// SPDX-License-Identifier: GPL-2.0-or-later
use super::{Error, Receiver};
use crate::{
    block::BlockError,
    wifi::intel::{
        boot::BootIo,
        dma::tests::{Domain, Source},
        firmware::legacy::{Sku, rx::Queue},
        power::{AccessError, Csr},
    },
};
use alloc::{collections::BTreeMap, rc::Rc, sync::Arc, vec::Vec};
use core::cell::RefCell;

const INT: u32 = 0x08;
const INT_MASK: u32 = 0x0c;
const FH_INT: u32 = 0x10;
const RESET: u32 = 0x20;
const GP_CONTROL: u32 = 0x24;
const ICT: u32 = 0xa0;
const SHADOW: u32 = 0xa8;
const WADDR: u32 = 0x444;
const RADDR: u32 = 0x448;
const WDATA: u32 = 0x44c;
const RDATA: u32 = 0x450;
const RX_CREDITS: u32 = 0x1c80;

const FREE: u32 = 0xa08000;
const USED: u32 = 0xa08100;
const STATUS_BASE: u32 = 0xa08200;
const FREE_WRITE: u32 = 0xa08080;
const FREE_READ: u32 = 0xa080c0;
const USED_WRITE: u32 = 0xa08180;
const CONFIG: u32 = 0xa09800;
const RFH_STATUS: u32 = 0xa09808;
const ACTIVE: u32 = 0xa0980c;
const DMA: u32 = 0xa09820;

const MAC_REQUEST: u32 = 1 << 3;
const RFH_IDLE: u32 = 1 << 31;
const STOP_MASTER: u32 = 0x200;
const MASTER_DISABLED: u32 = 0x100;
const SW_RESET: u32 = 0x80;
const FH_TX: u32 = 1 << 27;
const RX_IRQ: u32 = 1 << 31;
const ALIVE_IRQ: u32 = 1;
const RX_FH: u32 = 1 << 16;
const RX_MASK: u32 = 0xa200_0009;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    Read(u32, u32),
    Write(u32, u32),
    PrphRead(u32, u32),
    PrphWrite(u32, u32),
    Byte(u32, u8),
    Delay(u32),
}

struct Device {
    csr: BTreeMap<u32, u32>,
    prph: BTreeMap<u32, u32>,
    events: Vec<Event>,
    read_address: u32,
    write_address: u32,
    hold_rfh: bool,
    rfh_delay: Option<u32>,
    rfh_status_override: Option<u32>,
    ack_master: bool,
    fail_enable_once: bool,
}

impl Device {
    fn new(mask: u32) -> Self {
        Self {
            csr: BTreeMap::from([(GP_CONTROL, 1), (INT_MASK, mask), (SHADOW, 0x800f_ffff)]),
            prph: BTreeMap::from([(RFH_STATUS, RFH_IDLE)]),
            events: Vec::new(),
            read_address: 0,
            write_address: 0,
            hold_rfh: false,
            rfh_delay: None,
            rfh_status_override: None,
            ack_master: true,
            fail_enable_once: false,
        }
    }

    fn csr(&self, register: u32) -> u32 {
        self.csr.get(&register).copied().unwrap_or_default()
    }

    fn prph(&self, register: u32) -> u32 {
        self.prph.get(&register).copied().unwrap_or_default()
    }

    fn base(&self, register: u32) -> u64 {
        u64::from(self.prph(register)) | (u64::from(self.prph(register + 4)) << 32)
    }

    fn event_at(&self, event: Event) -> usize {
        self.events.iter().position(|item| *item == event).unwrap()
    }
}

#[derive(Clone)]
struct Io(Rc<RefCell<Device>>);

impl Csr for Io {
    fn read32(&mut self, register: u32) -> Result<u32, AccessError> {
        let mut device = self.0.borrow_mut();
        if register == RDATA {
            assert_ne!(device.csr(GP_CONTROL) & MAC_REQUEST, 0);
            let address = device.read_address;
            let value = if address == RFH_STATUS {
                device
                    .rfh_status_override
                    .unwrap_or_else(|| device.prph(address))
            } else {
                device.prph(address)
            };
            device.events.push(Event::PrphRead(address, value));
            return Ok(value);
        }
        let value = device.csr(register);
        device.events.push(Event::Read(register, value));
        Ok(value)
    }

    fn write32(&mut self, register: u32, value: u32) -> Result<(), AccessError> {
        let mut device = self.0.borrow_mut();
        device.events.push(Event::Write(register, value));
        match register {
            INT | FH_INT => {
                let remaining = device.csr(register) & !value;
                device.csr.insert(register, remaining);
            }
            WADDR | RADDR => {
                assert_ne!(device.csr(GP_CONTROL) & MAC_REQUEST, 0);
                assert_eq!(value & !0x000f_ffff, 3 << 24);
                let address = 0xa00000 | (value & 0x000f_ffff);
                if register == RADDR {
                    device.read_address = address;
                } else {
                    device.write_address = address;
                }
            }
            WDATA => {
                assert_ne!(device.csr(GP_CONTROL) & MAC_REQUEST, 0);
                let address = device.write_address;
                device.events.push(Event::PrphWrite(address, value));
                device.prph.insert(address, value);
                if address == DMA {
                    if value != 0 {
                        device.prph.insert(RFH_STATUS, 0);
                        if core::mem::take(&mut device.fail_enable_once) {
                            return Err(AccessError::Mapping);
                        }
                    } else if !device.hold_rfh && device.rfh_delay.is_none() {
                        device.prph.insert(RFH_STATUS, RFH_IDLE);
                    }
                }
            }
            RESET => {
                let acknowledged = if device.ack_master && value & STOP_MASTER != 0 {
                    MASTER_DISABLED
                } else {
                    0
                };
                device
                    .csr
                    .insert(RESET, value & !MASTER_DISABLED | acknowledged);
            }
            _ => {
                device.csr.insert(register, value);
            }
        }
        Ok(())
    }

    fn delay_us(&mut self, microseconds: u32) -> Result<(), AccessError> {
        let mut device = self.0.borrow_mut();
        device.events.push(Event::Delay(microseconds));
        if device.prph(DMA) == 0
            && let Some(remaining) = device.rfh_delay
        {
            let remaining = remaining.saturating_sub(microseconds);
            if remaining == 0 {
                device.rfh_delay = None;
                device.prph.insert(RFH_STATUS, RFH_IDLE);
            } else {
                device.rfh_delay = Some(remaining);
            }
        }
        Ok(())
    }
}

// SAFETY: fixture clones share a synchronous, exclusive requester model.
// A register write has taken effect before either success or an injected
// uncertain error is returned, and no physical device or IRQ task exists.
unsafe impl BootIo for Io {
    fn write8(&mut self, register: u32, value: u8) -> Result<(), AccessError> {
        self.0
            .borrow_mut()
            .events
            .push(Event::Byte(register, value));
        Ok(())
    }

    fn reset_function(&mut self) -> Result<bool, BlockError> {
        panic!("RFH bootstrap shutdown uses the legacy CSR reset")
    }
}

struct Fixture {
    receiver: Receiver<Io>,
    device: Rc<RefCell<Device>>,
    domain: Arc<Domain>,
    mappings: usize,
}

impl Fixture {
    fn new(mask: u32, sku: Sku) -> Self {
        let device = Rc::new(RefCell::new(Device::new(mask)));
        let domain = Arc::new(Domain::default());
        domain.records.lock().unwrap().coherent = true;
        let queue = Queue::prepare(&Source::default(), domain.clone(), (1 << 36) - 1, 1).unwrap();
        let mappings = domain.records.lock().unwrap().live.len();
        // SAFETY: this device is exclusively owned by the fixture and paired
        // with the exact mock domain that owns its still-unpublished queue.
        let receiver = unsafe { Receiver::new(Io(device.clone()), queue, sku) };
        Self {
            receiver,
            device,
            domain,
            mappings,
        }
    }

    fn write_dma(&self, address: u64, bytes: &[u8], complete: bool) {
        // The fixture's Domain maps its allocated pages at this fixed stride.
        let id = (address - 0x100000) / 0x10000;
        assert_eq!(address, 0x100000 + id * 0x10000);
        let mut records = self.domain.records.lock().unwrap();
        let ptr = records.live[&id].address(0, bytes.len()).unwrap();
        // SAFETY: these are live mappings in a synchronous model. The device
        // writes are ordered by the fixture before the consumer can poll.
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr(), bytes.len()) };
        if complete {
            assert!(records.completed.insert(id));
        }
    }

    fn first_free_address(&self) -> u64 {
        let free = self.device.borrow().base(FREE);
        let id = (free - 0x100000) / 0x10000;
        let records = self.domain.records.lock().unwrap();
        let ptr = records.live[&id].address(0, 8).unwrap();
        // SAFETY: the fixture is the device and holds the immutable, live
        // descriptor mapping. The Rust host never races this read.
        let descriptor = u64::from_le_bytes(unsafe { ptr.as_ptr().cast::<[u8; 8]>().read() });
        assert_eq!(descriptor & 0xfff, 1);
        descriptor & !0xfff
    }
}

#[test]
fn publishes_queue_zero_in_order_without_stealing_fh_interrupts() {
    let mut fixture = Fixture::new(FH_TX, Sku::Cnvi9000);
    fixture.receiver.start().unwrap();
    {
        let device = fixture.device.borrow();
        let prph: Vec<_> = device
            .events
            .iter()
            .filter_map(|event| match event {
                Event::PrphWrite(address, value) => Some((*address, *value)),
                _ => None,
            })
            .collect();
        let free = device.base(FREE);
        let used = device.base(USED);
        let status = device.base(STATUS_BASE);
        assert_ne!(free, 0);
        assert_ne!(free, used);
        assert_ne!(status, used);
        assert_eq!(
            prph,
            [
                (DMA, 0),
                (ACTIVE, 0),
                (FREE, free as u32),
                (FREE + 4, (free >> 32) as u32),
                (USED, used as u32),
                (USED + 4, (used >> 32) as u32),
                (STATUS_BASE, status as u32),
                (STATUS_BASE + 4, (status >> 32) as u32),
                (FREE_WRITE, 0),
                (FREE_READ, 0),
                (USED_WRITE, 0),
                (DMA, 0x8794_0000),
                (CONFIG, 3),
                (ACTIVE, 0x0001_0001),
            ]
        );
        assert_eq!(device.csr(INT_MASK), FH_TX);
        assert_eq!(device.csr(GP_CONTROL) & MAC_REQUEST, 0);
        let active = device.event_at(Event::PrphWrite(ACTIVE, 0x0001_0001));
        let credits = device.event_at(Event::Write(RX_CREDITS, 504));
        assert!(active < credits);
        assert!(device.event_at(Event::Byte(4, 0x40)) < credits);
    }
    fixture.receiver.stop().unwrap();
    assert!(fixture.receiver.finish().is_ok());
}

#[test]
fn refuses_an_active_ict_route_before_publishing_dma() {
    let mut fixture = Fixture::new(FH_TX, Sku::Cnvi9000);
    fixture.device.borrow_mut().csr.insert(ICT, 1 << 31);
    assert_eq!(fixture.receiver.start(), Err(Error::InterruptMode));
    assert_eq!(fixture.device.borrow().events, [Event::Read(ICT, 1 << 31)]);
    assert_eq!(fixture.domain.records.lock().unwrap().unmaps, 0);

    fixture.receiver.stop().unwrap();
    assert!(fixture.receiver.finish().is_ok());
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}

#[test]
fn interrupt_is_only_a_hint_and_producer_completion_needs_no_irq() {
    let mut fixture = Fixture::new(0, Sku::Cnvi9000);
    fixture.receiver.start().unwrap();
    fixture.receiver.enable_interrupts().unwrap();
    assert_eq!(fixture.device.borrow().csr(INT_MASK), RX_MASK);
    {
        let mut device = fixture.device.borrow_mut();
        device.csr.insert(INT, RX_IRQ | ALIVE_IRQ | FH_TX);
        device.csr.insert(FH_INT, RX_FH | 1);
    }
    assert!(fixture.receiver.poll().unwrap().is_none());
    assert_eq!(fixture.domain.records.lock().unwrap().cpu_syncs, 0);
    {
        let device = fixture.device.borrow();
        assert_eq!(device.csr(INT), FH_TX);
        assert_eq!(device.csr(FH_INT), 1);
    }

    let mut frame = [0; 4096];
    frame[..4].copy_from_slice(&4u32.to_le_bytes());
    frame[4] = 4;
    frame[64..68].copy_from_slice(&0x5555_0000u32.to_le_bytes());
    let first = fixture.first_free_address();
    let (used, status) = {
        let device = fixture.device.borrow();
        (device.base(USED), device.base(STATUS_BASE))
    };
    fixture.write_dma(first, &frame, true);
    fixture.write_dma(used, &1u32.to_le_bytes(), false);
    fixture.write_dma(status, &1u32.to_le_bytes(), false);
    let received = fixture.receiver.poll().unwrap().unwrap();
    let packet = received.packets().unwrap().next().unwrap().unwrap();
    assert_eq!((packet.group, packet.command), (0, 4));
    assert_eq!(fixture.domain.records.lock().unwrap().cpu_syncs, 1);
    assert!(fixture.receiver.poll().unwrap().is_none());

    let observed = fixture.domain.records.lock().unwrap().coherent_reads.len();
    fixture.device.borrow_mut().csr.insert(INT_MASK, 0);
    assert!(fixture.receiver.poll().is_err());
    assert_eq!(
        fixture.domain.records.lock().unwrap().coherent_reads.len(),
        observed,
        "a reset interrupt mask must prevent even a coherent producer read"
    );
    fixture.receiver.stop().unwrap();
}

#[test]
fn uncertain_rfh_enable_keeps_the_published_mappings_and_requester() {
    let mut fixture = Fixture::new(0, Sku::Cnvi9000);
    fixture.device.borrow_mut().fail_enable_once = true;
    assert!(fixture.receiver.start().is_err());
    assert!(fixture.receiver.poll().is_err());
    assert!(fixture.receiver.finish().is_err());
    {
        let records = fixture.domain.records.lock().unwrap();
        assert_eq!(records.live.len(), fixture.mappings);
        assert_eq!(records.unmaps, 0);
    }
    fixture.receiver.stop().unwrap();
    assert!(fixture.receiver.finish().is_ok());
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}

#[test]
fn bad_idle_status_and_master_timeout_cannot_release_dma() {
    let mut fixture = Fixture::new(0, Sku::Cnvi9000);
    fixture.receiver.start().unwrap();
    fixture.receiver.enable_interrupts().unwrap();
    fixture.device.borrow_mut().hold_rfh = true;
    assert!(fixture.receiver.stop().is_err());
    assert!(fixture.receiver.finish().is_err());
    assert_eq!(fixture.domain.records.lock().unwrap().unmaps, 0);
    fixture.device.borrow_mut().rfh_status_override = Some(u32::MAX);
    assert!(fixture.receiver.stop().is_err());
    assert_eq!(fixture.domain.records.lock().unwrap().unmaps, 0);
    {
        let mut device = fixture.device.borrow_mut();
        device.hold_rfh = false;
        device.rfh_status_override = None;
        device.ack_master = false;
    }
    assert!(fixture.receiver.stop().is_err());
    assert_eq!(fixture.domain.records.lock().unwrap().unmaps, 0);
    assert_eq!(fixture.device.borrow().csr(RESET) & SW_RESET, 0);

    fixture.device.borrow_mut().ack_master = true;
    fixture.receiver.stop().unwrap();
    assert!(fixture.receiver.finish().is_ok());
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}

#[test]
fn shutdown_waits_for_rfh_and_master_then_retries_only_invalidation() {
    let mut fixture = Fixture::new(0, Sku::Cnvi9000);
    fixture.receiver.start().unwrap();
    fixture.receiver.enable_interrupts().unwrap();
    {
        let mut device = fixture.device.borrow_mut();
        device.events.clear();
        device.rfh_delay = Some(70);
    }
    fixture.domain.records.lock().unwrap().unmap_fail = Some(1);
    assert!(fixture.receiver.stop().is_err());
    assert!(fixture.receiver.finish().is_err());
    let writes = {
        let device = fixture.device.borrow();
        let mask = device.event_at(Event::Write(INT_MASK, 0));
        let disable = device.event_at(Event::PrphWrite(DMA, 0));
        let idle = device.event_at(Event::PrphRead(RFH_STATUS, RFH_IDLE));
        let stop = device.event_at(Event::Write(RESET, STOP_MASTER));
        let ack = device.event_at(Event::Read(RESET, STOP_MASTER | MASTER_DISABLED));
        let reset = device.event_at(Event::Write(
            RESET,
            STOP_MASTER | MASTER_DISABLED | SW_RESET,
        ));
        let delay = device.event_at(Event::Delay(5000));
        assert!(
            mask < disable
                && disable < idle
                && idle < stop
                && stop < ack
                && ack < reset
                && reset < delay
        );
        device.events.len()
    };
    assert_eq!(
        fixture.domain.records.lock().unwrap().live.len(),
        fixture.mappings - 1
    );
    fixture.receiver.stop().unwrap();
    assert_eq!(fixture.device.borrow().events.len(), writes);
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
    assert!(fixture.receiver.finish().is_ok());
}
