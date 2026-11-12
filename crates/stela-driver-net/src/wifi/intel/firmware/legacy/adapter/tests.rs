// SPDX-License-Identifier: GPL-2.0-or-later
extern crate std;

use super::{Adapter, Error};
use crate::{
    block::BlockError,
    bus::dma::{
        mapping::{Direction, DmaDomain, DmaError, DomainMapping},
        pages::DmaPages,
    },
    wifi::intel::{
        boot::BootIo,
        dma::tests::{Domain, Source},
        firmware::{
            CMD_VERSIONS, Firmware, SEC_INIT,
            legacy::{
                CPU_SEPARATOR, Sku,
                alive::{Image, Progress},
                tests::{image, section},
            },
        },
        power::{AccessError, Csr},
    },
};
use alloc::{collections::BTreeMap, sync::Arc, vec::Vec};
use core::{num::NonZeroUsize, ops::Range};
use std::sync::Mutex;
use stela_driver_base::{Clock, test_clock};

mod bluetooth;
mod calibration;
mod dqa;
mod nvm;
mod paging;
mod phy_configuration;
mod phy_db;
mod regular;
mod scheduler;
mod soc;

#[test]
fn inherited_stop_masks_and_halts_engines_before_reset_without_new_dma() {
    let device = Arc::new(Mutex::new(Device::new()));
    let domain = Arc::new(Domain::default());
    let mut io = Io {
        device: device.clone(),
        domain: domain.clone(),
    };
    let mut stop = super::InheritedStop::default();
    device.lock().unwrap().gone = true;
    // SAFETY: this fixture is the exclusive original requester accessor.
    assert!(unsafe { stop.halt(&mut io) }.is_err());
    {
        let mut model = device.lock().unwrap();
        assert!(
            !model
                .events
                .iter()
                .any(|event| matches!(event, Event::Write(..)))
        );
        model.gone = false;
        model.ack_master = false;
        model.events.clear();
    }
    // SAFETY: the original fixture and stop state stay in this attempt.
    assert_eq!(
        unsafe { stop.halt(&mut io) },
        Err(Error::Receiver(super::rfh::Error::StopTimeout {
            register: RESET
        }))
    );
    {
        let mut model = device.lock().unwrap();
        let events = &model.events;
        let position = |event| events.iter().position(|seen| *seen == event).unwrap();
        assert!(position(Event::Write(MASK, 0)) < position(Event::PeripheralWrite(SCD_TXFACT, 0)));
        assert!(
            position(Event::PeripheralWrite(SCD_TXFACT, 0))
                < position(Event::PeripheralWrite(RFH_DMA, 0))
        );
        assert!(
            position(Event::PeripheralWrite(RFH_DMA, 0))
                < position(Event::Write(RESET, MASTER_STOP))
        );
        assert!(!events.iter().any(
            |event| matches!(event, Event::Write(RESET, value) if value & SOFTWARE_RESET != 0)
        ));
        model.ack_master = true;
    }
    // SAFETY: this retries the same exclusively owned BAR and custody state.
    unsafe { stop.halt(&mut io) }.unwrap();
    assert!(device.lock().unwrap().reset_settled);
    assert_eq!(domain.records.lock().unwrap().maps, 0);
}

const INT: u32 = 0x08;
const MASK: u32 = 0x0c;
const FH_INT: u32 = 0x10;
const RESET: u32 = 0x20;
const GP: u32 = 0x24;
const RFKILL_CLEAR: u32 = 0x5c;
const PRPH_WRITE_ADDR: u32 = 0x444;
const PRPH_READ_ADDR: u32 = 0x448;
const PRPH_WRITE_DATA: u32 = 0x44c;
const PRPH_READ_DATA: u32 = 0x450;
const COMMAND_DOORBELL: u32 = 0x460;
const LOAD_STATUS: u32 = 0x1af0;
const FH_TX_STATUS: u32 = 0x1eb0;
const FH_TX_CHANNELS: [u32; 8] = [
    0x1d00, 0x1d20, 0x1d40, 0x1d60, 0x1d80, 0x1da0, 0x1dc0, 0x1de0,
];
const TX_CONFIG: u32 = 0x1e20;
const FH_DESTINATION: u32 = 0x19c8;
const FH_SIZE: u32 = 0x194c;
const RX_DOORBELL: u32 = 0x1c80;
const FREE_BASE: u32 = 0x08000;
const USED_BASE: u32 = 0x08100;
const STATUS_BASE: u32 = 0x08200;
const RFH_STATUS: u32 = 0x09808;
const RFH_DMA: u32 = 0x09820;
const CPU_RELEASE: u32 = 0x300c;
const SCD_TXFACT: u32 = 0x02c10;

const FH_MASK: u32 = 1 << 27;
const RX_MASK: u32 = 0xa200_0009;
const TX_ENABLE: u32 = 0x8010_0000;
const MASTER_STOP: u32 = 0x200;
const MASTER_DISABLED: u32 = 0x100;
const SOFTWARE_RESET: u32 = 0x80;
const MAC_ACCESS: u32 = 1 << 3;
const RFH_IDLE: u32 = 1 << 31;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    Read(u32, u32),
    Write(u32, u32),
    PeripheralWrite(u32, u32),
    SramWrite(u32, u32),
    FhComplete,
    FhSubmission {
        destination: u32,
        bytes: u32,
        mask: u32,
    },
    Drain(usize),
}

struct Device {
    csr: BTreeMap<u32, u32>,
    peripheral: BTreeMap<u32, u32>,
    events: Vec<Event>,
    read_address: u32,
    write_address: u32,
    active_fh: bool,
    fail_posted_enable: bool,
    fail_fh_destination: Option<u32>,
    regular_upload: bool,
    lose_fh_nic_release: bool,
    fail_final_health: bool,
    gone: bool,
    ack_master: bool,
    reset_settled: bool,
    fail_drain: bool,
    drained: bool,
    command: nvm::CommandDevice,
    scheduler: scheduler::DeviceMemory,
}

impl Device {
    fn new() -> Self {
        Self {
            csr: BTreeMap::from([(GP, 1)]),
            peripheral: BTreeMap::from([(RFH_STATUS, RFH_IDLE)]),
            events: Vec::new(),
            read_address: 0,
            write_address: 0,
            active_fh: false,
            fail_posted_enable: false,
            fail_fh_destination: None,
            regular_upload: false,
            lose_fh_nic_release: false,
            fail_final_health: false,
            gone: false,
            ack_master: true,
            reset_settled: false,
            fail_drain: false,
            drained: false,
            command: nvm::CommandDevice::default(),
            scheduler: scheduler::DeviceMemory::default(),
        }
    }

    fn csr(&self, register: u32) -> u32 {
        self.csr.get(&register).copied().unwrap_or_default()
    }

    fn peripheral(&self, register: u32) -> u32 {
        self.peripheral.get(&register).copied().unwrap_or_default()
    }

    fn address(&self, register: u32) -> u64 {
        u64::from(self.peripheral(register)) | (u64::from(self.peripheral(register + 4)) << 32)
    }
}

#[derive(Clone)]
struct Io {
    device: Arc<Mutex<Device>>,
    domain: Arc<Domain>,
}

impl Csr for Io {
    fn read32(&mut self, register: u32) -> Result<u32, AccessError> {
        let mut device = self.device.lock().unwrap();
        let value = if device.gone {
            u32::MAX
        } else if register == PRPH_READ_DATA {
            assert_ne!(device.csr(GP) & MAC_ACCESS, 0);
            device.peripheral(device.read_address)
        } else if register == FH_TX_STATUS {
            device.command.tx_status()
        } else if register == scheduler::MEM_READ_DATA {
            assert_ne!(device.csr(GP) & MAC_ACCESS, 0);
            device.scheduler.read_data()
        } else {
            device.csr(register)
        };
        device.events.push(Event::Read(register, value));
        if register == LOAD_STATUS && value == u32::MAX && device.fail_final_health {
            device.gone = true;
        }
        Ok(value)
    }

    fn write32(&mut self, register: u32, value: u32) -> Result<(), AccessError> {
        let mut device = self.device.lock().unwrap();
        device.events.push(Event::Write(register, value));
        match register {
            INT | FH_INT => {
                let remaining = device.csr(register) & !value;
                device.csr.insert(register, remaining);
            }
            PRPH_READ_ADDR | PRPH_WRITE_ADDR => {
                assert_ne!(device.csr(GP) & MAC_ACCESS, 0);
                assert_eq!(value & !0x000f_ffff, 3 << 24);
                let address = value & 0x000f_ffff;
                if register == PRPH_READ_ADDR {
                    device.read_address = address;
                } else {
                    device.write_address = address;
                }
            }
            PRPH_WRITE_DATA => {
                assert_ne!(device.csr(GP) & MAC_ACCESS, 0);
                let address = device.write_address;
                device.peripheral.insert(address, value);
                device.events.push(Event::PeripheralWrite(address, value));
                if address == RFH_DMA {
                    device
                        .peripheral
                        .insert(RFH_STATUS, if value == 0 { RFH_IDLE } else { 0 });
                }
                if address == SCD_TXFACT && value == 0 {
                    device.command.stop_scheduler();
                }
                device.scheduler_prph_write(address, value)?;
            }
            RESET => {
                let acknowledged = if device.ack_master && value & MASTER_STOP != 0 {
                    MASTER_DISABLED
                } else {
                    0
                };
                device
                    .csr
                    .insert(RESET, value & !MASTER_DISABLED | acknowledged);
                if value & SOFTWARE_RESET != 0 {
                    assert_ne!(acknowledged, 0);
                    device.active_fh = false;
                    device.command.reset();
                }
            }
            GP if value & MAC_ACCESS == 0 && device.active_fh && device.lose_fh_nic_release => {
                // A posted MAC release reached the BAR but did not clear
                // the bit. The next owner-controlled cleanup can retry it.
                device.lose_fh_nic_release = false;
            }
            TX_CONFIG if value == TX_ENABLE => {
                let mask = device.csr(MASK);
                assert!(mask == FH_MASK || device.regular_upload && mask == (FH_MASK | RX_MASK));
                assert_ne!(device.csr(GP) & MAC_ACCESS, 0);
                assert!(!device.active_fh);
                assert_eq!(device.csr(INT) & FH_MASK, 0);
                assert_eq!(device.csr(FH_INT) & 3, 0);
                let records = self.domain.records.lock().unwrap();
                assert_eq!(
                    records.live.len(),
                    508 + device.command.prepared_pins(),
                    "only one staging pin alongside the rings"
                );
                assert!(
                    records
                        .live
                        .contains_key(&u64::try_from(records.maps).unwrap())
                );
                let destination = device.csr(FH_DESTINATION);
                let bytes = device.csr(FH_SIZE) & 0x0fff_ffff;
                assert_ne!(bytes, 0);
                device.events.push(Event::FhSubmission {
                    destination,
                    bytes,
                    mask,
                });
                device.active_fh = true;
                if device.fail_fh_destination == Some(destination) {
                    device.fail_fh_destination = None;
                    return Err(AccessError::Mapping);
                }
                if core::mem::take(&mut device.fail_posted_enable) {
                    return Err(AccessError::Mapping);
                }
            }
            COMMAND_DOORBELL => {
                assert_eq!(device.csr(MASK), RX_MASK);
                assert_ne!(device.csr(GP) & MAC_ACCESS, 0);
                if value & 0xff != 0 {
                    device.command.publish(&self.domain, value)?;
                }
                device.csr.insert(register, value);
            }
            register if FH_TX_CHANNELS.contains(&register) => {
                let channel = (register - FH_TX_CHANNELS[0]) / 0x20;
                if value == 0 {
                    device.command.stop_channel(channel);
                } else {
                    device.command.enable_channel(channel);
                }
                device.csr.insert(register, value);
            }
            scheduler::MEM_READ_ADDRESS | scheduler::MEM_WRITE_ADDRESS => {
                assert_ne!(device.csr(GP) & MAC_ACCESS, 0);
                assert_eq!(value & 3, 0);
                device.scheduler.select(register, value);
            }
            scheduler::MEM_WRITE_DATA => {
                assert_ne!(device.csr(GP) & MAC_ACCESS, 0);
                let address = device.scheduler.write_data(value);
                device.events.push(Event::SramWrite(address, value));
            }
            _ => {
                device.csr.insert(register, value);
                device.scheduler_csr_write(&self.domain, register, value)?;
            }
        }
        Ok(())
    }

    fn delay_us(&mut self, microseconds: u32) -> Result<(), AccessError> {
        let mut device = self.device.lock().unwrap();
        if microseconds == 5000 && device.csr(RESET) & SOFTWARE_RESET != 0 {
            device.reset_settled = true;
        }
        if microseconds == 1000 && device.active_fh {
            assert_eq!(device.csr(GP) & MAC_ACCESS, 0);
            let shared_rx = device.regular_upload && device.csr(MASK) == (FH_MASK | RX_MASK);
            let csr = device.csr(INT) | FH_MASK | (u32::from(shared_rx) << 31);
            let fh = device.csr(FH_INT) | 1 | (u32::from(shared_rx) << 16);
            device.csr.insert(INT, csr);
            device.csr.insert(FH_INT, fh);
            device.events.push(Event::FhComplete);
            device.active_fh = false;
        }
        Ok(())
    }
}

// SAFETY: the test owns one synchronous, synthetic PCI function and domain.
// MMIO effects happen in program order, including the simulated posted error.
unsafe impl BootIo for Io {
    fn write8(&mut self, register: u32, value: u8) -> Result<(), AccessError> {
        self.device
            .lock()
            .unwrap()
            .events
            .push(Event::Write(register, value.into()));
        Ok(())
    }

    fn reset_function(&mut self) -> Result<bool, BlockError> {
        panic!("this legacy transport shuts down via its CSR reset")
    }
}

struct PostedDomain {
    inner: Arc<Domain>,
    device: Arc<Mutex<Device>>,
}

// SAFETY: mapping and cache operations retain the original synthetic domain's
// guarantees. The drain verifies this exact requester's modeled reset and idle.
unsafe impl DmaDomain for PostedDomain {
    unsafe fn map(
        &self,
        pages: &DmaPages,
        direction: Direction,
        limit: u64,
    ) -> Result<DomainMapping, DmaError> {
        // SAFETY: the caller's pins, direction and limit apply to this domain.
        unsafe { self.inner.map(pages, direction, limit) }
    }

    unsafe fn unmap(&self, mapping: DomainMapping) -> Result<(), DmaError> {
        let device = self.device.lock().unwrap();
        assert!(!device.active_fh);
        device.command.assert_can_unmap(mapping.id);
        let exposed = [FREE_BASE, USED_BASE, STATUS_BASE]
            .into_iter()
            .any(|register| device.address(register) != 0)
            || device.command.prepared_pins() != 0;
        if exposed && device.csr(RESET) & MASTER_STOP != 0 {
            assert!(
                device.drained,
                "stop must drain before invalidating any lease"
            );
        }
        // SAFETY: the caller's quiescence and live mapping apply here.
        unsafe { self.inner.unmap(mapping) }
    }

    unsafe fn sync_for_device(
        &self,
        mapping: DomainMapping,
        range: Range<usize>,
        direction: Direction,
    ) -> Result<(), DmaError> {
        // SAFETY: the caller's live mapping, range and direction apply here.
        unsafe { self.inner.sync_for_device(mapping, range, direction) }
    }

    unsafe fn sync_for_cpu(
        &self,
        mapping: DomainMapping,
        range: Range<usize>,
        direction: Direction,
    ) -> Result<(), DmaError> {
        // SAFETY: the caller's completion, live mapping and range apply here.
        unsafe { self.inner.sync_for_cpu(mapping, range, direction) }
    }

    unsafe fn read_coherent_u32(
        &self,
        mapping: DomainMapping,
        offset: usize,
        direction: Direction,
    ) -> Result<u32, DmaError> {
        // SAFETY: the caller's live mapping, direction and offset apply here.
        unsafe { self.inner.read_coherent_u32(mapping, offset, direction) }
    }

    unsafe fn drain_posted_writes(&self) -> Result<(), DmaError> {
        let mut device = self.device.lock().unwrap();
        assert!(device.reset_settled);
        assert_eq!(device.csr(MASK), 0);
        assert_eq!(device.peripheral(RFH_DMA), 0);
        assert_ne!(device.peripheral(RFH_STATUS) & RFH_IDLE, 0);
        assert!(!device.active_fh);
        assert!(!device.drained);
        let unmaps = self.inner.records.lock().unwrap().unmaps;
        device.events.push(Event::Drain(unmaps));
        if core::mem::take(&mut device.fail_drain) {
            return Err(DmaError::DrainUnavailable);
        }
        device.drained = true;
        Ok(())
    }
}

struct Fixture<'a> {
    adapter: Adapter<'a, Io>,
    device: Arc<Mutex<Device>>,
    domain: Arc<Domain>,
    rx_index: u16,
}

impl<'a> Fixture<'a> {
    fn dma_counts(&self) -> (usize, usize, usize) {
        let records = self.domain.records.lock().unwrap();
        (records.maps, records.unmaps, records.live.len())
    }

    fn new(source: &'a Source) -> Self {
        let device = Arc::new(Mutex::new(Device::new()));
        let domain = Arc::new(Domain::default());
        domain.records.lock().unwrap().coherent = true;
        let io = Io {
            device: device.clone(),
            domain: domain.clone(),
        };
        let posted = Arc::new(PostedDomain {
            inner: domain.clone(),
            device: device.clone(),
        });
        let clock = Clock {
            ticks_per_microsecond: 1,
        };
        // SAFETY: no physical device is involved, and this is the only owner
        // of the fresh modeled requester and its DMA domain.
        let adapter = unsafe { Adapter::new(io, source, posted, Sku::Cnvi9000, 1, clock) }.unwrap();
        Self {
            adapter,
            device,
            domain,
            rx_index: 0,
        }
    }

    fn publish_rx(&mut self, frame: &[u8; 4096]) {
        publish_rx_frame(&self.device, &self.domain, &mut self.rx_index, frame);
    }

    fn publish_alive(&mut self) {
        self.publish_alive_with_scheduler(0);
    }

    fn publish_alive_with_scheduler(&mut self, scheduler: u32) {
        self.publish_rx(&alive_frame(scheduler));
        self.device.lock().unwrap().scheduler.alive = true;
    }
}

fn alive_frame(scheduler: u32) -> [u8; 4096] {
    let mut frame = [0; 4096];
    frame[..4].copy_from_slice(&72u32.to_le_bytes());
    frame[4] = 1;
    frame[8..10].copy_from_slice(&0xcafeu16.to_le_bytes());
    frame[48..52].copy_from_slice(&scheduler.to_le_bytes());
    frame[128..132].copy_from_slice(&0x5555_0000u32.to_le_bytes());
    frame
}

fn publish_rx_frame(
    device: &Mutex<Device>,
    domain: &Domain,
    rx_index: &mut u16,
    frame: &[u8; 4096],
) {
    let (free, used, status) = {
        let device = device.lock().unwrap();
        (
            device.address(FREE_BASE),
            device.address(USED_BASE),
            device.address(STATUS_BASE),
        )
    };
    let index = usize::from(*rx_index);
    let descriptor = u64::from_le_bytes(read_dma(domain, free, index * 8));
    *rx_index += 1;
    let vid = u32::from(*rx_index);
    assert_eq!(descriptor & 0xfff, u64::from(vid));
    write_dma(domain, descriptor & !0xfff, 0, frame, true);
    write_dma(domain, used, index * 4, &vid.to_le_bytes(), false);
    write_dma(domain, status, 0, &vid.to_le_bytes(), false);
}

fn write_dma(domain: &Domain, address: u64, offset: usize, bytes: &[u8], complete: bool) {
    let (id, within) = dma_offset(address);
    let mut records = domain.records.lock().unwrap();
    assert!(matches!(
        records.permissions[&id],
        Direction::FromDevice | Direction::Bidirectional
    ));
    let ptr = records.live[&id]
        .address(within + offset, bytes.len())
        .unwrap();
    // SAFETY: synchronous writes by the synthetic device to a live
    // mapping, with completion published only after the bytes are ready.
    unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr(), bytes.len()) };
    if complete {
        assert!(records.completed.insert(id));
    }
}

fn dma_offset(address: u64) -> (u64, usize) {
    let relative = address - 0x100000;
    (relative / 0x10000, (relative % 0x10000) as usize)
}

fn read_dma<const N: usize>(domain: &Domain, address: u64, offset: usize) -> [u8; N] {
    let mut bytes = [0; N];
    read_dma_into(domain, address, offset, &mut bytes);
    bytes
}

fn read_dma_into(domain: &Domain, address: u64, offset: usize, bytes: &mut [u8]) {
    let (id, within) = dma_offset(address);
    let records = domain.records.lock().unwrap();
    assert!(matches!(
        records.permissions[&id],
        Direction::ToDevice | Direction::Bidirectional
    ));
    let ptr = records.live[&id]
        .address(within + offset, bytes.len())
        .unwrap();
    // SAFETY: the synchronous device owns the initialized range and the
    // domain lock keeps its live DMA mapping through the read.
    unsafe { core::ptr::copy_nonoverlapping(ptr.as_ptr(), bytes.as_mut_ptr(), bytes.len()) };
}

fn api46_image() -> Vec<u8> {
    let mut bytes = image();
    bytes.extend(CMD_VERSIONS.to_le_bytes());
    bytes.extend(4u32.to_le_bytes());
    bytes.extend([1, 0, 99, 3]);
    bytes
}

#[test]
fn init_uploads_chunks_then_hands_off_empty_cpu2_to_rfh_alive() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut bytes = api46_image();
    section(&mut bytes, SEC_INIT, 0x41000, &[2, 3, 4]);
    let firmware = Firmware::parse(&bytes).unwrap();
    fixture
        .adapter
        .start_init(&firmware, NonZeroUsize::new(2).unwrap())
        .unwrap();
    {
        let device = fixture.device.lock().unwrap();
        let relevant: Vec<_> = device
            .events
            .iter()
            .copied()
            .filter(|event| {
                matches!(
                    event,
                    Event::Write(RX_DOORBELL | MASK | LOAD_STATUS | RFKILL_CLEAR, _)
                        | Event::Write(TX_CONFIG, TX_ENABLE)
                        | Event::PeripheralWrite(CPU_RELEASE, _)
                        | Event::FhComplete
                )
            })
            .collect();
        assert_eq!(
            relevant,
            [
                Event::Write(RFKILL_CLEAR, 2),
                Event::Write(RFKILL_CLEAR, 4),
                Event::Write(RX_DOORBELL, 504),
                Event::Write(MASK, FH_MASK),
                Event::Write(RFKILL_CLEAR, 2),
                Event::Write(RFKILL_CLEAR, 2),
                Event::PeripheralWrite(CPU_RELEASE, 1 << 24),
                Event::Write(TX_CONFIG, TX_ENABLE),
                Event::FhComplete,
                Event::Write(LOAD_STATUS, 1),
                Event::Write(TX_CONFIG, TX_ENABLE),
                Event::FhComplete,
                Event::Write(TX_CONFIG, TX_ENABLE),
                Event::FhComplete,
                Event::Write(LOAD_STATUS, 3),
                Event::Write(MASK, RX_MASK),
                Event::Write(LOAD_STATUS, 0xffff),
                Event::Write(LOAD_STATUS, u32::MAX),
            ]
        );
    }
    {
        let records = fixture.domain.records.lock().unwrap();
        assert_eq!(records.live.len(), 507);
        assert_eq!(records.unmaps, 3);
        assert_eq!(
            &records.copied[records.copied.len() - 3..],
            &[vec![1], vec![2, 3], vec![4]]
        );
    }
    assert_eq!(
        fixture.adapter.poll_init_alive().unwrap(),
        Progress::Pending
    );
    fixture.publish_alive();
    assert!(matches!(
        fixture.adapter.poll_init_alive(),
        Ok(Progress::Alive {
            image: Image::Init,
            ..
        })
    ));
    fixture.adapter.stop().unwrap();
    assert!(fixture.adapter.finish().is_ok());
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
    assert!(
        fixture
            .device
            .lock()
            .unwrap()
            .events
            .contains(&Event::Drain(3))
    );
}

#[test]
fn rejects_bad_firmware_before_interrupts_or_dma_publication() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let before = fixture.domain.records.lock().unwrap().maps;
    let mut wrong_api = api46_image();
    wrong_api[72..76].copy_from_slice(&77u32.to_le_bytes());
    let mut oversized_init = api46_image();
    for n in 1..16 {
        section(&mut oversized_init, SEC_INIT, 0x42000 + n, &[0x18]);
    }
    for bytes in [wrong_api, image(), oversized_init] {
        let firmware = Firmware::parse(&bytes).unwrap();
        assert!(
            fixture
                .adapter
                .start_init(&firmware, NonZeroUsize::MIN)
                .is_err()
        );
        assert!(fixture.device.lock().unwrap().events.is_empty());
        assert_eq!(fixture.domain.records.lock().unwrap().maps, before);
    }
    let mut nonempty_cpu2 = api46_image();
    section(&mut nonempty_cpu2, SEC_INIT, CPU_SEPARATOR, &[]);
    section(&mut nonempty_cpu2, SEC_INIT, 0xc0080000, &[2]);
    let firmware = Firmware::parse(&nonempty_cpu2).unwrap();
    assert_eq!(
        fixture.adapter.start_init(&firmware, NonZeroUsize::MIN),
        Err(Error::Cpu2Sections)
    );
    assert!(fixture.device.lock().unwrap().events.is_empty());
    assert_eq!(fixture.domain.records.lock().unwrap().maps, before);
    fixture.adapter.stop().unwrap();
    assert!(fixture.adapter.finish().is_ok());
}

#[test]
fn posted_fh_error_keeps_staging_until_verified_stop_and_retryable_unmap() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let bytes = api46_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    {
        let mut device = fixture.device.lock().unwrap();
        device.fail_posted_enable = true;
        device.lose_fh_nic_release = true;
    }
    assert!(
        fixture
            .adapter
            .start_init(&firmware, NonZeroUsize::MIN)
            .is_err()
    );
    assert!(fixture.adapter.finish().is_err());
    let attempted = fixture.device.lock().unwrap().events.len();
    {
        let device = fixture.device.lock().unwrap();
        assert!(device.active_fh);
        assert_ne!(device.csr(GP) & MAC_ACCESS, 0);
    }
    assert!(
        fixture
            .adapter
            .start_init(&firmware, NonZeroUsize::MIN)
            .is_err()
    );
    assert_eq!(fixture.device.lock().unwrap().events.len(), attempted);
    {
        let records = fixture.domain.records.lock().unwrap();
        assert_eq!(records.live.len(), 508);
        assert_eq!(records.unmaps, 0);
    }

    fixture.device.lock().unwrap().ack_master = false;
    assert!(fixture.adapter.stop().is_err());
    assert!(fixture.adapter.finish().is_err());
    assert_eq!(fixture.domain.records.lock().unwrap().unmaps, 0);
    {
        let device = fixture.device.lock().unwrap();
        assert_eq!(device.csr(GP) & MAC_ACCESS, 0);
        let writes = &device.events[attempted..];
        assert!(writes.contains(&Event::Write(GP, 1)));
        let acquire = writes
            .iter()
            .position(|event| *event == Event::Write(GP, 1 | MAC_ACCESS))
            .unwrap();
        let release = writes
            .iter()
            .position(|event| *event == Event::Write(GP, 1))
            .unwrap();
        assert!(release < acquire);
    }
    fixture.device.lock().unwrap().ack_master = true;
    fixture.device.lock().unwrap().fail_drain = true;
    assert!(matches!(
        fixture.adapter.stop(),
        Err(Error::Drain(DmaError::DrainUnavailable))
    ));
    assert!(fixture.adapter.finish().is_err());
    assert_eq!(fixture.domain.records.lock().unwrap().unmaps, 0);
    assert_eq!(fixture.domain.records.lock().unwrap().live.len(), 508);
    let before_drain_retry = fixture.device.lock().unwrap().events.len();

    fixture.domain.records.lock().unwrap().unmap_fail = Some(0);
    assert!(fixture.adapter.stop().is_err());
    assert!(fixture.adapter.finish().is_err());
    assert_eq!(fixture.domain.records.lock().unwrap().live.len(), 508);
    {
        let device = fixture.device.lock().unwrap();
        assert_eq!(&device.events[before_drain_retry..], &[Event::Drain(0)]);
    }
    fixture.adapter.stop().unwrap();
    assert_eq!(
        fixture.device.lock().unwrap().events.len(),
        before_drain_retry + 1
    );
    assert!(fixture.adapter.finish().is_ok());
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}

#[test]
fn stop_refuses_to_clear_a_mac_request_without_custody() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let bytes = api46_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    fixture
        .adapter
        .start_init(&firmware, NonZeroUsize::MIN)
        .unwrap();
    let before_stop = {
        let mut device = fixture.device.lock().unwrap();
        device.csr.insert(GP, 1 | MAC_ACCESS);
        device.events.len()
    };
    assert!(fixture.adapter.stop().is_err());
    assert!(fixture.adapter.finish().is_err());
    {
        let mut device = fixture.device.lock().unwrap();
        assert_eq!(device.csr(GP), 1 | MAC_ACCESS);
        assert!(
            !device.events[before_stop..]
                .iter()
                .any(|event| matches!(event, Event::Write(GP | RESET, _) | Event::Drain(_)))
        );
        device.csr.insert(GP, 1);
    }
    assert_eq!(fixture.domain.records.lock().unwrap().live.len(), 507);
    fixture.adapter.stop().unwrap();
    assert!(fixture.adapter.finish().is_ok());
}

#[test]
fn all_ones_cpu2_status_cannot_hide_a_removed_requester() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let bytes = api46_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    fixture.device.lock().unwrap().fail_final_health = true;
    assert!(
        fixture
            .adapter
            .start_init(&firmware, NonZeroUsize::MIN)
            .is_err()
    );
    {
        let device = fixture.device.lock().unwrap();
        assert!(device.events.contains(&Event::Write(LOAD_STATUS, u32::MAX)));
        assert!(device.events.contains(&Event::Read(LOAD_STATUS, u32::MAX)));
        assert!(device.gone);
    }
    assert!(fixture.adapter.poll_init_alive().is_err());
    assert!(fixture.adapter.finish().is_err());
    assert_eq!(fixture.domain.records.lock().unwrap().live.len(), 507);
    fixture.device.lock().unwrap().gone = false;
    fixture.adapter.stop().unwrap();
}

fn pinned_cnvi9000_api46_image() -> Vec<u8> {
    use sha2::{Digest, Sha256};

    let root = std::env::var_os("STELA_IWLWIFI_IMAGES").expect("actual pinned image directory");
    let path = std::path::Path::new(&root).join(Sku::Cnvi9000.firmware_name());
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "9a0c5d938646942f9798e6212579c25771e4bb44301cd409b82abe361ca2c3e0"
    );
    bytes
}

#[test]
#[ignore = "oracle: STELA_IWLWIFI_IMAGES"]
fn pinned_cnvi9000_api46_uploads_nine_init_sections_and_receives_alive() {
    let _clock = test_clock::virtualize();
    let bytes = pinned_cnvi9000_api46_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    fixture
        .adapter
        .start_init(&firmware, NonZeroUsize::new(0x10000).unwrap())
        .unwrap();
    {
        let device = fixture.device.lock().unwrap();
        let status: Vec<_> = device
            .events
            .iter()
            .filter_map(|event| match event {
                Event::Write(LOAD_STATUS, status) => Some(*status),
                _ => None,
            })
            .collect();
        assert_eq!(
            status,
            [1, 3, 7, 15, 31, 63, 127, 255, 511, 0xffff, u32::MAX]
        );
    }
    fixture.publish_alive();
    assert!(matches!(
        fixture.adapter.poll_init_alive(),
        Ok(Progress::Alive {
            image: Image::Init,
            ..
        })
    ));
    fixture.adapter.stop().unwrap();
    assert!(fixture.adapter.finish().is_ok());
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}
