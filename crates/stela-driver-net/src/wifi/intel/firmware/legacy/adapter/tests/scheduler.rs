// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::wifi::intel::firmware::legacy::{hcmd, nvm::Section};

pub(super) const MEM_READ_ADDRESS: u32 = 0x40c;
pub(super) const MEM_WRITE_ADDRESS: u32 = 0x410;
pub(super) const MEM_WRITE_DATA: u32 = 0x418;
pub(super) const MEM_READ_DATA: u32 = 0x41c;

const KEEP_WARM: u32 = 0x197c;
const FH_CHICKEN: u32 = 0x1e98;
const SCD_SRAM_BASE: u32 = 0x02c00;
const SCD_BYTE_COUNT: u32 = 0x02c08;
const SCD_QUEUE_READ: u32 = 0x02c68;
const SCD_QUEUE_CHAIN: u32 = 0x02ce8;
const SCD_QUEUE_STATUS: u32 = 0x02d0c;
const SCD_GP_CONTROL: u32 = 0x02da8;
const SCD_AGGR: u32 = 0x02e48;
const SCD_ENABLE: u32 = 0x02e54;
pub(super) const SRAM_BASE: u32 = 0x800000;
const SRAM_CONTEXT: u32 = SRAM_BASE + 0x600;
const SRAM_LAST: u32 = SRAM_BASE + 0x818;
const INACTIVE_QUEUE: u32 = 0x0008_0000;
const ACTIVE_QUEUE: u32 = 0x017f_001f;
const ENABLE_TX_CHANNEL: u32 = 0x8000_0008;
const TFD_BYTES: usize = 128;
const TX_PINS: usize = 35;
const RX_PINS: usize = 507;

#[derive(Default)]
pub(super) struct DeviceMemory {
    words: BTreeMap<u32, u32>,
    read: u32,
    write: u32,
    pub(super) alive: bool,
    fail_stage_at: Option<u32>,
    fail_queue_enable: bool,
}

impl DeviceMemory {
    pub(super) fn select(&mut self, register: u32, value: u32) {
        if register == MEM_READ_ADDRESS {
            self.read = value;
        } else {
            self.write = value;
        }
    }

    pub(super) fn read_data(&mut self) -> u32 {
        let word = self.words.get(&self.read).copied().unwrap_or_default();
        self.read += 4;
        word
    }

    pub(super) fn write_data(&mut self, value: u32) -> u32 {
        let address = self.write;
        self.words.insert(address, value);
        self.write += 4;
        address
    }
}

impl Device {
    pub(super) fn scheduler_csr_write(
        &mut self,
        domain: &Domain,
        register: u32,
        value: u32,
    ) -> Result<(), AccessError> {
        if register == cbcc(0) {
            assert_eq!(self.peripheral(SCD_TXFACT), 0);
            assert_eq!(self.peripheral(CPU_RELEASE), 0);
            let ring = u64::from(value) << 8;
            let pins = domain.records.lock().unwrap().live.len() - RX_PINS;
            assert_eq!(pins, TX_PINS);
            self.command.prepare(ring, pins);
        }
        if self.scheduler.fail_stage_at == Some(register) {
            self.scheduler.fail_stage_at = None;
            return Err(AccessError::Mapping);
        }
        Ok(())
    }

    pub(super) fn scheduler_prph_write(
        &mut self,
        address: u32,
        value: u32,
    ) -> Result<(), AccessError> {
        if address == SCD_QUEUE_STATUS && value & (1 << 3) != 0 {
            assert!(self.scheduler.alive);
            assert_eq!(self.csr(MASK), RX_MASK);
            self.command.activate();
            if core::mem::take(&mut self.scheduler.fail_queue_enable) {
                return Err(AccessError::Mapping);
            }
        }
        Ok(())
    }
}

fn cbcc(queue: u32) -> u32 {
    match queue {
        0..=15 => 0x19d0 + 4 * queue,
        16..=19 => 0x1bf0 + 4 * (queue - 16),
        20..=30 => 0x1b20 + 4 * (queue - 20),
        _ => panic!("invalid Gen1 TX queue"),
    }
}

fn event_index(device: &Device, event: Event) -> usize {
    device
        .events
        .iter()
        .position(|observed| *observed == event)
        .unwrap()
}

pub(super) fn configure_scheduler(fixture: &Fixture<'_>, base: u32) {
    let mut device = fixture.device.lock().unwrap();
    device
        .peripheral
        .extend([(SCD_SRAM_BASE, base), (SCD_GP_CONTROL, 0x80), (SCD_AGGR, 1)]);
    device.csr.insert(FH_CHICKEN, 0x80);
    for address in (SRAM_CONTEXT - 4..=SRAM_LAST + 4).step_by(4) {
        device.scheduler.words.insert(address, 0xabcd_0001);
    }
}

fn start(fixture: &mut Fixture<'_>) -> Result<(), Error> {
    let bytes = nvm::image_with_nvm_version([0x88, 1, 3, 0]);
    fixture
        .adapter
        .start_nvm_init(&Firmware::parse(&bytes).unwrap(), NonZeroUsize::MIN)
}

pub(super) fn accept_alive(fixture: &mut Fixture<'_>) {
    fixture.publish_alive_with_scheduler(SRAM_BASE);
    assert_eq!(fixture.adapter.activate_nvm_queue(), Err(Error::State));
    assert!(matches!(
        fixture.adapter.poll_init_alive(),
        Ok(Progress::Alive {
            image: Image::Init,
            ..
        })
    ));
}

#[test]
fn stages_all_gen1_rings_before_cpu_release_then_activates_only_queue_zero() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    configure_scheduler(&fixture, SRAM_BASE);
    start(&mut fixture).unwrap();

    let before_alive = {
        let device = fixture.device.lock().unwrap();
        let release = event_index(&device, Event::PeripheralWrite(CPU_RELEASE, 1 << 24));
        let keep_warm = device.csr(KEEP_WARM);
        assert_ne!(keep_warm, 0);
        assert!(event_index(&device, Event::Write(KEEP_WARM, keep_warm)) < release);
        assert!(event_index(&device, Event::PeripheralWrite(SCD_TXFACT, 0)) < release);
        assert!(
            event_index(&device, Event::PeripheralWrite(SCD_GP_CONTROL, 0x0004_0081)) < release
        );
        assert_eq!(
            read_dma::<8>(&fixture.domain, u64::from(keep_warm) << 4, 0),
            [0; 8]
        );
        let mut rings = alloc::collections::BTreeSet::new();
        let first: [u8; TFD_BYTES] =
            read_dma(&fixture.domain, u64::from(device.csr(cbcc(0))) << 8, 0);
        assert_eq!(first[..4], [0, 0, 0, 1]);
        assert_eq!(u16::from_le_bytes([first[8], first[9]]) >> 4, 8);
        for queue in 0..31 {
            let register = cbcc(queue);
            let value = device.csr(register);
            assert_ne!(value, 0);
            assert!(rings.insert(value));
            assert!(event_index(&device, Event::Write(register, value)) < release);
            let address = u64::from(value) << 8;
            assert_eq!(read_dma::<TFD_BYTES>(&fixture.domain, address, 0), first);
            assert_eq!(
                read_dma::<TFD_BYTES>(&fixture.domain, address, 255 * TFD_BYTES),
                first
            );
        }
        assert_eq!(device.peripheral(SCD_TXFACT), 0);
        assert_eq!(device.peripheral(SCD_ENABLE), 0);
        assert!(FH_TX_CHANNELS.into_iter().all(|reg| device.csr(reg) == 0));
        assert!(
            !device
                .events
                .iter()
                .any(|event| matches!(event, Event::SramWrite(..)))
        );
        assert_eq!(device.csr(MASK), RX_MASK);
        assert_eq!(device.csr(0xa0), 0);
        device.events.len()
    };
    assert_eq!(fixture.adapter.activate_nvm_queue(), Err(Error::State));
    assert_eq!(fixture.device.lock().unwrap().events.len(), before_alive);

    accept_alive(&mut fixture);
    fixture.adapter.activate_nvm_queue().unwrap();
    {
        let device = fixture.device.lock().unwrap();
        assert_eq!(device.peripheral(SCD_ENABLE), 1);
        assert_eq!(device.peripheral(SCD_QUEUE_STATUS), ACTIVE_QUEUE);
        for queue in 1..31 {
            let status = if queue < 20 {
                SCD_QUEUE_STATUS
            } else {
                0x02f34
            } + 4 * queue;
            assert_eq!(device.peripheral(status) & (1 << 3), 0);
        }
        let inactive = event_index(
            &device,
            Event::PeripheralWrite(SCD_QUEUE_STATUS, INACTIVE_QUEUE),
        );
        let active = event_index(
            &device,
            Event::PeripheralWrite(SCD_QUEUE_STATUS, ACTIVE_QUEUE),
        );
        let read_pointer = event_index(&device, Event::PeripheralWrite(SCD_QUEUE_READ, 0));
        let write_pointer = event_index(&device, Event::Write(COMMAND_DOORBELL, 0));
        assert!(inactive < read_pointer && inactive < write_pointer);
        assert!(read_pointer < active && write_pointer < active);
        let tx_fact = event_index(&device, Event::PeripheralWrite(SCD_TXFACT, 0xff));
        let disable_queues = event_index(&device, Event::PeripheralWrite(SCD_ENABLE, 0));
        let enable_queue_zero = event_index(&device, Event::PeripheralWrite(SCD_ENABLE, 1));
        assert!(active < enable_queue_zero && enable_queue_zero < tx_fact);
        assert!(disable_queues < event_index(&device, Event::SramWrite(SRAM_CONTEXT, 0)));
        assert!(event_index(&device, Event::SramWrite(SRAM_LAST, 0)) < active);
        let count = u64::from(device.peripheral(SCD_BYTE_COUNT)) << 10;
        assert_eq!(read_dma::<8>(&fixture.domain, count, 0), [0; 8]);
        for address in (SRAM_CONTEXT..=SRAM_LAST).step_by(4) {
            let expected = if address == SRAM_CONTEXT + 4 {
                0x0040_0040
            } else {
                0
            };
            assert_eq!(
                device.scheduler.words[&address], expected,
                "SRAM {address:#x}"
            );
        }
        for address in [SRAM_CONTEXT - 4, SRAM_LAST + 4] {
            assert_eq!(device.scheduler.words[&address], 0xabcd_0001);
        }
        assert_eq!(device.peripheral(SCD_AGGR) & 1, 0);
        assert_eq!(device.csr(FH_CHICKEN), 0x82);
        assert!(
            FH_TX_CHANNELS
                .into_iter()
                .all(|reg| device.csr(reg) == ENABLE_TX_CHANNEL)
        );
        assert_eq!(device.csr(MASK), RX_MASK);
        assert_eq!(device.csr(0xa0), 0);
    }
    nvm::complete_read(&mut fixture, Section::SW, 0, b"test");
    fixture.adapter.stop().unwrap();
    assert!(fixture.adapter.finish().is_ok());
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}

#[test]
#[ignore = "oracle: STELA_IWLWIFI_IMAGES"]
fn pinned_cnvi9000_api46_runs_concrete_scheduler_and_modeled_nvm_read() {
    let _clock = test_clock::virtualize();
    let bytes = pinned_cnvi9000_api46_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    configure_scheduler(&fixture, SRAM_BASE);
    fixture
        .adapter
        .start_nvm_init(&firmware, NonZeroUsize::new(0x10000).unwrap())
        .unwrap();
    accept_alive(&mut fixture);
    fixture.adapter.activate_nvm_queue().unwrap();
    nvm::complete_read(
        &mut fixture,
        Section::SW,
        0,
        &[0x1d, 0x0a, 2, 1, 0, 0, 5, 0, 0, 0],
    );
    fixture.adapter.stop().unwrap();
    assert!(fixture.adapter.finish().is_ok());
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}

#[test]
fn scheduler_requires_the_alive_sram_address_and_direct_rfh_causes() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    for ict_enabled in [false, true] {
        let mut fixture = Fixture::new(&source);
        let base = if ict_enabled {
            SRAM_BASE
        } else {
            SRAM_BASE + 4
        };
        configure_scheduler(&fixture, base);
        start(&mut fixture).unwrap();
        accept_alive(&mut fixture);
        if ict_enabled {
            fixture.device.lock().unwrap().csr.insert(0xa0, 1 << 31);
            assert!(matches!(
                fixture.adapter.activate_nvm_queue(),
                Err(Error::Service(_))
            ));
            fixture.device.lock().unwrap().csr.insert(0xa0, 0);
        } else {
            assert_eq!(
                fixture.adapter.activate_nvm_queue(),
                Err(Error::Command(hcmd::Error::SchedulerSram {
                    register: base,
                    alive: SRAM_BASE
                }))
            );
        }
        {
            let device = fixture.device.lock().unwrap();
            assert_eq!(device.peripheral(SCD_ENABLE), 0);
            assert_eq!(device.peripheral(SCD_TXFACT), 0);
            assert!(
                !device
                    .events
                    .iter()
                    .any(|event| matches!(event, Event::SramWrite(..)))
            );
        }
        fixture.adapter.stop().unwrap();
        assert!(fixture.adapter.finish().is_ok());
    }
}

#[test]
fn scheduler_rejects_chained_command_queue_before_publishing_queue_pointers() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    configure_scheduler(&fixture, SRAM_BASE);
    start(&mut fixture).unwrap();
    accept_alive(&mut fixture);
    fixture
        .device
        .lock()
        .unwrap()
        .peripheral
        .insert(SCD_QUEUE_CHAIN, 1);
    assert_eq!(
        fixture.adapter.activate_nvm_queue(),
        Err(Error::Command(hcmd::Error::Rejected {
            register: 0xa02ce8,
            observed: 1
        }))
    );
    {
        let device = fixture.device.lock().unwrap();
        assert_eq!(device.peripheral(SCD_ENABLE), 0);
        assert_eq!(device.peripheral(SCD_TXFACT), 0);
        assert!(!device.events.iter().any(|event| matches!(
            event,
            Event::Write(COMMAND_DOORBELL, _) | Event::PeripheralWrite(SCD_QUEUE_READ, _)
        )));
    }
    fixture.adapter.stop().unwrap();
    assert!(fixture.adapter.finish().is_ok());
}

#[test]
fn posted_gen1_staging_error_keeps_all_rings_and_never_releases_the_cpu() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    configure_scheduler(&fixture, SRAM_BASE);
    fixture.device.lock().unwrap().scheduler.fail_stage_at = Some(cbcc(7));
    assert!(start(&mut fixture).is_err());
    {
        let device = fixture.device.lock().unwrap();
        assert_ne!(
            device.csr(cbcc(7)),
            0,
            "the failed posted write took effect"
        );
        assert_eq!(device.csr(cbcc(8)), 0);
        assert_eq!(device.peripheral(CPU_RELEASE), 0);
    }
    assert_eq!(start(&mut fixture), Err(Error::State));
    assert!(fixture.adapter.finish().is_err());
    assert_eq!(fixture.domain.records.lock().unwrap().unmaps, 0);
    assert_eq!(
        fixture.domain.records.lock().unwrap().live.len(),
        RX_PINS + TX_PINS
    );
    fixture.adapter.stop().unwrap();
    assert!(fixture.adapter.finish().is_ok());
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}

#[test]
fn posted_scheduler_activation_failure_retains_requester_until_drain_and_invalidation_retry() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    configure_scheduler(&fixture, SRAM_BASE);
    start(&mut fixture).unwrap();
    accept_alive(&mut fixture);
    fixture.device.lock().unwrap().scheduler.fail_queue_enable = true;
    assert!(fixture.adapter.activate_nvm_queue().is_err());
    assert_eq!(fixture.adapter.activate_nvm_queue(), Err(Error::State));
    let unmaps = fixture.domain.records.lock().unwrap().unmaps;
    let stopping = {
        let mut device = fixture.device.lock().unwrap();
        assert_eq!(device.peripheral(SCD_QUEUE_STATUS), ACTIVE_QUEUE);
        device.fail_drain = true;
        device.events.len()
    };
    assert_eq!(
        fixture.adapter.stop(),
        Err(Error::Drain(DmaError::DrainUnavailable))
    );
    {
        let device = fixture.device.lock().unwrap();
        let stopped = &device.events[stopping..];
        let find = |event| {
            stopped
                .iter()
                .position(|observed| *observed == event)
                .unwrap()
        };
        let mask = find(Event::Write(MASK, 0));
        let scheduler = find(Event::PeripheralWrite(SCD_TXFACT, 0));
        let rf = find(Event::PeripheralWrite(RFH_DMA, 0));
        let master = find(Event::Write(RESET, MASTER_STOP));
        let drain = find(Event::Drain(unmaps));
        assert!(mask < scheduler && rf < master && master < drain);
        for channel in FH_TX_CHANNELS {
            let stop = find(Event::Write(channel, 0));
            assert!(scheduler < stop && stop < rf);
        }
    }
    assert!(fixture.adapter.finish().is_err());
    assert_eq!(fixture.domain.records.lock().unwrap().unmaps, unmaps);
    assert_eq!(
        fixture.domain.records.lock().unwrap().live.len(),
        RX_PINS + TX_PINS
    );
    fixture.domain.records.lock().unwrap().unmap_fail = Some(unmaps);
    assert!(fixture.adapter.stop().is_err());
    assert!(fixture.adapter.finish().is_err());
    fixture.adapter.stop().unwrap();
    assert!(fixture.adapter.finish().is_ok());
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}
