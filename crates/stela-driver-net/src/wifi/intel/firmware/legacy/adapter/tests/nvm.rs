// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::wifi::intel::{
    firmware::legacy::{adapter::NvmProgress, nvm},
    nvm::AddressSource,
};

const TFD_BYTES: usize = 128;
const TB0_BYTES: usize = 64;
const PAGING_TB1_BYTES: usize = 33 * 4;
const NVM_OPCODE: u8 = 0x88;
const NVM_WIRE_GROUP: u8 = 1;
const TX_IDLE: u32 = 0x00ff_0000;
const TEST_CLOCK: Clock = Clock {
    ticks_per_microsecond: 1,
};

#[derive(Clone)]
pub(super) struct Submitted {
    slot: usize,
    tb0: u64,
    tb1: Option<u64>,
    bytes: Vec<u8>,
}

impl Submitted {
    pub(super) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(super) fn sequence(&self) -> u16 {
        u16::from_le_bytes([self.bytes[2], self.bytes[3]])
    }

    pub(super) fn remainder(&self) -> Option<u64> {
        self.tb1
    }
}

#[derive(Default)]
pub(super) struct CommandDevice {
    ring: Option<u64>,
    pins: usize,
    active: bool,
    next_slot: usize,
    inflight: Option<Submitted>,
    paging: Vec<u64>,
    fail_doorbell: bool,
    hold_tx_busy: bool,
    stopped_channels: u32,
}

impl CommandDevice {
    pub(super) fn prepare(&mut self, ring: u64, pins: usize) {
        self.ring = Some(ring);
        self.pins = pins;
    }

    pub(super) fn activate(&mut self) {
        assert!(self.ring.is_some());
        assert!(!self.active);
        self.active = true;
    }

    pub(super) fn prepared_pins(&self) -> usize {
        self.pins
    }

    pub(super) fn publish(&mut self, domain: &Domain, pointer: u32) -> Result<(), AccessError> {
        assert!(self.active, "a command needs an activated SCD queue");
        assert!(self.inflight.is_none());
        let slot = self.next_slot;
        assert_eq!(pointer, (slot + 1) as u32, "Gen1 queue zero pointer");
        let descriptor: [u8; TFD_BYTES] = read_dma(domain, self.ring.unwrap(), slot * TFD_BYTES);
        assert_eq!(descriptor[..3], [0; 3]);
        let count = usize::from(descriptor[3]);
        assert!((1..=2).contains(&count));
        assert!(descriptor[4 + count * 6..].iter().all(|byte| *byte == 0));
        let (tb0, length) = tb0(&descriptor);
        let first = usize::from(length);
        assert!((8..=TB0_BYTES).contains(&first));
        let mut bytes = read_dma::<TB0_BYTES>(domain, tb0, 0)[..first].to_vec();
        let tb1 = if count == 2 {
            assert_eq!(first, 20);
            let (address, size) = tb(&descriptor, 10);
            bytes.resize(first + usize::from(size), 0);
            read_dma_into(domain, address, 0, &mut bytes[first..]);
            assert_eq!(
                domain.records.lock().unwrap().permissions[&dma_offset(address).0],
                Direction::ToDevice
            );
            if bytes[0] == 0x4f {
                assert_eq!(usize::from(size), PAGING_TB1_BYTES);
                self.paging = bytes[first..]
                    .chunks_exact(4)
                    .map(|word| u64::from(u32::from_le_bytes(word.try_into().unwrap())) << 12)
                    .filter(|address| *address != 0)
                    .collect();
            }
            Some(address)
        } else {
            None
        };
        assert_eq!(
            bytes.len(),
            8 + usize::from(u16::from_le_bytes([bytes[4], bytes[5]]))
        );
        assert_eq!(tb0 % TB0_BYTES as u64, 0);
        self.next_slot += 1;
        self.inflight = Some(Submitted {
            slot,
            tb0,
            tb1,
            bytes,
        });
        if core::mem::take(&mut self.fail_doorbell) {
            return Err(AccessError::Mapping);
        }
        Ok(())
    }

    pub(super) fn complete(&mut self, domain: &Domain, sequence: u16) {
        let command = self.inflight.take().unwrap();
        assert_eq!(command.sequence(), sequence);
        let (ring, ring_base) = dma_offset(self.ring.unwrap());
        let tfd_start = ring_base + command.slot * TFD_BYTES;
        let (tb0, tb0_start) = dma_offset(command.tb0);
        let mut records = domain.records.lock().unwrap();
        assert!(
            records
                .completed_ranges
                .insert((ring, tfd_start, tfd_start + TFD_BYTES))
        );
        assert!(
            records
                .completed_ranges
                .insert((tb0, tb0_start, tb0_start + TB0_BYTES))
        );
    }

    pub(super) fn assert_can_unmap(&self, id: u64) {
        assert!(
            self.paging
                .iter()
                .all(|&address| dma_offset(address).0 != id)
        );
        if let Some(command) = &self.inflight {
            assert_ne!(id, dma_offset(self.ring.unwrap()).0);
            assert_ne!(id, dma_offset(command.tb0).0);
            if let Some(address) = command.tb1 {
                assert_ne!(id, dma_offset(address).0);
            }
        }
    }

    pub(super) fn fail_next_doorbell(&mut self) {
        self.fail_doorbell = true;
    }

    pub(super) fn stop_scheduler(&mut self) {
        self.active = false;
    }

    pub(super) fn stop_channel(&mut self, channel: u32) {
        assert!(!self.active, "scheduler must stop before FH channels");
        self.stopped_channels |= 1 << (16 + channel);
    }

    pub(super) fn enable_channel(&mut self, channel: u32) {
        self.stopped_channels &= !(1 << (16 + channel));
    }

    pub(super) fn tx_status(&self) -> u32 {
        if self.hold_tx_busy {
            0
        } else {
            self.stopped_channels
        }
    }

    pub(super) fn reset(&mut self) {
        if self.ring.is_some() {
            assert!(!self.active);
            assert_eq!(self.tx_status(), TX_IDLE);
        }
        self.inflight = None;
        self.paging.clear();
    }
}

fn tb0(descriptor: &[u8; TFD_BYTES]) -> (u64, u16) {
    tb(descriptor, 4)
}

fn tb(descriptor: &[u8; TFD_BYTES], offset: usize) -> (u64, u16) {
    let low = u32::from_le_bytes(descriptor[offset..offset + 4].try_into().unwrap());
    let high_and_length =
        u16::from_le_bytes(descriptor[offset + 4..offset + 6].try_into().unwrap());
    (
        u64::from(low) | (u64::from(high_and_length & 0xf) << 32),
        high_and_length >> 4,
    )
}

pub(super) fn image_with_nvm_version(nvm_abi: [u8; 4]) -> Vec<u8> {
    let mut bytes = api46_image();
    // The last TLV has one four-byte ALIVE entry. Extend that same TLV
    // with NVM_ACCESS rather than adding a second command-version table.
    let length = bytes.len() - 8;
    bytes[length..length + 4].copy_from_slice(&8u32.to_le_bytes());
    bytes.extend(nvm_abi);
    bytes
}

#[test]
fn rejects_missing_or_wrong_nvm_abi_before_setup_and_command_mappings() {
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let before = fixture.domain.records.lock().unwrap().maps;
    for (bytes, expected) in [
        (api46_image(), nvm::Error::MissingApi),
        (
            image_with_nvm_version([0x88, 0, 3, 0]),
            nvm::Error::MissingApi,
        ),
        (
            image_with_nvm_version([0x88, 1, 2, 0]),
            nvm::Error::UnsupportedApi {
                command: 2,
                notification: 0,
            },
        ),
        (
            image_with_nvm_version([0x88, 1, 3, 1]),
            nvm::Error::UnsupportedApi {
                command: 3,
                notification: 1,
            },
        ),
    ] {
        let firmware = Firmware::parse(&bytes).unwrap();
        // SAFETY: the fake has no prior ownership or hardware activity.
        // An invalid ABI is rejected before entering the preparation callback.
        let result = unsafe {
            fixture
                .adapter
                .start_init_with_nvm(&firmware, NonZeroUsize::MIN, |_, _| {
                    panic!("NVM preparation ran without the supported firmware ABI")
                })
        };
        assert_eq!(result, Err(Error::Nvm(expected)));
        assert!(fixture.device.lock().unwrap().events.is_empty());
        assert_eq!(fixture.domain.records.lock().unwrap().maps, before);
    }
    fixture.adapter.stop().unwrap();
    assert!(fixture.adapter.finish().is_ok());
}

fn start_nvm_fixture(fixture: &mut Fixture<'_>) {
    let bytes = image_with_nvm_version([0x88, 1, 3, 0]);
    let firmware = Firmware::parse(&bytes).unwrap();
    start_nvm_with_firmware(fixture, &firmware, NonZeroUsize::MIN);
}

fn start_nvm_with_firmware(
    fixture: &mut Fixture<'_>,
    firmware: &Firmware<'_>,
    staging_capacity: NonZeroUsize,
) {
    // SAFETY: the callback stands in for platform FH preparation on this
    // synchronous fake. It records the driver's exact device-visible ring.
    unsafe {
        fixture
            .adapter
            .start_init_with_nvm(firmware, staging_capacity, |io, layout| {
                assert_eq!(layout.entries, 256);
                assert_eq!(layout.entry_bytes, TFD_BYTES);
                assert_ne!(layout.keep_warm, layout.byte_count);
                assert_eq!(read_dma::<8>(&io.domain, layout.keep_warm, 0), [0; 8]);
                assert_eq!(read_dma::<8>(&io.domain, layout.byte_count, 0), [0; 8]);
                let descriptor: [u8; TFD_BYTES] = read_dma(&io.domain, layout.ring, 0);
                assert_eq!(descriptor[..4], [0, 0, 0, 1]);
                let (invalid, length) = tb0(&descriptor);
                assert_eq!(length, 8);
                assert_eq!(
                    read_dma::<8>(&io.domain, invalid, 0),
                    [6, 0x0f, 0xff, 0xff, 0, 0, 0, 0]
                );
                let mut device = io.device.lock().unwrap();
                let pins = io.domain.records.lock().unwrap().live.len() - 507;
                device.command.prepare(layout.ring, pins);
                assert_eq!(pins, 5);
                Ok(())
            })
    }
    .unwrap();
    assert_eq!(fixture.adapter.start_nvm_read(request()), Err(Error::State));
    fixture.publish_alive();
    assert!(matches!(
        fixture.adapter.poll_init_alive(),
        Ok(Progress::Alive {
            image: Image::Init,
            ..
        })
    ));
    // SAFETY: this callback admits the mock scheduler only for the already
    // recorded layout. No other queue or requester shares its DMA.
    unsafe {
        fixture.adapter.activate_nvm_commands(|io, layout| {
            let mut device = io.device.lock().unwrap();
            assert_eq!(device.command.ring, Some(layout.ring));
            device.command.activate();
            device.peripheral.insert(SCD_TXFACT, 1);
            Ok(())
        })
    }
    .unwrap();
}

fn request() -> nvm::Request {
    nvm::Request::new(nvm::Section::SW, 16, 4).unwrap()
}

fn reply_payload(data: [u8; 4]) -> [u8; 12] {
    [16, 0, 4, 0, 1, 0, 0, 0, data[0], data[1], data[2], data[3]]
}

fn reply(opcode: u8, sequence: u16, payload: &[u8]) -> [u8; 4096] {
    let mut frame = [0; 4096];
    frame[..4].copy_from_slice(&(4 + payload.len() as u32).to_le_bytes());
    frame[4] = opcode;
    frame[5] = NVM_WIRE_GROUP;
    frame[6..8].copy_from_slice(&sequence.to_le_bytes());
    frame[8..8 + payload.len()].copy_from_slice(payload);
    frame[64..68].copy_from_slice(&0x5555_0000u32.to_le_bytes());
    frame
}

pub(super) fn submitted(fixture: &Fixture<'_>) -> Submitted {
    fixture
        .device
        .lock()
        .unwrap()
        .command
        .inflight
        .clone()
        .unwrap()
}

pub(super) fn complete_read(
    fixture: &mut Fixture<'_>,
    section: nvm::Section,
    offset: u16,
    data: &[u8],
) {
    let length = u16::try_from(data.len()).unwrap();
    fixture
        .adapter
        .start_nvm_read(nvm::Request::new(section, offset, length).unwrap())
        .unwrap();
    let sequence = submitted(fixture).sequence();
    fixture
        .device
        .lock()
        .unwrap()
        .command
        .complete(&fixture.domain, sequence);
    let mut payload = Vec::new();
    for word in [offset, length, section.id(), 0] {
        payload.extend(word.to_le_bytes());
    }
    payload.extend_from_slice(data);
    fixture.publish_rx(&reply(NVM_OPCODE, sequence, &payload));
    assert!(
        matches!(fixture.adapter.poll_nvm_read(), Ok(NvmProgress::Data(bytes)) if bytes == data)
    );
}

#[test]
fn identity_uses_matched_zero_offset_nvm_data_and_same_owner_mac_reads() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    start_nvm_fixture(&mut fixture);
    let sw = [0x1d, 0x0a, 0x02, 0x01, 0, 0, 5, 0, 0, 0];
    let phy_sku = [0x3c, 0x5b, 0x10, 0xa7, 0x2f, 0, 0, 0];
    let events_before = fixture.device.lock().unwrap().events.len();
    assert_eq!(fixture.adapter.nvm_identity(), Err(Error::State));
    complete_read(&mut fixture, nvm::Section::SW, 1, &sw);
    complete_read(&mut fixture, nvm::Section::PHY_SKU, 0, &phy_sku[..7]);
    assert_eq!(fixture.adapter.nvm_identity(), Err(Error::State));
    complete_read(&mut fixture, nvm::Section::SW, 0, &sw);
    assert_eq!(fixture.adapter.nvm_identity(), Err(Error::State));
    complete_read(&mut fixture, nvm::Section::PHY_SKU, 0, &phy_sku);

    let strap_reads = [Event::Read(0x388, 0x0223_4567), Event::Read(0x38c, 0x89ab)];
    let otp_reads = [Event::Read(0x380, 0x0467_89ab), Event::Read(0x384, 0xcdef)];
    {
        let mut device = fixture.device.lock().unwrap();
        assert!(mac_reads(&device.events[events_before..]).is_empty());
        device.csr.extend([
            (0x388, 0x0223_4567),
            (0x38c, 0x89ab),
            (0x380, 0x0467_89ab),
            (0x384, 0xcdef),
        ]);
    }
    let identity = fixture.adapter.nvm_identity().unwrap();
    assert_eq!(identity.address.source, AddressSource::Strap);
    assert_eq!(identity.address.bytes, [2, 0x23, 0x45, 0x67, 0x89, 0xab]);
    assert_eq!(identity.nvm_version, 0x0102_0a1d);
    assert_eq!(identity.hardware_addresses, 5);
    assert_eq!(identity.radio_configuration, 0xa710_5b3c);
    assert_eq!(identity.sku_flags, 0x2f);
    let before_fallback = {
        let mut device = fixture.device.lock().unwrap();
        assert_eq!(mac_reads(&device.events[events_before..]), strap_reads);
        device.csr.insert(0x388, 0);
        device.csr.insert(0x38c, 0);
        device.events.len()
    };
    let identity = fixture.adapter.nvm_identity().unwrap();
    assert_eq!(identity.address.source, AddressSource::Otp);
    assert_eq!(identity.address.bytes, [4, 0x67, 0x89, 0xab, 0xcd, 0xef]);
    assert_eq!(
        mac_reads(&fixture.device.lock().unwrap().events[before_fallback..]),
        [
            Event::Read(0x388, 0),
            Event::Read(0x38c, 0),
            otp_reads[0],
            otp_reads[1],
        ]
    );
    fixture.adapter.stop().unwrap();
    assert!(fixture.adapter.finish().is_ok());
}

fn mac_reads(events: &[Event]) -> Vec<Event> {
    events
        .iter()
        .copied()
        .filter(|event| matches!(event, Event::Read(0x380 | 0x384 | 0x388 | 0x38c, _)))
        .collect()
}

#[test]
fn nvm_read_matches_fresh_full_ticket_and_reclaims_its_slot() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    start_nvm_fixture(&mut fixture);
    let before = fixture.domain.records.lock().unwrap().live.len();

    fixture.publish_rx(&reply(NVM_OPCODE, 0, &reply_payload(*b"old!")));
    fixture.adapter.start_nvm_read(request()).unwrap();
    let command = submitted(&fixture);
    assert_eq!(
        command.bytes(),
        [0x88, 1, 0, 0, 8, 0, 0, 0, 0, 0, 1, 0, 16, 0, 4, 0]
    );
    assert!(matches!(
        fixture.adapter.poll_nvm_read(),
        Ok(NvmProgress::Pending)
    ));

    for (group, opcode, sequence) in [
        (1, NVM_OPCODE, 0x2000),
        (1, NVM_OPCODE, 0x8000),
        (1, 0x89, 0),
        (0, NVM_OPCODE, 0),
    ] {
        let mut frame = reply(opcode, sequence, &reply_payload(*b"nope"));
        frame[5] = group;
        fixture.publish_rx(&frame);
        assert!(matches!(
            fixture.adapter.poll_nvm_read(),
            Ok(NvmProgress::Pending)
        ));
        assert!(fixture.device.lock().unwrap().command.inflight.is_some());
        assert_eq!(fixture.domain.records.lock().unwrap().live.len(), before);
    }

    let sequence = command.sequence();
    fixture
        .device
        .lock()
        .unwrap()
        .command
        .complete(&fixture.domain, sequence);
    fixture.publish_rx(&reply(NVM_OPCODE, sequence, &reply_payload(*b"test")));
    assert!(
        matches!(fixture.adapter.poll_nvm_read(), Ok(NvmProgress::Data(bytes)) if bytes == b"test")
    );
    let ring = fixture.device.lock().unwrap().command.ring.unwrap();
    let (_, length) = tb0(&read_dma(&fixture.domain, ring, 0));
    assert_eq!(
        length, 8,
        "the reclaimed TFD holds the invalid command again"
    );

    fixture.adapter.start_nvm_read(request()).unwrap();
    let sequence = submitted(&fixture).sequence();
    assert_eq!(sequence, 1);
    fixture
        .device
        .lock()
        .unwrap()
        .command
        .complete(&fixture.domain, sequence);
    fixture.publish_rx(&reply(NVM_OPCODE, sequence, &[16, 0, 0, 0, 1, 0, 1, 0]));
    assert!(matches!(
        fixture.adapter.poll_nvm_read(),
        Ok(NvmProgress::End)
    ));

    let at_start = nvm::Request::new(nvm::Section::SW, 0, 4).unwrap();
    fixture.adapter.start_nvm_read(at_start).unwrap();
    let sequence = submitted(&fixture).sequence();
    assert_eq!(sequence, 2);
    fixture
        .device
        .lock()
        .unwrap()
        .command
        .complete(&fixture.domain, sequence);
    fixture.publish_rx(&reply(NVM_OPCODE, sequence, &[0, 0, 0, 0, 1, 0, 1, 0]));
    assert!(matches!(
        fixture.adapter.poll_nvm_read(),
        Ok(NvmProgress::Absent)
    ));
    fixture.adapter.stop().unwrap();
    assert!(fixture.adapter.finish().is_ok());
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}

#[test]
fn malformed_or_duplicate_packet_in_one_rb_never_reclaims_nvm() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    for second_header in [0u32, 16] {
        let mut fixture = Fixture::new(&source);
        start_nvm_fixture(&mut fixture);
        fixture.adapter.start_nvm_read(request()).unwrap();
        let sequence = submitted(&fixture).sequence();
        let valid = reply(NVM_OPCODE, sequence, &reply_payload(*b"test"));
        let mut malformed = valid;
        malformed[64..128].copy_from_slice(&valid[..64]);
        malformed[64..68].copy_from_slice(&second_header.to_le_bytes());
        malformed[128..132].copy_from_slice(&0x5555_0000u32.to_le_bytes());
        fixture.publish_rx(&malformed);
        let result = fixture.adapter.poll_nvm_read();
        if second_header == 16 {
            assert_eq!(result, Err(Error::DuplicateNvmReply));
        } else {
            assert!(matches!(result, Err(Error::Receiver(_))));
        }
        assert!(fixture.device.lock().unwrap().command.inflight.is_some());
        fixture.adapter.stop().unwrap();
        assert!(fixture.adapter.finish().is_ok());
    }
}

#[test]
fn nvm_timeout_rejects_a_late_closed_reply_and_requires_stop() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    start_nvm_fixture(&mut fixture);
    fixture.adapter.start_nvm_read(request()).unwrap();
    test_clock::advance(TEST_CLOCK, 2_000_001);
    let sequence = submitted(&fixture).sequence();
    fixture.publish_rx(&reply(NVM_OPCODE, sequence, &reply_payload(*b"late")));
    assert_eq!(fixture.adapter.poll_nvm_read(), Err(Error::CommandTimeout));
    assert_eq!(fixture.adapter.start_nvm_read(request()), Err(Error::State));
    assert!(fixture.adapter.finish().is_err());
    assert!(fixture.device.lock().unwrap().command.inflight.is_some());
    fixture.adapter.stop().unwrap();
    assert!(fixture.adapter.finish().is_ok());
}

#[test]
fn posted_nvm_error_keeps_dma_until_tx_idle_drain_and_invalidation_retry() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    start_nvm_fixture(&mut fixture);
    let (unmaps, live) = {
        let records = fixture.domain.records.lock().unwrap();
        (records.unmaps, records.live.len())
    };
    {
        let mut device = fixture.device.lock().unwrap();
        device.command.fail_doorbell = true;
        device.command.hold_tx_busy = true;
    }
    assert!(matches!(
        fixture.adapter.start_nvm_read(request()),
        Err(Error::Command(_))
    ));
    let attempted = fixture.device.lock().unwrap().events.len();
    assert_eq!(fixture.adapter.start_nvm_read(request()), Err(Error::State));
    assert_eq!(fixture.device.lock().unwrap().events.len(), attempted);
    assert!(fixture.adapter.stop().is_err());
    {
        let mut device = fixture.device.lock().unwrap();
        assert!(device.command.inflight.is_some());
        assert!(!device.reset_settled);
        assert_eq!(device.command.stopped_channels, TX_IDLE);
        assert!(!device.command.active);
        assert!(!device.drained);
        device.command.hold_tx_busy = false;
        device.fail_drain = true;
    }
    assert_eq!(fixture.domain.records.lock().unwrap().unmaps, unmaps);
    assert!(matches!(
        fixture.adapter.stop(),
        Err(Error::Drain(DmaError::DrainUnavailable))
    ));
    assert!(fixture.adapter.finish().is_err());
    assert_eq!(fixture.domain.records.lock().unwrap().unmaps, unmaps);
    assert_eq!(fixture.domain.records.lock().unwrap().live.len(), live);
    fixture.domain.records.lock().unwrap().unmap_fail = Some(unmaps);
    assert!(fixture.adapter.stop().is_err());
    assert!(fixture.adapter.finish().is_err());
    assert_eq!(fixture.domain.records.lock().unwrap().live.len(), live);
    let stopped = fixture.device.lock().unwrap().events.len();
    fixture.adapter.stop().unwrap();
    assert_eq!(fixture.device.lock().unwrap().events.len(), stopped);
    assert!(fixture.adapter.finish().is_ok());
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}
