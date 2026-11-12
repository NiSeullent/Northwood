// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::wifi::intel::firmware::legacy::{
    adapter::CalibrationProgress,
    alive::{self, Completion},
    nvm::Section,
    phy_db::Kind,
};

const TX_ANTENNA: u8 = 0x98;
const PHY_CONFIGURATION: u8 = 0x6a;
const PHY_DB: u8 = 0x6b;
const INIT_COMPLETE: u8 = 0x04;
const UNSOLICITED: u16 = 1 << 15;
const RF_ENABLED: u32 = 1 << 27;
const FLOW: u32 = 0x1f8e_51db;
const FIRMWARE_PHY: u32 = 0x0033_0018;
const EFFECTIVE_PHY: u32 = 0x8012_0018;

pub(super) fn append_tlv(image: &mut Vec<u8>, kind: u32, payload: &[u8]) {
    image.extend(kind.to_le_bytes());
    image.extend((payload.len() as u32).to_le_bytes());
    image.extend_from_slice(payload);
    image.resize(image.len().next_multiple_of(4), 0);
}

pub(super) fn calibration_image() -> Vec<u8> {
    calibration_image_with_commands(&[])
}

pub(super) fn calibration_image_with_commands(additional_versions: &[u8]) -> Vec<u8> {
    let mut bytes = image();
    let mut command_versions = vec![
        1,
        0,
        99,
        3,
        0x88,
        1,
        3,
        0,
        TX_ANTENNA,
        1,
        1,
        0,
        PHY_CONFIGURATION,
        1,
        2,
        0,
        0x4f,
        1,
        1,
        0,
    ];
    command_versions.extend_from_slice(additional_versions);
    append_tlv(&mut bytes, CMD_VERSIONS, &command_versions);
    append_tlv(&mut bytes, 23, &FIRMWARE_PHY.to_le_bytes());
    let mut defaults = Vec::new();
    for word in [1, FLOW, 1] {
        defaults.extend(word.to_le_bytes());
    }
    append_tlv(&mut bytes, 22, &defaults);
    bytes
}

fn ready(fixture: &mut Fixture<'_>, rf_enabled: bool) {
    let bytes = calibration_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    ready_with_firmware(fixture, &firmware, NonZeroUsize::MIN, rf_enabled);
}

pub(super) fn ready_with_firmware(
    fixture: &mut Fixture<'_>,
    firmware: &Firmware<'_>,
    staging: NonZeroUsize,
    rf_enabled: bool,
) {
    scheduler::configure_scheduler(fixture, scheduler::SRAM_BASE);
    fixture.adapter.start_nvm_init(firmware, staging).unwrap();
    scheduler::accept_alive(fixture);
    fixture.adapter.activate_nvm_queue().unwrap();
    nvm::complete_read(
        fixture,
        Section::SW,
        0,
        &[0x1d, 0x0a, 2, 1, 0, 0, 5, 0, 0, 0],
    );
    // The NVM masks narrow the firmware's two chains to TX B and RX A.
    nvm::complete_read(
        fixture,
        Section::PHY_SKU,
        0,
        &[0x3c, 0x5b, 0x10, 0x12, 0x2f, 0, 0, 0],
    );
    {
        let mut device = fixture.device.lock().unwrap();
        device.csr.extend([
            (0x388, 0x0223_4567),
            (0x38c, 0x89ab),
            (GP, 1 | (u32::from(rf_enabled) * RF_ENABLED)),
        ]);
    }
    let identity = fixture.adapter.nvm_identity().unwrap();
    assert_eq!((identity.tx_antennas, identity.rx_antennas), (2, 1));
}

fn start(fixture: &mut Fixture<'_>) -> Result<(), Error> {
    // SAFETY: the fake is this adapter's exclusively owned 31dc board and
    // has no antenna transmit path, external producer or scheduler writer.
    unsafe { fixture.adapter.start_31dc_init_calibration() }
}

pub(super) type Packet<'a> = (u8, u8, u16, &'a [u8]);

pub(super) fn buffer(packets: &[Packet<'_>]) -> [u8; 4096] {
    let mut bytes = [0; 4096];
    let mut start = 0;
    for &(command, group, sequence, data) in packets {
        let end = start + 8 + data.len();
        bytes[start..start + 4].copy_from_slice(&(4 + data.len() as u32).to_le_bytes());
        bytes[start + 4] = command;
        bytes[start + 5] = group;
        bytes[start + 6..start + 8].copy_from_slice(&sequence.to_le_bytes());
        bytes[start + 8..end].copy_from_slice(data);
        start = end.next_multiple_of(64);
    }
    bytes[start..start + 4].copy_from_slice(&0x5555_0000u32.to_le_bytes());
    bytes
}

fn complete_command(fixture: &Fixture<'_>, sequence: u16) {
    fixture
        .device
        .lock()
        .unwrap()
        .command
        .complete(&fixture.domain, sequence);
}

fn pending(fixture: &mut Fixture<'_>) {
    assert!(matches!(
        fixture.adapter.poll_init_calibration(),
        Ok(CalibrationProgress::Pending)
    ));
}

pub(super) fn complete_current_init(fixture: &mut Fixture<'_>) {
    complete_current_init_with(
        fixture,
        &[(PHY_DB, 0, UNSOLICITED, &[1, 0, 2, 0, 0x10, 0x11])],
    );
}

pub(super) fn complete_current_init_with(fixture: &mut Fixture<'_>, notifications: &[Packet<'_>]) {
    start(fixture).unwrap();
    let antenna = nvm::submitted(fixture).sequence();
    complete_command(fixture, antenna);
    fixture.publish_rx(&buffer(&[(TX_ANTENNA, 1, antenna, &[])]));
    pending(fixture);
    let phy = nvm::submitted(fixture).sequence();
    complete_command(fixture, phy);
    let mut packets = notifications.to_vec();
    packets.extend([
        (PHY_CONFIGURATION, 1, phy, &[][..]),
        (INIT_COMPLETE, 0, UNSOLICITED, &[][..]),
    ]);
    fixture.publish_rx(&buffer(&packets));
    assert!(matches!(
        fixture.adapter.poll_init_calibration(),
        Ok(CalibrationProgress::Complete)
    ));
}

#[test]
fn calibration_collects_only_current_notifications_and_hands_off_after_stop() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    ready(&mut fixture, true);
    assert!(fixture.adapter.take_regular_handoff().is_err());

    // Two NVM reads used slots zero and one. This preclosed exact response
    // must not reclaim the first calibration command's still-live DMA.
    fixture.publish_rx(&buffer(&[(TX_ANTENNA, 1, 2, &[])]));
    start(&mut fixture).unwrap();
    let tx = nvm::submitted(&fixture);
    assert_eq!(tx.bytes(), [TX_ANTENNA, 1, 2, 0, 4, 0, 0, 0, 2, 0, 0, 0]);
    pending(&mut fixture);
    fixture.publish_rx(&buffer(&[
        (TX_ANTENNA, 0, 2, &[]),
        (TX_ANTENNA, 1, 0x2002, &[]),
    ]));
    pending(&mut fixture);
    assert_eq!(nvm::submitted(&fixture).sequence(), tx.sequence());

    complete_command(&fixture, tx.sequence());
    fixture.publish_rx(&buffer(&[(TX_ANTENNA, 1, tx.sequence(), &[])]));
    // These are closed while still in the TX_ANT stage. Even an exact PHY
    // response and early INIT_COMPLETE cannot pass the PHY doorbell cutoff.
    fixture.publish_rx(&buffer(&[
        (PHY_DB, 0, UNSOLICITED, &[2, 0, 1, 0, 0xff]),
        (PHY_CONFIGURATION, 1, 3, &[]),
        (INIT_COMPLETE, 0, UNSOLICITED, &[]),
    ]));
    pending(&mut fixture);
    let phy = nvm::submitted(&fixture);
    let mut expected = vec![PHY_CONFIGURATION, 1, 3, 0, 12, 0, 0, 0];
    for word in [EFFECTIVE_PHY, FLOW, 1] {
        expected.extend(word.to_le_bytes());
    }
    assert_eq!(phy.bytes(), expected);
    pending(&mut fixture);
    assert_eq!(nvm::submitted(&fixture).sequence(), phy.sequence());

    fixture.publish_rx(&buffer(&[
        (PHY_DB, 0, UNSOLICITED, &[1, 0, 2, 0, 0x10, 0x11]),
        (PHY_DB, 1, UNSOLICITED, &[4, 0, 3, 0, 2, 0, 0x12]),
        (PHY_DB, 0, UNSOLICITED, &[5, 0, 6, 0, 1, 0, 0, 0, 9, 0]),
        (PHY_DB, 0, phy.sequence(), &[1, 0, 1, 0, 0xee]),
        (INIT_COMPLETE, 1, UNSOLICITED, &[]),
        (PHY_DB, 0, UNSOLICITED, &[2, 0, 1, 0, 0xdd]),
        (PHY_CONFIGURATION, 1, phy.sequence() | UNSOLICITED, &[]),
    ]));
    pending(&mut fixture);
    assert_eq!(nvm::submitted(&fixture).sequence(), phy.sequence());
    assert!(fixture.adapter.take_regular_handoff().is_err());

    complete_command(&fixture, phy.sequence());
    fixture.publish_rx(&buffer(&[(PHY_CONFIGURATION, 1, phy.sequence(), &[])]));
    assert!(matches!(
        fixture.adapter.poll_init_calibration(),
        Ok(CalibrationProgress::Complete)
    ));
    assert!(fixture.adapter.take_regular_handoff().is_err());

    let (unmaps, live) = {
        let records = fixture.domain.records.lock().unwrap();
        (records.unmaps, records.live.len())
    };
    fixture.device.lock().unwrap().fail_drain = true;
    assert_eq!(
        fixture.adapter.stop(),
        Err(Error::Drain(DmaError::DrainUnavailable))
    );
    assert!(fixture.adapter.take_regular_handoff().is_err());
    assert_eq!(fixture.domain.records.lock().unwrap().unmaps, unmaps);
    assert_eq!(fixture.domain.records.lock().unwrap().live.len(), live);
    fixture.domain.records.lock().unwrap().unmap_fail = Some(unmaps);
    assert!(fixture.adapter.stop().is_err());
    assert!(fixture.adapter.take_regular_handoff().is_err());
    fixture.adapter.stop().unwrap();
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());

    let handoff = fixture.adapter.take_regular_handoff().unwrap();
    let database: Vec<_> = handoff
        .phy_db()
        .sections()
        .map(|section| (section.kind, section.group, section.data))
        .collect();
    assert_eq!(
        database,
        [
            (Kind::Configuration, None, &[0x10, 0x11][..]),
            (Kind::Papd, Some(2), &[2, 0, 0x12][..]),
            (Kind::TxPower, Some(1), &[1, 0, 0, 0, 9, 0][..]),
        ]
    );
    let (io, mut sequence, _) = handoff.into_parts();
    assert!(Arc::ptr_eq(&io.device, &fixture.device));
    assert!(Arc::ptr_eq(&io.domain, &fixture.domain));
    assert!(fixture.adapter.take_regular_handoff().is_err());

    assert_eq!(
        sequence.wait_regular_after_init_stop(1),
        Err(alive::Error::Generation {
            previous: 1,
            next: 1
        })
    );
    sequence.wait_regular_after_init_stop(2).unwrap();
    let mut payload = [0; 68];
    payload[..2].copy_from_slice(&0xcafeu16.to_le_bytes());
    let receipt = |generation| Completion {
        generation,
        index: 0,
        group: 0,
        command: 1,
        payload: &payload,
    };
    assert_eq!(sequence.poll(Some(receipt(1))), Ok(Progress::Pending));
    assert!(matches!(
        sequence.poll(Some(receipt(2))),
        Ok(Progress::Alive {
            image: Image::Regular,
            ..
        })
    ));
}

#[test]
fn missing_phy_ack_times_out_even_after_init_complete() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    ready(&mut fixture, true);
    start(&mut fixture).unwrap();
    let tx = nvm::submitted(&fixture).sequence();
    complete_command(&fixture, tx);
    fixture.publish_rx(&buffer(&[(TX_ANTENNA, 1, tx, &[])]));
    pending(&mut fixture);
    let phy = nvm::submitted(&fixture);
    fixture.publish_rx(&buffer(&[(INIT_COMPLETE, 0, UNSOLICITED, &[])]));
    pending(&mut fixture);
    test_clock::advance(
        Clock {
            ticks_per_microsecond: 1,
        },
        2_000_001,
    );
    assert!(matches!(
        fixture.adapter.poll_init_calibration(),
        Err(Error::CommandTimeout)
    ));
    assert_eq!(nvm::submitted(&fixture).sequence(), phy.sequence());
    assert!(fixture.adapter.take_regular_handoff().is_err());
    fixture.adapter.stop().unwrap();
    assert!(fixture.adapter.take_regular_handoff().is_err());
    assert!(fixture.adapter.finish().is_ok());
}

#[test]
fn rfkill_refuses_calibration_before_publishing_a_command() {
    let _clock = test_clock::virtualize();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    ready(&mut fixture, false);
    let before = fixture.domain.records.lock().unwrap().syncs;
    assert_eq!(start(&mut fixture), Err(Error::RfKilled));
    assert_eq!(fixture.domain.records.lock().unwrap().syncs, before);
    assert_eq!(fixture.device.lock().unwrap().csr(COMMAND_DOORBELL), 2);
    assert!(fixture.adapter.take_regular_handoff().is_err());
    fixture.adapter.stop().unwrap();
    assert!(fixture.adapter.take_regular_handoff().is_err());
    assert!(fixture.adapter.finish().is_ok());
}
