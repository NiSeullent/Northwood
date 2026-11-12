// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::wifi::intel::firmware::legacy::adapter::{
    PagingProgress, PhyDbProgress, RegularAdapter,
};

const PAGING: u8 = 0x4f;
const TX_ANTENNA: u8 = 0x98;
const PHY_DB_NOTIFICATION: u8 = 0x6b;
const PHY_DB_COMMAND: u8 = 0x6c;
const UNSOLICITED: u16 = 1 << 15;

const CFG: &[u8] = &[0x10, 0x11];
const PAPD_LOW: &[u8] = &[0, 0, 0x23];
const PAPD_HIGH: &[u8] = &[2, 0, 0x32, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
const TXP_LOW: &[u8] = &[0, 0, 0, 0, 4, 0];
const TXP_HIGH: &[u8] = &[1, 0, 0, 0, 9, 0, 0x40, 7, 8, 9, 10, 11];

pub(super) fn start_regular_with_db<'a, 'fw>(
    fixture: &mut Fixture<'a>,
    firmware: &Firmware<'fw>,
    configuration: &[u8],
) -> RegularAdapter<'a, 'fw, Io> {
    calibration::ready_with_firmware(fixture, firmware, regular::STAGING, true);
    // The first notification in each grouped family declares its highest
    // group ID. Lower groups can arrive out of replay order and leave gaps.
    let notifications: Vec<_> = [
        (5u16, TXP_HIGH),
        (4, PAPD_HIGH),
        (1, &[0xaa][..]),
        (4, PAPD_LOW),
        (2, &[]),
        (1, configuration),
        (5, TXP_LOW),
    ]
    .into_iter()
    .map(|(kind, data)| {
        let mut payload = kind.to_le_bytes().to_vec();
        payload.extend((data.len() as u16).to_le_bytes());
        payload.extend_from_slice(data);
        payload
    })
    .collect();
    let packets: Vec<_> = notifications
        .iter()
        .map(|payload| (PHY_DB_NOTIFICATION, 0, UNSOLICITED, payload.as_slice()))
        .collect();
    calibration::complete_current_init_with(fixture, &packets);
    paging::handoff_to_regular(fixture, firmware)
}

pub(super) fn acknowledge(fixture: &Fixture<'_>, index: &mut u16, opcode: u8) {
    acknowledge_in_group(fixture, index, opcode, 1);
}

pub(super) fn acknowledge_in_group(fixture: &Fixture<'_>, index: &mut u16, opcode: u8, group: u8) {
    let command = nvm::submitted(fixture);
    assert_eq!(command.bytes()[..2], [opcode, group]);
    let sequence = command.sequence();
    fixture
        .device
        .lock()
        .unwrap()
        .command
        .complete(&fixture.domain, sequence);
    paging::post(fixture, index, &[(opcode, group, sequence, &[])]);
}

pub(super) fn complete_paging(
    fixture: &Fixture<'_>,
    regular: &mut RegularAdapter<'_, '_, Io>,
    index: &mut u16,
) -> Vec<u64> {
    regular.start_paging().unwrap();
    let command = nvm::submitted(fixture);
    let pages = command.bytes()[20..]
        .chunks_exact(4)
        .map(|word| u64::from(u32::from_le_bytes(word.try_into().unwrap())) << 12)
        .filter(|&address| address != 0)
        .collect();
    acknowledge(fixture, index, PAGING);
    assert!(matches!(
        regular.poll_paging(),
        Ok(PagingProgress::Complete)
    ));
    pages
}

pub(super) fn paging_remains_pinned(fixture: &Fixture<'_>, pages: &[u64]) {
    let records = fixture.domain.records.lock().unwrap();
    assert!(
        pages
            .iter()
            .all(|&address| records.live.contains_key(&dma_offset(address).0))
    );
}

fn pending(regular: &mut RegularAdapter<'_, '_, Io>) {
    assert!(matches!(
        regular.poll_phy_db_replay(),
        Ok(PhyDbProgress::Pending)
    ));
}

#[test]
#[ignore = "oracle: STELA_IWLWIFI_IMAGES"]
fn pinned_31dc_replays_captured_db_in_wire_order_and_waits_for_each_fresh_ack() {
    let _clock = test_clock::virtualize();
    let bytes = pinned_cnvi9000_api46_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut regular = start_regular_with_db(&mut fixture, &firmware, CFG);
    let mut index = 0;
    paging::accept_alive(&fixture, &mut regular, &mut index);
    assert_eq!(regular.start_phy_db_replay(), Err(Error::State));
    let pages = complete_paging(&fixture, &mut regular, &mut index);
    assert_eq!(pages.len(), 9);

    paging::post(&fixture, &mut index, &[(TX_ANTENNA, 1, 1, &[])]);
    regular.start_phy_db_replay().unwrap();
    assert_eq!(
        nvm::submitted(&fixture).bytes(),
        [TX_ANTENNA, 1, 1, 0, 4, 0, 0, 0, 2, 0, 0, 0]
    );
    pending(&mut regular);
    acknowledge(&fixture, &mut index, TX_ANTENNA);
    // This exact CFG ACK is already closed while TX_ANT is pending. After
    // its ACK posts CFG, the new per-command cutoff must still exclude it.
    paging::post(&fixture, &mut index, &[(PHY_DB_COMMAND, 1, 2, &[])]);
    pending(&mut regular);
    assert_eq!(nvm::submitted(&fixture).sequence(), 2);
    pending(&mut regular);
    paging::post(
        &fixture,
        &mut index,
        &[
            (PHY_DB_COMMAND, 0, 2, &[]),
            (PHY_DB_COMMAND, 1, 0x2002, &[]),
            (PHY_DB_COMMAND, 1, UNSOLICITED | 2, &[]),
            (TX_ANTENNA, 1, 2, &[]),
        ],
    );
    pending(&mut regular);

    let expected = [
        (1u16, CFG),
        (2, &[][..]),
        (4, PAPD_LOW),
        (4, PAPD_HIGH),
        (5, TXP_LOW),
        (5, TXP_HIGH),
    ];
    for (i, &(kind, data)) in expected.iter().enumerate() {
        let command = nvm::submitted(&fixture);
        let sequence = i as u8 + 2;
        let payload_len = (4 + data.len()) as u8;
        assert_eq!(
            &command.bytes()[..8],
            [PHY_DB_COMMAND, 1, sequence, 0, payload_len, 0, 0, 0]
        );
        assert_eq!(&command.bytes()[8..10], kind.to_le_bytes());
        assert_eq!(&command.bytes()[10..12], (data.len() as u16).to_le_bytes());
        assert_eq!(&command.bytes()[12..], data);
        let remainder = command.remainder().map(|address| dma_offset(address).0);
        assert_eq!(remainder.is_some(), data.len() > 8);
        if let Some(id) = remainder {
            assert_eq!(
                fixture.domain.records.lock().unwrap().permissions[&id],
                Direction::ToDevice
            );
        }
        acknowledge(&fixture, &mut index, PHY_DB_COMMAND);
        let progress = regular.poll_phy_db_replay();
        if i + 1 == expected.len() {
            assert!(matches!(progress, Ok(PhyDbProgress::Complete)));
        } else {
            assert!(matches!(progress, Ok(PhyDbProgress::Pending)));
        }
        if let Some(id) = remainder {
            assert!(
                !fixture
                    .domain
                    .records
                    .lock()
                    .unwrap()
                    .live
                    .contains_key(&id)
            );
        }
        paging_remains_pinned(&fixture, &pages);
    }
    assert_eq!(fixture.device.lock().unwrap().csr(COMMAND_DOORBELL), 8);
    assert_eq!(regular.start_phy_db_replay(), Err(Error::State));
    regular.stop().unwrap();
    assert!(regular.finish().is_ok());
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}

#[test]
fn posted_long_replay_error_preserves_tb1_and_paging_until_confirmed_stop() {
    let _clock = test_clock::virtualize();
    let bytes = calibration::calibration_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let configuration = [0x51; 16];
    let mut regular = start_regular_with_db(&mut fixture, &firmware, &configuration);
    let mut index = 0;
    paging::accept_alive(&fixture, &mut regular, &mut index);
    let pages = complete_paging(&fixture, &mut regular, &mut index);
    regular.start_phy_db_replay().unwrap();
    fixture.device.lock().unwrap().command.fail_next_doorbell();
    let unmaps = fixture.domain.records.lock().unwrap().unmaps;
    acknowledge(&fixture, &mut index, TX_ANTENNA);
    assert!(matches!(
        regular.poll_phy_db_replay(),
        Err(Error::Command(_))
    ));
    let command = nvm::submitted(&fixture);
    assert_eq!(command.bytes()[0], PHY_DB_COMMAND);
    assert_eq!(&command.bytes()[12..], configuration);
    let remainder = dma_offset(command.remainder().unwrap()).0;
    {
        let records = fixture.domain.records.lock().unwrap();
        assert_eq!(records.unmaps, unmaps);
        assert!(records.live.contains_key(&remainder));
    }
    paging_remains_pinned(&fixture, &pages);
    assert!(regular.poll_phy_db_replay().is_err());
    assert_eq!(regular.start_phy_db_replay(), Err(Error::State));
    assert!(regular.finish().is_err());
    paging::stop_with_retries(&fixture, &mut regular);
}

#[test]
fn missing_calibration_families_refuse_replay_before_antenna_or_dma() {
    let _clock = test_clock::virtualize();
    let bytes = calibration::calibration_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut regular = paging::start_regular(&mut fixture, &firmware);
    let mut index = 0;
    paging::accept_alive(&fixture, &mut regular, &mut index);
    let pages = complete_paging(&fixture, &mut regular, &mut index);
    let (maps, syncs) = {
        let records = fixture.domain.records.lock().unwrap();
        (records.maps, records.syncs)
    };
    let events = fixture.device.lock().unwrap().events.len();
    assert!(regular.start_phy_db_replay().is_err());
    {
        let records = fixture.domain.records.lock().unwrap();
        assert_eq!((records.maps, records.syncs), (maps, syncs));
    }
    {
        let device = fixture.device.lock().unwrap();
        assert_eq!(device.csr(COMMAND_DOORBELL), 1);
        assert!(
            device.events[events..]
                .iter()
                .all(|event| matches!(event, Event::Read(..)))
        );
    }
    paging_remains_pinned(&fixture, &pages);
    regular.stop().unwrap();
    assert!(regular.finish().is_ok());
}
