// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::wifi::intel::firmware::legacy::{
    adapter::{PhyConfigurationProgress, PhyDbProgress, RegularAdapter},
    phy,
};

pub(super) const PHY_CONFIGURATION: u8 = 0x6a;
const TX_ANTENNA: u8 = 0x98;
const PHY_DB_COMMAND: u8 = 0x6c;
const CONFIG_SEQUENCE: u16 = 8;
const EFFECTIVE_PHY: u32 = 0x8012_0018;
const REGULAR_FLOW: u32 = 0x0f06_15db;
const REGULAR_EVENT: u32 = 0x0f02_958b;
const SYNTHETIC_FLOW: u32 = 0x2468_ace0;
const SYNTHETIC_EVENT: u32 = 0x1357_9bdf;
const TEST_CLOCK: Clock = Clock {
    ticks_per_microsecond: 1,
};

fn synthetic_regular_image() -> Vec<u8> {
    synthetic_regular_image_with_commands(&[])
}

pub(super) fn synthetic_regular_image_with_commands(additional_versions: &[u8]) -> Vec<u8> {
    let mut bytes = calibration::calibration_image_with_commands(additional_versions);
    // Default calibration TLV 22, twelve bytes, REGULAR image zero.
    for word in [22u32, 12, 0, SYNTHETIC_FLOW, SYNTHETIC_EVENT] {
        bytes.extend(word.to_le_bytes());
    }
    bytes
}

pub(super) fn complete_replay<'a, 'fw>(
    fixture: &mut Fixture<'a>,
    firmware: &Firmware<'fw>,
    index: &mut u16,
) -> (RegularAdapter<'a, 'fw, Io>, Vec<u64>) {
    let mut regular = phy_db::start_regular_with_db(fixture, firmware, &[0x10, 0x11]);
    paging::accept_alive(fixture, &mut regular, index);
    assert_eq!(regular.start_phy_configuration(), Err(Error::State));
    let pages = phy_db::complete_paging(fixture, &mut regular, index);
    assert_eq!(regular.start_phy_configuration(), Err(Error::State));
    regular.start_phy_db_replay().unwrap();
    assert_eq!(regular.start_phy_configuration(), Err(Error::State));
    phy_db::acknowledge(fixture, index, TX_ANTENNA);
    assert!(matches!(
        regular.poll_phy_db_replay(),
        Ok(PhyDbProgress::Pending)
    ));
    for slot in 2..CONFIG_SEQUENCE {
        assert_eq!(nvm::submitted(fixture).sequence(), slot);
        assert!(!regular.phy_ready());
        assert_eq!(regular.start_phy_configuration(), Err(Error::State));
        phy_db::acknowledge(fixture, index, PHY_DB_COMMAND);
        let progress = regular.poll_phy_db_replay();
        if slot + 1 == CONFIG_SEQUENCE {
            assert!(matches!(progress, Ok(PhyDbProgress::Complete)));
        } else {
            assert!(matches!(progress, Ok(PhyDbProgress::Pending)));
        }
    }
    assert!(!regular.phy_ready());
    (regular, pages)
}

fn expected_configuration(flow: u32, event: u32) -> Vec<u8> {
    let mut command = vec![PHY_CONFIGURATION, 1, CONFIG_SEQUENCE as u8, 0, 12, 0, 0, 0];
    for word in [EFFECTIVE_PHY, flow, event] {
        command.extend(word.to_le_bytes());
    }
    command
}

fn pending(regular: &mut RegularAdapter<'_, '_, Io>) {
    assert!(matches!(
        regular.poll_phy_configuration(),
        Ok(PhyConfigurationProgress::Pending)
    ));
    assert!(!regular.phy_ready());
}

#[test]
#[ignore = "oracle: STELA_IWLWIFI_IMAGES"]
fn pinned_31dc_uses_regular_phy_defaults_and_requires_a_fresh_exact_reply() {
    let _clock = test_clock::virtualize();
    let bytes = pinned_cnvi9000_api46_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut index = 0;
    let (mut regular, pages) = complete_replay(&mut fixture, &firmware, &mut index);
    assert_eq!(pages.len(), 9);
    assert!(regular.poll_phy_configuration().is_err());
    let (maps, unmaps, live) = {
        let records = fixture.domain.records.lock().unwrap();
        (records.maps, records.unmaps, records.live.len())
    };

    paging::post(
        &fixture,
        &mut index,
        &[(PHY_CONFIGURATION, 1, CONFIG_SEQUENCE, &[])],
    );
    regular.start_phy_configuration().unwrap();
    let command = nvm::submitted(&fixture);
    assert_eq!(
        command.bytes(),
        expected_configuration(REGULAR_FLOW, REGULAR_EVENT)
    );
    assert_eq!(command.remainder(), None);
    pending(&mut regular);
    assert_eq!(regular.start_phy_configuration(), Err(Error::State));
    paging::post(
        &fixture,
        &mut index,
        &[
            (PHY_CONFIGURATION, 0, CONFIG_SEQUENCE, &[]),
            (PHY_CONFIGURATION, 1, CONFIG_SEQUENCE | 0x2000, &[]),
            (PHY_CONFIGURATION, 1, CONFIG_SEQUENCE | 0x8000, &[]),
            (PHY_DB_COMMAND, 1, CONFIG_SEQUENCE, &[]),
        ],
    );
    pending(&mut regular);
    assert_eq!(nvm::submitted(&fixture).sequence(), CONFIG_SEQUENCE);
    phy_db::acknowledge(&fixture, &mut index, PHY_CONFIGURATION);
    assert!(matches!(
        regular.poll_phy_configuration(),
        Ok(PhyConfigurationProgress::Complete)
    ));
    assert!(regular.phy_ready());
    let events = fixture.device.lock().unwrap().events.len();
    assert_eq!(regular.start_phy_configuration(), Err(Error::State));
    assert!(regular.poll_phy_configuration().is_err());
    assert_eq!(fixture.device.lock().unwrap().events.len(), events);
    assert_eq!(fixture.device.lock().unwrap().csr(COMMAND_DOORBELL), 9);
    {
        let records = fixture.domain.records.lock().unwrap();
        assert_eq!(
            (records.maps, records.unmaps, records.live.len()),
            (maps, unmaps, live)
        );
    }
    phy_db::paging_remains_pinned(&fixture, &pages);
    regular.stop().unwrap();
    assert!(!regular.phy_ready());
    assert!(regular.finish().is_ok());
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}

#[test]
fn posted_regular_phy_error_retains_paging_and_command_pins_until_stop() {
    let _clock = test_clock::virtualize();
    let bytes = synthetic_regular_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut index = 0;
    let (mut regular, pages) = complete_replay(&mut fixture, &firmware, &mut index);
    let (maps, unmaps, live) = {
        let records = fixture.domain.records.lock().unwrap();
        (records.maps, records.unmaps, records.live.len())
    };
    fixture.device.lock().unwrap().command.fail_next_doorbell();
    assert!(matches!(
        regular.start_phy_configuration(),
        Err(Error::Command(_))
    ));
    let command = nvm::submitted(&fixture);
    assert_eq!(
        command.bytes(),
        expected_configuration(SYNTHETIC_FLOW, SYNTHETIC_EVENT)
    );
    assert_eq!(command.remainder(), None);
    assert!(!regular.phy_ready());
    assert!(regular.poll_phy_configuration().is_err());
    assert_eq!(regular.start_phy_configuration(), Err(Error::State));
    assert!(regular.finish().is_err());
    {
        let records = fixture.domain.records.lock().unwrap();
        assert_eq!(
            (records.maps, records.unmaps, records.live.len()),
            (maps, unmaps, live)
        );
    }
    phy_db::paging_remains_pinned(&fixture, &pages);
    paging::stop_with_retries(&fixture, &mut regular);
}

#[test]
fn exact_late_regular_phy_reply_times_out_and_preserves_paging() {
    let _clock = test_clock::virtualize();
    let bytes = synthetic_regular_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut index = 0;
    let (mut regular, pages) = complete_replay(&mut fixture, &firmware, &mut index);
    regular.start_phy_configuration().unwrap();
    let sequence = nvm::submitted(&fixture).sequence();
    let cpu_syncs = fixture.domain.records.lock().unwrap().cpu_syncs;
    test_clock::advance(TEST_CLOCK, 2_000_001);
    paging::post(
        &fixture,
        &mut index,
        &[(PHY_CONFIGURATION, 1, sequence, &[])],
    );
    assert!(matches!(
        regular.poll_phy_configuration(),
        Err(Error::CommandTimeout)
    ));
    assert_eq!(fixture.domain.records.lock().unwrap().cpu_syncs, cpu_syncs);
    assert_eq!(nvm::submitted(&fixture).sequence(), sequence);
    assert!(!regular.phy_ready());
    assert_eq!(regular.start_phy_configuration(), Err(Error::State));
    assert!(regular.finish().is_err());
    phy_db::paging_remains_pinned(&fixture, &pages);
    paging::stop_with_retries(&fixture, &mut regular);
}

#[test]
fn missing_regular_defaults_never_use_the_init_phy_configuration() {
    let _clock = test_clock::virtualize();
    let bytes = calibration::calibration_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut index = 0;
    let (mut regular, pages) = complete_replay(&mut fixture, &firmware, &mut index);
    let events = fixture.device.lock().unwrap().events.len();
    let (maps, syncs) = {
        let records = fixture.domain.records.lock().unwrap();
        (records.maps, records.syncs)
    };
    assert_eq!(
        regular.start_phy_configuration(),
        Err(Error::Phy(phy::Error::MissingRegularCalibration))
    );
    assert!(!regular.phy_ready());
    assert_eq!(fixture.device.lock().unwrap().events.len(), events);
    assert_eq!(fixture.device.lock().unwrap().csr(COMMAND_DOORBELL), 8);
    {
        let records = fixture.domain.records.lock().unwrap();
        assert_eq!((records.maps, records.syncs), (maps, syncs));
    }
    phy_db::paging_remains_pinned(&fixture, &pages);
    regular.stop().unwrap();
    assert!(regular.finish().is_ok());
}
