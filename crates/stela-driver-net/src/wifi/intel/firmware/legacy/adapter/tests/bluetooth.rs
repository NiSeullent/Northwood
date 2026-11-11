// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::wifi::intel::{
    firmware::legacy::adapter::{
        BluetoothCoexistenceProgress, PhyConfigurationProgress, RegularAdapter,
    },
    radio::BluetoothConfigError,
};

pub(super) const BT_CONFIG: u8 = 0x9b;
pub(super) const BT_VERSION: [u8; 4] = [BT_CONFIG, 1, 6, 0];
const BT_SEQUENCE: u16 = 9;

pub(super) fn acknowledge_phy(
    fixture: &Fixture<'_>,
    regular: &mut RegularAdapter<'_, '_, Io>,
    index: &mut u16,
) {
    phy_db::acknowledge(fixture, index, phy_configuration::PHY_CONFIGURATION);
    assert!(matches!(
        regular.poll_phy_configuration(),
        Ok(PhyConfigurationProgress::Complete)
    ));
    assert!(regular.phy_ready());
}

fn assert_command(fixture: &Fixture<'_>, modules: u8) {
    let command = nvm::submitted(fixture);
    assert_eq!(
        command.bytes(),
        [
            BT_CONFIG,
            1,
            BT_SEQUENCE as u8,
            0,
            8,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            modules,
            0,
            0,
            0
        ]
    );
    assert_eq!(command.remainder(), None);
    assert!(
        fixture
            .device
            .lock()
            .unwrap()
            .events
            .contains(&Event::Write(COMMAND_DOORBELL, u32::from(BT_SEQUENCE) + 1))
    );
}

fn pending(regular: &mut RegularAdapter<'_, '_, Io>) {
    assert!(matches!(
        regular.poll_bluetooth_coexistence(),
        Ok(BluetoothCoexistenceProgress::Pending)
    ));
    assert!(regular.phy_ready());
    assert!(!regular.bluetooth_coexistence_ready());
}

pub(super) fn acknowledge(
    fixture: &Fixture<'_>,
    regular: &mut RegularAdapter<'_, '_, Io>,
    index: &mut u16,
) {
    phy_db::acknowledge(fixture, index, BT_CONFIG);
    assert!(matches!(
        regular.poll_bluetooth_coexistence(),
        Ok(BluetoothCoexistenceProgress::Complete)
    ));
    assert!(regular.phy_ready());
    assert!(regular.bluetooth_coexistence_ready());
}

#[test]
#[ignore = "oracle: STELA_IWLWIFI_IMAGES"]
fn pinned_31dc_bluetooth_uses_normal_mode_and_requires_a_fresh_exact_reply() {
    let _clock = test_clock::virtualize();
    let bytes = pinned_cnvi9000_api46_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut index = 0;
    let (mut regular, pages) =
        phy_configuration::complete_replay(&mut fixture, &firmware, &mut index);
    assert_eq!(regular.start_bluetooth_coexistence(), Err(Error::State));
    regular.start_phy_configuration().unwrap();
    assert_eq!(regular.start_bluetooth_coexistence(), Err(Error::State));
    assert!(!regular.bluetooth_coexistence_ready());
    acknowledge_phy(&fixture, &mut regular, &mut index);
    let dma = fixture.dma_counts();

    paging::post(&fixture, &mut index, &[(BT_CONFIG, 1, BT_SEQUENCE, &[])]);
    regular.start_bluetooth_coexistence().unwrap();
    assert_command(&fixture, 0x15);
    assert_eq!(
        fixture.device.lock().unwrap().csr(COMMAND_DOORBELL),
        u32::from(BT_SEQUENCE) + 1
    );
    pending(&mut regular);
    assert_eq!(regular.start_bluetooth_coexistence(), Err(Error::State));
    paging::post(
        &fixture,
        &mut index,
        &[
            (BT_CONFIG, 0, BT_SEQUENCE, &[]),
            (BT_CONFIG, 1, BT_SEQUENCE | 0x2000, &[]),
            (BT_CONFIG, 1, BT_SEQUENCE | 0x8000, &[]),
            (phy_configuration::PHY_CONFIGURATION, 1, BT_SEQUENCE, &[]),
        ],
    );
    pending(&mut regular);
    assert_command(&fixture, 0x15);
    acknowledge(&fixture, &mut regular, &mut index);
    let events = fixture.device.lock().unwrap().events.len();
    assert_eq!(regular.start_bluetooth_coexistence(), Err(Error::State));
    assert!(regular.poll_bluetooth_coexistence().is_err());
    assert_eq!(fixture.device.lock().unwrap().events.len(), events);
    assert_eq!(fixture.dma_counts(), dma);
    phy_db::paging_remains_pinned(&fixture, &pages);
    regular.stop().unwrap();
    assert!(!regular.bluetooth_coexistence_ready());
    assert!(regular.finish().is_ok());
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}

#[test]
fn posted_bluetooth_error_retains_paging_and_command_pins_until_stop() {
    let _clock = test_clock::virtualize();
    let bytes = phy_configuration::synthetic_regular_image_with_commands(&BT_VERSION);
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut index = 0;
    let (mut regular, pages) =
        phy_configuration::complete_replay(&mut fixture, &firmware, &mut index);
    regular.start_phy_configuration().unwrap();
    acknowledge_phy(&fixture, &mut regular, &mut index);
    let dma = fixture.dma_counts();

    fixture.device.lock().unwrap().command.fail_next_doorbell();
    assert!(matches!(
        regular.start_bluetooth_coexistence(),
        Err(Error::Command(_))
    ));
    assert_command(&fixture, 0x14);
    assert!(!regular.bluetooth_coexistence_ready());
    assert!(regular.poll_bluetooth_coexistence().is_err());
    assert_eq!(regular.start_bluetooth_coexistence(), Err(Error::State));
    assert!(regular.finish().is_err());
    assert_eq!(fixture.dma_counts(), dma);
    phy_db::paging_remains_pinned(&fixture, &pages);
    paging::stop_with_retries(&fixture, &mut regular);
}

#[test]
fn invalid_bluetooth_metadata_refuses_posting_without_changing_phy_ready() {
    let _clock = test_clock::virtualize();
    for (versions, error) in [
        (&[][..], BluetoothConfigError::MissingApi),
        (
            &[BT_CONFIG, 1, 5, 0][..],
            BluetoothConfigError::Version {
                command: 5,
                notification: 0,
            },
        ),
        (
            &[BT_CONFIG, 1, 6, 1][..],
            BluetoothConfigError::Version {
                command: 6,
                notification: 1,
            },
        ),
        (
            &[BT_CONFIG, 1, 6, 0, BT_CONFIG, 1, 6, 0][..],
            BluetoothConfigError::DuplicateApi,
        ),
    ] {
        let bytes = phy_configuration::synthetic_regular_image_with_commands(versions);
        let firmware = Firmware::parse(&bytes).unwrap();
        let source = Source::default();
        let mut fixture = Fixture::new(&source);
        let mut index = 0;
        let (mut regular, pages) =
            phy_configuration::complete_replay(&mut fixture, &firmware, &mut index);
        regular.start_phy_configuration().unwrap();
        acknowledge_phy(&fixture, &mut regular, &mut index);
        let events = fixture.device.lock().unwrap().events.len();
        let dma = fixture.dma_counts();

        assert_eq!(
            regular.start_bluetooth_coexistence(),
            Err(Error::Bluetooth(error))
        );
        assert!(regular.phy_ready());
        assert!(!regular.bluetooth_coexistence_ready());
        assert_eq!(fixture.device.lock().unwrap().events.len(), events);
        assert_eq!(
            fixture.device.lock().unwrap().csr(COMMAND_DOORBELL),
            u32::from(BT_SEQUENCE)
        );
        assert_eq!(fixture.dma_counts(), dma);
        phy_db::paging_remains_pinned(&fixture, &pages);
        regular.stop().unwrap();
        assert!(regular.finish().is_ok());
    }
}
