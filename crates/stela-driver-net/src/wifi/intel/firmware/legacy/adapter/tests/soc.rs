// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::wifi::intel::{
    firmware::{
        CAPABILITIES,
        legacy::adapter::{RegularAdapter, SocConfigurationProgress},
    },
    radio::SocConfigurationError,
};

const SYSTEM_GROUP: u8 = 2;
const SOC_CONFIG: u8 = 1;
const SOC_SEQUENCE: u16 = 10;
const SOC_CAPABILITY: u32 = 37;

pub(super) fn synthetic_image(soc_capable: bool, soc_versions: &[u8]) -> Vec<u8> {
    let mut versions = bluetooth::BT_VERSION.to_vec();
    versions.extend_from_slice(soc_versions);
    let mut bytes = phy_configuration::synthetic_regular_image_with_commands(&versions);
    if soc_capable {
        let mut word = [0; 8];
        word[..4].copy_from_slice(&(SOC_CAPABILITY / 32).to_le_bytes());
        word[4..].copy_from_slice(&(1u32 << (SOC_CAPABILITY % 32)).to_le_bytes());
        calibration::append_tlv(&mut bytes, CAPABILITIES, &word);
    }
    bytes
}

pub(super) fn start_bluetooth<'a, 'fw>(
    fixture: &mut Fixture<'a>,
    firmware: &Firmware<'fw>,
    index: &mut u16,
) -> (RegularAdapter<'a, 'fw, Io>, Vec<u64>) {
    let (mut regular, pages) = phy_configuration::complete_replay(fixture, firmware, index);
    regular.start_phy_configuration().unwrap();
    bluetooth::acknowledge_phy(fixture, &mut regular, index);
    assert_eq!(regular.start_soc_configuration(), Err(Error::State));
    regular.start_bluetooth_coexistence().unwrap();
    assert_eq!(regular.start_soc_configuration(), Err(Error::State));
    assert!(!regular.soc_configuration_ready());
    (regular, pages)
}

fn assert_command(fixture: &Fixture<'_>) {
    let command = nvm::submitted(fixture);
    assert_eq!(
        command.bytes(),
        [
            SOC_CONFIG,
            SYSTEM_GROUP,
            SOC_SEQUENCE as u8,
            0,
            8,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0x9e,
            2,
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
            .contains(&Event::Write(COMMAND_DOORBELL, u32::from(SOC_SEQUENCE) + 1))
    );
}

fn pending(regular: &mut RegularAdapter<'_, '_, Io>) {
    assert!(matches!(
        regular.poll_soc_configuration(),
        Ok(SocConfigurationProgress::Pending)
    ));
    assert!(regular.phy_ready());
    assert!(regular.bluetooth_coexistence_ready());
    assert!(!regular.soc_configuration_ready());
}

pub(super) fn acknowledge(
    fixture: &Fixture<'_>,
    regular: &mut RegularAdapter<'_, '_, Io>,
    index: &mut u16,
) {
    phy_db::acknowledge_in_group(fixture, index, SOC_CONFIG, SYSTEM_GROUP);
    assert!(matches!(
        regular.poll_soc_configuration(),
        Ok(SocConfigurationProgress::Complete)
    ));
    assert!(regular.phy_ready());
    assert!(regular.bluetooth_coexistence_ready());
    assert!(regular.soc_configuration_ready());
}

#[test]
#[ignore = "oracle: STELA_IWLWIFI_IMAGES"]
fn pinned_31dc_soc_configuration_uses_670_us_and_waits_for_a_fresh_system_reply() {
    let _clock = test_clock::virtualize();
    let bytes = pinned_cnvi9000_api46_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    assert!(firmware.supports_capability(SOC_CAPABILITY));
    assert!(firmware.command_version(SYSTEM_GROUP, SOC_CONFIG).is_none());
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut index = 0;
    let (mut regular, pages) = start_bluetooth(&mut fixture, &firmware, &mut index);
    bluetooth::acknowledge(&fixture, &mut regular, &mut index);
    let dma = fixture.dma_counts();

    paging::post(
        &fixture,
        &mut index,
        &[(SOC_CONFIG, SYSTEM_GROUP, SOC_SEQUENCE, &[])],
    );
    regular.start_soc_configuration().unwrap();
    assert_command(&fixture);
    assert_eq!(
        fixture.device.lock().unwrap().csr(COMMAND_DOORBELL),
        u32::from(SOC_SEQUENCE) + 1
    );
    pending(&mut regular);
    assert_eq!(regular.start_soc_configuration(), Err(Error::State));
    paging::post(
        &fixture,
        &mut index,
        &[
            (SOC_CONFIG, 1, SOC_SEQUENCE, &[]),
            (SOC_CONFIG, SYSTEM_GROUP, SOC_SEQUENCE | 0x2000, &[]),
            (SOC_CONFIG, SYSTEM_GROUP, SOC_SEQUENCE | 0x8000, &[]),
            (bluetooth::BT_CONFIG, SYSTEM_GROUP, SOC_SEQUENCE, &[]),
        ],
    );
    pending(&mut regular);
    assert_command(&fixture);
    acknowledge(&fixture, &mut regular, &mut index);
    let events = fixture.device.lock().unwrap().events.len();
    assert_eq!(regular.start_soc_configuration(), Err(Error::State));
    assert!(regular.poll_soc_configuration().is_err());
    assert_eq!(fixture.device.lock().unwrap().events.len(), events);
    assert_eq!(fixture.dma_counts(), dma);
    phy_db::paging_remains_pinned(&fixture, &pages);
    regular.stop().unwrap();
    assert!(!regular.soc_configuration_ready());
    assert!(regular.finish().is_ok());
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}

#[test]
fn absent_capability_or_invalid_soc_metadata_preserves_bluetooth_ready_without_posting() {
    let _clock = test_clock::virtualize();
    for (capable, versions, error) in [
        (false, &[][..], SocConfigurationError::MissingCapability),
        (
            true,
            &[SOC_CONFIG, SYSTEM_GROUP, 1, 1][..],
            SocConfigurationError::Version {
                command: 1,
                notification: 1,
            },
        ),
        (
            true,
            &[
                SOC_CONFIG,
                SYSTEM_GROUP,
                1,
                0,
                SOC_CONFIG,
                SYSTEM_GROUP,
                2,
                0,
            ][..],
            SocConfigurationError::DuplicateApi,
        ),
    ] {
        let bytes = synthetic_image(capable, versions);
        let firmware = Firmware::parse(&bytes).unwrap();
        assert_eq!(firmware.supports_capability(SOC_CAPABILITY), capable);
        let source = Source::default();
        let mut fixture = Fixture::new(&source);
        let mut index = 0;
        let (mut regular, pages) = start_bluetooth(&mut fixture, &firmware, &mut index);
        bluetooth::acknowledge(&fixture, &mut regular, &mut index);
        let events = fixture.device.lock().unwrap().events.len();
        let dma = fixture.dma_counts();

        assert_eq!(
            regular.start_soc_configuration(),
            Err(Error::SocConfiguration(error))
        );
        assert!(regular.phy_ready());
        assert!(regular.bluetooth_coexistence_ready());
        assert!(!regular.soc_configuration_ready());
        assert_eq!(fixture.device.lock().unwrap().events.len(), events);
        assert_eq!(
            fixture.device.lock().unwrap().csr(COMMAND_DOORBELL),
            u32::from(SOC_SEQUENCE)
        );
        assert_eq!(fixture.dma_counts(), dma);
        phy_db::paging_remains_pinned(&fixture, &pages);
        regular.stop().unwrap();
        assert!(regular.finish().is_ok());
    }
}

#[test]
fn posted_soc_error_retains_paging_and_command_pins_until_stop() {
    let _clock = test_clock::virtualize();
    let bytes = synthetic_image(true, &[SOC_CONFIG, SYSTEM_GROUP, 2, 0]);
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut index = 0;
    let (mut regular, pages) = start_bluetooth(&mut fixture, &firmware, &mut index);
    bluetooth::acknowledge(&fixture, &mut regular, &mut index);
    let dma = fixture.dma_counts();

    fixture.device.lock().unwrap().command.fail_next_doorbell();
    assert!(matches!(
        regular.start_soc_configuration(),
        Err(Error::Command(_))
    ));
    assert_command(&fixture);
    assert!(!regular.soc_configuration_ready());
    assert!(regular.poll_soc_configuration().is_err());
    assert_eq!(regular.start_soc_configuration(), Err(Error::State));
    assert!(regular.finish().is_err());
    assert_eq!(fixture.dma_counts(), dma);
    phy_db::paging_remains_pinned(&fixture, &pages);
    paging::stop_with_retries(&fixture, &mut regular);
}
