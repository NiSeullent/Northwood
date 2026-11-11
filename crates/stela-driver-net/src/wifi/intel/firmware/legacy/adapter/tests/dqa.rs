// SPDX-License-Identifier: GPL-2.0-or-later
//! Exercise DQA transport in isolation after the SoC configuration fixture.
use super::*;
use crate::wifi::intel::{
    firmware::{CAPABILITIES, legacy::adapter::RegularAdapter},
    radio::control::Control,
};

const DATA_PATH_GROUP: u8 = 5;
const DQA_ENABLE: u8 = 0;
const DQA_SEQUENCE: u16 = 11;
const DQA_CAPABILITY: u32 = 12;

fn complete_soc<'a, 'fw>(
    fixture: &mut Fixture<'a>,
    firmware: &Firmware<'fw>,
    index: &mut u16,
) -> (RegularAdapter<'a, 'fw, Io>, Vec<u64>) {
    let (mut regular, pages) = soc::start_bluetooth(fixture, firmware, index);
    bluetooth::acknowledge(fixture, &mut regular, index);
    regular.start_soc_configuration().unwrap();
    soc::acknowledge(fixture, &mut regular, index);
    (regular, pages)
}

fn assert_command(fixture: &Fixture<'_>) {
    let command = nvm::submitted(fixture);
    assert_eq!(
        command.bytes(),
        [
            DQA_ENABLE,
            DATA_PATH_GROUP,
            DQA_SEQUENCE as u8,
            0,
            4,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ]
    );
    assert_eq!(command.remainder(), None);
}

fn synthetic_dqa_image() -> Vec<u8> {
    let mut bytes = soc::synthetic_image(true, &[]);
    let mut word = [0; 8];
    word[4..].copy_from_slice(&(1u32 << DQA_CAPABILITY).to_le_bytes());
    calibration::append_tlv(&mut bytes, CAPABILITIES, &word);
    bytes
}

#[test]
#[ignore = "oracle: STELA_IWLWIFI_IMAGES"]
fn pinned_31dc_dqa_queue_zero_transport_retires_only_the_fresh_exact_reply() {
    let _clock = test_clock::virtualize();
    let bytes = pinned_cnvi9000_api46_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    assert!(firmware.supports_capability(DQA_CAPABILITY));
    assert!(
        firmware
            .command_version(DATA_PATH_GROUP, DQA_ENABLE)
            .is_none()
    );
    let request = Control::dqa(&firmware).unwrap().unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut index = 0;
    let (mut regular, pages) = complete_soc(&mut fixture, &firmware, &mut index);
    let dma = fixture.dma_counts();
    let first_event = fixture.device.lock().unwrap().events.len();

    paging::post(
        &fixture,
        &mut index,
        &[(DQA_ENABLE, DATA_PATH_GROUP, DQA_SEQUENCE, &[])],
    );
    let pending = regular.submit_dqa_for_test(&request).unwrap();
    assert_command(&fixture);
    assert!(!regular.poll_dqa_for_test(pending).unwrap());
    paging::post(
        &fixture,
        &mut index,
        &[
            (DQA_ENABLE, 1, DQA_SEQUENCE, &[]),
            (DQA_ENABLE + 1, DATA_PATH_GROUP, DQA_SEQUENCE, &[]),
            (DQA_ENABLE, DATA_PATH_GROUP, DQA_SEQUENCE - 1, &[]),
            (DQA_ENABLE, DATA_PATH_GROUP, DQA_SEQUENCE | 0x2000, &[]),
            (DQA_ENABLE, DATA_PATH_GROUP, DQA_SEQUENCE | 0x8000, &[]),
        ],
    );
    assert!(!regular.poll_dqa_for_test(pending).unwrap());
    assert_command(&fixture);
    phy_db::acknowledge_in_group(&fixture, &mut index, DQA_ENABLE, DATA_PATH_GROUP);
    assert!(regular.poll_dqa_for_test(pending).unwrap());
    assert!(matches!(
        regular.submit_dqa_for_test(&request),
        Err(Error::State)
    ));
    assert!(regular.finish().is_err());
    assert_eq!(fixture.dma_counts(), dma);
    phy_db::paging_remains_pinned(&fixture, &pages);
    {
        let device = fixture.device.lock().unwrap();
        let events = &device.events[first_event..];
        let doorbells: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::Write(COMMAND_DOORBELL, pointer) => Some(*pointer),
                _ => None,
            })
            .collect();
        assert_eq!(doorbells, [u32::from(DQA_SEQUENCE) + 1]);
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Event::PeripheralWrite(..) | Event::SramWrite(..)))
        );
    }
    regular.stop().unwrap();
    assert!(regular.finish().is_ok());
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}

#[test]
fn posted_dqa_transport_error_retains_command_and_paging_pins_until_stop() {
    let _clock = test_clock::virtualize();
    let bytes = synthetic_dqa_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let request = Control::dqa(&firmware).unwrap().unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut index = 0;
    let (mut regular, pages) = complete_soc(&mut fixture, &firmware, &mut index);
    let dma = fixture.dma_counts();

    fixture.device.lock().unwrap().command.fail_next_doorbell();
    assert!(matches!(
        regular.submit_dqa_for_test(&request),
        Err(Error::Command(_))
    ));
    assert_command(&fixture);
    assert!(matches!(
        regular.submit_dqa_for_test(&request),
        Err(Error::State)
    ));
    assert!(regular.finish().is_err());
    assert_eq!(fixture.dma_counts(), dma);
    phy_db::paging_remains_pinned(&fixture, &pages);
    paging::stop_with_retries(&fixture, &mut regular);
}

#[test]
fn late_dqa_reply_preserves_command_ownership_until_stop() {
    let _clock = test_clock::virtualize();
    let bytes = synthetic_dqa_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let request = Control::dqa(&firmware).unwrap().unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut index = 0;
    let (mut regular, pages) = complete_soc(&mut fixture, &firmware, &mut index);

    let pending = regular.submit_dqa_for_test(&request).unwrap();
    let cpu_syncs = fixture.domain.records.lock().unwrap().cpu_syncs;
    let dma = fixture.dma_counts();
    test_clock::advance(
        Clock {
            ticks_per_microsecond: 1,
        },
        super::super::COMMAND_TIMEOUT_US + 1,
    );
    paging::post(
        &fixture,
        &mut index,
        &[(DQA_ENABLE, DATA_PATH_GROUP, DQA_SEQUENCE, &[])],
    );
    assert!(matches!(
        regular.poll_dqa_for_test(pending),
        Err(Error::CommandTimeout)
    ));
    assert_eq!(fixture.domain.records.lock().unwrap().cpu_syncs, cpu_syncs);
    assert_eq!(fixture.dma_counts(), dma);
    assert_command(&fixture);
    assert!(regular.finish().is_err());
    phy_db::paging_remains_pinned(&fixture, &pages);
    paging::stop_with_retries(&fixture, &mut regular);
}
