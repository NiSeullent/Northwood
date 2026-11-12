// SPDX-License-Identifier: GPL-2.0-or-later
use super::{Firmware, image, tlv};
use crate::wifi::intel::{
    firmware::{CMD_VERSIONS, legacy::alive::*},
    receive,
};
use alloc::vec::Vec;
use stela_driver_base::{Clock, ClockError, test_clock};

const CLOCK: Clock = Clock {
    ticks_per_microsecond: 1,
};
const INIT_GENERATION: u64 = 12;
const REGULAR_GENERATION: u64 = 13;
const CALIBRATION_START: u16 = 9;

fn firmware_bytes(version: Option<u8>) -> Vec<u8> {
    let mut bytes = image();
    if let Some(version) = version {
        tlv(&mut bytes, CMD_VERSIONS, &[1, 0, 99, version]);
    }
    bytes
}

fn sequence() -> Sequence {
    let bytes = firmware_bytes(Some(3));
    let firmware = Firmware::parse(&bytes).unwrap();
    Sequence::wait_init(&firmware, CLOCK, INIT_GENERATION).unwrap()
}

fn payload(status: u16, subtype: u8) -> [u8; 68] {
    let mut bytes = core::array::from_fn(|index| index as u8);
    bytes[0..2].copy_from_slice(&status.to_le_bytes());
    bytes[2..4].copy_from_slice(&0x8201u16.to_le_bytes());
    bytes[12] = subtype;
    bytes
}

fn completion(generation: u64, payload: &[u8]) -> Completion<'_> {
    Completion {
        generation,
        index: 0,
        group: 0,
        command: 1,
        payload,
    }
}

fn init_complete(generation: u64, index: u16) -> Completion<'static> {
    Completion {
        generation,
        index,
        group: 0,
        command: 4,
        payload: &[],
    }
}

fn finish_init(sequence: &mut Sequence) {
    sequence
        .wait_init_complete(INIT_GENERATION, CALIBRATION_START)
        .unwrap();
    assert_eq!(
        sequence.poll(Some(init_complete(INIT_GENERATION, CALIBRATION_START))),
        Ok(Progress::InitComplete)
    );
}

fn expect_alive(progress: Progress, expected: Image) -> Report {
    let Progress::Alive { image, report } = progress else {
        panic!("expected ALIVE, got {progress:?}");
    };
    assert_eq!(image, expected);
    report
}

pub(super) fn check_pinned_alive_sequence(firmware: &Firmware<'_>) {
    let _time = test_clock::virtualize();
    let mut sequence = Sequence::wait_init(firmware, CLOCK, INIT_GENERATION).unwrap();
    let bytes = payload(0xcafe, 9);
    expect_alive(
        sequence
            .poll(Some(completion(INIT_GENERATION, &bytes)))
            .unwrap(),
        Image::Init,
    );
    finish_init(&mut sequence);
    sequence
        .wait_regular_after_init_stop(REGULAR_GENERATION)
        .unwrap();
    let bytes = payload(0xcafe, 0);
    expect_alive(
        sequence
            .poll(Some(completion(REGULAR_GENERATION, &bytes)))
            .unwrap(),
        Image::Regular,
    );
}

#[test]
fn init_and_regular_decode_v3_and_each_complete_once() {
    let _time = test_clock::virtualize();
    let mut sequence = sequence();
    assert_eq!(
        sequence.wait_regular_after_init_stop(REGULAR_GENERATION),
        Err(Error::State)
    );

    // Subtype is metadata. Only the active image and RX generation select the stage.
    let bytes = payload(0xcafe, 9);
    let report = expect_alive(
        sequence
            .poll(Some(completion(INIT_GENERATION, &bytes)))
            .unwrap(),
        Image::Init,
    );
    assert_eq!(report.flags, 0x8201);
    assert_eq!(report.lmac.major, 0x07060504);
    assert_eq!(report.lmac.minor, 0x0b0a0908);
    assert_eq!(report.lmac.subtype, 9);
    assert_eq!(report.lmac.kind, 13);
    assert_eq!(report.lmac.mac, 14);
    assert_eq!(report.lmac.options, 15);
    assert_eq!(report.lmac.timestamp, 0x13121110);
    assert_eq!(report.lmac.debug_addresses[0], 0x17161514);
    assert_eq!(report.lmac.debug_addresses[5], 0x2b2a2928);
    assert_eq!(report.lmac.debug_addresses[7], 0x33323130);
    assert_eq!(report.umac.major, 0x37363534);
    assert_eq!(report.umac.minor, 0x3b3a3938);
    assert_eq!(report.umac.debug_addresses, [0x3f3e3d3c, 0x43424140]);
    assert_eq!(
        sequence.poll(Some(completion(INIT_GENERATION, &bytes))),
        Err(Error::State)
    );
    finish_init(&mut sequence);

    for next in [INIT_GENERATION - 1, INIT_GENERATION] {
        assert_eq!(
            sequence.wait_regular_after_init_stop(next),
            Err(Error::Generation {
                previous: INIT_GENERATION,
                next
            })
        );
    }
    sequence
        .wait_regular_after_init_stop(REGULAR_GENERATION)
        .unwrap();
    assert_eq!(
        sequence.poll(Some(init_complete(REGULAR_GENERATION, CALIBRATION_START))),
        Ok(Progress::Pending)
    );
    for generation in [INIT_GENERATION, REGULAR_GENERATION + 1] {
        assert_eq!(
            sequence.poll(Some(completion(generation, &bytes))),
            Ok(Progress::Pending)
        );
    }
    assert_eq!(
        sequence.poll(Some(completion(INIT_GENERATION, &payload(0xdead, 9)))),
        Ok(Progress::Pending)
    );
    let mut notification = completion(REGULAR_GENERATION, &bytes);
    notification.group = 1; // Linux's default-group alias for a legacy wait.
    assert_eq!(
        expect_alive(sequence.poll(Some(notification)).unwrap(), Image::Regular),
        report
    );
    assert_eq!(
        sequence.poll(Some(completion(REGULAR_GENERATION, &bytes))),
        Err(Error::State)
    );
    assert_eq!(
        sequence.wait_regular_after_init_stop(REGULAR_GENERATION + 1),
        Err(Error::State)
    );
}

#[test]
fn malformed_or_unrelated_notifications_leave_the_original_wait_armed() {
    let _time = test_clock::virtualize();
    let mut sequence = sequence();
    let bytes = payload(0xcafe, 0);
    for actual in 0..bytes.len() {
        assert_eq!(
            sequence.poll(Some(completion(INIT_GENERATION, &bytes[..actual]))),
            Ok(Progress::Malformed { actual })
        );
    }
    for actual in [69, 144, 152] {
        assert_eq!(
            sequence.poll(Some(completion(INIT_GENERATION, &vec![0; actual]))),
            Ok(Progress::Malformed { actual })
        );
    }
    for (group, command) in [(0, 4), (2, 1)] {
        assert_eq!(
            sequence.poll(Some(Completion {
                group,
                command,
                ..completion(INIT_GENERATION, &[])
            })),
            Ok(Progress::Pending)
        );
    }
    assert_eq!(sequence.poll(None), Ok(Progress::Pending));
    expect_alive(
        sequence
            .poll(Some(completion(INIT_GENERATION, &bytes)))
            .unwrap(),
        Image::Init,
    );
}

#[test]
fn non_success_status_retires_each_image_wait() {
    let _time = test_clock::virtualize();
    let good = payload(0xcafe, 0);
    for (image, generation) in [
        (Image::Init, INIT_GENERATION),
        (Image::Regular, REGULAR_GENERATION),
    ] {
        for status in [0xdead, 0, 0xcaff] {
            let mut sequence = sequence();
            if image == Image::Regular {
                expect_alive(
                    sequence
                        .poll(Some(completion(INIT_GENERATION, &good)))
                        .unwrap(),
                    Image::Init,
                );
                finish_init(&mut sequence);
                sequence.wait_regular_after_init_stop(generation).unwrap();
            }
            let bad = payload(status, 0);
            assert_eq!(
                sequence.poll(Some(completion(generation, &bad))),
                Err(Error::Notification {
                    image,
                    error: receive::Error::AliveRejected {
                        status,
                        flags: 0x8201
                    },
                })
            );
            assert_eq!(
                sequence.poll(Some(completion(generation, &good))),
                Err(Error::State)
            );
            assert_eq!(
                sequence.wait_regular_after_init_stop(generation + 1),
                Err(Error::State)
            );
        }
    }
}

#[test]
fn truncated_then_silent_init_and_late_regular_expire_independently() {
    let _time = test_clock::virtualize();
    let mut init = sequence();
    let bytes = payload(0xcafe, 0);
    test_clock::advance(CLOCK, 1_999_700);
    assert_eq!(
        init.poll(Some(completion(INIT_GENERATION, &bytes[..67]))),
        Ok(Progress::Malformed { actual: 67 })
    );
    test_clock::advance(CLOCK, 100);
    assert_eq!(init.poll(None), Err(Error::Timeout(Image::Init)));
    assert_eq!(
        init.poll(Some(completion(INIT_GENERATION, &bytes))),
        Err(Error::State)
    );

    let mut regular = sequence();
    expect_alive(
        regular
            .poll(Some(completion(INIT_GENERATION, &bytes)))
            .unwrap(),
        Image::Init,
    );
    test_clock::advance(CLOCK, 2_000_000);
    finish_init(&mut regular);
    regular
        .wait_regular_after_init_stop(REGULAR_GENERATION)
        .unwrap();
    assert_eq!(regular.poll(None), Ok(Progress::Pending));
    test_clock::advance(CLOCK, 2_000_000);
    assert_eq!(
        regular.poll(Some(completion(REGULAR_GENERATION, &bytes))),
        Err(Error::Timeout(Image::Regular))
    );
    assert_eq!(regular.poll(None), Err(Error::State));
}

#[test]
fn init_complete_requires_armed_calibration_and_current_rx_at_or_after_cutoff() {
    let _time = test_clock::virtualize();
    let mut sequence = sequence();
    let current = || init_complete(INIT_GENERATION, CALIBRATION_START);
    assert_eq!(sequence.poll(Some(current())), Ok(Progress::Pending));
    assert_eq!(
        sequence.wait_init_complete(INIT_GENERATION, CALIBRATION_START),
        Err(Error::State)
    );
    let bytes = payload(0xcafe, 9);
    expect_alive(
        sequence
            .poll(Some(completion(INIT_GENERATION, &bytes)))
            .unwrap(),
        Image::Init,
    );
    assert_eq!(sequence.poll(Some(current())), Err(Error::State));
    assert_eq!(
        sequence.wait_regular_after_init_stop(REGULAR_GENERATION),
        Err(Error::State)
    );
    assert_eq!(
        sequence.wait_init_complete(REGULAR_GENERATION, CALIBRATION_START),
        Err(Error::Generation {
            previous: INIT_GENERATION,
            next: REGULAR_GENERATION,
        })
    );
    sequence
        .wait_init_complete(INIT_GENERATION, CALIBRATION_START)
        .unwrap();
    assert_eq!(
        sequence.wait_init_complete(INIT_GENERATION, 0),
        Err(Error::State)
    );
    for packet in [
        init_complete(INIT_GENERATION - 1, CALIBRATION_START),
        init_complete(INIT_GENERATION, CALIBRATION_START - 1),
        Completion {
            command: 0x6b, // CALIB_RES_NOTIF_PHY_DB cannot complete the wait.
            ..current()
        },
        Completion {
            group: 2,
            ..current()
        },
    ] {
        assert_eq!(sequence.poll(Some(packet)), Ok(Progress::Pending));
        assert_eq!(
            sequence.wait_regular_after_init_stop(REGULAR_GENERATION),
            Err(Error::State)
        );
    }
    assert_eq!(
        sequence.poll(Some(Completion {
            group: 1,         // Linux's default-group alias for a legacy wait.
            payload: &[0x5a], // The callback leaves this payload opaque.
            ..current()
        })),
        Ok(Progress::InitComplete)
    );
    assert_eq!(sequence.poll(Some(current())), Err(Error::State));
    sequence
        .wait_regular_after_init_stop(REGULAR_GENERATION)
        .unwrap();
}

#[test]
fn init_complete_has_an_independent_deadline_and_cannot_revive_expired_init() {
    let _time = test_clock::virtualize();
    let mut sequence = sequence();
    let bytes = payload(0xcafe, 9);
    expect_alive(
        sequence
            .poll(Some(completion(INIT_GENERATION, &bytes)))
            .unwrap(),
        Image::Init,
    );
    test_clock::advance(CLOCK, 2_000_000);
    sequence
        .wait_init_complete(INIT_GENERATION, CALIBRATION_START)
        .unwrap();
    test_clock::advance(CLOCK, 1_999_800);
    assert_eq!(sequence.poll(None), Ok(Progress::Pending));
    assert_eq!(
        sequence.poll(Some(init_complete(INIT_GENERATION, CALIBRATION_START))),
        Err(Error::InitCompleteTimeout)
    );
    assert_eq!(
        sequence.poll(Some(init_complete(INIT_GENERATION, CALIBRATION_START))),
        Err(Error::State)
    );
    assert_eq!(
        sequence.wait_regular_after_init_stop(REGULAR_GENERATION),
        Err(Error::State)
    );
}

#[test]
fn only_api46_and_its_declared_notification_layout_arm_the_model() {
    let _time = test_clock::virtualize();
    for version in [None, Some(2), Some(6), Some(8), Some(255)] {
        let bytes = firmware_bytes(version);
        let firmware = Firmware::parse(&bytes).unwrap();
        assert_eq!(
            Sequence::wait_init(&firmware, CLOCK, INIT_GENERATION).unwrap_err(),
            Error::Version(version)
        );
    }
    let mut bytes = firmware_bytes(Some(3));
    let firmware = Firmware::parse(&bytes).unwrap();
    let uncalibrated = Clock {
        ticks_per_microsecond: 0,
    };
    assert_eq!(
        Sequence::wait_init(&firmware, uncalibrated, INIT_GENERATION).unwrap_err(),
        Error::Clock(ClockError::Uncalibrated)
    );
    bytes[72..76].copy_from_slice(&77u32.to_le_bytes());
    let firmware = Firmware::parse(&bytes).unwrap();
    assert_eq!(
        Sequence::wait_init(&firmware, CLOCK, INIT_GENERATION).unwrap_err(),
        Error::FirmwareApi(77)
    );
}
