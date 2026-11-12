// SPDX-License-Identifier: GPL-2.0-or-later
use super::{Model, *};
use crate::wifi::intel::firmware::{CMD_VERSIONS, Firmware, legacy::tests::image};
use stela_driver_base::{Clock, test_clock};

fn wait_init(queue: &mut Queue) -> Result<Sequence, Error> {
    let mut bytes = image();
    bytes.extend(CMD_VERSIONS.to_le_bytes());
    bytes.extend(4u32.to_le_bytes());
    bytes.extend([1, 0, 99, 3]);
    let firmware = Firmware::parse(&bytes).unwrap();
    let clock = Clock {
        ticks_per_microsecond: 1,
    };
    queue.wait_init(&firmware, clock)
}

fn frame(group: u8, command: u8, payload: &[u8]) -> [u8; BUFFER_BYTES] {
    let mut bytes = [0; BUFFER_BYTES];
    let end = 8 + payload.len();
    bytes[..4].copy_from_slice(&(payload.len() as u32 + 4).to_le_bytes());
    bytes[4] = command;
    bytes[5] = group;
    bytes[8..end].copy_from_slice(payload);
    let next = end.next_multiple_of(64);
    bytes[next..next + 4].copy_from_slice(&0x5555_0000u32.to_le_bytes());
    bytes
}

fn alive_frame() -> [u8; BUFFER_BYTES] {
    let mut payload = [0; 68];
    payload[..2].copy_from_slice(&0xcafeu16.to_le_bytes());
    frame(0, 1, &payload)
}

fn post(model: &mut Model, index: u16, bytes: &[u8]) {
    let vid = index + 1;
    model.write_buffer(vid, bytes);
    model.used(index, u32::from(vid));
    model.producer(u32::from(index + 1));
}

#[test]
fn command_cutoff_requires_current_generation_and_new_closed_slot() {
    let response = frame(1, 0x88, &[0; 8]);
    let mut old = Model::new(41);
    post(&mut old, 0, &response);
    let stale = old.queue.complete().unwrap().unwrap();
    // SAFETY: the old synchronous model can no longer post DMA.
    unsafe { old.queue.release_after_stop().unwrap() };

    let mut current = Model::new(42);
    post(&mut current, 0, &response);
    let first_cutoff = current.queue.response_cutoff().unwrap();
    let early = current.queue.complete().unwrap().unwrap();
    post(&mut current, 1, &response);
    let exact = current.queue.complete().unwrap().unwrap();
    assert!(!first_cutoff.accepts(&stale));
    assert!(!first_cutoff.accepts(&early));
    assert!(first_cutoff.accepts(&exact));
    let next_cutoff = current.queue.response_cutoff().unwrap();
    assert!(!next_cutoff.accepts(&exact));
    // SAFETY: the current synchronous model can no longer post DMA.
    unsafe { current.queue.release_after_stop().unwrap() };
}

#[test]
fn dma_completion_and_init_cutoff_gate_regular_boot() {
    let _time = test_clock::virtualize();
    let mut init = Model::new(41);
    let mut sequence = wait_init(&mut init.queue).unwrap();
    assert!(matches!(wait_init(&mut init.queue), Err(Error::State)));
    assert!(init.queue.complete().unwrap().is_none());
    assert_eq!(
        poll_notification(&mut sequence, None),
        Ok(Progress::Pending)
    );

    let mut packed = frame(0, 1, &[0xca]);
    packed[64..256].copy_from_slice(&alive_frame()[..192]);
    post(&mut init, 0, &packed);
    let completed = init.queue.complete().unwrap().unwrap();
    assert!(matches!(
        poll_notification(&mut sequence, Some(&completed)),
        Ok(Progress::Alive {
            image: alive::Image::Init,
            ..
        })
    ));
    assert!(sequence.wait_regular_after_init_stop(42).is_err());

    // This descriptor closes before the PHY calibration wait is armed. The
    // DMA owner observes its index even though the bytes are not polled yet.
    post(&mut init, 1, &frame(0, 4, &[]));
    init.queue.wait_init_complete(&mut sequence).unwrap();
    let early = init.queue.complete().unwrap().unwrap();
    assert_eq!(
        poll_notification(&mut sequence, Some(&early)),
        Ok(Progress::Pending)
    );
    assert!(sequence.wait_regular_after_init_stop(42).is_err());

    post(&mut init, 2, &frame(0, 0x6b, &[1, 2]));
    let phy_db = init.queue.complete().unwrap().unwrap();
    assert_eq!(
        poll_notification(&mut sequence, Some(&phy_db)),
        Ok(Progress::Pending)
    );
    assert_eq!(
        phy_db.packets().unwrap().next().unwrap().unwrap().payload(),
        &[1, 2]
    );

    post(&mut init, 3, &frame(1, 4, &[]));
    let completed = init.queue.complete().unwrap().unwrap();
    assert_eq!(
        poll_notification(&mut sequence, Some(&completed)),
        Ok(Progress::InitComplete)
    );
    assert!(poll_notification(&mut sequence, Some(&completed)).is_err());

    // Retain one valid ALIVE from INIT across stop; the original generation
    // stays attached to the CPU copy, even when its DMA pins are reclaimed.
    post(&mut init, 4, &alive_frame());
    let old_alive = init.queue.complete().unwrap().unwrap();
    // SAFETY: the synchronous fixture's DMA model has stopped publishing.
    unsafe { init.queue.release_after_stop().unwrap() };
    let mut regular = Model::new(42);
    regular
        .queue
        .wait_regular_after_init_stop(&mut sequence)
        .unwrap();
    assert_eq!(
        poll_notification(&mut sequence, Some(&old_alive)),
        Ok(Progress::Pending)
    );
    post(&mut regular, 0, &frame(0, 4, &[]));
    let init_complete = regular.queue.complete().unwrap().unwrap();
    assert_eq!(
        poll_notification(&mut sequence, Some(&init_complete)),
        Ok(Progress::Pending)
    );
    post(&mut regular, 1, &alive_frame());
    let completed = regular.queue.complete().unwrap().unwrap();
    assert!(matches!(
        poll_notification(&mut sequence, Some(&completed)),
        Ok(Progress::Alive {
            image: alive::Image::Regular,
            ..
        })
    ));
    // SAFETY: all modeled DMA activity is synchronous and has ceased.
    unsafe { regular.queue.release_after_stop().unwrap() };
}

#[test]
fn invalid_frames_and_old_generation_never_satisfy_init_complete() {
    let _time = test_clock::virtualize();
    let mut previous = Model::new(8);
    post(&mut previous, 0, &frame(0, 4, &[]));
    let stale = previous.queue.complete().unwrap().unwrap();
    // SAFETY: all modeled DMA activity is synchronous and has ceased.
    unsafe { previous.queue.release_after_stop().unwrap() };

    let mut init = Model::new(9);
    let mut sequence = wait_init(&mut init.queue).unwrap();
    post(&mut init, 0, &alive_frame());
    let completed = init.queue.complete().unwrap().unwrap();
    poll_notification(&mut sequence, Some(&completed)).unwrap();
    assert_eq!(
        previous.queue.wait_init_complete(&mut sequence),
        Err(Error::Generation)
    );
    init.queue.wait_init_complete(&mut sequence).unwrap();
    assert_eq!(
        poll_notification(&mut sequence, Some(&stale)),
        Ok(Progress::Pending)
    );

    let mut wrong_queue = frame(0, 4, &[]);
    wrong_queue[2] = 1;
    post(&mut init, 1, &wrong_queue);
    let completed = init.queue.complete().unwrap().unwrap();
    assert_eq!(
        poll_notification(&mut sequence, Some(&completed)),
        Err(Error::Wire(receive::Error::Queue {
            expected: 0,
            actual: 1
        }))
    );

    let mut truncated = frame(0, 4, &[]);
    // A valid prefix cannot transition the sequence when a later frame in
    // the same completed DMA snapshot claims bytes beyond its backing.
    truncated[64..68].copy_from_slice(&0x3fffu32.to_le_bytes());
    post(&mut init, 2, &truncated);
    let completed = init.queue.complete().unwrap().unwrap();
    assert_eq!(
        poll_notification(&mut sequence, Some(&completed)),
        Err(Error::Wire(receive::Error::Length {
            offset: 64,
            raw: 0x3fff
        }))
    );

    post(&mut init, 3, &frame(2, 4, &[]));
    let wrong_group = init.queue.complete().unwrap().unwrap();
    assert_eq!(
        poll_notification(&mut sequence, Some(&wrong_group)),
        Ok(Progress::Pending)
    );
    assert!(sequence.wait_regular_after_init_stop(10).is_err());
    post(&mut init, 4, &frame(0, 4, &[]));
    let completed = init.queue.complete().unwrap().unwrap();
    assert_eq!(
        poll_notification(&mut sequence, Some(&completed)),
        Ok(Progress::InitComplete)
    );
    // SAFETY: all modeled DMA activity is synchronous and has ceased.
    unsafe { init.queue.release_after_stop().unwrap() };
}
