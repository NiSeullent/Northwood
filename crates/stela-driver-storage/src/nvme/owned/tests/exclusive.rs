// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;
use crate::block::Request;

const SLOT_BYTES: usize = 16 * 1024;
const SLOT_LBA: u64 = 32;

fn namespace(fixture: &Fixture) -> Namespace {
    let mut attached = fixture.attach().unwrap();
    assert!(attached.failures.is_empty());
    assert_eq!(attached.namespaces.len(), 1);
    attached.namespaces.pop().unwrap()
}

fn exclusive(fixture: &Fixture) -> ExclusiveNamespace {
    namespace(fixture)
        .into_exclusive()
        .unwrap_or_else(|(_, error)| panic!("exclusive namespace refused: {error}"))
}

fn write(data: Vec<u8>) -> Request {
    Request::Write {
        lba: SLOT_LBA,
        data,
        durable: true,
    }
}

#[test]
fn exclusive_namespace_completes_synchronous_requests() {
    let fixture = Fixture::new();
    let mut disk = exclusive(&fixture);
    let data: Vec<u8> = (0..SLOT_BYTES).map(|byte| (byte * 17) as u8).collect();
    assert_eq!(write(data.clone()).run(&mut disk).unwrap(), None);
    let read = Request::Read {
        lba: SLOT_LBA,
        bytes: SLOT_BYTES,
    };
    assert_eq!(read.run(&mut disk).unwrap(), Some(data));
    let model = fixture.model.lock().unwrap();
    assert_eq!(model.sq_heads[1], 2);
    assert!(!model.outstanding);
    let write_command = model.read_iova(model.queues[1].0, 64);
    let flags = u32::from_le_bytes(write_command[48..52].try_into().unwrap());
    assert_ne!(flags & FORCE_UNIT_ACCESS, 0);
}

#[test]
fn exclusive_namespace_notice_refuses_before_publishing_io() {
    let fixture = Fixture::new();
    fixture.model.lock().unwrap().async_events = 1 << 8;
    let mut disk = exclusive(&fixture);
    {
        let mut model = fixture.model.lock().unwrap();
        model.blocks *= 2;
        model.post_event(2);
    }

    assert!(write(vec![0x5a; SLOT_BYTES]).run(&mut disk).is_err());
    assert!(disk.flush_cache().is_err());
    assert_eq!(disk.identity().blocks, 1024);
    let model = fixture.model.lock().unwrap();
    assert_eq!(model.sq_heads[1], 0);
    assert!(model.disk.iter().all(|&byte| byte == 0));
}

#[test]
fn exclusive_namespace_does_not_publish_a_read_completed_with_a_notice() {
    let fixture = Fixture::new();
    fixture.model.lock().unwrap().async_events = 1 << 8;
    let mut disk = exclusive(&fixture);
    {
        let mut model = fixture.model.lock().unwrap();
        model.disk.fill(0x37);
        model.notice_on_io = Some(2);
    }

    let mut output = [0xa5; SLOT_BYTES];
    assert!(disk.read(SLOT_LBA, &mut output).is_err());
    assert_eq!(output, [0xa5; SLOT_BYTES]);
    assert_eq!(fixture.model.lock().unwrap().sq_heads[1], 1);

    assert!(disk.flush_cache().is_err());
    let model = fixture.model.lock().unwrap();
    assert_eq!(model.sq_heads[1], 1);
    assert!(!model.outstanding);
}

#[test]
fn exclusive_namespace_does_not_reissue_a_retryable_completion() {
    let fixture = Fixture::new();
    let mut disk = exclusive(&fixture);
    let before = {
        let mut model = fixture.model.lock().unwrap();
        model.disk.fill(0x37);
        model.statuses.push_back(2);
        model.commands
    };

    let mut output = [0xa5; SLOT_BYTES];
    assert_eq!(
        disk.read(SLOT_LBA, &mut output),
        Err(BlockError::Device { status: 2 })
    );
    assert_eq!(output, [0xa5; SLOT_BYTES]);
    assert_eq!(fixture.model.lock().unwrap().commands, before + 1);

    disk.read(SLOT_LBA, &mut output).unwrap();
    assert_eq!(output, [0x37; SLOT_BYTES]);
    assert_eq!(fixture.model.lock().unwrap().commands, before + 2);
}

#[test]
fn exclusive_namespace_timeout_contains_dma_without_replaying_the_write() {
    let _clock = crate::test_clock::virtualize();
    let fixture = Fixture::new();
    let mut disk = exclusive(&fixture);
    let before = {
        let mut model = fixture.model.lock().unwrap();
        model.delay_io = 1;
        model.events.len()
    };

    assert!(write(vec![0x5a; SLOT_BYTES]).run(&mut disk).is_err());
    assert!(disk.flush_cache().is_err());
    drop(disk);

    let model = fixture.model.lock().unwrap();
    assert_eq!(model.delay_io, 0);
    assert!(model.disk.iter().all(|&byte| byte == 0));
    assert!(model.released && model.retired && model.maps.is_empty());
    assert_eq!(fixture.source.released.load(Ordering::SeqCst), BUFFERS);
    let events = &model.events[before..];
    let at = |event| events.iter().position(|&entry| entry == event).unwrap();
    assert!(at("command") < at("stop"));
    assert!(at("stop") < at("bme_off"));
    assert!(at("bme_off") < at("drain"));
    assert!(at("drain") < at("unmap"));
}

#[test]
fn exclusive_namespace_keeps_ambiguous_dma_through_failed_drain() {
    let _clock = crate::test_clock::virtualize();
    let fixture = Fixture::new();
    let mut disk = exclusive(&fixture);
    let syncs = {
        let mut model = fixture.model.lock().unwrap();
        model.disk.fill(0x37);
        model.delay_io = 1;
        model.fail_after_doorbell = true;
        model.fail_drain = true;
        model.sync_cpu
    };

    let mut output = [0xa5; SLOT_BYTES];
    assert!(disk.read(SLOT_LBA, &mut output).is_err());
    assert_eq!(output, [0xa5; SLOT_BYTES]);
    {
        let controller = disk.namespace.controller.borrow();
        assert_eq!(controller.held.iter().map(Vec::len).sum::<usize>(), 5);
        assert_eq!(controller.pool.free.len(), POOL_PAGES - 5);
    }
    {
        let mut model = fixture.model.lock().unwrap();
        assert_eq!(model.sync_cpu, syncs);
        assert_eq!(model.maps.len(), BUFFERS);
        let (queue, command) = model.pending.take().expect("device still holds the read");
        model.complete(queue, command);
        model.fail_read = false;
    }
    assert!(disk.flush_cache().is_err());
    assert_eq!(output, [0xa5; SLOT_BYTES]);
    drop(disk);

    let model = fixture.model.lock().unwrap();
    assert!(model.events.contains(&"drain"));
    assert!(model.stopped && !model.bme && !model.drained);
    assert_eq!(model.maps.len(), BUFFERS);
    assert_eq!(model.sync_cpu, syncs);
    assert_eq!(model.unmap_attempts, 0);
    assert!(!model.retired && !model.released);
    assert_eq!(
        (model.register_owner_drops, model.lifecycle_owner_drops),
        (0, 0)
    );
    assert_eq!(fixture.source.released.load(Ordering::SeqCst), 0);
}

#[test]
fn exclusive_namespace_refuses_existing_controller_work() {
    let _clock = crate::test_clock::virtualize();
    let fixture = Fixture::new();
    let mut disk = namespace(&fixture);
    fixture.model.lock().unwrap().hold_io = true;
    let tag = disk.start(Command::Read { lba: 0, bytes: 512 }).unwrap();
    let Err((mut disk, _)) = disk.into_exclusive() else {
        panic!("exclusive admission accepted an existing tag");
    };
    fixture.model.lock().unwrap().complete_held(0);
    assert_eq!(disk.finish(tag).unwrap().unwrap(), Some(vec![0; 512]));
    assert!(disk.into_exclusive().is_ok());
}

#[test]
fn exclusive_namespace_refuses_a_sibling_route_before_any_command() {
    let fixture = Fixture::new();
    let disk = namespace(&fixture);
    let sibling = Namespace {
        controller: disk.controller.clone(),
        namespace: disk.namespace,
        identity: disk.identity.clone(),
        ids: disk.ids.clone(),
        scanned: disk.scanned,
        removed: None,
    };
    let commands = fixture.model.lock().unwrap().commands;

    let Err((disk, _)) = disk.into_exclusive() else {
        panic!("exclusive admission accepted a controller with a sibling route");
    };
    assert_eq!(fixture.model.lock().unwrap().commands, commands);

    drop(sibling);
    assert!(disk.into_exclusive().is_ok());
}

#[test]
fn exclusive_namespace_reidentifies_the_namespace_without_a_notice() {
    let fixture = Fixture::new();
    let disk = namespace(&fixture);
    let commands = {
        let mut model = fixture.model.lock().unwrap();
        assert!(model.event_request.is_none());
        model.eui64 = [0x22; 8];
        model.commands
    };

    let Err((mut disk, error)) = disk.into_exclusive() else {
        panic!("exclusive admission accepted a different namespace identifier");
    };
    assert_eq!(error, NAMESPACE_REMOVED);
    {
        let model = fixture.model.lock().unwrap();
        assert!(model.commands > commands);
        assert_eq!(model.sq_heads[1], 0);
    }
    disk.retire_owned().unwrap();
}
