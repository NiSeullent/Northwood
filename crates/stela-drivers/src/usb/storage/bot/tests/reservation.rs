// SPDX-License-Identifier: GPL-2.0-or-later

use super::{NativeFixture, native_fixture, scsi};
use crate::{
    block::BlockDevice,
    usb::{
        storage::{
            Data, ReservationError, UsbDisk, UsbDiskReservation, UsbStorage, media::TransportOwner,
        },
        xhci::{SUPER_SPEED, model::ResetHardware},
    },
};
use alloc::{vec, vec::Vec};
use core::cell::RefCell;

fn activity(hardware: &RefCell<ResetHardware>) -> [usize; 4] {
    let hardware = hardware.borrow();
    [
        hardware.scsi.len(),
        hardware.commands_seen.len(),
        hardware.reset_writes,
        hardware.accesses,
    ]
}

fn reserve(storage: UsbStorage, disk: UsbDisk) -> UsbDiskReservation {
    match storage.try_reserve_disk(disk) {
        Ok(reservation) => reservation,
        Err((_, _, error)) => panic!("reservation unexpectedly refused: {error:?}"),
    }
}

fn refused(
    storage: UsbStorage,
    disk: UsbDisk,
    expected: ReservationError,
) -> (UsbStorage, UsbDisk) {
    match storage.try_reserve_disk(disk) {
        Ok(_) => panic!("reservation should have refused: {expected:?}"),
        Err((storage, disk, error)) => {
            assert_eq!(error, expected);
            (storage, disk)
        }
    }
}

#[test]
fn idle_fixed_lun_reservation_preserves_identity_without_io() {
    let mut memory = vec![0u64; 512 * 1024];
    let NativeFixture {
        storage,
        disk,
        hardware,
        dma,
    } = native_fixture(&mut memory, SUPER_SPEED, 64, true);
    let expected = disk.identity().clone();
    assert_eq!(expected.block_size, 512);
    assert_eq!(expected.blocks, 1024);
    assert!(!expected.read_only);
    assert!(expected.features.write_cache);
    let before = activity(&hardware);
    let remaining = dma.remaining();
    let reservation = reserve(storage, disk);
    assert_eq!(reservation.identity(), &expected);
    assert_eq!(activity(&hardware), before);
    assert_eq!(dma.remaining(), remaining);
}

#[test]
fn owner_and_rescan_disk_aliases_refuse_without_consuming_the_pair() {
    let mut memory = vec![0u64; 512 * 1024];
    let NativeFixture {
        storage,
        disk,
        hardware,
        ..
    } = native_fixture(&mut memory, SUPER_SPEED, 64, true);
    let owner_alias = storage.clone();
    let before = activity(&hardware);
    let (mut storage, disk) = refused(storage, disk, ReservationError::Aliased);
    assert_eq!(activity(&hardware), before);
    assert_eq!(owner_alias.location(), storage.location());
    drop(owner_alias);

    let mut failures = Vec::new();
    let mut aliases = storage.rescan(&mut failures).unwrap();
    assert!(failures.is_empty());
    assert_eq!(aliases.len(), 1);
    let mut disk_alias = aliases.pop().unwrap();
    assert_eq!(disk_alias.identity(), disk.identity());
    let before = activity(&hardware);
    let (storage, mut disk) = refused(storage, disk, ReservationError::Aliased);
    assert_eq!(activity(&hardware), before);

    let mut bytes = [0; 512];
    disk.read(7, &mut bytes).unwrap();
    assert_eq!(bytes, [0x6d; 512]);
    bytes.fill(0);
    disk_alias.read(7, &mut bytes).unwrap();
    assert_eq!(bytes, [0x6d; 512]);
    drop(disk_alias);

    let before = activity(&hardware);
    let expected = disk.identity().clone();
    assert_eq!(reserve(storage, disk).identity(), &expected);
    assert_eq!(activity(&hardware), before);
}

#[test]
fn identical_reported_identities_do_not_allow_pairing_different_owners() {
    let mut left_memory = vec![0u64; 512 * 1024];
    let mut right_memory = vec![0u64; 512 * 1024];
    let left = native_fixture(&mut left_memory, SUPER_SPEED, 64, true);
    let right = native_fixture(&mut right_memory, SUPER_SPEED, 64, true);
    let expected = left.disk.identity().clone();
    assert_eq!(right.disk.identity(), &expected);
    let left_before = activity(&left.hardware);
    let right_before = activity(&right.hardware);

    let (left_storage, right_disk) =
        refused(left.storage, right.disk, ReservationError::DifferentOwner);
    assert_eq!(activity(&left.hardware), left_before);
    assert_eq!(activity(&right.hardware), right_before);

    let left_reservation = reserve(left_storage, left.disk);
    let right_reservation = reserve(right.storage, right_disk);
    assert_eq!(left_reservation.identity(), &expected);
    assert_eq!(right_reservation.identity(), &expected);
    assert_eq!(activity(&left.hardware), left_before);
    assert_eq!(activity(&right.hardware), right_before);
}

#[test]
fn in_flight_bot_read_refuses_reservation_and_can_still_finish_once() {
    let mut memory = vec![0u64; 512 * 1024];
    let NativeFixture {
        storage,
        disk,
        hardware,
        ..
    } = native_fixture(&mut memory, SUPER_SPEED, 64, true);
    let scsi_start = hardware.borrow().scsi.len();
    let TransportOwner::Bot(owner) = &storage.owner else {
        panic!("native BOT fixture published another transport");
    };
    let read = scsi::read_write(false, false, 7, 1);
    owner
        .borrow_mut()
        .transport
        .begin(0, read.as_ref(), Data::In(512))
        .unwrap();
    let before = activity(&hardware);

    let (storage, disk) = refused(storage, disk, ReservationError::Busy);
    assert_eq!(activity(&hardware), before);
    let TransportOwner::Bot(owner) = &storage.owner else {
        panic!("refusal changed the transport");
    };
    let result = (0..32)
        .find_map(|_| owner.borrow_mut().transport.end())
        .expect("native read completion");
    assert_eq!(result, Ok(vec![0x6d; 512]));
    assert!(owner.borrow_mut().transport.end().is_none());
    assert_eq!(&hardware.borrow().scsi[scsi_start..], &[(0, scsi::READ_10)]);

    let before = activity(&hardware);
    let expected = disk.identity().clone();
    assert_eq!(reserve(storage, disk).identity(), &expected);
    assert_eq!(activity(&hardware), before);
}

#[test]
fn stale_publication_is_refused_and_fresh_rescan_can_reserve() {
    let mut memory = vec![0u64; 512 * 1024];
    let NativeFixture {
        storage,
        disk,
        hardware,
        ..
    } = native_fixture(&mut memory, SUPER_SPEED, 64, true);
    let TransportOwner::Bot(owner) = &storage.owner else {
        panic!("native BOT fixture published another transport");
    };
    owner.borrow_mut().invalidate();
    let before = activity(&hardware);
    let (mut storage, mut disk) = refused(storage, disk, ReservationError::Stale);
    let mut output = [0x5a; 512];
    assert!(disk.read(7, &mut output).is_err());
    assert_eq!(output, [0x5a; 512]);
    assert_eq!(activity(&hardware), before);
    drop(disk);

    let mut failures = Vec::new();
    let mut disks = storage.rescan(&mut failures).unwrap();
    assert!(failures.is_empty());
    assert_eq!(disks.len(), 1);
    let disk = disks.pop().unwrap();
    let expected = disk.identity().clone();
    let before = activity(&hardware);
    assert_eq!(reserve(storage, disk).identity(), &expected);
    assert_eq!(activity(&hardware), before);
}

#[test]
fn quarantined_controller_slot_refuses_an_idle_class_owner() {
    let mut memory = vec![0u64; 512 * 1024];
    let NativeFixture {
        storage,
        disk,
        hardware,
        ..
    } = native_fixture(&mut memory, SUPER_SPEED, 64, true);
    let expected = disk.identity().clone();
    let before = activity(&hardware);
    {
        let TransportOwner::Bot(owner) = &storage.owner else {
            panic!("native BOT fixture published another transport");
        };
        let owner = owner.borrow();
        let transport = &owner.transport;
        let native = &transport.io;
        let mut controller = native.controller.borrow_mut();
        controller.device(native.id).unwrap().quarantine();
        assert!(controller.validate_device_id(native.id).is_err());
        assert!(!transport.poisoned());
        assert!(!transport.command_pending());
    }
    assert_eq!(activity(&hardware), before);
    let (_, disk) = refused(storage, disk, ReservationError::Stale);
    assert_eq!(disk.identity(), &expected);
    assert_eq!(activity(&hardware), before);
}
