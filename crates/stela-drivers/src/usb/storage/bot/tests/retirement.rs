// SPDX-License-Identifier: GPL-2.0-or-later

use super::{NativeFixture, native_fixture};
use crate::{
    block::{BlockDevice, BlockError},
    usb::{
        descriptor::{DEVICE_DESCRIPTOR, GET_DESCRIPTOR, setup},
        storage::{RetiredUsbDisk, UsbDiskReservation, media::TransportOwner},
        xhci::{
            SET_DEQUEUE, STOP_ENDPOINT, SUPER_SPEED, Shared, SlotId,
            model::{ResetFault, ResetHardware},
        },
    },
};
use alloc::{rc::Rc, vec};
use core::cell::RefCell;

// Disable Slot is private to the xHCI reset module. The model records the
// command TRB type on the wire, where xHCI assigns Disable Slot type 10.
const DISABLE_SLOT: u32 = 10;

fn reserve(fixture: NativeFixture) -> (UsbDiskReservation, Shared, SlotId) {
    let NativeFixture { storage, disk, .. } = fixture;
    let (controller, id) = {
        let TransportOwner::Bot(owner) = &storage.owner else {
            panic!("fixture published another transport");
        };
        let owner = owner.borrow();
        let native = &owner.transport.io;
        (native.controller.clone(), native.id)
    };
    controller.borrow_mut().test_claim_native(id).unwrap();
    let reservation = match storage.try_reserve_disk(disk) {
        Ok(reservation) => reservation,
        Err((_, _, error)) => panic!("class reservation failed: {error:?}"),
    };
    (reservation, controller, id)
}

fn retire(reservation: UsbDiskReservation) -> RetiredUsbDisk {
    match reservation.retire_xhci() {
        Ok(retired) => retired,
        Err((_, error)) => panic!("terminal retirement failed: {error:?}"),
    }
}

fn lifecycle(hardware: &RefCell<ResetHardware>) -> [usize; 4] {
    let hardware = hardware.borrow();
    [
        hardware.scsi.len(),
        hardware.reset_writes,
        hardware.resets,
        hardware.enables.len(),
    ]
}

fn assert_old_id_fenced(controller: &Shared, id: SlotId, hardware: &RefCell<ResetHardware>) {
    let accesses = hardware.borrow().accesses;
    assert!(controller.borrow_mut().device(id).is_err());
    assert_eq!(hardware.borrow().accesses, accesses);
}

#[test]
fn retirement_stops_every_endpoint_before_disabling_the_original_slot() {
    let mut memory = vec![0u64; 512 * 1024];
    let fixture = native_fixture(&mut memory, SUPER_SPEED, 64, true);
    let expected = fixture.disk.identity().clone();
    let hardware = fixture.hardware.clone();
    let (reservation, controller, id) = reserve(fixture);
    let slot = id.hardware_slot();
    assert_ne!(controller.borrow().test_slot_dma_entry(id), 0);
    let before = lifecycle(&hardware);
    let first = hardware.borrow().commands_seen.len();

    let retired = retire(reservation);
    assert_eq!(retired.identity(), &expected);
    assert_eq!(
        &hardware.borrow().commands_seen[first..],
        &[
            (STOP_ENDPOINT, slot, 1),
            (SET_DEQUEUE, slot, 1),
            (STOP_ENDPOINT, slot, 2),
            (SET_DEQUEUE, slot, 2),
            (STOP_ENDPOINT, slot, 3),
            (SET_DEQUEUE, slot, 3),
            (DISABLE_SLOT, slot, 0),
        ]
    );
    assert_eq!(lifecycle(&hardware), before);
    assert_old_id_fenced(&controller, id, &hardware);
    assert_eq!(controller.borrow().test_slot_dma_entry(id), 0);
}

#[test]
fn unproved_hardware_retirement_pins_the_reservation_and_fences_the_slot() {
    let endpoint = "retire every endpoint before Disable Slot";
    for (fault, command, dci, operation) in [
        (ResetFault::Stop(2), STOP_ENDPOINT, 2, endpoint),
        (ResetFault::Dequeue(2), SET_DEQUEUE, 2, endpoint),
        (
            ResetFault::Disable,
            DISABLE_SLOT,
            0,
            "acknowledge Disable Slot",
        ),
        (
            ResetFault::WrongDisableSlot,
            DISABLE_SLOT,
            0,
            "acknowledge Disable Slot identity",
        ),
        (
            ResetFault::SilentDisable,
            DISABLE_SLOT,
            0,
            "acknowledge Disable Slot",
        ),
    ] {
        let mut memory = vec![0u64; 512 * 1024];
        let fixture = native_fixture(&mut memory, SUPER_SPEED, 64, true);
        let expected = fixture.disk.identity().clone();
        let hardware = fixture.hardware.clone();
        let (reservation, controller, id) = reserve(fixture);
        let before = lifecycle(&hardware);
        let first = hardware.borrow().commands_seen.len();
        let dma = controller.borrow().test_slot_dma_entry(id);
        assert_ne!(dma, 0);
        hardware.borrow_mut().fault = fault;

        let Err((reservation, error)) = reservation.retire_xhci() else {
            panic!("retirement succeeded after {fault:?}");
        };
        assert!(error.ownership_uncertain);
        assert_eq!(error.operation, operation);
        assert_eq!(reservation.identity(), &expected);
        {
            let hardware = hardware.borrow();
            let commands = &hardware.commands_seen[first..];
            assert!(commands.contains(&(command, id.hardware_slot(), dci)));
            if command != DISABLE_SLOT {
                assert!(commands.iter().all(|&(kind, _, _)| kind != DISABLE_SLOT));
            }
        }
        assert_eq!(lifecycle(&hardware), before);
        assert_old_id_fenced(&controller, id, &hardware);
        assert_eq!(controller.borrow().test_slot_dma_entry(id), dma);

        let accesses = hardware.borrow().accesses;
        let last = hardware.borrow().commands_seen.len();
        let Err((reservation, error)) = reservation.retire_xhci() else {
            panic!("a fenced reservation was allowed to retry retirement");
        };
        assert!(error.ownership_uncertain);
        assert_eq!(reservation.identity(), &expected);
        assert_eq!(hardware.borrow().accesses, accesses);
        assert_eq!(hardware.borrow().commands_seen.len(), last);
        let refs = Rc::strong_count(&controller);
        drop(reservation);
        assert_eq!(Rc::strong_count(&controller), refs);
    }
}

#[test]
fn unread_controller_transfer_refuses_before_any_retirement_effect() {
    let mut memory = vec![0u64; 512 * 1024];
    let fixture = native_fixture(&mut memory, SUPER_SPEED, 64, true);
    let expected = fixture.disk.identity().clone();
    let hardware = fixture.hardware.clone();
    let (reservation, controller, id) = reserve(fixture);
    let request = setup(
        0x80,
        GET_DESCRIPTOR,
        u16::from(DEVICE_DESCRIPTOR) << 8,
        0,
        18,
    );
    let mut operation = controller
        .borrow_mut()
        .device(id)
        .unwrap()
        .start_control(request, &[])
        .unwrap();
    let before = lifecycle(&hardware);
    let first = hardware.borrow().commands_seen.len();
    let accesses = hardware.borrow().accesses;

    let Err((reservation, error)) = reservation.retire_xhci() else {
        panic!("retirement accepted the pending control transfer");
    };
    assert!(!error.ownership_uncertain);
    assert_eq!(error.operation, "preflight reserved BOT slot retirement");
    assert_eq!(
        error.cause,
        BlockError::Unsupported("BOT slot or controller still has pending work")
    );
    assert_eq!(reservation.identity(), &expected);
    assert_eq!(lifecycle(&hardware), before);
    assert_eq!(hardware.borrow().commands_seen.len(), first);
    assert_eq!(hardware.borrow().accesses, accesses);

    let descriptor = {
        let mut controller = controller.borrow_mut();
        let mut device = controller.device(id).unwrap();
        (0..32)
            .find_map(|_| device.poll(&mut operation))
            .expect("model control completion")
            .unwrap()
    };
    assert_eq!(descriptor.len(), 18);
    assert_eq!(descriptor[1], DEVICE_DESCRIPTOR);
    assert_eq!(retire(reservation).identity(), &expected);
    assert_eq!(lifecycle(&hardware), before);
    assert_old_id_fenced(&controller, id, &hardware);
}
