// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;
use alloc::sync::Arc;
use core::{cell::UnsafeCell, ptr::NonNull};
use stela_compiler::context::memory::MemoryStatus;
use stela_engine::{
    machine::engine::CpuMappingsRevokedFrame,
    memory::{RegisteredSystemRam, ReservedSystemRamRange, system_ram},
};

const FRAME: u64 = 0x10000;
const BLOCK: usize = 512;

#[repr(align(16384))]
struct FrameBytes(UnsafeCell<[u8; PAGE_BYTES]>);

// SAFETY: the test keeps the bytes pinned and only accesses them through the
// exact registered RAM owner. It does not retain an alternate reference.
unsafe impl Sync for FrameBytes {}

fn ram_with(bytes: [u8; PAGE_BYTES]) -> Arc<RegisteredSystemRam> {
    let storage = Arc::new(FrameBytes(UnsafeCell::new(bytes)));
    let range = ReservedSystemRamRange {
        source_pa: FRAME,
        host_pa: 0x800000,
        host_base: NonNull::new(storage.0.get().cast::<u8>()).unwrap(),
        bytes: PAGE_BYTES,
    };
    // SAFETY: the single range covers exactly the initialized pinned array.
    let mut ram = unsafe { system_ram::fixture(&[range], storage) };
    ram.enable_descriptor_frame_leases().unwrap();
    Arc::new(ram)
}

fn slot_start(plan: &partition::Plan, slot: u64) -> usize {
    plan.partition.first_lba as usize * BLOCK
        + plan.header.layout().slot_offset(slot).unwrap() as usize
}

fn assert_nonresident(ram: &RegisteredSystemRam) {
    assert_eq!(ram.nonresident_frame(FRAME, PAGE_BYTES), Some(FRAME));
    let mut output = [0xa5; 16];
    // SAFETY: this output is private, and the fixture keeps the same backing.
    let result = unsafe { ram.read_physical(FRAME, &mut output) };
    assert_eq!(result, Err(MemoryStatus::Nonresident));
    assert_eq!(output, [0xa5; 16]);
}

fn assert_resident_bytes(ram: &RegisteredSystemRam, expected: &[u8; PAGE_BYTES]) {
    assert_eq!(ram.nonresident_frame(FRAME, PAGE_BYTES), None);
    let mut bytes = [0xa5; PAGE_BYTES];
    // SAFETY: this output is private, and the fixture keeps the same backing.
    unsafe { ram.read_physical(FRAME, &mut bytes) }.unwrap();
    assert_eq!(&bytes, expected);
}

fn assert_backing_bytes(stored: &StoredFrame<'_, '_, DiskFixture>, expected: &[u8; PAGE_BYTES]) {
    let mut bytes = [0xa5; PAGE_BYTES];
    assert!(stored.reference_copy_backing(&mut bytes).unwrap());
    assert_eq!(&bytes, expected);
}

#[test]
fn held_frame_restore_refuses_persistent_ciphertext_corruption_then_retries() {
    let original = core::array::from_fn(|offset| (offset % 251) as u8);
    let ram = ram_with(original);
    let (disk, _, plan) = fixture(BLOCK as u32);
    let trace = Rc::clone(&disk.trace);
    let mut session = open(disk, 0x49).unwrap();
    let neighbor = [0x63; PAGE_BYTES];
    let neighbor_proof = session.write_slot(1, 0x54, &neighbor).unwrap();
    let adjacent = slot_start(&plan, 1)..slot_start(&plan, 1) + PAGE_BYTES;
    let adjacent_ciphertext = session.disk.device().disk.bytes[adjacent.clone()].to_vec();
    assert_ne!(adjacent_ciphertext, neighbor);
    let start = slot_start(&plan, 0);
    let flip = start + PAGE_BYTES - 1;
    let first_write = trace.borrow().writes.len();
    let first_flush = trace.borrow().flushes;
    trace.borrow_mut().observe_flush = Some((Arc::clone(&ram), FRAME));

    // SAFETY: this exact private fixture has no other CPU mapping or DMA user.
    unsafe {
        CpuMappingsRevokedFrame::with_reference_revoked_frame(Arc::clone(&ram), FRAME, |revoked| {
            assert_eq!(revoked.physical(), FRAME);
            let mut stored = session.store_revoked_frame(0, revoked).unwrap();
            assert_eq!(stored.physical(), FRAME);
            assert_nonresident(&ram);
            {
                let trace = trace.borrow();
                let lba = (start / BLOCK) as u64;
                assert_eq!(
                    &trace.writes[first_write..],
                    &[
                        (lba, 4096, false),
                        (lba + 8, 4096, false),
                        (lba + 16, 4096, false),
                        (lba + 24, 4096, false),
                    ]
                );
                assert_eq!(trace.flushes, first_flush + 1);
                assert_eq!(trace.resident_at_flush, [true]);
            }

            assert_backing_bytes(&stored, &original);
            assert!(stored.try_discard_retained_bytes().unwrap());
            assert_nonresident(&ram);
            assert_backing_bytes(&stored, &[0; PAGE_BYTES]);
            let first_read = trace.borrow().reads;
            trace.borrow_mut().flip_byte_on_read = Some(flip);
            for attempt in 1..=2 {
                assert!(matches!(
                    stored.try_restore(),
                    Err(FrameSwapError::Storage(Error::Authentication))
                ));
                assert_nonresident(&ram);
                assert_backing_bytes(&stored, &[0; PAGE_BYTES]);
                let trace = trace.borrow();
                assert!(trace.flip_byte_on_read.is_none());
                assert_eq!(trace.reads, first_read + 4 * attempt);
                assert_eq!(trace.writes.len(), first_write + 4);
                assert_eq!(trace.flushes, first_flush + 1);
            }

            trace.borrow_mut().flip_byte_on_read = Some(flip);
            assert!(stored.try_restore().unwrap());
            assert_resident_bytes(&ram, &original);
            assert_eq!(trace.borrow().reads, first_read + 12);
            assert!(stored.try_restore().unwrap());
            assert_eq!(trace.borrow().reads, first_read + 12);
            drop(stored);
        })
    }
    .unwrap();

    assert_resident_bytes(&ram, &original);
    drop(ram.reference_execution_loan().unwrap());
    let bytes = &session.disk.device().disk.bytes;
    assert_eq!(&bytes[adjacent], adjacent_ciphertext.as_slice());
    assert_ne!(&bytes[start..start + PAGE_BYTES], &original);
    let mut output = [0; PAGE_BYTES];
    session
        .read_slot(1, 0x54, &neighbor_proof, &mut output)
        .unwrap();
    assert_eq!(output, neighbor);
}

#[test]
fn unrelated_slot_proofs_cannot_restore_a_discarded_frame() {
    const PAGE: u64 = FRAME / PAGE_BYTES as u64;
    let original = [0x39; PAGE_BYTES];
    let ram = ram_with(original);
    let (disk, _, _) = fixture(BLOCK as u32);
    let trace = Rc::clone(&disk.trace);
    let mut session = open(disk, 0x59).unwrap();
    let mut wrong_page = session
        .write_slot(0, PAGE + 1, &[0x21; PAGE_BYTES])
        .unwrap();
    let mut stale_write = session.write_slot(0, PAGE, &[0x42; PAGE_BYTES]).unwrap();
    let mut wrong_slot = session.write_slot(1, PAGE, &[0x63; PAGE_BYTES]).unwrap();
    let held_counter = wrong_slot.counter + 1;
    let mut stale_generation = {
        let (disk, _, _) = fixture(BLOCK as u32);
        let mut other = open(disk, 0x5a).unwrap();
        let mut proof = other.write_slot(0, PAGE, &[0x84; PAGE_BYTES]).unwrap();
        for _ in 1..held_counter {
            proof = other.write_slot(0, PAGE, &[0x84; PAGE_BYTES]).unwrap();
        }
        proof
    };
    assert_eq!(stale_generation.counter, held_counter);
    assert_ne!(stale_generation.generation, session.header().generation());

    // SAFETY: this exact private fixture has no other CPU mapping or DMA user.
    unsafe {
        CpuMappingsRevokedFrame::with_reference_revoked_frame(Arc::clone(&ram), FRAME, |revoked| {
            let mut stored = session.store_revoked_frame(0, revoked).unwrap();
            let before = {
                let trace = trace.borrow();
                (trace.reads, trace.writes.len(), trace.flushes)
            };
            stored.reference_swap_proof(&mut wrong_slot);
            let discard = stored.try_discard_retained_bytes();
            stored.reference_swap_proof(&mut wrong_slot);
            assert!(matches!(
                discard,
                Err(FrameSwapError::Storage(Error::Authentication))
            ));
            assert_nonresident(&ram);
            assert_backing_bytes(&stored, &original);

            assert!(stored.try_discard_retained_bytes().unwrap());
            assert_backing_bytes(&stored, &[0; PAGE_BYTES]);
            for proof in [
                &mut wrong_slot,
                &mut wrong_page,
                &mut stale_write,
                &mut stale_generation,
            ] {
                stored.reference_swap_proof(proof);
                let result = stored.try_restore();
                stored.reference_swap_proof(proof);
                assert!(matches!(
                    result,
                    Err(FrameSwapError::Storage(Error::Authentication))
                ));
                assert_nonresident(&ram);
                assert_backing_bytes(&stored, &[0; PAGE_BYTES]);
                let trace = trace.borrow();
                assert_eq!((trace.reads, trace.writes.len(), trace.flushes), before);
            }

            assert!(stored.try_restore().unwrap());
            assert_resident_bytes(&ram, &original);
        })
    }
    .unwrap();
    assert_resident_bytes(&ram, &original);
    drop(ram.reference_execution_loan().unwrap());
}

#[test]
fn flush_failure_leaves_the_revoked_frame_resident_and_the_slot_unclaimed() {
    let original = [0x86; PAGE_BYTES];
    let ram = ram_with(original);
    let (disk, _, _) = fixture(BLOCK as u32);
    let trace = Rc::clone(&disk.trace);
    let mut session = open(disk, 0x67).unwrap();
    let first_flush = trace.borrow().flushes;
    {
        let mut trace = trace.borrow_mut();
        trace.observe_flush = Some((Arc::clone(&ram), FRAME));
        trace.fail_flush = true;
    }

    // SAFETY: this exact private fixture has no other CPU mapping or DMA user.
    unsafe {
        CpuMappingsRevokedFrame::with_reference_revoked_frame(Arc::clone(&ram), FRAME, |revoked| {
            assert!(matches!(
                session.store_revoked_frame(0, revoked),
                Err(FrameSwapError::Storage(Error::Device(BlockError::Timeout)))
            ));
            assert_resident_bytes(&ram, &original);
            let trace = trace.borrow();
            assert_eq!(trace.flushes, first_flush + 1);
            assert_eq!(trace.resident_at_flush, [true]);
        })
    }
    .unwrap();

    assert_resident_bytes(&ram, &original);
    drop(ram.reference_execution_loan().unwrap());
    trace.borrow_mut().fail_flush = false;
    let proof = session.write_slot(0, 7, &original).unwrap();
    let mut output = [0; PAGE_BYTES];
    session.read_slot(0, 7, &proof, &mut output).unwrap();
    assert_eq!(output, original);
}

#[test]
fn abandoned_nonresident_frame_keeps_its_slot_claimed() {
    for forget in [false, true] {
        let original = [0x96; PAGE_BYTES];
        let ram = ram_with(original);
        let (disk, _, _) = fixture(BLOCK as u32);
        let trace = Rc::clone(&disk.trace);
        let mut session = open(disk, 0x88).unwrap();

        let stopped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: this exact private fixture has no other CPU mapping or DMA user.
            unsafe {
                CpuMappingsRevokedFrame::with_reference_revoked_frame(
                    Arc::clone(&ram),
                    FRAME,
                    |revoked| {
                        let stored = session.store_revoked_frame(0, revoked).unwrap();
                        if forget {
                            core::mem::forget(stored);
                        } else {
                            drop(stored);
                        }
                    },
                )
            }
        }));
        assert!(stopped.is_err());
        assert_eq!(ram.nonresident_frame(FRAME, PAGE_BYTES), Some(FRAME));
        assert_eq!(
            ram.reference_execution_loan().err(),
            Some(MemoryStatus::Domain)
        );

        let before = {
            let trace = trace.borrow();
            (trace.writes.len(), trace.flushes)
        };
        assert!(matches!(
            session.write_slot(0, 7, &original),
            Err(Error::SlotHeld(0))
        ));
        let trace = trace.borrow();
        assert_eq!((trace.writes.len(), trace.flushes), before);
    }
}
