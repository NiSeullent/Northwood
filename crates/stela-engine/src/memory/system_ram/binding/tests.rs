// SPDX-License-Identifier: GPL-2.0-or-later
use super::super::{ReservedSystemRamRange, descriptor_frames::FRAME_BYTES, fixture};
use super::*;
use alloc::sync::Arc;
use core::{cell::UnsafeCell, ptr::NonNull};

const FRAME: u64 = FRAME_BYTES as u64;
const FIRST: u64 = 0x8000;
const NEXT: u64 = FIRST + FRAME;
const HOLE: u64 = NEXT + FRAME;
const LATER: u64 = HOLE + FRAME;

#[repr(align(16384))]
struct Storage(UnsafeCell<[u8; 3 * FRAME_BYTES]>);

// SAFETY: only the fixture's registered RAM may access these bytes.
unsafe impl Sync for Storage {}

fn ram() -> RegisteredSystemRam {
    let storage = Arc::new(Storage(UnsafeCell::new([0; 3 * FRAME_BYTES])));
    let base = storage.0.get().cast::<u8>();
    let ranges =
        [(FIRST, FRAME_BYTES), (NEXT, 0), (LATER, 2 * FRAME_BYTES)].map(|(source_pa, offset)| {
            ReservedSystemRamRange {
                source_pa,
                host_pa: 0x1000_0000 + offset as u64,
                host_base: NonNull::new(base.wrapping_add(offset)).unwrap(),
                bytes: FRAME_BYTES,
            }
        });
    // SAFETY: the fixture retains the aligned, initialized and disjoint extents.
    let mut ram = unsafe { fixture(&ranges, storage) };
    ram.enable_descriptor_frame_leases().unwrap();
    ram
}

#[test]
fn classify_page_distinguishes_complete_registration_from_partial_coverage() {
    let ram = ram();
    let loan = ram.cpu_execution_loan().unwrap();
    let binding = ram.cpu_binding(&loan).unwrap();
    assert_eq!(
        binding.classify_page(FIRST, 4096),
        Ok(RamPageKind::Resident)
    );
    // Adjacent GPAs can have discontiguous HPA runs.
    assert_eq!(
        binding.classify_page(NEXT - 8, 16),
        Ok(RamPageKind::Resident)
    );
    assert_eq!(
        binding.classify_page(HOLE, FRAME_BYTES),
        Ok(RamPageKind::Unmapped)
    );
    assert_eq!(
        binding.classify_page(u64::MAX, 1),
        Ok(RamPageKind::Unmapped)
    );
    for (guest, bytes) in [
        (HOLE - 8, 16),
        (LATER - 8, 16),
        (FIRST, 3 * FRAME_BYTES + 16),
    ] {
        assert_eq!(
            binding.classify_page(guest, bytes),
            Err(MemoryStatus::Translation)
        );
    }
    assert_eq!(
        binding.classify_page(FIRST, 0),
        Err(MemoryStatus::InvalidRequest)
    );
    assert_eq!(
        binding.classify_page(u64::MAX, 2),
        Err(MemoryStatus::Translation)
    );
}

#[test]
fn classify_page_reports_the_exact_missing_frame_and_rejects_a_stale_epoch() {
    let ram = ram();
    ram.model_set_nonresident(NEXT);
    let loan = ram.cpu_execution_loan().unwrap();
    let mut binding = ram.cpu_binding(&loan).unwrap();
    assert_eq!(
        binding.classify_page(NEXT + 4096, 4096),
        Ok(RamPageKind::Nonresident { frame: NEXT })
    );
    assert_eq!(
        binding.classify_page(NEXT - 8, 16),
        Ok(RamPageKind::Nonresident { frame: NEXT })
    );
    assert_eq!(
        binding.classify_page(LATER, 4096),
        Ok(RamPageKind::Resident)
    );
    binding.binding.generation += 1;
    assert_eq!(binding.classify_page(HOLE, 4096), Err(MemoryStatus::Domain));
    drop(loan);

    ram.model_set_resident(NEXT);
    let loan = ram.cpu_execution_loan().unwrap();
    let binding = ram.cpu_binding(&loan).unwrap();
    assert_eq!(binding.classify_page(NEXT, 4096), Ok(RamPageKind::Resident));
}
