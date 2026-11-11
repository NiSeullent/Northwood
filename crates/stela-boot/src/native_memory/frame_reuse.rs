// SPDX-License-Identifier: GPL-2.0-or-later
//! Exact PFN reuse in the standalone native host check.

use super::{PAGE, SOURCE};
use stela_drivers::bus::dma::memory::{
    CacheAttribute, CpuMapping, MemoryError, PageSource, PinnedPages,
};

const FRAME: usize = 4 * PAGE;

fn page_numbers(pages: &PinnedPages) -> Result<[u64; 4], MemoryError> {
    let addresses: [u64; 4] = pages
        .physical_pages()
        .try_into()
        .map_err(|_| MemoryError::InvalidPageList)?;
    if pages.bytes() != FRAME
        || pages.logical_bytes() != FRAME
        || pages.page_shift() != PAGE.trailing_zeros() as u8
        || !addresses[0].is_multiple_of(FRAME as u64)
        || addresses
            .windows(2)
            .any(|pair| pair[0].checked_add(PAGE as u64) != Some(pair[1]))
    {
        return Err(MemoryError::InvalidPageList);
    }
    Ok(addresses.map(|address| address / PAGE as u64))
}

fn write_pattern(mapping: &CpuMapping, seed: u8) -> Result<(), MemoryError> {
    let base = mapping.address(0, FRAME)?.as_ptr();
    for page in 0..4 {
        // SAFETY: this diagnostic exclusively owns the checked WB extent.
        unsafe { base.add(page * PAGE).write_bytes(seed + page as u8, PAGE) };
    }
    Ok(())
}

fn expect_pattern(mapping: &CpuMapping, seed: Option<u8>) -> Result<(), MemoryError> {
    let base = mapping.address(0, FRAME)?.as_ptr();
    for offset in 0..FRAME {
        let expected = seed.map_or(0, |seed| seed + (offset / PAGE) as u8);
        // SAFETY: the entire range belongs to this live exclusive mapping.
        if unsafe { base.add(offset).read_volatile() } != expected {
            return Err(MemoryError::InvalidPageList);
        }
    }
    Ok(())
}

pub(super) fn verify() -> Result<(), MemoryError> {
    let heap = SOURCE.registration();
    let heap_end = heap
        .physical
        .checked_add(heap.bytes as u64)
        .ok_or(MemoryError::OutsideReservation)?;
    let allocate = |end| SOURCE.allocate_below(FRAME, FRAME, CacheAttribute::WriteBack, end);
    // The first allocation uses the lowest fit. Its end then admits only
    // that original candidate in this cold diagnostic.
    let original = allocate(heap_end)?;
    let old = page_numbers(&original)?;
    let old_end = (old[3] + 1) * PAGE as u64;
    let original_mapping = original.cpu_mapping(CacheAttribute::WriteBack)?;
    write_pattern(&original_mapping, 0x31)?;
    drop(original);
    if !matches!(allocate(old_end), Err(MemoryError::OutOfMemory)) {
        return Err(MemoryError::InvalidPageList);
    }
    expect_pattern(&original_mapping, Some(0x31))?;
    drop(original_mapping);

    let sentinel = allocate(old_end)?;
    let reused = page_numbers(&sentinel)?;
    if reused != old {
        return Err(MemoryError::InvalidPageList);
    }
    let sentinel_mapping = sentinel.cpu_mapping(CacheAttribute::WriteBack)?;
    expect_pattern(&sentinel_mapping, None)?;
    write_pattern(&sentinel_mapping, 0x63)?;

    let spare = allocate(heap_end)?;
    let replacement = page_numbers(&spare)?;
    if replacement.iter().any(|page| reused.contains(page)) {
        return Err(MemoryError::InvalidPageList);
    }
    let spare_mapping = spare.cpu_mapping(CacheAttribute::WriteBack)?;
    expect_pattern(&spare_mapping, None)?;
    write_pattern(&spare_mapping, 0x97)?;
    expect_pattern(&spare_mapping, Some(0x97))?;
    expect_pattern(&sentinel_mapping, Some(0x63))?;
    crate::output::arguments(format_args!(
        "CHECK native-frame-reuse PASS page_bytes={PAGE} old_pfn={old:x?} reused_pfn={reused:x?} replacement_pfn={replacement:x?} live_pin_refused=true zeroed=true sentinel_preserved=true\n"
    ));
    Ok(())
}
