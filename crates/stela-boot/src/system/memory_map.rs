// SPDX-License-Identifier: GPL-2.0-or-later

//! The same physical-memory accounting for the loader and the engine handoff.
use super::{LARGE, MAX_GUEST_RANGES, PAGE, is_ram, output};
use stela_boot_core::ram::{aligned_range, merge_aligned_ranges};
use uefi::{boot::MemoryType, mem::memory_map::MemoryMap};

/// Bounds scratch space even after ExitBootServices. An oversized map reports
/// the omitted memory and never offers it to the guest.
const CANDIDATES: usize = 1024;
const STANDARD_TYPES: usize = 16;

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct TypeTotals {
    entries: usize,
    bytes: u64,
    aligned: u64,
}

pub(super) struct Map {
    pub(super) ranges: [(u64, u64); MAX_GUEST_RANGES],
    pub(super) count: usize,
    pub(super) ledger: Ledger,
}

/// The final observation travels with the handoff so the engine can persist
/// it after opening the boot logs with its own storage drivers.
#[repr(C)]
pub(crate) struct Ledger {
    types: [TypeTotals; STANDARD_TYPES + 1],
    entries: usize,
    pub(super) installed: u64,
    pub(super) usable: u64,
    pub(super) separate_usable: u64,
    pub(super) align_loss: u64,
    separate_loss: u64,
    largest: (u64, u64),
    dropped: u64,
    omitted: u64,
}

impl Ledger {
    pub(crate) fn report(&self, stage: &str) {
        for (kind, totals) in self
            .types
            .iter()
            .enumerate()
            .filter(|(_, ty)| ty.entries != 0)
        {
            let name = (kind < STANDARD_TYPES).then_some(MemoryType(kind as u32));
            output::arguments(format_args!(
                "HOST MAP stage={stage} type={name:?} entries={} bytes={:#x} aligned={:#x}\n",
                totals.entries, totals.bytes, totals.aligned,
            ));
        }
        let (base, largest) = self.largest;
        output::arguments(format_args!(
            "HOST MAP stage={stage} installed={:#x} usable={:#x} largest={largest:#x}@{base:#x} \
             entries={} align_loss={:#x} merged_loss={:#x} dropped={:#x} candidates_omitted={:#x}\n",
            self.installed,
            self.usable,
            self.entries,
            self.separate_loss,
            self.align_loss,
            self.dropped,
            self.omitted,
        ));
    }
}

pub(super) fn summarize(map: &impl MemoryMap, stage: Option<&str>) -> Map {
    let mut types = [TypeTotals::default(); STANDARD_TYPES + 1];
    let mut candidates = [(0, 0); CANDIDATES];
    let (mut found, mut entries, mut omitted) = (0, 0, 0);
    let (mut installed, mut raw, mut separate) = (0, 0, 0);
    for entry in map.entries() {
        entries += 1;
        let bytes = entry.page_count * PAGE;
        let end = entry.phys_start + bytes;
        let (_, aligned) = aligned_range(entry.phys_start, end, LARGE);
        let totals = &mut types[(entry.ty.0 as usize).min(STANDARD_TYPES)];
        totals.entries += 1;
        totals.bytes += bytes;
        totals.aligned += aligned;
        if is_ram(entry.ty) {
            installed += bytes;
        }
        if !matches!(
            entry.ty,
            MemoryType::CONVENTIONAL
                | MemoryType::BOOT_SERVICES_CODE
                | MemoryType::BOOT_SERVICES_DATA
        ) || entry.att.bits() & super::RUNTIME_ATTRIBUTE != 0
        {
            continue;
        }
        // The retained identity map leaves physical page zero unmapped.
        let start = entry.phys_start.max(PAGE);
        if start >= end {
            continue;
        }
        separate += aligned_range(start, end, LARGE).1;
        if let Some(slot) = candidates.get_mut(found) {
            *slot = (start, end);
            found += 1;
            raw += end - start;
        } else {
            omitted += end - start;
        }
    }
    let mut ranges = [(0, 0); MAX_GUEST_RANGES];
    let (count, dropped) = merge_aligned_ranges(&mut candidates[..found], &mut ranges, LARGE);
    let ranges_kept = &ranges[..count];
    let usable = ranges_kept.iter().map(|&(_, bytes)| bytes).sum::<u64>();
    let ledger = Ledger {
        types,
        entries,
        installed,
        usable,
        separate_usable: separate,
        align_loss: raw.saturating_sub(usable + dropped),
        separate_loss: (raw + omitted).saturating_sub(separate),
        largest: ranges_kept
            .iter()
            .max_by_key(|r| r.1)
            .copied()
            .unwrap_or_default(),
        dropped,
        omitted,
    };
    if let Some(stage) = stage {
        ledger.report(stage);
    }
    Map {
        ranges,
        count,
        ledger,
    }
}
