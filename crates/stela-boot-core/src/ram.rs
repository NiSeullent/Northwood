//! Copy admission over immutable, guest-address-sorted RAM ranges. A refused
//! physical transfer must not modify either RAM or the caller's read buffer.

use stela_compiler::context::memory::MemoryStatus;
use stela_engine::memory::{
    RegisteredSystemRam, domain::MemoryExecutionLoan, system_ram::CpuRamBinding,
};

/// Guest DRAM backed by one contiguous host physical range.
#[derive(Debug, Clone, Copy)]
pub struct RamRange {
    pub guest: u64,
    pub host: u64,
    pub bytes: u64,
}

/// Lay `ram` out end to end in guest physical memory from `base`, in order,
/// keeping each range's host backing.
pub fn place(ram: &mut [RamRange], base: u64) {
    let mut guest = base;
    for range in ram {
        range.guest = guest;
        guest += range.bytes;
    }
}

/// The longest RAM run starting at `guest`, as (host physical, bytes).
pub fn run(ram: &[RamRange], guest: u64) -> Option<(u64, u64)> {
    let (range, offset) = range_at(ram, guest)?;
    Some((range.host.checked_add(offset)?, range.bytes - offset))
}

fn range_at(ram: &[RamRange], guest: u64) -> Option<(&RamRange, u64)> {
    let index = ram.partition_point(|range| range.guest <= guest);
    let range = ram.get(index.checked_sub(1)?)?;
    let offset = guest - range.guest;
    (offset < range.bytes).then_some((range, offset))
}

/// Whether the retained layout names this GPA, without looking up an HPA.
pub fn contains(ram: &[RamRange], guest: u64) -> bool {
    range_at(ram, guest).is_some()
}

enum NativeRamMap<'a> {
    Legacy,
    Registered(CpuRamBinding<'a>),
}

/// Owner-local native physical lookup for one execution entry.
pub struct NativeRam<'a> {
    legacy: &'a [RamRange],
    map: NativeRamMap<'a>,
}

impl<'a> NativeRam<'a> {
    /// A registered entry requires its live execution loan and expected generation.
    pub fn new(
        legacy: &'a [RamRange],
        registered: Option<&'a RegisteredSystemRam>,
        entry: Option<(&'a MemoryExecutionLoan, u64)>,
    ) -> Result<Self, MemoryStatus> {
        let map = match (registered, entry) {
            (None, None) => NativeRamMap::Legacy,
            (Some(ram), Some((loan, generation))) => {
                let binding = ram.cpu_binding(loan)?;
                if binding.generation() != generation {
                    return Err(MemoryStatus::Domain);
                }
                NativeRamMap::Registered(binding)
            }
            _ => return Err(MemoryStatus::Domain),
        };
        Ok(Self { legacy, map })
    }

    /// Return the host physical run starting at `guest`, limited by `max_bytes`.
    pub fn physical_run(
        &self,
        guest: u64,
        max_bytes: u64,
    ) -> Result<Option<(u64, u64)>, MemoryStatus> {
        match &self.map {
            NativeRamMap::Legacy if max_bytes == 0 => Ok(None),
            NativeRamMap::Legacy => {
                Ok(run(self.legacy, guest).map(|(host, bytes)| (host, bytes.min(max_bytes))))
            }
            NativeRamMap::Registered(binding) => {
                let current = binding.physical_run(guest, max_bytes)?;
                if max_bytes != 0 && current.is_none() && contains(self.legacy, guest) {
                    return Err(MemoryStatus::Domain);
                }
                Ok(current)
            }
        }
    }

    /// Return the HPA of a page. Registered RAM must cover the entire page
    /// and its base must fit the address field used by the caller.
    pub fn physical_page(
        &self,
        guest: u64,
        page_bytes: u64,
        host_address_mask: u64,
    ) -> Result<Option<u64>, MemoryStatus> {
        if matches!(self.map, NativeRamMap::Legacy) {
            return Ok(run(self.legacy, guest).map(|(host, _)| host));
        }
        if page_bytes == 0 {
            return Err(MemoryStatus::InvalidRequest);
        }
        let Some((host, bytes)) = self.physical_run(guest, page_bytes)? else {
            return Ok(None);
        };
        if bytes != page_bytes || host & !host_address_mask != 0 {
            return Err(MemoryStatus::Domain);
        }
        Ok(Some(host))
    }
}

/// Calls `action` with each host run behind `bytes` of guest RAM at `guest`, as
/// (host physical, bytes). False, partway, if part of it is not RAM.
pub fn runs(
    ram: &[RamRange],
    mut guest: u64,
    mut bytes: u64,
    mut action: impl FnMut(u64, u64),
) -> bool {
    while bytes != 0 {
        let Some((host, length)) = run(ram, guest) else {
            return false;
        };
        let length = length.min(bytes);
        action(host, length);
        guest += length;
        bytes -= length;
    }
    true
}

/// The host ranges `candidates` (`start..end`, in the firmware map's order,
/// which UEFI does not sort) hold, sorted by address with touching and
/// overlapping ones joined, into `ranges` as (start, bytes): how many, and
/// the bytes of those that did not fit. Empty candidates are skipped.
pub fn merge_ranges(candidates: &mut [(u64, u64)], ranges: &mut [(u64, u64)]) -> (usize, u64) {
    merge_aligned_ranges(candidates, ranges, 1)
}

/// Join touching candidates before trimming their outer edges to whole
/// `alignment`-byte pages. Return the ranges written and the aligned bytes
/// omitted because the destination was full.
pub fn merge_aligned_ranges(
    candidates: &mut [(u64, u64)],
    ranges: &mut [(u64, u64)],
    alignment: u64,
) -> (usize, u64) {
    assert!(alignment.is_power_of_two());
    candidates.sort_unstable();
    let (mut count, mut dropped) = (0usize, 0u64);
    let mut store = |(start, end)| {
        let (start, bytes) = aligned_range(start, end, alignment);
        if bytes != 0 {
            if let Some(slot) = ranges.get_mut(count) {
                *slot = (start, bytes);
                count += 1;
            } else {
                dropped += bytes;
            }
        }
    };
    let mut pending = None;
    for &(start, end) in candidates.iter().filter(|(start, end)| start < end) {
        match pending {
            Some((base, top)) if start <= top => pending = Some((base, top.max(end))),
            _ => {
                if let Some(previous) = pending.replace((start, end)) {
                    store(previous);
                }
            }
        }
    }
    if let Some(last) = pending {
        store(last);
    }
    (count, dropped)
}

/// Trim the half-open range to whole pages, as (start, bytes).
pub fn aligned_range(start: u64, end: u64, alignment: u64) -> (u64, u64) {
    let Some(start) = start.checked_next_multiple_of(alignment) else {
        return (0, 0);
    };
    let end = end / alignment * alignment;
    (start, end.saturating_sub(start))
}

/// Insert a host extent into the dense guest address space described by the
/// first `count` `(host, bytes)` pairs, which the caller keeps physically
/// disjoint and in guest order. `guest_offset` is relative to the start of
/// that space and may be at its end. All extents use the given alignment.
/// Returns the enlarged count, or leaves the buffer untouched on refusal.
pub fn insert_backing(
    ranges: &mut [(u64, u64)],
    count: usize,
    guest_offset: u64,
    backing: (u64, u64),
    alignment: u64,
) -> Option<usize> {
    let (backing_host, backing_bytes) = backing;
    if !alignment.is_power_of_two()
        || !guest_offset.is_multiple_of(alignment)
        || !backing_host.is_multiple_of(alignment)
        || backing_bytes == 0
        || !backing_bytes.is_multiple_of(alignment)
    {
        return None;
    }
    let backing_end = backing_host.checked_add(backing_bytes)?;
    let mut guest_end = 0u64;
    let mut insertion = (count, 0);
    for (index, &(host, bytes)) in ranges.get(..count)?.iter().enumerate() {
        if !host.is_multiple_of(alignment) || bytes == 0 || !bytes.is_multiple_of(alignment) {
            return None;
        }
        let host_end = host.checked_add(bytes)?;
        if backing_host < host_end && host < backing_end {
            return None;
        }
        let guest_start = guest_end;
        guest_end = guest_end.checked_add(bytes)?;
        if (guest_start..guest_end).contains(&guest_offset) {
            insertion = (index, guest_offset - guest_start);
        }
    }
    if guest_offset > guest_end {
        return None;
    }
    guest_end.checked_add(backing_bytes)?;
    let (index, split) = insertion;
    let new_count = count.checked_add(1 + usize::from(split != 0))?;
    if new_count > ranges.len() {
        return None;
    }

    if split == 0 {
        ranges.copy_within(index..count, index + 1);
        ranges[index] = backing;
    } else {
        let (host, bytes) = ranges[index];
        ranges.copy_within(index + 1..count, index + 3);
        ranges[index..index + 3].copy_from_slice(&[
            (host, split),
            backing,
            (host + split, bytes - split),
        ]);
    }
    Some(new_count)
}

/// The immutable map and all host backing remain owned by the caller through
/// both passes. `action` is infallible and receives only complete admitted
/// chunks. UAT/DART page-by-page prefix semantics belong to their outer owner.
pub fn copy(
    ram: &[RamRange],
    guest: u64,
    length: usize,
    mut action: impl FnMut(u64, usize, usize),
) -> bool {
    if length == 0 {
        return true;
    }
    let Ok(bytes) = u64::try_from(length) else {
        return false;
    };
    if guest.checked_add(bytes - 1).is_none() {
        return false;
    }
    let Some((host, available)) = run(ram, guest) else {
        return false;
    };
    // The common one-run transfer needs one lookup and no replay.
    if bytes <= available {
        if host.checked_add(bytes - 1).is_none() {
            return false;
        }
        action(host, 0, length);
        return true;
    }
    if !walk(ram, guest, length, |_, _, _| {}) {
        return false;
    }
    // The map is immutably borrowed and unchanged, so the same walk cannot
    // discover a new hole after an earlier chunk has had an effect.
    walk(ram, guest, length, action)
}

fn walk(
    ram: &[RamRange],
    guest: u64,
    length: usize,
    mut action: impl FnMut(u64, usize, usize),
) -> bool {
    let mut done = 0;
    while done < length {
        let Some((host, available)) = run(ram, guest + done as u64) else {
            return false;
        };
        let chunk = usize::try_from(available)
            .unwrap_or(usize::MAX)
            .min(length - done);
        if host.checked_add(chunk as u64 - 1).is_none() {
            return false;
        }
        action(host, done, chunk);
        done += chunk;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_ram_keeps_legacy_page_lookup_for_unregistered_entries() {
        let legacy = [RamRange {
            guest: 0x4000,
            host: 0x8020,
            bytes: 64,
        }];
        let memory = NativeRam::new(&legacy, None, None).unwrap();
        assert_eq!(memory.physical_run(0x4000, 16), Ok(Some((0x8020, 16))));
        assert_eq!(memory.physical_run(0x4000, 4096), Ok(Some((0x8020, 64))));
        assert_eq!(memory.physical_run(0x4000, 0), Ok(None));
        assert_eq!(memory.physical_page(0x4000, 4096, !0xfff), Ok(Some(0x8020)));
        assert_eq!(memory.physical_page(0x5000, 4096, !0xfff), Ok(None));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn native_registered_ram_requires_its_exact_live_binding() {
        use alloc::sync::Arc;
        use core::{cell::UnsafeCell, ptr::NonNull};
        use stela_engine::memory::{ReservedSystemRamRange, system_ram::fixture};

        const PAGE: u64 = 4096;
        const BASE: u64 = 0x8000;
        const HPA: u64 = 0x2_0000;
        const HOLE: u64 = BASE + 2 * PAGE;
        const SHORT: u64 = HOLE + 2 * PAGE;
        const MASK: u64 = 0x000f_ffff_ffff_f000;

        #[repr(align(4096))]
        struct Storage(UnsafeCell<[u8; 3 * PAGE as usize]>);
        // SAFETY: only the registered owner may access these bytes.
        unsafe impl Sync for Storage {}

        let make_ram = || {
            let storage = Arc::new(Storage(UnsafeCell::new([0; 3 * PAGE as usize])));
            let pointer = storage.0.get().cast::<u8>();
            let ranges = [
                ReservedSystemRamRange {
                    source_pa: BASE,
                    host_pa: HPA,
                    host_base: NonNull::new(pointer).unwrap(),
                    bytes: 2 * PAGE as usize,
                },
                ReservedSystemRamRange {
                    source_pa: SHORT,
                    host_pa: 2 * HPA,
                    host_base: NonNull::new(pointer.wrapping_add(2 * PAGE as usize)).unwrap(),
                    bytes: PAGE as usize / 2,
                },
            ];
            // SAFETY: the owner keeps the disjoint pinned and initialized extents.
            unsafe { fixture(&ranges, storage) }
        };
        let ram = make_ram();
        let other = make_ram();
        let loan = ram.cpu_execution_loan().unwrap();
        let foreign = other.cpu_execution_loan().unwrap();
        assert!(ram.try_service().unwrap().is_none());
        let generation = ram.generation().unwrap();
        assert_eq!(other.generation().unwrap(), generation);
        let legacy = [RamRange {
            guest: BASE,
            host: 3 * HPA,
            bytes: 3 * PAGE,
        }];

        for rejected in [
            None,
            Some((&loan, generation + 1)),
            Some((&foreign, generation)),
        ] {
            assert!(matches!(
                NativeRam::new(&legacy, Some(&ram), rejected),
                Err(MemoryStatus::Domain)
            ));
        }
        assert!(matches!(
            NativeRam::new(&legacy, None, Some((&loan, generation))),
            Err(MemoryStatus::Domain)
        ));

        let memory = NativeRam::new(&legacy, Some(&ram), Some((&loan, generation))).unwrap();
        assert_eq!(
            memory.physical_run(BASE, 4 * PAGE),
            Ok(Some((HPA, 2 * PAGE)))
        );
        assert_eq!(
            memory.physical_page(BASE + PAGE, PAGE, MASK),
            Ok(Some(HPA + PAGE))
        );
        assert_eq!(memory.physical_run(HOLE, PAGE), Err(MemoryStatus::Domain));
        assert_eq!(
            memory.physical_page(HOLE, PAGE, MASK),
            Err(MemoryStatus::Domain)
        );
        assert_eq!(memory.physical_page(HOLE + PAGE, PAGE, MASK), Ok(None));
        assert_eq!(
            memory.physical_run(SHORT, PAGE),
            Ok(Some((2 * HPA, PAGE / 2)))
        );
        assert_eq!(
            memory.physical_page(SHORT, PAGE, MASK),
            Err(MemoryStatus::Domain)
        );
        assert_eq!(
            memory.physical_page(BASE, PAGE, 0xf000),
            Err(MemoryStatus::Domain)
        );
        drop(loan);
        assert!(ram.try_service().unwrap().is_some());
    }

    #[test]
    fn merged_ranges_are_sorted_joined_and_count_what_does_not_fit() {
        let mut candidates = [
            (0x4000, 0x6000),
            (0x1000, 0x2000),
            (0x9000, 0x9000),
            (0x2000, 0x3000),
            (0x8000, 0x9000),
            (0x5000, 0x7000),
        ];
        let mut ranges = [(0, 0); 2];
        assert_eq!(merge_ranges(&mut candidates, &mut ranges), (2, 0x1000));
        assert_eq!(ranges, [(0x1000, 0x2000), (0x4000, 0x3000)]);
    }

    #[test]
    fn two_hundred_unsorted_fragments_merge_in_address_order() {
        let mut candidates: [(u64, u64); 200] = core::array::from_fn(|index| {
            let page = (index as u64 * 37) % 200;
            // Every other pair of pages touches its neighbour.
            (page << 21, (page << 21) + (1 << 21) - ((page & 2) << 19))
        });
        let mut ranges = [(0, 0); 128];
        let (count, dropped) = merge_ranges(&mut candidates, &mut ranges);
        assert_eq!((count, dropped), (100, 0));
        assert!(
            ranges[..count]
                .windows(2)
                .all(|pair| pair[0].0 + pair[0].1 < pair[1].0)
        );
    }

    #[test]
    fn whole_pages_can_span_descriptor_boundaries_but_not_holes() {
        let mut candidates = [(25, 31), (2, 13), (33, 40), (13, 24), (26, 32), (32, 33)];
        let mut ranges = [(0, 0); 2];
        assert_eq!(
            merge_aligned_ranges(&mut candidates, &mut ranges, 8),
            (2, 0)
        );
        assert_eq!(ranges, [(8, 16), (32, 8)]);
        assert_eq!(
            merge_aligned_ranges(&mut candidates, &mut ranges[..1], 8),
            (1, 8)
        );
        assert_eq!(aligned_range(u64::MAX - 1, u64::MAX, 8), (0, 0));
    }

    #[test]
    fn insert_backing_at_start_boundary_interior_and_end() {
        let high = (0x9000, 0x2000);
        let low = (0x1000, 0x3000);
        let backing = (0x4000, 0x2000);
        let unused = (0xdead, 0xbeef);
        let cases: [(u64, &[(u64, u64)]); 4] = [
            (0, &[backing, high, low]),
            (0x2000, &[high, backing, low]),
            (0x1000, &[(0x9000, 0x1000), backing, (0xa000, 0x1000), low]),
            (0x5000, &[high, low, backing]),
        ];
        for (offset, expected) in cases {
            let mut ranges = [high, low, unused, unused, unused];
            let count = insert_backing(&mut ranges, 2, offset, backing, 0x1000);
            assert_eq!(count, Some(expected.len()));
            assert_eq!(&ranges[..expected.len()], expected);
            assert!(ranges[expected.len()..].iter().all(|&pair| pair == unused));
        }
        let mut empty = [unused];
        assert_eq!(insert_backing(&mut empty, 0, 0, backing, 0x1000), Some(1));
        assert_eq!(empty, [backing]);
    }

    #[test]
    fn overlapping_backing_and_insufficient_slots_leave_the_map_unchanged() {
        let original = [(0x9000, 0x2000), (0x1000, 0x3000), (0xdead, 0xbeef), (0, 0)];
        let mut ranges = original;
        for backing in [(0, 0x2000), (0x2000, 0x1000), (0xa000, 0x2000)] {
            assert_eq!(insert_backing(&mut ranges, 2, 0, backing, 0x1000), None);
            assert_eq!(ranges, original);
        }
        let backing = (0x4000, 0x2000);
        assert_eq!(
            insert_backing(&mut ranges[..2], 2, 0, backing, 0x1000),
            None
        );
        assert_eq!(ranges, original);
        assert_eq!(
            insert_backing(&mut ranges[..3], 2, 0x1000, backing, 0x1000),
            None
        );
        assert_eq!(ranges, original);
        assert_eq!(
            insert_backing(&mut ranges[..3], 2, 0, backing, 0x1000),
            Some(3)
        );
    }

    #[test]
    fn insert_backing_rejects_malformed_geometry_before_mutation() {
        let existing = [(0x4000, 0x2000), (0x9000, 0x1000), (0xdead, 0xbeef)];
        let backing = (0x1000, 0x1000);
        for (count, offset, backing, alignment) in [
            (4, 0, backing, 0x1000),
            (2, 0x4000, backing, 0x1000),
            (2, 0, backing, 0),
            (2, 0, backing, 3),
            (2, 1, backing, 0x1000),
            (2, 0, (0x1001, 0x1000), 0x1000),
            (2, 0, (0x1000, 0), 0x1000),
            (2, 0, (0x1000, 1), 0x1000),
            (2, 0, (u64::MAX - 0xfff, 0x1000), 0x1000),
        ] {
            let mut ranges = existing;
            assert_eq!(
                insert_backing(&mut ranges, count, offset, backing, alignment),
                None
            );
            assert_eq!(ranges, existing);
        }
        for invalid in [
            (0x9001, 0x1000),
            (0x9000, 0),
            (0x9000, 1),
            (u64::MAX - 0xfff, 0x1000),
        ] {
            let original = [existing[0], invalid, existing[2]];
            let mut ranges = original;
            assert_eq!(insert_backing(&mut ranges, 2, 0, backing, 0x1000), None);
            assert_eq!(ranges, original);
        }
    }

    #[test]
    fn dense_guest_size_overflow_is_rejected() {
        for length in [2, 3] {
            // Even if the caller supplied overlapping original ranges,
            // neither the old nor the enlarged dense guest size may wrap.
            let original = [(0, u64::MAX - 2), (0, length), (0, 0)];
            let mut ranges = original;
            assert_eq!(
                insert_backing(&mut ranges, 2, 0, (u64::MAX - 2, 1), 1),
                None
            );
            assert_eq!(ranges, original);
        }
    }

    #[test]
    fn a_guest_range_is_its_host_runs() {
        let ram = [
            RamRange {
                guest: 0x1000,
                host: 0x8000,
                bytes: 0x1000,
            },
            RamRange {
                guest: 0x2000,
                host: 0x4000,
                bytes: 0x2000,
            },
        ];
        let mut found = [(0, 0); 2];
        let mut count = 0;
        let whole = runs(&ram, 0x1800, 0x1000, |host, bytes| {
            found[count] = (host, bytes);
            count += 1;
        });
        assert!(whole);
        assert_eq!(found, [(0x8800, 0x800), (0x4000, 0x800)]);
        assert!(!runs(&ram, 0x3800, 0x1000, |_, _| {}));
        assert!(runs(&ram, 0x9000, 0, |_, _| {}));
    }

    #[test]
    fn holes_and_outer_boundaries_refuse_without_read_or_write_prefixes() {
        let ram = [
            RamRange {
                guest: 8,
                host: 0,
                bytes: 8,
            },
            RamRange {
                guest: 16,
                host: 24,
                bytes: 8,
            },
            RamRange {
                guest: 32,
                host: 8,
                bytes: 8,
            },
        ];
        for start in 0..=48u64 {
            for length in 0..=49usize {
                let admitted = (start..start + length as u64)
                    .all(|byte| (8..24).contains(&byte) || (32..40).contains(&byte));
                let mut host = [0xa5u8; 40];
                let mut output = [0x5au8; 49];
                let mut calls = 0;
                assert_eq!(
                    copy(&ram, start, length, |physical, done, chunk| {
                        calls += 1;
                        host[physical as usize..physical as usize + chunk].fill(0x33);
                        output[done..done + chunk].fill(0x77);
                    }),
                    admitted,
                    "start={start} length={length}"
                );
                if !admitted || length == 0 {
                    assert_eq!(calls, 0);
                    assert_eq!(host, [0xa5; 40]);
                    assert_eq!(output, [0x5a; 49]);
                } else {
                    assert!(output[..length].iter().all(|byte| *byte == 0x77));
                    assert!(output[length..].iter().all(|byte| *byte == 0x5a));
                }
            }
        }
    }

    #[test]
    fn adjacent_guest_ranges_keep_their_distinct_host_offsets() {
        let ram = [
            RamRange {
                guest: 8,
                host: 0x1000,
                bytes: 8,
            },
            RamRange {
                guest: 16,
                host: 0x9000,
                bytes: 8,
            },
            RamRange {
                guest: 24,
                host: 0x4000,
                bytes: 8,
            },
        ];
        let mut chunks = std::vec::Vec::new();
        assert!(copy(&ram, 12, 16, |host, done, bytes| chunks
            .push((host, done, bytes))));
        assert_eq!(chunks, [(0x1004, 0, 4), (0x9000, 4, 8), (0x4000, 12, 4)]);
        assert_eq!(run(&ram, 32), None);
    }

    #[test]
    fn placing_moves_guest_addresses_and_keeps_host_backing() {
        let mut ram = [
            RamRange {
                guest: 0,
                host: 0x9000,
                bytes: 0x2000,
            },
            RamRange {
                guest: 0x2000,
                host: 0x1000,
                bytes: 0x1000,
            },
        ];
        place(&mut ram, 0x8_0000_0000);
        assert_eq!(run(&ram, 0x8_0000_0000), Some((0x9000, 0x2000)));
        assert_eq!(run(&ram, 0x8_0000_2800), Some((0x1800, 0x800)));
        assert_eq!(run(&ram, 0x2000), None);
    }

    #[test]
    fn address_wrap_refuses_before_effects_and_last_address_is_representable() {
        let ram = [RamRange {
            guest: u64::MAX - 7,
            host: 0x1000,
            bytes: 8,
        }];
        let mut calls = 0;
        assert!(!copy(&ram, u64::MAX - 3, 5, |_, _, _| calls += 1));
        assert_eq!(calls, 0);
        assert!(copy(&ram, u64::MAX - 3, 4, |host, done, bytes| {
            assert_eq!((host, done, bytes), (0x1004, 0, 4));
            calls += 1;
        }));
        assert_eq!(calls, 1);
        let ram = [
            RamRange {
                guest: 0,
                host: 0x1000,
                bytes: 4,
            },
            RamRange {
                guest: 4,
                host: u64::MAX - 1,
                bytes: 4,
            },
        ];
        assert!(!copy(&ram, 0, 8, |_, _, _| calls += 1));
        assert_eq!(
            calls, 1,
            "a later host overflow cannot transfer an earlier chunk"
        );
        assert_eq!(run(&ram, 7), None);
    }
}
