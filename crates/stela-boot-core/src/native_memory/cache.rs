//! Read-only validation of the native heap's effective WB cache policy.
//! This does not change PAT/MTRRs or authorize aliases with another policy.
use core::ops::Range;
use stela_drivers::bus::dma::memory::MemoryError;

pub const WRITE_BACK: u8 = 6;
const FIXED_TOP: u64 = 1024 * 1024;
pub const ADDRESS: u64 = 0x000f_ffff_ffff_f000;
const PRESENT: u64 = 1;
const WRITABLE: u64 = 1 << 1;
const LARGE: u64 = 1 << 7;
#[cfg(test)]
const NX: u64 = 1 << 63;

/// Prove each current leaf maps the expected physical pages with a WB
/// PAT selection. Intermediate cache bits affect table fetches, not data PAT.
pub fn verify_mappings(
    cpu: u64,
    physical: u64,
    bytes: usize,
    root: u64,
    levels: u32,
    pat: u64,
    mut read: impl FnMut(u64, usize) -> Option<u64>,
) -> Result<(), MemoryError> {
    if !matches!(levels, 4 | 5) || bytes == 0 {
        return Err(MemoryError::CacheAttributeUnavailable);
    }
    let mut offset = 0u64;
    while offset < bytes as u64 {
        let address = cpu.checked_add(offset).ok_or(MemoryError::InvalidSize)?;
        let canonical_bits = 12 + 9 * levels;
        if ((address << (64 - canonical_bits)) as i64 >> (64 - canonical_bits)) as u64 != address {
            return Err(MemoryError::OutsideReservation);
        }
        let mut table = root;
        let mut writable = true;
        for level in (0..levels).rev() {
            let index = ((address >> (12 + 9 * level)) & 511) as usize;
            let entry = read(table, index).ok_or(MemoryError::OutsideReservation)?;
            if entry & PRESENT == 0 {
                return Err(MemoryError::OutsideReservation);
            }
            writable &= entry & WRITABLE != 0;
            let large = entry & LARGE != 0;
            if level == 0 || (large && matches!(level, 1 | 2)) {
                let size = 1u64 << (12 + 9 * level);
                let pat_bit = if level == 0 { 7 } else { 12 };
                let index = ((entry >> 3) & 3) | ((entry >> pat_bit) & 1) << 2;
                let cache_type = (pat >> (index * 8)) as u8;
                let leaf_offset = address & (size - 1);
                let base = entry & ADDRESS & !(size - 1);
                if !writable
                    || cache_type != WRITE_BACK
                    || base.checked_add(leaf_offset) != physical.checked_add(offset)
                    || (level != 0 && entry & ADDRESS & (size - 1) & !(1 << 12) != 0)
                {
                    return Err(MemoryError::CacheAttributeUnavailable);
                }
                offset += (size - leaf_offset).min(bytes as u64 - offset);
                break;
            }
            if large {
                return Err(MemoryError::OutsideReservation);
            }
            table = entry & ADDRESS;
        }
    }
    Ok(())
}

/// One enabled variable-range MTRR.
#[derive(Clone)]
pub struct Variable {
    pub range: Range<u64>,
    pub kind: u8,
}

/// The MTRR state and CR0 cache controls that decide a range's type.
pub struct Policy {
    pub enabled: bool,
    pub caches_enabled: bool,
    pub default_kind: u8,
    pub physical_bits: u8,
}

impl Policy {
    /// The whole range is write-back: no non-WB overlap and full coverage.
    /// Each clone of the iterator must read the same bounded, immutable
    /// register state. A decoding error invalidates the whole proof.
    pub fn writeback(
        &self,
        start: u64,
        bytes: usize,
        variables: impl Clone + Iterator<Item = Result<Variable, MemoryError>>,
    ) -> Result<(), MemoryError> {
        let end = start
            .checked_add(bytes as u64)
            .ok_or(MemoryError::InvalidSize)?;
        if !self.enabled
            || !self.caches_enabled
            || bytes == 0
            || start < FIXED_TOP
            || !(32..=52).contains(&self.physical_bits)
            || end > 1u64 << self.physical_bits
        {
            return Err(MemoryError::CacheAttributeUnavailable);
        }
        // Check every register even when the default is WB. Otherwise extend
        // coverage across unsorted WB ranges one endpoint at a time.
        let mut cursor = start;
        loop {
            let mut covered = cursor;
            for entry in variables.clone() {
                let entry = entry?;
                if entry.range.start >= end || entry.range.end <= start {
                    continue;
                }
                if entry.kind != WRITE_BACK {
                    return Err(MemoryError::CacheAttributeUnavailable);
                }
                if entry.range.start <= cursor {
                    covered = covered.max(entry.range.end);
                }
            }
            if self.default_kind == WRITE_BACK || covered >= end {
                return Ok(());
            }
            if covered == cursor {
                return Err(MemoryError::CacheAttributeUnavailable);
            }
            cursor = covered;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{collections::BTreeMap, vec::Vec};

    fn check_policy(
        policy: &Policy,
        start: u64,
        bytes: usize,
        variables: &[Variable],
    ) -> Result<(), MemoryError> {
        policy.writeback(start, bytes, variables.iter().cloned().map(Ok))
    }

    fn tables(
        levels: u32,
        cpu: u64,
        physical: u64,
        leaf: u32,
        attributes: u64,
    ) -> BTreeMap<(u64, usize), u64> {
        let mut tables = BTreeMap::new();
        let mut table = 0x1000;
        for level in (leaf..levels).rev() {
            let index = ((cpu >> (12 + 9 * level)) & 511) as usize;
            let value = if level == leaf {
                physical | attributes | PRESENT | WRITABLE | NX | if leaf != 0 { LARGE } else { 0 }
            } else {
                (table + 0x1000) | PRESENT | WRITABLE
            };
            tables.insert((table, index), value);
            table += 0x1000;
        }
        tables
    }

    #[test]
    fn mapping_check_rejects_wrong_pat_or_address_on_any_page() {
        for levels in [4, 5] {
            let cpu = if levels == 5 {
                0x0040_0000_0040_0000
            } else {
                0xffff_8000_0040_0000
            };
            let mut pages = tables(levels, cpu, 0x800000, 0, 0);
            let leaf_table = u64::from(levels) * 0x1000;
            let index = ((cpu >> 12) & 511) as usize;
            pages.insert((leaf_table, index + 1), 0x801000 | PRESENT | WRITABLE | NX);
            let check = |pages: &BTreeMap<_, _>, pat| {
                verify_mappings(cpu, 0x800000, 8192, 0x1000, levels, pat, |table, index| {
                    pages.get(&(table, index)).copied()
                })
            };
            assert_eq!(check(&pages, 6), Ok(()));
            assert!(check(&pages, 0).is_err());
            pages.insert((leaf_table, index + 1), 0x802000 | PRESENT | WRITABLE | NX);
            assert!(check(&pages, 6).is_err());
            // PAT bit 7 in a 4 KiB PTE selects entry 4; PAT0 WB alone is
            // insufficient when the leaf selects an uncached entry.
            pages.insert(
                (leaf_table, index),
                0x800000 | PRESENT | WRITABLE | NX | LARGE,
            );
            pages.insert(
                (leaf_table, index + 1),
                0x801000 | PRESENT | WRITABLE | NX | LARGE,
            );
            assert!(check(&pages, 6).is_err());
            assert_eq!(check(&pages, 6u64 << 32), Ok(()));
        }
    }

    #[test]
    fn large_leaf_pat_bit_and_effective_permissions_are_not_guessed() {
        for leaf in [1, 2] {
            let cpu = 0xffff_8000_4000_0000;
            let pages = tables(4, cpu, 0x80000000, leaf, 1 << 12);
            let check = |pages: &BTreeMap<_, _>, pat| {
                verify_mappings(
                    cpu + 4096,
                    0x80001000,
                    8192,
                    0x1000,
                    4,
                    pat,
                    |table, index| pages.get(&(table, index)).copied(),
                )
            };
            assert!(check(&pages, 6).is_err());
            assert_eq!(check(&pages, 6u64 << 32), Ok(()));
            let mut read_only = pages.clone();
            *read_only
                .get_mut(&(0x1000, ((cpu >> 39) & 511) as usize))
                .unwrap() &= !WRITABLE;
            assert!(check(&read_only, 6u64 << 32).is_err());
        }
    }

    #[test]
    fn effective_wb_requires_pat_mtrr_and_complete_coverage() {
        let mut policy = Policy {
            enabled: true,
            caches_enabled: true,
            default_kind: WRITE_BACK,
            physical_bits: 48,
        };
        let mut variables = Vec::new();
        let range = 0x200000..0x204000;
        assert_eq!(
            check_policy(&policy, range.start, 0x4000, &variables),
            Ok(())
        );
        let invalid = Err(MemoryError::CacheAttributeUnavailable);
        assert_eq!(
            policy.writeback(range.start, 0x4000, [invalid].into_iter()),
            Err(MemoryError::CacheAttributeUnavailable)
        );
        policy.enabled = false;
        assert!(check_policy(&policy, range.start, 0x4000, &variables).is_err());
        policy.enabled = true;
        variables.push(Variable {
            range: 0x203000..0x204000,
            kind: 0,
        });
        assert!(check_policy(&policy, range.start, 0x4000, &variables).is_err());
        variables.clear();
        policy.default_kind = 0;
        variables.push(Variable {
            range: 0x200000..0x202000,
            kind: WRITE_BACK,
        });
        assert!(check_policy(&policy, range.start, 0x4000, &variables).is_err());
        variables.push(Variable {
            range: 0x202000..0x204000,
            kind: WRITE_BACK,
        });
        assert_eq!(
            check_policy(&policy, range.start, 0x4000, &variables),
            Ok(())
        );
        assert!(check_policy(&policy, 0, 4096, &variables).is_err());
        assert!(check_policy(&policy, u64::MAX - 4095, 8192, &variables).is_err());
        assert!(check_policy(&policy, 1 << 48, 4096, &variables).is_err());
    }

    #[test]
    fn unsorted_wb_coverage_and_gaps_require_no_allocation() {
        use super::super::tests::allocation_failure;
        let policy = Policy {
            enabled: true,
            caches_enabled: true,
            default_kind: 0,
            physical_bits: 48,
        };
        let mut variables = [
            0x203000..0x205000,
            0x201000..0x204000,
            0x205000..0x206000,
            0x200000..0x202000,
        ]
        .into_iter()
        .map(|range| Variable {
            range,
            kind: WRITE_BACK,
        })
        .collect::<Vec<_>>();
        allocation_failure::arm(0);
        let covered = check_policy(&policy, 0x200000, 0x6000, &variables);
        let gap = check_policy(&policy, 0x200000, 0x7000, &variables);
        variables[1].kind = 0;
        let conflict = check_policy(&policy, 0x200000, 0x6000, &variables);
        let observation = allocation_failure::disarm();
        assert_eq!(covered, Ok(()));
        assert_eq!(gap, Err(MemoryError::CacheAttributeUnavailable));
        assert_eq!(conflict, Err(MemoryError::CacheAttributeUnavailable));
        assert_eq!(observation.attempts, 0);
    }

    /// Intel desktop layouts: WB default with a UC PCI hole (common on
    /// Alder/Raptor Lake boards), and a UC default whose WB ranges stop
    /// short of the top of a 96 GiB memory (DDR5 48 GiB modules).
    #[test]
    fn intel_desktop_mtrr_layouts_decide_exactly() {
        let hole = Policy {
            enabled: true,
            caches_enabled: true,
            default_kind: WRITE_BACK,
            physical_bits: 39,
        };
        let uc = |range: Range<u64>| Variable { range, kind: 0 };
        let wb = |range: Range<u64>| Variable {
            range,
            kind: WRITE_BACK,
        };
        let pci = [uc(0x8000_0000..0x1_0000_0000)];
        assert!(check_policy(&hole, 0x1_0000_0000, 32 << 20, &pci).is_ok());
        assert!(check_policy(&hole, 0x7000_0000, 32 << 20, &pci).is_ok());
        assert!(check_policy(&hole, 0x7ff0_0000, 2 << 20, &pci).is_err());
        let short = Policy {
            default_kind: 0,
            ..hole
        };
        let cover = [
            wb(0..64 << 30),
            wb(64 << 30..80 << 30),
            uc(0x8000_0000..0x1_0000_0000),
        ];
        assert!(check_policy(&short, 40 << 30, 32 << 20, &cover).is_ok());
        assert!(check_policy(&short, (64 << 30) - (16 << 20), 32 << 20, &cover).is_ok());
        assert!(check_policy(&short, 88 << 30, 32 << 20, &cover).is_err());
        assert!(check_policy(&short, (80 << 30) - (16 << 20), 32 << 20, &cover).is_err());
        // WT over WB is WT, never WB.
        let mixed = [
            wb(0..64 << 30),
            Variable {
                range: 8 << 30..9 << 30,
                kind: 4,
            },
        ];
        assert!(check_policy(&short, 8 << 30, 4096, &mixed).is_err());
    }
}
