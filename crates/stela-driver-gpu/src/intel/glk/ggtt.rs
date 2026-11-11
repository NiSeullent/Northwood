// SPDX-License-Identifier: GPL-2.0-or-later
//! Read-only geometry for Gemini Lake's GGTT page-table window.

use core::ops::Range;
use stela_driver_base::bus::pci::Bar;

use super::{MIB, MemoryLayout, ggtt_bytes};

// Pinned i915 cc9de5d4, intel_ggtt.c: gen6_gttmmadr_size,
// gen6_gttadr_offset and gen8_gmch_probe; intel_gtt.h: gen8_pte_t.
const BAR0_BYTES: u64 = 16 * MIB;
const TABLE_OFFSET: u64 = 8 * MIB;
const PTE_BYTES: u64 = size_of::<u64>() as u64;
const GPU_PAGE_BYTES: u64 = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    InvalidBar0,
    InvalidGraphicsControl,
    InvalidTableSize,
}

/// The GGTT window in PCI bus address space, before any host translation.
/// This geometry confers no permission to map or access the table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Geometry {
    pci_start: u64,
    table_bytes: u64,
}

impl Geometry {
    /// Use the BAR0 descriptor and graphics control inventory of the same
    /// Gemini Lake function. A valid BAR0 with GGMS zero has no table window.
    pub fn from_bar0(bar: Bar, layout: MemoryLayout) -> Result<Option<Self>, Error> {
        Self::from_size(bar, layout.ggtt_bytes)
    }

    /// Read only the GGMS field. The stolen-memory base is independent of
    /// the page-table window and need not be read for this observation.
    pub fn from_control(bar: Bar, control: u16) -> Result<Option<Self>, Error> {
        if control == u16::MAX {
            return Err(Error::InvalidGraphicsControl);
        }
        Self::from_size(bar, ggtt_bytes(control))
    }

    fn from_size(bar: Bar, table_bytes: u64) -> Result<Option<Self>, Error> {
        let Bar::Memory {
            address,
            size: BAR0_BYTES,
            prefetchable: false,
            wide,
        } = bar
        else {
            return Err(Error::InvalidBar0);
        };
        let range = bar.memory().ok_or(Error::InvalidBar0)?;
        if address == 0 || !address.is_multiple_of(BAR0_BYTES) || (!wide && range.end > 1 << 32) {
            return Err(Error::InvalidBar0);
        }

        if table_bytes == 0 {
            return Ok(None);
        }
        if !table_bytes.is_power_of_two() || !(2 * MIB..=8 * MIB).contains(&table_bytes) {
            return Err(Error::InvalidTableSize);
        }
        Ok(Some(Self {
            pci_start: address + TABLE_OFFSET,
            table_bytes,
        }))
    }

    pub fn pci_range(self) -> Range<u64> {
        self.pci_start..self.pci_start + self.table_bytes
    }

    pub fn entry_count(self) -> u64 {
        self.table_bytes / PTE_BYTES
    }

    pub fn address_space_bytes(self) -> u64 {
        self.entry_count() * GPU_PAGE_BYTES
    }

    /// Byte offset from the start of the table for one aligned GPU page.
    pub fn entry_offset(self, gpu_page: u64) -> Option<u64> {
        if !gpu_page.is_multiple_of(GPU_PAGE_BYTES) || gpu_page >= self.address_space_bytes() {
            return None;
        }
        Some(gpu_page / GPU_PAGE_BYTES * PTE_BYTES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intel::glk::StolenMemory;

    const BASE: u64 = 0x1_c000_0000;

    fn bar(address: u64, size: u64, prefetchable: bool, wide: bool) -> Bar {
        Bar::Memory {
            address,
            size,
            prefetchable,
            wide,
        }
    }

    fn memory(ggtt_bytes: u64) -> MemoryLayout {
        MemoryLayout {
            ggtt_bytes,
            stolen: Some(StolenMemory {
                base: 0x9be0_0000,
                bytes: 64 * MIB,
            }),
        }
    }

    #[test]
    fn reads_only_ggms_from_available_graphics_control() {
        let bar = bar(BASE, BAR0_BYTES, false, true);
        for (ggms, bytes) in [(0, 0), (1, 2 * MIB), (2, 4 * MIB), (3, 8 * MIB)] {
            assert_eq!(
                Geometry::from_control(bar, 0x3f00 | ggms << 6),
                Geometry::from_bar0(bar, memory(bytes)),
            );
        }
        assert_eq!(
            Geometry::from_control(bar, u16::MAX),
            Err(Error::InvalidGraphicsControl),
        );
    }

    #[test]
    fn four_mib_table_covers_two_gib_of_gpu_pages() {
        let ggtt = Geometry::from_bar0(bar(BASE, BAR0_BYTES, false, true), memory(4 * MIB))
            .unwrap()
            .unwrap();
        assert_eq!(ggtt.pci_range(), BASE + 8 * MIB..BASE + 12 * MIB);
        assert_eq!(ggtt.entry_count(), 524_288);
        assert_eq!(ggtt.address_space_bytes(), 2 << 30);
        assert_eq!(ggtt.entry_offset(0), Some(0));
        assert_eq!(ggtt.entry_offset(GPU_PAGE_BYTES), Some(8));
        assert_eq!(
            ggtt.entry_offset((2 << 30) - GPU_PAGE_BYTES),
            Some(4 * MIB - 8)
        );
        assert_eq!(ggtt.entry_offset(1), None);
        assert_eq!(ggtt.entry_offset(2 << 30), None);
        assert_eq!(ggtt.entry_offset(u64::MAX), None);
    }

    #[test]
    fn decoded_sizes_fit_at_the_top_of_a_32_bit_bar() {
        let base = (1 << 32) - BAR0_BYTES;
        let bar = bar(base, BAR0_BYTES, false, false);
        assert_eq!(Geometry::from_bar0(bar, memory(0)), Ok(None));
        for (table_mib, gpu_mib) in [(2, 1024), (8, 4096)] {
            let ggtt = Geometry::from_bar0(bar, memory(table_mib * MIB))
                .unwrap()
                .unwrap();
            assert_eq!(
                ggtt.pci_range(),
                base + 8 * MIB..base + (8 + table_mib) * MIB
            );
            assert_eq!(ggtt.address_space_bytes(), gpu_mib * MIB);
        }
    }

    #[test]
    fn invalid_bars_and_table_sizes_are_refused() {
        for bad in [
            Bar::Io {
                port: 0x1000,
                size: 4,
            },
            bar(0, BAR0_BYTES, false, true),
            bar(BASE + GPU_PAGE_BYTES, BAR0_BYTES, false, true),
            bar(BASE, BAR0_BYTES / 2, false, true),
            bar(BASE, BAR0_BYTES * 2, false, true),
            bar(BASE, BAR0_BYTES, true, true),
            bar(BASE, BAR0_BYTES, false, false),
            bar(u64::MAX & !(BAR0_BYTES - 1), BAR0_BYTES, false, true),
        ] {
            assert_eq!(
                Geometry::from_bar0(bad, memory(4 * MIB)),
                Err(Error::InvalidBar0)
            );
        }
        let bar = bar(BASE, BAR0_BYTES, false, true);
        for bad in [1, MIB, 3 * MIB, 16 * MIB] {
            assert_eq!(
                Geometry::from_bar0(bar, memory(bad)),
                Err(Error::InvalidTableSize)
            );
        }
    }
}
