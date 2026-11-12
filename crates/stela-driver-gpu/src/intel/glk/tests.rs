// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;
use core::cell::Cell;
use stela_driver_base::bus::pci::{Address, ConfigFailure};

use crate::Adapter;

struct Config {
    identity: u32,
    class: u8,
    control: Result<u16, ConfigError>,
    base: Result<u32, ConfigError>,
    control_reads: Cell<usize>,
    base_reads: Cell<usize>,
}

impl ConfigSpace for Config {
    fn read32(&self, offset: u16) -> Result<u32, ConfigError> {
        match offset {
            0x00 => Ok(self.identity),
            0x08 => Ok(u32::from(self.class) << 24),
            STOLEN_BASE => {
                self.base_reads.set(self.base_reads.get() + 1);
                self.base
            }
            _ => panic!("unexpected graphics configuration read at {offset:#x}"),
        }
    }

    fn read16(&self, offset: u16) -> Result<u16, ConfigError> {
        assert_eq!(offset, GRAPHICS_CONTROL);
        self.control_reads.set(self.control_reads.get() + 1);
        self.control
    }

    fn read8(&self, offset: u16) -> Result<u8, ConfigError> {
        panic!("unexpected graphics configuration read at {offset:#x}");
    }

    fn write32(&mut self, _: u16, _: u32) -> Result<(), ConfigError> {
        panic!("graphics inventory wrote a dword");
    }

    fn write16(&mut self, _: u16, _: u16) -> Result<(), ConfigError> {
        panic!("graphics inventory wrote a word");
    }

    fn write8(&mut self, _: u16, _: u8) -> Result<(), ConfigError> {
        panic!("graphics inventory wrote a byte");
    }
}

fn function(vendor: u16, device: u16, class: u8, control: u16, base: u32) -> Function<Config> {
    Function::new(
        Address {
            segment: 0,
            bus: 0,
            device: 2,
            function: 0,
        },
        Config {
            identity: (u32::from(device) << 16) | u32::from(vendor),
            class,
            control: Ok(control),
            base: Ok(base),
            control_reads: Cell::new(0),
            base_reads: Cell::new(0),
        },
    )
    .unwrap()
}

fn glk(device: u16, control: u16, base: u32) -> Function<Config> {
    function(INTEL_VENDOR, device, DISPLAY_CLASS, control, base)
}

#[test]
fn display_inventory_reads_both_gemini_lake_ids() {
    for id in [0x3184, 0x3185] {
        let mut pci = glk(id, 0x0280, 0x9be0_0001);
        assert_eq!(Adapter::discover(&pci).unwrap().glk_memory, None);
        assert_eq!(pci.config().control_reads.get(), 0);

        let adapter = Adapter::inspect(&mut pci).unwrap();
        assert_eq!(adapter.device, id);
        assert_eq!(
            adapter.glk_memory,
            Some(Ok(MemoryLayout {
                ggtt_bytes: 4 * MIB,
                stolen: Some(StolenMemory {
                    base: 0x9be0_0000,
                    bytes: 64 * MIB,
                }),
            }))
        );
        assert_eq!(pci.config().control_reads.get(), 1);
        assert_eq!(pci.config().base_reads.get(), 1);
    }
}

#[test]
fn unsupported_identities_keep_inventory_without_vendor_reads() {
    for (vendor, id, class) in [
        (INTEL_VENDOR, 0x5a85, DISPLAY_CLASS), // Apollo Lake
        (INTEL_VENDOR, 0x1912, DISPLAY_CLASS), // Skylake
        (INTEL_VENDOR, 0x46a6, DISPLAY_CLASS), // Alder Lake
        (0x10de, 0x3185, DISPLAY_CLASS),
        (INTEL_VENDOR, 0x3185, 0x02),
    ] {
        let mut pci = function(vendor, id, class, u16::MAX, u32::MAX);
        let adapter = Adapter::inspect(&mut pci);
        assert_eq!(adapter.is_some(), class == DISPLAY_CLASS);
        assert_eq!(adapter.and_then(|adapter| adapter.glk_memory), None);
        assert_eq!(pci.config().control_reads.get(), 0);
        assert_eq!(pci.config().base_reads.get(), 0);
    }
}

#[test]
fn zero_stolen_memory_skips_the_base_and_decodes_each_ggtt_size() {
    for (ggms, mib) in [(0, 0), (1, 2), (2, 4), (3, 8)] {
        let mut pci = glk(0x3185, ggms << 6, u32::MAX);
        assert_eq!(
            inspect_memory(&mut pci),
            Some(Ok(MemoryLayout {
                ggtt_bytes: mib * MIB,
                stolen: None,
            }))
        );
        assert_eq!(pci.config().base_reads.get(), 0);
    }
}

#[test]
fn stolen_size_boundaries_and_upper_physical_boundary_are_observed() {
    for (gms, mib) in [(1, 32), (0x10, 512), (0x3f, 2016)] {
        let mut pci = glk(0x3185, gms << 8, 0x4000_0000);
        assert_eq!(
            inspect_memory(&mut pci).unwrap().unwrap().stolen,
            Some(StolenMemory {
                base: 0x4000_0000,
                bytes: mib * MIB,
            })
        );
    }
    let mut pci = glk(0x3185, 0x0100, 0xfe00_0000);
    assert!(inspect_memory(&mut pci).unwrap().is_ok());
}

#[test]
fn invalid_or_denied_memory_observations_preserve_the_adapter() {
    for control in [0x4000, 0xf000, u16::MAX] {
        let mut pci = glk(0x3185, control, 0x9000_0000);
        assert_eq!(
            Adapter::inspect(&mut pci).unwrap().glk_memory,
            Some(Err(Error::ReservedStolenSize))
        );
        assert_eq!(pci.config().base_reads.get(), 0);
    }

    for base in [0, 0x800, 0xfe10_0000, u32::MAX] {
        let mut pci = glk(0x3185, 0x0100, base);
        assert_eq!(
            inspect_memory(&mut pci),
            Some(Err(Error::InvalidStolenRange))
        );
    }

    let error = ConfigError::before_effect(ConfigFailure::Access {
        offset: GRAPHICS_CONTROL,
        width: 2,
    });
    let mut pci = glk(0x3185, 0x0100, 0x9000_0000);
    pci.config().control = Err(error);
    assert_eq!(
        Adapter::inspect(&mut pci).unwrap().glk_memory,
        Some(Err(Error::Config(error)))
    );
    assert_eq!(pci.config().base_reads.get(), 0);
    let error = ConfigError::before_effect(ConfigFailure::Access {
        offset: STOLEN_BASE,
        width: 4,
    });
    pci.config().control = Ok(0x0100);
    pci.config().base = Err(error);
    assert_eq!(inspect_memory(&mut pci), Some(Err(Error::Config(error))));
}

#[test]
fn glk_forcewake_keeps_the_three_gen9_lp_register_pairs() {
    for (domain, request, acknowledge) in [
        (ForcewakeDomain::Render, 0xa278, 0xd84),
        (ForcewakeDomain::Gt, 0xa188, 0x130044),
        (ForcewakeDomain::Media, 0xa270, 0xd88),
    ] {
        assert_eq!(
            domain.registers(),
            ForcewakeRegisters {
                request,
                acknowledge,
            }
        );
    }
}
