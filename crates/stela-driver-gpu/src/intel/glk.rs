// SPDX-License-Identifier: GPL-2.0-or-later
//! Gemini Lake PCI memory inventory and GT forcewake register definitions.
//! Configuration mirrors describe firmware allocations. They confer no
//! reservation, DMA, MMIO or scanout authority.

use stela_driver_base::bus::pci::{ConfigError, ConfigSpace, Function};

use crate::{DISPLAY_CLASS, INTEL_VENDOR};

pub mod forcewake;
pub mod ggtt;
pub mod native;

// The pinned i915 PCI ID and graphics control headers describe the
// read-only mirrors on the integrated graphics function.
pub const GRAPHICS_CONTROL: u16 = 0x50;
const STOLEN_BASE: u16 = 0x5c;
const MIB: u64 = 1 << 20;
const STOLEN_BASE_MASK: u32 = 0xfff0_0000;

/// Firmware-declared graphics memory. These physical addresses are inventory
/// data until reconciled with the host memory map.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryLayout {
    /// Size of the GGTT page-table storage, in bytes.
    pub ggtt_bytes: u64,
    pub stolen: Option<StolenMemory>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StolenMemory {
    pub base: u64,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Config(ConfigError),
    ReservedStolenSize,
    InvalidStolenRange,
}

/// Observe the memory layout of a Gemini Lake display function. Other
/// identities leave the vendor-specific registers unread.
pub fn inspect_memory<C: ConfigSpace>(
    function: &mut Function<C>,
) -> Option<Result<MemoryLayout, Error>> {
    (function.vendor == INTEL_VENDOR
        && matches!(function.device, 0x3184 | 0x3185)
        && function.class[0] == DISPLAY_CLASS)
        .then(|| read_memory(function.config()))
}

fn read_memory(config: &impl ConfigSpace) -> Result<MemoryLayout, Error> {
    let control = config.read16(GRAPHICS_CONTROL).map_err(Error::Config)?;
    let ggtt_bytes = ggtt_bytes(control);
    // Gemini Lake's datasheet 336561 §2.10 reserves GMS 0x40–0xff.
    let gms = u64::from(control >> 8);
    if gms > 0x3f {
        return Err(Error::ReservedStolenSize);
    }
    let stolen_bytes = gms * 32 * MIB;
    let stolen = if stolen_bytes == 0 {
        None
    } else {
        let base = u64::from(config.read32(STOLEN_BASE).map_err(Error::Config)? & STOLEN_BASE_MASK);
        if base == 0 || base + stolen_bytes > 1 << 32 {
            return Err(Error::InvalidStolenRange);
        }
        Some(StolenMemory {
            base,
            bytes: stolen_bytes,
        })
    };
    Ok(MemoryLayout { ggtt_bytes, stolen })
}

fn ggtt_bytes(control: u16) -> u64 {
    match (control >> 6) & 0x3 {
        0 => 0,
        order => MIB << order,
    }
}

/// Gen9 LP GT domains on GLK. Offsets are relative to the register BAR,
/// and must be used through an owned uncore mapping after attachment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForcewakeDomain {
    Render,
    Gt,
    Media,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForcewakeRegisters {
    pub request: usize,
    pub acknowledge: usize,
}

impl ForcewakeDomain {
    /// i915's Gen9/10 domain pairs from intel_uncore.c and intel_gt_regs.h.
    pub const fn registers(self) -> ForcewakeRegisters {
        let (request, acknowledge) = match self {
            Self::Render => (0xa278, 0xd84),
            Self::Gt => (0xa188, 0x130044),
            Self::Media => (0xa270, 0xd88),
        };
        ForcewakeRegisters {
            request,
            acknowledge,
        }
    }
}

#[cfg(test)]
mod tests;
