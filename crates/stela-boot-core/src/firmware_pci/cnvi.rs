// SPDX-License-Identifier: GPL-2.0-or-later

//! Read-only observations for the firmware handle of the original 31dc.
use super::{BarRegion, observe_memory_bar};

const BAR_BYTES: u64 = 16 * 1024;
const PM_CAPABILITY: u8 = 1;

pub fn matches(header: &[u8; 64]) -> bool {
    header[..4] == [0x86, 0x80, 0xdc, 0x31] && header[9..12] == [0, 0x80, 2]
}

pub fn bar0<'a>(header: &[u8; 64], regions: &'a [BarRegion]) -> Option<&'a BarRegion> {
    let word = |offset| u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    let (base, bits) = observe_memory_bar(word(0x0c), word(0x10), word(0x14), 4096)?;
    let [region] = regions else { return None };
    (region.resource_type == 0
        && region.granularity == bits
        && region.pci.start == base
        && region.pci.end.checked_sub(base)? >= BAR_BYTES
        && region.host.end.checked_sub(region.host.start)? >= BAR_BYTES)
        .then_some(region)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerError {
    Missing,
    Invalid,
    ReadFailed,
}

/// Find and read the conventional PCI PMCSR through the same firmware handle.
pub fn power_state(
    header: &[u8; 64],
    mut read16: impl FnMut(u8) -> Option<u16>,
) -> Result<u8, PowerError> {
    if header[6] & 0x10 == 0 {
        return Err(PowerError::Missing);
    }
    let mut offset = header[0x34];
    let mut visited = 0u64;
    while offset != 0 {
        if !(0x40..=0xfc).contains(&offset) || offset & 3 != 0 {
            return Err(PowerError::Invalid);
        }
        let bit = 1u64 << (offset >> 2);
        if visited & bit != 0 {
            return Err(PowerError::Invalid);
        }
        visited |= bit;
        let entry = read16(offset)
            .filter(|&word| word != u16::MAX)
            .ok_or(PowerError::ReadFailed)?;
        if entry as u8 == PM_CAPABILITY {
            let pmcsr = offset.checked_add(4).ok_or(PowerError::Invalid)?;
            return read16(pmcsr)
                .filter(|&word| word != u16::MAX)
                .map(|word| (word & 3) as u8)
                .ok_or(PowerError::ReadFailed);
        }
        offset = (entry >> 8) as u8;
    }
    Err(PowerError::Missing)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cnvi_d0_requires_readable_pmcsr_on_the_same_bounded_capability_chain() {
        let mut header = [0; 64];
        assert_eq!(power_state(&header, |_| None), Err(PowerError::Missing));
        header[6] = 0x10;
        header[0x34] = 0x40;
        let config = |offset| match offset {
            0x40 => Some(0x5010),
            0x50 => Some(1),
            0x54 => Some(0x0100),
            _ => None,
        };
        assert_eq!(power_state(&header, config), Ok(0));
        assert_eq!(
            power_state(&header, |offset| match offset {
                0x54 => Some(3),
                _ => config(offset),
            }),
            Ok(3)
        );
        assert_eq!(
            power_state(&header, |_| Some(0x4010)),
            Err(PowerError::Invalid)
        );
        assert_eq!(
            power_state(&header, |_| Some(u16::MAX)),
            Err(PowerError::ReadFailed)
        );
    }
}
