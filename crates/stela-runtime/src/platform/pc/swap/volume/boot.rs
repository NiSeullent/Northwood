// SPDX-License-Identifier: GPL-2.0-or-later
//! Identify the DATA sibling on the disk the firmware booted from.

use super::BASIC_DATA;
use crate::pc::gpt::{self, Guid, Partition};
use alloc::boxed::Box;
use core::fmt;
use stela_drivers::block::BlockDevice;

/// GPT signature and extent from the firmware's loaded image device path.
#[derive(Clone, Copy)]
pub struct BootEsp {
    guid: Guid,
    first_lba: u64,
    last_lba: u64,
}

impl BootEsp {
    pub fn new(guid: Guid, first_lba: u64, blocks: u64) -> Option<Self> {
        if guid.is_zero() {
            return None;
        }
        Some(Self {
            guid,
            first_lba,
            last_lba: first_lba.checked_add(blocks.checked_sub(1)?)?,
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum BootDataError {
    MissingEsp,
    AmbiguousEsp,
    InvalidEsp,
    MissingData,
    AmbiguousData,
    InvalidData,
}

impl fmt::Display for BootDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::MissingEsp => "boot ESP not found on the probed disks",
            Self::AmbiguousEsp => "duplicate boot ESP GUID",
            Self::InvalidEsp => "boot ESP type or extent disagrees with the firmware",
            Self::MissingData => "boot disk has no basic data partition",
            Self::AmbiguousData => "boot disk has several basic data partitions",
            Self::InvalidData => {
                "boot data partition has a zero GUID, duplicate GUID, or overlapping extent"
            }
        })
    }
}

/// Locate a single, disjoint DATA partition on the GPT containing the boot
/// ESP. Return the disk index so the caller can read directly from that disk
/// even when another disk has a matching DATA GUID.
pub fn boot_data(
    disks: &mut [Box<dyn BlockDevice>],
    esp: BootEsp,
) -> Result<(usize, Partition), BootDataError> {
    let mut boot = None;
    for (index, disk) in disks.iter_mut().enumerate() {
        let Ok(table) = gpt::read(&mut **disk) else {
            continue;
        };
        let mut matches = table
            .partitions
            .iter()
            .filter(|partition| partition.unique_guid == esp.guid);
        let Some(partition) = matches.next() else {
            continue;
        };
        if matches.next().is_some() || boot.is_some() {
            return Err(BootDataError::AmbiguousEsp);
        }
        if partition.type_guid != Guid::ESP
            || partition.first_lba != esp.first_lba
            || partition.last_lba != esp.last_lba
        {
            return Err(BootDataError::InvalidEsp);
        }
        boot = Some((index, table));
    }

    let (index, table) = boot.ok_or(BootDataError::MissingEsp)?;
    let mut candidates = table
        .partitions
        .iter()
        .filter(|partition| partition.type_guid == BASIC_DATA);
    let data = candidates.next().ok_or(BootDataError::MissingData)?;
    if candidates.next().is_some() {
        return Err(BootDataError::AmbiguousData);
    }
    if data.unique_guid.is_zero()
        || table.partitions.iter().any(|other| {
            !core::ptr::eq(other, data)
                && (other.unique_guid == data.unique_guid
                    || (data.first_lba <= other.last_lba && other.first_lba <= data.last_lba))
        })
    {
        return Err(BootDataError::InvalidData);
    }
    Ok((index, data.clone()))
}

#[cfg(test)]
mod tests;
