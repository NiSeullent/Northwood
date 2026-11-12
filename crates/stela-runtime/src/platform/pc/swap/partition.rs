// SPDX-License-Identifier: GPL-2.0-or-later
//! Select an existing swap partition, or plan one in an explicitly usable
//! GPT range. These operations never issue a write.

use super::{Error as FormatError, HEADER_BYTES, Header};
use crate::pc::gpt::{
    self, GptError, Guid, NORTHWOOD_SWAP_NAME, NORTHWOOD_SWAP_TYPE, Partition, Table,
};
use alloc::{collections::BTreeSet, vec::Vec};
use core::{fmt, ops::RangeInclusive};
use stela_drivers::block::{BlockDevice, BlockError, Request};
use stela_formats::payload::gpt::ALIGNMENT;

#[derive(Debug)]
pub enum Error {
    Geometry,
    Existing,
    NoSpace,
    Missing,
    Ambiguous,
    WrongType,
    Table(GptError),
    Device(BlockError),
    Format(FormatError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry => f.write_str("invalid swap partition or GPT geometry"),
            Self::Existing => f.write_str("a swap partition or the proposed GUID already exists"),
            Self::NoSpace => f.write_str("no aligned GPT gap fits the requested swap partition"),
            Self::Missing => f.write_str("no matching Northwood Swap partition"),
            Self::Ambiguous => f.write_str("several partitions match the swap selection"),
            Self::WrongType => f.write_str("the selected partition is not typed as Northwood Swap"),
            Self::Table(error) => error.fmt(f),
            Self::Device(error) => error.fmt(f),
            Self::Format(error) => error.fmt(f),
        }
    }
}

/// A new GPT entry and its matching, idle header. The setup caller owns
/// the partition-table transaction and obtains the disk's usable range
/// from the selected GPT or from the table it is about to create.
#[derive(Debug)]
pub struct Plan {
    pub partition: Partition,
    pub header: Header,
}

/// Find a gap at a 1 MiB boundary, without moving existing partitions.
/// `usable` is inclusive and excludes all GPT metadata and any area setup
/// intends to reserve separately. `bytes` includes the swap metadata.
pub fn plan(
    table: &Table,
    usable: RangeInclusive<u64>,
    block_bytes: u64,
    bytes: u64,
    uuid: Guid,
) -> Result<Plan, Error> {
    if usable.is_empty() || !valid_block_size(block_bytes) {
        return Err(Error::Geometry);
    }
    let end = usable.end().checked_add(1).ok_or(Error::Geometry)?;
    let blocks = bytes.div_ceil(block_bytes);
    let capacity = blocks.checked_mul(block_bytes).ok_or(Error::Geometry)?;
    let header = Header::new(capacity, uuid).map_err(Error::Format)?;
    if table
        .partitions
        .iter()
        .any(|entry| entry.unique_guid == uuid || entry.type_guid == NORTHWOOD_SWAP_TYPE)
        || table.disk_guid == uuid
    {
        return Err(Error::Existing);
    }
    let partitions = ordered(table, *usable.start(), end)?;
    let mut cursor = *usable.start();
    let alignment = ALIGNMENT / block_bytes;
    for entry in partitions
        .into_iter()
        .map(Some)
        .chain(core::iter::once(None))
    {
        let gap_end = entry.map_or(end, |entry| entry.first_lba);
        let first = cursor.div_ceil(alignment).checked_mul(alignment);
        if let Some((first, next)) = first
            .and_then(|first| first.checked_add(blocks).map(|next| (first, next)))
            .filter(|(_, next)| *next <= gap_end)
        {
            return Ok(Plan {
                partition: Partition {
                    type_guid: NORTHWOOD_SWAP_TYPE,
                    unique_guid: uuid,
                    first_lba: first,
                    last_lba: next - 1,
                    attributes: 0,
                    name: NORTHWOOD_SWAP_NAME.into(),
                },
                header,
            });
        }
        if let Some(entry) = entry {
            cursor = entry.last_lba + 1;
        }
    }
    Err(Error::NoSpace)
}

/// Read the header of the exact partition on `disk`. When no UUID was
/// specified, exactly one partition of the swap type must exist. The
/// type and header UUID both have to match before callers can use it.
pub fn inspect(
    disk: &mut dyn BlockDevice,
    uuid: Option<Guid>,
) -> Result<(Partition, Header), Error> {
    inspect_table(disk, gpt::read, uuid)
}

pub(super) fn inspect_for_write(
    disk: &mut dyn BlockDevice,
    uuid: Guid,
) -> Result<(Partition, Header), Error> {
    inspect_table(disk, gpt::read_for_write, Some(uuid))
}

fn inspect_table(
    disk: &mut dyn BlockDevice,
    read_table: impl FnOnce(&mut dyn BlockDevice) -> Result<Table, GptError>,
    uuid: Option<Guid>,
) -> Result<(Partition, Header), Error> {
    let block_bytes = u64::from(disk.identity().block_size);
    if !valid_block_size(block_bytes) {
        return Err(Error::Geometry);
    }
    let table = read_table(disk).map_err(Error::Table)?;
    let mut matches = table.partitions.iter().filter(|entry| {
        uuid.map_or(entry.type_guid == NORTHWOOD_SWAP_TYPE, |uuid| {
            entry.unique_guid == uuid
        })
    });
    let entry = matches.next().ok_or(Error::Missing)?;
    if matches.next().is_some() {
        return Err(Error::Ambiguous);
    }
    if entry.type_guid != NORTHWOOD_SWAP_TYPE {
        return Err(Error::WrongType);
    }
    ordered(&table, 0, disk.identity().blocks)?;
    let header_blocks = HEADER_BYTES as u64 / block_bytes;
    if entry.blocks() < header_blocks || entry.last_lba >= disk.identity().blocks {
        return Err(Error::Geometry);
    }
    let bytes = Request::Read {
        lba: entry.first_lba,
        bytes: HEADER_BYTES,
    }
    .run(disk)
    .map_err(Error::Device)?
    .ok_or(Error::Device(BlockError::Protocol(
        "swap header read returned no bytes",
    )))?;
    let header = Header::decode_partition(&bytes, entry, block_bytes).map_err(Error::Format)?;
    Ok((entry.clone(), header))
}

fn valid_block_size(block_bytes: u64) -> bool {
    matches!(block_bytes, 512 | 1024 | 2048 | 4096)
}

/// GPT parsing checks individual partition bounds. Check the relationships
/// between entries before using a gap or accepting a swap candidate.
fn ordered(table: &Table, first: u64, end: u64) -> Result<Vec<&Partition>, Error> {
    let mut partitions: Vec<&Partition> = table.partitions.iter().collect();
    partitions.sort_unstable_by_key(|entry| entry.first_lba);
    let mut cursor = first;
    let mut ids = BTreeSet::new();
    for entry in &partitions {
        if entry.first_lba < cursor
            || entry.last_lba < entry.first_lba
            || entry.last_lba >= end
            || entry.type_guid.is_zero()
            || entry.unique_guid.is_zero()
            || !ids.insert(entry.unique_guid)
        {
            return Err(Error::Geometry);
        }
        cursor = entry.last_lba + 1;
    }
    Ok(partitions)
}

#[cfg(test)]
mod tests;
