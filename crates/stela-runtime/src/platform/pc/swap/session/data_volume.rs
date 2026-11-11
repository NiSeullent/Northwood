// SPDX-License-Identifier: GPL-2.0-or-later
//! Qualify the FAT32 placement under the session's whole-disk owner.

use super::{Error, ExclusiveDisk, check_disk, read_blocks};
use crate::pc::{
    gpt::{self, Guid, Partition, PartitionDevice},
    swap::volume::{self, BASIC_DATA, VerifiedLayout},
};
use stela_drivers::block::{BlockDevice, BlockError, Identity};
use stela_formats::payload::ReadAt;

pub(super) fn check_features(identity: &Identity) -> Result<(), Error> {
    let block = identity.block_size as usize;
    let features = identity.features;
    if identity.read_only {
        return Err(Error::Device(BlockError::ReadOnly));
    }
    if !matches!(block, 512 | 1024 | 2048 | 4096)
        || identity.blocks.checked_mul(block as u64).is_none()
        || (features.max_transfer != 0 && features.max_transfer < block)
        || features
            .max_read_transfer
            .is_some_and(|limit| limit.get() < block)
    {
        return Err(Error::Device(BlockError::Unsupported(
            "invalid FAT32 swap block geometry or transfer limits",
        )));
    }
    Ok(())
}

pub(super) fn partition(disk: &mut dyn BlockDevice, uuid: Guid) -> Result<Partition, Error> {
    if uuid.is_zero() {
        return Err(Error::DataPartition);
    }
    let table = gpt::read_for_write(disk).map_err(Error::DataTable)?;
    let data = table
        .partitions
        .iter()
        .find(|entry| entry.unique_guid == uuid)
        .ok_or(Error::DataPartition)?;
    if data.type_guid != BASIC_DATA
        || table.partitions.iter().any(|other| {
            !core::ptr::eq(other, data)
                && (other.unique_guid == uuid
                    || (data.first_lba <= other.last_lba && other.first_lba <= data.last_lba))
        })
    {
        return Err(Error::DataPartition);
    }
    Ok(data.clone())
}

pub(super) fn verify_layout<D: ExclusiveDisk>(
    disk: &mut PartitionDevice<D>,
    expected: &Identity,
    maximum_bytes: u64,
) -> Result<VerifiedLayout, Error> {
    let block = u64::from(expected.block_size);
    let reader = PartitionReader {
        disk,
        expected,
        block,
    };
    let layout = volume::verify_layout(reader, block, maximum_bytes).map_err(Error::Volume)?;
    check_disk(disk.disk(), expected)?;
    Ok(layout)
}

/// Metadata reads can start or stop within a logical block. Use the block
/// planner for both aligned reads and the one-block boundary reads.
struct PartitionReader<'a, D> {
    disk: &'a mut PartitionDevice<D>,
    expected: &'a Identity,
    block: u64,
}

impl<D: ExclusiveDisk> ReadAt for PartitionReader<'_, D> {
    type Error = Error;

    fn len(&self) -> u64 {
        self.disk.identity().bytes()
    }

    fn read_at(&mut self, mut offset: u64, mut output: &mut [u8]) -> Result<(), Error> {
        if offset
            .checked_add(output.len() as u64)
            .is_none_or(|end| end > self.len())
        {
            return Err(Error::Device(BlockError::Protocol(
                "FAT32 read exceeds DATA",
            )));
        }
        while !output.is_empty() {
            let within = (offset % self.block) as usize;
            let block = self.block as usize;
            let aligned = output.len() / block * block;
            let (read, take) = if within == 0 && aligned != 0 {
                (aligned, aligned)
            } else {
                (block, output.len().min(block - within))
            };
            let data = read_blocks(self.disk, self.expected, offset / self.block, read)?;
            let (chunk, rest) = output.split_at_mut(take);
            chunk.copy_from_slice(&data[within..within + take]);
            offset += take as u64;
            output = rest;
        }
        Ok(())
    }
}
