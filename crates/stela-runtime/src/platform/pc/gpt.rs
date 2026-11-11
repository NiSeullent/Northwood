//! A disk's GUID partition table as the mount reads it, and a block device
//! confined to one partition.
//!
//! The table is read by the one GPT reader every Stela tool uses
//! ([`stela_formats::payload::gpt::read`]): the primary, or the backup when
//! an interrupted write left the primary damaged, as boot selection reads
//! it. So a disk the mount selected is a disk it can open.
use alloc::format;
use core::fmt;
use stela_drivers::block::{BlockDevice, BlockError, Command, Finished, Identity, Request};
use stela_formats::payload::{
    PayloadError,
    gpt::{self as shared, Disk, Found},
};

pub use shared::{Guid, Partition, Table};

/// The Northwood Swap partition type, E9AFCDBE-E70E-40C5-8A5F-AF97B70BD75D.
pub const NORTHWOOD_SWAP_TYPE: Guid = shared::Role::NORTHWOOD_SWAP_TYPE;
pub const NORTHWOOD_SWAP_NAME: &str = shared::Role::NORTHWOOD_SWAP_NAME;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GptError {
    Device(BlockError),
    /// Neither copy of the GPT is present, or the disk is too small or of a
    /// block size no GPT here uses.
    NoTable,
    Invalid(&'static str),
}

impl From<BlockError> for GptError {
    fn from(error: BlockError) -> Self {
        Self::Device(error)
    }
}

impl fmt::Display for GptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device(error) => write!(f, "GPT read failed: {error}"),
            Self::NoTable => f.write_str("no GPT"),
            Self::Invalid(why) => write!(f, "invalid GPT: {why}"),
        }
    }
}

/// The geometry [`stela_formats::payload::gpt`] reads `identity`'s disk with.
pub fn geometry(identity: &Identity) -> Disk {
    Disk {
        block_bytes: u64::from(identity.block_size),
        blocks: identity.blocks,
    }
}

/// Read `disk`'s GPT: the primary table, or the backup when the primary is
/// damaged or missing. Both copies damaged is an error; a disk with neither
/// has no table.
pub fn read(disk: &mut dyn BlockDevice) -> Result<Table, GptError> {
    read_table(disk, false)
}

/// Require agreeing primary and backup GPTs and a pure protective MBR.
pub(super) fn read_for_write(disk: &mut dyn BlockDevice) -> Result<Table, GptError> {
    read_table(disk, true)
}

fn read_table(disk: &mut dyn BlockDevice, writable: bool) -> Result<Table, GptError> {
    let geometry = geometry(disk.identity());
    if !matches!(geometry.block_bytes, 512 | 1024 | 2048 | 4096)
        || geometry.blocks <= 2 * geometry.first_usable()
    {
        return Err(GptError::NoTable);
    }
    let mut failed = None;
    let read = |lba, blocks| {
        let bytes = usize::try_from(blocks * geometry.block_bytes)
            .map_err(|_| PayloadError::Disk("GPT read too large"))?;
        read_blocks(disk, lba, bytes).map_err(|error| {
            failed = Some(error.clone());
            PayloadError::io(error)
        })
    };
    let found = if writable {
        shared::read_for_write(geometry, read).map(Some)
    } else {
        shared::read(geometry, read).map(|found| match found {
            Found::Table { table, .. } => Some(table),
            Found::Blank | Found::Interrupted => None,
        })
    };
    if let Some(error) = failed {
        return Err(GptError::Device(error));
    }
    match found {
        Ok(Some(table)) => Ok(table),
        Ok(None) => Err(GptError::NoTable),
        Err(PayloadError::Disk(why)) => Err(GptError::Invalid(why)),
        Err(_) => Err(GptError::Invalid("unreadable")),
    }
}

fn read_blocks(
    disk: &mut dyn BlockDevice,
    lba: u64,
    bytes: usize,
) -> Result<alloc::vec::Vec<u8>, BlockError> {
    Request::Read { lba, bytes }
        .run(disk)?
        .ok_or(BlockError::Protocol("GPT read returned no bytes"))
}

/// A block device that reaches only the blocks of one partition.
pub struct PartitionDevice<D> {
    disk: D,
    first_lba: u64,
    partition: Partition,
    identity: Identity,
}

impl<D: BlockDevice> PartitionDevice<D> {
    pub fn new(disk: D, partition: Partition) -> Self {
        let mut identity = disk.identity().clone();
        identity.location = format!("{} partition {}", identity.location, partition.unique_guid);
        identity.blocks = partition.blocks();
        Self {
            disk,
            first_lba: partition.first_lba,
            partition,
            identity,
        }
    }

    /// The device holding the partition.
    pub fn device(&self) -> &D {
        &self.disk
    }

    pub fn partition(&self) -> &Partition {
        &self.partition
    }

    /// The whole disk's identity, as the device reported it.
    pub fn disk(&self) -> &Identity {
        self.disk.identity()
    }
}

impl<D: BlockDevice> BlockDevice for PartitionDevice<D> {
    fn identity(&self) -> &Identity {
        &self.identity
    }

    fn read(&mut self, lba: u64, buffer: &mut [u8]) -> Result<(), BlockError> {
        self.identity.check(lba, buffer.len())?;
        self.disk.read(self.first_lba + lba, buffer)
    }

    fn write(&mut self, lba: u64, data: &[u8]) -> Result<(), BlockError> {
        self.identity.check_write(lba, data.len())?;
        self.disk.write(self.first_lba + lba, data)
    }

    fn flush_cache(&mut self) -> Result<(), BlockError> {
        self.disk.flush_cache()
    }

    fn write_fua(&mut self, lba: u64, data: &[u8]) -> Result<(), BlockError> {
        self.identity.check_write(lba, data.len())?;
        self.disk.write_fua(self.first_lba + lba, data)
    }

    fn discard(&mut self, lba: u64, blocks: u64) -> Result<(), BlockError> {
        let command = Command::discard(lba, blocks).within(&self.identity, self.first_lba)?;
        self.disk.execute(command).map(drop)
    }

    fn write_zeroes(&mut self, lba: u64, blocks: u64, unmap: bool) -> Result<(), BlockError> {
        let command =
            Command::WriteZeroes { lba, blocks, unmap }.within(&self.identity, self.first_lba)?;
        self.disk.execute(command).map(drop)
    }

    fn depth(&self) -> usize {
        self.disk.depth()
    }

    fn room(&self) -> bool {
        self.disk.room()
    }

    fn start(&mut self, command: Command) -> Result<u32, BlockError> {
        self.disk
            .start(command.within(&self.identity, self.first_lba)?)
    }

    fn finish(&mut self, tag: u32) -> Option<Finished> {
        self.disk.finish(tag)
    }

    fn reaps_together(&self) -> bool {
        self.disk.reaps_together()
    }

    fn finished(&mut self, tag: u32) -> Option<Finished> {
        self.disk.finished(tag)
    }

    fn posted(&self) -> bool {
        self.disk.posted()
    }

    fn signals(&self) -> bool {
        self.disk.signals()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use alloc::string::String;
    use stela_drivers::block::RamDisk;

    pub(crate) const DATA: &str = "0FC63DAF-8483-4772-8E79-3D69D8477DE4";

    pub(crate) fn partition(kind: Guid, unique: &str, first: u64, last: u64) -> Partition {
        Partition {
            type_guid: kind,
            unique_guid: unique.parse().unwrap(),
            first_lba: first,
            last_lba: last,
            attributes: 0,
            name: String::from("test"),
        }
    }

    pub(crate) fn formatted(serial: &str, partitions: &[Partition]) -> RamDisk {
        let mut disk = RamDisk::new(serial, 256, 512);
        let table = Table {
            disk_guid: "11111111-2222-3333-4444-555555555555".parse().unwrap(),
            partitions: partitions.to_vec(),
        };
        for write in shared::encode(geometry(disk.identity()), &table).unwrap() {
            disk.write(write.lba, &write.bytes).unwrap();
        }
        disk
    }

    #[test]
    fn a_formatted_table_reads_back_and_damage_is_reported() {
        let data = DATA.parse().unwrap();
        let partitions = [
            partition(Guid::ESP, "AAAAAAAA-0000-0000-0000-000000000001", 34, 99),
            partition(data, "AAAAAAAA-0000-0000-0000-000000000002", 100, 199),
        ];
        let mut disk = formatted("s", &partitions);
        let table = read(&mut disk).unwrap();
        assert_eq!(table.partitions, partitions);
        // The backup header sits at the last block.
        assert_eq!(&disk.bytes[255 * 512..255 * 512 + 8], shared::SIGNATURE);

        // Damage to either primary structure falls back to the backup.
        for at in [2 * 512 + 40, 512 + 40] {
            let mut damaged = formatted("s", &partitions);
            damaged.bytes[at] ^= 1;
            assert_eq!(read(&mut damaged).unwrap().partitions, partitions);
        }
        let mut damaged = formatted("s", &partitions);
        damaged.bytes[512 + 40] ^= 1;
        damaged.bytes[255 * 512 + 40] ^= 1;
        assert_eq!(
            read(&mut damaged),
            Err(GptError::Invalid("GPT header CRC mismatch"))
        );
        assert_eq!(
            read(&mut RamDisk::new("s", 256, 512)),
            Err(GptError::NoTable)
        );
        assert_eq!(read(&mut RamDisk::new("s", 8, 512)), Err(GptError::NoTable));
    }

    #[test]
    fn a_partition_device_never_reaches_outside_its_partition() {
        let data = DATA.parse().unwrap();
        let target = partition(data, "AAAAAAAA-0000-0000-0000-000000000002", 100, 103);
        let mut disk = formatted("s", core::slice::from_ref(&target));
        // A cache to flush: the view passes its flush through.
        disk.identity.features.write_cache = true;
        let before = disk.bytes.clone();
        let mut view = PartitionDevice::new(disk, target);
        assert_eq!(view.identity().blocks, 4);
        view.write(0, &[0xa5; 2048]).unwrap();
        assert_eq!(
            view.write(3, &[0; 1024]),
            Err(BlockError::OutOfRange {
                lba: 3,
                blocks: 2,
                capacity: 4
            })
        );
        let mut buffer = [0; 512];
        assert!(view.read(4, &mut buffer).is_err());
        view.read(3, &mut buffer).unwrap();
        assert_eq!(buffer, [0xa5; 512]);
        view.flush().unwrap();
        let after = &view.disk.bytes;
        assert_eq!(after[..100 * 512], before[..100 * 512]);
        assert_eq!(after[104 * 512..], before[104 * 512..]);
        assert_eq!(view.disk.flushes, 1);
    }
}
