// SPDX-License-Identifier: GPL-2.0-or-later

//! Plan a four-sector update to an admitted GPT without normalizing its
//! headers, entry layout or protective MBR.
use anyhow::{Context, Result, anyhow, ensure};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    ops::RangeInclusive,
};
use stela_formats::payload::{
    PayloadError,
    gpt::{self, Disk, HeaderInfo, Partition, Table},
};
use stela_runtime::pc::swap::partition;

const STANDARD_ENTRY_BYTES: usize = 128;

#[derive(Debug)]
pub(super) struct Change {
    /// Logical sector to compare and replace during the transaction.
    pub(super) lba: u64,
    /// The full original and replacement sectors, for preflight and readback.
    pub(super) old: Vec<u8>,
    pub(super) new: Vec<u8>,
}

pub(super) struct Snapshot {
    disk: Disk,
    pub(super) table: Table,
    pub(super) protective_mbr: Vec<u8>,
    primary: Copy,
    backup: Copy,
}

struct Copy {
    header_lba: u64,
    info: HeaderInfo,
    header: Vec<u8>,
    entries: Vec<u8>,
}

/// Read exactly the sectors checked by the shared GPT admission path.
/// Both copies must also have identical raw entry arrays. Header extensions
/// and each copy's block padding are preserved independently.
pub(super) fn read(file: &mut File, disk: Disk) -> Result<Snapshot> {
    let bytes = disk
        .blocks
        .checked_mul(disk.block_bytes)
        .context("GPT image geometry overflows")?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file() && metadata.len() == bytes,
        "GPT geometry does not match a regular disk image"
    );
    let mut sectors = BTreeMap::new();
    let table = gpt::read_for_write(disk, |lba, blocks| {
        let mut bytes = vec![0; (blocks * disk.block_bytes) as usize];
        file.seek(SeekFrom::Start(lba * disk.block_bytes))
            .and_then(|_| file.read_exact(&mut bytes))
            .map_err(PayloadError::io)?;
        sectors.insert(lba, bytes.clone());
        Ok(bytes)
    })
    .map_err(|error| anyhow!("{error}"))?;
    ensure!(!table.disk_guid.is_zero(), "the GPT disk GUID is zero");
    let primary = Copy::from_sectors(disk, 1, &mut sectors)?;
    let backup = Copy::from_sectors(disk, disk.blocks - 1, &mut sectors)?;
    ensure!(
        primary.entry_array() == backup.entry_array(),
        "the primary and backup GPT raw entries differ"
    );
    let protective_mbr = sectors.remove(&0).context("missing protective MBR")?;
    Ok(Snapshot {
        disk,
        table,
        protective_mbr,
        primary,
        backup,
    })
}

impl Snapshot {
    pub(super) fn usable(&self) -> RangeInclusive<u64> {
        self.primary.info.first_usable..=self.primary.info.last_usable
    }

    pub(super) fn has_empty_slot(&self) -> bool {
        self.empty_slot().is_some()
    }

    fn empty_slot(&self) -> Option<usize> {
        self.primary
            .entry_array()
            .chunks_exact(self.primary.info.entry_bytes as usize)
            .position(|entry| entry.iter().all(|&byte| byte == 0))
    }

    /// Use the exact placement from the shared swap planner. Returned changes
    /// are backup entries, backup header, primary entries and primary header.
    /// The caller must persist the backup copy before modifying the primary.
    pub(super) fn insert_swap(&self, partition: &Partition) -> Result<[Change; 4]> {
        let bytes = partition
            .last_lba
            .checked_sub(partition.first_lba)
            .and_then(|blocks| blocks.checked_add(1))
            .and_then(|blocks| blocks.checked_mul(self.disk.block_bytes))
            .context("invalid swap partition capacity")?;
        let expected = partition::plan(
            &self.table,
            self.usable(),
            self.disk.block_bytes,
            bytes,
            partition.unique_guid,
        )
        .map_err(|error| anyhow!("{error}"))?;
        ensure!(
            expected.partition == *partition,
            "the swap entry does not match the verified GPT placement"
        );

        let entry_bytes = self.primary.info.entry_bytes as usize;
        let slot = self
            .empty_slot()
            .context("the GPT has no entirely empty partition entry")?;
        let mut entries = self.primary.entry_array().to_vec();
        let offset = slot * entry_bytes;
        let entry = &mut entries[offset..offset + STANDARD_ENTRY_BYTES];
        entry[..16].copy_from_slice(&partition.type_guid.0);
        entry[16..32].copy_from_slice(&partition.unique_guid.0);
        entry[32..40].copy_from_slice(&partition.first_lba.to_le_bytes());
        entry[40..48].copy_from_slice(&partition.last_lba.to_le_bytes());
        entry[48..56].copy_from_slice(&partition.attributes.to_le_bytes());
        for (unit, field) in partition
            .name
            .encode_utf16()
            .zip(entry[56..].chunks_exact_mut(2))
        {
            field.copy_from_slice(&unit.to_le_bytes());
        }

        let crc = gpt::crc32(&entries);
        let block = self.disk.block_bytes as usize;
        let entry = &entries[offset..offset + STANDARD_ENTRY_BYTES];
        let [backup_entries, backup_header] = self.backup.changes(block, offset, entry, crc);
        let [primary_entries, primary_header] = self.primary.changes(block, offset, entry, crc);
        Ok([
            backup_entries,
            backup_header,
            primary_entries,
            primary_header,
        ])
    }
}

impl Copy {
    fn from_sectors(
        disk: Disk,
        header_lba: u64,
        sectors: &mut BTreeMap<u64, Vec<u8>>,
    ) -> Result<Self> {
        let header = sectors.remove(&header_lba).context("missing GPT header")?;
        let info = gpt::parse_header(disk, header_lba, &header)
            .map_err(|error| anyhow!("{error}"))?
            .context("missing GPT header signature")?;
        let entries = sectors
            .remove(&info.entries_lba)
            .context("missing GPT entry array")?;
        Ok(Self {
            header_lba,
            info,
            header,
            entries,
        })
    }

    fn entry_array(&self) -> &[u8] {
        &self.entries[..self.info.entry_array_bytes() as usize]
    }

    fn changes(&self, block: usize, offset: usize, entry: &[u8], crc: u32) -> [Change; 2] {
        // Entry sizes and logical blocks are powers of two, both multiples
        // of 128. Its standard fields always fit within one sector.
        let sector = offset / block;
        let within = offset % block;
        let old_entries = self.entries[sector * block..(sector + 1) * block].to_vec();
        let mut new_entries = old_entries.clone();
        new_entries[within..within + entry.len()].copy_from_slice(entry);

        let mut header = self.header.clone();
        let size = u32::from_le_bytes(header[12..16].try_into().expect("admitted GPT header"));
        header[88..92].copy_from_slice(&crc.to_le_bytes());
        header[16..20].fill(0);
        let header_crc = gpt::crc32(&header[..size as usize]);
        header[16..20].copy_from_slice(&header_crc.to_le_bytes());
        [
            Change {
                lba: self.info.entries_lba + sector as u64,
                old: old_entries,
                new: new_entries,
            },
            Change {
                lba: self.header_lba,
                old: self.header.clone(),
                new: header,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use stela_formats::payload::gpt::{ALIGNMENT, Guid, Role};
    use stela_runtime::pc::swap::MIN_STORE_BYTES;

    const ENTRY_BYTES: usize = 256;
    const ENTRY_COUNT: usize = 5;

    fn guid(value: u8) -> Guid {
        Guid::from_entropy([value; 16])
    }

    fn write_sector(file: &mut File, disk: Disk, lba: u64, bytes: &[u8]) {
        file.seek(SeekFrom::Start(lba * disk.block_bytes)).unwrap();
        file.write_all(bytes).unwrap();
    }

    fn extended_image(block_bytes: u64, have_empty: bool) -> (File, Disk) {
        let disk = Disk {
            block_bytes,
            blocks: 32 * ALIGNMENT / block_bytes,
        };
        let align = ALIGNMENT / block_bytes;
        let table = Table {
            disk_guid: guid(1),
            partitions: [(Role::AppleRegion, 2, 8, 10), (Role::Esp, 3, 22, 23)]
                .into_iter()
                .map(|(role, id, start, end)| Partition {
                    type_guid: role.type_guid(),
                    unique_guid: guid(id),
                    first_lba: start * align,
                    last_lba: end * align - 1,
                    attributes: id.into(),
                    name: role.name().into(),
                })
                .collect(),
        };
        let writes = gpt::encode(disk, &table).unwrap();
        let mut file = tempfile::tempfile().unwrap();
        file.set_len(disk.blocks * block_bytes).unwrap();

        let size = ENTRY_BYTES * ENTRY_COUNT;
        let mut entries = vec![0u8; size.div_ceil(block_bytes as usize) * block_bytes as usize];
        let encoded_entries = &writes[2].bytes;
        entries[ENTRY_BYTES..ENTRY_BYTES + STANDARD_ENTRY_BYTES]
            .copy_from_slice(&encoded_entries[..STANDARD_ENTRY_BYTES]);
        entries[3 * ENTRY_BYTES..3 * ENTRY_BYTES + STANDARD_ENTRY_BYTES]
            .copy_from_slice(&encoded_entries[STANDARD_ENTRY_BYTES..2 * STANDARD_ENTRY_BYTES]);
        entries[145] = 0x11; // Type zero, with unrelated bytes in its extension.
        entries[ENTRY_BYTES + 145] = 0x22;
        entries[3 * ENTRY_BYTES + 145] = 0x33;
        if !have_empty {
            entries[2 * ENTRY_BYTES + 145] = 0x44;
            entries[4 * ENTRY_BYTES + 145] = 0x55;
        }
        let entry_crc = gpt::crc32(&entries[..size]);

        for (header_write, entries_lba, extra) in
            [(&writes[1], disk.blocks - 12, 0x41), (&writes[3], 8, 0x42)]
        {
            let mut header = header_write.bytes.clone();
            header[12..16].copy_from_slice(&112u32.to_le_bytes());
            header[40..48].copy_from_slice(&(2 * align + 1).to_le_bytes());
            header[48..56].copy_from_slice(&(28 * align - 1).to_le_bytes());
            header[72..80].copy_from_slice(&entries_lba.to_le_bytes());
            header[80..84].copy_from_slice(&(ENTRY_COUNT as u32).to_le_bytes());
            header[84..88].copy_from_slice(&(ENTRY_BYTES as u32).to_le_bytes());
            header[88..92].copy_from_slice(&entry_crc.to_le_bytes());
            header[101] = extra;
            header[119] = extra + 1;
            header[16..20].fill(0);
            let crc = gpt::crc32(&header[..112]);
            header[16..20].copy_from_slice(&crc.to_le_bytes());
            entries[size] = extra;
            write_sector(&mut file, disk, entries_lba, &entries);
            write_sector(&mut file, disk, header_write.lba, &header);
        }
        let mut mbr = writes[4].bytes.clone();
        mbr[23] = 0x71;
        write_sector(&mut file, disk, 0, &mbr);
        (file, disk)
    }

    fn swap(snapshot: &Snapshot, disk: Disk) -> Partition {
        partition::plan(
            &snapshot.table,
            snapshot.usable(),
            disk.block_bytes,
            MIN_STORE_BYTES,
            guid(9),
        )
        .unwrap()
        .partition
    }

    #[test]
    fn inserts_at_an_empty_index_and_preserves_all_other_raw_metadata() {
        for block_bytes in [512, 4096] {
            let (mut file, disk) = extended_image(block_bytes, true);
            let before = read(&mut file, disk).unwrap();
            let swap = swap(&before, disk);
            assert!(before.has_empty_slot());
            let changes = before.insert_swap(&swap).unwrap();
            let slot_sector = (2 * ENTRY_BYTES) as u64 / block_bytes;
            assert_eq!(
                changes.each_ref().map(|change| change.lba),
                [
                    disk.blocks - 12 + slot_sector,
                    disk.blocks - 1,
                    8 + slot_sector,
                    1
                ]
            );
            for change in &changes {
                let mut bytes = vec![0; block_bytes as usize];
                file.seek(SeekFrom::Start(change.lba * block_bytes))
                    .unwrap();
                file.read_exact(&mut bytes).unwrap();
                assert_eq!(bytes, change.old);
                assert_eq!(change.new.len(), block_bytes as usize);
                write_sector(&mut file, disk, change.lba, &change.new);
            }

            let after = read(&mut file, disk).unwrap();
            assert_eq!(after.usable(), before.usable());
            assert_eq!(after.protective_mbr, before.protective_mbr);
            assert_eq!(
                after.table.partitions,
                [
                    before.table.partitions[0].clone(),
                    swap,
                    before.table.partitions[1].clone(),
                ]
            );
            for (old, new) in [
                (&before.primary, &after.primary),
                (&before.backup, &after.backup),
            ] {
                assert_eq!(old.info.entries_lba, new.info.entries_lba);
                assert_eq!(old.info.entry_count, new.info.entry_count);
                assert_eq!(old.info.entry_bytes, new.info.entry_bytes);
                let range = 2 * ENTRY_BYTES..2 * ENTRY_BYTES + STANDARD_ENTRY_BYTES;
                assert_eq!(old.entries[..range.start], new.entries[..range.start]);
                assert_eq!(old.entries[range.end..], new.entries[range.end..]);
                assert_eq!(old.header[..16], new.header[..16]);
                assert_eq!(old.header[20..88], new.header[20..88]);
                assert_eq!(old.header[92..], new.header[92..]);
            }
        }
    }

    #[test]
    fn a_zero_type_does_not_authorize_overwriting_residual_entry_bytes() {
        let (mut file, disk) = extended_image(512, false);
        let snapshot = read(&mut file, disk).unwrap();
        assert!(!snapshot.has_empty_slot());
        let error = snapshot.insert_swap(&swap(&snapshot, disk)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("no entirely empty partition entry")
        );
    }
}
