// SPDX-License-Identifier: GPL-2.0-or-later

//! Check where a swap partition could fit in an existing GPT disk image.
use super::gpt_edit::{self, Snapshot};
use anyhow::{Context, Result, anyhow, bail, ensure};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    ops::RangeInclusive,
    path::Path,
};
use stela_formats::payload::gpt::{
    ALIGNMENT, DEFAULT_MIN_APPLE_BYTES, Disk, Guid, Partition, Role, Table,
};
use stela_runtime::pc::swap::{Layout, partition};

#[derive(Debug)]
pub(super) struct Preview {
    pub(super) disk_guid: Guid,
    pub(super) target: Partition,
    pub(super) target_bytes: u64,
    pub(super) block_bytes: u64,
    pub(super) required_bytes: u64,
    pub(super) largest_gap_bytes: u64,
    pub(super) slots: u64,
    pub(super) placement: Option<partition::Plan>,
}

pub(super) fn disk_guid(path: &Path) -> Result<Guid> {
    let mut file = open_regular(path, false)?;
    Ok(read_image(&mut file, path)?.gpt.table.disk_guid)
}

pub(super) fn plan(
    path: &Path,
    target: Guid,
    boot_disk: Guid,
    requested_bytes: u64,
) -> Result<Preview> {
    let mut file = open_regular(path, false)?;
    read_image(&mut file, path)?.plan(path, target, boot_disk, requested_bytes, None)
}

impl Preview {
    pub(super) fn placement(&self) -> Result<&partition::Plan> {
        ensure!(
            self.target_bytes >= DEFAULT_MIN_APPLE_BYTES,
            "the selected macOS target partition is below the setup minimum"
        );
        self.placement.as_ref().ok_or_else(|| {
            anyhow!(
                "the swap partition needs {} more contiguous bytes",
                self.required_bytes - self.largest_gap_bytes
            )
        })
    }
}

impl ImageGpt {
    pub(super) fn plan(
        &self,
        path: &Path,
        target: Guid,
        boot_disk: Guid,
        requested_bytes: u64,
        uuid: Option<Guid>,
    ) -> Result<Preview> {
        let table = &self.gpt.table;
        ensure!(!boot_disk.is_zero(), "the boot disk has no GPT disk GUID");
        ensure!(
            table.disk_guid != boot_disk,
            "{} is the protected boot disk (GPT GUID {})",
            path.display(),
            boot_disk
        );
        let mut targets = table
            .partitions
            .iter()
            .filter(|entry| entry.unique_guid == target);
        let target = targets
            .next()
            .with_context(|| format!("{} has no target partition {target}", path.display()))?;
        ensure!(
            targets.next().is_none()
                && !target.unique_guid.is_zero()
                && matches!(target.role(), None | Some(Role::AppleRegion)),
            "partition {} is not an unambiguous macOS target",
            target.unique_guid
        );
        ensure!(
            self.gpt.has_empty_slot(),
            "the GPT has no free partition entry"
        );

        let block_bytes = self.disk.block_bytes;
        let required_bytes = requested_bytes
            .div_ceil(block_bytes)
            .checked_mul(block_bytes)
            .context("swap partition size overflows after rounding to a logical block")?;
        let slots = Layout::for_store(required_bytes)
            .map_err(|error| anyhow!("{error}"))?
            .slot_count();
        let usable = self.gpt.usable();
        let placement = match partition::plan(
            table,
            usable.clone(),
            block_bytes,
            required_bytes,
            uuid.unwrap_or_else(|| unused_preview_guid(table)),
        ) {
            Ok(plan) => Some(plan),
            Err(partition::Error::NoSpace) => None,
            Err(error) => bail!("{error}"),
        };

        Ok(Preview {
            disk_guid: table.disk_guid,
            target: target.clone(),
            target_bytes: target.blocks() * block_bytes,
            block_bytes,
            required_bytes,
            largest_gap_bytes: largest_gap_bytes(table, usable, block_bytes),
            slots,
            placement,
        })
    }
}

/// The planner needs a free GUID to check the format. The preview returns
/// only LBAs, so the caller never treats this GUID as a partition identity.
fn unused_preview_guid(table: &Table) -> Guid {
    let used: BTreeSet<_> = table
        .partitions
        .iter()
        .map(|entry| entry.unique_guid)
        .chain([table.disk_guid])
        .collect();
    (1..=used.len() as u64 + 1)
        .map(|number| {
            let mut bytes = [0; 16];
            bytes[..8].copy_from_slice(&number.to_le_bytes());
            Guid::from_entropy(bytes)
        })
        .find(|guid| !used.contains(guid))
        .expect("more preview GUIDs than GPT entries")
}

/// Report the maximum capacity after applying the same start alignment
/// as the partition planner. The planner already checked entry geometry.
fn largest_gap_bytes(table: &Table, usable: RangeInclusive<u64>, block_bytes: u64) -> u64 {
    let mut partitions: Vec<_> = table.partitions.iter().collect();
    partitions.sort_unstable_by_key(|entry| entry.first_lba);
    let alignment = ALIGNMENT / block_bytes;
    let mut cursor = *usable.start();
    let mut largest = 0;
    for entry in partitions
        .into_iter()
        .map(Some)
        .chain(std::iter::once(None))
    {
        let end = entry.map_or(usable.end() + 1, |entry| entry.first_lba);
        let start = cursor.div_ceil(alignment) * alignment;
        largest = largest.max(end.saturating_sub(start));
        if let Some(entry) = entry {
            cursor = entry.last_lba + 1;
        }
    }
    largest * block_bytes
}

pub(super) struct ImageGpt {
    pub(super) disk: Disk,
    pub(super) gpt: Snapshot,
}

pub(super) fn open_regular(path: &Path, writable: bool) -> Result<File> {
    let metadata =
        fs::metadata(path).with_context(|| format!("cannot inspect {}", path.display()))?;
    ensure!(
        metadata.is_file(),
        "{} is not a regular disk image",
        path.display()
    );
    let file = OpenOptions::new()
        .read(true)
        .write(writable)
        .open(path)
        .with_context(|| format!("cannot open {}", path.display()))?;
    ensure!(
        file.metadata()?.is_file(),
        "{} is not a regular disk image",
        path.display()
    );
    Ok(file)
}

pub(super) fn read_image(file: &mut File, path: &Path) -> Result<ImageGpt> {
    let bytes = file.metadata()?.len();
    let mut found = None;
    let mut errors = Vec::new();
    for block_bytes in [512, 1024, 2048, 4096] {
        if !bytes.is_multiple_of(block_bytes) {
            continue;
        }
        let disk = Disk {
            block_bytes,
            blocks: bytes / block_bytes,
        };
        match gpt_edit::read(file, disk) {
            Ok(gpt) => {
                ensure!(
                    found.is_none(),
                    "{} has ambiguous GPT block geometry",
                    path.display()
                );
                found = Some(ImageGpt { disk, gpt });
            }
            Err(error) => errors.push(format!("{block_bytes}-byte blocks: {error}")),
        }
    }
    let image = found.with_context(|| {
        let reason = if errors.is_empty() {
            String::from("image length does not contain whole logical blocks")
        } else {
            errors.join("; ")
        };
        format!("{} has no verified GPT ({reason})", path.display())
    })?;
    ensure!(
        !image.gpt.table.disk_guid.is_zero(),
        "{} has a zero GPT disk GUID",
        path.display()
    );
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};
    use stela_formats::payload::gpt;
    use stela_runtime::pc::swap::MIN_STORE_BYTES;

    fn guid(value: u8) -> Guid {
        Guid::from_entropy([value; 16])
    }

    fn image(path: &Path, block_bytes: u64) {
        let disk = Disk {
            block_bytes,
            blocks: 64 * ALIGNMENT / block_bytes,
        };
        let alignment = ALIGNMENT / block_bytes;
        let partitions = [
            (Role::Esp, 2, 5, 6),
            (Role::AppleRegion, 3, 8, 10),
            (Role::MachineState, 4, 22, 24),
        ]
        .map(|(role, id, start, end)| Partition {
            type_guid: role.type_guid(),
            unique_guid: guid(id),
            first_lba: start * alignment,
            last_lba: end * alignment - 1,
            attributes: 0,
            name: role.name().into(),
        });
        let table = Table {
            disk_guid: guid(1),
            partitions: partitions.into(),
        };
        let mut file = File::create(path).unwrap();
        file.set_len(disk.blocks * block_bytes).unwrap();
        for mut write in gpt::encode(disk, &table).unwrap() {
            if write.lba == 1 || write.lba == disk.blocks - 1 {
                // Both headers reserve more than the usual GPT metadata.
                let first = 2 * alignment + 1;
                let last = 29 * alignment - 1;
                write.bytes[40..48].copy_from_slice(&first.to_le_bytes());
                write.bytes[48..56].copy_from_slice(&last.to_le_bytes());
                write.bytes[16..20].fill(0);
                let size = u32::from_le_bytes(write.bytes[12..16].try_into().unwrap());
                let crc = gpt::crc32(&write.bytes[..size as usize]);
                write.bytes[16..20].copy_from_slice(&crc.to_le_bytes());
            }
            file.seek(SeekFrom::Start(write.lba * block_bytes)).unwrap();
            file.write_all(&write.bytes).unwrap();
        }
    }

    #[test]
    fn preview_uses_verified_usable_bounds_and_contiguous_aligned_capacity() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("target.img");
        for block_bytes in [512, 4096] {
            image(&path, block_bytes);
            assert_eq!(disk_guid(&path).unwrap(), guid(1));
            let result = plan(&path, guid(3), guid(8), MIN_STORE_BYTES + 1).unwrap();
            assert_eq!(result.disk_guid, guid(1));
            assert_eq!(result.target.unique_guid, guid(3));
            assert_eq!(result.target_bytes, 2 * ALIGNMENT);
            assert_eq!(result.block_bytes, block_bytes);
            assert_eq!(result.required_bytes, MIN_STORE_BYTES + block_bytes);
            assert_eq!(result.slots, 1);
            assert_eq!(result.largest_gap_bytes, 12 * ALIGNMENT);
            let first = 10 * ALIGNMENT / block_bytes;
            assert_eq!(
                result
                    .placement
                    .map(|plan| plan.partition.first_lba..=plan.partition.last_lba),
                Some(first..=first + result.required_bytes / block_bytes - 1)
            );
            let result = plan(&path, guid(3), guid(8), 12 * ALIGNMENT + 1).unwrap();
            assert!(result.placement.is_none());
            assert_eq!(result.required_bytes, 12 * ALIGNMENT + block_bytes);
            assert_eq!(result.largest_gap_bytes, 12 * ALIGNMENT);
        }
    }

    #[test]
    fn preview_refuses_the_boot_disk_unsupported_targets_and_a_damaged_gpt() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("target.img");
        image(&path, 512);
        let error = plan(&path, guid(3), guid(1), MIN_STORE_BYTES).unwrap_err();
        assert!(error.to_string().contains("protected boot disk"));
        assert!(plan(&path, guid(2), guid(8), MIN_STORE_BYTES).is_err());
        assert!(plan(&path, guid(4), guid(8), MIN_STORE_BYTES).is_err());
        assert!(plan(&path, guid(3), guid(8), 1).is_err());
        assert!(disk_guid(directory.path()).is_err());

        let mut file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::End(-512)).unwrap();
        file.write_all(&[0; 512]).unwrap();
        assert!(disk_guid(&path).is_err());
    }
}
