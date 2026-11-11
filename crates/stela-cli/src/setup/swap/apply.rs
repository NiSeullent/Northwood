// SPDX-License-Identifier: GPL-2.0-or-later

//! Apply a MEM-04 partition to a private, regular GPT image.
use super::{PlanImage, gpt_edit, preview, random_guid, summary};
use anyhow::{Context, Result, anyhow, bail, ensure};
use std::{
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom, Write},
};
use stela_runtime::pc::swap::{HEADER_BYTES, Header, partition};

pub(super) fn run(selection: &PlanImage) -> Result<String> {
    let mut boot = preview::open_regular(&selection.boot_disk, false)?;
    boot.try_lock_shared()
        .with_context(|| format!("cannot lock boot image {}", selection.boot_disk.display()))?;
    let boot_guid = preview::read_image(&mut boot, &selection.boot_disk)?
        .gpt
        .table
        .disk_guid;
    // The shared boot lock would otherwise hide the identity error for a
    // same-path selection or symlink alias when we request the write lock.
    if fs::canonicalize(&selection.disk)? == fs::canonicalize(&selection.boot_disk)? {
        bail!(
            "{} is the protected boot disk (GPT GUID {boot_guid})",
            selection.disk.display()
        );
    }

    let mut image = preview::open_regular(&selection.disk, true)?;
    image.try_lock().with_context(|| {
        format!(
            "cannot acquire the exclusive image lock for {}",
            selection.disk.display()
        )
    })?;
    let before = preview::read_image(&mut image, &selection.disk)?;
    let plan = before.plan(
        &selection.disk,
        selection.target,
        boot_guid,
        selection.bytes,
        Some(random_guid()?),
    )?;
    let planned = plan.placement()?;
    let updates = Updates::prepare(&mut image, &before, planned)?;
    commit(&mut image, &updates, |file| {
        verify_image(file, &before, planned)
    })?;
    Ok(format!(
        "Applied and verified in {}\n  target disk GUID  {}\n  target partition  {}\n{}",
        selection.disk.display(),
        plan.disk_guid,
        plan.target.unique_guid,
        summary(
            planned.header,
            &format!(
                "GPT LBA {}..={}",
                planned.partition.first_lba, planned.partition.last_lba
            )
        )
    ))
}

fn verify_image(
    file: &mut File,
    before: &preview::ImageGpt,
    planned: &partition::Plan,
) -> Result<()> {
    let after = gpt_edit::read(file, before.disk).context("verifying both GPT copies")?;
    let existing = after
        .table
        .partitions
        .iter()
        .filter(|entry| entry.unique_guid != planned.partition.unique_guid);
    ensure!(
        after.table.disk_guid == before.gpt.table.disk_guid
            && after.usable() == before.gpt.usable()
            && after.protective_mbr == before.gpt.protective_mbr
            && existing.eq(&before.gpt.table.partitions)
            && after.table.partitions.contains(&planned.partition),
        "the installed GPT differs from the verified swap placement"
    );
    let bytes = read_at(
        file,
        planned.partition.first_lba * before.disk.block_bytes,
        HEADER_BYTES,
    )?;
    let header = Header::decode_partition(&bytes, &planned.partition, before.disk.block_bytes)
        .map_err(|error| anyhow!("{error}"))?;
    ensure!(
        header == planned.header,
        "the installed swap header differs from the plan"
    );
    Ok(())
}

struct Update {
    offset: u64,
    old: Vec<u8>,
    new: Vec<u8>,
}

struct Updates {
    header: Update,
    gpt: [Update; 4],
}

impl Updates {
    fn prepare(
        file: &mut File,
        before: &preview::ImageGpt,
        planned: &partition::Plan,
    ) -> Result<Self> {
        let block = before.disk.block_bytes;
        let offset = planned.partition.first_lba * block;
        let gpt = before
            .gpt
            .insert_swap(&planned.partition)?
            .map(|change| Update {
                offset: change.lba * block,
                old: change.old,
                new: change.new,
            });
        Ok(Self {
            header: Update {
                offset,
                old: read_at(file, offset, HEADER_BYTES)?,
                new: planned.header.encode().to_vec(),
            },
            gpt,
        })
    }
}

#[derive(Clone, Copy)]
enum Stage {
    Header,
    Backup,
    Primary,
}

trait ImageIo: Read + Write + Seek {
    fn durable(&mut self) -> io::Result<()>;
}

impl ImageIo for File {
    fn durable(&mut self) -> io::Result<()> {
        self.sync_all()
    }
}

fn commit<F: ImageIo>(
    image: &mut F,
    updates: &Updates,
    verify: impl FnOnce(&mut F) -> Result<()>,
) -> Result<()> {
    let header = std::slice::from_ref(&updates.header);
    let (backup, primary) = updates.gpt.split_at(2);
    check(image, &updates.gpt, |update| &update.old).context("GPT changed before applying swap")?;
    check(image, header, |update| &update.old)
        .context("swap header area changed before applying")?;
    let mut stage = Stage::Header;
    let applied = (|| -> Result<()> {
        publish(image, header, |update| &update.new).context("writing the idle swap header")?;
        stage = Stage::Backup;
        publish(image, backup, |update| &update.new).context("publishing the backup GPT")?;
        stage = Stage::Primary;
        publish(image, primary, |update| &update.new).context("publishing the primary GPT")?;
        verify(image)
    })();
    let Err(error) = applied else {
        return Ok(());
    };
    // Keep a verified GPT copy in place. After touching the primary, the
    // new backup remains usable until the old primary has been restored.
    let restored = (|| -> Result<()> {
        if matches!(stage, Stage::Primary) {
            publish(image, primary, |update| &update.old).context("restoring the primary GPT")?;
        }
        if matches!(stage, Stage::Primary | Stage::Backup) {
            publish(image, backup, |update| &update.old).context("restoring the backup GPT")?;
        }
        publish(image, header, |update| &update.old).context("restoring the original header area")
    })();
    match restored {
        Ok(()) => {
            Err(error.context("swap application failed and the original bytes were restored"))
        }
        Err(rollback) => Err(anyhow!(
            "swap application failed: {error:#}. Image rollback failed: {rollback:#}"
        )),
    }
}

type Contents = fn(&Update) -> &[u8];

fn publish(image: &mut impl ImageIo, updates: &[Update], contents: Contents) -> Result<()> {
    for update in updates {
        image.seek(SeekFrom::Start(update.offset))?;
        image.write_all(contents(update))?;
    }
    image.durable()?;
    check(image, updates, contents)
}

fn check(image: &mut (impl Read + Seek), updates: &[Update], contents: Contents) -> Result<()> {
    for update in updates {
        let expected = contents(update);
        ensure!(
            read_at(image, update.offset, expected.len())? == expected,
            "readback mismatch at image byte {}",
            update.offset
        );
    }
    Ok(())
}

fn read_at(image: &mut (impl Read + Seek), offset: u64, length: usize) -> Result<Vec<u8>> {
    let mut bytes = vec![0; length];
    image.seek(SeekFrom::Start(offset))?;
    image.read_exact(&mut bytes)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use stela_formats::payload::gpt::{self, ALIGNMENT, Disk, Guid, Partition, Role, Table};
    use stela_runtime::pc::swap::MIN_STORE_BYTES;

    fn guid(value: u8) -> Guid {
        Guid::from_entropy([value; 16])
    }

    fn fixture(disk_guid: Guid, block_bytes: u64) -> tempfile::NamedTempFile {
        let mut image = tempfile::NamedTempFile::new().unwrap();
        let file = image.as_file_mut();
        let disk = Disk {
            block_bytes,
            blocks: 16 * ALIGNMENT / block_bytes,
        };
        let alignment = ALIGNMENT / disk.block_bytes;
        let table = Table {
            disk_guid,
            partitions: vec![Partition {
                type_guid: Role::AppleRegion.type_guid(),
                unique_guid: guid(3),
                first_lba: alignment,
                last_lba: 3 * alignment - 1,
                attributes: 0,
                name: Role::AppleRegion.name().into(),
            }],
        };
        file.set_len(disk.blocks * disk.block_bytes).unwrap();
        for write in gpt::encode(disk, &table).unwrap() {
            file.seek(SeekFrom::Start(write.lba * disk.block_bytes))
                .unwrap();
            file.write_all(&write.bytes).unwrap();
        }
        // A failed transaction must restore even nonzero bytes in the free gap.
        file.seek(SeekFrom::Start(3 * ALIGNMENT)).unwrap();
        file.write_all(&[0xa5; HEADER_BYTES]).unwrap();
        image
    }

    #[derive(Clone, Copy, Debug)]
    enum Fault {
        Write(usize),
        Sync(usize),
        Verify,
    }

    struct Faulty<'a> {
        file: &'a mut File,
        fault: Fault,
        writes: usize,
        syncs: usize,
    }

    impl Read for Faulty<'_> {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            self.file.read(buffer)
        }
    }

    impl Seek for Faulty<'_> {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.file.seek(position)
        }
    }

    impl Write for Faulty<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            if matches!(self.fault, Fault::Write(index) if index == self.writes) {
                self.file.write_all(&bytes[..bytes.len() / 2])?;
                return Err(io::Error::other("injected partial image write"));
            }
            self.file.write(bytes)
        }

        fn flush(&mut self) -> io::Result<()> {
            self.file.flush()
        }
    }

    impl ImageIo for Faulty<'_> {
        fn durable(&mut self) -> io::Result<()> {
            self.syncs += 1;
            if matches!(self.fault, Fault::Sync(index) if index == self.syncs) {
                return Err(io::Error::other("injected sync error"));
            }
            self.file.sync_all()
        }
    }

    #[test]
    fn restores_the_private_image_after_write_sync_or_verification_errors() {
        let faults = (1..=5)
            .map(Fault::Write)
            .chain((1..=3).map(Fault::Sync))
            .chain([Fault::Verify]);
        let image = fixture(guid(1), 512);
        let baseline = fs::read(image.path()).unwrap();
        let mut file = preview::open_regular(image.path(), true).unwrap();
        file.try_lock().unwrap();
        for fault in faults {
            let before = preview::read_image(&mut file, image.path()).unwrap();
            let planned = partition::plan(
                &before.gpt.table,
                before.gpt.usable(),
                before.disk.block_bytes,
                MIN_STORE_BYTES,
                guid(4),
            )
            .unwrap();
            let updates = Updates::prepare(&mut file, &before, &planned).unwrap();
            let mut faulty = Faulty {
                file: &mut file,
                fault,
                writes: 0,
                syncs: 0,
            };
            let error = commit(&mut faulty, &updates, |file| {
                verify_image(file.file, &before, &planned)?;
                bail!("injected final verification error")
            })
            .unwrap_err();
            assert!(
                error.to_string().contains("original bytes were restored"),
                "{fault:?}: {error:#}"
            );
            assert!(
                read_at(&mut file, 0, baseline.len()).unwrap() == baseline,
                "{fault:?}"
            );
        }
    }

    #[test]
    fn refuses_an_image_locked_by_another_handle() {
        let image = fixture(guid(1), 512);
        let boot = fixture(guid(2), 512);
        let baseline = fs::read(image.path()).unwrap();
        let lock = preview::open_regular(image.path(), true).unwrap();
        lock.try_lock().unwrap();
        let options = PlanImage {
            disk: image.path().to_path_buf(),
            target: guid(3),
            boot_disk: boot.path().to_path_buf(),
            bytes: MIN_STORE_BYTES,
        };
        let error = run(&options).unwrap_err();
        assert!(
            error.to_string().contains("exclusive image lock"),
            "{error:#}"
        );
        drop(lock);
        assert!(fs::read(image.path()).unwrap() == baseline);
    }

    #[test]
    fn installed_header_is_readable_by_the_inspector_at_all_image_sector_sizes() {
        for block_bytes in [512, 1024, 2048, 4096] {
            let image = fixture(guid(1), block_bytes);
            let mut file = preview::open_regular(image.path(), true).unwrap();
            file.try_lock().unwrap();
            let before = preview::read_image(&mut file, image.path()).unwrap();
            let planned = partition::plan(
                &before.gpt.table,
                before.gpt.usable(),
                block_bytes,
                MIN_STORE_BYTES,
                guid(4),
            )
            .unwrap();
            let updates = Updates::prepare(&mut file, &before, &planned).unwrap();
            commit(&mut file, &updates, |file| {
                verify_image(file, &before, &planned)
            })
            .unwrap();
            drop(file);
            let (mut reader, _) = crate::boot_log::open_gpt_image(image.path()).unwrap();
            let (entry, header) = partition::inspect(&mut reader, Some(guid(4))).unwrap();
            assert_eq!(entry, planned.partition);
            assert_eq!(header, planned.header);
        }
    }
}
