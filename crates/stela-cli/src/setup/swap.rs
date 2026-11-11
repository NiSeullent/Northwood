// SPDX-License-Identifier: GPL-2.0-or-later
use super::{Reported, host::bytes_text};
use anyhow::{Context, Result, anyhow, ensure};
use clap::Args;
use std::{
    fmt::Write as _,
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};
use stela_formats::payload::gpt::{self, ALIGNMENT, DEFAULT_MIN_APPLE_BYTES, Disk, Table};
use stela_runtime::pc::{
    gpt::Guid,
    swap::{DEFAULT_STORE_BYTES, DEFAULT_VOLUME_MAX_BYTES, Header, SLOT_BYTES, partition, volume},
};

mod apply;
mod fat_image;
mod gpt_edit;
mod prepare_volume;
mod preview;

#[derive(Args)]
pub(super) struct Inspect {
    /// A whole-disk GPT image.
    #[arg(long, required_unless_present = "volume", conflicts_with = "volume")]
    disk: Option<PathBuf>,
    /// A raw FAT32 volume image holding the preallocated swap files.
    #[arg(long)]
    volume: Option<PathBuf>,
    /// The unique GPT partition GUID. The image may otherwise have one swap partition.
    #[arg(long, requires = "disk")]
    partition: Option<Guid>,
    /// Logical block size of the FAT32 volume. Defaults to 512 bytes.
    #[arg(long, requires = "volume")]
    block_bytes: Option<u64>,
    /// FAT32 policy cap in bytes. Defaults to 8 GiB.
    #[arg(long, requires = "volume")]
    max_bytes: Option<u64>,
}

impl Inspect {
    pub(super) fn run(&self) -> Result<String> {
        if let Some(path) = &self.volume {
            let image = crate::apfs::open_image(path)?;
            let metadata = volume::inspect(
                image,
                self.block_bytes.unwrap_or(512),
                self.max_bytes.unwrap_or(DEFAULT_VOLUME_MAX_BYTES),
            )
            .map_err(|error| anyhow!("{}: {error}", path.display()))?;
            return Ok(summary(
                metadata.header,
                &format!("FAT32 files, {} bytes configured", metadata.config.bytes()),
            ));
        }
        let path = self
            .disk
            .as_ref()
            .ok_or_else(|| anyhow!("a swap image is required"))?;
        let (mut disk, _) = crate::boot_log::open_gpt_image(path)?;
        let (partition, header) = partition::inspect(&mut disk, self.partition)
            .map_err(|error| anyhow!("{}: {error}", path.display()))?;
        Ok(summary(
            header,
            &format!("GPT LBA {}..={}", partition.first_lba, partition.last_lba),
        ))
    }
}

fn summary(header: Header, location: &str) -> String {
    let layout = header.layout();
    format!(
        "Northwood Swap {}\n  location {location}\n  state {:?}\n  generation {}\n  encrypted {}\n  slots {}\n  bytes used by the layout {}\n",
        header.store_uuid(),
        header.state(),
        header.generation(),
        header.encrypted(),
        layout.slot_count(),
        layout.used_bytes(),
    )
}

#[derive(Args)]
pub(super) struct PlanImage {
    /// Existing whole-disk GPT image containing the macOS target partition.
    #[arg(long, value_name = "IMAGE")]
    disk: PathBuf,
    /// Unique GPT partition GUID of the selected macOS target in --disk.
    #[arg(long, value_name = "GUID")]
    target: Guid,
    /// Whole-disk GPT image of the boot disk to protect from selection.
    #[arg(long, value_name = "IMAGE")]
    boot_disk: PathBuf,
    /// Swap partition size in bytes, including metadata. Defaults to 32 GiB.
    #[arg(long, default_value_t = DEFAULT_STORE_BYTES)]
    bytes: u64,
}

impl PlanImage {
    pub(super) fn run(&self) -> Result<String> {
        let mut report = format!(
            "Northwood Swap plan (read only)\n  target disk       {}\n  target partition  {}\n  boot disk         {}\n  requested         {}\n",
            self.disk.display(),
            self.target,
            self.boot_disk.display(),
            capacity(self.bytes),
        );
        if let Err(error) = self.describe(&mut report) {
            return Err(Reported {
                report,
                summary: format!("{error:#}"),
            }
            .into());
        }
        Ok(report)
    }

    fn describe(&self, report: &mut String) -> Result<()> {
        let boot_guid =
            preview::disk_guid(&self.boot_disk).context("checking the boot disk image")?;
        writeln!(report, "  boot disk GUID    {boot_guid}")?;
        let plan = preview::plan(&self.disk, self.target, boot_guid, self.bytes)?;
        writeln!(report, "  target disk GUID  {}", plan.disk_guid)?;
        writeln!(
            report,
            "  boot disk check   disk GUID differs from the supplied boot disk"
        )?;
        writeln!(report, "  logical sector    {} bytes", plan.block_bytes)?;
        writeln!(
            report,
            "  macOS target LBAs {}..={}",
            plan.target.first_lba, plan.target.last_lba
        )?;
        writeln!(
            report,
            "  macOS region      {} (setup minimum {})",
            capacity(plan.target_bytes),
            capacity(DEFAULT_MIN_APPLE_BYTES),
        )?;
        writeln!(
            report,
            "  swap required     {}",
            capacity(plan.required_bytes)
        )?;
        writeln!(
            report,
            "  largest GPT gap   {} (1 MiB aligned)",
            capacity(plan.largest_gap_bytes)
        )?;
        let partition = &plan.placement()?.partition;
        writeln!(
            report,
            "  proposed GPT LBAs {}..={}",
            partition.first_lba, partition.last_lba
        )?;
        writeln!(
            report,
            "  swap slots        {} ({})",
            plan.slots,
            capacity(plan.slots * SLOT_BYTES)
        )?;
        Ok(())
    }
}

fn capacity(bytes: u64) -> String {
    format!("{bytes} bytes ({})", bytes_text(bytes))
}

#[derive(Args)]
pub(super) struct ApplyImage {
    #[command(flatten)]
    selection: PlanImage,
}

impl ApplyImage {
    pub(super) fn run(&self) -> Result<String> {
        apply::run(&self.selection)
    }
}

#[derive(Args)]
pub(super) struct PrepareVolumeImage {
    /// Existing raw FAT32 installer data partition image.
    #[arg(long, value_name = "IMAGE")]
    volume: PathBuf,
    /// Read-only raw FAT32 ESP image with the completed setup record.
    #[arg(long, value_name = "IMAGE")]
    esp_image: PathBuf,
    /// New raw FAT32 data image. An existing path is refused.
    #[arg(long, value_name = "IMAGE")]
    output: PathBuf,
    /// Requested swap bytes, including metadata. Defaults to the ESP's policy (32 GiB).
    #[arg(long)]
    bytes: Option<u64>,
}

impl PrepareVolumeImage {
    pub(super) fn run(&self) -> Result<String> {
        prepare_volume::run(self)
    }
}

#[derive(Args)]
pub(super) struct CreateImage {
    /// The new GPT image. An existing path is refused.
    #[arg(long)]
    image: PathBuf,
    /// Swap partition size in bytes, including metadata.
    #[arg(long, default_value_t = DEFAULT_STORE_BYTES)]
    bytes: u64,
    /// Logical sector size for the image.
    #[arg(long, default_value_t = 512)]
    block_bytes: u64,
}

impl CreateImage {
    pub(super) fn run(&self) -> Result<String> {
        ensure!(
            matches!(self.block_bytes, 512 | 4096),
            "a swap image needs a 512- or 4096-byte logical block"
        );
        let blocks = self.bytes.div_ceil(self.block_bytes);
        let disk = Disk {
            block_bytes: self.block_bytes,
            blocks: blocks
                .checked_add(2 * ALIGNMENT / self.block_bytes)
                .context("swap image capacity overflows")?,
        };
        let image_bytes = disk
            .blocks
            .checked_mul(disk.block_bytes)
            .context("swap image size overflows")?;
        let mut table = Table {
            disk_guid: random_guid()?,
            partitions: Vec::new(),
        };
        let planned = partition::plan(
            &table,
            disk.first_usable()..=disk.last_usable(),
            disk.block_bytes,
            self.bytes,
            random_guid()?,
        )
        .map_err(|error| anyhow!("{error}"))?;
        let first = planned.partition.first_lba;
        table.partitions.push(planned.partition);
        let parent = self
            .image
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
        let file = temporary.as_file_mut();
        file.set_len(image_bytes)?;
        file.seek(SeekFrom::Start(first * disk.block_bytes))?;
        file.write_all(&planned.header.encode())?;
        for write in gpt::encode(disk, &table).map_err(|error| anyhow!("{error}"))? {
            file.seek(SeekFrom::Start(write.lba * disk.block_bytes))?;
            file.write_all(&write.bytes)?;
        }
        file.sync_all()?;
        temporary
            .persist_noclobber(&self.image)
            .with_context(|| format!("cannot publish swap image {}", self.image.display()))?;
        Ok(summary(planned.header, &self.image.display().to_string()))
    }
}

pub(super) fn random_guid() -> Result<Guid> {
    let mut bytes = [0; 16];
    getrandom::getrandom(&mut bytes)
        .map_err(|error| anyhow!("cannot make a swap GUID: {error}"))?;
    Ok(Guid::from_entropy(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use stela_runtime::pc::swap::MIN_STORE_BYTES;

    #[test]
    fn a_new_sparse_swap_image_is_inspected_and_never_replaces_an_existing_file() {
        let directory = tempfile::tempdir().unwrap();
        for block_bytes in [512, 4096] {
            let path = directory.path().join(format!("swap-{block_bytes}.img"));
            let create = CreateImage {
                image: path.clone(),
                bytes: MIN_STORE_BYTES,
                block_bytes,
            };
            assert!(create.run().unwrap().contains("slots 1"));
            let inspect = Inspect {
                disk: Some(path.clone()),
                partition: None,
                volume: None,
                block_bytes: None,
                max_bytes: None,
            };
            let before = inspect.run().unwrap();
            assert!(before.contains("state Idle"));
            assert!(create.run().is_err());
            assert_eq!(inspect.run().unwrap(), before);
        }
    }

    #[test]
    fn setup_routes_swap_commands_without_an_installer_usb() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("swap.img");
        let image = path.to_str().unwrap();
        let create = crate::Cli::try_parse_from([
            "stela",
            "setup",
            "swap",
            "create-image",
            "--image",
            image,
            "--bytes",
            &MIN_STORE_BYTES.to_string(),
        ])
        .unwrap();
        create.command.run().unwrap();
        let inspect =
            crate::Cli::try_parse_from(["stela", "setup", "swap", "inspect", "--disk", image])
                .unwrap();
        inspect.command.run().unwrap();
        assert!(crate::Cli::try_parse_from(["stela", "setup", "swap", "inspect"]).is_err());
    }

    #[test]
    fn swap_plan_requires_an_explicit_target_and_boot_disk_image() {
        let target = "1f71f26e-8b97-45ac-8a9a-67edc9b00e82";
        let args = ["stela", "setup", "swap", "plan", "--disk", "target.img"];
        assert!(crate::Cli::try_parse_from(args).is_err());
        assert!(crate::Cli::try_parse_from(args.into_iter().chain(["--target", target])).is_err());
        let parsed = crate::Cli::try_parse_from(args.into_iter().chain([
            "--target",
            target,
            "--boot-disk",
            "boot.img",
        ]))
        .unwrap();
        let crate::Command::Setup(setup) = parsed.command else {
            panic!("expected a setup command");
        };
        let crate::setup::Action::PlanSwapImage(plan) = setup.request().action else {
            panic!("expected a swap plan");
        };
        assert_eq!(plan.disk, Path::new("target.img"));
        assert_eq!(plan.boot_disk, Path::new("boot.img"));
        assert_eq!(plan.target, Guid::parse(target).unwrap());
        assert_eq!(plan.bytes, DEFAULT_STORE_BYTES);
    }
}
