// SPDX-License-Identifier: GPL-2.0-or-later

//! Build a new data image from an existing installer and its ESP policy.
use super::{PrepareVolumeImage, capacity, fat_image, preview, random_guid, summary};
use crate::setup::{
    nwconfig_publish::default_swap_policy,
    record::{RECORD_KIND, RECORD_SCHEMA},
};
use anyhow::{Context, Result, anyhow, bail, ensure};
use northwood_config::{Swap, SwapLocation};
use serde::Deserialize;
use std::{
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom, Write},
    path::Path,
};
use stela_formats::payload::ReadAt;
use stela_loader::mount_interface::layout::{self, Layout};
use stela_runtime::pc::fat32::{self, Fat32, FatError};

pub(super) fn run(request: &PrepareVolumeImage) -> Result<String> {
    let mut esp = read_locked(&request.esp_image)?;
    let (layout, policy) = setup_policy(&mut esp)
        .with_context(|| format!("checking setup on {}", request.esp_image.display()))?;
    let requested = request.bytes.unwrap_or(policy.bytes);
    let mut source = read_locked(&request.volume)?;
    let prepared = stage_image(&mut source, &request.output, |stage| {
        fat_image::prepare(
            stage,
            layout,
            requested,
            policy.volume_max_bytes,
            random_guid()?,
        )
    })?;
    let actual = prepared.metadata.config.bytes();
    let mut report = format!(
        "Prepared FAT32 data image {}\n  setup record      {} on {}\n  requested         {}\n  FAT32 policy cap  {}\n  free before       {}\n  sector / cluster  {} / {} bytes\n  prepared          {}\n  store             {}\n",
        request.output.display(),
        layout.esp_path(layout::RECORD),
        request.esp_image.display(),
        capacity(requested),
        capacity(policy.volume_max_bytes),
        capacity(prepared.before.free_bytes),
        prepared.before.block_bytes,
        prepared.before.cluster_bytes,
        capacity(actual),
        if prepared.kept {
            "kept existing verified store"
        } else {
            "allocated and verified"
        },
    );
    if actual < requested {
        report.push_str(&format!(
            "  shortfall         {}\n",
            capacity(requested - actual)
        ));
    }
    let capped = requested.min(policy.volume_max_bytes);
    if requested > capped {
        report.push_str(&format!(
            "  policy reduction  {}\n",
            capacity(requested - capped)
        ));
    }
    if actual < capped {
        report.push_str(&format!(
            "  space reduction   {} (available aligned clusters)\n",
            capacity(capped - actual)
        ));
    }
    report.push_str(&summary(
        prepared.metadata.header,
        &layout.data_path(layout::DATA_SWAP),
    ));
    Ok(report)
}

fn read_locked(path: &Path) -> Result<File> {
    let file = preview::open_regular(path, false)?;
    file.try_lock_shared()
        .with_context(|| format!("cannot lock image {}", path.display()))?;
    Ok(file)
}

#[derive(Deserialize)]
struct SetupIdentity {
    kind: String,
    schema_version: u32,
    complete: bool,
    #[serde(default)]
    volume_swap: Option<crate::setup::volume_swap::Store>,
}

fn setup_policy(file: &mut File) -> Result<(Layout, Swap)> {
    let bytes = file.metadata()?.len();
    let mut reader = Reader { file, bytes };
    let mut fat = Fat32::open(&mut reader).map_err(|error| anyhow!("{error}"))?;
    let [preferred, fallback] = Layout::PREFERRED;
    let modern = find(&mut fat, &preferred.esp_path(layout::RECORD))?;
    let legacy = find(&mut fat, &fallback.esp_path(layout::RECORD))?;
    let (layout, record) = match (modern, legacy) {
        (Some(record), None) => (preferred, record),
        (None, Some(record)) => (fallback, record),
        (None, None) => bail!("the ESP image has no setup record"),
        (Some(_), Some(_)) => bail!("the ESP image has both Northwood and Stela setup records"),
    };
    let plist = find(&mut fat, &Layout::Northwood.esp_path(layout::CONFIG))?;
    let reader = fat.into_inner();
    let record: SetupIdentity = serde_json::from_slice(&contents(reader, &record, 1 << 20)?)
        .context("reading setup.json")?;
    ensure!(
        record.kind == RECORD_KIND && record.schema_version == RECORD_SCHEMA && record.complete,
        "the ESP image must contain a completed setup record of schema {RECORD_SCHEMA}"
    );
    ensure!(
        record.volume_swap.is_none(),
        "the setup record already owns the FAT32 swap store; use setup repair to maintain it"
    );
    let policy = match plist {
        Some(plist) => {
            let bytes = contents(reader, &plist, northwood_config::plist::MAX_BYTES)?;
            let parsed = northwood_config::parse(&bytes).context("reading nwconfig.plist")?;
            default_swap_policy(&parsed)
        }
        None => Swap::default(),
    };
    ensure!(
        policy.enabled
            && matches!(
                policy.location,
                SwapLocation::Auto | SwapLocation::VolumeFile
            ),
        "the ESP configuration does not enable FAT32 volume swap"
    );
    Ok((layout, policy))
}

fn find<R: ReadAt>(fat: &mut Fat32<R>, path: &str) -> Result<Option<fat32::File>> {
    match fat.file(path) {
        Ok(file) => Ok(Some(file)),
        Err(FatError::Missing(_)) => Ok(None),
        Err(error) => Err(anyhow!("{path}: {error}")),
    }
}

fn contents(reader: &mut Reader<'_>, file: &fat32::File, limit: usize) -> Result<Vec<u8>> {
    ensure!(
        file.bytes <= limit as u64,
        "ESP setup metadata exceeds {limit} bytes"
    );
    let mut bytes = vec![0; file.bytes as usize];
    let mut remaining = bytes.as_mut_slice();
    for extent in &file.extents {
        let length = remaining.len().min(extent.bytes as usize);
        let (buffer, tail) = remaining.split_at_mut(length);
        reader.read_at(extent.offset, buffer)?;
        remaining = tail;
    }
    ensure!(
        remaining.is_empty(),
        "ESP setup metadata has an incomplete FAT chain"
    );
    Ok(bytes)
}

struct Reader<'a> {
    file: &'a mut File,
    bytes: u64,
}

impl ReadAt for Reader<'_> {
    type Error = io::Error;

    fn len(&self) -> u64 {
        self.bytes
    }

    fn read_at(&mut self, offset: u64, bytes: &mut [u8]) -> io::Result<()> {
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(bytes)
    }
}

fn stage_image<T>(
    source: &mut File,
    output: &Path,
    prepare: impl FnOnce(&tempfile::NamedTempFile) -> Result<T>,
) -> Result<T> {
    match fs::symlink_metadata(output) {
        Ok(_) => bail!("{} already exists", output.display()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).with_context(|| format!("checking {}", output.display())),
    }
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut stage = tempfile::NamedTempFile::new_in(parent)?;
    copy_sparse(source, stage.as_file_mut()).context("copying the installer data image")?;
    let prepared = prepare(&stage)?;
    stage.as_file().sync_all()?;
    stage
        .persist_noclobber(output)
        .with_context(|| format!("cannot publish {}", output.display()))?;
    let _ = File::open(parent).and_then(|directory| directory.sync_all());
    Ok(prepared)
}

fn copy_sparse(source: &mut File, stage: &mut File) -> Result<()> {
    let before = source.metadata()?;
    stage.set_len(before.len())?;
    source.seek(SeekFrom::Start(0))?;
    let mut buffer = vec![0; 1 << 20];
    let zeros = vec![0; buffer.len()];
    let mut offset = 0;
    while offset < before.len() {
        let length = (before.len() - offset).min(buffer.len() as u64) as usize;
        let bytes = &mut buffer[..length];
        source.read_exact(bytes)?;
        if *bytes != zeros[..length] {
            stage.seek(SeekFrom::Start(offset))?;
            stage.write_all(bytes)?;
        }
        offset += length as u64;
    }
    let after = source.metadata()?;
    ensure!(
        before.len() == after.len() && before.modified()? == after.modified()?,
        "the source data image changed while it was being copied"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_failure_does_not_publish_or_change_the_source() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("source.img");
        let output = directory.path().join("output.img");
        fs::write(&input, b"original restore bytes").unwrap();
        let mut source = read_locked(&input).unwrap();
        let error = stage_image(&mut source, &output, |stage| -> Result<()> {
            fs::write(stage.path(), b"incomplete staged changes")?;
            bail!("injected allocation failure")
        })
        .unwrap_err();
        assert!(error.to_string().contains("injected allocation failure"));
        assert!(!output.exists());
        assert_eq!(fs::read(&input).unwrap(), b"original restore bytes");
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);

        fs::write(&output, b"previous output").unwrap();
        let error = stage_image(&mut source, &output, |_| -> Result<()> {
            panic!("a pre-existing output must never be staged over")
        })
        .unwrap_err();
        assert!(error.to_string().contains("already exists"));
        assert_eq!(fs::read(&output).unwrap(), b"previous output");
    }
}
