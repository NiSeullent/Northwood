// SPDX-License-Identifier: GPL-2.0-or-later

//! Plain-text loader snapshots kept on the visible FAT32 data partition.
use super::Record;
use crate::efi;
use anyhow::{Context, Result};
use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};
use stela_loader::mount_interface::{
    VOLUME_LOG_BYTES, VOLUME_LOG_EMPTY, VOLUME_LOG_SLOTS,
    layout::{self, Layout},
    volume_log_sequence,
};

fn data_volume(record: &Record) -> Option<&Path> {
    (record.volume != record.esp && record.volume.is_dir()).then_some(record.volume.as_path())
}

fn directory(record: &Record) -> Option<PathBuf> {
    data_volume(record).map(|volume| volume.join(record.layout.data_path(layout::DATA_LOGS)))
}

fn slot(path: &Path) -> Option<File> {
    let metadata = fs::symlink_metadata(path).ok()?;
    (metadata.is_file() && metadata.len() == VOLUME_LOG_BYTES as u64)
        .then(|| File::open(path).ok())
        .flatten()
}

/// Allocate only missing slots. Older Stela loaders ignore these files.
pub(super) fn prepare(record: &Record) -> Result<()> {
    let Some(directory) = directory(record) else {
        return Ok(());
    };
    fs::create_dir_all(&directory)
        .with_context(|| format!("cannot prepare {}", directory.display()))?;
    let mut empty = VOLUME_LOG_EMPTY.as_bytes().to_vec();
    empty.resize(VOLUME_LOG_BYTES, b' ');
    for name in VOLUME_LOG_SLOTS {
        let path = directory.join(name);
        if !path.exists() {
            efi::write_atomic(&path, &empty, true)
                .with_context(|| format!("cannot prepare {}", path.display()))?;
        }
    }
    Ok(())
}

pub(super) fn latest(record: &Record, require_output: bool) -> Option<(PathBuf, Vec<u8>)> {
    latest_at(data_volume(record)?, record.layout, require_output)
}

fn latest_at(volume: &Path, layout: Layout, require_output: bool) -> Option<(PathBuf, Vec<u8>)> {
    let read = |layout: Layout| {
        latest_in(
            &volume.join(layout.data_path(layout::DATA_LOGS)),
            require_output,
        )
    };
    read(layout).or_else(|| {
        Layout::PREFERRED
            .into_iter()
            .filter(|candidate| *candidate != layout)
            .find_map(read)
    })
}

fn latest_in(directory: &Path, require_output: bool) -> Option<(PathBuf, Vec<u8>)> {
    let (path, _, bytes) = VOLUME_LOG_SLOTS
        .iter()
        .filter_map(|name| {
            let path = directory.join(name);
            let mut bytes = Vec::new();
            slot(&path)?
                .take(VOLUME_LOG_BYTES as u64)
                .read_to_end(&mut bytes)
                .ok()?;
            let sequence = volume_log_sequence(&bytes)?;
            if require_output
                && !bytes
                    .splitn(2, |&byte| byte == b'\n')
                    .nth(1)
                    .is_some_and(|body| body.iter().any(|byte| !byte.is_ascii_whitespace()))
            {
                return None;
            }
            (sequence != 0).then_some((path, sequence, bytes))
        })
        .max_by_key(|(_, sequence, _)| *sequence)?;
    Some((path, bytes))
}

pub(super) fn owned_slots(record: &Record) -> Vec<PathBuf> {
    let Some(directory) = directory(record) else {
        return Vec::new();
    };
    VOLUME_LOG_SLOTS
        .iter()
        .filter_map(|name| {
            let path = directory.join(name);
            let mut header = [0; 128];
            slot(&path)?.read_exact(&mut header).ok()?;
            volume_log_sequence(&header).map(|_| path)
        })
        .collect()
}

pub(super) fn remove(paths: Vec<PathBuf>) -> Result<()> {
    for path in &paths {
        fs::remove_file(path)?;
    }
    if let Some(directory) = paths.first().and_then(|path| path.parent())
        && fs::read_dir(directory).is_ok_and(|mut entries| entries.next().is_none())
    {
        fs::remove_dir(directory)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_slot(volume: &Path, layout: Layout, slot: usize, sequence: u64, line: &str) {
        let directory = volume.join(layout.data_path(layout::DATA_LOGS));
        fs::create_dir_all(&directory).unwrap();
        let mut bytes = format!("STELA loader boot log #{sequence} test\n{line}\n").into_bytes();
        bytes.resize(VOLUME_LOG_BYTES, b' ');
        fs::write(directory.join(VOLUME_LOG_SLOTS[slot]), bytes).unwrap();
    }

    #[test]
    fn prefers_the_latest_northwood_slot_and_falls_back_to_stela() {
        let temporary = tempfile::tempdir().unwrap();
        let volume = temporary.path();
        let latest = |layout, require_output| {
            let (_, bytes) = latest_at(volume, layout, require_output).unwrap();
            String::from_utf8(bytes).unwrap()
        };
        write_slot(volume, Layout::Legacy, 0, 30, "legacy log");
        write_slot(volume, Layout::Northwood, 0, 0, "empty slot");
        assert!(latest(Layout::Northwood, false).contains("legacy log"));
        write_slot(volume, Layout::Northwood, 1, 3, "older Northwood log");
        write_slot(volume, Layout::Northwood, 2, 4, "latest Northwood log");
        write_slot(volume, Layout::Northwood, 3, 5, "");
        let text = latest(Layout::Northwood, true);
        assert!(text.contains("latest Northwood log"));
        assert!(!text.contains("legacy log"));
        assert!(latest(Layout::Northwood, false).starts_with("STELA loader boot log #5"));
        assert!(latest(Layout::Legacy, false).contains("legacy log"));
    }
}
