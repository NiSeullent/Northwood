// SPDX-License-Identifier: GPL-2.0-or-later

//! Plain-text loader snapshots kept on the visible FAT32 data partition.
use super::Record;
#[cfg(not(target_os = "macos"))]
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
        prepare_slot(&path, &empty)
            .with_context(|| format!("cannot prepare {}", path.display()))?;
    }
    // FAT may refuse to sync a directory. Its files were already synced.
    #[cfg(target_os = "macos")]
    let _ = File::open(&directory).and_then(|directory| directory.sync_all());
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn prepare_slot(path: &Path, empty: &[u8]) -> Result<()> {
    if !path.exists() {
        efi::write_atomic(path, empty, true)?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn prepare_slot(path: &Path, empty: &[u8]) -> Result<()> {
    darwin::prepare(path, empty)?;
    Ok(())
}

#[cfg(target_os = "macos")]
mod darwin {
    use super::{File, Path, VOLUME_LOG_BYTES, VOLUME_LOG_EMPTY};
    use std::{
        fs::OpenOptions,
        io::{self, Read, Seek, SeekFrom, Write},
        os::unix::fs::OpenOptionsExt,
    };

    // The loader accepts the slot only after its first byte is committed.
    // An exact staged marker identifies a file that this path can finish.
    pub(super) const PENDING: &[u8] = b"\xa7TELA loader boot log #0 (unused)\n";

    pub(super) fn prepare(path: &Path, empty: &[u8]) -> io::Result<()> {
        // macOS FAT32 rejects renameatx_np(RENAME_EXCL). Create the slot
        // exclusively and publish its marker only after the space is allocated.
        let mut file = match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(mut file) => {
                file.lock()?;
                file.write_all(PENDING)?;
                file.sync_all()?;
                file
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                let Some(file) = pending(path)? else {
                    return Ok(());
                };
                file
            }
            Err(error) => return Err(error),
        };
        file.seek(SeekFrom::Start(VOLUME_LOG_EMPTY.len() as u64))?;
        file.write_all(&empty[VOLUME_LOG_EMPTY.len()..])?;
        file.sync_all()?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&empty[..1])?;
        file.sync_all()
    }

    fn pending(path: &Path) -> io::Result<Option<File>> {
        let Ok(metadata) = path.symlink_metadata() else {
            return Ok(None);
        };
        if !metadata.is_file() || metadata.len() > VOLUME_LOG_BYTES as u64 {
            return Ok(None);
        }
        let mut options = OpenOptions::new();
        options
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        // Existing arbitrary or unreadable paths were always left alone.
        let Ok(mut original) = options.open(path) else {
            return Ok(None);
        };
        if !has_pending_marker(&mut original)? {
            return Ok(None);
        }
        let mut file = options.write(true).open(path)?;
        file.lock()?;
        // Another setup may have finished after the first open. Check again
        // under the lock before touching any existing bytes.
        if !has_pending_marker(&mut file)? {
            return Ok(None);
        }
        let mut buffer = [0; 4096];
        let mut remaining = VOLUME_LOG_BYTES - PENDING.len();
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                return Ok(Some(file));
            }
            if read > remaining || buffer[..read].iter().any(|&byte| byte != b' ' && byte != 0) {
                return Ok(None);
            }
            remaining -= read;
        }
    }

    fn has_pending_marker(file: &mut File) -> io::Result<bool> {
        let metadata = file.metadata()?;
        if !metadata.is_file()
            || !(PENDING.len() as u64..=VOLUME_LOG_BYTES as u64).contains(&metadata.len())
        {
            return Ok(false);
        }
        let mut marker = [0; VOLUME_LOG_EMPTY.len()];
        match file.read_exact(&mut marker) {
            Ok(()) => Ok(marker == PENDING),
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
            Err(error) => Err(error),
        }
    }
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

    #[cfg(target_os = "macos")]
    fn empty_slot() -> Vec<u8> {
        let mut bytes = VOLUME_LOG_EMPTY.as_bytes().to_vec();
        bytes.resize(VOLUME_LOG_BYTES, b' ');
        bytes
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn darwin_slot_preparation_preserves_existing_paths() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join(VOLUME_LOG_SLOTS[0]);
        let empty = empty_slot();
        prepare_slot(&path, &empty).unwrap();
        assert_eq!(fs::read(&path).unwrap(), empty);

        let mut previous_boot = b"STELA loader boot log #5 test\nKept boot log\n".to_vec();
        previous_boot.resize(VOLUME_LOG_BYTES, b' ');
        fs::write(&path, &previous_boot).unwrap();
        prepare_slot(&path, &empty).unwrap();
        assert_eq!(fs::read(&path).unwrap(), previous_boot);

        let mut foreign = darwin::PENDING.to_vec();
        foreign.extend_from_slice(b"arbitrary existing bytes");
        for data in [
            b"".as_slice(),
            b"other log",
            &darwin::PENDING[..8],
            &foreign,
        ] {
            fs::write(&path, data).unwrap();
            prepare_slot(&path, &empty).unwrap();
            assert_eq!(fs::read(&path).unwrap(), data);
        }
        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        prepare_slot(&path, &empty).unwrap();
        assert!(path.is_dir());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn darwin_slot_preparation_finishes_recognized_interrupted_writes() {
        let temporary = tempfile::tempdir().unwrap();
        let empty = empty_slot();
        let lengths = [darwin::PENDING.len(), 4097, VOLUME_LOG_BYTES];
        for (name, length) in VOLUME_LOG_SLOTS.into_iter().zip(lengths) {
            let path = temporary.path().join(name);
            let mut pending = darwin::PENDING.to_vec();
            pending.resize(length, 0);
            assert_eq!(volume_log_sequence(&pending), None);
            fs::write(&path, pending).unwrap();
            prepare_slot(&path, &empty).unwrap();
            let finished = fs::read(&path).unwrap();
            assert_eq!(finished, empty);
            assert_eq!(volume_log_sequence(&finished), Some(0));
        }
    }

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
