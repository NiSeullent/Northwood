// SPDX-License-Identifier: GPL-2.0-or-later
//! Read-only inspection of a Northwood Swap store on the selected FAT32 boot
//! volume. The FAT32 reader validates each requested chain. It cannot prove
//! exclusive cluster ownership against every other file and directory or
//! provide an exclusive volume lease. The separate layout check verifies
//! allocation across the volume and returns block locations without I/O
//! authority.

use super::{EXTENT_BYTES, HEADER_BYTES, Header, Layout, MIN_STORE_BYTES, SLOT_BYTES};
use crate::pc::fat32::{Fat32, FatError, File, MAX_EXTENTS};
use alloc::{format, string::String, vec::Vec};
use core::fmt;
use stela_formats::payload::{ReadAt, pieces::Extent};

mod boot;
pub use boot::{BootDataError, BootEsp, boot_data};

pub(super) const BASIC_DATA: crate::pc::gpt::Guid = crate::pc::gpt::Guid([
    0xa2, 0xa0, 0xd0, 0xeb, 0xe5, 0xb9, 0x33, 0x44, 0x87, 0xc0, 0x68, 0xb6, 0xb7, 0x26, 0x99, 0xc7,
]);

pub const DIRECTORY: &str = "stela/swap";
pub const CONFIG_PATH: &str = "stela/swap/swap.cfg";
const NORTHWOOD_DIRECTORY: &str = "northwood/swap";
const NORTHWOOD_CONFIG_PATH: &str = "northwood/swap/swap.cfg";
const CONFIG_LIMIT: usize = b"bytes ".len() + 20 + b"\r\n".len();
const DIRECTORY_SCAN_BYTES: u64 = 2 << 20;
const MAX_FILES: u16 = 10_000;
const MAX_STORE_BYTES: u64 = MAX_FILES as u64 * EXTENT_BYTES;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    bytes: u64,
    layout: Layout,
}

impl Config {
    /// Parse the single `bytes N` record, with an optional LF or CRLF.
    pub fn parse(input: &[u8], block_bytes: u64, maximum_bytes: u64) -> Result<Self, Error> {
        if input.len() > CONFIG_LIMIT {
            return Err(Error::Config);
        }
        let line = input
            .strip_suffix(b"\r\n")
            .or_else(|| input.strip_suffix(b"\n"))
            .unwrap_or(input);
        let digits = line.strip_prefix(b"bytes ").ok_or(Error::Config)?;
        if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
            return Err(Error::Config);
        }
        let bytes = core::str::from_utf8(digits)
            .ok()
            .and_then(|number| number.parse().ok())
            .ok_or(Error::Config)?;
        Self::new(bytes, block_bytes, maximum_bytes)
    }

    /// Validate a capacity supplied by setup or read from `swap.cfg`.
    pub fn new(bytes: u64, block_bytes: u64, maximum_bytes: u64) -> Result<Self, Error> {
        if !matches!(block_bytes, 512 | 1024 | 2048 | 4096) {
            return Err(Error::BlockSize(block_bytes));
        }
        let maximum = maximum_bytes.min(MAX_STORE_BYTES);
        if bytes < MIN_STORE_BYTES || bytes > maximum || !bytes.is_multiple_of(block_bytes) {
            return Err(Error::Capacity { bytes, maximum });
        }
        Ok(Self {
            bytes,
            layout: Layout::for_store(bytes).map_err(Error::Header)?,
        })
    }

    pub fn bytes(self) -> u64 {
        self.bytes
    }

    /// The numbered files in logical store order, with exactly the last file
    /// shortened when the capacity is not a whole multiple of 1 GiB.
    pub fn files(self) -> impl ExactSizeIterator<Item = ExtentFile> {
        let count = self.bytes.div_ceil(EXTENT_BYTES) as u16;
        (0..count).map(move |index| ExtentFile {
            index,
            bytes: (self.bytes - u64::from(index) * EXTENT_BYTES).min(EXTENT_BYTES),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtentFile {
    index: u16,
    bytes: u64,
}

impl ExtentFile {
    pub fn path(self) -> String {
        self.path_in(DIRECTORY)
    }

    fn path_in(self, directory: &str) -> String {
        format!("{directory}/{:04}.swp", self.index)
    }

    pub fn bytes(self) -> u64 {
        self.bytes
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Metadata {
    pub config: Config,
    pub header: Header,
}

/// Validated logical order of an observed FAT32 swap store. A caller
/// intending to write must hold exclusive access to the same media from
/// before verification until all writes and DMA have finished.
pub struct VerifiedLayout {
    metadata: Metadata,
    runs: Vec<(u64, Extent)>,
}

impl VerifiedLayout {
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Partition-relative byte ranges of the header, in store order.
    pub fn header_runs(&self) -> impl Iterator<Item = Extent> + '_ {
        self.runs_at(0, HEADER_BYTES as u64)
    }

    /// Partition-relative byte ranges of one 16 KiB slot, in store order.
    /// Every range starts and ends on a logical device block boundary.
    pub fn slot_runs(&self, slot: u64) -> Option<impl Iterator<Item = Extent> + '_> {
        let start = self.metadata.header.layout().slot_offset(slot)?;
        Some(self.runs_at(start, SLOT_BYTES))
    }

    pub(super) fn runs_at(&self, start: u64, bytes: u64) -> impl Iterator<Item = Extent> + '_ {
        let end = start + bytes;
        let first = self
            .runs
            .partition_point(|(logical, _)| *logical <= start)
            .saturating_sub(1);
        self.runs[first..]
            .iter()
            .take_while(move |(logical, _)| *logical < end)
            .map(move |&(logical, run)| {
                let skip = start.saturating_sub(logical);
                Extent {
                    offset: run.offset + skip,
                    bytes: run.bytes.min(end - logical) - skip,
                }
            })
    }
}

/// Verify allocation for the whole FAT32 volume before retaining the swap
/// placement. Like `inspect`, this accepts only a read interface.
pub fn verify_layout<R: ReadAt>(
    mut reader: R,
    block_bytes: u64,
    maximum_bytes: u64,
) -> Result<VerifiedLayout, Error> {
    let store = inspect_store(&mut reader, block_bytes, maximum_bytes, true)?;
    let mut physical = store.runs.clone();
    physical.push(store.config_cluster);
    validate_no_overlap(&mut physical)?;
    drop(physical);
    let metadata = store.metadata()?;
    Fat32::open(reader)
        .and_then(|mut volume| volume.verify_allocation())
        .map_err(Error::Fat)?;
    let mut logical = 0;
    let runs = store
        .runs
        .into_iter()
        .map(|run| {
            let start = logical;
            logical += run.bytes;
            (start, run)
        })
        .collect();
    Ok(VerifiedLayout { metadata, runs })
}

/// Inspect an already selected boot volume with the caller's capacity limit.
/// Reject aliases between swap files and with the configuration file, and
/// validate the header against the exact capacity described by `swap.cfg`.
pub fn inspect<R: ReadAt>(
    mut reader: R,
    block_bytes: u64,
    maximum_bytes: u64,
) -> Result<Metadata, Error> {
    let mut store = inspect_store(&mut reader, block_bytes, maximum_bytes, false)?;
    store.runs.push(store.config_cluster);
    validate_no_overlap(&mut store.runs)?;
    store.metadata()
}

struct Store {
    config: Config,
    config_cluster: Extent,
    header: [u8; HEADER_BYTES],
    runs: Vec<Extent>,
}

impl Store {
    fn metadata(&self) -> Result<Metadata, Error> {
        let header = Header::decode(&self.header, self.config.bytes).map_err(Error::Header)?;
        if header.layout() != self.config.layout {
            return Err(Error::HeaderCapacity);
        }
        Ok(Metadata {
            config: self.config,
            header,
        })
    }
}

fn inspect_store<R: ReadAt>(
    reader: &mut R,
    block_bytes: u64,
    maximum_bytes: u64,
    require_unique: bool,
) -> Result<Store, Error> {
    let volume_bytes = reader.len();
    let mut volume = Fat32::open(&mut *reader).map_err(Error::Fat)?;
    let cluster_bytes = volume.cluster_bytes();
    let (directory, cfg) = config_file(&mut volume, require_unique)?;
    let [cfg_extent] = cfg.extents.as_slice() else {
        return Err(Error::Config);
    };
    let mut input = [0; CONFIG_LIMIT];
    let input = &mut input[..cfg.bytes as usize];
    read_prefix(volume.into_inner(), &cfg.extents, input)?;
    let config = Config::parse(input, block_bytes, maximum_bytes.min(volume_bytes))?;

    let mut volume = Fat32::open(&mut *reader).map_err(Error::Fat)?;
    let mut runs = Vec::new();
    for expected in config.files() {
        let file = volume
            .file_bounded_by_name(
                &expected.path_in(directory),
                DIRECTORY_SCAN_BYTES,
                expected.bytes..=expected.bytes,
                MAX_EXTENTS - runs.len(),
                require_unique,
            )
            .map_err(|error| match error {
                FatError::FileSize { actual, .. } => Error::FileSize {
                    index: expected.index,
                    expected: expected.bytes,
                    actual,
                },
                FatError::Fragmented(_) | FatError::ExtentLimit(..) => Error::Fragmented,
                error => Error::Fat(error),
            })?;
        validate_file(&file, block_bytes, volume_bytes)?;
        runs.extend(file.extents);
    }

    let mut header = [0; HEADER_BYTES];
    read_prefix(volume.into_inner(), &runs, &mut header)?;
    // The final byte of the small config file still owns the whole cluster.
    let config_cluster = Extent {
        offset: cfg_extent.offset,
        bytes: cluster_bytes,
    };
    Ok(Store {
        config,
        config_cluster,
        header,
        runs,
    })
}

fn config_file<R: ReadAt>(
    volume: &mut Fat32<R>,
    require_unique: bool,
) -> Result<(&'static str, File), Error> {
    let mut find = |path| {
        volume
            .file_bounded_by_name(
                path,
                DIRECTORY_SCAN_BYTES,
                1..=CONFIG_LIMIT as u64,
                1,
                require_unique,
            )
            .map_err(|error| match error {
                FatError::FileSize { .. } => Error::Config,
                error => Error::Fat(error),
            })
    };
    match find(NORTHWOOD_CONFIG_PATH) {
        Ok(file) => Ok((NORTHWOOD_DIRECTORY, file)),
        Err(Error::Fat(FatError::Missing(_))) => find(CONFIG_PATH).map(|file| (DIRECTORY, file)),
        Err(error) => Err(error),
    }
}

fn validate_file(file: &File, block_bytes: u64, volume_bytes: u64) -> Result<(), Error> {
    let mut total = 0u64;
    for &run in &file.extents {
        if run.bytes == 0
            || !run.offset.is_multiple_of(block_bytes)
            || !run.bytes.is_multiple_of(block_bytes)
            || run
                .offset
                .checked_add(run.bytes)
                .is_none_or(|end| end > volume_bytes)
        {
            return Err(Error::Run);
        }
        total = total.checked_add(run.bytes).ok_or(Error::Run)?;
    }
    if total != file.bytes {
        return Err(Error::Run);
    }
    Ok(())
}

fn validate_no_overlap(runs: &mut [Extent]) -> Result<(), Error> {
    runs.sort_unstable_by_key(|run| run.offset);
    for [previous, next] in runs.array_windows::<2>() {
        if previous.offset + previous.bytes > next.offset {
            return Err(Error::Overlap(next.offset));
        }
    }
    Ok(())
}

fn read_prefix<R: ReadAt>(
    reader: &mut R,
    runs: &[Extent],
    mut buffer: &mut [u8],
) -> Result<(), Error> {
    for run in runs {
        if buffer.is_empty() {
            break;
        }
        let count = (buffer.len() as u64).min(run.bytes) as usize;
        let (chunk, rest) = buffer.split_at_mut(count);
        reader
            .read_at(run.offset, chunk)
            .map_err(|error| Error::Read(format!("{error:?}")))?;
        buffer = rest;
    }
    if buffer.is_empty() {
        Ok(())
    } else {
        Err(Error::Run)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Config,
    BlockSize(u64),
    Capacity {
        bytes: u64,
        maximum: u64,
    },
    FileSize {
        index: u16,
        expected: u64,
        actual: u64,
    },
    Fragmented,
    Run,
    Overlap(u64),
    HeaderCapacity,
    Fat(FatError),
    Header(super::Error),
    Read(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config => f.write_str("invalid Northwood Swap volume configuration"),
            Self::BlockSize(bytes) => write!(f, "unsupported Northwood Swap block size {bytes}"),
            Self::Capacity { bytes, maximum } => write!(
                f,
                "Northwood Swap volume size {bytes} must be block-aligned and between {MIN_STORE_BYTES} and {maximum} bytes"
            ),
            Self::FileSize {
                index,
                expected,
                actual,
            } => write!(
                f,
                "Northwood Swap extent {index:04}.swp has {actual} bytes, expected {expected}"
            ),
            Self::Fragmented => f.write_str("Northwood Swap volume store has too many runs"),
            Self::Run => f.write_str("invalid Northwood Swap volume run"),
            Self::Overlap(offset) => write!(f, "Northwood Swap files overlap at byte {offset}"),
            Self::HeaderCapacity => {
                f.write_str("Northwood Swap header capacity differs from swap.cfg")
            }
            Self::Fat(error) => error.fmt(f),
            Self::Header(error) => error.fmt(f),
            Self::Read(what) => write!(f, "Northwood Swap volume read failed: {what}"),
        }
    }
}

impl core::error::Error for Error {}

#[cfg(test)]
mod tests;
