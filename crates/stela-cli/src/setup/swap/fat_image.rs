// SPDX-License-Identifier: GPL-2.0-or-later

//! Provision a store on a private raw FAT32 image. GNU mtools owns file and
//! directory updates. dosfstools checks cluster ownership before replacement
//! and verifies the resulting filesystem without repairing it.
use anyhow::{Context, Result, anyhow, ensure};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
};
use stela_formats::payload::{ReadAt, gpt::Guid};
use stela_loader::mount_interface::layout::{self, Layout};
use stela_runtime::pc::{
    fat32::{Fat32, FatError},
    swap::{EXTENT_BYTES, Header, MIN_STORE_BYTES, State, volume},
};
use tempfile::NamedTempFile;

const FAT_MASK: u32 = 0x0fff_ffff;
const FAT_WINDOW: usize = 64 << 10;
const CONFIG: &str = "swap.cfg";

#[derive(Debug)]
pub(super) struct Capacity {
    pub(super) block_bytes: u64,
    pub(super) cluster_bytes: u64,
    pub(super) free_bytes: u64,
}

#[derive(Debug)]
pub(super) struct Prepared {
    pub(super) before: Capacity,
    pub(super) metadata: volume::Metadata,
    pub(super) kept: bool,
}

/// Free data-cluster capacity, excluding FAT metadata and occupied clusters.
pub(super) fn capacity(path: &Path) -> Result<Capacity> {
    ensure!(
        fs::symlink_metadata(path)?.is_file(),
        "{} is not a regular FAT32 image",
        path.display()
    );
    Ok(scan(&mut File::open(path)?)?.capacity)
}

/// Prepare a caller-owned staging image for later publication. Only files
/// named by a verified previous swap configuration may be removed.
pub(super) fn prepare(
    stage: &NamedTempFile,
    layout: Layout,
    requested_bytes: u64,
    max_bytes: u64,
    guid: Guid,
) -> Result<Prepared> {
    let tools = Tools::new(stage)?;
    let mut image = stage.as_file().try_clone()?;
    let before = capacity(stage.path())?;
    let block_bytes = before.block_bytes;
    let desired = requested_bytes.min(max_bytes);
    let desired = desired - desired % block_bytes;
    volume::Config::new(desired, block_bytes, max_bytes)?;

    let directory = layout.data_path(layout::DATA_SWAP);
    let config_path = format!("{directory}/{CONFIG}");
    let existing = previous_store(&mut image, layout, block_bytes)?;
    if let Some(store) = &existing {
        let listing = tools.require_directory(&directory)?;
        check_names(&listing, Some(store.config))?;
    }
    if let Some(existing) = existing.as_ref()
        && existing.config.bytes() == desired
        && existing.header.state() == State::Idle
    {
        tools.check_fat()?;
        tools.require_installer(layout)?;
        return Ok(Prepared {
            before,
            metadata: volume::Metadata {
                config: existing.config,
                header: existing.header,
            },
            kept: true,
        });
    }

    scan(&mut image)?.refresh_fsinfo(&mut image)?;
    tools.check_fat()?;
    tools.require_installer(layout)?;
    let listing = tools.ensure_directory(&directory)?;
    check_names(&listing, existing.as_ref().map(|store| store.config))?;
    if let Some(existing) = existing {
        for index in 0..existing.config.files().len() {
            tools.remove(&extent_path(&directory, index))?;
        }
        tools.remove(&config_path)?;
    }

    let mut payload = NamedTempFile::new_in(&tools.directory)?;
    reserve(&tools, &mut image, &payload, &config_path)?;
    let mut slots = 0;
    let configured = loop {
        let free = scan(&mut image)?.capacity;
        // The config's directory entry is reserved, but its final text
        // will occupy one data cluster.
        let available = desired.min(free.free_bytes.saturating_sub(free.cluster_bytes));
        ensure!(
            available >= MIN_STORE_BYTES,
            "the FAT32 image has {available} allocatable bytes after reserving swap metadata. Northwood Swap needs at least {MIN_STORE_BYTES}"
        );
        if available <= slots as u64 * EXTENT_BYTES {
            break available;
        }
        reserve(
            &tools,
            &mut image,
            &payload,
            &extent_path(&directory, slots),
        )?;
        slots += 1;
    };
    let config = volume::Config::new(configured, block_bytes, max_bytes)?;
    let header = Header::new(configured, guid)?;
    for (index, extent) in config.files().enumerate() {
        let prefix = if index == 0 {
            header.encode().to_vec()
        } else {
            Vec::new()
        };
        write_payload(payload.as_file_mut(), extent.bytes(), &prefix)?;
        tools.copy(&payload, &extent_path(&directory, index), true)?;
    }
    // Reserving the next slot can itself consume the last free directory
    // cluster. If that reduces capacity to a whole extent count, drop the
    // surplus placeholder. Other files in the directory are untouched.
    for index in config.files().len()..slots {
        tools.remove(&extent_path(&directory, index))?;
    }
    let text = format!("bytes {configured}\n");
    write_payload(payload.as_file_mut(), text.len() as u64, text.as_bytes())?;
    tools.copy(&payload, &config_path, true)?;
    scan(&mut image)?.refresh_fsinfo(&mut image)?;
    tools.check_fat()?;

    let metadata = inspect(&mut image, block_bytes, max_bytes)?;
    ensure!(
        metadata.config == config && metadata.header == header,
        "the FAT32 swap store did not read back as planned"
    );
    Ok(Prepared {
        before,
        metadata,
        kept: false,
    })
}

fn extent_path(directory: &str, index: usize) -> String {
    format!("{directory}/{index:04}.swp")
}

fn reserve(tools: &Tools<'_>, image: &mut File, source: &NamedTempFile, path: &str) -> Result<()> {
    let mut fat = Fat32::open(Image::new(image)?).map_err(|error| anyhow!("{error}"))?;
    ensure!(
        matches!(fat.file(path), Err(FatError::Missing(_))),
        "an unmanaged FAT32 entry or alias occupies {path}"
    );
    tools.copy(source, path, false)?;
    let mut fat = Fat32::open(Image::new(image)?).map_err(|error| anyhow!("{error}"))?;
    ensure!(
        fat.file(path).is_ok_and(|file| file.bytes == 0),
        "the directory entry reserved for {path} did not read back"
    );
    Ok(())
}

fn write_payload(file: &mut File, bytes: u64, prefix: &[u8]) -> Result<()> {
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(prefix)?;
    file.set_len(bytes)?;
    Ok(())
}

fn previous_store(
    file: &mut File,
    layout: Layout,
    block_bytes: u64,
) -> Result<Option<volume::Metadata>> {
    let other = Layout::PREFERRED
        .into_iter()
        .find(|candidate| *candidate != layout)
        .expect("two FAT layouts");
    let mut fat = Fat32::open(Image::new(file)?).map_err(|error| anyhow!("{error}"))?;
    let mut exists = |candidate: Layout| -> Result<bool> {
        let path = format!("{}/{CONFIG}", candidate.data_path(layout::DATA_SWAP));
        match fat.file(&path) {
            Ok(_) => Ok(true),
            Err(FatError::Missing(_)) => Ok(false),
            Err(error) => Err(anyhow!("{error}")),
        }
    };
    ensure!(
        !exists(other)?,
        "a swap configuration already exists in {}; use its layout to maintain it",
        other.data_path(layout::DATA_SWAP)
    );
    let present = exists(layout)?;
    if present {
        let bytes = file.metadata()?.len();
        Ok(Some(inspect(file, block_bytes, bytes)?))
    } else {
        Ok(None)
    }
}

fn check_names(listing: &[u8], previous: Option<volume::Config>) -> Result<()> {
    let mut seen = BTreeSet::new();
    for name in listing.split(|&byte| byte == b'\n') {
        let name = String::from_utf8_lossy(name);
        let name = name
            .trim_end_matches(['\r', '/'])
            .rsplit('/')
            .next()
            .unwrap_or_default();
        let name = name.to_ascii_lowercase();
        let index = name
            .strip_suffix(".swp")
            .filter(|digits| digits.len() == 4 && digits.bytes().all(|byte| byte.is_ascii_digit()))
            .and_then(|digits| digits.parse::<usize>().ok());
        if name == CONFIG || index.is_some() {
            ensure!(
                previous
                    .is_some_and(|config| index.is_none_or(|index| index < config.files().len())),
                "unmanaged swap entry {name:?} occupies the FAT32 image"
            );
            ensure!(
                seen.insert(name.clone()),
                "ambiguous FAT32 swap entry {name:?}"
            );
        }
    }
    ensure!(
        previous.is_none_or(|config| seen.len() == config.files().len() + 1),
        "the FAT32 swap store uses an unmanaged filename alias"
    );
    Ok(())
}

fn inspect(file: &mut File, block_bytes: u64, limit: u64) -> Result<volume::Metadata> {
    volume::inspect(Image::new(file)?, block_bytes, limit).map_err(anyhow::Error::new)
}

struct Image<'a> {
    file: &'a mut File,
    bytes: u64,
}

impl<'a> Image<'a> {
    fn new(file: &'a mut File) -> Result<Self> {
        let metadata = file.metadata()?;
        ensure!(metadata.is_file(), "this is not a regular FAT32 image");
        Ok(Self {
            file,
            bytes: metadata.len(),
        })
    }
}

impl ReadAt for Image<'_> {
    type Error = io::Error;

    fn len(&self) -> u64 {
        self.bytes
    }

    fn read_at(&mut self, offset: u64, bytes: &mut [u8]) -> io::Result<()> {
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(bytes)
    }
}

struct Scan {
    capacity: Capacity,
    first_free: u32,
    fsinfo: Vec<u64>,
}

impl Scan {
    /// A stale FAT32 hint must not reduce the available capacity. Update
    /// recognized FSInfo sectors on the private image before invoking tools.
    fn refresh_fsinfo(&self, file: &mut File) -> Result<()> {
        let free = (self.capacity.free_bytes / self.capacity.cluster_bytes) as u32;
        let mut info = [0; 512];
        for &offset in &self.fsinfo {
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(&mut info)?;
            if le32(&info, 0) == 0x4161_5252
                && le32(&info, 484) == 0x6141_7272
                && le32(&info, 508) == 0xaa55_0000
                && (le32(&info, 488) != free || le32(&info, 492) != self.first_free)
            {
                file.seek(SeekFrom::Start(offset + 488))?;
                let mut fields = [0; 8];
                fields[..4].copy_from_slice(&free.to_le_bytes());
                fields[4..].copy_from_slice(&self.first_free.to_le_bytes());
                file.write_all(&fields)?;
            }
        }
        Ok(())
    }
}

fn scan(file: &mut File) -> Result<Scan> {
    let mut image = Image::new(file)?;
    let mut boot = [0; 512];
    image.read_at(0, &mut boot)?;
    let cluster_bytes = Fat32::open(&mut image)
        .map_err(|error| anyhow!("{error}"))?
        .cluster_bytes();
    let block_bytes = u64::from(le16(&boot, 11));
    let reserved = u64::from(le16(&boot, 14));
    let copies = u64::from(boot[16]);
    let sectors = u64::from(le32(&boot, 32));
    let fat_bytes = u64::from(le32(&boot, 36)) * block_bytes;
    ensure!(
        matches!(copies, 1 | 2) && le16(&boot, 40) & 0x80 == 0 && le16(&boot, 42) == 0,
        "only mirrored FAT32 version 0 images are supported"
    );
    let volume_bytes = sectors * block_bytes;
    ensure!(
        volume_bytes <= image.len()
            && image.len().is_multiple_of(block_bytes)
            && le16(&boot, 19) == 0,
        "FAT32 geometry does not match the raw partition image"
    );
    let start = reserved * block_bytes;
    // A raw partition can have trailing sectors outside the declared FAT
    // volume. They are neither allocatable nor part of the FAT update.
    let clusters = (volume_bytes - start - copies * fat_bytes) / cluster_bytes;
    ensure!(
        clusters <= u64::from(0x0fff_ffeeu32),
        "too many FAT32 clusters"
    );
    let mut primary = vec![0; FAT_WINDOW];
    let mut mirror = vec![0; FAT_WINDOW];
    let mut first_free = u32::MAX;
    let mut free = 0;
    for cluster in (2..clusters + 2).step_by(FAT_WINDOW / 4) {
        let size = ((clusters + 2 - cluster) * 4).min(FAT_WINDOW as u64) as usize;
        let offset = start + cluster * 4;
        image.read_at(offset, &mut primary[..size])?;
        if copies == 2 {
            image.read_at(offset + fat_bytes, &mut mirror[..size])?;
        }
        for at in (0..size).step_by(4) {
            let entry = le32(&primary, at) & FAT_MASK;
            ensure!(
                copies == 1 || entry == le32(&mirror, at) & FAT_MASK,
                "the FAT32 copies disagree at cluster {}",
                cluster + at as u64 / 4
            );
            if entry == 0 {
                first_free = first_free.min((cluster + at as u64 / 4) as u32);
                free += 1;
            }
        }
    }
    let info = u64::from(le16(&boot, 48));
    let backup = u64::from(le16(&boot, 50));
    let fsinfo = [info, backup + info]
        .into_iter()
        .enumerate()
        .filter_map(|(index, sector)| {
            (info != 0 && sector < reserved && (index == 0 || backup != 0))
                .then_some(sector * block_bytes)
        })
        .collect();
    Ok(Scan {
        capacity: Capacity {
            block_bytes,
            cluster_bytes,
            free_bytes: free * cluster_bytes,
        },
        first_free,
        fsinfo,
    })
}

fn le16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().expect("two FAT bytes"))
}

fn le32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four FAT bytes"))
}

struct Tools<'a> {
    stage: &'a NamedTempFile,
    directory: PathBuf,
    image: String,
}

impl<'a> Tools<'a> {
    fn new(stage: &'a NamedTempFile) -> Result<Self> {
        let directory = stage
            .path()
            .parent()
            .context("the FAT32 stage has no parent")?;
        let tools = Self {
            stage,
            directory: directory.to_path_buf(),
            image: tool_name(stage.path())?,
        };
        tools.check_stage()?;
        Ok(tools)
    }

    fn check_stage(&self) -> Result<()> {
        let path = fs::symlink_metadata(self.stage.path())?;
        let file = self.stage.as_file().metadata()?;
        ensure!(
            path.is_file() && file.is_file() && path.len() == file.len(),
            "the private FAT32 stage changed"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            ensure!(
                path.ino() == file.ino() && path.dev() == file.dev() && file.nlink() == 1,
                "the private FAT32 stage changed or acquired an alias"
            );
        }
        Ok(())
    }

    fn run(&self, program: &str, args: &[&str]) -> Result<Output> {
        self.check_stage()?;
        let mut command = Command::new(program);
        command
            .current_dir(&self.directory)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("LC_ALL", "C")
            .env("MTOOLSRC", if cfg!(windows) { "NUL" } else { "/dev/null" })
            .stdin(Stdio::null());
        if program != "fsck.fat" {
            command.args(["-i", &self.image]);
        }
        command.args(args).output().with_context(|| {
            format!(
                "cannot run {program}; FAT32 image provisioning requires GNU mtools and dosfstools"
            )
        })
    }

    fn checked(&self, program: &str, args: &[&str]) -> Result<Output> {
        let output = self.run(program, args)?;
        ensure!(
            output.status.success(),
            "{program} failed: {} {}",
            String::from_utf8_lossy(&output.stderr).trim(),
            String::from_utf8_lossy(&output.stdout).trim()
        );
        Ok(output)
    }

    fn check_fat(&self) -> Result<()> {
        self.checked("fsck.fat", &["-n", &self.image])?;
        Ok(())
    }

    fn require_installer(&self, layout: Layout) -> Result<()> {
        self.require_directory(&layout.data_path(layout::DATA_BOOT))?;
        self.require_directory(&layout.data_path(layout::DATA_INSTALL))?;
        Ok(())
    }

    fn require_directory(&self, path: &str) -> Result<Vec<u8>> {
        self.checked("mdir", &["-a", "-b", &format!("::/{path}/")])
            .with_context(|| format!("the FAT32 data image needs an existing {path} directory"))
            .map(|output| output.stdout)
    }

    fn ensure_directory(&self, path: &str) -> Result<Vec<u8>> {
        let target = format!("::/{path}");
        let directory = format!("{target}/");
        let args = ["-a", "-b", &directory];
        let first = self.run("mdir", &args)?;
        if first.status.success() {
            return Ok(first.stdout);
        }
        self.checked("mmd", &["-D", "s", "-D", "S", &target])?;
        Ok(self.checked("mdir", &args)?.stdout)
    }

    fn remove(&self, path: &str) -> Result<()> {
        self.checked("mdel", &[&format!("::/{path}")])?;
        Ok(())
    }

    fn copy(&self, source: &NamedTempFile, path: &str, replace: bool) -> Result<()> {
        self.checked(
            "mcopy",
            &[
                "-D",
                if replace { "o" } else { "s" },
                "-D",
                "S",
                &tool_name(source.path())?,
                &format!("::/{path}"),
            ],
        )?;
        Ok(())
    }
}

/// Keep the image argument free of mtools' `@@offset` and drive syntax.
fn tool_name(path: &Path) -> Result<String> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("invalid temporary image name")?;
    ensure!(
        name.bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b".-_".contains(&byte)),
        "the temporary image name is not safe for mtools"
    );
    Ok(format!("./{name}"))
}
