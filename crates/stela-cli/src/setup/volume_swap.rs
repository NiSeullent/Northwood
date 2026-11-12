// SPDX-License-Identifier: GPL-2.0-or-later

//! The installer journal owns a MEM-04 store on its FAT data partition.
//! The store's durable marker identifies partial work. Swap becomes
//! visible to the loader only after the extents and configuration are synced.
use super::{Record, host::Host, medium};
use crate::efi;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, Metadata, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};
use stela_formats::payload::gpt::Guid;
use stela_loader::mount_interface::layout::{self, Layout};
use stela_runtime::pc::swap::{HEADER_BYTES, Header, MIN_STORE_BYTES, volume::Config};

const CONFIG: &str = "swap.cfg";
const OWNER: &str = "owner.cfg";
const LOCK: &str = "owner.lck";
const BLOCK_BYTES: u64 = 4096;
// The supported FAT sector/cluster range can round each metadata allocation
// to 512 KiB. Reserve room for the directories, markers, config and rounding.
const METADATA_RESERVE: u64 = 4 << 20;
const ZERO_BYTES: usize = 1 << 20;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Store {
    pub(super) requested_bytes: u64,
    pub(super) bytes: u64,
    pub(super) guid: String,
    pub(super) partition_guid: String,
}

impl Store {
    pub(super) fn plan(
        requested_bytes: u64,
        max_bytes: u64,
        available_bytes: u64,
        guid: Guid,
        partition_guid: Guid,
    ) -> Result<Option<Self>> {
        ensure!(
            !guid.is_zero() && !partition_guid.is_zero(),
            "the swap store and FAT data partition need nonzero GUIDs"
        );
        let bytes = requested_bytes
            .min(max_bytes)
            .min(available_bytes.saturating_sub(METADATA_RESERVE));
        let bytes = bytes - bytes % BLOCK_BYTES;
        if bytes < MIN_STORE_BYTES {
            return Ok(None);
        }
        Config::new(bytes, BLOCK_BYTES, max_bytes)?;
        Ok(Some(Self {
            requested_bytes,
            bytes,
            guid: guid.to_string(),
            partition_guid: partition_guid.to_string(),
        }))
    }

    /// The data partition must still be the one setup recorded, beside the
    /// ESP on the same disk. A stale mount path cannot authorize an update.
    pub(super) fn validate_volume(&self, host: &dyn Host, record: &Record) -> Result<()> {
        ensure!(
            record.volume != record.esp,
            "swap needs the installer's FAT data partition"
        );
        let (volume, guid) = medium::locate(host, &record.esp, Some(&record.volume))?;
        let actual = parse_guid(&guid)?;
        let expected = parse_guid(&self.partition_guid)?;
        let disk = host.volume(&record.esp)?.disk;
        ensure!(
            actual == expected && disk.is_some() && host.volume(&volume)?.disk == disk,
            "{} does not match the recorded swap data partition {expected}",
            volume.display()
        );
        Ok(())
    }

    pub(super) fn directory(&self, volume: &Path, layout: Layout) -> PathBuf {
        volume.join(layout.data_path(layout::DATA_SWAP))
    }

    pub(super) fn problems(&self, volume: &Path, layout: Layout) -> Vec<String> {
        self.repair_plan(volume, layout)
            .unwrap_or_else(|error| vec![format!("{error:#}")])
    }

    /// A present but foreign or damaged managed file is a conflict. Missing
    /// files in an owned store are repairable without deleting any data.
    pub(super) fn repair_plan(&self, volume: &Path, layout: Layout) -> Result<Vec<String>> {
        let paths = self.paths(volume, layout)?;
        self.check_alternate(volume, layout)?;
        let (live, stage) = self.inspect(&paths)?;
        let mut problems = Vec::new();
        if !live.complete() {
            problems.push(format!(
                "finish preparing the swap store at {}",
                paths.live.display()
            ));
        }
        if stage.owner.is_some() && live.complete() {
            problems.push(format!(
                "clean up the completed swap copy at {}",
                paths.stage.display()
            ));
        }
        Ok(problems)
    }

    pub(super) fn publish(&self, volume: &Path, layout: Layout) -> Result<()> {
        let paths = self.paths(volume, layout)?;
        self.check_alternate(volume, layout)?;
        let (live, _) = self.inspect(&paths)?;
        create_directory(&paths.base)?;
        create_directory(&paths.live)?;
        let marker = self.marker()?;
        let owner = live.owner.unwrap_or_else(|| paths.live.join(OWNER));
        if regular(&owner)?.is_none() {
            efi::write_atomic(&owner, marker.as_bytes(), true)?;
        }
        let _lock = lock_folder(&owner, live.lock.as_deref(), marker.as_bytes())?;
        let (live, stage) = self.inspect(&paths)?;
        if live.complete() {
            return remove_folder(&paths.stage, &stage);
        }
        let header = self.header()?.encode();
        let config = self.config()?;
        let mut pending = Vec::new();
        for (index, extent) in config.files().enumerate() {
            if live.extents[index].is_none() {
                let name = extent_name(index);
                let path = stage.extents[index]
                    .clone()
                    .unwrap_or_else(|| paths.stage.join(&name));
                pending.push((index, extent.bytes(), path, paths.live.join(name)));
            }
        }
        if !pending.is_empty() {
            create_directory(&paths.stage)?;
            if stage.owner.is_none() {
                efi::write_atomic(&paths.stage.join(OWNER), marker.as_bytes(), true)?;
            }
            let zeros = vec![0; ZERO_BYTES];
            for (index, bytes, path, _) in &pending {
                let prefix = if *index == 0 { &header[..] } else { &[] };
                allocate(path, *bytes, prefix, &zeros)?;
            }
            sync_directory(&paths.stage);
            for (_, _, source, destination) in pending {
                move_fresh(&source, &destination)?;
            }
            sync_directory(&paths.live);
        }
        if live.config.is_none() {
            efi::write_atomic(
                &paths.live.join(CONFIG),
                self.config_text().as_bytes(),
                true,
            )?;
        }
        let (live, stage) = self.inspect(&paths)?;
        ensure!(
            live.complete(),
            "the prepared swap store did not read back complete"
        );
        remove_folder(&paths.stage, &stage)
    }

    /// Verify both folders before unlinking anything. The plan contains
    /// only the store's names, never unrelated contents of either folder.
    pub(super) fn removable_paths(&self, volume: &Path, layout: Layout) -> Result<Vec<PathBuf>> {
        let (live, stage) = self.inspect(&self.paths(volume, layout)?)?;
        Ok(stage.removals().chain(live.removals()).cloned().collect())
    }

    pub(super) fn remove(&self, volume: &Path, layout: Layout) -> Result<()> {
        let paths = self.paths(volume, layout)?;
        let (live, stage) = self.inspect(&paths)?;
        let marker = self.marker()?;
        let lock_folder_state = if live.owner.is_some() { &live } else { &stage };
        let _lock = lock_folder_state
            .owner
            .as_deref()
            .map(|owner| lock_folder(owner, lock_folder_state.lock.as_deref(), marker.as_bytes()))
            .transpose()?;
        let (live, stage) = self.inspect(&paths)?;
        remove_folder(&paths.stage, &stage)?;
        remove_folder(&paths.live, &live)
    }

    fn paths(&self, volume: &Path, layout: Layout) -> Result<Paths> {
        let guid = parse_guid(&self.guid)?;
        let base = volume.join(layout.data_dir());
        Ok(Paths {
            stage: base.join(format!(".swap-{guid}.partial")),
            live: self.directory(volume, layout),
            base,
        })
    }

    fn check_alternate(&self, volume: &Path, layout: Layout) -> Result<()> {
        let alternate = Layout::PREFERRED
            .into_iter()
            .find(|candidate| *candidate != layout)
            .expect("two installer layouts");
        let base = volume.join(alternate.data_dir());
        let directory = self.directory(volume, alternate);
        if directory_exists(&base)? && directory_exists(&directory)? {
            for entry in fs::read_dir(&directory)? {
                let entry = entry?;
                ensure!(
                    !entry.file_name().eq_ignore_ascii_case(CONFIG),
                    "a swap configuration already exists in {}",
                    directory.display()
                );
            }
        }
        Ok(())
    }

    fn config(&self) -> Result<Config> {
        ensure!(
            self.bytes <= self.requested_bytes,
            "invalid swap store size in setup record"
        );
        Ok(Config::new(self.bytes, BLOCK_BYTES, self.bytes)?)
    }

    fn config_text(&self) -> String {
        format!("bytes {}\n", self.bytes)
    }

    fn header(&self) -> Result<Header> {
        Ok(Header::new(
            self.config()?.bytes(),
            parse_guid(&self.guid)?,
        )?)
    }

    fn marker(&self) -> Result<String> {
        Ok(format!(
            "Northwood Swap owner 1\nstore {}\npartition {}\nbytes {}\n",
            parse_guid(&self.guid)?,
            parse_guid(&self.partition_guid)?,
            self.bytes
        ))
    }

    fn inspect(&self, paths: &Paths) -> Result<(Folder, Folder)> {
        directory_exists(&paths.base)?;
        Ok((
            self.inspect_folder(&paths.live, false)?,
            self.inspect_folder(&paths.stage, true)?,
        ))
    }

    fn inspect_folder(&self, path: &Path, staging: bool) -> Result<Folder> {
        let config = self.config()?;
        let mut folder = Folder {
            owner: None,
            lock: None,
            config: None,
            extents: vec![None; config.files().len()],
        };
        if !directory_exists(path)? {
            return Ok(folder);
        }
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let folded = name.to_ascii_lowercase();
            let slot = match folded.as_str() {
                OWNER => &mut folder.owner,
                LOCK => &mut folder.lock,
                CONFIG if !staging => &mut folder.config,
                CONFIG => anyhow::bail!(
                    "unexpected {} in the staged swap copy",
                    entry.path().display()
                ),
                name => match extent_index(name) {
                    Some(index) => folder.extents.get_mut(index).with_context(|| {
                        format!("unmanaged swap extent {}", entry.path().display())
                    })?,
                    None => continue,
                },
            };
            ensure!(
                slot.is_none(),
                "ambiguous FAT filename {}",
                entry.path().display()
            );
            regular(&entry.path())?.context("a swap path disappeared")?;
            *slot = Some(entry.path());
        }
        let Some(owner) = &folder.owner else {
            ensure!(
                folder.lock.is_none()
                    && folder.config.is_none()
                    && folder.extents.iter().all(Option::is_none),
                "{} has swap files without the recorded ownership marker",
                path.display()
            );
            return Ok(folder);
        };
        check_contents(owner, self.marker()?.as_bytes())?;
        if let Some(lock) = &folder.lock {
            ensure!(
                regular(lock)?.is_some_and(|metadata| metadata.len() == 0),
                "{} is not a swap lock file",
                lock.display()
            );
        }
        if let Some(config) = &folder.config {
            check_contents(config, self.config_text().as_bytes())?;
        }
        let header = self.header()?;
        for (index, extent) in config.files().enumerate() {
            let Some(path) = &folder.extents[index] else {
                continue;
            };
            let bytes = regular(path)?.context("a swap extent disappeared")?.len();
            ensure!(
                bytes == extent.bytes() || staging && bytes < extent.bytes(),
                "{} has {bytes} bytes; the setup record expects {}",
                path.display(),
                extent.bytes()
            );
            if index == 0 {
                check_header(path, header, self.bytes, bytes, staging)?;
            }
        }
        Ok(folder)
    }
}

struct Paths {
    base: PathBuf,
    live: PathBuf,
    stage: PathBuf,
}

struct Folder {
    owner: Option<PathBuf>,
    lock: Option<PathBuf>,
    config: Option<PathBuf>,
    extents: Vec<Option<PathBuf>>,
}

impl Folder {
    fn complete(&self) -> bool {
        self.owner.is_some() && self.config.is_some() && self.extents.iter().all(Option::is_some)
    }

    fn removals(&self) -> impl Iterator<Item = &PathBuf> {
        self.config
            .iter()
            .chain(self.extents.iter().rev().flatten())
            .chain(self.lock.iter())
            .chain(self.owner.iter())
    }
}

fn parse_guid(text: &str) -> Result<Guid> {
    Guid::parse(text)
        .filter(|guid| !guid.is_zero())
        .with_context(|| format!("invalid swap GUID {text:?}"))
}

fn extent_name(index: usize) -> String {
    format!("{index:04}.swp")
}

fn extent_index(name: &str) -> Option<usize> {
    let digits = name.strip_suffix(".swp")?;
    (digits.len() == 4 && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| digits.parse().ok())
        .flatten()
}

fn metadata(path: &Path) -> Result<Option<Metadata>> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("checking {}", path.display())),
    }
}

fn regular(path: &Path) -> Result<Option<Metadata>> {
    let Some(file) = metadata(path)? else {
        return Ok(None);
    };
    ensure!(
        file.is_file(),
        "{} is not a regular swap file",
        path.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        ensure!(file.nlink() == 1, "{} has a hard link", path.display());
    }
    Ok(Some(file))
}

fn directory_exists(path: &Path) -> Result<bool> {
    let current = metadata(path)?;
    if let Some(parent) = path.parent()
        && let Some(name) = path.file_name()
        && parent.is_dir()
    {
        let mut aliases = 0;
        for entry in fs::read_dir(parent)? {
            if entry?.file_name().eq_ignore_ascii_case(name) {
                aliases += 1;
            }
        }
        ensure!(
            aliases <= usize::from(current.is_some()),
            "{} has an ambiguous FAT filename",
            path.display()
        );
    }
    let Some(directory) = current else {
        return Ok(false);
    };
    ensure!(
        directory.is_dir(),
        "{} is not a regular swap directory",
        path.display()
    );
    Ok(true)
}

fn create_directory(path: &Path) -> Result<()> {
    if !directory_exists(path)? {
        fs::create_dir(path).with_context(|| format!("creating {}", path.display()))?;
        if let Some(parent) = path.parent() {
            sync_directory(parent);
        }
    }
    Ok(())
}

fn check_contents(path: &Path, expected: &[u8]) -> Result<()> {
    let mut found = Vec::new();
    File::open(path)?
        .take(expected.len() as u64 + 1)
        .read_to_end(&mut found)?;
    ensure!(
        found == expected,
        "{} does not match the setup record",
        path.display()
    );
    Ok(())
}

fn check_header(path: &Path, expected: Header, bytes: u64, len: u64, staging: bool) -> Result<()> {
    let mut prefix = [0; HEADER_BYTES];
    let count = len.min(HEADER_BYTES as u64) as usize;
    File::open(path)?.read_exact(&mut prefix[..count])?;
    if staging {
        ensure!(
            prefix[..count] == expected.encode()[..count],
            "{} has a foreign staged swap header",
            path.display()
        );
    } else {
        let header = Header::decode(&prefix, bytes)
            .with_context(|| format!("checking the swap header at {}", path.display()))?;
        ensure!(
            header.store_uuid() == expected.store_uuid() && header.layout() == expected.layout(),
            "{} has a swap header that differs from the setup record",
            path.display()
        );
    }
    Ok(())
}

fn lock_folder(owner: &Path, lock: Option<&Path>, marker: &[u8]) -> Result<File> {
    let default_lock = owner.with_file_name(LOCK);
    let path = lock.unwrap_or(&default_lock);
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    let file = match regular(path)? {
        Some(_) => options.open(path)?,
        None => match options.create_new(true).open(path) {
            Ok(file) => {
                file.sync_all()?;
                if let Some(directory) = path.parent() {
                    sync_directory(directory);
                }
                file
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                regular(path)?.context("the swap lock disappeared")?;
                OpenOptions::new().read(true).write(true).open(path)?
            }
            Err(error) => return Err(error.into()),
        },
    };
    file.try_lock()
        .with_context(|| format!("locking {}", path.display()))?;
    // Windows denies reads through another handle to a locked byte range.
    // The lock file has no payload; the separate owner marker remains readable.
    check_contents(owner, marker)?;
    Ok(file)
}

fn allocate(path: &Path, bytes: u64, prefix: &[u8], zeros: &[u8]) -> Result<()> {
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    if regular(path)?.is_none() {
        options.create_new(true);
    }
    let mut file = options
        .open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let mut written = file.seek(SeekFrom::End(0))?;
    ensure!(
        written <= bytes,
        "{} exceeds the planned extent size",
        path.display()
    );
    if written < prefix.len() as u64 {
        file.write_all(&prefix[written as usize..])?;
        written = prefix.len() as u64;
    }
    while written < bytes {
        let next = (bytes - written).min(zeros.len() as u64) as usize;
        file.write_all(&zeros[..next])
            .with_context(|| format!("allocating {}", path.display()))?;
        written += next as u64;
    }
    file.sync_all()
        .with_context(|| format!("syncing {}", path.display()))
}

fn move_fresh(source: &Path, destination: &Path) -> Result<()> {
    let mut staged = tempfile::TempPath::try_from_path(source)?;
    staged.disable_cleanup(true);
    staged
        .persist_noclobber(destination)
        .map_err(|error| error.error)
        .with_context(|| format!("publishing {}", destination.display()))
}

fn sync_directory(path: &Path) {
    let _ = File::open(path).and_then(|directory| directory.sync_all());
}

fn remove_folder(path: &Path, folder: &Folder) -> Result<()> {
    if folder.owner.is_none() {
        return Ok(());
    }
    for file in folder.removals() {
        fs::remove_file(file).with_context(|| format!("removing {}", file.display()))?;
    }
    sync_directory(path);
    if fs::read_dir(path)?.next().is_none() {
        fs::remove_dir(path)?;
        if let Some(parent) = path.parent() {
            sync_directory(parent);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> Store {
        Store::plan(
            MIN_STORE_BYTES,
            MIN_STORE_BYTES,
            MIN_STORE_BYTES + METADATA_RESERVE,
            Guid::from_entropy([1; 16]),
            Guid::from_entropy([2; 16]),
        )
        .unwrap()
        .unwrap()
    }

    #[test]
    fn resumes_partial_allocation_and_repairs_missing_files_without_touching_other_files() {
        let root = tempfile::tempdir().unwrap();
        let layout = Layout::Northwood;
        let store = store();
        let paths = store.paths(root.path(), layout).unwrap();
        fs::create_dir_all(&paths.stage).unwrap();
        fs::create_dir(&paths.live).unwrap();
        let marker = store.marker().unwrap();
        fs::write(paths.stage.join(OWNER), &marker).unwrap();
        fs::write(paths.live.join(OWNER), &marker).unwrap();
        fs::write(paths.live.join("unrelated.txt"), "kept").unwrap();
        let first = extent_name(0);
        fs::write(
            paths.stage.join(&first),
            &store.header().unwrap().encode()[..1024],
        )
        .unwrap();
        assert!(!store.repair_plan(root.path(), layout).unwrap().is_empty());

        store.publish(root.path(), layout).unwrap();
        assert!(store.problems(root.path(), layout).is_empty());
        assert!(!paths.stage.exists());
        fs::remove_file(paths.live.join(&first)).unwrap();
        fs::remove_file(paths.live.join(CONFIG)).unwrap();
        store.publish(root.path(), layout).unwrap();
        assert!(store.problems(root.path(), layout).is_empty());
        assert_eq!(fs::read(paths.live.join("unrelated.txt")).unwrap(), b"kept");
        assert_eq!(store.removable_paths(root.path(), layout).unwrap().len(), 4);
        store.remove(root.path(), layout).unwrap();
        store.remove(root.path(), layout).unwrap();
        assert!(
            store
                .removable_paths(root.path(), layout)
                .unwrap()
                .is_empty()
        );
        assert_eq!(fs::read(paths.live.join("unrelated.txt")).unwrap(), b"kept");
    }

    #[test]
    fn refuses_foreign_filenames_and_damaged_owned_contents_before_mutation() {
        let root = tempfile::tempdir().unwrap();
        let layout = Layout::Legacy;
        let store = store();
        let live = store.directory(root.path(), layout);
        fs::create_dir_all(&live).unwrap();
        let foreign = live.join("0000.SWP");
        fs::write(&foreign, "foreign").unwrap();
        assert!(store.publish(root.path(), layout).is_err());
        assert!(store.remove(root.path(), layout).is_err());
        assert_eq!(fs::read(&foreign).unwrap(), b"foreign");

        fs::remove_file(&foreign).unwrap();
        store.publish(root.path(), layout).unwrap();
        fs::write(live.join(CONFIG), "foreign configuration").unwrap();
        assert!(store.repair_plan(root.path(), layout).is_err());
        assert!(store.removable_paths(root.path(), layout).is_err());
        assert!(store.remove(root.path(), layout).is_err());
        assert!(live.join(extent_name(0)).is_file());
        assert_eq!(
            fs::read(live.join(CONFIG)).unwrap(),
            b"foreign configuration"
        );
    }
}
