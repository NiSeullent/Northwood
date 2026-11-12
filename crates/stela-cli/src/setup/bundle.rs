//! The boot bundles on the USB: one for each macOS build setup keeps, each
//! Stela machine (its ECID is in the bundle's `/chosen`), [`Generation`]
//! (the boot-bundle schema `stela.efi` reads it with, and the revision of
//! the loader that prepared it) and [`Kind`], in
//! `northwood/boot/<build>/<leaf>` of the USB's FAT32 data partition, or the
//! corresponding directory under `stela` on legacy media.
//!
//! - `restore-<ECID>`: a machine's restore bundle (`stela-dev boot prepare
//!   --restore`), which the Install macOS choice boots;
//! - `update-<ECID>`: a machine's update restore bundle (`stela-dev boot
//!   prepare --restore --update`), which earlier setups' Update macOS
//!   (Stela) choice booted; setup prepares none now;
//! - any of these with `-s<schema>` for another schema and `-r<revision>`
//!   for another revision, so a Stela update that reads a new schema or
//!   prepares a new revision writes its bundles beside the old ones before
//!   it removes them.
//!
//! A bundle is copied to a `.partial` sibling and renamed into place, so an
//! interrupted copy is never taken for a bundle; the bundle it replaces is
//! set aside as `.previous` until then, so a stop between the two renames
//! leaves it whole, and the next install or `repair` puts it back
//! ([`recover`]), while `status` reports what is left ([`leftovers`]).
//! The setup record keeps the
//! SHA-256 of its `bundle.json`, whose file list carries every file's
//! SHA-256, so [`verify`] checks the whole bundle.
use super::ecid::Ecid;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Read as _,
    path::{Path, PathBuf},
};
use stela_loader::mount_interface::layout::{DATA_BOOT, Layout};

pub(super) const MANIFEST: &str = "bundle.json";
/// The schema bundles without a suffix have.
const FIRST_SCHEMA: u32 = 1;
/// The revision bundles without a suffix have, and that of a `bundle.json`
/// without one.
const FIRST_REVISION: u32 = 0;

/// What a bundle's content is: its boot-bundle schema, the format
/// `stela.efi` reads (`stela_formats::boot_bundle::SCHEMA_VERSION`), and its
/// revision, that of the loader that prepared it
/// (`stela_loader::boot_bundle::REVISION`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Generation {
    pub(super) schema: u32,
    pub(super) revision: u32,
}

#[cfg(test)]
impl Generation {
    /// Schema 1, revision 0: every bundle from before setups recorded
    /// revisions.
    pub(super) const FIRST: Self = Self {
        schema: FIRST_SCHEMA,
        revision: FIRST_REVISION,
    };
}

impl std::fmt::Display for Generation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "schema {} revision {}", self.schema, self.revision)
    }
}

/// What a bundle boots.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Kind {
    /// The restore ramdisk (`boot prepare --restore`).
    #[default]
    Restore,
    /// The update restore ramdisk (`boot prepare --restore --update`),
    /// which earlier setups prepared.
    Update,
}

/// The kinds the install medium's restores boot: the erase restore alone.
/// The medium holds the recovery members only
/// (`restore::payload::trim`), which have no update identity; the installed
/// macOS updates itself, as on a Mac.
pub(super) const RESTORE_KINDS: [Kind; 1] = [Kind::Restore];

impl Kind {
    fn is_restore(&self) -> bool {
        *self == Self::Restore
    }

    fn prefix(self) -> &'static str {
        match self {
            Self::Restore => "restore",
            Self::Update => "update",
        }
    }

    /// What it is called in reports.
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Restore => "restore",
            Self::Update => "update restore",
        }
    }
}

/// One boot bundle setup wrote.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Bundle {
    pub(super) build: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) ecid: Option<Ecid>,
    pub(super) schema: u32,
    /// Revision 0 for a bundle an older setup wrote without recording it.
    #[serde(default)]
    pub(super) revision: u32,
    /// SHA-256 of its `bundle.json`; empty for a bundle an older setup
    /// wrote without recording it.
    #[serde(default)]
    pub(super) manifest_sha256: String,
    #[serde(default, skip_serializing_if = "Kind::is_restore")]
    pub(super) kind: Kind,
    /// A restore bundle that boots under the AP ticket Apple's signing
    /// service personalized for its machine (`boot prepare --restore --ecid`), not the
    /// IPSW's global ticket.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) personalized: bool,
    /// The PEs of its Mac; 0 for a bundle that did not record them.
    #[serde(default)]
    pub(super) processors: usize,
}

/// Fields selecting one bundle within a setup's build/machine generations.
#[derive(Clone, Copy)]
pub(super) struct Identity<'a> {
    pub(super) kind: Kind,
    pub(super) build: &'a str,
    pub(super) ecid: Option<Ecid>,
    pub(super) generation: Generation,
}

impl Bundle {
    pub(super) fn generation(&self) -> Generation {
        Generation {
            schema: self.schema,
            revision: self.revision,
        }
    }

    pub(super) fn same_kind(&self, identity: Identity<'_>) -> bool {
        let Identity {
            kind,
            build,
            ecid,
            generation,
        } = identity;
        self.kind == kind
            && self.build == build
            && self.ecid == ecid
            && self.generation() == generation
    }

    /// Its directory on `volume`.
    #[cfg(test)]
    pub(super) fn directory(&self, volume: &Path) -> PathBuf {
        self.directory_in(volume, Layout::Legacy)
    }

    /// Its directory on `volume` in the setup's layout.
    pub(super) fn directory_in(&self, volume: &Path, layout: Layout) -> PathBuf {
        directory_of_in(
            volume,
            layout,
            Identity {
                kind: self.kind,
                build: &self.build,
                ecid: self.ecid,
                generation: self.generation(),
            },
        )
    }
}

pub(super) fn leaf_of(kind: Kind, ecid: Option<Ecid>, generation: Generation) -> String {
    let prefix = kind.prefix();
    let mut leaf = match ecid {
        Some(ecid) => format!("{prefix}-{}", ecid.digits()),
        None => prefix.into(),
    };
    if generation.schema != FIRST_SCHEMA {
        leaf.push_str(&format!("-s{}", generation.schema));
    }
    if generation.revision != FIRST_REVISION {
        leaf.push_str(&format!("-r{}", generation.revision));
    }
    leaf
}

/// The directory of a build's bundles below a data partition's root.
#[cfg(test)]
pub(super) const BOOT_DIRECTORY: &str = "stela/boot";

/// The active layout's boot-bundle build directory on `volume`.
pub(super) fn build_directory_in(volume: &Path, layout: Layout, build: &str) -> PathBuf {
    volume.join(layout.data_path(DATA_BOOT)).join(build)
}

#[cfg(test)]
pub(super) fn directory_of(volume: &Path, identity: Identity<'_>) -> PathBuf {
    directory_of_in(volume, Layout::Legacy, identity)
}

pub(super) fn directory_of_in(volume: &Path, layout: Layout, identity: Identity<'_>) -> PathBuf {
    let Identity {
        kind,
        build,
        ecid,
        generation,
    } = identity;
    build_directory_in(volume, layout, build).join(leaf_of(kind, ecid, generation))
}

#[cfg(test)]
fn firmware_path_of(identity: Identity<'_>) -> String {
    firmware_path_of_in(Layout::Legacy, identity)
}

pub(super) fn firmware_path_of_in(layout: Layout, identity: Identity<'_>) -> String {
    let Identity {
        kind,
        build,
        ecid,
        generation,
    } = identity;
    format!(
        "\\{}\\{build}\\{}",
        layout.data_path(DATA_BOOT).replace('/', "\\"),
        leaf_of(kind, ecid, generation)
    )
}

/// What `bundle.json` says of itself and its files.
#[derive(Deserialize)]
struct Listing {
    schema_version: u32,
    #[serde(default)]
    revision: u32,
    #[serde(default)]
    processors: usize,
    #[serde(default)]
    files: Vec<Listed>,
}

#[derive(Deserialize)]
struct Listed {
    name: String,
    bytes: u64,
    sha256: String,
}

/// The generation a bundle directory's `bundle.json` declares.
pub(super) fn generation(directory: &Path) -> Result<Generation> {
    let listing: Listing = stela_engine::read_json(&directory.join(MANIFEST))?;
    Ok(Generation {
        schema: listing.schema_version,
        revision: listing.revision,
    })
}

/// The PEs of the Mac a prepared bundle directory boots.
pub(super) fn processors(directory: &Path) -> Result<usize> {
    let listing: Listing = stela_engine::read_json(&directory.join(MANIFEST))?;
    Ok(listing.processors)
}

/// Recover the count from the recorded manifest when an older setup omitted
/// it from the record. The live disk topology is still checked at boot.
pub(super) fn recorded_processors(directory: &Path, expected: &str) -> Option<usize> {
    use stela_formats::boot_bundle::MAX_MANIFEST_BYTES;

    if expected.is_empty() {
        return None;
    }
    let mut bytes = Vec::new();
    fs::File::open(directory.join(MANIFEST))
        .ok()?
        .take((MAX_MANIFEST_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() > MAX_MANIFEST_BYTES || stela_formats::sha256(&bytes) != expected {
        return None;
    }
    let listing: Listing = serde_json::from_slice(&bytes).ok()?;
    Some(listing.processors).filter(|&count| count != 0)
}

/// Bytes of the files in a prepared bundle directory.
pub(super) fn bytes(directory: &Path) -> Result<u64> {
    let mut total = 0;
    for file in fs::read_dir(directory)? {
        total += file?.metadata()?.len();
    }
    Ok(total)
}

fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut file =
        fs::File::open(path).with_context(|| format!("cannot read {}", path.display()))?;
    let mut hash = Sha256::new();
    std::io::copy(&mut file, &mut hash)?;
    Ok(hex::encode(hash.finalize()))
}

/// The siblings [`install`] writes beside `destination`: the copy in
/// progress, and the bundle it replaces until the copy is in place.
pub(super) fn siblings(destination: &Path) -> Result<(PathBuf, PathBuf)> {
    let name = destination
        .file_name()
        .context("a bundle directory without a name")?
        .to_string_lossy()
        .into_owned();
    Ok((
        destination.with_file_name(format!("{name}.partial")),
        destination.with_file_name(format!("{name}.previous")),
    ))
}

/// Put back what an [`install`] that stopped part way left beside
/// `destination`: a `.previous` bundle without a bundle in place is the
/// last complete one (the stop came between its two renames), so it is
/// renamed back; a `.previous` beside a bundle, and any `.partial`, are
/// removed. Returns what was done, for the report.
pub(super) fn recover(destination: &Path) -> Result<Vec<String>> {
    let (partial, previous) = siblings(destination)?;
    let mut done = Vec::new();
    if previous.exists() && !destination.exists() {
        fs::rename(&previous, destination)?;
        sync_parent(destination);
        done.push(format!(
            "put back {}, which an interrupted copy had set aside",
            destination.display()
        ));
    }
    for stale in [&partial, &previous] {
        if stale.exists() {
            fs::remove_dir_all(stale)?;
            done.push(format!(
                "removed {}, left by an interrupted copy",
                stale.display()
            ));
        }
    }
    Ok(done)
}

/// Where `destination`'s bundle is: its `.previous` when an interrupted
/// [`install`] set it aside and [`recover`] has not put it back.
pub(super) fn in_place(destination: &Path) -> PathBuf {
    match siblings(destination) {
        Ok((_, previous)) if !destination.exists() && previous.is_dir() => previous,
        _ => destination.to_path_buf(),
    }
}

/// What interrupted [`install`]s left in a build directory: each bundle
/// with a `.partial` or `.previous` sibling, and what is wrong.
pub(super) fn leftovers(build_directory: &Path) -> Vec<(PathBuf, String)> {
    let Ok(entries) = fs::read_dir(build_directory) else {
        return Vec::new();
    };
    let mut found: Vec<(PathBuf, String)> = Vec::new();
    let mut names: Vec<String> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    for name in names {
        let (leaf, what) = if let Some(leaf) = name.strip_suffix(".partial") {
            (leaf, "a copy that stopped part way")
        } else if let Some(leaf) = name.strip_suffix(".previous") {
            let destination = build_directory.join(leaf);
            if destination.exists() {
                (leaf, "the bundle an interrupted copy replaced")
            } else {
                (
                    leaf,
                    "the bundle itself, set aside by a copy that stopped between its renames",
                )
            }
        } else {
            continue;
        };
        found.push((
            build_directory.join(leaf),
            format!("{} is {what}", build_directory.join(&name).display()),
        ));
    }
    found
}

/// Make the directory's renames durable; FAT may refuse to sync a
/// directory, and its files are already synced.
fn sync_parent(path: &Path) {
    if let Some(parent) = path.parent() {
        let _ = fs::File::open(parent).and_then(|directory| directory.sync_all());
    }
}

/// Copy the prepared bundle in `prepared` to `destination`, replacing what
/// is there only once the copy is complete; returns its `bundle.json`'s
/// SHA-256. What an earlier, interrupted install left is recovered first
/// ([`recover`]).
pub(super) fn install(prepared: &Path, destination: &Path) -> Result<String> {
    recover(destination)?;
    let (partial, previous) = siblings(destination)?;
    fs::create_dir_all(&partial)?;
    let progress = crate::progress::Phase::bytes("Writing boot files", bytes(prepared)?);
    for file in fs::read_dir(prepared)? {
        let file = file?;
        let target = partial.join(file.file_name());
        crate::progress::copy(&file.path(), &target, &progress)?;
    }
    sync_parent(&partial.join(MANIFEST));
    progress.finish();
    if destination.exists() {
        fs::rename(destination, &previous)?;
    }
    fs::rename(&partial, destination)?;
    sync_parent(destination);
    if previous.exists() {
        fs::remove_dir_all(&previous)?;
    }
    sha256_file(&destination.join(MANIFEST))
}

/// What is wrong with the bundle in `directory`, checked against its
/// `bundle.json` and, when known, that file's recorded SHA-256; empty when
/// it is intact.
pub(super) fn verify(directory: &Path, manifest_sha256: &str) -> Vec<String> {
    let mut problems = Vec::new();
    let manifest = directory.join(MANIFEST);
    if !manifest.is_file() {
        problems.push(format!("{} is missing", manifest.display()));
        return problems;
    }
    match sha256_file(&manifest) {
        Ok(digest) if !manifest_sha256.is_empty() && digest != manifest_sha256 => problems.push(
            format!("{} differs from the one setup wrote", manifest.display()),
        ),
        Ok(_) => {}
        Err(error) => problems.push(format!("{error:#}")),
    }
    let listing: Listing = match stela_engine::read_json(&manifest) {
        Ok(listing) => listing,
        Err(error) => {
            problems.push(format!("{} cannot be read: {error:#}", manifest.display()));
            return problems;
        }
    };
    if listing.files.is_empty() {
        problems.push(format!("{} lists no files", manifest.display()));
    }
    for file in &listing.files {
        let path = directory.join(&file.name);
        let check = || -> Result<()> {
            ensure!(
                !file.name.contains(['/', '\\']) && file.name != MANIFEST,
                "lists an unexpected file name {:?}",
                file.name
            );
            let bytes = fs::metadata(&path)
                .with_context(|| format!("{} is missing", path.display()))?
                .len();
            ensure!(
                bytes == file.bytes,
                "{} holds {bytes} bytes, not {}",
                path.display(),
                file.bytes
            );
            ensure!(
                sha256_file(&path)? == file.sha256,
                "{} differs from its SHA-256 in {MANIFEST}",
                path.display()
            );
            Ok(())
        };
        if let Err(error) = check() {
            problems.push(format!("{error:#}"));
        }
    }
    problems
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[test]
    fn legacy_processor_count_comes_from_the_recorded_manifest() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join(MANIFEST);
        let bytes = br#"{"schema_version":4,"processors":8}"#;
        let digest = stela_formats::sha256(bytes);
        fs::write(&path, bytes).unwrap();
        assert_eq!(recorded_processors(temp.path(), &digest), Some(8));
        assert_eq!(recorded_processors(temp.path(), ""), None);
        let replacement = br#"{"schema_version":4,"processors":1}"#;
        fs::write(&path, replacement).unwrap();
        assert_eq!(recorded_processors(temp.path(), &digest), None);
        let omitted = br#"{"schema_version":4}"#;
        fs::write(&path, omitted).unwrap();
        assert_eq!(
            recorded_processors(temp.path(), &stela_formats::sha256(omitted)),
            None
        );
    }

    pub(in super::super) fn of(schema: u32, revision: u32) -> Generation {
        Generation { schema, revision }
    }

    /// A bundle directory of one file, listed in a `bundle.json` of
    /// `generation`, which names no revision for revision 0, as those of
    /// loaders before revisions do not.
    pub(in super::super) fn write(
        directory: &Path,
        generation: Generation,
        contents: &[u8],
    ) -> Result<()> {
        use sha2::{Digest, Sha256};
        fs::create_dir_all(directory)?;
        fs::write(directory.join("ramdisk.img"), contents)?;
        let mut listing = serde_json::json!({
            "schema_version": generation.schema,
            "files": [{
                "name": "ramdisk.img",
                "bytes": contents.len(),
                "sha256": hex::encode(Sha256::digest(contents)),
            }],
        });
        if generation.revision != FIRST_REVISION {
            listing["revision"] = generation.revision.into();
        }
        Ok(stela_engine::write_json(
            &directory.join(MANIFEST),
            &listing,
        )?)
    }

    #[test]
    fn bundles_are_named_by_machine_schema_and_revision() {
        let restore = |ecid, generation| leaf_of(Kind::Restore, ecid, generation);
        assert_eq!(restore(None, Generation::FIRST), "restore");
        assert_eq!(
            restore(Some(Ecid(0xab)), Generation::FIRST),
            "restore-00000000000000AB"
        );
        assert_eq!(restore(None, of(3, 0)), "restore-s3");
        assert_eq!(
            restore(Some(Ecid(0xab)), of(1, 4)),
            "restore-00000000000000AB-r4"
        );
        assert_eq!(restore(None, of(2, 1)), "restore-s2-r1");
        assert_eq!(
            leaf_of(Kind::Restore, Some(Ecid(0xab)), of(2, 0)),
            "restore-00000000000000AB-s2"
        );
        assert_eq!(
            firmware_path_of(Identity {
                kind: Kind::Restore,
                build: "26A434",
                ecid: Some(Ecid(1)),
                generation: Generation::FIRST
            }),
            "\\stela\\boot\\26A434\\restore-0000000000000001"
        );
        let identity = Identity {
            kind: Kind::Restore,
            build: "26A434",
            ecid: Some(Ecid(1)),
            generation: Generation::FIRST,
        };
        assert_eq!(
            directory_of_in(Path::new("/mnt/data"), Layout::Northwood, identity),
            Path::new("/mnt/data/northwood/boot/26A434/restore-0000000000000001")
        );
        assert_eq!(
            firmware_path_of_in(Layout::Northwood, identity),
            "\\northwood\\boot\\26A434\\restore-0000000000000001"
        );
        assert_eq!(
            firmware_path_of(Identity {
                kind: Kind::Update,
                build: "26A434",
                ecid: Some(Ecid(1)),
                generation: of(1, 2)
            }),
            "\\stela\\boot\\26A434\\update-0000000000000001-r2"
        );
    }

    #[test]
    fn missing_bundle_revision_defaults_to_zero() {
        let dir = tempfile::tempdir().unwrap();
        // As every loader before revisions wrote it.
        let listing = serde_json::json!({"schema_version": 1, "files": []});
        stela_engine::write_json(&dir.path().join(MANIFEST), &listing).unwrap();
        assert_eq!(generation(dir.path()).unwrap(), Generation::FIRST);
        let listing = serde_json::json!({"schema_version": 1, "revision": 3, "files": []});
        stela_engine::write_json(&dir.path().join(MANIFEST), &listing).unwrap();
        assert_eq!(generation(dir.path()).unwrap(), of(1, 3));
        // So is a bundle an older setup recorded.
        let kept: Bundle =
            serde_json::from_str(r#"{"build": "B", "schema": 1, "manifest_sha256": ""}"#).unwrap();
        assert_eq!(kept.generation(), Generation::FIRST);
    }

    #[test]
    fn installed_bundles_verify_and_damage_is_found() {
        let dir = tempfile::tempdir().unwrap();
        let prepared = dir.path().join("prepared");
        write(&prepared, Generation::FIRST, &[0x5a; 4096]).unwrap();
        let destination = dir.path().join("boot/B/restore");
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        let digest = install(&prepared, &destination).unwrap();
        assert!(verify(&destination, &digest).is_empty());
        assert_eq!(generation(&destination).unwrap(), Generation::FIRST);
        // A second install replaces it whole.
        write(&prepared, of(2, 1), &[0x33; 100]).unwrap();
        let second = install(&prepared, &destination).unwrap();
        assert_ne!(second, digest);
        assert_eq!(generation(&destination).unwrap(), of(2, 1));
        assert!(!dir.path().join("boot/B/restore.partial").exists());
        assert!(!verify(&destination, &digest).is_empty());
        fs::write(destination.join("ramdisk.img"), [0x34; 100]).unwrap();
        let problems = verify(&destination, &second);
        assert!(
            problems[0].contains("differs from its SHA-256"),
            "{problems:?}"
        );
        fs::remove_file(destination.join("ramdisk.img")).unwrap();
        assert!(verify(&destination, "")[0].contains("missing"));
    }

    #[test]
    fn every_stop_inside_an_install_leaves_one_whole_bundle() {
        let dir = tempfile::tempdir().unwrap();
        let prepared = dir.path().join("prepared");
        let build = dir.path().join("boot/B");
        let destination = build.join("restore");
        fs::create_dir_all(&build).unwrap();
        write(&prepared, Generation::FIRST, &[0x5a; 4096]).unwrap();
        let first = install(&prepared, &destination).unwrap();
        let (partial, previous) = siblings(&destination).unwrap();
        let half = |path: &Path| {
            fs::create_dir_all(path).unwrap();
            fs::write(path.join("ramdisk.img"), b"half").unwrap();
        };
        // Stopped while copying: the bundle in place is whole.
        half(&partial);
        assert!(leftovers(&build)[0].1.contains("stopped part way"));
        assert_eq!(in_place(&destination), destination);
        // Stopped between the renames: the set-aside bundle is the whole one.
        fs::rename(&destination, &previous).unwrap();
        assert_eq!(in_place(&destination), previous);
        assert!(verify(&in_place(&destination), &first).is_empty());
        let found = leftovers(&build);
        assert_eq!(found.len(), 2);
        assert!(found.iter().all(|(d, _)| *d == destination));
        assert!(found[1].1.contains("set aside"), "{found:?}");
        assert_eq!(recover(&destination).unwrap().len(), 2);
        assert!(verify(&destination, &first).is_empty());
        assert!(leftovers(&build).is_empty());
        // Stopped before the set-aside copy was removed.
        half(&previous);
        assert!(leftovers(&build)[0].1.contains("replaced"));
        recover(&destination).unwrap();
        assert!(verify(&destination, &first).is_empty() && !previous.exists());
        // An install after any of these replaces the bundle whole.
        half(&partial);
        fs::rename(&destination, &previous).unwrap();
        write(&prepared, of(2, 0), &[0x33; 100]).unwrap();
        let second = install(&prepared, &destination).unwrap();
        assert!(verify(&destination, &second).is_empty());
        assert!(leftovers(&build).is_empty());
    }
}
