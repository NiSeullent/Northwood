//! Preparing a setup's boot bundles on this computer and fetching the
//! restore images they are prepared from.
use super::{
    Environment, InstallImage, MARGIN, Request,
    bundle::{self, Kind},
    ecid::Ecid,
    host::Host,
    record::Record,
};
use crate::{boot, restore};
use anyhow::{Context, Result, bail, ensure};
use std::{
    fs,
    path::{Path, PathBuf},
};
use stela_formats::builds;

/// A boot bundle prepared on this computer, not yet copied.
pub(super) struct Prepared {
    pub(super) build: String,
    pub(super) ecid: Option<Ecid>,
    pub(super) kind: Kind,
    pub(super) personalized: bool,
    pub(super) generation: bundle::Generation,
    pub(super) processors: usize,
    pub(super) directory: tempfile::TempDir,
    pub(super) bytes: u64,
}

pub(super) type PreparedSetup = (String, String, Vec<Prepared>);

impl Prepared {
    pub(super) fn path(&self) -> PathBuf {
        self.directory.path().join(BUNDLE_LEAF)
    }
}

/// The directory a bundle is prepared into, inside its temporary one.
pub(super) const BUNDLE_LEAF: &str = "bundle";

/// Prepare `ipsw`'s restore bundles of every kind setup writes
/// ([`bundle::RESTORE_KINDS`]) for the machine `ecid`, under Apple's
/// personalized ticket. Returns the image's version and build too.
pub(super) fn prepare_restore_bundles(
    environment: &Environment<'_>,
    ipsw: &Path,
    model: &str,
    ecid: Ecid,
) -> Result<PreparedSetup> {
    let (product_version, build) = restore::payload::identity(ipsw, model)?;
    let mut prepared = Vec::new();
    for kind in bundle::RESTORE_KINDS {
        let (_, _, bundles) = prepare_kind(
            environment,
            BundleOptions {
                kind,
                ipsw,
                model,
                build: Some(&build),
                ecids: &[Some(ecid)],
            },
        )?;
        prepared.extend(bundles);
    }
    Ok((product_version, build, prepared))
}

/// Refuse `ipsw`, of macOS `build`, unless setup supports that build.
/// A macOS 27 build Stela was not tested with is admitted with a note.
pub(super) fn supported(environment: &Environment<'_>, ipsw: &Path, build: &str) -> Result<()> {
    let entry = builds::find_in(environment.builds, build)
        .map_err(|error| anyhow::anyhow!("{}: {error}", ipsw.display()))?;
    if entry.build != build {
        eprintln!(
            "note: macOS build {build} is not tested yet; Stela uses build {}'s device \
             interfaces for it",
            entry.build
        );
    }
    Ok(())
}

/// Source identity and personalization choices for one kind of boot bundle.
pub(super) struct BundleOptions<'a> {
    pub(super) kind: Kind,
    pub(super) ipsw: &'a Path,
    pub(super) model: &'a str,
    pub(super) build: Option<&'a str>,
    pub(super) ecids: &'a [Option<Ecid>],
}

/// Prepare the selected bundle kind for each machine; verify the requested
/// build and return its product version, build and prepared bundles.
pub(super) fn prepare_kind(
    environment: &Environment<'_>,
    options: BundleOptions<'_>,
) -> Result<(String, String, Vec<Prepared>)> {
    let BundleOptions {
        kind,
        ipsw,
        model,
        build,
        ecids,
    } = options;
    let (product_version, found) = restore::payload::identity(ipsw, model)?;
    if let Some(build) = build {
        ensure!(
            found == build,
            "{} holds macOS build {found}, not {build}",
            ipsw.display()
        );
    }
    supported(environment, ipsw, &found)?;
    let mut prepared = Vec::new();
    for &ecid in ecids {
        let directory = tempfile::tempdir()?;
        let output = directory.path().join(BUNDLE_LEAF);
        let prepare = match kind {
            Kind::Restore => environment.prepare_restore,
            Kind::Update => environment.prepare_update,
        };
        // A restore bundle for a known machine boots under its personalized
        // ticket from Apple's signing service (TSS).
        let personalized = ecid.is_some();
        let progress =
            crate::progress::Phase::spinner(format!("Preparing {} boot files", kind.label()));
        prepare(RamdiskOptions {
            ipsw,
            model,
            ecid,
            output: &output,
        })
        .with_context(|| format!("preparing the {} boot bundle", kind.label()))?;
        progress.finish();
        let generation = bundle::generation(&output)?;
        ensure!(
            generation == environment.generation(),
            "the prepared bundle is {generation}, not the {} this stela and its stela.efi use",
            environment.generation()
        );
        let bytes = bundle::bytes(&output)?;
        prepared.push(Prepared {
            build: found.clone(),
            ecid,
            kind,
            personalized,
            generation,
            processors: bundle::processors(&output)?,
            directory,
            bytes,
        });
    }
    Ok((product_version, found, prepared))
}

/// Copy prepared bundles to the record's volume and record them, as
/// bundles of its active generation.
pub(super) fn copy_bundles(record: &mut Record, prepared: &[Prepared]) -> Result<()> {
    let generation = record.generation();
    for bundle in prepared {
        ensure!(
            bundle.generation == generation,
            "a {} bundle cannot be one of this setup's {generation} bundles",
            bundle.generation
        );
        let destination = bundle::directory_of_in(
            &record.volume,
            record.layout,
            bundle::Identity {
                kind: bundle.kind,
                build: &bundle.build,
                ecid: bundle.ecid,
                generation,
            },
        );
        fs::create_dir_all(destination.parent().unwrap())?;
        let digest = bundle::install(&bundle.path(), &destination)?;
        // Read back, so a USB that loses writes is found now, not at boot.
        let problems = bundle::verify(&destination, &digest);
        ensure!(
            problems.is_empty(),
            "the copied bundle does not read back as written: {}",
            problems.join("; ")
        );
        record.bundles.retain(|b| {
            !b.same_kind(bundle::Identity {
                kind: bundle.kind,
                build: &bundle.build,
                ecid: bundle.ecid,
                generation,
            })
        });
        record.bundles.push(bundle::Bundle {
            build: bundle.build.clone(),
            ecid: bundle.ecid,
            schema: generation.schema,
            revision: generation.revision,
            manifest_sha256: digest,
            kind: bundle.kind,
            personalized: bundle.personalized,
            processors: bundle.processors,
        });
    }
    Ok(())
}

/// Whether `volume` has room for `prepared`.
pub(super) fn check_space(host: &dyn Host, volume: &Path, prepared: &[Prepared]) -> Result<()> {
    let need: u64 = prepared.iter().map(|p| p.bytes).sum::<u64>() + MARGIN;
    let free = host.volume(volume)?.free_bytes;
    ensure!(
        free >= need,
        "{} has {:.1} GB free; the boot bundles need {:.1} GB",
        volume.display(),
        free as f64 / 1e9,
        need as f64 / 1e9
    );
    Ok(())
}

/// Where setup keeps downloaded restore images without `--download-dir`:
/// `images` in Stela's per-user cache ([`cache_dir`]).
pub(crate) fn download_dir() -> PathBuf {
    cache_dir().join("images")
}

/// Stela's per-user cache: `stela` in the platform's cache directory, or in
/// the current directory when the environment names none. Never the system
/// temp directory, which is often too small for macOS.
pub(crate) fn cache_dir() -> PathBuf {
    let var = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty());
    cache_dir_in(std::env::consts::OS, var).unwrap_or_else(|| PathBuf::from("stela-cache"))
}

fn rooted(path: &Path) -> bool {
    path.is_absolute() || path.to_string_lossy().starts_with('/')
}

/// [`cache_dir`] on operating system `os` with environment `var`: Linux and
/// other Unix `$XDG_CACHE_HOME/stela` or `~/.cache/stela`, macOS
/// `~/Library/Caches/stela`, Windows `%LOCALAPPDATA%\stela\cache`.
pub(super) fn cache_dir_in(
    os: &str,
    var: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Option<PathBuf> {
    let home = || var("HOME").map(PathBuf::from).filter(|home| rooted(home));
    match os {
        "windows" => var("LOCALAPPDATA")
            .map(PathBuf::from)
            .filter(|path| path.is_absolute() || path.to_string_lossy().contains(':'))
            .map(|path| path.join("stela").join("cache")),
        "macos" => home().map(|home| home.join("Library/Caches/stela")),
        _ => var("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .filter(|path| rooted(path))
            .or_else(|| home().map(|home| home.join(".cache")))
            .map(|cache| cache.join("stela")),
    }
}

/// Refuse to write `need` bytes into `directory` (created on demand) unless
/// its file system has room for them and [`MARGIN`]; `what` names the
/// download.
pub(crate) fn ensure_room(directory: &Path, what: &str, need: u64) -> Result<()> {
    fs::create_dir_all(directory)
        .with_context(|| format!("cannot create {}", directory.display()))?;
    let free = super::host::System.volume(directory)?.free_bytes;
    match shortfall(directory, what, need, free) {
        Some(text) => bail!("{text}"),
        None => Ok(()),
    }
}

/// Why `free` bytes in `directory` cannot take `need` bytes and [`MARGIN`].
pub(super) fn shortfall(directory: &Path, what: &str, need: u64, free: u64) -> Option<String> {
    (free < need.saturating_add(MARGIN)).then(|| {
        format!(
            "{what} needs {} in {}, which has {} free; free some space there or choose another \
             place with --download-dir",
            super::host::bytes_text(need.saturating_add(MARGIN)),
            directory.display(),
            super::host::bytes_text(free)
        )
    })
}

/// Download `--build`'s image, or reuse a complete earlier download: the
/// whole IPSW for a full install medium, else its recovery members,
/// which prepare the boot bundles.
pub(super) fn fetch(command: &Request) -> Result<PathBuf> {
    let directory = command.download_dir.clone().unwrap_or_else(download_dir);
    match command.install_image {
        Some(InstallImage::Full) => fetch_image(&directory, &command.build, command.model()),
        _ => fetch_recovery(&directory, &command.build, command.model()),
    }
}

/// Copy `build`'s recovery members for `model` from Apple's servers into
/// `directory` (`restore::fetch_recovery`), or reuse an earlier complete
/// copy: setup's install medium needs no more of the image.
pub(super) fn fetch_recovery(directory: &Path, build: &str, model: &str) -> Result<PathBuf> {
    ensure!(
        build.bytes().all(|c| c.is_ascii_alphanumeric())
            && model
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b','),
        "unexpected build or model identifier"
    );
    let path = directory.join(format!("Recovery_{build}_{}.ipsw", model.replace(',', "_")));
    if path.is_file() {
        let archive = zip::ZipArchive::new(fs::File::open(&path)?)
            .with_context(|| format!("{} is not a complete archive", path.display()))?;
        ensure!(
            archive.comment() == restore::payload::RECOVERY_COMMENT,
            "{} is not a recovery archive; move it away",
            path.display()
        );
        eprintln!("Using downloaded boot files");
        return Ok(path);
    }
    let images = restore::catalog()?;
    let image = restore::select(&images, Some(build), Some(model))?;
    fs::create_dir_all(directory)?;
    let bytes = restore::fetch_recovery(&image, model, &path)?;
    eprintln!("Downloaded {:.2} GB of boot files", bytes as f64 / 1e9);
    Ok(path)
}

/// Download `build`'s restore image for `model` from Apple's catalog into
/// `directory`, or reuse a complete earlier download from the same entry.
pub(crate) fn fetch_image(directory: &Path, build: &str, model: &str) -> Result<PathBuf> {
    let images = restore::catalog()?;
    let image = restore::select(&images, Some(build), Some(model))?;
    ensure!(
        image.build.bytes().all(|c| c.is_ascii_alphanumeric())
            && image
                .version
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'.'),
        "unexpected catalog identifiers"
    );
    let path = directory.join(format!(
        "UniversalMac_{}_{}_Restore.ipsw",
        image.version, image.build
    ));
    if path.is_file()
        && let Some(receipt) = restore::receipt(&path)?
    {
        // Reused only as the catalog entry it was downloaded from.
        ensure!(
            receipt.image == image && receipt.catalog_sha1_verified,
            "{} was downloaded from another catalog entry than Apple lists for macOS {} now; \
             move it away, or pass it with --ipsw",
            path.display(),
            image.build
        );
        eprintln!("Using the downloaded macOS image");
    } else {
        restore::download(&image, &path)?;
    }
    Ok(path)
}

/// The console boot-args of the bundles setup prepares: the kernel's text
/// on the screen (`-v`) and on the debug UART, which the engine records in
/// the boot log as `GUEST|` lines, so a run on a PC can be read afterwards.
pub(super) const CONSOLE_BOOT_ARGS: &str = "--boot-args=-v serial=3 serial-device-name=uart0";

/// Inputs and destination for one erase or update restore bundle.
pub(super) struct RamdiskOptions<'a> {
    pub(super) ipsw: &'a Path,
    pub(super) model: &'a str,
    /// The machine; its restore bundle boots under the AP ticket Apple's
    /// signing service (TSS) personalizes for it.
    pub(super) ecid: Option<Ecid>,
    pub(super) output: &'a Path,
}

/// `stela-dev boot prepare --restore` for the IPSW, model and machine into
/// `output`: the restore bundle the Install macOS choice boots.
pub(super) fn prepare_restore(options: RamdiskOptions<'_>) -> Result<()> {
    prepare_ramdisk(options, false)
}

/// `stela-dev boot prepare --restore --update`: the update restore bundle the
/// Update macOS choice boots.
pub(super) fn prepare_update(options: RamdiskOptions<'_>) -> Result<()> {
    prepare_ramdisk(options, true)
}

/// A restore bundle of the model's erase or, with `update`, update restore
/// identity, for `ecid` under its personalized AP ticket. Then
/// the engine's own board check of the prepared DeviceTree
/// (`machine::soc::board`), so a model the engine cannot run is refused
/// before the USB changes rather than when the PC boots it.
fn prepare_ramdisk(options: RamdiskOptions<'_>, update: bool) -> Result<()> {
    let RamdiskOptions {
        ipsw,
        model,
        ecid,
        output,
    } = options;
    let mut line: Vec<String> = [
        "--ipsw",
        ipsw.to_str().context("IPSW path is not UTF-8")?,
        "--model",
        model,
        "--restore",
        "--output",
        output.to_str().context("temporary path is not UTF-8")?,
        CONSOLE_BOOT_ARGS,
    ]
    .map(str::to_owned)
    .into();
    if update {
        line.push("--update".into());
    }
    if let Some(ecid) = ecid {
        line.extend(["--ecid".into(), ecid.to_string()]);
    }
    boot::prepare(&line)?;
    let tree = fs::read(output.join("devicetree.adt"))?;
    stela_soc_data::board::board(&tree)
        .map_err(|reason| anyhow::anyhow!("Stela cannot run {model} yet: {reason}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::BTreeMap, ffi::OsString};

    fn cache(os: &str, vars: &[(&str, &str)]) -> Option<PathBuf> {
        let vars: BTreeMap<String, OsString> = vars
            .iter()
            .map(|(name, value)| (name.to_string(), OsString::from(value)))
            .collect();
        cache_dir_in(os, |name| vars.get(name).cloned())
    }

    #[test]
    fn downloads_go_to_the_per_user_cache_never_the_temp_directory() {
        let home = [("HOME", "/home/someone")];
        assert_eq!(
            cache("linux", &home),
            Some("/home/someone/.cache/stela".into())
        );
        assert_eq!(
            cache(
                "linux",
                &[("HOME", "/home/someone"), ("XDG_CACHE_HOME", "/data/cache")]
            ),
            Some("/data/cache/stela".into())
        );
        // A relative XDG_CACHE_HOME is invalid by the specification.
        assert_eq!(
            cache(
                "linux",
                &[("HOME", "/home/someone"), ("XDG_CACHE_HOME", "cache")]
            ),
            Some("/home/someone/.cache/stela".into())
        );
        assert_eq!(
            cache("freebsd", &home),
            Some("/home/someone/.cache/stela".into())
        );
        assert_eq!(
            cache(
                "macos",
                &[("HOME", "/Users/someone"), ("XDG_CACHE_HOME", "/x")]
            ),
            Some("/Users/someone/Library/Caches/stela".into())
        );
        let windows = cache(
            "windows",
            &[("LOCALAPPDATA", r"C:\Users\someone\AppData\Local")],
        )
        .unwrap();
        assert!(windows.ends_with("stela/cache") || windows.ends_with(r"stela\cache"));
        assert!(windows.starts_with(r"C:\Users\someone\AppData\Local"));
        for os in ["linux", "macos", "windows"] {
            assert_eq!(cache(os, &[]), None, "{os}");
            assert_eq!(cache(os, &[("HOME", "relative")]), None, "{os}");
        }
        let chosen = cache_dir();
        assert!(
            !chosen.starts_with(std::env::temp_dir()),
            "{}",
            chosen.display()
        );
        assert_eq!(download_dir(), chosen.join("images"));
    }

    #[test]
    fn a_download_needs_its_size_and_the_margin_free() {
        let directory = Path::new("/cache/stela/images");
        let gb = 1_000_000_000;
        assert_eq!(
            shortfall(directory, "Downloading macOS", 17 * gb, 18 * gb),
            None
        );
        assert_eq!(
            shortfall(directory, "Downloading macOS", 17 * gb, 17 * gb).as_deref(),
            Some(
                "Downloading macOS needs 17.1 GB in /cache/stela/images, which has 17.0 GB free; \
                 free some space there or choose another place with --download-dir"
            )
        );
        assert!(shortfall(directory, "x", u64::MAX, u64::MAX - 1).is_some());
        let dir = tempfile::tempdir().unwrap();
        let fresh = dir.path().join("made/on/demand");
        ensure_room(&fresh, "Downloading macOS", 1).unwrap();
        assert!(fresh.is_dir());
        let error = ensure_room(&fresh, "Downloading macOS", u64::MAX / 2)
            .unwrap_err()
            .to_string();
        assert!(error.contains("--download-dir"), "{error}");
    }
}
