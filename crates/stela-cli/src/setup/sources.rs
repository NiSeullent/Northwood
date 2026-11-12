//! Where an update finds the restore image members it prepares boot
//! bundles from, trying in order:
//!
//! 1. `--ipsw`;
//! 2. an image or boot archive of the build already on this computer: in
//!    `--download-dir`, this user's cache, the invoking user's cache under
//!    `sudo` and root's cache when it is readable ([`image_caches`]);
//! 3. the USB's own full install medium, which holds the image byte for
//!    byte (`medium::image`);
//! 4. the image's boot members alone, read by byte ranges from Apple's
//!    servers (`restore::fetch_recovery`): about 2.2 GB of macOS 27.2's
//!    27 GB.
//!
//! [`locate`] decides without downloading or changing anything, so the
//! update says what it reads and downloads before it starts.
use super::{Environment, host::bytes_text, medium};
use crate::restore;
use anyhow::{Context, Result};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Where the restore image members come from.
#[derive(Debug)]
pub(super) enum Source {
    /// `--ipsw`.
    Given(PathBuf),
    /// An image or boot archive of the build on this computer.
    Cached(PathBuf),
    /// The USB's full install medium of the build.
    Medium(medium::Medium),
    /// The image's boot members on Apple's servers: `bytes` of the image's
    /// `whole`.
    Remote { bytes: u64, whole: u64 },
}

/// What [`locate`] looks for, and where.
pub(super) struct Lookup<'a> {
    pub(super) build: &'a str,
    pub(super) model: &'a str,
    pub(super) ipsw: Option<&'a Path>,
    pub(super) download_dir: Option<&'a Path>,
    /// The setup's install medium of the build.
    pub(super) medium: Option<&'a medium::Medium>,
}

impl Lookup<'_> {
    /// The directories searched for a downloaded image, `--download-dir`
    /// first.
    fn directories(&self, environment: &Environment<'_>) -> Vec<PathBuf> {
        let mut directories: Vec<PathBuf> = self
            .download_dir
            .map(Path::to_path_buf)
            .into_iter()
            .collect();
        for directory in environment.image_caches {
            if !directories.contains(directory) {
                directories.push(directory.clone());
            }
        }
        directories
    }

    /// Where downloads and boot archives read from the USB go.
    fn download_dir(&self, environment: &Environment<'_>) -> PathBuf {
        self.download_dir
            .map(Path::to_path_buf)
            .or_else(|| environment.image_caches.first().cloned())
            .unwrap_or_else(super::download_dir)
    }
}

/// Decide where `lookup`'s members come from, downloading nothing. Fails
/// only when none is found and Apple's servers cannot be asked for the
/// image's boot members.
pub(super) fn locate(environment: &Environment<'_>, lookup: &Lookup<'_>) -> Result<Source> {
    if let Some(ipsw) = lookup.ipsw {
        return Ok(Source::Given(ipsw.to_path_buf()));
    }
    let directories = lookup.directories(environment);
    if let Some(path) = directories
        .iter()
        .find_map(|directory| cached(directory, lookup.build, lookup.model))
    {
        return Ok(Source::Cached(path));
    }
    if let Some(medium) = lookup
        .medium
        .filter(|m| !m.remote && m.build == lookup.build && m.problems().is_empty())
    {
        return Ok(Source::Medium(medium.clone()));
    }
    let searched = directories
        .iter()
        .map(|d| d.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let medium = match lookup.medium {
        Some(m) if m.remote => "this USB streams the image from Apple's servers",
        Some(_) => "this USB's copy of the image is damaged",
        None => "this USB holds no copy of the image",
    };
    match (environment.recovery_size)(lookup.build, lookup.model) {
        Ok((bytes, whole)) => Ok(Source::Remote { bytes, whole }),
        Err(error) => anyhow::bail!(
            "the macOS {build} restore image is not on this computer (looked in {searched}) and \
             {medium}, and Apple's servers could not be asked for its boot files ({error:#}). Pass \
             the image with --ipsw FILE, or connect to the network and run this again, which \
             downloads only its boot files, not the whole image",
            build = lookup.build,
        ),
    }
}

/// The plan line that names `source`.
pub(super) fn describe(source: &Source, build: &str) -> String {
    match source {
        Source::Given(path) => format!("from {}", path.display()),
        Source::Cached(path) => format!("from {}, downloaded earlier", path.display()),
        Source::Medium(medium) => format!(
            "from this USB's copy of macOS {build} in {}; nothing is downloaded",
            medium.directory().display()
        ),
        Source::Remote { bytes, whole } => format!(
            "download {} of the macOS {build} image's {} from Apple: no copy of it is on this \
             USB or this computer (pass --ipsw FILE to use one you have)",
            bytes_text(*bytes),
            bytes_text(*whole)
        ),
    }
}

/// A restore image file to prepare from, and the temporary directory that
/// holds it when it was read from the USB.
pub(super) struct Obtained {
    pub(super) path: PathBuf,
    _temporary: Option<tempfile::TempDir>,
}

/// The file `source` names, reading or downloading what it takes.
pub(super) fn obtain(
    environment: &Environment<'_>,
    source: &Source,
    lookup: &Lookup<'_>,
) -> Result<Obtained> {
    let file = |path: &Path| Obtained {
        path: path.to_path_buf(),
        _temporary: None,
    };
    match source {
        Source::Given(path) | Source::Cached(path) => Ok(file(path)),
        Source::Medium(medium) => {
            // Beside downloads, never the system temp directory, which is
            // often too small; removed once the bundles are prepared.
            let directory = lookup.download_dir(environment);
            fs::create_dir_all(&directory)
                .with_context(|| format!("cannot create {}", directory.display()))?;
            let temporary = tempfile::Builder::new()
                .prefix("usb-")
                .tempdir_in(&directory)?;
            let path = temporary.path().join("boot.ipsw");
            eprintln!(
                "Reading the boot files of macOS {} from this USB",
                lookup.build
            );
            restore::payload::trim(
                medium::image::open(medium)?,
                lookup.model,
                &path,
                "Reading boot files from the USB",
            )
            .with_context(|| format!("reading {}", medium.directory().display()))?;
            Ok(Obtained {
                path,
                _temporary: Some(temporary),
            })
        }
        Source::Remote { bytes, whole } => {
            eprintln!(
                "Downloading {} of the macOS {} image's {}: its boot files, read by range. No \
                 copy of the image is on this USB or this computer; pass --ipsw FILE to use one \
                 you have",
                bytes_text(*bytes),
                lookup.build,
                bytes_text(*whole)
            );
            let path = (environment.fetch_recovery)(
                &lookup.download_dir(environment),
                lookup.build,
                lookup.model,
            )?;
            Ok(file(&path))
        }
    }
}

/// A complete image or boot archive of `build` for `model` in `directory`:
/// the boot archive an update or setup downloaded, the one a full setup
/// left, or a whole image.
fn cached(directory: &Path, build: &str, model: &str) -> Option<PathBuf> {
    let mut candidates = vec![
        directory.join(format!("Recovery_{build}_{}.ipsw", model.replace(',', "_"))),
        // Where a first setup keeps its downloads until it finishes.
        directory.join("setup").join("boot.ipsw"),
    ];
    if let Ok(entries) = fs::read_dir(directory) {
        let mut images: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with("UniversalMac_")
                            && name.ends_with(&format!("_{build}_Restore.ipsw"))
                    })
            })
            .collect();
        images.sort();
        candidates.extend(images);
    }
    candidates.into_iter().find(|path| {
        path.is_file()
            && restore::payload::identity(path, model).is_ok_and(|(_, found)| found == build)
    })
}

/// The directories downloaded images may be in: this user's cache, and on
/// Unix the invoking user's under `sudo` and root's, so a setup run with
/// `sudo` and an update run without find each other's downloads, and a
/// dry run looks where the real run does.
pub(super) fn image_caches() -> Vec<PathBuf> {
    let own = super::download_dir();
    #[cfg(not(unix))]
    let directories = vec![own];
    #[cfg(unix)]
    let directories = {
        let mut directories = vec![own];
        let homes = std::env::var("SUDO_USER")
            .ok()
            .filter(|user| !user.is_empty() && user != "root")
            .and_then(|user| home_of(&user))
            .into_iter()
            .chain(std::iter::once(PathBuf::from(match std::env::consts::OS {
                "macos" => "/var/root",
                _ => "/root",
            })));
        for home in homes {
            let home = home.into_os_string();
            if let Some(cache) = super::prepare::cache_dir_in(std::env::consts::OS, |name| {
                (name == "HOME").then(|| home.clone())
            }) {
                let images = cache.join("images");
                if !directories.contains(&images) {
                    directories.push(images);
                }
            }
        }
        directories
    };
    directories
}

/// The home directory of the user `name`: `/etc/passwd`'s, or macOS's
/// `/Users/<name>`.
#[cfg(unix)]
fn home_of(name: &str) -> Option<PathBuf> {
    if !name
        .bytes()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
        || name.starts_with('.')
    {
        return None;
    }
    let passwd = fs::read_to_string("/etc/passwd").unwrap_or_default();
    passwd
        .lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(':').collect();
            (fields.len() >= 6 && fields[0] == name).then(|| PathBuf::from(fields[5]))
        })
        .find(|home| home.is_absolute())
        .or_else(|| (std::env::consts::OS == "macos").then(|| Path::new("/Users").join(name)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caches_start_with_this_users() {
        let caches = image_caches();
        assert_eq!(caches[0], super::super::download_dir());
        let mut unique = caches.clone();
        unique.dedup();
        assert_eq!(unique, caches);
    }

    #[cfg(unix)]
    #[test]
    fn invoking_user_names_are_checked() {
        assert_eq!(home_of("../etc"), None);
        assert_eq!(home_of(".hidden"), None);
        assert_eq!(home_of("a/b"), None);
        if cfg!(target_os = "linux") {
            assert_eq!(home_of("root"), Some(PathBuf::from("/root")));
        }
    }
}
