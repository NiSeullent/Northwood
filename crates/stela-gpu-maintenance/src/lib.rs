//! GPU maintenance payload validation, shared by its producer and installer.
use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Component, Path, PathBuf},
};
use stela_loader::aux_kc::Kext;

#[cfg(any(target_os = "macos", test))]
mod executable;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(any(target_os = "macos", test))]
mod policy;
#[cfg(any(target_os = "macos", all(test, unix)))]
mod publication;

pub const MANIFEST: &str = "gpu-install.json";
/// The sealed payload as a guest disk, which maintenance reads with `tar`.
pub const MEDIUM: &str = "gpu-install.tar";
pub const KEXT: &str = "StelaGPU.kext";
pub const PLUGIN: &str = "StelaGPU.bundle";
pub const BUNDLE_PATH: &str = "/Library/Extensions/StelaGPU.kext";

#[derive(Parser)]
#[command(
    name = "stela-gpu-maintenance",
    version,
    about = "Install GPU components from macOS maintenance",
    arg_required_else_help = true
)]
pub struct Gpu {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate an installation payload without changing a disk.
    Check { payload: PathBuf },
    /// Run as root in macOS maintenance; uses macOS's APFS and trust APIs.
    Install(Install),
}

#[derive(Args)]
struct Install {
    /// The complete target/guest directory from cargo xtask guest.
    #[arg(long)]
    payload: PathBuf,
    /// Mounted writable Data volume of the system being installed.
    #[arg(long)]
    data_volume: PathBuf,
    /// Mounted Preboot volume, containing the volume group's directory.
    #[arg(long)]
    preboot: PathBuf,
    #[arg(long, value_parser = parse_uuid)]
    volume_group: [u8; 16],
    /// The installed system's next-stage IM4M under this Preboot group.
    #[arg(long)]
    manifest: PathBuf,
    /// The installed system's splat manifest, when present.
    #[arg(long)]
    splat: Option<PathBuf>,
    /// Apple's native linker, required for the on-disk arm64e kext.
    #[arg(long, default_value = "/usr/bin/ld")]
    linker: PathBuf,
}

impl Gpu {
    pub fn run(self) -> Result<()> {
        match self.command {
            Command::Check { payload } => {
                let manifest = Payload::load(&payload)?;
                println!(
                    "GPU payload verified: {} files, kext UUID {}",
                    manifest.files.len(),
                    manifest.kext_uuid
                );
                Ok(())
            }
            Command::Install(options) => {
                #[cfg(target_os = "macos")]
                {
                    macos::install(options)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = options;
                    anyhow::bail!(
                        "GPU installation requires a macOS maintenance boot; build and payload checking work on this host"
                    )
                }
            }
        }
    }
}

fn parse_uuid(text: &str) -> Result<[u8; 16], String> {
    let bytes = text.as_bytes();
    if bytes.len() != 36 || [8, 13, 18, 23].iter().any(|&at| bytes[at] != b'-') {
        return Err("expected a hyphenated volume-group UUID".into());
    }
    let digits: String = text.chars().filter(|&ch| ch != '-').collect();
    let mut uuid = [0; 16];
    hex::decode_to_slice(digits, &mut uuid).map_err(|error| error.to_string())?;
    if uuid == [0; 16] {
        return Err("volume-group UUID is zero".into());
    }
    Ok(uuid)
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Payload {
    format: u32,
    pub kext_uuid: String,
    /// Each relative regular file and its SHA-256, in bytewise path order.
    pub files: BTreeMap<String, String>,
}

impl Payload {
    /// Seal only the two installable components. Build intermediates stay out.
    pub fn write(root: &Path) -> Result<()> {
        let files = inventory(root)?;
        let mut payload = Self {
            format: 1,
            kext_uuid: String::new(),
            files,
        };
        payload.kext_uuid = hex::encode(payload.source_uuid(root)?);
        let manifest = serde_json::to_vec_pretty(&payload)?;
        fs::write(root.join(MANIFEST), &manifest)?;
        fs::write(root.join(MEDIUM), payload.medium(root, &manifest)?)?;
        Ok(())
    }

    /// A bare ustar stream of the manifest, then every file: single-user
    /// macOS mounts no removable filesystem, but its `tar` reads a disk.
    /// The manifest comes first so maintenance can recognize the disk.
    fn medium(&self, root: &Path, manifest: &[u8]) -> Result<Vec<u8>> {
        let mut stream = Vec::new();
        append_ustar(&mut stream, MANIFEST, manifest)?;
        for path in self.files.keys() {
            append_ustar(&mut stream, path, &fs::read(root.join(path))?)?;
        }
        // Two zero blocks end the archive; whole mebibytes make it a disk.
        stream.resize(
            (stream.len() + 2 * USTAR_BLOCK).next_multiple_of(1 << 20),
            0,
        );
        Ok(stream)
    }

    pub fn load(root: &Path) -> Result<Self> {
        let payload: Self = serde_json::from_slice(&fs::read(root.join(MANIFEST))?)?;
        ensure!(
            payload.format == 1,
            "unsupported GPU payload format {}",
            payload.format
        );
        for (path, hash) in &payload.files {
            ensure!(
                safe_relative(path) && hash.len() == 64 && hex::decode(hash).is_ok(),
                "invalid GPU payload entry {path}"
            );
        }
        ensure!(
            payload.files == inventory(root)?,
            "GPU payload file inventory or SHA-256 differs from its manifest"
        );
        ensure!(
            payload.kext_uuid == hex::encode(payload.source_uuid(root)?),
            "GPU payload source UUID differs from its kext objects"
        );
        use stela_loader::plugin::{self, BundleSeal, CODE_RESOURCES_PATH};
        let bundle = root.join(PLUGIN);
        let executable = fs::read(bundle.join("Contents/MacOS/StelaGPU"))?;
        let mut actual = match fs::read(bundle.join(CODE_RESOURCES_PATH)) {
            Ok(resources) => {
                let info = fs::read(bundle.join("Contents/Info.plist"))?;
                plugin::bundle_code_hashes(&executable, BundleSeal::new(&info, &resources))?
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                plugin::code_hashes(&executable)?
            }
            Err(error) => return Err(error).context("reading GPU bundle CodeResources"),
        };
        let mut listed = stela_loader::trust_cache::code_hashes(&fs::read(
            root.join(KEXT)
                .join(stela_loader::aux_kc::TRUSTED_CODE_FILE),
        )?)?;
        actual.sort_unstable();
        listed.sort_unstable();
        ensure!(
            actual == listed,
            "GPU payload code hashes differ from its Metal executable"
        );
        Ok(payload)
    }

    pub fn source_uuid(&self, root: &Path) -> Result<[u8; 16]> {
        let info = fs::read(root.join(KEXT).join("Info.plist"))?;
        let objects = self
            .files
            .keys()
            .filter(|name| name.starts_with(&format!("{KEXT}/")) && name.ends_with(".o"))
            .map(|name| fs::read(root.join(name)))
            .collect::<std::io::Result<Vec<_>>>()?;
        ensure!(!objects.is_empty(), "GPU payload has no kext objects");
        let properties = plist::Value::from_reader_xml(&info[..])?;
        ensure!(
            properties
                .as_dictionary()
                .and_then(|dict| dict.get("CFBundleIdentifier"))
                .and_then(plist::Value::as_string)
                == Some("com.stela.StelaGPU"),
            "GPU payload has another kext identifier"
        );
        Ok(Kext {
            info: &info,
            objects: objects.iter().map(Vec::as_slice).collect(),
            bundle_path: BUNDLE_PATH.into(),
            trusted_code: vec![],
        }
        .uuid())
    }
}

const USTAR_BLOCK: usize = 512;

/// One regular file, owned by root, with no timestamp so builds reproduce.
fn append_ustar(stream: &mut Vec<u8>, path: &str, data: &[u8]) -> Result<()> {
    ensure!(path.len() < 100, "GPU payload path is too long: {path}");
    let mut header = [0; USTAR_BLOCK];
    let size = format!("{:011o}", data.len());
    for (at, field) in [
        (0, path.as_bytes()),
        (100, b"0000644"),
        (108, b"0000000"),
        (116, b"0000000"),
        (124, size.as_bytes()),
        (136, b"00000000000"),
        (148, b"        "),
        (156, b"0"),
        (257, b"ustar\x0000"),
    ] {
        header[at..at + field.len()].copy_from_slice(field);
    }
    let sum: u32 = header.iter().map(|&byte| u32::from(byte)).sum();
    header[148..155].copy_from_slice(format!("{sum:06o}\0").as_bytes());
    stream.extend_from_slice(&header);
    stream.extend_from_slice(data);
    stream.resize(stream.len().next_multiple_of(USTAR_BLOCK), 0);
    Ok(())
}

fn safe_relative(path: &str) -> bool {
    !path.contains(['\\', ':'])
        && path
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
        && Path::new(path)
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
}

fn inventory(root: &Path) -> Result<BTreeMap<String, String>> {
    fn walk(root: &Path, relative: &Path, files: &mut BTreeMap<String, String>) -> Result<()> {
        let path = root.join(relative);
        let metadata =
            fs::symlink_metadata(&path).with_context(|| format!("reading {}", path.display()))?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "GPU payload contains a symlink: {}",
            path.display()
        );
        if metadata.is_dir() {
            for child in fs::read_dir(&path)? {
                walk(root, &relative.join(child?.file_name()), files)?;
            }
        } else {
            ensure!(
                metadata.is_file() && metadata.len() <= 512 * 1024 * 1024,
                "GPU payload is not a bounded regular file: {}",
                path.display()
            );
            let name = relative
                .to_str()
                .context("GPU payload path is not UTF-8")?
                .replace('\\', "/");
            ensure!(safe_relative(&name), "invalid GPU payload path {name}");
            files.insert(name, stela_formats::sha256(&fs::read(&path)?));
        }
        Ok(())
    }
    let mut files = BTreeMap::new();
    for component in [KEXT, PLUGIN] {
        walk(root, Path::new(component), &mut files)?;
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_and_volume_groups_cannot_escape_the_target() {
        for path in ["/a", "../a", "a/../b", "a\\b", "C:/a", "a//b", "a/./b"] {
            assert!(!safe_relative(path), "{path}");
        }
        assert!(safe_relative("StelaGPU.bundle/Contents/MacOS/StelaGPU"));
        assert_eq!(
            parse_uuid("ABCDEF01-2345-6789-ABCD-EF0123456789").unwrap()[0],
            0xab
        );
        assert!(parse_uuid("00000000-0000-0000-0000-000000000000").is_err());
        assert!(parse_uuid("ABCDEF0123456789ABCDEF0123456789").is_err());
    }

    #[test]
    fn maintenance_command_requires_explicit_target_mounts_and_group() {
        use clap::Parser;
        assert!(
            Gpu::try_parse_from([
                "stela-gpu-maintenance",
                "install",
                "--payload",
                "guest",
                "--data-volume",
                "/Volumes/Data",
                "--preboot",
                "/Volumes/Preboot",
                "--volume-group",
                "ABCDEF01-2345-6789-ABCD-EF0123456789",
                "--manifest",
                "/Volumes/Preboot/group/boot/apticket.im4m",
            ])
            .is_ok()
        );
        assert!(
            Gpu::try_parse_from(["stela-gpu-maintenance", "install", "--payload", "guest"])
                .is_err()
        );
    }

    #[test]
    fn medium_opens_with_the_manifest_and_is_a_whole_disk() -> Result<()> {
        let root = tempfile::tempdir()?;
        for directory in [KEXT, PLUGIN] {
            fs::create_dir(root.path().join(directory))?;
        }
        fs::write(root.path().join(KEXT).join("unit.o"), b"object")?;
        let payload = Payload {
            format: 1,
            kext_uuid: String::new(),
            files: inventory(root.path())?,
        };
        let stream = payload.medium(root.path(), b"{}")?;
        assert_eq!(stream.len(), 1 << 20);
        assert_eq!(&stream[..MANIFEST.len() + 1], b"gpu-install.json\0");
        assert_eq!(&stream[USTAR_BLOCK..USTAR_BLOCK + 2], b"{}");
        let file = &stream[2 * USTAR_BLOCK..];
        assert!(file.starts_with(b"StelaGPU.kext/unit.o\0"));
        assert_eq!(&file[124..135], b"00000000006");
        assert_eq!(&file[257..265], b"ustar\x0000");
        assert_eq!(&file[USTAR_BLOCK..USTAR_BLOCK + 6], b"object");

        let recorded = u32::from_str_radix(str::from_utf8(&file[148..154])?, 8)?;
        let blanked = file[..USTAR_BLOCK].iter().enumerate();
        let sum: u32 = blanked
            .map(|(at, &byte)| match at {
                148..156 => u32::from(b' '),
                _ => u32::from(byte),
            })
            .sum();
        assert_eq!(recorded, sum);

        let mut stream = Vec::new();
        assert!(append_ustar(&mut stream, &"a".repeat(100), b"").is_err());
        Ok(())
    }

    #[test]
    fn inventory_detects_modified_and_extra_files() -> Result<()> {
        let root = tempfile::tempdir()?;
        for directory in [KEXT, PLUGIN] {
            fs::create_dir(root.path().join(directory))?;
        }
        let path = root.path().join(KEXT).join("unit.o");
        fs::write(&path, b"first")?;
        let before = inventory(root.path())?;
        fs::write(&path, b"other")?;
        assert_ne!(before, inventory(root.path())?);
        fs::write(root.path().join(PLUGIN).join("extra"), b"extra")?;
        assert_eq!(inventory(root.path())?.len(), 2);
        Ok(())
    }

    #[test]
    #[cfg(unix)]
    fn payload_symlinks_are_refused_before_following_their_targets() -> Result<()> {
        let root = tempfile::tempdir()?;
        fs::create_dir(root.path().join(KEXT))?;
        fs::create_dir(root.path().join(PLUGIN))?;
        std::os::unix::fs::symlink("/etc/passwd", root.path().join(KEXT).join("outside"))?;
        assert!(
            inventory(root.path())
                .unwrap_err()
                .to_string()
                .contains("symlink")
        );
        Ok(())
    }
}
