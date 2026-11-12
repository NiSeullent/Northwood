//! Install the UEFI mount on a USB's EFI system partition as its
//! removable-media boot loader, `\EFI\BOOT\BOOTX64.EFI`, which a PC's
//! firmware starts when it boots from the USB; no other boot manager is
//! involved.
//!
//! `install` copies `stela.efi` there and writes the mount's
//! `EFI/stela/stela.cfg` and `EFI/stela/source.bin`. A partition that
//! already boots another loader is refused and left unchanged: the loader
//! there is replaced only when it is a `stela.efi` (it carries a mount
//! interface marker, [`mount_interface::find`]). Files are replaced
//! atomically. `uninstall` removes what `install` wrote.
//!
//! With `--ipsw`, `install` also writes a restore payload for `stela.efi`
//! under `EFI/stela/images/<build>/` on the ESP or another volume of the
//! same USB: FAT32-safe slices of the model's members, or the whole IPSW,
//! and `payload.json` with every member's SHA-256 and the public AEA key
//! responses. `uninstall` leaves payloads alone.
pub use crate::restore::payload::{
    PayloadLayout, SYNTHETIC_BUILD, SYNTHETIC_MODEL, WriteOptions as PayloadOptions,
    synthetic_ipsw, write as write_payload,
};
use anyhow::{Context, Result, bail, ensure};
use clap::{Args, Subcommand};
use serde_json::json;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};
use stela_loader::mount_interface::{
    self,
    layout::{self, Layout},
};
use stela_runtime::pc::boot_log;

use stela_formats::payload::MAX_SLICE_BYTES;

/// The removable-media boot loader path for x64 (UEFI 2.10, Removable Media
/// Boot Behavior).
pub const LOADER: &str = stela_loader::mount_interface::layout::LOADER;
const LOADER_DIRECTORY: &str = "EFI/BOOT";
#[cfg(test)]
pub(crate) const OPTIONS: &str = "EFI/stela/stela.cfg";
/// `mount_interface::BOOT_LOG_FILE`.
#[cfg(test)]
pub(crate) const BOOT_LOG: &str = "EFI/stela/boot.log";
const SOURCE: &str = "EFI/stela/source.bin";

#[derive(Subcommand)]
pub enum EfiCommand {
    /// Install `stela.efi` as an EFI partition's removable-media boot loader.
    Install(Install),
    /// Remove everything `install` added.
    Uninstall(Uninstall),
}

impl EfiCommand {
    pub fn run(self) -> Result<()> {
        let report = match self {
            Self::Install(command) => install(&command)?,
            Self::Uninstall(command) => uninstall(&command.esp, false)?,
        };
        println!("{report:#}");
        Ok(())
    }
}

#[derive(Args)]
pub struct Install {
    /// The EFI system partition root.
    #[arg(long)]
    pub esp: PathBuf,
    /// The mount application, `stela.efi`.
    #[arg(long)]
    pub image: PathBuf,
    /// Original ARM bytes; omit to keep the installed `source.bin`.
    #[arg(long)]
    pub source: Option<PathBuf>,
    /// Mount options for `stela.cfg`; omit to keep the installed file.
    #[arg(long, allow_hyphen_values = true)]
    pub options: Option<String>,
    /// Write this IPSW's restore payload as well.
    #[arg(long)]
    pub ipsw: Option<PathBuf>,
    /// The Mac model whose identities the payload carries.
    #[arg(long, default_value = crate::setup::PINNED_MODEL, requires = "ipsw")]
    pub model: String,
    /// The volume root for the payload; defaults to `--esp`.
    #[arg(long, requires = "ipsw")]
    pub payload_root: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = PayloadLayout::Split, requires = "ipsw")]
    pub payload_layout: PayloadLayout,
    /// Largest slice of a split payload; FAT32 allows 4 GiB - 1.
    #[arg(long, default_value_t = MAX_SLICE_BYTES, hide = true)]
    pub slice_bytes: u64,
    /// Leave out the public AEA keys; the restore then needs the network.
    #[arg(long, requires = "ipsw")]
    pub no_aea_keys: bool,
}

#[derive(Args)]
pub struct Uninstall {
    /// The EFI system partition root.
    #[arg(long)]
    esp: PathBuf,
}

/// Replace `path` through a sibling temporary file, so a failed write leaves
/// the old file in place; with `fresh`, only when nothing is there yet.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8], fresh: bool) -> Result<()> {
    let parent = path.parent().context("a file without a parent directory")?;
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    file.write_all(bytes)?;
    file.as_file().sync_all()?;
    match fresh {
        true => file.persist_noclobber(path).map_err(|error| error.error)?,
        false => file.persist(path).map_err(|error| error.error)?,
    };
    // FAT may refuse to sync a directory; the file itself is already synced.
    let _ = fs::File::open(parent).and_then(|directory| directory.sync_all());
    Ok(())
}

/// Whether the partition's boot loader is a `stela.efi`; `None` when it has
/// none.
pub(crate) fn loader_is_stela(esp: &Path) -> Result<Option<bool>> {
    let path = esp.join(LOADER);
    match fs::read(&path) {
        Ok(image) => Ok(Some(mount_interface::find(&image).is_some())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("cannot read {}", path.display())),
    }
}

/// Refuse a partition whose boot loader is not Stela's, changing nothing.
pub(crate) fn check_loader(esp: &Path) -> Result<()> {
    ensure!(
        loader_is_stela(esp)? != Some(false),
        "{} already boots another loader ({}). Stela boots the USB itself, as its only boot \
         loader, and does not replace another one: use a USB whose EFI partition has no \
         {LOADER}, or erase the USB first.",
        esp.display(),
        esp.join(LOADER).display()
    );
    Ok(())
}

/// The standalone legacy EFI command cannot maintain a Northwood setup.
fn check_legacy_command(esp: &Path) -> Result<()> {
    let record = esp.join(Layout::Northwood.esp_path(layout::RECORD));
    ensure!(
        !record.try_exists()?,
        "{} belongs to a Northwood setup; use the setup commands to maintain it",
        record.display()
    );
    Ok(())
}

/// Install or replace the mount, and return a report of what changed.
pub fn install(command: &Install) -> Result<serde_json::Value> {
    let esp = &command.esp;
    check_legacy_command(esp)?;
    let source = match &command.source {
        Some(source) => {
            Some(fs::read(source).with_context(|| format!("cannot read {}", source.display()))?)
        }
        None if esp.join(SOURCE).is_file() => None,
        None => bail!("pass --source: {} has no {SOURCE}", esp.display()),
    };
    // Check the partition before writing the payload, which goes before the
    // loader, so the USB never boots Stela with a half-written payload.
    check_loader(esp)?;
    let payload = match &command.ipsw {
        Some(ipsw) => Some(crate::restore::payload::write(
            &crate::restore::payload::WriteOptions {
                image: ipsw,
                model: &command.model,
                root: command.payload_root.as_deref().unwrap_or(esp),
                layout: command.payload_layout,
                slice_bytes: command.slice_bytes,
                fetch_keys: !command.no_aea_keys,
            },
        )?),
        None => None,
    };
    let image = fs::read(&command.image)
        .with_context(|| format!("cannot read {}", command.image.display()))?;
    let mut report = add_mount(esp, &image, source.as_deref(), command.options.as_deref())?;
    report["payload"] = json!(payload);
    Ok(report)
}

/// Copy the ARM source when given and the options, then the loader, over
/// whatever loader is there: callers check it first. `stela setup` uses this
/// without a source: install mode runs none.
pub(crate) fn add_mount(
    esp: &Path,
    image: &[u8],
    source: Option<&[u8]>,
    options: Option<&str>,
) -> Result<serde_json::Value> {
    add_mount_in(esp, image, source, options, Layout::Legacy)
}

pub(crate) fn add_mount_in(
    esp: &Path,
    image: &[u8],
    source: Option<&[u8]>,
    options: Option<&str>,
    layout: Layout,
) -> Result<serde_json::Value> {
    validate_mount_image(image, layout)?;
    prepare_mount(esp, source, options, layout)?;
    publish_loader(esp, image)?;
    let [loader, options, source, boot_log] = installed_paths(esp, layout);
    Ok(json!({
        "loader": loader,
        "source": source.is_file().then_some(source),
        "options": options.is_file().then_some(options),
        "boot_log": boot_log,
    }))
}

/// Northwood setup needs both the path and plist contracts.
pub(crate) fn supports_northwood_layout(image: &[u8]) -> bool {
    let marker = northwood_config::MOUNT_CAPABILITY;
    Layout::supports_northwood(image) && image.windows(marker.len()).any(|window| window == marker)
}

pub(crate) fn validate_mount_image(image: &[u8], layout: Layout) -> Result<()> {
    ensure!(image.starts_with(b"MZ"), "the image is not a PE image");
    ensure!(
        mount_interface::find(image).is_some(),
        "the image is not a stela.efi: it carries no mount interface marker"
    );
    ensure!(
        layout == Layout::Legacy || supports_northwood_layout(image),
        "the EFI image lacks support for the Northwood setup layout or nwconfig.plist"
    );
    Ok(())
}

/// Put the selected layout's files in place before activating the loader.
pub(crate) fn prepare_mount(
    esp: &Path,
    source: Option<&[u8]>,
    options: Option<&str>,
    layout: Layout,
) -> Result<()> {
    fs::create_dir_all(esp.join(layout.esp_dir()))?;
    if let Some(source) = source {
        write_atomic(&esp.join(layout.esp_path(layout::SOURCE)), source, false)?;
    }
    if let Some(options) = options {
        write_atomic(
            &esp.join(layout.esp_path(layout::OPTIONS)),
            options.as_bytes(),
            false,
        )?;
    }
    // An existing log keeps its earlier boots. `stela.efi` writes its
    // clusters wherever the volume put them.
    let path = esp.join(layout.esp_path(layout::BOOT_LOG));
    if !path.is_file() {
        write_atomic(&path, &vec![0; boot_log::LOG_BYTES as usize], true)?;
    }
    Ok(())
}

/// Activate a previously prepared mount at the shared UEFI entry point.
pub(crate) fn publish_loader(esp: &Path, image: &[u8]) -> Result<()> {
    fs::create_dir_all(esp.join(LOADER_DIRECTORY))?;
    write_atomic(&esp.join(LOADER), image, false)
}

/// The shared loader and the selected layout's mount files, for reports.
pub(crate) fn installed_paths(esp: &Path, layout: Layout) -> [PathBuf; 4] {
    [
        esp.join(LOADER),
        esp.join(layout.esp_path(layout::OPTIONS)),
        esp.join(layout.esp_path(layout::SOURCE)),
        esp.join(layout.esp_path(layout::BOOT_LOG)),
    ]
}

/// Remove the loader, when it is Stela's, and the files `install` wrote,
/// and return a report.
pub fn uninstall(esp: &Path, owned: bool) -> Result<serde_json::Value> {
    check_legacy_command(esp)?;
    uninstall_in(esp, owned, Layout::Legacy)
}

pub(crate) fn uninstall_in(esp: &Path, owned: bool, layout: Layout) -> Result<serde_json::Value> {
    let stela = match owned {
        true => Some(true),
        false => loader_is_stela(esp)?,
    };
    let [loader, options, source, boot_log] = installed_paths(esp, layout);
    let remove = (stela == Some(true))
        .then_some(loader)
        .into_iter()
        .chain([options, source, boot_log]);
    let mut files = Vec::new();
    for file in remove {
        if file.is_file() {
            fs::remove_file(&file)?;
            files.push(file);
        }
    }
    // Directories go only when nothing else is left in them.
    for directory in [esp.join(layout.esp_dir()), esp.join(LOADER_DIRECTORY)] {
        if fs::read_dir(&directory).is_ok_and(|mut entries| entries.next().is_none()) {
            fs::remove_dir(&directory)?;
        }
    }
    Ok(json!({
        "files": files,
        "kept_loader": (stela == Some(false)).then(|| esp.join(LOADER)),
    }))
}
