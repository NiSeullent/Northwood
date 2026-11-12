//! `stela setup`: prepare a USB to install macOS 27 onto
//! a disk of the PC and boot it, and keep it working over the installed
//! system's life.
//!
//! Setup writes the install medium (`setup::medium`): on a USB of 32 GB or
//! more the whole IPSW on its FAT32 data partition, else the IPSW's
//! inventory, from which `stela.efi` streams the IPSW from Apple's servers
//! during the install ([`default_install_image`]). It prepares each machine's restore bundle (`stela
//! boot prepare --restore`) and adds an Install macOS startup choice
//! (`stela.efi --install`), which restores macOS and its paired recoveryOS
//! and System Recovery onto the Mac's internal SSD, as a Mac's erase
//! restore does. An erase restore needs the macOS system image, so a
//! download's recovery members (`restore::payload::trim`) only prepare the
//! boot bundles. The macOS choice
//! boots the installed system afterwards. The restore personalizes its root
//! tickets, its Cryptex1 ticket and the recoveryOS LocalPolicy with Apple's
//! signing server (TSS); setup asks for them while it prepares the bundle
//! (`boot::ticket::staged`), so an install with macOS on the USB needs no
//! network.
//!
//! One command checks the USB and the restore image without writing
//! anything, prepares the restore bundle on this computer, then copies it
//! to the USB's FAT32 data partition, which the PC's firmware can read and
//! which also holds the install medium and the translation store. Setup
//! installs the loader as `EFI/BOOT/BOOTX64.EFI` and records its files
//! beside the boot settings in `EFI/northwood/setup.json` on new Northwood
//! media. Existing Stela media keep their layout. `--dry-run` stops after
//! the checks. `uninstall` removes the files the record names.
//!
//! Each Stela machine has its own ECID (`setup::ecid`), drawn once, kept in
//! the record and given to every bundle prepared for it. Without `--target`,
//! Install macOS asks for a disk of the PC in `stela.efi`, or a raw
//! partition of the USB is the Mac's internal SSD. With `--target`, setup
//! runs on the PC itself and names the PC's own disk (`setup::installed`):
//! one empty partition becomes the Mac's internal SSD and keeps the machine
//! state (NVRAM, the SEP's state, the boot log) in its last bytes, outside
//! what the guest sees, or `--nor` names a partition of its own for that
//! state. The USB then boots the installed macOS from that disk, or the
//! disk's own recoveryOS when NVRAM asks for it or the disk holds no system,
//! as a Mac does. Further disks become further installs of the same setup
//! (`setup::machines`), each its own machine, chosen at boot from
//! `stela.efi`'s startup choices. Setup writes no disk but the USB, except
//! fresh machine state with `--initialize-nor` and NVRAM with
//! `nvram next-boot`.
//!
//! An existing setup is updated in place (`setup::update`): `update stela`
//! installs a new `stela.efi` and prepares the boot bundles again when it
//! reads another schema or this `stela` prepares another bundle revision
//! than theirs, and `update os` a new macOS build's install medium and
//! restore bundles. Every path takes `--dry-run`. A `stela.efi`
//! whose mount interface (`stela_loader::mount_interface`) differs from this
//! `stela`'s is refused.
mod addresses;
mod assets;
mod bundle;
mod command;
mod create;
mod doctor;
mod ecid;
mod elevation;
mod fdr;
mod firmware;
mod format;
pub(crate) mod host;
mod installed;
mod machines;
mod medium;
mod northwood_config;
mod nwconfig_publish;
mod prepare;
mod record;
mod sources;
mod state;
mod swap;
mod translations;
mod uninstall;
mod update;
mod volume_log;
mod volume_swap;
mod wifi;
mod wizard;

pub(crate) use command::Setup;
pub(crate) use elevation::hold_window;
pub(crate) use installed::NextBoot;
pub(crate) use prepare::{cache_dir, download_dir, ensure_room};

use crate::{efi, restore};
use anyhow::{Context, Result, bail, ensure};
use bundle::Kind;
use clap::Args;
use create::{apply_install, describe_install, plan_install, restore_bundles, setup, state_line};
use ecid::Ecid;
use host::{FileSystem, Host};
use prepare::{
    Prepared, check_space, copy_bundles, fetch, prepare_kind, prepare_restore, prepare_update,
    supported,
};
use record::{Record, publish, read_record, remove_bundles, write_record};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};
use stela_formats::builds;
use stela_loader::mount_interface::{
    self, Interface,
    layout::{self, Layout},
};
use uninstall::uninstall;

/// `relative`, spelled with `/`, below `base` with the host's own separators.
fn join_slashed(base: &Path, relative: &str) -> PathBuf {
    let mut path = base.to_path_buf();
    path.extend(relative.split('/'));
    path
}

/// The Mac models setup presents, with their names: the M2 Mac mini, the
/// board the engine is tested on. Other Macs stay listed in the wizard,
/// greyed out, and are refused by name.
pub(crate) const SUPPORTED_MODELS: &[(&str, &str)] = &[("Mac14,3", "Mac mini M2")];
/// The Mac model Stela presents by default.
pub(crate) const PINNED_MODEL: &str = SUPPORTED_MODELS[0].0;

/// Setup can present Mac `model`.
pub(crate) fn supported_model(model: &str) -> bool {
    SUPPORTED_MODELS
        .iter()
        .any(|(supported, _)| *supported == model)
}

/// Refuse a Mac model `models` does not hold, naming the ones it does.
fn check_model(models: &[(&str, &str)], model: &str) -> Result<()> {
    if models.iter().any(|(supported, _)| *supported == model) {
        return Ok(());
    }
    let supported: Vec<String> = models
        .iter()
        .map(|(model, name)| format!("{model} ({name})"))
        .collect();
    bail!(
        "Mac model {model} is not supported yet; Stela supports {}",
        supported.join(", ")
    )
}
#[cfg(test)]
const RECORD: &str = "EFI/stela/setup.json";
/// A margin so no volume is filled to its last byte.
pub(super) const MARGIN: u64 = 64 * 1024 * 1024;
/// The boot loader `stela setup` installs: `stela.efi` itself.
pub(super) const LOADER_PATH: &str = efi::LOADER;
/// The startup choices beside `stela.cfg`.
#[cfg(test)]
const CHOICES_PATH: &str = "EFI/stela/choices.cfg";
/// The startup choices' timeout with several installs; with one, the list
/// opens only when a key is held.
const CHOOSER_SECONDS: u32 = 5;
/// The startup choice that installs macOS onto a setup's one install.
const INSTALL_CHOICE: &str = "Install macOS";
/// The startup choice that updates a setup's one install in place.
const UPDATE_CHOICE: &str = "Update macOS";
/// `stela setup {words} --usb {usb}` in backticks, as a user types it: every
/// maintenance command needs `--usb`.
fn fix(usb: &Path, words: &str) -> String {
    let words = if words.is_empty() {
        String::new()
    } else {
        format!("{words} ")
    };
    format!(
        "`stela setup {words}--usb {}`",
        quoted(&usb.display().to_string())
    )
}

/// `word` as a shell takes it as one argument.
fn quoted(word: &str) -> String {
    if word.is_empty() || word.contains(|c: char| c.is_whitespace() || c == '"') {
        format!("\"{}\"", word.replace('"', "\\\""))
    } else {
        word.into()
    }
}

/// The command that finishes the interrupted first setup `record` describes:
/// the same setup run again resumes it (`create::plan`).
fn rerun(record: &Record, usb: &Path) -> String {
    let mut words = String::new();
    if record.build != builds::DEFAULT.build {
        let _ = write!(words, " --build {}", record.build);
    }
    if record.model != PINNED_MODEL {
        let _ = write!(words, " --model {}", record.model);
    }
    if let [install] = &record.installs[..] {
        let disk = &install.disk;
        if !disk.is_chosen() && !disk.external_disks {
            let _ = write!(words, " --target {}", disk.apple_region);
            if disk.machine_state_offset.is_none() {
                let _ = write!(words, " --nor {}", disk.nor);
            }
        }
        if install.name != machines::FIRST_NAME {
            let _ = write!(words, " --install {}", quoted(&install.name));
        }
    }
    if let Some(network) = &record.wifi {
        let _ = write!(words, " --wifi {}", quoted(&network.ssid));
        if let Some(country) = &network.country {
            let _ = write!(words, " --wifi-country {country}");
        }
    }
    if let Some(ipv4) = record.addresses.ipv4 {
        let _ = write!(words, " --ipv4 {}/{}", ipv4.address, ipv4.prefix);
        if let Some(gateway) = ipv4.gateway {
            let _ = write!(words, " --ipv4-gateway {gateway}");
        }
    }
    if !record.addresses.dns.is_empty() {
        let servers: Vec<String> = record.addresses.dns.iter().map(|s| s.to_string()).collect();
        let _ = write!(words, " --dns {}", servers.join(","));
    }
    for device in &record.usb4_approved {
        let _ = write!(words, " --usb4-approve {device}");
    }
    if record.profile {
        words.push_str(" --profile");
    }
    if record.verify_on_read {
        words.push_str(" --verify-on-read");
    }
    fix(usb, words.trim_start())
}

/// What to do about an interrupted first setup on `usb`.
fn interrupted_text(record: &Record, usb: &Path) -> String {
    format!(
        "the first setup on {} was interrupted. To finish it, run it again: {} (add --ipsw FILE \
         to use a local image). To start over: {}",
        record.esp.display(),
        rerun(record, usb),
        fix(usb, "uninstall")
    )
}

/// A command that found problems: its whole report, which goes to standard
/// output, and a one-line summary, which is the error. Its `Display` is
/// both, so callers that only print the error lose nothing.
#[derive(Debug)]
pub(super) struct Reported {
    pub(super) report: String,
    pub(super) summary: String,
}

impl std::fmt::Display for Reported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.report, self.summary)
    }
}

impl std::error::Error for Reported {}

/// Split a command's result into what goes to standard output and the
/// error: a [`Reported`] failure prints its report and fails with its
/// summary alone.
fn split_report(result: Result<String>) -> (String, Result<()>) {
    match result {
        Ok(text) => (text, Ok(())),
        Err(error) => match error.downcast::<Reported>() {
            Ok(reported) => (reported.report, Err(anyhow::anyhow!(reported.summary))),
            Err(error) => (String::new(), Err(error)),
        },
    }
}

/// Write `report` to `path` instead of the terminal, and say where it went.
fn save_report(path: &Path, report: &str) -> Result<()> {
    fs::write(path, report).with_context(|| format!("cannot write {}", path.display()))?;
    println!(
        "Saved the report to {}",
        crate::boot_log::shown(path).display()
    );
    Ok(())
}

/// The startup choice that restores with a bundle of `kind` on a setup
/// with one install.
fn restore_choice_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Update => UPDATE_CHOICE,
        _ => INSTALL_CHOICE,
    }
}

/// Where an install reads the macOS restore image: on the USB when it is
/// large enough ([`default_install_image`]), else streamed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum InstallImage {
    /// The whole IPSW on the USB: with the signing answers setup fetched,
    /// the install needs no network. Records written before the name
    /// changed say `offline`.
    #[serde(alias = "offline")]
    Full,
    /// The IPSW on Apple's servers, streamed during the install. Records
    /// written before the name changed say `online`.
    #[serde(alias = "online")]
    Network,
}

/// The smallest USB, as the PC reports it, that a full installer is made
/// on: a stick sold as 32 GB holds about 28.6 GiB.
pub(super) const FULL_MIN: u64 = 28 * 1024 * 1024 * 1024;

/// A new setup's installer for a USB of `capacity` bytes (0 when unknown):
/// the whole macOS image on a USB of 32 GB or more, so the install needs no
/// network, and the network installer on a smaller one.
pub(super) fn default_install_image(capacity: u64) -> InstallImage {
    match capacity >= FULL_MIN {
        true => InstallImage::Full,
        false => InstallImage::Network,
    }
}

fn image_is_remote(ipsw: &Path, choice: Option<InstallImage>) -> Result<bool> {
    let complete = restore::payload::holds_system_image(ipsw)?;
    match choice {
        Some(InstallImage::Network) => Ok(true),
        Some(InstallImage::Full) => {
            ensure!(
                complete,
                "{} holds only the recovery members, without the macOS system image this USB's \
                 install reads from it; pass the whole IPSW with --ipsw",
                ipsw.display()
            );
            Ok(false)
        }
        None => Ok(true),
    }
}

/// What one `stela setup` command does.
#[derive(Default)]
enum Action {
    /// Create a setup, or add an install to one with `--target`.
    #[default]
    Create,
    Status,
    Repair,
    UpdateStela,
    UpdateOs,
    Uninstall,
    NextBoot(NextBoot),
    ResetNvram,
    BackupState(PathBuf),
    RestoreState(PathBuf),
    BootLog,
    Configure,
    HostFirmwareInstall(PathBuf),
    HostFirmwareFetch {
        requirements: Option<PathBuf>,
        gpu: Option<String>,
        profile: Option<firmware::Profile>,
        offline: bool,
    },
    InspectSwap(swap::Inspect),
    CreateSwapImage(swap::CreateImage),
    PlanSwapImage(swap::PlanImage),
    ApplySwapImage(swap::ApplyImage),
    PrepareSwapVolumeImage(swap::PrepareVolumeImage),
}

/// One `stela setup` command: its action and the options it was given
/// (`setup::command` describes each).
#[derive(Default)]
struct Request {
    action: Action,
    usb: PathBuf,
    ipsw: Option<PathBuf>,
    fetch: bool,
    install_image: Option<InstallImage>,
    download_dir: Option<PathBuf>,
    efi: Option<PathBuf>,
    model: Option<String>,
    build: String,
    dry_run: bool,
    target: Option<String>,
    nor: Option<String>,
    disk_serial: Option<String>,
    erase_target: bool,
    initialize_nor: bool,
    install: Option<String>,
    ecid: Option<Ecid>,
    new_ecid: bool,
    profile: bool,
    no_profile: bool,
    verify_on_read: bool,
    install_volume: Option<PathBuf>,
    default_install: Option<String>,
    remove_install: Option<String>,
    chooser_timeout: Option<command::Wait>,
    translation_store_bytes: Option<u64>,
    pub(super) volume_swap_bytes: Option<u64>,
    host_firmware_cache: Option<PathBuf>,
    confirm_disk: Option<String>,
    /// The person running setup confirmed erasing the USB, as the wizard
    /// asks; otherwise a scripted erase at a terminal asks.
    confirmed: bool,
    wifi: Option<String>,
    wifi_passphrase_file: Option<PathBuf>,
    wifi_country: Option<String>,
    no_wifi: bool,
    usb4_approve: Vec<String>,
    no_usb4_approve: bool,
    /// `--wifi`'s passphrase, once read or typed.
    wifi_passphrase: Option<String>,
    ipv4: Option<stela_boot_core::options::StaticIpv4>,
    ipv4_gateway: Option<std::net::Ipv4Addr>,
    dns: Option<Vec<std::net::IpAddr>>,
    automatic_ipv4: bool,
    automatic_dns: bool,
    /// `boot-log`: the file the report goes to instead of the terminal.
    output: Option<PathBuf>,
}

#[derive(Args)]
pub struct InstallMedium {
    /// The macOS 27 restore image (IPSW); a download receipt beside it is
    /// checked.
    #[arg(
        long,
        required_unless_present = "fetch_build",
        conflicts_with = "fetch_build"
    )]
    ipsw: Option<PathBuf>,
    /// Download this catalog build straight into verified FAT32 pieces.
    #[arg(long, value_name = "BUILD")]
    fetch_build: Option<String>,
    /// Check the download identity, destination and capacity without writing.
    #[arg(long, requires = "fetch_build")]
    dry_run: bool,
    /// The Mac model whose identities the inventory lists.
    #[arg(long, default_value = PINNED_MODEL)]
    model: String,
    /// The mounted FAT32 volume to write `stela/install/<build>` to.
    #[arg(long)]
    volume: PathBuf,
}

impl InstallMedium {
    pub fn run(self) -> Result<()> {
        let host = host::System;
        let info = host.volume(&self.volume)?;
        ensure!(
            info.filesystem == FileSystem::Fat,
            "{} is {}, not FAT32",
            self.volume.display(),
            info.filesystem
        );
        if let Some(build) = self.fetch_build.as_deref() {
            return self.fetch(&host, build);
        }
        let ipsw = self
            .ipsw
            .as_deref()
            .context("supply --ipsw or --fetch-build")?;
        let (_, build) = restore::payload::identity(ipsw, &self.model)?;
        let bytes = fs::metadata(ipsw)?.len();
        let piece_bytes = stela_formats::payload::PIECE_BYTES;
        medium::check_space(
            &host,
            medium::SpaceOptions {
                layout: Layout::Legacy,
                volume: &self.volume,
                build: &build,
                ipsw_bytes: bytes,
                piece_bytes,
            },
        )?;
        let expected = medium::Expected::of(restore::receipt(ipsw)?.as_ref());
        let inventory = restore::payload::inventory(
            ipsw,
            &self.model,
            restore::payload::KeyOptions {
                fetch: true,
                output: None,
                input: None,
            },
        )?;
        let written = medium::write(medium::WriteOptions {
            layout: Layout::Legacy,
            volume: &self.volume,
            ipsw,
            inventory,
            expected: &expected,
            piece_bytes,
        })?;
        self.report(&build, &written);
        if expected.sha1.is_none() {
            println!("warning: no download receipt; IPSW checked against its BuildManifest only");
        }
        Ok(())
    }

    fn fetch(&self, host: &host::System, build: &str) -> Result<()> {
        ensure!(
            !build.is_empty() && build.bytes().all(|byte| byte.is_ascii_alphanumeric()),
            "unexpected build identifier"
        );
        let directory = medium::directory(&self.volume, build);
        ensure!(
            !directory.try_exists()?,
            "{} already exists",
            directory.display()
        );
        let image = restore::select(&restore::catalog()?, Some(build), Some(&self.model))?;
        let download = medium::download::Download::open(image)?;
        let piece_bytes = stela_formats::payload::PIECE_BYTES;
        medium::check_space(
            host,
            medium::SpaceOptions {
                layout: Layout::Legacy,
                volume: &self.volume,
                build,
                ipsw_bytes: download.bytes,
                piece_bytes,
            },
        )?;
        if self.dry_run {
            println!(
                "Dry run: {} catalog bytes for macOS {build}, FAT32 destination {}",
                download.bytes,
                directory.display()
            );
            return Ok(());
        }
        let written = download.write_fresh_in(&self.volume, &self.model, piece_bytes)?;
        self.report(build, &written);
        Ok(())
    }

    fn report(&self, build: &str, written: &medium::Written) {
        let directory = medium::directory(&self.volume, build);
        println!(
            "Install medium {}: {} pieces kept, {} written; payload.json SHA-256 {}",
            directory.display(),
            written.kept,
            written.wrote,
            written.manifest_sha256
        );
        println!(
            "Engine options: --install \\{}\\{} --install-source <the partition's GPT unique \
             GUID>",
            stela_formats::payload::install_directory(build).replace('/', "\\"),
            stela_formats::payload::MANIFEST_FILE
        );
    }
}

/// Writes a restore bundle for an IPSW, model and machine into a
/// directory, under the machine's personalized ticket.
type Prepare = dyn for<'a> Fn(prepare::RamdiskOptions<'a>) -> Result<()>;

/// An IPSW's restore inventory for a model.
type InventoryOf = dyn Fn(&Path, &str) -> Result<restore::payload::Inventory>;
/// The bytes of a build's boot members for a model on Apple's servers, and
/// of its whole image (`restore::recovery_size`).
type RecoverySizeOf<'a> = dyn Fn(&str, &str) -> Result<(u64, u64)> + 'a;
/// Download a build's boot members for a model into a directory
/// (`prepare::fetch_recovery`).
type FetchRecoveryOf<'a> = dyn Fn(&Path, &str, &str) -> Result<PathBuf> + 'a;

/// Inputs that the command line fixes and tests replace.
#[derive(Clone, Copy)]
struct Environment<'a> {
    host: &'a dyn Host,
    download: Option<&'a medium::download::Download>,
    /// `stela-dev boot prepare --restore`; tests use synthetic IPSWs, which
    /// have no kernel, and write a stand-in.
    prepare_restore: &'a Prepare,
    /// `stela-dev boot prepare --restore --update`, likewise.
    prepare_update: &'a Prepare,
    /// The install medium's inventory, with the public AEA keys fetched.
    inventory: &'a InventoryOf,
    /// The inventory of the IPSW's build on Apple's servers, which a remote
    /// medium names (`restore::remote_inventory`).
    remote_inventory: &'a InventoryOf,
    /// The install medium's piece size (`payload::PIECE_BYTES`).
    piece_bytes: u64,
    /// The mount interface this `stela` writes for.
    interface: Interface,
    /// The revision of the bundles this `stela` prepares
    /// (`stela_loader::boot_bundle::REVISION`).
    bundle_revision: u32,
    /// Where downloaded images may be, this user's cache first
    /// (`sources::image_caches`).
    image_caches: &'a [PathBuf],
    recovery_size: &'a RecoverySizeOf<'a>,
    fetch_recovery: &'a FetchRecoveryOf<'a>,
    /// The macOS builds setup prepares (`stela_formats::builds::SUPPORTED`);
    /// tests add their synthetic images' builds.
    builds: &'a [builds::Build],
    /// The Mac models setup presents ([`SUPPORTED_MODELS`]); tests add
    /// their synthetic images' model.
    models: &'a [(&'a str, &'a str)],
}

impl Environment<'_> {
    /// The generation of the bundles this `stela` prepares.
    fn generation(&self) -> bundle::Generation {
        bundle::Generation {
            schema: self.interface.bundle_schema,
            revision: self.bundle_revision,
        }
    }
}

/// The `stela.efi` at `path`, which must be a UEFI application with this
/// `stela`'s mount interface.
fn read_efi(environment: &Environment<'_>, path: &Path) -> Result<Vec<u8>> {
    let image = fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    ensure!(
        image.starts_with(b"MZ"),
        "{} is not a UEFI application",
        path.display()
    );
    if let Some(reason) =
        mount_interface::mismatch(environment.interface, mount_interface::find(&image))
    {
        bail!(
            "{} is not the stela.efi this stela installs: {reason}. Omit --efi to install the \
             stela.efi built into this stela.",
            path.display()
        );
    }
    Ok(image)
}

fn check_layout_efi(layout: Layout, image: &[u8]) -> Result<()> {
    ensure!(
        layout == Layout::Legacy || efi::supports_northwood_layout(image),
        "this EFI image cannot boot a Northwood setup; use a current Northwood EFI image"
    );
    Ok(())
}

/// Refuse to change a setup whose installed `stela.efi` reads another
/// interface than this `stela` writes; `update stela` replaces it first.
fn check_installed_efi(environment: &Environment<'_>, record: &Record) -> Result<()> {
    let tool = record.esp.join(LOADER_PATH);
    let image = fs::read(&tool).with_context(|| format!("cannot read {}", tool.display()))?;
    if let Some(reason) =
        mount_interface::mismatch(environment.interface, mount_interface::find(&image))
    {
        bail!(
            "the stela.efi on {} does not read what this stela writes ({reason}); install this \
             stela's stela.efi first with {}, or use the stela that installed it",
            record.esp.display(),
            fix(&record.esp, "update stela")
        );
    }
    check_layout_efi(record.layout, &image)
}

/// Refuse to prepare bundles for `record` when its active ones are of
/// another revision than this `stela` prepares: `update stela` prepares
/// them all at this one.
fn check_bundle_revision(environment: &Environment<'_>, record: &Record) -> Result<()> {
    match revision_mismatch(environment, record) {
        Some(reason) => bail!("{reason}"),
        None => Ok(()),
    }
}

/// Why `record`'s active bundles are not of the revision this `stela`
/// prepares, with the command that brings them to it.
fn revision_mismatch(environment: &Environment<'_>, record: &Record) -> Option<String> {
    let (theirs, ours) = (record.bundle_revision, environment.bundle_revision);
    if theirs == ours {
        return None;
    }
    Some(if theirs < ours {
        format!(
            "this setup's boot bundles are revision {theirs}, and this stela prepares revision \
             {ours}: prepare them again with {}",
            fix(&record.esp, "update stela")
        )
    } else {
        format!(
            "this setup's boot bundles are revision {theirs}, newer than the revision {ours} \
             this stela prepares: use the stela that prepared them"
        )
    })
}

/// The actions that find their USB when `--usb` is not given.
fn finds_usb(action: &Action) -> bool {
    matches!(
        action,
        Action::Status
            | Action::Repair
            | Action::UpdateStela
            | Action::UpdateOs
            | Action::HostFirmwareInstall(_)
            | Action::HostFirmwareFetch { .. }
    )
}

/// The mounted volume of each of this computer's disks that holds a Stela
/// setup, with the disk it is on: an EFI partition with the setup record,
/// or else a data volume with Stela's `stela/boot` directory. One per disk.
fn find_setups(host: &dyn Host) -> Result<Vec<(PathBuf, String)>> {
    let mut found = Vec::new();
    for disk in host.disks()? {
        let mounts = || disk.partitions.iter().filter_map(|p| p.mount.as_ref());
        let volume = mounts()
            .find(|mount| {
                Layout::PREFERRED
                    .into_iter()
                    .any(|layout| join_slashed(mount, &layout.esp_path(layout::RECORD)).is_file())
            })
            .or_else(|| {
                mounts().find(|mount| {
                    Layout::PREFERRED.into_iter().any(|layout| {
                        join_slashed(mount, &layout.data_path(layout::DATA_BOOT)).is_dir()
                    })
                })
            });
        if let Some(volume) = volume {
            found.push((
                volume.clone(),
                format!("{} ({})", disk.identifier, disk.name),
            ));
        }
    }
    Ok(found)
}

/// The USB of the one disk that holds a setup ([`find_setups`]), named on
/// the error output. None or several are an error that lists them; setup
/// never guesses between two.
fn choose_setup(host: &dyn Host) -> Result<PathBuf> {
    let found = find_setups(host)?;
    match found.as_slice() {
        [(volume, disk)] => {
            eprintln!("Using the Stela USB at {} on {disk}", volume.display());
            Ok(volume.clone())
        }
        [] => bail!(
            "no mounted volume on this computer holds a Stela setup: plug in the USB, mount its \
             STELA volume, or name it with --usb"
        ),
        several => {
            let list: String = several
                .iter()
                .map(|(volume, disk)| format!("  {} on {disk}\n", volume.display()))
                .collect();
            bail!(
                "{} disks hold a Stela setup:\n{list}Name the one to use with --usb, such as \
                 --usb {}",
                several.len(),
                several[0].0.display()
            )
        }
    }
}

impl Request {
    fn run(mut self) -> Result<()> {
        // Maintenance commands can find a single setup without --usb.
        if finds_usb(&self.action) && self.usb.as_os_str().is_empty() {
            self.usb = choose_setup(&host::System)?;
        }
        let boot_log = matches!(self.action, Action::BootLog);
        let swap_image = matches!(
            self.action,
            Action::InspectSwap(_)
                | Action::CreateSwapImage(_)
                | Action::PlanSwapImage(_)
                | Action::ApplySwapImage(_)
                | Action::PrepareSwapVolumeImage(_)
        );
        let interactive = !boot_log && !swap_image && self.usb.as_os_str().is_empty();
        // Windows opens a new elevated window for the wizard or raw logs.
        let elevate = !swap_image && (cfg!(unix) || interactive || boot_log);
        if elevate && !self.dry_run && elevation::relaunch()? {
            if boot_log {
                eprintln!(
                    "Reading the disks needs administrator rights: the boot log opens in a new \
                     window."
                );
            }
            return Ok(());
        }
        // Without a USB, `boot-log` reads every disk.
        if boot_log && self.usb.as_os_str().is_empty() {
            let logs = crate::boot_log::scan(&host::System)?;
            return crate::boot_log::write_logs(&logs, true, self.output.as_deref());
        }
        debug_assert_eq!(
            stela_loader::boot_bundle::SCHEMA_VERSION,
            mount_interface::CURRENT.bundle_schema
        );
        let image_caches = sources::image_caches();
        let environment = Environment {
            host: &host::System,
            image_caches: &image_caches,
            recovery_size: &|build, model| {
                let image = restore::select(&restore::catalog()?, Some(build), Some(model))?;
                restore::recovery_size(&image, model)
            },
            fetch_recovery: &|directory, build, model| {
                prepare::fetch_recovery(directory, build, model)
            },
            download: None,
            prepare_restore: &prepare_restore,
            prepare_update: &prepare_update,
            inventory: &|ipsw, model| {
                restore::payload::inventory(
                    ipsw,
                    model,
                    restore::payload::KeyOptions {
                        fetch: true,
                        output: None,
                        input: None,
                    },
                )
            },
            remote_inventory: &|ipsw, model| {
                let (_, build) = restore::payload::identity(ipsw, model)?;
                let image = restore::select(&restore::catalog()?, Some(&build), Some(model))?;
                restore::remote_inventory(&image, model, None)
            },
            piece_bytes: stela_formats::payload::PIECE_BYTES,
            interface: mount_interface::CURRENT,
            bundle_revision: stela_loader::boot_bundle::REVISION,
            builds: builds::SUPPORTED,
            models: SUPPORTED_MODELS,
        };
        // A stela without its stela.efi says so before the wizard asks
        // anything.
        if interactive && self.efi.is_none() {
            self.efi = Some(assets::resolve()?);
        }
        if let Some(path) = self.efi.as_ref().filter(|_| interactive) {
            read_efi(&environment, path)?;
        }
        if interactive && !wizard::configure(&mut self, environment.host)? {
            return Ok(());
        }
        let dry_run = self.dry_run;
        let profile = self.profile;
        let output = self.output.take();
        let (report, result) = split_report(self.dispatch(&environment));
        if let Err(error) = result {
            print!("{report}");
            return Err(error);
        }
        if interactive && !dry_run {
            let profile = if profile {
                "\nProfiling is on. Save your boot log with stela setup boot-log."
            } else {
                ""
            };
            cliclack::note(
                "Next steps",
                format!(
                    "Boot the target PC from this USB and choose Install macOS.\nThe installer \
                     checks whether it needs the internet before it starts.{profile}"
                ),
            )?;
            cliclack::outro("Your installer USB is ready")?;
        } else if let Some(path) = output {
            save_report(&path, &report)?;
        } else {
            print!("{report}");
        }
        Ok(())
    }

    fn model(&self) -> &str {
        self.model.as_deref().unwrap_or(PINNED_MODEL)
    }

    fn dispatch(mut self, environment: &Environment<'_>) -> Result<String> {
        let action = std::mem::take(&mut self.action);
        match action {
            Action::Create => return self.create(environment),
            Action::Status => return doctor::status(environment, &self.usb),
            Action::BootLog => return state::boot_log(environment, &self),
            Action::Uninstall => return uninstall(environment, &self.usb, self.dry_run),
            Action::InspectSwap(ref inspect) => return inspect.run(),
            Action::CreateSwapImage(ref create) => return create.run(),
            Action::PlanSwapImage(ref plan) => return plan.run(),
            Action::ApplySwapImage(ref apply) => return apply.run(),
            Action::PrepareSwapVolumeImage(ref prepare) => return prepare.run(),
            _ => {}
        }
        let mut record = read_record(environment, &self.usb)?;
        // A maintenance `--fetch` downloads the image of the model the setup
        // presents.
        self.model.get_or_insert_with(|| record.model.clone());
        // A USB that holds the whole macOS image keeps holding it: its
        // repairs and updates download whole images.
        if self.install_image.is_none() {
            self.install_image = record.install_image.or_else(|| {
                record
                    .media
                    .iter()
                    .any(|medium| !medium.remote)
                    .then_some(InstallImage::Full)
            });
        }
        // Every maintenance command works on the active build's files, so
        // what it downloads is that build; only `update os` names another.
        if !matches!(action, Action::UpdateOs) {
            self.build = record.build.clone();
        }
        match action {
            Action::Repair => doctor::repair(environment, &self),
            Action::ResetNvram => state::reset_nvram(environment, &record, &self),
            Action::BackupState(path) => state::backup(environment, &record, &self, &path),
            Action::RestoreState(path) => state::restore(environment, &record, &self, &path),
            Action::UpdateStela => {
                let image = match &self.efi {
                    Some(path) => path.clone(),
                    None => assets::resolve()?,
                };
                // The update finds the restore image itself when it needs
                // one, downloading only its boot files, never the whole
                // image; `--fetch` adds nothing to that.
                update::update_stela(
                    environment,
                    record,
                    update::UpdateOptions {
                        usb: &self.usb,
                        image_path: &image,
                        ipsw: self.ipsw.as_deref(),
                        download_dir: self.download_dir.as_deref(),
                        dry_run: self.dry_run,
                    },
                )
            }
            Action::UpdateOs => {
                check_installed_efi(environment, &record)?;
                record.install_image = self.install_image.or(record.install_image);
                let ipsw = self.image()?;
                update::update_os(environment, record, &ipsw, &self.usb, self.dry_run)
            }
            Action::NextBoot(next) => self.next_boot(environment, &record, next),
            Action::Configure => {
                self.read_wifi_passphrase()?;
                machines::configure(environment, &self)
            }
            Action::HostFirmwareInstall(source) => {
                let mount_image = host_firmware_mount(environment, &record)?;
                let package =
                    stela_loader::host_firmware::Package::read_for_mount(&source, &mount_image)?;
                let (images, bytes) = package.inventory();
                let (destination, _) =
                    firmware::publication_paths(&record.esp, record.layout, &mount_image)?;
                let mut report = format!(
                    "Host firmware: {images} verified resources, {bytes} bytes to {}\n",
                    destination.display()
                );
                if self.dry_run {
                    report.push_str("Dry run: nothing was changed.\n");
                } else {
                    package.publish(&destination)?;
                    report.push_str(
                        "Published host firmware manifest. Device initialization is checked at boot.\n",
                    );
                }
                Ok(report)
            }
            Action::HostFirmwareFetch {
                requirements,
                gpu,
                profile,
                offline,
            } => {
                let mount_image = host_firmware_mount(environment, &record)?;
                let cache = self
                    .host_firmware_cache
                    .clone()
                    .unwrap_or_else(|| cache_dir().join("firmware"));
                let bundled = stela_loader::host_firmware::reads_bundle(&mount_image);
                ensure!(
                    bundled || requirements.is_some() || gpu.is_some() || profile.is_some(),
                    "this USB's stela.efi reads no firmware bundle: name a requirements file, a \
                     GPU or a profile, or run {} first",
                    fix(&record.esp, "update stela")
                );
                if bundled && requirements.is_none() && profile.is_none() {
                    let mut report = format!(
                        "Host firmware: the USB's bundle holds {}.\n",
                        firmware::bundle_summary()
                    );
                    if self.dry_run {
                        report.push_str("Dry run: nothing was changed.\n");
                        return Ok(report);
                    }
                    for note in firmware::provision_bundle_from(
                        environment.host,
                        &record.volume,
                        &record.esp,
                        &mount_image,
                        &cache,
                        offline,
                    )? {
                        writeln!(report, "Note: {note}")?;
                    }
                    report.push_str("Done.\n");
                    return Ok(report);
                }
                let (destination, installed) =
                    firmware::publication_paths(&record.esp, record.layout, &mount_image)?;
                firmware::provision_from(
                    firmware::ProvisionOptions {
                        requirements: requirements.as_deref(),
                        gpu: gpu.as_deref(),
                        wifi: None,
                        profile,
                        offline,
                        destination: &destination,
                        cache: &cache,
                        dry_run: self.dry_run,
                        mount_image: &mount_image,
                    },
                    &installed,
                )
            }
            Action::Create
            | Action::Status
            | Action::BootLog
            | Action::Uninstall
            | Action::InspectSwap(_)
            | Action::CreateSwapImage(_)
            | Action::PlanSwapImage(_)
            | Action::ApplySwapImage(_)
            | Action::PrepareSwapVolumeImage(_) => {
                unreachable!("these run without the setup record")
            }
        }
    }

    /// Create a setup, or add an install (`--target`) to an existing one.
    fn create(mut self, environment: &Environment<'_>) -> Result<String> {
        if let Some(model) = &self.model {
            check_model(environment.models, model)?;
        }
        // A first setup that was interrupted is resumed by `setup`, not
        // added to.
        let configured = locate_esp(environment.host, &self.usb).is_ok_and(|esp| {
            record::layout_at(&esp).is_ok_and(|layout| layout.is_some())
                && record::interrupted(environment.host, &esp, &self.usb).is_none()
        });
        if configured && self.target.is_some() {
            ensure!(
                self.volume_swap_bytes.is_none(),
                "--volume-swap-bytes goes with a new setup"
            );
            ensure!(
                self.wifi.is_none(),
                "--wifi goes with a new setup; {} changes an existing one's",
                fix(&self.usb, "configure --wifi SSID")
            );
            // Every install restores the active build, which `--fetch`
            // downloads.
            if let Ok(record) = read_record(environment, &self.usb) {
                self.build = record.build;
            }
            return machines::add(environment, &self);
        }
        setup(environment, self)
    }

    /// Read `--wifi`'s passphrase: the wizard's, `--wifi-passphrase-file`'s,
    /// or one typed at a prompt, which a dry run skips.
    fn read_wifi_passphrase(&mut self) -> Result<()> {
        let Some(ssid) = &self.wifi else {
            return Ok(());
        };
        if self.wifi_passphrase.is_some() {
            return Ok(());
        }
        self.wifi_passphrase = match &self.wifi_passphrase_file {
            Some(file) => Some(wifi::read_passphrase(file)?),
            None if self.dry_run => None,
            None => Some(wifi::ask_passphrase(ssid)?),
        };
        Ok(())
    }

    /// `nvram next-boot`.
    fn next_boot(
        &self,
        environment: &Environment<'_>,
        record: &Record,
        next: NextBoot,
    ) -> Result<String> {
        check_installed_efi(environment, record)?;
        let install = record.install(self.install.as_deref())?;
        let disk = &install.disk;
        let changes = installed::next_boot(environment.host, disk, next, self.dry_run)?;
        let mut text = format!(
            "NVRAM of {} (install \"{}\"'s machine state):\n",
            match disk.machine_state_offset {
                Some(offset) => format!("{} from byte {offset:#x}", disk.nor),
                None => disk.nor.clone(),
            },
            install.name
        );
        if changes.is_empty() {
            text.push_str("  nothing to change\n");
        }
        for change in changes {
            writeln!(text, "  {change}")?;
        }
        text.push_str(if self.dry_run {
            "Dry run: nothing was changed.\n"
        } else {
            "Done.\n"
        });
        Ok(text)
    }

    /// The restore image: `--ipsw`, or the build setup downloads.
    fn image(&self) -> Result<PathBuf> {
        match (&self.ipsw, self.fetch) {
            (Some(path), false) => Ok(path.clone()),
            (None, true) if self.dry_run => {
                bail!(
                    "a dry run needs the image first: pass --ipsw, or run once without \
                     --dry-run to download it"
                )
            }
            (None, true) => fetch(self),
            _ => bail!("pass --ipsw IPSW, or --fetch to download macOS 27 from Apple"),
        }
    }

    /// The restore image when the command names one.
    fn optional_image(&self) -> Result<Option<PathBuf>> {
        (self.ipsw.is_some() || self.fetch)
            .then(|| self.image())
            .transpose()
    }
}

/// The installed `stela.efi`, which must read host firmware resources.
fn host_firmware_mount(environment: &Environment<'_>, record: &Record) -> Result<Vec<u8>> {
    check_installed_efi(environment, record)?;
    let mount_image = fs::read(record.esp.join(LOADER_PATH))?;
    ensure!(
        stela_loader::host_firmware::supports_mount(&mount_image),
        "the installed stela.efi cannot load host firmware; run {} first",
        fix(&record.esp, "update stela")
    );
    Ok(mount_image)
}

/// The ESP to change: the EFI partition of `usb`'s disk, which must be
/// mounted, sit on a removable disk and boot no loader but Stela's.
fn locate_esp(host: &dyn Host, usb: &Path) -> Result<PathBuf> {
    locate_esp_with_record(host, usb, |esp| Ok(record::mutation_layout(esp)?.is_some()))
}

fn locate_esp_with_record(
    host: &dyn Host,
    usb: &Path,
    has_record: impl FnOnce(&Path) -> Result<bool>,
) -> Result<PathBuf> {
    ensure!(usb.is_dir(), "{} is not a mounted volume", usb.display());
    let volume = host.volume(usb)?;
    let disks = host.disks()?;
    let disk = volume
        .disk
        .as_ref()
        .and_then(|id| disks.iter().find(|d| &d.identifier == id));
    let partition = disk.and_then(|d| d.partitions.iter().find(|p| p.content == "EFI"));
    let esp = match partition {
        Some(partition) => match &partition.mount {
            Some(mount) => mount.clone(),
            // Hosts leave a USB's EFI partition unmounted.
            None => host.mount(&partition.identifier).with_context(|| {
                format!(
                    "{} is on the same disk as EFI partition {}, which could not be mounted \
                     (setup needs {})",
                    usb.display(),
                    partition.identifier,
                    host::ADMIN
                )
            })?,
        },
        None => bail!(
            "{} is on a disk without an EFI partition. Setup erases a USB into one when --usb \
             names its whole disk (such as {}), which deletes everything on it.",
            usb.display(),
            host::DISK_EXAMPLE
        ),
    };
    // A USB setup prepared owns its loader, damaged or old as it may be.
    if !has_record(&esp)? {
        efi::check_loader(&esp)?;
    }
    if let Some(disk) = host
        .volume(&esp)?
        .disk
        .and_then(|id| disks.into_iter().find(|d| d.identifier == id))
    {
        ensure!(
            disk.erasable(),
            "{} is on {} ({}), an internal disk{}. Setup changes only a removable USB: \
             connect one and name it with --usb.",
            esp.display(),
            disk.identifier,
            disk.name,
            if disk.system {
                " that runs this computer"
            } else {
                ""
            }
        );
    }
    Ok(esp)
}

#[cfg(test)]
mod tests;
