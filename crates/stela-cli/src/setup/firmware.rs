//! Vendor firmware provisioning for a native-driver requirement inventory.
//! Network/cache policy stays in the CLI; the loader validates pins/packages.
use anyhow::{Context, Result, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use reqwest::{blocking::Client, header, redirect::Policy};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};
use stela_drivers::firmware::{Requirement, Target};
use stela_loader::host_firmware::{
    self, MAX_MANIFEST_BYTES,
    recipes::{self, Artifact, Plan},
    requirements::{self, Inventory, Request},
};
use stela_loader::mount_interface::layout::{BOOT_LOG, FIRMWARE, Layout};

pub(super) fn directory(esp: &Path, layout: Layout) -> PathBuf {
    esp.join(layout.esp_path(FIRMWARE))
}

pub(super) fn read_inventory(path: &Path) -> Result<Inventory> {
    let file = File::open(path)?;
    ensure!(
        file.metadata()?.is_file() && file.metadata()?.len() <= MAX_MANIFEST_BYTES as u64,
        "firmware requirement inventory exceeds its file budget"
    );
    let mut bytes = Vec::new();
    file.take(MAX_MANIFEST_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    Ok(Inventory::parse(&bytes)?)
}

fn read_cached(path: &Path, artifact: &Artifact) -> Result<Vec<u8>> {
    ensure!(
        !fs::symlink_metadata(path)?.is_symlink(),
        "firmware cache object is a symlink"
    );
    let mut file = File::open(path)?;
    ensure!(
        file.metadata()?.is_file() && file.metadata()?.len() == artifact.length as u64,
        "firmware cache object length differs from its pin"
    );
    let mut bytes = vec![0; artifact.length];
    file.read_exact(&mut bytes)?;
    ensure!(
        file.read(&mut [0])? == 0,
        "firmware cache object grew during read"
    );
    artifact.verify(&bytes)?;
    Ok(bytes)
}

fn download(client: &Client, artifact: &Artifact) -> Result<Vec<u8>> {
    let primary = download_member(client, artifact, &artifact.url(), false);
    if artifact.source != recipes::WIRELESS_REGDB {
        return primary;
    }
    primary.or_else(|error| {
        let mirror = format!(
            "https://kernel.googlesource.com/pub/scm/linux/kernel/git/wens/wireless-regdb/+/{}/{}?format=TEXT",
            artifact.revision, artifact.member,
        );
        download_member(client, artifact, &mirror, true)
            .with_context(|| format!("wireless-regdb primary source failed: {error:#}"))
    })
}

fn download_member(
    client: &Client,
    artifact: &Artifact,
    url: &str,
    encoded: bool,
) -> Result<Vec<u8>> {
    let wire_length = if encoded {
        artifact.length.div_ceil(3) * 4
    } else {
        artifact.length
    };
    // A minute, and the file at 32 kB/s: a GPU's 64 MB firmware may take
    // many minutes on a slow network.
    let timeout = Duration::from_secs(60 + wire_length as u64 / 32_000);
    let mut response = crate::net::send(&format!("downloading {}", artifact.member), || {
        client
            .get(url)
            .header(header::ACCEPT_ENCODING, "identity")
            .timeout(timeout)
    })?
    .error_for_status()?;
    ensure!(
        response.status() == reqwest::StatusCode::OK,
        "firmware source redirected or returned an unexpected status"
    );
    ensure!(
        response
            .content_length()
            .is_none_or(|n| n == wire_length as u64),
        "firmware source length differs from its pin"
    );
    let mut bytes = Vec::with_capacity(wire_length);
    response
        .by_ref()
        .take(wire_length as u64 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() == wire_length,
        "firmware source length differs from its pin"
    );
    if encoded {
        bytes = STANDARD
            .decode(bytes)
            .context("wireless-regdb mirror returned invalid base64")?;
    }
    artifact.verify(&bytes)?;
    Ok(bytes)
}

fn cached(client: Option<&Client>, directory: &Path, artifact: &Artifact) -> Result<Vec<u8>> {
    let path = directory.join(artifact.sha256);
    match fs::symlink_metadata(&path) {
        Ok(_) => return read_cached(&path, artifact),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let client = client.with_context(|| {
        format!(
            "offline firmware cache is missing {} ({})",
            artifact.member, artifact.sha256
        )
    })?;
    let bytes = download(client, artifact)?;
    fs::create_dir_all(directory)?;
    ensure!(
        !fs::symlink_metadata(directory)?.is_symlink(),
        "firmware cache directory is a symlink"
    );
    let mut temporary = tempfile::NamedTempFile::new_in(directory)?;
    temporary.write_all(&bytes)?;
    temporary.as_file().sync_all()?;
    if let Err(error) = temporary.persist_noclobber(&path)
        && error.error.kind() != std::io::ErrorKind::AlreadyExists
    {
        return Err(error).with_context(|| {
            format!("cannot publish immutable firmware cache {}", path.display())
        });
    }
    host_firmware::sync_directory(directory)?;
    read_cached(&path, artifact)
}

/// The chips of this computer's NVIDIA GPUs, whose firmware setup adds to
/// the USB, and a note for each NVIDIA GPU Stela cannot run.
pub(super) fn host_chips(host: &dyn super::host::Host) -> Result<(Vec<&'static str>, Vec<String>)> {
    let mut chips = Vec::new();
    let mut notes = Vec::new();
    for device in host.nvidia_gpus()? {
        match recipes::nvidia_chip(device) {
            Some(chip) if !chips.contains(&chip) => chips.push(chip),
            Some(_) => {}
            None => notes.push(format!(
                "this computer's NVIDIA GPU (device {device:04x}) is not one Stela supports: \
                 Stela needs {}",
                stela_boot_core::gpu_status::SUPPORTED
            )),
        }
    }
    Ok((chips, notes))
}

/// Why a USB made on this computer has no GPU firmware, and how to add it:
/// `fix` is the command, with CHIP for the chip the PC's boot screen names.
pub(super) fn missing(fix: &str) -> String {
    format!(
        "this computer has no NVIDIA GPU Stela supports, so the USB gets no GPU firmware, and \
         macOS cannot start without it. Run setup on the PC itself, or add the firmware later: \
         boot the PC from the USB, which names its GPU chip (such as ga104) when its firmware \
         is missing, then run {fix} on this computer"
    )
}

/// Add what a Wi-Fi network needs to the firmware on the USB's EFI partition
/// `esp`, whose boot program is `mount_image`: the regulatory database, and
/// the files of this computer's Wi-Fi adapters whose PCI IDs name them.
pub(super) fn provision_wifi_in(
    host: &dyn super::host::Host,
    esp: &Path,
    layout: Layout,
    mount_image: &[u8],
) -> Result<()> {
    // A boot program that reads no firmware from the USB has no Wi-Fi driver
    // either.
    if !host_firmware::supports_mount(mount_image) {
        return Ok(());
    }
    let (destination, installed) = publication_paths(esp, layout, mount_image)?;
    provision_from(
        ProvisionOptions {
            requirements: None,
            gpu: None,
            wifi: Some(&host.network_adapters()?),
            profile: None,
            offline: false,
            destination: &destination,
            cache: &super::cache_dir().join("firmware"),
            dry_run: false,
            mount_image,
        },
        &installed,
    )
    .context("the Wi-Fi network is set, but the USB did not get the Wi-Fi firmware")?;
    Ok(())
}

/// A note for a setup with a Wi-Fi network on a computer none of whose
/// adapters' files setup can add.
pub(super) fn wifi_note(host: &dyn super::host::Host) -> Result<Option<String>> {
    let mut inventory = Inventory::default();
    for (vendor, device) in host.network_adapters()? {
        if recipes::require_wifi(&mut inventory, vendor, device)? {
            return Ok(None);
        }
    }
    Ok(Some(NO_WIFI_FILES.into()))
}

/// A note for each of this computer's Bluetooth controllers. Setup adds no
/// Bluetooth firmware itself: every vendor setup chooses its files by
/// registers only a boot of the PC reads, whose log then names them for
/// `repair`, the command that adds them.
pub(super) fn bluetooth_notes(host: &dyn super::host::Host, repair: &str) -> Result<Vec<String>> {
    Ok(describe_bluetooth(&host.bluetooth_controllers()?, repair))
}

/// What [`bluetooth_notes`] says for `controllers`.
fn describe_bluetooth(controllers: &[super::host::UsbBluetooth], repair: &str) -> Vec<String> {
    use stela_drivers::usb::bluetooth::Setup;
    let mut notes = Vec::new();
    for controller in controllers {
        let ids = format!("USB {:04x}:{:04x}", controller.vendor, controller.product);
        let setup = Setup::of(
            controller.vendor,
            controller.product,
            controller.vendor_interface,
        );
        notes.push(match setup {
            // Recipes pin images for these setups' drivers only
            // (`recipes::MEDIATEK_BLUETOOTH`, `recipes::INTEL_RSA`).
            Some(setup @ (Setup::MediaTek | Setup::Intel)) => format!(
                "this computer's Bluetooth controller ({}, {ids}) loads firmware that its \
                 registers choose, so the USB has none yet: boot the PC from the USB once, \
                 then run {repair} on a computer with internet, which adds the file the boot \
                 asked for when Stela has a pinned source for it",
                setup.name()
            ),
            Some(setup) => format!(
                "this computer's Bluetooth controller ({}, {ids}) needs firmware Stela has no \
                 pinned source for, so it stays off",
                setup.name()
            ),
            // A vendor-class interface no setup takes is no HCI.
            None if controller.vendor_interface => continue,
            None => format!(
                "this computer's Bluetooth controller ({ids}) is not one Stela has a setup for, \
                 so it stays off"
            ),
        });
    }
    notes
}

/// What [`wifi_note`] says.
const NO_WIFI_FILES: &str = "this computer has no Wi-Fi adapter whose firmware setup \
    can add (Intel AX200 to BE200, Realtek RTL8822C, MediaTek MT7921, MT7922 or MT7925), \
    so the USB has none: the PC needs wired \
    Ethernet, or run setup on the PC itself";

#[derive(Clone, Copy, clap::ValueEnum)]
pub(super) enum Profile {
    #[value(name = "lenovo-ideapad-1-15igl7", alias = "15igl7")]
    Lenovo15Igl7,
}

/// Firmware inputs, cache and destination for one provisioning request.
pub(super) struct ProvisionOptions<'a> {
    pub(super) requirements: Option<&'a Path>,
    /// A chip to add nouveau's files for, to what the destination holds.
    pub(super) gpu: Option<&'a str>,
    /// The setup computer's network adapters, when a Wi-Fi network is set:
    /// add the regulatory database and their files to what the destination
    /// holds.
    pub(super) wifi: Option<&'a [(u16, u16)]>,
    pub(super) profile: Option<Profile>,
    pub(super) offline: bool,
    pub(super) destination: &'a Path,
    pub(super) cache: &'a Path,
    pub(super) dry_run: bool,
    pub(super) mount_image: &'a [u8],
}

/// The firmware manifest in `destination`, when it has one.
fn installed_manifest(destination: &Path) -> Result<Option<Vec<u8>>> {
    let previous = destination.join("manifest.json");
    if !previous.try_exists()? {
        return Ok(None);
    }
    let mut bytes = Vec::new();
    File::open(&previous)?
        .take(MAX_MANIFEST_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    Ok(Some(bytes))
}

/// Whether `image` is the exact binding `request` asks for.
fn binds(image: &host_firmware::Image<'_>, request: &Request) -> bool {
    request.name == image.name
        && request.target.driver == image.target.driver
        && request.target.hardware == image.target.hardware
        && request.target.revision == image.target.revision
        && request.target.board.as_deref() == image.target.board
}

/// What the newest firmware inventory in a USB's boot log asks for that the
/// USB's firmware lacks: files with a pinned source, which `repair` and
/// `update stela` add, and files without one, which they name.
#[derive(Debug)]
pub(super) struct Requested {
    layout: Layout,
    pub(super) pinned: Vec<Request>,
    pub(super) unpinned: Vec<Request>,
    /// Why the newest inventory line holds none.
    pub(super) unreadable: Option<String>,
}

impl Default for Requested {
    fn default() -> Self {
        Self {
            layout: Layout::Legacy,
            pinned: Vec::new(),
            unpinned: Vec::new(),
            unreadable: None,
        }
    }
}

/// A missing file as the reports name it: its name and the driver's target.
pub(super) fn describe(request: &Request) -> String {
    let target = &request.target;
    let board = target
        .board
        .as_deref()
        .map(|board| format!(" board {board}"))
        .unwrap_or_default();
    format!(
        "{} ({} {} {}{board})",
        request.name, target.driver, target.hardware, target.revision
    )
}

impl Requested {
    pub(super) fn is_empty(&self) -> bool {
        self.pinned.is_empty() && self.unpinned.is_empty() && self.unreadable.is_none()
    }

    /// The lines `repair` and `update stela` report.
    pub(super) fn lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if let Some(why) = &self.unreadable {
            lines.push(format!(
                "the boot log's newest firmware request is unreadable: {why}"
            ));
        }
        for request in &self.pinned {
            lines.push(format!(
                "add {}, which the last boot asked for",
                describe(request)
            ));
        }
        for request in &self.unpinned {
            lines.push(format!(
                "the last boot asked for {}, which Stela has no pinned source for; the device \
                 stays off",
                describe(request)
            ));
        }
        lines
    }

    /// Download the pinned files and publish them with what the USB's
    /// firmware on `esp` already holds.
    pub(super) fn provision(&self, esp: &Path, mount_image: &[u8]) -> Result<()> {
        self.plan(
            esp,
            mount_image,
            &super::cache_dir().join("firmware"),
            false,
        )
        .context("the USB did not get the firmware the last boot asked for")?;
        Ok(())
    }

    /// [`Self::provision`] through `cache`; a dry run only plans.
    fn plan(
        &self,
        esp: &Path,
        mount_image: &[u8],
        cache: &Path,
        dry_run: bool,
    ) -> Result<Option<String>> {
        if self.pinned.is_empty() {
            return Ok(None);
        }
        let (destination, installed) = publication_paths(esp, self.layout, mount_image)?;
        let mut inventory = Inventory::default();
        for request in &self.pinned {
            inventory.observe(request.view())?;
        }
        keep_installed(&mut inventory, &installed, true)?;
        publish_plan(&inventory, &destination, cache, dry_run, false, mount_image).map(Some)
    }
}

/// What the newest inventory in the boot log on `esp` asks for that the
/// USB's firmware lacks, for a boot program `mount_image`. A USB without a
/// log or without such a line, and a boot program that reads no firmware
/// from the USB, need nothing.
#[cfg(test)]
fn requested(esp: &Path, mount_image: &[u8]) -> Result<Requested> {
    requested_in(esp, Layout::Legacy, mount_image)
}

pub(super) fn requested_in(esp: &Path, layout: Layout, mount_image: &[u8]) -> Result<Requested> {
    let empty = || Requested {
        layout,
        ..Requested::default()
    };
    if !host_firmware::supports_mount(mount_image) {
        return Ok(empty());
    }
    let Some(log) = fs::read(esp.join(layout.esp_path(BOOT_LOG)))
        .ok()
        .and_then(|region| stela_runtime::pc::boot_log::read(&region).ok())
    else {
        return Ok(empty());
    };
    let inventory = match requirements::newest(&log.stream) {
        None => return Ok(empty()),
        Some(Ok(inventory)) => inventory,
        Some(Err(error)) => {
            return Ok(Requested {
                unreadable: Some(error.0),
                ..empty()
            });
        }
    };
    let manifest = installed_manifest(&catalog_directory(esp, layout, mount_image)?)?;
    let installed = manifest
        .as_deref()
        .map(host_firmware::Manifest::parse)
        .transpose()?;
    let mut requested = empty();
    for request in inventory.requests() {
        if requirements::read_by_boot(&request.view().target)
            || installed
                .as_ref()
                .is_some_and(|manifest| manifest.images.iter().any(|image| binds(image, request)))
        {
            continue;
        }
        if recipes::recipe(request).is_some() {
            requested.pinned.push(request.clone());
        } else {
            requested.unpinned.push(request.clone());
        }
    }
    Ok(requested)
}

/// The catalog visible to this EFI, which may be the legacy fallback during
/// a layout transition. A present modern manifest takes precedence.
pub(super) fn catalog_directory(esp: &Path, layout: Layout, image: &[u8]) -> Result<PathBuf> {
    if layout == Layout::Northwood || Layout::supports_northwood(image) {
        let modern = directory(esp, Layout::Northwood);
        if installed_manifest(&modern)?.is_some() {
            return Ok(modern);
        }
    }
    Ok(directory(esp, Layout::Legacy))
}

/// The catalog destination and the active catalog whose bindings it must keep.
pub(super) fn publication_paths(
    esp: &Path,
    layout: Layout,
    image: &[u8],
) -> Result<(PathBuf, PathBuf)> {
    let destination = directory(esp, layout);
    let installed = catalog_directory(esp, layout, image)?;
    ensure!(
        layout == Layout::Northwood || destination == installed,
        "the current EFI reads host firmware from {}; writing {} would leave that catalog in use",
        installed.display(),
        destination.display()
    );
    Ok((destination, installed))
}

/// Require what the firmware in `destination` holds, when `add` says the
/// inventory adds to it; refuse an inventory that would remove a binding.
fn keep_installed(inventory: &mut Inventory, destination: &Path, add: bool) -> Result<()> {
    if let Some(bytes) = installed_manifest(destination)? {
        let previous = host_firmware::Manifest::parse(&bytes)?;
        if add {
            for image in &previous.images {
                inventory.observe(Requirement {
                    name: image.name,
                    target: Target {
                        driver: image.target.driver,
                        hardware: image.target.hardware,
                        revision: image.target.revision,
                        board: image.target.board,
                    },
                    minimum_bytes: 1,
                })?;
            }
        }
        for image in &previous.images {
            ensure!(
                inventory
                    .requests()
                    .iter()
                    .any(|request| binds(image, request)),
                "observed requirements do not cover installed firmware binding {}; refusing to remove it",
                image.name
            );
        }
    }
    Ok(())
}

#[cfg(test)]
fn provision(options: ProvisionOptions<'_>) -> Result<String> {
    let installed = options.destination;
    provision_from(options, installed)
}

pub(super) fn provision_from(options: ProvisionOptions<'_>, installed: &Path) -> Result<String> {
    let ProvisionOptions {
        requirements,
        gpu,
        wifi,
        profile,
        offline,
        destination,
        cache,
        dry_run,
        mount_image,
    } = options;
    let mut inventory = match requirements {
        Some(path) => read_inventory(path)?,
        None => Inventory::default(),
    };
    keep_installed(
        &mut inventory,
        installed,
        gpu.is_some() || wifi.is_some() || profile.is_some(),
    )?;
    if let Some(chip) = gpu {
        recipes::require_gsp(&mut inventory, chip)?;
    }
    if let Some(adapters) = wifi {
        recipes::require_regulatory(&mut inventory)?;
        for &(vendor, device) in adapters {
            recipes::require_wifi(&mut inventory, vendor, device)?;
        }
    }
    if let Some(Profile::Lenovo15Igl7) = profile {
        recipes::require_15igl7_wifi(&mut inventory)?;
    }
    let mut report = publish_plan(
        &inventory,
        destination,
        cache,
        dry_run,
        offline,
        mount_image,
    )?;
    if profile.is_some() {
        report.push_str("Profile lenovo-ideapad-1-15igl7 selects Wi-Fi firmware candidates. Live device initialization is checked at boot.\n");
    }
    Ok(report)
}

/// Plan `inventory`'s pinned files and, unless `dry_run`, download them into
/// `cache` and publish them to `destination`.
fn publish_plan(
    inventory: &Inventory,
    destination: &Path,
    cache: &Path,
    dry_run: bool,
    offline: bool,
    mount_image: &[u8],
) -> Result<String> {
    let plan = Plan::select(inventory)?;
    host_firmware::MountLimits::from_image(mount_image)
        .context("stela.efi has no supported host firmware mount capability")?
        .check_lengths(
            plan.selections()
                .iter()
                .map(|selection| selection.recipe.image.length),
        )?;
    let mut report = format!(
        "Host firmware: {} exact firmware bindings, {} pinned image/license artifacts\n",
        plan.selections().len(),
        plan.artifacts().len()
    );
    for selection in plan.selections() {
        use std::fmt::Write as _;
        writeln!(
            report,
            "  {} {:?}: {} bytes SHA-256 {}",
            selection.request.name,
            selection.request.target,
            selection.recipe.image.length,
            selection.recipe.image.sha256
        )?;
    }
    if dry_run {
        report.push_str("Dry run: no downloads or writes.\n");
        return Ok(report);
    }
    let client = (!offline)
        .then(|| {
            Client::builder()
                .user_agent(concat!("stela/", env!("CARGO_PKG_VERSION")))
                .connect_timeout(Duration::from_secs(30))
                .timeout(Duration::from_secs(120))
                .redirect(Policy::none())
                .build()
        })
        .transpose()?;
    let package = plan.package(|artifact| {
        cached(client.as_ref(), cache, artifact)
            .map_err(|e| host_firmware::Error(format!("firmware provisioning: {e:#}")))
    })?;
    package.check_mount(mount_image)?;
    package.publish(destination)?;
    report.push_str("Published pinned firmware and accompanying redistribution notices.\n");
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn each_bluetooth_controller_is_named_with_what_a_boot_and_repair_do() {
        use super::super::host::UsbBluetooth;
        let controller = |vendor, product, vendor_interface| UsbBluetooth {
            vendor,
            product,
            vendor_interface,
        };
        let notes = describe_bluetooth(
            &[
                controller(0x13d3, 0x3602, false),
                controller(0x0bda, 0x8771, false),
                controller(0x0a12, 0x0001, false),
                // A vendor-class interface no setup takes is no controller.
                controller(0x0bda, 0x8153, true),
            ],
            "`stela setup repair --usb E:`",
        );
        assert_eq!(notes.len(), 3, "{notes:?}");
        assert!(
            notes[0].starts_with("this computer's Bluetooth controller (MediaTek, USB 13d3:3602)")
                && notes[0].contains(
                    "boot the PC from the USB once, then run `stela setup repair --usb E:`"
                ),
            "{}",
            notes[0]
        );
        assert!(
            notes[1].contains("(Realtek, USB 0bda:8771) needs firmware Stela has no pinned source"),
            "{}",
            notes[1]
        );
        assert!(
            notes[2].contains("(USB 0a12:0001) is not one Stela has a setup for"),
            "{}",
            notes[2]
        );
    }

    #[test]
    fn cache_reuse_rechecks_exact_length_hash_and_symlink_identity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("object");
        let artifact = Artifact {
            source: "https://vendor.invalid",
            revision: "synthetic",
            member: "one.bin",
            length: 3,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        };
        fs::write(&path, b"abc").unwrap();
        assert_eq!(read_cached(&path, &artifact).unwrap(), b"abc");
        for bytes in [b"abd".as_slice(), b"ab", b"abcd"] {
            fs::write(&path, bytes).unwrap();
            assert!(read_cached(&path, &artifact).is_err());
        }
        #[cfg(unix)]
        {
            let alias = dir.path().join("alias");
            std::os::unix::fs::symlink(&path, &alias).unwrap();
            assert!(read_cached(&alias, &artifact).is_err());
        }
    }

    #[test]
    fn offline_cache_miss_names_the_pin_and_reuse_verifies_it() {
        let dir = tempfile::tempdir().unwrap();
        let artifact = Artifact {
            source: "https://vendor.invalid",
            revision: "synthetic",
            member: "one.bin",
            length: 3,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        };
        let missing = cached(None, dir.path(), &artifact).unwrap_err().to_string();
        assert!(missing.contains(artifact.member) && missing.contains(artifact.sha256));
        let path = dir.path().join(artifact.sha256);
        assert!(!path.exists());
        fs::write(&path, b"abc").unwrap();
        assert_eq!(cached(None, dir.path(), &artifact).unwrap(), b"abc");
        fs::write(&path, b"xyz").unwrap();
        assert!(cached(None, dir.path(), &artifact).is_err());
    }

    #[test]
    fn lenovo_profile_dry_run_plans_exact_cnvi_and_regulatory_resources() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("firmware");
        let cache = dir.path().join("cache");
        let provision = |dry_run| {
            provision(ProvisionOptions {
                requirements: None,
                gpu: None,
                wifi: None,
                profile: Some(Profile::Lenovo15Igl7),
                offline: true,
                destination: &destination,
                cache: &cache,
                dry_run,
                mount_image: &host_firmware::MOUNT_CAPABILITY,
            })
        };
        let report = provision(true).unwrap();
        assert!(report.contains("iwlwifi-9000-pu-b0-jf-b0-46.ucode"));
        assert!(report.contains("pci:8086:31dc"));
        assert!(report.contains("regulatory.db.p7s"));
        let error = provision(false).unwrap_err().to_string();
        assert!(
            error.contains("offline firmware cache is missing"),
            "{error}"
        );
        assert!(!destination.exists());
        assert!(!cache.exists());
    }

    #[test]
    #[ignore = "network oracle: pinned vendor firmware and license downloads"]
    fn lenovo_profile_provisions_the_same_package_online_and_offline() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache");
        let online = dir.path().join("online");
        let offline = dir.path().join("offline");
        for (destination, cache_only) in [(&online, false), (&offline, true)] {
            provision(ProvisionOptions {
                requirements: None,
                gpu: None,
                wifi: None,
                profile: Some(Profile::Lenovo15Igl7),
                offline: cache_only,
                destination,
                cache: &cache,
                dry_run: false,
                mount_image: &host_firmware::MOUNT_CAPABILITY,
            })
            .unwrap();
            let package = host_firmware::Package::read(destination).unwrap();
            assert_eq!(package.inventory(), (3, 1527001));
        }
        assert_eq!(
            fs::read(online.join("manifest.json")).unwrap(),
            fs::read(offline.join("manifest.json")).unwrap()
        );
    }
    #[test]
    fn dry_run_plans_pins_without_cache_or_destination_writes() {
        let dir = tempfile::tempdir().unwrap();
        let inventory = dir.path().join("requests.json");
        fs::write(&inventory, br#"{"schema":1,"complete":true,"requirements":[{"name":"rtl_nic/rtl8168d-1.fw","target":{"driver":"realtek","hardware":"10ec:8168","revision":"xid-281","board":null},"minimum_bytes":4}]}"#).unwrap();
        let cache = dir.path().join("cache");
        let destination = dir.path().join("destination");
        let text = provision(ProvisionOptions {
            requirements: Some(&inventory),
            gpu: None,
            wifi: None,
            profile: None,
            offline: false,
            destination: &destination,
            cache: &cache,
            dry_run: true,
            mount_image: &host_firmware::MOUNT_CAPABILITY,
        })
        .unwrap();
        assert!(text.contains("1492 bytes"));
        assert!(!cache.exists());
        assert!(!destination.exists());
        fs::write(
            &inventory,
            br#"{"schema":1,"complete":false,"requirements":[]}"#,
        )
        .unwrap();
        assert!(
            provision(ProvisionOptions {
                requirements: Some(&inventory),
                gpu: None,
                wifi: None,
                profile: None,
                offline: false,
                destination: &destination,
                cache: &cache,
                dry_run: true,
                mount_image: &host_firmware::MOUNT_CAPABILITY
            })
            .is_err()
        );
    }

    #[test]
    fn a_wifi_network_plans_the_regulatory_database_and_the_host_adapters_files() {
        let dir = tempfile::tempdir().unwrap();
        let plan = |adapters: &[(u16, u16)]| {
            provision(ProvisionOptions {
                requirements: None,
                gpu: None,
                wifi: Some(adapters),
                profile: None,
                offline: false,
                destination: &dir.path().join("destination"),
                cache: &dir.path().join("cache"),
                dry_run: true,
                mount_image: &host_firmware::MOUNT_CAPABILITY,
            })
            .unwrap()
        };
        // An Intel Ethernet controller and an MT7925.
        let text = plan(&[(0x8086, 0x125c), (0x14c3, 0x7925)]);
        for name in [
            "regulatory.db",
            "mediatek/mt7925/WIFI_MT7925_PATCH_MCU_1_1_hdr.bin",
            "mediatek/mt7925/WIFI_RAM_CODE_MT7925_1_1.bin",
        ] {
            assert!(text.contains(name), "{name}: {text}");
        }
        assert!(text.contains("3 exact firmware bindings"), "{text}");
        // An AX210 and an RTL8822CE.
        let text = plan(&[(0x8086, 0x2725), (0x10ec, 0xc822)]);
        for name in [
            "regulatory.db",
            "iwlwifi-ty-a0-gf-a0-89.ucode",
            "iwlwifi-ty-a0-gf-a0.pnvm",
            "rtw88/rtw8822c_fw.bin",
        ] {
            assert!(text.contains(name), "{name}: {text}");
        }
        assert!(text.contains("4 exact firmware bindings"), "{text}");
        // No adapter setup knows: the database alone.
        let text = plan(&[(0x8086, 0x125c)]);
        assert!(text.contains("1 exact firmware bindings"), "{text}");
        assert!(!dir.path().join("destination").exists());
    }

    #[test]
    fn transport_rejects_redirect_truncation_growth_and_wrong_digest() {
        use std::{net::TcpListener, thread};
        for (response, valid) in [
            (
                "HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc",
                true,
            ),
            (
                "HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nab",
                false,
            ),
            ("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nabcd", false),
            (
                "HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\nabcd",
                false,
            ),
            (
                "HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabd",
                false,
            ),
            (
                "HTTP/1.1 302 Found\r\nLocation: https://vendor.invalid/alternate\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                false,
            ),
        ] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let server = thread::spawn(move || {
                let (mut socket, _) = listener.accept().unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let _ = socket.read(&mut [0; 4096]);
                let _ = socket.write_all(response.as_bytes());
            });
            let artifact = Artifact {
                source: Box::leak(format!("http://{address}").into_boxed_str()),
                revision: "synthetic-v1",
                member: "one.bin",
                length: 3,
                sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            };
            let client = Client::builder()
                .timeout(Duration::from_secs(2))
                .redirect(Policy::none())
                .build()
                .unwrap();
            assert_eq!(download(&client, &artifact).is_ok(), valid);
            server.join().unwrap();
        }
    }

    #[test]
    fn a_gpu_joins_what_is_installed() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("firmware");
        fs::create_dir_all(&destination).unwrap();
        let manifest = serde_json::json!({"schema":1,"images":[{
            "name":"rtl_nic/rtl8168e-3.fw", "path":"rtl/one.bin",
            "target":{"driver":"realtek","hardware":"10ec:8168","revision":"xid-2c8","board":null},
            "provenance":{"source":"synthetic-fixture","revision":"v1","member":"one.bin","license":"CC0-1.0"},
            "length":3,"sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]});
        fs::write(
            destination.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let provision = |gpu| {
            provision(ProvisionOptions {
                requirements: None,
                gpu,
                wifi: None,
                profile: None,
                offline: false,
                destination: &destination,
                cache: &dir.path().join("cache"),
                dry_run: true,
                mount_image: &host_firmware::MOUNT_CAPABILITY,
            })
        };
        let report = provision(Some("tu104")).unwrap();
        assert!(report.contains("rtl_nic/rtl8168e-3.fw"));
        assert!(report.contains("nvidia/tu104/gsp/gsp-570.144.bin"));
        assert!(provision(None).is_err(), "nothing covers what is installed");
        assert!(provision(Some("gp104")).is_err());
    }

    /// A boot log on `esp` whose boots print `lines`.
    fn boot_log(esp: &Path, boots: &[&str]) {
        boot_log_in(esp, Layout::Legacy, boots);
    }

    fn boot_log_in(esp: &Path, layout: Layout, boots: &[&str]) {
        fs::create_dir_all(esp.join(layout.esp_dir())).unwrap();
        fs::write(
            esp.join(layout.esp_path(BOOT_LOG)),
            crate::boot_log::tests::region(boots),
        )
        .unwrap();
    }

    /// One inventory line as stela.efi prints it, of these requests.
    fn requests_line(requests: &[serde_json::Value]) -> String {
        format!(
            "[    3.250] {}{}\n",
            requirements::LOG_LINE,
            serde_json::json!({"schema":1,"complete":true,"requirements":requests})
        )
    }

    fn realtek(name: &str, xid: &str) -> serde_json::Value {
        serde_json::json!({"name":name,"target":{"driver":"realtek","hardware":"10ec:8168",
            "revision":xid,"board":null},"minimum_bytes":4})
    }

    #[test]
    fn modern_requests_use_the_modern_boot_log_and_publication_directory() {
        let dir = tempfile::tempdir().unwrap();
        let esp = dir.path();
        let legacy = requests_line(&[realtek("rtl_nic/rtl8168f-1.fw", "xid-480")]);
        let modern = requests_line(&[realtek("rtl_nic/rtl8168d-1.fw", "xid-281")]);
        boot_log_in(esp, Layout::Legacy, &[&legacy]);
        boot_log_in(esp, Layout::Northwood, &[&modern]);
        let image = host_firmware::MOUNT_CAPABILITY;
        let found = requested_in(esp, Layout::Northwood, &image).unwrap();
        assert_eq!(found.layout, Layout::Northwood);
        assert_eq!(found.pinned[0].name, "rtl_nic/rtl8168d-1.fw");
        let plan = found
            .plan(esp, &image, &esp.join("cache"), true)
            .unwrap()
            .unwrap();
        assert!(plan.contains("rtl_nic/rtl8168d-1.fw"), "{plan}");
        assert_eq!(
            directory(esp, found.layout),
            esp.join("EFI/northwood/firmware")
        );
        assert!(!directory(esp, Layout::Legacy).exists());
    }

    #[test]
    fn firmware_catalog_precedence_follows_the_efi_capability() {
        let dir = tempfile::tempdir().unwrap();
        let esp = dir.path();
        let old = host_firmware::MOUNT_CAPABILITY;
        let mut modern = old.to_vec();
        modern.extend_from_slice(&Layout::MOUNT_CAPABILITY);
        let legacy_path = directory(esp, Layout::Legacy);
        let modern_path = directory(esp, Layout::Northwood);
        assert_eq!(
            catalog_directory(esp, Layout::Northwood, &modern).unwrap(),
            legacy_path
        );
        fs::create_dir_all(&modern_path).unwrap();
        fs::write(
            modern_path.join("manifest.json"),
            br#"{"schema":1,"images":[]}"#,
        )
        .unwrap();
        assert_eq!(
            catalog_directory(esp, Layout::Legacy, &modern).unwrap(),
            modern_path
        );
        assert_eq!(
            catalog_directory(esp, Layout::Legacy, &old).unwrap(),
            legacy_path
        );
        assert!(publication_paths(esp, Layout::Legacy, &modern).is_err());
        assert_eq!(
            publication_paths(esp, Layout::Northwood, &modern)
                .unwrap()
                .0,
            modern_path
        );
    }

    #[test]
    fn the_boot_log_s_newest_requests_split_into_what_setup_adds_and_names() {
        let dir = tempfile::tempdir().unwrap();
        let esp = dir.path().join("esp");
        let destination = esp.join(host_firmware::DIRECTORY);
        fs::create_dir_all(&destination).unwrap();
        // The USB holds one Realtek file already.
        let manifest = serde_json::json!({"schema":1,"images":[{
            "name":"rtl_nic/rtl8168e-3.fw", "path":"rtl/one.bin",
            "target":{"driver":"realtek","hardware":"10ec:8168","revision":"xid-2c8","board":null},
            "provenance":{"source":"synthetic-fixture","revision":"v1","member":"one.bin","license":"CC0-1.0"},
            "length":3,"sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]});
        fs::write(
            destination.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let image = host_firmware::MOUNT_CAPABILITY;
        // No log, or a log without requests: nothing.
        assert!(requested(&esp, &image).unwrap().is_empty());
        boot_log(&esp, &["CHOICE 1 (macOS)\n"]);
        assert!(requested(&esp, &image).unwrap().is_empty());
        let unpinned = serde_json::json!({"name":"brcm/synthetic.bin","target":{
            "driver":"synthetic-bluetooth","hardware":"pci:14e4:5fa0","revision":"c0",
            "board":null},"minimum_bytes":1});
        let variable = serde_json::json!({"name":"intel/wifi/uefi/UefiCnvWlanUATS.bin",
            "target":{"driver":"wifi-intel","hardware":"uefi:92daaf2f-c02b-455b-b2ec-f5a3594f4aea",
            "revision":"CnvUefiWlanUATS","board":null},"minimum_bytes":1});
        let older = requests_line(&[realtek("rtl_nic/rtl8168f-1.fw", "xid-480")]);
        let newer = requests_line(&[
            realtek("rtl_nic/rtl8168e-3.fw", "xid-2c8"),
            realtek("rtl_nic/rtl8168d-1.fw", "xid-281"),
            unpinned,
            variable,
        ]);
        boot_log(
            &esp,
            &[&older, &format!("USB added 1-4\n{newer}HOST WARNING x\n")],
        );
        let found = requested(&esp, &image).unwrap();
        let names = |requests: &[Request]| -> Vec<String> {
            requests
                .iter()
                .map(|request| request.name.clone())
                .collect()
        };
        assert_eq!(names(&found.pinned), ["rtl_nic/rtl8168d-1.fw"]);
        assert_eq!(names(&found.unpinned), ["brcm/synthetic.bin"]);
        assert!(found.unreadable.is_none());
        let lines = found.lines().join("\n");
        assert!(
            lines.contains("add rtl_nic/rtl8168d-1.fw (realtek 10ec:8168 xid-281)"),
            "{lines}"
        );
        assert!(
            lines.contains("brcm/synthetic.bin (synthetic-bluetooth pci:14e4:5fa0 c0), which Stela has no pinned source for"),
            "{lines}"
        );
        // The plan keeps what the USB holds and downloads nothing in a dry run.
        let cache = dir.path().join("cache");
        let plan = found.plan(&esp, &image, &cache, true).unwrap().unwrap();
        assert!(plan.contains("2 exact firmware bindings"), "{plan}");
        assert!(plan.contains("rtl_nic/rtl8168e-3.fw"), "{plan}");
        assert!(plan.contains("rtl_nic/rtl8168d-1.fw"), "{plan}");
        assert!(plan.contains("Dry run"), "{plan}");
        assert!(!cache.exists());
        // Unpinned requests alone plan nothing.
        let only = Requested {
            pinned: Vec::new(),
            ..found
        };
        assert!(only.plan(&esp, &image, &cache, true).unwrap().is_none());
        only.provision(&esp, &image).unwrap();
        // A boot program that reads no firmware from the USB asked for none.
        assert!(requested(&esp, b"no marker").unwrap().is_empty());
        // A line cut short is named, not followed.
        boot_log(&esp, &[&newer[..60]]);
        let cut = requested(&esp, &image).unwrap();
        assert!(cut.pinned.is_empty() && cut.unreadable.is_some(), "{cut:?}");
    }

    #[test]
    #[ignore = "oracle: STELA_FIRMWARE_TEST_OUTPUT"]
    fn vendor_provisioning_binds_pins_and_notices() {
        let output = std::path::PathBuf::from(
            std::env::var_os("STELA_FIRMWARE_TEST_OUTPUT")
                .expect("explicit fresh evidence directory"),
        );
        assert!(!output.exists(), "never overwrite historical evidence");
        fs::create_dir_all(&output).unwrap();
        let requests = [
            ("rtl_nic/rtl8168d-1.fw", "xid-281"), ("rtl_nic/rtl8168d-2.fw", "xid-280"),
            ("rtl_nic/rtl8168e-1.fw", "xid-2c1"), ("rtl_nic/rtl8168e-2.fw", "xid-2c0"),
            ("rtl_nic/rtl8168e-3.fw", "xid-2c8"), ("rtl_nic/rtl8168f-1.fw", "xid-480"),
            ("rtl_nic/rtl8168f-2.fw", "xid-481"), ("rtl_nic/rtl8411-1.fw", "xid-488"),
        ].map(|(name, revision)| serde_json::json!({"name":name,"target":{"driver":"realtek","hardware":"10ec:8168","revision":revision,"board":null},"minimum_bytes":4}));
        let inventory = output.join("synthetic-observed-requirements.json");
        fs::write(
            &inventory,
            serde_json::to_vec_pretty(
                &serde_json::json!({"schema":1,"complete":true,"requirements":requests}),
            )
            .unwrap(),
        )
        .unwrap();
        let destination = output.join("firmware");
        let cache = output.join("cache");
        let report = provision(ProvisionOptions {
            requirements: Some(&inventory),
            gpu: None,
            wifi: None,
            profile: None,
            offline: false,
            destination: &destination,
            cache: &cache,
            dry_run: false,
            mount_image: &host_firmware::MOUNT_CAPABILITY,
        })
        .unwrap();
        fs::write(output.join("provisioning.txt"), report).unwrap();
        let manifest_bytes = fs::read(destination.join("manifest.json")).unwrap();
        let manifest = host_firmware::Manifest::parse(&manifest_bytes).unwrap();
        let observed = read_inventory(&inventory).unwrap();
        assert_eq!(manifest.images.len(), 8);
        for (image, request) in manifest.images.iter().zip(observed.requests()) {
            assert_eq!(image.name, request.name);
            assert_eq!(image.target.revision, request.target.revision);
            assert_eq!(image.target.board, request.target.board.as_deref());
            let bytes = fs::read(destination.join(image.path)).unwrap();
            let recipe = Plan::select(&observed).unwrap();
            recipe
                .selections()
                .iter()
                .find(|selected| selected.request.name == image.name)
                .unwrap()
                .recipe
                .image
                .verify(&bytes)
                .unwrap();
        }
        let objects = destination
            .join(manifest.images[0].path)
            .parent()
            .unwrap()
            .to_path_buf();
        let notice = fs::read(objects.join("license-0.txt")).unwrap();
        assert_eq!(notice.len(), 219);
        assert!(
            String::from_utf8(notice)
                .unwrap()
                .contains("Realtek Semiconductor Corporation")
        );
        assert!(objects.join("provenance.json").is_file());
        assert_eq!(
            host_firmware::Package::read(&destination)
                .unwrap()
                .inventory(),
            (8, 22876)
        );
        let again = provision(ProvisionOptions {
            requirements: Some(&inventory),
            gpu: None,
            wifi: None,
            profile: None,
            offline: false,
            destination: &destination,
            cache: &cache,
            dry_run: false,
            mount_image: &host_firmware::MOUNT_CAPABILITY,
        })
        .unwrap();
        assert!(again.contains("Published pinned firmware"));
        assert_eq!(
            fs::read(destination.join("manifest.json")).unwrap(),
            manifest_bytes
        );
    }
}
