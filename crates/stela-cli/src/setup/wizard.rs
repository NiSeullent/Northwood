//! A small Clack-style front end to the same setup command used by scripts.
use super::{
    InstallImage, Request, firmware,
    host::{self, Disk, Host, Removal},
    record::{self, Record},
    wifi,
};
use crate::{progress::Phase, restore};
use anyhow::{Context, Result, ensure};
use northwood_config::Swap;
use std::{
    io::IsTerminal,
    path::{Path, PathBuf},
};

mod catalog;
mod choices;
mod menu;
mod theme;

// Advertised USB capacity classes; preparation also checks exact image sizes.
const NETWORK_MIN: u64 = 7 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Usb {
    id: String,
    name: String,
    bytes: u64,
    removal: Removal,
    serial: Option<String>,
}

impl Usb {
    pub(super) fn from_disk(disk: &Disk, host: &dyn Host) -> Self {
        Self {
            id: disk.identifier.clone(),
            name: disk.name.clone(),
            bytes: disk.bytes,
            removal: disk.removal,
            serial: host.serial(&disk.identifier).ok().flatten(),
        }
    }

    pub(super) fn verify(&self, host: &dyn Host) -> Result<()> {
        let disk = host
            .disks()?
            .into_iter()
            .find(|d| d.identifier == self.id)
            .context("the selected USB was disconnected; run setup again")?;
        ensure!(
            disk.erasable() && *self == Self::from_disk(&disk, host),
            "the selected USB changed; run setup again and choose it again"
        );
        Ok(())
    }

    fn path(&self) -> PathBuf {
        PathBuf::from(&self.id)
    }

    fn size(&self) -> String {
        host::size_text(self.bytes)
    }

    fn description(&self) -> String {
        format!("{} · {} · {}", self.name, self.size(), self.id)
    }

    /// Its model or size, typed back to erase a disk other than a USB.
    fn typed_back(&self) -> Result<String> {
        cliclack::log::warning(format!(
            "{} ({}, {}) is not a USB disk.",
            self.name,
            self.size(),
            self.id
        ))?;
        let (name, bytes) = (self.name.clone(), self.bytes);
        let expected = format!("Type {} or {}", self.name, self.size());
        Ok(cliclack::input("Type its model or size to erase it")
            .placeholder(&format!("{} or {}", self.name, self.size()))
            .validate(move |typed: &String| {
                if host::names_disk(&name, bytes, typed) {
                    Ok(())
                } else {
                    Err(expected.clone())
                }
            })
            .interact()?)
    }

    /// The name and size people recognize, for the erase warning.
    fn short(&self) -> String {
        format!("{} ({})", self.name, self.size())
    }
}

#[derive(serde::Deserialize)]
struct LocalImage {
    #[serde(rename = "ProductVersion")]
    product_version: String,
    #[serde(rename = "ProductBuildVersion")]
    product_build: String,
    #[serde(rename = "SupportedProductTypes")]
    supported_models: Vec<String>,
}

fn local_image(path: &Path) -> Result<LocalImage> {
    // Read only metadata into the wizard, rather than keeping every board's
    // component map. Preparation validates the selected identity before erase.
    let bytes = restore::archive::read_member(path, "BuildManifest.plist", 64 * 1024 * 1024)?;
    Ok(plist::from_reader(std::io::Cursor::new(bytes))?)
}

/// `false` is cancellation. No disk is written by this function.
pub(super) fn configure(command: &mut Request, host: &dyn Host) -> Result<bool> {
    ensure!(
        std::io::stdin().is_terminal() && std::io::stderr().is_terminal(),
        "interactive setup needs a terminal; scripts pass --usb, and --ipsw for a local image"
    );
    match configure_inner(command, host) {
        Err(error)
            if error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|e| e.kind() == std::io::ErrorKind::Interrupted) =>
        {
            cliclack::outro_cancel("Setup cancelled. Your USB was not changed.")?;
            Ok(false)
        }
        result => result,
    }
}

/// The EFI partition of `usb` and the interrupted first setup it holds,
/// which the same choices again resume (`create::plan`). The partition is
/// mounted to look, as setup mounts it anyway.
pub(super) fn interrupted(host: &dyn Host, usb: &Disk) -> Option<(PathBuf, Record)> {
    let partition = usb.partitions.iter().find(|p| p.content == "EFI")?;
    let esp = match &partition.mount {
        Some(mount) => mount.clone(),
        None => host.mount(&partition.identifier).ok()?,
    };
    let record = record::interrupted(host, &esp, &esp)?;
    Some((esp, record))
}

/// Make `command` the interrupted setup `record` describes, on its EFI
/// partition `esp`, so it resumes instead of erasing the USB again. A local
/// IPSW given on the command line stays; otherwise setup downloads the
/// recorded build.
pub(super) fn resume(command: &mut Request, esp: PathBuf, record: &Record) {
    command.usb = esp;
    command.build = record.build.clone();
    command.model = Some(record.model.clone());
    // A resumed setup is a new one, and streams the install as every new
    // setup does, also when an older stela began it with the whole image
    // (`stela setup` run again does the same).
    command.install_image = Some(InstallImage::Network);
    if command.ipsw.is_none() {
        command.fetch = true;
    }
    command.install = record.installs.first().map(|install| install.name.clone());
    command.wifi = record.wifi.as_ref().map(|network| network.ssid.clone());
    command.wifi_country = record
        .wifi
        .as_ref()
        .and_then(|network| network.country.clone());
    command.ipv4 = record
        .addresses
        .ipv4
        .map(|ipv4| stela_boot_core::options::StaticIpv4 {
            address: ipv4.address,
            prefix: ipv4.prefix,
        });
    command.ipv4_gateway = record.addresses.ipv4.and_then(|ipv4| ipv4.gateway);
    command.dns = (!record.addresses.dns.is_empty()).then(|| record.addresses.dns.clone());
    command.usb4_approve = record.usb4_approved.clone();
    command.profile = record.profile;
}

fn configure_inner(command: &mut Request, host: &dyn Host) -> Result<bool> {
    cliclack::set_theme(theme::Stela);
    crate::progress::follow(theme::rail());
    cliclack::intro(
        theme::accent()
            .bold()
            .apply_to("Stela · Create a macOS installer"),
    )?;
    let disks = host.disks()?;
    let choices: Vec<_> = disks
        .iter()
        .filter(|d| d.erasable())
        .map(|d| Usb::from_disk(d, host))
        .collect();
    ensure!(
        !choices.is_empty(),
        "no removable USB found. Connect a USB and run setup again."
    );
    let mut images = None;
    let mut names = catalog::cached_names();
    let mut first = true;
    // `--build`, or the newest supported build.
    let mut preferred_build = Some(command.build.clone());
    loop {
        let rows: Vec<_> = choices
            .iter()
            .enumerate()
            .map(|(i, usb)| menu::Item {
                value: i,
                label: usb.description(),
                hint: if usb.bytes == 0 {
                    "Capacity unavailable".into()
                } else if usb.bytes < NETWORK_MIN {
                    "Too small · Requires an 8 GB or larger USB".into()
                } else if usb.bytes < super::FULL_MIN {
                    "Network installer · macOS downloads during installation".into()
                } else {
                    "Full installer · macOS goes on the USB".into()
                },
                enabled: usb.bytes >= NETWORK_MIN,
            })
            .collect();
        if !rows.iter().any(|row| row.enabled) {
            cliclack::note(
                "USBs found",
                rows.iter()
                    .map(|row| format!("{} · {}", row.label, row.hint))
                    .collect::<Vec<_>>()
                    .join("\n"),
            )?;
            anyhow::bail!("no USB of 8 GB or more found; connect one and run setup again");
        }
        let initial = choices
            .iter()
            .position(|usb| usb.path() == command.usb)
            .unwrap_or(0);
        let usb = &choices[menu::select("1 / 6  Choose a USB", &rows, initial)?];
        command.usb = usb.path();
        // A setup this USB holds that stopped part way is finished, not
        // erased and begun again.
        let disk = disks.iter().find(|d| d.identifier == usb.id);
        if let Some((esp, record)) = disk.and_then(|disk| interrupted(host, disk)) {
            let choice = cliclack::select("This USB holds a setup that was interrupted")
                .item(0, "Resume it", "Keeps what it wrote and finishes the rest")
                .item(1, "Start over", "Erases the USB")
                .initial_value(0)
                .interact()?;
            if choice == 0 {
                resume(command, esp, &record);
                if command.wifi.is_some() {
                    cliclack::log::warning("The passphrase is stored in plaintext on the USB.")?;
                    let ssid = command.wifi.clone().unwrap_or_default();
                    command.wifi_passphrase = Some(wifi::ask_passphrase(&ssid)?);
                }
                return Ok(true);
            }
        }
        // A USB of 32 GB or more takes the whole macOS image, so the
        // install needs no network; a smaller one gets the network
        // installer. A local IPSW named on the command line supplies the
        // boot files, and macOS too when it holds the whole image.
        let mut install_image = super::default_install_image(usb.bytes);
        let prefilled = first && command.ipsw.is_some();
        first = false;
        let (version, models) = if prefilled {
            let path = command.ipsw.clone().unwrap();
            let progress = Phase::spinner("Reading IPSW metadata");
            let manifest =
                local_image(&path).with_context(|| format!("reading {}", path.display()))?;
            progress.finish();
            stela_formats::builds::find(&manifest.product_build)
                .with_context(|| path.display().to_string())?;
            command.ipsw = Some(path);
            command.fetch = false;
            command.build = manifest.product_build.clone();
            preferred_build = Some(command.build.clone());
            if !crate::restore::payload::holds_system_image(command.ipsw.as_deref().unwrap())? {
                install_image = InstallImage::Network;
            }
            command.install_image = Some(install_image);
            cliclack::log::info(format!(
                "macOS {} ({}) · local IPSW{}",
                manifest.product_version,
                manifest.product_build,
                if stela_formats::builds::tested(&manifest.product_build) {
                    ""
                } else {
                    " · not tested yet"
                }
            ))?;
            (manifest.product_version, manifest.supported_models)
        } else {
            if images.is_none() {
                let progress = Phase::spinner("Loading available macOS versions");
                images = Some(catalog::load()?);
                if let Ok(fresh) = catalog::model_names() {
                    names = fresh;
                }
                progress.finish();
            }
            let entries = images.as_ref().unwrap();
            let image =
                &entries[choices::release(entries, preferred_build.as_deref().unwrap_or(""))?];
            command.build = image.build.clone();
            preferred_build = Some(command.build.clone());
            command.ipsw = None;
            command.fetch = true;
            command.install_image = Some(install_image);
            (image.version.clone(), image.models.clone())
        };
        ensure!(
            !models.is_empty(),
            "this image lists no compatible Mac models"
        );
        let mut models = models;
        models.sort();
        models.dedup();
        command.model = Some(choices::model(&models, &names, command.model.as_deref())?);
        // What the install needs the network for, and what is not known
        // about the installed system's first start.
        let full = install_image == InstallImage::Full;
        for note in super::create::network_notes(!full, full) {
            cliclack::log::info(note)?;
        }
        choose_wifi(command)?;
        super::addresses::choose(command)?;
        choose_swap(command)?;
        cliclack::note(
            "6 / 6  Review",
            format!(
                "USB        {}\nInstaller  {}\nmacOS      {version} ({})\nMac        {}\nUSB swap   {}\n{}{}{}",
                usb.serial.as_ref().map_or_else(
                    || usb.description(),
                    |serial| format!("{} · serial {serial}", usb.description())
                ),
                match full {
                    true => "Full · macOS goes on the USB, the install needs no network",
                    false => "Network · macOS downloads during installation",
                },
                command.build,
                names
                    .get(command.model())
                    .map_or(command.model(), String::as_str),
                match command.volume_swap_bytes {
                    Some(bytes) => format!("Prepare · {} requested", swap_size(bytes)),
                    None => "Skip".into(),
                },
                match &command.wifi {
                    Some(ssid) => format!("Wi-Fi      {ssid}\n"),
                    None => "Wi-Fi      None (Ethernet)\n".into(),
                },
                super::addresses::review(command),
                // Only a scripted --profile turns it on.
                if command.profile {
                    "Profiling  On\n"
                } else {
                    ""
                },
            ),
        )?;
        let (chips, notes) = firmware::host_chips(host)?;
        for note in notes {
            cliclack::log::warning(note)?;
        }
        if chips.is_empty() {
            cliclack::log::warning(firmware::missing(
                "`stela setup host-firmware fetch --gpu CHIP --usb USB` (USB: the installer's \
                 data volume)",
            ))?;
        }
        cliclack::log::info(super::create::SIGNING_NOTE)?;
        cliclack::log::warning(format!("Everything on {} will be erased.", usb.short()))?;
        match cliclack::select("Create the installer?")
            .item(
                0,
                "Create installer",
                format!("Erase {} and write the installer", usb.short()),
            )
            .item(1, "Edit choices", "")
            .item(2, "Cancel", "")
            .initial_value(0)
            .interact()?
        {
            0 => {
                if usb.removal == Removal::Typed {
                    command.confirm_disk = Some(usb.typed_back()?);
                }
                usb.verify(host)?;
                command.confirmed = true;
                return Ok(true);
            }
            1 => continue,
            _ => {
                cliclack::outro_cancel("Setup cancelled. Your USB was not changed.")?;
                return Ok(false);
            }
        }
    }
}

fn swap_size(bytes: u64) -> String {
    const GIB: u64 = 1 << 30;
    if bytes.is_multiple_of(GIB) {
        format!("{} GiB", bytes / GIB)
    } else {
        host::bytes_text(bytes)
    }
}

fn choose_swap(command: &mut Request) -> Result<()> {
    let policy = Swap::default();
    let requested = command.volume_swap_bytes.unwrap_or(policy.bytes);
    let prepare =
        cliclack::select("Memory and swap · Does the target PC have 4 GiB of RAM or less?")
            .item(
                false,
                "Skip USB swap",
                "For PCs with more than 4 GiB of RAM or unknown RAM",
            )
            .item(
                true,
                "Prepare USB swap",
                format!(
                    "Requests {} · Up to {} on FAT32, depending on free space",
                    swap_size(requested),
                    swap_size(policy.volume_max_bytes),
                ),
            )
            .initial_value(command.volume_swap_bytes.is_some())
            .interact()?;
    command.volume_swap_bytes = prepare.then_some(requested);
    Ok(())
}

/// A network installer's Wi-Fi: a network this computer sees or knows, one
/// typed in, or none.
fn choose_wifi(command: &mut Request) -> Result<()> {
    let progress = Phase::spinner("Looking for Wi-Fi networks");
    let networks = wifi::host_networks();
    progress.finish();
    const ETHERNET: usize = usize::MAX;
    const TYPED: usize = usize::MAX - 1;
    let mut select = cliclack::select("4 / 6  Wi-Fi during installation").item(
        ETHERNET,
        "No Wi-Fi",
        "Connect the PC to Ethernet during installation",
    );
    for (index, ssid) in networks.iter().enumerate() {
        select = select.item(index, ssid, "");
    }
    let initial = command
        .wifi
        .as_ref()
        .and_then(|ssid| networks.iter().position(|known| known == ssid))
        .unwrap_or(if command.wifi.is_some() {
            TYPED
        } else {
            ETHERNET
        });
    let choice = select
        .item(TYPED, "Another network", "Type its name")
        .initial_value(initial)
        .interact()?;
    let ssid = match choice {
        ETHERNET => {
            (command.wifi, command.wifi_passphrase) = (None, None);
            return Ok(());
        }
        TYPED => cliclack::input("Network name")
            .default_input(command.wifi.as_deref().unwrap_or_default())
            .validate(|name: &String| wifi::ssid(name).map(|_| ()))
            .interact()?,
        index => networks[index].clone(),
    };
    cliclack::log::warning("The passphrase is stored in plaintext on the USB.")?;
    command.wifi_passphrase = Some(wifi::ask_passphrase(&ssid)?);
    command.wifi = Some(ssid);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_manifest_checks_version_and_model() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("local.ipsw");
        crate::efi::synthetic_ipsw(&path)?;
        let manifest = local_image(&path)?;
        assert_eq!(manifest.product_build, restore::payload::SYNTHETIC_BUILD);
        assert!(
            manifest
                .supported_models
                .contains(&restore::payload::SYNTHETIC_MODEL.into())
        );
        assert!(local_image(&dir.path().join("missing.ipsw")).is_err());
        Ok(())
    }

    #[test]
    fn maintenance_without_a_usb_is_refused_before_opening_prompts() {
        use clap::Parser;
        for words in [
            &["status"][..],
            &["repair"],
            &["uninstall"],
            &["update", "stela"],
            &["configure", "--no-profile"],
        ] {
            let line = ["stela", "setup"].into_iter().chain(words.iter().copied());
            assert!(crate::Cli::try_parse_from(line).is_err(), "{words:?}");
        }
    }
}
