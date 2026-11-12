//! The setup journal on the ESP, read by later commands and written before
//! each change. The journal's directory determines its layout.
use super::{
    CHOOSER_SECONDS, Environment, InstallImage, LOADER_PATH, addresses,
    bundle::{self, Kind},
    ecid::Ecid,
    host::{FileSystem, Host},
    image_is_remote, installed, locate_esp, locate_esp_with_record, machines, medium,
    restore_choice_name, translations, wifi,
};
use crate::efi;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};
use stela_boot_core::options::{System, Uuid};
use stela_loader::mount_interface::{
    Choice, Choices,
    layout::{self, Layout},
};

/// What setup writes, recorded on the ESP so `uninstall` can remove it.
/// Records before schema 8 also named a recoveryOS bundle on the USB, and
/// records before schema 9 kept the data partition's files under `EFI/`;
/// this `stela` reads them only to uninstall them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Record {
    pub(super) schema_version: u32,
    pub(super) kind: String,
    /// Selected from the journal's directory on read, never from its JSON.
    #[serde(skip, default = "legacy_layout")]
    pub(super) layout: Layout,
    /// The active macOS build, which Install macOS restores.
    pub(super) build: String,
    pub(super) model: String,
    pub(super) esp: PathBuf,
    /// The USB's FAT32 data partition.
    #[serde(alias = "payload_volume")]
    pub(super) volume: PathBuf,
    /// A schema 1 setup's restore payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) payload_directory: Option<PathBuf>,
    /// `stela/boot/<build>` of the active build.
    pub(super) bundle_directory: PathBuf,
    /// Directories setup created on the payload volume, deepest first.
    pub(super) created_directories: Vec<PathBuf>,
    /// `stela.cfg`.
    pub(super) options: String,
    pub(super) complete: bool,
    /// Schema 3: the PC's own disk; read into `installs`.
    #[serde(default, skip_serializing)]
    pub(super) installed: Option<installed::InstalledDisk>,
    /// Schema 3: the installed `stela.efi`'s version.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) stela_version: Option<String>,
    /// Schema 4: the PC disks macOS installs onto, each its own machine.
    #[serde(default)]
    pub(super) installs: Vec<machines::Install>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) default_install: Option<String>,
    /// Schema 4: the boot-bundle schema the active `stela.efi` reads.
    #[serde(default = "first_schema")]
    pub(super) bundle_schema: u32,
    /// Schema 7: the revision of the active bundles, those of the loader
    /// that prepared them (`stela_loader::boot_bundle::REVISION`); 0 before.
    #[serde(default)]
    pub(super) bundle_revision: u32,
    /// Schema 4: every boot bundle setup wrote.
    #[serde(default)]
    pub(super) bundles: Vec<bundle::Bundle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) chooser_timeout: Option<u32>,
    /// Schema 5: the install medium; read into `media`.
    #[serde(default, skip_serializing)]
    pub(super) medium: Option<medium::Medium>,
    /// Schema 6: the install media, each a full IPSW on the USB's FAT32
    /// data partition: the active build's, and the next one's while
    /// `update os` writes it.
    #[serde(default)]
    pub(super) media: Vec<medium::Medium>,
    /// Schema 5: the Install choice checks the IPSW as it reads it
    /// (`--verify-on-read`).
    #[serde(default)]
    pub(super) verify_on_read: bool,
    /// Every choice times the engine (`--profile`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) profile: bool,
    /// Where installs read the restore image; streamed when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) install_image: Option<InstallImage>,
    /// The translation store `stela.efi` keeps its packs in, if the USB
    /// has room for one (`setup::translations`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) translations: Option<translations::Store>,
    /// Files setup owns on the bound FAT32 data partition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) volume_swap: Option<super::volume_swap::Store>,
    /// The Wi-Fi network every choice joins (`--wifi`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) wifi: Option<wifi::Network>,
    /// The USB4 devices every choice admits without an IOMMU
    /// (`--usb4-approve`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) usb4_approved: Vec<String>,
    /// The network addresses every choice sets by hand (`--ipv4`, `--dns`).
    #[serde(default, skip_serializing_if = "addresses::Addresses::is_automatic")]
    pub(super) addresses: addresses::Addresses,
}

pub(super) fn first_schema() -> u32 {
    1
}

fn legacy_layout() -> Layout {
    Layout::Legacy
}

pub(super) const RECORD_KIND: &str = "stela_setup";

/// Schema 8: no recoveryOS bundle; every setup has an install. Schema 9:
/// the data partition's files are under `stela/`, not `EFI/stela/`.
pub(super) const RECORD_SCHEMA: u32 = 9;

impl Record {
    fn set_layout(&mut self, layout: Layout) {
        self.layout = layout;
        for medium in self.media.iter_mut().chain(self.medium.as_mut()) {
            medium.layout = layout;
        }
        if let Some(store) = self.translations.as_mut() {
            store.layout = layout;
        }
    }

    /// Whether installs stream `ipsw`'s build from Apple's servers rather
    /// than read it from the USB, as every setup does but one that already
    /// holds whole images. An erase restore needs the macOS system image,
    /// which a recovery archive (`restore::payload::trim`) leaves out.
    pub(super) fn remote_install(&self, ipsw: &Path) -> Result<bool> {
        image_is_remote(ipsw, self.install_image)
    }

    /// Bring an older schema's record to one `uninstall` reads: its
    /// install medium is listed and its installed disk becomes the first
    /// install.
    pub(super) fn migrate(&mut self) {
        if let Some(medium) = self.medium.take()
            && !self.media.iter().any(|m| m.build == medium.build)
        {
            self.media.push(medium);
        }
        if let Some(disk) = self.installed.take() {
            self.installs.push(machines::Install {
                name: machines::FIRST_NAME.into(),
                ecid: None,
                disk,
            });
        }
    }

    /// The install `stela.cfg` boots.
    pub(super) fn default(&self) -> Option<&machines::Install> {
        self.default_install
            .as_ref()
            .and_then(|name| self.installs.iter().find(|i| &i.name == name))
            .or(self.installs.first())
    }

    /// The install `name` names, or the default one.
    pub(super) fn install(&self, name: Option<&str>) -> Result<&machines::Install> {
        match name {
            Some(name) => self
                .installs
                .iter()
                .find(|i| i.name == name)
                .with_context(|| format!("this setup has no install named {name:?}")),
            None => self.default().with_context(|| {
                format!(
                    "the setup on {} names no installed disk (--target)",
                    self.esp.display()
                )
            }),
        }
    }

    /// The generation of the active bundles.
    pub(super) fn generation(&self) -> bundle::Generation {
        bundle::Generation {
            schema: self.bundle_schema,
            revision: self.bundle_revision,
        }
    }

    pub(super) fn line(&self, mut system: System) -> String {
        // Every bundle of a setup is of its one Mac.
        system.guest_pes = self
            .bundles
            .iter()
            .map(|bundle| bundle.processors)
            .find(|&processors| processors != 0);
        system.profile = self.profile;
        system.ssid = self.wifi.as_ref().map(|network| network.ssid.clone());
        if self.layout == Layout::Northwood && system.ssid.is_some() && system.wifi_key.is_none() {
            system.wifi_key = Some(self.layout.uefi_path(layout::WIFI_KEY));
        }
        system.wifi_country = self
            .wifi
            .as_ref()
            .and_then(|network| network.country.clone());
        system.usb4_approved = self
            .usb4_approved
            .iter()
            .filter_map(|uuid| Uuid::parse(uuid))
            .collect();
        self.addresses.apply(&mut system);
        format!("{system}\n")
    }

    /// The option line that boots `install`.
    pub(super) fn boot_line(&self, install: &machines::Install) -> Result<String> {
        Ok(self.line(install.disk.options(install.ecid)?))
    }

    /// `stela.cfg`: the default install. Every setup this `stela` reads
    /// has one ([`read_record`]).
    pub(super) fn options_text(&self) -> Result<String> {
        self.default()
            .map(|install| self.boot_line(install))
            .transpose()
            .map(Option::unwrap_or_default)
    }

    /// The active build's restore bundle of `kind` (erase or update) for
    /// `ecid`'s machine, under the AP ticket Apple's signing service
    /// personalized for it. A bundle under the IPSW's global ticket, which
    /// an older setup without the signing opt-in prepared, counts as
    /// missing, so `repair --ipsw` prepares it again.
    pub(super) fn restore_bundle(&self, kind: Kind, ecid: Option<Ecid>) -> Option<&bundle::Bundle> {
        self.bundles.iter().find(|b| {
            b.same_kind(bundle::Identity {
                kind,
                build: &self.build,
                ecid,
                generation: self.generation(),
            }) && b.personalized
        })
    }

    /// The install medium of `build`.
    pub(super) fn medium_of(&self, build: &str) -> Option<&medium::Medium> {
        self.media.iter().find(|m| m.build == build)
    }

    /// The active build's install medium.
    pub(super) fn active_medium(&self) -> Option<&medium::Medium> {
        self.medium_of(&self.build)
    }

    /// Whether the installs restore from install media: the setup has
    /// written a medium or a restore bundle, as every setup does.
    pub(super) fn installs_media(&self) -> bool {
        !self.installs.is_empty() && (!self.media.is_empty() || !self.bundles.is_empty())
    }

    /// The machines of the installs, each once.
    pub(super) fn install_machines(&self) -> Vec<Option<Ecid>> {
        let mut machines = Vec::new();
        for install in &self.installs {
            if !machines.contains(&install.ecid) {
                machines.push(install.ecid);
            }
        }
        machines
    }

    /// The option line of `install`'s Install macOS choice
    /// ([`Kind::Restore`]) or Update macOS choice
    /// ([`Kind::Update`]): its restore bundle of that kind, the install
    /// medium and its partitions. None without the active build's medium or
    /// the machine's personalized bundle.
    pub(super) fn restore_line(
        &self,
        kind: Kind,
        install: &machines::Install,
    ) -> Result<Option<String>> {
        let Some(medium) = self.active_medium() else {
            return Ok(None);
        };
        if self.restore_bundle(kind, install.ecid).is_none() {
            return Ok(None);
        }
        let restore = bundle::firmware_path_of_in(
            self.layout,
            bundle::Identity {
                kind,
                build: &self.build,
                ecid: install.ecid,
                generation: self.generation(),
            },
        );
        let mut system = install.disk.bundle_options(&restore, install.ecid)?;
        medium.restore_from(&mut system)?;
        system.update_restore = kind == Kind::Update;
        system.verify_on_read = self.verify_on_read;
        system.development_restore = kind == Kind::Restore;
        system.input = kind == Kind::Restore;
        Ok(Some(self.line(system)))
    }

    /// The Install macOS choice's option line.
    pub(super) fn install_line(&self, install: &machines::Install) -> Result<Option<String>> {
        self.restore_line(Kind::Restore, install)
    }

    /// The name of `install`'s Install macOS or Update macOS
    /// (Stela) choice.
    pub(super) fn restore_choice(&self, kind: Kind, install: &machines::Install) -> String {
        if self.installs.len() == 1 {
            return restore_choice_name(kind).into();
        }
        let several = match kind {
            Kind::Update => "Update macOS",
            _ => "Install macOS",
        };
        format!("{} - {several}", install.name)
    }

    /// The restore bundles the installs' machines need of the active build
    /// and lack: both kinds of every install's while the setup has the
    /// build's install medium.
    pub(super) fn missing_restore_bundles(&self) -> Vec<(Kind, Option<Ecid>)> {
        if self.active_medium().is_none() {
            return Vec::new();
        }
        let mut missing = Vec::new();
        for kind in bundle::RESTORE_KINDS {
            for ecid in self.install_machines() {
                let present = self.restore_bundle(kind, ecid).is_some_and(|b| {
                    b.directory_in(&self.volume, self.layout)
                        .join(bundle::MANIFEST)
                        .is_file()
                });
                if !present {
                    missing.push((kind, ecid));
                }
            }
        }
        missing
    }

    /// The startup choices: each install, default first, and its Install
    /// macOS. None without an installed disk.
    pub(super) fn choices(&self) -> Result<Option<Choices>> {
        let Some(default) = self.default() else {
            return Ok(None);
        };
        let mut entries = Vec::new();
        let others = self.installs.iter().filter(|i| i.name != default.name);
        for install in std::iter::once(default).chain(others) {
            entries.push(Choice {
                name: install.name.clone(),
                options: self.boot_line(install)?.trim().into(),
            });
            for kind in bundle::RESTORE_KINDS {
                if let Some(line) = self.restore_line(kind, install)? {
                    entries.push(Choice {
                        name: self.restore_choice(kind, install),
                        options: line.trim().into(),
                    });
                }
            }
        }
        // The list opens on its own with several installs, and while the
        // USB can install macOS.
        let mut installing = false;
        for install in &self.installs {
            installing |= self.install_line(install)?.is_some();
        }
        let seconds = if self.installs.len() > 1 || installing {
            CHOOSER_SECONDS
        } else {
            0
        };
        Ok(Some(Choices {
            timeout_seconds: self.chooser_timeout.unwrap_or(seconds),
            entries,
        }))
    }

    pub(super) fn choices_text(&self) -> Result<Option<String>> {
        self.choices()?
            .map(|choices| choices.encode().map_err(anyhow::Error::msg))
            .transpose()
    }

    /// Forget the bundles of machines the setup no longer has; returns
    /// them for [`remove_bundles`] once the record no longer names them.
    pub(super) fn drop_stale_bundles(&mut self) -> Vec<bundle::Bundle> {
        let machines = self.install_machines();
        let (kept, stale) = self
            .bundles
            .drain(..)
            .partition(|b| machines.contains(&b.ecid));
        self.bundles = kept;
        stale
    }

    /// Whether the record names bundles or media of another build or
    /// generation than the active one: what an interrupted update left,
    /// which `update::prune` removes.
    pub(super) fn has_leftovers(&self) -> bool {
        let generation = self.generation();
        self.bundles
            .iter()
            .any(|b| b.build != self.build || b.generation() != generation)
            || self.media.iter().any(|m| m.build != self.build)
    }

    /// Every build directory the record or a bundle names.
    pub(super) fn build_directories(&self) -> Vec<PathBuf> {
        let mut builds = vec![&self.build];
        builds.extend(self.bundles.iter().map(|b| &b.build));
        let mut directories = Vec::new();
        for build in builds {
            let directory = bundle::build_directory_in(&self.volume, self.layout, build);
            if !directories.contains(&directory) {
                directories.push(directory);
            }
        }
        directories
    }
}

/// Delete the directories of bundles the record no longer names.
pub(super) fn remove_bundles(record: &Record, stale: &[bundle::Bundle]) -> Result<()> {
    for kept in stale {
        let directory = kept.directory_in(&record.volume, record.layout);
        if directory.is_dir() {
            fs::remove_dir_all(directory)?;
        }
    }
    Ok(())
}

/// Write the boot configuration and its compatibility files.
pub(super) fn publish(record: &mut Record) -> Result<()> {
    let image = fs::read(record.esp.join(LOADER_PATH))?;
    publish_with_image(record, &image)
}

/// Publish a complete startup configuration before putting its loader at
/// the removable-media entry point.
pub(super) fn publish_with_image(record: &mut Record, image: &[u8]) -> Result<()> {
    let layout = record.layout;
    efi::validate_mount_image(image, layout)?;
    super::volume_log::prepare(record)?;
    if record.bundles.iter().all(|bundle| bundle.processors == 0) {
        let generation = record.generation();
        for kept in &mut record.bundles {
            if kept.build == record.build
                && kept.generation() == generation
                && let Some(processors) = bundle::recorded_processors(
                    &kept.directory_in(&record.volume, record.layout),
                    &kept.manifest_sha256,
                )
            {
                kept.processors = processors;
                break;
            }
        }
    }
    record.options = record.options_text()?;
    let choices = record.choices()?;
    let choices_text = choices
        .as_ref()
        .map(Choices::encode)
        .transpose()
        .map_err(anyhow::Error::msg)?;
    let northwood = super::nwconfig_publish::prepare(
        &record.esp,
        image,
        &record.options,
        choices.as_ref(),
        layout,
    )?;
    // The existing mirrors are the merge baseline. Keep them intact until
    // the canonical config is written, so an interrupted publish can retry.
    if let Some(northwood) = northwood {
        northwood.write()?;
    }
    efi::prepare_mount(&record.esp, None, Some(&record.options), layout)?;
    let path = record.esp.join(layout.esp_path(layout::CHOICES));
    match choices_text {
        Some(text) => efi::write_atomic(&path, text.as_bytes(), false)?,
        None if path.exists() => fs::remove_file(&path)?,
        None => {}
    }
    efi::publish_loader(&record.esp, image)
}

/// The record on the ESP of `usb`, which must be a complete setup's.
pub(super) fn read_record(environment: &Environment<'_>, usb: &Path) -> Result<Record> {
    let esp = locate_esp(environment.host, usb)?;
    let layout = mutation_layout(&esp)?.unwrap_or(Layout::Legacy);
    let (record, current) = read_record_at(environment, usb, &esp, layout)?;
    check_mutation_volume(environment.host, &esp, &current, &record)?;
    Ok(record)
}

fn record_exists(esp: &Path, layout: Layout) -> Result<bool> {
    let path = esp.join(layout.esp_path(layout::RECORD));
    path.try_exists()
        .with_context(|| format!("cannot inspect {}", path.display()))
}

/// Read-only commands prefer the Northwood journal when both are present.
pub(super) fn layout_at(esp: &Path) -> Result<Option<Layout>> {
    for layout in Layout::PREFERRED {
        if record_exists(esp, layout)? {
            return Ok(Some(layout));
        }
    }
    Ok(None)
}

/// Writes need one unambiguous owner of the shared loader and configuration.
pub(super) fn mutation_layout(esp: &Path) -> Result<Option<Layout>> {
    let [preferred, fallback] = Layout::PREFERRED;
    let modern = record_exists(esp, preferred)?;
    let legacy = record_exists(esp, fallback)?;
    ensure!(
        !(modern && legacy),
        "{} holds both Northwood and Stela setup records, which share the boot loader. \
         Resolve the two records before changing the mounted installer",
        esp.display()
    );
    Ok(modern.then_some(preferred).or(legacy.then_some(fallback)))
}

/// The setup record for read-only commands, with its actual path and layout.
pub(super) fn read_inspection_record(
    environment: &Environment<'_>,
    usb: &Path,
) -> Result<(Record, Layout, PathBuf)> {
    let esp = locate_esp_with_record(environment.host, usb, |esp| Ok(layout_at(esp)?.is_some()))?;
    let layout = layout_at(&esp)?.unwrap_or(Layout::Legacy);
    let path = esp.join(layout.esp_path(layout::RECORD));
    let (record, _) = read_record_at(environment, usb, &esp, layout)?;
    Ok((record, layout, path))
}

fn read_record_at(
    environment: &Environment<'_>,
    usb: &Path,
    esp: &Path,
    layout: Layout,
) -> Result<(Record, PathBuf)> {
    let path = esp.join(layout.esp_path(layout::RECORD));
    ensure!(
        path.is_file(),
        "{} has no setup; run `stela setup` first",
        esp.display()
    );
    let header: serde_json::Value = stela_engine::read_json(&path)?;
    let schema = header["schema_version"].as_u64().unwrap_or(0);
    ensure!(
        header["kind"] == RECORD_KIND && schema <= u64::from(RECORD_SCHEMA),
        "{} is not a setup record this version understands; it may come from a newer stela",
        path.display()
    );
    ensure!(
        schema == u64::from(RECORD_SCHEMA),
        "{} is a schema {schema} setup, which this stela no longer reads: {}. Run `stela setup \
         uninstall --usb {}` and set the USB up again",
        path.display(),
        match schema {
            8 => "it keeps its files under EFI/ on the data partition",
            _ => "it booted a recoveryOS bundle from the USB",
        },
        usb.display()
    );
    let mut value = header;
    let current = current_volume(environment.host, esp, usb);
    rebase(&mut value, esp, &current);
    let mut record: Record = serde_json::from_value(value)
        .with_context(|| format!("cannot read setup record {}", path.display()))?;
    ensure!(
        !record.installs.is_empty(),
        "{} names no install; run `stela setup uninstall --usb {}` and set the USB up again",
        path.display(),
        usb.display()
    );
    record.set_layout(layout);
    Ok((record, current))
}

/// Where the data partition of the USB whose EFI partition is `esp` is
/// mounted now: `usb` when it names it, else the FAT32 data partition beside
/// the EFI partition, else the EFI partition itself, as for a setup with one
/// partition.
pub(super) fn current_volume(host: &dyn Host, esp: &Path, usb: &Path) -> PathBuf {
    if usb != esp {
        return usb.to_path_buf();
    }
    medium::locate(host, esp, None).map_or_else(|_| esp.to_path_buf(), |(volume, _)| volume)
}

/// A record can name a path from an earlier mount. Mutations may use that
/// data only after it has been rebased to a mounted partition of this USB.
fn check_mutation_volume(
    host: &dyn Host,
    esp: &Path,
    current: &Path,
    record: &Record,
) -> Result<()> {
    let data = &record.volume;
    if data != esp {
        let esp_disk = host.volume(esp)?.disk;
        let mounted = if data == current {
            true
        } else if current == esp {
            // The data mount can still be usable when its GPT GUID is
            // unavailable. Require the exact recorded mount on this USB.
            host.disks()?.iter().any(|disk| {
                Some(&disk.identifier) == esp_disk.as_ref()
                    && disk.partitions.iter().any(|partition| {
                        partition.content != "EFI" && partition.mount.as_ref() == Some(data)
                    })
            })
        } else {
            false
        };
        ensure!(
            mounted,
            "mount the setup's data partition beside {} before changing the installer",
            esp.display()
        );
        let volume = host.volume(data)?;
        ensure!(
            esp_disk.is_some() && volume.disk == esp_disk,
            "{} is not mounted on the same disk as {}",
            data.display(),
            esp.display()
        );
        ensure!(
            volume.filesystem == FileSystem::Fat,
            "{} is not a FAT32 setup data partition",
            data.display()
        );
    }
    ensure!(
        record
            .media
            .iter()
            .chain(record.medium.as_ref())
            .all(|medium| medium.volume == *data)
            && record
                .translations
                .as_ref()
                .is_none_or(|store| store.volume == *data),
        "the setup record names files outside its data partition {}",
        data.display()
    );
    Ok(())
}

/// Move every path of the record `value` to where the USB is mounted now:
/// `esp`, and the data partition at `volume` ([`current_volume`]). The record
/// holds the paths of the computer and user that wrote it, and another user,
/// computer or system mounts the same USB somewhere else.
fn rebase(value: &mut serde_json::Value, esp: &Path, volume: &Path) {
    let text = |key: &str| value[key].as_str().map(str::to_owned);
    let (Some(old_esp), Some(old_volume)) = (
        text("esp"),
        text("volume").or_else(|| text("payload_volume")),
    ) else {
        return;
    };
    // A setup with one partition keeps everything on its EFI partition. With
    // two, the data partition's paths stay as they were only when no data
    // partition is mounted now.
    let mut moves = Vec::new();
    if old_volume != old_esp && volume != esp {
        moves.push((old_volume, volume));
    }
    moves.push((old_esp, esp));
    fn walk(value: &mut serde_json::Value, moves: &[(String, &Path)]) {
        match value {
            serde_json::Value::String(text) => {
                for (old, new) in moves {
                    let Some(rest) = text.strip_prefix(old.as_str()) else {
                        continue;
                    };
                    // The whole old path, not another one that starts alike.
                    let separator = |c: char| c == '/' || c == '\\';
                    if !(rest.is_empty() || rest.starts_with(separator) || old.ends_with(separator))
                    {
                        continue;
                    }
                    let mut path = new.to_path_buf();
                    path.extend(rest.split(separator).filter(|part| !part.is_empty()));
                    *text = path.to_string_lossy().into_owned();
                    break;
                }
            }
            serde_json::Value::Array(values) => values.iter_mut().for_each(|v| walk(v, moves)),
            serde_json::Value::Object(fields) => {
                for (key, value) in fields {
                    // The boot options name UEFI paths, which do not move.
                    if key != "options" {
                        walk(value, moves);
                    }
                }
            }
            _ => {}
        }
    }
    walk(value, &moves);
}

/// Any schema's record, for `uninstall`: the bundles of an older
/// schema's record that name no kind or `recovery` are recoveryOS bundles,
/// which this `stela` has no kind for; their build directories are removed
/// with the rest. Before schema 9 the data partition's files were under
/// `EFI/`, which the record's volumes then gain, so every directory derived
/// from them is where that setup wrote it. Its paths follow the USB to where
/// it is mounted now ([`rebase`]): the EFI partition `esp`, and the data
/// partition `usb` names or beside it.
pub(super) fn read_any_record(host: &dyn Host, esp: &Path, usb: &Path) -> Result<Record> {
    let layout = mutation_layout(esp)?.unwrap_or(Layout::Legacy);
    let path = &esp.join(layout.esp_path(layout::RECORD));
    let mut value: serde_json::Value = stela_engine::read_json(path)?;
    let schema = value["schema_version"].as_u64().unwrap_or(0);
    ensure!(
        value["kind"] == RECORD_KIND && (1..=u64::from(RECORD_SCHEMA)).contains(&schema),
        "{} is not a setup record this version understands",
        path.display()
    );
    let current = current_volume(host, esp, usb);
    rebase(&mut value, esp, &current);
    if schema < 8
        && let Some(bundles) = value
            .get_mut("bundles")
            .and_then(serde_json::Value::as_array_mut)
    {
        bundles.retain(|bundle| {
            bundle
                .get("kind")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|kind| kind != "recovery")
        });
    }
    let mut record: Record = serde_json::from_value(value)
        .with_context(|| format!("{} cannot be read", path.display()))?;
    check_mutation_volume(host, esp, &current, &record)?;
    record.migrate();
    if schema < 9 {
        record.volume.push("EFI");
        for medium in record.media.iter_mut().chain(record.medium.as_mut()) {
            medium.volume.push("EFI");
        }
        if let Some(store) = &mut record.translations {
            store.volume.push("EFI");
        }
    }
    record.set_layout(layout);
    Ok(record)
}

/// The record on the EFI partition `esp` when it is this schema's and its
/// first setup was interrupted (`complete` is false): what `create::plan`
/// resumes. Its paths follow the USB to where it is mounted now, as
/// [`read_record`]'s do: a USB taken out part way is often mounted somewhere
/// else when setup runs again.
pub(super) fn interrupted(host: &dyn Host, esp: &Path, usb: &Path) -> Option<Record> {
    let layout = mutation_layout(esp).ok()??;
    let mut value: serde_json::Value =
        stela_engine::read_json(&esp.join(layout.esp_path(layout::RECORD))).ok()?;
    let current = current_volume(host, esp, usb);
    rebase(&mut value, esp, &current);
    let mut record = serde_json::from_value::<Record>(value)
        .ok()
        .filter(|record| {
            record.kind == RECORD_KIND && record.schema_version == RECORD_SCHEMA && !record.complete
        })?;
    check_mutation_volume(host, esp, &current, &record).ok()?;
    record.set_layout(layout);
    Some(record)
}

/// Replace the record through a synced sibling file, so a stop at any point
/// leaves the previous record or this one whole: the record is the journal
/// every interrupted setup and update is resumed from.
pub(super) fn write_record(esp: &Path, record: &Record) -> Result<()> {
    let current = mutation_layout(esp)?;
    ensure!(
        current.is_none_or(|current| current == record.layout),
        "{} already has a setup record in another layout",
        esp.display()
    );
    let path = esp.join(record.layout.esp_path(layout::RECORD));
    fs::create_dir_all(path.parent().unwrap())?;
    let mut record = record.clone();
    record.schema_version = RECORD_SCHEMA;
    let mut bytes = serde_json::to_vec_pretty(&record)?;
    bytes.push(b'\n');
    efi::write_atomic(&path, &bytes, current.is_none())
}

#[cfg(test)]
mod rebase_tests {
    use super::*;

    #[test]
    fn a_record_follows_the_usb_to_where_it_is_mounted_now() {
        let mut value = serde_json::json!({
            "esp": "/run/media/root/EFI",
            "volume": "/run/media/root/STELA",
            "bundle_directory": "/run/media/root/STELA/stela/boot/27A1",
            "created_directories": ["/run/media/root/STELA/stela", "/run/media/root/STELATE"],
            "options": "--bundle \\stela\\boot /run/media/root/STELA",
            "media": [{"volume": "/run/media/root/STELA"}],
            "translations": {"volume": "/run/media/root/STELA", "pack_bytes": 1},
        });
        rebase(
            &mut value,
            Path::new("/media/user/EFI"),
            Path::new("/media/user/STELA1"),
        );
        assert_eq!(value["esp"], "/media/user/EFI");
        assert_eq!(value["volume"], "/media/user/STELA1");
        assert_eq!(
            value["bundle_directory"],
            "/media/user/STELA1/stela/boot/27A1"
        );
        assert_eq!(value["created_directories"][0], "/media/user/STELA1/stela");
        // Another path that only starts alike, and the boot options, stay.
        assert_eq!(value["created_directories"][1], "/run/media/root/STELATE");
        assert_eq!(
            value["options"],
            "--bundle \\stela\\boot /run/media/root/STELA"
        );
        assert_eq!(value["media"][0]["volume"], "/media/user/STELA1");
        assert_eq!(value["translations"]["volume"], "/media/user/STELA1");
        // A record written on Windows, read where `/` separates.
        let mut value = serde_json::json!({
            "esp": "E:\\",
            "volume": "F:\\",
            "bundle_directory": "F:\\stela\\boot\\27A1",
        });
        rebase(&mut value, Path::new("/mnt/efi"), Path::new("/mnt/data"));
        assert_eq!(value["esp"], "/mnt/efi");
        assert_eq!(
            Path::new(value["bundle_directory"].as_str().unwrap()),
            Path::new("/mnt/data/stela/boot/27A1")
        );
        // One partition: everything follows the EFI partition.
        let mut value = serde_json::json!({
            "esp": "/a",
            "volume": "/a",
            "bundle_directory": "/a/stela/boot/27A1",
        });
        rebase(&mut value, Path::new("/b"), Path::new("/b"));
        assert_eq!(value["volume"], "/b");
        assert_eq!(value["bundle_directory"], "/b/stela/boot/27A1");
    }
}
