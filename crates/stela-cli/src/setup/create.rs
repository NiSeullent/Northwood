//! Creating a setup: the plan of what it writes, its description, and
//! writing it.
use super::{
    CHOOSER_SECONDS, Environment, INSTALL_CHOICE, InstallImage, LOADER_PATH, MARGIN, Request,
    assets,
    bundle::{self, Kind},
    ecid::Ecid,
    firmware, format,
    host::{self, Host},
    image_is_remote, installed, locate_esp, machines, medium,
    prepare::{
        Prepared, PreparedSetup, copy_bundles, download_dir, prepare_kind, prepare_restore_bundles,
        supported,
    },
    read_efi,
    record::{RECORD_KIND, RECORD_SCHEMA, Record, write_record},
    translations, update, volume_log, volume_swap, wifi, wizard,
};
use crate::{efi, restore};
use anyhow::{Context, Result, bail, ensure};
use std::{
    fmt::Write as _,
    fs,
    io::IsTerminal as _,
    path::{Path, PathBuf},
};
use stela_formats::builds;
use stela_loader::mount_interface::{
    VOLUME_LOG_BYTES, VOLUME_LOG_SLOTS,
    layout::{self, Layout},
};
use stela_runtime::pc::swap::{DEFAULT_VOLUME_MAX_BYTES, MIN_STORE_BYTES};

/// The setup USB's one unmounted partition besides its EFI partition: a
/// raw partition, which becomes the Mac's internal SSD. `None` when the USB
/// has none; several are refused, since setup cannot tell them apart.
pub(super) fn stick_ssd(host: &dyn Host, usb_disk: Option<&str>) -> Result<Option<String>> {
    let Some(usb_disk) = usb_disk else {
        return Ok(None);
    };
    let disks = host.disks()?;
    let Some(disk) = disks.iter().find(|disk| disk.identifier == usb_disk) else {
        return Ok(None);
    };
    let raw: Vec<&str> = disk
        .partitions
        .iter()
        .filter(|p| p.content != "EFI" && p.mount.is_none())
        .map(|p| p.identifier.as_str())
        .collect();
    match raw[..] {
        [] => Ok(None),
        [partition] => Ok(Some(partition.into())),
        _ => bail!(
            "{usb_disk} has several unmounted partitions ({}); name the Mac's internal SSD with \
             --target",
            raw.join(", ")
        ),
    }
}

/// The install medium setup writes.
pub(super) struct MediumPlan {
    pub(super) volume: PathBuf,
    pub(super) guid: String,
    /// It names the IPSW on Apple's servers (`medium::write_remote`).
    pub(super) remote: bool,
    pub(super) ipsw_bytes: u64,
    /// Bytes still to write, with the hash files and a margin.
    pub(super) need: u64,
    pub(super) expected: medium::Expected,
    /// Directories it creates, deepest first.
    pub(super) created: Vec<PathBuf>,
}

/// What installing macOS needs: each machine's restore bundles and, unless
/// the USB holds the build's intact medium, the install medium.
pub(super) struct InstallPlan {
    /// The macOS build it installs.
    pub(super) build: String,
    pub(super) restore: Vec<Prepared>,
    pub(super) medium: Option<MediumPlan>,
}

/// The restore bundles of both kinds for each of `ecids`.
pub(super) fn restore_bundles(ecids: &[Option<Ecid>]) -> Vec<(Kind, Option<Ecid>)> {
    bundle::RESTORE_KINDS
        .into_iter()
        .flat_map(|kind| ecids.iter().map(move |&ecid| (kind, ecid)))
        .collect()
}

/// Restore image, selected machine bundles and additional volume bytes.
pub(super) struct InstallOptions<'a> {
    pub(super) build: &'a str,
    pub(super) ipsw: &'a Path,
    pub(super) bundles: &'a [(Kind, Option<Ecid>)],
    pub(super) additional_bytes: u64,
}

/// Plan the install of `build` from `ipsw` for `record`: the restore
/// bundles `bundles` names and the build's install medium; and check the
/// data partition's space: `also` is what else setup writes to it.
pub(super) fn plan_install(
    environment: &Environment<'_>,
    record: &Record,
    options: InstallOptions<'_>,
) -> Result<InstallPlan> {
    let InstallOptions {
        build,
        ipsw,
        bundles,
        additional_bytes: also,
    } = options;
    let host = environment.host;
    let mut restore = Vec::new();
    for kind in bundle::RESTORE_KINDS {
        let ecids: Vec<Option<Ecid>> = bundles
            .iter()
            .filter(|(k, _)| *k == kind)
            .map(|&(_, ecid)| ecid)
            .collect();
        if !ecids.is_empty() {
            let (_, _, prepared) = prepare_kind(
                environment,
                super::prepare::BundleOptions {
                    kind,
                    ipsw,
                    model: &record.model,
                    build: Some(build),
                    ecids: &ecids,
                },
            )?;
            restore.extend(prepared);
        }
    }
    let remote = environment.download.is_none() && record.remote_install(ipsw)?;
    let intact = record
        .medium_of(build)
        .is_some_and(|m| m.remote == remote && m.problems().is_empty());
    let medium = match intact {
        true => None,
        false if remote => Some(MediumPlan {
            volume: record.volume.clone(),
            guid: String::new(),
            remote,
            ipsw_bytes: 0,
            need: MARGIN,
            expected: medium::Expected::default(),
            created: missing_directories(
                &record.volume,
                &medium::directory_in(&record.volume, record.layout, build),
            ),
        }),
        false => {
            let receipt = restore::receipt(ipsw)?;
            let (volume, guid) = medium::locate(host, &record.esp, Some(&record.volume))?;
            let ipsw_bytes = match environment.download {
                Some(download) => download.bytes,
                None => fs::metadata(ipsw)?.len(),
            };
            let need = medium::needed_bytes_in(
                &volume,
                record.layout,
                build,
                ipsw_bytes,
                environment.piece_bytes,
            );
            let created = missing_directories(
                &volume,
                &medium::directory_in(&volume, record.layout, build),
            );
            Some(MediumPlan {
                volume,
                guid,
                remote,
                ipsw_bytes,
                need,
                expected: match environment.download {
                    Some(download) => medium::Expected {
                        sha256: None,
                        sha1: download.image.sha1.clone(),
                    },
                    None => medium::Expected::of(receipt.as_ref()),
                },
                created,
            })
        }
    };
    // The medium and the bundles share the data partition.
    let restore_bytes: u64 = restore.iter().map(|p| p.bytes).sum();
    let need = medium.as_ref().map_or(MARGIN, |plan| plan.need) + restore_bytes + also;
    let free = host.volume(&record.volume)?.free_bytes;
    ensure!(
        free >= need,
        "{} has {} free; the install medium and boot bundles need {}{}. Free space on it, or \
         use a larger USB.",
        record.volume.display(),
        host::bytes_text(free),
        host::bytes_text(need),
        match &medium {
            Some(plan) if !plan.remote => {
                format!(
                    " (the macOS {build} IPSW is {})",
                    host::bytes_text(plan.ipsw_bytes)
                )
            }
            _ => String::new(),
        }
    );
    Ok(InstallPlan {
        build: build.into(),
        restore,
        medium,
    })
}

/// The plan's lines, after their two spaces.
pub(super) fn describe_install(
    record: &Record,
    plan: &InstallPlan,
    text: &mut String,
) -> Result<()> {
    for bundle in &plan.restore {
        writeln!(
            text,
            "  {:<14} {} ({})",
            bundle.kind.label(),
            bundle::directory_of_in(
                &record.volume,
                record.layout,
                bundle::Identity {
                    kind: bundle.kind,
                    build: &bundle.build,
                    ecid: bundle.ecid,
                    generation: record.generation()
                }
            )
            .display(),
            host::bytes_text(bundle.bytes)
        )?;
    }
    if let Some(medium) = plan.medium.as_ref().filter(|m| m.remote) {
        writeln!(
            text,
            "  install image  {}: file list only; Install macOS downloads macOS {} from Apple \
             over the PC's network (Ethernet, or the Wi-Fi network setup names)",
            medium::directory_in(&medium.volume, record.layout, &plan.build).display(),
            plan.build,
        )?;
    } else if let Some(medium) = &plan.medium {
        writeln!(
            text,
            "  install medium {}: the macOS {} IPSW in pieces on partition {} ({} to write)",
            medium::directory_in(&medium.volume, record.layout, &plan.build).display(),
            plan.build,
            medium.guid,
            host::bytes_text(medium.need)
        )?;
        if medium.expected.sha1.is_none() {
            writeln!(
                text,
                "  note           no download receipt: the IPSW is checked against its \
                 BuildManifest only"
            )?;
        }
    }
    Ok(())
}

/// Copy the restore bundles and write the install medium.
pub(super) fn apply_install(
    environment: &Environment<'_>,
    record: &mut Record,
    ipsw: &Path,
    plan: InstallPlan,
) -> Result<()> {
    copy_bundles(record, &plan.restore)?;
    write_record(&record.esp.clone(), record)?;
    let build = plan.build;
    let Some(plan) = plan.medium else {
        return Ok(());
    };
    // A medium of the other kind or volume this one replaces, removed once
    // this one is written.
    let previous = record.medium_of(&build).cloned();
    // Recorded before the first piece, so an interrupted copy is undone.
    record.media.retain(|m| m.build != build);
    record.media.push(medium::Medium {
        layout: record.layout,
        build: build.clone(),
        volume: plan.volume.clone(),
        partition_guid: plan.guid.clone(),
        manifest_sha256: String::new(),
        remote: plan.remote,
    });
    let mut created = record.created_directories.clone();
    for directory in plan.created {
        if !created.contains(&directory) {
            created.push(directory);
        }
    }
    created.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
    record.created_directories = created;
    write_record(&record.esp.clone(), record)?;
    if plan.remote {
        let progress = crate::progress::Phase::spinner("Preparing the network install inventory");
        let inventory = (environment.remote_inventory)(ipsw, &record.model)?;
        progress.finish();
        ensure!(
            inventory.manifest.product_build == build,
            "Apple's servers hold macOS build {}, not {build}",
            inventory.manifest.product_build,
        );
        let sha256 = medium::write_remote_in(&plan.volume, record.layout, inventory)?;
        if let Some(medium) = record.media.iter_mut().find(|m| m.build == build) {
            medium.manifest_sha256 = sha256;
        }
        write_record(&record.esp.clone(), record)?;
        return remove_replaced(record, &build, previous.as_ref());
    }
    if let Some(download) = environment.download {
        let written = download.write_in(
            &plan.volume,
            record.layout,
            &record.model,
            environment.piece_bytes,
        )?;
        if let Some(medium) = record.media.iter_mut().find(|m| m.build == build) {
            medium.manifest_sha256 = written.manifest_sha256;
        }
        write_record(&record.esp.clone(), record)?;
        return remove_replaced(record, &build, previous.as_ref());
    }
    // The inventory shows the restore image's hashing progress.
    let inventory = (environment.inventory)(ipsw, &record.model)?;
    ensure!(
        inventory.manifest.product_build == build,
        "{} holds macOS build {}, not {build}",
        ipsw.display(),
        inventory.manifest.product_build,
    );
    let written = medium::write(medium::WriteOptions {
        layout: record.layout,
        volume: &plan.volume,
        ipsw,
        inventory,
        expected: &plan.expected,
        piece_bytes: environment.piece_bytes,
    })?;
    if let Some(medium) = record.media.iter_mut().find(|m| m.build == build) {
        medium.manifest_sha256 = written.manifest_sha256;
    }
    write_record(&record.esp.clone(), record)?;
    remove_replaced(record, &build, previous.as_ref())
}

/// Remove what the medium of `build` that the one just written replaced
/// left (`medium::remove_replaced`).
pub(super) fn remove_replaced(
    record: &Record,
    build: &str,
    previous: Option<&medium::Medium>,
) -> Result<()> {
    match (previous, record.medium_of(build)) {
        (Some(previous), Some(current)) => medium::remove_replaced(previous, current),
        _ => Ok(()),
    }
}

/// What a resumed first setup keeps of the interrupted one.
struct Resume {
    name: String,
    ecid: Ecid,
    /// Its restore bundles, when every one is intact; else empty, and
    /// they are prepared again.
    bundles: Vec<bundle::Bundle>,
    /// Its install medium of the build: `plan_install` keeps it when intact
    /// and `medium::write` keeps each piece whose hash matches.
    media: Vec<medium::Medium>,
    translations: Option<translations::Store>,
    created_directories: Vec<PathBuf>,
}

/// The setup a resume is checked against.
struct ResumeRequest<'a> {
    command: &'a Request,
    ipsw: &'a Path,
    disk: &'a installed::InstalledDisk,
    volume: &'a Path,
}

impl Resume {
    /// Refuse a resume of a setup of another build, model, target, data
    /// volume, install name or bundle generation, naming each difference.
    fn check(
        environment: &Environment<'_>,
        previous: &Record,
        request: ResumeRequest<'_>,
    ) -> Result<Self> {
        let ResumeRequest {
            command,
            ipsw,
            disk,
            volume,
        } = request;
        let [install] = &previous.installs[..] else {
            bail!(
                "the interrupted setup on {} names {} installs; remove it with {} first",
                previous.esp.display(),
                previous.installs.len(),
                super::fix(&command.usb, "uninstall")
            );
        };
        let mut differences = Vec::new();
        let (_, build) = restore::payload::identity(ipsw, command.model())?;
        if build != previous.build {
            differences.push(format!("macOS {} (this one {build})", previous.build));
        }
        if command.model() != previous.model {
            differences.push(format!(
                "model {} (this one {})",
                previous.model,
                command.model()
            ));
        }
        let same_disk = |a: &installed::InstalledDisk, b: &installed::InstalledDisk| {
            a.is_chosen() == b.is_chosen()
                && a.apple_region_guid
                    .eq_ignore_ascii_case(&b.apple_region_guid)
                && a.nor_guid.eq_ignore_ascii_case(&b.nor_guid)
                && a.machine_state_offset == b.machine_state_offset
        };
        if !same_disk(&install.disk, disk) {
            differences.push(format!(
                "target {} (this one {})",
                install.disk.partitions(),
                disk.partitions()
            ));
        }
        if previous.volume != volume {
            differences.push(format!(
                "data volume {} (this one {})",
                previous.volume.display(),
                volume.display()
            ));
        }
        if let Some(name) = command
            .install
            .as_ref()
            .filter(|name| **name != install.name)
        {
            differences.push(format!(
                "install \"{}\" (this one \"{name}\")",
                install.name
            ));
        }
        let generation = environment.generation();
        if previous.generation() != generation {
            differences.push(format!(
                "boot bundles {} (this stela prepares {generation})",
                previous.generation()
            ));
        }
        let Some(ecid) = install.ecid else {
            bail!(
                "the interrupted setup on {} names no ECID; remove it with {} first",
                previous.esp.display(),
                super::fix(&command.usb, "uninstall")
            );
        };
        if !differences.is_empty() {
            bail!(
                "{} holds an interrupted setup that differs from this one: {}. Finish it with {}, \
                 or remove it with {} first",
                previous.esp.display(),
                differences.join(", "),
                super::rerun(previous, &command.usb),
                super::fix(&command.usb, "uninstall")
            );
        }
        // Every restore bundle intact, or all prepared again.
        let intact = bundle::RESTORE_KINDS.into_iter().all(|kind| {
            previous
                .restore_bundle(kind, Some(ecid))
                .is_some_and(|kept| {
                    bundle::verify(
                        &kept.directory_in(&previous.volume, previous.layout),
                        &kept.manifest_sha256,
                    )
                    .is_empty()
                })
        });
        let bundles = match intact {
            true => previous
                .bundles
                .iter()
                .filter(|b| b.build == build && b.generation() == generation)
                .cloned()
                .collect(),
            false => Vec::new(),
        };
        Ok(Self {
            name: install.name.clone(),
            ecid,
            bundles,
            media: previous
                .media
                .iter()
                .filter(|m| m.build == build)
                .cloned()
                .collect(),
            translations: previous.translations.clone(),
            created_directories: previous.created_directories.clone(),
        })
    }
}

/// What an install sends to Apple's signing service; the plan says it
/// before setup writes the USB.
pub(super) const SIGNING_NOTE: &str = "Apple's signing service (gs.apple.com) receives this \
    machine's ECID, the restore's nonces and its component hashes, to sign the restore's root \
    tickets, its Cryptex1 ticket and the recoveryOS LocalPolicy: setup asks while it prepares \
    the boot files, and the install asks only for what the USB lacks.";

/// The full installer's promise, and its limit.
pub(super) const OFFLINE_NOTE: &str = "Full installer: macOS is on the USB, the install needs \
    no network.";
/// What is not known yet about the installed system without a network.
pub(super) const FIRST_START_NOTE: &str = "After the install, macOS may ask for a network the \
    first time it starts: whether Setup Assistant's activation finishes without one is not \
    tested yet.";

/// Whether the restore bundle in `directory` holds Apple's answers to every
/// signing request its install makes (`tss::staged`).
fn staged(directory: &Path) -> bool {
    fs::read(directory.join(stela_runtime::restore::tss::staged::FILE))
        .ok()
        .and_then(|bytes| stela_runtime::restore::tss::staged::Set::decode(&bytes).ok())
        .is_some_and(|set| set.complete())
}

/// What the plan says about the network: none needed for the install with
/// macOS and every signing answer on the USB, else why it is needed.
pub(super) fn network_notes(remote: bool, offline: bool) -> Vec<String> {
    match offline {
        true => vec![OFFLINE_NOTE.into(), FIRST_START_NOTE.into()],
        false => vec![stela_runtime::restore::tss::network_reason(remote).into()],
    }
}

/// The checks' result: everything setup will write, and where.
pub(super) struct Plan {
    pub(super) record: Record,
    pub(super) ipsw: PathBuf,
    pub(super) product_version: String,
    /// The prepared restore bundles, in a host directory until setup
    /// copies them.
    pub(super) prepared: Vec<Prepared>,
    pub(super) efi_image: Vec<u8>,
    /// The install's disk, and the NOR image setup writes to it.
    pub(super) installed: installed::Checked,
    /// The install medium.
    pub(super) install: InstallPlan,
    pub(super) notes: Vec<String>,
    /// Optional USB swap request that could not fit after installer reservations.
    swap_note: Option<String>,
    /// The key file's passphrase, for `--wifi`.
    pub(super) wifi_passphrase: Option<String>,
    /// The chips of this computer's NVIDIA GPUs, whose firmware setup adds.
    pub(super) gpu_chips: Vec<&'static str>,
}

pub(super) fn plan(
    environment: &Environment<'_>,
    command: &Request,
    ipsw: PathBuf,
    early: Option<PreparedSetup>,
    efi_image: Vec<u8>,
) -> Result<Plan> {
    let host = environment.host;
    let esp = locate_esp(host, &command.usb)?;
    // An interrupted first setup is resumed; any other is refused.
    let previous = match super::record::layout_at(&esp)? {
        None => None,
        Some(layout) => match super::record::interrupted(host, &esp, &command.usb) {
            Some(previous) => Some(previous),
            None => bail!(
                "{} is already set up ({}). Update it with {} or {}, or remove it with {} first",
                esp.display(),
                esp.join(layout.esp_path(layout::RECORD)).display(),
                super::fix(&command.usb, "update stela"),
                super::fix(&command.usb, "update os"),
                super::fix(&command.usb, "uninstall")
            ),
        },
    };
    let usb_disk = host.volume(&esp)?.disk;
    // Without --target, a raw partition of the USB is the Mac's internal
    // SSD, and the PC's disks its external drives.
    let stick = match &command.target {
        Some(_) => None,
        None => stick_ssd(host, usb_disk.as_deref())?,
    };
    let target = command.target.clone().or(stick.clone());
    // With no disk named here, Install macOS chooses the PC's disk in
    // stela.efi.
    let installed = match &target {
        None => installed::Checked {
            disk: installed::InstalledDisk::chosen(),
            nor_image: None,
            sep_provisioned: false,
            foreign: None,
            notes: Vec::new(),
        },
        Some(target) => installed::check(
            host,
            &installed::Request {
                apple_region: target,
                nor: command.nor.as_deref(),
                serial: command.disk_serial.as_deref(),
                erase_target: command.erase_target,
                initialize_nor: command.initialize_nor,
                usb_disk: usb_disk.as_deref(),
                external_disks: stick.is_some(),
            },
        )?,
    };
    let mut notes = Vec::new();
    // The bundles, the install medium and the translation store go on the
    // USB's FAT32 data partition, which the firmware reads.
    let (volume, volume_guid) = medium::locate(host, &esp, command.install_volume.as_deref())?;
    let resume = match &previous {
        Some(previous) => Some(Resume::check(
            environment,
            previous,
            ResumeRequest {
                command,
                ipsw: &ipsw,
                disk: &installed.disk,
                volume: &volume,
            },
        )?),
        None => None,
    };
    let ecid = machines::machine_ecid(
        machines::EcidOptions {
            given: command.ecid,
            recorded: resume.as_ref().map(|r| r.ecid),
            sep_provisioned: installed.sep_provisioned,
            new_ecid: command.new_ecid,
        },
        &mut notes,
    )?;
    let layout = previous
        .as_ref()
        .map_or_else(|| layout_for_image(&efi_image), |record| record.layout);
    update::check_firmware_support(&esp, layout, &efi_image)?;
    // The restore bundles are prepared on this computer before the USB
    // changes, which also measures them for the space check. A resumed
    // setup keeps its intact ones.
    let (product_version, build, prepared) = match (early, &resume) {
        (Some(prepared), _) => prepared,
        (None, Some(resume)) if !resume.bundles.is_empty() => {
            let (version, build) = restore::payload::identity(&ipsw, command.model())?;
            notes.push("the interrupted setup's boot files are intact and kept".into());
            (version, build, Vec::new())
        }
        (None, _) => prepare_restore_bundles(environment, &ipsw, command.model(), ecid)?,
    };
    let bundles: u64 = prepared.iter().map(|bundle| bundle.bytes).sum();
    let install = machines::Install {
        name: resume.as_ref().map_or_else(
            || {
                command
                    .install
                    .clone()
                    .unwrap_or_else(|| machines::FIRST_NAME.into())
            },
            |resume| resume.name.clone(),
        ),
        ecid: Some(ecid),
        disk: installed.disk.clone(),
    };
    machines::check_name(&install.name)?;
    let bundle_directory = bundle::build_directory_in(&volume, layout, &build);
    let mut record = Record {
        layout,
        schema_version: RECORD_SCHEMA,
        kind: RECORD_KIND.into(),
        build: build.clone(),
        model: command.model().into(),
        esp: esp.clone(),
        created_directories: missing_directories(&volume, &bundle_directory),
        bundle_directory,
        volume: volume.clone(),
        payload_directory: None,
        options: String::new(),
        complete: false,
        installed: None,
        stela_version: Some(update::version_id(&efi_image)),
        installs: vec![install],
        default_install: None,
        bundle_schema: environment.interface.bundle_schema,
        bundle_revision: environment.bundle_revision,
        bundles: Vec::new(),
        chooser_timeout: match command.chooser_timeout {
            Some(super::command::Wait::Seconds(seconds)) => Some(seconds),
            _ => None,
        },
        medium: None,
        media: Vec::new(),
        verify_on_read: command.verify_on_read,
        profile: command.profile,
        install_image: command.install_image,
        translations: None,
        volume_swap: previous
            .as_ref()
            .and_then(|previous| previous.volume_swap.clone()),
        wifi: command.wifi.clone().map(|ssid| wifi::Network {
            ssid,
            country: command.wifi_country.clone(),
        }),
        usb4_approved: command.usb4_approve.clone(),
        addresses: super::addresses::Addresses::requested(command)?,
    };
    // What the interrupted setup wrote and this one keeps: its intact
    // bundles, its medium (kept whole when intact, else piece by piece by
    // its hashes), its translation store and the directories it created.
    if let Some(resume) = resume {
        notes.push(format!(
            "resumes the setup interrupted on {}",
            esp.join(layout.esp_path(layout::RECORD)).display()
        ));
        record.bundles = resume.bundles;
        record.media = resume.media;
        record.translations = resume.translations;
        for directory in resume.created_directories {
            if !record.created_directories.contains(&directory) {
                record.created_directories.push(directory);
            }
        }
        record
            .created_directories
            .sort_by_key(|p| std::cmp::Reverse(p.components().count()));
    }
    // The restore bundles are prepared above: the install plan adds the
    // install medium.
    let install = plan_install(
        environment,
        &record,
        InstallOptions {
            build: &record.build,
            ipsw: &ipsw,
            bundles: &[],
            additional_bytes: bundles,
        },
    )?;
    // The translation store takes what the data partition has left after
    // everything above.
    let medium_bytes = install.medium.as_ref().map_or(0, |m| m.need);
    if record.translations.is_none() {
        let (store, note) = translations::choose_in(
            host,
            layout,
            &[record.volume.as_path()],
            |_| medium_bytes + bundles,
            command.translation_store_bytes,
        )?;
        notes.extend(note);
        record.translations = store;
    }
    let mut swap_note = None;
    if command.volume_swap_bytes.is_some() || record.volume_swap.is_some() {
        check_volume_swap_efi(&efi_image)?;
    }
    if let Some(store) = &record.volume_swap {
        ensure!(
            command
                .volume_swap_bytes
                .is_none_or(|bytes| bytes == store.requested_bytes),
            "the interrupted setup already requested {} bytes of USB swap",
            store.requested_bytes
        );
    } else if let Some(requested) = command.volume_swap_bytes {
        let cache_bytes = record.translations.as_ref().map_or(0, |store| {
            translations::needed_bytes(store.pack_bytes).saturating_sub(store.used_bytes())
        });
        let logs = (VOLUME_LOG_SLOTS.len() - volume_log::owned_slots(&record).len()) as u64
            * VOLUME_LOG_BYTES as u64;
        let reserved = medium_bytes + bundles + cache_bytes + logs;
        let available = host
            .volume(&record.volume)?
            .free_bytes
            .saturating_sub(reserved);
        let policy = super::nwconfig_publish::status(&record.esp)?
            .as_ref()
            .map(super::nwconfig_publish::default_swap_policy)
            .unwrap_or_default();
        ensure!(
            policy.enabled
                && matches!(
                    policy.location,
                    northwood_config::SwapLocation::Auto
                        | northwood_config::SwapLocation::VolumeFile
                ),
            "the default boot entry does not enable USB volume swap"
        );
        record.volume_swap = volume_swap::Store::plan(
            requested,
            policy.volume_max_bytes,
            available,
            super::swap::random_guid()?,
            volume_guid.parse().map_err(anyhow::Error::msg)?,
        )?;
        if record.volume_swap.is_none() {
            swap_note = Some(format!(
                "USB swap requested {}. Only {} is available after reserving installer files, the translation store and logs.",
                host::bytes_text(requested),
                host::bytes_text(available)
            ));
        }
    }
    if let Some(store) = &record.volume_swap {
        store.validate_volume(host, &record)?;
        store.repair_plan(&record.volume, record.layout)?;
    }
    record.options = record.options_text()?;
    record.choices_text()?;
    // Said up front: whether the install needs the network, and what is
    // not known about the installed system's first start.
    let remote = install.medium.as_ref().is_some_and(|medium| medium.remote);
    let restore_bundles: Vec<PathBuf> = prepared
        .iter()
        .filter(|bundle| bundle.kind == Kind::Restore)
        .map(Prepared::path)
        .chain(
            record
                .bundles
                .iter()
                .filter(|bundle| bundle.kind == Kind::Restore)
                .map(|bundle| bundle.directory_in(&record.volume, layout)),
        )
        .collect();
    let offline =
        !remote && !restore_bundles.is_empty() && restore_bundles.iter().all(|b| staged(b));
    notes.extend(network_notes(remote, offline));
    notes.push(SIGNING_NOTE.into());
    notes.extend(installed.notes.iter().cloned());
    // The firmware of this computer's GPU; a USB made elsewhere gets none,
    // which the PC's boot screen names, so say how to add it.
    let (gpu_chips, gpu_notes) = firmware::host_chips(environment.host)?;
    notes.extend(gpu_notes);
    if gpu_chips.is_empty() {
        notes.push(firmware::missing(&super::fix(
            &command.usb,
            "host-firmware fetch --gpu CHIP",
        )));
    }
    if record.wifi.is_some() {
        notes.extend(firmware::wifi_note(environment.host)?);
    }
    notes.extend(firmware::bluetooth_notes(
        environment.host,
        &super::fix(&command.usb, "repair"),
    )?);
    Ok(Plan {
        record,
        ipsw,
        product_version,
        prepared,
        efi_image,
        installed,
        install,
        notes,
        swap_note,
        wifi_passphrase: command.wifi_passphrase.clone(),
        gpu_chips,
    })
}

pub(super) fn describe(plan: &Plan) -> Result<String> {
    let record = &plan.record;
    let mut text = String::new();
    writeln!(
        text,
        "Stela setup for macOS {} ({}) on {}:",
        plan.product_version, record.build, record.model
    )?;
    writeln!(text, "  restore image  {}", plan.ipsw.display())?;
    writeln!(text, "  EFI partition  {}", record.esp.display())?;
    if !plan.gpu_chips.is_empty() {
        writeln!(text, "  GPU firmware   {}", plan.gpu_chips.join(", "))?;
    }
    let install = &record.installs[0];
    writeln!(
        text,
        "  ECID           {}",
        install.ecid.expect("a new setup's install has an ECID")
    )?;
    for prepared in &plan.prepared {
        writeln!(
            text,
            "  {:<14} {} ({})",
            prepared.kind.label(),
            bundle::directory_of_in(
                &record.volume,
                record.layout,
                bundle::Identity {
                    kind: prepared.kind,
                    build: &prepared.build,
                    ecid: prepared.ecid,
                    generation: record.generation()
                }
            )
            .display(),
            host::bytes_text(prepared.bytes)
        )?;
    }
    if let Some(store) = &record.translations {
        writeln!(text, "  translations   {}", store.describe())?;
    }
    if let Some(store) = &record.volume_swap {
        writeln!(
            text,
            "  USB swap       {}",
            describe_volume_swap(record, store)
        )?;
    } else if let Some(note) = &plan.swap_note {
        writeln!(text, "  {note}")?;
    }
    writeln!(
        text,
        "  boot loader    {}: stela.efi",
        record.esp.join(LOADER_PATH).display()
    )?;
    let checked = &plan.installed;
    let disk = &checked.disk;
    if record.layout == Layout::Northwood {
        writeln!(
            text,
            "  nwconfig.plist {}",
            record
                .esp
                .join(record.layout.esp_path(layout::CONFIG))
                .display()
        )?;
    }
    writeln!(
        text,
        "  stela.cfg      {}",
        record
            .esp
            .join(record.layout.esp_path(layout::OPTIONS))
            .display()
    )?;
    writeln!(
        text,
        "  choices.cfg    {}: \"{}\", \"{INSTALL_CHOICE}\" ({CHOOSER_SECONDS} s menu)",
        record
            .esp
            .join(record.layout.esp_path(layout::CHOICES))
            .display(),
        install.name
    )?;
    describe_install(record, &plan.install, &mut text)?;
    if let Some(network) = &record.wifi {
        writeln!(
            text,
            "  {}",
            wifi::describe_in(network, &record.esp, record.layout)
        )?;
    }
    for line in record.addresses.describe() {
        writeln!(text, "  {line}")?;
    }
    match disk.is_chosen() {
        true => writeln!(
            text,
            "  Mac's SSD      a PC disk chosen at {INSTALL_CHOICE}"
        )?,
        false => writeln!(
            text,
            "  Mac's SSD      {} (serial {}, partition {})",
            disk.apple_region, disk.apple_region_serial, disk.apple_region_guid
        )?,
    }
    writeln!(text, "  {}", state_line(disk))?;
    match checked.nor_image {
        Some(_) => writeln!(
            text,
            "  writes         this USB, machine state on {}",
            disk.state_description()
        )?,
        None => writeln!(text, "  writes         this USB only")?,
    }
    for note in &plan.notes {
        writeln!(text, "  note           {note}")?;
    }
    Ok(text)
}

/// The plan's line for an install's machine state, after its two spaces.
pub(super) fn state_line(disk: &installed::InstalledDisk) -> String {
    format!("{:<14} {}", "machine state", disk.state_description())
}

pub(super) fn setup(environment: &Environment<'_>, mut command: Request) -> Result<String> {
    let host = environment.host;
    // Before anything is erased.
    if let Some(bytes) = command.volume_swap_bytes {
        ensure!(
            bytes >= MIN_STORE_BYTES,
            "USB swap needs at least {MIN_STORE_BYTES} bytes"
        );
    }
    command.read_wifi_passphrase()?;
    if command.fetch {
        builds::find_in(environment.builds, &command.build)?;
    }
    let mut erase = None;
    if let Some(disk) = host.whole_disk(&command.usb)? {
        let found = host.disks()?.into_iter().find(|d| d.identifier == disk);
        let found = found.with_context(|| format!("this computer lists no disk {disk}"))?;
        ensure!(
            found.erasable(),
            "{disk} ({}) is an internal disk; setup erases only a removable USB",
            found.name
        );
        ensure!(
            found.removal == host::Removal::Usb
                || command
                    .confirm_disk
                    .as_deref()
                    .is_some_and(|typed| found.confirmed_by(typed)),
            "{disk} ({}, {}) is not a USB disk; pass --confirm-disk with its model or size to \
             erase it",
            found.name,
            host::size_text(found.bytes)
        );
        if command.dry_run {
            let layout = match &command.efi {
                Some(path) => {
                    let image = read_efi(environment, path)?;
                    if command.volume_swap_bytes.is_some() {
                        check_volume_swap_efi(&image)?;
                    }
                    layout_for_image(&image)
                }
                None => Layout::Northwood,
            };
            let mut report = format!(
                "Setup would erase {disk} ({}) into an EFI partition and a FAT32 volume named \
                 {}, then set it up.\n",
                found.name,
                layout.volume_label(),
            );
            if let Some(bytes) = command.volume_swap_bytes {
                writeln!(
                    report,
                    "  USB swap       {} requested in {}, up to {} after reserving installer files",
                    host::bytes_text(bytes),
                    layout.data_path(layout::DATA_SWAP),
                    host::bytes_text(DEFAULT_VOLUME_MAX_BYTES)
                )?;
            }
            report.push_str("Dry run: nothing was changed.\n");
            return Ok(report);
        }
        host.check_erase_access(&disk)?;
        let identity = wizard::Usb::from_disk(&found, host);
        if !command.confirmed && std::io::stdin().is_terminal() {
            confirm_erase(host, &found)?;
        }
        // A new setup puts macOS on a USB large enough for it.
        command
            .install_image
            .get_or_insert(super::default_install_image(found.bytes));
        erase = Some((disk, found.name, found.bytes, identity));
    } else {
        // Inspect the mounted installer's layout before downloading any files.
        locate_esp(host, &command.usb)?;
    }
    // Downloaded boot archives are kept in one place until setup finishes,
    // so a setup run again after a stop reuses them, and removed once it
    // has. A full image goes directly to the USB after preparation and
    // signing; an explicit cache directory keeps what it holds.
    let temporary = (command.fetch && command.download_dir.is_none() && !command.dry_run)
        .then(|| download_dir().join("setup"));
    if let Some(temporary) = &temporary {
        fs::create_dir_all(temporary)?;
    }
    let mut download = None;
    let ipsw = if let Some(temporary) = &temporary
        && command.install_image == Some(InstallImage::Full)
    {
        let image = restore::select(
            &restore::catalog()?,
            Some(&command.build),
            Some(command.model()),
        )?;
        let boot = temporary.join("boot.ipsw");
        download = Some(medium::download::Download::prepare(
            image,
            command.model(),
            &boot,
        )?);
        boot
    } else {
        if let Some(temporary) = &temporary {
            command.download_dir = Some(temporary.clone());
        }
        command.image()?
    };
    if ipsw.is_file() {
        let (_, build) = restore::payload::identity(&ipsw, command.model())?;
        supported(environment, &ipsw, &build)?;
    }
    let environment = &Environment {
        download: download.as_ref(),
        ..*environment
    };
    if command.efi.is_none() {
        command.efi = Some(assets::resolve()?);
    }
    let efi_image = read_efi(environment, command.efi.as_ref().unwrap())?;
    if command.volume_swap_bytes.is_some() {
        check_volume_swap_efi(&efi_image)?;
    }
    let mut early = None;
    if let Some((disk, _, capacity, _)) = &erase {
        for path in std::iter::once(&ipsw).chain(command.efi.iter()) {
            ensure!(
                host.volume(path)?.disk.as_deref() != Some(disk),
                "{} is on the USB being erased; move installer inputs to another disk first",
                path.display()
            );
        }
        let remote = download.is_none() && image_is_remote(&ipsw, command.install_image)?;
        // The portable installer knows its identity before the target PC is
        // connected. Prepare and sign once, and measure it before erasing USB.
        if command.target.is_none() {
            let ecid = command.ecid.map_or_else(Ecid::generate, Ok)?;
            command.ecid = Some(ecid);
            let (version, build, prepared) =
                prepare_restore_bundles(environment, &ipsw, command.model(), ecid)?;
            let image_bytes = if remote {
                0
            } else {
                match &download {
                    Some(download) => download.bytes,
                    None => fs::metadata(&ipsw)?.len(),
                }
            };
            let required = prepared.iter().map(|p| p.bytes).sum::<u64>()
                + image_bytes
                + format::ESP_BYTES
                + 4 * MARGIN;
            ensure!(
                *capacity == 0 || *capacity >= required,
                "this installer needs a USB of at least {} (this one holds {}): use a larger USB",
                host::bytes_text(required),
                host::bytes_text(*capacity)
            );
            early = Some((version, build, prepared));
        }
    }
    if let Some((disk, name, _, identity)) = erase {
        identity.verify(host)?;
        eprintln!("Erasing {disk} ({name})");
        host.erase(&disk, layout_for_image(&efi_image).volume_label())?;
        let volume = host
            .disks()?
            .into_iter()
            .find(|d| d.identifier == disk)
            .and_then(|d| d.partitions.into_iter().find_map(|p| p.mount))
            .with_context(|| format!("{disk} has no mounted volume after it was erased"))?;
        command.usb = volume;
    }
    let plan = plan(environment, &command, ipsw, early, efi_image)?;
    let mut report = describe(&plan)?;
    if command.dry_run {
        report.push_str("Dry run: nothing was changed.\n");
        return Ok(report);
    }
    let disk = plan.installed.disk.clone();
    let volumes = [plan.record.esp.clone(), plan.record.volume.clone()];
    apply(environment, plan)?;
    // Before "Done": a USB taken out right after holds everything.
    for volume in &volumes {
        host.flush(volume)?;
    }
    if let Some(temporary) = &temporary {
        // The boot files are on the USB now.
        let _ = fs::remove_dir_all(temporary);
    }
    let start = match disk {
        disk if disk.is_chosen() => format!(
            "choose \"{INSTALL_CHOICE}\" in Stela's startup choices, pick the PC disk and the \
             space macOS gets, and the install restores macOS onto it; Stela then starts the \
             installed macOS from that disk"
        ),
        disk if disk.external_disks => format!(
            "its internal SSD ({}) is blank, so Stela offers \"{INSTALL_CHOICE}\", which \
             restores macOS onto it; Stela then starts that macOS from the USB, and macOS \
             sees the PC's other disks as external drives",
            disk.apple_region
        ),
        disk => format!(
            "choose \"{INSTALL_CHOICE}\" in Stela's startup choices to install macOS onto \
             {target}, then boot from the USB again, which starts the installed macOS",
            target = disk.apple_region
        ),
    };
    writeln!(
        report,
        "Done. Eject the USB before you unplug it. Boot the PC from this USB (the firmware's \
         boot menu lists it as a UEFI USB device): {start}.\nCheck it with {}\nUndo it with {}",
        super::fix(&command.usb, "status"),
        super::fix(&command.usb, "uninstall")
    )?;
    Ok(report)
}

/// Ask before erasing `disk` at a terminal: a mistyped disk name would
/// erase another removable disk.
fn confirm_erase(host: &dyn Host, disk: &host::Disk) -> Result<()> {
    let serial = host
        .serial(&disk.identifier)
        .ok()
        .flatten()
        .map(|serial| format!(", serial {serial}"))
        .unwrap_or_default();
    eprint!(
        "Erase {} ({}, {}{serial})? Everything on it is deleted. [y/N] ",
        disk.identifier,
        disk.name,
        host::size_text(disk.bytes)
    );
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    ensure!(
        matches!(answer.trim(), "y" | "Y" | "yes" | "Yes" | "YES"),
        "{} was not erased; setup stopped",
        disk.identifier
    );
    Ok(())
}

/// Explicit upstream loaders still require the legacy layout.
fn layout_for_image(image: &[u8]) -> Layout {
    if efi::supports_northwood_layout(image) {
        Layout::Northwood
    } else {
        Layout::Legacy
    }
}

fn check_volume_swap_efi(image: &[u8]) -> Result<()> {
    ensure!(
        efi::supports_northwood_layout(image),
        "USB swap needs a current Northwood EFI"
    );
    Ok(())
}

fn describe_volume_swap(record: &Record, store: &volume_swap::Store) -> String {
    format!(
        "{} ({} of {} requested)",
        store.directory(&record.volume, record.layout).display(),
        host::bytes_text(store.bytes),
        host::bytes_text(store.requested_bytes)
    )
}

/// Directories from `root` down to `path` that do not exist yet, deepest
/// first.
pub(super) fn missing_directories(root: &Path, path: &Path) -> Vec<PathBuf> {
    let mut missing: Vec<PathBuf> = path
        .ancestors()
        .take_while(|p| p.starts_with(root) && *p != root)
        .filter(|p| !p.exists())
        .map(Path::to_path_buf)
        .collect();
    missing.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
    missing
}

pub(super) fn apply(environment: &Environment<'_>, plan: Plan) -> Result<()> {
    let mut record = plan.record;
    let esp = record.esp.clone();
    // Recorded before the first write, so an interrupted setup can be undone.
    write_record(&esp, &record)?;
    copy_bundles(&mut record, &plan.prepared)?;
    write_record(&esp, &record)?;
    if let Some(store) = &record.translations {
        store.publish()?;
    }
    apply_install(environment, &mut record, &plan.ipsw, plan.install)?;
    if let Some(store) = &record.volume_swap {
        store.validate_volume(environment.host, &record)?;
        volume_log::prepare(&record)?;
        eprintln!(
            "Preparing USB swap at {}",
            describe_volume_swap(&record, store)
        );
        store.publish(&record.volume, record.layout)?;
    } else if let Some(note) = &plan.swap_note {
        eprintln!("{note}");
    }
    let checked = &plan.installed;
    if let Some(region) = &checked.foreign {
        installed::erase_foreign(environment.host, region)?;
    }
    if let Some(nor) = &checked.nor_image {
        installed::initialize_nor(environment.host, &checked.disk, nor)?;
    }
    if let Some(passphrase) = &plan.wifi_passphrase {
        wifi::write_key_in(&esp, passphrase, record.layout)?;
    }
    super::record::publish_with_image(&mut record, &plan.efi_image)?;
    record.complete = true;
    write_record(&esp, &record)?;
    for chip in plan.gpu_chips {
        let (destination, installed) =
            firmware::publication_paths(&esp, record.layout, &plan.efi_image)?;
        firmware::provision_from(
            firmware::ProvisionOptions {
                requirements: None,
                gpu: Some(chip),
                wifi: None,
                profile: None,
                offline: false,
                destination: &destination,
                cache: &super::cache_dir().join("firmware"),
                dry_run: false,
                mount_image: &plan.efi_image,
            },
            &installed,
        )
        .with_context(|| {
            format!(
                "the USB is set up, but without {chip}'s GPU firmware, which macOS needs: add \
                 it with {}",
                super::fix(&esp, &format!("host-firmware fetch --gpu {chip}"))
            )
        })?;
    }
    if record.wifi.is_some() {
        firmware::provision_wifi_in(environment.host, &esp, record.layout, &plan.efi_image)?;
    }
    Ok(())
}
