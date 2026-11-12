//! `status` and `repair`.
//!
//! Status reads everything a setup consists of and changes nothing: the
//! record, the installed `stela.efi` against the recorded version and this
//! `stela`'s mount interface, the USB's boot loader,
//! `stela.cfg` and the startup choices against the record, every restore
//! bundle file against its `bundle.json` and the active ones'
//! revision against the one this `stela` prepares, the install medium's
//! manifest and pieces, and for each install its
//! partitions (still the recorded ones), the systems on its Apple region,
//! its NVRAM (what the next boot will be, and a failed update's
//! `ota-result`) and whether its SEP state is provisioned, and the firmware
//! the newest inventory in the USB's boot log asks for and the USB lacks.
//! Each problem names the command that fixes it, and any problem fails the
//! command.
//!
//! Repair puts back what the record says: `stela.cfg` and the startup
//! choices, and, from the active build's restore image (`--ipsw` or
//! `--fetch`), every damaged or missing restore bundle and a damaged install
//! medium; with an image it prepares every restore bundle of the active
//! build again, with each machine's own ECID, and gives an install without
//! one its ECID (`machines::machine_ecid`). It removes the bundles and media
//! of other builds and generations that an interrupted update left. A
//! missing or foreign `stela.efi` is `update stela`'s to replace: no copy of
//! it is kept. With `--initialize-nor` it replaces machine state `stela.efi`
//! would refuse (`installed::reinitialize`), blanking a foreign SEP area
//! only for a machine that never booted. It adds the pinned firmware the
//! last boot asked for and the USB lacks (`firmware::requested`), keeping
//! what the USB holds, and names files without a pinned source.
use super::{
    Environment, LOADER_PATH, Record, Reported, Request, apply_install, bundle,
    check_bundle_revision, describe_install, ecid::Ecid, fix, host::Host, host::bytes_text,
    installed, interrupted_text, machines, plan_install, publish, read_record,
    record::read_inspection_record, revision_mismatch, state, update, write_record,
};
use anyhow::{Context, Result, bail, ensure};
use std::{
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};
use stela_loader::mount_interface::layout::{self, Layout};
use stela_loader::{boot_selection, mount_interface, recovery};
use {
    stela_formats::apfs, stela_formats::apfs::Container,
    stela_formats::apfs::volume::APFS_VOL_ROLE_UPDATE, stela_formats::payload::ReadAt,
    stela_formats::preboot, stela_soc_data::nor::CAPACITY,
};

/// A host partition as an APFS source.
struct Partition<'a> {
    host: &'a dyn Host,
    identifier: String,
    bytes: u64,
}

impl ReadAt for Partition<'_> {
    type Error = String;

    fn len(&self) -> u64 {
        self.bytes
    }

    fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> Result<(), String> {
        let bytes = self
            .host
            .read(&self.identifier, offset, buffer.len())
            .map_err(|error| format!("{error:#}"))?;
        buffer.copy_from_slice(&bytes);
        Ok(())
    }
}

/// The installed systems on an Apple region: each container's boot
/// directories, whether it has a staged update, and the boot directories
/// of its paired recoveryOS and System Recovery (`stela_loader::recovery`).
fn systems(host: &dyn Host, identifier: &str, bytes: u64) -> Vec<String> {
    let mut disk = Partition {
        host,
        identifier: identifier.into(),
        bytes,
    };
    let partitions = match apfs::container_partitions(&mut disk) {
        Ok(partitions) => partitions,
        Err(_) => return Vec::new(),
    };
    let mut found = Vec::new();
    for (offset, _) in partitions {
        let Ok(mut container) = Container::open(&mut disk, offset) else {
            continue;
        };
        let update = container.volume_with_role(APFS_VOL_ROLE_UPDATE).is_ok();
        match preboot::candidates(&mut container) {
            Ok((_, sets)) => {
                for set in sets {
                    found.push(format!(
                        "container at {offset:#x}: {}{}",
                        set.directory(),
                        if update {
                            "; an update is staged on its Update volume"
                        } else {
                            ""
                        }
                    ));
                }
            }
            Err(error) => found.push(format!(
                "container at {offset:#x}: no boot objects ({error})"
            )),
        }
    }
    // The recoveryOS copies stela.efi boots when NVRAM asks for one.
    for kind in [recovery::Kind::Paired, recovery::Kind::System] {
        match recovery::find(&mut disk, kind, None) {
            Ok(copies) => {
                for copy in copies {
                    for set in &copy.sets {
                        found.push(format!(
                            "{} at {:#x}: {}",
                            kind.label(),
                            copy.offset,
                            set.directory()
                        ));
                    }
                }
            }
            Err(error) => found.push(format!("{}: unreadable ({error:#})", kind.label())),
        }
    }
    found
}

/// The report and its problems.
struct Report {
    text: String,
    problems: Vec<String>,
    /// The USB, as the fixes name it.
    usb: PathBuf,
}

impl Report {
    /// The command that fixes a problem (`super::fix`).
    fn fix(&self, words: &str) -> String {
        fix(&self.usb, words)
    }

    fn line(&mut self, label: &str, value: impl std::fmt::Display) {
        let _ = writeln!(self.text, "  {label:<14} {value}");
    }

    fn problem(&mut self, problem: String) {
        self.problems.push(problem);
    }
}

/// The translation store's packs, and what `repair` would fix.
fn translations_status(record: &Record, report: &mut Report) {
    let Some(store) = &record.translations else {
        report.line("translations", "none");
        return;
    };
    report.line("translations", store.describe());
    for pack in store.packs() {
        report.line(
            "  pack",
            format!("{} ({})", pack.path.display(), bytes_text(pack.bytes)),
        );
    }
    // The first boot renames the spare into its pack by design: a note, with
    // the repair that prepares the next one.
    let taken = store.spare_taken().then(|| store.spare_repair());
    for repair in store.repairs() {
        let fix = report.fix("repair");
        let text = format!("translation store: {repair}: {fix}");
        match taken.as_ref() == Some(&repair) {
            true => report.line(
                "note",
                format!("{text} (the first boot took the spare for its pack, as it should)"),
            ),
            false => report.problem(text),
        }
    }
}

fn swap_status(host: &dyn Host, record: &Record, report: &mut Report) {
    let Some(store) = &record.volume_swap else {
        return;
    };
    let directory = store.directory(&record.volume, record.layout);
    report.line(
        "swap store",
        format!("{} ({})", directory.display(), bytes_text(store.bytes)),
    );
    if let Err(error) = store.validate_volume(host, record) {
        report.problem(format!(
            "Cannot verify the recorded swap volume ({error:#})"
        ));
        return;
    }
    for problem in store.problems(&record.volume, record.layout) {
        report.problem(format!("Swap store needs attention ({problem})"));
    }
}

fn efi_status(environment: &Environment<'_>, record: &Record, layout: Layout, report: &mut Report) {
    let tool = record.esp.join(LOADER_PATH);
    match fs::read(&tool) {
        Ok(image) => {
            let id = update::version_id(&image);
            let interface = mount_interface::find(&image);
            report.line(
                "stela.efi",
                format!(
                    "version {id}{}",
                    interface
                        .map(|i| format!(", reads boot-bundle schema {}", i.bundle_schema))
                        .unwrap_or_default()
                ),
            );
            let fix = report.fix("update stela");
            if record.stela_version.as_deref() != Some(id.as_str()) {
                report.problem(format!(
                    "{} is not the version setup installed ({}): {fix}",
                    tool.display(),
                    record.stela_version.as_deref().unwrap_or("unknown")
                ));
            }
            if let Some(reason) = mount_interface::mismatch(environment.interface, interface) {
                report.problem(format!(
                    "the installed stela.efi does not read what this stela writes ({reason}): \
                     {fix}"
                ));
            }
        }
        Err(_) => {
            let fix = report.fix("update stela");
            report.problem(format!("{} is missing: {fix}", tool.display()))
        }
    }
    let canonical = match super::nwconfig_publish::status(&record.esp) {
        Ok(Some(parsed)) => {
            report.line("nwconfig.plist", "valid");
            for warning in parsed.warnings {
                report.line("note", format!("nwconfig.plist unknown key {warning:?}"));
            }
            true
        }
        Ok(None) => false,
        Err(error) => {
            report.problem(format!("nwconfig.plist: {error:#}"));
            false
        }
    };
    let repair = report.fix("repair");
    let mirrors_required = layout == Layout::Legacy || !canonical;
    let options = record.esp.join(layout.esp_path(layout::OPTIONS));
    if mirrors_required || options.try_exists().unwrap_or(true) {
        let options = fs::read_to_string(options).unwrap_or_default();
        match record.options_text() {
            Ok(expected) if options != expected => {
                report.problem(format!("stela.cfg differs from the setup record: {repair}"))
            }
            Ok(_) => {}
            Err(error) => report.problem(format!(
                "stela.cfg cannot be written from the setup record: {error:#}"
            )),
        }
    }
    let choices = record.esp.join(layout.esp_path(layout::CHOICES));
    if mirrors_required || choices.try_exists().unwrap_or(true) {
        let choices = fs::read_to_string(choices).ok();
        match record.choices_text() {
            Ok(expected) if choices != expected => report.problem(format!(
                "the startup choices (choices.cfg) differ from the setup record: {repair}"
            )),
            Ok(_) => {}
            Err(error) => report.problem(format!("the startup choices: {error:#}")),
        }
    }
}

/// The firmware the newest inventory in the USB's boot log asks for and the
/// USB lacks: files `repair` adds are problems, the others are named.
fn firmware_status(record: &Record, report: &mut Report) {
    let Ok(tool) = fs::read(record.esp.join(LOADER_PATH)) else {
        return;
    };
    let requested = match super::firmware::requested_in(&record.esp, record.layout, &tool) {
        Ok(requested) => requested,
        Err(error) => {
            report.problem(format!("the USB's host firmware: {error:#}"));
            return;
        }
    };
    if let Some(why) = &requested.unreadable {
        report.line(
            "firmware",
            format!("the boot log's newest firmware request is unreadable: {why}"),
        );
    }
    for request in &requested.unpinned {
        report.line(
            "firmware",
            format!(
                "the last boot asked for {}, which Stela has no pinned source for",
                super::firmware::describe(request)
            ),
        );
    }
    let repair = report.fix("repair");
    for request in &requested.pinned {
        report.problem(format!(
            "the last boot asked for firmware {}, which the USB lacks: {repair} adds it",
            super::firmware::describe(request)
        ));
    }
}

/// The most recent boot captured by the USB, separate from media integrity.
fn memory_status(record: &Record, report: &mut Report) {
    let Some(log) = super::state::diagnostic_boot_log(record) else {
        report.line("last boot", "no readable USB boot log");
        return;
    };
    let diagnostics =
        crate::boot_log::diagnostics::Diagnostics::latest(&log.bytes, log.overwritten);
    report.line("last boot", log.path.display());
    match diagnostics.usable_mib {
        Some(usable) => match diagnostics
            .minimum_mib
            .filter(|_| diagnostics.memory_refused)
        {
            Some(minimum) => report.line(
                "usable RAM",
                format!("{usable} MiB ({minimum} MiB required)"),
            ),
            None => report.line("usable RAM", format!("{usable} MiB")),
        },
        None => report.line("usable RAM", "not recorded"),
    }
    if diagnostics.memory_refused {
        let state = if diagnostics.estimate {
            "refused by diagnostic memory check"
        } else {
            "boot refused before selecting a profile"
        };
        report.line("memory profile", state);
    } else {
        match diagnostics.profile {
            Some(profile) if diagnostics.estimate => {
                report.line(
                    "memory profile",
                    format!("{profile} (one-worker diagnostic estimate)"),
                );
            }
            Some(profile) => report.line("memory profile", profile),
            None => report.line("memory profile", "not recorded"),
        }
    }
    report.line(
        "swap",
        diagnostics.swap.as_deref().unwrap_or("not recorded"),
    );
    if diagnostics.memory_refused || diagnostics.swap_refused {
        report.line(
            "next diagnostic",
            format!(
                "On the target PC, press a key at USB startup and choose Diagnostics if available. Read the log with {}.",
                report.fix("boot-log")
            ),
        );
    }
}

/// The command that puts back the active build's restore files: `--fetch`
/// downloads that build, the whole image when the medium holds it.
fn image_fix(report: &Report) -> String {
    report.fix("repair --fetch")
}

fn bundle_status(environment: &Environment<'_>, record: &Record, report: &mut Report) {
    let repair = report.fix("repair");
    let image = image_fix(report);
    for kept in &record.bundles {
        let directory = kept.directory_in(&record.volume, record.layout);
        let role = if kept.build == record.build && kept.generation() == record.generation() {
            "active"
        } else {
            "left by an interrupted update"
        };
        let problems = bundle::verify(&directory, &kept.manifest_sha256);
        report.line(
            kept.kind.label(),
            format!(
                "{} ({role}, {}{}): {}",
                directory.display(),
                kept.generation(),
                kept.ecid
                    .map(|ecid| format!(", ECID {ecid}"))
                    .unwrap_or_default(),
                if problems.is_empty() {
                    "intact"
                } else {
                    "DAMAGED"
                }
            ),
        );
        if role != "active" {
            report.problem(format!(
                "{} is left by an interrupted update: run that update again, or {repair} \
                 removes it",
                directory.display()
            ));
            continue;
        }
        for problem in problems {
            report.problem(format!(
                "{problem}: {image} (or --ipsw with the macOS {} restore image)",
                kept.build
            ));
        }
    }
    for directory in record.build_directories() {
        for (_, what) in bundle::leftovers(&directory) {
            report.problem(format!("{what}: {repair} puts it right"));
        }
    }
    if let Some(reason) = revision_mismatch(environment, record) {
        report.problem(reason);
    }
    for (kind, ecid) in record.missing_restore_bundles() {
        report.problem(format!(
            "the active build {} has no {} bundle for {}, so its {} choice is missing: {image}",
            record.build,
            kind.label(),
            ecid.map_or_else(|| "the setup".into(), |e| format!("ECID {e}")),
            super::restore_choice_name(kind)
        ));
    }
    if record.installs_media() && record.active_medium().is_none() {
        report.problem(format!(
            "the active build {} has no install medium, so its Install macOS choice is \
             missing: {image}",
            record.build
        ));
    }
    for medium in &record.media {
        let problems = medium.problems();
        if medium.build != record.build {
            report.line(
                "leftover",
                format!("{} (macOS {})", medium.directory().display(), medium.build),
            );
            report.problem(format!(
                "{} is left by an interrupted update: run that update again, or {repair} \
                 removes it",
                medium.directory().display()
            ));
            continue;
        }
        report.line(
            "install medium",
            format!(
                "{} (macOS {}, partition {}): {}",
                medium.directory().display(),
                medium.build,
                medium.partition_guid,
                if problems.is_empty() {
                    "intact"
                } else {
                    "DAMAGED"
                }
            ),
        );
        for problem in problems {
            report.problem(format!(
                "{problem}: {image} (or --ipsw with the macOS {} restore image)",
                medium.build
            ));
        }
    }
}

/// `--install NAME` when the setup has several installs, for a fix that
/// acts on one.
fn install_words(record: &Record, install: &machines::Install) -> String {
    match record.installs.len() {
        1 => String::new(),
        _ => format!(" --install {}", super::quoted(&install.name)),
    }
}

fn install_status(
    environment: &Environment<'_>,
    record: &Record,
    install: &machines::Install,
    report: &mut Report,
) {
    let host = environment.host;
    let default = record.default().is_some_and(|d| d.name == install.name);
    let _ = writeln!(
        report.text,
        "Install \"{}\"{}:",
        install.name,
        if default { " (default)" } else { "" }
    );
    match install.ecid {
        Some(ecid) => report.line("ECID", ecid),
        None => {
            let fix = report.fix("repair --fetch");
            report.problem(format!(
                "install \"{}\" has no ECID (an older setup made it): {fix}, with --ecid or \
                 --new-ecid when its SEP state is provisioned",
                install.name
            ))
        }
    }
    let disk = &install.disk;
    // A portable installer's disk is chosen in stela.efi, out of this
    // computer's reach.
    if disk.is_chosen() {
        report.line("Mac's SSD", "a PC disk chosen at Install macOS");
        return;
    }
    match installed::confirm(host, disk, installed::Role::AppleRegion, None) {
        Ok(region) => {
            let bytes = host.partition(&region).map(|p| p.bytes).unwrap_or(0);
            // The guest's SSD ends where a machine state in its tail starts.
            let bytes = disk.machine_state_offset.unwrap_or(bytes).min(bytes);
            let systems = systems(host, &region, bytes);
            report.line(
                "Mac's SSD",
                format!(
                    "{region} (serial {}, partition {}): {}",
                    disk.apple_region_serial,
                    disk.apple_region_guid,
                    if systems.is_empty() {
                        "no installed macOS yet; choose Install macOS"
                    } else {
                        "macOS installed"
                    }
                ),
            );
            if region != disk.apple_region {
                report.line(
                    "",
                    format!("was {}; `repair` records the new name", disk.apple_region),
                );
            }
            for system in systems {
                report.line("", system);
            }
        }
        Err(error) => report.problem(format!(
            "install \"{}\"'s SSD partition: {error:#}",
            install.name
        )),
    }
    if let Ok(Some(region)) = installed::foreign(host, disk) {
        let fix = report.fix("repair --erase-target");
        report.problem(format!(
            "{}: {fix} erases it",
            foreign_problem(&install.name, &region.identifier)
        ));
    }
    let area = match installed::state_area(host, disk, None) {
        Ok(area) => area,
        Err(error) => {
            report.problem(format!(
                "install \"{}\"'s machine state: {error:#}",
                install.name
            ));
            return;
        }
    };
    report.line(
        "machine state",
        format!("{} ({})", disk.state_description(), bytes_text(area.bytes)),
    );
    let bytes = match area.read(
        host,
        0,
        CAPACITY + stela_formats::machine_state::SEP_BYTES as usize,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            report.problem(format!("{} cannot be read: {error:#}", area.describe()));
            return;
        }
    };
    let (nor_bytes, sep_area) = bytes.split_at(CAPACITY);
    // An interrupted `--target` can leave a machine that never booted with
    // foreign state, which `repair --initialize-nor` replaces.
    let never_booted = installed::never_booted(host, &area)
        .ok()
        .flatten()
        .unwrap_or(false);
    let nor = area.describe();
    let which = install_words(record, install);
    match state::nvram(nor_bytes) {
        state::Nvram::Blank => report.line("NVRAM", "blank"),
        state::Nvram::Valid(variables) => {
            report.line("next boot", state::next_boot(&variables));
            for name in [
                boot_selection::OTA_RESULT,
                boot_selection::OTA_FAILURE_REASON,
            ] {
                if let Some(value) = boot_selection::lookup(&variables, name) {
                    report.line(
                        "update",
                        format!(
                            "{}: {}",
                            String::from_utf8_lossy(name),
                            String::from_utf8_lossy(value)
                        ),
                    );
                }
            }
        }
        state::Nvram::Invalid(error) if never_booted => {
            let fix = report.fix("repair --initialize-nor");
            report.problem(format!("{nor}'s NVRAM is unreadable ({error}): {fix}"))
        }
        state::Nvram::Invalid(error) => {
            let fix = report.fix(&format!("nvram reset{which}"));
            report.problem(format!("{nor}'s NVRAM is unreadable ({error}): {fix}"))
        }
    }
    match state::sep(sep_area) {
        Ok(true) => report.line("SEP state", "provisioned"),
        Ok(false) => report.line("SEP state", "blank"),
        Err(error) if never_booted => {
            let fix = report.fix("repair --initialize-nor");
            report.problem(format!(
                "{nor}'s SEP area: {error}; the machine never booted, so it holds no keys: {fix} \
                 blanks it"
            ))
        }
        Err(error) => {
            let lost = installed::lost_keys(install, &report.usb, record.installs.len() > 1);
            report.problem(format!("{nor}'s SEP area: {error}. {lost}"))
        }
    }
}

/// An install's SSD partition holds another file system.
fn foreign_problem(install: &str, region: &str) -> String {
    format!(
        "install \"{install}\"'s SSD partition {region} holds another file system. The PC's own \
         OS mounts it and can overwrite Stela's machine state in it"
    )
}

/// `status`.
pub(super) fn status(environment: &Environment<'_>, usb: &Path) -> Result<String> {
    let (record, layout, path) = read_inspection_record(environment, usb)?;
    let mut report = Report {
        text: String::new(),
        problems: Vec::new(),
        usb: usb.to_path_buf(),
    };
    let _ = writeln!(
        report.text,
        "{} setup on {} ({}):",
        match layout {
            Layout::Northwood => "Northwood",
            Layout::Legacy => "Stela",
        },
        record.esp.display(),
        path.display()
    );
    if !record.complete {
        report.problem(interrupted_text(&record, usb));
    }
    report.line("model", &record.model);
    report.line("macOS build", &record.build);
    efi_status(environment, &record, layout, &mut report);
    memory_status(&record, &mut report);
    firmware_status(&record, &mut report);
    bundle_status(environment, &record, &mut report);
    translations_status(&record, &mut report);
    swap_status(environment.host, &record, &mut report);
    for install in &record.installs {
        install_status(environment, &record, install, &mut report);
    }
    let mut text = report.text;
    if report.problems.is_empty() {
        text.push_str("No problems found.\n");
        return Ok(text);
    }
    let count = report.problems.len();
    let _ = match count {
        1 => writeln!(text, "1 problem:"),
        _ => writeln!(text, "{count} problems:"),
    };
    for problem in &report.problems {
        let _ = writeln!(text, "  - {problem}");
    }
    Err(Reported {
        report: text,
        summary: match count {
            1 => "1 problem; the report above names its fix".into(),
            _ => format!("{count} problems; the report above names each fix"),
        },
    }
    .into())
}

/// `repair`.
pub(super) fn repair(environment: &Environment<'_>, command: &Request) -> Result<String> {
    let usb = &command.usb;
    let mut record = read_record(environment, usb)?;
    update::ready(&record, usb)?;
    let esp = record.esp.clone();
    let mut text = format!("Stela repair on {}:\n", esp.display());
    let Ok(tool) = fs::read(esp.join(LOADER_PATH)) else {
        bail!(
            "{} has no stela.efi: {}",
            esp.display(),
            fix(usb, "update stela")
        );
    };
    if let Some(reason) =
        mount_interface::mismatch(environment.interface, mount_interface::find(&tool))
    {
        bail!(
            "the setup's stela.efi does not read what this stela writes ({reason}): run {} first",
            fix(usb, "update stela")
        );
    }
    update::check_firmware_support(&esp, record.layout, &tool)?;
    let swap_repairs = if let Some(store) = &record.volume_swap {
        store.validate_volume(environment.host, &record)?;
        store.repair_plan(&record.volume, record.layout)?
    } else {
        Vec::new()
    };
    // The firmware the last boot asked for and the USB lacks.
    let requested = super::firmware::requested_in(&esp, record.layout, &tool)?;
    // Machines without an ECID get one; their bundles are prepared again.
    let ipsw = command.optional_image()?;
    if ipsw.is_some() {
        check_bundle_revision(environment, &record)?;
    }
    record.install_image = command.install_image.or(record.install_image);
    let mut notes = Vec::new();
    // Disk names change with plug order and reboots: each install's
    // partitions are found again by their GUIDs and serials, and recorded
    // under their new names.
    for install in &mut record.installs {
        if install.disk.is_chosen() {
            continue;
        }
        for (role, recorded) in [
            (
                installed::Role::AppleRegion,
                install.disk.apple_region.clone(),
            ),
            (installed::Role::Nor, install.disk.nor.clone()),
        ] {
            let Ok(found) = installed::confirm(environment.host, &install.disk, role, None) else {
                continue;
            };
            if found == recorded {
                continue;
            }
            writeln!(
                text,
                "  disks          install \"{}\": {recorded} is now {found}",
                install.name
            )?;
            match role {
                installed::Role::AppleRegion => install.disk.apple_region = found.clone(),
                installed::Role::Nor => install.disk.nor = found,
            }
            // A machine state in the region's tail is named by the region.
            if install.disk.machine_state_offset.is_some() {
                install.disk.nor = install.disk.apple_region.clone();
            }
        }
    }
    let mut assigned: Vec<(String, Ecid)> = Vec::new();
    if ipsw.is_some() {
        for index in 0..record.installs.len() {
            if record.installs[index].ecid.is_some() {
                continue;
            }
            let install = record.installs[index].clone();
            let area = installed::state_area(environment.host, &install.disk, None)
                .and_then(|area| {
                    area.read(
                        environment.host,
                        CAPACITY as u64,
                        stela_formats::machine_state::SEP_ROOTS_BYTES,
                    )
                })
                .with_context(|| {
                    format!(
                        "install \"{}\" needs an ECID, which depends on its machine state; \
                         connect its disk",
                        install.name
                    )
                })?;
            let provisioned = state::sep(&area).unwrap_or(false);
            let ecid = machines::machine_ecid(
                machines::EcidOptions {
                    given: command.ecid,
                    recorded: None,
                    sep_provisioned: provisioned,
                    new_ecid: command.new_ecid,
                },
                &mut notes,
            )?;
            record.installs[index].ecid = Some(ecid);
            assigned.push((install.name.clone(), ecid));
        }
    }
    for (name, ecid) in &assigned {
        writeln!(text, "  ECID           install \"{name}\" is now {ecid}")?;
    }
    // What interrupted copies left, before any bundle is judged; a dry run
    // judges a set-aside bundle where it would be put back.
    for directory in record.build_directories() {
        for (destination, what) in bundle::leftovers(&directory) {
            if command.dry_run {
                writeln!(text, "  bundle         would put right: {what}")?;
                continue;
            }
            for done in bundle::recover(&destination)? {
                writeln!(text, "  bundle         {done}")?;
            }
        }
    }
    let stale = record.drop_stale_bundles();
    for old in &stale {
        writeln!(
            text,
            "  remove         {}",
            old.directory_in(&record.volume, record.layout).display()
        )?;
    }
    // What an interrupted update left beside the active build and
    // generation.
    let leftovers = record.has_leftovers();
    if leftovers {
        writeln!(
            text,
            "  remove         the files an interrupted update left beside macOS {}'s",
            record.build
        )?;
    }
    // The install's restore bundles and medium: all again with an image.
    // A setup that has written media needs the active build's: it is put
    // back when damaged or missing.
    let has_media = record.installs_media();
    let damaged_medium = has_media
        && record
            .active_medium()
            .is_none_or(|medium| !medium.problems().is_empty());
    let mut restore: Vec<(bundle::Kind, Option<Ecid>)> = Vec::new();
    if has_media {
        for (kind, ecid) in super::restore_bundles(&record.install_machines()) {
            let intact = record.restore_bundle(kind, ecid).is_some_and(|b| {
                bundle::verify(
                    &bundle::in_place(&b.directory_in(&record.volume, record.layout)),
                    &b.manifest_sha256,
                )
                .is_empty()
            });
            if ipsw.is_some() || !intact {
                restore.push((kind, ecid));
            }
        }
    }
    let install = match (&ipsw, restore.is_empty() && !damaged_medium) {
        (_, true) => None,
        (Some(ipsw), false) => {
            let plan = plan_install(
                environment,
                &record,
                super::create::InstallOptions {
                    build: &record.build,
                    ipsw,
                    bundles: &restore,
                    additional_bytes: 0,
                },
            )?;
            describe_install(&record, &plan, &mut text)?;
            Some(plan)
        }
        (None, false) => {
            writeln!(
                text,
                "  install        {} restore bundles of macOS {} are damaged or missing{}",
                restore.len(),
                record.build,
                if damaged_medium {
                    ", and so is the install medium"
                } else {
                    ""
                }
            )?;
            return Err(Reported {
                report: text,
                summary: format!(
                    "repair needs the macOS {} restore image: run {} (or --ipsw FILE in place \
                     of --fetch)",
                    record.build,
                    fix(usb, "repair --fetch")
                ),
            }
            .into());
        }
    };
    writeln!(
        text,
        "  boot loader    stela.efi, stela.cfg and the startup choices as recorded"
    )?;
    for line in requested.lines() {
        writeln!(text, "  firmware       {line}")?;
    }
    // Machine state stela.efi would refuse: replaced with
    // --initialize-nor, else named.
    // A disk that is not connected keeps its machine state as it is; the
    // USB's own files are repaired all the same.
    let mut reinitialize = Vec::new();
    for install in &record.installs {
        let lost = installed::lost_keys(install, usb, record.installs.len() > 1);
        let fresh = match installed::reinitialize(environment.host, &install.disk, &lost) {
            Ok(Some(fresh)) => fresh,
            Ok(None) => continue,
            Err(error) => {
                notes.push(unreachable_note(&install.name, &error));
                continue;
            }
        };
        match (&fresh.refusal, command.initialize_nor) {
            (Some(refusal), true) => {
                return Err(Reported {
                    report: text,
                    summary: format!("install \"{}\": {refusal}", install.name),
                }
                .into());
            }
            (Some(refusal), false) => {
                notes.push(format!("install \"{}\": {refusal}", install.name))
            }
            (None, true) => {
                for change in &fresh.changes {
                    writeln!(text, "  disks          {change}")?;
                }
                reinitialize.push(fresh);
            }
            (None, false) => notes.push(format!(
                "install \"{}\"'s machine state would be refused by stela.efi; {} replaces it",
                install.name,
                fix(usb, "repair --initialize-nor")
            )),
        }
    }
    // An SSD partition holding another file system: erased with
    // --erase-target, else named.
    let mut foreign = Vec::new();
    for install in &record.installs {
        let region = match installed::foreign(environment.host, &install.disk) {
            Ok(Some(region)) => region,
            Ok(None) => continue,
            Err(error) => {
                // Named once, as the machine-state check above names it.
                let note = unreachable_note(&install.name, &error);
                if !notes.contains(&note) {
                    notes.push(note);
                }
                continue;
            }
        };
        if command.erase_target {
            writeln!(
                text,
                "  disks          erase the file system on {}, install \"{}\"'s SSD partition",
                region.identifier, install.name
            )?;
            foreign.push(region);
        } else {
            notes.push(format!(
                "{}: {} erases it",
                foreign_problem(&install.name, &region.identifier),
                fix(usb, "repair --erase-target")
            ));
        }
    }
    for note in &notes {
        writeln!(text, "  note           {note}")?;
    }
    let store_repairs = record
        .translations
        .as_ref()
        .map(super::translations::Store::repairs)
        .unwrap_or_default();
    for repair in &store_repairs {
        writeln!(text, "  translations   {repair}")?;
    }
    for repair in &swap_repairs {
        writeln!(text, "  swap store     {repair}")?;
    }
    if command.dry_run {
        text.push_str("Dry run: nothing was changed.\n");
        return Ok(text);
    }
    if let Some(store) = &record.translations
        && !store_repairs.is_empty()
    {
        store.repair()?;
    }
    for region in &foreign {
        installed::erase_foreign(environment.host, region)?;
    }
    for fresh in &reinitialize {
        fresh.area.write(environment.host, 0, &fresh.bytes)?;
        ensure!(
            fresh.area.read(environment.host, 0, fresh.bytes.len())? == fresh.bytes,
            "{} did not read back as written",
            fresh.area.describe()
        );
    }
    if let (Some(plan), Some(ipsw)) = (install, &ipsw) {
        apply_install(environment, &mut record, ipsw, plan)?;
    }
    if let Some(store) = &record.volume_swap
        && !swap_repairs.is_empty()
    {
        store.validate_volume(environment.host, &record)?;
        super::volume_log::prepare(&record)?;
        store.publish(&record.volume, record.layout)?;
    }
    publish(&mut record)?;
    write_record(&esp, &record)?;
    super::remove_bundles(&record, &stale)?;
    if leftovers {
        update::prune(&mut record)?;
        write_record(&esp, &record)?;
    }
    requested.provision(&esp, &tool)?;
    writeln!(text, "Done. Check it with {}", fix(usb, "status"))?;
    Ok(text)
}

/// Why an install's disk was left alone.
fn unreachable_note(install: &str, error: &anyhow::Error) -> String {
    format!("install \"{install}\"'s disk is left as it is: {error:#}")
}
