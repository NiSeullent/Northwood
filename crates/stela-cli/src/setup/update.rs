//! Updates of an existing setup: a Stela update replaces `stela.efi`, a
//! macOS update replaces the install medium and restore bundles. Each one
//! writes the new files first and removes the old ones last, so one that was
//! interrupted is finished by running it again. The record names the new
//! version or build only once it is active, and `repair` removes what a stop
//! left of the other one (`Record::has_leftovers`). There is no rollback.
//!
//! - Stela: the USB's boot loader is replaced atomically by the new one. A
//!   new `stela.efi` must carry this `stela`'s mount interface. When it reads
//!   another boot-bundle schema than the setup's bundles have, or this
//!   `stela` prepares another bundle revision
//!   (`stela_loader::boot_bundle::REVISION`) than theirs, the update prepares
//!   the active build's restore bundles for every machine again (while the
//!   setup has the build's install medium) at the new schema and revision,
//!   from `--ipsw` or `--fetch`; this makes it an update even with the same
//!   `stela.efi`. The cache store's spare is prepared before activating it:
//!   its new translator takes that spare. Repeating an already-installed
//!   update replenishes a consumed spare without replacing EFI or boot
//!   configuration.
//! - macOS: an installed macOS updates itself: Software Update stages the
//!   new system on the container's Update volume and sets `boot-command
//!   upgrade`, which `stela.efi` boots from that volume as iBEC would
//!   (`stela_loader::upgrade`), and every later boot builds from the
//!   Preboot volume's new objects. `update os` brings the setup's Install
//!   macOS to a new build: the build's install medium and each
//!   machine's restore bundles, so the USB needs room for both while it runs.
//!
//! Neither touches the installed disks' partitions.
use super::{
    Environment, LOADER_PATH, Prepared, Record, apply_install, bundle, check_bundle_revision,
    check_space, copy_bundles, describe_install, plan_install, prepare_kind, publish, read_efi,
    restore_bundles, write_record,
};
use crate::efi;
use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use std::{fmt::Write as _, fs, path::Path};
use stela_loader::mount_interface::layout::Layout;

pub(super) fn version_id(image: &[u8]) -> String {
    let digest = hex::encode(Sha256::digest(image));
    digest[..16].into()
}

pub(super) fn check_firmware_support(esp: &Path, layout: Layout, image: &[u8]) -> Result<()> {
    super::check_layout_efi(layout, image)?;
    let catalog = super::firmware::catalog_directory(esp, layout, image)?;
    stela_loader::host_firmware::check_installed_mount(&catalog, image)
        .context("stela.efi cannot load installed host firmware")?;
    Ok(())
}

/// Make `build` the active build, which Install macOS restores; its
/// restore bundles and medium are the startup choices' from now on.
fn activate_build(record: &mut Record, build: &str) -> Result<()> {
    record.build = build.into();
    record.bundle_directory = bundle::build_directory_in(&record.volume, record.layout, build);
    publish(record)
}

/// Remove the bundles and media of everything but the active build and
/// generation.
pub(super) fn prune(record: &mut Record) -> Result<()> {
    let (build, generation) = (record.build.clone(), record.generation());
    let (active, stale): (Vec<_>, Vec<_>) = record
        .bundles
        .drain(..)
        .partition(|b| b.build == build && b.generation() == generation);
    record.bundles = active;
    super::remove_bundles(record, &stale)?;
    for build in stale.iter().map(|b| &b.build) {
        let directory = bundle::build_directory_in(&record.volume, record.layout, build);
        if *build != record.build && directory.is_dir() {
            fs::remove_dir_all(directory)?;
        }
    }
    let (active, stale): (Vec<_>, Vec<_>) = record.media.drain(..).partition(|m| m.build == build);
    record.media = active;
    for medium in stale {
        remove_medium(&medium)?;
    }
    Ok(())
}

/// Delete a medium the record no longer names.
fn remove_medium(medium: &super::medium::Medium) -> Result<()> {
    let directory = medium.removable_directory()?;
    if directory.is_dir() {
        fs::remove_dir_all(&directory)
            .with_context(|| format!("cannot remove {}", directory.display()))?;
    }
    Ok(())
}

/// Refuse a change to an interrupted setup, naming the command that
/// finishes it.
pub(super) fn ready(record: &Record, usb: &Path) -> Result<()> {
    ensure!(record.complete, "{}", super::interrupted_text(record, usb));
    Ok(())
}

/// Image sources and write policy for a Stela update.
pub(super) struct UpdateOptions<'a> {
    /// The USB, as `--usb` named it.
    pub(super) usb: &'a Path,
    pub(super) image_path: &'a Path,
    pub(super) ipsw: Option<&'a Path>,
    pub(super) dry_run: bool,
}

/// Replace `stela.efi` with `image_path`, preparing bundles of its schema
/// and this `stela`'s revision from `ipsw` when the setup's are of another.
pub(super) fn update_stela(
    environment: &Environment<'_>,
    mut record: Record,
    options: UpdateOptions<'_>,
) -> Result<String> {
    let UpdateOptions {
        usb,
        image_path,
        ipsw,
        dry_run,
    } = options;
    let image = read_efi(environment, image_path)?;
    check_firmware_support(&record.esp, record.layout, &image)?;
    let to = version_id(&image);
    let to_generation = environment.generation();
    ready(&record, usb)?;
    let esp = record.esp.clone();
    // A missing or damaged loader is what an update also puts right.
    let from = fs::read(esp.join(LOADER_PATH)).map_or("none".into(), |image| version_id(&image));
    let from_generation = record.generation();
    let mut text = String::new();
    writeln!(text, "Stela update on {}:", esp.display())?;
    text.push_str(&super::fdr::prepare(&record, true)?);
    // The firmware the last boot asked for and the USB lacks, which every
    // update that changes something adds.
    let requested = super::firmware::requested_in(&esp, record.layout, &image)?;
    for line in requested.lines() {
        writeln!(text, "  firmware       {line}")?;
    }
    if from == to && from_generation == to_generation {
        if !dry_run {
            super::fdr::prepare(&record, false)?;
        }
        // An update stopped after its loader went in: the record still names
        // the previous version, or the bundles and media it replaced.
        if record.stela_version.as_deref() != Some(to.as_str()) || record.has_leftovers() {
            writeln!(
                text,
                "  stela.efi      version {to} is installed; its update is finished"
            )?;
            writeln!(
                text,
                "  boot loader    stela.cfg and the startup choices as recorded"
            )?;
            if dry_run {
                text.push_str("Dry run: nothing was changed.\n");
                return Ok(text);
            }
            if let Some(store) = &record.translations {
                store.publish()?;
            }
            activate(&mut record, to)?;
            requested.provision(&esp, &image)?;
            text.push_str("Done.\n");
            return Ok(text);
        }
        writeln!(text, "  stela.efi      version {to} is already installed")?;
        let spare = record
            .translations
            .as_ref()
            .filter(|store| !store.config_intact() || !store.spare_ready());
        if spare.is_some() {
            writeln!(
                text,
                "  translations   prepare the cache configuration and spare"
            )?;
        }
        if spare.is_some() || !requested.is_empty() {
            if dry_run {
                text.push_str("Dry run: nothing was changed.\n");
            } else {
                if let Some(store) = spare {
                    store.publish()?;
                }
                requested.provision(&esp, &image)?;
            }
        }
        return Ok(text);
    }
    if from == to {
        writeln!(text, "  stela.efi      version {to} stays")?;
    } else {
        writeln!(text, "  stela.efi      version {from} -> {to}")?;
    }
    // Restore bundles of the new generation for every machine of the active
    // build while it has an install medium.
    let mut next = record.clone();
    next.bundle_schema = to_generation.schema;
    next.bundle_revision = to_generation.revision;
    let missing_restore = match from_generation == to_generation {
        true => Vec::new(),
        false => next.missing_restore_bundles(),
    };
    let prepared = if missing_restore.is_empty() {
        Vec::new()
    } else {
        let Some(ipsw) = ipsw else {
            bail!(
                "this stela and stela.efi {to} boot bundles of {to_generation}, and this \
                 setup's are {from_generation}. The update prepares new ones from the macOS {} \
                 restore image: run {} (or --ipsw FILE in place of --fetch)",
                record.build,
                super::fix(usb, "update stela --fetch")
            );
        };
        let mut prepared: Vec<Prepared> = Vec::new();
        for kind in bundle::RESTORE_KINDS {
            let ecids: Vec<_> = missing_restore
                .iter()
                .filter(|(k, _)| *k == kind)
                .map(|&(_, ecid)| ecid)
                .collect();
            if !ecids.is_empty() {
                let (_, _, restore) = prepare_kind(
                    environment,
                    super::prepare::BundleOptions {
                        kind,
                        ipsw,
                        model: &record.model,
                        build: Some(&record.build),
                        ecids: &ecids,
                    },
                )?;
                prepared.extend(restore);
            }
        }
        check_space(environment.host, &record.volume, &prepared)?;
        prepared
    };
    for bundle in &prepared {
        writeln!(
            text,
            "  {:<14} {} ({to_generation}, {}), replaces {from_generation}",
            bundle.kind.label(),
            bundle::directory_of_in(
                &record.volume,
                record.layout,
                bundle::Identity {
                    kind: bundle.kind,
                    build: &bundle.build,
                    ecid: bundle.ecid,
                    generation: to_generation
                }
            )
            .display(),
            super::host::bytes_text(bundle.bytes)
        )?;
    }
    match &record.translations {
        Some(store) => {
            if from == to {
                writeln!(text, "  translations   {}", store.describe())?;
            } else {
                writeln!(
                    text,
                    "  translations   {}; new pack for the new stela.efi",
                    store.describe()
                )?;
            }
            for pack in store.stale() {
                writeln!(text, "  remove         {}", pack.path.display())?;
            }
        }
        None => writeln!(text, "  translations   no store on this USB")?,
    }
    if dry_run {
        text.push_str("Dry run: nothing was changed.\n");
        return Ok(text);
    }
    record.bundle_schema = to_generation.schema;
    record.bundle_revision = to_generation.revision;
    copy_bundles(&mut record, &prepared)?;
    write_record(&esp, &record)?;
    // A new translator must have its bounded spare before its EFI becomes
    // active. Failure leaves the old loader active.
    if let Some(store) = &record.translations {
        store.publish()?;
    }
    super::fdr::prepare(&record, false)?;
    efi::add_mount_in(&esp, &image, None, None, record.layout)?;
    activate(&mut record, to)?;
    // A USB whose setup left its Wi-Fi network without firmware gets it now.
    if record.wifi.is_some() {
        super::firmware::provision_wifi_in(environment.host, &esp, record.layout, &image)?;
    }
    requested.provision(&esp, &image)?;
    text.push_str("Done.\n");
    Ok(text)
}

/// The steps after the new loader is in place: `stela.cfg` and the startup
/// choices for the record, the record naming version `to`, and the bundles,
/// media and packs it replaced removed. The record names the old version
/// until it is written here, so `update stela` run again after a stop
/// anywhere before finishes these steps.
fn activate(record: &mut Record, to: String) -> Result<()> {
    publish(record)?;
    record.stela_version = Some(to);
    write_record(&record.esp.clone(), record)?;
    prune(record)?;
    if let Some(store) = &record.translations {
        store.prune()?;
    }
    write_record(&record.esp.clone(), record)
}

/// Add `ipsw`'s install medium and every machine's restore bundles, and
/// make its build the one Install macOS restores.
pub(super) fn update_os(
    environment: &Environment<'_>,
    mut record: Record,
    ipsw: &Path,
    usb: &Path,
    dry_run: bool,
) -> Result<String> {
    check_bundle_revision(environment, &record)?;
    let (product_version, build) = crate::restore::payload::identity(ipsw, &record.model)?;
    super::supported(environment, ipsw, &build)?;
    ready(&record, usb)?;
    let from = record.build.clone();
    if build == from && record.has_leftovers() {
        // Stopped after the new build became the active one: remove what it
        // replaces.
        let mut text = format!("macOS {product_version} ({build}) install files:\n");
        writeln!(
            text,
            "  remove         the restore files of earlier builds and generations"
        )?;
        if dry_run {
            text.push_str("Dry run: nothing was changed.\n");
            return Ok(text);
        }
        prune(&mut record)?;
        write_record(&record.esp.clone(), &record)?;
        text.push_str("Done.\n");
        return Ok(text);
    }
    ensure!(
        build != from,
        "macOS {build} is already the build this setup installs"
    );
    // The build's install medium and restore bundles, so Install macOS
    // restores the new build.
    let install = plan_install(
        environment,
        &record,
        super::create::InstallOptions {
            build: &build,
            ipsw,
            bundles: &restore_bundles(&record.install_machines()),
            additional_bytes: 0,
        },
    )?;
    let mut text = String::new();
    writeln!(text, "macOS {product_version} ({build}) install files:")?;
    describe_install(&record, &install, &mut text)?;
    if let Some(medium) = record.active_medium() {
        writeln!(
            text,
            "  remove         {} and the {from} restore files",
            medium.directory().display()
        )?;
    }
    if dry_run {
        text.push_str("Dry run: nothing was changed.\n");
        return Ok(text);
    }
    apply_install(environment, &mut record, ipsw, install)?;
    activate_build(&mut record, &build)?;
    // The record names the new build before the old one's files go, so a
    // stop while they are removed leaves them as leftovers, which running
    // this again or `repair` removes.
    write_record(&record.esp.clone(), &record)?;
    prune(&mut record)?;
    write_record(&record.esp.clone(), &record)?;
    text.push_str(
        "Done. An installed macOS keeps updating itself. To install this build on a new or \
         erased disk, boot the USB and choose Install macOS, which erases that disk.\n",
    );
    Ok(text)
}
