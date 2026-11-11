//! `stela setup uninstall`: remove everything the setup record names.
use super::{
    Environment, locate_esp,
    record::{Record, mutation_layout, read_any_record},
    translations, wifi,
};
use crate::efi;
use anyhow::{Context, Result, ensure};
use std::{
    fmt::Write as _,
    fs,
    path::{Component, Path, PathBuf},
};
use stela_loader::mount_interface::layout::{self, Layout};

pub(super) fn uninstall(
    environment: &Environment<'_>,
    usb: &Path,
    dry_run: bool,
) -> Result<String> {
    let esp = locate_esp(environment.host, usb)?;
    let selected = mutation_layout(&esp)?;
    let layout = selected.unwrap_or(Layout::Legacy);
    let record: Option<Record> = selected
        .map(|_| read_any_record(environment.host, &esp, usb))
        .transpose()?;
    let mut text = String::new();
    let mut removals: Vec<PathBuf> = Vec::new();
    if let Some(record) = &record {
        let boot = record.volume.join(layout.data_path(layout::DATA_BOOT));
        let images = record.volume.join("stela/images");
        let older_images = record.volume.join("EFI/stela/images");
        let directories: Vec<PathBuf> = record
            .payload_directory
            .iter()
            .cloned()
            .chain([record.bundle_directory.clone()])
            .chain(record.build_directories())
            .collect();
        for directory in directories {
            ensure!(
                directory.file_name().is_some()
                    && directory.parent().is_some_and(|p| {
                        p == boot || layout == Layout::Legacy && (p == images || p == older_images)
                    }),
                "the setup record names an unexpected directory {}",
                directory.display()
            );
            if directory.exists() && !removals.contains(&directory) {
                removals.push(directory);
            }
        }
        for medium in &record.media {
            let directory = medium.removable_directory()?;
            if directory.exists() {
                removals.push(directory);
            }
        }
        if let Some(store) = &record.translations {
            let directory = translations::checked_directory(store)?;
            if directory.is_dir() {
                removals.push(directory);
            }
        }
    } else {
        writeln!(
            text,
            "No setup record on {}; removing only Stela's boot loader and files.",
            esp.display()
        )?;
    }
    let log_slots = record
        .as_ref()
        .map(super::volume_log::owned_slots)
        .unwrap_or_default();
    let swap = record
        .as_ref()
        .and_then(|record| record.volume_swap.as_ref().map(|store| (record, store)));
    let swap_files = swap
        .map(|(record, store)| {
            store.validate_volume(environment.host, record)?;
            store.removable_paths(&record.volume, record.layout)
        })
        .transpose()?
        .unwrap_or_default();
    let nwconfig = super::nwconfig_publish::generated(&esp, layout);
    let choices = esp.join(layout.esp_path(layout::CHOICES));
    let key = esp.join(wifi::key_path(layout));
    let configuration = [&choices, &key];
    writeln!(text, "Stela uninstall from {}:", esp.display())?;
    for directory in &removals {
        writeln!(text, "  remove   {}", directory.display())?;
    }
    for path in log_slots.iter().chain(&swap_files) {
        writeln!(text, "  remove   {}", path.display())?;
    }
    if let Some(path) = &nwconfig {
        writeln!(text, "  remove   {}", path.display())?;
    }
    let installed = efi::installed_paths(&esp, layout);
    for file in installed.iter().chain(configuration) {
        if file.is_file() {
            writeln!(text, "  remove   {}", file.display())?;
        }
    }
    for install in record.iter().flat_map(|r| r.installs.iter()) {
        writeln!(
            text,
            "  keep     install \"{}\": {} keep macOS, its NVRAM and SEP state{}; delete them \
             with a disk utility when no longer wanted",
            install.name,
            install.disk.partitions(),
            install
                .ecid
                .map(|ecid| format!(" (ECID {ecid}, which a new setup reuses with --ecid)"))
                .unwrap_or_default()
        )?;
    }
    if dry_run {
        text.push_str("Dry run: nothing was changed.\n");
        return Ok(text);
    }
    if let Some((record, store)) = swap {
        store.validate_volume(environment.host, record)?;
        store.remove(&record.volume, record.layout)?;
    }
    for directory in &removals {
        fs::remove_dir_all(directory)
            .with_context(|| format!("cannot remove {}", directory.display()))?;
    }
    super::volume_log::remove(log_slots)?;
    if let Some(path) = nwconfig {
        fs::remove_file(path)?;
    }
    for file in configuration {
        if file.is_file() {
            fs::remove_file(file)?;
        }
    }
    if let Some(record) = &record {
        let data = record.volume.join(layout.data_dir());
        for directory in &record.created_directories {
            if directory.strip_prefix(&data).is_ok_and(|relative| {
                relative
                    .components()
                    .all(|part| matches!(part, Component::Normal(_)))
            }) && fs::read_dir(directory).is_ok_and(|mut entries| entries.next().is_none())
            {
                fs::remove_dir(directory)?;
            }
        }
    }
    efi::uninstall_in(&esp, record.is_some(), layout)?;
    if record.is_some() {
        fs::remove_file(esp.join(layout.esp_path(layout::RECORD)))?;
        let directory = esp.join(layout.esp_dir());
        if fs::read_dir(&directory).is_ok_and(|mut entries| entries.next().is_none()) {
            fs::remove_dir(directory)?;
        }
    }
    text.push_str("Done.\n");
    Ok(text)
}
