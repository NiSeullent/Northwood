// SPDX-License-Identifier: GPL-2.0-or-later
//! Maintain the setup-managed parts of the Northwood plist during migration.
use super::northwood_config as writer;
use ::northwood_config::{self as config, Parsed, plist};
use anyhow::{Context, Result, ensure};
use std::{
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
};
use stela_boot_core::options::System;
use stela_loader::mount_interface::{
    Choices,
    layout::{self, Layout},
};

pub(super) struct Publication {
    path: PathBuf,
    bytes: String,
    fresh: bool,
}

impl Publication {
    pub(super) fn write(self) -> Result<()> {
        fs::create_dir_all(self.path.parent().context("nwconfig path has no parent")?)?;
        crate::efi::write_atomic(&self.path, self.bytes.as_bytes(), self.fresh)
    }
}

fn read(path: &Path) -> Result<Option<Vec<u8>>> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    };
    let mut bytes = Vec::new();
    file.take(plist::MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    Ok(Some(bytes))
}

fn compatibility(esp: &Path, layout: Layout) -> Option<plist::Dictionary> {
    let line = read(&esp.join(layout.esp_path(layout::OPTIONS))).ok()??;
    let options = System::parse(std::str::from_utf8(&line).ok()?).ok()?;
    let choices = match read(&esp.join(layout.esp_path(layout::CHOICES))).ok()? {
        Some(bytes) => Some(Choices::parse(std::str::from_utf8(&bytes).ok()?).ok()?),
        None => None,
    };
    writer::dictionary(&options, choices.as_ref()).ok()
}

pub(super) fn prepare(
    esp: &Path,
    image: &[u8],
    options: &str,
    choices: Option<&Choices>,
    layout: Layout,
) -> Result<Option<Publication>> {
    let marker = config::MOUNT_CAPABILITY;
    if !image.windows(marker.len()).any(|window| window == marker) {
        ensure!(
            layout == Layout::Legacy,
            "the EFI image cannot read nwconfig.plist for the Northwood setup"
        );
        return Ok(None);
    }
    let path = esp.join(writer::PATH);
    let current = read(&path)?;
    if layout == Layout::Legacy && current.is_none() {
        return Ok(None);
    }
    let options = System::parse(options).map_err(anyhow::Error::msg)?;
    let next = writer::dictionary(&options, choices)?;
    let fresh = current.is_none();
    let document = if let Some(bytes) = current {
        if let Err(error) = validate(&bytes) {
            if layout == Layout::Northwood {
                return Err(error).with_context(|| format!("reading {}", path.display()));
            }
            return Ok(None);
        }
        let current = plist::parse(&bytes)?;
        let previous = match compatibility(esp, layout) {
            Some(previous) => previous,
            // Keep existing authored values when there is no previous mirror
            // to identify the fields setup generated.
            None if layout == Layout::Northwood => plist::Dictionary::new(),
            None => return Ok(None),
        };
        writer::merge_generated(&previous, next, current)
    } else {
        next
    };
    let bytes = plist::to_xml(&document);
    validate(bytes.as_bytes()).context("checking nwconfig.plist before publishing setup")?;
    Ok(Some(Publication { path, bytes, fresh }))
}

pub(super) fn validate(bytes: &[u8]) -> Result<Parsed> {
    let parsed = config::parse(bytes)?;
    for config in core::iter::once(&parsed.config)
        .chain(parsed.config.boot.entries.iter().map(|entry| &entry.config))
    {
        let options = config::render::options_line(config, true)?;
        System::parse(&options).map_err(anyhow::Error::msg)?;
    }
    Ok(parsed)
}

pub(super) fn status(esp: &Path) -> Result<Option<Parsed>> {
    read(&esp.join(writer::PATH))?
        .map(|bytes| validate(&bytes))
        .transpose()
}

/// The swap policy of the entry the firmware selects by default.
pub(super) fn default_swap_policy(parsed: &Parsed) -> config::Swap {
    let boot = &parsed.config.boot;
    let entry = if boot.default_install.is_empty() {
        boot.entries.first()
    } else {
        boot.entries
            .iter()
            .find(|entry| entry.name == boot.default_install)
    };
    entry
        .map_or(&parsed.config.memory.swap, |entry| {
            &entry.config.memory.swap
        })
        .clone()
}

pub(super) fn generated(esp: &Path, layout: Layout) -> Option<PathBuf> {
    let path = esp.join(writer::PATH);
    let bytes = read(&path).ok()??;
    let current = plist::parse(&bytes).ok()?;
    (compatibility(esp, layout).as_ref() == Some(&current)).then_some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORIGINAL: &str = "--system --preboot DISK:667cb4d7-f8ac-43bb-a744-f5cd657908b1 --storage DISK:667cb4d7-f8ac-43bb-a744-f5cd657908b1";
    const CONFIGURED: &str = "--system --preboot DISK:667cb4d7-f8ac-43bb-a744-f5cd657908b1 --storage DISK:667cb4d7-f8ac-43bb-a744-f5cd657908b1 --wifi Test%20Network";

    #[test]
    fn setup_publishes_only_for_a_compatible_loader_and_preserves_custom_fields() {
        let esp = tempfile::tempdir().unwrap();
        let esp = esp.path();
        let legacy_path = esp.join(crate::efi::OPTIONS);
        fs::create_dir_all(legacy_path.parent().unwrap()).unwrap();
        fs::write(&legacy_path, ORIGINAL).unwrap();

        assert!(
            prepare(esp, b"older loader", ORIGINAL, None, Layout::Legacy)
                .unwrap()
                .is_none()
        );
        assert!(status(esp).unwrap().is_none());
        let marker = config::MOUNT_CAPABILITY;
        assert!(
            prepare(esp, &marker, ORIGINAL, None, Layout::Legacy)
                .unwrap()
                .is_none()
        );
        let path = esp.join(writer::PATH);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let original = System::parse(ORIGINAL).unwrap();
        fs::write(
            &path,
            plist::to_xml(&writer::dictionary(&original, None).unwrap()),
        )
        .unwrap();
        assert_eq!(generated(esp, Layout::Legacy), Some(path.clone()));

        let mut edited = plist::parse(&fs::read(&path).unwrap()).unwrap();
        edited.insert("FutureSection".into(), plist::Value::Boolean(true));
        fs::write(&path, plist::to_xml(&edited)).unwrap();
        let plan = prepare(esp, &marker, CONFIGURED, None, Layout::Legacy)
            .unwrap()
            .unwrap();
        plan.write().unwrap();
        fs::write(&legacy_path, CONFIGURED).unwrap();

        let parsed = status(esp).unwrap().unwrap();
        assert_eq!(parsed.config.drivers.network.wifi.ssid, "Test Network");
        assert_eq!(parsed.warnings, ["FutureSection"]);
        assert!(generated(esp, Layout::Legacy).is_none());
    }

    #[test]
    fn northwood_publication_uses_only_its_own_compatibility_mirror() {
        let esp = tempfile::tempdir().unwrap();
        let esp = esp.path();
        let marker = config::MOUNT_CAPABILITY;
        prepare(esp, &marker, ORIGINAL, None, Layout::Northwood)
            .unwrap()
            .unwrap()
            .write()
            .unwrap();
        let options = esp.join(Layout::Northwood.esp_path(layout::OPTIONS));
        fs::write(&options, ORIGINAL).unwrap();
        assert!(generated(esp, Layout::Northwood).is_some());
        assert!(generated(esp, Layout::Legacy).is_none());
        let plan = prepare(esp, &marker, CONFIGURED, None, Layout::Northwood)
            .unwrap()
            .unwrap();
        plan.write().unwrap();
        prepare(esp, &marker, CONFIGURED, None, Layout::Northwood)
            .unwrap()
            .unwrap()
            .write()
            .unwrap();
        fs::write(options, CONFIGURED).unwrap();
        let parsed = status(esp).unwrap().unwrap();
        assert_eq!(parsed.config.drivers.network.wifi.ssid, "Test Network");
        assert!(generated(esp, Layout::Northwood).is_some());
    }
}
