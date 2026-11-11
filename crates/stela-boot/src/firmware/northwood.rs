// SPDX-License-Identifier: GPL-2.0-or-later
//! Bridge the Northwood plist to the established option line and menu.
use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use northwood_config::{BootTimeout, NwConfig, plist::MAX_BYTES, profiles, render};
use stela_boot_core::options::{DRAM_BASE, HOST_CHECK, HostCheck, System};
use stela_loader::mount_interface::{Choice, Choices};
use uefi::{CStr16, cstr16, fs::FileSystem};

pub(super) const CONFIG: &CStr16 = cstr16!("\\EFI\\northwood\\nwconfig.plist");

pub(super) struct Settings {
    pub options: String,
    pub choices: Option<Choices>,
    pub unknown_keys: usize,
    pub xml: Vec<u8>,
    pub explicit_entries: bool,
    pub profile: Option<&'static str>,
    pub volume_log: bool,
}

pub(super) fn settings(
    volume: &mut FileSystem,
    legacy: Option<&[u8]>,
    choices: Option<&Choices>,
) -> Result<Settings, &'static str> {
    let info = volume
        .metadata(CONFIG)
        .map_err(|_| "cannot read file metadata")?;
    if info.file_size() > MAX_BYTES as u64 {
        return Err("exceeds the configuration size limit");
    }
    let bytes = volume.read(CONFIG).map_err(|_| "cannot read the file")?;
    let (parsed, profile) = profiles::parse(&bytes, profiles::Hardware::default())
        .map_err(|_| "contains an invalid setting or hardware profile")?;
    let legacy = legacy.and_then(|line| core::str::from_utf8(line).ok());
    let default = |config: &NwConfig| -> Result<String, &'static str> {
        let matches = legacy.into_iter().chain(
            choices
                .iter()
                .flat_map(|choices| choices.entries.iter().map(|entry| entry.options.as_str())),
        );
        options(config, matches)
    };

    let boot = &parsed.config.boot;
    let force_diagnostics = parsed.config.misc.debug.host_check;
    let explicit_entries = !force_diagnostics && !boot.entries.is_empty();
    let timeout = match boot.timeout {
        BootTimeout::Auto => choices
            .as_ref()
            .map_or(5, |choices| choices.timeout_seconds),
        BootTimeout::Seconds(seconds) => seconds,
    };
    let (options, choices) = if force_diagnostics {
        (default(&parsed.config)?, None)
    } else if boot.entries.is_empty() {
        let options = default(&parsed.config)?;
        let choices = choices.cloned().map(|mut choices| {
            choices.timeout_seconds = timeout;
            if let Some(first) = choices.entries.first_mut() {
                first.options = options.clone();
            }
            choices
        });
        (options, choices)
    } else {
        let mut entries = Vec::with_capacity(boot.entries.len());
        for entry in &boot.entries {
            entries.push(Choice {
                name: entry.name.clone(),
                options: default(&entry.config)?,
            });
        }
        if !boot.default_install.is_empty() {
            let selected = entries
                .iter()
                .position(|entry| entry.name == boot.default_install)
                .ok_or("default install is absent from the boot entries")?;
            let entry = entries.remove(selected);
            entries.insert(0, entry);
        }
        let options = entries[0].options.clone();
        let choices = Choices {
            timeout_seconds: timeout,
            entries,
        };
        choices.encode().map_err(|_| "boot entries were rejected")?;
        (options, Some(choices))
    };

    Ok(Settings {
        options,
        choices,
        unknown_keys: parsed.warnings.len(),
        xml: bytes,
        explicit_entries,
        profile,
        volume_log: parsed.config.misc.boot_log.volume,
    })
}

fn options<'a>(
    config: &NwConfig,
    legacy: impl Iterator<Item = &'a str>,
) -> Result<String, &'static str> {
    let line = render::options_line(config, cfg!(feature = "diagnostics"))
        .map_err(|_| "cannot render the system options")?;
    if line.split_ascii_whitespace().next() == Some(HOST_CHECK) {
        return HostCheck::parse(&line)
            .map(|options| options.to_string())
            .map_err(|_| "host-check options were rejected");
    }
    let mut options = System::parse(&line).map_err(|_| "system options were rejected")?;
    if !(options.bundle || cfg!(feature = "diagnostics") && options.entry != DRAM_BASE) {
        return Err("does not select a boot source");
    }
    // A personalized identity belongs to the setup machine state. During
    // migration recover it from the legacy options for that exact source.
    if let Some(old) = legacy
        .filter_map(|line| System::parse(line).ok())
        .find(|old| {
            old.storage == options.storage
                && old.nor == options.nor
                && old.preboot == options.preboot
                && old.bundle_directory == options.bundle_directory
                && old.install == options.install
                && old.install_source == options.install_source
        })
    {
        options.ecid = old.ecid;
        options.development_restore |= !options.update_restore && old.development_restore;
        options.profile |= old.profile;
    }
    Ok(options.to_string())
}
