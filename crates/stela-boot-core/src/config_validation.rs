// SPDX-License-Identifier: GPL-2.0-or-later
//! Read-only nwconfig checks for the host commands.

extern crate std;

use crate::options::{Constraint, Error, HOST_CHECK, HostCheck, System};
use alloc::{format, vec::Vec};
use anyhow::{Context, Result};
use northwood_config::{NwConfig, Parsed, plist::MAX_BYTES, profiles, render};
use std::{fs::File, io::Read, path::Path};

pub fn validate_file(file: &Path) -> Result<()> {
    let mut bytes = Vec::new();
    File::open(file)
        .and_then(|file| file.take(MAX_BYTES as u64 + 1).read_to_end(&mut bytes))
        .with_context(|| format!("reading {}", file.display()))?;
    let parsed = validate(&bytes).with_context(|| format!("validating {}", file.display()))?;
    for warning in parsed.warnings {
        std::eprintln!("warning: unknown key {warning:?}");
    }
    std::println!("{} is valid", file.display());
    Ok(())
}

fn validate(bytes: &[u8]) -> Result<Parsed> {
    let (parsed, _) = profiles::parse(bytes, profiles::Hardware::default())?;
    let config = &parsed.config;
    if config.misc.debug.host_check || config.boot.entries.is_empty() {
        validate_options(config).context("default configuration (Boot)")?;
    } else {
        for (index, entry) in config.boot.entries.iter().enumerate() {
            validate_options(&entry.config)
                .with_context(|| format!("Boot/Entries[{index}] ({:?})", entry.name))?;
        }
    }
    Ok(parsed)
}

fn validate_options(config: &NwConfig) -> Result<()> {
    // Include the option grammar of diagnostics builds, and allow partial
    // configurations and disk placeholders before a source is selected.
    let line = render::options_line(config, true)?;
    if line.split_ascii_whitespace().next() == Some(HOST_CHECK) {
        HostCheck::parse(&line)?;
    } else {
        System::parse(&line).map_err(|error| {
            let field = match error {
                Error::Constraint(Constraint::InstallSourceDependency) => "Boot/InstallSource",
                Error::Constraint(Constraint::VerifyDependency) => "Boot/VerifyOnRead",
                Error::Constraint(Constraint::PrebootSelection) => "Boot/Preboot",
                Error::Constraint(Constraint::Ipv4Gateway) => "Drivers/Network/Gateway",
                _ => return anyhow::Error::new(error),
            };
            anyhow::Error::new(error).context(field)
        })?;
    }
    Ok(())
}
