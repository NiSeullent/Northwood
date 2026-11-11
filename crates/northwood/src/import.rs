// SPDX-License-Identifier: GPL-2.0-or-later
//! Convert selected settings into host-side hints for manual review.
use crate::oc::Config;
use anyhow::{Context, Result};
use plist::{Dictionary, Value};
use serde::Serialize;
use std::{
    collections::BTreeSet,
    fs, io,
    path::{Path, PathBuf},
};

const TOUCHPAD: &str = "SYNA2BA6";
const DEFAULT_POLL_INTERVAL_US: u64 = 8_000;

#[derive(Serialize)]
struct Report {
    schema_version: u8,
    acpi_add_entries: usize,
    enabled_acpi_patches: usize,
    enabled_acpi_deletes: usize,
    kext_entries: usize,
    uefi_driver_entries: usize,
    device_property_paths: usize,
    enabled_kernel_quirks: usize,
    enabled_booter_quirks: usize,
    has_model: bool,
    identity_fields_redacted: usize,
    local_efi_entries: Option<[usize; 3]>,
    hints: Vec<&'static str>,
    notes: Vec<&'static str>,
}

fn inventory(directory: &Path) -> Result<[usize; 3]> {
    let mut counts = [0; 3];
    for (index, name) in ["ACPI", "Kexts", "Drivers"].iter().enumerate() {
        match fs::read_dir(directory.join(name)) {
            Ok(entries) => {
                for entry in entries {
                    let ty = entry?.file_type()?;
                    if ty.is_file() || ty.is_dir() {
                        counts[index] += 1;
                    }
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(counts)
}

fn dictionary(entries: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    let mut values = Dictionary::new();
    for (key, value) in entries {
        values.insert(key.into(), value);
    }
    Value::Dictionary(values)
}

fn touchpad_hint(kind: &'static str, value: u64, comment: &'static str) -> Value {
    dictionary([
        ("Match", Value::String(TOUCHPAD.into())),
        ("Kind", Value::String(kind.into())),
        ("Value", Value::Integer(value.into())),
        ("Comment", Value::String(comment.into())),
    ])
}

fn touchpad_match(config: &Config) -> bool {
    config
        .enabled_acpi("Patch")
        .chain(config.enabled_acpi("Add"))
        .filter_map(|entry| entry.get("Comment").and_then(Value::as_string))
        .any(|comment| comment.to_ascii_uppercase().contains(TOUCHPAD))
        || config
            .acpi_add()
            .iter()
            .any(|entry| entry.enabled && entry.path.to_ascii_uppercase().contains(TOUCHPAD))
}

fn collect(config: &mut Config, local_efi_entries: Option<[usize; 3]>) -> (Value, Report) {
    let mut report = Report {
        schema_version: 1,
        acpi_add_entries: config.acpi_add().len(),
        enabled_acpi_patches: config.enabled_acpi("Patch").count(),
        enabled_acpi_deletes: config.enabled_acpi("Delete").count(),
        kext_entries: config.kexts().len(),
        uefi_driver_entries: config.drivers().len(),
        device_property_paths: config.device_properties().len(),
        enabled_kernel_quirks: config.active_quirks("Kernel"),
        enabled_booter_quirks: config.active_quirks("Booter"),
        has_model: config.model().is_some(),
        identity_fields_redacted: config.redact_identity().len(),
        local_efi_entries,
        hints: Vec::new(),
        notes: Vec::new(),
    };
    let mut root = Dictionary::new();
    root.insert(
        "Northwood".into(),
        dictionary([("SchemaVersion", Value::Integer(1u64.into()))]),
    );

    let args = config.boot_args();
    let polling = args.iter().any(|arg| arg == "-vi2c-force-polling");
    let touchpad = touchpad_match(config);
    let mut quirks = Vec::new();
    if touchpad {
        let has_sbfs = config.enabled_acpi("Patch").any(|entry| {
            entry.get("Replace").and_then(Value::as_data) == Some(b"SBFS")
                || entry
                    .get("Comment")
                    .and_then(Value::as_string)
                    .is_some_and(|text| text.to_ascii_uppercase().contains("SBFS"))
        });
        if has_sbfs {
            quirks.push(touchpad_hint(
                "I2cHidDescriptor",
                0x2c,
                "Candidate from the Lenovo 15IGL7 SYNA2BA6 SBFS workaround. Verify against host ACPI.",
            ));
            report.hints.push("I2cHidDescriptor");
        }
        if polling {
            quirks.push(touchpad_hint(
                "I2cHidForcePolling",
                DEFAULT_POLL_INTERVAL_US,
                "Candidate from -vi2c-force-polling and the SYNA2BA6 match.",
            ));
            report.hints.push("I2cHidForcePolling");
        }
    } else if polling {
        report
            .notes
            .push("Polling requested without an identified touchpad HID.");
    }
    if !quirks.is_empty() {
        root.insert(
            "ACPI".into(),
            dictionary([("Quirks", Value::Array(quirks))]),
        );
    }

    let mut swap_values = BTreeSet::new();
    let mut invalid_size = false;
    for size in args
        .iter()
        .filter_map(|arg| arg.strip_prefix("nmxswapreserve="))
    {
        match size.parse::<u64>() {
            Ok(value) if (1..=1024).contains(&value) => {
                swap_values.insert(value);
            }
            _ => invalid_size = true,
        }
    }
    if !invalid_size && swap_values.len() == 1 {
        if let Some(size) = swap_values.first() {
            root.insert(
                "Memory".into(),
                dictionary([(
                    "Swap",
                    dictionary([("SizeGiB", Value::Integer((*size).into()))]),
                )]),
            );
            report.hints.push("Memory/Swap/SizeGiB");
        }
    } else if invalid_size || swap_values.len() > 1 {
        report
            .notes
            .push("The source has conflicting or invalid nmxswapreserve values.");
    }
    (Value::Dictionary(root), report)
}

/// Resolve existing directory aliases before creating any missing directories.
fn resolved_location(path: &Path) -> io::Result<PathBuf> {
    let mut location = std::path::absolute(path)?;
    let mut missing = Vec::new();
    loop {
        match location.canonicalize() {
            Ok(mut resolved) => {
                resolved.extend(missing.into_iter().rev());
                return Ok(resolved);
            }
            Err(error)
                if error.kind() == io::ErrorKind::NotFound
                    && location
                        .symlink_metadata()
                        .is_err_and(|error| error.kind() == io::ErrorKind::NotFound) =>
            {
                missing.push(location.file_name().ok_or(error)?.to_os_string());
                location.pop();
            }
            Err(error) => return Err(error),
        }
    }
}

fn oc_tree(path: &Path) -> Option<&Path> {
    let named = |path: &Path, name: &[u8]| {
        path.file_name()
            .is_some_and(|part| part.as_encoded_bytes().eq_ignore_ascii_case(name))
    };
    path.ancestors().find(|path| {
        named(path, b"OC") && path.parent().is_some_and(|parent| named(parent, b"EFI"))
    })
}

fn output_location(config: &Path, efi: Option<&Path>, output: &Path) -> Result<PathBuf> {
    let requested = std::path::absolute(output)?;
    let resolved = resolved_location(&requested)
        .with_context(|| format!("resolving output directory {}", output.display()))?;
    let config = std::path::absolute(config)?;
    let protected = efi
        .into_iter()
        .chain(oc_tree(&config))
        .chain(oc_tree(&requested));
    let mut inside_oc = oc_tree(&resolved).is_some();
    for directory in protected {
        let directory = resolved_location(directory)
            .with_context(|| format!("resolving OpenCore directory {}", directory.display()))?;
        inside_oc |= resolved.starts_with(directory);
    }
    anyhow::ensure!(
        !inside_oc,
        "output directory must be outside the OpenCore input tree"
    );
    Ok(resolved)
}

pub(crate) fn write(config: &Path, efi: Option<&Path>, output: &Path) -> Result<[PathBuf; 2]> {
    let output = output_location(config, efi, output)?;
    let mut config = Config::read(config)?;
    let local_efi = efi.map(inventory).transpose()?;
    let (hints, report) = collect(&mut config, local_efi);
    let mut hint_bytes = Vec::new();
    hints.to_writer_xml(&mut hint_bytes)?;
    let mut report_bytes = serde_json::to_vec_pretty(&report)?;
    report_bytes.push(b'\n');
    fs::create_dir_all(&output).with_context(|| format!("creating {}", output.display()))?;
    let paths = [
        output.join("nwconfig-hints.plist"),
        output.join("report.json"),
    ];
    for path in &paths {
        match path.symlink_metadata() {
            Ok(_) => anyhow::bail!(
                "{} already exists; choose another output directory",
                path.display()
            ),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| format!("checking {}", path.display()));
            }
        }
    }
    for (path, bytes) in paths.iter().zip([hint_bytes, report_bytes]) {
        use io::Write;
        let mut temporary = tempfile::NamedTempFile::new_in(&output)?;
        temporary.write_all(&bytes)?;
        temporary
            .persist_noclobber(path)
            .with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oc::APPLE_BOOT_GUID;

    #[test]
    fn refuses_output_in_an_opencore_tree_before_creating_directories() {
        let scratch = tempfile::tempdir().unwrap();
        let config = scratch.path().join("config.plist");
        dictionary([]).to_file_xml(&config).unwrap();
        let source = fs::read(&config).unwrap();
        let inventory = scratch.path().join("open-core-input");
        fs::create_dir(&inventory).unwrap();

        for (efi, output) in [
            (Some(inventory.as_path()), inventory.join("results/new")),
            (None, scratch.path().join("efi/oc/results")),
        ] {
            let error = write(&config, efi, &output).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("outside the OpenCore input tree")
            );
            assert!(!output.parent().unwrap().exists());
        }
        assert_eq!(fs::read(&config).unwrap(), source);
    }

    #[cfg(unix)]
    #[test]
    fn refuses_output_through_an_alias_to_the_inventoried_tree() {
        let scratch = tempfile::tempdir().unwrap();
        let inventory = scratch.path().join("open-core-input");
        fs::create_dir(&inventory).unwrap();
        let config = inventory.join("config.plist");
        dictionary([]).to_file_xml(&config).unwrap();
        let alias = scratch.path().join("alias");
        std::os::unix::fs::symlink(&inventory, &alias).unwrap();

        let error = write(&config, Some(&inventory), &alias.join("results/new")).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("outside the OpenCore input tree")
        );
        assert!(!inventory.join("results").exists());
    }

    #[test]
    fn imports_allowlisted_hints_and_keeps_identifiers_private() {
        let value = dictionary([
            (
                "ACPI",
                dictionary([(
                    "Patch",
                    Value::Array(vec![dictionary([
                        ("Enabled", Value::Boolean(true)),
                        (
                            "Comment",
                            Value::String("SYNA2BA6 SBFS machine-name-private".into()),
                        ),
                    ])]),
                )]),
            ),
            (
                "PlatformInfo",
                dictionary([(
                    "Generic",
                    dictionary([(
                        "SystemSerialNumber",
                        Value::String("serial-number-private".into()),
                    )]),
                )]),
            ),
            (
                "NVRAM",
                dictionary([(
                    "Add",
                    dictionary([(
                        APPLE_BOOT_GUID,
                        dictionary([(
                            "boot-args",
                            Value::String(
                                "-vi2c-force-polling nmxswapreserve=32 key=secret".into(),
                            ),
                        )]),
                    )]),
                )]),
            ),
        ]);
        let mut config = Config {
            root: value.into_dictionary().unwrap(),
        };
        let (hints, report) = collect(&mut config, None);
        assert_eq!(
            report.hints,
            [
                "I2cHidDescriptor",
                "I2cHidForcePolling",
                "Memory/Swap/SizeGiB"
            ]
        );
        assert_eq!(report.identity_fields_redacted, 1);
        let mut bytes = Vec::new();
        hints.to_writer_xml(&mut bytes).unwrap();
        northwood_config::parse(&bytes).unwrap();
        for exported in [bytes, serde_json::to_vec(&report).unwrap()] {
            let output = String::from_utf8(exported).unwrap();
            for private in [
                "machine-name-private",
                "serial-number-private",
                "key=secret",
            ] {
                assert!(!output.contains(private));
            }
        }
    }
}
