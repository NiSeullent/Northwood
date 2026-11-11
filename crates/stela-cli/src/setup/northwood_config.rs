// SPDX-License-Identifier: GPL-2.0-or-later
//! Project setup's typed boot options into the shared Northwood schema.

use ::northwood_config::{
    self as config, NwConfig,
    plist::{self, Dictionary, Value},
    render,
};
use anyhow::{Context, Result, ensure};
use std::fmt::{Display, Write as _};
use stela_boot_core::options::{DEFAULT_WIFI_KEY, DRAM_BASE, System, is_chosen};
use stela_loader::mount_interface::{CHOSEN_DISK, Choices};
use stela_runtime::pc::storage::Target;

pub(super) const PATH: &str = "EFI/northwood/nwconfig.plist";

/// Encode setup's default and menu without writing to the ESP. The first menu
/// entry names the same options as the default, as in the legacy files.
pub(super) fn dictionary(default: &System, choices: Option<&Choices>) -> Result<Dictionary> {
    let mut document = options(default);
    let systems = choices
        .map(|choices| -> Result<_> {
            choices.encode().map_err(anyhow::Error::msg)?;
            choices
                .entries
                .iter()
                .map(|entry| {
                    System::parse(&entry.options)
                        .map_err(anyhow::Error::msg)
                        .with_context(|| format!("reading setup boot entry {:?}", entry.name))
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();
    if let Some(choices) = choices {
        ensure!(
            systems
                .first()
                .is_some_and(|first| *first == canonical(default)),
            "the first setup boot entry differs from the default"
        );
        let entries = choices
            .entries
            .iter()
            .zip(&systems)
            .map(|(entry, system)| {
                dict([
                    ("Name", string(&entry.name)),
                    (
                        "Overrides",
                        Value::Dictionary(differences(&document, options(system))),
                    ),
                ])
            })
            .collect();

        let Some(Value::Dictionary(boot)) = document.get_mut("Boot") else {
            unreachable!("options always creates a Boot dictionary");
        };
        boot.insert(
            "Timeout".into(),
            Value::Integer(choices.timeout_seconds.into()),
        );
        boot.insert("DefaultInstall".into(), string(&choices.entries[0].name));
        boot.insert("Entries".into(), Value::Array(entries));
    }
    document.insert(
        "Northwood".into(),
        dict([(
            "SchemaVersion",
            Value::Integer(config::SCHEMA_VERSION.into()),
        )]),
    );

    // Validate the same bytes that setup would publish, including the
    // firmware parser's size and nesting limits.
    let parsed = config::parse(plist::to_xml(&document).as_bytes())
        .context("validating setup nwconfig.plist")?;
    check_options(default, &parsed.config).context("checking the default setup boot entry")?;
    ensure!(
        systems.len() == parsed.config.boot.entries.len(),
        "setup boot entry count changed in nwconfig.plist"
    );
    for (original, entry) in systems.iter().zip(&parsed.config.boot.entries) {
        check_options(original, &entry.config)
            .with_context(|| format!("checking setup boot entry {:?}", entry.name))?;
    }
    Ok(document)
}

fn canonical(system: &System) -> System {
    let mut options = system.clone();
    if options.ssid.is_some() && options.wifi_key.is_none() {
        options.wifi_key = Some(DEFAULT_WIFI_KEY.into());
    }
    if options.native_rounds || options.native_bootstrap_check || options.native_shared_check {
        options.parallel = false;
    }
    options
}

fn check_options(system: &System, parsed: &NwConfig) -> Result<()> {
    let mut expected = canonical(system);
    // The loader recovers machine identity, restore policy and setup's
    // profiling preference from the exact matching legacy source.
    expected.ecid = None;
    expected.development_restore = false;
    expected.profile = false;
    let line = render::options_line(parsed, true)?;
    let actual = System::parse(&line).map_err(anyhow::Error::msg)?;
    ensure!(
        actual == expected,
        "setup system options cannot be represented by nwconfig.plist"
    );
    Ok(())
}

fn differences(base: &Dictionary, entry: Dictionary) -> Dictionary {
    let mut overrides = Dictionary::new();
    for (name, value) in entry {
        match (base.get(&name), value) {
            (Some(Value::Dictionary(old)), Value::Dictionary(new)) => {
                let nested = differences(old, new);
                if !nested.is_empty() {
                    overrides.insert(name, Value::Dictionary(nested));
                }
            }
            (Some(old), new) if *old == new => {}
            (_, new) => {
                overrides.insert(name, new);
            }
        }
    }
    overrides
}

/// Apply a setup update to generated leaves the user has left unchanged.
/// Startup entries keep their names as identities across reordering.
pub(super) fn merge_generated(
    previous: &Dictionary,
    next: Dictionary,
    current: Dictionary,
) -> Dictionary {
    merge_dictionary(previous, next, current, Scope::Root)
}

#[derive(Clone, Copy)]
enum Scope {
    Root,
    Boot,
    Entries,
    Other,
}

impl Scope {
    fn child(self, key: &str) -> Self {
        match (self, key) {
            (Self::Root, "Boot") => Self::Boot,
            (Self::Boot, "Entries") => Self::Entries,
            _ => Self::Other,
        }
    }
}

fn merge_dictionary(
    previous: &Dictionary,
    mut next: Dictionary,
    mut current: Dictionary,
    scope: Scope,
) -> Dictionary {
    for (key, old) in previous {
        let replacement = next.remove(key);
        let existing = current.remove(key);
        if let Some(value) = merge_value(Some(old), replacement, existing, scope.child(key)) {
            current.insert(key.clone(), value);
        }
    }
    for (key, value) in next {
        let existing = current.remove(&key);
        if let Some(value) = merge_value(None, Some(value), existing, scope.child(&key)) {
            current.insert(key, value);
        }
    }
    current
}

fn merge_value(
    previous: Option<&Value>,
    next: Option<Value>,
    current: Option<Value>,
    scope: Scope,
) -> Option<Value> {
    let Some(current) = current else { return next };
    if Some(&current) == previous {
        return next;
    }
    match current {
        Value::Dictionary(current)
            if previous.is_none_or(|value| matches!(value, Value::Dictionary(_)))
                && next
                    .as_ref()
                    .is_none_or(|value| matches!(value, Value::Dictionary(_))) =>
        {
            let empty = Dictionary::new();
            let previous = match previous {
                Some(Value::Dictionary(previous)) => previous,
                _ => &empty,
            };
            let (has_next, next) = match next {
                Some(Value::Dictionary(next)) => (true, next),
                _ => (false, Dictionary::new()),
            };
            let dictionary = merge_dictionary(previous, next, current, scope);
            (has_next || !dictionary.is_empty()).then_some(Value::Dictionary(dictionary))
        }
        Value::Array(current)
            if matches!(scope, Scope::Entries)
                && previous.is_none_or(|value| matches!(value, Value::Array(_)))
                && next
                    .as_ref()
                    .is_none_or(|value| matches!(value, Value::Array(_))) =>
        {
            let previous = match previous {
                Some(Value::Array(previous)) => previous.as_slice(),
                _ => &[],
            };
            let next = match next {
                Some(Value::Array(next)) => next,
                _ => Vec::new(),
            };
            Some(Value::Array(merge_entries(previous, next, current)))
        }
        current => Some(current),
    }
}

fn merge_entries(previous: &[Value], next: Vec<Value>, mut current: Vec<Value>) -> Vec<Value> {
    let mut entries = Vec::new();
    for entry in next {
        let Some(name) = entry_name(&entry) else {
            entries.push(entry);
            continue;
        };
        if named_entry(&entries, name).is_some() {
            continue;
        }
        let old = named_entry(previous, name);
        let existing = current
            .iter()
            .position(|item| entry_name(item) == Some(name))
            .map(|index| current.remove(index));
        current.retain(|item| entry_name(item) != Some(name));
        if let Some(entry) = merge_value(old, Some(entry), existing, Scope::Other) {
            entries.push(entry);
        }
    }
    for entry in current {
        if let Some(name) = entry_name(&entry)
            && (named_entry(&entries, name).is_some()
                || named_entry(previous, name) == Some(&entry))
        {
            continue;
        }
        entries.push(entry);
    }
    entries
}

fn named_entry<'a>(entries: &'a [Value], name: &str) -> Option<&'a Value> {
    entries.iter().find(|entry| entry_name(entry) == Some(name))
}

fn entry_name(entry: &Value) -> Option<&str> {
    let Value::Dictionary(entry) = entry else {
        return None;
    };
    let Value::String(name) = entry.get("Name")? else {
        return None;
    };
    Some(name)
}

fn options(system: &System) -> Dictionary {
    let boot = dict([
        ("Bundle", optional(system.bundle_directory.as_deref())),
        ("Preboot", target(system.preboot.as_ref())),
        ("Install", optional(system.install.as_deref())),
        ("InstallSource", optional(system.install_source)),
        ("VerifyOnRead", flag(system.verify_on_read)),
        ("UpdateRestore", flag(system.update_restore)),
        ("SingleUser", flag(system.single_user)),
        ("MaintenanceWrite", flag(system.maintenance_write)),
    ]);
    let wifi_key = system
        .ssid
        .as_ref()
        .map(|_| system.wifi_key.as_deref().unwrap_or(DEFAULT_WIFI_KEY));
    let wifi = dict([
        ("SSID", optional(system.ssid.as_deref())),
        ("KeyFile", optional(wifi_key)),
        ("Country", optional(system.wifi_country.as_deref())),
    ]);
    let network = dict([
        ("WiFi", wifi),
        ("IPv4", optional(system.ipv4)),
        ("Gateway", optional(system.ipv4_gateway)),
        ("DNS", string_array(&system.dns)),
    ]);
    let drivers = dict([
        ("Input", flag(system.input)),
        ("GuestNetwork", flag(system.guest_network)),
        ("ExternalDisks", flag(system.external_disks)),
        ("UsbStorage", string_array(&system.usb_storage)),
        ("Usb4Approve", string_array(&system.usb4_approved)),
        ("Network", network),
    ]);
    let graphics = dict([("ScanoutMirror", flag(system.scanout_mirror))]);
    let memory = dict([(
        "StorageRamBytes",
        integer(system.storage_ram.unwrap_or(0).into()),
    )]);
    let guest = dict([
        ("Storage", target(system.storage.as_ref())),
        ("Nor", target(system.nor.as_ref())),
        ("Cores", integer(system.native_cores.unwrap_or(0) as i128)),
        ("PEs", integer(system.guest_pes.unwrap_or(0) as i128)),
        ("Parallel", flag(system.parallel)),
        ("IdleSkip", flag(system.idle_skip)),
        ("NativeRounds", flag(system.native_rounds)),
    ]);
    let misc = dict([
        ("BootLog", dict([("Volume", flag(!system.no_volume_log))])),
        ("Debug", debug_options(system)),
    ]);
    fields([
        ("Boot", boot),
        ("Drivers", drivers),
        ("Graphics", graphics),
        ("Memory", memory),
        ("Guest", guest),
        ("Misc", misc),
    ])
}

fn debug_options(system: &System) -> Value {
    dict([
        ("Survey", flag(system.survey)),
        ("SepObserve", flag(system.sep_observe)),
        ("MonitorTrace", flag(system.monitor_trace)),
        ("DisplayTrace", flag(system.display_trace)),
        (
            "Break",
            array(&system.breakpoints, |address| integer(i128::from(*address))),
        ),
        (
            "Trace",
            array(&system.traces, |address| integer(i128::from(*address))),
        ),
        (
            "Watch",
            array(&system.watches, |(base, bytes)| {
                string(format!("{base:#x},{bytes:#x}"))
            }),
        ),
        (
            "Dump",
            array(&system.dumps, |(base, bytes)| {
                string(format!("{base},{bytes:#x}"))
            }),
        ),
        ("NetworkCheck", optional(system.network_check.as_deref())),
        (
            "AudioCheck",
            optional(system.audio_check.map(|mode| mode.name())),
        ),
        ("AudioOwnHpet", flag(system.audio_own_hpet)),
        ("OwnIoapic", flag(system.native_pci_own_ioapic)),
        ("NativeBootstrapCheck", flag(system.native_bootstrap_check)),
        ("NativeSharedCheck", flag(system.native_shared_check)),
        ("NativeArtifacts", flag(system.native_artifacts)),
        ("Raw", string(raw_options(system))),
    ])
}

fn raw_options(system: &System) -> String {
    let mut line = String::new();
    for (flag, value, set) in [
        ("--dram-base", system.dram_base, system.dram_explicit),
        ("--load", system.load, system.load != DRAM_BASE),
        ("--entry", system.entry, system.entry != DRAM_BASE),
        ("--x0", system.x0, system.x0 != 0),
        ("--budget", system.budget, system.budget != 0),
    ] {
        if set {
            let _ = write!(line, " {flag} {value:#x}");
        }
    }
    if system.restore {
        line.push_str(" --restore-study");
    }
    if let Some(target) = &system.restore_source {
        let _ = write!(
            line,
            " --restore-source {}:{}",
            target.serial, target.partition
        );
    }
    if let Some(manifest) = &system.restore_manifest_path {
        let _ = write!(line, " --restore-manifest {manifest}");
    }
    if let Some(identity) = system.restore_identity {
        let _ = write!(line, " --restore-identity {identity}");
    }
    line.trim_start().to_owned()
}

fn fields<const N: usize>(entries: [(&str, Value); N]) -> Dictionary {
    entries
        .into_iter()
        .map(|(key, value)| (key.into(), value))
        .collect()
}

fn dict<const N: usize>(entries: [(&str, Value); N]) -> Value {
    Value::Dictionary(fields(entries))
}

fn string(value: impl Display) -> Value {
    Value::String(value.to_string())
}

fn optional(value: Option<impl Display>) -> Value {
    Value::String(value.map(|value| value.to_string()).unwrap_or_default())
}

fn flag(value: bool) -> Value {
    Value::Boolean(value)
}

fn integer(value: i128) -> Value {
    Value::Integer(value)
}

fn array<T>(values: &[T], encode: impl FnMut(&T) -> Value) -> Value {
    Value::Array(values.iter().map(encode).collect())
}

fn string_array<T: Display>(values: &[T]) -> Value {
    array(values, |value| Value::String(value.to_string()))
}

fn target(target: Option<&Target>) -> Value {
    Value::String(match target {
        Some(target) if is_chosen(target) => CHOSEN_DISK.into(),
        Some(target) => format!("{}:{}", target.serial, target.partition),
        None => String::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use stela_loader::mount_interface::Choice;

    const INSTALL: &str = "--system --bundle stela/boot/fixture --storage @disk --nor @disk --install stela/install/fixture/payload.json --install-source c1d5d644-f9b9-11f0-8a7f-8f97d8adcd4f --development-restore --verify-on-read --no-volume-log --wifi Test%20%25%20Network --wifi-country KR --ecid 0x1234 --input";
    const PREBOOT: &str = "--system --preboot USB:DEVICE:667cb4d7-f8ac-43bb-a744-f5cd657908b1 --storage USB:DEVICE:667cb4d7-f8ac-43bb-a744-f5cd657908b1 --guest-pes 4 --ecid 0x1234";

    #[test]
    fn serializes_default_and_install_choice_without_local_identity() {
        let default = System::parse(PREBOOT).unwrap();
        let choices = Choices {
            timeout_seconds: 12,
            entries: vec![
                Choice {
                    name: "macOS".into(),
                    options: PREBOOT.into(),
                },
                Choice {
                    name: "Install macOS".into(),
                    options: INSTALL.into(),
                },
            ],
        };
        let xml = plist::to_xml(&dictionary(&default, Some(&choices)).unwrap());
        assert!(!xml.contains("Ecid"));
        assert!(!xml.contains("0x1234"));
        assert!(!xml.contains("DevelopmentRestore"));
        let parsed = config::parse(xml.as_bytes()).unwrap().config;
        assert_eq!(parsed.boot.timeout, config::BootTimeout::Seconds(12));
        assert_eq!(parsed.boot.default_install, "macOS");
        assert!(!parsed.boot.entries[0].config.drivers.input);
        let entry = &parsed.boot.entries[1];
        assert_eq!(entry.name, "Install macOS");
        assert!(entry.config.drivers.input);
        assert_eq!(entry.config.guest.storage, "@disk");
        assert_eq!(entry.config.guest.nor, "@disk");
        assert!(entry.config.boot.preboot.is_empty());
        assert_eq!(entry.config.guest.pes, 0);
        assert!(!entry.config.misc.boot_log.volume);
        assert_eq!(entry.config.drivers.network.wifi.ssid, "Test % Network");
        assert_eq!(entry.config.drivers.network.wifi.key_file, DEFAULT_WIFI_KEY);
        let line = render::options_line(&entry.config, false).unwrap();
        let actual = System::parse(&line).unwrap();
        assert!(is_chosen(actual.storage.as_ref().unwrap()));
        assert_eq!(actual.wifi_key.as_deref(), Some(DEFAULT_WIFI_KEY));
        assert_eq!(actual.wifi_country.as_deref(), Some("KR"));
        assert!(actual.no_volume_log);
        assert!(actual.verify_on_read);
        assert!(actual.input);
        assert!(actual.ecid.is_none());
        assert!(!actual.development_restore);
    }

    #[test]
    fn rejects_a_menu_that_disagrees_with_its_default() {
        let default = System::parse(PREBOOT).unwrap();
        let choices = Choices {
            timeout_seconds: 0,
            entries: vec![Choice {
                name: "Install macOS".into(),
                options: INSTALL.into(),
            }],
        };
        assert!(dictionary(&default, Some(&choices)).is_err());
    }

    #[test]
    fn refuses_to_drop_transient_or_unrepresentable_flags() {
        let mut options = System::parse(PREBOOT).unwrap();
        options.safe_boot = true;
        assert!(dictionary(&options, None).is_err());
        let zero_ram = System::parse("--system --storage-ram 0").unwrap();
        assert!(dictionary(&zero_ram, None).is_err());
    }

    #[test]
    fn writes_explicit_ipv4_dns_and_diagnostics_with_their_original_values() {
        let options = System::parse(
            "--system --bundle stela/boot/fixture --ipv4 192.168.10.4/24 --ipv4-gateway 192.168.10.1 --dns 1.1.1.1,2606:4700:4700::1111 --trace 0xabcdef --break 0x1234 --watch 0x1000,0x20 --load 0x800004000",
        ).unwrap();
        let document = dictionary(&options, None).unwrap();
        let parsed = config::from_dictionary(document).unwrap();
        let line = render::options_line(&parsed.config, true).unwrap();
        assert_eq!(System::parse(&line).unwrap(), options);
    }

    #[test]
    fn keeps_setup_profile_in_the_legacy_options() {
        let options = System::parse("--system --bundle stela/boot/fixture --profile").unwrap();
        let document = dictionary(&options, None).unwrap();
        let parsed = config::from_dictionary(document).unwrap();
        let line = render::options_line(&parsed.config, false).unwrap();
        assert!(options.profile);
        assert!(!System::parse(&line).unwrap().profile);
    }

    #[test]
    fn merges_generated_leaves_without_replacing_authored_or_future_fields() {
        let current = fields([
            ("Northwood", dict([("AutoProfile", flag(false))])),
            (
                "Memory",
                dict([
                    ("StorageRamBytes", integer(3)),
                    ("Swap", dict([("SizeGiB", integer(8))])),
                ]),
            ),
            (
                "Misc",
                dict([(
                    "BootLog",
                    dict([("Volume", flag(true)), ("Esp", flag(false))]),
                )]),
            ),
            ("Future", string("untouched")),
        ]);
        let previous = fields([
            ("Memory", dict([("StorageRamBytes", integer(2))])),
            ("Misc", dict([("BootLog", dict([("Volume", flag(true))]))])),
        ]);
        let next = fields([
            ("Northwood", dict([("SchemaVersion", integer(1))])),
            ("Memory", dict([("StorageRamBytes", integer(4))])),
            ("Misc", dict([("BootLog", dict([("Volume", flag(false))]))])),
        ]);
        let expected = fields([
            (
                "Northwood",
                dict([("SchemaVersion", integer(1)), ("AutoProfile", flag(false))]),
            ),
            (
                "Memory",
                dict([
                    ("StorageRamBytes", integer(3)),
                    ("Swap", dict([("SizeGiB", integer(8))])),
                ]),
            ),
            (
                "Misc",
                dict([(
                    "BootLog",
                    dict([("Volume", flag(false)), ("Esp", flag(false))]),
                )]),
            ),
            ("Future", string("untouched")),
        ]);
        assert_eq!(merge_generated(&previous, next, current), expected);
    }

    #[test]
    fn merges_entries_by_name_and_keeps_authored_overrides() {
        fn entries(values: Vec<Value>) -> Dictionary {
            fields([("Boot", dict([("Entries", Value::Array(values))]))])
        }
        fn entry(name: &str, overrides: Value) -> Value {
            dict([("Name", string(name)), ("Overrides", overrides)])
        }
        let a = entry(
            "A",
            dict([("Guest", dict([("Cores", integer(2)), ("PEs", integer(8))]))]),
        );
        let b = entry("B", dict([]));
        let c = entry(
            "C",
            dict([("Graphics", dict([("ScanoutMirror", flag(true))]))]),
        );
        let authored_a = entry(
            "A",
            dict([(
                "Guest",
                dict([
                    ("Cores", integer(2)),
                    ("PEs", integer(8)),
                    ("BootArgs", string("-v")),
                ]),
            )]),
        );
        let authored_c = entry(
            "C",
            dict([
                ("Graphics", dict([("ScanoutMirror", flag(true))])),
                ("Memory", dict([("Swap", dict([("SizeGiB", integer(8))]))])),
            ]),
        );
        let custom = entry("Custom", dict([("Drivers", dict([("Input", flag(true))]))]));
        let added = entry("Added", dict([]));
        let new_a = entry("A", dict([("Guest", dict([("Cores", integer(4))]))]));
        let expected_a = entry(
            "A",
            dict([(
                "Guest",
                dict([("Cores", integer(4)), ("BootArgs", string("-v"))]),
            )]),
        );
        let previous = entries(vec![a, b.clone(), c]);
        let next = entries(vec![added.clone(), new_a]);
        let current = entries(vec![
            b,
            authored_a,
            authored_c.clone(),
            custom.clone(),
            custom.clone(),
        ]);
        let expected = entries(vec![added, expected_a, authored_c, custom]);
        assert_eq!(merge_generated(&previous, next, current), expected);
    }
}
