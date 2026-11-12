// SPDX-License-Identifier: GPL-2.0-or-later
//! Read-only views of an existing OpenCore plist for the hint importer.
use anyhow::{Context, Result, anyhow};
use plist::{Dictionary, Value};
use std::path::Path;

/// The GUID under `NVRAM/Add` that holds `boot-args` and `csr-active-config`.
pub const APPLE_BOOT_GUID: &str = "7C436110-AB2A-4BBB-A880-FE41995C9F82";

/// The identity fields OpenCore's `PlatformInfo` may hold, under `Generic`
/// and in the hand-written `DataHub`, `PlatformNVRAM` and `SMBIOS` tables.
/// A published profile carries none of them.
const IDENTITY_FIELDS: [&str; 7] = [
    "SystemSerialNumber",
    "MLB",
    "SystemUUID",
    "ROM",
    "BoardSerialNumber",
    "ChassisSerialNumber",
    "ProcessorSerial",
];

/// Marker used while inspecting the input. The original values are never exported.
pub const REDACTED: &str = "NORTHWOOD-REDACTED";

/// A config.plist as OpenCore reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub root: Dictionary,
}

/// One `Kernel/Add` entry.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KextEntry {
    pub bundle_path: String,
    pub enabled: bool,
    pub executable_path: String,
    pub plist_path: String,
    pub min_kernel: String,
    pub max_kernel: String,
    pub comment: String,
}

/// One `ACPI/Add` entry.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AcpiEntry {
    pub path: String,
    pub enabled: bool,
    pub comment: String,
}

/// One `UEFI/Drivers` entry, in either of OpenCore's two forms (a string
/// before 0.7.7, a dictionary since).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DriverEntry {
    pub path: String,
    pub enabled: bool,
    pub arguments: String,
    pub comment: String,
}

fn string(dictionary: &Dictionary, key: &str) -> String {
    dictionary
        .get(key)
        .and_then(Value::as_string)
        .unwrap_or_default()
        .to_string()
}

fn boolean(dictionary: &Dictionary, key: &str) -> bool {
    dictionary
        .get(key)
        .and_then(Value::as_boolean)
        .unwrap_or(false)
}

impl Config {
    /// Read `path`, an XML or binary plist whose root is a dictionary.
    pub fn read(path: &Path) -> Result<Self> {
        let value =
            Value::from_file(path).with_context(|| format!("{} is not a plist", path.display()))?;
        let root = value
            .into_dictionary()
            .ok_or_else(|| anyhow!("{} is not a dictionary plist", path.display()))?;
        Ok(Self { root })
    }

    #[cfg(test)]
    pub fn write(&self, path: &Path) -> Result<()> {
        Value::Dictionary(self.root.clone())
            .to_file_xml(path)
            .with_context(|| format!("could not write {}", path.display()))
    }

    fn section(&self, name: &str) -> Option<&Dictionary> {
        self.root.get(name).and_then(Value::as_dictionary)
    }

    fn array(&self, section: &str, key: &str) -> &[Value] {
        self.section(section)
            .and_then(|section| section.get(key))
            .and_then(Value::as_array)
            .map_or(&[], Vec::as_slice)
    }

    pub fn enabled_acpi(&self, key: &str) -> impl Iterator<Item = &Dictionary> {
        self.array("ACPI", key)
            .iter()
            .filter_map(Value::as_dictionary)
            .filter(|entry| boolean(entry, "Enabled"))
    }

    pub fn active_quirks(&self, section: &str) -> usize {
        self.section(section)
            .and_then(|section| section.get("Quirks"))
            .and_then(Value::as_dictionary)
            .map_or(0, |quirks| {
                quirks
                    .values()
                    .filter(|value| value.as_boolean() == Some(true))
                    .count()
            })
    }

    /// `Kernel/Add`, in order.
    pub fn kexts(&self) -> Vec<KextEntry> {
        self.array("Kernel", "Add")
            .iter()
            .filter_map(Value::as_dictionary)
            .map(|entry| KextEntry {
                bundle_path: string(entry, "BundlePath"),
                enabled: boolean(entry, "Enabled"),
                executable_path: string(entry, "ExecutablePath"),
                plist_path: string(entry, "PlistPath"),
                min_kernel: string(entry, "MinKernel"),
                max_kernel: string(entry, "MaxKernel"),
                comment: string(entry, "Comment"),
            })
            .collect()
    }

    /// `ACPI/Add`, in order.
    pub fn acpi_add(&self) -> Vec<AcpiEntry> {
        self.array("ACPI", "Add")
            .iter()
            .filter_map(Value::as_dictionary)
            .map(|entry| AcpiEntry {
                path: string(entry, "Path"),
                enabled: boolean(entry, "Enabled"),
                comment: string(entry, "Comment"),
            })
            .collect()
    }

    /// `UEFI/Drivers`, in order.
    pub fn drivers(&self) -> Vec<DriverEntry> {
        self.array("UEFI", "Drivers")
            .iter()
            .filter_map(|entry| match entry {
                Value::String(path) => Some(DriverEntry {
                    path: path.clone(),
                    enabled: true,
                    arguments: String::new(),
                    comment: String::new(),
                }),
                Value::Dictionary(entry) => Some(DriverEntry {
                    path: string(entry, "Path"),
                    enabled: boolean(entry, "Enabled"),
                    arguments: string(entry, "Arguments"),
                    comment: string(entry, "Comment"),
                }),
                _ => None,
            })
            .collect()
    }

    /// `DeviceProperties/Add`: each device path's properties, values as
    /// text (`data:` plus hex for data, the number or string otherwise).
    pub fn device_properties(&self) -> Vec<(String, Vec<(String, String)>)> {
        self.section("DeviceProperties")
            .and_then(|section| section.get("Add"))
            .and_then(Value::as_dictionary)
            .map(|add| {
                add.iter()
                    .map(|(path, properties)| {
                        let properties = properties
                            .as_dictionary()
                            .map(|properties| {
                                properties
                                    .iter()
                                    .map(|(key, value)| (key.clone(), text(value)))
                                    .collect()
                            })
                            .unwrap_or_default();
                        (path.clone(), properties)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The NVRAM variable `name` under the Apple boot GUID, as text.
    pub fn apple_nvram(&self, name: &str) -> Option<String> {
        self.section("NVRAM")?
            .get("Add")?
            .as_dictionary()?
            .get(APPLE_BOOT_GUID)?
            .as_dictionary()?
            .get(name)
            .map(text)
    }

    /// `boot-args`, split on spaces.
    pub fn boot_args(&self) -> Vec<String> {
        self.apple_nvram("boot-args")
            .unwrap_or_default()
            .split_whitespace()
            .map(str::to_string)
            .collect()
    }

    /// `PlatformInfo/Generic/SystemProductName`, or the SMBIOS table's.
    pub fn model(&self) -> Option<String> {
        let platform = self.section("PlatformInfo")?;
        for table in ["Generic", "SMBIOS"] {
            if let Some(name) = platform
                .get(table)
                .and_then(Value::as_dictionary)
                .and_then(|table| table.get("SystemProductName"))
                .and_then(Value::as_string)
                .filter(|name| !name.is_empty())
            {
                return Some(name.to_string());
            }
        }
        None
    }

    /// Replace every identity field under `PlatformInfo` with [`REDACTED`]
    /// and return the `Table/Field` names replaced. A field that is empty
    /// already is left alone and not listed.
    pub fn redact_identity(&mut self) -> Vec<String> {
        let mut redacted = Vec::new();
        let Some(Value::Dictionary(platform)) = self.root.get_mut("PlatformInfo") else {
            return redacted;
        };
        for (table, value) in platform.iter_mut() {
            let Value::Dictionary(table_values) = value else {
                continue;
            };
            for field in IDENTITY_FIELDS {
                let Some(current) = table_values.get_mut(field) else {
                    continue;
                };
                let filled = match &*current {
                    Value::String(text) => !text.is_empty() && text != REDACTED,
                    Value::Data(bytes) => !bytes.is_empty() && bytes != REDACTED.as_bytes(),
                    _ => true,
                };
                if !filled {
                    continue;
                }
                *current = match &*current {
                    Value::Data(_) => Value::Data(REDACTED.as_bytes().to_vec()),
                    _ => Value::String(REDACTED.to_string()),
                };
                redacted.push(format!("{table}/{field}"));
            }
        }
        redacted
    }
}

/// A plist value as one line of text.
pub fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Data(bytes) => format!("data:{}", hex::encode(bytes)),
        Value::Boolean(flag) => flag.to_string(),
        Value::Integer(number) => number.to_string(),
        Value::Real(number) => number.to_string(),
        Value::Array(items) => format!("[{} items]", items.len()),
        Value::Dictionary(items) => format!("{{{} keys}}", items.len()),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Config {
        let mut root = Dictionary::new();
        let mut kernel = Dictionary::new();
        let mut lilu = Dictionary::new();
        lilu.insert("BundlePath".into(), Value::String("Lilu.kext".into()));
        lilu.insert("Enabled".into(), Value::Boolean(true));
        lilu.insert(
            "ExecutablePath".into(),
            Value::String("Contents/MacOS/Lilu".into()),
        );
        lilu.insert("MinKernel".into(), Value::String("".into()));
        kernel.insert("Add".into(), Value::Array(vec![Value::Dictionary(lilu)]));
        root.insert("Kernel".into(), Value::Dictionary(kernel));
        let mut uefi = Dictionary::new();
        let mut runtime = Dictionary::new();
        runtime.insert("Path".into(), Value::String("OpenRuntime.efi".into()));
        runtime.insert("Enabled".into(), Value::Boolean(true));
        uefi.insert(
            "Drivers".into(),
            Value::Array(vec![
                Value::Dictionary(runtime),
                Value::String("HfsPlus.efi".into()),
            ]),
        );
        root.insert("UEFI".into(), Value::Dictionary(uefi));
        let mut platform = Dictionary::new();
        let mut generic = Dictionary::new();
        generic.insert(
            "SystemProductName".into(),
            Value::String("MacBookPro16,2".into()),
        );
        generic.insert(
            "SystemSerialNumber".into(),
            Value::String("C02ABCDEFGHI".into()),
        );
        generic.insert("MLB".into(), Value::String("".into()));
        generic.insert("ROM".into(), Value::Data(vec![1, 2, 3, 4, 5, 6]));
        platform.insert("Generic".into(), Value::Dictionary(generic));
        root.insert("PlatformInfo".into(), Value::Dictionary(platform));
        let mut nvram = Dictionary::new();
        let mut add = Dictionary::new();
        let mut apple = Dictionary::new();
        apple.insert("boot-args".into(), Value::String("-v keepsyms=1".into()));
        add.insert(APPLE_BOOT_GUID.into(), Value::Dictionary(apple));
        nvram.insert("Add".into(), Value::Dictionary(add));
        root.insert("NVRAM".into(), Value::Dictionary(nvram));
        Config { root }
    }

    #[test]
    fn the_typed_views_read_both_driver_forms_and_the_boot_arguments() {
        let config = sample();
        assert_eq!(config.kexts()[0].bundle_path, "Lilu.kext");
        assert!(config.kexts()[0].enabled);
        let drivers = config.drivers();
        assert_eq!(
            drivers.iter().map(|d| d.path.as_str()).collect::<Vec<_>>(),
            ["OpenRuntime.efi", "HfsPlus.efi"]
        );
        assert!(drivers.iter().all(|d| d.enabled));
        assert_eq!(config.boot_args(), ["-v", "keepsyms=1"]);
        assert_eq!(config.model().as_deref(), Some("MacBookPro16,2"));
        assert!(config.acpi_add().is_empty());
    }

    #[test]
    fn redaction_replaces_filled_identity_fields_only_and_keeps_the_model() {
        let mut config = sample();
        let redacted = config.redact_identity();
        assert_eq!(redacted, ["Generic/SystemSerialNumber", "Generic/ROM"]);
        let generic = config.root["PlatformInfo"].as_dictionary().unwrap()["Generic"]
            .as_dictionary()
            .unwrap();
        assert_eq!(generic["SystemSerialNumber"].as_string(), Some(REDACTED));
        assert_eq!(generic["ROM"].as_data(), Some(REDACTED.as_bytes()));
        assert_eq!(generic["MLB"].as_string(), Some(""));
        assert_eq!(config.model().as_deref(), Some("MacBookPro16,2"));
        // Twice changes nothing more.
        assert!(config.redact_identity().is_empty());
    }

    #[test]
    fn a_config_survives_a_write_and_read_unchanged() {
        let config = sample();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.plist");
        config.write(&path).unwrap();
        let again = Config::read(&path).unwrap();
        assert_eq!(again, config);
    }
}
