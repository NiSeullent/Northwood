// SPDX-License-Identifier: GPL-2.0-or-later
use crate::plist::{self, Dictionary, Value};
use alloc::{collections::BTreeMap, format, string::String, vec::Vec};
use core::{
    fmt,
    net::{IpAddr, Ipv4Addr},
};

const GIB: u64 = 1 << 30;
pub const SCHEMA_VERSION: u32 = 1;
pub const MAX_CHOICES: usize = 35;
pub const MAX_CHOICE_NAME: usize = 40;
pub const MAX_TIMEOUT_SECONDS: u32 = 60;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Xml(plist::XmlError),
    Field { path: String, reason: &'static str },
}

impl Error {
    pub(crate) fn field(path: impl Into<String>, reason: &'static str) -> Self {
        Self::Field {
            path: path.into(),
            reason,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Xml(error) => error.fmt(f),
            Self::Field { path, reason } => write!(f, "{path}: {reason}"),
        }
    }
}

impl core::error::Error for Error {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parsed {
    pub config: NwConfig,
    /// Paths to keys this schema does not recognize.
    pub warnings: Vec<String>,
}

/// A strict parse for the host validator and firmware.
pub fn parse(bytes: &[u8]) -> Result<Parsed, Error> {
    from_dictionary(plist::parse(bytes).map_err(Error::Xml)?)
}

/// Parse a configuration assembled by setup with the same schema as the loader.
pub fn from_dictionary(dictionary: Dictionary) -> Result<Parsed, Error> {
    let mut base = dictionary;
    let entries = match base.get_mut("Boot") {
        Some(Value::Dictionary(boot)) => boot.remove("Entries"),
        _ => None,
    };
    let mut warnings = Vec::new();
    let mut config = NwConfig::read(base.clone(), &mut warnings)?;
    if let Some(entries) = entries {
        let mut field = Reader::new(
            String::from("Boot"),
            Dictionary::from([(String::from("Entries"), entries)]),
        );
        config.boot.entries = read_entries(field.dictionaries("Entries")?, &base, &mut warnings)?;
        if !config.boot.entries.is_empty()
            && !config.boot.default_install.is_empty()
            && !config
                .boot
                .entries
                .iter()
                .any(|entry| entry.name == config.boot.default_install)
        {
            return Err(Error::field(
                "Boot/DefaultInstall",
                "does not name an entry in Boot/Entries",
            ));
        }
    }
    Ok(Parsed { config, warnings })
}

/// A malformed configuration produces the entire default record and an error
/// for the caller to display. A valid file keeps its unknown-key warnings.
pub fn parse_or_failsafe(bytes: &[u8]) -> (Parsed, Option<Error>) {
    match parse(bytes) {
        Ok(parsed) => (parsed, None),
        Err(error) => (
            Parsed {
                config: NwConfig::default(),
                warnings: Vec::new(),
            },
            Some(error),
        ),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NwConfig {
    pub northwood: Northwood,
    pub acpi: Acpi,
    pub boot: Boot,
    pub drivers: Drivers,
    pub graphics: Graphics,
    pub memory: Memory,
    pub guest: Guest,
    pub misc: Misc,
    pub nvram: Nvram,
    pub platform_info: PlatformInfo,
    pub uefi: Uefi,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Northwood {
    pub schema_version: u32,
    pub hardware_profile: String,
    pub auto_profile: bool,
}

impl Default for Northwood {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            hardware_profile: String::new(),
            auto_profile: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Acpi {
    pub quirks: Vec<AcpiQuirk>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcpiQuirk {
    pub matcher: String,
    pub kind: String,
    pub value: Option<Value>,
    pub comment: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BootTimeout {
    #[default]
    Auto,
    Seconds(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootEntry {
    pub name: String,
    /// The effective config, with overrides applied to the top-level config.
    pub config: NwConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Boot {
    pub timeout: BootTimeout,
    pub entries: Vec<BootEntry>,
    pub default_install: String,
    pub install: String,
    pub install_source: String,
    pub verify_on_read: bool,
    pub update_restore: bool,
    pub single_user: bool,
    pub maintenance_write: bool,
    pub bundle: String,
    pub preboot: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DriverMode {
    #[default]
    Auto,
    Safe,
    Manual,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverBlock {
    pub matcher: String,
    pub comment: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverForce {
    pub module: String,
    pub matcher: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Drivers {
    pub mode: DriverMode,
    pub enable: Vec<String>,
    pub disable: Vec<String>,
    pub arguments: BTreeMap<String, String>,
    pub block: Vec<DriverBlock>,
    pub force: Vec<DriverForce>,
    pub input: bool,
    pub guest_network: bool,
    pub external_disks: bool,
    pub usb_storage: Vec<String>,
    pub usb4_approve: Vec<String>,
    pub network: Network,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Network {
    pub wifi: Wifi,
    pub ipv4: String,
    pub gateway: String,
    pub dns: Vec<IpAddr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wifi {
    pub ssid: String,
    pub key_file: String,
    pub country: String,
}

impl Default for Wifi {
    fn default() -> Self {
        Self {
            ssid: String::new(),
            key_file: String::from(crate::DEFAULT_WIFI_KEY),
            country: String::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphicsBackend {
    #[default]
    Auto,
    NvModern,
    I915,
    Xe,
    Northbridge,
    Gop,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Graphics {
    pub backend: GraphicsBackend,
    pub scanout_mirror: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryProfile {
    #[default]
    Auto,
    Standard,
    Compact,
    Paged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SwapLocation {
    #[default]
    Auto,
    Partition,
    VolumeFile,
    None,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Memory {
    pub profile: MemoryProfile,
    pub swap: Swap,
    pub paged: Paged,
    pub translation_store_bytes: u64,
    pub storage_ram_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Swap {
    pub enabled: bool,
    pub bytes: u64,
    pub location: SwapLocation,
    pub partition: String,
    pub volume_max_bytes: u64,
    pub encrypt: bool,
    pub guest_swap: bool,
}

impl Default for Swap {
    fn default() -> Self {
        Self {
            enabled: true,
            bytes: 32 * GIB,
            location: SwapLocation::Auto,
            partition: String::new(),
            volume_max_bytes: 8 * GIB,
            encrypt: true,
            guest_swap: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paged {
    pub reported_bytes: u64,
    pub heap_floor_bytes: u64,
}

impl Default for Paged {
    fn default() -> Self {
        Self {
            reported_bytes: 2 * GIB,
            heap_floor_bytes: GIB / 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Guest {
    pub storage: String,
    pub nor: String,
    pub cores: u64,
    pub pes: u64,
    pub parallel: bool,
    pub idle_skip: bool,
    pub native_rounds: bool,
    pub boot_args: String,
}

impl Default for Guest {
    fn default() -> Self {
        Self {
            storage: String::new(),
            nor: String::new(),
            cores: 0,
            pes: 0,
            parallel: true,
            idle_skip: false,
            native_rounds: false,
            boot_args: String::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Misc {
    pub debug: DebugOptions,
    pub boot_log: BootLog,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DebugOptions {
    pub host_check: bool,
    pub profile: bool,
    pub survey: bool,
    pub sep_observe: bool,
    pub monitor_trace: bool,
    pub display_trace: bool,
    pub breaks: Vec<String>,
    pub trace: Vec<String>,
    pub watch: Vec<String>,
    pub dump: Vec<String>,
    pub network_check: String,
    pub audio_check: String,
    pub audio_own_hpet: bool,
    pub own_ioapic: bool,
    pub native_bootstrap_check: bool,
    pub native_shared_check: bool,
    pub native_artifacts: bool,
    pub development_restore: bool,
    pub raw: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootLog {
    pub esp: bool,
    pub volume: bool,
}

impl Default for BootLog {
    fn default() -> Self {
        Self {
            esp: true,
            volume: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Nvram {
    pub add: Dictionary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformInfo {
    pub model: String,
}

impl Default for PlatformInfo {
    fn default() -> Self {
        Self {
            model: String::from("Mac14,3"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Uefi {
    pub connect_all: bool,
    pub reclaim_boot_services: bool,
    pub firmware_path: String,
}

impl Default for Uefi {
    fn default() -> Self {
        Self {
            connect_all: true,
            reclaim_boot_services: true,
            firmware_path: String::from(crate::DEFAULT_FIRMWARE_PATH),
        }
    }
}

struct Reader {
    path: String,
    fields: Dictionary,
}

impl Reader {
    fn new(path: String, fields: Dictionary) -> Self {
        Self { path, fields }
    }

    fn path(&self, key: &str) -> String {
        if self.path.is_empty() {
            String::from(key)
        } else {
            format!("{}/{key}", self.path)
        }
    }

    fn error(&self, key: &str, reason: &'static str) -> Error {
        Error::field(self.path(key), reason)
    }

    fn finish(self, warnings: &mut Vec<String>) {
        warnings.extend(self.fields.keys().map(|key| self.path(key)));
    }

    fn section(&mut self, key: &str) -> Result<Self, Error> {
        Ok(Self::new(self.path(key), self.dictionary(key)?))
    }

    fn dictionary(&mut self, key: &str) -> Result<Dictionary, Error> {
        match self.fields.remove(key) {
            Some(Value::Dictionary(fields)) => Ok(fields),
            None => Ok(Dictionary::new()),
            _ => Err(self.error(key, "expected a dictionary")),
        }
    }

    fn string(&mut self, key: &str, default: &str) -> Result<String, Error> {
        match self.fields.remove(key) {
            Some(Value::String(text)) => Ok(text),
            None => Ok(String::from(default)),
            _ => Err(self.error(key, "expected a string")),
        }
    }

    fn required_string(&mut self, key: &str) -> Result<String, Error> {
        let value = self.string(key, "")?;
        if value.trim().is_empty() {
            return Err(self.error(key, "expected a nonempty string"));
        }
        Ok(value)
    }

    fn boolean(&mut self, key: &str, default: bool) -> Result<bool, Error> {
        match self.fields.remove(key) {
            Some(Value::Boolean(value)) => Ok(value),
            None => Ok(default),
            _ => Err(self.error(key, "expected a boolean")),
        }
    }

    fn integer(&mut self, key: &str, default: u64) -> Result<u64, Error> {
        match self.fields.remove(key) {
            Some(Value::Integer(value)) => u64::try_from(value)
                .map_err(|_| self.error(key, "expected a nonnegative 64-bit integer")),
            None => Ok(default),
            _ => Err(self.error(key, "expected an integer")),
        }
    }

    fn array(&mut self, key: &str) -> Result<Vec<Value>, Error> {
        match self.fields.remove(key) {
            Some(Value::Array(values)) => Ok(values),
            None => Ok(Vec::new()),
            _ => Err(self.error(key, "expected an array")),
        }
    }

    fn strings(&mut self, key: &str) -> Result<Vec<String>, Error> {
        self.array(key)?
            .into_iter()
            .map(|value| match value {
                Value::String(text) if !text.trim().is_empty() => Ok(text),
                _ => Err(self.error(key, "expected an array of nonempty strings")),
            })
            .collect()
    }

    fn dictionaries(&mut self, key: &str) -> Result<Vec<Self>, Error> {
        self.array(key)?
            .into_iter()
            .enumerate()
            .map(|(index, value)| match value {
                Value::Dictionary(fields) => {
                    Ok(Self::new(format!("{}[{index}]", self.path(key)), fields))
                }
                _ => Err(self.error(key, "expected an array of dictionaries")),
            })
            .collect()
    }

    fn diagnostic_words(&mut self, key: &str, integers: bool) -> Result<Vec<String>, Error> {
        let values = match self.fields.remove(key) {
            None => return Ok(Vec::new()),
            Some(Value::Array(values)) => values,
            Some(value) => alloc::vec![value],
        };
        values
            .into_iter()
            .map(|value| match value {
                Value::String(text) if !text.is_empty() => Ok(text),
                Value::Integer(value) if integers && value >= 0 => Ok(format!("{value:#x}")),
                _ => Err(self.error(key, "invalid diagnostic value")),
            })
            .collect()
    }
}

const PLATFORM_IDENTITY_FIELDS: &[&str] = &[
    "Ecid",
    "SystemSerialNumber",
    "BoardSerialNumber",
    "ChassisSerialNumber",
    "ProcessorSerial",
    "MLB",
    "SystemUUID",
    "ROM",
];

// Recognize credential fields at their known locations, including the
// OpenCore dictionaries users may copy into this configuration.
const EXTERNAL_STATE_FIELDS: &[(&[&str], &[&str])] = &[
    (
        &["Drivers", "Network"],
        &[
            "Password",
            "Passphrase",
            "WiFiPassword",
            "WiFiPassphrase",
            "PSK",
            "PreSharedKey",
        ],
    ),
    (
        &["Drivers", "Network", "WiFi"],
        &[
            "Password",
            "Passphrase",
            "WiFiPassword",
            "WiFiPassphrase",
            "PSK",
            "PreSharedKey",
            "WpaPsk",
            "Key",
        ],
    ),
    (
        &["Misc", "Security"],
        &["PasswordHash", "PasswordSalt", "ApECID"],
    ),
    (&["PlatformInfo"], PLATFORM_IDENTITY_FIELDS),
    (&["PlatformInfo", "Generic"], PLATFORM_IDENTITY_FIELDS),
    (&["PlatformInfo", "SMBIOS"], PLATFORM_IDENTITY_FIELDS),
    (&["PlatformInfo", "DataHub"], PLATFORM_IDENTITY_FIELDS),
    (&["PlatformInfo", "NVRAM"], PLATFORM_IDENTITY_FIELDS),
    (&["PlatformInfo", "PlatformNVRAM"], PLATFORM_IDENTITY_FIELDS),
];

fn external_field(dictionary: &Dictionary, sections: &[&str], names: &[&str]) -> Option<String> {
    let Some((section, rest)) = sections.split_first() else {
        return dictionary
            .keys()
            .find(|key| names.iter().any(|name| key.eq_ignore_ascii_case(name)))
            .cloned();
    };
    dictionary.iter().find_map(|(key, value)| {
        if key.eq_ignore_ascii_case(section)
            && let Value::Dictionary(dictionary) = value
        {
            return external_field(dictionary, rest, names).map(|path| format!("{key}/{path}"));
        }
        None
    })
}

impl NwConfig {
    fn read(dictionary: Dictionary, warnings: &mut Vec<String>) -> Result<Self, Error> {
        if let Some(path) = EXTERNAL_STATE_FIELDS
            .iter()
            .find_map(|(sections, fields)| external_field(&dictionary, sections, fields))
        {
            return Err(Error::field(
                path,
                "identity and secrets belong in external state",
            ));
        }
        let mut root = Reader::new(String::new(), dictionary);
        let mut northwood = root.section("Northwood")?;
        if northwood.integer("SchemaVersion", u64::from(SCHEMA_VERSION))?
            != u64::from(SCHEMA_VERSION)
        {
            return Err(northwood.error("SchemaVersion", "unsupported schema version"));
        }
        let northwood_config = Northwood {
            hardware_profile: northwood.string("HardwareProfile", "")?,
            auto_profile: northwood.boolean("AutoProfile", true)?,
            ..Northwood::default()
        };
        northwood.finish(warnings);

        let mut acpi = root.section("ACPI")?;
        let mut quirks = Vec::new();
        for mut entry in acpi.dictionaries("Quirks")? {
            let matcher = entry.required_string("Match")?;
            let kind = entry.required_string("Kind")?;
            let value = entry.fields.remove("Value");
            // These three bounds match the settings accepted by the host
            // HID-over-I²C driver. Other kinds retain their extensible values.
            let constraint = match kind.as_str() {
                "I2cHidAddress" => Some((0..=0x3ff, "expected an integer from 0 to 1023")),
                "I2cHidDescriptor" => Some((0..=0xffff, "expected an integer from 0 to 65535")),
                "I2cHidForcePolling" => {
                    Some((1_000..=50_000, "expected an integer from 1000 to 50000 µs"))
                }
                _ => None,
            };
            if let Some((range, reason)) = constraint
                && !matches!(&value, Some(Value::Integer(number)) if range.contains(number))
            {
                return Err(entry.error("Value", reason));
            }
            quirks.push(AcpiQuirk {
                matcher,
                kind,
                value,
                comment: entry.string("Comment", "")?,
            });
            entry.finish(warnings);
        }
        acpi.finish(warnings);

        let mut graphics = root.section("Graphics")?;
        let backend = match graphics.string("Backend", "Auto")?.as_str() {
            "Auto" => GraphicsBackend::Auto,
            "NVModern" => GraphicsBackend::NvModern,
            "i915" => GraphicsBackend::I915,
            "Xe" => GraphicsBackend::Xe,
            "Northbridge" => GraphicsBackend::Northbridge,
            "GOP" => GraphicsBackend::Gop,
            _ => return Err(graphics.error("Backend", "unknown graphics backend")),
        };
        let scanout_mirror = graphics.boolean("ScanoutMirror", false)?;
        graphics.finish(warnings);

        let mut nvram = root.section("NVRAM")?;
        let nvram_config = Nvram {
            add: nvram.dictionary("Add")?,
        };
        nvram.finish(warnings);

        let mut platform = root.section("PlatformInfo")?;
        let platform_info = PlatformInfo {
            model: platform.string("Model", "Mac14,3")?,
        };
        platform.finish(warnings);

        let mut uefi = root.section("UEFI")?;
        let mut uefi_quirks = uefi.section("Quirks")?;
        let uefi_config = Uefi {
            connect_all: uefi.boolean("ConnectAll", true)?,
            reclaim_boot_services: uefi_quirks.boolean("ReclaimBootServices", true)?,
            firmware_path: uefi.string("FirmwarePath", crate::DEFAULT_FIRMWARE_PATH)?,
        };
        uefi_quirks.finish(warnings);
        uefi.finish(warnings);

        let config = Self {
            northwood: northwood_config,
            acpi: Acpi { quirks },
            boot: read_boot(root.section("Boot")?, warnings)?,
            drivers: read_drivers(root.section("Drivers")?, warnings)?,
            graphics: Graphics {
                backend,
                scanout_mirror,
            },
            memory: read_memory(root.section("Memory")?, warnings)?,
            guest: read_guest(root.section("Guest")?, warnings)?,
            misc: read_misc(root.section("Misc")?, warnings)?,
            nvram: nvram_config,
            platform_info,
            uefi: uefi_config,
        };
        root.finish(warnings);
        Ok(config)
    }
}

fn read_boot(mut reader: Reader, warnings: &mut Vec<String>) -> Result<Boot, Error> {
    let timeout = match reader.fields.remove("Timeout") {
        None => BootTimeout::Auto,
        Some(Value::String(value)) if value == "auto" => BootTimeout::Auto,
        Some(Value::Integer(value)) if (0..=i128::from(MAX_TIMEOUT_SECONDS)).contains(&value) => {
            BootTimeout::Seconds(value as u32)
        }
        _ => return Err(reader.error("Timeout", "expected auto or an integer from 0 to 60")),
    };
    let boot = Boot {
        timeout,
        entries: Vec::new(),
        default_install: reader.string("DefaultInstall", "")?,
        install: reader.string("Install", "")?,
        install_source: reader.string("InstallSource", "")?,
        verify_on_read: reader.boolean("VerifyOnRead", false)?,
        update_restore: reader.boolean("UpdateRestore", false)?,
        single_user: reader.boolean("SingleUser", false)?,
        maintenance_write: reader.boolean("MaintenanceWrite", false)?,
        bundle: reader.string("Bundle", "")?,
        preboot: reader.string("Preboot", "")?,
    };
    if !boot.install_source.is_empty() && !guid(&boot.install_source) {
        return Err(reader.error("InstallSource", "expected a partition GUID"));
    }
    reader.finish(warnings);
    Ok(boot)
}

fn read_entries(
    entries: Vec<Reader>,
    base: &Dictionary,
    warnings: &mut Vec<String>,
) -> Result<Vec<BootEntry>, Error> {
    if entries.len() > MAX_CHOICES {
        return Err(Error::field("Boot/Entries", "too many boot entries"));
    }
    let base_warnings = warnings.clone();
    let mut choices: Vec<BootEntry> = Vec::with_capacity(entries.len());
    for mut entry in entries {
        let name = entry.required_string("Name")?;
        if name.len() > MAX_CHOICE_NAME || !name.bytes().all(|ch| (0x20..=0x7e).contains(&ch)) {
            return Err(entry.error(
                "Name",
                "expected a printable ASCII name of at most 40 bytes",
            ));
        }
        if choices.iter().any(|choice| choice.name == name) {
            return Err(entry.error("Name", "duplicate boot entry name"));
        }
        let overrides = entry.dictionary("Overrides")?;
        if matches!(overrides.get("Boot"), Some(Value::Dictionary(boot)) if boot.contains_key("Entries"))
        {
            return Err(entry.error("Overrides/Boot/Entries", "nested boot entries are invalid"));
        }
        let mut effective = base.clone();
        merge(&mut effective, overrides);
        let mut entry_warnings = Vec::new();
        let config =
            NwConfig::read(effective, &mut entry_warnings).map_err(|error| match error {
                Error::Field { path, reason } => entry.error(&format!("Overrides/{path}"), reason),
                error => error,
            })?;
        warnings.extend(
            entry_warnings
                .into_iter()
                .filter(|path| !base_warnings.contains(path))
                .map(|path| entry.path(&format!("Overrides/{path}"))),
        );
        entry.finish(warnings);
        choices.push(BootEntry { name, config });
    }
    Ok(choices)
}

pub(crate) fn merge(base: &mut Dictionary, overrides: Dictionary) {
    for (key, value) in overrides {
        match (base.get_mut(&key), value) {
            (Some(Value::Dictionary(old)), Value::Dictionary(new)) => merge(old, new),
            (_, value) => {
                base.insert(key, value);
            }
        }
    }
}

fn read_drivers(mut reader: Reader, warnings: &mut Vec<String>) -> Result<Drivers, Error> {
    let mode = match reader.string("Mode", "Auto")?.as_str() {
        "Auto" => DriverMode::Auto,
        "Safe" => DriverMode::Safe,
        "Manual" => DriverMode::Manual,
        _ => return Err(reader.error("Mode", "expected Auto, Safe or Manual")),
    };
    let enable = reader.strings("Enable")?;
    let disable = reader.strings("Disable")?;
    if enable.iter().any(|module| disable.contains(module)) {
        return Err(reader.error("Enable", "a module is both enabled and disabled"));
    }
    let arguments = reader
        .dictionary("Arguments")?
        .into_iter()
        .map(|(key, value)| match value {
            Value::String(value) if !key.trim().is_empty() => Ok((key, value)),
            _ => Err(reader.error(
                "Arguments",
                "expected a dictionary of module names and argument strings",
            )),
        })
        .collect::<Result<_, _>>()?;
    let mut block: Vec<DriverBlock> = Vec::new();
    for mut item in reader.dictionaries("Block")? {
        let matcher = pci_selector(&mut item)?;
        if block.iter().any(|rule| rule.matcher == matcher) {
            return Err(item.error("Match", "duplicate blocked device"));
        }
        block.push(DriverBlock {
            matcher,
            comment: item.string("Comment", "")?,
        });
        item.finish(warnings);
    }
    let mut force: Vec<DriverForce> = Vec::new();
    for mut item in reader.dictionaries("Force")? {
        let module = item.required_string("Module")?;
        let matcher = pci_selector(&mut item)?;
        if disable.contains(&module) || (mode == DriverMode::Manual && !enable.contains(&module)) {
            return Err(item.error("Module", "forced module must be enabled"));
        }
        if force.iter().any(|rule| rule.matcher == matcher) {
            return Err(item.error("Match", "duplicate forced device"));
        }
        if block.iter().any(|rule| rule.matcher == matcher) {
            return Err(item.error("Match", "forced device is also blocked"));
        }
        force.push(DriverForce { module, matcher });
        item.finish(warnings);
    }
    let usb4_approve = reader.strings("Usb4Approve")?;
    if !usb4_approve.iter().all(|uuid| guid(uuid)) {
        return Err(reader.error("Usb4Approve", "expected USB4 UUIDs"));
    }
    let drivers = Drivers {
        mode,
        enable,
        disable,
        arguments,
        block,
        force,
        input: reader.boolean("Input", false)?,
        guest_network: reader.boolean("GuestNetwork", false)?,
        external_disks: reader.boolean("ExternalDisks", false)?,
        usb_storage: reader.strings("UsbStorage")?,
        usb4_approve,
        network: read_network(reader.section("Network")?, warnings)?,
    };
    reader.finish(warnings);
    Ok(drivers)
}

fn pci_selector(reader: &mut Reader) -> Result<String, Error> {
    let matcher = reader.required_string("Match")?;
    let valid = matcher
        .strip_prefix("pci:")
        .and_then(|ids| ids.split_once(':'))
        .is_some_and(|(vendor, device)| {
            [vendor, device]
                .into_iter()
                .all(|id| id.len() == 4 && id.bytes().all(|byte| byte.is_ascii_hexdigit()))
        });
    if !valid {
        return Err(reader.error("Match", "expected pci:vvvv:dddd"));
    }
    Ok(matcher.to_ascii_lowercase())
}

fn read_network(mut reader: Reader, warnings: &mut Vec<String>) -> Result<Network, Error> {
    let mut wifi = reader.section("WiFi")?;
    let wifi_config = Wifi {
        ssid: wifi.string("SSID", "")?,
        key_file: wifi.string("KeyFile", crate::DEFAULT_WIFI_KEY)?,
        country: wifi.string("Country", "")?,
    };
    if wifi_config.ssid.len() > 32 {
        return Err(wifi.error("SSID", "SSID must be at most 32 UTF-8 bytes"));
    }
    if !wifi_config.country.is_empty()
        && (wifi_config.country.len() != 2
            || !wifi_config
                .country
                .bytes()
                .all(|ch| ch.is_ascii_alphabetic()))
    {
        return Err(wifi.error("Country", "expected a two-letter country code"));
    }
    wifi.finish(warnings);
    let ipv4 = reader.string("IPv4", "")?;
    if !ipv4.is_empty() {
        let address = ipv4.split_once('/').and_then(|(address, prefix)| {
            let address = address.parse::<Ipv4Addr>().ok()?;
            let prefix = prefix.parse::<u8>().ok().filter(|n| (1..=32).contains(n))?;
            Some((address, prefix))
        });
        if address.is_none() {
            return Err(reader.error("IPv4", "expected IPv4 address and prefix length"));
        }
    }
    let gateway = reader.string("Gateway", "")?;
    if !gateway.is_empty() && (ipv4.is_empty() || gateway.parse::<Ipv4Addr>().is_err()) {
        return Err(reader.error(
            "Gateway",
            "expected an IPv4 gateway and a static IPv4 address",
        ));
    }
    let dns_words = reader.strings("DNS")?;
    if dns_words.len() > 3 {
        return Err(reader.error("DNS", "expected at most three DNS servers"));
    }
    let mut dns = Vec::new();
    for word in dns_words {
        let server = word
            .parse::<IpAddr>()
            .ok()
            .filter(|server| {
                !server.is_unspecified() && !server.is_multicast() && !dns.contains(server)
            })
            .ok_or_else(|| reader.error("DNS", "expected distinct unicast IP addresses"))?;
        dns.push(server);
    }
    reader.finish(warnings);
    Ok(Network {
        wifi: wifi_config,
        ipv4,
        gateway,
        dns,
    })
}

fn read_memory(mut reader: Reader, warnings: &mut Vec<String>) -> Result<Memory, Error> {
    let profile = match reader.string("Profile", "Auto")?.as_str() {
        "Auto" => MemoryProfile::Auto,
        "Standard" => MemoryProfile::Standard,
        "Compact" => MemoryProfile::Compact,
        "Paged" => MemoryProfile::Paged,
        _ => return Err(reader.error("Profile", "unknown memory profile")),
    };
    let mut swap = reader.section("Swap")?;
    let location = match swap.string("Location", "Auto")?.as_str() {
        "Auto" => SwapLocation::Auto,
        "Partition" => SwapLocation::Partition,
        "VolumeFile" | "Volume" => SwapLocation::VolumeFile,
        "None" => SwapLocation::None,
        _ => return Err(swap.error("Location", "unknown swap location")),
    };
    let has_size_gib = swap.fields.contains_key("SizeGiB");
    let size_gib = swap.integer("SizeGiB", 32)?;
    let size_bytes = size_gib
        .checked_mul(GIB)
        .ok_or_else(|| swap.error("SizeGiB", "size exceeds 64-bit capacity"))?;
    let bytes = swap.integer("Bytes", size_bytes)?;
    if has_size_gib && bytes != size_bytes {
        return Err(swap.error("Bytes", "conflicts with SizeGiB"));
    }
    if bytes < GIB || bytes % 4096 != 0 {
        return Err(swap.error("Bytes", "expected at least 1 GiB, aligned to 4 KiB"));
    }
    let has_partition = swap.fields.contains_key("Partition");
    let partition = swap.string("Partition", "")?;
    let partition_uuid = swap.string("PartitionUUID", &partition)?;
    if has_partition && !partition.eq_ignore_ascii_case(&partition_uuid) {
        return Err(swap.error("PartitionUUID", "conflicts with Partition"));
    }
    if !partition_uuid.is_empty() && !guid(&partition_uuid) {
        return Err(swap.error("Partition", "expected a partition GUID"));
    }
    let has_volume_max_gib = swap.fields.contains_key("VolumeMaxGiB");
    let volume_max_gib = swap.integer("VolumeMaxGiB", 8)?;
    let volume_max_default = volume_max_gib
        .checked_mul(GIB)
        .ok_or_else(|| swap.error("VolumeMaxGiB", "size exceeds 64-bit capacity"))?;
    let volume_max_bytes = swap.integer("VolumeMaxBytes", volume_max_default)?;
    if has_volume_max_gib && volume_max_bytes != volume_max_default {
        return Err(swap.error("VolumeMaxBytes", "conflicts with VolumeMaxGiB"));
    }
    if volume_max_bytes < GIB || volume_max_bytes % 4096 != 0 {
        return Err(swap.error(
            "VolumeMaxBytes",
            "expected at least 1 GiB, aligned to 4 KiB",
        ));
    }
    let swap_config = Swap {
        enabled: swap.boolean("Enabled", true)? && location != SwapLocation::None,
        bytes,
        location,
        partition: partition_uuid,
        volume_max_bytes,
        encrypt: swap.boolean("Encrypt", true)?,
        guest_swap: swap.boolean("GuestSwap", true)?,
    };
    swap.finish(warnings);
    let mut paged = reader.section("Paged")?;
    let paged_config = Paged {
        reported_bytes: paged.integer("ReportedBytes", 2 * GIB)?,
        heap_floor_bytes: paged.integer("HeapFloorBytes", GIB / 2)?,
    };
    for (field, bytes) in [
        ("ReportedBytes", paged_config.reported_bytes),
        ("HeapFloorBytes", paged_config.heap_floor_bytes),
    ] {
        if bytes == 0 || bytes % (2 << 20) != 0 {
            return Err(paged.error(field, "expected a nonzero multiple of 2 MiB"));
        }
    }
    paged.finish(warnings);
    let memory = Memory {
        profile,
        swap: swap_config,
        paged: paged_config,
        translation_store_bytes: reader.integer("TranslationStoreBytes", 0)?,
        storage_ram_bytes: reader.integer("StorageRamBytes", 0)?,
    };
    reader.finish(warnings);
    Ok(memory)
}

fn read_guest(mut reader: Reader, warnings: &mut Vec<String>) -> Result<Guest, Error> {
    let guest = Guest {
        storage: reader.string("Storage", "")?,
        nor: reader.string("Nor", "")?,
        cores: reader.integer("Cores", 0)?,
        pes: reader.integer("PEs", 0)?,
        parallel: reader.boolean("Parallel", true)?,
        idle_skip: reader.boolean("IdleSkip", false)?,
        native_rounds: reader.boolean("NativeRounds", false)?,
        boot_args: reader.string("BootArgs", "")?,
    };
    if guest.cores > 64 {
        return Err(reader.error("Cores", "expected at most 64 native cores"));
    }
    if !guest.storage.is_empty() && !target(&guest.storage) {
        return Err(reader.error("Storage", "expected SERIAL:GUID or @disk"));
    }
    if !guest.nor.is_empty() && !target(&guest.nor) {
        return Err(reader.error("Nor", "expected SERIAL:GUID or @disk"));
    }
    reader.finish(warnings);
    Ok(guest)
}

fn read_misc(mut reader: Reader, warnings: &mut Vec<String>) -> Result<Misc, Error> {
    let mut debug = reader.section("Debug")?;
    let mut log = reader.section("BootLog")?;
    let misc = Misc {
        debug: DebugOptions {
            host_check: debug.boolean("HostCheck", false)?,
            profile: debug.boolean("Profile", false)?,
            survey: debug.boolean("Survey", false)?,
            sep_observe: debug.boolean("SepObserve", false)?,
            monitor_trace: debug.boolean("MonitorTrace", false)?,
            display_trace: debug.boolean("DisplayTrace", false)?,
            breaks: debug.diagnostic_words("Break", true)?,
            trace: debug.diagnostic_words("Trace", true)?,
            watch: debug.diagnostic_words("Watch", false)?,
            dump: debug.diagnostic_words("Dump", false)?,
            network_check: debug.string("NetworkCheck", "")?,
            audio_check: debug.string("AudioCheck", "")?,
            audio_own_hpet: debug.boolean("AudioOwnHpet", false)?,
            own_ioapic: debug.boolean("OwnIoapic", false)?,
            native_bootstrap_check: debug.boolean("NativeBootstrapCheck", false)?,
            native_shared_check: debug.boolean("NativeSharedCheck", false)?,
            native_artifacts: debug.boolean("NativeArtifacts", false)?,
            development_restore: debug.boolean("DevelopmentRestore", false)?,
            raw: debug.string("Raw", "")?,
        },
        boot_log: BootLog {
            esp: log.boolean("Esp", true)?,
            volume: log.boolean("Volume", true)?,
        },
    };
    debug.finish(warnings);
    log.finish(warnings);
    reader.finish(warnings);
    Ok(misc)
}

fn target(value: &str) -> bool {
    value == "@disk"
        || value.rsplit_once(':').is_some_and(|(serial, partition)| {
            !serial.is_empty() && !serial.chars().any(char::is_whitespace) && guid(partition)
        })
}

fn guid(value: &str) -> bool {
    value.len() == 36
        && value.bytes().enumerate().all(|(index, byte)| match index {
            8 | 13 | 18 | 23 => byte == b'-',
            _ => byte.is_ascii_hexdigit(),
        })
}
