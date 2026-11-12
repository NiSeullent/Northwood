// SPDX-License-Identifier: GPL-2.0-or-later
//! Hardware inventory from numeric PCI/USB listings and observed boot outcomes.

use anyhow::{Context, Result, ensure};
use clap::{ArgGroup, Args};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{self, Write as _},
    fs::File,
    io::{self, Read, Write as _},
    path::{Path, PathBuf},
};
use stela_drivers::{
    bus::pci::{Address, Identity},
    gpu::intel::glk::{MemoryLayout, StolenMemory},
    net::wifi::{intel, unsupported},
    probe::registry::{self, Firmware, Manifest, Status},
    usb::{bluetooth, ethernet},
};
use stela_loader::host_firmware::{recipes, requirements::Inventory};

mod glk_attachment;
mod lpss_attachment;

const MAX_INPUT_BYTES: u64 = 1024 * 1024;
const MAX_BOOT_LOG_BYTES: u64 = 8 * MAX_INPUT_BYTES;

#[derive(Args)]
#[command(group(ArgGroup::new("inventory").required(true).multiple(true).args(["lspci", "lsusb", "boot_log"])))]
pub(crate) struct Report {
    /// Read a file captured with `lspci -nn` or `lspci -Dnn`.
    #[arg(long, value_name = "FILE")]
    lspci: Option<PathBuf>,
    /// Read a file captured with `lsusb`.
    #[arg(long, value_name = "FILE")]
    lsusb: Option<PathBuf>,
    /// Use the latest boot from one serial capture or one `stela boot-log` export.
    #[arg(long, value_name = "FILE")]
    boot_log: Option<PathBuf>,
}

impl Report {
    pub(crate) fn run(self) -> Result<()> {
        let pci = self
            .lspci
            .as_deref()
            .map(|path| read(path, "lspci", parse_pci))
            .transpose()?;
        let usb = self
            .lsusb
            .as_deref()
            .map(|path| read(path, "lsusb", parse_usb))
            .transpose()?;
        let boot = self.boot_log.as_deref().map(read_boot).transpose()?;
        let report = render(pci.as_deref(), usb.as_deref(), boot.as_ref())?;
        io::stdout().lock().write_all(report.as_bytes())?;
        Ok(())
    }
}

fn read<T>(path: &Path, source: &str, parse: fn(&str) -> Option<T>) -> Result<Vec<T>> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut input = String::new();
    file.take(MAX_INPUT_BYTES + 1)
        .read_to_string(&mut input)
        .with_context(|| format!("cannot read {source} text from {}", path.display()))?;
    ensure!(
        input.len() as u64 <= MAX_INPUT_BYTES,
        "{source} inventory exceeds 1 MiB"
    );
    input
        .lines()
        .map(str::trim)
        .enumerate()
        .filter(|(_, text)| !text.is_empty())
        .map(|(line, text)| {
            parse(text).with_context(|| {
                format!(
                    "{} line {} is not a valid {source} inventory row",
                    path.display(),
                    line + 1
                )
            })
        })
        .collect()
}

fn hex(word: &str, digits: usize) -> Option<u16> {
    if word.len() != digits || !word.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u16::from_str_radix(word, 16).ok()
}

fn ids(word: &str) -> Option<(u16, u16)> {
    let (vendor, device) = word.split_once(':')?;
    Some((hex(vendor, 4)?, hex(device, 4)?))
}

fn address(word: &str) -> Option<Address> {
    let (bus, slot) = word.rsplit_once(':')?;
    let (segment, bus) = match bus.split_once(':') {
        Some((segment, bus)) => (hex(segment, 4)?, bus),
        None => (0, bus),
    };
    let (device, function) = slot.split_once('.')?;
    let (device, function) = (hex(device, 2)? as u8, hex(function, 1)? as u8);
    (device < 32 && function < 8).then_some(Address {
        segment,
        bus: hex(bus, 2)? as u8,
        device,
        function,
    })
}

/// `lspci -nn` supplies base class and subclass. The last byte is a
/// placeholder until `pci_candidates` checks every possible interface.
fn parse_pci(line: &str) -> Option<Identity> {
    let (slot, text) = line.split_once(char::is_whitespace)?;
    let (class, text) = text.split_once("]:")?;
    let class = hex(class.rsplit_once('[')?.1, 4)?;
    let (vendor, device) = text
        .split('[')
        .skip(1)
        .filter_map(|field| field.split_once(']').map(|(value, _)| value))
        .find_map(ids)?;
    Some(Identity {
        address: address(slot)?,
        vendor,
        device,
        class: [(class >> 8) as u8, class as u8, 0],
    })
}

#[derive(Debug, PartialEq, Eq)]
struct Usb {
    bus: u16,
    address: u16,
    vendor: u16,
    product: u16,
}

fn parse_usb(line: &str) -> Option<Usb> {
    fn number(word: &str) -> Option<u16> {
        let value: u16 = word.parse().ok()?;
        (word.len() == 3 && word.bytes().all(|byte| byte.is_ascii_digit()) && value != 0)
            .then_some(value)
    }
    let mut fields = line.split_ascii_whitespace();
    if fields.next()? != "Bus" {
        return None;
    }
    let bus = number(fields.next()?)?;
    if fields.next()? != "Device" {
        return None;
    }
    let address = number(fields.next()?.strip_suffix(':')?)?;
    if fields.next()? != "ID" {
        return None;
    }
    let (vendor, product) = ids(fields.next()?)?;
    Some(Usb {
        bus,
        address,
        vendor,
        product,
    })
}

#[derive(Default)]
struct BootReport {
    pci: BTreeMap<Address, BootDevice>,
    acpi: BTreeMap<String, BTreeSet<AcpiOutcome>>,
}

#[derive(Default)]
struct BootDevice {
    identity: Option<Identity>,
    class: Option<[u8; 3]>,
    glk_id: Option<u16>,
    glk_memory: Option<GlkMemory>,
    glk_attachment: glk_attachment::Attachment,
    lpss_id: Option<u16>,
    lpss_attachment: lpss_attachment::Attachment,
    inconsistent: bool,
    outcomes: BTreeSet<Outcome>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Outcome {
    Network(Option<&'static str>),
    GuestNetwork,
    Storage(&'static str),
    ProbeFailed,
    ChildFailed,
    Unsupported,
    Disabled,
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Network(Some(driver)) => write!(f, "Network adapter attached ({driver})"),
            Self::Network(None) => f.write_str("Network adapter attached"),
            Self::GuestNetwork => f.write_str("Selected for guest networking"),
            Self::Storage(transport) => write!(f, "{transport} disk observed"),
            Self::ProbeFailed => f.write_str("PCI probe failed"),
            Self::ChildFailed => f.write_str("Device behind this function failed"),
            Self::Unsupported => f.write_str("Reported unsupported"),
            Self::Disabled => f.write_str("Disabled by nwconfig"),
        }
    }
}

impl BootDevice {
    fn observe_class(&mut self, class: [u8; 3]) {
        self.inconsistent |= self
            .class
            .replace(class)
            .is_some_and(|previous| previous != class);
    }

    fn observe_glk_id(&mut self, device: u16) {
        self.inconsistent |= self
            .glk_id
            .replace(device)
            .is_some_and(|previous| previous != device);
    }

    fn observe_lpss_id(&mut self, device: u16) {
        self.inconsistent |= self
            .lpss_id
            .replace(device)
            .is_some_and(|previous| previous != device);
    }

    fn agrees_with(&self, pci: Option<&Identity>) -> bool {
        let observed_glk = self.glk_id.is_some() || self.glk_attachment.observed();
        let matches_glk = |identity: &Identity| {
            identity.vendor == 0x8086
                && matches!(identity.device, 0x3184 | 0x3185)
                && self.glk_id.is_none_or(|glk| glk == identity.device)
                && identity.class[0] == 3
        };
        !self.inconsistent
            && self.lpss_id.is_none_or(|device| {
                let matches_lpss =
                    |identity: &Identity| (identity.vendor, identity.device) == (0x8086, device);
                !observed_glk
                    && self.identity.as_ref().is_none_or(matches_lpss)
                    && pci.is_none_or(matches_lpss)
            })
            && (!observed_glk
                || self.class.is_none_or(|class| class[0] == 3)
                    && self.identity.as_ref().is_none_or(matches_glk)
                    && pci.is_none_or(matches_glk))
            && !pci.is_some_and(|pci| {
                self.class.is_some_and(|class| class[..2] != pci.class[..2])
                    || self.identity.is_some_and(|identity| {
                        (identity.vendor, identity.device) != (pci.vendor, pci.device)
                    })
            })
    }

    fn describe(&self) -> String {
        self.outcomes
            .iter()
            .map(ToString::to_string)
            .chain(self.glk_memory.map(GlkMemory::describe))
            .chain(self.glk_attachment.describe())
            .chain(self.lpss_attachment.describe())
            .collect::<Vec<_>>()
            .join("; ")
    }
}

#[derive(Clone, Copy)]
struct GlkMemory(Option<MemoryLayout>);

impl GlkMemory {
    fn describe(self) -> String {
        let Some(memory) = self.0 else {
            return "GLK PCI memory inventory unavailable".into();
        };
        let stolen = match memory.stolen {
            Some(stolen) => format!("stolen {} bytes at {:#x}", stolen.bytes, stolen.base),
            None => "no stolen memory reported".into(),
        };
        format!(
            "GLK PCI memory inventory only (GGTT {} bytes, {stolen})",
            memory.ggtt_bytes
        )
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum AcpiOutcome {
    StopReceiptUnavailable,
    LpssStopReceiptUnavailable(Address, (u16, u16)),
    LpssControllerMismatch(Address, (u16, u16)),
    LpssResources(Address, u16, lpss_attachment::Acpi),
    HidDescriptorUnavailable,
    HidPolling,
}

impl AcpiOutcome {
    fn describe(&self) -> String {
        match self {
            Self::StopReceiptUnavailable => {
                "Probe refused: native PCI endpoint stop receipt unavailable".into()
            }
            Self::LpssStopReceiptUnavailable(slot, (vendor, device)) => format!(
                "LPSS I²C {slot} {vendor:04x}:{device:04x} refused: native PCI endpoint stop receipt unavailable"
            ),
            Self::LpssControllerMismatch(slot, (vendor, device)) => format!(
                "I²C-HID probe refused: ACPI controller {slot} {vendor:04x}:{device:04x} is not a supported LPSS I²C function"
            ),
            Self::LpssResources(slot, device, resources) => resources.describe(*slot, *device),
            Self::HidDescriptorUnavailable => {
                "I²C-HID probe refused: _DSM did not return the descriptor register".into()
            }
            Self::HidPolling => "I²C-HID input device listed for host polling".into(),
        }
    }
}

fn read_boot(path: &Path) -> Result<BootReport> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut input = Vec::new();
    file.take(MAX_BOOT_LOG_BYTES + 1)
        .read_to_end(&mut input)
        .with_context(|| format!("cannot read boot log {}", path.display()))?;
    ensure!(
        input.len() as u64 <= MAX_BOOT_LOG_BYTES,
        "boot log exceeds 8 MiB"
    );
    boot_observations(&input)
}

/// Both the on-disk text mirror and the exported ring can prefix timestamps.
fn boot_line(line: &str) -> &str {
    let line = line.trim_ascii();
    if let Some((stamp, message)) = line.strip_prefix('[').and_then(|line| line.split_once(']'))
        && let Some((seconds, fraction)) = stamp.trim_ascii().split_once('.')
        && !seconds.is_empty()
        && !fraction.is_empty()
        && seconds
            .bytes()
            .chain(fraction.bytes())
            .all(|byte| byte.is_ascii_digit())
    {
        return message.trim_ascii_start();
    }
    line
}

fn pci_class(field: &str) -> Option<[u8; 3]> {
    let field = field.strip_prefix("class=")?;
    if field.len() != 6 || !field.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let class = u32::from_str_radix(field, 16).ok()?.to_be_bytes();
    Some([class[1], class[2], class[3]])
}

/// The host-check census is identity evidence. Its candidate and inventory
/// labels do not report the result of a probe.
fn boot_identity(fields: &str) -> Option<Identity> {
    let mut fields = fields.split_ascii_whitespace();
    let address = address(fields.next()?)?;
    let (vendor, device) = ids(fields.next()?)?;
    let class = pci_class(fields.next()?)?;
    let (kind, _) = fields.next()?.split_once('=')?;
    if !matches!(kind, "candidate" | "inventory") {
        return None;
    }
    Some(Identity {
        address,
        vendor,
        device,
        class,
    })
}

fn network_driver(driver: &str) -> Option<&'static str> {
    registry::manifests()
        .map(|manifest| manifest.implementation)
        .find(|implementation| *implementation == driver)
}

fn observed_outcome(line: &str) -> Option<(Address, Outcome)> {
    if let Some(fields) = line.strip_prefix("HOST DEVICE FAILED pci=") {
        let (slot, detail) = fields.split_once(" (")?;
        detail.split_once(") error=")?;
        return Some((address(slot)?, Outcome::ProbeFailed));
    }
    if let Some(fields) = line.strip_prefix("HOST DEVICE FAILED ") {
        let (location, _) = fields.split_once(": ")?;
        let (slot, _) = location.split_once(' ')?;
        return Some((address(slot)?, Outcome::ChildFailed));
    }
    if let Some(fields) = line.strip_prefix("NETWORK link=") {
        let (location, details) = fields.rsplit_once(" driver=")?;
        let (driver, details) = details.split_once(" transport=")?;
        let (transport, carrier) = details.split_once(" carrier=")?;
        if !matches!(transport, "Ethernet" | "Wi-Fi") || !matches!(carrier, "true" | "false") {
            return None;
        }
        return Some((
            address(location.split_ascii_whitespace().next()?)?,
            Outcome::Network(network_driver(driver)),
        ));
    }
    if let Some(fields) = line.strip_prefix("NETWORK candidate driver=") {
        let (driver, fields) = fields.split_once(" adapter=")?;
        let (location, _) = fields.rsplit_once(" mac=")?;
        return Some((
            address(location.split_ascii_whitespace().next()?)?,
            Outcome::Network(network_driver(driver)),
        ));
    }
    if let Some(fields) = line.strip_prefix("GUEST NETWORK adapter ") {
        let (location, _) = fields.rsplit_once(" mac=")?;
        return Some((
            address(location.split_ascii_whitespace().next()?)?,
            Outcome::GuestNetwork,
        ));
    }
    let disk = line
        .strip_prefix("DISK ")
        .or_else(|| line.strip_prefix("STORAGE DEVICE "))?;
    let (transport, fields) = disk.split_once(" at ")?;
    let transport = match transport {
        "Ahci" => "AHCI",
        "Nvme" => "NVMe",
        "SdMmc" => "SD/MMC",
        "VirtioBlk" => "Virtio block",
        _ => return None,
    };
    let (location, _) = fields.split_once(" serial=")?;
    Some((
        address(location.split_ascii_whitespace().next()?)?,
        Outcome::Storage(transport),
    ))
}

fn observed_glk(line: &str) -> Option<(Address, u16, GlkMemory)> {
    const MIB: u64 = 1 << 20;
    let fields = line.strip_prefix("HOST GPU GLK pci=")?;
    let (slot, fields) = fields.split_once(" device=")?;
    let slot = address(slot)?;
    let (device, facts) = fields.split_once(' ')?;
    let device = hex(device, 4)?;
    if !matches!(device, 0x3184 | 0x3185) {
        return None;
    }
    let memory = if let Some(error) = facts.strip_prefix("memory unavailable: ") {
        if error.is_empty() {
            return None;
        }
        // Keep the outcome without quoting potentially identifying log details.
        None
    } else {
        let (ggtt, fields) = facts
            .strip_prefix("ggtt_bytes=")?
            .split_once(" stolen_base=0x")?;
        let (base, bytes) = fields.split_once(" stolen_bytes=")?;
        if !ggtt.bytes().all(|byte| byte.is_ascii_digit())
            || !base.bytes().all(|byte| byte.is_ascii_hexdigit())
            || !bytes.bytes().all(|byte| byte.is_ascii_digit())
        {
            return None;
        }
        let ggtt_bytes = ggtt.parse::<u64>().ok()?;
        if ![0, 2 * MIB, 4 * MIB, 8 * MIB].contains(&ggtt_bytes) {
            return None;
        }
        let base = u64::from_str_radix(base, 16).ok()?;
        let bytes = bytes.parse::<u64>().ok()?;
        let stolen = if (base, bytes) == (0, 0) {
            None
        } else if base > 0
            && base.is_multiple_of(MIB)
            && (32 * MIB..=2016 * MIB).contains(&bytes)
            && bytes.is_multiple_of(32 * MIB)
            && base.checked_add(bytes).is_some_and(|end| end <= 1 << 32)
        {
            Some(StolenMemory { base, bytes })
        } else {
            return None;
        };
        Some(MemoryLayout { ggtt_bytes, stolen })
    };
    Some((slot, device, GlkMemory(memory)))
}

fn observed_acpi(line: &str) -> Option<(&str, AcpiOutcome)> {
    const STOP_RECEIPT_UNAVAILABLE: &str = "native endpoint engine stop receipt unavailable";

    fn pci_location(fields: &str) -> Option<(Address, (u16, u16))> {
        let (slot, id) = fields.split_once(' ')?;
        (slot.len() == 12).then_some(())?;
        Some((address(slot)?, ids(id)?))
    }
    let (path, outcome) = if let Some(path) = line.strip_prefix("INPUT host device I2C-HID ") {
        (path, AcpiOutcome::HidPolling)
    } else {
        let (path, reason) = line.strip_prefix("PLATFORM FAILED ")?.split_once(": ")?;
        let outcome = match reason {
            STOP_RECEIPT_UNAVAILABLE => AcpiOutcome::StopReceiptUnavailable,
            "HID-over-I2C _DSM did not return the descriptor register" => {
                AcpiOutcome::HidDescriptorUnavailable
            }
            reason => {
                if let Some(location) = reason
                    .strip_prefix("LPSS I2C ")
                    .and_then(|fields| fields.split_once(" handoff: "))
                    .filter(|(_, error)| *error == STOP_RECEIPT_UNAVAILABLE)
                    .map(|(location, _)| location)
                {
                    let (slot, id) = pci_location(location)?;
                    AcpiOutcome::LpssStopReceiptUnavailable(slot, id)
                } else {
                    let (slot, id) = pci_location(
                        reason
                            .strip_prefix("I2C ACPI controller ")?
                            .strip_suffix(" is not a supported LPSS I2C function")?,
                    )?;
                    AcpiOutcome::LpssControllerMismatch(slot, id)
                }
            }
        };
        (path, outcome)
    };
    valid_acpi_path(path).then_some((path, outcome))
}

fn valid_acpi_path(path: &str) -> bool {
    path.strip_prefix('\\').is_some_and(|path| {
        path.split('.').all(|component| {
            let Some((&head, tail)) = component.as_bytes().split_first() else {
                return false;
            };
            component.len() <= 4
                && (head.is_ascii_uppercase() || head == b'_')
                && tail
                    .iter()
                    .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || *byte == b'_')
        })
    })
}

fn boot_observations(input: &[u8]) -> Result<BootReport> {
    let mut boot = BootReport::default();
    let mut setup_stream = false;
    for bytes in input.split_inclusive(|&byte| byte == b'\n') {
        // A truncated serial line cannot establish a probe outcome.
        let Ok(line) = std::str::from_utf8(bytes) else {
            continue;
        };
        if !line.ends_with('\n') {
            continue;
        }
        let line = boot_line(line);
        ensure!(
            !line.starts_with("=== partition ") && !line.starts_with("=== the file"),
            "boot log contains multiple streams; supply one serial capture or exported log"
        );
        if [
            "Boot log on the USB (",
            "Boot log of install \"",
            "Loader boot log on the data volume (",
        ]
        .iter()
        .any(|&header| line.starts_with(header))
        {
            ensure!(
                !setup_stream,
                "setup export contains multiple streams; supply one serial capture or exported log"
            );
            setup_stream = true;
        }
        if line.starts_with("STELA UEFI MOUNT version=")
            || line
                .strip_prefix("=== STELA BOOT ")
                .and_then(|s| s.strip_suffix(" ==="))
                .is_some_and(|n| n.parse::<u64>().is_ok())
        {
            boot = BootReport::default();
        } else if let Some(identity) = line.strip_prefix("DEVICE PCI ").and_then(boot_identity) {
            let device = boot.pci.entry(identity.address).or_default();
            device.observe_class(identity.class);
            device.inconsistent |= device
                .identity
                .replace(identity)
                .is_some_and(|previous| previous != identity);
        } else if let Some((slot, class, reason)) = line
            .strip_prefix("HOST UNSUPPORTED pci=")
            .and_then(|fields| {
                let (slot, fields) = fields.split_once(' ')?;
                let (class, reason) = fields.split_once(": ")?;
                Some((address(slot)?, pci_class(class)?, reason))
            })
        {
            let device = boot.pci.entry(slot).or_default();
            device.observe_class(class);
            device
                .outcomes
                .insert(if reason == "driver module disabled by nwconfig" {
                    Outcome::Disabled
                } else {
                    Outcome::Unsupported
                });
        } else if let Some((slot, outcome)) = observed_outcome(line) {
            boot.pci.entry(slot).or_default().outcomes.insert(outcome);
        } else if let Some((slot, id, memory)) = observed_glk(line) {
            let device = boot.pci.entry(slot).or_default();
            device.observe_glk_id(id);
            device.glk_memory = Some(memory);
        } else if let Some((slot, observation)) = glk_attachment::parse(line) {
            let device = boot.pci.entry(slot).or_default();
            if let Some(id) = observation.device() {
                device.observe_glk_id(id);
            }
            device.glk_attachment.observe(observation);
        } else if let Some((slot, id, record)) = lpss_attachment::parse(line) {
            let device = boot.pci.entry(slot).or_default();
            device.observe_lpss_id(id);
            if let lpss_attachment::Record::Acpi(path, resources) = record {
                boot.acpi
                    .entry(path.into())
                    .or_default()
                    .insert(AcpiOutcome::LpssResources(slot, id, resources));
            } else {
                device.lpss_attachment.observe(record);
            }
        } else if let Some((path, outcome)) = observed_acpi(line) {
            boot.acpi.entry(path.into()).or_default().insert(outcome);
        }
    }
    Ok(boot)
}

fn pci_modules(mut identity: Identity, prog_if_known: bool) -> Vec<(&'static Manifest, usize)> {
    let mut matches: Vec<(&Manifest, usize)> = Vec::new();
    let interfaces = if prog_if_known {
        identity.class[2]..=identity.class[2]
    } else {
        u8::MIN..=u8::MAX
    };
    for interface in interfaces {
        identity.class[2] = interface;
        for manifest in registry::pci_candidates(&identity) {
            if let Some((_, count)) = matches.iter_mut().find(|(item, _)| *item == manifest) {
                *count += 1;
            } else {
                matches.push((manifest, 1));
            }
        }
    }
    matches
}

fn describe_pci(identity: Identity, prog_if_known: bool) -> Result<(String, Vec<String>)> {
    let mut firmware = Inventory::default();
    let mut candidates = Vec::new();
    let wifi = identity.class[..2] == [2, 0x80];
    if wifi && let Some(reason) = unsupported::reason(identity.vendor, identity.device) {
        if (identity.vendor, identity.device) == (0x8086, 0x31dc) {
            recipes::require_15igl7_wifi(&mut firmware)?;
        } else if let Some(sku) =
            intel::firmware::legacy::Sku::from_pci(identity.vendor, identity.device)
        {
            candidates.push(sku.firmware_name().into());
            recipes::require_regulatory(&mut firmware)?;
        }
        candidates.extend(
            firmware
                .requests()
                .iter()
                .map(|request| request.name.clone()),
        );
        return Ok((reason.into(), candidates));
    }
    if wifi && recipes::require_wifi(&mut firmware, identity.vendor, identity.device)? {
        recipes::require_regulatory(&mut firmware)?;
    }

    let mut drivers = Vec::new();
    for (manifest, interfaces) in pci_modules(identity, prog_if_known) {
        let status = match manifest.status {
            Status::Implemented => "probe candidate",
            Status::CheckOnly => "check only",
        };
        let interface = if prog_if_known || interfaces == (u8::MAX as usize + 1) {
            ""
        } else {
            "; requires matching prog-if"
        };
        drivers.push(format!(
            "{} / {} ({status}{interface})",
            manifest.id, manifest.implementation
        ));
        match manifest.firmware {
            Firmware::None => {}
            Firmware::Required(names) => candidates.extend(names.iter().map(|name| (*name).into())),
            Firmware::ProfileSelected(source) => {
                candidates.push(format!("Live selection via {source}"));
            }
        }
    }

    if identity.class[0] == 3
        && identity.vendor == 0x10de
        && let Some(chip) = recipes::nvidia_chip(identity.device)
        && recipes::require_gsp(&mut firmware, chip).is_ok()
    {
        drivers.push(format!("nouveau / {chip} (PCI ID candidate)"));
    }
    candidates.extend(
        firmware
            .requests()
            .iter()
            .map(|request| request.name.clone()),
    );
    candidates.sort_unstable();
    candidates.dedup();
    Ok((drivers.join("; "), candidates))
}

fn describe_usb(device: &Usb) -> Vec<String> {
    let mut candidates = Vec::new();
    // The vendor Ethernet table only reads the two USB ID fields.
    let mut descriptor = [0; 18];
    descriptor[8..10].copy_from_slice(&device.vendor.to_le_bytes());
    descriptor[10..12].copy_from_slice(&device.product.to_le_bytes());
    if ethernet::listed(&descriptor) {
        candidates.push("USB Ethernet (ID candidate)".into());
    }
    for vendor_interface in [false, true] {
        if let Some(setup) = bluetooth::Setup::of(device.vendor, device.product, vendor_interface) {
            let label = format!("{} Bluetooth (ID candidate)", setup.name());
            if !candidates.contains(&label) {
                candidates.push(label);
            }
        }
    }
    candidates
}

fn or_dash(text: &str) -> &str {
    if text.is_empty() { "—" } else { text }
}

fn render(
    pci: Option<&[Identity]>,
    usb: Option<&[Usb]>,
    boot: Option<&BootReport>,
) -> Result<String> {
    let mut report = String::from(
        "# Hardware report\n\nDriver candidates need live probing and firmware checks. \
         Offline listings omit PCI programming interfaces and USB interface descriptors.\n",
    );
    if boot.is_some() {
        report.push_str("\nBoot observations cover the latest captured boot. Use an `lspci` listing from the same machine when correlating by address.\n");
    }
    let mut rows: BTreeMap<_, _> = pci
        .unwrap_or_default()
        .iter()
        .map(|identity| (identity.address, Some(identity)))
        .collect();
    if let Some(boot) = boot {
        for slot in boot.pci.keys() {
            rows.entry(*slot).or_insert(None);
        }
    }
    if pci.is_some() || boot.is_some() {
        report.push_str("\n## PCI devices\n\n");
        if rows.is_empty() {
            report.push_str("No PCI records in the supplied inputs.\n");
        } else {
            report
                .push_str("| Address | PCI ID | Class | Driver candidates | Firmware candidates |");
            if boot.is_some() {
                report.push_str(" Boot observations |");
            }
            report.push_str("\n| --- | --- | --- | --- | --- |");
            if boot.is_some() {
                report.push_str(" --- |");
            }
            report.push('\n');
        }
        for (&slot, &supplied) in &rows {
            let observed = boot.and_then(|boot| boot.pci.get(&slot));
            let agreed = observed.filter(|entry| entry.agrees_with(supplied));
            let identity = supplied.copied().or_else(|| agreed?.identity);
            let boot_class = agreed.and_then(|entry| entry.class);
            let class = if let Some([base, sub, interface]) = boot_class {
                format!("{base:02x}{sub:02x}{interface:02x}")
            } else if let Some(identity) = identity {
                format!("{:02x}{:02x}", identity.class[0], identity.class[1])
            } else {
                String::new()
            };
            if let Some(mut identity) = identity {
                if let Some(class) = boot_class {
                    identity.class = class;
                }
                let (drivers, firmware) = describe_pci(identity, boot_class.is_some())?;
                write!(
                    report,
                    "| {slot} | {:04x}:{:04x} | {class} | {} | {} |",
                    identity.vendor,
                    identity.device,
                    or_dash(&drivers),
                    or_dash(&firmware.join("; "))
                )?;
            } else {
                let id = agreed
                    .and_then(|entry| entry.glk_id.or(entry.lpss_id))
                    .map_or_else(String::new, |device| format!("8086:{device:04x}"));
                write!(
                    report,
                    "| {slot} | {} | {} | — | — |",
                    or_dash(&id),
                    or_dash(&class)
                )?;
            }
            if boot.is_some() {
                let outcome = if let Some(entry) = agreed {
                    entry.describe()
                } else if observed.is_some() {
                    "Conflicting PCI identity or class. Correlation withheld".into()
                } else {
                    String::new()
                };
                write!(report, " {} |", or_dash(&outcome))?;
            }
            report.push('\n');
        }
    }
    if let Some(boot) = boot
        && !boot.acpi.is_empty()
    {
        report.push_str(
            "\n## ACPI devices\n\n| Namespace path | Boot observations |\n| --- | --- |\n",
        );
        for (path, outcomes) in &boot.acpi {
            let mut resources = outcomes
                .iter()
                .filter(|outcome| matches!(outcome, AcpiOutcome::LpssResources(..)));
            let resources_conflict = resources.next().is_some() && resources.next().is_some();
            let outcomes = outcomes
                .iter()
                .map(|outcome| match outcome {
                    AcpiOutcome::LpssResources(..) if resources_conflict => {
                        "Conflicting ACPI resource observations. Resource details withheld".into()
                    }
                    AcpiOutcome::LpssResources(slot, _, _)
                        if !boot.pci[slot].agrees_with(rows[slot]) =>
                    {
                        "Conflicting PCI identity. ACPI resource correlation withheld".into()
                    }
                    _ => outcome.describe(),
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
                .join("; ");
            writeln!(report, "| `{path}` | {outcomes} |")?;
        }
    }
    if let Some(usb) = usb {
        report.push_str(
            "\n## USB devices\n\n| Bus | Device | USB ID | Drivers |\n| --- | --- | --- | --- |\n",
        );
        let mut usb: Vec<_> = usb.iter().collect();
        usb.sort_by_key(|device| (device.bus, device.address));
        for device in usb {
            writeln!(
                report,
                "| {:03} | {:03} | {:04x}:{:04x} | {} |",
                device.bus,
                device.address,
                device.vendor,
                device.product,
                or_dash(&describe_usb(device).join("; "))
            )?;
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_inputs_keep_only_device_fields() {
        let gpu = parse_pci("0002:00:02.0 VGA compatible controller [0300]: Intel UHD Graphics 600 [8086:3185] (rev 06)").unwrap();
        assert_eq!(gpu.address.segment, 2);
        assert_eq!(gpu.class, [3, 0, 0]);
        assert_eq!((gpu.vendor, gpu.device), (0x8086, 0x3185));
        let usb =
            parse_usb("Bus 003 Device 005: ID 8087:0aaa Intel Bluetooth serial=secret").unwrap();
        assert_eq!(
            usb,
            Usb {
                bus: 3,
                address: 5,
                vendor: 0x8087,
                product: 0x0aaa
            }
        );
        assert!(
            !render(Some(&[gpu]), Some(&[usb]), None)
                .unwrap()
                .contains("secret")
        );
    }

    #[test]
    fn malformed_rows_fail_without_echoing_their_contents() {
        for row in [
            "00:14.3 Network controller: Intel [8086:31dc]",
            "00:14.3 Network controller [0280]: Intel [8086:zzzz]",
            "00:20.3 Network controller [0280]: Intel [8086:31dc]",
            "00:14.8 Network controller [0280]: Intel [8086:31dc]",
        ] {
            assert!(parse_pci(row).is_none(), "{row}");
        }
        assert!(parse_usb("Bus 001 Device 000: ID 8087:0aaa").is_none());
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("lsusb.txt");
        std::fs::write(&file, "\nsecret serial\n").unwrap();
        let error = read(&file, "lsusb", parse_usb).unwrap_err().to_string();
        assert!(
            error.contains("line 2") && !error.contains("secret"),
            "{error}"
        );
    }

    #[test]
    fn lenovo_inventory_distinguishes_legacy_wifi_and_conditional_xhci() {
        let pci = [
            "00:15.0 Network controller [0280]: Intel CNVi [8086:31dc] (rev 06)",
            "00:02.0 VGA compatible controller [0300]: Intel UHD Graphics 600 [8086:3185]",
            "00:13.0 USB controller [0c03]: Intel xHCI [8086:31a8]",
        ]
        .map(|row| parse_pci(row).unwrap());
        let usb = parse_usb("Bus 001 Device 003: ID 0bda:8153 adapter aa:bb:cc:dd:ee:ff").unwrap();
        let output = render(Some(&pci), Some(&[usb]), None).unwrap();
        assert!(
            output.contains("DisplayInventory / pci-display (check only)"),
            "{output}"
        );
        assert!(output.contains("requires matching prog-if"), "{output}");
        assert!(
            output.contains("iwlwifi-9000-pu-b0-jf-b0-46.ucode"),
            "{output}"
        );
        assert!(output.contains("regulatory.db.p7s"), "{output}");
        assert!(
            output.contains("this generation has no driver here"),
            "{output}"
        );
        assert!(output.contains("USB Ethernet (ID candidate)"), "{output}");
        assert!(
            !output.contains("IntelWiFi / iwlwifi") && !output.contains("aa:bb"),
            "{output}"
        );
    }

    #[test]
    fn latest_boot_joins_census_and_probe_outcomes_without_exposing_log_details() {
        let log = b"\
=== STELA BOOT 1 ===\n\
STELA UEFI MOUNT version=0.1.0\r\n\
NETWORK link=0000:00:02.0 driver=virtio-net transport=Ethernet carrier=true\n\
=== STELA BOOT 2 ===\n\
[    0.000] STELA UEFI MOUNT version=0.1.0\r\n\
[    5.021] DEVICE PCI 0000:00:03.0 1af4:1000 class=020000 candidate=VirtioNet.nwdrv (virtio-net)\r\n\
DEVICE PCI 0000:00:04.0 1af4:1000 class=020000 candidate=VirtioNet.nwdrv (virtio-net)\n\
NETWORK candidate driver=virtio-net adapter=0000:00:03.0 description=secret mac=aa:bb:cc:dd:ee:ff\n\
NETWORK link=0000:00:03.0 driver=virtio-net transport=Ethernet carrier=true\n\
GUEST NETWORK adapter 0000:00:03.0 description=secret mac=aa:bb:cc:dd:ee:ff\n\
STORAGE DEVICE Nvme at 0000:00:05.0 namespace 1 serial=private model=private blocks=4 block_size=4096\n\
DISK Usb at xhci 0000:00:06.0 port 1 serial=private model=private blocks=4 block_size=512\n\
HOST DEVICE FAILED pci=0000:00:05.0 (storage) error=secret\n\
HOST DEVICE FAILED 0000:00:05.0 SD slot 0: secret\n\
HOST UNSUPPORTED pci=0000:00:07.0 class=010000: driver module disabled by nwconfig\n\
HOST UNSUPPORTED pci=0000:00:08.0 class=028000: unknown secret\n\
GUEST| HOST DEVICE FAILED pci=0000:00:09.0 (storage) error=secret\n\
HOST UNSUPPORTED pci=0000:00:0a.0 class=028000: truncated";
        let pci = [parse_pci("00:03.0 Ethernet controller [0200]: Virtio [1af4:1000]").unwrap()];
        let boot = boot_observations(log).unwrap();
        let output = render(Some(&pci), None, Some(&boot)).unwrap();
        let row = |address: &str| {
            output
                .lines()
                .find(|line| line.starts_with(&format!("| {address} |")))
        };
        let attached = row("0000:00:03.0").unwrap();
        assert_eq!(
            attached
                .matches("Network adapter attached (virtio-net)")
                .count(),
            1,
            "{output}"
        );
        assert!(
            attached.contains("Selected for guest networking"),
            "{output}"
        );
        assert!(row("0000:00:04.0").unwrap().ends_with(" | — |"), "{output}");
        let disk = row("0000:00:05.0").unwrap();
        assert!(
            disk.contains(
                "NVMe disk observed; PCI probe failed; Device behind this function failed"
            ),
            "{output}"
        );
        assert!(
            row("0000:00:07.0")
                .unwrap()
                .ends_with("Disabled by nwconfig |"),
            "{output}"
        );
        assert!(
            row("0000:00:08.0")
                .unwrap()
                .ends_with("Reported unsupported |"),
            "{output}"
        );
        for absent in [
            "0000:00:02.0",
            "0000:00:06.0",
            "0000:00:09.0",
            "0000:00:0a.0",
            "aa:bb",
            "private",
            "secret",
        ] {
            assert!(!output.contains(absent), "{output}");
        }
    }

    #[test]
    fn boot_boundary_and_conflicting_identities_prevent_stale_associations() {
        let log = b"\
STELA UEFI MOUNT version=0.1.0\n\
DEVICE PCI 0000:00:03.0 1af4:1000 class=020000 candidate=VirtioNet.nwdrv (virtio-net)\n\
NETWORK link=0000:00:03.0 driver=virtio-net transport=Ethernet carrier=false\n\
STELA UEFI MOUNT version=0.1.0\n\
DEVICE PCI 0000:00:04.0 1af4:1000 class=020000 candidate=VirtioNet.nwdrv (virtio-net)\n\
NETWORK link=0000:00:04.0 driver=virtio-net transport=Ethernet carrier=false\n\
HOST UNSUPPORTED pci=0000:00:05.0 class=010000: reason\n";
        let boot = boot_observations(log).unwrap();
        assert_eq!(boot.pci.len(), 2);
        let pci = [
            parse_pci("00:04.0 Ethernet controller [0200]: Intel [8086:100e]").unwrap(),
            parse_pci("00:05.0 USB controller [0c03]: Intel [8086:31a8]").unwrap(),
        ];
        let output = render(Some(&pci), None, Some(&boot)).unwrap();
        assert_eq!(
            output.matches("Conflicting PCI identity or class").count(),
            2,
            "{output}"
        );
        assert!(
            !output.contains("Network adapter attached")
                && !output.contains("Reported unsupported"),
            "{output}"
        );
        assert!(!output.contains("0000:00:03.0"), "{output}");
        let log = [log.as_slice(), b"=== STELA BOOT 9 ===\n"].concat();
        assert!(boot_observations(&log).unwrap().pci.is_empty());
        assert!(boot_observations(b"=== partition private ===\n").is_err());
        assert!(
            boot_observations(
                b"Boot log on the USB (private): present\nLoader boot log on the data volume (private):\n"
            )
            .is_err()
        );
    }

    #[test]
    fn observed_programming_interface_constrains_candidates_without_promoting_them() {
        let pci = [parse_pci("00:02.0 USB controller [0c03]: QEMU xHCI [1b36:000d]").unwrap()];
        let boot = boot_observations(
            b"DEVICE PCI 0000:00:02.0 1b36:000d class=0c0330 candidate=XHCI.nwdrv (xhci)\n",
        )
        .unwrap();
        let report = render(Some(&pci), None, Some(&boot)).unwrap();
        assert!(report.contains("1b36:000d | 0c0330 | XHCI"), "{report}");
        for absent in [
            "UHCI",
            "OHCI",
            "EHCI",
            "requires matching prog-if",
            "attached",
        ] {
            assert!(!report.contains(absent), "{report}");
        }
    }

    #[test]
    fn glk_memory_and_acpi_probe_outcomes_keep_their_actual_scope() {
        let log = br"=== STELA BOOT 1 ===
HOST GPU GLK pci=0000:00:09.0 device=3184 ggtt_bytes=2097152 stolen_base=0x0 stolen_bytes=0
INPUT host device I2C-HID \_SB_.OLD0
=== STELA BOOT 2 ===
[    5.021] HOST GPU GLK pci=0000:00:02.0 device=3185 ggtt_bytes=4194304 stolen_base=0x9be00000 stolen_bytes=67108864
HOST GPU GLK pci=0000:00:03.0 device=3184 memory unavailable: Config(secret hardware detail)
PLATFORM FAILED \_SB_.PCI0.NWIC.NWT0: native endpoint engine stop receipt unavailable
PLATFORM FAILED \_SB_.PCI0.NWIC.TPD1: HID-over-I2C _DSM did not return the descriptor register
INPUT host device I2C-HID \_SB_.PCI0.I2C1.TPD2
INPUT host device I2C-HID \_SB_.PCI0.I2C1.TPD2
";
        let boot = boot_observations(log).unwrap();
        let output = render(None, None, Some(&boot)).unwrap();
        for observed in [
            "| 0000:00:02.0 | 8086:3185 | — | — | — | GLK PCI memory inventory only (GGTT 4194304 bytes, stolen 67108864 bytes at 0x9be00000) |",
            "| 0000:00:03.0 | 8086:3184 | — | — | — | GLK PCI memory inventory unavailable |",
            "| `\\_SB_.PCI0.NWIC.NWT0` | Probe refused: native PCI endpoint stop receipt unavailable |",
            "| `\\_SB_.PCI0.NWIC.TPD1` | I²C-HID probe refused: _DSM did not return the descriptor register |",
            "| `\\_SB_.PCI0.I2C1.TPD2` | I²C-HID input device listed for host polling |",
        ] {
            assert_eq!(output.matches(observed).count(), 1, "{output}");
        }
        for absent in ["00:09.0", "OLD0", "secret", "driver started"] {
            assert!(!output.contains(absent), "{output}");
        }
    }

    #[test]
    fn mismatched_glk_ids_and_malformed_evidence_do_not_correlate() {
        let log = br"DEVICE PCI 0000:00:02.0 8086:3185 class=030000 inventory=display
HOST GPU GLK pci=0000:00:02.0 device=3184 ggtt_bytes=0 stolen_base=0x0 stolen_bytes=0
HOST GPU GLK pci=0000:00:03.0 device=3185 ggtt_bytes=2097152 stolen_base=0x0 stolen_bytes=0
HOST GPU GLK pci=0000:00:04.0 device=3185 ggtt_bytes=1 stolen_base=0x0 stolen_bytes=0
HOST GPU GLK pci=0000:00:05.0 device=3185 ggtt_bytes=0 stolen_base=0xfffffffffff00000 stolen_bytes=33554432
HOST GPU GLK pci=0000:00:06.0 device=9999 ggtt_bytes=0 stolen_base=0x0 stolen_bytes=0
PLATFORM FAILED \_SB_.PRIVATE|serial: native endpoint engine stop receipt unavailable
GUEST| PLATFORM FAILED \_SB_.GUES: native endpoint engine stop receipt unavailable
PLATFORM FAILED \_SB_.LAST: native endpoint engine stop receipt unavailable";
        let boot = boot_observations(log).unwrap();
        let supplied = [parse_pci("00:03.0 Ethernet [0200]: Intel [8086:3185]").unwrap()];
        let output = render(Some(&supplied), None, Some(&boot)).unwrap();
        assert_eq!(
            output.matches("Conflicting PCI identity or class").count(),
            2,
            "{output}"
        );
        assert!(!output.contains("GLK PCI memory inventory"), "{output}");
        for absent in ["00:04.0", "00:05.0", "00:06.0", "PRIVATE", "GUES", "LAST"] {
            assert!(!output.contains(absent), "{output}");
        }
        assert!(boot.acpi.is_empty());
    }

    #[test]
    fn captured_qemu_glk_attachment_keeps_uefi_and_post_ebs_evidence_on_the_same_row() {
        // Verbatim GLK lines from the 82V7 QEMU attachment-diagnostics boot.
        let boot = boot_observations(include_bytes!("report/glk-qemu.txt")).unwrap();
        let output = render(None, None, Some(&boot)).unwrap();
        let row = output
            .lines()
            .find(|line| line.starts_with("| 0000:00:06.0 | 8086:3185 |"))
            .unwrap();
        for observation in [
            "GLK PCI memory inventory only (GGTT 4194304 bytes, stolen 67108864 bytes at 0x9be00000)",
            "GLK UEFI (memory decode on, bus mastering on, BAR0 unassigned)",
            "GLK post-EBS PCI (memory decode on, bus mastering off, INTx masked and deasserted, containment verified, GOP on another PCI function)",
            "GLK attach refused (engine-stop receipt unavailable, firmware exclusion and IRQ drain unproven)",
            "GLK post-EBS capabilities (PM capability absent, MSI capability absent, MSI-X capability absent, PCIe capability absent)",
            "GLK post-EBS BAR0 configuration refused",
        ] {
            assert!(row.contains(observation), "{row}");
        }
        assert!(
            !row.contains("attached") && !row.contains("driver started"),
            "{row}"
        );
    }

    #[test]
    fn translated_root_bar_and_engine_stop_are_independent_observations() {
        let log = br"FW GLK ATTACH pci=0000:00:02.0 device=3185 mse=true bme=true bar0_bus=0xc0000000 bar0_host=0xb0000000 bar0_bytes=16777216 bar0_source=uefi-pci-io
[    5.021] HOST GLK ATTACH pci=0000:00:02.0 device=3185 mse=true bme=false intx_disabled=true intx_asserted=false pci_controls=Ok(()) gop=same-function-verified firmware_exclusion=unproven irq_drain=unproven attach=refused-no-engine-stop-receipt
HOST GLK ATTACH pci=0000:00:02.0 pm_state=Some(0) msi_enabled=Some(0) msix_enabled=None pcie_pending=Some(1)
HOST GLK ATTACH pci=0000:00:02.0 bar0_bus=0xc0000000 bar0_host=0x1c0000000 bar0_expected_bytes=16777216 bar0_source=native-root-window bar0=translated
HOST GLK ATTACH pci=0002:00:02.0 config=unavailable error=InvalidPlatformReply attach=refused-no-engine-stop-receipt
";
        let boot = boot_observations(log).unwrap();
        let output = render(None, None, Some(&boot)).unwrap();
        let mut rows = output.lines().filter(|line| line.starts_with("| 000"));
        let row = rows.next().unwrap();
        for observation in [
            "| 0000:00:02.0 | 8086:3185 |",
            "PCI_IO BAR0 bus 0xc0000000, host 0xb0000000, 16777216 bytes",
            "GLK post-EBS BAR0 bus 0xc0000000, host 0x1c0000000 (translation unsupported by GLK adapter)",
            "GLK BAR0 addresses differ across the UEFI and post-EBS observations",
            "GOP same PCI function (verified)",
            "GLK post-EBS capabilities (D0, MSI off, MSI-X capability absent, PCIe transactions pending)",
            "GLK attach refused (engine-stop receipt unavailable, firmware exclusion and IRQ drain unproven)",
        ] {
            assert!(row.contains(observation), "{row}");
        }
        let other = rows.next().unwrap();
        assert!(other.starts_with("| 0002:00:02.0 | — |"), "{other}");
        assert!(
            other.contains("GLK post-EBS PCI configuration unavailable"),
            "{other}"
        );
        assert!(!other.contains("0xc0000000"), "{other}");
    }

    #[test]
    fn conflicting_glk_phases_or_repeats_withhold_unreliable_attachment_details() {
        let log = br"FW GLK ATTACH pci=0000:00:02.0 device=3184 mse=true bme=true bar0_low=0x0 bar0_next=0x0 bar0=raw-bar0-refused
HOST GLK ATTACH pci=0000:00:02.0 device=3185 mse=true bme=false intx_disabled=true intx_asserted=false pci_controls=Ok(()) gop=absent firmware_exclusion=unproven irq_drain=unproven attach=refused-no-engine-stop-receipt
HOST GPU GLK pci=0000:00:03.0 device=3185 ggtt_bytes=0 stolen_base=0x0 stolen_bytes=0
FW GLK ATTACH pci=0000:00:03.0 device=3185 mse=true bme=true bar0_low=0x0 bar0_next=0x0 bar0=raw-bar0-refused
FW GLK ATTACH pci=0000:00:03.0 device=3185 mse=true bme=true bar0_low=0x0 bar0_next=0x0 bar0=raw-bar0-refused
FW GLK ATTACH pci=0000:00:03.0 device=3185 mse=true bme=false bar0_low=0x0 bar0_next=0x0 bar0=raw-bar0-refused
HOST GLK ATTACH pci=0000:00:04.0 bar0=raw-bar0-refused
";
        let boot = boot_observations(log).unwrap();
        let supplied = [parse_pci("00:04.0 Ethernet [0200]: Intel [8086:100e]").unwrap()];
        let output = render(Some(&supplied), None, Some(&boot)).unwrap();
        assert_eq!(
            output.matches("Conflicting PCI identity or class").count(),
            2,
            "{output}"
        );
        assert!(
            output.contains("Conflicting GLK attachment observations. Attachment details withheld"),
            "{output}"
        );
        assert!(output.contains("GLK PCI memory inventory only"), "{output}");
        assert!(
            !output.contains("GLK UEFI") && !output.contains("GLK attach refused"),
            "{output}"
        );
    }

    #[test]
    fn malformed_glk_attachment_lines_do_not_create_evidence() {
        let log = br"FW GLK ATTACH pci=0000:00:02.8 device=3185 mse=true bme=true bar0_low=0x0 bar0_next=0x0 bar0=raw-bar0-refused
FW GLK ATTACH pci=0000:00:03.0 device=3186 mse=true bme=true bar0_low=0x0 bar0_next=0x0 bar0=raw-bar0-refused
FW GLK ATTACH pci=0000:00:04.0 device=3185 mse=unknown bme=true bar0_low=0x0 bar0_next=0x0 bar0=raw-bar0-refused
FW GLK ATTACH pci=0000:00:05.0 device=3185 mse=true bme=true bar0_low=0x0 bar0_next=0x0 bar0=raw-bar0-refused extra=secret
HOST GLK ATTACH pci=0000:00:06.0 device=3185 mse=true bme=false intx_disabled=true intx_asserted=false pci_controls=Ok(()) gop=absent firmware_exclusion=unproven irq_drain=unproven attach=ready
HOST GLK ATTACH pci=0000:00:07.0 pm_state=Some(4) msi_enabled=None msix_enabled=None pcie_pending=None
HOST GLK ATTACH pci=0000:00:08.0 bar0_bus=0xc0000000 bar0_host=0xc0000000 bar0_expected_bytes=16777216 bar0_source=native-root-window bar0=translated
HOST GLK ATTACH pci=0000:00:09.0 config=unavailable error= attach=refused-no-engine-stop-receipt
HOST GLK ATTACH pci=0000:00:0a.0 bar0=raw-bar0-refused";
        let boot = boot_observations(log).unwrap();
        assert!(boot.pci.is_empty());
    }

    #[test]
    fn structured_lpss_probe_outcomes_require_valid_pci_and_acpi_fields() {
        let log = br"PLATFORM FAILED \_SB_.PCI0.I2C7.TPD0: LPSS I2C 0000:00:07.0 8086:31ba handoff: native endpoint engine stop receipt unavailable
PLATFORM FAILED \_SB_.PCI0.I2C7.TPD1: I2C ACPI controller 0000:00:06.0 1b36:000d is not a supported LPSS I2C function
PLATFORM FAILED \_SB_.BAD0: LPSS I2C 0000:00:07.8 8086:31ba handoff: native endpoint engine stop receipt unavailable
PLATFORM FAILED \_SB_.BAD1: LPSS I2C 0000:00:07.0 secret:31ba handoff: native endpoint engine stop receipt unavailable
PLATFORM FAILED \_SB_.BAD2: LPSS I2C 0000:00:07.0 8086:31ba handoff: native endpoint engine stop receipt unavailable secret
PLATFORM FAILED \_SB_.BAD3: I2C ACPI controller 00:06.0 1b36:000d is not a supported LPSS I2C function
PLATFORM FAILED \_SB_.BAD4|serial: LPSS I2C 0000:00:07.0 8086:31ba handoff: native endpoint engine stop receipt unavailable
PLATFORM FAILED \_SB_.BAD5: LPSS I2C 0000:00:07.0 8086:31ba handoff: native endpoint engine stop receipt unavailable";
        let boot = boot_observations(log).unwrap();
        assert_eq!(boot.acpi.len(), 2);
        let output = render(None, None, Some(&boot)).unwrap();
        assert!(output.contains(
            "| `\\_SB_.PCI0.I2C7.TPD0` | LPSS I²C 0000:00:07.0 8086:31ba refused: native PCI endpoint stop receipt unavailable |"
        ));
        assert!(output.contains(
            "| `\\_SB_.PCI0.I2C7.TPD1` | I²C-HID probe refused: ACPI controller 0000:00:06.0 1b36:000d is not a supported LPSS I²C function |"
        ));
        assert!(!output.contains("secret") && !output.contains("BAD"));
    }

    const LPSS_UEFI: &str = "FW LPSS-I2C pci=0000:00:07.0 id=8086:31ba command=0x0007 ioe=true mse=true bme=true intx-disabled=false bar0=advertised pci-handle-i2c-master=present pci-handle-i2c-host=absent stop-evidence=unavailable";
    const LPSS_FW_BAR: &str =
        "FW LPSS-I2C BAR0 pci=0000:00:07.0 id=8086:31ba bus=0x2000 host=0x8000 bytes=4096";
    const LPSS_HOST_BAR: &str = "HOST LPSS-I2C BAR0 pci=0000:00:07.0 id=8086:31ba bus=0x2000 host=0x8000 bytes=4096 state=translated";
    const LPSS_HOST_PCI: &str = "HOST LPSS-I2C PCI pci=0000:00:07.0 id=8086:31ba mse=false bme=false intx-disabled=true intx-asserted=false contained=true stop-evidence=unavailable";
    const LPSS_HOST_CAPS: &str = "HOST LPSS-I2C CAPS pci=0000:00:07.0 id=8086:31ba pm-state=Some(3) msi-enabled=None msix-enabled=Some(0) pcie-pending=Some(1)";
    const LPSS_ACPI: &str = r"HOST LPSS-I2C ACPI pci=0000:00:07.0 id=8086:31ba hid=SYNA2BA6 device=\_SB_.PCI0.I2C7.TPD0 controller=\_SB_.PCI0.I2C7 addr=0x2c addr-bits=7 speed=400000 irq=apic-polled descriptor=0x20";

    #[test]
    fn lpss_report_joins_firmware_and_original_root_by_exact_pci_identity() {
        let log = format!(
            "=== STELA BOOT 1 ===\n{}\n=== STELA BOOT 2 ===\n[    3.021] {LPSS_UEFI}\r\n{LPSS_UEFI}\n{LPSS_FW_BAR}\n{LPSS_HOST_PCI}\n{LPSS_HOST_CAPS}\n{LPSS_HOST_BAR}\n{LPSS_ACPI}\n{LPSS_ACPI}\n",
            LPSS_UEFI.replace("00:07.0", "00:08.0"),
        );
        let boot = boot_observations(log.as_bytes()).unwrap();
        let output = render(None, None, Some(&boot)).unwrap();
        assert!(
            output.contains("| 0000:00:07.0 | 8086:31ba | — |"),
            "{output}"
        );
        assert!(!output.contains("00:08.0"), "{output}");
        assert_eq!(
            output.matches("LPSS UEFI PCI_IO (command").count(),
            1,
            "{output}"
        );
        assert_eq!(
            output.matches("LPSS I²C resources for SYNA2BA6").count(),
            1,
            "{output}"
        );
        assert!(
            output.contains("(raw _CRS 7-bit address 0x2c, 400000 Hz, APIC polling, effective HID descriptor 0x20)"),
            "{output}"
        );
        assert!(
            output.contains("D3hot, MSI absent, MSI-X off, PCIe transactions pending"),
            "{output}"
        );
        assert!(
            output.contains("LPSS BAR0 range agrees with the original root window"),
            "{output}"
        );
        assert!(
            output.contains("LPSS I²C endpoint stop evidence unavailable"),
            "{output}"
        );
    }

    #[test]
    fn lpss_report_keeps_firmware_extent_and_host_refusals_distinct() {
        let log = format!(
            "{}\n{LPSS_HOST_BAR}\n{}\n{}\n",
            LPSS_FW_BAR.replace("bytes=4096", "bytes=8192"),
            LPSS_FW_BAR.replace("00:07.0", "00:08.0"),
            "HOST LPSS-I2C BAR0 pci=0000:00:08.0 id=8086:31ba state=original-root-unavailable",
        );
        let boot = boot_observations(log.as_bytes()).unwrap();
        let output = render(None, None, Some(&boot)).unwrap();
        assert!(
            output.contains("LPSS BAR0 base agrees with the original root window, extent differs"),
            "{output}"
        );
        assert!(
            output.contains("LPSS post-EBS BAR0 original PCI root unavailable"),
            "{output}"
        );
        assert!(!output.contains("range agrees"), "{output}");
    }

    #[test]
    fn malformed_lpss_records_do_not_add_evidence() {
        let log = [
            LPSS_UEFI.replace("ioe=true", "ioe=false"),
            LPSS_UEFI.replace("i2c-master=present", "i2c-master=secret"),
            LPSS_UEFI.replace("stop-evidence=unavailable", "stop-evidence=granted"),
            LPSS_FW_BAR.replace("31ba", "3185"),
            LPSS_FW_BAR.replace("bus=0x2000", "bus=0xfffffffffffff000"),
            LPSS_HOST_BAR.replace("bytes=4096", "bytes=8192"),
            LPSS_HOST_BAR.replace("state=translated", "state=identity"),
            LPSS_HOST_CAPS.replace("Some(3)", "Some(4)"),
            LPSS_ACPI.replace("SYNA2BA6", "secret"),
            LPSS_ACPI.replace("TPD0", "TPD|"),
            LPSS_ACPI.replace("addr=0x2c", "addr=0x100"),
            format!("{LPSS_HOST_PCI} extra=secret"),
            format!("GUEST| {LPSS_UEFI}"),
            // An incomplete trailing serial line never establishes evidence.
            LPSS_UEFI.to_owned(),
        ]
        .join("\n");
        let boot = boot_observations(log.as_bytes()).unwrap();
        assert!(boot.pci.is_empty() && boot.acpi.is_empty());
    }

    #[test]
    fn conflicting_lpss_records_withhold_pci_and_acpi_correlations() {
        let log = format!(
            "{LPSS_FW_BAR}\n{LPSS_HOST_BAR}\n{LPSS_ACPI}\n{}\n{}\n",
            LPSS_HOST_BAR.replace("host=0x8000", "host=0x9000"),
            LPSS_ACPI.replace("descriptor=0x20", "descriptor=0x2c"),
        );
        let boot = boot_observations(log.as_bytes()).unwrap();
        let output = render(None, None, Some(&boot)).unwrap();
        assert!(
            output.contains("Conflicting LPSS I²C PCI observations"),
            "{output}"
        );
        assert!(
            output.contains("Conflicting ACPI resource observations"),
            "{output}"
        );
        assert!(
            !output.contains("range agrees") && !output.contains("descriptor 0x"),
            "{output}"
        );

        let log = format!(
            "{LPSS_FW_BAR}\n{}\n{LPSS_ACPI}\n",
            LPSS_HOST_BAR.replace("31ba", "31b8")
        );
        let boot = boot_observations(log.as_bytes()).unwrap();
        let output = render(None, None, Some(&boot)).unwrap();
        assert!(
            output.contains("Conflicting PCI identity or class"),
            "{output}"
        );
        assert!(
            output.contains("Conflicting PCI identity. ACPI resource correlation withheld"),
            "{output}"
        );
        assert!(!output.contains("range agrees"), "{output}");
    }

    #[test]
    fn report_subcommand_accepts_combined_inputs_and_requires_at_least_one() {
        use clap::Parser;
        assert!(crate::Cli::try_parse_from(["stela", "report"]).is_err());
        assert!(crate::Cli::try_parse_from(["stela", "report", "--lspci", "pci.txt"]).is_ok());
        assert!(crate::Cli::try_parse_from(["stela", "report", "--lsusb", "usb.txt"]).is_ok());
        assert!(crate::Cli::try_parse_from(["stela", "report", "--boot-log", "boot.txt"]).is_ok());
        let command = crate::Cli::try_parse_from([
            "stela",
            "report",
            "--lspci",
            "pci.txt",
            "--lsusb",
            "usb.txt",
            "--boot-log",
            "boot.txt",
        ])
        .unwrap();
        assert!(matches!(command.command, crate::Command::Report(_)));
    }
}
