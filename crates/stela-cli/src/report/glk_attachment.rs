// SPDX-License-Identifier: GPL-2.0-or-later
//! Firmware and post-EBS GLK attachment observations from the boot log.

use super::{Address, address, hex};
use std::str::SplitAsciiWhitespace;
use stela_boot_core::firmware_pci::glk::BAR0_BYTES;
use stela_drivers::bus::mmio::owned::PAGE_BYTES;

const GGTT_OFFSET: u64 = BAR0_BYTES / 2;
const MIB: u64 = 1 << 20;
const PTE_BYTES: u64 = size_of::<u64>() as u64;
const GPU_PAGE_BYTES: u64 = 4096;

#[derive(Default)]
pub(super) struct Attachment {
    firmware: Option<Firmware>,
    ggtt: Option<FirmwareGgtt>,
    ggtt_conflict: bool,
    host: Option<Host>,
    capabilities: Option<Option<Capabilities>>,
    bar: Option<HostBar>,
    conflict: bool,
}

pub(super) enum Record {
    Firmware(u16, Firmware),
    Ggtt(FirmwareGgtt),
    Host(u16, Controls),
    ConfigUnavailable,
    Capabilities(Option<Capabilities>),
    Bar(HostBar),
}

impl Record {
    pub(super) fn device(&self) -> Option<u16> {
        match self {
            Self::Firmware(device, _) | Self::Host(device, _) => Some(*device),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Firmware {
    mse: bool,
    bme: bool,
    bar: FirmwareBar,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FirmwareBar {
    Descriptor { bus: u64, host: u64 },
    Unassigned,
    Invalid,
    DescriptorUnavailable,
    DescriptorMismatch,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FirmwareGgtt {
    Observed {
        pci_start: u64,
        pci_end: u64,
        entries: u64,
        gpu_va_bytes: u64,
    },
    Disabled,
    ConfigUnavailable,
    Refused,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Host {
    Controls(Controls),
    ConfigUnavailable,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Controls {
    mse: bool,
    bme: bool,
    intx_disabled: bool,
    intx_asserted: bool,
    contained: bool,
    gop: &'static str,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Capabilities {
    pm: Option<u8>,
    msi: Option<u8>,
    msix: Option<u8>,
    pending: Option<u8>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum HostBar {
    Invalid,
    Unresolved {
        bus: u64,
        reason: &'static str,
    },
    RootWindow {
        bus: u64,
        host: u64,
        status: BarStatus,
        original: bool,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum BarStatus {
    Identity,
    Translated,
    HostMemoryConflict,
    InvalidHostRange,
}

impl Attachment {
    pub(super) fn observed(&self) -> bool {
        self.firmware.is_some()
            || self.host.is_some()
            || self.capabilities.is_some()
            || self.bar.is_some()
    }

    pub(super) fn observe(&mut self, record: Record) {
        self.conflict |= match record {
            Record::Firmware(_, value) => changed(&mut self.firmware, value),
            Record::Ggtt(ggtt) => {
                self.ggtt_conflict |= changed(&mut self.ggtt, ggtt);
                return;
            }
            Record::Host(_, controls) => changed(&mut self.host, Host::Controls(controls)),
            Record::ConfigUnavailable => changed(&mut self.host, Host::ConfigUnavailable),
            Record::Capabilities(value) => changed(&mut self.capabilities, value),
            Record::Bar(value) => changed(&mut self.bar, value),
        };
    }

    pub(super) fn describe(&self) -> Vec<String> {
        if self.conflict {
            return vec![
                "Conflicting GLK attachment observations. Attachment details withheld".into(),
            ];
        }
        let mut notes = Vec::new();
        if let Some(Firmware { mse, bme, bar }) = self.firmware {
            let bar = match bar {
                FirmwareBar::Descriptor { bus, host } => {
                    format!("PCI_IO BAR0 bus {bus:#x}, host {host:#x}, {BAR0_BYTES} bytes")
                }
                FirmwareBar::Unassigned => "BAR0 unassigned".into(),
                FirmwareBar::Invalid => "BAR0 configuration refused".into(),
                FirmwareBar::DescriptorUnavailable => "BAR0 descriptor unavailable".into(),
                FirmwareBar::DescriptorMismatch => {
                    "BAR0 descriptor disagrees with PCI configuration".into()
                }
            };
            notes.push(format!(
                "GLK UEFI (memory decode {}, bus mastering {}, {bar})",
                on_off(mse),
                on_off(bme)
            ));
        }
        notes.extend(self.describe_ggtt());
        match self.host {
            Some(Host::Controls(controls)) => {
                let pci = if controls.contained {
                    "verified"
                } else {
                    "recheck refused"
                };
                let intx = if controls.intx_disabled {
                    "masked"
                } else {
                    "unmasked"
                };
                let asserted = if controls.intx_asserted {
                    "asserted"
                } else {
                    "deasserted"
                };
                notes.push(format!(
                    "GLK post-EBS PCI (memory decode {}, bus mastering {}, INTx {intx} and {asserted}, containment {pci}, GOP {})",
                    on_off(controls.mse), on_off(controls.bme), controls.gop
                ));
                notes.push("GLK attach refused (engine-stop receipt unavailable, firmware exclusion and IRQ drain unproven)".into());
            }
            Some(Host::ConfigUnavailable) => {
                notes.push("GLK post-EBS PCI configuration unavailable".into());
                notes.push("GLK attach refused (engine-stop receipt unavailable)".into());
            }
            None => {}
        }
        if let Some(capabilities) = self.capabilities {
            notes.push(match capabilities {
                Some(Capabilities {
                    pm,
                    msi,
                    msix,
                    pending,
                }) => {
                    let pm = match pm {
                        Some(0) => "D0",
                        Some(1) => "D1",
                        Some(2) => "D2",
                        Some(3) => "D3hot",
                        _ => "PM capability absent",
                    };
                    let pending = match pending {
                        Some(0) => "transactions clear",
                        Some(_) => "transactions pending",
                        None => "capability absent",
                    };
                    format!(
                        "GLK post-EBS capabilities ({pm}, MSI {}, MSI-X {}, PCIe {pending})",
                        enabled(msi),
                        enabled(msix)
                    )
                }
                None => "GLK post-EBS PCI capabilities unavailable".into(),
            });
        }
        if let Some(bar) = self.bar {
            notes.push(match bar {
                HostBar::Invalid => "GLK post-EBS BAR0 configuration refused".into(),
                HostBar::Unresolved { bus, reason } => {
                    format!("GLK post-EBS BAR0 bus {bus:#x} ({reason})")
                }
                HostBar::RootWindow {
                    bus, host, status, ..
                } => {
                    let detail = match status {
                        BarStatus::Identity => "identity translation observed",
                        BarStatus::Translated => "translation unsupported by GLK adapter",
                        BarStatus::HostMemoryConflict => "host memory conflict",
                        BarStatus::InvalidHostRange => "invalid host range",
                    };
                    format!("GLK post-EBS BAR0 bus {bus:#x}, host {host:#x} ({detail})")
                }
            });
        }
        if let Some(Firmware {
            bar: FirmwareBar::Descriptor { bus, host },
            ..
        }) = self.firmware
            && let Some(HostBar::RootWindow {
                bus: post_bus,
                host: post_host,
                status,
                original,
            }) = self.bar
        {
            if (bus, host) != (post_bus, post_host) {
                notes.push(
                    "GLK BAR0 addresses differ across the UEFI and post-EBS observations".into(),
                );
            } else if original && status == BarStatus::Identity {
                notes.push(
                    "GLK BAR0 ranges agree across UEFI PCI_IO and the original PCI root window"
                        .into(),
                );
            }
        }
        notes
    }

    fn describe_ggtt(&self) -> Option<String> {
        if self.ggtt_conflict {
            return Some("Conflicting GLK UEFI GGTT observations. Geometry withheld".into());
        }
        Some(match self.ggtt? {
            FirmwareGgtt::Observed {
                pci_start,
                pci_end,
                entries,
                gpu_va_bytes,
            } => {
                let matches_bar = matches!(self.firmware, Some(Firmware {
                    bar: FirmwareBar::Descriptor { bus, .. }, ..
                }) if bus.checked_add(GGTT_OFFSET) == Some(pci_start));
                if matches_bar {
                    format!(
                        "GLK UEFI GGTT (PCI bus {pci_start:#x}..{pci_end:#x} exclusive, {entries} entries, GPU VA {gpu_va_bytes} bytes)"
                    )
                } else {
                    "GLK UEFI GGTT geometry withheld (matching PCI_IO BAR0 descriptor unavailable or disagrees)".into()
                }
            }
            FirmwareGgtt::Disabled => "GLK UEFI GGTT disabled".into(),
            FirmwareGgtt::ConfigUnavailable => "GLK UEFI GGTT configuration unavailable".into(),
            FirmwareGgtt::Refused => "GLK UEFI GGTT observation refused".into(),
        })
    }
}

fn changed<T: PartialEq>(old: &mut Option<T>, value: T) -> bool {
    let conflict = old.as_ref().is_some_and(|previous| *previous != value);
    *old = Some(value);
    conflict
}

fn on_off(value: bool) -> &'static str {
    if value { "on" } else { "off" }
}

fn enabled(value: Option<u8>) -> &'static str {
    value.map_or("capability absent", |value| on_off(value != 0))
}

pub(super) fn parse(line: &str) -> Option<(Address, Record)> {
    let (prefix, fields) = line.split_once(" pci=")?;
    let (slot, fields) = fields.split_once(' ')?;
    if slot.len() != 12 {
        return None;
    }
    let record = match prefix {
        "FW GLK ATTACH" => parse_firmware(fields)?,
        "FW GLK GGTT" => Record::Ggtt(parse_ggtt(fields)?),
        "HOST GLK ATTACH" => parse_host(fields)?,
        _ => return None,
    };
    Some((address(slot)?, record))
}

fn parse_host(fields: &str) -> Option<Record> {
    if fields.starts_with("device=") {
        parse_controls(fields)
    } else if let Some(detail) = fields.strip_prefix("config=unavailable error=") {
        let detail = detail.strip_suffix(" attach=refused-no-engine-stop-receipt")?;
        if detail.is_empty() {
            return None;
        }
        Some(Record::ConfigUnavailable)
    } else if fields.starts_with("pm_state=") {
        parse_capabilities(fields)
    } else if let Some(detail) = fields.strip_prefix("capabilities=unavailable error=") {
        if detail.is_empty() {
            return None;
        }
        Some(Record::Capabilities(None))
    } else {
        parse_host_bar(fields).map(Record::Bar)
    }
}

fn value<'a>(words: &mut SplitAsciiWhitespace<'a>, key: &str) -> Option<&'a str> {
    words.next()?.strip_prefix(key)
}

fn boolean(word: &str) -> Option<bool> {
    match word {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn device(word: &str) -> Option<u16> {
    let device = hex(word, 4)?;
    matches!(device, 0x3184 | 0x3185).then_some(device)
}

fn number(word: &str) -> Option<u64> {
    let word = word.strip_prefix("0x")?;
    if word.is_empty() || !word.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(word, 16).ok()
}

fn bus_address(word: &str) -> Option<u64> {
    let bus = number(word)?;
    (bus != 0 && bus.is_multiple_of(BAR0_BYTES)).then_some(bus)
}

fn valid_bar_length(word: &str) -> bool {
    word.bytes().all(|byte| byte.is_ascii_digit()) && word.parse::<u64>().ok() == Some(BAR0_BYTES)
}

fn parse_ggtt(fields: &str) -> Option<FirmwareGgtt> {
    if fields == "status=disabled" {
        return Some(FirmwareGgtt::Disabled);
    }
    if let Some(detail) = fields.strip_prefix("status=config-unavailable error=") {
        return (!detail.trim_ascii().is_empty()).then_some(FirmwareGgtt::ConfigUnavailable);
    }
    if let Some(detail) = fields.strip_prefix("status=refused error=") {
        return (!detail.trim_ascii().is_empty()).then_some(FirmwareGgtt::Refused);
    }

    fn decimal(word: &str) -> Option<u64> {
        if !word.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        word.parse().ok()
    }

    let mut words = fields.split_ascii_whitespace();
    if words.next()? != "status=observed" {
        return None;
    }
    let pci_start = number(value(&mut words, "pci_start=")?)?;
    let pci_end = number(value(&mut words, "pci_end_exclusive=")?)?;
    let entries = decimal(value(&mut words, "entries=")?)?;
    let gpu_va_bytes = decimal(value(&mut words, "gpu_va_bytes=")?)?;
    let bus = pci_start.checked_sub(GGTT_OFFSET)?;
    let bar_end = bus.checked_add(BAR0_BYTES)?;
    let table_bytes = pci_end.checked_sub(pci_start)?;
    if bus == 0
        || !bus.is_multiple_of(BAR0_BYTES)
        || pci_end > bar_end
        || !table_bytes.is_power_of_two()
        || !(2 * MIB..=8 * MIB).contains(&table_bytes)
        || entries != table_bytes / PTE_BYTES
        || gpu_va_bytes != (table_bytes / PTE_BYTES) * GPU_PAGE_BYTES
        || words.next()? != "source=uefi-pci-io"
        || words.next().is_some()
    {
        return None;
    }
    Some(FirmwareGgtt::Observed {
        pci_start,
        pci_end,
        entries,
        gpu_va_bytes,
    })
}

fn parse_firmware(fields: &str) -> Option<Record> {
    let mut words = fields.split_ascii_whitespace();
    let device = device(value(&mut words, "device=")?)?;
    let mse = boolean(value(&mut words, "mse=")?)?;
    let bme = boolean(value(&mut words, "bme=")?)?;
    let field = words.next()?;
    let bar = if let Some(bus) = field.strip_prefix("bar0_bus=") {
        let bus = bus_address(bus)?;
        let host = number(value(&mut words, "bar0_host=")?)?;
        if host > bus
            || bus.checked_add(BAR0_BYTES).is_none()
            || !valid_bar_length(value(&mut words, "bar0_bytes=")?)
            || words.next()? != "bar0_source=uefi-pci-io"
        {
            return None;
        }
        FirmwareBar::Descriptor { bus, host }
    } else {
        let low = u32::try_from(number(field.strip_prefix("bar0_low=")?)?).ok()?;
        let high = u32::try_from(number(value(&mut words, "bar0_next=")?)?).ok()?;
        match value(&mut words, "bar0=")? {
            "raw-bar0-refused" if low == 0 || (low == 4 && high == 0) => FirmwareBar::Unassigned,
            "raw-bar0-refused" => FirmwareBar::Invalid,
            "descriptor-unavailable" => FirmwareBar::DescriptorUnavailable,
            "descriptor-mismatch" => FirmwareBar::DescriptorMismatch,
            _ => return None,
        }
    };
    words
        .next()
        .is_none()
        .then_some(Record::Firmware(device, Firmware { mse, bme, bar }))
}

fn parse_controls(fields: &str) -> Option<Record> {
    let (head, tail) = fields.split_once(" pci_controls=")?;
    let (containment, tail) = tail.rsplit_once(" gop=")?;
    let contained = if containment == "Ok(())" {
        true
    } else if containment.starts_with("Err(ContainmentError { ") && containment.ends_with(" })") {
        false
    } else {
        return None;
    };
    let mut head = head.split_ascii_whitespace();
    let device = device(value(&mut head, "device=")?)?;
    let mse = boolean(value(&mut head, "mse=")?)?;
    let bme = boolean(value(&mut head, "bme=")?)?;
    let intx_disabled = boolean(value(&mut head, "intx_disabled=")?)?;
    let intx_asserted = boolean(value(&mut head, "intx_asserted=")?)?;
    let mut tail = tail.split_ascii_whitespace();
    let gop = match tail.next()? {
        "absent" => "absent",
        "same-function-verified" => "same PCI function (verified)",
        "same-function-unverified" => "same PCI function (unverified)",
        "other-function" => "on another PCI function",
        "source-unavailable" => "PCI source unavailable",
        _ => return None,
    };
    if head.next().is_some()
        || tail.next()? != "firmware_exclusion=unproven"
        || tail.next()? != "irq_drain=unproven"
        || tail.next()? != "attach=refused-no-engine-stop-receipt"
        || tail.next().is_some()
    {
        return None;
    }
    Some(Record::Host(
        device,
        Controls {
            mse,
            bme,
            intx_disabled,
            intx_asserted,
            contained,
            gop,
        },
    ))
}

fn parse_capabilities(fields: &str) -> Option<Record> {
    fn read_option(word: &str, max: u8) -> Option<Option<u8>> {
        if word == "None" {
            return Some(None);
        }
        let word = word.strip_prefix("Some(")?.strip_suffix(')')?;
        let [digit] = word.as_bytes() else {
            return None;
        };
        let value = digit.checked_sub(b'0')?;
        (value <= max).then_some(Some(value))
    }
    let mut words = fields.split_ascii_whitespace();
    let pm = read_option(value(&mut words, "pm_state=")?, 3)?;
    let msi = read_option(value(&mut words, "msi_enabled=")?, 1)?;
    let msix = read_option(value(&mut words, "msix_enabled=")?, 1)?;
    let pending = read_option(value(&mut words, "pcie_pending=")?, 1)?;
    words
        .next()
        .is_none()
        .then_some(Record::Capabilities(Some(Capabilities {
            pm,
            msi,
            msix,
            pending,
        })))
}

fn parse_host_bar(fields: &str) -> Option<HostBar> {
    if fields == "bar0=raw-bar0-refused" {
        return Some(HostBar::Invalid);
    }
    let mut words = fields.split_ascii_whitespace();
    let bus = bus_address(value(&mut words, "bar0_bus=")?)?;
    let field = words.next()?;
    if let Some(reason) = field.strip_prefix("bar0=") {
        let reason = match reason {
            "no-covering-root-window" => "no covering root window",
            "ambiguous-root-window" => "multiple original root windows describe the same host span",
            "conflicting-root-windows" => "original root windows describe conflicting host spans",
            "invalid-root-translation" => "original root translation overflows",
            "original-root-unavailable" => "original PCI root unavailable",
            _ => return None,
        };
        return words
            .next()
            .is_none()
            .then_some(HostBar::Unresolved { bus, reason });
    }
    let host = number(field.strip_prefix("bar0_host=")?)?;
    if host < bus || !valid_bar_length(value(&mut words, "bar0_expected_bytes=")?) {
        return None;
    }
    let original = match value(&mut words, "bar0_source=")? {
        "native-root-window" => false,
        "original-root-window" => true,
        _ => return None,
    };
    let valid = host.is_multiple_of(PAGE_BYTES) && host.checked_add(BAR0_BYTES).is_some();
    let status = match value(&mut words, "bar0=")? {
        "identity" if valid && bus == host => BarStatus::Identity,
        "translated" if valid && bus != host => BarStatus::Translated,
        "host-memory-conflict" if valid => BarStatus::HostMemoryConflict,
        "invalid-host-range" if !valid => BarStatus::InvalidHostRange,
        _ => return None,
    };
    words.next().is_none().then_some(HostBar::RootWindow {
        bus,
        host,
        status,
        original,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIRMWARE: &str = "FW GLK ATTACH pci=0000:00:02.0 device=3185 mse=true bme=true bar0_bus=0xc0000000 bar0_host=0xc0000000 bar0_bytes=16777216 bar0_source=uefi-pci-io";
    const ORIGINAL: &str = "HOST GLK ATTACH pci=0000:00:02.0 bar0_bus=0xc0000000 bar0_host=0xc0000000 bar0_expected_bytes=16777216 bar0_source=original-root-window bar0=identity";
    const AGREEMENT: &str =
        "GLK BAR0 ranges agree across UEFI PCI_IO and the original PCI root window";
    const GGTT: &str = "FW GLK GGTT pci=0000:00:02.0 status=observed pci_start=0xc0800000 pci_end_exclusive=0xc0c00000 entries=524288 gpu_va_bytes=2147483648 source=uefi-pci-io";
    const NO_STOP: &str = "HOST GLK ATTACH pci=0000:00:02.0 config=unavailable error=InvalidPlatformReply attach=refused-no-engine-stop-receipt";

    fn report(host: &str) -> String {
        let (firmware_address, firmware) = parse(FIRMWARE).unwrap();
        let (host_address, host) = parse(host).unwrap();
        assert_eq!(firmware_address, host_address);
        let mut attachment = Attachment::default();
        attachment.observe(firmware);
        attachment.observe(host);
        attachment.describe().join("\n")
    }

    #[test]
    fn range_agreement_requires_a_unique_original_root_window() {
        assert!(report(ORIGINAL).contains(AGREEMENT));
        let legacy = ORIGINAL.replace("original-root-window", "native-root-window");
        assert!(!report(&legacy).contains(AGREEMENT));
        let conflict = ORIGINAL.replace("bar0=identity", "bar0=host-memory-conflict");
        assert!(!report(&conflict).contains(AGREEMENT));
        let changed = ORIGINAL.replace("0xc0000000", "0xd0000000");
        assert!(
            report(&changed).contains("addresses differ across the UEFI and post-EBS observations")
        );
    }

    #[test]
    fn ambiguous_or_conflicting_root_windows_have_no_selected_host() {
        for (status, expected) in [
            (
                "ambiguous-root-window",
                "multiple original root windows describe the same host span",
            ),
            (
                "conflicting-root-windows",
                "original root windows describe conflicting host spans",
            ),
        ] {
            let host =
                format!("HOST GLK ATTACH pci=0000:00:02.0 bar0_bus=0xc0000000 bar0={status}");
            let output = report(&host);
            assert!(output.contains(expected), "{output}");
            assert!(
                !output.contains(AGREEMENT)
                    && !output.contains("post-EBS BAR0 bus 0xc0000000, host")
            );
            assert!(parse(&format!("{host} bar0_host=0xc0000000")).is_none());
        }
    }

    fn render_log(log: &str) -> String {
        let boot = crate::report::boot_observations(log.as_bytes()).unwrap();
        crate::report::render(None, None, Some(&boot)).unwrap()
    }

    #[test]
    fn ggtt_numbers_require_the_matching_firmware_bar_on_the_same_pci_function() {
        let firmware = FIRMWARE.replace("bar0_host=0xc0000000", "bar0_host=0xb0000000");
        let other_segment = GGTT.replace("0000:00:02.0", "0002:00:02.0");
        let output = render_log(&format!("{GGTT}\n{other_segment}\n{firmware}\n{NO_STOP}\n"));
        let mut rows = output.lines().filter(|line| line.starts_with("| 000"));
        let matching = rows.next().unwrap();
        assert!(matching.contains("| 0000:00:02.0 | 8086:3185 |"));
        assert!(matching.contains(
            "PCI bus 0xc0800000..0xc0c00000 exclusive, 524288 entries, GPU VA 2147483648 bytes"
        ));
        assert!(matching.contains("GLK attach refused (engine-stop receipt unavailable)"));
        let orphan = rows.next().unwrap();
        assert!(orphan.starts_with("| 0002:00:02.0 | — |"));
        assert!(orphan.contains("GGTT geometry withheld"));
        assert!(!orphan.contains("524288"));

        let wrong_bar = GGTT.replace("0xc0", "0xd0");
        let output = render_log(&format!("{firmware}\n{wrong_bar}\n"));
        assert!(output.contains("GGTT geometry withheld"));
        assert!(!output.contains("524288"));
    }

    #[test]
    fn ggtt_wire_checks_span_counts_and_source() {
        for (before, after) in [
            ("00:02.0", "00:02.8"),
            ("pci_start=0xc0800000", "pci_start=0xc0800001"),
            (
                "pci_end_exclusive=0xc0c00000",
                "pci_end_exclusive=0xc0b00000",
            ),
            (
                "pci_end_exclusive=0xc0c00000",
                "pci_end_exclusive=0xc0000000",
            ),
            ("entries=524288", "entries=524289"),
            ("entries=524288", "entries=+524288"),
            ("gpu_va_bytes=2147483648", "gpu_va_bytes=2147483649"),
            ("source=uefi-pci-io", "source=original-root-window"),
            ("source=uefi-pci-io", "source=uefi-pci-io extra=field"),
        ] {
            let malformed = GGTT.replace(before, after);
            assert!(parse(&malformed).is_none(), "{malformed}");
        }
    }

    #[test]
    fn ggtt_status_and_conflict_do_not_change_attachment() {
        let prefix = "FW GLK GGTT pci=0000:00:02.0";
        for (status, description) in [
            ("status=disabled", "disabled"),
            (
                "status=config-unavailable error=Firmware(DEVICE_ERROR)",
                "configuration unavailable",
            ),
            ("status=refused error=PRIVATE", "observation refused"),
        ] {
            let output = render_log(&format!("{FIRMWARE}\n{NO_STOP}\n{prefix} {status}\n"));
            assert!(output.contains(&format!("GLK UEFI GGTT {description}")));
            assert!(output.contains("GLK attach refused (engine-stop receipt unavailable)"));
            assert!(!output.contains("PRIVATE"));
        }
        assert!(parse(&format!("{prefix} status=refused error=")).is_none());
        assert!(parse(&format!("{prefix} status=disabled extra=field")).is_none());

        let output = render_log(&format!(
            "{FIRMWARE}\n{NO_STOP}\n{GGTT}\n{prefix} status=disabled\n"
        ));
        assert!(output.contains("Conflicting GLK UEFI GGTT observations. Geometry withheld"));
        assert!(output.contains("GLK attach refused (engine-stop receipt unavailable)"));
        assert!(!output.contains("524288"));
    }
}
