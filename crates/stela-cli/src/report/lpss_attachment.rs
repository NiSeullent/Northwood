// SPDX-License-Identifier: GPL-2.0-or-later
//! Read-only LPSS I²C observations from a single boot.

use super::{Address, address, hex, ids, valid_acpi_path};
use std::str::SplitAsciiWhitespace;
use stela_boot_core::firmware_pci::lpss::{self, BAR0_PROBE_BYTES};
use stela_drivers::bus::{lpss as lpss_driver, mmio::owned::PAGE_BYTES, pci::regs};

#[derive(Default)]
pub(super) struct Attachment {
    firmware: Option<Firmware>,
    firmware_bar: Option<FirmwareBar>,
    host_bar: Option<HostBar>,
    host_pci: Option<HostPci>,
    host_caps: Option<Option<Capabilities>>,
    conflict: bool,
}

pub(super) enum Record<'a> {
    Firmware(Firmware),
    FirmwareBar(FirmwareBar),
    HostBar(HostBar),
    HostPci(HostPci),
    HostCaps(Option<Capabilities>),
    Acpi(&'a str, Acpi),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Firmware {
    command: u16,
    bar: &'static str,
    master: &'static str,
    host: &'static str,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct HostPci {
    mse: bool,
    bme: bool,
    intx_disabled: bool,
    intx_asserted: bool,
    contained: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Capabilities {
    pm: Option<u8>,
    msi: Option<u8>,
    msix: Option<u8>,
    pending: Option<u8>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FirmwareBar {
    Descriptor(Range),
    Unavailable(&'static str),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum HostBar {
    Window(Range, &'static str),
    Unavailable(&'static str),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Range {
    bus: u64,
    host: u64,
    bytes: u64,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Acpi {
    hid: String,
    controller: String,
    address: u16,
    bits: u8,
    speed: u32,
    irq: &'static str,
    descriptor: u16,
}

impl Acpi {
    pub(super) fn describe(&self, slot: Address, device: u16) -> String {
        format!(
            "LPSS I²C resources for {} on `{}`, PCI {slot} 8086:{device:04x} (raw _CRS {}-bit address {:#x}, {} Hz, {}, effective HID descriptor {:#x})",
            self.hid,
            self.controller,
            self.bits,
            self.address,
            self.speed,
            self.irq,
            self.descriptor
        )
    }
}

impl Attachment {
    pub(super) fn observe(&mut self, record: Record<'_>) {
        fn changed<T: PartialEq>(slot: &mut Option<T>, value: T) -> bool {
            let conflict = slot.as_ref().is_some_and(|previous| *previous != value);
            *slot = Some(value);
            conflict
        }

        self.conflict |= match record {
            Record::Firmware(value) => changed(&mut self.firmware, value),
            Record::FirmwareBar(value) => changed(&mut self.firmware_bar, value),
            Record::HostBar(value) => changed(&mut self.host_bar, value),
            Record::HostPci(value) => changed(&mut self.host_pci, value),
            Record::HostCaps(value) => changed(&mut self.host_caps, value),
            Record::Acpi(..) => return,
        };
    }

    pub(super) fn describe(&self) -> Vec<String> {
        if self.conflict
            || matches!(self.firmware_bar, Some(FirmwareBar::Descriptor(_)))
                && self.firmware.is_some_and(|value| value.bar != "advertised")
        {
            return vec![
                "Conflicting LPSS I²C PCI observations. Attachment details withheld".into(),
            ];
        }
        let mut notes = Vec::new();
        if let Some(Firmware {
            command,
            bar,
            master,
            host,
        }) = self.firmware
        {
            let on = |mask| on_off(command & mask != 0);
            notes.push(format!(
                "LPSS UEFI PCI_IO (command {command:#06x}, I/O decode {}, memory decode {}, bus mastering {}, INTx {}, BAR0 {bar}, I²C Master {master}, I²C Host {host})",
                on(regs::COMMAND_IO), on(regs::COMMAND_MEMORY), on(regs::COMMAND_MASTER),
                masked(command & regs::COMMAND_INTX_DISABLE != 0),
            ));
        }
        if let Some(firmware) = self.firmware_bar {
            notes.push(match firmware {
                FirmwareBar::Descriptor(range) => format!("LPSS UEFI PCI_IO BAR0 {range}"),
                FirmwareBar::Unavailable(reason) => format!("LPSS UEFI PCI_IO BAR0 {reason}"),
            });
        }
        if let Some(HostPci {
            mse,
            bme,
            intx_disabled,
            intx_asserted,
            contained,
        }) = self.host_pci
        {
            notes.push(format!(
                "LPSS post-EBS PCI (memory decode {}, bus mastering {}, INTx {} and {}, containment {})",
                on_off(mse), on_off(bme), masked(intx_disabled),
                if intx_asserted { "asserted" } else { "deasserted" },
                if contained { "verified" } else { "recheck refused" },
            ));
        }
        if let Some(caps) = self.host_caps {
            notes.push(match caps {
                Some(Capabilities { pm, msi, msix, pending }) => {
                    let pm = match pm {
                        Some(0) => "D0",
                        Some(1) => "D1",
                        Some(2) => "D2",
                        Some(3) => "D3hot",
                        _ => "PM capability absent",
                    };
                    let enabled = |value: Option<u8>| value.map_or("absent", |value| on_off(value != 0));
                    let pending = pending.map_or("capability absent", |value| if value == 0 { "clear" } else { "pending" });
                    format!("LPSS post-EBS PCI capabilities ({pm}, MSI {}, MSI-X {}, PCIe transactions {pending})", enabled(msi), enabled(msix))
                }
                None => "LPSS post-EBS PCI capabilities unavailable".into(),
            });
        }
        if let Some(host) = self.host_bar {
            notes.push(match host {
                HostBar::Window(range, state) => {
                    format!("LPSS post-EBS BAR0 {range} (original root window, {state})")
                }
                HostBar::Unavailable(reason) => format!("LPSS post-EBS BAR0 {reason}"),
            });
        }
        if let (
            Some(FirmwareBar::Descriptor(fw)),
            Some(HostBar::Window(host, "identity" | "translated")),
        ) = (self.firmware_bar, self.host_bar)
        {
            notes.push(if fw == host {
                "LPSS BAR0 range agrees with the original root window".into()
            } else if (fw.bus, fw.host) == (host.bus, host.host) {
                "LPSS BAR0 base agrees with the original root window, extent differs".into()
            } else {
                "LPSS BAR0 range differs from the original root window".into()
            });
        }
        if self.firmware.is_some() || self.host_pci.is_some() {
            notes.push("LPSS I²C endpoint stop evidence unavailable".into());
        }
        notes
    }
}

fn on_off(value: bool) -> &'static str {
    if value { "on" } else { "off" }
}

fn masked(value: bool) -> &'static str {
    if value { "masked" } else { "unmasked" }
}

impl std::fmt::Display for Range {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "bus {:#x}, host {:#x}, {} bytes",
            self.bus, self.host, self.bytes
        )
    }
}

fn field<'a>(words: &mut SplitAsciiWhitespace<'a>, name: &str) -> Option<&'a str> {
    words.next()?.strip_prefix(name)
}

fn boolean(words: &mut SplitAsciiWhitespace<'_>, name: &str) -> Option<bool> {
    match field(words, name)? {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn hex_number(word: &str) -> Option<u64> {
    let value = word.strip_prefix("0x")?;
    if value.is_empty() || value.len() > 16 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(value, 16).ok()
}

fn decimal(word: &str) -> Option<u64> {
    if !word.is_empty() && word.bytes().all(|byte| byte.is_ascii_digit()) {
        word.parse().ok()
    } else {
        None
    }
}

fn stop_evidence(words: &mut SplitAsciiWhitespace<'_>) -> Option<()> {
    (field(words, "stop-evidence=")? == "unavailable").then_some(())
}

fn firmware(words: &mut SplitAsciiWhitespace<'_>) -> Option<Firmware> {
    let command = hex(field(words, "command=")?.strip_prefix("0x")?, 4)?;
    for (name, mask) in [
        ("ioe=", regs::COMMAND_IO),
        ("mse=", regs::COMMAND_MEMORY),
        ("bme=", regs::COMMAND_MASTER),
        ("intx-disabled=", regs::COMMAND_INTX_DISABLE),
    ] {
        if boolean(words, name)? != (command & mask != 0) {
            return None;
        }
    }
    let bar = match field(words, "bar0=")? {
        "advertised" => "advertised",
        "not-advertised" => "not advertised",
        "unavailable" => "unavailable",
        _ => return None,
    };
    let protocol = |value| match value {
        "present" => Some("present"),
        "absent" => Some("absent"),
        "unavailable" => Some("unavailable"),
        _ => None,
    };
    let master = protocol(field(words, "pci-handle-i2c-master=")?)?;
    let host = protocol(field(words, "pci-handle-i2c-host=")?)?;
    stop_evidence(words)?;
    Some(Firmware {
        command,
        bar,
        master,
        host,
    })
}

fn range(first: &str, words: &mut SplitAsciiWhitespace<'_>) -> Option<Range> {
    let bus = hex_number(first.strip_prefix("bus=")?)?;
    let host = hex_number(field(words, "host=")?)?;
    let bytes = decimal(field(words, "bytes=")?)?;
    (bytes != 0).then_some(())?;
    bus.checked_add(bytes)?;
    host.checked_add(bytes)?;
    Some(Range { bus, host, bytes })
}

fn firmware_bar(words: &mut SplitAsciiWhitespace<'_>) -> Option<FirmwareBar> {
    let first = words.next()?;
    if let Some(state) = first.strip_prefix("state=") {
        let reason = match state {
            "raw-bar0-refused" => "configuration refused",
            "descriptor-unavailable" => "descriptor unavailable",
            "descriptor-mismatch" => "descriptor disagrees with PCI configuration",
            _ => return None,
        };
        Some(FirmwareBar::Unavailable(reason))
    } else {
        Some(FirmwareBar::Descriptor(range(first, words)?))
    }
}

fn host_bar(words: &mut SplitAsciiWhitespace<'_>) -> Option<HostBar> {
    let first = words.next()?;
    if let Some(state) = first.strip_prefix("state=") {
        let reason = match state {
            "raw-bar0-refused" => "configuration refused",
            "no-covering-root-window" => "no covering original root window",
            "ambiguous-root-window" => "ambiguous original root window",
            "conflicting-root-windows" => "conflicting original root windows",
            "invalid-root-translation" => "original root translation overflows",
            "original-root-unavailable" => "original PCI root unavailable",
            "config-unavailable" => "PCI configuration unavailable",
            _ => return None,
        };
        return Some(HostBar::Unavailable(reason));
    }
    let range = range(first, words)?;
    if range.bytes != BAR0_PROBE_BYTES {
        return None;
    }
    let state = match field(words, "state=")? {
        "identity"
            if range.bus == range.host
                && range.host != 0
                && range.host.is_multiple_of(PAGE_BYTES) =>
        {
            "identity"
        }
        "translated"
            if range.bus != range.host
                && range.host != 0
                && range.host.is_multiple_of(PAGE_BYTES) =>
        {
            "translated"
        }
        "host-memory-conflict" => "host memory conflict",
        "invalid-host-range" => "invalid host range",
        _ => return None,
    };
    Some(HostBar::Window(range, state))
}

fn host_pci(words: &mut SplitAsciiWhitespace<'_>) -> Option<HostPci> {
    let value = HostPci {
        mse: boolean(words, "mse=")?,
        bme: boolean(words, "bme=")?,
        intx_disabled: boolean(words, "intx-disabled=")?,
        intx_asserted: boolean(words, "intx-asserted=")?,
        contained: boolean(words, "contained=")?,
    };
    stop_evidence(words)?;
    Some(value)
}

fn host_caps(words: &mut SplitAsciiWhitespace<'_>) -> Option<Option<Capabilities>> {
    fn value(field: &str, max: u8) -> Option<Option<u8>> {
        if field == "None" {
            return Some(None);
        }
        let value = field.strip_prefix("Some(")?.strip_suffix(')')?;
        let value = u8::try_from(decimal(value)?).ok()?;
        (value <= max).then_some(Some(value))
    }

    let first = words.next()?;
    if first == "state=unavailable" {
        return Some(None);
    }
    Some(Some(Capabilities {
        pm: value(first.strip_prefix("pm-state=")?, 3)?,
        msi: value(field(words, "msi-enabled=")?, 1)?,
        msix: value(field(words, "msix-enabled=")?, 1)?,
        pending: value(field(words, "pcie-pending=")?, 1)?,
    }))
}

fn acpi<'a>(words: &mut SplitAsciiWhitespace<'a>) -> Option<(&'a str, Acpi)> {
    let hid = field(words, "hid=")?;
    if !matches!(hid.len(), 7 | 8)
        || !hid
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit())
    {
        return None;
    }
    let device = field(words, "device=")?;
    let controller = field(words, "controller=")?;
    if !valid_acpi_path(device) || !valid_acpi_path(controller) {
        return None;
    }
    let address = u16::try_from(hex_number(field(words, "addr=")?)?).ok()?;
    let bits = match field(words, "addr-bits=")? {
        "7" if address < (1 << 7) => 7,
        "10" if address < (1 << 10) => 10,
        _ => return None,
    };
    let speed = u32::try_from(decimal(field(words, "speed=")?)?).ok()?;
    let irq = match field(words, "irq=")? {
        "gpio" => "GPIO interrupt resource",
        "apic-polled" => "APIC polling",
        _ => return None,
    };
    let descriptor = u16::try_from(hex_number(field(words, "descriptor=")?)?).ok()?;
    Some((
        device,
        Acpi {
            hid: hid.into(),
            controller: controller.into(),
            address,
            bits,
            speed,
            irq,
            descriptor,
        },
    ))
}

pub(super) fn parse(line: &str) -> Option<(Address, u16, Record<'_>)> {
    let mut words = line.split_ascii_whitespace();
    let phase = words.next()?;
    if words.next()? != "LPSS-I2C" {
        return None;
    }
    let kind = words.next()?;
    let slot = if phase == "FW" && kind.starts_with("pci=") {
        kind.strip_prefix("pci=")?
    } else {
        field(&mut words, "pci=")?
    };
    if slot.len() != 12 {
        return None;
    }
    let slot = address(slot)?;
    let (vendor, device) = ids(field(&mut words, "id=")?)?;
    if (phase, kind) == ("HOST", "ACPI") {
        lpss_driver::Profile::select(lpss_driver::Identity {
            vendor,
            device,
            subsystem_vendor: 0,
            subsystem_device: 0,
        })
        .ok()?;
    } else if !lpss::matches(u32::from(device) << 16 | u32::from(vendor)) {
        return None;
    }
    let record = match (phase, kind) {
        ("FW", "BAR0") => Record::FirmwareBar(firmware_bar(&mut words)?),
        ("FW", kind) if kind.starts_with("pci=") => Record::Firmware(firmware(&mut words)?),
        ("HOST", "BAR0") => Record::HostBar(host_bar(&mut words)?),
        ("HOST", "PCI") => Record::HostPci(host_pci(&mut words)?),
        ("HOST", "CAPS") => Record::HostCaps(host_caps(&mut words)?),
        ("HOST", "ACPI") => {
            let (path, acpi) = acpi(&mut words)?;
            Record::Acpi(path, acpi)
        }
        _ => return None,
    };
    words.next().is_none().then_some((slot, device, record))
}
