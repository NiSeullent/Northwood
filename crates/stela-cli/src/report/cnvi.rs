// SPDX-License-Identifier: GPL-2.0-or-later
//! Read-only 31dc observations from one boot. A route record is not a claim.

use super::{Address, address};
use std::str::SplitAsciiWhitespace;
use stela_drivers::net::wifi::intel::firmware::legacy::identity::Snapshot;

#[derive(Default)]
pub(super) struct Observations {
    pci: Option<Pci>,
    csr: Option<Option<Snapshot>>,
    dmar: Option<Dmar>,
    root: Option<Root>,
    conflict: bool,
}

pub(super) enum Record {
    Pci(Pci),
    Csr(Option<Snapshot>),
    Dmar(Dmar),
    Root(Root),
}

#[derive(PartialEq, Eq)]
pub(super) struct Pci {
    power: &'static str,
    memory: bool,
    bus_master: bool,
    bar: Option<(u64, u64, u64)>,
}

#[derive(PartialEq, Eq)]
pub(super) struct Dmar {
    direct: Vec<u64>,
    include_all: Vec<u64>,
    reserved: usize,
}

#[derive(PartialEq, Eq)]
pub(super) struct Root {
    rid: u16,
    role: &'static str,
    route: Route,
}

#[derive(PartialEq, Eq)]
enum Route {
    Unresolved(&'static str),
    NotTaken(u64),
    Taken {
        unit: u64,
        initial: &'static str,
        allows: bool,
        blocked_fabric: bool,
        initial_lines: bool,
    },
}

impl Observations {
    pub(super) fn observed(&self) -> bool {
        self.pci.is_some() || self.csr.is_some() || self.dmar.is_some() || self.root.is_some()
    }

    pub(super) fn observe(&mut self, record: Record) {
        fn changed<T: PartialEq>(old: &mut Option<T>, value: T) -> bool {
            let conflict = old.as_ref().is_some_and(|previous| previous != &value);
            *old = Some(value);
            conflict
        }
        self.conflict |= match record {
            Record::Pci(value) => changed(&mut self.pci, value),
            Record::Csr(value) => changed(&mut self.csr, value),
            Record::Dmar(value) => changed(&mut self.dmar, value),
            Record::Root(value) => changed(&mut self.root, value),
        };
    }

    pub(super) fn describe(&self) -> Vec<String> {
        if self.conflict
            || self.csr.is_some()
                && self
                    .pci
                    .as_ref()
                    .is_some_and(|pci| pci.power != "D0" || !pci.memory || pci.bar.is_none())
        {
            return vec!["Conflicting Intel CNVi observations. Correlation withheld".into()];
        }
        let mut notes = Vec::new();
        if let Some(pci) = &self.pci {
            let bar = pci.bar.map_or_else(
                || "BAR0 unverified".into(),
                |(bus, host, size)| format!("BAR0 bus {bus:#x}, host {host:#x}, {size} bytes"),
            );
            notes.push(format!(
                "CNVi UEFI PCI {} (memory decode {}, bus master {}, {bar})",
                pci.power,
                on_off(pci.memory),
                on_off(pci.bus_master)
            ));
        }
        if let Some(csr) = self.csr {
            notes.push(csr.map_or_else(
                || "CNVi UEFI CSR read failed".into(),
                |s| format!("CNVi UEFI CSR (HW_REV {:#010x}, RF_ID {:#010x}, GP_CNTRL {:#010x}, identity {:?}, clock/access {:?})",
                            s.hw_revision, s.rf_id, s.gp_control, s.qualify_31dc(), s.qualify_awake()),
            ));
        }
        if let Some(dmar) = &self.dmar {
            notes.push(format!("CNVi firmware DMAR listing (direct units {:x?}, include-all units {:x?}, direct reserved ranges {})",
                               dmar.direct, dmar.include_all, dmar.reserved));
        }
        if let Some(root) = &self.root {
            let route = match root.route {
                Route::Unresolved(reason) => format!("route {reason}"),
                Route::NotTaken(unit) => format!("unit {unit:#x}, takeover not taken"),
                Route::Taken {
                    unit,
                    initial,
                    allows,
                    blocked_fabric,
                    initial_lines,
                } => format!(
                    "unit {unit:#x}, initial {initial}, allows {allows}, blocked fabric {blocked_fabric}, initial lines {initial_lines}"
                ),
            };
            notes.push(format!(
                "CNVi post-EBS DMA RID {:#06x} ({}, {route}, claim not attempted)",
                root.rid, root.role
            ));
        }
        notes
    }
}

fn on_off(value: bool) -> &'static str {
    if value { "on" } else { "off" }
}

fn field<'a>(fields: &mut SplitAsciiWhitespace<'a>, key: &str) -> Option<&'a str> {
    fields.next()?.strip_prefix(key)
}

fn flag(fields: &mut SplitAsciiWhitespace<'_>, key: &str) -> Option<bool> {
    match field(fields, key)? {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn hexadecimal(value: &str) -> Option<u64> {
    u64::from_str_radix(value.strip_prefix("0x")?, 16).ok()
}

fn pci(mut fields: SplitAsciiWhitespace<'_>) -> Option<Record> {
    let power = match field(&mut fields, "power=")? {
        "Ok(0)" => "D0",
        "Ok(1)" => "D1",
        "Ok(2)" => "D2",
        "Ok(3)" => "D3hot",
        "Err(Missing)" => "PM capability missing",
        "Err(Invalid)" => "PM capability invalid",
        "Err(ReadFailed)" => "PM capability read failed",
        _ => return None,
    };
    let memory = flag(&mut fields, "memory_decode=")?;
    let bus_master = flag(&mut fields, "bus_master=")?;
    let bar = match fields.next()? {
        "bar0=unverified" => None,
        first => {
            let bus = hexadecimal(first.strip_prefix("bar0_bus=")?)?;
            let host = hexadecimal(field(&mut fields, "bar0_host=")?)?;
            let bytes = field(&mut fields, "bar0_bytes=")?.parse().ok()?;
            if bytes < 16 * 1024 || fields.next()? != "source=uefi-pci-io" {
                return None;
            }
            Some((bus, host, bytes))
        }
    };
    fields.next().is_none().then_some(Record::Pci(Pci {
        power,
        memory,
        bus_master,
        bar,
    }))
}

fn csr(mut fields: SplitAsciiWhitespace<'_>) -> Option<Record> {
    let first = fields.next()?;
    if first == "read=failed" {
        return fields.next().map(|_| Record::Csr(None));
    }
    let hw_revision = u32::try_from(hexadecimal(first.strip_prefix("hw_rev=")?)?).ok()?;
    let mut word = |key| u32::try_from(hexadecimal(field(&mut fields, key)?)?).ok();
    let snapshot = Snapshot {
        hw_revision,
        rf_id: word("rf_id=")?,
        gp_control: word("gp=")?,
    };
    (fields.next()? == format!("identity={:?}", snapshot.qualify_31dc())
        && fields.next()? == format!("awake={:?}", snapshot.qualify_awake())
        && fields.next().is_none())
    .then_some(Record::Csr(Some(snapshot)))
}

fn dmar(fields: &str) -> Option<Record> {
    fn units(text: &str) -> Option<Vec<u64>> {
        let list = text.strip_prefix('[')?.strip_suffix(']')?;
        if list.is_empty() {
            return Some(Vec::new());
        }
        list.split(", ")
            .map(|unit| u64::from_str_radix(unit, 16).ok())
            .collect()
    }
    let (direct, rest) = fields
        .strip_prefix("direct_units=")?
        .split_once(" include_all_units=")?;
    let (include_all, reserved) = rest.split_once(" direct_reserved=")?;
    let reserved = reserved
        .strip_suffix(" scope=firmware-listing")?
        .parse()
        .ok()?;
    Some(Record::Dmar(Dmar {
        direct: units(direct)?,
        include_all: units(include_all)?,
        reserved,
    }))
}

fn root(slot: Address, mut fields: SplitAsciiWhitespace<'_>) -> Option<Record> {
    let rid = u16::try_from(hexadecimal(field(&mut fields, "rid=")?)?).ok()?;
    let expected =
        u16::from(slot.bus) << 8 | u16::from(slot.device) << 3 | u16::from(slot.function);
    if rid != expected {
        return None;
    }
    let role = match field(&mut fields, "role=")? {
        "Physical" => "physical",
        "Virtual" => "virtual",
        "Unverified" => "unverified",
        _ => return None,
    };
    let next = fields.next()?;
    let route = if let Some(unit) = next.strip_prefix("unit=") {
        let unit = hexadecimal(unit)?;
        match fields.next()? {
            "takeover=not-taken" => Route::NotTaken(unit),
            "takeover=taken" => Route::Taken {
                unit,
                initial: match field(&mut fields, "initial=")? {
                    "window" => "window",
                    "blocked" => "blocked",
                    _ => return None,
                },
                allows: flag(&mut fields, "allows=")?,
                blocked_fabric: flag(&mut fields, "blocked_fabric=")?,
                initial_lines: flag(&mut fields, "initial_lines=")?,
            },
            _ => return None,
        }
    } else {
        Route::Unresolved(match next {
            "route=no-unit" => "no unit",
            "route=unresolved" => "unresolved",
            "route=not-observed" => "not observed",
            _ => return None,
        })
    };
    (fields.next()? == "claim=not-attempted" && fields.next().is_none())
        .then_some(Record::Root(Root { rid, role, route }))
}

pub(super) fn parse(line: &str) -> Option<(Address, Record)> {
    let (kind, rest) = line.strip_prefix("HOST WIFI CNVI ")?.split_once(' ')?;
    let (slot, fields) = rest.strip_prefix("pci=")?.split_once(' ')?;
    let slot = address(slot)?;
    let record = match kind {
        "PCI" => pci(fields.split_ascii_whitespace()),
        "CSR" => csr(fields.split_ascii_whitespace()),
        "DMAR" => dmar(fields),
        "ROOT" => root(slot, fields.split_ascii_whitespace()),
        _ => None,
    }?;
    Some((slot, record))
}
