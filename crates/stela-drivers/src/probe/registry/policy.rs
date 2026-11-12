// SPDX-License-Identifier: GPL-2.0-or-later
//! Evaluate driver policy on copied PCI identities.
use super::{Class, Driver, Early, Entry, Identity, Manifest, REGISTRY, Route, Status, handoff};
use crate::gpu;
use alloc::string::String;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Auto,
    Safe,
    Manual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PciId {
    pub vendor: u16,
    pub device: u16,
}

impl PciId {
    /// Parse the exact `pci:vvvv:dddd` selector used in nwconfig.
    pub fn parse(value: &str) -> Option<Self> {
        let (vendor, device) = value.strip_prefix("pci:")?.split_once(':')?;
        let hex = |part: &str| {
            (part.len() == 4 && part.bytes().all(|byte| byte.is_ascii_hexdigit()))
                .then(|| u16::from_str_radix(part, 16).ok())
                .flatten()
        };
        Some(Self {
            vendor: hex(vendor)?,
            device: hex(device)?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForcedDriver<'a> {
    pub pci: PciId,
    pub module: &'a str,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DriverPolicy<'a> {
    pub mode: Mode,
    pub enable: &'a [String],
    pub disable: &'a [String],
    pub block: &'a [PciId],
    pub force: &'a [ForcedDriver<'a>],
}

impl DriverPolicy<'_> {
    fn allows(&self, manifest: &Manifest) -> bool {
        if self.disable.iter().any(|id| id == manifest.id) {
            return false;
        }
        match self.mode {
            Mode::Auto => true,
            Mode::Safe => matches!(
                manifest.class,
                Class::Storage | Class::Input | Class::UsbHost
            ),
            Mode::Manual => self.enable.iter().any(|id| id == manifest.id),
        }
    }

    /// Whether the post-scan NVIDIA path may try to claim this adapter.
    /// The live GPU and firmware admission checks still apply.
    pub fn allows_nvmodern(&self, adapter: &gpu::Adapter) -> bool {
        self.admit_nvmodern(adapter).is_ok()
    }

    /// Resolve post-scan NVIDIA admission while retaining a policy refusal.
    pub fn admit_nvmodern(&self, adapter: &gpu::Adapter) -> Result<(), Resolution> {
        if adapter.vendor != gpu::Vendor::Nvidia {
            return Err(Resolution::Unknown);
        }
        let identity = Identity {
            address: adapter.address,
            vendor: 0x10de,
            device: adapter.device,
            class: adapter.class,
        };
        match select_pci(&identity, self) {
            Selection::Entry(entry) if matches!(entry.route, Route::AfterScan) => Ok(()),
            Selection::Refused(reason) => Err(reason),
            _ => Err(Resolution::Unknown),
        }
    }
}

/// Policy result for PCI inventory. A candidate still needs normal ownership,
/// device and firmware validation before it can attach.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolution {
    Candidate(&'static Manifest),
    Disabled(&'static Manifest),
    Blocked,
    Unsupported(&'static str),
    ForceRejected,
    Unknown,
}

/// Inspect which registration could serve this PCI identity under `policy`.
/// Forcing may select another matching driver. It still requires a positive
/// match from that driver's own predicate.
pub fn resolve_pci(identity: &Identity, policy: &DriverPolicy<'_>) -> Resolution {
    match select_pci(identity, policy) {
        Selection::Entry(entry) => Resolution::Candidate(&entry.manifest),
        Selection::Refused(reason) => reason,
        Selection::Unknown => Resolution::Unknown,
    }
}

#[derive(Clone, Copy)]
pub(in crate::probe) enum Selection {
    Entry(&'static Entry),
    Refused(Resolution),
    Unknown,
}

impl Selection {
    pub(in crate::probe) fn driver(self) -> Option<Driver> {
        match self {
            Self::Entry(entry) => match entry.route {
                Route::Driver(driver) => Some(driver),
                _ => None,
            },
            _ => None,
        }
    }

    pub(in crate::probe) fn refusal(self) -> Option<&'static str> {
        match self {
            Self::Refused(Resolution::Disabled(_)) => Some("driver module disabled by nwconfig"),
            Self::Refused(Resolution::Blocked) => Some("PCI attachment blocked by nwconfig"),
            Self::Refused(Resolution::Unsupported(why)) => Some(why),
            Self::Refused(Resolution::ForceRejected) => {
                Some("forced driver is disabled or does not admit this PCI function")
            }
            _ => None,
        }
    }

    pub(in crate::probe) fn attaches_ehci(self) -> bool {
        matches!(
            self,
            Self::Entry(entry) if matches!(entry.route, Route::Early(Early::Usb(handoff::Kind::Ehci)))
        )
    }
}

pub(in crate::probe) fn select_pci(identity: &Identity, policy: &DriverPolicy<'_>) -> Selection {
    let pci = PciId {
        vendor: identity.vendor,
        device: identity.device,
    };
    if policy.block.contains(&pci) {
        return Selection::Refused(Resolution::Blocked);
    }
    if identity.class[..2] == [2, 0x80]
        && let Some(why) = crate::net::wifi::unsupported::reason(pci.vendor, pci.device)
    {
        return Selection::Refused(Resolution::Unsupported(why));
    }
    let candidates = || REGISTRY.iter().filter(|entry| (entry.matches)(identity));
    let early = candidates().find(|entry| matches!(entry.route, Route::Early(_)));
    let exclusive = early.is_some_and(|entry| match entry.route {
        Route::Early(Early::Display | Early::Usb(handoff::Kind::Ehci)) => true,
        #[cfg(feature = "native-acpi")]
        Route::Early(Early::Smbus) => true,
        _ => false,
    });
    let primary = if exclusive {
        candidates()
            .find(|entry| matches!(entry.route, Route::AfterScan))
            .or(early)
    } else {
        candidates()
            .find(|entry| matches!(entry.route, Route::Driver(_)))
            .or(early)
    };
    if let Some(force) = policy.force.iter().find(|force| force.pci == pci) {
        let valid = |entry: &&Entry| {
            entry.manifest.id == force.module
                && policy.allows(&entry.manifest)
                && entry.manifest.status == Status::Implemented
        };
        let chosen = if exclusive {
            primary.filter(valid)
        } else {
            candidates().find(|entry| matches!(entry.route, Route::Driver(_)) && valid(entry))
        };
        return chosen.map_or(
            Selection::Refused(Resolution::ForceRejected),
            Selection::Entry,
        );
    }
    match primary {
        Some(entry) if entry.manifest.status == Status::CheckOnly => Selection::Entry(entry),
        Some(entry) if policy.allows(&entry.manifest) => Selection::Entry(entry),
        Some(entry) => Selection::Refused(Resolution::Disabled(&entry.manifest)),
        None => Selection::Unknown,
    }
}
