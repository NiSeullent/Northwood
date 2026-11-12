// SPDX-License-Identifier: GPL-2.0-or-later
//! Resolve the firmware-selected Northwood configuration and host profile.
use crate::{handoff::Handoff, layout, output};
use alloc::vec::Vec;
use northwood_config::{Acpi, DriverMode, Drivers, NwConfig, handoff, plist, profiles};
use plist::Value;
use stela_drivers::firmware::{Catalog, smbios};
use stela_drivers::probe::registry::{DriverPolicy, ForcedDriver, Mode, PciId};
use stela_drivers::probe::{HidMatch, HidQuirk, HidSetting};

#[cfg(feature = "diagnostics")]
mod firmware_hid;
#[cfg(feature = "diagnostics")]
pub(crate) use firmware_hid::firmware_primary_hid;

pub(super) struct Policy<'a> {
    drivers: Option<&'a Drivers>,
    blocked: Vec<PciId>,
    forced: Vec<ForcedDriver<'a>>,
}

impl<'a> Policy<'a> {
    pub(super) fn new(drivers: Option<&'a Drivers>) -> Self {
        Self {
            drivers,
            blocked: drivers
                .into_iter()
                .flat_map(|config| &config.block)
                .filter_map(|rule| PciId::parse(&rule.matcher))
                .collect(),
            forced: drivers
                .into_iter()
                .flat_map(|config| &config.force)
                .filter_map(|rule| {
                    Some(ForcedDriver {
                        pci: PciId::parse(&rule.matcher)?,
                        module: &rule.module,
                    })
                })
                .collect(),
        }
    }

    pub(super) fn pci(&self) -> DriverPolicy<'_> {
        let Some(drivers) = self.drivers else {
            return DriverPolicy::default();
        };
        DriverPolicy {
            mode: match drivers.mode {
                DriverMode::Auto => Mode::Auto,
                DriverMode::Safe => Mode::Safe,
                DriverMode::Manual => Mode::Manual,
            },
            enable: &drivers.enable,
            disable: &drivers.disable,
            block: &self.blocked,
            force: &self.forced,
        }
    }
}

pub(super) fn hid_quirks(config: Option<&Acpi>) -> Vec<HidQuirk<'_>> {
    config
        .into_iter()
        .flat_map(|config| &config.quirks)
        .filter_map(|rule| {
            let setting = match rule.kind.as_str() {
                "I2cHidAddress" => HidSetting::Address,
                "I2cHidDescriptor" => HidSetting::DescriptorRegister,
                "I2cHidForcePolling" => HidSetting::PollIntervalUs,
                _ => return None,
            };
            let target = if rule.matcher.starts_with('\\') {
                HidMatch::Path(&rule.matcher)
            } else {
                HidMatch::Hid(&rule.matcher)
            };
            match rule.value.as_ref() {
                Some(Value::Integer(value)) => {
                    HidQuirk::new(target, setting, u64::try_from(*value).ok()?)
                }
                _ => None,
            }
        })
        .collect()
}

fn board_text(value: smbios::Observation<'_>) -> &str {
    let smbios::Observation::Value(bytes) = value else {
        return "";
    };
    stela_boot_core::host_identity::smbios_text(bytes)
        .and_then(|bytes| core::str::from_utf8(bytes).ok())
        .unwrap_or_default()
}

pub(crate) struct ProfileConfiguration {
    pub(crate) config: NwConfig,
    pub(crate) profile: Option<&'static str>,
    /// A candidate matched SMBIOS and PCI but could not be checked for its HID.
    #[cfg(feature = "diagnostics")]
    pub(crate) hid_unavailable: bool,
}

/// Merge the same built-in defaults for normal startup and Diagnostics.
/// An unobservable primary HID never admits an automatic match.
pub(crate) fn profile_configuration(
    wire: Option<&[u8]>,
    firmware: Option<&Catalog>,
    pci_ids: &[(u16, u16)],
    mut observe_primary_hid: impl FnMut(&str) -> Option<bool>,
) -> Result<ProfileConfiguration, northwood_config::Error> {
    let identity = firmware
        .filter(|_| !pci_ids.is_empty())
        .and_then(|catalog| smbios::load(catalog).ok())
        .map(|snapshot| snapshot.identity);
    let (manufacturer, product) = identity.map_or(("", ""), |identity| {
        (
            board_text(identity.system_vendor),
            board_text(identity.product_name),
        )
    });
    let hardware = profiles::Hardware {
        manufacturer,
        product,
        pci_ids,
        acpi_ids: &[],
    };
    #[cfg(feature = "diagnostics")]
    let mut hid_unavailable = false;
    let primary_hid = |hid: &str| {
        let observed = observe_primary_hid(hid);
        #[cfg(feature = "diagnostics")]
        {
            hid_unavailable |= observed.is_none();
        }
        observed.unwrap_or(false)
    };
    let (parsed, profile) = match wire {
        Some(bytes) => handoff::decode_with_observed_hid(bytes, hardware, primary_hid),
        None => profiles::apply_with_observed_hid(hardware, plist::Dictionary::new(), primary_hid),
    }?;
    Ok(ProfileConfiguration {
        config: parsed.config,
        profile,
        #[cfg(feature = "diagnostics")]
        hid_unavailable,
    })
}

pub(super) fn configuration(
    boot: &Handoff,
    firmware: Option<&Catalog>,
    pci_ids: &[(u16, u16)],
    acpi: Option<&'static super::aml::HostAcpi>,
) -> Option<NwConfig> {
    let region = boot.northwood_config;
    if region.bytes > handoff::MAX_HANDOFF_BYTES as u64 {
        output::arguments(format_args!("NWCONFIG engine handoff is too large\n"));
        return None;
    }
    let wire = (region.bytes != 0).then(|| {
        // SAFETY: the loader retained the validated configuration in its mapped
        // allocations. The engine owns the handoff and reads it before use.
        unsafe {
            core::slice::from_raw_parts(
                layout::physical(region.physical) as *const u8,
                region.bytes as usize,
            )
        }
    });
    match profile_configuration(wire, firmware, pci_ids, |hid| {
        acpi.map(|host| host.has_primary_hid(hid))
    }) {
        Ok(effective) if wire.is_none() && effective.profile.is_none() => None,
        Ok(ProfileConfiguration {
            config, profile, ..
        }) => {
            if let Some(profile) = profile {
                crate::host_inventory::text("NorthwoodHardwareProfile", profile);
                output::arguments(format_args!("NWCONFIG hardware_profile={profile}\n"));
            }
            let swap = &config.memory.swap;
            output::arguments(format_args!(
                "NWCONFIG engine handoff swap={} location={:?} bytes={}\n",
                swap.enabled, swap.location, swap.bytes
            ));
            Some(config)
        }
        Err(_) => {
            output::arguments(format_args!("NWCONFIG engine handoff could not be read\n"));
            None
        }
    }
}
