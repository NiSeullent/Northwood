// SPDX-License-Identifier: GPL-2.0-or-later
//! Built-in defaults selected by observed SMBIOS, ACPI and PCI identifiers.
use crate::{
    Error, Parsed, Swap, from_dictionary,
    plist::{self, Dictionary, Value},
    schema::merge,
};

struct Profile {
    id: &'static str,
    xml: &'static [u8],
}

const PROFILES: &[Profile] = &[Profile {
    id: "lenovo-ideapad-1-15igl7",
    xml: include_bytes!("../../../hardware/profiles/lenovo-ideapad-1-15igl7.plist"),
}];

#[derive(Debug, Clone, Copy, Default)]
pub struct Hardware<'a> {
    pub manufacturer: &'a str,
    pub product: &'a str,
    pub acpi_ids: &'a [&'a str],
    pub pci_ids: &'a [(u16, u16)],
}

pub fn parse(bytes: &[u8], host: Hardware<'_>) -> Result<(Parsed, Option<&'static str>), Error> {
    apply(host, plist::parse(bytes).map_err(Error::Xml)?)
}

/// Parse and apply a profile using a live query for the primary ACPI HID.
pub fn parse_with_observed_hid(
    bytes: &[u8],
    host: Hardware<'_>,
    observe_primary_hid: impl FnMut(&str) -> bool,
) -> Result<(Parsed, Option<&'static str>), Error> {
    apply_with_observed_hid(
        host,
        plist::parse(bytes).map_err(Error::Xml)?,
        observe_primary_hid,
    )
}

fn matching_hid<'a>(dictionary: &'a Dictionary, host: Hardware<'_>) -> Option<&'a str> {
    let text = |name| match dictionary.get(name) {
        Some(Value::String(text)) => Some(text.as_str()),
        _ => None,
    };
    let integer = |name| match dictionary.get(name) {
        Some(Value::Integer(number)) => u16::try_from(*number).ok(),
        _ => None,
    };
    let (Some(manufacturer), Some(product), Some(hid), Some(vendor), Some(device)) = (
        text("Manufacturer"),
        text("Product"),
        text("AcpiHid"),
        integer("PciVendor"),
        integer("PciDevice"),
    ) else {
        return None;
    };
    (host.manufacturer.trim().eq_ignore_ascii_case(manufacturer)
        && host.product.trim().eq_ignore_ascii_case(product)
        && host.pci_ids.contains(&(vendor, device)))
    .then_some(hid)
}

/// Merge matching defaults below the supplied settings. Returns the
/// selected profile's name for the boot log and diagnostics.
pub fn apply(
    host: Hardware<'_>,
    user: Dictionary,
) -> Result<(Parsed, Option<&'static str>), Error> {
    apply_with_observed_hid(host, user, |hid| {
        host.acpi_ids.iter().any(|id| id.eq_ignore_ascii_case(hid))
    })
}

/// Query the primary ACPI HID only for automatic candidates whose SMBIOS
/// and PCI identifiers match. The callback supplies the HID evidence in
/// place of `Hardware::acpi_ids`.
pub fn apply_with_observed_hid(
    host: Hardware<'_>,
    user: Dictionary,
    mut observe_primary_hid: impl FnMut(&str) -> bool,
) -> Result<(Parsed, Option<&'static str>), Error> {
    let parsed = from_dictionary(user.clone())?;
    let config = &parsed.config.northwood;
    let requested = config.hardware_profile.as_str();
    if requested.is_empty()
        && (!config.auto_profile || host.manufacturer.is_empty() || host.product.is_empty())
    {
        return Ok((parsed, None));
    }
    for profile in PROFILES {
        let mut defaults = plist::parse(profile.xml).map_err(Error::Xml)?;
        let Some(Value::Dictionary(matcher)) = defaults.remove("Match") else {
            return Err(Error::field(
                "Match",
                "built-in profile is missing its match dictionary",
            ));
        };
        let selected = if requested.is_empty() {
            matching_hid(&matcher, host).is_some_and(&mut observe_primary_hid)
        } else {
            requested == profile.id
        };
        if selected {
            let user_sets_bytes = matches!(user.get("Memory"), Some(Value::Dictionary(memory))
                if matches!(memory.get("Swap"), Some(Value::Dictionary(swap)) if swap.contains_key("Bytes")));
            if let Some(Value::Dictionary(memory)) = defaults.get_mut("Memory")
                && let Some(Value::Dictionary(swap)) = memory.get_mut("Swap")
                && (user_sets_bytes
                    || swap.get("SizeGiB")
                        == Some(&Value::Integer(i128::from(Swap::default().bytes >> 30))))
            {
                // Keep a redundant schema default from conflicting with an
                // entry's legacy Bytes override.
                swap.remove("SizeGiB");
            }
            merge(&mut defaults, user);
            return Ok((from_dictionary(defaults)?, Some(profile.id)));
        }
    }
    if requested.is_empty() {
        return Ok((parsed, None));
    }
    Err(Error::field(
        "Northwood/HardwareProfile",
        "unknown hardware profile",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lenovo() -> Hardware<'static> {
        Hardware {
            manufacturer: " Lenovo ",
            product: "82V7",
            acpi_ids: &["SYNA2BA6"],
            pci_ids: &[(0x8086, 0x3185)],
        }
    }

    #[test]
    fn requires_all_detected_identifiers_for_automatic_selection() {
        let (_, profile) = apply(lenovo(), Dictionary::new()).unwrap();
        assert_eq!(profile, Some("lenovo-ideapad-1-15igl7"));
        for host in [
            Hardware {
                manufacturer: "Other",
                ..lenovo()
            },
            Hardware {
                product: "82V8",
                ..lenovo()
            },
            Hardware {
                acpi_ids: &[],
                ..lenovo()
            },
            Hardware {
                pci_ids: &[],
                ..lenovo()
            },
        ] {
            assert_eq!(apply(host, Dictionary::new()).unwrap().1, None);
        }
    }

    #[test]
    fn observes_primary_hid_only_after_smbios_and_pci_match() {
        for host in [
            Hardware {
                manufacturer: "Other",
                ..lenovo()
            },
            Hardware {
                product: "82V8",
                ..lenovo()
            },
            Hardware {
                pci_ids: &[],
                ..lenovo()
            },
        ] {
            let (_, profile) = apply_with_observed_hid(host, Dictionary::new(), |_| {
                panic!("hardware prefilter must run before the HID query")
            })
            .unwrap();
            assert_eq!(profile, None);
        }

        for present in [false, true] {
            let mut observed = 0;
            let (_, profile) = apply_with_observed_hid(lenovo(), Dictionary::new(), |hid| {
                assert_eq!(hid, "SYNA2BA6");
                observed += 1;
                present
            })
            .unwrap();
            assert_eq!(observed, 1);
            assert_eq!(profile, present.then_some("lenovo-ideapad-1-15igl7"));
        }
    }

    #[test]
    fn explicit_and_disabled_profiles_do_not_query_hid() {
        let no_query = |_: &str| panic!("only automatic matching may query ACPI");
        let disabled = br#"<plist><dict><key>Northwood</key><dict>
            <key>AutoProfile</key><false/>
        </dict></dict></plist>"#;
        assert_eq!(
            parse_with_observed_hid(disabled, lenovo(), no_query)
                .unwrap()
                .1,
            None,
        );
        let forced = br#"<plist><dict><key>Northwood</key><dict>
            <key>AutoProfile</key><false/>
            <key>HardwareProfile</key><string>lenovo-ideapad-1-15igl7</string>
        </dict></dict></plist>"#;
        assert_eq!(
            parse_with_observed_hid(forced, Hardware::default(), no_query)
                .unwrap()
                .1,
            Some("lenovo-ideapad-1-15igl7"),
        );
    }

    #[test]
    fn user_values_override_profile_and_auto_can_be_disabled() {
        let user = plist::parse(
            br#"<plist><dict>
            <key>Memory</key><dict><key>Swap</key><dict>
            <key>SizeGiB</key><integer>16</integer></dict></dict>
        </dict></plist>"#,
        )
        .unwrap();
        let (merged, _) = apply(lenovo(), user).unwrap();
        assert_eq!(merged.config.memory.swap.bytes, 16 << 30);
        assert_eq!(merged.config.acpi.quirks.len(), 2);
        let user = plist::parse(
            br#"<plist><dict><key>Northwood</key><dict>
            <key>AutoProfile</key><false/></dict></dict></plist>"#,
        )
        .unwrap();
        assert_eq!(apply(lenovo(), user).unwrap().1, None);
    }

    #[test]
    fn explicit_selection_and_legacy_size_alias_preserve_user_values() {
        let input = br#"<plist><dict>
            <key>Northwood</key><dict>
                <key>HardwareProfile</key><string>lenovo-ideapad-1-15igl7</string>
            </dict>
            <key>Memory</key><dict><key>Swap</key><dict>
                <key>Bytes</key><integer>8589934592</integer>
            </dict></dict>
        </dict></plist>"#;
        let (parsed, name) = parse(input, Hardware::default()).unwrap();
        assert_eq!(name, Some("lenovo-ideapad-1-15igl7"));
        assert_eq!(parsed.config.memory.swap.bytes, 8 << 30);
    }
}
