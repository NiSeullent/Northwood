// SPDX-License-Identifier: GPL-2.0-or-later
//! Versioned transport for the validated boot configuration and selected entry.
use crate::{Error, MAX_CHOICE_NAME, Parsed, plist::MAX_BYTES, profiles};
use alloc::vec::Vec;

const VERSION: &[u8] = b"NWCF\x01";
pub const MAX_HANDOFF_BYTES: usize = VERSION.len() + 1 + MAX_CHOICE_NAME + MAX_BYTES;

pub fn encode(xml: &[u8], entry: Option<&str>) -> Option<Vec<u8>> {
    let name = entry.unwrap_or("").as_bytes();
    if xml.is_empty() || xml.len() > MAX_BYTES || name.len() > MAX_CHOICE_NAME {
        return None;
    }
    let mut wire = Vec::with_capacity(VERSION.len() + 1 + name.len() + xml.len());
    wire.extend_from_slice(VERSION);
    wire.push(name.len() as u8);
    wire.extend_from_slice(name);
    wire.extend_from_slice(xml);
    Some(wire)
}

pub fn decode(wire: &[u8]) -> Result<Parsed, Error> {
    decode_with_observed_hid(wire, profiles::Hardware::default(), |_| false)
        .map(|(parsed, _)| parsed)
}

/// Decode the selected entry and apply defaults from an explicitly selected
/// or observed profile. The hardware observations belong to this boot and
/// are never carried in the wire.
pub fn decode_with_observed_hid(
    wire: &[u8],
    host: profiles::Hardware<'_>,
    observe_primary_hid: impl FnMut(&str) -> bool,
) -> Result<(Parsed, Option<&'static str>), Error> {
    let malformed = || Error::field("nwconfig handoff", "invalid version, length or entry name");
    if wire.len() > MAX_HANDOFF_BYTES {
        return Err(malformed());
    }
    let (length, wire) = wire
        .strip_prefix(VERSION)
        .and_then(<[u8]>::split_first)
        .ok_or_else(malformed)?;
    let length = usize::from(*length);
    if length > MAX_CHOICE_NAME {
        return Err(malformed());
    }
    let (name, xml) = wire.split_at_checked(length).ok_or_else(malformed)?;
    let name = core::str::from_utf8(name).map_err(|_| malformed())?;
    let (mut parsed, profile) = profiles::parse_with_observed_hid(xml, host, observe_primary_hid)?;
    if !name.is_empty() {
        parsed.config = parsed
            .config
            .boot
            .entries
            .into_iter()
            .find(|entry| entry.name == name)
            .ok_or_else(malformed)?
            .config;
    }
    Ok((parsed, profile))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &[u8] = br#"<plist><dict>
        <key>Memory</key><dict><key>Swap</key><dict><key>SizeGiB</key><integer>32</integer></dict></dict>
        <key>Boot</key><dict><key>Entries</key><array>
            <dict><key>Name</key><string>Recovery</string><key>Overrides</key><dict>
                <key>Memory</key><dict><key>Swap</key><dict><key>SizeGiB</key><integer>8</integer></dict></dict>
            </dict></dict>
        </array></dict>
    </dict></plist>"#;

    #[test]
    fn restores_the_selected_entry_and_its_override() {
        let base = encode(CONFIG, None).unwrap();
        let recovery = encode(CONFIG, Some("Recovery")).unwrap();
        assert_eq!(decode(&base).unwrap().config.memory.swap.bytes, 32 << 30);
        assert_eq!(decode(&recovery).unwrap().config.memory.swap.bytes, 8 << 30);
    }

    #[test]
    fn observed_profile_survives_the_wire_and_preserves_entry_overrides() {
        let xml = br#"<plist><dict>
            <key>Boot</key><dict><key>Entries</key><array>
                <dict><key>Name</key><string>Recovery</string><key>Overrides</key><dict>
                    <key>Memory</key><dict><key>Swap</key><dict>
                        <key>Bytes</key><integer>8589934592</integer>
                    </dict></dict>
                </dict></dict>
            </array></dict>
        </dict></plist>"#;
        let wire = encode(xml, Some("Recovery")).unwrap();
        let host = profiles::Hardware {
            manufacturer: "LENOVO",
            product: "82V7",
            pci_ids: &[(0x8086, 0x3185)],
            ..Default::default()
        };
        let mut observations = 0;
        let (parsed, profile) = decode_with_observed_hid(&wire, host, |hid| {
            observations += 1;
            hid == "SYNA2BA6"
        })
        .unwrap();
        assert_eq!(observations, 1);
        assert_eq!(profile, Some("lenovo-ideapad-1-15igl7"));
        assert_eq!(parsed.config.acpi.quirks.len(), 2);
        assert_eq!(parsed.config.memory.swap.bytes, 8 << 30);
        assert!(decode(&wire).unwrap().config.acpi.quirks.is_empty());
    }

    #[test]
    fn rejects_corrupted_wire_and_unknown_entry() {
        let wire = encode(CONFIG, Some("Unknown")).unwrap();
        assert!(decode(&wire).is_err());
        assert!(decode(&wire[..VERSION.len()]).is_err());
        assert!(decode(&[0; MAX_HANDOFF_BYTES + 1]).is_err());
    }
}
