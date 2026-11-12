//! The Wi-Fi network `stela.efi` joins for a network install (`--wifi`):
//! its name and country go in `stela.cfg`, and its passphrase in the key
//! file beside it in the selected setup directory, in plaintext.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
use stela_loader::mount_interface::layout::{self, Layout};

/// The key file on the ESP, as `stela.cfg`'s `--wifi` reads it by default.
#[cfg(test)]
pub(super) const KEY_PATH: &str = "EFI/stela/wifi.key";

pub(super) fn key_path(layout: Layout) -> String {
    layout.esp_path(layout::WIFI_KEY)
}

/// The network a setup joins, as its record keeps it. The passphrase is
/// only in the key file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct Network {
    pub(super) ssid: String,
    /// The ISO 3166 alpha-2 country whose radio rules apply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) country: Option<String>,
}

/// A network name of 1 to 32 bytes.
pub(super) fn ssid(name: &str) -> Result<String, String> {
    if (1..=32).contains(&name.len()) {
        Ok(name.to_owned())
    } else {
        Err("a Wi-Fi network name is 1 to 32 bytes".into())
    }
}

/// A WPA2 passphrase of 8 to 63 printable ASCII characters, or a PSK of 64
/// hexadecimal digits.
pub(super) fn check_passphrase(passphrase: &str) -> Result<(), String> {
    let printable = passphrase.bytes().all(|byte| (0x20..0x7f).contains(&byte));
    let psk = passphrase.len() == 64 && passphrase.bytes().all(|byte| byte.is_ascii_hexdigit());
    if psk || printable && (8..=63).contains(&passphrase.len()) {
        Ok(())
    } else {
        Err("a Wi-Fi passphrase is 8 to 63 printable ASCII characters, or 64 hex digits".into())
    }
}

/// An ISO 3166 alpha-2 country code, in capitals.
pub(super) fn country(code: &str) -> Result<String, String> {
    if code.len() == 2 && code.bytes().all(|byte| byte.is_ascii_alphabetic()) {
        Ok(code.to_ascii_uppercase())
    } else {
        Err("a Wi-Fi country is a two-letter ISO 3166 code, such as US or DE".into())
    }
}

/// The passphrase in `file`, without its line ending.
pub(super) fn read_passphrase(file: &Path) -> Result<String> {
    let text = fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
    let passphrase = text.trim_end_matches(['\r', '\n']).to_owned();
    check_passphrase(&passphrase)
        .map_err(anyhow::Error::msg)
        .with_context(|| file.display().to_string())?;
    Ok(passphrase)
}

/// Ask for the passphrase without echoing it.
pub(super) fn ask_passphrase(ssid: &str) -> Result<String> {
    use std::io::IsTerminal;
    ensure!(
        std::io::stdin().is_terminal(),
        "--wifi needs the passphrase typed at a terminal, or --wifi-passphrase-file FILE"
    );
    Ok(cliclack::password(format!("Passphrase for {ssid}"))
        .mask('•')
        .validate(|text: &String| check_passphrase(text))
        .interact()?)
}

/// Write the key file onto the ESP at `esp`.
pub(super) fn write_key_in(esp: &Path, passphrase: &str, layout: Layout) -> Result<()> {
    let path = esp.join(key_path(layout));
    fs::create_dir_all(path.parent().unwrap())?;
    crate::efi::write_atomic(&path, passphrase.as_bytes(), false)
        .with_context(|| format!("writing {}", path.display()))
}

/// The plan's line for the network, after its two spaces.
pub(super) fn describe_in(network: &Network, esp: &Path, layout: Layout) -> String {
    format!(
        "{:<14} joins \"{}\"{}; its passphrase is in {} in plaintext",
        "Wi-Fi",
        network.ssid,
        network
            .country
            .as_ref()
            .map(|country| format!(" under {country}'s rules"))
            .unwrap_or_default(),
        esp.join(key_path(layout)).display()
    )
}

/// The networks this computer sees or knows, strongest first where the
/// host says; empty when it has no Wi-Fi or cannot list them.
pub(super) fn host_networks() -> Vec<String> {
    let mut networks = scan().unwrap_or_default();
    let mut seen = std::collections::BTreeSet::new();
    networks.retain(|name| ssid(name).is_ok() && seen.insert(name.clone()));
    networks
}

#[cfg(target_os = "linux")]
fn scan() -> Option<Vec<String>> {
    let output = std::process::Command::new("nmcli")
        .args(["-t", "-f", "SSID", "device", "wifi", "list"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| nmcli(&String::from_utf8_lossy(&output.stdout)))
}

#[cfg(windows)]
fn scan() -> Option<Vec<String>> {
    let output = std::process::Command::new("netsh")
        .args(["wlan", "show", "networks"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| netsh(&String::from_utf8_lossy(&output.stdout)))
}

#[cfg(target_os = "macos")]
fn scan() -> Option<Vec<String>> {
    // The Wi-Fi interface, then the networks it has joined before.
    let ports = std::process::Command::new("networksetup")
        .arg("-listallhardwareports")
        .output()
        .ok()?;
    let device = wifi_device(&String::from_utf8_lossy(&ports.stdout))?;
    let output = std::process::Command::new("networksetup")
        .args(["-listpreferredwirelessnetworks", &device])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| networksetup(&String::from_utf8_lossy(&output.stdout)))
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn scan() -> Option<Vec<String>> {
    None
}

/// `nmcli -t -f SSID device wifi list`: one name a line, `\:` for a colon,
/// hidden networks empty.
#[cfg(any(test, target_os = "linux"))]
fn nmcli(text: &str) -> Vec<String> {
    text.lines()
        .map(|line| line.replace("\\:", ":").replace("\\\\", "\\"))
        .filter(|ssid| !ssid.is_empty())
        .collect()
}

/// `netsh wlan show networks`: `SSID 1 : name` lines.
#[cfg(any(test, windows))]
fn netsh(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            let key = key.trim();
            (key.starts_with("SSID ") && !key.starts_with("BSSID")).then(|| value.trim().to_owned())
        })
        .filter(|ssid| !ssid.is_empty())
        .collect()
}

/// `networksetup -listallhardwareports`: the device after `Hardware Port:
/// Wi-Fi`.
#[cfg(any(test, target_os = "macos"))]
fn wifi_device(text: &str) -> Option<String> {
    let mut lines = text.lines();
    lines.find(|line| line.trim() == "Hardware Port: Wi-Fi")?;
    lines
        .next()?
        .trim()
        .strip_prefix("Device: ")
        .map(str::to_owned)
}

/// `networksetup -listpreferredwirelessnetworks`: a heading, then one
/// tab-indented name a line.
#[cfg(any(test, target_os = "macos"))]
fn networksetup(text: &str) -> Vec<String> {
    text.lines()
        .skip(1)
        .map(|line| line.trim_start_matches('\t').to_owned())
        .filter(|ssid| !ssid.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passphrases_are_8_to_63_printable_characters_or_a_64_digit_psk() {
        assert!(check_passphrase("correct horse").is_ok());
        assert!(check_passphrase(&"a".repeat(63)).is_ok());
        assert!(check_passphrase(&"0f".repeat(32)).is_ok());
        for refused in ["short", &"z".repeat(64), "tab\there!", "caf\u{e9} au lait"] {
            assert!(check_passphrase(refused).is_err(), "{refused:?}");
        }
    }

    #[test]
    fn countries_are_two_letters_in_capitals() {
        assert_eq!(country("de").unwrap(), "DE");
        for refused in ["USA", "1A", "", "D"] {
            assert!(country(refused).is_err(), "{refused:?}");
        }
    }

    #[test]
    fn each_hosts_listing_yields_its_network_names() {
        assert_eq!(nmcli("Home\n\nCafe\\: Guest\n"), ["Home", "Cafe: Guest"]);
        let netsh_text = "Interface name : Wi-Fi\nThere are 2 networks currently visible.\n\n\
            SSID 1 : Home\n    Network type            : Infrastructure\n\
            SSID 2 : Office Net\n    BSSID 1                 : 00:11:22:33:44:55\n";
        assert_eq!(netsh(netsh_text), ["Home", "Office Net"]);
        let ports = "Hardware Port: Ethernet\nDevice: en1\n\nHardware Port: Wi-Fi\nDevice: en0\n";
        assert_eq!(wifi_device(ports).as_deref(), Some("en0"));
        let preferred = "Preferred networks on en0:\n\tHome\n\tOffice Net\n";
        assert_eq!(networksetup(preferred), ["Home", "Office Net"]);
    }
}
