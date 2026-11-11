// SPDX-License-Identifier: GPL-2.0-or-later
//! Bridge the plist to the established Stela option-line parser.

use crate::{DEFAULT_WIFI_KEY, Error, NwConfig};
use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use core::fmt::Write;

/// Render only keys with an existing system option. New consumers read the
/// typed configuration. The result still goes through Stela's System parser.
pub fn options_line(config: &NwConfig, diagnostics: bool) -> Result<String, Error> {
    if config.misc.debug.host_check {
        if !diagnostics {
            return Err(Error::field(
                "Misc/Debug/HostCheck",
                "requires a diagnostics build",
            ));
        }
        let mut line = Line(String::from("--host-check"));
        line.flag("--no-volume-log", !config.misc.boot_log.volume);
        return Ok(line.0);
    }
    let mut line = Line(String::from("--system"));
    let boot = &config.boot;
    line.word("--bundle", "Boot/Bundle", &boot.bundle)?;
    line.word("--preboot", "Boot/Preboot", &boot.preboot)?;
    line.word("--install", "Boot/Install", &boot.install)?;
    line.word(
        "--install-source",
        "Boot/InstallSource",
        &boot.install_source,
    )?;
    line.flag("--verify-on-read", boot.verify_on_read);
    line.flag("--update-restore", boot.update_restore);
    line.flag("--single-user", boot.single_user);
    line.flag("--maintenance-write", boot.maintenance_write);
    line.flag("--no-volume-log", !config.misc.boot_log.volume);

    let drivers = &config.drivers;
    line.flag("--input", drivers.input);
    line.flag("--guest-network", drivers.guest_network);
    line.flag("--external-disks", drivers.external_disks);
    for serial in &drivers.usb_storage {
        line.word("--usb-storage", "Drivers/UsbStorage", serial)?;
    }
    for uuid in &drivers.usb4_approve {
        line.word("--usb4-approve", "Drivers/Usb4Approve", uuid)?;
    }

    let network = &drivers.network;
    let wifi = &network.wifi;
    if !wifi.ssid.is_empty() {
        let mut encoded = String::with_capacity(wifi.ssid.len());
        for ch in wifi.ssid.chars() {
            if ch == '%' || ch.is_control() || ch.is_whitespace() {
                let mut bytes = [0; 4];
                for byte in ch.encode_utf8(&mut bytes).bytes() {
                    let _ = write!(encoded, "%{byte:02X}");
                }
            } else {
                encoded.push(ch);
            }
        }
        // An SSID may legitimately start with a dash; all bytes are treated
        // as an operand by the legacy parser.
        line.operand("--wifi", &encoded);
        let key = if wifi.key_file.is_empty() {
            DEFAULT_WIFI_KEY
        } else {
            &wifi.key_file
        };
        line.word("--wifi-key", "Drivers/Network/WiFi/KeyFile", key)?;
        line.word(
            "--wifi-country",
            "Drivers/Network/WiFi/Country",
            &wifi.country,
        )?;
    } else if !wifi.country.is_empty()
        || (!wifi.key_file.is_empty() && wifi.key_file != DEFAULT_WIFI_KEY)
    {
        return Err(Error::field(
            "Drivers/Network/WiFi",
            "KeyFile and Country need an SSID",
        ));
    }
    line.word("--ipv4", "Drivers/Network/IPv4", &network.ipv4)?;
    line.word(
        "--ipv4-gateway",
        "Drivers/Network/Gateway",
        &network.gateway,
    )?;
    if !network.dns.is_empty() {
        let addresses = network
            .dns
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        line.operand("--dns", &addresses.join(","));
    }

    line.flag("--scanout-mirror", config.graphics.scanout_mirror);
    line.number("--storage-ram", config.memory.storage_ram_bytes);
    let guest = &config.guest;
    line.word("--storage", "Guest/Storage", &guest.storage)?;
    line.word("--nor", "Guest/Nor", &guest.nor)?;
    line.number("--native-cores", guest.cores);
    line.number("--guest-pes", guest.pes);
    line.flag("--serialized", !guest.parallel && !guest.native_rounds);
    line.flag("--idle-skip", guest.idle_skip);
    line.flag("--native-rounds", guest.native_rounds);

    let debug = &config.misc.debug;
    if !diagnostics && *debug != crate::DebugOptions::default() {
        return Err(Error::field(
            "Misc/Debug",
            "diagnostic options require a diagnostics build",
        ));
    }
    for (flag, enabled) in [
        ("--profile", debug.profile),
        ("--survey", debug.survey),
        ("--sep-observe", debug.sep_observe),
        ("--monitor-trace", debug.monitor_trace),
        ("--display-trace", debug.display_trace),
        ("--audio-own-hpet", debug.audio_own_hpet),
        ("--native-pci-own-ioapic", debug.own_ioapic),
        ("--native-bootstrap-check", debug.native_bootstrap_check),
        ("--native-shared-check", debug.native_shared_check),
        ("--native-artifacts", debug.native_artifacts),
        ("--development-restore", debug.development_restore),
    ] {
        line.flag(flag, enabled);
    }
    for (flag, path, values) in [
        ("--break", "Misc/Debug/Break", &debug.breaks),
        ("--trace", "Misc/Debug/Trace", &debug.trace),
        ("--watch", "Misc/Debug/Watch", &debug.watch),
        ("--dump", "Misc/Debug/Dump", &debug.dump),
    ] {
        for value in values {
            line.word(flag, path, value)?;
        }
    }
    line.word(
        "--network-check",
        "Misc/Debug/NetworkCheck",
        &debug.network_check,
    )?;
    line.word("--audio-check", "Misc/Debug/AudioCheck", &debug.audio_check)?;
    line.raw(&debug.raw)?;
    Ok(line.0)
}

struct Line(String);

impl Line {
    fn flag(&mut self, flag: &str, enabled: bool) {
        if enabled {
            self.0.push(' ');
            self.0.push_str(flag);
        }
    }

    fn operand(&mut self, flag: &str, value: &str) {
        self.flag(flag, true);
        self.0.push(' ');
        self.0.push_str(value);
    }

    fn word(&mut self, flag: &str, path: &str, value: &str) -> Result<(), Error> {
        if value.is_empty() {
            return Ok(());
        }
        if value.starts_with("--")
            || value
                .chars()
                .any(|ch| ch.is_whitespace() || ch.is_control())
        {
            return Err(Error::field(
                path,
                "expected one option value without whitespace",
            ));
        }
        self.operand(flag, value);
        Ok(())
    }

    fn number(&mut self, flag: &str, value: u64) {
        if value != 0 {
            let _ = write!(self.0, " {flag} {value}");
        }
    }

    fn raw(&mut self, raw: &str) -> Result<(), Error> {
        const PATH: &str = "Misc/Debug/Raw";
        let mut words = raw.split_ascii_whitespace();
        while let Some(flag) = words.next() {
            match flag {
                "--restore-study" => self.flag(flag, true),
                "--dram-base" | "--load" | "--entry" | "--x0" | "--budget" | "--restore-source"
                | "--restore-manifest" | "--restore-identity" => {
                    let value = words
                        .next()
                        .ok_or_else(|| Error::field(PATH, "missing diagnostic value"))?;
                    self.word(flag, PATH, value)?;
                }
                _ => {
                    return Err(Error::field(
                        PATH,
                        "unknown or reserved system diagnostic option",
                    ));
                }
            }
        }
        Ok(())
    }
}
