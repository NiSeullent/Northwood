// SPDX-License-Identifier: GPL-2.0-or-later
//! The two on-disk layouts understood by Northwood's setup and boot program.
use alloc::{format, string::String};

/// UEFI removable-media entry point, shared by both layouts.
pub const LOADER: &str = "EFI/BOOT/BOOTX64.EFI";
pub const OPTIONS: &str = "stela.cfg";
pub const CHOICES: &str = "choices.cfg";
pub const SELECTED_DISK: &str = "disk.cfg";
pub const SOURCE: &str = "source.bin";
pub const CONFIG: &str = "nwconfig.plist";
pub const WIFI_KEY: &str = "wifi.key";
pub const BOOT_LOG: &str = "boot.log";
pub const RECORD: &str = "setup.json";
pub const FIRMWARE: &str = "firmware";
pub const FIRMWARE_MANIFEST: &str = "firmware/manifest.json";
pub const DATA_BOOT: &str = "boot";
pub const DATA_INSTALL: &str = "install";
pub const DATA_STORE: &str = "store";
pub const DATA_LOGS: &str = "logs";
pub const DATA_SWAP: &str = "swap";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Layout {
    Northwood,
    Legacy,
}

impl Layout {
    pub const PREFERRED: [Self; 2] = [Self::Northwood, Self::Legacy];
    /// Advertised by loaders that read the Northwood ESP and data paths.
    pub const MOUNT_CAPABILITY: [u8; 18] = *b"NORTHWOOD-LAYOUT\0\x01";

    pub fn supports_northwood(image: &[u8]) -> bool {
        image
            .windows(Self::MOUNT_CAPABILITY.len())
            .any(|window| window == Self::MOUNT_CAPABILITY)
    }

    pub const fn esp_dir(self) -> &'static str {
        match self {
            Self::Northwood => "EFI/northwood",
            Self::Legacy => "EFI/stela",
        }
    }

    pub const fn data_dir(self) -> &'static str {
        match self {
            Self::Northwood => "northwood",
            Self::Legacy => "stela",
        }
    }

    pub const fn volume_label(self) -> &'static str {
        match self {
            Self::Northwood => "NORTHWOOD",
            Self::Legacy => "STELA",
        }
    }

    pub fn esp_path(self, name: &str) -> String {
        format!("{}/{name}", self.esp_dir())
    }

    pub fn uefi_path(self, name: &str) -> String {
        format!("\\{}", self.esp_path(name).replace('/', "\\"))
    }

    pub fn data_path(self, name: &str) -> String {
        format!("{}/{name}", self.data_dir())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn northwood_precedes_stela_and_both_use_the_standard_loader() {
        assert_eq!(LOADER, "EFI/BOOT/BOOTX64.EFI");
        assert_eq!(
            Layout::PREFERRED.map(|layout| layout.uefi_path(FIRMWARE_MANIFEST)),
            [
                "\\EFI\\northwood\\firmware\\manifest.json",
                "\\EFI\\stela\\firmware\\manifest.json",
            ]
        );
        assert_eq!(
            Layout::PREFERRED.map(|layout| layout.data_path(DATA_STORE)),
            ["northwood/store", "stela/store"],
        );
        assert!(
            Layout::PREFERRED
                .iter()
                .all(|layout| layout.volume_label().len() <= 11)
        );
    }
}
