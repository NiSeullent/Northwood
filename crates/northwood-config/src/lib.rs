// SPDX-License-Identifier: GPL-2.0-or-later
//! Shared configuration for the Northwood boot program and setup tools.
#![no_std]

extern crate alloc;

pub mod handoff;
pub mod plist;
pub mod profiles;
pub mod render;
mod schema;

pub use schema::*;

pub const CONFIG_FILE: &str = "\\EFI\\northwood\\nwconfig.plist";
pub const MOUNT_CAPABILITY: [u8; 20] = *b"NORTHWOOD-NWCONFIG\0\x01";
pub const DEFAULT_WIFI_KEY: &str = "\\EFI\\northwood\\wifi.key";
pub const DEFAULT_FIRMWARE_PATH: &str = "EFI/northwood/firmware";

#[cfg(test)]
mod tests;
