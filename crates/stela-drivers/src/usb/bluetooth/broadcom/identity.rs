// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2015 Intel Corporation
//! Broadcom USB Bluetooth identities: USB and HCI versions, chip names and patch-file names.
use super::{Error, Result};
use crate::{firmware::Target, usb::descriptor::Interface, usb::xhci::Device};
use alloc::{format, string::String, vec::Vec};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UsbIdentity {
    pub vendor: u16,
    pub product: u16,
    pub revision: u16,
}
impl UsbIdentity {
    pub fn read(device: &Device) -> Self {
        let identity = super::super::identity::read(&device.descriptor);
        Self {
            vendor: identity.vendor,
            product: identity.product,
            revision: identity.revision,
        }
    }
    pub fn hardware_label(self) -> String {
        format!(
            "usb:{:04x}:{:04x}:{:04x}",
            self.vendor, self.product, self.revision
        )
    }
}

/// Linux btusb vendor-interface admission for the separate Broadcom PatchRAM
/// and Apple ROM setups. PatchRAM still checks controller identity.
pub fn is_vendor_interface(device: &Device, interface: &Interface) -> bool {
    is_patchram_interface(device, interface) || is_apple_interface(device, interface)
}

fn is_patchram_interface(device: &Device, interface: &Interface) -> bool {
    let usb = UsbIdentity::read(device);
    interface.number == 0
        && interface.alternate == 0
        && (interface.class, interface.subclass, interface.protocol) == (0xff, 1, 1)
        && listed(usb.vendor, usb.product, true)
}

pub(crate) const APPLE_VENDOR: u16 = 0x05ac;

/// Linux btusb's BTUSB_BCM_APPLE | BTUSB_IFNUM_2 entry. Apple uses
/// interface 2 when a composite controller exposes other functions first.
pub(crate) fn is_apple_interface(device: &Device, interface: &Interface) -> bool {
    UsbIdentity::read(device).vendor == APPLE_VENDOR
        && matches!(interface.number, 0 | 2)
        && interface.alternate == 0
        && (interface.class, interface.subclass, interface.protocol) == (0xff, 1, 1)
}

/// Whether a controller with these IDs is a btusb BCM_PATCHRAM entry when
/// its interface 0 is of vendor class FF/01/01 (`vendor_interface`) or the
/// standard E0/01/01.
pub(crate) fn listed(vendor: u16, product: u16, vendor_interface: bool) -> bool {
    (vendor, product) == (0x19ff, 0x0239)
        || vendor_interface
            && matches!(
                vendor,
                0x105b
                    | 0x0bb4
                    | 0x0489
                    | 0x04ca
                    | 0x0a5c
                    | 0x0b05
                    | 0x050d
                    | 0x13d3
                    | 0x413c
                    | 0x0930
            )
}

pub(super) fn is_candidate(device: &Device) -> bool {
    crate::usb::descriptor::interfaces(&device.configuration)
        .iter()
        .any(|i| {
            is_patchram_interface(device, i)
                || (i.number == 0
                    && i.alternate == 0
                    && (i.class, i.subclass, i.protocol) == (0xe0, 1, 1)
                    && {
                        let usb = UsbIdentity::read(device);
                        listed(usb.vendor, usb.product, false)
                    })
        })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalVersion {
    pub raw: [u8; 9],
}
impl LocalVersion {
    #[expect(
        clippy::result_large_err,
        reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
    )]
    pub(super) fn parse(bytes: &[u8]) -> Result<Self> {
        let raw = bytes
            .try_into()
            .map_err(|_| Error::protocol("Read Local Version length", bytes))?;
        if bytes[0] != 0 || u16::from_le_bytes([bytes[5], bytes[6]]) != 15 {
            return Err(Error::protocol("Broadcom manufacturer/status", bytes));
        }
        Ok(Self { raw })
    }
    pub fn subversion(&self) -> u16 {
        u16::from_le_bytes([self.raw[7], self.raw[8]])
    }
    pub fn revision(&self) -> u16 {
        u16::from_le_bytes([self.raw[2], self.raw[3]])
    }
    pub fn hardware_name(&self) -> Option<&'static str> {
        Some(match self.subversion() {
            0x2105 => "BCM20703A1",
            0x210b => "BCM43142A0",
            0x2112 => "BCM4314A0",
            0x2118 => "BCM20702A0",
            0x2126 => "BCM4335A0",
            0x220e => "BCM20702A1",
            0x230f => "BCM4356A2",
            0x4106 => "BCM4335B0",
            0x410e => "BCM20702B0",
            0x6109 => "BCM4335C0",
            0x610c => "BCM4354",
            0x6607 => "BCM4350C5",
            _ => return None,
        })
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    pub usb: UsbIdentity,
    pub version: LocalVersion,
    pub verbose: [u8; 7],
    pub features: [u8; 9],
    pub local_name: Vec<u8>,
    /// Fc5a response, may differ from descriptor identity; no substitution.
    pub product: [u8; 5],
}
impl Identity {
    pub fn revision_label(&self) -> String {
        let v = &self.version.raw;
        format!(
            "hci-{:02x}-{:04x}:lmp-{:02x}-{:04x}:manufacturer-000f:chip-{:02x}:product-{:04x}-{:04x}",
            v[1],
            self.version.revision(),
            v[4],
            self.version.subversion(),
            self.verbose[1],
            u16::from_le_bytes([self.product[1], self.product[2]]),
            u16::from_le_bytes([self.product[3], self.product[4]])
        )
    }
    pub fn target<'a>(&self, hardware: &'a str, revision: &'a str) -> Target<'a> {
        Target {
            driver: "broadcom-bluetooth-usb-patchram",
            hardware,
            revision,
            board: None,
        }
    }
    pub fn firmware_names(&self) -> Vec<String> {
        let suffix = format!(
            "-{:04x}-{:04x}.hcd",
            u16::from_le_bytes([self.product[1], self.product[2]]),
            u16::from_le_bytes([self.product[3], self.product[4]])
        );
        let mut names = Vec::new();
        if let Some(name) = self.version.hardware_name() {
            names.push(format!("brcm/{name}{suffix}"));
        }
        names.push(format!("brcm/BCM{suffix}"));
        names
    }
}
