// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2015 Intel Corporation
//! Broadcom/Cypress USB PatchRAM initialization adapted from pinned Linux btbcm.
//! Controller identification, exact Provider resources and acknowledged
//! HCD commands are required. Success is historical protocol evidence, not
//! vendor authentication, Bluetooth service readiness or physical validation.
//! Also supports the Apple USB ROM setup from the pinned Linux btbcm.
//! USB media and the distinct BCM4377 PCIe transport remain separate backends.
pub mod diagnostics;
mod identity;
mod initialize;
mod patch;

use super::{FirmwareReceipt, Hci, Io, Packet};
use alloc::{boxed::Box, vec::Vec};
pub(crate) use identity::listed;
pub(crate) use identity::{APPLE_VENDOR, is_apple_interface};
pub use identity::{Identity, LocalVersion, UsbIdentity, is_vendor_interface};
pub use patch::{Hcd, Record};

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Attempt {
    pub version: Option<LocalVersion>,
    pub verbose: Option<[u8; 7]>,
    pub features: Option<[u8; 9]>,
    pub local_name: Option<Vec<u8>>,
    pub product: Option<[u8; 5]>,
    pub download_mode_entered: bool,
    pub after_features: Option<[u8; 9]>,
    pub after_local_name: Option<Vec<u8>>,
    pub identity: Option<Identity>,
    pub firmware: Option<FirmwareReceipt>,
    /// Commands completed from the selected image, never submitted-only work.
    pub patch_commands: usize,
    pub after: Option<LocalVersion>,
    pub address: Option<[u8; 6]>,
    /// Unrelated asynchronous packets are held in wire order, not discarded.
    pub asynchronous: Vec<Packet>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedReport {
    pub before: Identity,
    pub after: LocalVersion,
    pub address: [u8; 6],
    pub after_features: [u8; 9],
    pub after_local_name: Vec<u8>,
    pub firmware: Box<FirmwareReceipt>,
    pub patch_commands: usize,
    pub command_credits: u8,
    /// Packets observed while initializing; the eventual host must deliver them.
    pub asynchronous: Vec<Packet>,
    /// Source-required policy: controller duplicate filtering is strict.
    pub strict_duplicate_filter: bool,
}

/// Observations from the Apple ROM setup. Rejected optional commands have no
/// observation. No host firmware was downloaded by this setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppleReport {
    pub usb: UsbIdentity,
    pub interface: u8,
    pub verbose: Option<[u8; 7]>,
    pub product: Option<[u8; 5]>,
    pub features: Option<[u8; 9]>,
    pub local_name: Option<Vec<u8>>,
    pub command_credits: u8,
    pub asynchronous: Vec<Packet>,
    pub strict_duplicate_filter: bool,
}

pub(super) use initialize::run_apple as initialize_apple;

#[derive(Debug)]
pub enum Error {
    Unsupported(&'static str),
    Protocol {
        requirement: &'static str,
        observed: Vec<u8>,
    },
    Resource(crate::firmware::Error),
    MalformedPatch {
        offset: usize,
    },
    Controller {
        opcode: u16,
        status: u8,
        parameters: Vec<u8>,
    },
    InvalidAddress([u8; 6]),
    Timeout(&'static str),
    Transport(super::Error),
    Attempt {
        cause: Box<Error>,
        observation: Box<Attempt>,
    },
    Cleanup {
        cause: Box<Error>,
        cleanup: super::Error,
    },
}
impl Error {
    fn protocol(requirement: &'static str, observed: &[u8]) -> Self {
        Self::Protocol {
            requirement,
            observed: observed.to_vec(),
        }
    }
}
impl From<super::Error> for Error {
    fn from(value: super::Error) -> Self {
        Self::Transport(value)
    }
}
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Broadcom USB Bluetooth initialization {self:?}")
    }
}
impl core::error::Error for Error {}

pub(super) fn is_candidate(device: &crate::usb::xhci::Device) -> bool {
    identity::is_candidate(device)
}

/// Download the exact USB-controller PatchRAM image with no guessed address.
/// Use the native initialization wrapper to keep a single attempt per
/// attachment. Failure closes/quarantines that attachment; no firmware retry.
#[expect(
    clippy::result_large_err,
    reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
)]
pub fn initialize<B: Io>(
    hci: &mut Hci<B>,
    provider: &dyn crate::firmware::Provider,
) -> Result<OwnedReport> {
    initialize::run(hci, provider, None)
}

/// A platform-supplied controller address can replace a source-listed
/// default address. This is the host-order six-byte BD_ADDR as transmitted by
/// HCI, not a generated/random identity. No platform value is invented.
#[expect(
    clippy::result_large_err,
    reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
)]
pub fn initialize_with_address<B: Io>(
    hci: &mut Hci<B>,
    provider: &dyn crate::firmware::Provider,
    address: [u8; 6],
) -> Result<OwnedReport> {
    initialize::run(hci, provider, Some(address))
}

#[cfg(test)]
mod tests;
