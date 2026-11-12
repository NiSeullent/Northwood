// SPDX-License-Identifier: GPL-2.0-only
// Copyright(c) 1999 - 2018 Intel Corporation.
// Copyright (c) 2026 Stela contributors
//! Probe admission of the PCH cohorts this composition implements, as
//! Linux e1000e's `e1000_probe` binds them: the 82579 (PCH2), the
//! ULP-exempt LPT I217/I218 IDs, the I219 SPT/CNP/TGP/ADP table entries and
//! the MTP/ARL/LNP and PTP/NVL entries. The cohort comes from the PCI device
//! ID, as Linux's `board_*` driver data does.
//!
//! BAR0 holds the MAC. Before SPT the NVM flash registers are BAR1; from SPT
//! on they are BAR0 at `E1000_FLASH_BASE_ADDR` (0xe000), as `e1000_probe`
//! maps them. The function's configuration space stays with the adapter for
//! the live LTR and descriptor-flush-status reads LPT and later make.
//!
//! The LPT-LP I218 (155a/1559) and I218-3 (15a2/15a3) need the K1 hang
//! sequence and ULP entry/exit, which are not implemented; they stay
//! unclaimed, as do the ICH8-10 and first-generation PCH parts. Firmware
//! owned (managed) MACs and an initially inaccessible PHY are refused by
//! the cold controller with their typed error.
use super::{Controller, Error, lpt::LptRegisters, mtp, queue::Ethernet};
use crate::{
    Handoff, Link, LinkError, Mac,
    block::BlockError,
    bus::{
        dma::DmaAllocator,
        mmio::Mmio,
        pci::{self, ConfigSpace, Function},
    },
};
use alloc::{format, rc::Rc, string::String, vec::Vec};

pub use mtp::{HostIdentity, Observation};

/// `E1000_FLASH_BASE_ADDR`: the SPT+ flash registers' offset in BAR0.
const FLASH_IN_MAC: u64 = 0xe000;
/// The MAC window the cold composition reaches: the highest register it
/// names is FEXTNVM12 at 0x5bc0, and the SPT+ flash follows at 0xe000.
const MAC_MINIMUM: u64 = 0x5bc4;

/// The Linux `board_*` group a PCH device ID selects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cohort {
    /// `board_pch2lan`: 82579LM/V.
    Pch2,
    /// `board_pch_lpt`, the IDs `e1000_enable_ulp_lpt_lp` exempts.
    Lpt,
    /// `board_pch_spt`, `board_pch_cnp`, `board_pch_tgp`, `board_pch_adp`.
    I219,
    /// `board_pch_mtp` and `board_pch_ptp`.
    Mtp,
}

impl Cohort {
    pub fn of(device: u16) -> Option<Self> {
        Some(match device {
            0x1502 | 0x1503 => Self::Pch2,
            0x153a | 0x153b | 0x15a0 | 0x15a1 => Self::Lpt,
            device if super::i219::Family::from_device(device).is_some() => Self::I219,
            device if mtp::Family::from_device(device).is_some() => Self::Mtp,
            _ => return None,
        })
    }

    /// Whether the flash registers are their own BAR1 (before SPT).
    fn separate_flash(self) -> bool {
        matches!(self, Self::Pch2 | Self::Lpt)
    }

    fn name(self) -> &'static str {
        match self {
            Self::Pch2 => "82579",
            Self::Lpt => "I217/I218",
            Self::I219 | Self::Mtp => "I219",
        }
    }
}

/// Candidate selection from copied identity data.
pub fn matches_identity(identity: &pci::Identity) -> bool {
    identity.vendor == 0x8086
        && identity.class == [2, 0, 0]
        && Cohort::of(identity.device).is_some()
}

pub fn matches<C: ConfigSpace>(function: &Function<C>) -> bool {
    matches_identity(&function.identity())
}

/// Why a PCH function did not attach.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AttachError {
    Device(BlockError),
    Controller(Error),
}

impl core::fmt::Display for AttachError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Device(error) => write!(f, "e1000e PCH: {error}"),
            Self::Controller(error) => write!(f, "e1000e PCH: {error}"),
        }
    }
}

impl core::error::Error for AttachError {}

impl From<BlockError> for AttachError {
    fn from(error: BlockError) -> Self {
        Self::Device(error)
    }
}

impl From<pci::ConfigError> for AttachError {
    fn from(error: pci::ConfigError) -> Self {
        Self::Device(BlockError::Config(error))
    }
}

impl From<Error> for AttachError {
    fn from(error: Error) -> Self {
        Self::Controller(error)
    }
}

/// Bounds and alignment of one 32-bit access in a `bytes`-long window.
fn checked(at: u32, width: u32, bytes: usize) -> Result<usize, ()> {
    let at = usize::try_from(at).map_err(|_| ())?;
    let width = width as usize;
    if !at.is_multiple_of(width) || at.checked_add(width).is_none_or(|end| end > bytes) {
        return Err(());
    }
    Ok(at)
}

/// The MAC and flash windows of one PCH function and its configuration.
pub struct PciIo<C: ConfigSpace + 'static> {
    mac: Mmio,
    flash: Mmio,
    function: Function<C>,
}

impl<C: ConfigSpace + 'static> PciIo<C> {
    /// # Safety
    /// `mac` and `flash` are uncached mappings of this function's MAC and
    /// flash register windows, and nothing else drives them or the function.
    pub unsafe fn new(mac: Mmio, flash: Mmio, function: Function<C>) -> Self {
        Self {
            mac,
            flash,
            function,
        }
    }
}

// SAFETY: the constructor's mappings belong to this one function and are
// owned by it; every access is bounds- and alignment-checked first.
unsafe impl<C: ConfigSpace + 'static> super::Registers for PciIo<C> {
    fn read32(&mut self, at: u32) -> Result<u32, super::Fault> {
        let at = checked(at, 4, self.mac.len()).map_err(|()| super::Fault::OutsideBar)?;
        Ok(self.mac.read32(at))
    }
    fn write32(&mut self, at: u32, value: u32) -> Result<(), super::Fault> {
        let at = checked(at, 4, self.mac.len()).map_err(|()| super::Fault::OutsideBar)?;
        self.mac.write32(at, value);
        Ok(())
    }
}

// SAFETY: as above; the flash window is this function's own.
unsafe impl<C: ConfigSpace + 'static> super::nvm::Registers for PciIo<C> {
    fn read_mac32(&mut self, at: u32) -> Result<u32, super::nvm::Error> {
        super::Registers::read32(self, at).map_err(|_| super::nvm::Error::Access)
    }
    fn read16(&mut self, at: u32) -> Result<u16, super::nvm::Error> {
        let at = checked(at, 2, self.flash.len()).map_err(|()| super::nvm::Error::Bounds)?;
        Ok(self.flash.read16(at))
    }
    fn read32(&mut self, at: u32) -> Result<u32, super::nvm::Error> {
        let at = checked(at, 4, self.flash.len()).map_err(|()| super::nvm::Error::Bounds)?;
        Ok(self.flash.read32(at))
    }
    fn write16(&mut self, at: u32, value: u16) -> Result<(), super::nvm::Error> {
        let at = checked(at, 2, self.flash.len()).map_err(|()| super::nvm::Error::Bounds)?;
        self.flash.write16(at, value);
        Ok(())
    }
    fn write32(&mut self, at: u32, value: u32) -> Result<(), super::nvm::Error> {
        let at = checked(at, 4, self.flash.len()).map_err(|()| super::nvm::Error::Bounds)?;
        self.flash.write32(at, value);
        Ok(())
    }
}

// SAFETY: the configuration space is the same function's, owned with its
// register windows; every read is live and a refusal stays an error.
unsafe impl<C: ConfigSpace + 'static> LptRegisters for PciIo<C> {
    fn read_config16(&mut self, offset: u16) -> Result<u16, super::Fault> {
        self.function
            .config()
            .read16(offset)
            .map_err(|_| super::Fault::Access)
    }
}

/// This host's SMBIOS system fields as the MTP/PTP board quirk reads them.
pub fn host_identity<'a>(system: &crate::firmware::smbios::SystemIdentity<'a>) -> HostIdentity<'a> {
    use crate::firmware::smbios::Observation as Smbios;
    let field = |observation: Smbios<'a>| match observation {
        Smbios::NotObserved => Observation::NotObserved,
        Smbios::Absent => Observation::Absent,
        Smbios::Value(bytes) => Observation::Value(bytes),
    };
    HostIdentity {
        system_vendor: field(system.system_vendor),
        product_name: field(system.product_name),
        product_version: field(system.product_version),
    }
}

type Timer = Rc<dyn crate::bus::pci::power::Timer>;

/// An attached PCH adapter.
pub struct Pch {
    link: Ethernet<PciIo<pci::EcamConfig>, Timer>,
    pub location: String,
}

impl Pch {
    /// Initialize one exclusively owned PCH function after ExitBootServices.
    /// `host` carries this machine's SMBIOS system fields, which the MTP and
    /// PTP cohorts need for Linux's K1 board quirk; the others ignore it.
    ///
    /// # Safety
    /// The caller exclusively owns the function, its configuration and both
    /// register BARs, which the handoff mapped uncached. Firmware, SMM
    /// excepted as the cold controller's FWSM check decides, and every other
    /// driver have stopped using it. `dma` is identity-mapped DMA memory.
    pub unsafe fn attach(
        mut function: Function,
        handoff: &Handoff,
        dma: &mut dyn DmaAllocator,
        minimum: Timer,
        host: HostIdentity<'_>,
    ) -> Result<Self, AttachError> {
        let cohort = Cohort::of(function.device)
            .filter(|_| matches(&function))
            .ok_or(BlockError::Unsupported(
                "not an implemented e1000e PCH device",
            ))?;
        let device = function.device;
        let location = format!("{} Intel {}", function.address, cohort.name());
        function.power_up(minimum.as_ref())?;
        let bars = function.bars()?;
        let length = |bar: Option<pci::Bar>| {
            bar.and_then(|bar| bar.memory())
                .map(|range| range.end - range.start)
                .ok_or(BlockError::Unsupported(
                    "e1000e PCH register BAR is not memory",
                ))
        };
        let mac_bytes = length(bars[0])?;
        if mac_bytes < MAC_MINIMUM || (!cohort.separate_flash() && mac_bytes < FLASH_IN_MAC + 20) {
            return Err(BlockError::Unsupported("e1000e PCH MAC BAR is too small").into());
        }
        let mac = crate::registers(handoff, bars[0], 0, mac_bytes)?;
        let flash = if cohort.separate_flash() {
            crate::registers(handoff, bars[1], 0, length(bars[1])?)?
        } else {
            crate::registers(handoff, bars[0], FLASH_IN_MAC, mac_bytes - FLASH_IN_MAC)?
        };
        function.enable_polled()?;
        // SAFETY: the caller's exclusive grant covers both windows, which
        // `registers` checked are mapped and inside this function's BARs.
        let io = unsafe { PciIo::new(mac, flash, function) };
        // SAFETY: as above; this is a new cold epoch of this function, and
        // `minimum` is the platform's real elapsed-time source.
        let controller = unsafe {
            match cohort {
                Cohort::Pch2 => Controller::new(device, io, minimum),
                Cohort::Lpt => Controller::new_lpt(device, io, minimum),
                Cohort::I219 => Controller::new_i219(device, io, minimum),
                Cohort::Mtp => Controller::new_mtp(device, io, minimum, host),
            }
        }?;
        // SAFETY: memory decoding and bus mastering are on, and `dma` is the
        // caller's identity-mapped DMA memory.
        let link = unsafe { Ethernet::new(controller, dma) }?;
        Ok(Self { link, location })
    }

    /// Stop both queues and hand the PHY back, as `e1000e_down` does.
    pub fn stop(&mut self) -> Result<(), Error> {
        self.link.shutdown()
    }
}

impl Link for Pch {
    fn mac(&self) -> Mac {
        self.link.mac()
    }
    fn carrier(&mut self) -> bool {
        self.link.carrier()
    }
    fn transmit(&mut self, frame: &[u8]) -> Result<(), LinkError> {
        self.link.transmit(frame)
    }
    fn receive(&mut self) -> Result<Option<Vec<u8>>, LinkError> {
        self.link.receive()
    }
    fn receive_reusing(&mut self, frame: &mut Vec<u8>) -> Result<bool, LinkError> {
        self.link.receive_reusing(frame)
    }
}

#[cfg(test)]
mod tests;
