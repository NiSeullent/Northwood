// SPDX-License-Identifier: GPL-2.0-only
// Register definitions follow Linux include/uapi/linux/pci_regs.h, copyright
// 1994 Drew Eckhardt and 1997-1999 Martin Mares; stopped PRI/PASID/VF behavior
// follows drivers/pci/ats.c, Copyright (C) 2009 Intel Corporation,
// Yu Zhao <yu.zhao@intel.com>, Copyright (C) 2011 Advanced Micro Devices,
// PRI and PASID additions by Joerg Roedel <joerg.roedel@amd.com>, at commit
// cc9de5d4b5ab6459f5774946aa3195ad51236fa8. See NOTICE.
//! Stopped PCI endpoints. A claim grants neither alias isolation nor
//! a remapping-unit takeover. Engine/IRQ/work drainage and physical
//! function identity are unsafe producer obligations, not config-bit inferences.
use super::regs;
use super::{
    Address, BARS, Bar, CapabilityError, ConfigSpace, Function, Identity,
    owned::{ConfigRetention, OwnedConfig},
};
use crate::bus::mmio::owned::{Error as MappingError, Mapping};
use core::mem::ManuallyDrop;

mod irq_identity;
pub use irq_identity::RequesterId;
pub mod containment;
pub mod preparation;
pub mod roles;

const PM: u16 = 1;
const MSI: u16 = 5;
const PCIX: u16 = 7;
const VENDOR: u16 = 9;
const PCIE: u16 = 0x10;
const MSIX: u16 = 0x11;
const ATS: u16 = 0xf;
const PRI: u16 = 0x13;
const PASID: u16 = 0x1b;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FunctionRole {
    Physical,
    Virtual,
    Unverified,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClaimError {
    Config(super::ConfigError),
    Mapping(MappingError),
    AlreadyBound,
    IdentityChanged,
    ReadFailed { offset: u16 },
    Active { offset: u16 },
    MalformedCapability { offset: u16 },
    UnsupportedHeader(u8),
    UnsupportedRole(FunctionRole),
    UnsupportedCapability { id: u16, extended: bool },
    UnsupportedVersion { id: u16, version: u8 },
    UnsupportedPcieType(u8),
}
impl From<super::ConfigError> for ClaimError {
    fn from(error: super::ConfigError) -> Self {
        Self::Config(error)
    }
}

/// The original configuration owner, with its immutable ECAM BDF and root/
/// resource/reset epoch kept.
pub struct StoppedRequester {
    function: Function<OwnedConfig>,
    address: Address,
    dma_bound: bool,
}

/// Non-cloneable binding derived from the original validated stopped function.
/// It keeps its native epoch but supplies no second config writer, new
/// coherency/alias proof, or observation of current endpoint quiescence. The
/// original unsafe stopped/attachment grant persists until explicit retirement.
pub struct RequesterBinding {
    address: Address,
    retention: ConfigRetention,
}
impl core::fmt::Debug for RequesterBinding {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RequesterBinding")
            .field("address", &self.address)
            .finish_non_exhaustive()
    }
}
impl RequesterBinding {
    pub fn address(&self) -> Address {
        self.address
    }
    pub fn requester(&self) -> u16 {
        u16::from(self.address.bus) << 8
            | u16::from(self.address.device) << 3
            | u16::from(self.address.function)
    }
    pub fn check_live(&self) -> Result<(), MappingError> {
        self.retention.check_live()
    }
    pub fn matches_mapping(&self, mapping: &Mapping) -> Result<bool, MappingError> {
        self.retention.matches_mapping(mapping)
    }
}

impl StoppedRequester {
    /// # Safety
    /// The endpoint's engines, DMA/posted requests, IRQs and workers are
    /// stopped and drained. Payload WB DMA coherency has been independently
    /// checked for this device and platform. `Physical` must come from the
    /// complete PF/VF inventory, never local capability absence. Firmware,
    /// reset, removal and alias/config owners cannot change this epoch.
    /// This grants no isolation from aliases and no control over other requesters.
    /// The caller and final release obey process/deferred-context requirements.
    /// Rejection returns the original function and mapping owner.
    pub unsafe fn from_stopped_pci(
        function: Function<OwnedConfig>,
        role: FunctionRole,
    ) -> Result<Self, (ClaimError, Function<OwnedConfig>)> {
        let mut function = ManuallyDrop::new(function);
        if role != FunctionRole::Physical {
            return Err((
                ClaimError::UnsupportedRole(role),
                ManuallyDrop::into_inner(function),
            ));
        }
        let address = function.config().address();
        if function.address != address {
            return Err((
                ClaimError::IdentityChanged,
                ManuallyDrop::into_inner(function),
            ));
        }
        if let Err(error) = validate_stopped(&mut function) {
            return Err((error, ManuallyDrop::into_inner(function)));
        }
        Ok(Self {
            function: ManuallyDrop::into_inner(function),
            address,
            dma_bound: false,
        })
    }

    pub fn address(&self) -> Address {
        self.address
    }

    pub fn identity(&self) -> Identity {
        self.function.identity()
    }

    /// Size BARs while retaining the original stop and DMA binding state.
    /// On refusal, retain this owner and do not retry as though no write ran.
    pub fn bars(&mut self) -> Result<[Option<Bar>; BARS], ClaimError> {
        self.check_stopped()?;
        let bars = self.function.bars()?;
        self.check_stopped()?;
        Ok(bars)
    }

    pub fn requester(&self) -> u16 {
        u16::from(self.address.bus) << 8
            | u16::from(self.address.device) << 3
            | u16::from(self.address.function)
    }

    /// Recheck configuration observations without inferring new engine
    /// quiescence or renewing the original producer's epoch grant.
    pub fn check_stopped(&mut self) -> Result<(), ClaimError> {
        if self.function.address != self.address || self.function.config().address() != self.address
        {
            return Err(ClaimError::IdentityChanged);
        }
        validate_stopped(&mut self.function)
    }

    /// Transfer the owner. Consuming the claim does not end its
    /// mapping lifetime or authorize bus mastering; the recipient owns config.
    pub fn into_function(self) -> Function<OwnedConfig> {
        self.function
    }

    /// Return this original stopped owner and keep its
    /// mapping/root epoch for one DMA domain. No new function or access path is
    /// manufactured. A private flag forbids a second safe split. Any failure
    /// returns the same original stopped owner.
    /// The sole original owner must not reset, remove or reassign in a way
    /// invalidating the resource/routing/coherency/translation epoch.
    /// Only a specifically checked bounded reset preserving those owners and
    /// ATS/NoSnoop policy is admissible. Reset supplies neither stopped/drained
    /// proof nor remapper completion; keeping a mapping cannot enforce it.
    pub fn into_dma_binding(mut self) -> Result<(Self, RequesterBinding), (ClaimError, Self)> {
        if self.dma_bound {
            return Err((ClaimError::AlreadyBound, self));
        }
        if let Err(error) = self.check_stopped() {
            return Err((error, self));
        }
        let retention = match self.function.config().retain_epoch() {
            Ok(retention) => retention,
            Err(error) => return Err((ClaimError::Mapping(error), self)),
        };
        let binding = RequesterBinding {
            address: self.address,
            retention,
        };
        self.dma_bound = true;
        Ok((self, binding))
    }
}

fn read16<C: ConfigSpace>(config: &C, offset: u16) -> Result<u16, ClaimError> {
    let value = config.read16(offset)?;
    if value == u16::MAX {
        Err(ClaimError::ReadFailed { offset })
    } else {
        Ok(value)
    }
}

fn inactive<C: ConfigSpace>(config: &C, offset: u16, mask: u16) -> Result<u16, ClaimError> {
    let value = read16(config, offset)?;
    if value & mask != 0 {
        return Err(ClaimError::Active { offset });
    }
    Ok(value)
}

impl From<CapabilityError> for ClaimError {
    fn from(error: CapabilityError) -> Self {
        match error {
            CapabilityError::Config { error, .. } => Self::Config(error),
            CapabilityError::FailedRead { offset } => Self::ReadFailed { offset },
            error => Self::MalformedCapability {
                offset: error.offset(),
            },
        }
    }
}

/// Wait for PCIe outstanding transactions to finish before changing DMA
/// ownership. `expired` uses the caller's retained clock/deadline. An idle
/// status is only a configuration observation; engine, IRQ and worker
/// quiescence remain the stopped requester's independent producer obligations.
pub fn wait_for_pending_transactions<C: ConfigSpace>(
    config: &C,
    mut expired: impl FnMut() -> bool,
) -> Result<(), ClaimError> {
    let mut express = None;
    for entry in config.capabilities() {
        let entry = entry?;
        if u16::from(entry.id) == PCIE && express.replace(entry.offset).is_some() {
            return Err(ClaimError::MalformedCapability {
                offset: entry.offset,
            });
        }
    }
    let Some(express) = express else {
        return Ok(());
    };
    let status = express + 10;
    while read16(config, status)? & (1 << 5) != 0 {
        if expired() {
            return Err(ClaimError::Active { offset: status });
        }
        core::hint::spin_loop();
    }
    Ok(())
}

/// Extended capabilities that report or tune the link and make no request of
/// their own, by their `pci_regs.h` IDs: the null entry, AER, virtual channels,
/// the serial number, power budgeting, vendor-specific, ACS, ARI, Resizable BAR
/// and its VF twin, LTR, secondary PCIe, L1 substates, data link features, the
/// 16 and 32 GT/s physical layers and lane margining.
const INERT: [u16; 17] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x0b, 0x0d, 0x0e, 0x15, 0x18, 0x19, 0x1e, 0x24, 0x25, 0x26, 0x27,
    0x2a,
];
/// SR-IOV: inert while its virtual functions, which are requesters, are off.
const SRIOV: u16 = 0x10;
/// `PCI_SRIOV_CTRL`, and its `PCI_SRIOV_CTRL_VFE`.
const SRIOV_CONTROL: u16 = 8;
const SRIOV_VF_ENABLE: u16 = 1;

fn standard<C: ConfigSpace>(config: &C, status: u16) -> Result<(), ClaimError> {
    if status & regs::STATUS_CAP_LIST == 0 {
        return Ok(());
    }
    let mut singleton = 0u32;
    for entry in config.capabilities() {
        let entry = entry?;
        let offset = entry.offset;
        let id = u16::from(entry.id);
        match id {
            PM => {
                let version = (read16(config, offset + 2)? & 7) as u8;
                if !(1..=3).contains(&version) {
                    return Err(ClaimError::UnsupportedVersion { id, version });
                }
            }
            PCIX => {
                let version = ((read16(config, offset + 2)? >> 12) & 3) as u8;
                if !matches!(version, 0..=2) {
                    return Err(ClaimError::UnsupportedVersion { id, version });
                }
            }
            MSI => {
                let flags = inactive(config, offset + 2, 1)?;
                let supported = (flags >> 1) & 7;
                let selected = (flags >> 4) & 7;
                if flags & !0x1ff != 0 || supported > 5 || selected > supported {
                    return Err(ClaimError::MalformedCapability { offset });
                }
            }
            MSIX => {
                inactive(config, offset + 2, 1 << 15)?;
            }
            PCIE => {
                let flags = read16(config, offset + 2)?;
                let version = (flags & 15) as u8;
                if !matches!(version, 1 | 2) {
                    return Err(ClaimError::UnsupportedVersion { id, version });
                }
                let kind = ((flags >> 4) & 15) as u8;
                if !matches!(kind, 0 | 1 | 9) || flags & 0x100 != 0 {
                    return Err(ClaimError::UnsupportedPcieType(kind));
                }
                inactive(config, offset + 8, 1 << 11)?;
                inactive(config, offset + 10, 1 << 5)?;
            }
            VENDOR => {}
            _ => {
                return Err(ClaimError::UnsupportedCapability {
                    id,
                    extended: false,
                });
            }
        }
        if id != VENDOR {
            let bit = 1 << id;
            if singleton & bit != 0 {
                return Err(ClaimError::MalformedCapability { offset });
            }
            singleton |= bit;
        }
    }
    Ok(())
}

fn extended<C: ConfigSpace>(config: &C) -> Result<(), ClaimError> {
    // Linux `pci_cfg_space_size_ext`: an all-ones first extended header means
    // no extended space, as on every conventional PCI function. A removal is
    // caught by the identity read that follows the walk.
    if config.read32(0x100)? == u32::MAX {
        return Ok(());
    }
    let mut singleton = 0u32;
    for entry in config.extended_capabilities() {
        let entry = entry?;
        let (offset, id, version) = (entry.offset, entry.id, entry.version);
        if INERT.contains(&id) {
            continue;
        }
        if id == SRIOV {
            inactive(config, offset + SRIOV_CONTROL, SRIOV_VF_ENABLE)?;
            continue;
        }
        if !matches!(id, ATS | PASID | PRI) {
            return Err(ClaimError::UnsupportedCapability { id, extended: true });
        }
        if version != 1 {
            return Err(ClaimError::UnsupportedVersion { id, version });
        }
        let bit = 1 << id;
        if singleton & bit != 0 {
            return Err(ClaimError::MalformedCapability { offset });
        }
        singleton |= bit;
        match id {
            ATS => {
                inactive(config, offset + 6, 1 << 15)?;
            }
            PASID => {
                inactive(config, offset + 6, 1)?;
            }
            _ => {
                inactive(config, offset + 4, 3)?;
                if read16(config, offset + 6)? & 0x100 == 0 {
                    return Err(ClaimError::Active { offset: offset + 6 });
                }
            }
        }
    }
    Ok(())
}

/// Recheck PCI containment observations only. This supplies no engine stop,
/// payload/IRQ/work retirement, physical-role proof or new requester capability.
pub fn validate_stopped<C: ConfigSpace>(function: &mut Function<C>) -> Result<(), ClaimError> {
    let expected = u32::from(function.vendor) | u32::from(function.device) << 16;
    let class = function.class;
    let config = function.config();
    let identity = config.read32(0)?;
    if identity == u32::MAX {
        return Err(ClaimError::ReadFailed { offset: 0 });
    }
    let actual_class = config.read32(8)?;
    if actual_class == u32::MAX {
        return Err(ClaimError::ReadFailed { offset: 8 });
    }
    if identity != expected
        || [
            (actual_class >> 24) as u8,
            (actual_class >> 16) as u8,
            (actual_class >> 8) as u8,
        ] != class
    {
        return Err(ClaimError::IdentityChanged);
    }
    let header = config.read8(0x0e)?;
    if header == u8::MAX {
        return Err(ClaimError::ReadFailed { offset: 0x0e });
    }
    if header & 0x7f != 0 {
        return Err(ClaimError::UnsupportedHeader(header & 0x7f));
    }
    if inactive(config, regs::COMMAND, regs::COMMAND_MASTER)? & regs::COMMAND_INTX_DISABLE == 0 {
        return Err(ClaimError::Active {
            offset: regs::COMMAND,
        });
    }
    standard(config, read16(config, regs::STATUS)?)?;
    extended(config)?;
    // A second identity read catches observed removal/reset during traversal.
    // The kept epoch grant, not this observation, excludes concurrent change.
    if config.read32(0)? != identity {
        return Err(ClaimError::IdentityChanged);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
