// SPDX-License-Identifier: GPL-2.0-only
// Copyright(c) 1999 - 2018 Intel Corporation.
// Copyright (c) 2026 Stela contributors
//! Initial SPT I219 LM/V (156f/1570), with the integrated flash window,
//! unmanaged force-ULP exit and descriptor-flush status. Later I219 generations
//! are not selected by their common I217 PHY ID. Initial PHY identity must be
//! accessible; firmware/SMM, ME ownership and other users remain excluded.
use super::{
    Controller, Delay, Error, Fault, Identity, Kind, Phy, Profile, Register, Registers, i219, lpt,
    nvm,
};

pub(super) const DESCRIPTOR_STATUS: u16 = 0xe4;
pub(super) const FLUSH_REQUIRED: u16 = 0x100;
pub(super) const FEXTNVM11: u32 = 0x5bbc;
pub(super) const DISABLE_MULR_FIX: u32 = 0x2000;
pub(super) const FEXTNVM7: u32 = 0xe4;
pub(super) const SIDE_CLOCK_UNGATE: u32 = 4;
pub(super) const FEXTNVM9: u32 = 0x5bb4;
pub(super) const IOSF_CLOCK_GATING_DISABLE: u32 = 0x1800;
pub(super) const IOSFPC: u32 = 0xf28;
pub(super) const CORRUPTION_THRESHOLD: u32 = 1 << 16;
pub(super) const TARC_REQUEST_MASK: u32 = 0x3000_0000;
pub(super) const TARC_TWO_REQUESTS: u32 = 0x2000_0000;
pub(super) const PCIE_ANALOG: u32 = 0xf18;
pub(super) const K1_OFF_ENABLE: u32 = 1 << 31;
pub(super) const FLUSH_BYTES: u32 = 512;
const DISABLE_SMB_PERST: u32 = 0x20;
const CTRL_EXT: u32 = 0x18;
const FORCE_SMBUS: u32 = 1 << 11;
const POWER_CYCLE_DONE: u32 = 4;
const LANPHYPC_OVERRIDE: u32 = 1 << 16;
const LANPHYPC_VALUE: u32 = 1 << 17;
const SMB_CONTROL: Register = Register {
    page: 769,
    index: 23,
};
const PM_CONTROL: Register = Register {
    page: 770,
    index: 17,
};
const K1_ENABLE: u16 = 1 << 14;
const ULP_CONFIG: Register = Register {
    page: 779,
    index: 16,
};
const ULP_CLEAR: u16 = 0x4 | 0x10 | 0x20 | 0x40 | 0x100 | 0x400 | 0x800 | 0x1000;
const ULP_START: u16 = 1;

impl<R: lpt::LptRegisters, D: Delay + Clone> Controller<R, D> {
    /// Construct only the initial SPT I219 LM/V cohort.
    ///
    /// # Safety
    /// All `Controller::new` same-function physical ownership/reset/exclusion
    /// requirements apply, with vendor8086/class020000/device156f/1570.
    /// Config/MMIO/integrated MAC+e000 flash/PHY share one kept
    /// Function/reset epoch. LTR words remain immutable, while config e4 is a
    /// real live read at every required reset/stop point. The initial PHY is
    /// accessible and identifies as I217 before the forced interconnect cycle.
    /// A foreign ring requiring flush is refused without touching foreign RAM.
    /// No native config/MMIO/DMA provider or probe admission is supplied here.
    pub unsafe fn new_spt(device: u16, io: R, delay: D) -> Result<Self, Error> {
        // SAFETY: caller supplies the live same-function cold epoch.
        let mut this = unsafe { Self::claim_spt(device, io, delay) }?;
        if let Err(error) = this.initialize_i219() {
            this.poisoned = true;
            this.phy.poisoned = true;
            return Err(error);
        }
        Ok(this)
    }
    pub(super) unsafe fn claim_spt(device: u16, io: R, delay: D) -> Result<Self, Error> {
        if !matches!(device, 0x156f | 0x1570) {
            return Err(Error::UnsupportedDevice(device));
        }
        // SAFETY: same exact initial SPT grant passed to the shared owner.
        unsafe { Self::claim_i219(device, io, delay) }
    }
    pub(super) unsafe fn claim_i219(device: u16, io: R, delay: D) -> Result<Self, Error> {
        let family = i219::Family::from_device(device).ok_or(Error::UnsupportedDevice(device))?;
        // SAFETY: source-selected family under the same owner grant.
        unsafe { Self::claim_integrated(io, delay, |limits| Profile::I219(limits, family)) }
    }
    pub(super) unsafe fn claim_integrated(
        io: R,
        delay: D,
        profile: impl FnOnce(lpt::Limits) -> Profile,
    ) -> Result<Self, Error> {
        let (bus, delay) = super::qualified_bus(io, delay)?;
        let limits = (|| {
            let snoop = bus.0.borrow_mut().read_config16(0xa8)?;
            let nosnoop = bus.0.borrow_mut().read_config16(0xaa)?;
            lpt::Limits::new(snoop, nosnoop)
        })();
        let limits = match limits {
            Ok(value) => value,
            Err(error) => {
                core::mem::forget((bus, delay));
                return Err(error);
            }
        };
        // SAFETY: the SPT function owns its integrated window.
        let nvm = unsafe { nvm::Nvm::new(nvm::Layout::Spt, bus.clone(), delay.clone()) }?;
        // SAFETY: exact PCI cohort selects the reviewed I217 transport.
        let phy = unsafe { Phy::new(Kind::I217, bus, delay) };
        Ok(Self {
            phy,
            nvm,
            station: crate::Mac([0; 6]),
            identity: Identity {
                kind: Kind::I217,
                id: 0,
                revision: 0,
            },
            poisoned: false,
            profile: profile(limits),
            config_read: Some(R::read_config16),
        })
    }
}

impl<R: Registers + nvm::Registers, D: Delay + Clone> Controller<R, D> {
    pub(super) fn i219_family(&self) -> Option<i219::Family> {
        match self.profile {
            Profile::I219(_, family) => Some(family),
            _ => None,
        }
    }
    pub(super) fn is_i219(&self) -> bool {
        self.i219_family().is_some() || self.mtp_family().is_some()
    }
    pub(super) fn is_spt_mac(&self) -> bool {
        self.i219_family() == Some(i219::Family::Spt)
    }
    pub(super) fn spt_config_status(&mut self) -> Result<u16, Error> {
        self.phy.check_minimum()?;
        let read = self.config_read.ok_or(Error::Configuration)?;
        let io = self.phy.io.as_ref().expect("retained same-function owner");
        let value = read(&mut *io.0.borrow_mut(), DESCRIPTOR_STATUS)?;
        if value == u16::MAX {
            return Err(Fault::Removed.into());
        }
        Ok(value)
    }
    fn inherited_ring_check(&mut self) -> Result<(), Error> {
        let length = self.read(0x3808)?;
        let status = self.spt_config_status()?;
        if length != 0 && status & FLUSH_REQUIRED != 0 {
            return Err(Error::DescriptorFlushRequired);
        }
        Ok(())
    }
    pub(super) fn initialize_i219(&mut self) -> Result<(), Error> {
        self.qualify()?;
        self.inherited_ring_check()?;
        self.identity = self.phy.identify()?;
        self.validate_nvm()?;
        self.station = self.nvm.station()?;
        self.gate(true)?;
        self.spt_ulp_exit()?;
        // The donor releases SWFLAG before this hardware PHY reset. No PHY
        // transaction wrapper may span it, or the later full-chip reset.
        self.phy_reset()?;
        self.configuration_done()?;
        self.post_reset()?;
        self.phy.wait_minimum(50_000)?;
        self.qualify()?;
        self.identity = self.phy.identify()?;
        if self.phy.read(ULP_CONFIG)? & ULP_CLEAR != 0 {
            return Err(Error::UlpState);
        }
        if self.phy.read(SMB_CONTROL)? & 1 != 0 || self.read(CTRL_EXT)? & FORCE_SMBUS != 0 {
            return Err(Error::SmbusMode);
        }
        self.mtp_initial_k1(true)?;
        // The source initialization flow follows forced ULP exit with another
        // generic PHY reset before normal MAC reset/PHY post-configuration.
        self.phy_reset()?;
        self.configuration_done()?;
        self.mtp_initial_k1(false)?;
        self.qualify()?;
        self.identity = self.phy.identify()?;
        self.write(super::PBA, super::RX_ALLOCATION_KIB)?;
        self.inherited_ring_check()?;
        let value = self.read(FEXTNVM11)?;
        self.write(FEXTNVM11, value | DISABLE_MULR_FIX)?;
        self.global_reset()?;
        self.configuration_done()?;
        self.post_reset()?;
        self.write(super::IMC, u32::MAX)?;
        self.read(super::ICR)?;
        let bias = self.read(super::KABGTXD)?;
        self.write(super::KABGTXD, bias | super::BGSQLBIAS)?;
        self.mtp_reset_finish()?;
        self.identity = self.phy.identify()?;
        Ok(())
    }
    /// Complete Intel's forced ULP exit even when the power-cycle status
    /// never arrives. The subsequent MDIC accesses still verify PHY access.
    fn spt_ulp_exit(&mut self) -> Result<(), Error> {
        self.phy.transaction(|p| {
            let value = p.read_mac(super::FEXTNVM3)?;
            p.write_mac(
                super::FEXTNVM3,
                (value & !super::CONFIG_COUNTER) | super::CONFIG_50MS,
            )?;
            let value = p.read_mac(super::CTRL)?;
            let value = (value | LANPHYPC_OVERRIDE) & !LANPHYPC_VALUE;
            p.write_mac(super::CTRL, value)?;
            p.read_mac(super::STATUS)?;
            p.wait_minimum(10)?;
            p.write_mac(super::CTRL, value & !LANPHYPC_OVERRIDE)?;
            p.read_mac(super::STATUS)?;
            for _ in 0..20 {
                p.wait_minimum(5_000)?;
                if p.read_mac(CTRL_EXT)? & POWER_CYCLE_DONE != 0 {
                    break;
                }
            }
            p.wait_minimum(30_000)?;
            p.retry_enabled = false;
            // A mode-switch MDIC error is a source-defined transition, not an
            // access/removal fault. Only the exact CV_SMB_CTRL response may
            // take this path; final readback must prove PCIe mode.
            p.page(SMB_CONTROL.page)?;
            let mut word = match p.mdic(1, SMB_CONTROL.index, None) {
                Ok(value) => value,
                Err(error) if smbus_read_unavailable(error) => {
                    let value = p.read_mac(CTRL_EXT)?;
                    p.write_mac(CTRL_EXT, value | FORCE_SMBUS)?;
                    p.wait_minimum(50_000)?;
                    p.page(SMB_CONTROL.page)?;
                    p.mdic(1, SMB_CONTROL.index, None)?
                }
                Err(error) => return Err(error),
            };
            let switching = word & 1 != 0;
            word &= !1;
            match p.mdic(1, SMB_CONTROL.index, Some(word)) {
                Ok(_) => {}
                Err(error) if switching && smbus_response_error(error) => {}
                Err(error) => return Err(error),
            }
            p.retry_enabled = true;
            let value = p.read_mac(CTRL_EXT)?;
            p.write_mac(CTRL_EXT, value & !FORCE_SMBUS)?;
            let value = p.access(PM_CONTROL, None)?;
            p.access(PM_CONTROL, Some(value | K1_ENABLE))?;
            let value = p.access(ULP_CONFIG, None)? & !ULP_CLEAR;
            p.access(ULP_CONFIG, Some(value))?;
            p.access(ULP_CONFIG, Some(value | ULP_START))?;
            let value = p.read_mac(FEXTNVM7)?;
            p.write_mac(FEXTNVM7, value & !DISABLE_SMB_PERST)?;
            Ok(())
        })?;
        Ok(())
    }
}
fn smbus_response_error(error: Fault) -> bool {
    let Fault::MdicResponse(response) = error else {
        return false;
    };
    let mask = super::super::MDIC_READY
        | super::super::MDIC_ERROR
        | super::super::MDIC_REG_MASK
        | (31 << 21);
    let expected = super::super::MDIC_READY
        | super::super::MDIC_ERROR
        | (u32::from(SMB_CONTROL.index) << 16)
        | (1 << 21);
    response & mask == expected
}
fn smbus_read_unavailable(error: Fault) -> bool {
    matches!(error, Fault::MdicTimeout) || smbus_response_error(error)
}

#[cfg(test)]
pub(super) mod tests;
