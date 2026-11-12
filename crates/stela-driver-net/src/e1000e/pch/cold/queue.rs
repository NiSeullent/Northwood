// SPDX-License-Identifier: GPL-2.0-only
// Copyright(c) 1999 - 2018 Intel Corporation.
// Copyright (c) 2026 Stela contributors
//! One polled queue for the declared cold PCH profiles. Native RX follows e1000e's
//! ordinary-MTU path; TX uses legacy descriptors without offloads. The PHY
//! jumbo workaround is explicitly restored to its non-jumbo settings. FCS is
//! received and removed in software as in that source profile. No IRQ, EEE,
//! pause, VLAN, jumbo, managed firmware or power-transition profile is claimed.
use super::{Controller, Delay, Error, Registers, i219, lpt, nvm, spt};
use crate::{Link, LinkError, MAX_FRAME, MIN_FRAME, Mac, bus::dma::DmaAllocator};
use alloc::{format, vec::Vec};
use core::sync::atomic::{Ordering, fence};

// Common descriptor storage only; the 82574-specific programming/reset/start
// routines are never called. Addresses are boot-lifetime coherent DMA RAM.
use crate::ring::{DESCRIPTOR, DESCRIPTORS, Ring, write_transmit_frame};

const CTRL_EXT: u32 = 0x18;
const FEXTNVM4: u32 = 0x24;
const KMRN: u32 = 0x34;
const TIPG: u32 = 0x410;
const RXCSUM: u32 = 0x5000;
const RFCTL: u32 = 0x5008;
const MTA: u32 = 0x5200;
const MTA_REGISTERS: u32 = 32;
const RAL: u32 = 0x5400;
const RAH: u32 = 0x5404;
const SHRAL: u32 = 0x5438;
const SHRAH: u32 = 0x543c;
const SHRAL_LPT: u32 = 0x5408;
const SHRAH_LPT: u32 = 0x540c;
const PBECCSTS: u32 = 0x100c;
const ECC_ENABLE: u32 = 1 << 16;
const MEMORY_ERROR_HANDLING: u32 = 1 << 19;
const FFLT_DBG: u32 = 0x5f04;
const GCR: u32 = 0x5b00;
const PBA: u32 = 0x1000;
const RDBAL: u32 = 0x2800;
const RDBAH: u32 = 0x2804;
const RDLEN: u32 = 0x2808;
const RDH: u32 = 0x2810;
const RDT: u32 = 0x2818;
const RDTR: u32 = 0x2820;
const RXDCTL: u32 = 0x2828;
const RADV: u32 = 0x282c;
const TDBAL: u32 = 0x3800;
const TDBAH: u32 = 0x3804;
const TDLEN: u32 = 0x3808;
const TDH: u32 = 0x3810;
const TDT: u32 = 0x3818;
const TIDV: u32 = 0x3820;
const TXDCTL: u32 = 0x3828;
const TADV: u32 = 0x382c;
const TARC: u32 = 0x3840;
const QUEUE_STRIDE: u32 = 0x100;
const ENABLE: u32 = 2;
const LINK_UP: u32 = 2;
const LINK_EVENT: u32 = 1 << 2;
const FULL_DUPLEX: u32 = 1;
const SPEED_MASK: u32 = 3 << 6;
const LINK_SUMMARY: u32 = LINK_UP | FULL_DUPLEX | SPEED_MASK;
const SLU: u32 = 1 << 6;
const FORCE_LINK: u32 = (1 << 11) | (1 << 12);
const PAUSE: u32 = (1 << 27) | (1 << 28);
const VLAN: u32 = 1 << 30;
const DRIVER_LOADED: u32 = 1 << 28;
const CTRL_EXT_POLICY: u32 = (1 << 17) | (1 << 20) | (1 << 22) | DRIVER_LOADED;
const NFS_DISABLE: u32 = (1 << 6) | (1 << 7);
const EXTENDED_RX: u32 = 1 << 15;
const VALID_ADDRESS: u32 = 1 << 31;
const TX_POLICY: u32 = (1 << 24) | (1 << 22) | (1 << 16) | 31;
const RX_POLICY: u32 = (1 << 16) | (4 << 8) | 8;
const RECEIVE_POLICY: u32 = ENABLE | (1 << 4) | (1 << 15);
const TRANSMIT_POLICY: u32 = ENABLE | super::PAD_SHORT | (15 << 4) | (63 << 12) | (1 << 24);
const RX_DONE: u32 = 1;
const RX_EOP: u32 = 2;
const RX_ERRORS: u32 = 0x9700_0000;
const TX_DONE: u8 = 1;
const TX_COMMAND: u32 = (1 << 24) | (1 << 25) | (1 << 27);
const FCS_BYTES: usize = 4;
const PHY_LINK_READY: u16 = (1 << 2) | (1 << 5);
const ADVERTISEMENT_SPEED: u16 = 0x1e0;
const ADVERTISEMENT_PAUSE: u16 = 0xc00;
const GIGABIT_ADVERTISEMENT: u16 = 0x300;
const GIGABIT_FULL: u16 = 0x200;
const MASTER_OVERRIDE: u16 = 0x1000;
const PHY_AN_RESTART: u16 = (1 << 12) | (1 << 9);
const PHY_QUIET: u16 = (1 << 14) | (1 << 11) | (1 << 10);
const RX_CONFIG: u16 = 0x3412;
const LPI_PLL_SHUT: u16 = 0x4412;
const EEE_PCS_STATUS: u16 = 0x182e;
const EEE_ADVERTISEMENT: u16 = 0x040e;
const LPI_ENABLE: u16 = 0x6000;

/// The controller owner and DMA storage remain held after an
/// uncertain doorbell, timeout, removal or control failure. A fresh object is
/// not a retry mechanism; only an external real reset/recovery grants one.
pub struct Ethernet<R: Registers + nvm::Registers, D: Delay + Clone> {
    controller: Controller<R, D>,
    receive: Ring,
    transmit: Ring,
    next_rx: usize,
    next_tx: usize,
    clean_tx: usize,
    discard_rx: bool,
    link: Option<u32>,
    stopped: bool,
    failure: Option<Error>,
}
impl<R: Registers + nvm::Registers, D: Delay + Clone> Ethernet<R, D> {
    /// # Safety
    /// `dma` grants coherent boot-lifetime CPU/DMA RAM reachable by the
    /// same PCI requester as `controller`, using the allocator's real
    /// addressability/identity mapping contract. PCI memory decoding and bus
    /// mastering are enabled by the exclusive platform owner, with firmware,
    /// SMM and native interrupts excluded. No failed epoch may be reconstructed
    /// without reset/recovery. This constructor requires those grants;
    /// it neither synthesizes PCI ownership nor enables a broad probe match.
    pub unsafe fn new(
        mut controller: Controller<R, D>,
        dma: &mut dyn DmaAllocator,
    ) -> Result<Self, Error> {
        if let Some(error) = controller.minimum_failure() {
            return Err(super::Fault::Minimum(error).into());
        }
        if controller.is_poisoned() {
            return Err(Error::Poisoned);
        }
        controller.phy.check_minimum()?;
        let receive = Ring::new(dma).map_err(Error::Dma)?;
        let transmit = Ring::new(dma).map_err(Error::Dma)?;
        let mut this = Self {
            controller,
            receive,
            transmit,
            next_rx: 0,
            next_tx: 0,
            clean_tx: 0,
            discard_rx: false,
            link: None,
            stopped: false,
            failure: None,
        };
        if let Err(error) = this.configure() {
            this.poison();
            return Err(error);
        }
        Ok(this)
    }
    fn poison(&mut self) {
        self.controller.poisoned = true;
        self.controller.phy.poisoned = true;
    }
    pub fn failure(&self) -> Option<&Error> {
        self.failure.as_ref()
    }
    fn check(&mut self) -> Result<(), Error> {
        if let Some(error) = &self.failure
            && error.minimum_cause().is_some()
        {
            return Err(error.clone());
        }
        if let Some(error) = self.controller.minimum_failure() {
            return Err(super::Fault::Minimum(error).into());
        }
        if self.controller.is_poisoned() {
            Err(Error::Poisoned)
        } else if self.stopped {
            Err(Error::Stopped)
        } else {
            self.controller.phy.check_minimum()?;
            Ok(())
        }
    }
    fn modify_mac(&mut self, at: u32, clear: u32, set: u32) -> Result<(), Error> {
        let value = self.controller.read(at)?;
        self.controller.write(at, (value & !clear) | set)
    }
    fn kmrn(&mut self, index: u8, write: Option<u16>) -> Result<u16, Error> {
        Ok(self.controller.phy.transaction(|p| {
            p.write_mac(
                KMRN,
                (u32::from(index) << 16) | write.map_or(1 << 21, u32::from),
            )?;
            p.read_mac(super::STATUS)?;
            p.wait_minimum(2)?;
            if write.is_none() {
                Ok(p.read_mac(KMRN)? as u16)
            } else {
                Ok(0)
            }
        })?)
    }
    fn standard_mtu(&mut self) -> Result<(), Error> {
        // Unchanged source's disabled-jumbo branch, with every formerly ignored
        // read failure made terminal. Do not restore RX on an uncertain write.
        let reg = super::super::Register::new(769, 20)?;
        let old = self.controller.phy.read(reg)?;
        self.controller.phy.write(reg, old | (1 << 14))?;
        self.modify_mac(FFLT_DBG, 0xf << 14, 0)?;
        self.modify_mac(super::RCTL, 1 << 26, 0)?;
        let value = self.kmrn(1, None)?;
        self.kmrn(1, Some(value & !1))?;
        let value = self.kmrn(16, None)?;
        self.kmrn(16, Some((value & !(0xf << 8)) | (0xb << 8)))?;
        self.controller.modify(769, 23, 0x7f << 5, 0)?;
        self.controller.modify(769, 16, 0, 1 << 13)?;
        self.controller.modify(776, 20, 0x3ff << 2, 8 << 2)?;
        self.controller
            .phy
            .write(super::super::Register::new(776, 23)?, 0x7e00)?;
        self.controller.modify(770, 17, 1 << 10, 0)?;
        self.controller.phy.write(reg, old & !(1 << 14))?;
        Ok(())
    }
    fn configure(&mut self) -> Result<(), Error> {
        self.controller.qualify()?;
        self.controller.write(0x5800, 0)?; // WUC: no wake ownership/profile
        self.controller.write(0x38, 0x8100)?; // VET: ordinary 802.1Q recognition
        // Real FIFO allocation must fit an ordinary frame (including metadata
        // in TX and received FCS in RX); never assume a reset/NVM buffer size.
        let pba = self.controller.read(PBA)?;
        if (pba & 0xffff) < 2 || (pba >> 16) < 2 {
            return Err(Error::Configuration);
        }
        self.modify_mac(CTRL_EXT, 0, CTRL_EXT_POLICY)?;
        for queue in 0..2 {
            self.modify_mac(TXDCTL + queue * QUEUE_STRIDE, 0x003f_003f, TX_POLICY)?;
        }
        self.modify_mac(TARC, 0, (1 << 23) | (1 << 24) | (1 << 26) | (1 << 27))?;
        let mulr = self.controller.read(super::TCTL)? & (1 << 28) != 0;
        self.modify_mac(
            TARC + QUEUE_STRIDE,
            1 << 28,
            (1 << 24) | (1 << 26) | (1 << 30) | if mulr { 0 } else { 1 << 28 },
        )?;
        if self.controller.is_spt_mac() {
            self.modify_mac(spt::IOSFPC, 0, spt::CORRUPTION_THRESHOLD)?; // SPT IOSFPC data-corruption erratum
            self.modify_mac(TARC, spt::TARC_REQUEST_MASK, spt::TARC_TWO_REQUESTS)?;
        }
        if self.controller.is_i219() {
            // Masked legacy/polled profile, without MSI/MSI-X clock requests.
            self.modify_mac(spt::FEXTNVM7, 0, spt::SIDE_CLOCK_UNGATE)?;
            self.modify_mac(spt::FEXTNVM9, 0, spt::IOSF_CLOCK_GATING_DISABLE)?;
        }
        self.modify_mac(GCR, 0x3f, 0)?;
        self.modify_mac(RFCTL, 0, NFS_DISABLE | EXTENDED_RX)?;
        if self.controller.profile.is_lpt_or_newer() {
            self.modify_mac(PBECCSTS, 0, ECC_ENABLE)?;
            self.modify_mac(super::CTRL, 0, MEMORY_ERROR_HANDLING)?;
        }
        self.controller.mtp_init_k1()?;
        self.station_filters()?;
        for index in 0..MTA_REGISTERS {
            self.controller.write(MTA + index * 4, 0)?;
        }
        self.standard_mtu()?;
        self.setup_link()?;
        self.i219_clock_policy()?;
        self.controller.write(RXCSUM, 0)?;
        self.controller.write(TIPG, 8 | (8 << 10) | (6 << 20))?;
        for (low, high, length, head, tail, address) in [
            (
                RDBAL,
                RDBAH,
                RDLEN,
                RDH,
                RDT,
                self.receive.descriptors.bus_address(),
            ),
            (
                TDBAL,
                TDBAH,
                TDLEN,
                TDH,
                TDT,
                self.transmit.descriptors.bus_address(),
            ),
        ] {
            self.controller.write(low, address as u32)?;
            self.controller.write(high, (address >> 32) as u32)?;
            self.controller
                .write(length, (DESCRIPTORS * DESCRIPTOR) as u32)?;
            self.controller.write(head, 0)?;
            self.controller.write(tail, 0)?;
        }
        self.controller.write(RXDCTL, RX_POLICY)?;
        self.controller.write(RXDCTL + QUEUE_STRIDE, RX_POLICY)?;
        let tx_policy = self.controller.read(TXDCTL)?;
        self.controller.write(TXDCTL + QUEUE_STRIDE, tx_policy)?;
        for at in [RDTR, RADV, TIDV, TADV] {
            self.controller.write(at, 0)?;
        }
        // Pause is explicitly disabled, including stale MAC control and
        // thresholds. PHY advertisement uses the same no-pause policy.
        for at in [0x2160, 0x2168] {
            self.controller.write(at, 0)?;
        }
        self.controller.write(0x170, 0xffff)?;
        self.controller.write(0x5f40, 0xffff)?;
        self.controller
            .phy
            .write(super::super::Register::new(769, 27)?, 0xffff)?;
        self.controller.write(super::IMC, u32::MAX)?;
        for index in 0..DESCRIPTORS {
            self.offer(index);
        }
        fence(Ordering::SeqCst);
        self.controller.write(RDT, (DESCRIPTORS - 1) as u32)?;
        self.modify_mac(
            super::CTRL,
            super::MASTER_DISABLE | FORCE_LINK | PAUSE | VLAN,
            SLU,
        )?;
        self.controller.write(super::RCTL, RECEIVE_POLICY)?;
        self.controller.write(super::TCTL, TRANSMIT_POLICY)?;
        self.controller.read(super::STATUS)?;
        self.update_link()?;
        Ok(())
    }
    fn i219_clock_policy(&mut self) -> Result<(), Error> {
        if self
            .controller
            .i219_family()
            .is_some_and(i219::Family::dma_clock_workaround)
            || self.controller.mtp_family().is_some()
        {
            self.modify_mac(FFLT_DBG, 0, i219::DONT_GATE_WAKE_DMA_CLOCK)?;
        }
        Ok(())
    }
    fn station_filters(&mut self) -> Result<(), Error> {
        let a = self.controller.station.0;
        let low = u32::from_le_bytes([a[0], a[1], a[2], a[3]]);
        let high = u32::from(u16::from_le_bytes([a[4], a[5]])) | VALID_ADDRESS;
        self.controller.write(RAL, low)?;
        self.controller.read(super::STATUS)?;
        self.controller.write(RAH, high)?;
        self.controller.read(super::STATUS)?;
        // PCH2 preserves RAR1..6 and uses SHRA0..3 at5438. LPT's eleven SHRA
        // slots begin at5408 and have FWSM ownership limits instead.
        let is_lpt = self.controller.profile.is_lpt_or_newer();
        let (low, high, count) = if is_lpt {
            (
                SHRAL_LPT,
                SHRAH_LPT,
                lpt::shared_count(self.controller.read(super::super::FWSM)?),
            )
        } else {
            (SHRAL, SHRAH, 4)
        };
        for index in 0..count {
            self.controller.phy.transaction(|p| {
                if is_lpt && lpt::shared_count(p.read_mac(super::super::FWSM)?) <= index {
                    return Err(super::super::Fault::FirmwareOwned);
                }
                p.write_mac(low + index * 8, 0)?;
                p.read_mac(super::STATUS)?;
                p.write_mac(high + index * 8, 0)?;
                p.read_mac(super::STATUS)?;
                Ok(())
            })?;
            if self.controller.read(low + index * 8)? != 0
                || self.controller.read(high + index * 8)? != 0
            {
                return Err(Error::Configuration);
            }
        }
        Ok(())
    }
    fn setup_link(&mut self) -> Result<(), Error> {
        self.modify_mac(super::CTRL, FORCE_LINK | PAUSE | VLAN, SLU)?;
        if !self.controller.profile.is_lpt_or_newer() {
            self.kmrn(4, Some(0xffff))?;
            let value = self.kmrn(9, None)?;
            self.kmrn(9, Some(value | 0x3f))?;
        }
        self.controller.modify(0, 22, 0, (1 << 15) | (3 << 10))?;
        self.controller.modify(0, 18, 0x600, 0x400)?;
        self.controller.modify(0, 9, MASTER_OVERRIDE, 0)?;
        self.controller.modify(
            0,
            4,
            ADVERTISEMENT_SPEED | ADVERTISEMENT_PAUSE,
            ADVERTISEMENT_SPEED,
        )?;
        self.controller
            .modify(0, 9, GIGABIT_ADVERTISEMENT, GIGABIT_FULL)?;
        self.disable_eee()?;
        self.controller.modify(0, 0, PHY_QUIET, PHY_AN_RESTART)?;
        Ok(())
    }
    fn disable_eee(&mut self) -> Result<(), Error> {
        let lpt = self.controller.profile.is_lpt_or_newer();
        self.controller.phy.transaction(|p| {
            Controller::<R, D>::emi(
                p,
                if lpt {
                    lpt::EEE_ADVERTISEMENT
                } else {
                    EEE_ADVERTISEMENT
                },
                0,
            )?;
            let control = super::super::Register {
                page: 772,
                index: 20,
            };
            let value = p.access(control, None)? & !LPI_ENABLE;
            if !lpt {
                p.access(
                    super::super::Register { page: 0, index: 16 },
                    Some(LPI_PLL_SHUT),
                )?;
                let pll = p.access(super::super::Register { page: 0, index: 17 }, None)?;
                Controller::<R, D>::emi(p, LPI_PLL_SHUT, pll & !4)?;
            }
            p.access(
                super::super::Register { page: 0, index: 16 },
                Some(if lpt {
                    lpt::EEE_PCS_STATUS
                } else {
                    EEE_PCS_STATUS
                }),
            )?;
            p.access(super::super::Register { page: 0, index: 17 }, None)?;
            p.access(control, Some(value))?;
            Ok(())
        })?;
        Ok(())
    }
    fn update_link(&mut self) -> Result<bool, Error> {
        self.check()?;
        let status = self.controller.read(super::STATUS)? & LINK_SUMMARY;
        // IMC masks interrupts, but ICR still latches link-status changes.
        // Consume that event so same-speed reconnects cannot bypass
        // PHY/K1 workarounds because the MAC summary is unchanged.
        let event = self.controller.read(super::ICR)?;
        if self.link == Some(status) && event & LINK_EVENT == 0 {
            return Ok(status & LINK_UP != 0);
        }
        let mii = super::super::Register::new(0, 1)?;
        self.controller.phy.read(mii)?; // clear latched-low observation
        let phy = self.controller.phy.read(mii)?;
        if status & LINK_UP == 0 || phy & PHY_LINK_READY != PHY_LINK_READY {
            self.controller
                .lpt_link_common(if status & LINK_UP != 0 && phy & 4 != 0 {
                    Some(status)
                } else {
                    None
                })?;
            self.link = if status & LINK_UP == 0 {
                Some(status)
            } else {
                None
            };
            return Ok(false);
        }
        let half_ten = status & (FULL_DUPLEX | SPEED_MASK) == 0;
        self.modify_mac(
            TIPG,
            0x3ff,
            if half_ten {
                0xff
            } else if self.controller.is_i219()
                && status & FULL_DUPLEX != 0
                && status & (1 << 7) == 0
            {
                0xc
            } else {
                8
            },
        )?;
        let lpt = self.controller.profile.is_lpt_or_newer();
        let mtp = self.controller.mtp_family().is_some();
        self.controller.phy.transaction(|p| {
            Controller::<R, D>::emi(
                p,
                if lpt { lpt::RX_CONFIG } else { RX_CONFIG },
                u16::from(!half_ten),
            )?;
            if lpt {
                let gate = lpt::PLL_CLOCK_GATE;
                let word = p.access(gate, None)?;
                p.access(
                    gate,
                    Some(
                        (word & !lpt::PLL_CLOCK_GATE_MASK)
                            | if status & (1 << 7) != 0 {
                                if mtp {
                                    super::mtp::GIGABIT_PLL_CLOCK_GATE
                                } else {
                                    lpt::PLL_CLOCK_GATE_GIGABIT
                                }
                            } else {
                                lpt::PLL_CLOCK_GATE_SLOW
                            },
                    ),
                )?;
                if status & (1 << 7) != 0 {
                    let pm = super::super::Register {
                        page: 770,
                        index: 17,
                    };
                    let word = p.access(pm, None)?;
                    p.access(pm, Some(word | lpt::K1_CLOCK_REQUEST))?;
                }
            }
            Ok(())
        })?;
        if self.controller.is_i219() {
            let gap = super::super::Register {
                page: 776,
                index: 20,
            };
            if status & (1 << 7) != 0 {
                let word = self.controller.phy.read(gap)?;
                if ((word >> 2) & 0x3ff) < 0x18 {
                    self.controller
                        .phy
                        .write(gap, (word & !(0x3ff << 2)) | (0x18 << 2))?;
                }
            } else {
                self.controller.phy.write(gap, 0xc023)?;
            }
        }
        if lpt {
            self.controller.lpt_link_common(Some(status))?;
            self.disable_eee()?;
            self.modify_mac(super::TCTL, 0x3ff << 12, 63 << 12)?;
            self.modify_mac(super::CTRL, PAUSE, 0)?;
            self.link = Some(status);
            return Ok(true);
        }
        // 82579 K1 and half-duplex preamble workarounds apply on every observed
        // speed/duplex transition, including a cable reconnection while polling.
        let hv = self
            .controller
            .phy
            .read(super::super::Register::new(0, 26)?)?;
        if hv & 0x1040 != 0x1040 || u32::from((hv & 0x300) >> 8) != (status & SPEED_MASK) >> 6 {
            self.link = None;
            return Ok(false);
        }
        if hv & 0x300 != 0 {
            self.controller.modify(770, 17, 1 << 14, 0)?;
        } else {
            self.modify_mac(FEXTNVM4, 7, 3)?;
        }
        self.controller.modify(
            770,
            16,
            0x7000,
            if status & FULL_DUPLEX == 0 { 0x1000 } else { 0 },
        )?;
        self.modify_mac(super::TCTL, 0x3ff << 12, 63 << 12)?;
        self.modify_mac(super::CTRL, PAUSE, 0)?;
        self.link = Some(status);
        Ok(true)
    }
    fn offer(&mut self, index: usize) {
        let base = index * DESCRIPTOR;
        self.receive
            .descriptors
            .write(base, self.receive.address(index));
        self.receive.descriptors.write(base + 8, 0u64);
    }
    fn reclaim(&mut self) -> bool {
        while self.clean_tx != self.next_tx {
            if self
                .transmit
                .descriptors
                .read::<u8>(self.clean_tx * DESCRIPTOR + 12)
                & TX_DONE
                == 0
            {
                break;
            }
            fence(Ordering::Acquire);
            self.clean_tx = (self.clean_tx + 1) % DESCRIPTORS;
        }
        (self.next_tx + 1) % DESCRIPTORS != self.clean_tx
    }
    fn spt_flush_descriptors(&mut self) -> Result<(), Error> {
        if !self.controller.is_i219() {
            return Ok(());
        }
        self.modify_mac(spt::FEXTNVM11, 0, spt::DISABLE_MULR_FIX)?;
        let length = self.controller.read(TDLEN)?;
        let status = self.controller.spt_config_status()?;
        if status & spt::FLUSH_REQUIRED == 0 || length == 0 {
            return Ok(());
        }
        let control = self.controller.read(super::TCTL)?;
        self.controller.write(super::TCTL, control | ENABLE)?;
        let tail = self.controller.read(TDT)?;
        if tail as usize != self.next_tx {
            return Err(Error::Configuration);
        }
        // Source's non-EOP flush descriptor reads512 bytes from the owned
        // descriptor ring itself. The single-empty-slot invariant reserves
        // this descriptor even while earlier packets await retirement.
        let index = self.next_tx;
        let base = index * DESCRIPTOR;
        self.transmit
            .descriptors
            .write(base, self.transmit.descriptors.bus_address());
        self.transmit
            .descriptors
            .write(base + 8, spt::FLUSH_BYTES | (1 << 25));
        self.transmit.descriptors.write(base + 12, 0u32);
        fence(Ordering::SeqCst);
        self.next_tx = (index + 1) % DESCRIPTORS;
        self.controller.write(TDT, self.next_tx as u32)?;
        self.controller.phy.wait_minimum(200)?;
        if self.controller.spt_config_status()? & spt::FLUSH_REQUIRED != 0 {
            let control = self.controller.read(super::RCTL)?;
            self.controller.write(super::RCTL, control & !ENABLE)?;
            self.controller.read(super::STATUS)?;
            self.controller.phy.wait_minimum(100)?;
            let value = self.controller.read(RXDCTL)?;
            self.controller
                .write(RXDCTL, (value & 0xffff_c000) | 0x1f | (1 << 8) | (1 << 24))?;
            self.controller.write(super::RCTL, control | ENABLE)?;
            self.controller.read(super::STATUS)?;
            self.controller.phy.wait_minimum(100)?;
            self.controller.write(super::RCTL, control & !ENABLE)?;
            // Source only rechecks after TX. An unresolved RX flush is not
            // permission to reset/power down or release the owner.
            if self.controller.spt_config_status()? & spt::FLUSH_REQUIRED != 0 {
                return Err(Error::DescriptorFlushRequired);
            }
        }
        Ok(())
    }
    /// Stop both packet engines and observe PCIe master drain before releasing
    /// native resources. Any failure permanently keeps the owner.
    pub fn shutdown(&mut self) -> Result<(), Error> {
        if self.stopped {
            return Ok(());
        }
        self.check()?;
        let result = (|| {
            self.controller.write(super::IMC, u32::MAX)?;
            self.spt_flush_descriptors()?;
            self.modify_mac(super::RCTL, ENABLE, 0)?;
            self.modify_mac(super::TCTL, ENABLE, 0)?;
            self.controller.read(super::STATUS)?;
            self.controller.phy.wait_minimum(10_000)?;
            self.modify_mac(super::CTRL, 0, super::MASTER_DISABLE)?;
            for _ in 0..800 {
                if self.controller.read(super::STATUS)? & super::MASTER_ENABLED == 0 {
                    self.controller.lpt_platform_pm(None)?;
                    self.modify_mac(CTRL_EXT, DRIVER_LOADED, 0)?;
                    if self.controller.read(CTRL_EXT)? & DRIVER_LOADED != 0 {
                        return Err(Error::Configuration);
                    }
                    return Ok(());
                }
                self.controller.phy.wait_minimum(100)?;
            }
            Err(Error::MasterTimeout)
        })();
        match result {
            Ok(()) => {
                self.stopped = true;
                Ok(())
            }
            Err(e) => {
                self.failure.get_or_insert_with(|| e.clone());
                self.poison();
                Err(e)
            }
        }
    }
    fn error(&mut self, error: Error) -> LinkError {
        if error == Error::Stopped {
            return LinkError::Adapter("PCH Ethernet is stopped".into());
        }
        self.failure.get_or_insert_with(|| error.clone());
        self.poison();
        if let Some(cause) = error.minimum_cause() {
            return LinkError::Pci(crate::block::BlockError::PciPower(cause));
        }
        LinkError::Adapter(format!("PCH: {error}"))
    }
}
impl<R: Registers + nvm::Registers, D: Delay + Clone> Drop for Ethernet<R, D> {
    fn drop(&mut self) {
        if !self.stopped && !self.controller.is_poisoned() {
            let _ = self.shutdown();
        }
        // A poisoned PHY's Drop keeps the shared function owner.
        // DMA RAM is independently boot-lifetime, including after uncertainty.
    }
}
impl<R: Registers + nvm::Registers, D: Delay + Clone> Link for Ethernet<R, D> {
    fn mac(&self) -> Mac {
        self.controller.station
    }
    fn carrier(&mut self) -> bool {
        match self.update_link() {
            Ok(up) => up,
            Err(Error::Stopped) => false,
            Err(e) => {
                self.failure.get_or_insert(e);
                self.poison();
                false
            }
        }
    }
    fn transmit(&mut self, frame: &[u8]) -> Result<(), LinkError> {
        if !(14..=MAX_FRAME).contains(&frame.len()) {
            return Err(LinkError::Invalid("frame length is outside 14..=1514"));
        }
        self.check().map_err(|e| self.error(e))?;
        if !self.update_link().map_err(|e| self.error(e))? {
            return Err(LinkError::Adapter("PCH Ethernet link is down".into()));
        }
        let mut space = self.reclaim();
        for _ in 0..1000 {
            if space {
                break;
            }
            self.controller
                .phy
                .wait_minimum(100)
                .map_err(|fault| self.error(Error::from(fault)))?;
            space = self.reclaim();
        }
        if !space {
            return Err(self.error(Error::QueueStalled));
        }
        let index = self.next_tx;
        let length = frame.len().max(MIN_FRAME);
        write_transmit_frame(self.transmit.buffer(index), frame, length);
        let base = index * DESCRIPTOR;
        self.transmit
            .descriptors
            .write(base, self.transmit.address(index));
        self.transmit
            .descriptors
            .write(base + 8, length as u32 | TX_COMMAND);
        self.transmit.descriptors.write(base + 12, 0u32);
        fence(Ordering::SeqCst);
        let tail = (index + 1) % DESCRIPTORS;
        if let Err(error) = self.controller.write(TDT, tail as u32) {
            return Err(self.error(error));
        }
        self.next_tx = tail;
        Ok(())
    }
    fn receive(&mut self) -> Result<Option<Vec<u8>>, LinkError> {
        let mut frame = Vec::new();
        Ok(self.receive_reusing(&mut frame)?.then_some(frame))
    }
    /// The stack's receive path: the frame is copied into its reused
    /// buffer, so a received frame allocates nothing. On an error the
    /// buffer is unchanged.
    fn receive_reusing(&mut self, frame: &mut Vec<u8>) -> Result<bool, LinkError> {
        self.check().map_err(|e| self.error(e))?;
        self.update_link().map_err(|e| self.error(e))?;
        for _ in 0..DESCRIPTORS {
            let index = self.next_rx;
            let base = index * DESCRIPTOR;
            let status = self.receive.descriptors.read::<u32>(base + 8);
            if status & RX_DONE == 0 {
                if let Some(tail) = self.receive.idle() {
                    fence(Ordering::SeqCst);
                    if let Err(error) = self.controller.write(RDT, tail) {
                        return Err(self.error(error));
                    }
                }
                return Ok(false);
            }
            fence(Ordering::Acquire);
            let length = usize::from(self.receive.descriptors.read::<u16>(base + 12));
            let eop = status & RX_EOP != 0;
            let good = !self.discard_rx
                && eop
                && status & RX_ERRORS == 0
                && (14 + FCS_BYTES..=MAX_FRAME + FCS_BYTES).contains(&length);
            if good {
                let length = length - FCS_BYTES;
                frame
                    .try_reserve_exact(length.saturating_sub(frame.len()))
                    .map_err(|_| {
                        LinkError::Adapter("PCH Ethernet receive allocation failed".into())
                    })?;
                frame.clear();
                frame.extend_from_slice(&self.receive.buffer(index)[..length]);
            }
            self.discard_rx = !eop;
            self.offer(index);
            if let Some(tail) = self.receive.offered(index) {
                fence(Ordering::SeqCst);
                if let Err(error) = self.controller.write(RDT, tail) {
                    return Err(self.error(error));
                }
            }
            self.next_rx = (index + 1) % DESCRIPTORS;
            if good {
                return Ok(true);
            }
        }
        if let Some(tail) = self.receive.idle() {
            fence(Ordering::SeqCst);
            if let Err(error) = self.controller.write(RDT, tail) {
                return Err(self.error(error));
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
pub(super) mod tests;
