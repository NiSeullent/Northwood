// SPDX-License-Identifier: GPL-2.0-or-later
// Sequences follow Linux `atl1c_hw.c` and `atl1c_main.c`, Copyright (c) 2008
// Atheros Communications Inc., Copyright (c) 2007 Atheros Corporation.
// Copyright (c) 2026 Stela contributors
//! The MAC, PHY and PCIe sequences of the atl1c family in Linux's order and
//! with its values, over a register window and the shared timing owner.
use super::registers::*;
use crate::{alx::Registers, block::BlockError, timing::Timing};

/// `atl1c_nic_type`, chosen by PCI device ID as `atl1c_get_mac_type` does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// AR8131 (L1C, gigabit).
    L1c,
    /// AR8132 (L2C, fast Ethernet).
    L2c,
    /// AR8152 v1.1 (L2C_B).
    L2cB,
    /// AR8152 v2.0 (L2C_B2).
    L2cB2,
    /// AR8151 v1.0 (L1D, gigabit).
    L1d,
    /// AR8151 v2.0 (L1D_2_0, gigabit).
    L1d2,
}

impl Kind {
    pub(super) fn from_device(device: u16) -> Option<Self> {
        Some(match device {
            0x1063 => Self::L1c,
            0x1062 => Self::L2c,
            0x2060 => Self::L2cB,
            0x2062 => Self::L2cB2,
            0x1073 => Self::L1d,
            0x1083 => Self::L1d2,
            _ => return None,
        })
    }
    /// `atl1c_setup_mac_funcs`: `ATL1C_LINK_CAP_1000M`.
    pub(super) fn gigabit(self) -> bool {
        matches!(self, Self::L1c | Self::L1d | Self::L1d2)
    }
    fn first_generation(self) -> bool {
        matches!(self, Self::L1c | Self::L2c)
    }
    fn second_revision(self) -> bool {
        matches!(self, Self::L2cB2 | Self::L1d2)
    }
    fn l2cb_family(self) -> bool {
        matches!(self, Self::L2cB | Self::L2cB2)
    }
}

/// The PCI identity `atl1c_sw_init` and `atl1c_patch_assign` read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Chip {
    pub(super) kind: Kind,
    pub(super) device: u16,
    pub(super) revision: u8,
    /// `msi_lnkpatch` from the platform table.
    pub(super) link_patch: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Speed {
    Ten,
    Hundred,
    Thousand,
}

/// The negotiated link, `None` while down.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct LinkState {
    pub(super) speed: Option<Speed>,
    pub(super) full_duplex: bool,
}

pub(super) const TX_RING: usize = 256;
pub(super) const RX_RING: usize = 256;
/// `AT_RX_BUF_SIZE`: a full frame, one VLAN tag and the FCS.
pub(super) const RX_BUFFER: u32 = 1514 + 4 + 4;
/// `atl1c_sw_init`'s timers: interrupt clear 100 ms, statistics 400 ms,
/// and the moderation of both directions.
const ICT: u32 = 50_000;
const SMB_TIMER: u32 = 200_000;
const RX_IMT: u32 = 200;
const TX_IMT: u32 = 1000;
const RFD_BURST: u32 = 8;
const PREAMBLE: u32 = 7;
/// The standard MTU with Ethernet header, one VLAN tag and FCS.
const FRAME: u32 = 1500 + 14 + 4 + 4;
/// `AT_TWSI_EEPROM_TIMEOUT`: EEPROM load polls, ten milliseconds apart.
const TWSI_EEPROM_TIMEOUT: u32 = 100;

pub(super) struct Hardware<R> {
    pub(super) registers: R,
    pub(super) timer: Timing,
    pub(super) chip: Chip,
    pub(super) link: LinkState,
    /// `hw->hibernate`: true until a link comes up.
    pub(super) hibernate: bool,
    /// `hw->dmar_block` after `atl1c_configure_tx`'s correction.
    pub(super) dmar_block: u32,
}

impl<R: Registers> Hardware<R> {
    pub(super) fn new(registers: R, timer: Timing, chip: Chip, dmar_block: u32) -> Self {
        Self {
            registers,
            timer,
            chip,
            link: LinkState::default(),
            hibernate: true,
            dmar_block,
        }
    }

    pub(super) fn read(&self, offset: usize) -> u32 {
        self.registers.read32(offset)
    }
    fn write(&self, offset: usize, value: u32) {
        self.registers.write32(offset, value)
    }
    fn delay(&self, us: u64) -> Result<(), BlockError> {
        self.timer.delay(us)
    }

    fn wait_mdio_idle(&self) -> Result<u32, BlockError> {
        for _ in 0..MDIO_MAX_AC_TO {
            let value = self.read(MDIO_CTRL);
            if value & (MDIO_CTRL_BUSY | MDIO_CTRL_START) == 0 {
                return Ok(value);
            }
            self.delay(10)?;
        }
        Err(BlockError::Timeout)
    }

    /// `atl1c_read_phy_core` and `atl1c_write_phy_core`: only L2C_B2 and
    /// L1D_2 use the slow clock, and only while hibernating.
    fn phy_core(
        &self,
        extended: Option<u8>,
        register: u16,
        write: Option<u16>,
    ) -> Result<u16, BlockError> {
        let clock = if self.chip.kind.second_revision() && self.hibernate {
            MDIO_CTRL_CLK_25_128
        } else {
            MDIO_CTRL_CLK_25_4
        };
        let mut value =
            MDIO_CTRL_SPRES_PRMBL | (clock << MDIO_CTRL_CLK_SEL_SHIFT) | MDIO_CTRL_START;
        match extended {
            Some(device) => {
                self.write(
                    MDIO_EXTN,
                    (u32::from(device) << MDIO_EXTN_DEVAD_SHIFT) | u32::from(register),
                );
                value |= MDIO_CTRL_MODE_EXT;
            }
            None => value |= u32::from(register) << MDIO_CTRL_REG_SHIFT,
        }
        value |= match write {
            Some(data) => u32::from(data),
            None => MDIO_CTRL_OP_READ,
        };
        self.write(MDIO_CTRL, value);
        let done = self.wait_mdio_idle()?;
        Ok(if write.is_some() { 0 } else { done as u16 })
    }

    pub(super) fn read_phy(&self, register: u16) -> Result<u16, BlockError> {
        self.phy_core(None, register, None)
    }
    pub(super) fn write_phy(&self, register: u16, value: u16) -> Result<(), BlockError> {
        self.phy_core(None, register, Some(value)).map(|_| ())
    }
    fn read_ext(&self, device: u8, register: u16) -> Result<u16, BlockError> {
        self.phy_core(Some(device), register, None)
    }
    fn write_ext(&self, device: u8, register: u16, value: u16) -> Result<(), BlockError> {
        self.phy_core(Some(device), register, Some(value))
            .map(|_| ())
    }
    fn read_dbg(&self, register: u16) -> Result<u16, BlockError> {
        self.write_phy(MII_DBG_ADDR, register)?;
        self.read_phy(MII_DBG_DATA)
    }
    fn write_dbg(&self, register: u16, value: u16) -> Result<(), BlockError> {
        self.write_phy(MII_DBG_ADDR, register)?;
        self.write_phy(MII_DBG_DATA, value)
    }

    /// `atl1c_wait_until_idle`: the busy bits still set, or zero.
    fn wait_until_idle(&self, bits: u32) -> Result<u32, BlockError> {
        let mut value = 0;
        for _ in 0..MAX_IDLE_DELAY {
            value = self.read(IDLE_STATUS);
            if value & bits == 0 {
                return Ok(0);
            }
            self.delay(1000)?;
        }
        Ok(value)
    }

    /// `atl1c_pcie_patch`.
    fn pcie_patch(&self) {
        let master = self.read(MASTER_CTRL) & !MASTER_CTRL_CLK_SEL_DIS;
        self.write(MASTER_CTRL, master);
        if self.chip.kind.first_generation() {
            let value = self.read(PCIE_PHYMISC);
            self.write(PCIE_PHYMISC, value | PCIE_PHYMISC_FORCE_RCV_DET);
        } else if master & MASTER_CTRL_WAKEN_25M == 0 {
            self.write(MASTER_CTRL, master | MASTER_CTRL_WAKEN_25M);
        }
        if self.chip.kind == Kind::L2cB && self.chip.revision == L2CB_V10 {
            let mut value = self.read(PCIE_PHYMISC2);
            value = set_field(value, PCIE_PHYMISC2_CDR_BW, L2CB1_PCIE_PHYMISC2_CDR_BW);
            value = set_field(value, PCIE_PHYMISC2_L0S_TH, L2CB1_PCIE_PHYMISC2_L0S_TH);
            self.write(PCIE_PHYMISC2, value);
            let value = self.read(LINK_CTRL);
            self.write(LINK_CTRL, value | LINK_CTRL_EXT_SYNC);
        }
        if matches!(self.chip.kind, Kind::L2cB | Kind::L1d) {
            let value = self.read(PM_CTRL);
            self.write(PM_CTRL, value | PM_CTRL_L0S_BUFSRX_EN);
            let value = self.read(DMA_DBG);
            self.write(DMA_DBG, value & !DMA_DBG_VENDOR_MSG);
        }
    }

    /// `atl1c_reset_pcie`'s register part with `ATL1C_PCIE_L0S_L1_DISABLE`:
    /// WoL status cleared, LTSSM write-once ID off, the patch, ASPM off.
    /// The command register and the AER severity are the PCI owner's.
    pub(super) fn reset_pcie(&self) -> Result<(), BlockError> {
        self.read(WOL_CTRL);
        self.write(WOL_CTRL, 0);
        let value = self.read(LTSSM_ID_CTRL);
        self.write(LTSSM_ID_CTRL, value & !LTSSM_ID_EN_WRO);
        self.pcie_patch();
        self.set_aspm(None, false, false);
        self.delay(5000)
    }

    /// `atl1c_phy_reset` with hibernation allowed (`ATL1C_HIB_DISABLE` is
    /// never set by `atl1c_setup_mac_funcs`).
    pub(super) fn reset_phy(&self) -> Result<(), BlockError> {
        let kind = self.chip.kind;
        let mut control = self.read(GPHY_CTRL);
        control &= !(GPHY_CTRL_EXT_RESET
            | GPHY_CTRL_PHY_IDDQ
            | GPHY_CTRL_GATE_25M_EN
            | GPHY_CTRL_PWDOWN_HW
            | GPHY_CTRL_CLS);
        control |= GPHY_CTRL_SEL_ANA_RST | GPHY_CTRL_HIB_EN | GPHY_CTRL_HIB_PULSE;
        self.write(GPHY_CTRL, control);
        self.delay(10)?;
        self.write(GPHY_CTRL, control | GPHY_CTRL_EXT_RESET);
        self.delay(10 * GPHY_CTRL_EXT_RST_TO)?;
        if kind == Kind::L2cB {
            let value = self.read_dbg(MIIDBG_CFGLPSPD)?;
            self.write_dbg(MIIDBG_CFGLPSPD, value & !CFGLPSPD_RSTCNT_CLK125SW)?;
        }
        if kind.l2cb_family() {
            let value = self.read_dbg(MIIDBG_CABLE1TH_DET)?;
            self.write_dbg(MIIDBG_CABLE1TH_DET, value | CABLE1TH_DET_EN)?;
            let value = self.read_dbg(MIIDBG_VOLT_CTRL)?;
            self.write_dbg(MIIDBG_VOLT_CTRL, value & !VOLT_CTRL_SWLOWEST)?;
        }
        let legacy = if matches!(kind, Kind::L1d | Kind::L1d2) {
            L1D_LEGCYPS_DEF
        } else {
            L1C_LEGCYPS_DEF
        };
        self.write_dbg(MIIDBG_LEGCYPS, legacy)?;
        self.write_dbg(MIIDBG_SYSMODCTRL, SYSMODCTRL_IECHOADJ_DEF)?;
        if matches!(kind, Kind::L1d | Kind::L1d2 | Kind::L2cB2) {
            let lpi = self.read(LPI_CTRL);
            self.write(LPI_CTRL, lpi & !LPI_CTRL_EN);
            self.write_ext(MIIEXT_ANEG, MIIEXT_LOCAL_EEEADV, 0)?;
            self.write_ext(MIIEXT_PCS, MIIEXT_CLDCTRL3, L2CB_CLDCTRL3)?;
        }
        self.write_dbg(MIIDBG_ANACTRL, ANACTRL_DEF)?;
        self.write_dbg(MIIDBG_SRDSYSMOD, SRDSYSMOD_DEF)?;
        self.write_dbg(MIIDBG_TST10BTCFG, TST10BTCFG_DEF)?;
        self.write_dbg(MIIDBG_TST100BTCFG, TST100BTCFG_DEF | TST100BTCFG_LITCH_EN)?;
        self.write_phy(MII_IER, IER_LINK_UP | IER_LINK_DOWN)
    }

    /// `atl1c_phy_init` for `MEDIA_TYPE_AUTO_SENSOR`: the PHY identity,
    /// then `atl1c_phy_setup_adv` with `ADVERTISED_Autoneg` and a restart.
    pub(super) fn init_phy(&self) -> Result<u32, BlockError> {
        let high = self.read_phy(MII_PHYSID1)?;
        let low = self.read_phy(MII_PHYSID2)?;
        self.write_phy(MII_ADVERTISE, ADVERTISE_AUTONEG)?;
        let giga = if self.chip.kind.gigabit() {
            ADVERTISE_1000HALF | ADVERTISE_1000FULL
        } else {
            0
        };
        self.write_phy(MII_CTRL1000, giga)?;
        self.write_phy(MII_BMCR, BMCR_RESET | BMCR_ANENABLE | BMCR_ANRESTART)?;
        Ok(u32::from(high) << 16 | u32::from(low))
    }

    /// `atl1c_stop_mac`; a unit that stays busy is reported by the reset.
    fn stop_mac(&self) -> Result<(), BlockError> {
        let value = self.read(RXQ_CTRL);
        self.write(RXQ_CTRL, value & !RXQ_CTRL_EN);
        let value = self.read(TXQ_CTRL);
        self.write(TXQ_CTRL, value & !TXQ_CTRL_EN);
        self.wait_until_idle(IDLE_STATUS_RXQ_BUSY | IDLE_STATUS_TXQ_BUSY)?;
        let value = self.read(MAC_CTRL);
        self.write(MAC_CTRL, value & !(MAC_CTRL_TX_EN | MAC_CTRL_RX_EN));
        self.wait_until_idle(IDLE_STATUS_TXMAC_BUSY | IDLE_STATUS_RXMAC_BUSY)?;
        Ok(())
    }

    /// `atl1c_reset_mac`: queues and MAC stopped, the self-clearing soft
    /// reset, driver-controlled speed, and the per-chip clock slowdown.
    pub(super) fn reset_mac(&self) -> Result<(), BlockError> {
        self.stop_mac()?;
        let master = self.read(MASTER_CTRL) | MASTER_CTRL_OOB_DIS;
        self.write(MASTER_CTRL, master | MASTER_CTRL_SOFT_RST);
        self.delay(10_000)?;
        if self.wait_until_idle(IDLE_STATUS_MASK)? != 0 {
            return Err(BlockError::Unsupported(
                "atl1c MAC stayed busy after its soft reset",
            ));
        }
        self.write(MASTER_CTRL, master);
        let value = self.read(MAC_CTRL);
        self.write(MAC_CTRL, value | MAC_CTRL_SPEED_MODE_SW);
        let value = self.read(SERDES);
        match self.chip.kind {
            Kind::L2cB => self.write(
                SERDES,
                value & !(SERDES_PHY_CLK_SLOWDOWN | SERDES_MAC_CLK_SLOWDOWN),
            ),
            Kind::L2cB2 | Kind::L1d2 => self.write(
                SERDES,
                value | SERDES_PHY_CLK_SLOWDOWN | SERDES_MAC_CLK_SLOWDOWN,
            ),
            _ => {}
        }
        Ok(())
    }

    /// `atl1c_set_aspm` for a link speed (`None` while down), with the L0s
    /// and L1 support flags; `atl1c_disable_l0s_l1` clears both.
    pub(super) fn set_aspm(&self, speed: Option<Speed>, l0s: bool, l1: bool) {
        let kind = self.chip.kind;
        let fast = matches!(speed, Some(Speed::Thousand | Speed::Hundred));
        let mut power = self.read(PM_CTRL);
        power &= !(PM_CTRL_ASPM_L1_EN | PM_CTRL_ASPM_L0S_EN | PM_CTRL_MAC_ASPM_CHK);
        if kind.second_revision() {
            power &= !PMCTRL_TXL1_AFTER_L0S;
            let timer = if fast { L1D_PMCTRL_L1_ENTRY_TM_16US } else { 1 };
            power = set_field(power, L1D_PMCTRL_L1_ENTRY_TM, timer);
        } else {
            let mut timer = if kind == Kind::L2cB {
                L2CB1_PM_CTRL_L1_ENTRY_TM
            } else {
                L1C_PM_CTRL_L1_ENTRY_TM
            };
            if !fast {
                timer = 1;
            }
            power = set_field(power, PM_CTRL_L1_ENTRY_TIMER, timer);
        }
        if l0s && speed.is_some() {
            power |= PM_CTRL_ASPM_L0S_EN | PM_CTRL_MAC_ASPM_CHK;
        }
        if l1 {
            power |= PM_CTRL_ASPM_L1_EN | PM_CTRL_MAC_ASPM_CHK;
        }
        if !kind.first_generation() {
            power = set_field(power, PM_CTRL_PM_REQ_TIMER, PM_CTRL_PM_REQ_TO_DEF);
            power |= PM_CTRL_RCVR_WT_TIMER | PM_CTRL_SERDES_PD_EX_L1 | PM_CTRL_CLK_SWH_L1;
            power &= !(PM_CTRL_SERDES_L1_EN
                | PM_CTRL_SERDES_PLL_L1_EN
                | PM_CTRL_SERDES_BUFS_RX_L1_EN
                | PM_CTRL_SA_DLY_EN
                | PM_CTRL_HOTRST);
            if speed.is_none() || kind == Kind::L2cB {
                power &= !PM_CTRL_ASPM_L0S_EN;
            }
        } else {
            power = set_field(power, PM_CTRL_L1_ENTRY_TIMER, 0);
            if speed.is_some() {
                power |=
                    PM_CTRL_SERDES_L1_EN | PM_CTRL_SERDES_PLL_L1_EN | PM_CTRL_SERDES_BUFS_RX_L1_EN;
                power &= !(PM_CTRL_SERDES_PD_EX_L1
                    | PM_CTRL_CLK_SWH_L1
                    | PM_CTRL_ASPM_L0S_EN
                    | PM_CTRL_ASPM_L1_EN);
            } else {
                power |= PM_CTRL_CLK_SWH_L1;
                power &= !(PM_CTRL_SERDES_L1_EN
                    | PM_CTRL_SERDES_PLL_L1_EN
                    | PM_CTRL_SERDES_BUFS_RX_L1_EN
                    | PM_CTRL_ASPM_L0S_EN);
            }
        }
        self.write(PM_CTRL, power);
    }

    fn station(&self) -> Option<crate::Mac> {
        let low = self.read(MAC_STA_ADDR).to_be_bytes();
        let high = (self.read(MAC_STA_ADDR + 4) as u16).to_be_bytes();
        let mac = crate::Mac([high[0], high[1], low[0], low[1], low[2], low[3]]);
        (!mac.is_multicast() && mac.0 != [0; 6]).then_some(mac)
    }

    /// `atl1c_get_permanent_address`: what firmware left in the station
    /// registers, otherwise the EEPROM or OTP load. Linux falls back to a
    /// random address; this port refuses the adapter instead.
    pub(super) fn permanent_mac(&self) -> Result<crate::Mac, BlockError> {
        if let Some(mac) = self.station() {
            return Ok(mac);
        }
        let kind = self.chip.kind;
        let mut otp = self.read(OTP_CTRL);
        let eeprom = self.read(TWSI_DEBUG) & TWSI_DEBUG_DEV_EXIST != 0
            || self.read(MASTER_CTRL) & MASTER_CTRL_OTP_SEL != 0;
        let mut raised = false;
        if eeprom {
            if kind.first_generation() && otp & OTP_CTRL_CLK_EN == 0 {
                otp |= OTP_CTRL_CLK_EN;
                self.write(OTP_CTRL, otp);
                self.delay(1000)?;
            }
            if kind.l2cb_family() {
                let value = self.read_dbg(MIIDBG_ANACTRL)?;
                self.write_dbg(MIIDBG_ANACTRL, value & !ANACTRL_HB_EN)?;
                let value = self.read_dbg(MIIDBG_VOLT_CTRL)?;
                self.write_dbg(MIIDBG_VOLT_CTRL, value | VOLT_CTRL_SWLOWEST)?;
                self.delay(20)?;
                raised = true;
            }
            let value = self.read(TWSI_CTRL);
            self.write(TWSI_CTRL, value | TWSI_CTRL_SW_LDSTART);
            let mut loaded = false;
            for _ in 0..TWSI_EEPROM_TIMEOUT {
                self.delay(10_000)?;
                if self.read(TWSI_CTRL) & TWSI_CTRL_SW_LDSTART == 0 {
                    loaded = true;
                    break;
                }
            }
            if !loaded {
                return Err(BlockError::Unsupported(
                    "atl1c station address load timed out",
                ));
            }
        }
        if kind.first_generation() {
            self.write(OTP_CTRL, otp & !OTP_CTRL_CLK_EN);
            self.delay(1000)?;
        }
        if raised {
            let value = self.read_dbg(MIIDBG_ANACTRL)?;
            self.write_dbg(MIIDBG_ANACTRL, value | ANACTRL_HB_EN)?;
            let value = self.read_dbg(MIIDBG_VOLT_CTRL)?;
            self.write_dbg(MIIDBG_VOLT_CTRL, value & !VOLT_CTRL_SWLOWEST)?;
            self.delay(20)?;
        }
        self.station().ok_or(BlockError::Unsupported(
            "atl1c without a valid station address",
        ))
    }

    /// `atl1c_hw_set_mac_addr`.
    pub(super) fn set_mac(&self, mac: crate::Mac) {
        let [a, b, c, d, e, f] = mac.0;
        self.write(MAC_STA_ADDR, u32::from_be_bytes([c, d, e, f]));
        self.write(MAC_STA_ADDR + 4, u32::from(u16::from_be_bytes([a, b])));
    }

    /// `atl1c_set_multi` with `IFF_ALLMULTI` and an empty list.
    pub(super) fn receive_filter(&self) {
        let mut mac = self.read(MAC_CTRL);
        mac |= MAC_CTRL_MC_ALL_EN;
        mac &= !MAC_CTRL_PROMIS_EN;
        self.write(MAC_CTRL, mac);
        self.write(RX_HASH_TABLE, 0);
        self.write(RX_HASH_TABLE + 4, 0);
    }

    /// `atl1c_configure_des_ring`: one used and one idle transmit ring (the
    /// high-priority ring is always programmed), one receive queue.
    pub(super) fn ring_pointers(
        &self,
        transmit: u64,
        idle_transmit: u64,
        receive_free: u64,
        receive_return: u64,
    ) {
        self.write(TX_BASE_ADDR_HI, (transmit >> 32) as u32);
        self.write(TPD_PRI0_ADDR_LO, transmit as u32);
        self.write(TPD_PRI1_ADDR_LO, idle_transmit as u32);
        self.write(TPD_RING_SIZE, TX_RING as u32);
        self.write(RX_BASE_ADDR_HI, (receive_free >> 32) as u32);
        self.write(RFD0_HEAD_ADDR_LO, receive_free as u32);
        self.write(RFD_RING_SIZE, RX_RING as u32);
        self.write(RX_BUF_SIZE, RX_BUFFER);
        self.write(RRD0_HEAD_ADDR_LO, receive_return as u32);
        self.write(RRD_RING_SIZE, RX_RING as u32);
        if self.chip.kind == Kind::L2cB {
            self.write(SRAM_RXF_LEN, 0x02a0);
            self.write(SRAM_TXF_LEN, 0x0100);
            self.write(SRAM_RXF_ADDR, 0x029f_0000);
            self.write(SRAM_RFD0_INFO, 0x02bf_02a0);
            self.write(SRAM_TXF_ADDR, 0x03bf_02c0);
            self.write(SRAM_TRD_ADDR, 0x03df_03c0);
            self.write(TXF_WATER_MARK, 0);
            self.write(RXD_DMA_CTRL, 0);
        }
        self.write(LOAD_PTR, 1);
    }

    /// `atl1c_configure_mac` around the ring pointers it loads: status and
    /// WoL cleared, no clock gating, moderation timers, then
    /// `atl1c_configure_tx`, `_rx` and `_dma`. The interrupt mask stays
    /// zero: this driver polls.
    pub(super) fn configure_mac(&self, rings: impl FnOnce(&Self)) {
        let mut master = self.read(MASTER_CTRL);
        master &= !(MASTER_CTRL_TX_ITIMER_EN | MASTER_CTRL_RX_ITIMER_EN | MASTER_CTRL_INT_RDCLR);
        self.write(ISR, u32::MAX);
        self.write(WOL_CTRL, 0);
        self.write(CLK_GATING_CTRL, 0);
        self.write(INT_RETRIG_TIMER, ICT & 0xffff);
        rings(self);
        self.write(
            IRQ_MODRT_TIMER_INIT,
            (TX_IMT & 0xffff) | ((RX_IMT & 0xffff) << IRQ_MODRT_RX_TIMER_SHIFT),
        );
        master |= MASTER_CTRL_TX_ITIMER_EN | MASTER_CTRL_RX_ITIMER_EN | MASTER_CTRL_SA_TIMER_EN;
        self.write(MASTER_CTRL, master);
        self.write(SMB_STAT_TIMER, SMB_TIMER & 0x00ff_ffff);
        self.write(MTU, FRAME);
        self.write(
            TX_TSO_OFFLOAD_THRESH,
            (MAX_TSO_FRAME_SIZE >> 3) & TX_TSO_OFFLOAD_THRESH_MASK,
        );
        let transmit = if self.chip.kind.l2cb_family() {
            L2CB_TXQ_CFGV
        } else {
            L1C_TXQ_CFGV
        };
        self.write(TXQ_CTRL, transmit);
        let mut receive = (RFD_BURST & 0x3f) << RXQ_RFD_BURST_NUM_SHIFT;
        if self.chip.kind != Kind::L1d2 && self.chip.device & 1 != 0 {
            receive = (receive & !3) | ASPM_THRUPUT_LIMIT_100M;
        }
        self.write(RXQ_CTRL, receive);
        self.write(
            DMA_CTRL,
            DMA_CTRL_RORDER_MODE_OUT
                | DMA_CTRL_RREQ_PRI_DATA
                | ((self.dmar_block & 7) << DMA_CTRL_RREQ_BLEN_SHIFT)
                | (DMA_CTRL_WDLY_CNT_DEF << DMA_CTRL_WDLY_CNT_SHIFT)
                | (DMA_CTRL_RDLY_CNT_DEF << DMA_CTRL_RDLY_CNT_SHIFT),
        );
    }

    /// `atl1c_start_mac`.
    pub(super) fn start_mac(&self) {
        let transmit = self.read(TXQ_CTRL) | TXQ_CTRL_EN;
        let receive = self.read(RXQ_CTRL) | RXQ_CTRL_EN;
        let mut mac = self.read(MAC_CTRL);
        mac |= MAC_CTRL_TX_EN
            | MAC_CTRL_TX_FLOW
            | MAC_CTRL_RX_EN
            | MAC_CTRL_RX_FLOW
            | MAC_CTRL_ADD_CRC
            | MAC_CTRL_PAD
            | MAC_CTRL_BC_EN
            | MAC_CTRL_SINGLE_PAUSE_EN
            | MAC_CTRL_HASH_ALG_CRC32;
        if self.link.full_duplex {
            mac |= MAC_CTRL_DUPLX;
        } else {
            mac &= !MAC_CTRL_DUPLX;
        }
        let speed = if self.link.speed == Some(Speed::Thousand) {
            MAC_CTRL_SPEED_1000
        } else {
            MAC_CTRL_SPEED_10_100
        };
        mac = set_field(mac, MAC_CTRL_SPEED, speed);
        mac = set_field(mac, MAC_CTRL_PRMLEN, PREAMBLE);
        self.write(TXQ_CTRL, transmit);
        self.write(RXQ_CTRL, receive);
        self.write(MAC_CTRL, mac);
    }

    /// `atl1c_get_link_status` and `atl1c_get_speed_and_duplex`.
    pub(super) fn read_phy_link(&self) -> Result<LinkState, BlockError> {
        self.read_phy(MII_BMSR)?;
        if self.read_phy(MII_BMSR)? & BMSR_LSTATUS == 0 {
            return Ok(LinkState::default());
        }
        let specific = self.read_phy(MII_GIGA_PSSR)?;
        if specific & GIGA_PSSR_SPD_DPLX_RESOLVED == 0 {
            return Err(BlockError::Unsupported(
                "atl1c PHY speed and duplex unresolved",
            ));
        }
        let speed = match specific & GIGA_PSSR_SPEED {
            GIGA_PSSR_1000MBS => Speed::Thousand,
            GIGA_PSSR_100MBS => Speed::Hundred,
            GIGA_PSSR_10MBS => Speed::Ten,
            _ => {
                return Err(BlockError::Unsupported(
                    "atl1c PHY reports an invalid speed",
                ));
            }
        };
        Ok(LinkState {
            speed: Some(speed),
            full_duplex: specific & GIGA_PSSR_DPLX != 0,
        })
    }

    /// `atl1c_post_phy_linkchg`.
    pub(super) fn post_phy_link(&self, speed: Option<Speed>) -> Result<(), BlockError> {
        let kind = self.chip.kind;
        let adjust = !kind.first_generation();
        match speed {
            Some(speed) => {
                if kind == Kind::L1d2 {
                    let length =
                        self.read_ext(MIIEXT_PCS, MIIEXT_CLDCTRL6)? & CLDCTRL6_CAB_LEN_MASK;
                    let value = if length > CLDCTRL6_CAB_LEN_SHORT {
                        AZ_ANADECT_LONG
                    } else {
                        AZ_ANADECT_DEF
                    };
                    self.write_dbg(MIIDBG_AZ_ANADECT, value)?;
                }
                if adjust && speed == Speed::Hundred && self.chip.link_patch {
                    self.write_dbg(MIIDBG_MSE16DB, L1D_MSE16DB_UP)?;
                    self.write_dbg(MIIDBG_SYSMODCTRL, L1D_SYSMODCTRL_IECHOADJ_DEF)?;
                }
            }
            None => {
                if adjust && self.chip.link_patch {
                    self.write_dbg(MIIDBG_SYSMODCTRL, SYSMODCTRL_IECHOADJ_DEF)?;
                    self.write_dbg(MIIDBG_MSE16DB, L1D_MSE16DB_DOWN)?;
                }
            }
        }
        Ok(())
    }

    /// `atl1c_irq_disable`'s register part.
    pub(super) fn mask_interrupts(&self) {
        self.write(IMR, 0);
        self.write(ISR, ISR_DIS_INT);
    }

    pub(super) fn produce_receive(&self, index: usize) {
        self.write(MB_RFD0_PROD_IDX, index as u32 & 0xffff);
    }
    pub(super) fn produce_transmit(&self, index: usize) {
        self.registers.write16(TPD_PRI0_PIDX, index as u16);
    }
    pub(super) fn transmit_consumer(&self) -> usize {
        usize::from(self.registers.read16(TPD_PRI0_CIDX))
    }
}
