// SPDX-License-Identifier: GPL-2.0-or-later
// Register values: Linux `atl1c_hw.h`, Copyright (c) 2008 Atheros
// Communications Inc., Copyright (c) 2007 Atheros Corporation.
// Copyright (c) 2026 Stela contributors
//! The registers, fields and PHY addresses of the atl1c family that the
//! polled single-queue path uses, with the values of Linux `atl1c_hw.h`.

/// PCI Express Device Control's maximum read request size, as 128 << n.
pub(super) const DEVICE_CTRL_MAXRRS_MIN: u32 = 2;
/// L2C_B revision 1.0 and L2C_B2 revision 2.1 PCI revision IDs.
pub(super) const L2CB_V10: u8 = 0xc0;
pub(super) const L2CB_V21: u8 = 0xc1;

/// The indirect configuration window `atl1c_patch_assign` uses.
pub(super) const PCIE_IND_ACC_ADDR: u16 = 0x80;
pub(super) const PCIE_IND_ACC_DATA: u16 = 0x84;
pub(super) const PCIE_DEV_MISC_CTRL: u32 = 0x21c;

pub(super) const LINK_CTRL: usize = 0x68;
pub(super) const LINK_CTRL_EXT_SYNC: u32 = 0x80;

pub(super) const TWSI_CTRL: usize = 0x218;
pub(super) const TWSI_CTRL_SW_LDSTART: u32 = 1 << 11;
pub(super) const PCIE_PHYMISC: usize = 0x1000;
pub(super) const PCIE_PHYMISC_FORCE_RCV_DET: u32 = 1 << 2;
pub(super) const PCIE_PHYMISC2: usize = 0x1004;
pub(super) const PCIE_PHYMISC2_L0S_TH: (u32, u32) = (3, 18);
pub(super) const PCIE_PHYMISC2_CDR_BW: (u32, u32) = (3, 16);
pub(super) const L2CB1_PCIE_PHYMISC2_L0S_TH: u32 = 3;
pub(super) const L2CB1_PCIE_PHYMISC2_CDR_BW: u32 = 3;
pub(super) const TWSI_DEBUG: usize = 0x1108;
pub(super) const TWSI_DEBUG_DEV_EXIST: u32 = 1 << 29;
pub(super) const DMA_DBG: usize = 0x1114;
pub(super) const DMA_DBG_VENDOR_MSG: u32 = 1 << 0;
pub(super) const OTP_CTRL: usize = 0x12f0;
pub(super) const OTP_CTRL_CLK_EN: u32 = 1 << 1;

pub(super) const PM_CTRL: usize = 0x12f8;
pub(super) const PM_CTRL_HOTRST: u32 = 1 << 31;
pub(super) const PM_CTRL_MAC_ASPM_CHK: u32 = 1 << 30;
pub(super) const PM_CTRL_SA_DLY_EN: u32 = 1 << 29;
pub(super) const PM_CTRL_L0S_BUFSRX_EN: u32 = 1 << 28;
pub(super) const PM_CTRL_PM_REQ_TIMER: (u32, u32) = (0xf, 20);
pub(super) const PM_CTRL_PM_REQ_TO_DEF: u32 = 0xf;
pub(super) const PMCTRL_TXL1_AFTER_L0S: u32 = 1 << 19;
pub(super) const L1D_PMCTRL_L1_ENTRY_TM: (u32, u32) = (7, 16);
pub(super) const L1D_PMCTRL_L1_ENTRY_TM_16US: u32 = 4;
pub(super) const PM_CTRL_L1_ENTRY_TIMER: (u32, u32) = (0xf, 16);
pub(super) const L2CB1_PM_CTRL_L1_ENTRY_TM: u32 = 7;
pub(super) const L1C_PM_CTRL_L1_ENTRY_TM: u32 = 0xf;
pub(super) const PM_CTRL_RCVR_WT_TIMER: u32 = 1 << 15;
pub(super) const PM_CTRL_CLK_SWH_L1: u32 = 1 << 13;
pub(super) const PM_CTRL_ASPM_L0S_EN: u32 = 1 << 12;
pub(super) const PM_CTRL_SERDES_BUFS_RX_L1_EN: u32 = 1 << 7;
pub(super) const PM_CTRL_SERDES_PD_EX_L1: u32 = 1 << 6;
pub(super) const PM_CTRL_SERDES_PLL_L1_EN: u32 = 1 << 5;
pub(super) const PM_CTRL_SERDES_L1_EN: u32 = 1 << 4;
pub(super) const PM_CTRL_ASPM_L1_EN: u32 = 1 << 3;

pub(super) const LTSSM_ID_CTRL: usize = 0x12fc;
pub(super) const LTSSM_ID_EN_WRO: u32 = 0x1000;

pub(super) const MASTER_CTRL: usize = 0x1400;
pub(super) const MASTER_CTRL_OTP_SEL: u32 = 1 << 31;
pub(super) const MASTER_CTRL_INT_RDCLR: u32 = 1 << 14;
pub(super) const MASTER_CTRL_CLK_SEL_DIS: u32 = 1 << 12;
pub(super) const MASTER_CTRL_RX_ITIMER_EN: u32 = 1 << 11;
pub(super) const MASTER_CTRL_TX_ITIMER_EN: u32 = 1 << 10;
pub(super) const MASTER_CTRL_SA_TIMER_EN: u32 = 1 << 7;
pub(super) const MASTER_CTRL_OOB_DIS: u32 = 1 << 6;
pub(super) const MASTER_CTRL_WAKEN_25M: u32 = 1 << 5;
pub(super) const MASTER_CTRL_SOFT_RST: u32 = 1 << 0;

pub(super) const IRQ_MODRT_TIMER_INIT: usize = 0x1408;
pub(super) const IRQ_MODRT_RX_TIMER_SHIFT: u32 = 16;

pub(super) const GPHY_CTRL: usize = 0x140c;
pub(super) const GPHY_CTRL_PWDOWN_HW: u32 = 1 << 14;
pub(super) const GPHY_CTRL_PHY_PLL_ON: u32 = 1 << 13;
pub(super) const GPHY_CTRL_SEL_ANA_RST: u32 = 1 << 12;
pub(super) const GPHY_CTRL_HIB_PULSE: u32 = 1 << 11;
pub(super) const GPHY_CTRL_HIB_EN: u32 = 1 << 10;
pub(super) const GPHY_CTRL_PHY_IDDQ: u32 = 1 << 7;
pub(super) const GPHY_CTRL_GATE_25M_EN: u32 = 1 << 5;
pub(super) const GPHY_CTRL_100AB_EN: u32 = 1 << 17;
pub(super) const GPHY_CTRL_LED_MODE: u32 = 1 << 2;
pub(super) const GPHY_CTRL_EXT_RESET: u32 = 1 << 0;
/// `GPHY_CTRL_EXT_RST_TO`: the PHY reset settles in 80 × 10 µs.
pub(super) const GPHY_CTRL_EXT_RST_TO: u64 = 80;
pub(super) const GPHY_CTRL_CLS: u32 =
    GPHY_CTRL_LED_MODE | GPHY_CTRL_100AB_EN | GPHY_CTRL_PHY_PLL_ON;

pub(super) const IDLE_STATUS: usize = 0x1410;
pub(super) const IDLE_STATUS_TXQ_BUSY: u32 = 1 << 3;
pub(super) const IDLE_STATUS_RXQ_BUSY: u32 = 1 << 2;
pub(super) const IDLE_STATUS_TXMAC_BUSY: u32 = 1 << 1;
pub(super) const IDLE_STATUS_RXMAC_BUSY: u32 = 1 << 0;
pub(super) const IDLE_STATUS_MASK: u32 =
    IDLE_STATUS_TXQ_BUSY | IDLE_STATUS_RXQ_BUSY | IDLE_STATUS_TXMAC_BUSY | IDLE_STATUS_RXMAC_BUSY;
/// `AT_HW_MAX_IDLE_DELAY`: idle polls, one millisecond apart.
pub(super) const MAX_IDLE_DELAY: u32 = 10;

pub(super) const MDIO_CTRL: usize = 0x1414;
pub(super) const MDIO_CTRL_MODE_EXT: u32 = 1 << 30;
pub(super) const MDIO_CTRL_BUSY: u32 = 1 << 27;
pub(super) const MDIO_CTRL_CLK_SEL_SHIFT: u32 = 24;
pub(super) const MDIO_CTRL_CLK_25_4: u32 = 0;
pub(super) const MDIO_CTRL_CLK_25_128: u32 = 7;
pub(super) const MDIO_CTRL_START: u32 = 1 << 23;
pub(super) const MDIO_CTRL_SPRES_PRMBL: u32 = 1 << 22;
pub(super) const MDIO_CTRL_OP_READ: u32 = 1 << 21;
pub(super) const MDIO_CTRL_REG_SHIFT: u32 = 16;
/// `MDIO_MAX_AC_TO`: MDIO polls, ten microseconds apart.
pub(super) const MDIO_MAX_AC_TO: u32 = 120;
pub(super) const MDIO_EXTN: usize = 0x1448;
pub(super) const MDIO_EXTN_DEVAD_SHIFT: u32 = 16;

pub(super) const SERDES: usize = 0x1424;
pub(super) const SERDES_PHY_CLK_SLOWDOWN: u32 = 1 << 18;
pub(super) const SERDES_MAC_CLK_SLOWDOWN: u32 = 1 << 17;
pub(super) const LPI_CTRL: usize = 0x1440;
pub(super) const LPI_CTRL_EN: u32 = 1 << 0;

pub(super) const MAC_CTRL: usize = 0x1480;
pub(super) const MAC_CTRL_SPEED_MODE_SW: u32 = 1 << 30;
pub(super) const MAC_CTRL_HASH_ALG_CRC32: u32 = 1 << 29;
pub(super) const MAC_CTRL_SINGLE_PAUSE_EN: u32 = 1 << 28;
pub(super) const MAC_CTRL_BC_EN: u32 = 1 << 26;
pub(super) const MAC_CTRL_MC_ALL_EN: u32 = 1 << 25;
pub(super) const MAC_CTRL_SPEED: (u32, u32) = (3, 20);
pub(super) const MAC_CTRL_SPEED_10_100: u32 = 1;
pub(super) const MAC_CTRL_SPEED_1000: u32 = 2;
pub(super) const MAC_CTRL_PROMIS_EN: u32 = 1 << 15;
pub(super) const MAC_CTRL_PRMLEN: (u32, u32) = (0xf, 10);
pub(super) const MAC_CTRL_PAD: u32 = 1 << 7;
pub(super) const MAC_CTRL_ADD_CRC: u32 = 1 << 6;
pub(super) const MAC_CTRL_DUPLX: u32 = 1 << 5;
pub(super) const MAC_CTRL_RX_FLOW: u32 = 1 << 3;
pub(super) const MAC_CTRL_TX_FLOW: u32 = 1 << 2;
pub(super) const MAC_CTRL_RX_EN: u32 = 1 << 1;
pub(super) const MAC_CTRL_TX_EN: u32 = 1 << 0;
pub(super) const WOL_CTRL: usize = 0x14a0;
pub(super) const MAC_STA_ADDR: usize = 0x1488;
pub(super) const RX_HASH_TABLE: usize = 0x1490;
pub(super) const MTU: usize = 0x149c;

pub(super) const SRAM_RFD0_INFO: usize = 0x1500;
pub(super) const SRAM_TRD_ADDR: usize = 0x1518;
pub(super) const SRAM_RXF_ADDR: usize = 0x1520;
pub(super) const SRAM_RXF_LEN: usize = 0x1524;
pub(super) const SRAM_TXF_ADDR: usize = 0x1528;
pub(super) const SRAM_TXF_LEN: usize = 0x152c;
pub(super) const LOAD_PTR: usize = 0x1534;
pub(super) const RX_BASE_ADDR_HI: usize = 0x1540;
pub(super) const TX_BASE_ADDR_HI: usize = 0x1544;
pub(super) const RFD0_HEAD_ADDR_LO: usize = 0x1550;
pub(super) const RFD_RING_SIZE: usize = 0x1560;
pub(super) const RX_BUF_SIZE: usize = 0x1564;
pub(super) const RRD0_HEAD_ADDR_LO: usize = 0x1568;
pub(super) const RRD_RING_SIZE: usize = 0x1578;
pub(super) const TPD_PRI1_ADDR_LO: usize = 0x157c;
pub(super) const TPD_PRI0_ADDR_LO: usize = 0x1580;
pub(super) const TPD_RING_SIZE: usize = 0x1584;
pub(super) const TXQ_CTRL: usize = 0x1590;
pub(super) const TXQ_CTRL_EN: u32 = 1 << 5;
pub(super) const TX_TSO_OFFLOAD_THRESH: usize = 0x1594;
pub(super) const TX_TSO_OFFLOAD_THRESH_MASK: u32 = 0x07ff;
/// `MAX_TSO_FRAME_SIZE`.
pub(super) const MAX_TSO_FRAME_SIZE: u32 = 7 * 1024;
pub(super) const TXF_WATER_MARK: usize = 0x1598;
pub(super) const RXQ_CTRL: usize = 0x15a0;
pub(super) const RXQ_CTRL_EN: u32 = 1 << 31;
pub(super) const RXQ_RFD_BURST_NUM_SHIFT: u32 = 20;
pub(super) const ASPM_THRUPUT_LIMIT_100M: u32 = 3;
pub(super) const RXD_DMA_CTRL: usize = 0x15ac;
pub(super) const DMA_CTRL: usize = 0x15c0;
pub(super) const DMA_CTRL_WDLY_CNT_SHIFT: u32 = 16;
pub(super) const DMA_CTRL_WDLY_CNT_DEF: u32 = 4;
pub(super) const DMA_CTRL_RDLY_CNT_SHIFT: u32 = 11;
pub(super) const DMA_CTRL_RDLY_CNT_DEF: u32 = 15;
pub(super) const DMA_CTRL_RREQ_PRI_DATA: u32 = 1 << 10;
pub(super) const DMA_CTRL_RREQ_BLEN_SHIFT: u32 = 4;
pub(super) const DMA_CTRL_RORDER_MODE_OUT: u32 = 4;
pub(super) const SMB_STAT_TIMER: usize = 0x15c4;
pub(super) const MB_RFD0_PROD_IDX: usize = 0x15e0;
pub(super) const TPD_PRI0_PIDX: usize = 0x15f2;
pub(super) const TPD_PRI0_CIDX: usize = 0x15f6;

/// `TXQ_CFGV`: five descriptors a burst, enhanced mode, 802.3 large send
/// and IP options enabled, as Linux programs it whatever offloads are used.
const TXQ_CFGV: u32 = 5 | (1 << 6) | (1 << 7) | (1 << 4);
pub(super) const L1C_TXQ_CFGV: u32 = TXQ_CFGV | (0x200 << 16);
pub(super) const L2CB_TXQ_CFGV: u32 = TXQ_CFGV | (0x40 << 16);

pub(super) const ISR: usize = 0x1600;
pub(super) const ISR_DIS_INT: u32 = 1 << 31;
pub(super) const IMR: usize = 0x1604;
pub(super) const INT_RETRIG_TIMER: usize = 0x1608;
pub(super) const CLK_GATING_CTRL: usize = 0x1814;
pub(super) const MT_MAGIC_REG: usize = 0x1f00;
pub(super) const MT_MAGIC: u32 = 0xaabb_1234;

// Clause 22 PHY registers and the vendor PHY's specific ones.
pub(super) const MII_BMCR: u16 = 0x00;
pub(super) const MII_BMSR: u16 = 0x01;
pub(super) const MII_PHYSID1: u16 = 0x02;
pub(super) const MII_PHYSID2: u16 = 0x03;
pub(super) const MII_ADVERTISE: u16 = 0x04;
pub(super) const MII_CTRL1000: u16 = 0x09;
pub(super) const MII_GIGA_PSSR: u16 = 0x11;
pub(super) const MII_IER: u16 = 0x12;
pub(super) const MII_DBG_ADDR: u16 = 0x1d;
pub(super) const MII_DBG_DATA: u16 = 0x1e;
pub(super) const BMCR_RESET: u16 = 0x8000;
pub(super) const BMCR_ANENABLE: u16 = 0x1000;
pub(super) const BMCR_ANRESTART: u16 = 0x0200;
pub(super) const BMSR_LSTATUS: u16 = 0x0004;
/// `ADVERTISE_DEFAULT_CAP` with all four 10/100 abilities: 10/100 half and
/// full, symmetric and asymmetric pause (no selector bits, as Linux).
pub(super) const ADVERTISE_AUTONEG: u16 = 0x01e0 | 0x0400 | 0x0800;
pub(super) const ADVERTISE_1000HALF: u16 = 0x0100;
pub(super) const ADVERTISE_1000FULL: u16 = 0x0200;
pub(super) const GIGA_PSSR_SPD_DPLX_RESOLVED: u16 = 0x0800;
pub(super) const GIGA_PSSR_DPLX: u16 = 0x2000;
pub(super) const GIGA_PSSR_SPEED: u16 = 0xc000;
pub(super) const GIGA_PSSR_10MBS: u16 = 0x0000;
pub(super) const GIGA_PSSR_100MBS: u16 = 0x4000;
pub(super) const GIGA_PSSR_1000MBS: u16 = 0x8000;
pub(super) const IER_LINK_UP: u16 = 0x0400;
pub(super) const IER_LINK_DOWN: u16 = 0x0800;

// The PHY's debug port.
pub(super) const MIIDBG_ANACTRL: u16 = 0x00;
pub(super) const ANACTRL_HB_EN: u16 = 0x80;
pub(super) const ANACTRL_DEF: u16 = 0x02ef;
pub(super) const MIIDBG_SYSMODCTRL: u16 = 0x04;
pub(super) const SYSMODCTRL_IECHOADJ_DEF: u16 = 0x88bb;
pub(super) const L1D_SYSMODCTRL_IECHOADJ_DEF: u16 = 0x4fbb;
pub(super) const MIIDBG_SRDSYSMOD: u16 = 0x05;
pub(super) const SRDSYSMOD_DEF: u16 = 0x2c46;
pub(super) const MIIDBG_CFGLPSPD: u16 = 0x0a;
pub(super) const CFGLPSPD_RSTCNT_CLK125SW: u16 = 0x2000;
pub(super) const MIIDBG_TST10BTCFG: u16 = 0x12;
pub(super) const TST10BTCFG_DEF: u16 = 0x4c04;
pub(super) const MIIDBG_AZ_ANADECT: u16 = 0x15;
pub(super) const AZ_ANADECT_DEF: u16 = 0x3220;
pub(super) const AZ_ANADECT_LONG: u16 = 0xb210;
pub(super) const MIIDBG_MSE16DB: u16 = 0x18;
pub(super) const L1D_MSE16DB_UP: u16 = 0x05ea;
pub(super) const L1D_MSE16DB_DOWN: u16 = 0x02ea;
pub(super) const MIIDBG_LEGCYPS: u16 = 0x29;
pub(super) const L1D_LEGCYPS_DEF: u16 = 0x129d;
pub(super) const L1C_LEGCYPS_DEF: u16 = 0x36dd;
pub(super) const MIIDBG_TST100BTCFG: u16 = 0x36;
pub(super) const TST100BTCFG_DEF: u16 = 0xe12c;
pub(super) const TST100BTCFG_LITCH_EN: u16 = 0x80;
pub(super) const MIIDBG_VOLT_CTRL: u16 = 0x3b;
pub(super) const VOLT_CTRL_SWLOWEST: u16 = 0x8;
pub(super) const MIIDBG_CABLE1TH_DET: u16 = 0x3e;
pub(super) const CABLE1TH_DET_EN: u16 = 0x8000;

// Extended (clause 45 style) PHY registers.
pub(super) const MIIEXT_PCS: u8 = 3;
pub(super) const MIIEXT_CLDCTRL3: u16 = 0x8003;
pub(super) const L2CB_CLDCTRL3: u16 = 0x4d19;
pub(super) const MIIEXT_CLDCTRL6: u16 = 0x8006;
pub(super) const CLDCTRL6_CAB_LEN_MASK: u16 = 0x1ff;
pub(super) const CLDCTRL6_CAB_LEN_SHORT: u16 = 0x50;
pub(super) const MIIEXT_ANEG: u8 = 7;
pub(super) const MIIEXT_LOCAL_EEEADV: u16 = 0x3c;

// Descriptor fields (`atl1c.h`).
pub(super) const TPD_EOP: u32 = 1 << 31;
pub(super) const RRS_RX_RFD_CNT: (u32, u32) = (0xf, 16);
pub(super) const RRS_RX_RFD_INDEX: (u32, u32) = (0xfff, 20);
pub(super) const RRS_PKT_SIZE_MASK: u32 = 0x3fff;
pub(super) const RRS_RX_ERR_SUM: u32 = 1 << 20;
pub(super) const RRS_802_3_LEN_ERR: u32 = 1 << 30;
pub(super) const RRS_RXD_UPDATED: u32 = 1 << 31;

/// `FIELD_GETX`.
pub(super) fn field(value: u32, (mask, shift): (u32, u32)) -> u32 {
    (value >> shift) & mask
}

/// `FIELD_SETX`.
pub(super) fn set_field(value: u32, (mask, shift): (u32, u32), field: u32) -> u32 {
    (value & !(mask << shift)) | ((field & mask) << shift)
}
