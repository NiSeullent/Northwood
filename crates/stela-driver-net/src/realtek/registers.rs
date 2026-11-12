// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2002 ShuChen <shuchen@realtek.com.tw>
// Copyright (c) 2003-2007 Francois Romieu <romieu@fr.zoreil.com>
// Copyright (c) a lot of people too. Please respect their work.
// Copyright (c) 2026 Stela contributors
//! Register and descriptor encodings from the pinned Linux r8169 driver.
pub(super) const MAC0: usize = 0;
pub(super) const MAC4: usize = 4;
pub(super) const MAR0: usize = 8;
pub(super) const TX_LOW: usize = 0x20;
pub(super) const TX_HIGH: usize = 0x24;
pub(super) const COMMAND: usize = 0x37;
pub(super) const TX_POLL: usize = 0x38;
pub(super) const INTERRUPT_MASK: usize = 0x3c;
pub(super) const INTERRUPT_STATUS: usize = 0x3e;
pub(super) const NBASE_INTERRUPT_MASK: usize = 0x38;
pub(super) const NBASE_INTERRUPT_STATUS: usize = 0x3c;
/// Interrupt events: a frame received, a frame received with an error, and
/// a link change.
pub(super) const RX_OK: u16 = 1 << 0;
pub(super) const RX_ERROR: u16 = 1 << 1;
pub(super) const LINK_CHANGE: u16 = 1 << 5;
pub(super) const NBASE_TX_POLL: usize = 0x90;
pub(super) const NBASE_MAC0_BACKUP: usize = 0x19e0;
pub(super) const NBASE_RSS: usize = 0x4500;
pub(super) const NBASE_QUEUE: usize = 0x4800;
pub(super) const NBASE_EEE_TXIDLE: usize = 0x6048;
pub(super) const NBASE_REGISTER_BYTES: u64 = (NBASE_EEE_TXIDLE + 2) as u64;
pub(super) const NBASE_AN_CONTROL: u16 = 0xa5d4;
pub(super) const NBASE_EEE_ADV2: u16 = 0xa6d4;
pub(super) const NBASE_ADV_2500: u16 = 0x0080;
pub(super) const NBASE_ADV_5000: u16 = 0x0100;
pub(super) const NBASE_ADV_10000: u16 = 0x1000;
pub(super) const TX_CONFIG: usize = 0x40;
pub(super) const RX_CONFIG: usize = 0x44;
pub(super) const CONFIG_LOCK: usize = 0x50;
pub(super) const CONFIG1: usize = 0x52;
pub(super) const CONFIG2: usize = 0x53;
pub(super) const CONFIG3: usize = 0x54;
pub(super) const CONFIG4: usize = 0x55;
pub(super) const CONFIG5: usize = 0x56;
pub(super) const PHY_ACCESS: usize = 0x60;
pub(super) const CSI_DATA: usize = 0x64;
pub(super) const CSI_ADDRESS: usize = 0x68;
pub(super) const PHY_STATUS: usize = 0x6c;
pub(super) const ERI_DATA: usize = 0x70;
pub(super) const ERI_ADDRESS: usize = 0x74;
pub(super) const EPHY_ACCESS: usize = 0x80;
pub(super) const MAC_OCP: usize = 0xb0;
pub(super) const DP_OCP_DATA: usize = 0xb0;
pub(super) const DP_OCP_ACCESS: usize = 0xb4;
pub(super) const DP_MDIO_GATE: usize = 0xd0;
pub(super) const DP_MDIO_GATED: u32 = 1 << 17;
pub(super) const PHY_OCP: usize = 0xb8;
pub(super) const DLL_POWER: usize = 0xd0;
pub(super) const MISC_1: usize = 0xf2;
pub(super) const IBCR0: usize = 0xf8;
pub(super) const IBCR2: usize = 0xf9;
pub(super) const IBISR0: usize = 0xfb;
pub(super) const EFUSE_ACCESS: usize = 0xdc;
pub(super) const DEBUG: usize = 0xd1;
pub(super) const MCU: usize = 0xd3;
pub(super) const RX_MAXIMUM: usize = 0xda;
pub(super) const CPLUS: usize = 0xe0;
pub(super) const INTERRUPT_MITIGATION: usize = 0xe2;
pub(super) const RX_LOW: usize = 0xe4;
pub(super) const RX_HIGH: usize = 0xe8;
pub(super) const EARLY_TX: usize = 0xec;
pub(super) const MISC: usize = 0xf0;
/// Cfg9346 unlock and lock values (Linux `Cfg9346_Unlock`, `Cfg9346_Lock`).
pub(super) const CONFIG_UNLOCKED: u8 = 0xc0;
pub(super) const CONFIG_LOCKED: u8 = 0;
/// RxConfig bits (Linux `RxConfigBits` and `rtl_init_rxcfg`).
pub(super) const RX_FIFO_THRESHOLD: u32 = 7 << 13;
pub(super) const RX_DMA_BURST: u32 = 7 << 8;
pub(super) const RX_128_INTERRUPTS: u32 = 1 << 15;
pub(super) const RX_MULTI_ENABLE: u32 = 1 << 14;
pub(super) const RX_EARLY_OFF: u32 = 1 << 11;
pub(super) const RX_FETCH_8125: u32 = 8 << 27;
/// Accept broadcast, multicast and frames to the station address only.
pub(super) const RX_ACCEPT: u32 = 0x08 | 0x04 | 0x02;
/// TxConfig bits (Linux `TX_DMA_BURST`, `InterFrameGap`, `TXCFG_AUTO_FIFO`).
pub(super) const TX_DMA_BURST: u32 = 7 << 8;
pub(super) const TX_INTER_FRAME_GAP: u32 = 3 << 24;
pub(super) const TX_AUTO_FIFO: u32 = 1 << 7;
/// PHYstatus link bit (Linux `LinkStatus`).
pub(super) const LINK_STATUS: u8 = 0x02;
/// PHYstatus `TBI_Enable`, on the 8169 family alone.
pub(super) const TBI_ENABLE: u8 = 0x80;
/// Clause 22 BMCR restart and BMSR link/autonegotiation-complete bits.
pub(super) const MII_STATUS: u16 = 1;
pub(super) const BMCR_RESTART_AUTONEGOTIATION: u16 = 1 << 9;
pub(super) const BMSR_LINK: u16 = 1 << 2;
pub(super) const BMSR_AUTONEGOTIATION_COMPLETE: u16 = 1 << 5;
pub(super) const RESET: u8 = 0x10;
pub(super) const ENABLE: u8 = 0x0c;
pub(super) const POLL_NORMAL: u8 = 0x40;
pub(super) const ACCESS: u32 = 1 << 31;
pub(super) const OCP_PHY_BASE: u16 = 0xa400;
pub(super) const PHY_PAGE: u16 = 0x1f;
pub(super) const PHY_CONTROL: u16 = 0;
pub(super) const PHY_RESET: u16 = 1 << 15;
pub(super) const PHY_POWER_DOWN: u16 = 1 << 11;
pub(super) const PHY_OCP_EEE_ADV: u16 = 0xa5d0;
pub(super) const MCU_OOB: u8 = 1 << 7;
pub(super) const MCU_TX_EMPTY: u8 = 1 << 5;
pub(super) const MCU_RX_EMPTY: u8 = 1 << 4;
pub(super) const LINK_LIST_READY: u8 = 1 << 1;
pub(super) const TX_FIFO_EMPTY: u32 = 1 << 11;
pub(super) const RXDV_GATED: u32 = 1 << 19;
/// Linux r8169's `NUM_RX_DESC` and `NUM_TX_DESC`.
pub(super) const DESCRIPTORS: usize = 256;
pub(super) const DESCRIPTOR: usize = 16;
pub(super) const BUFFER: usize = 2048;
pub(super) const OWN: u32 = 1 << 31;
pub(super) const END: u32 = 1 << 30;
pub(super) const FIRST: u32 = 1 << 29;
pub(super) const LAST: u32 = 1 << 28;
pub(super) const RX_ERRORS: u32 = 0xf << 19;
pub(super) const LENGTH: u32 = 0x3fff;

// The source extended-XID register lies beyond the normal NBASE timer aperture.
pub(super) const TX_CONFIG_V2: usize = 0x60b0;
pub(super) const EXTENDED_REGISTER_BYTES: u64 = TX_CONFIG_V2 as u64 + 4;
