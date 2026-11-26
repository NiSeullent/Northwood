// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2002 ShuChen <shuchen@realtek.com.tw>
// Copyright (c) 2003-2007 Francois Romieu <romieu@fr.zoreil.com>
// Copyright (c) a lot of people too. Please respect their work.
// Copyright (c) 2026 Stela contributors
//! Ordered MAC-version lookup, adapted from Linux's rtl_chip_infos.
//! Linux's symbolic revision suffix is held, rather than its enum ordinal.

/// Detected MAC revision; this does not imply the startup path is implemented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Revision {
    pub version: u8,
    pub name: &'static str,
    pub firmware: Option<&'static str>,
    pub xid: u16,
    /// TX_CONFIG_V2 word, present only for the primary extended-XID marker.
    pub extended_xid: Option<u32>,
}

impl Revision {
    pub fn implemented(self) -> bool {
        matches!(self.version, 2..=6 | 17..=26 | 28 | 31..=36 | 38 | 40 | 42 | 44 | 46 | 51 | 52 | 61 | 63..=66 | 70)
    }

    /// Pinned donor classification, kept separate from the raw firmware identity.
    pub fn extended_classification(self) -> Option<u32> {
        self.extended_xid.map(|raw| raw & 0x7fff_ffff)
    }
    pub(super) fn gigabit_pcie(self) -> bool {
        self.version > 6 && self.version < 61
    }
    pub(super) fn modern_phy(self) -> bool {
        self.version >= 40
    }
    pub(super) fn nbase(self) -> bool {
        self.version >= 61
    }
}

struct Entry(u16, u16, u8, &'static str, Option<&'static str>);
const fn fw(name: &'static str) -> Option<&'static str> {
    Some(name)
}
const ENTRIES: &[Entry] = &[
    Entry(0x7cf, 0x64a, 70, "RTL8126A", fw("rtl_nic/rtl8126a-3.fw")),
    Entry(0x7cf, 0x649, 70, "RTL8126A", fw("rtl_nic/rtl8126a-2.fw")),
    Entry(0x7cf, 0x681, 66, "RTL8125BP", fw("rtl_nic/rtl8125bp-2.fw")),
    Entry(0x7cf, 0x708, 65, "RTL8125CP", fw("rtl_nic/rtl8125cp-1.fw")),
    Entry(0x7cf, 0x68b, 64, "RTL9151A", fw("rtl_nic/rtl9151a-1.fw")),
    Entry(0x7cf, 0x68a, 64, "RTL8125K", fw("rtl_nic/rtl8125k-1.fw")),
    Entry(0x7cf, 0x689, 64, "RTL8125D", fw("rtl_nic/rtl8125d-2.fw")),
    Entry(0x7cf, 0x688, 64, "RTL8125D", fw("rtl_nic/rtl8125d-1.fw")),
    Entry(0x7cf, 0x641, 63, "RTL8125B", fw("rtl_nic/rtl8125b-2.fw")),
    Entry(0x7cf, 0x609, 61, "RTL8125A", fw("rtl_nic/rtl8125a-3.fw")),
    Entry(0x7cf, 0x54b, 52, "RTL8168FP/RTL8117", None),
    Entry(
        0x7cf,
        0x54a,
        52,
        "RTL8168FP/RTL8117",
        fw("rtl_nic/rtl8168fp-3.fw"),
    ),
    Entry(0x7cf, 0x502, 51, "RTL8168EP/RTL8111EP", None),
    Entry(
        0x7cf,
        0x541,
        46,
        "RTL8168H/RTL8111H",
        fw("rtl_nic/rtl8168h-2.fw"),
    ),
    Entry(
        0x7cf,
        0x6c0,
        46,
        "RTL8168M/RTL8111M",
        fw("rtl_nic/rtl8168h-2.fw"),
    ),
    Entry(0x7cf, 0x5c8, 44, "RTL8411B", fw("rtl_nic/rtl8411-2.fw")),
    Entry(
        0x7cf,
        0x509,
        42,
        "RTL8168G/RTL8111G",
        fw("rtl_nic/rtl8168g-3.fw"),
    ),
    Entry(
        0x7cf,
        0x4c0,
        40,
        "RTL8168G/RTL8111G",
        fw("rtl_nic/rtl8168g-2.fw"),
    ),
    Entry(0x7c8, 0x488, 38, "RTL8411", fw("rtl_nic/rtl8411-1.fw")),
    Entry(
        0x7cf,
        0x481,
        36,
        "RTL8168F/RTL8111F",
        fw("rtl_nic/rtl8168f-2.fw"),
    ),
    Entry(
        0x7cf,
        0x480,
        35,
        "RTL8168F/RTL8111F",
        fw("rtl_nic/rtl8168f-1.fw"),
    ),
    Entry(
        0x7c8,
        0x2c8,
        34,
        "RTL8168E-VL/RTL8111E-VL",
        fw("rtl_nic/rtl8168e-3.fw"),
    ),
    Entry(
        0x7cf,
        0x2c1,
        32,
        "RTL8168E/RTL8111E",
        fw("rtl_nic/rtl8168e-1.fw"),
    ),
    Entry(
        0x7c8,
        0x2c0,
        33,
        "RTL8168E/RTL8111E",
        fw("rtl_nic/rtl8168e-2.fw"),
    ),
    Entry(
        0x7cf,
        0x281,
        25,
        "RTL8168D/RTL8111D",
        fw("rtl_nic/rtl8168d-1.fw"),
    ),
    Entry(
        0x7c8,
        0x280,
        26,
        "RTL8168D/RTL8111D",
        fw("rtl_nic/rtl8168d-2.fw"),
    ),
    Entry(0x7cf, 0x28a, 28, "RTL8168DP/RTL8111DP", None),
    Entry(0x7cf, 0x28b, 31, "RTL8168DP/RTL8111DP", None),
    Entry(0x7cf, 0x3c9, 23, "RTL8168CP/RTL8111CP", None),
    Entry(0x7cf, 0x3c8, 18, "RTL8168CP/RTL8111CP", None),
    Entry(0x7c8, 0x3c8, 24, "RTL8168CP/RTL8111CP", None),
    Entry(0x7cf, 0x3c0, 19, "RTL8168C/RTL8111C", None),
    Entry(0x7cf, 0x3c2, 20, "RTL8168C/RTL8111C", None),
    Entry(0x7cf, 0x3c3, 21, "RTL8168C/RTL8111C", None),
    Entry(0x7c8, 0x3c0, 22, "RTL8168C/RTL8111C", None),
    Entry(0x7c8, 0x380, 17, "RTL8168B/RTL8111B", None),
    Entry(0xfc8, 0x980, 6, "RTL8169SC/RTL8110SC", None),
    Entry(0xfc8, 0x180, 5, "RTL8169SC/RTL8110SC", None),
    Entry(0xfc8, 0x100, 4, "RTL8169SB/RTL8110SB", None),
    Entry(0xfc8, 0x040, 3, "RTL8110S", None),
    Entry(0xfc8, 0x008, 2, "RTL8169S", None),
];

pub(super) fn detect(txconfig: u32) -> Option<Revision> {
    if txconfig == u32::MAX {
        return None;
    }
    let xid = ((txconfig >> 20) & 0xfcf) as u16;
    ENTRIES.iter().find(|e| xid & e.0 == e.1).map(|e| Revision {
        version: e.2,
        name: e.3,
        firmware: e.4,
        xid,
        extended_xid: None,
    })
}

// Same-vendor aliases use the pinned donor's common XID dispatch; they are
// candidates, not a claim that any unknown or unsupported revision can run.
const PCI_ALIASES: &[u16] = &[
    0x2502, 0x2600, 0x8129, 0x8162, 0x8167, 0x3000, 0x5000, 0x0e10,
];
/// The Realtek ID whose family rules apply to this PCI function. The
/// rebadged boards are listed in Linux's r8169 table without subsystem IDs.
pub(super) fn pci_device(vendor: u16, device: u16) -> Option<u16> {
    match (vendor, device) {
        (0x10ec, 0x8161 | 0x8168 | 0x8169 | 0x8125 | 0x8126) => Some(device),
        (0x10ec, alias) if PCI_ALIASES.contains(&alias) => Some(alias),
        (0x1186, 0x4300 | 0x4302) | (0x1259, 0xc107) | (0x16ec, 0x0116) => Some(0x8167),
        (0x10ff, 0x8168) => Some(0x8168),
        _ => None,
    }
}

pub(super) fn pci_family(device: u16, revision: Revision) -> bool {
    match device {
        0x8169 => revision.version <= 6,
        0x8168 | 0x8161 => revision.gigabit_pcie(),
        0x8125 => matches!(revision.version, 61 | 63 | 64 | 65 | 66),
        0x8126 => revision.version == 70,
        alias if PCI_ALIASES.contains(&alias) => revision.implemented(),
        _ => false,
    }
}

/// The primary source table delegates this exact marker to TX_CONFIG_V2.
pub(super) fn needs_extended(txconfig: u32) -> bool {
    txconfig != u32::MAX && ((txconfig >> 20) & 0x7cf) == 0x7c8
}

pub(super) fn detect_extended(txconfig: u32, raw: u32) -> Option<Revision> {
    if !needs_extended(txconfig) || raw & 0x7fff_ffff != 0 {
        return None;
    }
    Some(Revision {
        version: 64,
        name: "RTL9151AS",
        firmware: fw("rtl_nic/rtl9151a-1.fw"),
        xid: ((txconfig >> 20) & 0xfcf) as u16,
        extended_xid: Some(raw),
    })
}
