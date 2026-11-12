// SPDX-License-Identifier: GPL-2.0-or-later
//! Static PCI registrations over the copied discovery headers.
use super::{legacy, native::Kind, pch};
use crate::{
    audio, bluetooth,
    bus::{pci::Identity, thunderbolt, vmd},
    gpu, input, net,
    storage::{ahci, nvme, rtsx, sdhci, virtio},
    usb::{handoff, xhci},
};

mod policy;
pub use policy::{DriverPolicy, ForcedDriver, Mode, PciId, Resolution, resolve_pci};
pub(super) use policy::{Selection, select_pci};

/// The host subsystem served by a registered module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Storage,
    Ethernet,
    Wifi,
    Bluetooth,
    Audio,
    Input,
    UsbHost,
    Gpu,
    Platform,
}

/// What a registration establishes about an implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// An existing attachment implementation with no verification claim here.
    Implemented,
    /// Inventory, identification or firmware handoff only.
    CheckOnly,
}

/// Revision of the statically linked driver manifest interface.
pub const ABI: (u16, u16) = (1, 0);

/// Licensing of statically linked registration source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LicenseTier {
    /// Permissive terms are available.
    A,
    /// Copyleft terms apply to linked source.
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct License {
    /// SPDX expression from the primary implementation's source notice.
    pub spdx: &'static str,
    pub tier: LicenseTier,
}

/// How to determine the external firmware needed for an admitted device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Firmware {
    None,
    /// File names are stable for this registration.
    Required(&'static [&'static str]),
    /// The named profile must select images after matching or reading the device.
    ProfileSelected(&'static str),
}

/// Metadata shared by boot probing and host-side inventory tools.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Manifest {
    pub id: &'static str,
    pub implementation: &'static str,
    pub class: Class,
    pub status: Status,
    /// Source implementation in this tree. NOTICE records donor attribution.
    pub source: &'static str,
    pub license: License,
    pub firmware: Firmware,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Early {
    Display,
    #[cfg(feature = "native-acpi")]
    Smbus,
    Usb(handoff::Kind),
}

#[derive(Clone, Copy)]
pub(super) enum Driver {
    Native(Kind),
    Legacy(legacy::Attach),
}

#[derive(Clone, Copy)]
enum Route {
    Early(Early),
    Driver(Driver),
    AfterScan,
}

pub(super) struct Entry {
    pub(super) manifest: Manifest,
    matches: fn(&Identity) -> bool,
    pre_exit_claim: bool,
    route: Route,
}

/// Rows are ordered by the original probe priority within each scan stage.
/// USB firmware handoff precedes driver attachment, including xHCI.
macro_rules! entry {
    (@firmware [$($required:literal),*]) => { Firmware::Required(&[$($required),*]) };
    (@firmware $profile:literal) => { Firmware::ProfileSelected($profile) };
    (@firmware none) => { Firmware::None };
    ($id:literal, $implementation:literal, $class:ident, $status:ident, $claim:literal, $matches:expr, $route:expr; $source:literal, $tier:ident $spdx:literal, $firmware:tt) => {
        Entry {
            manifest: Manifest {
                id: $id,
                implementation: $implementation,
                class: Class::$class,
                status: Status::$status,
                source: $source,
                license: License { spdx: $spdx, tier: LicenseTier::$tier },
                firmware: entry!(@firmware $firmware),
            },
            matches: $matches,
            pre_exit_claim: $claim,
            route: $route,
        }
    };
    ($id:literal, $implementation:literal, $class:ident, $status:ident, $claim:literal, $matches:expr, native $kind:ident; $source:literal, $tier:ident $spdx:literal, $firmware:tt) => {
        entry!(
            $id,
            $implementation,
            $class,
            $status,
            $claim,
            $matches,
            Route::Driver(Driver::Native(Kind::$kind)); $source, $tier $spdx, $firmware
        )
    };
    ($id:literal, $implementation:literal, $class:ident, $matches:expr, legacy $attach:ident; $source:literal, $tier:ident $spdx:literal, $firmware:tt) => {
        entry!(
            $id,
            $implementation,
            $class,
            Implemented,
            true,
            $matches,
            Route::Driver(Driver::Legacy(legacy::$attach)); $source, $tier $spdx, $firmware
        )
    };
}

static REGISTRY: &[Entry] = &[
    entry!(
        "DisplayInventory",
        "pci-display",
        Gpu,
        CheckOnly,
        false,
        |id| id.class[0] == 0x03,
        Route::Early(Early::Display);
        "crates/stela-driver-gpu/src/lib.rs", B "GPL-2.0-or-later", none
    ),
    entry!(
        "NVModern",
        "nouveau-nvk",
        Gpu,
        Implemented,
        false,
        |id| id.class[0] == 0x03 && gpu::Vendor::from_pci(id.vendor) == gpu::Vendor::Nvidia,
        Route::AfterScan;
        "third_party/linux/drivers/gpu/drm/nouveau/nvkm/engine/device/base.c", A "MIT",
        "nouveau PMC_BOOT_0 chipset and pinned GSP-RM recipes"
    ),
    #[cfg(feature = "native-acpi")]
    entry!(
        "IntelSMBus",
        "i801",
        Platform,
        Implemented,
        false,
        super::smbus::matches_identity,
        Route::Early(Early::Smbus);
        "crates/stela-driver-base/src/bus/smbus/i801.rs", B "GPL-2.0-or-later", none
    ),
    entry!(
        "UHCI",
        "uhci-handoff",
        UsbHost,
        CheckOnly,
        true,
        |id| handoff::Kind::of(id.class) == Some(handoff::Kind::Uhci),
        Route::Early(Early::Usb(handoff::Kind::Uhci));
        "crates/stela-drivers/src/usb/handoff.rs", B "GPL-2.0-only", none
    ),
    entry!(
        "OHCI",
        "ohci-handoff",
        UsbHost,
        CheckOnly,
        true,
        |id| handoff::Kind::of(id.class) == Some(handoff::Kind::Ohci),
        Route::Early(Early::Usb(handoff::Kind::Ohci));
        "crates/stela-drivers/src/usb/handoff.rs", B "GPL-2.0-only", none
    ),
    entry!(
        "EHCI",
        "ehci",
        UsbHost,
        Implemented,
        true,
        |id| handoff::Kind::of(id.class) == Some(handoff::Kind::Ehci),
        Route::Early(Early::Usb(handoff::Kind::Ehci));
        "crates/stela-drivers/src/usb/ehci.rs", B "GPL-2.0-or-later", none
    ),
    entry!(
        "XHCI",
        "xhci-handoff",
        UsbHost,
        CheckOnly,
        true,
        |id| handoff::Kind::of(id.class) == Some(handoff::Kind::Xhci),
        Route::Early(Early::Usb(handoff::Kind::Xhci));
        "crates/stela-drivers/src/usb/handoff.rs", B "GPL-2.0-only", none
    ),
    entry!("IntelVMD", "vmd", Storage, Implemented, false, |id| vmd::Profile::identify(id.vendor, id.device).is_ok(), native Vmd;
        "crates/stela-driver-base/src/bus/vmd.rs", B "GPL-2.0-only", none
    ),
    entry!("BroadcomEth", "tg3", Ethernet, Implemented, false, |id| net::tg3::candidate(id.vendor, id.device, id.class), native Tg3;
        "crates/stela-driver-net/src/tg3.rs", B "GPL-2.0-only", none
    ),
    entry!("AquantiaEth", "atlantic", Ethernet, Implemented, false, |id| net::aqtion::Profile::identify(id.vendor, id.device).is_some() && id.class == [2, 0, 0], native Aqtion;
        "crates/stela-driver-net/src/aqtion.rs", B "GPL-2.0-only", none
    ),
    entry!("AMDSFH", "amd-sfh", Input, CheckOnly, false, |id| (id.vendor, id.device) == (0x1022, 0x15e4), native Sfh;
        "crates/stela-drivers/src/input/sfh.rs", B "GPL-2.0-or-later", none
    ),
    entry!("IntelISH", "intel-ish", Input, CheckOnly, false, |id| (id.vendor, id.device) == (0x8086, 0x9d35), native Ish;
        "crates/stela-drivers/src/input/ish.rs", B "GPL-2.0-only", none
    ),
    entry!("IntelWiFi", "iwlwifi", Wifi, Implemented, false, |id| wifi_vendor(id, 0x8086), native IntelWifi;
        "crates/stela-driver-net/src/wifi/intel.rs", A "GPL-2.0-only OR BSD-3-Clause", "iwlwifi family, board and observed RF identity; wireless-regdb"
    ),
    entry!("MediaTekWiFi", "mt7925e", Wifi, Implemented, false, |id| wifi_vendor(id, 0x14c3), native MediatekWifi;
        "crates/stela-driver-net/src/wifi/mediatek.rs", A "BSD-3-Clause-Clear", ["mediatek/mt7925/WIFI_MT7925_PATCH_MCU_1_1_hdr.bin", "mediatek/mt7925/WIFI_RAM_CODE_MT7925_1_1.bin", "regulatory.db"]
    ),
    entry!("RealtekWiFi", "rtw88_8822ce", Wifi, Implemented, false, |id| wifi_vendor(id, 0x10ec) && net::wifi::realtek::identity::RTL8822C_DEVICES.contains(&id.device), native RealtekWifi;
        "crates/stela-driver-net/src/wifi/realtek.rs", A "GPL-2.0-only OR BSD-3-Clause", ["rtw88/rtw8822c_fw.bin", "regulatory.db"]
    ),
    entry!("IntelBT", "btintel_pcie", Bluetooth, Implemented, false, |id| id.vendor == 0x8086 && bluetooth::intel_pcie::DEVICE_IDS.contains(&id.device), native IntelBluetooth;
        "crates/stela-drivers/src/bluetooth/intel_pcie.rs", B "GPL-2.0-or-later", "btintel PCIe controller identity"
    ),
    entry!("BroadcomBT", "hci_bcm4377", Bluetooth, Implemented, false, |id| wifi_vendor(id, 0x14e4) && bluetooth::bcm4377::Chip::of_device(id.device).is_some(), native Bcm4377;
        "crates/stela-drivers/src/bluetooth/bcm4377.rs", A "MIT", "BCM4377 chip, board and controller identity"
    ),
    entry!("HDA", "snd_hda_intel", Audio, Implemented, true, |id| audio::hda::pci::identify_identity(id).is_ok(), native Hda;
        "crates/stela-driver-audio/src/hda/pci.rs", B "GPL-2.0-only", none
    ),
    entry!("USB4", "thunderbolt", UsbHost, Implemented, false, |id| thunderbolt::controller::Profile::identify(id.vendor, id.device, u32::from_be_bytes([0, id.class[0], id.class[1], id.class[2]])).is_ok(), native Usb4;
        "crates/stela-driver-base/src/bus/thunderbolt.rs", B "GPL-2.0-only", none
    ),
    entry!("VirtioBlock", "virtio-blk", Storage, virtio::matches_identity, legacy virtio_block;
        "crates/stela-driver-storage/src/virtio.rs", B "GPL-2.0-or-later", none
    ),
    entry!("NVMe", "nvme", Storage, |id| id.class == nvme::CLASS, legacy nvme;
        "crates/stela-driver-storage/src/nvme.rs", B "GPL-2.0-only", none
    ),
    entry!("XHCI", "xhci", UsbHost, |id| id.class == xhci::CLASS, legacy xhci;
        "crates/stela-drivers/src/usb/xhci.rs", B "GPL-2.0-or-later", none
    ),
    entry!("VirtioNet", "virtio-net", Ethernet, net::virtio::matches_identity, legacy virtio_net;
        "crates/stela-driver-net/src/virtio.rs", B "GPL-2.0-or-later", none
    ),
    entry!("IntelEth", "igc", Ethernet, net::igc::matches_identity, legacy igc;
        "crates/stela-driver-net/src/igc.rs", B "GPL-2.0-only", none
    ),
    entry!("IntelEth", "e1000e", Ethernet, net::e1000e::matches_identity, legacy e1000e;
        "crates/stela-driver-net/src/e1000e.rs", B "BSD-3-Clause AND GPL-2.0-only", none
    ),
    entry!("IntelEth", "e1000e-pch", Ethernet, pch::matches_identity, legacy pch;
        "crates/stela-driver-net/src/e1000e/pch/cold/attach.rs", B "GPL-2.0-only", none
    ),
    entry!("IntelEth", "igb", Ethernet, net::igb::matches_identity, legacy igb;
        "crates/stela-driver-net/src/igb.rs", B "GPL-2.0-only", none
    ),
    entry!("AtherosEth", "alx", Ethernet, net::alx::matches_identity, legacy alx;
        "crates/stela-driver-net/src/alx.rs", B "GPL-2.0-or-later AND ISC", none
    ),
    entry!("AtherosEth", "atl1c", Ethernet, net::atl1c::matches_identity, legacy atl1c;
        "crates/stela-driver-net/src/atl1c.rs", B "GPL-2.0-or-later", none
    ),
    entry!("RealtekEth", "r8169", Ethernet, net::realtek::matches_identity, legacy realtek;
        "crates/stela-driver-net/src/realtek.rs", B "GPL-2.0-only AND GPL-2.0-or-later", "r8169 MAC revision"
    ),
    entry!("VirtioInput", "virtio-input", Input, input::virtio::matches_identity, legacy virtio_input;
        "crates/stela-drivers/src/input/virtio.rs", B "GPL-2.0-or-later", none
    ),
    entry!("RtsxReader", "rtsx_pci", Storage, rtsx::matches_identity, legacy rtsx;
        "crates/stela-driver-storage/src/rtsx.rs", B "GPL-2.0-or-later", none
    ),
    entry!("SDHCI", "sdhci", Storage, sdhci::matches_identity, legacy sdhci;
        "crates/stela-driver-storage/src/sdhci.rs", B "GPL-2.0-or-later", none
    ),
    entry!("AHCI", "ahci", Storage, ahci::matches_identity, legacy ahci;
        "crates/stela-driver-storage/src/ahci.rs", B "GPL-2.0-or-later", none
    ),
];

fn wifi_vendor(identity: &Identity, vendor: u16) -> bool {
    identity.vendor == vendor && identity.class[..2] == [2, 0x80]
}

/// All compiled registration rows, including separate phases and variants
/// of a module. No match or attached-device state is implied.
pub fn manifests() -> impl Iterator<Item = &'static Manifest> {
    REGISTRY.iter().map(|entry| &entry.manifest)
}

/// Candidate rows from the same copied PCI identity used at boot.
/// The iterator includes handoff rows and overlaps, in scan priority order.
/// It establishes neither attachment nor device verification.
pub fn pci_candidates(identity: &Identity) -> impl Iterator<Item = &'static Manifest> + '_ {
    REGISTRY
        .iter()
        .filter(move |entry| (entry.matches)(identity))
        .map(|entry| &entry.manifest)
}

pub(super) fn early(identity: &Identity) -> Option<Early> {
    REGISTRY.iter().find_map(|entry| match entry.route {
        Route::Early(action) if (entry.matches)(identity) => Some(action),
        _ => None,
    })
}

pub(super) fn pre_exit_claim(identity: &Identity) -> bool {
    // This pass intentionally checks any match, even if the scan later
    // prefers another row. Some firmware owners rely on those BAR mappings.
    REGISTRY
        .iter()
        .any(|entry| entry.pre_exit_claim && (entry.matches)(identity))
}

#[cfg(test)]
mod tests;
