// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::{
    bluetooth::{bcm4377, intel_pcie},
    bus::pci::Address,
    net::{aqtion, tg3, wifi::realtek as realtek_wifi},
};
use alloc::string::String;

fn identity(vendor: u16, device: u16, class: [u8; 3]) -> Identity {
    Identity {
        address: Address {
            segment: 0,
            bus: 0,
            device: 0,
            function: 0,
        },
        vendor,
        device,
        class,
    }
}

// The original selection order, expressed over a copied header so the
// oracle cannot read configuration space or touch the device.
fn old_driver(function: &Identity) -> Option<Driver> {
    if let Some(kind) = old_native(function) {
        return Some(Driver::Native(kind));
    }
    let attach: legacy::Attach = if virtio::matches_identity(function) {
        legacy::virtio_block
    } else if function.class == nvme::CLASS {
        legacy::nvme
    } else if function.class == xhci::CLASS {
        legacy::xhci
    } else if net::virtio::matches_identity(function) {
        legacy::virtio_net
    } else if net::igc::matches_identity(function) {
        legacy::igc
    } else if net::e1000e::matches_identity(function) {
        legacy::e1000e
    } else if pch::matches_identity(function) {
        legacy::pch
    } else if net::igb::matches_identity(function) {
        legacy::igb
    } else if net::alx::matches_identity(function) {
        legacy::alx
    } else if net::realtek::matches_identity(function) {
        legacy::realtek
    } else if input::virtio::matches_identity(function) {
        legacy::virtio_input
    } else if rtsx::matches_identity(function) {
        legacy::rtsx
    } else if sdhci::matches_identity(function) {
        legacy::sdhci
    } else if ahci::matches_identity(function) {
        legacy::ahci
    } else {
        return None;
    };
    Some(Driver::Legacy(attach))
}

fn old_native(function: &Identity) -> Option<Kind> {
    if vmd::Profile::identify(function.vendor, function.device).is_ok() {
        Some(Kind::Vmd)
    } else if tg3::candidate(function.vendor, function.device, function.class) {
        Some(Kind::Tg3)
    } else if aqtion::Profile::identify(function.vendor, function.device).is_some()
        && function.class == [2, 0, 0]
    {
        Some(Kind::Aqtion)
    } else if (function.vendor, function.device) == (0x1022, 0x15e4) {
        Some(Kind::Sfh)
    } else if (function.vendor, function.device) == (0x8086, 0x9d35) {
        Some(Kind::Ish)
    } else if function.vendor == 0x8086 && function.class[..2] == [2, 0x80] {
        Some(Kind::IntelWifi)
    } else if function.vendor == 0x14c3 && function.class[..2] == [2, 0x80] {
        Some(Kind::MediatekWifi)
    } else if function.vendor == 0x10ec
        && function.class[..2] == [2, 0x80]
        && realtek_wifi::identity::RTL8822C_DEVICES.contains(&function.device)
    {
        Some(Kind::RealtekWifi)
    } else if function.vendor == 0x8086 && intel_pcie::DEVICE_IDS.contains(&function.device) {
        Some(Kind::IntelBluetooth)
    } else if function.vendor == 0x14e4
        && function.class[..2] == [2, 0x80]
        && bcm4377::Chip::of_device(function.device).is_some()
    {
        Some(Kind::Bcm4377)
    } else if audio::hda::pci::identify_identity(function).is_ok() {
        Some(Kind::Hda)
    } else if crate::bus::thunderbolt::controller::Profile::identify(
        function.vendor,
        function.device,
        u32::from_be_bytes([0, function.class[0], function.class[1], function.class[2]]),
    )
    .is_ok()
    {
        Some(Kind::Usb4)
    } else {
        None
    }
}

fn old_claims(function: &Identity) -> bool {
    virtio::matches_identity(function)
        || input::virtio::matches_identity(function)
        || net::virtio::matches_identity(function)
        || net::igc::matches_identity(function)
        || net::e1000e::matches_identity(function)
        || pch::matches_identity(function)
        || net::igb::matches_identity(function)
        || net::alx::matches_identity(function)
        || net::realtek::matches_identity(function)
        || audio::hda::pci::identify_identity(function).is_ok()
        || function.class == nvme::CLASS
        || ahci::matches_identity(function)
        || sdhci::matches_identity(function)
        || rtsx::matches_identity(function)
        || function.class == xhci::CLASS
        || handoff::Kind::of(function.class).is_some()
}

fn check(identity: Identity) {
    let selected = select_pci(&identity, &DriverPolicy::default());
    let exclusive_early =
        identity.class[0] == 3 || handoff::Kind::of(identity.class) == Some(handoff::Kind::Ehci);
    #[cfg(feature = "native-acpi")]
    let exclusive_early = exclusive_early || super::super::smbus::matches_identity(&identity);
    let unsupported_wifi = identity.class[..2] == [2, 0x80]
        && net::wifi::unsupported::reason(identity.vendor, identity.device).is_some();
    let observed = selected.driver();
    let expected = if exclusive_early || unsupported_wifi {
        None
    } else {
        old_driver(&identity)
    };
    let agrees = match (observed, expected) {
        (Some(Driver::Legacy(found)), Some(Driver::Legacy(expected))) => {
            core::ptr::fn_addr_eq(found, expected)
        }
        (Some(Driver::Native(found)), Some(Driver::Native(expected))) => found == expected,
        (None, None) => true,
        _ => false,
    };
    assert!(agrees, "driver differs for {identity:?}");
    assert_eq!(
        selected.attaches_ehci(),
        handoff::Kind::of(identity.class) == Some(handoff::Kind::Ehci),
        "{identity:?}"
    );
    assert_eq!(
        pre_exit_claim(&identity),
        old_claims(&identity),
        "{identity:?}"
    );

    let expected = if identity.class[0] == 3 {
        Some(Early::Display)
    } else {
        #[cfg(feature = "native-acpi")]
        if super::super::smbus::matches_identity(&identity) {
            assert_eq!(early(&identity), Some(Early::Smbus));
            return;
        }
        handoff::Kind::of(identity.class).map(Early::Usb)
    };
    assert_eq!(early(&identity), expected, "{identity:?}");
}

#[test]
fn original_probe_and_pre_exit_admission_agree() {
    // Check real IDs with their normal classes and with overlapping classes.
    // The old probe prioritizes identity-only routes in some of these cases.
    for (vendor, device) in [
        (0x8086, 0x9a0b), // VMD
        (0x14e4, 0x165f), // Tigon3
        (0x1d6a, 0x07b1), // Aquantia
        (0x1022, 0x15e4), // SFH
        (0x8086, 0x9d35), // ISH
        (0x8086, 0x31dc), // Unsupported CNVi before the broad Intel match
        (0x8086, 0x2723), // AX200
        (0x14c3, 0x7925), // MediaTek
        (0x10ec, 0xc822), // Realtek Wi-Fi
        (0x8086, 0xa876), // PCIe Bluetooth
        (0x14e4, 0x5fa0), // BCM4377
        (0x8086, 0xa170), // HDA
        (0x8086, 0x1137), // Thunderbolt
        (0x8086, 0x1c22), // SMBus
        (0x1af4, 0x1001), // Virtio block
        (0x1af4, 0x1041), // Virtio net
        (0x1af4, 0x1052), // Virtio input
        (0x8086, 0x15f3), // igc
        (0x8086, 0x10d3), // e1000e
        (0x8086, 0x1502), // PCH
        (0x8086, 0x1533), // igb
        (0x1969, 0x1091), // alx
        (0x10ec, 0x8168), // Realtek Ethernet
        (0x10ec, 0x5229), // Realtek card reader
        (0xffff, 0xffff), // Class-only matches and unrelated functions
    ] {
        for class in [
            [0, 0, 0],
            [1, 4, 0],
            [1, 6, 1],
            [1, 8, 2],
            [2, 0, 0],
            [2, 0x80, 0],
            [3, 0, 0],
            [4, 3, 0],
            [8, 5, 0],
            [8, 5, 1],
            [0x0c, 3, 0],
            [0x0c, 3, 0x10],
            [0x0c, 3, 0x20],
            [0x0c, 3, 0x30],
            [0x0c, 5, 0],
            [0x0c, 0x0a, 0],
            [0x0d, 0x11, 0],
            [0xff, 0, 0],
        ] {
            check(identity(vendor, device, class));
        }
    }

    let mut sample = 0x25e4_1696_3909_ea14_u64;
    for _ in 0..50_000 {
        sample ^= sample << 13;
        sample ^= sample >> 7;
        sample ^= sample << 17;
        let [a, b, c, d, e, f, g, _] = sample.to_le_bytes();
        check(identity(
            u16::from_le_bytes([a, b]),
            u16::from_le_bytes([c, d]),
            [e, f, g],
        ));
    }
}

#[test]
fn driver_policy_keeps_identity_validation_and_priority() {
    let nic = identity(0x8086, 0x15f3, [2, 0, 0]);
    let virtio = identity(0x1af4, 0x1001, [0, 0, 0]);
    assert!(
        matches!(resolve_pci(&nic, &DriverPolicy::default()), Resolution::Candidate(m) if m.id == "IntelEth")
    );
    let safe = DriverPolicy {
        mode: Mode::Safe,
        ..DriverPolicy::default()
    };
    assert!(matches!(resolve_pci(&nic, &safe), Resolution::Disabled(m) if m.id == "IntelEth"));
    assert!(
        matches!(resolve_pci(&virtio, &safe), Resolution::Candidate(m) if m.id == "VirtioBlock")
    );
    let enable = [String::from("IntelEth")];
    let manual = DriverPolicy {
        mode: Mode::Manual,
        enable: &enable,
        ..DriverPolicy::default()
    };
    assert!(matches!(resolve_pci(&nic, &manual), Resolution::Candidate(m) if m.id == "IntelEth"));
    assert!(
        matches!(resolve_pci(&virtio, &manual), Resolution::Disabled(m) if m.id == "VirtioBlock")
    );

    let pci = PciId::parse("pci:8086:15F3").unwrap();
    let force = [ForcedDriver {
        pci,
        module: "RealtekEth",
    }];
    let policy = DriverPolicy {
        force: &force,
        ..DriverPolicy::default()
    };
    assert_eq!(resolve_pci(&nic, &policy), Resolution::ForceRejected);
    let block = [pci];
    assert_eq!(
        resolve_pci(
            &nic,
            &DriverPolicy {
                block: &block,
                ..policy
            }
        ),
        Resolution::Blocked
    );
    assert!(PciId::parse("pci:+123:15f3").is_none());
    assert!(PciId::parse("pci:8086:15f3:0000").is_none());
    assert!(
        matches!(resolve_pci(&identity(0x8086, 0x31dc, [2, 0x80, 0]), &DriverPolicy::default()), Resolution::Unsupported(why) if why.contains("9000 series"))
    );
}

#[test]
fn nvmodern_metadata_matches_nvidia_displays_only() {
    let display = identity(0x10de, 0x2684, [3, 0, 0]);
    assert!(matches!(
        resolve_pci(&display, &DriverPolicy::default()),
        Resolution::Candidate(manifest) if manifest.id == "NVModern"
    ));
    assert_eq!(early(&display), Some(Early::Display));
    assert!(
        select_pci(&display, &DriverPolicy::default())
            .driver()
            .is_none()
    );
    assert!(!pre_exit_claim(&display));

    let audio = identity(0x10de, 0x22ba, [4, 3, 0]);
    let intel = identity(0x8086, 0x3185, [3, 0, 0]);
    assert!(pci_candidates(&audio).all(|manifest| manifest.id != "NVModern"));
    assert!(pci_candidates(&intel).all(|manifest| manifest.id != "NVModern"));
}

#[test]
fn policy_gates_host_drivers_and_keeps_inventory_stages() {
    let ehci = identity(0x8086, 0x1c26, [0x0c, 3, 0x20]);
    let xhci = identity(0x8086, 0x31a8, [0x0c, 3, 0x30]);
    let nvidia = gpu::Adapter {
        address: ehci.address,
        vendor: gpu::Vendor::Nvidia,
        device: 0x2684,
        class: [3, 0, 0],
        glk_memory: None,
    };
    let disable = [
        String::from("EHCI"),
        String::from("XHCI"),
        String::from("NVModern"),
    ];
    let disabled = DriverPolicy {
        disable: &disable,
        ..DriverPolicy::default()
    };
    assert!(select_pci(&ehci, &DriverPolicy::default()).attaches_ehci());
    assert!(!select_pci(&ehci, &disabled).attaches_ehci());
    assert!(select_pci(&xhci, &disabled).driver().is_none());
    assert_eq!(early(&ehci), Some(Early::Usb(handoff::Kind::Ehci)));
    assert_eq!(early(&xhci), Some(Early::Usb(handoff::Kind::Xhci)));
    assert!(DriverPolicy::default().allows_nvmodern(&nvidia));
    assert!(matches!(
        disabled.admit_nvmodern(&nvidia),
        Err(Resolution::Disabled(manifest)) if manifest.id == "NVModern"
    ));

    let block = [PciId {
        vendor: 0x10de,
        device: nvidia.device,
    }];
    let blocked = DriverPolicy {
        block: &block,
        ..DriverPolicy::default()
    };
    assert_eq!(blocked.admit_nvmodern(&nvidia), Err(Resolution::Blocked));

    let enable = [String::from("NVModern")];
    let force = [ForcedDriver {
        pci: block[0],
        module: "NVModern",
    }];
    let manual = DriverPolicy {
        mode: Mode::Manual,
        enable: &enable,
        force: &force,
        ..DriverPolicy::default()
    };
    assert!(manual.allows_nvmodern(&nvidia));
    assert_eq!(
        DriverPolicy {
            disable: &disable,
            ..manual
        }
        .admit_nvmodern(&nvidia),
        Err(Resolution::ForceRejected)
    );
    assert!(!manual.allows_nvmodern(&gpu::Adapter {
        vendor: gpu::Vendor::Intel,
        ..nvidia
    }));
}
