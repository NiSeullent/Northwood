// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026 Stela contributors
//! Admission cases.
use super::*;

#[test]
fn phy_status_bit_7_is_tbi_only_on_the_8169_family() {
    let at = |xid: u32| revision::detect(xid << 20).unwrap();
    // RTL8169SC (version 06): the bit is its TBI interface.
    assert!(tbi_mode(at(0x980), 0x80));
    assert!(!tbi_mode(at(0x980), 0x7f));
    // RTL8111H/8168M (version 46) and RTL8168G (40) set it with copper.
    for xid in [0x541, 0x6c0, 0x4c0] {
        assert!(!tbi_mode(at(xid), 0x80 | 0x13), "{xid:03x}");
    }
}

#[test]
fn exact_ordered_revision_fixtures_refuse_unknown_ancient_and_wrong_pci_family() {
    // Independent source-table cases include overlapping specific/generic masks.
    for (xid, version, fw) in [
        (0x3c8, 18, None),
        (0x3c9, 23, None),
        (0x3ca, 24, None),
        (0x3c0, 19, None),
        (0x3c2, 20, None),
        (0x3c3, 21, None),
        (0x3c4, 22, None),
        (0x281, 25, Some("rtl_nic/rtl8168d-1.fw")),
        (0x280, 26, Some("rtl_nic/rtl8168d-2.fw")),
        (0x2c1, 32, Some("rtl_nic/rtl8168e-1.fw")),
        (0x2c0, 33, Some("rtl_nic/rtl8168e-2.fw")),
        (0x2c8, 34, Some("rtl_nic/rtl8168e-3.fw")),
        (0x641, 63, Some("rtl_nic/rtl8125b-2.fw")),
        (0x609, 61, Some("rtl_nic/rtl8125a-3.fw")),
        (0x688, 64, Some("rtl_nic/rtl8125d-1.fw")),
        (0x689, 64, Some("rtl_nic/rtl8125d-2.fw")),
        (0x68a, 64, Some("rtl_nic/rtl8125k-1.fw")),
        (0x68b, 64, Some("rtl_nic/rtl9151a-1.fw")),
        (0x708, 65, Some("rtl_nic/rtl8125cp-1.fw")),
        (0x681, 66, Some("rtl_nic/rtl8125bp-2.fw")),
        (0x649, 70, Some("rtl_nic/rtl8126a-2.fw")),
        (0x64a, 70, Some("rtl_nic/rtl8126a-3.fw")),
        (0x980, 6, None),
    ] {
        let r = revision::detect(xid << 20).unwrap();
        assert_eq!((r.version, r.firmware), (version, fw));
        assert_eq!(u32::from(r.xid), xid);
        if version >= 61 && !matches!(version, 61 | 63..=66 | 70) {
            assert!(!r.implemented());
        }
    }
    for xid in [0u32, 0x300, 0x7c8, 0x6c9, 0xfcf] {
        assert!(revision::detect(xid << 20).is_none());
    }
    assert!(revision::detect(u32::MAX).is_none());
    assert!(revision::pci_family(0x8169, revision(2)));
    assert!(!revision::pci_family(0x8168, revision(2)));
    assert!(!revision::pci_family(0x8125, revision(34)));
}

#[test]
fn exact_phy_id_and_capability_validation_has_no_synthetic_success() {
    for (reg, value) in [(2, 0xffff), (3, 0xc910), (1, 0), (15, 0)] {
        let time = Time::default();
        let model = Model::ready(&time);
        model.state.borrow_mut().phy.insert((0, reg), value);
        let mut hw = Hardware::new(&model, &time, revision(18), 0);
        assert!(hw.configure_phy(None, false).is_err());
        assert!(!model.state.borrow().phy.contains_key(&(2, 0x0c)));
    }
}

#[test]
fn station_mac_rejects_multicast_zero_and_broadcast() {
    assert_eq!(station(0, 0), None);
    assert_eq!(station(1, 0), None);
    assert_eq!(station(u32::MAX, u32::MAX), None);
    assert_eq!(station(0x0504_0302, 0x0706), Some(Mac([2, 3, 4, 5, 6, 7])));
}

#[test]
fn generic_phy_identical_id_nbase_variant_is_refused_before_analogue_programming() {
    let time = Time::default();
    let model = Model::ready(&time);
    model.state.borrow_mut().phy.insert((0xa61, 0x13), 0x2000);
    let mut hw = Hardware::new(&model, &time, revision(18), 0);
    assert_eq!(
        hw.configure_phy(None, false),
        Err(Error::UnsupportedPhy(0x001cc800))
    );
    assert_eq!(model.state.borrow().page, 0);
    assert!(!model.state.borrow().phy.contains_key(&(2, 0xc)));
}

#[test]
fn firmware_catalog_binding_requires_exact_pci_xid_board_and_filename() {
    use resources::{Catalog, ErrorKind, Provenance, Resource, Target};
    let bytes = 0x801f0000u32.to_le_bytes().to_vec();
    // Fixed SHA-256 fixture; identity tests do not derive their expected digest
    // through the same Catalog validation implementation.
    let catalog_digest = [
        0x23, 0xc3, 0xcf, 0xd9, 0x9c, 0xa0, 0x28, 0xa0, 0xba, 0x72, 0xe8, 0x92, 0x40, 0xe4, 0xb4,
        0xf8, 0x93, 0xc1, 0xb4, 0x26, 0xa2, 0x1a, 0x6e, 0x89, 0x36, 0xfd, 0x43, 0x13, 0x6b, 0x13,
        0xc1, 0xba,
    ];
    let catalog = Catalog::from_handoff(vec![Resource {
        name: "rtl_nic/rtl8168e-3.fw",
        target: Target {
            driver: "realtek",
            hardware: "10ec:8168",
            revision: "xid-2c8",
            board: None,
        },
        provenance: Provenance {
            source: "synthetic-test",
            revision: "1",
            member: "rtl_nic/rtl8168e-3.fw",
            license: "MIT",
        },
        expected_length: 4,
        expected_sha256: catalog_digest,
        bytes,
    }])
    .unwrap();
    let image = load_firmware(&catalog, "10ec:8168", revision(34))
        .unwrap()
        .unwrap();
    assert_eq!(image.bytes(), &0x801f0000u32.to_le_bytes());
    assert!(matches!(
        load_firmware(&catalog, "10ec:8161", revision(34)),
        Err(Error::Firmware(resources::Error {
            kind: ErrorKind::WrongTarget,
            ..
        }))
    ));
    let mut other = revision(34);
    other.xid = 0x2c9;
    assert!(load_firmware(&catalog, "10ec:8168", other).is_err());
    assert!(
        load_firmware(&catalog, "10ec:8168", revision(18))
            .unwrap()
            .is_none()
    );
    other.firmware = Some("rtl_nic/rtl8168e-2.fw");
    assert!(matches!(
        load_firmware(&catalog, "10ec:8168", other),
        Err(Error::Firmware(resources::Error {
            kind: ErrorKind::Missing,
            ..
        }))
    ));
}
