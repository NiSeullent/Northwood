// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026 Stela contributors
//! Aliases cases.
use super::*;

fn alias_function(device: u16, express: bool) -> Function<AliasConfig> {
    let mut config = AliasConfig {
        words: [0; 64],
        writes: Vec::new(),
    };
    config.words[0] = (u32::from(device) << 16) | 0x10ec;
    config.words[2] = 0x0200_0000;
    if express {
        config.words[1] = 1 << 20; // Status: conventional capability list.
        config.words[0x34 / 4] = 0x40;
        config.words[0x40 / 4] = 0x0001_0010;
        config.words[0x50 / 4] = 0xa5fc;
    }
    Function::new(
        pci::Address {
            segment: 0,
            bus: 0,
            device: 1,
            function: 2,
        },
        config,
    )
    .unwrap()
}

struct AliasConfig {
    words: [u32; 64],
    writes: Vec<(u16, u16)>,
}

impl ConfigSpace for AliasConfig {
    fn read8(&self, offset: u16) -> core::result::Result<u8, crate::bus::pci::ConfigError> {
        Ok((self.words[usize::from(offset / 4)] >> ((offset & 3) * 8)) as u8)
    }
    fn read16(&self, offset: u16) -> core::result::Result<u16, crate::bus::pci::ConfigError> {
        Ok((self.words[usize::from(offset / 4)] >> ((offset & 3) * 8)) as u16)
    }
    fn read32(&self, offset: u16) -> core::result::Result<u32, crate::bus::pci::ConfigError> {
        Ok(self.words[usize::from(offset / 4)] >> ((offset & 3) * 8))
    }
    fn write8(&mut self, _: u16, _: u8) -> core::result::Result<(), crate::bus::pci::ConfigError> {
        Err(crate::bus::pci::ConfigError::before_effect(
            crate::bus::pci::ConfigFailure::UnqualifiedControl,
        ))
    }
    fn write32(
        &mut self,
        _: u16,
        _: u32,
    ) -> core::result::Result<(), crate::bus::pci::ConfigError> {
        Err(crate::bus::pci::ConfigError::before_effect(
            crate::bus::pci::ConfigFailure::UnqualifiedControl,
        ))
    }
    fn write16(
        &mut self,
        offset: u16,
        value: u16,
    ) -> core::result::Result<(), crate::bus::pci::ConfigError> {
        self.writes.push((offset, value));
        let word = &mut self.words[usize::from(offset / 4)];
        let shift = (offset & 2) * 8;
        *word = (*word & !(0xffff << shift)) | (u32::from(value) << shift);

        Ok(())
    }
}

mod firmware;

#[test]
fn source_same_vendor_pci_aliases_still_require_exact_implemented_revision_and_link() {
    let aliases = [
        0x2502, 0x2600, 0x8129, 0x8162, 0x8167, 0x3000, 0x5000, 0x0e10,
    ];
    // Independent representative inputs exercise all41 ordered source entries,
    // including generic-mask fallbacks3ca/3c4 that overlap specific entries.
    let xids = [
        0x64a, 0x649, 0x681, 0x708, 0x68b, 0x68a, 0x689, 0x688, 0x641, 0x609, 0x54b, 0x54a, 0x502,
        0x541, 0x6c0, 0x5c8, 0x509, 0x4c0, 0x488, 0x481, 0x480, 0x2c8, 0x2c1, 0x2c0, 0x281, 0x280,
        0x28a, 0x28b, 0x3c9, 0x3c8, 0x3ca, 0x3c0, 0x3c2, 0x3c3, 0x3c4, 0x380, 0x980, 0x180, 0x100,
        0x040, 0x008,
    ];
    for alias in aliases {
        let mut function = alias_function(alias, true);
        assert!(matches(&function));
        function.vendor = 0x1186;
        assert!(!matches(&function));
        function.vendor = 0x10ec;
        function.class = [2, 0, 1];
        assert!(!matches(&function));
        function.class = [1, 6, 1];
        assert!(!matches(&function));
        for xid in xids {
            let revision = revision::detect(xid << 20).unwrap();
            assert!(admit_revision(alias, Some(0x50), revision).is_ok());
            assert_eq!(
                admit_revision(alias, None, revision).is_ok(),
                revision.version <= 6
            );
        }
        for raw in [0, 0x8000_0000] {
            let extended = revision::detect_extended(0x7c8 << 20, raw).unwrap();
            assert!(admit_revision(alias, Some(0x50), extended).is_ok());
            assert!(admit_revision(alias, None, extended).is_err());
        }
        let mut unsupported = revision(64);
        unsupported.version = 80;
        assert_eq!(
            admit_revision(alias, Some(0x50), unsupported),
            Err(Error::UnsupportedRevision(unsupported))
        );
    }
    for device in [0x8136, 0x8127, 0x8139, 0x8124, 0x2501, 0x2601, 0xffff] {
        assert!(!matches(&alias_function(device, true)));
        assert!(admit_revision(device, Some(0x50), revision(64)).is_err());
    }
    // Historical primary PCI families keep their narrower source boundaries.
    assert!(admit_revision(0x8125, Some(0x50), revision(70)).is_err());
    assert!(admit_revision(0x8169, Some(0x50), revision(64)).is_err());
    assert!(admit_revision(0x8168, Some(0x50), revision(6)).is_err());
}

#[test]
fn pci_alias_aspm_selection_uses_capability_before_clock_or_enable() {
    for device in [
        0x8169, 0x8129, 0x2502, 0x8162, 0x8167, 0x3000, 0x5000, 0x0e10,
    ] {
        for express in [false, true] {
            let mut function = alias_function(device, express);
            let called = Cell::new(false);
            let control = prepare_pcie_link(&mut function, |function| {
                called.set(true);
                assert_eq!(function.config().read16(4).unwrap() & 6, 0); // No MMIO/bus-master enable yet.
                assert_eq!(function.config().read16(0x50).unwrap(), 0xa5fc);
                assert!(function.config().writes.is_empty());
                Ok(())
            })
            .unwrap();
            assert_eq!(called.get(), express);
            assert_eq!(control, express.then_some(0x50));
            assert_eq!(function.config().read16(4).unwrap() & 6, 0);
            assert_eq!(
                function.config().writes,
                if express {
                    vec![(0x50, 0xa4fc)]
                } else {
                    vec![]
                }
            );
        }
    }
    let mut function = alias_function(0x2502, true);
    assert_eq!(
        prepare_pcie_link(&mut function, |_| Err(BlockError::Timeout)),
        Err(BlockError::Timeout)
    );
    assert!(function.config().writes.is_empty());
    function.config().words[0x50 / 4] = 0xffff;
    assert_eq!(
        prepare_pcie_link(&mut function, |_| Ok(())),
        Err(BlockError::Device { status: 65535 })
    );
    assert!(function.config().writes.is_empty());
    for invalid in [0, 1] {
        let mut function = alias_function(0x2502, true);
        if invalid == 0 {
            function.config().words[0x40 / 4] |= 0x4000;
        }
        // Looping PCIe capability.
        else {
            function.config().words[0x34 / 4] = 0xf0;
            function.config().words[0xf0 / 4] = 0x10;
        }
        assert!(
            prepare_pcie_link(&mut function, |_| panic!(
                "invalid list must fail before ASPM"
            ))
            .is_err()
        );
        assert!(function.config().writes.is_empty());
    }
}

#[test]
fn alias_detected_layout_requires_full_bar_before_hardware_or_dma() {
    for alias in [
        0x2502, 0x2600, 0x8129, 0x8162, 0x8167, 0x3000, 0x5000, 0x0e10,
    ] {
        for xid in [0x609, 0x641, 0x688, 0x708, 0x681, 0x649] {
            let time = Time::default();
            let model = Model::nbase(&time);
            let ranges = RefCell::new(Vec::new());
            assert_eq!(
                detect_revision(&model, xid << 20, |bytes| {
                    ranges.borrow_mut().push(bytes);
                    Err(BlockError::Unsupported("unmapped detected BAR layout"))
                }),
                Err(Error::Device(BlockError::Unsupported(
                    "unmapped detected BAR layout"
                )))
            );
            assert_eq!(*ranges.borrow(), vec![0x604a]);
            assert!(model.state.borrow().writes.is_empty());
            let revision = detect_revision(&model, xid << 20, |bytes| {
                assert_eq!(bytes, 0x604a);
                Ok(())
            })
            .unwrap();
            admit_revision(alias, Some(0x50), revision).unwrap();
            assert_eq!(register_aperture(revision), 0x604a);
        }
        let time = Time::default();
        let model = Model::nbase(&time);
        assert!(
            detect_revision(&model, 0x7c8 << 20, |bytes| {
                assert_eq!(bytes, 0x60b4);
                Err(BlockError::Unsupported("short extended BAR"))
            })
            .is_err()
        );
        assert!(model.state.borrow().writes.is_empty());
        let old = detect_revision(&model, 0x380 << 20, |_| {
            panic!("legacy layout already checked at initial256-byte map")
        })
        .unwrap();
        admit_revision(alias, Some(0x50), old).unwrap();
        assert_eq!(register_aperture(old), 256);
        assert!(
            detect_revision(&model, 0x300 << 20, |_| panic!(
                "unknown XID must not expand aperture"
            ))
            .is_err()
        );
    }
}

#[test]
fn other_vendors_realtek_boards_take_the_family_of_their_realtek_id() {
    for (vendor, device, realtek) in [
        (0x1186, 0x4300, 0x8167),
        (0x1186, 0x4302, 0x8167),
        (0x1259, 0xc107, 0x8167),
        (0x16ec, 0x0116, 0x8167),
        (0x10ff, 0x8168, 0x8168),
    ] {
        let mut function = alias_function(device, true);
        function.vendor = vendor;
        assert!(matches(&function));
        assert_eq!(revision::pci_device(vendor, device), Some(realtek));
        function.class = [2, 0x80, 0];
        assert!(!matches(&function));
    }
    // A Realtek ID under another vendor, or another vendor's ID under
    // Realtek's, is no candidate.
    assert_eq!(revision::pci_device(0x1186, 0x8168), None);
    assert_eq!(revision::pci_device(0x10ec, 0x4300), None);
    assert_eq!(revision::pci_device(0x1737, 0x1032), None);
    // The RTL8169SC boards admit what Realtek's 8167 does; NCube's what 8168 does.
    assert!(admit_revision(0x8167, None, revision(4)).is_ok());
    assert!(admit_revision(0x8168, Some(0x50), revision(4)).is_err());
}
