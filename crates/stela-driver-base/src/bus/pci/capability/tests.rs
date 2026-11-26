use super::*;
use crate::bus::pci::regs;
use alloc::vec::Vec;

struct Space([u8; 4096]);

impl Space {
    fn new() -> Self {
        let mut space = Self([0; 4096]);
        space.put16(regs::STATUS, regs::STATUS_CAP_LIST);
        space
    }
    fn put16(&mut self, offset: u16, value: u16) {
        let at = usize::from(offset);
        self.0[at..at + 2].copy_from_slice(&value.to_le_bytes());
    }
    fn put32(&mut self, offset: u16, value: u32) {
        let at = usize::from(offset);
        self.0[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    fn standard(&mut self, offset: u16, id: u8, next: u8) {
        self.put16(offset, u16::from(id) | u16::from(next) << 8);
    }
    fn extended(&mut self, offset: u16, id: u16, next: u16) {
        self.put32(offset, u32::from(id) | 1 << 16 | u32::from(next) << 20);
    }
    fn walk(&self) -> Vec<Result<(u16, u8, u16), CapabilityError>> {
        capabilities(self)
            .map(|entry| entry.map(|c| (c.offset, c.id, c.bytes)))
            .collect()
    }
    fn walk_extended(&self) -> Vec<Result<(u16, u16, u16), CapabilityError>> {
        extended_capabilities(self)
            .map(|entry| entry.map(|c| (c.offset, c.id, c.bytes)))
            .collect()
    }
}

impl ConfigSpace for Space {
    fn read32(&self, offset: u16) -> Result<u32, crate::bus::pci::ConfigError> {
        Ok({
            let at = usize::from(offset);
            u32::from_le_bytes(self.0[at..at + 4].try_into().unwrap())
        })
    }
    fn write32(&mut self, offset: u16, value: u32) -> Result<(), crate::bus::pci::ConfigError> {
        self.put32(offset, value);
        Ok(())
    }

    fn read8(&self, offset: u16) -> Result<u8, crate::bus::pci::ConfigError> {
        Ok((self.read32(offset & !3)? >> ((offset & 3) * 8)) as u8)
    }

    fn write8(&mut self, offset: u16, value: u8) -> Result<(), crate::bus::pci::ConfigError> {
        let shift = (offset & 3) * 8;
        let word = self.read32(offset & !3)? & !(0xff << shift) | u32::from(value) << shift;
        self.write32(offset & !3, word)
    }

    fn read16(&self, offset: u16) -> Result<u16, crate::bus::pci::ConfigError> {
        Ok((self.read32(offset & !3)? >> ((offset & 2) * 8)) as u16)
    }

    fn write16(&mut self, offset: u16, value: u16) -> Result<(), crate::bus::pci::ConfigError> {
        let shift = (offset & 2) * 8;
        let word = self.read32(offset & !3)? & !(0xffff << shift) | u32::from(value) << shift;
        self.write32(offset & !3, word)
    }
}

#[test]
fn walks_sized_bodies_in_list_order() {
    let mut space = Space::new();
    space.0[0x34] = 0x50;
    space.standard(0x50, 0x05, 0x40);
    space.put16(0x52, 1 << 7 | 1 << 8); // 64-bit, maskable MSI
    space.standard(0x40, 0x01, 0x70);
    space.standard(0x70, 0x09, 0);
    space.0[0x72] = 0x10;
    assert_eq!(
        space.walk(),
        [Ok((0x50, 5, 24)), Ok((0x40, 1, 8)), Ok((0x70, 9, 0x10))]
    );
}

#[test]
fn status_without_capability_list_is_empty() {
    let mut space = Space::new();
    space.put16(regs::STATUS, 0);
    space.0[0x34] = 0x40;
    space.standard(0x40, 0x01, 0);
    assert!(space.walk().is_empty());
}

#[test]
fn a_pointer_with_reserved_bits_is_misaligned() {
    let mut space = Space::new();
    space.0[0x34] = 0x40;
    space.standard(0x40, 0x01, 0x52);
    assert_eq!(
        space.walk(),
        [
            Ok((0x40, 1, 8)),
            Err(CapabilityError::Misaligned { offset: 0x52 })
        ]
    );
}

#[test]
fn a_pointer_below_the_header_is_out_of_range() {
    let mut space = Space::new();
    space.0[0x34] = 0x3c;
    assert_eq!(
        space.walk(),
        [Err(CapabilityError::OutOfRange { offset: 0x3c })]
    );
}

#[test]
fn a_loop_is_a_cycle() {
    let mut space = Space::new();
    space.0[0x34] = 0x40;
    space.standard(0x40, 0x01, 0x48);
    space.standard(0x48, 0x11, 0x40);
    assert_eq!(
        space.walk(),
        [
            Ok((0x40, 1, 8)),
            Ok((0x48, 0x11, 12)),
            Err(CapabilityError::Cycle { offset: 0x40 })
        ]
    );
}

#[test]
fn a_header_and_a_body_overlap_in_either_order() {
    // A header inside an earlier body, and a body over an earlier header.
    for (first, second, expected) in [
        (
            (0x40, 0x11, 0x48),
            (0x48, 0x01),
            [
                Ok((0x40, 0x11, 12)),
                Err(CapabilityError::Overlap { offset: 0x48 }),
            ],
        ),
        (
            (0x48, 0x01, 0x40),
            (0x40, 0x11),
            [
                Ok((0x48, 0x01, 8)),
                Err(CapabilityError::Overlap { offset: 0x40 }),
            ],
        ),
    ] {
        let mut space = Space::new();
        space.0[0x34] = first.0 as u8;
        space.standard(first.0, first.1, first.2);
        space.standard(second.0, second.1, 0);
        assert_eq!(space.walk(), expected);
    }
}

#[test]
fn a_body_past_the_end_is_out_of_range() {
    let mut space = Space::new();
    space.0[0x34] = 0xf8;
    space.standard(0xf8, 0x11, 0);
    assert_eq!(
        space.walk(),
        [Err(CapabilityError::OutOfRange { offset: 0xf8 })]
    );
}

#[test]
fn an_all_ones_header_is_a_failed_read() {
    let mut space = Space::new();
    space.0[0x34] = 0x40;
    space.put16(0x40, u16::MAX);
    assert_eq!(
        space.walk(),
        [Err(CapabilityError::FailedRead { offset: 0x40 })]
    );
}

/// Intel's PCH xHCI lists a vendor-specific entry whose length byte is 0
/// after a sized one. Only its header is reserved and the walk goes on to
/// the entries after it, as Linux's does.
#[test]
fn a_vendor_entry_shorter_than_its_header_reserves_only_the_header() {
    for length in [0, 1, 2] {
        let mut space = Space::new();
        space.0[0x34] = 0x70;
        space.standard(0x70, 0x01, 0x80);
        space.standard(0x80, 0x09, 0xb0);
        space.0[0x82] = 0x14;
        space.standard(0xb0, 0x09, 0xb4);
        space.0[0xb2] = length;
        space.standard(0xb4, 0x00, 0);
        assert_eq!(
            space.walk(),
            [
                Ok((0x70, 1, 8)),
                Ok((0x80, 9, 0x14)),
                Ok((0xb0, 9, 2)),
                Ok((0xb4, 0, 2))
            ]
        );
        let layouts: Vec<_> = capabilities(&space)
            .map(|entry| entry.unwrap().layout)
            .collect();
        assert_eq!(
            layouts,
            [Layout::Sized, Layout::Sized, Layout::Header, Layout::Sized]
        );
    }
}

#[test]
fn express_size_follows_version_type_and_slot() {
    for (flags, bytes) in [
        (0x0001, 0x14),
        (0x0091, 12),
        (0x0041, 0x24),
        (0x0161, 0x1c),
        (0x0002, 0x34),
        (0x0092, 0x2c),
        (0x0042, 0x3c),
        (0x0162, 0x3c),
    ] {
        let mut space = Space::new();
        space.0[0x34] = 0x40;
        space.standard(0x40, 0x10, 0);
        space.put16(0x42, flags);
        assert_eq!(space.walk(), [Ok((0x40, 0x10, bytes))], "{flags:#x}");
    }
    let mut space = Space::new();
    space.0[0x34] = 0x40;
    space.standard(0x40, 0x10, 0);
    space.put16(0x42, 3);
    let entry = capabilities(&space).next().unwrap().unwrap();
    assert_eq!((entry.bytes, entry.layout), (2, Layout::Unsupported));
}

#[test]
fn a_headers_walk_accepts_a_body_shorter_than_its_layout() {
    let mut space = Space::new();
    space.0[0x34] = 0x58;
    space.standard(0x58, 0x05, 0x5c); // 32-bit MSI: ten bytes from 0x58
    space.standard(0x5c, 0x08, 0);
    assert_eq!(
        space.walk(),
        [
            Ok((0x58, 5, 10)),
            Err(CapabilityError::Overlap { offset: 0x5c })
        ]
    );
    let headers: Vec<_> = capability_headers(&space)
        .map(|entry| entry.map(|c| (c.offset, c.id, c.bytes, c.layout)))
        .collect();
    assert_eq!(
        headers,
        [
            Ok((0x58, 5, 2, Layout::Header)),
            Ok((0x5c, 8, 2, Layout::Header))
        ]
    );
}

#[test]
fn unknown_ids_reserve_their_header() {
    let mut space = Space::new();
    space.0[0x34] = 0x40;
    space.standard(0x40, 0x0a, 0x44);
    space.standard(0x44, 0x01, 0);
    let entries: Vec<_> = capabilities(&space).map(Result::unwrap).collect();
    assert_eq!(entries[0].layout, Layout::Header);
    assert_eq!(entries[0].bytes, 2);
    assert_eq!(entries[1].offset, 0x44);
}

#[test]
fn transport_errors_end_the_walk() {
    #[derive(Debug, PartialEq)]
    enum Error {
        Link,
        Capability(CapabilityError),
    }
    impl From<CapabilityError> for Error {
        fn from(error: CapabilityError) -> Self {
            Self::Capability(error)
        }
    }
    let mut space = Space::new();
    space.0[0x34] = 0x40;
    space.standard(0x40, 0x01, 0x48);
    let walk: Vec<_> = Capabilities::new(Transport(|offset, bytes| {
        if offset == 0x40 {
            Err(Error::Link)
        } else {
            Config(&space).read(offset, bytes).map_err(Error::from)
        }
    }))
    .collect();
    assert_eq!(walk, [Err(Error::Link)]);
}

#[test]
fn extended_walks_sized_bodies() {
    let mut space = Space::new();
    space.extended(0x100, 0x1e, 0x140);
    space.extended(0x140, 0x0b, 0x180);
    space.put32(0x144, 0x018 << 20);
    space.extended(0x180, 0x01, 0);
    assert_eq!(
        space.walk_extended(),
        [
            Ok((0x100, 0x1e, 16)),
            Ok((0x140, 0x0b, 0x18)),
            Ok((0x180, 1, 4))
        ]
    );
}

#[test]
fn extended_without_configuration_space_is_empty() {
    let mut space = Space::new();
    assert!(space.walk_extended().is_empty());
    space.put32(0x100, u32::MAX);
    assert!(space.walk_extended().is_empty());
}

#[test]
fn extended_walks_through_a_null_entry() {
    let mut space = Space::new();
    space.extended(0x100, 0, 0x140);
    space.extended(0x140, 0x18, 0);
    assert_eq!(
        space.walk_extended(),
        [Ok((0x100, 0, 4)), Ok((0x140, 0x18, 8))]
    );
}

#[test]
fn extended_refuses_loops_overlaps_and_null_entries() {
    let mut space = Space::new();
    space.extended(0x100, 0x0e, 0x104);
    assert_eq!(
        space.walk_extended()[1],
        Err(CapabilityError::Overlap { offset: 0x104 })
    );
    space.extended(0x100, 0x01, 0x100);
    assert_eq!(
        space.walk_extended()[1],
        Err(CapabilityError::Cycle { offset: 0x100 })
    );
    space.extended(0x100, 0x01, 0x200);
    assert_eq!(
        space.walk_extended()[1],
        Err(CapabilityError::Malformed { offset: 0x200 })
    );
    space.extended(0x100, 0x01, 0xff8);
    space.extended(0xff8, 0x13, 0);
    assert_eq!(
        space.walk_extended()[1],
        Err(CapabilityError::OutOfRange { offset: 0xff8 })
    );
}
