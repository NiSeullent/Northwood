use super::*;
use crate::Clock;
use crate::bus::pci::regs;
use alloc::{rc::Rc, vec, vec::Vec};
use core::cell::RefCell;
mod timer;
#[test]
fn owned_pio_activation_never_grants_dma_or_writes_adjacent_pci_status() {
    #[derive(Clone)]
    struct Config {
        bytes: Rc<RefCell<Vec<u8>>>,
        writes: Rc<RefCell<Vec<(u16, usize, u32)>>>,
    }
    impl ConfigSpace for Config {
        fn read8(&self, at: u16) -> Result<u8, crate::bus::pci::ConfigError> {
            Ok(self.bytes.borrow()[usize::from(at)])
        }
        fn read16(&self, at: u16) -> Result<u16, crate::bus::pci::ConfigError> {
            let value = {
                let at = usize::from(at);
                u16::from_le_bytes(self.bytes.borrow()[at..at + 2].try_into().unwrap())
            };
            Ok(value)
        }
        fn read32(&self, at: u16) -> Result<u32, crate::bus::pci::ConfigError> {
            let value = {
                let at = usize::from(at);
                u32::from_le_bytes(self.bytes.borrow()[at..at + 4].try_into().unwrap())
            };
            Ok(value)
        }
        fn write8(&mut self, _: u16, _: u8) -> Result<(), crate::bus::pci::ConfigError> {
            panic!("PIO activation must use actual word stores");
        }
        fn write32(&mut self, _: u16, _: u32) -> Result<(), crate::bus::pci::ConfigError> {
            panic!("PCI status/control RMW would clear adjacent W1C status");
        }
        fn write16(&mut self, at: u16, value: u16) -> Result<(), crate::bus::pci::ConfigError> {
            assert!(matches!(at, 0x04 | 0x52 | 0x62 | 0x92 | 0xa2));
            if at == 0x04 {
                assert_eq!(value & regs::COMMAND_MASTER, 0, "transient DMA grant");
            }
            self.writes.borrow_mut().push((at, 2, u32::from(value)));
            let at = usize::from(at);
            self.bytes.borrow_mut()[at..at + 2].copy_from_slice(&value.to_le_bytes());

            Ok(())
        }
    }
    let bytes = Rc::new(RefCell::new(vec![0; 256]));
    let writes = Rc::new(RefCell::new(Vec::new()));
    {
        let mut b = bytes.borrow_mut();
        b[..4].copy_from_slice(&0x8220_1217u32.to_le_bytes());
        b[4..6].copy_from_slice(&7u16.to_le_bytes());
        b[6..8].copy_from_slice(&0xa010u16.to_le_bytes());
        b[8..12].copy_from_slice(&0x0805_0100u32.to_le_bytes());
        b[0x34] = 0x50;
        b[0x50] = 5;
        b[0x51] = 0x60;
        b[0x52..0x54].copy_from_slice(&0x81u16.to_le_bytes());
        b[0x60] = 0x11;
        b[0x62..0x64].copy_from_slice(&0x8001u16.to_le_bytes());
    }
    let valid = bytes.borrow().clone();
    let mut function = Function::new(
        crate::bus::pci::Address {
            segment: 0,
            bus: 0,
            device: 1,
            function: 0,
        },
        Config {
            bytes: bytes.clone(),
            writes: writes.clone(),
        },
    )
    .unwrap();
    quiesce_pio(&mut function).unwrap();
    assert_eq!(function.config().read16(4).unwrap(), 0x401);
    set_pio_command(&mut function, true).unwrap();
    let c = function.config();
    assert_eq!(c.read16(4).unwrap(), 0x403);
    assert_eq!(c.read16(6).unwrap(), 0xa010);
    assert_eq!(c.read16(0x52).unwrap(), 0x80);
    assert_eq!(c.read16(0x62).unwrap(), 1);
    assert_eq!(
        &*writes.borrow(),
        &[(4, 2, 0x401), (0x52, 2, 0x80), (0x62, 2, 1), (4, 2, 0x403)]
    );
    // Refuse malformed inventories without even a PCI command write: otherwise
    // the tolerant shared inventory could hide a still-enabled interrupt route.
    for (name, changes) in [
        ("below capability range", vec![(0x34, 0x3c)]),
        ("unaligned head", vec![(0x34, 0x51)]),
        ("unaligned next", vec![(0x51, 0x63)]),
        ("self loop", vec![(0x51, 0x50)]),
        ("multi-node loop", vec![(0x61, 0x50)]),
        ("duplicate MSI", vec![(0x60, 5)]),
        ("duplicate MSI-X", vec![(0x50, 0x11)]),
        ("all-ones ID", vec![(0x60, 0xff)]),
        ("reserved ID", vec![(0x60, 0)]),
        (
            "PM body beyond standard config",
            vec![(0x34, 0xfc), (0xfc, 1)],
        ),
        (
            "PMCSR overlaps MSI header",
            vec![(0x34, 0x40), (0x40, 1), (0x41, 0x44), (0x44, 5)],
        ),
        (
            "duplicate PM",
            vec![(0x34, 0x40), (0x40, 1), (0x41, 0x48), (0x48, 1)],
        ),
        (
            "MSI-X body beyond standard config",
            vec![(0x34, 0xfc), (0xfc, 0x11)],
        ),
        (
            "MSI body beyond standard config",
            vec![(0x34, 0xf8), (0xf8, 5)],
        ),
        (
            "capability overlaps MSI control",
            vec![(0x51, 0x54), (0x54, 1)],
        ),
        (
            "PCIe v1 body hides PM",
            vec![
                (0x34, 0x40),
                (0x40, 0x10),
                (0x41, 0x44),
                (0x42, 1),
                (0x44, 1),
                (0x48, 1), // real PCIe Device Control, not a PMCSR
            ],
        ),
        (
            "PCIe v1 body hides MSI",
            vec![(0x34, 0x40), (0x40, 0x10), (0x41, 0x50), (0x42, 1)],
        ),
        (
            "PCIe v2 tail hides PM",
            vec![
                (0x34, 0x40),
                (0x40, 0x10),
                (0x41, 0x70),
                (0x42, 2),
                (0x70, 1),
            ],
        ),
        (
            "PCIe v2 tail hides MSI",
            vec![
                (0x34, 0x40),
                (0x40, 0x10),
                (0x41, 0x70),
                (0x42, 2),
                (0x70, 5),
            ],
        ),
        (
            "VPD body hides PM",
            vec![(0x34, 0x40), (0x40, 3), (0x41, 0x44), (0x44, 1)],
        ),
        (
            "VPD body hides MSI",
            vec![(0x34, 0x40), (0x40, 3), (0x41, 0x44), (0x44, 5)],
        ),
        (
            "vendor body hides PM",
            vec![(0x34, 0x40), (0x40, 9), (0x41, 0x44), (0x42, 16), (0x44, 1)],
        ),
        (
            "vendor body hides MSI",
            vec![(0x34, 0x40), (0x40, 9), (0x41, 0x44), (0x42, 16), (0x44, 5)],
        ),
        (
            "subsystem body hides MSI",
            vec![(0x34, 0x40), (0x40, 0x0d), (0x41, 0x44), (0x44, 5)],
        ),
        (
            "PCIe unknown version",
            vec![(0x34, 0x40), (0x40, 0x10), (0x41, 0), (0x42, 3)],
        ),
        (
            "PCIe unqualified port type",
            vec![(0x34, 0x40), (0x40, 0x10), (0x41, 0), (0x42, 0x42)],
        ),
        (
            "PCIe endpoint with slot",
            vec![(0x34, 0x40), (0x40, 0x10), (0x41, 0), (0x42, 2), (0x43, 1)],
        ),
        (
            "PCIe truncated v2 body",
            vec![(0x34, 0xe0), (0xe0, 0x10), (0xe2, 2)],
        ),
        ("vendor overflow", vec![(0x34, 0xfc), (0xfc, 9), (0xfe, 5)]),
        ("unknown extent", vec![(0x34, 0x40), (0x40, 0x7f)]),
        ("unsupported known extent", vec![(0x34, 0x40), (0x40, 7)]),
        (
            "duplicate PCIe",
            vec![
                (0x34, 0x40),
                (0x40, 0x10),
                (0x41, 0x80),
                (0x42, 1),
                (0x80, 0x10),
                (0x82, 1),
            ],
        ),
        (
            "duplicate VPD",
            vec![(0x34, 0x40), (0x40, 3), (0x41, 0x48), (0x48, 3)],
        ),
        (
            "duplicate subsystem capability",
            vec![(0x34, 0x40), (0x40, 0x0d), (0x41, 0x48), (0x48, 0x0d)],
        ),
        ("non-endpoint header", vec![(0x0e, 1)]),
    ] {
        let mut corrupt = valid.clone();
        for (at, value) in changes {
            corrupt[at] = value;
        }
        let writes = Rc::new(RefCell::new(Vec::new()));
        let mut function = Function::new(
            function.address,
            Config {
                bytes: Rc::new(RefCell::new(corrupt)),
                writes: writes.clone(),
            },
        )
        .unwrap();
        assert!(quiesce_pio(&mut function).is_err(), "{name}");
        assert!(
            writes.borrow().is_empty(),
            "{name}: refused before hardware attachment"
        );
    }
    // A valid PM capability before MSI is admitted without treating PMCSR as
    // another header; the later power-up sees a real, disjoint PMCSR in D0.
    let mut with_pm = valid.clone();
    with_pm[0x34] = 0x40;
    with_pm[0x40] = 1;
    with_pm[0x41] = 0x50;
    let writes = Rc::new(RefCell::new(Vec::new()));
    let mut function = Function::new(
        function.address,
        Config {
            bytes: Rc::new(RefCell::new(with_pm)),
            writes: writes.clone(),
        },
    )
    .unwrap();
    quiesce_pio(&mut function).unwrap();
    function
        .power_up(&timer::Minimum(Clock {
            ticks_per_microsecond: 1000,
        }))
        .unwrap();
    assert_eq!(
        &*writes.borrow(),
        &[(4, 2, 0x401), (0x52, 2, 0x80), (0x62, 2, 1)]
    );
    // Source endpoint lengths differ by version and link presence.
    // All headers following these real bodies remain separate from PMCSR
    // and interrupt controls, even when body bytes resemble header IDs.
    for flags in [0x01u16, 0x02, 0x11, 0x12, 0x91, 0x92] {
        let mut mixed = valid.clone();
        mixed[0x34] = 0x40;
        mixed[0x40] = 0x10;
        mixed[0x41] = 0x80;
        mixed[0x42..0x44].copy_from_slice(&flags.to_le_bytes());
        mixed[0x80] = 1;
        mixed[0x81] = 0x90;
        mixed[0x82] = 3;
        mixed[0x90] = 5;
        mixed[0x91] = 0xa0;
        mixed[0x92..0x94].copy_from_slice(&1u16.to_le_bytes());
        mixed[0xa0] = 0x11;
        mixed[0xa1] = 0xb0;
        mixed[0xa2..0xa4].copy_from_slice(&0x8001u16.to_le_bytes());
        mixed[0xb0] = 3;
        mixed[0xb1] = 0xb8;
        mixed[0xb8] = 9;
        mixed[0xb9] = 0xc0;
        mixed[0xba] = 8;
        mixed[0xc0] = 0x0d;
        let writes = Rc::new(RefCell::new(Vec::new()));
        let mut function = Function::new(
            function.address,
            Config {
                bytes: Rc::new(RefCell::new(mixed)),
                writes: writes.clone(),
            },
        )
        .unwrap();
        quiesce_pio(&mut function).unwrap();
        function
            .power_up(&timer::Minimum(Clock {
                ticks_per_microsecond: 1000,
            }))
            .unwrap();
        assert_eq!(
            &*writes.borrow(),
            &[(4, 2, 0x401), (0x92, 2, 0), (0xa2, 2, 1)],
            "flags={flags:x}"
        );
    }
    // Exact endpoint bodies and vendor lengths may end at byte255. These
    // are boundary profiles, not headers admitted by truncation. A vendor
    // length below the entry's own header reserves the header alone.
    for (start, id, extra) in [
        (0xec, 0x10, 0x01u8),
        (0xcc, 0x10, 0x02),
        (0xf4, 0x10, 0x91),
        (0xd4, 0x10, 0x92),
        (0xfc, 9, 3),
        (0x40, 9, 0xc0),
        (0x40, 9, 0),
        (0x40, 9, 2),
    ] {
        let mut boundary = valid.clone();
        boundary[0x34] = start as u8;
        boundary[start] = id;
        boundary[start + 1] = 0;
        boundary[start + 2] = extra;
        if id == 0x10 {
            boundary[start + 3] = 0;
        }
        let writes = Rc::new(RefCell::new(Vec::new()));
        let mut function = Function::new(
            function.address,
            Config {
                bytes: Rc::new(RefCell::new(boundary)),
                writes: writes.clone(),
            },
        )
        .unwrap();
        quiesce_pio(&mut function).unwrap();
        assert_eq!(
            &*writes.borrow(),
            &[(4, 2, 0x401)],
            "start={start:x}, extra={extra:x}"
        );
    }
    // A complete maximum-length list of four-byte vendor capabilities
    // terminates normally without a forgiving iteration cutoff.
    let mut complete = valid;
    complete[0x34] = 0x40;
    for at in (0x40..=0xfc).step_by(4) {
        complete[at] = 9;
        complete[at + 2] = 4; // native vendor-capability byte length
        complete[at + 1] = if at == 0xfc { 0 } else { (at + 4) as u8 };
    }
    let writes = Rc::new(RefCell::new(Vec::new()));
    let mut function = Function::new(
        function.address,
        Config {
            bytes: Rc::new(RefCell::new(complete)),
            writes: writes.clone(),
        },
    )
    .unwrap();
    quiesce_pio(&mut function).unwrap();
    assert_eq!(&*writes.borrow(), &[(4, 2, 0x401)]);
}

mod access_failures;
