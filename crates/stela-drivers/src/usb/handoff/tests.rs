use super::*;
use alloc::{collections::BTreeMap, vec::Vec};

fn clock() -> Clock {
    Clock {
        ticks_per_microsecond: 1,
    }
}

/// A model's reaction to one write: its words, the offset and the value.
type React = fn(&mut BTreeMap<usize, u32>, usize, u32);

/// Dword registers with an optional reaction to each write.
#[derive(Default)]
struct Model {
    words: BTreeMap<usize, u32>,
    writes: Vec<(usize, u32)>,
    react: Option<React>,
}

impl Model {
    fn with(words: &[(usize, u32)]) -> Self {
        Self {
            words: words.iter().copied().collect(),
            ..Self::default()
        }
    }
    fn word(&self, offset: usize) -> u32 {
        self.words.get(&offset).copied().unwrap_or(0)
    }
}

impl Dwords for Model {
    fn read32(&mut self, offset: usize) -> u32 {
        self.word(offset)
    }
    fn write32(&mut self, offset: usize, value: u32) {
        self.writes.push((offset, value));
        self.words.insert(offset, value);
        if let Some(react) = self.react {
            react(&mut self.words, offset, value);
        }
    }
}

impl Registers for Model {
    fn read8(&mut self, offset: usize) -> u8 {
        (self.word(offset & !3) >> ((offset & 3) * 8)) as u8
    }
    fn read16(&mut self, offset: usize) -> u16 {
        (self.word(offset & !3) >> ((offset & 2) * 8)) as u16
    }
    fn write16(&mut self, offset: usize, value: u16) {
        self.write32(offset, u32::from(value));
    }
}

/// Configuration space as dwords, with a reaction to each byte or dword write.
struct Config {
    words: [u32; 64],
    react: fn(&mut [u32; 64], u16),
}

impl Config {
    fn new(react: fn(&mut [u32; 64], u16)) -> Self {
        Self {
            words: [0; 64],
            react,
        }
    }
}

impl ConfigSpace for Config {
    fn read32(&self, offset: u16) -> core::result::Result<u32, crate::bus::pci::ConfigError> {
        Ok(self.words[usize::from(offset / 4)])
    }
    fn write32(
        &mut self,
        offset: u16,
        value: u32,
    ) -> core::result::Result<(), crate::bus::pci::ConfigError> {
        self.words[usize::from(offset / 4)] = value;
        (self.react)(&mut self.words, offset);

        Ok(())
    }
    fn write8(
        &mut self,
        offset: u16,
        value: u8,
    ) -> core::result::Result<(), crate::bus::pci::ConfigError> {
        let word = &mut self.words[usize::from(offset / 4)];
        let shift = (offset & 3) * 8;
        *word = *word & !(0xff << shift) | u32::from(value) << shift;
        (self.react)(&mut self.words, offset);

        Ok(())
    }

    fn read8(&self, offset: u16) -> core::result::Result<u8, crate::bus::pci::ConfigError> {
        Ok((self.read32(offset & !3)? >> ((offset & 3) * 8)) as u8)
    }
    fn read16(&self, offset: u16) -> core::result::Result<u16, crate::bus::pci::ConfigError> {
        Ok((self.read32(offset & !3)? >> ((offset & 2) * 8)) as u16)
    }
    fn write16(
        &mut self,
        offset: u16,
        value: u16,
    ) -> core::result::Result<(), crate::bus::pci::ConfigError> {
        let shift = (offset & 2) * 8;
        let word = self.read32(offset & !3)? & !(0xffff << shift) | u32::from(value) << shift;
        self.write32(offset & !3, word)
    }
}

const EECP: u16 = 0x68;

fn ehci_registers() -> Model {
    // CAPLENGTH 0x20, EECP 0x68; the controller halts once RUN clears.
    let mut model = Model::with(&[(0, 0x20), (EHCI_HCCPARAMS, u32::from(EECP) << 8)]);
    model.react = Some(|words, offset, _| {
        let stopped = words.get(&(0x20 + EHCI_USBCMD)).copied().unwrap_or(0) & EHCI_RUN == 0;
        if (offset == 0x20 + EHCI_USBCMD || offset == 0x20 + EHCI_USBSTS) && stopped {
            words.insert(0x20 + EHCI_USBSTS, EHCI_HALTED);
        }
    });
    model
}

#[test]
fn ehci_firmware_releases_its_semaphore_and_its_smis_go_off() {
    let _time = crate::test_clock::virtualize();
    let mut config = Config::new(|words, offset| {
        // Firmware lets go once the OS Owned byte is set.
        if offset == EECP + 3 {
            words[usize::from(EECP / 4)] &= !EHCI_BIOS_OWNED;
        }
    });
    config.words[usize::from(EECP / 4)] = EHCI_LEGSUP | EHCI_BIOS_OWNED;
    config.words[usize::from(EECP / 4) + 1] = 0xe03f_0001;
    let mut registers = ehci_registers();
    registers.words.insert(0x20 + EHCI_USBCMD, EHCI_RUN);
    let handed = ehci(
        &mut config,
        &mut registers,
        0x400,
        clock(),
        crate::minimum_model::owned(clock()).as_ref(),
    )
    .unwrap();
    assert_eq!(
        handed,
        Handed {
            from_firmware: true,
            forced: false,
            legacy_released: true,
        }
    );
    assert_eq!(config.words[usize::from(EECP / 4)], EHCI_LEGSUP | 1 << 24);
    assert_eq!(config.words[usize::from(EECP / 4) + 1], 0);
    assert!(registers.writes.contains(&(0x20 + EHCI_CONFIGFLAG, 0)));
    assert_eq!(registers.word(0x20 + EHCI_USBINTR), 0);
    assert_eq!(registers.word(0x20 + EHCI_USBCMD) & EHCI_RUN, 0);
}

#[test]
fn ehci_firmware_that_keeps_its_semaphore_is_forced_out() {
    let _time = crate::test_clock::virtualize();
    let mut config = Config::new(|_, _| {});
    config.words[usize::from(EECP / 4)] = EHCI_LEGSUP | EHCI_BIOS_OWNED;
    let handed = ehci(
        &mut config,
        &mut ehci_registers(),
        0x400,
        clock(),
        crate::minimum_model::owned(clock()).as_ref(),
    )
    .unwrap();
    assert!(handed.from_firmware && handed.forced);
    assert_eq!(config.words[usize::from(EECP / 4)] & EHCI_BIOS_OWNED, 0);
}

#[test]
fn ehci_capability_loop_without_legacy_support_is_malformed() {
    let _time = crate::test_clock::virtualize();
    let mut config = Config::new(|_, _| {});
    // A vendor capability pointing at itself.
    config.words[usize::from(EECP / 4)] = u32::from(EECP) << 8 | 0x0a;
    assert_eq!(
        ehci(
            &mut config,
            &mut ehci_registers(),
            0x400,
            clock(),
            crate::minimum_model::owned(clock()).as_ref()
        ),
        Err(Error::Malformed)
    );
}

#[test]
fn ohci_routing_returns_on_ownership_change_or_is_cleared() {
    let _time = crate::test_clock::virtualize();
    for releases in [true, false] {
        let mut registers = Model::with(&[
            (OHCI_CONTROL, OHCI_CTRL_IR | OHCI_CTRL_RWC | 0x80),
            (OHCI_FMINTERVAL, 0x2edf),
        ]);
        if releases {
            registers.react = Some(|words, offset, value| {
                if offset == OHCI_CMDSTATUS && value & OHCI_OCR != 0 {
                    *words.get_mut(&OHCI_CONTROL).unwrap() &= !OHCI_CTRL_IR;
                }
            });
        }
        let handed = ohci(
            &mut registers,
            0x1000,
            clock(),
            crate::minimum_model::owned(clock()).as_ref(),
        )
        .unwrap();
        assert_eq!(
            handed,
            Handed {
                from_firmware: true,
                forced: !releases,
                legacy_released: true,
            }
        );
        assert_eq!(registers.word(OHCI_CONTROL), OHCI_CTRL_RWC);
        assert_eq!(registers.word(OHCI_FMINTERVAL), 0x2edf);
        assert!(registers.writes.contains(&(OHCI_INTRDISABLE, u32::MAX)));
    }
}

#[test]
fn uhci_set_up_by_firmware_is_reset_with_legacy_support_cleared() {
    let _time = crate::test_clock::virtualize();
    let mut config = Config::new(|_, _| {});
    config.words[usize::from(UHCI_LEGSUP / 4)] = 0x2010;
    let mut ports = Model::with(&[(UHCI_USBCMD, u32::from(UHCI_RUN | UHCI_CONFIGURE))]);
    let handed = uhci(
        &mut config,
        &mut ports,
        crate::minimum_model::owned(clock()).as_ref(),
    )
    .unwrap();
    assert!(handed.from_firmware);
    assert!(handed.legacy_released);
    assert_eq!(config.read16(UHCI_LEGSUP).unwrap(), UHCI_LEGSUP_RWC);
    assert_eq!(
        ports.writes,
        [
            (UHCI_USBCMD, u32::from(UHCI_HCRESET)),
            (UHCI_USBINTR, 0),
            (UHCI_USBCMD, 0)
        ]
    );
}

#[test]
fn uhci_left_suspended_by_firmware_is_untouched() {
    let _time = crate::test_clock::virtualize();
    let mut config = Config::new(|_, _| {});
    let mut ports = Model::with(&[(UHCI_USBCMD, u32::from(UHCI_CONFIGURE | UHCI_EGSM))]);
    assert_eq!(
        uhci(
            &mut config,
            &mut ports,
            crate::minimum_model::owned(clock()).as_ref()
        ),
        Ok(Handed::default())
    );
    assert!(ports.writes.is_empty());
}

#[test]
fn xhci_legacy_support_is_taken_and_the_controller_halted() {
    let _time = crate::test_clock::virtualize();
    // CAPLENGTH 0x20, xECP at dword 0x100 / 4; a protocol capability first.
    let mut registers = Model::with(&[
        (0, 0x20),
        (XHCI_HCCPARAMS1, (0x100 / 4) << 16),
        (0x100, 2 | 4 << 8),
        (0x110, XHCI_LEGACY | XHCI_BIOS_OWNED),
        (0x114, 0xe001_e011),
        (0x20 + XHCI_USBCMD, XHCI_RUN | XHCI_IRQS),
    ]);
    registers.react = Some(|words, offset, value| {
        if offset == 0x110 && value & XHCI_OS_OWNED != 0 {
            *words.get_mut(&0x110).unwrap() &= !XHCI_BIOS_OWNED;
        }
        if offset == 0x20 + XHCI_USBCMD && value & XHCI_RUN == 0 {
            words.insert(0x20 + XHCI_USBSTS, XHCI_HALTED);
        }
    });
    let handed = xhci(
        &mut registers,
        0x1000,
        clock(),
        crate::minimum_model::owned(clock()).as_ref(),
    )
    .unwrap();
    assert_eq!(
        handed,
        Handed {
            from_firmware: true,
            forced: false,
            legacy_released: true,
        }
    );
    assert_eq!(registers.word(0x110), XHCI_LEGACY | XHCI_OS_OWNED);
    // The SMI enables are clear and the RW1C events written back.
    assert_eq!(registers.word(0x114), XHCI_LEGACY_SMI_EVENTS);
    assert_eq!(registers.word(0x20 + XHCI_USBCMD), 0);
}

#[test]
fn xhci_legacy_readback_can_withhold_i8042_without_refusing_usb() {
    let _time = crate::test_clock::virtualize();
    let minimum = crate::minimum_model::owned(clock());
    let mut registers = Model::with(&[(0x110, XHCI_LEGACY)]);
    registers.react = Some(|words, offset, _| {
        if offset == 0x114 {
            *words.get_mut(&offset).unwrap() |= 1;
        }
    });
    let handed = xhci_legacy(&mut registers, 0x110, clock(), minimum.as_ref()).unwrap();
    assert_eq!(registers.word(0x110), XHCI_LEGACY | XHCI_OS_OWNED);
    assert!(!handed.legacy_released);

    registers.react = Some(|words, offset, _| {
        if offset == 0x110 {
            *words.get_mut(&offset).unwrap() &= !XHCI_OS_OWNED;
        }
    });
    let handed = xhci_legacy(&mut registers, 0x110, clock(), minimum.as_ref()).unwrap();
    assert_eq!(registers.word(0x114) & XHCI_LEGACY_SMI_ENABLES, 0);
    assert!(!handed.legacy_released);
}

#[test]
fn registers_reading_all_ones_refuse_the_handoff() {
    let _time = crate::test_clock::virtualize();
    let mut gone = Model::with(&[(XHCI_HCCPARAMS1, u32::MAX), (OHCI_CONTROL, u32::MAX)]);
    assert_eq!(
        xhci(
            &mut gone,
            0x1000,
            clock(),
            crate::minimum_model::owned(clock()).as_ref()
        ),
        Err(Error::Removed)
    );
    assert_eq!(
        ohci(
            &mut gone,
            0x1000,
            clock(),
            crate::minimum_model::owned(clock()).as_ref()
        ),
        Err(Error::Removed)
    );
}

#[test]
fn only_usb_host_interfaces_have_a_kind() {
    assert_eq!(Kind::of([0x0c, 0x03, 0x20]), Some(Kind::Ehci));
    assert_eq!(Kind::of([0x0c, 0x03, 0xfe]), None);
    assert_eq!(Kind::of([0x0c, 0x05, 0x00]), None);
}

/// An Intel xHCI's configuration space with its port-routing masks.
fn intel_xhci(device: u16, subsystem: u32, superspeed: u32, usb2: u32) -> Function<Config> {
    let mut config = Config::new(|_, _| {});
    config.words[0] = u32::from(INTEL) | u32::from(device) << 16;
    config.words[2] = 0x0c03_3000;
    config.words[usize::from(regs::SUBSYSTEM_VENDOR_ID / 4)] = subsystem;
    config.words[usize::from(INTEL_USB3PRM / 4)] = superspeed;
    config.words[usize::from(INTEL_USB2PRM / 4)] = usb2;
    Function::new(
        crate::bus::pci::Address {
            segment: 0,
            bus: 0,
            device: 0x14,
            function: 0,
        },
        config,
    )
    .unwrap_or_else(|_| panic!("modelled function"))
}

#[test]
fn intel_xhci_takes_every_switchable_port_from_ehci() {
    for device in [0x1e31, 0x8c31, 0x9c31, 0x9cb1] {
        let mut function = intel_xhci(device, 0, 0xf, 0x3fff);
        assert_eq!(
            switch_intel_ports(&mut function, true),
            Ok(Some(Switched {
                superspeed: 0xf,
                usb2: 0x3fff
            }))
        );
        let words = &function.config().words;
        assert_eq!(words[usize::from(INTEL_USB3_PSSEN / 4)], 0xf);
        assert_eq!(words[usize::from(INTEL_XUSB2PR / 4)], 0x3fff);
    }
}

#[test]
fn ports_stay_put_without_an_intel_ehci_or_a_switchable_port() {
    let untouched = |function: &mut Function<Config>| {
        let words = &function.config().words;
        words[usize::from(INTEL_USB3_PSSEN / 4)] == 0 && words[usize::from(INTEL_XUSB2PR / 4)] == 0
    };
    let mut function = intel_xhci(0x8c31, 0, 0xf, 0x3fff);
    assert_eq!(switch_intel_ports(&mut function, false), Ok(None));
    assert!(untouched(&mut function));
    let mut function = intel_xhci(0x8c31, SONY_FIXED_PORTS, 0xf, 0x3fff);
    assert_eq!(switch_intel_ports(&mut function, true), Ok(None));
    assert!(untouched(&mut function));
    let mut function = intel_xhci(0xa12f, 0, 0, 0);
    assert_eq!(switch_intel_ports(&mut function, true), Ok(None));
    assert!(untouched(&mut function));
    let mut amd = intel_xhci(0x43ba, 0, 0xf, 0x3fff);
    amd.vendor = 0x1022;
    assert_eq!(switch_intel_ports(&mut amd, true), Ok(None));
    assert!(untouched(&mut amd));
}

/// One page of EHCI registers the early handoff reaches through BAR0.
#[repr(align(4096))]
struct Page([u32; 1024]);

#[test]
fn early_ehci_handoff_takes_the_semaphore_and_clears_the_smis() {
    let _time = crate::test_clock::virtualize();
    let mut page = alloc::boxed::Box::new(Page([0; 1024]));
    // CAPLENGTH 0x20 and EECP; the controller is halted.
    page.0[0] = 0x20;
    page.0[EHCI_HCCPARAMS / 4] = u32::from(EECP) << 8;
    page.0[(0x20 + EHCI_USBSTS) / 4] = EHCI_HALTED;
    let base = page.0.as_mut_ptr() as u64;
    let mut config = Config::new(|words, offset| {
        // BAR0 decodes 4 KiB, 64-bit; firmware lets go of the semaphore.
        if offset == 0x10 && words[4] == u32::MAX {
            words[4] = 0xffff_f004;
        }
        if offset == EECP + 3 {
            words[usize::from(EECP / 4)] &= !EHCI_BIOS_OWNED;
        }
    });
    config.words[0] = 0x8086 | 0x1e2d << 16;
    config.words[2] = 0x0c03_2000;
    config.words[4] = base as u32 | 4;
    config.words[5] = (base >> 32) as u32;
    config.words[usize::from(EECP / 4)] = EHCI_LEGSUP | EHCI_BIOS_OWNED;
    config.words[usize::from(EECP / 4) + 1] = 0xe03f_0001;
    let mut function = Function::new(
        crate::bus::pci::Address {
            segment: 0,
            bus: 0,
            device: 0x1d,
            function: 0,
        },
        config,
    )
    .unwrap_or_else(|_| panic!("modelled function"));
    let handoff = Handoff {
        ecam: Vec::new(),
        mapped: alloc::vec![base..base + 4096],
        dma: 0..0,
        clock: clock(),
        firmware: None,
        completion: None,
    };
    // SAFETY: BAR0 is the page above, which nothing else reaches.
    let handed = unsafe {
        early(
            &mut function,
            &handoff,
            crate::minimum_model::owned(clock()).as_ref(),
        )
    };
    assert_eq!(
        handed,
        Some(Ok(Handed {
            from_firmware: true,
            forced: false,
            legacy_released: true,
        }))
    );
    let words = &function.config().words;
    assert_eq!(words[usize::from(EECP / 4)], EHCI_LEGSUP | 1 << 24);
    assert_eq!(words[usize::from(EECP / 4) + 1], 0);
    // The command register is restored, and the ports go to the companions.
    assert_eq!(words[1], 0);
    assert_eq!(page.0[(0x20 + EHCI_CONFIGFLAG) / 4], 0);
}

mod minimum;
