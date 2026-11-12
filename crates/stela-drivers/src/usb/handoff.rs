// SPDX-License-Identifier: GPL-2.0
//! Taking USB host controllers from firmware before any driver binds, as
//! Linux's `quirk_usb_early_handoff` in `drivers/usb/host/pci-quirks.c` does.
//!
//! Firmware's USB legacy support emulates an i8042 keyboard and mouse through
//! SMIs and can trap ports 0x60 and 0x64 until every host controller is handed
//! over. Each handoff takes the controller through its ownership semaphore
//! (EHCI, xHCI), its interrupt-routing bit (OHCI) or its legacy-support
//! register (UHCI), turns its SMIs off, and halts it with its interrupts
//! masked. Firmware that does not let go in time loses the controller anyway,
//! as in Linux; the result records that.
use crate::bus::pci::regs;
use crate::{
    Clock, Handoff,
    bus::{
        mmio::Mmio,
        pci::{Bar, ConfigSpace, Function},
    },
};
use core::fmt;
use stela_driver_base::bus::port;

/// The USB host controller interfaces, by PCI programming interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Uhci,
    Ohci,
    Ehci,
    Xhci,
}

impl Kind {
    /// The interface of a serial-bus USB function, or none for other classes.
    pub fn of(class: [u8; 3]) -> Option<Self> {
        match class {
            [0x0c, 0x03, 0x00] => Some(Self::Uhci),
            [0x0c, 0x03, 0x10] => Some(Self::Ohci),
            [0x0c, 0x03, 0x20] => Some(Self::Ehci),
            [0x0c, 0x03, 0x30] => Some(Self::Xhci),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Uhci => "UHCI",
            Self::Ohci => "OHCI",
            Self::Ehci => "EHCI",
            Self::Xhci => "xHCI",
        }
    }
}

/// Why a controller could not be handed over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Minimum(crate::bus::pci::power::Error),
    Config(crate::bus::pci::ConfigError),
    /// The function has no BAR of the kind its interface decodes registers in.
    NoRegisters,
    /// Its register BAR lies outside the ranges the mount mapped uncached.
    Unmapped,
    /// Its registers read as all ones.
    Removed,
    /// Its capability registers point outside their space, or its extended
    /// capability list loops before any legacy-support capability.
    Malformed,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(f),
            Self::Minimum(error) => write!(f, "USB handoff minimum delay: {error:?}"),
            Self::NoRegisters => f.write_str("no register BAR"),
            Self::Unmapped => f.write_str("register BAR not mapped"),
            Self::Removed => f.write_str("registers read as all ones"),
            Self::Malformed => f.write_str("malformed capability registers"),
        }
    }
}

/// What a completed handoff found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Handed {
    /// Firmware owned the controller.
    pub from_firmware: bool,
    /// Firmware did not release it in time; the forced handoff was attempted.
    pub forced: bool,
    /// No legacy interception capability, or release and disabled SMIs read back.
    pub(crate) legacy_released: bool,
}

impl Handed {
    const NO_LEGACY: Self = Self {
        from_firmware: false,
        forced: false,
        legacy_released: true,
    };
}

impl Default for Handed {
    fn default() -> Self {
        Self::NO_LEGACY
    }
}

/// Ordered dword register accesses: what the xHCI legacy-support step needs.
pub(crate) trait Dwords {
    fn read32(&mut self, offset: usize) -> u32;
    fn write32(&mut self, offset: usize, value: u32);
}

/// Ordered register accesses of one host controller.
pub(crate) trait Registers: Dwords {
    fn read8(&mut self, offset: usize) -> u8;
    fn read16(&mut self, offset: usize) -> u16;
    fn write16(&mut self, offset: usize, value: u16);
}

impl Dwords for Mmio {
    fn read32(&mut self, offset: usize) -> u32 {
        Mmio::read32(self, offset)
    }
    fn write32(&mut self, offset: usize, value: u32) {
        Mmio::write32(self, offset, value)
    }
}

impl Registers for Mmio {
    fn read8(&mut self, offset: usize) -> u8 {
        Mmio::read8(self, offset)
    }
    fn read16(&mut self, offset: usize) -> u16 {
        Mmio::read16(self, offset)
    }
    fn write16(&mut self, offset: usize, value: u16) {
        Mmio::write16(self, offset, value)
    }
}

/// A UHCI I/O BAR.
struct Ports(u16);

impl Registers for Ports {
    fn read8(&mut self, offset: usize) -> u8 {
        // SAFETY: `early`'s caller owns the controller and its decoded ports.
        unsafe { port::read8(self.0 + offset as u16) }
    }
    fn read16(&mut self, offset: usize) -> u16 {
        // SAFETY: as for `read8`.
        unsafe { port::read16(self.0 + offset as u16) }
    }
    fn write16(&mut self, offset: usize, value: u16) {
        // SAFETY: as for `read8`.
        unsafe { port::write16(self.0 + offset as u16, value) }
    }
}

impl Dwords for Ports {
    fn read32(&mut self, offset: usize) -> u32 {
        // SAFETY: as for `read8`.
        unsafe { port::read32(self.0 + offset as u16) }
    }
    fn write32(&mut self, offset: usize, value: u32) {
        // SAFETY: as for `read8`.
        unsafe { port::write32(self.0 + offset as u16, value) }
    }
}

/// Hand the USB host controller `function` over from firmware; none for a
/// function of another class. Decoding of its register BAR is on while this
/// runs and the command register is restored after success. Any failure leaves
/// the complete function with the caller, without a successful handoff receipt.
///
/// # Safety
/// `handoff` describes the running machine after ExitBootServices, and
/// nothing else drives the controller while this runs.
pub unsafe fn early<C: ConfigSpace>(
    function: &mut Function<C>,
    handoff: &Handoff,
    minimum: &dyn crate::bus::pci::power::Timer,
) -> Option<Result<Handed, Error>> {
    let kind = Kind::of(function.class)?;
    Some(prepare(function, handoff, kind, minimum))
}

fn prepare<C: ConfigSpace>(
    function: &mut Function<C>,
    handoff: &Handoff,
    kind: Kind,
    minimum: &dyn crate::bus::pci::power::Timer,
) -> Result<Handed, Error> {
    minimum.check().map_err(Error::Minimum)?;
    let clock = handoff.clock;
    let bars = function.bars().map_err(Error::Config)?;
    let config = function.config();
    let command = config.read16(regs::COMMAND).map_err(Error::Config)?;
    if command == u16::MAX {
        return Err(Error::Removed);
    }
    let handed = if kind == Kind::Uhci {
        let port = bars
            .iter()
            .find_map(|bar| match bar {
                Some(Bar::Io { port, size }) if *size >= 0x20 && *port <= 0xffe0 => {
                    Some(*port as u16)
                }
                _ => None,
            })
            .ok_or(Error::NoRegisters)?;
        config
            .write16(regs::COMMAND, command | regs::COMMAND_IO)
            .map_err(Error::Config)?;
        uhci(config, &mut Ports(port), minimum)?
    } else {
        let (mut mmio, bytes) = memory(&bars, handoff)?;
        config
            .write16(regs::COMMAND, command | regs::COMMAND_MEMORY)
            .map_err(Error::Config)?;
        match kind {
            Kind::Ohci => ohci(&mut mmio, bytes, clock, minimum)?,
            Kind::Ehci => ehci(config, &mut mmio, bytes, clock, minimum)?,
            _ => xhci(&mut mmio, bytes, clock, minimum)?,
        }
    };
    // A failed access supplies no permission for rollback. Return immediately
    // with the function still owned by the caller; restore only after success.
    config
        .write16(regs::COMMAND, command)
        .map_err(Error::Config)?;
    Ok(handed)
}

/// BAR0's registers and their size, after checking that the mount mapped them.
fn memory(bars: &[Option<Bar>], handoff: &Handoff) -> Result<(Mmio, usize), Error> {
    let range = bars[0]
        .and_then(|bar| bar.memory())
        .ok_or(Error::NoRegisters)?;
    if !handoff.is_mapped(&range) {
        return Err(Error::Unmapped);
    }
    let bytes = usize::try_from(range.end - range.start).map_err(|_| Error::Malformed)?;
    // SAFETY: mapped uncached by the mount and owned by the caller.
    Ok((unsafe { Mmio::new(range.start, bytes) }, bytes))
}

const UHCI_LEGSUP: u16 = 0xc0;
const UHCI_LEGSUP_RO: u16 = 0x5040;
const UHCI_LEGSUP_RWC: u16 = 0x8f00;
const UHCI_LEGSUP_ENABLE: u16 = !(UHCI_LEGSUP_RO | UHCI_LEGSUP_RWC);
const UHCI_USBCMD: usize = 0;
const UHCI_USBINTR: usize = 4;
const UHCI_RUN: u16 = 1;
const UHCI_HCRESET: u16 = 2;
const UHCI_EGSM: u16 = 8;
const UHCI_CONFIGURE: u16 = 0x40;
const UHCI_INTR_RESUME: u16 = 2;

/// `uhci_check_and_reset_hc`: reset a controller firmware set up, which
/// clears the legacy-support SMI and PIRQ enables.
fn uhci<C: ConfigSpace>(
    config: &mut C,
    ports: &mut impl Registers,
    minimum: &dyn crate::bus::pci::power::Timer,
) -> Result<Handed, Error> {
    minimum.check().map_err(Error::Minimum)?;
    let legsup = config.read16(UHCI_LEGSUP).map_err(Error::Config)?;
    let command = ports.read16(UHCI_USBCMD);
    if legsup == u16::MAX || command == u16::MAX {
        return Err(Error::Removed);
    }
    let from_firmware = legsup & UHCI_LEGSUP_ENABLE != 0;
    if from_firmware
        || command & UHCI_RUN != 0
        || command & UHCI_CONFIGURE == 0
        || command & UHCI_EGSM == 0
        || ports.read16(UHCI_USBINTR) & !UHCI_INTR_RESUME != 0
    {
        config
            .write16(UHCI_LEGSUP, UHCI_LEGSUP_RWC)
            .map_err(Error::Config)?;
        ports.write16(UHCI_USBCMD, UHCI_HCRESET);
        minimum.wait_us(5).map_err(Error::Minimum)?;
        ports.write16(UHCI_USBINTR, 0);
        ports.write16(UHCI_USBCMD, 0);
    }
    Ok(Handed {
        from_firmware,
        forced: false,
        legacy_released: config
            .read16(UHCI_LEGSUP)
            .is_ok_and(|value| value & UHCI_LEGSUP_ENABLE == 0),
    })
}

const OHCI_CONTROL: usize = 0x04;
const OHCI_CMDSTATUS: usize = 0x08;
const OHCI_INTRENABLE: usize = 0x10;
const OHCI_INTRDISABLE: usize = 0x14;
const OHCI_FMINTERVAL: usize = 0x34;
const OHCI_CTRL_IR: u32 = 1 << 8;
const OHCI_CTRL_RWC: u32 = 1 << 9;
const OHCI_INTR_OC: u32 = 1 << 30;
const OHCI_OCR: u32 = 1 << 3;
const OHCI_HCR: u32 = 1;
const OHCI_HANDOFF_US: u64 = 500_000;

/// `quirk_usb_handoff_ohci`: request ownership change while firmware routes
/// interrupts to SMI, then reset the controller into USB_RESET with
/// InterruptRouting clear and HcFmInterval preserved.
fn ohci(
    registers: &mut impl Registers,
    bytes: usize,
    clock: Clock,
    minimum: &dyn crate::bus::pci::power::Timer,
) -> Result<Handed, Error> {
    minimum.check().map_err(Error::Minimum)?;
    if bytes < 0x40 {
        return Err(Error::Malformed);
    }
    let control = registers.read32(OHCI_CONTROL);
    if control == u32::MAX {
        return Err(Error::Removed);
    }
    let from_firmware = control & OHCI_CTRL_IR != 0;
    let mut forced = false;
    if from_firmware {
        registers.write32(OHCI_INTRENABLE, OHCI_INTR_OC);
        registers.write32(OHCI_CMDSTATUS, OHCI_OCR);
        let released = clock.wait(OHCI_HANDOFF_US, || {
            registers.read32(OHCI_CONTROL) & OHCI_CTRL_IR == 0
        });
        if !released {
            minimum.wait_us(OHCI_HANDOFF_US).map_err(Error::Minimum)?;
            forced = registers.read32(OHCI_CONTROL) & OHCI_CTRL_IR != 0;
        }
    }
    registers.write32(OHCI_INTRDISABLE, u32::MAX);
    registers.write32(OHCI_CONTROL, control & OHCI_CTRL_RWC);
    registers.read32(OHCI_CONTROL);
    let interval = registers.read32(OHCI_FMINTERVAL);
    registers.write32(OHCI_CMDSTATUS, OHCI_HCR);
    clock.wait(30, || registers.read32(OHCI_CMDSTATUS) & OHCI_HCR == 0);
    registers.write32(OHCI_FMINTERVAL, interval);
    Ok(Handed {
        from_firmware,
        forced,
        legacy_released: registers.read32(OHCI_CONTROL) & OHCI_CTRL_IR == 0,
    })
}

const EHCI_HCCPARAMS: usize = 0x08;
const EHCI_USBCMD: usize = 0;
const EHCI_USBSTS: usize = 4;
const EHCI_USBINTR: usize = 8;
const EHCI_CONFIGFLAG: usize = 0x40;
const EHCI_RUN: u32 = 1;
const EHCI_HALTED: u32 = 1 << 12;
const EHCI_LEGSUP: u32 = 1;
const EHCI_BIOS_OWNED: u32 = 1 << 16;
pub(super) const EHCI_LEGACY_SMI_ENABLES: u32 = 0xe03f;
const EHCI_HANDOFF_US: u64 = 1_000_000;

/// `quirk_usb_disable_ehci` and `ehci_bios_handoff`: take the legacy-support
/// semaphore in configuration space, clear every SMI enable, then halt the
/// controller with its interrupts off.
fn ehci<C: ConfigSpace>(
    config: &mut C,
    registers: &mut impl Registers,
    bytes: usize,
    clock: Clock,
    minimum: &dyn crate::bus::pci::power::Timer,
) -> Result<Handed, Error> {
    minimum.check().map_err(Error::Minimum)?;
    let parameters = registers.read32(EHCI_HCCPARAMS);
    if parameters == u32::MAX {
        return Err(Error::Removed);
    }
    let operational = usize::from(registers.read8(0));
    if operational < 0x10 || operational + EHCI_CONFIGFLAG + 4 > bytes {
        return Err(Error::Malformed);
    }
    let mut handed = None;
    let mut offset = ((parameters >> 8) & 0xff) as u16;
    let mut count = 64;
    while offset != 0 {
        count -= 1;
        if count == 0 {
            break;
        }
        if offset < 0x40 || offset & 3 != 0 {
            return Err(Error::Malformed);
        }
        let capability = config.read32(offset).map_err(Error::Config)?;
        if capability == u32::MAX {
            return Err(Error::Removed);
        }
        match capability & 0xff {
            EHCI_LEGSUP => {
                let current = ehci_bios(config, offset, capability, clock, minimum)?;
                handed = Some(Handed {
                    legacy_released: current.legacy_released
                        && handed.is_none_or(|earlier: Handed| earlier.legacy_released),
                    ..current
                });
                if handed.is_some_and(|handed| handed.from_firmware) {
                    registers.write32(operational + EHCI_CONFIGFLAG, 0);
                }
            }
            0 => break,
            _ => {}
        }
        offset = ((capability >> 8) & 0xff) as u16;
    }
    if count == 0 && handed.is_none() {
        return Err(Error::Malformed);
    }
    if registers.read32(operational + EHCI_USBSTS) & EHCI_HALTED == 0 {
        let command = registers.read32(operational + EHCI_USBCMD);
        registers.write32(operational + EHCI_USBCMD, command & !EHCI_RUN);
        for _ in 0..20 {
            registers.write32(operational + EHCI_USBSTS, 0x3f);
            minimum.wait_us(100).map_err(Error::Minimum)?;
            let status = registers.read32(operational + EHCI_USBSTS);
            if status == u32::MAX || status & EHCI_HALTED != 0 {
                break;
            }
        }
    }
    registers.write32(operational + EHCI_USBINTR, 0);
    registers.write32(operational + EHCI_USBSTS, 0x3f);
    let mut handed = handed.unwrap_or_default();
    handed.legacy_released &= count != 0;
    Ok(handed)
}

fn ehci_bios<C: ConfigSpace>(
    config: &mut C,
    offset: u16,
    capability: u32,
    clock: Clock,
    minimum: &dyn crate::bus::pci::power::Timer,
) -> Result<Handed, Error> {
    minimum.check().map_err(Error::Minimum)?;
    let from_firmware = capability & EHCI_BIOS_OWNED != 0;
    if from_firmware {
        config.write8(offset + 3, 1).map_err(Error::Config)?;
    }
    let mut failure = None;
    let released = clock.wait(EHCI_HANDOFF_US, || match config.read32(offset) {
        Ok(value) => value & EHCI_BIOS_OWNED == 0,
        Err(error) => {
            failure = Some(error);
            true
        }
    });
    if let Some(error) = failure {
        return Err(Error::Config(error));
    }
    let released = if released {
        true
    } else {
        minimum.wait_us(EHCI_HANDOFF_US).map_err(Error::Minimum)?;
        config.read32(offset).map_err(Error::Config)? & EHCI_BIOS_OWNED == 0
    };
    if !released {
        config.write8(offset + 2, 0).map_err(Error::Config)?;
    }
    config.write32(offset + 4, 0).map_err(Error::Config)?;
    Ok(Handed {
        from_firmware,
        forced: !released,
        legacy_released: config
            .read32(offset)
            .is_ok_and(|header| header & EHCI_BIOS_OWNED == 0)
            && config
                .read32(offset + 4)
                .is_ok_and(|control| control & EHCI_LEGACY_SMI_ENABLES == 0),
    })
}

const XHCI_HCCPARAMS1: usize = 0x10;
const XHCI_LEGACY: u32 = 1;
const XHCI_BIOS_OWNED: u32 = 1 << 16;
const XHCI_OS_OWNED: u32 = 1 << 24;
const XHCI_LEGACY_DISABLE_SMI: u32 = (0x7 << 1) | (0xff << 5) | (0x7 << 17);
const XHCI_LEGACY_SMI_ENABLES: u32 = 1 | (1 << 4) | (0x7 << 13);
const XHCI_LEGACY_SMI_EVENTS: u32 = 0x7 << 29;
const XHCI_USBCMD: usize = 0;
const XHCI_USBSTS: usize = 4;
const XHCI_RUN: u32 = 1;
const XHCI_IRQS: u32 = (1 << 2) | (1 << 3) | (1 << 10);
const XHCI_HALTED: u32 = 1;
const XHCI_NOT_READY: u32 = 1 << 11;
const XHCI_HANDOFF_US: u64 = 1_000_000;
const XHCI_READY_US: u64 = 5_000_000;
const XHCI_HALT_US: u64 = 16_000;

/// The USB Legacy Support step of the xHCI handoff (xHCI 7.1.1) on the
/// capability at `at`: set the OS Owned semaphore, take ownership once
/// firmware lets go or its time runs out, then clear the SMI enables and the
/// RW1C SMI events.
pub(crate) fn xhci_legacy<R: Dwords + ?Sized>(
    registers: &mut R,
    at: usize,
    clock: Clock,
    minimum: &dyn crate::bus::pci::power::Timer,
) -> Result<Handed, crate::bus::pci::power::Error> {
    minimum.check()?;
    let header = registers.read32(at);
    registers.write32(at, header | XHCI_OS_OWNED);
    let released = clock.wait(XHCI_HANDOFF_US, || {
        registers.read32(at) & XHCI_BIOS_OWNED == 0
    });
    let released = if released {
        true
    } else {
        // Nominal timeout data cannot shorten the physical firmware grace.
        minimum.wait_us(XHCI_HANDOFF_US)?;
        registers.read32(at) & XHCI_BIOS_OWNED == 0
    };
    if !released {
        let header = registers.read32(at);
        registers.write32(at, header & !XHCI_BIOS_OWNED | XHCI_OS_OWNED);
    }
    let control = registers.read32(at + 4);
    registers.write32(
        at + 4,
        control & XHCI_LEGACY_DISABLE_SMI | XHCI_LEGACY_SMI_EVENTS,
    );
    Ok(Handed {
        from_firmware: header & XHCI_BIOS_OWNED != 0,
        forced: !released,
        legacy_released: registers.read32(at) & (XHCI_BIOS_OWNED | XHCI_OS_OWNED) == XHCI_OS_OWNED
            && registers.read32(at + 4) & XHCI_LEGACY_SMI_ENABLES == 0,
    })
}

/// `quirk_usb_handoff_xhci`: take the USB Legacy Support semaphore, clear
/// the SMI enables and events, then halt the controller with its interrupts
/// off once it is ready.
fn xhci(
    registers: &mut impl Registers,
    bytes: usize,
    clock: Clock,
    minimum: &dyn crate::bus::pci::power::Timer,
) -> Result<Handed, Error> {
    let parameters = registers.read32(XHCI_HCCPARAMS1);
    if parameters == u32::MAX {
        return Err(Error::Removed);
    }
    let operational = usize::from(registers.read8(0));
    if operational < 0x20 || operational + XHCI_USBSTS + 4 > bytes {
        return Err(Error::Malformed);
    }
    let mut handed = Handed::NO_LEGACY;
    let mut offset = (parameters >> 16) as usize * 4;
    while offset != 0 {
        if offset + 8 > bytes {
            return Err(Error::Malformed);
        }
        let header = registers.read32(offset);
        if header == u32::MAX {
            return Err(Error::Removed);
        }
        if header & 0xff == XHCI_LEGACY {
            handed = xhci_legacy(registers, offset, clock, minimum).map_err(Error::Minimum)?;
            break;
        }
        let next = (header >> 8) & 0xff;
        offset = if next == 0 {
            0
        } else {
            offset + next as usize * 4
        };
    }
    clock.wait(XHCI_READY_US, || {
        registers.read32(operational + XHCI_USBSTS) & XHCI_NOT_READY == 0
    });
    let command = registers.read32(operational + XHCI_USBCMD);
    registers.write32(operational + XHCI_USBCMD, command & !(XHCI_RUN | XHCI_IRQS));
    clock.wait(XHCI_HALT_US, || {
        registers.read32(operational + XHCI_USBSTS) & XHCI_HALTED != 0
    });
    Ok(handed)
}

const INTEL: u16 = 0x8086;
const INTEL_XUSB2PR: u16 = 0xd0;
const INTEL_USB2PRM: u16 = 0xd4;
const INTEL_USB3_PSSEN: u16 = 0xd8;
const INTEL_USB3PRM: u16 = 0xdc;
/// Sony VAIO T-series (subsystem 104d:90a8), which cannot switch its ports.
const SONY_FIXED_PORTS: u32 = 0x90a8_104d;

/// The ports an Intel xHCI took from its EHCI, as the routing registers read
/// back after the switch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Switched {
    /// USB3_PSSEN: ports whose SuperSpeed terminations are on.
    pub superspeed: u32,
    /// XUSB2PR: USB 2 ports routed to the xHCI.
    pub usb2: u32,
}

/// `usb_enable_intel_xhci_ports`, which Linux runs at the end of
/// `quirk_usb_handoff_xhci` for every Intel xHCI: on Intel 7-, 8- and
/// 9-series PCHs firmware may leave the switchable ports on EHCI, and the
/// xHCI then sees no device on them. When an Intel EHCI exists
/// (`intel_ehci`), turn on SuperSpeed for every port USB3PRM allows and route
/// every USB 2 port XUSB2PRM allows to the xHCI. None when the function is
/// no Intel xHCI, there is no Intel EHCI, or no port is switchable.
pub fn switch_intel_ports<C: ConfigSpace>(
    function: &mut Function<C>,
    intel_ehci: bool,
) -> Result<Option<Switched>, Error> {
    if function.vendor != INTEL || Kind::of(function.class) != Some(Kind::Xhci) || !intel_ehci {
        return Ok(None);
    }
    let config = function.config();
    if config
        .read32(regs::SUBSYSTEM_VENDOR_ID)
        .map_err(Error::Config)?
        == SONY_FIXED_PORTS
    {
        return Ok(None);
    }
    let superspeed = config.read32(INTEL_USB3PRM).map_err(Error::Config)?;
    let usb2 = config.read32(INTEL_USB2PRM).map_err(Error::Config)?;
    if superspeed == u32::MAX || usb2 == u32::MAX {
        return Err(Error::Removed);
    }
    if superspeed == 0 && usb2 == 0 {
        return Ok(None);
    }
    config
        .write32(INTEL_USB3_PSSEN, superspeed)
        .map_err(Error::Config)?;
    config.write32(INTEL_XUSB2PR, usb2).map_err(Error::Config)?;
    Ok(Some(Switched {
        superspeed: config.read32(INTEL_USB3_PSSEN).map_err(Error::Config)?,
        usb2: config.read32(INTEL_XUSB2PR).map_err(Error::Config)?,
    }))
}

#[cfg(test)]
mod tests;
