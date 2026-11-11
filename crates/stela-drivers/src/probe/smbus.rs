//! Intel i801 SMBus hosts on PCI, attached as Linux `i801_probe` attaches
//! them: through the I/O window BAR 4 decodes, enabled in SMBus mode, under
//! the ACPI arbitration `i801_acpi_probe` sets up, with Host Notify enabled
//! for the touchpad companions of the PS/2 mouse port. The I/O window is used
//! rather than the memory BAR Linux prefers: both reach the same registers,
//! and the owned mappings offer no byte-wide access to them.
use super::{NativeServices, PlatformError, native};
use crate::{
    Clock,
    bus::{
        pci::{Address, Bar, ConfigSpace, owned::OwnedConfig, regs},
        smbus::{
            self, Bus, ErrorKind, Transaction,
            i801::{
                self, ControllerInfo, I801, PortIo,
                i2c_block::{Controller, OwnedConfiguration},
            },
        },
    },
    platform::acpi::{IoWindow, Session},
};
use alloc::{rc::Rc, string::String};
use core::cell::RefCell;

/// The i801's SMBus I/O window, `SMBBAR`.
const SMBUS_BAR: usize = 4;
/// SMBHSTCFG and its bits, as Linux `i801_setup_hstcfg` sets them.
const HOST_CONFIGURATION: u16 = 0x40;
const HOST_ENABLED: u8 = 1;
const HOST_SMI: u8 = 1 << 1;
const HOST_I2C: u8 = 1 << 2;

type Owned = Controller<PortIo, Clock, OwnedConfiguration<OwnedConfig, ()>>;

/// An attached i801, shared by the companions on its bus.
#[derive(Clone)]
pub struct Host(Rc<RefCell<State>>);

struct State {
    location: String,
    controller: Owned,
    /// The AML arbitration of the host's I/O window, when its ACPI companion
    /// took a handler.
    firmware: Option<IoWindow>,
    host_notify: bool,
    /// Host Notify senders not yet consumed, a bit per seven-bit address.
    notified: u128,
}

impl State {
    /// The controller, unless AML has accessed its window, after which
    /// firmware owns it as Linux `i801_access` then refuses.
    fn controller(&mut self) -> Result<&mut Owned, smbus::Error> {
        if self.firmware.as_ref().is_some_and(IoWindow::firmware_owned) {
            return Err(ErrorKind::FirmwareOwned.into());
        }
        Ok(&mut self.controller)
    }
}

impl Host {
    /// The host's PCI location, such as `0000:00:1f.4`.
    pub fn location(&self) -> String {
        self.0.borrow().location.clone()
    }

    /// Whether the host takes Host Notify, which touchpad companions need.
    pub fn host_notify(&self) -> bool {
        self.0.borrow().host_notify
    }

    /// Whether a peripheral answers at `address`, as Linux
    /// `i2c_default_probe` asks with an SMBus Quick write.
    pub fn present(&self, address: smbus::Address) -> bool {
        self.0
            .borrow_mut()
            .controller()
            .and_then(|controller| controller.quick(address, false))
            .is_ok()
    }

    /// Whether a Host Notify from `address` arrived and is not yet retired;
    /// `retire` retires it. Every notification the host holds is collected
    /// first, so each sender keeps its own.
    pub fn notified(&self, address: smbus::Address, retire: bool) -> Result<bool, smbus::Error> {
        let mut state = self.0.borrow_mut();
        while let Some(sender) = state.controller()?.take_host_notify()? {
            // Linux `i2c_handle_smbus_host_notify` ignores a sender outside
            // the seven-bit address space.
            if sender < 0x80 {
                state.notified |= 1u128 << sender;
            }
        }
        let bit = 1u128 << address.get();
        let notified = state.notified & bit != 0;
        if retire {
            state.notified &= !bit;
        }
        Ok(notified)
    }
}

impl Bus for Host {
    fn transaction(
        &mut self,
        address: smbus::Address,
        transaction: Transaction<'_>,
        pec: bool,
    ) -> Result<(), smbus::Error> {
        self.0
            .borrow_mut()
            .controller()?
            .transaction(address, transaction, pec)
    }
}

/// Whether `function` is an i801 SMBus host.
pub(super) fn matches_identity(function: &crate::bus::pci::Identity) -> bool {
    function.class[..2] == [0x0c, 0x05]
        && ControllerInfo::from_pci_id(function.vendor, function.device).is_some()
}

/// Attach the i801 at `address`, as Linux `i801_probe` does.
///
/// # Safety
/// The mount grants port I/O to the function's I/O window, as the
/// [`NativeServices`] contract states, and nothing else drives the host.
pub(super) unsafe fn attach(
    services: &dyn NativeServices,
    session: &'static Session,
    address: Address,
    bridges: &[native::Bridge],
    clock: Clock,
) -> Result<Host, PlatformError> {
    let requester = services
        .claim_stopped(address)
        .map_err(PlatformError::Native)?;
    let mut function = requester.into_function();
    let info = ControllerInfo::from_pci_id(function.vendor, function.device)
        .ok_or(PlatformError::Refused("not an i801 SMBus host"))?;
    let Some(Bar::Io { port, size }) = function.bars().map_err(PlatformError::Config)?[SMBUS_BAR]
    else {
        return Err(PlatformError::Refused(
            "SMBus base address uninitialized, as Linux i801_probe reports it",
        ));
    };
    let window = u16::try_from(port)
        .ok()
        .zip(u16::try_from(port + size).ok())
        .filter(|&(start, _)| start != 0)
        .map(|(start, end)| start..end)
        .ok_or(PlatformError::Refused(
            "SMBus I/O window outside the port space",
        ))?;
    // i801_acpi_probe: a SystemIO handler at the companion, else no AML
    // operation region may overlap the window.
    let firmware = match native::companion(session, address, bridges)
        .map(|companion| companion.arbitrate_io(window.clone()))
    {
        Some(Ok(firmware)) => Some(firmware),
        _ => {
            if session
                .io_regions_overlapping(window.clone())
                .map_err(PlatformError::AcpiMethod)?
                != 0
            {
                return Err(PlatformError::Refused(
                    "ACPI operation regions overlap the SMBus I/O window, which no handler arbitrates",
                ));
            }
            None
        }
    };
    let config = function.config();
    let command = config
        .read16(regs::COMMAND)
        .map_err(PlatformError::Config)?;
    if command & regs::COMMAND_IO == 0 {
        config
            .write16(regs::COMMAND, command | regs::COMMAND_IO)
            .map_err(PlatformError::Config)?;
    }
    // i801_setup_hstcfg: SMBus timing, host enabled. A host whose interrupts
    // go to SMI# stays with firmware.
    let original = config
        .read8(HOST_CONFIGURATION)
        .map_err(PlatformError::Config)?;
    let setup = (original & !HOST_I2C) | HOST_ENABLED;
    if setup & HOST_SMI != 0 {
        return Err(PlatformError::Smbus(ErrorKind::FirmwareSmi.into()));
    }
    if setup != original {
        config
            .write8(HOST_CONFIGURATION, setup)
            .map_err(PlatformError::Config)?;
    }
    // SAFETY: the window is the function's decoded I/O BAR, which the mount
    // grants; AML reaches it only through the arbitration above.
    let registers = unsafe { PortIo::claim(window.start, info) }.map_err(PlatformError::Smbus)?;
    let host = I801::new(registers, clock, info, setup, i801::Config::default())
        .map_err(PlatformError::Smbus)?;
    // SAFETY: the claimed function's configuration is this probe's alone.
    let configuration = unsafe { OwnedConfiguration::claim(function, ()) }
        .map_err(|_| PlatformError::Refused("not an i801 SMBus host"))?;
    // SAFETY: as above; the host has no operation in flight.
    let mut controller = unsafe { Controller::claim(host, configuration) }
        .map_err(|(kind, _)| PlatformError::Smbus(kind.into()))?;
    let host_notify = info.has_host_notify();
    if host_notify {
        controller
            .enable_host_notify()
            .map_err(PlatformError::Smbus)?;
    }
    Ok(Host(Rc::new(RefCell::new(State {
        location: alloc::format!("{address}"),
        controller,
        firmware,
        host_notify,
        notified: 0,
    }))))
}
