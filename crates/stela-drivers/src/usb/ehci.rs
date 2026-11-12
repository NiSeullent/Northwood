//! Polled native EHCI 1.0 high-speed control, bulk and interrupt backend.
//!
//! This library owns its PCI, uncached BAR and requester-domain DMA resources.
//! It has no xHCI identities or native class-driver adapter. One
//! transfer runs at a time. Every inspection/cancellation stops the complete
//! active schedule, observes ASS=PSS=0 and drains posted DMA writes
//! before touching its reusable QH/qTD/bounce arena. Uncertain transitions
//! permanently quarantine the complete attachment, not just the failed TD.
//!
//! Hubs are driven through owned port operations ([`hub`]). Full- and
//! low-speed devices behind a high-speed hub run as split transactions
//! through its transaction translator: control, and interrupt for boot
//! keyboards and mice ([`input`]); full-speed bulk is refused. On a root
//! port they are handed to the companion, which has no driver here. ISO,
//! controller/PCI sleep and controller-specific quirks remain unsupported.
//! These boundaries make this an incomplete EHCI-family implementation.
mod descriptor;
mod detach;
pub mod host;
pub mod hub;
pub mod input;
mod native;
mod periodic;
mod recovery;
mod schedule;
mod suspend;
#[cfg(test)]
mod tests;

use crate::bus::dma::{
    mapping::{Direction, DmaError, DmaMapping},
    memory::{CacheAttribute, CpuMapping, MemoryError},
};
use alloc::{boxed::Box, rc::Rc, vec::Vec};
use core::{mem, ptr};
use native::Hardware;
pub use native::{Failure, NativeResources, attach};
use schedule::{Arena, Pending};

pub const CLASS: [u8; 3] = [0x0c, 0x03, 0x20];
pub const MAX_TRANSFER: usize = 65_536;
const TRANSITION_US: u64 = 100_000;
const HALT_US: u64 = 2_000; // Linux ehci_halt:16*125us.
const RESET_US: u64 = 250_000; // Linux ehci_reset:250*1000us.
const BIOS_HANDOFF_US: u64 = 1_000_000;
const CMD: usize = 0;
const STS: usize = 4;
const INTR: usize = 8;
const SEGMENT: usize = 0x10;
const PERIODIC: usize = 0x14;
const ASYNC: usize = 0x18;
const CONFIG: usize = 0x40;
const PORT: usize = 0x44;
const RUN: u32 = 1;
const RESET: u32 = 2;
const ASE: u32 = 1 << 5;
const PSE: u32 = 1 << 4;
const IAAD: u32 = 1 << 6;
const HALT: u32 = 1 << 12;
const ASS: u32 = 1 << 15;
const PSS: u32 = 1 << 14;
const FATAL: u32 = 1 << 4;
const CONNECT: u32 = 1;
const ENABLE: u32 = 1 << 2;
const OVERCURRENT: u32 = 1 << 4;
const PORT_RESET: u32 = 1 << 8;
const PORT_SUSPEND: u32 = 1 << 7;
const PORT_RESUME: u32 = 1 << 6;
const POWER: u32 = 1 << 12;
const OWNER: u32 = 1 << 13;
const PORT_CHANGES: u32 = (1 << 1) | (1 << 3) | (1 << 5);
/// Devices the table holds behind hubs, past the root ports.
const CHILDREN: usize = 32;
/// Hub tiers below a root port, as USB 2.0 allows.
const MAX_TIERS: usize = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Minimum(crate::bus::pci::power::Error),
    Config(crate::bus::pci::ConfigError),
    Mapping(crate::bus::mmio::owned::Error),
    Irq(crate::bus::irq::IrqError),
    Unsupported(&'static str),
    Invalid,
    Stale,
    Busy,
    Timeout,
    Removed,
    Device(u32),
    Quarantined,
    Dma(DmaError),
    Preparation(crate::bus::pci::requester::preparation::Error),
    Memory(MemoryError),
    Register,
    Exhausted,
    Clock,
}
impl Error {
    /// A pure STALL: the endpoint halted with no other transaction error.
    pub fn stalled(&self) -> bool {
        matches!(self, Self::Device(token) if token & schedule::ERRORS == schedule::HALTED)
    }
}
impl From<DmaError> for Error {
    fn from(value: DmaError) -> Self {
        Self::Dma(value)
    }
}
impl From<MemoryError> for Error {
    fn from(value: MemoryError) -> Self {
        Self::Memory(value)
    }
}

/// An unforgeable identity minted only after this owner's HS root reset.
/// Copies remain tombstones after disconnect, another reset or controller stop.
#[derive(Clone)]
pub struct DeviceId {
    owner: Rc<()>,
    generation: u64,
    /// The root port the device hangs from, itself or through hubs.
    port: u8,
    /// Its device-table index plus one: the root port for a root device.
    slot: u8,
}
/// A bulk endpoint from a configuration fetched and selected over USB.
#[derive(Clone)]
pub struct BulkPipe {
    device: DeviceId,
    configuration: u64,
    endpoint: u8,
}
/// A high-speed interrupt endpoint from an acknowledged configuration.
/// Configuration/root generations permanently invalidate old copies.
#[derive(Clone)]
pub struct InterruptPipe {
    device: DeviceId,
    configuration: u64,
    endpoint: u8,
    interval_microframes: u16,
}
impl InterruptPipe {
    pub fn endpoint_address(&self) -> u8 {
        self.endpoint
    }
    pub fn interval_microframes(&self) -> u16 {
        self.interval_microframes
    }
}
impl BulkPipe {
    pub fn endpoint_address(&self) -> u8 {
        self.endpoint
    }
}
impl DeviceId {
    pub fn root_port(&self) -> u8 {
        self.port
    }
    /// Whether the device is a root port's own device, not one behind a hub.
    pub fn is_root(&self) -> bool {
        self.slot == self.port
    }
}
/// A USB 2.0 device's bus speed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Speed {
    Low,
    Full,
    High,
}
/// Where a device behind a hub hangs: its hub's device-table entry, the
/// hub's generation when the device was attached, and the hub port.
#[derive(Clone, Copy)]
struct Parent {
    index: usize,
    generation: u64,
    port: u8,
}
/// One specific transfer; old tickets cannot cancel or consume a newer transfer.
pub struct TransferId {
    device: DeviceId,
    serial: u64,
}
/// Private ownership of one suspended root. Dropping it keeps the
/// suspended device, USB address and all native resources. It never reenables
/// the old public device/pipe identities or authorizes platform sleep/reset.
pub struct SuspendedPort {
    owner: Rc<()>,
    generation: u64,
    port: u8,
}
#[derive(Debug, Eq, PartialEq)]
pub struct Completion {
    pub actual: usize,
    pub data: Vec<u8>,
}

struct Device {
    generation: u64,
    address: u8,
    /// The root port it hangs from, itself or through hubs.
    root_port: u8,
    /// The hub port it hangs from; None for a root device.
    parent: Option<Parent>,
    speed: Speed,
    /// A full- or low-speed device's transaction translator: the high-speed
    /// hub's address and the hub port the translated traffic leaves on.
    tt: Option<(u8, u8)>,
    /// EP0's maximum packet size.
    control_packet: u16,
    configuration: u64,
    configuration_value: u8,
    endpoints: [Option<Endpoint>; 32],
    suspended: bool,
}
impl Device {
    /// Whether the device hangs from root port `port` through hubs.
    fn root(&self, port: u8) -> bool {
        self.parent.is_some() && self.root_port == port
    }
}
#[derive(Clone, Copy)]
struct Endpoint {
    toggle: bool,
    ping: bool,
    halted: bool,
    periodic: Option<periodic::Recipe>,
}
#[derive(Clone, Copy, Eq, PartialEq)]
enum State {
    Running,
    Failed,
    Retiring,
    Stopped,
}

enum RootReset {
    Empty,
    Ready,
}

pub struct Controller {
    hardware: Option<Box<dyn Hardware>>,
    arena: Option<Arena>,
    owner: Rc<()>,
    state: State,
    fault: Option<Error>,
    legacy_released: Option<bool>,
    operational: usize,
    /// The device table: each root port's device, then the devices behind
    /// hubs ([`CHILDREN`] entries).
    ports: Vec<Option<Device>>,
    /// The root ports, the table's first entries.
    roots: usize,
    next_generation: u64,
    next_transfer: u64,
    default_port: Option<u8>,
    power_control: bool,
    companions: bool,
    pending: Option<Pending>,
    /// The interrupt transfer another transfer preempted, by serial, and
    /// what became of it ([`Controller::preempt`]).
    preempted: Option<(u64, Result<Completion, Error>)>,
}
impl Controller {
    fn hw(&mut self) -> &mut dyn Hardware {
        &mut **self.hardware.as_mut().expect("retained attachment")
    }
    fn read(&mut self, at: usize) -> Result<u32, Error> {
        if let Some(error) = self.fault {
            return Err(error);
        }
        let value = self.hw().read32(at)?;
        if value == u32::MAX {
            Err(Error::Removed)
        } else {
            Ok(value)
        }
    }
    fn write(&mut self, at: usize, value: u32) -> Result<(), Error> {
        if let Some(error) = self.fault {
            return Err(error);
        }
        self.hw().write32(at, value)
    }
    fn op(&mut self, at: usize) -> Result<u32, Error> {
        self.read(self.operational + at)
    }
    fn put(&mut self, at: usize, value: u32) -> Result<(), Error> {
        self.write(self.operational + at, value)
    }
    fn ready(&self) -> Result<(), Error> {
        if let Some(error) = self.fault {
            return Err(error);
        }
        if self.state == State::Running {
            Ok(())
        } else {
            Err(Error::Quarantined)
        }
    }
    fn identity(&self, id: &DeviceId) -> Result<usize, Error> {
        if !Rc::ptr_eq(&self.owner, &id.owner) {
            return Err(Error::Stale);
        }
        let index = usize::from(id.slot).checked_sub(1).ok_or(Error::Stale)?;
        if self
            .ports
            .get(index)
            .and_then(Option::as_ref)
            .is_none_or(|device| device.generation != id.generation || device.suspended)
            || !self.attached(index)
        {
            return Err(Error::Stale);
        }
        self.ready()?;
        Ok(index)
    }
    /// Whether every hub between the device at `index` and its root is
    /// still the one it was attached through.
    fn attached(&self, mut index: usize) -> bool {
        for _ in 0..=MAX_TIERS {
            let Some(device) = self.ports.get(index).and_then(Option::as_ref) else {
                return false;
            };
            let Some(parent) = device.parent else {
                return true;
            };
            if self
                .ports
                .get(parent.index)
                .and_then(Option::as_ref)
                .is_none_or(|hub| hub.generation != parent.generation)
            {
                return false;
            }
            index = parent.index;
        }
        false
    }
    /// Retire every device behind a hub on root port `port`: a reset or
    /// disconnect of the root takes them all off the bus.
    fn retire_children(&mut self, port: u8) {
        let roots = self.roots;
        for (index, entry) in self.ports.iter_mut().enumerate().skip(roots) {
            if entry.as_ref().is_some_and(|device| device.root(port)) {
                *entry = None;
                if self.default_port == Some((index + 1) as u8) {
                    self.default_port = None;
                }
            }
        }
    }
    fn port_at(&self, index: usize) -> usize {
        self.operational + PORT + 4 * index
    }
    fn live(&mut self, id: &DeviceId) -> Result<usize, Error> {
        let index = self.identity(id)?;
        let root = usize::from(id.port) - 1;
        let result = self.read(self.port_at(root));
        let value = self.guard(result)?;
        if value
            & (CONNECT
                | ENABLE
                | OWNER
                | OVERCURRENT
                | PORT_RESET
                | PORT_CHANGES
                | PORT_SUSPEND
                | PORT_RESUME)
            != CONNECT | ENABLE
        {
            self.retire_observed_root(root, value)?;
            return Err(Error::Stale);
        }
        let result = self.health();
        self.guard(result)?;
        Ok(index)
    }
    fn retire_observed_root(&mut self, index: usize, value: u32) -> Result<(), Error> {
        let port = (index + 1) as u8;
        self.ports[index] = None; // Revoke public identity before any other work.
        self.retire_children(port);
        // Losing an identity is not physical isolation. A connected device can
        // still answer its old assigned address (or address0) despite PEC/CSC,
        // overcurrent or an unfinished reset. Never free that bus namespace.
        if value & CONNECT != 0 {
            return self.fail(Error::Stale);
        }
        if self.default_port == Some(port) {
            self.default_port = None;
        }
        if self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.root_port() == port)
        {
            return self.fail(Error::Stale);
        }
        Ok(())
    }
    fn health(&mut self) -> Result<(), Error> {
        let status = self.op(STS)?;
        if status & (FATAL | HALT) != 0
            || status & PSS != 0 && !self.pending.as_ref().is_some_and(|p| p.is_periodic())
        {
            return Err(Error::Device(status));
        }
        Ok(())
    }
    fn wait(&mut self, at: usize, mask: u32, wanted: u32, timeout: u64) -> Result<(), Error> {
        let start = self.hw().now_us();
        loop {
            let value = self.read(at)?;
            if value & mask == wanted {
                return Ok(());
            }
            if self.hw().now_us().wrapping_sub(start) >= timeout {
                return Err(Error::Timeout);
            }
            self.hw().pause();
        }
    }
    fn fail<T>(&mut self, error: Error) -> Result<T, Error> {
        // Retirement cannot reopen after a later detach or release refusal.
        if !matches!(self.state, State::Retiring | State::Stopped) {
            self.state = State::Failed;
        }
        if matches!(
            error,
            Error::Minimum(_) | Error::Config(_) | Error::Mapping(_) | Error::Irq(_)
        ) {
            Err(*self.fault.get_or_insert(error))
        } else {
            // Ordinary USB/protocol faults retain their existing explicit
            // contained recovery policy. They issue no new native grant.
            Err(error)
        }
    }
    fn delay_minimum(&mut self, microseconds: u64) -> Result<(), Error> {
        if let Some(error) = self.fault {
            return Err(error);
        }
        // Admission closes before the actual source callback, including unwind.
        self.fault = Some(Error::Quarantined);
        match self.hw().delay_us(microseconds) {
            Ok(()) => {
                self.fault = None;
                Ok(())
            }
            Err(error) => {
                self.fault = Some(error);
                self.state = State::Failed;
                Err(error)
            }
        }
    }
    /// Wait through this owner's actual physical minimum provider. A failed
    /// or unwound source permanently closes traffic and keeps all resources.
    pub fn wait_minimum(&mut self, microseconds: u64) -> Result<(), Error> {
        self.ready()?;
        self.delay_minimum(microseconds)
    }
    fn guard<T>(&mut self, result: Result<T, Error>) -> Result<T, Error> {
        match result {
            Ok(value) => Ok(value),
            Err(error) => self.fail(error),
        }
    }
    fn schedule_off(&mut self) -> Result<(), Error> {
        self.health()?;
        let command = self.op(CMD)?;
        // Never reverse a pending transition or stop an unowned engine.
        let engine = if self.pending.as_ref().is_some_and(|p| p.is_periodic()) {
            PSE
        } else {
            ASE
        };
        if command & (PSE | ASE) & !engine != 0 {
            return Err(Error::Unsupported("EHCI unexpected active schedule"));
        }
        self.wait(
            self.operational + STS,
            ASS | PSS,
            (if command & ASE != 0 { ASS } else { 0 }) | (if command & PSE != 0 { PSS } else { 0 }),
            TRANSITION_US,
        )?;
        self.put(CMD, command & !(ASE | PSE))?;
        self.wait(self.operational + STS, ASS | PSS, 0, TRANSITION_US)?;
        self.health()?;
        // ASS0/PSS0 is engine quiescence, not a remapper payload-write drain.
        self.hw().drain()?;
        let arena = self.arena.as_ref().ok_or(Error::Invalid)?;
        // SAFETY: ASS0/PSS0 + requester path drain.
        unsafe {
            arena.dma.sync_for_cpu(0..arena.bytes())?;
        }
        Ok(())
    }
    fn schedule_on(&mut self) -> Result<(), Error> {
        self.health()?;
        self.wait(self.operational + STS, ASS | PSS, 0, TRANSITION_US)?;
        let arena = self.arena.as_ref().ok_or(Error::Invalid)?;
        arena.dma.sync_for_device(0..arena.bytes())?;
        let command = self.op(CMD)?;
        let periodic = self.pending.as_ref().ok_or(Error::Invalid)?.is_periodic();
        if command & (ASE | PSE) != 0 {
            return Err(Error::Unsupported(
                "EHCI schedule not disabled before publication",
            ));
        }
        self.put(CMD, command | if periodic { PSE } else { ASE })?;
        let expected = if periodic { PSS } else { ASS };
        self.wait(self.operational + STS, ASS | PSS, expected, TRANSITION_US)?;
        self.health()
    }

    /// The controller's root ports, numbered from one.
    pub fn ports(&self) -> u8 {
        self.roots as u8
    }
    /// The lowest USB address no device here holds.
    pub fn unused_address(&self) -> Option<u8> {
        (1..=127).find(|address| {
            !self
                .ports
                .iter()
                .flatten()
                .any(|device| device.address == *address)
        })
    }
    /// Whether `id` still names a live device on a running controller.
    pub fn validate(&self, id: &DeviceId) -> Result<(), Error> {
        self.identity(id).map(drop)
    }

    /// Reset a powered root with no active TD. Another default-address owner
    /// must first be addressed or explicitly disconnected. Reset retires all
    /// old handles before its first hardware effect, including failed resets.
    pub fn reset_port(&mut self, port: u8) -> Result<DeviceId, Error> {
        self.ready()?;
        let index = usize::from(port)
            .checked_sub(1)
            .filter(|index| *index < self.roots)
            .ok_or(Error::Invalid)?;
        if self.pending.is_some() || self.default_port.is_some_and(|owner| owner != port) {
            return Err(Error::Busy);
        }
        if self.ports[index]
            .as_ref()
            .is_some_and(|device| device.suspended)
        {
            return Err(Error::Unsupported(
                "EHCI suspended root requires its retained resume owner",
            ));
        }
        let generation = self
            .next_generation
            .checked_add(1)
            .ok_or(Error::Exhausted)?;
        self.next_generation = generation;
        let untracked = self.ports[index].take().is_none() && self.default_port != Some(port);
        self.retire_children(port);
        // Keep any old address0 reservation throughout the owned transition.
        // A failed reset can leave that exact device reachable at address0.
        let result = match self.reset_root(index) {
            Ok(RootReset::Ready) => Ok(()),
            Ok(RootReset::Empty) if untracked => return Err(Error::Stale),
            Ok(RootReset::Empty) => Err(Error::Stale),
            Err(error) => Err(error),
        };
        if let Err(error) = result {
            // Only these paths observed OWNER handoff or a disabled
            // FS/LS port after completed reset. Every ambiguous failure closes
            // the whole controller, including old assigned address aliases.
            if matches!(
                error,
                Error::Unsupported("EHCI companion-owned root" | "EHCI FS/LS companion traffic")
            ) {
                if self.default_port == Some(port) {
                    self.default_port = None;
                }
            } else {
                self.state = State::Failed;
            }
            return Err(error);
        }
        self.ports[index] = Some(Device {
            generation,
            address: 0,
            root_port: port,
            parent: None,
            speed: Speed::High,
            tt: None,
            control_packet: 64,
            configuration: 0,
            configuration_value: 0,
            endpoints: [None; 32],
            suspended: false,
        });
        self.default_port = Some(port);
        Ok(DeviceId {
            owner: self.owner.clone(),
            generation,
            port,
            slot: port,
        })
    }
    fn reset_root(&mut self, index: usize) -> Result<RootReset, Error> {
        self.schedule_off()?;
        let at = self.port_at(index);
        let mut value = self.read(at)?;
        if value & OWNER != 0 {
            return Err(Error::Unsupported("EHCI companion-owned root"));
        }
        if self.power_control && value & POWER == 0 {
            self.write(at, (value & !PORT_CHANGES) | POWER)?;
            self.delay_minimum(20_000)?;
            value = self.read(at)?;
            if value & POWER == 0 {
                return Err(Error::Device(value));
            }
        }
        // An idle port that has no connected device never enters reset.
        // Keep this observation distinct from a disconnect during reset.
        if value & (CONNECT | ENABLE | OVERCURRENT | PORT_RESET | PORT_RESUME | PORT_SUSPEND) == 0 {
            return Ok(RootReset::Empty);
        }
        if value & CONNECT == 0 || value & OVERCURRENT != 0 {
            return Err(Error::Stale);
        }
        // Debounce connectivity, acknowledging only observed changes.
        self.write(at, (value & !PORT_CHANGES) | (value & PORT_CHANGES))?;
        self.delay_minimum(100_000)?;
        value = self.read(at)?;
        if value & CONNECT == 0 || value & (OWNER | OVERCURRENT | PORT_CHANGES) != 0 {
            return Err(Error::Stale);
        }
        self.write(at, (value & !(PORT_CHANGES | ENABLE)) | PORT_RESET)?;
        self.delay_minimum(50_000)?;
        value = self.read(at)?;
        self.write(at, value & !(PORT_CHANGES | PORT_RESET))?;
        self.wait(at, PORT_RESET, 0, TRANSITION_US)?;
        self.delay_minimum(10_000)?;
        value = self.read(at)?;
        if value & CONNECT == 0
            || value & (OVERCURRENT | OWNER | PORT_RESET | (1 << 1) | (1 << 5)) != 0
        {
            return Err(Error::Stale);
        }
        // PE is only set by a successful high-speed reset/chirp; never set it.
        if value & ENABLE == 0 {
            if self.companions {
                self.write(at, (value & !PORT_CHANGES) | OWNER)?;
                self.wait(at, OWNER, OWNER, TRANSITION_US)?;
            }
            return Err(Error::Unsupported("EHCI FS/LS companion traffic"));
        }
        self.write(at, (value & !PORT_CHANGES) | (value & PORT_CHANGES))?;
        self.health()?;
        Ok(RootReset::Ready)
    }

    /// Observe kept roots. Disconnect/change invalidates old IDs
    /// before any arena access. An active affected TD permanently quarantines
    /// the controller; an idle disconnect releases only its address0 owner.
    pub fn refresh_roots(&mut self) -> Result<Vec<u8>, Error> {
        self.ready()?;
        let mut removed = Vec::new();
        removed
            .try_reserve_exact(self.roots)
            .map_err(|_| Error::Exhausted)?;
        for index in 0..self.roots {
            if self.ports[index].is_none() {
                continue;
            }
            let result = self.read(self.port_at(index));
            let value = self.guard(result)?;
            let suspended = self.ports[index].as_ref().expect("retained").suspended;
            if value
                & (CONNECT
                    | ENABLE
                    | OWNER
                    | OVERCURRENT
                    | PORT_RESET
                    | PORT_CHANGES
                    | PORT_SUSPEND
                    | PORT_RESUME)
                != CONNECT | ENABLE | if suspended { PORT_SUSPEND } else { 0 }
            {
                let port = (index + 1) as u8;
                self.retire_observed_root(index, value)?;
                removed.push(port);
                // Leave PORTSC unchanged: replaying an unexpected reset/
                // resume/owner bit is not authorized by observing a change.
                // A later explicit root reset owns its W1C acknowledgments.
            }
        }
        let result = self.health();
        self.guard(result)?;
        Ok(removed)
    }

    /// Select a unique USB address through a real successful SET_ADDRESS TD.
    pub fn set_address(&mut self, id: &DeviceId, address: u8) -> Result<(), Error> {
        let index = self.identity(id)?;
        if address == 0
            || address > 127
            || self.ports[index].as_ref().expect("validated").address != 0
            || self
                .ports
                .iter()
                .flatten()
                .any(|device| device.address == address)
        {
            return Err(Error::Invalid);
        }
        self.control_internal(id, [0, 5, address, 0, 0, 0, 0, 0], &[], 0)?;
        let delay = self.delay_minimum(2_000);
        self.guard(delay)?;
        self.ports[index].as_mut().expect("retained").address = address;
        self.default_port = None;
        Ok(())
    }

    /// Execute a bounded control request. Standard address/configuration and
    /// endpoint-halt mutations use their owning methods, preserving pipe state.
    pub fn control_in(&mut self, id: &DeviceId, setup: [u8; 8]) -> Result<Vec<u8>, Error> {
        self.identity(id)?;
        Self::public_control(setup)?;
        if setup[0] & 0x80 == 0 {
            return Err(Error::Invalid);
        }
        self.control_internal(
            id,
            setup,
            &[],
            usize::from(u16::from_le_bytes([setup[6], setup[7]])),
        )
    }
    pub fn control_out(&mut self, id: &DeviceId, setup: [u8; 8], data: &[u8]) -> Result<(), Error> {
        self.identity(id)?;
        Self::public_control(setup)?;
        if setup[0] & 0x80 != 0
            || usize::from(u16::from_le_bytes([setup[6], setup[7]])) != data.len()
        {
            return Err(Error::Invalid);
        }
        self.control_internal(id, setup, data, 0).map(|_| ())
    }
    fn public_control(setup: [u8; 8]) -> Result<(), Error> {
        let kind = setup[0] & 0x60;
        // This backend owns roots only. Hub-port class requests can reset or
        // power an untracked downstream address0 owner, bypassing the
        // bus-address exclusion. They require a real topology owner first.
        if kind == 0x60 || setup[0] & 0x1f > 2 {
            return Err(Error::Unsupported(
                "EHCI topology-owned or reserved control recipient",
            ));
        }
        if kind == 0 && (setup[0] & 0x80 == 0 || matches!(setup[1], 1 | 3 | 5 | 9 | 11)) {
            return Err(Error::Unsupported("EHCI state-changing standard request"));
        }
        Ok(())
    }
}
impl Drop for Controller {
    fn drop(&mut self) {
        // No Drop can invent a hardware ACK or drain. Preserve PCI/UC roots,
        // domain, CPU aliases and pages together for any running/failed owner.
        if self.state != State::Stopped {
            if let Some(arena) = self.arena.take() {
                mem::forget(arena);
            }
            if let Some(hardware) = self.hardware.take() {
                mem::forget(hardware);
            }
        }
    }
}
