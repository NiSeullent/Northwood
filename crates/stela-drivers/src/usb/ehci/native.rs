//! Native ownership, never a BAR address interpreted as a CPU pointer.
use super::*;
use crate::{
    Clock,
    bus::{
        dma::{mapping::DmaDomain, memory::PageSource},
        mmio::owned::{Mapping, Protection},
        pci::{
            ConfigSpace, Function, owned::OwnedConfig, power::Timer,
            requester::preparation::PreparingRequester,
        },
    },
};
use alloc::sync::Arc;
mod requester;
mod time;
use requester::Requester;

/// Original requester/root/BAR owners and the RAM allocator. Construction
/// grants no engine stop, root drain, private domain or DMA allocation.
pub struct NativeResources {
    pub requester: PreparingRequester,
    pub registers: Mapping,
    pub memory: &'static dyn PageSource,
    pub clock: Clock,
    pub minimum: Rc<dyn crate::bus::pci::power::Timer>,
}

/// Attach a native EHCI requester. Initialization errors keep the
/// entire owner in `Failure`; dropping it permanently quarantines resources.
///
/// # Safety
/// These are the paired requester/configuration/BAR resources, with
/// exclusive D0, reset, removal, bridge-routing and assignment ownership for
/// their complete lifetime. The BAR is this function's real BAR0;
/// numeric geometry alone is insufficient. No other firmware/driver accessor
/// may use this controller. BME is initially clear; the requester keeps its
/// original root and DMA backing without a private-domain grant. Initialization
/// stops both vendor engines, drains that original path and consumes the
/// current-epoch receipt before private binding or allocating a fresh arena.
/// If assignment changed, every older route must already be retired by its
/// original owner; a new root cannot certify a different old path.
/// Companion controllers and their port/reset ownership must be stopped and
/// exclusively coordinated before CONFIGFLAG routes roots to EHCI. The clock
/// must be calibrated, monotonic and advancing in this context. All operations
/// run in one serialized process/deferred context, including final release.
/// The platform keeps masked IRQ-route ownership (MSI/MSI-X/INTx),
/// acknowledges prior posted IRQ delivery, and stops/drains every prior IRQ,
/// deferred callback and firmware accessor before attachment and until release.
/// USBINTR0 or BMEclear alone cannot establish those IRQ/callback facts.
pub unsafe fn attach(
    resources: NativeResources,
) -> Result<(Controller, Arc<dyn DmaDomain>), Failure> {
    let mut controller = Controller::initialize(Box::new(Native {
        elapsed: time::Elapsed::new(Clock::ticks()),
        resources: Some(Resources {
            requester: Requester::Preparing(resources.requester),
            registers: resources.registers,
            memory: resources.memory,
            clock: resources.clock,
            minimum: resources.minimum,
        }),
    }))?;
    match controller.hw().domain() {
        Ok(domain) => Ok((controller, domain)),
        Err(error) => {
            controller.state = State::Failed;
            Err(Failure {
                error,
                retained: Box::new(controller),
            })
        }
    }
}

pub struct Failure {
    pub error: Error,
    retained: Box<Controller>,
}
impl Failure {
    /// The exact quarantined attachment remains owned; this exposes no route
    /// to clear uncertainty or republish its old device identities.
    pub fn is_quarantined(&self) -> bool {
        self.retained.state == State::Failed
    }

    /// The legacy handoff's result, if initialization reached that step.
    /// Later controller failures do not undo a verified SMI disable.
    pub(crate) fn legacy_released(&self) -> Option<bool> {
        self.retained.legacy_released
    }
}

// Internal trait lets the DMA peer execute the identical production controller.
// Native is the sole production implementation; no public synthetic adapter.
pub(super) trait Hardware {
    fn validate(&mut self) -> Result<(), Error>;
    fn read32(&mut self, offset: usize) -> Result<u32, Error>;
    fn write32(&mut self, offset: usize, value: u32) -> Result<(), Error>;
    fn config_read(&mut self, offset: u16) -> Result<u32, Error>;
    fn config_write(&mut self, offset: u16, value: u32) -> Result<(), Error>;
    fn config_os_owned(&mut self, offset: u16) -> Result<(), Error>;
    fn bytes(&self) -> usize;
    fn now_us(&mut self) -> u64;
    fn pause(&mut self);
    fn delay_us(&mut self, us: u64) -> Result<(), Error>;
    fn allocate(&mut self, limit: u64) -> Result<Arena, Error>;
    fn bus_master(&mut self, enabled: bool) -> Result<(), Error>;
    fn prepare_dma(&mut self) -> Result<(), Error>;
    fn drain(&mut self) -> Result<(), Error>;
    fn detach_requester(&mut self) -> Result<(), Error>;
    fn retire(&mut self) -> Result<(), Error>;
    fn domain(&mut self) -> Result<Arc<dyn DmaDomain>, Error>;
}
struct Resources {
    requester: Requester,
    registers: Mapping,
    memory: &'static dyn PageSource,
    clock: Clock,
    minimum: Rc<dyn crate::bus::pci::power::Timer>,
}
struct Native {
    resources: Option<Resources>,
    elapsed: time::Elapsed,
}
impl Native {
    fn r(&mut self) -> &mut Resources {
        self.resources.as_mut().expect("retained native owner")
    }
    fn registers(&self) -> Result<&Mapping, Error> {
        Ok(&self.resources.as_ref().ok_or(Error::Register)?.registers)
    }
}
impl Hardware for Native {
    fn validate(&mut self) -> Result<(), Error> {
        let r = self.r();
        r.minimum.check().map_err(Error::Minimum)?;
        if r.requester.function()?.class != CLASS {
            return Err(Error::Invalid);
        }
        // The shared native timer checks duration multiplication before effects
        // and uses wrapping elapsed TSC ticks, including counter rollover.
        r.clock
            .check_duration(BIOS_HANDOFF_US)
            .map_err(|_| Error::Clock)?;
        if r.registers.attribute() != Ok(CacheAttribute::Uncached)
            || r.registers.protection() != Ok(Protection::ReadWrite)
        {
            return Err(Error::Register);
        }
        let bar = r
            .requester
            .function()?
            .config()
            .read32(0x10)
            .map_err(Error::Config)?;
        if bar & 1 != 0 || bar & 6 == 2 || bar & 6 == 6 {
            return Err(Error::Unsupported("EHCI BAR format"));
        }
        let upper = if bar & 6 == 4 {
            u64::from(
                r.requester
                    .function()?
                    .config()
                    .read32(0x14)
                    .map_err(Error::Config)?,
            ) << 32
        } else {
            0
        };
        if r.registers
            .physical_span()
            .map_err(|_| Error::Register)?
            .start()
            != (u64::from(bar & !15) | upper)
        {
            return Err(Error::Register);
        }
        let command = r
            .requester
            .function()?
            .config()
            .read16(4)
            .map_err(Error::Config)?;
        if command & 4 != 0 {
            return Err(Error::Unsupported("EHCI initially uncontained requester"));
        }
        r.requester
            .function()?
            .config()
            .write16(4, command | 2 | 0x400)
            .map_err(Error::Config)?; // memory decode + INTx disabled, BME stays clear
        if r.requester
            .function()?
            .config()
            .read16(4)
            .map_err(Error::Config)?
            & 0x406
            != 0x402
        {
            return Err(Error::Register);
        }
        self.registers()?
            .address(0, 4)
            .ok()
            .filter(|pointer| (pointer.as_ptr() as usize).is_multiple_of(4))
            .ok_or(Error::Register)?;
        Ok(())
    }
    fn read32(&mut self, offset: usize) -> Result<u32, Error> {
        self.registers()?
            .read32(offset)
            .map_err(|_| Error::Register)
    }
    fn write32(&mut self, offset: usize, value: u32) -> Result<(), Error> {
        self.registers()?
            .write32(offset, value)
            .map_err(|_| Error::Register)
    }
    fn config_read(&mut self, offset: u16) -> Result<u32, Error> {
        if offset > 252 || !offset.is_multiple_of(4) {
            return Err(Error::Register);
        }
        self.r()
            .requester
            .function()?
            .config()
            .read32(offset)
            .map_err(Error::Config)
    }
    fn config_write(&mut self, offset: u16, value: u32) -> Result<(), Error> {
        if offset > 252 || !offset.is_multiple_of(4) {
            return Err(Error::Register);
        }
        self.r()
            .requester
            .function()?
            .config()
            .write32(offset, value)
            .map_err(Error::Config)?;
        Ok(())
    }
    fn config_os_owned(&mut self, offset: u16) -> Result<(), Error> {
        if !(0x40..=0xfc).contains(&offset) || !offset.is_multiple_of(4) {
            return Err(Error::Register);
        }
        // Narrow semaphore write preserves BIOS ownership and capability bytes.
        self.r()
            .requester
            .function()?
            .config()
            .write8(offset + 3, 1)
            .map_err(Error::Config)?;
        Ok(())
    }
    fn bytes(&self) -> usize {
        self.resources
            .as_ref()
            .and_then(|r| r.registers.bytes().ok())
            .unwrap_or(0)
    }
    fn now_us(&mut self) -> u64 {
        let rate = self.r().clock.ticks_per_microsecond;
        self.elapsed.sample(Clock::ticks(), rate)
    }
    fn pause(&mut self) {
        core::hint::spin_loop();
    }
    fn delay_us(&mut self, us: u64) -> Result<(), Error> {
        self.r().minimum.wait_us(us).map_err(Error::Minimum)
    }
    fn allocate(&mut self, limit: u64) -> Result<Arena, Error> {
        let r = self.r();
        let pages = r
            .memory
            .allocate(schedule::ARENA_BYTES, 4096, CacheAttribute::WriteBack)?;
        let cpu = pages.cpu_mapping(CacheAttribute::WriteBack)?;
        let dma = DmaMapping::map(
            r.requester.domain()?,
            pages,
            Direction::Bidirectional,
            limit,
        )?;
        Arena::new(cpu, dma)
    }
    fn bus_master(&mut self, enabled: bool) -> Result<(), Error> {
        let config = self.r().requester.function()?.config();
        let old = config.read16(4).map_err(Error::Config)?;
        config
            .write16(4, if enabled { old | 6 | 0x400 } else { old & !4 })
            .map_err(Error::Config)?;
        if (config.read16(4).map_err(Error::Config)? & 4 != 0) != enabled {
            return Err(Error::Register);
        }
        Ok(())
    }
    fn prepare_dma(&mut self) -> Result<(), Error> {
        // SAFETY: initialize_inner acknowledged full HALT/ASS0/PSS0 with
        // device IRQs masked and original BAR/config/root owners kept. Its
        // first-attach grant excludes prior firmware/IRQ/work/bypass users.
        unsafe { self.r().requester.bind() }
    }
    fn drain(&mut self) -> Result<(), Error> {
        // SAFETY: the caller observed schedule stop or full HALT. This is the
        // same already bound domain that owns every current EHCI DMA lease.
        unsafe { self.r().requester.domain()?.drain_posted_writes() }.map_err(Error::Dma)
    }
    fn detach_requester(&mut self) -> Result<(), Error> {
        // SAFETY: shutdown observed full HALT/ASS0/PSS0, BMEclear, original
        // payload drain and successful unmap of its arena. The attachment
        // grant excludes IRQ/work/firmware producers through final retirement.
        unsafe { self.r().requester.retire() }
    }
    fn retire(&mut self) -> Result<(), Error> {
        let resources = self.resources.as_mut().ok_or(Error::Quarantined)?;
        if !resources.requester.is_retired() {
            return Err(Error::Quarantined);
        }
        // SAFETY: Controller::shutdown has stopped engines/BME/DMA and
        // completed original payload, IRQ/users and unique attachment/pin
        // retirement. Keep SAME native requester/function/provider graph
        // before either external retirement callback can fail or unwind.
        unsafe { resources.registers.retire() }.map_err(Error::Mapping)?;
        // SAFETY: every register/table accessor is retired; original actual
        // configuration custody remains inside that same Retired requester.
        unsafe { resources.requester.retire_configuration() }
    }
    fn domain(&mut self) -> Result<Arc<dyn DmaDomain>, Error> {
        self.r().requester.domain()
    }
}

impl Controller {
    pub(super) fn initialize(hardware: Box<dyn Hardware>) -> Result<Self, Failure> {
        let mut controller = Self {
            hardware: Some(hardware),
            arena: None,
            owner: Rc::new(()),
            state: State::Failed,
            fault: None,
            legacy_released: None,
            operational: 0,
            ports: Vec::new(),
            roots: 0,
            preempted: None,
            next_generation: 0,
            next_transfer: 0,
            default_port: None,
            power_control: false,
            companions: false,
            pending: None,
        };
        match controller.initialize_inner() {
            Ok(()) => {
                controller.state = State::Running;
                Ok(controller)
            }
            Err(error) => Err(Failure {
                error,
                retained: Box::new(controller),
            }),
        }
    }
    pub(super) fn initialize_inner(&mut self) -> Result<(), Error> {
        self.legacy_released = None;
        self.hw().validate()?;
        let caps = self.read(0)?;
        if caps >> 16 != 0x100 {
            return Err(Error::Unsupported(
                "EHCI revision (only standard 1.0 qualified)",
            ));
        }
        self.operational = (caps & 0xff) as usize;
        if self.operational < 0x10 || !self.operational.is_multiple_of(4) {
            return Err(Error::Invalid);
        }
        let hcs = self.read(4)?;
        let hcc = self.read(8)?;
        let ports = (hcs & 15) as usize;
        if ports == 0 || self.operational + PORT + ports * 4 > self.hw().bytes() {
            return Err(Error::Register);
        }
        self.roots = ports;
        self.ports = (0..ports + CHILDREN).map(|_| None).collect();
        self.power_control = hcs & 16 != 0;
        self.companions = hcs & (15 << 12) != 0;
        self.legacy_released = Some(false);
        self.legacy(((hcc >> 8) & 255) as u16)?;
        self.legacy_released = Some(true);
        self.put(INTR, 0)?;
        let command = self.op(CMD)?;
        if self.op(STS)? & HALT != 0 {
            // A halted controller runs no schedule, whatever ASE, PSE and
            // their statuses still say: the early PCI handoff clears only RUN,
            // as Linux's `quirk_usb_disable_ehci` does, and a halted schedule
            // status need not follow its enable. Linux's `ehci_halt` then
            // waits for HALT alone.
            self.put(CMD, command & !(RUN | ASE | PSE | IAAD))?;
            self.wait(self.operational + STS, HALT, HALT, HALT_US)?;
        } else {
            self.wait(
                self.operational + STS,
                ASS | PSS,
                (if command & ASE != 0 { ASS } else { 0 })
                    | (if command & PSE != 0 { PSS } else { 0 }),
                TRANSITION_US,
            )?;
            self.put(CMD, command & !(RUN | ASE | PSE | IAAD))?;
            self.wait(self.operational + STS, HALT | ASS | PSS, HALT, HALT_US)?;
        }
        // Initial BMEclear/reset alone cannot retire old posted writes before
        // the allocator/domain reuses RAM or IOVAs for this fresh arena.
        self.hw().prepare_dma()?;
        self.put(CMD, RESET)?;
        self.wait(self.operational + CMD, RESET, 0, RESET_US)?;
        if self.op(STS)? & HALT == 0 {
            return Err(Error::Invalid);
        }
        let limit = if hcc & 1 != 0 {
            u64::MAX
        } else {
            u64::from(u32::MAX)
        };
        self.arena = Some(self.hw().allocate(limit)?);
        let address = self.arena.as_ref().expect("retained").address;
        if hcc & 1 != 0 {
            self.put(SEGMENT, (address >> 32) as u32)?;
        }
        self.put(ASYNC, address as u32)?;
        self.put(PERIODIC, (address + schedule::FRAME_LIST as u64) as u32)?;
        self.put(INTR, 0)?;
        self.put(STS, 0x3f)?; // Only EHCI 1.0 W1C interrupt statuses.
        self.hw().bus_master(true)?;
        self.put(CMD, 8 << 16 | RUN)?; // Periodic and async both disabled.
        self.wait(self.operational + STS, HALT, 0, TRANSITION_US)?;
        self.put(CONFIG, 1)?;
        self.wait(self.operational + CONFIG, 1, 1, TRANSITION_US)?;
        self.delay_minimum(5_000)?; // Root-routing settle, not register readback.
        // HCRST is not USB device-address retirement. Before minting any
        // default-address owner, prove every untracked EHCI root isolated.
        for index in 0..self.roots {
            let at = self.port_at(index);
            let value = self.read(at)?;
            if value & OWNER != 0 {
                continue;
            } // Companion route, not ours.
            if value & (PORT_RESET | PORT_RESUME | PORT_SUSPEND) != 0 {
                return Err(Error::Unsupported("EHCI inherited root transition"));
            }
            if value & ENABLE != 0 {
                self.write(at, value & !(ENABLE | PORT_CHANGES))?;
                self.wait(at, ENABLE, 0, TRANSITION_US)?;
            }
        }
        self.health()
    }
    fn legacy(&mut self, mut offset: u16) -> Result<(), Error> {
        let mut seen = [false; 64];
        while offset != 0 {
            if !(0x40..=0xf8).contains(&offset)
                || !offset.is_multiple_of(4)
                || seen[usize::from(offset / 4)]
            {
                return Err(Error::Invalid);
            }
            seen[usize::from(offset / 4)] = true;
            let cap = self.hw().config_read(offset)?;
            if cap == u32::MAX {
                return Err(Error::Removed);
            }
            if cap & 255 == 1 {
                self.hw().config_os_owned(offset)?;
                let start = self.hw().now_us();
                loop {
                    let ownership = self.hw().config_read(offset)?;
                    if ownership == u32::MAX {
                        return Err(Error::Removed);
                    }
                    if ownership & (1 << 16) == 0 && ownership & (1 << 24) != 0 {
                        break;
                    }
                    if self.hw().now_us().wrapping_sub(start) >= BIOS_HANDOFF_US {
                        return Err(Error::Timeout);
                    }
                    self.hw().pause();
                }
                // EHCI USBLEGCTLSTS SMI enables0..5,13..15; high RW1C
                // statuses29..31. Do not replay reserved/RO lower bits.
                self.hw().config_write(offset + 4, 0xe000_0000)?;
                if self.hw().config_read(offset + 4)? & crate::usb::handoff::EHCI_LEGACY_SMI_ENABLES
                    != 0
                {
                    return Err(Error::Register);
                }
            }
            offset = ((cap >> 8) & 255) as u16;
        }
        Ok(())
    }

    /// Explicit acknowledged shutdown. Failed shutdown preserves the complete
    /// attachment indefinitely. A stopped owner cannot be restarted/rebound.
    pub fn shutdown(&mut self) -> Result<(), Error> {
        self.ready()?;
        if self.pending.is_some() {
            return Err(Error::Busy);
        }
        self.state = State::Retiring;
        self.schedule_off()?;
        self.put(CMD, 0)?;
        self.wait(self.operational + STS, HALT | ASS | PSS, HALT, HALT_US)?;
        self.hw().bus_master(false)?;
        self.hw().drain()?;
        if let Some(arena) = self.arena.as_mut() {
            // SAFETY: engines halted, BMEclear, drain, serialized no IRQ.
            unsafe {
                arena.dma.unmap()?;
            }
        }
        self.arena = None;
        let detached = self.hw().detach_requester();
        self.guard(detached)?;
        let retired = self.hw().retire();
        self.guard(retired)?;
        self.ports.iter_mut().for_each(|port| *port = None);
        self.state = State::Stopped;
        Ok(())
    }
}
