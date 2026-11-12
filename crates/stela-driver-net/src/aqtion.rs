// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2014-2019 aQuantia Corporation
// Copyright (C) 2019-2020 Marvell International Ltd.
// Copyright (C) 2026 Stela contributors
//! AQtion PCIe Ethernet. Linux atlantic at a804c5742bff224d4cb4d25dd383554d598be32b.
//! One polled queue, full Ethernet frames, no checksum/VLAN/segmentation offloads.
//! Firmware must be provisioned in the controller's NVM; host firmware download
//! is explicitly unsupported, as in the pinned donor. See NOTICE.
use super::{HEADER, Link, LinkError, MAX_FRAME, MIN_FRAME, Mac, check_frame};
use crate::bus::{
    dma::{
        mapping::{DmaDomain, DmaError},
        memory::MemoryError,
    },
    mmio::owned,
    pci::{ConfigSpace, Function, owned::OwnedConfig},
};
use alloc::{format, sync::Arc, vec::Vec};
use core::cell::Cell;
mod firmware;
mod hardware;
mod memory;
pub mod native;
mod registers;
use memory::{COUNT, Memory};
use registers::Field;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Config(crate::bus::pci::ConfigError),
    PciCapability(crate::bus::pci::CapabilityError),
    Pci(crate::block::BlockError),
    Unsupported(&'static str),
    Timeout { register: usize },
    Removed,
    Firmware { status: u32 },
    Dma(DmaError),
    Memory(MemoryError),
    Mapping(owned::Error),
    Readback { register: usize, observed: u32 },
    RingHead(u32),
    Stopped,
}
pub type Result<T> = core::result::Result<T, Error>;
impl From<DmaError> for Error {
    fn from(e: DmaError) -> Self {
        Self::Dma(e)
    }
}
impl From<MemoryError> for Error {
    fn from(e: MemoryError) -> Self {
        Self::Memory(e)
    }
}
impl From<owned::Error> for Error {
    fn from(e: owned::Error) -> Self {
        Self::Mapping(e)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Generation {
    Atlantic,
    Antigua,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    pub generation: Generation,
    pub device: u16,
    pub rates: u32,
}
impl Profile {
    pub fn identify(vendor: u16, device: u16) -> Option<Self> {
        if vendor != 0x1d6a {
            return None;
        }
        let (generation, rates) = match device {
            0x07b1 | 0x87b1 => (Generation::Atlantic, 0xf20),
            0x08b1 | 0x88b1 | 0x11b1 | 0x91b1 => (Generation::Atlantic, 0x720),
            0x12b1 | 0x92b1 => (Generation::Atlantic, 0x320),
            0x00c0 | 0x94c0 | 0x93c0 | 0x04c0 | 0x14c0 | 0x34c0 => (Generation::Antigua, 0xff00),
            _ => return None,
        };
        Some(Self {
            generation,
            device,
            rates,
        })
    }
}

/// Exclusive register effects, also used by the independently authored model.
/// # Safety
/// This is the UC BAR of this same function and reset epoch.
/// A read/write performs the native DWORD operation or returns failure.
/// The owner must preserve mappings/reservations if dropped during active DMA.
pub unsafe trait Registers {
    fn read(&self, offset: usize) -> Result<u32>;
    fn write(&self, offset: usize, value: u32) -> Result<()>;
}
/// Bounded real UC mapping; no raw numeric address can create this owner.
pub struct MappedRegisters {
    mapping: owned::Mapping,
}
impl MappedRegisters {
    pub fn new(mapping: owned::Mapping) -> core::result::Result<Self, (Error, owned::Mapping)> {
        let valid = (|| {
            if mapping.attribute()? != owned::CacheAttribute::Uncached
                || mapping.protection()? != owned::Protection::ReadWrite
                || mapping.bytes()? < 0x10000
            {
                return Err(Error::Unsupported(
                    "AQtion requires a complete UC read/write register BAR",
                ));
            }
            Ok(())
        })();
        match valid {
            Ok(()) => Ok(Self { mapping }),
            Err(e) => Err((e, mapping)),
        }
    }
}
// SAFETY: the owned mapping checks bounds/cache policy on every exact-width access.
unsafe impl Registers for MappedRegisters {
    fn read(&self, offset: usize) -> Result<u32> {
        if !offset.is_multiple_of(4) {
            return Err(Error::Unsupported("unaligned AQtion register"));
        }
        let p = self.mapping.address(offset, 4)?;
        // SAFETY: checked live UC DWORD, aligned by mapping and offset.
        Ok(unsafe { p.as_ptr().cast::<u32>().read_volatile() })
    }
    fn write(&self, offset: usize, value: u32) -> Result<()> {
        if !offset.is_multiple_of(4) {
            return Err(Error::Unsupported("unaligned AQtion register"));
        }
        let p = self.mapping.address(offset, 4)?;
        // SAFETY: checked live UC DWORD.
        unsafe { p.as_ptr().cast::<u32>().write_volatile(value) };
        Ok(())
    }
}
pub use crate::bus::pci::power::Timer;
struct Hardware<R, T> {
    registers: R,
    timer: T,
    reset_ownership: Cell<firmware::ResetOwnership>,
}
impl<R: Registers, T: Timer> Hardware<R, T> {
    fn check_timing(&self) -> Result<()> {
        self.timer
            .check()
            .map_err(|error| Error::Pci(crate::block::BlockError::PciPower(error)))
    }
    fn delay(&self, microseconds: u64) -> Result<()> {
        self.timer
            .wait_us(microseconds)
            .map_err(|error| Error::Pci(crate::block::BlockError::PciPower(error)))
    }
    fn read(&self, offset: usize) -> Result<u32> {
        let value = self.registers.read(offset)?;
        // Linux aq_hw_read_reg permits all-ones data unless the independent
        // alive register also reports removal. Both declared hardware caps
        // select0x10; firmware RAM and MAC padding may legitimately be all ones.
        if value == u32::MAX && self.registers.read(0x10)? == u32::MAX {
            return Err(Error::Removed);
        }
        Ok(value)
    }
    fn write(&self, offset: usize, value: u32) -> Result<()> {
        self.registers.write(offset, value)
    }
    fn set(&self, field: Field, index: usize, value: u32) -> Result<()> {
        let address = field.address + field.stride * index;
        let word = if field.mask == u32::MAX {
            value
        } else {
            (self.read(address)? & !field.mask) | ((value << field.shift) & field.mask)
        };
        self.write(address, word)
    }
    fn get(&self, field: Field, index: usize) -> Result<u32> {
        Ok((self.read(field.address + field.stride * index)? & field.mask) >> field.shift)
    }
    fn wait(
        &self,
        register: usize,
        interval: u64,
        timeout: u64,
        mut ready: impl FnMut(u32) -> bool,
    ) -> Result<u32> {
        for _ in 0..=timeout / interval {
            let word = self.read(register)?;
            if ready(word) {
                return Ok(word);
            }
            self.delay(interval)?;
        }
        Err(Error::Timeout { register })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Stopped,
    Running,
    Suspended,
    Faulted,
}
// Independent of queue health: a failed PMCSR transaction may have reached
// the device even when readback fails. Only native power_up can restore D0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PciPower {
    Unknown,
    D0,
    D3hot,
}
fn power_up<C: ConfigSpace>(
    function: &mut Function<C>,
    timer: &impl Timer,
    power: &mut PciPower,
) -> Result<()> {
    timer
        .check()
        .map_err(|error| Error::Pci(crate::block::BlockError::PciPower(error)))?;
    *power = PciPower::Unknown;
    if function.config().read16(4)? == u16::MAX || function.config().read16(6)? == u16::MAX {
        return Err(Error::Removed);
    }
    function.power_up(timer).map_err(Error::Pci)?;
    *power = PciPower::D0;
    Ok(())
}
fn power_down<C: ConfigSpace>(function: &mut Function<C>, power: &mut PciPower) -> Result<()> {
    let pm = function
        .capabilities()?
        .into_iter()
        .find_map(|(offset, id)| (id == 1).then_some(u16::from(offset)))
        .ok_or(Error::Unsupported("AQtion PCI power capability absent"))?;
    let pmcsr = pm + 4;
    let control = function.config().read16(pmcsr)?;
    if control == u16::MAX {
        return Err(Error::Removed);
    }
    *power = PciPower::Unknown;
    // Native WORD store: leave PME_Status W1C pending and disable PME.
    function.config().write16(pmcsr, (control & !0x8103) | 3)?;
    let observed = function.config().read16(pmcsr)?;
    if observed == u16::MAX {
        return Err(Error::Removed);
    }
    if observed & 3 != 3 {
        return Err(Error::Readback {
            register: usize::from(pmcsr),
            observed: u32::from(observed),
        });
    }
    *power = PciPower::D3hot;
    Ok(())
}
/// Owns the PCI configuration, UC register window, DMA domain and pins.
/// Drop quarantines every active physical owner; it never invents DMA completion.
/// Native attachment uses the cold/root owner in [`native`]. The defaulted configuration
/// parameter lets protocol peers exercise the same lifecycle implementation;
/// it supplies no constructor or physical resource permission.
pub struct Aqtion<
    R: Registers = MappedRegisters,
    T: Timer = alloc::rc::Rc<dyn Timer>,
    C: ConfigSpace = OwnedConfig,
> {
    function: Option<Function<C>>,
    hardware: Option<Hardware<R, T>>,
    domain: Option<Arc<dyn DmaDomain>>,
    memory: Option<Memory>,
    profile: Profile,
    firmware: Option<firmware::Firmware>,
    mac: Mac,
    state: State,
    power: PciPower,
    tx_tail: usize,
    tx_head: usize,
    rx_head: usize,
    dropping: bool,
}
impl<R: Registers, T: Timer, C: ConfigSpace> Aqtion<R, T, C> {
    fn hw(&self) -> &Hardware<R, T> {
        self.hardware.as_ref().unwrap()
    }
    pub fn state(&self) -> State {
        self.state
    }
    pub fn profile(&self) -> Profile {
        self.profile
    }
    pub fn firmware_version(&self) -> Option<u32> {
        self.firmware.map(|f| f.version)
    }
    fn resume_owned(&mut self) -> Result<()> {
        self.require_rebuildable()?;
        if self.power == PciPower::D0 {
            return Err(Error::Unsupported("AQtion is already in D0"));
        }
        power_up(
            self.function.as_mut().unwrap(),
            &self.hardware.as_ref().unwrap().timer,
            &mut self.power,
        )?;
        if self.state == State::Suspended {
            self.state = State::Stopped;
        }
        self.start()
    }
    fn require_d0(&self) -> Result<()> {
        if self.power != PciPower::D0 {
            return Err(Error::Unsupported("AQtion requires acknowledged PCI D0"));
        }
        Ok(())
    }
    fn require_rebuildable(&self) -> Result<()> {
        if self.memory.as_ref().is_none_or(Memory::is_retiring) {
            return Err(Error::Stopped);
        }
        Ok(())
    }
    fn start(&mut self) -> Result<()> {
        self.require_rebuildable()?;
        self.hw().check_timing()?;
        self.require_d0()?;
        if self.state == State::Running {
            return Err(Error::Unsupported("AQtion already running"));
        }
        self.state = State::Faulted;
        let function = self.function.as_mut().unwrap();
        if function.config().read16(4)? == u16::MAX {
            return Err(Error::Removed);
        }
        // No MSI/MSI-X registration exists: stop all device sources and use polling.
        let command = function.config().read16(4)?;
        function
            .config()
            .write16(4, (command & !4) | 2 | (1 << 10))?;
        self.hw().mask_interrupts()?;
        self.hw().reset(self.profile)?;
        self.hw().stop_queues()?;
        let fw = self.hw().firmware(self.profile)?;
        self.start_queues(fw)
    }
    /// Publish fresh queues after firmware reset and queue stop. Both entry
    /// paths keep acknowledged D0 and their checked clock/root through this call.
    fn start_queues(&mut self, fw: firmware::Firmware) -> Result<()> {
        self.mac = fw.mac;
        self.firmware = Some(fw);
        self.hw().configure(self.profile, &fw)?;
        self.tx_tail = 0;
        self.tx_head = 0;
        self.rx_head = 0;
        self.dropping = false;
        self.memory.as_mut().unwrap().prepare()?;
        self.hw().queues(self.memory.as_ref().unwrap())?;
        self.function.as_mut().unwrap().enable_polled()?;
        let command = self.function.as_mut().unwrap().config().read16(4)?;
        if command == u16::MAX {
            return Err(Error::Removed);
        }
        if command & 0x406 != 0x406 {
            return Err(Error::Readback {
                register: 4,
                observed: u32::from(command),
            });
        }
        self.hw().enable_queues()?;
        self.state = State::Running;
        Ok(())
    }
    /// Ordinary stop masks IRQs, disables engines, acknowledges their state,
    /// clears BusMaster and drains the real remapper's posted payload writes.
    /// A failure keeps every owner and requires recovery before I/O resumes.
    pub fn stop(&mut self) -> Result<()> {
        if matches!(self.state, State::Stopped | State::Suspended) {
            return Ok(());
        }
        self.hw().check_timing()?;
        self.require_d0()?;
        self.state = State::Faulted;
        self.hw().stop_queues()?;
        if let Some(fw) = self.firmware {
            self.hw().firmware_link(self.profile, fw, false)?;
        }
        let f = self.function.as_mut().unwrap();
        let command = f.config().read16(4)?;
        if command == u16::MAX {
            return Err(Error::Removed);
        }
        f.config().write16(4, command & !4)?;
        let command = f.config().read16(4)?;
        if command == u16::MAX {
            return Err(Error::Removed);
        }
        if command & 4 != 0 {
            return Err(Error::Readback {
                register: 4,
                observed: u32::from(command),
            });
        }
        // SAFETY: engine-disable readbacks and stopped submissions precede this
        // domain write-drain. There are no registered IRQ/work callbacks.
        unsafe { self.domain.as_ref().unwrap().drain_posted_writes() }?;
        self.state = State::Stopped;
        Ok(())
    }
    /// Stop/drain the real queue epoch before requesting PCI D3hot. The
    /// configuration owner and DMA domain remain held for ordinary resume.
    pub fn suspend(&mut self) -> Result<()> {
        self.stop()?;
        if self.state == State::Suspended {
            return Ok(());
        }
        self.require_d0()?;
        // stop() already proved this queue epoch drained. Keep Stopped on an
        // ambiguous D3 readback; retire can reclaim it without powered MMIO.
        power_down(self.function.as_mut().unwrap(), &mut self.power)?;
        self.state = State::Suspended;
        Ok(())
    }
    pub fn restart(&mut self) -> Result<()> {
        self.stop()?;
        self.start()
    }
    /// Explicitly reclaim DMA only after the acknowledged ordinary stop/drain.
    /// A failed unmap keeps the complete owner for retirement retry. Starting
    /// retirement permanently revokes queue rebuild, including on first failure.
    pub fn retire(&mut self) -> Result<()> {
        self.stop()?;
        if let Some(m) = self.memory.as_mut() {
            m.retire()?;
        }
        self.memory = None;
        Ok(())
    }
    fn healthy(&mut self) -> Result<bool> {
        if self.state != State::Running {
            return Err(Error::Stopped);
        }
        let result = self.hw().link(self.profile, self.firmware.unwrap());
        if result.is_err() {
            self.state = State::Faulted;
        } else if result == Ok(false) && self.tx_head != self.tx_tail {
            // Link loss can strand hardware TX descriptors. Retire the
            // engine epoch before reusing their bytes; failures keep it held.
            self.restart()?;
        }
        result
    }
    fn tx_reap(&mut self) -> Result<()> {
        let head = self.hw().get(registers::TX_HEAD, 0)?;
        if head as usize >= COUNT {
            return Err(Error::RingHead(head));
        }
        // Hardware head must advance only through submitted entries.
        let completed = (head as usize + COUNT - self.tx_head) % COUNT;
        let pending = (self.tx_tail + COUNT - self.tx_head) % COUNT;
        if completed > pending {
            return Err(Error::RingHead(head));
        }
        self.memory
            .as_ref()
            .unwrap()
            .complete_tx(self.tx_head, completed)?;
        self.tx_head = head as usize;
        Ok(())
    }
    fn send(&mut self, frame: &[u8]) -> Result<()> {
        self.tx_reap()?;
        let next = (self.tx_tail + 1) % COUNT;
        for _ in 0..2000 {
            if next != self.tx_head {
                break;
            }
            self.hw().delay(100)?;
            self.tx_reap()?;
        }
        if next == self.tx_head {
            return Err(Error::Timeout {
                register: registers::TX_HEAD.address,
            });
        }
        self.memory
            .as_mut()
            .unwrap()
            .transmit(self.tx_tail, frame)?;
        self.hw().set(registers::TX_TAIL, 0, next as u32)?;
        self.tx_tail = next;
        Ok(())
    }
    /// Copy the next good frame into `frame`; true when one arrived.
    fn poll_rx(&mut self, frame: &mut Vec<u8>) -> Result<bool> {
        for _ in 0..COUNT {
            let Some((kind, status, length)) =
                self.memory.as_ref().unwrap().receive_status(self.rx_head)?
            else {
                return Ok(false);
            };
            let end = status & 2 != 0;
            let good = !self.dropping
                && end
                && status & 4 == 0
                && kind & 0x1000 == 0
                && status & 0xf000 == 0
                && (HEADER..=MAX_FRAME).contains(&length);
            if good {
                self.memory
                    .as_ref()
                    .unwrap()
                    .receive_bytes(self.rx_head, length, frame)?;
            }
            self.dropping = !end;
            let index = self.rx_head;
            self.memory.as_mut().unwrap().offer(index)?;
            self.rx_head = (index + 1) % COUNT;
            // Tail names the first unoffered descriptor; one slot remains reserved.
            self.hw().set(registers::RX_TAIL, 0, index as u32)?;
            if good {
                return Ok(true);
            }
        }
        Ok(false)
    }
}
impl<R: Registers, T: Timer, C: ConfigSpace> Link for Aqtion<R, T, C> {
    fn mac(&self) -> Mac {
        self.mac
    }
    fn carrier(&mut self) -> bool {
        self.healthy().unwrap_or(false)
    }
    fn transmit(&mut self, frame: &[u8]) -> core::result::Result<(), LinkError> {
        check_frame(frame)?;
        if !self.healthy().map_err(adapter)? {
            return Err(LinkError::NoCarrier);
        }
        self.send(frame)
            .inspect_err(|_| self.state = State::Faulted)
            .map_err(adapter)
    }
    fn receive(&mut self) -> core::result::Result<Option<Vec<u8>>, LinkError> {
        let mut frame = Vec::new();
        Ok(self.receive_reusing(&mut frame)?.then_some(frame))
    }
    /// The stack's receive path: the frame is copied into its reused
    /// buffer, so a received frame allocates nothing.
    fn receive_reusing(&mut self, frame: &mut Vec<u8>) -> core::result::Result<bool, LinkError> {
        self.healthy().map_err(adapter)?;
        self.poll_rx(frame)
            .inspect_err(|_| self.state = State::Faulted)
            .map_err(adapter)
    }
}
fn adapter(error: Error) -> LinkError {
    match error {
        Error::Config(error) => LinkError::Pci(crate::block::BlockError::Config(error)),
        Error::PciCapability(error) => {
            LinkError::Pci(crate::block::BlockError::PciCapability(error))
        }
        Error::Pci(error) => LinkError::Pci(error),
        error => LinkError::Adapter(format!("AQtion: {error:?}")),
    }
}
impl<R: Registers, T: Timer, C: ConfigSpace> Drop for Aqtion<R, T, C> {
    fn drop(&mut self) {
        // Even stopped owners keep their MMIO epoch until the platform retires
        // its mapping explicitly. An unretired DMA mapping quarantines its pins.
        if self.state != State::Stopped || self.power != PciPower::D0 {
            if let Some(f) = self.function.take() {
                core::mem::forget(f);
            }
            if let Some(h) = self.hardware.take() {
                core::mem::forget(h);
            }
            if let Some(d) = self.domain.take() {
                core::mem::forget(d);
            }
        }
    }
}

#[cfg(test)]
mod tests;

impl From<crate::bus::pci::ConfigError> for Error {
    fn from(e: crate::bus::pci::ConfigError) -> Self {
        Self::Config(e)
    }
}
impl From<crate::bus::pci::CapabilityError> for Error {
    fn from(e: crate::bus::pci::CapabilityError) -> Self {
        Self::PciCapability(e)
    }
}
impl From<crate::block::BlockError> for Error {
    fn from(e: crate::block::BlockError) -> Self {
        Self::Pci(e)
    }
}

#[cfg(test)]
use crate::Clock;
