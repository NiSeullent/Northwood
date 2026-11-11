// SPDX-License-Identifier: GPL-2.0-only
// Intel LPSS private initialization adapted from Linux intel-lpss.c and
// intel-lpss-pci.c at a804c5742bff224d4cb4d25dd383554d598be32b.
// Copyright (C) 2015 Intel Corporation.
// Authors: Andy Shevchenko, Mika Westerberg, Heikki Krogerus, Jarkko Nikula.
//! Owned PCI LPSS I²C platform resources. [`spi`] supplies the separate kept
//! SPI reset/remap/clock and SSP PIO composition. UART/iDMA and platform PM
//! remain separate. The explicit I²C owner composes private resource
//! initialization with the shared DesignWare PIO engine and held-D0 lifetime.
use super::{
    mmio::owned::{CacheAttribute, Mapping, Protection},
    pci::{ConfigSpace, Function, owned::OwnedConfig, requester::StoppedRequester},
};
use crate::bus::pci::regs;
use core::mem::ManuallyDrop;

mod child;
pub mod cold;
pub mod spi;
pub use child::{AttachError, AttachFailure, AttachStage, I2c, I2cFailure, Retired};

const PRIVATE: usize = 0x200;
const WINDOW_BYTES: usize = 0x300;
const RESET: usize = PRIVATE + 0x04;
const REMAP_LOW: usize = PRIVATE + 0x40;
const REMAP_HIGH: usize = REMAP_LOW + 4;
const CAPS: usize = PRIVATE + 0xfc;
const CAPS_TYPE: u32 = 0xf << 4;
const RESET_FUNCTION_AND_IDMA: u32 = 7;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Claim(crate::bus::pci::requester::ClaimError),
    Config(crate::bus::pci::ConfigError),
    Unsupported,
    Identity,
    Mapping,
    Mmio(super::mmio::owned::Error),
    Configuration,
    Register {
        offset: usize,
        expected: u32,
        observed: u32,
    },
    Io,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Identity {
    pub vendor: u16,
    pub device: u16,
    pub subsystem_vendor: u16,
    pub subsystem_device: u16,
}
/// Source clock/property group. Names follow the donor's info
/// structures; later chips can select an earlier group's properties.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProfileKind {
    Sunrisepoint,
    CannonLake,
    Broxton,
    ApolloLake,
    GeminiLake,
    ElkhartLake,
}
/// One source-selected I²C profile. These properties are not board SCL
/// counts or permission to ignore an overlapping firmware resource reservation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Profile {
    device: u16,
    kind: ProfileKind,
}
impl Profile {
    pub fn select(identity: Identity) -> Result<Self, Error> {
        if identity.vendor != 0x8086 {
            return Err(Error::Unsupported);
        }
        let kind = match identity.device {
            0x02c5 | 0x02c6 | 0x02e8 | 0x02e9 | 0x02ea | 0x02eb | 0x06e8 | 0x06e9 | 0x06ea
            | 0x06eb | 0x9dc5 | 0x9dc6 | 0x9de8 | 0x9de9 | 0x9dea | 0x9deb | 0xa368 | 0xa369
            | 0xa36a | 0xa36b => ProfileKind::CannonLake,
            0x0aac | 0x0aae | 0x0ab0 | 0x0ab2 | 0x0ab4 | 0x0ab6 | 0x0ab8 | 0x0aba | 0x1aac
            | 0x1aae | 0x1ab0 | 0x1ab2 | 0x1ab4 | 0x1ab6 | 0x1ab8 | 0x1aba | 0x34c5 | 0x34c6
            | 0x34e8 | 0x34e9 | 0x34ea | 0x34eb | 0x43ad | 0x43ae | 0x43d8 | 0x43e8 | 0x43e9
            | 0x43ea | 0x43eb | 0x4dc5 | 0x4dc6 | 0x4de8 | 0x4de9 | 0x4dea | 0x4deb | 0x51c5
            | 0x51c6 | 0x51d8 | 0x51d9 | 0x51e8 | 0x51e9 | 0x51ea | 0x51eb | 0x54c5 | 0x54c6
            | 0x54e8 | 0x54e9 | 0x54ea | 0x54eb | 0x7750 | 0x7751 | 0x7778 | 0x7779 | 0x777a
            | 0x777b | 0x7a4c | 0x7a4d | 0x7a4e | 0x7a4f | 0x7a7c | 0x7a7d | 0x7acc | 0x7acd
            | 0x7ace | 0x7acf | 0x7afc | 0x7afd | 0x7e50 | 0x7e51 | 0x7e78 | 0x7e79 | 0x7e7a
            | 0x7e7b | 0x7f4c | 0x7f4d | 0x7f4e | 0x7f4f | 0x7f7a | 0x7f7b | 0x98c5 | 0x98c6
            | 0x98e8 | 0x98e9 | 0x98ea | 0x98eb => ProfileKind::Broxton,
            0x31ac | 0x31ae | 0x31b0 | 0x31b2 | 0x31b4 | 0x31b6 | 0x31b8 | 0x31ba => {
                ProfileKind::GeminiLake
            }
            0x4b44 | 0x4b45 | 0x4b4b | 0x4b4c | 0x4b78 | 0x4b79 | 0x4b7a | 0x4b7b | 0x4d50
            | 0x4d51 | 0x4d78 | 0x4d79 | 0x4d7a | 0x4d7b | 0x6e4c | 0x6e4d | 0x6e4e | 0x6e4f
            | 0x6e7a | 0x6e7b | 0xa850 | 0xa851 | 0xa878 | 0xa879 | 0xa87a | 0xa87b | 0xd350
            | 0xd351 | 0xd378 | 0xd379 | 0xd37a | 0xd37b | 0xe350 | 0xe351 | 0xe378 | 0xe379
            | 0xe37a | 0xe37b | 0xe450 | 0xe451 | 0xe478 | 0xe479 | 0xe47a | 0xe47b => {
                ProfileKind::ElkhartLake
            }
            0x5aac | 0x5aae | 0x5ab0 | 0x5ab2 | 0x5ab4 | 0x5ab6 | 0x5ab8 | 0x5aba => {
                ProfileKind::ApolloLake
            }
            0x9d60 | 0x9d61 | 0x9d62 | 0x9d63 | 0x9d64 | 0x9d65 | 0xa0c5 | 0xa0c6 | 0xa0d8
            | 0xa0d9 | 0xa0e8 | 0xa0e9 | 0xa0ea | 0xa0eb | 0xa160 | 0xa161 | 0xa162 | 0xa2e0
            | 0xa2e1 | 0xa2e2 | 0xa2e3 | 0xa3e0 | 0xa3e1 | 0xa3e2 | 0xa3e3 => {
                ProfileKind::Sunrisepoint
            }
            _ => return Err(Error::Unsupported),
        };
        Ok(Self {
            device: identity.device,
            kind,
        })
    }
    pub fn device(self) -> u16 {
        self.device
    }
    pub fn input_clock_hz(self) -> u32 {
        match self.kind {
            ProfileKind::Sunrisepoint => 120_000_000,
            ProfileKind::CannonLake => 216_000_000,
            ProfileKind::Broxton | ProfileKind::ApolloLake | ProfileKind::GeminiLake => 133_000_000,
            ProfileKind::ElkhartLake => 100_000_000,
        }
    }
    pub fn source_sda_hold_ns(self) -> u32 {
        match self.kind {
            ProfileKind::Sunrisepoint | ProfileKind::CannonLake => 230,
            ProfileKind::Broxton | ProfileKind::ElkhartLake => 42,
            ProfileKind::ApolloLake => 207,
            ProfileKind::GeminiLake => 313,
        }
    }
    pub fn source_sda_fall_ns(self) -> Option<u32> {
        match self.kind {
            ProfileKind::Sunrisepoint | ProfileKind::CannonLake => None,
            _ => Some(171),
        }
    }
    pub fn source_scl_fall_ns(self) -> Option<u32> {
        match self.kind {
            ProfileKind::Sunrisepoint | ProfileKind::CannonLake => None,
            ProfileKind::GeminiLake => Some(290),
            _ => Some(208),
        }
    }
}
/// Function/root/reset and UC register permission.
///
/// # Safety
/// The provider owns the same powered D0 physical function, complete
/// BAR and configuration/reset epoch. Firmware, AML/SMM, shared PUNIT/semaphore users, IRQs and other CPUs
/// cannot access/revoke it. Prior engines, posted DMA, callbacks and work are
/// stopped/drained; bus mastering and interrupts remain disabled. All addresses
/// and identity are observations from those resources. Leaking this
/// provider keeps its grants; no borrowed temporary owner is allowed.
pub unsafe trait Registers: 'static {
    fn identity(&mut self) -> Result<Identity, Error>;
    fn physical_base(&self) -> u64;
    fn read32(&mut self, offset: usize) -> Result<u32, Error>;
    fn write32(&mut self, offset: usize, value: u32) -> Result<(), Error>;
}

/// Fallible child MMIO under the same retained mapping.
/// # Safety
/// The full 0x200-byte child aperture has stable ordered native DWORD access.
/// Offsets are the DesignWare register definitions. Hardware fault values
/// remain observable; no invented success or temporary pointer is permitted.
/// The same provider keeps power/clock/root grants if quarantined.
pub unsafe trait ChildAccess: Registers {
    fn child_read32(&mut self, offset: usize) -> Result<u32, Error>;
    fn child_write32(&mut self, offset: usize, value: u32) -> Result<(), Error>;
}

pub type NativeResources = (Function<OwnedConfig>, Mapping);
/// Real owned UC mapping and the canonical stopped requester's original
/// configuration owner.
pub struct NativeRegisters {
    function: Function<OwnedConfig>,
    mapping: Mapping,
    base: u64,
}

/// Match the retained register mapping to BAR0 and current PCI decoding.
fn native_window(config: &OwnedConfig, mapping: &Mapping) -> Result<u64, Error> {
    let low = config.read32(0x10)?;
    if low == u32::MAX || low & 9 != 0 || !matches!(low & 6, 0 | 4) {
        return Err(Error::Mapping);
    }
    let high = if low & 6 == 4 {
        config.read32(0x14)?
    } else {
        0
    };
    if high == u32::MAX {
        return Err(Error::Mapping);
    }
    let base = u64::from(high) << 32 | u64::from(low & !15);
    let span = mapping.physical_span().map_err(Error::Mmio)?;
    if base == 0
        || span.start() != base
        || span.bytes() < WINDOW_BYTES as u64
        || mapping.attribute().map_err(Error::Mmio)? != CacheAttribute::Uncached
        || mapping.protection().map_err(Error::Mmio)? != Protection::ReadWrite
    {
        return Err(Error::Mapping);
    }
    let pointer = mapping.address(0, WINDOW_BYTES).map_err(Error::Mmio)?;
    if !(pointer.as_ptr() as usize).is_multiple_of(4) {
        return Err(Error::Mapping);
    }
    // Keep the source's memory decode, with no DMA or interrupt publish.
    let command = config.read16(regs::COMMAND)?;
    if command == u16::MAX
        || command & (regs::COMMAND_MASTER | regs::COMMAND_MEMORY | regs::COMMAND_INTX_DISABLE)
            != regs::COMMAND_MEMORY | regs::COMMAND_INTX_DISABLE
    {
        return Err(Error::Configuration);
    }
    Ok(base)
}

impl NativeRegisters {
    /// # Safety
    /// Mapping is this stopped function's exclusive BAR0 reservation at
    /// the same root/firmware epoch, powered D0. It cannot be revoked
    /// through aliases. The original stopped producer's drainage contract holds.
    /// Power/clock and firmware/PUNIT exclusions are boot-lifetime or kept
    /// in the consumed resources' backings, including quarantine. A
    /// borrowed external guard that can end first is insufficient. No implicit
    /// identity mapping or firmware-conflict override is permitted.
    pub unsafe fn claim(
        requester: StoppedRequester,
        mapping: Mapping,
    ) -> Result<Self, (Error, NativeResources)> {
        let mut inputs = ManuallyDrop::new((requester, mapping));
        let stopped = inputs.0.check_stopped();
        let (requester, mapping) = ManuallyDrop::into_inner(inputs);
        let mut resources = ManuallyDrop::new((requester.into_function(), mapping));
        let checked = (|| {
            stopped.map_err(Error::Claim)?;
            let (function, mapping) = &mut *resources;
            native_window(function.config(), mapping)
        })();
        match checked {
            Ok(base) => {
                let (function, mapping) = ManuallyDrop::into_inner(resources);
                Ok(Self {
                    function,
                    mapping,
                    base,
                })
            }
            Err(error) => Err((error, ManuallyDrop::into_inner(resources))),
        }
    }
    /// `offset` as a register of the window.
    fn register(offset: usize) -> Result<usize, Error> {
        if !offset.is_multiple_of(4) || offset > WINDOW_BYTES - 4 {
            return Err(Error::Mapping);
        }
        Ok(offset)
    }
    fn into_resources(self) -> NativeResources {
        (self.function, self.mapping)
    }
}
// SAFETY: unsafe claim consumes canonical stopped ownership and UC
// mapping, validates real BAR/configuration and keeps both for every access.
unsafe impl Registers for NativeRegisters {
    fn identity(&mut self) -> Result<Identity, Error> {
        let expected = u32::from(self.function.vendor) | u32::from(self.function.device) << 16;
        let config = self.function.config();
        if config.read32(0)? != expected {
            return Err(Error::Identity);
        }
        let subsystem = config.read32(0x2c)?;
        if subsystem == u32::MAX {
            return Err(Error::Identity);
        }
        Ok(Identity {
            vendor: expected as u16,
            device: (expected >> 16) as u16,
            subsystem_vendor: subsystem as u16,
            subsystem_device: (subsystem >> 16) as u16,
        })
    }
    fn physical_base(&self) -> u64 {
        self.base
    }
    fn read32(&mut self, offset: usize) -> Result<u32, Error> {
        self.mapping
            .read32(Self::register(offset)?)
            .map_err(Error::Mmio)
    }
    fn write32(&mut self, offset: usize, value: u32) -> Result<(), Error> {
        self.mapping
            .write32(Self::register(offset)?, value)
            .map_err(Error::Mmio)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Unmodified,
    Programming,
    Ready,
    Retired,
}
pub struct Controller<R: Registers> {
    registers: Option<R>,
    profile: Option<Profile>,
    phase: Phase,
}
/// Every initialization or retirement refusal retains its original provider.
pub struct ControllerFailure<R: Registers> {
    pub error: Error,
    owner: ManuallyDrop<Controller<R>>,
}
impl<R: Registers> ControllerFailure<R> {
    pub fn profile(&self) -> Option<Profile> {
        self.owner.profile
    }
}
impl<R: Registers> core::fmt::Debug for ControllerFailure<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ControllerFailure")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}
impl<R: Registers> Controller<R> {
    /// Park the same provider before metadata callbacks, reset and remap.
    pub fn initialize(registers: R) -> Result<Self, ControllerFailure<R>> {
        let mut owner = ManuallyDrop::new(Self {
            registers: Some(registers),
            profile: None,
            phase: Phase::Unmodified,
        });
        let result = (|| {
            let profile = owner
                .registers
                .as_mut()
                .unwrap()
                .identity()
                .and_then(Profile::select)?;
            owner.profile = Some(profile);
            owner.program()
        })();
        match result {
            Ok(()) => Ok(ManuallyDrop::into_inner(owner)),
            Err(error) => Err(ControllerFailure { error, owner }),
        }
    }
    fn program(&mut self) -> Result<(), Error> {
        let caps = self.registers.as_mut().unwrap().read32(CAPS)?;
        if caps == u32::MAX || caps & CAPS_TYPE != 0 {
            return Err(Error::Unsupported);
        }
        let base = self.registers.as_ref().unwrap().physical_base();
        if base == 0
            || !base.is_multiple_of(4096)
            || base.checked_add(WINDOW_BYTES as u64).is_none()
        {
            return Err(Error::Mapping);
        }
        self.phase = Phase::Programming;
        self.write_verified(RESET, 0)?;
        self.write_verified(RESET, RESET_FUNCTION_AND_IDMA)?;
        // Source lo_hi_writeq uses ordered native low then high DWORD stores.
        self.registers
            .as_mut()
            .unwrap()
            .write32(REMAP_LOW, base as u32)?;
        self.registers
            .as_mut()
            .unwrap()
            .write32(REMAP_HIGH, (base >> 32) as u32)?;
        self.verify(REMAP_LOW, base as u32)?;
        self.verify(REMAP_HIGH, (base >> 32) as u32)?;
        self.phase = Phase::Ready;
        Ok(())
    }
    fn verify(&mut self, offset: usize, expected: u32) -> Result<(), Error> {
        let observed = self.registers.as_mut().unwrap().read32(offset)?;
        if observed != expected {
            return Err(Error::Register {
                offset,
                expected,
                observed,
            });
        }
        Ok(())
    }
    fn write_verified(&mut self, offset: usize, value: u32) -> Result<(), Error> {
        self.registers.as_mut().unwrap().write32(offset, value)?;
        self.verify(offset, value)
    }
    /// # Safety
    /// The child has acknowledged disable, or no child has ever been
    /// published. No child register/IRQ/work users remain. This is not a DMA or
    /// external SDA/SCL recovery primitive; this PIO owner never enabled DMA.
    unsafe fn retire(self) -> Result<Retired<R>, ControllerFailure<R>> {
        let mut owner = ManuallyDrop::new(self);
        owner.phase = Phase::Programming;
        match owner.write_verified(RESET, 0) {
            Err(error) => Err(ControllerFailure { error, owner }),
            Ok(()) => {
                owner.phase = Phase::Retired;
                Ok(Retired {
                    registers: owner.registers.take().unwrap(),
                })
            }
        }
    }
    pub fn profile(&self) -> Profile {
        self.profile.unwrap()
    }
}
impl<R: Registers> Drop for Controller<R> {
    fn drop(&mut self) {
        if matches!(self.phase, Phase::Programming | Phase::Ready)
            && let Some(registers) = self.registers.take()
        {
            core::mem::forget(registers);
        }
    }
}

// SAFETY: NativeRegisters keeps the complete UC aperture and original
// function/root epoch. DesignWare offsets are bounded below private registers.
unsafe impl ChildAccess for NativeRegisters {
    fn child_read32(&mut self, offset: usize) -> Result<u32, Error> {
        if offset >= PRIVATE {
            return Err(Error::Mapping);
        }
        self.read32(offset)
    }
    fn child_write32(&mut self, offset: usize, value: u32) -> Result<(), Error> {
        if offset >= PRIVATE {
            return Err(Error::Mapping);
        }
        self.write32(offset, value)
    }
}

#[cfg(test)]
pub(crate) mod tests;

impl From<crate::bus::pci::ConfigError> for Error {
    fn from(error: crate::bus::pci::ConfigError) -> Self {
        Self::Config(error)
    }
}
