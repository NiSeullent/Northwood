// SPDX-License-Identifier: GPL-2.0-or-later
//! Opt-in forcewake transport for a stopped Gemini Lake PCI function.

use core::mem::ManuallyDrop;
use stela_driver_base::{
    Clock,
    bus::{
        dma::memory::CacheAttribute,
        mmio::owned::{self, Mapping, Protection, Resource, ResourceKind},
        pci::{
            Bar, CAPABILITY_POWER, CapabilityError, ConfigError, ConfigSpace, Function,
            owned::OwnedConfig,
            regs,
            requester::{ClaimError, StoppedRequester},
        },
    },
};

use super::{ForcewakeDomain, forcewake};
use crate::{DISPLAY_CLASS, INTEL_VENDOR};

const BAR0: u16 = 0x10;
const BAR_TYPE_64: u32 = 4;
const BAR_FLAGS: u32 = 0xf;
const PMCSR: u16 = 4;
const PMCSR_STATE: u16 = 3;

/// Map only the low register aperture of GTTMMADR. The GGTT storage uses
/// a separate range in the same BAR, as in i915's intel_uncore_setup_mmio
/// at cc9de5d4b5ab6459f5774946aa3195ad51236fa8.
pub const REGISTER_BYTES: usize = 2 << 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Claim(ClaimError),
    UnsupportedFunction,
    Config(ConfigError),
    Capability(CapabilityError),
    DecodeDisabled,
    PowerState,
    Bar,
    Mapping(owned::Error),
}

/// The original owners. Any detach must still follow their own retirement
/// protocols.
pub struct Parts<P> {
    pub function: Function<OwnedConfig>,
    pub resource: Resource,
    pub mapping: Mapping,
    pub platform: P,
}

pub struct AttachRefusal<P> {
    pub error: Error,
    _parts: ManuallyDrop<Parts<P>>,
}

pub struct WakeRefusal<P> {
    pub error: forcewake::Error,
    _retained: forcewake::Refusal<Registers<P>>,
}

/// BAR0 and its original configuration, root and power owners. Dropping the
/// transport preserves those owners without issuing any register operations.
pub struct Uncore<P> {
    parts: ManuallyDrop<Parts<P>>,
    clock: Clock,
}

impl<P: 'static> Uncore<P> {
    /// Consume the stopped requester for register-only attachment.
    ///
    /// # Safety
    /// `bar` is the size/type of BAR0 in this exact stopped function's frozen
    /// PCI epoch. `resource` is the original platform BAR0 reservation and
    /// `mapping` is its exclusive UC view of the first [REGISTER_BYTES].
    /// The original physical
    /// bridge translation for that BAR is identity. All firmware, IRQ and
    /// driver paths using the BAR or forcewake bits are excluded and drained.
    /// The consumed owners, including `platform`, keep the original DMA/root
    /// containment, powered D0 uncore, calibrated stable clock and CPU
    /// context valid through the operation, including any refusal. The
    /// stopped producer's engine-quiescence grant continues to hold.
    pub unsafe fn attach(
        requester: StoppedRequester,
        bar: Bar,
        resource: Resource,
        mapping: Mapping,
        platform: P,
        clock: Clock,
    ) -> Result<Self, AttachRefusal<P>> {
        let mut inputs = ManuallyDrop::new((requester, resource, mapping, platform));
        let stopped = inputs.0.check_stopped();
        let (requester, resource, mapping, platform) = ManuallyDrop::into_inner(inputs);
        let mut parts = ManuallyDrop::new(Parts {
            function: requester.into_function(),
            resource,
            mapping,
            platform,
        });
        let checked = stopped
            .map_err(Error::Claim)
            .and_then(|()| check_window(&mut parts, bar));
        match checked {
            Ok(()) => Ok(Self { parts, clock }),
            Err(error) => Err(AttachRefusal {
                error,
                _parts: parts,
            }),
        }
    }

    /// Observe one acquire/release handshake. The same owned transport is
    /// returned only after the domain ACKs both transitions.
    pub fn forcewake_cycle(self, domain: ForcewakeDomain) -> Result<Self, WakeRefusal<P>> {
        let clock = self.clock;
        // SAFETY: attach consumed the exclusive register, power and root
        // owners. The private transport permits no other register accesses.
        let awake =
            unsafe { forcewake::acquire(Registers(self), clock, domain) }.map_err(keep_refusal)?;
        awake
            .release()
            .map(|registers| registers.0)
            .map_err(keep_refusal)
    }

    /// Transfer the exact configuration and BAR owners without unmapping.
    ///
    /// # Safety
    /// The recipient keeps the original ownership and retirement obligations.
    /// Consuming a stopped token at attach does not grant a new DMA binding.
    pub unsafe fn into_parts(self) -> Parts<P> {
        ManuallyDrop::into_inner(self.parts)
    }
}

fn check_window<P>(parts: &mut Parts<P>, bar: Bar) -> Result<(), Error> {
    let function = &mut parts.function;
    if function.vendor != INTEL_VENDOR
        || !matches!(function.device, 0x3184 | 0x3185)
        || function.class[0] != DISPLAY_CLASS
    {
        return Err(Error::UnsupportedFunction);
    }
    let Bar::Memory {
        address,
        size,
        prefetchable: false,
        wide,
    } = bar
    else {
        return Err(Error::Bar);
    };
    let config = function.config();
    let low = config.read32(BAR0).map_err(Error::Config)?;
    let flags = if wide { BAR_TYPE_64 } else { 0 };
    if low & BAR_FLAGS != flags {
        return Err(Error::Bar);
    }
    let high = if wide {
        config.read32(BAR0 + 4).map_err(Error::Config)?
    } else {
        0
    };
    let base = u64::from(high) << 32 | u64::from(low & !BAR_FLAGS);
    let mapping = &parts.mapping;
    let span = mapping.physical_span().map_err(Error::Mapping)?;
    let bar_span = parts.resource.physical_span();
    if base == 0
        || base != address
        || bar_span.start() != base
        || bar_span.bytes() != size
        || size < REGISTER_BYTES as u64
        || span.start() != base
        || span.bytes() != REGISTER_BYTES as u64
        || parts.resource.kind() != ResourceKind::Registers
        || mapping.attribute().map_err(Error::Mapping)? != CacheAttribute::Uncached
        || mapping.protection().map_err(Error::Mapping)? != Protection::ReadWrite
        || !mapping
            .retains_resource(&parts.resource)
            .map_err(Error::Mapping)?
    {
        return Err(Error::Bar);
    }
    if !(mapping
        .address(0, REGISTER_BYTES)
        .map_err(Error::Mapping)?
        .as_ptr() as usize)
        .is_multiple_of(size_of::<u32>())
    {
        return Err(Error::Bar);
    }
    let command = config.read16(regs::COMMAND).map_err(Error::Config)?;
    if command == u16::MAX
        || command & (regs::COMMAND_MEMORY | regs::COMMAND_MASTER | regs::COMMAND_INTX_DISABLE)
            != regs::COMMAND_MEMORY | regs::COMMAND_INTX_DISABLE
    {
        return Err(Error::DecodeDisabled);
    }
    for capability in config.capabilities() {
        let capability = capability.map_err(Error::Capability)?;
        if capability.id == CAPABILITY_POWER {
            let status = config
                .read16(capability.offset + PMCSR)
                .map_err(Error::Config)?;
            if status == u16::MAX || status & PMCSR_STATE != 0 {
                return Err(Error::PowerState);
            }
        }
    }
    Ok(())
}

struct Registers<P>(Uncore<P>);

impl<P> forcewake::Registers for Registers<P> {
    fn read32(&mut self, offset: usize) -> Result<u32, owned::Error> {
        self.0.parts.mapping.read32(offset)
    }

    fn write32(&mut self, offset: usize, value: u32) -> Result<(), owned::Error> {
        self.0.parts.mapping.write32(offset, value)
    }
}

fn keep_refusal<P>(refusal: forcewake::Refusal<Registers<P>>) -> WakeRefusal<P> {
    WakeRefusal {
        error: refusal.error,
        _retained: refusal,
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests;
