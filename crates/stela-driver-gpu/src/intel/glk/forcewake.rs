// SPDX-License-Identifier: GPL-2.0-or-later
//! One Gen9 LP forcewake request over a caller-qualified uncore owner.
//! Register pairs, kernel bit and ACK interval were checked against i915
//! intel_uncore.c and intel_gt_regs.h at cc9de5d4b5ab6459f5774946aa3195ad51236fa8.
//! GLK's WaRsForcewakeAddDelayForAck applies to pre-acquire and acquire
//! acknowledgements in the Render, GT and Media domains.

use super::{ForcewakeDomain, ForcewakeRegisters};
use core::mem::ManuallyDrop;
use stela_driver_base::{Clock, ClockError, bus::mmio::owned, waiting};

const ACK_TIMEOUT_US: u64 = 50_000;
const KERNEL: u32 = 1;
const FALLBACK: u32 = 1 << 15;
const FALLBACK_ATTEMPTS: u64 = 10;
const FALLBACK_DELAY_STEP_US: u64 = 10;

/// Ordered 32-bit reads and masked writes of the retained uncore registers.
pub trait Registers {
    fn read32(&mut self, offset: usize) -> Result<u32, owned::Error>;
    fn write32(&mut self, offset: usize, value: u32) -> Result<(), owned::Error>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Idle,
    Acquire,
    Release,
    FallbackIdle,
    FallbackSet,
    FallbackRelease,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Clock(ClockError),
    Access {
        phase: Phase,
        offset: usize,
        error: owned::Error,
    },
    UnreadableAck(Phase),
    Timeout(Phase),
}

/// Failure keeps the original register/power owner. An unacknowledged effect
/// supplies no permission to issue a second request or retire the aperture.
pub struct Refusal<R> {
    pub error: Error,
    _registers: ManuallyDrop<R>,
}

/// A kernel request acknowledged for one domain. Call [Self::release] to
/// clear it. Abandonment retains the owner and issues no register effects.
pub struct Awake<R> {
    registers: ManuallyDrop<R>,
    clock: Clock,
    domain: ForcewakeDomain,
}

/// Wait for an idle kernel ACK, publish the request, then wait for its ACK.
///
/// # Safety
/// The original, exclusive GLK BAR owner keeps the uncached aperture,
/// platform power, clocks and execution context valid for both this call and
/// release. No firmware, other driver or nested request uses this domain's
/// kernel or kernel fallback bit. The clock is calibrated and stable in that
/// same context. A returned observation qualifies only the forcewake handshake.
pub unsafe fn acquire<R: Registers>(
    registers: R,
    clock: Clock,
    domain: ForcewakeDomain,
) -> Result<Awake<R>, Refusal<R>> {
    let mut registers = ManuallyDrop::new(registers);
    let pair = domain.registers();
    let result = (|| {
        wait_kernel_ack(&mut *registers, clock, pair, Phase::Idle, 0)?;
        request(
            &mut *registers,
            pair.request,
            Phase::Acquire,
            KERNEL,
            KERNEL,
        )?;
        wait_kernel_ack(&mut *registers, clock, pair, Phase::Acquire, KERNEL)
    })();
    match result {
        Ok(()) => Ok(Awake {
            registers,
            clock,
            domain,
        }),
        Err(error) => Err(Refusal {
            error,
            _registers: registers,
        }),
    }
}

impl<R: Registers> Awake<R> {
    /// Clear the same kernel request and return the original owner only after
    /// the same domain acknowledges deassertion.
    pub fn release(self) -> Result<R, Refusal<R>> {
        let Self {
            mut registers,
            clock,
            domain,
        } = self;
        let pair = domain.registers();
        let result =
            request(&mut *registers, pair.request, Phase::Release, KERNEL, 0).and_then(|()| {
                wait_ack(
                    &mut *registers,
                    clock,
                    pair.acknowledge,
                    Phase::Release,
                    KERNEL,
                    0,
                )
            });
        match result {
            Ok(()) => Ok(ManuallyDrop::into_inner(registers)),
            Err(error) => Err(Refusal {
                error,
                _registers: registers,
            }),
        }
    }
}

fn request(
    registers: &mut impl Registers,
    offset: usize,
    phase: Phase,
    mask: u32,
    value: u32,
) -> Result<(), Error> {
    registers
        .write32(offset, (mask << 16) | value)
        .map_err(|error| Error::Access {
            phase,
            offset,
            error,
        })
}

fn wait_kernel_ack(
    registers: &mut impl Registers,
    clock: Clock,
    pair: ForcewakeRegisters,
    phase: Phase,
    expected: u32,
) -> Result<(), Error> {
    match wait_ack(registers, clock, pair.acknowledge, phase, KERNEL, expected) {
        Err(Error::Timeout(_)) => {}
        result => return result,
    }
    for attempt in 1..=FALLBACK_ATTEMPTS {
        // i915 toggles the independent kernel fallback bit to recover a
        // lost normal ACK. Each fallback transition needs its own ACK
        // before this owner can continue or hand back the aperture.
        wait_ack(
            registers,
            clock,
            pair.acknowledge,
            Phase::FallbackIdle,
            FALLBACK,
            0,
        )?;
        request(
            registers,
            pair.request,
            Phase::FallbackSet,
            FALLBACK,
            FALLBACK,
        )?;
        clock
            .try_delay(FALLBACK_DELAY_STEP_US * attempt)
            .map_err(Error::Clock)?;
        wait_ack(
            registers,
            clock,
            pair.acknowledge,
            Phase::FallbackSet,
            FALLBACK,
            FALLBACK,
        )?;
        let kernel_ack = read_ack(registers, pair.acknowledge, phase)? & KERNEL;
        request(registers, pair.request, Phase::FallbackRelease, FALLBACK, 0)?;
        wait_ack(
            registers,
            clock,
            pair.acknowledge,
            Phase::FallbackRelease,
            FALLBACK,
            0,
        )?;
        if kernel_ack == expected {
            break;
        }
    }
    // Confirm the normal ACK and released fallback bit in the same sample.
    wait_ack(
        registers,
        clock,
        pair.acknowledge,
        phase,
        KERNEL | FALLBACK,
        expected,
    )
}

fn read_ack(registers: &mut impl Registers, offset: usize, phase: Phase) -> Result<u32, Error> {
    let value = registers.read32(offset).map_err(|error| Error::Access {
        phase,
        offset,
        error,
    })?;
    if value == u32::MAX {
        return Err(Error::UnreadableAck(phase));
    }
    Ok(value)
}

fn wait_ack(
    registers: &mut impl Registers,
    clock: Clock,
    offset: usize,
    phase: Phase,
    mask: u32,
    expected: u32,
) -> Result<(), Error> {
    let deadline = clock.try_deadline(ACK_TIMEOUT_US).map_err(Error::Clock)?;
    while !deadline.passed() {
        let value = read_ack(registers, offset, phase)?;
        if deadline.passed() {
            break;
        }
        if value & mask == expected {
            return Ok(());
        }
        waiting();
        core::hint::spin_loop();
    }
    Err(Error::Timeout(phase))
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests;
