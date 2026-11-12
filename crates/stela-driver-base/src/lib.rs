//! The foundation every Stela PC driver builds on: PCI and the other buses,
//! DMA, MMIO ownership, block devices, platform services (ACPI, native work,
//! embedded controllers), the calibrated [`Clock`] and the [`Handoff`] the
//! UEFI mount records before it leaves boot services.
//!
//! The device-class drivers live in their own crates above this one; the
//! `stela-drivers` facade probes them.
#![no_std]
#[macro_use]
extern crate alloc;

use alloc::{sync::Arc, vec::Vec};
use core::{
    ops::Range,
    sync::atomic::{AtomicPtr, Ordering},
};

pub mod block;
pub mod budget;
pub mod bus;
pub mod entropy;
pub mod input_observation;
pub mod platform;
pub mod sync;
#[cfg(any(test, feature = "test-support"))]
pub mod test_clock;
pub mod timing;

use stela_driver_core::firmware;

use block::BlockError;
use bus::{mmio::Mmio, pci};

/// What `stela.efi` records for the host drivers before it leaves boot
/// services.
#[derive(Clone, Debug)]
pub struct Handoff {
    /// ECAM windows from the ACPI MCFG table.
    pub ecam: Vec<pci::EcamWindow>,
    /// MMIO ranges mapped uncached in Stela's tables: the ECAM windows and
    /// every memory BAR of a storage controller or input device.
    pub mapped: Vec<Range<u64>>,
    /// Identity-mapped, physically contiguous pages for DMA, below 4 GiB.
    pub dma: Range<u64>,
    /// The time-stamp counter rate, measured against the firmware's stall.
    pub clock: Clock,
    /// Verified host-device firmware; sharing a handoff never copies image bytes.
    pub firmware: Option<Arc<firmware::Catalog>>,
    /// The message a disk sends when it completes a command, where the
    /// engine takes one: it wakes the core that looks for completions
    /// ([`block::BlockDevice::signals`]). `None` leaves every disk polled.
    pub completion: Option<bus::irq::MsiMessage>,
}

impl Handoff {
    /// Whether `range` lies inside one mapped MMIO range.
    pub fn is_mapped(&self, range: &Range<u64>) -> bool {
        self.mapped
            .iter()
            .any(|mapped| mapped.start <= range.start && range.end <= mapped.end)
    }
}

/// The registers at `offset..offset + bytes` of a memory BAR, after checking
/// that Stela mapped them uncached.
pub fn registers(
    handoff: &Handoff,
    bar: Option<pci::Bar>,
    offset: u64,
    bytes: u64,
) -> Result<Mmio, BlockError> {
    let bar = bar
        .and_then(|bar| bar.memory())
        .ok_or(BlockError::Unsupported("register BAR is not memory"))?;
    let start = bar
        .start
        .checked_add(offset)
        .filter(|start| start.saturating_add(bytes) <= bar.end)
        .ok_or(BlockError::Unsupported("registers outside their BAR"))?;
    if !handoff.is_mapped(&(start..start + bytes)) {
        return Err(BlockError::Unsupported("registers are not mapped"));
    }
    // SAFETY: mapped uncached by stela.efi and owned by this one driver.
    Ok(unsafe { Mmio::new(start, bytes as usize) })
}

/// What runs while a driver waits: the mount's boot screen moves its
/// marker. A `fn()` pointer, null when none is set.
static WAITING: AtomicPtr<()> = AtomicPtr::new(core::ptr::null_mut());

/// Call `hook` while any driver waits, often and briefly; it must not
/// drive a device.
pub fn on_wait(hook: fn()) {
    WAITING.store(hook as *mut (), Ordering::Relaxed);
}

/// What runs as a driver starts on a device, with the device named: the
/// mount's boot screen shows it, so a device that never answers is named
/// there. A `fn(&str)` pointer, null when none is set.
static WORKING: AtomicPtr<()> = AtomicPtr::new(core::ptr::null_mut());

/// Call `hook` with a device's name as a driver starts on it; it must not
/// drive a device.
pub fn on_work(hook: fn(&str)) {
    WORKING.store(hook as *mut (), Ordering::Relaxed);
}

/// Name `device` to the [`on_work`] hook as a driver starts on it.
pub fn working(device: &str) {
    let hook = WORKING.load(Ordering::Relaxed);
    if !hook.is_null() {
        // SAFETY: `on_work` stored a `fn(&str)`, the pointer's only source.
        let hook: fn(&str) = unsafe { core::mem::transmute(hook) };
        hook(device);
    }
}

/// What a processor runs while it spins for a lock another one holds: the
/// platform's safe point, where it answers what the holder may be waiting
/// on. A `fn()` pointer, null when none is set.
static SPINNING: AtomicPtr<()> = AtomicPtr::new(core::ptr::null_mut());

/// Call `hook` on every turn of a driver's lock spin; it must not take a
/// driver lock or drive a device.
pub fn on_spin(hook: fn()) {
    SPINNING.store(hook as *mut (), Ordering::Release);
}

/// One turn of a lock spin: the [`on_spin`] hook, then a pause.
pub fn spinning() {
    let hook = SPINNING.load(Ordering::Acquire);
    if !hook.is_null() {
        // SAFETY: `on_spin` stored a `fn()`, the pointer's only source.
        let hook: fn() = unsafe { core::mem::transmute(hook) };
        hook();
    }
    core::hint::spin_loop();
}

/// Run the [`on_wait`] hook; drivers call it as they wait.
pub fn waiting() {
    let hook = WAITING.load(Ordering::Relaxed);
    if !hook.is_null() {
        // SAFETY: `on_wait` stored a `fn()`, the pointer's only source.
        let hook: fn() = unsafe { core::mem::transmute(hook) };
        hook();
    }
}

/// An unavailable calibration or duration which cannot fit the counter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClockError {
    Uncalibrated,
    DurationOverflow,
    DurationTooLong,
}

/// Microsecond waits from a producer-calibrated time-stamp counter.
/// The producer must preserve a stable TSC rate and processor/clock context.
/// Native attachment must check its maximum duration before device effects.
#[derive(Clone, Copy, Debug)]
pub struct Clock {
    pub ticks_per_microsecond: u64,
}

impl Clock {
    /// The raw time-stamp counter, for intervals another party converts.
    pub fn ticks() -> u64 {
        #[cfg(any(test, feature = "test-support"))]
        if let Some(ticks) = test_clock::read() {
            return ticks;
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: RDTSC has no memory effects; x86-64 always has it.
        return unsafe { core::arch::x86_64::_rdtsc() };
        #[cfg(not(target_arch = "x86_64"))]
        unreachable!("the time-stamp counter read outside x86-64")
    }

    fn duration_ticks(&self, microseconds: u64) -> Result<u64, ClockError> {
        if self.ticks_per_microsecond == 0 {
            return Err(ClockError::Uncalibrated);
        }
        let ticks = self
            .ticks_per_microsecond
            .checked_mul(microseconds)
            .ok_or(ClockError::DurationOverflow)?;
        // Leave a quarter of a raw counter revolution in which an expired
        // deadline can be observed and another quarter in which a counter
        // read before the start, on a core whose counter lags by a few
        // cycles, is told apart from one long past it ([`Deadline`]).
        if ticks > u64::MAX / 4 {
            return Err(ClockError::DurationTooLong);
        }
        Ok(ticks)
    }

    /// Validate calibration and the complete required interval before any
    /// device programming. Intervals may span at most a quarter of a raw
    /// counter revolution. Zero duration still requires a calibrated clock.
    pub fn check_duration(&self, microseconds: u64) -> Result<(), ClockError> {
        self.duration_ticks(microseconds).map(|_| ())
    }

    /// Keep an opaque raw-counter origin for a wrapping elapsed-time timer.
    /// The same calibrated clock/context must own every later observation.
    pub fn try_start(&self) -> Result<u64, ClockError> {
        self.check_duration(0)?;
        Ok(Self::ticks())
    }

    pub fn elapsed_at(&self, start: u64, counter: u64) -> Result<u64, ClockError> {
        self.check_duration(0)?;
        Ok(counter.wrapping_sub(start) / self.ticks_per_microsecond)
    }

    /// Elapsed whole microseconds from this clock's own raw start. Subtract
    /// before division, preserving a rollover inside the measured interval.
    /// The origin must be observed within one complete raw TSC revolution.
    pub fn try_elapsed_us(&self, start: u64) -> Result<u64, ClockError> {
        self.check_duration(0)?;
        self.elapsed_at(start, Self::ticks())
    }

    /// Microseconds since the raw counter's arbitrary origin. This quotient
    /// rolls over when the raw TSC does; use Deadline for wrap-safe intervals.
    pub fn try_now_us(&self) -> Result<u64, ClockError> {
        self.check_duration(0)?;
        Ok(Self::ticks() / self.ticks_per_microsecond)
    }

    /// Infallible compatibility entry for a previously validated clock.
    /// Invalid calibration panics explicitly; it never supplies a fake time.
    #[track_caller]
    pub fn now_us(&self) -> u64 {
        self.try_now_us().expect("Clock requires a calibrated TSC")
    }

    fn deadline_with(
        &self,
        microseconds: u64,
        counter: impl FnOnce() -> u64,
    ) -> Result<Deadline, ClockError> {
        let ticks = self.duration_ticks(microseconds)?;
        Ok(Deadline {
            start: counter(),
            ticks,
        })
    }

    /// A checked interval keeping its raw start counter and full duration.
    /// No absolute counter addition or saturation can shorten this interval.
    pub fn try_deadline(&self, microseconds: u64) -> Result<Deadline, ClockError> {
        self.deadline_with(microseconds, Self::ticks)
    }

    /// A deadline for a validated calibration and representable duration.
    /// Invalid requests panic; native drivers should preflight or use try_*.
    #[track_caller]
    pub fn deadline(&self, microseconds: u64) -> Deadline {
        self.try_deadline(microseconds)
            .expect("Clock requires calibrated, representable timing")
    }

    fn delay_with(
        &self,
        microseconds: u64,
        mut counter: impl FnMut() -> u64,
        mut relax: impl FnMut(),
    ) -> Result<(), ClockError> {
        let deadline = self.deadline_with(microseconds, &mut counter)?;
        while !deadline.passed_at(counter()) {
            relax();
        }
        Ok(())
    }

    fn relax() {
        waiting();
        core::hint::spin_loop();
    }

    /// Spin for at least the checked full interval, including TSC rollover.
    pub fn try_delay(&self, microseconds: u64) -> Result<(), ClockError> {
        self.delay_with(microseconds, Self::ticks, Self::relax)
    }

    /// Wait using a previously validated clock. Invalid requests panic instead
    /// of silently completing a required hardware delay.
    #[track_caller]
    pub fn delay(&self, microseconds: u64) {
        self.try_delay(microseconds)
            .expect("Clock requires calibrated, representable timing")
    }

    fn wait_with(
        &self,
        microseconds: u64,
        mut ready: impl FnMut() -> bool,
        mut counter: impl FnMut() -> u64,
        mut relax: impl FnMut(),
        at: &'static core::panic::Location<'static>,
    ) -> Result<bool, ClockError> {
        let deadline = self.deadline_with(microseconds, &mut counter)?;
        loop {
            if ready() {
                return Ok(true);
            }
            let now = counter();
            if deadline.passed_at(now) || budget::passed_at(now, at) {
                return Ok(ready());
            }
            relax();
        }
    }

    /// Poll until ready or elapsed timeout, keeping the final observation at
    /// expiry. Invalid timing fails before invoking the device predicate.
    /// A passed device start [`budget`] ends it as a timeout.
    #[track_caller]
    pub fn try_wait(
        &self,
        microseconds: u64,
        ready: impl FnMut() -> bool,
    ) -> Result<bool, ClockError> {
        let at = core::panic::Location::caller();
        self.wait_with(microseconds, ready, Self::ticks, Self::relax, at)
    }

    /// Poll using validated timing. Invalid requests panic explicitly; they
    /// cannot turn an already-ready predicate into an uncalibrated success.
    #[track_caller]
    pub fn wait(&self, microseconds: u64, ready: impl FnMut() -> bool) -> bool {
        self.try_wait(microseconds, ready)
            .expect("Clock requires calibrated, representable timing")
    }
}

/// One raw-counter interval. Poll more often than once per half TSC revolution
/// and retire it after expiry; its total age must not reach a full revolution.
/// The start/duration representation permits a rollover inside that interval.
/// It grants no device completion beyond elapsed time.
#[derive(Clone, Copy, Debug)]
pub struct Deadline {
    start: u64,
    ticks: u64,
}

impl Deadline {
    /// Whether `counter` lies at least the deadline's ticks after its
    /// start. A counter before the start, as another core's may read by a
    /// few cycles, has not passed it: intervals span at most a quarter of
    /// the counter's revolution ([`Clock::check_duration`]).
    pub fn passed_at(&self, counter: u64) -> bool {
        let elapsed = counter.wrapping_sub(self.start) as i64;
        elapsed >= 0 && elapsed as u64 >= self.ticks
    }
    /// [`Self::passed_at`] now, or a passed device start [`budget`].
    #[track_caller]
    pub fn passed(&self) -> bool {
        let now = Clock::ticks();
        self.passed_at(now) || budget::passed_at(now, core::panic::Location::caller())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_ranges_must_contain_the_whole_range() {
        let handoff = Handoff {
            firmware: None,
            completion: None,
            ecam: Vec::new(),
            mapped: vec![0x1000..0x3000, 0x8000..0x9000],
            dma: 0..0,
            clock: Clock {
                ticks_per_microsecond: 1,
            },
        };
        assert!(handoff.is_mapped(&(0x1000..0x3000)));
        assert!(handoff.is_mapped(&(0x2000..0x2800)));
        assert!(!handoff.is_mapped(&(0x2000..0x4000)));
        assert!(!handoff.is_mapped(&(0x3000..0x8800)));
    }

    #[test]
    fn cloning_a_driver_handoff_shares_the_immutable_firmware_catalog() {
        let catalog = Arc::new(firmware::Catalog::from_handoff(vec![]).unwrap());
        let handoff = Handoff {
            ecam: Vec::new(),
            mapped: Vec::new(),
            dma: 0..0,
            clock: Clock {
                ticks_per_microsecond: 1,
            },
            firmware: Some(catalog.clone()),
            completion: None,
        };
        let cloned = handoff.clone();
        assert!(Arc::ptr_eq(cloned.firmware.as_ref().unwrap(), &catalog));
        drop(handoff);
        assert_eq!(cloned.firmware.as_ref().unwrap().inventory(), (0, 0));
    }

    #[test]
    fn a_zero_wait_still_polls_once() {
        let clock = Clock {
            ticks_per_microsecond: 1,
        };
        let mut calls = 0;
        assert!(!clock.wait(0, || {
            calls += 1;
            false
        }));
        assert!(calls >= 1);
    }

    #[test]
    fn clock_checked_duration_rejects_missing_calibration_and_overflow() {
        let invalid = Clock {
            ticks_per_microsecond: 0,
        };
        assert_eq!(invalid.check_duration(0), Err(ClockError::Uncalibrated));
        assert_eq!(invalid.try_now_us(), Err(ClockError::Uncalibrated));
        assert_eq!(invalid.try_start(), Err(ClockError::Uncalibrated));
        assert_eq!(invalid.try_elapsed_us(0), Err(ClockError::Uncalibrated));
        assert_eq!(invalid.try_delay(1), Err(ClockError::Uncalibrated));
        let high = Clock {
            ticks_per_microsecond: u64::MAX,
        };
        assert_eq!(high.duration_ticks(1), Err(ClockError::DurationTooLong));
        assert_eq!(high.check_duration(2), Err(ClockError::DurationOverflow));
        assert_eq!(high.try_delay(2), Err(ClockError::DurationOverflow));
        let ordinary = Clock {
            ticks_per_microsecond: 3_000,
        };
        assert_eq!(ordinary.duration_ticks(10_000), Ok(30_000_000));
        assert_eq!(ordinary.duration_ticks(0), Ok(0));
    }

    #[test]
    fn clock_deadline_keeps_complete_interval_across_raw_counter_rollover() {
        let clock = Clock {
            ticks_per_microsecond: 3,
        };
        for start in [100, u64::MAX - 4] {
            let deadline = clock.deadline_with(2, || start).unwrap();
            for elapsed in 0..6 {
                assert!(!deadline.passed_at(start.wrapping_add(elapsed)));
            }
            assert!(deadline.passed_at(start.wrapping_add(6)));
            assert!(deadline.passed_at(start.wrapping_add(7)));
        }
        let largest = Clock {
            ticks_per_microsecond: 1,
        }
        .deadline_with(u64::MAX / 4, || u64::MAX - 4)
        .unwrap();
        assert!(!largest.passed_at((u64::MAX - 4).wrapping_add(u64::MAX / 4 - 1)));
        assert!(largest.passed_at((u64::MAX - 4).wrapping_add(u64::MAX / 4)));
        assert!(largest.passed_at((u64::MAX - 4).wrapping_add(u64::MAX / 4 + 1)));
        // Another core's counter a few cycles behind the start.
        assert!(!largest.passed_at(u64::MAX - 7));
        assert_eq!(
            Clock {
                ticks_per_microsecond: 1
            }
            .check_duration(u64::MAX / 4 + 1),
            Err(ClockError::DurationTooLong)
        );
    }

    #[test]
    fn clock_delay_uses_shared_loop_without_truncation_or_wrap_hang() {
        let clock = Clock {
            ticks_per_microsecond: 3,
        };
        for start in [100, u64::MAX - 4] {
            let mut counter = start;
            let mut reads = 0;
            let mut relaxations = 0;
            clock
                .delay_with(
                    2,
                    || {
                        reads += 1;
                        assert!(reads <= 7, "shared delay did not finish at wrap");
                        let value = counter;
                        counter = counter.wrapping_add(1);
                        value
                    },
                    || {
                        relaxations += 1;
                    },
                )
                .unwrap();
            assert_eq!(reads, 7, "the entire six-tick interval must elapse");
            assert_eq!(relaxations, 5);
            assert_eq!(counter, start.wrapping_add(7));
        }
    }

    #[test]
    fn clock_shared_loops_allow_counter_observations_to_skip_exact_expiry() {
        let clock = Clock {
            ticks_per_microsecond: 1,
        };
        for (start, duration) in [(100, 6), (u64::MAX - 4, 6), (u64::MAX - 4, u64::MAX / 4)] {
            let observations = [
                start,
                start.wrapping_add(duration - 1),
                start.wrapping_add(duration + 1),
            ];
            let mut next = observations.into_iter();
            let mut relaxations = 0;
            clock
                .delay_with(
                    duration,
                    || next.next().expect("delay hung past expiry"),
                    || {
                        relaxations += 1;
                    },
                )
                .unwrap();
            assert_eq!(relaxations, 1);
            assert!(next.next().is_none());
            let mut next = observations.into_iter();
            let mut polls = 0;
            assert_eq!(
                clock.wait_with(
                    duration,
                    || {
                        polls += 1;
                        polls == 3
                    },
                    || next.next().expect("poll hung past expiry"),
                    || {},
                    core::panic::Location::caller()
                ),
                Ok(true)
            );
            assert_eq!(polls, 3);
            assert!(next.next().is_none());
        }
    }

    #[test]
    fn clock_invalid_wait_has_no_device_counter_or_wait_hook_effects() {
        for (rate, us, error) in [
            (0, 0, ClockError::Uncalibrated),
            (0, 10, ClockError::Uncalibrated),
            (u64::MAX, 2, ClockError::DurationOverflow),
            (u64::MAX, 1, ClockError::DurationTooLong),
            (1, u64::MAX / 4 + 1, ClockError::DurationTooLong),
        ] {
            let clock = Clock {
                ticks_per_microsecond: rate,
            };
            assert_eq!(
                clock.wait_with(
                    us,
                    || panic!("device predicate on invalid timing"),
                    || panic!("counter origin on invalid timing"),
                    || panic!("wait hook on invalid timing"),
                    core::panic::Location::caller()
                ),
                Err(error)
            );
            assert_eq!(
                clock.delay_with(
                    us,
                    || panic!("counter origin on invalid delay"),
                    || panic!("wait hook on invalid delay")
                ),
                Err(error)
            );
            assert_eq!(
                clock.try_wait(us, || panic!("native predicate on invalid timing")),
                Err(error)
            );
        }
    }

    #[test]
    fn clock_poll_preserves_final_observation_and_zero_timeout_order() {
        let clock = Clock {
            ticks_per_microsecond: 1,
        };
        let mut polls = 0;
        let ready = clock
            .wait_with(
                0,
                || {
                    polls += 1;
                    polls == 2
                },
                || 77,
                || panic!("no relaxation at zero timeout"),
                core::panic::Location::caller(),
            )
            .unwrap();
        assert!(ready, "final expiry observation remains part of the API");
        assert_eq!(polls, 2);
        let mut counter = u64::MAX - 2;
        let mut polls = 0;
        let mut relaxations = 0;
        let ready = clock
            .wait_with(
                4,
                || {
                    polls += 1;
                    false
                },
                || {
                    let value = counter;
                    counter = counter.wrapping_add(1);
                    value
                },
                || {
                    relaxations += 1;
                },
                core::panic::Location::caller(),
            )
            .unwrap();
        assert!(!ready);
        assert_eq!(polls, 5);
        assert_eq!(relaxations, 3);
    }

    #[test]
    fn clock_elapsed_subtracts_raw_ticks_before_microsecond_conversion() {
        let clock = Clock {
            ticks_per_microsecond: 3,
        };
        for start in [100, u64::MAX - 4] {
            for elapsed in 0..=15 {
                assert_eq!(
                    clock.elapsed_at(start, start.wrapping_add(elapsed)),
                    Ok(elapsed / 3)
                );
            }
        }
    }

    #[test]
    fn clock_legacy_invalid_requests_fail_explicitly_instead_of_completing() {
        extern crate std;
        let invalid = Clock {
            ticks_per_microsecond: 0,
        };
        assert!(std::panic::catch_unwind(|| invalid.now_us()).is_err());
        assert!(std::panic::catch_unwind(|| invalid.deadline(1)).is_err());
        assert!(std::panic::catch_unwind(|| invalid.delay(1)).is_err());
        assert!(std::panic::catch_unwind(|| invalid.wait(1, || true)).is_err());
        let overflow = Clock {
            ticks_per_microsecond: u64::MAX,
        };
        assert!(std::panic::catch_unwind(|| overflow.delay(2)).is_err());
    }
}
