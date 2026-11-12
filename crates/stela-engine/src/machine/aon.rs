//! What the CLPC core performance ("AON") counters count: the work a PE
//! does, which the board's storage of each counter accumulates
//! (`platform::apple::core_counters`, S3_7_C15_C(2n)_3).
//!
//! AppleT8112CLPC's `CorePerfSampler::enableCore` selects 0x8000, 0x8100 and
//! 0x8700 for counters 0-2 and then sets S3_7_C15_C5_0 bit 0
//! (`machine::impdef`). No public source names the events. What the
//! 26A434 T8112 kernelcache shows (`kernelcache.release.mac14g`):
//!
//! - `SchedulerInterface::contextSwitch` reads only counters 0 and 1
//!   (0xfffffe0009f382ac, 0xfffffe0009f382c4) as wrapping 32-bit deltas
//!   into the off-core thread's sample: the pair a scheduler charges a
//!   thread with, cycles and instructions (inference).
//! - The core sample's consumer at 0xfffffe0009f397f4 divides counter 2's
//!   delta, shifted left by 24, by counter 0's (`lsl x13, x13, #24; udiv
//!   x13, x13, x0`), filters the 8.24 ratio with two tunable rates and
//!   compares it with 0x28f5c, 1% in 8.24 (0xfffffe0009f39934): counter 2
//!   counts a subset of counter 0's units, so counter 0 is the denominator
//!   of a fraction, cycles (inference).
//!
//! So selector 0x8000 counts core cycles and 0x8100 retired instructions
//! (inference from the uses above), each as `machine::pmu` counts PMC0 and
//! PMC1: cycles at the PE's core clock while it does not wait in WFI or
//! WFE, instructions as the engine retires them. Every other selector
//! (0x8700, whose 1% fraction of cycles is not identified, and 0x8300 on
//! later SoCs) counts nothing, which reads as a zero fraction, and the
//! counter keeps what software wrote. Counting needs S3_7_C15_C5_0 bit 0.
//!
//! The core clock is the cluster's highest operating point, as for PMC0: the
//! engine runs a PE no slower for a lower DVFS state. The counts are added
//! to the board's storage when the engine brings the PE's monitors up to
//! date (exception entry and return, the start and end of a wait), so a
//! native read between those points needs no exit and reads the count as
//! of the last one: monotonic, behind by at most the time since then.

/// The core performance counters a SoC selects: ten on the T6050, the most.
pub const SLOTS: usize = 10;

/// Selector of core cycles (inference, see the module documentation).
pub const CYCLES: u64 = 0x8000;
/// Selector of retired instructions (inference).
pub const INSTRUCTIONS: u64 = 0x8100;

/// The enable and event selectors in force over an interval.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Controls {
    /// S3_7_C15_C5_0 bit 0.
    pub enabled: bool,
    /// Each counter's selector, `None` before software writes one.
    pub selectors: [Option<u64>; SLOTS],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    Cycles,
    Instructions,
}

impl Controls {
    fn event(&self, slot: usize) -> Option<Event> {
        if !self.enabled {
            return None;
        }
        match self.selectors[slot]? {
            CYCLES => Some(Event::Cycles),
            INSTRUCTIONS => Some(Event::Instructions),
            _ => None,
        }
    }
}

/// One PE's counting of its core performance counters.
#[derive(Debug, Clone)]
pub struct CoreActivity {
    /// The core clock, in Hz, and the system counter frequency.
    clock: u64,
    frequency: u64,
    synced_count: u64,
    synced_retired: u64,
    waiting: bool,
    controls: Controls,
    /// The cycle fraction, in 1/`frequency` cycle units.
    remainder: u64,
    /// Counts not yet added to the board's storage.
    pending: [u64; SLOTS],
}

impl CoreActivity {
    /// A PE's counting from reset: disabled, nothing selected. A zero clock
    /// or frequency counts nothing.
    pub fn new(clock: u64, frequency: u64) -> Option<Self> {
        (clock != 0 && frequency != 0).then_some(Self {
            clock,
            frequency,
            synced_count: 0,
            synced_retired: 0,
            waiting: false,
            controls: Controls::default(),
            remainder: 0,
            pending: [0; SLOTS],
        })
    }

    /// Count up to the system count `now` and `retired` instructions under
    /// the waiting state and controls held since the last call, then
    /// continue with `waiting` and `controls`.
    pub fn sync(&mut self, now: u64, retired: u64, waiting: bool, controls: Controls) {
        let ticks = now.saturating_sub(self.synced_count);
        let instructions = retired.wrapping_sub(self.synced_retired);
        let mut cycles = None;
        for slot in 0..SLOTS {
            let increment = match self.controls.event(slot) {
                Some(Event::Cycles) if !self.waiting => *cycles.get_or_insert_with(|| {
                    let scaled =
                        u128::from(ticks) * u128::from(self.clock) + u128::from(self.remainder);
                    let frequency = u128::from(self.frequency);
                    self.remainder = (scaled % frequency) as u64;
                    u64::try_from(scaled / frequency).unwrap_or(u64::MAX)
                }),
                Some(Event::Instructions) => instructions,
                _ => 0,
            };
            self.pending[slot] = self.pending[slot].wrapping_add(increment);
        }
        self.synced_count = self.synced_count.max(now);
        self.synced_retired = retired;
        self.waiting = waiting;
        self.controls = controls;
    }

    /// The counts since the last call, per counter, for the board's storage.
    pub fn take(&mut self) -> [u64; SLOTS] {
        core::mem::replace(&mut self.pending, [0; SLOTS])
    }

    /// Whether any counter counts under the current controls.
    pub fn counting(&self) -> bool {
        (0..SLOTS).any(|slot| self.controls.event(slot).is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FREQUENCY: u64 = 24_000_000;
    const CLOCK: u64 = 3_504_000_000;

    fn enabled() -> Controls {
        let mut controls = Controls {
            enabled: true,
            ..Controls::default()
        };
        controls.selectors[0] = Some(CYCLES);
        controls.selectors[1] = Some(INSTRUCTIONS);
        controls.selectors[2] = Some(0x8700);
        controls
    }

    #[test]
    fn counts_cycles_at_the_core_clock_and_retired_instructions() {
        let mut activity = CoreActivity::new(CLOCK, FREQUENCY).unwrap();
        activity.sync(0, 0, false, enabled());
        // One millisecond of system count, 1000 instructions.
        activity.sync(FREQUENCY / 1000, 1000, false, enabled());
        let counts = activity.take();
        assert_eq!(counts[0], CLOCK / 1000);
        assert_eq!(counts[1], 1000);
        assert_eq!(counts[2], 0, "an unidentified event counts nothing");
        assert_eq!(activity.take(), [0; SLOTS]);
    }

    #[test]
    fn frequency_scales_cycles() {
        let mut low = CoreActivity::new(2_424_000_000, FREQUENCY).unwrap();
        let mut high = CoreActivity::new(CLOCK, FREQUENCY).unwrap();
        for activity in [&mut low, &mut high] {
            activity.sync(0, 0, false, enabled());
            activity.sync(FREQUENCY, 0, false, enabled());
        }
        assert_eq!(low.take()[0], 2_424_000_000);
        assert_eq!(high.take()[0], CLOCK);
    }

    #[test]
    fn waiting_counts_no_cycles() {
        let mut activity = CoreActivity::new(CLOCK, FREQUENCY).unwrap();
        activity.sync(0, 0, false, enabled());
        activity.sync(240, 10, true, enabled());
        let busy = activity.take()[0];
        assert_eq!(busy, 240 * CLOCK / FREQUENCY);
        // A second of WFI.
        activity.sync(240 + FREQUENCY, 10, false, enabled());
        assert_eq!(activity.take()[0], 0);
        activity.sync(480 + FREQUENCY, 20, false, enabled());
        assert_eq!(activity.take(), {
            let mut expected = [0; SLOTS];
            expected[0] = busy;
            expected[1] = 10;
            expected
        });
    }

    #[test]
    fn cycles_are_monotonic_over_fractional_steps() {
        let mut activity = CoreActivity::new(CLOCK, FREQUENCY).unwrap();
        activity.sync(0, 0, false, enabled());
        let mut total = 0u64;
        for step in 1..=FREQUENCY / 1000 {
            activity.sync(step, step, false, enabled());
            total += activity.take()[0];
        }
        // Fractions carry: no cycle is lost or invented over 1 ms.
        assert_eq!(total, CLOCK / 1000);
    }

    #[test]
    fn reset_and_disable_count_nothing() {
        let mut activity = CoreActivity::new(CLOCK, FREQUENCY).unwrap();
        assert!(!activity.counting());
        activity.sync(FREQUENCY, 100, false, Controls::default());
        assert_eq!(activity.take(), [0; SLOTS], "nothing counts from reset");
        activity.sync(2 * FREQUENCY, 200, false, enabled());
        assert_eq!(
            activity.take(),
            [0; SLOTS],
            "controls apply from their write"
        );
        assert!(activity.counting());
        let disabled = Controls {
            enabled: false,
            ..enabled()
        };
        activity.sync(3 * FREQUENCY, 300, false, disabled);
        assert_eq!(activity.take()[1], 100);
        activity.sync(4 * FREQUENCY, 400, false, disabled);
        assert_eq!(activity.take(), [0; SLOTS], "the enable stops counting");
        assert!(CoreActivity::new(0, FREQUENCY).is_none());
    }
}
