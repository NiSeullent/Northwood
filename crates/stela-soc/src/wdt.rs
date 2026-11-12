//! The Apple SoC watchdog (`apple,t8112-wdt`, `apple,wdt`; ADT `/arm-io/wdt`).
//! A bite never resets the host: it becomes a [`super::bus::ResetRequest`]
//! that the engine reports when it stops the guest.
//!
//! Documented by Linux `drivers/watchdog/apple_wdt.c` and m1n1 `src/wdt.c`:
//! three watchdogs, each with CUR_TIME (a writable counter that increments
//! once per reference-clock tick), BITE_TIME and CTRL (IRQ_EN 0, IRQ_STATUS 1,
//! RESET_EN 2). The machine resets while RESET_EN is set and
//! CUR_TIME >= BITE_TIME. Every bite, WD0's included, is a plain restart
//! request: the host restarts into its normal boot selection, and no
//! watchdog selects recovery.
//! WD0 also has BARK_TIME; whenever IRQ_EN is set and CUR_TIME >= BARK_TIME
//! its interrupt and IRQ_STATUS are raised; writing 1 clears IRQ_STATUS. The reference
//! clock is the 24 MHz `clkref`, the system counter's rate.
//!
//! Inferred, not documented: CUR_TIME is a 32-bit count that wraps; the
//! interrupt output follows IRQ_STATUS while IRQ_EN is set; every watchdog starts
//! disarmed at zero, as m1n1 leaves the primary one, and BITE_TIME/BARK_TIME
//! start at zero as Stela's disabled boot handoff image.
//!
//! The DT reg[1] panic-save scratch (panic-save-flag-bit) keeps full32
//! RMWs (`AppleARMWatchdogTimer::_handlePEHaltRestart`, kernel file 0x1ca4a88).
//! reg[2] commands the separate watchdog mechanism. Writing 0 disarms it
//! (`AppleARMWatchdogTimer::_disarmWatchdogMechanism`, 0x1ca6114);
//! `_armWatchdogMechanism` (0x1ca5e28) writes an absolute deadline in whole
//! seconds: uptime converted to nanoseconds plus the timeout (10 s from
//! 0x1ca5f28), divided by 10^9. The deadline counts the system counter's
//! seconds, the time base the driver read; when the counter reaches it while
//! armed, the mechanism requests a reset. That comparator is inferred from
//! the driver's arithmetic. It starts disarmed; command readback is unproved
//! and refuses. reg[3], the AP watchdog configuration, and reg[4], the panic
//! force-power-off flags, keep full32 RMWs: `start` sets configuration
//! bits [5:4] to 1, `_writeApWatchdogConfiguration` (0x1ca5da0) replaces its
//! low byte with expiry flags and `_clearApWatchdogConfiguration`
//! (0x1ca6134) clears it; the panic path sets the force-power-off bit. Both
//! choose what the SoC does when a watchdog expires or the machine resets,
//! and Stela stops the guest at either, so no later guest-visible behavior
//! depends on them. Every side register starts zero in Stela's boot handoff.
//! No side register changes the three primary timers.
//! When the DT assigns panic-save and force-power-off the same physical
//! word, both views share its full32 storage; separate layouts keep
//! separate words. Command/configuration aliases require separate evidence.
//!
//! `CoreTrigger` is one `trigger-config` word in a CPU's CoreSight region,
//! which the driver sets to its value under the record's mask on CPU start
//! events (`writeCoreTriggerConfigurationStateChange`, 0x1ca5cf8). Bits
//! outside the mask refuse; like the configuration, it acts only on expiry.
use stela_contracts::Refusal;
use stela_contracts::refuse;
use {
    stela_contracts::bus::Access, stela_contracts::bus::Direction,
    stela_contracts::bus::MmioDevice, stela_contracts::bus::MmioError, stela_contracts::bus::Width,
    stela_contracts::bus::refuse,
};
use {stela_contracts::interrupt::AicInput, stela_contracts::interrupt::Wire};

const NAME: &str = "apple-wdt";

const CTRL_IRQ_EN: u32 = 1;
const CTRL_IRQ_STATUS: u32 = 1 << 1;
const CTRL_RESET_EN: u32 = 1 << 2;

const CUR_TIME: u64 = 0x0;
const BITE_TIME: u64 = 0x4;
const BARK_TIME: u64 = 0x8;
const CTRL: u64 = 0xc;
const STRIDE: u64 = 0x10;
const WATCHDOGS: usize = 3;
/// The `clkref` reference clock and system counter rate.
const REFERENCE_HZ: u64 = crate::counter::FREQUENCY_HZ;
pub const PANIC_SAVE: u64 = 0x4000;
pub const MECHANISM: u64 = 0x4004;
pub const CONFIGURATION: u64 = 0x4008;
pub const POWER_OFF: u64 = 0x400c;

/// One write to the watchdog mechanism: XNU arms it on its halt, restart
/// and panic path, so the record dates the start of a reset that follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MechanismEvent {
    /// The deadline armed, in system counter seconds; `None` disarms.
    pub deadline: Option<u32>,
    /// The system counter at the write.
    pub at: u64,
    /// The PE that wrote it.
    pub cpu: usize,
}

/// The most mechanism writes kept between two reports; later ones are
/// counted, not kept.
const MECHANISM_LOG: usize = 8;

enum PowerOff {
    Separate(u32),
    /// Both DT views name one physical flags word, not a copied value.
    PanicSave,
}

#[derive(Debug, Clone, Copy, Default)]
struct Watchdog {
    /// Counter value at which CUR_TIME was 0.
    base: u64,
    /// Counter value of the last evaluation.
    evaluated: u64,
    bite: Option<u32>,
    bark: Option<u32>,
    control: u32,
}
impl Watchdog {
    fn current(&self, now: u64) -> u32 {
        now.wrapping_sub(self.base) as u32
    }
    /// Whether CUR_TIME reached `target` in `(evaluated, now]`, or already
    /// was at or past it at `evaluated`.
    fn reached(&self, target: u32, now: u64) -> bool {
        let start = self.current(self.evaluated);
        start >= target || now - self.evaluated >= u64::from(target - start)
    }
    fn deadline(&self, target: u32, now: u64) -> u64 {
        let current = self.current(now);
        if current >= target {
            now
        } else {
            now.saturating_add(u64::from(target - current))
        }
    }
}

pub struct AppleWdt {
    watchdogs: [Watchdog; WATCHDOGS],
    irq: Option<AicInput>,
    irq_level: bool,
    side: bool,
    panic_save: u32,
    /// The armed mechanism deadline, in seconds of the system counter.
    mechanism: Option<u32>,
    /// Mechanism writes since the last [`AppleWdt::take_mechanism_log`].
    mechanism_log: alloc::vec::Vec<MechanismEvent>,
    /// Writes not kept since then, past [`MECHANISM_LOG`].
    mechanism_dropped: u32,
    configuration: u32,
    power_off: PowerOff,
}
impl AppleWdt {
    /// `irq` is the node's DeviceTree `interrupts` AIC input; `now` the
    /// counter value at which every CUR_TIME reads 0.
    pub fn new(irq: Option<AicInput>, now: u64) -> Self {
        let watchdog = Watchdog {
            base: now,
            evaluated: now,
            bite: Some(0),
            bark: Some(0),
            ..Watchdog::default()
        };
        Self {
            watchdogs: [watchdog; WATCHDOGS],
            irq,
            irq_level: false,
            side: false,
            panic_save: 0,
            mechanism: None,
            mechanism_log: alloc::vec::Vec::new(),
            mechanism_dropped: 0,
            configuration: 0,
            power_off: PowerOff::Separate(0),
        }
    }
    /// The mechanism writes since the last call, and how many more were
    /// not kept.
    pub fn take_mechanism_log(&mut self) -> (alloc::vec::Vec<MechanismEvent>, u32) {
        (
            core::mem::take(&mut self.mechanism_log),
            core::mem::take(&mut self.mechanism_dropped),
        )
    }

    /// The optional DT side controls, separate from the timer window.
    pub fn with_side_controls(mut self) -> Self {
        self.side = true;
        self
    }

    /// DT reg[1] and reg[4] name the same full32 flags word (original
    /// T6000: panic-save bit 0 and force-power-off bit 2 at 0x2922b8020).
    /// Both logical views keep the word, including unrelated bits.
    pub fn with_shared_panic_flags(mut self) -> Self {
        self.side = true;
        self.power_off = PowerOff::PanicSave;
        self
    }
    fn evaluate(&mut self, access: &mut Access<'_>) -> Option<u64> {
        let now = access.now;
        let mut next: Option<u64> = None;
        let mut earliest =
            |deadline: u64| next = Some(next.map_or(deadline, |next| next.min(deadline)));
        if let Some(seconds) = self.mechanism {
            let deadline = u64::from(seconds) * REFERENCE_HZ;
            if now >= deadline {
                access.request_reset(NAME, "watchdog mechanism deadline passed: reset requested");
            } else {
                earliest(deadline);
            }
        }
        for (index, watchdog) in self.watchdogs.iter_mut().enumerate() {
            if watchdog.control & CTRL_RESET_EN != 0
                && let Some(bite) = watchdog.bite
            {
                if watchdog.reached(bite, now) {
                    access.request_reset(
                        NAME,
                        if index == 0 {
                            "watchdog 0 bite: system reset requested"
                        } else {
                            "watchdog bite: system reset requested"
                        },
                    );
                } else {
                    earliest(watchdog.deadline(bite, now));
                }
            }
            if watchdog.control & (CTRL_IRQ_EN | CTRL_IRQ_STATUS) == CTRL_IRQ_EN
                && let Some(bark) = watchdog.bark
            {
                if watchdog.reached(bark, now) {
                    watchdog.control |= CTRL_IRQ_STATUS;
                } else {
                    earliest(watchdog.deadline(bark, now));
                }
            }
            watchdog.evaluated = now;
        }
        let level = self.watchdogs[0].control & (CTRL_IRQ_EN | CTRL_IRQ_STATUS)
            == CTRL_IRQ_EN | CTRL_IRQ_STATUS;
        if level != self.irq_level {
            self.irq_level = level;
            if let Some(irq) = self.irq {
                access.interrupts.set_level(Wire::Aic(irq), level);
            }
        }
        next
    }
    fn decode(offset: u64) -> Option<(usize, u64)> {
        let index = usize::try_from(offset / STRIDE)
            .ok()
            .filter(|&index| index < WATCHDOGS)?;
        let register = offset % STRIDE;
        (matches!(register, CUR_TIME | BITE_TIME | CTRL) || register == BARK_TIME && index == 0)
            .then_some((index, register))
    }
}

impl MmioDevice for AppleWdt {
    fn name(&self) -> &'static str {
        NAME
    }
    fn read(
        &mut self,
        offset: u64,
        width: Width,
        access: &mut Access<'_>,
    ) -> Result<u64, MmioError> {
        let fail = |reason: Refusal| refuse(NAME, offset, width, Direction::Read, reason);
        if width != Width::Word {
            return Err(fail(refuse!("watchdog registers are 32-bit")));
        }
        if self.side {
            match offset {
                PANIC_SAVE => return Ok(u64::from(self.panic_save)),
                CONFIGURATION => return Ok(u64::from(self.configuration)),
                POWER_OFF => {
                    return Ok(u64::from(match self.power_off {
                        PowerOff::Separate(word) => word,
                        PowerOff::PanicSave => self.panic_save,
                    }));
                }
                _ => {}
            }
            if offset == MECHANISM {
                return Err(fail(refuse!(
                    "watchdog mechanism command readback is not established"
                )));
            }
        }
        let (index, register) =
            Self::decode(offset).ok_or_else(|| fail(refuse!("unknown watchdog register")))?;
        self.evaluate(access);
        let watchdog = &self.watchdogs[index];
        let value = match register {
            CUR_TIME => watchdog.current(access.now),
            BITE_TIME => watchdog
                .bite
                .ok_or_else(|| fail(refuse!("BITE_TIME read before the guest wrote it")))?,
            BARK_TIME => watchdog
                .bark
                .ok_or_else(|| fail(refuse!("BARK_TIME read before the guest wrote it")))?,
            _ => watchdog.control,
        };
        Ok(u64::from(value))
    }
    fn write(
        &mut self,
        offset: u64,
        width: Width,
        value: u64,
        access: &mut Access<'_>,
    ) -> Result<(), MmioError> {
        let fail = |reason: Refusal| refuse(NAME, offset, width, Direction::Write, reason);
        if width != Width::Word {
            return Err(fail(refuse!("watchdog registers are 32-bit")));
        }
        if self.side {
            let retained = match offset {
                PANIC_SAVE => Some(&mut self.panic_save),
                CONFIGURATION => Some(&mut self.configuration),
                POWER_OFF => Some(match &mut self.power_off {
                    PowerOff::Separate(word) => word,
                    PowerOff::PanicSave => &mut self.panic_save,
                }),
                _ => None,
            };
            if let Some(register) = retained {
                *register = value as u32;
                return Ok(());
            }
            if offset == MECHANISM {
                self.mechanism = (value != 0).then_some(value as u32);
                if self.mechanism_log.len() < MECHANISM_LOG {
                    self.mechanism_log.push(MechanismEvent {
                        deadline: self.mechanism,
                        at: access.now,
                        cpu: access.cpu,
                    });
                } else {
                    self.mechanism_dropped = self.mechanism_dropped.saturating_add(1);
                }
                self.evaluate(access);
                return Ok(());
            }
        }
        let (index, register) =
            Self::decode(offset).ok_or_else(|| fail(refuse!("unknown watchdog register")))?;
        self.evaluate(access);
        let value = value as u32;
        let watchdog = &mut self.watchdogs[index];
        match register {
            CUR_TIME => watchdog.base = access.now.wrapping_sub(u64::from(value)),
            BITE_TIME => watchdog.bite = Some(value),
            BARK_TIME => watchdog.bark = Some(value),
            _ => {
                if value & !(CTRL_IRQ_EN | CTRL_IRQ_STATUS | CTRL_RESET_EN) != 0 {
                    return Err(fail(refuse!("watchdog CTRL reserved field is not modeled")));
                }
                if index != 0 && value & CTRL_IRQ_EN != 0 {
                    return Err(fail(refuse!("only watchdog 0 has a bark interrupt")));
                }
                let status = watchdog.control & CTRL_IRQ_STATUS & !value;
                watchdog.control = value & !CTRL_IRQ_STATUS | status;
            }
        }
        self.evaluate(access);
        Ok(())
    }
    fn advance(&mut self, access: &mut Access<'_>) -> Result<Option<u64>, MmioError> {
        Ok(self.evaluate(access))
    }
}

/// One watchdog `trigger-config` word: keeps the bits of its mask.
pub struct CoreTrigger {
    mask: u32,
    value: u32,
}
impl CoreTrigger {
    pub fn new(mask: u32) -> Self {
        Self { mask, value: 0 }
    }
}
impl MmioDevice for CoreTrigger {
    fn name(&self) -> &'static str {
        "watchdog-core-trigger"
    }
    fn read(&mut self, offset: u64, width: Width, _: &mut Access<'_>) -> Result<u64, MmioError> {
        if offset != 0 || width != Width::Word {
            return Err(refuse(
                self.name(),
                offset,
                width,
                Direction::Read,
                "the core trigger word is one full32 register",
            ));
        }
        Ok(u64::from(self.value))
    }
    fn write(
        &mut self,
        offset: u64,
        width: Width,
        value: u64,
        _: &mut Access<'_>,
    ) -> Result<(), MmioError> {
        if offset != 0 || width != Width::Word || value as u32 & !self.mask != 0 {
            return Err(refuse(
                self.name(),
                offset,
                width,
                Direction::Write,
                "core trigger bits outside the trigger-config mask are not described",
            ));
        }
        self.value = value as u32;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stela_contracts::interrupt::InterruptLatch;

    const WD1: u64 = STRIDE;
    const IRQ: AicInput = AicInput::new(0, 379);

    #[test]
    fn shared_panic_flags_are_one_word_through_both_logical_views() {
        let mut latch = InterruptLatch::new();
        let mut access = Access::new(0, 0, &mut latch);
        let mut wdt = AppleWdt::new(None, 0).with_shared_panic_flags();
        wdt.write(PANIC_SAVE, Width::Word, 0xabcd_0001, &mut access)
            .unwrap();
        assert_eq!(
            wdt.read(POWER_OFF, Width::Word, &mut access),
            Ok(0xabcd_0001)
        );
        let word = wdt.read(POWER_OFF, Width::Word, &mut access).unwrap();
        wdt.write(POWER_OFF, Width::Word, word | 4, &mut access)
            .unwrap();
        assert_eq!(
            wdt.read(PANIC_SAVE, Width::Word, &mut access),
            Ok(0xabcd_0005)
        );
        assert_eq!(wdt.read(CONFIGURATION, Width::Word, &mut access), Ok(0));
        assert_eq!(wdt.read(CUR_TIME, Width::Word, &mut access), Ok(0));
        assert!(access.reset_request().is_none());
        // A distinct physical layout continues to have two independent words.
        let mut separate = AppleWdt::new(None, 0).with_side_controls();
        separate
            .write(PANIC_SAVE, Width::Word, 1, &mut access)
            .unwrap();
        assert_eq!(separate.read(POWER_OFF, Width::Word, &mut access), Ok(0));
    }

    fn write(
        wdt: &mut AppleWdt,
        latch: &mut InterruptLatch,
        now: u64,
        offset: u64,
        value: u32,
    ) -> Option<&'static str> {
        let mut access = Access::new(0, now, latch);
        wdt.write(offset, Width::Word, value.into(), &mut access)
            .unwrap();
        access.reset_request().map(|request| request.reason)
    }
    fn advance(wdt: &mut AppleWdt, latch: &mut InterruptLatch, now: u64) -> (Option<u64>, bool) {
        let mut access = Access::new(0, now, latch);
        let next = wdt.advance(&mut access).unwrap();
        (next, access.reset_request().is_some())
    }

    /// Linux `apple_wdt_set_timeout`, `apple_wdt_start` and `apple_wdt_ping`.
    #[test]
    fn optional_side_controls_hold_panic_scratch_and_only_disarm_the_mechanism() {
        let mut sink = InterruptLatch::new();
        let mut a = Access::new(0, 0, &mut sink);
        let mut w = AppleWdt::new(None, 0).with_side_controls();
        assert_eq!(w.read(BITE_TIME, Width::Word, &mut a), Ok(0));
        assert_eq!(w.read(PANIC_SAVE, Width::Word, &mut a), Ok(0));
        w.write(PANIC_SAVE, Width::Word, 0xabc0, &mut a).unwrap();
        let v = w.read(PANIC_SAVE, Width::Word, &mut a).unwrap();
        w.write(PANIC_SAVE, Width::Word, v | 1, &mut a).unwrap();
        assert_eq!(w.read(PANIC_SAVE, Width::Word, &mut a), Ok(0xabc1));
        // Arming sets a deadline in counter seconds; write0 disarms.
        w.write(MECHANISM, Width::Word, 10, &mut a).unwrap();
        assert_eq!(w.mechanism, Some(10));
        w.write(MECHANISM, Width::Word, 0, &mut a).unwrap();
        assert_eq!(w.mechanism, None);
        let (log, dropped) = w.take_mechanism_log();
        assert_eq!(dropped, 0);
        assert_eq!(
            log,
            [
                MechanismEvent {
                    deadline: Some(10),
                    at: 0,
                    cpu: 0
                },
                MechanismEvent {
                    deadline: None,
                    at: 0,
                    cpu: 0
                },
            ]
        );
        assert_eq!(w.take_mechanism_log(), (alloc::vec::Vec::new(), 0));
        assert!(w.read(MECHANISM, Width::Word, &mut a).is_err());
        assert!(w.read(PANIC_SAVE, Width::Double, &mut a).is_err());
        assert!(w.read(PANIC_SAVE + 0x10, Width::Word, &mut a).is_err());
        // The driver's configuration RMWs: start's bits [5:4], then the
        // expiry flags in the low byte, then a clear of that byte.
        let configuration = w.read(CONFIGURATION, Width::Word, &mut a).unwrap();
        w.write(
            CONFIGURATION,
            Width::Word,
            configuration & !0x30 | 0x10,
            &mut a,
        )
        .unwrap();
        w.write(CONFIGURATION, Width::Word, 0x1d, &mut a).unwrap();
        assert_eq!(w.read(CONFIGURATION, Width::Word, &mut a), Ok(0x1d));
        w.write(POWER_OFF, Width::Word, 1 << 2, &mut a).unwrap();
        assert_eq!(w.read(POWER_OFF, Width::Word, &mut a), Ok(4));
        assert_eq!(w.read(CTRL, Width::Word, &mut a), Ok(0));
        assert!(
            AppleWdt::new(None, 0)
                .read(PANIC_SAVE, Width::Word, &mut a)
                .is_err()
        );
    }

    #[test]
    fn a_core_trigger_word_keeps_only_its_mask() {
        let mut sink = InterruptLatch::new();
        let mut a = Access::new(0, 0, &mut sink);
        let mut trigger = CoreTrigger::new(0xf);
        assert_eq!(trigger.read(0, Width::Word, &mut a), Ok(0));
        trigger.write(0, Width::Word, 1, &mut a).unwrap();
        assert_eq!(trigger.read(0, Width::Word, &mut a), Ok(1));
        assert!(trigger.write(0, Width::Word, 0x11, &mut a).is_err());
        assert!(trigger.read(0, Width::Double, &mut a).is_err());
        assert_eq!(trigger.read(0, Width::Word, &mut a), Ok(1));
    }

    #[test]
    fn mechanism_writes_are_kept_up_to_the_log_bound_then_counted() {
        let mut sink = InterruptLatch::new();
        let mut w = AppleWdt::new(None, 0).with_side_controls();
        for second in 1..=10u64 {
            let mut a = Access::new(3, second * REFERENCE_HZ, &mut sink);
            w.write(MECHANISM, Width::Word, second + 10, &mut a)
                .unwrap();
        }
        let (log, dropped) = w.take_mechanism_log();
        assert_eq!(log.len(), MECHANISM_LOG);
        assert_eq!(dropped, 2);
        assert_eq!(
            log[0],
            MechanismEvent {
                deadline: Some(11),
                at: REFERENCE_HZ,
                cpu: 3
            }
        );
    }

    #[test]
    fn an_armed_mechanism_resets_at_its_deadline_unless_rearmed() {
        let mut latch = InterruptLatch::new();
        let mut w = AppleWdt::new(None, 0).with_side_controls();
        let second = REFERENCE_HZ;
        let mut a = Access::new(0, 5 * second, &mut latch);
        w.write(MECHANISM, Width::Word, 15, &mut a).unwrap();
        assert_eq!(w.advance(&mut a), Ok(Some(15 * second)));
        let mut a = Access::new(0, 14 * second, &mut latch);
        w.write(MECHANISM, Width::Word, 24, &mut a).unwrap();
        let mut a = Access::new(0, 20 * second, &mut latch);
        w.advance(&mut a).unwrap();
        assert!(a.reset_request().is_none());
        let mut a = Access::new(0, 24 * second, &mut latch);
        w.advance(&mut a).unwrap();
        assert!(a.reset_request().is_some());
    }

    #[test]
    fn holes_and_misaligned_offsets_never_alias_control_registers() {
        let mut latch = InterruptLatch::new();
        let mut access = Access::new(0, 0, &mut latch);
        let mut wdt = AppleWdt::new(Some(IRQ), 0);
        for offset in 0..WATCHDOGS as u64 * STRIDE {
            if AppleWdt::decode(offset).is_some() {
                continue;
            }
            assert!(wdt.read(offset, Width::Word, &mut access).is_err());
            assert!(
                wdt.write(offset, Width::Word, CTRL_RESET_EN.into(), &mut access)
                    .is_err()
            );
            assert!(access.reset_request().is_none());
        }
        for watchdog in 0..WATCHDOGS as u64 {
            assert_eq!(
                wdt.read(watchdog * STRIDE + CTRL, Width::Word, &mut access),
                Ok(0)
            );
        }
    }

    #[test]
    fn linux_keepalive_then_expiry() {
        let mut latch = InterruptLatch::new();
        let mut wdt = AppleWdt::new(Some(IRQ), 0);
        let timeout = 24_000_000 * 30;
        assert_eq!(write(&mut wdt, &mut latch, 0, WD1 + CUR_TIME, 0), None);
        write(&mut wdt, &mut latch, 0, WD1 + BITE_TIME, timeout);
        write(&mut wdt, &mut latch, 10, WD1 + CUR_TIME, 0);
        assert_eq!(
            write(&mut wdt, &mut latch, 10, WD1 + CTRL, CTRL_RESET_EN),
            None
        );
        assert_eq!(
            advance(&mut wdt, &mut latch, 1000),
            (Some(10 + u64::from(timeout)), false)
        );
        let mut access = Access::new(0, 1000, &mut latch);
        assert_eq!(wdt.read(WD1 + CUR_TIME, Width::Word, &mut access), Ok(990));
        write(&mut wdt, &mut latch, 1000, WD1 + CUR_TIME, 0);
        assert_eq!(
            advance(&mut wdt, &mut latch, 1000 + u64::from(timeout) - 1),
            (Some(1000 + u64::from(timeout)), false)
        );
        assert!(advance(&mut wdt, &mut latch, 1000 + u64::from(timeout)).1);
    }

    /// Linux `apple_wdt_restart` and m1n1 `wdt_reboot` request a reset; the
    /// model reports it rather than resetting.
    #[test]
    fn restart_is_surfaced_as_a_reset_request() {
        let mut latch = InterruptLatch::new();
        let mut wdt = AppleWdt::new(None, 0);
        assert_eq!(
            write(&mut wdt, &mut latch, 5, WD1 + CTRL, CTRL_RESET_EN),
            Some("watchdog bite: system reset requested"),
            "zero handoff BITE_TIME expires immediately when enabled"
        );
        assert_eq!(
            write(&mut wdt, &mut latch, 6, WD1 + BITE_TIME, 0),
            Some("watchdog bite: system reset requested")
        );
        let mut wdt = AppleWdt::new(None, 0);
        write(&mut wdt, &mut latch, 0, WD1 + BITE_TIME, 0x10_0000);
        write(&mut wdt, &mut latch, 0, WD1 + CUR_TIME, 0);
        write(&mut wdt, &mut latch, 0, WD1 + CTRL, CTRL_RESET_EN);
        assert!(advance(&mut wdt, &mut latch, 0x10_0000).1);
    }

    /// No bite selects recovery: WD0's is the same plain restart as the
    /// others', and says so.
    #[test]
    fn a_watchdog_0_bite_requests_a_plain_restart() {
        let mut latch = InterruptLatch::new();
        let mut wdt = AppleWdt::new(None, 0);
        let mut access = Access::new(0, 5, &mut latch);
        wdt.write(CTRL, Width::Word, CTRL_RESET_EN.into(), &mut access)
            .unwrap();
        let request = access.reset_request().unwrap();
        assert_eq!(
            request.transition,
            stela_contracts::bus::Transition::Restart
        );
        assert_eq!(request.reason, "watchdog 0 bite: system reset requested");
    }

    #[test]
    fn late_evaluation_still_bites_after_wrap() {
        let mut latch = InterruptLatch::new();
        let mut wdt = AppleWdt::new(None, 0);
        write(&mut wdt, &mut latch, 0, WD1 + BITE_TIME, 100);
        write(&mut wdt, &mut latch, 0, WD1 + CTRL, CTRL_RESET_EN);
        assert!(advance(&mut wdt, &mut latch, (1 << 32) + 50).1);
    }

    #[test]
    fn bark_interrupt_latches_and_clears() {
        let mut latch = InterruptLatch::new();
        let mut wdt = AppleWdt::new(Some(IRQ), 0);
        write(&mut wdt, &mut latch, 0, BARK_TIME, 50);
        write(&mut wdt, &mut latch, 0, CTRL, CTRL_IRQ_EN);
        assert_eq!(advance(&mut wdt, &mut latch, 10), (Some(50), false));
        advance(&mut wdt, &mut latch, 60);
        assert!(latch.is_asserted(Wire::Aic(IRQ)));
        let mut access = Access::new(0, 60, &mut latch);
        assert_eq!(
            wdt.read(CTRL, Width::Word, &mut access),
            Ok(u64::from(CTRL_IRQ_EN | CTRL_IRQ_STATUS))
        );
        write(
            &mut wdt,
            &mut latch,
            60,
            CTRL,
            CTRL_IRQ_EN | CTRL_IRQ_STATUS,
        );
        assert!(
            latch.is_asserted(Wire::Aic(IRQ)),
            "the condition still holds"
        );
        write(&mut wdt, &mut latch, 60, CUR_TIME, 0);
        write(
            &mut wdt,
            &mut latch,
            60,
            CTRL,
            CTRL_IRQ_EN | CTRL_IRQ_STATUS,
        );
        assert!(!latch.is_asserted(Wire::Aic(IRQ)));
        assert_eq!(advance(&mut wdt, &mut latch, 70), (Some(110), false));
    }

    #[test]
    fn refuses_unknown_registers_and_fields() {
        let mut latch = InterruptLatch::new();
        let mut access = Access::new(0, 0, &mut latch);
        let mut wdt = AppleWdt::new(None, 0);
        for offset in [WD1 + BARK_TIME, 0x30] {
            assert!(
                wdt.read(offset, Width::Word, &mut access).is_err(),
                "{offset:#x}"
            );
        }
        assert!(
            wdt.write(WD1 + CTRL, Width::Word, u64::from(CTRL_IRQ_EN), &mut access)
                .is_err()
        );
        assert!(wdt.write(CTRL, Width::Word, 1 << 3, &mut access).is_err());
        assert!(wdt.read(CUR_TIME, Width::Byte, &mut access).is_err());
    }
}
