//! Step-loop timing, host-time exclusion, idle skipping and wait limits.

use super::*;

impl<H: Host, D: Devices> Machine<H, D> {
    /// Run host-side work on the devices, such as an external peer's
    /// computation, between steps: the host time it takes is not guest
    /// time, so a serialized count does not advance across it. A parallel
    /// PE's count is the host's: stopped in the kernel, where it may hold a
    /// lock another PE times, the count waits for it only once it has been
    /// stopped `clock::STALL_MILLIS`; at EL0 it holds no such lock.
    pub fn exclude_host_time<T>(&mut self, work: impl FnOnce(&mut D) -> T) -> T {
        let progress = self.work();
        let at_el0 = self.cpu.pstate.el == 0;
        let released = self.clock.leave(|| self.host.counter(), progress, at_el0);
        let before = self.host.counter();
        let result = work(&mut self.devices);
        let after = self.host.counter();
        if released {
            self.clock.resume(|| self.host.counter(), progress);
        }
        self.clock.exclude(after.saturating_sub(before));
        result
    }

    /// Mark the running PE's progress for the shared count's stall guard
    /// every few blocks (`clock::Clock::mark`): a compare at each step.
    #[inline]
    pub(super) fn mark_progress(&mut self) {
        if self.clock.mark_due(self.cpu.retired) {
            self.mark_progress_now();
        }
    }

    #[cold]
    fn mark_progress_now(&mut self) {
        let phases = self.clock_phases();
        let retired = self.cpu.retired;
        self.clock.mark(|| self.host.counter(), retired, phases);
    }

    /// The profile clock and the step loop's phase totals, under step
    /// timing: what the stall report names a gap's phase by.
    pub(super) fn clock_phases(&mut self) -> Option<clock::Phases> {
        let phases = self.step_timing.as_ref()?.phases;
        Some((self.host.profile_ticks(), phases))
    }

    /// End every wait of all PEs at once by leaping the system counter to
    /// the next deadline instead of waiting on the host: guest sleeps cost no
    /// host time, and time of day runs ahead of the host's (`clock`).
    /// Parallel machines share host time; advancing just one PE would
    /// disagree with the counters the other PEs observe.
    pub fn set_idle_skip(&mut self, skip: bool) -> Result<(), &'static str> {
        if skip && self.parallel.is_some() {
            return Err("idle skipping requires all PEs to share one scheduler");
        }
        self.idle_skip = skip;
        Ok(())
    }

    /// Time the step loop's phases and exits from now on ([`step_timing`]).
    pub fn set_step_timing(&mut self, on: bool) {
        self.step_timing = on.then(step_timing::StepTiming::default);
        // Self-loops count their entries only while timing is on.
        self.native_system = None;
    }

    /// The step loop's timing so far, while it is on.
    pub fn step_timing(&self) -> Option<&step_timing::StepTiming> {
        self.step_timing.as_ref()
    }

    /// The profile clock, while timing is on.
    #[inline]
    pub(super) fn timing_start(&mut self) -> u64 {
        match self.step_timing {
            Some(_) => self.host.profile_ticks(),
            None => 0,
        }
    }

    /// The profile clock's ticks since `since`, or none without step timing.
    #[inline]
    pub(super) fn timing_since(&mut self, since: u64) -> u64 {
        match self.step_timing {
            Some(_) => self.host.profile_ticks().wrapping_sub(since),
            None => 0,
        }
    }

    /// Charge the ticks since `since` to `phase`; the profile clock now.
    #[inline]
    pub(super) fn timing_charge(&mut self, phase: step_timing::Phase, since: u64) -> u64 {
        if self.step_timing.is_none() {
            return 0;
        }
        let now = self.host.profile_ticks();
        self.host.profile_boundary(phase, now);
        if let Some(timing) = &mut self.step_timing {
            let spent = now.wrapping_sub(since);
            timing.phases[phase as usize] += spent;
            timing.lifetime_maxima[phase as usize].observe(
                spent,
                now,
                self.cpu.pc,
                self.cpu.pstate.el,
                self.cpu.pstate.daif,
            );
            if phase != step_timing::Phase::Schedule {
                let level = step_timing::Level::of(self.cpu.pstate.el, self.cpu.pstate.guarded);
                timing.levels.charge(level, spent);
            }
        }
        now
    }

    /// Charge the ticks since `since` to handling an exit in `slot`, a native
    /// device fault's or device site's to the register it reached, and a
    /// System exit's to its key ([`Self::take_system_exits`]).
    #[inline]
    pub(super) fn timing_exit(&mut self, slot: usize, since: u64) {
        let device = self.timed_device.take().filter(|_| {
            slot == step_timing::DEVICE_FAULT_SLOT || slot == step_timing::DEVICE_SITE_SLOT
        });
        let Some(timing) = &mut self.step_timing else {
            return;
        };
        let check = timing.check.take();
        let slot = match check {
            Some(check) if slot == step_timing::CHECK_FAULT_SLOT => check.slot(),
            _ => slot,
        };
        let now = self.host.profile_ticks();
        self.host.profile_boundary(step_timing::Phase::Exit, now);
        let closed = core::mem::take(&mut self.timed_closed);
        let waited = device.map_or(0, |_| self.devices.take_lock_wait());
        if let Some(timing) = &mut self.step_timing {
            let spent = now.wrapping_sub(since);
            if let Some(physical) = device {
                let page = step_timing::device_page(physical);
                self.system_exits.devices.add(page, 1, spent, waited);
            }
            if let Some(parts) = self.timed_write.take() {
                self.system_exits.devices.writes.add(spent, parts);
            }
            let level = step_timing::Level::of(self.cpu.pstate.el, self.cpu.pstate.guarded);
            timing.levels.charge(level, spent);
            if let Some(word) = self.timed_word.take() {
                // The interrupt's handler ends with this return, whose own
                // ticks were its last.
                if step_timing::exception_return(word) {
                    timing.levels.in_interrupt = false;
                }
                if slot != stela_compiler::block::ExitKind::FloatingPoint as usize {
                    let timed = self.system_words.entry(word).or_default();
                    timed.accesses += 1;
                    timed.ticks += spent;
                }
                let key = step_timing::system_key(word);
                self.system_exits.add(key, 1, spent, u64::from(closed));
            }
            if let Some(physical) = device
                && let Some(register) = self.device_registers.get_mut(&physical)
            {
                register.ticks += spent;
            }
            timing.phases[step_timing::Phase::Exit as usize] += spent;
            timing.lifetime_maxima[step_timing::Phase::Exit as usize].observe(
                spent,
                now,
                self.cpu.pc,
                self.cpu.pstate.el,
                self.cpu.pstate.daif,
            );
            if let Some(exit) = timing.exits.get_mut(slot) {
                *exit = (exit.0 + 1, exit.1 + spent);
            }
        }
    }

    /// Under step timing, charge the native check fault being handled to
    /// what the engine found at it ([`step_timing::Check`]).
    #[inline]
    pub(super) fn timed_check(&mut self, check: step_timing::Check) {
        if let Some(timing) = &mut self.step_timing {
            timing.check = Some(check);
        }
    }

    /// The word a System exit of block `index` at `pc` completes, which
    /// [`Self::timing_exit`] charges: [`step_timing::DEVICE_SITE_KEY`] at a
    /// device site's stub, else the instruction at `pc`, which is the
    /// block's last unless an instruction the block completes natively fell
    /// back to the engine.
    pub(super) fn timed_system_word(&mut self, index: usize, pc: u64) -> u32 {
        if self.device_stub(index, pc) {
            return step_timing::DEVICE_SITE_KEY;
        }
        let block = &self.code.blocks[index];
        let terminal = block.terminal;
        if pc == block.terminal_pc() {
            return terminal;
        }
        let mut bytes = [0; 4];
        let physical = block.physical.wrapping_add(pc.wrapping_sub(block.pc));
        if self.host.read_ram(physical, &mut bytes) {
            self.admit(u32::from_le_bytes(bytes))
        } else {
            terminal
        }
    }

    /// The System exits by key since the last call, under step timing
    /// ([`step_timing::SystemExits`]).
    pub fn take_system_exits(&mut self) -> step_timing::SystemExits {
        core::mem::take(&mut *self.system_exits)
    }

    /// Whether hot-block samples wait to be taken ([`Self::take_hot_samples`]).
    pub fn has_hot_samples(&self) -> bool {
        !self.hot_samples.is_empty()
    }

    /// Add the hot-block samples since the last call to `hot`
    /// ([`step_timing::HotBlocks`]).
    pub fn take_hot_samples(&mut self, hot: &mut step_timing::HotBlocks) {
        for (pc, level) in self.hot_samples.drain(..) {
            hot.add(pc, level, 1);
        }
        hot.add_dynamic(&self.dynamic_samples);
        self.dynamic_samples.clear();
    }

    /// The transfer budget of the next native run: `budget`, or under step
    /// timing for one run in [`step_timing::HOT_SAMPLE_EVERY`] a uniform draw
    /// from 1..=[`step_timing::HOT_SAMPLE_SPAN`] when that is smaller, with
    /// whether the run is sampled ([`step_timing::HotBlocks`]).
    pub(super) fn sampled_budget(&mut self, budget: u64) -> (u64, bool) {
        if self.step_timing.is_none() || budget == 0 {
            return (budget, false);
        }
        self.hot_countdown -= 1;
        if self.hot_countdown != 0 {
            return (budget, false);
        }
        self.hot_countdown = step_timing::HOT_SAMPLE_EVERY;
        let mut x = self.hot_seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.hot_seed = x;
        let drawn = 1 + x % step_timing::HOT_SAMPLE_SPAN;
        (budget.min(drawn), drawn < budget)
    }

    /// The architectural system count and each generic timer's `(CTL, CVAL)`
    /// in [`timer::CHANNELS`] order, sampled at that count, of the bank that
    /// counts it: the Apple generic timer's from the M3 on, where the kernel
    /// keeps its timer (`timer::agt`). Diagnostics must use this count,
    /// including slice lag and idle leaps, just as guest register reads and
    /// timer delivery do; the host counter can differ.
    pub fn timer_snapshot(&mut self) -> (u64, [(u64, u64); 4]) {
        let now = self.now();
        let bank = timer::system_bank(&self.cpu);
        let timers = timer::CHANNELS.map(|channel| {
            (
                channel.read_control(&self.cpu, bank, now),
                timer::value(&self.cpu, bank, channel.compare),
            )
        });
        (now, timers)
    }

    /// Bound the waits of every PE on the host in one `run` to `limit`
    /// host counter ticks in all, after which it returns [`Stop::Waiting`]
    /// so the host can take inputs that reach the board from outside it,
    /// such as its power button. Host time, as those inputs arrive in it
    /// whether or not a stop holds the count; in all, so that a guest
    /// whose timers end each wait early still returns. An idle skip, which
    /// leaps to a deadline at once, is not bounded. Without a limit a wait
    /// lasts until a deadline.
    pub fn set_wait_limit(&mut self, limit: Option<u64>) {
        self.wait_limit = limit.map(|limit| limit.max(1));
    }
}
