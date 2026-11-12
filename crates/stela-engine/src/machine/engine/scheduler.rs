//! The serial run loop: device polling, stepping, native execution and block
//! linking.

use super::*;

/// Native runs between two promotion steps, and the stamped cells each
/// looks at: every stamped cell once in 256 steps, a lock and 256 loads
/// in 1024 runs.
const PROMOTION_RUNS: u32 = 1024;
const PROMOTION_CELLS: u32 = 256;

pub(super) fn native_instruction_incomplete(step: &Step) -> bool {
    matches!(
        step,
        Err(Stop::Unsupported(Requirement {
            kind: RequirementKind::NativePreparation(_)
                | RequirementKind::NativeRamServiceBusy
                | RequirementKind::DeviceSiteRamServiceBusy
                | RequirementKind::NonresidentRam { .. }
                | RequirementKind::RamTransfer { .. },
            ..
        }))
    )
}

impl<H: Host, D: Devices> Machine<H, D> {
    /// Run until a stop. `budget` limits executed blocks (0 = unlimited).
    pub fn run(&mut self, budget: u64) -> Stop {
        self.run_with_budget(RunBudget {
            blocks: budget,
            host_ticks: None,
        })
    }

    /// Run within block and optional host-time limits, yielding between
    /// engine steps. An entered native chain or host callback may overrun.
    pub fn run_with_budget(&mut self, budget: RunBudget) -> Stop {
        self.latch_canonical_native_failure();
        if let Some(failure) = &self.native_failure {
            return Stop::NativeFailure(failure.summary.clone());
        }
        if let Some(pending) = self.pending_native_device_access() {
            return pending;
        }
        if self.cpu.has_canonical_memory() || self.peers.iter().any(|peer|
            matches!(peer, processors::Peer::Parked(parked) if parked.cpu().has_canonical_memory())) {
            return self.unsupported(RequirementKind::Native(
                "converted shared-memory PEs run only through run_serialized or the native scheduler",
            ));
        }
        self.run_blocks_with_budget(budget)
    }

    /// The serial scheduler and chained native blocks, until a stop.
    pub(super) fn run_blocks(&mut self, budget: u64) -> Stop {
        self.run_blocks_with_budget(RunBudget {
            blocks: budget,
            host_ticks: None,
        })
    }

    pub(super) fn run_blocks_with_budget(&mut self, budget: RunBudget) -> Stop {
        self.latch_canonical_native_failure();
        if let Some(failure) = &self.native_failure {
            return Stop::NativeFailure(failure.summary.clone());
        }
        let deadline = budget
            .host_ticks
            .map(|ticks| budget::Deadline::new(self.host.counter(), ticks));
        self.observe_parallel_source(SourcePhase::SliceEnter);
        if let Some(hub) = &self.parallel {
            // The loop's first schedule reads the mail posted meanwhile.
            hub.set_outside(self.current, false);
        }
        self.block_limit = if budget.blocks == 0 {
            u64::MAX
        } else {
            budget.blocks
        };
        self.waited = 0;
        // The host may have changed the devices between runs (`devices` is
        // theirs too): their lines are asked again.
        self.device_lines = None;
        loop {
            if self.statistics.blocks >= self.block_limit
                || deadline
                    .is_some_and(|deadline| deadline.remaining(self.host.counter()).is_none())
            {
                self.leave_run_loop();
                return Stop::Budget;
            }
            if let Some(hub) = &self.parallel {
                hub.pass(self.current);
            }
            let started = self.timing_start();
            let scheduled = match deadline {
                Some(deadline) => self.schedule_with_deadline(Some(deadline)),
                None => self.schedule(),
            };
            self.timing_charge(step_timing::Phase::Schedule, started);
            if let Err(stop) = scheduled
                .and_then(|()| self.step())
                .or_else(|stop| self.undefined_instead(stop))
            {
                self.leave_run_loop();
                return stop;
            }
        }
    }

    /// Leave the run loop: other PEs' broadcasts stop waiting for this one
    /// until it enters again.
    fn leave_run_loop(&mut self) {
        // What this PE kept of shared counts, for the reports and the ring.
        self.flush_exits();
        self.flush_entered();
        // Out of the wait, a PE in WFE has guest work the host's work keeps
        // from it (`clock`); one waiting for an interrupt or off has none.
        if !self.waits_for_interrupt() && !self.processor_paused(self.current) {
            let (retired, phases) = (self.cpu.retired, self.clock_phases());
            self.clock.busy(|| self.host.counter(), retired, phases);
        }
        self.observe_parallel_source(SourcePhase::SliceExit);
        if let Some(hub) = &self.parallel {
            hub.set_outside(self.current, true);
        }
    }

    /// The same RAM publication path as MMIO-triggered DMA, reached with no
    /// translated invocation live. Called while running and while all PEs wait.
    pub(super) fn poll_devices(&mut self, now: u64) -> Step {
        if let Some(deadline) = self.devices.dma_deadline() {
            self.poll_devices_at(now, deadline)?;
        }
        Ok(())
    }

    /// Dispatch need not observe time when no asynchronous DMA is pending.
    /// Scheduler callers already have a timestamp and use `poll_devices`.
    pub(super) fn poll_due_devices(&mut self) -> Step {
        if let Some(deadline) = self.devices.dma_deadline()
            && !self.clock.below(deadline, || self.host.counter())
        {
            let now = self.now();
            self.poll_devices_at(now, deadline)?;
        }
        Ok(())
    }

    pub(super) fn poll_devices_at(&mut self, now: u64, deadline: u64) -> Step {
        if now >= deadline {
            let started = self.timing_start();
            self.settle_devices(now);
            self.statistics.due_transfers += 1;
            self.statistics.transfer_ticks += self.timing_since(started);
            self.device_lines = None;
        }
        Ok(())
    }

    pub(super) fn step(&mut self) -> Step {
        let started = self.timing_start();
        self.poll_due_devices()?;
        let delivered = self.deliver_interrupts();
        self.timing_charge(step_timing::Phase::Devices, started);
        if let Some(reason) = self.devices.take_reset() {
            return Err(Stop::Reset(reason));
        }
        if delivered {
            return Ok(());
        }
        let pc = self.cpu.pc;
        let mode = self.debug_mode();
        // Active-pending software step comes before every other exception
        // of the instruction (Arm `CheckSoftwareStep`).
        if mode.step && !self.cpu.pstate.ss {
            let exception = self.debug_exception(Class::StepLower, Class::StepSame, 0x22, pc);
            self.take(exception);
            return Ok(());
        }
        if !pc.is_multiple_of(INSTRUCTION) {
            let mut exception = Exception::synchronous(Class::PcAlignment, 0, pc);
            exception.fault_address = Some(pc);
            self.take(exception);
            return Ok(());
        }
        if (mode.breakpoints || self.cpu.pstate.il)
            && let Some(exception) = self.instruction_checks(pc, mode.breakpoints)?
        {
            self.take(exception);
            return Ok(());
        }
        if self.breakpoints.binary_search(&pc).is_ok() {
            return Err(Stop::Breakpoint(pc));
        }
        if self.traces.binary_search(&pc).is_ok() {
            self.trace(pc);
        }
        // Active-not-pending: this one instruction advances the step.
        self.cpu.step_advance = mode.step;
        // PSTATE.PACM applies to the one instruction after PACM.
        let pacm = self.cpu.pstate.pacm;
        let single = mode.breakpoints || mode.step || pacm;
        let fetching = self.timing_start();
        let fetched = self.fetch_block(pc, single);
        self.timing_charge(step_timing::Phase::Fetch, fetching);
        let index = match fetched? {
            Fetched::Block(index) => index,
            Fetched::Exception(exception) => {
                self.cpu.step_advance = false;
                self.take(exception);
                return Ok(());
            }
            Fetched::Checked(word) => {
                let step = self.checked_instruction(pc, word);
                if native_instruction_incomplete(&step) {
                    self.cpu.step_advance = false;
                    return step;
                }
                if !core::mem::take(&mut self.pacm_written) && pacm {
                    self.cpu.pstate.pacm = false;
                }
                self.advance_step();
                return step;
            }
        };
        // A block table entry that admitted the BTYPE answered the check.
        if self.cpu.pstate.btype != 0 && !self.landing_admitted && self.fetch_guarded(pc) {
            let word = self.word(index, pc)?;
            if let Some(exception) = self.branch_target_exception(pc, word) {
                self.take(exception);
                return Ok(());
            }
        }
        let step = self.execute(index, if single { 0 } else { self.chain_budget() });
        if self.native_failure.is_some() {
            return step;
        }
        if native_instruction_incomplete(&step) {
            // A stopped instruction does not consume PACM or advance
            // architectural single-step state.
            self.cpu.step_advance = false;
            return step;
        }
        if !core::mem::take(&mut self.pacm_written) && pacm {
            self.cpu.pstate.pacm = false;
        }
        self.advance_step();
        step
    }

    /// Blocks the next native run may chain into: none past the time slice
    /// or the run's limit.
    pub(super) fn chain_budget(&self) -> u64 {
        let next = self.statistics.blocks + 1;
        CHAIN_LIMIT
            .min(self.slice_end.saturating_sub(next))
            .min(self.block_limit.saturating_sub(next))
    }

    /// The end of the native lower range, which EL0 blocks' one-sided
    /// checks compare with (`GuestState::lower_limit`).
    pub(super) fn lower_limit(&self) -> stela_compiler::Limit {
        stela_compiler::Limit(self.host.placement().check().lower_limit())
    }

    /// What native code finds this PE's data through.
    pub(super) fn anchors(&self) -> Anchors {
        let (pointer, memo) = self.pointer.anchors();
        let mut anchors = Anchors::default();
        anchors[Anchor::Control] = self.chain.address();
        anchors[Anchor::Mirrors] = &*self.mirrors as *const Mirrors as u64;
        anchors[Anchor::Cells] = self.mirrors.cells().as_ptr() as u64;
        anchors[Anchor::Pointer] = pointer;
        anchors[Anchor::Memo] = memo;
        anchors[Anchor::Count] = self.clock.reader();
        anchors[Anchor::StackZero] = self
            .cpu
            .registers
            .address(SystemRegister::SP_EL0, 0)
            .expect("SP_EL0 has storage");
        anchors[Anchor::Amx] = &self.cpu.amx.native as *const _ as u64;
        anchors
    }

    /// Run compiled block `index` and the blocks it chains into, up to
    /// `budget` of them, and complete the last one's exit or fault.
    pub(super) fn execute(&mut self, index: usize, budget: u64) -> Step {
        let (budget, sampled) = self.sampled_budget(budget);
        let entry = self.code.blocks[index].code;
        self.finish_transition(budget);
        self.refresh_transition_contexts(budget);
        // A transition may continue in this run unless counters tell the
        // levels apart (`transitions`).
        self.mirrors.transitions.chain =
            u64::from(self.cpu.pmu.as_ref().is_none_or(|pmu| pmu.alike()));
        self.chain.begin(index, budget);
        self.cpu.state.daif = u32::from(self.cpu.pstate.daif);
        self.mirrors.selected = u64::from(self.cpu.pstate.sp);
        self.mirrors.transition(self.mirror_state());
        self.mirrors.enter(self.cpu.pstate);
        self.cpu.state.pan = u64::from(self.cpu.pstate.pan);
        self.enter_amx_cell();
        self.cpu.state.anchors = self.anchors();
        self.cpu.state.lower_limit = self.lower_limit();
        let entered = self.timing_start();
        // SAFETY: installed by this machine since the last discard, for the system-mode block ABI;
        // chained blocks and thunks too.
        let exit = unsafe { self.host.try_execute(entry, &mut self.cpu.state) };
        if self.mirrors.switches.len != 0 {
            self.take_thread_switches();
        }
        let transitions = self.mirrors.transitions.take_counts();
        let transitioned = transitions != [0; transition::KINDS];
        if transitioned {
            self.adopt_native_state();
        }
        let returned = self.timing_charge(step_timing::Phase::Native, entered);
        self.leave_amx_cell();
        self.cpu.pstate.daif = self.cpu.state.daif as u8;
        self.cpu.pstate.sp = self.mirrors.selected != 0;
        self.mirrors.leave(&mut self.cpu.pstate);
        if self.cpu.pstate.pan != (self.cpu.state.pan != 0) {
            self.cpu.pstate.pan = !self.cpu.pstate.pan;
            self.sync_host_regime();
        }
        let exit = exit.map_err(|reason| {
            self.unsupported(match reason {
                NativeEntryError::RamServiceBusy => RequirementKind::NativeRamServiceBusy,
                NativeEntryError::Refused(detail) => {
                    RequirementKind::NativePreparation(detail.into())
                }
            })
        })?;
        let remaining = self.chain.budget().load(Ordering::Relaxed);
        let (block, retired, hits, _) = self.chain.completed();
        self.promotion_runs = self.promotion_runs.wrapping_add(1);
        if self.promotion_runs.is_multiple_of(PROMOTION_RUNS) {
            self.promote_hot_exits();
        }
        // A link spends a transfer and counts nothing else.
        let chained = if remaining > budget {
            budget
        } else {
            budget - remaining
        };
        let last = (block & u64::from(u32::MAX)) as usize;
        // A block another PE retired since this run entered it is still
        // in its slot: the slot is given again only after this PE's grace.
        if let NativeExit::Fault(fault) = exit
            && self.code.blocks.get(last).is_none()
        {
            return Err(self.fail_native(index, fault, NativeFailureReason::UnknownCode));
        }
        // A sampled run whose budget ran out was in `last` at a uniformly
        // chosen transfer (`step_timing::HotBlocks`).
        if sampled
            && remaining > budget
            && let Some(block) = self.code.blocks.get(last)
        {
            let level = step_timing::Level::of(self.cpu.pstate.el, self.cpu.pstate.guarded);
            // Kernel and guarded PCs as offsets from this PE's vector base
            // at its EL, which moves with the kernel's slide.
            let pc = if level == step_timing::Level::User {
                block.pc
            } else {
                let vbar = if self.cpu.pstate.el == 2 {
                    SystemRegister::VBAR_EL2
                } else {
                    SystemRegister::VBAR_EL1
                };
                block.pc.wrapping_sub(self.cpu.read(vbar) & !0x7ff)
            };
            self.hot_samples.push((pc, level));
            // Its RET, BLR or BR form's transfer, at most as many as the
            // hot samples a take would keep.
            if let NativeExit::Exit { next_pc, kind } = exit
                && let Some((form, _)) = super::dynamic_branch(block.terminal)
                && ExitWord::split(kind).0 == form as u64
                && self.dynamic_samples.len() < step_timing::DYNAMIC_SAMPLES
            {
                // Bit 11 selects a PAC form.
                let pac = block.terminal & 1 << 11 != 0;
                let form = match (form, pac) {
                    (ExitKind::Return, false) => step_timing::DynamicForm::Ret,
                    (ExitKind::Call, false) => step_timing::DynamicForm::Blr,
                    (_, false) => step_timing::DynamicForm::Br,
                    (ExitKind::Return, true) => step_timing::DynamicForm::Reta,
                    (ExitKind::Call, true) => step_timing::DynamicForm::Blra,
                    (_, true) => step_timing::DynamicForm::Bra,
                };
                self.dynamic_samples.push(step_timing::DynamicSample {
                    form,
                    site: block.pc,
                    target: next_pc,
                });
            }
        }
        self.statistics.blocks += 1 + chained;
        self.statistics.chained += chained;
        self.statistics.lookup_hits += hits;
        // A transfer entered the last block, with the BTYPE it names: 0
        // from every thunk but those of BR and BLR forms.
        let btype = (chained > 0).then_some((block >> chain::BLOCK_BTYPE_SHIFT) as u8);
        match exit {
            NativeExit::Exit { next_pc, kind } => {
                let (kind, skipped) = ExitWord::split(kind);
                self.cpu.retired = self
                    .cpu
                    .retired
                    .wrapping_add(retired)
                    .wrapping_sub(u64::from(skipped));
                if transitioned {
                    // The last block's instructions, which its exit counts,
                    // are the old level's when its transition ended the run.
                    let ending = if kind == ExitKind::Return as u64 {
                        u64::from(self.code.blocks[last].instructions)
                    } else {
                        0
                    };
                    self.adopt_transitions(transitions, ending);
                }
                if let Some(btype) = btype {
                    self.cpu.pstate.btype = btype;
                }
                if let Some(count) = self.statistics.exit_kinds.get_mut(kind as usize) {
                    *count += 1;
                }
                if let Some(miss) = self.lookup_miss(last, next_pc, kind, remaining > budget) {
                    self.statistics.lookup_misses[miss] += 1;
                }
                let mut slot = kind as usize;
                if kind == ExitKind::System as u64 && self.step_timing.is_some() {
                    let word = self.timed_system_word(last, next_pc);
                    if word == step_timing::DEVICE_SITE_KEY {
                        slot = step_timing::DEVICE_SITE_SLOT;
                    }
                    self.timed_word = Some(word);
                }
                let exited = self.exit(last, next_pc, kind);
                if exited.is_ok() {
                    let btype = self.cpu.pstate.btype;
                    match ExitKind::try_from(kind) {
                        Ok(kind @ (ExitKind::Fallthrough | ExitKind::Direct | ExitKind::Call))
                            if btype == 0 =>
                        {
                            self.link(last, next_pc, ExitWord { kind, skipped });
                        }
                        Ok(ExitKind::Return | ExitKind::Indirect | ExitKind::Call) => {
                            self.link_dynamic(last, btype, next_pc);
                        }
                        _ => {}
                    }
                }
                self.timing_exit(slot, returned);
                exited
            }
            NativeExit::Fault(mut fault) => {
                let at = (entry.0 as u64).wrapping_add(fault.offset as u64);
                fault.offset = at.wrapping_sub(self.code.blocks[last].code.0 as u64) as usize;
                let slot = step_timing::fault_slot(&fault.cause);
                self.timed_device = None;
                if transitioned {
                    self.adopt_transitions(transitions, 0);
                }
                // A self-loop's earlier iterations in this entry
                // (`GuestState::loop_retired`), which only its exits move.
                let iterations = core::mem::take(&mut self.cpu.state.loop_retired);
                let handled =
                    self.native_fault(last, fault, retired.wrapping_add(iterations), btype);
                self.timing_exit(slot, returned);
                handled
            }
        }
    }

    /// Why block `index`'s exit of `kind` to `next_pc` took no lookup
    /// transfer, as an index of `Statistics::lookup_misses`, when it is one
    /// a lookup dispatcher serves; `spent` when a dispatcher found no budget left.
    pub(super) fn lookup_miss(
        &self,
        index: usize,
        next_pc: u64,
        kind: u64,
        spent: bool,
    ) -> Option<usize> {
        let block = &self.code.blocks[index];
        let record = self.code.blocks.record(index);
        let kind = ExitKind::try_from(kind).ok()?;
        let dynamic = dynamic_branch(block.terminal)
            .or_else(|| transitions::transition_branch(block.terminal));
        let unlinked = if dynamic.is_some() && kind != ExitKind::Direct {
            block.dynamic_site.is_some() && record.dynamic_open()
        } else if (next_pc ^ block.pc) & !(FETCH_PAGE - 1) != 0
            && matches!(
                kind,
                ExitKind::Fallthrough | ExitKind::Direct | ExitKind::Call
            )
        {
            block.successors.iter().enumerate().any(|(position, exit)| {
                !record.linked(position) && exit.target.0 == next_pc && exit.kind == kind as u8
            })
        } else {
            return None;
        };
        if self.variants[block.variant.0 as usize].single {
            return None;
        }
        if unlinked {
            return Some(0);
        }
        let Some(key) = self.current_key(next_pc) else {
            return Some(1);
        };
        let table = &self.code.table;
        let entry = table.find(table.slot(next_pc, key), next_pc, key);
        Some(if entry.pc != next_pc || entry.key != key {
            2
        } else if spent {
            3
        } else {
            4
        })
    }

    /// Redirect block `index`'s direct exit to `next_pc` into the block a
    /// fetch there would run, once that block exists: one in the same
    /// page, so the same translation and permissions, compiled for the same
    /// variant. Its BTYPE is 0, as no branch that sets it ends in a direct
    /// exit, and every System register write, exception and device access
    /// that changes what the next fetch checks returns to the engine. An
    /// exit out of the page goes through a cell of the lookup table
    /// instead ([`Self::link_through_cell`]). An exit in the page whose
    /// target has no block yet waits for the fetch that compiles it
    /// ([`Machine::direct_exit`]). Whether the exit is linked now.
    pub(super) fn link(&mut self, index: usize, next_pc: u64, word: impl Into<ExitWord>) -> bool {
        let word = word.into();
        let unlinked = |code: &board::Code| {
            let block = &code.blocks[index];
            let record = code.blocks.record(index);
            block
                .successors
                .iter()
                .enumerate()
                .position(|(position, exit)| {
                    !record.linked(position) && exit.target.0 == next_pc && exit.word() == word
                })
        };
        // Nearly every exit finds nothing to link: decided on the board
        // borrowed, without a clone of its `Arc`, whose count every PE's
        // exits would write.
        {
            let block = &self.code.blocks[index];
            if block.successors.is_empty()
                || self.variants[block.variant.0 as usize].single
                || self.traces.binary_search(&next_pc).is_ok()
                || unlinked(&self.code).is_none()
            {
                return false;
            }
            // An exit in the page to a PC without a block yet, as the
            // finder says without the lock: it waits for the fetch that
            // compiles the block, which links it under the lock and checks
            // it all again there ([`Self::link_direct_exit`]). In a launch
            // nearly every new block is reached so, and the lock taken here
            // found nothing to do.
            if (next_pc ^ block.pc) & !(FETCH_PAGE - 1) == 0 {
                let physical = block.physical.wrapping_add(next_pc.wrapping_sub(block.pc));
                let code = &self.code;
                if code
                    .finder
                    .find(&code.blocks, next_pc, physical, block.variant)
                    == Some(None)
                {
                    let incarnation = code.blocks.record(index).incarnation();
                    self.direct_exit = Some((index, incarnation, next_pc, word));
                    return false;
                }
            }
        }
        let code = Arc::clone(&self.code);
        let block = &code.blocks[index];
        let record = code.blocks.record(index);
        let find = || unlinked(&code);
        // Another PE may link or detach it meanwhile: found again under the
        // lock, for a block still installed.
        let mut cache = code.cache_at(board::LockSite::Link);
        // Only an indexed block links out: a detach finds the sources of
        // its links among the indexed blocks of the target's page, so a
        // link from a block an IC IVAU forgot would outlive its target.
        let indexed = cache
            .index
            .get(&code.blocks, block.pc, block.physical, block.variant)
            == Some(index);
        let Some(position) = find().filter(|_| indexed) else {
            return false;
        };
        let retired = u32::from(block.instructions) - u32::from(word.skipped);
        if (next_pc ^ block.pc) & !(FETCH_PAGE - 1) != 0 {
            let point = chain::direct_patch(block.successors[position].site);
            if self.link_through_cell(&mut cache, point, next_pc, retired, word, 0) {
                record.settle(position);
                return true;
            }
            return false;
        }
        let physical = block.physical.wrapping_add(next_pc.wrapping_sub(block.pc));
        let Some(target) = cache
            .index
            .get(&code.blocks, next_pc, physical, block.variant)
        else {
            self.direct_exit = Some((index, record.incarnation(), next_pc, word));
            return false;
        };
        let exit = block.successors[position];
        let site = exit.site;
        let entry = code.blocks[target].code.0 as u64;
        // A slot's body runs in the block's own code, once: its patch
        // point falls into it once it is written ([`chain::link`]).
        if exit.slot != 0 && !record.used(position) {
            let at = site - u64::from(exit.slot);
            let point = chain::slot_patch(at);
            let Some((body, bytes)) =
                chain::link(at, exit.slot.into(), retired, target as u32, entry)
                    .filter(|_| chain::patchable(point))
            else {
                return false;
            };
            self.patch_native(NativeCode(body as usize), &bytes);
            self.patch_point(NativeCode(point as usize), chain::NOP5);
            record.set_linked(position, true, true);
            self.statistics.links += 1;
            return true;
        }
        // Otherwise a thunk: from the slot's point, back to the exit when
        // no transfer is allowed; or from the exit's own point, which has
        // loaded the target, returning.
        let (point, fallback) = match exit.slot {
            0 => (chain::direct_patch(site), chain::Fallback::Return(word)),
            slot => (
                chain::slot_patch(site - u64::from(slot)),
                chain::Fallback::Jump(site),
            ),
        };
        if !chain::patchable(point) {
            return false;
        }
        self.install_stream = self.stream_at(point);
        let Ok(thunk) = self.install_native(&[0; chain::THUNK_BYTES], false) else {
            return false;
        };
        let at = thunk.0 as u64;
        // The thunk's part was chosen meanwhile: the exit stays unlinked.
        if !self.note_reference(point, at) {
            return false;
        }
        let (Some(bytes), Some(jump)) = (
            chain::thunk(at, retired, target as u32, entry, fallback),
            chain::jump(point, at),
        ) else {
            return false;
        };
        self.patch_native(thunk, &bytes);
        self.patch_point(NativeCode(point as usize), jump);
        record.set_linked(position, true, false);
        self.statistics.links += 1;
        true
    }

    /// Link the direct exit waiting for a block at `pc`
    /// ([`Machine::direct_exit`]), which the fetch at `pc` just found or
    /// compiled: whether it is linked now, so that the exit, not a lookup
    /// entry, enters the block. Every block discard the exit's block may
    /// meet forgets it first ([`Self::evict_code`]).
    pub(super) fn link_direct_exit(&mut self, pc: u64) -> bool {
        match self.direct_exit.take() {
            Some((block, incarnation, target, word))
                if target == pc && self.code.blocks.record(block).incarnation() == incarnation =>
            {
                let linked = self.link(block, target, word);
                // The block of the fetch is another variant's.
                self.direct_exit = None;
                linked
            }
            _ => false,
        }
    }

    /// Redirect block `index`'s exit to the target in RAX, which just left
    /// BTYPE `btype`, through a lookup dispatcher, whose target any fetch may
    /// have run. It sets the BTYPE the form sets from the block's
    /// page: 0 for RET, 0b10 for BLR forms, 0b01 for BR forms through X16
    /// or X17, and for BR forms through another register 0b11 in a guarded
    /// page and 0b01 elsewhere. Blocks are compiled per guarded page, so
    /// the page's BTYPE is the block's.
    ///
    /// The exit of a BRA* thunk, which just authenticated `target`, first
    /// compares the target with it ([`chain::predicted_tail`]): an arm64e
    /// auth stub's BRAA X16, X17 has one target, and most BRA* transfers are
    /// at such sites, so an equal target takes a direct exit linked through
    /// a cell, which a hot site turns into a direct jump into the target's
    /// block ([`chain::stamped_branch_tail`]). The thunk's memo still
    /// authenticates every transfer: only which block an authenticated
    /// target enters is cached, and validated as every cell is. RETA* and
    /// BLRA* sites mostly see several targets, where the compare would cost
    /// a mispredicted branch at each predicted one: they keep the lookup.
    pub(super) fn link_dynamic(&mut self, index: usize, btype: u8, target: u64) {
        // Decided on the board borrowed: nearly every such exit has its
        // dispatcher already, and a clone of the board's `Arc` would write
        // a count every PE's exits write.
        let (point, kind, retired, predicted) = {
            let block = &self.code.blocks[index];
            let record = self.code.blocks.record(index);
            let Some(site) = block.dynamic_site.filter(|_| {
                !self.variants[block.variant.0 as usize].single && record.dynamic_open()
            }) else {
                return;
            };
            let Some((kind, fixed)) = dynamic_branch(block.terminal)
                .or_else(|| transitions::transition_branch(block.terminal))
            else {
                return;
            };
            let page = if self.variants[block.variant.0 as usize].guarded_page {
                0b11
            } else {
                0b01
            };
            if fixed.unwrap_or(page) != btype {
                return;
            }
            let retired = u32::from(block.instructions);
            // Bit 11 selects the PAC forms of RET, BR and BLR, whose exit
            // is their thunk's.
            let predicted = matches!(
                dynamic_branch(block.terminal),
                Some((ExitKind::Indirect, _))
            ) && block.terminal & 1 << 11 != 0;
            (chain::dynamic_patch(site.get()), kind, retired, predicted)
        };
        let code = Arc::clone(&self.code);
        let record = code.blocks.record(index);
        // Another PE may serve it meanwhile: checked again under the lock.
        let mut cache = code.cache_at(board::LockSite::Dynamic);
        if !record.dynamic_open() || !record.live() {
            return;
        }
        let word = kind.into();
        if (predicted
            && self.redirect_through_prediction(&mut cache, point, retired, word, btype, target))
            || self.redirect_through_lookup(&mut cache, point, retired, word, btype)
        {
            record.close_dynamic();
        }
    }

    /// Send the exit of `word` whose patch point is at host address `site`,
    /// from a block that ran `retired` instructions by a branch that sets
    /// `btype`, to a [`chain::predicted_tail`] of its own for `target`: its
    /// direct exit to `target` is linked through a cell before the point
    /// jumps to the tail, and any other target takes the tail's lookup, as
    /// [`Self::redirect_through_lookup`] would send it. False, with the
    /// exit as it was, when the tail or the cell link cannot be made.
    fn redirect_through_prediction(
        &mut self,
        cache: &mut board::Cache,
        site: u64,
        retired: u32,
        word: ExitWord,
        btype: u8,
        target: u64,
    ) -> bool {
        if !chain::patchable(site) {
            return false;
        }
        let Some((probe, tail)) =
            self.probe_and_tail(cache, site, btype, chain::PREDICTED_TAIL_BYTES)
        else {
            return false;
        };
        let at = tail.0 as u64;
        let (Some(code), Some(jump)) = (
            chain::predicted_tail(
                at,
                GuestAddress(target),
                word,
                retired,
                probe + chain::PROBE_ENTRY as u64,
            ),
            chain::jump(site, at),
        ) else {
            return false;
        };
        // Nothing reaches the tail before the point's jump: its direct exit
        // is linked first, so no predicted transfer returns to the engine.
        self.patch_native(tail, &code);
        let exit = chain::direct_patch(at + chain::PREDICTED_EXIT as u64);
        if !self.link_through_cell(cache, exit, target, retired, word, btype) {
            return false;
        }
        self.patch_point(NativeCode(site as usize), jump);
        self.statistics.lookups += 1;
        self.statistics.predicted_tails += 1;
        true
    }

    /// Send the exit of `word` whose patch point is at host address `site`
    /// ([`chain::dynamic_patch`]), from a block that ran `retired`
    /// instructions by a branch that sets `btype`, to a tail of its own that
    /// calls the table probe of `btype` ([`chain::lookup_tail`]), written
    /// before the point's jump to it: the jump into each found block is the
    /// site's, so the host predicts a site's targets apart from every other
    /// site's. The probe validates every transfer against the ITLB, whose
    /// entries a TLBI or a context change forgets, and the block table,
    /// whose entries a release of the block forgets, so a return that
    /// unwinds past its caller (longjmp, an exception) or returns to a
    /// poisoned PAC address misses and exits.
    pub(super) fn redirect_through_lookup(
        &mut self,
        cache: &mut board::Cache,
        site: u64,
        retired: u32,
        word: ExitWord,
        btype: u8,
    ) -> bool {
        if !chain::patchable(site) {
            return false;
        }
        let Some((probe, tail)) = self.probe_and_tail(cache, site, btype, chain::TAIL_BYTES) else {
            return false;
        };
        let at = tail.0 as u64;
        let (Some(code), Some(jump)) = (
            chain::lookup_tail(at, word, retired, probe + chain::PROBE_ENTRY as u64),
            chain::jump(site, at),
        ) else {
            return false;
        };
        self.patch_native(tail, &code);
        self.patch_point(NativeCode(site as usize), jump);
        self.statistics.lookups += 1;
        true
    }

    /// Link the direct exit of `word` whose patch point is at host address
    /// `site` ([`chain::direct_patch`]), from a block that ran
    /// `retired` instructions, to `next_pc` in another page, through a cell
    /// of the lookup table ([`chain::LookupTable::link_cell`]) and a tail of
    /// its own ([`chain::cell_tail`]). The cell holds the entry of `next_pc`
    /// the table holds for the key its page translates to now, and takes the
    /// next one the table takes: a transfer through it translates the
    /// target's page through the ITLB and compares the target and the key,
    /// as a lookup does, without hashing or probing the table, and its jump
    /// is the site's own. A cell that does not hold the target with that key
    /// sends the transfer to the table probe. The cell goes with the exit's
    /// code ([`Self::evict_code`]), and its entry with the block's.
    ///
    /// The direct exit of a [`chain::predicted_tail`] of a branch that sets
    /// a nonzero `btype` uses the probes of `btype`, whose cell probes take
    /// only entries that accept it, and gets a stamped tail only for a block
    /// that accepts it.
    pub(super) fn link_through_cell(
        &mut self,
        cache: &mut board::Cache,
        site: u64,
        next_pc: u64,
        retired: u32,
        word: ExitWord,
        btype: u8,
    ) -> bool {
        if !chain::patchable(site) {
            return false;
        }
        // An exit whose cell probe transfers made it hot gets a stamped
        // tail ([`chain::stamped_tail`]); every other a cell tail.
        let hot = cache
            .targets
            .site_cell_number(site)
            .is_some_and(|cell| cache.targets.cell_heat(cell) >= chain::HOT_CELL);
        let bytes = if hot {
            chain::STAMPED_TAIL_BYTES
        } else {
            chain::CELL_TAIL_BYTES
        };
        let Some((probe, tail)) = self.probe_and_tail(cache, site, btype, bytes) else {
            return false;
        };
        let current = self.current_key(next_pc).and_then(|key| {
            let held = cache
                .targets
                .find(cache.targets.slot(next_pc, key), next_pc, key);
            (held.pc == next_pc && held.key == key).then_some(held)
        });
        let cell = cache.targets.link_cell(site, next_pc, current);
        let number = cache.targets.site_cell_number(site);
        if let Some(number) = number {
            let link = chain::CellLink {
                site,
                pc: next_pc,
                retired,
                word,
                btype,
            };
            cache.targets.note_link(number, link);
        }
        let at = tail.0 as u64;
        let entry = probe + chain::CELL_PROBE_ENTRY as u64;
        let stamped = hot
            .then(|| {
                let (current, number) = (current?, number?);
                if btype != 0 && current.targets & 1 << btype == 0 {
                    return None;
                }
                chain::stamped_branch_tail(
                    at,
                    cell,
                    number,
                    next_pc,
                    word,
                    retired,
                    current.block,
                    current.entry,
                    probe + chain::STAMPING_PROBE_ENTRY as u64,
                    btype,
                )
                .map(|code| (code, number))
            })
            .flatten();
        let stamped_cell = stamped.map(|(_, number)| number);
        let code = match stamped {
            Some((code, _)) => Some(code.to_vec()),
            None => chain::cell_tail(at, cell, word, retired, entry).map(|code| code.to_vec()),
        };
        let (Some(code), Some(jump)) = (code, chain::jump(site, at)) else {
            cache.targets.release_cells(site..=site);
            return false;
        };
        if let Some(number) = stamped_cell {
            cache.targets.note_stamped(number);
            self.statistics.stamped_tails += 1;
        }
        self.patch_native(tail, &code);
        self.patch_point(NativeCode(site as usize), jump);
        self.statistics.links += 1;
        true
    }

    /// Give the exits of the next [`PROMOTION_CELLS`] stamped cells whose
    /// cell probe transfers made them hot a stamped tail
    /// ([`chain::stamped_tail`]): a bounded step every [`PROMOTION_RUNS`]
    /// native runs, sweeping every stamped cell in turn.
    pub(super) fn promote_hot_exits(&mut self) {
        let from = self.promotion_cursor;
        let to = from + PROMOTION_CELLS;
        self.promotion_cursor = if to as usize >= chain::STAMPED_CELLS {
            0
        } else {
            to
        };
        let code = Arc::clone(&self.code);
        let mut cache = code.cache_at(board::LockSite::Link);
        for link in cache.targets.hot_links(from..to) {
            if let Some(cell) = cache.targets.site_cell_number(link.site) {
                // Hot once: tried now whatever the link finds.
                cache.targets.settle_heat(cell);
            }
            self.link_through_cell(
                &mut cache,
                link.site,
                link.pc,
                link.retired,
                link.word,
                link.btype,
            );
        }
        self.statistics.stamped_in_use = cache.targets.stamped_tails() as u64;
    }

    /// The probes of `btype` and room for a tail of `bytes` that calls
    /// them, in one part: a tail is taken away with its part, never after
    /// its probes. In the board's arena the probes are installed once per
    /// part, under the board's lock (`cache`); a tail that met a new part
    /// installs them there first.
    fn probe_and_tail(
        &mut self,
        cache: &mut board::Cache,
        site: u64,
        btype: u8,
        bytes: usize,
    ) -> Option<(u64, NativeCode)> {
        let code = Arc::clone(&self.code);
        let Some(arena) = &code.arena else {
            let probe = self.lookup_dispatcher(btype)?;
            let tail = self
                .host
                .install(&[0; chain::STAMPED_TAIL_BYTES][..bytes])
                .ok()?;
            return Some((probe, tail));
        };
        // In the generation of the exit's block, which the tail outlives
        // there ([`board::Ring::refer`]).
        let stream = self.stream_at(site);
        self.install_stream = stream;
        let tail = self
            .install_native(&[0; chain::STAMPED_TAIL_BYTES][..bytes], false)
            .ok()?;
        // The tail's part was chosen meanwhile: the exit stays as it is.
        if !self.note_reference(site, tail.0 as u64) {
            return None;
        }
        let part_of = |address: usize| ((address - arena.window.exec) / arena.part_bytes) as u32;
        let part = part_of(tail.0);
        let at = match cache.dispatchers.iter().position(|&(held, _)| held == part) {
            Some(at) => at,
            None => {
                cache.dispatchers.push((part, [None; 4]));
                cache.dispatchers.len() - 1
            }
        };
        if let Some(probe) = cache.dispatchers[at].1[usize::from(btype)] {
            return Some((probe, tail));
        }
        self.install_stream = stream;
        let probe = self
            .install_native(&chain::lookup_dispatcher(btype)?, false)
            .ok()?;
        // The part filled between the two: the tail waits for the next.
        if part_of(probe.0) != part {
            return None;
        }
        cache.dispatchers[at].1[usize::from(btype)] = Some(probe.0 as u64);
        Some((probe.0 as u64, tail))
    }

    /// The probes of `btype` ([`chain::lookup_dispatcher`]), in the part of
    /// the arena new code fills: like a thunk, they stay until after every
    /// tail that calls them, as parts go oldest first, every tail is in the
    /// part of its probes, and the exits that jump to a tail are in that part
    /// or older ones.
    fn lookup_dispatcher(&mut self, btype: u8) -> Option<u64> {
        if let Some(code) = self.lookup_dispatchers[usize::from(btype)] {
            return Some(code.0 as u64);
        }
        let code = chain::lookup_dispatcher(btype)?;
        let installed = self.host.install(&code).ok()?;
        self.lookup_dispatchers[usize::from(btype)] = Some(installed);
        Some(installed.0 as u64)
    }
}
