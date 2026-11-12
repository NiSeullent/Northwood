//! Instruction fetch: lookup contexts, translation, branch-target and FP access
//! checks.

use super::*;

/// Which BTYPEs may land on this instruction. PAC signing instructions use
/// the current SCTLR to decide whether they also admit BTYPE 0b11.
#[derive(Clone, Copy)]
#[repr(u8)]
pub(super) enum Landing {
    None = 0b0000,
    Bti = 0b0001,
    Call = 0b0111,
    Jump = 0b1011,
    Any = 0b1111,
    Pac = 0b1_0000,
}

impl Landing {
    pub(super) fn of(word: u32) -> Self {
        const PACIASP: u32 = 0xd503_233f;
        const PACIBSP: u32 = 0xd503_237f;
        if word & 0xffff_ff3f == 0xd503_241f {
            return match (word >> 6) & 3 {
                0b00 => Self::Bti,
                0b01 => Self::Call,
                0b10 => Self::Jump,
                _ => Self::Any,
            };
        }
        if word == PACIASP || word == PACIBSP || block::pauth_lr::signs_with_implicit_bti(word) {
            return Self::Pac;
        }
        if matches!(word & 0xffe0_001f, 0xd420_0000 | 0xd440_0000) {
            Self::Any // BRK and HLT
        } else {
            Self::None
        }
    }

    pub(super) fn targets(self, cpu: &Cpu) -> u32 {
        match self {
            Self::Pac => {
                const SCTLR_BT0: u64 = 1 << 35;
                const SCTLR_BT1: u64 = 1 << 36;
                let forbid = if cpu.pstate.el == 0 {
                    SCTLR_BT0
                } else {
                    SCTLR_BT1
                };
                if cpu.regime_sctlr() & forbid == 0 {
                    Self::Any as u32
                } else {
                    Self::Call as u32
                }
            }
            static_targets => static_targets as u32,
        }
    }
}

impl<H: Host, D: Devices> Machine<H, D> {
    /// Arm `BranchTargetCheck`: the Branch Target exception (EC 0x0D)
    /// reports BTYPE in ISS and leaves it set for SPSR.
    pub(super) fn branch_target_exception(&self, pc: u64, word: u32) -> Option<Exception> {
        let btype = self.cpu.pstate.btype;
        (!self.branch_target_compatible(word))
            .then(|| Exception::synchronous(Class::BranchTarget, u32::from(btype), pc))
    }

    /// An instruction whose FP, SVE, SME and FPMR access checks the engine
    /// applies before it runs: after a Branch Target exception, the first
    /// check that fails traps, or is UNDEFINED (`CheckFPMREnabled`). One
    /// that passes runs on its own: the AdvSIMD FP8 conversions here
    /// ([`Self::fp8_conversion`]), SVE and SME forms through [`scalable`],
    /// and scalar FP in streaming mode as a block that then zeroes the upper
    /// Z bits it wrote.
    #[cold]
    pub(super) fn checked_instruction(&mut self, pc: u64, word: u32) -> Step {
        if self.cpu.pstate.btype != 0
            && self.fetch_guarded(pc)
            && let Some(exception) = self.branch_target_exception(pc, word)
        {
            self.cpu.step_advance = false;
            self.take(exception);
            return Ok(());
        }
        let checks = self.profile.chip.checks(word);
        let exception = match streaming::check(&self.cpu, checks) {
            Err(Denial::FloatingPoint(el)) => {
                Exception::synchronous(Class::FloatingPointAccess, TRAPPED_AARCH64, pc).to(el)
            }
            Err(Denial::Sme { el, smtc }) => {
                Exception::synchronous(Class::SmeAccess, smtc, pc).to(el)
            }
            Err(Denial::Undefined) => return self.undefined(),
            Err(Denial::Unimplemented(feature)) => {
                return Err(self.unsupported(RequirementKind::Instruction {
                    word,
                    detail: format!("its access check needs {feature}, which the engine lacks"),
                }));
            }
            Ok(()) if let Some(operand) = fp8_conversion::decode(word) => {
                return self.fp8_conversion(word, &operand);
            }
            Ok(()) if checks.iter().all(|check| *check == Check::Fp) => {
                return self.streaming_scalar(pc, word);
            }
            Ok(()) => return self.scalable_instruction(pc, word),
        };
        self.cpu.step_advance = false;
        // Arm decodes before it checks access: an SVE or SME word UNDEFINED
        // on this chip takes that instead of the trap.
        if checks.iter().any(|check| *check != Check::Fp) && self.scalable_undefined(pc, word) {
            return self.undefined();
        }
        self.take(exception);
        Ok(())
    }

    /// F1CVTL, F2CVTL, BF1CVTL, BF2CVTL and FCVTN (FP8) and their upper
    /// forms, after their checks passed: the shared FP8 conversion with
    /// FPMR's formats and scales and FPCR, the cumulative exceptions ORed
    /// into FPSR. In streaming mode the V write zeroes the upper Z bits.
    pub(super) fn fp8_conversion(&mut self, word: u32, operand: &fp8_conversion::Operand) -> Step {
        let Some(fpmr) = self.cpu.scalable.as_ref().map(ScalableState::fpmr) else {
            return Err(self.unsupported(RequirementKind::Instruction {
                word,
                detail: "an FP8 conversion on a PE without FPMR".into(),
            }));
        };
        let fpcr = self.cpu.state.fpcr;
        let register = |index: u8| self.cpu.state.v[usize::from(index)];
        let converted = crate::floating::scalable::fixed_fp8_conversion(
            operand,
            register(operand.source),
            register(operand.second_source),
            register(operand.destination),
            fpcr,
            fpmr,
        );
        let Ok((value, exceptions)) = converted else {
            return Err(self.unsupported(RequirementKind::Instruction {
                word,
                detail: format!("FPCR {fpcr:#x} enables a trapped FP exception"),
            }));
        };
        let scalable = self.cpu.scalable.as_mut().expect("checked above");
        if let Err(error) = scalable.write_v(&mut self.cpu.state, operand.destination, value) {
            return Err(self.unsupported(RequirementKind::Instruction {
                word,
                detail: format!("its FP8 result cannot be written: {error}"),
            }));
        }
        self.cpu.state.fpsr |= exceptions;
        self.next()
    }

    /// The number of the context a fetch in `half` runs in now: the EL with
    /// the guarded level, the variant and the half's translation inputs
    /// ([`contexts::Key`]). ITLB entries name it.
    pub(super) fn context(&mut self, el: u8, variant: Variant, half: usize) -> u64 {
        let epoch = self.fetch_epoch(half);
        if let Some(recent) = self.recent[half]
            .iter()
            .flatten()
            .find(|recent| (recent.el, recent.variant, recent.epoch) == (el, variant, epoch))
        {
            return recent.number;
        }
        let (controls, hcr) = self.half_inputs(half, el);
        let key = contexts::Key {
            ttbr0: controls.ttbr0,
            permissions: if el & 3 == 0 {
                controls.sprr.el0
            } else {
                controls.sprr.el1
            },
            frame: self.context_frame(controls, hcr),
            el,
            variant,
        };
        let number = match self.spaces[half].find(&key) {
            Some(number) => number,
            None => self.make_context(half, key, &controls),
        };
        self.salt_clock += 1;
        self.salt_uses[(number % chain::LOOKUP_SALTS as u64) as usize] = self.salt_clock;
        self.recent[half].rotate_right(1);
        self.recent[half][0] = Some(Recent {
            el,
            variant,
            // A frame's table may have filled and forgotten every context.
            epoch: self.fetch_epoch(half),
            number,
        });
        number
    }

    /// A new context of `half` for `key`, whose inputs are `controls`.
    fn make_context(&mut self, half: usize, key: contexts::Key, controls: &mmu::Controls) -> u64 {
        let salt = self.context_salt(half, key.variant, key.ttbr0);
        let made = self.next_context;
        self.next_context += 1;
        let (number, evicted) =
            self.spaces[half].insert(key, mmu::current_asid(controls), controls.regime, |slot| {
                contexts::context_number(made, slot, salt)
            });
        if evicted.is_some() {
            self.statistics.itlb_drops[2] += 1;
            // A transition record may name the context that went.
            self.mirrors.transitions.contexts_moved();
        }
        self.statistics.itlb_sources[ItlbSource::Made as usize] += 1;
        if let Some(pages) = self.pending_refills.remove(&(half, key)) {
            self.refills.extend(pages);
        }
        number
    }

    /// The number of the frame of the inputs `controls` and `hcr` in
    /// [`Self::context_frames`]: all but TTBR0 and the SPRR permissions.
    /// Past [`CONTEXT_FRAMES`] values every context goes with the frames
    /// they name ([`Self::contexts`] then takes both halves' anew).
    fn context_frame(&mut self, mut controls: mmu::Controls, hcr: u64) -> u32 {
        controls.ttbr0 = 0;
        controls.sprr.el0 = 0;
        controls.sprr.el1 = 0;
        let frame = (controls, hcr);
        if let Some(at) = self.context_frames.iter().rposition(|held| *held == frame) {
            return at as u32;
        }
        if self.context_frames.len() == CONTEXT_FRAMES {
            self.forget_spaces(0..=1, 3);
            self.context_frames.clear();
            // Their keys name frames by number.
            self.pending_refills.clear();
        }
        self.context_frames.push(frame);
        (self.context_frames.len() - 1) as u32
    }

    /// The salt of a new context of `half` in `variant` under `table`: the
    /// one its address space's earlier contexts had while no other owner
    /// took it since, so that its ITLB entries take the places of a
    /// forgotten context's; else the least recently used, so that the
    /// address spaces that ran last keep slots apart.
    fn context_salt(&mut self, half: usize, variant: Variant, table: u64) -> u64 {
        let id = u64::from(self.variant_id(variant).0);
        let owner = table ^ (half as u64) ^ id << 6;
        let salt = self.salt_owners.iter().position(|known| *known == owner);
        salt.unwrap_or_else(|| {
            let salt = (0..chain::LOOKUP_SALTS)
                .min_by_key(|&salt| self.salt_uses[salt])
                .expect("salts");
            if self.salt_uses[salt] != 0 {
                self.statistics.itlb_drops[3] += 1;
            }
            self.salt_owners[salt] = owner;
            salt
        }) as u64
    }

    /// Both halves' contexts at `el` in `variant`: of one translation
    /// generation, so that neither names a context the other's frame
    /// forgot.
    pub(super) fn contexts(&mut self, el: u8, variant: Variant) -> [u64; 2] {
        loop {
            let generation = self.translation_generation;
            let contexts = [0, 1].map(|half| self.context(el, variant, half));
            if self.translation_generation == generation {
                return contexts;
            }
        }
    }

    /// What decides how a fetch of `half` at `el` (with the guarded level in
    /// bit 4) translates besides the tables' contents: the controls of the
    /// regime it walks and HCR_EL2, which selects that regime (Arm
    /// `S1TranslationRegime`; E2H is RES1). Of SPRR's permissions a fetch
    /// reads only its own level's register, SPRR_PERM_EL0 at EL0 and
    /// SPRR_PERM_EL1 above, and neither PAN nor the unlock fields
    /// (`mmu::sprr::permissions`), so the kernel's contexts outlive the
    /// SPRR_PERM_EL0 writes XNU makes at every thread switch. The upper
    /// half's leave out TTBR0's table address, which it never walks, and
    /// keep TTBR0's ASID field only while TCR.A1 selects it and a fetch of
    /// the upper half since its contexts were last forgotten was
    /// non-global: the ASID selects nothing else of it, so a kernel's
    /// global text keeps one context across every process's ASID.
    pub(super) fn half_inputs(&self, half: usize, el: u8) -> (mmu::Controls, u64) {
        const TCR_A1: u64 = 1 << 22;
        const TTBR_ASID: u64 = 0xffff << 48;
        use crate::machine::cpu::TranslationRegime;
        let el = el & 3;
        let regime = if el == 2 || el == 0 && self.cpu.in_host() {
            TranslationRegime::El20
        } else {
            TranslationRegime::El10
        };
        let mut controls = self.cpu.translating_controls(regime);
        if el == 0 {
            controls.sprr.el1 = 0;
        } else {
            controls.sprr.el0 = 0;
        }
        controls.sprr.fields = [0; 2];
        if half == 1 {
            controls.ttbr0 &= if self.private_fetches[1] && controls.tcr & TCR_A1 == 0 {
                TTBR_ASID
            } else {
                0
            };
        }
        (controls, self.cpu.read(SystemRegister::HCR_EL2))
    }

    /// The checks before an instruction executes, in Arm's priority after PC
    /// alignment: an Instruction Abort of its fetch, a breakpoint (Arm
    /// `AArch64_CheckBreakpoint`), then the Illegal Execution state (Arm
    /// `AArch64_CheckIllegalState`).
    #[cold]
    pub(super) fn instruction_checks(
        &mut self,
        pc: u64,
        breakpoints: bool,
    ) -> Result<Option<Exception>, Stop> {
        match self.fetch_physical(pc) {
            Ok(_) => {}
            Err(Ok(exception)) => return Ok(Some(exception)),
            Err(Err(stop)) => return Err(stop),
        }
        if breakpoints && debug::breakpoint(&self.cpu, pc) {
            return Ok(Some(self.debug_exception(
                Class::BreakpointLower,
                Class::BreakpointSame,
                0x22,
                pc,
            )));
        }
        Ok(self
            .cpu
            .pstate
            .il
            .then(|| Exception::synchronous(Class::IllegalExecution, 0, pc)))
    }

    /// Translate a PC for execution at the current EL.
    pub(super) fn fetch_physical(&mut self, pc: u64) -> Result<u64, Result<Exception, Stop>> {
        let page = pc & !(FETCH_PAGE - 1);
        let slot = (page / FETCH_PAGE) as usize % FETCH_CACHE;
        // The cache key folds the guarded level in with the EL.
        let el = self.cpu.pstate.el | u8::from(self.cpu.pstate.guarded) << 4;
        if let Some(entry) = self.fetch[slot]
            && entry.page == page
            && entry.el == el
            && entry.epoch == self.fetch_epoch(chain::half(pc))
        {
            return Ok(entry.physical | (pc & (FETCH_PAGE - 1)));
        }
        let requester = self.cpu.requester();
        let controls = self.cpu.controls(self.cpu.regime());
        let mut tables = RamTables(&mut self.host);
        match mmu::translate(&controls, pc, Access::Execute, requester, &mut tables) {
            Translated::Mapped(_) if !self.host.fetch_permitted(pc) => {
                let fault =
                    mmu::Fault::stage1(mmu::FaultKind::Permission, HOST_REFUSED_FETCH_LEVEL);
                Err(Ok(self.abort(
                    fault,
                    [Class::InstructionAbortLower, Class::InstructionAbortSame],
                    fault.instruction_syndrome(),
                    pc,
                    pc,
                )))
            }
            Translated::Mapped(translation) => {
                let half = chain::half(pc);
                if half == 0 && translation.global && self.widen_lower_globals(pc, translation.size)
                {
                    return self.fetch_physical(pc);
                }
                self.fetched_through(half, pc, translation.size, translation.global);
                let physical = translation.physical & !(FETCH_PAGE - 1);
                self.fetch[slot] = Some(FetchEntry {
                    page,
                    physical,
                    guarded: translation.guarded,
                    global: translation.global,
                    el,
                    epoch: self.fetch_epoch(half),
                });
                Ok(translation.physical)
            }
            Translated::Fault(fault)
                if fault.kind == mmu::FaultKind::Permission
                    && requester.el != 0
                    && !requester.guarded
                    && controls.sprr.enabled()
                    && matches!(
                        mmu::translate(
                            &controls,
                            pc,
                            Access::Execute,
                            mmu::Requester {
                                guarded: true,
                                ..requester
                            },
                            &mut RamTables(&mut self.host),
                        ),
                        Translated::Mapped(_)
                    ) =>
            {
                // On hardware an EL fetch of GL-only code enters GL at
                // GXF_ABORT_EL1 (as reported for GL2); the saved state is
                // not documented.
                Err(Err(self.unsupported(RequirementKind::Native(
                    "kernel-level fetch of guarded-only code (GXF abort entry)",
                ))))
            }
            Translated::Fault(fault) => Err(Ok(self.abort(
                fault,
                [Class::InstructionAbortLower, Class::InstructionAbortSame],
                fault.instruction_syndrome(),
                pc,
                pc,
            ))),
            Translated::Unsupported(unsupported) => Err(Err(
                self.unsupported(RequirementKind::Translation(unsupported))
            )),
        }
    }

    /// The page frame of `pc` for execution at the current EL and whether
    /// it is guarded, when its walk translates and the host permits the
    /// fetch: [`Self::fetch_physical`] without a fault or report, for a walk
    /// no instruction asked for ([`Self::refill_pages`]). The walker makes
    /// no descriptor updates (no FEAT_HAFDBS), so the walk has no effect on
    /// memory. A global lower page that widens what the PE may hold is
    /// left to its fetch, which walks again.
    pub(super) fn quiet_fetch_physical(&mut self, pc: u64) -> Option<(u64, bool, bool)> {
        let requester = self.cpu.requester();
        let controls = self.cpu.controls(self.cpu.regime());
        let mut tables = RamTables(&mut self.host);
        let Translated::Mapped(translation) =
            mmu::translate(&controls, pc, Access::Execute, requester, &mut tables)
        else {
            return None;
        };
        if !self.host.fetch_permitted(pc) {
            return None;
        }
        let half = chain::half(pc);
        if half == 0 && translation.global && self.widen_lower_globals(pc, translation.size) {
            return None;
        }
        self.fetched_through(half, pc, translation.size, translation.global);
        Some((
            translation.physical & !(FETCH_PAGE - 1),
            translation.guarded,
            translation.global,
        ))
    }

    /// Whether the walk of the page of `pc` that the fetch cache holds for
    /// the current EL ended in a global leaf; true when the cache does not
    /// hold it, which only makes more TLBIs reach the page's ITLB entry.
    pub(super) fn fetched_global(&self, pc: u64) -> bool {
        let page = pc & !(FETCH_PAGE - 1);
        let slot = (page / FETCH_PAGE) as usize % FETCH_CACHE;
        let el = self.cpu.pstate.el | u8::from(self.cpu.pstate.guarded) << 4;
        self.fetch[slot].is_none_or(|entry| {
            entry.page != page
                || entry.el != el
                || entry.epoch != self.fetch_epoch(chain::half(pc))
                || entry.global
        })
    }

    /// Whether the page of `pc`, which executes at the current EL, is guarded
    /// (Arm `InGuardedPage`).
    pub(super) fn fetch_guarded(&mut self, pc: u64) -> bool {
        if self.fetch_physical(pc).is_err() {
            return false;
        }
        let page = pc & !(FETCH_PAGE - 1);
        let slot = (page / FETCH_PAGE) as usize % FETCH_CACHE;
        self.fetch[slot].is_some_and(|entry| entry.page == page && entry.guarded)
    }

    /// FEAT_BTI's BTYPE after an indirect branch at `pc` through `register`
    /// (Arm BR, BLR, BRA* and BLRA*): BLR forms give 0b10; BR forms give
    /// 0b01 through X16 or X17 or from an unguarded page, else 0b11.
    pub(super) fn indirect_branch_type(&mut self, pc: u64, register: u8, link: bool) -> u8 {
        if link {
            0b10
        } else if register == 16 || register == 17 || !self.fetch_guarded(pc) {
            0b01
        } else {
            0b11
        }
    }

    /// Whether the first instruction of a guarded-page target is compatible
    /// with PSTATE.BTYPE (Arm `BTypeCompatible_BTI`, `BTypeCompatible_PAC`):
    /// BTI with a matching target, PACIASP and PACIBSP unless BTYPE is 0b11
    /// and SCTLR_ELx.BT0 (EL0) or BT1 forbids it, BRK and HLT.
    pub(super) fn branch_target_compatible(&self, word: u32) -> bool {
        self.branch_target_accepts(word, self.cpu.pstate.btype)
    }

    /// [`Self::branch_target_compatible`] for BTYPE `btype`.
    pub(super) fn branch_target_accepts(&self, word: u32, btype: u8) -> bool {
        Landing::of(word).targets(&self.cpu) & (1 << btype) != 0
    }

    /// The EL an FP/SIMD access traps to, if it traps (Arm
    /// `AArch64_CheckFPEnabled`): CPACR_EL1.FPEN at EL0 and EL1 outside the
    /// host, then CPTR_EL2.FPEN, which has the CPACR layout because
    /// HCR_EL2.E2H is RES1.
    pub(super) fn floating_point_trap(&mut self) -> Option<u8> {
        *self
            .floating_point
            .get_or_insert_with(|| self.cpu.floating_point_trap())
    }

    pub(super) fn floating_point_enabled(&mut self) -> bool {
        self.floating_point_trap().is_none()
    }

    pub(super) fn floating_point_exception(&mut self, pc: u64) -> Exception {
        let exception = Exception::synchronous(Class::FloatingPointAccess, TRAPPED_AARCH64, pc);
        match self.floating_point_trap() {
            Some(el) => exception.to(el),
            None => exception,
        }
    }

    /// BTYPE after a block that a native branch or fallthrough ended: BR and
    /// BLR set it, every other instruction clears it.
    pub(super) fn native_branch_type(
        &mut self,
        index: usize,
        kind: Result<ExitKind, stela_compiler::block::InvalidExitKind>,
    ) -> Result<u8, Stop> {
        const REGISTER_BRANCH: u32 = 0xffff_fc1f;
        const BR: u32 = 0xd61f_0000;
        const BLR: u32 = 0xd63f_0000;
        if !matches!(kind, Ok(ExitKind::Indirect | ExitKind::Call)) {
            return Ok(0);
        }
        let terminal = self.code.blocks[index].terminal_pc();
        let word = self.word(index, terminal)?;
        let register = ((word >> 5) & 31) as u8;
        Ok(match word & REGISTER_BRANCH {
            BR => self.indirect_branch_type(terminal, register, false),
            BLR => self.indirect_branch_type(terminal, register, true),
            // BRA* and BLRA* that a PAC thunk completed.
            _ => match block::decode_pointer_authentication(GuestAddress(terminal), word) {
                Ok(Some(instruction)) => self.pointer_branch_type(terminal, instruction.operation),
                _ => 0,
            },
        })
    }

    /// Both FP service boundaries share the same sole semantic completion.
    /// An error is an enabled FP exception the instruction raised, which
    /// [`Self::take_floating_point_exception`] takes.
    #[inline]
    pub(super) fn complete_floating_point(
        &mut self,
        word: u32,
        operation: &FloatingPointOperation,
    ) -> Result<(), crate::floating::Trapped> {
        if self.step_timing.is_none() {
            return self.complete_floating_point_untimed(word, operation);
        }
        // Under step timing, what the completion found
        // (`step_timing::FloatingReason`): its fresh cumulative flags, with
        // FPSR's earlier ones put back after it, so FPSR ends as it would.
        const CUMULATIVE: u32 = 0x9f;
        let fpcr = self.cpu.state.fpcr;
        let before = self.cpu.state.fpsr;
        self.cpu.state.fpsr = before & !CUMULATIVE;
        let completed = self.complete_floating_point_untimed(word, operation);
        let fresh = self.cpu.state.fpsr & CUMULATIVE;
        self.cpu.state.fpsr |= before;
        let reason = super::step_timing::FloatingReason::of(fpcr, fresh, completed.is_err());
        self.system_exits.floating[reason as usize] += 1;
        if super::step_timing::FloatingReason::moded(fpcr) {
            self.system_exits.floating[super::step_timing::FloatingReason::Mode as usize] += 1;
        }
        completed
    }

    fn complete_floating_point_untimed(
        &mut self,
        word: u32,
        operation: &FloatingPointOperation,
    ) -> Result<(), crate::floating::Trapped> {
        #[cfg(all(feature = "std", feature = "native-profile"))]
        if let Some(profile) = &mut self.floating_profile
            && profile.tick()
        {
            return profile.complete_at(&mut self.cpu, word, operation);
        }
        #[cfg(not(all(feature = "std", feature = "native-profile")))]
        let _ = word;
        crate::floating::execute(&mut self.cpu.state, operation)
    }

    /// Take the TrappedFP exception of the FP instruction at the PC (Arm
    /// `AArch64.FPTrappedException`): TFV set, with the trapped exceptions
    /// in IOF, DZF, OFF, UFF, IXF and IDF, to the current EL above EL1, to
    /// EL2 under HCR_EL2.TGE, else to EL1. The syndrome is TFV's for
    /// scalar and vector instructions alike, which Arm permits.
    pub(super) fn take_floating_point_exception(
        &mut self,
        trapped: crate::floating::Trapped,
    ) -> Step {
        const TFV: u32 = 1 << 23;
        self.take(Exception::synchronous(
            Class::TrappedFloatingPoint,
            TFV | trapped.exceptions,
            self.cpu.pc,
        ));
        Ok(())
    }
}
