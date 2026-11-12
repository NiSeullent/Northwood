//! System instructions, exception entry and return, and System register
//! accesses.

use super::*;

/// A PE often completes the same architectural instruction words through
/// many different compiled blocks. Cache decoding only; the operation still
/// runs with the current registers, privilege and feature checks.
pub(super) struct DecodeCache([Option<(u32, SystemOperation)>; Self::SLOTS]);

impl Default for DecodeCache {
    fn default() -> Self {
        // Copy the immutable empty table rather than building a temporary
        // on each worker's stack.
        const { Self([None; Self::SLOTS]) }
    }
}

impl DecodeCache {
    const SLOTS: usize = 512;

    fn operation(&mut self, pc: u64, word: u32) -> Option<SystemOperation> {
        let slot = (word.wrapping_mul(0x9e37_79b9) >> (u32::BITS - Self::SLOTS.ilog2())) as usize;
        let entry = &mut self.0[slot];
        if let Some((cached_word, operation)) = *entry
            && cached_word == word
            // Keep the compiler's alignment and checked fallthrough rules
            // even when an earlier address supplied the same word.
            && pc.is_multiple_of(INSTRUCTION)
            && pc <= u64::MAX - INSTRUCTION
        {
            return Some(operation);
        }
        let decoded = match block::decode_system_mode_instruction(GuestAddress(pc), word) {
            Ok(Some(instruction)) => instruction.operation,
            // Compiled code completes MRS and MSR of NZCV inline
            // (`aarch64::flags`); the engine still completes one it is given.
            Ok(None) if stela_compiler::aarch64::flags::accesses_nzcv(word) => {
                SystemOperation::Register(decode_register(word)?)
            }
            _ => return None,
        };
        *entry = Some((word, decoded));
        Some(decoded)
    }
}

impl<H: Host, D: Devices> Machine<H, D> {
    pub(super) fn system_instruction(&mut self, word: u32) -> Step {
        let operation = self.decode_system_operation(word);
        self.complete_system_instruction(word, operation)
    }

    pub(super) fn decode_system_operation(&mut self, word: u32) -> Option<SystemOperation> {
        self.system_decode.operation(self.cpu.pc, word)
    }

    pub(super) fn complete_system_instruction(
        &mut self,
        word: u32,
        operation: Option<SystemOperation>,
    ) -> Step {
        let Some(operation) = operation else {
            return Err(self.refuse_system(word, "System exit at a word that is not System"));
        };
        // Apple's SDSB (`0x00201460` plus the domain), WKDMC and WKDMD;
        // `guarded` checks the GXF instructions.
        let apple = match operation {
            SystemOperation::Synchronization(_) if word & !3 == 0x0020_1460 => {
                Some(hacr::Operation::SynchronizationBarrier)
            }
            SystemOperation::Compression(_) => Some(hacr::Operation::Compression),
            _ => None,
        };
        if let Some(operation) = apple
            && let Some(step) = self.hacr_guest(word, operation)
        {
            return step;
        }
        match operation {
            SystemOperation::Exception(operation) => self.exception_instruction(operation),
            SystemOperation::Register(access) => self.register_instruction(word, access),
            SystemOperation::PstateImmediate(encoding) => {
                let before = self.cpu.pstate;
                let outcome = self.cpu.write_pstate_field(encoding);
                self.pstate_immediate(word, before, outcome)
            }
            SystemOperation::Maintenance(access) => self.maintenance(word, access),
            SystemOperation::InstructionSynchronization { .. } => self.next(),
            SystemOperation::Synchronization(operation) => match operation {
                // DSB nXS waits for a subset of what DSB waits for (accesses
                // to memory with XS 0), and is UNDEFINED without FEAT_XS.
                Synchronization::DataCompletion { nxs: true, .. } => {
                    match self.cpu.registers.implemented(Feature::FEAT_XS) {
                        Ok(true) => {
                            data_barrier();
                            self.complete_maintenance()?;
                            self.next()
                        }
                        Ok(false) => self.undefined(),
                        Err(error) => {
                            Err(self.unsupported(RequirementKind::Native(error.reason())))
                        }
                    }
                }
                Synchronization::ErrorSynchronization => self.error_synchronization(),
                // The TLB and instruction cache maintenance this PE posted
                // completes for the other PEs (`parallel` `post`).
                Synchronization::DataCompletion { .. } => {
                    data_barrier();
                    self.complete_maintenance()?;
                    self.open_barrier();
                    self.next()
                }
                Synchronization::StoreBypassBarrier { .. }
                | Synchronization::BranchTarget { .. }
                | Synchronization::TraceSynchronization
                | Synchronization::OtherHint { .. }
                | Synchronization::Yield
                | Synchronization::ConsumptionBarrier
                | Synchronization::SpeculationBarrier => {
                    // The engine completes each instruction before the
                    // next, and x86 keeps the PE's own accesses in program
                    // order for the others (TSO) apart from store to load,
                    // which only DSB and DMB order: these have no further
                    // effect.
                    self.next()
                }
                Synchronization::SendEvent { .. } | Synchronization::Wait { .. } => {
                    Err(self.refuse_system(word, "unexpected synchronization family"))
                }
                Synchronization::InstructionSynchronization { .. } => self.next(),
            },
            SystemOperation::SendEvent { local } => {
                self.send_event(local);
                self.next()
            }
            SystemOperation::Wait {
                kind,
                timeout_register,
            } => {
                self.adopt_native_monitor();
                self.wait(word, kind, timeout_register)
            }
            SystemOperation::TagMemory(operand) => self.tag_memory(word, operand),
            SystemOperation::TagOffset(operand) => self.tag_offset(word, operand),
            // No halting debug: Debug-state-only instructions are UNDEFINED.
            SystemOperation::DebugStateOnly(_) => self.undefined(),
            SystemOperation::Unprivileged(operand) => self.unprivileged(operand),
            SystemOperation::Exclusive(Ok(operand)) => self.exclusive(operand),
            SystemOperation::Exclusive(Err(_)) => self.undefined(),
            SystemOperation::CheckedPointer(operand) => self.checked_pointer(operand),
            SystemOperation::Compression(instruction) => self.compression(word, instruction),
            SystemOperation::Guarded(operation) => self.guarded(word, operation),
            SystemOperation::AppleMaintenance(operation) => self.apple_maintenance(word, operation),
        }
    }

    /// An ERET or GEXIT to `spsr`, guarded when `guarded`, that enters an
    /// EL1&0 guest of the EL2 host (EL1 or EL0, unguarded, HCR_EL2.TGE
    /// clear) while an Apple VM register holds state this machine does not
    /// apply to a guest (`impdef::Registers::guest_state`): SPTM and the
    /// kernel's hypervisor load a VM's values into them before it runs
    /// (`machine::impdef` `GUEST_EVIDENCE`). The guest is entered without
    /// that state, reported once.
    pub(super) fn guest_entry(&mut self, spsr: u64, guarded: bool) {
        let enters_guest = !guarded
            && !self.cpu.in_host()
            && super::super::exception::spsr_target(spsr).is_ok_and(|(el, _)| el < 2);
        if enters_guest && let Some(entry) = self.cpu.apple.impdef.guest_state() {
            self.degrade(self.requirement(RequirementKind::VmState(entry.name)));
        }
    }

    pub(super) fn exception_instruction(&mut self, operation: ExceptionOperation) -> Step {
        let pc = self.cpu.pc;
        match operation {
            // HFGITR_EL2.SVC_EL1 and SVC_EL0 trap SVC from EL1, and from EL0
            // outside the host, to EL2 at the SVC itself (Arm
            // `AArch64_CheckForSVCTrap`).
            ExceptionOperation::SupervisorCall { immediate }
                if self.cpu.pstate.el != 2
                    && self.fine_grained_trap(if self.cpu.pstate.el == 1 {
                        HFGITR_SVC_EL1
                    } else {
                        HFGITR_SVC_EL0
                    }) =>
            {
                self.take(
                    Exception::synchronous(Class::SupervisorCall, u32::from(immediate), pc).to(2),
                );
                Ok(())
            }
            // To EL1, or to EL2 from EL2 and from EL0 in the host.
            ExceptionOperation::SupervisorCall { immediate } => {
                self.advance_step();
                self.note_transition_miss(transition::SUPERVISOR_CALL);
                let start = self.transition_start();
                self.take(Exception::synchronous(
                    Class::SupervisorCall,
                    u32::from(immediate),
                    pc + INSTRUCTION,
                ));
                self.note_transition(transition::SUPERVISOR_CALL, start, 0, 0);
                Ok(())
            }
            // BRK: to EL2 from EL2, or under HCR_EL2.TGE or MDCR_EL2.TDE
            // (Arm `AArch64_SoftwareBreakpoint`).
            ExceptionOperation::SoftwareBreakpoint { immediate } => {
                let exception = self.debug_exception(
                    Class::Breakpoint,
                    Class::Breakpoint,
                    u32::from(immediate),
                    pc,
                );
                self.take(exception);
                Ok(())
            }
            ExceptionOperation::ExceptionReturn { key } => self.exception_return(key),
            // HVC from EL1 and EL2 goes to EL2, unless HCR_EL2.HCD disables it.
            ExceptionOperation::HypervisorCall { immediate }
                if self.cpu.pstate.el != 0 && !self.cpu.hcr(HCR_HCD) =>
            {
                self.advance_step();
                self.take(
                    Exception::synchronous(
                        Class::HypervisorCall,
                        u32::from(immediate),
                        pc + INSTRUCTION,
                    )
                    .to(2),
                );
                Ok(())
            }
            // Without EL3, SMC is UNDEFINED, except that HCR_EL2.TSC traps it
            // from EL1 to EL2, returning to the SMC itself (Arm
            // `AArch64_CheckForSMCUndefOrTrap`).
            ExceptionOperation::SecureMonitorCall { immediate }
                if self.cpu.pstate.el == 1 && self.cpu.hcr(HCR_TSC) =>
            {
                self.take(
                    Exception::synchronous(Class::MonitorCall, u32::from(immediate), pc).to(2),
                );
                Ok(())
            }
            // No EL3, debug state or halting, and HVC from EL0 or with HCD:
            // all UNDEFINED.
            ExceptionOperation::HypervisorCall { .. }
            | ExceptionOperation::SecureMonitorCall { .. }
            | ExceptionOperation::Halt { .. }
            | ExceptionOperation::DebugChange { .. }
            | ExceptionOperation::DebugReturn
            | ExceptionOperation::Undefined { .. } => self.undefined(),
        }
    }

    /// Whether HFGITR_EL2 bit `bit` traps an instruction from EL1, or from
    /// EL0 outside the host (FEAT_FGT without EL3, so always enabled).
    pub(super) fn fine_grained_trap(&self, bit: u32) -> bool {
        let el = self.cpu.pstate.el;
        (el == 1 || (el == 0 && !self.cpu.in_host()))
            && self.cpu.read(SystemRegister::HFGITR_EL2) & (1 << bit) != 0
    }

    /// ERET, ERETAA and ERETAB (Arm `AArch64_CheckForERetTrap`, `AuthIA`,
    /// `AuthIB` and `AArch64_ExceptionReturn`).
    pub(super) fn exception_return(&mut self, key: Option<AuthenticationKey>) -> Step {
        let pc = self.cpu.pc;
        if self.cpu.pstate.el == 0 {
            return self.undefined();
        }
        // HFGITR_EL2.ERET, or HCR_EL2.NV in effect (no FEAT_NV3), traps an
        // EL1 ERET to EL2 (Arm `AArch64_CheckForERetTrap`).
        if self.fine_grained_trap(HFGITR_ERET)
            || self.cpu.pstate.el == 1 && self.cpu.nested() & 1 != 0
        {
            let iss = match key {
                None => 0,
                Some(AuthenticationKey::A) => 0b10,
                Some(AuthenticationKey::B) => 0b11,
            };
            self.take(Exception::synchronous(Class::ExceptionReturnTrap, iss, pc).to(2));
            return Ok(());
        }
        let mut target = self.cpu.exception_link();
        if let Some(key) = key {
            let key = match key {
                AuthenticationKey::A => PointerAuthenticationKey::Ia,
                AuthenticationKey::B => PointerAuthenticationKey::Ib,
            };
            if self.key_enabled(key) {
                if self.pointer_authentication_trapped() {
                    return self.pointer_authentication_trap();
                }
                let runtime_key = match key {
                    PointerAuthenticationKey::Ia => stela_contracts::pauth::Key::Ia,
                    _ => stela_contracts::pauth::Key::Ib,
                };
                // A combined authentication: a failure corrupts the pointer,
                // or with FEAT_FPACCOMBINE takes the PAC Fail exception at
                // the ERETA* (Arm `AuthIA`, `Authenticate`).
                match self.pointer_context()?.authenticate_operation(
                    runtime_key,
                    target,
                    self.cpu.state.sp,
                    true,
                ) {
                    Ok(result) => target = result.pointer,
                    Err(pauth::AuthenticationError::Failure { code }) => {
                        self.take(Exception::synchronous(
                            Class::PointerAuthenticationFailure,
                            u32::from(code),
                            pc,
                        ));
                        return Ok(());
                    }
                    Err(_) => {
                        return Err(self.unsupported(RequirementKind::Native(
                            "ERETAA/ERETAB authentication without its key",
                        )));
                    }
                }
            }
        }
        self.guest_entry(self.cpu.exception_state(), self.cpu.pstate.guarded);
        let trace = self.monitor_before(MonitorKind::ExceptionReturn);
        if key.is_none() {
            self.note_transition_miss(transition::EXCEPTION_RETURN);
        }
        let start = self.transition_start();
        let spsr = self.cpu.exception_state();
        self.cpu.exception_return(target).map_err(|_| {
            self.unsupported(RequirementKind::Native(
                "canonical exclusive/event settlement failed before exception return",
            ))
        })?;
        self.pacm_written = true;
        self.host.report(&Event::Return {
            to_el: self.cpu.pstate.el,
            guarded: self.cpu.pstate.guarded,
            pc: self.cpu.pc,
        });
        self.monitor_after(trace);
        self.transitioned();
        if key.is_none() {
            self.note_transition(transition::EXCEPTION_RETURN, start, spsr, 0);
        }
        Ok(())
    }

    /// ESB: at EL0 or EL1 outside the host with HCR_EL2.AMO set, a virtual
    /// SError pending but masked by PSTATE.A is deferred into VDISR_EL2 (Arm
    /// `AArch64_vESBOperation`). No physical SError is ever pending.
    pub(super) fn error_synchronization(&mut self) -> Step {
        let hcr = self.cpu.read(SystemRegister::HCR_EL2);
        let guest = self.cpu.pstate.el != 2 && hcr & (1 << HCR_TGE) == 0;
        if guest
            && hcr & (1 << HCR_AMO) != 0
            && hcr & (1 << HCR_VSE) != 0
            && self.cpu.pstate.daif & DAIF_A != 0
        {
            let syndrome = self.cpu.read(SystemRegister::VSESR_EL2) & 0x01ff_ffff;
            self.cpu
                .registers
                .set(SystemRegister::VDISR_EL2, 0, 1 << 31 | syndrome);
            self.cpu
                .registers
                .set(SystemRegister::HCR_EL2, 0, hcr & !(1 << HCR_VSE));
        }
        self.next()
    }

    /// MRS and MSR (`register_access`); a read of a [`MIRRORED`] register
    /// that completes as a plain read, taking no exception, opens its cell
    /// with the value read.
    pub(super) fn register_instruction(&mut self, word: u32, access: RegisterAccess) -> Step {
        self.mirrors.transition(self.mirror_state());
        if access.encoding() == registers::encodings::SPRR_PERM_EL0
            && access.direction() == Direction::Write
        {
            return self.permission_write(word, access);
        }
        if matches!(
            access.encoding(),
            registers::encodings::SPRR_PERM_EL1 | registers::encodings::SPRR_PERM_EL12
        ) && access.direction() == Direction::Write
        {
            return self.kernel_permission_write(word, access);
        }
        if timer_word(word).is_some() {
            return self.timer_access(word, access);
        }
        if let Some(index) = written(access) {
            return self.stored_write(word, access, index);
        }
        if access.encoding() == TABLE_BASE && access.direction() == Direction::Write {
            return self.table_base_write(word, access);
        }
        if let Some(index) = guarded_written(access) {
            return self.guarded_access(word, access, index);
        }
        if access.direction() == Direction::Write {
            return self.stored_write_access(word, access);
        }
        let Some(index) = mirrored(access) else {
            return self.stored_read(word, access);
        };
        if GUARDED_MIRRORS.contains(&index) {
            return self.guarded_access(word, access, index);
        }
        if COUNTER_MIRRORS.contains(&index) && self.cpu.pmu.is_none() {
            return self.register_access(word, access);
        }
        let before = self.completion_point();
        let step = self.register_access(word, access);
        let after = self.position();
        let read = self.cpu.state.x[usize::from(access.rt())];
        // A count read opens its cell only where it read the clock's last
        // count less the offset: a chip that answers through another view,
        // or a later read of the clock, leaves it to the exits.
        let value = if COUNT_MIRRORS.contains(&index) {
            let offset = registers::count_alias_offset(&self.cpu);
            (read == self.clock.last().wrapping_sub(offset)).then_some(offset)
        } else {
            Some(read)
        };
        if let (Ok(()), true, Some(value)) = (&step, after == before, value) {
            self.mirrors[index] = register_read::Cell { readable: 1, value };
            let plain = match index {
                _ if index == COUNTER_MIRRORS[0] => Some(CYCLES_PLAIN),
                _ if index == COUNTER_MIRRORS[1] => Some(INSTRUCTIONS_PLAIN),
                _ => None,
            };
            if let Some(plain) = plain {
                self.mirrors[plain] = register_read::Cell {
                    readable: 1,
                    value: 0,
                };
            }
        }
        step
    }

    /// Open [`Mirrors::events`] on the PE's event register as native code is
    /// entered: its storage, the PE's own or its canonical memory's.
    pub(super) fn enter_events(&mut self) {
        self.mirrors.events = register_read::Cell {
            readable: 1,
            value: self.cpu.event_storage(),
        };
        self.mirrors.exclusive = register_read::Cell {
            readable: u64::from(self.cpu.monitor.is_none() && !self.monitor_tracing),
            value: self.cpu.reservation_word(),
        };
    }

    /// Fill [`Mirrors::cycles`] and [`Mirrors::instructions`] as native code
    /// is entered: the monitor's PMC0 and PMC1, which no native code
    /// changes, PMC1 at the instructions retired so far, each unless a read
    /// of it answers a degraded value instead of the count.
    pub(super) fn enter_cycles(&mut self) {
        let degraded = |index: usize| {
            !self.cpu.degraded.is_empty()
                && self
                    .cpu
                    .degraded
                    .contains_key(&MIRRORED[COUNTER_MIRRORS[index]].selector())
        };
        let (cycles_degraded, instructions_degraded) = (degraded(0), degraded(1));
        let context = self.cpu.pmu_context(false);
        let retired = self.cpu.retired;
        (self.mirrors.cycles, self.mirrors.instructions) = match &self.cpu.pmu {
            Some(pmu) => (
                if cycles_degraded {
                    Default::default()
                } else {
                    pmu.native_cycles()
                },
                if instructions_degraded {
                    Default::default()
                } else {
                    pmu.native_instructions(retired, context)
                },
            ),
            None => Default::default(),
        };
    }

    /// MRS or MSR of a [`GUARDED_MIRRORS`] register: in a guarded level, one
    /// that completes plainly opens its cell on the bank's storage.
    fn guarded_access(&mut self, word: u32, access: RegisterAccess, index: usize) -> Step {
        if !self.cpu.pstate.guarded {
            return self.register_access(word, access);
        }
        let before = self.completion_point();
        let step = self.register_access(word, access);
        let value = registers::guarded_storage(&self.cpu, access.encoding());
        if let (Ok(()), true, Some(value)) = (&step, self.position() == before, value) {
            self.mirrors[index] = register_read::Cell { readable: 1, value };
        }
        step
    }

    /// The cell of a read of `access` that one was made for, if a read
    /// that completes plainly opens it on the register's live storage. A
    /// read of FPCR or FPSR completes from the frame ([`Self::stored_cell`]).
    fn stored_read(&mut self, word: u32, access: RegisterAccess) -> Step {
        use registers::encodings::{FPCR, FPSR, PAN};
        let encoding = access.encoding();
        let cell = (access.direction() == Direction::Read && access.rt() != ZERO_REGISTER)
            .then(|| self.stored_reads.iter().position(|&read| read == encoding))
            .flatten();
        let Some(cell) = cell else {
            return self.register_access(word, access);
        };
        let before = self.completion_point();
        self.plain_read = None;
        let step = self.register_access(word, access);
        let value = if matches!(encoding, FPCR | FPSR | PAN) {
            Some(0)
        } else {
            self.plain_read
                .filter(|&(register, index)| {
                    self.cpu.registers.get(register, index)
                        == self.cpu.state.x[usize::from(access.rt())]
                })
                .and_then(|(register, index)| self.cpu.registers.address(register, index))
                .or_else(|| self.devices.register_storage(self.current, encoding))
                .or_else(|| self.cpu.apple_read_storage(encoding))
                .or_else(|| {
                    // A reading at the clock's last count, as native code
                    // reads the clock.
                    let read = self.cpu.state.x[usize::from(access.rt())];
                    self.cpu
                        .energy_storage(encoding, self.clock.last())
                        .filter(|&(_, reading)| reading == read)
                        .map(|(storage, _)| storage)
                })
                .or_else(|| {
                    pstate_field(encoding).map(|(index, _)| self.mirrors.field_storage(index))
                })
        };
        if let (Ok(()), true, Some(value)) = (&step, self.position() == before, value) {
            self.mirrors[MIRRORED.len() + cell] = register_read::Cell { readable: 1, value };
        }
        step
    }

    /// MSR of a register with a [`Self::stored_writes`] cell, which one that
    /// completes plainly opens on the register's value.
    fn stored_write_access(&mut self, word: u32, access: RegisterAccess) -> Step {
        let encoding = access.encoding();
        let cell = self
            .stored_writes
            .iter()
            .position(|&write| write == encoding);
        let Some(cell) = cell else {
            return self.register_access(word, access);
        };
        let before = self.completion_point();
        let step = self.register_access(word, access);
        let storage = self.write_storage(encoding);
        if let (Ok(()), true, Some((value, _))) = (&step, self.position() == before, storage) {
            self.mirrors[stored_write_cell(cell)] = register_read::Cell { readable: 1, value };
        }
        step
    }

    /// MSR CNTV_TVAL_EL0, CNTV_CVAL_EL0 or CNTV_CTL_EL0, or MRS CNTV_CTL_EL0:
    /// one that completes plainly where it reaches CNTHV_* with no offset
    /// ([`TIMER_COMPARE`]) opens the timer cells on that storage, once the
    /// registers show the access did what the native one does.
    fn timer_access(&mut self, word: u32, access: RegisterAccess) -> Step {
        const HCR_E2H: u64 = 1 << 34;
        let before = self.completion_point();
        let written = self.cpu.state.x.get(usize::from(access.rt())).copied();
        let written = if access.rt() == ZERO_REGISTER {
            0
        } else {
            written.unwrap_or(0)
        };
        let step = self.register_access(word, access);
        let pstate = self.cpu.pstate;
        if step.is_err()
            || self.position() != before
            || pstate.el != 2
            || pstate.guarded
            || timer::agt::present(self.cpu.registers.chip())
            || self.cpu.read(SystemRegister::HCR_EL2) & HCR_E2H == 0
        {
            return step;
        }
        let compare = self.cpu.registers.get(SystemRegister::CNTHV_CVAL_EL2, 0);
        let control = self.cpu.registers.get(SystemRegister::CNTHV_CTL_EL2, 0);
        let fixed = self.register_fixed_bits(SystemRegister::CNTHV_CTL_EL2);
        let did = match timer_word(word).map(|(access, _)| access) {
            Some(TimerWord::Value) => {
                compare == self.clock.last().wrapping_add(written as u32 as i32 as u64)
            }
            Some(TimerWord::Compare) => compare == written,
            Some(TimerWord::ControlWrite) => control == written & fixed.keep | fixed.set,
            Some(TimerWord::ControlRead) => {
                let met = control & timer::ENABLE != 0 && self.clock.last() >= compare;
                access.rt() == ZERO_REGISTER
                    || self.cpu.state.x[usize::from(access.rt())]
                        == control & !timer::ISTATUS | if met { timer::ISTATUS } else { 0 }
            }
            None => false,
        };
        let storage = (
            self.cpu
                .registers
                .address(SystemRegister::CNTHV_CVAL_EL2, 0),
            self.cpu.registers.address(SystemRegister::CNTHV_CTL_EL2, 0),
        );
        if let (true, (Some(compare), Some(control))) = (did, storage) {
            self.mirrors[TIMER_COMPARE] = register_read::Cell {
                readable: 1,
                value: compare,
            };
            self.mirrors[TIMER_CONTROL] = register_read::Cell {
                readable: 1,
                value: control,
            };
        }
        step
    }

    /// MSR of `WRITTEN[index]`: one that only reaches the register's
    /// storage opens its cell there, for a register that takes what the
    /// thunk's fixed bits say: the register the write reached may be
    /// another EL's.
    fn stored_write(&mut self, word: u32, access: RegisterAccess, index: usize) -> Step {
        let (_, named) = WRITTEN[index];
        let before = self.completion_point();
        self.plain_read = None;
        let step = self.register_access(word, access);
        let value = self
            .plain_read
            .filter(|&(register, _)| {
                self.register_fixed_bits(register) == self.register_fixed_bits(named)
            })
            // A native thread pointer write also updates TPIDR_EL1's read
            // cells: it must reach TPIDR_EL1 itself. (A completed write
            // leaves none of its reads degraded: `degraded_written`.)
            .filter(|&(register, _)| {
                index != THREAD_POINTER_WRITE || register == SystemRegister::TPIDR_EL1
            })
            .and_then(|(register, index)| match register {
                SystemRegister::FPSR => Some(0),
                _ => self
                    .cpu
                    .el2_pointer_key_storage(register)
                    .or_else(|| self.cpu.registers.address(register, index)),
            });
        if let (Ok(()), true, Some(value)) = (&step, self.position() == before, value) {
            self.mirrors[written_cell(index)] = register_read::Cell { readable: 1, value };
        }
        step
    }

    /// MSR TTBR0_EL1: one that only reaches the register's storage opens
    /// [`TABLE_BASE_KEPT`] there, for a register that takes what the
    /// thunk's fixed bits say, so a later write of the same value completes
    /// natively. A write of another value changes translation and stays the
    /// engine's.
    fn table_base_write(&mut self, word: u32, access: RegisterAccess) -> Step {
        let before = self.completion_point();
        self.plain_read = None;
        let step = self.register_access(word, access);
        let value = self
            .plain_read
            .filter(|&(register, _)| {
                matches!(
                    register,
                    SystemRegister::TTBR0_EL1 | SystemRegister::TTBR0_EL2
                ) && self.register_fixed_bits(register)
                    == self.register_fixed_bits(SystemRegister::TTBR0_EL1)
            })
            .and_then(|(register, index)| self.cpu.registers.address(register, index));
        if let (Ok(()), true, Some(value)) = (&step, self.position() == before, value) {
            self.mirrors[TABLE_BASE_KEPT] = register_read::Cell { readable: 1, value };
        }
        step
    }

    /// MSR SPRR_PERM_EL0. Like every SPRR_PERM write it closes the
    /// register's cells, but a cell open before it reopens with what a read
    /// now returns: the write changes neither the controls the read's
    /// plainness follows from (SPRR_CONFIG_EL1, HACR_EL2, HCR_EL2 and the
    /// EL) nor which SPRR_PERM register it reads, so dyld's read-back after
    /// each toggle completes natively. That holds in every state whose read
    /// names the register this write reached ([`user_permission_register`]):
    /// a kernel writes it at EL2 at each thread switch and reads it back in
    /// other states of the host, whose cells then stay open too.
    pub(super) fn permission_write(&mut self, word: u32, access: RegisterAccess) -> Step {
        let cells = self.mirrors.banks.map(|cells| cells[SPRR_PERM_MIRROR]);
        let tge = self.cpu.in_host();
        let current = self.mirror_state();
        let reached = user_permission_register(current, tge);
        let reopen: Vec<usize> = (0..MIRROR_STATES)
            .filter(|&state| {
                cells[state].readable != 0 && user_permission_register(state, tge) == reached
            })
            .collect();
        let before = self.completion_point();
        let step = self.register_access(word, access);
        let plain = step.is_ok() && self.position() == before;
        if plain {
            self.keep_permission(word, access);
        }
        if !plain || reopen.is_empty() {
            return step;
        }
        let read = decode_register(word | MRS_READ).expect("MSR has an MRS form");
        let physical = self.now();
        if let RegisterOutcome::Read(value) = self.cpu.apple_access(read, physical) {
            let value = self.degraded_read(read, value);
            for state in reopen {
                self.mirrors.banks[state][SPRR_PERM_MIRROR] =
                    register_read::Cell { readable: 1, value };
            }
        }
        step
    }

    /// MSR SPRR_PERM_EL1 or SPRR_PERM_EL12, which XNU makes around its
    /// writes to protected data: they write the EL1 and EL2 permissions
    /// alone, so SPRR_PERM_EL0's cells, closed by every SPRR_PERM write,
    /// reopen as they were in every state. Neither the register an
    /// SPRR_PERM_EL0 read names nor what decides its plainness changed.
    fn kernel_permission_write(&mut self, word: u32, access: RegisterAccess) -> Step {
        let cells = self.mirrors.banks.map(|cells| cells[SPRR_PERM_MIRROR]);
        let before = self.completion_point();
        let step = self.register_access(word, access);
        if step.is_ok() && self.position() == before {
            for (bank, cell) in self.mirrors.banks.iter_mut().zip(cells) {
                bank[SPRR_PERM_MIRROR] = cell;
            }
            self.keep_permission(word, access);
        }
        step
    }

    /// After a plain MSR SPRR_PERM_EL0 or SPRR_PERM_EL1: open its
    /// [`PERMISSION_KEPT`] cell on the storage the write reached in the
    /// state, which a read there returns, so a later write of the value it
    /// holds completes natively.
    fn keep_permission(&mut self, word: u32, access: RegisterAccess) {
        let Some((kept, _)) = permission_write_word(word) else {
            return;
        };
        if self
            .cpu
            .degraded
            .contains_key(&access.encoding().selector())
        {
            return;
        }
        if let Some(value) = self.cpu.apple_read_storage(access.encoding()) {
            self.mirrors[kept] = register_read::Cell { readable: 1, value };
        }
    }

    /// The PC, EL, GXF state and exception count of the PE, which an
    /// instruction that completes without an exception leaves as
    /// [`Self::completion_point`] gives before it.
    pub(super) fn position(&self) -> (u64, u8, bool, u64) {
        (
            self.cpu.pc,
            self.cpu.pstate.el,
            self.cpu.pstate.guarded,
            self.statistics.exceptions,
        )
    }

    /// [`Self::position`] once the current instruction completes plainly.
    pub(super) fn completion_point(&self) -> (u64, u8, bool, u64) {
        let (pc, el, guarded, exceptions) = self.position();
        (pc + INSTRUCTION, el, guarded, exceptions)
    }

    /// MRS and MSR: the generated accessor of an architectural register, or
    /// Apple's implementation-defined dispatch; any other encoding is
    /// UNDEFINED.
    pub(super) fn register_access(&mut self, word: u32, access: RegisterAccess) -> Step {
        use registers::encodings::{FPCR, FPSR};
        let encoding = access.encoding();
        if matches!(encoding, FPCR | FPSR) && !self.floating_point_enabled() {
            let exception = self.floating_point_exception(self.cpu.pc);
            self.take(exception);
            return Ok(());
        }
        // Apple's vGIC CPU interface answers the ICH_* and ICC_* encodings
        // before the generated accessors, which follow the profile's absent
        // FEAT_GICv3.
        if super::super::vgic::CpuInterface::may_access(encoding) {
            let write = (access.direction() == Direction::Write).then(|| match access.rt() {
                31 => 0,
                rt => self.cpu.state.x[usize::from(rt)],
            });
            let context = self.cpu.vgic_context(write.is_some());
            if let Some(outcome) = self.cpu.vgic.access(encoding, write, &context) {
                return self.register_outcome(word, Some(access), outcome);
            }
        }
        if self.hacr_reaches(access)
            && let Some(step) = self.hacr_guest(word, hacr::Operation::Register(encoding))
        {
            return step;
        }
        if let Some(step) = self.system_access(word, access) {
            return step;
        }
        if let Some(outcome) = self.board_register(access) {
            return self.register_outcome(word, Some(access), outcome);
        }
        let physical = self.now();
        // A write's outcome forgets derived state, and a trap's exception
        // does; a read changes none of it.
        let outcome = self.cpu.apple_access(access, physical);
        self.register_outcome(word, Some(access), outcome)
    }

    /// Whether HACR_EL2 decides an MRS or MSR of a guest: AIDR_EL1 from EL1,
    /// an op0 3, CRn 11 or 15 encoding from EL1 that HCR_EL2.TIDCP does not
    /// already trap to EL2, and from EL0 the two such encodings EL0 reaches
    /// (SPRR_PERM_EL0, and reads of the counter alias); the others are
    /// UNDEFINED there.
    pub(super) fn hacr_reaches(&self, access: RegisterAccess) -> bool {
        let encoding = access.encoding();
        if self.cpu.pstate.el == 2 || self.cpu.in_host() {
            return false;
        }
        let apple = encoding.op0() == 3 && matches!(encoding.crn(), 11 | 15);
        match self.cpu.pstate.el {
            1 => encoding == registers::encodings::AIDR_EL1 || (apple && !self.cpu.hcr(HCR_TIDCP)),
            _ => {
                encoding == registers::encodings::SPRR_PERM_EL0
                    || (encoding == registers::encodings::CNTVCT_ALIAS_EL0
                        && access.direction() == Direction::Read)
            }
        }
    }

    /// HACR_EL2 on an Apple-specific operation of an EL1&0 guest
    /// (`machine::hacr`): the step when a set bit traps it to EL2 (EC 0x18,
    /// the instruction's ISS) or refuses it, `None` when it runs.
    pub(super) fn hacr_guest(&mut self, word: u32, operation: hacr::Operation) -> Option<Step> {
        if self.cpu.pstate.el == 2 || self.cpu.in_host() {
            return None;
        }
        let value = self.cpu.read(SystemRegister::HACR_EL2);
        if value == 0 {
            return None;
        }
        let context = hacr::Context {
            el: self.cpu.pstate.el,
            guarded: self.cpu.pstate.guarded,
        };
        match hacr::guest(value, operation, context) {
            hacr::Outcome::Runs => None,
            hacr::Outcome::Trap => {
                let iss = registers::word_trap_iss(word);
                let exception = Exception::synchronous(Class::SystemRegister, iss, self.cpu.pc);
                self.take(exception.to(2));
                Some(Ok(()))
            }
            hacr::Outcome::Refused(detail) => Some(Err(
                self.unsupported(RequirementKind::Instruction { word, detail })
            )),
        }
    }

    /// An Apple IMPLEMENTATION DEFINED register of the board (`Devices::
    /// system_register`), at EL1 and EL2. EL0 accesses and EL1 accesses that
    /// HCR_EL2.TIDCP traps outside the host are left to `Cpu::apple_access`,
    /// which makes them UNDEFINED or traps them as for the core's own.
    pub(super) fn board_register(&mut self, access: RegisterAccess) -> Option<RegisterOutcome> {
        let encoding = access.encoding();
        let el = self.cpu.pstate.el;
        if encoding.op0() != 3
            || encoding.crn() != 15
            || el == 0
            || (el == 1 && !self.cpu.in_host() && self.cpu.hcr(HCR_TIDCP))
        {
            return None;
        }
        let write = (access.direction() == Direction::Write).then(|| match access.rt() {
            31 => 0,
            rt => self.cpu.state.x[usize::from(rt)],
        });
        let now = self.now();
        let result = self
            .devices
            .system_register(self.current, encoding, write, now)?;
        self.device_lines = None;
        // A fast-IPI write moves only device lines, which the step reads
        // again: nothing derived from the PE's registers changes.
        let fast_ipi = crate::machine::fast_ipi::Register::decode(encoding).is_some();
        Some(match (result, write) {
            (Ok(value), None) => RegisterOutcome::Read(value),
            (Ok(_), Some(_)) => RegisterOutcome::Written(registers::Changes {
                reads_kept: !fast_ipi,
                stored: fast_ipi,
                ..registers::Changes::default()
            }),
            (Err(reason), _) => RegisterOutcome::Refused(Refusal::UnsupportedSetting(reason)),
        })
    }

    /// MSR (immediate) after `write_pstate_field` from `before`. A write
    /// that changes only the A, I or F masks, only SPSel or only PAN leaves
    /// System registers and the other PSTATE fields as they were, so state
    /// derived from them stays: the DAIF mirror takes the new masks, PAN
    /// changes the host's requester, and SPSel only which SP `GuestState`
    /// holds, which native code can then switch ([`STACK_SELECT`]). The step
    /// re-evaluates interrupt delivery after any of them.
    /// Every other change forgets derived state as an MSR does.
    pub(super) fn pstate_immediate(
        &mut self,
        word: u32,
        before: Pstate,
        outcome: RegisterOutcome,
    ) -> Step {
        let after = self.cpu.pstate;
        let RegisterOutcome::Written(changes) = outcome else {
            return self.register_outcome(word, None, outcome);
        };
        if changes.unmodeled.is_some() || changes.irq_steering.is_some() {
            return self.register_outcome(word, None, RegisterOutcome::Written(changes));
        }
        let only = |field: Pstate| field == after;
        if only(Pstate {
            daif: after.daif,
            ..before
        }) && (before.daif ^ after.daif) & DAIF_D == 0
        {
            let cell = &mut self.mirrors[DAIF_MIRROR];
            if cell.readable != 0 {
                cell.value = daif_register(after.daif);
            }
            self.mirrors[MASK_WRITES].readable = 1;
        } else if only(Pstate {
            pan: after.pan,
            ..before
        }) {
            if before.pan != after.pan {
                self.sync_host_regime();
            }
            if self.host.pan_in_flags() {
                self.mirrors[PAN_WRITES].readable = 1;
            }
        } else if !only(Pstate {
            sp: after.sp,
            ..before
        }) {
            return self.register_outcome(word, None, RegisterOutcome::Written(changes));
        } else {
            let level = match after.el {
                1 => SystemRegister::SP_EL1,
                _ => SystemRegister::SP_EL2,
            };
            let stack = match after.guarded {
                true => Some(self.cpu.guarded_stack()),
                false => self.cpu.registers.address(level, 0),
            };
            if let Some(value) = stack {
                self.mirrors[STACK_SELECT] = register_read::Cell { readable: 1, value };
            }
        }
        self.next()
    }

    pub(super) fn register_outcome(
        &mut self,
        word: u32,
        access: Option<RegisterAccess>,
        outcome: RegisterOutcome,
    ) -> Step {
        match outcome {
            RegisterOutcome::Read(value) => {
                let access = access.expect("reads come from MRS");
                let value = self.degraded_read(access, value);
                if access.rt() != 31 {
                    self.cpu.state.x[usize::from(access.rt())] = value;
                }
            }
            RegisterOutcome::Written(changes) => {
                if let Some(access) = access {
                    self.degraded_written(access);
                }
                if let Some(missing) = changes.unmodeled {
                    let refusal = Refusal::UnsupportedSetting(missing);
                    self.degrade(self.register_requirement(word, access, refusal));
                }
                if let Some(refuses) = changes.irq_steering {
                    let now = self.now();
                    self.devices.irq_steering(self.current, refuses, now);
                    self.device_lines = None;
                }
                // PSTATE.D and Apple registers can change what is derived.
                if changes.reads_kept && changes.pointer {
                    self.forget_pointer_derived();
                } else if changes.reads_kept {
                    self.forget_derived();
                } else if !changes.stored && !changes.permissions {
                    self.derived_changed();
                }
                if changes.translation {
                    self.translations_changed();
                } else if changes.controls {
                    self.controls_changed();
                } else if changes.permissions {
                    self.permissions_changed();
                } else if changes.regime {
                    self.sync_regime();
                }
                if changes.floating_point {
                    // Blocks are keyed by FP enablement; nothing to discard.
                }
            }
            RegisterOutcome::Undefined => return self.undefined(),
            RegisterOutcome::Trapped(el) => {
                let iss = registers::word_trap_iss(word);
                let exception = Exception::synchronous(Class::SystemRegister, iss, self.cpu.pc);
                self.take(match el {
                    Some(el) => exception.to(el),
                    None => exception,
                });
                return Ok(());
            }
            RegisterOutcome::SmeTrapped(el) => {
                let exception = Exception::synchronous(Class::SmeAccess, 0, self.cpu.pc);
                self.take(exception.to(el));
                return Ok(());
            }
            RegisterOutcome::Refused(refusal) => {
                return self.register_refused(word, access, refusal);
            }
            RegisterOutcome::View(encoding, bank) => {
                let access = access.expect("views come from MRS and MSR");
                return self
                    .system_access_as(word, access, encoding, bank)
                    .unwrap_or_else(|| {
                        Err(Stop::Unsupported(self.register_requirement(
                            word,
                            Some(access),
                            Refusal::Unimplemented,
                        )))
                    });
            }
        }
        self.next()
    }

    /// A System register access whose model refused it (an encoding the
    /// chip's components use without a model, or a setting or state the
    /// model does not carry out), reported once: the register behaves as
    /// storage. A read returns the value last written
    /// to it, 0 before the first write, and a write is held without the
    /// refused effect ([`Cpu::degraded`](crate::machine::cpu::Cpu)).
    pub(super) fn register_refused(
        &mut self,
        word: u32,
        access: Option<RegisterAccess>,
        refusal: Refusal,
    ) -> Step {
        self.degrade(self.register_requirement(word, access, refusal));
        if let Some(access) = access {
            let key = access.encoding().selector();
            let rt = usize::from(access.rt());
            if access.direction() == Direction::Read {
                let value = self.cpu.degraded.get(&key).copied().unwrap_or(0);
                if rt != 31 {
                    self.cpu.state.x[rt] = value;
                }
            } else {
                let value = if rt == 31 { 0 } else { self.cpu.state.x[rt] };
                self.cpu.degraded.insert(key, value);
                // A read now answers what the write held, not the storage.
                for kept in PERMISSION_KEPT {
                    self.mirrors.close(kept);
                }
            }
        }
        self.next()
    }

    /// A completed read of `access`: the value a refused write left in
    /// degraded storage, if one did, else the model's `value`.
    pub(super) fn degraded_read(&self, access: RegisterAccess, value: u64) -> u64 {
        if self.cpu.degraded.is_empty() {
            return value;
        }
        let key = access.encoding().selector();
        self.cpu.degraded.get(&key).copied().unwrap_or(value)
    }

    /// A completed write of `access`: its model holds the register again.
    pub(super) fn degraded_written(&mut self, access: RegisterAccess) {
        if !self.cpu.degraded.is_empty() {
            self.cpu.degraded.remove(&access.encoding().selector());
        }
    }

    /// The requirement of a System register access this machine cannot
    /// carry out in full, with the syndrome its trap would report.
    pub(super) fn register_requirement(
        &self,
        word: u32,
        access: Option<RegisterAccess>,
        refusal: Refusal,
    ) -> Requirement {
        let mut requirement = self.requirement(RequirementKind::SystemRegister { word, refusal });
        requirement.syndrome = Some(super::super::exception::syndrome(
            Class::SystemRegister,
            access.map_or(0, registers::trap_iss),
        ));
        requirement
    }

    /// SYS and SYSL: the generated accessor of an architectural System
    /// instruction; every other encoding is UNDEFINED. In the
    /// IMPLEMENTATION DEFINED space (CRn 11 and 15) HCR_EL2.TIDCP first
    /// traps EL1 outside the host to EL2, as Arm's accessor for that space
    /// does (the profile has no FEAT_TIDCP1, so SCTLR_ELx.TIDCP is RES0).
    /// Its encodings are then UNDEFINED, Arm's behavior for an unallocated
    /// encoding: no 26A434 SPTM, TXM, kernelcache or system kext of any
    /// chip executes one (the census has no `S1_` row), so
    /// that the chips allocate none is an inference from the census.
    pub(super) fn maintenance(&mut self, word: u32, access: RegisterAccess) -> Step {
        if let Some(step) = self.system_access(word, access) {
            return step;
        }
        if matches!(access.encoding().crn(), 11 | 15)
            && self.cpu.pstate.el == 1
            && !self.cpu.in_host()
            && self.cpu.hcr(HCR_TIDCP)
        {
            return self.register_outcome(word, Some(access), RegisterOutcome::Trapped(Some(2)));
        }
        self.undefined()
    }

    pub(super) fn take(&mut self, exception: Exception) {
        // The instruction being completed traps or faults: it does not
        // retire. SVC, HVC and SMC return after themselves and retire.
        if self
            .completing
            .take_if(|pc| {
                exception.vector == Vector::Synchronous && exception.return_address == *pc
            })
            .is_some()
        {
            self.cpu.retired -= 1;
        }
        let trace = self.monitor_before(MonitorKind::Exception {
            vector: exception.vector,
            syndrome: exception.syndrome,
            return_address: exception.return_address,
            fault_address: exception.fault_address,
        });
        let (from_el, from_guarded) = (self.cpu.pstate.el, self.cpu.pstate.guarded);
        self.cpu.take(exception);
        self.statistics.exceptions += 1;
        self.host.report(&Event::Exception {
            vector: exception.vector,
            from_el,
            from_guarded,
            syndrome: exception.syndrome,
            return_address: exception.return_address,
            fault_address: exception.fault_address,
        });
        self.monitor_after(trace);
        self.transitioned();
    }

    /// Resume a stopped architectural exception after a validation fixture has
    /// repaired its backing. Apply the ELR/SPSR and synchronize all
    /// native controls, without retiring a guest instruction or supplying OS
    /// services. Production guest kernels return through their own ERET.
    #[cfg(feature = "test-support")]
    pub fn resume_exception(
        &mut self,
    ) -> Result<(), stela_compiler::context::memory::MemoryStatus> {
        let target = self.cpu.exception_link();
        self.cpu.exception_return(target)?;
        self.pacm_written = true;
        self.sync_regime();
        Ok(())
    }

    /// What iBoot left in the core lock registers (`machine::ctrr`): every
    /// cluster and PE takes it now, and again when it starts. The boot
    /// mount calls this before the first instruction.
    pub fn hand_off_core_locks(&mut self, handoff: ctrr::CoreLocks) {
        let handoff = alloc::sync::Arc::new(handoff);
        let parked = self.peers.iter_mut().filter_map(|peer| match peer {
            processors::Peer::Parked(parked) => Some(parked.cpu_mut()),
            _ => None,
        });
        for cpu in core::iter::once(&mut self.cpu).chain(parked) {
            cpu.ctrr.replace(ctrr::Cluster::handed_off(&handoff));
            cpu.forget_ctrr_locks();
            cpu.ctrr_core = ctrr::Core::handed_off(&handoff);
            cpu.core_locks = handoff.clone();
        }
        self.translations_changed();
    }

    /// Complete the current instruction: the PC advances and BTYPE clears.
    pub(super) fn next(&mut self) -> Step {
        self.cpu.pc += INSTRUCTION;
        self.cpu.pstate.btype = 0;
        Ok(())
    }
}

impl<H: Host, D: Devices> Machine<H, D> {
    /// A DSB completed: where it left none of this PE's maintenance
    /// outstanding, later DSBs complete natively ([`Mirrors::barrier`])
    /// until it has some again.
    ///
    /// What a DSB waits for here, all of it this PE's own (Arm ARM, DSB:
    /// it completes the PE's earlier explicit accesses and the cache and
    /// TLB maintenance it issued, and waits for nothing other PEs issued):
    ///
    /// - Its earlier memory accesses, for every later one. The engine and
    ///   native code make each access in program order and x86 keeps them
    ///   ordered for other PEs (TSO) but for a store before a later load,
    ///   which the thunk's MFENCE orders, as [`data_barrier`] does here.
    ///   Device accesses are no part of it: each completes in the engine,
    ///   at its own exit, before the next instruction.
    /// - Its TLB and instruction cache maintenance. This PE applies a TLBI
    ///   or IC IVAU to itself as the engine completes it; what the other PEs
    ///   must apply waits in `outgoing` and `posted`, and `code_posted` names
    ///   a page posted, until [`Self::complete_maintenance`] sends it and
    ///   waits until each applied it. That empties them, and only
    ///   [`Self::post`] fills them again, which closes the cell, as does an
    ///   open [`CODE_PAGE`] cell, which the DSB that ends a run of lines must
    ///   close ([`Mirrors::open_code_page`]). Native code posts nothing and
    ///   opens no code page cell: only the engine does, so the cell stays
    ///   true for a whole native run.
    ///
    /// Maintenance other PEs posted to this one is theirs to wait for: this
    /// PE applies it at its next return to the engine, to which their mail
    /// brings it, whether or not it runs a DSB. Neither System registers
    /// nor the PE's state decide any of this, so the cell is the PE's
    /// alone, outside the cell banks: derived state changes and
    /// transitions leave it open.
    pub(super) fn open_barrier(&mut self) {
        let quiet =
            self.outgoing.is_empty() && self.posted.is_empty() && self.code_posted.is_none();
        self.mirrors.barrier = register_read::Cell {
            readable: u64::from(quiet),
            value: 0,
        };
    }
}

/// The host fence of a DSB the engine completes: every store this PE made
/// before it is visible to the other PEs before any later load (x86 orders
/// store to load only with a fence or a locked instruction). Native code
/// issues its own MFENCE only while [`Mirrors::barrier`] is open.
fn data_barrier() {
    #[cfg(test)]
    tests::DATA_BARRIERS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
}

/// Whether an SPRR_PERM_EL0 read in mirror state `state` names the EL2&0
/// regime's EL0 permissions (`cpu::PERM_EL0_20`), as at EL2 and at EL0 under
/// HCR_EL2.TGE, rather than the EL1&0 regime's (`Cpu::regime`,
/// `Cpu::apple_el0`). HCR_EL2 writes close every cell, so the TGE a write
/// finds is that of every open cell.
fn user_permission_register(state: usize, tge: bool) -> bool {
    match state & 3 {
        2 => true,
        0 => tge,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host fences [`data_barrier`] has issued, across every test.
    pub(super) static DATA_BARRIERS: core::sync::atomic::AtomicU64 =
        core::sync::atomic::AtomicU64::new(0);

    #[test]
    fn a_dsb_with_no_posted_maintenance_still_fences_the_host() {
        use core::sync::atomic::Ordering::Relaxed;
        const DSB_ISH: u32 = 0xd503_3b9f;
        const DSB_ISHLD: u32 = 0xd503_399f;
        let mut machine = super::super::tests::machine(&[]);
        for word in [DSB_ISH, DSB_ISHLD] {
            let before = DATA_BARRIERS.load(Relaxed);
            machine.system_instruction(word).unwrap();
            assert!(DATA_BARRIERS.load(Relaxed) > before, "{word:#x}");
        }
    }

    #[test]
    fn decoded_operation_reuses_words_across_pcs_and_still_checks_fallthrough() {
        let mut cache = DecodeCache::default();
        let word = 0xd503_205f; // WFE
        let expected = SystemOperation::Wait {
            kind: WaitKind::Event,
            timeout_register: None,
        };
        assert_eq!(cache.operation(0x1000, word), Some(expected));
        assert_eq!(cache.operation(0x2000, word), Some(expected));
        assert_eq!(cache.operation(u64::MAX - 3, word), None);
        assert_eq!(cache.operation(0x3000, word), Some(expected));
    }
}
