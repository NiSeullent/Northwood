//! The context System register models read and write the PE, its translation
//! and its devices through.

use super::*;

/// SCTLR_ELx.EnALS: a trap of LD64B and ST64B, which reaches no MRS or MSR
/// and no translation.
const SCTLR_ENALS: u64 = 1 << 56;

impl<H: Host, D: Devices> Context for Access<'_, H, D> {
    fn chip(&self) -> Chip {
        self.machine.cpu.registers.chip()
    }

    fn implemented(&self, feature: Feature) -> Result<bool, Effect> {
        self.machine
            .cpu
            .registers
            .implemented(feature)
            .map_err(|error| Effect::Refused(error.reason()))
    }

    fn el(&self) -> u64 {
        u64::from(self.machine.cpu.pstate.el)
    }

    fn pstate(&self, field: Pstate) -> u64 {
        let cpu = &self.machine.cpu;
        let state = cpu.pstate;
        let nzcv = u64::from(cpu.state.nzcv >> 28);
        let daif = u64::from(state.daif);
        match field {
            Pstate::N => nzcv >> 3 & 1,
            Pstate::Z => nzcv >> 2 & 1,
            Pstate::C => nzcv >> 1 & 1,
            Pstate::V => nzcv & 1,
            Pstate::D => u64::from(daif & u64::from(DAIF_D) != 0),
            Pstate::A => u64::from(daif & u64::from(DAIF_A) != 0),
            Pstate::I => u64::from(daif & u64::from(DAIF_I) != 0),
            Pstate::F => u64::from(daif & u64::from(DAIF_F) != 0),
            Pstate::SP => u64::from(state.sp),
            Pstate::PAN => u64::from(state.pan),
            Pstate::UAO => u64::from(state.uao),
            Pstate::DIT => u64::from(state.dit),
            Pstate::SSBS => u64::from(state.ssbs),
            Pstate::TCO => u64::from(state.tco),
        }
    }

    fn set_pstate(&mut self, field: Pstate, value: u64) {
        let cpu = &mut self.machine.cpu;
        let bit = value & 1 != 0;
        let flag = |shift: u32| 1u32 << (28 + shift);
        let mut next: PstateBits = cpu.pstate;
        let daif = |next: &mut PstateBits, mask: u8| {
            if bit {
                next.daif |= mask;
            } else {
                next.daif &= !mask;
            }
        };
        match field {
            Pstate::N | Pstate::Z | Pstate::C | Pstate::V => {
                let shift = match field {
                    Pstate::N => 3,
                    Pstate::Z => 2,
                    Pstate::C => 1,
                    _ => 0,
                };
                if bit {
                    cpu.state.nzcv |= flag(shift);
                } else {
                    cpu.state.nzcv &= !flag(shift);
                }
                return;
            }
            Pstate::D => daif(&mut next, DAIF_D),
            Pstate::A => daif(&mut next, DAIF_A),
            Pstate::I => daif(&mut next, DAIF_I),
            Pstate::F => daif(&mut next, DAIF_F),
            Pstate::SP => {
                next.sp = bit;
                self.regime = true;
            }
            Pstate::PAN => {
                next.pan = bit;
                self.regime = true;
            }
            Pstate::UAO => next.uao = bit,
            Pstate::DIT => next.dit = bit,
            Pstate::SSBS => next.ssbs = bit,
            // PSTATE.TCO decides which accesses are tag checked.
            Pstate::TCO => {
                next.tco = bit;
                self.regime = true;
            }
        }
        let masks = cpu.pstate.daif;
        cpu.set_pstate(next);
        if matches!(field, Pstate::D | Pstate::A | Pstate::I | Pstate::F) {
            // The masks derive nothing but an open DAIF cell's value, as
            // for MSR DAIFSet and DAIFClr (`Machine::pstate_immediate`).
            let cell = &mut self.machine.mirrors[super::super::mirrors::DAIF_MIRROR];
            if cell.readable != 0 {
                cell.value = super::super::mirrors::daif_register(next.daif);
            }
        }
        if (masks ^ next.daif) & DAIF_D != 0 {
            // PSTATE.D masks same-EL debug exceptions, watchpoints included.
            self.regime = true;
        }
    }

    fn x(&self) -> u64 {
        if self.rt == 31 {
            0
        } else {
            self.machine.cpu.state.x[usize::from(self.rt)]
        }
    }

    fn set_x(&mut self, value: u64) {
        self.result = Some(value);
    }

    fn read(&mut self, register: Register, index: u64) -> u64 {
        let machine = &*self.machine;
        let cpu = &machine.cpu;
        // The reset address, where the board holds it (Apple's
        // `cpu-impl-reg` RVBAR); with none it has no profile value.
        if register == Register::RVBAR_EL2
            && let Some(address) = machine.devices.reset_vector(machine.current)
        {
            return address;
        }
        if machine.profile_missing.contains(&register) {
            self.refuse("identification register without a profile value");
            return 0;
        }
        if let Some(value) = cpu.read_el2_pointer_key(register) {
            return value.unwrap_or_else(|reason| {
                self.refuse(reason);
                0
            });
        }
        match register {
            Register::AMEVCNTR0_EL0 | Register::AMCNTENSET0_EL0 | Register::AMCNTENCLR0_EL0 => {
                self.read_activity_monitor(register, index)
            }
            Register::AMEVTYPER0_EL0 => amu::EVENTS[index as usize],
            Register::SP_EL0 => {
                self.plain.set(Some((register, 0)));
                cpu.sp(0)
            }
            Register::SP_EL1 => cpu.sp(1),
            Register::SP_EL2 => cpu.sp(2),
            Register::FPCR => u64::from(cpu.state.fpcr),
            Register::FPSR => u64::from(cpu.state.fpsr),
            Register::SVCR => cpu.svcr(),
            Register::FPMR => match cpu.scalable.as_ref() {
                Some(scalable) => scalable.fpmr(),
                None => {
                    self.refuse("FPMR on a PE without its scalable state");
                    0
                }
            },
            Register::ISR_EL1 => self.pending(),
            Register::CCSIDR_EL1 => match cpu.caches {
                Some(caches) => caches.ccsidr(cpu.registers.get(Register::CSSELR_EL1, 0)),
                None => cpu.registers.get(register, 0),
            },
            register if lor_region(register) && no_lor_descriptors(cpu) => 0,
            Register::MPAM1_EL1 => {
                cpu.registers.get(register, 0) & !MPAMEN
                    | cpu.registers.get(Register::MPAM2_EL2, 0) & MPAMEN
            }
            Register::DCZID_EL0 => {
                cpu.registers.get(register, 0) & !DCZID_DZP
                    | if zero_block_prohibited(cpu) {
                        DCZID_DZP
                    } else {
                        0
                    }
            }
            // RXfull and TXfull mirror the debug communications channel.
            Register::MDCCSR_EL0 => cpu.dcc.status(),
            Register::MDSCR_EL1 => cpu.registers.get(register, 0) & !DCC_FULL | cpu.dcc.status(),
            // Every claim tag is implemented; DBGCLAIMCLR reads the claim.
            Register::DBGCLAIMSET_EL1 => debug::CLAIM_TAGS,
            Register::DBGCLAIMCLR_EL1 => u64::from(cpu.claim),
            Register::OSDTRRX_EL1 => u64::from(cpu.dcc.rx),
            Register::OSDTRTX_EL1 => u64::from(cpu.dcc.tx),
            _ => match Channel::by_control(register) {
                Some(channel) => {
                    let now = self.now();
                    channel.read_control(&self.machine.cpu, self.bank, now)
                }
                None => match (self.bank, cpu.apple.generic_timer.get(register)) {
                    (Bank::Apple, Some(value)) => value,
                    _ => {
                        if super::super::mirrors::reads_storage(register) {
                            self.plain.set(Some((register, index)));
                        }
                        cpu.registers.get(register, index)
                    }
                },
            },
        }
    }

    fn write(&mut self, register: Register, index: u64, value: u64) -> Result<(), Effect> {
        if let Some(reason) =
            crate::machine::registers::vmsa_locked(&self.machine.cpu, register, value)
        {
            return Err(Effect::Refused(reason));
        }
        if let Some(written) = self.write_activity_monitor(register, index, value) {
            return written;
        }
        // RMR_EL2.RR requests a Warm reset, which the PE takes once the write
        // completes (`Machine::warm_reset`), at the reset vector the board
        // holds for it.
        if register == Register::RMR_EL2 && value & RMR_RR != 0 {
            if self
                .machine
                .devices
                .reset_vector(self.machine.current)
                .is_none()
            {
                return Err(Effect::Refused(
                    "RMR_EL2.RR Warm reset request without a reset vector for the PE",
                ));
            }
            self.reset = true;
        }
        let cpu = &mut self.machine.cpu;
        if let Some(reason) = implementation_defined_refusal(register, value) {
            return Err(Effect::Refused(reason));
        }
        let value = if register == Register::MPAM1_EL1 {
            value & !MPAMEN
        } else {
            value
        };
        // Without LORegion descriptors the LORegion registers are RAZ/WI.
        if lor_region(register) && no_lor_descriptors(cpu) {
            return Ok(());
        }
        match register {
            // OSLAR_EL1.OSLK locks and unlocks the OS lock (OSLSR_EL1.OSLK).
            Register::OSLAR_EL1 => {
                let oslsr = cpu.read(Register::OSLSR_EL1) & !0b10 | (value & 1) << 1;
                cpu.registers.set_raw(Register::OSLSR_EL1, 0, oslsr);
                self.regime = true;
                return Ok(());
            }
            // RXfull and TXfull are writable only while the OS lock is set,
            // for saving and restoring the channel.
            Register::MDSCR_EL1 if cpu.field(Register::OSLSR_EL1, 1, 1) != 0 => {
                cpu.dcc.tx_full = value >> 29 & 1 != 0;
                cpu.dcc.rx_full = value >> 30 & 1 != 0;
            }
            Register::OSDTRRX_EL1 => {
                cpu.dcc.rx = value as u32;
                return Ok(());
            }
            Register::DBGCLAIMSET_EL1 => {
                cpu.claim |= (value & debug::CLAIM_TAGS) as u8;
                return Ok(());
            }
            Register::DBGCLAIMCLR_EL1 => {
                cpu.claim &= !((value & debug::CLAIM_TAGS) as u8);
                return Ok(());
            }
            Register::OSDTRTX_EL1 => {
                cpu.dcc.tx = value as u32;
                return Ok(());
            }
            // Once GXF is enabled HCR_EL2 is writable only from GL2 (m1n1
            // `gxf.c`, `exception_asm.S`); what a write from EL2 does then is
            // not documented.
            Register::HCR_EL2 if cpu.guarded_enabled() && !cpu.pstate.guarded => {
                return Err(Effect::Refused(
                    "HCR_EL2 write outside GL2 with GXF enabled",
                ));
            }
            Register::SVCR => {
                return cpu
                    .set_svcr(value & 1 != 0, value & 2 != 0)
                    .map_err(|error| Effect::Refused(error.reason()));
            }
            Register::SP_EL0 | Register::SP_EL1 | Register::SP_EL2 => {
                let el = match register {
                    Register::SP_EL0 => 0,
                    Register::SP_EL1 => 1,
                    _ => 2,
                };
                cpu.set_sp(el, value);
                if register == Register::SP_EL0 && cpu.pstate.sp {
                    self.plain.set(Some((register, 0)));
                }
                return Ok(());
            }
            Register::FPCR => {
                cpu.state.fpcr = value as u32 & fpcr_fields(&cpu.registers);
                return Ok(());
            }
            // Without AArch32, FPSR keeps only QC and the cumulative flags.
            Register::FPSR => {
                cpu.state.fpsr = (value & !cpu.registers.res0(Register::FPSR)) as u32;
                self.plain.set(Some((register, 0)));
                return Ok(());
            }
            // FPMR lives with the scalable state, which resets it whenever
            // PSTATE.SM changes (ResetSVEState).
            Register::FPMR => {
                let value = value & !cpu.registers.res0(Register::FPMR);
                return match cpu.scalable.as_mut() {
                    Some(scalable) => {
                        scalable.set_fpmr(value);
                        Ok(())
                    }
                    None => Err(Effect::Refused("FPMR on a PE without its scalable state")),
                };
            }
            _ => {}
        }
        if matches!(
            register,
            Register::CNTKCTL_EL1
                | Register::CNTHCTL_EL2
                | Register::CNTVOFF_EL2
                | Register::HCR_EL2
        ) {
            let now = self.now();
            self.machine.sample_events(now);
        }
        if register == Register::TPIDR_EL1 {
            self.machine.switch_thread(value);
        }
        let cpu = &mut self.machine.cpu;
        let sctlr = matches!(register, Register::SCTLR_EL1 | Register::SCTLR_EL2);
        let before = if sctlr { cpu.read(register) } else { 0 };
        let held = cpu.registers.get(register, index);
        // The Apple bank's copy keeps the register's reserved bits.
        let banked = cpu.write_el2_pointer_key(register, value)
            || self.bank == Bank::Apple && {
                let (res0, res1) = (cpu.registers.res0(register), cpu.registers.res1(register));
                cpu.apple
                    .generic_timer
                    .set(register, value & !res0 & !res1 | res1)
            };
        let key = super::super::mirrors::pointer_key(register);
        if banked && key {
            self.plain.set(Some((register, index)));
        }
        if !banked {
            cpu.registers.set(register, index, value);
            if key
                || matches!(
                    register,
                    Register::ELR_EL1
                        | Register::ELR_EL2
                        | Register::SPSR_EL1
                        | Register::SPSR_EL2
                        | Register::TPIDR_EL0
                        | Register::TPIDRRO_EL0
                        | Register::TPIDR_EL1
                        | Register::CONTEXTIDR_EL1
                        | Register::CONTEXTIDR_EL2
                        | Register::TTBR0_EL1
                        | Register::TTBR0_EL2
                )
            {
                self.plain.set(Some((register, index)));
            }
        }
        let changed = held ^ cpu.registers.get(register, index);
        self.machine
            .profile_missing
            .retain(|missing| *missing != register);
        // A guarded level writes back the controls it finds around each of
        // its calls: what derives from a value stays while the value does.
        if !banked && (changed == 0 || sctlr && changed == SCTLR_ENALS) {
            if changed != 0 {
                self.machine.forget_derived();
            }
            return Ok(());
        }
        self.machine.register_written(register);
        if table_base(register) {
            if self.translation == Translation::Unchanged {
                self.translation = Translation::TableBase;
            }
        } else if sctlr && (before ^ self.machine.cpu.read(register)) & mmu::SCTLR_TRANSLATION == 0
        {
            self.regime = true;
        } else if translates(register) {
            self.translation = Translation::All;
        }
        if watches(register) {
            self.regime = true;
        }
        // A CONTEXTIDR write changes neither translation, the host's regime
        // nor the prediction context (`PredictionContext`): breakpoints
        // read it live.
        Ok(())
    }

    fn physical_count(&mut self) -> u64 {
        let now = self.now();
        timer::count(&self.machine.cpu, self.bank, now)
    }

    fn nv_mem(&mut self, offset: u64) -> Result<u64, Effect> {
        let mut bytes = [0; 8];
        self.nested_access(offset, false, &mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn set_nv_mem(&mut self, offset: u64, value: u64) -> Result<(), Effect> {
        self.nested_access(offset, true, &mut value.to_le_bytes())
    }

    fn operation(&mut self, operation: Operation) -> Result<(), Effect> {
        match operation {
            // Fetch translations go whole; the shadow, a cache of the
            // current context alone, loses what the TLBI reaches of it, a
            // superset of its scope and level.
            Operation::Tlbi(tlbi) => {
                let (reach, everywhere, asid, broadcast) = reach(&self.machine.cpu, &tlbi);
                self.translation = Translation::Maintenance(reach, everywhere, asid, broadcast);
                Ok(())
            }
            Operation::At {
                stage,
                el,
                access,
                address,
            } => self.address_translation(stage, el, access, address),
            // The tag forms (FEAT_MTE2) act as their data forms: the host
            // has no caches, of data or of Allocation Tags.
            Operation::Dc {
                op, scope, operand, ..
            } => {
                if scope == CacheScope::SetWay {
                    return Ok(());
                }
                // The host is coherent: maintenance by VA only checks that
                // the address translates, as a write for invalidation.
                let write = op == CacheOp::Invalidate;
                self.check_maintenance(operand, write)?;
                // DC IVAC is watched as a write of the DataCacheWatchpointSize
                // block around it (64 bytes, CTR_EL0's line: Stela's choice).
                if write && scope == CacheScope::PoC {
                    let access = debug::Access {
                        address: operand & !(DC_WATCH_BYTES - 1),
                        size: DC_WATCH_BYTES,
                        read: false,
                        write: false,
                        maintenance: true,
                        el: self.machine.cpu.pstate.el,
                    };
                    if let Some(exception) = self
                        .machine
                        .watchpoint_exception(&access, self.machine.cpu.pc)
                    {
                        return Err(Effect::Exception(exception));
                    }
                }
                Ok(())
            }
            Operation::Ic { scope, operand } => {
                self.code = if scope == CacheScope::PoU {
                    Code::Page {
                        address: operand,
                        physical: self.check_maintenance(operand, false)?,
                    }
                } else {
                    Code::All
                };
                Ok(())
            }
            Operation::ZeroBlock {
                kind: CacheType::Data,
                address,
            } => self.zero_block(address),
            // DC GVA and DC GZVA.
            Operation::ZeroBlock {
                kind: kind @ (CacheType::Tag | CacheType::DataTag),
                address,
            } => self
                .machine
                .set_block_tags(address, kind == CacheType::DataTag)
                .map_err(|failure| match failure {
                    super::super::tags::Failure::Exception(exception) => {
                        Effect::Exception(exception)
                    }
                    super::super::tags::Failure::Translation(_) => {
                        Effect::Refused("DC GVA or GZVA address needs an unsupported translation")
                    }
                    super::super::tags::Failure::Refused(reason) => Effect::Refused(reason),
                }),
            Operation::ZeroBlock {
                kind: CacheType::Instruction,
                ..
            } => Err(Effect::Refused("an instruction cache zero operation")),
            // No speculation to restrict.
            Operation::RestrictPrediction { .. } => Ok(()),
        }
    }

    fn implementation_defined(&mut self, _read: bool) -> Result<Effect, Effect> {
        Err(Effect::Refused(
            "the IMPLEMENTATION DEFINED space is handled by machine::registers",
        ))
    }

    fn read_dbgdtr(&mut self, width: u32) -> u64 {
        self.machine.cpu.dcc.read(width)
    }

    fn write_dbgdtr(&mut self, width: u32, value: u64) -> Result<(), Effect> {
        self.machine.cpu.dcc.write(width, value);
        Ok(())
    }

    /// The profile has no error records (ERRIDR_EL1.NUM is 0), so the
    /// ERX*_EL1 window reads as zero and ignores writes.
    fn error_record(&self, _register: ErrorRecord, _index: u64) -> u64 {
        0
    }

    fn set_error_record(
        &mut self,
        _register: ErrorRecord,
        _index: u64,
        _value: u64,
    ) -> Result<(), Effect> {
        Ok(())
    }

    fn auxiliary_monitor(&self, _index: u64) -> bool {
        false
    }

    fn limit(&self, limit: Limit) -> u64 {
        let dfr0 = self.machine.cpu.read(Register::ID_AA64DFR0_EL1);
        match limit {
            Limit::Breakpoints => (dfr0 >> 12 & 0xf) + 1,
            Limit::Watchpoints => (dfr0 >> 20 & 0xf) + 1,
            Limit::AuxiliaryMonitors => 0,
        }
    }
}
