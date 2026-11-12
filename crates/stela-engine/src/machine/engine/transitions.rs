//! Native GENTER, GEXIT, SVC and ERET (`x86_64::transition`): the record an
//! engine-path transition leaves for the next native entry at its target,
//! the thunks that complete the instructions from records, and what the
//! engine takes back of a native run that made transitions.
use super::*;
use crate::machine::cpu::Cpu;
use stela_compiler::x86_64::transition::{
    EXCEPTION_RETURN, GUARDED_ENTER, GUARDED_EXIT, KINDS, Record, SUPERVISOR_CALL,
};

/// The SPSR bits a GEXIT record fixes: M[4:0], D, BTYPE, IL, SS, PAN, TCO
/// and PACM. The rest are operands (NZCV, A, I, F, DIT, UAO, SSBS) or no
/// part of PSTATE.
const EXIT_CHECKED: u64 = 0x1f | 1 << 9 | 3 << 10 | 3 << 20 | 1 << 22 | 1 << 25 | 1 << 35;
/// Of those, the bits a record needs clear: BTYPE, IL, SS and PACM, which
/// native code keeps clear.
const EXIT_CLEAR: u64 = 3 << 10 | 3 << 20 | 1 << 35;
/// The SPSR bits an ERET record fixes: all but the operands NZCV, A, I,
/// F, DIT, UAO and SSBS, so the mode, D, PAN, BTYPE, IL, SS, TCO, PACM and
/// whatever else SPSR holds are the record's.
const RETURN_CHECKED: u64 = !(0xf << 28 | 7 << 6 | 1 << 24 | 1 << 23 | 1 << 12);
const GENTER: u32 = 0x0020_1420;
const GEXIT: u32 = 0x0020_1400;
/// SVC #imm16 (imm16 in bits 20:5) and ERET.
const SVC: u32 = 0xd400_0001;
const SVC_IMMEDIATE: u32 = 0xffff << 5;
const ERET: u32 = 0xd69f_03e0;

/// The kind of native transition `word` makes, with GENTER's or SVC's
/// immediate.
pub(super) fn transition_word(word: u32) -> Option<(usize, u16)> {
    if word & !0x1f == GENTER {
        Some((GUARDED_ENTER, (word & 0x1f) as u16))
    } else if word == GEXIT {
        Some((GUARDED_EXIT, 0))
    } else if word & !SVC_IMMEDIATE == SVC {
        Some((SUPERVISOR_CALL, ((word & SVC_IMMEDIATE) >> 5) as u16))
    } else if word == ERET {
        Some((EXCEPTION_RETURN, 0))
    } else {
        None
    }
}

/// Why GENTER, GEXIT, SVC or ERET took the engine path, as its thunk would
/// have decided there ([`Machine::note_transition_miss`]): no record was ever made
/// from the state; one was, but the table's generation moved since; it was
/// made for another frame or stack selection; a deadline was due or none
/// was known (GEXIT, ERET); an interrupt notification came (GEXIT, ERET);
/// the operands the record fixes differ, or the engine holds a monitor
/// (ERET); an interrupt waiting would be unmasked (GEXIT, ERET);
/// or the record held, so no thunk ran (none installed, the block ran in
/// the engine, or the host's root moved).
pub mod miss {
    pub const NEVER: usize = 0;
    pub const FORGOTTEN: usize = 1;
    pub const FRAME: usize = 2;
    pub const DEADLINE: usize = 3;
    pub const EPOCH: usize = 4;
    pub const CHECK: usize = 5;
    pub const INTERRUPT: usize = 6;
    pub const OTHER: usize = 7;
    pub const REASONS: usize = 8;
}

/// [`dynamic_branch`] of a block ending with a native transition: its
/// thunk's exit to the target, which sets no BTYPE.
pub(super) fn transition_branch(word: u32) -> Option<(ExitKind, Option<u8>)> {
    transition_word(word).map(|_| (ExitKind::Return, Some(0)))
}

/// An engine-path transition whose record the next native entry at its
/// target makes, when nothing a record depends on changed meanwhile.
#[derive(Debug, Clone, Copy)]
pub(super) struct Note {
    kind: usize,
    from: Pstate,
    to: usize,
    pc: u64,
    generation: u64,
    /// GENTER: the ASPSR it banked. GEXIT: ASPSR's guarded bit.
    aspsr: u64,
    /// GEXIT and ERET: the SPSR it restored.
    spsr: u64,
    prediction: Option<PredictionContext>,
    /// SVC and ERET between levels whose PAC key enables differ: the PAC
    /// service's tag words before and after the engine reopened them, and
    /// its generation.
    pointer: Option<([PointerTags; 2], u64)>,
}

/// The PAC service's tag words (`PointerService::tags`).
type PointerTags = [u64; x86_64_pointer::TAG_WORDS];

/// What a transition starts from ([`Machine::transition_start`]).
#[derive(Debug, Clone, Copy)]
pub(super) struct Start {
    pstate: Pstate,
    prediction: Option<PredictionContext>,
    watch: Watch,
    pointer: PointerUse,
    tags: PointerTags,
}

/// How PAC instructions act at a level: the four address keys' enables
/// (`Machine::pointer_key_enables`) and whether EL2 traps them.
type PointerUse = (u64, bool);

impl<H: Host, D: Devices> Machine<H, D> {
    /// GENTER, GEXIT, SVC or ERET (`kind`) is about to take the engine
    /// path: count why ([`miss`]) in [`Statistics::transition_misses`], by
    /// kind. Only states that make records count: EL2 for all, and EL0 of
    /// the host for SVC.
    pub(super) fn note_transition_miss(&mut self, kind: usize) {
        let pstate = self.cpu.pstate;
        let counted = pstate.el == 2
            || kind == SUPERVISOR_CALL && pstate.el == 0 && !pstate.guarded && self.cpu.in_host();
        if !counted {
            return;
        }
        let returning = matches!(kind, GUARDED_EXIT | EXCEPTION_RETURN);
        let table = &self.mirrors.transitions;
        let record = table.records[self.mirror_state()][kind];
        let frame = &self.cpu.state as *const GuestState as u64;
        let reason = if record.generation == 0 {
            miss::NEVER
        } else if record.generation != table.generation {
            miss::FORGOTTEN
        } else if record.frame != frame || record.selected != u64::from(pstate.sp) {
            miss::FRAME
        } else if returning
            && self.mirrors.lines_until != u64::MAX
            && (self.mirrors.lines_until == 0 || self.now() >= self.mirrors.lines_until)
        {
            miss::DEADLINE
        } else if returning && self.chain.interrupt_moved() {
            miss::EPOCH
        } else if self.transition_operands_differ(&record, kind) {
            miss::CHECK
        } else if returning && self.transition_unmasks(kind) {
            miss::INTERRUPT
        } else {
            miss::OTHER
        };
        self.statistics.transition_misses[kind][reason] += 1;
    }

    /// The SPSR a GEXIT or ERET of `kind` restores, with GEXIT's ASPSR.
    fn returned_state(&self, kind: usize) -> Option<(u64, u64)> {
        if kind == EXCEPTION_RETURN {
            return Some((self.cpu.read(SystemRegister::SPSR_EL2), 0));
        }
        let bank = usize::from(self.cpu.pstate.el)
            .checked_sub(1)
            .and_then(|level| self.cpu.apple.guarded.get(level))?;
        Some((bank.spsr, bank.aspsr))
    }

    /// Whether the operands `record` fixes differ from the PE's, as the
    /// thunk checks them: GENTER's and SVC's PAN; GEXIT's ASPSR guarded bit
    /// and SPSR's fixed bits; ERET's SPSR fixed bits, or a monitor the
    /// engine holds (`Places::monitor`).
    fn transition_operands_differ(&self, record: &Record, kind: usize) -> bool {
        if matches!(kind, GUARDED_ENTER | SUPERVISOR_CALL) {
            return u64::from(self.cpu.pstate.pan) != record.check;
        }
        let Some((spsr, aspsr)) = self.returned_state(kind) else {
            return true;
        };
        let monitor = kind == EXCEPTION_RETURN
            && (self.cpu.monitor.is_some() || self.cpu.has_canonical_memory());
        let guarded = kind == GUARDED_EXIT && aspsr & 1 != record.aspsr_value;
        monitor || guarded || spsr & record.check_mask != record.check
    }

    /// Whether a GEXIT or ERET of `kind` would unmask an interrupt waiting
    /// at native entry.
    fn transition_unmasks(&self, kind: usize) -> bool {
        let Some((spsr, _)) = self.returned_state(kind) else {
            return false;
        };
        let masks = spsr >> 6 & 3;
        u64::from(self.cpu.state.pending) & 3 & !masks != 0
    }

    pub(super) fn transition_start(&self) -> Start {
        Start {
            pstate: self.cpu.pstate,
            prediction: self.prediction_context,
            watch: self.watch,
            pointer: self.pointer_use(),
            tags: self.pointer.tags,
        }
    }

    /// PAC's behavior at the current level: native transitions keep the PAC
    /// service, which the engine path reopens at the new level
    /// (`Machine::reopen_pointers`), so both levels must agree. Above EL0,
    /// APCTL's kernel-keys bit enables every key whatever SCTLR says.
    fn pointer_use(&self) -> PointerUse {
        use PointerAuthenticationKey::{Da, Db, Ia, Ib};
        let keys = [Ia, Ib, Da, Db].into_iter().fold(0, |keys, key| {
            keys | pointer_authentication::key_enable_bit(key)
        });
        (
            self.pointer_key_enables() & keys,
            self.pointer_authentication_trapped(),
        )
    }

    /// GENTER, GEXIT, SVC or ERET (`kind`) completed in the engine from
    /// `start`, with the `spsr` and `aspsr` the note keeps. Only transitions
    /// with neither MTE (TCO stays) nor monitor tracing (whose events native
    /// code would not report), and with PAC acting alike on both sides
    /// ([`Self::pointer_use`]), make records: GENTER and GEXIT at EL2, where
    /// HACR traps none; SVC and ERET between EL0 and EL2 of the host
    /// (HCR_EL2.E2H and TGE: one translation regime, PAC keys and controls,
    /// and no fine-grained trap of either) or within EL2, unguarded, with
    /// the watchpoints the same on both sides and no single-step in
    /// progress. ERET also needs the local monitor kept in the frame alone:
    /// a PE with canonical memory (the home PE of a parallel board) clears
    /// its reservation through the shared registry (`Cpu::exception_return`),
    /// which native code cannot reach. An SVC leaves the monitor alone
    /// (`Cpu::take`), so it makes records there too.
    pub(super) fn note_transition(&mut self, kind: usize, start: Start, spsr: u64, aspsr: u64) {
        self.statistics.engine_transitions[kind] += 1;
        let (from, to) = (start.pstate, self.cpu.pstate);
        let plain = |pstate: Pstate| !pstate.il && !pstate.ss && !pstate.pacm;
        let levels = match kind {
            GUARDED_ENTER | GUARDED_EXIT => from.el == 2 && to.el == 2,
            _ => {
                let host = |pstate: Pstate| {
                    !pstate.guarded && (pstate.el == 2 || pstate.el == 0 && self.cpu.in_host())
                };
                host(from)
                    && host(to)
                    && self.cpu.in_host()
                    && start.watch == self.watch
                    && !self.cpu.step_advance
                    && (kind == SUPERVISOR_CALL || !self.cpu.has_canonical_memory())
                    && if kind == SUPERVISOR_CALL {
                        to.el == 2
                    } else {
                        from.el == 2
                    }
            }
        };
        // Enables that differ are switched natively from a service with a
        // tag word open (which the engine reopens whole, a function of its
        // generation, enables and traps) to the tags it just reopened
        // (`transition::check_pointer`).
        let pointer = (start.pointer != self.pointer_use()
            && matches!(kind, SUPERVISOR_CALL | EXCEPTION_RETURN)
            && start.tags.iter().any(|&tag| tag != 0))
        .then_some(([start.tags, self.pointer.tags], self.pointer.generation));
        let usable = !self.monitor_tracing
            && levels
            && plain(from)
            && plain(to)
            && from.tco == to.tco
            && start.pointer.1 == self.pointer_use().1
            && (start.pointer == self.pointer_use() || pointer.is_some())
            && !self.cpu.mte()
            && match kind {
                GUARDED_EXIT => spsr & EXIT_CLEAR == 0,
                EXCEPTION_RETURN => spsr & EXIT_CLEAR == 0,
                _ => true,
            };
        self.transition_note = usable.then(|| Note {
            kind,
            from,
            to: self.mirror_state(),
            pc: self.cpu.pc,
            generation: self.mirrors.transitions.generation,
            aspsr,
            spsr,
            prediction: start.prediction,
            pointer,
        });
    }

    /// Native code is entered with `budget` transfers: the record of the
    /// noted transition is made when this is its target, in its state, with
    /// nothing a record depends on changed since and chaining allowed. The
    /// host's view follows natively where the host gives one
    /// ([`Host::native_view`]); else the record's thunk ends the run.
    pub(super) fn finish_transition(&mut self, budget: u64) {
        let Some(note) = self.transition_note.take() else {
            return;
        };
        let to = self.cpu.pstate;
        let state = self.mirror_state();
        if budget == 0
            || note.pc != self.cpu.pc
            || note.to != state
            || note.generation != self.mirrors.transitions.generation
            || note.prediction != self.prediction_context
            || to.btype != 0
            || note.pointer.is_some_and(|([_, to], generation)| {
                to != self.pointer.tags || generation != self.pointer.generation
            })
        {
            return;
        }
        let from = note.from;
        let from_state = mirror_state(from.el, from.guarded);
        let regime = Regime {
            controls: self.cpu.translating_controls(self.cpu.regime()),
            requester: self.cpu.requester(),
            watch: self.watch,
            tag_checks: self.tag_checks,
        };
        let before = Regime {
            requester: mmu::Requester {
                el: from.el,
                pan: from.pan,
                unprivileged: false,
                guarded: from.guarded,
            },
            ..regime
        };
        let same = before == regime;
        let view = if same {
            None
        } else {
            self.host.native_view(&before, &regime)
        };
        let at = |field: &u64| field as *const u64 as u64;
        let guarded = matches!(note.kind, GUARDED_ENTER | GUARDED_EXIT);
        let (spsr, elr, esr, aspsr, enter) = if guarded {
            let bank = &self.cpu.apple.guarded[usize::from(to.el) - 1];
            (
                at(&bank.spsr),
                at(&bank.elr),
                at(&bank.esr),
                at(&bank.aspsr),
                at(&bank.enter),
            )
        } else {
            // SVC and ERET reach EL2's registers alone (`Cpu::take` and
            // `Cpu::exception_return` from or to EL2).
            let address = |register| self.cpu.registers.address(register, 0);
            let (Some(spsr), Some(elr), Some(esr)) = (
                address(SystemRegister::SPSR_EL2),
                address(SystemRegister::ELR_EL2),
                address(SystemRegister::ESR_EL2),
            ) else {
                return;
            };
            (spsr, elr, esr, 0, 0)
        };
        let scratch = at(&self.mirrors.transitions.records[from_state][note.kind].scratch);
        let aspsr = if guarded { aspsr } else { scratch };
        let (sp_save, sp_load) = if Cpu::same_stack(from, to) {
            (scratch, scratch)
        } else {
            (self.cpu.stack_storage(from), self.cpu.stack_storage(to))
        };
        let salts = self.chain.salts();
        let entering = matches!(note.kind, GUARDED_ENTER | SUPERVISOR_CALL);
        // An entry's SPSR without its operands: the mode (and TCO, which
        // stays).
        let fixed = Pstate {
            daif: 0,
            pan: false,
            uao: false,
            dit: false,
            ssbs: false,
            btype: 0,
            ..from
        }
        .to_spsr(0);
        let view = view.unwrap_or(NativeView {
            cr3: 0,
            root_at: 0,
            root: 0,
            stale_at: 0,
            view_at: 0,
            view: 0,
            ac: 0,
        });
        // Cells the engine refreshes at each transition close: the counters'
        // (their count of the last transition) and, where mask reads take
        // the cell's value rather than the frame's, DAIF's.
        let exclusive = self.native_system().exclusive;
        let cell = |index: usize| at(&self.mirrors.banks[state][index].readable);
        let daif = if exclusive {
            scratch
        } else {
            cell(DAIF_MIRROR)
        };
        let close = [cell(COUNTER_MIRRORS[0]), cell(COUNTER_MIRRORS[1]), daif];
        // SVC: DIT stays, UAO clears, SSBS takes SCTLR_EL2.DSSBS
        // (`exception::entry_pstate`), as the engine path just did.
        let ssbs = PSTATE_FIELDS[2].1;
        let fields_set = [0, 0, if to.ssbs { ssbs } else { 0 }];
        if note.kind == SUPERVISOR_CALL && (to.uao || to.dit != from.dit) {
            return;
        }
        // ERET: the tag the target loses in each half (`Cpu::branch_address`
        // at the EL returned to).
        const LOWER_TAGGED: u64 = 1 << 56;
        const UPPER_TAGGED: u64 = 1 << 55;
        let untag = [
            u64::from(self.cpu.branch_address(LOWER_TAGGED) != LOWER_TAGGED),
            u64::from(self.cpu.branch_address(UPPER_TAGGED) != UPPER_TAGGED),
        ];
        let (check_mask, check) = match note.kind {
            GUARDED_ENTER | SUPERVISOR_CALL => (0, u64::from(from.pan)),
            GUARDED_EXIT => (EXIT_CHECKED, note.spsr & EXIT_CHECKED),
            _ => (RETURN_CHECKED, note.spsr & RETURN_CHECKED),
        };
        let record = Record {
            generation: self.mirrors.transitions.generation,
            frame: &self.cpu.state as *const GuestState as u64,
            selected: u64::from(from.sp),
            to: state as u64,
            cells: self.mirrors.banks[state].as_ptr() as u64,
            contexts: self.chain.contexts(),
            salts: u64::from(salts[0]) | u64::from(salts[1]) << 32,
            contexts_generation: self.mirrors.transitions.contexts_generation,
            sp_save,
            sp_load,
            spsr,
            elr,
            esr,
            aspsr,
            target: match note.kind {
                GUARDED_ENTER => enter,
                GUARDED_EXIT => elr,
                // The record's own `vector`, set below.
                _ => 0,
            },
            aspsr_value: note.aspsr,
            spsr_fixed: if entering { fixed } else { 0 },
            check_mask,
            check,
            chain: u64::from(same || view.cr3 != 0),
            cr3: view.cr3,
            root_at: view.root_at,
            root: view.root,
            stale_at: view.stale_at,
            view_at: view.view_at,
            view: view.view,
            ac: view.ac,
            vector: if note.kind == SUPERVISOR_CALL {
                note.pc
            } else {
                0
            },
            pan: u64::from(to.pan),
            fields_keep: [u64::MAX, 0, 0],
            fields_set,
            untag,
            event: self.cpu.event_storage(),
            close,
            scratch: 0,
            pointer_swap: u64::from(note.pointer.is_some()),
            pointer_generation: note.pointer.map_or(0, |(_, generation)| generation),
            pointer_from: note
                .pointer
                .map_or([0; x86_64_pointer::TAG_WORDS], |([from, _], _)| from),
            pointer_to: note
                .pointer
                .map_or([0; x86_64_pointer::TAG_WORDS], |([_, to], _)| to),
        };
        let slot = &mut self.mirrors.transitions.records[from_state][note.kind];
        *slot = record;
        if !guarded {
            slot.target = at(&slot.vector);
        }
    }

    /// Native code is entered with `budget` transfers in the PE's state,
    /// whose lookup contexts the fetch just took: records into this state
    /// whose contexts went take these, and may continue here again.
    pub(super) fn refresh_transition_contexts(&mut self, budget: u64) {
        let state = self.mirror_state();
        let table = &mut self.mirrors.transitions;
        if budget == 0 || table.stale_contexts >> state & 1 == 0 {
            return;
        }
        table.stale_contexts &= !(1 << state);
        let contexts = self.chain.contexts();
        let salts = self.chain.salts();
        let (generation, contexts_generation) = (table.generation, table.contexts_generation);
        for record in table.records.as_flattened_mut() {
            if record.generation == generation && record.to == state as u64 {
                record.contexts = contexts;
                record.salts = u64::from(salts[0]) | u64::from(salts[1]) << 32;
                record.contexts_generation = contexts_generation;
            }
        }
    }

    /// Native code returned after transitions: PSTATE's EL, guarded level
    /// and PAN are those it left, before anything reads them.
    pub(super) fn adopt_native_state(&mut self) {
        let state = self.mirrors.native_state();
        self.cpu.pstate.el = (state & 3) as u8;
        self.cpu.pstate.guarded = state & 4 != 0;
        self.cpu.pstate.pan = self.cpu.state.pan != 0;
    }

    /// What an engine-path transition does after its own effects
    /// ([`Self::transitioned`]) for the `counts` native code made, the
    /// last block's `ending` instructions not yet retired. The PAC service
    /// is native code's: records are made only where both states take the
    /// same keys, version and controls, and where the key enables differ
    /// (SVC, ERET) the thunk swaps the service's tag words to those the
    /// engine path reopened (`transition::check_pointer`); traps agree at
    /// both levels ([`Self::pointer_use`]). With counters that tell the levels
    /// apart a run ends at its first transition, so the counts before it
    /// stay the old level's.
    pub(super) fn adopt_transitions(&mut self, counts: [u64; KINDS], ending: u64) {
        for (total, count) in self.statistics.native_transitions.iter_mut().zip(counts) {
            *total += count;
        }
        // An SVC takes an exception (`Machine::take`).
        self.statistics.exceptions += counts[SUPERVISOR_CALL];
        self.cpu.retired += ending;
        self.forget_state_derived();
        self.cells_moved();
        self.cpu.retired -= ending;
    }

    /// Install the thunk completing the GENTER, GEXIT, SVC or ERET `word` that ends
    /// block `code` (its bytes `native`) at `pc`; the address of its exit
    /// to the target, which may be linked.
    pub(super) fn complete_transition_natively(
        &mut self,
        code: NativeCode,
        native: &[u8],
        pc: u64,
        word: u32,
    ) -> Option<u64> {
        let (kind, immediate) = transition_word(word)?;
        let site = native.len().checked_sub(chain::DIRECT_EXIT_BYTES)?;
        if !chain::is_direct_exit(&native[site..], GuestAddress(pc), ExitKind::System) {
            return None;
        }
        // GEXIT and ERET read the count (`Places::lines_until`).
        if matches!(kind, GUARDED_EXIT | EXCEPTION_RETURN) {
            self.clock.native(self.host.tsc_counter()?);
        }
        let pc_at = GuestAddress(pc);
        let thunk = match kind {
            GUARDED_ENTER => {
                let syndrome = Cpu::guarded_enter_syndrome(immediate as u8);
                let syndrome = u32::try_from(syndrome).ok()?;
                transition::guarded_enter_thunk(pc_at, syndrome, &TRANSITIONS)
            }
            GUARDED_EXIT => transition::guarded_exit_thunk(pc_at, &TRANSITIONS),
            SUPERVISOR_CALL => {
                let syndrome = crate::machine::exception::syndrome(
                    crate::machine::exception::Class::SupervisorCall,
                    u32::from(immediate),
                );
                let syndrome = u32::try_from(syndrome).ok()?;
                transition::supervisor_call_thunk(pc_at, syndrome, &TRANSITIONS)
            }
            _ => transition::exception_return_thunk(pc_at, &TRANSITIONS),
        };
        self.install_stream = self.stream_at(code.0 as u64);
        let installed = self.install_native(&thunk.code, true).ok()?;
        let site = code.0 as u64 + site as u64;
        let jump = chain::jump(site, installed.0 as u64)?;
        // Its part was chosen meanwhile: the instruction stays an exit.
        if !self.note_reference(site, installed.0 as u64) {
            return None;
        }
        self.patch_native(NativeCode(site as usize), &jump);
        let at = installed.0 as u64;
        thunk
            .dynamic
            .filter(|&exit| chain::dynamic_ready(&thunk.code, exit, at))
            .map(|exit| at + exit as u64)
    }
}

#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux")
))]
pub(super) mod tests {
    use super::super::NoDevices;
    use super::super::tests::{BASE, FakeHost, machine_on};
    use super::*;
    use stela_contracts::Chip;

    const GENTER_WORD: u32 = 0x0020_1423;
    const ENTER: u64 = BASE + 0x9000;

    /// A machine at EL2 with SPRR and EL2's GXF enabled and GXF_ENTER_EL2
    /// at [`ENTER`].
    fn gxf2() -> Machine<FakeHost, NoDevices> {
        let mut machine = machine_on(Chip::AppleM2, &[]);
        machine.cpu.pstate = Pstate::reset(2);
        machine.cpu.apple.sprr_config |= mmu::SPRR_CONFIG_EN;
        machine.cpu.apple.gxf_config[1] |= crate::machine::cpu::GXF_CONFIG_EN;
        machine.cpu.apple.guarded[1].enter = ENTER;
        machine.derived_changed();
        machine
    }

    /// Run `code` on the host on the machine's frame, entered and left as
    /// `Machine::execute` enters and leaves native code; the exit's PC and
    /// kind.
    pub(in crate::machine::engine) fn run_native(
        machine: &mut Machine<FakeHost, NoDevices>,
        code: &[u8],
    ) -> (u64, u64) {
        use core::ffi::c_void;
        unsafe extern "C" {
            fn mmap(
                address: *mut c_void,
                bytes: usize,
                protection: i32,
                flags: i32,
                fd: i32,
                offset: i64,
            ) -> *mut c_void;
            fn mprotect(address: *mut c_void, bytes: usize, protection: i32) -> i32;
            fn munmap(address: *mut c_void, bytes: usize) -> i32;
        }
        #[cfg(target_os = "macos")]
        const MAP_ANONYMOUS: i32 = 0x1000;
        #[cfg(target_os = "linux")]
        const MAP_ANONYMOUS: i32 = 0x20;
        const PAGE: usize = 4096;
        machine.cpu.state.daif = u32::from(machine.cpu.pstate.daif);
        machine.mirrors.selected = u64::from(machine.cpu.pstate.sp);
        machine.mirrors.monitor =
            u64::from(machine.cpu.monitor.is_some() || machine.cpu.has_canonical_memory());
        let state = machine.mirror_state();
        machine.mirrors.transition(state);
        machine.mirrors.enter(machine.cpu.pstate);
        machine.cpu.state.pan = u64::from(machine.cpu.pstate.pan);
        machine.enter_cycles();
        machine.cpu.state.anchors = machine.anchors();
        machine.cpu.state.lower_limit = machine.lower_limit();
        let code = stela_compiler::x86_64::pinned::entered(code);
        assert!(code.len() <= PAGE);
        // SAFETY: private anonymous memory, RX after the copy, unmapped after
        // the call; the thunk touches only the machine's frame and the data
        // its anchors and records name, which outlive it.
        let exit = unsafe {
            let page = mmap(core::ptr::null_mut(), PAGE, 3, 2 | MAP_ANONYMOUS, -1, 0);
            assert_ne!(page as usize, usize::MAX);
            core::ptr::copy_nonoverlapping(code.as_ptr(), page.cast(), code.len());
            assert_eq!(mprotect(page, PAGE, 5), 0);
            let entry: unsafe extern "sysv64" fn(*mut GuestState) -> [u64; 2] =
                core::mem::transmute(page);
            let exit = entry(&mut machine.cpu.state);
            assert_eq!(munmap(page, PAGE), 0);
            exit
        };
        let counts = machine.mirrors.transitions.take_counts();
        if counts != [0; KINDS] {
            machine.adopt_native_state();
        }
        machine.cpu.pstate.daif = machine.cpu.state.daif as u8;
        machine.cpu.pstate.sp = machine.mirrors.selected != 0;
        machine.mirrors.leave(&mut machine.cpu.pstate);
        if counts != [0; KINDS] {
            machine.adopt_transitions(counts, 0);
            machine.cpu.pc = exit[0];
        }
        (exit[0], exit[1])
    }

    /// Everything a transition changes, for comparison.
    fn observed(machine: &Machine<FakeHost, NoDevices>) -> alloc::string::String {
        let cpu = &machine.cpu;
        // The frame's copies of DAIF and PAN, its anchors and its lower
        // limit serve native runs alone: PSTATE holds the masks and PAN
        // between them, and the placement holds the limit.
        let state = GuestState {
            daif: 0,
            pan: 0,
            anchors: Default::default(),
            lower_limit: Default::default(),
            ..cpu.state.clone()
        };
        alloc::format!(
            "{:?} {:?} pc={:#x} banks={:?} sp={:#x},{:#x},{:#x} state={}",
            cpu.pstate,
            state,
            cpu.pc,
            cpu.apple.guarded,
            cpu.sp(0),
            cpu.sp(1),
            cpu.sp(2),
            machine.mirror_state(),
        )
    }

    /// Both twins: the operands the transitions bank and restore.
    fn prepare(machine: &mut Machine<FakeHost, NoDevices>, pc: u64) {
        machine.cpu.pc = pc;
        machine.cpu.state.nzcv = 0x6000_0000;
        machine.cpu.pstate.daif = 0b0011;
        machine.cpu.pstate.pan = true;
        machine.cpu.pstate.dit = true;
        machine.cpu.state.sp = 0xffff_fe00_1234_5670;
        machine.cpu.apple.guarded[1].sp = 0xffff_fe00_0777_0000;
    }

    /// Engine-path GEXITs count why their thunk would not have completed
    /// them: no record yet, a record forgotten, a deadline, a notification,
    /// a changed operand, or none (no thunk ran).
    #[test]
    fn engine_path_transitions_count_their_miss() {
        let pc = BASE + 0x100;
        let mut machine = gxf2();
        prepare(&mut machine, pc);
        let gexit = |machine: &mut Machine<FakeHost, NoDevices>| {
            machine.cpu.pc = pc;
            machine.system_instruction(GENTER_WORD).unwrap();
            machine.finish_transition(1);
            machine.system_instruction(GEXIT).unwrap();
            machine.finish_transition(1);
            machine.statistics.transition_misses[GUARDED_EXIT]
        };
        let mut expected = [0; miss::REASONS];
        expected[miss::NEVER] += 1;
        assert_eq!(gexit(&mut machine), expected);
        assert_eq!(
            machine.statistics.transition_misses[GUARDED_ENTER][miss::NEVER],
            1,
            "GENTER too"
        );
        machine.mirrors.lines_until = u64::MAX;
        expected[miss::OTHER] += 1;
        assert_eq!(gexit(&mut machine), expected);
        machine.mirrors.lines_until = 0;
        expected[miss::DEADLINE] += 1;
        assert_eq!(gexit(&mut machine), expected);
        machine.mirrors.lines_until = u64::MAX;
        machine.chain.weak_storage().upgrade().unwrap().notify();
        expected[miss::EPOCH] += 1;
        assert_eq!(gexit(&mut machine), expected);
        let epoch = machine.chain.snapshot().interrupt_epoch;
        machine.chain.acknowledge_interrupt(epoch);
        machine.mirrors.transitions.forget();
        expected[miss::FORGOTTEN] += 1;
        assert_eq!(gexit(&mut machine), expected);
    }

    /// Native GENTER and GEXIT from the records the engine path left change
    /// state exactly as the engine path does, and count as native.
    #[test]
    fn native_guarded_transitions_match_the_engine_path() {
        let pc = BASE + 0x100;
        let (mut engine, mut native) = (gxf2(), gxf2());
        // Twice: the first transitions set the prediction context up.
        for machine in [&mut engine, &mut native] {
            prepare(machine, pc);
            for _ in 0..2 {
                machine.system_instruction(GENTER_WORD).unwrap();
                machine.finish_transition(1);
                machine.system_instruction(GEXIT).unwrap();
                machine.finish_transition(1);
                machine.cpu.pc = pc;
            }
        }
        let table = &native.mirrors.transitions;
        let generation = table.generation;
        assert_eq!(table.records[2][GUARDED_ENTER].generation, generation);
        assert_eq!(table.records[6][GUARDED_EXIT].generation, generation);
        prepare(&mut engine, pc);
        prepare(&mut native, pc);
        // No deadline: GEXIT reads no count.
        native.mirrors.lines_until = u64::MAX;
        assert_eq!(observed(&engine), observed(&native));
        engine.system_instruction(GENTER_WORD).unwrap();
        let syndrome = Cpu::guarded_enter_syndrome(3) as u32;
        let enter = transition::guarded_enter_thunk(GuestAddress(pc), syndrome, &TRANSITIONS);
        let exit = run_native(&mut native, &enter.code);
        assert_eq!(exit, (ENTER, ExitKind::Return as u64));
        assert_eq!(observed(&engine), observed(&native));
        assert!(native.cpu.pstate.guarded);
        // GEXIT back, with the masks the GL left in SPSR changed meanwhile.
        for machine in [&mut engine, &mut native] {
            machine.cpu.pc = ENTER + 0x40;
            machine.cpu.state.nzcv = 0x8000_0000;
            machine.cpu.apple.guarded[1].spsr &= !(0b0001 << 6);
        }
        engine.system_instruction(GEXIT).unwrap();
        let leave = transition::guarded_exit_thunk(GuestAddress(ENTER + 0x40), &TRANSITIONS);
        let exit = run_native(&mut native, &leave.code);
        assert_eq!(exit, (pc + 4, ExitKind::Return as u64));
        assert_eq!(observed(&engine), observed(&native));
        assert!(!native.cpu.pstate.guarded && native.cpu.pstate.daif == 0b0010);
        assert_eq!(native.statistics.native_transitions, [0, 0, 1, 1]);
        // A waiting IRQ the return would unmask stays with the engine.
        native.cpu.state.pending = 0b10;
        native.cpu.pc = pc;
        native.system_instruction(GENTER_WORD).unwrap();
        native.cpu.pc = ENTER + 0x40;
        let before = observed(&native);
        native.cpu.apple.guarded[1].spsr &= !(0b0010 << 6);
        let after_write = observed(&native);
        assert_ne!(before, after_write);
        let exit = run_native(&mut native, &leave.code);
        assert_eq!(exit, (ENTER + 0x40, ExitKind::System as u64));
        assert_eq!(observed(&native), after_write);
    }

    /// A write of SPRR's field registers (`S3_6_C15_C3_0`), which SPTM
    /// makes at every address-space switch, keeps the records: no access
    /// and nothing derived reads them; as does a stored IMPLEMENTATION
    /// DEFINED register's.
    #[test]
    fn stored_register_writes_keep_the_records() {
        const MSR_SPRR_FIELDS_0: u32 = 0xd51e_f300; // msr s3_6_c15_c3_0, x0
        let mut machine = gxf2();
        machine.cpu.pc = BASE + 0x100;
        for _ in 0..2 {
            machine.system_instruction(GENTER_WORD).unwrap();
            machine.finish_transition(1);
            machine.system_instruction(GEXIT).unwrap();
            machine.finish_transition(1);
        }
        let generation = machine.mirrors.transitions.generation;
        assert_eq!(
            machine.mirrors.transitions.records[2][GUARDED_ENTER].generation,
            generation
        );
        for value in [0x40800, 0] {
            machine.cpu.state.x[0] = value;
            machine.system_instruction(MSR_SPRR_FIELDS_0).unwrap();
            assert_eq!(machine.cpu.apple.sprr_fields[0], value);
            assert_eq!(machine.mirrors.transitions.generation, generation);
        }
        // So does a write of an IMPLEMENTATION DEFINED register a store
        // takes in full: XNU's thread keys at each thread switch.
        const MSR_THREAD_KEY_0: u32 = 0xd51c_fdc6; // msr s3_4_c15_c13_6, x6
        machine.cpu.state.x[6] = 0x1234_5678;
        machine.system_instruction(MSR_THREAD_KEY_0).unwrap();
        assert_eq!(machine.mirrors.transitions.generation, generation);
    }

    /// MSR APCTL_EL1, x0 (`S3_4_C15_C0_4`), which SPTM makes on entry and
    /// exit and XNU at thread switches.
    const MSR_APCTL: u32 = 0xd51c_f080;
    /// APCTL's kernel-keys bit (`pointer_authentication::APCTL_KERNEL_KEYS`).
    const KERNEL_KEYS: u64 = 1 << 4;

    /// A write of PAC's inputs (APCTL, a key) keeps the GENTER and GEXIT
    /// records, whose two sides share every PAC input, and drops the SVC and
    /// ERET records, whose two levels' PAC use it may set apart.
    #[test]
    fn pointer_writes_keep_only_the_guarded_records() {
        let mut machine = gxf2();
        machine.cpu.pc = BASE + 0x100;
        for _ in 0..2 {
            machine.system_instruction(GENTER_WORD).unwrap();
            machine.finish_transition(1);
            machine.system_instruction(GEXIT).unwrap();
            machine.finish_transition(1);
            machine.cpu.pc = BASE + 0x100;
        }
        let table = &mut machine.mirrors.transitions;
        let generation = table.generation;
        assert_eq!(table.records[2][GUARDED_ENTER].generation, generation);
        assert_eq!(table.records[6][GUARDED_EXIT].generation, generation);
        // As an SVC from the host's EL0 and an ERET from EL2 would leave.
        table.records[0][SUPERVISOR_CALL].generation = generation;
        table.records[2][EXCEPTION_RETURN].generation = generation;
        let holds = |machine: &Machine<FakeHost, NoDevices>, state: usize, kind: usize| {
            let table = &machine.mirrors.transitions;
            table.records[state][kind].generation == table.generation
        };
        machine.cpu.state.x[0] = KERNEL_KEYS;
        machine.system_instruction(MSR_APCTL).unwrap();
        assert_eq!(machine.cpu.apple.apctl[1], KERNEL_KEYS);
        assert_ne!(machine.mirrors.transitions.generation, generation);
        assert!(holds(&machine, 2, GUARDED_ENTER) && holds(&machine, 6, GUARDED_EXIT));
        assert!(!holds(&machine, 0, SUPERVISOR_CALL) && !holds(&machine, 2, EXCEPTION_RETURN));
        // An architectural key's write alike.
        machine
            .cpu
            .registers
            .set(SystemRegister::APIAKeyLo_EL1, 0, 0x5555);
        machine.register_written(SystemRegister::APIAKeyLo_EL1);
        assert!(holds(&machine, 2, GUARDED_ENTER) && holds(&machine, 6, GUARDED_EXIT));
        // Any other derived change still drops them.
        machine.derived_changed();
        assert!(!holds(&machine, 2, GUARDED_ENTER) && !holds(&machine, 6, GUARDED_EXIT));
    }

    /// A native GENTER, an APCTL write at GL2 that turns the kernel keys
    /// off, and a native GEXIT leave the PE, the PAC service included, as
    /// the engine path does.
    #[test]
    fn native_guarded_transitions_around_an_apctl_write_match_the_engine_path() {
        let pc = BASE + 0x100;
        let (mut engine, mut native) = (gxf2(), gxf2());
        for machine in [&mut engine, &mut native] {
            machine.cpu.apple.apctl[1] = KERNEL_KEYS;
            // EL2's own keys (S3_6_C15_C13_0 onward), every half known.
            for index in 0..10u8 {
                let encoding = Encoding::new(3, 6, 15, 13 + index / 8, index % 8);
                let register = crate::machine::pointer_keys::Register::decode(encoding).unwrap();
                machine
                    .cpu
                    .apple
                    .pointer_keys
                    .write(register, 0x1111 * u64::from(index + 1));
            }
            machine.derived_changed();
            prepare(machine, pc);
            for _ in 0..2 {
                machine.system_instruction(GENTER_WORD).unwrap();
                machine.finish_transition(1);
                machine.system_instruction(GEXIT).unwrap();
                machine.finish_transition(1);
                machine.cpu.pc = pc;
            }
            prepare(machine, pc);
            machine.revalidate_pointers();
        }
        native.mirrors.lines_until = u64::MAX;
        let pointer = |machine: &Machine<FakeHost, NoDevices>| {
            (machine.pointer.tags, machine.pointer.generation)
        };
        assert_eq!(pointer(&engine), pointer(&native));
        assert!(native.pointer.tags.iter().any(|&tag| tag != 0), "keys open");
        engine.system_instruction(GENTER_WORD).unwrap();
        let syndrome = Cpu::guarded_enter_syndrome(3) as u32;
        let enter = transition::guarded_enter_thunk(GuestAddress(pc), syndrome, &TRANSITIONS);
        assert_eq!(
            run_native(&mut native, &enter.code),
            (ENTER, ExitKind::Return as u64)
        );
        for machine in [&mut engine, &mut native] {
            machine.cpu.pc = ENTER;
            machine.cpu.state.x[0] = 0;
            machine.system_instruction(MSR_APCTL).unwrap();
            assert_eq!(machine.cpu.apple.apctl[1], 0);
            machine.cpu.pc = ENTER + 0x40;
        }
        assert_eq!(observed(&engine), observed(&native));
        assert_eq!(pointer(&engine), pointer(&native));
        engine.system_instruction(GEXIT).unwrap();
        let leave = transition::guarded_exit_thunk(GuestAddress(ENTER + 0x40), &TRANSITIONS);
        assert_eq!(
            run_native(&mut native, &leave.code),
            (pc + 4, ExitKind::Return as u64)
        );
        assert_eq!(observed(&engine), observed(&native));
        assert_eq!(pointer(&engine), pointer(&native));
        assert_eq!(native.statistics.native_transitions, [0, 0, 1, 1]);
    }

    /// Any change of what records depend on drops them; lookup contexts
    /// going only keeps them from continuing in their new state until the
    /// next native entry there gives them its contexts.
    #[test]
    fn derived_changes_forget_the_records_and_contexts_refresh() {
        let mut machine = gxf2();
        let generation = machine.mirrors.transitions.generation;
        machine.derived_changed();
        assert_ne!(machine.mirrors.transitions.generation, generation);
        machine.cpu.pc = BASE + 0x100;
        for _ in 0..2 {
            machine.system_instruction(GENTER_WORD).unwrap();
            machine.finish_transition(1);
            machine.system_instruction(GEXIT).unwrap();
            machine.finish_transition(1);
        }
        let table = &machine.mirrors.transitions;
        let (generation, contexts) = (table.generation, table.contexts_generation);
        assert_eq!(
            table.records[2][GUARDED_ENTER].contexts_generation,
            contexts
        );
        machine.advance_translation_generation();
        let table = &machine.mirrors.transitions;
        assert_eq!(table.generation, generation);
        assert_ne!(table.contexts_generation, contexts);
        assert_ne!(
            table.records[2][GUARDED_ENTER].contexts_generation,
            table.contexts_generation
        );
        // In GL2, the target of GENTER from EL2.
        machine.chain.set_contexts([0x51, 0x52]);
        machine.system_instruction(GENTER_WORD).unwrap();
        machine.refresh_transition_contexts(1);
        let table = &machine.mirrors.transitions;
        let record = &table.records[2][GUARDED_ENTER];
        assert_eq!(record.contexts_generation, table.contexts_generation);
        assert_eq!(record.contexts, [0x51, 0x52]);
        assert_ne!(
            table.records[6][GUARDED_EXIT].contexts_generation,
            table.contexts_generation
        );
    }

    /// A timer that falls due while the PE is at GL2 (every interrupt
    /// masked) is taken right after the native GEXIT's unmask, as after the
    /// engine path's, and so is another PE's interrupt: the thunk leaves the
    /// unmask to the engine once the count reaches the step's deadline or
    /// the interrupt epoch moved. Before the deadline it completes natively.
    #[test]
    fn interrupts_due_in_gl2_are_taken_at_a_native_gexit_as_at_the_engine_path() {
        use crate::machine::Lines;
        use core::sync::atomic::Ordering::Relaxed;
        use stela_compiler::x86_64::register_read::{Count, CountState};
        const CNTV_CVAL_X0: u32 = 0xd51b_e340;
        const CNTV_CTL_X0: u32 = 0xd51b_e320;
        const DUE: u64 = 1500;
        /// What happens while the PE is at GL2.
        type Cause = fn(&mut Machine<FakeHost, NoDevices>, &CountState);
        let pc = BASE + 0x100;
        let leave = transition::guarded_exit_thunk(GuestAddress(ENTER + 0x40), &TRANSITIONS);
        let syndrome = Cpu::guarded_enter_syndrome(3) as u32;
        let enter = transition::guarded_enter_thunk(GuestAddress(pc), syndrome, &TRANSITIONS);
        // A machine with records, the virtual timer armed for DUE and FIQ
        // unmasked at EL2, at the GENTER at `pc`. Records hold for the
        // frame they were made with: the machine stays where it is.
        let armed = |machine: &mut Machine<FakeHost, NoDevices>| {
            // The timer first: its writes drop records.
            machine.cpu.state.x[0] = DUE;
            machine.system_instruction(CNTV_CVAL_X0).unwrap();
            machine.cpu.state.x[0] = 1;
            machine.system_instruction(CNTV_CTL_X0).unwrap();
            prepare(machine, pc);
            for _ in 0..2 {
                machine.system_instruction(GENTER_WORD).unwrap();
                machine.finish_transition(1);
                machine.system_instruction(GEXIT).unwrap();
                machine.finish_transition(1);
                machine.cpu.pc = pc;
            }
            machine.cpu.pstate.daif = 0b0010;
            machine.host.now = 100;
            let quiet = Lines {
                irq: false,
                fiq: false,
            };
            machine.device_lines = Some((quiet, u64::MAX, machine.current));
        };
        // The engine path: GENTER, the timer falls due, GEXIT, the FIQ.
        let mut engine = gxf2();
        armed(&mut engine);
        engine.system_instruction(GENTER_WORD).unwrap();
        engine.cpu.pc = ENTER + 0x40;
        engine.host.now = DUE;
        assert!(!engine.deliver_interrupts(), "GL2 masks the FIQ");
        engine.system_instruction(GEXIT).unwrap();
        assert!(engine.deliver_interrupts(), "the FIQ after GEXIT");
        // The native path, its count read from `clock`.
        let clock = alloc::boxed::Box::new(CountState::default());
        clock.ceiling.store(u64::MAX, Relaxed);
        let native_path = |native: &mut Machine<FakeHost, NoDevices>, cause: Cause| {
            armed(native);
            let reader = native.clock.reader() as *mut Count;
            // SAFETY: the machine's own reader, which no native code runs
            // over meanwhile; `clock` outlives every run here.
            unsafe {
                (*reader).per_tsc = 0;
                (*reader).last = 0;
                (*reader).state = &*clock as *const CountState as u64;
            }
            clock.offset.store(100, Relaxed);
            assert_eq!(run_native(native, &enter.code).0, ENTER);
            native.cpu.pc = ENTER + 0x40;
            // The step before the run at GL2 finds the timer's deadline.
            assert!(!native.deliver_interrupts());
            assert_eq!(native.mirrors.lines_until, DUE);
            cause(native, &clock);
        };
        // The timer fell due, or another PE interrupted: the engine unmasks.
        let causes: [Cause; 2] = [
            |_, clock| clock.offset.store(DUE, Relaxed),
            |machine, _| {
                machine
                    .chain
                    .weak_storage()
                    .upgrade()
                    .unwrap()
                    .interrupt_epoch()
                    .fetch_add(1, Relaxed);
            },
        ];
        for (index, cause) in causes.into_iter().enumerate() {
            let mut native = gxf2();
            native_path(&mut native, cause);
            let exit = run_native(&mut native, &leave.code);
            assert_eq!(
                exit,
                (ENTER + 0x40, ExitKind::System as u64),
                "cause {index}"
            );
            native.system_instruction(GEXIT).unwrap();
            native.host.now = DUE;
            if index == 0 {
                assert!(native.deliver_interrupts(), "the FIQ after GEXIT");
                assert_eq!(observed(&engine), observed(&native));
            }
        }
        // Before the deadline, with no other interrupt, it completes.
        let mut native = gxf2();
        native_path(&mut native, |_, clock| clock.offset.store(DUE - 1, Relaxed));
        assert_eq!(
            run_native(&mut native, &leave.code),
            (pc + 4, ExitKind::Return as u64)
        );
    }

    /// The pieces of one native run, laid out as an installer would and
    /// joined by `links` (piece, offset of its patch point, target piece):
    /// the body `run_native` takes, entered at the first piece.
    fn joined(pieces: &[&[u8]], links: &[(usize, usize, usize)]) -> alloc::vec::Vec<u8> {
        let header = stela_compiler::x86_64::pinned::entered(&[]).len() as u64;
        let mut at = alloc::vec::Vec::new();
        let mut end = chain::JUMP_BYTES as u64;
        for piece in pieces {
            let start = (header + end).next_multiple_of(64) - header;
            at.push(start);
            end = start + piece.len() as u64;
        }
        let mut body = alloc::vec![0xcc_u8; end as usize];
        body[..chain::JUMP_BYTES].copy_from_slice(&chain::jump(0, at[0]).unwrap());
        for (piece, &start) in pieces.iter().zip(&at) {
            body[start as usize..][..piece.len()].copy_from_slice(piece);
        }
        for &(from, point, to) in links {
            let site = at[from] + point as u64;
            let jump = chain::jump(site, at[to]).unwrap();
            body[site as usize..][..jump.len()].copy_from_slice(&jump);
        }
        body
    }

    /// An installed thunk's patch point of its dynamic exit, in it.
    fn dynamic_point(thunk: &stela_compiler::x86_64::pointer::Thunk) -> usize {
        let header = stela_compiler::x86_64::pinned::entered(&[]).len() as u64;
        // `joined` places each piece 64-aligned in the page.
        let at = (header + 64).next_multiple_of(64);
        let exit = thunk.dynamic.expect("a dynamic exit");
        assert!(chain::dynamic_ready(&thunk.code, exit, at));
        (chain::dynamic_patch(at + exit as u64) - at) as usize
    }

    /// An in-block MRS TPIDR_EL1 (`x86_64::mirrored_read`) reads the value
    /// the last MSR wrote whatever native GENTER and GEXIT ran around it:
    /// before or after the write, in other runs or chained in the same run
    /// as the reads, and with the write made in the guarded level.
    #[test]
    fn inline_thread_pointer_reads_follow_writes_across_native_transitions() {
        const MRS_X8: u32 = 0xd538_d088; // mrs x8, tpidr_el1
        const MRS_X9: u32 = 0xd538_d089; // mrs x9, tpidr_el1
        const MRS_X10: u32 = 0xd538_d08a; // mrs x10, tpidr_el1
        const MSR_X11: u32 = 0xd518_d08b; // msr tpidr_el1, x11
        const ADD: u32 = 0x9100_0400; // add x0, x0, #1
        let pc = BASE + 0x100;
        let thread = MIRRORED
            .iter()
            .position(|&encoding| encoding == registers::encodings::TPIDR_EL1)
            .unwrap();
        let mut machine = gxf2();
        prepare(&mut machine, pc);
        for _ in 0..2 {
            machine.system_instruction(GENTER_WORD).unwrap();
            machine.finish_transition(1);
            machine.system_instruction(GEXIT).unwrap();
            machine.finish_transition(1);
            machine.cpu.pc = pc;
        }
        // Each run may continue past its transitions, with no deadline
        // (GEXIT reads no count).
        machine.mirrors.lines_until = u64::MAX;
        machine.mirrors.transitions.chain = 1;
        machine.mirrors.transitions.records[2][GUARDED_ENTER].chain = 1;
        machine.mirrors.transitions.records[6][GUARDED_EXIT].chain = 1;
        // Open the cells of EL2 and GL2 on the first value.
        let write = |machine: &mut Machine<FakeHost, NoDevices>, value: u64| {
            machine.cpu.state.x[11] = value;
            machine.system_instruction(MSR_X11).unwrap();
            assert_eq!(
                machine.cpu.registers.get(SystemRegister::TPIDR_EL1, 0),
                value
            );
        };
        write(&mut machine, 0x1000);
        machine.system_instruction(MRS_X8).unwrap();
        machine.system_instruction(GENTER_WORD).unwrap();
        machine.system_instruction(MRS_X8).unwrap();
        machine.system_instruction(GEXIT).unwrap();
        assert_eq!(machine.mirror_state(), 2);
        for state in [2, 6] {
            let cell = machine.mirrors.banks[state][thread];
            assert_eq!((cell.readable, cell.value), (1, 0x1000), "state {state}");
        }
        // Blocks as the engine compiles them: A at `pc` ends in GENTER,
        // B at its target in GEXIT, C after GENTER falls through.
        let options = stela_compiler::Options::new(
            stela_compiler::block::BlockProfile::new(stela_compiler::block::Features::SYSTEM_MODE),
            stela_compiler::TargetCpu::detect(),
        )
        .with_memory_checks(machine.memory_checks())
        .with_native_system(machine.native_system());
        let compile = |at: u64, words: &[u32]| {
            let bytes: alloc::vec::Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            stela_compiler::compile(&bytes, GuestAddress(at), &options)
                .unwrap()
                .native
                .x86_64_code
        };
        let (genter, gexit, after) = (pc + 4, ENTER + 4, pc + 8);
        let a = compile(pc, &[MRS_X8, GENTER_WORD]);
        let b = compile(ENTER, &[MRS_X9, GEXIT]);
        let c = compile(after, &[MRS_X10, ADD]);
        let site = |code: &[u8], at: u64| {
            let site = code.len() - chain::DIRECT_EXIT_BYTES;
            assert!(chain::is_direct_exit(
                &code[site..],
                GuestAddress(at),
                ExitKind::System
            ));
            site
        };
        let (a_site, b_site) = (site(&a, genter), site(&b, gexit));
        let syndrome = Cpu::guarded_enter_syndrome(3) as u32;
        let enter = transition::guarded_enter_thunk(GuestAddress(genter), syndrome, &TRANSITIONS);
        let leave = transition::guarded_exit_thunk(GuestAddress(gexit), &TRANSITIONS);
        let (enter_point, leave_point) = (dynamic_point(&enter), dynamic_point(&leave));
        let reads = |machine: &mut Machine<FakeHost, NoDevices>| {
            let x = &mut machine.cpu.state.x;
            (x[8], x[9], x[10]) = (0, 0, 0);
        };
        let natives = |machine: &Machine<FakeHost, NoDevices>| {
            machine.statistics.native_transitions[2..].to_vec()
        };
        let fallthrough = (after + 8, ExitKind::Fallthrough as u64);

        // 1. MSR, then a native round trip, then the reads: every read in
        // one run, A to C through both thunks.
        write(&mut machine, 0x2000);
        reads(&mut machine);
        machine.cpu.pc = pc;
        let run = joined(
            &[&a, &enter.code, &b, &leave.code, &c],
            &[
                (0, a_site, 1),
                (1, enter_point, 2),
                (2, b_site, 3),
                (3, leave_point, 4),
            ],
        );
        assert_eq!(run_native(&mut machine, &run), fallthrough);
        assert_eq!(natives(&machine), [1, 1]);
        let x = &machine.cpu.state.x;
        assert_eq!((x[8], x[9], x[10]), (0x2000, 0x2000, 0x2000));
        assert_eq!(machine.mirror_state(), 2);

        // 2. The reverse: a native round trip, then MSR, then the reads,
        // each read in a run of its own after a thunk.
        let round = joined(
            &[&enter.code, &b, &leave.code, &c],
            &[(0, enter_point, 1), (1, b_site, 2), (2, leave_point, 3)],
        );
        machine.cpu.pc = genter;
        assert_eq!(run_native(&mut machine, &round), fallthrough);
        write(&mut machine, 0x3000);
        reads(&mut machine);
        machine.cpu.pc = after;
        assert_eq!(run_native(&mut machine, &c), fallthrough);
        assert_eq!(machine.cpu.state.x[10], 0x3000);
        // GENTER's thunk, then B in GL2 on the cells it named.
        let entering = joined(&[&enter.code, &b], &[(0, enter_point, 1)]);
        machine.cpu.pc = genter;
        assert_eq!(
            run_native(&mut machine, &entering),
            (gexit, ExitKind::System as u64)
        );
        assert_eq!(machine.cpu.state.x[9], 0x3000);
        assert_eq!(machine.mirror_state(), 6);

        // 3. A write in GL2, between native GENTER and GEXIT, then GEXIT's
        // thunk and C on the cells it named.
        write(&mut machine, 0x4000);
        reads(&mut machine);
        let leaving = joined(&[&leave.code, &c], &[(0, leave_point, 1)]);
        machine.cpu.pc = gexit;
        assert_eq!(run_native(&mut machine, &leaving), fallthrough);
        assert_eq!(machine.cpu.state.x[10], 0x4000);
        assert_eq!(machine.mirror_state(), 2);

        // 4. Chained again after that, every read takes the last value.
        reads(&mut machine);
        machine.cpu.pc = pc;
        assert_eq!(run_native(&mut machine, &run), fallthrough);
        let x = &machine.cpu.state.x;
        assert_eq!((x[8], x[9], x[10]), (0x4000, 0x4000, 0x4000));

        // 5. A cell a change closed sends its read to the engine, at the
        // MRS, wherever native transitions left the run.
        machine.mirrors.close_all();
        machine.cpu.pc = pc;
        assert_eq!(
            run_native(&mut machine, &run),
            (pc, ExitKind::System as u64)
        );
    }

    /// A run chained A -> GENTER -> B (GL2) -> GEXIT -> C stops at GEXIT,
    /// leaving the unmask to the engine, when another PE's mail moved the
    /// interrupt epoch or the count reached the step's deadline while the
    /// run was at GL2; otherwise it runs through to C.
    #[test]
    fn chained_runs_stop_at_gexit_for_mail_and_due_deadlines() {
        use core::sync::atomic::Ordering::Relaxed;
        use stela_compiler::x86_64::register_read::{Count, CountState};
        const ADD: u32 = 0x9100_0400; // add x0, x0, #1
        const DUE: u64 = 1500;
        let pc = BASE + 0x100;
        let (genter, gexit, after) = (pc + 4, ENTER + 4, pc + 8);
        let clock = alloc::boxed::Box::new(CountState::default());
        clock.ceiling.store(u64::MAX, Relaxed);
        for case in 0..3 {
            let mut machine = gxf2();
            prepare(&mut machine, pc);
            for _ in 0..2 {
                machine.system_instruction(GENTER_WORD).unwrap();
                machine.finish_transition(1);
                machine.system_instruction(GEXIT).unwrap();
                machine.finish_transition(1);
                machine.cpu.pc = pc;
            }
            machine.mirrors.transitions.chain = 1;
            machine.mirrors.transitions.records[2][GUARDED_ENTER].chain = 1;
            machine.mirrors.transitions.records[6][GUARDED_EXIT].chain = 1;
            machine.mirrors.lines_until = DUE;
            let reader = machine.clock.reader() as *mut Count;
            // SAFETY: the machine's own reader, which no native code runs
            // over meanwhile; `clock` outlives every run here.
            unsafe {
                (*reader).per_tsc = 0;
                (*reader).last = 0;
                (*reader).state = &*clock as *const CountState as u64;
            }
            clock
                .offset
                .store(if case == 2 { DUE } else { DUE - 1 }, Relaxed);
            if case == 1 {
                machine
                    .chain
                    .weak_storage()
                    .upgrade()
                    .unwrap()
                    .interrupt_epoch()
                    .fetch_add(1, Relaxed);
            }
            let options = stela_compiler::Options::new(
                stela_compiler::block::BlockProfile::new(
                    stela_compiler::block::Features::SYSTEM_MODE,
                ),
                stela_compiler::TargetCpu::detect(),
            )
            .with_memory_checks(machine.memory_checks())
            .with_native_system(machine.native_system());
            let compile = |at: u64, words: &[u32]| {
                let bytes: alloc::vec::Vec<u8> =
                    words.iter().flat_map(|w| w.to_le_bytes()).collect();
                stela_compiler::compile(&bytes, GuestAddress(at), &options)
                    .unwrap()
                    .native
                    .x86_64_code
            };
            let a = compile(pc, &[ADD, GENTER_WORD]);
            let b = compile(ENTER, &[ADD, GEXIT]);
            let c = compile(after, &[ADD, ADD]);
            let site = |code: &[u8]| code.len() - chain::DIRECT_EXIT_BYTES;
            let syndrome = Cpu::guarded_enter_syndrome(3) as u32;
            let enter =
                transition::guarded_enter_thunk(GuestAddress(genter), syndrome, &TRANSITIONS);
            let leave = transition::guarded_exit_thunk(GuestAddress(gexit), &TRANSITIONS);
            let run = joined(
                &[&a, &enter.code, &b, &leave.code, &c],
                &[
                    (0, site(&a), 1),
                    (1, dynamic_point(&enter), 2),
                    (2, site(&b), 3),
                    (3, dynamic_point(&leave), 4),
                ],
            );
            machine.cpu.state.x[0] = 0;
            let exit = run_native(&mut machine, &run);
            if case == 0 {
                assert_eq!(exit, (after + 8, ExitKind::Fallthrough as u64));
                assert_eq!(machine.statistics.native_transitions[2..], [1, 1]);
            } else {
                assert_eq!(exit, (gexit, ExitKind::System as u64), "case {case}");
                assert_eq!(machine.mirror_state(), 6, "case {case}");
                assert_eq!(machine.statistics.native_transitions[2..], [1, 0]);
            }
        }
    }
}
