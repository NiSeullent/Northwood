//! System register reads that native code completes from mirror cells.

use super::*;

/// The System registers whose MRS native code completes from a mirror
/// cell (`x86_64::register_read`): reads without side effects whose result
/// and plainness follow from System registers and PSTATE alone, so a read
/// the engine found plain stays plain, with the same value, until derived
/// state may change ([`Machine::derived_changed`]). They are the thread
/// pointers, Apple's GL2 thread pointer and GXF status, the GL1 exception
/// registers a guarded level saves and restores ([`GUARDED_MIRRORS`]),
/// SPRR_PERM_EL0 ([`SPRR_PERM_MIRROR`]) and DAIF; and the virtual count and
/// Apple's alias of it, whose cells hold the offset they subtract from the
/// clock's count ([`COUNT_MIRRORS`]); CurrentEL and MPIDR_EL1, which each
/// state's cell answers for good; and the cycle and instruction counters a
/// kernel reads around each exception to charge threads, which a transition
/// brings up to date ([`COUNTER_MIRRORS`]).
pub(super) const MIRRORED: [Encoding; 15] = {
    use registers::encodings::{
        ASPSR_GL1, CNTVCT_ALIAS_EL0, CURRENT_EL, DAIF, ELR_GL1, GXF_STATUS_EL1, MPIDR_EL1,
        SPRR_PERM_EL0, SPSR_GL1, TPIDR_EL1, TPIDR_EL2, TPIDR_GL2,
    };
    use stela_compiler::aarch64::system::registers::CNTVCT_EL0;
    [
        TPIDR_EL1,
        TPIDR_EL2,
        TPIDR_GL2,
        GXF_STATUS_EL1,
        SPSR_GL1,
        ASPSR_GL1,
        ELR_GL1,
        CURRENT_EL,
        super::super::pmu::PMC[0],
        super::super::pmu::PMC[1],
        MPIDR_EL1,
        SPRR_PERM_EL0,
        CNTVCT_EL0,
        CNTVCT_ALIAS_EL0,
        DAIF,
    ]
};

/// Rt 31 in MRS: XZR, which discards the value.
pub(super) const ZERO_REGISTER: u8 = 31;

/// MRS's L bit (21), which makes an MSR (register) word the MRS of the
/// same register and Xt.
pub(super) const MRS_READ: u32 = 1 << 21;

/// The registers read other than [`MIRRORED`] ones that get a cell of their
/// own the first time code naming them is installed
/// ([`Machine::stored_cell`]), which reads the register's live storage: the
/// ones a kernel reads around each exception and thread switch, FPCR and
/// FPSR, and the core counters a sampler reads in bursts
/// ([`Devices::register_storage`]). When there are more, the rest take System
/// exits. Cells go to registers in the order code naming them is first
/// installed, which for a kernel is boot's, so there are cells for every
/// register a boot reads: macOS reads over 64 before its first process, and
/// its processes' every MRS of TPIDRRO_EL0, their thread pointer, then took
/// an exit, over 100,000 a second on each PE running them.
pub(super) const STORED_CELLS: usize = 256;

/// The cells of the [`MIRRORED`] registers, in its order, and the
/// [`STORED_CELLS`]: native code reads those of the PE's EL and GXF state,
/// the bank of `banks` that `state` names. A read's value and plainness
/// follow from System registers and PSTATE, of which an exception entry or
/// return, GENTER or GEXIT changes these and the masks alone: each state
/// keeps its cells, which it finds again ([`Mirrors::transition`]). Only
/// SP_EL0 follows SPSel, whose thunks read `selected`, PSTATE.SP while native
/// code runs, which can change it
/// (`x86_64::register_read::stack_select_thunk`).
/// Anything else that changes derived state closes every cell of every bank.
/// The last of them says whether writes of the masks complete natively
/// ([`MASK_WRITES`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Mirrors {
    pub(super) banks: [[register_read::Cell; CELLS]; MIRROR_STATES],
    state: usize,
    /// A bit per bank that was the PE's since cells last closed, which only
    /// those can hold open.
    entered: u8,
    pub(super) selected: u64,
    /// PSTATE.DIT, UAO and SSBS as their registers read, in
    /// [`PSTATE_FIELDS`] order, while native code runs: their reads and
    /// writes complete here ([`pstate_field`]).
    pub(super) fields: [u64; 3],
    /// What a native ISR_EL1 read checks
    /// (`x86_64::register_read::interrupt_status_thunk`): `readable` the
    /// count before which time alone leaves the interrupt lines of the last
    /// step as they were (0 while the read is not the lines' alone),
    /// `value` the interrupt epoch they reflect
    /// ([`Machine::refresh_interrupt_status`]).
    pub(super) interrupt_status: register_read::Cell,
    /// What a native DSB checks (`x86_64::register_read::barrier_thunk`):
    /// open while this PE has no maintenance outstanding, whatever its
    /// state or derived state ([`Machine::open_barrier`]).
    pub(super) barrier: register_read::Cell,
    /// What native SEVL and WFE complete on
    /// (`x86_64::register_read::send_event_thunk`, `event_wait_thunk`): open
    /// on the PE's event register as native code is entered
    /// ([`Machine::enter_events`]). Neither depends on any System register
    /// or the PE's state: SEVL never traps, and a WFE that finds the event
    /// register set clears it and completes before any trap check (Arm
    /// `Hint_WFE`); one that finds it clear exits for the engine to wait.
    pub(super) events: register_read::Cell,
    /// What native CLREX completes on
    /// (`x86_64::register_read::clear_exclusive_thunk`): open as native
    /// code is entered while the engine holds no monitor of its own and
    /// traces none ([`Machine::enter_events`]), its `value` the PE's
    /// reservation word (`Cpu::reservation_word`), which a held canonical
    /// reservation makes nonzero. CLREX never traps, so nothing else
    /// decides it.
    pub(super) exclusive: register_read::Cell,
    /// The records native GENTER and GEXIT complete from
    /// (`x86_64::transition`), indexed by `state`, which they move.
    pub(super) transitions: transition::Table,
    /// The count before which time alone changes no interrupt line, as the
    /// last step at EL2 found it (0 elsewhere): native GEXIT unmasks only
    /// before it ([`Machine::refresh_interrupt_status`]).
    pub(super) lines_until: u64,
    /// The thread switches native TPIDR_EL1 writes made since native code
    /// was entered, which the engine takes when it returns
    /// ([`Machine::take_thread_switches`]).
    pub(super) switches: register_read::SwitchLog,
    /// Nonzero while the engine holds an exclusive monitor of its own
    /// (`Cpu::monitor`), set as native code is entered: native ERET then
    /// leaves clearing it to the engine. A canonical reservation has its
    /// own word, which the exclusive service keeps (`Record::reservation`).
    pub(super) monitor: u64,
    /// The performance monitor's cycle counter as native code is entered
    /// ([`Machine::enter_cycles`]), from which a cycle counter read computes
    /// the count where only its cell's value went stale.
    pub(super) cycles: register_read::CycleCounter,
    /// The performance monitor's instruction counter as native code is
    /// entered ([`Machine::enter_cycles`]), from which an instruction
    /// counter read computes the count where only its cell's value went
    /// stale.
    pub(super) instructions: register_read::InstructionCounter,
}

/// Where native code finds the cell at `index` of the PE's state.
pub(super) const fn cell_place(index: usize) -> Place {
    Place::new(Anchor::Cells, index * size_of::<register_read::Cell>())
}

/// Where native code finds [`Mirrors::selected`].
pub(super) const SELECTED: Place = Place::new(Anchor::Mirrors, offset_of!(Mirrors, selected));

/// Where native code finds [`Mirrors::interrupt_status`].
pub(super) const INTERRUPT_STATUS: Place =
    Place::new(Anchor::Mirrors, offset_of!(Mirrors, interrupt_status));

/// Where native code finds [`Mirrors::barrier`].
pub(super) const BARRIER: Place = Place::new(Anchor::Mirrors, offset_of!(Mirrors, barrier));

/// Where native code finds [`Mirrors::events`].
pub(super) const EVENTS: Place = Place::new(Anchor::Mirrors, offset_of!(Mirrors, events));

/// Where native code finds [`Mirrors::exclusive`].
pub(super) const CLEAR_EXCLUSIVE: Place =
    Place::new(Anchor::Mirrors, offset_of!(Mirrors, exclusive));

/// MRS ISR_EL1 to Xt, Rt in bits 4:0.
pub(super) const READ_INTERRUPT_STATUS: u32 = 0xd538_c100;

/// Xt of the MRS ISR_EL1 `word`, if it is one.
pub(super) fn interrupt_status_read(word: u32) -> Option<u8> {
    (word & !31 == READ_INTERRUPT_STATUS).then_some((word & 31) as u8)
}

/// The cell of MSR SPSel, open once one completed plainly in the state:
/// its `value` is the address of the stack SPSel 1 selects there.
pub(super) const STACK_SELECT: usize = MASK_WRITES + 1 + WRITTEN.len();

/// The cell of IC IVAU (`x86_64::register_read::code_page_thunk`), open
/// from one's completion until something could make another do more: code
/// compiled, a translation changed or the DSB that ends the kernel's run of
/// lines. Its `value` is an address in the page.
pub(super) const CODE_PAGE: usize = STACK_SELECT + 1;

/// The cell of MSR PAN (`x86_64::register_read::pan_thunk`), open once one
/// completed plainly in the state on a host that keeps PAN in RFLAGS.AC.
pub(super) const PAN_WRITES: usize = CODE_PAGE + 1;

/// The cell of MSR TTBR0_EL1 (`x86_64::register_read::unchanged_write_thunk`),
/// open on the storage a plain write reached: a write of the value the
/// register holds completes natively and changes nothing, any other exits.
/// A kernel rewrites the same table base at most switches into a thread of
/// the address space it already runs.
pub(super) const TABLE_BASE_KEPT: usize = PAN_WRITES + 1;

/// TTBR0_EL1 (TTBR0_EL2 at EL2 with E2H).
pub(super) const TABLE_BASE: Encoding = Encoding::new(3, 0, 2, 0, 0);

/// The cells of the EL2 virtual timer's compare and control
/// (`x86_64::register_read::timer_thunk`), whose `value`s are the addresses
/// of CNTHV_CVAL_EL2's and CNTHV_CTL_EL2's storage. An access of CNTV_TVAL,
/// CNTV_CVAL or CNTV_CTL that completes plainly opens both, only where
/// such an access always reaches that channel with no offset: at EL2,
/// outside a guarded level, with HCR_EL2.E2H set (Arm `AArch64_CNTV_*`
/// redirect to CNTHV_*), on a chip whose one timer bank counts the system
/// count unscaled (before the M3: `timer::agt`). No CNTHCTL_EL2 trap applies
/// at EL2.
pub(super) const TIMER_COMPARE: usize = TABLE_BASE_KEPT + 1;
pub(super) const TIMER_CONTROL: usize = TIMER_COMPARE + 1;

/// Where native timer accesses find the channel and the engine's words.
pub(super) const TIMER: register_read::TimerPlaces = register_read::TimerPlaces {
    compare: cell_place(TIMER_COMPARE),
    control: cell_place(TIMER_CONTROL),
    lines_until: Place::new(Anchor::Mirrors, offset_of!(Mirrors, lines_until)),
    count: Place::new(Anchor::Count, 0),
};

/// MSR CNTV_TVAL_EL0, CNTV_CTL_EL0 and CNTV_CVAL_EL0, and MRS
/// CNTV_CTL_EL0, with Rt in bits 4:0.
const TIMER_VALUE_WRITE: u32 = 0xd51b_e300;
const TIMER_CONTROL_WRITE: u32 = 0xd51b_e320;
const TIMER_COMPARE_WRITE: u32 = 0xd51b_e340;
const TIMER_CONTROL_READ: u32 = 0xd53b_e320;

/// What a native timer access of `word` does, and Rt, if `word` is one:
/// [`TimerWord`].
pub(super) fn timer_word(word: u32) -> Option<(TimerWord, u8)> {
    let access = match word & !31 {
        TIMER_VALUE_WRITE => TimerWord::Value,
        TIMER_CONTROL_WRITE => TimerWord::ControlWrite,
        TIMER_COMPARE_WRITE => TimerWord::Compare,
        TIMER_CONTROL_READ => TimerWord::ControlRead,
        _ => return None,
    };
    Some((access, (word & 31) as u8))
}

/// The timer accesses native code completes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TimerWord {
    Value,
    Compare,
    ControlWrite,
    ControlRead,
}

/// The cell that says the cycle counter's read is plain in the state
/// (`x86_64::register_read::cycle_counter_thunk`): it opens with the
/// counter's own cell ([`COUNTER_MIRRORS`]) and closes only with every cell,
/// as plainness follows from System registers and PSTATE alone, while a
/// native transition closes the counter's cell because its value, the
/// count, went stale.
pub(super) const CYCLES_PLAIN: usize = TIMER_CONTROL + 1;

/// Where native code finds [`Mirrors::cycles`].
pub(super) const CYCLES: Place = Place::new(Anchor::Mirrors, offset_of!(Mirrors, cycles));

/// The cell that says the instruction counter's read is plain in the state
/// (`x86_64::register_read::instruction_counter_thunk`), as
/// [`CYCLES_PLAIN`] is the cycle counter's.
pub(super) const INSTRUCTIONS_PLAIN: usize = CYCLES_PLAIN + 1;

/// Where native code finds [`Mirrors::instructions`].
pub(super) const INSTRUCTION_COUNTER: Place =
    Place::new(Anchor::Mirrors, offset_of!(Mirrors, instructions));

/// The cells of MSR SPRR_PERM_EL0 and MSR SPRR_PERM_EL1
/// (`x86_64::register_read::unchanged_write_thunk`), open on the storage a
/// plain write reached in the state: a write of the value the register
/// holds completes natively and changes nothing (no access changes, as the
/// engine's own write of it finds), any other exits. A kernel writes the
/// user permissions at each return to a thread and the kernel ones around
/// each write of protected data, mostly with the value they hold. A write
/// whose model refused it closes them ([`Machine::register_refused`]).
pub(super) const PERMISSION_KEPT: [usize; 2] = [INSTRUCTIONS_PLAIN + 1, INSTRUCTIONS_PLAIN + 2];

/// The [`PERMISSION_KEPT`] cell of the MSR `word`, and Rt, if it writes
/// SPRR_PERM_EL0 or SPRR_PERM_EL1.
pub(super) fn permission_write_word(word: u32) -> Option<(usize, u8)> {
    const MSR_SPRR_PERM_EL0: u32 = 0xd51e_f1a0;
    const MSR_SPRR_PERM_EL1: u32 = 0xd51e_f1c0;
    let kept = match word & !31 {
        MSR_SPRR_PERM_EL0 => PERMISSION_KEPT[0],
        MSR_SPRR_PERM_EL1 => PERMISSION_KEPT[1],
        _ => return None,
    };
    Some((kept, (word & 31) as u8))
}

/// The cell of the first of the [`STORED_WRITES`].
const STORED_WRITE_CELLS: usize = PERMISSION_KEPT[1] + 1;

/// The IMPLEMENTATION DEFINED registers that a store completes writes of
/// that get a cell of their own the first time code naming them is installed
/// ([`Machine::stored_write_cell`]), open on the register's value once a
/// write completes plainly. A kernel writes them around each thread switch.
/// When there are more, the rest take System exits. Numbers go in install
/// order and never free, and boot installs writes of dozens of tuning,
/// control and counter registers (26A434 XNU alone writes over 40 stored
/// ones) before the first thread switch, so room for the ones a kernel
/// writes at every switch (thread keys, SPRR's field registers) must
/// outlast those.
pub(super) const STORED_WRITES: usize = 128;

const CELLS: usize = STORED_WRITE_CELLS + STORED_WRITES;

/// The cell of `stored_writes[index]`.
pub(super) const fn stored_write_cell(index: usize) -> usize {
    STORED_WRITE_CELLS + index
}

/// The cell of the mask writes (`x86_64::register_read::mask_thunk`), open
/// once one completed plainly in the state.
pub(super) const MASK_WRITES: usize = MIRRORED.len() + STORED_CELLS;

/// The registers whose MSR native code completes, which a kernel writes
/// around each exception and whose writes change nothing else: a cell each,
/// after [`MASK_WRITES`], that a plain write opens on the storage it reached
/// (FPSR's lives in the frame, so names none). They stay open as reads do.
/// A PAC key's halves also close the keys to PAC thunks, which then reopen
/// together (`Machine::open_pointers`). TPIDR_EL1's also update its open
/// read cells in every state and log the thread switch for the engine's
/// diagnostics ([`THREAD_SWITCH`]).
pub(super) const WRITTEN: [(Encoding, SystemRegister); 18] = [
    (Encoding::new(3, 0, 4, 0, 1), SystemRegister::ELR_EL1),
    (Encoding::new(3, 0, 4, 0, 0), SystemRegister::SPSR_EL1),
    (STACK_ZERO, SystemRegister::SP_EL0),
    (Encoding::new(3, 3, 4, 4, 1), SystemRegister::FPSR),
    (registers::encodings::TPIDR_EL0, SystemRegister::TPIDR_EL0),
    (
        registers::encodings::TPIDRRO_EL0,
        SystemRegister::TPIDRRO_EL0,
    ),
    (Encoding::new(3, 0, 2, 1, 0), SystemRegister::APIAKeyLo_EL1),
    (Encoding::new(3, 0, 2, 1, 1), SystemRegister::APIAKeyHi_EL1),
    (Encoding::new(3, 0, 2, 1, 2), SystemRegister::APIBKeyLo_EL1),
    (Encoding::new(3, 0, 2, 1, 3), SystemRegister::APIBKeyHi_EL1),
    (Encoding::new(3, 0, 2, 2, 0), SystemRegister::APDAKeyLo_EL1),
    (Encoding::new(3, 0, 2, 2, 1), SystemRegister::APDAKeyHi_EL1),
    (Encoding::new(3, 0, 2, 2, 2), SystemRegister::APDBKeyLo_EL1),
    (Encoding::new(3, 0, 2, 2, 3), SystemRegister::APDBKeyHi_EL1),
    (Encoding::new(3, 0, 2, 3, 0), SystemRegister::APGAKeyLo_EL1),
    (Encoding::new(3, 0, 2, 3, 1), SystemRegister::APGAKeyHi_EL1),
    (CONTEXT_ID, SystemRegister::CONTEXTIDR_EL1),
    (registers::encodings::TPIDR_EL1, SystemRegister::TPIDR_EL1),
];

/// The [`WRITTEN`] index of TPIDR_EL1, the kernel's thread pointer.
pub(super) const THREAD_POINTER_WRITE: usize = WRITTEN.len() - 1;

/// The [`MIRRORED`] index of TPIDR_EL1.
pub(super) const THREAD_POINTER_MIRROR: usize = 0;
const _: () = assert!(matches!(
    MIRRORED[THREAD_POINTER_MIRROR],
    registers::encodings::TPIDR_EL1
));

/// Where a native TPIDR_EL1 write finds the register's read cells and the
/// switch log (`x86_64::register_read::thread_switch_thunk`).
pub(super) const THREAD_SWITCH: register_read::ThreadSwitch = register_read::ThreadSwitch {
    reads: Place::new(
        Anchor::Mirrors,
        offset_of!(Mirrors, banks) + THREAD_POINTER_MIRROR * size_of::<register_read::Cell>(),
    ),
    stride: size_of::<[register_read::Cell; CELLS]>(),
    banks: MIRROR_STATES,
    log: Place::new(Anchor::Mirrors, offset_of!(Mirrors, switches)),
};

/// CONTEXTIDR_EL1 (CONTEXTIDR_EL2 at EL2 with E2H), which a kernel writes
/// at each thread or address space switch: only debug and trace read it,
/// context-aware breakpoints live from its storage, and it is not part of
/// the prediction context (`PredictionContext`).
pub(super) const CONTEXT_ID: Encoding = Encoding::new(3, 0, 13, 0, 1);

/// SP_EL0, plain only with SP_ELx selected.
pub(super) const STACK_ZERO: Encoding = Encoding::new(3, 0, 4, 1, 0);

impl Default for Mirrors {
    fn default() -> Self {
        let cell = register_read::Cell::default();
        Self {
            banks: [[cell; CELLS]; MIRROR_STATES],
            state: 0,
            entered: 1,
            selected: 0,
            fields: [0; 3],
            interrupt_status: register_read::Cell::default(),
            barrier: register_read::Cell::default(),
            events: register_read::Cell::default(),
            exclusive: register_read::Cell::default(),
            transitions: transition::Table::default(),
            lines_until: 0,
            switches: register_read::SwitchLog::default(),
            monitor: 0,
            cycles: register_read::CycleCounter::default(),
            instructions: register_read::InstructionCounter::default(),
        }
    }
}

/// Where native transitions find the mirrors' words.
pub(super) const TRANSITIONS: transition::Places = transition::Places {
    state: Place::new(Anchor::Mirrors, offset_of!(Mirrors, state)),
    table: Place::new(Anchor::Mirrors, offset_of!(Mirrors, transitions)),
    selected: SELECTED,
    fields: Place::new(Anchor::Mirrors, offset_of!(Mirrors, fields)),
    field_bits: [
        PSTATE_FIELDS[0].1 as u32,
        PSTATE_FIELDS[1].1 as u32,
        PSTATE_FIELDS[2].1 as u32,
    ],
    interrupt_status: INTERRUPT_STATUS,
    lines_until: Place::new(Anchor::Mirrors, offset_of!(Mirrors, lines_until)),
    count: Place::new(Anchor::Count, 0),
    monitor: Place::new(Anchor::Mirrors, offset_of!(Mirrors, monitor)),
    pointer_tags: super::pointer::TAGS,
    pointer_generation: Place::new(
        Anchor::Pointer,
        offset_of!(super::pointer::PointerService, generation),
    ),
};
const _: () = assert!(size_of::<usize>() == size_of::<u64>());

/// The PSTATE fields with registers of their own that a kernel saves and
/// restores around each exception, which only the engine reads besides
/// MRS: DIT, UAO and SSBS, by encoding and register bit. A plain MRS or MSR
/// of one opens a stored cell on its [`Mirrors::fields`] word, which
/// [`Mirrors::enter`] and [`Mirrors::leave`] keep with PSTATE around native
/// code. A write takes the field's bit alone, as the register's do.
pub(super) const PSTATE_FIELDS: [(Encoding, u64); 3] = [
    (Encoding::new(3, 3, 4, 2, 5), 1 << 24),
    (Encoding::new(3, 0, 4, 2, 4), 1 << 23),
    (Encoding::new(3, 3, 4, 2, 6), 1 << 12),
];

/// The [`PSTATE_FIELDS`] index and bit of `encoding`, if it is one.
pub(super) fn pstate_field(encoding: Encoding) -> Option<(usize, u64)> {
    PSTATE_FIELDS
        .iter()
        .position(|&(field, _)| field == encoding)
        .map(|index| (index, PSTATE_FIELDS[index].1))
}

/// The cell of `WRITTEN[index]`.
pub(super) const fn written_cell(index: usize) -> usize {
    MASK_WRITES + 1 + index
}

/// The ELs and GXF states, `Mirrors::banks` per [`mirror_state`].
pub(super) const MIRROR_STATES: usize = 8;
const _: () = assert!(MIRROR_STATES == transition::STATES);

pub(super) fn mirror_state(el: u8, guarded: bool) -> usize {
    usize::from(el) | usize::from(guarded) << 2
}

impl Mirrors {
    /// The cells of the PE's state, at what [`Anchor::Cells`] holds.
    pub(super) fn cells(&self) -> &[register_read::Cell; CELLS] {
        &self.banks[self.state]
    }

    /// The state native code left current, which a native transition may
    /// have moved: its cells may hold open values from now on.
    pub(super) fn native_state(&mut self) -> usize {
        self.state &= MIRROR_STATES - 1;
        self.entered |= 1 << self.state;
        self.state
    }

    /// Native code is entered with PSTATE `pstate`.
    pub(super) fn enter(&mut self, pstate: Pstate) {
        let bit = |set: bool, index: usize| if set { PSTATE_FIELDS[index].1 } else { 0 };
        self.fields = [bit(pstate.dit, 0), bit(pstate.uao, 1), bit(pstate.ssbs, 2)];
    }

    /// Native code returned: `pstate` takes the fields it wrote.
    pub(super) fn leave(&self, pstate: &mut Pstate) {
        let [dit, uao, ssbs] = self.fields.map(|word| word != 0);
        (pstate.dit, pstate.uao, pstate.ssbs) = (dit, uao, ssbs);
    }

    /// The address native code reads and writes field `index` at.
    pub(super) fn field_storage(&self, index: usize) -> u64 {
        &self.fields[index] as *const u64 as u64
    }

    /// The PE is in state `to`: native code reads its cells.
    pub(super) fn transition(&mut self, to: usize) {
        self.state = to;
        self.entered |= 1 << to;
    }

    /// Close every cell. A closed cell keeps no value a mispredicted native
    /// read could use (FEAT_CSV3's System register rule).
    pub(super) fn close_all(&mut self) {
        for (state, cells) in self.banks.iter_mut().enumerate() {
            if self.entered >> state & 1 != 0 {
                *cells = [register_read::Cell::default(); CELLS];
            }
        }
        self.entered = 1 << self.state;
    }

    /// Whether every cell is closed.
    #[cfg(test)]
    pub(super) fn closed(&self) -> bool {
        self.banks.iter().flatten().all(|cell| cell.readable == 0)
    }

    /// The register of cell `index` now holds `value`, as its open cells
    /// in every state read it.
    pub(super) fn update(&mut self, index: usize, value: u64) {
        for cells in &mut self.banks {
            if cells[index].readable != 0 {
                cells[index].value = value;
            }
        }
    }

    /// IC IVAU of `address` completed: its page's lines need no other.
    pub(super) fn open_code_page(&mut self, address: u64) {
        self[CODE_PAGE] = register_read::Cell {
            readable: 1,
            value: address,
        };
        // The DSB that ends the run of lines closes the cell again.
        self.barrier = register_read::Cell::default();
    }

    /// Another IC IVAU of the page could do more.
    pub(super) fn close_code_page(&mut self) {
        self.close(CODE_PAGE);
    }

    /// Close the cells of one register in every state.
    pub(super) fn close(&mut self, index: usize) {
        for cells in &mut self.banks {
            cells[index] = register_read::Cell::default();
        }
    }
}

/// System registers whose writes leave the derived caches intact.
/// Exception registers and thread pointers are read when used. TTBR0
/// changes translation, and its write still synchronizes the host regime.
/// CONTEXTIDR changes only live debug matches, which read it when checked.
pub(super) fn stored(register: SystemRegister) -> bool {
    use SystemRegister::*;
    matches!(
        register,
        TTBR0_EL1
            | TTBR0_EL2
            | CONTEXTIDR_EL1
            | CONTEXTIDR_EL2
            | ELR_EL1
            | ELR_EL2
            | SPSR_EL1
            | SPSR_EL2
            | ESR_EL1
            | ESR_EL2
            | FAR_EL1
            | FAR_EL2
            | PAR_EL1
            | TPIDR_EL0
            | TPIDRRO_EL0
            | TPIDR_EL1
            | TPIDR_EL2
    )
}

/// Whether `register` is half of a PAC key.
pub(super) fn pointer_key(register: SystemRegister) -> bool {
    use SystemRegister::*;
    matches!(
        register,
        APIAKeyLo_EL1
            | APIAKeyHi_EL1
            | APIBKeyLo_EL1
            | APIBKeyHi_EL1
            | APDAKeyLo_EL1
            | APDAKeyHi_EL1
            | APDBKeyLo_EL1
            | APDBKeyHi_EL1
            | APGAKeyLo_EL1
            | APGAKeyHi_EL1
    )
}

/// Whether a generated read that ends with `register`'s value is the plain
/// read of its storage, which a cell may point at: only state those reads
/// return as stored, never a view composed from it.
pub(super) fn reads_storage(register: SystemRegister) -> bool {
    use SystemRegister::*;
    stored(register)
        || matches!(
            register,
            SP_EL0
                | HCR_EL2
                | SCTLR_EL1
                | SCTLR_EL2
                | TCR_EL1
                | TCR_EL2
                | TTBR1_EL1
                | TTBR1_EL2
                | VBAR_EL1
                | VBAR_EL2
                | CPTR_EL2
                | CPACR_EL1
        )
}

/// Whether MRS of `encoding` can have a stored cell of its own storage: not
/// the ID space, whose reads are few, nor the IMPLEMENTATION DEFINED one,
/// which Apple's state answers.
pub(super) fn storable(encoding: Encoding) -> bool {
    !(encoding.op0() == 3 && matches!(encoding.crn(), 0 | 11 | 15))
}

impl core::ops::Index<usize> for Mirrors {
    type Output = register_read::Cell;
    fn index(&self, index: usize) -> &Self::Output {
        &self.banks[self.state][index]
    }
}

impl core::ops::IndexMut<usize> for Mirrors {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.banks[self.state][index]
    }
}

/// The [`MIRRORED`] index of DAIF.
pub(super) const DAIF_MIRROR: usize = MIRRORED.len() - 1;
const _: () = assert!(matches!(MIRRORED[DAIF_MIRROR], registers::encodings::DAIF));
/// The [`MIRRORED`] indices of CNTVCT_EL0 and CNTVCT_ALIAS_EL0, count
/// reads completed natively whenever the host's counter is its TSC's
/// ([`Host::tsc_counter`]).
pub(super) const COUNT_MIRRORS: [usize; 2] = [DAIF_MIRROR - 2, DAIF_MIRROR - 1];
const _: () = assert!(matches!(
    MIRRORED[COUNT_MIRRORS[0]],
    stela_compiler::aarch64::system::registers::CNTVCT_EL0
));
const _: () = assert!(matches!(
    MIRRORED[COUNT_MIRRORS[1]],
    registers::encodings::CNTVCT_ALIAS_EL0
));

/// The [`MIRRORED`] indices of SPSR_GL1, ASPSR_GL1 and ELR_GL1, which a
/// guarded level reads when it is entered and writes before it leaves. Their
/// cells open only in a guarded level, where they are plain, on the bank's
/// storage: reads and writes both go there, as GENTER and an exception
/// taken to the level do ([`registers::guarded_storage`]).
pub(super) const GUARDED_MIRRORS: [usize; 3] = [4, 5, 6];
const _: () = assert!(matches!(
    [
        MIRRORED[GUARDED_MIRRORS[0]],
        MIRRORED[GUARDED_MIRRORS[1]],
        MIRRORED[GUARDED_MIRRORS[2]]
    ],
    [
        registers::encodings::SPSR_GL1,
        registers::encodings::ASPSR_GL1,
        registers::encodings::ELR_GL1
    ]
));

/// The [`MIRRORED`] indices of the cycle and instruction counters. They
/// move at every instruction but no read depends on it: between transitions
/// they read the count of the last one, as exceptions are what a kernel
/// charges to threads, and later reads never go back.
pub(super) const COUNTER_MIRRORS: [usize; 2] = [8, 9];
const _: () = assert!(COUNTER_MIRRORS[0] == 8);

/// MRS X0 of `encoding`.
pub(super) fn read_word(encoding: Encoding) -> u32 {
    0xd530_0000
        | u32::from(encoding.op0() & 1) << 19
        | u32::from(encoding.op1()) << 16
        | u32::from(encoding.crn()) << 12
        | u32::from(encoding.crm()) << 8
        | u32::from(encoding.op2()) << 5
}

/// The [`MIRRORED`] index of SPRR_PERM_EL0, which dyld and JIT runtimes
/// write around each write to their protected pages and read back at once.
/// An SPRR_PERM write keeps derived state ([`Machine::permissions_changed`])
/// but closes these cells; a write of SPRR_PERM_EL0 itself reopens the one
/// open before it ([`Machine::permission_write`]).
pub(super) const SPRR_PERM_MIRROR: usize = COUNT_MIRRORS[0] - 1;
const _: () = assert!(matches!(
    MIRRORED[SPRR_PERM_MIRROR],
    registers::encodings::SPRR_PERM_EL0
));

/// MRS DAIF's value for PSTATE's D, A, I and F masks: bits 9:6.
pub(super) fn daif_register(daif: u8) -> u64 {
    u64::from(daif) << DAIF_REGISTER_SHIFT
}

pub(super) const DAIF_REGISTER_SHIFT: u32 = 6;

/// The index in [`WRITTEN`] of a write of `access`.
pub(super) fn written(access: RegisterAccess) -> Option<usize> {
    if access.direction() != Direction::Write {
        return None;
    }
    let encoding = access.encoding();
    WRITTEN.iter().position(|&(written, _)| written == encoding)
}

/// The [`MIRRORED`] index of a read of `access`, other than to XZR.
pub(super) fn mirrored(access: RegisterAccess) -> Option<usize> {
    if access.direction() != Direction::Read || access.rt() == ZERO_REGISTER {
        return None;
    }
    MIRRORED
        .iter()
        .position(|&encoding| encoding == access.encoding())
}

/// The [`GUARDED_MIRRORS`] index of a write of `access`.
pub(super) fn guarded_written(access: RegisterAccess) -> Option<usize> {
    if access.direction() != Direction::Write {
        return None;
    }
    let encoding = access.encoding();
    GUARDED_MIRRORS
        .into_iter()
        .find(|&index| MIRRORED[index] == encoding)
}

/// The MSR `word` at `pc` of a [`GUARDED_MIRRORS`] register: its index and Xt.
pub(super) fn guarded_written_register(pc: u64, word: u32) -> Option<(usize, u8)> {
    let instruction = block::decode_system_mode_instruction(GuestAddress(pc), word).ok()??;
    match instruction.operation {
        SystemOperation::Register(access) => Some((guarded_written(access)?, access.rt())),
        _ => None,
    }
}

/// The MSR `word` at `pc` of a [`WRITTEN`] register: its index and Xt.
pub(super) fn written_register(pc: u64, word: u32) -> Option<(usize, u8)> {
    let instruction = block::decode_system_mode_instruction(GuestAddress(pc), word).ok()??;
    match instruction.operation {
        SystemOperation::Register(access) => Some((written(access)?, access.rt())),
        _ => None,
    }
}

/// Xt of the MSR TTBR0_EL1 `word` at `pc`, if it is one.
pub(super) fn table_base_write(pc: u64, word: u32) -> Option<u8> {
    let instruction = block::decode_system_mode_instruction(GuestAddress(pc), word).ok()??;
    match instruction.operation {
        SystemOperation::Register(access)
            if access.encoding() == TABLE_BASE && access.direction() == Direction::Write =>
        {
            Some(access.rt())
        }
        _ => None,
    }
}

/// The selection the MSR SPSel `word` writes, if it is one.
pub(super) fn stack_select(word: u32) -> Option<bool> {
    match word {
        0xd500_40bf => Some(false),
        0xd500_41bf => Some(true),
        _ => None,
    }
}

/// The immediate of the MSR PAN `word`, if it is one.
pub(super) fn pan_write(word: u32) -> Option<bool> {
    const MSR_PAN: u32 = 0xd500_409f;
    const IMMEDIATE: u32 = 1 << 8;
    (word & !IMMEDIATE == MSR_PAN).then_some(word & IMMEDIATE != 0)
}

/// Whether `word` at `pc` is a DSB that only completes maintenance.
pub(super) fn data_barrier(pc: u64, word: u32) -> bool {
    let decoded = block::decode_system_mode_instruction(GuestAddress(pc), word);
    matches!(
        decoded
            .ok()
            .flatten()
            .map(|instruction| instruction.operation),
        Some(SystemOperation::Synchronization(
            Synchronization::DataCompletion { nxs: false, .. }
        ))
    )
}

/// Whether `word` at `pc` is CLREX.
pub(super) fn clear_exclusive(pc: u64, word: u32) -> bool {
    matches!(
        block::decode_system_mode_instruction(GuestAddress(pc), word)
            .ok()
            .flatten()
            .map(|instruction| instruction.operation),
        Some(SystemOperation::Exclusive(Ok(
            stela_compiler::aarch64::exclusive::Operand::Clear
        )))
    )
}

/// Whether `word` at `pc` is SEVL (true) or WFE without a timeout (false),
/// if it is one.
pub(super) fn event_instruction(pc: u64, word: u32) -> Option<bool> {
    use stela_compiler::aarch64::system::synchronization::WaitKind;
    let instruction = block::decode_system_mode_instruction(GuestAddress(pc), word).ok()??;
    match instruction.operation {
        SystemOperation::SendEvent { local: true } => Some(true),
        SystemOperation::Wait {
            kind: WaitKind::Event,
            timeout_register: None,
        } => Some(false),
        _ => None,
    }
}

/// Xt of the IC IVAU `word`, if it is one.
pub(super) fn code_page_invalidation(word: u32) -> Option<u8> {
    const IC_IVAU: u32 = 0xd50b_7520;
    (word & !31 == IC_IVAU).then_some((word & 31) as u8)
}

/// The MSR `word` at `pc`, if it writes a register.
pub(super) fn msr_access(pc: u64, word: u32) -> Option<RegisterAccess> {
    let instruction = block::decode_system_mode_instruction(GuestAddress(pc), word).ok()??;
    match instruction.operation {
        SystemOperation::Register(access) if access.direction() == Direction::Write => Some(access),
        _ => None,
    }
}

/// The MRS `word` at `pc`, if it reads to a register.
pub(super) fn mrs_access(pc: u64, word: u32) -> Option<RegisterAccess> {
    let instruction = block::decode_system_mode_instruction(GuestAddress(pc), word).ok()??;
    match instruction.operation {
        SystemOperation::Register(access)
            if access.direction() == Direction::Read && access.rt() != ZERO_REGISTER =>
        {
            Some(access)
        }
        _ => None,
    }
}

/// The [`MIRRORED`] index and Xt of `word` at `pc` if it is MRS of a
/// mirrored register to Xt, other than XZR.
pub(super) fn mirrored_read(pc: u64, word: u32) -> Option<(usize, u8)> {
    let instruction = block::decode_system_mode_instruction(GuestAddress(pc), word).ok()??;
    match instruction.operation {
        SystemOperation::Register(access) => Some((mirrored(access)?, access.rt())),
        _ => None,
    }
}

/// Whether `word` at `pc` is a DC ZVA that blocks compiled for `system`
/// complete natively, inside a block as well as at its end.
pub(super) fn zeroes_natively(pc: u64, word: u32, system: NativeSystem) -> bool {
    system.zero_block.is_some()
        && block::decode_system_mode_instruction(GuestAddress(pc), word)
            .ok()
            .flatten()
            .is_some_and(|instruction| zero_block::operand(&instruction).is_some())
}

/// Whether `word` at `pc` is a System instruction blocks compiled for
/// `system` complete natively: DC ZVA, one of its DC clean operations or an
/// exclusive of one register.
pub(super) fn completes_natively(pc: u64, word: u32, system: NativeSystem) -> bool {
    block::decode_system_mode_instruction(GuestAddress(pc), word)
        .ok()
        .flatten()
        .is_some_and(|instruction| {
            system.zero_block.is_some() && zero_block::operand(&instruction).is_some()
                || cache_clean::operand(&instruction)
                    .is_some_and(|(operation, _)| system.cache_clean.contains(operation))
                || system.exclusive && native_exclusive::operand(&instruction).is_some()
        })
}
