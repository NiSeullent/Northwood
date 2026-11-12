//! Native completion of GENTER and GEXIT from a transition record.
//!
//! The instruction ends its block with a System exit. A thunk the engine
//! installs over that exit completes it natively from the [`Record`] the
//! engine left for the PE's state and the transition's kind in its
//! [`Table`]: what the last engine-path transition of that kind from that
//! state showed of everything the transition does but its operands — the
//! state it enters, that state's cells, its ITLB contexts, where its stack
//! pointers and GXF bank registers live, and the host's view of it. A record
//! holds while the table's `generation` is the record's: the engine moves
//! the generation on whenever state any record depends on may change.
//!
//! GENTER banks SPSR (PSTATE as the frame holds it), ELR (the next
//! instruction), ESR (the immediate) and ASPSR in the level's GXF registers
//! and continues at its entry point on its stack with every exception
//! masked (`Cpu::guarded_enter`). GEXIT restores PSTATE from SPSR and
//! continues at ELR (`Cpu::guarded_exit`), while SPSR's bits that are no
//! operand of the native path (mode, D, PAN, TCO, IL, SS, BTYPE, PACM) are
//! the record's, ASPSR's guarded bit is the record's, no interrupt waiting
//! since native code was entered ([`GuestState::pending`]) is unmasked, no
//! interrupt line changed since and the count has not reached the one at
//! which time alone could raise one (`Places::lines_until`). Otherwise either takes its original
//! exit to the engine.
//!
//! The thunk counts its completion in the table, writes the state word and
//! the PE's cells anchor, ITLB contexts and salts for the next lookup,
//! closes the native ISR_EL1 read and the cells of the new state the
//! engine refreshes at each transition (the record's `close`), and moves the host to the new view when
//! the record names one: CR3 for the view's root, which must still be the
//! one the record names and not stale, the host's view word and RFLAGS.AC
//! for PAN. Then it leaves through the exit to the target in RAX: linked
//! (`Thunk::dynamic`) while the table's and the record's `chain` words allow
//! the next state's code to run in this native run, else through an exit
//! that is never linked. It clobbers RAX, RCX, RDX and flags, and the pinned SP.
use super::asm::{Alu, Asm, Cond, Gpr, Mem, Patch, ShiftOp, Size};
use super::flag_word::{ARM_MASK, ARM_SHIFT, PACK_MULTIPLIER, UNPACK_MULTIPLIER, WORD_MASK};
use super::pointer::Thunk;
use crate::{Anchor, GuestAddress, GuestState, Place, aarch64::Register, block::ExitKind};
use alloc::vec::Vec;
use core::mem::offset_of;

/// The kinds of transition a [`Table`] keeps records of.
pub const SUPERVISOR_CALL: usize = 0;
pub const EXCEPTION_RETURN: usize = 1;
pub const GUARDED_ENTER: usize = 2;
pub const GUARDED_EXIT: usize = 3;
pub const KINDS: usize = 4;
/// The PE states a [`Table`] keeps records for (the engine's mirror states).
pub const STATES: usize = 8;

/// What one transition of one kind from one state does but its operands.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Record {
    /// The table's generation the record holds in; 0 never does.
    pub generation: u64,
    /// The frame (`GuestState`) the addresses below belong with.
    pub frame: u64,
    /// The stack selection (PSTATE.SP) it was made with.
    pub selected: u64,
    /// The state word after the transition.
    pub to: u64,
    /// The cells of that state (`Anchor::Cells`).
    pub cells: u64,
    /// The ITLB contexts and salts there (`chain::ChainControl`), good for
    /// a lookup while `contexts_generation` is the table's.
    pub contexts: [u64; 2],
    pub salts: u64,
    pub contexts_generation: u64,
    /// Where the SP left is saved and the SP taken is loaded from; both
    /// [`Self::scratch`] when they are one stack.
    pub sp_save: u64,
    pub sp_load: u64,
    /// The level's SPSR, ELR, ESR and ASPSR.
    pub spsr: u64,
    pub elr: u64,
    pub esr: u64,
    pub aspsr: u64,
    /// Where the target is read: GXF_ENTER for GENTER, ELR for GEXIT.
    pub target: u64,
    /// GENTER: ASPSR's value. GEXIT: ASPSR's guarded bit.
    pub aspsr_value: u64,
    /// GENTER: SPSR's bits the state fixes (mode, TCO).
    pub spsr_fixed: u64,
    /// GEXIT: SPSR & `check_mask` must be `check`. GENTER: PSTATE.PAN
    /// must be `check`.
    pub check_mask: u64,
    pub check: u64,
    /// Nonzero when the host's view after it is exact for native code to
    /// continue there ([`Table::chain`] permitting).
    pub chain: u64,
    /// The host's CR3 for the new view, 0 when the view stays.
    pub cr3: u64,
    /// The host's root of that view, which must be `root`, and its stale
    /// flag (a byte): a stale context loads without the no-flush bit (bit
    /// 63), which clears the flag.
    pub root_at: u64,
    pub root: u64,
    pub stale_at: u64,
    /// The host's view word and its new value.
    pub view_at: u64,
    pub view: u64,
    /// RFLAGS.AC after the switch: [`AC_KEEP`], [`AC_SET`] or [`AC_CLEAR`].
    pub ac: u64,
    /// SVC: the vector it continues at. ERET: the target the thunk
    /// computed. `target` names this word for both.
    pub vector: u64,
    /// SVC: PSTATE.PAN after it (the frame's word).
    pub pan: u64,
    /// SVC: each PSTATE field word ([`Places::fields`]) after it is the
    /// word before AND `fields_keep` OR `fields_set`.
    pub fields_keep: [u64; 3],
    pub fields_set: [u64; 3],
    /// ERET: nonzero where the target's top byte is a tag the return
    /// removes (`AArch64_BranchAddr`), for the lower and the upper half.
    pub untag: [u64; 2],
    /// ERET: the event register byte the return sets.
    pub event: u64,
    /// SVC and ERET between levels whose PAC key enables differ: nonzero,
    /// and the PAC service's tag words ([`Places::pointer_tags`]) must be
    /// `pointer_from` (an open service of the level left) and become
    /// `pointer_to`, the level entered's, as the engine reopens them.
    pub pointer_swap: u64,
    /// The service's generation (its keys, version and layout) both are of.
    pub pointer_generation: u64,
    pub pointer_from: [u64; super::pointer::TAG_WORDS],
    pub pointer_to: [u64; super::pointer::TAG_WORDS],
    /// The `readable` words of cells of the new state that close: those
    /// whose value the engine refreshes at each transition. [`Self::scratch`]
    /// where there are fewer.
    pub close: [u64; 3],
    pub scratch: u64,
}

pub const AC_KEEP: u64 = 0;
pub const AC_SET: u64 = 1;
pub const AC_CLEAR: u64 = 2;

/// A PE's transition records. Two tables are equal when the same records
/// hold in them, whatever their generations.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Table {
    /// Starts at 1.
    pub generation: u64,
    /// Nonzero while native code may continue in the new state.
    pub chain: u64,
    /// Moves when lookup contexts go: records keep holding, but continue
    /// in the new state only once their contexts are taken again.
    pub contexts_generation: u64,
    /// The states whose records' contexts are older than that, a bit each
    /// (the engine's, which no thunk reads).
    pub stale_contexts: u64,
    /// Native completions of each kind since the engine last took them.
    pub counts: [u64; KINDS],
    pub records: [[Record; KINDS]; STATES],
}

impl Default for Table {
    fn default() -> Self {
        Self {
            generation: 1,
            chain: 0,
            contexts_generation: 1,
            stale_contexts: 0,
            counts: [0; KINDS],
            records: [[Record::default(); KINDS]; STATES],
        }
    }
}

impl PartialEq for Table {
    fn eq(&self, other: &Self) -> bool {
        fn holding(table: &Table) -> impl Iterator<Item = Option<Record>> + '_ {
            table.records.as_flattened().iter().map(|record| {
                (record.generation == table.generation).then_some(Record {
                    generation: 0,
                    ..*record
                })
            })
        }
        self.chain == other.chain && self.counts == other.counts && holding(self).eq(holding(other))
    }
}

impl Eq for Table {}

impl Table {
    /// Every record stops holding.
    pub fn forget(&mut self) {
        self.generation = self.generation.wrapping_add(1).max(1);
    }

    /// Every record stops holding but those of the `kept` kinds that held,
    /// which hold in the new generation.
    pub fn forget_but(&mut self, kept: &[usize]) {
        let (old, new) = (self.generation, self.generation.wrapping_add(1).max(1));
        for records in &mut self.records {
            for &kind in kept {
                if records[kind].generation == old {
                    records[kind].generation = new;
                }
            }
        }
        self.generation = new;
    }

    /// Lookup contexts went: no record continues in its new state until
    /// the engine gives it that state's contexts again.
    pub fn contexts_moved(&mut self) {
        self.contexts_generation = self.contexts_generation.wrapping_add(1).max(1);
        self.stale_contexts = u64::MAX;
    }

    /// The completions counted since the last call, which it clears.
    pub fn take_counts(&mut self) -> [u64; KINDS] {
        core::mem::take(&mut self.counts)
    }
}

/// Where a thunk finds the engine's words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Places {
    /// The PE's state word, an index of [`Table::records`].
    pub state: Place,
    pub table: Place,
    /// The stack selection (PSTATE.SP) while native code runs.
    pub selected: Place,
    /// PSTATE.DIT, UAO and SSBS, each its SPSR bit when set.
    pub fields: Place,
    /// The SPSR bit of each of `fields`.
    pub field_bits: [u32; 3],
    /// The `readable` word of the native ISR_EL1 read.
    pub interrupt_status: Place,
    /// The count before which time alone changes no interrupt line, as the
    /// last step found it (`u64::MAX` none, 0 now), and the reader of the
    /// count (`register_read::Count`).
    pub lines_until: Place,
    pub count: Place,
    /// Nonzero while the engine holds an exclusive monitor of its own,
    /// which only it clears: ERET leaves it to the engine.
    pub monitor: Place,
    /// The PAC service's tag words (`x86_64::pointer`) and generation.
    pub pointer_tags: Place,
    pub pointer_generation: Place,
}

const RECORD_BYTES: usize = size_of::<Record>();
const RECORDS: usize = offset_of!(Table, records);

fn record_field(offset: usize) -> Mem {
    Mem::base(Gpr::Rdx, offset as i32)
}

fn frame_field(offset: usize) -> Mem {
    Mem::base(Gpr::Rdi, offset as i32)
}

/// SPSR's DAIF shift and PAN bit.
const SPSR_DAIF_SHIFT: u8 = 6;
const SPSR_PAN_SHIFT: u8 = 22;
const MOV_CR3_RCX: [u8; 3] = [0x0f, 0x22, 0xd9];
/// CR3's bit that keeps the PCID's entries (with CR4.PCIDE).
const KEEP_ENTRIES: u64 = 1 << 63;
const STAC: [u8; 3] = [0x0f, 0x01, 0xcb];
const CLAC: [u8; 3] = [0x0f, 0x01, 0xca];

/// RDX at the record of `kind` for the state word, RCX at the table, and
/// the branches to the original exit when it does not hold for this frame.
fn locate(asm: &mut Asm<'_>, places: &Places, kind: usize, misses: &mut Vec<Patch>) {
    use Gpr::{Rax, Rcx, Rdi, Rdx};
    use Size::Qword;
    asm.place(Rdx, places.state);
    asm.mov_load(Qword, Rdx, Mem::base(Rdx, 0));
    asm.alu_imm8(Alu::And, Qword, Rdx, (STATES - 1) as i8);
    asm.imul_imm32(Qword, Rdx, Rdx, (KINDS * RECORD_BYTES) as i32);
    asm.place(Rcx, places.table);
    asm.alu(Alu::Add, Qword, Rdx, Rcx);
    asm.alu_imm32(Alu::Add, Qword, Rdx, (RECORDS + kind * RECORD_BYTES) as i32);
    asm.mov_load(
        Qword,
        Rax,
        Mem::base(Rcx, offset_of!(Table, generation) as i32),
    );
    asm.alu_load(
        Alu::Cmp,
        Qword,
        Rax,
        record_field(offset_of!(Record, generation)),
    );
    misses.push(asm.jcc(Cond::Ne));
    asm.alu_load(
        Alu::Cmp,
        Qword,
        Rdi,
        record_field(offset_of!(Record, frame)),
    );
    misses.push(asm.jcc(Cond::Ne));
    asm.place(Rcx, places.selected);
    asm.mov_load(Qword, Rax, Mem::base(Rcx, 0));
    asm.alu_load(
        Alu::Cmp,
        Qword,
        Rax,
        record_field(offset_of!(Record, selected)),
    );
    misses.push(asm.jcc(Cond::Ne));
}

/// The branch to the original exit unless the host's root of the record's
/// view is still the record's and fresh.
fn check_host(asm: &mut Asm<'_>, misses: &mut Vec<Patch>) {
    use Gpr::{Rax, Rcx};
    use Size::Qword;
    asm.alu_imm8(Alu::Cmp, Qword, record_field(offset_of!(Record, cr3)), 0);
    let stays = asm.jcc8(Cond::E);
    asm.mov_load(Qword, Rcx, record_field(offset_of!(Record, root_at)));
    asm.mov_load(Qword, Rax, Mem::base(Rcx, 0));
    asm.alu_load(Alu::Cmp, Qword, Rax, record_field(offset_of!(Record, root)));
    misses.push(asm.jcc(Cond::Ne));
    asm.bind(stays);
}

/// Move SP to the record's save slot and load it from its load slot.
fn swap_stack(asm: &mut Asm<'_>) {
    use Gpr::Rcx;
    use Size::Qword;
    let sp = super::pinned::host(Register::Sp).expect("SP is pinned");
    asm.mov_load(Qword, Rcx, record_field(offset_of!(Record, sp_save)));
    asm.mov_store(Qword, Mem::base(Rcx, 0), sp);
    asm.mov_load(Qword, Rcx, record_field(offset_of!(Record, sp_load)));
    asm.mov_load(Qword, sp, Mem::base(Rcx, 0));
}

/// What every completion does after its own effects, RDX at the record,
/// then the exits: RAX the target.
fn commit(
    mut code: Vec<u8>,
    places: &Places,
    kind: usize,
    misses: Vec<Patch>,
    pc: GuestAddress,
) -> Thunk {
    use Gpr::{Rax, Rcx};
    use Size::Qword;
    let mut asm = Asm::new(&mut code);
    let to = |field| record_field(field);
    asm.place(Rcx, places.state);
    asm.mov_load(Qword, Rax, to(offset_of!(Record, to)));
    asm.mov_store(Qword, Mem::base(Rcx, 0), Rax);
    let count = Place::new(
        places.table.anchor,
        places.table.offset + offset_of!(Table, counts) + kind * 8,
    );
    asm.place(Rcx, count);
    asm.alu_imm8(Alu::Add, Qword, Mem::base(Rcx, 0), 1);
    asm.place(Rcx, places.interrupt_status);
    asm.mov_rm_imm32(Qword, Mem::base(Rcx, 0), 0);
    asm.mov_load(Qword, Rax, to(offset_of!(Record, cells)));
    let cells = offset_of!(GuestState, anchors) + Anchor::Cells as usize * 8;
    asm.mov_store(Qword, frame_field(cells), Rax);
    for close in 0..3 {
        asm.mov_load(Qword, Rcx, to(offset_of!(Record, close) + close * 8));
        asm.mov_rm_imm32(Qword, Mem::base(Rcx, 0), 0);
    }
    let contexts = offset_of!(super::chain::ChainControl, contexts);
    asm.place(Rcx, Place::new(Anchor::Control, contexts));
    for half in 0..2 {
        asm.mov_load(Qword, Rax, to(offset_of!(Record, contexts) + half * 8));
        asm.mov_store(Qword, Mem::base(Rcx, (half * 8) as i32), Rax);
    }
    let salts = offset_of!(super::chain::ChainControl, salts);
    asm.place(Rcx, Place::new(Anchor::Control, salts));
    asm.mov_load(Qword, Rax, to(offset_of!(Record, salts)));
    asm.mov_store(Qword, Mem::base(Rcx, 0), Rax);
    // The host's view.
    asm.alu_imm8(Alu::Cmp, Qword, to(offset_of!(Record, cr3)), 0);
    let stays = asm.jcc(Cond::E);
    asm.mov_load(Qword, Rax, to(offset_of!(Record, view)));
    asm.mov_load(Qword, Rcx, to(offset_of!(Record, view_at)));
    asm.mov_store(Qword, Mem::base(Rcx, 0), Rax);
    // A stale context flushes as it loads (`load_view`).
    asm.mov_load(Qword, Rcx, to(offset_of!(Record, stale_at)));
    asm.movzx(Size::Dword, Rax, Size::Byte, Mem::base(Rcx, 0));
    asm.alu(Alu::Or, Size::Dword, Rax, Rax);
    let fresh = asm.jcc8(Cond::E);
    asm.alu(Alu::Xor, Size::Dword, Rax, Rax);
    asm.mov_store(Size::Byte, Mem::base(Rcx, 0), Rax);
    asm.mov_load(Qword, Rcx, to(offset_of!(Record, cr3)));
    asm.mov_imm64(Rax, !KEEP_ENTRIES);
    asm.alu(Alu::And, Qword, Rcx, Rax);
    let load = asm.jmp8();
    asm.bind(fresh);
    asm.mov_load(Qword, Rcx, to(offset_of!(Record, cr3)));
    asm.bind(load);
    asm.fixed(&MOV_CR3_RCX);
    asm.alu_imm8(Alu::Cmp, Qword, to(offset_of!(Record, ac)), AC_SET as i8);
    let not_set = asm.jcc8(Cond::Ne);
    asm.fixed(&STAC);
    asm.bind(not_set);
    asm.alu_imm8(Alu::Cmp, Qword, to(offset_of!(Record, ac)), AC_CLEAR as i8);
    let not_clear = asm.jcc8(Cond::Ne);
    asm.fixed(&CLAC);
    asm.bind(not_clear);
    asm.bind(stays);
    // The target, and whether this run may continue there.
    asm.place(Rcx, places.table);
    asm.mov_load(Qword, Rax, to(offset_of!(Record, target)));
    asm.mov_load(Qword, Rax, Mem::base(Rax, 0));
    asm.alu_imm8(
        Alu::Cmp,
        Qword,
        Mem::base(Rcx, offset_of!(Table, chain) as i32),
        0,
    );
    let ends = asm.jcc(Cond::E);
    asm.alu_imm8(Alu::Cmp, Qword, to(offset_of!(Record, chain)), 0);
    let stops = asm.jcc(Cond::E);
    asm.mov_load(
        Qword,
        Rcx,
        Mem::base(Rcx, offset_of!(Table, contexts_generation) as i32),
    );
    asm.alu_load(
        Alu::Cmp,
        Qword,
        Rcx,
        to(offset_of!(Record, contexts_generation)),
    );
    let older = asm.jcc(Cond::Ne);
    let dynamic = asm.len();
    super::block::exit_kind_with(&mut asm, ExitKind::Return);
    asm.bind(ends);
    asm.bind(stops);
    asm.bind(older);
    super::block::exit_kind_with(&mut asm, ExitKind::Return);
    for miss in misses {
        asm.bind(miss);
    }
    super::block::direct_exit_with(&mut asm, pc, ExitKind::System);
    Thunk::new(code, None, Some(dynamic))
}

/// The thunk completing `GENTER #immediate` at `pc`, whose ESR is
/// `syndrome` (the immediate in it), else exiting at `pc` for the engine.
pub fn guarded_enter_thunk(pc: GuestAddress, syndrome: u32, places: &Places) -> Thunk {
    use Gpr::{Rax, Rcx};
    use Size::{Dword, Qword};
    let mut code = Vec::with_capacity(512);
    let mut misses = Vec::new();
    let mut asm = Asm::new(&mut code);
    locate(&mut asm, places, GUARDED_ENTER, &mut misses);
    asm.mov_load(Qword, Rax, frame_field(offset_of!(GuestState, pan)));
    asm.alu_load(
        Alu::Cmp,
        Qword,
        Rax,
        record_field(offset_of!(Record, check)),
    );
    misses.push(asm.jcc(Cond::Ne));
    check_host(&mut asm, &mut misses);
    saved_state(&mut asm, places);
    asm.mov_load(Qword, Rcx, record_field(offset_of!(Record, spsr)));
    asm.mov_store(Qword, Mem::base(Rcx, 0), Rax);
    asm.mov_load(Qword, Rcx, record_field(offset_of!(Record, elr)));
    asm.mov_imm64(Rax, pc.0.wrapping_add(4));
    asm.mov_store(Qword, Mem::base(Rcx, 0), Rax);
    asm.mov_load(Qword, Rcx, record_field(offset_of!(Record, esr)));
    asm.mov_imm32(Rax, syndrome);
    asm.mov_store(Qword, Mem::base(Rcx, 0), Rax);
    asm.mov_load(Qword, Rcx, record_field(offset_of!(Record, aspsr)));
    asm.mov_load(Qword, Rax, record_field(offset_of!(Record, aspsr_value)));
    asm.mov_store(Qword, Mem::base(Rcx, 0), Rax);
    swap_stack(&mut asm);
    asm.mov_rm_imm32(Dword, frame_field(offset_of!(GuestState, daif)), 0xf);
    asm.place(Rcx, places.selected);
    asm.mov_rm_imm32(Qword, Mem::base(Rcx, 0), 1);
    commit(code, places, GUARDED_ENTER, misses, pc)
}

/// The thunk completing GEXIT at `pc`, else exiting at `pc` for the engine.
pub fn guarded_exit_thunk(pc: GuestAddress, places: &Places) -> Thunk {
    use Gpr::{Rax, Rcx};
    use Size::Qword;
    let mut code = Vec::with_capacity(512);
    let mut misses = Vec::new();
    let mut asm = Asm::new(&mut code);
    check_deadline(&mut asm, places, &mut misses);
    locate(&mut asm, places, GUARDED_EXIT, &mut misses);
    check_epoch(&mut asm, &mut misses);
    check_host(&mut asm, &mut misses);
    // ASPSR's guarded bit and SPSR's fixed bits are the record's.
    asm.mov_load(Qword, Rcx, record_field(offset_of!(Record, aspsr)));
    asm.mov_load(Qword, Rax, Mem::base(Rcx, 0));
    asm.alu_imm8(Alu::And, Qword, Rax, 1);
    asm.alu_load(
        Alu::Cmp,
        Qword,
        Rax,
        record_field(offset_of!(Record, aspsr_value)),
    );
    misses.push(asm.jcc(Cond::Ne));
    check_return_state(&mut asm, &mut misses);
    restore_pstate(&mut asm, places);
    swap_stack(&mut asm);
    commit(code, places, GUARDED_EXIT, misses, pc)
}

/// The branch to the original exit unless no interrupt line falls due by
/// time alone: a timer, device or PMU deadline that passed since the last
/// step leaves the unmask to the engine, which takes what fell due at once.
fn check_deadline(asm: &mut Asm<'_>, places: &Places, misses: &mut Vec<Patch>) {
    use Gpr::{Rax, Rdx};
    use Size::Qword;
    asm.place(Rax, places.lines_until);
    asm.alu_imm8(Alu::Cmp, Qword, Mem::base(Rax, 0), -1);
    let timeless = asm.jcc(Cond::E);
    asm.alu_imm8(Alu::Cmp, Qword, Mem::base(Rax, 0), 0);
    misses.push(asm.jcc(Cond::E));
    super::register_read::read_count(asm, places.count);
    asm.place(Rax, places.lines_until);
    asm.alu_load(Alu::Cmp, Qword, Rdx, Mem::base(Rax, 0));
    misses.push(asm.jcc(Cond::Ae));
    asm.bind(timeless);
}

/// The branch to the original exit unless no interrupt line changed since
/// native code was entered.
fn check_epoch(asm: &mut Asm<'_>, misses: &mut Vec<Patch>) {
    use Gpr::{Rax, Rcx};
    use Size::Qword;
    let epoch = offset_of!(super::chain::ChainControl, interrupt_epoch);
    let observed = offset_of!(super::chain::ChainControl, observed_interrupt_epoch);
    asm.place(Rcx, Place::new(Anchor::Control, epoch));
    asm.mov_load(Qword, Rax, Mem::base(Rcx, 0));
    asm.alu_load(
        Alu::Cmp,
        Qword,
        Rax,
        Mem::base(Rcx, (observed - epoch) as i32),
    );
    misses.push(asm.jcc(Cond::Ne));
}

/// SPSR into RAX, then the branches to the original exit unless its bits
/// `check_mask` names are the record's `check` and it unmasks no interrupt
/// waiting since entry. RDX at the record.
fn check_return_state(asm: &mut Asm<'_>, misses: &mut Vec<Patch>) {
    use Gpr::{Rax, Rcx};
    use Size::{Dword, Qword};
    asm.mov_load(Qword, Rcx, record_field(offset_of!(Record, spsr)));
    asm.mov_load(Qword, Rax, Mem::base(Rcx, 0));
    asm.mov(Qword, Rcx, Rax);
    asm.alu_load(
        Alu::And,
        Qword,
        Rcx,
        record_field(offset_of!(Record, check_mask)),
    );
    asm.alu_load(
        Alu::Cmp,
        Qword,
        Rcx,
        record_field(offset_of!(Record, check)),
    );
    misses.push(asm.jcc(Cond::Ne));
    // No interrupt waiting since entry is unmasked: I and F are bits 1 and
    // 0 of both.
    asm.mov(Qword, Rcx, Rax);
    asm.shift_imm(ShiftOp::Shr, Qword, Rcx, SPSR_DAIF_SHIFT);
    asm.unary(super::asm::Unary::Not, Dword, Rcx);
    asm.alu_load(
        Alu::And,
        Dword,
        Rcx,
        frame_field(offset_of!(GuestState, pending)),
    );
    asm.alu_imm8(Alu::And, Dword, Rcx, 0b11);
    misses.push(asm.jcc(Cond::Ne));
}

/// PSTATE's operands from the SPSR in RAX: NZCV, the masks, PAN, the
/// fields and the stack selection. RDX stays at the record.
fn restore_pstate(asm: &mut Asm<'_>, places: &Places) {
    use Gpr::{Rax, Rcx, Rdx};
    use Size::{Dword, Qword};
    asm.mov(Qword, Rcx, Rax);
    asm.shift_imm(ShiftOp::Shr, Qword, Rcx, ARM_SHIFT);
    asm.alu_imm8(Alu::And, Dword, Rcx, 0xf);
    asm.imul_imm32(Dword, Rcx, Rcx, PACK_MULTIPLIER as i32);
    asm.alu_imm32(Alu::And, Dword, Rcx, WORD_MASK as i32);
    asm.mov_store(Dword, frame_field(offset_of!(GuestState, nzcv)), Rcx);
    asm.mov(Qword, Rcx, Rax);
    asm.shift_imm(ShiftOp::Shr, Qword, Rcx, SPSR_DAIF_SHIFT);
    asm.alu_imm8(Alu::And, Dword, Rcx, 0xf);
    asm.mov_store(Dword, frame_field(offset_of!(GuestState, daif)), Rcx);
    asm.mov(Qword, Rcx, Rax);
    asm.shift_imm(ShiftOp::Shr, Qword, Rcx, SPSR_PAN_SHIFT);
    asm.alu_imm8(Alu::And, Dword, Rcx, 1);
    asm.mov_store(Qword, frame_field(offset_of!(GuestState, pan)), Rcx);
    asm.push(Rdx);
    asm.place(Rdx, places.fields);
    for (field, bit) in places.field_bits.into_iter().enumerate() {
        asm.mov(Qword, Rcx, Rax);
        asm.alu_imm32(Alu::And, Dword, Rcx, bit as i32);
        asm.mov_store(Qword, Mem::base(Rdx, (field * 8) as i32), Rcx);
    }
    asm.place(Rdx, places.selected);
    asm.mov(Qword, Rcx, Rax);
    asm.alu_imm8(Alu::And, Dword, Rcx, 1);
    asm.mov_store(Qword, Mem::base(Rdx, 0), Rcx);
    asm.pop(Rdx);
}

/// The SPSR an exception entry saves from the PE's operands into RAX:
/// NZCV from the flag word, the masks, PAN and the fields, with the
/// record's `spsr_fixed` bits (the mode and TCO). Clobbers RCX.
fn saved_state(asm: &mut Asm<'_>, places: &Places) {
    use Gpr::{Rax, Rcx};
    use Size::{Dword, Qword};
    asm.mov_load(Dword, Rax, frame_field(offset_of!(GuestState, nzcv)));
    asm.alu_imm32(Alu::And, Dword, Rax, WORD_MASK as i32);
    asm.imul_imm32(Dword, Rax, Rax, UNPACK_MULTIPLIER as i32);
    asm.alu_imm32(Alu::And, Dword, Rax, ARM_MASK as i32);
    asm.mov_load(Dword, Rcx, frame_field(offset_of!(GuestState, daif)));
    asm.shift_imm(ShiftOp::Shl, Dword, Rcx, SPSR_DAIF_SHIFT);
    asm.alu(Alu::Or, Qword, Rax, Rcx);
    asm.mov_load(Qword, Rcx, frame_field(offset_of!(GuestState, pan)));
    asm.shift_imm(ShiftOp::Shl, Qword, Rcx, SPSR_PAN_SHIFT);
    asm.alu(Alu::Or, Qword, Rax, Rcx);
    asm.place(Rcx, places.fields);
    for field in 0..3 {
        asm.alu_load(Alu::Or, Qword, Rax, Mem::base(Rcx, field * 8));
    }
    asm.alu_load(
        Alu::Or,
        Qword,
        Rax,
        record_field(offset_of!(Record, spsr_fixed)),
    );
}

/// The thunk completing `SVC #immediate` at `pc` from EL0 to EL2 in the
/// host, or within EL2, whose ESR is `syndrome`, else exiting at `pc` for
/// the engine (`Cpu::take`). The record fixes everything but the operands:
/// The branch to the original exit unless the PAC service's tag words and
/// generation are the record's, where it swaps them; RAX and RCX are
/// clobbered.
fn check_pointer(asm: &mut Asm<'_>, places: &Places, misses: &mut Vec<Patch>) {
    use Gpr::{Rax, Rcx};
    use Size::Qword;
    asm.alu_imm8(
        Alu::Cmp,
        Qword,
        record_field(offset_of!(Record, pointer_swap)),
        0,
    );
    let kept = asm.jcc(Cond::E);
    asm.place(Rcx, places.pointer_generation);
    asm.mov_load(Qword, Rax, Mem::base(Rcx, 0));
    asm.alu_load(
        Alu::Cmp,
        Qword,
        Rax,
        record_field(offset_of!(Record, pointer_generation)),
    );
    misses.push(asm.jcc(Cond::Ne));
    asm.place(Rcx, places.pointer_tags);
    for word in 0..super::pointer::TAG_WORDS {
        asm.mov_load(Qword, Rax, Mem::base(Rcx, (word * 8) as i32));
        asm.alu_load(
            Alu::Cmp,
            Qword,
            Rax,
            record_field(offset_of!(Record, pointer_from) + word * 8),
        );
        misses.push(asm.jcc(Cond::Ne));
    }
    asm.bind(kept);
}

/// The PAC service's tag words become the record's `pointer_to` where it
/// swaps them; RAX and RCX are clobbered.
fn swap_pointer(asm: &mut Asm<'_>, places: &Places) {
    use Gpr::{Rax, Rcx};
    use Size::Qword;
    asm.alu_imm8(
        Alu::Cmp,
        Qword,
        record_field(offset_of!(Record, pointer_swap)),
        0,
    );
    let kept = asm.jcc8(Cond::E);
    asm.place(Rcx, places.pointer_tags);
    for word in 0..super::pointer::TAG_WORDS {
        asm.mov_load(
            Qword,
            Rax,
            record_field(offset_of!(Record, pointer_to) + word * 8),
        );
        asm.mov_store(Qword, Mem::base(Rcx, (word * 8) as i32), Rax);
    }
    asm.bind(kept);
}

/// SPSR takes NZCV, the masks, PAN and the fields with the record's mode;
/// ELR the next instruction; ESR the syndrome; the SP moves to the
/// target's; every exception is masked; PAN, UAO and SSBS become the
/// record's (with PSTATE.PAN the record's `check`), DIT stays; and the PE
/// continues at the record's vector.
pub fn supervisor_call_thunk(pc: GuestAddress, syndrome: u32, places: &Places) -> Thunk {
    use Gpr::{Rax, Rcx};
    use Size::{Dword, Qword};
    let mut code = Vec::with_capacity(512);
    let mut misses = Vec::new();
    let mut asm = Asm::new(&mut code);
    locate(&mut asm, places, SUPERVISOR_CALL, &mut misses);
    asm.mov_load(Qword, Rax, frame_field(offset_of!(GuestState, pan)));
    asm.alu_load(
        Alu::Cmp,
        Qword,
        Rax,
        record_field(offset_of!(Record, check)),
    );
    misses.push(asm.jcc(Cond::Ne));
    check_host(&mut asm, &mut misses);
    check_pointer(&mut asm, places, &mut misses);
    swap_pointer(&mut asm, places);
    saved_state(&mut asm, places);
    asm.mov_load(Qword, Rcx, record_field(offset_of!(Record, spsr)));
    asm.mov_store(Qword, Mem::base(Rcx, 0), Rax);
    asm.mov_load(Qword, Rcx, record_field(offset_of!(Record, elr)));
    asm.mov_imm64(Rax, pc.0.wrapping_add(4));
    asm.mov_store(Qword, Mem::base(Rcx, 0), Rax);
    asm.mov_load(Qword, Rcx, record_field(offset_of!(Record, esr)));
    asm.mov_imm32(Rax, syndrome);
    asm.mov_store(Qword, Mem::base(Rcx, 0), Rax);
    swap_stack(&mut asm);
    asm.mov_rm_imm32(Dword, frame_field(offset_of!(GuestState, daif)), 0xf);
    asm.place(Rcx, places.selected);
    asm.mov_rm_imm32(Qword, Mem::base(Rcx, 0), 1);
    asm.mov_load(Qword, Rax, record_field(offset_of!(Record, pan)));
    asm.mov_store(Qword, frame_field(offset_of!(GuestState, pan)), Rax);
    asm.place(Rcx, places.fields);
    for field in 0..3 {
        let word = Mem::base(Rcx, (field * 8) as i32);
        asm.mov_load(Qword, Rax, word);
        asm.alu_load(
            Alu::And,
            Qword,
            Rax,
            record_field(offset_of!(Record, fields_keep) + field * 8),
        );
        asm.alu_load(
            Alu::Or,
            Qword,
            Rax,
            record_field(offset_of!(Record, fields_set) + field * 8),
        );
        asm.mov_store(Qword, word, Rax);
    }
    commit(code, places, SUPERVISOR_CALL, misses, pc)
}

/// The thunk completing ERET at `pc` from EL2 to EL0 in the host or to EL2,
/// else exiting at `pc` for the engine (`Cpu::exception_return`). As for
/// GEXIT: no deadline due, no notification since entry, SPSR's bits other
/// than the operands (NZCV, A, I, F, DIT, UAO, SSBS) the record's, so
/// neither an illegal return nor PACM, single-step or another EL, and no
/// waiting interrupt unmasked; and the engine holds no exclusive monitor of
/// its own (`Places::monitor`). PSTATE's operands come from SPSR, the PC
/// from ELR with its tag removed where the record says, the SP moves, the
/// frame's local monitor clears and the event register is set.
pub fn exception_return_thunk(pc: GuestAddress, places: &Places) -> Thunk {
    use Gpr::{Rax, Rcx};
    use Size::{Byte, Dword, Qword};
    let mut code = Vec::with_capacity(640);
    let mut misses = Vec::new();
    let mut asm = Asm::new(&mut code);
    check_deadline(&mut asm, places, &mut misses);
    locate(&mut asm, places, EXCEPTION_RETURN, &mut misses);
    check_epoch(&mut asm, &mut misses);
    asm.place(Rcx, places.monitor);
    asm.alu_imm8(Alu::Cmp, Qword, Mem::base(Rcx, 0), 0);
    misses.push(asm.jcc(Cond::Ne));
    check_host(&mut asm, &mut misses);
    // The target: ELR, its top byte bit 55's copy where the half's TBI
    // applies.
    asm.mov_load(Qword, Rcx, record_field(offset_of!(Record, elr)));
    asm.mov_load(Qword, Rcx, Mem::base(Rcx, 0));
    asm.mov(Qword, Rax, Rcx);
    asm.shift_imm(ShiftOp::Shr, Qword, Rax, 55);
    asm.alu_imm8(Alu::And, Dword, Rax, 1);
    let upper = asm.jcc8(Cond::Ne);
    asm.alu_imm8(Alu::Cmp, Qword, record_field(offset_of!(Record, untag)), 0);
    let lower_kept = asm.jcc8(Cond::E);
    let lower_untag = asm.jmp8();
    asm.bind(upper);
    asm.alu_imm8(
        Alu::Cmp,
        Qword,
        record_field(offset_of!(Record, untag) + 8),
        0,
    );
    let upper_kept = asm.jcc8(Cond::E);
    asm.bind(lower_untag);
    asm.shift_imm(ShiftOp::Shl, Qword, Rcx, 8);
    asm.shift_imm(ShiftOp::Sar, Qword, Rcx, 8);
    asm.bind(lower_kept);
    asm.bind(upper_kept);
    asm.mov_store(Qword, record_field(offset_of!(Record, vector)), Rcx);
    check_pointer(&mut asm, places, &mut misses);
    check_return_state(&mut asm, &mut misses);
    restore_pstate(&mut asm, places);
    swap_pointer(&mut asm, places);
    swap_stack(&mut asm);
    asm.mov_rm_imm32(
        Dword,
        frame_field(offset_of!(GuestState, exclusive_bytes)),
        0,
    );
    asm.mov_load(Qword, Rcx, record_field(offset_of!(Record, event)));
    asm.mov_imm32(Rax, 1);
    asm.mov_store(Byte, Mem::base(Rcx, 0), Rax);
    commit(code, places, EXCEPTION_RETURN, misses, pc)
}

#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux")
))]
mod tests {
    use super::*;
    use crate::x86_64::chain::ChainControl;
    use crate::x86_64::register_read::{Count, CountState};
    use alloc::boxed::Box;

    #[repr(C)]
    #[derive(Debug, PartialEq, Eq)]
    struct Exit {
        pc: u64,
        kind: u64,
    }

    /// The engine's words a thunk reads besides the table.
    #[repr(C)]
    #[derive(Debug, Default)]
    struct Words {
        state: u64,
        selected: u64,
        fields: [u64; 3],
        status: u64,
        lines_until: u64,
        monitor: u64,
        pointer_tags: [u64; crate::x86_64::pointer::TAG_WORDS],
        pointer_generation: u64,
    }

    /// A level's GXF registers and two stack slots.
    #[repr(C)]
    #[derive(Debug, Default)]
    struct Bank {
        spsr: u64,
        elr: u64,
        esr: u64,
        aspsr: u64,
        enter: u64,
        stacks: [u64; 2],
    }

    const PC: GuestAddress = GuestAddress(0xffff_fe00_0000_8000);
    const FIELD_BITS: [u32; 3] = [1 << 24, 1 << 23, 1 << 12];
    const PLACES: Places = Places {
        state: Place::new(Anchor::Mirrors, offset_of!(Words, state)),
        table: Place::new(Anchor::Pointer, 0),
        selected: Place::new(Anchor::Mirrors, offset_of!(Words, selected)),
        fields: Place::new(Anchor::Mirrors, offset_of!(Words, fields)),
        field_bits: FIELD_BITS,
        interrupt_status: Place::new(Anchor::Mirrors, offset_of!(Words, status)),
        lines_until: Place::new(Anchor::Mirrors, offset_of!(Words, lines_until)),
        count: Place::new(Anchor::Count, 0),
        monitor: Place::new(Anchor::Mirrors, offset_of!(Words, monitor)),
        pointer_tags: Place::new(Anchor::Mirrors, offset_of!(Words, pointer_tags)),
        pointer_generation: Place::new(Anchor::Mirrors, offset_of!(Words, pointer_generation)),
    };
    const MISS: Exit = Exit {
        pc: PC.0,
        kind: ExitKind::System as u64,
    };

    /// Run a thunk on the host as the engine enters native code.
    fn run(code: &[u8], state: &mut GuestState) -> Exit {
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
        let code = super::super::pinned::entered(code);
        assert!(code.len() <= PAGE);
        // SAFETY: private anonymous memory, RX after the copy, unmapped after
        // the call; the thunk touches only `state` and the boxes its anchors
        // and record name, which outlive it.
        unsafe {
            let page = mmap(core::ptr::null_mut(), PAGE, 3, 2 | MAP_ANONYMOUS, -1, 0);
            assert_ne!(page as usize, usize::MAX);
            core::ptr::copy_nonoverlapping(code.as_ptr(), page.cast(), code.len());
            assert_eq!(mprotect(page, PAGE, 5), 0);
            let entry: unsafe extern "sysv64" fn(*mut GuestState) -> Exit =
                core::mem::transmute(page);
            let exit = entry(state);
            assert_eq!(munmap(page, PAGE), 0);
            exit
        }
    }

    struct Rig {
        state: Box<GuestState>,
        words: Box<Words>,
        table: Box<Table>,
        control: Box<ChainControl>,
        bank: Box<Bank>,
        clock: Box<CountState>,
        reader: Box<Count>,
    }

    impl Rig {
        fn new() -> Self {
            let mut rig = Self {
                state: Box::default(),
                words: Box::default(),
                table: Box::default(),
                control: Box::default(),
                bank: Box::default(),
                clock: Box::default(),
                reader: Box::default(),
            };
            rig.clock
                .ceiling
                .store(u64::MAX, core::sync::atomic::Ordering::Relaxed);
            rig.reader.state = &*rig.clock as *const CountState as u64;
            rig.state.anchors[Anchor::Count] = &*rig.reader as *const Count as u64;
            rig.state.anchors[Anchor::Mirrors] = &*rig.words as *const Words as u64;
            rig.state.anchors[Anchor::Pointer] = &*rig.table as *const Table as u64;
            rig.state.anchors[Anchor::Control] = &*rig.control as *const ChainControl as u64;
            rig
        }

        /// The record of `kind` from `state`, holding for this rig.
        fn record(&mut self, state: usize, kind: usize, to: u64) -> &mut Record {
            let bank = &mut *self.bank as *mut Bank;
            let frame = &*self.state as *const GuestState as u64;
            let generation = self.table.generation;
            let record = &mut self.table.records[state][kind];
            let at = |field: *mut u64| field as u64;
            // SAFETY: field addresses of the live bank box, never read here.
            unsafe {
                *record = Record {
                    generation,
                    frame,
                    to,
                    cells: 0x1234_5000,
                    contexts: [11, 12],
                    salts: 0x0000_0007_0000_0005,
                    sp_save: at(&raw mut (*bank).stacks[0]),
                    sp_load: at(&raw mut (*bank).stacks[1]),
                    spsr: at(&raw mut (*bank).spsr),
                    elr: at(&raw mut (*bank).elr),
                    esr: at(&raw mut (*bank).esr),
                    aspsr: at(&raw mut (*bank).aspsr),
                    ..Record::default()
                };
            }
            record.close = [&raw const record.scratch as u64; 3];
            record
        }

        fn run(&mut self, thunk: &Thunk) -> Exit {
            run(&thunk.code, &mut self.state)
        }

        /// What every completion leaves of the common commit.
        fn committed(&self, kind: usize, to: u64) {
            assert_eq!(self.words.state, to);
            assert_eq!(self.table.counts[kind], 1);
            assert_eq!(self.words.status, 0);
            assert_eq!(self.state.anchors[Anchor::Cells], 0x1234_5000);
            assert_eq!(self.control.contexts, [11, 12]);
            assert_eq!(self.control.salts, [5, 7]);
        }
    }

    const ENTER: u64 = 0xffff_fe00_0001_0000;

    fn entering() -> Rig {
        let mut rig = Rig::new();
        rig.bank.enter = ENTER;
        rig.bank.stacks = [0, 0x7000];
        rig.words.state = 2;
        rig.words.selected = 1;
        rig.words.fields = [1 << 24, 0, 0];
        rig.words.status = 9;
        rig.state.nzcv = 0xa000_0000;
        rig.state.daif = 0b0011;
        rig.state.pan = 1;
        rig.state.sp = 0x5000;
        let enter = &raw const rig.bank.enter as u64;
        let record = rig.record(2, GUARDED_ENTER, 6);
        record.selected = 1;
        record.check = 1;
        record.target = enter;
        record.spsr_fixed = (2 << 2) | 1;
        rig
    }

    /// GENTER banks PSTATE as the frame holds it, the next instruction and
    /// the immediate, and continues at the entry point on the level's stack
    /// with every exception masked, the record's state current.
    #[test]
    fn guarded_enter_completes_from_the_record() {
        let syndrome = 0xfe01_0003;
        for chain in [0, 1] {
            let mut rig = entering();
            rig.table.chain = chain;
            let thunk = guarded_enter_thunk(PC, syndrome, &PLACES);
            assert!(thunk.dynamic.is_some() && thunk.next.is_none());
            let exit = rig.run(&thunk);
            assert_eq!(
                exit,
                Exit {
                    pc: ENTER,
                    kind: ExitKind::Return as u64,
                }
            );
            let spsr = 0xa000_0000 | (0b0011 << 6) | (1 << 22) | (1 << 24) | 9;
            assert_eq!(rig.bank.spsr, spsr);
            assert_eq!(rig.bank.elr, PC.0 + 4);
            assert_eq!(rig.bank.esr, u64::from(syndrome));
            assert_eq!(rig.bank.aspsr, 0);
            assert_eq!(rig.bank.stacks[0], 0x5000);
            assert_eq!(rig.state.sp, 0x7000);
            assert_eq!(rig.state.daif, 0xf);
            assert_eq!(rig.state.nzcv, 0xa000_0000);
            assert_eq!(rig.words.selected, 1);
            rig.committed(GUARDED_ENTER, 6);
        }
    }

    /// Without a record that holds for the frame, the state and the stack
    /// selection, GENTER exits with nothing written.
    #[test]
    fn guarded_enter_leaves_the_rest_to_the_engine() {
        let thunk = guarded_enter_thunk(PC, 0xfe01_0000, &PLACES);
        let changes: [fn(&mut Rig); 5] = [
            |rig| rig.table.forget(),
            |rig| rig.words.selected = 0,
            |rig| rig.state.pan = 0,
            |rig| rig.words.state = 6,
            |rig| rig.table.records[2][GUARDED_ENTER].frame += 8,
        ];
        for change in changes {
            let mut rig = entering();
            change(&mut rig);
            let selected = rig.words.selected;
            assert_eq!(rig.run(&thunk), MISS);
            assert_eq!(rig.bank.spsr, 0);
            assert_eq!(rig.state.sp, 0x5000);
            assert_eq!(rig.state.daif, 0b0011);
            assert_eq!(rig.words.selected, selected);
            assert_eq!(rig.table.counts, [0; KINDS]);
        }
    }

    const RETURN: u64 = 0xffff_fe00_0000_4004;
    /// SPSR's bits a GEXIT record fixes: M, D, BTYPE, PAN, TCO, IL, SS, PACM.
    const CHECKED: u64 =
        0x1f | (1 << 9) | (3 << 10) | (1 << 22) | (1 << 25) | (3 << 20) | (1 << 35);

    fn exiting() -> Rig {
        let mut rig = Rig::new();
        let spsr = 0x6000_0000 | (0b0100 << 6) | (1 << 12) | 9;
        rig.bank.spsr = spsr;
        rig.bank.elr = RETURN;
        rig.bank.stacks = [0, 0x9000];
        rig.words.lines_until = u64::MAX;
        rig.words.state = 6;
        rig.words.selected = 1;
        rig.words.fields = [1 << 24, 1 << 23, 0];
        rig.state.nzcv = 0x8000_0000;
        rig.state.daif = 0xf;
        rig.state.pan = 1;
        rig.state.sp = 0x7000;
        let elr = &raw const rig.bank.elr as u64;
        let record = rig.record(6, GUARDED_EXIT, 2);
        record.selected = 1;
        record.target = elr;
        record.check_mask = CHECKED;
        record.check = spsr & CHECKED;
        rig
    }

    /// GEXIT restores NZCV, the masks, PAN, the fields and the stack from
    /// SPSR and continues at ELR, the record's state current.
    #[test]
    fn guarded_exit_completes_from_the_record() {
        let mut rig = exiting();
        let thunk = guarded_exit_thunk(PC, &PLACES);
        assert_eq!(
            rig.run(&thunk),
            Exit {
                pc: RETURN,
                kind: ExitKind::Return as u64,
            }
        );
        assert_eq!(rig.state.nzcv, 0x6000_0000);
        assert_eq!(rig.state.daif, 0b0100);
        assert_eq!(rig.state.pan, 0);
        assert_eq!(rig.words.fields, [0, 0, 1 << 12]);
        assert_eq!(rig.words.selected, 1);
        assert_eq!(rig.bank.stacks[0], 0x7000);
        assert_eq!(rig.state.sp, 0x9000);
        rig.committed(GUARDED_EXIT, 2);
    }

    /// GEXIT exits with nothing written when an interrupt waiting since
    /// entry would be unmasked, a line changed, ASPSR or a fixed SPSR bit
    /// differs from the record's, or the record does not hold.
    #[test]
    fn guarded_exit_leaves_the_rest_to_the_engine() {
        let thunk = guarded_exit_thunk(PC, &PLACES);
        let changes: [fn(&mut Rig); 7] = [
            |rig| rig.words.selected = 0,
            |rig| rig.state.pending = 0b10,
            |rig| rig.control.interrupt_epoch = 1,
            |rig| rig.bank.aspsr = 1,
            |rig| rig.bank.spsr |= 1 << 22,
            |rig| rig.bank.spsr |= 1 << 10,
            |rig| rig.table.forget(),
        ];
        for change in changes {
            let mut rig = exiting();
            change(&mut rig);
            assert_eq!(rig.run(&thunk), MISS);
            assert_eq!(rig.state.nzcv, 0x8000_0000);
            assert_eq!(rig.state.daif, 0xf);
            assert_eq!(rig.state.sp, 0x7000);
            assert_eq!(rig.words.state, 6);
            assert_eq!(rig.table.counts, [0; KINDS]);
        }
        // An interrupt that stays masked does not stop it.
        let mut rig = exiting();
        rig.state.pending = 0b100;
        assert_eq!(rig.run(&thunk).kind, ExitKind::Return as u64);
    }

    /// GEXIT completes only before the count at which time alone could
    /// raise an interrupt line: from there the engine unmasks, and takes
    /// what fell due at once, as for an engine-path GEXIT.
    #[test]
    fn guarded_exit_leaves_a_due_deadline_to_the_engine() {
        use core::sync::atomic::Ordering::Relaxed;
        let thunk = guarded_exit_thunk(PC, &PLACES);
        for (count, until, completes) in [
            (99, 100, true),
            (100, 100, false),
            (5, 0, false),
            (1 << 40, u64::MAX, true),
        ] {
            let mut rig = exiting();
            rig.words.lines_until = until;
            rig.clock.offset.store(count, Relaxed);
            let exit = rig.run(&thunk);
            if completes {
                assert_eq!(exit.pc, RETURN, "count {count} until {until}");
                rig.committed(GUARDED_EXIT, 2);
            } else {
                assert_eq!(exit, MISS, "count {count} until {until}");
                assert_eq!((rig.words.state, rig.state.sp), (6, 0x7000));
            }
        }
    }

    /// A forget that keeps some kinds keeps only those of them that held;
    /// the rest, and a record that had stopped holding, do not hold.
    #[test]
    fn forgetting_but_some_kinds_keeps_only_their_holding_records() {
        let mut table = Table::default();
        let generation = table.generation;
        for kind in 0..KINDS {
            table.records[3][kind].generation = generation;
        }
        table.records[5][GUARDED_EXIT].generation = generation + 7;
        table.forget_but(&[GUARDED_ENTER, GUARDED_EXIT]);
        assert_ne!(table.generation, generation);
        let holds = |table: &Table, state: usize, kind: usize| {
            table.records[state][kind].generation == table.generation
        };
        assert!(holds(&table, 3, GUARDED_ENTER) && holds(&table, 3, GUARDED_EXIT));
        assert!(!holds(&table, 3, SUPERVISOR_CALL) && !holds(&table, 3, EXCEPTION_RETURN));
        assert!(!holds(&table, 5, GUARDED_EXIT));
        table.forget();
        assert!(!holds(&table, 3, GUARDED_ENTER));
    }
}
