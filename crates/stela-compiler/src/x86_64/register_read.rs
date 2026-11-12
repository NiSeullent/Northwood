//! Native completion of MRS reads of a System register whose value the
//! system-mode engine keeps in a [`Cell`] at a fixed host address.
//!
//! The MRS ends its block with a System exit. A [`thunk`] the engine
//! installs over that exit completes it natively while the cell's
//! `readable` word is nonzero: it writes `value` to Xt (nothing for XZR) and
//! leaves through a direct exit to the next instruction, which the engine
//! may chain like a block's. Otherwise it takes the original exit.
//!
//! The engine sets `readable` only while the read at the current EL is the
//! plain register read of its AARCHMRS accessor, neither UNDEFINED nor
//! trapped nor redirected, with `value` the register's contents, and clears
//! it whenever a System register write, an exception, an exception return
//! or a PE switch may change either. Thunks clobber only RAX and flags.
//!
//! A [`stored_thunk`] reads the live register instead, through the address
//! the cell's `value` holds, so that what an exception or an MSR stores needs
//! no System exit to reach it; a [`frame_thunk`] reads a field of the frame.
//!
//! A [`stored_write_thunk`] completes MSR of a register the same way: it
//! writes through that address, keeping the register's fixed bits, and a
//! [`frame_write_thunk`] writes a field of the frame. It also clobbers RCX
//! and RDX.
//!
//! A [`pan_read_thunk`] completes MRS PAN from the frame, where a
//! [`pan_thunk`] leaves it.
//!
//! A [`key_write_thunk`] completes MSR of half of a PAC key, closing the keys
//! to PAC thunks when it changes one.
//!
//! A [`checked_write_thunk`] completes MSR of a register that takes any value
//! without the bits it refuses, which the engine decides.
//!
//! A [`stack_select_thunk`] completes MSR SPSel: the frame's SP moves to the
//! stack it leaves and takes the one it selects. The selection is a word of
//! the engine's while native code runs, which reads of SP_EL0 consult
//! ([`selected_stored_thunk`], [`selected_write_thunk`]): they are plain only
//! with SP_ELx selected, whatever their cell says.
//!
//! A [`mask_thunk`] completes a write of PSTATE's D, A, I and F masks in the
//! frame, which the engine takes back when native code returns, unless the
//! write changes D, which debug state follows, or clears the mask of an
//! interrupt waiting since native code was entered ([`GuestState::pending`]):
//! one that comes later stops native code as ever. It also clobbers RCX and
//! RDX, as a [`mask_read_thunk`] reads the masks from the frame.
//!
//! A [`code_page_thunk`] completes IC IVAU of the page the cell's `value`
//! names, which a kernel cleans line by line: the first line's invalidation
//! was the whole page's. It also clobbers RCX.
//!
//! A [`count_thunk`] reads a count instead: the system count a clock
//! publishes in a [`CountState`] at the host's TSC, through the reader's
//! [`Count`], less the cell's `value`. It also clobbers RCX and RDX, which
//! are dead at every block exit (the direct exit returns in RAX and RDX,
//! and chaining uses RCX). An [`energy_thunk`] scales the count it reads
//! into an energy accumulator's reading.
//!
//! A [`send_event_thunk`] sets the event register byte the cell's `value`
//! names, and an [`event_wait_thunk`] consumes a set one; both clobber RCX
//! and RDX.
//!
//! A [`cycle_counter_thunk`] computes a cycle counter at the count where
//! its cell went stale but the read stays plain, and opens the cell on it;
//! an [`instruction_counter_thunk`] does the same for a retired-instruction
//! counter from the instructions native code retired before it.
//!
//! An [`interrupt_status_thunk`] reads ISR_EL1 from the interrupts waiting
//! at entry while no line changed since; it clobbers RCX and RDX.
use super::asm::{Alu, Asm, Cond, Gpr, Mem, Patch, Rm, ShiftOp, Size, Unary};
use super::pointer::Thunk;
use crate::{GuestAddress, GuestState, Place, aarch64::Register, block::ExitKind};
use alloc::vec::Vec;
use core::sync::atomic::AtomicU64;

/// What a thunk reads.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Cell {
    /// Nonzero while native code may complete the read with `value`.
    pub readable: u64,
    pub value: u64,
}

const VALUE: i8 = core::mem::offset_of!(Cell, value) as i8;
/// Bits of an address within the page a [`code_page_thunk`] compares.
const CODE_PAGE_BITS: u8 = 12;
const _: () = assert!(core::mem::offset_of!(Cell, readable) == 0);

/// A clock's count as every reader computes it: at host count `host`
/// (`((tsc - base) * per_tsc) >> 64`), `min(host + offset, ceiling)`, with
/// wrapping addition. A writer makes `sequence` odd, changes `offset` and
/// `ceiling` and makes it even again; a reader takes both from one even
/// sequence and samples the TSC before that sequence ends, so readers on
/// different cores see one count (a seqlock).
#[repr(C, align(64))]
#[derive(Debug, Default)]
pub struct CountState {
    pub sequence: AtomicU64,
    pub offset: AtomicU64,
    pub ceiling: AtomicU64,
}

const SEQUENCE: i8 = core::mem::offset_of!(CountState, sequence) as i8;
const STATE_OFFSET: i8 = core::mem::offset_of!(CountState, offset) as i8;
const CEILING: i8 = core::mem::offset_of!(CountState, ceiling) as i8;

/// One reader of a [`CountState`] at `state`: its TSC scale, and `last`,
/// the count it last read, below which it never reads, so its own count
/// never runs backwards. `seen_*` hold the read in progress.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Count {
    pub base: u64,
    pub per_tsc: u64,
    pub last: u64,
    pub state: u64,
    pub seen_sequence: u64,
    pub seen_offset: u64,
    pub seen_ceiling: u64,
}

const BASE: i8 = core::mem::offset_of!(Count, base) as i8;
const PER_TSC: i8 = core::mem::offset_of!(Count, per_tsc) as i8;
const LAST: i8 = core::mem::offset_of!(Count, last) as i8;
const STATE: i8 = core::mem::offset_of!(Count, state) as i8;
const SEEN_SEQUENCE: i8 = core::mem::offset_of!(Count, seen_sequence) as i8;
const SEEN_OFFSET: i8 = core::mem::offset_of!(Count, seen_offset) as i8;
const SEEN_CEILING: i8 = core::mem::offset_of!(Count, seen_ceiling) as i8;

/// Xt's storage in [`pinned`](super::pinned) code.
fn register(rt: u8) -> Rm {
    Asm::pinned(&mut Vec::new())
        .guest(Register::X(rt))
        .expect("X0..X30")
}

/// The thunk completing `MRS Xt, <register>` at `pc` from the cell at
/// `cell`, else exiting at `pc` for the engine.
pub fn thunk(pc: GuestAddress, rt: u8, cell: Place) -> Thunk {
    let (mut code, miss) = open(cell);
    if rt != 31 {
        let mut asm = Asm::new(&mut code);
        asm.mov_load(Size::Qword, Gpr::Rax, Mem::base_disp8(Gpr::Rax, VALUE));
        asm.mov_store(Size::Qword, register(rt), Gpr::Rax);
    }
    finish(code, miss, pc)
}

/// The thunk completing `MRS Xt, <register>` at `pc` from the storage the
/// `value` of the cell at `cell` points at, else exiting at `pc`.
pub fn stored_thunk(pc: GuestAddress, rt: u8, cell: Place) -> Thunk {
    let (mut code, miss) = open(cell);
    let mut asm = Asm::new(&mut code);
    asm.mov_load(Size::Qword, Gpr::Rax, Mem::base_disp8(Gpr::Rax, VALUE));
    asm.mov_load(Size::Qword, Gpr::Rax, Mem::base(Gpr::Rax, 0));
    asm.mov_store(Size::Qword, register(rt), Gpr::Rax);
    finish(code, miss, pc)
}

/// [`stored_thunk`] of a read that is plain only while the word at `select`
/// is nonzero.
pub fn selected_stored_thunk(pc: GuestAddress, rt: u8, cell: Place, select: Place) -> Thunk {
    let (mut code, miss) = open(cell);
    let unselected = selected(&mut code, select);
    let mut asm = Asm::new(&mut code);
    asm.mov_load(Size::Qword, Gpr::Rax, Mem::base_disp8(Gpr::Rax, VALUE));
    asm.mov_load(Size::Qword, Gpr::Rax, Mem::base(Gpr::Rax, 0));
    asm.mov_store(Size::Qword, register(rt), Gpr::Rax);
    finish_unless(code, [miss, unselected], pc)
}

/// Branch away while the word at `select` is zero.
fn selected(code: &mut Vec<u8>, select: Place) -> Patch {
    let mut asm = Asm::new(code);
    asm.place(Gpr::Rcx, select);
    asm.alu_imm8(Alu::Cmp, Size::Qword, Mem::base(Gpr::Rcx, 0), 0);
    asm.jcc(Cond::E)
}

/// The thunk completing `MRS Xt, <register>` at `pc` from the 32-bit frame
/// field at `offset` while the cell at `cell` is open, else exiting at `pc`.
pub fn frame_thunk(pc: GuestAddress, rt: u8, cell: Place, offset: usize) -> Thunk {
    let (mut code, miss) = open(cell);
    let mut asm = Asm::new(&mut code);
    let field = Mem::base_disp32(Gpr::Rdi, offset as i32);
    asm.mov_load(Size::Dword, Gpr::Rax, field);
    asm.mov_store(Size::Qword, register(rt), Gpr::Rax);
    finish(code, miss, pc)
}

/// The thunk completing `MRS Xt, PAN` at `pc` while the cell at `cell` is
/// open, else exiting at `pc`.
pub fn pan_read_thunk(pc: GuestAddress, rt: u8, cell: Place) -> Thunk {
    /// PAN's bit of the register.
    const PAN_SHIFT: u8 = 22;
    let (mut code, miss) = open(cell);
    let mut asm = Asm::new(&mut code);
    let field = frame_field(core::mem::offset_of!(GuestState, pan));
    asm.mov_load(Size::Qword, Gpr::Rax, field);
    asm.shift_imm(ShiftOp::Shl, Size::Qword, Gpr::Rax, PAN_SHIFT);
    asm.mov_store(Size::Qword, register(rt), Gpr::Rax);
    finish(code, miss, pc)
}

/// The bits of a register a write takes and those it always holds, as
/// [`Self::apply`] gives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Written {
    pub keep: u64,
    pub set: u64,
}

impl Written {
    /// Leave `Xt`'s value as the register stores it in RAX.
    fn apply(self, asm: &mut Asm<'_>, rt: u8) {
        if rt == 31 {
            asm.alu(Alu::Xor, Size::Dword, Gpr::Rax, Gpr::Rax);
        } else {
            asm.mov_load(Size::Qword, Gpr::Rax, register(rt));
        }
        if self.keep != u64::MAX {
            asm.mov_imm64(Gpr::Rdx, self.keep);
            asm.alu(Alu::And, Size::Qword, Gpr::Rax, Gpr::Rdx);
        }
        if self.set != 0 {
            asm.mov_imm64(Gpr::Rdx, self.set);
            asm.alu(Alu::Or, Size::Qword, Gpr::Rax, Gpr::Rdx);
        }
    }
}

/// The thunk completing `MSR <register>, Xt` at `pc` through the storage the
/// `value` of the cell at `cell` points at, else exiting at `pc`.
pub fn stored_write_thunk(pc: GuestAddress, rt: u8, cell: Place, written: Written) -> Thunk {
    let (mut code, miss) = open(cell);
    let mut asm = Asm::new(&mut code);
    asm.mov_load(Size::Qword, Gpr::Rcx, Mem::base_disp8(Gpr::Rax, VALUE));
    written.apply(&mut asm, rt);
    asm.mov_store(Size::Qword, Mem::base(Gpr::Rcx, 0), Gpr::Rax);
    finish(code, miss, pc)
}

/// The thunk completing `MSR <register>, Xt` at `pc` when the value, as the
/// register takes it, equals what the storage the `value` of the cell at
/// `cell` points at holds already: such a write changes nothing. Any other
/// value, or a closed cell, exits at `pc`.
pub fn unchanged_write_thunk(pc: GuestAddress, rt: u8, cell: Place, written: Written) -> Thunk {
    let (mut code, miss) = open(cell);
    let mut asm = Asm::new(&mut code);
    asm.mov_load(Size::Qword, Gpr::Rcx, Mem::base_disp8(Gpr::Rax, VALUE));
    written.apply(&mut asm, rt);
    asm.alu_load(Alu::Cmp, Size::Qword, Gpr::Rax, Mem::base(Gpr::Rcx, 0));
    let changed = asm.jcc(Cond::Ne);
    finish_unless(code, [miss, changed], pc)
}

/// [`stored_write_thunk`] of a register that keeps what it is written, unless
/// the value has one of the bits `refused`.
pub fn checked_write_thunk(pc: GuestAddress, rt: u8, cell: Place, refused: u64) -> Thunk {
    let (mut code, miss) = open(cell);
    let mut asm = Asm::new(&mut code);
    asm.mov_load(Size::Qword, Gpr::Rcx, Mem::base_disp8(Gpr::Rax, VALUE));
    let all = Written {
        keep: u64::MAX,
        set: 0,
    };
    all.apply(&mut asm, rt);
    if rt == 31 || refused == 0 {
        asm.mov_store(Size::Qword, Mem::base(Gpr::Rcx, 0), Gpr::Rax);
        return finish(code, miss, pc);
    }
    asm.mov_imm64(Gpr::Rdx, refused);
    asm.test(Size::Qword, Gpr::Rax, Gpr::Rdx);
    let unplain = asm.jcc(Cond::Ne);
    asm.mov_store(Size::Qword, Mem::base(Gpr::Rcx, 0), Gpr::Rax);
    finish_unless(code, [miss, unplain], pc)
}

/// One thread switch a [`thread_switch_thunk`] completed: the thread
/// pointer it replaced and the one it wrote, the frame pointer (X29) and
/// link register (X30) then, and the MSR's PC.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Switch {
    pub previous: u64,
    pub next: u64,
    pub frame: u64,
    pub link: u64,
    pub pc: u64,
}

/// The thread switches native code completed since the engine last took
/// them, in order: `len` of `entries`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SwitchLog {
    pub len: u64,
    pub entries: [Switch; SWITCHES],
}

/// The switches a [`SwitchLog`] holds; a thunk finding it full exits.
pub const SWITCHES: usize = 8;

/// Where a [`thread_switch_thunk`] finds the register's read cells and the
/// engine's [`SwitchLog`]: `reads` is the read cell of the first of `banks`
/// banks, each `stride` bytes after the one before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadSwitch {
    pub reads: Place,
    pub stride: usize,
    pub banks: usize,
    pub log: Place,
}

/// Leave Xt's value in RAX (0 for XZR).
fn load_x(asm: &mut Asm<'_>, rt: u8) {
    if rt == 31 {
        asm.alu(Alu::Xor, Size::Dword, Gpr::Rax, Gpr::Rax);
    } else {
        asm.mov_load(Size::Qword, Gpr::Rax, register(rt));
    }
}

/// The thunk completing `MSR <thread pointer>, Xt` at `pc` through the
/// storage the `value` of the cell at `cell` points at, for a register that
/// keeps what it is written and has read cells of its own in every bank
/// (TPIDR_EL1): every open read cell takes the value, as the engine's
/// write updates them. A value that replaces another is logged in the
/// [`SwitchLog`] for the engine's diagnostics, which see the switch as an
/// engine-path write would show it; with the log full, or the cell closed,
/// it exits at `pc`. It also clobbers RCX and RDX.
pub fn thread_switch_thunk(pc: GuestAddress, rt: u8, cell: Place, places: &ThreadSwitch) -> Thunk {
    use Gpr::{Rax, Rcx, Rdx};
    use Size::Qword;
    let (mut code, miss) = open(cell);
    let mut asm = Asm::new(&mut code);
    asm.mov_load(Qword, Rcx, Mem::base_disp8(Rax, VALUE));
    load_x(&mut asm, rt);
    // The same value changes nothing: no cell, no thread.
    asm.alu_load(Alu::Cmp, Qword, Rax, Mem::base(Rcx, 0));
    let same = asm.jcc(Cond::E);
    asm.place(Rdx, places.log);
    let len = core::mem::offset_of!(SwitchLog, len) as i32;
    asm.alu_imm8(Alu::Cmp, Qword, Mem::base(Rdx, len), SWITCHES as i8);
    let full = asm.jcc(Cond::Ae);
    asm.mov_load(Qword, Rax, Mem::base(Rdx, len));
    asm.alu_imm8(Alu::Add, Qword, Mem::base(Rdx, len), 1);
    asm.imul_imm32(Qword, Rax, Rax, size_of::<Switch>() as i32);
    asm.alu(Alu::Add, Qword, Rdx, Rax);
    let entry = |field: usize| {
        Mem::base(
            Rdx,
            (core::mem::offset_of!(SwitchLog, entries) + field) as i32,
        )
    };
    asm.mov_load(Qword, Rax, Mem::base(Rcx, 0));
    asm.mov_store(Qword, entry(core::mem::offset_of!(Switch, previous)), Rax);
    load_x(&mut asm, 29);
    asm.mov_store(Qword, entry(core::mem::offset_of!(Switch, frame)), Rax);
    load_x(&mut asm, 30);
    asm.mov_store(Qword, entry(core::mem::offset_of!(Switch, link)), Rax);
    asm.mov_imm64(Rax, pc.0);
    asm.mov_store(Qword, entry(core::mem::offset_of!(Switch, pc)), Rax);
    load_x(&mut asm, rt);
    asm.mov_store(Qword, entry(core::mem::offset_of!(Switch, next)), Rax);
    asm.mov_store(Qword, Mem::base(Rcx, 0), Rax);
    // Every open read cell reads the new value; closed ones keep none.
    asm.place(Rdx, places.reads);
    for bank in 0..places.banks {
        let at = (bank * places.stride) as i32;
        asm.alu_imm8(Alu::Cmp, Qword, Mem::base(Rdx, at), 0);
        let closed = asm.jcc8(Cond::E);
        asm.mov_store(Qword, Mem::base(Rdx, at + VALUE as i32), Rax);
        asm.bind(closed);
    }
    asm.bind(same);
    finish_unless(code, [miss, full], pc)
}

/// The thunk completing `IC IVAU, Xt` at `pc` of an address in the page the
/// `value` of the cell at `cell` names, else exiting at `pc`.
pub fn code_page_thunk(pc: GuestAddress, rt: u8, cell: Place) -> Thunk {
    let (mut code, miss) = open(cell);
    let mut asm = Asm::new(&mut code);
    asm.mov_load(Size::Qword, Gpr::Rcx, Mem::base_disp8(Gpr::Rax, VALUE));
    asm.alu_load(Alu::Xor, Size::Qword, Gpr::Rcx, register(rt));
    asm.shift_imm(ShiftOp::Shr, Size::Qword, Gpr::Rcx, CODE_PAGE_BITS);
    let other = asm.jcc(Cond::Ne);
    finish_unless(code, [miss, other], pc)
}

/// [`stored_write_thunk`] of a write that is plain only while the word at
/// `select` is nonzero.
pub fn selected_write_thunk(
    pc: GuestAddress,
    rt: u8,
    cell: Place,
    select: Place,
    written: Written,
) -> Thunk {
    let (mut code, miss) = open(cell);
    let unselected = selected(&mut code, select);
    let mut asm = Asm::new(&mut code);
    asm.mov_load(Size::Qword, Gpr::Rcx, Mem::base_disp8(Gpr::Rax, VALUE));
    written.apply(&mut asm, rt);
    asm.mov_store(Size::Qword, Mem::base(Gpr::Rcx, 0), Gpr::Rax);
    finish_unless(code, [miss, unselected], pc)
}

/// The thunk completing `MSR SPSel, #select` at `pc` while the cell at
/// `cell` is open, its `value` the address of the level's SP_ELx. The word
/// at `current` holds the selection and `sp0` is the address of SP_EL0.
pub fn stack_select_thunk(
    pc: GuestAddress,
    select: bool,
    cell: Place,
    current: Place,
    sp0: Place,
) -> Thunk {
    use Gpr::{Rax, Rcx, Rdx};
    use Size::Qword;
    let (mut code, miss) = open(cell);
    let sp = super::pinned::host(Register::Sp).expect("SP is pinned");
    let mut asm = Asm::new(&mut code);
    asm.place(Rcx, current);
    asm.alu_imm8(Alu::Cmp, Qword, Mem::base(Rcx, 0), i8::from(select));
    let unchanged = asm.jcc(Cond::E);
    asm.mov_imm32(Rdx, u32::from(select));
    asm.mov_store(Qword, Mem::base(Rcx, 0), Rdx);
    asm.mov_load(Qword, Rax, Mem::base_disp8(Rax, VALUE));
    asm.place(Rcx, sp0);
    let (from, to) = if select { (Rcx, Rax) } else { (Rax, Rcx) };
    asm.mov_store(Qword, Mem::base(from, 0), sp);
    asm.mov_load(Qword, sp, Mem::base(to, 0));
    asm.bind(unchanged);
    finish(code, miss, pc)
}

/// [`stored_write_thunk`] of half of a PAC key. A value that changes it clears
/// the `words` tag words at `tags`, as a thread switch writes several halves
/// and only the last leaves a key worth opening: the next PAC instruction
/// exits and the engine reopens them under the whole keys.
pub fn key_write_thunk(pc: GuestAddress, rt: u8, cell: Place, tags: Place, words: usize) -> Thunk {
    let (mut code, miss) = open(cell);
    let mut asm = Asm::new(&mut code);
    asm.mov_load(Size::Qword, Gpr::Rcx, Mem::base_disp8(Gpr::Rax, VALUE));
    let all = Written {
        keep: u64::MAX,
        set: 0,
    };
    all.apply(&mut asm, rt);
    asm.alu_load(Alu::Cmp, Size::Qword, Gpr::Rax, Mem::base(Gpr::Rcx, 0));
    let unchanged = asm.jcc(Cond::E);
    asm.mov_store(Size::Qword, Mem::base(Gpr::Rcx, 0), Gpr::Rax);
    asm.place(Gpr::Rcx, tags);
    asm.alu(Alu::Xor, Size::Dword, Gpr::Rax, Gpr::Rax);
    for word in 0..words {
        let at = Mem::base_disp8(Gpr::Rcx, (word * size_of::<u64>()) as i8);
        asm.mov_store(Size::Qword, at, Gpr::Rax);
    }
    asm.bind(unchanged);
    finish(code, miss, pc)
}

/// The thunk completing `MSR <register>, Xt` at `pc` in the 32-bit frame
/// field at `offset` while the cell at `cell` is open, else exiting at `pc`.
pub fn frame_write_thunk(
    pc: GuestAddress,
    rt: u8,
    cell: Place,
    offset: usize,
    written: Written,
) -> Thunk {
    let (mut code, miss) = open(cell);
    let mut asm = Asm::new(&mut code);
    written.apply(&mut asm, rt);
    asm.mov_store(Size::Dword, frame_field(offset), Gpr::Rax);
    finish(code, miss, pc)
}

/// A write of PSTATE's masks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskWrite {
    /// MSR DAIFSet, #imm.
    Set(u8),
    /// MSR DAIFClr, #imm.
    Clear(u8),
    /// MSR DAIF, Xt.
    Register(u8),
}

impl MaskWrite {
    /// The write `word` makes, if it is one.
    pub fn of(word: u32) -> Option<Self> {
        let immediate = (word >> 8 & 0xf) as u8;
        match word & 0xffff_f0ff {
            0xd503_40df => Some(Self::Set(immediate)),
            0xd503_40ff => Some(Self::Clear(immediate)),
            _ if word & !0x1f == 0xd51b_4220 => Some(Self::Register((word & 0x1f) as u8)),
            _ => None,
        }
    }
}

/// PSTATE.D in [`GuestState::daif`].
const MASK_D: i8 = 8;
/// DAIF's bit 6, of F, the lowest mask.
const MASK_SHIFT: u8 = 6;

fn frame_field(offset: usize) -> Mem {
    Mem::base_disp32(Gpr::Rdi, offset as i32)
}

/// The thunk completing `MSR PAN, #pan` at `pc` while the cell at `cell` is
/// open, on a host that keeps PSTATE.PAN in RFLAGS.AC, else exiting at `pc`.
pub fn pan_thunk(pc: GuestAddress, pan: bool, cell: Place) -> Thunk {
    const CLAC: [u8; 3] = [0x0f, 0x01, 0xca];
    const STAC: [u8; 3] = [0x0f, 0x01, 0xcb];
    let (mut code, miss) = open(cell);
    let mut asm = Asm::new(&mut code);
    asm.mov_imm32(Gpr::Rax, u32::from(pan));
    let field = frame_field(core::mem::offset_of!(GuestState, pan));
    asm.mov_store(Size::Qword, field, Gpr::Rax);
    asm.fixed(if pan { &CLAC } else { &STAC });
    finish(code, miss, pc)
}

/// The thunk completing a DSB at `pc` while the cell at `cell` is open, that
/// is while the engine has no maintenance for it to complete, else exiting
/// at `pc`.
pub fn barrier_thunk(pc: GuestAddress, cell: Place) -> Thunk {
    let (mut code, miss) = open(cell);
    Asm::new(&mut code).mfence();
    finish(code, miss, pc)
}

/// The thunk completing CLREX at `pc` while the cell at `cell` is open and
/// the word its `value` points at is zero, that is while the PE's local
/// monitor lives in the frame alone: it clears the frame's monitor; else it
/// exits at `pc` for the engine to clear the rest. It also clobbers RCX.
pub fn clear_exclusive_thunk(pc: GuestAddress, cell: Place) -> Thunk {
    let (mut code, closed) = open(cell);
    let mut asm = Asm::new(&mut code);
    asm.mov_load(Size::Qword, Gpr::Rcx, Mem::base_disp8(Gpr::Rax, VALUE));
    asm.alu_imm8(Alu::Cmp, Size::Qword, Mem::base(Gpr::Rcx, 0), 0);
    let held = asm.jcc(Cond::Ne);
    let bytes = frame_field(core::mem::offset_of!(GuestState, exclusive_bytes));
    asm.mov_rm_imm32(Size::Dword, bytes, 0);
    finish_unless(code, [closed, held], pc)
}

/// The thunk completing SEVL at `pc` while the cell at `cell` is open: it
/// sets the event register, the byte the cell's `value` points at, else
/// exits at `pc`. It also clobbers RCX and RDX.
pub fn send_event_thunk(pc: GuestAddress, cell: Place) -> Thunk {
    let (mut code, miss) = open(cell);
    let mut asm = Asm::new(&mut code);
    asm.mov_load(Size::Qword, Gpr::Rcx, Mem::base_disp8(Gpr::Rax, VALUE));
    asm.mov_imm32(Gpr::Rdx, 1);
    asm.mov_store(Size::Byte, Mem::base(Gpr::Rcx, 0), Gpr::Rdx);
    finish(code, miss, pc)
}

/// The thunk completing WFE at `pc` while the cell at `cell` is open and the
/// event register, the byte the cell's `value` points at, is set: it clears
/// the register in one exchange, as another PE may set it, and completes;
/// else it exits at `pc` for the engine to wait. It also clobbers RCX and
/// RDX.
pub fn event_wait_thunk(pc: GuestAddress, cell: Place) -> Thunk {
    let (mut code, miss) = open(cell);
    let mut asm = Asm::new(&mut code);
    asm.mov_load(Size::Qword, Gpr::Rcx, Mem::base_disp8(Gpr::Rax, VALUE));
    asm.alu(Alu::Xor, Size::Dword, Gpr::Rdx, Gpr::Rdx);
    asm.xchg(Size::Byte, Mem::base(Gpr::Rcx, 0), Gpr::Rdx);
    asm.test(Size::Byte, Gpr::Rdx, Gpr::Rdx);
    let clear = asm.jcc(Cond::E);
    finish_unless(code, [miss, clear], pc)
}

/// The thunk completing `write` at `pc` in the frame while the cell at
/// `cell` is open, else exiting at `pc`.
pub fn mask_thunk(write: MaskWrite, pc: GuestAddress, cell: Place) -> Thunk {
    use Gpr::{Rax, Rcx, Rdx};
    use Size::{Dword, Qword};
    let (mut code, closed) = open(cell);
    let masks = frame_field(core::mem::offset_of!(GuestState, daif));
    let pending = frame_field(core::mem::offset_of!(GuestState, pending));
    let mut asm = Asm::new(&mut code);
    asm.mov_load(Dword, Rcx, masks);
    match write {
        MaskWrite::Set(immediate) => {
            asm.mov(Dword, Rax, Rcx);
            asm.alu_imm8(Alu::Or, Dword, Rax, immediate as i8);
        }
        MaskWrite::Clear(immediate) => {
            asm.mov(Dword, Rax, Rcx);
            asm.alu_imm8(Alu::And, Dword, Rax, !immediate as i8);
        }
        MaskWrite::Register(31) => asm.alu(Alu::Xor, Dword, Rax, Rax),
        MaskWrite::Register(rt) => {
            asm.mov_load(Qword, Rax, register(rt));
            asm.shift_imm(ShiftOp::Shr, Qword, Rax, MASK_SHIFT);
            asm.alu_imm8(Alu::And, Dword, Rax, 0xf);
        }
    }
    // RDX the masks that change, RCX those of them that clear.
    asm.mov(Dword, Rdx, Rax);
    asm.alu(Alu::Xor, Dword, Rdx, Rcx);
    asm.alu(Alu::And, Dword, Rcx, Rdx);
    asm.alu_imm8(Alu::And, Dword, Rdx, MASK_D);
    let debug = asm.jcc(Cond::Ne);
    asm.alu_load(Alu::And, Dword, Rcx, pending);
    let waiting = asm.jcc(Cond::Ne);
    asm.mov_store(Dword, masks, Rax);
    finish_unless(code, [closed, debug, waiting], pc)
}

/// The thunk completing `MRS Xt, DAIF` at `pc` from the frame while the cell
/// at `cell` is open, else exiting at `pc`.
pub fn mask_read_thunk(pc: GuestAddress, rt: u8, cell: Place) -> Thunk {
    let (mut code, miss) = open(cell);
    let mut asm = Asm::new(&mut code);
    let masks = frame_field(core::mem::offset_of!(GuestState, daif));
    asm.mov_load(Size::Dword, Gpr::Rax, masks);
    asm.shift_imm(ShiftOp::Shl, Size::Dword, Gpr::Rax, MASK_SHIFT);
    asm.mov_store(Size::Qword, register(rt), Gpr::Rax);
    finish(code, miss, pc)
}

/// The thunk completing `MRS Xt, <count>` at `pc` with the count of the
/// reader at `count` less the `value` of the cell at `cell`, else exiting
/// at `pc`. It spins while a writer publishes; `lfence` keeps the TSC
/// sample between the two sequence reads (Intel SDM Vol. 2B, RDTSC).
pub fn count_thunk(pc: GuestAddress, rt: u8, cell: Place, count: Place) -> Thunk {
    use Gpr::{Rax, Rdx};
    use Size::Qword;
    let (mut code, miss) = open(cell);
    let mut asm = Asm::new(&mut code);
    read_count(&mut asm, count);
    if rt != 31 {
        asm.place(Rax, cell);
        asm.alu_load(Alu::Sub, Qword, Rdx, Mem::base_disp8(Rax, VALUE));
        asm.mov_store(Qword, register(rt), Rdx);
    }
    finish(code, miss, pc)
}

/// What an [`energy_thunk`]'s cell `value` points at: an accumulator's
/// reading is `(count - origin) * scale / frequency` of the system count.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Energy {
    pub origin: u64,
    pub scale: u64,
    pub frequency: u64,
}

/// The thunk completing `MRS Xt, <accumulator>` at `pc` with the reading the
/// [`Energy`] the `value` of the cell at `cell` points at gives at the count
/// of the reader at `count`, else exiting at `pc`: as the cell's register's
/// model computes it, so a reading never goes back. A quotient that would
/// not fit in 64 bits exits too. It also clobbers RCX and RDX.
pub fn energy_thunk(pc: GuestAddress, rt: u8, cell: Place, count: Place) -> Thunk {
    use Gpr::{Rax, Rcx, Rdx};
    use Size::Qword;
    let (mut code, miss) = open(cell);
    let mut asm = Asm::new(&mut code);
    read_count(&mut asm, count);
    asm.place(Rax, cell);
    asm.mov_load(Qword, Rcx, Mem::base_disp8(Rax, VALUE));
    let energy = |field| Mem::base_disp8(Rcx, field as i8);
    // The ticks since the origin, none before it.
    asm.mov(Qword, Rax, Rdx);
    asm.alu(Alu::Xor, Size::Dword, Rdx, Rdx);
    asm.alu_load(
        Alu::Sub,
        Qword,
        Rax,
        energy(core::mem::offset_of!(Energy, origin)),
    );
    asm.cmov(Cond::B, Qword, Rax, Rdx);
    asm.unary(
        Unary::Mul,
        Qword,
        energy(core::mem::offset_of!(Energy, scale)),
    );
    let frequency = energy(core::mem::offset_of!(Energy, frequency));
    asm.alu_load(Alu::Cmp, Qword, Rdx, frequency);
    let wide = asm.jcc(Cond::Ae);
    asm.unary(Unary::Div, Qword, frequency);
    if rt != 31 {
        asm.mov_store(Qword, register(rt), Rax);
    }
    finish_unless(code, [miss, wide], pc)
}

/// What a [`cycle_counter_thunk`] computes a cycle counter from, as the
/// engine's performance monitor counts it: `counter` at system count
/// `synced` with `remainder` of a cycle in 1/`frequency` units, then
/// `clock` cycles a `frequency` ticks (0 while the counter does not count),
/// the sum wrapped to `mask`; `remainder` is below `frequency`. `open` is nonzero while these are the
/// monitor's; `frequency` is never 0 then.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CycleCounter {
    pub open: u64,
    pub counter: u64,
    pub synced: u64,
    pub remainder: u64,
    pub clock: u64,
    pub frequency: u64,
    pub mask: u64,
}

impl CycleCounter {
    /// The counter at system count `now`, `None` where the thunk exits
    /// (closed, or a quotient that does not fit in 64 bits).
    pub fn at(&self, now: u64) -> Option<u64> {
        if self.open == 0 || self.frequency == 0 {
            return None;
        }
        let ticks = now.saturating_sub(self.synced);
        let scaled = u128::from(ticks) * u128::from(self.clock) + u128::from(self.remainder);
        let increment = u64::try_from(scaled / u128::from(self.frequency)).ok()?;
        Some(self.counter.wrapping_add(increment) & self.mask)
    }
}

/// The thunk completing `MRS Xt, <cycle counter>` at `pc`. While the cell
/// at `cell` is open it reads its `value` (the count the engine or this
/// thunk last took in the PE's state). While it is closed but the cell at
/// `plain` is open (the read is plain in the state: only the value went
/// stale, at a native transition), it computes the count at the reader at
/// `count` from the [`CycleCounter`] at `cycles` ([`CycleCounter::at`]),
/// opens `cell` on it, as an engine read does, and completes with it. Else
/// it exits at `pc`. It also clobbers RCX and RDX.
pub fn cycle_counter_thunk(
    pc: GuestAddress,
    rt: u8,
    cell: Place,
    plain: Place,
    cycles: Place,
    count: Place,
) -> Thunk {
    use Gpr::{Rax, Rcx, Rdx};
    use Size::{Dword, Qword};
    let field = |offset: usize| Mem::base_disp8(Rcx, offset as i8);
    let mut code = Vec::with_capacity(256);
    let mut asm = Asm::new(&mut code);
    asm.place(Rax, cell);
    asm.alu_imm8(Alu::Cmp, Qword, Mem::base(Rax, 0), 0);
    let stale = asm.jcc(Cond::E);
    asm.mov_load(Qword, Rdx, Mem::base_disp8(Rax, VALUE));
    let read = asm.jmp();
    asm.bind(stale);
    asm.place(Rax, plain);
    asm.alu_imm8(Alu::Cmp, Qword, Mem::base(Rax, 0), 0);
    let unplain = asm.jcc(Cond::E);
    asm.place(Rcx, cycles);
    asm.alu_imm8(
        Alu::Cmp,
        Qword,
        field(core::mem::offset_of!(CycleCounter, open)),
        0,
    );
    let closed = asm.jcc(Cond::E);
    read_count(&mut asm, count);
    asm.place(Rcx, cycles);
    // The ticks since the monitor's count, none before it.
    asm.mov(Qword, Rax, Rdx);
    asm.alu(Alu::Xor, Dword, Rdx, Rdx);
    asm.alu_load(
        Alu::Sub,
        Qword,
        Rax,
        field(core::mem::offset_of!(CycleCounter, synced)),
    );
    asm.cmov(Cond::B, Qword, Rax, Rdx);
    asm.unary(
        Unary::Mul,
        Qword,
        field(core::mem::offset_of!(CycleCounter, clock)),
    );
    asm.alu_load(
        Alu::Add,
        Qword,
        Rax,
        field(core::mem::offset_of!(CycleCounter, remainder)),
    );
    asm.alu_imm8(Alu::Adc, Qword, Rdx, 0);
    let frequency = field(core::mem::offset_of!(CycleCounter, frequency));
    asm.alu_load(Alu::Cmp, Qword, Rdx, frequency);
    let wide = asm.jcc(Cond::Ae);
    asm.unary(Unary::Div, Qword, frequency);
    asm.alu_load(
        Alu::Add,
        Qword,
        Rax,
        field(core::mem::offset_of!(CycleCounter, counter)),
    );
    asm.alu_load(
        Alu::And,
        Qword,
        Rax,
        field(core::mem::offset_of!(CycleCounter, mask)),
    );
    asm.mov(Qword, Rdx, Rax);
    asm.place(Rax, cell);
    asm.mov_store(Qword, Mem::base_disp8(Rax, VALUE), Rdx);
    asm.mov_imm32(Rcx, 1);
    asm.mov_store(Qword, Mem::base(Rax, 0), Rcx);
    asm.bind(read);
    if rt != 31 {
        asm.mov_store(Qword, register(rt), Rdx);
    }
    finish_unless(code, [unplain, closed, wide], pc)
}

/// What an [`instruction_counter_thunk`] computes a retired-instruction
/// counter from, as the engine's performance monitor counts it: `base`,
/// the count as native code was entered, plus the instructions retired
/// since, masked by `step` (all ones while the counter counts, 0 while it
/// does not), the sum wrapped to `mask`. `open` is nonzero while these are
/// the monitor's.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct InstructionCounter {
    pub open: u64,
    pub base: u64,
    pub step: u64,
    pub mask: u64,
}

impl InstructionCounter {
    /// The counter after native code retired `retired` instructions, `None`
    /// where the thunk exits (closed).
    pub fn at(&self, retired: u64) -> Option<u64> {
        (self.open != 0).then(|| self.base.wrapping_add(retired & self.step) & self.mask)
    }
}

/// The thunk completing `MRS Xt, <instruction counter>` at `pc`, the
/// instruction after `before` others of its block. While the cell at `cell`
/// is open it reads its `value`. While it is closed but the cell at `plain`
/// is open (only the value went stale, at a native transition), it computes
/// the count from the [`InstructionCounter`] at `counter`
/// ([`InstructionCounter::at`]) with the instructions native code retired
/// before this one: the chain's count of the blocks it left
/// (`ChainControl::retired`) and `before`. It opens `cell` on it, as an
/// engine read does, and completes with it. Else it exits at `pc`. It also
/// clobbers RCX and RDX.
pub fn instruction_counter_thunk(
    pc: GuestAddress,
    rt: u8,
    cell: Place,
    plain: Place,
    counter: Place,
    before: u32,
) -> Option<Thunk> {
    use Gpr::{Rax, Rcx, Rdx};
    use Size::Qword;
    let before = i32::try_from(before).ok()?;
    let field = |offset: usize| Mem::base_disp8(Rcx, offset as i8);
    let mut code = Vec::with_capacity(160);
    let mut asm = Asm::new(&mut code);
    asm.place(Rax, cell);
    asm.alu_imm8(Alu::Cmp, Qword, Mem::base(Rax, 0), 0);
    let stale = asm.jcc(Cond::E);
    asm.mov_load(Qword, Rdx, Mem::base_disp8(Rax, VALUE));
    let read = asm.jmp();
    asm.bind(stale);
    asm.place(Rax, plain);
    asm.alu_imm8(Alu::Cmp, Qword, Mem::base(Rax, 0), 0);
    let unplain = asm.jcc(Cond::E);
    asm.place(Rcx, counter);
    asm.alu_imm8(
        Alu::Cmp,
        Qword,
        field(core::mem::offset_of!(InstructionCounter, open)),
        0,
    );
    let closed = asm.jcc(Cond::E);
    asm.place(Rax, super::chain::CONTROL_PLACE);
    asm.mov_load(Qword, Rdx, Mem::base(Rax, super::chain::RETIRED_OFFSET));
    if before != 0 {
        asm.alu_imm32(Alu::Add, Qword, Rdx, before);
    }
    asm.alu_load(
        Alu::And,
        Qword,
        Rdx,
        field(core::mem::offset_of!(InstructionCounter, step)),
    );
    asm.alu_load(
        Alu::Add,
        Qword,
        Rdx,
        field(core::mem::offset_of!(InstructionCounter, base)),
    );
    asm.alu_load(
        Alu::And,
        Qword,
        Rdx,
        field(core::mem::offset_of!(InstructionCounter, mask)),
    );
    asm.place(Rax, cell);
    asm.mov_store(Qword, Mem::base_disp8(Rax, VALUE), Rdx);
    asm.mov_imm32(Rcx, 1);
    asm.mov_store(Qword, Mem::base(Rax, 0), Rcx);
    asm.bind(read);
    if rt != 31 {
        asm.mov_store(Qword, register(rt), Rdx);
    }
    Some(finish_unless(code, [unplain, closed], pc))
}

/// The thunk completing `MRS Xt, ISR_EL1` at `pc` where the register reads
/// the physical IRQ and FIQ lines alone (EL2): from the masks of the
/// interrupts waiting when native code was entered ([`GuestState::pending`],
/// I in bit 1 and F in bit 0, ISR's bits 7 and 6), while the lines are still
/// those. The cell at `cell` says so: open (`readable` nonzero), the epoch at
/// `epoch` still its `value` (no line changed since it was read), and the
/// count of the reader at `count` below its `readable` (no timer or device
/// deadline passed; `u64::MAX` names none and reads no count). Else it exits
/// at `pc`. It also clobbers RCX and RDX.
pub fn interrupt_status_thunk(
    pc: GuestAddress,
    rt: u8,
    cell: Place,
    epoch: Place,
    count: Place,
) -> Thunk {
    use Gpr::{Rax, Rcx, Rdx};
    use Size::{Dword, Qword};
    /// ISR_EL1.F, the lowest of the bits [`GuestState::pending`] gives.
    const ISR_SHIFT: u8 = 6;
    let (mut code, closed) = open(cell);
    let mut asm = Asm::new(&mut code);
    asm.place(Rcx, epoch);
    asm.mov_load(Qword, Rdx, Mem::base(Rcx, 0));
    asm.alu_load(Alu::Cmp, Qword, Rdx, Mem::base_disp8(Rax, VALUE));
    let changed = asm.jcc(Cond::Ne);
    asm.alu_imm8(Alu::Cmp, Qword, Mem::base(Rax, 0), -1);
    let timeless = asm.jcc(Cond::E);
    read_count(&mut asm, count);
    asm.place(Rax, cell);
    asm.alu_load(Alu::Cmp, Qword, Rdx, Mem::base(Rax, 0));
    let due = asm.jcc(Cond::Ae);
    asm.bind(timeless);
    if rt != 31 {
        let pending = frame_field(core::mem::offset_of!(GuestState, pending));
        asm.mov_load(Dword, Rax, pending);
        asm.alu_imm8(Alu::And, Dword, Rax, 0b11);
        asm.shift_imm(ShiftOp::Shl, Dword, Rax, ISR_SHIFT);
        asm.mov_store(Qword, register(rt), Rax);
    }
    finish_unless(code, [closed, changed, due], pc)
}

/// What a [`timer_thunk`] completes, of one generic timer channel whose
/// count is the system count with no offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimerAccess {
    /// MSR TVAL: the compare becomes the count plus Xt's low 32 bits,
    /// sign-extended.
    WriteValue,
    /// MSR CVAL: the compare becomes Xt.
    WriteCompare,
    /// MSR CTL: the control becomes Xt as the register takes it.
    WriteControl(Written),
    /// MRS CTL: ENABLE and IMASK as stored, ISTATUS while enabled with the
    /// count at or past the compare.
    ReadControl,
}

/// Where a [`timer_thunk`] finds its channel and the engine's words: two
/// cells whose `value` is the address of the compare's storage and the
/// control's, the count before which the engine looks at the lines again
/// ([`crate::x86_64::transition::Places::lines_until`]) and the count reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimerPlaces {
    pub compare: Place,
    pub control: Place,
    pub lines_until: Place,
    pub count: Place,
}

/// ENABLE and IMASK of a timer control.
const TIMER_ARMING: i8 = 0b11;
/// ENABLE alone: the timer's output follows the compare.
const TIMER_ARMED: i8 = 0b01;
/// ISTATUS's bit.
const TIMER_STATUS_SHIFT: u8 = 2;

/// The thunk completing `access` of a timer channel at `pc` with Xt, else
/// exiting at `pc`. Both cells must be open. A write completes only when it
/// leaves the channel's output low and its next rise no earlier than the
/// engine's `lines_until`, which must name a count (not 0): the output was
/// low (disabled, masked or before the compare), and once the write is done
/// it is disabled, masked or armed for a compare after the count and at or
/// past `lines_until`. The engine, which looks at the lines again at
/// `lines_until`, then finds the new compare in storage; anything else,
/// which could raise or lower the line or move its rise earlier, exits. A
/// count that passes the new compare after the thunk read it is at or past
/// `lines_until` too. It clobbers RAX, RCX and RDX.
pub fn timer_thunk(pc: GuestAddress, rt: u8, access: TimerAccess, places: TimerPlaces) -> Thunk {
    use Gpr::{Rax, Rcx, Rdx};
    use Size::{Dword, Qword};
    let (mut code, closed) = open(places.compare);
    let mut asm = Asm::new(&mut code);
    asm.place(Rax, places.control);
    asm.alu_imm8(Alu::Cmp, Qword, Mem::base(Rax, 0), 0);
    let control_closed = asm.jcc(Cond::E);
    let storage = |asm: &mut Asm<'_>, destination: Gpr, cell: Place| {
        asm.place(destination, cell);
        asm.mov_load(Qword, destination, Mem::base_disp8(destination, VALUE));
    };
    if access == TimerAccess::ReadControl {
        read_count(&mut asm, places.count);
        storage(&mut asm, Rax, places.compare);
        asm.alu_load(Alu::Cmp, Qword, Rdx, Mem::base(Rax, 0));
        asm.setcc(Cond::Ae, Rcx);
        asm.movzx(Dword, Rcx, Size::Byte, Rcx);
        storage(&mut asm, Rax, places.control);
        asm.mov_load(Qword, Rax, Mem::base(Rax, 0));
        asm.alu(Alu::And, Dword, Rcx, Rax);
        asm.shift_imm(ShiftOp::Shl, Dword, Rcx, TIMER_STATUS_SHIFT);
        asm.alu_imm8(Alu::And, Qword, Rax, !(1 << TIMER_STATUS_SHIFT));
        asm.alu(Alu::Or, Qword, Rax, Rcx);
        if rt != 31 {
            asm.mov_store(Qword, register(rt), Rax);
        }
        return finish_unless(code, [closed, control_closed], pc);
    }
    asm.place(Rcx, places.lines_until);
    asm.alu_imm8(Alu::Cmp, Qword, Mem::base(Rcx, 0), 0);
    let unknown = asm.jcc(Cond::E);
    read_count(&mut asm, places.count);
    // The output was low: not armed, or armed before its compare.
    storage(&mut asm, Rax, places.control);
    asm.mov_load(Dword, Rax, Mem::base(Rax, 0));
    asm.alu_imm8(Alu::And, Dword, Rax, TIMER_ARMING);
    asm.alu_imm8(Alu::Cmp, Dword, Rax, TIMER_ARMED);
    let was_quiet = asm.jcc(Cond::Ne);
    storage(&mut asm, Rax, places.compare);
    asm.alu_load(Alu::Cmp, Qword, Rdx, Mem::base(Rax, 0));
    let was_high = asm.jcc(Cond::Ae);
    asm.bind(was_quiet);
    // The new register value into RCX, and the compare it arms into RAX
    // while the control arms one, else straight to the store.
    let write = |asm: &mut Asm<'_>| {
        if rt == 31 {
            asm.alu(Alu::Xor, Dword, Rcx, Rcx);
        } else {
            asm.mov_load(Qword, Rcx, register(rt));
        }
    };
    let quiet = match access {
        TimerAccess::WriteValue | TimerAccess::WriteCompare => {
            if access == TimerAccess::WriteValue {
                if rt == 31 {
                    asm.alu(Alu::Xor, Dword, Rcx, Rcx);
                } else {
                    asm.movsxd(Rcx, register(rt));
                }
                asm.alu(Alu::Add, Qword, Rcx, Rdx);
            } else {
                write(&mut asm);
            }
            storage(&mut asm, Rax, places.control);
            asm.mov_load(Dword, Rax, Mem::base(Rax, 0));
            asm.alu_imm8(Alu::And, Dword, Rax, TIMER_ARMING);
            asm.alu_imm8(Alu::Cmp, Dword, Rax, TIMER_ARMED);
            let quiet = asm.jcc(Cond::Ne);
            asm.mov(Qword, Rax, Rcx);
            quiet
        }
        TimerAccess::WriteControl(written) => {
            write(&mut asm);
            if written.keep != u64::MAX {
                asm.mov_imm64(Rax, written.keep);
                asm.alu(Alu::And, Qword, Rcx, Rax);
            }
            if written.set != 0 {
                asm.mov_imm64(Rax, written.set);
                asm.alu(Alu::Or, Qword, Rcx, Rax);
            }
            asm.mov(Dword, Rax, Rcx);
            asm.alu_imm8(Alu::And, Dword, Rax, TIMER_ARMING);
            asm.alu_imm8(Alu::Cmp, Dword, Rax, TIMER_ARMED);
            let quiet = asm.jcc(Cond::Ne);
            storage(&mut asm, Rax, places.compare);
            asm.mov_load(Qword, Rax, Mem::base(Rax, 0));
            quiet
        }
        TimerAccess::ReadControl => unreachable!("handled above"),
    };
    // Armed: after the count, and no earlier than the engine looks again.
    asm.alu(Alu::Cmp, Qword, Rax, Rdx);
    let at_once = asm.jcc(Cond::Be);
    asm.place(Rdx, places.lines_until);
    asm.alu_load(Alu::Cmp, Qword, Rax, Mem::base(Rdx, 0));
    let earlier = asm.jcc(Cond::B);
    asm.bind(quiet);
    let target = match access {
        TimerAccess::WriteControl(_) => places.control,
        _ => places.compare,
    };
    storage(&mut asm, Rax, target);
    asm.mov_store(Qword, Mem::base(Rax, 0), Rcx);
    finish_unless(
        code,
        [closed, control_closed, unknown, was_high, at_once, earlier],
        pc,
    )
}

/// The count of the reader at `count` into RDX, as [`count_thunk`] reads
/// it. Clobbers RAX and RCX.
pub(super) fn read_count(asm: &mut Asm<'_>, count: Place) {
    use Gpr::{Rax, Rcx, Rdx};
    use Size::Qword;
    let field = |offset| Mem::base_disp8(Rcx, offset);
    let state = |offset| Mem::base_disp8(Rax, offset);
    asm.place(Rcx, count);
    let start = asm.jmp8();
    let busy = asm.len();
    asm.pause();
    asm.bind(start);
    asm.mov_load(Qword, Rax, field(STATE));
    asm.mov_load(Qword, Rdx, state(SEQUENCE));
    asm.test_imm(Size::Byte, Rdx, 1);
    let publishing = asm.jcc8(Cond::Ne);
    asm.bind_back(publishing, busy);
    asm.mov_store(Qword, field(SEEN_SEQUENCE), Rdx);
    asm.mov_load(Qword, Rdx, state(STATE_OFFSET));
    asm.mov_store(Qword, field(SEEN_OFFSET), Rdx);
    asm.mov_load(Qword, Rdx, state(CEILING));
    asm.mov_store(Qword, field(SEEN_CEILING), Rdx);
    asm.lfence();
    asm.rdtsc();
    asm.lfence();
    asm.shift_imm(ShiftOp::Shl, Qword, Rdx, 32);
    asm.alu(Alu::Or, Qword, Rax, Rdx);
    asm.alu_load(Alu::Sub, Qword, Rax, field(BASE));
    asm.unary(Unary::Mul, Qword, field(PER_TSC));
    asm.mov_load(Qword, Rax, field(STATE));
    asm.mov_load(Qword, Rax, state(SEQUENCE));
    asm.alu_load(Alu::Cmp, Qword, Rax, field(SEEN_SEQUENCE));
    let changed = asm.jcc8(Cond::Ne);
    asm.bind_back(changed, busy);
    asm.alu_load(Alu::Add, Qword, Rdx, field(SEEN_OFFSET));
    asm.alu_load(Alu::Cmp, Qword, Rdx, field(SEEN_CEILING));
    asm.cmov(Cond::A, Qword, Rdx, field(SEEN_CEILING));
    asm.alu_load(Alu::Cmp, Qword, Rdx, field(LAST));
    asm.cmov(Cond::B, Qword, Rdx, field(LAST));
    asm.mov_store(Qword, field(LAST), Rdx);
}

/// Load `cell` and branch to the miss while it is not readable.
fn open(cell: Place) -> (Vec<u8>, Patch) {
    let mut code = Vec::with_capacity(128);
    let mut asm = Asm::new(&mut code);
    asm.place(Gpr::Rax, cell);
    asm.alu_imm8(Alu::Cmp, Size::Qword, Mem::base(Gpr::Rax, 0), 0);
    let miss = asm.jcc(Cond::E);
    (code, miss)
}

/// The direct exit to the next instruction, then the `miss` branch's
/// target: the original exit at `pc`.
fn finish(code: Vec<u8>, miss: Patch, pc: GuestAddress) -> Thunk {
    finish_unless(code, [miss], pc)
}

/// [`finish`] with several branches to the original exit.
fn finish_unless<const N: usize>(mut code: Vec<u8>, misses: [Patch; N], pc: GuestAddress) -> Thunk {
    let next = code.len();
    let mut asm = Asm::new(&mut code);
    super::block::direct_exit_with(
        &mut asm,
        GuestAddress(pc.0.wrapping_add(4)),
        ExitKind::Fallthrough,
    );
    for miss in misses {
        asm.bind(miss);
    }
    super::block::direct_exit_with(&mut asm, pc, ExitKind::System);
    Thunk::new(code, Some(next), None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Anchor;
    use crate::x86_64::chain::{DIRECT_EXIT_BYTES, direct_exit};

    /// The anchors of these tests, each at the data itself.
    const CELL: Place = Place::new(Anchor::Mirrors, 0);
    const WORD: Place = Place::new(Anchor::Pointer, 0);
    const READER: Place = Place::new(Anchor::Count, 0);
    const SP0: Place = Place::new(Anchor::StackZero, 0);

    #[test]
    fn pan_writes_move_the_flag_and_the_frame() {
        for (pan, flag) in [(true, [0x0f, 0x01, 0xca]), (false, [0x0f, 0x01, 0xcb])] {
            let thunk = pan_thunk(GuestAddress(0x1000), pan, CELL);
            assert!(thunk.code.windows(3).any(|bytes| bytes == flag));
            assert!(thunk.next.is_some());
        }
    }

    #[test]
    fn mask_writes_name_their_form() {
        assert_eq!(MaskWrite::of(0xd503_47df), Some(MaskWrite::Set(7)));
        assert_eq!(MaskWrite::of(0xd503_4fdf), Some(MaskWrite::Set(15)));
        assert_eq!(MaskWrite::of(0xd503_47ff), Some(MaskWrite::Clear(7)));
        assert_eq!(MaskWrite::of(0xd51b_4229), Some(MaskWrite::Register(9)));
        for other in [0xd500_40bf_u32, 0xd500_409f, 0xd53b_4228, 0xd503_201f] {
            assert_eq!(MaskWrite::of(other), None, "{other:#x}");
        }
    }

    /// The emitted code runs on the host: masks change in the frame, and a
    /// change of D or of a waiting interrupt's mask stays with the engine.
    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    #[test]
    fn mask_writes_complete_in_the_frame() {
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
        const PC: GuestAddress = GuestAddress(0x4000);
        let open = Cell {
            readable: 1,
            value: 0,
        };
        // The PC of the exit `write` takes.
        fn run(write: MaskWrite, cell: &Cell, state: &mut GuestState) -> u64 {
            let thunk = mask_thunk(write, PC, CELL);
            state.anchors[Anchor::Mirrors] = cell as *const Cell as u64;
            // SAFETY: private anonymous memory, RX after the copy, unmapped
            // after the call; the code reads and writes only `state` and
            // `cell`.
            unsafe {
                let page = mmap(core::ptr::null_mut(), PAGE, 3, 2 | MAP_ANONYMOUS, -1, 0);
                assert_ne!(page as usize, usize::MAX);
                core::ptr::copy_nonoverlapping(thunk.code.as_ptr(), page.cast(), thunk.code.len());
                assert_eq!(mprotect(page, PAGE, 5), 0);
                let entry: unsafe extern "sysv64" fn(*mut GuestState) -> u128 =
                    core::mem::transmute(page);
                let exit = entry(state);
                assert_eq!(munmap(page, PAGE), 0);
                exit as u64
            }
        }
        let (next, here) = (PC.0 + 4, PC.0);
        let mut state = GuestState {
            daif: 0b1000,
            ..GuestState::default()
        };
        assert_eq!(run(MaskWrite::Set(7), &open, &mut state), next);
        assert_eq!(state.daif, 0b1111);
        assert_eq!(run(MaskWrite::Clear(2), &open, &mut state), next);
        assert_eq!(state.daif, 0b1101);
        state.x[9] = 0b0011 << 6 | 0b1000 << 6 | 1 << 40;
        assert_eq!(run(MaskWrite::Register(9), &open, &mut state), next);
        assert_eq!(state.daif, 0b1011);
        // An interrupt waits on I.
        state.pending = 0b0010;
        assert_eq!(run(MaskWrite::Clear(2), &open, &mut state), here);
        assert_eq!(state.daif, 0b1011);
        assert_eq!(run(MaskWrite::Clear(1), &open, &mut state), next);
        assert_eq!(state.daif, 0b1010);
        // Masking more is always fine, and D stays with the engine.
        assert_eq!(run(MaskWrite::Set(1), &open, &mut state), next);
        assert_eq!(run(MaskWrite::Clear(8), &open, &mut state), here);
        state.daif = 0b0011;
        assert_eq!(run(MaskWrite::Set(8), &open, &mut state), here);
        assert_eq!(state.daif, 0b0011);
        // Closed, every write is the engine's.
        let closed = Cell::default();
        assert_eq!(run(MaskWrite::Set(1), &closed, &mut state), here);
    }

    #[test]
    fn thunks_end_in_the_fallthrough_then_the_original_exit() {
        let pc = GuestAddress(0x4000);
        let thunk = thunk(pc, 8, CELL);
        let end = thunk.code.len();
        assert_eq!(
            thunk.code[end - DIRECT_EXIT_BYTES..],
            direct_exit(pc, ExitKind::System)
        );
        let next = thunk.next.unwrap();
        assert_eq!(next + DIRECT_EXIT_BYTES, end - DIRECT_EXIT_BYTES);
        // Padded for its offset, where a link patches it.
        assert_eq!(
            thunk.code[next..next + DIRECT_EXIT_BYTES],
            crate::x86_64::chain::direct_exit_at(
                next as u64,
                GuestAddress(0x4004),
                ExitKind::Fallthrough
            )
        );
    }

    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    #[repr(C)]
    #[derive(Debug, PartialEq, Eq)]
    struct Exit {
        pc: u64,
        kind: u64,
    }

    /// Run a thunk on the host.
    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    fn run(code: &[u8], state: &mut crate::GuestState) -> Exit {
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
        // SAFETY: private anonymous memory, RX after the copy, unmapped after
        // the call; the thunk touches only `state` and its cells, which
        // outlive it.
        unsafe {
            let page = mmap(core::ptr::null_mut(), PAGE, 3, 2 | MAP_ANONYMOUS, -1, 0);
            assert_ne!(page as usize, usize::MAX);
            core::ptr::copy_nonoverlapping(code.as_ptr(), page.cast(), code.len());
            assert_eq!(mprotect(page, PAGE, 5), 0);
            let entry: unsafe extern "sysv64" fn(*mut crate::GuestState) -> Exit =
                core::mem::transmute(page);
            let exit = entry(state);
            assert_eq!(munmap(page, PAGE), 0);
            exit
        }
    }

    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    const PC: GuestAddress = GuestAddress(0x4000);
    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    const MISS: Exit = Exit {
        pc: PC.0,
        kind: ExitKind::System as u64,
    };
    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    const NEXT: Exit = Exit {
        pc: PC.0 + 4,
        kind: ExitKind::Fallthrough as u64,
    };

    /// A thread pointer write runs on the host: with its cell open it writes
    /// the storage and every open read cell, logs a switch to another value
    /// with the frame and link registers, changes nothing for the same
    /// value, and leaves everything to the engine with the cell closed or
    /// the log full.
    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    #[test]
    fn thread_switches_write_the_register_its_reads_and_the_log() {
        use crate::GuestState;
        use core::mem::offset_of;
        #[repr(C)]
        #[derive(Default)]
        struct Data {
            write: Cell,
            storage: u64,
            reads: [Cell; 3],
            log: SwitchLog,
        }
        let (pc, miss, next) = (PC, MISS, NEXT);
        let mut data = alloc::boxed::Box::new(Data::default());
        let base = &mut *data as *mut Data;
        let place = |offset: usize| Place::new(Anchor::Mirrors, offset);
        let places = ThreadSwitch {
            reads: place(offset_of!(Data, reads)),
            stride: size_of::<Cell>(),
            banks: 3,
            log: place(offset_of!(Data, log)),
        };
        let cell = place(offset_of!(Data, write));
        let mut state = GuestState::default();
        state.anchors[Anchor::Mirrors] = base as u64;
        state.x[29] = 0xffff_fe00_0000_2900;
        state.x[30] = 0xffff_fe00_0000_3000;
        let storage = base as u64 + offset_of!(Data, storage) as u64;
        // SAFETY: `base` points into the live box; reads and writes happen
        // between thunk calls.
        let data = |f: &mut dyn FnMut(&mut Data)| unsafe { f(&mut *base) };
        let open = Cell {
            readable: 1,
            value: 0x1000,
        };
        data(&mut |d| {
            d.write = Cell {
                readable: 0,
                value: storage,
            };
            d.storage = 0x1000;
            d.reads = [open, Cell::default(), open];
        });
        state.x[5] = 0x2000;
        let write = thread_switch_thunk(pc, 5, cell, &places);
        assert_eq!(run(&write.code, &mut state), miss);
        data(&mut |d| assert_eq!((d.storage, d.log.len), (0x1000, 0)));
        data(&mut |d| d.write.readable = 1);
        assert_eq!(run(&write.code, &mut state), next);
        data(&mut |d| {
            assert_eq!(d.storage, 0x2000);
            let new = Cell {
                readable: 1,
                value: 0x2000,
            };
            assert_eq!(d.reads, [new, Cell::default(), new]);
            assert_eq!(d.log.len, 1);
            let switch = Switch {
                previous: 0x1000,
                next: 0x2000,
                frame: 0xffff_fe00_0000_2900,
                link: 0xffff_fe00_0000_3000,
                pc: pc.0,
            };
            assert_eq!(d.log.entries[0], switch);
        });
        // The same value again: nothing logged.
        assert_eq!(run(&write.code, &mut state), next);
        data(&mut |d| assert_eq!(d.log.len, 1));
        // XZR, then values until the log is full: the next one exits.
        assert_eq!(
            run(&thread_switch_thunk(pc, 31, cell, &places).code, &mut state),
            next
        );
        data(&mut |d| assert_eq!((d.storage, d.log.entries[1].next), (0, 0)));
        for value in 2..SWITCHES as u64 {
            state.x[5] = value;
            assert_eq!(run(&write.code, &mut state), next);
        }
        data(&mut |d| assert_eq!((d.log.len, d.storage), (SWITCHES as u64, 7)));
        state.x[5] = 0x9000;
        assert_eq!(run(&write.code, &mut state), miss);
        data(&mut |d| {
            assert_eq!(d.storage, 7);
            assert_eq!(d.reads[0].value, 7);
        });
    }

    /// Thunks run on the host: a readable cell completes the read into Xt,
    /// or into nothing for XZR; a cleared one exits with no write.
    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    #[test]
    fn thunks_read_the_cell_only_while_it_is_readable() {
        use crate::GuestState;
        let (pc, miss, next) = (PC, MISS, NEXT);
        let mut cell = alloc::boxed::Box::new(Cell::default());
        let address = &mut *cell as *mut Cell;
        // SAFETY: address points into the live Cell box; the closure runs between thunk calls.
        let set = |value: Cell| unsafe { address.write(value) };
        let mut state = GuestState::default();
        state.x[19] = 7;
        state.anchors[Anchor::Mirrors] = address as u64;
        let read = thunk(pc, 19, CELL);
        set(Cell {
            readable: 0,
            value: 0xffff_fe00_1234_5000,
        });
        assert_eq!(run(&read.code, &mut state), miss);
        assert_eq!(state.x[19], 7);
        set(Cell {
            readable: 1,
            value: 0xffff_fe00_1234_5000,
        });
        assert_eq!(run(&read.code, &mut state), next);
        assert_eq!(state.x[19], 0xffff_fe00_1234_5000);
        let before = state.x;
        assert_eq!(run(&thunk(pc, 31, CELL).code, &mut state), next);
        assert_eq!(state.x, before);
    }

    /// IC IVAU of the cell's page completes; of another, or with the cell
    /// closed, it exits.
    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    #[test]
    fn the_code_page_thunk_completes_only_in_the_open_cells_page() {
        use crate::GuestState;
        let open = Cell {
            readable: 1,
            value: 0xffff_fe00_1234_5040,
        };
        let closed = Cell::default();
        let mut state = GuestState::default();
        for (cell, xt, exit) in [
            (&open, 0xffff_fe00_1234_5fc0, NEXT),
            (&open, 0xffff_fe00_1234_6000, MISS),
            (&closed, 0xffff_fe00_1234_5fc0, MISS),
        ] {
            state.x[19] = xt;
            state.anchors[Anchor::Mirrors] = cell as *const Cell as u64;
            let thunk = code_page_thunk(PC, 19, CELL);
            assert_eq!(run(&thunk.code, &mut state), exit);
            assert_eq!(state.x[19], xt);
        }
    }

    #[test]
    fn write_thunks_store_the_kept_bits_only_while_the_cell_is_open() {
        use crate::GuestState;
        let written = Written {
            keep: 0xff00,
            set: 1,
        };
        let mut storage = alloc::boxed::Box::new(0u64);
        let target = &mut *storage as *mut u64;
        let mut cell = alloc::boxed::Box::new(Cell::default());
        let address = &mut *cell as *mut Cell;
        let mut state = GuestState::default();
        state.x[3] = 0xabcd;
        state.anchors[Anchor::Mirrors] = address as u64;
        let code = stored_write_thunk(PC, 3, CELL, written).code;
        assert_eq!(run(&code, &mut state), MISS);
        // SAFETY: the live Cell box and the live storage, between runs.
        unsafe {
            address.write(Cell {
                readable: 1,
                value: target as u64,
            });
        }
        assert_eq!(run(&code, &mut state), NEXT);
        // SAFETY: as above.
        assert_eq!(unsafe { target.read() }, 0xab01);
        let zero = stored_write_thunk(PC, 31, CELL, written).code;
        assert_eq!(run(&zero, &mut state), NEXT);
        // SAFETY: as above.
        assert_eq!(unsafe { target.read() }, 1);
        let offset = core::mem::offset_of!(GuestState, fpsr);
        let frame = frame_write_thunk(PC, 3, CELL, offset, written).code;
        assert_eq!(run(&frame, &mut state), NEXT);
        assert_eq!(state.fpsr, 0xab01);
    }

    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    #[test]
    fn key_writes_close_the_tag_words_only_when_they_change_the_key() {
        use crate::GuestState;
        let mut key = alloc::boxed::Box::new(5u64);
        let target = &mut *key as *mut u64;
        let mut cell = alloc::boxed::Box::new(Cell {
            readable: 1,
            value: target as u64,
        });
        let mut tags = alloc::boxed::Box::new([7u64; 3]);
        let mut state = GuestState::default();
        state.anchors[Anchor::Mirrors] = &*cell as *const Cell as u64;
        state.anchors[Anchor::Pointer] = tags.as_mut_ptr() as u64;
        let place = Place::new(Anchor::Pointer, 0);
        let code = key_write_thunk(PC, 3, CELL, place, 2).code;
        state.x[3] = 5;
        assert_eq!(run(&code, &mut state), NEXT);
        assert_eq!(*tags, [7; 3]);
        state.x[3] = 6;
        assert_eq!(run(&code, &mut state), NEXT);
        // SAFETY: the live key, between runs.
        assert_eq!(unsafe { target.read() }, 6);
        assert_eq!(*tags, [0, 0, 7]);
        tags[0] = 7;
        cell.readable = 0;
        state.x[3] = 8;
        assert_eq!(run(&code, &mut state), MISS);
        // SAFETY: as above.
        assert_eq!(unsafe { target.read() }, 6);
        assert_eq!(tags[0], 7);
    }

    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    #[test]
    fn checked_writes_leave_the_refused_bits_to_the_exit() {
        use crate::GuestState;
        let mut storage = alloc::boxed::Box::new(5u64);
        let target = &mut *storage as *mut u64;
        let cell = alloc::boxed::Box::new(Cell {
            readable: 1,
            value: target as u64,
        });
        let mut state = GuestState::default();
        state.anchors[Anchor::Mirrors] = &*cell as *const Cell as u64;
        let code = checked_write_thunk(PC, 3, CELL, 0xf0).code;
        state.x[3] = 0x10f;
        assert_eq!(run(&code, &mut state), NEXT);
        // SAFETY: the live storage, between runs.
        assert_eq!(unsafe { target.read() }, 0x10f);
        state.x[3] = 0x10;
        assert_eq!(run(&code, &mut state), MISS);
        // SAFETY: as above.
        assert_eq!(unsafe { target.read() }, 0x10f);
        let zero = checked_write_thunk(PC, 31, CELL, u64::MAX).code;
        assert_eq!(run(&zero, &mut state), NEXT);
        // SAFETY: as above.
        assert_eq!(unsafe { target.read() }, 0);
    }

    #[test]
    fn stack_select_moves_the_frame_sp_between_the_stacks() {
        use crate::GuestState;
        let mut cell = alloc::boxed::Box::new(Cell::default());
        let address = &mut *cell as *mut Cell as u64;
        let mut stacks = alloc::boxed::Box::new([0x1000u64, 0x2000]);
        let (sp0, spx) = (stacks.as_mut_ptr() as u64, stacks.as_mut_ptr() as u64 + 8);
        let mut current = alloc::boxed::Box::new(1u64);
        let word = &mut *current as *mut u64 as u64;
        let mut state = GuestState {
            sp: 0x3000,
            ..GuestState::default()
        };
        state.anchors[Anchor::Mirrors] = address;
        state.anchors[Anchor::Pointer] = word;
        state.anchors[Anchor::StackZero] = sp0;
        let to_sp0 = stack_select_thunk(PC, false, CELL, WORD, SP0).code;
        let to_spx = stack_select_thunk(PC, true, CELL, WORD, SP0).code;
        assert_eq!(run(&to_sp0, &mut state), MISS);
        cell.readable = 1;
        cell.value = spx;
        // Cross the box with a raw read: the thunk wrote through addresses.
        let read = |at: u64| {
            // SAFETY: a live word of `stacks` or `current`.
            unsafe { (at as *const u64).read_volatile() }
        };
        assert_eq!(run(&to_sp0, &mut state), NEXT);
        assert_eq!((state.sp, read(spx), read(word)), (0x1000, 0x3000, 0));
        // Selecting what is selected moves nothing.
        assert_eq!(run(&to_sp0, &mut state), NEXT);
        assert_eq!((state.sp, read(spx)), (0x1000, 0x3000));
        state.sp = 0x1800;
        assert_eq!(run(&to_spx, &mut state), NEXT);
        assert_eq!((state.sp, read(sp0), read(word)), (0x3000, 0x1800, 1));
    }

    /// A count read adds the state's offset to the scaled TSC (zero here,
    /// as `per_tsc` is), stays at or below its ceiling, never reads below
    /// `last` and keeps what it read there, and subtracts the cell's value;
    /// XZR still moves `last`.
    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    #[test]
    fn count_thunks_read_a_monotonic_bounded_count_less_the_cell_value() {
        use crate::GuestState;
        use core::sync::atomic::Ordering::Relaxed;
        let mut cell = alloc::boxed::Box::new(Cell::default());
        let published = alloc::boxed::Box::new(CountState::default());
        published.offset.store(100, Relaxed);
        published.ceiling.store(u64::MAX, Relaxed);
        let mut count = alloc::boxed::Box::new(Count {
            last: 50,
            state: &*published as *const CountState as u64,
            ..Count::default()
        });
        let (cell_at, count_at) = (&mut *cell as *mut Cell, &mut *count as *mut Count);
        let read = count_thunk(PC, 3, CELL, READER);
        let mut state = GuestState::default();
        state.anchors[Anchor::Mirrors] = cell_at as u64;
        state.anchors[Anchor::Count] = count_at as u64;
        assert_eq!(run(&read.code, &mut state), MISS);
        // SAFETY: the boxes outlive every thunk run.
        unsafe {
            cell_at.write(Cell {
                readable: 1,
                value: 30,
            })
        };
        let last = || {
            // SAFETY: the Count box is live and every thunk run has returned.
            unsafe { (*count_at).last }
        };
        assert_eq!(run(&read.code, &mut state), NEXT);
        assert_eq!((state.x[3], last()), (70, 100));
        published.offset.store(80, Relaxed);
        assert_eq!(run(&read.code, &mut state), NEXT);
        assert_eq!((state.x[3], last()), (70, 100));
        published.offset.store(500, Relaxed);
        published.ceiling.store(300, Relaxed);
        assert_eq!(run(&read.code, &mut state), NEXT);
        assert_eq!((state.x[3], last()), (270, 300));
        published.ceiling.store(u64::MAX, Relaxed);
        let before = state.x;
        let discard = count_thunk(PC, 31, CELL, READER);
        assert_eq!(run(&discard.code, &mut state), NEXT);
        assert_eq!((state.x, last()), (before, 500));
    }

    /// SEVL sets the event register; WFE clears a set one and completes, and
    /// leaves a clear one, or a closed cell, to the engine.
    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    #[test]
    fn event_thunks_set_and_consume_the_event_register() {
        use crate::GuestState;
        let mut event = alloc::boxed::Box::new(0u8);
        let mut cell = alloc::boxed::Box::new(Cell::default());
        let (send, wait) = (send_event_thunk(PC, CELL), event_wait_thunk(PC, CELL));
        let mut state = GuestState::default();
        state.anchors[Anchor::Mirrors] = &mut *cell as *mut Cell as u64;
        assert_eq!(run(&send.code, &mut state), MISS);
        *cell = Cell {
            readable: 1,
            value: &mut *event as *mut u8 as u64,
        };
        assert_eq!(run(&wait.code, &mut state), MISS);
        assert_eq!(run(&send.code, &mut state), NEXT);
        assert_eq!(*event, 1);
        assert_eq!(run(&wait.code, &mut state), NEXT);
        assert_eq!(*event, 0);
        assert_eq!(run(&wait.code, &mut state), MISS);
    }

    /// An energy read scales the ticks of the count since the origin, none
    /// before it, and leaves a reading too wide for 64 bits to the engine.
    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    #[test]
    fn energy_thunks_scale_the_ticks_since_the_origin() {
        use crate::GuestState;
        use core::sync::atomic::Ordering::Relaxed;
        let mut energy = alloc::boxed::Box::new(Energy {
            origin: 40,
            scale: 3_000,
            frequency: 24,
        });
        let mut cell = alloc::boxed::Box::new(Cell {
            readable: 1,
            value: &mut *energy as *mut Energy as u64,
        });
        let published = alloc::boxed::Box::new(CountState::default());
        published.offset.store(100, Relaxed);
        published.ceiling.store(u64::MAX, Relaxed);
        let mut count = alloc::boxed::Box::new(Count {
            state: &*published as *const CountState as u64,
            ..Count::default()
        });
        let read = energy_thunk(PC, 3, CELL, READER);
        let mut state = GuestState::default();
        state.anchors[Anchor::Mirrors] = &mut *cell as *mut Cell as u64;
        state.anchors[Anchor::Count] = &mut *count as *mut Count as u64;
        assert_eq!(run(&read.code, &mut state), NEXT);
        assert_eq!(state.x[3], 60 * 3_000 / 24);
        energy.origin = 200;
        assert_eq!(run(&read.code, &mut state), NEXT);
        assert_eq!(state.x[3], 0);
        energy.origin = 0;
        energy.scale = u64::MAX;
        energy.frequency = 50;
        assert_eq!(run(&read.code, &mut state), MISS);
    }

    /// A cycle counter read takes an open cell's value; with the cell
    /// closed but the read plain it computes the count as
    /// [`CycleCounter::at`] does, carrying the remainder and wrapping to the
    /// mask, and opens the cell on it; anything else, and a quotient too
    /// wide, exits with Xt untouched.
    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    #[test]
    fn cycle_counter_thunks_compute_the_count_where_only_the_value_went_stale() {
        use crate::GuestState;
        use core::sync::atomic::Ordering::Relaxed;
        #[repr(C)]
        struct Cells {
            cell: Cell,
            plain: Cell,
            cycles: CycleCounter,
        }
        const PLAIN: Place = Place::new(Anchor::Mirrors, size_of::<Cell>());
        const CYCLES: Place = Place::new(Anchor::Mirrors, 2 * size_of::<Cell>());
        let mut cells = alloc::boxed::Box::new(Cells {
            cell: Cell::default(),
            plain: Cell::default(),
            cycles: CycleCounter {
                open: 1,
                counter: 1_000,
                synced: 90,
                remainder: 2,
                clock: 7,
                frequency: 3,
                mask: u64::MAX >> 16,
            },
        });
        let published = alloc::boxed::Box::new(CountState::default());
        published.ceiling.store(u64::MAX, Relaxed);
        let mut count = alloc::boxed::Box::new(Count {
            state: &*published as *const CountState as u64,
            ..Count::default()
        });
        let read = cycle_counter_thunk(PC, 3, CELL, PLAIN, CYCLES, READER);
        let mut state = GuestState::default();
        state.anchors[Anchor::Mirrors] = &mut *cells as *mut Cells as u64;
        state.anchors[Anchor::Count] = &mut *count as *mut Count as u64;
        // The reader never reads below its last count: each read starts
        // afresh at `now`.
        let reader = &mut *count as *mut Count;
        let run_at = |_: &mut Cells, now: u64, state: &mut GuestState| {
            published.offset.store(now, Relaxed);
            // SAFETY: the test's reader, which no thunk runs over meanwhile.
            unsafe { (*reader).last = 0 };
            run(&read.code, state)
        };
        // Closed and not plain: the engine.
        state.x[3] = 0xdead;
        assert_eq!(run_at(&mut cells, 100, &mut state), MISS);
        assert_eq!(state.x[3], 0xdead);
        // Plain: (10 * 7 + 2) / 3 = 24 cycles past 1000, the cell opens.
        cells.plain.readable = 1;
        assert_eq!(run_at(&mut cells, 100, &mut state), NEXT);
        assert_eq!(state.x[3], 1_024);
        assert_eq!(cells.cycles.at(100), Some(1_024));
        assert_eq!(
            cells.cell,
            Cell {
                readable: 1,
                value: 1_024
            }
        );
        // Open: its value until the cell closes again.
        assert_eq!(run_at(&mut cells, 400, &mut state), NEXT);
        assert_eq!(state.x[3], 1_024);
        cells.cell.readable = 0;
        for now in [0, 89, 90, 91, 92, 400, 1 << 40] {
            assert_eq!(run_at(&mut cells, now, &mut state), NEXT, "{now}");
            assert_eq!(Some(state.x[3]), cells.cycles.at(now), "{now}");
            cells.cell.readable = 0;
        }
        // Wrapped to the counter's width.
        cells.cycles.counter = u64::MAX >> 16;
        assert_eq!(run_at(&mut cells, 100, &mut state), NEXT);
        assert_eq!(state.x[3], 23);
        cells.cell.readable = 0;
        // Not counting: the count as it stands.
        cells.cycles.clock = 0;
        assert_eq!(run_at(&mut cells, 1 << 40, &mut state), NEXT);
        assert_eq!(state.x[3], u64::MAX >> 16);
        cells.cell.readable = 0;
        // Too wide for one division, and the monitor's view closed.
        cells.cycles.clock = u64::MAX;
        state.x[3] = 0xdead;
        assert_eq!(run_at(&mut cells, u64::MAX, &mut state), MISS);
        assert_eq!(cells.cycles.at(u64::MAX), None);
        cells.cycles.clock = 7;
        cells.cycles.open = 0;
        assert_eq!(run_at(&mut cells, 100, &mut state), MISS);
        assert_eq!((state.x[3], cells.cell.readable), (0xdead, 0));
        // XZR discards it but the cell still opens.
        cells.cycles.open = 1;
        cells.cycles.counter = 1_000;
        let discard = cycle_counter_thunk(PC, 31, CELL, PLAIN, CYCLES, READER);
        published.offset.store(100, Relaxed);
        // SAFETY: as above.
        unsafe { (*reader).last = 0 };
        assert_eq!(run(&discard.code, &mut state), NEXT);
        assert_eq!((state.x[3], cells.cell.value), (0xdead, 1_024));
    }

    /// An instruction counter read takes an open cell's value; with the
    /// cell closed but the read plain it adds the chain's retired count and
    /// the instructions before it in its block to the base, as
    /// [`InstructionCounter::at`] does, wrapping to the mask, and opens the
    /// cell on it; anything else exits with Xt untouched.
    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    #[test]
    fn instruction_counter_thunks_add_what_native_code_retired() {
        use crate::GuestState;
        use crate::x86_64::chain::ChainControl;
        #[repr(C)]
        struct Cells {
            cell: Cell,
            plain: Cell,
            counter: InstructionCounter,
        }
        const PLAIN: Place = Place::new(Anchor::Mirrors, size_of::<Cell>());
        const COUNTER: Place = Place::new(Anchor::Mirrors, 2 * size_of::<Cell>());
        let mut cells = alloc::boxed::Box::new(Cells {
            cell: Cell::default(),
            plain: Cell::default(),
            counter: InstructionCounter {
                open: 1,
                base: 1_000,
                step: u64::MAX,
                mask: u64::MAX >> 16,
            },
        });
        let mut control = alloc::boxed::Box::new(ChainControl {
            retired: 40,
            ..ChainControl::default()
        });
        let read = instruction_counter_thunk(PC, 3, CELL, PLAIN, COUNTER, 5).unwrap();
        let mut state = GuestState::default();
        state.anchors[Anchor::Mirrors] = &mut *cells as *mut Cells as u64;
        state.anchors[Anchor::Control] = &mut *control as *mut ChainControl as u64;
        // Closed and not plain: the engine.
        state.x[3] = 0xdead;
        assert_eq!(run(&read.code, &mut state), MISS);
        assert_eq!(state.x[3], 0xdead);
        // Plain: 40 chained and 5 before it past 1000, the cell opens.
        cells.plain.readable = 1;
        assert_eq!(run(&read.code, &mut state), NEXT);
        assert_eq!(state.x[3], 1_045);
        assert_eq!(cells.counter.at(45), Some(1_045));
        assert_eq!(
            cells.cell,
            Cell {
                readable: 1,
                value: 1_045
            }
        );
        // Open: its value until the cell closes again.
        control.retired = 400;
        assert_eq!(run(&read.code, &mut state), NEXT);
        assert_eq!(state.x[3], 1_045);
        cells.cell.readable = 0;
        for retired in [0, 1, 400, 1 << 40] {
            control.retired = retired;
            assert_eq!(run(&read.code, &mut state), NEXT, "{retired}");
            assert_eq!(Some(state.x[3]), cells.counter.at(retired + 5), "{retired}");
            cells.cell.readable = 0;
        }
        // Wrapped to the counter's width.
        control.retired = 0;
        cells.counter.base = u64::MAX >> 16;
        assert_eq!(run(&read.code, &mut state), NEXT);
        assert_eq!(state.x[3], 4);
        cells.cell.readable = 0;
        // Not counting: the count as it stands.
        cells.counter.step = 0;
        control.retired = 1 << 40;
        assert_eq!(run(&read.code, &mut state), NEXT);
        assert_eq!(state.x[3], u64::MAX >> 16);
        cells.cell.readable = 0;
        // The monitor's view closed.
        cells.counter.open = 0;
        state.x[3] = 0xdead;
        assert_eq!(run(&read.code, &mut state), MISS);
        assert_eq!(cells.counter.at(0), None);
        assert_eq!((state.x[3], cells.cell.readable), (0xdead, 0));
        // XZR discards it but the cell still opens; the first instruction
        // of its block adds no immediate.
        cells.counter = InstructionCounter {
            open: 1,
            base: 1_000,
            step: u64::MAX,
            mask: u64::MAX,
        };
        control.retired = 7;
        let discard = instruction_counter_thunk(PC, 31, CELL, PLAIN, COUNTER, 0).unwrap();
        assert!(discard.code.len() < read.code.len());
        assert_eq!(run(&discard.code, &mut state), NEXT);
        assert_eq!((state.x[3], cells.cell.value), (0xdead, 1_007));
    }

    /// ISR_EL1 reads the waiting I and F masks as bits 7 and 6 while the
    /// cell is open, the epoch is the cell's and the count is below its
    /// deadline, and leaves every other case, Xt untouched, to the engine.
    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    #[test]
    fn interrupt_status_thunks_read_the_waiting_lines_while_nothing_changed() {
        use crate::GuestState;
        use core::sync::atomic::Ordering::Relaxed;
        const EPOCH: Place = Place::new(Anchor::Control, 0);
        let mut cell = alloc::boxed::Box::new(Cell::default());
        let mut epoch = alloc::boxed::Box::new(7u64);
        let published = alloc::boxed::Box::new(CountState::default());
        published.offset.store(100, Relaxed);
        published.ceiling.store(u64::MAX, Relaxed);
        let mut count = alloc::boxed::Box::new(Count {
            state: &*published as *const CountState as u64,
            ..Count::default()
        });
        let mut state = GuestState::default();
        state.anchors[Anchor::Mirrors] = &mut *cell as *mut Cell as u64;
        state.anchors[Anchor::Control] = &mut *epoch as *mut u64 as u64;
        state.anchors[Anchor::Count] = &mut *count as *mut Count as u64;
        let read = interrupt_status_thunk(PC, 5, CELL, EPOCH, READER).code;
        let mut expect = |cell_now: Cell, pending: u32, exit: Exit, x5: u64| {
            *cell = cell_now;
            state.pending = pending;
            state.x[5] = 0xdead;
            assert_eq!(run(&read, &mut state), exit, "{cell_now:?} {pending:#x}");
            assert_eq!(state.x[5], x5, "{cell_now:?} {pending:#x}");
        };
        let open = |deadline: u64, value: u64| Cell {
            readable: deadline,
            value,
        };
        // No deadline: no count is read, every mask combination reads.
        for (pending, isr) in [(0, 0), (0b01, 0x40), (0b10, 0x80), (0b11, 0xc0)] {
            expect(open(u64::MAX, 7), pending, NEXT, isr);
        }
        // The count (100: `per_tsc` is 0) is below 101, not below 100.
        expect(open(101, 7), 0b10, NEXT, 0x80);
        expect(open(100, 7), 0b10, MISS, 0xdead);
        // Closed, or a line changed since: the engine reads.
        expect(Cell::default(), 0b01, MISS, 0xdead);
        expect(open(u64::MAX, 6), 0b01, MISS, 0xdead);
        *epoch = 6;
        expect(open(u64::MAX, 6), 0b01, NEXT, 0x40);
        // XZR reads nothing.
        let discard = interrupt_status_thunk(PC, 31, CELL, EPOCH, READER).code;
        let before = state.x;
        assert_eq!(run(&discard, &mut state), NEXT);
        assert_eq!(state.x, before);
    }

    /// Timer thunks run on the host against a count the test moves: writes
    /// that keep the output low and arm no compare before `lines_until`
    /// complete into storage, every other write exits with storage as it
    /// was, and CTL reads give ENABLE and IMASK as stored with ISTATUS from
    /// the count.
    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    #[test]
    fn timer_thunks_complete_only_writes_that_raise_nothing_before_the_engine_looks() {
        use crate::GuestState;
        use core::mem::offset_of;
        use core::sync::atomic::Ordering::Relaxed;
        #[repr(C)]
        #[derive(Default)]
        struct Timer {
            compare: Cell,
            control: Cell,
            cval: u64,
            ctl: u64,
            lines_until: u64,
        }
        const ENABLE: u64 = 1;
        const IMASK: u64 = 2;
        const ISTATUS: u64 = 4;
        let place = |offset: usize| Place::new(Anchor::Mirrors, offset);
        let places = TimerPlaces {
            compare: place(offset_of!(Timer, compare)),
            control: place(offset_of!(Timer, control)),
            lines_until: place(offset_of!(Timer, lines_until)),
            count: READER,
        };
        let mut timer = alloc::boxed::Box::new(Timer::default());
        let base = &mut *timer as *mut Timer;
        let published = alloc::boxed::Box::new(CountState::default());
        published.ceiling.store(u64::MAX, Relaxed);
        let mut count = alloc::boxed::Box::new(Count {
            state: &*published as *const CountState as u64,
            ..Count::default()
        });
        let mut state = GuestState::default();
        state.anchors[Anchor::Mirrors] = base as u64;
        state.anchors[Anchor::Count] = &mut *count as *mut Count as u64;
        let (cval_at, ctl_at) = (
            base as u64 + offset_of!(Timer, cval) as u64,
            base as u64 + offset_of!(Timer, ctl) as u64,
        );
        // SAFETY: `base` points into the live box; every access happens
        // between thunk runs.
        let data = |f: &mut dyn FnMut(&mut Timer)| unsafe { f(&mut *base) };
        let open = |d: &mut Timer| {
            d.compare = Cell {
                readable: 1,
                value: cval_at,
            };
            d.control = Cell {
                readable: 1,
                value: ctl_at,
            };
        };
        data(&mut |d| open(d));
        let keep = Written {
            keep: ENABLE | IMASK | ISTATUS,
            set: 0,
        };
        let (value, compare) = (
            timer_thunk(PC, 5, TimerAccess::WriteValue, places).code,
            timer_thunk(PC, 5, TimerAccess::WriteCompare, places).code,
        );
        let control = timer_thunk(PC, 5, TimerAccess::WriteControl(keep), places).code;
        // count, lines_until, (cval, ctl) before, the thunk, Xt, the exit
        // and (cval, ctl) after.
        let mut expect = |now: u64,
                          until: u64,
                          before: (u64, u64),
                          code: &[u8],
                          xt: u64,
                          exit: Exit,
                          after: (u64, u64)| {
            published.offset.store(now, Relaxed);
            data(&mut |d| {
                (d.cval, d.ctl) = before;
                d.lines_until = until;
            });
            state.x[5] = xt;
            let ran = run(code, &mut state);
            data(&mut |d| {
                assert_eq!(
                    (ran.pc, d.cval, d.ctl),
                    (exit.pc, after.0, after.1),
                    "now {now} until {until} before {before:?} xt {xt:#x}"
                );
            });
            assert_eq!(state.x[5], xt, "a write leaves Xt");
        };
        let armed = (200, ENABLE);
        // TVAL: count plus the sign-extended low word. 180 is past 150.
        expect(100, 150, armed, &value, 80, NEXT, (180, ENABLE));
        expect(
            100,
            150,
            armed,
            &value,
            0xdead_0000_0000_0050,
            NEXT,
            (180, ENABLE),
        );
        // Earlier than the engine looks again: exit.
        expect(100, 150, armed, &value, 30, MISS, armed);
        // Due at once, or already past: exit.
        expect(100, 150, armed, &value, 0, MISS, armed);
        expect(100, 150, armed, &value, 0xffff_fff6, MISS, armed);
        // CVAL: at lines_until itself completes.
        expect(100, 150, armed, &compare, 150, NEXT, (150, ENABLE));
        expect(100, 150, armed, &compare, 149, MISS, armed);
        // The output is high (armed and past its compare): every write exits.
        expect(100, 150, (90, ENABLE), &compare, 500, MISS, (90, ENABLE));
        expect(100, 150, (90, ENABLE), &control, 0, MISS, (90, ENABLE));
        // Masked or disabled, any compare completes and arms nothing.
        expect(
            100,
            150,
            (90, ENABLE | IMASK),
            &compare,
            120,
            NEXT,
            (120, ENABLE | IMASK),
        );
        expect(100, 150, (90, 0), &value, 5, NEXT, (105, 0));
        // CTL: disabling or masking completes; ISTATUS and RES0 bits as the
        // register takes them.
        expect(
            100,
            150,
            armed,
            &control,
            !ENABLE,
            NEXT,
            (200, IMASK | ISTATUS),
        );
        expect(100, 150, armed, &control, 0, NEXT, (200, 0));
        // Unmasking an expired compare raises the line: exit.
        expect(
            100,
            150,
            (90, ENABLE | IMASK),
            &control,
            ENABLE,
            MISS,
            (90, ENABLE | IMASK),
        );
        // Enabling arms the stored compare: before lines_until exits.
        expect(100, 150, (120, 0), &control, ENABLE, MISS, (120, 0));
        expect(100, 150, (160, 0), &control, ENABLE, NEXT, (160, ENABLE));
        // No count named (0): every write exits; none armed (MAX): only a
        // write that arms a compare exits.
        expect(100, 0, armed, &control, 0, MISS, armed);
        expect(100, u64::MAX, armed, &control, 0, NEXT, (200, 0));
        expect(100, u64::MAX, armed, &value, 1 << 20, MISS, armed);
        expect(
            100,
            u64::MAX,
            (200, 0),
            &value,
            1 << 20,
            NEXT,
            (100 + (1 << 20), 0),
        );
        // Closed cells: the engine's.
        for closed in [offset_of!(Timer, compare), offset_of!(Timer, control)] {
            data(&mut |d| {
                open(d);
                // SAFETY: `closed` is the offset of a Cell field of `d`.
                unsafe { (*((d as *mut Timer as *mut u8).add(closed) as *mut Cell)).readable = 0 };
            });
            expect(100, 150, armed, &control, 0, MISS, armed);
        }
        data(&mut |d| open(d));
        // The count passes the new compare right after the thunk read it:
        // the native ISR read the engine left open until lines_until (150)
        // exits there, before the compare (180) the line now rises at.
        expect(100, 150, armed, &value, 80, NEXT, (180, ENABLE));
        published.offset.store(180, Relaxed);
        let mut status = Cell {
            readable: 150,
            value: 0,
        };
        let mut epoch = 0u64;
        let mut isr_state = GuestState::default();
        isr_state.anchors[Anchor::Mirrors] = &mut status as *mut Cell as u64;
        isr_state.anchors[Anchor::Control] = &mut epoch as *mut u64 as u64;
        isr_state.anchors[Anchor::Count] = state.anchors[Anchor::Count];
        let isr = interrupt_status_thunk(PC, 5, CELL, Place::new(Anchor::Control, 0), READER);
        assert_eq!(run(&isr.code, &mut isr_state), MISS);
        // CTL reads: all eight ENABLE, IMASK and count cases, and a stored
        // ISTATUS bit is never what the read gives.
        let read = timer_thunk(PC, 7, TimerAccess::ReadControl, places).code;
        let mut now = 180;
        for ctl in 0..8u64 {
            for past in [false, true] {
                now += 10;
                published.offset.store(now, Relaxed);
                let cval = if past { now } else { now + 1 };
                data(&mut |d| (d.cval, d.ctl) = (cval, ctl));
                state.x[7] = 0xdead;
                assert_eq!(run(&read, &mut state), NEXT);
                let status = if ctl & ENABLE != 0 && past {
                    ISTATUS
                } else {
                    0
                };
                assert_eq!(state.x[7], ctl & !ISTATUS | status, "ctl {ctl} past {past}");
                data(&mut |d| assert_eq!((d.cval, d.ctl), (cval, ctl)));
            }
        }
        let discard = timer_thunk(PC, 31, TimerAccess::ReadControl, places).code;
        let before = state.x;
        assert_eq!(run(&discard, &mut state), NEXT);
        assert_eq!(state.x, before);
    }
}
