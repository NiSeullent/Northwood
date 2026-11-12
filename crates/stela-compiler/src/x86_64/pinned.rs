//! Guest registers that live in host registers while native code runs.
//!
//! A frame access is a load and a store around nearly every guest instruction,
//! 7 bytes each, and the code they make outgrows the instruction cache. So the
//! guest registers a kernel and its programs use most stay in host registers
//! across blocks, chained ones too. `GuestState` holds them only outside native
//! code: the entry from the engine loads them and the return stores them, and
//! a fault's settlement ([`settle`]) takes them from the host's capture. No
//! block spills or fills them, which leaves nothing to reconcile at a chain.
//!
//! These host registers are the ones emitters do not take for scratch; RAX,
//! RCX, RDX and R8 to R11 stay free. An emitter that needs one regardless goes
//! through the frame, [`frame_bound`], with the registers around it written
//! back and reloaded.
use super::asm::{Alu, Asm, Code, Gpr, Mem, Rm, ShiftOp, Size};
use super::{flag_word, logical, nzcv, right_operand, store};
use crate::{
    GuestAddress, GuestState, TargetCpu,
    aarch64::{
        Instruction, Logical, Operation, Register, Shift, WideMove, Width, pc_relative, simd,
    },
    block::HostGprSnapshot,
};
use alloc::vec::Vec;

/// Each pinned guest register and the host register that holds it. SP takes
/// RBX, the host register an emitter may borrow around an access
/// (`atomic::compare_exchange_pair`): SP is never a CASP operand, and the
/// recovery row of that access names where RBX went.
pub(crate) const PINNED: [(Register, Gpr); 7] = [
    (Register::Sp, Gpr::Rbx),
    (Register::X(8), Gpr::Rsi),
    (Register::X(19), Gpr::R14),
    (Register::X(0), Gpr::R15),
    (Register::X(1), Gpr::Rbp),
    (Register::X(16), Gpr::R12),
    (Register::X(17), Gpr::R13),
];

/// The host register holding `register`.
pub(crate) fn host(register: Register) -> Option<Gpr> {
    PINNED
        .iter()
        .find(|(pinned, _)| *pinned == register)
        .map(|(_, host)| *host)
}

mod span;
pub(super) use span::{Dirty, Run, Span, write_back as write_back_locals};

const LOCAL_HOSTS: [Gpr; 4] = [Gpr::R8, Gpr::R9, Gpr::R10, Gpr::R11];
pub(super) type LocalRegisters = [Option<Register>; LOCAL_HOSTS.len()];

pub(super) fn host_or_local(register: Register, locals: Option<LocalRegisters>) -> Option<Gpr> {
    host(register).or_else(|| {
        locals?
            .iter()
            .position(|local| *local == Some(register))
            .map(|index| LOCAL_HOSTS[index])
    })
}

/// Whether the frame slot at `offset` holds no Arm state while native code
/// runs.
#[cfg(debug_assertions)]
pub(crate) fn slot(offset: i32) -> bool {
    const NZCV: i32 = core::mem::offset_of!(GuestState, nzcv) as i32;
    offset & !3 == NZCV
        || PINNED.iter().any(|(register, _)| {
            super::offset(*register).is_some_and(|slot| offset & !7 == slot as i32)
        })
}

/// The pinned registers' values in a fault's `capture`, into `state`.
pub fn settle(capture: &HostGprSnapshot, state: &mut GuestState) {
    for (register, host) in PINNED {
        let value = capture.get(host.number()).unwrap_or_default();
        match register {
            Register::X(index) => state.x[usize::from(index)] = value,
            Register::Sp => state.sp = value,
            Register::Zero => {}
        }
    }
    state.nzcv = flag_word::arm(state.nzcv);
}

/// Whether `operation`'s emitter reaches guest registers and flags in the
/// frame only, or takes a host register that holds one.
pub(crate) fn frame_bound(operation: &Operation) -> bool {
    matches!(
        operation,
        Operation::Flags(_)
            | Operation::TagArithmetic(_)
            | Operation::Simd(simd::Operand::Sha512(_))
    )
}

/// The host register a result for `register` is best computed in: its own,
/// or RAX for [`finish`] to store.
fn working(code: &mut Code, register: Register) -> Gpr {
    code.host(register).unwrap_or(Gpr::Rax)
}

/// `destination` = the guest `register` at `size`, or zero for XZR.
fn take(asm: &mut Asm, size: Size, destination: Gpr, register: Register) {
    match asm.guest(register) {
        Some(Rm::Reg(host)) if host == destination && size == Size::Qword => {}
        Some(storage) => asm.mov_load(size, destination, storage),
        None => asm.alu(Alu::Xor, Size::Dword, destination, destination),
    }
}

/// `destination` = `value`, in the fewest bytes.
fn constant(asm: &mut Asm, destination: Gpr, value: u64) {
    match u32::try_from(value) {
        Ok(low) => asm.mov_imm32(destination, low),
        Err(_) => asm.mov_imm64(destination, value),
    }
}

/// The result in `work` stored if it is not already home, then NZCV.
fn finish(code: &mut Code, work: Gpr, destination: Register, flags: bool, subtract: bool) {
    finish_with(code, work, destination, flags, subtract, nzcv);
}

fn finish_with(
    code: &mut Code,
    work: Gpr,
    destination: Register,
    flags: bool,
    subtract: bool,
    capture: impl FnOnce(&mut Code, bool),
) {
    if work == Gpr::Rax {
        store(code, destination);
    }
    if flags {
        capture(code, subtract);
    }
}

/// The right operand of an ALU instruction whose result goes to `work`: its
/// host register when it may be used as it is, else staged in RCX.
fn right(
    code: &mut Code,
    register: Register,
    width: Width,
    shift: Shift,
    amount: u8,
    work: Gpr,
) -> Gpr {
    match code.host(register) {
        Some(host) if amount == 0 && host != work => host,
        _ => {
            right_operand(code, register, width, shift, amount);
            Gpr::Rcx
        }
    }
}

/// A binary operation without a shifted operand can read a frame slot
/// directly. A commutative operation can start with the right operand
/// when it already lives in the destination.
pub(super) fn unshifted_binary(
    asm: &mut Asm,
    op: Alu,
    size: Size,
    work: Gpr,
    mut left: Register,
    mut right: Register,
) -> bool {
    if asm.guest(right) == Some(Rm::Reg(work)) && right != left {
        if op == Alu::Sub {
            return false;
        }
        debug_assert_ne!(op, Alu::Cmp, "CMP has no destination register");
        core::mem::swap(&mut left, &mut right);
    }
    let destination = match asm.guest(left) {
        Some(Rm::Reg(host)) if op == Alu::Cmp => host,
        _ => {
            // The ALU operation itself clears the upper half of a W result.
            take(asm, Size::Qword, work, left);
            work
        }
    };
    match asm.guest(right) {
        Some(Rm::Reg(host)) => asm.alu(op, size, destination, host),
        Some(frame) => asm.alu_load(op, size, destination, frame),
        None => asm.alu_imm8(op, size, destination, 0),
    }
    true
}

/// The preceding flag capture left native CF equal to the guest carry.
/// Keep it through operand preparation, then use the direct ADC or SBB.
pub(super) fn emit_live_carry(code: &mut Code, operation: &Operation) {
    emit_carry(code, operation, Carry::Native { borrow: false }, true);
}

#[derive(Clone, Copy)]
enum Carry {
    Frame,
    Native { borrow: bool },
}

/// The body uses only GPRs and the owned frame. Each result reaches its
/// canonical home immediately, but only the final row materializes NZCV.
pub(super) fn emit_carry_run(
    code: &mut Code,
    source_map: &mut Vec<(GuestAddress, usize)>,
    instructions: &[Instruction],
    live: bool,
) -> Option<usize> {
    const MAX_ROWS: usize = 8;
    let start = instructions.first()?.pc.0;
    let count = instructions
        .iter()
        .take(MAX_ROWS)
        .enumerate()
        .take_while(|&(index, instruction)| {
            start.checked_add(index as u64 * 4) == Some(instruction.pc.0)
                && matches!(
                    instruction.operation,
                    Operation::AddSubCarry { flags: true, .. }
                )
        })
        .count();
    if count < 2 {
        return None;
    }
    let mut carry = if live {
        Carry::Native { borrow: false }
    } else {
        Carry::Frame
    };
    for (index, instruction) in instructions[..count].iter().enumerate() {
        source_map.push((instruction.pc, code.len()));
        carry = emit_carry(code, &instruction.operation, carry, index + 1 == count);
    }
    Some(count)
}

fn emit_carry(code: &mut Code, operation: &Operation, incoming: Carry, commit: bool) -> Carry {
    let Operation::AddSubCarry {
        width,
        subtract,
        flags,
        destination,
        left,
        right,
    } = *operation
    else {
        unreachable!("carry consumer")
    };
    let work = working(code, destination);
    let size = Size::of(width == Width::X64);
    let mut asm = code.asm();
    let left = asm.guest(left);
    let mut right = asm.guest(right);
    // Loading a different left operand may overwrite the pinned right.
    if right == Some(Rm::Reg(work)) && left != right {
        asm.mov(size, Gpr::Rcx, work);
        right = Some(Rm::Reg(Gpr::Rcx));
    }
    match left {
        None => asm.mov_imm32(work, 0), // XOR would destroy CF.
        Some(storage) if storage != Rm::Reg(work) => asm.mov_load(size, work, storage),
        _ => {}
    }
    let incoming_borrow = match incoming {
        Carry::Frame => {
            flag_word::carry(&mut asm, flag_word::ARM_SHIFT + 1);
            false
        }
        Carry::Native { borrow } => borrow,
    };
    // ADC consumes carry, SBB consumes borrow. Within a run, equally
    // oriented operations keep CF unchanged between producer and consumer.
    if incoming_borrow != subtract {
        asm.cmc();
    }
    let alu = if subtract { Alu::Sbb } else { Alu::Adc };
    match right {
        None => asm.alu_imm8(alu, size, work, 0),
        Some(Rm::Reg(right)) => asm.alu(alu, size, work, right),
        Some(Rm::Mem(right)) => asm.alu_load(alu, size, work, right),
        Some(Rm::Xmm(_)) => unreachable!("scalar guest operand"),
    }
    let commit = flags && commit;
    finish(code, work, destination, commit, subtract);
    Carry::Native {
        borrow: subtract && !commit,
    }
}

/// Compare the left operand against the prior instruction's result in RAX.
pub(super) fn compare_held_right(code: &mut Code, width: Width, left: Register) {
    let mut asm = code.asm();
    let left = asm.guest(left).expect("CMP left register");
    asm.alu(Alu::Cmp, Size::of(width == Width::X64), left, Gpr::Rax);
}

/// Compare the guest value that the previous instruction left in RAX.
pub(super) fn compare_held_immediate(code: &mut Code, width: Width, immediate: u32) {
    let size = Size::of(width == Width::X64);
    code.asm()
        .alu_imm(Alu::Cmp, size, Gpr::Rax, immediate as i32);
}

pub(super) fn emit_pc_relative(code: &mut Code, operand: pc_relative::Operand) {
    // Fixed in size at every address, for a stored block's rebase.
    let work = working(code, operand.destination);
    code.asm().mov_imm64(work, operand.address);
    finish(code, work, operand.destination, false, false);
}

/// Common integer operations on their native register or frame operands.
pub(super) fn emit_direct(code: &mut Code, operation: &Operation, target: TargetCpu) -> bool {
    emit_direct_with_add_sub_capture(code, operation, target, nzcv)
}

/// Use a different capture for ADD/SUB when its successor can read live flags.
pub(super) fn emit_direct_with_add_sub_capture(
    code: &mut Code,
    operation: &Operation,
    target: TargetCpu,
    capture: impl FnOnce(&mut Code, bool),
) -> bool {
    match *operation {
        Operation::AddSub {
            width,
            subtract,
            flags,
            destination,
            source,
            immediate,
        } => {
            let size = Size::of(width == Width::X64);
            let op = match (subtract, destination) {
                (true, Register::Zero) => Alu::Cmp,
                (true, _) => Alu::Sub,
                (false, _) => Alu::Add,
            };
            let work = working(code, destination);
            let immediate = immediate as i32;
            let mut asm = code.asm();
            match asm.guest(source) {
                Some(operand) if op == Alu::Cmp => {
                    asm.alu_imm(op, size, operand, immediate);
                }
                Some(Rm::Reg(host)) if !flags && host != work => {
                    let distance = if subtract { -immediate } else { immediate };
                    asm.lea(size, work, Mem::base(host, distance));
                }
                _ => {
                    take(&mut asm, Size::Qword, work, source);
                    asm.alu_imm(op, size, work, immediate);
                }
            }
            finish_with(code, work, destination, flags, subtract, capture);
        }
        Operation::AddSubRegister {
            width,
            subtract,
            flags,
            destination,
            left,
            right: operand,
            shift,
            amount,
        } => {
            let size = Size::of(width == Width::X64);
            let op = match (subtract, destination) {
                (true, Register::Zero) => Alu::Cmp,
                (true, _) => Alu::Sub,
                (false, _) => Alu::Add,
            };
            let work = working(code, destination);
            if amount != 0 || !unshifted_binary(&mut code.asm(), op, size, work, left, operand) {
                let frame = !code.is_pinned();
                if frame {
                    take(&mut code.asm(), Size::Qword, work, left);
                }
                let second = right(code, operand, width, shift, amount, work);
                let mut asm = code.asm();
                match asm.guest(left) {
                    Some(Rm::Reg(host)) if op == Alu::Cmp => asm.alu(op, size, host, second),
                    _ => {
                        if !frame {
                            take(&mut asm, Size::Qword, work, left);
                        }
                        asm.alu(op, size, work, second);
                    }
                }
            }
            finish_with(code, work, destination, flags, subtract, capture);
        }
        Operation::LogicalRegister {
            width,
            kind,
            invert,
            flags,
            destination,
            left,
            right: operand,
            shift,
            amount,
        } => {
            let size = Size::of(width == Width::X64);
            let work = working(code, destination);
            if kind == Logical::And
                && invert
                && target.supports(TargetCpu::Bmi1)
                && !target.data_independent_timing()
                && let Some(left) = code.asm().guest(left)
            {
                let inverted = match code.host(operand) {
                    Some(host) if amount == 0 => host,
                    _ => {
                        right_operand(code, operand, width, shift, amount);
                        Gpr::Rcx
                    }
                };
                // ANDN folds a frame operand and sets N/Z with C/V clear.
                code.asm().andn(size, work, inverted, left);
            } else if kind == Logical::And
                && invert
                && amount == 0
                && !target.data_independent_timing()
                && let Some(inverted) = code.host(operand)
                && inverted != work
            {
                // (left | right) ^ right is left & !right. Keep the resident
                // right unchanged and avoid staging its complement. XOR
                // also sets N/Z and clears C/V for BICS.
                let mut asm = code.asm();
                take(&mut asm, Size::Qword, work, left);
                asm.alu(Alu::Or, size, work, inverted);
                asm.alu(Alu::Xor, size, work, inverted);
            } else if kind == Logical::Or
                && left == Register::Zero
                && !invert
                && amount == 0
                && !flags
            {
                // MOV.
                take(&mut code.asm(), size, work, operand);
            } else if invert
                || amount != 0
                || !unshifted_binary(&mut code.asm(), logical(kind), size, work, left, operand)
            {
                let frame = !code.is_pinned();
                if frame {
                    take(&mut code.asm(), Size::Qword, work, left);
                }
                let second = if invert {
                    right_operand(code, operand, width, shift, amount);
                    code.asm().unary(super::asm::Unary::Not, size, Gpr::Rcx);
                    Gpr::Rcx
                } else {
                    right(code, operand, width, shift, amount, work)
                };
                let mut asm = code.asm();
                if !frame {
                    take(&mut asm, Size::Qword, work, left);
                }
                asm.alu(logical(kind), size, work, second);
            }
            finish(code, work, destination, flags, false);
        }
        Operation::LogicalImmediate {
            width,
            kind,
            flags,
            destination,
            source,
            immediate,
        } => {
            let size = Size::of(width == Width::X64);
            let work = working(code, destination);
            let mut asm = code.asm();
            take(&mut asm, Size::Qword, work, source);
            match i32::try_from(immediate as i64) {
                Ok(short) => asm.alu_imm(logical(kind), size, work, short),
                Err(_) => {
                    asm.mov_imm64(Gpr::Rcx, immediate);
                    asm.alu(logical(kind), size, work, Gpr::Rcx);
                }
            }
            finish(code, work, destination, flags, false);
        }
        Operation::MoveWide {
            width,
            kind,
            destination,
            immediate,
            shift,
        } if code.is_pinned() => {
            let work = working(code, destination);
            let value = u64::from(immediate) << shift;
            let mut asm = code.asm();
            match kind {
                WideMove::Zero => constant(&mut asm, work, value),
                WideMove::Not if width == Width::W32 => {
                    asm.mov_imm32(work, !value as u32);
                }
                WideMove::Not => asm.mov_imm64(work, !value),
                WideMove::Keep => {
                    take(&mut asm, Size::Qword, work, destination);
                    // The field brought down to bits 15:0, replaced, and
                    // taken back up.
                    if shift != 0 {
                        asm.shift_imm(ShiftOp::Ror, Size::Qword, work, shift);
                    }
                    asm.mov_imm16(work, immediate);
                    if shift != 0 {
                        asm.shift_imm(ShiftOp::Rol, Size::Qword, work, shift);
                    }
                    if width == Width::W32 {
                        asm.mov(Size::Dword, work, work);
                    }
                }
            }
            finish(code, work, destination, false, false);
        }
        Operation::PcRelative(operand)
            if code.is_pinned() && operand.destination != Register::Zero =>
        {
            emit_pc_relative(code, operand);
        }
        _ => return false,
    }
    true
}

/// The host registers a fault in native code finds `state`'s pinned
/// registers in, for what stands in for native code.
pub fn capture(state: &mut GuestState) -> HostGprSnapshot {
    state.nzcv = flag_word::packed(state.nzcv);
    let mut registers = [0; 16];
    for (register, host) in PINNED {
        registers[usize::from(host.number())] = match register {
            Register::X(index) => state.x[usize::from(index)],
            Register::Sp => state.sp,
            Register::Zero => 0,
        };
    }
    HostGprSnapshot::legacy(registers)
}

/// `body`, whose exits return, in code that runs it as the engine does: the
/// pinned registers loaded around it. The engine's own entry does this; this
/// is for running a block anywhere else.
pub fn entered(body: &[u8]) -> alloc::vec::Vec<u8> {
    const SAVE: [u8; 10] = [0x53, 0x55, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57];
    const RESTORE: [u8; 11] = [
        0x41, 0x5f, 0x41, 0x5e, 0x41, 0x5d, 0x41, 0x5c, 0x5d, 0x5b, 0xc3,
    ];
    let mut loads = Code::new();
    reload(&mut loads);
    let mut stores = Code::new();
    write_back(&mut stores);
    let mut code = SAVE.to_vec();
    code.extend_from_slice(&loads);
    code.push(0xe8);
    code.extend(((stores.len() + RESTORE.len()) as i32).to_le_bytes());
    code.extend_from_slice(&stores);
    code.extend(RESTORE);
    code.extend_from_slice(body);
    code
}

/// Code that runs the block at RCX as the engine does, for a host that keeps
/// the callee-saved registers itself and calls this with RDI at the frame.
pub fn entry() -> alloc::vec::Vec<u8> {
    const TAKE_BLOCK: [u8; 3] = [0x49, 0x89, 0xcb]; // mov r11,rcx
    const CALL_BLOCK: [u8; 3] = [0x41, 0xff, 0xd3]; // call r11
    const RET: u8 = 0xc3;
    let mut loads = Code::new();
    reload(&mut loads);
    let mut stores = Code::new();
    write_back(&mut stores);
    let mut code = TAKE_BLOCK.to_vec();
    code.extend_from_slice(&loads);
    code.extend(CALL_BLOCK);
    code.extend_from_slice(&stores);
    code.push(RET);
    code
}

/// The frame made exact for a [`frame_bound`] emission.
pub(crate) fn write_back(code: &mut Code) {
    flag_word::open(code);
    let mut asm = code.asm();
    for (register, host) in PINNED {
        let slot = super::offset(register).expect("pinned registers have frame slots");
        asm.mov_store(Size::Qword, super::frame(slot), host);
    }
}

/// The pinned registers taken from the frame after a [`frame_bound`] emission.
pub(crate) fn reload(code: &mut Code) {
    flag_word::close(code);
    let mut asm = code.asm();
    for (register, host) in PINNED {
        let slot = super::offset(register).expect("pinned registers have frame slots");
        asm.mov_load(Size::Qword, host, super::frame(slot));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_settled_fault_finds_each_register_in_its_host() {
        let mut registers = [0; 16];
        for (index, (_, host)) in PINNED.iter().enumerate() {
            registers[usize::from(host.number())] = index as u64 + 1;
        }
        let mut state = GuestState::default();
        settle(&HostGprSnapshot::legacy(registers), &mut state);
        assert_eq!(state.sp, 1);
        assert_eq!(state.x[8], 2);
        assert_eq!(state.x[19], 3);
        assert_eq!(state.x[0], 4);
        assert_eq!(state.x[1], 5);
        assert_eq!(state.x[16], 6);
        assert_eq!(state.x[17], 7);
    }

    #[test]
    fn pinned_code_moves_registers_and_frame_code_keeps_its_slots() {
        let mut pinned = Code::pinned();
        super::super::load(&mut pinned, Register::X(0));
        super::super::store(&mut pinned, Register::Sp);
        // mov rax,r15; mov rbx,rax.
        assert_eq!(&pinned[..], [0x49, 0x8b, 0xc7, 0x48, 0x89, 0xc3]);
        let mut frame = Code::new();
        super::super::load(&mut frame, Register::X(0));
        assert_eq!(&frame[..], [0x48, 0x8b, 0x87, 0, 0, 0, 0]);
    }

    #[test]
    fn immediate_cmp_reads_the_frame_and_preserves_its_operand() {
        let mut initial = GuestState {
            sp: 0xfff000,
            ..GuestState::default()
        };
        for width in [Width::W32, Width::X64] {
            initial.x[2] = match width {
                Width::W32 => 0x1234_5678_8000_0000,
                Width::X64 => 1 << 63,
            };
            for (source, immediate, expected) in
                [(Register::X(2), 31, 3), (Register::Sp, 0xfff000, 6)]
            {
                let operation = Operation::AddSub {
                    width,
                    subtract: true,
                    flags: true,
                    destination: Register::Zero,
                    source,
                    immediate,
                };
                let mut frame = Code::new();
                assert!(emit_direct(&mut frame, &operation, TargetCpu::Baseline));
                let start = usize::from(width == Width::X64);
                assert_eq!(frame[start], if immediate <= 127 { 0x83 } else { 0x81 });
                assert_eq!(frame[start + 1], 0xbf); // CMP with a frame displacement.

                let mut pinned = direct(operation);
                pinned.asm().ret();
                let pinned = entered(&pinned);
                for bytes in [&frame[..], &pinned[..]] {
                    let mut state = initial.clone();
                    super::super::native::run(bytes, &mut state);
                    assert_eq!(state.x, initial.x);
                    assert_eq!(state.sp, initial.sp);
                    assert_eq!(state.nzcv, expected << 28);
                }
            }
        }
    }

    fn direct(operation: Operation) -> Code {
        let mut code = Code::pinned();
        assert!(emit_direct(&mut code, &operation, TargetCpu::Baseline));
        code
    }

    #[test]
    fn direct_forms_compute_where_their_result_lives() {
        // add x8,x0,#16: lea rsi,[r15+16].
        let add = direct(Operation::AddSub {
            width: Width::X64,
            subtract: false,
            flags: false,
            destination: Register::X(8),
            source: Register::X(0),
            immediate: 16,
        });
        assert_eq!(&add[..], [0x49, 0x8d, 0x77, 0x10]);
        // mov x0,x19: mov r15,r14.
        let mov = direct(Operation::LogicalRegister {
            width: Width::X64,
            kind: Logical::Or,
            invert: false,
            flags: false,
            destination: Register::X(0),
            left: Register::Zero,
            right: Register::X(19),
            shift: Shift::Lsl,
            amount: 0,
        });
        assert_eq!(&mov[..], [0x4d, 0x8b, 0xfe]);
        // mov w0,w0 keeps the low half.
        let narrow = direct(Operation::LogicalRegister {
            width: Width::W32,
            kind: Logical::Or,
            invert: false,
            flags: false,
            destination: Register::X(0),
            left: Register::Zero,
            right: Register::X(0),
            shift: Shift::Lsl,
            amount: 0,
        });
        assert!(!narrow.is_empty());
    }

    #[test]
    fn bmi1_bic_folds_the_frame_operand_and_keeps_older_targets() {
        // BIC X8,X9,X10 from archived desktop code (8a2a0128).
        let operation = Operation::LogicalRegister {
            width: Width::X64,
            kind: Logical::And,
            invert: true,
            flags: false,
            destination: Register::X(8),
            left: Register::X(9),
            right: Register::X(10),
            shift: Shift::Lsl,
            amount: 0,
        };
        let emit = |target| {
            let mut code = Code::pinned();
            assert!(emit_direct(&mut code, &operation, target));
            code
        };
        let fast = emit(TargetCpu::Bmi1);
        // mov rcx,[rdi+80]; andn rsi,rcx,[rdi+72].
        assert_eq!(
            &fast[..],
            [0x48, 0x8b, 0x4f, 0x50, 0xc4, 0xe2, 0xf0, 0xf2, 0x77, 0x48]
        );
        let baseline = emit(TargetCpu::Baseline);
        assert_eq!(baseline.instructions(), 4);
        assert_eq!(fast.instructions(), 2);
        for target in [
            TargetCpu::Sse42,
            TargetCpu::Bmi2,
            TargetCpu::Bmi1.with_data_independent_timing(),
        ] {
            assert_eq!(emit(target), baseline);
        }
    }

    #[test]
    fn unshifted_bic_uses_two_ops_for_a_resident_inverted_operand_without_bmi1() {
        // BIC W16,W16,W15 from the 26B123 kernel. W16 is pinned to R12D and
        // the hot block keeps W15 in its second local register, R9D.
        let operation = Operation::LogicalRegister {
            width: Width::W32,
            kind: Logical::And,
            invert: true,
            flags: false,
            destination: Register::X(16),
            left: Register::X(16),
            right: Register::X(15),
            shift: Shift::Lsl,
            amount: 0,
        };
        let emit = |target| {
            let mut code = Code::pinned();
            code.set_local_registers(Some([None, Some(Register::X(15)), None, None]));
            assert!(emit_direct(&mut code, &operation, target));
            code
        };
        let target = TargetCpu::X86_64_V2.with_lzcnt();
        let compact = emit(target);
        // or r12d,r9d; xor r12d,r9d.
        assert_eq!(&compact[..], [0x45, 0x09, 0xcc, 0x45, 0x31, 0xcc]);
        assert_eq!(compact.instructions(), 2);
        assert_eq!(
            emit(target.with_data_independent_timing()).instructions(),
            3
        );
    }

    #[test]
    fn bic_preserves_aliases_and_bics_flags_on_both_paths() {
        let host = TargetCpu::detect();
        let mut initial = GuestState::default();
        initial.x[0] = 0x89ab_cdef_8000_0000;
        initial.x[8] = 0x5555_5555_0000_0000;
        initial.x[9] = u64::MAX;
        initial.nzcv = 0xf000_0000;
        for width in [Width::W32, Width::X64] {
            for (shift, amount) in [(Shift::Lsl, 0), (Shift::Asr, 1), (Shift::Ror, 3)] {
                for (destination, left, right) in [
                    (Register::X(8), Register::X(8), Register::X(0)),
                    (Register::X(8), Register::X(0), Register::X(8)),
                    (Register::X(8), Register::X(9), Register::X(9)),
                    (Register::Zero, Register::X(0), Register::X(8)),
                ] {
                    let operation = Operation::LogicalRegister {
                        width,
                        kind: Logical::And,
                        invert: true,
                        flags: true,
                        destination,
                        left,
                        right,
                        shift,
                        amount,
                    };
                    let run = |target, pinned| {
                        let mut code = if pinned { Code::pinned() } else { Code::new() };
                        assert!(emit_direct(&mut code, &operation, target));
                        code.asm().ret();
                        let bytes = if pinned {
                            entered(&code)
                        } else {
                            code.into_bytes()
                        };
                        let mut state = initial.clone();
                        super::super::native::run(&bytes, &mut state);
                        state
                    };
                    for pinned in [false, true] {
                        let state = run(TargetCpu::Baseline, pinned);
                        assert_eq!(
                            state,
                            run(TargetCpu::Baseline.with_data_independent_timing(), pinned)
                        );
                        if host.supports(TargetCpu::Bmi1) {
                            assert_eq!(state, run(TargetCpu::Bmi1, pinned));
                        }
                        assert_eq!(state.nzcv & 0x3000_0000, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn binary_operands_keep_aliases_widths_and_flags() {
        use crate::{
            GuestAddress, Options, TargetCpu,
            block::{BlockProfile, Features},
        };
        const BASE: GuestAddress = GuestAddress(0x1000);
        let pinned = Options::new(
            BlockProfile::new(Features::SYSTEM_MODE),
            TargetCpu::Baseline,
        );
        let frame = Options::new(BlockProfile::ORDINARY, TargetCpu::Baseline);
        let mut initial = GuestState::default();
        initial.x[0] = 0x1234_5678_7fff_fffe;
        initial.x[1] = 0xfedc_ba98_0000_0003;
        initial.x[3] = 0x9182_7364_ffff_ffff;
        initial.x[16] = 0xaaff_5500_8000_0000;
        initial.x[17] = 0x0246_8ace_ffff_ff00;
        initial.nzcv = 0x3000_0000;
        for base in [
            0x0b00_0000, // ADD
            0x0b00_0800, // ADD, LSL #2
            0x2b00_0000, // ADDS
            0x4b00_0000, // SUB
            0x6b00_0000, // SUBS
            0x4a00_0000, // EOR
            0x6a00_0000, // ANDS
        ] {
            for (rd, rn, rm) in [
                (1, 0, 1),
                (16, 17, 1),
                (17, 1, 16),
                (0, 3, 0),
                (3, 0, 3),
                (1, 1, 1),
                (31, 16, 3),
                (1, 31, 1),
                (16, 17, 31),
            ] {
                for width in [0u32, 1 << 31] {
                    let word = base | width | rm << 16 | rn << 5 | rd;
                    let run = |options, wrap| {
                        let native = crate::compile(&word.to_le_bytes(), BASE, options).unwrap();
                        let code = if wrap {
                            entered(&native.native.x86_64_code)
                        } else {
                            native.native.x86_64_code
                        };
                        let mut state = initial.clone();
                        crate::x86_64::native::run(&code, &mut state);
                        state
                    };
                    assert_eq!(run(&pinned, true), run(&frame, false), "{word:08x}");
                }
            }
        }
    }

    struct Random(u64);
    impl Random {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 16) as u32
        }
        fn below(&mut self, bound: u32) -> u32 {
            self.next() % bound
        }
        fn pick(&mut self, choices: &[u32]) -> u32 {
            choices[self.below(choices.len() as u32) as usize]
        }
    }

    /// Registers instructions compute, pinned and not, and those that stay
    /// addresses: bases into the test's buffer and an index.
    const DATA: [u32; 9] = [0, 1, 3, 8, 9, 16, 17, 21, 30];
    const BASES: [u32; 3] = [19, 20, 31];
    const INDEX: u32 = 5;

    /// One random instruction of the common forms, with selects, compares
    /// and, given `branches`, forward conditional branches past the next one.
    fn instruction(random: &mut Random, branches: bool) -> u32 {
        let wide = random.below(2);
        let (rd, rn, rm) = (random.pick(&DATA), random.pick(&DATA), random.pick(&DATA));
        let zero_or = |random: &mut Random, register: u32| {
            if random.below(6) == 0 { 31 } else { register }
        };
        let rn = zero_or(random, rn);
        let rm = zero_or(random, rm);
        let base = random.pick(&BASES);
        match random.below(19) {
            // ADD and SUB immediate; a target or source of 31 is SP, so
            // only compares use it.
            0 => {
                let (op, flags) = (random.below(2), random.below(2));
                let rn = random.pick(&DATA);
                wide << 31
                    | op << 30
                    | flags << 29
                    | 0x1100_0000
                    | random.below(4096) << 10
                    | rn << 5
                    | rd
            }
            1 => {
                let op = random.below(2);
                let rn = random.pick(&DATA);
                wide << 31 | op << 30 | 1 << 29 | 0x1100_001f | random.below(4096) << 10 | rn << 5
            }
            // ADD and SUB shifted register, CMP among them.
            2 => {
                let (op, flags) = (random.below(2), random.below(2));
                let rd = if flags == 1 { zero_or(random, rd) } else { rd };
                wide << 31
                    | op << 30
                    | flags << 29
                    | 0x0b00_0000
                    | random.below(3) << 22
                    | rm << 16
                    | random.below(if wide == 1 { 64 } else { 32 }) << 10
                    | rn << 5
                    | rd
            }
            // AND, ORR, EOR, ANDS with every shift, BIC and the like.
            3 => {
                wide << 31
                    | random.below(4) << 29
                    | 0x0a00_0000
                    | random.below(4) << 22
                    | random.below(2) << 21
                    | rm << 16
                    | random.below(if wide == 1 { 64 } else { 32 }) << 10
                    | rn << 5
                    | rd
            }
            // MOV, and from ZR.
            4 => wide << 31 | 1 << 29 | 0x0a00_0000 | rm << 16 | 31 << 5 | rd,
            // Logical immediates, wide in N and rotation.
            5 => {
                let opc = random.below(4);
                let immediate = if wide == 1 {
                    1 << 22 | random.below(64) << 16 | random.below(63) << 10
                } else {
                    random.below(32) << 16 | random.below(31) << 10
                };
                let rn = random.pick(&DATA);
                let rd = if opc == 3 { rd } else { random.pick(&DATA) };
                wide << 31 | opc << 29 | 0x1200_0000 | immediate | rn << 5 | rd
            }
            // MOVN, MOVZ, MOVK at each position.
            6 => {
                let opc = random.pick(&[0, 2, 3]);
                let position = random.below(if wide == 1 { 4 } else { 2 });
                wide << 31
                    | opc << 29
                    | 0x1280_0000
                    | position << 21
                    | random.below(0x1_0000) << 5
                    | rd
            }
            // ADR and ADRP.
            7 => random.below(2) << 31 | 0x1000_0000 | random.below(0x4000) << 5 | rd,
            // Unsigned-offset loads and stores of each size and sign.
            8 => {
                let (size, opc) = [
                    (0, 0),
                    (0, 1),
                    (0, 2),
                    (0, 3),
                    (1, 0),
                    (1, 1),
                    (1, 2),
                    (1, 3),
                    (2, 0),
                    (2, 1),
                    (2, 2),
                    (3, 0),
                    (3, 1),
                ][random.below(13) as usize];
                size << 30 | opc << 22 | 0x3900_0000 | random.below(64) << 10 | base << 5 | rd
            }
            // LDP and STP of X and W.
            9 => {
                let (load, rt2) = (random.below(2), random.pick(&DATA));
                let rt2 = if rt2 == rd { 4 } else { rt2 };
                wide << 31
                    | load << 22
                    | 0x2900_0000
                    | random.below(32) << 15
                    | rt2 << 10
                    | base << 5
                    | rd
            }
            // Pre- and post-index.
            10 => {
                let (load, mode) = (random.below(2), random.pick(&[1, 3]));
                // SP stays aligned.
                let base = random.pick(&BASES[..2]);
                3 << 30
                    | load << 22
                    | 0x3800_0000
                    | (random.below(32) * 8) << 12
                    | mode << 10
                    | base << 5
                    | rd
            }
            // Register offset.
            11 => {
                let load = random.below(2);
                3 << 30 | load << 22 | 0x3820_6800 | INDEX << 16 | base << 5 | rd
            }
            // SP moves as code moves it.
            12 => 0x9100_43ff,
            13 => 0xd100_43ff,
            // CSEL, CSINC, CSINV and CSNEG.
            14 => {
                wide << 31
                    | random.below(2) << 30
                    | 0x1a80_0000
                    | rm << 16
                    | random.below(14) << 12
                    | random.below(2) << 10
                    | rn << 5
                    | rd
            }
            // CCMP and CCMN of a register.
            15 => {
                wide << 31
                    | random.below(2) << 30
                    | 0x3a40_0000
                    | rm << 16
                    | random.below(14) << 12
                    | rn << 5
                    | random.below(16)
            }
            // B.cond, CBZ and CBNZ past the next instruction.
            16 if branches => 0x5400_0040 | random.below(14),
            17 if branches => wide << 31 | random.below(2) << 24 | 0x3400_0040 | rd,
            _ => 0xd503_201f,
        }
    }

    #[test]
    fn pinned_code_computes_what_frame_code_computes() {
        use crate::{
            GuestAddress, Options, TargetCpu,
            block::{BlockProfile, Features},
        };
        const BASE: GuestAddress = GuestAddress(0x4000_0000);
        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        let mut memory = alloc::vec![0u8; 1 << 16];
        for byte in &mut memory {
            *byte = random.next() as u8;
        }
        let start = memory.clone();
        let target = TargetCpu::Baseline;
        let pinned = Options::new(BlockProfile::new(Features::SYSTEM_MODE), target);
        let frame = Options::new(BlockProfile::ORDINARY, target);
        let middle = (memory.as_mut_ptr() as u64 + (1 << 15)) & !15;
        for round in 0..3000 {
            let words: alloc::vec::Vec<u32> = (0..1 + random.below(24))
                .map(|_| instruction(&mut random, false))
                .collect();
            let source: alloc::vec::Vec<u8> =
                words.iter().flat_map(|word| word.to_le_bytes()).collect();
            let (Ok(one), Ok(other)) = (
                crate::compile(&source, BASE, &pinned),
                crate::compile(&source, BASE, &frame),
            ) else {
                continue;
            };
            assert_eq!(one.source_bytes, other.source_bytes);
            let mut initial = GuestState::default();
            for slot in &mut initial.x {
                *slot = u64::from(random.next()) << 32 | u64::from(random.next());
            }
            initial.x[19] = middle;
            initial.x[20] = middle - 0x2000;
            initial.x[INDEX as usize] = 0x20;
            initial.sp = middle + 0x2000;
            initial.nzcv = random.below(16) << 28;
            memory.copy_from_slice(&start);
            let mut expected = initial.clone();
            crate::x86_64::native::run(&other.native.x86_64_code, &mut expected);
            let kept = memory.clone();
            memory.copy_from_slice(&start);
            let mut state = initial.clone();
            crate::x86_64::native::run(&entered(&one.native.x86_64_code), &mut state);
            assert_eq!(state.x, expected.x, "round {round}: {words:08x?}");
            assert_eq!(state.sp, expected.sp, "round {round}: {words:08x?}");
            assert_eq!(state.nzcv, expected.nzcv, "round {round}: {words:08x?}");
            assert!(memory == kept, "round {round}: {words:08x?}");
        }
    }

    /// What a field of a [`FORMS`] word holds.
    #[derive(Clone, Copy, PartialEq)]
    enum Field {
        /// Left as the form has it.
        Fixed,
        /// A computed general register.
        Data,
        /// The even register of a consecutive pair.
        Even,
        /// A general register holding an address in the test's buffer.
        Base,
        /// One no form moves, so that its alignment holds.
        Aligned,
        /// One an indexed form moves.
        Moved,
        /// A general register holding a small offset.
        Index,
        /// A vector register.
        Vector,
        /// The register bit 5 names.
        Own,
    }
    use Field::{
        Aligned as A, Base as B, Data as D, Even as E, Fixed as F, Index as I, Moved as M,
        Own as O, Vector as V,
    };

    /// Encodings with their random bits and what bits 0, 5, 10 and 16 name.
    const FORMS: &[(u32, u32, [Field; 4])] = &[
        // The flag instructions, frame-bound: CFINV, XAFLAG, AXFLAG, RMIF,
        // SETF8 and SETF16.
        (0xd500_401f, 0, [F, F, F, F]),
        (0xd500_403f, 0, [F, F, F, F]),
        (0xd500_405f, 0, [F, F, F, F]),
        (0xba00_0400, 0x001f_800f, [F, D, F, F]),
        (0x3a00_080d, 0, [F, D, F, F]),
        (0x3a00_480d, 0, [F, D, F, F]),
        // Conditional selects and compares.
        (0x1a80_0000, 0x8000_f000, [D, D, F, D]),
        (0x1a80_0400, 0x8000_f000, [D, D, F, D]),
        (0x5a80_0000, 0x8000_f000, [D, D, F, D]),
        (0x5a80_0400, 0x8000_f000, [D, D, F, D]),
        (0x7a40_0000, 0x8000_f00f, [F, D, F, D]),
        (0x3a40_0000, 0x8000_f00f, [F, D, F, D]),
        (0x7a40_0800, 0x801f_f00f, [F, D, F, F]),
        (0x3a40_0800, 0x801f_f00f, [F, D, F, F]),
        // Add and subtract with carry.
        (0x1a00_0000, 0xe000_0000, [D, D, F, D]),
        // EXTR, bitfields.
        (0x93c0_0000, 0x0000_fc00, [D, D, F, D]),
        (0x1380_0000, 0x0000_7c00, [D, D, F, D]),
        (0x9340_0000, 0x003f_fc00, [D, D, F, F]),
        (0xb340_0000, 0x003f_fc00, [D, D, F, F]),
        (0xd340_0000, 0x003f_fc00, [D, D, F, F]),
        (0x1300_0000, 0x001f_7c00, [D, D, F, F]),
        (0x3300_0000, 0x001f_7c00, [D, D, F, F]),
        (0x5300_0000, 0x001f_7c00, [D, D, F, F]),
        // Multiplies, divides, shifts, bit operations.
        (0x1b00_0000, 0x8000_8000, [D, D, D, D]),
        (0x9b20_0000, 0x0080_8000, [D, D, D, D]),
        (0x9b40_7c00, 0x0080_0000, [D, D, F, D]),
        (0x1ac0_0800, 0x8000_0400, [D, D, F, D]),
        (0x1ac0_2000, 0x8000_0c00, [D, D, F, D]),
        (0x5ac0_0000, 0x8000_0c00, [D, D, F, F]),
        (0x5ac0_1000, 0x8000_0400, [D, D, F, F]),
        // Extended registers, and what reads SP.
        (0x0b20_0000, 0xe000_fc00, [D, D, F, D]),
        (0x9100_03e0, 0x003f_fc00, [D, F, F, F]),
        (0xcb20_63e0, 0x0000_0000, [D, F, F, D]),
        // Atomics.
        (0x3820_0000, 0xc0c0_7000, [D, A, F, D]),
        (0x3820_8000, 0xc0c0_0000, [D, A, F, D]),
        (0x08a0_7c00, 0xc040_8000, [D, A, F, D]),
        (0x0820_7c00, 0x4040_8000, [E, A, F, E]),
        // Exclusives and ordered accesses.
        (0x885f_7c00, 0x4000_8000, [D, A, F, F]),
        (0x8800_7c00, 0x4000_8000, [D, A, F, D]),
        (0x88df_fc00, 0x4000_0000, [D, A, F, F]),
        (0x889f_fc00, 0x4000_0000, [D, A, F, F]),
        (0xc87f_0000, 0x0000_8000, [E, A, F, F]),
        // Unscaled, indexed and register-offset accesses of every size.
        (0x3800_0000, 0xc0df_f000, [D, B, F, F]),
        (0x3800_0400, 0xc0df_0800, [D, M, F, F]),
        (0x3820_4800, 0xc0c0_b000, [D, B, F, I]),
        // Pairs.
        (0x6940_0000, 0x001f_8000, [D, B, D, F]),
        (0xa8c0_0000, 0x013f_8000, [D, M, D, F]),
        (0xa880_0000, 0x013f_8000, [D, M, D, F]),
        // SIMD and FP accesses.
        (0x3d00_0000, 0xc003_fc00, [V, B, F, F]),
        (0x3dc0_0000, 0x0043_fc00, [V, B, F, F]),
        (0x3c00_0400, 0xc0df_0800, [V, M, F, F]),
        (0x3c20_4800, 0xc040_b000, [V, B, F, I]),
        (0x2d00_0000, 0x805f_8000, [V, B, V, F]),
        (0xad00_0000, 0x005f_8000, [V, B, V, F]),
        (0x4c00_7000, 0x0040_0000, [V, B, F, F]),
        (0x4c9f_7000, 0x0040_0000, [V, M, F, F]),
        (0x4c80_7000, 0x0040_0000, [V, M, F, I]),
        (0x4c00_8000, 0x0040_0000, [V, B, F, F]),
        (0x4d00_0000, 0x0040_1c00, [V, B, F, F]),
        (0x4d40_c000, 0x0000_0000, [V, B, F, F]),
        // Prefetches: immediate, unscaled and register-offset.
        (0xf980_0000, 0x003f_fc1f, [F, B, F, F]),
        (0xf880_0000, 0x001f_f01f, [F, B, F, F]),
        (0xf8a0_6800, 0x0000_101f, [F, B, F, I]),
        // General registers to and from vectors.
        (0x9e66_0000, 0x0000_0000, [D, V, F, F]),
        (0x9e67_0000, 0x0000_0000, [V, D, F, F]),
        (0x1e26_0000, 0x0000_0000, [D, V, F, F]),
        (0x1e27_0000, 0x0000_0000, [V, D, F, F]),
        (0x4e08_3c00, 0x0000_0000, [D, V, F, F]),
        (0x0e04_3c00, 0x0000_0000, [D, V, F, F]),
        (0x4e04_2c00, 0x0000_0000, [D, V, F, F]),
        (0x4e08_1c00, 0x0000_0000, [V, D, F, F]),
        (0x4e08_0c00, 0x0000_0000, [V, D, F, F]),
        (0x9e62_0000, 0x0000_0000, [V, D, F, F]),
        (0x9e78_0000, 0x0000_0000, [D, V, F, F]),
    ];

    /// Terminals, whose next PC the comparison also reads.
    const ENDS: &[(u32, u32, [Field; 4])] = &[
        (0x3400_0040, 0x8100_0000, [D, F, F, F]),
        (0x3600_0040, 0x80f8_0000, [D, F, F, F]),
        (0x5400_0040, 0x0000_000f, [F, F, F, F]),
        (0xd61f_0000, 0x0020_0000, [F, D, F, F]),
        // Loads into their own base, which nothing after them reads.
        (0xf940_0000, 0x0000_fc00, [O, B, F, F]),
        (0xb940_0000, 0x0000_fc00, [O, B, F, F]),
        (0x3940_0000, 0x0080_fc00, [O, B, F, F]),
        (0xf840_0000, 0x001f_f000, [O, B, F, F]),
        (0xa940_0000, 0x001f_8000, [O, B, D, F]),
        (0xa940_0000, 0x001f_8000, [D, B, O, F]),
        (0x2940_0000, 0x001f_8000, [O, B, D, F]),
        (0x6940_0000, 0x001f_8000, [D, B, O, F]),
    ];

    const COMPUTED: [u32; 7] = [0, 3, 8, 9, 16, 21, 30];
    const EVEN: [u32; 2] = [2, 26];
    /// The registers a round deals out to the roles above, so that each
    /// host register serves as each of them.
    const DEALT: [u32; 13] = [0, 1, 5, 8, 9, 16, 17, 19, 20, 21, 22, 23, 30];

    /// `register` in a round that dealt the [`DEALT`] registers as `dealt`.
    fn named(register: u32, dealt: &[u32; 13]) -> u32 {
        DEALT
            .iter()
            .position(|&role| role == register)
            .map_or(register, |at| dealt[at])
    }
    const ADDRESSES: [u32; 4] = [1, 19, 20, 31];
    const ALIGNED: [u32; 2] = [19, 31];
    const MOVED: [u32; 2] = [1, 20];
    const OFFSETS: [u32; 2] = [5, 17];

    /// `form` with its fields filled in and its random bits drawn.
    fn filled(random: &mut Random, form: &(u32, u32, [Field; 4]), dealt: &[u32; 13]) -> u32 {
        let (word, free, fields) = *form;
        let mut word = word | random.next() & free;
        let mut pair = None;
        for (field, at) in fields.into_iter().zip([0, 5, 10, 16]) {
            let register = match field {
                Field::Fixed | Field::Own => continue,
                Field::Data => {
                    let mut register = random.pick(&COMPUTED);
                    // A load pair's two registers differ, as Arm requires.
                    if pair == Some(register) {
                        register = if register == 3 { 21 } else { 3 };
                    }
                    pair = Some(register);
                    register
                }
                Field::Even => random.pick(&EVEN),
                Field::Base => random.pick(&ADDRESSES),
                Field::Aligned => random.pick(&ALIGNED),
                Field::Moved => random.pick(&MOVED),
                Field::Index => random.pick(&OFFSETS),
                Field::Vector => {
                    let mut register = random.below(32);
                    if pair == Some(register) {
                        register = (register + 1) % 32;
                    }
                    pair = Some(register);
                    register
                }
            };
            word |= named(register, dealt) << at;
        }
        for (field, at) in fields.into_iter().zip([0, 5, 10, 16]) {
            if field == Field::Own {
                word |= (word >> 5 & 31) << at;
            }
        }
        word
    }

    /// A block runs past forward conditional branches; a taken one leaves
    /// by a side exit, with the instructions it skipped in its exit.
    #[test]
    fn a_block_runs_past_forward_branches_and_leaves_by_side_exits() {
        use crate::{
            GuestAddress, Options, TargetCpu,
            block::{BlockProfile, ExitKind, Features},
        };
        const BASE: GuestAddress = GuestAddress(0x4000_0000);
        const WORDS: [u32; 5] = [
            0xb400_0080, // cbz x0, +16
            0x9100_0421, // add x1, x1, #1
            0xb500_0081, // cbnz x1, +16
            0x9100_0442, // add x2, x2, #1
            0x17ff_fffc, // b -16
        ];
        let source: alloc::vec::Vec<u8> =
            WORDS.iter().flat_map(|word| word.to_le_bytes()).collect();
        let options = Options::new(
            BlockProfile::new(Features::SYSTEM_MODE),
            TargetCpu::detect(),
        );
        let block = crate::compile(&source, BASE, &options).unwrap();
        assert_eq!(block.source_bytes, source.len());
        let exits: alloc::vec::Vec<_> = block
            .direct_exits
            .iter()
            .map(|exit| (exit.target.0 - BASE.0, exit.kind, exit.skipped))
            .collect();
        assert_eq!(
            exits,
            [
                (0, ExitKind::Direct, 0),
                (0x10, ExitKind::Direct, 4),
                (0x18, ExitKind::Direct, 2),
            ]
        );
        let code = entered(&block.native.x86_64_code);
        for (x0, x1, next, expected) in [
            (0, 7, BASE.0 + 0x10, [7, 5]),
            (1, 7, BASE.0 + 0x18, [8, 5]),
            (1, u64::MAX, BASE.0, [0, 6]),
        ] {
            let mut state = GuestState::default();
            state.x[0] = x0;
            state.x[1] = x1;
            state.x[2] = 5;
            assert_eq!(crate::x86_64::native::run(&code, &mut state), next);
            assert_eq!([state.x[1], state.x[2]], expected);
        }
    }

    /// Every form against the frame code, as a block runs it in the engine.
    #[test]
    fn every_form_computes_what_frame_code_computes() {
        use crate::{
            GuestAddress, Options, TargetCpu,
            block::{BlockProfile, Features},
        };
        const BASE: GuestAddress = GuestAddress(0x4000_0000);
        let mut random = Random(0x2545_f491_4f6c_dd1d);
        let mut memory = alloc::vec![0u8; 1 << 17];
        for byte in &mut memory {
            *byte = random.next() as u8;
        }
        let start = memory.clone();
        let target = TargetCpu::detect();
        let pinned = Options::new(BlockProfile::new(Features::SYSTEM_MODE), target);
        let frame = Options::new(BlockProfile::ORDINARY, target);
        let middle = (memory.as_mut_ptr() as u64 + (1 << 16)) & !0xff;
        let mut compared = 0;
        for round in 0..200_000 {
            let mut dealt = DEALT;
            for at in (1..dealt.len()).rev() {
                dealt.swap(at, random.below(at as u32 + 1) as usize);
            }
            let mut words: alloc::vec::Vec<u32> = (0..1 + random.below(6))
                .map(|_| {
                    let form = FORMS[random.below(FORMS.len() as u32) as usize];
                    filled(&mut random, &form, &dealt)
                })
                .collect();
            if random.below(3) == 0 {
                let form = ENDS[random.below(ENDS.len() as u32) as usize];
                words.push(filled(&mut random, &form, &dealt));
            }
            let source: alloc::vec::Vec<u8> =
                words.iter().flat_map(|word| word.to_le_bytes()).collect();
            let Ok(one) = crate::compile(&source, BASE, &pinned) else {
                continue;
            };
            // What the engine takes ends the pinned block before it.
            if one.source_bytes == 0 {
                continue;
            }
            let Ok(other) = crate::compile(&source[..one.source_bytes], BASE, &frame) else {
                continue;
            };
            if other.source_bytes != one.source_bytes {
                continue;
            }
            let mut initial = GuestState::default();
            for slot in &mut initial.x {
                *slot = u64::from(random.next()) << 32 | u64::from(random.next());
            }
            for register in &mut initial.v {
                *register = [
                    u64::from(random.next()) << 32 | u64::from(random.next()),
                    u64::from(random.next()) << 32 | u64::from(random.next()),
                ];
            }
            initial.x[named(1, &dealt) as usize] = middle + 0x3000;
            initial.x[named(19, &dealt) as usize] = middle;
            initial.x[named(20, &dealt) as usize] = middle - 0x3000;
            initial.x[named(5, &dealt) as usize] = 0x20;
            initial.x[named(17, &dealt) as usize] = 0x40;
            initial.sp = middle + 0x6000;
            initial.nzcv = random.below(16) << 28;
            memory.copy_from_slice(&start);
            let mut expected = initial.clone();
            let next = crate::x86_64::native::run(&other.native.x86_64_code, &mut expected);
            let kept = memory.clone();
            memory.copy_from_slice(&start);
            let mut state = initial.clone();
            let pc = crate::x86_64::native::run(&entered(&one.native.x86_64_code), &mut state);
            let source = alloc::format!("round {round}: {words:08x?}");
            assert_eq!(pc, next, "{source}");
            assert_eq!(state.x, expected.x, "{source}");
            assert_eq!(state.sp, expected.sp, "{source}");
            assert_eq!(state.nzcv, expected.nzcv, "{source}");
            assert_eq!(state.v, expected.v, "{source}");
            assert!(memory == kept, "{source}");
            compared += 1;
        }
        assert!(compared > 1000, "{compared} blocks compared");
    }

    /// Blocks whose runs keep guest registers in scratch registers
    /// ([`Span`]) against the same blocks without them, as data independent
    /// timing compiles them, through side exits and both kinds of checks.
    #[test]
    fn runs_compute_what_pinned_code_without_them_computes() {
        use crate::{
            GuestAddress, Options, TargetCpu,
            block::{BlockProfile, Features},
        };
        const BASE: GuestAddress = GuestAddress(0x4000_0000);
        let mut random = Random(0x6a09_e667_f3bc_c908);
        let mut memory = alloc::vec![0u8; 1 << 16];
        for byte in &mut memory {
            *byte = random.next() as u8;
        }
        let start = memory.clone();
        let middle = (memory.as_mut_ptr() as u64 + (1 << 15)) & !15;
        let mut differing = 0;
        for target in [TargetCpu::Baseline, TargetCpu::detect()] {
            for masked in [false, true] {
                let checks = super::super::MemoryChecks::system(true, false, true, masked);
                let options = |target| {
                    Options::new(BlockProfile::new(Features::SYSTEM_MODE), target)
                        .with_memory_checks(checks)
                };
                let runs = options(target);
                let without = options(target.with_data_independent_timing());
                for round in 0..3000 {
                    let words: alloc::vec::Vec<u32> = (0..1 + random.below(24))
                        .map(|_| instruction(&mut random, true))
                        .collect();
                    let source: alloc::vec::Vec<u8> =
                        words.iter().flat_map(|word| word.to_le_bytes()).collect();
                    let (Ok(one), Ok(other)) = (
                        crate::compile(&source, BASE, &runs),
                        crate::compile(&source, BASE, &without),
                    ) else {
                        continue;
                    };
                    assert_eq!(one.source_bytes, other.source_bytes);
                    differing += usize::from(one.native.x86_64_code != other.native.x86_64_code);
                    let mut initial = GuestState::default();
                    for slot in &mut initial.x {
                        *slot = u64::from(random.next()) << 32 | u64::from(random.next());
                    }
                    initial.x[19] = middle;
                    initial.x[20] = middle - 0x2000;
                    initial.x[INDEX as usize] = 0x20;
                    initial.sp = middle + 0x2000;
                    initial.nzcv = random.below(16) << 28;
                    memory.copy_from_slice(&start);
                    let mut expected = initial.clone();
                    let next = crate::x86_64::native::run(
                        &entered(&other.native.x86_64_code),
                        &mut expected,
                    );
                    let kept = memory.clone();
                    memory.copy_from_slice(&start);
                    let mut state = initial.clone();
                    let pc =
                        crate::x86_64::native::run(&entered(&one.native.x86_64_code), &mut state);
                    let source = alloc::format!("round {round}: {words:08x?}");
                    assert_eq!(pc, next, "{source}");
                    assert_eq!(state.x, expected.x, "{source}");
                    assert_eq!(state.sp, expected.sp, "{source}");
                    assert_eq!(state.nzcv, expected.nzcv, "{source}");
                    assert!(memory == kept, "{source}");
                }
            }
        }
        assert!(differing > 1000, "{differing} blocks kept locals");
    }

    /// The crypto forms, which random vector words seldom reach, with their
    /// register bits.
    const CRYPTO: &[(u32, u32)] = &[
        (0x5e00_0000, 0x001f_73ff),
        (0x5e28_0800, 0x0000_33ff),
        (0xce60_8000, 0x001f_0fff),
        (0xcec0_8000, 0x0000_07ff),
        (0xce40_c000, 0x001f_0fff),
        (0xce40_8000, 0x001f_0fff),
        (0xce00_0000, 0x009f_fc00 | 0x03ff),
        (0xce20_0000, 0x001f_7fff),
        (0x4e28_0800, 0x0000_33ff),
    ];

    /// Vector and crypto forms' scratch registers, on both of their tiers,
    /// are none of the pinned ones.
    #[test]
    fn vector_forms_compute_what_frame_code_computes() {
        use crate::{
            GuestAddress, Options, TargetCpu,
            block::{BlockProfile, Features},
        };
        const BASE: GuestAddress = GuestAddress(0x4000_0000);
        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        for target in [TargetCpu::Baseline, TargetCpu::detect()] {
            let pinned = Options::new(BlockProfile::new(Features::SYSTEM_MODE), target);
            let frame = Options::new(BlockProfile::ORDINARY, target);
            let mut compared = 0;
            for round in 0..40_000 {
                let word = match round % 2 {
                    0 => {
                        let (form, registers) = CRYPTO[round / 2 % CRYPTO.len()];
                        form | random.next() & registers
                    }
                    _ => random.next() | 0x0e00_0000,
                };
                let source = word.to_le_bytes();
                let Ok(one) = crate::compile(&source, BASE, &pinned) else {
                    continue;
                };
                if one.source_bytes == 0 {
                    continue;
                }
                let Ok(other) = crate::compile(&source, BASE, &frame) else {
                    continue;
                };
                if other.source_bytes != one.source_bytes {
                    continue;
                }
                let mut initial = GuestState::default();
                for slot in &mut initial.x {
                    *slot = u64::from(random.next()) << 32 | u64::from(random.next());
                }
                for register in &mut initial.v {
                    *register = [
                        u64::from(random.next()) << 32 | u64::from(random.next()),
                        u64::from(random.next()) << 32 | u64::from(random.next()),
                    ];
                }
                initial.sp = 0x1234_5678_9abc_def0;
                initial.nzcv = random.below(16) << 28;
                let mut expected = initial.clone();
                let next = crate::x86_64::native::run(&other.native.x86_64_code, &mut expected);
                let mut state = initial.clone();
                let pc = crate::x86_64::native::run(&entered(&one.native.x86_64_code), &mut state);
                let source = alloc::format!("{word:08x}");
                assert_eq!(pc, next, "{source}");
                assert_eq!(state.x, expected.x, "{source}");
                assert_eq!(state.sp, expected.sp, "{source}");
                assert_eq!(state.nzcv, expected.nzcv, "{source}");
                assert_eq!(state.v, expected.v, "{source}");
                compared += 1;
            }
            assert!(compared > 1000, "{compared} blocks compared");
        }
    }

    #[test]
    fn no_emitter_reaches_a_pinned_slot() {
        use crate::{
            GuestAddress, Options, TargetCpu,
            block::{BlockProfile, Features},
        };
        let mut random = Random(0x0123_4567_89ab_cdef);
        for target in [TargetCpu::Baseline, TargetCpu::detect()] {
            let options = Options::new(BlockProfile::new(Features::SYSTEM_MODE), target);
            for _ in 0..200_000 {
                // A debug assertion in the encoder panics at such a slot.
                let word = random.next();
                let _ = crate::compile(&word.to_le_bytes(), GuestAddress(0x4000), &options);
            }
        }
    }
}
