//! Scalar and vector Half FADD, FSUB, FABD, FMUL, FMULX, FNMUL and FDIV in
//! packed AVX2 lanes, for every finite input: subnormal inputs, zeros,
//! cancellation and tiny or subnormal results complete inline.
//!
//! VCVTPH2PS and VCVTPS2PD widen each group of four Half lanes exactly to
//! Double, where the sum or difference (at most 40 significant bits, the
//! smallest Half step being 2^-24 and the largest operand below 2^16) and
//! the product (22 bits, at least 2^-48) are exact and raise no host status.
//! The exact Double is then rounded once to Half in integer qword lanes under
//! FPCR's rounding mode, on the normal grid or, below 2^-14, on the
//! subnormal grid (a variable shift), so no double rounding occurs.
//!
//! A quotient is not exact in Double, but its one rounding there never
//! decides a Half result: an inexact quotient a/b of Half fields lies at
//! least 2^-23 (relative) from every number of 12 significant bits ((a -
//! m*b)/b with integer fields), far beyond the Double rounding error of
//! 2^-53, so its dropped bits are never all zero nor an exact tie, it is on
//! the same side of 2^-14 as the exact quotient, and the Half result,
//! tininess and inexact status are the exact quotient's in every rounding
//! mode. The quotient raises host status, so the caller's MXCSR is saved
//! and restored around it; a zero divisor gives an infinity or NaN, which
//! takes the helper as an overflow.
//!
//! A scalar form computes its one lane in the low group, whose other lanes
//! are set to 0 and 1.0 (the left and right inputs), which complete
//! exactly; the result is the lane alone, or, under FPCR.NEP, the lane in
//! Vn's other bits.
//!
//! An exact zero sum takes its sign from the rounding mode: -0 under
//! round-toward-minus for opposite signs. The host adds in nearest-even, so
//! under round-toward-minus both inputs and the result are negated around the
//! exact operation: -((-a) + (-b)) has the toward-minus zero sign and the same
//! nonzero value.
//!
//! Arm detects tininess before rounding with FPCR.AH zero, which is the
//! subnormal-grid test here: UFC for a tiny inexact lane. FZ16 (which
//! flushes tiny results and subnormal inputs) and AH (tininess after
//! rounding) send lanes with a subnormal input or a tiny nonzero result to
//! the helper, and so do NaN and infinite inputs, overflow and enabled traps.
use super::super::super::numerical::{Condition, branch, jump, resolve, test_control};
use super::super::super::{Host, packed};
use crate::x86_64::asm::{Alu, Code, Cond, Gpr, Map, Mem, Pp, Rm, ShiftOp, Size, Unary, Vex};
use crate::{
    GuestAddress, GuestState, TargetCpu,
    aarch64::{
        floating::{
            BinaryOperation, FPCR_AH, FPCR_EXCEPTION_ENABLE_MASK, FPCR_FZ16, FPCR_ROUNDING_MASK,
            FPCR_ROUNDING_SHIFT, INEXACT,
        },
        simd::arithmetic::{Inputs, Operand, Shape},
    },
    block::ExitKind,
};
use Gpr::{R8, R9, R10, Rax, Rcx, Rsp};
use alloc::vec::Vec;

/// The Half inputs, as `half::prepare_inputs` leaves them, and the result.
const LEFT: u8 = 0;
const RIGHT: u8 = 1;
const RESULT: u8 = 7;
/// Group scratch.
const T2: u8 = 2;
const T3: u8 = 3;
const T4: u8 = 4;
const T5: u8 = 5;
const T6: u8 = 6;
const T8: u8 = 8;
const T9: u8 = 9;
const T10: u8 = 10;
const T11: u8 = 11;
/// The sign flip under round-toward-minus, and the rounding policies:
/// all ones in each qword when FPCR selects nearest, toward plus, toward
/// minus.
const FLIP: u8 = 12;
const NEAREST: u8 = 13;
const POSITIVE: u8 = 14;
const NEGATIVE: u8 = 15;

const FPSR_UNDERFLOW: u32 = 1 << 3;
const HALF_EXPONENT_FIELD: u16 = 0x7c00;
const HALF_MAGNITUDE: u16 = 0x7fff;
/// Double biased exponent of 2^-15: below this plus one a result is tiny.
const SUBNORMAL_LIMIT: u32 = 1008;
/// The normal-grid shift (52 - 10) plus `SUBNORMAL_LIMIT`.
const SHIFT_BASE: u32 = 1050;
/// Every shift past 54 rounds a value below 2^53 like 54 does.
const SHIFT_LIMIT: u32 = 54;
const LARGEST_FINITE: u32 = 0x7bff;
const DOUBLE_FRACTION_BITS: u8 = 52;
const SIGN_POSITION: u8 = 63;
const HALF_SIGN_POSITION: u8 = 15;
const UPPER_QWORD_BYTES: u8 = 8;
/// R10's bits for the tiny nonzero lanes and, shifted, the overflowed ones.
const TINY_LANES: i32 = 0xf;
const OVERFLOW_SHIFT: u8 = 8;
/// The low Half lane of a qword, and 1.0 in its other three lanes.
const HALF_LANE: i32 = 0xffff;
const ONES_ABOVE: u64 = 0x3c00_3c00_3c00_0000;
/// The aligned stack slot holding the caller's MXCSR around a quotient.
const MXCSR_SLOT_BYTES: i8 = 16;

const fn vex(map: Map, pp: Pp, w: bool, l: bool) -> Vex {
    Vex { map, pp, w, l }
}

/// `op ymm reg, ymm vvvv, ymm rm` (`l`) or the xmm form.
fn v(code: &mut Code, prefix: Vex, opcode: u8, reg: u8, vvvv: u8, rm: u8) {
    code.asm().vex(prefix, opcode, reg, vvvv, Rm::Xmm(rm));
}

fn ymm(code: &mut Code, map: Map, opcode: u8, w: bool, reg: u8, vvvv: u8, rm: u8) {
    v(code, vex(map, Pp::P66, w, true), opcode, reg, vvvv, rm);
}

fn and(code: &mut Code, d: u8, a: u8, b: u8) {
    ymm(code, Map::M0f, 0xdb, false, d, a, b);
}
/// `d = !a & b`.
fn and_not(code: &mut Code, d: u8, a: u8, b: u8) {
    ymm(code, Map::M0f, 0xdf, false, d, a, b);
}
fn or(code: &mut Code, d: u8, a: u8, b: u8) {
    ymm(code, Map::M0f, 0xeb, false, d, a, b);
}
fn xor(code: &mut Code, d: u8, a: u8, b: u8) {
    ymm(code, Map::M0f, 0xef, false, d, a, b);
}
fn add_q(code: &mut Code, d: u8, a: u8, b: u8) {
    ymm(code, Map::M0f, 0xd4, false, d, a, b);
}
fn sub_q(code: &mut Code, d: u8, a: u8, b: u8) {
    ymm(code, Map::M0f, 0xfb, false, d, a, b);
}
fn equal_q(code: &mut Code, d: u8, a: u8, b: u8) {
    ymm(code, Map::M0f38, 0x29, false, d, a, b);
}
/// `d = a > b`, signed.
fn greater_q(code: &mut Code, d: u8, a: u8, b: u8) {
    ymm(code, Map::M0f38, 0x37, false, d, a, b);
}
/// `d = a >> b` per qword (VPSRLVQ).
fn right_variable(code: &mut Code, d: u8, a: u8, b: u8) {
    ymm(code, Map::M0f38, 0x45, true, d, a, b);
}
/// `d = a << b` per qword (VPSLLVQ).
fn left_variable(code: &mut Code, d: u8, a: u8, b: u8) {
    ymm(code, Map::M0f38, 0x47, true, d, a, b);
}
fn ones(code: &mut Code, d: u8) {
    equal_q(code, d, d, d);
}
fn zero(code: &mut Code, d: u8) {
    xor(code, d, d, d);
}
/// VPSRLQ (`group` 2) or VPSLLQ (6) by `amount`.
fn shift_q(code: &mut Code, group: u8, d: u8, a: u8, amount: u8) {
    ymm(code, Map::M0f, 0x73, false, group, d, a);
    code.asm().data(&[amount]);
}
fn right_q(code: &mut Code, d: u8, a: u8, amount: u8) {
    shift_q(code, 2, d, a, amount);
}
fn left_q(code: &mut Code, d: u8, a: u8, amount: u8) {
    shift_q(code, 6, d, a, amount);
}
/// `d = mask ? b : a` per qword (VBLENDVPD).
fn blend(code: &mut Code, d: u8, a: u8, b: u8, mask: u8) {
    ymm(code, Map::M0f3a, 0x4b, false, d, a, b);
    code.asm().data(&[mask << 4]);
}
/// Every qword of `d` = the dword `value`, zero-extended. Clobbers EAX.
fn broadcast(code: &mut Code, d: u8, value: u32) {
    code.asm().mov_imm32(Rax, value);
    broadcast_rax(code, d);
}
/// Every qword of `d` = RAX.
fn broadcast_rax(code: &mut Code, d: u8) {
    // VMOVQ xmm, r64; VPBROADCASTQ ymm, xmm.
    code.asm().vex(
        vex(Map::M0f, Pp::P66, true, false),
        0x6e,
        d,
        0,
        Rm::Reg(Rax),
    );
    ymm(code, Map::M0f38, 0x59, false, d, 0, d);
}
/// EAX = the sign bits of the qwords of `source` (VMOVMSKPD).
fn lanes(code: &mut Code, source: u8) {
    code.asm().vex(
        vex(Map::M0f, Pp::P66, false, true),
        0x50,
        0,
        0,
        Rm::Xmm(source),
    );
}

/// Whether the packed recipe completes every form on `target`.
pub(super) fn supports(target: TargetCpu) -> bool {
    target.supports(TargetCpu::Avx2) && target.supports(TargetCpu::F16c)
}

/// The FPCR policies in NEAREST, POSITIVE, NEGATIVE and FLIP. Clobbers RAX
/// and RCX.
fn policies(code: &mut Code, flip: bool) {
    code.asm().mov_load(
        Size::Dword,
        Rcx,
        super::super::super::numerical::state(core::mem::offset_of!(GuestState, fpcr)),
    );
    let mut asm = code.asm();
    asm.shift_imm(ShiftOp::Shr, Size::Dword, Rcx, FPCR_ROUNDING_SHIFT as u8);
    asm.alu_imm8(Alu::And, Size::Dword, Rcx, FPCR_ROUNDING_MASK as i8);
    for (mode, register) in [(0, NEAREST), (1, POSITIVE), (2, NEGATIVE)] {
        let mut asm = code.asm();
        asm.alu(Alu::Xor, Size::Dword, Rax, Rax);
        asm.alu_imm8(Alu::Cmp, Size::Dword, Rcx, mode);
        asm.setcc(Cond::E, Rax);
        asm.unary(Unary::Neg, Size::Qword, Rax);
        broadcast_rax(code, register);
    }
    if flip {
        // RAX is still the toward-minus policy.
        code.asm()
            .shift_imm(ShiftOp::Shl, Size::Qword, Rax, SIGN_POSITION);
        broadcast_rax(code, FLIP);
    }
}

/// Branch to `helper` when an active lane of LEFT or RIGHT is infinite or
/// NaN, or, under FZ16 or AH, subnormal. Clobbers XMM2 to XMM6 and EAX.
fn guard_inputs(code: &mut Code, active: u32, helper: &mut Vec<usize>) {
    let word = |value: u16| u32::from(value) * 0x0001_0001;
    super::super::super::precision::dword_constant(code, T6, word(HALF_EXPONENT_FIELD));
    for (input, scratch) in [(LEFT, T2), (RIGHT, T3)] {
        packed(code, Host::MoveAligned, scratch, input);
        packed(code, Host::And, scratch, T6);
        packed(code, Host::EqualWord, scratch, T6);
    }
    packed(code, Host::Or, T2, T3);
    Host::ByteSignMask.register(code, 0, T2, TargetCpu::Baseline);
    code.asm()
        .alu_imm32(Alu::And, Size::Dword, Rax, active as i32);
    helper.push(branch(code, Condition::NotZero));
    test_control(code, FPCR_FZ16 | FPCR_AH);
    let ordinary = branch(code, Condition::Zero);
    // Subnormal: exponent field zero, magnitude not.
    super::super::super::precision::dword_constant(code, T5, word(HALF_MAGNITUDE));
    packed(code, Host::Xor, T4, T4);
    for (input, scratch) in [(LEFT, T2), (RIGHT, T3)] {
        packed(code, Host::MoveAligned, scratch, input);
        packed(code, Host::And, scratch, T5);
        packed(code, Host::EqualWord, scratch, T4);
        packed(code, Host::MoveAligned, T6, input);
        super::super::super::precision::dword_constant(code, T5, word(HALF_EXPONENT_FIELD));
        packed(code, Host::And, T6, T5);
        packed(code, Host::EqualWord, T6, T4);
        packed(code, Host::AndNot, scratch, T6);
        super::super::super::precision::dword_constant(code, T5, word(HALF_MAGNITUDE));
    }
    packed(code, Host::Or, T2, T3);
    Host::ByteSignMask.register(code, 0, T2, TargetCpu::Baseline);
    code.asm()
        .alu_imm32(Alu::And, Size::Dword, Rax, active as i32);
    helper.push(branch(code, Condition::NotZero));
    let end = code.len();
    resolve(code, ordinary, end);
}

/// One group of four lanes: the low (`upper` false) or high qword of LEFT
/// and RIGHT, its Half results in the low qword of RESULT (or, for the high
/// group, the high qword). R8D, R9D and R10D collect the lanes that are
/// inexact, underflow, and tiny and nonzero; R10D also the lanes that
/// overflow, above [`OVERFLOW_SHIFT`]. It branches nowhere.
fn group(code: &mut Code, operation: BinaryOperation, upper: bool, flip: bool) {
    // Widen exactly: VCVTPH2PS xmm, VCVTPS2PD ymm.
    for (input, wide) in [(LEFT, T8), (RIGHT, T9)] {
        let source = if upper {
            // VPSRLDQ xmm, xmm, 8.
            v(
                code,
                vex(Map::M0f, Pp::P66, false, false),
                0x73,
                3,
                wide,
                input,
            );
            code.asm().data(&[UPPER_QWORD_BYTES]);
            wide
        } else {
            input
        };
        v(
            code,
            vex(Map::M0f38, Pp::P66, false, false),
            0x13,
            wide,
            0,
            source,
        );
        v(
            code,
            vex(Map::M0f, Pp::None, false, true),
            0x5a,
            wide,
            0,
            wide,
        );
    }
    if flip {
        xor(code, T8, T8, FLIP);
        xor(code, T9, T9, FLIP);
    }
    let opcode = match operation {
        BinaryOperation::Add => 0x58,
        BinaryOperation::Subtract | BinaryOperation::AbsoluteDifference => 0x5c,
        BinaryOperation::Divide => 0x5e,
        _ => 0x59,
    };
    ymm(code, Map::M0f, opcode, false, T10, T8, T9);
    if flip {
        xor(code, T10, T10, FLIP);
    }
    // T11 = negative lanes; T8 = magnitude; T10 = zero lanes; T2 = E - 1.
    zero(code, T9);
    greater_q(code, T11, T9, T10);
    left_q(code, T8, T10, 1);
    right_q(code, T8, T8, 1);
    equal_q(code, T10, T8, T9);
    right_q(code, T2, T8, DOUBLE_FRACTION_BITS);
    ones(code, T9);
    add_q(code, T2, T2, T9);
    // T3 = tiny (or zero) lanes; T2 = min(E - 1, 1008).
    broadcast(code, T9, SUBNORMAL_LIMIT);
    greater_q(code, T3, T9, T2);
    blend(code, T2, T9, T2, T3);
    and_not(code, T4, T10, T3);
    lanes(code, T4);
    code.asm().alu(Alu::Or, Size::Dword, R10, Rax);
    // T5 = the shift: min(1050 - T2, 54).
    broadcast(code, T9, SHIFT_BASE);
    sub_q(code, T5, T9, T2);
    broadcast(code, T9, SHIFT_LIMIT);
    greater_q(code, T6, T5, T9);
    blend(code, T5, T5, T9, T6);
    // T2 = V = magnitude - (T2 << 52), zero for a zero lane.
    left_q(code, T2, T2, DOUBLE_FRACTION_BITS);
    sub_q(code, T2, T8, T2);
    and_not(code, T2, T10, T2);
    // T8 = the dropped-bit mask; T10 = the dropped bits.
    ones(code, T9);
    left_variable(code, T6, T9, T5);
    xor(code, T8, T6, T9);
    and(code, T10, T2, T8);
    zero(code, T9);
    equal_q(code, T4, T10, T9);
    lanes(code, T4);
    let mut asm = code.asm();
    asm.unary(Unary::Not, Size::Dword, Rax);
    asm.alu_imm8(Alu::And, Size::Dword, Rax, 0xf);
    asm.alu(Alu::Or, Size::Dword, R8, Rax);
    and_not(code, T4, T4, T3);
    lanes(code, T4);
    code.asm().alu(Alu::Or, Size::Dword, R9, Rax);
    // Increment: (half mask + kept parity) when nearest, the mask when
    // toward plus and positive or toward minus and negative.
    right_variable(code, T4, T2, T5);
    left_q(code, T4, T4, SIGN_POSITION);
    right_q(code, T4, T4, SIGN_POSITION);
    right_q(code, T10, T8, 1);
    add_q(code, T10, T10, T4);
    and(code, T10, T10, NEAREST);
    and_not(code, T4, T11, T8);
    and(code, T4, T4, POSITIVE);
    or(code, T10, T10, T4);
    and(code, T4, T11, T8);
    and(code, T4, T4, NEGATIVE);
    or(code, T10, T10, T4);
    add_q(code, T2, T2, T10);
    right_variable(code, T2, T2, T5);
    // Overflow (to infinity or the largest finite), and an infinite or NaN
    // quotient, takes the helper.
    broadcast(code, T9, LARGEST_FINITE);
    greater_q(code, T4, T2, T9);
    lanes(code, T4);
    let mut asm = code.asm();
    asm.shift_imm(ShiftOp::Shl, Size::Dword, Rax, OVERFLOW_SHIFT);
    asm.alu(Alu::Or, Size::Dword, R10, Rax);
    // The sign, transformed as FNMUL and FABD transform the rounded result.
    match operation {
        BinaryOperation::AbsoluteDifference => {}
        BinaryOperation::NegatedMultiply => {
            ones(code, T9);
            xor(code, T11, T11, T9);
        }
        _ => {}
    }
    if operation != BinaryOperation::AbsoluteDifference {
        right_q(code, T11, T11, SIGN_POSITION);
        left_q(code, T11, T11, HALF_SIGN_POSITION);
        or(code, T2, T2, T11);
    }
    // Four qwords below 2^16 to four words: VEXTRACTI128, VPACKUSDW twice.
    code.asm().vex(
        vex(Map::M0f3a, Pp::P66, false, true),
        0x39,
        T2,
        0,
        Rm::Xmm(T4),
    );
    code.asm().data(&[1]);
    let pack = vex(Map::M0f38, Pp::P66, false, false);
    v(code, pack, 0x2b, T2, T2, T4);
    if upper {
        v(code, pack, 0x2b, T2, T2, T2);
        // VPUNPCKLQDQ xmm, xmm, xmm.
        v(
            code,
            vex(Map::M0f, Pp::P66, false, false),
            0x6c,
            RESULT,
            RESULT,
            T2,
        );
    } else {
        v(code, pack, 0x2b, RESULT, T2, T2);
    }
}

/// Complete `operand` (a [`supports`] form) after `half::prepare_inputs`,
/// write its Half lanes to the destination and the fresh status to FPSR.
/// Every guard fails before a guest effect, into the exit at `pc`.
pub(super) fn emit(code: &mut Code, operand: &Operand, pc: GuestAddress) {
    let mut helper = Vec::new();
    test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
    helper.push(branch(code, Condition::NotZero));
    super::prepare_inputs(code, operand);
    let wide = operand.shape == Shape::Vector128;
    let scalar = operand.shape == Shape::Scalar;
    if scalar {
        // Lane 0 alone; 0 and 1.0 in the others.
        let mut asm = code.asm();
        asm.movd_from_xmm(Rax, LEFT);
        asm.alu_imm32(Alu::And, Size::Dword, Rax, HALF_LANE);
        asm.movq_to_xmm(LEFT, Rax);
        asm.movd_from_xmm(Rax, RIGHT);
        asm.alu_imm32(Alu::And, Size::Dword, Rax, HALF_LANE);
        asm.mov_imm64(Rcx, ONES_ABOVE);
        asm.alu(Alu::Or, Size::Qword, Rax, Rcx);
        asm.movq_to_xmm(RIGHT, Rax);
    }
    guard_inputs(code, if wide { 0xffff } else { 0xff }, &mut helper);
    let flip = matches!(
        operand.operation,
        BinaryOperation::Add | BinaryOperation::Subtract | BinaryOperation::AbsoluteDifference
    );
    policies(code, flip);
    let mut asm = code.asm();
    asm.alu(Alu::Xor, Size::Dword, R8, R8);
    asm.alu(Alu::Xor, Size::Dword, R9, R9);
    asm.alu(Alu::Xor, Size::Dword, R10, R10);
    let divide = operand.operation == BinaryOperation::Divide;
    if divide {
        let mut asm = code.asm();
        asm.alu_imm8(Alu::Sub, Size::Qword, Rsp, MXCSR_SLOT_BYTES);
        asm.stmxcsr(Mem::base(Rsp, 0));
    }
    group(code, operand.operation, false, flip);
    if wide {
        group(code, operand.operation, true, flip);
    }
    if divide {
        let mut asm = code.asm();
        asm.ldmxcsr(Mem::base(Rsp, 0));
        asm.alu_imm8(Alu::Add, Size::Qword, Rsp, MXCSR_SLOT_BYTES);
    }
    code.asm().vzeroupper();
    code.asm()
        .test_imm(Size::Dword, R10, TINY_LANES << OVERFLOW_SHIFT);
    helper.push(branch(code, Condition::NotZero));
    // A tiny nonzero lane under FZ16 or AH takes the helper.
    code.asm().test_imm(Size::Dword, R10, TINY_LANES);
    let none = branch(code, Condition::Zero);
    test_control(code, FPCR_FZ16 | FPCR_AH);
    helper.push(branch(code, Condition::NotZero));
    let end = code.len();
    resolve(code, none, end);
    // Fresh IXC and UFC.
    let mut asm = code.asm();
    asm.alu(Alu::Xor, Size::Dword, Rax, Rax);
    asm.test(Size::Dword, R8, R8);
    asm.setcc(Cond::Ne, Rax);
    asm.shift_imm(
        ShiftOp::Shl,
        Size::Dword,
        Rax,
        INEXACT.trailing_zeros() as u8,
    );
    asm.alu(Alu::Xor, Size::Dword, Rcx, Rcx);
    asm.test(Size::Dword, R9, R9);
    asm.setcc(Cond::Ne, Rcx);
    asm.shift_imm(
        ShiftOp::Shl,
        Size::Dword,
        Rcx,
        FPSR_UNDERFLOW.trailing_zeros() as u8,
    );
    asm.alu(Alu::Or, Size::Dword, Rax, Rcx);
    asm.alu(
        Alu::Or,
        Size::Dword,
        super::super::super::numerical::state(core::mem::offset_of!(GuestState, fpsr)),
        Rax,
    );
    if scalar {
        let mut asm = code.asm();
        asm.movd_from_xmm(Rax, RESULT);
        asm.alu_imm32(Alu::And, Size::Dword, Rax, HALF_LANE);
        asm.movq_to_xmm(LEFT, Rax);
        // FPCR.NEP keeps Vn's other bits (`half::merge_scalar`).
        let left = match operand.inputs {
            Inputs::Elementwise { left, .. } | Inputs::Indexed { left, .. } => Some(left),
            Inputs::Pairwise { .. } | Inputs::PairwiseScalar { .. } => None,
        };
        if let Some(left) = left {
            super::super::super::read_vector(code, left, super::super::super::half::ORIGINAL_LEFT);
            super::super::super::half::merge_scalar(code);
        }
    } else if wide {
        packed(code, Host::MoveAligned, LEFT, RESULT);
    } else {
        Host::MoveQword.register(code, LEFT, RESULT, TargetCpu::Baseline);
    }
    super::super::super::commit(code, operand.destination, LEFT);
    let done = jump(code);
    let helper_start = code.len();
    code.asm().vzeroupper();
    crate::x86_64::block::direct_exit(code, pc, ExitKind::FloatingPoint);
    let finished = code.len();
    for patch in helper {
        resolve(code, patch, helper_start);
    }
    resolve(code, done, finished);
}
