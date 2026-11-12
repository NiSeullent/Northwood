//! Half element-wise lanes with a signed zero input: their results are exact
//! and raise no status, so they complete from the inputs without the helper.
use super::super::super::super::isa::ShiftOpcode;
use super::super::super::numerical::{
    Condition, branch, read_numerical_vector, vector_normal_or_zero_range,
};
use super::super::super::{Host, half as lanes, packed, packed_imm};
use crate::x86_64::asm::{Alu, Code, Cond, Gpr, ShiftOp, Size, Unary};
use crate::x86_64::simd::encode::guest;
use crate::{
    TargetCpu,
    aarch64::{
        floating::{BinaryOperation, FPCR_ROUNDING_MASK, FPCR_ROUNDING_SHIFT, Format},
        simd::arithmetic::{Inputs, Operand, Shape},
    },
};
use alloc::vec::Vec;

/// Vector element-wise sums, differences, products and quotients whose lanes have a
/// zero input: the result is a signed zero or the other input, exact and
/// without status under every FPCR (a quotient needs a nonzero divisor).
pub(super) fn supported(operand: &Operand) -> bool {
    operand.shape != Shape::Scalar
        && matches!(operand.inputs, Inputs::Elementwise { .. })
        && matches!(
            operand.operation,
            BinaryOperation::Add
                | BinaryOperation::Subtract
                | BinaryOperation::Multiply
                | BinaryOperation::MultiplyExtended
                | BinaryOperation::NegatedMultiply
                | BinaryOperation::Divide
        )
}

const ZERO_LEFT_INPUT: u8 = 1;
const ZERO_LEFT: u8 = 2;
const ZERO_RIGHT: u8 = 3;
const ZERO_ANY: u8 = 4;
const ZERO_SCRATCH: u8 = 5;
const ZERO_RIGHT_INPUT: u8 = 6;
const ZERO_TEMPORARY: u8 = 7;
/// Substitute inputs 1.0 and 0.5: their sum, difference, product and
/// quotient are exact normal Halves.
const SUBSTITUTE_LEFT: u16 = 0x3c00;
const SUBSTITUTE_RIGHT: u16 = 0x3800;
/// FPCR.RMode toward minus infinity.
const ROUND_TOWARD_MINUS: i8 = 2;

/// `value` in every word of `xmm`. Clobbers EAX.
fn word_constant(code: &mut Code, xmm: u8, value: u16) {
    let mut asm = code.asm();
    asm.mov_imm32(Gpr::Rax, u32::from(value) * 0x0001_0001);
    asm.movd_to_xmm(xmm, Gpr::Rax);
    packed_imm(code, Host::ShuffleDword, xmm, xmm, 0);
}

/// The active lanes of `shape` as all-ones words in `xmm`.
fn active_lanes(code: &mut Code, shape: Shape, xmm: u8) {
    packed(code, Host::EqualWord, xmm, xmm);
    let inactive_bytes = match shape {
        Shape::Vector128 => 0,
        Shape::Vector64 => 8,
        _ => 14,
    };
    if inactive_bytes != 0 {
        ShiftOpcode::RightBytes.emit(code, xmm, inactive_bytes);
    }
}

/// Word masks: `ZERO_LEFT` and `ZERO_RIGHT` for the zero lanes of `left`
/// and `right`, `ZERO_ANY` for the active lanes with either.
fn zero_masks(code: &mut Code, shape: Shape, left: u8, right: u8) {
    packed(code, Host::Xor, ZERO_SCRATCH, ZERO_SCRATCH);
    for (mask, source) in [(ZERO_LEFT, left), (ZERO_RIGHT, right)] {
        packed(code, Host::MoveAligned, mask, source);
        ShiftOpcode::LeftWord.emit(code, mask, 1);
        packed(code, Host::EqualWord, mask, ZERO_SCRATCH);
    }
    active_lanes(code, shape, ZERO_ANY);
    packed(code, Host::MoveAligned, ZERO_SCRATCH, ZERO_LEFT);
    packed(code, Host::Or, ZERO_SCRATCH, ZERO_RIGHT);
    packed(code, Host::And, ZERO_ANY, ZERO_SCRATCH);
}

/// Before the lane recipe: active lanes with a zero input compute 1.0 op 0.5
/// instead, which is exact and normal; [`merge`] replaces their
/// results. A zero divisor takes the helper.
pub(super) fn substitute(code: &mut Code, operand: &Operand, helper: &mut Vec<usize>) {
    // The other input of a zero lane must itself be normal or zero.
    for source in [lanes::LEFT, lanes::RIGHT] {
        vector_normal_or_zero_range(code, Format::Half, operand.shape, source, lanes::RESULT);
        helper.push(branch(code, Condition::NotZero));
    }
    zero_masks(code, operand.shape, lanes::LEFT, lanes::RIGHT);
    if operand.operation == BinaryOperation::Divide {
        active_lanes(code, operand.shape, ZERO_SCRATCH);
        packed(code, Host::And, ZERO_SCRATCH, ZERO_RIGHT);
        Host::ByteSignMask.register(code, 0, ZERO_SCRATCH, TargetCpu::Baseline);
        code.asm().test(Size::Dword, Gpr::Rax, Gpr::Rax);
        helper.push(branch(code, Condition::NotZero));
    }
    for (source, value) in [
        (lanes::LEFT, SUBSTITUTE_LEFT),
        (lanes::RIGHT, SUBSTITUTE_RIGHT),
    ] {
        word_constant(code, ZERO_SCRATCH, value);
        packed(code, Host::And, ZERO_SCRATCH, ZERO_ANY);
        packed(code, Host::MoveAligned, ZERO_LEFT, ZERO_ANY);
        packed(code, Host::AndNot, ZERO_LEFT, source);
        packed(code, Host::Or, ZERO_LEFT, ZERO_SCRATCH);
        packed(code, Host::MoveAligned, source, ZERO_LEFT);
    }
}

/// After the lane recipe, with the result in `lanes::LEFT` and the inputs
/// still in the guest registers: the zero lanes' exact results. A product or
/// quotient is the zero with the XOR of the signs (negated for FNMUL). A sum
/// with one zero input is the other input; two zeros sum to -0 under RM when
/// either is -0, else only when both are. Preserves R10D and XMM0 outside
/// the zero lanes.
pub(super) fn merge(code: &mut Code, operand: &Operand) {
    let Inputs::Elementwise { left, right } = operand.inputs else {
        unreachable!("element-wise zero lanes")
    };
    read_numerical_vector(code, operand.shape, left, ZERO_LEFT_INPUT);
    read_numerical_vector(code, operand.shape, right, ZERO_RIGHT_INPUT);
    let sign = Format::Half.sign_mask() as u16;
    if operand.operation == BinaryOperation::Subtract {
        word_constant(code, ZERO_SCRATCH, sign);
        packed(code, Host::Xor, ZERO_RIGHT_INPUT, ZERO_SCRATCH);
    }
    zero_masks(code, operand.shape, ZERO_LEFT_INPUT, ZERO_RIGHT_INPUT);
    // ZERO_SCRATCH becomes the zero lanes' results.
    packed(code, Host::MoveAligned, ZERO_SCRATCH, ZERO_LEFT_INPUT);
    packed(code, Host::Xor, ZERO_SCRATCH, ZERO_RIGHT_INPUT);
    if matches!(
        operand.operation,
        BinaryOperation::Add | BinaryOperation::Subtract
    ) {
        // Two zeros: L & R, or L | R = (L & R) | (L ^ R) under RM.
        let mut asm = code.asm();
        asm.mov_load(
            Size::Dword,
            Gpr::Rax,
            guest(core::mem::offset_of!(crate::GuestState, fpcr) as u32),
        );
        asm.shift_imm(
            ShiftOp::Shr,
            Size::Dword,
            Gpr::Rax,
            FPCR_ROUNDING_SHIFT as u8,
        );
        asm.alu_imm8(Alu::And, Size::Dword, Gpr::Rax, FPCR_ROUNDING_MASK as i8);
        asm.alu_imm8(Alu::Cmp, Size::Dword, Gpr::Rax, ROUND_TOWARD_MINUS);
        asm.setcc(Cond::E, Gpr::Rax);
        asm.movzx(Size::Dword, Gpr::Rax, Size::Byte, Gpr::Rax);
        asm.unary(Unary::Neg, Size::Dword, Gpr::Rax);
        asm.movd_to_xmm(ZERO_TEMPORARY, Gpr::Rax);
        packed_imm(code, Host::ShuffleDword, ZERO_TEMPORARY, ZERO_TEMPORARY, 0);
        packed(code, Host::And, ZERO_SCRATCH, ZERO_TEMPORARY);
        packed(code, Host::MoveAligned, ZERO_TEMPORARY, ZERO_LEFT_INPUT);
        packed(code, Host::And, ZERO_TEMPORARY, ZERO_RIGHT_INPUT);
        packed(code, Host::Or, ZERO_SCRATCH, ZERO_TEMPORARY);
        // Lanes with both zero keep that; with one zero, the other input.
        packed(code, Host::MoveAligned, ZERO_TEMPORARY, ZERO_LEFT);
        packed(code, Host::And, ZERO_TEMPORARY, ZERO_RIGHT);
        packed(code, Host::And, ZERO_SCRATCH, ZERO_TEMPORARY);
        packed(code, Host::And, ZERO_RIGHT_INPUT, ZERO_LEFT);
        packed(code, Host::And, ZERO_LEFT_INPUT, ZERO_RIGHT);
        packed(code, Host::Or, ZERO_RIGHT_INPUT, ZERO_LEFT_INPUT);
        packed(code, Host::AndNot, ZERO_TEMPORARY, ZERO_RIGHT_INPUT);
        packed(code, Host::Or, ZERO_SCRATCH, ZERO_TEMPORARY);
    } else {
        word_constant(code, ZERO_LEFT, sign);
        packed(code, Host::And, ZERO_SCRATCH, ZERO_LEFT);
        if operand.operation == BinaryOperation::NegatedMultiply {
            packed(code, Host::Xor, ZERO_SCRATCH, ZERO_LEFT);
        }
    }
    packed(code, Host::And, ZERO_SCRATCH, ZERO_ANY);
    packed(code, Host::AndNot, ZERO_ANY, lanes::LEFT);
    packed(code, Host::Or, ZERO_ANY, ZERO_SCRATCH);
    packed(code, Host::MoveAligned, lanes::LEFT, ZERO_ANY);
}
