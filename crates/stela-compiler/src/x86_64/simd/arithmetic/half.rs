//! Half-precision arithmetic through exact widened intermediates, or one
//! AVX512-FP16 instruction.
use crate::x86_64::asm::{
    Code, Gpr,
    vector_extensions::{self, HalfOpcode},
};
use alloc::{vec, vec::Vec};
mod packed;
mod product;
mod zero;

use super::super::super::isa::ShiftOpcode;
use super::super::numerical::{
    ArithmeticOpcode, Condition, branch, emit_opcode_to, prepare_vector_input_range,
    read_numerical_right, read_numerical_vector, test_control, vector_exponent_range,
};
use super::super::{Host, commit, half as lanes, packed, packed_imm, read_vector};
use crate::{
    GuestAddress, TargetCpu,
    aarch64::{
        floating::{BinaryOperation, FPCR_EXCEPTION_ENABLE_MASK, Format},
        simd::arithmetic::{Inputs, Operand, Shape},
    },
};

const LEFT_ODD: u8 = 6;
const RIGHT_ODD: u8 = 3;
const EVEN_HALF_WORDS: u8 = 0x88;
const SECOND_HALF_WORD: u8 = 0x55;
/// The sign constant of the AVX512-FP16 path; free after its range checks.
const NATIVE_SIGN: u8 = 2;
/// Half lanes in the low qword.
const QWORD_LANES: u8 = 4;
/// The aligned stack slot holding the caller MXCSR around a Half quotient.
const MXCSR_SLOT_BYTES: i8 = 16;

fn compact_even_words(code: &mut Code, source: u8) {
    packed_imm(code, Host::ShuffleLowWord, source, source, EVEN_HALF_WORDS);
}

/// Normalize adjacent Vn pairs followed by Vm pairs into two Half vectors.
/// Q0 inputs were loaded with MOVQ; only the assembled active words are guarded.
fn prepare_pairs(code: &mut Code, shape: Shape) {
    packed(code, Host::MoveAligned, LEFT_ODD, lanes::LEFT);
    if shape == Shape::Vector128 {
        packed(code, Host::MoveAligned, RIGHT_ODD, lanes::RIGHT);
        // Sign extension makes every raw Half field exactly PACKSSDW-safe.
        for source in [lanes::LEFT, lanes::RIGHT] {
            ShiftOpcode::LeftDword.emit(code, source, Format::Half.bits() as u8);
            ShiftOpcode::ArithmeticDword.emit(code, source, Format::Half.bits() as u8);
        }
        packed(code, Host::PackSignedDwordToWord, lanes::LEFT, lanes::RIGHT);
        for source in [LEFT_ODD, RIGHT_ODD] {
            ShiftOpcode::ArithmeticDword.emit(code, source, Format::Half.bits() as u8);
        }
        packed(code, Host::PackSignedDwordToWord, LEFT_ODD, RIGHT_ODD);
        packed(code, Host::MoveAligned, lanes::RIGHT, LEFT_ODD);
        return;
    }
    ShiftOpcode::RightDword.emit(code, LEFT_ODD, Format::Half.bits() as u8);
    packed(code, Host::MoveAligned, RIGHT_ODD, lanes::RIGHT);
    ShiftOpcode::RightDword.emit(code, RIGHT_ODD, Format::Half.bits() as u8);
    for source in [lanes::LEFT, lanes::RIGHT, LEFT_ODD, RIGHT_ODD] {
        compact_even_words(code, source);
    }
    packed(code, Host::UnpackLowDword, lanes::LEFT, lanes::RIGHT);
    packed(code, Host::UnpackLowDword, LEFT_ODD, RIGHT_ODD);
    packed(code, Host::MoveAligned, lanes::RIGHT, LEFT_ODD);
}

fn prepare_inputs(code: &mut Code, operand: &Operand) {
    match operand.inputs {
        Inputs::Elementwise { left, right } => {
            read_numerical_vector(code, operand.shape, left, lanes::LEFT);
            read_numerical_vector(code, operand.shape, right, lanes::RIGHT);
        }
        Inputs::Indexed {
            left,
            right,
            right_lane,
        } => {
            read_numerical_vector(code, operand.shape, left, lanes::LEFT);
            // Read all of Vm: a 64-bit form may select its upper qword.
            read_numerical_right(code, Format::Half, operand.shape, right, Some(right_lane));
        }
        Inputs::Pairwise { left, right } => {
            read_numerical_vector(code, operand.shape, left, lanes::LEFT);
            read_numerical_vector(code, operand.shape, right, lanes::RIGHT);
            prepare_pairs(code, operand.shape);
        }
        Inputs::PairwiseScalar { source } => {
            read_vector(code, source, lanes::LEFT);
            packed_imm(
                code,
                Host::ShuffleLowWord,
                lanes::RIGHT,
                lanes::LEFT,
                SECOND_HALF_WORD,
            );
        }
    }
    if operand.shape == Shape::Scalar && !matches!(operand.inputs, Inputs::PairwiseScalar { .. }) {
        packed(code, Host::MoveAligned, lanes::ORIGINAL_LEFT, lanes::LEFT);
    }
}

fn operation(operation: BinaryOperation) -> (ArithmeticOpcode, lanes::ResultSign) {
    match operation {
        BinaryOperation::Add => (ArithmeticOpcode::Add, lanes::ResultSign::Preserve),
        BinaryOperation::Subtract => (ArithmeticOpcode::Subtract, lanes::ResultSign::Preserve),
        BinaryOperation::Multiply | BinaryOperation::MultiplyExtended => {
            (ArithmeticOpcode::Multiply, lanes::ResultSign::Preserve)
        }
        BinaryOperation::NegatedMultiply => (ArithmeticOpcode::Multiply, lanes::ResultSign::Negate),
        BinaryOperation::AbsoluteDifference => {
            (ArithmeticOpcode::Subtract, lanes::ResultSign::Absolute)
        }
        // The Double quotient of two normal Halves rounds once more, but
        // stays on the same side of every Half rounding boundary: an inexact
        // quotient a/b lies at least 2^-23 (relative) from any number of 12
        // significant bits ((a - m*b)/b with integer fields), far beyond the
        // Double rounding error of 2^-53. Its discarded bits are then never
        // all zero and never an exact tie, so the Half result and its
        // inexact status are those of the exact quotient under every
        // rounding mode.
        BinaryOperation::Divide => (ArithmeticOpcode::Divide, lanes::ResultSign::Preserve),
    }
}

fn native_opcode(operation: BinaryOperation) -> HalfOpcode {
    match operation {
        BinaryOperation::Add => HalfOpcode::Add,
        BinaryOperation::Subtract | BinaryOperation::AbsoluteDifference => HalfOpcode::Subtract,
        BinaryOperation::Multiply
        | BinaryOperation::NegatedMultiply
        | BinaryOperation::MultiplyExtended => HalfOpcode::Multiply,
        BinaryOperation::Divide => HalfOpcode::Divide,
    }
}

/// FNMUL and FABD transform the rounded result, here in the `lanes` active
/// lanes; the cleared inactive lanes stay zero.
fn native_sign(code: &mut Code, operation: BinaryOperation, lanes: u8) {
    let (word, opcode) = match operation {
        BinaryOperation::NegatedMultiply => (Format::Half.sign_mask(), Host::Xor),
        BinaryOperation::AbsoluteDifference => (Format::Half.sign_mask() - 1, Host::And),
        _ => return,
    };
    let qword = (0..lanes.min(QWORD_LANES)).fold(0, |qword, lane| {
        qword | word << (u32::from(lane) * Format::Half.bits())
    });
    let mut asm = code.asm();
    asm.mov_imm64(Gpr::Rax, qword);
    asm.movq_to_xmm(NATIVE_SIGN, Gpr::Rax);
    if lanes > QWORD_LANES {
        packed(code, Host::UnpackLowQword, NATIVE_SIGN, NATIVE_SIGN);
    }
    packed(code, opcode, lanes::LEFT, NATIVE_SIGN);
}

/// One AVX512-FP16 operation under the same input guards as the lane recipe.
fn emit_native(code: &mut Code, operand: &Operand, pc: GuestAddress) {
    test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
    let mut helper = vec![branch(code, Condition::NotZero)];
    prepare_inputs(code, operand);
    prepare_vector_input_range(code, Format::Half);
    for source in [lanes::LEFT, lanes::RIGHT] {
        vector_exponent_range(code, Format::Half, operand.shape, source);
        helper.push(branch(code, Condition::NotZero));
    }
    lanes::run_native(code, operand.shape, &mut helper, |code| {
        vector_extensions::half_arithmetic(
            code,
            native_opcode(operand.operation),
            lanes::LEFT,
            lanes::RIGHT,
        );
    });
    native_sign(
        code,
        operand.operation,
        operand.shape.lane_count(Format::Half) as u8,
    );
    let merge =
        operand.shape == Shape::Scalar && !matches!(operand.inputs, Inputs::PairwiseScalar { .. });
    lanes::finish_native(code, operand.destination, merge, helper, pc);
}

pub(super) fn emit(code: &mut Code, operand: &Operand, pc: GuestAddress, target: TargetCpu) {
    debug_assert_eq!(operand.format, Format::Half);
    if vector_extensions::half_supported(target) {
        emit_native(code, operand, pc);
        return;
    }
    if packed::supports(operand, target) {
        packed::emit(code, operand, pc);
        return;
    }
    let mut helper = Vec::new();
    let mut rollback = Vec::new();
    let packed_product = product::supports(operand);
    let checked_widening =
        !packed_product && operand.shape != Shape::Scalar && target.supports(TargetCpu::F16c);
    if checked_widening {
        code.asm().vzeroupper();
    }
    // Ordinary binary keeps kept enables and fresh status under every AH.
    test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
    helper.push(branch(code, Condition::NotZero));
    prepare_inputs(code, operand);
    let zero_lanes = zero::supported(operand);
    if zero_lanes {
        prepare_vector_input_range(code, Format::Half);
        zero::substitute(code, operand, &mut helper);
    }
    prepare_vector_input_range(code, Format::Half);
    for source in [lanes::LEFT, lanes::RIGHT] {
        vector_exponent_range(code, Format::Half, operand.shape, source);
        helper.push(branch(code, Condition::NotZero));
    }
    if packed_product {
        product::emit(code, operand.shape, helper, pc);
    } else {
        lanes::allocate_frame(code);
        if checked_widening {
            lanes::set_pack_bias(code);
        } else {
            lanes::set_biases(code, lanes::LeftScale::Identity);
        }
        let loop_start = lanes::start_lanes(code);
        if checked_widening {
            lanes::widen_lane_checked(code, target);
        } else {
            lanes::widen_lane(code);
        }
        // Normal H sums/differences need at most 41 significant bits, products 22.
        // Nonzero results are normal Double values, exact under every caller
        // MXCSR; a quotient rounds once, harmlessly (`operation`).
        // Cancellation zero and unsafe Half outputs roll back before guest effects.
        let (opcode, sign) = operation(operand.operation);
        packed(code, Host::MoveAligned, lanes::RESULT, lanes::LEFT);
        // An inexact quotient raises host precision status, which the
        // caller's MXCSR must not keep: save and restore it around DIVSD.
        let divide = operand.operation == BinaryOperation::Divide;
        if divide {
            let mut asm = code.asm();
            asm.alu_imm8(
                crate::x86_64::asm::Alu::Sub,
                crate::x86_64::asm::Size::Qword,
                Gpr::Rsp,
                MXCSR_SLOT_BYTES,
            );
            asm.stmxcsr(crate::x86_64::asm::Mem::base(Gpr::Rsp, 0));
        }
        emit_opcode_to(
            code,
            Format::Double,
            Shape::Scalar,
            opcode,
            lanes::RESULT,
            lanes::RIGHT,
        );
        if divide {
            let mut asm = code.asm();
            asm.ldmxcsr(crate::x86_64::asm::Mem::base(Gpr::Rsp, 0));
            asm.alu_imm8(
                crate::x86_64::asm::Alu::Add,
                crate::x86_64::asm::Size::Qword,
                Gpr::Rsp,
                MXCSR_SLOT_BYTES,
            );
        }
        lanes::pack_lane(code, &mut rollback, lanes::RoundingPolicy::Fpcr, sign);
        lanes::finish_lanes(
            code,
            operand.shape.lane_count(Format::Half) as u8,
            loop_start,
            &helper,
            &rollback,
            pc,
        );
    }
    if zero_lanes {
        zero::merge(code, operand);
    }
    lanes::commit_inexact(code);
    if operand.shape == Shape::Scalar && !matches!(operand.inputs, Inputs::PairwiseScalar { .. }) {
        lanes::merge_scalar(code);
    }
    // Both the lane frame and packed product path clear inactive output bits.
    commit(code, operand.destination, lanes::LEFT);
}
