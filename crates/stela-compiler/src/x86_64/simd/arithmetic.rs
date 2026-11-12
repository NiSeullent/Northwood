//! FP binary arithmetic: guarded SSE2 values and precise software fallback.
use crate::x86_64::asm::BitTest;
use crate::x86_64::asm::{Alu, Code, Gpr, Rm, Size};
use alloc::vec;
use alloc::vec::Vec;
mod half;
#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux")
))]
mod half_tests;
#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux")
))]
mod tiny_tests;
use super::super::isa::ShiftOpcode;
use super::numerical::{
    ArithmeticOpcode, ColdSplit, Condition, EXPONENT_VECTOR, Entry, FPSR_UNDERFLOW,
    INPUT_MINIMUM_EXPONENT, Input, LEFT_VECTOR, RESULT_MINIMUM_EXPONENT, RIGHT_VECTOR,
    SINGLE_EXPONENT_BIAS_BITS, ZeroProduct, branch, clear_high_qword, clear_scalar, commit_inexact,
    commit_inexact_underflow, emit_opcode, environment, exponent_range, fpsr_has, jump,
    nonfinite_lane, prepare_vector_input_range, prepare_vector_result_range, read_inputs,
    read_numerical_right, read_numerical_vector, reject_other_status,
    reject_other_status_but_underflow, resolve, restore_environment, scalar_ordered_range,
    scalar_range_or_zero, sticky_inexact, sticky_result_check, test_control, vector_exponent_range,
    vector_normal_or_zero_range, vector_ordered_range, zero_product_check,
};
use super::{Host, commit, packed, read_vector, shuffle};
use crate::{
    GuestAddress, TargetCpu,
    aarch64::{
        floating::{
            BinaryOperation, FPCR_AH, FPCR_EXCEPTION_ENABLE_MASK, FPCR_FZ, FPCR_NEP, Format,
        },
        simd::arithmetic::{Inputs, Operand, Shape},
    },
    block::ExitKind,
};

const PAIRED_RIGHT: u8 = 6;
const ZERO_RANGE_SCRATCH: u8 = 7;
const SIGN_MASK_VECTOR: u8 = 2;
const EVEN_SINGLE_LANES: u8 = 0x88;
const ODD_SINGLE_LANES: u8 = 0xdd;
const FIRST_LANE: u8 = 0;
const REPLICATE_SINGLE_LANE: u8 = 0x55;
const REPLICATE_LOW_DOUBLE: u8 = 0x44;
const REPLICATE_HIGH_DOUBLE: u8 = 0xee;
const SIGN_BITS: u8 = 1;

fn arithmetic_opcode(operation: BinaryOperation) -> ArithmeticOpcode {
    match operation {
        BinaryOperation::Add => ArithmeticOpcode::Add,
        BinaryOperation::Subtract | BinaryOperation::AbsoluteDifference => {
            ArithmeticOpcode::Subtract
        }
        BinaryOperation::Multiply
        | BinaryOperation::NegatedMultiply
        | BinaryOperation::MultiplyExtended => ArithmeticOpcode::Multiply,
        BinaryOperation::Divide => ArithmeticOpcode::Divide,
    }
}

fn shuffle_single(code: &mut Code, destination: u8, source: u8, lanes: u8) {
    code.asm().shufps(destination, Rm::Xmm(source), lanes);
}

/// Stage adjacent Vn pairs before adjacent Vm pairs without numerical work.
/// Return the right host register; classifier scratch XMM2/3/4/5 must not alias it.
fn prepare_pairs(code: &mut Code, format: Format, shape: Shape) -> u8 {
    match (format, shape) {
        (Format::Single, Shape::Vector64) => {
            // Q0 loads zero both high qwords. The extra packed sums are normal
            // values plus zero, so they cannot add exceptions to active lanes.
            let mut asm = code.asm();
            asm.unpcklps(LEFT_VECTOR, Rm::Xmm(RIGHT_VECTOR));
            asm.movhlps(RIGHT_VECTOR, Rm::Xmm(LEFT_VECTOR));
            RIGHT_VECTOR
        }
        (Format::Single, Shape::Vector128) => {
            packed(code, Host::MoveAligned, PAIRED_RIGHT, LEFT_VECTOR);
            shuffle_single(code, LEFT_VECTOR, RIGHT_VECTOR, EVEN_SINGLE_LANES);
            shuffle_single(code, PAIRED_RIGHT, RIGHT_VECTOR, ODD_SINGLE_LANES);
            PAIRED_RIGHT
        }
        (Format::Double, Shape::Vector128) => {
            packed(code, Host::MoveAligned, PAIRED_RIGHT, LEFT_VECTOR);
            packed(code, Host::UnpackLowQword, LEFT_VECTOR, RIGHT_VECTOR);
            packed(code, Host::UnpackHighQword, PAIRED_RIGHT, RIGHT_VECTOR);
            PAIRED_RIGHT
        }
        _ => unreachable!("allocated S/D vector pairwise shape"),
    }
}

fn prepare_inputs(code: &mut Code, operand: &Operand) -> u8 {
    match operand.inputs {
        Inputs::Elementwise { left, right } => {
            read_numerical_vector(code, operand.shape, left, LEFT_VECTOR);
            read_numerical_vector(code, operand.shape, right, RIGHT_VECTOR);
            RIGHT_VECTOR
        }
        Inputs::Indexed {
            left,
            right,
            right_lane,
        } => {
            read_numerical_vector(code, operand.shape, left, LEFT_VECTOR);
            // Both policies select from the whole Vm. The 128-bit form uses one
            // shuffle; the 64-bit form keeps its shift and broadcast and can
            // still select an upper lane.
            if operand.shape == Shape::Vector128 {
                read_vector(code, right, RIGHT_VECTOR);
                let lanes = match operand.format {
                    Format::Single => right_lane * REPLICATE_SINGLE_LANE,
                    Format::Double if right_lane == FIRST_LANE => REPLICATE_LOW_DOUBLE,
                    Format::Double => REPLICATE_HIGH_DOUBLE,
                    Format::Half => unreachable!("Half has its own mapping policy"),
                };
                shuffle(code, RIGHT_VECTOR, RIGHT_VECTOR, lanes);
            } else {
                read_numerical_right(code, operand.format, operand.shape, right, Some(right_lane));
                // Inactive lanes multiply zeros, so an infinite element
                // raises no host invalid there.
                packed(code, Host::MoveQword, RIGHT_VECTOR, RIGHT_VECTOR);
            }
            RIGHT_VECTOR
        }
        Inputs::Pairwise { left, right } => {
            read_numerical_vector(code, operand.shape, left, LEFT_VECTOR);
            read_numerical_vector(code, operand.shape, right, RIGHT_VECTOR);
            prepare_pairs(code, operand.format, operand.shape)
        }
        Inputs::PairwiseScalar { source } => {
            read_vector(code, source, LEFT_VECTOR);
            packed(code, Host::MoveAligned, RIGHT_VECTOR, LEFT_VECTOR);
            ShiftOpcode::RightBytes.emit(
                code,
                RIGHT_VECTOR,
                (operand.format.bits() / u8::BITS) as u8,
            );
            RIGHT_VECTOR
        }
    }
}

/// FNMUL/FABD transform the rounded result. Only normal, zero or infinite results reach
/// this point, so AH's NaN sign policy stays on the exact helper path.
fn result_sign(code: &mut Code, operand: &Operand) {
    let negate = match operand.operation {
        BinaryOperation::NegatedMultiply => true,
        BinaryOperation::AbsoluteDifference => false,
        _ => return,
    };
    if operand.shape == Shape::Scalar {
        match operand.format {
            Format::Single => {
                let sign = operand.format.sign_mask() as u32;
                let mut asm = code.asm();
                asm.movd_from_xmm(Gpr::Rax, LEFT_VECTOR);
                if negate {
                    asm.alu_accumulator_imm32(Alu::Xor, Size::Dword, sign as i32);
                } else {
                    asm.alu_accumulator_imm32(Alu::And, Size::Dword, !sign as i32);
                }
                asm.movd_to_xmm(SIGN_MASK_VECTOR, Gpr::Rax);
                Host::MoveScalarSingle.register(
                    code,
                    LEFT_VECTOR,
                    SIGN_MASK_VECTOR,
                    TargetCpu::Baseline,
                );
            }
            Format::Double => {
                let mut asm = code.asm();
                asm.movq_from_xmm(Gpr::Rax, LEFT_VECTOR);
                asm.bit_imm(
                    if negate { BitTest::Btc } else { BitTest::Btr },
                    Size::Qword,
                    Gpr::Rax,
                    operand.format.bits() as u8 - SIGN_BITS,
                );
                asm.movq_to_xmm(SIGN_MASK_VECTOR, Gpr::Rax);
                Host::MoveScalarDouble.register(
                    code,
                    LEFT_VECTOR,
                    SIGN_MASK_VECTOR,
                    TargetCpu::Baseline,
                );
            }
            Format::Half => unreachable!("Half has its own mapping policy"),
        }
    } else {
        packed(code, Host::EqualDword, SIGN_MASK_VECTOR, SIGN_MASK_VECTOR);
        let shift = match (operand.format, negate) {
            (Format::Single, true) => ShiftOpcode::LeftDword,
            (Format::Double, true) => ShiftOpcode::LeftQword,
            (Format::Single, false) => ShiftOpcode::RightDword,
            (Format::Double, false) => ShiftOpcode::RightQword,
            (Format::Half, _) => unreachable!("Half has its own mapping policy"),
        };
        shift.emit(
            code,
            SIGN_MASK_VECTOR,
            if negate {
                operand.format.bits() as u8 - SIGN_BITS
            } else {
                SIGN_BITS
            },
        );
        packed(
            code,
            if negate { Host::Xor } else { Host::And },
            LEFT_VECTOR,
            SIGN_MASK_VECTOR,
        );
    }
}

/// Addition and subtraction results below the minimum normal are exact, so
/// they raise no guest status.
fn exact_tiny(operation: BinaryOperation) -> bool {
    matches!(
        operation,
        BinaryOperation::Add | BinaryOperation::Subtract | BinaryOperation::AbsoluteDifference
    )
}

/// A product whose rounded result is a zero is exact, without status,
/// exactly when an input is a zero; otherwise it underflowed. FMULX's
/// zero times infinity is two, not a zero.
fn zero_product(operation: BinaryOperation) -> bool {
    matches!(
        operation,
        BinaryOperation::Multiply
            | BinaryOperation::NegatedMultiply
            | BinaryOperation::MultiplyExtended
    )
}

/// The full path's status checks distinguish exact zeros from underflowed
/// nonzero results, and infinite results of an infinite input from
/// overflow: the host raises overflow for the latter, and invalid for
/// infinity times zero (FMULX's two included) and infinity minus infinity,
/// which `reject_other_status` sends to the helper. Neither side raises
/// status for an infinity with a finite, non-subnormal operand, and its
/// result is the same infinity in every rounding mode, FZ and AH. Division
/// keeps its existing conservative domain.
fn accepts_zero(operation: BinaryOperation) -> bool {
    match operation {
        BinaryOperation::Add
        | BinaryOperation::Subtract
        | BinaryOperation::Multiply
        | BinaryOperation::NegatedMultiply
        | BinaryOperation::MultiplyExtended
        | BinaryOperation::AbsoluteDifference => true,
        BinaryOperation::Divide => false,
    }
}

/// An FDIV lane's zero dividend over a normal divisor: the quotient is
/// an exact signed zero, with no status, whatever FPCR holds. Code divides
/// zero often (a ratio, a normalized silent sample or transparent pixel, an
/// animation's progress at its start), and each one would otherwise leave
/// compiled code. A zero quotient from a normal
/// dividend underflowed, which the host's status rejects; the divisor's
/// range check still sends a zero or infinite divisor to the helper.
fn zero_quotient(operand: &Operand) -> bool {
    operand.operation == BinaryOperation::Divide
        && matches!(operand.inputs, Inputs::Elementwise { .. })
}

/// The left source, whose register the operation overwrote, for the
/// out-of-line zero-product check: still unchanged in the frame.
fn product_left(operand: &Operand) -> u8 {
    match operand.inputs {
        Inputs::Elementwise { left, .. } | Inputs::Indexed { left, .. } => left,
        Inputs::Pairwise { .. } | Inputs::PairwiseScalar { .. } => {
            unreachable!("pairwise forms add")
        }
    }
}

/// The sticky-inexact path (`numerical::sticky_inexact`): the operation in
/// the entry MXCSR, its result classification and the commit, which clears
/// scalar upper bits as NEP is zero there. Products that the classification
/// rejects first take an out-of-line check for exact zeros
/// ([`zero_product`]). Returns the branches to the full path, to the helper
/// and past the full path, and those between the sticky path and the
/// rest ([`ColdSplit`]).
fn sticky_path(code: &mut Code, operand: &Operand, entry: Entry, target: TargetCpu) -> StickyPath {
    let full = sticky_inexact(code, entry);
    // Zeroing loads leave a scalar result's upper bits clear.
    let zeroed =
        operand.shape == Shape::Scalar && matches!(operand.inputs, Inputs::Elementwise { .. });
    let right = if let Inputs::Elementwise { left, right } = operand.inputs {
        let input = if zeroed {
            Input::Scalar(operand.format)
        } else {
            Input::Vector(operand.shape)
        };
        read_inputs(
            code,
            entry.xmm0,
            target,
            &[(input, left, LEFT_VECTOR), (input, right, RIGHT_VECTOR)],
        );
        RIGHT_VECTOR
    } else {
        prepare_inputs(code, operand)
    };
    emit_opcode(
        code,
        operand.format,
        operand.shape,
        arithmetic_opcode(operand.operation),
        right,
    );
    let rejected = sticky_result_check(
        code,
        operand.format,
        operand.shape,
        LEFT_VECTOR,
        exact_tiny(operand.operation),
        target,
    );
    let resume = code.len();
    result_sign(code, operand);
    match operand.shape {
        Shape::Scalar if !zeroed => clear_scalar(code, operand.format, LEFT_VECTOR),
        Shape::Vector64 => clear_high_qword(code, 0),
        Shape::Scalar | Shape::Vector128 => {}
    }
    commit(code, operand.destination, LEFT_VECTOR);
    let inline_end = code.checkpoint();
    let done = jump(code);
    let cold = code.checkpoint();
    if !zero_product(operand.operation) {
        // A zero quotient may be exact, and an infinite sum or difference
        // of an infinite input is: the full path tells, from its inputs
        // and the host's status, rather than the helper.
        let mut full = full;
        full.push(rejected);
        return StickyPath {
            full,
            helper: vec![],
            done,
            cold,
            inline_end,
            crossing: vec![],
        };
    }
    let mut crossing = vec![rejected];
    let check = code.len();
    resolve(code, rejected, check);
    // With FPSR.UFC set too, a tiny or zero product adds no status, and its
    // value is the host's: the entry MXCSR flushes nothing. Only a NaN or
    // infinite lane needs more: the full path, which applies Arm's NaN
    // selection, DN and IOC. (The zero-product check cannot take it: its
    // accepted-lane mask is the one the nonfinite test has just replaced.)
    let known = fpsr_has(code, FPSR_UNDERFLOW);
    let unknown = jump(code);
    let finite_check = code.len();
    resolve(code, known, finite_check);
    let nonfinite = nonfinite_lane(
        code,
        operand.format,
        operand.shape,
        LEFT_VECTOR,
        EXPONENT_VECTOR,
    );
    let finite = jump(code);
    resolve(code, finite, resume);
    crossing.push(finite);
    let zero_check = code.len();
    resolve(code, unknown, zero_check);
    // Other rejected products take the full path, which completes tiny
    // results ([`tiny_product_status`]).
    let mut full = full;
    full.push(nonfinite);
    full.extend(zero_product_check(
        code,
        &ZeroProduct {
            format: operand.format,
            shape: operand.shape,
            right,
            left: product_left(operand),
            addend: None,
        },
        resume,
        &mut crossing,
    ));
    StickyPath {
        full,
        helper: vec![],
        done,
        cold,
        inline_end,
        crossing,
    }
}

/// The full path's status of a product ([`zero_product`]) under FPCR.FZ and
/// AH zero, when it is not `None`: results below the minimum normal
/// complete as the host computed them (no FTZ in the environment), with
/// MXCSR underflow as FPSR.UFC: SSE, like Arm with AH zero, detects
/// tininess before rounding and, with underflow masked, reports it for an
/// inexact tiny result (measured: the minimum normal times 1 - 2^-24 rounds
/// to the minimum normal with PE and UE). Returns the branches to the
/// success path; `helper` receives the others. With FZ or AH set (AH
/// detects tininess after rounding), the caller's checks follow.
fn tiny_product_status(
    code: &mut Code,
    operand: &Operand,
    helper: &mut Vec<usize>,
) -> Option<(Vec<usize>, usize)> {
    if !zero_product(operand.operation) {
        return None;
    }
    test_control(code, FPCR_FZ | FPCR_AH);
    let strict = branch(code, Condition::NotZero);
    helper.push(reject_other_status_but_underflow(code));
    Some((vec![jump(code)], strict))
}

/// A sticky path's open branches ([`sticky_path`]).
struct StickyPath {
    full: Vec<usize>,
    helper: Vec<usize>,
    done: usize,
    cold: crate::x86_64::asm::Checkpoint,
    inline_end: crate::x86_64::asm::Checkpoint,
    crossing: Vec<usize>,
}

/// Resolve the sticky path's branches to the full path, which begins here.
fn enter_full(code: &mut Code, sticky: &mut StickyPath) -> usize {
    let full = code.len();
    for &patch in &sticky.full {
        resolve(code, patch, full);
    }
    sticky.crossing.append(&mut sticky.full);
    full
}

fn emit_scalar(
    code: &mut Code,
    operand: &Operand,
    pc: GuestAddress,
    entry: Entry,
    target: TargetCpu,
) -> ColdSplit {
    let mut sticky = sticky_path(code, operand, entry, target);
    let full = enter_full(code, &mut sticky);
    let mut helper = core::mem::take(&mut sticky.helper);
    test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
    helper.push(branch(code, Condition::NotZero));
    let right = prepare_inputs(code, operand);
    for source in [LEFT_VECTOR, right] {
        if accepts_zero(operand.operation) {
            helper.push(scalar_ordered_range(
                code,
                operand.format,
                source,
                INPUT_MINIMUM_EXPONENT,
            ));
        } else if source == LEFT_VECTOR && zero_quotient(operand) {
            helper.push(scalar_range_or_zero(
                code,
                operand.format,
                source,
                INPUT_MINIMUM_EXPONENT,
            ));
        } else {
            exponent_range(code, operand.format, source, INPUT_MINIMUM_EXPONENT);
            helper.push(branch(code, Condition::AboveOrEqual));
        }
    }
    environment(code);
    emit_opcode(
        code,
        operand.format,
        operand.shape,
        arithmetic_opcode(operand.operation),
        right,
    );
    restore_environment(code);
    // Caller environment and stack are restored before every branch from here.
    let tiny = tiny_product_status(code, operand, &mut helper);
    if let Some((_, strict)) = &tiny {
        let strict_start = code.len();
        resolve(code, *strict, strict_start);
    }
    helper.push(reject_other_status(code));
    let success = if accepts_zero(operand.operation) {
        helper.push(scalar_ordered_range(
            code,
            operand.format,
            LEFT_VECTOR,
            RESULT_MINIMUM_EXPONENT,
        ));
        jump(code)
    } else if zero_quotient(operand) {
        helper.push(scalar_range_or_zero(
            code,
            operand.format,
            LEFT_VECTOR,
            RESULT_MINIMUM_EXPONENT,
        ));
        jump(code)
    } else {
        exponent_range(code, operand.format, LEFT_VECTOR, RESULT_MINIMUM_EXPONENT);
        branch(code, Condition::Below)
    };
    let helper_start = code.len();
    super::super::block::direct_exit(code, pc, ExitKind::FloatingPoint);
    let success_start = code.len();
    for patch in helper {
        resolve(code, patch, helper_start);
    }
    resolve(code, success, success_start);
    commit_status(code, success_start, tiny);
    result_sign(code, operand);
    if matches!(operand.inputs, Inputs::PairwiseScalar { .. }) {
        clear_scalar(code, operand.format, LEFT_VECTOR);
    } else {
        test_control(code, FPCR_NEP);
        let merge = branch(code, Condition::NotZero);
        clear_scalar(code, operand.format, LEFT_VECTOR);
        let merge_target = code.len();
        resolve(code, merge, merge_target);
    }
    commit(code, operand.destination, LEFT_VECTOR);
    finish(code, sticky, full, operand)
}

/// Resolve the tiny-product successes ([`tiny_product_status`]) to
/// `success_start`, and commit the status of the full path.
fn commit_status(code: &mut Code, success_start: usize, tiny: Option<(Vec<usize>, usize)>) {
    match tiny {
        Some((successes, _)) => {
            for patch in successes {
                resolve(code, patch, success_start);
            }
            commit_inexact_underflow(code);
        }
        None => commit_inexact(code),
    }
}

/// Resolve the jump past the full path, which ends here.
fn finish(code: &mut Code, sticky: StickyPath, full: usize, operand: &Operand) -> ColdSplit {
    let done = code.len();
    resolve(code, sticky.done, done);
    ColdSplit {
        cold: sticky.cold,
        inline_end: sticky.inline_end,
        full,
        crossing: sticky.crossing,
        result: operand.destination,
    }
}

fn emit_vector(
    code: &mut Code,
    operand: &Operand,
    pc: GuestAddress,
    entry: Entry,
    target: TargetCpu,
) -> ColdSplit {
    let mut sticky = sticky_path(code, operand, entry, target);
    let full = enter_full(code, &mut sticky);
    let mut helper = core::mem::take(&mut sticky.helper);
    test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
    helper.push(branch(code, Condition::NotZero));
    let right = prepare_inputs(code, operand);
    prepare_vector_input_range(code, operand.format);
    for source in [LEFT_VECTOR, right] {
        if accepts_zero(operand.operation) {
            vector_ordered_range(
                code,
                operand.format,
                operand.shape,
                source,
                ZERO_RANGE_SCRATCH,
            );
        } else if source == LEFT_VECTOR && zero_quotient(operand) {
            vector_normal_or_zero_range(
                code,
                operand.format,
                operand.shape,
                source,
                ZERO_RANGE_SCRATCH,
            );
        } else {
            vector_exponent_range(code, operand.format, operand.shape, source);
        }
        helper.push(branch(code, Condition::NotZero));
    }
    if operand.shape == Shape::Vector64 && operand.operation == BinaryOperation::Divide {
        // Inactive numerators are zero; 1.0 denominators keep 0/0 from raising
        // host invalid.
        packed(code, Host::EqualDword, EXPONENT_VECTOR, EXPONENT_VECTOR);
        ShiftOpcode::RightDword.emit(
            code,
            EXPONENT_VECTOR,
            u32::BITS as u8 - SINGLE_EXPONENT_BIAS_BITS,
        );
        ShiftOpcode::LeftDword.emit(code, EXPONENT_VECTOR, Format::Single.fraction_bits() as u8);
        code.asm().movlhps(1, crate::x86_64::asm::Rm::Xmm(2));
    }
    environment(code);
    emit_opcode(
        code,
        operand.format,
        operand.shape,
        arithmetic_opcode(operand.operation),
        right,
    );
    restore_environment(code);
    let tiny = tiny_product_status(code, operand, &mut helper);
    if let Some((_, strict)) = &tiny {
        let strict_start = code.len();
        resolve(code, *strict, strict_start);
    }
    helper.push(reject_other_status(code));
    prepare_vector_result_range(code);
    if accepts_zero(operand.operation) {
        vector_ordered_range(
            code,
            operand.format,
            operand.shape,
            LEFT_VECTOR,
            ZERO_RANGE_SCRATCH,
        );
    } else if zero_quotient(operand) {
        vector_normal_or_zero_range(
            code,
            operand.format,
            operand.shape,
            LEFT_VECTOR,
            ZERO_RANGE_SCRATCH,
        );
    } else {
        vector_exponent_range(code, operand.format, operand.shape, LEFT_VECTOR);
    }
    let success = branch(code, Condition::Zero);
    let helper_start = code.len();
    super::super::block::direct_exit(code, pc, ExitKind::FloatingPoint);
    let success_start = code.len();
    for patch in helper {
        resolve(code, patch, helper_start);
    }
    resolve(code, success, success_start);
    commit_status(code, success_start, tiny);
    result_sign(code, operand);
    if operand.shape == Shape::Vector64 {
        clear_high_qword(code, 0);
    }
    commit(code, operand.destination, LEFT_VECTOR);
    finish(code, sticky, full, operand)
}

/// No binary form ends its block: Half division runs the lane recipe.
pub(super) fn always_exits(_operand: &Operand) -> bool {
    false
}

/// The S/D forms return their [`ColdSplit`].
pub(in crate::x86_64) fn emit(
    code: &mut Code,
    operand: &Operand,
    pc: GuestAddress,
    target: TargetCpu,
    entry: Entry,
) -> Option<ColdSplit> {
    if operand.format == Format::Half {
        half::emit(code, operand, pc, target);
        None
    } else if operand.shape == Shape::Scalar {
        Some(emit_scalar(code, operand, pc, entry, target))
    } else {
        Some(emit_vector(code, operand, pc, entry, target))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn numerical_instructions_require_the_block_exit_protocol() {
        const FADD_S: u32 = 0x1e222820;
        const FADD_2S: u32 = 0x0e22d420;
        const RET: u32 = 0xd65f03c0;
        for word in [
            FADD_S,
            FADD_2S,
            0x1ee2_2820, // fadd h0, h1, h2
            0x1ee2_3820, // fsub h0, h1, h2
            0x1ee2_0820, // fmul h0, h1, h2
            0x0e42_1420, // fadd v0.4h, v1.4h, v2.4h
            0x0ec2_1420, // fsub v0.4h, v1.4h, v2.4h
            0x2e42_1c20, // fmul v0.4h, v1.4h, v2.4h
            0x4e42_1420, // fadd v0.8h, v1.8h, v2.8h
            0x4ec2_1420, // fsub v0.8h, v1.8h, v2.8h
            0x6e42_1c20, // fmul v0.8h, v1.8h, v2.8h
            0x1ee2_1820, // fdiv h0, h1, h2
            0x2e42_3c20, // fdiv v0.4h, v1.4h, v2.4h
            0x6e42_3c20, // fdiv v0.8h, v1.8h, v2.8h
        ] {
            let bytes: Vec<u8> = [word, RET]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            assert!(matches!(
                crate::testing::leaf(&bytes, GuestAddress(0x1000))
                    .unwrap_err()
                    .kind,
                crate::CompileErrorKind::Unsupported(
                    crate::UnsupportedReason::NumericalRuntimeRequired { .. }
                )
            ));
            let block = crate::testing::block(&bytes, GuestAddress(0x1000)).unwrap();
            assert_eq!(block.source_bytes, bytes.len());
            assert_eq!(block.native.source_map.len(), 2);
            assert_eq!(block.native.source_map[1].0, GuestAddress(0x1004));
        }
    }
}
