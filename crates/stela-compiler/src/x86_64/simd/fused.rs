//! Fused multiply-add: FMA3 when available, otherwise an exact-product SSE2 path.
use super::super::isa::ShiftOpcode;
use super::numerical::{
    ArithmeticOpcode, ColdSplit, Condition, Entry, FusedOpcode, INPUT_MINIMUM_EXPONENT, Input,
    RESULT_MINIMUM_EXPONENT, ZeroProduct, branch, clear_scalar, commit_inexact, emit_fma,
    emit_opcode, environment, jump, prepare_vector_input_range, prepare_vector_result_range,
    read_inputs, read_numerical_right, read_numerical_vector, reject_other_status,
    reject_product_status, resolve, restore_environment, restore_environment_with_product,
    save_product_status, scalar_ordered_range, sticky_inexact, sticky_result_check, test_control,
    vector_ordered_range, zero_product_check,
};
use super::{Host, commit, packed};
use crate::x86_64::asm::Code;
use crate::{
    GuestAddress, TargetCpu,
    aarch64::{
        floating::{FPCR_EXCEPTION_ENABLE_MASK, FPCR_NEP, Format},
        simd::{arithmetic::Shape, fused::Operand},
    },
    block::ExitKind,
};
use alloc::vec;
use alloc::vec::Vec;
mod half;
#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux")
))]
mod tests;

const LEFT: u8 = 0;
const RIGHT: u8 = 1;
const SIGN_MASK: u8 = 2;
const ADDEND: u8 = 6;
const ORIGINAL_ADDEND: u8 = 7;
/// The zero classifier's scratch: the scalar-only original addend's register.
const ZERO_RANGE_SCRATCH: u8 = ORIGINAL_ADDEND;

fn emit_checked_fma(code: &mut Code, operand: &Operand, target: TargetCpu) {
    emit_fma(
        code,
        operand.format,
        operand.shape,
        FusedOpcode::from(operand.operation),
        target,
    );
}

fn prepare_right(code: &mut Code, operand: &Operand) {
    read_numerical_right(
        code,
        operand.format,
        operand.shape,
        operand.right,
        operand.right_lane,
    );
    if operand.shape == Shape::Vector64 && operand.right_lane.is_some() {
        // Inactive lanes multiply zeros, so an infinite element raises no
        // host invalid there.
        packed(code, Host::MoveQword, RIGHT, RIGHT);
    }
}

fn prepare_negation(code: &mut Code, operand: &Operand) -> u8 {
    let merge = if operand.shape == Shape::Scalar && operand.operation.negate_addend() {
        packed(code, Host::MoveAligned, ORIGINAL_ADDEND, ADDEND);
        ORIGINAL_ADDEND
    } else {
        ADDEND
    };
    if operand.operation.negate_addend() || operand.operation.negate_left() {
        packed(code, Host::EqualDword, SIGN_MASK, SIGN_MASK);
        let shift = if operand.format == Format::Single {
            ShiftOpcode::LeftDword
        } else {
            ShiftOpcode::LeftQword
        };
        shift.emit(code, SIGN_MASK, (operand.format.bits() - 1) as u8);
        if operand.operation.negate_left() {
            packed(code, Host::Xor, LEFT, SIGN_MASK);
        }
        if operand.operation.negate_addend() {
            packed(code, Host::Xor, ADDEND, SIGN_MASK);
        }
    }
    merge
}

/// With FMA3, S/D forms return their [`ColdSplit`].
pub(in crate::x86_64) fn emit(
    code: &mut Code,
    operand: &Operand,
    pc: GuestAddress,
    target: TargetCpu,
    entry: Entry,
) -> Option<ColdSplit> {
    if operand.format == Format::Half {
        half::emit(code, operand, pc, target);
        return None;
    }
    let native_fma = target.supports(TargetCpu::Fma);
    let (mut helper, sticky) = if native_fma {
        let StickyPath {
            full,
            helper,
            done,
            mut crossing,
            cold,
            inline_end,
        } = sticky_path(code, operand, target, entry);
        let full_start = code.len();
        for &patch in &full {
            resolve(code, patch, full_start);
        }
        crossing.extend(full);
        (helper, Some((done, full_start, crossing, cold, inline_end)))
    } else {
        (Vec::new(), None)
    };
    test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
    helper.push(branch(code, Condition::NotZero));
    read_numerical_vector(code, operand.shape, operand.left, LEFT);
    read_numerical_vector(code, operand.shape, operand.addend, ADDEND);
    prepare_right(code, operand);
    if operand.shape != Shape::Scalar {
        prepare_vector_input_range(code, operand.format);
    }
    // A zero input or lane is exact on both sides: FPMulAdd and the host's
    // FMA follow IEEE 754's signed-zero sums under the installed rounding
    // (an exact zero sum is +0, -0 rounding toward minus infinity, and a sum
    // of two zeros of one sign keeps it), a zero product adds exactly, and
    // a zero is not subnormal, so FZ and FIZ flush nothing, and neither side
    // raises status for it. An infinite input is exact too: its result is an
    // infinity in every rounding mode, without status, unless it meets a zero
    // factor or an opposite infinity, which raise the host's invalid (the
    // separate product's too) and go to the helper; a finite product that
    // overflowed raised overflow. The SSE2 product's own status rejects any
    // rounding before the sum.
    for source in [LEFT, RIGHT, ADDEND] {
        if operand.shape == Shape::Scalar {
            helper.push(scalar_ordered_range(
                code,
                operand.format,
                source,
                INPUT_MINIMUM_EXPONENT,
            ));
        } else {
            vector_ordered_range(
                code,
                operand.format,
                operand.shape,
                source,
                ZERO_RANGE_SCRATCH,
            );
            helper.push(branch(code, Condition::NotZero));
        }
    }
    let merge = if native_fma {
        ADDEND
    } else {
        prepare_negation(code, operand)
    };
    environment(code);
    if native_fma {
        emit_checked_fma(code, operand, target);
        restore_environment(code);
    } else {
        emit_opcode(
            code,
            operand.format,
            operand.shape,
            ArithmeticOpcode::Multiply,
            RIGHT,
        );
        save_product_status(code);
        emit_opcode(
            code,
            operand.format,
            operand.shape,
            ArithmeticOpcode::Add,
            ADDEND,
        );
        restore_environment_with_product(code);
        // All host status and stack state are restored before either side exit.
        helper.push(reject_product_status(code));
    }
    helper.push(reject_other_status(code));
    let success = if operand.shape == Shape::Scalar {
        // An exact zero result raised no status; a nonzero one rounded to
        // zero raised underflow, which went to the helper above.
        helper.push(scalar_ordered_range(
            code,
            operand.format,
            LEFT,
            RESULT_MINIMUM_EXPONENT,
        ));
        jump(code)
    } else {
        prepare_vector_result_range(code);
        // A zero result is exact: one that underflowed raised the host's
        // underflow status, which went to the helper above.
        vector_ordered_range(
            code,
            operand.format,
            operand.shape,
            LEFT,
            ZERO_RANGE_SCRATCH,
        );
        branch(code, Condition::Zero)
    };
    let helper_start = code.len();
    super::super::block::direct_exit(code, pc, ExitKind::FloatingPoint);
    let success_start = code.len();
    for patch in helper {
        resolve(code, patch, helper_start);
    }
    resolve(code, success, success_start);
    commit_inexact(code);
    if operand.shape == Shape::Scalar {
        // Register MOVSS/MOVSD replaces only the result lane of A.
        if operand.format == Format::Single {
            Host::MoveScalarSingle
        } else {
            Host::MoveScalarDouble
        }
        .register(code, merge, LEFT, crate::TargetCpu::Baseline);
        test_control(code, FPCR_NEP);
        let merging = branch(code, Condition::NotZero);
        clear_scalar(code, operand.format, merge);
        let commit_start = code.len();
        resolve(code, merging, commit_start);
        commit(code, operand.destination, merge);
        // XMM0 holds the destination wherever the emission ends.
        packed(code, Host::MoveAligned, LEFT, merge);
    } else {
        if operand.shape == Shape::Vector64 {
            packed(code, Host::MoveQword, LEFT, LEFT);
        }
        commit(code, operand.destination, LEFT);
    }
    let (done, full, crossing, cold, inline_end) = sticky?;
    let end = code.len();
    resolve(code, done, end);
    Some(ColdSplit {
        cold,
        inline_end,
        full,
        crossing,
        result: operand.destination,
    })
}

/// The sticky-inexact path (`numerical::sticky_inexact`) for a host with
/// FMA3, whose single rounding is the guest's: the operation in the entry
/// MXCSR, its result classification and the commit, which clears scalar
/// upper bits as NEP is zero there. A rejected zero takes the out-of-line
/// zero-product check, and other rejected results the full path. Returns the branches to the full path, to the helper
/// and past the full path, and those between the sticky path and the rest
/// ([`ColdSplit`]).
struct StickyPath {
    full: Vec<usize>,
    helper: Vec<usize>,
    done: usize,
    crossing: Vec<usize>,
    cold: crate::x86_64::asm::Checkpoint,
    inline_end: crate::x86_64::asm::Checkpoint,
}

fn sticky_path(code: &mut Code, operand: &Operand, target: TargetCpu, entry: Entry) -> StickyPath {
    let full = sticky_inexact(code, entry);
    // A zeroing load of the destination LEFT leaves a scalar result's
    // upper bits clear.
    let left = if operand.shape == Shape::Scalar {
        Input::Scalar(operand.format)
    } else {
        Input::Vector(operand.shape)
    };
    let vector = Input::Vector(operand.shape);
    let mut inputs = vec![(left, operand.left, LEFT), (vector, operand.addend, ADDEND)];
    if operand.right_lane.is_none() {
        inputs.push((vector, operand.right, RIGHT));
    }
    read_inputs(code, entry.xmm0, target, &inputs);
    if operand.right_lane.is_some() {
        prepare_right(code, operand);
    }
    emit_checked_fma(code, operand, target);
    let rejected = sticky_result_check(code, operand.format, operand.shape, LEFT, false, target);
    let resume = code.len();
    if operand.shape == Shape::Vector64 {
        packed(code, Host::MoveQword, LEFT, LEFT);
    }
    commit(code, operand.destination, LEFT);
    let inline_end = code.checkpoint();
    let done = jump(code);
    let cold = code.checkpoint();
    let check = code.len();
    resolve(code, rejected, check);
    let mut crossing = vec![rejected];
    // An infinite result may be exact: the full path tells, from its inputs
    // and the host's status, rather than the helper.
    let mut full = full;
    full.extend(zero_product_check(
        code,
        &ZeroProduct {
            format: operand.format,
            shape: operand.shape,
            right: RIGHT,
            left: operand.left,
            addend: Some(ADDEND),
        },
        resume,
        &mut crossing,
    ));
    StickyPath {
        full,
        helper: vec![],
        done,
        crossing,
        cold,
        inline_end,
    }
}
