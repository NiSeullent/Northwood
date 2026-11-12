//! FP comparisons: exception-free native ordering, software path for signaling cases.
use super::super::isa::ShiftOpcode;
use super::numerical::{
    Condition, EXPONENT_VECTOR, INPUT_MINIMUM_EXPONENT, MINIMUM_EXPONENT_VECTOR, branch,
    clear_scalar, exponent_range, prepare_vector_input_range, read_numerical_vector, resolve,
    test_control, vector_exponent_range,
};
use super::numerical::{helper_section, jump};
use super::{Host, commit, packed, read_vector};
use crate::x86_64::asm::FloatLanes;
use crate::x86_64::asm::{Alu, Code, Cond, Gpr, Rm, ShiftOp, Size};
use crate::x86_64::flag_word;
use crate::{
    GuestAddress, TargetCpu,
    aarch64::{
        flags::{NZCV_C, NZCV_N, NZCV_SHIFT, NZCV_Z},
        floating::{FPCR_EXCEPTION_ENABLE_MASK, FPCR_NEP, Format, comparison::Predicate},
        simd::{
            arithmetic::Shape,
            comparison::{CompareSource, Operand},
        },
    },
};
use alloc::vec::Vec;

const LEFT_VECTOR: u8 = 0;
const RIGHT_VECTOR: u8 = 1;
const ZERO_RESULT_VECTOR: u8 = 2;
const ZERO_VECTOR: u8 = 3;
const QWORD_EQUAL_VECTOR: u8 = 4;
const MERGE_VECTOR: u8 = 6;
const TEMP_VECTOR: u8 = 7;
const SIGN_BITS: u8 = 1;
const HALF_SIGN_SHIFT: u8 = u16::BITS as u8 - SIGN_BITS;
const SINGLE_SIGN_SHIFT: u8 = u32::BITS as u8 - SIGN_BITS;
/// Moves a Half's sign to the top of a dword.
const HALF_SIGN_TO_DWORD_TOP: u8 = (u32::BITS - u16::BITS) as u8;
const COMPARE_EQUAL: u8 = 0;
const COMPARE_LESS: u8 = 1;
const COMPARE_LESS_EQUAL: u8 = 2;
const HALF_LOW_MASK: u32 = 0xff;
const HALF_FULL_MASK: u32 = 0xffff;
const SINGLE_LOW_MASK: u32 = 0x03;
const SINGLE_FULL_MASK: u32 = 0x0f;
const DOUBLE_FULL_MASK: u32 = 0x05;
const ADJACENT_DWORDS: u8 = 0xb1;
const QWORD_HIGH_DWORDS: u8 = 0xf5;

/// The lane-width operand size of a scalar `format` value.
fn lane_size(format: Format) -> Size {
    match format {
        Format::Half => Size::Word,
        Format::Single => Size::Dword,
        Format::Double => Size::Qword,
    }
}

/// The GPR width that holds a scalar `format` value: a quadword for Double,
/// otherwise a doubleword.
fn register_size(format: Format) -> Size {
    if format == Format::Double {
        Size::Qword
    } else {
        Size::Dword
    }
}

/// Move the low `format` lane of `vector` into `register`.
fn move_scalar(code: &mut Code, format: Format, register: Gpr, vector: u8) {
    if format == Format::Double {
        code.asm().movq_from_xmm(register, vector);
    } else {
        code.asm().movd_from_xmm(register, vector);
    }
}

fn exception_guard(code: &mut Code, helper: &mut Vec<usize>) {
    test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
    helper.push(branch(code, Condition::NotZero));
}

fn scalar_normal_guard(code: &mut Code, format: Format, source: u8, helper: &mut Vec<usize>) {
    exponent_range(code, format, source, INPUT_MINIMUM_EXPONENT);
    helper.push(branch(code, Condition::AboveOrEqual));
}

/// A zero comparison also accepts signed zero and infinity. A zero fraction
/// proves one of those classes or an exact normal power of two; remaining
/// values must pass the ordinary normal guard. No subnormal or NaN executes.
fn scalar_zero_guard(code: &mut Code, format: Format, source: u8, helper: &mut Vec<usize>) {
    move_scalar(code, format, Gpr::Rax, source);
    let register_bits = if format == Format::Double {
        u64::BITS
    } else {
        u32::BITS
    };
    code.asm().shift_imm(
        ShiftOp::Shl,
        register_size(format),
        Gpr::Rax,
        (register_bits - format.fraction_bits()) as u8,
    );
    let zero_fraction = branch(code, Condition::Zero);
    scalar_normal_guard(code, format, source, helper);
    let accepted = code.len();
    resolve(code, zero_fraction, accepted);
}

fn active_mask(format: Format, shape: Shape) -> u32 {
    match (format, shape) {
        (Format::Half, Shape::Vector64) => HALF_LOW_MASK,
        (Format::Half, Shape::Vector128) => HALF_FULL_MASK,
        (Format::Single, Shape::Vector64) => SINGLE_LOW_MASK,
        (Format::Single, Shape::Vector128) => SINGLE_FULL_MASK,
        (Format::Double, Shape::Vector128) => DOUBLE_FULL_MASK,
        _ => unreachable!("decoded vector comparison arrangement"),
    }
}

fn shift_left(format: Format) -> ShiftOpcode {
    match format {
        Format::Half => ShiftOpcode::LeftWord,
        Format::Single => ShiftOpcode::LeftDword,
        Format::Double => ShiftOpcode::LeftQword,
    }
}

fn shift_right(format: Format) -> ShiftOpcode {
    match format {
        Format::Half => ShiftOpcode::RightWord,
        Format::Single => ShiftOpcode::RightDword,
        Format::Double => ShiftOpcode::RightQword,
    }
}

fn equal_zero(code: &mut Code, format: Format, value: u8, zero: u8, temporary: u8) {
    packed(
        code,
        if format == Format::Half {
            Host::EqualWord
        } else {
            Host::EqualDword
        },
        value,
        zero,
    );
    if format == Format::Double {
        super::shuffle(code, temporary, value, ADJACENT_DWORDS);
        packed(code, Host::And, value, temporary);
    }
}

fn vector_zero_guard(code: &mut Code, format: Format, shape: Shape, helper: &mut Vec<usize>) {
    prepare_vector_input_range(code, format);
    vector_zero_or_normal_guard(code, format, shape, LEFT_VECTOR, helper);
}

/// Send `source` to the helper unless every active lane is normal, a zero
/// or an infinity, after [`prepare_vector_input_range`]. Uses TEMP_VECTOR
/// and MERGE_VECTOR.
fn vector_zero_or_normal_guard(
    code: &mut Code,
    format: Format,
    shape: Shape,
    source: u8,
    helper: &mut Vec<usize>,
) {
    // The shared range leaves per-lane normal validity in EXPONENT_VECTOR.
    // OR fraction-zero validity into it: only zeros/infinities are newly
    // admitted, and none can change host MXCSR or guest FPSR.
    vector_exponent_range(code, format, shape, source);
    packed(code, Host::MoveAligned, TEMP_VECTOR, source);
    shift_left(format).emit(
        code,
        TEMP_VECTOR,
        (format.bits() - format.fraction_bits()) as u8,
    );
    equal_zero(
        code,
        format,
        TEMP_VECTOR,
        MINIMUM_EXPONENT_VECTOR,
        MERGE_VECTOR,
    );
    packed(code, Host::Or, EXPONENT_VECTOR, TEMP_VECTOR);
    if format == Format::Half {
        Host::ByteSignMask.register(code, 0, EXPONENT_VECTOR, TargetCpu::Baseline);
    } else {
        Host::DwordSignMask.register(code, 0, EXPONENT_VECTOR, TargetCpu::Baseline);
    }
    let mask = active_mask(format, shape) as i32;
    let mut asm = code.asm();
    asm.alu_accumulator_imm32(Alu::And, Size::Dword, mask);
    asm.alu_accumulator_imm32(Alu::Cmp, Size::Dword, mask);
    helper.push(branch(code, Condition::NotZero));
}

/// Signed order key for nonzero normal half values. Sign-complementing the
/// fraction/exponent reverses negative magnitudes without a host FP operation.
fn half_order_key(code: &mut Code, vector: u8) {
    packed(code, Host::MoveAligned, TEMP_VECTOR, vector);
    ShiftOpcode::ArithmeticWord.emit(code, TEMP_VECTOR, HALF_SIGN_SHIFT);
    ShiftOpcode::RightWord.emit(code, TEMP_VECTOR, SIGN_BITS);
    packed(code, Host::Xor, vector, TEMP_VECTOR);
}

/// `register` takes the low Half of `vector` as a signed integer of its
/// magnitude: both zeros become 0, and finite values and infinities keep
/// their order. Uses EDX.
fn half_signed_magnitude(code: &mut Code, register: Gpr, vector: u8) {
    const MAGNITUDE: i32 = 0x7fff;
    let mut asm = code.asm();
    asm.movd_from_xmm(register, vector);
    asm.mov(Size::Dword, Gpr::Rdx, register);
    asm.alu_imm32(Alu::And, Size::Dword, register, MAGNITUDE);
    // EDX becomes all ones for a negative value, then negates the magnitude.
    asm.shift_imm(ShiftOp::Shl, Size::Dword, Gpr::Rdx, HALF_SIGN_TO_DWORD_TOP);
    asm.shift_imm(ShiftOp::Sar, Size::Dword, Gpr::Rdx, SINGLE_SIGN_SHIFT);
    asm.alu(Alu::Xor, Size::Dword, register, Gpr::Rdx);
    asm.alu(Alu::Sub, Size::Dword, register, Gpr::Rdx);
}

/// EAX takes EDX's flags when `condition` holds.
fn select_flags(code: &mut Code, condition: Cond) {
    code.asm().cmov(condition, Size::Dword, Gpr::Rax, Gpr::Rdx);
}

fn store_flags(code: &mut Code) {
    flag_word::store(&mut code.asm(), Gpr::Rax);
}

/// The flags of an ordered comparison, as `code` holds them: greater, less
/// and equal.
fn ordered_words(code: &Code) -> [u32; 3] {
    [NZCV_C, NZCV_N, NZCV_Z | NZCV_C].map(|flags| flag_word::constant(code, flags))
}

fn ordered_flags(code: &mut Code, integer: bool) {
    let [greater, less, equal] = ordered_words(code);
    let mut asm = code.asm();
    asm.mov_imm32(Gpr::Rax, greater);
    asm.mov_imm32(Gpr::Rdx, less);
    select_flags(code, if integer { Cond::L } else { Cond::B });
    code.asm().mov_imm32(Gpr::Rdx, equal);
    select_flags(code, Cond::E);
    store_flags(code);
}

fn zero_flags(code: &mut Code, format: Format) {
    move_scalar(code, format, Gpr::Rcx, LEFT_VECTOR);
    let [greater, less, equal] = ordered_words(code);
    let mut asm = code.asm();
    asm.test(lane_size(format), Gpr::Rcx, Gpr::Rcx);
    asm.mov_imm32(Gpr::Rax, greater);
    asm.mov_imm32(Gpr::Rdx, less);
    select_flags(code, Cond::S);
    // Shift off the sign at the lane width; both +0 and -0 set ZF.
    let mut asm = code.asm();
    asm.shift_imm(ShiftOp::Shl, lane_size(format), Gpr::Rcx, 1);
    asm.mov_imm32(Gpr::Rdx, equal);
    select_flags(code, Cond::E);
    store_flags(code);
}

/// Whether FCMP/FCMPE/FCCMP of S/D registers take [`quick_flags`]: not in
/// the DIT profile, which keeps its reviewed classification.
fn quick_flags_supported(format: Format, right: CompareSource, target: TargetCpu) -> bool {
    format != Format::Half
        && matches!(right, CompareSource::Register(_))
        && !target.data_independent_timing()
}

/// With no trap enabled and FZ, FIZ and AH clear, UCOMIS orders every
/// non-NaN pair (zeros of either sign, subnormals, infinities) as Arm does
/// and Arm raises no flag: only an unordered pair (a NaN, whose
/// IOC depends on FCMP or FCMPE and on signalling) needs the helper. UCOMIS
/// may set MXCSR status bits (IE for an SNaN, DE for a subnormal), which
/// nothing reads: every status reader installs a clean control first.
/// Returns the branch, taken with UCOMIS's flags, to the flag mapping;
/// every other control falls through to the full classification.
fn quick_flags(
    code: &mut Code,
    format: Format,
    left: u8,
    right: u8,
    helper: &mut Vec<usize>,
) -> usize {
    use crate::aarch64::floating::{FPCR_AH, FPCR_FIZ, FPCR_FZ};
    test_control(
        code,
        FPCR_EXCEPTION_ENABLE_MASK | FPCR_FZ | FPCR_FIZ | FPCR_AH,
    );
    let other = branch(code, Condition::NotZero);
    read_vector(code, left, LEFT_VECTOR);
    read_vector(code, right, RIGHT_VECTOR);
    if format == Format::Double {
        code.asm().ucomisd(LEFT_VECTOR, Rm::Xmm(RIGHT_VECTOR));
    } else {
        code.asm().ucomiss(LEFT_VECTOR, Rm::Xmm(RIGHT_VECTOR));
    }
    helper.push(code.asm().jcc(Cond::P).offset());
    let ordered = jump(code);
    let full = code.len();
    resolve(code, other, full);
    ordered
}

fn emit_flags(
    code: &mut Code,
    format: Format,
    left: u8,
    right: CompareSource,
    pc: GuestAddress,
    target: TargetCpu,
) {
    let mut helper = Vec::new();
    let quick = match right {
        CompareSource::Register(register) if quick_flags_supported(format, right, target) => {
            Some(quick_flags(code, format, left, register, &mut helper))
        }
        _ => None,
    };
    exception_guard(code, &mut helper);
    read_vector(code, left, LEFT_VECTOR);
    if let CompareSource::Register(right) = right {
        read_vector(code, right, RIGHT_VECTOR);
        // UCOMIS, and the signed magnitudes of Half values, order zeros of
        // either sign and infinities with no host or guest flag; NaNs and
        // subnormals take the helper.
        for source in [LEFT_VECTOR, RIGHT_VECTOR] {
            scalar_zero_guard(code, format, source, &mut helper);
        }
        if format == Format::Half {
            half_signed_magnitude(code, Gpr::Rax, LEFT_VECTOR);
            half_signed_magnitude(code, Gpr::Rcx, RIGHT_VECTOR);
            code.asm().alu(Alu::Cmp, Size::Dword, Gpr::Rax, Gpr::Rcx);
        } else if format == Format::Double {
            code.asm().ucomisd(LEFT_VECTOR, Rm::Xmm(RIGHT_VECTOR));
        } else {
            code.asm().ucomiss(LEFT_VECTOR, Rm::Xmm(RIGHT_VECTOR));
        }
        // JMP and the unexecuted helper preserve the comparison flags.
        helper_section(code, &helper, pc);
        if let Some(quick) = quick {
            let mapping = code.len();
            resolve(code, quick, mapping);
        }
        ordered_flags(code, format == Format::Half);
    } else {
        scalar_zero_guard(code, format, LEFT_VECTOR, &mut helper);
        helper_section(code, &helper, pc);
        zero_flags(code, format);
    }
}

fn emit_conditional_flags(
    code: &mut Code,
    format: Format,
    (left, right): (u8, u8),
    condition: u8,
    fallback: u8,
    pc: GuestAddress,
    target: TargetCpu,
) {
    // A false condition bypasses numerical enables and every operand effect.
    let failed = super::super::condition_holds(code, condition, false)
        .map(|holds| code.asm().jcc(holds.negate()).offset());
    emit_flags(
        code,
        format,
        left,
        CompareSource::Register(right),
        pc,
        target,
    );
    if let Some(failed) = failed {
        let completed = jump(code);
        let fallback_start = code.len();
        resolve(code, failed, fallback_start);
        let fallback = flag_word::constant(code, u32::from(fallback) << NZCV_SHIFT);
        flag_word::store_constant(&mut code.asm(), fallback);
        let completed_target = code.len();
        resolve(code, completed, completed_target);
    }
}

fn absolute(operation: Predicate) -> bool {
    matches!(
        operation,
        Predicate::AbsoluteGreaterEqual | Predicate::AbsoluteGreater
    )
}

fn magnitude(code: &mut Code, format: Format) {
    packed(code, Host::EqualDword, TEMP_VECTOR, TEMP_VECTOR);
    shift_right(format).emit(code, TEMP_VECTOR, SIGN_BITS);
    packed(code, Host::And, LEFT_VECTOR, TEMP_VECTOR);
    packed(code, Host::And, RIGHT_VECTOR, TEMP_VECTOR);
}

fn invert_mask(code: &mut Code, vector: u8) {
    packed(code, Host::EqualDword, TEMP_VECTOR, TEMP_VECTOR);
    packed(code, Host::Xor, vector, TEMP_VECTOR);
}

fn half_predicate(code: &mut Code, operation: Predicate) -> u8 {
    if operation == Predicate::Equal {
        packed(code, Host::EqualWord, LEFT_VECTOR, RIGHT_VECTOR);
        return LEFT_VECTOR;
    }
    if absolute(operation) {
        magnitude(code, Format::Half);
    } else {
        half_order_key(code, LEFT_VECTOR);
        half_order_key(code, RIGHT_VECTOR);
    }
    // Right is in XMM0, left in XMM1.
    match operation {
        Predicate::Greater | Predicate::AbsoluteGreater => {
            packed(code, Host::GreaterWord, RIGHT_VECTOR, LEFT_VECTOR);
            RIGHT_VECTOR
        }
        Predicate::GreaterEqual | Predicate::AbsoluteGreaterEqual => {
            packed(code, Host::GreaterWord, LEFT_VECTOR, RIGHT_VECTOR);
            invert_mask(code, LEFT_VECTOR);
            LEFT_VECTOR
        }
        Predicate::Less => {
            packed(code, Host::GreaterWord, LEFT_VECTOR, RIGHT_VECTOR);
            LEFT_VECTOR
        }
        Predicate::LessEqual => {
            packed(code, Host::GreaterWord, RIGHT_VECTOR, LEFT_VECTOR);
            invert_mask(code, RIGHT_VECTOR);
            RIGHT_VECTOR
        }
        Predicate::Equal => unreachable!("equality needs no half order transform"),
    }
}

fn floating_predicate(code: &mut Code, operation: Predicate, format: Format, shape: Shape) -> u8 {
    if absolute(operation) {
        magnitude(code, format);
    }
    let (destination, source, predicate) = match operation {
        Predicate::Equal => (LEFT_VECTOR, RIGHT_VECTOR, COMPARE_EQUAL),
        Predicate::Greater | Predicate::AbsoluteGreater => {
            (LEFT_VECTOR, RIGHT_VECTOR, COMPARE_LESS)
        }
        Predicate::GreaterEqual | Predicate::AbsoluteGreaterEqual => {
            (LEFT_VECTOR, RIGHT_VECTOR, COMPARE_LESS_EQUAL)
        }
        Predicate::Less => (RIGHT_VECTOR, LEFT_VECTOR, COMPARE_LESS),
        Predicate::LessEqual => (RIGHT_VECTOR, LEFT_VECTOR, COMPARE_LESS_EQUAL),
    };
    let lanes = match (shape == Shape::Scalar, format == Format::Double) {
        (true, true) => FloatLanes::ScalarDouble,
        (true, false) => FloatLanes::ScalarSingle,
        (false, true) => FloatLanes::PackedDouble,
        (false, false) => FloatLanes::PackedSingle,
    };
    code.asm()
        .float_compare(lanes, destination, Rm::Xmm(source), predicate);
    destination
}

/// Zero predicates use signed magnitude tests instead of numerical host FP.
/// This also gives exact equality for both signs of zero at every lane width.
fn zero_predicate(code: &mut Code, operation: Predicate, format: Format) -> u8 {
    packed(code, Host::MoveAligned, ZERO_RESULT_VECTOR, LEFT_VECTOR);
    shift_left(format).emit(code, ZERO_RESULT_VECTOR, SIGN_BITS);
    shift_right(format).emit(code, ZERO_RESULT_VECTOR, SIGN_BITS);
    packed(code, Host::Xor, ZERO_VECTOR, ZERO_VECTOR);
    equal_zero(
        code,
        format,
        ZERO_RESULT_VECTOR,
        ZERO_VECTOR,
        QWORD_EQUAL_VECTOR,
    );
    if operation == Predicate::Equal {
        return ZERO_RESULT_VECTOR;
    }
    match format {
        Format::Half => ShiftOpcode::ArithmeticWord.emit(code, LEFT_VECTOR, HALF_SIGN_SHIFT),
        Format::Single => ShiftOpcode::ArithmeticDword.emit(code, LEFT_VECTOR, SINGLE_SIGN_SHIFT),
        Format::Double => {
            super::shuffle(code, LEFT_VECTOR, LEFT_VECTOR, QWORD_HIGH_DWORDS);
            ShiftOpcode::ArithmeticDword.emit(code, LEFT_VECTOR, SINGLE_SIGN_SHIFT);
        }
    }
    match operation {
        Predicate::Less => {
            packed(code, Host::AndNot, ZERO_RESULT_VECTOR, LEFT_VECTOR);
            ZERO_RESULT_VECTOR
        }
        Predicate::LessEqual => {
            packed(code, Host::Or, LEFT_VECTOR, ZERO_RESULT_VECTOR);
            LEFT_VECTOR
        }
        Predicate::Greater | Predicate::GreaterEqual => {
            packed(code, Host::EqualDword, ZERO_VECTOR, ZERO_VECTOR);
            packed(code, Host::Xor, LEFT_VECTOR, ZERO_VECTOR);
            if operation == Predicate::Greater {
                packed(code, Host::AndNot, ZERO_RESULT_VECTOR, LEFT_VECTOR);
                ZERO_RESULT_VECTOR
            } else {
                packed(code, Host::Or, LEFT_VECTOR, ZERO_RESULT_VECTOR);
                LEFT_VECTOR
            }
        }
        _ => unreachable!("decoded ordinary comparison with zero"),
    }
}

fn merge_scalar(code: &mut Code, format: Format, result: u8) -> u8 {
    if result != LEFT_VECTOR && format != Format::Half {
        packed(code, Host::MoveAligned, LEFT_VECTOR, result);
    }
    match format {
        Format::Half => {
            let mut asm = code.asm();
            asm.movd_from_xmm(Gpr::Rax, result);
            asm.pinsrw(MERGE_VECTOR, Gpr::Rax, 0);
        }
        Format::Single => {
            Host::MoveScalarSingle.register(code, MERGE_VECTOR, LEFT_VECTOR, TargetCpu::Baseline)
        }
        Format::Double => {
            Host::MoveScalarDouble.register(code, MERGE_VECTOR, LEFT_VECTOR, TargetCpu::Baseline)
        }
    }
    MERGE_VECTOR
}

fn commit_predicate(
    code: &mut Code,
    format: Format,
    shape: Shape,
    destination: u8,
    result: u8,
    merge: bool,
    zero: bool,
) {
    if shape == Shape::Scalar {
        let result = if merge {
            merge_scalar(code, format, result)
        } else {
            result
        };
        if zero {
            clear_scalar(code, format, result);
        } else {
            test_control(code, FPCR_NEP);
            let merging = branch(code, Condition::NotZero);
            clear_scalar(code, format, result);
            let merge_target = code.len();
            resolve(code, merging, merge_target);
        }
        commit(code, destination, result);
    } else {
        if shape == Shape::Vector64 {
            clear_scalar(code, Format::Double, result);
        }
        commit(code, destination, result);
    }
}

fn emit_predicate(code: &mut Code, operand: &Operand, pc: GuestAddress) {
    let Operand::Predicate {
        operation,
        format,
        shape,
        destination,
        left,
        right,
    } = *operand
    else {
        unreachable!("predicate comparison dispatch");
    };
    let mut helper = Vec::new();
    exception_guard(code, &mut helper);
    if shape != Shape::Scalar
        && operation == Predicate::Equal
        && right == CompareSource::Register(left)
    {
        // Normal values, both zero signs and infinities equal themselves
        // without changing FPSR. One guard suffices for the shared source.
        read_numerical_vector(code, shape, left, LEFT_VECTOR);
        vector_zero_guard(code, format, shape, &mut helper);
        helper_section(code, &helper, pc);
        packed(code, Host::EqualDword, LEFT_VECTOR, LEFT_VECTOR);
        commit_predicate(code, format, shape, destination, LEFT_VECTOR, false, false);
        return;
    }
    let zero = right == CompareSource::Zero;
    let merge = shape == Shape::Scalar
        && !zero
        && (format == Format::Half
            || absolute(operation)
            || matches!(operation, Predicate::Less | Predicate::LessEqual));
    if let CompareSource::Register(right) = right {
        // Reversed operands let ordered LT/LE implement GT/GE, and keep Vm's
        // upper bits for the scalar case.
        read_numerical_vector(code, shape, right, LEFT_VECTOR);
        read_numerical_vector(code, shape, left, RIGHT_VECTOR);
        if merge {
            packed(code, Host::MoveAligned, MERGE_VECTOR, LEFT_VECTOR);
        }
        // CMPSS/CMPPS order zeros of either sign and infinities as Arm
        // does, without a status; NaNs (IOC) and subnormals (FZ) take the
        // helper. The Half order key holds for normal values alone.
        if shape == Shape::Scalar {
            for source in [LEFT_VECTOR, RIGHT_VECTOR] {
                if format == Format::Half {
                    scalar_normal_guard(code, format, source, &mut helper);
                } else {
                    scalar_zero_guard(code, format, source, &mut helper);
                }
            }
        } else {
            prepare_vector_input_range(code, format);
            for source in [LEFT_VECTOR, RIGHT_VECTOR] {
                if format == Format::Half {
                    vector_exponent_range(code, format, shape, source);
                    helper.push(branch(code, Condition::NotZero));
                } else {
                    vector_zero_or_normal_guard(code, format, shape, source, &mut helper);
                }
            }
        }
    } else {
        read_numerical_vector(code, shape, left, LEFT_VECTOR);
        if shape == Shape::Scalar {
            scalar_zero_guard(code, format, LEFT_VECTOR, &mut helper);
        } else {
            vector_zero_guard(code, format, shape, &mut helper);
        }
    }
    let result = if zero {
        zero_predicate(code, operation, format)
    } else if format == Format::Half {
        half_predicate(code, operation)
    } else {
        floating_predicate(code, operation, format, shape)
    };
    helper_section(code, &helper, pc);
    commit_predicate(code, format, shape, destination, result, merge, zero);
}

pub(in crate::x86_64) fn emit(
    code: &mut Code,
    operand: &Operand,
    pc: GuestAddress,
    target: TargetCpu,
) {
    match *operand {
        Operand::Flags {
            format,
            left,
            right,
            ..
        } => emit_flags(code, format, left, right, pc, target),
        Operand::ConditionalFlags {
            format,
            left,
            right,
            condition,
            fallback,
            ..
        } => emit_conditional_flags(code, format, (left, right), condition, fallback, pc, target),
        Operand::Predicate { .. } => emit_predicate(code, operand, pc),
    }
}
