//! Conversions between FP and integers.
use super::numerical::jump;
use super::numerical::{
    ArithmeticOpcode, Condition, INPUT_MINIMUM_EXPONENT, MXCSR_INEXACT, Rounding, branch,
    clear_scalar, commit_inexact, emit_opcode, environment_round, exponent_range,
    prepare_vector_input_range, read_numerical_vector, reject_other_status, resolve,
    restore_environment, test_control, vector_exponent_range, vector_normal_or_zero_range,
};
use super::{Host, commit, packed, read_vector, vector_offset};
use crate::TargetCpu;
use crate::x86_64::asm::{Alu, Code, Gpr, Rm, ShiftOp, Size, vector_extensions};
use crate::x86_64::asm::{BitTest, FloatLanes};
use crate::x86_64::flag_word;
use crate::x86_64::isa::ShiftOpcode;
use crate::x86_64::simd::encode::guest;
use crate::{
    GuestAddress,
    aarch64::{
        Register,
        floating::{
            FPCR_EXCEPTION_ENABLE_MASK, FPCR_FIZ, FPCR_FZ, FPCR_NEP, FPCR_ROUNDING_MASK,
            FPCR_ROUNDING_SHIFT, Format, INEXACT,
            integer::{IntegerFormat, IntegerRounding, IntegerWidth},
        },
        simd::{
            arithmetic::Shape,
            conversion::{IntegerConversion, IntegerOperand},
        },
    },
    block::ExitKind,
};
use Gpr::{Rax, Rcx, Rdx};
use alloc::vec;
use alloc::vec::Vec;

const BROADCAST_LOW_SINGLE: u8 = 0;
const SIGN_BIT: u8 = 63;
const RESULT_VECTOR: u8 = 0;
const TEMPORARY_VECTOR: u8 = 1;
const INTEGER_ACCUMULATOR: u8 = 6;
const BITS_PER_BYTE: u32 = u8::BITS;
const SINGLE_EXPONENT_BIAS: i16 = 127;
const DOUBLE_EXPONENT_BIAS: i16 = 1023;
const UNSIGNED_HIGH_EXPONENT: i16 = 63;
const SINGLE_SUBNORMAL_CLASS_SHIFT: u8 = Format::Single.fraction_bits() as u8 + 1;

fn helper(code: &mut Code, pc: GuestAddress) {
    super::super::block::direct_exit(code, pc, ExitKind::FloatingPoint);
}

fn scalar_lanes(format: Format) -> FloatLanes {
    match format {
        Format::Single => FloatLanes::ScalarSingle,
        Format::Double => FloatLanes::ScalarDouble,
        Format::Half => unreachable!("half conversions use the exact core"),
    }
}

fn width(wide: bool) -> Size {
    if wide { Size::Qword } else { Size::Dword }
}

/// `cvtsi2ss/sd xmm0, eax/rax`.
fn to_float(code: &mut Code, format: Format, wide: bool) {
    code.asm()
        .cvtsi2float(scalar_lanes(format), width(wide), RESULT_VECTOR, Rax);
}

/// `cvt(t)ss/sd2si eax/rax, xmm0`.
fn from_float(code: &mut Code, format: Format, truncate: bool, wide: bool) {
    code.asm().cvtfloat2si(
        scalar_lanes(format),
        truncate,
        width(wide),
        Rax,
        Rm::Xmm(RESULT_VECTOR),
    );
}

/// Move the low `width` lane of `vector` into RAX/EAX.
fn lane_from_vector(code: &mut Code, wide: bool, vector: u8) {
    if wide {
        code.asm().movq_from_xmm(Rax, vector);
    } else {
        code.asm().movd_from_xmm(Rax, vector);
    }
}

/// Move RAX/EAX into the low lane of `vector`, zeroing the rest.
fn lane_to_vector(code: &mut Code, wide: bool, vector: u8) {
    if wide {
        code.asm().movq_to_xmm(vector, Rax);
    } else {
        code.asm().movd_to_xmm(vector, Rax);
    }
}

fn rounding(rounding: IntegerRounding) -> Rounding {
    match rounding {
        IntegerRounding::NearestEven => Rounding::NearestEven,
        IntegerRounding::TowardPositive => Rounding::Positive,
        IntegerRounding::TowardNegative => Rounding::Negative,
        IntegerRounding::TowardZero => Rounding::Zero,
        IntegerRounding::NearestAway => unreachable!("ties-away conversions use the exact core"),
    }
}

fn integer_offset(integer: IntegerOperand, lane: u32) -> u32 {
    let IntegerOperand::Simd {
        register, format, ..
    } = integer
    else {
        unreachable!("general register operands have no SIMD offset")
    };
    vector_offset(register) + lane * (format.width.bits() / BITS_PER_BYTE)
}

/// AVX512-FP16 converts Half in one rounding; a fixed-point scale would add
/// another.
pub(super) fn half_supported(operand: &IntegerConversion, target: TargetCpu) -> bool {
    vector_extensions::half_supported(target) && operand.frac_bits == 0
}

fn read_integer(code: &mut Code, integer: IntegerOperand, lane: u32) {
    let format = integer.format();
    match integer {
        IntegerOperand::General { register, .. } => super::super::load(code, register),
        IntegerOperand::Simd { .. } if format.width == IntegerWidth::Bits16 => {
            let element = guest(integer_offset(integer, lane));
            if format.signed {
                code.asm().movsx(Size::Dword, Rax, Size::Word, element);
            } else {
                code.asm().movzx(Size::Dword, Rax, Size::Word, element);
            }
        }
        IntegerOperand::Simd { .. } => {
            code.asm().mov_load(
                width(format.width == IntegerWidth::Bits64),
                Rax,
                guest(integer_offset(integer, lane)),
            );
        }
    }
    if !format.signed && format.width == IntegerWidth::Bits32 {
        // The shared GPR load is 64-bit; unsigned W conversion reads only Wn.
        code.asm().mov(Size::Dword, Rax, Rax);
    }
}

fn float_constant(code: &mut Code, format: Format, exponent: i16, shape: Shape) {
    let (bias, fraction_bits) = match format {
        Format::Single => (SINGLE_EXPONENT_BIAS, Format::Single.fraction_bits()),
        Format::Double => (DOUBLE_EXPONENT_BIAS, Format::Double.fraction_bits()),
        Format::Half => unreachable!("half conversions use the exact core"),
    };
    let mut asm = code.asm();
    asm.mov_imm64(
        Rdx,
        u64::try_from(bias + exponent).expect("normal conversion scale") << fraction_bits,
    );
    asm.movq_to_xmm(TEMPORARY_VECTOR, Rdx);
    if shape != Shape::Scalar {
        if format == Format::Single {
            asm.shufps(
                TEMPORARY_VECTOR,
                Rm::Xmm(TEMPORARY_VECTOR),
                BROADCAST_LOW_SINGLE,
            );
        } else {
            asm.unpcklpd(TEMPORARY_VECTOR, Rm::Xmm(TEMPORARY_VECTOR));
        }
    }
}

fn scale(code: &mut Code, operand: &IntegerConversion, shape: Shape, direction: i16) {
    if operand.frac_bits != 0 {
        float_constant(
            code,
            operand.format,
            direction * i16::from(operand.frac_bits),
            shape,
        );
        emit_opcode(
            code,
            operand.format,
            shape,
            ArithmeticOpcode::Multiply,
            TEMPORARY_VECTOR,
        );
    }
}

fn convert_integer_lane(
    code: &mut Code,
    operand: &IntegerConversion,
    lane: u32,
    target: TargetCpu,
) {
    let integer = operand.integer.format();
    read_integer(code, operand.integer, lane);
    if operand.format == Format::Half {
        let wide = integer.width == IntegerWidth::Bits64;
        if integer.signed {
            vector_extensions::half_from_signed(code, RESULT_VECTOR, wide, Rax);
        } else {
            vector_extensions::scalar_from_unsigned(
                code,
                RESULT_VECTOR,
                Format::Half,
                wide,
                Rax,
                target,
            );
        }
        return;
    }
    let wide = integer.width == IntegerWidth::Bits64 || !integer.signed;
    if integer.signed || integer.width != IntegerWidth::Bits64 {
        to_float(code, operand.format, wide);
    } else if !vector_extensions::scalar_from_unsigned(
        code,
        RESULT_VECTOR,
        operand.format,
        true,
        Rax,
        target,
    ) {
        // High unsigned values do not fit CVTSI's signed input. Retaining a
        // sticky low bit before halving preserves all four rounding modes.
        let mut asm = code.asm();
        asm.mov(Size::Qword, Rcx, Rax);
        asm.shift_imm(ShiftOp::Shr, Size::Qword, Rcx, SIGN_BIT);
        let low = branch(code, Condition::Zero);
        let mut asm = code.asm();
        asm.mov(Size::Qword, Rcx, Rax);
        asm.alu_imm8(Alu::And, Size::Dword, Rcx, 1);
        asm.shift_one(ShiftOp::Shr, Size::Qword, Rax);
        asm.alu(Alu::Or, Size::Qword, Rax, Rcx);
        to_float(code, operand.format, true);
        emit_opcode(
            code,
            operand.format,
            Shape::Scalar,
            ArithmeticOpcode::Add,
            RESULT_VECTOR,
        );
        let done = jump(code);
        let low_start = code.len();
        resolve(code, low, low_start);
        to_float(code, operand.format, true);
        let end = code.len();
        resolve(code, done, end);
    }
    // For S/D and fbits<=64, this power-of-two rescaling stays normal and exact.
    scale(code, operand, Shape::Scalar, -1);
}

/// SSE4.1 ROUNDSS/ROUNDSD immediate: round toward negative or positive
/// infinity by the immediate, not MXCSR, without the precision exception.
const ROUND_TOWARD_NEGATIVE: u8 = 0x9;
const ROUND_TOWARD_POSITIVE: u8 = 0xa;

/// Leave ECX as MXCSR status with only the inexact flag, set exactly when the
/// `size` values in RDX and `other` differ. Clobbers RDX.
fn inexact_unless_equal(code: &mut Code, size: Size, other: Gpr) {
    let mut asm = code.asm();
    asm.alu(Alu::Sub, size, Rdx, other);
    // CF is set exactly when the difference is zero.
    asm.alu_imm8(Alu::Cmp, size, Rdx, 1);
    asm.alu(Alu::Sbb, Size::Dword, Rcx, Rcx);
    asm.alu_imm8(Alu::Add, Size::Dword, Rcx, 1);
    asm.shift_imm(
        ShiftOp::Shl,
        Size::Dword,
        Rcx,
        MXCSR_INEXACT.trailing_zeros() as u8,
    );
}

/// The largest unbiased input exponent whose conversion stays inside the
/// integer: below 2^(n-1) for truncation, and one power lower for the rounding
/// forms, which can round up to the next power of two. Unsigned forms convert
/// with the signed 64-bit instruction (unsigned W results then fit 32 bits).
fn fast_exponent_limit(integer: IntegerFormat, truncate: bool) -> u32 {
    let magnitude_bits = match (integer.signed, integer.width) {
        (true, width) => width.bits() - 1,
        (false, IntegerWidth::Bits32) => 32,
        (false, _) => 63,
    };
    magnitude_bits - 1 - u32::from(!truncate)
}

/// The scalar S/D to W/X conversion without an MXCSR environment, taken
/// whatever FPSR holds: zeros, and normal and subnormal-free inputs whose
/// result fits the integer (`fast_exponent_limit`), with no enabled trap.
/// Then the only status is inexact, which a conversion back finds exactly:
/// the rounded integer of an S/D input in range is exactly representable in
/// its format. Truncation and nearest-even (the block ABI's MXCSR rounding,
/// see `numerical::sticky_inexact`) run on any tier; SSE4.1's ROUNDSS/SD
/// adds the directed forms. Ties away from zero (FCVTAS/FCVTAU) runs on any
/// tier: it truncates the exact double 2x to t and takes
/// (t + 1 + (t < 0 ? -1 : 0)) >> 1 arithmetically, which is x rounded half
/// away from zero; the rounding limit keeps 2x inside the integer. Returns
/// the branches to the full path and to its completion, which takes RAX and
/// ECX as the full path leaves them; `None` when the form has no fast path.
/// FJCVTZS (`negative_zero_exact` false) sends -0, whose Z flag differs, to
/// the full path.
fn fast_to_integer(
    code: &mut Code,
    operand: &IntegerConversion,
    mode: IntegerRounding,
    target: TargetCpu,
    negative_zero_exact: bool,
) -> Option<(Vec<usize>, Vec<usize>)> {
    let integer = operand.integer.format();
    let directed = matches!(
        mode,
        IntegerRounding::TowardPositive | IntegerRounding::TowardNegative
    );
    if operand.integer.shape() != Shape::Scalar
        || operand.frac_bits != 0
        || operand.format == Format::Half
        || integer.width == IntegerWidth::Bits16
        || directed && !target.supports(TargetCpu::Sse41)
        || target.data_independent_timing()
    {
        return None;
    }
    let format = operand.format;
    let double = format == Format::Double;
    let lanes = scalar_lanes(format);
    let input_size = Size::of(double);
    let truncate = mode == IntegerRounding::TowardZero;
    let result_size = width(!(integer.signed && integer.width == IntegerWidth::Bits32));
    let mut full = Vec::new();
    test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
    full.push(branch(code, Condition::NotZero));
    read_float_lane(code, operand, 0);
    let mut asm = code.asm();
    if double {
        asm.movq_from_xmm(Rcx, RESULT_VECTOR);
    } else {
        asm.movd_from_xmm(Rcx, RESULT_VECTOR);
    }
    asm.mov(input_size, Rdx, Rcx);
    asm.shift_one(ShiftOp::Shl, input_size, Rdx);
    if !negative_zero_exact {
        asm.test(input_size, Rcx, Rcx);
    }
    let zero = branch(code, Condition::Zero);
    let mut asm = code.asm();
    if !integer.signed {
        asm.bit_imm(BitTest::Bt, input_size, Rcx, (format.bits() - 1) as u8);
        full.push(branch(code, Condition::Below));
    }
    let mut asm = code.asm();
    asm.shift_imm(
        ShiftOp::Shr,
        input_size,
        Rdx,
        format.fraction_bits() as u8 + 1,
    );
    asm.alu_imm8(Alu::Cmp, Size::Dword, Rdx, 1);
    full.push(branch(code, Condition::Below));
    let bias = format.exponent_bias();
    code.asm().alu_imm32(
        Alu::Cmp,
        Size::Dword,
        Rdx,
        (bias + fast_exponent_limit(integer, truncate)) as i32,
    );
    full.push(branch(code, Condition::Above));
    let source = if directed {
        let opcode = if double {
            Host::RoundScalarDouble
        } else {
            Host::RoundScalarSingle
        };
        let immediate = if mode == IntegerRounding::TowardNegative {
            ROUND_TOWARD_NEGATIVE
        } else {
            ROUND_TOWARD_POSITIVE
        };
        opcode.register_imm(code, TEMPORARY_VECTOR, RESULT_VECTOR, immediate, target);
        TEMPORARY_VECTOR
    } else {
        RESULT_VECTOR
    };
    let away = mode == IntegerRounding::NearestAway;
    if away {
        // 2x is exact and normal: the input is normal and below the limit.
        emit_opcode(
            code,
            format,
            Shape::Scalar,
            ArithmeticOpcode::Add,
            RESULT_VECTOR,
        );
    }
    // Ties away works on 64 bits: t + 1 can leave a signed W result.
    code.asm().cvtfloat2si(
        lanes,
        truncate || directed || away,
        if away { Size::Qword } else { result_size },
        Rax,
        Rm::Xmm(source),
    );
    let mut asm = code.asm();
    if away {
        // RDX is -1 for a negative t and 0 otherwise.
        asm.mov(Size::Qword, Rdx, Rax);
        asm.shift_imm(ShiftOp::Sar, Size::Qword, Rdx, SIGN_BIT);
        asm.alu(Alu::Add, Size::Qword, Rax, Rdx);
        asm.alu_imm8(Alu::Add, Size::Qword, Rax, 1);
        asm.shift_one(ShiftOp::Sar, Size::Qword, Rax);
        if result_size == Size::Dword {
            asm.mov(Size::Dword, Rax, Rax);
        }
    }
    asm.cvtsi2float(lanes, result_size, TEMPORARY_VECTOR, Rax);
    if double {
        asm.movq_from_xmm(Rdx, TEMPORARY_VECTOR);
    } else {
        asm.movd_from_xmm(Rdx, TEMPORARY_VECTOR);
    }
    inexact_unless_equal(code, input_size, Rcx);
    let mut success = vec![jump(code)];
    let zero_start = code.len();
    resolve(code, zero, zero_start);
    let mut asm = code.asm();
    asm.alu(Alu::Xor, Size::Dword, Rax, Rax);
    asm.alu(Alu::Xor, Size::Dword, Rcx, Rcx);
    success.push(jump(code));
    Some((full, success))
}

/// The scalar W/X to S/D conversion without an MXCSR environment, under
/// FPCR's nearest-even rounding (the block ABI's MXCSR rounding) and no
/// enabled trap. Its only status is inexact, which the truncating
/// conversion back finds exactly: a result that rounded up to 2^31 or 2^63
/// converts back to the integer indefinite value, never the input. Unsigned
/// X inputs from 2^63 take the full path. Returns the branches to the full
/// path and to its completion, which takes XMM0 and ECX as the full path
/// leaves them.
fn fast_from_integer(
    code: &mut Code,
    operand: &IntegerConversion,
    target: TargetCpu,
) -> Option<(Vec<usize>, Vec<usize>)> {
    let integer = operand.integer.format();
    if operand.integer.shape() != Shape::Scalar
        || operand.frac_bits != 0
        || operand.format == Format::Half
        || integer.width == IntegerWidth::Bits16
        || target.data_independent_timing()
    {
        return None;
    }
    let lanes = scalar_lanes(operand.format);
    let size = width(integer.width == IntegerWidth::Bits64 || !integer.signed);
    let mut full = Vec::new();
    test_control(
        code,
        FPCR_EXCEPTION_ENABLE_MASK | FPCR_ROUNDING_MASK << FPCR_ROUNDING_SHIFT,
    );
    full.push(branch(code, Condition::NotZero));
    read_vector(code, operand.float, RESULT_VECTOR);
    read_integer(code, operand.integer, 0);
    if !integer.signed && integer.width == IntegerWidth::Bits64 {
        code.asm().bit_imm(BitTest::Bt, Size::Qword, Rax, SIGN_BIT);
        full.push(branch(code, Condition::Below));
    }
    let mut asm = code.asm();
    asm.cvtsi2float(lanes, size, RESULT_VECTOR, Rax);
    asm.cvtfloat2si(lanes, true, size, Rdx, Rm::Xmm(RESULT_VECTOR));
    inexact_unless_equal(code, size, Rax);
    Some((full, vec![jump(code)]))
}

fn pack_integer_lane(code: &mut Code, width: IntegerWidth, lane: u32) {
    lane_to_vector(code, width == IntegerWidth::Bits64, TEMPORARY_VECTOR);
    if lane != 0 {
        ShiftOpcode::LeftBytes.emit(
            code,
            TEMPORARY_VECTOR,
            (lane * (width.bits() / BITS_PER_BYTE)) as u8,
        );
    }
    packed(code, Host::Or, INTEGER_ACCUMULATOR, TEMPORARY_VECTOR);
}

fn finish_float(code: &mut Code, operand: &IntegerConversion) {
    commit_inexact(code);
    if operand.integer.shape() == Shape::Scalar {
        test_control(code, FPCR_NEP);
        let merge = branch(code, Condition::NotZero);
        clear_scalar(code, operand.format, RESULT_VECTOR);
        let end = code.len();
        resolve(code, merge, end);
    }
    commit(code, operand.float, RESULT_VECTOR);
}

fn word_to_double(
    code: &mut Code,
    operand: &IntegerConversion,
    register: Register,
    signed: bool,
    pc: GuestAddress,
) {
    // The shared FP completion refuses enabled exception traps. Keep that
    // boundary even though every W input converts exactly to binary64.
    test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
    let enabled = branch(code, Condition::NotZero);

    // Exact integer conversion changes neither MXCSR nor FPSR. Start with
    // a zero upper lane and read the old upper lane only for NEP merging.
    packed(code, Host::Xor, RESULT_VECTOR, RESULT_VECTOR);
    let mut asm = code.asm();
    match (signed, asm.guest(register)) {
        (true, Some(source)) => {
            asm.cvtsi2float(FloatLanes::ScalarDouble, Size::Dword, RESULT_VECTOR, source);
        }
        (_, source) => {
            if let Some(source) = source {
                asm.mov_load(Size::Dword, Rax, source);
            } else {
                asm.alu(Alu::Xor, Size::Dword, Rax, Rax);
            }
            asm.cvtsi2float(FloatLanes::ScalarDouble, Size::Qword, RESULT_VECTOR, Rax);
        }
    }
    test_control(code, FPCR_NEP);
    let no_merge = branch(code, Condition::Zero);
    code.asm()
        .movhps(RESULT_VECTOR, guest(vector_offset(operand.float) + 8));
    let commit_start = code.len();
    resolve(code, no_merge, commit_start);
    commit(code, operand.float, RESULT_VECTOR);
    let done = jump(code);
    let helper_start = code.len();
    resolve(code, enabled, helper_start);
    helper(code, pc);
    let end = code.len();
    resolve(code, done, end);
}

pub(in crate::x86_64) fn from_integer(
    code: &mut Code,
    operand: &IntegerConversion,
    pc: GuestAddress,
    target: TargetCpu,
) {
    if let IntegerOperand::General {
        register,
        format:
            IntegerFormat {
                width: IntegerWidth::Bits32,
                signed,
            },
    } = operand.integer
        && operand.format == Format::Double
        && operand.frac_bits == 0
        && !target.data_independent_timing()
    {
        word_to_double(code, operand, register, signed, pc);
        return;
    }
    if super::half_integer::try_from_integer(code, operand, pc, target) {
        return;
    }
    if operand.format == Format::Half && !half_supported(operand, target) {
        // S->H after a general integer->S can double round; there is no
        // baseline H CVT.
        helper(code, pc);
        return;
    }
    let fast = fast_from_integer(code, operand, target);
    if let Some((full, _)) = &fast {
        let full_start = code.len();
        for &patch in full {
            resolve(code, patch, full_start);
        }
    }
    let shape = operand.integer.shape();
    let integer = operand.integer.format();
    test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
    let enabled = branch(code, Condition::NotZero);
    if shape == Shape::Scalar {
        read_vector(code, operand.float, RESULT_VECTOR);
    }
    environment_round(code, Rounding::Fpcr);
    let signed_single = operand.format == Format::Single && integer.signed;
    if shape != Shape::Scalar
        && (signed_single
            || vector_extensions::packed_conversion_supported(operand.format, integer, target))
    {
        let IntegerOperand::Simd { register, .. } = operand.integer else {
            unreachable!("vector integer source")
        };
        read_numerical_vector(code, shape, register, RESULT_VECTOR);
        if signed_single {
            Host::ConvertPackedDwordToSingle.register(
                code,
                RESULT_VECTOR,
                RESULT_VECTOR,
                TargetCpu::Baseline,
            );
        } else {
            vector_extensions::packed_from_integer(
                code,
                RESULT_VECTOR,
                operand.format,
                integer,
                target,
            );
        }
        scale(code, operand, shape, -1);
    } else if shape == Shape::Scalar {
        convert_integer_lane(code, operand, 0, target);
    } else {
        packed(code, Host::Xor, INTEGER_ACCUMULATOR, INTEGER_ACCUMULATOR);
        for lane in 0..shape.lane_count(operand.format) {
            convert_integer_lane(code, operand, lane, target);
            lane_from_vector(code, operand.format == Format::Double, RESULT_VECTOR);
            pack_integer_lane(code, integer.width, lane);
        }
        packed(code, Host::MoveAligned, RESULT_VECTOR, INTEGER_ACCUMULATOR);
    }
    restore_environment(code);
    let status = reject_other_status(code);
    let success = jump(code);
    let helper_start = code.len();
    resolve(code, enabled, helper_start);
    resolve(code, status, helper_start);
    helper(code, pc);
    let success_start = code.len();
    resolve(code, success, success_start);
    for patch in fast.map(|(_, success)| success).unwrap_or_default() {
        resolve(code, patch, success_start);
    }
    finish_float(code, operand);
}

fn guard_float_input(code: &mut Code, operand: &IntegerConversion) -> usize {
    let shape = operand.integer.shape();
    if operand.format == Format::Half {
        // AVX512-FP16 converts either sign of zero exactly. Keep mixed zero
        // lanes inline and let the helper handle subnormals and specials.
        prepare_vector_input_range(code, Format::Half);
        vector_normal_or_zero_range(code, Format::Half, shape, RESULT_VECTOR, TEMPORARY_VECTOR);
        branch(code, Condition::NotZero)
    } else if shape == Shape::Scalar {
        exponent_range(code, operand.format, RESULT_VECTOR, INPUT_MINIMUM_EXPONENT);
        branch(code, Condition::AboveOrEqual)
    } else {
        prepare_vector_input_range(code, operand.format);
        vector_exponent_range(code, operand.format, shape, RESULT_VECTOR);
        branch(code, Condition::NotZero)
    }
}

/// Guard only the lanes whose conversion depends on Arm input flushing.
/// XMM0 holds the input and any inactive high lanes are already zero.
fn guard_single_subnormal(code: &mut Code) -> usize {
    let magnitude = TEMPORARY_VECTOR;
    let exponent = 2;
    let zero = 3;
    packed(code, Host::MoveAligned, magnitude, RESULT_VECTOR);
    // Drop the sign and test the magnitude and exponent independently.
    // Only exponent-zero lanes with a nonzero magnitude can flush.
    ShiftOpcode::LeftDword.emit(code, magnitude, 1);
    packed(code, Host::MoveAligned, exponent, magnitude);
    ShiftOpcode::RightDword.emit(code, exponent, SINGLE_SUBNORMAL_CLASS_SHIFT);
    packed(code, Host::Xor, zero, zero);
    packed(code, Host::EqualDword, magnitude, zero);
    packed(code, Host::EqualDword, exponent, zero);
    packed(code, Host::AndNot, magnitude, exponent);
    Host::DwordSignMask.register(code, 0, magnitude, TargetCpu::Baseline);
    code.asm().test(Size::Dword, Rax, Rax);
    branch(code, Condition::NotZero)
}

fn read_float_lane(code: &mut Code, operand: &IntegerConversion, lane: u32) {
    let element =
        guest(vector_offset(operand.float) + lane * (operand.format.bits() / BITS_PER_BYTE));
    if operand.format == Format::Double {
        Host::MoveQword.emit(code, RESULT_VECTOR, element, TargetCpu::Baseline);
    } else {
        code.asm().movd_to_xmm(RESULT_VECTOR, element);
    }
}

/// VCVT(T)SH2(U)SI of the low Half lane. Half's largest finite value fits a
/// dword, so an H integer checks the range of the dword result.
fn half_to_general(
    code: &mut Code,
    integer: IntegerFormat,
    truncate: bool,
    failures: &mut Vec<usize>,
    target: TargetCpu,
) {
    let wide = integer.width == IntegerWidth::Bits64;
    if integer.signed {
        vector_extensions::half_to_signed(code, Rax, RESULT_VECTOR, wide, truncate);
    } else {
        vector_extensions::scalar_to_unsigned(
            code,
            Rax,
            RESULT_VECTOR,
            Format::Half,
            wide,
            truncate,
            target,
        );
    }
    if integer.width == IntegerWidth::Bits16 {
        let mut asm = code.asm();
        asm.mov(Size::Dword, Rdx, Rax);
        if integer.signed {
            asm.alu_imm32(Alu::Add, Size::Dword, Rdx, i32::from(i16::MAX) + 1);
        }
        asm.shift_imm(ShiftOp::Shr, Size::Dword, Rdx, u16::BITS as u8);
        failures.push(branch(code, Condition::NotZero));
    }
}

fn convert_float_lane(
    code: &mut Code,
    operand: &IntegerConversion,
    mode: IntegerRounding,
    failures: &mut Vec<usize>,
    target: TargetCpu,
) {
    let integer = operand.integer.format();
    let truncate = mode == IntegerRounding::TowardZero;
    let wide = integer.width == IntegerWidth::Bits64;
    if operand.format == Format::Half {
        half_to_general(code, integer, truncate, failures, target);
    } else if integer.signed {
        from_float(code, operand.format, truncate, wide);
    } else if !vector_extensions::scalar_to_unsigned(
        code,
        Rax,
        RESULT_VECTOR,
        operand.format,
        wide,
        truncate,
        target,
    ) {
        if integer.width == IntegerWidth::Bits32 {
            from_float(code, operand.format, truncate, true);
            let mut asm = code.asm();
            asm.mov(Size::Qword, Rdx, Rax);
            asm.shift_imm(ShiftOp::Shr, Size::Qword, Rdx, u32::BITS as u8);
            asm.test(Size::Dword, Rdx, Rdx);
            failures.push(branch(code, Condition::NotZero));
        } else {
            float_constant(code, operand.format, UNSIGNED_HIGH_EXPONENT, Shape::Scalar);
            if operand.format == Format::Single {
                code.asm().ucomiss(RESULT_VECTOR, Rm::Xmm(TEMPORARY_VECTOR));
            } else {
                code.asm().ucomisd(RESULT_VECTOR, Rm::Xmm(TEMPORARY_VECTOR));
            }
            let low = branch(code, Condition::Below);
            // Subtraction is exact throughout [2^63,2^64) (Sterbenz). Anything
            // beyond that interval raises native invalid and is retried by the core.
            emit_opcode(
                code,
                operand.format,
                Shape::Scalar,
                ArithmeticOpcode::Subtract,
                TEMPORARY_VECTOR,
            );
            from_float(code, operand.format, truncate, true);
            code.asm().bit_imm(BitTest::Bts, Size::Qword, Rax, SIGN_BIT);
            let done = jump(code);
            let low_start = code.len();
            resolve(code, low, low_start);
            from_float(code, operand.format, truncate, true);
            code.asm().bit_imm(BitTest::Bt, Size::Qword, Rax, SIGN_BIT);
            failures.push(branch(code, Condition::Below));
            let end = code.len();
            resolve(code, done, end);
        }
    }
}

/// Whether ties-away FP-to-integer `operand` has an inline path on some
/// target (`fast_to_integer`); other forms always take the exact core.
pub(super) fn nearest_away_inline(operand: &IntegerConversion) -> bool {
    operand.integer.shape() == Shape::Scalar
        && operand.frac_bits == 0
        && operand.format != Format::Half
        && operand.integer.format().width != IntegerWidth::Bits16
}

fn finish_integer(code: &mut Code, operand: &IntegerConversion, packed_result: bool) {
    commit_inexact(code);
    match operand.integer {
        IntegerOperand::General { register, .. } => super::super::store(code, register),
        IntegerOperand::Simd {
            register,
            format,
            shape,
        } => {
            if shape != Shape::Scalar {
                if !packed_result {
                    packed(code, Host::MoveAligned, RESULT_VECTOR, INTEGER_ACCUMULATOR);
                }
            } else {
                read_vector(code, register, RESULT_VECTOR);
                if format.width == IntegerWidth::Bits16 {
                    code.asm().pinsrw(RESULT_VECTOR, Rax, 0);
                } else {
                    let wide = format.width == IntegerWidth::Bits64;
                    lane_to_vector(code, wide, TEMPORARY_VECTOR);
                    if wide {
                        Host::MoveScalarDouble
                    } else {
                        Host::MoveScalarSingle
                    }
                    .register(
                        code,
                        RESULT_VECTOR,
                        TEMPORARY_VECTOR,
                        TargetCpu::Baseline,
                    );
                }
                test_control(code, FPCR_NEP);
                let merge = branch(code, Condition::NotZero);
                clear_scalar(code, operand.format, RESULT_VECTOR);
                let end = code.len();
                resolve(code, merge, end);
            }
            commit(code, register, RESULT_VECTOR);
        }
    }
}

pub(in crate::x86_64) fn to_integer(
    code: &mut Code,
    operand: &IntegerConversion,
    mode: IntegerRounding,
    pc: GuestAddress,
    target: TargetCpu,
) {
    to_integer_with(code, operand, mode, pc, target, true);
}

fn to_integer_with(
    code: &mut Code,
    operand: &IntegerConversion,
    mode: IntegerRounding,
    pc: GuestAddress,
    target: TargetCpu,
    negative_zero_exact: bool,
) {
    if super::half_integer::try_to_integer(code, operand, mode, pc, target) {
        return;
    }
    if operand.format == Format::Half && !half_supported(operand, target) {
        helper(code, pc);
        return;
    }
    let fast = fast_to_integer(code, operand, mode, target, negative_zero_exact);
    if mode == IntegerRounding::NearestAway {
        // SSE2 has no ties-away control, and adding 0.5 before truncation
        // double rounds near half ties: inputs outside the fast path take
        // the exact integer-core route.
        let Some((full, success)) = fast else {
            helper(code, pc);
            return;
        };
        let helper_start = code.len();
        for patch in full {
            resolve(code, patch, helper_start);
        }
        helper(code, pc);
        let success_start = code.len();
        for patch in success {
            resolve(code, patch, success_start);
        }
        finish_integer(code, operand, false);
        return;
    }
    if let Some((full, _)) = &fast {
        let full_start = code.len();
        for &patch in full {
            resolve(code, patch, full_start);
        }
    }
    let shape = operand.integer.shape();
    let integer = operand.integer.format();
    let signed_single = operand.format == Format::Single && integer.signed;
    let packed_result = shape != Shape::Scalar
        && (signed_single
            || vector_extensions::packed_conversion_supported(operand.format, integer, target));
    let packed_truncate = signed_single
        && packed_result
        && mode == IntegerRounding::TowardZero
        && operand.frac_bits == 0;
    let mut helper_patches = Vec::new();
    let other_controls = if packed_truncate {
        read_numerical_vector(code, shape, operand.float, RESULT_VECTOR);
        // With input flushing disabled, CVTTPS2DQ reports inexact for
        // fractional inputs and invalid for NaNs or out-of-range values.
        // Its status distinguishes those cases before any guest commit.
        test_control(code, FPCR_EXCEPTION_ENABLE_MASK | FPCR_FIZ | FPCR_FZ);
        Some(branch(code, Condition::NotZero))
    } else {
        test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
        helper_patches.push(branch(code, Condition::NotZero));
        read_numerical_vector(code, shape, operand.float, RESULT_VECTOR);
        helper_patches.push(guard_float_input(code, operand));
        None
    };
    let convert = code.len();
    environment_round(code, rounding(mode));
    let mut environment_failures = Vec::new();
    if packed_result {
        scale(code, operand, shape, 1);
        if !signed_single {
            vector_extensions::packed_to_integer(
                code,
                RESULT_VECTOR,
                operand.format,
                integer,
                mode == IntegerRounding::TowardZero,
                target,
            );
        } else if mode == IntegerRounding::TowardZero {
            code.asm().cvttps2dq(RESULT_VECTOR, Rm::Xmm(RESULT_VECTOR));
        } else {
            Host::ConvertPackedSingleToDword.register(
                code,
                RESULT_VECTOR,
                RESULT_VECTOR,
                TargetCpu::Baseline,
            );
        }
    } else {
        if shape != Shape::Scalar {
            packed(code, Host::Xor, INTEGER_ACCUMULATOR, INTEGER_ACCUMULATOR);
        }
        for lane in 0..shape.lane_count(operand.format) {
            read_float_lane(code, operand, lane);
            scale(code, operand, Shape::Scalar, 1);
            convert_float_lane(code, operand, mode, &mut environment_failures, target);
            if shape != Shape::Scalar {
                pack_integer_lane(code, integer.width, lane);
            }
        }
    }
    restore_environment(code);
    helper_patches.push(reject_other_status(code));
    let success = jump(code);
    if !environment_failures.is_empty() {
        // Range failures occur before commit. Their cold path restores exactly
        // the same caller environment as the normal completion path.
        let restore_start = code.len();
        for patch in environment_failures {
            resolve(code, patch, restore_start);
        }
        restore_environment(code);
    }
    if let Some(other_controls) = other_controls {
        let check_controls = code.len();
        resolve(code, other_controls, check_controls);
        test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
        helper_patches.push(branch(code, Condition::NotZero));
        // Signed zero is safe under every flush mode. NaN and range errors
        // are still caught by CVTTPS2DQ's invalid status before commit.
        helper_patches.push(guard_single_subnormal(code));
        let resume = jump(code);
        resolve(code, resume, convert);
    }
    let helper_start = code.len();
    for patch in helper_patches {
        resolve(code, patch, helper_start);
    }
    helper(code, pc);
    let success_start = code.len();
    resolve(code, success, success_start);
    for patch in fast.map(|(_, success)| success).unwrap_or_default() {
        resolve(code, patch, success_start);
    }
    finish_integer(code, operand, packed_result);
}

pub(in crate::x86_64) fn javascript(
    code: &mut Code,
    destination: Register,
    source: u8,
    pc: GuestAddress,
    target: TargetCpu,
) {
    const NZCV_ZERO_SHIFT: u32 = 30;
    const INEXACT_TO_NZCV_SHIFT: u8 = (NZCV_ZERO_SHIFT - INEXACT.trailing_zeros()) as u8;
    let operand = IntegerConversion {
        integer: IntegerOperand::General {
            register: destination,
            format: IntegerFormat {
                width: IntegerWidth::Bits32,
                signed: true,
            },
        },
        float: source,
        format: Format::Double,
        frac_bits: 0,
    };
    // A normal input whose truncated value fits in 32 bits gives the JavaScript
    // result directly. Other inputs exit to software completion of FJCVTZS,
    // which handles wrapping, special values and zeros.
    to_integer_with(
        code,
        &operand,
        IntegerRounding::TowardZero,
        pc,
        target,
        false,
    );
    // commit_inexact leaves only fresh IXC in ECX. Accumulated FPSR.IXC must
    // not clear Z for a later exact input, including a discarded WZR result.
    let shift = match code.is_pinned() {
        true => (flag_word::ZERO.trailing_zeros() - INEXACT.trailing_zeros()) as u8,
        false => INEXACT_TO_NZCV_SHIFT,
    };
    let mut asm = code.asm();
    asm.alu_imm8(Alu::Xor, Size::Dword, Rcx, INEXACT as i8);
    asm.shift_imm(ShiftOp::Shl, Size::Dword, Rcx, shift);
    flag_word::store(&mut asm, Rcx);
}

#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux")
))]
#[test]
fn native_half_conversion_guard_admits_zero_and_checks_only_active_lanes() {
    use crate::{GuestState, x86_64::native::run};

    let source = 4;
    for (shape, lanes) in [
        (Shape::Scalar, 1),
        (Shape::Vector64, 4),
        (Shape::Vector128, 8),
    ] {
        let operand = IntegerConversion {
            integer: IntegerOperand::Simd {
                register: 3,
                format: IntegerFormat {
                    width: IntegerWidth::Bits16,
                    signed: true,
                },
                shape,
            },
            float: source,
            format: Format::Half,
            frac_bits: 0,
        };
        // The guard uses SSE2, so an AMD host can verify this part of the
        // AVX512-FP16 conversion path even without the FP16 conversion itself.
        let mut code = Code::new();
        read_numerical_vector(&mut code, shape, source, RESULT_VECTOR);
        let failure = guard_float_input(&mut code, &operand);
        code.asm().mov_imm32(Rax, 1);
        code.asm().ret();
        let rejected = code.len();
        resolve(&mut code, failure, rejected);
        code.asm().mov_imm32(Rax, 0);

        for (lane, input, expected) in [
            (0, 0, 1),
            (0, 0x8000, 1),
            (0, 0x0001, 0),
            (0, 0x8001, 0),
            (0, 0x7c00, 0),
            (0, 0x7e01, 0),
            (7, 0x7c00, u64::from(lanes < 8)),
        ] {
            let mut input_lanes = [0x0400u16, 0x8000, 0, 0x3c00, 0x4a1f, 0, 0x06b6, 0];
            input_lanes[lane] = input;
            let mut state = GuestState::default();
            for (slot, lanes) in state.v[usize::from(source)]
                .iter_mut()
                .zip(input_lanes.chunks_exact(4))
            {
                *slot = lanes.iter().enumerate().fold(0, |limb, (index, &value)| {
                    limb | u64::from(value) << (16 * index)
                });
            }
            assert_eq!(
                run(&code, &mut state),
                expected,
                "{shape:?} {input_lanes:x?}"
            );
        }
    }
}

#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux")
))]
#[test]
fn exact_word_to_double_keeps_source_rows_status_and_nep() {
    use crate::{
        GuestState, Options,
        aarch64::floating::{FPCR_AH, FPCR_ROUNDING_SHIFT},
        block::{BlockProfile, CompilationPlan},
        x86_64::native::run,
    };

    let pc = GuestAddress(0x1000120c0);
    let target = TargetCpu::Baseline;
    let compile = |word: u32, target| {
        let source = [word, 0xd65f_03c0].map(u32::to_le_bytes).concat();
        CompilationPlan::new(&source, pc, &Options::new(BlockProfile::ORDINARY, target))
            .unwrap()
            .compile()
            .unwrap()
            .native
    };
    let uses_environment = |code: &[u8]| code.windows(2).any(|word| word == [0x0f, 0xae]);
    for (word, source, destination, signed) in [
        (0x1e62_0262, 19, 2, true), // SCVTF D2,W19 from Geekbench.
        (0x1e63_0120, 9, 0, false), // UCVTF D0,W9 from Geekbench.
        (0x1e62_03e2, 31, 2, true),
        (0x1e63_03e0, 31, 0, false),
    ] {
        let native = compile(word, target);
        assert_eq!(native.source_map.len(), 2);
        assert_eq!(native.source_map[0], (pc, 0));
        assert_eq!(native.source_map[1].0, GuestAddress(pc.0 + 4));
        assert!(!uses_environment(&native.x86_64_code));
        let dit = compile(word, target.with_data_independent_timing());
        assert!(uses_environment(&dit.x86_64_code));
        for bits in [0u32, 1, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff] {
            for nep in [0, FPCR_NEP] {
                for rounding in 0..4 {
                    let mut initial = GuestState {
                        fpcr: nep
                            | FPCR_FIZ
                            | FPCR_FZ
                            | FPCR_AH
                            | (rounding << FPCR_ROUNDING_SHIFT),
                        fpsr: 0x0800_009f,
                        nzcv: 0xb000_0000,
                        ..GuestState::default()
                    };
                    initial.x[30] = 0x4000;
                    if source != 31 {
                        initial.x[source] = 0xdead_beef_0000_0000 | u64::from(bits);
                    }
                    initial.v[destination] = [u64::MAX, 0x1234_5678_9abc_def0];
                    let mut expected = initial.clone();
                    let value = if source == 31 { 0 } else { bits };
                    let result = if signed {
                        f64::from(value as i32)
                    } else {
                        f64::from(value)
                    };
                    expected.v[destination][0] = result.to_bits();
                    if nep == 0 {
                        expected.v[destination][1] = 0;
                    }
                    let mut actual = initial.clone();
                    assert_eq!(run(&native.x86_64_code, &mut actual), 0x4000);
                    assert_eq!(actual, expected);

                    initial.fpcr |= FPCR_EXCEPTION_ENABLE_MASK;
                    let mut actual = initial.clone();
                    assert_eq!(run(&native.x86_64_code, &mut actual), pc.0);
                    assert_eq!(actual, initial);
                }
            }
        }
    }
    for word in [0x1e22_0262, 0x9e62_0262, 0x1e42_c120] {
        assert!(uses_environment(&compile(word, target).x86_64_code));
    }
}
