//! FRECPE/FRSQRTE/FRECPX through exact integer arithmetic.
use super::super::isa::ShiftOpcode;
use super::encode::guest;
use super::numerical::helper_section;
use super::numerical::{
    ArithmeticOpcode, Condition, INPUT_MINIMUM_EXPONENT, Rounding, branch, clear_scalar,
    emit_opcode, environment_round, exponent_range, exponent_range_below, jump,
    positive_exponent_range, prepare_vector_input_range, prepare_vector_input_range_below,
    read_numerical_vector, resolve, restore_environment, test_control, vector_exponent_range,
    vector_positive_exponent_range,
};
use super::{Host, commit, packed, read_vector};
use crate::x86_64::asm::{Alu, Code, FloatLanes, Gpr, Mem, Rm, ShiftOp, Size, Unary};
use crate::{
    GuestAddress, TargetCpu,
    aarch64::{
        floating::{
            DIVIDE_BY_ZERO, FPCR_AH, FPCR_DN, FPCR_EXCEPTION_ENABLE_MASK, FPCR_FIZ, FPCR_FZ,
            FPCR_NEP, Format, INPUT_DENORMAL, INVALID, UNDERFLOW,
        },
        simd::{
            arithmetic::Shape,
            estimate::{Kind, Operand},
        },
    },
};
use Gpr::{R8, R9, R10, R11, Rax, Rcx, Rdx};
use alloc::vec::Vec;

const RESULT: u8 = 0;
const SOURCE: u8 = 1;
const DENOMINATOR: u8 = 2;
const TEMP: u8 = 3;
const CONSTANT: u8 = 4;
const EXPONENT: u8 = 5;
const COMPACT: u8 = 6;
const SAVED: u8 = 7;
const ESTIMATE_FRACTION_BITS: u8 = 8;
const ESTIMATE_UNIT: u32 = 1 << ESTIMATE_FRACTION_BITS;
const RECIPROCAL_QUOTIENT_EXPONENT: u32 = 2 * ESTIMATE_FRACTION_BITS as u32 + 2;
const RECIPROCAL_SQRT_QUOTIENT_EXPONENT: u32 = 3 * ESTIMATE_FRACTION_BITS as u32 + 2;
const SIGN_BITS: u8 = 1;
const DWORD_BITS: u8 = u32::BITS as u8;
const HALF_BITS: u8 = u16::BITS as u8;
const UNSIGNED_TABLE_SHIFT: u8 = DWORD_BITS - ESTIMATE_FRACTION_BITS - SIGN_BITS;
const UPPER_HALF_BYTES: u8 = (u64::BITS / u8::BITS) as u8;
const COMPACT_QWORD_LOW_DWORDS: u8 = 0x88;
const BROADCAST_LOW_DWORD: u8 = 0;
const FIRST_LANE: u8 = 0;
const ACCUMULATOR_GPR: u8 = 0;

/// Move the low `format` lane of `vector` into RAX/EAX.
fn lane_to_rax(code: &mut Code, format: Format, vector: u8) {
    if format == Format::Double {
        code.asm().movq_from_xmm(Rax, vector);
    } else {
        code.asm().movd_from_xmm(Rax, vector);
    }
}

fn reciprocal_sqrt(kind: Kind) -> bool {
    matches!(
        kind,
        Kind::ReciprocalSquareRoot | Kind::UnsignedReciprocalSquareRoot
    )
}

fn exponent_bits(format: Format) -> u8 {
    (format.bits() - format.fraction_bits()) as u8 - SIGN_BITS
}

fn fraction_shift(format: Format) -> u8 {
    format.fraction_bits() as u8 - ESTIMATE_FRACTION_BITS
}

fn checked_reciprocal(operand: &Operand, target: TargetCpu) -> bool {
    operand.kind == Kind::Reciprocal
        && operand.shape != Shape::Scalar
        && operand.format != Format::Double
        && target.supports(TargetCpu::Sse41)
}

/// Exact table rounding: r=(2^18+a)/(2a+1), a=256+fraction_top8.
/// The odd denominator cannot tie, so integer DIV needs no FP environment.
fn scalar_reciprocal(code: &mut Code, format: Format) {
    let fraction_mask = (ESTIMATE_UNIT - 1) as i32;
    let fraction_bits = format.fraction_bits() as u8;
    let sign_bit = format.bits() as u8 - SIGN_BITS;
    lane_to_rax(code, format, SOURCE);
    let mut asm = code.asm();
    asm.mov(Size::Qword, R8, Rax);
    asm.shift_imm(ShiftOp::Shr, Size::Qword, Rax, fraction_shift(format));
    asm.alu_accumulator_imm32(Alu::And, Size::Dword, fraction_mask);
    asm.alu_accumulator_imm32(Alu::Or, Size::Dword, ESTIMATE_UNIT as i32);
    // ECX = 2a+1.
    asm.lea(Size::Dword, Rcx, Mem::indexed(Rax, Rax, 1, 1));
    asm.alu_accumulator_imm32(
        Alu::Add,
        Size::Dword,
        (1u32 << RECIPROCAL_QUOTIENT_EXPONENT) as i32,
    );
    asm.alu(Alu::Xor, Size::Dword, Rdx, Rdx);
    asm.unary(Unary::Div, Size::Dword, Rcx);
    asm.alu_accumulator_imm32(Alu::And, Size::Dword, fraction_mask);
    asm.shift_imm(ShiftOp::Shl, Size::Qword, Rax, fraction_shift(format));
    asm.mov(Size::Qword, R9, Rax);
    asm.mov(Size::Qword, Rax, R8);
    asm.shift_imm(ShiftOp::Shr, Size::Qword, Rax, fraction_bits);
    asm.alu_accumulator_imm32(
        Alu::And,
        Size::Dword,
        (2 * format.exponent_bias() + 1) as i32,
    );
    asm.mov_imm32(Rcx, 2 * format.exponent_bias() - 1);
    asm.alu(Alu::Sub, Size::Dword, Rcx, Rax);
    asm.shift_imm(ShiftOp::Shl, Size::Qword, Rcx, fraction_bits);
    asm.alu(Alu::Or, Size::Qword, R9, Rcx);
    asm.shift_imm(ShiftOp::Shr, Size::Qword, R8, sign_bit);
    asm.shift_imm(ShiftOp::Shl, Size::Qword, R8, sign_bit);
    asm.alu(Alu::Or, Size::Qword, R9, R8);
    asm.movq_to_xmm(RESULT, R9);
}

fn dword_constant(code: &mut Code, vector: u8, value: u32) {
    let mut asm = code.asm();
    asm.mov_imm32(Rax, value);
    asm.movd_to_xmm(vector, Rax);
    super::shuffle(code, vector, vector, BROADCAST_LOW_DWORD);
}

/// Whether inputs outside the table's domain take [`special_lanes`]
/// rather than the helper.
fn special_supported(operand: &Operand) -> bool {
    matches!(operand.kind, Kind::Reciprocal | Kind::ReciprocalSquareRoot)
        && operand.format != Format::Half
}

fn source_guard(
    code: &mut Code,
    operand: &Operand,
    helper: &mut Vec<usize>,
    special: &mut Vec<usize>,
) {
    let to_special = special_supported(operand);
    if !operand.kind.is_unsigned() {
        // AH suppresses every estimate exception, including kept enables.
        test_control(code, FPCR_AH);
        let suppressed = branch(code, Condition::NotZero);
        test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
        helper.push(branch(code, Condition::NotZero));
        let accepted = code.len();
        resolve(code, suppressed, accepted);
    }
    read_numerical_vector(code, operand.shape, operand.source, SOURCE);
    if operand.kind.is_unsigned() {
        return;
    }
    let maximum = 2 * operand.format.exponent_bias() - 1;
    if operand.shape == Shape::Scalar {
        if operand.kind == Kind::Reciprocal {
            exponent_range_below(
                code,
                operand.format,
                SOURCE,
                INPUT_MINIMUM_EXPONENT,
                maximum,
            );
        } else if reciprocal_sqrt(operand.kind) {
            positive_exponent_range(code, operand.format, SOURCE, INPUT_MINIMUM_EXPONENT);
        } else {
            exponent_range(code, operand.format, SOURCE, INPUT_MINIMUM_EXPONENT);
        }
        let range = branch(code, Condition::AboveOrEqual);
        if to_special {
            special.push(range)
        } else {
            helper.push(range)
        }
    } else {
        if operand.kind == Kind::Reciprocal {
            // Exclude the top two normal source exponents: their reciprocal
            // can be subnormal or flushed and belongs to precise handling.
            prepare_vector_input_range_below(code, operand.format, maximum);
        } else {
            prepare_vector_input_range(code, operand.format);
        }
        if reciprocal_sqrt(operand.kind) {
            vector_positive_exponent_range(code, operand.format, operand.shape, SOURCE);
        } else {
            vector_exponent_range(code, operand.format, operand.shape, SOURCE);
        }
        let range = branch(code, Condition::NotZero);
        if to_special {
            special.push(range)
        } else {
            helper.push(range)
        }
    }
}

/// Keep sign, biased exponent and the top eight fraction bits in each dword.
/// D metadata is compacted before any host floating operation; H expands first.
fn compact_source(code: &mut Code, format: Format, upper: bool) {
    packed(code, Host::MoveAligned, COMPACT, SOURCE);
    if upper {
        ShiftOpcode::RightBytes.emit(code, COMPACT, UPPER_HALF_BYTES);
    }
    if format == Format::Half {
        packed(code, Host::Xor, TEMP, TEMP);
        packed(code, Host::UnpackLowWord, COMPACT, TEMP);
    }
    if format == Format::Double {
        ShiftOpcode::RightQword.emit(code, COMPACT, fraction_shift(format));
        super::shuffle(code, COMPACT, COMPACT, COMPACT_QWORD_LOW_DWORDS);
    } else {
        ShiftOpcode::RightDword.emit(code, COMPACT, fraction_shift(format));
    }
}

fn result_exponent(code: &mut Code, format: Format, sqrt: bool) {
    let bits = exponent_bits(format);
    packed(code, Host::MoveAligned, TEMP, COMPACT);
    ShiftOpcode::LeftDword.emit(code, TEMP, DWORD_BITS - bits - ESTIMATE_FRACTION_BITS);
    ShiftOpcode::RightDword.emit(code, TEMP, DWORD_BITS - bits);
    let base = (if sqrt { 3 } else { 2 }) * format.exponent_bias() - 1;
    dword_constant(code, EXPONENT, base);
    packed(code, Host::SubDword, EXPONENT, TEMP);
    if sqrt {
        ShiftOpcode::RightDword.emit(code, EXPONENT, 1);
    }
    ShiftOpcode::LeftDword.emit(code, EXPONENT, ESTIMATE_FRACTION_BITS);
}

fn floating_denominator(code: &mut Code, sqrt: bool) {
    packed(code, Host::MoveAligned, DENOMINATOR, COMPACT);
    ShiftOpcode::LeftDword.emit(code, DENOMINATOR, DWORD_BITS - ESTIMATE_FRACTION_BITS);
    ShiftOpcode::RightDword.emit(code, DENOMINATOR, DWORD_BITS - ESTIMATE_FRACTION_BITS);
    if sqrt {
        ShiftOpcode::RightDword.emit(code, DENOMINATOR, 1);
        ShiftOpcode::LeftDword.emit(code, DENOMINATOR, 1);
        dword_constant(code, CONSTANT, ESTIMATE_UNIT + 1);
        packed(code, Host::AddDword, DENOMINATOR, CONSTANT);
        // Y=(fraction_top8 & ~1)+257. Odd exponent uses A=Y;
        // even exponent uses A=2Y, matching both source table branches.
        packed(code, Host::MoveAligned, TEMP, COMPACT);
        ShiftOpcode::LeftDword.emit(code, TEMP, DWORD_BITS - SIGN_BITS - ESTIMATE_FRACTION_BITS);
        ShiftOpcode::ArithmeticDword.emit(code, TEMP, DWORD_BITS - SIGN_BITS);
        packed(code, Host::AndNot, TEMP, DENOMINATOR);
        packed(code, Host::AddDword, DENOMINATOR, TEMP);
    } else {
        ShiftOpcode::LeftDword.emit(code, DENOMINATOR, 1);
        dword_constant(code, CONSTANT, 2 * ESTIMATE_UNIT + 1);
        packed(code, Host::AddDword, DENOMINATOR, CONSTANT);
    }
}

/// Bounded denominators are exact Single integers. Primary-derived midpoint
/// bounds prove nearest(DIV[+SQRT]) followed by nearest integer conversion
/// reproduces every standard estimate key. Host PE is not guest estimate IXC.
fn integer_estimate(code: &mut Code, sqrt: bool) {
    packed(
        code,
        Host::ConvertPackedDwordToSingle,
        DENOMINATOR,
        DENOMINATOR,
    );
    let power = if sqrt {
        RECIPROCAL_SQRT_QUOTIENT_EXPONENT
    } else {
        RECIPROCAL_QUOTIENT_EXPONENT
    };
    let numerator = (Format::Single.exponent_bias() + power) << Format::Single.fraction_bits();
    dword_constant(code, RESULT, numerator);
    emit_opcode(
        code,
        Format::Single,
        Shape::Vector128,
        ArithmeticOpcode::Divide,
        DENOMINATOR,
    );
    if sqrt {
        packed(code, Host::SquareRootPackedSingle, RESULT, RESULT);
    }
    packed(code, Host::ConvertPackedSingleToDword, RESULT, RESULT);
}

/// RCP raises no exceptions. Its error bound makes floor(q') either r or r - 1;
/// the exact odd product A*(2*r0+1) decides the increment.
fn corrected_reciprocal(code: &mut Code, target: TargetCpu) {
    packed(code, Host::ConvertPackedDwordToSingle, RESULT, DENOMINATOR);
    packed(code, Host::ReciprocalPackedSingle, RESULT, RESULT);
    ShiftOpcode::RightDword.emit(code, RESULT, fraction_shift(Format::Single));
    let reciprocal_exponent = Format::Single.exponent_bias() - ESTIMATE_FRACTION_BITS as u32 - 2;
    let integer_bias = (reciprocal_exponent << ESTIMATE_FRACTION_BITS) - ESTIMATE_UNIT;
    dword_constant(code, CONSTANT, integer_bias);
    packed(code, Host::SubDword, RESULT, CONSTANT);
    packed(code, Host::MoveAligned, CONSTANT, RESULT);
    ShiftOpcode::LeftDword.emit(code, CONSTANT, 1);
    Host::MultiplyLowDword.register(code, CONSTANT, DENOMINATOR, target);
    packed(code, Host::AddDword, CONSTANT, DENOMINATOR);
    dword_constant(code, TEMP, 1 << (RECIPROCAL_QUOTIENT_EXPONENT + 1));
    packed(code, Host::GreaterDword, TEMP, CONSTANT);
    packed(code, Host::SubDword, RESULT, TEMP);
}

fn pack_half(code: &mut Code, signed: bool, target: TargetCpu) {
    if signed && target.supports(TargetCpu::Sse41) {
        Host::PackUnsignedDwordToWord.register(code, RESULT, RESULT, target);
    } else {
        if signed {
            // Reconstructed negative H raw bits exceed signedword range.
            // Sign-extend true16 bits before signed saturation packing.
            ShiftOpcode::LeftDword.emit(code, RESULT, DWORD_BITS - HALF_BITS);
            ShiftOpcode::ArithmeticDword.emit(code, RESULT, DWORD_BITS - HALF_BITS);
        }
        packed(code, Host::PackSignedDwordToWord, RESULT, RESULT);
    }
}

fn floating_group(code: &mut Code, operand: &Operand, upper: bool, target: TargetCpu) {
    let sqrt = reciprocal_sqrt(operand.kind);
    compact_source(code, operand.format, upper);
    result_exponent(code, operand.format, sqrt);
    floating_denominator(code, sqrt);
    if !sqrt {
        let sign = exponent_bits(operand.format) + ESTIMATE_FRACTION_BITS;
        ShiftOpcode::RightDword.emit(code, COMPACT, sign);
        ShiftOpcode::LeftDword.emit(code, COMPACT, sign);
    }
    if checked_reciprocal(operand, target) {
        corrected_reciprocal(code, target);
    } else {
        integer_estimate(code, sqrt);
    }
    ShiftOpcode::LeftDword.emit(code, RESULT, DWORD_BITS - ESTIMATE_FRACTION_BITS);
    ShiftOpcode::RightDword.emit(code, RESULT, DWORD_BITS - ESTIMATE_FRACTION_BITS);
    packed(code, Host::Or, RESULT, EXPONENT);
    if !sqrt {
        packed(code, Host::Or, RESULT, COMPACT);
    }
    if operand.format == Format::Double {
        packed(code, Host::Xor, TEMP, TEMP);
        packed(code, Host::UnpackLowDword, RESULT, TEMP);
        ShiftOpcode::LeftQword.emit(code, RESULT, fraction_shift(operand.format));
    } else {
        ShiftOpcode::LeftDword.emit(code, RESULT, fraction_shift(operand.format));
        if operand.format == Format::Half {
            pack_half(code, !sqrt, target);
        }
    }
}

fn floating_estimate(code: &mut Code, operand: &Operand, target: TargetCpu) {
    if operand.shape == Shape::Scalar {
        clear_scalar(code, operand.format, SOURCE);
    }
    if operand.kind == Kind::Reciprocal && operand.shape == Shape::Scalar {
        scalar_reciprocal(code, operand.format);
        return;
    }
    let checked = checked_reciprocal(operand, target);
    if !checked {
        environment_round(code, Rounding::NearestEven);
    }
    floating_group(code, operand, false, target);
    if operand.format == Format::Half && operand.shape == Shape::Vector128 {
        packed(code, Host::MoveAligned, SAVED, RESULT);
        floating_group(code, operand, true, target);
        packed(code, Host::UnpackLowQword, SAVED, RESULT);
        packed(code, Host::MoveAligned, RESULT, SAVED);
    }
    // The domain proves every host operation finite and safe for conversion.
    // Approximation PE is discarded; no guest status arises.
    if !checked {
        restore_environment(code);
    }
}

fn unsigned_estimate(code: &mut Code, kind: Kind, target: TargetCpu) {
    let sqrt = reciprocal_sqrt(kind);
    let checked = !sqrt && target.supports(TargetCpu::Sse41);
    if !checked {
        environment_round(code, Rounding::NearestEven);
    }
    packed(code, Host::MoveAligned, SAVED, SOURCE);
    if sqrt {
        ShiftOpcode::RightDword.emit(code, SAVED, DWORD_BITS - 2);
        packed(code, Host::Xor, TEMP, TEMP);
        packed(code, Host::GreaterDword, SAVED, TEMP);
    } else {
        ShiftOpcode::ArithmeticDword.emit(code, SAVED, DWORD_BITS - SIGN_BITS);
    }
    packed(code, Host::MoveAligned, DENOMINATOR, SOURCE);
    ShiftOpcode::RightDword.emit(code, DENOMINATOR, UNSIGNED_TABLE_SHIFT);
    dword_constant(code, CONSTANT, 1);
    if sqrt {
        packed(code, Host::MoveAligned, TEMP, SOURCE);
        ShiftOpcode::ArithmeticDword.emit(code, TEMP, DWORD_BITS - SIGN_BITS);
        packed(code, Host::MoveAligned, COMPACT, DENOMINATOR);
        ShiftOpcode::LeftDword.emit(code, COMPACT, DWORD_BITS - SIGN_BITS);
        ShiftOpcode::ArithmeticDword.emit(code, COMPACT, DWORD_BITS - SIGN_BITS);
        packed(code, Host::Or, COMPACT, CONSTANT); // even a -> +1; odd a -> -1
        packed(code, Host::And, COMPACT, TEMP); // only a>=256 adjusts A
    }
    ShiftOpcode::LeftDword.emit(code, DENOMINATOR, 1);
    packed(code, Host::AddDword, DENOMINATOR, CONSTANT);
    if sqrt {
        packed(code, Host::AddDword, DENOMINATOR, COMPACT);
    }
    if checked {
        // Invalid unsigned domains are saturated below. Clamp their A to
        // 513..1023 here; valid lanes already contain this denominator bit.
        ShiftOpcode::LeftDword.emit(code, CONSTANT, ESTIMATE_FRACTION_BITS + 1);
        packed(code, Host::Or, DENOMINATOR, CONSTANT);
        corrected_reciprocal(code, target);
    } else {
        integer_estimate(code, sqrt);
    }
    ShiftOpcode::LeftDword.emit(code, RESULT, UNSIGNED_TABLE_SHIFT);
    packed(code, Host::And, RESULT, SAVED);
    packed(code, Host::EqualDword, TEMP, TEMP);
    packed(code, Host::AndNot, SAVED, TEMP);
    packed(code, Host::Or, RESULT, SAVED);
    if !checked {
        restore_environment(code);
    }
}

fn reciprocal_exponent(code: &mut Code, format: Format) {
    lane_to_rax(code, format, SOURCE);
    let kept = format.sign_mask() | format.exponent_mask();
    let mut asm = code.asm();
    if format == Format::Double {
        asm.mov_imm64(Rdx, format.exponent_mask());
        asm.alu(Alu::Xor, Size::Qword, Rax, Rdx);
        asm.mov_imm64(Rcx, kept);
        asm.alu(Alu::And, Size::Qword, Rax, Rcx);
        packed(code, Host::MoveGprQword, RESULT, ACCUMULATOR_GPR);
    } else {
        asm.alu_accumulator_imm32(Alu::Xor, Size::Dword, format.exponent_mask() as i32);
        asm.alu_accumulator_imm32(Alu::And, Size::Dword, kept as i32);
        asm.movd_to_xmm(RESULT, Rax);
    }
}

const LESS: u8 = 1;

fn quiet_bit(format: Format) -> u64 {
    1 << (format.fraction_bits() - 1)
}

const LESS_OR_EQUAL: u8 = 2;
const UNORDERED: u8 = 3;
const NOT_EQUAL: u8 = 4;

/// One `format` lane's `bits` in every lane of `xmm`.
fn lane_constant(code: &mut Code, format: Format, xmm: u8, bits: u64) {
    if format == Format::Double {
        code.asm().mov_imm64(Rax, bits);
        packed(code, Host::MoveGprQword, xmm, ACCUMULATOR_GPR);
        packed(code, Host::UnpackLowQword, xmm, xmm);
    } else {
        dword_constant(code, xmm, bits as u32);
    }
}

/// Packed CMPPS/CMPPD `predicate` of `destination` against `source`. Its
/// MXCSR status is never guest state.
fn compare(code: &mut Code, format: Format, destination: u8, source: u8, predicate: u8) {
    let lanes = if format == Format::Double {
        FloatLanes::PackedDouble
    } else {
        FloatLanes::PackedSingle
    };
    code.asm()
        .float_compare(lanes, destination, Rm::Xmm(source), predicate);
}

/// The MOVMSKPS bits of the active lanes: a Double lane's high dword.
fn active_lanes(operand: &Operand) -> i32 {
    match (operand.format, operand.shape) {
        (Format::Double, Shape::Scalar) => 0b10,
        (Format::Double, _) => 0b1010,
        (_, Shape::Scalar) => 0b1,
        (_, Shape::Vector64) => 0b11,
        _ => 0b1111,
    }
}

/// Branch taken when no active lane of the lane mask `xmm` is set.
fn none_active(code: &mut Code, xmm: u8, active: i32) -> usize {
    Host::DwordSignMask.register(code, ACCUMULATOR_GPR, xmm, TargetCpu::Baseline);
    code.asm().test_imm(Size::Dword, Rax, active);
    branch(code, Condition::Zero)
}

/// Add FPSR `flag` to R11D's fresh flags where an active lane of `xmm` is set.
fn flag_if_active(code: &mut Code, xmm: u8, active: i32, flag: u32) {
    let none = none_active(code, xmm, active);
    code.asm().alu_imm32(Alu::Or, Size::Dword, R11, flag as i32);
    let next = code.len();
    resolve(code, none, next);
}

/// Where an active lane of `xmm` is set: take the helper unless FPCR (R10D)
/// has one of `accepted`, and add `flag` where it has `flagged`.
fn lanes_need_control(
    code: &mut Code,
    xmm: u8,
    active: i32,
    (accepted, flagged, flag): (u32, u32, u32),
    helper: &mut Vec<usize>,
) {
    let none = none_active(code, xmm, active);
    code.asm().test_imm(Size::Dword, R10, accepted as i32);
    helper.push(branch(code, Condition::Zero));
    code.asm().test_imm(Size::Dword, R10, flagged as i32);
    let quiet = branch(code, Condition::Zero);
    code.asm().alu_imm32(Alu::Or, Size::Dword, R11, flag as i32);
    let next = code.len();
    resolve(code, none, next);
    resolve(code, quiet, next);
}

/// FPCR's (R10D) default NaN in every lane of `xmm`: negative under AH.
fn default_nan(code: &mut Code, format: Format, xmm: u8) {
    let sign_shift = format.bits() as u8 - 1 - FPCR_AH.trailing_zeros() as u8;
    let quiet = format.exponent_mask() | quiet_bit(format);
    let size = Size::of(format == Format::Double);
    let mut asm = code.asm();
    asm.mov(size, Rax, R10);
    asm.alu_imm32(Alu::And, size, Rax, FPCR_AH as i32);
    asm.shift_imm(ShiftOp::Shl, size, Rax, sign_shift);
    asm.mov_imm64(Rcx, quiet);
    asm.alu(Alu::Or, size, Rax, Rcx);
    if format == Format::Double {
        packed(code, Host::MoveGprQword, xmm, ACCUMULATOR_GPR);
        packed(code, Host::UnpackLowQword, xmm, xmm);
    } else {
        code.asm().movd_to_xmm(xmm, Rax);
        super::shuffle(code, xmm, xmm, BROADCAST_LOW_DWORD);
    }
}

/// Each lane's NaN result in `xmm` (`quiet` holds the quiet bit in every
/// lane): the default NaN under DN, else the input (SAVED) made quiet.
fn nan_result(code: &mut Code, format: Format, xmm: u8, quiet: u8) {
    packed(code, Host::MoveAligned, xmm, SAVED);
    packed(code, Host::Or, xmm, quiet);
    code.asm().test_imm(Size::Dword, R10, FPCR_DN as i32);
    let kept = branch(code, Condition::Zero);
    default_nan(code, format, xmm);
    let next = code.len();
    resolve(code, kept, next);
}

/// FRECPE and FRSQRTE where some active lane is outside the table's
/// domain (zero, subnormal, infinity, NaN, FRSQRTE's negative inputs and
/// FRECPE's inputs whose estimate is subnormal): the table path runs with
/// those lanes replaced by one, and each takes Arm's result for its class,
/// with the cumulative flags its class raises (none under AH). Only a
/// subnormal input that no FZ, FIZ or AH flushes, and an FRECPE input whose
/// subnormal estimate no FZ or AH flushes, take the helper.
fn special_lanes(code: &mut Code, operand: &Operand, target: TargetCpu, helper: &mut Vec<usize>) {
    let format = operand.format;
    let sqrt = reciprocal_sqrt(operand.kind);
    let active = active_lanes(operand);
    let fraction_bits = format.fraction_bits();
    let magnitude_mask = format.sign_mask() - 1;
    let infinity = format.exponent_mask();
    let minimum = 1u64 << fraction_bits;
    let one = u64::from(format.exponent_bias()) << fraction_bits;
    // FRECPE's estimate is subnormal from this magnitude on.
    let subnormal_estimate = u64::from(2 * format.exponent_bias() - 1) << fraction_bits;
    if operand.shape == Shape::Scalar {
        clear_scalar(code, format, SOURCE);
    }
    packed(code, Host::MoveAligned, SAVED, SOURCE);
    // In the domain: minimum <= |x| < limit, for FRSQRTE on x itself so
    // that negative inputs leave it. NaN compares false.
    let magnitude = if sqrt {
        SOURCE
    } else {
        lane_constant(code, format, CONSTANT, magnitude_mask);
        packed(code, Host::MoveAligned, TEMP, SOURCE);
        packed(code, Host::And, TEMP, CONSTANT);
        TEMP
    };
    lane_constant(code, format, DENOMINATOR, minimum);
    compare(code, format, DENOMINATOR, magnitude, LESS_OR_EQUAL);
    let limit = if sqrt { infinity } else { subnormal_estimate };
    lane_constant(code, format, CONSTANT, limit);
    packed(code, Host::MoveAligned, COMPACT, magnitude);
    compare(code, format, COMPACT, CONSTANT, LESS);
    packed(code, Host::And, DENOMINATOR, COMPACT);
    lane_constant(code, format, CONSTANT, one);
    packed(code, Host::And, SOURCE, DENOMINATOR);
    packed(code, Host::AndNot, DENOMINATOR, CONSTANT);
    packed(code, Host::Or, SOURCE, DENOMINATOR);
    floating_estimate(code, operand, target);

    // RESULT holds the table results, SAVED the inputs.
    const MAGNITUDE: u8 = SOURCE;
    const ZERO_CLASS: u8 = DENOMINATOR;
    const NAN: u8 = CONSTANT;
    const SCRATCH: u8 = TEMP;
    const VALUE: u8 = EXPONENT;
    const LANES: u8 = COMPACT;
    lane_constant(code, format, SCRATCH, magnitude_mask);
    packed(code, Host::MoveAligned, MAGNITUDE, SAVED);
    packed(code, Host::And, MAGNITUDE, SCRATCH);
    lane_constant(code, format, SCRATCH, minimum);
    packed(code, Host::MoveAligned, ZERO_CLASS, MAGNITUDE);
    compare(code, format, ZERO_CLASS, SCRATCH, LESS);
    packed(code, Host::MoveAligned, NAN, SAVED);
    compare(code, format, NAN, SAVED, UNORDERED);
    {
        let mut asm = code.asm();
        asm.mov_load(
            Size::Dword,
            R10,
            guest(core::mem::offset_of!(crate::GuestState, fpcr) as u32),
        );
        asm.alu(Alu::Xor, Size::Dword, R11, R11);
    }
    // Subnormal inputs are zeros where flushed, IDC under FZ.
    packed(code, Host::Xor, LANES, LANES);
    compare(code, format, LANES, MAGNITUDE, NOT_EQUAL);
    packed(code, Host::And, LANES, ZERO_CLASS);
    lanes_need_control(
        code,
        LANES,
        active,
        (FPCR_FZ | FPCR_FIZ | FPCR_AH, FPCR_FZ, INPUT_DENORMAL),
        helper,
    );
    flag_if_active(code, ZERO_CLASS, active, DIVIDE_BY_ZERO);
    // Signalling NaNs: NaN with the quiet bit clear.
    lane_constant(code, format, SCRATCH, quiet_bit(format));
    packed(code, Host::MoveAligned, LANES, SAVED);
    packed(code, Host::And, LANES, SCRATCH);
    packed(code, Host::Xor, VALUE, VALUE);
    packed(code, Host::EqualDword, LANES, VALUE);
    packed(code, Host::And, LANES, NAN);
    flag_if_active(code, LANES, active, INVALID);
    // VALUE: the sign alone.
    packed(code, Host::MoveAligned, VALUE, SAVED);
    packed(code, Host::Xor, VALUE, MAGNITUDE);
    lane_constant(code, format, SCRATCH, infinity);
    if sqrt {
        // Zeros give the signed infinity, +infinity zero, other negative
        // inputs the default NaN with IOC.
        packed(code, Host::Or, VALUE, SCRATCH);
        packed(code, Host::And, VALUE, ZERO_CLASS);
        const DOMAIN: u8 = SCRATCH;
        packed(code, Host::MoveAligned, LANES, SAVED);
        compare(code, format, LANES, SCRATCH, LESS);
        lane_constant(code, format, DOMAIN, minimum);
        compare(code, format, DOMAIN, SAVED, LESS_OR_EQUAL);
        packed(code, Host::And, DOMAIN, LANES);
        const NEGATIVE: u8 = ZERO_CLASS;
        packed(code, Host::Xor, LANES, LANES);
        packed(code, Host::MoveAligned, MAGNITUDE, SAVED);
        compare(code, format, MAGNITUDE, LANES, LESS);
        packed(code, Host::AndNot, NEGATIVE, MAGNITUDE);
        flag_if_active(code, NEGATIVE, active, INVALID);
        default_nan(code, format, MAGNITUDE);
        packed(code, Host::And, MAGNITUDE, NEGATIVE);
        packed(code, Host::AndNot, NEGATIVE, VALUE);
        packed(code, Host::Or, NEGATIVE, MAGNITUDE);
        // NEGATIVE now holds the non-NaN values.
        lane_constant(code, format, MAGNITUDE, quiet_bit(format));
        nan_result(code, format, LANES, MAGNITUDE);
        packed(code, Host::And, LANES, NAN);
        packed(code, Host::AndNot, NAN, NEGATIVE);
        packed(code, Host::Or, NAN, LANES);
        packed(code, Host::And, RESULT, DOMAIN);
        packed(code, Host::AndNot, DOMAIN, NAN);
        packed(code, Host::Or, RESULT, DOMAIN);
    } else {
        // Zeros give the signed infinity, infinities and inputs whose
        // estimate flushes the signed zero (UFC under FZ).
        packed(code, Host::MoveAligned, LANES, MAGNITUDE);
        compare(code, format, LANES, SCRATCH, LESS);
        packed(code, Host::And, SCRATCH, ZERO_CLASS);
        packed(code, Host::Or, VALUE, SCRATCH);
        const SPECIAL: u8 = SCRATCH;
        lane_constant(code, format, SPECIAL, subnormal_estimate);
        compare(code, format, SPECIAL, MAGNITUDE, LESS_OR_EQUAL);
        packed(code, Host::And, LANES, SPECIAL);
        lanes_need_control(
            code,
            LANES,
            active,
            (FPCR_FZ | FPCR_AH, FPCR_FZ, UNDERFLOW),
            helper,
        );
        packed(code, Host::Or, SPECIAL, ZERO_CLASS);
        packed(code, Host::Or, SPECIAL, NAN);
        lane_constant(code, format, MAGNITUDE, quiet_bit(format));
        nan_result(code, format, LANES, MAGNITUDE);
        packed(code, Host::And, LANES, NAN);
        packed(code, Host::AndNot, NAN, VALUE);
        packed(code, Host::Or, NAN, LANES);
        packed(code, Host::And, NAN, SPECIAL);
        packed(code, Host::AndNot, SPECIAL, RESULT);
        packed(code, Host::Or, SPECIAL, NAN);
        packed(code, Host::MoveAligned, RESULT, SPECIAL);
    }
    // AH keeps every estimate flag.
    code.asm().test_imm(Size::Dword, R10, FPCR_AH as i32);
    let alternative = branch(code, Condition::NotZero);
    code.asm().alu(
        Alu::Or,
        Size::Dword,
        guest(core::mem::offset_of!(crate::GuestState, fpsr) as u32),
        R11,
    );
    let next = code.len();
    resolve(code, alternative, next);
}

fn commit_result(code: &mut Code, operand: &Operand) {
    let result = if operand.shape == Shape::Scalar {
        read_vector(code, operand.destination, SAVED);
        match operand.format {
            Format::Half => {
                let mut asm = code.asm();
                asm.movd_from_xmm(Rax, RESULT);
                asm.pinsrw(SAVED, Rax, FIRST_LANE);
            }
            Format::Single => {
                Host::MoveScalarSingle.register(code, SAVED, RESULT, TargetCpu::Baseline)
            }
            Format::Double => {
                Host::MoveScalarDouble.register(code, SAVED, RESULT, TargetCpu::Baseline)
            }
        }
        test_control(code, FPCR_NEP);
        let merging = branch(code, Condition::NotZero);
        clear_scalar(code, operand.format, SAVED);
        let merged = code.len();
        resolve(code, merging, merged);
        SAVED
    } else {
        if operand.shape == Shape::Vector64 {
            clear_scalar(code, Format::Double, RESULT);
        }
        RESULT
    };
    commit(code, operand.destination, result);
}

pub(in crate::x86_64) fn emit(
    code: &mut Code,
    operand: &Operand,
    pc: GuestAddress,
    target: TargetCpu,
) {
    let mut helper = Vec::new();
    let mut special = Vec::new();
    source_guard(code, operand, &mut helper, &mut special);
    if operand.kind.is_unsigned() {
        unsigned_estimate(code, operand.kind, target);
    } else if operand.kind == Kind::ReciprocalExponent {
        reciprocal_exponent(code, operand.format);
    } else {
        floating_estimate(code, operand, target);
    }
    let mut done = Vec::new();
    if !special.is_empty() {
        done.push(jump(code));
        let start = code.len();
        for patch in special {
            resolve(code, patch, start);
        }
        special_lanes(code, operand, target, &mut helper);
    }
    if !helper.is_empty() {
        helper_section(code, &helper, pc);
    }
    let end = code.len();
    for patch in done {
        resolve(code, patch, end);
    }
    commit_result(code, operand);
}
