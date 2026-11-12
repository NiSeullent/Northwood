//! Half <-> 16-bit integer conversions from the fields, in SSE2 integer
//! lanes on every x86 tier: FCVT{N,A,P,M,Z}{S,U} and SCVTF/UCVTF, vector
//! 4H/8H and SIMD scalar H, without fixed-point scaling.
//!
//! No host FP status is raised and MXCSR is untouched: the only host FP
//! instructions are CVTDQ2PS of 16-bit integers and CVTTPS2DQ of powers of
//! two, both exact. F16C would need the MXCSR environment around VCVTPS2PH
//! (about 35 TSC ticks) and gains nothing for the truncation, so the same
//! field recipe serves every tier; AVX512-FP16 hosts keep their native
//! recipe. Inputs the recipes do not cover take the exact helper.
use super::numerical::{
    Condition, branch, clear_scalar, commit_inexact, jump, resolve, test_control,
};
use super::precision::{dword_constant, field_narrow_lanes};
use super::{Host, commit, encode::guest, packed, read_vector, vector_offset};
use crate::TargetCpu;
use crate::x86_64::asm::{Alu, Code, Gpr, Rm, Size, Unary, vector_extensions};
use crate::x86_64::isa::ShiftOpcode;
use crate::{
    GuestAddress,
    aarch64::{
        floating::{
            FPCR_EXCEPTION_ENABLE_MASK, FPCR_NEP, FPCR_ROUNDING_MASK, FPCR_ROUNDING_SHIFT, Format,
            integer::{IntegerRounding, IntegerWidth},
        },
        simd::{
            arithmetic::Shape,
            conversion::{IntegerConversion, IntegerOperand},
        },
    },
    block::ExitKind,
};
use Gpr::{Rax, Rcx, Rdx};
use alloc::vec::Vec;

const RESULT: u8 = 0;
const INPUT: u8 = 1;
/// Results of the low four lanes while the high four convert.
const LOW_HALVES: u8 = 7;
const HALF_FRACTION_BITS: u8 = 10;
const HALF_EXPONENT_BIAS: u32 = 15;
const HALF_EXPONENT_ALL_ONES: u32 = 31;
const SINGLE_FRACTION_BITS: u8 = 23;
const SINGLE_EXPONENT_BIAS: u32 = 127;
/// Shift between a 16-bit result of the product (M * 2^(E-15)) >> 10 and
/// its high word.
const PRODUCT_HIGH_SHIFT: u8 = 16 - HALF_FRACTION_BITS;
const UPPER_QWORD_BYTES: u8 = 8;

fn simd_halves(operand: &IntegerConversion) -> Option<(u8, Shape, bool)> {
    match operand.integer {
        IntegerOperand::Simd {
            register,
            format,
            shape,
        } if format.width == IntegerWidth::Bits16
            && operand.format == Format::Half
            && operand.frac_bits == 0 =>
        {
            Some((register, shape, format.signed))
        }
        _ => None,
    }
}

fn applies(target: TargetCpu) -> bool {
    !vector_extensions::half_supported(target) && !target.data_independent_timing()
}

/// Whether FCVT{N,A,P,M,Z}{S,U} `operand` has the field recipe on some
/// target; every rounding the instruction names has one.
pub(super) fn to_integer_inline(operand: &IntegerConversion) -> bool {
    simd_halves(operand).is_some()
}

/// Whether SCVTF/UCVTF `operand` has the field recipe on some target.
pub(super) fn from_integer_inline(operand: &IntegerConversion) -> bool {
    simd_halves(operand).is_some()
}

/// `value` in every word of `xmm`. Clobbers EAX.
fn word_constant(code: &mut Code, xmm: u8, value: u16) {
    dword_constant(code, xmm, u32::from(value) * 0x0001_0001);
}

/// The active 16-bit lanes of guest vector `register` in XMM1, the others
/// zero.
fn read_halves(code: &mut Code, register: u8, shape: Shape) {
    match shape {
        Shape::Scalar => {
            let mut asm = code.asm();
            asm.movzx(Size::Dword, Rax, Size::Word, guest(vector_offset(register)));
            asm.movd_to_xmm(INPUT, Rax);
        }
        Shape::Vector64 => Host::MoveQword.emit(
            code,
            INPUT,
            guest(vector_offset(register)),
            TargetCpu::Baseline,
        ),
        _ => read_vector(code, register, INPUT),
    }
}

/// Write the active 16-bit lanes in `xmm` to guest vector `register`: a
/// vector form as it is (inactive lanes are zero), a scalar one under NEP.
fn write_halves(code: &mut Code, register: u8, shape: Shape, xmm: u8) {
    if shape != Shape::Scalar {
        commit(code, register, xmm);
        return;
    }
    code.asm().movd_from_xmm(Rax, xmm);
    read_vector(code, register, RESULT);
    code.asm().pinsrw(RESULT, Rax, 0);
    test_control(code, FPCR_NEP);
    let merge = branch(code, Condition::NotZero);
    clear_scalar(code, Format::Half, RESULT);
    let end = code.len();
    resolve(code, merge, end);
    commit(code, register, RESULT);
}

/// ECX = all ones when EAX is nonzero, else zero.
fn status_from(code: &mut Code) {
    let mut asm = code.asm();
    asm.mov(Size::Dword, Rcx, Rax);
    asm.unary(Unary::Neg, Size::Dword, Rcx);
    asm.alu(Alu::Sbb, Size::Dword, Rcx, Rcx);
}

/// Branch to `fallback` when some word of `mask` is set.
fn fallback_if_any(code: &mut Code, mask: u8, fallback: &mut Vec<usize>) {
    Host::ByteSignMask.register(code, 0, mask, TargetCpu::Baseline);
    code.asm().test(Size::Dword, Rax, Rax);
    fallback.push(branch(code, Condition::NotZero));
}

/// FCVT{N,A,P,M,Z}{S,U} from Half to 16-bit lanes. A lane with biased
/// exponent E and significand M (hidden bit included) has the value
/// M * 2^(E-25); for E >= 15 its truncation is (M * 2^(E-15)) >> 10, with
/// the product below 2^26 in a high and low word, and the dropped bits are
/// the low ten of the low word. E < 15 truncates to zero. The other
/// roundings then add one to the magnitude (`round_magnitudes`). NaN,
/// infinite and nonzero subnormal lanes (whose FZ16 flushing decides IXC),
/// and lanes out of the integer range (IOC), take the helper; -32768 does
/// too. A lane is inexact exactly when bits were dropped, whatever the
/// rounding.
fn emit_to(code: &mut Code, operand: &IntegerConversion, mode: IntegerRounding, pc: GuestAddress) {
    const MAGNITUDE: u8 = 2;
    const EXPONENT: u8 = 3;
    const BAD: u8 = 4;
    const SCRATCH: u8 = 5;
    const MULTIPLIER: u8 = 6;
    const LOW: u8 = 7;
    let (register, shape, signed) = simd_halves(operand).expect("field form");
    let mut fallback = Vec::new();
    test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
    fallback.push(branch(code, Condition::NotZero));
    read_halves(code, operand.float, shape);
    packed(code, Host::MoveAligned, MAGNITUDE, INPUT);
    ShiftOpcode::LeftWord.emit(code, MAGNITUDE, 1);
    ShiftOpcode::RightWord.emit(code, MAGNITUDE, 1);
    packed(code, Host::MoveAligned, EXPONENT, MAGNITUDE);
    ShiftOpcode::RightWord.emit(code, EXPONENT, HALF_FRACTION_BITS);
    // NaN and infinity.
    word_constant(code, BAD, HALF_EXPONENT_ALL_ONES as u16);
    packed(code, Host::EqualWord, BAD, EXPONENT);
    // Nonzero subnormals: exponent zero, magnitude not.
    packed(code, Host::Xor, SCRATCH, SCRATCH);
    packed(code, Host::EqualWord, SCRATCH, EXPONENT);
    packed(code, Host::Xor, MULTIPLIER, MULTIPLIER);
    packed(code, Host::EqualWord, MULTIPLIER, MAGNITUDE);
    packed(code, Host::AndNot, MULTIPLIER, SCRATCH);
    packed(code, Host::Or, BAD, MULTIPLIER);
    if signed {
        // |x| >= 2^15.
        word_constant(code, SCRATCH, (HALF_EXPONENT_BIAS + 14) as u16);
        packed(code, Host::MoveAligned, MULTIPLIER, EXPONENT);
        packed(code, Host::GreaterWord, MULTIPLIER, SCRATCH);
    } else {
        // Negative with |x| >= 1.
        word_constant(code, SCRATCH, (HALF_EXPONENT_BIAS - 1) as u16);
        packed(code, Host::MoveAligned, MULTIPLIER, EXPONENT);
        packed(code, Host::GreaterWord, MULTIPLIER, SCRATCH);
        packed(code, Host::MoveAligned, SCRATCH, INPUT);
        ShiftOpcode::ArithmeticWord.emit(code, SCRATCH, 15);
        packed(code, Host::And, MULTIPLIER, SCRATCH);
    }
    packed(code, Host::Or, BAD, MULTIPLIER);
    fallback_if_any(code, BAD, &mut fallback);
    // BAD becomes the lanes below one: E < 15.
    word_constant(code, BAD, HALF_EXPONENT_BIAS as u16);
    packed(code, Host::GreaterWord, BAD, EXPONENT);
    // MULTIPLIER = 2^(max(E, 15) - 15) from the exact CVTTPS2DQ of the
    // Single 2^(max(E, 15) - 15).
    word_constant(code, SCRATCH, HALF_EXPONENT_BIAS as u16);
    packed(code, Host::MaxSignedWord, EXPONENT, SCRATCH);
    packed(code, Host::Xor, SCRATCH, SCRATCH);
    packed(code, Host::MoveAligned, MULTIPLIER, EXPONENT);
    packed(code, Host::UnpackLowWord, MULTIPLIER, SCRATCH);
    packed(code, Host::UnpackHighWord, EXPONENT, SCRATCH);
    dword_constant(code, SCRATCH, SINGLE_EXPONENT_BIAS - HALF_EXPONENT_BIAS);
    packed(code, Host::AddDword, MULTIPLIER, SCRATCH);
    packed(code, Host::AddDword, EXPONENT, SCRATCH);
    for xmm in [MULTIPLIER, EXPONENT] {
        ShiftOpcode::LeftDword.emit(code, xmm, SINGLE_FRACTION_BITS);
        code.asm().cvttps2dq(xmm, Rm::Xmm(xmm));
        // 2^15 sign-extends to the word 0x8000, which packs exactly.
        ShiftOpcode::LeftDword.emit(code, xmm, 16);
        ShiftOpcode::ArithmeticDword.emit(code, xmm, 16);
    }
    packed(code, Host::PackSignedDwordToWord, MULTIPLIER, EXPONENT);
    // EXPONENT = M: the fraction with the hidden bit.
    packed(code, Host::MoveAligned, EXPONENT, MAGNITUDE);
    ShiftOpcode::LeftWord.emit(code, EXPONENT, 16 - HALF_FRACTION_BITS);
    ShiftOpcode::RightWord.emit(code, EXPONENT, 16 - HALF_FRACTION_BITS);
    word_constant(code, SCRATCH, 1 << HALF_FRACTION_BITS);
    packed(code, Host::Or, EXPONENT, SCRATCH);
    packed(code, Host::MoveAligned, LOW, EXPONENT);
    packed(code, Host::MultiplyLowWord, LOW, MULTIPLIER);
    packed(code, Host::MultiplyHighUnsignedWord, EXPONENT, MULTIPLIER);
    ShiftOpcode::LeftWord.emit(code, EXPONENT, PRODUCT_HIGH_SHIFT);
    packed(code, Host::MoveAligned, RESULT, LOW);
    ShiftOpcode::RightWord.emit(code, RESULT, HALF_FRACTION_BITS);
    packed(code, Host::Or, RESULT, EXPONENT);
    packed(code, Host::AndNot, BAD, RESULT);
    // RESULT = the truncated magnitudes, zero below one.
    packed(code, Host::MoveAligned, RESULT, BAD);
    if mode != IntegerRounding::TowardZero {
        round_magnitudes(code, mode, signed, &mut fallback);
    }
    // Inexact lanes: dropped product bits at E >= 15, a nonzero magnitude
    // below one. Reuse BAD's complement: recompute the below-one lanes.
    word_constant(code, BAD, HALF_EXPONENT_BIAS as u16);
    packed(code, Host::MoveAligned, SCRATCH, MAGNITUDE);
    ShiftOpcode::RightWord.emit(code, SCRATCH, HALF_FRACTION_BITS);
    packed(code, Host::GreaterWord, BAD, SCRATCH);
    ShiftOpcode::LeftWord.emit(code, LOW, PRODUCT_HIGH_SHIFT);
    packed(code, Host::Xor, SCRATCH, SCRATCH);
    packed(code, Host::EqualWord, LOW, SCRATCH);
    packed(code, Host::Or, LOW, BAD);
    // LOW: lanes exact at E >= 15 or below one. Below one, exact means zero.
    packed(code, Host::EqualWord, SCRATCH, MAGNITUDE);
    packed(code, Host::And, SCRATCH, BAD);
    packed(code, Host::AndNot, BAD, LOW);
    packed(code, Host::Or, BAD, SCRATCH);
    // BAD: exact lanes, every bit set.
    Host::ByteSignMask.register(code, 0, BAD, TargetCpu::Baseline);
    code.asm()
        .alu_imm32(Alu::Xor, Size::Dword, Rax, u16::MAX.into());
    status_from(code);
    if signed {
        packed(code, Host::MoveAligned, SCRATCH, INPUT);
        ShiftOpcode::ArithmeticWord.emit(code, SCRATCH, 15);
        packed(code, Host::Xor, RESULT, SCRATCH);
        packed(code, Host::SubWord, RESULT, SCRATCH);
    }
    commit_inexact(code);
    write_halves(code, register, shape, RESULT);
    finish(code, fallback, pc);
}

/// Add one to the truncated magnitudes in RESULT where `mode` rounds them
/// up, from the input in INPUT, its magnitude in XMM2 and the low product
/// word in XMM7, which stay; XMM3 to XMM6 and EAX are clobbered.
///
/// At E >= 15 the dropped bits are R = the low ten bits of the product: the
/// even-ties rounding adds one when R + (truncation & 1) passes one half
/// (0x200), the away-ties one when R reaches it, the directed ones when R is
/// nonzero and the sign is the rounding's. Below one (E < 15, a truncation
/// of zero) the same tests run on the magnitude moved so that one half
/// (0x3800) meets the same thresholds, and on the plain magnitude for the
/// directed roundings. No rounded magnitude leaves the integer: fractions
/// exist only below 2^10. An unsigned lane that is negative with a nonzero
/// rounded magnitude (IOC) takes the helper.
fn round_magnitudes(
    code: &mut Code,
    mode: IntegerRounding,
    signed: bool,
    fallback: &mut Vec<usize>,
) {
    const MAGNITUDE: u8 = 2;
    const EXPONENT: u8 = 3;
    const UP: u8 = 4;
    const SCRATCH: u8 = 5;
    const DROPPED: u8 = 6;
    const LOW: u8 = 7;
    const ONE_HALF: u16 = 0x3800;
    const HALF_DROPPED: u16 = 1 << (HALF_FRACTION_BITS - 1);
    let nearest = matches!(
        mode,
        IntegerRounding::NearestEven | IntegerRounding::NearestAway
    );
    // UP = the lanes below one.
    packed(code, Host::MoveAligned, EXPONENT, MAGNITUDE);
    ShiftOpcode::RightWord.emit(code, EXPONENT, HALF_FRACTION_BITS);
    word_constant(code, UP, HALF_EXPONENT_BIAS as u16);
    packed(code, Host::GreaterWord, UP, EXPONENT);
    packed(code, Host::MoveAligned, DROPPED, LOW);
    word_constant(code, SCRATCH, (1 << HALF_FRACTION_BITS) - 1);
    packed(code, Host::And, DROPPED, SCRATCH);
    if mode == IntegerRounding::NearestEven {
        packed(code, Host::MoveAligned, SCRATCH, RESULT);
        ShiftOpcode::LeftWord.emit(code, SCRATCH, 15);
        ShiftOpcode::RightWord.emit(code, SCRATCH, 15);
        packed(code, Host::AddWord, DROPPED, SCRATCH);
    }
    packed(code, Host::MoveAligned, EXPONENT, MAGNITUDE);
    if nearest {
        word_constant(code, SCRATCH, ONE_HALF - HALF_DROPPED);
        packed(code, Host::SubWord, EXPONENT, SCRATCH);
    }
    packed(code, Host::And, EXPONENT, UP);
    packed(code, Host::AndNot, UP, DROPPED);
    packed(code, Host::Or, UP, EXPONENT);
    let threshold = match mode {
        IntegerRounding::NearestEven => HALF_DROPPED,
        IntegerRounding::NearestAway => HALF_DROPPED - 1,
        _ => 0,
    };
    if threshold == 0 {
        packed(code, Host::Xor, SCRATCH, SCRATCH);
    } else {
        word_constant(code, SCRATCH, threshold);
    }
    // UP = the lanes whose magnitude rounds up.
    packed(code, Host::GreaterWord, UP, SCRATCH);
    if !nearest {
        packed(code, Host::MoveAligned, SCRATCH, INPUT);
        ShiftOpcode::ArithmeticWord.emit(code, SCRATCH, 15);
        if mode == IntegerRounding::TowardPositive {
            packed(code, Host::AndNot, SCRATCH, UP);
        } else {
            packed(code, Host::And, SCRATCH, UP);
        }
        packed(code, Host::MoveAligned, UP, SCRATCH);
    }
    packed(code, Host::SubWord, RESULT, UP);
    if !signed {
        packed(code, Host::MoveAligned, SCRATCH, INPUT);
        ShiftOpcode::ArithmeticWord.emit(code, SCRATCH, 15);
        packed(code, Host::Xor, EXPONENT, EXPONENT);
        packed(code, Host::EqualWord, EXPONENT, RESULT);
        packed(code, Host::AndNot, EXPONENT, SCRATCH);
        fallback_if_any(code, EXPONENT, fallback);
    }
}

/// SCVTF/UCVTF from 16-bit lanes to Half under FPCR nearest-even: each lane
/// widens to a dword and converts exactly to Single (16 bits fit its
/// significand), which the Single field narrowing then rounds once. Lanes
/// that round to infinity (overflow) take the helper.
fn emit_from(code: &mut Code, operand: &IntegerConversion, pc: GuestAddress) {
    let (register, shape, signed) = simd_halves(operand).expect("field form");
    let mut fallback = Vec::new();
    test_control(
        code,
        FPCR_EXCEPTION_ENABLE_MASK | FPCR_ROUNDING_MASK << FPCR_ROUNDING_SHIFT,
    );
    fallback.push(branch(code, Condition::NotZero));
    let quarters: &[bool] = if shape == Shape::Vector128 {
        &[false, true]
    } else {
        &[false]
    };
    for &upper in quarters {
        read_halves(code, register, shape);
        if upper {
            ShiftOpcode::RightBytes.emit(code, INPUT, UPPER_QWORD_BYTES);
        }
        if signed {
            packed(code, Host::UnpackLowWord, INPUT, INPUT);
            ShiftOpcode::ArithmeticDword.emit(code, INPUT, 16);
        } else {
            packed(code, Host::Xor, RESULT, RESULT);
            packed(code, Host::UnpackLowWord, INPUT, RESULT);
        }
        Host::ConvertPackedDwordToSingle.register(code, INPUT, INPUT, TargetCpu::Baseline);
        field_narrow_lanes(code, &mut fallback);
        if upper {
            let mut asm = code.asm();
            asm.alu(Alu::Or, Size::Dword, Rcx, Rdx);
            asm.movlhps(LOW_HALVES, Rm::Xmm(RESULT));
        } else if quarters.len() > 1 {
            code.asm().mov(Size::Dword, Rdx, Rcx);
            packed(code, Host::MoveAligned, LOW_HALVES, RESULT);
        }
    }
    let result = if quarters.len() > 1 {
        LOW_HALVES
    } else {
        RESULT
    };
    commit_inexact(code);
    write_halves(code, operand.float, shape, result);
    finish(code, fallback, pc);
}

fn finish(code: &mut Code, fallback: Vec<usize>, pc: GuestAddress) {
    let done = jump(code);
    let helper = code.len();
    for patch in fallback {
        resolve(code, patch, helper);
    }
    super::super::block::direct_exit(code, pc, ExitKind::FloatingPoint);
    let end = code.len();
    resolve(code, done, end);
}

/// Emit FCVT{N,A,P,M,Z}{S,U} `operand` when the field recipe applies on
/// `target`.
pub(super) fn try_to_integer(
    code: &mut Code,
    operand: &IntegerConversion,
    mode: IntegerRounding,
    pc: GuestAddress,
    target: TargetCpu,
) -> bool {
    if !to_integer_inline(operand) || !applies(target) {
        return false;
    }
    emit_to(code, operand, mode, pc);
    true
}

/// Emit SCVTF/UCVTF `operand` when the field recipe applies on `target`.
pub(super) fn try_from_integer(
    code: &mut Code,
    operand: &IntegerConversion,
    pc: GuestAddress,
    target: TargetCpu,
) -> bool {
    if !from_integer_inline(operand) || !applies(target) {
        return false;
    }
    emit_from(code, operand, pc);
    true
}
