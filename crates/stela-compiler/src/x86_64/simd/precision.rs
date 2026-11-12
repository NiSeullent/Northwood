//! FCVT between FP formats: SSE2, F16C, AVX512-FP16 or integer paths, software
//! fallback otherwise.
use super::super::isa::ShiftOpcode;
use super::numerical::helper_section;
use super::numerical::{
    ArithmeticOpcode, Condition, INPUT_MINIMUM_EXPONENT, MINIMUM_EXPONENT_VECTOR,
    RESULT_MINIMUM_EXPONENT, Rounding, branch, clear_scalar, commit_inexact, environment_round,
    exponent_range, jump, prepare_half_result_range, prepare_vector_input_range,
    prepare_vector_result_range, read_numerical_vector, reject_other_status,
    reject_other_status_but_underflow, resolve, restore_environment, scalar_range_or_zero,
    single_vector_arithmetic, test_control, vector_exponent_range, vector_normal_or_zero_range,
};
use super::{Host, commit, encode::guest, packed, read_vector, vector_offset};
use crate::x86_64::asm::FloatLanes;
use crate::x86_64::asm::{
    Alu, BitTest, Code, Cond, Gpr, Rm, ShiftOp, Size, Unary, vector_extensions,
};
use crate::{
    GuestAddress, TargetCpu,
    aarch64::{
        FloatingPointOperation, Instruction, Operation,
        floating::{
            FPCR_AH, FPCR_AHP, FPCR_EXCEPTION_ENABLE_MASK, FPCR_NEP, FPCR_ROUNDING_MASK,
            FPCR_ROUNDING_SHIFT, Format, UNDERFLOW, precision::PrecisionRounding,
        },
        simd::{
            arithmetic::Shape,
            conversion::{Operand, PrecisionMode, PrecisionOperand},
        },
    },
    block::ExitKind,
};
use Gpr::{Rax, Rcx, Rdx};
use alloc::vec::Vec;

const RESULT_VECTOR: u8 = 0;
const SOURCE_VECTOR: u8 = 1;
const JAM_VECTOR: u8 = 6;
/// The low Half of a scalar narrowing, ahead of its merge into old Vd.
const HALF_LANE_VECTOR: u8 = 6;
/// The zero classifier's scratch in vector Half narrowing.
const NARROW_ZERO_SCRATCH: u8 = 7;
/// VCVTPS2PH's immediate: round by MXCSR.RC.
const CONVERT_USING_MXCSR: u8 = 1 << 2;
const COMPARE_EQUAL: u8 = 0;
const UPPER_QWORD_BYTES: u8 = 8;
const HALF_EXPONENT_BITS: u8 = 5;
const HALF_EXPONENT_MASK: u32 = (1 << HALF_EXPONENT_BITS) - 1;
const HALF_NORMAL_EXPONENT_COUNT: u8 = HALF_EXPONENT_MASK as u8 - 1;
const HALF_SIGN: u32 = 1 << (u16::BITS - 1);
const HALF_MAGNITUDE: u32 = HALF_SIGN - 1;
const HALF_TO_SINGLE_FRACTION_SHIFT: u8 = 13;
const HALF_TO_SINGLE_SIGN_SHIFT: u8 = 16;
const SINGLE_EXPONENT_MASK: u32 =
    (1 << (Format::Single.bits() - Format::Single.fraction_bits() - 1)) - 1;
const HALF_TO_SINGLE_BIAS: u32 =
    ((SINGLE_EXPONENT_MASK >> 1) - (HALF_EXPONENT_MASK >> 1)) << Format::Single.fraction_bits();
const HALF_FIELD_TO_DWORD_TOP: u8 = u32::BITS as u8 - HALF_TO_SINGLE_SIGN_SHIFT + 1;
const HALF_EXPONENT_TO_DWORD_BOTTOM: u8 = u32::BITS as u8 - HALF_EXPONENT_BITS;
const HALF_MAXIMUM_TO_BIAS_BITS: u8 = 2;
const SINGLE_BIAS_POSITION: u8 = HALF_TO_SINGLE_BIAS.trailing_zeros() as u8;
const HALF_VALIDITY_MASK: i8 = 0x0f;
const HALF_NORMAL_VECTOR: u8 = 2;
const HALF_FINITE_VECTOR: u8 = 3;
const HALF_MAXIMUM_VECTOR: u8 = 4;
const HALF_ZERO_VECTOR: u8 = 5;
const HALF_SCALED_VECTOR: u8 = 6;
const HALF_PAIR_SOURCE: u8 = 7;
const HALF_PAIR_SWAP_QWORDS: u8 = 0x4e;
const HALF_SUBNORMAL_SCALE: u32 = (Format::Single.exponent_bias() + 1
    - Format::Half.exponent_bias()
    - Format::Half.fraction_bits())
    << Format::Single.fraction_bits();
const DOUBLE_TO_SINGLE_DISCARDED_BITS: u8 = 29;
const DOUBLE_TO_SINGLE_DISCARDED_MASK: u32 = (1 << DOUBLE_TO_SINGLE_DISCARDED_BITS) - 1;
const COMPACT_QWORD_LOW_DWORDS: u8 = 0x88;
const DWORD_SIGN_TO_LSB_SHIFT: u8 = u32::BITS as u8 - 1;
/// Field narrowing of Single to Half ([`emit_field_narrow_half`]) converts
/// zero magnitudes and those in [`FIELD_NARROW_MINIMUM`,
/// `FIELD_NARROW_MAXIMUM`]: from the smallest normal Half (2^-14) to the
/// largest value that rounds to a finite Half under nearest-even (below
/// 65520).
const FIELD_NARROW_MINIMUM: u32 = 0x3880_0000;
const FIELD_NARROW_MAXIMUM: u32 = 0x477f_efff;
/// The Single fraction bits a Half drops.
const FIELD_NARROW_DROPPED_BITS: u8 = 13;
const FIELD_NARROW_DROPPED_MASK: u32 = (1 << FIELD_NARROW_DROPPED_BITS) - 1;
/// The exponent rebias from Single to Half, in the Single exponent field.
const FIELD_NARROW_REBIAS: u32 = (Format::Single.exponent_bias() - Format::Half.exponent_bias())
    << Format::Single.fraction_bits();
/// Nearest-even before the shift: half the dropped weight less one; the kept
/// least significant bit adds the last one.
const FIELD_NARROW_ROUND: u32 = FIELD_NARROW_DROPPED_MASK >> 1;
const ALL_DWORD_LANES: i8 = 0x0f;

fn exception_guard(code: &mut Code, helper: &mut Vec<usize>) {
    test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
    helper.push(branch(code, Condition::NotZero));
}

fn scalar_commit(code: &mut Code, destination: u8, format: Format) {
    test_control(code, FPCR_NEP);
    let merging = branch(code, Condition::NotZero);
    clear_scalar(code, format, RESULT_VECTOR);
    let merge_target = code.len();
    resolve(code, merging, merge_target);
    commit(code, destination, RESULT_VECTOR);
}

fn half_normal_guard(code: &mut Code, helper: &mut Vec<usize>) {
    let mut asm = code.asm();
    asm.movd_from_xmm(Rax, SOURCE_VECTOR);
    asm.shift_imm(
        ShiftOp::Shr,
        Size::Dword,
        Rax,
        Format::Half.fraction_bits() as u8,
    );
    asm.alu_accumulator_imm32(Alu::And, Size::Dword, HALF_EXPONENT_MASK as i32);
    asm.alu_imm8(Alu::Sub, Size::Dword, Rax, 1);
    asm.alu_imm8(Alu::Cmp, Size::Dword, Rax, HALF_NORMAL_EXPONENT_COUNT as i8);
    helper.push(branch(code, Condition::AboveOrEqual));
}

/// Exact normal half widening uses only field construction. In particular,
/// conversion ignores FZ16, and no host numerical half extension is assumed.
fn half_to_single(code: &mut Code) {
    let mut asm = code.asm();
    asm.movd_from_xmm(Rax, SOURCE_VECTOR);
    asm.mov(Size::Dword, Rcx, Rax);
    asm.alu_accumulator_imm32(Alu::And, Size::Dword, HALF_MAGNITUDE as i32);
    asm.shift_imm(
        ShiftOp::Shl,
        Size::Dword,
        Rax,
        HALF_TO_SINGLE_FRACTION_SHIFT,
    );
    asm.alu_accumulator_imm32(Alu::Add, Size::Dword, HALF_TO_SINGLE_BIAS as i32);
    asm.alu_imm32(Alu::And, Size::Dword, Rcx, HALF_SIGN as i32);
    asm.shift_imm(ShiftOp::Shl, Size::Dword, Rcx, HALF_TO_SINGLE_SIGN_SHIFT);
    asm.alu(Alu::Or, Size::Dword, Rax, Rcx);
    asm.movd_to_xmm(SOURCE_VECTOR, Rax);
}

fn scalar_odd_jam(code: &mut Code) {
    // The source is a normal double and the result a normal single, so exactly
    // the low 29 fraction bits are discarded.
    let mut asm = code.asm();
    asm.movq_from_xmm(Rax, SOURCE_VECTOR);
    asm.alu_accumulator_imm32(
        Alu::And,
        Size::Dword,
        DOUBLE_TO_SINGLE_DISCARDED_MASK as i32,
    );
    asm.setcc(Cond::Ne, Rax);
    asm.movzx(Size::Dword, Rax, Size::Byte, Rax);
    asm.movd_to_xmm(JAM_VECTOR, Rax);
    packed(code, Host::Or, RESULT_VECTOR, JAM_VECTOR);
}

pub(in crate::x86_64) fn jam_odd_pair(code: &mut Code, one_vector: u8) {
    // The normal single widens exactly and is compared with its source double
    // lane without raising host status. A lane's sticky bit never comes from
    // the aggregate PE flag. The range preparation leaves 1 in every dword of
    // MINIMUM_EXPONENT_VECTOR.
    let mut asm = code.asm();
    asm.cvtps2pd(JAM_VECTOR, Rm::Xmm(RESULT_VECTOR));
    asm.float_compare(
        FloatLanes::PackedDouble,
        JAM_VECTOR,
        Rm::Xmm(SOURCE_VECTOR),
        COMPARE_EQUAL,
    );
    super::shuffle(code, JAM_VECTOR, JAM_VECTOR, COMPACT_QWORD_LOW_DWORDS);
    packed(code, Host::AndNot, JAM_VECTOR, one_vector);
    // PSHUFD duplicates masks into inactive upper lanes. Clear those masks so
    // CVTPD2PS's architecturally zero upper qword remains zero.
    clear_scalar(code, Format::Double, JAM_VECTOR);
    packed(code, Host::Or, RESULT_VECTOR, JAM_VECTOR);
}

fn vector_odd_jam(code: &mut Code) {
    jam_odd_pair(code, MINIMUM_EXPONENT_VECTOR);
}

fn emit_scalar(
    code: &mut Code,
    operand: &PrecisionOperand,
    source: Format,
    destination: Format,
    rounding: PrecisionRounding,
    pc: GuestAddress,
) {
    let mut helper = Vec::new();
    exception_guard(code, &mut helper);
    read_vector(code, operand.source, SOURCE_VECTOR);
    // Precision conversions merge old Vd, unlike binary arithmetic's Vn merge.
    read_vector(code, operand.destination, RESULT_VECTOR);
    if source == Format::Half {
        // A zero widens to the zero of its sign, exactly and without status.
        code.asm().movd_from_xmm(Rax, SOURCE_VECTOR);
        code.asm().test_imm(Size::Dword, Rax, HALF_MAGNITUDE as i32);
        let zero = branch(code, Condition::Zero);
        half_normal_guard(code, &mut helper);
        half_to_single(code);
        let widened = jump(code);
        let zero_start = code.len();
        resolve(code, zero, zero_start);
        let mut asm = code.asm();
        asm.alu_imm32(Alu::And, Size::Dword, Rax, HALF_SIGN as i32);
        asm.shift_imm(ShiftOp::Shl, Size::Dword, Rax, HALF_TO_SINGLE_SIGN_SHIFT);
        asm.movd_to_xmm(SOURCE_VECTOR, Rax);
        let joined = code.len();
        resolve(code, widened, joined);
        if destination == Format::Double {
            code.asm().cvtss2sd(RESULT_VECTOR, Rm::Xmm(SOURCE_VECTOR));
        } else {
            Host::MoveScalarSingle.register(
                code,
                RESULT_VECTOR,
                SOURCE_VECTOR,
                crate::TargetCpu::Baseline,
            );
        }
    } else if destination == Format::Double {
        // Normal or zero S→D is exact, unaffected by ambient RC/DAZ/FTZ/masks,
        // and raises neither Intel Invalid nor Denormal, nor any FPSR status
        // (a zero is no denormal input, whatever FZ holds). Preserve MXCSR.
        helper.push(scalar_range_or_zero(
            code,
            source,
            SOURCE_VECTOR,
            INPUT_MINIMUM_EXPONENT,
        ));
        code.asm().cvtss2sd(RESULT_VECTOR, Rm::Xmm(SOURCE_VECTOR));
    } else {
        exponent_range(code, source, SOURCE_VECTOR, INPUT_MINIMUM_EXPONENT);
        helper.push(branch(code, Condition::AboveOrEqual));
        environment_round(
            code,
            if rounding == PrecisionRounding::ToOdd {
                Rounding::Zero
            } else {
                Rounding::Fpcr
            },
        );
        code.asm().cvtsd2ss(RESULT_VECTOR, Rm::Xmm(SOURCE_VECTOR));
        restore_environment(code);
        helper.push(reject_other_status(code));
        exponent_range(code, destination, RESULT_VECTOR, RESULT_MINIMUM_EXPONENT);
        helper.push(branch(code, Condition::AboveOrEqual));
        if rounding == PrecisionRounding::ToOdd {
            scalar_odd_jam(code);
        }
    }
    helper_section(code, &helper, pc);
    if source == Format::Double && destination == Format::Single {
        commit_inexact(code);
    }
    scalar_commit(code, operand.destination, destination);
}

/// Branch if any exponent is outside the normal IEEE half range. Keep the
/// normal and finite lane masks for the zero and subnormal path.
fn half_normal_widening_guard(code: &mut Code) -> usize {
    packed(
        code,
        Host::EqualDword,
        HALF_MAXIMUM_VECTOR,
        HALF_MAXIMUM_VECTOR,
    );
    ShiftOpcode::RightDword.emit(code, HALF_MAXIMUM_VECTOR, HALF_EXPONENT_TO_DWORD_BOTTOM);
    packed(code, Host::MoveAligned, HALF_NORMAL_VECTOR, SOURCE_VECTOR);
    ShiftOpcode::LeftDword.emit(code, HALF_NORMAL_VECTOR, HALF_FIELD_TO_DWORD_TOP);
    ShiftOpcode::RightDword.emit(code, HALF_NORMAL_VECTOR, HALF_EXPONENT_TO_DWORD_BOTTOM);
    packed(
        code,
        Host::MoveAligned,
        HALF_FINITE_VECTOR,
        HALF_MAXIMUM_VECTOR,
    );
    packed(
        code,
        Host::GreaterDword,
        HALF_FINITE_VECTOR,
        HALF_NORMAL_VECTOR,
    );
    packed(
        code,
        Host::GreaterDword,
        HALF_NORMAL_VECTOR,
        HALF_ZERO_VECTOR,
    );
    packed(code, Host::And, HALF_NORMAL_VECTOR, HALF_FINITE_VECTOR);
    Host::DwordSignMask.register(code, 0, HALF_NORMAL_VECTOR, TargetCpu::Baseline);
    code.asm()
        .alu_imm8(Alu::Cmp, Size::Dword, Rax, HALF_VALIDITY_MASK);
    branch(code, Condition::NotZero)
}

fn add_half_widening_bias(code: &mut Code) {
    // The guard keeps exponent mask 31; 31 >> 2 gives 7, and 7 << 27 is
    // exactly the S-vs-H exponent bias adjustment. Reuse that constant.
    ShiftOpcode::RightDword.emit(code, HALF_MAXIMUM_VECTOR, HALF_MAXIMUM_TO_BIAS_BITS);
    ShiftOpcode::LeftDword.emit(code, HALF_MAXIMUM_VECTOR, SINGLE_BIAS_POSITION);
    packed(code, Host::AddDword, SOURCE_VECTOR, HALF_MAXIMUM_VECTOR);
}

fn widen_finite_half(code: &mut Code) -> usize {
    // Exponent 31 needs exact completion for AHP, infinity, and NaNs.
    Host::DwordSignMask.register(code, 0, HALF_FINITE_VECTOR, TargetCpu::Baseline);
    code.asm()
        .alu_imm8(Alu::Cmp, Size::Dword, Rax, HALF_VALIDITY_MASK);
    let exceptional = branch(code, Condition::NotZero);
    // Every 15-bit integer converts exactly to single, and scaling by 2^-24
    // produces a normal single or +0. The temporary leaves MXCSR untouched
    // even in lanes where the result comes from the normal representation.
    ShiftOpcode::LeftDword.emit(code, SOURCE_VECTOR, HALF_FIELD_TO_DWORD_TOP);
    ShiftOpcode::RightDword.emit(code, SOURCE_VECTOR, HALF_FIELD_TO_DWORD_TOP);
    packed(
        code,
        Host::ConvertPackedDwordToSingle,
        HALF_SCALED_VECTOR,
        SOURCE_VECTOR,
    );
    code.asm().mov_imm32(Rax, HALF_SUBNORMAL_SCALE);
    code.asm().movd_to_xmm(HALF_FINITE_VECTOR, Rax);
    super::shuffle(code, HALF_FINITE_VECTOR, HALF_FINITE_VECTOR, 0);
    single_vector_arithmetic(
        code,
        ArithmeticOpcode::Multiply,
        HALF_SCALED_VECTOR,
        HALF_FINITE_VECTOR,
    );
    ShiftOpcode::LeftDword.emit(code, SOURCE_VECTOR, HALF_TO_SINGLE_FRACTION_SHIFT);
    add_half_widening_bias(code);
    packed(code, Host::And, SOURCE_VECTOR, HALF_NORMAL_VECTOR);
    packed(code, Host::AndNot, HALF_NORMAL_VECTOR, HALF_SCALED_VECTOR);
    packed(code, Host::Or, SOURCE_VECTOR, HALF_NORMAL_VECTOR);
    exceptional
}

/// Widen the low four halves of SOURCE_VECTOR. Keeps the packed pair source
/// and the zero constant live across an accepted instruction.
fn widen_half_lanes(code: &mut Code, destination: u8, pc: GuestAddress, control: Option<usize>) {
    packed(code, Host::UnpackLowWord, SOURCE_VECTOR, HALF_ZERO_VECTOR);
    packed(code, Host::MoveAligned, RESULT_VECTOR, SOURCE_VECTOR);
    let needs_scaling = half_normal_widening_guard(code);
    // Normal half lanes need only an exponent bias and a field shift.
    ShiftOpcode::LeftDword.emit(code, SOURCE_VECTOR, HALF_FIELD_TO_DWORD_TOP);
    ShiftOpcode::RightDword.emit(
        code,
        SOURCE_VECTOR,
        HALF_FIELD_TO_DWORD_TOP - HALF_TO_SINGLE_FRACTION_SHIFT,
    );
    add_half_widening_bias(code);
    let normal = jump(code);
    let other = code.len();
    resolve(code, needs_scaling, other);
    let exceptional = widen_finite_half(code);
    match control {
        Some(control) => helper_section(code, &[control, exceptional], pc),
        None => helper_section(code, &[exceptional], pc),
    }
    let complete = code.len();
    resolve(code, normal, complete);
    ShiftOpcode::LeftDword.emit(code, RESULT_VECTOR, HALF_TO_SINGLE_SIGN_SHIFT);
    ShiftOpcode::RightDword.emit(code, RESULT_VECTOR, DWORD_SIGN_TO_LSB_SHIFT);
    ShiftOpcode::LeftDword.emit(code, RESULT_VECTOR, DWORD_SIGN_TO_LSB_SHIFT);
    packed(code, Host::Or, RESULT_VECTOR, SOURCE_VECTOR);
    commit(code, destination, RESULT_VECTOR);
}

fn emit_widen_half(code: &mut Code, operand: &PrecisionOperand, upper: bool, pc: GuestAddress) {
    test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
    let control = branch(code, Condition::NotZero);
    let part = vector_offset(operand.source) + u32::from(upper) * u32::from(UPPER_QWORD_BYTES);
    Host::MoveQword.emit(code, SOURCE_VECTOR, guest(part), TargetCpu::Baseline);
    packed(code, Host::Xor, HALF_ZERO_VECTOR, HALF_ZERO_VECTOR);
    widen_half_lanes(code, operand.destination, pc, Some(control));
}

fn half_widen(instruction: &Instruction) -> Option<(PrecisionOperand, bool)> {
    let Operation::FloatingPoint(FloatingPointOperation::Conversion(Operand::Precision(operand))) =
        instruction.operation
    else {
        return None;
    };
    let PrecisionMode::Widen {
        source: Format::Half,
        upper,
    } = operand.mode
    else {
        return None;
    };
    Some((operand, upper))
}

/// Two independent four-lane conversions from the same packed input. A
/// destructive first destination would change the second input and cannot
/// use the snapshot. Each half keeps its original exact completion boundary.
pub(in crate::x86_64) fn emit_half_widen_pair(
    code: &mut Code,
    source_map: &mut Vec<(GuestAddress, usize)>,
    instructions: &[Instruction],
) -> Option<u8> {
    let [first, second, ..] = instructions else {
        return None;
    };
    let (left, left_upper) = half_widen(first)?;
    let (right, right_upper) = half_widen(second)?;
    if first.pc.0.checked_add(4) != Some(second.pc.0)
        || left.source != right.source
        || left.destination == left.source
        || left_upper == right_upper
    {
        return None;
    }

    for (index, (instruction, operand, upper)) in
        [(first, left, left_upper), (second, right, right_upper)]
            .into_iter()
            .enumerate()
    {
        code.set_instruction(source_map.len());
        source_map.push((instruction.pc, code.len()));
        let control = if index == 0 {
            // Neither widening can update FPCR. Its conservative trap gate
            // belongs to the first instruction, before either write.
            test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
            let enabled = branch(code, Condition::NotZero);
            read_vector(code, left.source, HALF_PAIR_SOURCE);
            packed(code, Host::Xor, HALF_ZERO_VECTOR, HALF_ZERO_VECTOR);
            Some(enabled)
        } else {
            None
        };
        if upper {
            super::shuffle(code, SOURCE_VECTOR, HALF_PAIR_SOURCE, HALF_PAIR_SWAP_QWORDS);
        } else {
            packed(code, Host::MoveAligned, SOURCE_VECTOR, HALF_PAIR_SOURCE);
        }
        widen_half_lanes(code, operand.destination, instruction.pc, control);
    }
    Some(right.destination)
}

fn emit_widen(code: &mut Code, operand: &PrecisionOperand, upper: bool, pc: GuestAddress) {
    let mut helper = Vec::new();
    exception_guard(code, &mut helper);
    read_vector(code, operand.source, SOURCE_VECTOR);
    if upper {
        ShiftOpcode::RightBytes.emit(code, SOURCE_VECTOR, UPPER_QWORD_BYTES);
    }
    prepare_vector_input_range(code, Format::Single);
    vector_exponent_range(code, Format::Single, Shape::Vector64, SOURCE_VECTOR);
    helper.push(branch(code, Condition::NotZero));
    code.asm().cvtps2pd(RESULT_VECTOR, Rm::Xmm(SOURCE_VECTOR));
    helper_section(code, &helper, pc);
    commit(code, operand.destination, RESULT_VECTOR);
}

fn emit_narrow(
    code: &mut Code,
    operand: &PrecisionOperand,
    upper: bool,
    rounding: PrecisionRounding,
    pc: GuestAddress,
) {
    let mut helper = Vec::new();
    exception_guard(code, &mut helper);
    read_numerical_vector(code, Shape::Vector128, operand.source, SOURCE_VECTOR);
    prepare_vector_input_range(code, Format::Double);
    vector_exponent_range(code, Format::Double, Shape::Vector128, SOURCE_VECTOR);
    helper.push(branch(code, Condition::NotZero));
    environment_round(
        code,
        if rounding == PrecisionRounding::ToOdd {
            Rounding::Zero
        } else {
            Rounding::Fpcr
        },
    );
    code.asm().cvtpd2ps(RESULT_VECTOR, Rm::Xmm(SOURCE_VECTOR));
    restore_environment(code);
    helper.push(reject_other_status(code));
    // Conversion changes both element size and lane count. Rebuild the bounds
    // for the S result and check only its two active output lanes.
    prepare_vector_input_range(code, Format::Single);
    prepare_vector_result_range(code);
    vector_exponent_range(code, Format::Single, Shape::Vector64, RESULT_VECTOR);
    helper.push(branch(code, Condition::NotZero));
    if rounding == PrecisionRounding::ToOdd {
        vector_odd_jam(code);
    }
    helper_section(code, &helper, pc);
    commit_inexact(code);
    if upper {
        read_vector(code, operand.destination, SOURCE_VECTOR);
        code.asm().movlhps(SOURCE_VECTOR, Rm::Xmm(RESULT_VECTOR));
        commit(code, operand.destination, SOURCE_VECTOR);
    } else {
        commit(code, operand.destination, RESULT_VECTOR);
    }
}

/// After a vector narrowing of normal or zero Single lanes in SOURCE_VECTOR
/// to Half in RESULT_VECTOR, which VCVTPS2PH rounds correctly to the
/// subnormal grid and to zero: EDX = UFC when a lane was tiny and inexact.
/// Arm (FPCR.AH zero) finds a lane tiny before rounding, below 2^-14,
/// which compares the input's magnitude field, and inexact where the Half
/// widened back (exactly) differs from the input. FZ16 flushes no result of
/// a conversion and FZ no normal input; FPCR.AH, whose tininess follows
/// rounding, takes `helper` when a lane is tiny. Clobbers EAX and XMM2 to
/// XMM5; ECX survives.
fn narrow_half_underflow(code: &mut Code, helper: &mut Vec<usize>, target: TargetCpu) {
    const TINY: u8 = 2;
    const SCRATCH: u8 = 3;
    const MAGNITUDE: u8 = 4;
    const WIDE: u8 = 5;
    const SINGLE_MAGNITUDE: u32 = u32::MAX >> 1;
    /// 2^-14, Half's least normal, as Single.
    const HALF_MINIMUM_NORMAL: u32 = 0x3880_0000;
    dword_constant(code, SCRATCH, SINGLE_MAGNITUDE);
    packed(code, Host::MoveAligned, MAGNITUDE, SOURCE_VECTOR);
    packed(code, Host::And, MAGNITUDE, SCRATCH);
    packed(code, Host::Xor, TINY, TINY);
    packed(code, Host::EqualDword, TINY, MAGNITUDE);
    dword_constant(code, SCRATCH, HALF_MINIMUM_NORMAL);
    packed(code, Host::GreaterDword, SCRATCH, MAGNITUDE);
    // Nonzero and below the least normal.
    packed(code, Host::AndNot, TINY, SCRATCH);
    code.asm().alu(Alu::Xor, Size::Dword, Rdx, Rdx);
    Host::ByteSignMask.register(code, 0, TINY, TargetCpu::Baseline);
    code.asm().test(Size::Dword, Rax, Rax);
    let none = branch(code, Condition::Zero);
    test_control(code, FPCR_AH);
    helper.push(branch(code, Condition::NotZero));
    Host::ConvertHalfToSingle.register(code, WIDE, RESULT_VECTOR, target);
    packed(code, Host::EqualDword, WIDE, SOURCE_VECTOR);
    packed(code, Host::AndNot, WIDE, TINY);
    Host::ByteSignMask.register(code, 0, WIDE, TargetCpu::Baseline);
    let mut asm = code.asm();
    asm.test(Size::Dword, Rax, Rax);
    asm.setcc(Cond::Ne, Rdx);
    asm.shift_imm(
        ShiftOp::Shl,
        Size::Dword,
        Rdx,
        UNDERFLOW.trailing_zeros() as u8,
    );
    let end = code.len();
    resolve(code, none, end);
}

/// VCVTPS2PH narrows Single in one rounding, and AVX512-FP16's VCVTSD2SH
/// narrows Double. The DIT profile keeps its reviewed recipe.
pub(super) fn half_narrowing_supported(mode: PrecisionMode, target: TargetCpu) -> bool {
    match mode {
        PrecisionMode::Narrow {
            destination: Format::Half,
            ..
        }
        | PrecisionMode::Scalar {
            source: Format::Single,
            destination: Format::Half,
            ..
        } => target.supports(TargetCpu::F16c) && !target.data_independent_timing(),
        PrecisionMode::Scalar {
            source: Format::Double,
            destination: Format::Half,
            ..
        } => vector_extensions::half_supported(target),
        _ => false,
    }
}

/// Narrow to Half under the guards of the Half arithmetic path: normal inputs
/// (and zero vector lanes), and a helper for other status or, for a scalar,
/// a result below exponent 2, so x86's tininess rule never decides a guest
/// result. A vector narrows tiny lanes too ([`narrow_half_underflow`]).
/// FPCR.AHP selects a format the host lacks.
fn emit_narrow_half(
    code: &mut Code,
    operand: &PrecisionOperand,
    pc: GuestAddress,
    target: TargetCpu,
) {
    let mut helper = Vec::new();
    exception_guard(code, &mut helper);
    test_control(code, FPCR_AHP);
    helper.push(branch(code, Condition::NotZero));
    let scalar = match operand.mode {
        PrecisionMode::Scalar { source, .. } => Some(source),
        _ => None,
    };
    if let Some(source) = scalar {
        read_vector(code, operand.source, SOURCE_VECTOR);
        exponent_range(code, source, SOURCE_VECTOR, INPUT_MINIMUM_EXPONENT);
        helper.push(branch(code, Condition::AboveOrEqual));
        clear_scalar(code, source, SOURCE_VECTOR);
    } else {
        // A zero lane narrows to the zero of its sign, exactly and without
        // status on both sides; FZ and FIZ flush no zero.
        read_numerical_vector(code, Shape::Vector128, operand.source, SOURCE_VECTOR);
        prepare_vector_input_range(code, Format::Single);
        vector_normal_or_zero_range(
            code,
            Format::Single,
            Shape::Vector128,
            SOURCE_VECTOR,
            NARROW_ZERO_SCRATCH,
        );
        helper.push(branch(code, Condition::NotZero));
    }
    let result = if scalar.is_some() {
        HALF_LANE_VECTOR
    } else {
        RESULT_VECTOR
    };
    environment_round(code, Rounding::Fpcr);
    if scalar == Some(Format::Double) {
        vector_extensions::half_from_double(code, result, SOURCE_VECTOR);
    } else {
        // VCVTPS2PH encodes its source in ModRM.reg, and clears above the low
        // qword of its destination.
        Host::ConvertSingleToHalf.register_imm(
            code,
            SOURCE_VECTOR,
            result,
            CONVERT_USING_MXCSR,
            target,
        );
    }
    restore_environment(code);
    if scalar.is_some() {
        helper.push(reject_other_status(code));
        prepare_vector_input_range(code, Format::Half);
        prepare_half_result_range(code);
        vector_exponent_range(code, Format::Half, Shape::Scalar, result);
        helper.push(branch(code, Condition::NotZero));
    } else {
        // Normal or zero inputs: an overflow is the only other status a
        // result may raise, and the host's underflow (tininess after
        // rounding) is not Arm's, which the lanes give.
        helper.push(reject_other_status_but_underflow(code));
        narrow_half_underflow(code, &mut helper, target);
    }
    helper_section(code, &helper, pc);
    commit_inexact(code);
    if scalar.is_none() {
        code.asm().alu(
            Alu::Or,
            Size::Dword,
            guest(core::mem::offset_of!(crate::GuestState, fpsr) as u32),
            Rdx,
        );
    }
    match operand.mode {
        PrecisionMode::Narrow { upper: true, .. } => {
            read_vector(code, operand.destination, SOURCE_VECTOR);
            code.asm().movlhps(SOURCE_VECTOR, Rm::Xmm(RESULT_VECTOR));
            commit(code, operand.destination, SOURCE_VECTOR);
        }
        PrecisionMode::Narrow { .. } => commit(code, operand.destination, RESULT_VECTOR),
        _ => {
            read_vector(code, operand.destination, RESULT_VECTOR);
            let mut asm = code.asm();
            asm.movd_from_xmm(Rax, HALF_LANE_VECTOR);
            asm.pinsrw(RESULT_VECTOR, Rax, 0);
            scalar_commit(code, operand.destination, Format::Half);
        }
    }
}

/// Whether Single to Half `mode` has the field narrowing path of
/// [`emit_field_narrow_half`], which runs on every target.
pub(super) fn field_narrowing(mode: PrecisionMode) -> bool {
    matches!(
        mode,
        PrecisionMode::Scalar {
            source: Format::Single,
            destination: Format::Half,
            rounding: PrecisionRounding::Fpcr,
        } | PrecisionMode::Narrow {
            destination: Format::Half,
            rounding: PrecisionRounding::Fpcr,
            ..
        }
    )
}

/// Broadcast `value` to the four dwords of `xmm`. Clobbers EAX.
pub(super) fn dword_constant(code: &mut Code, xmm: u8, value: u32) {
    let mut asm = code.asm();
    asm.mov_imm32(Rax, value);
    asm.movd_to_xmm(xmm, Rax);
    super::shuffle(code, xmm, xmm, 0);
}

/// FCVT Hd, Sn from the Single fields in general registers: a zero, or a
/// magnitude in the field narrowing range, else `fallback`. Leaves the Half
/// in EAX, and ECX with every bit set when inexact and zero otherwise.
fn scalar_field_narrow(code: &mut Code, source: u8, fallback: &mut Vec<usize>) {
    let mut asm = code.asm();
    asm.mov_load(Size::Dword, Rax, guest(vector_offset(source)));
    asm.mov(Size::Dword, Rdx, Rax);
    asm.alu_imm32(Alu::And, Size::Dword, Rdx, i32::MAX);
    let zero = branch(code, Condition::Zero);
    let mut asm = code.asm();
    asm.alu_imm32(Alu::Sub, Size::Dword, Rdx, FIELD_NARROW_MINIMUM as i32);
    asm.alu_imm32(
        Alu::Cmp,
        Size::Dword,
        Rdx,
        (FIELD_NARROW_MAXIMUM - FIELD_NARROW_MINIMUM) as i32,
    );
    fallback.push(branch(code, Condition::Above));
    // The minimum has no fraction bits, so EDX keeps the dropped bits and the
    // kept least significant bit of the magnitude.
    let mut asm = code.asm();
    asm.mov(Size::Dword, Rcx, Rdx);
    asm.alu_imm32(Alu::And, Size::Dword, Rcx, FIELD_NARROW_DROPPED_MASK as i32);
    asm.unary(Unary::Neg, Size::Dword, Rcx);
    asm.alu(Alu::Sbb, Size::Dword, Rcx, Rcx);
    asm.bit_imm(BitTest::Bt, Size::Dword, Rdx, FIELD_NARROW_DROPPED_BITS);
    asm.alu_imm32(
        Alu::Adc,
        Size::Dword,
        Rdx,
        (FIELD_NARROW_MINIMUM - FIELD_NARROW_REBIAS + FIELD_NARROW_ROUND) as i32,
    );
    asm.shift_imm(ShiftOp::Shr, Size::Dword, Rdx, FIELD_NARROW_DROPPED_BITS);
    asm.shift_imm(ShiftOp::Shr, Size::Dword, Rax, HALF_TO_SINGLE_SIGN_SHIFT);
    asm.alu_imm32(Alu::And, Size::Dword, Rax, HALF_SIGN as i32);
    asm.alu(Alu::Or, Size::Dword, Rax, Rdx);
    let done = jump(code);
    let zero_start = code.len();
    resolve(code, zero, zero_start);
    // Only the sign is left above a zero magnitude.
    let mut asm = code.asm();
    asm.shift_imm(ShiftOp::Shr, Size::Dword, Rax, HALF_TO_SINGLE_SIGN_SHIFT);
    asm.alu(Alu::Xor, Size::Dword, Rcx, Rcx);
    let end = code.len();
    resolve(code, done, end);
}

/// FCVTN Vd.4H, Vn.4S from the Single fields in SSE2 integer lanes: every
/// lane a zero or a magnitude in the field narrowing range, else `fallback`.
/// Leaves the four Halves in the low qword of `RESULT_VECTOR` with a zero
/// high qword, and ECX with every bit set when some lane is inexact and zero
/// otherwise.
fn vector_field_narrow(code: &mut Code, source: u8, fallback: &mut Vec<usize>) {
    read_vector(code, source, SOURCE_VECTOR);
    field_narrow_lanes(code, fallback);
}

/// [`vector_field_narrow`] of the four Single lanes already in XMM1, under
/// FPCR nearest-even, with the same results. Clobbers only XMM0-XMM6, EAX
/// and ECX.
pub(super) fn field_narrow_lanes(code: &mut Code, fallback: &mut Vec<usize>) {
    const MAGNITUDE: u8 = 2;
    const VALID: u8 = 3;
    const SCRATCH: u8 = 4;
    const ZERO: u8 = 5;
    const BOUND: u8 = 6;
    packed(code, Host::MoveAligned, MAGNITUDE, SOURCE_VECTOR);
    ShiftOpcode::LeftDword.emit(code, MAGNITUDE, 1);
    ShiftOpcode::RightDword.emit(code, MAGNITUDE, 1);
    dword_constant(code, BOUND, FIELD_NARROW_MINIMUM - 1);
    packed(code, Host::MoveAligned, SCRATCH, MAGNITUDE);
    packed(code, Host::GreaterDword, SCRATCH, BOUND);
    dword_constant(code, VALID, FIELD_NARROW_MAXIMUM + 1);
    packed(code, Host::GreaterDword, VALID, MAGNITUDE);
    packed(code, Host::And, VALID, SCRATCH);
    packed(code, Host::Xor, ZERO, ZERO);
    packed(code, Host::MoveAligned, SCRATCH, MAGNITUDE);
    packed(code, Host::EqualDword, SCRATCH, ZERO);
    packed(code, Host::Or, SCRATCH, VALID);
    Host::DwordSignMask.register(code, 0, SCRATCH, TargetCpu::Baseline);
    code.asm()
        .alu_imm8(Alu::Cmp, Size::Dword, Rax, ALL_DWORD_LANES);
    fallback.push(branch(code, Condition::NotZero));
    // A zero lane has no dropped bits either.
    packed(code, Host::MoveAligned, SCRATCH, MAGNITUDE);
    ShiftOpcode::LeftDword.emit(code, SCRATCH, u32::BITS as u8 - FIELD_NARROW_DROPPED_BITS);
    packed(code, Host::EqualDword, SCRATCH, ZERO);
    Host::DwordSignMask.register(code, 0, SCRATCH, TargetCpu::Baseline);
    let mut asm = code.asm();
    asm.mov(Size::Dword, Rcx, Rax);
    asm.alu_imm8(Alu::Xor, Size::Dword, Rcx, ALL_DWORD_LANES);
    asm.unary(Unary::Neg, Size::Dword, Rcx);
    asm.alu(Alu::Sbb, Size::Dword, Rcx, Rcx);
    // Round each valid lane: its kept least significant bit, then the rebias
    // and half the dropped weight less one.
    packed(code, Host::MoveAligned, SCRATCH, MAGNITUDE);
    ShiftOpcode::RightDword.emit(code, SCRATCH, FIELD_NARROW_DROPPED_BITS);
    ShiftOpcode::LeftDword.emit(code, SCRATCH, DWORD_SIGN_TO_LSB_SHIFT);
    ShiftOpcode::RightDword.emit(code, SCRATCH, DWORD_SIGN_TO_LSB_SHIFT);
    packed(code, Host::AddDword, SCRATCH, MAGNITUDE);
    dword_constant(
        code,
        BOUND,
        FIELD_NARROW_ROUND.wrapping_sub(FIELD_NARROW_REBIAS),
    );
    packed(code, Host::AddDword, SCRATCH, BOUND);
    ShiftOpcode::RightDword.emit(code, SCRATCH, FIELD_NARROW_DROPPED_BITS);
    packed(code, Host::And, SCRATCH, VALID);
    ShiftOpcode::RightDword.emit(code, SOURCE_VECTOR, DWORD_SIGN_TO_LSB_SHIFT);
    ShiftOpcode::LeftDword.emit(code, SOURCE_VECTOR, HALF_SIGN.trailing_zeros() as u8);
    packed(code, Host::Or, SCRATCH, SOURCE_VECTOR);
    // Sign-extend each Half to its dword so that the saturating pack is
    // exact, and pack against zero.
    ShiftOpcode::LeftDword.emit(code, SCRATCH, u16::BITS as u8);
    ShiftOpcode::ArithmeticDword.emit(code, SCRATCH, u16::BITS as u8);
    packed(code, Host::PackSignedDwordToWord, SCRATCH, ZERO);
    packed(code, Host::MoveAligned, RESULT_VECTOR, SCRATCH);
}

/// Single to Half narrowing (`field_narrowing`) under FPCR nearest-even, AHP
/// clear and no enabled trap, from the Single fields alone: zeros, and
/// magnitudes whose Half is normal and finite. Such an input is not tiny, so
/// the only status is inexact, from the dropped fraction bits; no host FP
/// instruction runs and MXCSR stays untouched, on every target. Other inputs
/// and controls take the F16C recipe where the host has it, else the helper.
fn emit_field_narrow_half(
    code: &mut Code,
    operand: &PrecisionOperand,
    pc: GuestAddress,
    target: TargetCpu,
) {
    let mut fallback = Vec::new();
    test_control(
        code,
        FPCR_EXCEPTION_ENABLE_MASK | FPCR_AHP | FPCR_ROUNDING_MASK << FPCR_ROUNDING_SHIFT,
    );
    fallback.push(branch(code, Condition::NotZero));
    match operand.mode {
        PrecisionMode::Scalar { .. } => {
            scalar_field_narrow(code, operand.source, &mut fallback);
            commit_inexact(code);
            read_vector(code, operand.destination, RESULT_VECTOR);
            code.asm().pinsrw(RESULT_VECTOR, Rax, 0);
            scalar_commit(code, operand.destination, Format::Half);
        }
        PrecisionMode::Narrow { upper, .. } => {
            vector_field_narrow(code, operand.source, &mut fallback);
            commit_inexact(code);
            if upper {
                read_vector(code, operand.destination, SOURCE_VECTOR);
                code.asm().movlhps(SOURCE_VECTOR, Rm::Xmm(RESULT_VECTOR));
                commit(code, operand.destination, SOURCE_VECTOR);
            } else {
                commit(code, operand.destination, RESULT_VECTOR);
            }
        }
        PrecisionMode::Widen { .. } => unreachable!("field narrowing narrows"),
    }
    let done = jump(code);
    let fallback_start = code.len();
    for patch in fallback {
        resolve(code, patch, fallback_start);
    }
    if half_narrowing_supported(operand.mode, target) {
        emit_narrow_half(code, operand, pc, target);
    } else {
        super::super::block::direct_exit(code, pc, ExitKind::FloatingPoint);
    }
    let end = code.len();
    resolve(code, done, end);
}

pub(in crate::x86_64) fn emit(
    code: &mut Code,
    operand: &PrecisionOperand,
    pc: GuestAddress,
    target: TargetCpu,
) {
    if field_narrowing(operand.mode) && !target.data_independent_timing() {
        emit_field_narrow_half(code, operand, pc, target);
        return;
    }
    if half_narrowing_supported(operand.mode, target) {
        emit_narrow_half(code, operand, pc, target);
        return;
    }
    match operand.mode {
        PrecisionMode::Scalar {
            source,
            destination,
            rounding,
        } if destination != Format::Half => {
            emit_scalar(code, operand, source, destination, rounding, pc);
        }
        PrecisionMode::Widen {
            source: Format::Half,
            upper,
        } => emit_widen_half(code, operand, upper, pc),
        PrecisionMode::Widen {
            source: Format::Single,
            upper,
        } => emit_widen(code, operand, upper, pc),
        PrecisionMode::Narrow {
            destination: Format::Single,
            upper,
            rounding,
        } => emit_narrow(code, operand, upper, rounding, pc),
        _ => super::super::block::direct_exit(code, pc, ExitKind::FloatingPoint),
    }
}
