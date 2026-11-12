//! Shared FP emission: integer lane classification and a private MXCSR.
use crate::x86_64::asm::{Alu, Code, Cond, Gpr, Map, Mem, Pp, Rm, ShiftOp, Size, Vex};
use crate::x86_64::asm::{FloatLanes, FloatOp};
use alloc::vec;
use alloc::vec::Vec;
mod control;

use super::super::isa::ShiftOpcode;
use super::{Host, packed, packed_imm, read_vector, vector_offset};
use crate::{
    GuestState,
    aarch64::{floating::Format, simd::arithmetic::Shape},
};
use Gpr::{Rax, Rcx, Rdi, Rsp};
pub(in crate::x86_64) use control::rounding_control;

const MXCSR_EXCEPTION_MASKS: u32 = 0x3f << 7;
pub(super) const MXCSR_INEXACT: u32 = 1 << 5;
const MXCSR_OTHER_STATUS: u32 = MXCSR_INEXACT - 1;
/// Every MXCSR status flag.
const MXCSR_STATUS_FLAGS: u8 = 0x3f;
const MXCSR_POSITIVE_ROUNDING: u32 = 1 << 14;
const MXCSR_NEGATIVE_ROUNDING: u32 = 1 << 13;
const FPCR_POSITIVE_TO_MXCSR_SHIFT: u8 = 8;
const FPCR_NEGATIVE_TO_MXCSR_SHIFT: u8 = 10;
const MXCSR_TO_FPSR_INEXACT_SHIFT: u8 = 1;
pub(super) const INPUT_MINIMUM_EXPONENT: u8 = 1;
pub(super) const RESULT_MINIMUM_EXPONENT: u8 = 2;
pub(super) const LEFT_VECTOR: u8 = 0;
pub(super) const RIGHT_VECTOR: u8 = 1;
pub(super) const EXPONENT_VECTOR: u8 = 2;
const VALIDITY_VECTOR: u8 = 3;
const MAXIMUM_EXPONENT_VECTOR: u8 = 4;
pub(super) const MINIMUM_EXPONENT_VECTOR: u8 = 5;
const SINGLE_VECTOR_MASK: u8 = 0x0f;
const SINGLE_LOW_VECTOR_MASK: u8 = 0x03;
const DOUBLE_VECTOR_MASK: u8 = 0x05;
const SCALAR_VECTOR_MASK: u8 = 1;
const HALF_SCALAR_BYTE_MASK: u32 = 3;
const HALF_LOW_BYTE_MASK: u32 = 0xff;
const HALF_VECTOR_BYTE_MASK: u32 = 0xffff;
const SIGN_BITS: u8 = 1;
pub(super) const SINGLE_EXPONENT_BIAS_BITS: u8 = 7;
pub(super) const ENVIRONMENT_STACK_BYTES: u8 = 16;
/// The environment frame: the saved MXCSR, then the installed control,
/// then the captured status, then a product's status.
const MXCSR_SAVED_OFFSET: u8 = 0;
const MXCSR_CONTROL_OFFSET: u8 = 4;
const MXCSR_STATUS_OFFSET: u8 = 8;
const MXCSR_PRODUCT_OFFSET: u8 = 12;
/// The FMA addend of the shared numerical allocation.
const ADDEND_VECTOR: u8 = 6;
const FMA_SCALAR_OPCODE_BIT: u8 = 1;

pub(in crate::x86_64) enum Condition {
    Below,
    AboveOrEqual,
    Zero,
    NotZero,
    Above,
    /// PF set: an unordered UCOMISS/UCOMISD.
    Parity,
}

impl Condition {
    const fn cond(self) -> Cond {
        match self {
            Self::Below => Cond::B,
            Self::AboveOrEqual => Cond::Ae,
            Self::Zero => Cond::E,
            Self::NotZero => Cond::Ne,
            Self::Above => Cond::A,
            Self::Parity => Cond::P,
        }
    }
}

/// `[rdi + offset]` in GuestState, with a 32-bit displacement.
pub(in crate::x86_64) fn state(offset: usize) -> Mem {
    Mem::base_disp32(Rdi, offset as i32)
}

/// `[rsp + offset]` in the environment frame.
fn frame(offset: u8) -> Mem {
    Mem::base(Rsp, offset.into())
}

#[repr(u8)]
pub(in crate::x86_64) enum ArithmeticOpcode {
    Add = 0x58,
    Subtract = 0x5c,
    Multiply = 0x59,
    Divide = 0x5e,
}

impl From<ArithmeticOpcode> for crate::x86_64::asm::vector_extensions::HalfOpcode {
    fn from(opcode: ArithmeticOpcode) -> Self {
        match opcode {
            ArithmeticOpcode::Add => Self::Add,
            ArithmeticOpcode::Subtract => Self::Subtract,
            ArithmeticOpcode::Multiply => Self::Multiply,
            ArithmeticOpcode::Divide => Self::Divide,
        }
    }
}

impl ArithmeticOpcode {
    const fn float_op(self) -> FloatOp {
        match self {
            Self::Add => FloatOp::Add,
            Self::Subtract => FloatOp::Sub,
            Self::Multiply => FloatOp::Mul,
            Self::Divide => FloatOp::Div,
        }
    }
}

/// The SSE lane form of a `format` operation over `shape`: scalar Half
/// operates on its exact Double widening.
pub(in crate::x86_64) const fn float_lanes(format: Format, shape: Shape) -> FloatLanes {
    match (shape, format) {
        (Shape::Scalar, Format::Single) => FloatLanes::ScalarSingle,
        (Shape::Scalar, _) => FloatLanes::ScalarDouble,
        (_, Format::Double) => FloatLanes::PackedDouble,
        _ => FloatLanes::PackedSingle,
    }
}

#[repr(u8)]
pub(in crate::x86_64) enum FusedOpcode {
    MultiplyAdd = 0x98,
    MultiplySubtract = 0x9a,
    NegativeMultiplyAdd = 0x9c,
    NegativeMultiplySubtract = 0x9e,
}

impl From<crate::aarch64::floating::fused::FusedOperation> for FusedOpcode {
    fn from(operation: crate::aarch64::floating::fused::FusedOperation) -> Self {
        use crate::aarch64::floating::fused::FusedOperation;
        match operation {
            FusedOperation::Add => Self::MultiplyAdd,
            FusedOperation::Subtract => Self::NegativeMultiplyAdd,
            FusedOperation::NegatedAdd => Self::NegativeMultiplySubtract,
            FusedOperation::NegatedSubtract => Self::MultiplySubtract,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Rounding {
    Fpcr,
    NearestEven,
    Positive,
    Negative,
    Zero,
}

mod bfloat16;
pub(in crate::x86_64) use bfloat16::{quantize_bfloat_packed, quantize_bfloat_scalar_bits};
mod quantization;

pub(in crate::x86_64) use quantization::{
    IntegerRoundingRegister, broadcast_integer_rounding, integer_rounding_policy,
};

/// Checked FMA132 with the shared numerical allocation: XMM0 is the product
/// destination, XMM1 the multiplier and XMM6 the addend. All are caller-volatile.
pub(in crate::x86_64) fn emit_fma(
    code: &mut Code,
    format: Format,
    shape: Shape,
    opcode: FusedOpcode,
    target: crate::TargetCpu,
) {
    assert!(
        target.supports(crate::TargetCpu::Fma),
        "FMA3 requires usable AVX/FMA state"
    );
    // 132 order: XMM0 = XMM0 * XMM1 + XMM6, with the selected sign variant.
    code.asm().vex(
        Vex {
            map: Map::M0f38,
            pp: Pp::P66,
            w: format == Format::Double,
            l: false,
        },
        opcode as u8
            | if shape == Shape::Scalar {
                FMA_SCALAR_OPCODE_BIT
            } else {
                0
            },
        LEFT_VECTOR,
        ADDEND_VECTOR,
        Rm::Xmm(RIGHT_VECTOR),
    );
}

/// A `jcc rel32` to resolve later, returning its displacement field.
pub(in crate::x86_64) fn branch(code: &mut Code, condition: Condition) -> usize {
    code.asm().jcc(condition.cond()).offset()
}
pub(in crate::x86_64) fn resolve(code: &mut Code, patch: usize, target: usize) {
    let target = i64::try_from(target).expect("numerical branch target fits i64");
    let next = i64::try_from(patch + size_of::<i32>()).expect("numerical branch source fits i64");
    let distance = i32::try_from(target - next).expect("bounded numerical branch");
    code.patch(patch, distance.to_le_bytes());
}
/// Refuse kept numerical enables unless AH suppresses exceptions.
pub(super) fn guard_ah0_exceptions(code: &mut Code, helper: &mut Vec<usize>) {
    use crate::aarch64::floating::{FPCR_AH, FPCR_EXCEPTION_ENABLE_MASK};
    test_control(code, FPCR_AH);
    let suppressed = branch(code, Condition::NotZero);
    test_control(code, FPCR_EXCEPTION_ENABLE_MASK);
    helper.push(branch(code, Condition::NotZero));
    let enabled = code.len();
    resolve(code, suppressed, enabled);
}

/// Select guest RC for AH0 and mandatory nearest-even for AH1.
/// Inputs are already normal, so the forced AH1 flush controls are immaterial.
pub(super) fn environment_alternative_nearest(code: &mut Code) {
    use crate::aarch64::floating::FPCR_AH;
    test_control(code, FPCR_AH);
    let nearest = branch(code, Condition::NotZero);
    environment(code);
    let done = jump(code);
    let nearest_start = code.len();
    resolve(code, nearest, nearest_start);
    environment_round(code, Rounding::NearestEven);
    let end = code.len();
    resolve(code, done, end);
}

/// Commit the accepted host inexact status, only when FPCR.AH allows fresh
/// status.
pub(super) fn commit_ah0_inexact(code: &mut Code) {
    use crate::aarch64::floating::FPCR_AH;
    test_control(code, FPCR_AH);
    let suppressed = branch(code, Condition::NotZero);
    commit_inexact(code);
    let end = code.len();
    resolve(code, suppressed, end);
}

/// The FPCR fields outside whose defaults the sticky-inexact path
/// ([`sticky_inexact`]) declines: FIZ, AH, the exception enables, RMode
/// and FZ change some S/D result or its status that the path's result
/// classification cannot see, and NEP, which the path does not merge,
/// changes scalar upper bits. DN only selects a NaN result, which the
/// classification sends to the helper; FZ16 and AHP concern Half only.
pub(super) const FPCR_STICKY_MASK: u32 = {
    use crate::aarch64::floating::{
        FPCR_AH, FPCR_EXCEPTION_ENABLE_MASK, FPCR_FIZ, FPCR_FZ, FPCR_NEP, FPCR_ROUNDING_MASK,
        FPCR_ROUNDING_SHIFT,
    };
    FPCR_FIZ
        | FPCR_AH
        | FPCR_NEP
        | FPCR_EXCEPTION_ENABLE_MASK
        | FPCR_ROUNDING_MASK << FPCR_ROUNDING_SHIFT
        | FPCR_FZ
};
/// FPSR.IXC.
const FPSR_INEXACT: u8 = 1 << 4;

/// Enter the sticky-inexact path, or take one of the returned branches to
/// the full path. With FPCR at its defaults ([`FPCR_STICKY_MASK`]) and
/// FPSR.IXC already set, an S/D operation whose result its caller then
/// classifies needs neither its own MXCSR control nor its status. It runs
/// in the MXCSR native code is entered with: the block ABI's power-on
/// control (all exceptions masked, nearest rounding, no DAZ or FTZ), which
/// the SysV ABI preserves across calls and every environment sequence here
/// restores; its status flags are never read. Without DAZ, FZ and FIZ both
/// sides compute subnormal inputs exactly and without guest status, so no
/// input needs classifying. A fresh inexact result would only set IXC
/// again; every other status the guest would see needs a NaN or infinite
/// result, or, for an operation whose tiny results may be inexact, a result
/// below twice the minimum normal, which the caller's classification sends
/// to the exact helper.
///
/// Unless `entry.checked`, the caller knows the conditions hold
/// ([`Entry`]). There are then no branches.
pub(super) fn sticky_inexact(code: &mut Code, entry: Entry) -> Vec<usize> {
    if !entry.checked {
        return vec![];
    }
    test_control(code, FPCR_STICKY_MASK);
    let control = branch(code, Condition::NotZero);
    code.asm().test_imm(
        Size::Byte,
        state(core::mem::offset_of!(GuestState, fpsr)),
        FPSR_INEXACT.into(),
    );
    let status = branch(code, Condition::Zero);
    vec![control, status]
}

/// What the block knows where a numerical instruction begins.
#[derive(Clone, Copy)]
pub(in crate::x86_64) struct Entry {
    /// Whether the sticky-inexact path checks its conditions
    /// ([`sticky_inexact`]). The block may omit the check where the
    /// previous instruction's sticky path falls through, if that
    /// instruction's full path continues in this one's ([`ColdSplit`]).
    pub(in crate::x86_64) checked: bool,
    /// The guest vector XMM0 holds whole, which the sticky path's inputs
    /// may take from there ([`read_inputs`]).
    pub(in crate::x86_64) xmm0: Option<u8>,
}

impl Entry {
    /// Nothing known.
    pub(in crate::x86_64) const UNKNOWN: Self = Self {
        checked: true,
        xmm0: None,
    };
}

/// An S/D emission with a sticky-inexact path ([`sticky_inexact`]), laid
/// out inline: the sticky path, a `jmp rel32` past the rest, then the rest
/// (zero-product check, full path, helper exit), which ends where the
/// emission ends. A block may move the rest out of line, re-resolving
/// `crossing` and ending the moved part with a jump to where the emission
/// ended. Neither part accesses guest memory.
pub(in crate::x86_64) struct ColdSplit {
    /// Start of the rest, just after the `jmp` past it.
    pub(in crate::x86_64) cold: crate::x86_64::asm::Checkpoint,
    /// End of the inline path, before its jump over the cold path.
    pub(in crate::x86_64) inline_end: crate::x86_64::asm::Checkpoint,
    /// The full path's entry, correct in every FPCR and FPSR state.
    pub(in crate::x86_64) full: usize,
    /// Every rel32 branch field in one part that targets the other.
    pub(in crate::x86_64) crossing: Vec<usize>,
    /// The destination vector, which XMM0 holds whole wherever the
    /// emission ends: every path there ends with its commit from XMM0.
    pub(in crate::x86_64) result: u8,
}

/// Jump over a cold section that exits to software completion at `pc`, and
/// point each `helper` branch at that section. Every branch enters it with
/// the caller's MXCSR and stack restored; success continues after it.
pub(in crate::x86_64) fn helper_section(
    code: &mut Code,
    helper: &[usize],
    pc: crate::GuestAddress,
) {
    let completed = jump(code);
    let helper_start = code.len();
    super::super::block::direct_exit(code, pc, crate::block::ExitKind::FloatingPoint);
    let success = code.len();
    for &patch in helper {
        resolve(code, patch, helper_start);
    }
    resolve(code, completed, success);
}

/// Packed single-precision `opcode` on whole XMM registers.
pub(in crate::x86_64) fn single_vector_arithmetic(
    code: &mut Code,
    opcode: ArithmeticOpcode,
    destination: u8,
    source: u8,
) {
    emit_opcode_to(
        code,
        crate::aarch64::floating::Format::Single,
        crate::aarch64::simd::arithmetic::Shape::Vector128,
        opcode,
        destination,
        source,
    );
}

/// An unconditional `jmp rel32` to resolve later.
pub(in crate::x86_64) fn jump(code: &mut Code) -> usize {
    code.asm().jmp().offset()
}

/// Classify the active result lanes in `source` for the sticky-inexact
/// path and return the branch to the helper. Every NaN or infinite lane
/// takes it; unless `exact_tiny` (addition and subtraction, whose results
/// below the minimum normal are exact, and roots, which are never tiny),
/// so does every lane below twice the minimum normal, which may be tiny
/// before rounding, zero included.
pub(super) fn sticky_result_check(
    code: &mut Code,
    format: Format,
    shape: Shape,
    source: u8,
    exact_tiny: bool,
    target: crate::TargetCpu,
) -> usize {
    if exact_tiny && shape != Shape::Scalar && quick_finite_supported(target) {
        quick_finite(code, format, shape, source, target);
        return classified_rejection(code, format, shape, source, exact_tiny);
    }
    if !exact_tiny && shape == Shape::Vector128 && quick_normal_supported(target) {
        quick_normal(code, format, source);
        return classified_rejection(code, format, shape, source, exact_tiny);
    }
    if shape == Shape::Scalar {
        let minimum = if exact_tiny {
            0
        } else {
            RESULT_MINIMUM_EXPONENT
        };
        if format == Format::Single {
            single_exponent_range(code, source, minimum);
        } else {
            exponent_range(code, format, source, minimum);
        }
        return branch(code, Condition::AboveOrEqual);
    }
    vector_result_classification(code, format, shape, source, exact_tiny)
}

/// The full lane classification of [`sticky_result_check`] for a vector:
/// the branch to the rejection path, which finds the range constants in
/// XMM4 (and XMM5 for products) and the accepted-lane mask in
/// EXPONENT_VECTOR ([`zero_product_check`] completes from it).
fn vector_result_classification(
    code: &mut Code,
    format: Format,
    shape: Shape,
    source: u8,
    exact_tiny: bool,
) -> usize {
    prepare_vector_maximum(code, format);
    if exact_tiny {
        vector_below_maximum(code, format, shape, source);
    } else {
        prepare_vector_result_range(code);
        vector_exponent_range(code, format, shape, source);
    }
    branch(code, Condition::NotZero)
}

/// After a quick classification that set ZF exactly when it accepts every
/// lane: accepted vectors branch past the full classification, which the
/// others then run to leave the registers its rejection path reads. Both
/// reject the same lanes, so the returned branch is always taken there.
fn classified_rejection(
    code: &mut Code,
    format: Format,
    shape: Shape,
    source: u8,
    exact_tiny: bool,
) -> usize {
    let accepted = branch(code, Condition::Zero);
    let rejected = vector_result_classification(code, format, shape, source, exact_tiny);
    let end = code.len();
    resolve(code, accepted, end);
    rejected
}

/// Whether [`quick_finite`] runs on `target`: PTEST (SSE4.1). The DIT
/// profile keeps its reviewed classification.
fn quick_finite_supported(target: crate::TargetCpu) -> bool {
    super::three::test_zero_supported(target) && !target.data_independent_timing()
}

/// ZF clear when an active S/D lane of the vector in `source` is NaN or
/// infinite: `x - x` in the entry MXCSR (nearest, no DAZ) is +0 for
/// every finite lane and NaN otherwise, so one PTEST of the difference
/// classifies every lane. An 8-byte vector's inactive high qword is cleared
/// first. Clobbers EXPONENT_VECTOR and the entry MXCSR's status, which is
/// never read.
fn quick_finite(
    code: &mut Code,
    format: Format,
    shape: Shape,
    source: u8,
    target: crate::TargetCpu,
) {
    let lanes = if format == Format::Double {
        FloatLanes::PackedDouble
    } else {
        FloatLanes::PackedSingle
    };
    let held = if shape == Shape::Vector64 {
        packed(code, Host::MoveQword, EXPONENT_VECTOR, source);
        EXPONENT_VECTOR
    } else {
        source
    };
    if super::three::vex(target) {
        super::three::float(code, FloatOp::Sub, lanes, EXPONENT_VECTOR, held, held);
    } else {
        if held != EXPONENT_VECTOR {
            packed(code, Host::MoveAligned, EXPONENT_VECTOR, held);
        }
        code.asm().float(
            FloatOp::Sub,
            lanes,
            EXPONENT_VECTOR,
            Rm::Xmm(EXPONENT_VECTOR),
        );
    }
    super::three::test_zero(code, EXPONENT_VECTOR, target);
}

/// Whether [`quick_normal`] runs on `target`: AVX-512 F, VL and DQ
/// (VFPCLASS). The DIT profile keeps its reviewed classification.
fn quick_normal_supported(target: crate::TargetCpu) -> bool {
    use crate::TargetCpu;
    target.supports(TargetCpu::Avx512f.with_avx512vl().with_avx512dq())
        && !target.data_independent_timing()
}

/// ZF clear when a lane of the 16-byte S/D vector in `source` is NaN,
/// infinite or below twice the minimum normal (zero included), the
/// lanes [`sticky_result_check`] rejects for a product: half of each lane
/// in the entry MXCSR (nearest, no DAZ or FTZ) is zero, subnormal,
/// infinite or NaN exactly for those lanes (an exponent field of 0, 1 or
/// all ones; halving is exact above exponent 1, and a subnormal half stays
/// subnormal or rounds to zero), which one VFPCLASS tells. Clobbers RAX,
/// EXPONENT_VECTOR, k1 and the entry MXCSR's status, which is never read.
fn quick_normal(code: &mut Code, format: Format, source: u8) {
    const ZERO_SUBNORMAL_INFINITE_NAN: u8 = 0xbf;
    const VFPCLASS: u8 = 0x66;
    const KORTESTW: u8 = 0x98;
    const MASK: u8 = 1;
    let double = format == Format::Double;
    let lanes = if double {
        FloatLanes::PackedDouble
    } else {
        FloatLanes::PackedSingle
    };
    let mut asm = code.asm();
    if double {
        asm.mov_imm64(Rax, 0.5f64.to_bits());
    } else {
        asm.mov_imm32(Rax, 0.5f32.to_bits());
    }
    super::three::broadcast_general(code, EXPONENT_VECTOR, Rax, double);
    super::three::float(
        code,
        FloatOp::Mul,
        lanes,
        EXPONENT_VECTOR,
        EXPONENT_VECTOR,
        source,
    );
    let mut asm = code.asm();
    asm.evex(
        crate::x86_64::asm::Evex {
            map: Map::M0f3a,
            pp: Pp::P66,
            w: double,
            length: 0,
            mask: 0,
            zeroing: false,
            broadcast: false,
            scale: 16,
        },
        VFPCLASS,
        MASK,
        0,
        Rm::Xmm(EXPONENT_VECTOR),
    );
    asm.data(&[ZERO_SUBNORMAL_INFINITE_NAN]);
    asm.vex(
        Vex {
            map: Map::M0f,
            pp: Pp::None,
            w: false,
            l: false,
        },
        KORTESTW,
        MASK,
        0,
        Rm::Xmm(MASK),
    );
}

/// [`exponent_range`] of a Single in `source` (XMM0 or XMM1) with one
/// instruction fewer: doubling its bits drops the sign and puts the
/// exponent field at the top, so the range bounds shift with it.
fn single_exponent_range(code: &mut Code, source: u8, minimum: u8) {
    const FIELD_SHIFT: u32 = u32::BITS - 8;
    let maximum = (Format::Single.exponent_mask() >> Format::Single.fraction_bits()) as u32;
    assert!(
        matches!(source, 0 | 1),
        "allocated numerical source register"
    );
    let mut asm = code.asm();
    asm.movd_from_xmm(Rax, source);
    asm.alu(Alu::Add, Size::Dword, Rax, Rax);
    if minimum != 0 {
        asm.alu_accumulator_imm32(
            Alu::Sub,
            Size::Dword,
            (u32::from(minimum) << FIELD_SHIFT) as i32,
        );
    }
    asm.alu_accumulator_imm32(
        Alu::Cmp,
        Size::Dword,
        ((maximum - u32::from(minimum)) << FIELD_SHIFT) as i32,
    );
}

/// `op eax, imm8`.
fn eax_imm8(code: &mut Code, op: Alu, value: u8) {
    code.asm().alu_imm8(op, Size::Dword, Rax, value as i8);
}

/// `op eax, imm32`, the accumulator short form.
fn eax_imm32(code: &mut Code, op: Alu, value: u32) {
    code.asm()
        .alu_accumulator_imm32(op, Size::Dword, value as i32);
}

/// Leave ZF set when every active S/D lane of `source` has an exponent
/// below all ones, with the maximum from [`prepare_vector_input_range`].
/// The high dwords of Double lanes hold zero, which is below it too.
fn vector_below_maximum(code: &mut Code, format: Format, shape: Shape, source: u8) {
    vector_exponent_fields(code, format, source, false);
    packed(
        code,
        Host::MoveAligned,
        VALIDITY_VECTOR,
        MAXIMUM_EXPONENT_VECTOR,
    );
    packed(code, Host::GreaterDword, VALIDITY_VECTOR, EXPONENT_VECTOR);
    Host::DwordSignMask.register(code, 0, VALIDITY_VECTOR, crate::TargetCpu::Baseline);
    let mask = match (shape, format) {
        (Shape::Vector64, Format::Single) => {
            eax_imm8(code, Alu::And, SINGLE_LOW_VECTOR_MASK);
            SINGLE_LOW_VECTOR_MASK
        }
        (Shape::Vector128, Format::Single | Format::Double) => SINGLE_VECTOR_MASK,
        _ => unreachable!("allocated S/D vector shape"),
    };
    eax_imm8(code, Alu::Cmp, mask);
}

/// The operands of a product whose result the sticky classification
/// rejected, for [`zero_product_check`]: the result is in LEFT_VECTOR, the
/// right input in `right`, the left source (overwritten by the result)
/// still in guest vector `left` of the frame, and a fused addend in
/// `addend`.
pub(super) struct ZeroProduct {
    pub format: Format,
    pub shape: Shape,
    pub right: u8,
    pub left: u8,
    pub addend: Option<u8>,
}

/// Zero-product check registers; EXPONENT_VECTOR keeps the classifier's
/// lane validity, and the right input's register is reused for the left
/// once its own zeros are known.
const ZERO_RESULT: u8 = 3;
const ZERO_INPUT: u8 = 4;
const ZERO_PAIR: u8 = 5;
const ZERO_VECTOR: u8 = 7;
/// PSHUFD selector exchanging the two dwords of each qword.
const SWAP_DWORD_PAIRS: u8 = 0xb1;
/// MOVMSKPS lanes that are active: the low dword of each D lane holds its
/// classification.
const DOUBLE_LOW_DWORDS: u8 = 0b0101;
/// MOVMSKPS lanes of the high dword of each D lane, which holds its exponent.
const DOUBLE_HIGH_DWORDS: u8 = 0b1010;

/// Out of line after a sticky-inexact product (`sticky_inexact`): return
/// to `resume` when every active lane the classification rejected is a
/// zero whose product has a zero input, with a zero addend for a fused
/// product, which makes it exact and without status, and otherwise take
/// one of the returned branches to the helper; `resumes` receives the
/// branches to `resume`. Other zeros and tiny
/// results may have underflowed, and need the helper.
pub(super) fn zero_product_check(
    code: &mut Code,
    product: &ZeroProduct,
    resume: usize,
    resumes: &mut Vec<usize>,
) -> Vec<usize> {
    let ZeroProduct {
        format,
        shape,
        right,
        left,
        addend,
    } = *product;
    if shape == Shape::Scalar {
        let mut helper = vec![];
        for source in core::iter::once(LEFT_VECTOR).chain(addend) {
            zero_scalar(code, format, source);
            helper.push(branch(code, Condition::NotZero));
        }
        zero_scalar(code, format, right);
        let right_zero = branch(code, Condition::Zero);
        resolve(code, right_zero, resume);
        resumes.push(right_zero);
        let size = Size::of(format == Format::Double);
        let mut asm = code.asm();
        asm.mov_load(size, Rax, state(vector_offset(left) as usize));
        asm.alu(Alu::Add, size, Rax, Rax);
        let left_zero = branch(code, Condition::Zero);
        resolve(code, left_zero, resume);
        resumes.push(left_zero);
        helper.push(jump(code));
        return helper;
    }
    packed(code, Host::Xor, ZERO_VECTOR, ZERO_VECTOR);
    zero_lanes(code, format, ZERO_INPUT, right);
    read_vector(code, left, right);
    zero_lanes(code, format, right, right);
    packed(code, Host::Or, ZERO_INPUT, right);
    zero_lanes(code, format, ZERO_RESULT, LEFT_VECTOR);
    packed(code, Host::And, ZERO_INPUT, ZERO_RESULT);
    if let Some(addend) = addend {
        zero_lanes(code, format, addend, addend);
        packed(code, Host::And, ZERO_INPUT, addend);
    }
    packed(code, Host::Or, EXPONENT_VECTOR, ZERO_INPUT);
    Host::DwordSignMask.register(code, 0, EXPONENT_VECTOR, crate::TargetCpu::Baseline);
    let mask = match (shape, format) {
        (Shape::Vector64, Format::Single) => SINGLE_LOW_VECTOR_MASK,
        (Shape::Vector128, Format::Single) => SINGLE_VECTOR_MASK,
        (Shape::Vector128, Format::Double) => DOUBLE_LOW_DWORDS,
        _ => unreachable!("allocated S/D vector shape"),
    };
    eax_imm8(code, Alu::And, mask);
    eax_imm8(code, Alu::Cmp, mask);
    let passed = branch(code, Condition::Zero);
    resolve(code, passed, resume);
    resumes.push(passed);
    vec![jump(code)]
}

/// Leave ZF set when the low S/D lane of `xmm` is a zero of either sign.
pub(super) fn zero_scalar(code: &mut Code, format: Format, xmm: u8) {
    let mut asm = code.asm();
    if format == Format::Double {
        asm.movq_from_xmm(Rax, xmm);
    } else {
        asm.movd_from_xmm(Rax, xmm);
    }
    let size = Size::of(format == Format::Double);
    asm.alu(Alu::Add, size, Rax, Rax);
}

/// Return the failure branch for a finite S/D exponent range or a
/// signed zero. Preserve ECX, which may hold the pending fresh host status.
pub(super) fn scalar_range_or_zero(
    code: &mut Code,
    format: Format,
    source: u8,
    minimum: u8,
) -> usize {
    scalar_range_or_special(code, format, source, minimum, false)
}

/// [`scalar_range_or_zero`] that also accepts an infinity of either sign:
/// every ordered value but a subnormal. Preserves ECX.
pub(super) fn scalar_ordered_range(
    code: &mut Code,
    format: Format,
    source: u8,
    minimum: u8,
) -> usize {
    scalar_range_or_special(code, format, source, minimum, true)
}

fn scalar_range_or_special(
    code: &mut Code,
    format: Format,
    source: u8,
    minimum: u8,
    infinite: bool,
) -> usize {
    assert!(format != Format::Half);
    let infinity = infinite.then(|| {
        infinity_scalar(code, format, source);
        branch(code, Condition::Zero)
    });
    zero_scalar(code, format, source);
    let zero = branch(code, Condition::Zero);
    let maximum = (format.exponent_mask() >> format.fraction_bits()) as u32;
    if format == Format::Single {
        // Doubling discarded the sign and moved the exponent above the fraction.
        let shift = format.fraction_bits() + u32::from(SIGN_BITS);
        eax_imm32(code, Alu::Sub, u32::from(minimum) << shift);
        eax_imm32(code, Alu::Cmp, (maximum - u32::from(minimum)) << shift);
    } else {
        code.asm().shift_imm(
            ShiftOp::Shr,
            Size::Qword,
            Rax,
            (format.fraction_bits() + u32::from(SIGN_BITS)) as u8,
        );
        eax_imm8(code, Alu::Sub, minimum);
        eax_imm32(code, Alu::Cmp, maximum - u32::from(minimum));
    }
    let failed = branch(code, Condition::AboveOrEqual);
    let end = code.len();
    resolve(code, zero, end);
    if let Some(infinity) = infinity {
        resolve(code, infinity, end);
    }
    failed
}

/// Set every S lane, or both dwords of every D lane, of `destination` to
/// all ones where `source` holds a zero of either sign, using ZERO_VECTOR
/// and, for D, ZERO_PAIR.
fn zero_lanes(code: &mut Code, format: Format, destination: u8, source: u8) {
    if destination != source {
        packed(code, Host::MoveAligned, destination, source);
    }
    if format == Format::Double {
        ShiftOpcode::LeftQword.emit(code, destination, SIGN_BITS);
    } else {
        ShiftOpcode::LeftDword.emit(code, destination, SIGN_BITS);
    }
    packed(code, Host::EqualDword, destination, ZERO_VECTOR);
    if format == Format::Double {
        packed_imm(
            code,
            Host::ShuffleDword,
            ZERO_PAIR,
            destination,
            SWAP_DWORD_PAIRS,
        );
        packed(code, Host::And, destination, ZERO_PAIR);
    }
}

pub(super) fn test_control(code: &mut Code, mask: u32) {
    code.asm().test_imm(
        Size::Dword,
        state(core::mem::offset_of!(GuestState, fpcr)),
        mask as i32,
    );
}

/// Leave flags from (exponent-minimum) versus the exclusive range length.
/// Below is allocated normal; zero/all-ones exponents fail one unsigned test.
pub(super) fn exponent_range(code: &mut Code, format: Format, source: u8, minimum: u8) {
    let maximum = (format.exponent_mask() >> format.fraction_bits()) as u32;
    exponent_range_impl(code, format, source, minimum, maximum, false);
}

/// A stricter exclusive upper exponent bounds finite arithmetic outputs too.
pub(super) fn exponent_range_below(
    code: &mut Code,
    format: Format,
    source: u8,
    minimum: u8,
    maximum: u32,
) {
    exponent_range_impl(code, format, source, minimum, maximum, false);
}

/// Keep the sign alongside the exponent so every negative input also fails.
pub(super) fn positive_exponent_range(code: &mut Code, format: Format, source: u8, minimum: u8) {
    let maximum = (format.exponent_mask() >> format.fraction_bits()) as u32;
    exponent_range_impl(code, format, source, minimum, maximum, true);
}

fn exponent_range_impl(
    code: &mut Code,
    format: Format,
    source: u8,
    minimum: u8,
    maximum: u32,
    positive_only: bool,
) {
    assert!(
        matches!(source, 0 | 1 | 6),
        "allocated numerical source register"
    );
    let size = Size::of(format == Format::Double);
    let mut asm = code.asm();
    if format == Format::Double {
        asm.movq_from_xmm(Rax, source);
    } else {
        asm.movd_from_xmm(Rax, source);
    }
    asm.shift_imm(ShiftOp::Shr, size, Rax, format.fraction_bits() as u8);
    let exponent_mask = (1u32 << (format.bits() - format.fraction_bits() - 1)) - 1;
    let field_mask = if positive_only {
        exponent_mask * 2 + 1
    } else {
        exponent_mask
    };
    eax_imm32(code, Alu::And, field_mask);
    if minimum != 0 {
        eax_imm8(code, Alu::Sub, minimum);
    }
    eax_imm32(code, Alu::Cmp, maximum - u32::from(minimum));
}

pub(super) fn environment(code: &mut Code) {
    environment_round(code, Rounding::Fpcr);
}

/// Save the complete caller environment and install masked, nonflushing SSE
/// control. Conversion instructions may select rounding independently of FPCR.
pub(super) fn environment_round(code: &mut Code, rounding: Rounding) {
    let mut asm = code.asm();
    asm.alu_imm8(Alu::Sub, Size::Qword, Rsp, ENVIRONMENT_STACK_BYTES as i8);
    asm.stmxcsr(frame(MXCSR_SAVED_OFFSET));
    if matches!(rounding, Rounding::Fpcr) {
        asm.mov_load(
            Size::Dword,
            Rcx,
            state(core::mem::offset_of!(GuestState, fpcr)),
        );
        asm.mov(Size::Dword, Rax, Rcx);
        asm.shift_imm(ShiftOp::Shr, Size::Dword, Rax, FPCR_POSITIVE_TO_MXCSR_SHIFT);
        asm.alu_accumulator_imm32(Alu::And, Size::Dword, MXCSR_POSITIVE_ROUNDING as i32);
        asm.shift_imm(ShiftOp::Shr, Size::Dword, Rcx, FPCR_NEGATIVE_TO_MXCSR_SHIFT);
        asm.alu_imm32(Alu::And, Size::Dword, Rcx, MXCSR_NEGATIVE_ROUNDING as i32);
        asm.alu(Alu::Or, Size::Dword, Rax, Rcx);
        asm.alu_accumulator_imm32(Alu::Or, Size::Dword, MXCSR_EXCEPTION_MASKS as i32);
    } else {
        let direction = match rounding {
            Rounding::NearestEven => 0,
            Rounding::Positive => MXCSR_POSITIVE_ROUNDING,
            Rounding::Negative => MXCSR_NEGATIVE_ROUNDING,
            Rounding::Zero => MXCSR_POSITIVE_ROUNDING | MXCSR_NEGATIVE_ROUNDING,
            Rounding::Fpcr => unreachable!("FPCR control emitted above"),
        };
        asm.mov_imm32(Rax, MXCSR_EXCEPTION_MASKS | direction);
    }
    asm.mov_store(Size::Dword, frame(MXCSR_CONTROL_OFFSET), Rax);
    asm.ldmxcsr(frame(MXCSR_CONTROL_OFFSET));
}

pub(super) fn restore_environment(code: &mut Code) {
    let mut asm = code.asm();
    asm.stmxcsr(frame(MXCSR_STATUS_OFFSET));
    asm.ldmxcsr(frame(MXCSR_SAVED_OFFSET));
    asm.mov_load(Size::Dword, Rcx, frame(MXCSR_STATUS_OFFSET));
    asm.alu_imm8(Alu::Add, Size::Qword, Rsp, ENVIRONMENT_STACK_BYTES as i8);
}

/// Observe a real arithmetic stage without leaving the private environment.
pub(super) fn capture_environment_status(code: &mut Code) {
    let mut asm = code.asm();
    asm.stmxcsr(frame(MXCSR_STATUS_OFFSET));
    asm.mov_load(Size::Dword, Rcx, frame(MXCSR_STATUS_OFFSET));
}

/// Discard private scratch-operation status before the next arithmetic stage.
pub(super) fn clear_environment_status(code: &mut Code) {
    code.asm().ldmxcsr(frame(MXCSR_CONTROL_OFFSET));
}

/// MXCSR underflow status: with underflow masked, raised for an inexact
/// result that was tiny before rounding, as Arm's FPSR.UFC with FPCR.AH
/// zero.
pub(super) const MXCSR_UNDERFLOW: u32 = 1 << 4;
/// FPSR.UFC.
pub(super) const FPSR_UNDERFLOW: u8 = 1 << 3;

/// Branch taken when FPSR has every flag of `flags`.
pub(super) fn fpsr_has(code: &mut Code, flags: u8) -> usize {
    let mut asm = code.asm();
    asm.movzx(
        Size::Dword,
        Rax,
        Size::Byte,
        state(core::mem::offset_of!(GuestState, fpsr)),
    );
    asm.alu_imm8(Alu::And, Size::Dword, Rax, flags as i8);
    asm.alu_imm8(Alu::Cmp, Size::Dword, Rax, flags as i8);
    branch(code, Condition::Zero)
}

/// Branch taken when some lane of the S/D `source` (`shape`) is a NaN or
/// an infinity. Clobbers EAX and `scratch`.
pub(super) fn nonfinite_lane(
    code: &mut Code,
    format: Format,
    shape: Shape,
    source: u8,
    scratch: u8,
) -> usize {
    if shape == Shape::Scalar {
        let size = Size::of(format == Format::Double);
        let mut asm = code.asm();
        if format == Format::Double {
            asm.movq_from_xmm(Rax, source);
        } else {
            asm.movd_from_xmm(Rax, source);
        }
        // Exponent all ones: the shifted-out sign leaves the exponent on top,
        // and adding the lowest exponent bit carries out of all ones.
        asm.alu(Alu::Add, size, Rax, Rax);
        asm.shift_imm(ShiftOp::Shr, size, Rax, format.fraction_bits() as u8 + 1);
        asm.alu_imm32(
            Alu::Cmp,
            size,
            Rax,
            (format.exponent_mask() >> format.fraction_bits()) as i32,
        );
        return branch(code, Condition::Zero);
    }
    // The high dword of a Double lane holds its exponent: test those.
    packed(code, Host::MoveAligned, scratch, source);
    ShiftOpcode::LeftDword.emit(code, scratch, SIGN_BITS);
    let exponent_shift = match format {
        Format::Double => (Format::Double.fraction_bits() - 32) as u8 + 1,
        _ => format.fraction_bits() as u8 + 1,
    };
    ShiftOpcode::RightDword.emit(code, scratch, exponent_shift);
    let all_ones = match format {
        Format::Double => 0x7ff,
        _ => 0xff,
    };
    let mut asm = code.asm();
    asm.mov_imm32(Rax, all_ones);
    asm.movd_to_xmm(ZERO_VECTOR, Rax);
    packed_imm(code, Host::ShuffleDword, ZERO_VECTOR, ZERO_VECTOR, 0);
    packed(code, Host::EqualDword, scratch, ZERO_VECTOR);
    Host::DwordSignMask.register(code, 0, scratch, crate::TargetCpu::Baseline);
    let mask = match (shape, format) {
        (Shape::Vector64, Format::Single) => SINGLE_LOW_VECTOR_MASK,
        (Shape::Vector128, Format::Single) => SINGLE_VECTOR_MASK,
        (Shape::Vector128, Format::Double) => DOUBLE_HIGH_DWORDS,
        _ => unreachable!("allocated S/D vector shape"),
    };
    eax_imm8(code, Alu::And, mask);
    branch(code, Condition::NotZero)
}

/// Accumulate IXC and UFC from ECX's MXCSR status (`commit_inexact` with
/// underflow), leaving ECX with only those fresh FPSR flags.
pub(super) fn commit_inexact_underflow(code: &mut Code) {
    let mut asm = code.asm();
    asm.alu_imm32(
        Alu::And,
        Size::Dword,
        Rcx,
        (MXCSR_INEXACT | MXCSR_UNDERFLOW) as i32,
    );
    asm.shift_imm(ShiftOp::Shr, Size::Dword, Rcx, MXCSR_TO_FPSR_INEXACT_SHIFT);
    asm.alu(
        Alu::Or,
        Size::Dword,
        state(core::mem::offset_of!(GuestState, fpsr)),
        Rcx,
    );
}

/// The status a guest-visible result rejects ([`reject_other_status`])
/// except underflow.
pub(super) fn reject_other_status_but_underflow(code: &mut Code) -> usize {
    code.asm().test_imm(
        Size::Dword,
        Rcx,
        (MXCSR_OTHER_STATUS & !MXCSR_UNDERFLOW) as i32,
    );
    branch(code, Condition::NotZero)
}

/// Accumulate status and leave ECX holding only fresh guest IXC (zero or INEXACT).
pub(super) fn commit_inexact(code: &mut Code) {
    let mut asm = code.asm();
    asm.alu_imm32(Alu::And, Size::Dword, Rcx, MXCSR_INEXACT as i32);
    asm.shift_imm(ShiftOp::Shr, Size::Dword, Rcx, MXCSR_TO_FPSR_INEXACT_SHIFT);
    asm.alu(
        Alu::Or,
        Size::Dword,
        state(core::mem::offset_of!(GuestState, fpsr)),
        Rcx,
    );
}

/// Extract exponent magnitudes into words/dwords and compare an open range.
/// Double exponents occupy the low dword of each qword; their high dwords and
/// inactive S lanes are excluded from the dword sign mask. No numerical FP runs here.
pub(in crate::x86_64) fn vector_exponent_range(
    code: &mut Code,
    format: Format,
    shape: Shape,
    source: u8,
) {
    vector_exponent_range_impl(code, format, shape, source, false);
}

/// Reject negative inputs as well as zero and all-ones exponents.
pub(super) fn vector_positive_exponent_range(
    code: &mut Code,
    format: Format,
    shape: Shape,
    source: u8,
) {
    vector_exponent_range_impl(code, format, shape, source, true);
}

/// Copy `source` to EXPONENT_VECTOR and reduce every lane to its exponent
/// field, with the sign above it if `positive_only`.
fn vector_exponent_fields(code: &mut Code, format: Format, source: u8, positive_only: bool) {
    packed(code, Host::MoveAligned, EXPONENT_VECTOR, source);
    let (left, right) = match format {
        Format::Single => (ShiftOpcode::LeftDword, ShiftOpcode::RightDword),
        Format::Double => (ShiftOpcode::LeftQword, ShiftOpcode::RightQword),
        Format::Half => (ShiftOpcode::LeftWord, ShiftOpcode::RightWord),
    };
    if positive_only {
        right.emit(code, EXPONENT_VECTOR, format.fraction_bits() as u8);
    } else {
        left.emit(code, EXPONENT_VECTOR, SIGN_BITS);
        right.emit(
            code,
            EXPONENT_VECTOR,
            format.fraction_bits() as u8 + SIGN_BITS,
        );
    }
}

fn vector_exponent_validity(code: &mut Code, format: Format, source: u8, positive_only: bool) {
    vector_exponent_fields(code, format, source, positive_only);
    packed(
        code,
        Host::MoveAligned,
        VALIDITY_VECTOR,
        MAXIMUM_EXPONENT_VECTOR,
    );
    let compare = if format == Format::Half {
        Host::GreaterWord
    } else {
        Host::GreaterDword
    };
    packed(code, compare, VALIDITY_VECTOR, EXPONENT_VECTOR);
    packed(code, compare, EXPONENT_VECTOR, MINIMUM_EXPONENT_VECTOR);
    packed(code, Host::And, EXPONENT_VECTOR, VALIDITY_VECTOR);
}

fn half_validity_mask(code: &mut Code, byte_mask: u32) {
    Host::ByteSignMask.register(code, 0, EXPONENT_VECTOR, crate::TargetCpu::Baseline);
    eax_imm32(code, Alu::And, byte_mask);
    eax_imm32(code, Alu::Cmp, byte_mask);
}

/// Classify a caller-selected low Half lane span without inventing a vector shape.
pub(in crate::x86_64) fn vector_half_exponent_range(code: &mut Code, source: u8, byte_mask: u32) {
    vector_exponent_validity(code, Format::Half, source, false);
    half_validity_mask(code, byte_mask);
}

fn vector_exponent_range_impl(
    code: &mut Code,
    format: Format,
    shape: Shape,
    source: u8,
    positive_only: bool,
) {
    vector_exponent_validity(code, format, source, positive_only);
    vector_validity_mask(code, format, shape);
}

/// Accept only normal values and signed zeros. The range constants
/// remain live in XMM4/5; XMM3 and `scratch` are classifier scratch. Unlike a zero
/// fraction test, clearing just the sign rejects infinities and subnormals.
pub(super) fn vector_normal_or_zero_range(
    code: &mut Code,
    format: Format,
    shape: Shape,
    source: u8,
    scratch: u8,
) {
    vector_normal_or_zero_fields(code, format, source, scratch);
    vector_validity_mask(code, format, shape);
}

/// Accept normal values, infinities and, for S/D, signed zeros: every
/// ordered value except subnormals (and Half zeros). NaN lanes fail. Same
/// register contract as [`vector_normal_or_zero_range`].
pub(super) fn vector_ordered_range(
    code: &mut Code,
    format: Format,
    shape: Shape,
    source: u8,
    scratch: u8,
) {
    if format == Format::Half {
        vector_exponent_validity(code, format, source, false);
    } else {
        vector_normal_or_zero_fields(code, format, source, scratch);
    }
    infinity_lanes(code, format, source, scratch);
    vector_validity_mask(code, format, shape);
}

/// OR an all-ones mask for every infinite lane of `source` into
/// EXPONENT_VECTOR. Shifting out the sign leaves an infinity equal to all
/// ones shifted above the fraction; a NaN has fraction bits and differs.
/// Clobbers VALIDITY_VECTOR and `scratch`.
fn infinity_lanes(code: &mut Code, format: Format, source: u8, scratch: u8) {
    let (left, equal) = match format {
        Format::Double => (ShiftOpcode::LeftQword, Host::EqualDword),
        Format::Single => (ShiftOpcode::LeftDword, Host::EqualDword),
        Format::Half => (ShiftOpcode::LeftWord, Host::EqualWord),
    };
    packed(code, Host::MoveAligned, VALIDITY_VECTOR, source);
    left.emit(code, VALIDITY_VECTOR, SIGN_BITS);
    packed(code, Host::EqualDword, scratch, scratch);
    left.emit(code, scratch, format.fraction_bits() as u8 + SIGN_BITS);
    packed(code, equal, VALIDITY_VECTOR, scratch);
    if format == Format::Double {
        packed_imm(
            code,
            Host::ShuffleDword,
            scratch,
            VALIDITY_VECTOR,
            SWAP_DWORD_PAIRS,
        );
        packed(code, Host::And, VALIDITY_VECTOR, scratch);
    }
    packed(code, Host::Or, EXPONENT_VECTOR, VALIDITY_VECTOR);
}

/// Leave ZF set when the low `format` lane of `source` is an infinity of
/// either sign. Clobbers RAX.
pub(super) fn infinity_scalar(code: &mut Code, format: Format, source: u8) {
    let mut asm = code.asm();
    if format == Format::Double {
        asm.movq_from_xmm(Rax, source);
        // Drop the sign, then rotate the exponent below the fraction.
        asm.shift_imm(ShiftOp::Shl, Size::Qword, Rax, SIGN_BITS);
        asm.shift_imm(
            ShiftOp::Rol,
            Size::Qword,
            Rax,
            (Format::Double.bits() - Format::Double.fraction_bits() - u32::from(SIGN_BITS)) as u8,
        );
        let exponent = (format.exponent_mask() >> format.fraction_bits()) as i32;
        asm.alu_imm32(Alu::Cmp, Size::Qword, Rax, exponent);
    } else {
        asm.movd_from_xmm(Rax, source);
        // Shift out the higher lanes and the sign.
        let shift = u32::BITS - format.bits() + u32::from(SIGN_BITS);
        asm.shift_imm(ShiftOp::Shl, Size::Dword, Rax, shift as u8);
        asm.alu_accumulator_imm32(
            Alu::Cmp,
            Size::Dword,
            (format.exponent_mask() << shift) as u32 as i32,
        );
    }
}

fn vector_normal_or_zero_fields(code: &mut Code, format: Format, source: u8, scratch: u8) {
    assert!(
        ![
            source,
            EXPONENT_VECTOR,
            VALIDITY_VECTOR,
            MAXIMUM_EXPONENT_VECTOR,
            MINIMUM_EXPONENT_VECTOR
        ]
        .contains(&scratch)
    );
    vector_exponent_validity(code, format, source, false);
    packed(code, Host::MoveAligned, VALIDITY_VECTOR, source);
    let (left, equal) = match format {
        Format::Double => (ShiftOpcode::LeftQword, Host::EqualDword),
        Format::Single => (ShiftOpcode::LeftDword, Host::EqualDword),
        Format::Half => (ShiftOpcode::LeftWord, Host::EqualWord),
    };
    left.emit(code, VALIDITY_VECTOR, SIGN_BITS);
    // The result-range lower bound is one rather than zero. Use an independent
    // zero so output classification still recognizes both signed zeros.
    packed(code, Host::Xor, scratch, scratch);
    packed(code, equal, VALIDITY_VECTOR, scratch);
    if format == Format::Double {
        packed_imm(
            code,
            Host::ShuffleDword,
            scratch,
            VALIDITY_VECTOR,
            SWAP_DWORD_PAIRS,
        );
        packed(code, Host::And, VALIDITY_VECTOR, scratch);
    }
    packed(code, Host::Or, EXPONENT_VECTOR, VALIDITY_VECTOR);
}

fn vector_validity_mask(code: &mut Code, format: Format, shape: Shape) {
    if format == Format::Half {
        let mask = match shape {
            Shape::Scalar => HALF_SCALAR_BYTE_MASK,
            Shape::Vector64 => HALF_LOW_BYTE_MASK,
            Shape::Vector128 => HALF_VECTOR_BYTE_MASK,
        };
        half_validity_mask(code, mask);
        return;
    }
    Host::DwordSignMask.register(code, 0, EXPONENT_VECTOR, crate::TargetCpu::Baseline);
    let mask = match (shape, format) {
        (Shape::Scalar, _) => {
            eax_imm8(code, Alu::And, SCALAR_VECTOR_MASK);
            SCALAR_VECTOR_MASK
        }
        (Shape::Vector64, Format::Single) => {
            eax_imm8(code, Alu::And, SINGLE_LOW_VECTOR_MASK);
            SINGLE_LOW_VECTOR_MASK
        }
        (Shape::Vector128, Format::Single) => SINGLE_VECTOR_MASK,
        (Shape::Vector128, Format::Double) => {
            // Normal exponent validity occupies the low dword; exact-zero
            // validity fills both. Reduce one bit per architectural D lane.
            eax_imm8(code, Alu::And, DOUBLE_VECTOR_MASK);
            DOUBLE_VECTOR_MASK
        }
        _ => unreachable!("allocated vector shape"),
    };
    eax_imm8(code, Alu::Cmp, mask);
}

/// Widen four staged Half fields with integer operations. Callers guard active
/// inputs and supply zero and the broadcast Single-minus-Half exponent bias.
/// Inactive raw fields also become finite normal Single values without status.
pub(in crate::x86_64) fn widen_half_to_single(
    code: &mut Code,
    destination: u8,
    source: u8,
    sign: u8,
    zero: u8,
    bias: u8,
) {
    const WIDTH_DIFFERENCE: u8 = (Format::Single.bits() - Format::Half.bits()) as u8;
    const SINGLE_SIGN_POSITION: u8 = (Format::Single.bits() - 1) as u8;
    const FRACTION_DIFFERENCE: u8 =
        (Format::Single.fraction_bits() - Format::Half.fraction_bits()) as u8;
    if destination != source {
        packed(code, Host::MoveAligned, destination, source);
    }
    packed(code, Host::UnpackLowWord, destination, zero);
    packed(code, Host::MoveAligned, sign, destination);
    ShiftOpcode::LeftDword.emit(code, sign, WIDTH_DIFFERENCE);
    ShiftOpcode::RightDword.emit(code, sign, SINGLE_SIGN_POSITION);
    ShiftOpcode::LeftDword.emit(code, sign, SINGLE_SIGN_POSITION);
    ShiftOpcode::LeftDword.emit(code, destination, WIDTH_DIFFERENCE + SIGN_BITS);
    ShiftOpcode::RightDword.emit(
        code,
        destination,
        WIDTH_DIFFERENCE + SIGN_BITS - FRACTION_DIFFERENCE,
    );
    packed(code, Host::AddDword, destination, bias);
    packed(code, Host::Or, destination, sign);
}

pub(super) fn read_numerical_vector(code: &mut Code, shape: Shape, source: u8, xmm: u8) {
    if shape == Shape::Vector64 {
        // MOVQ loads only active bits and clears the high qword, even for DIV.
        Host::MoveQword.emit(
            code,
            xmm,
            state(vector_offset(source) as usize),
            crate::TargetCpu::Baseline,
        );
    } else {
        read_vector(code, source, xmm);
    }
}

/// The low S or D element of guest vector `source` in `xmm`, the rest of
/// it clear (MOVSS or MOVSD from memory), so that a scalar operation with
/// `xmm` as its destination leaves the upper bits an Arm scalar write
/// clears already clear.
pub(super) fn read_numerical_scalar(code: &mut Code, format: Format, source: u8, xmm: u8) {
    match format {
        Format::Single => Host::MoveScalarSingle,
        Format::Double => Host::MoveScalarDouble,
        Format::Half => unreachable!("Half has its own mapping policy"),
    }
    .emit(
        code,
        xmm,
        state(vector_offset(source) as usize),
        crate::TargetCpu::Baseline,
    );
}

/// How a sticky path reads one input from the frame: a zeroing scalar
/// load ([`read_numerical_scalar`]) or a vector one
/// ([`read_numerical_vector`]).
#[derive(Clone, Copy)]
pub(super) enum Input {
    Scalar(Format),
    Vector(Shape),
}

/// INSERTPS selector copying source lane 0 to destination lane 0 and
/// clearing lanes 1 to 3.
const INSERT_LOW_LANE_ALONE: u8 = 0b1110;

/// Read each `(input, source vector, xmm)`, taking a source that XMM0
/// holds whole (`xmm0`) from there rather than the frame, and producing
/// the bits a frame read would. Copies out of XMM0 come first, before a
/// read into XMM0 replaces it. A Single scalar copy needs SSE4.1's
/// INSERTPS; without it the frame provides the input.
pub(super) fn read_inputs(
    code: &mut Code,
    xmm0: Option<u8>,
    target: crate::TargetCpu,
    inputs: &[(Input, u8, u8)],
) {
    let held = |input: Input, source: u8| {
        xmm0 == Some(source)
            && !matches!(input, Input::Scalar(Format::Single) if !target.supports(crate::TargetCpu::Sse41))
    };
    let copy = |code: &mut Code, input: Input, xmm: u8| match input {
        Input::Scalar(Format::Single) => {
            Host::InsertSingle.emit_imm(
                code,
                xmm,
                Rm::Xmm(LEFT_VECTOR),
                INSERT_LOW_LANE_ALONE,
                target,
            );
        }
        Input::Scalar(Format::Double) | Input::Vector(Shape::Vector64) => {
            packed(code, Host::MoveQword, xmm, LEFT_VECTOR);
        }
        Input::Scalar(Format::Half) => unreachable!("Half has its own mapping policy"),
        Input::Vector(Shape::Vector128 | Shape::Scalar) => {
            if xmm != LEFT_VECTOR {
                packed(code, Host::MoveAligned, xmm, LEFT_VECTOR);
            }
        }
    };
    for &(input, source, xmm) in inputs {
        if held(input, source) && xmm != LEFT_VECTOR {
            copy(code, input, xmm);
        }
    }
    for &(input, source, xmm) in inputs {
        if !held(input, source) {
            match input {
                Input::Scalar(format) => read_numerical_scalar(code, format, source, xmm),
                Input::Vector(shape) => read_numerical_vector(code, shape, source, xmm),
            }
        } else if xmm == LEFT_VECTOR {
            copy(code, input, xmm);
        }
    }
}

/// Select from the whole indexed source before broadcasting. A 64-bit indexed
/// form may read a lane in the upper qword.
pub(super) fn read_numerical_right(
    code: &mut Code,
    format: Format,
    shape: Shape,
    source: u8,
    right_lane: Option<u8>,
) {
    const BROADCAST_LOW_LANE: u8 = 0;
    if let Some(lane) = right_lane {
        read_vector(code, source, RIGHT_VECTOR);
        ShiftOpcode::RightBytes.emit(
            code,
            RIGHT_VECTOR,
            (u32::from(lane) * format.bits() / u8::BITS) as u8,
        );
        if shape != Shape::Scalar {
            if format == Format::Half {
                packed_imm(
                    code,
                    Host::ShuffleLowWord,
                    RIGHT_VECTOR,
                    RIGHT_VECTOR,
                    BROADCAST_LOW_LANE,
                );
            }
            if format != Format::Double {
                packed_imm(
                    code,
                    Host::ShuffleDword,
                    RIGHT_VECTOR,
                    RIGHT_VECTOR,
                    BROADCAST_LOW_LANE,
                );
            } else {
                packed(code, Host::UnpackLowQword, RIGHT_VECTOR, RIGHT_VECTOR);
            }
        }
    } else {
        read_numerical_vector(code, shape, source, RIGHT_VECTOR);
    }
}

pub(in crate::x86_64) fn prepare_vector_input_range(code: &mut Code, format: Format) {
    prepare_vector_maximum(code, format);
    packed(
        code,
        Host::Xor,
        MINIMUM_EXPONENT_VECTOR,
        MINIMUM_EXPONENT_VECTOR,
    );
}

/// The all-ones exponent of `format` in every lane of
/// MAXIMUM_EXPONENT_VECTOR.
fn prepare_vector_maximum(code: &mut Code, format: Format) {
    packed(
        code,
        Host::EqualDword,
        MAXIMUM_EXPONENT_VECTOR,
        MAXIMUM_EXPONENT_VECTOR,
    );
    let exponent_bits = format.bits() - format.fraction_bits() - u32::from(SIGN_BITS);
    let (shift, width) = if format == Format::Half {
        (ShiftOpcode::RightWord, u16::BITS)
    } else {
        (ShiftOpcode::RightDword, u32::BITS)
    };
    shift.emit(code, MAXIMUM_EXPONENT_VECTOR, (width - exponent_bits) as u8);
}
/// Set an exclusive upper bound, keeping the normal lower bound used by
/// `vector_exponent_range`. Half precision repeats the limit in both words.
pub(super) fn prepare_vector_input_range_below(code: &mut Code, format: Format, maximum: u32) {
    const BROADCAST_LOW_DWORD: u8 = 0;
    let limit = if format == Format::Half {
        maximum | maximum << u16::BITS
    } else {
        maximum
    };
    let mut asm = code.asm();
    asm.mov_imm32(Rax, limit);
    asm.movd_to_xmm(MAXIMUM_EXPONENT_VECTOR, Rax);
    packed_imm(
        code,
        Host::ShuffleDword,
        MAXIMUM_EXPONENT_VECTOR,
        MAXIMUM_EXPONENT_VECTOR,
        BROADCAST_LOW_DWORD,
    );
    packed(
        code,
        Host::Xor,
        MINIMUM_EXPONENT_VECTOR,
        MINIMUM_EXPONENT_VECTOR,
    );
}

/// [`prepare_vector_result_range`] for Half's word lanes.
pub(in crate::x86_64) fn prepare_half_result_range(code: &mut Code) {
    packed(
        code,
        Host::EqualDword,
        MINIMUM_EXPONENT_VECTOR,
        MINIMUM_EXPONENT_VECTOR,
    );
    ShiftOpcode::RightWord.emit(code, MINIMUM_EXPONENT_VECTOR, u16::BITS as u8 - SIGN_BITS);
}

pub(in crate::x86_64) fn prepare_vector_result_range(code: &mut Code) {
    packed(
        code,
        Host::EqualDword,
        MINIMUM_EXPONENT_VECTOR,
        MINIMUM_EXPONENT_VECTOR,
    );
    ShiftOpcode::RightDword.emit(code, MINIMUM_EXPONENT_VECTOR, u32::BITS as u8 - SIGN_BITS);
}
pub(super) fn emit_opcode(
    code: &mut Code,
    format: Format,
    shape: Shape,
    opcode: ArithmeticOpcode,
    right: u8,
) {
    emit_opcode_to(code, format, shape, opcode, LEFT_VECTOR, right);
}

/// Arithmetic encoding with an explicit destination register.
pub(in crate::x86_64) fn emit_opcode_to(
    code: &mut Code,
    format: Format,
    shape: Shape,
    opcode: ArithmeticOpcode,
    destination: u8,
    source: u8,
) {
    code.asm().float(
        opcode.float_op(),
        float_lanes(format, shape),
        destination,
        Rm::Xmm(source),
    );
}
pub(super) fn reject_other_status(code: &mut Code) -> usize {
    code.asm()
        .test_imm(Size::Dword, Rcx, MXCSR_OTHER_STATUS as i32);
    branch(code, Condition::NotZero)
}
pub(super) fn save_product_status(code: &mut Code) {
    code.asm().stmxcsr(frame(MXCSR_PRODUCT_OFFSET));
}
pub(super) fn restore_environment_with_product(code: &mut Code) {
    let mut asm = code.asm();
    asm.stmxcsr(frame(MXCSR_STATUS_OFFSET));
    asm.ldmxcsr(frame(MXCSR_SAVED_OFFSET));
    asm.mov_load(Size::Dword, Rax, frame(MXCSR_PRODUCT_OFFSET));
    asm.mov_load(Size::Dword, Rcx, frame(MXCSR_STATUS_OFFSET));
    asm.alu_imm8(Alu::Add, Size::Qword, Rsp, ENVIRONMENT_STACK_BYTES as i8);
}
pub(super) fn reject_product_status(code: &mut Code) -> usize {
    code.asm()
        .test_accumulator_imm(Size::Byte, MXCSR_STATUS_FLAGS.into());
    branch(code, Condition::NotZero)
}
/// `movq xmm, xmm`: clear the high qword of `xmm`.
pub(super) fn clear_high_qword(code: &mut Code, xmm: u8) {
    Host::MoveQword.register(code, xmm, xmm, crate::TargetCpu::Baseline);
}
pub(super) fn clear_scalar(code: &mut Code, format: Format, xmm: u8) {
    if format == Format::Double {
        Host::MoveQword.register(code, xmm, xmm, crate::TargetCpu::Baseline);
    } else {
        code.asm().movd_from_xmm(Rax, xmm);
        if format == Format::Half {
            eax_imm32(code, Alu::And, u32::from(u16::MAX));
        }
        code.asm().movd_to_xmm(xmm, Rax);
    }
}
