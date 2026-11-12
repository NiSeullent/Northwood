//! AdvSIMD and FP emission with SSE2.
use super::isa::SimdOpcode;
use super::isa::SimdOpcode as Host;
use super::{MemoryChecks, memory_address, memory_writeback};
use crate::aarch64::simd::{Comparison, IntegerOperation, Operand};
use crate::x86_64::asm::{Code, Gpr, Mem, vector_extensions};
pub(in crate::x86_64) mod absolute_selection;
pub(super) mod aes;
pub(super) mod arithmetic;
pub(in crate::x86_64) mod bfloat16;
mod comparison;
mod complex;
#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux")
))]
mod conversion_costs;
pub(in crate::x86_64) mod encode;
mod estimate;
pub(in crate::x86_64) mod exponent_scale;
mod extrema_scale;
#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux")
))]
mod fast_conversion_tests;
#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux")
))]
mod fast_float_tests;
mod floating;
mod fused;
pub(in crate::x86_64) mod half;
mod half_integer;
#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux")
))]
mod half_integer_tests;
mod halving;
mod i8mm;
mod immediates;
mod integer_conversion;
mod lut;
pub(super) mod memory;
pub(crate) mod memory_checked;
mod minmax;
pub(in crate::x86_64) mod numerical;
pub(in crate::x86_64) mod precision;
#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux")
))]
mod product_nan_tests;
mod products;
pub(super) mod rearrange;
mod reductions;
mod refinement;
mod rounding;
pub(super) use rounding::kernel as rounding_kernel;
#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux")
))]
mod recipe_tests;
mod saturating;
mod sha1;
pub(super) mod sha256;
mod sha3;
mod sha32;
mod sha512;
pub(crate) mod shifts;
mod sm3;
mod sm4;
mod sqrt;
mod structure_checked;
mod structure_memory;
mod three;
mod widening;

/// Fixed-width SIMD memory routing; checked bodies never use private-RAM emitters.
#[derive(Clone, Copy)]
pub(crate) enum CheckedMemoryOperand {
    Fixed(memory_checked::CheckedOperand),
    Structure(structure_checked::CheckedOperand),
}

impl CheckedMemoryOperand {
    pub(crate) fn from_operation(
        operation: &crate::aarch64::Operation,
        profile: crate::block::CheckedMemoryProfile,
    ) -> Option<Self> {
        memory_checked::CheckedOperand::from_operation(operation, profile)
            .map(Self::Fixed)
            .or_else(|| {
                structure_checked::CheckedOperand::from_operation(operation, profile)
                    .map(Self::Structure)
            })
    }

    pub(crate) fn requirements(&self) -> crate::context::memory::MemoryRequirements {
        match self {
            Self::Fixed(operand) => operand.requirements(),
            Self::Structure(operand) => operand.requirements(),
        }
    }

    pub(crate) fn emit_context(&self, code: &mut Code, pc: u64, target: crate::TargetCpu) {
        match self {
            Self::Fixed(operand) => operand.emit_context(code, pc),
            Self::Structure(operand) => operand.emit_context(code, pc, target),
        }
    }
}

fn vector_offset(vector: u8) -> u32 {
    (core::mem::offset_of!(crate::GuestState, v) + usize::from(vector) * 16) as u32
}

/// Guest vector `vector` in GuestState, `[rdi+disp32]`.
fn vector_memory(vector: u8) -> Mem {
    Mem::base_disp32(Gpr::Rdi, vector_offset(vector) as i32)
}

/// Write guest vector `vector` from `xmm`: to its home where the block
/// keeps vectors in registers ([`super::vector_cache`]), else to the state.
fn commit(code: &mut Code, vector: u8, xmm: u8) {
    if code.vector_cache().writes_through() {
        code.discard_vector(vector);
    } else if code.vector_cache().active() {
        let host = code.allocate_vector(vector);
        if host != xmm {
            code.asm_unguarded().sse(
                Some(0x66),
                &[0x0f, 0x6f],
                host,
                crate::x86_64::asm::Rm::Xmm(xmm),
            );
        }
        code.mark_vector_dirty(vector);
        return;
    }
    // GuestState has only 8-byte alignment; V0 begins at byte264.
    Host::StoreUnaligned.emit(code, xmm, vector_memory(vector), crate::TargetCpu::Baseline);
}

/// Copy guest vector `vector` to `xmm`, which the caller may then change:
/// from its home, if it has one, else from the state.
fn read_vector(code: &mut Code, vector: u8, xmm: u8) {
    if let Some(host) = code.vector_home(vector) {
        packed(code, Host::MoveAligned, xmm, host);
        return;
    }
    Host::LoadUnaligned.emit(code, xmm, vector_memory(vector), crate::TargetCpu::Baseline);
}

/// A register that holds guest vector `vector` for reading only: its
/// home, which it gets here where the block keeps vectors in registers
/// ([`super::vector_cache`]), else `scratch` loaded from the state. The
/// caller never writes the register returned.
fn vector_source(code: &mut Code, vector: u8, scratch: u8) -> u8 {
    if let Some(host) = code.vector_home(vector) {
        return host;
    }
    if code.vector_cache().active() {
        let host = code.allocate_vector(vector);
        code.asm_unguarded()
            .sse(Some(0xf3), &[0x0f, 0x6f], host, vector_memory(vector));
        return host;
    }
    read_vector(code, vector, scratch);
    scratch
}

/// Where a recipe writes guest vector `vector`'s new value
/// ([`vector_destination`], [`vector_accumulator`]): `xmm`, the vector's
/// home, which the recipe's last instruction writes whole, or the recipe's
/// scratch register, which [`finish`] commits.
#[derive(Debug, Clone, Copy)]
struct Destination {
    vector: u8,
    xmm: u8,
    /// The scratch register the caller offered, distinct from every
    /// source register it still reads unless equal to `xmm`.
    scratch: u8,
}

impl Destination {
    /// Whether `xmm` is the vector's home.
    const fn home(&self) -> bool {
        self.xmm >= super::vector_cache::FIRST_HOST
    }
}

/// The register a recipe writes guest vector `vector`'s whole new value to
/// with its last instruction, after reading every source: its home where
/// the block keeps vectors in registers (and commits do not go to the
/// state), else `scratch`. Every source is taken before this call: the
/// home's old value is gone, and [`super::vector_cache`] refuses to hand
/// it out again before [`finish`].
fn vector_destination(code: &mut Code, vector: u8, scratch: u8) -> Destination {
    let cache = code.vector_cache();
    let xmm = if cache.active() && !cache.writes_through() {
        code.begin_vector_write(vector, true)
    } else {
        scratch
    };
    Destination {
        vector,
        xmm,
        scratch,
    }
}

/// [`vector_destination`] holding the vector's current value, for a recipe
/// that reads and then writes it in place (an accumulator, an inserted
/// half): its home, loaded if new, else `scratch` loaded.
fn vector_accumulator(code: &mut Code, vector: u8, scratch: u8) -> Destination {
    let cache = code.vector_cache();
    let xmm = if cache.active() && !cache.writes_through() {
        let loaded = code.vector_cache().peek(vector).is_some();
        let host = code.begin_vector_write(vector, false);
        if !loaded {
            code.asm_unguarded()
                .sse(Some(0xf3), &[0x0f, 0x6f], host, vector_memory(vector));
        }
        host
    } else {
        read_vector(code, vector, scratch);
        scratch
    };
    Destination {
        vector,
        xmm,
        scratch,
    }
}

/// End the write of `destination`: its home is newer than the state, or
/// the scratch register is committed.
fn finish(code: &mut Code, destination: Destination) {
    if destination.home() {
        code.end_vector_write(destination.vector);
    } else {
        commit(code, destination.vector, destination.xmm);
    }
}

/// `destination` = `result`, all 16 bytes (`bytes` 16) or the low 8 with
/// the high cleared (`bytes` 8), then [`finish`].
fn finish_from(code: &mut Code, destination: Destination, result: u8, bytes: u8) {
    if bytes == 8 {
        packed(code, Host::MoveQword, destination.xmm, result);
    } else {
        debug_assert_eq!(bytes, 16);
        if destination.xmm != result {
            packed(code, Host::MoveAligned, destination.xmm, result);
        }
    }
    finish(code, destination);
}

/// `destination` = `left op right` (16 bytes), the last instruction: one
/// VEX form, or a copy and the SSE form, swapping the sources of a
/// `commutative` operation that would overwrite its right source, else
/// through the destination's scratch register.
fn binary(
    code: &mut Code,
    opcode: SimdOpcode,
    destination: Destination,
    left: u8,
    right: u8,
    commutative: bool,
    target: crate::TargetCpu,
) {
    let xmm = destination.xmm;
    if xmm == left {
        // The legacy form in place.
        opcode.register(code, xmm, right, target);
    } else if three::vex(target) || xmm != right {
        three::three(code, opcode, xmm, left, right, target);
    } else if commutative {
        three::three(code, opcode, xmm, right, left, target);
    } else {
        three::three(code, opcode, destination.scratch, left, right, target);
        packed(code, Host::MoveAligned, xmm, destination.scratch);
    }
    finish(code, destination);
}

/// Guest vector `vector` as a source operand of an instruction that may
/// take memory without an alignment requirement (VEX, EVEX): its home, if
/// it has one, else its slot in the state.
fn vector_operand(code: &mut Code, vector: u8) -> crate::x86_64::asm::Rm {
    match code.vector_home(vector) {
        Some(host) => crate::x86_64::asm::Rm::Xmm(host),
        None => vector_memory(vector).into(),
    }
}

fn packed(code: &mut Code, opcode: SimdOpcode, destination: u8, source: u8) {
    opcode.register(code, destination, source, crate::TargetCpu::Baseline);
}

/// [`packed`] with an 8-bit immediate.
fn packed_imm(code: &mut Code, opcode: SimdOpcode, destination: u8, source: u8, immediate: u8) {
    opcode.emit_imm(
        code,
        destination,
        crate::x86_64::asm::Rm::Xmm(source),
        immediate,
        crate::TargetCpu::Baseline,
    );
}

fn subtract_opcode(lane_bytes: u8) -> SimdOpcode {
    match lane_bytes {
        1 => Host::SubByte,
        2 => Host::SubWord,
        4 => Host::SubDword,
        8 => Host::SubQword,
        _ => unreachable!("reviewed lane size"),
    }
}

fn shuffle(code: &mut Code, destination: u8, source: u8, control: u8) {
    packed_imm(code, Host::ShuffleDword, destination, source, control);
}

/// `movabs rcx, value`, then `movq xmm, rcx`, which clears the high qword.
fn move_constant(code: &mut Code, xmm: u8, value: u64) {
    let mut asm = code.asm();
    asm.mov_imm64(Gpr::Rcx, value);
    asm.movq_to_xmm(xmm, Gpr::Rcx);
}

fn unsigned_bias(code: &mut Code, lane_bytes: u8) {
    let mask: u64 = match lane_bytes {
        1 => 0x8080808080808080,
        2 => 0x8000800080008000,
        4 | 8 => 0x8000000080000000,
        _ => unreachable!("reviewed comparison lane"),
    };
    move_constant(code, 4, mask);
    packed(code, Host::UnpackLowQword, 4, 4);
    packed(code, Host::Xor, 0, 4);
    packed(code, Host::Xor, 1, 4);
}

fn equal(code: &mut Code, lane_bytes: u8, target: crate::TargetCpu) {
    if lane_bytes == 8 && target >= crate::TargetCpu::Sse41 {
        Host::EqualQword.register(code, 0, 1, target); // pcmpeqq xmm0,xmm1
        return;
    }
    packed(
        code,
        match lane_bytes {
            1 => Host::EqualByte,
            2 => Host::EqualWord,
            4 | 8 => Host::EqualDword,
            _ => unreachable!("reviewed comparison lane"),
        },
        0,
        1,
    );
    if lane_bytes == 8 {
        shuffle(code, 2, 0, 0xb1); // adjacent dword equality masks must both hold
        packed(code, Host::And, 0, 2);
    }
}

pub(super) fn greater(code: &mut Code, lane_bytes: u8, unsigned: bool, target: crate::TargetCpu) {
    // Unsigned lanes compare directly instead of after a sign bias.
    if unsigned
        && vector_extensions::compare(
            code,
            lane_bytes,
            false,
            vector_extensions::Predicate::Greater,
            0,
            1,
            target,
        )
    {
        return;
    }
    if lane_bytes == 8 && target >= crate::TargetCpu::Sse42 {
        if unsigned {
            move_constant(code, 4, 0x8000000000000000);
            packed(code, Host::UnpackLowQword, 4, 4);
            packed(code, Host::Xor, 0, 4);
            packed(code, Host::Xor, 1, 4);
        }
        Host::GreaterQword.register(code, 0, 1, target); // pcmpgtq xmm0,xmm1
        return;
    }
    if unsigned {
        unsigned_bias(code, lane_bytes);
    }
    if lane_bytes != 8 {
        packed(
            code,
            match lane_bytes {
                1 => Host::GreaterByte,
                2 => Host::GreaterWord,
                4 => Host::GreaterDword,
                _ => unreachable!("reviewed comparison lane"),
            },
            0,
            1,
        );
        return;
    }
    // Compare high dwords first; equal high halves use unsigned low ordering.
    packed(code, Host::MoveAligned, 2, 0);
    packed(code, Host::GreaterDword, 2, 1);
    shuffle(code, 2, 2, 0xf5);
    packed(code, Host::MoveAligned, 3, 0);
    packed(code, Host::EqualDword, 3, 1);
    shuffle(code, 3, 3, 0xf5);
    if !unsigned {
        unsigned_bias(code, 8);
    }
    packed(code, Host::GreaterDword, 0, 1);
    shuffle(code, 0, 0, 0xa0);
    packed(code, Host::And, 0, 3);
    packed(code, Host::Or, 0, 2);
}

/// Whether [`emit_numerical`] completes on `target` an `operation` that
/// [`numerical_always_exits`] ends its block with.
pub(super) fn numerical_terminal_inline(
    operation: &crate::aarch64::FloatingPointOperation,
    target: crate::TargetCpu,
) -> bool {
    use crate::aarch64::{
        FloatingPointOperation,
        floating::{Format, integer::IntegerRounding},
        simd::conversion::Operand,
    };
    match operation {
        FloatingPointOperation::Binary(operand) => {
            operand.format == Format::Half
                && crate::x86_64::asm::vector_extensions::half_supported(target)
        }
        FloatingPointOperation::Conversion(Operand::Precision(operand)) => {
            precision::half_narrowing_supported(operand.mode, target)
        }
        FloatingPointOperation::Conversion(Operand::FromInteger(operand)) => {
            integer_conversion::half_supported(operand, target)
        }
        FloatingPointOperation::Conversion(Operand::ToInteger { operand, rounding }) => {
            integer_conversion::half_supported(operand, target)
                && *rounding != IntegerRounding::NearestAway
        }
        _ => false,
    }
}

pub(super) fn numerical_always_exits(operation: &crate::aarch64::FloatingPointOperation) -> bool {
    use crate::aarch64::{
        FloatingPointOperation,
        floating::{Format, integer::IntegerRounding},
        simd::conversion::Operand,
    };
    match operation {
        FloatingPointOperation::ExtremaScale(operand) => extrema_scale::always_exits(operand),
        FloatingPointOperation::Binary(operand) => arithmetic::always_exits(operand),
        FloatingPointOperation::Fused(_)
        | FloatingPointOperation::Widening(_)
        | FloatingPointOperation::BFloat16(_)
        | FloatingPointOperation::Complex(_) => false,
        FloatingPointOperation::Conversion(operand) => match operand {
            Operand::Precision(operand) => {
                operand.mode.destination_format() == Format::Half
                    && !precision::field_narrowing(operand.mode)
            }
            Operand::FromInteger(operand) => {
                operand.format == Format::Half && !half_integer::from_integer_inline(operand)
            }
            Operand::ToInteger { operand, rounding } => {
                if operand.format == Format::Half {
                    !half_integer::to_integer_inline(operand)
                } else {
                    *rounding == IntegerRounding::NearestAway
                        && !integer_conversion::nearest_away_inline(operand)
                }
            }
            Operand::JavaScript { .. } => false,
        },
        FloatingPointOperation::Comparison(_)
        | FloatingPointOperation::MinMax(_)
        | FloatingPointOperation::Rounding(_)
        | FloatingPointOperation::SquareRoot(_)
        | FloatingPointOperation::Estimate(_)
        | FloatingPointOperation::Refinement(_) => false,
    }
}

pub(super) fn emit_numerical(
    code: &mut Code,
    operation: &crate::aarch64::FloatingPointOperation,
    pc: crate::GuestAddress,
    target: crate::TargetCpu,
) {
    emit_numerical_split(code, operation, pc, target, numerical::Entry::UNKNOWN);
}

/// [`emit_numerical`] after what `entry` says, returning the
/// [`numerical::ColdSplit`] of an emission with a sticky-inexact path.
pub(super) fn emit_numerical_split(
    code: &mut Code,
    operation: &crate::aarch64::FloatingPointOperation,
    pc: crate::GuestAddress,
    target: crate::TargetCpu,
    entry: numerical::Entry,
) -> Option<numerical::ColdSplit> {
    match operation {
        crate::aarch64::FloatingPointOperation::Estimate(operand) => {
            estimate::emit(code, operand, pc, target);
        }
        crate::aarch64::FloatingPointOperation::Refinement(operand) => {
            refinement::emit(code, operand, pc, target);
        }
        crate::aarch64::FloatingPointOperation::Binary(operand) => {
            return arithmetic::emit(code, operand, pc, target, entry);
        }
        crate::aarch64::FloatingPointOperation::Fused(operand) => {
            return fused::emit(code, operand, pc, target, entry);
        }
        crate::aarch64::FloatingPointOperation::Widening(operand) => {
            widening::emit(code, operand, pc, target)
        }
        crate::aarch64::FloatingPointOperation::BFloat16(operand) => {
            bfloat16::emit(code, operand, pc, target)
        }
        crate::aarch64::FloatingPointOperation::Complex(operand) => {
            complex::emit(code, operand, pc, target)
        }
        crate::aarch64::FloatingPointOperation::Comparison(operand) => {
            comparison::emit(code, operand, pc, target)
        }
        crate::aarch64::FloatingPointOperation::ExtremaScale(operand) => {
            extrema_scale::emit(code, operand, pc, target);
        }
        crate::aarch64::FloatingPointOperation::MinMax(operand) => {
            minmax::emit(code, operand, pc, target)
        }
        crate::aarch64::FloatingPointOperation::Rounding(operand) => {
            rounding::emit(code, operand, pc, target)
        }
        crate::aarch64::FloatingPointOperation::SquareRoot(operand) => {
            return sqrt::emit(code, operand, pc, target, entry);
        }
        crate::aarch64::FloatingPointOperation::Conversion(operand) => {
            use crate::aarch64::simd::conversion::Operand;
            match operand {
                Operand::Precision(operand) => precision::emit(code, operand, pc, target),
                Operand::FromInteger(operand) => {
                    integer_conversion::from_integer(code, operand, pc, target)
                }
                Operand::ToInteger { operand, rounding } => {
                    integer_conversion::to_integer(code, operand, *rounding, pc, target)
                }
                Operand::JavaScript {
                    destination,
                    source,
                } => integer_conversion::javascript(code, *destination, *source, pc, target),
            }
        }
    }
    None
}

#[cfg(test)]
pub(super) fn emit(code: &mut Code, operand: &Operand) {
    emit_for_target(
        code,
        operand,
        crate::TargetCpu::Baseline,
        MemoryChecks::DEFAULT,
    );
}
pub(super) fn emit_for_target(
    code: &mut Code,
    operand: &Operand,
    target: crate::TargetCpu,
    checks: MemoryChecks,
) {
    emit_after(code, operand, target, checks, None, None);
}

/// Whether `operand` keeps the block's vector homes
/// ([`super::vector_cache`]): its recipe reaches V only through
/// [`read_vector`], [`vector_source`], [`vector_operand`] and [`commit`] and
/// names only the scratch registers XMM0 to XMM7. Accesses store the dirty
/// homes before their fault sites.
pub(super) fn vector_use(operand: &Operand, target: crate::TargetCpu) -> super::block::VectorUse {
    use super::block::VectorUse::*;
    match operand {
        Operand::Integer { .. }
        | Operand::Compare { .. }
        | Operand::MoveImmediate { .. }
        | Operand::Product(_)
        | Operand::Saturate(_)
        | Operand::Halving(_)
        | Operand::Reduce(_)
        | Operand::Immediate(crate::aarch64::simd::immediates::Operand::Constant { .. }) => Keep,
        Operand::Immediate(crate::aarch64::simd::immediates::Operand::LaneMove(lane))
        | Operand::Rearrange(lane) => rearrange::vector_use(lane, target),
        // The register forms and scalar saturating shifts stage lanes
        // through the state (`shifts::register_shift`).
        Operand::Shift(shift) => match shift.kind {
            crate::aarch64::simd::shifts::Kind::Register { .. } => Settle,
            crate::aarch64::simd::shifts::Kind::SaturatingImmediate { .. }
                if shift.bytes <= shift.lane_bytes =>
            {
                Settle
            }
            _ => Keep,
        },
        Operand::Memory { .. } | Operand::MemoryPair { .. } | Operand::MemoryRegisterOffset(_) => {
            Flush
        }
        _ => Settle,
    }
}

/// CMEQ, CMGT (by register or zero) and CMLT #0 that one PCMPEQ/PCMPGT
/// computes, from the sources where they are into the destination; returns
/// whether it emitted the comparison.
#[allow(clippy::too_many_arguments)]
fn single_compare(
    code: &mut Code,
    operation: Comparison,
    destination: u8,
    left: u8,
    right: Option<u8>,
    bytes: u8,
    lane_bytes: u8,
    target: crate::TargetCpu,
) -> bool {
    let opcode = match (operation, lane_bytes) {
        (Comparison::Equal, 1) => Host::EqualByte,
        (Comparison::Equal, 2) => Host::EqualWord,
        (Comparison::Equal, 4) => Host::EqualDword,
        (Comparison::Equal, _) => Host::EqualQword,
        (Comparison::Greater | Comparison::LessZero, 1) => Host::GreaterByte,
        (Comparison::Greater | Comparison::LessZero, 2) => Host::GreaterWord,
        (Comparison::Greater | Comparison::LessZero, 4) => Host::GreaterDword,
        (Comparison::Greater | Comparison::LessZero, _) => Host::GreaterQword,
        _ => return false,
    };
    if target < opcode.requires() {
        return false;
    }
    let left_xmm = vector_source(code, left, 0);
    let right_xmm = match right {
        Some(right) => vector_source(code, right, 1),
        None => {
            packed(code, Host::Xor, 1, 1);
            1
        }
    };
    // CMLT #0 is 0 > x.
    let (first, second) = if operation == Comparison::LessZero {
        (right_xmm, left_xmm)
    } else {
        (left_xmm, right_xmm)
    };
    let commutative = operation == Comparison::Equal;
    if bytes == 8 {
        three::three(code, opcode, 2, first, second, target);
        let destination = vector_destination(code, destination, 2);
        finish_from(code, destination, 2, 8);
        return true;
    }
    // XMM2 is free and distinct from both sources.
    let destination = vector_destination(code, destination, 2);
    binary(
        code,
        opcode,
        destination,
        first,
        second,
        commutative,
        target,
    );
    true
}

/// AND, ORR (and MOV, its alias with equal sources), EOR, ADD and SUB from
/// the sources where they are into the destination: one VEX instruction
/// writes a 16-byte result to the destination's home.
#[allow(clippy::too_many_arguments)]
fn integer_binary(
    code: &mut Code,
    operation: IntegerOperation,
    destination: u8,
    left: u8,
    right: u8,
    bytes: u8,
    lane_bytes: u8,
    target: crate::TargetCpu,
) {
    use IntegerOperation::*;
    let left_xmm = vector_source(code, left, 0);
    if operation == Orr && left == right {
        let destination = vector_destination(code, destination, 0);
        finish_from(code, destination, left_xmm, bytes);
        return;
    }
    let right_xmm = vector_source(code, right, 1);
    let (opcode, commutative) = match operation {
        And => (Host::And, true),
        Orr => (Host::Or, true),
        Eor => (Host::Xor, true),
        Add => (
            match lane_bytes {
                1 => Host::AddByte,
                2 => Host::AddWord,
                4 => Host::AddDword,
                8 => Host::AddQword,
                _ => unreachable!("reviewed lane size"),
            },
            true,
        ),
        Sub => (subtract_opcode(lane_bytes), false),
        _ => unreachable!("a binary integer operation"),
    };
    if bytes == 8 {
        three::three(code, opcode, 0, left_xmm, right_xmm, target);
        let destination = vector_destination(code, destination, 0);
        finish_from(code, destination, 0, 8);
        return;
    }
    let destination = vector_destination(code, destination, 0);
    binary(
        code,
        opcode,
        destination,
        left_xmm,
        right_xmm,
        commutative,
        target,
    );
}

/// [`emit_for_target`] with guest vectors already held whole in XMM0, or
/// in XMM0 and XMM1 for a pair transfer.
pub(super) fn emit_after(
    code: &mut Code,
    operand: &Operand,
    target: crate::TargetCpu,
    checks: MemoryChecks,
    xmm0: Option<u8>,
    xmm_pair: Option<[u8; 2]>,
) {
    match *operand {
        Operand::Lut(ref operand) => lut::emit(code, operand, target),
        Operand::Aes(ref operand) => aes::emit(code, operand, target),
        Operand::I8mm(ref operand) => i8mm::emit(code, operand, target),
        Operand::Sha1(ref operand) => sha1::emit(code, operand, target),
        Operand::Sha256(ref operand) => sha256::emit(code, operand, target),
        Operand::Sha3(ref operand) => sha3::emit(code, operand, target),
        Operand::Sha512(ref operand) => sha512::emit(code, operand, target),
        Operand::Sm3(ref operand) => sm3::emit(code, operand),
        Operand::Sm4(ref operand) => sm4::emit(code, operand, target),
        Operand::Floating(ref operand) => floating::emit(code, operand, target),
        Operand::MemoryRegisterOffset(ref operand) => {
            memory::emit_register_offset(code, operand, checks, xmm0 == Some(operand.vector))
        }
        Operand::StructureMemory(ref operand) => {
            structure_memory::emit(code, operand, target, checks)
        }
        Operand::Immediate(crate::aarch64::simd::immediates::Operand::LaneMove(
            ref lane @ crate::aarch64::simd::rearrange::Operand::Extract {
                source, byte: 0, ..
            },
        ))
        | Operand::Rearrange(
            ref lane @ crate::aarch64::simd::rearrange::Operand::Extract {
                source, byte: 0, ..
            },
        ) if xmm0 == Some(source) => rearrange::extract(code, lane, true, target),
        Operand::Immediate(ref operand) => immediates::emit(code, operand, target),
        Operand::Halving(ref operand) => halving::emit(code, operand, target),
        Operand::Product(ref operand) => products::emit(code, operand, target),
        Operand::Saturate(ref operand) => saturating::emit(code, operand, target),
        Operand::Reduce(ref operand) => reductions::emit(code, operand, target),
        Operand::Shift(ref operand) => shifts::emit(code, operand, target),
        Operand::Rearrange(ref operand) => rearrange::emit(code, operand, target, xmm0),
        Operand::Compare {
            operation,
            destination,
            left,
            right,
            bytes,
            lane_bytes,
        } => {
            use Comparison::*;
            if single_compare(
                code,
                operation,
                destination,
                left,
                right,
                bytes,
                lane_bytes,
                target,
            ) {
                return;
            }
            // AVX-512 compares unsigned lanes without a bias (CMHI through
            // `greater`) and tests lanes directly. Signed relations keep one
            // SSE compare, inverted where needed: a mask round trip is slower.
            let extended = vector_extensions::compare_supported(lane_bytes, target)
                && matches!(operation, HigherEqual | Test);
            let reverse = !extended && matches!(operation, GreaterEqual | HigherEqual | LessZero);
            let left_xmm = u8::from(reverse);
            let right_xmm = u8::from(!reverse);
            read_vector(code, left, left_xmm);
            if let Some(right) = right {
                read_vector(code, right, right_xmm);
            } else {
                packed(code, Host::Xor, right_xmm, right_xmm);
            }
            if extended && operation == Test {
                vector_extensions::test(code, lane_bytes, 0, 1, target);
            } else if extended {
                let predicate = vector_extensions::Predicate::GreaterEqual;
                vector_extensions::compare(code, lane_bytes, false, predicate, 0, 1, target);
            } else {
                match operation {
                    Equal => equal(code, lane_bytes, target),
                    Test => {
                        packed(code, Host::And, 0, 1);
                        packed(code, Host::Xor, 1, 1);
                        equal(code, lane_bytes, target);
                    }
                    _ => greater(
                        code,
                        lane_bytes,
                        matches!(operation, Higher | HigherEqual),
                        target,
                    ),
                }
                if matches!(operation, GreaterEqual | HigherEqual | LessEqualZero | Test) {
                    packed(code, Host::EqualDword, 1, 1);
                    packed(code, Host::Xor, 0, 1);
                }
            }
            if bytes == 8 {
                packed(code, Host::MoveQword, 0, 0);
            }
            commit(code, destination, 0);
        }
        Operand::Integer {
            operation,
            destination,
            left,
            right,
            bytes,
            lane_bytes,
        } => {
            use IntegerOperation::*;
            // MOV is the ORR alias with two identical sources.
            let copy = operation == Orr && left == right;
            if copy || matches!(operation, And | Orr | Eor | Add | Sub) {
                integer_binary(
                    code,
                    operation,
                    destination,
                    left,
                    right,
                    bytes,
                    lane_bytes,
                    target,
                );
                return;
            }
            // Stage the sources before writing any overlapping V register.
            read_vector(code, left, 0);
            let mut result = 0;
            // The right source, which these operations only read, stays in
            // its home where it has one.
            let mut right_xmm = 1;
            if !copy && !matches!(operation, Abs | Neg) {
                if matches!(operation, And | Orr | Eor | Add | Sub | Bsl | Bit) {
                    right_xmm = vector_source(code, right, 1);
                } else {
                    read_vector(code, right, 1);
                }
            }
            match operation {
                And => packed(code, Host::And, 0, right_xmm),
                Bic => {
                    packed(code, Host::AndNot, 1, 0);
                    result = 1;
                }
                Orr if copy => {}
                Orr => packed(code, Host::Or, 0, right_xmm),
                Orn => {
                    packed(code, Host::EqualDword, 2, 2); // pcmpeqd: all ones, including dirty Q=0 upper bits
                    packed(code, Host::Xor, 1, 2);
                    packed(code, Host::Or, 0, 1);
                }
                Eor => packed(code, Host::Xor, 0, right_xmm),
                Bsl | Bit | Bif => {
                    // Ternary inputs are left, right and the old destination.
                    let table = match operation {
                        Bsl => 0xe4,
                        Bit => 0xe2,
                        Bif => 0xb8,
                        _ => unreachable!(),
                    };
                    let old = vector_operand(code, destination);
                    if !vector_extensions::ternary(code, 0, right_xmm, old, table, target) {
                        let old = vector_source(code, destination, 2);
                        if operation == Bsl {
                            packed(code, Host::Xor, 0, right_xmm);
                            packed(code, Host::And, 0, old);
                            packed(code, Host::Xor, 0, right_xmm);
                        } else {
                            packed(code, Host::Xor, 0, old);
                            if operation == Bit {
                                packed(code, Host::And, 0, right_xmm);
                                packed(code, Host::Xor, 0, old);
                            } else {
                                packed(code, Host::AndNot, 1, 0);
                                packed(code, Host::Xor, 1, old);
                                result = 1;
                            }
                        }
                    }
                }
                Add => packed(
                    code,
                    match lane_bytes {
                        1 => Host::AddByte,
                        2 => Host::AddWord,
                        4 => Host::AddDword,
                        8 => Host::AddQword,
                        _ => unreachable!("reviewed lane size"),
                    },
                    0,
                    right_xmm,
                ),
                Sub => packed(code, subtract_opcode(lane_bytes), 0, right_xmm),
                Neg => {
                    packed(code, Host::Xor, 1, 1);
                    packed(code, subtract_opcode(lane_bytes), 1, 0);
                    result = 1;
                }
                Abs => {
                    if lane_bytes <= 4 && target >= crate::TargetCpu::Ssse3 {
                        let opcode = match lane_bytes {
                            1 => Host::AbsByte,
                            2 => Host::AbsWord,
                            4 => Host::AbsDword,
                            _ => unreachable!(),
                        };
                        opcode.register(code, 0, 0, target);
                    } else if lane_bytes != 8
                        || !vector_extensions::quadword_absolute(code, 0, target)
                    {
                        packed(code, Host::Xor, 1, 1);
                        packed(
                            code,
                            match lane_bytes {
                                1 => Host::GreaterByte,
                                2 => Host::GreaterWord,
                                4 | 8 => Host::GreaterDword,
                                _ => unreachable!("reviewed lane size"),
                            },
                            1,
                            0,
                        );
                        if lane_bytes == 8 {
                            // SSE2 lacks a signed qword comparison: repeat each high
                            // dword's sign across its qword, using only its sign bit.
                            shuffle(code, 1, 1, 0xf5);
                        }
                        packed(code, Host::Xor, 0, 1);
                        packed(code, subtract_opcode(lane_bytes), 0, 1);
                    }
                }
            }
            if bytes == 8 {
                // Arm clears the complete upper half on scalar/D vector writes.
                packed(code, Host::MoveQword, result, result);
            }
            commit(code, destination, result);
        }
        Operand::MemoryPair {
            load,
            bytes,
            vectors,
            base,
            displacement,
            mode,
        } => {
            let access = super::Access::of(!load, 2 * u16::from(bytes));
            memory_address(
                code,
                checks,
                crate::x86_64::ImmediateAddress {
                    base,
                    displacement,
                    mode,
                },
                bytes,
                access,
            );
            if load {
                memory::wide_transfer(code, true, bytes, 0, 0);
                memory::wide_transfer(code, true, bytes, 1, bytes);
                // Neither guest destination changes if either load faults.
                // Duplicate destinations were declined by the decoder.
                commit(code, vectors[0], 0);
                commit(code, vectors[1], 1);
            } else {
                // A matching pair already holds both sources. A translated
                // address check can call a helper that clobbers XMM registers,
                // so read both sources after it on that path.
                if xmm_pair != Some(vectors) || checks.translated {
                    read_vector(code, vectors[0], 0);
                    read_vector(code, vectors[1], 1);
                }
                memory::wide_transfer(code, false, bytes, 0, 0);
                memory::wide_transfer(code, false, bytes, 1, bytes);
            }
            // The first element stays written if the second faults. Base
            // writeback requires both to complete.
            memory_writeback(code, base, mode);
        }
        Operand::MoveImmediate {
            vector,
            mask,
            two_lanes,
        } => {
            if mask == 0 {
                packed(code, Host::Xor, 0, 0);
                commit(code, vector, 0);
                return;
            }
            move_constant(code, 0, mask);
            if two_lanes {
                packed(code, Host::UnpackLowQword, 0, 0);
            }
            commit(code, vector, 0);
        }
        Operand::Memory {
            load,
            bytes,
            vector,
            base,
            displacement,
            mode,
        } => {
            let access = super::Access::of(!load, bytes.into());
            memory_address(
                code,
                checks,
                crate::x86_64::ImmediateAddress {
                    base,
                    displacement,
                    mode,
                },
                bytes,
                access,
            );
            memory::transfer(code, load, bytes, vector, xmm0 == Some(vector));
            // RDX keeps the potential writeback across all SSE/GPR staging.
            memory_writeback(code, base, mode);
        }
    }
}

/// The guest vector XMM0 holds whole after `operand`'s emission, `code`,
/// where it held `held` before: a vector load's or MOVI's destination,
/// committed from it, or a vector store's source, transferred from it; the
/// destination of a branch-free emission that ends with its commit from
/// XMM0; or, after a lane move to a general register, which leaves XMM0
/// alone, what it held.
pub(super) fn xmm0_after(
    scratch: &mut Code,
    operand: &Operand,
    code: &[u8],
    held: Option<u8>,
    homes: &super::vector_cache::VectorCache,
) -> Option<u8> {
    let cached = homes.active();
    let mut committed = |vector: u8| {
        scratch.clear();
        match homes.peek(vector) {
            // The commit to a home from XMM0.
            Some(host) => scratch.asm_unguarded().sse(
                Some(0x66),
                &[0x0f, 0x6f],
                host,
                crate::x86_64::asm::Rm::Xmm(0),
            ),
            None => commit(scratch, vector, 0),
        }
        code.ends_with(scratch).then_some(vector)
    };
    match *operand {
        Operand::Immediate(crate::aarch64::simd::immediates::Operand::LaneMove(
            crate::aarch64::simd::rearrange::Operand::Extract { .. },
        ))
        | Operand::Rearrange(crate::aarch64::simd::rearrange::Operand::Extract { .. }) => held,
        Operand::Rearrange(crate::aarch64::simd::rearrange::Operand::ByteExtract(operand)) => {
            committed(operand.destination)
        }
        Operand::Reduce(ref operand) => committed(operand.destination),
        Operand::Compare { destination, .. } | Operand::Integer { destination, .. } => {
            committed(destination)
        }
        Operand::Immediate(crate::aarch64::simd::immediates::Operand::Constant {
            destination,
            ..
        }) => committed(destination),
        // With vectors in host registers, a transfer may use the home.
        Operand::Memory { .. } | Operand::MemoryRegisterOffset(_) if cached => None,
        Operand::Memory {
            load,
            bytes,
            vector,
            ..
        } => memory::transfers_xmm0(load, bytes).then_some(vector),
        Operand::MemoryRegisterOffset(ref operand) => {
            memory::transfers_xmm0(operand.load, operand.bytes).then_some(operand.vector)
        }
        Operand::MoveImmediate { vector, .. } => Some(vector),
        _ => None,
    }
}

/// [`read_vector`]'s bytes for `vector` into XMM0, which an emission that
/// begins with them does not need where XMM0 holds the vector already.
pub(super) fn xmm0_load(scratch: &mut Code, vector: u8) -> &[u8] {
    scratch.clear();
    read_vector(scratch, vector, 0);
    scratch
}

/// [`read_vector`]'s bytes for a vector whose home is `host` into XMM0.
pub(super) fn xmm0_home_load(scratch: &mut Code, host: u8) -> &[u8] {
    scratch.clear();
    packed(scratch, Host::MoveAligned, 0, host);
    scratch
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aarch64::{AddressMode, Register};
    use alloc::vec;

    mod numerical_boundary {
        use crate::GuestAddress;
        use alloc::vec::Vec;

        const FCVT_H_D: u32 = 0x1e63c020;
        const SCVTF_H_W: u32 = 0x1ee20020;
        const FCVTAS_4S: u32 = 0x4e21c820;
        const FCVT_H_S: u32 = 0x1e23c020;
        const FCVTN_4H_4S: u32 = 0x0e216820;
        const FCVTAS_W_S: u32 = 0x1e240020;
        const FJCVTZS_W_D: u32 = 0x1e7e0020;
        const FCVT_S_H: u32 = 0x1ee24020;
        const SCVTF_S_W: u32 = 0x1e220020;
        const FCVTL_4S_4H: u32 = 0x0e217820;
        const UNSUPPORTED_SUFFIX: u32 = u32::MAX;

        #[test]
        fn terminal_conversions_do_not_consume_or_validate_an_unreached_suffix() {
            for word in [FCVT_H_D, SCVTF_H_W, FCVTAS_4S] {
                let bytes: Vec<u8> = [word, UNSUPPORTED_SUFFIX]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                let compiled = crate::testing::block(&bytes, GuestAddress(0x1000)).unwrap();
                assert_eq!(compiled.source_bytes, size_of::<u32>());
                assert_eq!(compiled.native.source_map, [(GuestAddress(0x1000), 0)]);
            }
            // Inline on every target: the block continues into the suffix.
            for word in [
                FCVT_S_H,
                SCVTF_S_W,
                FCVTL_4S_4H,
                FJCVTZS_W_D,
                FCVT_H_S,
                FCVTN_4H_4S,
                FCVTAS_W_S,
            ] {
                let bytes: Vec<u8> = [word, UNSUPPORTED_SUFFIX]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                assert!(crate::testing::block(&bytes, GuestAddress(0x1000)).is_err());
            }
        }
    }

    #[test]
    fn bit_select_uses_ternary_logic_only_with_f_and_vl() {
        use crate::TargetCpu;

        let baseline = TargetCpu::Baseline;
        let extended = baseline.with_avx512f().with_avx512vl();
        for operation in [
            IntegerOperation::Bsl,
            IntegerOperation::Bit,
            IntegerOperation::Bif,
        ] {
            for bytes in [8, 16] {
                let operand = Operand::Integer {
                    operation,
                    destination: 31,
                    left: 1,
                    right: 2,
                    bytes,
                    lane_bytes: 1,
                };
                let emit = |target| {
                    let mut code = Code::new();
                    emit_for_target(&mut code, &operand, target, MemoryChecks::DEFAULT);
                    code
                };
                let old = emit(baseline);
                let new = emit(extended);
                assert_eq!(old.instructions(), new.instructions() + 3);
                assert!(new.len() < old.len());
                assert!(
                    new.windows(5)
                        .any(|word| word[0] == 0x62 && word[4] == 0x25)
                );
                for target in [
                    TargetCpu::Ssse3,
                    TargetCpu::Sse41,
                    TargetCpu::X86_64_V3,
                    baseline.with_avx512f(),
                    baseline.with_avx512vl(),
                    extended.with_data_independent_timing(),
                ] {
                    assert_eq!(&emit(target)[..], &old[..], "{operation:?} {target:?}");
                }
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
    #[test]
    fn bit_select_matches_arm_masks_for_each_width_and_alias() {
        use crate::{
            GuestAddress, GuestState, Options, TargetCpu,
            block::{BlockProfile, CompilationPlan},
        };

        let pc = GuestAddress(0x2a96b0);
        let baseline = TargetCpu::Baseline;
        let extended = baseline.with_avx512f().with_avx512vl();
        // BIF V0.16B,V2.16B,V4.16B is from the ripgrep archive.
        for (form, mask) in [(0x2e60_1c00_u32, 0), (0x2ea0_1c00, 1), (0x2ee0_1c00, 2)] {
            for q in [0, 1] {
                for [d, n, m] in [[0, 2, 4], [0, 0, 4], [0, 2, 0], [0, 2, 2], [31, 31, 31]] {
                    let word = form | (q << 30) | (m << 16) | (n << 5) | d;
                    for target in [baseline, extended] {
                        if !TargetCpu::detect().supports(target) {
                            continue;
                        }
                        let block = CompilationPlan::new(
                            &word.to_le_bytes(),
                            pc,
                            &Options::new(BlockProfile::ORDINARY, target),
                        )
                        .unwrap()
                        .compile()
                        .unwrap();
                        let mut state = GuestState::default();
                        for (register, lanes) in state.v.iter_mut().enumerate() {
                            let bits = (register as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                            *lanes = [bits ^ 0xcccc_cccc_cccc_cccc, !bits.rotate_left(23)];
                        }
                        let mut expected = state.clone();
                        for lane in 0..=q as usize {
                            let [old, left, right] = [d, n, m].map(|r| state.v[r as usize][lane]);
                            expected.v[d as usize][lane] = match mask {
                                0 => (old & left) | (!old & right),
                                1 => (right & left) | (!right & old),
                                _ => (!right & left) | (right & old),
                            };
                        }
                        if q == 0 {
                            expected.v[d as usize][1] = 0;
                        }
                        assert_eq!(
                            crate::x86_64::native::run(&block.native.x86_64_code, &mut state),
                            pc.0 + 4
                        );
                        assert_eq!(state, expected, "{word:08x} {target:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn q_load_is_unaligned_staged_and_committed_before_writeback() {
        let mut code = Code::new();
        emit(
            &mut code,
            &Operand::Memory {
                load: true,
                bytes: 16,
                vector: 31,
                base: Register::Sp,
                displacement: -16,
                mode: AddressMode::PreIndex,
            },
        );
        assert_eq!(&code[7..13], &[0xa8, 15, 0x74, 2, 0x0f, 0x0b]);
        let suffix = [
            0xf3, 0x0f, 0x6f, 0x00, 0xf3, 0x0f, 0x7f, 0x87, 0xf8, 2, 0, 0, 0x48, 0x89, 0x97, 248,
            0, 0, 0,
        ];
        assert!(code.ends_with(&suffix));
    }

    #[test]
    fn scalar_accesses_use_exact_width_and_zero_high_vector_bits() {
        for (bytes, load_bytes, store_bytes) in [
            (
                1,
                vec![0x0f, 0xb6, 0x08, 0x66, 0x0f, 0x6e, 0xc1],
                vec![0x88, 0x08],
            ),
            (
                2,
                vec![0x0f, 0xb7, 0x08, 0x66, 0x0f, 0x6e, 0xc1],
                vec![0x66, 0x89, 0x08],
            ),
            (4, vec![0x66, 0x0f, 0x6e, 0], vec![0x66, 0x0f, 0x7e, 0]),
            (8, vec![0xf3, 0x0f, 0x7e, 0], vec![0x66, 0x0f, 0xd6, 0]),
        ] {
            for load in [false, true] {
                let mut code = Code::new();
                emit(
                    &mut code,
                    &Operand::Memory {
                        load,
                        bytes,
                        vector: 0,
                        base: Register::X(0),
                        displacement: 0,
                        mode: AddressMode::Offset,
                    },
                );
                if load {
                    assert!(code[7..].starts_with(&load_bytes));
                    assert!(code.ends_with(&[0xf3, 0x0f, 0x7f, 0x87, 8, 1, 0, 0]));
                } else {
                    assert!(code.ends_with(&store_bytes));
                }
            }
        }
    }

    #[test]
    fn movi_d_clears_high_half_and_two_d_replicates_without_new_cpu_features() {
        for two_lanes in [false, true] {
            let mut code = Code::new();
            emit(
                &mut code,
                &Operand::MoveImmediate {
                    vector: 31,
                    mask: 0xff00ff00ff00ff00,
                    two_lanes,
                },
            );
            assert_eq!(&code[..2], &[0x48, 0xb9]);
            assert_eq!(&code[2..10], &0xff00ff00ff00ff00u64.to_le_bytes());
            assert_eq!(&code[10..15], &[0x66, 0x48, 0x0f, 0x6e, 0xc1]);
            if two_lanes {
                assert_eq!(&code[15..19], &[0x66, 0x0f, 0x6c, 0xc0]);
            }
            assert!(code.ends_with(&[0xf3, 0x0f, 0x7f, 0x87, 0xf8, 2, 0, 0]));
        }
    }

    #[test]
    fn zero_byte_masks_share_compact_full_vector_zeroing() {
        for two_lanes in [false, true] {
            let mut code = Code::new();
            emit(
                &mut code,
                &Operand::MoveImmediate {
                    vector: 0,
                    mask: 0,
                    two_lanes,
                },
            );
            assert_eq!(
                code,
                [0x66, 0x0f, 0xef, 0xc0, 0xf3, 0x0f, 0x7f, 0x87, 8, 1, 0, 0]
            );
        }
    }

    #[test]
    fn pair_loads_stage_both_exact_width_values_before_guest_commits_and_writeback() {
        for (bytes, prefix, opcode) in [(4, 0x66, 0x6e), (8, 0xf3, 0x7e), (16, 0xf3, 0x6f)] {
            let mut code = Code::new();
            emit(
                &mut code,
                &Operand::MemoryPair {
                    load: true,
                    bytes,
                    vectors: [0, 31],
                    base: Register::X(0),
                    displacement: -64,
                    mode: AddressMode::PreIndex,
                },
            );
            assert_eq!(
                &code[14..23],
                &[prefix, 0x0f, opcode, 0, prefix, 0x0f, opcode, 0x48, bytes]
            );
            assert_eq!(&code[23..31], &[0xf3, 0x0f, 0x7f, 0x87, 8, 1, 0, 0]);
            assert_eq!(&code[31..39], &[0xf3, 0x0f, 0x7f, 0x8f, 0xf8, 2, 0, 0]);
            assert_eq!(&code[39..], &[0x48, 0x89, 0x97, 0, 0, 0, 0]);
        }
    }

    #[test]
    fn pair_stores_capture_duplicate_sources_before_two_writes_and_delay_writeback() {
        for (bytes, prefix, opcode) in [(4, 0x66, 0x7e), (8, 0x66, 0xd6), (16, 0xf3, 0x7f)] {
            let mut code = Code::new();
            emit(
                &mut code,
                &Operand::MemoryPair {
                    load: false,
                    bytes,
                    vectors: [31, 31],
                    base: Register::Sp,
                    displacement: 32,
                    mode: AddressMode::PostIndex,
                },
            );
            assert_eq!(&code[7..13], &[0xa8, 15, 0x74, 2, 0x0f, 0x0b]);
            assert_eq!(&code[17..25], &[0xf3, 0x0f, 0x6f, 0x87, 0xf8, 2, 0, 0]);
            assert_eq!(&code[25..33], &[0xf3, 0x0f, 0x6f, 0x8f, 0xf8, 2, 0, 0]);
            assert_eq!(
                &code[33..42],
                &[prefix, 0x0f, opcode, 0, prefix, 0x0f, opcode, 0x48, bytes]
            );
            assert_eq!(&code[42..], &[0x48, 0x89, 0x97, 248, 0, 0, 0]);
        }
    }

    #[test]
    fn vector_move_keeps_one_source_load_and_one_commit() {
        use crate::TargetCpu;

        for bytes in [8, 16] {
            let emit = |right, target| {
                let mut code = Code::new();
                emit_for_target(
                    &mut code,
                    &Operand::Integer {
                        operation: IntegerOperation::Orr,
                        destination: 18,
                        left: 0,
                        right,
                        bytes,
                        lane_bytes: 1,
                    },
                    target,
                    MemoryChecks::DEFAULT,
                );
                code
            };
            let baseline = TargetCpu::Baseline;
            let mov = emit(0, baseline);
            let orr = emit(1, baseline);
            let extra = usize::from(bytes == 8);
            assert_eq!((mov.instructions(), mov.len()), (2 + extra, 16 + 4 * extra));
            assert_eq!((orr.instructions(), orr.len()), (4 + extra, 28 + 4 * extra));
            for target in [
                TargetCpu::Ssse3,
                TargetCpu::Sse41,
                TargetCpu::X86_64_V3,
                baseline.with_data_independent_timing(),
            ] {
                assert_eq!(&emit(0, target)[..], &mov[..]);
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
    #[test]
    fn vector_move_preserves_frame_and_zeroes_d_upper_half() {
        use crate::{
            GuestAddress, GuestState, Options, TargetCpu,
            block::{BlockProfile, CompilationPlan},
        };

        let pc = GuestAddress(0x259f6c);
        // MOV V18.16B,V0.16B from the OpenSSL SHA-256 block prologue.
        for (destination, source) in [(18, 0), (0, 0), (31, 31)] {
            for q in [0, 1] {
                let word =
                    0x0ea0_1c00_u32 | (q << 30) | (source << 16) | (source << 5) | destination;
                let block = CompilationPlan::new(
                    &word.to_le_bytes(),
                    pc,
                    &Options::new(BlockProfile::ORDINARY, TargetCpu::Baseline),
                )
                .unwrap()
                .compile()
                .unwrap();
                let mut state = GuestState::default();
                for (register, lanes) in state.v.iter_mut().enumerate() {
                    *lanes = [
                        0xfedc_ba98_7654_3210 ^ register as u64,
                        0x0123_4567_89ab_cdef,
                    ];
                }
                let mut expected = state.clone();
                expected.v[destination as usize] = state.v[source as usize];
                if q == 0 {
                    expected.v[destination as usize][1] = 0;
                }
                assert_eq!(block.native.source_map, [(pc, 0)]);
                assert_eq!(
                    crate::x86_64::native::run(&block.native.x86_64_code, &mut state),
                    pc.0 + 4
                );
                assert_eq!(state, expected, "{word:08x}");
            }
        }
    }
}
