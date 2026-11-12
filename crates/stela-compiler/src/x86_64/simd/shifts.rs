//! AdvSIMD shifts by immediate and per-lane register, including width changes.
use super::{
    Host, binary, commit, finish, finish_from, greater, packed, read_vector, rearrange::constant,
    shuffle, three, vector_accumulator, vector_destination, vector_offset, vector_source,
};
use crate::x86_64::asm::{Alu, Code, Cond, Gpr, ShiftOp, Size, Unary, vector_extensions};
use crate::x86_64::simd::encode::guest;
use crate::{
    TargetCpu,
    aarch64::simd::shifts::{Kind, Operand},
    x86_64::isa::ShiftOpcode as Shift,
};
use Gpr::{R8, R9, R10, R11, Rax, Rcx, Rdx};
use alloc::vec::Vec;

const BITS_PER_BYTE: u8 = 8;
const VECTOR_BYTES: u8 = 16;
const LOW_DWORDS_OF_QWORDS: u8 = 0x88;
const HIGH_DWORDS_OF_QWORDS: u8 = 0xdd;

fn lane_mask(code: &mut Code, xmm: u8, lane_bytes: u8, value: u64) {
    let bytes = value.to_le_bytes();
    constant(
        code,
        xmm,
        core::array::from_fn(|i| bytes[i % usize::from(lane_bytes)]),
    );
}
fn add(code: &mut Code, destination: u8, source: u8, lane_bytes: u8) {
    packed(
        code,
        match lane_bytes {
            1 => Host::AddByte,
            2 => Host::AddWord,
            4 => Host::AddDword,
            8 => Host::AddQword,
            _ => unreachable!(),
        },
        destination,
        source,
    );
}
/// Shift each byte of `xmm` by `amount` with one GFNI affine transform: left,
/// logically right, or right filling with the sign (`signed`). The matrix
/// goes through `gpr` into XMM5.
pub(crate) fn byte_shift(
    code: &mut Code,
    xmm: u8,
    amount: u8,
    left: bool,
    signed: bool,
    gpr: Gpr,
    target: TargetCpu,
) -> bool {
    let matrix = vector_extensions::byte_matrix(|bit| {
        if left {
            bit.checked_sub(amount)
        } else if signed {
            Some((bit + amount).min(BITS_PER_BYTE - 1))
        } else {
            Some(bit + amount).filter(|from| *from < BITS_PER_BYTE)
        }
    });
    vector_extensions::byte_affine(code, xmm, 5, matrix, gpr, target)
}

/// Byte lanes use [`byte_shift`] when available, else a word shift and a lane
/// mask; either clobbers XMM5 and RCX (and the mask XMM7).
pub(crate) fn logical_shift(
    code: &mut Code,
    xmm: u8,
    lane_bytes: u8,
    amount: u8,
    left: bool,
    target: TargetCpu,
) {
    if amount == 0 {
        return;
    }
    if lane_bytes == 1 && byte_shift(code, xmm, amount, left, false, Rcx, target) {
        return;
    }
    let opcode = match (left, lane_bytes) {
        (true, 1 | 2) => Shift::LeftWord,
        (true, 4) => Shift::LeftDword,
        (true, 8) => Shift::LeftQword,
        (false, 1 | 2) => Shift::RightWord,
        (false, 4) => Shift::RightDword,
        (false, 8) => Shift::RightQword,
        _ => unreachable!("reviewed lane width"),
    };
    opcode.emit(code, xmm, amount);
    if lane_bytes == 1 {
        // Word shifts otherwise move bits across the neighboring byte lane.
        let mask = if left {
            (0xffu16 << amount) & 0xff
        } else {
            0xffu16 >> amount
        };
        lane_mask(code, 5, 1, u64::from(mask));
        packed(code, Host::And, xmm, 5);
    }
}
/// Signed qwords use VPSRAQ and signed bytes a GFNI affine transform when
/// available; the replicated-sign recipes clobber XMM4 as well.
pub(crate) fn right_shift(
    code: &mut Code,
    xmm: u8,
    lane_bytes: u8,
    amount: u8,
    signed: bool,
    target: TargetCpu,
) {
    if amount == 0 {
        return;
    }
    if !signed {
        logical_shift(code, xmm, lane_bytes, amount, false, target);
        return;
    }
    let extended = match lane_bytes {
        1 => byte_shift(code, xmm, amount, false, true, Rcx, target),
        8 => vector_extensions::arithmetic_qword_right(code, xmm, amount, target),
        _ => false,
    };
    if extended {
        return;
    }
    match lane_bytes {
        2 => Shift::ArithmeticWord.emit(code, xmm, amount),
        4 => Shift::ArithmeticDword.emit(code, xmm, amount),
        1 | 8 => {
            // XMM4 is the replicated lane sign. Signed qwords use their high
            // dword; signed bytes use PCMPGTB zero,source. Both handle shift=width.
            if lane_bytes == 8 {
                packed(code, Host::MoveAligned, 4, xmm);
                Shift::ArithmeticDword.emit(code, 4, 31);
                shuffle(code, 4, 4, 0xf5);
            } else {
                packed(code, Host::Xor, 4, 4);
                packed(code, Host::GreaterByte, 4, xmm);
            }
            logical_shift(code, xmm, lane_bytes, amount, false, target);
            logical_shift(code, 4, lane_bytes, lane_bytes * 8 - amount, true, target);
            packed(code, Host::Or, xmm, 4);
        }
        _ => unreachable!("reviewed lane width"),
    }
}
pub(crate) fn rounded_right(
    code: &mut Code,
    lane_bytes: u8,
    amount: u8,
    signed: bool,
    round: bool,
    target: TargetCpu,
) {
    rounded_right_from(code, 0, 0, lane_bytes, amount, signed, round, target);
}

/// The key of the per-block constant 1 in every lane ([`Code::derive`]).
///
/// [`Code::derive`]: crate::x86_64::asm::Code::derive
const LANE_ONES: u32 = 0x200;

/// A derived bank register holding 1 in every `lane_bytes` lane, kept for
/// the block (PCMPEQD, one shift and one EVEX move through XMM1 the first
/// time).
fn lane_ones(code: &mut Code, lane_bytes: u8) -> u8 {
    let key = crate::x86_64::vector_cache::Derived {
        key: LANE_ONES | u32::from(lane_bytes),
        source: None,
    };
    code.derive(key, |code, host| {
        packed(code, Host::EqualDword, 1, 1);
        let right = match lane_bytes {
            2 => Shift::RightWord,
            _ => Shift::RightDword,
        };
        right.emit(code, 1, lane_bytes * BITS_PER_BYTE - 1);
        three::evex_move(code, host, 1);
    })
}

/// The single host shift by immediate that shifts `lane_bytes` lanes
/// right, where there is one: logical H, S and D lanes, arithmetic H and S.
fn single_right_shift(lane_bytes: u8, signed: bool) -> Option<Shift> {
    match (signed, lane_bytes) {
        (false, 2) => Some(Shift::RightWord),
        (false, 4) => Some(Shift::RightDword),
        (false, 8) => Some(Shift::RightQword),
        (true, 2) => Some(Shift::ArithmeticWord),
        (true, 4) => Some(Shift::ArithmeticDword),
        _ => None,
    }
}

/// [`rounded_right`] of `source` (a home or XMM0) into `destination` (XMM0
/// or a destination home), which only the last instruction writes. VEX
/// forms read the source in place. Clobbers XMM0 and XMM1 (and the
/// temporaries of [`right_shift`]).
#[allow(clippy::too_many_arguments)]
fn rounded_right_from(
    code: &mut Code,
    destination: u8,
    source: u8,
    lane_bytes: u8,
    amount: u8,
    signed: bool,
    round: bool,
    target: TargetCpu,
) {
    let single = single_right_shift(lane_bytes, signed).filter(|_| amount != 0);
    if let Some(opcode) = single
        && round
        && amount >= 2
        && lane_bytes <= 4
        && code.vector_cache().active()
        && three::derived_bank(target)
    {
        // With ones per lane kept for the block: (x >> (n-1)) + 1, then
        // >> 1. The first shift leaves room for the + 1 at the lane's
        // maximum, so the sum is exact where x + 2^(n-1) would wrap.
        let ones = lane_ones(code, lane_bytes);
        if source == 0 {
            opcode.emit(code, 0, amount - 1);
        } else {
            three::shift(code, opcode, 0, source, amount - 1, target);
        }
        let add = super::rearrange::add_or_subtract(lane_bytes, false);
        three::evex_three(code, add, 0, 0, ones);
        if destination == 0 {
            opcode.emit(code, 0, 1);
        } else {
            three::shift(code, opcode, destination, 0, 1, target);
        }
        return;
    }
    if round && lane_bytes > 1 && !target.data_independent_timing() {
        // The rounding bit, bit amount-1, alone in bit 0 of XMM1: shift it to
        // the top of the lane, then down to the bottom. No constant.
        let bits = lane_bytes * BITS_PER_BYTE;
        let (left, right) = match lane_bytes {
            2 => (Shift::LeftWord, Shift::RightWord),
            4 => (Shift::LeftDword, Shift::RightDword),
            _ => (Shift::LeftQword, Shift::RightQword),
        };
        three::shift(code, left, 1, source, bits - amount, target);
        three::shift(code, right, 1, 1, bits - 1, target);
    } else if round {
        packed(code, Host::MoveAligned, 1, source);
        logical_shift(code, 1, lane_bytes, amount - 1, false, target);
        lane_mask(code, 2, lane_bytes, 1);
        packed(code, Host::And, 1, 2);
    }
    let shifted = if round { 0 } else { destination };
    match single {
        // The legacy form in place, a VEX form from the source.
        Some(opcode) if shifted == source => opcode.emit(code, source, amount),
        Some(opcode) => three::shift(code, opcode, shifted, source, amount, target),
        None => {
            if source != 0 {
                packed(code, Host::MoveAligned, 0, source);
            }
            right_shift(code, 0, lane_bytes, amount, signed, target);
            if !round && destination != 0 {
                packed(code, Host::MoveAligned, destination, 0);
            }
        }
    }
    if round {
        let add = super::rearrange::add_or_subtract(lane_bytes, false);
        if destination == 0 {
            packed(code, add, 0, 1);
        } else {
            three::three(code, add, destination, 0, 1, target);
        }
    }
}
#[allow(clippy::too_many_arguments)]
fn narrow(
    code: &mut Code,
    source: u8,
    lane_bytes: u8,
    amount: u8,
    round: bool,
    upper: bool,
    destination: u8,
    target: TargetCpu,
) {
    let source = vector_source(code, source, 0);
    rounded_right_from(
        code,
        0,
        source,
        lane_bytes * 2,
        amount,
        false,
        round,
        target,
    );
    pack_truncated(code, lane_bytes, upper, destination, target);
}

/// Shift first, then clamp. Rounding is computed as shifted value plus the
/// discarded high bit, so signed overflow and unsigned carry are never lost.
fn saturating_narrow(
    code: &mut Code,
    operand: &Operand,
    signed_source: bool,
    unsigned_result: bool,
    round: bool,
    upper: bool,
    target: TargetCpu,
) {
    let lane = operand.lane_bytes;
    let wide = lane * 2;
    let may_saturate =
        round || operand.amount != lane * BITS_PER_BYTE || signed_source == unsigned_result;
    let quick = may_saturate
        && operand.bytes != lane
        && quick_qc_supported(wide, signed_source, unsigned_result, target);
    if quick {
        let source = vector_source(code, operand.source, 0);
        rounded_right_from(
            code,
            0,
            source,
            wide,
            operand.amount,
            signed_source,
            round,
            target,
        );
        quick_qc_narrow(code, wide, signed_source, unsigned_result, target);
        let cleared = cleared_by(wide, signed_source, unsigned_result);
        finish_saturating_narrow(code, operand, upper, cleared, target);
        return;
    }
    read_vector(code, operand.source, 0);
    if !may_saturate {
        // Both signednesses extract the source's high half as bits. Arithmetic
        // H/S shifts put that half in signed pack range; D needs only a shuffle.
        // QC stays untouched, including poisoned inactive scalar lanes.
        match lane {
            1 | 2 => {
                right_shift(code, 0, wide, operand.amount, true, target);
                packed(
                    code,
                    if lane == 1 {
                        Host::PackSignedWordToByte
                    } else {
                        Host::PackSignedDwordToWord
                    },
                    0,
                    0,
                );
            }
            4 => shuffle(code, 0, 0, HIGH_DWORDS_OF_QWORDS),
            _ => unreachable!("allocated in-range narrowing width"),
        }
        packed(code, Host::MoveAligned, 6, 0);
    } else {
        rounded_right(code, wide, operand.amount, signed_source, round, target);
        if !vector_extensions::saturating_narrow(
            code,
            wide,
            [signed_source, !unsigned_result],
            6,
            0,
            Some(4),
            target,
        ) {
            if lane < 4 {
                saturating_narrow_pack(code, lane, signed_source, unsigned_result, target);
            } else {
                saturating_narrow_qword(code, unsigned_result, signed_source, target);
            }
        }
    }
    let scalar = operand.bytes == lane;
    if may_saturate {
        super::saturating::publish_qc(code, if scalar { wide } else { VECTOR_BYTES });
    }
    finish_saturating_narrow(code, operand, upper, false, target);
}

/// Write the narrowed XMM6: into the upper half of the destination, or
/// alone with the rest cleared (already, where `cleared`).
fn finish_saturating_narrow(
    code: &mut Code,
    operand: &Operand,
    upper: bool,
    cleared: bool,
    target: TargetCpu,
) {
    let lane = operand.lane_bytes;
    let scalar = operand.bytes == lane;
    if upper {
        let destination = vector_accumulator(code, operand.destination, 1);
        binary(
            code,
            Host::UnpackLowQword,
            destination,
            destination.xmm,
            6,
            false,
            target,
        );
        return;
    }
    if scalar {
        constant(
            code,
            1,
            core::array::from_fn(|i| if i < usize::from(lane) { u8::MAX } else { 0 }),
        );
        packed(code, Host::And, 6, 1);
    }
    let destination = vector_destination(code, operand.destination, 6);
    finish_from(code, destination, 6, if scalar || cleared { 16 } else { 8 });
}

/// Whether [`quick_qc_narrow`] narrows `wide` lanes on `target`: one host
/// pack (signed sources of H and S lanes) or an AVX-512 saturating
/// down-convert, and PTEST (SSE4.1) for QC. The DIT profile keeps the
/// branch-free recipe.
fn quick_qc_supported(
    wide: u8,
    signed_source: bool,
    unsigned_result: bool,
    target: TargetCpu,
) -> bool {
    three::test_zero_supported(target)
        && !target.data_independent_timing()
        && ((signed_source && wide <= 4)
            || vector_extensions::saturating_narrow_supported(wide, target))
        && (signed_source || unsigned_result)
}

/// Whether [`quick_qc_narrow`] leaves the high quadword of XMM6 clear.
fn cleared_by(wide: u8, signed_source: bool, _unsigned_result: bool) -> bool {
    !(signed_source && wide <= 4)
}

/// Saturate every `wide` lane of XMM0 to half width into the low quadword of
/// XMM6 and set FPSR.QC when any lane saturated. Signed sources of H and S
/// lanes use one PACKSS/PACKUS (both halves alike); other narrows one
/// VPMOVS/VPMOVUS. QC: a lane saturated exactly when its bits above the
/// unsigned result width are not all zero (negative sources included), or
/// when the sign-extended signed result differs from the source; one PTEST
/// of that difference, a branch over one sticky OR into FPSR. Clobbers
/// XMM1 and EFLAGS (as the branch-free recipe did); writes only FPSR.
fn quick_qc_narrow(
    code: &mut Code,
    wide: u8,
    signed_source: bool,
    unsigned_result: bool,
    target: TargetCpu,
) {
    if signed_source && wide <= 4 {
        let pack = match (wide, unsigned_result) {
            (2, true) => Host::PackUnsignedWordToByte,
            (2, false) => Host::PackSignedWordToByte,
            (4, true) => Host::PackUnsignedDwordToWord,
            (_, false) => Host::PackSignedDwordToWord,
            _ => unreachable!(),
        };
        three::three(code, pack, 6, 0, 0, target);
    } else {
        let narrowed = vector_extensions::saturating_narrow(
            code,
            wide,
            [signed_source, !unsigned_result],
            6,
            0,
            None,
            target,
        );
        assert!(narrowed, "checked by quick_qc_supported");
    }
    if unsigned_result {
        let high = match wide {
            2 => Shift::RightWord,
            4 => Shift::RightDword,
            _ => Shift::RightQword,
        };
        three::shift(code, high, 1, 0, wide * BITS_PER_BYTE / 2, target);
    } else {
        match wide {
            2 => Host::SignExtendByteToWord,
            4 => Host::SignExtendWordToDword,
            _ => Host::SignExtendDwordToQword,
        }
        .register(code, 1, 6, target);
        three::three(code, Host::Xor, 1, 1, 0, target);
    }
    three::test_zero(code, 1, target);
    let mut asm = code.asm();
    let none = asm.jcc8(Cond::E);
    // Sticky QC, preserving every other bit.
    asm.alu_imm32(
        Alu::Or,
        Size::Dword,
        guest(core::mem::offset_of!(crate::GuestState, fpsr) as u32),
        crate::aarch64::simd::saturating::FPSR_QC as i32,
    );
    asm.bind(none);
}

/// Native packs clamp H/S sources. Reconstructing the result and comparing it
/// with the shifted source derives QC without reimplementing the pack bounds.
/// XMM6 keeps the packed result and XMM4 receives the wide saturation mask.
fn saturating_narrow_pack(
    code: &mut Code,
    lane: u8,
    signed_source: bool,
    unsigned_result: bool,
    target: TargetCpu,
) {
    let wide = lane * 2;
    packed(code, Host::MoveAligned, 3, 0);
    if !signed_source {
        // With a shift of at least one, only one negative signed pattern is
        // possible: unsigned MAX rounded by one becomes exactly the wide sign
        // bit. Change it to signed MAX before the signed host pack (both
        // saturate to the same bound), comparing against the original value.
        super::saturating::sign_mask(code, 4, 0, wide);
        packed(code, Host::Xor, 0, 4);
    }
    match (unsigned_result, lane) {
        (false, 1) => packed(code, Host::PackSignedWordToByte, 0, 0),
        (true, 1) => packed(code, Host::PackUnsignedWordToByte, 0, 0),
        (false, 2) => packed(code, Host::PackSignedDwordToWord, 0, 0),
        (true, 2) if target >= TargetCpu::Sse41 => {
            Host::PackUnsignedDwordToWord.register(code, 0, 0, target);
        }
        (true, 2) => {
            // Translate [0,65535] into signed halfword bounds. The shifted
            // source range keeps this subtraction within signed dword range.
            let bias = 1u64 << (lane * BITS_PER_BYTE - 1);
            lane_mask(code, 1, wide, bias);
            packed(code, Host::SubDword, 0, 1);
            packed(code, Host::PackSignedDwordToWord, 0, 0);
            lane_mask(code, 1, lane, bias);
            packed(code, Host::Xor, 0, 1);
        }
        _ => unreachable!("allocated native narrowing pack"),
    }
    packed(code, Host::MoveAligned, 6, 0);
    if unsigned_result {
        packed(code, Host::Xor, 1, 1);
    } else {
        super::saturating::sign_mask(code, 1, 0, lane);
    }
    packed(
        code,
        if lane == 1 {
            Host::UnpackLowByte
        } else {
            Host::UnpackLowWord
        },
        0,
        1,
    );
    packed(code, Host::MoveAligned, 4, 0);
    packed(
        code,
        if wide == 2 {
            Host::EqualWord
        } else {
            Host::EqualDword
        },
        4,
        3,
    );
    packed(code, Host::EqualDword, 1, 1);
    packed(code, Host::Xor, 4, 1);
}

/// SSE2 has no saturating qword pack. Compare both bounds before selecting
/// clamped qwords; the checked SSE4.2 comparison replaces its baseline recipe.
fn saturating_narrow_qword(
    code: &mut Code,
    unsigned_result: bool,
    signed_source: bool,
    target: TargetCpu,
) {
    const DESTINATION_BYTES: u8 = 4;
    const SOURCE_BYTES: u8 = DESTINATION_BYTES * 2;
    let high = if unsigned_result {
        u64::from(u32::MAX)
    } else {
        i32::MAX as u64
    };
    let low = if unsigned_result {
        0
    } else {
        i64::from(i32::MIN) as u64
    };
    packed(code, Host::MoveAligned, 5, 0);
    lane_mask(code, 1, SOURCE_BYTES, high);
    greater(code, SOURCE_BYTES, !signed_source, target);
    packed(code, Host::MoveAligned, 6, 0);
    if signed_source {
        lane_mask(code, 0, SOURCE_BYTES, low);
        packed(code, Host::MoveAligned, 1, 5);
        greater(code, SOURCE_BYTES, false, target);
    } else {
        packed(code, Host::Xor, 0, 0);
    }
    lane_mask(code, 1, SOURCE_BYTES, low);
    packed(code, Host::Xor, 1, 5);
    packed(code, Host::And, 1, 0);
    packed(code, Host::Xor, 5, 1);
    lane_mask(code, 1, SOURCE_BYTES, high);
    packed(code, Host::Xor, 1, 5);
    packed(code, Host::And, 1, 6);
    packed(code, Host::Xor, 5, 1);
    packed(code, Host::MoveAligned, 4, 0);
    packed(code, Host::Or, 4, 6);
    shuffle(code, 6, 5, LOW_DWORDS_OF_QWORDS);
}
pub(super) fn pack_truncated(
    code: &mut Code,
    lane_bytes: u8,
    upper: bool,
    destination: u8,
    target: TargetCpu,
) {
    if !upper {
        // VPMOV{WB,DW,QD} clears the high quadword itself.
        let destination = vector_destination(code, destination, 0);
        if vector_extensions::truncating_narrow(code, lane_bytes * 2, destination.xmm, 0, target) {
            finish(code, destination);
            return;
        }
        truncate(code, lane_bytes);
        finish_from(code, destination, 0, 8);
        return;
    }
    if !vector_extensions::truncating_narrow(code, lane_bytes * 2, 0, 0, target) {
        truncate(code, lane_bytes);
    }
    let destination = vector_accumulator(code, destination, 1);
    binary(
        code,
        Host::UnpackLowQword,
        destination,
        destination.xmm,
        0,
        false,
        target,
    );
}

/// The low half of each lane of XMM0 packed into its low quadword, by
/// SSE2/SSE4.1 steps.
fn truncate(code: &mut Code, lane_bytes: u8) {
    match lane_bytes {
        1 => {
            lane_mask(code, 2, 2, 0xff);
            packed(code, Host::And, 0, 2);
            packed(code, Host::PackUnsignedWordToByte, 0, 0);
        }
        2 => {
            // Convert each low halfword to a signed dword before PACKSSDW:
            // every input is in range, so this truncates instead of saturating.
            Shift::LeftDword.emit(code, 0, 16);
            Shift::ArithmeticDword.emit(code, 0, 16);
            packed(code, Host::PackSignedDwordToWord, 0, 0);
        }
        4 => shuffle(code, 0, 0, 0x88), // select low dword of each qword
        _ => unreachable!("reviewed narrowing width"),
    }
}
fn widen(
    code: &mut Code,
    lane_bytes: u8,
    amount: u8,
    signed: bool,
    upper: bool,
    target: TargetCpu,
) {
    if upper {
        Shift::RightBytes.emit(code, 0, 8);
    }
    packed(code, Host::Xor, 1, 1);
    if signed {
        packed(
            code,
            match lane_bytes {
                1 => Host::GreaterByte,
                2 => Host::GreaterWord,
                4 => Host::GreaterDword,
                _ => unreachable!(),
            },
            1,
            0,
        );
    }
    packed(
        code,
        match lane_bytes {
            1 => Host::UnpackLowByte,
            2 => Host::UnpackLowWord,
            4 => Host::UnpackLowDword,
            _ => unreachable!(),
        },
        0,
        1,
    );
    logical_shift(code, 0, lane_bytes * 2, amount, true, target);
}

pub(super) fn emit(code: &mut Code, operand: &Operand, target: TargetCpu) {
    let Operand {
        kind,
        destination,
        source,
        lane_bytes,
        amount,
        bytes,
    } = *operand;
    match kind {
        Kind::Register {
            signed,
            round,
            saturating,
            counts,
        } => {
            if !round && !saturating && avx2_register_shift(code, operand, counts, signed, target) {
                return;
            }
            register_shift(
                code,
                operand,
                Some(counts),
                signed,
                !signed,
                round,
                saturating,
            );
            return;
        }
        Kind::SaturatingImmediate {
            signed_source,
            unsigned_result,
        } => {
            if bytes > lane_bytes {
                saturating_immediate_vector(code, operand, signed_source, unsigned_result, target);
                return;
            }
            register_shift(
                code,
                operand,
                None,
                signed_source,
                unsigned_result,
                false,
                true,
            );
            return;
        }
        _ => {}
    }
    if matches!(kind, Kind::Right { insert: true, .. }) && amount == lane_bytes * 8 {
        // SRI by the complete element width preserves every active old bit.
        // Keep a real load/store interval for precise source-map entries.
        read_vector(code, destination, 0);
        if bytes == 8 {
            packed(code, Host::MoveQword, 0, 0);
        }
        commit(code, destination, 0);
        return;
    }
    if let Kind::Right {
        signed,
        round,
        accumulate,
        insert: false,
    } = kind
    {
        // From the source where it is into the destination's home.
        let source = vector_source(code, source, 0);
        if bytes == 8 {
            rounded_right_from(code, 0, source, lane_bytes, amount, signed, round, target);
            if accumulate {
                let old = vector_source(code, destination, 1);
                let add = super::rearrange::add_or_subtract(lane_bytes, false);
                packed(code, add, 0, old);
            }
            let destination = vector_destination(code, destination, 0);
            finish_from(code, destination, 0, 8);
        } else if accumulate {
            rounded_right_from(code, 0, source, lane_bytes, amount, signed, round, target);
            let sum = vector_accumulator(code, destination, 1);
            let add = super::rearrange::add_or_subtract(lane_bytes, false);
            binary(code, add, sum, sum.xmm, 0, true, target);
        } else {
            let destination = vector_destination(code, destination, 0);
            rounded_right_from(
                code,
                destination.xmm,
                source,
                lane_bytes,
                amount,
                signed,
                round,
                target,
            );
            finish(code, destination);
        }
        return;
    }
    if let Kind::Narrow { round, upper } = kind {
        narrow(
            code,
            source,
            lane_bytes,
            amount,
            round,
            upper,
            destination,
            target,
        );
        return;
    }
    if !matches!(kind, Kind::SaturatingNarrow { .. }) {
        read_vector(code, source, 0);
    }
    match kind {
        Kind::Register { .. } | Kind::SaturatingImmediate { .. } | Kind::Narrow { .. } => {
            unreachable!()
        }
        Kind::SaturatingNarrow {
            signed_source,
            unsigned_result,
            round,
            upper,
        } => {
            saturating_narrow(
                code,
                operand,
                signed_source,
                unsigned_result,
                round,
                upper,
                target,
            );
            return;
        }
        Kind::Widen { signed, upper } => widen(code, lane_bytes, amount, signed, upper, target),
        Kind::Left { insert } => {
            logical_shift(code, 0, lane_bytes, amount, true, target);
            if insert && amount != 0 {
                read_vector(code, destination, 1);
                let preserve = (1u64 << amount) - 1; // amount is 0..lane_bits-1, including qword
                lane_mask(code, 2, lane_bytes, preserve);
                packed(code, Host::And, 1, 2);
                packed(code, Host::Or, 0, 1);
            }
        }
        Kind::Right {
            signed,
            round,
            accumulate,
            insert,
        } => {
            rounded_right(code, lane_bytes, amount, signed, round, target);
            if accumulate || insert {
                read_vector(code, destination, 1);
                if accumulate {
                    add(code, 0, 1, lane_bytes);
                } else {
                    let bits = lane_bytes * 8;
                    let preserve = (!(u128::from(u64::MAX) >> (64 - bits + amount))) as u64;
                    // A u128 shift allows bits = 64 and amount = 64 without
                    // Rust or x86 count masking; `lane_mask` keeps only this
                    // lane's bits.
                    lane_mask(code, 2, lane_bytes, preserve);
                    packed(code, Host::And, 1, 2);
                    packed(code, Host::Or, 0, 1);
                }
            }
        }
    }
    if bytes == 8 {
        packed(code, Host::MoveQword, 0, 0);
    }
    commit(code, destination, 0);
}

/// Constant counts allow packed SSE2 shifts. The general path stages wrapped
/// results in XMM3 and the originals in XMM6; the sign bits of XMM4's active
/// bytes set QC.
fn saturating_immediate_vector(
    code: &mut Code,
    operand: &Operand,
    signed_source: bool,
    unsigned_result: bool,
    target: TargetCpu,
) {
    let lane = operand.lane_bytes;
    read_vector(code, operand.source, 0);
    if operand.amount == 0 {
        if signed_source && unsigned_result {
            // SQSHLU clamps negative lanes to zero. Record their sign mask for
            // QC before PANDN consumes it; the recorder uses only ECX.
            super::saturating::sign_mask(code, 4, 0, lane);
            super::saturating::publish_qc(code, operand.bytes);
            packed(code, Host::AndNot, 4, 0);
            finish_immediate_vector(code, operand, 4);
        } else {
            // SQSHL/UQSHL zero cannot saturate, including inactive poison lanes.
            finish_immediate_vector(code, operand, 0);
        }
        return;
    }
    if operand.amount == 1 && lane <= 2 && signed_source != unsigned_result {
        // A shift by one is a saturating add of a value to itself, so no second
        // source copy is needed. Signed overflow shows as a sign change between
        // wrapped and saturated results; unsigned inputs saturate exactly when
        // the MSB is set, and replicating to signed words keeps low-byte MSBs
        // from setting QC.
        if !signed_source && lane == 2 {
            super::saturating::sign_mask(code, 4, 0, lane);
        } else {
            packed(code, Host::MoveAligned, 4, 0);
        }
        if signed_source {
            add(code, 4, 4, lane);
        }
        packed(
            code,
            match (signed_source, lane) {
                (true, 1) => Host::AddSignedSaturatingByte,
                (true, 2) => Host::AddSignedSaturatingWord,
                (false, 1) => Host::AddUnsignedSaturatingByte,
                (false, 2) => Host::AddUnsignedSaturatingWord,
                _ => unreachable!("reviewed doubling lane"),
            },
            0,
            0,
        );
        if signed_source {
            packed(code, Host::Xor, 4, 0);
        }
        super::saturating::publish_qc(code, operand.bytes);
        finish_immediate_vector(code, operand, 0);
        return;
    }
    packed(code, Host::MoveAligned, 6, 0);
    logical_shift(code, 0, lane, operand.amount, true, target);
    packed(code, Host::MoveAligned, 3, 0);
    right_shift(code, 0, lane, operand.amount, !unsigned_result, target);
    packed(code, Host::MoveAligned, 1, 6);
    super::equal(code, lane, crate::TargetCpu::Baseline);
    packed(code, Host::EqualDword, 4, 4);
    packed(code, Host::Xor, 4, 0); // reverse-shift mismatch means lost bits
    if signed_source {
        super::saturating::sign_mask(code, 5, 6, lane);
        if unsigned_result {
            // A negative source must clamp to zero even when a zero shift
            // leaves its bits unchanged (qword lanes included).
            packed(code, Host::Or, 4, 5);
            packed(code, Host::EqualDword, 1, 1);
            packed(code, Host::Xor, 5, 1); // positive bound=MAX, negative bound=0
        } else {
            packed(code, Host::EqualDword, 1, 1);
            if lane == 1 {
                // 0xffff >> 9 = 0x007f, in PACKUSWB's unsigned byte range.
                // One bound serves all sixteen lanes without immediate loads.
                Shift::RightWord.emit(code, 1, 9);
                packed(code, Host::PackUnsignedWordToByte, 1, 1);
            } else {
                logical_shift(code, 1, lane, 1, false, target);
            }
            packed(code, Host::Xor, 5, 1); // MAX for positive, MIN for negative
        }
    } else {
        packed(code, Host::EqualDword, 5, 5); // unsigned MAX
    }
    packed(code, Host::Xor, 5, 3);
    packed(code, Host::And, 5, 4);
    packed(code, Host::Xor, 3, 5);
    super::saturating::publish_qc(code, operand.bytes);
    finish_immediate_vector(code, operand, 3);
}

fn finish_immediate_vector(code: &mut Code, operand: &Operand, result: u8) {
    if operand.bytes == 8 {
        packed(code, Host::MoveQword, result, result);
    }
    commit(code, operand.destination, result);
}

/// Local near branches avoid assuming that every lane recipe fits in rel8.
#[derive(Clone, Copy)]
enum LaneBranch {
    Always,
    Nonnegative,
    Equal,
    Above,
    AboveOrEqual,
}
fn lane_branch(code: &mut Code, condition: LaneBranch) -> usize {
    let mut asm = code.asm();
    match condition {
        LaneBranch::Always => asm.jmp(),
        LaneBranch::Nonnegative => asm.jcc(Cond::Ns),
        LaneBranch::Equal => asm.jcc(Cond::E),
        LaneBranch::Above => asm.jcc(Cond::A),
        LaneBranch::AboveOrEqual => asm.jcc(Cond::Ae),
    }
    .offset()
}
fn resolve_lane_branch(code: &mut Code, patch: usize) {
    let relative = i32::try_from(code.len() - patch - 4).expect("lane recipe rel32");
    code.patch(patch, relative.to_le_bytes());
}
#[derive(Clone, Copy)]
enum LaneShift {
    Left,
    LogicalRight,
    ArithmeticRight,
}
impl LaneShift {
    /// RAX or R10, count CL, baseline 64-bit shift group.
    fn count(self, code: &mut Code, r10: bool) {
        let op = match self {
            Self::Left => ShiftOp::Shl,
            Self::LogicalRight => ShiftOp::Shr,
            Self::ArithmeticRight => ShiftOp::Sar,
        };
        code.asm()
            .shift_cl(op, Size::Qword, if r10 { R10 } else { Rax });
    }
}
fn normalize_lane(code: &mut Code, bits: u8, signed: bool) {
    if bits < 64 {
        let mut asm = code.asm();
        asm.shift_imm(ShiftOp::Shl, Size::Qword, Rax, 64 - bits);
        let right = if signed { ShiftOp::Sar } else { ShiftOp::Shr };
        asm.shift_imm(right, Size::Qword, Rax, 64 - bits);
    }
}
fn load_shift_lane(code: &mut Code, operand: &Operand, byte: u8, signed: bool) {
    let lane = guest(vector_offset(operand.source) + u32::from(byte));
    let mut asm = code.asm();
    match (operand.lane_bytes, signed) {
        (1, false) => asm.movzx(Size::Dword, Rax, Size::Byte, lane),
        (1, true) => asm.movsx(Size::Qword, Rax, Size::Byte, lane),
        (2, false) => asm.movzx(Size::Dword, Rax, Size::Word, lane),
        (2, true) => asm.movsx(Size::Qword, Rax, Size::Word, lane),
        (4, false) => asm.mov_load(Size::Dword, Rax, lane),
        (4, true) => asm.movsxd(Rax, lane),
        (8, _) => asm.mov_load(Size::Qword, Rax, lane),
        _ => unreachable!("allocated shift lane"),
    }
}
fn load_bound(code: &mut Code, bound: u64) {
    code.asm().mov_imm64(Rax, bound);
}
/// `xor eax, eax`.
fn zero_lane(code: &mut Code) {
    code.asm().alu(Alu::Xor, Size::Dword, Rax, Rax);
}
/// CL is 1 to 128. Rounding adds CF from the last discarded bit instead of
/// adding a bias that could overflow the qword.
fn register_right(code: &mut Code, signed: bool, round: bool) {
    code.asm().alu_imm8(Alu::Cmp, Size::Dword, Rcx, 64);
    let large = lane_branch(code, LaneBranch::AboveOrEqual);
    if signed {
        LaneShift::ArithmeticRight
    } else {
        LaneShift::LogicalRight
    }
    .count(code, false);
    if round {
        code.asm().alu_imm8(Alu::Adc, Size::Qword, Rax, 0);
    }
    let done = lane_branch(code, LaneBranch::Always);
    resolve_lane_branch(code, large);
    if signed && !round {
        // Sign fill.
        code.asm().shift_imm(ShiftOp::Sar, Size::Qword, Rax, 63);
    } else if !signed && round {
        // At exactly 64 the rounded result is the source top bit. Above 64
        // every unsigned 64-bit value rounds to zero.
        let beyond = lane_branch(code, LaneBranch::Above);
        code.asm().shift_imm(ShiftOp::Shr, Size::Qword, Rax, 63);
        let end = lane_branch(code, LaneBranch::Always);
        resolve_lane_branch(code, beyond);
        zero_lane(code);
        resolve_lane_branch(code, end);
    } else {
        zero_lane(code);
    }
    resolve_lane_branch(code, done);
}
fn register_left(
    code: &mut Code,
    bits: u8,
    signed_source: bool,
    unsigned_result: bool,
    saturating: bool,
) {
    let mut finished = Vec::new();
    let mut negative_mixed = None;
    if saturating {
        // RDX keeps the original.
        code.asm().mov(Size::Qword, Rdx, Rax);
    }
    if signed_source && unsigned_result {
        code.asm().test(Size::Qword, Rax, Rax);
        let nonnegative = lane_branch(code, LaneBranch::Nonnegative);
        negative_mixed = Some(lane_branch(code, LaneBranch::Always));
        resolve_lane_branch(code, nonnegative);
    }
    code.asm().alu_imm8(Alu::Cmp, Size::Dword, Rcx, bits as i8);
    let large = lane_branch(code, LaneBranch::AboveOrEqual);
    LaneShift::Left.count(code, false);
    if saturating {
        normalize_lane(code, bits, !unsigned_result);
        code.asm().mov(Size::Qword, R10, Rax);
        if unsigned_result {
            LaneShift::LogicalRight
        } else {
            LaneShift::ArithmeticRight
        }
        .count(code, true);
        code.asm().alu(Alu::Cmp, Size::Qword, R10, Rdx);
        finished.push(lane_branch(code, LaneBranch::Equal));
        let clamp = lane_branch(code, LaneBranch::Always);
        resolve_lane_branch(code, large);
        // Zero never saturates, even for huge counts.
        code.asm().test(Size::Qword, Rax, Rax);
        finished.push(lane_branch(code, LaneBranch::Equal));
        resolve_lane_branch(code, clamp);
        if let Some(patch) = negative_mixed {
            resolve_lane_branch(code, patch);
        }
        let maximum = if unsigned_result {
            u64::MAX >> (64 - bits)
        } else {
            (1u64 << (bits - 1)) - 1
        };
        load_bound(code, maximum);
        if signed_source {
            // The original sign.
            code.asm().test(Size::Qword, Rdx, Rdx);
            let positive = lane_branch(code, LaneBranch::Nonnegative);
            load_bound(
                code,
                if unsigned_result {
                    0
                } else {
                    0u64.wrapping_sub(1u64 << (bits - 1))
                },
            );
            resolve_lane_branch(code, positive);
        }
        // Sticky QC, other bits preserved.
        code.asm().alu_imm32(
            Alu::Or,
            Size::Dword,
            guest(core::mem::offset_of!(crate::GuestState, fpsr) as u32),
            crate::aarch64::simd::saturating::FPSR_QC as i32,
        );
    } else {
        finished.push(lane_branch(code, LaneBranch::Always));
        resolve_lane_branch(code, large);
        zero_lane(code);
    }
    for patch in finished {
        resolve_lane_branch(code, patch);
    }
}
/// SSHL and USHL of the extended lane in RAX by the signed count in ECX,
/// without a branch: both directions are computed and CMOVs pick one, so
/// the timing depends on neither the count nor the data (FEAT_DIT lists
/// both). A left count of 64 or more gives zero; a right count is clamped to
/// 63, which fills with the sign (SSHL) or, below -63, is replaced by zero
/// (USHL of a 64-bit lane). Narrower lanes are truncated by the caller.
fn register_plain(code: &mut Code, signed: bool) {
    const LEFT_LIMIT: i8 = 64;
    const RIGHT_LIMIT: u32 = 63;
    const RIGHT_ZERO: i8 = -64;
    let mut asm = code.asm();
    asm.mov(Size::Qword, R10, Rax);
    asm.shift_cl(ShiftOp::Shl, Size::Qword, R10);
    asm.alu(Alu::Xor, Size::Dword, Rdx, Rdx);
    asm.alu_imm8(Alu::Cmp, Size::Dword, Rcx, LEFT_LIMIT);
    asm.cmov(Cond::Ge, Size::Qword, R10, Rdx);
    asm.mov(Size::Dword, Rdx, Rcx);
    asm.unary(Unary::Neg, Size::Dword, Rdx);
    asm.mov_imm32(R11, RIGHT_LIMIT);
    asm.alu(Alu::Cmp, Size::Dword, Rdx, R11);
    asm.cmov(Cond::G, Size::Dword, Rdx, R11);
    // R11D keeps the signed count.
    asm.mov(Size::Dword, R11, Rcx);
    asm.mov(Size::Dword, Rcx, Rdx);
    if signed {
        LaneShift::ArithmeticRight
    } else {
        LaneShift::LogicalRight
    }
    .count(code, false);
    let mut asm = code.asm();
    if !signed {
        asm.alu(Alu::Xor, Size::Dword, Rdx, Rdx);
        asm.alu_imm8(Alu::Cmp, Size::Dword, R11, RIGHT_ZERO);
        asm.cmov(Cond::Le, Size::Qword, Rax, Rdx);
    }
    asm.test(Size::Dword, R11, R11);
    asm.cmov(Cond::Ns, Size::Qword, Rax, R10);
}

/// SSHL/USHL consume only each count lane's signed low byte. AVX2
/// logical shifts zero over-wide counts; arithmetic right shifts sign-fill.
fn avx2_register_shift(
    code: &mut Code,
    operand: &Operand,
    counts: u8,
    signed: bool,
    target: TargetCpu,
) -> bool {
    // Narrow and qword recipes lengthen dependent shifts, so automatic
    // selection uses full dword vectors. See the compiler ISA research record.
    if operand.lane_bytes != 4 || operand.bytes != 16 {
        return false;
    }
    emit_avx2_register_shift_candidate(code, operand, counts, signed, target)
}

fn emit_avx2_register_shift_candidate(
    code: &mut Code,
    operand: &Operand,
    counts: u8,
    signed: bool,
    target: TargetCpu,
) -> bool {
    use crate::x86_64::isa::VexShiftOpcode;
    if !target.supports(TargetCpu::Avx2)
        || target.data_independent_timing()
        || !matches!(operand.lane_bytes, 4 | 8)
        || !matches!(operand.bytes, 8 | 16)
    {
        return false;
    }
    crate::x86_64::isa::clear_upper_vector_state(code, target);
    read_vector(code, operand.source, 0);
    read_vector(code, counts, 1);
    Shift::LeftDword.emit(code, 1, 24);
    Shift::ArithmeticDword.emit(code, 1, 24);
    let qword = operand.lane_bytes == 8;
    if qword {
        // Select the low dword of each qword, then sign-extend to qwords.
        shuffle(code, 1, 1, LOW_DWORDS_OF_QWORDS);
        packed(code, Host::Xor, 2, 2);
        packed(code, Host::GreaterDword, 2, 1);
        packed(code, Host::UnpackLowDword, 1, 2);
    }
    packed(code, Host::Xor, 2, 2);
    packed(
        code,
        if qword {
            Host::SubQword
        } else {
            Host::SubDword
        },
        2,
        1,
    );
    let left = if qword {
        VexShiftOpcode::LeftQword
    } else {
        VexShiftOpcode::LeftDword
    };
    left.register(code, 3, 0, 1, target);
    if qword && signed {
        // (x XOR sign) >> count XOR sign gives arithmetic qword shifts,
        // including counts of 64 or more, which AVX2 lacks directly.
        packed(code, Host::MoveAligned, 4, 0);
        Shift::ArithmeticDword.emit(code, 4, 31);
        shuffle(code, 4, 4, 0xf5);
        packed(code, Host::Xor, 0, 4);
        VexShiftOpcode::RightQword.register(code, 0, 0, 2, target);
        packed(code, Host::Xor, 0, 4);
    } else {
        let right = if qword {
            VexShiftOpcode::RightQword
        } else if signed {
            VexShiftOpcode::ArithmeticDword
        } else {
            VexShiftOpcode::RightDword
        };
        right.register(code, 0, 0, 2, target);
    }
    if signed {
        // Positive counts use only the left result; a negated positive
        // count otherwise makes an arithmetic right shift sign-fill.
        Shift::ArithmeticDword.emit(code, 1, 31);
        packed(code, Host::And, 0, 1);
    }
    packed(code, Host::Or, 0, 3);
    finish_immediate_vector(code, operand, 0);
    true
}

fn register_shift(
    code: &mut Code,
    operand: &Operand,
    counts: Option<u8>,
    signed_source: bool,
    unsigned_result: bool,
    round: bool,
    saturating: bool,
) {
    // R8/R9 hold staged result halves. Guest sources remain unchanged until both
    // have been consumed, including d=n=m and scalar inactive-lane poisoning.
    let mut asm = code.asm();
    asm.alu(Alu::Xor, Size::Dword, R8, R8);
    asm.alu(Alu::Xor, Size::Dword, R9, R9);
    let bits = operand.lane_bytes * 8;
    for byte in (0..operand.bytes).step_by(usize::from(operand.lane_bytes)) {
        load_shift_lane(code, operand, byte, signed_source);
        if let Some(counts) = counts {
            // The count lane.
            code.asm().movsx(
                Size::Dword,
                Rcx,
                Size::Byte,
                guest(vector_offset(counts) + u32::from(byte)),
            );
            if !round && !saturating {
                register_plain(code, signed_source);
            } else {
                code.asm().test(Size::Dword, Rcx, Rcx);
                let left = lane_branch(code, LaneBranch::Nonnegative);
                // A signed byte -128 becomes +128.
                code.asm().unary(Unary::Neg, Size::Dword, Rcx);
                register_right(code, signed_source, round);
                let merged = lane_branch(code, LaneBranch::Always);
                resolve_lane_branch(code, left);
                register_left(code, bits, signed_source, unsigned_result, saturating);
                resolve_lane_branch(code, merged);
            }
        } else {
            code.asm().mov_imm32(Rcx, u32::from(operand.amount));
            register_left(code, bits, signed_source, unsigned_result, true);
        }
        normalize_lane(code, bits, false);
        let mut asm = code.asm();
        if byte % 8 != 0 {
            asm.shift_imm(ShiftOp::Shl, Size::Qword, Rax, byte % 8 * 8);
        }
        asm.alu(Alu::Or, Size::Qword, if byte < 8 { R8 } else { R9 }, Rax);
    }
    // Commit both staged halves.
    let mut asm = code.asm();
    asm.mov_store(Size::Qword, guest(vector_offset(operand.destination)), R8);
    asm.mov_store(
        Size::Qword,
        guest(vector_offset(operand.destination) + 8),
        R9,
    );
}

#[cfg(test)]
mod tests;
