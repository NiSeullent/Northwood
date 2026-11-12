//! Overflow-safe halving arithmetic and narrowing/widening helpers.
use super::rearrange::{add_or_subtract, lane_constant};
use super::{
    Host, binary, commit, packed,
    products::extend,
    read_vector,
    rearrange::constant,
    shifts::{pack_truncated, right_shift, rounded_right},
    three::three,
    vector_destination, vector_source,
};
use crate::x86_64::asm::Code;
use crate::{
    TargetCpu,
    aarch64::simd::halving::{Kind, Operand},
    x86_64::isa::ShiftOpcode as Shift,
};

fn half(code: &mut Code, lane: u8, signed: bool, subtract: bool, round: bool, target: TargetCpu) {
    if round && lane <= 2 {
        if signed {
            lane_constant(code, 2, lane, 1u64 << (lane * 8 - 1));
            packed(code, Host::Xor, 0, 2);
            packed(code, Host::Xor, 1, 2);
        }
        packed(
            code,
            if lane == 1 {
                Host::AverageByte
            } else {
                Host::AverageWord
            },
            0,
            1,
        );
        if signed {
            packed(code, Host::Xor, 0, 2);
        }
    } else if subtract {
        // floor((a-b)/2) = floor(a/2)-floor(b/2)-((~a & b) & 1).
        three(code, Host::AndNot, 2, 0, 1, target);
        lane_constant(code, 3, lane, 1);
        packed(code, Host::And, 2, 3);
        right_shift(code, 0, lane, 1, signed, target);
        right_shift(code, 1, lane, 1, signed, target);
        packed(code, add_or_subtract(lane, true), 0, 1);
        packed(code, add_or_subtract(lane, true), 0, 2);
    } else {
        // floor average: (a&b)+((a^b)>>1). Ceil average: (a|b)-((a^b)>>1).
        let both = if round { Host::Or } else { Host::And };
        three(code, both, 2, 0, 1, target);
        packed(code, Host::Xor, 0, 1);
        right_shift(code, 0, lane, 1, signed, target);
        if round && super::three::vex(target) {
            three(code, add_or_subtract(lane, true), 0, 2, 0, target);
        } else if round {
            packed(code, add_or_subtract(lane, true), 2, 0);
            packed(code, Host::MoveAligned, 0, 2);
        } else {
            packed(code, add_or_subtract(lane, false), 0, 2);
        }
    }
}
/// UHADD/SHADD/URHADD (and SRHADD of S and D lanes), 16 bytes, from the
/// sources where they are into the destination: PAVGB/PAVGW, else
/// (a&b)+((a^b)>>1) or (a|b)-((a^b)>>1). Clobbers XMM0-XMM2.
fn average(code: &mut Code, operand: &Operand, signed: bool, round: bool, target: TargetCpu) {
    let lane = operand.lane_bytes;
    let left = vector_source(code, operand.left, 0);
    let right = vector_source(code, operand.right, 1);
    if round && lane <= 2 {
        let opcode = if lane == 1 {
            Host::AverageByte
        } else {
            Host::AverageWord
        };
        let destination = vector_destination(code, operand.destination, 0);
        binary(code, opcode, destination, left, right, true, target);
        return;
    }
    let both = if round { Host::Or } else { Host::And };
    three(code, both, 2, left, right, target);
    three(code, Host::Xor, 0, left, right, target);
    right_shift(code, 0, lane, 1, signed, target);
    // XMM1, free now, is the scratch register of the subtraction.
    let destination = vector_destination(code, operand.destination, u8::from(round));
    if round {
        let subtract = add_or_subtract(lane, true);
        binary(code, subtract, destination, 2, 0, false, target);
    } else {
        binary(
            code,
            add_or_subtract(lane, false),
            destination,
            0,
            2,
            true,
            target,
        );
    }
}

fn pair_long(code: &mut Code, lane: u8, signed: bool, target: TargetCpu) {
    if lane == 1 && !signed && target >= TargetCpu::Ssse3 {
        constant(code, 1, [1; 16]);
        Host::MultiplyAddUnsignedBytes.register(code, 0, 1, target);
        return;
    }
    if lane == 2 && signed {
        lane_constant(code, 1, 2, 1);
        packed(code, Host::MultiplyAddWords, 0, 1);
        return;
    }
    if lane == 4 && target >= TargetCpu::Sse41 {
        packed(code, Host::MoveAligned, 2, 0);
        Shift::RightBytes.emit(code, 2, 8);
        let opcode = if signed {
            Host::SignExtendDwordToQword
        } else {
            Host::ZeroExtendDwordToQword
        };
        opcode.register(code, 0, 0, target);
        opcode.register(code, 2, 2, target);
        packed(code, Host::MoveAligned, 1, 0);
        packed(code, Host::UnpackLowQword, 0, 2);
        packed(code, Host::UnpackHighQword, 1, 2);
    } else {
        packed(code, Host::MoveAligned, 1, 0);
        if signed {
            match lane {
                1 => {
                    Shift::LeftWord.emit(code, 0, 8);
                    Shift::ArithmeticWord.emit(code, 0, 8);
                    Shift::ArithmeticWord.emit(code, 1, 8);
                }
                4 => {
                    Shift::LeftQword.emit(code, 0, 32);
                    right_shift(code, 0, 8, 32, true, target);
                    right_shift(code, 1, 8, 32, true, target);
                }
                _ => unreachable!("signed halfword path uses PMADDWD"),
            }
        } else {
            lane_constant(code, 2, lane * 2, (1u64 << (lane * 8)) - 1);
            packed(code, Host::And, 0, 2);
            match lane {
                1 => Shift::RightWord,
                2 => Shift::RightDword,
                4 => Shift::RightQword,
                _ => unreachable!(),
            }
            .emit(code, 1, lane * 8);
        }
    }
    packed(code, add_or_subtract(lane * 2, false), 0, 1);
}

pub(super) fn emit(code: &mut Code, operand: &Operand, target: TargetCpu) {
    let Operand {
        kind,
        destination,
        left,
        right,
        lane_bytes: lane,
        bytes,
    } = *operand;
    if let Kind::Half {
        signed,
        subtract: false,
        round,
    } = kind
        && bytes == 16
        && !(round && lane <= 2 && signed)
    {
        average(code, operand, signed, round, target);
        return;
    }
    read_vector(code, left, 0);
    match kind {
        Kind::Half {
            signed,
            subtract,
            round,
        } => {
            read_vector(code, right, 1);
            half(code, lane, signed, subtract, round, target);
        }
        Kind::HighNarrow {
            subtract,
            round,
            upper,
        } => {
            read_vector(code, right, 1);
            packed(code, add_or_subtract(lane * 2, subtract), 0, 1);
            rounded_right(code, lane * 2, lane * 8, false, round, target);
            pack_truncated(code, lane, upper, destination, target);
            return;
        }
        Kind::Truncate { upper } => {
            pack_truncated(code, lane, upper, destination, target);
            return;
        }
        Kind::ShiftLong { upper } => {
            extend(code, 0, lane, upper, false, target);
            match lane {
                1 => Shift::LeftWord,
                2 => Shift::LeftDword,
                4 => Shift::LeftQword,
                _ => unreachable!(),
            }
            .emit(code, 0, lane * 8);
        }
        Kind::PairLong { signed, accumulate } => {
            pair_long(code, lane, signed, target);
            if accumulate {
                read_vector(code, destination, 1);
                packed(code, add_or_subtract(lane * 2, false), 0, 1);
            }
        }
    }
    if bytes == 8 {
        packed(code, Host::MoveQword, 0, 0);
    }
    commit(code, destination, 0);
}
