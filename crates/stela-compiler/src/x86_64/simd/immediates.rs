//! Immediate construction and lane moves.
use super::{Host, commit, packed, read_vector, rearrange};
use crate::x86_64::asm::Code;
use crate::{
    TargetCpu,
    aarch64::simd::immediates::{ConstantOperation, Operand},
};

pub(super) fn emit(code: &mut Code, operand: &Operand, target: TargetCpu) {
    let Operand::Constant {
        destination,
        value,
        bytes,
        operation,
    } = *operand
    else {
        if let Operand::LaneMove(ref lane) = *operand {
            rearrange::emit(code, lane, target, None);
        }
        return;
    };
    if bytes == 16 {
        let limb = value.to_le_bytes();
        rearrange::constant(code, 0, core::array::from_fn(|i| limb[i % limb.len()]));
    } else if value == 0 {
        packed(code, Host::Xor, 0, 0);
    } else {
        // The expanded low 64 bits.
        code.asm().mov_imm64(crate::x86_64::asm::Gpr::Rcx, value);
        packed(code, Host::MoveGprQword, 0, 1);
    }
    match operation {
        ConstantOperation::Move => {}
        ConstantOperation::Or => {
            read_vector(code, destination, 1);
            packed(code, Host::Or, 0, 1);
        }
        ConstantOperation::BitClear => {
            read_vector(code, destination, 1);
            packed(code, Host::AndNot, 0, 1);
        }
    }
    if operation != ConstantOperation::Move && bytes == 8 {
        packed(code, Host::MoveQword, 0, 0);
    }
    commit(code, destination, 0);
}
