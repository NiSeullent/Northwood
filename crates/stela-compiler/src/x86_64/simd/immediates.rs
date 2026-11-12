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
    let dword = value as u32;
    let bytes16: [u8; 16] = core::array::from_fn(|i| value.to_le_bytes()[i % 8]);
    if bytes == 16 && (operation != ConstantOperation::BitClear || super::three::vex(target)) {
        // The constant straight into the destination's home, or ORR or BIC
        // into it.
        let destination = match operation {
            ConstantOperation::Move => super::vector_destination(code, destination, 0),
            _ => super::vector_accumulator(code, destination, 1),
        };
        let xmm = if operation == ConstantOperation::Move {
            destination.xmm
        } else {
            0
        };
        if value == 0 || rearrange::ones_constant(&bytes16) {
            rearrange::constant(code, xmm, bytes16);
        } else if super::three::broadcast_general_supported(target) {
            let periodic = value == u64::from(dword) * 0x1_0000_0001;
            let mut asm = code.asm();
            if periodic {
                asm.mov_imm32(crate::x86_64::asm::Gpr::Rcx, dword);
            } else {
                asm.mov_imm64(crate::x86_64::asm::Gpr::Rcx, value);
            }
            super::three::broadcast_general(code, xmm, crate::x86_64::asm::Gpr::Rcx, !periodic);
        } else {
            code.asm().mov_imm64(crate::x86_64::asm::Gpr::Rcx, value);
            packed(code, Host::MoveGprQword, xmm, 1);
            packed(code, Host::UnpackLowQword, xmm, xmm);
        }
        match operation {
            ConstantOperation::Move => super::finish(code, destination),
            ConstantOperation::Or => {
                let xmm = destination.xmm;
                super::binary(code, Host::Or, destination, xmm, 0, true, target);
            }
            ConstantOperation::BitClear => {
                // destination = ~constant & destination: VPANDN in place.
                let xmm = destination.xmm;
                super::three::three(code, Host::AndNot, xmm, 0, xmm, target);
                super::finish(code, destination);
            }
        }
        return;
    }
    if bytes == 16 && (value == 0 || rearrange::ones_constant(&bytes16)) {
        // PXOR, or PCMPEQD and at most one shift.
        rearrange::constant(code, 0, bytes16);
    } else if bytes == 16 && super::three::broadcast_general_supported(target) {
        // One immediate move and one EVEX broadcast; a dword-periodic value
        // takes the short MOV r32 form.
        let periodic = value == u64::from(dword) * 0x1_0000_0001;
        let mut asm = code.asm();
        if periodic {
            asm.mov_imm32(crate::x86_64::asm::Gpr::Rcx, dword);
        } else {
            asm.mov_imm64(crate::x86_64::asm::Gpr::Rcx, value);
        }
        super::three::broadcast_general(code, 0, crate::x86_64::asm::Gpr::Rcx, !periodic);
    } else if value == 0 {
        packed(code, Host::Xor, 0, 0);
    } else {
        // The expanded low 64 bits.
        code.asm().mov_imm64(crate::x86_64::asm::Gpr::Rcx, value);
        packed(code, Host::MoveGprQword, 0, 1);
        if bytes == 16 {
            packed(code, Host::UnpackLowQword, 0, 0);
        }
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
