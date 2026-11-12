//! Bitfield moves using shifts and masks. Guest NZCV is untouched.
use super::{Register, Shift, Width, load, right_operand, store};
use crate::aarch64::bitfield::{Kind, Operand};
use crate::x86_64::asm::{
    Alu, Code,
    Gpr::{Rax, Rcx, Rdx},
    Rm, ShiftOp, Size,
};

/// SXTB, UXTH, SXTW and the other extensions of a field at bit zero as one
/// MOVSX, MOVZX or MOV of (e/r)ax; false for any other field width.
fn extend(code: &mut Code, width: Width, signed: bool, field_width: u8) -> bool {
    let wide = width == Width::X64;
    let narrow = if field_width == 8 {
        Size::Byte
    } else {
        Size::Word
    };
    let mut asm = code.asm();
    match (field_width, signed) {
        (8 | 16, false) => asm.movzx(Size::Dword, Rax, narrow, Rax),
        (8 | 16, true) => asm.movsx(Size::of(wide), Rax, narrow, Rax),
        (32, false) if wide => asm.mov(Size::Dword, Rax, Rax),
        (32, true) if wide => asm.movsxd(Rax, Rax),
        _ => return false,
    }
    true
}

/// `op (e/r)ax,amount`, or nothing for a zero amount.
fn shift(code: &mut Code, width: Width, op: ShiftOp, amount: u8) {
    if amount != 0 {
        code.asm()
            .shift_imm(op, Size::of(width == Width::X64), Rax, amount);
    }
}

/// BFXIL Xd, Xn, #0, #32 can write its low half directly.
pub(super) fn is_low_dword_insert(operand: &Operand) -> bool {
    operand.width == Width::X64
        && operand.kind == Kind::Insert
        && operand.rotation == 0
        && operand.mask_end == 31
}

fn insert_low_dword(code: &mut Code, destination: Register, source: Register) {
    let mut asm = code.asm();
    if destination == source {
        // Keep the source row even though the architectural value is unchanged.
        asm.nop();
        return;
    }
    let destination = asm
        .guest(destination)
        .expect("discarded destination handled");
    let source = asm.guest(source);
    match destination {
        Rm::Mem(destination) => match source {
            Some(Rm::Reg(source)) => asm.mov_store(Size::Dword, destination, source),
            Some(source) => {
                asm.mov_load(Size::Dword, Rax, source);
                asm.mov_store(Size::Dword, destination, Rax);
            }
            None => asm.mov_rm_imm32(Size::Dword, destination, 0),
        },
        Rm::Reg(destination) => {
            // XOR the low-half delta into the full destination. EAX
            // zero-extends, so the high half survives the final XOR.
            asm.mov_load(Size::Dword, Rax, destination);
            if let Some(source) = source {
                asm.alu_load(Alu::Xor, Size::Dword, Rax, source);
            }
            asm.alu(Alu::Xor, Size::Qword, destination, Rax);
        }
        Rm::Xmm(_) => unreachable!("guest integer storage"),
    }
}

pub(super) fn emit(code: &mut Code, operand: &Operand) {
    if operand.destination == Register::Zero {
        // Discarding a result must still own a nonempty source-map interval.
        code.asm().nop();
        return;
    }
    if is_low_dword_insert(operand) {
        insert_low_dword(code, operand.destination, operand.source);
        return;
    }
    let bits = match operand.width {
        Width::W32 => 32,
        Width::X64 => 64,
    };
    debug_assert!(operand.rotation < bits && operand.mask_end < bits);
    let (source_shift, destination_shift, field_width) = if operand.rotation <= operand.mask_end {
        (operand.rotation, 0, operand.mask_end - operand.rotation + 1)
    } else {
        (0, bits - operand.rotation, operand.mask_end + 1)
    };

    load(code, operand.source);
    // A field from bit zero is either extended in place or, when it reaches
    // the top bit (LSL and the widest SBFIZ and UBFIZ), shifted there alone:
    // the bits above it leave either way.
    let from_bit_zero = operand.kind != Kind::Insert && source_shift == 0;
    if from_bit_zero
        && destination_shift == 0
        && extend(
            code,
            operand.width,
            operand.kind == Kind::Signed,
            field_width,
        )
    {
        store(code, operand.destination);
        return;
    }
    if from_bit_zero && destination_shift != 0 && destination_shift + field_width == bits {
        shift(code, operand.width, ShiftOp::Shl, destination_shift);
        store(code, operand.destination);
        return;
    }
    if operand.width == Width::W32 && field_width == bits {
        // A 32-bit shift clears the upper half, so only the full-width MOV
        // alias, which emits no shift, needs the zero-extending move.
        code.asm().mov(Size::Dword, Rax, Rax);
    }
    // Move the source field to the operand's MSB, then extract it. Arithmetic
    // extraction replicates source[imms], including wrapping SBFIZ aliases.
    // Zero shifts are omitted; emitted amounts are 1 through bits-1 because
    // x86 masks a shift by bits to zero.
    shift(
        code,
        operand.width,
        ShiftOp::Shl,
        bits - source_shift - field_width,
    );
    shift(
        code,
        operand.width,
        if operand.kind == Kind::Signed {
            ShiftOp::Sar
        } else {
            ShiftOp::Shr
        },
        bits - field_width,
    );
    shift(code, operand.width, ShiftOp::Shl, destination_shift);

    if operand.kind == Kind::Insert {
        let field_mask = (u64::MAX >> (64 - field_width)) << destination_shift;
        // Nothing has been written yet, so source and destination may alias.
        // RCX receives the old destination after the source is staged.
        right_operand(code, operand.destination, Width::X64, Shift::Lsl, 0);
        let mut asm = code.asm();
        if operand.width == Width::X64 {
            asm.mov_imm64(Rdx, !field_mask);
            asm.alu(Alu::And, Size::Qword, Rcx, Rdx);
        } else {
            asm.alu_imm32(Alu::And, Size::Dword, Rcx, !(field_mask as u32) as i32);
        }
        asm.alu(Alu::Or, Size::of(operand.width == Width::X64), Rax, Rcx);
    }
    store(code, operand.destination);
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec::Vec;

    fn code(width: Width, kind: Kind, rotation: u8, mask_end: u8) -> Vec<u8> {
        let mut result = Code::new();
        emit(
            &mut result,
            &Operand {
                width,
                kind,
                destination: Register::X(2),
                source: Register::X(1),
                rotation,
                mask_end,
            },
        );
        result.into_bytes()
    }

    #[test]
    fn low_dword_insert_updates_each_kind_of_destination_directly() {
        use crate::aarch64::Operation;
        use crate::x86_64::{MemoryChecks, Rax, rax_after};

        // A sampled macOS boot instruction: BFXIL X17, X1, #0, #32.
        let operand = crate::aarch64::bitfield::decode(0xb340_7c31).unwrap();
        for pinned in [false, true] {
            for dit in [false, true] {
                let mut code = Code::new();
                code.set_pinned(pinned);
                code.set_constant_time(dit);
                emit(&mut code, &operand);
                if pinned {
                    assert_eq!(code, [0x41, 0x8b, 0xc5, 0x33, 0xc5, 0x49, 0x31, 0xc5]);
                } else {
                    assert_eq!(code, [0x8b, 0x87, 8, 0, 0, 0, 0x89, 0x87, 136, 0, 0, 0]);
                }
                assert_eq!(
                    rax_after(&Operation::Bitfield(operand), &code, MemoryChecks::DEFAULT),
                    Rax::Unknown
                );
                for (destination, source) in [
                    (operand.destination, operand.destination),
                    (Register::Zero, operand.source),
                ] {
                    code.clear();
                    code.set_pinned(pinned);
                    emit(
                        &mut code,
                        &Operand {
                            destination,
                            source,
                            ..operand
                        },
                    );
                    assert_eq!(code, [0x90]);
                }
                code.clear();
                code.set_pinned(pinned);
                emit(
                    &mut code,
                    &Operand {
                        source: Register::Zero,
                        ..operand
                    },
                );
                if pinned {
                    assert_eq!(code, [0x41, 0x8b, 0xc5, 0x49, 0x31, 0xc5]);
                } else {
                    assert_eq!(code, [0xc7, 0x87, 136, 0, 0, 0, 0, 0, 0, 0]);
                }
            }
        }
    }

    #[test]
    fn low_dword_insert_leaves_a_following_full_width_read() {
        use crate::{GuestAddress, Options, TargetCpu, block::BlockProfile};

        // BFXIL X17,X1,#0,#32; ADD X14,X17,#1.
        let source: Vec<_> = [0xb340_7c31u32, 0x9100_062e]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let pc = GuestAddress(0xffff_fe00_3a24_52bc);
        let options = Options::new(BlockProfile::ORDINARY, TargetCpu::Baseline);
        let block = crate::compile(&source, pc, &options).unwrap();
        assert_eq!(
            block.native.source_map,
            [(pc, 0), (GuestAddress(pc.0 + 4), 12)]
        );
        assert!(block.native.x86_64_code[12..].starts_with(&[0x48, 0x8b, 0x87, 136, 0, 0, 0]));
    }

    #[test]
    fn full_width_extract_avoids_masked_x86_shifts_and_clears_w_upper_bits() {
        for kind in [Kind::Signed, Kind::Unsigned] {
            assert_eq!(
                code(Width::X64, kind, 0, 63),
                [0x48, 0x8b, 0x87, 8, 0, 0, 0, 0x48, 0x89, 0x87, 16, 0, 0, 0]
            );
            assert_eq!(
                code(Width::W32, kind, 0, 31),
                [
                    0x48, 0x8b, 0x87, 8, 0, 0, 0, 0x89, 0xc0, 0x48, 0x89, 0x87, 16, 0, 0, 0,
                ]
            );
        }
    }

    #[test]
    fn wrapping_signed_field_extends_before_placing_the_low_zero_bits() {
        let signed = code(Width::X64, Kind::Signed, 63, 0);
        assert_eq!(
            &signed[7..signed.len() - 7],
            [
                0x48, 0xc1, 0xe0, 63, 0x48, 0xc1, 0xf8, 63, 0x48, 0xc1, 0xe0, 1
            ]
        );
        let unsigned = code(Width::W32, Kind::Unsigned, 31, 0);
        assert_eq!(
            &unsigned[7..unsigned.len() - 7],
            [0xc1, 0xe0, 31, 0xc1, 0xe8, 31, 0xc1, 0xe0, 1]
        );
        let extraction = code(Width::X64, Kind::Unsigned, 8, 15);
        assert_eq!(
            &extraction[7..extraction.len() - 7],
            [0x48, 0xc1, 0xe0, 48, 0x48, 0xc1, 0xe8, 56]
        );
    }

    #[test]
    fn extensions_and_top_reaching_fields_take_one_instruction() {
        // GNU as: MOVZX EAX, AL; MOVSX RAX, AX; MOVSXD RAX, EAX; MOV EAX,
        // EAX; MOVSX EAX, AL; SHL RAX, 3; SHL EAX, 5.
        for (width, kind, rotation, mask_end, expected) in [
            (Width::W32, Kind::Unsigned, 0, 7, &[0x0f, 0xb6, 0xc0][..]),
            (Width::X64, Kind::Signed, 0, 15, &[0x48, 0x0f, 0xbf, 0xc0]),
            (Width::X64, Kind::Signed, 0, 31, &[0x48, 0x63, 0xc0]),
            (Width::X64, Kind::Unsigned, 0, 31, &[0x89, 0xc0]),
            (Width::W32, Kind::Signed, 0, 7, &[0x0f, 0xbe, 0xc0]),
            (Width::X64, Kind::Unsigned, 61, 60, &[0x48, 0xc1, 0xe0, 3]),
            (Width::W32, Kind::Signed, 27, 26, &[0xc1, 0xe0, 5]),
        ] {
            let emitted = code(width, kind, rotation, mask_end);
            assert_eq!(&emitted[7..emitted.len() - 7], expected);
        }
    }

    #[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
    #[test]
    fn every_extract_field_matches_the_arm_definition() {
        use crate::x86_64::asm::vector_extensions::execution::Native;
        use crate::{GuestState, context::ExecutionContext};
        let mut random = 0x2545_f491_4f6c_dd1d_u64;
        for (width, bits) in [(Width::W32, 32u8), (Width::X64, 64)] {
            for kind in [Kind::Signed, Kind::Unsigned] {
                for rotation in 0..bits {
                    for mask_end in 0..bits {
                        let mut function = code(width, kind, rotation, mask_end);
                        function.push(0xc3); // RET
                        let native = Native::new(&function);
                        for _ in 0..4 {
                            random ^= random << 13;
                            random ^= random >> 7;
                            random ^= random << 17;
                            let source = if bits == 32 {
                                random & u64::from(u32::MAX)
                            } else {
                                random
                            };
                            // UBFM/SBFM: the source field ends at bit mask_end;
                            // it moves right by rotation, or left by
                            // bits - rotation when it starts below rotation.
                            let (low, high, left) = if rotation <= mask_end {
                                (rotation, mask_end, 0)
                            } else {
                                (0, mask_end, bits - rotation)
                            };
                            let field_bits = u32::from(high - low + 1);
                            let field = source >> low & (u64::MAX >> (64 - field_bits));
                            let extended = if kind == Kind::Signed {
                                ((field << (64 - field_bits)) as i64 >> (64 - field_bits)) as u64
                            } else {
                                field
                            };
                            let mut expected = extended << left;
                            if bits == 32 {
                                expected &= u64::from(u32::MAX);
                            }
                            let mut state = GuestState::default();
                            state.x[1] = random;
                            native.run(&mut state, &mut ExecutionContext::default());
                            assert_eq!(
                                state.x[2], expected,
                                "{width:?} {kind:?} {rotation} {mask_end} {source:#x}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn insert_masks_preserve_only_bits_outside_the_selected_destination_field() {
        for (rotation, mask_end, keep_mask) in [
            (8, 15, 0xffffffffffffff00u64),
            (60, 7, 0xfffffffffffff00f),
            (0, 63, 0),
            (1, 0, 0x7fffffffffffffff),
        ] {
            let emitted = code(Width::X64, Kind::Insert, rotation, mask_end);
            let mask_offset = emitted
                .windows(2)
                .position(|bytes| bytes == [0x48, 0xba])
                .unwrap()
                + 2;
            assert_eq!(
                &emitted[mask_offset..mask_offset + 8],
                keep_mask.to_le_bytes()
            );
        }
        let emitted = code(Width::W32, Kind::Insert, 28, 7);
        let mask_offset = emitted
            .windows(2)
            .position(|bytes| bytes == [0x81, 0xe1])
            .unwrap()
            + 2;
        assert_eq!(
            &emitted[mask_offset..mask_offset + 4],
            0xfffff00fu32.to_le_bytes()
        );
        assert_eq!(&emitted[mask_offset + 4..mask_offset + 6], [0x09, 0xc8]);
    }

    #[test]
    fn aliased_insert_stages_both_original_operands_before_the_only_state_store() {
        let mut emitted = Code::new();
        emit(
            &mut emitted,
            &Operand {
                width: Width::X64,
                kind: Kind::Insert,
                destination: Register::X(30),
                source: Register::X(30),
                rotation: 60,
                mask_end: 7,
            },
        );
        assert_eq!(&emitted[..7], [0x48, 0x8b, 0x87, 240, 0, 0, 0]);
        let destination_load = [0x48, 0x8b, 0x8f, 240, 0, 0, 0];
        assert!(emitted.windows(7).any(|bytes| bytes == destination_load));
        assert_eq!(
            &emitted[emitted.len() - 7..],
            [0x48, 0x89, 0x87, 240, 0, 0, 0]
        );
        assert_eq!(
            emitted
                .windows(3)
                .filter(|bytes| *bytes == [0x48, 0x89, 0x87])
                .count(),
            1
        );
    }

    #[test]
    fn zero_register_reads_zero_and_discarded_results_keep_source_map_intervals() {
        for kind in [Kind::Signed, Kind::Insert, Kind::Unsigned] {
            let mut operand = Operand {
                width: Width::X64,
                kind,
                destination: Register::X(2),
                source: Register::Zero,
                rotation: 0,
                mask_end: 63,
            };
            let mut emitted = Code::new();
            emit(&mut emitted, &operand);
            assert_eq!(&emitted[..2], [0x31, 0xc0]);
            operand.destination = Register::Zero;
            emitted.clear();
            emit(&mut emitted, &operand);
            assert_eq!(emitted, [0x90]);
        }
        let source: Vec<_> = [0xd340001fu32, 0xd65f03c0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let leaf = crate::testing::leaf(&source, crate::GuestAddress(0x100000)).unwrap();
        let block = crate::testing::block(&source, crate::GuestAddress(0x100000)).unwrap();
        for native in [&leaf, &block.native] {
            assert_eq!(
                native.source_map,
                [
                    (crate::GuestAddress(0x100000), 0),
                    (crate::GuestAddress(0x100004), 1)
                ]
            );
        }
    }
}
