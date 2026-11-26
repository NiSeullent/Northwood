//! Lane moves and permutes. All inputs are read before the destination is written.
use super::{Host, commit, packed, packed_imm, read_vector, vector_offset};
use crate::x86_64::asm::{Alu, Code, Cond, Disp, Gpr, Mem, ShiftOp, Size, vector_extensions};
use crate::x86_64::isa::ShiftOpcode;
use crate::x86_64::simd::encode::guest;
use crate::{
    GuestAddress, TargetCpu,
    aarch64::{
        Instruction, Operation,
        simd::{
            Operand as Simd,
            rearrange::{ByteExtract, LaneSource, Operand},
        },
    },
};
use Gpr::{R8, Rax, Rcx, Rdi, Rdx};
use alloc::vec::Vec;

const REVERSE_DWORD_BYTES: [u8; 16] = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12];
const SWAP_WORDS_IN_DWORDS: u8 = 0xb1;
const REVERSE_MASK: u8 = 2;
const TABLE_INDEX_VECTOR: u8 = 1;
const TABLE_BIAS_VECTOR: u8 = 2;

/// REV32 Vd.16B is a one-source permutation. First swap bytes within
/// words, then swap the two words of each dword with SSE2 shuffles.
fn reverse_dwords(code: &mut Code, destination: u8, source: u8) {
    read_vector(code, source, 0);
    packed(code, Host::MoveAligned, 1, 0);
    ShiftOpcode::LeftWord.emit(code, 0, 8);
    ShiftOpcode::RightWord.emit(code, 1, 8);
    packed(code, Host::Or, 0, 1);
    packed_imm(code, Host::ShuffleLowWord, 0, 0, SWAP_WORDS_IN_DWORDS);
    packed_imm(code, Host::ShuffleHighWord, 0, 0, SWAP_WORDS_IN_DWORDS);
    commit(code, destination, 0);
}

fn reverse_dword_operands(instruction: &Instruction) -> Option<(u8, u8)> {
    match instruction.operation {
        Operation::Simd(Simd::Rearrange(Operand::Permute {
            destination,
            left,
            selection: REVERSE_DWORD_BYTES,
            ..
        })) => Some((destination, left)),
        _ => None,
    }
}

/// Use one SSSE3 mask for a short sequence of REV32 Vd.16B operations.
/// Each row still owns its frame load and commit, including aliased inputs.
pub(in crate::x86_64) fn emit_reverse_dword_run(
    code: &mut Code,
    source_map: &mut Vec<(GuestAddress, usize)>,
    instructions: &[Instruction],
    target: TargetCpu,
) -> Option<(usize, u8)> {
    debug_assert!(target.supports(TargetCpu::Ssse3));
    let [first, second, ..] = instructions else {
        return None;
    };
    reverse_dword_operands(first)?;
    reverse_dword_operands(second)?;
    if first.pc.0.checked_add(4) != Some(second.pc.0) {
        return None;
    }
    let mut last = None;
    for (index, instruction) in instructions.iter().take(4).enumerate() {
        let Some((destination, source)) = reverse_dword_operands(instruction) else {
            break;
        };
        if first.pc.0.checked_add(index as u64 * 4) != Some(instruction.pc.0) {
            break;
        }
        source_map.push((instruction.pc, code.len()));
        code.set_instruction(source_map.len() - 1);
        if index == 0 {
            constant(code, REVERSE_MASK, REVERSE_DWORD_BYTES);
        }
        read_vector(code, source, 0);
        Host::ByteShuffle.register(code, 0, REVERSE_MASK, target);
        commit(code, destination, 0);
        last = Some((index + 1, destination));
    }
    last
}

/// Load the 16 `bytes` into `xmm` without a constant pool (clobbers RCX and
/// XMM7). Zero, and a run of ones at either end of every word, dword or
/// qword lane, come from PXOR or PCMPEQD and one shift. A constant with
/// repeated dwords or qwords needs one immediate and one broadcast.
pub(super) fn constant(code: &mut Code, xmm: u8, bytes: [u8; 16]) {
    let [low, high] =
        core::array::from_fn(|i| u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap()));
    if low == high {
        let dword = low as u32;
        if low == 0 {
            packed(code, Host::Xor, xmm, xmm);
        } else if let Some((shift, amount)) = ones_run(low) {
            packed(code, Host::EqualDword, xmm, xmm);
            if let Some(shift) = shift {
                shift.emit(code, xmm, amount);
            }
        } else if low >> 32 == u64::from(dword) {
            let mut asm = code.asm();
            asm.mov_imm32(Rcx, dword);
            asm.movd_to_xmm(xmm, Rcx);
            packed_imm(code, Host::ShuffleDword, xmm, xmm, 0);
        } else {
            code.asm().mov_imm64(Rcx, low);
            packed(code, Host::MoveGprQword, xmm, 1);
            packed(code, Host::UnpackLowQword, xmm, xmm);
        }
        return;
    }
    // MOVQ from a GPR followed by PUNPCKLQDQ avoids constant-pool relocations.
    for (register, limb) in [(xmm, low), (7, high)] {
        code.asm().mov_imm64(Rcx, limb);
        packed(code, Host::MoveGprQword, register, 1);
    }
    packed(code, Host::UnpackLowQword, xmm, 7);
}

/// The shift that produces a word, dword or qword run from all ones.
/// The inner `None` indicates an all-ones pattern that needs no shift.
fn ones_run(limb: u64) -> Option<(Option<ShiftOpcode>, u8)> {
    if limb == 0 {
        return None;
    }
    for (bits, right, left) in [
        (16, ShiftOpcode::RightWord, ShiftOpcode::LeftWord),
        (32, ShiftOpcode::RightDword, ShiftOpcode::LeftDword),
        (64, ShiftOpcode::RightQword, ShiftOpcode::LeftQword),
    ] {
        if limb.rotate_left(bits) != limb {
            continue;
        }
        let full = u64::MAX >> (64 - bits);
        let value = limb & full;
        let shift = (bits - value.count_ones()) as u8;
        if value == full {
            return Some((None, 0));
        }
        if value == full >> shift {
            return Some((Some(right), shift));
        }
        if value == (full << shift) & full {
            return Some((Some(left), shift));
        }
    }
    None
}

/// Load `value` repeated in each `lane`-byte lane of `xmm` (clobbers RCX and XMM7).
pub(super) fn lane_constant(code: &mut Code, xmm: u8, lane: u8, value: u64) {
    let bytes = value.to_le_bytes();
    constant(
        code,
        xmm,
        core::array::from_fn(|i| bytes[i % usize::from(lane)]),
    );
}

/// Packed ADD or SUB for `lane`-byte lanes.
pub(super) fn add_or_subtract(lane: u8, subtract: bool) -> Host {
    match (subtract, lane) {
        (false, 1) => Host::AddByte,
        (false, 2) => Host::AddWord,
        (false, 4) => Host::AddDword,
        (false, 8) => Host::AddQword,
        (true, 1) => Host::SubByte,
        (true, 2) => Host::SubWord,
        (true, 4) => Host::SubDword,
        (true, 8) => Host::SubQword,
        _ => unreachable!("lane of 1, 2, 4 or 8 bytes"),
    }
}

fn load_lane(code: &mut Code, source: LaneSource, bytes: u8) {
    use crate::aarch64::Register;
    let mut asm = code.asm();
    let storage = match source {
        LaneSource::General(Register::Zero) => {
            asm.alu(Alu::Xor, Size::Dword, Rax, Rax);
            return;
        }
        LaneSource::General(register @ Register::X(_)) => {
            asm.guest(register).expect("X0..X30 have storage")
        }
        LaneSource::Vector { register, byte } => {
            guest(vector_offset(register) + u32::from(byte)).into()
        }
        LaneSource::General(Register::Sp) => unreachable!("lane transfers use ZR, not SP"),
    };
    // Zero-extending load of the exact width into RAX, from GuestState.
    match bytes {
        1 => asm.movzx(Size::Dword, Rax, Size::Byte, storage),
        2 => asm.movzx(Size::Dword, Rax, Size::Word, storage),
        4 => asm.mov_load(Size::Dword, Rax, storage),
        8 => asm.mov_load(Size::Qword, Rax, storage),
        _ => unreachable!("reviewed lane size"),
    }
}

/// UMOV, SMOV and FMOV to a general register ([`Operand::Extract`]),
/// which ends with the destination's store from RAX. With `held`, XMM0
/// holds the source vector whole and the lane is its lowest.
pub(super) fn extract(code: &mut Code, operand: &Operand, held: bool) {
    let Operand::Extract {
        destination,
        source,
        byte,
        lane_bytes,
        signed,
        x64,
    } = *operand
    else {
        unreachable!("an extract");
    };
    if held {
        debug_assert_eq!(byte, 0);
        let mut asm = code.asm();
        match lane_bytes {
            1 | 2 | 4 => asm.movd_from_xmm(Rax, 0),
            8 => asm.movq_from_xmm(Rax, 0),
            _ => unreachable!("reviewed lane size"),
        }
        match lane_bytes {
            1 => asm.movzx(Size::Dword, Rax, Size::Byte, Rax),
            2 => asm.movzx(Size::Dword, Rax, Size::Word, Rax),
            _ => {}
        }
    } else {
        load_lane(
            code,
            LaneSource::Vector {
                register: source,
                byte,
            },
            lane_bytes,
        );
    }
    if signed {
        let shift = 64 - lane_bytes * 8;
        let mut asm = code.asm();
        asm.shift_imm(ShiftOp::Shl, Size::Qword, Rax, shift);
        asm.shift_imm(ShiftOp::Sar, Size::Qword, Rax, shift);
        if !x64 {
            // W writes clear X high half.
            asm.mov(Size::Dword, Rax, Rax);
        }
    }
    // An unsigned lane of at most four bytes is zero-extended already.
    super::super::store(code, destination);
}

fn insert_lane(code: &mut Code, destination: u8, byte: u8, bytes: u8) {
    // Only the selected lane changes; even scalar-size INS preserves all others.
    let size = match bytes {
        1 => Size::Byte,
        2 => Size::Word,
        4 => Size::Dword,
        8 => Size::Qword,
        _ => unreachable!("reviewed lane size"),
    };
    code.asm().mov_store(
        size,
        guest(vector_offset(destination) + u32::from(byte)),
        Rax,
    );
}

fn byte_extract(code: &mut Code, operand: ByteExtract, target: TargetCpu, held: Option<u8>) {
    let ByteExtract {
        destination,
        left,
        right,
        bytes,
        offset,
    } = operand;
    if offset == 0 {
        if held != Some(left) {
            read_vector(code, left, 0);
        }
    } else if bytes == 16 && target.supports(TargetCpu::Ssse3) {
        // PALIGNR concatenates the old destination above its source.
        if left == right {
            if held != Some(left) {
                read_vector(code, left, 0);
            }
            Host::AlignBytes.register_imm(code, 0, 0, offset, target);
        } else {
            if held == Some(left) {
                packed(code, Host::MoveAligned, 1, 0);
            } else {
                read_vector(code, left, 1);
            }
            if held != Some(right) {
                read_vector(code, right, 0);
            }
            Host::AlignBytes.register_imm(code, 0, 1, offset, target);
        }
    } else {
        if left != right {
            if held == Some(right) {
                packed(code, Host::MoveAligned, 1, 0);
            } else {
                read_vector(code, right, 1);
            }
        }
        if held != Some(left) {
            read_vector(code, left, 0);
        }
        if bytes == 8 {
            packed(code, Host::UnpackLowQword, 0, u8::from(left != right));
            ShiftOpcode::RightBytes.emit(code, 0, offset);
        } else {
            if left == right {
                packed(code, Host::MoveAligned, 1, 0);
            }
            ShiftOpcode::RightBytes.emit(code, 0, offset);
            ShiftOpcode::LeftBytes.emit(code, 1, 16 - offset);
            packed(code, Host::Or, 0, 1);
        }
    }
    if bytes == 8 {
        packed(code, Host::MoveQword, 0, 0);
    }
    commit(code, destination, 0);
}

fn permutation(
    code: &mut Code,
    destination: u8,
    left: u8,
    right: u8,
    selection: [u8; 16],
    target: TargetCpu,
) {
    if selection == REVERSE_DWORD_BYTES && !code.constant_time() {
        reverse_dwords(code, destination, left);
        return;
    }
    if target >= TargetCpu::Ssse3 {
        read_vector(code, left, 0);
        read_vector(code, right, 1);
        for (source, xmm) in [(0, 0), (16, 1)] {
            let mask = selection.map(|index| {
                if (source..source + 16).contains(&index) {
                    index - source
                } else {
                    0x80
                }
            });
            constant(code, 2, mask);
            Host::ByteShuffle.register(code, xmm, 2, target);
        }
        packed(code, Host::Or, 0, 1);
        commit(code, destination, 0);
        return;
    }
    // Scalar byte gather fallback. RAX/RDX accumulate both halves before stores;
    // RCX is the zero-extended byte scratch. No guest RAM or host stack accesses.
    clear_halves(code);
    for (output, input) in selection.into_iter().enumerate() {
        if input == 255 {
            continue;
        }
        let vector = if input < 16 { left } else { right };
        code.asm().movzx(
            Size::Dword,
            Rcx,
            Size::Byte,
            guest(vector_offset(vector) + u32::from(input % 16)),
        );
        accumulate_byte(code, output, false);
    }
    commit_halves(code, destination);
}

/// Clear RAX and RDX, the low and high halves of a gathered vector.
fn clear_halves(code: &mut Code) {
    let mut asm = code.asm();
    asm.alu(Alu::Xor, Size::Dword, Rax, Rax);
    asm.alu(Alu::Xor, Size::Dword, Rdx, Rdx);
}

fn accumulate_byte(code: &mut Code, output: usize, r8: bool) {
    let shift = (output % 8 * 8) as u8;
    let byte = if r8 { R8 } else { Rcx };
    let mut asm = code.asm();
    if shift != 0 {
        asm.shift_imm(ShiftOp::Shl, Size::Qword, byte, shift);
    }
    let half = if output < 8 { Rax } else { Rdx };
    asm.alu(Alu::Or, Size::Qword, half, byte);
}
fn commit_halves(code: &mut Code, destination: u8) {
    let mut asm = code.asm();
    asm.mov_store(Size::Qword, guest(vector_offset(destination)), Rax);
    asm.mov_store(Size::Qword, guest(vector_offset(destination) + 8), Rdx);
}

fn prepare_single_table_indices(code: &mut Code, indices: u8) {
    read_vector(code, indices, TABLE_INDEX_VECTOR);
    let mut asm = code.asm();
    asm.mov_imm32(Rax, u32::from_le_bytes([0x70; 4]));
    asm.movd_to_xmm(TABLE_BIAS_VECTOR, Rax);
    super::shuffle(code, TABLE_BIAS_VECTOR, TABLE_BIAS_VECTOR, 0);
    // Saturating addition leaves the low nibble of 0..15 unchanged and
    // sets bit 7 for every index >=16. PSHUFB then zeroes those lanes.
    packed(
        code,
        Host::AddUnsignedSaturatingByte,
        TABLE_INDEX_VECTOR,
        TABLE_BIAS_VECTOR,
    );
}

fn single_table(code: &mut Code, destination: u8, first: u8, bytes: u8, target: TargetCpu) {
    read_vector(code, first, 0);
    Host::ByteShuffle.register(code, 0, TABLE_INDEX_VECTOR, target);
    if bytes == 8 {
        packed(code, Host::MoveQword, 0, 0);
    }
    commit(code, destination, 0);
}

struct SingleTable {
    destination: u8,
    first: u8,
    indices: u8,
    bytes: u8,
}

fn single_table_operands(instruction: &Instruction) -> Option<SingleTable> {
    let Operation::Simd(Simd::Rearrange(Operand::Table {
        destination,
        first,
        registers: 1,
        indices,
        bytes,
        preserve: false,
    })) = instruction.operation
    else {
        return None;
    };
    Some(SingleTable {
        destination,
        first,
        indices,
        bytes,
    })
}

/// Adjacent one-table lookups may share their range-checked index vector.
/// Each table is still read before its own write, and a write to the index
/// register ends the run after completing that lookup.
pub(in crate::x86_64) fn emit_single_table_run(
    code: &mut Code,
    source_map: &mut Vec<(GuestAddress, usize)>,
    instructions: &[Instruction],
    target: TargetCpu,
) -> Option<(usize, u8)> {
    let [first, second, ..] = instructions else {
        return None;
    };
    let head = single_table_operands(first)?;
    let next = single_table_operands(second)?;
    if head.destination == head.indices
        || head.indices != next.indices
        || first.pc.0.checked_add(4) != Some(second.pc.0)
    {
        return None;
    }
    let mut last = None;
    for (index, instruction) in instructions.iter().take(4).enumerate() {
        let Some(row) = single_table_operands(instruction) else {
            break;
        };
        if row.indices != head.indices
            || first.pc.0.checked_add(index as u64 * 4) != Some(instruction.pc.0)
        {
            break;
        }
        code.set_instruction(source_map.len());
        source_map.push((instruction.pc, code.len()));
        if index == 0 {
            prepare_single_table_indices(code, row.indices);
        }
        single_table(code, row.destination, row.first, row.bytes, target);
        last = Some((index + 1, row.destination));
        if row.destination == row.indices {
            break;
        }
    }
    last
}

fn table(code: &mut Code, operand: &Operand, target: TargetCpu) {
    let Operand::Table {
        destination,
        first,
        registers,
        indices,
        bytes,
        preserve,
    } = *operand
    else {
        unreachable!("a decoded table operation");
    };
    if vector_extensions::table_supported(1, target) {
        read_vector(code, indices, 1);
        if preserve {
            read_vector(code, destination, 0);
        }
        let tables: alloc::vec::Vec<_> = (0..registers)
            .map(|index| guest(vector_offset((first + index) % 32)))
            .collect();
        vector_extensions::table(code, 1, 0, 1, &tables, preserve, [2, 3], target);
        if bytes == 8 {
            packed(code, Host::MoveQword, 0, 0);
        }
        commit(code, destination, 0);
        return;
    }
    if target >= TargetCpu::Ssse3 {
        if registers == 1 && !preserve {
            prepare_single_table_indices(code, indices);
            single_table(code, destination, first, bytes, target);
            return;
        }
        read_vector(code, indices, 1);
        packed(code, Host::Xor, 0, 0); // result
        packed(code, Host::Xor, 6, 6); // valid-index mask for TBX
        constant(code, 4, [0xf0; 16]);
        packed(code, Host::MoveAligned, 3, 1);
        packed(code, Host::And, 3, 4); // high nibble identifies table register, including unsigned OOB
        for index in 0..registers {
            constant(code, 5, [index * 16; 16]);
            packed(code, Host::EqualByte, 5, 3);
            packed(code, Host::Or, 6, 5);
            read_vector(code, (first + index) % 32, 2);
            Host::ByteShuffle.register(code, 2, 1, target);
            packed(code, Host::And, 2, 5);
            packed(code, Host::Or, 0, 2);
        }
        if preserve {
            read_vector(code, destination, 2);
            packed(code, Host::AndNot, 6, 2);
            packed(code, Host::Or, 0, 6);
        }
        if bytes == 8 {
            packed(code, Host::MoveQword, 0, 0);
        }
        commit(code, destination, 0);
        return;
    }
    clear_halves(code);
    for output in 0..bytes {
        let mut asm = code.asm();
        if preserve {
            // The old destination byte.
            asm.movzx(
                Size::Dword,
                R8,
                Size::Byte,
                guest(vector_offset(destination) + u32::from(output)),
            );
        } else {
            asm.alu(Alu::Xor, Size::Dword, R8, R8);
        }
        asm.movzx(
            Size::Dword,
            Rcx,
            Size::Byte,
            guest(vector_offset(indices) + u32::from(output)),
        );
        asm.alu_imm8(Alu::Cmp, Size::Dword, Rcx, (registers * 16) as i8);
        let fallback = asm.jcc(Cond::Ae);
        // The first register's byte offset; the vector bank wraps at V31.
        asm.alu_imm32(Alu::Add, Size::Dword, Rcx, i32::from(first) * 16);
        asm.alu_imm32(Alu::And, Size::Dword, Rcx, 32 * 16 - 1);
        asm.movzx(
            Size::Dword,
            R8,
            Size::Byte,
            Mem::indexed(Rdi, Rcx, 1, vector_offset(0) as i32).with_width(Disp::Dword),
        );
        asm.bind(fallback);
        accumulate_byte(code, usize::from(output), true);
    }
    commit_halves(code, destination);
}

pub(super) fn emit(code: &mut Code, operand: &Operand, target: TargetCpu, held: Option<u8>) {
    match *operand {
        Operand::Duplicate {
            destination,
            source,
            lane_bytes,
            bytes,
        } => {
            load_lane(code, source, lane_bytes);
            packed(code, Host::MoveGprQword, 0, 0);
            // Unpacks broadcast the low element without SSE3+ requirements.
            if bytes > lane_bytes {
                for opcode in [
                    Host::UnpackLowByte,
                    Host::UnpackLowWord,
                    Host::UnpackLowDword,
                    Host::UnpackLowQword,
                ]
                .into_iter()
                .skip(lane_bytes.trailing_zeros() as usize)
                {
                    packed(code, opcode, 0, 0);
                }
            }
            if bytes <= 8 {
                packed(code, Host::MoveQword, 0, 0);
            }
            commit(code, destination, 0);
        }
        Operand::Insert {
            destination,
            source,
            byte,
            lane_bytes,
        } => {
            load_lane(code, source, lane_bytes);
            insert_lane(code, destination, byte, lane_bytes);
        }
        Operand::Extract { .. } => extract(code, operand, false),
        Operand::ByteExtract(operand) => byte_extract(code, operand, target, held),
        Operand::Permute {
            destination,
            left,
            right,
            selection,
        } => permutation(code, destination, left, right, selection, target),
        Operand::Table { .. } => table(code, operand, target),
    }
}

#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "macos")
))]
mod tests;
