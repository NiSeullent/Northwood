//! Lane moves and permutes. All inputs are read before the destination is written.
use super::{Host, commit, packed, packed_imm, read_vector, vector_offset};
use crate::x86_64::asm::{Alu, Code, Cond, Disp, Gpr, Mem, Rm, ShiftOp, Size, vector_extensions};
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
/// qword lane, come from PXOR or PCMPEQD and one shift; equal halves take one
/// MOVQ.
pub(super) fn constant(code: &mut Code, xmm: u8, bytes: [u8; 16]) {
    if bytes == [0; 16] {
        packed(code, Host::Xor, xmm, xmm);
        return;
    }
    if let Some((shift, amount)) = ones_run(&bytes) {
        packed(code, Host::EqualDword, xmm, xmm);
        if let Some(shift) = shift {
            shift.emit(code, xmm, amount);
        }
        return;
    }
    let (low, high) = bytes.split_at(8);
    if low == high {
        let limb = u64::from_le_bytes(low.try_into().expect("an eight-byte limb"));
        code.asm().mov_imm64(Rcx, limb);
        packed(code, Host::MoveGprQword, xmm, 1);
        packed(code, Host::UnpackLowQword, xmm, xmm);
        return;
    }
    // MOVQ from a GPR followed by PUNPCKLQDQ avoids constant-pool relocations.
    for (register, limb) in [(xmm, &bytes[..8]), (7, &bytes[8..])] {
        let limb = u64::from_le_bytes(limb.try_into().expect("an eight-byte limb"));
        code.asm().mov_imm64(Rcx, limb);
        packed(code, Host::MoveGprQword, register, 1);
    }
    packed(code, Host::UnpackLowQword, xmm, 7);
}

/// Whether [`constant`] loads `bytes` with PCMPEQD and at most one shift.
pub(super) fn ones_constant(bytes: &[u8; 16]) -> bool {
    ones_run(bytes).is_some()
}

/// The shift that turns all ones into `bytes`, when every word, dword or
/// qword lane of `bytes` holds the same run of ones at its low or high end:
/// `None` for all ones.
fn ones_run(bytes: &[u8; 16]) -> Option<(Option<ShiftOpcode>, u8)> {
    for (lane, right, left) in [
        (2, ShiftOpcode::RightWord, ShiftOpcode::LeftWord),
        (4, ShiftOpcode::RightDword, ShiftOpcode::LeftDword),
        (8, ShiftOpcode::RightQword, ShiftOpcode::LeftQword),
    ] {
        if bytes.chunks(lane).any(|chunk| chunk != &bytes[..lane]) {
            continue;
        }
        let mut value = [0; 8];
        value[..lane].copy_from_slice(&bytes[..lane]);
        let value = u64::from_le_bytes(value);
        let bits = lane as u32 * 8;
        let ones = value.count_ones();
        let full = u64::MAX >> (64 - bits);
        if value == full {
            return Some((None, 0));
        }
        if value == full >> (bits - ones) {
            return Some((Some(right), (bits - ones) as u8));
        }
        if value == (full << (bits - ones)) & full {
            return Some((Some(left), (bits - ones) as u8));
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

fn load_lane(code: &mut Code, source: LaneSource, bytes: u8, target: TargetCpu) {
    use crate::aarch64::Register;
    if let LaneSource::Vector { register, byte } = source
        && let Some(home) = code.vector_cache().peek(register)
        && (bytes == 2 || target >= TargetCpu::Sse41)
    {
        // The lane straight from the home, zero-extended into RAX: no
        // store of a newer home first. The home is only read (ModRM.reg
        // names it in PEXTRB/D/Q).
        let lane = byte / bytes;
        let mut asm = code.asm_unguarded();
        match bytes {
            1 => asm.sse_imm(Some(0x66), &[0x0f, 0x3a, 0x14], home, Rm::Reg(Rax), lane),
            2 => asm.pextrw(Rax, home, lane),
            4 => asm.pextrd(Rax, home, lane),
            _ => asm.pextrq(Rax, home, lane),
        }
        return;
    }
    if let LaneSource::Vector { register, .. } = source {
        // The lane is read from the slot: store a newer home first.
        code.write_back_vector(register);
    }
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
pub(super) fn extract(code: &mut Code, operand: &Operand, held: bool, target: TargetCpu) {
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
            target,
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

fn insert_lane(code: &mut Code, destination: u8, byte: u8, bytes: u8, target: TargetCpu) {
    let cache = code.vector_cache();
    if cache.active() && !cache.writes_through() && (bytes == 2 || target >= TargetCpu::Sse41) {
        // RAX's lane straight into the destination's home: one PINSR.
        let home = super::vector_accumulator(code, destination, 1);
        let lane = byte / bytes;
        let mut asm = code.asm();
        match bytes {
            1 => asm.sse_imm(
                Some(0x66),
                &[0x0f, 0x3a, 0x20],
                home.xmm,
                Rm::Reg(Rax),
                lane,
            ),
            2 => asm.pinsrw(home.xmm, Rm::Reg(Rax), lane),
            4 => asm.pinsrd(home.xmm, Rm::Reg(Rax), lane),
            _ => {
                asm.legacy(
                    Size::Qword,
                    Some(0x66),
                    &[0x0f, 0x3a, 0x22],
                    home.xmm,
                    false,
                    Rm::Reg(Rax),
                );
                asm.data(&[lane]);
            }
        }
        super::finish(code, home);
        return;
    }
    // Only the selected lane changes; even scalar-size INS preserves all others.
    let size = match bytes {
        1 => Size::Byte,
        2 => Size::Word,
        4 => Size::Dword,
        8 => Size::Qword,
        _ => unreachable!("reviewed lane size"),
    };
    // The lane is written to the slot: the home, stored if newer, goes.
    code.evict_vector(destination);
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

/// Every 16-byte lane of XMM0 = the `lane`-byte element of `source`. A
/// general source is read in place (its frame slot or host register), a
/// vector source with [`read_vector`]. AVX2 broadcasts with one VPBROADCAST;
/// SSE2 shuffles. Returns whether an eight-byte result has its high qword
/// zero already (ZR, and the SSE2 forms of a general source).
fn duplicate(code: &mut Code, source: LaneSource, lane: u8, bytes: u8, target: TargetCpu) -> bool {
    use crate::aarch64::Register;
    use crate::x86_64::asm::Rm;
    let avx2 = super::three::vex(target);
    match source {
        LaneSource::General(Register::Zero) => {
            packed(code, Host::Xor, 0, 0);
            true
        }
        LaneSource::General(register @ Register::X(_)) => {
            let storage = code.asm().guest(register).expect("X0..X30 have storage");
            if avx2 && matches!(storage, Rm::Mem(_)) {
                super::three::broadcast(code, lane, 0, storage);
                return false;
            }
            let mut asm = code.asm();
            match (lane, storage) {
                (8, Rm::Reg(gpr)) => asm.movq_to_xmm(0, gpr),
                // MOVQ xmm, m64 (F3 0F 7E).
                (8, storage) => asm.sse(Some(0xf3), &[0x0f, 0x7e], 0, storage),
                (_, storage) => asm.movd_to_xmm(0, storage),
            }
            if avx2 {
                super::three::broadcast(code, lane, 0, Rm::Xmm(0));
                return false;
            }
            // MOVD cleared bytes 4..15 (MOVQ bytes 8..15): the low qword is
            // filled first, and only a 16-byte vector copies it up.
            match lane {
                1 => {
                    packed(code, Host::UnpackLowByte, 0, 0);
                    packed_imm(code, Host::ShuffleLowWord, 0, 0, 0);
                }
                2 => packed_imm(code, Host::ShuffleLowWord, 0, 0, 0),
                4 if bytes <= 8 => packed_imm(code, Host::ShuffleDword, 0, 0, 0x50),
                4 => packed_imm(code, Host::ShuffleDword, 0, 0, 0),
                _ => {}
            }
            if bytes == 16 && lane != 4 {
                packed(code, Host::UnpackLowQword, 0, 0);
            }
            true
        }
        LaneSource::General(Register::Sp) => unreachable!("lane transfers use ZR, not SP"),
        LaneSource::Vector { register, byte } => {
            read_vector(code, register, 0);
            let index = byte / lane;
            match lane {
                8 => packed_imm(
                    code,
                    Host::ShuffleDword,
                    0,
                    0,
                    [0x44, 0xee][usize::from(index)],
                ),
                4 => packed_imm(code, Host::ShuffleDword, 0, 0, index * 0x55),
                2 if index < 4 => {
                    packed_imm(code, Host::ShuffleLowWord, 0, 0, index * 0x55);
                    packed(code, Host::UnpackLowQword, 0, 0);
                }
                2 => {
                    packed_imm(code, Host::ShuffleHighWord, 0, 0, (index - 4) * 0x55);
                    packed(code, Host::UnpackHighQword, 0, 0);
                }
                _ => {
                    if byte != 0 {
                        ShiftOpcode::RightBytes.emit(code, 0, byte);
                    }
                    if avx2 {
                        super::three::broadcast(code, 1, 0, Rm::Xmm(0));
                    } else {
                        packed(code, Host::UnpackLowByte, 0, 0);
                        packed_imm(code, Host::ShuffleLowWord, 0, 0, 0);
                        packed(code, Host::UnpackLowQword, 0, 0);
                    }
                }
            }
            false
        }
    }
}

/// Whether `operand` keeps the block's vector homes
/// ([`super::vector_use`]): lane moves store or give up the home of the
/// vector whose slot they address; the SSSE3 permutation and the byte
/// extraction use the operand layer alone; tables and the scalar byte
/// gather address slots.
pub(in crate::x86_64) fn vector_use(
    operand: &Operand,
    target: TargetCpu,
) -> crate::x86_64::block::VectorUse {
    use crate::x86_64::block::VectorUse::{Keep, Settle};
    match *operand {
        Operand::Duplicate { .. }
        | Operand::Insert { .. }
        | Operand::Extract { .. }
        | Operand::ByteExtract(_) => Keep,
        Operand::Permute { .. } if target >= TargetCpu::Ssse3 => Keep,
        Operand::Permute { .. } | Operand::Table { .. } => Settle,
    }
}

/// DUP of a 16-byte vector into the destination's home: one VPBROADCAST
/// from a general register's slot or a vector's low element, or one PSHUFD
/// of a vector's dword or qword element. Returns whether it emitted it.
fn duplicate_direct(
    code: &mut Code,
    destination: u8,
    source: LaneSource,
    lane: u8,
    target: TargetCpu,
) -> bool {
    use crate::aarch64::Register;
    use crate::x86_64::asm::Rm;
    if !super::three::vex(target) || !code.vector_cache().active() {
        return false;
    }
    match source {
        LaneSource::General(register @ Register::X(_)) => {
            let storage = code.asm().guest(register).expect("X0..X30 have storage");
            let from = match storage {
                Rm::Reg(gpr) => {
                    let mut asm = code.asm();
                    if lane == 8 {
                        asm.movq_to_xmm(0, gpr);
                    } else {
                        asm.movd_to_xmm(0, gpr);
                    }
                    Rm::Xmm(0)
                }
                storage => storage,
            };
            let destination = super::vector_destination(code, destination, 0);
            super::three::broadcast(code, lane, destination.xmm, from);
            super::finish(code, destination);
            true
        }
        LaneSource::Vector { register, byte } if byte == 0 || lane >= 4 => {
            let from = super::vector_source(code, register, 0);
            let destination = super::vector_destination(code, destination, 0);
            let index = byte / lane;
            match lane {
                8 => packed_imm(
                    code,
                    Host::ShuffleDword,
                    destination.xmm,
                    from,
                    [0x44, 0xee][usize::from(index)],
                ),
                4 => packed_imm(
                    code,
                    Host::ShuffleDword,
                    destination.xmm,
                    from,
                    index * 0x55,
                ),
                _ => super::three::broadcast(code, lane, destination.xmm, Rm::Xmm(from)),
            }
            super::finish(code, destination);
            true
        }
        _ => false,
    }
}

pub(super) fn emit(code: &mut Code, operand: &Operand, target: TargetCpu, held: Option<u8>) {
    if let Operand::Duplicate {
        destination,
        source,
        lane_bytes,
        bytes: 16,
    } = *operand
        && lane_bytes < 16
        && duplicate_direct(code, destination, source, lane_bytes, target)
    {
        return;
    }
    match *operand {
        Operand::Duplicate {
            destination,
            source,
            lane_bytes,
            bytes,
        } if bytes > lane_bytes && !target.data_independent_timing() => {
            let high_zero = duplicate(code, source, lane_bytes, bytes, target);
            if bytes <= 8 && !high_zero {
                packed(code, Host::MoveQword, 0, 0);
            }
            commit(code, destination, 0);
        }
        Operand::Duplicate {
            destination,
            source,
            lane_bytes,
            bytes,
        } => {
            load_lane(code, source, lane_bytes, target);
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
            load_lane(code, source, lane_bytes, target);
            insert_lane(code, destination, byte, lane_bytes, target);
        }
        Operand::Extract { .. } => extract(code, operand, false, target),
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
