//! Integer pairwise combines, across-lane reductions and bit counts.
use super::rearrange::lane_constant;
use super::{
    Host, commit, packed, read_vector, rearrange::constant, shifts::logical_shift, shuffle,
};
use crate::x86_64::asm::{Code, vector_extensions};
use crate::{
    TargetCpu,
    aarch64::simd::reductions::{BitCount, Combine, Kind, Operand, Shape},
    x86_64::isa::ShiftOpcode as Shift,
};

fn add_opcode(lane: u8) -> Host {
    match lane {
        1 => Host::AddByte,
        2 => Host::AddWord,
        4 => Host::AddDword,
        8 => Host::AddQword,
        _ => unreachable!(),
    }
}
fn compare_opcode(lane: u8) -> Host {
    match lane {
        1 => Host::GreaterByte,
        2 => Host::GreaterWord,
        4 => Host::GreaterDword,
        _ => unreachable!(),
    }
}
fn equal_opcode(lane: u8) -> Host {
    match lane {
        1 => Host::EqualByte,
        2 => Host::EqualWord,
        4 => Host::EqualDword,
        _ => unreachable!(),
    }
}
/// The host min/max of `lane`-byte lanes, which may need SSE4.1.
fn native_minmax(lane: u8, signed: bool, maximum: bool) -> Host {
    match (signed, maximum, lane) {
        (false, false, 1) => Host::MinUnsignedByte,
        (false, true, 1) => Host::MaxUnsignedByte,
        (true, false, 2) => Host::MinSignedWord,
        (true, true, 2) => Host::MaxSignedWord,
        (true, false, 1) => Host::MinSignedByte,
        (true, true, 1) => Host::MaxSignedByte,
        (false, false, 2) => Host::MinUnsignedWord,
        (false, true, 2) => Host::MaxUnsignedWord,
        (true, false, 4) => Host::MinSignedDword,
        (true, true, 4) => Host::MaxSignedDword,
        (false, false, 4) => Host::MinUnsignedDword,
        (false, true, 4) => Host::MaxUnsignedDword,
        _ => unreachable!("allocated min/max lane"),
    }
}

fn select_minmax(code: &mut Code, lane: u8, signed: bool, maximum: bool, target: TargetCpu) {
    let native = native_minmax(lane, signed, maximum);
    if target >= native.requires() {
        native.register(code, 0, 1, target);
        return;
    }
    packed(code, Host::MoveAligned, 2, 0);
    packed(code, Host::MoveAligned, 3, 1);
    if !signed {
        lane_constant(code, 4, lane, 1u64 << (lane * 8 - 1));
        packed(code, Host::Xor, 2, 4);
        packed(code, Host::Xor, 3, 4);
    }
    packed(code, compare_opcode(lane), 2, 3); // mask where original left > right
    if !maximum {
        packed(code, Host::EqualDword, 3, 3);
        packed(code, Host::Xor, 2, 3);
    }
    // right XOR ((left XOR right) AND selected-left mask)
    packed(code, Host::Xor, 0, 1);
    packed(code, Host::And, 0, 2);
    packed(code, Host::Xor, 0, 1);
}
fn combine(code: &mut Code, operation: Combine, lane: u8, target: TargetCpu) {
    match operation {
        Combine::Add | Combine::LongAdd { .. } => packed(code, add_opcode(lane), 0, 1),
        Combine::Minimum { signed } => select_minmax(code, lane, signed, false, target),
        Combine::Maximum { signed } => select_minmax(code, lane, signed, true, target),
    }
}

/// Produce adjacent even or odd lanes from XMM2/XMM3, respecting active width.
/// XMM6 preserves the first selection while the second uses XMM0/XMM4/XMM5/7.
fn pair_selection(code: &mut Code, lane: u8, bytes: u8, odd: bool) {
    packed(code, Host::MoveAligned, 0, 2);
    packed(code, Host::MoveAligned, 4, 3);
    match lane {
        1 => {
            if odd {
                Shift::RightWord.emit(code, 0, 8);
                Shift::RightWord.emit(code, 4, 8);
            } else {
                lane_constant(code, 5, 2, 0xff);
                packed(code, Host::And, 0, 5);
                packed(code, Host::And, 4, 5);
            }
            packed(code, Host::PackUnsignedWordToByte, 0, 4);
            if bytes == 8 {
                shuffle(code, 0, 0, 0x88);
            }
        }
        2 => {
            if !odd {
                Shift::LeftDword.emit(code, 0, 16);
                Shift::LeftDword.emit(code, 4, 16);
            }
            Shift::ArithmeticDword.emit(code, 0, 16);
            Shift::ArithmeticDword.emit(code, 4, 16);
            packed(code, Host::PackSignedDwordToWord, 0, 4);
            if bytes == 8 {
                shuffle(code, 0, 0, 0x88);
            }
        }
        4 => {
            let control = if odd { 0xdd } else { 0x88 };
            shuffle(code, 0, 0, control);
            shuffle(code, 4, 4, control);
            packed(
                code,
                if bytes == 8 {
                    Host::UnpackLowDword
                } else {
                    Host::UnpackLowQword
                },
                0,
                4,
            );
        }
        8 => packed(
            code,
            if odd {
                Host::UnpackHighQword
            } else {
                Host::UnpackLowQword
            },
            0,
            4,
        ),
        _ => unreachable!("allocated pair lane"),
    }
}
fn pair_operands(code: &mut Code, operand: &Operand) {
    read_vector(code, operand.left, 2);
    read_vector(code, operand.right, 3);
    pair_selection(code, operand.lane_bytes, operand.bytes, false);
    packed(code, Host::MoveAligned, 6, 0);
    pair_selection(code, operand.lane_bytes, operand.bytes, true);
    packed(code, Host::MoveAligned, 1, 0);
    packed(code, Host::MoveAligned, 0, 6);
}

fn long_sum_inputs(code: &mut Code, lane: u8, bytes: u8, signed: bool) {
    packed(code, Host::Xor, 1, 1);
    if signed {
        packed(code, compare_opcode(lane), 1, 0);
    }
    if bytes == 16 {
        packed(code, Host::MoveAligned, 2, 0);
        packed(
            code,
            match lane {
                1 => Host::UnpackHighByte,
                2 => Host::UnpackHighWord,
                4 => Host::UnpackHighDword,
                _ => unreachable!(),
            },
            2,
            1,
        );
    }
    packed(
        code,
        match lane {
            1 => Host::UnpackLowByte,
            2 => Host::UnpackLowWord,
            4 => Host::UnpackLowDword,
            _ => unreachable!(),
        },
        0,
        1,
    );
    if bytes == 16 {
        packed(code, add_opcode(lane * 2), 0, 2);
    }
}
/// UMINV or UMAXV of bytes or halfwords in XMM0 with SSE4.1's PHMINPOSUW,
/// the minimum of eight unsigned words, into its low word; the bits above
/// the result lane are left for the caller to clear. A maximum is the
/// complement of the minimum of the complements.
fn unsigned_word_extremum(code: &mut Code, maximum: bool, lane: u8, bytes: u8, target: TargetCpu) {
    const ONES: u8 = 1;
    const SCRATCH: u8 = 2;
    if maximum {
        packed(code, Host::EqualDword, ONES, ONES);
        packed(code, Host::Xor, 0, ONES);
    }
    match (lane, bytes) {
        (1, VECTOR_BYTES) => {
            packed(code, Host::MoveAligned, SCRATCH, 0);
            Shift::RightBytes.emit(code, SCRATCH, VECTOR_BYTES / 2);
            Host::MinUnsignedByte.register(code, 0, SCRATCH, target);
            Host::ZeroExtendByteToWord.register(code, 0, 0, target);
        }
        // Only the low eight bytes widen.
        (1, _) => Host::ZeroExtendByteToWord.register(code, 0, 0, target),
        (2, VECTOR_BYTES) => {}
        // Ones in the upper four words never win a minimum.
        (2, _) => {
            packed(code, Host::EqualDword, SCRATCH, SCRATCH);
            packed(code, Host::UnpackLowQword, 0, SCRATCH);
        }
        _ => unreachable!("byte or halfword lanes"),
    }
    Host::MinimumPositionWord.register(code, 0, 0, target);
    if maximum {
        packed(code, Host::Xor, 0, ONES);
    }
}

fn across(code: &mut Code, operation: Combine, mut lane: u8, mut bytes: u8, target: TargetCpu) {
    if let Combine::LongAdd { signed } = operation {
        long_sum_inputs(code, lane, bytes, signed);
        lane *= 2;
        bytes = 16;
    }
    let extremum = match operation {
        Combine::Minimum { signed: false } => Some(false),
        Combine::Maximum { signed: false } => Some(true),
        _ => None,
    };
    if let Some(maximum) = extremum
        && lane <= 2
        && target.supports(TargetCpu::Sse41)
    {
        unsigned_word_extremum(code, maximum, lane, bytes, target);
    } else {
        let mut stride = bytes / 2;
        while stride >= lane {
            packed(code, Host::MoveAligned, 1, 0);
            Shift::RightBytes.emit(code, 1, stride);
            combine(code, operation, lane, target);
            stride /= 2;
        }
    }
    // The low lane alone: shifting it to the top and back clears the rest.
    let clear = VECTOR_BYTES - lane;
    Shift::LeftBytes.emit(code, 0, clear);
    Shift::RightBytes.emit(code, 0, clear);
}

/// The bytes of an XMM register.
const VECTOR_BYTES: u8 = 16;

fn nibble_lookup(code: &mut Code, reverse: bool, target: TargetCpu) {
    let table = core::array::from_fn(|i| {
        if reverse {
            (i as u8).reverse_bits() >> 4
        } else {
            (i as u8).count_ones() as u8
        }
    });
    constant(code, 2, table);
    packed(code, Host::MoveAligned, 4, 2);
    packed(code, Host::MoveAligned, 1, 0);
    constant(code, 3, [15; 16]);
    packed(code, Host::And, 0, 3);
    Shift::RightWord.emit(code, 1, 4);
    packed(code, Host::And, 1, 3);
    Host::ByteShuffle.register(code, 2, 0, target);
    Host::ByteShuffle.register(code, 4, 1, target);
    if reverse {
        Shift::LeftWord.emit(code, 2, 4);
        packed(code, Host::Or, 2, 4);
    } else {
        packed(code, Host::AddByte, 2, 4);
    }
    packed(code, Host::MoveAligned, 0, 2);
}
fn bit_population(code: &mut Code, target: TargetCpu) {
    if vector_extensions::population_count(code, 1, 0, target) {
        return;
    }
    if target >= TargetCpu::Ssse3 {
        nibble_lookup(code, false, target);
        return;
    }
    packed(code, Host::MoveAligned, 1, 0);
    Shift::RightWord.emit(code, 1, 1);
    constant(code, 2, [0x55; 16]);
    packed(code, Host::And, 1, 2);
    packed(code, Host::SubByte, 0, 1);
    packed(code, Host::MoveAligned, 1, 0);
    Shift::RightWord.emit(code, 1, 2);
    constant(code, 2, [0x33; 16]);
    packed(code, Host::And, 0, 2);
    packed(code, Host::And, 1, 2);
    packed(code, Host::AddByte, 0, 1);
    packed(code, Host::MoveAligned, 1, 0);
    Shift::RightWord.emit(code, 1, 4);
    packed(code, Host::AddByte, 0, 1);
    constant(code, 2, [15; 16]);
    packed(code, Host::And, 0, 2);
}
fn bit_reverse(code: &mut Code, target: TargetCpu) {
    if vector_extensions::reverse_bytes(code, 0, 2, target) {
        return;
    }
    if target >= TargetCpu::Ssse3 {
        nibble_lookup(code, true, target);
        return;
    }
    for (amount, mask) in [(1, 0x55), (2, 0x33), (4, 0x0f)] {
        packed(code, Host::MoveAligned, 1, 0);
        Shift::RightWord.emit(code, 1, amount);
        constant(code, 2, [mask; 16]);
        packed(code, Host::And, 1, 2);
        packed(code, Host::And, 0, 2);
        Shift::LeftWord.emit(code, 0, amount);
        packed(code, Host::Or, 0, 1);
    }
}
fn leading_count(code: &mut Code, lane: u8, sign: bool, target: TargetCpu) {
    if sign {
        packed(code, Host::Xor, 1, 1);
        packed(code, compare_opcode(lane), 1, 0);
        packed(code, Host::Xor, 0, 1);
    }
    if vector_extensions::leading_zeros(code, lane, 0, target) {
        if sign {
            lane_constant(code, 1, lane, 1);
            packed(code, Host::SubDword, 0, 1);
        }
        return;
    }
    packed(code, Host::Xor, 2, 2); // accumulated per-lane count
    packed(code, Host::Xor, 3, 3); // zero comparison operand
    let mut step = lane * 4;
    while step != 0 {
        packed(code, Host::MoveAligned, 1, 0);
        logical_shift(code, 1, lane, lane * 8 - step, false, target);
        packed(code, equal_opcode(lane), 1, 3); // missing high half: increment and shift
        lane_constant(code, 4, lane, u64::from(step));
        packed(code, Host::And, 4, 1);
        packed(code, add_opcode(lane), 2, 4);
        packed(code, Host::MoveAligned, 4, 0);
        logical_shift(code, 4, lane, step, true, target);
        packed(code, Host::Xor, 4, 0);
        packed(code, Host::And, 4, 1);
        packed(code, Host::Xor, 0, 4);
        step /= 2;
    }
    packed(code, equal_opcode(lane), 0, 3);
    lane_constant(code, 1, lane, 1);
    packed(code, Host::And, 0, 1);
    packed(code, add_opcode(lane), 2, 0);
    if sign {
        packed(
            code,
            match lane {
                1 => Host::SubByte,
                2 => Host::SubWord,
                4 => Host::SubDword,
                _ => unreachable!(),
            },
            2,
            1,
        );
    }
    packed(code, Host::MoveAligned, 0, 2);
}

pub(super) fn emit(code: &mut Code, operand: &Operand, target: TargetCpu) {
    let Operand {
        kind,
        destination,
        left,
        right,
        bytes,
        lane_bytes,
    } = *operand;
    if (matches!(kind, Kind::Bits(BitCount::Population))
        && vector_extensions::population_supported(1, target))
        || (matches!(
            kind,
            Kind::Bits(BitCount::LeadingZeros | BitCount::LeadingSign)
        ) && vector_extensions::leading_supported(lane_bytes, target))
    {
        code.asm().vzeroupper();
    }
    if let Kind::Combine {
        operation: operation @ (Combine::Minimum { .. } | Combine::Maximum { .. }),
        shape: Shape::Lanes,
    } = kind
        && lane_bytes <= 4
    {
        let (signed, maximum) = match operation {
            Combine::Minimum { signed } => (signed, false),
            Combine::Maximum { signed } => (signed, true),
            _ => unreachable!(),
        };
        let native = native_minmax(lane_bytes, signed, maximum);
        if target >= native.requires() {
            // One PMIN/PMAX from the sources where they are.
            let left = super::vector_source(code, left, 0);
            let right = super::vector_source(code, right, 1);
            if bytes == 16 {
                let destination = super::vector_destination(code, destination, 0);
                super::binary(code, native, destination, left, right, true, target);
            } else {
                super::three::three(code, native, 0, left, right, target);
                let destination = super::vector_destination(code, destination, 0);
                super::finish_from(code, destination, 0, 8);
            }
            return;
        }
    }
    if !matches!(
        kind,
        Kind::Combine {
            shape: Shape::Pairs,
            ..
        }
    ) {
        read_vector(code, left, 0);
    }
    let mut scalar = false;
    match kind {
        Kind::Combine { operation, shape } => match shape {
            Shape::Lanes => {
                read_vector(code, right, 1);
                combine(code, operation, lane_bytes, target);
            }
            Shape::Pairs => {
                pair_operands(code, operand);
                combine(code, operation, lane_bytes, target);
            }
            Shape::Across => {
                across(code, operation, lane_bytes, bytes, target);
                scalar = true;
            }
        },
        Kind::Bits(operation) => match operation {
            BitCount::Population => bit_population(code, target),
            BitCount::Reverse => bit_reverse(code, target),
            BitCount::LeadingZeros => leading_count(code, lane_bytes, false, target),
            BitCount::LeadingSign => leading_count(code, lane_bytes, true, target),
            BitCount::Not => {
                packed(code, Host::EqualDword, 1, 1);
                packed(code, Host::Xor, 0, 1);
            }
        },
    }
    if !scalar && bytes == 8 {
        packed(code, Host::MoveQword, 0, 0);
    }
    commit(code, destination, 0);
}

// The test runs generated code on the host, which needs mmap.
#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux")
))]
mod tests {
    use super::*;
    use crate::GuestState;

    /// UMINV and UMAXV of bytes and halfwords, both widths, run on the host
    /// for each target tier and agree with Arm's definition: the extremum
    /// in the low lane, every other bit clear.
    #[test]
    fn unsigned_extrema_across_lanes_run_as_arm_defines_them() {
        use core::ffi::c_void;
        unsafe extern "C" {
            fn mmap(
                address: *mut c_void,
                bytes: usize,
                protection: i32,
                flags: i32,
                fd: i32,
                offset: i64,
            ) -> *mut c_void;
            fn mprotect(address: *mut c_void, bytes: usize, protection: i32) -> i32;
            fn munmap(address: *mut c_void, bytes: usize) -> i32;
        }
        #[cfg(target_os = "macos")]
        const MAP_ANONYMOUS: i32 = 0x1000;
        #[cfg(target_os = "linux")]
        const MAP_ANONYMOUS: i32 = 0x20;
        const PAGE: usize = 4096;
        const SOURCE: u8 = 1;
        const DESTINATION: u8 = 2;
        fn run(code: &[u8], state: &mut GuestState) {
            // SAFETY: private anonymous memory, RX after the copy, unmapped
            // after the call; the code touches only `state`.
            unsafe {
                let page = mmap(core::ptr::null_mut(), PAGE, 3, 2 | MAP_ANONYMOUS, -1, 0);
                assert_ne!(page as usize, usize::MAX);
                core::ptr::copy_nonoverlapping(code.as_ptr(), page.cast(), code.len());
                assert_eq!(mprotect(page, PAGE, 5), 0);
                let entry: unsafe extern "sysv64" fn(*mut GuestState) = core::mem::transmute(page);
                entry(state);
                assert_eq!(munmap(page, PAGE), 0);
            }
        }
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut random = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };
        for target in [TargetCpu::Baseline, TargetCpu::detect()] {
            for maximum in [false, true] {
                for (lane_bytes, bytes) in [(1u8, 16u8), (1, 8), (2, 16), (2, 8)] {
                    let operation = if maximum {
                        Combine::Maximum { signed: false }
                    } else {
                        Combine::Minimum { signed: false }
                    };
                    let operand = Operand {
                        kind: Kind::Combine {
                            operation,
                            shape: Shape::Across,
                        },
                        destination: DESTINATION,
                        left: SOURCE,
                        right: SOURCE,
                        bytes,
                        lane_bytes,
                    };
                    let mut code = Code::new();
                    emit(&mut code, &operand, target);
                    code.asm().ret();
                    for sample in 0..256 {
                        // Edge vectors: all zeros, all ones and lone extrema.
                        let value = match sample {
                            0 => [0, 0],
                            1 => [u64::MAX, u64::MAX],
                            2 => [u64::MAX, u64::MAX - 1],
                            3 => [1, 0],
                            _ => [random(), random()],
                        };
                        let mut state = GuestState::default();
                        state.v[usize::from(SOURCE)] = value;
                        state.v[usize::from(DESTINATION)] = [random(), random()];
                        run(&code, &mut state);
                        let bits = u32::from(lane_bytes) * u8::BITS;
                        let lanes = (0..u32::from(bytes / lane_bytes)).map(|index| {
                            let at = index * bits;
                            let word = value[(at / u64::BITS) as usize] >> (at % u64::BITS);
                            word & ((1u64 << bits) - 1)
                        });
                        let expected = if maximum { lanes.max() } else { lanes.min() };
                        assert_eq!(
                            state.v[usize::from(DESTINATION)],
                            [expected.unwrap(), 0],
                            "{target:?} {maximum} {lane_bytes} {bytes} {value:x?}"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn packed_bit_extensions_match_each_byte_and_clear_upper_lanes() {
        use crate::context::ExecutionContext;
        use crate::x86_64::asm::vector_extensions::execution::Native;
        let host = TargetCpu::detect();
        for target in [
            TargetCpu::Baseline,
            TargetCpu::Ssse3,
            TargetCpu::Gfni,
            TargetCpu::Avx512f.with_avx512vl().with_avx512bitalg(),
        ] {
            if !host.supports(target) {
                continue;
            }
            for operation in [BitCount::Population, BitCount::Reverse] {
                for bytes in [8u8, 16] {
                    for destination in [1u8, 2] {
                        let operand = Operand {
                            kind: Kind::Bits(operation),
                            destination,
                            left: 1,
                            right: 1,
                            bytes,
                            lane_bytes: 1,
                        };
                        let mut code = Code::new();
                        emit(&mut code, &operand, target);
                        code.asm().ret();
                        let native = Native::new(&code);
                        for first in 0..=u8::MAX {
                            let input: [u8; 16] =
                                core::array::from_fn(|i| first.wrapping_add((i * 37) as u8));
                            let expected: [u8; 16] = core::array::from_fn(|i| {
                                if i >= usize::from(bytes) {
                                    0
                                } else if operation == BitCount::Reverse {
                                    input[i].reverse_bits()
                                } else {
                                    input[i].count_ones() as u8
                                }
                            });
                            let words = |x: [u8; 16]| {
                                [
                                    u64::from_le_bytes(x[..8].try_into().unwrap()),
                                    u64::from_le_bytes(x[8..].try_into().unwrap()),
                                ]
                            };
                            let mut state = GuestState {
                                fpsr: 0x0800_001f,
                                nzcv: 0xa000_0000,
                                ..GuestState::default()
                            };
                            state.v[1] = words(input);
                            if destination != 1 {
                                state.v[2] = [u64::MAX; 2];
                            }
                            native.run(&mut state, &mut ExecutionContext::default());
                            assert_eq!(
                                state.v[usize::from(destination)],
                                words(expected),
                                "{target:?} {operation:?} {bytes} {destination} {first}"
                            );
                            assert_eq!(state.fpsr, 0x0800_001f);
                            assert_eq!(state.nzcv, 0xa000_0000);
                            if destination != 1 {
                                assert_eq!(state.v[1], words(input));
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn packed_dword_leading_counts_match_sign_and_zero_boundaries() {
        use crate::context::ExecutionContext;
        use crate::x86_64::asm::vector_extensions::execution::Native;
        let target = TargetCpu::Avx512f.with_avx512vl().with_avx512cd();
        if !TargetCpu::detect().supports(target) {
            return;
        }
        for operation in [BitCount::LeadingZeros, BitCount::LeadingSign] {
            for bytes in [8u8, 16] {
                for destination in [1u8, 2] {
                    let operand = Operand {
                        kind: Kind::Bits(operation),
                        destination,
                        left: 1,
                        right: 1,
                        bytes,
                        lane_bytes: 4,
                    };
                    let mut code = Code::new();
                    emit(&mut code, &operand, target);
                    let mut baseline = Code::new();
                    emit(&mut baseline, &operand, TargetCpu::Baseline);
                    assert!(code.len() < baseline.len());
                    assert!(code.instructions() < baseline.instructions());
                    code.asm().ret();
                    let native = Native::new(&code);
                    for bit in 0..32 {
                        for value in [0u32, u32::MAX, 1u32 << bit, !(1u32 << bit)] {
                            let input = [
                                value,
                                value.rotate_left(7),
                                !value,
                                value.wrapping_mul(0x9e3779b9),
                            ];
                            let mut raw = [0u8; 16];
                            let mut expected = [0u8; 16];
                            for i in 0..4 {
                                raw[i * 4..i * 4 + 4].copy_from_slice(&input[i].to_le_bytes());
                                if i * 4 < usize::from(bytes) {
                                    let n = if operation == BitCount::LeadingSign {
                                        let normalized = if (input[i] as i32) < 0 {
                                            !input[i]
                                        } else {
                                            input[i]
                                        };
                                        normalized.leading_zeros() - 1
                                    } else {
                                        input[i].leading_zeros()
                                    };
                                    expected[i * 4..i * 4 + 4].copy_from_slice(&n.to_le_bytes());
                                }
                            }
                            let words = |x: [u8; 16]| {
                                [
                                    u64::from_le_bytes(x[..8].try_into().unwrap()),
                                    u64::from_le_bytes(x[8..].try_into().unwrap()),
                                ]
                            };
                            let mut state = GuestState {
                                fpsr: 0x0800_001f,
                                nzcv: 0xa000_0000,
                                ..GuestState::default()
                            };
                            state.v[1] = words(raw);
                            if destination != 1 {
                                state.v[2] = [u64::MAX; 2];
                            }
                            native.run(&mut state, &mut ExecutionContext::default());
                            assert_eq!(
                                state.v[usize::from(destination)],
                                words(expected),
                                "{operation:?} {bytes} {destination} {value:08x}"
                            );
                            assert_eq!(state.fpsr, 0x0800_001f);
                            assert_eq!(state.nzcv, 0xa000_0000);
                            if destination != 1 {
                                assert_eq!(state.v[1], words(raw));
                            }
                        }
                    }
                }
            }
        }
    }
}
