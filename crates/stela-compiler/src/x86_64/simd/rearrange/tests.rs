use super::*;
use crate::{
    GuestState, Options,
    block::{BlockProfile, CompilationPlan, Features},
    x86_64::{native, pinned},
};

const PC: GuestAddress = GuestAddress(0x1af89b3ac);
const DESKTOP: [u32; 2] = [0x6e01_6000, 0x6e02_6001];
const SUFFIX: u32 = 0x9100_4231; // ADD X17,X17,#16

fn compile(words: &[u32], pc: GuestAddress, target: TargetCpu, system: bool) -> crate::NativeBlock {
    let source: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let profile = if system {
        BlockProfile::new(Features::SYSTEM_MODE)
    } else {
        BlockProfile::ORDINARY
    };
    CompilationPlan::new(&source, pc, &Options::new(profile, target))
        .unwrap()
        .compile()
        .unwrap()
        .native
}

fn reference(state: &mut GuestState, bytes: usize, offset: usize, registers: [usize; 3]) {
    let [d, n, m] = registers;
    let a = state.v[n].map(u64::to_le_bytes).concat();
    let b = state.v[m].map(u64::to_le_bytes).concat();
    let joined = [&a[..bytes], &b[..bytes]].concat();
    let mut result = [0; 16];
    result[..bytes].copy_from_slice(&joined[offset..offset + bytes]);
    state.v[d] = core::array::from_fn(|lane| {
        u64::from_le_bytes(result[lane * 8..lane * 8 + 8].try_into().unwrap())
    });
}

fn input() -> GuestState {
    let mut state = GuestState::default();
    for (register, value) in state.v.iter_mut().enumerate() {
        *value = core::array::from_fn(|half| {
            let base = (register * 19 + half * 8) as u8;
            u64::from_le_bytes(core::array::from_fn(|byte| base.wrapping_add(byte as u8)))
        });
    }
    state
}

#[test]
fn byte_extract_preserves_all_widths_offsets_and_aliases() {
    for target in [TargetCpu::Baseline, TargetCpu::Ssse3] {
        if !TargetCpu::detect().supports(target) {
            continue;
        }
        for bytes in [8, 16] {
            for offset in 0..bytes {
                for [d, n, m] in [[0, 1, 2], [1, 1, 2], [2, 1, 2], [0, 1, 1], [31, 31, 31]] {
                    let word = 0x2e00_0000
                        | ((bytes == 16) as u32) << 30
                        | (m as u32) << 16
                        | (offset as u32) << 11
                        | (n as u32) << 5
                        | d as u32;
                    let block = compile(&[word], PC, target, false);
                    let mut actual = input();
                    let mut expected = actual.clone();
                    reference(&mut expected, bytes, offset, [d, n, m]);
                    assert_eq!(native::run(&block.x86_64_code, &mut actual), PC.0 + 4);
                    assert_eq!(actual, expected, "{target:?} {word:08x}");
                }
            }
        }
    }
}

#[test]
fn desktop_extracts_forward_the_committed_left_source() {
    let second_pc = GuestAddress(PC.0 + 4);
    let suffix_pc = GuestAddress(PC.0 + 8);
    for system in [false, true] {
        for target in [TargetCpu::Baseline, TargetCpu::Ssse3] {
            let block = compile(&[DESKTOP[0], DESKTOP[1], SUFFIX], PC, target, system);
            let alone = compile(&[DESKTOP[1], SUFFIX], second_pc, target, system);
            let [(first, from), (second, middle), (suffix, end)] = block.source_map.as_slice()
            else {
                panic!("three source rows");
            };
            assert_eq!([*first, *second, *suffix], [PC, second_pc, suffix_pc]);
            assert!(*from < *middle && *middle < *end);
            assert!(end - middle < alone.source_map[1].1 - alone.source_map[0].1);
            let prefix = &block.x86_64_code[*from..*end];
            let alignments = prefix
                .windows(4)
                .filter(|bytes| *bytes == Host::AlignBytes.encoding())
                .count();
            assert_eq!(
                alignments,
                if target.supports(TargetCpu::Ssse3) {
                    2
                } else {
                    0
                }
            );
            assert!(
                !prefix
                    .windows(4)
                    .any(|bytes| bytes == Host::ByteShuffle.encoding())
            );

            if !TargetCpu::detect().supports(target) {
                continue;
            }
            let code = if system {
                pinned::entered(&block.x86_64_code)
            } else {
                block.x86_64_code
            };
            let mut actual = input();
            let mut expected = actual.clone();
            reference(&mut expected, 16, 12, [0, 0, 1]);
            reference(&mut expected, 16, 12, [1, 0, 2]);
            expected.x[17] += 16;
            assert_eq!(native::run(&code, &mut actual), PC.0 + 12);
            assert_eq!(actual, expected);
        }
    }
}

fn table_reference(state: &mut GuestState, word: u32) {
    let table = state.v[((word >> 5) & 31) as usize]
        .map(u64::to_le_bytes)
        .concat();
    let indices = state.v[((word >> 16) & 31) as usize]
        .map(u64::to_le_bytes)
        .concat();
    let width = if word & (1 << 30) == 0 { 8 } else { 16 };
    let mut result = [0; 16];
    for (destination, index) in result[..width].iter_mut().zip(indices) {
        *destination = table.get(usize::from(index)).copied().unwrap_or(0);
    }
    state.v[(word & 31) as usize] = core::array::from_fn(|lane| {
        u64::from_le_bytes(result[lane * 8..lane * 8 + 8].try_into().unwrap())
    });
}

#[test]
fn single_table_groups_share_indices_until_the_guest_overwrites_them() {
    const RIPGREP: [u32; 7] = [
        0x4e15_0016,
        0x4e15_0057,
        0x4e15_00d8,
        0x4e14_0039,
        0x4e14_007a,
        0x4e14_00fb,
        0x4e15_0215,
    ];
    const WRITES_INDEX: [u32; 3] = [0x4e15_0016, 0x0e15_0215, 0x4e15_0057];
    let start = GuestAddress(0x2dcef8);
    let target = TargetCpu::Ssse3;
    for system in [false, true] {
        for (words, prepared) in [(&RIPGREP[..], 3), (&WRITES_INDEX[..], 2)] {
            let block = compile(words, start, target, system);
            assert_eq!(block.source_map.len(), words.len());
            for (index, (pc, _)) in block.source_map.iter().enumerate() {
                assert_eq!(*pc, GuestAddress(start.0 + index as u64 * 4));
            }
            assert!(
                block
                    .source_map
                    .windows(2)
                    .all(|rows| rows[0].1 < rows[1].1)
            );
            assert_eq!(
                block
                    .x86_64_code
                    .windows(3)
                    .filter(|bytes| *bytes == Host::AddUnsignedSaturatingByte.encoding())
                    .count(),
                prepared,
            );

            if !TargetCpu::detect().supports(target) {
                continue;
            }
            let code = if system {
                pinned::entered(&block.x86_64_code)
            } else {
                block.x86_64_code
            };
            for base in (0..=u8::MAX).step_by(16) {
                let mut actual = input();
                for register in [20, 21] {
                    actual.v[register] = core::array::from_fn(|lane| {
                        u64::from_le_bytes(core::array::from_fn(|byte| {
                            base + (lane * 8 + byte) as u8
                        }))
                    });
                }
                let mut expected = actual.clone();
                for &word in words {
                    table_reference(&mut expected, word);
                }
                assert_eq!(
                    native::run(&code, &mut actual),
                    start.0 + (words.len() * 4) as u64
                );
                assert_eq!(actual, expected);
            }
        }
    }
}

#[test]
fn constants_load_their_exact_bytes_and_ones_runs_take_two_instructions() {
    let mut patterns: Vec<[u8; 16]> = alloc::vec![[0; 16], [0xff; 16]];
    for lane in [1u8, 2, 4, 8] {
        let bits = u32::from(lane) * 8;
        let mask = u64::MAX >> (64 - bits);
        for value in [
            1u64,
            0x7f,
            0x80,
            0xfe,
            1 << (bits - 1),
            (1 << (bits - 1)) - 1,
            1 << (bits - 2),
            0x5a5a_5a5a_5a5a_5a5a,
            u64::MAX << 3,
        ] {
            let bytes = (value & mask).to_le_bytes();
            patterns.push(core::array::from_fn(|i| bytes[i % usize::from(lane)]));
        }
    }
    patterns.push(core::array::from_fn(|i| i as u8));
    patterns.push(core::array::from_fn(|i| if i < 8 { 0xff } else { 0 }));
    for bytes in patterns {
        for xmm in [0, 1, 4, 6] {
            let mut code = Code::new();
            constant(&mut code, xmm, bytes);
            let length = code.len();
            commit(&mut code, 9, xmm);
            let mut state = GuestState::default();
            native::run(&code, &mut state);
            assert_eq!(
                state.v[9].map(u64::to_le_bytes).concat(),
                bytes,
                "{bytes:02x?} in xmm{xmm}"
            );
            if ones_run(&bytes).is_some() {
                // PCMPEQD and an immediate shift: four and five bytes.
                assert!(length <= 9, "{bytes:02x?}: {length} bytes");
            }
        }
    }
    // One per dword lane, the rounding bit of a rounding shift.
    let one_per_dword: [u8; 16] = core::array::from_fn(|i| u8::from(i % 4 == 0));
    assert_eq!(
        ones_run(&one_per_dword),
        Some((Some(ShiftOpcode::RightDword), 31))
    );
    assert_eq!(ones_run(&[0x7f; 16]), None);
}
