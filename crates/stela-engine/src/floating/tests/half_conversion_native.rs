//! Emitted Half conversions against exact completion on each host recipe.
use super::super::*;
use crate::execution::platform::Mapping;
use stela_compiler::{
    GuestAddress, TargetCpu,
    aarch64::floating::*,
    block::{BlockProfile, CompilationPlan, ExitKind, Features},
};

const PC: u64 = 0x1000;
const PREFIX: u32 = 0x9100_0508; // ADD X8,X8,#1
const SUFFIX: u32 = 0x9100_0529; // ADD X9,X9,#1
const SOURCE: usize = 1;

/// llvm-mc encodings, Rd = 0 and Rn = 1: FCVTZS 8H and 4H, FCVTZU, FCVTNS,
/// FCVTMS, FCVTPU, FCVTAS, SCVTF 8H, UCVTF 4H; FCVTZS, FCVTZU, FCVTNS, SCVTF
/// and UCVTF H; FCVTZS W and X, FCVTZU W and X, FCVTNS W, FCVTMU X; SCVTF and
/// UCVTF from W and X; FCVTN, FCVTN2; FCVT from S and D; fixed-point FCVTZS
/// and SCVTF, which keep completion.
const FORMS: [u32; 30] = [
    0x4ef9_b820,
    0x0ef9_b820,
    0x6ef9_b820,
    0x4e79_a820,
    0x0e79_b820,
    0x6ef9_a820,
    0x4e79_c820,
    0x4e79_d820,
    0x2e79_d820,
    0x5ef9_b820,
    0x7ef9_b820,
    0x5e79_a820,
    0x5e79_d820,
    0x7e79_d820,
    0x1ef8_0020,
    0x9ef8_0020,
    0x1ef9_0020,
    0x9ef9_0020,
    0x1ee0_0020,
    0x9ef1_0020,
    0x1ee2_0020,
    0x9ee2_0020,
    0x1ee3_0020,
    0x9ee3_0020,
    0x0e21_6820,
    0x4e21_6820,
    0x1e23_c020,
    0x1e63_c020,
    0x4f1c_fc20,
    0x1ec2_f420,
];

/// Inline on every target from the fields: FCVTN, FCVTN2 and FCVT Hd, Sn;
/// FCVT{N,M,P,A,Z}{S,U} 8H, 4H and H; SCVTF and UCVTF 8H, 4H and H.
const FIELD_RECIPES: [u32; 17] = [
    0x4e79_a820,
    0x0e79_b820,
    0x6ef9_a820,
    0x4e79_c820,
    0x5e79_a820,
    0x0e21_6820,
    0x4e21_6820,
    0x1e23_c020,
    0x4ef9_b820,
    0x0ef9_b820,
    0x6ef9_b820,
    0x5ef9_b820,
    0x7ef9_b820,
    0x4e79_d820,
    0x2e79_d820,
    0x5e79_d820,
    0x7e79_d820,
];

fn compile(word: u32, target: TargetCpu) -> Mapping {
    compile_words(&[PREFIX, word, SUFFIX], target)
}

fn compile_words(words: &[u32], target: TargetCpu) -> Mapping {
    let source: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let block = CompilationPlan::new(
        &source,
        GuestAddress(PC),
        &stela_compiler::Options::new(BlockProfile::new(Features::SYSTEM_MODE), target),
    )
    .unwrap()
    .compile()
    .unwrap();
    Mapping::new(&stela_compiler::x86_64::pinned::entered(
        &block.native.x86_64_code,
    ))
    .unwrap()
}

fn mxcsr() -> u32 {
    let mut value = 0;
    // SAFETY: valid writable output on this baseline SSE2 host.
    unsafe { core::arch::asm!("stmxcsr [{}]", in(reg) &raw mut value) };
    value
}

/// Run the block; the state must match exact completion, on the inline path
/// or after a helper exit that left only the prefix's effects. A conversion
/// that ends its block exits after itself; one in the block body runs on
/// through the suffix.
fn check(mapping: &Mapping, operation: &FloatingPointOperation, state: GuestState) -> bool {
    let mut expected = state.clone();
    expected.x[8] += 1;
    let prefix = expected.clone();
    complete(&mut expected, operation).unwrap();
    let mut actual = state;
    let saved = mxcsr();
    // SAFETY: the owned block contains only non-memory instructions, compiled
    // for this host. State satisfies its ordinary one-pointer block ABI.
    let exit = unsafe { mapping.call_block(&mut actual) }.unwrap();
    assert_eq!(mxcsr(), saved, "MXCSR {operation:?}");
    let inline = exit.kind == ExitKind::Fallthrough as u64;
    if inline {
        if exit.next_pc == PC + 12 {
            expected.x[9] += 1;
        } else {
            assert_eq!(exit.next_pc, PC + 8, "{operation:?}");
        }
    } else {
        assert_eq!(exit.kind, ExitKind::FloatingPoint as u64, "{operation:?}");
        assert_eq!(exit.next_pc, PC + 4, "{operation:?}");
        assert_eq!(actual, prefix, "helper exit {operation:?}");
        complete(&mut actual, operation).unwrap();
    }
    assert_eq!(actual, expected, "{operation:?} inline={inline}");
    inline
}

/// A random lane pattern for the input family `kind`.
fn limb(kind: u32, next: &mut impl FnMut() -> u64) -> u64 {
    let word = |next: &mut dyn FnMut() -> u64| next() & 0xffff;
    (0..4).fold(0, |limb, lane| match kind % 6 {
        // Half lanes: normal in integer range, then any class.
        0 => limb | (next() & 0x83ff | (8 + next() % 16) << 10) << (16 * lane),
        1 => limb | word(next) << (16 * lane),
        // Small signed integers, and raw ones.
        2 => limb | (next() & 63).wrapping_sub(32) << (16 * lane) & 0xffff << (16 * lane),
        // Single lanes in Half's range, or just past it.
        3 if lane % 2 == 0 => {
            limb | (next() & 0x807f_ffff | (113 + next() % 30) << 23) << (16 * lane)
        }
        // Double lanes.
        4 if lane == 0 => next() & 0x800f_ffff_ffff_ffff | (1008 + next() % 32) << 52,
        3 | 4 => limb,
        _ => next(),
    })
}

#[test]
fn half_conversions_match_completion_on_each_host_recipe() {
    let host = stela_compiler::x86_64::modeled::host();
    let mut random = 0x59f1_11f1_923f_82a4_u64;
    let mut next = move || {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        random
    };
    let targets = [
        TargetCpu::Baseline,
        TargetCpu::X86_64_V3,
        TargetCpu::X86_64_V4.with_avx512fp16(),
    ];
    for target in targets {
        if !host.supports(target) {
            continue;
        }
        let mut inline = 0;
        let mut inline_recipes = 0;
        for word in FORMS {
            let operation = FloatingPointOperation::decode(word).unwrap().unwrap();
            let mapping = compile(word, target);
            for sample in 0..96u32 {
                let mut state = GuestState {
                    fpcr: (sample & 3) << FPCR_ROUNDING_SHIFT
                        | [0, FPCR_NEP, FPCR_FZ16, FPCR_DN, FPCR_AHP][(sample % 5) as usize],
                    fpsr: next() as u32 & EXCEPTIONS,
                    ..GuestState::default()
                };
                for limb_at in state.v.as_flattened_mut() {
                    *limb_at = next();
                }
                let kind = sample / 4;
                state.v[SOURCE] = [limb(kind, &mut next), limb(kind, &mut next)];
                state.x[SOURCE] = match kind % 3 {
                    0 => next() & 0xfff,
                    1 => (next() & 0xfff).wrapping_neg(),
                    _ => next() >> (next() % 64),
                };
                let completed = check(&mapping, &operation, state);
                inline += u32::from(completed);
                inline_recipes += u32::from(completed && !FIELD_RECIPES.contains(&word));
            }
        }
        std::println!("{} inline={inline}", target.id());
        // The field recipes run on SSE2; the other forms need AVX512-FP16.
        assert_ne!(inline, 0, "{target:?}");
        assert_eq!(
            inline_recipes != 0,
            target.supports(TargetCpu::X86_64_V4.with_avx512fp16()),
            "{target:?}"
        );
    }
}

#[test]
fn half_widening_keeps_all_finite_lanes_native_on_sse2() {
    for (word, part) in [(0x0e21_7820, 0), (0x4e21_7820, 1)] {
        let operation = FloatingPointOperation::decode(word).unwrap().unwrap();
        let mapping = compile_words(&[PREFIX, word], TargetCpu::Baseline);
        for first in (0..=u16::MAX).step_by(4) {
            let mut state = GuestState {
                fpcr: FPCR_AH | FPCR_AHP | FPCR_FZ | FPCR_FIZ | FPCR_FZ16 | FPCR_DN,
                fpsr: FPSR_QC,
                ..GuestState::default()
            };
            // The unused half contains signaling NaNs, including for FCVTL2.
            state.v[SOURCE] = [0x7c01_7c01_7c01_7c01; 2];
            state.v[SOURCE][part] = (0..4).fold(0, |bits, lane| {
                bits | u64::from(first + lane) << (16 * lane)
            });
            let finite = first & 0x7c00 != 0x7c00;
            assert_eq!(check(&mapping, &operation, state), finite, "{first:#06x}");
        }

        for special in [0x7c00, 0xfc00, 0x7c01, 0xfe00] {
            let mut state = GuestState::default();
            state.v[SOURCE][part] = special;
            assert!(!check(&mapping, &operation, state));
        }

        let mut state = GuestState {
            fpcr: FPCR_EXCEPTION_ENABLE_MASK,
            ..GuestState::default()
        };
        state.v[0] = [u64::MAX; 2];
        assert!(complete(&mut state.clone(), &operation).is_err());
        let mut expected = state.clone();
        expected.x[8] += 1;
        let before_mxcsr = mxcsr();
        // SAFETY: compiled for the host baseline and only accesses its frame.
        let exit = unsafe { mapping.call_block(&mut state) }.unwrap();
        assert_eq!(exit.kind, ExitKind::FloatingPoint as u64);
        assert_eq!(exit.next_pc, PC + 4);
        assert_eq!(state, expected);
        assert_eq!(mxcsr(), before_mxcsr);
    }
}

#[test]
fn complementary_half_widening_preserves_each_instruction_boundary() {
    const LOW: u32 = 0x0e21_7820; // FCVTL V0.4S,V1.4H
    const HIGH: u32 = 0x4e21_7821; // FCVTL2 V1.4S,V1.8H
    const DESTRUCTIVE_LOW: u32 = 0x0e21_7821; // FCVTL V1.4S,V1.4H
    let normal = 0x3555_bc00_3c00_4000;
    let mixed = 0x8000_0000_83ff_0001;
    for profile in [
        BlockProfile::ORDINARY,
        BlockProfile::new(Features::SYSTEM_MODE),
    ] {
        let emit = |first, second| {
            let source: Vec<_> = [PREFIX, first, second, SUFFIX]
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            CompilationPlan::new(
                &source,
                GuestAddress(PC),
                &stela_compiler::Options::new(profile, TargetCpu::Baseline),
            )
            .unwrap()
            .compile()
            .unwrap()
            .native
        };
        let ungrouped = emit(DESTRUCTIVE_LOW, HIGH);
        for (first, second, first_part, grouped) in [
            (LOW, HIGH, 0, true),
            (0x4e21_7820, DESTRUCTIVE_LOW, 1, true),
            (DESTRUCTIVE_LOW, HIGH, 0, false),
        ] {
            let native = emit(first, second);
            assert_eq!(native.source_map.len(), 4);
            for (index, (pc, _)) in native.source_map.iter().enumerate() {
                assert_eq!(*pc, GuestAddress(PC + 4 * index as u64));
            }
            assert!(
                native
                    .source_map
                    .windows(2)
                    .all(|rows| rows[0].1 < rows[1].1)
            );
            if grouped {
                assert!(native.x86_64_code.len() < ungrouped.x86_64_code.len());
            }
            let code = if profile == BlockProfile::ORDINARY {
                native.x86_64_code
            } else {
                stela_compiler::x86_64::pinned::entered(&native.x86_64_code)
            };
            let mapping = Mapping::new(&code).unwrap();
            let [first, second] =
                [first, second].map(|word| FloatingPointOperation::decode(word).unwrap().unwrap());
            let modes = FPCR_FZ16 | FPCR_FZ | FPCR_FIZ | FPCR_DN;
            let mut first_failure = [normal; 2];
            first_failure[first_part] = 0x7c01;
            let mut second_failure = [normal; 2];
            second_failure[1 - first_part] = 0x7e00;
            let cases = [
                ([normal; 2], FPCR_AHP, None),
                ([mixed, normal], modes, None),
                ([normal; 2], FPCR_EXCEPTION_ENABLE_MASK, Some(PC + 4)),
                (first_failure, 0, Some(PC + 4)),
                (second_failure, 0, Some(PC + 8)),
            ];
            for (input, fpcr, failure) in cases {
                if !grouped && failure == Some(PC + 8) {
                    // The first write changes the input for this instruction.
                    continue;
                }
                let mut actual = GuestState {
                    fpcr,
                    fpsr: FPSR_QC,
                    ..GuestState::default()
                };
                actual.v[0] = [u64::MAX; 2];
                actual.v[SOURCE] = input;
                let mut expected = actual.clone();
                expected.x[8] += 1;
                if failure != Some(PC + 4) {
                    complete(&mut expected, &first).unwrap();
                }
                if failure.is_none() {
                    complete(&mut expected, &second).unwrap();
                    expected.x[9] += 1;
                }
                let saved_mxcsr = mxcsr();
                // SAFETY: the emitted block only accesses its owned state frame.
                let exit = unsafe { mapping.call_block(&mut actual) }.unwrap();
                assert_eq!(mxcsr(), saved_mxcsr);
                assert_eq!(actual, expected, "{profile:?} {first:?} {second:?}");
                assert_eq!(exit.next_pc, failure.unwrap_or(PC + 16));
                assert_eq!(
                    exit.kind,
                    if failure.is_some() {
                        ExitKind::FloatingPoint
                    } else {
                        ExitKind::Fallthrough
                    } as u64
                );
            }
        }
    }
}
