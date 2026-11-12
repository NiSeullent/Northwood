//! The Half <-> 16-bit integer field recipes (`half_integer`) against the
//! reference conversions on every target the host runs: every inline
//! completion matches the reference in its lanes and FPSR, every exit leaves
//! the state unchanged, and covered inputs must complete inline.
use crate::{
    GuestAddress, GuestState, Options, TargetCpu,
    aarch64::floating::{
        FPCR_AH, FPCR_AHP, FPCR_DN, FPCR_EXCEPTION_ENABLE_MASK, FPCR_FIZ, FPCR_FZ, FPCR_FZ16,
        FPCR_NEP, FPCR_ROUNDING_SHIFT, Format, INEXACT, INVALID,
        integer::{IntegerFormat, IntegerRounding, IntegerWidth, from_integer, to_integer},
    },
    block::{BlockProfile, CompilationPlan},
    x86_64::native::run,
};
use alloc::vec::Vec;

const PC: GuestAddress = GuestAddress(0x1000_0000);
const RETURN: u64 = 0x4000;
const D: usize = 3;
const N: usize = 4;
const UPPER: [u64; 2] = [0xa5a5_5a5a_c3c3_3c3c, 0x1234_5678_9abc_def0];

fn targets() -> Vec<TargetCpu> {
    let host = TargetCpu::detect();
    let mut targets: Vec<TargetCpu> = [
        TargetCpu::Baseline,
        TargetCpu::X86_64_V2,
        TargetCpu::X86_64_V2.with_f16c(),
        TargetCpu::X86_64_V3,
    ]
    .into_iter()
    .filter(|target| host.supports(*target))
    .collect();
    targets.push(host);
    targets
}

fn compile(word: u32, target: TargetCpu) -> Vec<u8> {
    let source = [word, 0xd65f_03c0].map(u32::to_le_bytes).concat();
    CompilationPlan::new(&source, PC, &Options::new(BlockProfile::ORDINARY, target))
        .unwrap()
        .compile()
        .unwrap()
        .native
        .x86_64_code
}

fn controls() -> Vec<u32> {
    let mut controls: Vec<u32> = (0..4).map(|mode| mode << FPCR_ROUNDING_SHIFT).collect();
    controls.extend([
        FPCR_AH,
        FPCR_FZ | FPCR_FIZ,
        FPCR_NEP,
        FPCR_DN | FPCR_FZ16,
        FPCR_FZ16 | FPCR_AH,
        FPCR_AHP,
        FPCR_EXCEPTION_ENABLE_MASK,
        1 << 12, // IXE alone
    ]);
    controls
}

/// Every 16-bit pattern once in some lane, then random mixtures weighted to
/// ordinary values and zeros.
fn cases() -> Vec<[u16; 8]> {
    let mut cases: Vec<[u16; 8]> = (0..8192u32)
        .map(|case| core::array::from_fn(|lane| (case * 8 + lane as u32) as u16))
        .collect();
    let mut seed = 0x1234_5678_9abc_def1_u64;
    for _ in 0..4096 {
        cases.push(core::array::from_fn(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            match seed % 4 {
                0 => (seed >> 8) as u16 & 0x7fff | (seed >> 40) as u16 & 0x8000,
                1 => 0x3c00 + (seed >> 8) as u16 % 0x3800,
                2 => 0,
                _ => (seed >> 24) as u16,
            }
        }));
    }
    cases
}

/// .4H, .8H and the SIMD scalar H form of the vector `base` word.
fn forms(base: u32) -> [(u32, usize); 3] {
    [(base, 4), (base | 1 << 30, 8), (base | 0x5000_0000, 1)]
}

/// Run `inputs` through `code` of `word`; compare with `reference` per
/// active lane. Returns whether it completed inline.
#[allow(clippy::too_many_arguments)]
fn check(
    code: &[u8],
    word: u32,
    target: TargetCpu,
    lanes: usize,
    inputs: [u16; 8],
    fpcr: u32,
    fpsr: u32,
    reference: &dyn Fn(u64) -> Option<(u64, u32)>,
) -> bool {
    let mut initial = GuestState {
        fpcr,
        fpsr,
        ..GuestState::default()
    };
    initial.x[30] = RETURN;
    let limb = |first: usize| {
        (first..first + 4).fold(0u64, |bits, lane| {
            bits | u64::from(inputs[lane]) << (16 * (lane - first))
        })
    };
    initial.v[N] = [limb(0), limb(4)];
    initial.v[D] = UPPER;
    let mut actual = initial.clone();
    let exit = run(code, &mut actual);
    if exit == PC.0 {
        assert_eq!(
            actual, initial,
            "{word:08x} {inputs:x?} {fpcr:#x} {target:?}"
        );
        return false;
    }
    assert_eq!(exit, RETURN, "{word:08x} {inputs:x?} {fpcr:#x} {target:?}");
    let mut exceptions = 0;
    let mut results = [0u64; 2];
    for (lane, &input) in inputs.iter().enumerate().take(lanes) {
        let (bits, flags) = reference(u64::from(input)).expect("no enabled traps");
        exceptions |= flags;
        results[lane / 4] |= bits << (16 * (lane % 4));
    }
    let mut expected = initial.clone();
    expected.fpsr |= exceptions;
    expected.v[D] = match lanes {
        1 if fpcr & FPCR_NEP != 0 => [UPPER[0] & !0xffff | results[0], UPPER[1]],
        _ => results,
    };
    assert_eq!(
        actual, expected,
        "{word:08x} {inputs:x?} {fpcr:#x} {target:?}"
    );
    true
}

fn run_forms(
    bases: [(u32, bool); 2],
    covered: &dyn Fn(bool, u16, u32) -> bool,
    reference: &dyn Fn(IntegerFormat, u32, u64) -> Option<(u64, u32)>,
) {
    let all = cases();
    let mut inline = 0usize;
    for target in targets() {
        for (base, signed) in bases {
            let integer = IntegerFormat {
                width: IntegerWidth::Bits16,
                signed,
            };
            for (word, lanes) in forms(base) {
                let word = word | (N as u32) << 5 | D as u32;
                let code = compile(word, target);
                for (index, &inputs) in all.iter().enumerate() {
                    let fpcrs: Vec<u32> = if index % 16 == 0 || index >= 8192 {
                        controls()
                    } else {
                        alloc::vec![0]
                    };
                    for fpcr in fpcrs {
                        for fpsr in [0, INEXACT] {
                            if fpsr != 0 && index % 4 != 0 {
                                continue;
                            }
                            let lane_reference = |bits| reference(integer, fpcr, bits);
                            let completed = check(
                                &code,
                                word,
                                target,
                                lanes,
                                inputs,
                                fpcr,
                                fpsr,
                                &lane_reference,
                            );
                            let must = inputs
                                .iter()
                                .take(lanes)
                                .all(|&bits| covered(signed, bits, fpcr));
                            assert!(
                                completed || !must,
                                "covered input exited: {word:08x} {inputs:x?} {fpcr:#x} {target:?}"
                            );
                            inline += usize::from(completed);
                        }
                    }
                }
            }
        }
    }
    assert!(inline > 100_000, "{inline}");
}

/// FCVTZS/FCVTZU and the rounding forms: every finite normal or zero lane
/// whose result fits completes inline under each rounding the word names.
fn half_to_integer_fields(mode: IntegerRounding, signed_base: u32) {
    let reference = move |integer, fpcr, bits| {
        to_integer(Format::Half, integer, 0, mode, fpcr, bits)
            .ok()
            .map(|outcome: crate::aarch64::floating::Outcome| (outcome.bits, outcome.exceptions))
    };
    run_forms(
        [(signed_base, true), (signed_base | 1 << 29, false)],
        &|signed, bits, fpcr| {
            let exponent = bits >> 10 & 31;
            let integer = IntegerFormat {
                width: IntegerWidth::Bits16,
                signed,
            };
            fpcr & FPCR_EXCEPTION_ENABLE_MASK == 0
                && exponent != 31
                && (exponent != 0 || bits & 0x7fff == 0)
                && (!signed || exponent < 30)
                && reference(integer, 0, u64::from(bits))
                    .is_some_and(|(_, flags)| flags & INVALID == 0)
        },
        &reference,
    );
}

#[test]
fn half_to_integer_fields_match_the_reference() {
    // FCVTZS, FCVTZU.
    half_to_integer_fields(IntegerRounding::TowardZero, 0x0ef9_b800);
}

#[test]
fn half_to_integer_nearest_even_fields_match_the_reference() {
    // FCVTNS, FCVTNU.
    half_to_integer_fields(IntegerRounding::NearestEven, 0x0e79_a800);
}

#[test]
fn half_to_integer_nearest_away_fields_match_the_reference() {
    // FCVTAS, FCVTAU.
    half_to_integer_fields(IntegerRounding::NearestAway, 0x0e79_c800);
}

#[test]
fn half_to_integer_toward_positive_fields_match_the_reference() {
    // FCVTPS, FCVTPU.
    half_to_integer_fields(IntegerRounding::TowardPositive, 0x0ef9_a800);
}

#[test]
fn half_to_integer_toward_negative_fields_match_the_reference() {
    // FCVTMS, FCVTMU.
    half_to_integer_fields(IntegerRounding::TowardNegative, 0x0e79_b800);
}

#[test]
fn half_from_integer_fields_match_the_reference() {
    // SCVTF, UCVTF.
    run_forms(
        [(0x0e79_d800, true), (0x2e79_d800, false)],
        &|signed, bits, fpcr| {
            let magnitude = if signed {
                i32::from(bits as i16).unsigned_abs()
            } else {
                u32::from(bits)
            };
            fpcr & (FPCR_EXCEPTION_ENABLE_MASK | 3 << FPCR_ROUNDING_SHIFT) == 0 && magnitude < 65520
        },
        &|integer, fpcr, bits| {
            from_integer(integer, Format::Half, 0, fpcr, bits)
                .ok()
                .map(|outcome| (outcome.bits, outcome.exceptions))
        },
    );
}

/// The forms outside the field recipes keep their old path: same results.
#[test]
fn other_half_integer_forms_still_match_the_reference() {
    // FCVTNS .8H.
    let word = 0x4e79_a800 | (N as u32) << 5 | D as u32;
    let integer = IntegerFormat {
        width: IntegerWidth::Bits16,
        signed: true,
    };
    for target in targets() {
        let code = compile(word, target);
        for inputs in cases().into_iter().step_by(64) {
            let reference = |bits| {
                to_integer(
                    Format::Half,
                    integer,
                    0,
                    IntegerRounding::NearestEven,
                    0,
                    bits,
                )
                .ok()
                .map(|outcome| (outcome.bits, outcome.exceptions))
            };
            check(&code, word, target, 8, inputs, 0, 0, &reference);
        }
    }
}
