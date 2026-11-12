//! Emitted comparison and conversion blocks against the exact software
//! completion, over special values, mixed lanes and FPCR modes: every
//! compiled result must equal [`complete`]'s, and every helper exit must
//! leave the state untouched.
use super::super::*;
use crate::execution::platform::Mapping;
use stela_compiler::{
    GuestAddress, TargetCpu,
    aarch64::{FloatingPointOperation, floating::*},
    block::{BlockProfile, CompilationPlan, ExitKind, Features},
};

const PC: u64 = 0x1000;
const PREFIX: u32 = 0x9100_0508; // ADD X8,X8,#1
const SUFFIX: u32 = 0x9100_0529; // ADD X9,X9,#1
const Q: u32 = 1 << 30;
const SZ: u32 = 1 << 22;

fn specials(format: Format) -> Vec<u64> {
    let (sign, quiet, exponent_shift) = match format {
        Format::Half => (0x8000u64, 0x0200u64, 10),
        Format::Single => (0x8000_0000, 0x0040_0000, 23),
        Format::Double => (0x8000_0000_0000_0000, 0x0008_0000_0000_0000, 52),
    };
    let infinity = (sign - 1) & !((1u64 << exponent_shift) - 1);
    let one = match format {
        Format::Half => 0x3c00,
        Format::Single => 0x3f80_0000,
        Format::Double => 0x3ff0_0000_0000_0000,
    };
    let minimum_normal = 1u64 << exponent_shift;
    let positive = [
        0,
        infinity,
        infinity | quiet,
        infinity | 1,
        1,
        minimum_normal - 1,
        minimum_normal,
        one,
        one + 1,
        one | (1 << (exponent_shift - 1)),
        infinity - 1,
        one + (2 << exponent_shift),
    ];
    positive
        .iter()
        .flat_map(|&value| [value, value | sign])
        .collect()
}

fn targets() -> Vec<TargetCpu> {
    let host = stela_compiler::x86_64::modeled::host();
    [TargetCpu::Baseline, host]
        .into_iter()
        .filter(|target| host.supports(*target))
        .collect()
}

fn compile(word: u32, target: TargetCpu) -> Mapping {
    let source: Vec<_> = [PREFIX, word, SUFFIX]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
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
    unsafe {
        core::arch::asm!("stmxcsr [{p}]", p = in(reg) &mut value, options(nostack, preserves_flags));
    }
    value
}

/// Run `mapping` on `state` and check it against [`complete`]. Returns
/// whether the block completed the instruction itself.
fn check(
    word: u32,
    mapping: &Mapping,
    operation: &FloatingPointOperation,
    input: &GuestState,
) -> bool {
    let mut state = input.clone();
    let before = mxcsr();
    // SAFETY: this host's owned RX image holds three non-memory instructions
    // under the one-pointer block ABI; the state is private.
    let exit = unsafe { mapping.call_block(&mut state) }.unwrap();
    // The control must survive. The six status bits may gain flags (UCOMISS
    // sets IE for an SNaN): no guest state derives from them, and every
    // reader of host status installs a clean control first.
    const MXCSR_STATUS: u32 = 0x3f;
    assert_eq!(
        mxcsr() & !MXCSR_STATUS,
        before & !MXCSR_STATUS,
        "MXCSR {word:#010x}"
    );
    let mut expected = input.clone();
    expected.x[8] += 1;
    let native = exit.kind == ExitKind::Fallthrough as u64;
    if native {
        // The block may end after the instruction, before the suffix.
        assert!([PC + 8, PC + 12].contains(&exit.next_pc), "{word:#010x}");
        if exit.next_pc == PC + 12 {
            expected.x[9] += 1;
        }
        let completed = complete(&mut expected, operation);
        assert!(
            completed.is_ok(),
            "{word:#010x} completed natively under an unsupported control: {input:x?}"
        );
    } else {
        assert_eq!(exit.kind, ExitKind::FloatingPoint as u64, "{word:#010x}");
        assert_eq!(exit.next_pc, PC + 4, "{word:#010x}");
    }
    assert_eq!(
        state, expected,
        "{word:#010x} native={native} fpcr={:#x} input v1={:x?} v2={:x?} v0={:x?} nzcv={:#x}",
        input.fpcr, input.v[1], input.v[2], input.v[0], input.nzcv
    );
    native
}

const MODES: [u32; 9] = [
    0,
    FPCR_FZ,
    FPCR_FZ16,
    FPCR_AH,
    FPCR_DN,
    FPCR_FIZ,
    FPCR_AH | FPCR_FZ | FPCR_FIZ,
    FPCR_FZ | FPCR_FZ16 | FPCR_DN | FPCR_AH | FPCR_NEP,
    FPCR_NEP | (3 << FPCR_ROUNDING_SHIFT),
];

/// Fill `register` with `values`, lane `i` taking `values[i % len]`.
fn fill(state: &mut GuestState, register: u8, format: Format, values: &[u64]) {
    let lanes = 128 / format.bits();
    for lane in 0..lanes {
        write_lane(
            &mut state.v[usize::from(register)],
            format.bits(),
            lane,
            values[lane as usize % values.len()],
        );
    }
}

/// Run every word over every ordered pair of special values in its first
/// lane, the remaining lanes holding the pairs that follow, under every
/// mode, with and without IXC, on every target. Returns (native, helper).
fn sweep(words: &[(u32, Format)]) -> (usize, usize) {
    let mut counts = (0, 0);
    for target in targets() {
        for &(word, format) in words {
            let operation = FloatingPointOperation::decode(word).unwrap().unwrap();
            let mapping = compile(word, target);
            let values = specials(format);
            let count = values.len();
            for left in 0..count {
                for right in 0..count {
                    let lanes = 128 / format.bits() as usize;
                    let lefts: Vec<_> = (0..lanes).map(|i| values[(left + i) % count]).collect();
                    let rights: Vec<_> = (0..lanes)
                        .map(|i| values[(right + 3 * i) % count])
                        .collect();
                    for &mode in &MODES {
                        for (fpsr, nzcv) in [(0, 0x6000_0000), (INEXACT, 0x9000_0000)] {
                            let mut state = GuestState {
                                x: std::array::from_fn(|i| 0x1234_5678_0000_0000 | i as u64),
                                v: std::array::from_fn(|i| {
                                    [0x0123_4567_89ab_cdef ^ i as u64, 0xfedc_ba98_7654_3210]
                                }),
                                fpcr: mode,
                                fpsr,
                                nzcv,
                                ..Default::default()
                            };
                            fill(&mut state, 1, format, &lefts);
                            fill(&mut state, 2, format, &rights);
                            if check(word, &mapping, &operation, &state) {
                                counts.0 += 1;
                            } else {
                                counts.1 += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    counts
}

fn vector_forms(single_double: &[u32], half: &[u32]) -> Vec<(u32, Format)> {
    let mut words = Vec::new();
    for &base in single_double {
        words.push((base, Format::Single));
        words.push((base | Q, Format::Single));
        words.push((base | Q | SZ, Format::Double));
    }
    for &base in half {
        words.push((base, Format::Half));
        words.push((base | Q, Format::Half));
    }
    words
}

#[test]
fn vector_register_and_zero_comparisons_match_completion() {
    // FCMEQ, FCMGE, FCMGT, FACGE, FACGT Vd,Vn,Vm with Rd=0, Rn=1, Rm=2.
    let register = vector_forms(
        &[
            0x0e22_e420,
            0x2e22_e420,
            0x2ea2_e420,
            0x2e22_ec20,
            0x2ea2_ec20,
        ],
        &[
            0x0e42_2420,
            0x2e42_2420,
            0x2ec2_2420,
            0x2e42_2c20,
            0x2ec2_2c20,
        ],
    );
    // FCMGT, FCMEQ, FCMLT, FCMGE, FCMLE Vd,Vn,#0.
    let zero = vector_forms(
        &[
            0x0ea0_c820,
            0x0ea0_d820,
            0x0ea0_e820,
            0x2ea0_c820,
            0x2ea0_d820,
        ],
        &[
            0x0ef8_c820,
            0x0ef8_d820,
            0x0ef8_e820,
            0x2ef8_c820,
            0x2ef8_d820,
        ],
    );
    let (native, helper) = sweep(&[register, zero].concat());
    assert!(native > 0 && helper > 0, "{native} {helper}");
}

#[test]
fn scalar_predicate_comparisons_match_completion() {
    let mut words = Vec::new();
    for base in [
        0x5e22_e420u32,
        0x7e22_e420,
        0x7ea2_e420,
        0x7e22_ec20,
        0x7ea2_ec20,
    ] {
        words.push((base, Format::Single));
        words.push((base | SZ, Format::Double));
    }
    for base in [
        0x5e42_2420u32,
        0x7e42_2420,
        0x7ec2_2420,
        0x7e42_2c20,
        0x7ec2_2c20,
    ] {
        words.push((base, Format::Half));
    }
    // Zero forms: FCMGT, FCMEQ, FCMLT, FCMGE, FCMLE Sd/Dd,Sn/Dn,#0, and Half.
    for base in [
        0x5ea0_c820u32,
        0x5ea0_d820,
        0x5ea0_e820,
        0x7ea0_c820,
        0x7ea0_d820,
    ] {
        words.push((base, Format::Single));
        words.push((base | SZ, Format::Double));
    }
    for base in [
        0x5ef8_c820u32,
        0x5ef8_d820,
        0x5ef8_e820,
        0x7ef8_c820,
        0x7ef8_d820,
    ] {
        words.push((base, Format::Half));
    }
    let (native, helper) = sweep(&words);
    assert!(native > 0 && helper > 0, "{native} {helper}");
}

#[test]
fn flag_comparisons_match_completion() {
    let mut words = Vec::new();
    for (format, kind) in [(Format::Single, 0), (Format::Double, 1), (Format::Half, 3)] {
        let base = 0x1e20_0000 | kind << 22;
        // FCMP/FCMPE Sn,Sm and #0.0 with Rn=1, Rm=2.
        for opc in [0x00, 0x08, 0x10, 0x18] {
            let right = if opc & 0x08 == 0 { 2 << 16 } else { 0 };
            words.push((base | 0x2000 | right | 1 << 5 | opc, format));
        }
        // FCCMP/FCCMPE with conditions EQ, GE, LT, HI and fallback NZCV 0b0110.
        for condition in [0x0u32, 0xa, 0xb, 0x8] {
            for signaling in [0, 0x10] {
                words.push((
                    base | 0x0400 | 2 << 16 | condition << 12 | 1 << 5 | signaling | 6,
                    format,
                ));
            }
        }
    }
    let (native, helper) = sweep(&words);
    assert!(native > 0 && helper > 0, "{native} {helper}");
}

#[test]
fn half_conversions_match_completion() {
    let words = [
        // FCVT S0,H1; FCVT D0,H1 (the left source is V1).
        (0x1ee2_4020, Format::Half),
        (0x1ee2_c020, Format::Half),
        // FCVTN V0.4H,V1.4S; FCVTN2 V0.8H,V1.4S.
        (0x0e21_6820, Format::Single),
        (0x4e21_6820, Format::Single),
        // FCVT H0,S1; FCVT H0,D1; FCVTL V0.4S,V1.4H; FCVTL2.
        (0x1e23_c020, Format::Single),
        (0x1e63_c020, Format::Double),
        (0x0e21_7820, Format::Half),
        (0x4e21_7820, Format::Half),
    ];
    let (native, helper) = sweep(&words);
    assert!(native > 0 && helper > 0, "{native} {helper}");
}
