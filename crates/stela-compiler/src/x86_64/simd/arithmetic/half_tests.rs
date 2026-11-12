//! Half arithmetic lanes (`half::emit`) against the reference on every
//! target the host runs: every inline completion matches the reference in
//! its lanes and FPSR, every exit leaves the state unchanged, and normal
//! inputs with a normal result at least twice the minimum, and vector lanes
//! with a signed zero input beside a normal or zero one, must complete
//! inline.
use crate::{
    GuestAddress, GuestState, Options, TargetCpu,
    aarch64::floating::{
        BinaryOperation, FPCR_AH, FPCR_DN, FPCR_EXCEPTION_ENABLE_MASK, FPCR_FZ16,
        FPCR_ROUNDING_SHIFT, Format, INEXACT, evaluate,
    },
    block::{BlockProfile, CompilationPlan},
    x86_64::native::run,
};
use alloc::vec::Vec;

const PC: GuestAddress = GuestAddress(0x1000_0000);
const RETURN: u64 = 0x4000;
const D: u32 = 3;
const N: u32 = 4;
const M: u32 = 5;

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

/// (word without registers, operation, lanes; 1 is scalar).
const FORMS: &[(u32, BinaryOperation, usize)] = &[
    (0x6e40_3c00, BinaryOperation::Divide, 8),
    (0x2e40_3c00, BinaryOperation::Divide, 4),
    (0x1ee0_1800, BinaryOperation::Divide, 1),
    (0x4e40_1400, BinaryOperation::Add, 8),
    (0x6e40_1c00, BinaryOperation::Multiply, 8),
    (0x0e40_1400, BinaryOperation::Add, 4),
    (0x4ec0_1400, BinaryOperation::Subtract, 8),
    (0x2e40_1c00, BinaryOperation::Multiply, 4),
];

fn controls() -> Vec<u32> {
    let mut controls: Vec<u32> = (0..4).map(|mode| mode << FPCR_ROUNDING_SHIFT).collect();
    controls.extend([
        FPCR_FZ16,
        FPCR_AH,
        FPCR_DN,
        1 << 12, // IXE
        FPCR_EXCEPTION_ENABLE_MASK,
    ]);
    controls
}

fn half(random: &mut u64) -> u64 {
    *random ^= *random << 13;
    *random ^= *random >> 7;
    *random ^= *random << 17;
    match *random % 16 {
        0 | 5 => (*random >> 40 & 1) << 15,
        1 => 0x7c00,
        2 => *random >> 20 & 0x3ff,
        // Exact quotients and products: powers of two and small integers.
        3 => 0x3c00 | (*random >> 24 & 0x7) << 10,
        4 => 0x4000 + (*random >> 30 & 0x1f) * 0x80,
        _ => ((*random >> 16 & 0x7fff) % 0x7800 + 0x400) | ((*random >> 40 & 1) << 15),
    }
}

fn normal(bits: u64) -> bool {
    let exponent = bits >> 10 & 31;
    exponent != 0 && exponent != 31
}

#[test]
fn half_arithmetic_lanes_match_the_reference() {
    let mut random = 0x1357_9bdf_2468_ace0_u64;
    let mut inline = 0usize;
    for target in targets() {
        for &(base, operation, lanes) in FORMS {
            let word = base | M << 16 | N << 5 | D;
            let code = compile(word, target);
            for case in 0..6000 {
                let inputs: Vec<(u64, u64)> = (0..lanes)
                    .map(|_| (half(&mut random), half(&mut random)))
                    .collect();
                let pack = |pick: &dyn Fn(&(u64, u64)) -> u64| {
                    let mut limbs = [0u64; 2];
                    for (lane, pair) in inputs.iter().enumerate() {
                        limbs[lane / 4] |= pick(pair) << (16 * (lane % 4));
                    }
                    limbs
                };
                let controls: Vec<u32> = if case % 8 == 0 {
                    controls()
                } else {
                    alloc::vec![0]
                };
                for fpcr in controls {
                    for fpsr in [0, INEXACT] {
                        let mut initial = GuestState {
                            fpcr,
                            fpsr,
                            ..GuestState::default()
                        };
                        initial.x[30] = RETURN;
                        initial.v[N as usize] = pack(&|pair| pair.0);
                        initial.v[M as usize] = pack(&|pair| pair.1);
                        initial.v[D as usize] = [u64::MAX, u64::MAX];
                        let mut actual = initial.clone();
                        let exit = run(&code, &mut actual);
                        let outcomes: Vec<_> = inputs
                            .iter()
                            .map(|&(left, right)| {
                                evaluate(Format::Half, operation, fpcr, left, right)
                            })
                            .collect();
                        let covered = fpcr & FPCR_EXCEPTION_ENABLE_MASK == 0
                            && inputs
                                .iter()
                                .zip(&outcomes)
                                .all(|(&(left, right), outcome)| {
                                    let outcome = outcome.as_ref().unwrap();
                                    let bits = outcome.bits;
                                    let zero = |bits: u64| bits & 0x7fff == 0;
                                    let finite = |bits| normal(bits) || zero(bits);
                                    // Vector lanes with a zero input and no zero divisor.
                                    let zero_lane = lanes > 1
                                        && (zero(left) || zero(right))
                                        && finite(left)
                                        && finite(right)
                                        && !(operation == BinaryOperation::Divide && zero(right));
                                    // Overflow and underflow take the helper.
                                    zero_lane
                                        || normal(left)
                                            && normal(right)
                                            && normal(bits)
                                            && bits >> 10 & 31 >= 2
                                            && outcome.exceptions & 0b1100 == 0
                                });
                        if exit == PC.0 {
                            assert_eq!(actual, initial, "{word:08x} {inputs:x?} {fpcr:#x}");
                            assert!(
                                !covered,
                                "covered lanes exited: {word:08x} {inputs:x?} {fpcr:#x} {target:?}"
                            );
                            continue;
                        }
                        assert_eq!(exit, RETURN, "{word:08x} {inputs:x?}");
                        let mut expected = initial.clone();
                        let mut limbs = [0u64; 2];
                        for (lane, outcome) in outcomes.iter().enumerate() {
                            let outcome = outcome.as_ref().expect("no enabled traps");
                            expected.fpsr |= outcome.exceptions;
                            limbs[lane / 4] |= outcome.bits << (16 * (lane % 4));
                        }
                        expected.v[D as usize] = limbs;
                        assert_eq!(
                            actual, expected,
                            "{word:08x} {inputs:x?} {fpcr:#x} {fpsr:#x} {target:?}"
                        );
                        inline += 1;
                    }
                }
            }
        }
    }
    assert!(inline > 20_000, "{inline}");
}

#[derive(Clone, Copy)]
enum Pairing {
    Elementwise,
    Pairwise,
    Indexed,
}

/// Finite Half values weighted to what leaves the lane recipe: subnormals,
/// values near the subnormal boundary, exact cancellation partners, tiny
/// products and ordinary values; `partner` is the other operand of the lane.
fn finite_half(random: &mut u64, partner: Option<u64>) -> u64 {
    *random ^= *random << 13;
    *random ^= *random >> 7;
    *random ^= *random << 17;
    let sign = (*random >> 40 & 1) << 15;
    match *random % 10 {
        0 => sign | (*random >> 20 & 0x3ff),
        1 => sign | (0x0400 + (*random >> 20 & 0x3ff)),
        2 => partner.map_or(0, |bits| bits ^ 0x8000),
        3 => partner.unwrap_or(0),
        4 => sign | ((*random >> 20 & 0x7ff) + 0x1000),
        5 => sign,
        6 => partner.map_or(0, |bits| {
            (bits ^ 0x8000).wrapping_add(*random >> 30 & 3) & 0xffff
        }),
        _ => sign | ((*random >> 16 & 0x7fff) % 0x7800 + 0x400),
    }
}

/// The packed recipe (`half::packed`) completes every finite vector lane
/// inline under each rounding mode, including subnormal inputs, exact
/// cancellation and tiny or subnormal results, with Arm's results and
/// status; FZ16, AH, overflow and NaN or infinite inputs exit or match.
#[test]
fn packed_half_arithmetic_completes_every_finite_lane_inline() {
    const PACKED_FORMS: &[(u32, BinaryOperation, usize, Pairing)] = &[
        (0x4e40_1400, BinaryOperation::Add, 8, Pairing::Elementwise),
        (0x0e40_1400, BinaryOperation::Add, 4, Pairing::Elementwise),
        (
            0x4ec0_1400,
            BinaryOperation::Subtract,
            8,
            Pairing::Elementwise,
        ),
        (
            0x6e40_1c00,
            BinaryOperation::Multiply,
            8,
            Pairing::Elementwise,
        ),
        (
            0x2e40_1c00,
            BinaryOperation::Multiply,
            4,
            Pairing::Elementwise,
        ),
        (
            0x6ec0_1400,
            BinaryOperation::AbsoluteDifference,
            8,
            Pairing::Elementwise,
        ),
        (
            0x4e40_1c00,
            BinaryOperation::MultiplyExtended,
            8,
            Pairing::Elementwise,
        ),
        (0x6e40_1400, BinaryOperation::Add, 8, Pairing::Pairwise),
        (0x4f00_9000, BinaryOperation::Multiply, 8, Pairing::Indexed),
    ];
    let host = TargetCpu::detect();
    let mut random = 0x2468_ace0_1357_9bdf_u64;
    let mut inline = 0usize;
    let mut exits = 0usize;
    for target in targets() {
        let packed = target.supports(TargetCpu::Avx2)
            && target.supports(TargetCpu::F16c)
            && !crate::x86_64::asm::vector_extensions::half_supported(target);
        assert!(host.supports(target));
        for &(base, operation, lanes, pairing) in PACKED_FORMS {
            let word = base | M << 16 | N << 5 | D;
            let code = compile(word, target);
            for case in 0..4000 {
                let mut n = [0u64; 8];
                let mut m = [0u64; 8];
                for lane in 0..8 {
                    n[lane] = finite_half(&mut random, None);
                    m[lane] = finite_half(&mut random, Some(n[lane]));
                }
                if case % 50 == 0 {
                    // An infinity or NaN somewhere.
                    n[case / 50 % lanes] = 0x7c00 | (case as u64 & 0x201);
                }
                let operands: Vec<(u64, u64)> = (0..lanes)
                    .map(|lane| match pairing {
                        Pairing::Elementwise => (n[lane], m[lane]),
                        Pairing::Indexed => (n[lane], m[0]),
                        Pairing::Pairwise => {
                            let source = if lane < lanes / 2 { &n } else { &m };
                            let first = 2 * (lane % (lanes / 2));
                            (source[first], source[first + 1])
                        }
                    })
                    .collect();
                let limbs = |values: &[u64; 8]| {
                    let mut limbs = [0u64; 2];
                    for (lane, value) in values.iter().enumerate().take(lanes) {
                        limbs[lane / 4] |= value << (16 * (lane % 4));
                    }
                    limbs
                };
                let controls: Vec<u32> = if case % 4 == 0 {
                    controls()
                } else {
                    alloc::vec![(case as u32 % 4) << FPCR_ROUNDING_SHIFT]
                };
                for fpcr in controls {
                    let mut initial = GuestState {
                        fpcr,
                        fpsr: if case % 3 == 0 { INEXACT } else { 0 },
                        ..GuestState::default()
                    };
                    initial.x[30] = RETURN;
                    initial.v[N as usize] = limbs(&n);
                    initial.v[M as usize] = limbs(&m);
                    initial.v[D as usize] = [u64::MAX, u64::MAX];
                    let mut actual = initial.clone();
                    let exit = run(&code, &mut actual);
                    let outcomes: Vec<_> = operands
                        .iter()
                        .map(|&(left, right)| evaluate(Format::Half, operation, fpcr, left, right))
                        .collect();
                    let finite = |bits: u64| bits >> 10 & 31 != 31;
                    let covered = packed
                        && fpcr & (FPCR_EXCEPTION_ENABLE_MASK | FPCR_FZ16 | FPCR_AH) == 0
                        && operands
                            .iter()
                            .zip(&outcomes)
                            .all(|(&(left, right), outcome)| {
                                finite(left)
                                    && finite(right)
                                    && outcome.as_ref().unwrap().exceptions & 0b100 == 0
                            });
                    if exit == PC.0 {
                        assert_eq!(actual, initial, "{word:08x} {operands:x?} {fpcr:#x}");
                        assert!(
                            !covered,
                            "covered lanes exited: {word:08x} {operands:x?} {fpcr:#x} {target:?}"
                        );
                        exits += 1;
                        continue;
                    }
                    assert_eq!(exit, RETURN, "{word:08x} {operands:x?}");
                    let mut expected = initial.clone();
                    let mut result = [0u64; 2];
                    for (lane, outcome) in outcomes.iter().enumerate() {
                        let outcome = outcome.as_ref().expect("no enabled traps");
                        expected.fpsr |= outcome.exceptions;
                        result[lane / 4] |= outcome.bits << (16 * (lane % 4));
                    }
                    expected.v[D as usize] = result;
                    assert_eq!(
                        actual, expected,
                        "{word:08x} {operands:x?} {fpcr:#x} {:#x} {target:?}",
                        initial.fpsr
                    );
                    inline += 1;
                }
            }
        }
    }
    assert!(inline > 50_000 && exits > 0, "{inline} {exits}");
}
