//! Products below the minimum normal (`tiny_product_status` and the sticky
//! path's known-underflow check) against the reference arithmetic, on every
//! target the host runs: every inline completion matches the reference in
//! its lanes and FPSR, every exit leaves the state unchanged, and covered
//! products must complete inline.
use crate::{
    GuestAddress, GuestState, Options, TargetCpu,
    aarch64::floating::{
        BinaryOperation, FPCR_AH, FPCR_DN, FPCR_EXCEPTION_ENABLE_MASK, FPCR_FIZ, FPCR_FZ,
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
const UNDERFLOW: u32 = 1 << 3;
const OVERFLOW: u32 = 1 << 2;

fn targets() -> Vec<TargetCpu> {
    let host = TargetCpu::detect();
    let mut targets: Vec<TargetCpu> = [
        TargetCpu::Baseline,
        TargetCpu::X86_64_V2,
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

/// (word without registers, operation, format, lanes; 1 is scalar).
const FORMS: &[(u32, BinaryOperation, Format, usize)] = &[
    (0x6e20_dc00, BinaryOperation::Multiply, Format::Single, 4),
    (0x2e20_dc00, BinaryOperation::Multiply, Format::Single, 2),
    (0x6e60_dc00, BinaryOperation::Multiply, Format::Double, 2),
    (0x1e20_0800, BinaryOperation::Multiply, Format::Single, 1),
    (0x1e60_0800, BinaryOperation::Multiply, Format::Double, 1),
    (
        0x1e20_8800,
        BinaryOperation::NegatedMultiply,
        Format::Single,
        1,
    ),
    (
        0x4e20_dc00,
        BinaryOperation::MultiplyExtended,
        Format::Single,
        4,
    ),
    (
        0x5e60_dc00,
        BinaryOperation::MultiplyExtended,
        Format::Double,
        1,
    ),
];

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// A normal of `format` with unbiased exponent `exponent`, a random
/// fraction and sign.
fn normal(format: Format, exponent: i32, random: &mut Random) -> u64 {
    let fraction_bits = format.fraction_bits();
    let bias = format.exponent_bias() as i32;
    let biased = (exponent + bias).clamp(1, 2 * bias) as u64;
    let fraction = match random.next() % 4 {
        0 => 0,
        1 => (1 << fraction_bits) - 1,
        _ => random.next() & ((1 << fraction_bits) - 1),
    };
    let sign = (random.next() & 1) << (format.bits() - 1);
    sign | biased << fraction_bits | fraction
}

/// Input pairs whose products are mostly tiny, near the minimum normal or
/// exact, with some ordinary, zero and special inputs.
fn pairs(format: Format, random: &mut Random) -> (u64, u64) {
    let minimum = 1 - format.exponent_bias() as i32;
    let left_exponent = (random.next() % 120) as i32 - 100;
    let delta = (random.next() % 40) as i32 - 32;
    let right_exponent = minimum - left_exponent + delta;
    let left = normal(format, left_exponent, random);
    let right = normal(format, right_exponent, random);
    match random.next() % 16 {
        0 => (left, 0),
        1 => (
            left,
            normal(format, (random.next() % 20) as i32 - 10, random),
        ),
        // Power-of-two products: exact tiny results, or the minimum normal.
        2 => (
            left & !((1 << format.fraction_bits()) - 1),
            right & !((1 << format.fraction_bits()) - 1),
        ),
        // Just below the minimum normal: rounds to it under nearest.
        3 => {
            let one_minus = (u64::from(format.exponent_bias()) << format.fraction_bits()) - 1;
            (1u64 << format.fraction_bits(), one_minus)
        }
        4 => (left, format.exponent_mask()),
        5 => (left, 1),
        _ => (left, right),
    }
}

fn controls() -> Vec<u32> {
    let mut controls: Vec<u32> = (0..4).map(|mode| mode << FPCR_ROUNDING_SHIFT).collect();
    controls.extend([
        FPCR_FZ,
        FPCR_AH,
        FPCR_FZ | FPCR_AH,
        FPCR_FIZ,
        FPCR_DN,
        1 << 11, // UFE
        FPCR_EXCEPTION_ENABLE_MASK,
    ]);
    controls
}

#[test]
fn tiny_products_match_the_reference() {
    let mut random = Random(0x5eed_1234_8765_4321);
    let mut inline = 0usize;
    let mut tiny_inline = 0usize;
    for target in targets() {
        for &(base, operation, format, lanes) in FORMS {
            let word = base | M << 16 | N << 5 | D;
            let code = compile(word, target);
            let width = format.bits() as usize;
            for case in 0..3000 {
                let inputs: Vec<(u64, u64)> =
                    (0..lanes).map(|_| pairs(format, &mut random)).collect();
                let pack = |pick: &dyn Fn(&(u64, u64)) -> u64| {
                    let mut limbs = [0u64; 2];
                    for (lane, pair) in inputs.iter().enumerate() {
                        let bit = lane * width;
                        limbs[bit / 64] |= pick(pair) << (bit % 64);
                    }
                    limbs
                };
                let controls: Vec<u32> = if case % 8 == 0 {
                    controls()
                } else {
                    alloc::vec![0]
                };
                for fpcr in controls {
                    for fpsr in [0, INEXACT, INEXACT | UNDERFLOW, UNDERFLOW] {
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
                            .map(|&(left, right)| evaluate(format, operation, fpcr, left, right))
                            .collect();
                        let covered = fpcr & (FPCR_EXCEPTION_ENABLE_MASK | FPCR_FZ | FPCR_AH) == 0
                            && inputs
                                .iter()
                                .zip(&outcomes)
                                .all(|(&(left, right), outcome)| {
                                    let normal_or_zero = |bits: u64| {
                                        let exponent = bits >> format.fraction_bits()
                                            & (format.exponent_mask() >> format.fraction_bits());
                                        let magnitude = bits & !(1 << (format.bits() - 1));
                                        magnitude == 0
                                            || exponent != 0
                                                && exponent
                                                    != format.exponent_mask()
                                                        >> format.fraction_bits()
                                    };
                                    let outcome = outcome.as_ref().unwrap();
                                    normal_or_zero(left)
                                        && normal_or_zero(right)
                                        && outcome.exceptions & OVERFLOW == 0
                                });
                        if exit == PC.0 {
                            assert_eq!(actual, initial, "{word:08x} {inputs:x?} {fpcr:#x}");
                            assert!(
                                !covered,
                                "covered product exited: {word:08x} {inputs:x?} {fpcr:#x} {fpsr:#x} {target:?}"
                            );
                            continue;
                        }
                        assert_eq!(exit, RETURN, "{word:08x} {inputs:x?}");
                        let mut expected = initial.clone();
                        let mut limbs = [0u64; 2];
                        let mut tiny = false;
                        for (lane, outcome) in outcomes.iter().enumerate() {
                            let outcome = outcome.as_ref().expect("no enabled traps");
                            expected.fpsr |= outcome.exceptions;
                            tiny |= outcome.exceptions & UNDERFLOW != 0;
                            let bit = lane * width;
                            limbs[bit / 64] |= outcome.bits << (bit % 64);
                        }
                        expected.v[D as usize] = limbs;
                        assert_eq!(
                            actual, expected,
                            "{word:08x} {inputs:x?} {fpcr:#x} {fpsr:#x} {target:?}"
                        );
                        inline += 1;
                        tiny_inline += usize::from(tiny);
                    }
                }
            }
        }
    }
    assert!(inline > 50_000, "{inline}");
    assert!(tiny_inline > 10_000, "{tiny_inline}");
}
