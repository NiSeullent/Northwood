//! FP products, quotients, FMLA/FMLS and FRECPS/FRSQRTS (scalar, vector and
//! by element) with NaN, infinite, zero and tiny operands against the
//! scalar Arm references, compiled as blocks for every tier this host has
//! (DIT included), with FPCR.DN on and off and FPSR with and without every
//! cumulative flag: every inline completion matches the reference in its
//! result, FPSR and the rest of the state; every other run exits at the
//! instruction with the state unchanged.
use crate::{
    GuestAddress, GuestState, Options, TargetCpu,
    aarch64::{
        floating::{
            self, BinaryOperation, FPCR_DN, FPCR_FZ, FPCR_ROUNDING_SHIFT, Format, INEXACT,
            fused::FusedOperation,
        },
        simd::refinement::Kind,
    },
    block::{BlockProfile, CompilationPlan},
    x86_64::native::run,
};
use alloc::vec::Vec;

const PC: GuestAddress = GuestAddress(0x1000_0000);
const RETURN: u64 = 0x4000;
const D: u32 = 3;
const N: u32 = 4;
const M: u32 = 17;
const ALL_FLAGS: u32 = 0x0800_009f;

fn targets() -> Vec<TargetCpu> {
    let host = TargetCpu::detect();
    [
        TargetCpu::Baseline,
        TargetCpu::Sse41,
        TargetCpu::X86_64_V3,
        TargetCpu::X86_64_V4,
        TargetCpu::X86_64_V2.with_data_independent_timing(),
        TargetCpu::X86_64_V4.with_data_independent_timing(),
        host,
    ]
    .into_iter()
    .filter(|target| host.supports(*target))
    .collect()
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

#[derive(Clone, Copy, Debug)]
enum Op {
    Binary(BinaryOperation),
    Fused(FusedOperation),
    Refinement(Kind),
}

/// Shapes: 0 scalar, else the lane count; `index` selects the by-element
/// form.
fn word(op: Op, format: Format, lanes: u32, index: Option<u32>) -> u32 {
    let double = u32::from(format == Format::Double);
    let q = u32::from(lanes * format.bits() == 128);
    if let Some(index) = index {
        let base = match op {
            Op::Binary(BinaryOperation::Multiply) => 0x0f80_9000,
            Op::Binary(_) => 0x2f80_9000,
            Op::Fused(FusedOperation::Add) => 0x0f80_1000,
            Op::Fused(_) => 0x0f80_5000,
            Op::Refinement(_) => unreachable!("no by-element form"),
        };
        let scalar = if lanes == 0 { 0x5000_0000 } else { q << 30 };
        // M = 17: Rm field 1 with the M bit (bit 20) set, for S; D uses
        // the M bit as Rm<4> too, with H as its index.
        let (h, l) = if double == 1 {
            (index, 0)
        } else {
            (index >> 1, index & 1)
        };
        return base | scalar | double << 22 | l << 21 | 1 << 20 | 1 << 16 | h << 11 | N << 5 | D;
    }
    let registers = M << 16 | N << 5 | D;
    if lanes == 0 {
        let base = match op {
            Op::Binary(BinaryOperation::Multiply) => 0x1e20_0800,
            Op::Binary(BinaryOperation::Divide) => 0x1e20_1800,
            Op::Binary(BinaryOperation::NegatedMultiply) => 0x1e20_8800,
            Op::Binary(_) => 0x5e20_dc00,
            Op::Refinement(Kind::Reciprocal) => 0x5e20_fc00,
            Op::Refinement(_) => 0x5ea0_fc00,
            Op::Fused(_) => unreachable!("scalar FMADD is another encoding"),
        };
        return base | double << 22 | registers;
    }
    let base = match op {
        Op::Binary(BinaryOperation::Multiply) => 0x2e20_dc00,
        Op::Binary(BinaryOperation::Divide) => 0x2e20_fc00,
        Op::Binary(_) => 0x0e20_dc00,
        Op::Fused(FusedOperation::Add) => 0x0e20_cc00,
        Op::Fused(_) => 0x0ea0_cc00,
        Op::Refinement(Kind::Reciprocal) => 0x0e20_fc00,
        Op::Refinement(_) => 0x0ea0_fc00,
    };
    base | q << 30 | double << 22 | registers
}

fn lane(vector: [u64; 2], format: Format, index: usize) -> u64 {
    if format == Format::Double {
        vector[index]
    } else {
        vector[index / 2] >> (32 * (index % 2)) & 0xffff_ffff
    }
}

fn set_lane(vector: &mut [u64; 2], format: Format, index: usize, bits: u64) {
    if format == Format::Double {
        vector[index] = bits;
    } else {
        let shift = 32 * (index % 2);
        vector[index / 2] = vector[index / 2] & !(0xffff_ffff << shift) | bits << shift;
    }
}

/// NaNs (quiet and signalling, payloads, both signs), infinities, zeros,
/// tiny and huge values and ones.
fn floats(format: Format) -> Vec<u64> {
    let double = format == Format::Double;
    let (sign, exponent_max, fraction_bits) = if double {
        (1u64 << 63, 0x7ffu64, 52)
    } else {
        (1u64 << 31, 0xffu64, 23)
    };
    let quiet = 1u64 << (fraction_bits - 1);
    let infinity = exponent_max << fraction_bits;
    let bias = exponent_max >> 1;
    let mut values = alloc::vec![
        infinity | quiet,
        infinity | quiet | 0x15,
        infinity | 1,
        infinity | 0x2a,
        infinity,
        0,
        1,
        1 << fraction_bits,
        bias << fraction_bits,
        (bias + 1) << fraction_bits | 3,
        (exponent_max - 1) << fraction_bits | (quiet - 1),
        (bias - 3) << fraction_bits | 0x1234,
    ];
    let positive = values.len();
    for index in 0..positive {
        values.push(values[index] | sign);
    }
    values
}

fn evaluate(
    op: Op,
    format: Format,
    fpcr: u32,
    left: u64,
    right: u64,
    addend: u64,
) -> Option<floating::Outcome> {
    match op {
        Op::Binary(operation) => floating::evaluate(format, operation, fpcr, left, right).ok(),
        Op::Fused(operation) => {
            floating::fused::evaluate(format, operation, fpcr, addend, left, right).ok()
        }
        Op::Refinement(kind) => {
            floating::refinement::evaluate(kind, format, fpcr, left, right).ok()
        }
    }
}

fn reference(
    op: Op,
    format: Format,
    lanes: u32,
    index: Option<u32>,
    initial: &GuestState,
) -> GuestState {
    let n = initial.v[N as usize];
    let m = initial.v[M as usize];
    let d = initial.v[D as usize];
    let mut expected = initial.clone();
    let mut result = [0u64; 2];
    for at in 0..lanes.max(1) as usize {
        let right = lane(m, format, index.map_or(at, |index| index as usize));
        let outcome = evaluate(
            op,
            format,
            initial.fpcr,
            lane(n, format, at),
            right,
            lane(d, format, at),
        )
        .expect("no enabled traps");
        expected.fpsr |= outcome.exceptions;
        set_lane(&mut result, format, at, outcome.bits);
    }
    expected.v[D as usize] = result;
    expected
}

#[test]
fn products_select_arm_nans_under_dn_on_every_tier() {
    let ops = [
        Op::Binary(BinaryOperation::Multiply),
        Op::Binary(BinaryOperation::Divide),
        Op::Binary(BinaryOperation::MultiplyExtended),
        Op::Binary(BinaryOperation::NegatedMultiply),
        Op::Fused(FusedOperation::Add),
        Op::Fused(FusedOperation::Subtract),
        Op::Refinement(Kind::Reciprocal),
        Op::Refinement(Kind::ReciprocalSquareRoot),
    ];
    let mut inline = 0usize;
    for target in targets() {
        for &op in &ops {
            for format in [Format::Single, Format::Double] {
                let values = floats(format);
                let lane_counts: &[u32] = if format == Format::Double {
                    &[0, 2]
                } else {
                    &[0, 2, 4]
                };
                for &lanes in lane_counts {
                    // FNMUL is scalar only; FMLA/FMLS by vector are vector only.
                    let negated = matches!(op, Op::Binary(BinaryOperation::NegatedMultiply));
                    let fused = matches!(op, Op::Fused(_));
                    let mut forms: Vec<Option<u32>> = Vec::new();
                    if (lanes == 0 || !negated) && (lanes != 0 || !fused) {
                        forms.push(None);
                    }
                    let elements = matches!(
                        op,
                        Op::Binary(BinaryOperation::Multiply | BinaryOperation::MultiplyExtended)
                            | Op::Fused(_)
                    );
                    if elements && (lanes != 0 || !fused) {
                        let count = if format == Format::Double { 2 } else { 4 };
                        forms.extend((0..count).map(Some));
                    }
                    for index in forms {
                        let code = compile(word(op, format, lanes, index), target);
                        for start in 0..values.len() * values.len() / 4 {
                            let mut n = [0x5a5a_a5a5_3c3c_c3c3u64, 0x0123_4567_89ab_cdef];
                            let mut m = [0x7654_3210_fedc_ba98u64, 0xc3c3_3c3c_a5a5_5a5a];
                            let mut d = [0xdead_beef_0bad_f00du64, 0xfeed_face_cafe_d00d];
                            for at in 0..4.min(if format == Format::Double { 2 } else { 4 }) {
                                set_lane(&mut n, format, at, values[(start + at) % values.len()]);
                                let other = (start / values.len() + at * 5) % values.len();
                                set_lane(&mut m, format, at, values[other]);
                                set_lane(
                                    &mut d,
                                    format,
                                    at,
                                    values[(start * 3 + at) % values.len()],
                                );
                            }
                            for fpcr in [0, FPCR_DN, FPCR_DN | FPCR_FZ, 2 << FPCR_ROUNDING_SHIFT] {
                                for fpsr in [0, INEXACT, ALL_FLAGS] {
                                    let mut initial = GuestState {
                                        fpcr,
                                        fpsr,
                                        nzcv: 0x6000_0000,
                                        ..GuestState::default()
                                    };
                                    initial.x[30] = RETURN;
                                    initial.v[N as usize] = n;
                                    initial.v[M as usize] = m;
                                    initial.v[D as usize] = d;
                                    let mut actual = initial.clone();
                                    let exit = run(&code, &mut actual);
                                    if exit == PC.0 {
                                        assert_eq!(
                                            actual, initial,
                                            "{op:?} {format:?} {lanes} {index:?} {target:?}"
                                        );
                                        continue;
                                    }
                                    assert_eq!(exit, RETURN);
                                    let expected = reference(op, format, lanes, index, &initial);
                                    assert_eq!(
                                        (actual.v[D as usize], actual.fpsr),
                                        (expected.v[D as usize], expected.fpsr),
                                        "{op:?} {format:?} {lanes} {index:?} n {n:x?} m {m:x?} d {d:x?} fpcr {fpcr:#x} fpsr {fpsr:#x} {target:?}"
                                    );
                                    assert_eq!(
                                        actual, expected,
                                        "{op:?} {format:?} {lanes} {index:?} {target:?}"
                                    );
                                    inline += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(inline > 10_000, "{inline}");
}
