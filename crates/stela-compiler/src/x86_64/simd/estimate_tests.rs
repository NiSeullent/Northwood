//! FRECPE and FRSQRTE (S and D, scalar and vector) against the Arm
//! reference, compiled as blocks for every tier this host has: every inline
//! completion matches the reference in every V register and FPSR, every
//! exit leaves the state unchanged, and zeros, infinities, NaNs and
//! negative inputs complete without an exit, as do subnormal inputs where
//! FZ, FIZ or AH flushes them and FRECPE's subnormal estimates where FZ or
//! AH flushes them.
use crate::{
    GuestAddress, GuestState, Options, TargetCpu,
    aarch64::{
        floating::{
            self, DIVIDE_BY_ZERO, FPCR_AH, FPCR_DN, FPCR_EXCEPTION_ENABLE_MASK, FPCR_FIZ, FPCR_FZ,
            FPCR_NEP, FPCR_ROUNDING_SHIFT, Format,
        },
        simd::estimate::Kind,
    },
    block::{BlockProfile, CompilationPlan},
    x86_64::native::run,
};
use alloc::vec::Vec;

const PC: GuestAddress = GuestAddress(0x1000_0000);
const RETURN: u64 = 0x4000;
const D: u32 = 3;
const N: u32 = 4;

fn targets() -> Vec<TargetCpu> {
    let host = TargetCpu::detect();
    [
        TargetCpu::Baseline,
        TargetCpu::Sse41,
        TargetCpu::X86_64_V2,
        TargetCpu::X86_64_V3,
        TargetCpu::X86_64_V4,
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

fn controls() -> [u32; 11] {
    [
        0,
        1 << FPCR_ROUNDING_SHIFT,
        3 << FPCR_ROUNDING_SHIFT,
        FPCR_FZ,
        FPCR_FIZ,
        FPCR_FZ | FPCR_DN,
        FPCR_FIZ | FPCR_AH,
        FPCR_AH | FPCR_DN,
        FPCR_DN,
        FPCR_NEP,
        FPCR_EXCEPTION_ENABLE_MASK,
    ]
}

/// Zeros, subnormals (tiny and large), normals across the range, the
/// largest finite values, infinities, quiet and signalling NaNs with
/// payloads (both signs) and random values.
fn floats(format: Format) -> Vec<u64> {
    let fraction_bits = format.fraction_bits();
    let sign = format.sign_mask();
    let exponent_max = format.exponent_mask() >> fraction_bits;
    let fraction_mask = format.fraction_mask();
    let quiet = 1u64 << (fraction_bits - 1);
    let bias = exponent_max >> 1;
    let mut values = Vec::new();
    for exponent in [
        0,
        1,
        2,
        bias - 1,
        bias,
        bias + 1,
        exponent_max - 3,
        exponent_max - 2,
        exponent_max - 1,
    ] {
        for fraction in [0, 1, quiet >> 1, quiet, fraction_mask] {
            values.push(exponent << fraction_bits | fraction);
        }
    }
    let infinity = format.exponent_mask();
    values.extend([
        infinity,
        infinity | quiet,
        infinity | quiet | 5,
        infinity | 1,
    ]);
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    for _ in 0..16 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        values.push(seed & (sign - 1));
    }
    let positive = values.len();
    for index in 0..positive {
        values.push(values[index] | sign);
    }
    values
}

fn word(kind: Kind, format: Format, lanes: u32) -> u32 {
    let double = u32::from(format == Format::Double);
    let sqrt = u32::from(kind == Kind::ReciprocalSquareRoot);
    let registers = N << 5 | D;
    if lanes == 0 {
        return 0x5ea1_d800 | sqrt << 29 | double << 22 | registers;
    }
    let q = u32::from(lanes * format.bits() == 128);
    0x0ea1_d800 | q << 30 | sqrt << 29 | double << 22 | registers
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

/// The reference state after the instruction, or `None` where the
/// reference declines the control (enabled traps).
fn reference(kind: Kind, format: Format, lanes: u32, initial: &GuestState) -> Option<GuestState> {
    let fpcr = initial.fpcr;
    let source = initial.v[N as usize];
    let mut expected = initial.clone();
    let mut result = if lanes == 0 && fpcr & FPCR_NEP != 0 {
        initial.v[D as usize]
    } else {
        [0; 2]
    };
    for index in 0..lanes.max(1) as usize {
        let outcome =
            floating::estimate::evaluate(kind, format, fpcr, lane(source, format, index)).ok()?;
        expected.fpsr |= outcome.exceptions;
        set_lane(&mut result, format, index, outcome.bits);
    }
    expected.v[D as usize] = result;
    Some(expected)
}

/// Whether the instruction must complete inline: no enabled trap, every
/// subnormal input flushed and every FRECPE subnormal estimate flushed.
fn must_inline(kind: Kind, format: Format, fpcr: u32, inputs: &[u64]) -> bool {
    let alternative = fpcr & FPCR_AH != 0;
    if fpcr & FPCR_EXCEPTION_ENABLE_MASK != 0 && !alternative {
        return false;
    }
    let flush_input = fpcr & (FPCR_FZ | FPCR_FIZ | FPCR_AH) != 0;
    let flush_output = fpcr & (FPCR_FZ | FPCR_AH) != 0;
    let magnitude = format.sign_mask() - 1;
    let subnormal_estimate = u64::from(2 * format.exponent_bias() - 1) << format.fraction_bits();
    inputs.iter().all(|&bits| {
        let bits = bits & magnitude;
        let subnormal = bits != 0 && bits & format.exponent_mask() == 0;
        let tiny_estimate =
            kind == Kind::Reciprocal && bits >= subnormal_estimate && bits < format.exponent_mask();
        (!subnormal || flush_input) && (!tiny_estimate || flush_output)
    })
}

#[test]
fn special_estimates_complete_inline() {
    let mut inline = 0usize;
    let mut runs = 0usize;
    for target in targets() {
        for kind in [Kind::Reciprocal, Kind::ReciprocalSquareRoot] {
            for format in [Format::Single, Format::Double] {
                let all = floats(format);
                let shapes: &[u32] = if format == Format::Double {
                    &[0, 2]
                } else {
                    &[0, 2, 4]
                };
                for &lanes in shapes {
                    let code = compile(word(kind, format, lanes), target);
                    let count = lanes.max(1) as usize;
                    for start in 0..all.len() {
                        let mut source = [0x5a5a_a5a5_3c3c_c3c3u64, 0x0123_4567_89ab_cdef];
                        let mut inputs = Vec::new();
                        for index in 0..count {
                            let bits = all[(start + index * 11) % all.len()];
                            inputs.push(bits);
                            set_lane(&mut source, format, index, bits);
                        }
                        for fpcr in controls() {
                            for fpsr in [0, 0x0800_009f] {
                                let mut initial = GuestState {
                                    fpcr,
                                    fpsr,
                                    nzcv: 0x6000_0000,
                                    ..GuestState::default()
                                };
                                initial.x[30] = RETURN;
                                initial.v[N as usize] = source;
                                initial.v[D as usize] =
                                    [0xdead_beef_0bad_f00d, 0xfeed_face_cafe_d00d];
                                let mut actual = initial.clone();
                                let exit = run(&code, &mut actual);
                                runs += 1;
                                let label = || {
                                    alloc::format!(
                                        "{kind:?} {format:?} {lanes} {inputs:x?} {fpcr:#x} {fpsr:#x} {target:?}"
                                    )
                                };
                                if exit == PC.0 {
                                    assert_eq!(actual, initial, "{}", label());
                                    assert!(
                                        !must_inline(kind, format, fpcr, &inputs),
                                        "exited: {}",
                                        label()
                                    );
                                    continue;
                                }
                                assert_eq!(exit, RETURN, "{}", label());
                                let expected = reference(kind, format, lanes, &initial)
                                    .expect("completed with enabled traps");
                                assert_eq!(
                                    actual,
                                    expected,
                                    "{} got fpsr {:#x} v {:x?} want {:#x} {:x?}",
                                    label(),
                                    actual.fpsr,
                                    actual.v[D as usize],
                                    expected.fpsr,
                                    expected.v[D as usize]
                                );
                                inline += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(inline > runs * 3 / 4, "{inline} of {runs}");
}

/// A 4S FRECPE with zero lanes, as hot loops issue it, completes without
/// an exit and sets DZC, under every flushing mode.
#[test]
fn zero_lanes_set_divide_by_zero_without_an_exit() {
    for target in targets() {
        let code = compile(word(Kind::Reciprocal, Format::Single, 4), target);
        for fpcr in [0, FPCR_FZ, FPCR_FZ | FPCR_DN, FPCR_DN] {
            let mut initial = GuestState {
                fpcr,
                ..GuestState::default()
            };
            initial.x[30] = RETURN;
            initial.v[N as usize] = [0x3f80_0000_0000_0000, 0x8000_0000_4000_0000];
            let mut actual = initial.clone();
            assert_eq!(run(&code, &mut actual), RETURN, "{fpcr:#x} {target:?}");
            assert_eq!(actual.fpsr, DIVIDE_BY_ZERO, "{fpcr:#x} {target:?}");
            assert_eq!(
                actual.v[D as usize],
                [0x3f7f_8000_7f80_0000, 0xff80_0000_3eff_8000],
                "{fpcr:#x} {target:?}"
            );
        }
    }
}
