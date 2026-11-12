//! FMAX/FMIN/FMAXNM/FMINNM (scalar, vector, pairwise) and vector
//! FADD/FSUB/FSQRT against the scalar Arm references, compiled as blocks
//! for every tier this host has: every inline completion matches the
//! reference in every V register, FPSR and FPCR; every other run exits at
//! the instruction with the state unchanged. Ordered min/max inputs with FZ
//! and FIZ clear, and finite vector sums and roots after IXC, must complete
//! inline.
use crate::{
    GuestAddress, GuestState, Options, TargetCpu,
    aarch64::floating::{
        self, BinaryOperation, FPCR_AH, FPCR_DN, FPCR_EXCEPTION_ENABLE_MASK, FPCR_FIZ, FPCR_FZ,
        FPCR_NEP, FPCR_ROUNDING_SHIFT, Format, INEXACT, minmax::MinMaxOperation,
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

fn controls() -> [u32; 9] {
    [
        0,
        1 << FPCR_ROUNDING_SHIFT,
        3 << FPCR_ROUNDING_SHIFT,
        FPCR_FZ,
        FPCR_FIZ | FPCR_AH,
        FPCR_AH,
        FPCR_DN,
        FPCR_NEP,
        FPCR_EXCEPTION_ENABLE_MASK,
    ]
}

/// Zeros, subnormals, the smallest normals, ones, the largest finite
/// values, infinities, quiet and signalling NaNs with payloads (both signs)
/// and random values.
fn floats(format: Format) -> Vec<u64> {
    let double = format == Format::Double;
    let (sign, exponent_max, fraction_bits) = if double {
        (1u64 << 63, 0x7ffu64, 52)
    } else {
        (1u64 << 31, 0xffu64, 23)
    };
    let fraction_mask = (1u64 << fraction_bits) - 1;
    let quiet = 1u64 << (fraction_bits - 1);
    let bias = exponent_max >> 1;
    let mut values = Vec::new();
    for exponent in [0, 1, 2, 3, bias - 1, bias, bias + 1, exponent_max - 1] {
        for fraction in [0, 1, quiet, fraction_mask] {
            values.push(exponent << fraction_bits | fraction);
        }
    }
    let infinity = exponent_max << fraction_bits;
    values.extend([
        infinity,
        infinity | quiet,
        infinity | quiet | 5,
        infinity | 1,
        infinity | 7,
    ]);
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    for _ in 0..24 {
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

#[derive(Clone, Copy, Debug)]
enum Op {
    MinMax(MinMaxOperation, bool),
    Binary(BinaryOperation),
    Sqrt,
    /// FMLA (false) or FMLS (true): Vd + Vn * Vm.
    Fused(bool),
}

/// The instruction word and its lanes (0 for a scalar).
fn word(op: Op, format: Format, lanes: u32) -> u32 {
    let double = u32::from(format == Format::Double);
    let q = u32::from(lanes * format.bits() == 128);
    let registers = M << 16 | N << 5 | D;
    if lanes == 0 {
        let base = match op {
            Op::MinMax(MinMaxOperation::Maximum, _) => 0x1e20_4800,
            Op::MinMax(MinMaxOperation::Minimum, _) => 0x1e20_5800,
            Op::MinMax(MinMaxOperation::MaximumNumber, _) => 0x1e20_6800,
            Op::MinMax(MinMaxOperation::MinimumNumber, _) => 0x1e20_7800,
            Op::Binary(BinaryOperation::Add) => 0x1e20_2800,
            Op::Binary(BinaryOperation::Subtract) => 0x1e20_3800,
            Op::Binary(BinaryOperation::Multiply) => 0x1e20_0800,
            Op::Binary(_) => 0x1e20_1800,
            Op::Sqrt => return 0x1e21_c000 | double << 22 | N << 5 | D,
            Op::Fused(_) => unreachable!("vector FMLA/FMLS only"),
        };
        return base | double << 22 | registers;
    }
    let base = match op {
        Op::MinMax(MinMaxOperation::Maximum, pairwise) => 0x0e20_f400 | u32::from(pairwise) << 29,
        Op::MinMax(MinMaxOperation::Minimum, pairwise) => 0x0ea0_f400 | u32::from(pairwise) << 29,
        Op::MinMax(MinMaxOperation::MaximumNumber, pairwise) => {
            0x0e20_c400 | u32::from(pairwise) << 29
        }
        Op::MinMax(MinMaxOperation::MinimumNumber, pairwise) => {
            0x0ea0_c400 | u32::from(pairwise) << 29
        }
        Op::Binary(BinaryOperation::Add) => 0x0e20_d400,
        Op::Binary(BinaryOperation::Subtract) => 0x0ea0_d400,
        Op::Binary(BinaryOperation::Multiply) => 0x2e20_dc00,
        Op::Binary(_) => 0x2e20_fc00,
        Op::Fused(subtract) => 0x0e20_cc00 | u32::from(subtract) << 23,
        Op::Sqrt => return 0x2ea1_f800 | q << 30 | double << 22 | N << 5 | D,
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

fn evaluate(
    op: Op,
    format: Format,
    fpcr: u32,
    left: u64,
    right: u64,
    addend: u64,
) -> Option<floating::Outcome> {
    use floating::fused::FusedOperation;
    match op {
        Op::Fused(subtract) => {
            let operation = if subtract {
                FusedOperation::Subtract
            } else {
                FusedOperation::Add
            };
            floating::fused::evaluate(format, operation, fpcr, addend, left, right).ok()
        }
        Op::MinMax(operation, _) => {
            floating::minmax::evaluate(format, operation, fpcr, left, right).ok()
        }
        Op::Binary(operation) => floating::evaluate(format, operation, fpcr, left, right).ok(),
        Op::Sqrt => floating::sqrt::evaluate(format, fpcr, left).ok(),
    }
}

/// The reference state after the instruction, or `None` where the
/// reference declines the control (enabled traps).
fn reference(op: Op, format: Format, lanes: u32, initial: &GuestState) -> Option<GuestState> {
    let fpcr = initial.fpcr;
    let n = initial.v[N as usize];
    let m = initial.v[M as usize];
    let mut expected = initial.clone();
    let mut exceptions = 0;
    let mut result = [0u64; 2];
    let count = lanes.max(1) as usize;
    for index in 0..count {
        let (left, right) = match op {
            Op::MinMax(_, true) => {
                let pair = 2 * index;
                let source = if pair < count { n } else { m };
                let pair = pair % count;
                (lane(source, format, pair), lane(source, format, pair + 1))
            }
            _ => (lane(n, format, index), lane(m, format, index)),
        };
        let addend = lane(initial.v[D as usize], format, index);
        let outcome = evaluate(op, format, fpcr, left, right, addend)?;
        exceptions |= outcome.exceptions;
        set_lane(&mut result, format, index, outcome.bits);
    }
    if lanes == 0 && fpcr & FPCR_NEP != 0 {
        // NEP keeps the first source's upper bits for a binary operation,
        // the destination's for FSQRT.
        let old = if matches!(op, Op::Sqrt) {
            initial.v[D as usize]
        } else {
            n
        };
        let mask = if format == Format::Double {
            u64::MAX
        } else {
            0xffff_ffff
        };
        result = [old[0] & !mask | result[0], old[1]];
    }
    expected.v[D as usize] = result;
    expected.fpsr |= exceptions;
    Some(expected)
}

#[test]
fn quick_float_guards_match_the_reference() {
    use MinMaxOperation::*;
    let mut ops: Vec<Op> = Vec::new();
    for operation in [Maximum, Minimum, MaximumNumber, MinimumNumber] {
        ops.extend([Op::MinMax(operation, false), Op::MinMax(operation, true)]);
    }
    ops.extend([
        Op::Binary(BinaryOperation::Add),
        Op::Binary(BinaryOperation::Subtract),
        Op::Binary(BinaryOperation::Multiply),
        Op::Binary(BinaryOperation::Divide),
        Op::Sqrt,
        Op::Fused(false),
        Op::Fused(true),
    ]);
    let mut runs = 0usize;
    let mut inline = 0usize;
    for target in targets() {
        for &op in &ops {
            for format in [Format::Single, Format::Double] {
                let all = floats(format);
                let shapes: &[u32] = if format == Format::Double {
                    &[0, 2]
                } else {
                    &[0, 2, 4]
                };
                for &lanes in shapes {
                    if matches!(op, Op::MinMax(_, true) | Op::Fused(_)) && lanes == 0 {
                        continue;
                    }
                    let code = compile(word(op, format, lanes), target);
                    let count = lanes.max(1) as usize;
                    for start in 0..all.len() {
                        let mut n = [0x5a5a_a5a5_3c3c_c3c3u64, 0x0123_4567_89ab_cdef];
                        let mut m = [0x7654_3210_fedc_ba98u64, 0xc3c3_3c3c_a5a5_5a5a];
                        for index in 0..count {
                            set_lane(&mut n, format, index, all[(start + index) % all.len()]);
                            let other = (start * 7 + index * 13 + 3) % all.len();
                            set_lane(&mut m, format, index, all[other]);
                        }
                        for fpcr in controls() {
                            for fpsr in [0, INEXACT, 0x0800_009f] {
                                let mut initial = GuestState {
                                    fpcr,
                                    fpsr,
                                    nzcv: 0x6000_0000,
                                    ..GuestState::default()
                                };
                                initial.x[30] = RETURN;
                                initial.v[N as usize] = n;
                                initial.v[M as usize] = m;
                                let mut d = [0xdead_beef_0bad_f00d, 0xfeed_face_cafe_d00d];
                                for index in 0..count {
                                    let other = (start * 3 + index * 5 + 11) % all.len();
                                    set_lane(&mut d, format, index, all[other]);
                                }
                                initial.v[D as usize] = d;
                                let mut actual = initial.clone();
                                let exit = run(&code, &mut actual);
                                runs += 1;
                                let expected = reference(op, format, lanes, &initial);
                                let ordered = (0..count).all(|index| {
                                    let finite = |bits: u64| {
                                        let mask = format.exponent_mask();
                                        bits & mask != mask
                                    };
                                    finite(lane(n, format, index)) && finite(lane(m, format, index))
                                });
                                if exit == PC.0 {
                                    assert_eq!(
                                        actual, initial,
                                        "{op:?} {format:?} {lanes} {target:?}"
                                    );
                                    let must_inline = matches!(op, Op::MinMax(..))
                                        && lanes != 0
                                        && ordered
                                        && fpcr
                                            & (FPCR_FZ
                                                | FPCR_FIZ
                                                | FPCR_AH
                                                | FPCR_EXCEPTION_ENABLE_MASK)
                                            == 0
                                        && !target.data_independent_timing()
                                        && target.supports(TargetCpu::Sse41);
                                    assert!(
                                        !must_inline,
                                        "ordered min/max exited: {op:?} {format:?} {lanes} {n:x?} {m:x?} {fpcr:#x} {target:?}"
                                    );
                                    continue;
                                }
                                assert_eq!(exit, RETURN, "{op:?} {format:?} {lanes} {target:?}");
                                let expected = expected.expect("completed with enabled traps");
                                assert_eq!(
                                    actual,
                                    expected,
                                    "{op:?} {format:?} {lanes} {n:x?} {m:x?} {fpcr:#x} {fpsr:#x} {target:?} got fpsr {:#x} v {:x?} want {:#x} {:x?}",
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
    assert!(inline > runs / 3, "{inline} of {runs}");
}

/// FCMP/FCMPE (register and zero) and FCCMP/FCCMPE (condition true and
/// false) of S and D against `comparison::flags`: NZCV and FPSR of every
/// inline completion, an unchanged state for every exit, and no exit for an
/// ordered register pair with traps, FZ, FIZ and AH clear on non-DIT tiers.
#[test]
fn quick_compares_match_the_reference() {
    use crate::aarch64::floating::comparison;
    let mut inline = 0usize;
    for target in targets() {
        for format in [Format::Single, Format::Double] {
            let all = floats(format);
            let double = u32::from(format == Format::Double);
            // (word, signal all NaNs, conditional, zero)
            let mut forms = Vec::new();
            for signal in [false, true] {
                let e = u32::from(signal) << 4;
                forms.push((
                    0x1e20_2000 | double << 22 | M << 16 | N << 5 | e,
                    signal,
                    false,
                    false,
                ));
                forms.push((0x1e20_2008 | double << 22 | N << 5 | e, signal, false, true));
                // FCCMP(E) ..., #0b0110, EQ
                forms.push((
                    0x1e20_0406 | double << 22 | M << 16 | N << 5 | e,
                    signal,
                    true,
                    false,
                ));
            }
            for (word, signal, conditional, zero) in forms {
                let code = compile(word, target);
                for (start, &left) in all.iter().enumerate() {
                    let right = if zero {
                        0
                    } else {
                        all[(start * 7 + 3) % all.len()]
                    };
                    for fpcr in controls() {
                        for nzcv in [0x6000_0000u32, 0x8000_0000] {
                            for fpsr in [0, 0x0800_009f] {
                                let mut initial = GuestState {
                                    fpcr,
                                    fpsr,
                                    nzcv,
                                    ..GuestState::default()
                                };
                                initial.x[30] = RETURN;
                                initial.v[N as usize] =
                                    [left | 0x5a5a_0000_0000_0000 & !width(format), 7];
                                initial.v[M as usize] =
                                    [right | 0x3c3c_0000_0000_0000 & !width(format), 9];
                                let mut actual = initial.clone();
                                let exit = run(&code, &mut actual);
                                let holds = !conditional || nzcv & 0x4000_0000 != 0;
                                let ordered = |bits: u64| {
                                    bits & format.exponent_mask() != format.exponent_mask()
                                        || bits & ((1u64 << format.fraction_bits()) - 1) == 0
                                };
                                if exit == PC.0 {
                                    assert_eq!(
                                        actual, initial,
                                        "{word:08x} {left:x} {right:x} {fpcr:#x} {target:?}"
                                    );
                                    let must_inline = holds
                                        && !zero
                                        && ordered(left)
                                        && ordered(right)
                                        && fpcr
                                            & (FPCR_FZ
                                                | FPCR_FIZ
                                                | FPCR_AH
                                                | FPCR_EXCEPTION_ENABLE_MASK)
                                            == 0
                                        && !target.data_independent_timing();
                                    assert!(
                                        !must_inline,
                                        "ordered compare exited: {word:08x} {left:x} {right:x} {fpcr:#x} {target:?}"
                                    );
                                    continue;
                                }
                                assert_eq!(exit, RETURN);
                                let mut expected = initial.clone();
                                if holds {
                                    let outcome =
                                        comparison::flags(format, signal, fpcr, left, right)
                                            .expect("completed with enabled traps");
                                    expected.nzcv = outcome.nzcv;
                                    expected.fpsr |= outcome.exceptions;
                                } else {
                                    expected.nzcv = 0x6 << 28;
                                }
                                assert_eq!(
                                    (actual.nzcv, actual.fpsr),
                                    (expected.nzcv, expected.fpsr),
                                    "{word:08x} {left:x} {right:x} {fpcr:#x} {nzcv:#x} {target:?}"
                                );
                                assert_eq!(actual, expected, "{word:08x} {target:?}");
                                inline += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(inline > 10_000, "{inline}");
}

fn width(format: Format) -> u64 {
    if format == Format::Double {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    }
}

/// MXCSR with every status bit set (IE, DE, ZE, OE, UE, PE) and the
/// power-on control.
const POLLUTED_MXCSR: u32 = 0x1fbf;

fn load_mxcsr(value: u32) {
    // SAFETY: a valid MXCSR value (no reserved bits) from a live local.
    unsafe { core::arch::asm!("ldmxcsr [{}]", in(reg) &value, options(nostack)) };
}

fn read_mxcsr() -> u32 {
    let mut value = 0u32;
    // SAFETY: stores the four-byte MXCSR into a live local.
    unsafe { core::arch::asm!("stmxcsr [{}]", in(reg) &mut value, options(nostack)) };
    value
}

/// FCMP S1,S2; FMAX V0.4S,V1.4S,V2.4S; FADD V3.4S,V1.4S,V2.4S; MRS X0,FPSR
/// as one block, entered with every host MXCSR status bit already set and
/// leaving whatever UCOMISS/CMPUNORDPS/ADDPS add: a completed block's FPSR
/// and the MRS result are the Arm cumulative flags of the three
/// instructions (from the references), never the host status. A block that
/// exits (a NaN operand, which the engine completes in software) stops at
/// one of the three instructions.
#[test]
fn guest_fpsr_ignores_host_mxcsr_status() {
    use crate::aarch64::floating::{comparison, minmax};
    const WORDS: [u32; 5] = [
        0x1e22_2020,
        0x4e22_f420,
        0x4e22_d423,
        0xd53b_4420,
        0xd65f_03c0,
    ];
    let source: Vec<u8> = WORDS.iter().flat_map(|word| word.to_le_bytes()).collect();
    let values = floats(Format::Single);
    let mut inline = 0usize;
    for target in targets() {
        let code = CompilationPlan::new(&source, PC, &Options::new(BlockProfile::ORDINARY, target))
            .unwrap()
            .compile()
            .unwrap()
            .native
            .x86_64_code;
        for start in 0..values.len() {
            let mut v1 = [0u64; 2];
            let mut v2 = [0u64; 2];
            for at in 0..4 {
                set_lane(
                    &mut v1,
                    Format::Single,
                    at,
                    values[(start + at) % values.len()],
                );
                set_lane(
                    &mut v2,
                    Format::Single,
                    at,
                    values[(start * 5 + at * 3 + 1) % values.len()],
                );
            }
            for fpsr in [0, INEXACT, 0x0800_009f] {
                let mut initial = GuestState {
                    fpsr,
                    ..GuestState::default()
                };
                initial.x[30] = RETURN;
                initial.v[1] = v1;
                initial.v[2] = v2;
                let mut actual = initial.clone();
                let before = read_mxcsr();
                load_mxcsr(POLLUTED_MXCSR);
                let exit = run(&code, &mut actual);
                let after = read_mxcsr();
                load_mxcsr(before);
                assert_eq!(
                    after & !0x3f,
                    POLLUTED_MXCSR & !0x3f,
                    "MXCSR control {target:?}"
                );
                if exit != RETURN {
                    assert!(
                        (0..3).any(|index| exit == PC.0 + 4 * index),
                        "exit {exit:#x} {target:?}"
                    );
                    continue;
                }
                let mut flags = fpsr;
                let compare = comparison::flags(
                    Format::Single,
                    false,
                    0,
                    lane(v1, Format::Single, 0),
                    lane(v2, Format::Single, 0),
                )
                .unwrap();
                flags |= compare.exceptions;
                for at in 0..4 {
                    let (left, right) =
                        (lane(v1, Format::Single, at), lane(v2, Format::Single, at));
                    flags |=
                        minmax::evaluate(Format::Single, MinMaxOperation::Maximum, 0, left, right)
                            .unwrap()
                            .exceptions;
                    flags |=
                        floating::evaluate(Format::Single, BinaryOperation::Add, 0, left, right)
                            .unwrap()
                            .exceptions;
                }
                assert_eq!(actual.nzcv, compare.nzcv, "{v1:x?} {v2:x?} {target:?}");
                assert_eq!(
                    (actual.fpsr, actual.x[0]),
                    (flags, u64::from(flags)),
                    "{v1:x?} {v2:x?} fpsr {fpsr:#x} {target:?}"
                );
                inline += 1;
            }
        }
    }
    assert!(inline > 100, "{inline}");
}
