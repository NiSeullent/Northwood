//! Execute real emitted blocks, including their interior helper exits.
use super::super::*;
use crate::execution::platform::Mapping;
use stela_compiler::{
    GuestAddress, TargetCpu,
    aarch64::{floating::*, simd::minmax::decode},
    block::{BlockProfile, CompilationPlan, ExitKind, Features},
};

const PC: u64 = 0x1000;
const PREFIX: u32 = 0x9100_0508; // ADD X8, X8, #1
const SUFFIX: u32 = 0x9100_0529; // ADD X9, X9, #1

fn fraction_bits(format: Format) -> u32 {
    match format {
        Format::Half => 10,
        Format::Single => 23,
        Format::Double => 52,
    }
}
fn sign_mask(format: Format) -> u64 {
    1 << (format.bits() - 1)
}
fn exponent_mask(format: Format) -> u64 {
    (sign_mask(format) - 1) & !((1 << fraction_bits(format)) - 1)
}
fn maximum_normal(format: Format) -> u64 {
    exponent_mask(format) - 1
}

fn forms() -> Vec<u32> {
    let mut forms = Vec::new();
    for operation in 0..4 {
        let scalar = 0x1e20_4800 | operation << 12;
        let vector = [0x0e20_f400, 0x0ea0_f400, 0x0e20_c400, 0x0ea0_c400][operation as usize];
        let pair = [0x7e30_f800, 0x7eb0_f800, 0x7e30_c800, 0x7eb0_c800][operation as usize];
        for format in [0, 1 << 22, 3 << 22] {
            forms.push(scalar | format);
        }
        for arrangement in [0, 1 << 30, (1 << 30) | (1 << 22)] {
            for pairwise in [0, 1 << 29] {
                forms.push(vector | arrangement | pairwise);
            }
        }
        for q in [0, 1 << 30] {
            for pairwise in [0, 1 << 29] {
                forms.push((vector ^ 0x0060_c000) | q | pairwise);
            }
            forms.push((pair ^ 0x7000_0000) | q);
        }
        forms.extend([pair, pair | (1 << 22), pair ^ (1 << 29), pair ^ (1 << 28)]);
    }
    assert_eq!(forms.len(), 76); // 44 S/D arrangements and 32 Half regressions
    forms
}

fn active_inputs(operand: &Operand) -> Vec<(u8, u32)> {
    let format = operand.format();
    let sources = match *operand {
        Operand::Binary {
            shape, left, right, ..
        } => vec![
            (left, shape.lane_count(format)),
            (right, shape.lane_count(format)),
        ],
        Operand::PairwiseScalar { source, .. } => vec![(source, 2)],
        Operand::Reduction { source, shape, .. } => vec![(source, shape.lane_count(format))],
    };
    sources
        .into_iter()
        .flat_map(|(source, count)| (0..count).map(move |lane| (source, lane)))
        .collect()
}

fn input(operand: &Operand, a: u64, b: u64, fpcr: u32) -> GuestState {
    let format = operand.format();
    let poison = exponent_mask(format) | 1; // inactive signaling NaNs
    let mut state = GuestState {
        x: std::array::from_fn(|i| 0x1234_5678_0000_0000 | i as u64),
        sp: 0x1234_0000,
        nzcv: 0xa000_0000,
        exclusive_bytes: 0x9e71_a53c,
        exclusive_address: 0,
        exclusive_value: 0,
        daif: 0,
        pending: 0,
        v: [[0; 2]; 32],
        fpcr,
        fpsr: FPSR_QC | EXCEPTIONS,
        ..Default::default()
    };
    for vector in &mut state.v {
        for lane in 0..Shape::Vector128.lane_count(format) {
            write_lane(vector, format.bits(), lane, poison);
        }
    }
    match *operand {
        Operand::Binary {
            shape,
            left,
            right,
            mode,
            ..
        } => {
            for lane in 0..shape.lane_count(format) {
                let (a, b) = if mode == BinaryMode::Pairwise && lane % 2 != 0 {
                    (b, a)
                } else {
                    (a, b)
                };
                write_lane(&mut state.v[usize::from(left)], format.bits(), lane, a);
                write_lane(&mut state.v[usize::from(right)], format.bits(), lane, b);
            }
        }
        Operand::PairwiseScalar { source, .. } => {
            write_lane(&mut state.v[usize::from(source)], format.bits(), 0, a);
            write_lane(&mut state.v[usize::from(source)], format.bits(), 1, b);
        }
        Operand::Reduction { source, shape, .. } => {
            for lane in 0..shape.lane_count(format) {
                write_lane(
                    &mut state.v[usize::from(source)],
                    format.bits(),
                    lane,
                    if lane % 2 == 0 { a } else { b },
                );
            }
        }
    }
    state
}

fn eligible(state: &GuestState, operand: &Operand) -> bool {
    if state.fpcr & FPCR_EXCEPTION_ENABLE_MASK != 0 {
        return false;
    }
    let format = operand.format();
    let valid = |source: u8, count| {
        (0..count).all(|lane| {
            let bits = read_lane(&state.v[usize::from(source)], format.bits(), lane);
            let exponent = bits & exponent_mask(format);
            let magnitude = bits & !sign_mask(format);
            (exponent != 0 && magnitude <= exponent_mask(format))
                || (format != Format::Half && magnitude == 0)
        })
    };
    match *operand {
        Operand::Binary {
            shape, left, right, ..
        } => valid(left, shape.lane_count(format)) && valid(right, shape.lane_count(format)),
        Operand::PairwiseScalar { source, .. } => valid(source, 2),
        Operand::Reduction { source, shape, .. } => valid(source, shape.lane_count(format)),
    }
}

fn mxcsr() -> u32 {
    let mut value = 0;
    // SAFETY: the output is valid writable storage on this x86-64 test host.
    unsafe {
        core::arch::asm!("stmxcsr [{p}]", p = in(reg) &mut value, options(nostack, preserves_flags));
    }
    value
}

fn set_mxcsr(value: u32) {
    // SAFETY: callers provide only defined baseline SSE control/status bits.
    unsafe {
        core::arch::asm!("ldmxcsr [{p}]", p = in(reg) &value, options(nostack, preserves_flags));
    }
}

fn check(mapping: &Mapping, operand: &Operand, mut state: GuestState, caller: u32) {
    let fast = eligible(&state, operand);
    let mut expected = state.clone();
    expected.x[8] += 1;
    if fast {
        complete(&mut expected, operand).unwrap();
        expected.x[9] += 1;
    }
    let saved = mxcsr();
    set_mxcsr(caller);
    // SAFETY: this owned RX image was compiled from three non-memory words for
    // this host target; state satisfies the ordinary one-pointer block ABI.
    let exit = unsafe { mapping.call_block(&mut state) };
    let actual_csr = mxcsr();
    set_mxcsr(saved);
    assert_eq!(actual_csr, caller, "MXCSR {operand:?}");
    let exit = exit.unwrap();
    assert_eq!(
        exit.next_pc,
        if fast { PC + 12 } else { PC + 4 },
        "{operand:?} fpcr={:#x} input={state:?}",
        state.fpcr
    );
    assert_eq!(
        exit.kind,
        if fast {
            ExitKind::Fallthrough as u64
        } else {
            ExitKind::FloatingPoint as u64
        }
    );
    assert_eq!(
        state, expected,
        "{operand:?} native={fast} fpcr={:#x}",
        state.fpcr
    );
}

#[test]
fn signed_zero_normal_and_infinite_selection_match_full_helper_state() {
    let host = stela_compiler::x86_64::modeled::host();
    let targets = [
        TargetCpu::Baseline,
        TargetCpu::detect(),
        TargetCpu::X86_64_V4.with_avx10_2(),
    ];
    for target in targets.into_iter().filter(|target| host.supports(*target)) {
        for base in forms() {
            for (d, n, m) in [
                (0, 1, 2),
                (1, 1, 2),
                (2, 1, 2),
                (0, 31, 31),
                (31, 0, 31),
                (31, 31, 31),
            ] {
                let binary = matches!(decode(base).unwrap().unwrap(), Operand::Binary { .. });
                let word = base | d | n << 5 | if binary { m << 16 } else { 0 };
                let operand = decode(word).unwrap().unwrap();
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
                let mapping = Mapping::new(&stela_compiler::x86_64::pinned::entered(
                    &block.native.x86_64_code,
                ))
                .unwrap();
                let format = operand.format();
                let one = match format {
                    Format::Half => 0x3c00,
                    Format::Single => 0x3f80_0000,
                    Format::Double => 0x3ff0_0000_0000_0000,
                };
                let sign = sign_mask(format);
                for (a, b) in [
                    (0, 0),
                    (0, sign),
                    (sign, 0),
                    (sign, sign),
                    (0, one),
                    (one | sign, 0),
                    (one, one | sign),
                    (one, one),
                    (1 << fraction_bits(format), one),
                    (maximum_normal(format), maximum_normal(format) | sign),
                    (exponent_mask(format), one),
                    (exponent_mask(format) | sign, one | sign),
                    (exponent_mask(format), exponent_mask(format) | sign),
                    (exponent_mask(format) | sign, exponent_mask(format) | sign),
                    (exponent_mask(format), exponent_mask(format)),
                    (maximum_normal(format), exponent_mask(format)),
                    (exponent_mask(format) | sign, 0),
                    (sign, exponent_mask(format)),
                ] {
                    for flags in 0..32 {
                        let mut fpcr = 0;
                        for (bit, mask) in [FPCR_AH, FPCR_NEP, FPCR_FZ, FPCR_FIZ, FPCR_DN]
                            .into_iter()
                            .enumerate()
                        {
                            if flags & (1 << bit) != 0 {
                                fpcr |= mask;
                            }
                        }
                        for rounding in 0..4 {
                            let mut state =
                                input(&operand, a, b, fpcr | rounding << FPCR_ROUNDING_SHIFT);
                            state.fpsr = if rounding % 2 == 0 {
                                0
                            } else {
                                FPSR_QC | EXCEPTIONS | 0x8000_0000
                            };
                            for caller in [0x1f80, 0xe07f, 0] {
                                // includes unmasked exceptions, DAZ and FTZ
                                check(&mapping, &operand, state.clone(), caller);
                            }
                        }
                    }
                }
                // Distinct zero signs and normal values in each source/lane
                // catch incorrect pair compaction and reduction subtree order.
                for pattern in 0..24 {
                    for fpcr in [
                        0,
                        FPCR_AH,
                        FPCR_NEP,
                        FPCR_AH | FPCR_NEP | FPCR_FZ | FPCR_FZ16 | FPCR_AHP,
                    ] {
                        let mut state = input(&operand, one, one, fpcr);
                        for (position, (source, lane)) in
                            active_inputs(&operand).into_iter().enumerate()
                        {
                            let bits = if pattern < 16 {
                                if pattern & (1 << (position % 4)) == 0 {
                                    0
                                } else {
                                    sign
                                }
                            } else {
                                [
                                    0,
                                    sign,
                                    one,
                                    one | sign,
                                    1 << fraction_bits(format),
                                    maximum_normal(format),
                                    one + 1,
                                    (one + 1) | sign,
                                    exponent_mask(format),
                                    exponent_mask(format) | sign,
                                ][(position + pattern) % 10]
                            };
                            write_lane(
                                &mut state.v[usize::from(source)],
                                format.bits(),
                                lane,
                                bits,
                            );
                        }
                        state.fpsr = 0x8000_0000;
                        check(&mapping, &operand, state, 0x6040);
                    }
                }
            }
        }
    }
}

#[test]
fn exceptional_active_lanes_and_enables_exit_before_any_effect() {
    for base in forms() {
        let binary = matches!(decode(base).unwrap().unwrap(), Operand::Binary { .. });
        let word = base | 31 << 5 | if binary { 1 << 16 } else { 0 };
        let operand = decode(word).unwrap().unwrap();
        let source: Vec<_> = [PREFIX, word, SUFFIX]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let block = CompilationPlan::new(
            &source,
            GuestAddress(PC),
            &stela_compiler::Options::new(
                BlockProfile::new(Features::SYSTEM_MODE),
                TargetCpu::Baseline,
            ),
        )
        .unwrap()
        .compile()
        .unwrap();
        let mapping = Mapping::new(&stela_compiler::x86_64::pinned::entered(
            &block.native.x86_64_code,
        ))
        .unwrap();
        let format = operand.format();
        let sign = sign_mask(format);
        let maximum = exponent_mask(format);
        for bad in [
            1,
            sign | 1,
            (1 << fraction_bits(format)) - 1,
            maximum | 1,
            maximum | sign | 1,
            maximum | (1 << (fraction_bits(format) - 1)),
            if format == Format::Double {
                0x0000_0001_0000_0000
            } else {
                1
            },
        ] {
            for fpcr in [
                0,
                FPCR_AH | FPCR_FZ | FPCR_FIZ | FPCR_NEP | FPCR_DN | FPCR_FZ16 | FPCR_AHP,
            ] {
                for (source, lane) in active_inputs(&operand) {
                    let mut state = input(&operand, 0, sign, fpcr);
                    write_lane(&mut state.v[usize::from(source)], format.bits(), lane, bad);
                    assert!(!eligible(&state, &operand));
                    check(&mapping, &operand, state, 0xe07f);
                }
            }
        }
        for enable in [
            INVALID,
            DIVIDE_BY_ZERO,
            OVERFLOW,
            UNDERFLOW,
            INEXACT,
            INPUT_DENORMAL,
            EXCEPTIONS,
        ] {
            let state = input(
                &operand,
                0,
                sign,
                FPCR_AH | FPCR_NEP | enable << FPCR_EXCEPTION_ENABLE_SHIFT,
            );
            check(&mapping, &operand, state, 0x1f80);
        }
    }
}
