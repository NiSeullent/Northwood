//! The environment-free scalar FP/integer conversions
//! (`integer_conversion::fast_to_integer`, `fast_from_integer`) and Half
//! narrowing (`precision::emit_field_narrow_half`, then F16C) against the
//! reference conversions,
//! on reduced targets: every inline completion matches the reference in its
//! result, FPSR and NZCV; every other run exits at the instruction with the
//! state unchanged. Zeros and in-range normals must complete inline, which
//! only the fast paths do for zeros.
use crate::{
    GuestAddress, GuestState, Options, TargetCpu,
    aarch64::floating::{
        FPCR_AH, FPCR_AHP, FPCR_DN, FPCR_EXCEPTION_ENABLE_MASK, FPCR_FIZ, FPCR_FZ, FPCR_FZ16,
        FPCR_NEP, FPCR_ROUNDING_SHIFT, Format, INEXACT,
        integer::{
            IntegerFormat, IntegerRounding, IntegerWidth, from_integer, to_integer, to_javascript,
        },
        precision::{self, PrecisionRounding},
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
    let mut targets = alloc::vec![
        TargetCpu::Baseline,
        TargetCpu::X86_64_V2,
        TargetCpu::X86_64_V2.with_f16c(),
    ];
    let host = TargetCpu::detect();
    targets.retain(|target| host.supports(*target));
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
        FPCR_FZ | FPCR_AH,
        FPCR_NEP,
        FPCR_DN | FPCR_FZ16,
        FPCR_EXCEPTION_ENABLE_MASK,
        1 << 12, // IXE alone
    ]);
    controls
}

/// Specials, boundaries of every integer range and random values of every
/// exponent, for `format`.
fn floats(format: Format) -> Vec<u64> {
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let double = format == Format::Double;
    let encode = |value: f64| -> u64 {
        if double {
            value.to_bits()
        } else {
            u64::from((value as f32).to_bits())
        }
    };
    let mut values: Vec<u64> = Vec::new();
    for power in [0, 1, 15, 16, 30, 31, 32, 52, 53, 62, 63, 64, 65] {
        let base = 2f64.powi(power);
        for delta in [
            -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, -128.0, 128.0,
        ] {
            values.push(encode(base + delta));
            values.push(encode(-base + delta));
        }
    }
    for value in [
        0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 0.1, -0.1, 1e-3, 65504.0, 65520.0, 6e-5,
    ] {
        values.push(encode(value));
    }
    let (sign, exponent_mask, fraction_bits) = if double {
        (1u64 << 63, 0x7ffu64, 52)
    } else {
        (1u64 << 31, 0xffu64, 23)
    };
    let fraction_mask = (1u64 << fraction_bits) - 1;
    for exponent in 0..=exponent_mask {
        for fraction in [0, 1, fraction_mask, next() & fraction_mask] {
            let bits = exponent << fraction_bits | fraction;
            values.push(bits);
            values.push(bits | sign);
        }
    }
    // NaNs: quiet and signaling, and both zeros.
    values.extend([0, sign, (exponent_mask << fraction_bits) | 1]);
    values
}

fn integers(width: IntegerWidth) -> Vec<u64> {
    let mut values = alloc::vec![0, 1, 2, 3, u64::MAX, u64::MAX - 1, 1 << 24, (1 << 24) + 1];
    for bit in 0..64 {
        values.extend([
            1u64 << bit,
            (1u64 << bit) - 1,
            (1u64 << bit) + 1,
            !(1u64 << bit),
        ]);
    }
    values.extend([
        0x7fff_ffff,
        0x8000_0000,
        0x8000_0001,
        0xffff_ffff,
        0x7fff_ffbf,
        0x7fff_ffc0,
    ]);
    values.iter().map(|value| value & width.mask()).collect()
}

fn width_mask(format: Format) -> u64 {
    if format == Format::Double {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    }
}

/// The vector destination of a scalar result under FPCR.NEP.
fn scalar_vector(old: [u64; 2], bits: u64, mask: u64, fpcr: u32) -> [u64; 2] {
    if fpcr & FPCR_NEP != 0 {
        [old[0] & !mask | bits, old[1]]
    } else {
        [bits, 0]
    }
}

struct Coverage {
    inline: usize,
    fast_inline: usize,
    runs: usize,
}

/// The FP-to-integer forms: (word without size fields, rounding, signed,
/// general destination).
const TO_INTEGER: &[(u32, IntegerRounding, bool, bool)] = &[
    (0x1e38_0000, IntegerRounding::TowardZero, true, true),
    (0x1e39_0000, IntegerRounding::TowardZero, false, true),
    (0x1e20_0000, IntegerRounding::NearestEven, true, true),
    (0x1e21_0000, IntegerRounding::NearestEven, false, true),
    (0x1e28_0000, IntegerRounding::TowardPositive, true, true),
    (0x1e29_0000, IntegerRounding::TowardPositive, false, true),
    (0x1e30_0000, IntegerRounding::TowardNegative, true, true),
    (0x1e31_0000, IntegerRounding::TowardNegative, false, true),
    (0x1e24_0000, IntegerRounding::NearestAway, true, true),
    (0x1e25_0000, IntegerRounding::NearestAway, false, true),
    (0x5ea1_b800, IntegerRounding::TowardZero, true, false),
    (0x7ea1_b800, IntegerRounding::TowardZero, false, false),
    (0x5e21_a800, IntegerRounding::NearestEven, true, false),
    (0x5ea1_a800, IntegerRounding::TowardPositive, true, false),
    (0x5e21_c800, IntegerRounding::NearestAway, true, false),
    (0x7e21_c800, IntegerRounding::NearestAway, false, false),
];

#[test]
fn environment_free_float_to_integer_matches_the_reference() {
    let mut coverage = Coverage {
        inline: 0,
        fast_inline: 0,
        runs: 0,
    };
    for target in targets() {
        for &(base, rounding, signed, general) in TO_INTEGER {
            for format in [Format::Single, Format::Double] {
                let double = format == Format::Double;
                for wide in [false, true] {
                    if !general && wide != double {
                        continue;
                    }
                    let word = if general {
                        base | u32::from(wide) << 31 | u32::from(double) << 22
                    } else {
                        base | u32::from(double) << 22
                    } | (N as u32) << 5
                        | D as u32;
                    let integer = IntegerFormat {
                        width: if wide {
                            IntegerWidth::Bits64
                        } else {
                            IntegerWidth::Bits32
                        },
                        signed,
                    };
                    let code = compile(word, target);
                    let fast = matches!(
                        rounding,
                        IntegerRounding::TowardZero
                            | IntegerRounding::NearestEven
                            | IntegerRounding::NearestAway
                    ) || target.supports(TargetCpu::Sse41);
                    for &bits in &floats(format) {
                        for fpcr in controls() {
                            for fpsr in [0, INEXACT] {
                                let mut initial = GuestState {
                                    fpcr,
                                    fpsr,
                                    nzcv: 0x6000_0000,
                                    ..GuestState::default()
                                };
                                initial.x[30] = RETURN;
                                initial.x[D] = 0xdead_beef_dead_beef;
                                initial.v[N] = [bits | (UPPER[0] & !width_mask(format)), UPPER[1]];
                                initial.v[D] = UPPER;
                                let mut actual = initial.clone();
                                let exit = run(&code, &mut actual);
                                coverage.runs += 1;
                                let reference =
                                    to_integer(format, integer, 0, rounding, fpcr, bits);
                                if exit == PC.0 {
                                    assert_eq!(actual, initial, "{word:08x} {bits:#x} {fpcr:#x}");
                                    let in_range = reference
                                        .is_ok_and(|outcome| outcome.exceptions & !INEXACT == 0);
                                    let exponent = if double {
                                        (bits >> 52) & 0x7ff
                                    } else {
                                        (bits >> 23) & 0xff
                                    };
                                    let zero = bits & !(1 << (format.bits() - 1)) == 0;
                                    let small = exponent != 0
                                        && exponent < format.exponent_bias() as u64 + 20
                                        && (signed || bits >> (format.bits() - 1) == 0);
                                    assert!(
                                        !(fast
                                            && in_range
                                            && fpcr & FPCR_EXCEPTION_ENABLE_MASK == 0
                                            && (zero || small)),
                                        "fast form exited: {word:08x} {bits:#x} {fpcr:#x} {target:?}"
                                    );
                                    continue;
                                }
                                assert_eq!(exit, RETURN, "{word:08x} {bits:#x}");
                                let outcome = reference.expect("completed with enabled traps");
                                let mut expected = initial.clone();
                                expected.fpsr |= outcome.exceptions;
                                if general {
                                    expected.x[D] = outcome.bits;
                                } else {
                                    expected.v[D] = scalar_vector(
                                        UPPER,
                                        outcome.bits,
                                        width_mask(format),
                                        fpcr,
                                    );
                                }
                                assert_eq!(
                                    actual, expected,
                                    "{word:08x} {bits:#x} {fpcr:#x} {target:?}"
                                );
                                coverage.inline += 1;
                                if bits & !(1 << (format.bits() - 1)) == 0 {
                                    coverage.fast_inline += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(coverage.fast_inline > 1000, "{}", coverage.fast_inline);
    assert!(
        coverage.inline > coverage.runs / 4,
        "{} of {}",
        coverage.inline,
        coverage.runs
    );
}

#[test]
fn environment_free_javascript_conversion_matches_the_reference() {
    for target in targets() {
        let word = 0x1e7e_0000 | (N as u32) << 5 | D as u32;
        let code = compile(word, target);
        let mut zeros = 0;
        for &bits in &floats(Format::Double) {
            for fpcr in controls() {
                for fpsr in [0, INEXACT] {
                    let mut initial = GuestState {
                        fpcr,
                        fpsr,
                        nzcv: 0xf000_0000,
                        ..GuestState::default()
                    };
                    initial.x[30] = RETURN;
                    initial.x[D] = 0xdead_beef_dead_beef;
                    initial.v[N] = [bits, UPPER[1]];
                    let mut actual = initial.clone();
                    let exit = run(&code, &mut actual);
                    if exit == PC.0 {
                        assert_eq!(actual, initial);
                        continue;
                    }
                    assert_eq!(exit, RETURN);
                    let outcome = to_javascript(fpcr, bits).expect("no enabled traps");
                    let mut expected = initial.clone();
                    expected.fpsr |= outcome.result.exceptions;
                    expected.x[D] = outcome.result.bits;
                    expected.nzcv = outcome.nzcv;
                    assert_eq!(actual, expected, "{bits:#x} {fpcr:#x} {target:?}");
                    zeros += usize::from(bits == 0);
                }
            }
        }
        // +0 completes inline under every control without enabled traps
        // (each zero input, both FPSR values).
        let zero_inputs = floats(Format::Double)
            .iter()
            .filter(|&&bits| bits == 0)
            .count();
        assert_eq!(
            zeros,
            zero_inputs * 2 * (controls().len() - 2),
            "{target:?}"
        );
    }
}

#[test]
fn environment_free_integer_to_float_matches_the_reference() {
    // (word without size fields, signed, general source)
    let forms: &[(u32, bool, bool)] = &[
        (0x1e22_0000, true, true),
        (0x1e23_0000, false, true),
        (0x5e21_d800, true, false),
        (0x7e21_d800, false, false),
    ];
    let mut fast_inline = 0;
    for target in targets() {
        for &(base, signed, general) in forms {
            for format in [Format::Single, Format::Double] {
                let double = format == Format::Double;
                for wide in [false, true] {
                    if !general && wide != double {
                        continue;
                    }
                    let word = if general {
                        base | u32::from(wide) << 31 | u32::from(double) << 22
                    } else {
                        base | u32::from(double) << 22
                    } | (N as u32) << 5
                        | D as u32;
                    let width = if wide {
                        IntegerWidth::Bits64
                    } else {
                        IntegerWidth::Bits32
                    };
                    let integer = IntegerFormat { width, signed };
                    let code = compile(word, target);
                    for &value in &integers(width) {
                        for fpcr in controls() {
                            for fpsr in [0, INEXACT] {
                                let mut initial = GuestState {
                                    fpcr,
                                    fpsr,
                                    ..GuestState::default()
                                };
                                initial.x[30] = RETURN;
                                initial.x[N] = value | 0xdead_beef_0000_0000 & !width.mask();
                                initial.v[N] = [value | UPPER[0] & !width.mask(), UPPER[1]];
                                initial.v[D] = UPPER;
                                let mut actual = initial.clone();
                                let exit = run(&code, &mut actual);
                                let rounding_default = fpcr & (3 << FPCR_ROUNDING_SHIFT) == 0
                                    && fpcr & FPCR_EXCEPTION_ENABLE_MASK == 0;
                                if exit == PC.0 {
                                    assert_eq!(actual, initial);
                                    assert!(
                                        !(rounding_default
                                            && (signed || !wide || value >> 63 == 0)),
                                        "fast form exited: {word:08x} {value:#x} {fpcr:#x} {target:?}"
                                    );
                                    continue;
                                }
                                assert_eq!(exit, RETURN);
                                let outcome = from_integer(integer, format, 0, fpcr, value)
                                    .expect("no enabled traps");
                                let mut expected = initial.clone();
                                expected.fpsr |= outcome.exceptions;
                                expected.v[D] =
                                    scalar_vector(UPPER, outcome.bits, width_mask(format), fpcr);
                                assert_eq!(
                                    actual, expected,
                                    "{word:08x} {value:#x} {fpcr:#x} {target:?}"
                                );
                                fast_inline += usize::from(rounding_default);
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(fast_inline > 1000, "{fast_inline}");
}

#[test]
fn environment_free_half_narrowing_matches_the_reference() {
    let mut targets = targets();
    let host = TargetCpu::detect();
    if host.supports(TargetCpu::X86_64_V3) {
        targets.push(TargetCpu::X86_64_V3);
    }
    let singles = floats(Format::Single);
    let mut seed = 0x9e37_79b9_u64;
    let mut pick = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        singles[(seed % singles.len() as u64) as usize]
    };
    let mut inline_rne = 0;
    let mut inline_tiny = 0;
    for target in targets {
        // FCVT Hd, Sn; FCVTN Vd.4H, Vn.4S; FCVTN2 Vd.8H, Vn.4S.
        for (word, lanes, upper) in [
            (0x1e23_c000u32, 1, false),
            (0x0e21_6800, 4, false),
            (0x4e21_6800, 4, true),
        ] {
            let word = word | (N as u32) << 5 | D as u32;
            let code = compile(word, target);
            if !target.supports(TargetCpu::F16c) {
                // Without F16C only the field path is inline: no MXCSR access.
                assert!(!code.windows(2).any(|pair| pair == [0x0f, 0xae]));
            }
            for case in 0..6000 {
                let mut inputs = [0u64; 4];
                for lane in inputs.iter_mut().take(lanes) {
                    *lane = pick();
                }
                if case % 3 == 0 {
                    // Mostly ordinary values, every lane of one magnitude class.
                    for lane in inputs.iter_mut().take(lanes) {
                        *lane = u64::from((1.0f32 + (*lane as u32 % 1000) as f32 / 7.0).to_bits());
                    }
                }
                if case % 7 == 0 && lanes == 4 {
                    inputs[case % 4] = 0;
                }
                if case % 5 == 0 {
                    // Zeros of both signs and the field range bounds.
                    const EDGES: [u64; 8] = [
                        0,
                        0x8000_0000,
                        0x3880_0000,
                        0x387f_ffff,
                        0x477f_efff,
                        0x477f_f000,
                        0x477f_e000,
                        0x3880_1000,
                    ];
                    for (lane, input) in inputs.iter_mut().take(lanes).enumerate() {
                        *input = EDGES[(case / 5 + lane * 3) % EDGES.len()]
                            | u64::from(case % 2 == 1) << 31;
                    }
                }
                let field = inputs.iter().take(lanes).all(|&bits| {
                    let magnitude = bits & 0x7fff_ffff;
                    magnitude == 0 || (0x3880_0000..=0x477f_efff).contains(&magnitude)
                });
                // A vector narrowing of normal or zero lanes that overflow in
                // no rounding mode completes inline with F16C, tiny lanes
                // included, while FPCR.AH leaves tininess to the helper.
                let narrowable = lanes == 4
                    && target.supports(TargetCpu::F16c)
                    && inputs.iter().all(|&bits| {
                        let magnitude = bits & 0x7fff_ffff;
                        magnitude == 0 || (0x0080_0000..=0x477f_e000).contains(&magnitude)
                    });
                let tiny = inputs.iter().any(|&bits| {
                    let magnitude = bits & 0x7fff_ffff;
                    magnitude != 0 && magnitude < 0x3880_0000
                });
                let mut controls = controls();
                controls.push(FPCR_AHP);
                for fpcr in controls {
                    for fpsr in [0, INEXACT] {
                        let mut initial = GuestState {
                            fpcr,
                            fpsr,
                            ..GuestState::default()
                        };
                        initial.x[30] = RETURN;
                        initial.v[N] = [inputs[0] | inputs[1] << 32, inputs[2] | inputs[3] << 32];
                        if lanes == 1 {
                            initial.v[N][0] |= UPPER[0] & !0xffff_ffff;
                            initial.v[N][1] = UPPER[1];
                        }
                        initial.v[D] = UPPER;
                        let mut actual = initial.clone();
                        let exit = run(&code, &mut actual);
                        if exit == PC.0 {
                            assert_eq!(actual, initial);
                            assert!(
                                !(narrowable
                                    && fpcr & (FPCR_EXCEPTION_ENABLE_MASK | FPCR_AHP | FPCR_AH)
                                        == 0),
                                "vector narrowing exited: {word:08x} {inputs:x?} {fpcr:#x} {target:?}"
                            );
                            assert!(
                                !(field
                                    && fpcr
                                        & (FPCR_EXCEPTION_ENABLE_MASK
                                            | FPCR_AHP
                                            | 3 << FPCR_ROUNDING_SHIFT)
                                        == 0),
                                "field form exited: {word:08x} {inputs:x?} {fpcr:#x} {target:?}"
                            );
                            continue;
                        }
                        // The conversion may end its block: completion then
                        // exits at the next instruction instead of the RET.
                        assert!(
                            exit == RETURN || exit == PC.0 + 4,
                            "{word:08x} {inputs:x?} {fpcr:#x} {fpsr:#x} {target:?}: {exit:#x}"
                        );
                        let mut exceptions = 0;
                        let mut halves = 0u64;
                        for (lane, &bits) in inputs.iter().enumerate().take(lanes) {
                            let outcome = precision::evaluate(
                                Format::Single,
                                Format::Half,
                                PrecisionRounding::Fpcr,
                                fpcr,
                                bits,
                            )
                            .expect("no enabled traps");
                            exceptions |= outcome.exceptions;
                            halves |= outcome.bits << (16 * lane);
                        }
                        let mut expected = initial.clone();
                        expected.fpsr |= exceptions;
                        expected.v[D] = match (lanes, upper) {
                            (1, _) => scalar_vector(UPPER, halves, 0xffff, fpcr),
                            (_, false) => [halves, 0],
                            (_, true) => [UPPER[0], halves],
                        };
                        assert_eq!(
                            actual, expected,
                            "{word:08x} {inputs:x?} {fpcr:#x} {target:?}"
                        );
                        inline_rne += usize::from(fpcr & (3 << FPCR_ROUNDING_SHIFT) == 0);
                        inline_tiny += usize::from(lanes == 4 && tiny);
                    }
                }
            }
        }
    }
    assert!(inline_rne > 10_000, "{inline_rne}");
    if TargetCpu::detect().supports(TargetCpu::F16c) {
        assert!(inline_tiny > 1_000, "{inline_tiny}");
    }
}
