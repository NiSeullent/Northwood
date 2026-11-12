//! Native block dispatch.
#![cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
use anyhow::Result;
use std::path::{Path, PathBuf};
use stela_compiler::{GuestAddress, GuestState};
use stela_engine::blocks::{BlockRuntime, StopReason, ThreadState};

const BASE: u64 = 0x1000;
const STOP: u64 = 0xdead_0000;
const RET: u32 = 0xd65f03c0;
const NOP: u32 = 0xd503201f;
const ADD_ONE: u32 = 0x91000400; // ADD X0, X0, #1

fn words(input: &[u32]) -> Vec<u8> {
    input.iter().flat_map(|word| word.to_le_bytes()).collect()
}
fn state(pc: u64, value: u64) -> ThreadState {
    let mut registers = GuestState::default();
    registers.x[0] = value;
    registers.x[30] = STOP;
    ThreadState {
        registers,
        pc: GuestAddress(pc),
    }
}
fn transfer(pc: u64, target: u64, call: bool) -> u32 {
    let displacement = (target as i64 - pc as i64) / 4;
    assert_eq!((target as i64 - pc as i64) % 4, 0);
    assert!((-(1 << 25)..(1 << 25)).contains(&displacement));
    (if call { 0x94000000 } else { 0x14000000 }) | (displacement as u32 & 0x03ffffff)
}
fn mov(destination: u32, source: u32) -> u32 {
    0xaa0003e0 | source << 16 | destination
}

mod numerical {
    use super::*;
    use stela_compiler::{
        TargetCpu,
        aarch64::floating::{INEXACT, INVALID},
        block::BlockProfile,
    };

    const FADD_S0_S1_S2: u32 = 0x1e222820;
    const FADD_H3_H4_H5: u32 = 0x1ee52883;
    const FMUL_D6_D7_D8: u32 = 0x1e6808e6;
    const FPCR_NEP: u32 = 1 << 2;
    const ROUNDING_SHIFT: u32 = 22;
    const EXCEPTION_ENABLE_SHIFT: u32 = 8;
    const FPSR_QC: u32 = 1 << 27;

    fn input(nep: bool) -> ThreadState {
        let mut thread = state(BASE, 19);
        thread.registers.v =
            std::array::from_fn(|i| [0xdeadbeef01234567 ^ i as u64, 0x89abcdef76543210 ^ i as u64]);
        thread.registers.v[1][0] = 0xdeadbeef3f800000;
        thread.registers.v[2][0] = 0x89abcdef40000000;
        thread.registers.v[4][0] = 0xdeadbeef01233c00;
        thread.registers.v[5][0] = 0x89abcdef76544000;
        thread.registers.v[7][0] = 0x4008000000000000;
        thread.registers.v[8][0] = 0x4000000000000000;
        thread.registers.fpcr = if nep { FPCR_NEP } else { 0 };
        thread.registers.fpsr = FPSR_QC | 0x80000000;
        thread.registers.nzcv = 0xa0000000;
        thread.registers.exclusive_bytes = 0xa53cf012;
        thread
    }
    fn publish(expected: &mut GuestState, d: usize, n: usize, bits: u64, width: u32) {
        let mask = u64::MAX >> (u64::BITS - width);
        let mut value = if expected.fpcr & FPCR_NEP != 0 {
            expected.v[n]
        } else {
            [0; 2]
        };
        value[0] = (value[0] & !mask) | bits;
        expected.v[d] = value;
    }

    /// A zero dividend over a normal divisor, and a zero widened from S to
    /// D, complete in compiled code with the exact signed zero and no
    /// status, on the full path (FPSR.IXC clear) and on the sticky one;
    /// a zero divisor still takes the helper, which raises DZC.
    #[test]
    fn zero_dividends_and_zero_widenings_stay_in_compiled_code() -> Result<()> {
        use stela_compiler::aarch64::floating::{DIVIDE_BY_ZERO, FPCR_FZ};
        const FDIV_S0_S1_S2: u32 = 0x1e22_1820;
        const FCVT_D3_S1: u32 = 0x1e22_c023;
        const THREE: u64 = 0x4040_0000;
        let cache = tempfile::tempdir()?;
        let mut runtime = BlockRuntime::new_for_target(
            76,
            cache.path().into(),
            BlockProfile::ORDINARY,
            TargetCpu::Baseline,
        );
        runtime.register_code(
            GuestAddress(BASE),
            words(&[ADD_ONE, FDIV_S0_S1_S2, FCVT_D3_S1, ADD_ONE, RET]),
        )?;
        for (zero, wide) in [(0u64, 0u64), (0x8000_0000, 0x8000_0000_0000_0000)] {
            for (fpcr, fpsr) in [(0, 0), (0, INEXACT), (FPCR_FZ, 0), (FPCR_NEP, INEXACT)] {
                let mut thread = input(false);
                thread.registers.fpcr = fpcr;
                thread.registers.fpsr = fpsr;
                thread.registers.v[1][0] = 0xdead_beef_0000_0000 | zero;
                thread.registers.v[2][0] = 0x89ab_cdef_0000_0000 | THREE;
                let mut expected = thread.registers.clone();
                expected.x[0] += 2;
                publish(&mut expected, 0, 1, zero, 32);
                let old = expected.v[3];
                expected.v[3] = if fpcr & FPCR_NEP != 0 {
                    [wide, old[1]]
                } else {
                    [wide, 0]
                };
                // SAFETY: the test program reads and writes only its own registered guest data, and nothing
                // else mutates it.
                let done = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 8)? };
                assert_eq!(done.reason, StopReason::StopAddress);
                assert_eq!(
                    done.floating_point_helpers, 0,
                    "{zero:#x} {fpcr:#x} {fpsr:#x}"
                );
                assert_eq!(thread.registers, expected, "{zero:#x} {fpcr:#x} {fpsr:#x}");
            }
        }
        let mut thread = input(false);
        thread.registers.fpsr = 0;
        thread.registers.v[1][0] = THREE;
        thread.registers.v[2][0] = 0;
        // SAFETY: as above.
        let done = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 8)? };
        assert_eq!(done.reason, StopReason::StopAddress);
        assert_eq!(done.floating_point_helpers, 1);
        assert_eq!(thread.registers.fpsr, DIVIDE_BY_ZERO);
        assert_eq!(thread.registers.v[0][0], 0x7f80_0000);
        Ok(())
    }

    #[test]
    fn packed_truncation_retires_from_native_status_when_flush_has_no_effect() -> Result<()> {
        use stela_compiler::aarch64::floating::{
            FPCR_AH, FPCR_FIZ, FPCR_FZ, Format,
            integer::{IntegerFormat, IntegerRounding, IntegerWidth, to_integer},
        };

        let cache = tempfile::tempdir()?;
        let mut runtime = BlockRuntime::new_for_target(
            76,
            cache.path().into(),
            BlockProfile::ORDINARY,
            TargetCpu::Baseline,
        );
        // The two archived 4S forms followed by a 2S form with poisoned high lanes.
        runtime.register_code(
            GuestAddress(BASE),
            words(&[ADD_ONE, 0x4ea1_b884, 0x4ea1_b863, 0x0ea1_b842, ADD_ONE, RET]),
        )?;
        let pack = |low: u32, high: u32| u64::from(low) | (u64::from(high) << 32);
        let integer = IntegerFormat {
            width: IntegerWidth::Bits32,
            signed: true,
        };
        let small = [0, 0x8000_0000, 1, 0x807f_ffff];
        let signed_zeros = [0, 0x8000_0000, 0x3fe0_0000, 0xc020_0000];
        let normals = [0x3fe0_0000, 0xc020_0000, 0x4eff_ffff, 0xcf00_0000];
        for (fpcr, lanes, helpers) in [
            (0, small, 0),
            (FPCR_AH, small, 0),
            (FPCR_FZ, small, 1),
            (FPCR_FIZ, small, 1),
            (FPCR_AH | FPCR_FZ, small, 1),
            (FPCR_FZ, signed_zeros, 0),
            (FPCR_FIZ, signed_zeros, 0),
            (FPCR_AH | FPCR_FZ, signed_zeros, 0),
            (FPCR_AH | FPCR_FZ | FPCR_FIZ, signed_zeros, 0),
            (FPCR_FZ, [0x0080_0000, 0x8080_0000, 0, 0x8000_0000], 0),
            (FPCR_FZ | FPCR_FIZ, normals, 0),
            (0, normals, 0),
            (0, [0x7f80_0001, 0x7fc0_1234, 0xc020_0000, 0xff80_0000], 1),
            (
                FPCR_FZ,
                [0x7f80_0001, 0x7fc0_1234, 0xc020_0000, 0xff80_0000],
                1,
            ),
            (0, [0x4f00_0000, 0xcf00_0001, 1, 0x8000_0000], 1),
            (FPCR_FIZ, [0x4f00_0000, 0xcf00_0001, 0, 0x8000_0000], 1),
        ] {
            let mut thread = input(false);
            thread.registers.fpcr = fpcr;
            thread.registers.v[4] = [pack(lanes[0], lanes[1]), pack(lanes[2], lanes[3])];
            thread.registers.v[3] = [pack(0, 0x3f80_0000), pack(0x8000_0000, 0xc000_0000)];
            thread.registers.v[2] = [pack(0x8000_0000, 0x4080_0000), 0x7f80_0001_7f80_0001];
            let mut expected = thread.registers.clone();
            let converted = lanes.map(|lane| {
                let result = to_integer(
                    Format::Single,
                    integer,
                    0,
                    IntegerRounding::TowardZero,
                    fpcr,
                    u64::from(lane),
                )
                .expect("supported FP controls");
                expected.fpsr |= result.exceptions;
                result.bits as u32
            });
            expected.x[0] += 2;
            expected.v[4] = [
                pack(converted[0], converted[1]),
                pack(converted[2], converted[3]),
            ];
            expected.v[3] = [pack(0, 1), pack(0, (-2i32) as u32)];
            expected.v[2] = [pack(0, 4), 0];
            // SAFETY: the program only operates on its own registered code and state.
            let report = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 8)? };
            assert_eq!(report.reason, StopReason::StopAddress);
            assert_eq!(report.floating_point_helpers, helpers);
            assert_eq!(thread.registers, expected);
        }
        Ok(())
    }

    #[test]
    fn interior_half_helper_resumes_after_prefix_and_reuses_disk_code() -> Result<()> {
        let cache = tempfile::tempdir()?;
        let source = words(&[ADD_ONE, FADD_S0_S1_S2, FADD_H3_H4_H5, FMUL_D6_D7_D8, RET]);
        for target in [TargetCpu::Baseline, TargetCpu::detect()] {
            for nep in [false, true] {
                for reused in [false, true] {
                    let mut runtime = BlockRuntime::new_for_target(
                        71,
                        cache.path().into(),
                        BlockProfile::ORDINARY,
                        target,
                    );
                    runtime.register_code(GuestAddress(BASE), source.clone())?;
                    let mut thread = input(nep);
                    // An infinity is outside every native Half domain, so this
                    // instruction takes its interior helper without replaying
                    // the completed prefix or prematurely executing the suffix.
                    thread.registers.v[5][0] = thread.registers.v[5][0] & !0xffff | 0x7c00;
                    let mut prefix = thread.registers.clone();
                    prefix.x[0] += 1;
                    publish(&mut prefix, 0, 1, 0x40400000, 32);
                    publish(&mut prefix, 3, 4, 0x7c00, 16);
                    // SAFETY: the test program reads and writes only its own registered guest data,
                    // and nothing else mutates it.
                    let first = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 1)? };
                    assert_eq!(first.reason, StopReason::BudgetExhausted);
                    assert_eq!(first.floating_point_helpers, 1);
                    assert_eq!(thread.pc, GuestAddress(BASE + 12));
                    assert_eq!(thread.registers, prefix);
                    let mut expected = prefix;
                    publish(&mut expected, 6, 7, 0x4018000000000000, 64);
                    // SAFETY: the test program reads and writes only its own registered guest data,
                    // and nothing else mutates it.
                    let rest = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 8)? };
                    assert_eq!(rest.reason, StopReason::StopAddress);
                    assert_eq!(rest.blocks_executed, 1);
                    assert_eq!(rest.floating_point_helpers, 0);
                    assert_eq!(thread.registers, expected);
                    if reused {
                        assert!(first.disk_cache_hits > 0 && rest.disk_cache_hits > 0);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn native_half_arithmetic_keeps_the_suffix_in_one_cached_block() -> Result<()> {
        let cache = tempfile::tempdir()?;
        let source = words(&[ADD_ONE, FADD_S0_S1_S2, FADD_H3_H4_H5, FMUL_D6_D7_D8, RET]);
        for target in [TargetCpu::Baseline, TargetCpu::detect()] {
            for nep in [false, true] {
                for reused in [false, true] {
                    let mut runtime = BlockRuntime::new_for_target(
                        72,
                        cache.path().into(),
                        BlockProfile::ORDINARY,
                        target,
                    );
                    runtime.register_code(GuestAddress(BASE), source.clone())?;
                    let mut thread = input(nep);
                    let mut expected = thread.registers.clone();
                    expected.x[0] += 1;
                    publish(&mut expected, 0, 1, 0x40400000, 32);
                    publish(&mut expected, 3, 4, 0x4200, 16);
                    publish(&mut expected, 6, 7, 0x4018000000000000, 64);
                    // SAFETY: the test program reads and writes only its own registered guest data,
                    // and nothing else mutates it.
                    let report = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 1)? };
                    assert_eq!(report.reason, StopReason::StopAddress);
                    assert_eq!(report.blocks_executed, 1);
                    assert_eq!(report.floating_point_helpers, 0);
                    assert_eq!(thread.registers, expected);
                    if reused {
                        assert!(report.disk_cache_hits > 0);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn enabled_traps_complete_an_instruction_that_raises_nothing() -> Result<()> {
        let cache = tempfile::tempdir()?;
        let mut runtime = BlockRuntime::new_for_target(
            72,
            cache.path().into(),
            BlockProfile::ORDINARY,
            TargetCpu::Baseline,
        );
        runtime.register_code(
            GuestAddress(BASE),
            words(&[ADD_ONE, FADD_S0_S1_S2, ADD_ONE, RET]),
        )?;
        let mut thread = input(true);
        thread.registers.fpcr |= INVALID << EXCEPTION_ENABLE_SHIFT;
        let mut expected = thread.registers.clone();
        expected.x[0] += 2;
        publish(&mut expected, 0, 1, 0x40400000, 32);
        // SAFETY: the test program reads and writes only its own registered guest data, and nothing
        // else mutates it.
        let done = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 8)? };
        assert_eq!(done.reason, StopReason::StopAddress);
        assert_eq!(thread.registers, expected);
        Ok(())
    }

    #[test]
    fn enabled_exception_traps_at_its_instruction_and_resumes_without_prefix_replay() -> Result<()>
    {
        use stela_compiler::aarch64::floating::{INEXACT, OVERFLOW};
        const SINGLE_MAX: u64 = 0x7f7f_ffff;
        let cache = tempfile::tempdir()?;
        let mut runtime = BlockRuntime::new_for_target(
            76,
            cache.path().into(),
            BlockProfile::ORDINARY,
            TargetCpu::Baseline,
        );
        runtime.register_code(
            GuestAddress(BASE),
            words(&[ADD_ONE, FADD_S0_S1_S2, ADD_ONE, RET]),
        )?;
        let mut thread = input(true);
        thread.registers.v[1][0] = 0xdead_beef_0000_0000 | SINGLE_MAX;
        thread.registers.v[2][0] = 0x89ab_cdef_0000_0000 | SINGLE_MAX;
        thread.registers.fpsr = 0;
        thread.registers.fpcr |= OVERFLOW << EXCEPTION_ENABLE_SHIFT;
        let mut prefix = thread.registers.clone();
        prefix.x[0] += 1;
        // The trapped overflow sets no FPSR bit; the inexact result, whose
        // trap is disabled, sets IXC.
        prefix.fpsr = INEXACT;
        // SAFETY: the test program reads and writes only its own registered guest data, and nothing
        // else mutates it.
        let stop = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 8)? };
        assert_eq!(
            stop.reason,
            StopReason::FloatingPointTrapped {
                instruction_pc: GuestAddress(BASE + 4),
                fpcr: prefix.fpcr,
                exceptions: OVERFLOW
            }
        );
        assert_eq!(stop.floating_point_helpers, 0);
        assert_eq!(thread.registers, prefix);
        assert_eq!(thread.pc, GuestAddress(BASE + 4));
        // SAFETY: the test program reads and writes only its own registered guest data, and nothing
        // else mutates it.
        let zero = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 0)? };
        assert_eq!(zero.blocks_executed, 0);
        assert_eq!(thread.registers, prefix);
        thread.registers.fpcr = FPCR_NEP;
        let mut expected = thread.registers.clone();
        expected.x[0] += 1;
        expected.fpsr |= OVERFLOW | INEXACT;
        publish(&mut expected, 0, 1, 0x7f800000, 32);
        // SAFETY: the test program reads and writes only its own registered guest data, and nothing
        // else mutates it.
        let done = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 8)? };
        assert_eq!(done.reason, StopReason::StopAddress);
        assert_eq!(thread.registers, expected);
        Ok(())
    }

    #[test]
    fn native_rounding_reads_each_current_control_through_same_cached_code() -> Result<()> {
        let cache = tempfile::tempdir()?;
        for rounding in 0..4 {
            let mut runtime = BlockRuntime::new_for_target(
                73,
                cache.path().into(),
                BlockProfile::ORDINARY,
                TargetCpu::Baseline,
            );
            runtime.register_code(GuestAddress(BASE), words(&[FADD_S0_S1_S2, RET]))?;
            let mut thread = input(true);
            thread.registers.fpcr |= rounding << ROUNDING_SHIFT;
            thread.registers.v[2][0] = 0x33800000; // half an S ulp at one
            let mut expected = thread.registers.clone();
            publish(
                &mut expected,
                0,
                1,
                if rounding == 1 {
                    0x3f800001
                } else {
                    0x3f800000
                },
                32,
            );
            expected.fpsr |= INEXACT;
            // SAFETY: the test program reads and writes only its own registered guest data, and
            // nothing else mutates it.
            let done = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 8)? };
            assert_eq!(done.reason, StopReason::StopAddress);
            assert_eq!(done.blocks_executed, 1);
            assert_eq!(done.floating_point_helpers, 0);
            assert_eq!(thread.registers, expected);
            if rounding > 0 {
                assert!(done.disk_cache_hits > 0);
            }
        }
        Ok(())
    }

    #[test]
    fn finite_half_widening_stays_native_for_both_source_halves() -> Result<()> {
        use stela_compiler::aarch64::floating::{FPCR_AH, FPCR_AHP, FPCR_FZ, FPCR_FZ16};

        let cache = tempfile::tempdir()?;
        let mut runtime = BlockRuntime::new_for_target(
            74,
            cache.path().into(),
            BlockProfile::ORDINARY,
            TargetCpu::Baseline,
        );
        // FCVTL V0.4S,V1.4H and FCVTL2 V2.4S,V1.8H.
        runtime.register_code(
            GuestAddress(BASE),
            words(&[ADD_ONE, 0x0e21_7820, 0x4e21_7822, ADD_ONE, RET]),
        )?;
        let mut initial = input(false);
        initial.registers.fpcr = FPCR_AH | FPCR_AHP | FPCR_FZ | FPCR_FZ16;
        initial.registers.v[1] = [0x3c00_0001_8000_0000, 0x0000_c000_0400_8001];
        let mut expected = initial.registers.clone();
        expected.x[0] += 2;
        expected.v[0] = [0x8000_0000_0000_0000, 0x3f80_0000_3380_0000];
        expected.v[2] = [0x3880_0000_b380_0000, 0x0000_0000_c000_0000];
        let mut thread = initial.clone();
        // SAFETY: the program only operates on its own registered code and state.
        let report = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 1)? };
        assert_eq!(report.reason, StopReason::StopAddress);
        assert_eq!(report.floating_point_helpers, 0);
        assert_eq!(thread.registers, expected);

        // An enabled trap the widening raises no exception for changes
        // nothing.
        initial.registers.fpcr |= INVALID << EXCEPTION_ENABLE_SHIFT;
        expected.fpcr = initial.registers.fpcr;
        // SAFETY: the program only operates on its own registered code and state.
        let report = unsafe { runtime.run(&mut initial, Some(GuestAddress(STOP)), 8)? };
        assert_eq!(report.reason, StopReason::StopAddress);
        assert_eq!(initial.registers, expected);
        Ok(())
    }

    #[test]
    fn vector_self_comparison_exits_only_for_exceptional_lanes() -> Result<()> {
        let cache = tempfile::tempdir()?;
        let mut runtime = BlockRuntime::new_for_target(
            75,
            cache.path().into(),
            BlockProfile::ORDINARY,
            TargetCpu::Baseline,
        );
        // FCMEQ V0.4S,V1.4S,V1.4S and FCMEQ V2.2S,V2.2S,V2.2S.
        runtime.register_code(
            GuestAddress(BASE),
            words(&[ADD_ONE, 0x4e21_e420, 0x0e22_e442, ADD_ONE, RET]),
        )?;
        for exceptional in [false, true] {
            let mut thread = input(false);
            thread.registers.v[1] = if exceptional {
                [0x8000_0000_7fc0_0001, 0x7f80_0000_7f80_0001]
            } else {
                [0x8000_0000_0000_0000, 0xff80_0000_7f80_0000]
            };
            // The aliased 2S form clears the old signaling NaNs in the high half.
            thread.registers.v[2] = [0x3f80_0000_8000_0000, 0x7f80_0001_7f80_0001];
            let mut expected = thread.registers.clone();
            expected.x[0] += 2;
            expected.v[2] = [u64::MAX, 0];
            if exceptional {
                expected.v[0] = [0xffff_ffff_0000_0000; 2];
                expected.fpsr |= INVALID;
            } else {
                expected.v[0] = [u64::MAX; 2];
            }
            // SAFETY: the program only operates on its own registered code and state.
            let report = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 8)? };
            assert_eq!(report.reason, StopReason::StopAddress);
            assert_eq!(report.floating_point_helpers, usize::from(exceptional));
            assert_eq!(thread.registers, expected);
        }
        Ok(())
    }

    /// Run `program` once with `target` and once with the baseline target,
    /// whose helper completes every listed FP word exactly, and return both
    /// final states with the native run's helper count.
    fn against_helper(
        program: &[u32],
        target: TargetCpu,
        thread: &ThreadState,
    ) -> Result<(GuestState, GuestState, usize)> {
        let mut states = Vec::new();
        let mut helpers = 0;
        for (index, target) in [target, TargetCpu::Baseline].into_iter().enumerate() {
            let cache = tempfile::tempdir()?;
            let mut runtime = BlockRuntime::new_for_target(
                77,
                cache.path().into(),
                BlockProfile::ORDINARY,
                target,
            );
            runtime.register_code(GuestAddress(BASE), words(program))?;
            let mut run = thread.clone();
            // SAFETY: the program only operates on its own registered code and state.
            let report = unsafe { runtime.run(&mut run, Some(GuestAddress(STOP)), 8)? };
            assert_eq!(report.reason, StopReason::StopAddress);
            if index == 0 {
                helpers = report.floating_point_helpers;
            }
            states.push(run.registers);
        }
        let reference = states.pop().unwrap();
        Ok((states.pop().unwrap(), reference, helpers))
    }

    /// Vector FMLA and FMLS of S and D lanes, with normal and zero lanes,
    /// finish in compiled code with FMA3 on the full path (FPSR.IXC clear)
    /// and on the sticky one, with the helper's exact results and status.
    #[test]
    fn vector_fused_products_stay_in_compiled_code_with_fma3() -> Result<()> {
        let target = TargetCpu::detect();
        if !target.supports(TargetCpu::Fma) {
            return Ok(());
        }
        // FMLA V0.4S,V1.4S,V2.4S; FMLS V3.4S,V1.4S,V2.4S; FMLA V5.2D,V6.2D,V7.2D.
        let program = [ADD_ONE, 0x4e22_cc20, 0x4ea2_cc23, 0x4e67_ccc5, ADD_ONE, RET];
        let singles = |lanes: [u32; 4]| {
            [
                u64::from(lanes[0]) | u64::from(lanes[1]) << 32,
                u64::from(lanes[2]) | u64::from(lanes[3]) << 32,
            ]
        };
        for fpsr in [0, INEXACT] {
            for rounding in 0..4 {
                let mut thread = input(false);
                thread.registers.fpsr = fpsr;
                thread.registers.fpcr = rounding << ROUNDING_SHIFT;
                thread.registers.v[0] = singles([0x3f80_0000, 0xbf2a_aaab, 0, 0x4049_0fdb]);
                thread.registers.v[1] = singles([0x3eaa_aaab, 0x3f35_04f3, 0x8000_0000, 0]);
                thread.registers.v[2] = singles([0x3dcc_cccd, 0xc0a0_0000, 0x3f80_0000, 0]);
                thread.registers.v[3] = singles([0x4120_0000, 0, 0x3e4c_cccd, 0x8000_0000]);
                thread.registers.v[5] = [0x3fd5_5555_5555_5555, 0];
                thread.registers.v[6] = [0x3fb9_9999_9999_999a, 0x4000_0000_0000_0000];
                thread.registers.v[7] = [0xbff0_0000_0000_0001, 0x8000_0000_0000_0000];
                let (native, reference, helpers) = against_helper(&program, target, &thread)?;
                assert_eq!(helpers, 0, "{fpsr:#x} {rounding}");
                assert_eq!(native, reference, "{fpsr:#x} {rounding}");
                assert_eq!(native.fpsr & INEXACT, INEXACT);
            }
        }
        Ok(())
    }

    /// Vector ordered comparisons order zero and infinite lanes in compiled
    /// code; a NaN (IOC) or a subnormal (FZ) lane still takes the helper.
    #[test]
    fn vector_comparisons_keep_zero_and_infinite_lanes_compiled() -> Result<()> {
        // FCMGE V4.4S,V7.4S,V18.4S; FCMGT V5.2D,V6.2D,V8.2D; FACGE V9.4S,V7.4S,V18.4S.
        let program = [ADD_ONE, 0x6e32_e4e4, 0x6ee8_e4c5, 0x6e32_ece9, ADD_ONE, RET];
        let cases = [
            ([0x8000_0000_0000_0000, 0xff80_0000_7f80_0000], 0, false),
            ([0x3f80_0000_7f80_0000, 0x0000_0000_bf80_0000], 0, false),
            ([0x7fc0_0000_0000_0000, 0], INVALID, true),
            ([0x0000_0001_0000_0000, 0], 0, true),
        ];
        for (left, status, exceptional) in cases {
            let mut thread = input(false);
            thread.registers.fpsr = 0;
            thread.registers.v[7] = left;
            thread.registers.v[18] = [0x0000_0000_8000_0000, 0x7f80_0000_ff80_0000];
            thread.registers.v[6] = [0x7ff0_0000_0000_0000, 0x8000_0000_0000_0000];
            thread.registers.v[8] = [0x3ff0_0000_0000_0000, 0];
            let (native, reference, helpers) =
                against_helper(&program, TargetCpu::detect(), &thread)?;
            // FCMGE and FACGE both read the exceptional lane.
            assert_eq!(helpers, 2 * usize::from(exceptional), "{left:x?}");
            assert_eq!(native, reference, "{left:x?}");
            assert_eq!(native.fpsr, status, "{left:x?}");
        }
        Ok(())
    }

    /// Half FCMP orders zeros of either sign and infinities in compiled
    /// code, and FCVT widens a zero Half there; a NaN or a subnormal still
    /// takes the helper.
    #[test]
    fn half_compares_and_widening_keep_zeros_and_infinities_compiled() -> Result<()> {
        // FCMP H2,H0; CSET X5,GE; FCVT S4,H4; FCVT D6,H6.
        let program = [
            ADD_ONE,
            0x1ee0_2040,
            0x9a9f_b7e5,
            0x1ee2_4084,
            0x1ee2_c0c6,
            ADD_ONE,
            RET,
        ];
        let cases = [
            (0x8000, 0x0000, 0x8000, 0),
            (0x0000, 0x8000, 0x0000, 0),
            (0xfc00, 0x3c00, 0xc000, 0),
            (0x7c00, 0x7c00, 0xbc00, 0),
            (0x3c00, 0xfbff, 0x3555, 0),
            (0x7e00, 0x3c00, 0x0000, 1),
            (0x0001, 0x3c00, 0x0000, 1),
        ];
        for (left, right, widened, helpers) in cases {
            for fpcr in [0, FPCR_NEP] {
                let mut thread = input(false);
                thread.registers.fpcr = fpcr;
                thread.registers.fpsr = 0;
                thread.registers.v[2][0] = 0xdead_0000 | left;
                thread.registers.v[0][0] = 0xbeef_0000 | right;
                thread.registers.v[4][0] = 0x1234_0000 | widened;
                thread.registers.v[6][0] = 0x5678_0000 | widened;
                let (native, reference, count) =
                    against_helper(&program, TargetCpu::detect(), &thread)?;
                assert_eq!(count, helpers, "{left:#x} {right:#x} {widened:#x}");
                assert_eq!(native, reference, "{left:#x} {right:#x} {widened:#x}");
            }
        }
        Ok(())
    }

    /// FCVTN and FCVTN2 from S to H narrow zero lanes in compiled code with
    /// F16C, with the helper's results.
    #[test]
    fn half_narrowing_keeps_zero_lanes_compiled_with_f16c() -> Result<()> {
        let target = TargetCpu::detect();
        if !target.supports(TargetCpu::F16c) {
            return Ok(());
        }
        // FCVTN V3.4H,V4.4S; FCVTN2 V3.8H,V5.4S.
        let program = [ADD_ONE, 0x0e21_6883, 0x4e21_68a3, ADD_ONE, RET];
        for fpsr in [0, INEXACT] {
            let mut thread = input(false);
            thread.registers.fpsr = fpsr;
            thread.registers.v[4] = [0x8000_0000_3f80_0000, 0x3eaa_aaab_0000_0000];
            thread.registers.v[5] = [0, 0xc120_0000_8000_0000];
            let (native, reference, helpers) = against_helper(&program, target, &thread)?;
            assert_eq!(helpers, 0, "{fpsr:#x}");
            assert_eq!(native, reference, "{fpsr:#x}");
            assert_eq!(native.v[3], [0x3555_0000_8000_3c00, 0xc900_8000_0000_0000]);
        }
        Ok(())
    }
}

#[test]
fn simd_family_chain_uses_shared_block_lowering_and_cpu_bound_disk_cache() -> Result<()> {
    use stela_compiler::aarch64::simd::saturating::FPSR_QC;
    use stela_compiler::{TargetCpu, block::BlockProfile};
    let cache = tempfile::tempdir()?;
    let program = words(&[
        0x4f0b5420, // SHL V0.16B,V1.16B,#3
        0x6f09a402, // USHLL2 V2.8H,V0.16B,#1
        0x4f0f8c42, // RSHRN2 V2.16B,V2.8H,#1: alias, preserve low half
        0x5f402483, // SRSHR D3,D4,#64
        0x7f4024c5, // URSHR D5,D6,#64
        0x6f404507, // SRI V7.2D,V8.2D,#64: preserve both old lanes
        0x6eab6d49, // UMIN V9.4S,V10.4S,V11.4S
        0x6eb0a92c, // UMAXV S12,V9.4S
        0x4e20598d, // CNT V13.16B,V12.16B
        0x4eedbdae, // ADDP V14.2D,V13.2D,V13.2D
        0x4e604a0f, // CLS V15.8H,V16.8H
        0x5e207a30, // SQABS B16,B17: active MIN sets QC, clears upper bits
        0x6e320e52, // UQADD V18.16B,V18.16B,V18.16B: zero sum keeps QC
        0x6ea12a93, // SQXTUN2 V19.4S,V20.2D: signed wide bounds, preserve low half
        0x4eb7c2d5, // SMULL2 V21.2D,V22.4S,V23.4S
        0x4eb612b5, // SADDW2 V21.2D,V21.2D,V22.4S: full wide left alias
        0x6eb772d8, // UABDL2 V24.2D,V22.4S,V23.4S
        0x4eb79ed9, // MUL V25.4S,V22.4S,V23.4S
        0x6e3716da, // URHADD V26.16B,V22.16B,V23.16B
        0x6e202b5b, // UADDLP V27.8H,V26.16B
        0x4ea02adc, // SADDLP V28.2D,V22.4S
        0x6eb842bd, // RADDHN2 V29.4S,V21.2D,V24.2D
        0x0e212b7e, // XTN V30.8B,V27.8H
        0x6e213b5f, // SHLL2 V31.8H,V26.16B,#8
        0x1e2603d8, // FMOV W24,S30: raw low bits and W upper clearing
        0x9eaf0311, // FMOV V17.D[1],X24: preserve low half
        0x1ee04234, // FMOV H20,H17: raw scalar payload, clear upper bits
        0x1eee1017, // FMOV H23,#1.0
        0x6f00c656, // MVNI V22.4S,#0x12,MSL#8
        0x2f00b5f6, // BIC V22.4H,#0x0f,LSL#8: Q=0 clears high half
        RET,
    ]);
    let mut initial = state(BASE, 19);
    initial.registers.nzcv = 0xf0000000;
    initial.registers.fpcr = 0x03c00000;
    initial.registers.fpsr = 0x1f;
    let source: [u8; 16] = std::array::from_fn(|i| (i * 17 + 3) as u8);
    let limbs = |bytes: [u8; 16]| {
        let value = u128::from_le_bytes(bytes);
        [value as u64, (value >> 64) as u64]
    };
    initial.registers.v[1] = limbs(source);
    initial.registers.v[4] = [u64::MAX, 0x12345678];
    initial.registers.v[6] = [u64::MAX, 0x87654321];
    initial.registers.v[7] = [0x0123456789abcdef, 0xfedcba9876543210];
    initial.registers.v[10] = [0x00000001ffffffff, 0x0000000080000000];
    initial.registers.v[11] = [0x0000000200000000, 0xffffffff7fffffff];
    initial.registers.v[16] = [u64::MAX; 2];
    initial.registers.v[17] = [0x80, u64::MAX];
    initial.registers.v[19] = [0x0123456789abcdef, u64::MAX];
    initial.registers.v[20] = [u64::MAX, 0x100000000];
    initial.registers.v[22] = [0x00000003fffffffe, 0x80000000ffffffff];
    initial.registers.v[23] = [0xfffffffd00000005, 0xffffffff00000002];
    initial.registers.v[29] = [0x0123456789abcdef, u64::MAX];
    let mut expected = initial.registers.clone();
    let shifted = source.map(|byte| byte.wrapping_shl(3));
    expected.v[0] = limbs(shifted);
    let mut narrowed = [0; 16];
    for i in 0..4 {
        narrowed[i * 2..i * 2 + 2].copy_from_slice(&(u16::from(shifted[8 + i]) * 2).to_le_bytes());
    }
    narrowed[8..].copy_from_slice(&shifted[8..]);
    expected.v[2] = limbs(narrowed);
    expected.v[3] = [0; 2];
    expected.v[5] = [1, 0];
    expected.v[9] = [0x0000000100000000, 0x000000007fffffff];
    expected.v[12] = [0x7fffffff, 0];
    expected.v[13] = [0x07080808, 0];
    expected.v[14] = [0x07080808; 2];
    expected.v[15] = [0x000f000f000f000f; 2];
    expected.v[16] = [0x7f, 0];
    expected.v[19] = [0x0123456789abcdef, 0xffffffff00000000];
    expected.fpsr |= FPSR_QC;
    expected.v[21] = [u64::MAX - 2, 0];
    expected.v[24] = [0xfffffffd, 0x7fffffff];
    expected.v[25] = [0xfffffff7fffffff6, 0x80000000fffffffe];
    let input_bytes = |v: [u64; 2]| (u128::from(v[0]) | (u128::from(v[1]) << 64)).to_le_bytes();
    let a = input_bytes(initial.registers.v[22]);
    let b = input_bytes(initial.registers.v[23]);
    let averaged: [u8; 16] =
        std::array::from_fn(|i| ((u16::from(a[i]) + u16::from(b[i]) + 1) >> 1) as u8);
    let pairs: [u16; 8] =
        std::array::from_fn(|i| u16::from(averaged[i * 2]) + u16::from(averaged[i * 2 + 1]));
    expected.v[26] = limbs(averaged);
    expected.v[27] = limbs(std::array::from_fn(|i| pairs[i / 2].to_le_bytes()[i % 2]));
    expected.v[28] = [1, 0xffffffff7fffffff];
    expected.v[29] = [0x0123456789abcdef, 1];
    expected.v[30] = limbs(std::array::from_fn(
        |i| if i < 8 { pairs[i] as u8 } else { 0 },
    ));
    expected.v[31] = limbs(std::array::from_fn(|i| {
        if i % 2 == 0 { 0 } else { averaged[8 + i / 2] }
    }));
    expected.x[24] = 2;
    expected.v[17] = [0x80, 2];
    expected.v[20] = [0x80, 0];
    expected.v[23] = [0x3c00, 0];
    expected.v[22] = [0xf0ffe000f0ffe000, 0];
    let mut targets = vec![TargetCpu::Baseline, TargetCpu::detect()];
    targets.dedup();
    for target in targets {
        for (phase, disk) in [false, true, true].into_iter().enumerate() {
            let mut runtime = BlockRuntime::new_for_target(
                802,
                cache.path().into(),
                BlockProfile::ORDINARY,
                target,
            );
            runtime.register_code(GuestAddress(BASE), program.clone())?;
            let mut thread = initial.clone();
            if phase == 2 {
                thread.registers.fpsr |= FPSR_QC;
            }
            // SAFETY: owned, terminating register-only code with no guest RAM
            // access.
            let result = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 1)? };
            assert_eq!(result.reason, StopReason::StopAddress);
            assert_eq!(thread.pc, GuestAddress(STOP));
            assert_eq!(thread.registers, expected);
            assert_eq!(result.compiled_blocks, usize::from(!disk));
            assert_eq!(result.disk_cache_hits, usize::from(disk));
        }
    }
    Ok(())
}

#[test]
fn simd_shifts_and_doubled_products_share_code() -> Result<()> {
    use stela_compiler::aarch64::simd::saturating::FPSR_QC;
    use stela_compiler::{TargetCpu, block::BlockProfile};
    let cache = tempfile::tempdir()?;
    let program = words(&[
        0x4e224420, // SSHL V0.16B,V1.16B,V2.16B: signed low-byte counts
        0x7ee55483, // URSHL D3,D4,D5: rounded unsigned full-width right shift
        0x5e284ce6, // SQSHL B6,B7,B8: saturating scalar MIN, clear upper bits
        0x6e6b5d49, // UQRSHL V9.8H,V10.8H,V11.8H: ignore upper count bits
        0x7f4065ac, // SQSHLU D12,D13,#0: negative source saturates even at zero
        0x6eb0b5ee, // SQRDMULH V14.4S,V15.4S,V16.4S: MIN*MIN and signed ties
        0x4eb39251, // SQDMLAL2 V17.2D,V18.4S,V19.4S: saturate product first
        0x5f5f72b4, // SQDMLSL S20,H21,V15.H[1]: selected MIN*MIN, upper clearing
        0x6e399f17, // PMUL V23.16B,V24.16B,V25.16B: low polynomial products
        0x4efce37a, // PMULL2 V26.1Q,V27.2D,V28.2D: full high-half polynomial
        0x6e3f4fdd, // UQSHL V29.16B,V30.16B,V31.16B: zero keeps earlier QC
        RET,
    ]);
    let limbs = |bytes: [u8; 16]| {
        let value = u128::from_le_bytes(bytes);
        [value as u64, (value >> 64) as u64]
    };
    let mut initial = state(BASE, 19);
    initial.registers.nzcv = 0xa0000000;
    initial.registers.fpcr = 0x03c00000;
    initial.registers.fpsr = 0x1f;
    initial.registers.v[1] = limbs([
        0, 1, 0x80, 0xff, 0x7f, 0x80, 3, 2, 0x55, 0xaa, 0x80, 0x7f, 0x42, 0xff, 1, 0x80,
    ]);
    initial.registers.v[2] = limbs([
        0, 1, 0xff, 0xf8, 8, 0x80, 0x7f, 0xfe, 0xfe, 0xff, 0xf9, 0xf9, 2, 0, 0xf8, 0xf7,
    ]);
    initial.registers.v[4] = [u64::MAX; 2];
    initial.registers.v[5] = [0x123456789abcdec0, u64::MAX]; // low byte -64
    initial.registers.v[7] = [0x80, u64::MAX];
    initial.registers.v[8] = [1, u64::MAX];
    initial.registers.v[10] = [u64::MAX; 2];
    initial.registers.v[11] = [0x55f055f055f055f0; 2]; // low byte -16 in each H
    initial.registers.v[13] = [u64::MAX; 2];
    initial.registers.v[15] = [0x8000000080000000, 0x0000000100000001];
    initial.registers.v[16] = [0x7fffffff80000000, 0xc000000040000000];
    initial.registers.v[17] = [u64::MAX, 1 << 63];
    initial.registers.v[18] = [0, 0x7fffffff80000000];
    initial.registers.v[19] = [0, 0x7fffffff80000000];
    initial.registers.v[20] = [0, u64::MAX];
    initial.registers.v[21] = [0x8000, u64::MAX];
    initial.registers.v[24] = [0x8383838383838383; 2];
    initial.registers.v[25] = [0x7d7d7d7d7d7d7d7d; 2];
    initial.registers.v[27] = [u64::MAX, 0x8000000000000001];
    initial.registers.v[28] = [u64::MAX, 3];
    initial.registers.v[31] = [0x7f7f7f7f7f7f7f7f; 2];
    let mut expected = initial.registers.clone();
    expected.v[0] = [0x0000ff00ffc00200, 0xff00ff0800ffd515];
    expected.v[3] = [1, 0];
    expected.v[6] = [0x80, 0];
    expected.v[9] = [0x0001000100010001; 2];
    expected.v[12] = [0, 0];
    expected.v[14] = [0x800000017fffffff, 1];
    // First: sat(2*MIN32*MIN32) + (-1); second: 2*MAX32^2 + MIN64.
    expected.v[17] = [0x7ffffffffffffffe, (-8589934590i64) as u64];
    expected.v[20] = [0x80000001, 0];
    expected.v[23] = [0x0707070707070707; 2];
    expected.v[26] = [0x8000000000000003, 1];
    expected.v[29] = [0, 0];
    expected.fpsr |= FPSR_QC;
    let mut targets = vec![TargetCpu::Baseline, TargetCpu::detect()];
    if TargetCpu::Pclmulqdq.supported() {
        targets.push(TargetCpu::Pclmulqdq);
    }
    targets.dedup();
    for target in targets {
        for (phase, disk) in [false, true, true].into_iter().enumerate() {
            let mut runtime = BlockRuntime::new_for_target(
                803,
                cache.path().into(),
                BlockProfile::ORDINARY,
                target,
            );
            runtime.register_code(GuestAddress(BASE), program.clone())?;
            let mut thread = initial.clone();
            if phase == 2 {
                thread.registers.fpsr |= FPSR_QC;
            }
            // SAFETY: owned, terminating register-only code returning to the
            // stop address.
            let result = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 1)? };
            assert_eq!(result.reason, StopReason::StopAddress);
            assert_eq!(thread.pc, GuestAddress(STOP));
            assert_eq!(thread.registers, expected);
            assert_eq!(result.compiled_blocks, usize::from(!disk));
            assert_eq!(result.disk_cache_hits, usize::from(disk));
        }
    }
    Ok(())
}

#[test]
fn nested_direct_and_indirect_calls_preserve_architectural_link_registers() -> Result<()> {
    let cache = tempfile::tempdir()?;
    let mut runtime = BlockRuntime::new(17, cache.path().to_path_buf());
    let mut main = vec![NOP; 13];
    main[..4].copy_from_slice(&[
        mov(19, 30),
        transfer(BASE + 4, BASE + 0x20, true),
        mov(30, 19),
        RET,
    ]);
    main[8..].copy_from_slice(&[
        mov(20, 30),
        0xd63f0120, // BLR X9
        0x91000800, // ADD X0, X0, #2
        mov(30, 20),
        RET,
    ]);
    runtime.register_code(GuestAddress(BASE), words(&main))?;
    runtime.register_code(GuestAddress(0x2000), words(&[0x91000c00, RET]))?;
    let mut thread = state(BASE, 7);
    thread.registers.x[9] = 0x2000;
    // SAFETY: owned, terminating code with no memory access; execution is
    // serialized.
    let result = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 10)? };
    assert_eq!(result.reason, StopReason::StopAddress);
    assert_eq!(result.blocks_executed, 5);
    assert_eq!(result.compiled_blocks, 5);
    assert_eq!(thread.pc, GuestAddress(STOP));
    assert_eq!(thread.registers.x[0], 12);
    assert_eq!(thread.registers.x[19], STOP);
    assert_eq!(thread.registers.x[20], BASE + 8);
    assert_eq!(thread.registers.x[30], STOP);
    Ok(())
}

#[test]
fn finite_budget_suspends_loop_at_real_guest_pc_and_resumes_from_cached_block() -> Result<()> {
    let cache = tempfile::tempdir()?;
    let mut runtime = BlockRuntime::new(18, cache.path().to_path_buf());
    runtime.register_code(GuestAddress(BASE), words(&[0xf1000400, 0x54ffffe1, RET]))?;
    let mut thread = state(BASE, 3);
    // SAFETY: a decrement loop without memory access, bounded by the dispatch
    // budget.
    let first = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 2)? };
    assert_eq!(first.reason, StopReason::BudgetExhausted);
    assert_eq!(first.blocks_executed, 2);
    assert_eq!(first.compiled_blocks, 1);
    // Both iterations run in the first loaded block; the internal edge makes no
    // second cache lookup.
    assert_eq!(first.memory_cache_hits, 0);
    assert_eq!(thread.pc, GuestAddress(BASE));
    assert_eq!(thread.registers.x[0], 1);
    // SAFETY: the same code now returns within two blocks.
    let resumed = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 2)? };
    assert_eq!(resumed.reason, StopReason::StopAddress);
    assert_eq!(resumed.blocks_executed, 2);
    assert_eq!(resumed.memory_cache_hits, 1);
    assert_eq!(thread.registers.x[0], 0);
    assert_eq!(thread.pc, GuestAddress(STOP));
    Ok(())
}

#[test]
fn straight_line_window_limit_falls_through_without_losing_an_instruction() -> Result<()> {
    let cache = tempfile::tempdir()?;
    let mut runtime = BlockRuntime::new(19, cache.path().to_path_buf());
    let mut program = vec![ADD_ONE; 257];
    program.push(RET);
    runtime.register_code(GuestAddress(BASE), words(&program))?;
    let mut thread = state(BASE, 0);
    // SAFETY: terminating additions without memory access, returning to the
    // stop address.
    let first = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 1)? };
    assert_eq!(first.reason, StopReason::BudgetExhausted);
    assert_eq!(thread.pc, GuestAddress(BASE + 256 * 4));
    assert_eq!(thread.registers.x[0], 256);
    // SAFETY: only one addition and RET remain.
    let second = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 1)? };
    assert_eq!(second.reason, StopReason::StopAddress);
    assert_eq!(second.blocks_executed, 1);
    assert_eq!(thread.registers.x[0], 257);
    assert_eq!(thread.pc, GuestAddress(STOP));
    Ok(())
}

#[test]
fn unsupported_callee_is_lazy_and_preserves_completed_caller_effects() -> Result<()> {
    let cache = tempfile::tempdir()?;
    let mut runtime = BlockRuntime::new(20, cache.path().to_path_buf());
    runtime.register_code(
        GuestAddress(BASE),
        words(&[ADD_ONE, transfer(BASE + 4, 0x2000, true)]),
    )?;
    runtime.register_code(GuestAddress(0x2000), words(&[0]))?;
    let mut thread = state(BASE, 40);
    // SAFETY: the supported caller is memory-free; unknown callee code must not run.
    let result = unsafe { runtime.run(&mut thread, None, 10)? };
    assert!(matches!(result.reason, StopReason::Unsupported { .. }));
    assert_eq!(result.blocks_executed, 1);
    assert_eq!(result.compiled_blocks, 1);
    assert_eq!(thread.pc, GuestAddress(0x2000));
    assert_eq!(thread.registers.x[0], 41);
    assert_eq!(thread.registers.x[30], BASE + 8);
    Ok(())
}

#[test]
fn hosted_dispatch_separates_source_admission_from_plan_and_memory_errors() -> Result<()> {
    use stela_compiler::{
        CompileErrorKind, TargetCpu,
        block::{BlockProfile, CheckedMemoryProfile, Features},
    };
    let cache = tempfile::tempdir()?;
    let mut runtime = BlockRuntime::new(20, cache.path().to_path_buf());
    let unsupported = words(&[0]);
    runtime.register_code(GuestAddress(BASE), unsupported.clone())?;
    let detail = stela_compiler::block::source_bytes(&unsupported, GuestAddress(BASE))
        .unwrap_err()
        .to_string();
    let mut thread = state(BASE, 40);
    // SAFETY: dispatch refuses the unsupported source before it can run.
    let stopped = unsafe { runtime.run(&mut thread, None, 1)? };
    assert_eq!(stopped.reason, StopReason::Unsupported { detail });
    assert_eq!(stopped.compiled_blocks, 0);
    assert_eq!(stopped.blocks_executed, 0);

    let mut leaf = BlockRuntime::new_for_target(
        20,
        cache.path().to_path_buf(),
        BlockProfile::new(Features::LEAF),
        TargetCpu::Baseline,
    );
    leaf.register_code(GuestAddress(BASE), words(&[ADD_ONE]))?;
    // SAFETY: only a register operation, and the leaf planner refuses its missing return.
    let error = unsafe { leaf.run(&mut thread, None, 1) }.unwrap_err();
    assert!(matches!(
        error,
        stela_engine::Error::Compile(error) if error.kind == CompileErrorKind::MissingLeafReturn
    ));

    let mut checked = BlockRuntime::new_for_target(
        20,
        cache.path().to_path_buf(),
        BlockProfile::ORDINARY.with_checked_memory(CheckedMemoryProfile::SpAlignmentEnabled),
        TargetCpu::Baseline,
    );
    checked.register_code(GuestAddress(BASE), unsupported)?;
    // SAFETY: checked-memory artifacts are refused at the direct entry.
    let error = unsafe { checked.run(&mut thread, None, 1) }.unwrap_err();
    assert_eq!(
        error.to_string(),
        "checked artifact cannot execute through direct entry"
    );
    assert_eq!(thread.pc, GuestAddress(BASE));
    assert_eq!(thread.registers.x[0], 40);
    assert!(named_files(cache.path(), "native.bin")?.is_empty());
    Ok(())
}

#[test]
fn indirect_unmapped_and_unaligned_targets_stop_without_host_jumps() -> Result<()> {
    for (target, expected) in [
        (0x9000, StopReason::UnmappedTarget),
        (BASE + 1, StopReason::UnalignedTarget),
    ] {
        let cache = tempfile::tempdir()?;
        let mut runtime = BlockRuntime::new(21, cache.path().to_path_buf());
        runtime.register_code(GuestAddress(BASE), words(&[0xd61f0120]))?; // BR X9
        let mut thread = state(BASE, 55);
        thread.registers.x[9] = target;
        // SAFETY: BR returns a guest PC to dispatch; it must never jump there on the host.
        let result = unsafe { runtime.run(&mut thread, None, 10)? };
        assert_eq!(result.reason, expected);
        assert_eq!(result.blocks_executed, 1);
        assert_eq!(thread.pc, GuestAddress(target));
        assert_eq!(thread.registers.x[0], 55);
    }
    Ok(())
}

#[test]
fn registration_replacement_removal_and_reregistration_invalidate_loaded_blocks() -> Result<()> {
    let cache = tempfile::tempdir()?;
    let mut runtime = BlockRuntime::new(22, cache.path().to_path_buf());
    let original = words(&[0xd2800020, RET]); // MOV X0, #1
    let first_generation = runtime.register_code(GuestAddress(BASE), original.clone())?;
    let mut thread = state(BASE, 0);
    // SAFETY: own memory-free constant-return code.
    unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 2)? };
    assert_eq!(thread.registers.x[0], 1);
    let second_generation = runtime.register_code(GuestAddress(BASE), words(&[0xd2800040, RET]))?;
    assert!(second_generation > first_generation);
    thread = state(BASE, 0);
    // SAFETY: the replacement is another owned constant-return program.
    let replaced = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 2)? };
    assert_eq!(replaced.reason, StopReason::StopAddress);
    assert_eq!(replaced.compiled_blocks, 1);
    assert_eq!(replaced.memory_cache_hits, 0);
    assert_eq!(thread.registers.x[0], 2);
    assert!(runtime.unregister_code(GuestAddress(BASE)));
    assert!(!runtime.unregister_code(GuestAddress(BASE)));
    thread = state(BASE, 99);
    // SAFETY: no code may execute after its region was removed.
    let removed = unsafe { runtime.run(&mut thread, None, 1)? };
    assert_eq!(removed.reason, StopReason::UnmappedTarget);
    assert_eq!(removed.blocks_executed, 0);
    assert_eq!(thread.registers.x[0], 99);
    assert!(runtime.register_code(GuestAddress(BASE), original)? > second_generation);
    // SAFETY: the original code has been registered again.
    let restored = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 2)? };
    assert_eq!(restored.reason, StopReason::StopAddress);
    assert_eq!(restored.memory_cache_hits, 0);
    assert_eq!(restored.disk_cache_hits, 1);
    assert_eq!(thread.registers.x[0], 1);
    Ok(())
}

#[test]
fn warm_execution_uses_rx_memory_and_new_runtime_revalidates_disk_cache() -> Result<()> {
    let cache = tempfile::tempdir()?;
    let program = words(&[ADD_ONE, RET]);
    let mut runtime = BlockRuntime::new(23, cache.path().to_path_buf());
    runtime.register_code(GuestAddress(BASE), program.clone())?;
    let mut thread = state(BASE, 1);
    // SAFETY: own finite memory-free code, serialized through this runtime.
    let cold = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 2)? };
    assert_eq!(cold.compiled_blocks, 1);
    thread = state(BASE, 7);
    // SAFETY: the same immutable executable copy remains valid.
    let warm = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 2)? };
    assert_eq!(warm.compiled_blocks, 0);
    assert_eq!(warm.disk_cache_hits, 0);
    assert_eq!(warm.memory_cache_hits, 1);
    assert_eq!(thread.registers.x[0], 8);
    let mut other = BlockRuntime::new(23, cache.path().to_path_buf());
    other.register_code(GuestAddress(BASE), program)?;
    thread = state(BASE, 12);
    // SAFETY: loading must revalidate the persisted native bytes before executing.
    let disk = unsafe { other.run(&mut thread, Some(GuestAddress(STOP)), 2)? };
    assert_eq!(disk.compiled_blocks, 0);
    assert_eq!(disk.disk_cache_hits, 1);
    assert_eq!(disk.memory_cache_hits, 0);
    assert_eq!(thread.registers.x[0], 13);
    Ok(())
}

fn named_files(root: &Path, name: &str) -> Result<Vec<PathBuf>> {
    let mut result = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            result.extend(named_files(&path, name)?);
        } else if path.file_name().and_then(|s| s.to_str()) == Some(name) {
            result.push(path);
        }
    }
    Ok(result)
}

#[test]
fn disk_corruption_cannot_replace_a_loaded_copy_and_is_rejected_on_reload() -> Result<()> {
    let cache = tempfile::tempdir()?;
    let program = words(&[ADD_ONE, RET]);
    let mut runtime = BlockRuntime::new(24, cache.path().to_path_buf());
    runtime.register_code(GuestAddress(BASE), program.clone())?;
    let mut thread = state(BASE, 0);
    // SAFETY: own memory-free code, first load before corrupting its disk artifact.
    unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 2)? };
    let files = named_files(cache.path(), "native.bin")?;
    assert_eq!(files.len(), 1);
    std::fs::write(&files[0], [0xcc])?;
    thread = state(BASE, 10);
    // SAFETY: loaded RX memory is independent of later untrusted disk edits.
    let warm = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 2)? };
    assert_eq!(warm.memory_cache_hits, 1);
    assert_eq!(thread.registers.x[0], 11);
    let mut other = BlockRuntime::new(24, cache.path().to_path_buf());
    other.register_code(GuestAddress(BASE), program)?;
    thread = state(BASE, 20);
    // SAFETY: corruption must be rejected before any new mapping or execution.
    assert!(unsafe { other.run(&mut thread, Some(GuestAddress(STOP)), 2) }.is_err());
    assert_eq!(thread.pc, GuestAddress(BASE));
    assert_eq!(thread.registers.x[0], 20);
    Ok(())
}

#[test]
fn rejects_overlapping_regions_and_never_pads_partial_final_instruction() -> Result<()> {
    let cache = tempfile::tempdir()?;
    let mut runtime = BlockRuntime::new(25, cache.path().to_path_buf());
    let mut program = words(&[ADD_ONE]);
    program.extend([0xc0, 0x03, 0x5f]); // Three bytes of RET, incomplete.
    runtime.register_code(GuestAddress(BASE), program)?;
    assert!(
        runtime
            .register_code(GuestAddress(BASE + 4), words(&[RET]))
            .is_err()
    );
    assert!(
        runtime
            .register_code(GuestAddress(BASE - 4), words(&[NOP, RET]))
            .is_err()
    );
    let mut thread = state(BASE, 40);
    // SAFETY: the complete addition is memory-free; truncated RET must never execute.
    let result = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 10)? };
    assert_eq!(result.reason, StopReason::TruncatedInstruction);
    assert_eq!(result.blocks_executed, 1);
    assert_eq!(thread.pc, GuestAddress(BASE + 4));
    assert_eq!(thread.registers.x[0], 41);
    assert_eq!(thread.registers.x[30], STOP);
    Ok(())
}

#[test]
fn direct_ram_effects_survive_call_and_return_dispatch_boundaries() -> Result<()> {
    let cache = tempfile::tempdir()?;
    let mut runtime = BlockRuntime::new(26, cache.path().to_path_buf());
    runtime.register_code(
        GuestAddress(BASE),
        words(&[
            mov(19, 30),
            ADD_ONE,
            0xf9000020, // STR X0, [X1]
            transfer(BASE + 12, 0x2000, true),
            mov(30, 19),
            RET,
        ]),
    )?;
    runtime.register_code(
        GuestAddress(0x2000),
        words(&[
            0xf9400020, // LDR X0, [X1]
            ADD_ONE, 0xf9000020, // STR X0, [X1]
            RET,
        ]),
    )?;
    let mut cell = Box::new(0u64);
    let mut thread = state(BASE, 40);
    thread.registers.x[1] = (&raw mut *cell) as u64;
    // SAFETY: X1 points to one exclusively owned aligned u64 for the whole run;
    // the code accesses only that cell, and no Rust reference to it is live.
    let result = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 4)? };
    assert_eq!(result.reason, StopReason::StopAddress);
    assert_eq!(result.blocks_executed, 3);
    assert_eq!(thread.registers.x[0], 42);
    assert_eq!(*cell, 42);
    assert_eq!(thread.pc, GuestAddress(STOP));
    Ok(())
}

#[test]
fn discovery_follows_control_flow_edges() -> Result<()> {
    use stela_compiler::block::{DynamicEdgeKind, SuccessorKind};
    let cache = tempfile::tempdir()?;
    let mut runtime = BlockRuntime::new(91, cache.path().into());
    runtime.register_code(
        GuestAddress(BASE),
        words(&[
            transfer(BASE, 0x2000, true),
            transfer(BASE + 4, 0x3000, false),
            0,
        ]),
    )?;
    runtime.register_code(
        GuestAddress(0x2000),
        words(&[0x54000040, RET, transfer(0x2008, 0x2000, false), 0]),
    )?;
    runtime.register_code(GuestAddress(0x3000), words(&[RET, 0]))?;
    let report = runtime.discover(&[GuestAddress(BASE)], 32)?;
    assert_eq!(
        report.blocks.iter().map(|b| b.entry.0).collect::<Vec<_>>(),
        [BASE, 0x2000, BASE + 4, 0x2008, 0x2004, 0x3000]
    );
    assert_eq!(report.compiled_blocks, 6);
    assert_eq!(report.attempted_entries, 6);
    assert!(report.issues.is_empty());
    assert!(report.pending_entries.is_empty());
    assert!(report.static_worklist_exhausted);
    assert_eq!(
        report.blocks[0]
            .analysis
            .static_successors
            .iter()
            .map(|e| e.kind)
            .collect::<Vec<_>>(),
        [SuccessorKind::CallTarget, SuccessorKind::CallContinuation]
    );
    assert_eq!(report.deferred_dynamic_edges.len(), 2);
    assert!(
        report
            .deferred_dynamic_edges
            .iter()
            .all(|e| e.edge.kind == DynamicEdgeKind::Return)
    );
    assert!(report.dynamic_targets_unresolved);
    assert!(
        !report.native_executed && !report.native_pages_loaded && !report.complete_code_discovered
    );
    for block in &report.blocks {
        assert!(runtime.loaded_manifest(block.entry).is_none());
    }
    Ok(())
}

#[test]
fn discovery_prepares_faulting_guest_memory_code_but_does_not_map_or_execute_it() -> Result<()> {
    let cache = tempfile::tempdir()?;
    let mut runtime = BlockRuntime::new(92, cache.path().into());
    // LDR X0,[X1]; RET would dereference an unspecified guest pointer if invoked.
    runtime.register_code(GuestAddress(BASE), words(&[0xf9400020, RET]))?;
    let prepared = runtime.prepare(GuestAddress(BASE))?;
    assert_eq!(prepared.analysis.source_bytes, 8);
    assert!(!prepared.prepared.cache_hit);
    assert!(runtime.loaded_manifest(GuestAddress(BASE)).is_none());
    let report = runtime.discover(&[GuestAddress(BASE)], 1)?;
    assert_eq!(report.disk_cache_hits, 1);
    assert!(!report.native_executed && !report.native_pages_loaded);
    assert!(runtime.loaded_manifest(GuestAddress(BASE)).is_none());
    Ok(())
}

#[test]
fn discovery_distinguishes_indirect_calls_returns_and_traps_without_inventing_targets() -> Result<()>
{
    use stela_compiler::block::DynamicEdgeKind;
    let cache = tempfile::tempdir()?;
    let mut runtime = BlockRuntime::new(93, cache.path().into());
    runtime.register_code(GuestAddress(BASE), words(&[0xd63f0120, RET, 0]))?; // BLR X9; RET; unreachable data.
    runtime.register_code(GuestAddress(0x2000), words(&[0xd61f0140, 0]))?; // BR X10.
    runtime.register_code(GuestAddress(0x3000), words(&[0xd4001001, 0]))?;
    let report = runtime.discover(
        &[
            GuestAddress(0x3000),
            GuestAddress(0x2000),
            GuestAddress(BASE),
        ],
        20,
    )?;
    assert_eq!(
        report.roots.iter().map(|p| p.0).collect::<Vec<_>>(),
        [BASE, 0x2000, 0x3000]
    );
    assert_eq!(report.blocks.len(), 4);
    assert!(report.issues.is_empty());
    assert_eq!(
        report
            .deferred_dynamic_edges
            .iter()
            .map(|e| (e.edge.kind, e.edge.register))
            .collect::<Vec<_>>(),
        [
            (DynamicEdgeKind::Call, 9),
            (DynamicEdgeKind::Branch, 10),
            (DynamicEdgeKind::Return, 30)
        ]
    );
    let trap = report.blocks.iter().find(|b| b.entry.0 == 0x3000).unwrap();
    assert!(trap.analysis.trap.is_some());
    assert!(trap.analysis.static_successors.is_empty());
    Ok(())
}

#[test]
fn discovery_reports_exact_unsupported_pc_missing_and_truncated_sources_with_bounded_work()
-> Result<()> {
    use stela_engine::blocks::DiscoveryProblem;
    let cache = tempfile::tempdir()?;
    let mut runtime = BlockRuntime::new(94, cache.path().into());
    runtime.register_code(GuestAddress(BASE), words(&[NOP, 0, RET]))?;
    let mut partial = words(&[NOP]);
    partial.extend([1, 2]);
    runtime.register_code(GuestAddress(0x2000), partial)?;
    let report = runtime.discover(
        &[
            GuestAddress(BASE),
            GuestAddress(0x2004),
            GuestAddress(0x9000),
        ],
        10,
    )?;
    assert!(report.blocks.is_empty());
    assert_eq!(report.issues.len(), 3);
    assert!(
        matches!(&report.issues[0].problem, DiscoveryProblem::Unsupported {
            failure_pc,
            encoding: Some(0),
            cause: stela_compiler::CompileErrorKind::Unsupported(_),
        } if failure_pc.0 == BASE + 4)
    );
    assert_eq!(
        report.issues[1].problem,
        DiscoveryProblem::TruncatedInstruction
    );
    assert_eq!(report.issues[2].problem, DiscoveryProblem::UnmappedTarget);
    assert!(!report.static_worklist_exhausted);
    assert_eq!(report.attempted_entries, 3);
    assert!(runtime.loaded_manifest(GuestAddress(BASE)).is_none());
    assert!(runtime.discover(&[], 1).is_err());
    assert!(
        runtime
            .discover(&[GuestAddress(BASE), GuestAddress(BASE)], 1)
            .is_err()
    );
    assert!(runtime.discover(&[GuestAddress(BASE + 1)], 1).is_err());
    assert!(runtime.discover(&[GuestAddress(BASE)], 4097).is_err());
    let zero = runtime.discover(&[GuestAddress(BASE)], 0)?;
    assert_eq!(zero.attempted_entries, 0);
    assert_eq!(zero.pending_entries, [GuestAddress(BASE)]);
    Ok(())
}

#[test]
fn prepared_aot_artifact_is_reused_by_jit_and_source_replacement_changes_identity() -> Result<()> {
    let cache = tempfile::tempdir()?;
    let mut runtime = BlockRuntime::new(95, cache.path().into());
    let original = words(&[ADD_ONE, RET]);
    runtime.register_code(GuestAddress(BASE), original.clone())?;
    let prepared = runtime.prepare(GuestAddress(BASE))?;
    let key = prepared.prepared.key;
    let generation = prepared.generation;
    assert!(!prepared.prepared.cache_hit);
    assert!(runtime.loaded_manifest(GuestAddress(BASE)).is_none());
    let mut thread = state(BASE, 9);
    // SAFETY: owned code without memory access, returning to the stop address.
    let ran = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), 2)? };
    assert_eq!(ran.compiled_blocks, 0);
    assert_eq!(ran.disk_cache_hits, 1);
    assert_eq!(thread.registers.x[0], 10);
    assert_eq!(
        runtime
            .loaded_manifest(GuestAddress(BASE))
            .unwrap()
            .identity,
        prepared.prepared.manifest.identity
    );
    runtime.register_code(GuestAddress(BASE), words(&[0x91000800, RET]))?;
    assert!(runtime.loaded_manifest(GuestAddress(BASE)).is_none());
    let replaced = runtime.prepare(GuestAddress(BASE))?;
    assert!(replaced.generation > generation);
    assert_ne!(replaced.prepared.key, key);
    assert!(!replaced.prepared.cache_hit);
    assert!(runtime.unregister_code(GuestAddress(BASE)));
    assert!(runtime.prepare(GuestAddress(BASE)).is_err());
    runtime.register_code(GuestAddress(BASE), original)?;
    let restored = runtime.prepare(GuestAddress(BASE))?;
    assert!(restored.prepared.cache_hit);
    assert_eq!(restored.prepared.key, key);
    Ok(())
}

#[test]
fn discovery_budget_leaves_deterministic_candidates_and_new_runtime_uses_same_disk_artifacts()
-> Result<()> {
    let cache = tempfile::tempdir()?;
    let code = words(&[0x54000040, RET, RET]);
    let mut runtime = BlockRuntime::new(96, cache.path().into());
    runtime.register_code(GuestAddress(BASE), code.clone())?;
    let first = runtime.discover(&[GuestAddress(BASE)], 1)?;
    assert_eq!(
        first.pending_entries,
        [GuestAddress(BASE + 8), GuestAddress(BASE + 4)]
    );
    assert_eq!(first.compiled_blocks, 1);
    assert!(!first.static_worklist_exhausted);
    let mut second = BlockRuntime::new(96, cache.path().into());
    second.register_code(GuestAddress(BASE), code)?;
    let warm = second.discover(&[GuestAddress(BASE)], 10)?;
    assert_eq!(
        warm.blocks.iter().map(|b| b.entry.0).collect::<Vec<_>>(),
        [BASE, BASE + 8, BASE + 4]
    );
    assert_eq!(warm.disk_cache_hits, 1);
    assert_eq!(warm.compiled_blocks, 2);
    assert!(warm.static_worklist_exhausted);
    assert_eq!(
        first.blocks[0].prepared.manifest.identity,
        warm.blocks[0].prepared.manifest.identity
    );
    assert_eq!(
        first.blocks[0].prepared.manifest.native_sha256,
        warm.blocks[0].prepared.manifest.native_sha256
    );
    Ok(())
}

/// FEAT_LSE2 under concurrency: one host thread STPs `(n, !n)` while another
/// LDPs the same 16-byte-aligned quadwords, and never sees halves of two
/// different stores. The same loops 8 bytes further, two element accesses
/// each, see hundreds of torn pairs. Ignored: it races two host cores.
#[test]
#[ignore = "manual: opt-in"]
fn aligned_x_pairs_are_single_copy_atomic_across_threads() -> Result<()> {
    use stela_compiler::{TargetCpu, block::BlockProfile};
    use stela_engine::memory::{MemoryRegionSpec, OwnedMemory};
    const DATA: u64 = 0x3d00_0000_0000;
    const ITERATIONS: u64 = 10_000_000;
    // add x1,x1,#1; mvn x2,x1; stp x1,x2,[x0]; subs x3,x3,#1; b.ne 0; ret
    const WRITER: [u32; 6] = [
        0x9100_0421,
        0xaa21_03e2,
        0xa900_0801,
        0xf100_0463,
        0x54ff_ff81,
        RET,
    ];
    // ldp x1,x2,[x0]; eor x4,x1,x2; cmn x4,#1; cinc x5,x5,ne; subs x3,x3,#1;
    // b.ne 0; ret
    const READER: [u32; 7] = [
        0xa940_0801,
        0xca02_0024,
        0xb100_049f,
        0x9a85_04a5,
        0xf100_0463,
        0x54ff_ff61,
        RET,
    ];
    let target = TargetCpu::detect();
    if !target.supports(TargetCpu::AtomicLoad16) {
        eprintln!("skipped: this host gives no atomic aligned 16-byte accesses");
        return Ok(());
    }
    let mut memory = OwnedMemory::map(vec![MemoryRegionSpec {
        guest_address: DATA,
        virtual_bytes: 4096,
        initial_bytes: vec![0; 4096],
        readable: true,
        writable: true,
        executable: false,
    }])?;
    memory.write(DATA, &[[0u8; 8], [0xff; 8]].concat())?;
    let run = |program: &'static [u32], seed: u64| -> Result<GuestState> {
        let cache = tempfile::tempdir()?;
        let mut runtime =
            BlockRuntime::new_for_target(seed, cache.path().into(), BlockProfile::ORDINARY, target);
        runtime.register_code(GuestAddress(BASE), words(program))?;
        let mut thread = state(BASE, DATA);
        thread.registers.x[3] = ITERATIONS;
        // SAFETY: the program only accesses the mapped 16 bytes at DATA,
        // which the other thread races on deliberately at the guest level.
        let outcome = unsafe { runtime.run(&mut thread, Some(GuestAddress(STOP)), usize::MAX)? };
        assert_eq!(outcome.reason, StopReason::StopAddress);
        Ok(thread.registers)
    };
    let (writer, reader) = std::thread::scope(|scope| {
        let writer = scope.spawn(|| run(&WRITER, 1));
        let reader = scope.spawn(|| run(&READER, 2));
        (writer.join().unwrap(), reader.join().unwrap())
    });
    assert_eq!(writer?.x[1], ITERATIONS);
    assert_eq!(reader?.x[5], 0, "torn LDP observations");
    drop(memory);
    Ok(())
}
