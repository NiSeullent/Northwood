use super::super::{
    NativeExit, NoDevices, Stop,
    tests::{BASE, FakeHost, machine, normal_memory},
};
use super::*;
use crate::machine::{exception::Pstate, system::Register};

type TestMachine = Machine<FakeHost, NoDevices>;

#[cfg(all(host_memory, target_arch = "x86_64"))]
mod canonical;

const PTRUE_P0_B: u32 = 0x2518_e3e0; // ptrue p0.b
const INDEX_Z1_0_1: u32 = 0x0421_4001; // index z1.b, #0, #1
const INDEX_Z2_3_0: u32 = 0x0420_4062; // index z2.b, #3, #0
const ADD_Z0_Z1_Z2: u32 = 0x0422_0020; // add z0.b, z1.b, z2.b
const MOVA_ZA0H_Z0: u32 = 0xc000_0000; // mova za0h.b[w12, 0], p0/m, z0.b
const MOVA_Z3_ZA0H: u32 = 0xc002_0003; // mova z3.b, p0/m, za0h.b[w12, 0]
const RDSVL_X5: u32 = 0x04bf_5825; // rdsvl x5, #1
const ZERO_ZA: u32 = 0xc008_00ff; // zero {za}
/// HCRX_EL2.EnFPM: FPMR at EL0 and EL1 outside the host.
const HCRX_ENFPM: u64 = 1 << 23;

/// A PE of the test chip at EL1 with FP, SVE and SME enabled, in
/// streaming mode with ZA, running `words` from BASE; `None` on a chip
/// without SME.
fn streaming(words: &[u32]) -> Option<TestMachine> {
    let mut machine = machine(words);
    machine.cpu.scalable.as_ref()?;
    enable_streaming(&mut machine);
    Some(machine)
}

/// Enable FP, SVE and SME at EL1 (vectors at 0x1_0000), and enter
/// streaming mode with ZA at BASE.
fn enable_streaming<D: Devices>(machine: &mut Machine<FakeHost, D>) {
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    for register in [Register::CPACR_EL1, Register::CPTR_EL2] {
        machine
            .cpu
            .registers
            .set(register, 0, 0b11 << 24 | 0b11 << 20 | 0b11 << 16);
    }
    machine.derived_changed();
    machine.cpu.set_pstate(Pstate::reset(1));
    machine.cpu.set_svcr(true, true).unwrap();
    machine.cpu.pc = BASE;
}

fn z(machine: &TestMachine, register: u8) -> Vec<u8> {
    let mut bytes = alloc::vec![0; STREAMING_VECTOR_BYTES];
    let scalable = machine.cpu.scalable.as_ref().unwrap();
    scalable
        .read_z(&machine.cpu.state, register, &mut bytes)
        .unwrap();
    bytes
}

#[test]
fn streaming_sve_and_sme_instructions_run_on_the_512_bit_state() {
    let words = [
        PTRUE_P0_B,
        INDEX_Z1_0_1,
        INDEX_Z2_3_0,
        ADD_Z0_Z1_Z2,
        MOVA_ZA0H_Z0,
        MOVA_Z3_ZA0H,
        RDSVL_X5,
        ZERO_ZA,
    ];
    // The scalar lowering, and the host's AVX2 and AVX-512 where it has
    // them.
    for target in [
        stela_compiler::TargetCpu::Baseline,
        stela_compiler::TargetCpu::detect(),
    ] {
        run_on(target, &words);
    }
}

fn run_on(target: stela_compiler::TargetCpu, words: &[u32]) {
    let Some(mut machine) = streaming(words) else {
        return;
    };
    machine.target = target;
    let compiled = machine.statistics.compiled;
    for index in 0..words.len() as u64 {
        assert_eq!(machine.step(), Ok(()), "{index}");
        assert_eq!(machine.cpu.pc, BASE + 4 * (index + 1));
    }
    assert_eq!(machine.statistics.compiled - compiled, words.len() as u64);
    // Every element of the 512-bit vectors: Z1 counts up, Z2 is 3.
    let expected: Vec<u8> = (3..3 + STREAMING_VECTOR_BYTES as u8).collect();
    assert_eq!(z(&machine, 0), expected);
    // Through ZA0H.B row 0 and back; then ZERO {ZA} clears the row.
    assert_eq!(z(&machine, 3), expected);
    let mut row = [0xffu8; STREAMING_VECTOR_BYTES];
    let scalable = machine.cpu.scalable.as_ref().unwrap();
    scalable.read_za(0, &mut row).unwrap();
    assert_eq!(row, [0; STREAMING_VECTOR_BYTES]);
    assert_eq!(machine.cpu.state.x[5], STREAMING_VECTOR_BYTES as u64);
    // Running an instruction again reuses its leaf.
    machine.cpu.pc = BASE + 4 * 3;
    let compiled = machine.statistics.compiled;
    machine.step().unwrap();
    assert_eq!(machine.statistics.compiled, compiled);
}

#[test]
fn scalable_floating_point_follows_fpcr_and_sets_fpsr_without_trapping() {
    const PTRUE_P0_S: u32 = 0x2598_e3e0; // ptrue p0.s
    const FDUP_Z0_1: u32 = 0x25b9_ce00; // fdup z0.s, #1.0
    const FDUP_Z1_2: u32 = 0x25b9_c001; // fdup z1.s, #2.0
    const FADD_Z0_Z1: u32 = 0x6580_8020; // fadd z0.s, p0/m, z0.s, z1.s
    const DUP_Z1_0: u32 = 0x25b8_c001; // dup z1.s, #0
    const FDIV_Z0_Z1: u32 = 0x658d_8020; // fdiv z0.s, p0/m, z0.s, z1.s
    const FPCR_DZE: u32 = 1 << 9;
    const FPSR_DZC: u32 = 1 << 1;
    const FMOPA_ZA0_Z0_Z1: u32 = 0x8081_0000; // fmopa za0.s, p0/m, p0/m, z0.s, z1.s
    let words = [
        PTRUE_P0_S,
        FDUP_Z0_1,
        FDUP_Z1_2,
        FADD_Z0_Z1,
        DUP_Z1_0,
        FDIV_Z0_Z1,
        FDUP_Z0_1,
        FDUP_Z1_2,
        FMOPA_ZA0_Z0_Z1,
    ];
    let Some(mut machine) = streaming(&words) else {
        return;
    };
    // Divide-by-zero trapping enabled: streaming mode without FA64
    // ignores it and only accumulates DZC.
    machine.cpu.state.fpcr = FPCR_DZE;
    machine.cpu.state.fpsr = 0;
    for index in 0..4 {
        assert_eq!(machine.step(), Ok(()), "{index}");
    }
    let lanes = |machine: &TestMachine| -> Vec<u32> {
        z(machine, 0)
            .chunks(4)
            .map(|lane| u32::from_le_bytes(lane.try_into().unwrap()))
            .collect()
    };
    assert_eq!(lanes(&machine), [3.0f32.to_bits(); 16]);
    assert_eq!(machine.cpu.state.fpsr, 0);
    for index in 4..6 {
        assert_eq!(machine.step(), Ok(()), "{index}");
    }
    assert_eq!(lanes(&machine), [f32::INFINITY.to_bits(); 16]);
    assert_eq!(machine.cpu.state.fpsr, FPSR_DZC);
    // The outer product of ones and twos: row 0 of ZA0.S is ZA's first
    // array vector.
    for index in 6..9 {
        assert_eq!(machine.step(), Ok(()), "{index}");
    }
    let mut row = [0u8; STREAMING_VECTOR_BYTES];
    let scalable = machine.cpu.scalable.as_ref().unwrap();
    scalable.read_za(0, &mut row).unwrap();
    assert!(
        row.chunks(4)
            .all(|lane| lane == 2.0f32.to_bits().to_le_bytes())
    );
    assert_eq!(machine.cpu.pc, BASE + 36);
}

#[test]
fn streaming_scalar_fp_runs_as_a_block_and_zeroes_the_upper_z_bits_it_writes() {
    const FADD_S7: u32 = 0x1e22_2827; // fadd s7, s1, s2
    const FCMP_S1_S2: u32 = 0x1e22_2020; // fcmp s1, s2
    let Some(mut machine) = streaming(&[FADD_S7, FCMP_S1_S2]) else {
        return;
    };
    // Z0 to Z7 all ones.
    let ones = alloc::vec![0xff; STREAMING_VECTOR_BYTES];
    for register in 0..8 {
        let scalable = machine.cpu.scalable.as_mut().unwrap();
        scalable
            .write_z(&mut machine.cpu.state, register, &ones)
            .unwrap();
    }
    // The unit-test host runs no native block: it reports the
    // fallthrough the FADD's block takes.
    let fallthrough = |pc: u64| NativeExit::Exit {
        next_pc: pc,
        kind: stela_compiler::block::ExitKind::Fallthrough as u64,
    };
    machine.host.exits.push(fallthrough(BASE + 4));
    assert_eq!(machine.step(), Ok(()));
    assert_eq!(machine.cpu.pc, BASE + 4);
    // S7's Z bits above 128 are zero; the other registers keep theirs.
    let mut upper_zero = ones.clone();
    upper_zero[16..].fill(0);
    assert_eq!(z(&machine, 7), upper_zero);
    assert_eq!(z(&machine, 1), ones);
    // FCMP writes no vector register.
    machine.host.exits.push(fallthrough(BASE + 8));
    assert_eq!(machine.step(), Ok(()));
    assert_eq!((z(&machine, 1), z(&machine, 2)), (ones.clone(), ones));
    // Outside streaming mode the FADD is part of an ordinary block.
    machine.cpu.set_svcr(false, false).unwrap();
    machine.cpu.pc = BASE;
    let Ok(Fetched::Block(index)) = machine.fetch_block(BASE, false) else {
        panic!("an ordinary block")
    };
    let block = &machine.code.blocks[index];
    assert_eq!((block.pc, block.instructions), (BASE, 2));
}

#[test]
fn movprfx_runs_as_its_move_and_its_follower_after_it() {
    const PTRUE_P1_S_VL2: u32 = 0x2598_e041; // ptrue p1.s, vl2
    const MOVPRFX_Z3_Z1: u32 = 0x0420_bc23; // movprfx z3, z1
    const MOVPRFX_Z4_P1Z_Z1: u32 = 0x0490_2424; // movprfx z4.s, p1/z, z1.s
    const MOVPRFX_Z5_P1M_Z1: u32 = 0x0491_2425; // movprfx z5.s, p1/m, z1.s
    const MOVPRFX_Z6_Z1: u32 = 0x0420_bc26; // movprfx z6, z1
    const ADD_Z6_P0M_Z2: u32 = 0x0400_0046; // add z6.b, p0/m, z6.b, z2.b
    let words = [
        PTRUE_P0_B,
        PTRUE_P1_S_VL2,
        INDEX_Z1_0_1,
        INDEX_Z2_3_0,
        MOVPRFX_Z3_Z1,
        MOVPRFX_Z4_P1Z_Z1,
        MOVPRFX_Z5_P1M_Z1,
        MOVPRFX_Z6_Z1,
        ADD_Z6_P0M_Z2,
    ];
    let Some(mut machine) = streaming(&words) else {
        return;
    };
    let ones = alloc::vec![0xff; STREAMING_VECTOR_BYTES];
    let scalable = machine.cpu.scalable.as_mut().unwrap();
    for register in [4, 5] {
        scalable
            .write_z(&mut machine.cpu.state, register, &ones)
            .unwrap();
    }
    for index in 0..words.len() as u64 {
        assert_eq!(machine.step(), Ok(()), "{index}");
        assert_eq!(machine.cpu.pc, BASE + 4 * (index + 1));
    }
    let counted: Vec<u8> = (0..STREAMING_VECTOR_BYTES as u8).collect();
    assert_eq!(z(&machine, 3), counted);
    // VL2 makes the first two words active.
    let mut zeroed = counted.clone();
    zeroed[8..].fill(0);
    assert_eq!(z(&machine, 4), zeroed);
    let mut merged = ones.clone();
    merged[..8].copy_from_slice(&counted[..8]);
    assert_eq!(z(&machine, 5), merged);
    let added: Vec<u8> = (3..3 + STREAMING_VECTOR_BYTES as u8).collect();
    assert_eq!(z(&machine, 6), added);
}

/// Every SVE and SME form of the M4, M5 and M6 that runs through
/// [`Machine::prepare_scalable`], its operand fields filled in 24 ways,
/// in each mode its access checks allow: each form has a filling that
/// runs or decodes as UNDEFINED, and no filling stops for a feature, the
/// M6's FPMR-checked FP8 forms included. Scalar FP, Advanced SIMD and
/// non-streaming SVE forms take other paths. The other fillings decline
/// the reserved encodings they hit.
#[test]
fn every_sve_and_sme_form_of_the_sme_chips_runs_or_is_undefined() {
    use {stela_contracts::Chip, stela_contracts::chip::Check};
    // BFSCALE, of FEAT_SVE_BFSCALE, which no Apple chip has, lies inside
    // FSCALE's encodings, and the compiler declines it as unknown: the
    // SVE form, and the multi-vector forms, FSCALE's with size 0.
    const FSCALE: [u32; 4] = [0xc120_a180, 0xc120_a980, 0xc120_b180, 0xc120_b980];
    const SIZE: u32 = 0b11 << 22;
    const BFSCALE: (u32, u32) = (0xffff_e000, 0x6509_8000);
    let mut runs = 0;
    let mut undefined = 0;
    for chip in [Chip::AppleM4, Chip::AppleM5, Chip::AppleM6] {
        let mut machine = super::super::tests::machine_on(chip, &[]);
        machine.target = stela_compiler::TargetCpu::detect();
        // HCRX_EL2.EnFPM gives EL1 FPMR, for the M6's FP8 forms.
        machine.cpu.registers.set(Register::HCRX_EL2, 0, HCRX_ENFPM);
        for (mask, value, checks) in chip.check_forms() {
            if checks
                .iter()
                .any(|check| matches!(check, Check::Fp | Check::FpAdvSimd | Check::NonStreamingSve))
            {
                continue;
            }
            let known = (mask, value) == BFSCALE
                || FSCALE.contains(&value) && mask & SIZE == SIZE && value & SIZE == 0;
            let streaming_only = checks.iter().any(|check| {
                matches!(
                    check,
                    Check::Sve | Check::StreamingSve | Check::StreamingSveAndZa
                )
            });
            for streaming in [true, false] {
                if !streaming && streaming_only {
                    continue;
                }
                machine.cpu.set_svcr(streaming, true).unwrap();
                let profile = machine.scalable_profile().unwrap();
                let (mut tried, mut handled) = (false, false);
                let mut declined = alloc::string::String::new();
                let mut seed = value ^ 0x9e37_79b9;
                for _ in 0..24 {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    let word = value | (seed & !mask);
                    if chip.admit(word) != word || chip.checks(word) != checks {
                        continue;
                    }
                    tried = true;
                    match machine.prepare_scalable(BASE, word, profile) {
                        Ok(Prepared::Undefined) => {
                            handled = true;
                            undefined += 1;
                        }
                        Ok(_) => {
                            handled = true;
                            runs += 1;
                        }
                        Err(error) => {
                            assert!(
                                    known || !matches!(&error, PreparationError::Compile(stela_compiler::CompileError {
                                        kind: stela_compiler::CompileErrorKind::GuestFeaturesUnavailable { .. },
                                        ..
                                    })),
                                    "{chip:?} {word:#010x}: {error}"
                                );
                            declined = error.to_string();
                        }
                    }
                }
                assert!(
                    !tried || handled || known,
                    "{chip:?} streaming {streaming}: {value:#010x} under {mask:#010x}: {declined}"
                );
            }
        }
    }
    assert!(runs > 25_000 && undefined > 1_000, "{runs} {undefined}");
}

#[test]
fn features_the_chip_lacks_and_debug_state_movt_are_undefined_before_their_checks() {
    use stela_contracts::Chip;
    const BFADD_Z0_Z1: u32 = 0x6500_8020; // bfadd z0.h, p0/m, z0.h, z1.h
    const MOVT_X0_ZT0: u32 = 0xc04c_03e0; // movt x0, zt0[0]
    const UNDEFINED: u64 = 1 << 25;
    let on = |chip, word| {
        let mut machine = super::super::tests::machine_on(chip, &[PTRUE_P0_B, word]);
        enable_streaming(&mut machine);
        machine.step().unwrap();
        machine
    };
    // The M4 lacks FEAT_SVE_B16B16: BFADD is UNDEFINED, also with SME
    // disabled, where it would otherwise trap; the M5 runs it.
    let mut m4 = on(Chip::AppleM4, BFADD_Z0_Z1);
    assert_eq!(m4.step(), Ok(()));
    assert_eq!(m4.cpu.read(Register::ESR_EL1), UNDEFINED);
    assert_eq!(m4.cpu.read(Register::ELR_EL1), BASE + 4);
    let mut m5 = on(Chip::AppleM5, BFADD_Z0_Z1);
    assert_eq!(m5.step(), Ok(()));
    assert_eq!(m5.cpu.pc, BASE + 8);
    for (chip, word) in [(Chip::AppleM4, BFADD_Z0_Z1), (Chip::AppleM5, MOVT_X0_ZT0)] {
        let mut machine = on(chip, word);
        for register in [Register::CPACR_EL1, Register::CPTR_EL2] {
            machine.cpu.registers.set(register, 0, 0);
        }
        machine.derived_changed();
        assert_eq!(machine.step(), Ok(()), "{chip:?}");
        assert_eq!(machine.cpu.read(Register::ESR_EL1), UNDEFINED, "{chip:?}");
    }
    // MOVT between ZT0 and an X register exists only in Debug state.
    let mut m6 = on(Chip::AppleM6, MOVT_X0_ZT0);
    assert_eq!(m6.step(), Ok(()));
    assert_eq!(m6.cpu.read(Register::ESR_EL1), UNDEFINED);
    assert_eq!(m6.cpu.read(Register::ELR_EL1), BASE + 4);
}

#[test]
fn dit_stops_explicitly() {
    let Some(mut machine) = streaming(&[ADD_Z0_Z1_Z2]) else {
        return;
    };
    machine.cpu.pstate.dit = true;
    assert!(stop(&mut machine).contains("PSTATE.DIT"));
    assert_eq!(machine.cpu.pc, BASE);
}

fn stop(machine: &mut TestMachine) -> alloc::string::String {
    match machine.step() {
        Err(Stop::Unsupported(requirement)) => alloc::format!("{:?}", requirement.kind),
        other => panic!("{other:?}"),
    }
}

/// RAM at `address`, `bytes` long.
fn ram(machine: &TestMachine, address: u64, bytes: usize) -> Vec<u8> {
    let offset = (address - BASE) as usize;
    machine.host.ram[offset..offset + bytes].to_vec()
}

#[test]
fn scalable_loads_and_stores_reach_translated_memory_element_by_element() {
    const ST1B_Z1_X5: u32 = 0xe400_e0a1; // st1b {z1.b}, p0, [x5]
    const LD1B_Z2_X5: u32 = 0xa400_a0a2; // ld1b {z2.b}, p0/z, [x5]
    const LDR_ZA_X5: u32 = 0xe100_00a0; // ldr za[w12, 0], [x5]
    const STR_ZA_X6: u32 = 0xe120_00c0; // str za[w12, 0], [x6]
    let words = [
        PTRUE_P0_B,
        INDEX_Z1_0_1,
        ST1B_Z1_X5,
        LD1B_Z2_X5,
        LDR_ZA_X5,
        STR_ZA_X6,
    ];
    let Some(mut machine) = streaming(&words) else {
        return;
    };
    normal_memory(&mut machine, false);
    // The ZA store crosses a page.
    let (source, destination) = (BASE + 0x8000, BASE + 0x9000 - 24);
    machine.cpu.state.x[5] = source;
    machine.cpu.state.x[6] = destination;
    for index in 0..words.len() as u64 {
        assert_eq!(machine.step(), Ok(()), "{index}");
        assert_eq!(machine.cpu.pc, BASE + 4 * (index + 1));
    }
    let counted: Vec<u8> = (0..STREAMING_VECTOR_BYTES as u8).collect();
    assert_eq!(ram(&machine, source, STREAMING_VECTOR_BYTES), counted);
    assert_eq!(z(&machine, 2), counted);
    assert_eq!(ram(&machine, destination, STREAMING_VECTOR_BYTES), counted);
}

/// The M6's FP8 SME forms run under FPMR's formats once
/// CheckFPMREnabled passes: the FP8 to FP16 (FEAT_SME_F8F16) and FP32
/// (FEAT_SME_F8F32) outer products, FSCALE (multiple vectors) and the
/// multi-vector FP8 conversions.
#[test]
fn the_m6_fp8_sme_forms_run_under_fpmr() {
    const PTRUE_P1_B: u32 = 0x2518_e3e1; // ptrue p1.b
    const DUP_Z0: u32 = 0x2538_c700; // mov z0.b, #0x38
    const DUP_Z1: u32 = 0x2538_c701; // mov z1.b, #0x38
    const FMOPA_ZA0_H: u32 = 0x80a1_2008; // fmopa za0.h, p0/m, p1/m, z0.b, z1.b
    const FMOPA_ZA1_S: u32 = 0x80a1_2001; // fmopa za1.s, p0/m, p1/m, z0.b, z1.b
    const MOVA_Z3_ZA0H_H: u32 = 0xc042_0003; // mov z3.h, p0/m, za0h.h[w12, 0]
    const MOVA_Z4_ZA1H_S: u32 = 0xc082_0084; // mov z4.s, p0/m, za1h.s[w12, 0]
    const DUP_Z6_3: u32 = 0x25b8_c066; // mov z6.s, #3
    const FSCALE: u32 = 0xc1a6_a184; // fscale {z4.s, z5.s}, {z4.s, z5.s}, z6.s
    const F1CVT: u32 = 0xc126_e006; // f1cvt {z6.h, z7.h}, z0.b
    // F8S1 and F8S2 E4M3, whose 0x38 is 1.0.
    const E4M3: u64 = 1 | 1 << 3;
    let words = [
        PTRUE_P0_B,
        PTRUE_P1_B,
        DUP_Z0,
        DUP_Z1,
        ZERO_ZA,
        FMOPA_ZA0_H,
        FMOPA_ZA1_S,
        MOVA_Z3_ZA0H_H,
        MOVA_Z4_ZA1H_S,
        DUP_Z6_3,
        FSCALE,
        F1CVT,
    ];
    let mut machine = super::super::tests::machine_on(stela_contracts::Chip::AppleM6, &words);
    enable_streaming(&mut machine);
    for _ in 0..5 {
        machine.step().unwrap();
    }
    // HCRX_EL2.EnFPM clear makes them UNDEFINED at EL1.
    assert_eq!(machine.step(), Ok(()));
    assert_eq!(
        (
            machine.cpu.read(Register::ESR_EL1),
            machine.cpu.read(Register::ELR_EL1)
        ),
        (1 << 25, BASE + 20)
    );
    machine.cpu.registers.set(Register::HCRX_EL2, 0, HCRX_ENFPM);
    machine.derived_changed();
    machine.cpu.set_pstate(Pstate::reset(1));
    machine.cpu.scalable.as_mut().unwrap().set_fpmr(E4M3);
    machine.cpu.pc = BASE + 20;
    for index in 5..words.len() as u64 {
        assert_eq!(machine.step(), Ok(()), "{index}");
        assert_eq!(machine.cpu.pc, BASE + 4 * (index + 1), "{index}");
    }
    let repeat = |bytes: &[u8]| -> Vec<u8> {
        bytes
            .iter()
            .copied()
            .cycle()
            .take(STREAMING_VECTOR_BYTES)
            .collect()
    };
    // Two products of 1.0 make each half 2.0, and four each single 4.0,
    // which FSCALE takes to 32.0 (Z5, zero, stays zero); F1CVT widens
    // E4M3 1.0 to half 1.0.
    assert_eq!(z(&machine, 3), repeat(&0x4000u16.to_le_bytes()));
    assert_eq!(z(&machine, 4), repeat(&32.0f32.to_le_bytes()));
    assert_eq!(z(&machine, 5), repeat(&[0]));
    assert_eq!(z(&machine, 6), repeat(&0x3c00u16.to_le_bytes()));
    assert_eq!(z(&machine, 7), repeat(&0x3c00u16.to_le_bytes()));
}

/// On the M5 and M6, with tag checks on, each element of a tag-checked
/// SVE access is checked against its granule's Allocation Tag.
#[test]
fn tag_checked_scalable_accesses_take_tag_check_faults_by_element() {
    const LD1B_Z2_X5: u32 = 0xa400_a0a2; // ld1b {z2.b}, p0/z, [x5]
    const TAGGED: u64 = 0xf0;
    const HCR_ATA: u64 = 1 << 56;
    const SCTLR_ATA_TCF_SYNC: u64 = 1 << 43 | 1 << 40;
    const TCR_TBI0: u64 = 1 << 37;
    for chip in [
        stela_contracts::Chip::AppleM5,
        stela_contracts::Chip::AppleM6,
    ] {
        let mut machine = super::super::tests::machine_on(chip, &[PTRUE_P0_B, LD1B_Z2_X5]);
        enable_streaming(&mut machine);
        normal_memory(&mut machine, false);
        machine.cpu.registers.set(Register::MAIR_EL1, 0, TAGGED);
        for (register, set) in [
            (Register::HCR_EL2, HCR_ATA),
            (Register::SCTLR_EL1, SCTLR_ATA_TCF_SYNC),
            (Register::TCR_EL1, TCR_TBI0),
        ] {
            let value = machine.cpu.read(register) | set;
            machine.cpu.registers.set(register, 0, value);
        }
        machine.translations_changed();
        let data = BASE + 0x8000;
        for offset in (0..STREAMING_VECTOR_BYTES as u64).step_by(16) {
            machine
                .tags
                .set(data + offset, if offset == 16 { 0xb } else { 0xa });
        }
        machine.step().unwrap();
        // The second granule's first element mismatches.
        let pointer = 0xa << 56 | data;
        machine.cpu.state.x[5] = pointer;
        machine.cpu.pc = BASE + 4;
        assert_eq!(machine.step(), Ok(()));
        assert_eq!(
            (
                machine.cpu.read(Register::ESR_EL1),
                machine.cpu.read(Register::FAR_EL1),
                machine.cpu.read(Register::ELR_EL1),
            ),
            (0x25 << 26 | 1 << 25 | 0b01_0001, pointer + 16, BASE + 4)
        );
        // With every tag matching the load completes.
        machine.tags.set(data + 16, 0xa);
        machine.cpu.set_pstate(Pstate::reset(1));
        machine.cpu.set_svcr(true, true).unwrap();
        machine.cpu.pc = BASE + 4;
        assert_eq!(machine.step(), Ok(()));
        assert_eq!(machine.cpu.pc, BASE + 8);
    }
}

#[test]
fn scalable_accesses_take_their_data_aborts_alignment_and_sp_faults() {
    const LD1B_Z2_X5: u32 = 0xa400_a0a2; // ld1b {z2.b}, p0/z, [x5]
    const ST1B_Z1_X5: u32 = 0xe400_e0a1; // st1b {z1.b}, p0, [x5]
    const LD1W_Z0_X5: u32 = 0xa540_a0a0; // ld1w {z0.s}, p0/z, [x5]
    const LD1B_Z2_SP: u32 = 0xa400_a3e2; // ld1b {z2.b}, p0/z, [sp]
    const SCTLR_A: u64 = 1 << 1;
    const SCTLR_SA: u64 = 1 << 3;
    let words = [PTRUE_P0_B, LD1B_Z2_X5, ST1B_Z1_X5, LD1W_Z0_X5, LD1B_Z2_SP];
    let Some(mut machine) = streaming(&words) else {
        return;
    };
    normal_memory(&mut machine, false);
    machine.step().unwrap();
    let run = |machine: &mut TestMachine, index: u64| {
        machine.cpu.set_pstate(Pstate::reset(1));
        machine.cpu.pc = BASE + 4 * index;
        machine.step()
    };
    let syndrome = |machine: &TestMachine| {
        (
            machine.cpu.read(Register::ESR_EL1),
            machine.cpu.read(Register::FAR_EL1),
            machine.cpu.read(Register::ELR_EL1),
        )
    };
    // Outside the mapped gigabyte: a level-1 translation fault, a write
    // for the store, at the instruction and its first element.
    let unmapped = BASE + 0x4000_0000;
    machine.cpu.state.x[5] = unmapped;
    for (index, write) in [(1, 0), (2, 1 << 6)] {
        assert_eq!(run(&mut machine, index), Ok(()));
        assert_eq!(
            syndrome(&machine),
            (
                0x25 << 26 | 1 << 25 | write | 0b00_0101,
                unmapped,
                BASE + 4 * index
            )
        );
    }
    // A store that reaches past RAM stops there, and keeps the elements
    // before it (Z1 is zero).
    let edge = BASE + 0x1_0000 - 16;
    let offset = (edge - BASE) as usize;
    machine.host.ram[offset..].fill(0xff);
    machine.cpu.state.x[5] = edge;
    machine.cpu.set_pstate(Pstate::reset(1));
    machine.cpu.pc = BASE + 8;
    assert!(stop(&mut machine).contains("no device model"));
    assert_eq!(ram(&machine, edge, 16), [0; 16]);
    // With SCTLR_EL1.A an unaligned word element faults (DFSC 0x21).
    let sctlr = machine.cpu.read(Register::SCTLR_EL1);
    machine
        .cpu
        .registers
        .set(Register::SCTLR_EL1, 0, sctlr | SCTLR_A);
    machine.derived_changed();
    machine.cpu.state.x[5] = BASE + 0x8002;
    assert_eq!(run(&mut machine, 3), Ok(()));
    assert_eq!(
        syndrome(&machine),
        (0x25 << 26 | 1 << 25 | 0x21, BASE + 0x8002, BASE + 12)
    );
    // An unaligned SP faults with EC 0x26 under SCTLR_EL1.SA, before
    // any element; without SA the leaf, which checks it, stops.
    machine.cpu.state.sp = BASE + 0x8001;
    machine
        .cpu
        .registers
        .set(Register::SCTLR_EL1, 0, sctlr | SCTLR_SA);
    machine.derived_changed();
    assert_eq!(run(&mut machine, 4), Ok(()));
    assert_eq!(machine.cpu.read(Register::ESR_EL1), 0x26 << 26 | 1 << 25);
    machine
        .cpu
        .registers
        .set(Register::SCTLR_EL1, 0, sctlr & !SCTLR_SA);
    machine.derived_changed();
    machine.cpu.set_pstate(Pstate::reset(1));
    machine.cpu.pc = BASE + 16;
    assert!(stop(&mut machine).contains("SCTLR_ELx.SA clear"));
}
