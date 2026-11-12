use super::super::tests::{BASE, machine, machine_on, normal_memory, one_timer_bank};
use super::{ACTLR_EN_TSO, ACTLR_ICDSB, Code, Translation};
use stela_compiler::aarch64::system::registers::Encoding;
use {
    crate::machine::Stop, crate::machine::cpu::HCR_HCD, crate::machine::cpu::HCR_TGE,
    crate::machine::cpu::HCR_TSC, crate::machine::exception::Pstate,
    crate::machine::exception::Vector, crate::machine::system::Register, stela_contracts::Chip,
    stela_contracts::Feature,
};

/// Configure an unlocked authored CTRR region through the existing
/// generation's register bank. This is not an original-image SMP test.
fn test_ctrr_region(
    machine: &mut TestMachine,
    el: u8,
    first: u64,
    last: u64,
    enabled: bool,
    execute_inside_only: bool,
) {
    use crate::machine::ctrr::{self, Field, LockRegion};
    use stela_contracts::ctrr::Generation;
    match stela_contracts::ctrr::generation(machine.profile.chip) {
        None => panic!("fixture requires an Apple CTRR model"),
        Some(Generation::A) => {
            let mut cluster = machine.cpu.ctrr.borrow_mut();
            let level = usize::from(el == 2);
            cluster.write(level, Field::Lower, first).unwrap();
            cluster.write(level, Field::Upper, last).unwrap();
            let control = if !enabled {
                0
            } else if execute_inside_only {
                ctrr::EXECUTE_INSIDE_ONLY
            } else {
                ctrr::WRITE_PROTECT_MMU_OFF | ctrr::WRITE_PROTECT_MMU_ON
            };
            cluster.write(level, Field::Control, control).unwrap();
        }
        Some(Generation::Ctrr3) => {
            assert!(
                !execute_inside_only,
                "CTXR execution fields remain unqualified"
            );
            let bank = if el == 2 {
                &mut machine.cpu.ctrr_core.el2
            } else {
                &mut machine.cpu.ctrr_core.el1
            };
            bank.write(LockRegion::CtrrC, Field::Lower, first).unwrap();
            bank.write(LockRegion::CtrrC, Field::Upper, last).unwrap();
            bank.write(
                LockRegion::CtrrC,
                Field::Control,
                if enabled { ctrr::CTRR_ENABLE } else { 0 },
            )
            .unwrap();
        }
    }
    machine.translations_changed();
}

/// The alias keeps the original fault VA visibly different from the
/// final protected PA. Both regimes use the fixture's 1 GiB block.
fn test_ctrr_alias(machine: &mut TestMachine) -> u64 {
    const ALIAS: u64 = BASE + (1 << 30);
    normal_memory(machine, false);
    let slot = 0xf000 + ((ALIAS >> 30) as usize) * 8;
    let block = BASE | 1 << 10 | 3 << 8 | 1;
    machine.host.ram[slot..slot + 8].copy_from_slice(&block.to_le_bytes());
    machine.translations_changed();
    ALIAS
}

#[test]
fn ctrr_store_faults_keep_virtual_far_registers_and_memory_then_allow_retry() {
    const LDTR_X3_X2: u32 = 0xf840_0843;
    const STTR_X1_X2: u32 = 0xf800_0841;
    const AT_S1E1W_X2: u32 = 0xd508_7822;
    const ERET: u32 = 0xd69f_03e0;
    const DATA_OFFSET: usize = 0x2080;
    const INITIAL: u64 = 0x0bad_f00d_0123_4567;
    const STORED: u64 = 0x7654_3210_5566_7788;
    for chip in Chip::ALL {
        for el in [1, 2] {
            let mut machine = machine_on(chip, &[]);
            machine.cpu.set_pstate(Pstate {
                uao: true,
                ..Pstate::reset(el)
            });
            machine
                .cpu
                .registers
                .set(Register::HCR_EL2, 0, if el == 2 { 1 << 34 } else { 0 });
            let alias = test_ctrr_alias(&mut machine);
            let (vbar, esr, far, elr, spsr) = if el == 2 {
                (
                    Register::VBAR_EL2,
                    Register::ESR_EL2,
                    Register::FAR_EL2,
                    Register::ELR_EL2,
                    Register::SPSR_EL2,
                )
            } else {
                (
                    Register::VBAR_EL1,
                    Register::ESR_EL1,
                    Register::FAR_EL1,
                    Register::ELR_EL1,
                    Register::SPSR_EL1,
                )
            };
            machine.cpu.registers.set(vbar, 0, 0x1_0000);
            machine.host.ram[DATA_OFFSET..DATA_OFFSET + 8].copy_from_slice(&INITIAL.to_le_bytes());
            machine.cpu.state.x[1] = STORED;
            machine.cpu.state.x[2] = alias + DATA_OFFSET as u64;
            machine.cpu.state.x[8] = 0x55aa;
            machine.cpu.state.sp = BASE + 0x8000;
            machine.cpu.state.nzcv = 0xa000_0000;
            test_ctrr_region(&mut machine, el, BASE + 0x2000, BASE + 0x2fff, true, false);
            // Ordinary reads remain allowed, and AT ignores CTRR's
            // subsequent physical access restriction.
            machine.system_instruction(LDTR_X3_X2).unwrap();
            assert_eq!(machine.cpu.state.x[3], INITIAL, "{chip:?} EL{el}");
            if el == 1 {
                machine.system_instruction(AT_S1E1W_X2).unwrap();
                let par = machine.cpu.read(Register::PAR_EL1);
                assert_eq!(par & 1, 0);
                assert_eq!(par & 0x0000_ffff_ffff_f000, BASE + 0x2000);
            }
            let pc = machine.cpu.pc;
            let before = machine.cpu.state.clone();
            let pstate = machine.cpu.pstate;
            let ram = machine.host.ram.clone();
            machine.system_instruction(STTR_X1_X2).unwrap();
            assert_eq!(machine.cpu.pc, 0x1_0200, "{chip:?} EL{el}");
            assert_eq!(machine.cpu.read(esr), 0x9600_004f, "{chip:?} EL{el}");
            assert_eq!(machine.cpu.read(far), alias + DATA_OFFSET as u64);
            assert_eq!(machine.cpu.read(elr), pc);
            assert_eq!(machine.cpu.read(spsr) & 0xf000_0000, u64::from(before.nzcv));
            assert_eq!(machine.cpu.state, before);
            assert_eq!(machine.host.ram, ram, "forbidden store changed RAM");
            assert!(machine.take_unmodeled().is_empty());
            // ERET restores the precise faulting context. Removing
            // the protection permits the same instruction to retry.
            test_ctrr_region(&mut machine, el, BASE + 0x2000, BASE + 0x2fff, false, false);
            machine.system_instruction(ERET).unwrap();
            assert_eq!((machine.cpu.pc, machine.cpu.pstate), (pc, pstate));
            assert_eq!(machine.cpu.state, before);
            machine.system_instruction(STTR_X1_X2).unwrap();
            assert_eq!(machine.cpu.pc, pc + 4);
            assert_eq!(
                machine.host.ram[DATA_OFFSET..DATA_OFFSET + 8],
                STORED.to_le_bytes()
            );
        }
    }
}

#[test]
fn ctrr_fetch_faults_report_instruction_permission_and_preserve_faulting_state() {
    const NOP: u32 = 0xd503_201f;
    const ERET: u32 = 0xd69f_03e0;
    // Only legacy CTRR's execution rule is established. M4+ CTXR
    // execution-context fields remain an explicit separate limitation.
    for chip in [Chip::AppleM1, Chip::AppleM2, Chip::AppleM3] {
        for el in [1, 2] {
            let mut machine = machine_on(chip, &[NOP]);
            machine.cpu.set_pstate(Pstate::reset(el));
            machine
                .cpu
                .registers
                .set(Register::HCR_EL2, 0, if el == 2 { 1 << 34 } else { 0 });
            let alias = test_ctrr_alias(&mut machine);
            let (vbar, esr, far, elr) = if el == 2 {
                (
                    Register::VBAR_EL2,
                    Register::ESR_EL2,
                    Register::FAR_EL2,
                    Register::ELR_EL2,
                )
            } else {
                (
                    Register::VBAR_EL1,
                    Register::ESR_EL1,
                    Register::FAR_EL1,
                    Register::ELR_EL1,
                )
            };
            machine.cpu.registers.set(vbar, 0, 0x1_0000);
            machine.cpu.pc = alias;
            machine.cpu.state.x[4] = 0x1234;
            machine.cpu.state.sp = BASE + 0x8000;
            machine.cpu.state.nzcv = 0x6000_0000;
            test_ctrr_region(&mut machine, el, BASE + 0x2000, BASE + 0x2fff, true, true);
            let before = machine.cpu.state.clone();
            let pstate = machine.cpu.pstate;
            let ram = machine.host.ram.clone();
            machine.step().unwrap();
            assert_eq!(machine.cpu.pc, 0x1_0200);
            assert_eq!(machine.cpu.read(esr), 0x8600_000f);
            assert_eq!(machine.cpu.read(far), alias);
            assert_eq!(machine.cpu.read(elr), alias);
            assert_eq!(machine.cpu.state, before);
            assert_eq!(machine.host.ram, ram);
            assert!(
                machine.host.installed.is_empty(),
                "forbidden fetch compiled code"
            );
            assert!(machine.take_unmodeled().is_empty());
            test_ctrr_region(&mut machine, el, BASE + 0x2000, BASE + 0x2fff, false, true);
            machine.system_instruction(ERET).unwrap();
            assert_eq!((machine.cpu.pc, machine.cpu.pstate), (alias, pstate));
            assert_eq!(machine.cpu.state, before);
            assert_eq!(machine.fetch_physical(alias), Ok(BASE));
        }
    }
}

#[test]
fn core_monitor_construction_needs_a_checked_layout_and_an_explicit_pe_clock() {
    use {crate::machine::pmu, stela_contracts::identity::Identity};
    for chip in Chip::ALL {
        let mut machine = machine_on(chip, &[]);
        assert!(machine.cpu.pmu.is_none());
        // Without a monitor the counter reads as storage, reported once.
        machine.cpu.state.x[0] = 0x5a;
        machine
            .system_instruction(access(true, pmu::PMC[0], 0))
            .unwrap();
        assert_eq!(machine.cpu.state.x[0], 0);
        assert_eq!(machine.take_unmodeled().len(), 1);
        // A synthetic profile clock tests ownership and routing; it is
        // not a hardware clock declaration for any board.
        machine.profile.processors = alloc::vec![Identity {
            midr: 0,
            mpidr: 0x8000_0000,
            clock: Some(1_000_000_000),
            caches: None,
        }];
        machine.load_profile();
        assert_eq!(machine.cpu.pmu.is_some(), pmu::Pmu::on(chip));
        if pmu::Pmu::on(chip) {
            machine.cpu.state.x[0] = 1 << 48;
            machine
                .system_instruction(access(false, pmu::PMC[1], 0))
                .unwrap();
            machine
                .system_instruction(access(true, pmu::PMC[1], 1))
                .unwrap();
            assert_eq!(
                machine.cpu.state.x[1],
                if chip == Chip::AppleM1 { 0 } else { 1 << 48 }
            );
        }
        if chip == Chip::AppleM6 {
            let extra = Encoding::new(3, 0, 11, 13, 4);
            machine.cpu.state.x[0] = 1 << 63;
            machine.system_instruction(access(false, extra, 0)).unwrap();
            machine.system_instruction(access(true, extra, 1)).unwrap();
            assert_eq!(machine.cpu.state.x[1], 1 << 63);
            machine
                .system_instruction(access(true, pmu::PMSR, 1))
                .unwrap();
            assert_eq!(machine.cpu.state.x[1], 1 << 36);
            machine
                .system_instruction(access(false, Encoding::new(3, 0, 11, 13, 5), 0))
                .unwrap();
            assert_eq!(machine.take_unmodeled().len(), 1);
        }
        machine.profile.processors[0].clock = None;
        machine.load_profile();
        assert!(machine.cpu.pmu.is_none());
    }
}

fn access(read: bool, encoding: Encoding, rt: u32) -> u32 {
    0xd510_0000
        | u32::from(read) << 21
        | u32::from(encoding.op0() & 1) << 19
        | u32::from(encoding.op1()) << 16
        | u32::from(encoding.crn()) << 12
        | u32::from(encoding.crm()) << 8
        | u32::from(encoding.op2()) << 5
        | rt
}

const SCTLR_EL1: Encoding = Encoding::new(3, 0, 1, 0, 0);
const SCTLR_EL12: Encoding = Encoding::new(3, 5, 1, 0, 0);
const CNTKCTL_EL1: Encoding = Encoding::new(3, 0, 14, 1, 0);
const CNTVCT_EL0: Encoding = Encoding::new(3, 3, 14, 0, 2);
const ID_AA64PFR0_EL1: Encoding = Encoding::new(3, 0, 0, 4, 0);
const MIDR_EL1: Encoding = Encoding::new(3, 0, 0, 0, 0);
const TLBI_VMALLE1: u32 = 0xd508_871f;
const HVC: u32 = 0xd400_0002;
const SMC: u32 = 0xd400_0003;
const WFI: u32 = 0xd503_207f;
const HCR_TVM: u64 = 1 << 26;
const HCR_TRVM: u64 = 1 << 30;
const HCR_TID3: u64 = 1 << 18;
const HCR_TTLB: u64 = 1 << 25;
const HCR_TWI: u64 = 1 << 13;

type TestMachine = super::super::Machine<super::super::tests::FakeHost, super::super::NoDevices>;

#[test]
fn system_access_samples_the_counter_and_interrupts_only_when_needed() {
    const ISR_EL1: Encoding = Encoding::new(3, 0, 12, 1, 0);
    const CNTV_TVAL_EL0: Encoding = Encoding::new(3, 3, 14, 3, 0);
    const CNTV_CTL_EL0: Encoding = Encoding::new(3, 3, 14, 3, 1);
    use crate::machine::timer::ENABLE;

    let mut machine = machine_on(Chip::AppleM2, &[]);
    machine.cpu.pstate = Pstate::reset(1);
    let reads = machine.host.counter_reads;
    let queries = machine.statistics.line_queries;

    machine
        .system_instruction(access(true, MIDR_EL1, 0))
        .unwrap();
    machine.system_instruction(TLBI_VMALLE1).unwrap();
    assert_eq!(machine.host.counter_reads, reads);
    assert_eq!(machine.statistics.line_queries, queries);

    machine.host.now = 1000;
    machine.cpu.state.x[1] = 10;
    machine
        .system_instruction(access(false, CNTV_TVAL_EL0, 1))
        .unwrap();
    assert_eq!(machine.cpu.read(Register::CNTV_CVAL_EL0), 1010);
    assert_eq!(machine.host.counter_reads, reads + 1);
    assert_eq!(machine.statistics.line_queries, queries);

    machine.cpu.state.x[1] = ENABLE;
    machine
        .system_instruction(access(false, CNTV_CTL_EL0, 1))
        .unwrap();
    machine.host.now = 1010;
    machine
        .system_instruction(access(true, ISR_EL1, 0))
        .unwrap();
    assert_eq!(machine.cpu.state.x[0] & (1 << 6), 1 << 6);
    assert_eq!(machine.statistics.line_queries, queries + 1);
}

fn el2_trap(machine: &mut TestMachine, word: u32) -> u64 {
    machine.cpu.pc = BASE;
    machine.cpu.pstate = Pstate::reset(1);
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x4_0000);
    machine.system_instruction(word).unwrap();
    // From EL1, a lower EL: the lower-EL AArch64 synchronous vector.
    assert_eq!(
        (machine.cpu.pstate.el, machine.cpu.pc),
        (2, 0x4_0400),
        "{word:#x}"
    );
    machine.cpu.read(Register::ESR_EL2)
}

/// FEAT_CSV2: the host isolates branch predictions at each change of the
/// guest's context (its table bases with their ASIDs and the VMID), and
/// only then: not at exceptions and returns within one address space, nor
/// at a write that keeps the value, nor at a CONTEXTIDR write, which Arm's
/// hardware-described context leaves out.
#[test]
fn guest_context_changes_isolate_branch_predictions() {
    const TTBR0_EL1: Encoding = Encoding::new(3, 0, 2, 0, 0);
    const TTBR1_EL1: Encoding = Encoding::new(3, 0, 2, 0, 1);
    const CONTEXTIDR_EL1: Encoding = Encoding::new(3, 0, 13, 0, 1);
    const VTTBR_EL2: Encoding = Encoding::new(3, 4, 2, 1, 0);
    const MRS_X0_TPIDR_EL1: u32 = 0xd538_d080;
    const ERET: u32 = 0xd69f_03e0;
    let mut machine = machine(&[]);
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    let write = |machine: &mut TestMachine, el, encoding, value| {
        machine.cpu.set_pstate(Pstate::reset(el));
        machine.cpu.state.x[1] = value;
        machine
            .system_instruction(access(false, encoding, 1))
            .unwrap();
        machine.host.isolations
    };
    let mut isolations = machine.host.isolations;
    let writes =
        |machine: &mut TestMachine, isolations: &mut usize, list: &[(u8, Encoding, u64, bool)]| {
            for &(el, encoding, value, changes) in list {
                let after = write(machine, el, encoding, value);
                assert_eq!(after, *isolations + usize::from(changes), "{encoding:?}");
                *isolations = after;
            }
        };
    writes(
        &mut machine,
        &mut isolations,
        &[
            // A new ASID in TTBR0, then the same value again.
            (1, TTBR0_EL1, 1 << 48 | 0x8000, true),
            (1, TTBR0_EL1, 1 << 48 | 0x8000, false),
            (1, TTBR1_EL1, 0xc000, true),
            (1, CONTEXTIDR_EL1, 7, false),
        ],
    );
    // An exception from EL0 to EL1 and the return keep the context.
    machine.cpu.set_pstate(Pstate::reset(0));
    machine.system_instruction(MRS_X0_TPIDR_EL1).unwrap();
    assert_eq!(machine.cpu.pstate.el, 1);
    machine.cpu.registers.set(Register::SPSR_EL1, 0, 0);
    machine.system_instruction(ERET).unwrap();
    assert_eq!(machine.cpu.pstate.el, 0);
    assert_eq!(machine.host.isolations, isolations);
    // A new VMID, written in the EL2&0 host regime, a context of its own.
    writes(
        &mut machine,
        &mut isolations,
        &[(2, VTTBR_EL2, 3 << 48 | 0x4000, true)],
    );
    assert_eq!(
        machine.statistics.prediction_isolations,
        machine.host.isolations as u64
    );
}

#[test]
fn contextidr_updates_live_debug_without_isolating_or_forgetting_cached_controls() {
    const CONTEXTIDR_EL1: Encoding = Encoding::new(3, 0, 13, 0, 1);
    let mut machine = machine_on(Chip::AppleM2, &[]);
    machine.cpu.pstate = Pstate {
        daif: 0,
        ..Pstate::reset(1)
    };
    let registers = &mut machine.cpu.registers;
    registers.set(Register::CNTHP_CTL_EL2, 0, 1);
    registers.set(Register::CNTHP_CVAL_EL2, 0, 400);
    registers.set(Register::OSLSR_EL1, 0, 0b1000);
    registers.set(Register::MDSCR_EL1, 0, 1 << 15 | 1 << 13);
    // Address breakpoint 0 is linked to context-aware breakpoint 5.
    registers.set(Register::DBGBVR_EL1, 0, 0x8000);
    registers.set(
        Register::DBGBCR_EL1,
        0,
        1 << 20 | 5 << 16 | 0b1111 << 5 | 0b01 << 1 | 1,
    );
    registers.set(Register::DBGBVR_EL1, 5, 0x42);
    registers.set(
        Register::DBGBCR_EL1,
        5,
        0b0011 << 20 | 0b1111 << 5 | 0b11 << 1 | 1,
    );
    assert_eq!(machine.timer_deadline(), Some(400));
    machine.floating_point_enabled();
    machine.memory_checks();
    machine.native_system();
    let debug = machine.debug_mode();
    assert!(!crate::machine::debug::breakpoint(&machine.cpu, 0x8000));
    let isolations = machine.host.isolations;
    let regimes = machine.host.regimes.len();
    machine.cpu.state.x[1] = 0x42;
    machine
        .system_instruction(access(false, CONTEXTIDR_EL1, 1))
        .unwrap();
    assert_eq!(machine.host.isolations, isolations);
    assert_eq!(machine.host.regimes.len(), regimes);
    assert_eq!(machine.timers, Some(Some(400)));
    assert!(machine.floating_point.is_some());
    assert!(machine.memory_checks.is_some());
    assert!(machine.native_system.is_some());
    assert_eq!(machine.debug, Some(debug));
    assert!(crate::machine::debug::breakpoint(&machine.cpu, 0x8000));
}

#[test]
fn ccsidr_reads_the_cache_csselr_selects() {
    use {
        stela_contracts::caches::Cache, stela_contracts::caches::Caches,
        stela_contracts::identity::Identity,
    };
    const CLIDR_EL1: Encoding = Encoding::new(3, 1, 0, 0, 1);
    const CCSIDR_EL1: Encoding = Encoding::new(3, 1, 0, 0, 0);
    const CSSELR_EL1: Encoding = Encoding::new(3, 2, 0, 0, 0);
    let cache = |bytes, ways| Cache {
        bytes,
        ways,
        line: 128,
    };
    let caches =
        Caches::split_unified(cache(128 << 10, 8), cache(64 << 10, 8), cache(4 << 20, 16)).unwrap();
    let identity = Identity {
        midr: 0x611f_0320,
        mpidr: 0x8000_0000,
        clock: None,
        caches: Some(caches),
    };
    let mut machine = super::super::tests::board(&[], super::super::NoDevices, vec![identity]);
    let run = |machine: &mut TestMachine, word| {
        machine.cpu.pc = BASE;
        machine.cpu.pstate = Pstate::reset(1);
        machine.system_instruction(word).unwrap();
        machine.cpu.state.x[2]
    };
    assert_eq!(run(&mut machine, access(true, CLIDR_EL1, 2)), 0b100_011);
    for csselr in [0, 1, 2, 3, 0x10] {
        machine.cpu.state.x[1] = csselr;
        run(&mut machine, access(false, CSSELR_EL1, 1));
        assert_eq!(
            run(&mut machine, access(true, CCSIDR_EL1, 2)),
            caches.ccsidr(csselr),
            "{csselr:#x}"
        );
    }
    // XNU's cache_line_size: 16 << the level 1 data cache's LineSize.
    assert_eq!(16 << (caches.ccsidr(0) & 7), 128);
}

#[test]
fn mdcr_el2_tdcc_traps_el0_and_el1_dcc_accesses_to_el2() {
    const MDCCSR_EL0: Encoding = Encoding::new(2, 3, 0, 1, 0);
    const DBGDTR_EL0: Encoding = Encoding::new(2, 3, 0, 4, 0);
    const MDCCINT_EL1: Encoding = Encoding::new(2, 0, 0, 2, 0);
    const TDCC: u64 = 1 << 27;
    let mut machine = machine(&[]);
    // TDCC is FEAT_FGT's; without it (the M1) the bit is RES0.
    let fgt = Feature::FEAT_FGT.on(machine.profile.chip);
    machine.cpu.registers.set(Register::MDCR_EL2, 0, TDCC);
    for word in [
        access(true, MDCCSR_EL0, 1),
        access(false, DBGDTR_EL0, 1),
        access(true, MDCCINT_EL1, 1),
    ] {
        if !fgt {
            machine.cpu.pc = BASE;
            machine.cpu.pstate = Pstate::reset(1);
            machine.system_instruction(word).unwrap();
            assert_eq!((machine.cpu.pstate.el, machine.cpu.pc), (1, BASE + 4));
            continue;
        }
        let esr = el2_trap(&mut machine, word);
        assert_eq!(esr >> 26, 0x18, "{word:#x}");
        assert_eq!(esr & 1, u64::from(word >> 21 & 1), "{word:#x}");
    }
    if fgt {
        // EL0 outside the host traps to EL2 as well, MDSCR_EL1.TDCC clear.
        machine.cpu.pc = BASE;
        machine.cpu.pstate = Pstate::reset(0);
        machine
            .system_instruction(access(true, DBGDTR_EL0, 1))
            .unwrap();
        assert_eq!((machine.cpu.pstate.el, machine.cpu.pc), (2, 0x4_0400));
        assert_eq!(machine.cpu.read(Register::ESR_EL2) >> 26, 0x18);
    }
    // Without TDCC the EL1 read completes.
    machine.cpu.registers.set(Register::MDCR_EL2, 0, 0);
    machine.cpu.pc = BASE;
    machine.cpu.pstate = Pstate::reset(1);
    machine
        .system_instruction(access(true, MDCCSR_EL0, 1))
        .unwrap();
    assert_eq!((machine.cpu.pstate.el, machine.cpu.pc), (1, BASE + 4));
    // MDCR_EL2 keeps TDE, TDA, TDOSA, TDRA and TDCC: without a PMU, SPE,
    // TRBE or MPAM debug the other fields are RES0.
    const MDCR_EL2: Encoding = Encoding::new(3, 4, 1, 1, 1);
    machine.cpu.pstate = Pstate::reset(2);
    machine.cpu.state.x[0] = u64::MAX;
    machine
        .system_instruction(access(false, MDCR_EL2, 0))
        .unwrap();
    machine
        .system_instruction(access(true, MDCR_EL2, 1))
        .unwrap();
    assert_eq!(machine.cpu.state.x[1], if fgt { TDCC } else { 0 } | 0xf00);
}

#[test]
fn hcr_el2_traps_route_el1_accesses_to_el2_with_their_syndromes() {
    let mut machine = machine(&[]);
    for (hcr, word) in [
        (HCR_TVM, access(false, SCTLR_EL1, 1)),
        (HCR_TRVM, access(true, SCTLR_EL1, 1)),
        (HCR_TID3, access(true, ID_AA64PFR0_EL1, 1)),
        (HCR_TTLB, TLBI_VMALLE1),
    ] {
        machine.cpu.registers.set(Register::HCR_EL2, 0, hcr);
        let esr = el2_trap(&mut machine, word);
        assert_eq!(esr >> 26, 0x18, "{word:#x}");
        // ISS: Op0, Op2, Op1, CRn, Rt, CRm and the direction (read = 1).
        assert_eq!(esr & 1, u64::from(word >> 21 & 1), "{word:#x}");
        assert_eq!(esr >> 10 & 0xf, u64::from(word >> 12 & 0xf), "{word:#x}");
    }
    // Without the trap the EL1 write reaches SCTLR_EL1.
    machine.cpu.registers.set(Register::HCR_EL2, 0, 0);
    machine.cpu.pstate = Pstate::reset(1);
    machine.cpu.state.x[1] = 0x1005;
    machine
        .system_instruction(access(false, SCTLR_EL1, 1))
        .unwrap();
    assert_eq!(machine.cpu.read(Register::SCTLR_EL1) & 0xffff, 0x1985);
    // HCR_EL2.TWI traps WFI and TSC traps SMC from EL1 to EL2.
    machine.cpu.registers.set(Register::HCR_EL2, 0, HCR_TWI);
    assert_eq!(el2_trap(&mut machine, WFI) >> 26, 0x01);
    machine
        .cpu
        .registers
        .set(Register::HCR_EL2, 0, 1 << HCR_TSC);
    assert_eq!(el2_trap(&mut machine, SMC) >> 26, 0x17);
    // HVC goes to EL2 unless HCR_EL2.HCD disables it.
    machine.cpu.registers.set(Register::HCR_EL2, 0, 0);
    assert_eq!(el2_trap(&mut machine, HVC) >> 26, 0x16);
    machine
        .cpu
        .registers
        .set(Register::HCR_EL2, 0, 1 << HCR_HCD);
    machine.cpu.pc = BASE;
    machine.cpu.pstate = Pstate::reset(1);
    machine.system_instruction(HVC).unwrap();
    assert_eq!(
        machine.cpu.read(Register::ESR_EL1) >> 26,
        0,
        "UNDEFINED at EL1"
    );
}

#[test]
fn svcr_smstart_and_the_sme_registers_follow_the_chips_sme() {
    const SVCR: Encoding = Encoding::new(3, 3, 4, 2, 2);
    const SMCR_EL1: Encoding = Encoding::new(3, 0, 1, 2, 6);
    const SMPRI_EL1: Encoding = Encoding::new(3, 0, 1, 2, 4);
    const TPIDR2_EL0: Encoding = Encoding::new(3, 3, 13, 0, 5);
    const SMSTART_ZA: u32 = 0xd503_457f;
    const SMSTOP_ZA: u32 = 0xd503_447f;
    const SMSTART: u32 = 0xd503_477f;
    const SMSTOP: u32 = 0xd503_467f;
    const SME_TRAP: u64 = 0x1d << 26 | 1 << 25;
    const SMEN: u64 = 0b11 << 24;
    let mut machine = machine(&[]);
    let chip = machine.profile.chip;
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x2_0000);
    machine.derived_changed();
    let run = |machine: &mut TestMachine, el: u8, word: u32| {
        machine.cpu.pstate = Pstate {
            sp: el != 0,
            ..Pstate::reset(el.max(1))
        };
        machine.cpu.pstate.el = el;
        machine.cpu.pc = BASE;
        machine.system_instruction(word)
    };
    if !Feature::FEAT_SME.on(chip) {
        for word in [SMSTART_ZA, access(true, SVCR, 0)] {
            run(&mut machine, 1, word).unwrap();
            assert_eq!(machine.cpu.read(Register::ESR_EL1), 0x0200_0000);
        }
        return;
    }
    // CPACR_EL1.SMEN and CPTR_EL2.SMEN clear trap SME accesses from EL1
    // with EC 0x1D, SMTC 0: to EL1, then to EL2.
    machine.cpu.registers.set(Register::CPTR_EL2, 0, SMEN);
    for word in [SMSTART_ZA, access(true, SVCR, 0)] {
        run(&mut machine, 1, word).unwrap();
        assert_eq!(
            (machine.cpu.pstate.el, machine.cpu.read(Register::ESR_EL1)),
            (1, SME_TRAP)
        );
    }
    // SMEN 0b01 enables EL1 but traps EL0.
    machine
        .cpu
        .registers
        .set(Register::CPACR_EL1, 0, 0b01 << 24);
    run(&mut machine, 0, SMSTART_ZA).unwrap();
    assert_eq!(
        (machine.cpu.pstate.el, machine.cpu.read(Register::ESR_EL1)),
        (1, SME_TRAP)
    );
    machine.cpu.registers.set(Register::CPTR_EL2, 0, 0);
    run(&mut machine, 1, SMSTART_ZA).unwrap();
    assert_eq!(
        (machine.cpu.pstate.el, machine.cpu.read(Register::ESR_EL2)),
        (2, SME_TRAP)
    );
    machine.cpu.registers.set(Register::CPTR_EL2, 0, SMEN);
    machine.cpu.registers.set(Register::CPACR_EL1, 0, SMEN);
    // SMSTART ZA and SMSTOP ZA set and clear SVCR.ZA, as MSR SVCR does;
    // FPSR keeps its value while SM does not change.
    machine.cpu.state.fpsr = 0x1f;
    run(&mut machine, 0, SMSTART_ZA).unwrap();
    run(&mut machine, 0, access(true, SVCR, 2)).unwrap();
    assert_eq!((machine.cpu.state.x[2], machine.cpu.pc), (0b10, BASE + 4));
    run(&mut machine, 0, SMSTOP_ZA).unwrap();
    assert_eq!(machine.cpu.svcr(), 0);
    machine.cpu.state.x[3] = !1;
    run(&mut machine, 1, access(false, SVCR, 3)).unwrap();
    assert_eq!((machine.cpu.svcr(), machine.cpu.state.fpsr), (0b10, 0x1f));
    // SMSTART enters streaming mode, which resets FPSR; MSR SVCR and
    // SMSTOP leave it.
    run(&mut machine, 1, SMSTART).unwrap();
    assert_eq!(
        (machine.cpu.svcr(), machine.cpu.state.fpsr),
        (0b11, 0x0800_009f)
    );
    machine.cpu.state.x[4] = 1;
    run(&mut machine, 1, access(false, SVCR, 4)).unwrap();
    assert_eq!(machine.cpu.svcr(), 0b01);
    run(&mut machine, 1, SMSTOP).unwrap();
    assert_eq!(machine.cpu.svcr(), 0);
    // SMIDR_EL1 derives from MPIDR_EL1, which this profile has no value
    // for: reading it stops.
    const SMIDR_EL1: Encoding = Encoding::new(3, 1, 0, 0, 6);
    assert!(matches!(
        run(&mut machine, 1, access(true, SMIDR_EL1, 9)),
        Err(Stop::Unsupported(_))
    ));
    // SMCR_EL1, SMPRI_EL1 and TPIDR2_EL0 hold the chip's fields, once
    // HFGRTR_EL2 and HFGWTR_EL2 set nSMPRI_EL1 and nTPIDR2_EL0 and
    // SCTLR_EL1.EnTP2 lets EL0 reach TPIDR2_EL0.
    for register in [Register::HFGRTR_EL2, Register::HFGWTR_EL2] {
        machine.cpu.registers.set(register, 0, 0b11 << 54);
    }
    let sctlr = machine.cpu.read(Register::SCTLR_EL1);
    machine
        .cpu
        .registers
        .set(Register::SCTLR_EL1, 0, sctlr | 1 << 60);
    for (el, encoding, register) in [
        (1, SMCR_EL1, Register::SMCR_EL1),
        (1, SMPRI_EL1, Register::SMPRI_EL1),
        (0, TPIDR2_EL0, Register::TPIDR2_EL0),
    ] {
        machine.cpu.state.x[5] = u64::MAX;
        run(&mut machine, el, access(false, encoding, 5)).unwrap();
        run(&mut machine, el, access(true, encoding, 6)).unwrap();
        assert_eq!(
            (machine.cpu.state.x[6], machine.cpu.pc),
            (!register.res0(chip), BASE + 4),
            "{register:?}"
        );
    }
    // At EL2, SMCR_EL1 names SMCR_EL2 and SMCR_EL12 reaches SMCR_EL1.
    const SMCR_EL2: Encoding = Encoding::new(3, 4, 1, 2, 6);
    const SMCR_EL12: Encoding = Encoding::new(3, 5, 1, 2, 6);
    const SMPRIMAP_EL2: Encoding = Encoding::new(3, 4, 1, 2, 5);
    for (encoding, length, register) in [
        (SMCR_EL1, 1, Register::SMCR_EL2),
        (SMCR_EL12, 2, Register::SMCR_EL1),
        (SMCR_EL2, 3, Register::SMCR_EL2),
    ] {
        machine.cpu.state.x[7] = length;
        run(&mut machine, 2, access(false, encoding, 7)).unwrap();
        assert_eq!(machine.cpu.read(register) & 0xf, length, "{encoding:?}");
        run(&mut machine, 2, access(true, encoding, 8)).unwrap();
        assert_eq!(machine.cpu.state.x[8] & 0xf, length, "{encoding:?}");
    }
    machine.cpu.state.x[7] = u64::MAX;
    run(&mut machine, 2, access(false, SMPRIMAP_EL2, 7)).unwrap();
    assert_eq!(
        machine.cpu.read(Register::SMPRIMAP_EL2),
        !Register::SMPRIMAP_EL2.res0(chip)
    );
}

#[test]
fn smidr_tpidr2_priority_mapping_and_mpamsm_follow_the_chips_sme() {
    use stela_contracts::identity::Identity;
    const SMIDR_EL1: Encoding = Encoding::new(3, 1, 0, 0, 6);
    const TPIDR2_EL0: Encoding = Encoding::new(3, 3, 13, 0, 5);
    const HCRX_EL2: Encoding = Encoding::new(3, 4, 1, 2, 2);
    const SCTLR_ENTP2: u64 = 1 << 60;
    const HCRX_SMPME: u64 = 1 << 5;
    const MPAM2_EL2: Encoding = Encoding::new(3, 4, 10, 5, 0);
    const MPAMSM_EL1: Encoding = Encoding::new(3, 0, 10, 5, 3);
    const MPAM2_ENMPAMSM: u64 = 1 << 50;
    // A P-cluster core: Aff2 1, Aff1 1.
    const MPIDR: u64 = 0x8001_0100;
    let identity = Identity {
        midr: 0x611f_0490,
        mpidr: MPIDR,
        clock: None,
        caches: None,
    };
    let mut machine = super::super::tests::board(&[], super::super::NoDevices, vec![identity]);
    let chip = machine.profile.chip;
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x2_0000);
    machine.derived_changed();
    let run = |machine: &mut TestMachine, el: u8, word: u32| {
        machine.cpu.pstate = Pstate {
            sp: el != 0,
            ..Pstate::reset(el.max(1))
        };
        machine.cpu.pstate.el = el;
        machine.cpu.pc = BASE;
        machine.system_instruction(word).unwrap();
    };
    run(&mut machine, 1, access(true, SMIDR_EL1, 1));
    if !Feature::FEAT_SME.on(chip) {
        // An unimplemented ID register traps with EC 0x18.
        assert_eq!(
            machine.cpu.read(Register::ESR_EL1),
            0x18 << 26 | 1 << 25 | trap_iss_of(true, SMIDR_EL1, 1)
        );
        return;
    }
    // Apple's implementer code, priorities supported, shared by the
    // cluster: Affinity2 and Affinity hold Aff2:Aff1.
    assert_eq!(
        (machine.cpu.state.x[1], machine.cpu.pc),
        (0x61 << 24 | 1 << 15 | 0b11 << 13 | 0x101, BASE + 4)
    );
    // In the host, SCTLR_EL2.EnTP2 clear traps EL0's TPIDR2_EL0 accesses
    // to EL2 with EC 0x18; set, they reach the register.
    machine
        .cpu
        .registers
        .set(Register::HCR_EL2, 0, 1 << 34 | 1 << 27);
    machine.cpu.registers.set(Register::TPIDR2_EL0, 0, 0x5a);
    machine.derived_changed();
    for (read, rt) in [(true, 2), (false, 2)] {
        run(&mut machine, 0, access(read, TPIDR2_EL0, rt));
        assert_eq!(
            (machine.cpu.pstate.el, machine.cpu.read(Register::ESR_EL2)),
            (2, 0x18 << 26 | 1 << 25 | trap_iss_of(read, TPIDR2_EL0, rt)),
            "{read}"
        );
    }
    let sctlr = machine.cpu.read(Register::SCTLR_EL2);
    machine
        .cpu
        .registers
        .set(Register::SCTLR_EL2, 0, sctlr | SCTLR_ENTP2);
    run(&mut machine, 0, access(true, TPIDR2_EL0, 3));
    assert_eq!((machine.cpu.state.x[3], machine.cpu.pc), (0x5a, BASE + 4));
    // HCRX_EL2.SMPME holds, as SMIDR_EL1.SMPS has priority mapping.
    machine.cpu.state.x[4] = HCRX_SMPME;
    run(&mut machine, 2, access(false, HCRX_EL2, 4));
    run(&mut machine, 2, access(true, HCRX_EL2, 5));
    assert_eq!(machine.cpu.state.x[5] & HCRX_SMPME, HCRX_SMPME);
    // MPAM2_EL2.EnMPAMSM clear traps EL1's MPAMSM_EL1 accesses to EL2;
    // set, MPAMSM_EL1 holds PARTID_D and PMG_D, which tag nothing with
    // MPAMIDR_EL1 0.
    machine.cpu.registers.set(Register::HCR_EL2, 0, 0);
    machine.cpu.state.x[6] = 0;
    run(&mut machine, 2, access(false, MPAM2_EL2, 6));
    machine.cpu.state.x[7] = u64::MAX;
    run(&mut machine, 1, access(false, MPAMSM_EL1, 7));
    assert_eq!(
        (machine.cpu.pstate.el, machine.cpu.read(Register::ESR_EL2)),
        (2, 0x18 << 26 | 1 << 25 | trap_iss_of(false, MPAMSM_EL1, 7))
    );
    machine.cpu.state.x[6] = MPAM2_ENMPAMSM;
    run(&mut machine, 2, access(false, MPAM2_EL2, 6));
    run(&mut machine, 2, access(true, MPAM2_EL2, 8));
    assert_eq!(machine.cpu.state.x[8], MPAM2_ENMPAMSM);
    run(&mut machine, 1, access(false, MPAMSM_EL1, 7));
    run(&mut machine, 1, access(true, MPAMSM_EL1, 9));
    assert_eq!(
        (machine.cpu.state.x[9], machine.cpu.pc),
        (0x0000_ff00_ffff_0000, BASE + 4)
    );
}

/// The ISS of a trapped MRS (`read`) or MSR of `encoding` with `rt`.
fn trap_iss_of(read: bool, encoding: Encoding, rt: u32) -> u64 {
    u64::from(encoding.op0()) << 20
        | u64::from(encoding.op2()) << 17
        | u64::from(encoding.op1()) << 14
        | u64::from(encoding.crn()) << 10
        | u64::from(rt) << 5
        | u64::from(encoding.crm()) << 1
        | u64::from(read)
}

#[test]
fn fpmr_holds_its_fields_behind_enfpm_and_clears_as_streaming_mode_changes() {
    const FPMR: Encoding = Encoding::new(3, 3, 4, 4, 2);
    const SMSTART: u32 = 0xd503_477f;
    const SMSTOP: u32 = 0xd503_467f;
    const ENABLED: u64 = 0b11 << 24 | 0b11 << 20;
    const HCRX_ENFPM: u64 = 1 << 23;
    const SCTLR_ENFPM: u64 = 1 << 34;
    // EC 0x18 with ISS Op0 3, Op2 2, Op1 3, CRn 4, CRm 4 and a read of Xt.
    let fpmr_trap = |rt: u64| 0x18 << 26 | 1 << 25 | 0x34_d009 | rt << 5;
    let mut machine = machine(&[]);
    let chip = machine.profile.chip;
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x2_0000);
    machine.cpu.registers.set(Register::CPACR_EL1, 0, ENABLED);
    machine.cpu.registers.set(Register::CPTR_EL2, 0, ENABLED);
    machine.derived_changed();
    let run = |machine: &mut TestMachine, el: u8, word: u32| {
        machine.cpu.pstate = Pstate {
            sp: el != 0,
            ..Pstate::reset(el.max(1))
        };
        machine.cpu.pstate.el = el;
        machine.cpu.pc = BASE;
        machine.system_instruction(word)
    };
    if !Feature::FEAT_FPMR.on(chip) {
        run(&mut machine, 1, access(true, FPMR, 0)).unwrap();
        assert_eq!(machine.cpu.read(Register::ESR_EL1), 0x0200_0000);
        return;
    }
    // HCRX_EL2.EnFPM clear traps EL1's accesses to EL2 with EC 0x18.
    run(&mut machine, 1, access(true, FPMR, 0)).unwrap();
    assert_eq!(
        (machine.cpu.pstate.el, machine.cpu.read(Register::ESR_EL2)),
        (2, fpmr_trap(0))
    );
    machine.cpu.registers.set(Register::HCRX_EL2, 0, HCRX_ENFPM);
    // FPMR holds every field and reads its RES0 bits as 0.
    machine.cpu.state.x[3] = u64::MAX;
    run(&mut machine, 1, access(false, FPMR, 3)).unwrap();
    run(&mut machine, 1, access(true, FPMR, 4)).unwrap();
    assert_eq!(machine.cpu.state.x[4], 0x0000_003f_ff7f_c1ff);
    assert_eq!(machine.cpu.pc, BASE + 4);
    // EL0 also needs SCTLR_EL1.EnFPM, whose trap goes to EL1.
    run(&mut machine, 0, access(true, FPMR, 5)).unwrap();
    assert_eq!(
        (machine.cpu.pstate.el, machine.cpu.read(Register::ESR_EL1)),
        (1, fpmr_trap(5))
    );
    let sctlr = machine.cpu.read(Register::SCTLR_EL1);
    machine
        .cpu
        .registers
        .set(Register::SCTLR_EL1, 0, sctlr | SCTLR_ENFPM);
    run(&mut machine, 0, access(true, FPMR, 5)).unwrap();
    assert_eq!(
        (machine.cpu.pstate.el, machine.cpu.state.x[5]),
        (0, 0x0000_003f_ff7f_c1ff)
    );
    // Entering and leaving streaming mode resets FPMR (ResetSVEState).
    run(&mut machine, 1, SMSTART).unwrap();
    assert_eq!(machine.cpu.svcr() & 1, 1);
    run(&mut machine, 1, access(true, FPMR, 6)).unwrap();
    assert_eq!(machine.cpu.state.x[6], 0);
    run(&mut machine, 1, access(false, FPMR, 3)).unwrap();
    run(&mut machine, 1, SMSTOP).unwrap();
    run(&mut machine, 1, access(true, FPMR, 6)).unwrap();
    assert_eq!((machine.cpu.svcr(), machine.cpu.state.x[6]), (0, 0));
}

#[test]
fn hcrx_and_vncr_hold_their_fields_at_el2_where_the_chip_has_hcx_and_nv2() {
    const HCRX_EL2: Encoding = Encoding::new(3, 4, 1, 2, 2);
    const VNCR_EL2: Encoding = Encoding::new(3, 4, 2, 2, 0);
    const HCR_NV: u64 = 1 << 42;
    let mut machine = machine(&[]);
    let chip = machine.profile.chip;
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x2_0000);
    machine.derived_changed();
    for (encoding, register, feature) in [
        (HCRX_EL2, Register::HCRX_EL2, Feature::FEAT_HCX),
        (VNCR_EL2, Register::VNCR_EL2, Feature::FEAT_NV2),
    ] {
        // At EL2, MSR then MRS of every bit returns the chip's fields
        // (HCRX_EL2 has none on the M3, FNB being RES0 without
        // FEAT_TLBID); without the feature both are UNDEFINED.
        machine.cpu.pstate = Pstate::reset(2);
        machine.cpu.pc = BASE;
        machine.cpu.registers.set(Register::HCR_EL2, 0, 0);
        machine.cpu.state.x[0] = u64::MAX;
        machine
            .system_instruction(access(false, encoding, 0))
            .unwrap();
        machine.cpu.state.x[1] = 1;
        machine
            .system_instruction(access(true, encoding, 1))
            .unwrap();
        if feature.on(chip) {
            assert_eq!(
                (machine.cpu.state.x[1], machine.cpu.pc),
                (!register.res0(chip), BASE + 8),
                "{register:?}"
            );
        } else {
            assert_eq!(machine.cpu.read(Register::ESR_EL2), 0x0200_0000);
            assert_eq!(machine.cpu.state.x[1], 1, "{register:?}");
        }
        // At EL1 it is UNDEFINED, or under HCR_EL2.NV traps to EL2
        // (EC 0x18).
        machine.cpu.pstate = Pstate::reset(1);
        machine.cpu.pc = BASE;
        machine
            .system_instruction(access(true, encoding, 1))
            .unwrap();
        assert_eq!(machine.cpu.read(Register::ESR_EL1), 0x0200_0000);
        if Feature::FEAT_NV.on(chip) {
            machine.cpu.registers.set(Register::HCR_EL2, 0, HCR_NV);
            let esr = el2_trap(&mut machine, access(true, encoding, 1));
            assert_eq!(esr >> 26, 0x18, "{register:?}");
        }
    }
}

#[test]
fn hcr_el2_at_traps_el1_address_translation_where_the_chip_has_nv() {
    const AT_S1E1R_X0: u32 = 0xd508_7800;
    const HCR_AT: u64 = 1 << 44;
    let mut machine = machine(&[]);
    let chip = machine.profile.chip;
    machine.cpu.registers.set(Register::HCR_EL2, 0, HCR_AT);
    if Feature::FEAT_NV.on(chip) {
        assert_eq!(machine.cpu.read(Register::HCR_EL2) & HCR_AT, HCR_AT);
        let esr = el2_trap(&mut machine, AT_S1E1R_X0);
        assert_eq!(esr >> 26, 0x18);
    } else {
        // RES0 without FEAT_NV: AT completes at EL1.
        assert_eq!(machine.cpu.read(Register::HCR_EL2) & HCR_AT, 0);
        machine.cpu.pstate = Pstate::reset(1);
        machine.cpu.pc = BASE;
        machine.system_instruction(AT_S1E1R_X0).unwrap();
        assert_eq!((machine.cpu.pstate.el, machine.cpu.pc), (1, BASE + 4));
    }
}

#[test]
fn the_m6_error_record_group_status_reads_zero_and_fgt2_traps_it() {
    const ERXGSR_EL1: Encoding = Encoding::new(3, 0, 5, 3, 2);
    const HFGRTR2_NERXGSR: u64 = 1 << 1;
    const HCR_TERR: u64 = 1 << 36;
    let on = |chip| {
        let mut machine = machine_on(chip, &[]);
        machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
        machine.derived_changed();
        machine
    };
    let mut machine = on(Chip::AppleM6);
    // HFGRTR2_EL2.nERXGSR_EL1 clear traps EL1's read to EL2 (EC 0x18).
    let esr = el2_trap(&mut machine, access(true, ERXGSR_EL1, 3));
    assert_eq!(esr >> 26, 0x18);
    // Set, the read returns 0: there is no error record group.
    machine
        .cpu
        .registers
        .set(Register::HFGRTR2_EL2, 0, HFGRTR2_NERXGSR);
    assert_eq!(machine.cpu.read(Register::HFGRTR2_EL2), HFGRTR2_NERXGSR);
    machine.cpu.pc = BASE;
    machine.cpu.pstate = Pstate::reset(1);
    machine.cpu.state.x[3] = u64::MAX;
    machine
        .system_instruction(access(true, ERXGSR_EL1, 3))
        .unwrap();
    assert_eq!((machine.cpu.state.x[3], machine.cpu.pc), (0, BASE + 4));
    // DISR_EL1 holds RASv2's WU (17:16), WnRV (14) and WnR (13) as
    // written.
    const DISR_EL1: Encoding = Encoding::new(3, 0, 12, 1, 1);
    const DISR_WRITE_FIELDS: u64 = 0b11 << 16 | 1 << 14 | 1 << 13;
    machine.cpu.state.x[4] = DISR_WRITE_FIELDS;
    machine
        .system_instruction(access(false, DISR_EL1, 4))
        .unwrap();
    assert_eq!(machine.cpu.read(Register::DISR_EL1), DISR_WRITE_FIELDS);
    // HCR_EL2.TERR traps it all the same.
    machine.cpu.registers.set(Register::HCR_EL2, 0, HCR_TERR);
    let esr = el2_trap(&mut machine, access(true, ERXGSR_EL1, 3));
    assert_eq!(esr >> 26, 0x18);
    // Without FEAT_RASv2 (the M5) it is UNDEFINED.
    let mut machine = on(Chip::AppleM5);
    machine
        .system_instruction(access(true, ERXGSR_EL1, 3))
        .unwrap();
    assert_eq!(machine.cpu.read(Register::ESR_EL1), 0x0200_0000);
}

#[test]
fn sctlr2_holds_the_m6_pacm_enables_under_hcrx_el2_sctlr2en() {
    const SCTLR2_EL1: Encoding = Encoding::new(3, 0, 1, 0, 3);
    const SCTLR2_EL12: Encoding = Encoding::new(3, 5, 1, 0, 3);
    const SCTLR2_EL3: Encoding = Encoding::new(3, 6, 1, 0, 3);
    const ENPACM: u64 = 1 << 7 | 1 << 8;
    const HCRX_SCTLR2EN: u64 = 1 << 15;
    let on = |chip| {
        let mut machine = machine_on(chip, &[]);
        machine.derived_changed();
        machine
    };
    let mut machine = on(Chip::AppleM6);
    // HCRX_EL2.SCTLR2En clear traps EL1's access to EL2 (EC 0x18).
    machine.cpu.state.x[1] = u64::MAX;
    let esr = el2_trap(&mut machine, access(false, SCTLR2_EL1, 1));
    assert_eq!(esr >> 26, 0x18);
    // Set, the write holds EnPACM and EnPACM0; every other bit is RES0.
    machine
        .cpu
        .registers
        .set(Register::HCRX_EL2, 0, HCRX_SCTLR2EN);
    machine.cpu.pstate = Pstate::reset(1);
    machine
        .system_instruction(access(false, SCTLR2_EL1, 1))
        .unwrap();
    assert_eq!(machine.cpu.read(Register::SCTLR2_EL1), ENPACM);
    // In the host SCTLR2_EL1 names SCTLR2_EL2 and SCTLR2_EL12 reaches
    // SCTLR2_EL1.
    machine.cpu.pstate = Pstate::reset(2);
    machine.cpu.state.x[2] = 1 << 7;
    machine
        .system_instruction(access(false, SCTLR2_EL1, 2))
        .unwrap();
    assert_eq!(machine.cpu.read(Register::SCTLR2_EL2), 1 << 7);
    machine
        .system_instruction(access(true, SCTLR2_EL12, 3))
        .unwrap();
    assert_eq!(machine.cpu.state.x[3], ENPACM);
    // SCTLR2_EL3 is UNDEFINED, there being no EL3.
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x4_0000);
    machine
        .system_instruction(access(true, SCTLR2_EL3, 3))
        .unwrap();
    assert_eq!(machine.cpu.read(Register::ESR_EL2), 0x0200_0000);
    // Without FEAT_SCTLR2 (the M5) SCTLR2_EL1 is UNDEFINED.
    let mut machine = on(Chip::AppleM5);
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    machine
        .system_instruction(access(true, SCTLR2_EL1, 3))
        .unwrap();
    assert_eq!(machine.cpu.read(Register::ESR_EL1), 0x0200_0000);
}

#[test]
fn a_vhe_host_redirects_el1_names_and_reaches_el1_through_el12() {
    let mut machine = machine_on(one_timer_bank(), &[]);
    machine.cpu.pstate = Pstate::reset(2);
    machine.cpu.state.x[1] = 0x1005;
    machine
        .system_instruction(access(false, SCTLR_EL1, 1))
        .unwrap();
    assert_eq!(machine.cpu.read(Register::SCTLR_EL2) & 1, 1);
    assert_eq!(machine.cpu.read(Register::SCTLR_EL1) & 1, 0);
    machine.cpu.state.x[2] = 1;
    machine
        .system_instruction(access(false, SCTLR_EL12, 2))
        .unwrap();
    assert_eq!(machine.cpu.read(Register::SCTLR_EL1) & 1, 1);
    machine.cpu.registers.set(Register::CNTHCTL_EL2, 0, 0x3);
    machine
        .system_instruction(access(true, CNTKCTL_EL1, 3))
        .unwrap();
    assert_eq!(
        machine.cpu.state.x[3] & 3,
        3,
        "CNTKCTL_EL1 names CNTHCTL_EL2"
    );
}

#[test]
fn el0_counter_traps_follow_cntkctl_or_cnthctl_in_the_host() {
    let mut machine = machine_on(one_timer_bank(), &[]);
    let user = Pstate {
        el: 0,
        sp: false,
        daif: 0,
        ..Pstate::reset(1)
    };
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x4_0000);
    machine.cpu.pstate = user;
    machine
        .system_instruction(access(true, CNTVCT_EL0, 0))
        .unwrap();
    assert_eq!((machine.cpu.pstate.el, machine.cpu.pc), (1, 0x1_0400));
    // In the host the trap goes to EL2 and CNTHCTL_EL2 governs it.
    machine
        .cpu
        .registers
        .set(Register::HCR_EL2, 0, 1 << HCR_TGE);
    machine.cpu.pstate = user;
    machine.cpu.pc = BASE;
    machine
        .system_instruction(access(true, CNTVCT_EL0, 0))
        .unwrap();
    assert_eq!((machine.cpu.pstate.el, machine.cpu.pc), (2, 0x4_0400));
    machine.cpu.registers.set(Register::CNTHCTL_EL2, 0, 1 << 1);
    machine.cpu.pstate = user;
    machine.cpu.pc = BASE;
    machine.host.now = 77;
    machine
        .system_instruction(access(true, CNTVCT_EL0, 4))
        .unwrap();
    assert_eq!((machine.cpu.state.x[4], machine.cpu.pstate.el), (77, 0));
}

/// The M3's two timer banks: the Apple registers reach the Apple bank
/// with VHE redirection and the architectural traps, which report the
/// Apple encoding; the architectural encodings reach the bank
/// AGTCNTRDIR selects for the regime, the Arm bank counting 1 GHz. The
/// M2 has one bank and none of the Apple registers.
#[test]
fn apple_generic_timer_banks_follow_the_counter_redirect_from_the_m3_on() {
    use crate::machine::timer::{ENABLE, ISTATUS, agt};
    const CNTFRQ_EL0: Encoding = Encoding::new(3, 3, 14, 0, 0);
    const CNTV_CTL_EL0: Encoding = Encoding::new(3, 3, 14, 3, 1);
    let mut machine = machine_on(Chip::AppleM3, &[]);
    machine
        .cpu
        .registers
        .set(Register::HCR_EL2, 0, 1 << HCR_TGE);
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x4_0000);
    let run = |machine: &mut TestMachine, el, read, encoding, value| {
        machine.cpu.set_pstate(Pstate::reset(el));
        machine.cpu.pc = BASE;
        machine.cpu.state.x[1] = value;
        machine
            .system_instruction(access(read, encoding, 1))
            .unwrap();
        machine.cpu.state.x[1]
    };
    let apple =
        |machine: &TestMachine, register| machine.cpu.apple.generic_timer.get(register).unwrap();
    // The kernel's timer at EL2 is the Apple bank's EL2 virtual timer.
    machine.host.now = 1000;
    run(&mut machine, 2, false, agt::AGTCNTV_TVAL_EL0, 50);
    run(&mut machine, 2, false, agt::AGTCNTV_CTL_EL0, ENABLE);
    assert_eq!(apple(&machine, Register::CNTHV_CVAL_EL2), 1050);
    assert_eq!(machine.cpu.read(Register::CNTHV_CTL_EL2), 0);
    machine.host.now = 1050;
    assert_eq!(
        run(&mut machine, 2, true, agt::AGTCNTV_CTL_EL0, 0),
        ENABLE | ISTATUS
    );
    assert_eq!(run(&mut machine, 2, true, agt::AGTCNTVCT_EL0, 0), 1050);
    // KCTL names CNTHCTL_EL2 at EL2, the EL12 and EL02 forms and
    // AGTCNTKCTL_NOREDIR the EL1&0 regime's registers.
    run(&mut machine, 2, false, agt::AGTCNTKCTL_EL1, 0xf);
    assert_eq!(apple(&machine, Register::CNTHCTL_EL2) & 0xf, 0xf);
    run(&mut machine, 2, false, agt::AGTCNTKCTL_EL12, 0x2);
    assert_eq!(apple(&machine, Register::CNTKCTL_EL1) & 0xf, 0x2);
    run(&mut machine, 2, false, agt::AGTCNTKCTL_NOREDIR_EL1, 0x1);
    assert_eq!(apple(&machine, Register::CNTKCTL_EL1) & 0xf, 0x1);
    run(&mut machine, 2, false, agt::AGTCNTV_CTL_EL02, ENABLE);
    assert_eq!(apple(&machine, Register::CNTV_CTL_EL0), ENABLE);
    // Firmware leaves the redirect disabled: the architectural
    // encodings reach the Apple bank. With it enabled for the host they
    // reach the Arm bank, on the 1 GHz count, unless EL0's bit gives a
    // thread the Apple bank.
    assert_eq!(run(&mut machine, 2, true, CNTFRQ_EL0, 0), 24_000_000);
    assert_eq!(
        run(&mut machine, 2, true, CNTV_CTL_EL0, 0),
        ENABLE | ISTATUS
    );
    run(&mut machine, 2, false, agt::AGTCNTRDIR_EL1, 0);
    assert_eq!(run(&mut machine, 2, true, CNTFRQ_EL0, 0), 1_000_000_000);
    assert_eq!(run(&mut machine, 2, true, CNTVCT_EL0, 0), 43_750);
    run(&mut machine, 2, false, CNTV_CTL_EL0, ENABLE);
    assert_eq!(machine.cpu.read(Register::CNTHV_CTL_EL2), ENABLE);
    machine.cpu.registers.set(Register::CNTHCTL_EL2, 0, 0b11);
    assert_eq!(run(&mut machine, 0, true, CNTFRQ_EL0, 0), 1_000_000_000);
    run(&mut machine, 2, false, agt::AGTCNTRDIR_EL1, 1);
    assert_eq!(run(&mut machine, 0, true, CNTVCT_EL0, 0), 1050);
    assert_eq!(run(&mut machine, 2, true, CNTFRQ_EL0, 0), 1_000_000_000);
    // EL0 reads of the Apple count trap under the Apple bank's controls,
    // with the Apple encoding in the syndrome.
    machine
        .cpu
        .apple
        .generic_timer
        .set(Register::CNTHCTL_EL2, 0);
    run(&mut machine, 0, true, agt::AGTCNTVCTSS_EL0, 0);
    assert_eq!((machine.cpu.pstate.el, machine.cpu.pc), (2, 0x4_0400));
    let iss = machine.cpu.read(Register::ESR_EL2) & 0x1ff_ffff;
    assert_eq!(iss >> 10 & 0xf, 15, "CRn of the Apple encoding");
    // AGTCNTRDIR: per regime; the EL12 form is EL2's alone, and only an
    // unknown bit is reported.
    assert_eq!(run(&mut machine, 2, true, agt::AGTCNTRDIR_EL12, 0), 3);
    run(&mut machine, 2, false, agt::AGTCNTRDIR_EL12, 0);
    assert!(machine.take_unmodeled().is_empty());
    machine.cpu.set_pstate(Pstate::reset(2));
    machine.cpu.pc = BASE;
    machine.cpu.state.x[1] = 1 << 2;
    run(&mut machine, 2, false, agt::AGTCNTRDIR_EL1, 1 << 2);
    assert_eq!(machine.take_unmodeled().len(), 1);
    machine.cpu.registers.set(Register::HCR_EL2, 0, 0);
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    run(&mut machine, 1, true, agt::AGTCNTRDIR_EL12, 0);
    assert_eq!(
        (machine.cpu.pc, machine.cpu.read(Register::ESR_EL1) >> 26),
        (0x1_0200, 0),
        "UNDEFINED"
    );
    let mut m2 = machine_on(Chip::AppleM2, &[]);
    assert_eq!(run(&mut m2, 2, true, CNTFRQ_EL0, 0), 24_000_000);
    m2.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    run(&mut m2, 1, true, agt::AGTCNTV_CTL_EL0, 0);
    assert_eq!(
        (m2.cpu.pc, m2.cpu.read(Register::ESR_EL1) >> 26),
        (0x1_0200, 0)
    );
}

#[test]
fn sctlr_el0_controls_trap_to_el1_or_to_el2_in_the_host_and_set_dzp() {
    const CTR_EL0: Encoding = Encoding::new(3, 3, 0, 0, 1);
    const DCZID_EL0: Encoding = Encoding::new(3, 3, 0, 0, 7);
    const DC_CVAC_X0: u32 = 0xd50b_7a20;
    const IC_IVAU_X0: u32 = 0xd50b_7520;
    const DC_ZVA_X0: u32 = 0xd50b_7420;
    const UCT: u64 = 1 << 15;
    const DZE: u64 = 1 << 14;
    const UCI: u64 = 1 << 26;
    const TDZ: u64 = 1 << 28;
    let mut machine = machine(&[]);
    normal_memory(&mut machine, true);
    let user = Pstate {
        el: 0,
        sp: false,
        daif: 0,
        ..Pstate::reset(1)
    };
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x4_0000);
    let sctlr = machine.cpu.read(Register::SCTLR_EL1) & !(UCT | DZE | UCI);
    machine.cpu.registers.set(Register::SCTLR_EL1, 0, sctlr);
    let run = |machine: &mut TestMachine, pstate: Pstate, word: u32| {
        machine.cpu.pstate = pstate;
        machine.cpu.pc = BASE;
        machine.cpu.state.x[0] = BASE + 0x100;
        machine.system_instruction(word).unwrap();
        (machine.cpu.pstate.el, machine.cpu.pc)
    };
    // With UCT, UCI and DZE clear, EL0 traps to EL1 with EC 0x18 and the
    // instruction's own Op0, Op2, Op1, CRn, Rt, CRm and direction.
    for (word, iss) in [
        (access(true, CTR_EL0, 0), 0x32_c001),
        (DC_CVAC_X0, 0x12_dc14),
        (IC_IVAU_X0, 0x12_dc0a),
        (DC_ZVA_X0, 0x12_dc08),
    ] {
        assert_eq!(run(&mut machine, user, word), (1, 0x1_0400), "{word:#x}");
        assert_eq!(
            machine.cpu.read(Register::ESR_EL1),
            0x18 << 26 | 1 << 25 | iss,
            "{word:#x}"
        );
    }
    // DCZID_EL0.DZP reports the DC ZVA trap; BS is the profile's 4.
    let dczid = access(true, DCZID_EL0, 1);
    assert_eq!(run(&mut machine, user, dczid), (0, BASE + 4));
    assert_eq!(machine.cpu.state.x[1], 0x14);
    machine
        .cpu
        .registers
        .set(Register::SCTLR_EL1, 0, sctlr | UCT | DZE | UCI);
    // With UCT set the read reaches CTR_EL0, which this machine's profile
    // does not give: reported once.
    machine.cpu.pstate = user;
    machine
        .system_instruction(access(true, CTR_EL0, 1))
        .unwrap();
    assert_eq!(machine.take_unmodeled().len(), 1);
    for word in [DC_CVAC_X0, IC_IVAU_X0] {
        assert_eq!(run(&mut machine, user, word), (0, BASE + 4), "{word:#x}");
    }
    machine.host.ram[0x100..0x140].fill(0xaa);
    assert_eq!(run(&mut machine, user, DC_ZVA_X0), (0, BASE + 4));
    assert!(machine.host.ram[0x100..0x140].iter().all(|&b| b == 0));
    run(&mut machine, user, dczid);
    assert_eq!(machine.cpu.state.x[1], 0x4);
    // HCR_EL2.TDZ prohibits DC ZVA at EL0 and EL1, trapping to EL2.
    machine.cpu.registers.set(Register::HCR_EL2, 0, TDZ);
    for pstate in [user, Pstate::reset(1)] {
        run(&mut machine, pstate, dczid);
        assert_eq!(machine.cpu.state.x[1], 0x14);
        assert_eq!(run(&mut machine, pstate, DC_ZVA_X0), (2, 0x4_0400));
    }
    // In the host SCTLR_EL2 governs EL0, trapping to EL2 (WFI and WFE
    // through nTWI and nTWE), and EL2 is never prohibited DC ZVA.
    machine
        .cpu
        .registers
        .set(Register::HCR_EL2, 0, 1 << HCR_TGE | 1 << 34);
    const NTWI: u64 = 1 << 16;
    const NTWE: u64 = 1 << 18;
    let sctlr2 = machine.cpu.read(Register::SCTLR_EL2) & !(UCT | DZE | UCI | NTWI | NTWE);
    machine.cpu.registers.set(Register::SCTLR_EL2, 0, sctlr2);
    run(&mut machine, user, dczid);
    assert_eq!(machine.cpu.state.x[1], 0x14);
    machine.cpu.event = false;
    for (word, esr) in [
        (access(true, CTR_EL0, 0), 0x18 << 26 | 1 << 25 | 0x32_c001),
        (DC_CVAC_X0, 0x18 << 26 | 1 << 25 | 0x12_dc14),
        (DC_ZVA_X0, 0x18 << 26 | 1 << 25 | 0x12_dc08),
        (WFI, 0x01 << 26 | 1 << 25 | 1 << 24 | 0xe << 20),
        (0xd503_205f, 0x01 << 26 | 1 << 25 | 1 << 24 | 0xe << 20 | 1),
    ] {
        assert_eq!(run(&mut machine, user, word), (2, 0x4_0400), "{word:#x}");
        assert_eq!(machine.cpu.read(Register::ESR_EL2), esr, "{word:#x}");
    }
    machine
        .cpu
        .registers
        .set(Register::SCTLR_EL2, 0, sctlr2 | UCI | DZE);
    run(&mut machine, user, dczid);
    assert_eq!(machine.cpu.state.x[1], 0x4);
    for word in [DC_CVAC_X0, DC_ZVA_X0] {
        assert_eq!(run(&mut machine, user, word), (0, BASE + 4), "{word:#x}");
    }
    machine.cpu.registers.set(Register::SCTLR_EL2, 0, sctlr2);
    run(&mut machine, Pstate::reset(2), dczid);
    assert_eq!(machine.cpu.state.x[1], 0x4);
}

/// The contexts the native-completion tests below run in: each EL, in
/// and outside the host, under every combination of five
/// controls (HCR_EL2.TDZ, SCTLR_EL1.DZE, SCTLR_EL2.DZE, HFGITR_EL2.DCZVA
/// and HFGRTR_EL2.TPIDR_EL1, by bit of `controls`).
fn native_completion_contexts(mut each: impl FnMut(&mut TestMachine, Pstate, String)) {
    const DZE: u64 = 1 << 14;
    const TDZ: u64 = 1 << 28;
    const E2H: u64 = 1 << 34;
    const HFGITR_DCZVA: u64 = 1 << 11;
    const HFGRTR_TPIDR_EL1: u64 = 1 << 33;
    let mut machine = machine(&[]);
    normal_memory(&mut machine, true);
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x4_0000);
    let user = Pstate {
        el: 0,
        sp: false,
        daif: 0,
        ..Pstate::reset(1)
    };
    let sctlr = [Register::SCTLR_EL1, Register::SCTLR_EL2]
        .map(|register| machine.cpu.read(register) & !DZE);
    for (pstate, host) in [
        (user, false),
        (user, true),
        (Pstate::reset(1), false),
        (Pstate::reset(2), false),
        (Pstate::reset(2), true),
    ] {
        for controls in 0..32u32 {
            let bit = |index: u32, value: u64| if controls & 1 << index != 0 { value } else { 0 };
            let hcr = if host { 1 << HCR_TGE | E2H } else { E2H } | bit(0, TDZ);
            for (register, value) in [
                (Register::HCR_EL2, hcr),
                (Register::SCTLR_EL1, sctlr[0] | bit(1, DZE)),
                (Register::SCTLR_EL2, sctlr[1] | bit(2, DZE)),
                (Register::HFGITR_EL2, bit(3, HFGITR_DCZVA)),
                (Register::HFGRTR_EL2, bit(4, HFGRTR_TPIDR_EL1)),
            ] {
                machine.cpu.registers.set(register, 0, value);
            }
            machine.derived_changed();
            machine.cpu.pstate = pstate;
            machine.cpu.pc = BASE;
            let case = format!("EL{} host {host} controls {controls:#b}", pstate.el);
            each(&mut machine, pstate, case);
        }
    }
}

/// Native code zeroes DC ZVA's block in exactly the contexts where the
/// generated accessor completes DC ZVA with no trap or stop.
#[test]
fn native_dc_zva_is_exactly_the_untrapped_dc_zva() {
    const DC_ZVA_X0: u32 = 0xd50b_7420;
    let mut zeroes = 0;
    native_completion_contexts(|machine, pstate, case| {
        machine.cpu.state.x[0] = BASE + 0x108;
        machine.host.ram[0x100..0x140].fill(0xaa);
        let native = super::native_zero_block(&machine.cpu, &machine.profile_missing);
        machine.system_instruction(DC_ZVA_X0).unwrap();
        let zeroed = (machine.cpu.pstate.el, machine.cpu.pc) == (pstate.el, BASE + 4)
            && machine.host.ram[0x100..0x140].iter().all(|&byte| byte == 0);
        assert_eq!(native, zeroed.then_some(64), "{case}");
        zeroes += usize::from(zeroed);
    });
    // DZE, TDZ and, with FEAT_FGT, HFGITR_EL2.DCZVA leave some contexts.
    assert!(zeroes > 0 && zeroes < 5 * 32, "{zeroes}");
    // With stage 1 off, data is Device-nGnRnE: DC ZVA takes the
    // Alignment fault (DFSC 0b100001, WnR) with FAR Xt, never natively.
    let mut machine = machine(&[]);
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    machine.cpu.state.x[0] = BASE + 0x108;
    assert_eq!(machine.cpu.read(Register::SCTLR_EL1) & 1, 0);
    assert_eq!(
        super::native_zero_block(&machine.cpu, &machine.profile_missing),
        None
    );
    machine.system_instruction(DC_ZVA_X0).unwrap();
    assert_eq!(machine.cpu.pc, 0x1_0200);
    assert_eq!(machine.cpu.read(Register::ESR_EL1), 0x9600_0061);
    assert_eq!(machine.cpu.read(Register::FAR_EL1), BASE + 0x108);
}

/// MRS of a mirrored register opens the cell native code reads in
/// exactly the contexts where the generated accessor reads the register
/// with no trap or UNDEFINED.
#[test]
fn only_writes_a_read_follows_close_the_cells() {
    use stela_compiler::x86_64::register_read::Cell;
    const MRS_X1_TPIDR_EL1: u32 = 0xd538_d081;
    const MSR_CONTEXTIDR_X0: u32 = 0xd518_d020;
    const MSR_APIAKEYLO_X0: u32 = 0xd518_2100;
    const MSR_TPIDR_EL1_X0: u32 = 0xd518_d080;
    const MSR_CPACR_X0: u32 = 0xd518_1040;
    let mut machine = machine(&[]);
    machine.cpu.pstate = Pstate::reset(1);
    // HCR_EL2.APK: the keys do not trap.
    machine.cpu.registers.set(Register::HCR_EL2, 0, 1 << 40);
    let index = super::super::MIRRORED
        .iter()
        .position(|&encoding| encoding == Encoding::new(3, 0, 13, 0, 4))
        .unwrap();
    machine.cpu.registers.set(Register::TPIDR_EL1, 0, 0x7000);
    machine.system_instruction(MRS_X1_TPIDR_EL1).unwrap();
    let open = machine.mirrors[index];
    assert_eq!(
        open,
        Cell {
            readable: 1,
            value: 0x7000
        }
    );
    for word in [MSR_CONTEXTIDR_X0, MSR_APIAKEYLO_X0] {
        machine.system_instruction(word).unwrap();
        assert_eq!(machine.mirrors[index], open);
    }
    // Its own write reaches the cell.
    machine.cpu.state.x[0] = 0x8000;
    machine.system_instruction(MSR_TPIDR_EL1_X0).unwrap();
    assert_eq!(
        machine.mirrors[index],
        Cell {
            readable: 1,
            value: 0x8000
        }
    );
    // CPACR_EL1 decides traps of reads.
    machine.cpu.state.x[0] = machine.cpu.registers.get(Register::CPACR_EL1, 0) ^ 3 << 20;
    machine.system_instruction(MSR_CPACR_X0).unwrap();
    assert_eq!(machine.mirrors[index], Cell::default());
}

#[test]
fn counter_cells_take_the_count_at_each_transition() {
    use super::super::COUNTER_MIRRORS;
    use crate::machine::pmu::{self, Context, Filter, Pmu};
    const MRS_X3_PMC1: u32 = 0xd53a_f103;
    let mut machine = machine(&[]);
    machine.cpu.pstate = Pstate::reset(1);
    let context = Context {
        el: 1,
        filter: Filter::Guest,
        waiting: false,
    };
    let mut monitor = Pmu::new(36_000_000, 24_000_000);
    monitor.write(pmu::PMCR1, 1 << 17, 0, 0, context).unwrap();
    monitor
        .write(pmu::PMCR1_EL12, 1 << 17, 0, 0, context)
        .unwrap();
    monitor.write(pmu::PMCR0, 2, 0, 0, context).unwrap();
    machine.cpu.pmu = Some(monitor);
    machine.cpu.retired = 10;
    machine.system_instruction(MRS_X3_PMC1).unwrap();
    let index = COUNTER_MIRRORS[1];
    assert_eq!(machine.mirrors[index].readable, 1);
    assert_eq!(machine.mirrors[index].value, machine.cpu.state.x[3]);
    let read = machine.cpu.state.x[3];
    machine.cpu.retired += 100;
    assert_eq!(machine.mirrors[index].value, read);
    machine.transitioned();
    assert_eq!(machine.mirrors[index].value, read + 100);
}

#[test]
fn written_registers_open_their_cells_on_plain_writes_only() {
    use super::super::{WRITTEN, written_cell};
    use stela_compiler::x86_64::register_read::Cell;
    const MSR_ELR_X0: u32 = 0xd518_4020;
    const MSR_SP_EL0_X0: u32 = 0xd518_4100;
    const MSR_FPSR_X0: u32 = 0xd51b_4420;
    let mut machine = machine(&[]);
    machine.cpu.pstate = Pstate::reset(1);
    machine.cpu.pstate.sp = true;
    // CPACR_EL1.FPEN: FPSR does not trap.
    machine.cpu.registers.set(Register::CPACR_EL1, 0, 3 << 20);
    machine.cpu.state.x[0] = 0x4000;
    let cell = |mirrors: &super::super::Mirrors, register| {
        let index = WRITTEN.iter().position(|&(_, named)| named == register);
        mirrors[written_cell(index.unwrap())]
    };
    machine.system_instruction(MSR_ELR_X0).unwrap();
    let address = machine.cpu.registers.address(Register::ELR_EL1, 0).unwrap();
    assert_eq!(
        cell(&machine.mirrors, Register::ELR_EL1),
        Cell {
            readable: 1,
            value: address
        }
    );
    machine.system_instruction(MSR_SP_EL0_X0).unwrap();
    let address = machine.cpu.registers.address(Register::SP_EL0, 0).unwrap();
    assert_eq!(
        cell(&machine.mirrors, Register::SP_EL0),
        Cell {
            readable: 1,
            value: address
        }
    );
    machine.system_instruction(MSR_FPSR_X0).unwrap();
    assert_eq!(
        cell(&machine.mirrors, Register::FPSR),
        Cell {
            readable: 1,
            value: 0
        }
    );
    // EL0 traps them: nothing opens.
    let mut machine = super::super::tests::machine(&[]);
    machine.cpu.pstate = Pstate::reset(0);
    machine.system_instruction(MSR_ELR_X0).unwrap();
    assert_eq!(cell(&machine.mirrors, Register::ELR_EL1), Cell::default());
}

#[test]
fn mirrored_cells_open_exactly_on_the_plain_reads() {
    const THREAD: u64 = 0xffff_fe00_1234_5000;
    let mut reads = [0; super::super::MIRRORED.len()];
    native_completion_contexts(|machine, pstate, case| {
        for (index, &encoding) in super::super::MIRRORED.iter().enumerate() {
            for register in [
                Register::TPIDR_EL0,
                Register::TPIDRRO_EL0,
                Register::TPIDR_EL1,
            ] {
                machine.cpu.registers.set(register, 0, THREAD);
            }
            machine.cpu.pstate = pstate;
            machine.cpu.pc = BASE;
            machine.cpu.state.x[1] = 0;
            let exceptions = machine.statistics.exceptions;
            // Apple's registers may stop the run where they are not modeled.
            let Ok(()) = machine.system_instruction(access(true, encoding, 1)) else {
                continue;
            };
            // Outside a guarded level a GL bank read is refused, a
            // degraded answer no cell keeps.
            let read = (machine.cpu.pstate.el, machine.cpu.pc) == (pstate.el, BASE + 4)
                && machine.statistics.exceptions == exceptions
                && (pstate.guarded || !super::super::GUARDED_MIRRORS.contains(&index))
                && (machine.cpu.pmu.is_some() || !super::super::COUNTER_MIRRORS.contains(&index));
            let value = machine.cpu.state.x[1];
            assert_eq!(
                machine.mirrors[index],
                stela_compiler::x86_64::register_read::Cell {
                    readable: u64::from(read),
                    value: if read { value } else { 0 },
                },
                "{case} {encoding:?}"
            );
            reads[index] += usize::from(read);
            *machine.mirrors = Default::default();
        }
    });
    // MRS TPIDR_EL1 is UNDEFINED at EL0; at EL1 HFGRTR_EL2.TPIDR_EL1 traps
    // with FEAT_FGT.
    let machine = machine(&[]);
    let fine_grained = Feature::FEAT_FGT.on(machine.cpu.registers.chip());
    let index = |encoding| {
        super::super::MIRRORED
            .iter()
            .position(|&e| e == encoding)
            .unwrap()
    };
    assert_eq!(
        reads[index(Encoding::new(3, 0, 13, 0, 4))],
        if fine_grained { 16 + 64 } else { 32 + 64 }
    );
}

#[test]
fn rmr_el2_resets_into_aarch64_and_a_warm_reset_needs_a_reset_vector() {
    const RMR_EL2: Encoding = Encoding::new(3, 4, 12, 0, 2);
    let mut machine = machine(&[]);
    machine.cpu.pstate = Pstate::reset(2);
    // Without AArch32, AA64 is RAO/WI and bits 63:2 are RES0.
    machine.cpu.state.x[0] = !0b11;
    machine
        .system_instruction(access(false, RMR_EL2, 0))
        .unwrap();
    machine
        .system_instruction(access(true, RMR_EL2, 1))
        .unwrap();
    assert_eq!(machine.cpu.state.x[1], 1);
    // A warm reset request without a reset vector is reported once and
    // leaves the register as it was.
    machine.cpu.state.x[0] = 0b10;
    machine
        .system_instruction(access(false, RMR_EL2, 0))
        .unwrap();
    assert_eq!(machine.take_unmodeled().len(), 1);
    assert_eq!(machine.cpu.registers.get(Register::RMR_EL2, 0), 1);
}

#[test]
fn rvbar_el2_reads_the_reset_address_the_board_holds() {
    const RVBAR_EL2: Encoding = Encoding::new(3, 4, 12, 0, 1);
    const RESET: u64 = 0x8_0300_c000;
    /// A board holding PE 0's reset address, as Apple's `cpu-impl-reg`
    /// RVBAR does.
    struct ResetBoard;
    impl crate::machine::Devices for ResetBoard {
        fn read(
            &mut self,
            _: usize,
            _: u64,
            _: u8,
            _: u64,
        ) -> Result<u64, crate::machine::DeviceError> {
            Err(crate::machine::DeviceError::NoDevice)
        }
        fn write(
            &mut self,
            _: usize,
            _: u64,
            _: u8,
            _: u64,
            _: u64,
        ) -> Result<(), crate::machine::DeviceError> {
            Err(crate::machine::DeviceError::NoDevice)
        }
        fn lines(&mut self, _: usize, _: u64) -> crate::machine::Lines {
            crate::machine::Lines::default()
        }
        fn reset_vector(&self, cpu: usize) -> Option<u64> {
            (cpu == 0).then_some(RESET)
        }
    }
    let mut machine = super::super::tests::board(&[], ResetBoard, Vec::new());
    machine.cpu.pstate = Pstate::reset(2);
    machine
        .system_instruction(access(true, RVBAR_EL2, 1))
        .unwrap();
    assert_eq!(machine.cpu.state.x[1], RESET);
    // RVBAR_EL2 is read-only and UNDEFINED below EL2.
    machine.cpu.pstate = Pstate::reset(1);
    machine.cpu.pc = BASE;
    machine
        .system_instruction(access(true, RVBAR_EL2, 1))
        .unwrap();
    assert_eq!(machine.cpu.read(Register::ESR_EL1) >> 26, 0);
    // A board without one leaves the reset address unknown: the read is
    // reported once.
    let mut bare = machine_on(stela_contracts::Chip::AppleM2, &[]);
    bare.cpu.pstate = Pstate::reset(2);
    bare.system_instruction(access(true, RVBAR_EL2, 1)).unwrap();
    assert_eq!(bare.take_unmodeled().len(), 1);
}

#[test]
fn el1_reads_vmpidr_el2_as_mpidr_el1_and_el2_the_physical_value() {
    const MPIDR_EL1: Encoding = Encoding::new(3, 0, 0, 0, 5);
    const VMPIDR_EL2: Encoding = Encoding::new(3, 4, 0, 0, 5);
    let mut machine = machine(&[]);
    let physical = 0x8000_0103;
    machine.cpu.registers.set(Register::MPIDR_EL1, 0, physical);
    machine
        .profile_missing
        .retain(|register| *register != Register::MPIDR_EL1);
    let read = |machine: &mut TestMachine, el: u8, encoding: Encoding| {
        machine.cpu.pstate = Pstate::reset(el);
        machine.cpu.state.x[1] = 0x5a5a;
        machine
            .system_instruction(access(true, encoding, 1))
            .unwrap();
        machine.cpu.state.x[1]
    };
    assert_eq!(read(&mut machine, 2, MPIDR_EL1), physical);
    // VMPIDR_EL2 keeps Aff3..Aff0, U and MT; bit 31 is RES1 and the
    // rest RES0.
    machine.cpu.pstate = Pstate::reset(2);
    machine.cpu.state.x[0] = u64::MAX;
    machine
        .system_instruction(access(false, VMPIDR_EL2, 0))
        .unwrap();
    let virtual_mpidr = 0xff_c1ff_ffff;
    assert_eq!(read(&mut machine, 2, VMPIDR_EL2), virtual_mpidr);
    assert_eq!(read(&mut machine, 1, MPIDR_EL1), virtual_mpidr);
    assert_eq!(read(&mut machine, 2, MPIDR_EL1), physical);
}

#[test]
fn fields_without_aarch32_or_lpa2_are_reserved() {
    const SCTLR_EL2: Encoding = Encoding::new(3, 4, 1, 0, 0);
    const TCR_EL1: Encoding = Encoding::new(3, 0, 2, 0, 2);
    const TCR_EL2: Encoding = Encoding::new(3, 4, 2, 0, 2);
    let mut machine = machine(&[]);
    let round_trip = |machine: &mut TestMachine, encoding: Encoding, value: u64| {
        machine.cpu.pstate = Pstate::reset(2);
        machine.cpu.state.x[0] = value;
        machine
            .system_instruction(access(false, encoding, 0))
            .unwrap();
        machine
            .system_instruction(access(true, encoding, 1))
            .unwrap();
        machine.cpu.state.x[1]
    };
    // With E2H set and no AArch32 at EL0, SCTLR_EL2.SED (bit 8) and bit 7
    // are RES1 and CP15BEN (bit 5) is RES0, as in SCTLR_EL1.
    let sed = 1 << 8;
    let cp15ben = 1 << 5;
    let res1 = sed | 1 << 7;
    assert_eq!(
        round_trip(&mut machine, SCTLR_EL2, cp15ben) & (res1 | cp15ben),
        res1
    );
    assert_eq!(
        machine.cpu.registers.get(Register::SCTLR_EL2, 0) & cp15ben,
        0
    );
    // Without FEAT_LPA2 the DS bit of either TCR is RES0.
    for tcr in [TCR_EL1, TCR_EL2] {
        assert_eq!(round_trip(&mut machine, tcr, 1 << 59 | 0x19), 0x19);
    }
}

#[test]
fn special_purpose_and_coprocessor_registers_keep_only_their_fields() {
    const NZCV: Encoding = Encoding::new(3, 3, 4, 2, 0);
    const DAIF: Encoding = Encoding::new(3, 3, 4, 2, 1);
    const CURRENT_EL: Encoding = Encoding::new(3, 0, 4, 2, 2);
    const FPSR: Encoding = Encoding::new(3, 3, 4, 4, 1);
    const CPACR_EL1: Encoding = Encoding::new(3, 0, 1, 0, 2);
    const CPTR_EL2: Encoding = Encoding::new(3, 4, 1, 1, 2);
    let mut machine = machine(&[]);
    let round_trip_value = |machine: &mut TestMachine, encoding: Encoding, value: u64| {
        machine.cpu.pstate = Pstate::reset(2);
        machine.cpu.state.x[0] = value;
        machine
            .system_instruction(access(false, encoding, 0))
            .unwrap();
        machine
            .system_instruction(access(true, encoding, 1))
            .unwrap();
        machine.cpu.state.x[1]
    };
    let round_trip = |machine: &mut TestMachine, el: u8, encoding: Encoding| {
        machine.cpu.pstate = Pstate::reset(el);
        machine.cpu.pc = BASE;
        machine.cpu.state.x[0] = u64::MAX;
        machine
            .system_instruction(access(false, encoding, 0))
            .unwrap();
        machine.cpu.state.x[1] = 0x5a5a;
        machine
            .system_instruction(access(true, encoding, 1))
            .unwrap();
        assert_eq!(machine.cpu.pc, BASE + 8);
        machine.cpu.state.x[1]
    };
    // CPTR_EL2 has its E2H layout: FPEN, TAM (with FEAT_AMUv1, from the
    // M4) and TCPAC; CPACR_EL1 keeps FPEN alone. With FEAT_SME (from the
    // M4) both also hold SMEN.
    let chip = machine.profile.chip;
    let smen = if Feature::FEAT_SME.on(chip) {
        0b11 << 24
    } else {
        0
    };
    let tam = if Feature::FEAT_AMUv1.on(chip) {
        1 << 30
    } else {
        0
    };
    assert_eq!(
        round_trip(&mut machine, 2, CPTR_EL2),
        0x8030_0000 | tam | smen
    );
    machine.cpu.registers.set(Register::CPTR_EL2, 0, 0b11 << 20);
    machine.derived_changed();
    assert_eq!(round_trip(&mut machine, 1, CPACR_EL1), 0x0030_0000 | smen);
    // NZCV, DAIF and FPSR hold their flags, masks and cumulative bits.
    assert_eq!(round_trip(&mut machine, 1, NZCV), 0xf000_0000);
    assert_eq!(machine.cpu.state.nzcv, 0xf000_0000);
    assert_eq!(round_trip(&mut machine, 1, DAIF), 0x3c0);
    assert_eq!(machine.cpu.pstate.daif, 0xf);
    assert_eq!(round_trip(&mut machine, 1, FPSR), 0x0800_009f);
    // CurrentEL reads the EL at bits 3:2.
    for el in [1, 2] {
        machine.cpu.pstate = Pstate::reset(el);
        machine
            .system_instruction(access(true, CURRENT_EL, 2))
            .unwrap();
        assert_eq!(machine.cpu.state.x[2], u64::from(el) << 2);
    }
    // HCR_EL2.BSU and SWIO are stored (one PE runs at a time and no cache
    // is visible, so neither has an effect); bits 39:38 are RES0 and RW
    // reads as one.
    const HCR_EL2: Encoding = Encoding::new(3, 4, 1, 1, 0);
    let hcr = 0b11 << 10 | 1 << 1 | 1 << 34;
    assert_eq!(
        round_trip_value(&mut machine, HCR_EL2, hcr | 0b11 << 38),
        hcr | 1 << 31
    );
    machine.cpu.registers.set(Register::HCR_EL2, 0, 1 << 34);
    // CPTR_EL2.TCPAC traps EL1 CPACR_EL1 accesses to EL2.
    machine
        .cpu
        .registers
        .set(Register::CPTR_EL2, 0, 1 << 31 | 0b11 << 20);
    let esr = el2_trap(&mut machine, access(true, CPACR_EL1, 1));
    assert_eq!(esr >> 26, 0x18);
}

#[test]
fn pstate_fields_vector_bases_and_context_ids_keep_only_their_bits() {
    const SPSEL: Encoding = Encoding::new(3, 0, 4, 2, 0);
    const PAN: Encoding = Encoding::new(3, 0, 4, 2, 3);
    const SSBS: Encoding = Encoding::new(3, 3, 4, 2, 6);
    const VBAR_EL1: Encoding = Encoding::new(3, 0, 12, 0, 0);
    const VBAR_EL2: Encoding = Encoding::new(3, 4, 12, 0, 0);
    const CONTEXTIDR_EL1: Encoding = Encoding::new(3, 0, 13, 0, 1);
    const CONTEXTIDR_EL2: Encoding = Encoding::new(3, 4, 13, 0, 1);
    const SCTLR_EL2: Encoding = Encoding::new(3, 4, 1, 0, 0);
    let mut machine = machine(&[]);
    let write_read = |machine: &mut TestMachine, el: u8, encoding: Encoding, value: u64| {
        machine.cpu.pstate = Pstate::reset(el);
        machine.cpu.pc = BASE;
        machine.cpu.state.x[0] = value;
        machine
            .system_instruction(access(false, encoding, 0))
            .unwrap();
        machine
            .system_instruction(access(true, encoding, 1))
            .unwrap();
        assert_eq!(machine.cpu.pc, BASE + 8, "{encoding:?}");
        machine.cpu.state.x[1]
    };
    // PSTATE.SP, PAN and SSBS at bits 0, 22 and 12.
    assert_eq!(write_read(&mut machine, 1, SPSEL, u64::MAX), 1);
    assert_eq!(write_read(&mut machine, 1, PAN, u64::MAX), 1 << 22);
    assert!(machine.cpu.pstate.pan);
    if Feature::FEAT_SSBS.on(machine.profile.chip) {
        assert_eq!(write_read(&mut machine, 1, SSBS, u64::MAX), 1 << 12);
        assert!(machine.cpu.pstate.ssbs);
    } else {
        // Apple clang's M4, M5 and M6 have no FEAT_SSBS: UNDEFINED.
        machine.cpu.pstate = Pstate::reset(1);
        machine.system_instruction(access(true, SSBS, 1)).unwrap();
        assert_eq!(machine.cpu.read(Register::ESR_EL1), 0x0200_0000);
    }
    // Vector bases are 2 KiB aligned; CONTEXTIDR holds a 32-bit PROCID.
    assert_eq!(write_read(&mut machine, 1, VBAR_EL1, u64::MAX), !0x7ff);
    assert_eq!(write_read(&mut machine, 2, VBAR_EL2, u64::MAX), !0x7ff);
    assert_eq!(
        write_read(&mut machine, 1, CONTEXTIDR_EL1, u64::MAX),
        0xffff_ffff
    );
    assert_eq!(
        write_read(&mut machine, 2, CONTEXTIDR_EL2, u64::MAX),
        0xffff_ffff
    );
    // SCTLR bits 17, 24, 25 and 53:50 are RES0 in both registers.
    let reserved = 1 << 17 | 3 << 24 | 0xf << 50;
    assert_eq!(
        write_read(&mut machine, 1, SCTLR_EL1, reserved) & reserved,
        0
    );
    assert_eq!(
        write_read(&mut machine, 2, SCTLR_EL2, reserved) & reserved,
        0
    );
    // HCR_EL2.TWE traps EL1 WFE to EL2 when no event is pending.
    machine.cpu.event = false;
    machine.cpu.registers.set(Register::HCR_EL2, 0, 1 << 14);
    let esr = el2_trap(&mut machine, 0xd503_205f);
    assert_eq!(esr, 0x01 << 26 | 1 << 25 | 1 << 24 | 0xe << 20 | 1);
}

#[test]
fn whole_register_state_holds_what_software_writes() {
    const STATE: &[(u8, Encoding)] = &[
        (1, Encoding::new(3, 3, 13, 0, 2)), // TPIDR_EL0
        (1, Encoding::new(3, 3, 13, 0, 3)), // TPIDRRO_EL0
        (1, Encoding::new(3, 0, 13, 0, 4)), // TPIDR_EL1
        (2, Encoding::new(3, 4, 13, 0, 2)), // TPIDR_EL2
        (1, Encoding::new(3, 0, 4, 0, 1)),  // ELR_EL1
        (2, Encoding::new(3, 4, 4, 0, 1)),  // ELR_EL2
        (1, Encoding::new(3, 0, 6, 0, 0)),  // FAR_EL1
        (2, Encoding::new(3, 4, 6, 0, 0)),  // FAR_EL2
        (1, Encoding::new(3, 0, 4, 1, 0)),  // SP_EL0
        (2, Encoding::new(3, 4, 4, 1, 0)),  // SP_EL1
        (2, Encoding::new(3, 3, 13, 0, 7)), // SCXTNUM_EL0
        (2, Encoding::new(3, 4, 13, 0, 7)), // SCXTNUM_EL2
    ];
    let mut machine = machine(&[]);
    // HCR_EL2.APK lets EL1 reach the keys.
    machine.cpu.registers.set(Register::HCR_EL2, 0, 1 << 40);
    let value = 0xfedc_ba98_7654_3210;
    // The PAC keys: APIA and APIB (CRm 1), APDA and APDB (CRm 2) and
    // APGA (CRm 3), Lo then Hi.
    let keys = [(1, 0..4), (2, 0..4), (3, 0..2)]
        .into_iter()
        .flat_map(|(crm, op2s)| op2s.map(move |op2| (1, Encoding::new(3, 0, 2, crm, op2))));
    for (el, encoding) in STATE.iter().copied().chain(keys) {
        machine.cpu.pstate = Pstate::reset(el);
        machine.cpu.pc = BASE;
        machine.cpu.state.x[0] = value;
        machine
            .system_instruction(access(false, encoding, 0))
            .unwrap();
        machine.cpu.state.x[1] = 0;
        machine
            .system_instruction(access(true, encoding, 1))
            .unwrap();
        assert_eq!(
            (machine.cpu.state.x[1], machine.cpu.pc),
            (value, BASE + 8),
            "{encoding:?}"
        );
    }
    // HSTR_EL2, which traps AArch32 EL1, is RES0; MVFR0-2, REVIDR_EL1 and
    // the debug-state DLR_EL0 and DSPSR_EL0 behave as the profile says:
    // AArch32 media features and revisions read 0, and outside Debug
    // state DLR and DSPSR are UNDEFINED.
    machine.cpu.pstate = Pstate::reset(2);
    machine.cpu.state.x[0] = u64::MAX;
    let hstr = Encoding::new(3, 4, 1, 1, 3);
    machine.system_instruction(access(false, hstr, 0)).unwrap();
    machine.system_instruction(access(true, hstr, 1)).unwrap();
    assert_eq!(machine.cpu.state.x[1], 0);
    for op2 in 0..3 {
        machine.cpu.pstate = Pstate::reset(1);
        machine.cpu.state.x[1] = 1;
        machine
            .system_instruction(access(true, Encoding::new(3, 0, 0, 3, op2), 1))
            .unwrap();
        assert_eq!(machine.cpu.state.x[1], 0, "MVFR{op2}_EL1");
    }
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    for encoding in [Encoding::new(3, 3, 4, 5, 0), Encoding::new(3, 3, 4, 5, 1)] {
        machine.cpu.pstate = Pstate::reset(1);
        machine
            .system_instruction(access(true, encoding, 1))
            .unwrap();
        assert_eq!(
            machine.cpu.read(Register::ESR_EL1),
            0x0200_0000,
            "{encoding:?}"
        );
    }
}

#[test]
fn replacing_the_thread_pointer_suspends_the_thread_it_named() {
    const TPIDR_EL1: Encoding = Encoding::new(3, 0, 13, 0, 4);
    let mut machine = machine(&[]);
    let switch = |machine: &mut TestMachine, next: u64, frame: u64| {
        machine.cpu.pstate = Pstate::reset(1);
        machine.cpu.pc = BASE;
        machine.cpu.state.x[0] = next;
        machine.cpu.state.x[29] = frame;
        machine.cpu.state.x[30] = frame + 4;
        machine
            .system_instruction(access(false, TPIDR_EL1, 0))
            .unwrap();
    };
    // The first write and a rewrite leave no suspension to timestamp.
    let reads = machine.host.counter_reads;
    switch(&mut machine, 0x100, 0);
    switch(&mut machine, 0x100, 0);
    assert!(machine.suspended().is_empty());
    assert_eq!(machine.host.counter_reads, reads);
    machine.host.now = 101;
    switch(&mut machine, 0x200, 0x8000);
    assert_eq!(machine.host.counter_reads, reads + 1);
    let suspension = machine.suspended()[&0x100];
    assert_eq!(suspension.at, 101);
    assert_eq!(
        (suspension.pc, suspension.frame, suspension.link),
        (BASE, 0x8000, 0x8004)
    );
    // Running again ends its suspension.
    switch(&mut machine, 0x100, 0x9000);
    assert_eq!(
        machine.suspended().keys().copied().collect::<Vec<_>>(),
        [0x200]
    );
}

#[test]
fn aarch32_banked_spsrs_are_res0_at_el2_and_undefined_at_el1() {
    let mut machine = machine(&[]);
    // Without AArch32 at EL1, SPSR_irq, SPSR_abt, SPSR_und and SPSR_fiq
    // are RES0 from EL2.
    for op2 in 0..4 {
        let spsr = Encoding::new(3, 4, 4, 3, op2);
        machine.cpu.pstate = Pstate::reset(2);
        machine.cpu.pc = BASE;
        machine.cpu.state.x[0] = u64::MAX;
        machine.system_instruction(access(false, spsr, 0)).unwrap();
        machine.cpu.state.x[1] = u64::MAX;
        machine.system_instruction(access(true, spsr, 1)).unwrap();
        assert_eq!(
            (machine.cpu.state.x[1], machine.cpu.pc),
            (0, BASE + 8),
            "{op2}"
        );
        machine.cpu.pstate = Pstate::reset(1);
        machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
        machine.system_instruction(access(true, spsr, 1)).unwrap();
        assert_eq!(machine.cpu.read(Register::ESR_EL1), 0x0200_0000, "{op2}");
    }
}

#[test]
fn ecv_self_synchronized_counts_and_el1_virtual_counter_traps() {
    const CNTPCTSS_EL0: Encoding = Encoding::new(3, 3, 14, 0, 5);
    const CNTVCTSS_EL0: Encoding = Encoding::new(3, 3, 14, 0, 6);
    const CNTV_CTL_EL0: Encoding = Encoding::new(3, 3, 14, 3, 1);
    const EL1PCTEN: u64 = 1 << 10;
    let mut machine = machine_on(one_timer_bank(), &[]);
    let kernel = machine.cpu.pstate;
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x4_0000);
    machine.cpu.registers.set(Register::CNTVOFF_EL2, 0, 100);
    machine
        .cpu
        .registers
        .set(Register::CNTHCTL_EL2, 0, EL1PCTEN);
    machine.host.now = 1000;
    if !Feature::FEAT_ECV.on(machine.profile.chip) {
        // Without FEAT_ECV (the M1) the self-synchronized counts are
        // UNDEFINED.
        for encoding in [CNTPCTSS_EL0, CNTVCTSS_EL0] {
            machine.cpu.pstate = kernel;
            machine.cpu.registers.set(Register::ESR_EL1, 0, 0);
            machine
                .system_instruction(access(true, encoding, 1))
                .unwrap();
            assert_eq!(machine.cpu.read(Register::ESR_EL1), 0x0200_0000);
        }
        return;
    }
    machine
        .system_instruction(access(true, CNTPCTSS_EL0, 1))
        .unwrap();
    machine
        .system_instruction(access(true, CNTVCTSS_EL0, 2))
        .unwrap();
    assert_eq!(
        (machine.cpu.state.x[1], machine.cpu.state.x[2]),
        (1000, 900)
    );
    // CNTHCTL_EL2.EL1TVCT traps EL1 reads of the virtual count and
    // EL1TVT the virtual timer registers to EL2 with EC 0x18.
    for (bit, encoding) in [(14, CNTVCTSS_EL0), (14, CNTVCT_EL0), (13, CNTV_CTL_EL0)] {
        machine
            .cpu
            .registers
            .set(Register::CNTHCTL_EL2, 0, EL1PCTEN | 1 << bit);
        machine.cpu.pstate = kernel;
        machine.cpu.pc = BASE;
        machine
            .system_instruction(access(true, encoding, 3))
            .unwrap();
        assert_eq!(
            (machine.cpu.pstate.el, machine.cpu.pc),
            (2, 0x4_0400),
            "{encoding:?}"
        );
        assert_eq!(machine.cpu.read(Register::ESR_EL2) >> 26, 0x18);
    }
}

/// For every generated accessor at every index, run at EL1 on a fresh
/// machine that `configure` prepares, whether it traps to EL2 with EC
/// 0x18.
fn el1_traps_to_el2(
    configure: impl FnOnce(&mut super::Machine<super::super::tests::FakeHost, super::super::NoDevices>),
) -> alloc::vec::Vec<bool> {
    use super::{ACCESSORS, Access, Cell, Effect};
    let mut machine = machine(&[]);
    configure(&mut machine);
    ACCESSORS
        .iter()
        .flat_map(|accessor| (0..1u64 << accessor.index.len()).map(move |m| (accessor, m)))
        .map(|(accessor, m)| {
            let mut context = Access {
                machine: &mut machine,
                rt: 0,
                now: None,
                result: None,
                translation: Translation::Unchanged,
                regime: false,
                reset: false,
                code: Code::Unchanged,
                refusal: Cell::new(None),
                plain: Cell::new(None),
                bank: super::Bank::Arm,
            };
            let effect = match (accessor.run)(&mut context, m) {
                Ok(effect) | Err(effect) => effect,
            };
            matches!(effect, Effect::Trap { el: 2, ec: 0x18 })
        })
        .collect()
}

/// HCR_EL2's APK, FIEN and EnSCXT set, so that their coarse traps do not
/// pre-empt the controls under test.
const HCR_ENABLES: u64 = 1 << 40 | 1 << 47 | 1 << 53;

/// Every FEAT_FGT control the profile implements changes whether some
/// EL1 access traps to EL2 with EC 0x18. HFGITR_EL2.ERET, SVC_EL0 and
/// SVC_EL1 trap instructions without accessors and are tested in
/// engine.rs. HAFGRTR_EL2 is left out: it applies only with
/// FEAT_AMUv1, and the activity monitor test covers it.
#[test]
fn every_fine_grained_trap_control_traps_an_el1_access() {
    const CONTROLS: [Register; 5] = [
        Register::HFGRTR_EL2,
        Register::HFGWTR_EL2,
        Register::HFGITR_EL2,
        Register::HDFGRTR_EL2,
        Register::HDFGWTR_EL2,
    ];
    const INSTRUCTIONS: u64 = 0b111 << 51;
    let enable = |machine: &mut super::Machine<_, _>| {
        let hcr = machine.cpu.read(Register::HCR_EL2) | HCR_ENABLES;
        machine.cpu.registers.set(Register::HCR_EL2, 0, hcr);
    };
    let chip = crate::machine::test_chip();
    if !Feature::FEAT_FGT.on(chip) {
        // Without FEAT_FGT (the M1) the controls are UNDEFINED at EL2.
        const HFGRTR_EL2: Encoding = Encoding::new(3, 4, 1, 1, 4);
        let mut machine = machine(&[]);
        machine.cpu.pstate = Pstate::reset(2);
        machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x4_0000);
        machine
            .system_instruction(access(true, HFGRTR_EL2, 1))
            .unwrap();
        assert_eq!(machine.cpu.read(Register::ESR_EL2), 0x0200_0000);
        return;
    }
    let baseline = el1_traps_to_el2(enable);
    let mut inert = alloc::vec::Vec::new();
    for register in CONTROLS {
        let mut fields = !(register.res0(chip) | register.res1(chip));
        if register == Register::HFGITR_EL2 {
            fields &= !INSTRUCTIONS;
        }
        for bit in (0..64).filter(|bit| fields >> bit & 1 != 0) {
            let traps = el1_traps_to_el2(|machine| {
                enable(machine);
                let value = machine.cpu.read(register) ^ 1 << bit;
                machine.cpu.registers.set(register, 0, value);
            });
            if traps == baseline {
                inert.push(alloc::format!("{}[{bit}]", register.name()));
            }
        }
    }
    assert!(inert.is_empty(), "controls that trap nothing: {inert:?}");
}

/// Every HCR_EL2 control that traps System register and System
/// instruction accesses changes whether some EL1 access traps to EL2:
/// TID1-4, TACR, TSW, TPCP, TPU, TTLB, TVM, TRVM, TDZ, TLOR and TERR,
/// FEAT_EVT's TICAB and TOCU, FEAT_EVT2's TTLBIS and TTLBOS, and APK,
/// FIEN and EnSCXT, which trap while clear.
#[test]
fn every_hcr_el2_access_trap_control_traps_an_el1_access() {
    const CONTROLS: [(&str, u32); 21] = [
        ("TID1", 16),
        ("TID2", 17),
        ("TID3", 18),
        ("TACR", 21),
        ("TSW", 22),
        ("TPCP", 23),
        ("TPU", 24),
        ("TTLB", 25),
        ("TVM", 26),
        ("TDZ", 28),
        ("TRVM", 30),
        ("TLOR", 35),
        ("TERR", 36),
        ("APK", 40),
        ("FIEN", 47),
        ("TID4", 49),
        ("TICAB", 50),
        ("TOCU", 52),
        ("EnSCXT", 53),
        ("TTLBIS", 54),
        ("TTLBOS", 55),
    ];
    let hcr = |value: u64| {
        move |machine: &mut super::Machine<_, _>| {
            machine.cpu.registers.set(Register::HCR_EL2, 0, value);
        }
    };
    let baseline = el1_traps_to_el2(hcr(HCR_ENABLES));
    let inert: alloc::vec::Vec<_> = CONTROLS
        .into_iter()
        .filter(|&(_, bit)| el1_traps_to_el2(hcr(HCR_ENABLES ^ 1 << bit)) == baseline)
        .map(|(name, _)| name)
        .collect();
    assert!(inert.is_empty(), "controls that trap nothing: {inert:?}");
}

#[test]
fn reserved_and_absent_id_registers_follow_feat_idst() {
    const RESERVED: Encoding = Encoding::new(3, 0, 0, 4, 3);
    const GMID_EL1: Encoding = Encoding::new(3, 1, 0, 0, 4);
    const HCR_TID3: u64 = 1 << 18;
    let mut machine = machine(&[]);
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x4_0000);
    let kernel = machine.cpu.pstate;
    let class = |machine: &super::Machine<_, _>, register| machine.cpu.read(register) >> 26;
    // RES0 from EL1.
    machine.cpu.state.x[3] = 1;
    machine
        .system_instruction(access(true, RESERVED, 3))
        .unwrap();
    assert_eq!((machine.cpu.state.x[3], machine.cpu.pc), (0, BASE + 4));
    // HCR_EL2.TID3 traps EL1 reads to EL2.
    machine.cpu.registers.set(Register::HCR_EL2, 0, HCR_TID3);
    machine.cpu.pc = BASE;
    machine
        .system_instruction(access(true, RESERVED, 3))
        .unwrap();
    assert_eq!((machine.cpu.pstate.el, machine.cpu.pc), (2, 0x4_0400));
    assert_eq!(class(&machine, Register::ESR_EL2), 0x18);
    machine.cpu.registers.set(Register::HCR_EL2, 0, 0);
    // FEAT_IDST: an EL0 read traps to EL1 with EC 0x18.
    machine.cpu.pstate = Pstate {
        el: 0,
        sp: false,
        ..kernel
    };
    machine.cpu.pc = BASE;
    machine
        .system_instruction(access(true, RESERVED, 3))
        .unwrap();
    assert_eq!((machine.cpu.pstate.el, machine.cpu.pc), (1, 0x1_0400));
    assert_eq!(class(&machine, Register::ESR_EL1), 0x18);
    // A write is UNDEFINED.
    machine.cpu.pstate = kernel;
    machine.cpu.pc = BASE;
    machine
        .system_instruction(access(false, RESERVED, 3))
        .unwrap();
    assert_eq!(
        (machine.cpu.pc, class(&machine, Register::ESR_EL1)),
        (0x1_0200, 0)
    );
    // GMID_EL1 without FEAT_MTE2 (before the M5) is an unimplemented ID
    // register: EC 0x18 at the current EL.
    if !Feature::FEAT_MTE2.on(machine.profile.chip) {
        machine.cpu.pstate = kernel;
        machine.cpu.pc = BASE;
        machine
            .system_instruction(access(true, GMID_EL1, 3))
            .unwrap();
        assert_eq!(
            (machine.cpu.pc, class(&machine, Register::ESR_EL1)),
            (0x1_0200, 0x18)
        );
    }
}

#[test]
fn cntfrq_reads_the_profile_counter_frequency() {
    const CNTFRQ_EL0: Encoding = Encoding::new(3, 3, 14, 0, 0);
    let mut machine = machine(&[]);
    machine
        .system_instruction(access(true, CNTFRQ_EL0, 5))
        .unwrap();
    assert_eq!(machine.cpu.state.x[5], machine.profile.counter_frequency);
}

#[test]
fn identification_reads_come_from_the_profile_or_refuse() {
    let mut machine = machine(&[]);
    // Without FEAT_AA32 the AArch32 ID registers are entirely reserved,
    // so they read their one value with no profile entry.
    for (index, encoding) in [Encoding::new(3, 0, 0, 2, 0), Encoding::new(3, 0, 0, 3, 0)]
        .into_iter()
        .enumerate()
    {
        machine.cpu.state.x[3] = 1;
        machine
            .system_instruction(access(true, encoding, 3))
            .unwrap_or_else(|stop| panic!("{index}: {stop:?}"));
        assert_eq!(machine.cpu.state.x[3], 0);
    }
    // The test profile has no MIDR_EL1: the read, through VPIDR_EL2 at
    // EL1, is reported once and reads as storage.
    machine
        .system_instruction(access(true, MIDR_EL1, 0))
        .unwrap();
    assert_eq!(machine.take_unmodeled().len(), 1);
    // A value written to VPIDR_EL2 at EL2 is what EL1 then reads.
    machine.cpu.pstate = Pstate::reset(2);
    machine.cpu.state.x[1] = 0x611f_0320;
    machine
        .system_instruction(access(false, Encoding::new(3, 4, 0, 0, 0), 1))
        .unwrap();
    machine.cpu.pstate = Pstate::reset(1);
    machine
        .system_instruction(access(true, MIDR_EL1, 2))
        .unwrap();
    assert_eq!(machine.cpu.state.x[2], 0x611f_0320);
}

#[test]
fn interrupts_route_by_hcr_el2_and_virtual_interrupts_reach_el1() {
    let mut machine = machine(&[]);
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x4_0000);
    // A virtual IRQ (HCR_EL2.IMO and VI) is taken at EL1 under PSTATE.I.
    machine
        .cpu
        .registers
        .set(Register::HCR_EL2, 0, 1 << 4 | 1 << 7);
    machine.cpu.pstate = Pstate {
        daif: 0,
        ..Pstate::reset(1)
    };
    machine.derived_changed();
    assert!(machine.deliver_interrupts());
    assert_eq!((machine.cpu.pstate.el, machine.cpu.pc), (1, 0x1_0280));
    // Physical IRQs go to EL2 while IMO is set.
    assert_eq!(machine.cpu.interrupt_target(Vector::Irq), 2);
    // ISR_EL1 shows the virtual lines while they are routed to EL2: VI,
    // VF and VSE as I, F and A at bits 7, 6 and 8.
    const ISR_EL1: Encoding = Encoding::new(3, 0, 12, 1, 0);
    for (hcr, isr) in [
        (1 << 4 | 1 << 7, 1 << 7),
        (1 << 3 | 1 << 6, 1 << 6),
        (1 << 5 | 1 << 8, 1 << 8),
        (1 << 4 | 1 << 3 | 1 << 5, 0),
    ] {
        machine.cpu.registers.set(Register::HCR_EL2, 0, hcr);
        machine.cpu.pstate = Pstate::reset(1);
        machine
            .system_instruction(access(true, ISR_EL1, 1))
            .unwrap();
        assert_eq!(machine.cpu.state.x[1], isr, "{hcr:#x}");
    }
}

#[test]
fn the_vgic_signals_a_virtual_irq_that_el1_acknowledges_through_icc_iar1() {
    const ICH_HCR_EL2: Encoding = Encoding::new(3, 4, 12, 11, 0);
    const ICH_VMCR_EL2: Encoding = Encoding::new(3, 4, 12, 11, 7);
    const ICH_LR0_EL2: Encoding = Encoding::new(3, 4, 12, 12, 0);
    const ICH_MISR_EL2: Encoding = Encoding::new(3, 4, 12, 11, 2);
    const ICC_IAR1_EL1: Encoding = Encoding::new(3, 0, 12, 12, 0);
    const ICC_EOIR1_EL1: Encoding = Encoding::new(3, 0, 12, 12, 1);
    const ICC_SGI1R_EL1: Encoding = Encoding::new(3, 0, 12, 11, 5);
    const HFGRTR_ICC_IGRPENN: u64 = 1 << 39;
    const ICC_IGRPEN1_EL1: Encoding = Encoding::new(3, 0, 12, 12, 7);
    let mut machine = machine(&[]);
    // No System register interface to a physical GIC: GIC reads 0.
    assert_eq!(machine.cpu.field(Register::ID_AA64PFR0_EL1, 24, 4), 0);
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x4_0000);
    let at = |machine: &mut TestMachine, el: u8, read: bool, encoding, value| {
        machine.cpu.pstate = Pstate::reset(el);
        machine.cpu.pc = BASE;
        machine.cpu.state.x[1] = value;
        machine
            .system_instruction(access(read, encoding, 1))
            .unwrap();
        machine.cpu.state.x[1]
    };
    // EL2 queues vINTID 27, group 1, priority 0x80, and enables the
    // interface with group 1 on and no priority mask.
    at(
        &mut machine,
        2,
        false,
        ICH_LR0_EL2,
        1 << 62 | 1 << 60 | 0x80 << 48 | 27,
    );
    at(&mut machine, 2, false, ICH_VMCR_EL2, 0xff << 24 | 1 << 1);
    at(&mut machine, 2, false, ICH_HCR_EL2, 1);
    // With HCR_EL2.IMO the guest at EL1 takes it as a virtual IRQ.
    machine.cpu.registers.set(Register::HCR_EL2, 0, 1 << 4);
    machine.cpu.pstate = Pstate {
        daif: 0,
        ..Pstate::reset(1)
    };
    machine.cpu.pc = BASE;
    machine.derived_changed();
    assert!(machine.deliver_interrupts());
    assert_eq!((machine.cpu.pstate.el, machine.cpu.pc), (1, 0x1_0280));
    // ICC_IAR1_EL1 reaches ICV_IAR1_EL1, which acknowledges it, and
    // ICC_EOIR1_EL1 completes it.
    assert_eq!(at(&mut machine, 1, true, ICC_IAR1_EL1, 0), 27);
    machine.cpu.pstate.daif = 0;
    assert!(!machine.deliver_interrupts(), "active, nothing pending");
    at(&mut machine, 1, false, ICC_EOIR1_EL1, 27);
    assert_eq!(at(&mut machine, 2, true, ICH_LR0_EL2, 0) >> 62, 0);
    // The SGI registers trap to EL2 with EC 0x18.
    at(&mut machine, 1, false, ICC_SGI1R_EL1, 0);
    assert_eq!(machine.cpu.pstate.el, 2);
    assert_eq!(machine.cpu.read(Register::ESR_EL2) >> 26, 0x18);
    // HFGRTR_EL2.ICC_IGRPENn_EL1 needs FEAT_GICv3, so it is RES0 and
    // the group enable reads VENG1 untrapped.
    machine
        .cpu
        .registers
        .set(Register::HFGRTR_EL2, 0, HFGRTR_ICC_IGRPENN);
    assert_eq!(machine.cpu.read(Register::HFGRTR_EL2), 0);
    let enable = at(&mut machine, 1, true, ICC_IGRPEN1_EL1, 0);
    assert_eq!((machine.cpu.pstate.el, enable), (1, 1));
    // Underflow with no valid list register raises the maintenance
    // interrupt on the core's IRQ, which IMO routes to EL2.
    at(&mut machine, 2, false, ICH_HCR_EL2, 1 | 1 << 1);
    assert_eq!(at(&mut machine, 2, true, ICH_MISR_EL2, 0), 1 << 1);
    machine.cpu.pstate = Pstate {
        daif: 0,
        ..Pstate::reset(1)
    };
    machine.cpu.pc = BASE;
    assert!(machine.deliver_interrupts());
    assert_eq!((machine.cpu.pstate.el, machine.cpu.pc), (2, 0x4_0480));
    // At EL2 the physical CPU interface is absent: UNDEFINED.
    machine.cpu.pstate = Pstate::reset(2);
    machine.cpu.pc = BASE;
    machine
        .system_instruction(access(true, ICC_IAR1_EL1, 1))
        .unwrap();
    assert_eq!(machine.cpu.read(Register::ESR_EL2) >> 26, 0);
}

#[test]
fn sctlr_endianness_bits_are_res0_and_alignment_follows_a_naa() {
    let mut machine = machine(&[]);
    // SCTLR_EL1.EE and E0E are RES0 without FEAT_MixedEnd.
    machine.cpu.state.x[0] = 3 << 24;
    machine.system_instruction(0xd518_1000).unwrap(); // msr sctlr_el1, x0
    assert_eq!(machine.cpu.read(Register::SCTLR_EL1) & 3 << 24, 0);
    assert!(!machine.memory_checks().alignment && machine.memory_checks().ordered_granule);
    // A and nAA select blocks with the matching native checks.
    machine.cpu.pc = BASE;
    machine.cpu.state.x[0] = 1 << 1 | 1 << 6;
    machine.system_instruction(0xd518_1000).unwrap();
    assert_eq!(machine.cpu.read(Register::SCTLR_EL1) & 0x42, 0x42);
    assert!(machine.memory_checks().alignment && !machine.memory_checks().ordered_granule);
}

#[test]
fn mpam_enable_is_mpam2s_and_the_vpm_and_el3_registers_are_undefined() {
    const MPAM0_EL1: Encoding = Encoding::new(3, 0, 10, 5, 1);
    const MPAM1_EL1: Encoding = Encoding::new(3, 0, 10, 5, 0);
    const MPAM2_EL2: Encoding = Encoding::new(3, 4, 10, 5, 0);
    let mut machine = machine(&[]);
    // The profile's MPAMIDR_EL1: PARTID 0 and PMG 0 only, no HCR.
    machine
        .profile_missing
        .retain(|register| *register != Register::MPAMIDR_EL1);
    machine.cpu.registers.set(Register::MPAMIDR_EL1, 0, 0);
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x4_0000);
    let access_at = |machine: &mut TestMachine, el: u8, read: bool, encoding, value| {
        machine.cpu.pstate = Pstate::reset(el);
        machine.cpu.pc = BASE;
        machine.cpu.state.x[1] = value;
        machine
            .system_instruction(access(read, encoding, 1))
            .unwrap();
        (machine.cpu.pc, machine.cpu.state.x[1])
    };
    // EL2 sets MPAMEN and clears the reset traps on MPAM0_EL1 and
    // MPAM1_EL1; without EL3, MPAM1_EL1.MPAMEN reads MPAM2_EL2's and
    // ignores writes.
    let partitions = 0x0000_0506_0007_0008;
    for (enable, expected) in [(1u64 << 63, 1u64 << 63), (0, 0)] {
        access_at(&mut machine, 2, false, MPAM2_EL2, enable | partitions);
        assert_eq!(
            access_at(&mut machine, 2, true, MPAM2_EL2, 0),
            (BASE + 4, enable | partitions)
        );
        access_at(
            &mut machine,
            1,
            false,
            MPAM1_EL1,
            !enable & 1 << 63 | partitions,
        );
        assert_eq!(
            access_at(&mut machine, 1, true, MPAM1_EL1, 0),
            (BASE + 4, expected | partitions)
        );
    }
    // MPAM0_EL1 has no MPAMEN.
    access_at(&mut machine, 1, false, MPAM0_EL1, u64::MAX);
    assert_eq!(
        access_at(&mut machine, 1, true, MPAM0_EL1, 0).1,
        0x0000_ffff_ffff_ffff
    );
    // MPAMIDR_EL1.HAS_HCR is 0, so MPAMHCR_EL2, MPAMVPM<n>_EL2 and
    // MPAMVPMV_EL2 are UNDEFINED; without EL3 so is MPAM3_EL3.
    for encoding in [
        Encoding::new(3, 4, 10, 4, 0),
        Encoding::new(3, 4, 10, 6, 0),
        Encoding::new(3, 4, 10, 6, 7),
        Encoding::new(3, 4, 10, 4, 1),
        Encoding::new(3, 6, 10, 5, 0),
    ] {
        assert_eq!(
            access_at(&mut machine, 2, true, encoding, 0).0,
            0x4_0200,
            "{encoding:?}"
        );
        assert_eq!(machine.cpu.read(Register::ESR_EL2), 0x0200_0000);
    }
}

#[test]
fn disr_el1_is_stored_and_reads_vdisr_el2_while_amo_routes_serrors() {
    const DISR_EL1: Encoding = Encoding::new(3, 0, 12, 1, 1);
    let mut machine = machine(&[]);
    let access_at = |machine: &mut TestMachine, el: u8, read: bool, value| {
        machine.cpu.pstate = Pstate::reset(el);
        machine.cpu.state.x[1] = value;
        machine
            .system_instruction(access(read, DISR_EL1, 1))
            .unwrap();
        machine.cpu.state.x[1]
    };
    // A, IDS, AET, EA and DFSC hold what EL2 writes; the rest is RES0.
    access_at(&mut machine, 2, false, u64::MAX);
    let disr = access_at(&mut machine, 2, true, 0);
    assert_eq!(disr, !Register::DISR_EL1.res0(machine.profile.chip));
    assert_eq!(disr & (1 << 31 | 1 << 24), 1 << 31 | 1 << 24);
    // With HCR_EL2.AMO, EL1 reads and writes VDISR_EL2 instead.
    machine.cpu.registers.set(Register::HCR_EL2, 0, 1 << 5);
    machine
        .cpu
        .registers
        .set(Register::VDISR_EL2, 0, 0x8000_0011);
    assert_eq!(access_at(&mut machine, 1, true, 0), 0x8000_0011);
    access_at(&mut machine, 1, false, 0);
    assert_eq!(machine.cpu.read(Register::VDISR_EL2), 0);
    assert_eq!(machine.cpu.read(Register::DISR_EL1), disr);
    // Without AMO, EL1 reaches DISR_EL1 itself.
    machine.cpu.registers.set(Register::HCR_EL2, 0, 0);
    assert_eq!(access_at(&mut machine, 1, true, 0), disr);
}

#[test]
fn apple_actlr_el12_alias_shares_the_architectural_owner_and_refusals() {
    use crate::machine::impdef;
    const ARCHITECTURAL: Encoding = Encoding::new(3, 5, 1, 0, 1);
    const HOST: Encoding = Encoding::new(3, 0, 1, 0, 1);
    let alias = impdef::ACTLR_EL12;
    for chip in Chip::ALL {
        if !impdef::consumed(alias, chip) {
            continue;
        }
        let mut machine = machine_on(chip, &[]);
        machine.cpu.set_pstate(Pstate::reset(2));
        let read = |machine: &mut TestMachine, encoding| {
            machine.cpu.pc = BASE;
            machine
                .system_instruction(access(true, encoding, 2))
                .unwrap();
            machine.cpu.state.x[2]
        };
        let write = |machine: &mut TestMachine, encoding, value| {
            machine.cpu.pc = BASE;
            machine.cpu.state.x[1] = value;
            machine.system_instruction(access(false, encoding, 1))
        };
        // First read reaches the canonical reset value with no degraded
        // storage slot. The host's ACTLR_EL2 remains independent.
        let reset = machine.cpu.read(Register::ACTLR_EL1);
        assert_eq!(read(&mut machine, alias), reset);
        assert!(machine.take_unmodeled().is_empty());
        write(&mut machine, HOST, ACTLR_ICDSB).unwrap();
        for value in [0, ACTLR_EN_TSO, ACTLR_EN_TSO | ACTLR_ICDSB] {
            write(&mut machine, alias, value).unwrap();
            assert_eq!(read(&mut machine, ARCHITECTURAL), value);
            write(&mut machine, ARCHITECTURAL, value ^ ACTLR_EN_TSO).unwrap();
            assert_eq!(read(&mut machine, alias), value ^ ACTLR_EN_TSO);
            assert_eq!(machine.cpu.read(Register::ACTLR_EL2), ACTLR_ICDSB);
        }
        let before = machine.cpu.read(Register::ACTLR_EL1);
        // A refused setting is held without its effect, reported once.
        for refused in [1, 1 << 4, 1 << 5, 1 << 6, 1 << 63] {
            write(&mut machine, alias, refused).unwrap();
            assert_eq!(machine.cpu.read(Register::ACTLR_EL1), before);
            assert_eq!(machine.cpu.pc, BASE + 4);
        }
        assert!(!machine.take_unmodeled().is_empty());
        // XZR is an ordinary discarded destination/zero source.
        machine
            .system_instruction(access(false, alias, 31))
            .unwrap();
        assert_eq!(read(&mut machine, ARCHITECTURAL), 0);
        let original_x = machine.cpu.state.x;
        machine.system_instruction(access(true, alias, 31)).unwrap();
        assert_eq!(machine.cpu.state.x, original_x);
        let peer = machine_on(chip, &[]);
        assert_eq!(peer.cpu.read(Register::ACTLR_EL1), reset);
    }
}

#[test]
fn apple_actlr_el12_alias_preserves_lower_el_permissions_and_original_trap_iss() {
    use crate::machine::{impdef, registers::trap_iss};
    for chip in Chip::ALL {
        if !impdef::consumed(impdef::ACTLR_EL12, chip) {
            continue;
        }
        for read in [false, true] {
            for el in [0, 1] {
                for tidcp in [false, true] {
                    let mut machine = machine_on(chip, &[]);
                    machine.cpu.set_pstate(Pstate::reset(el));
                    machine.cpu.registers.set(
                        Register::HCR_EL2,
                        0,
                        if tidcp {
                            1 << crate::machine::registers::HCR_TIDCP
                        } else {
                            0
                        },
                    );
                    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
                    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x4_0000);
                    machine.cpu.state.x[1] = ACTLR_EN_TSO;
                    let before = machine.cpu.read(Register::ACTLR_EL1);
                    let word = access(read, impdef::ACTLR_EL12, 1);
                    machine.system_instruction(word).unwrap();
                    let trapped = el == 1 && tidcp;
                    let target = if trapped {
                        Register::ESR_EL2
                    } else {
                        Register::ESR_EL1
                    };
                    let syndrome = machine.cpu.read(target);
                    assert_eq!(syndrome >> 26, if trapped { 0x18 } else { 0 });
                    if trapped {
                        let original =
                            stela_compiler::aarch64::system::registers::decode(word).unwrap();
                        assert_eq!(syndrome & 0x1ff_ffff, u64::from(trap_iss(original)));
                    }
                    assert_eq!(machine.cpu.read(Register::ACTLR_EL1), before);
                }
            }
        }
    }
}

#[test]
fn implementation_defined_registers_take_the_bits_the_census_models() {
    const ACTLR_EL1: Encoding = Encoding::new(3, 0, 1, 0, 1);
    const ACTLR_EL12: Encoding = Encoding::new(3, 5, 1, 0, 1);
    const AFSR0_EL1: Encoding = Encoding::new(3, 0, 5, 1, 0);
    const AFSR1_EL12: Encoding = Encoding::new(3, 5, 5, 1, 1);
    const AMAIR_EL1: Encoding = Encoding::new(3, 0, 10, 3, 0);
    const AMAIR_EL12: Encoding = Encoding::new(3, 5, 10, 3, 0);
    const HACR_EL2: Encoding = Encoding::new(3, 4, 1, 1, 7);
    const EN_TSO: u64 = 1 << 1;
    const EN_APFLG: u64 = 1 << 4;
    const EN_AFP: u64 = 1 << 5;
    const EN_PRSV: u64 = 1 << 6;
    const ICDSB: u64 = 1 << 9;
    const IC_IVAU_ASID: u64 = 1 << 12;
    let mut machine = machine(&[]);
    machine.cpu.set_pstate(Pstate::reset(2));
    machine.cpu.registers.set(Register::HCR_EL2, 0, 1 << 34);
    let write = |machine: &mut TestMachine, encoding, value| {
        machine.cpu.state.x[1] = value;
        machine.system_instruction(access(false, encoding, 1))
    };
    let read = |machine: &mut TestMachine, encoding| {
        machine
            .system_instruction(access(true, encoding, 2))
            .unwrap();
        machine.cpu.state.x[2]
    };
    // XNU clears ACTLR bits at CPU initialization, and every register
    // takes 0.
    for encoding in [ACTLR_EL1, ACTLR_EL12, AMAIR_EL1, AMAIR_EL12, HACR_EL2] {
        write(&mut machine, encoding, 0).unwrap();
    }
    // EnTSO is held, through the host's ACTLR_EL2 and the guest's
    // ACTLR_EL1, as the context switch and the EL0 return leave it, and
    // so is the bit the icdsb context switch sets.
    write(&mut machine, ACTLR_EL1, EN_TSO).unwrap();
    write(&mut machine, ACTLR_EL12, EN_TSO | ICDSB).unwrap();
    assert_eq!(machine.cpu.read(Register::ACTLR_EL2), EN_TSO);
    assert_eq!(read(&mut machine, ACTLR_EL12), EN_TSO | ICDSB);
    write(&mut machine, ACTLR_EL1, EN_TSO | ICDSB).unwrap();
    // The M3 CPU initialization's IC IVAU ASID enable is held too.
    write(&mut machine, ACTLR_EL1, ICDSB | IC_IVAU_ASID).unwrap();
    assert_eq!(machine.cpu.read(Register::ACTLR_EL2), ICDSB | IC_IVAU_ASID);
    write(&mut machine, ACTLR_EL1, EN_TSO | ICDSB).unwrap();
    // AMAIR holds any value, through the host's AMAIR_EL2 and the
    // guest's AMAIR_EL1, as the hypervisor's save and restore needs.
    write(&mut machine, AMAIR_EL1, 0x0123_4567_89ab_cdef).unwrap();
    write(&mut machine, AMAIR_EL12, u64::MAX).unwrap();
    assert_eq!(machine.cpu.read(Register::AMAIR_EL2), 0x0123_4567_89ab_cdef);
    assert_eq!(machine.cpu.read(Register::AMAIR_EL1), u64::MAX);
    assert_eq!(read(&mut machine, AMAIR_EL1), 0x0123_4567_89ab_cdef);
    assert_eq!(read(&mut machine, AMAIR_EL12), u64::MAX);
    // EnAPFLG, EnAFP and EnPRSV, controls of features AIDR_EL1 does not
    // advertise, are reported by name, as is HACR_EL2's guest timer mask.
    for (encoding, value, reason) in [
        (ACTLR_EL1, EN_TSO | EN_APFLG, "ACTLR EnAPFLG"),
        (ACTLR_EL12, EN_TSO | EN_AFP | EN_PRSV, "ACTLR EnAFP"),
        (ACTLR_EL1, EN_PRSV, "ACTLR EnPRSV"),
        (
            HACR_EL2,
            1 << 20,
            "HACR_EL2 bit 20 masks the guest's virtual timer, whose pending FIQ register is not modeled",
        ),
    ] {
        write(&mut machine, encoding, value).unwrap();
        let reports = machine.take_unmodeled();
        assert!(
            reports
                .iter()
                .any(|requirement| format!("{requirement}").contains(reason)),
            "{encoding:?}: {reports:?}"
        );
    }
    assert_eq!(machine.cpu.read(Register::ACTLR_EL2), EN_TSO | ICDSB);
    // HACR_EL2 holds the SPTM's VM entry values.
    for value in [0x061f_0000_6100_2000, 0x041f_0000_0000_0000] {
        write(&mut machine, HACR_EL2, value).unwrap();
        assert_eq!(read(&mut machine, HACR_EL2), value);
    }
    write(&mut machine, HACR_EL2, 0).unwrap();
    // The fault status registers hold what software writes, as the
    // hypervisor's save and restore needs, and no exception records in
    // them.
    write(&mut machine, AFSR0_EL1, 0xdead_beef_0000_0001).unwrap();
    write(&mut machine, AFSR1_EL12, 0x20).unwrap();
    machine.cpu.set_pstate(Pstate::reset(1));
    machine
        .system_instruction(access(true, HACR_EL2, 2))
        .unwrap();
    assert_eq!(machine.cpu.read(Register::ESR_EL1), 0x0200_0000);
    machine.cpu.set_pstate(Pstate::reset(2));
    assert_eq!(read(&mut machine, AFSR0_EL1), 0xdead_beef_0000_0001);
    assert_eq!(read(&mut machine, AFSR1_EL12), 0x20);
    assert_eq!(machine.cpu.read(Register::AFSR0_EL2), 0xdead_beef_0000_0001);
}

#[test]
fn mpam_ras_lor_and_scxtnum_follow_their_minimal_configurations() {
    let mut machine = machine(&[]);
    // HCR_EL2.EnSCXT lets EL1 reach SCXTNUM_EL1, and clearing the reset
    // traps of MPAM2_EL2 (TRAPMPAM0EL1, TRAPMPAM1EL1) lets it reach
    // MPAM1_EL1.
    machine.cpu.registers.set(Register::HCR_EL2, 0, 1 << 53);
    machine.cpu.registers.set(Register::MPAM2_EL2, 0, 0);
    let write = |machine: &mut super::super::Machine<_, _>, encoding, value| {
        machine.cpu.state.x[1] = value;
        machine
            .system_instruction(access(false, encoding, 1))
            .unwrap();
    };
    let read = |machine: &mut super::super::Machine<_, _>, encoding| {
        machine
            .system_instruction(access(true, encoding, 2))
            .unwrap();
        machine.cpu.state.x[2]
    };
    // MPAM1_EL1 and SCXTNUM_EL1 hold what EL1 writes.
    for (encoding, value) in [
        (Encoding::new(3, 0, 10, 5, 0), 0x0001_0002_0003),
        (Encoding::new(3, 0, 13, 0, 7), 0x1234_5678),
    ] {
        write(&mut machine, encoding, value);
        assert_eq!(read(&mut machine, encoding), value, "{encoding:?}");
    }
    // With no LORegion descriptor (LORID_EL1 0), LORSA_EL1, LOREA_EL1,
    // LORN_EL1 and LORC_EL1 are RES0: RAZ/WI.
    machine
        .profile_missing
        .retain(|register| *register != Register::LORID_EL1);
    machine.cpu.registers.set(Register::LORID_EL1, 0, 0);
    for op2 in 0..4 {
        let encoding = Encoding::new(3, 0, 10, 4, op2);
        write(&mut machine, encoding, 0x8000_0001);
        assert_eq!(read(&mut machine, encoding), 0, "{encoding:?}");
    }
    // No error record: ERX* read 0 and ignore writes, whatever ERRSELR
    // selects.
    write(&mut machine, Encoding::new(3, 0, 5, 3, 1), 3);
    write(&mut machine, Encoding::new(3, 0, 5, 4, 1), u64::MAX);
    assert_eq!(read(&mut machine, Encoding::new(3, 0, 5, 4, 1)), 0);
    assert_eq!(read(&mut machine, Encoding::new(3, 0, 5, 4, 2)), 0);
}

#[test]
fn activity_monitors_count_group_zero_and_trap_as_the_architecture_defines() {
    const AMCR_EL0: Encoding = Encoding::new(3, 3, 13, 2, 0);
    const AMCFGR_EL0: Encoding = Encoding::new(3, 3, 13, 2, 1);
    const AMUSERENR_EL0: Encoding = Encoding::new(3, 3, 13, 2, 3);
    const AMCNTENCLR0_EL0: Encoding = Encoding::new(3, 3, 13, 2, 4);
    const AMCNTENSET0_EL0: Encoding = Encoding::new(3, 3, 13, 2, 5);
    const AMCNTENSET1_EL0: Encoding = Encoding::new(3, 3, 13, 3, 1);
    const AMCNTENCLR1_EL0: Encoding = Encoding::new(3, 3, 13, 3, 0);
    const AMEVCNTR1_0: Encoding = Encoding::new(3, 3, 13, 12, 0);
    const AMEVTYPER1_0: Encoding = Encoding::new(3, 3, 13, 14, 0);
    const AMEVCNTVOFF1_0: Encoding = Encoding::new(3, 4, 13, 10, 0);
    const AMEVCNTVOFF0_1: Encoding = Encoding::new(3, 4, 13, 8, 1);
    const HAFGRTR_EL2: Encoding = Encoding::new(3, 4, 3, 1, 6);
    const CPTR_EL2: Encoding = Encoding::new(3, 4, 1, 1, 2);
    const CNTVOFF_EL2: Encoding = Encoding::new(3, 4, 14, 0, 3);
    const HCR_E2H: u64 = 1 << 34;
    const HCR_AMVOFFEN: u64 = 1 << 51;
    const TAM: u64 = 1 << 30;
    const EC_SYSTEM: u64 = 0x18;
    let counter = |n| Encoding::new(3, 3, 13, 4, n);
    let event = |n| Encoding::new(3, 3, 13, 6, n);
    let offset = |n| Encoding::new(3, 4, 13, 8, n);
    let mut machine = machine(&[]);
    let chip = machine.profile.chip;
    let at = |machine: &mut TestMachine, el: u8, read: bool, encoding, value: u64| {
        machine.cpu.pc = BASE;
        machine.cpu.pstate = Pstate::reset(el);
        machine.cpu.state.x[1] = value;
        machine
            .system_instruction(access(read, encoding, if read { 2 } else { 1 }))
            .unwrap();
        machine.cpu.state.x[2]
    };
    if !Feature::FEAT_AMUv1.on(chip) {
        // The M1 to M3 have no activity monitors: UNDEFINED at EL1.
        at(&mut machine, 1, true, AMCR_EL0, 0);
        assert_eq!(machine.cpu.read(Register::ESR_EL1), 0x0200_0000);
        return;
    }
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x2_0000);
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x4_0000);
    machine.cpu.registers.set(Register::HCR_EL2, 0, HCR_E2H);
    machine.cpu.registers.set(Register::CPTR_EL2, 0, 0);
    // The identification the boot builds (`identification::derive`).
    let derived = crate::machine::identification::derive(
        chip,
        stela_compiler::TargetCpu::Baseline,
        true,
        crate::machine::identification::Speculation::NONE,
    );
    let amcfgr = derived
        .iter()
        .find(|(encoding, _)| *encoding == AMCFGR_EL0)
        .map(|(_, value)| *value);
    machine
        .profile_missing
        .retain(|register| *register != Register::AMCFGR_EL0);
    machine
        .cpu
        .registers
        .set_raw(Register::AMCFGR_EL0, 0, amcfgr.unwrap());
    // Four architected 64-bit counters and their mandated events.
    assert_eq!(at(&mut machine, 2, true, AMCFGR_EL0, 0), 0x0100_3f03);
    let events: Vec<u64> = (0..4)
        .map(|n| at(&mut machine, 2, true, event(n), 0))
        .collect();
    assert_eq!(events, [0x11, 0x4004, 0x8, 0x4005]);
    // EL2, the highest EL, enables them; the instructions-retired
    // counter counts what the engine retires, and memory stalls never
    // occur on the modeled core.
    at(&mut machine, 2, false, AMCNTENSET0_EL0, u64::MAX);
    assert_eq!(at(&mut machine, 2, true, AMCNTENSET0_EL0, 0), 0xf);
    at(&mut machine, 2, false, counter(2), 100);
    machine.cpu.retired += 7;
    assert_eq!(at(&mut machine, 2, true, counter(2), 0), 107);
    assert_eq!(at(&mut machine, 2, true, counter(3), 0), 0);
    at(&mut machine, 2, false, AMCNTENCLR0_EL0, 1 << 2);
    assert_eq!(at(&mut machine, 2, true, AMCNTENCLR0_EL0, 0), 0b1011);
    machine.cpu.retired += 7;
    assert_eq!(at(&mut machine, 2, true, counter(2), 0), 107);
    // No group 1: its enables hold nothing and its counters are
    // UNDEFINED, as is the absent offset of the constant frequency
    // counter.
    at(&mut machine, 2, false, AMCNTENSET1_EL0, u64::MAX);
    assert_eq!(at(&mut machine, 2, true, AMCNTENSET1_EL0, 0), 0);
    assert_eq!(at(&mut machine, 2, true, AMCNTENCLR1_EL0, 0), 0);
    for encoding in [AMEVCNTR1_0, AMEVTYPER1_0, AMEVCNTVOFF1_0, AMEVCNTVOFF0_1] {
        at(&mut machine, 2, true, encoding, 0);
        assert_eq!(machine.cpu.pc, 0x4_0200, "{encoding:?}");
    }
    // EL1 reads; EL1 never writes the monitors.
    assert_eq!(at(&mut machine, 1, true, counter(2), 0), 107);
    at(&mut machine, 1, false, counter(2), 0);
    assert_eq!(machine.cpu.read(Register::ESR_EL1), 0x0200_0000);
    // FEAT_AMUv1p1's virtual offsets apply at EL1 under AMVOFFEN.
    if Feature::FEAT_AMUv1p1.on(chip) {
        at(&mut machine, 2, false, offset(2), 7);
        // The Arm bank's CNTVOFF_EL2, with the M3-and-later counter
        // redirect enabled at EL2 (`timer::agt`).
        machine.cpu.apple.generic_timer.set_redirect(1, 0);
        at(&mut machine, 2, false, CNTVOFF_EL2, 5);
        machine
            .cpu
            .registers
            .set(Register::HCR_EL2, 0, HCR_E2H | HCR_AMVOFFEN);
        assert_eq!(at(&mut machine, 1, true, counter(2), 0), 100);
        let physical = at(&mut machine, 2, true, counter(1), 0);
        let virtual_count = at(&mut machine, 1, true, counter(1), 0);
        // The test host's counter stands still between the reads.
        assert_eq!(physical.wrapping_sub(virtual_count), 5, "CNTVOFF_EL2");
        assert_eq!(at(&mut machine, 2, true, counter(2), 0), 107);
        machine.cpu.registers.set(Register::HCR_EL2, 0, HCR_E2H);
    }
    // CPTR_EL2.TAM traps EL1 to EL2, and a clear AMUSERENR_EL0.EN traps
    // EL0 to EL1 outside the host.
    at(&mut machine, 2, false, CPTR_EL2, TAM);
    at(&mut machine, 1, true, counter(0), 0);
    assert_eq!(machine.cpu.read(Register::ESR_EL2) >> 26, EC_SYSTEM);
    at(&mut machine, 2, false, CPTR_EL2, 0);
    at(&mut machine, 0, true, counter(0), 0);
    assert_eq!(machine.cpu.read(Register::ESR_EL1) >> 26, EC_SYSTEM);
    at(&mut machine, 1, false, AMUSERENR_EL0, 1);
    assert_eq!(at(&mut machine, 0, true, counter(3), 0), 0);
    // With FEAT_FGT, HAFGRTR_EL2 traps EL1 reads of the enables
    // (AMCNTEN0, bit 0) and of each counter (AMEVCNTR0<n>, bit n + 1).
    if Feature::FEAT_FGT.on(chip) {
        at(&mut machine, 2, false, HAFGRTR_EL2, 1 | 1 << 3);
        for encoding in [AMCNTENSET0_EL0, counter(2)] {
            machine.cpu.registers.set(Register::ESR_EL2, 0, 0);
            at(&mut machine, 1, true, encoding, 0);
            assert_eq!(
                machine.cpu.read(Register::ESR_EL2) >> 26,
                EC_SYSTEM,
                "{encoding:?}"
            );
        }
        assert_eq!(at(&mut machine, 1, true, counter(3), 0), 0);
    }
    // AMCR_EL0 holds HDBG (and CG1RZ with FEAT_AMUv1p1); nothing halts.
    at(&mut machine, 2, false, AMCR_EL0, u64::MAX);
    let cg1rz = if Feature::FEAT_AMUv1p1.on(chip) {
        1 << 17
    } else {
        0
    };
    assert_eq!(at(&mut machine, 2, true, AMCR_EL0, 0), 1 << 10 | cg1rz);
}

#[test]
fn sctlr_tscxt_traps_el0_scxtnum_to_el1_or_to_el2_in_the_host() {
    const SCXTNUM_EL0: Encoding = Encoding::new(3, 3, 13, 0, 7);
    const SCTLR_EL2: Encoding = Encoding::new(3, 4, 1, 0, 0);
    const TSCXT: u64 = 1 << 20;
    const HCR_ENSCXT: u64 = 1 << 53;
    const EC_SYSTEM: u64 = 0x18;
    let mut machine = machine(&[]);
    machine.cpu.registers.set(Register::VBAR_EL1, 0, 0x1_0000);
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x4_0000);
    machine.cpu.registers.set(Register::SCXTNUM_EL0, 0, 0x5c);
    let user = Pstate {
        el: 0,
        sp: false,
        daif: 0,
        ..Pstate::reset(1)
    };
    let write_read = |machine: &mut TestMachine, el: u8, encoding: Encoding, value: u64| {
        machine.cpu.pstate = Pstate::reset(el);
        machine.cpu.state.x[0] = value;
        machine
            .system_instruction(access(false, encoding, 0))
            .unwrap();
        machine
            .system_instruction(access(true, encoding, 1))
            .unwrap();
        machine.cpu.state.x[1]
    };
    let el0_read = |machine: &mut TestMachine| {
        machine.cpu.pstate = user;
        machine.cpu.pc = BASE;
        machine.cpu.state.x[2] = 0;
        machine
            .system_instruction(access(true, SCXTNUM_EL0, 2))
            .unwrap();
        (
            machine.cpu.pstate.el,
            machine.cpu.pc,
            machine.cpu.state.x[2],
        )
    };
    // Outside the host with HCR_EL2.EnSCXT set, EL0 reads the number
    // until SCTLR_EL1.TSCXT, which holds what EL1 writes, traps it to
    // EL1 with EC 0x18.
    machine.cpu.registers.set(Register::HCR_EL2, 0, HCR_ENSCXT);
    assert_eq!(el0_read(&mut machine), (0, BASE + 4, 0x5c));
    assert_eq!(write_read(&mut machine, 1, SCTLR_EL1, TSCXT) & TSCXT, TSCXT);
    assert_eq!(el0_read(&mut machine), (1, 0x1_0400, 0));
    assert_eq!(machine.cpu.read(Register::ESR_EL1) >> 26, EC_SYSTEM);
    // TSCXT is checked before EnSCXT: with both, the trap is EL1's.
    machine.cpu.registers.set(Register::HCR_EL2, 0, 0);
    assert_eq!(el0_read(&mut machine).0, 1);
    // Without TSCXT, a clear EnSCXT traps EL0 to EL2.
    assert_eq!(write_read(&mut machine, 1, SCTLR_EL1, 0) & TSCXT, 0);
    assert_eq!(el0_read(&mut machine), (2, 0x4_0400, 0));
    assert_eq!(machine.cpu.read(Register::ESR_EL2) >> 26, EC_SYSTEM);
    // In the host, SCTLR_EL1 and EnSCXT do not apply: SCTLR_EL2.TSCXT
    // traps EL0 to EL2.
    machine
        .cpu
        .registers
        .set(Register::HCR_EL2, 0, 1 << HCR_TGE);
    machine.cpu.registers.set(Register::SCTLR_EL1, 0, TSCXT);
    assert_eq!(el0_read(&mut machine), (0, BASE + 4, 0x5c));
    assert_eq!(write_read(&mut machine, 2, SCTLR_EL2, TSCXT) & TSCXT, TSCXT);
    assert_eq!(el0_read(&mut machine), (2, 0x4_0400, 0));
    assert_eq!(machine.cpu.read(Register::ESR_EL2) >> 26, EC_SYSTEM);
}

#[test]
fn pstate_dit_is_written_read_and_kept_across_exceptions() {
    const DIT: Encoding = Encoding::new(3, 3, 4, 2, 5);
    const DIT_BIT: u64 = 1 << 24;
    const MSR_DIT_ONE: u32 = 0xd503_415f;
    const MSR_DIT_ZERO: u32 = 0xd503_405f;
    const ERET: u32 = 0xd69f_03e0;
    let mut machine = machine(&[]);
    // Without the host's timing guarantee ID_AA64PFR0_EL1 reports no
    // FEAT_DIT, but the field is kept as on every Apple chip
    // (`dit_is_kept_without_the_host_timing_guarantee`).
    assert_eq!(machine.cpu.read(Register::ID_AA64PFR0_EL1) >> 48 & 0xf, 0);
    let read = |machine: &mut TestMachine| {
        machine.system_instruction(access(true, DIT, 1)).unwrap();
        machine.cpu.state.x[1]
    };
    machine.cpu.pstate = Pstate::reset(1);
    machine.cpu.pc = BASE;
    // MSR DIT, #imm and MRS DIT: the bit reads at its PSTATE position.
    machine.system_instruction(MSR_DIT_ONE).unwrap();
    assert!(machine.cpu.pstate.dit);
    assert_eq!(read(&mut machine), DIT_BIT);
    machine.system_instruction(MSR_DIT_ZERO).unwrap();
    assert_eq!(read(&mut machine), 0);
    // MSR DIT, Xt keeps only bit 24.
    machine.cpu.state.x[0] = u64::MAX;
    machine.system_instruction(access(false, DIT, 0)).unwrap();
    assert_eq!(read(&mut machine), DIT_BIT);
    // EL0 reaches it too.
    machine.cpu.pstate = Pstate {
        el: 0,
        sp: false,
        dit: false,
        ..Pstate::reset(1)
    };
    machine.system_instruction(MSR_DIT_ONE).unwrap();
    assert_eq!((machine.cpu.pstate.el, read(&mut machine)), (0, DIT_BIT));
    // An exception keeps DIT and saves it in SPSR_ELx.DIT; ERET restores
    // it from there.
    machine.cpu.pstate = Pstate {
        dit: true,
        ..Pstate::reset(1)
    };
    machine.cpu.registers.set(Register::VBAR_EL2, 0, 0x4_0000);
    machine.system_instruction(HVC).unwrap();
    assert_eq!(machine.cpu.pstate.el, 2);
    assert!(machine.cpu.pstate.dit);
    let spsr = machine.cpu.read(Register::SPSR_EL2);
    assert_eq!(spsr & DIT_BIT, DIT_BIT);
    machine
        .cpu
        .registers
        .set(Register::SPSR_EL2, 0, spsr & !DIT_BIT);
    machine.system_instruction(ERET).unwrap();
    assert_eq!(machine.cpu.pstate.el, 1);
    assert!(!machine.cpu.pstate.dit);
}
