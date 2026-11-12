//! Native SVC and ERET between EL0 and EL2 of the host
//! (`x86_64::transition::supervisor_call_thunk`, `exception_return_thunk`)
//! against the engine path: every architectural effect the engine's
//! `Cpu::take` and `Cpu::exception_return` make, and the normal exit for
//! each case the record does not cover.

use super::super::transitions::tests::run_native;
use super::*;
use crate::machine::cpu::{RESERVATION_GRANULE, Reservation};
use stela_compiler::x86_64::transition::{EXCEPTION_RETURN, SUPERVISOR_CALL};

/// HCR_EL2.E2H and TGE: the host.
const HCR_HOST: u64 = 1 << 34 | 1 << 27;
const VECTORS: u64 = BASE + 0x8000;
/// The lower-EL synchronous vector.
const SVC_VECTOR: u64 = VECTORS + 0x400;
const ERET_PC: u64 = BASE + 0x40;
const USER: u64 = BASE + 0x100;
const SVC_WORD: u32 = 0xd400_0001 | 0x80 << 5; // svc #0x80
const ERET: u32 = 0xd69f_03e0;
/// SPSR of the return to EL0t: Z and C, DIT and SSBS, every mask clear.
const USER_SPSR: u64 = 0x6 << 28 | 1 << 24 | 1 << 12;

/// A PE at EL2 of the host, its vectors at [`VECTORS`].
fn host() -> Machine<FakeHost, NoDevices> {
    let mut machine = machine(&[]);
    machine.cpu.pstate = Pstate::reset(2);
    machine
        .cpu
        .registers
        .set(SystemRegister::HCR_EL2, 0, HCR_HOST);
    machine
        .cpu
        .registers
        .set(SystemRegister::VBAR_EL2, 0, VECTORS);
    machine.sync_regime();
    machine
}

/// The operands both twins start a case from, at the ERET.
fn prepare(machine: &mut Machine<FakeHost, NoDevices>) {
    machine.cpu.pc = ERET_PC;
    machine.cpu.state.nzcv = 0;
    machine.cpu.state.sp = 0xffff_fe00_1234_5670;
    let registers = &mut machine.cpu.registers;
    registers.set(SystemRegister::SPSR_EL2, 0, USER_SPSR);
    registers.set(SystemRegister::ELR_EL2, 0, USER);
    registers.set(SystemRegister::SP_EL0, 0, 0x0000_7000_0000_1000);
}

/// Engine-path ERETs and SVCs until both have records.
fn warm(machine: &mut Machine<FakeHost, NoDevices>) {
    for _ in 0..2 {
        prepare(machine);
        machine.system_instruction(ERET).unwrap();
        machine.finish_transition(1);
        assert_eq!(machine.cpu.pc, USER);
        machine.system_instruction(SVC_WORD).unwrap();
        machine.finish_transition(1);
        assert_eq!(machine.cpu.pc, SVC_VECTOR);
    }
    let table = &machine.mirrors.transitions;
    assert_eq!(
        table.records[0][SUPERVISOR_CALL].generation,
        table.generation
    );
    assert_eq!(
        table.records[2][EXCEPTION_RETURN].generation,
        table.generation
    );
}

/// Everything an SVC or ERET changes, for comparison.
fn observed(machine: &Machine<FakeHost, NoDevices>) -> alloc::string::String {
    let cpu = &machine.cpu;
    let state = GuestState {
        daif: 0,
        pan: 0,
        anchors: Default::default(),
        lower_limit: Default::default(),
        ..cpu.state.clone()
    };
    let read = |register| cpu.registers.get(register, 0);
    alloc::format!(
        "{:?} {:?} pc={:#x} spsr={:#x} elr={:#x} esr={:#x} sp={:#x},{:#x} state={} \
         event={} monitor={:?} exceptions={} pointer={:x?}",
        cpu.pstate,
        state,
        cpu.pc,
        read(SystemRegister::SPSR_EL2),
        read(SystemRegister::ELR_EL2),
        read(SystemRegister::ESR_EL2),
        cpu.sp(0),
        cpu.sp(2),
        machine.mirror_state(),
        cpu.event_is_set(),
        cpu.monitor.is_some(),
        machine.statistics.exceptions,
        machine.pointer.tags,
    )
}

fn thunks() -> (x86_64_pointer::Thunk, x86_64_pointer::Thunk) {
    let syndrome =
        crate::machine::exception::syndrome(crate::machine::exception::Class::SupervisorCall, 0x80)
            as u32;
    (
        transition::exception_return_thunk(GuestAddress(ERET_PC), &TRANSITIONS),
        transition::supervisor_call_thunk(GuestAddress(USER), syndrome, &TRANSITIONS),
    )
}

/// ERET to EL0 and SVC back complete natively from the records as the
/// engine completes them: PSTATE (EL, SP selection, masks, NZCV, PAN, DIT,
/// UAO, SSBS), SPSR, ELR and ESR, both stacks, the cells' state, the event
/// register, the local monitor, the exception count and the PAC service
/// native PAC instructions run on.
#[test]
fn native_svc_and_eret_match_the_engine_path() {
    let (mut engine, mut native) = (host(), host());
    warm(&mut engine);
    warm(&mut native);
    let (eret, svc) = thunks();
    for machine in [&mut engine, &mut native] {
        prepare(machine);
        machine.cpu.state.nzcv = 0x9000_0000;
        machine.cpu.state.exclusive_bytes = 8;
        machine.cpu.consume_event();
    }
    native.mirrors.lines_until = u64::MAX;
    assert_eq!(observed(&engine), observed(&native));
    engine.system_instruction(ERET).unwrap();
    let exit = run_native(&mut native, &eret.code);
    assert_eq!(exit, (USER, ExitKind::Return as u64));
    assert_eq!(observed(&engine), observed(&native));
    assert_eq!((native.cpu.pstate.el, native.cpu.pstate.sp), (0, false));
    assert!(native.cpu.event_is_set());
    assert_eq!(native.cpu.state.exclusive_bytes, 0);
    // SVC from EL0 with other flags and masks.
    for machine in [&mut engine, &mut native] {
        machine.cpu.pc = USER;
        machine.cpu.state.nzcv = 0x4000_0000;
        machine.cpu.state.sp = 0x0000_7000_0000_0ff0;
        machine.cpu.pstate.daif = 0b0100;
    }
    engine.system_instruction(SVC_WORD).unwrap();
    let exit = run_native(&mut native, &svc.code);
    assert_eq!(exit, (SVC_VECTOR, ExitKind::Return as u64));
    assert_eq!(observed(&engine), observed(&native));
    assert_eq!((native.cpu.pstate.el, native.cpu.pstate.sp), (2, true));
    assert_eq!(native.statistics.native_transitions[..2], [1, 1]);
}

/// A tagged ELR loses its tag where TCR_EL2's TBI0 applies, as
/// `Cpu::branch_address` removes it.
#[test]
fn native_eret_removes_the_tag_the_engine_removes() {
    let (mut engine, mut native) = (host(), host());
    for machine in [&mut engine, &mut native] {
        let tcr = machine.cpu.registers.get(SystemRegister::TCR_EL2, 0);
        machine
            .cpu
            .registers
            .set(SystemRegister::TCR_EL2, 0, tcr | 1 << 37);
        machine.sync_regime();
        warm(machine);
        prepare(machine);
        let tagged = USER | 0x5a << 56;
        machine
            .cpu
            .registers
            .set(SystemRegister::ELR_EL2, 0, tagged);
    }
    native.mirrors.lines_until = u64::MAX;
    engine.system_instruction(ERET).unwrap();
    let (eret, _) = thunks();
    let exit = run_native(&mut native, &eret.code);
    assert_eq!(exit, (USER, ExitKind::Return as u64));
    assert_eq!(observed(&engine), observed(&native));
}

/// What a record does not cover takes the normal exit with nothing
/// changed: another mode, single-step, PACM or an illegal return in SPSR,
/// an interrupt waiting that the return unmasks, an engine monitor, a
/// deadline due or not known, a notification, and for SVC another PAN.
#[test]
fn uncovered_svc_and_eret_take_the_normal_exit() {
    let (eret, svc) = thunks();
    type Case = fn(&mut Machine<FakeHost, NoDevices>);
    let cases: [(&str, Case); 9] = [
        ("EL2h", |m| {
            m.cpu
                .registers
                .set(SystemRegister::SPSR_EL2, 0, USER_SPSR | 0b1001)
        }),
        ("single-step", |m| {
            m.cpu
                .registers
                .set(SystemRegister::SPSR_EL2, 0, USER_SPSR | 1 << 21)
        }),
        // Stored raw: SPSR keeps PACM only with FEAT_PAuth_LR.
        ("PACM", |m| {
            m.cpu
                .registers
                .set_raw(SystemRegister::SPSR_EL2, 0, USER_SPSR | 1 << 35)
        }),
        ("EL1 under TGE", |m| {
            m.cpu
                .registers
                .set(SystemRegister::SPSR_EL2, 0, USER_SPSR | 0b0101)
        }),
        ("IRQ unmasked", |m| m.cpu.state.pending = 0b10),
        ("monitor", |m| {
            m.cpu.monitor = Some(Reservation {
                granule: 0x1000,
                bytes: [0; RESERVATION_GRANULE as usize],
            })
        }),
        ("deadline", |m| m.mirrors.lines_until = 0),
        ("notification", |m| {
            m.chain.weak_storage().upgrade().unwrap().notify()
        }),
        ("SVC's PAN", |_| {}),
    ];
    for (name, case) in cases {
        let mut machine = host();
        warm(&mut machine);
        prepare(&mut machine);
        machine.mirrors.lines_until = u64::MAX;
        let code = if name == "SVC's PAN" {
            machine.system_instruction(ERET).unwrap();
            machine.cpu.pc = USER;
            machine.cpu.pstate.pan = !machine.cpu.pstate.pan;
            &svc.code
        } else {
            &eret.code
        };
        case(&mut machine);
        let before = observed(&machine);
        let pc = machine.cpu.pc;
        let exit = run_native(&mut machine, code);
        assert_eq!(exit, (pc, ExitKind::System as u64), "{name}");
        assert_eq!(observed(&machine), before, "{name}");
    }
}

/// SCTLR_EL2's four PAC key enables.
const KEY_ENABLES: u64 = 1 << 31 | 1 << 30 | 1 << 27 | 1 << 13;

/// With APCTL's kernel-keys bit, EL2 signs with every key while EL0 of a
/// process without JOP keys (SCTLR's enables clear) does not: the engine
/// path reopens the PAC service at each level. A native SVC or ERET
/// switches the service's tag words only from a service with one open, so
/// with the service never opened neither makes a record (the twin round
/// trip below covers an open one). With SCTLR enabling the keys too, both
/// levels sign alike and the records switch nothing.
#[test]
fn svc_and_eret_make_records_only_where_pac_acts_alike() {
    for (enables, recorded) in [(0, [false, false]), (KEY_ENABLES, [true, true])] {
        let mut machine = host();
        let sctlr = machine.cpu.registers.get(SystemRegister::SCTLR_EL2, 0);
        machine
            .cpu
            .registers
            .set(SystemRegister::SCTLR_EL2, 0, sctlr & !KEY_ENABLES | enables);
        machine.cpu.apple.apctl[1] = 1 << 4;
        machine.sync_regime();
        for _ in 0..2 {
            prepare(&mut machine);
            machine.system_instruction(ERET).unwrap();
            machine.finish_transition(1);
            machine.system_instruction(SVC_WORD).unwrap();
            machine.finish_transition(1);
        }
        let table = &machine.mirrors.transitions;
        for (index, (state, kind)) in [(0, SUPERVISOR_CALL), (2, EXCEPTION_RETURN)]
            .into_iter()
            .enumerate()
        {
            let record = &table.records[state][kind];
            let made = record.generation == table.generation;
            assert_eq!(made, recorded[index], "enables {enables:#x} kind {kind}");
            if made {
                let swaps = record.pointer_swap != 0;
                assert_eq!(swaps, enables == 0, "enables {enables:#x} kind {kind}");
            }
        }
    }
}

/// MSR TTBR0_EL1, X1: TTBR0_EL2 in the host.
const MSR_TTBR0: u32 = 0xd518_2001;

/// A process switch (TTBR0) or an SPRR_PERM write changes the lookup
/// contexts of the level returned to, which records do not forget: the
/// ERET to EL0 then completes but ends its run rather than look its target
/// up in the previous process's context, and the next native entry at EL0
/// gives the record the new contexts.
#[test]
fn svc_and_eret_records_take_new_contexts_after_ttbr0_and_permission_writes() {
    for case in ["TTBR0", "SPRR_PERM"] {
        let mut machine = host();
        warm(&mut machine);
        let current = |machine: &Machine<FakeHost, NoDevices>| {
            let table = &machine.mirrors.transitions;
            let record = &table.records[2][EXCEPTION_RETURN];
            assert_eq!(record.generation, table.generation, "{case}");
            record.contexts_generation == table.contexts_generation
        };
        assert!(current(&machine), "{case}");
        if case == "TTBR0" {
            machine.cpu.state.x[1] = 0x0001_0000_8000_0000;
            machine.system_instruction(MSR_TTBR0).unwrap();
            assert_eq!(
                machine.cpu.registers.get(SystemRegister::TTBR0_EL2, 0),
                0x0001_0000_8000_0000
            );
        } else {
            machine.permissions_changed();
        }
        assert!(!current(&machine), "{case}");
        machine.cpu.pstate = Pstate {
            el: 0,
            sp: false,
            ..machine.cpu.pstate
        };
        machine.refresh_transition_contexts(1);
        assert!(current(&machine), "{case}");
    }
}

/// The twin round trip where EL2 keeps every key on (APCTL) and EL0 of a
/// process without JOP keys has only IB: after an ERET to EL0 and an SVC
/// back, the PAC service native PAC code runs on is the engine's at each
/// level (IA, DA and DB off at EL0): the records switch it natively, and a
/// service in another state takes the normal exit.
#[test]
fn svc_and_eret_leave_the_pac_service_of_the_level_entered() {
    let configure = |machine: &mut Machine<FakeHost, NoDevices>| {
        let sctlr = machine.cpu.registers.get(SystemRegister::SCTLR_EL2, 0);
        machine
            .cpu
            .registers
            .set(SystemRegister::SCTLR_EL2, 0, sctlr & !KEY_ENABLES | 1 << 30);
        machine.cpu.apple.apctl[1] = 1 << 4;
        use SystemRegister as R;
        for (index, register) in [
            R::APIAKeyLo_EL1,
            R::APIAKeyHi_EL1,
            R::APIBKeyLo_EL1,
            R::APIBKeyHi_EL1,
            R::APDAKeyLo_EL1,
            R::APDAKeyHi_EL1,
            R::APDBKeyLo_EL1,
            R::APDBKeyHi_EL1,
        ]
        .into_iter()
        .enumerate()
        {
            let half = crate::machine::pointer_keys::Register::architectural(register).unwrap();
            let value = 0x0123_4567_89ab_cdef_u64.rotate_left(8 * index as u32);
            machine.cpu.apple.pointer_keys.write(half, value);
        }
        machine.sync_regime();
        // The service a first PAC exit opens.
        let inputs = PointerInputs {
            keys: machine.pointer_keys(),
            version: machine.pointer_version().unwrap(),
            tcr: machine.cpu.regime_tcr(),
        };
        machine.pointer.open(inputs);
        machine.revalidate_pointers();
        assert!(machine.pointer.any_open());
        for _ in 0..2 {
            prepare(machine);
            machine.system_instruction(ERET).unwrap();
            machine.finish_transition(1);
            machine.system_instruction(SVC_WORD).unwrap();
            machine.finish_transition(1);
        }
    };
    let (mut engine, mut native) = (host(), host());
    configure(&mut engine);
    configure(&mut native);
    let (eret, svc) = thunks();
    let table = &native.mirrors.transitions;
    for (state, kind) in [(0, SUPERVISOR_CALL), (2, EXCEPTION_RETURN)] {
        let record = &table.records[state][kind];
        assert_eq!(record.generation, table.generation, "kind {kind}");
        assert_ne!(record.pointer_swap, 0, "kind {kind}");
    }
    let complete = |machine: &mut Machine<FakeHost, NoDevices>, code: &[u8], _word: u32| {
        let pc = machine.cpu.pc;
        assert_ne!(run_native(machine, code), (pc, ExitKind::System as u64));
    };
    for machine in [&mut engine, &mut native] {
        prepare(machine);
    }
    native.mirrors.lines_until = u64::MAX;
    assert_eq!(observed(&engine), observed(&native));
    engine.system_instruction(ERET).unwrap();
    complete(&mut native, &eret.code, ERET);
    assert_eq!(observed(&engine), observed(&native));
    engine.system_instruction(SVC_WORD).unwrap();
    complete(&mut native, &svc.code, SVC_WORD);
    assert_eq!(observed(&engine), observed(&native));
    assert_eq!(native.statistics.native_transitions[..2], [1, 1]);
    // A service in another state (closed here) takes the normal exit.
    prepare(&mut native);
    native.pointer.close();
    let before = observed(&native);
    let exit = run_native(&mut native, &eret.code);
    assert_eq!(exit, (ERET_PC, ExitKind::System as u64));
    assert_eq!(observed(&native), before);
}
