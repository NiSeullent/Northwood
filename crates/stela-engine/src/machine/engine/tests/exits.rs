//! Host timing of the engine's most frequent System exits: SVC from EL0 and
//! the ERET back, and the GENTER/GEXIT pair, through the same completion as
//! a native System exit (`Machine::completing` and
//! `complete_system_instruction`), followed by the variant the next fetch
//! derives. The manual test prints nanoseconds per transition; the others
//! check that the repeated transitions keep their architectural results.

use super::super::guarded::tests::gxf;
use super::*;
use crate::machine::exception::{DAIF_ALL, Pstate};

const SVC: u32 = 0xd400_0001;
const ERET: u32 = 0xd69f_03e0;
const GENTER: u32 = 0x0020_1420;
const GEXIT: u32 = 0x0020_1400;
const USER: u64 = BASE + 0x4000;
const VECTORS: u64 = BASE + 0x8000;
/// The lower-EL AArch64 synchronous vector.
const LOWER_SYNCHRONOUS: u64 = 0x400;
const HCR_E2H_TGE: u64 = 1 << 34 | 1 << 27;

/// One timed piece of work on a machine.
type Round = fn(&mut Machine<FakeHost, NoDevices>);

/// Complete `word` at the PC as a native System exit does.
fn exit(machine: &mut Machine<FakeHost, NoDevices>, word: u32) {
    let pc = machine.cpu.pc;
    let operation = machine.decode_system_operation(word);
    machine
        .completing(pc, |machine| {
            machine.complete_system_instruction(word, operation)
        })
        .unwrap();
}

/// A machine at EL0 with stage 1 on, whose SVCs go to EL1, or with `host`
/// to EL2 under HCR_EL2.{E2H,TGE}.
fn user(host: bool) -> Machine<FakeHost, NoDevices> {
    user_on(crate::machine::test_chip(), host)
}

/// [`user`] on `chip`.
fn user_on(chip: stela_contracts::Chip, host: bool) -> Machine<FakeHost, NoDevices> {
    let mut machine = machine_on(chip, &[]);
    normal_memory(&mut machine, true);
    let registers = &mut machine.cpu.registers;
    registers.set(SystemRegister::VBAR_EL1, 0, VECTORS);
    registers.set(SystemRegister::VBAR_EL2, 0, VECTORS);
    if host {
        registers.set(SystemRegister::HCR_EL2, 0, HCR_E2H_TGE);
    }
    machine.cpu.set_pstate(Pstate {
        sp: false,
        daif: 0,
        ..Pstate::reset(0)
    });
    machine.sync_regime();
    machine.cpu.pc = USER;
    machine
}

/// One SVC and its ERET, each with the next fetch's variant.
fn system_call(machine: &mut Machine<FakeHost, NoDevices>) {
    machine.cpu.pc = USER;
    exit(machine, SVC);
    machine.current_variant(false, false);
    exit(machine, ERET);
    machine.current_variant(false, false);
    machine.host.events.clear();
    machine.host.regimes.clear();
}

/// One GENTER and its GEXIT at EL1, each with the next fetch's variant.
fn guarded_call(machine: &mut Machine<FakeHost, NoDevices>, at: u64) {
    machine.cpu.pc = at;
    exit(machine, GENTER);
    machine.current_variant(false, false);
    exit(machine, GEXIT);
    machine.current_variant(false, false);
    machine.host.events.clear();
    machine.host.regimes.clear();
}

#[test]
fn repeated_system_calls_enter_and_return_exactly() {
    for host in [false, true] {
        let target = if host { 2 } else { 1 };
        let mut machine = user(host);
        for round in 0..4u64 {
            machine.cpu.pc = USER;
            machine.cpu.state.nzcv = 0x6000_0000;
            let retired = machine.cpu.retired;
            exit(&mut machine, SVC);
            assert_eq!(machine.cpu.pstate.el, target, "{host} {round}");
            assert_eq!(machine.cpu.pstate.daif, DAIF_ALL);
            assert_eq!(machine.cpu.pc, VECTORS + LOWER_SYNCHRONOUS);
            assert_eq!(machine.cpu.retired, retired + 1);
            let (elr, spsr, esr) = if host {
                (
                    SystemRegister::ELR_EL2,
                    SystemRegister::SPSR_EL2,
                    SystemRegister::ESR_EL2,
                )
            } else {
                (
                    SystemRegister::ELR_EL1,
                    SystemRegister::SPSR_EL1,
                    SystemRegister::ESR_EL1,
                )
            };
            assert_eq!(machine.cpu.read(elr), USER + 4);
            assert_eq!(machine.cpu.read(spsr), 0x6000_0000);
            assert_eq!(machine.cpu.read(esr), 0x5600_0000);
            let entered = machine.current_variant(false, false);
            exit(&mut machine, ERET);
            assert_eq!(machine.cpu.pstate.el, 0);
            assert_eq!(machine.cpu.pstate.daif, 0);
            assert_eq!(machine.cpu.pc, USER + 4);
            let returned = machine.current_variant(false, false);
            // EL0 checks SP alignment with SA0, EL1 and EL2 with SA.
            let sctlr = machine.cpu.regime_sctlr();
            assert_eq!(returned.checks.stack_alignment, sctlr & 1 << 4 != 0);
            machine.cpu.pc = USER;
            exit(&mut machine, SVC);
            assert_eq!(machine.current_variant(false, false), entered);
            exit(&mut machine, ERET);
            assert_eq!(machine.current_variant(false, false), returned);
        }
    }
}

/// Nanoseconds per transition of `round`, which makes `transitions` of
/// them, the best of several runs.
fn time(
    machine: &mut Machine<FakeHost, NoDevices>,
    transitions: u64,
    mut round: impl FnMut(&mut Machine<FakeHost, NoDevices>),
) -> u64 {
    const ROUNDS: u64 = 20_000;
    for _ in 0..ROUNDS / 10 {
        round(machine);
    }
    (0..5)
        .map(|_| {
            let started = std::time::Instant::now();
            for _ in 0..ROUNDS {
                round(machine);
            }
            started.elapsed().as_nanos() as u64 / (ROUNDS * transitions)
        })
        .min()
        .unwrap()
}

/// `cargo test -p stela-engine --release --lib exit_transition_costs --
/// --ignored --nocapture`
#[test]
#[ignore = "manual: opt-in"]
fn exit_transition_costs() {
    for host in [false, true] {
        let mut machine = user(host);
        let ns = time(&mut machine, 2, system_call);
        std::println!("EXITS svc+eret host={host} ns_per_transition={ns}");
    }
    let mut machine = gxf(stela_contracts::Chip::AppleM2);
    let at = machine.cpu.pc;
    let ns = time(&mut machine, 2, |machine| guarded_call(machine, at));
    std::println!("EXITS genter+gexit ns_per_transition={ns}");
    // Four PEs of one M2 cluster at once, whose transitions all read the
    // cluster's CTRR groups: unlocked, under its lock; locked, without it.
    for locked in [false, true] {
        let ns = contended(locked);
        std::println!("EXITS svc+eret pes=4 ctrr_locked={locked} ns_per_transition={ns}");
    }
    // The parts of one transition, at EL1 after an SVC.
    let mut machine = user(false);
    exit(&mut machine, SVC);
    let parts = parts();
    for (name, part) in parts {
        let ns = time(&mut machine, 1, part);
        std::println!("EXITS part={name} ns={ns}");
    }
    // A direct exit's whole round trip through the run loop.
    let mut machine = branches();
    let ns = time(&mut machine, 2, direct_round);
    std::println!("EXITS direct round_trip ns_per_exit={ns}");
    // Its parts: the schedule, the fetch, and the run with its exit.
    let ns = time(&mut machine, 1, |machine| machine.schedule().unwrap());
    std::println!("EXITS direct part=schedule ns={ns}");
    let ns = time(&mut machine, 1, |machine| {
        let _ = machine.fetch_block(BASE, false).unwrap();
    });
    std::println!("EXITS direct part=fetch ns={ns}");
    let Ok(Fetched::Block(index)) = machine.fetch_block(BASE, false) else {
        panic!("a block");
    };
    let ns = time(&mut machine, 1, |machine| {
        machine.host.exits.push(NativeExit::Exit {
            next_pc: BASE + 8,
            kind: ExitKind::Direct as u64,
        });
        let budget = machine.chain_budget();
        machine.execute(index, budget).unwrap();
    });
    std::println!("EXITS direct part=execute ns={ns}");
}

/// `b .+8`, a word no run reaches, and `b .-8`: two blocks whose direct
/// exits lead to each other.
fn branches() -> Machine<FakeHost, NoDevices> {
    const FORWARD: u32 = 0x1400_0002;
    const BACK: u32 = 0x17ff_fffe;
    machine(&[FORWARD, 0, BACK])
}

/// Each block's run ending at its direct exit to the other, through the
/// whole loop step: the schedule, the fetch, the entry, the native return,
/// the exit and its link.
fn direct_round(machine: &mut Machine<FakeHost, NoDevices>) {
    for next_pc in [BASE + 8, BASE] {
        machine.host.exits.push(NativeExit::Exit {
            next_pc,
            kind: ExitKind::Direct as u64,
        });
        machine.schedule().unwrap();
        machine.step().unwrap();
        assert_eq!(machine.cpu.pc, next_pc);
    }
}

/// The round trip keeps the PC moving between the two blocks and leaves no
/// scripted exit behind.
#[test]
fn direct_rounds_alternate_between_the_blocks() {
    let mut machine = branches();
    for _ in 0..8 {
        direct_round(&mut machine);
    }
    assert!(machine.host.exits.is_empty());
    assert_eq!(machine.statistics.exit_kinds[ExitKind::Direct as usize], 16);
}

/// The parts of one transition, at EL1 after an SVC.
fn parts() -> [(&'static str, Round); 6] {
    [
        ("cpu-take+return", |machine| {
            let exception = crate::machine::cpu::Exception::synchronous(
                crate::machine::exception::Class::SupervisorCall,
                0,
                USER + 4,
            );
            let pstate = machine.cpu.pstate;
            machine.cpu.take(exception);
            machine.cpu.set_pstate(pstate);
        }),
        ("transitioned", |machine| {
            machine.transitioned();
            machine.host.regimes.clear();
        }),
        ("sync-host-regime", |machine| {
            machine.sync_host_regime();
            machine.host.regimes.clear();
        }),
        ("reopen-pointers", |machine| machine.reopen_pointers()),
        ("forget+variant", |machine| {
            machine.forget_derived();
            machine.current_variant(false, false);
        }),
        ("variant", |machine| {
            machine.current_variant(false, false);
        }),
    ]
}

/// Nanoseconds per transition of `round` (making `transitions`) when each
/// round starts after `evict` bytes of other data went through the caches,
/// as native guest code does between two exits: the median of `rounds`
/// rounds, each timed alone, so the eviction is not counted.
fn time_cold(
    machine: &mut Machine<FakeHost, NoDevices>,
    transitions: u64,
    evict: &mut [u64],
    mut round: impl FnMut(&mut Machine<FakeHost, NoDevices>),
) -> u64 {
    const ROUNDS: usize = 301;
    let mut samples = std::vec::Vec::with_capacity(ROUNDS);
    for index in 0..ROUNDS {
        // One write per line: the lines are dirty and must leave.
        for line in evict.iter_mut().step_by(8) {
            *line = line.wrapping_add(index as u64);
        }
        core::hint::black_box(&mut *evict);
        let started = std::time::Instant::now();
        round(machine);
        samples.push(started.elapsed().as_nanos() as u64);
    }
    samples.sort_unstable();
    samples[ROUNDS / 2] / transitions
}

/// `cargo test -p stela-engine --release --lib exit_transition_cold_costs
/// -- --ignored --nocapture`: [`exit_transition_costs`] with the caches
/// refilled by other data before every pair, which is how a guest's exits
/// find them: hot (none), past L2 (2 MiB), past L3 (256 MiB).
#[test]
#[ignore = "manual: opt-in"]
fn exit_transition_cold_costs() {
    for mib in [0usize, 2, 256] {
        let mut evict = std::vec![0u64; mib * 1024 * 1024 / 8];
        for host in [false, true] {
            let mut machine = user(host);
            let ns = time_cold(&mut machine, 2, &mut evict, system_call);
            std::println!("EXITS cold_mib={mib} svc+eret host={host} ns_per_transition={ns}");
        }
        let mut machine = gxf(stela_contracts::Chip::AppleM2);
        let at = machine.cpu.pc;
        let ns = time_cold(&mut machine, 2, &mut evict, |machine| {
            guarded_call(machine, at)
        });
        std::println!("EXITS cold_mib={mib} genter+gexit ns_per_transition={ns}");
    }
    // Hot code, cold data: SVC+ERET on each of `count` machines in turn,
    // so each finds its own state evicted by the others' but the engine's
    // code warm.
    for count in [1usize, 64] {
        let mut machines: std::vec::Vec<_> = (0..count).map(|_| user(true)).collect();
        let mut samples = std::vec::Vec::new();
        for pass in 0..8 {
            for machine in &mut machines {
                let started = std::time::Instant::now();
                system_call(machine);
                if pass > 0 {
                    samples.push(started.elapsed().as_nanos() as u64);
                }
            }
        }
        samples.sort_unstable();
        let ns = samples[samples.len() / 2] / 2;
        std::println!("EXITS machines={count} svc+eret host=true ns_per_transition={ns}");
    }
    // Each half alone, and each part of a transition (at EL1 after an
    // SVC), past L3.
    let mut evict = std::vec![0u64; 256 * 1024 * 1024 / 8];
    for (name, word) in [("svc", SVC), ("eret", ERET)] {
        let mut machine = user(true);
        let ns = time_cold(&mut machine, 1, &mut evict, |machine| {
            machine.cpu.pc = USER;
            exit(machine, SVC);
            if word == ERET {
                // Only the ERET is timed: the SVC before it runs warm and
                // is taken out by timing it again alone below.
            }
        });
        std::println!("EXITS cold_mib=256 {name}-alone host=true ns={ns}");
        if word == SVC {
            continue;
        }
        let mut samples = std::vec::Vec::new();
        for index in 0..301u64 {
            machine.cpu.pc = USER;
            exit(&mut machine, SVC);
            for line in evict.iter_mut().step_by(8) {
                *line = line.wrapping_add(index);
            }
            core::hint::black_box(&mut evict);
            let started = std::time::Instant::now();
            exit(&mut machine, ERET);
            samples.push(started.elapsed().as_nanos() as u64);
        }
        samples.sort_unstable();
        std::println!(
            "EXITS cold_mib=256 eret-alone host=true ns={}",
            samples[samples.len() / 2]
        );
    }
    let mut machine = user(false);
    exit(&mut machine, SVC);
    for (name, part) in parts() {
        let ns = time_cold(&mut machine, 1, &mut evict, part);
        std::println!("EXITS cold_mib=256 part={name} ns={ns}");
    }
    // Past L3, then the code warmed by another machine's SVC+ERET: what
    // is left of the cold cost is the timed machine's own data.
    let (mut warm, mut timed) = (user(true), user(true));
    let mut samples = std::vec::Vec::new();
    for index in 0..301u64 {
        for line in evict.iter_mut().step_by(8) {
            *line = line.wrapping_add(index);
        }
        core::hint::black_box(&mut evict);
        system_call(&mut warm);
        let started = std::time::Instant::now();
        system_call(&mut timed);
        samples.push(started.elapsed().as_nanos() as u64);
    }
    samples.sort_unstable();
    let ns = samples[samples.len() / 2] / 2;
    std::println!("EXITS cold_mib=256 code_warm svc+eret host=true ns_per_transition={ns}");
}

/// Nanoseconds per SVC or ERET transition of four PEs of one M2 cluster
/// running at once, the slowest PE's.
fn contended(locked: bool) -> u64 {
    use crate::machine::ctrr::Field;
    let cluster = crate::machine::shared::Shared::<crate::machine::ctrr::Cluster>::default();
    if locked {
        for level in 0..2 {
            cluster.borrow_mut().write(level, Field::Lock, 1).unwrap();
        }
    }
    let start = std::sync::Barrier::new(4);
    std::thread::scope(|scope| {
        let runs: std::vec::Vec<_> = (0..4)
            .map(|_| {
                let (cluster, start) = (cluster.clone(), &start);
                scope.spawn(move || {
                    let mut machine = user_on(stela_contracts::Chip::AppleM2, true);
                    machine.cpu.ctrr = cluster;
                    start.wait();
                    time(&mut machine, 2, system_call)
                })
            })
            .collect();
        runs.into_iter()
            .map(|run| run.join().unwrap())
            .max()
            .unwrap()
    })
}
