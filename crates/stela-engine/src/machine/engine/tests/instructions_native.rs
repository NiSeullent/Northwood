//! Native PMC1 reads (`x86_64::register_read::instruction_counter_thunk`)
//! run on the machine's own frame and cells against the engine path: where
//! a native transition closed the counter's cell but the read stays plain,
//! the thunk computes the count the engine's read would answer after the
//! same instructions retired and opens the cell on it.

use super::super::transitions::tests::run_native;
use super::*;
use crate::machine::pmu::{self, Context, Filter, Pmu};

const MRS_X10_PMC1: u32 = 0xd53a_f10a;
const ADD: u32 = 0x9100_0400; // add x0, x0, #1
const CONTEXT: Context = Context {
    el: 1,
    filter: Filter::Guest,
    waiting: false,
};

/// A PE at EL1 whose PMC1 took 1000 before it retired 10 instructions and
/// counts retired instructions at EL1 (or not at all with `counting`
/// false).
fn counting_machine(counting: bool) -> Machine<FakeHost, NoDevices> {
    let mut machine = machine(&[]);
    machine.cpu.pstate = Pstate::reset(1);
    let mut monitor = Pmu::new(36_000_000, 24_000_000);
    monitor.write(pmu::PMCR1, 1 << 17, 0, 0, CONTEXT).unwrap();
    monitor
        .write(pmu::PMCR1_EL12, 1 << 17, 0, 0, CONTEXT)
        .unwrap();
    monitor
        .write(pmu::PMCR0, u64::from(counting) << 1, 0, 0, CONTEXT)
        .unwrap();
    monitor.write(pmu::PMC[1], 1_000, 0, 0, CONTEXT).unwrap();
    machine.cpu.pmu = Some(monitor);
    machine.cpu.retired = 10;
    machine.derived_changed();
    machine
}

/// PMC1 as the engine reads it.
fn engine(machine: &mut Machine<FakeHost, NoDevices>) -> u64 {
    machine.cpu.pc = BASE;
    machine.system_instruction(MRS_X10_PMC1).unwrap();
    assert_eq!(machine.cpu.pc, BASE + 4, "completed plainly");
    machine.cpu.state.x[10]
}

/// PMC1 as the thunk reads it after native code entered with the machine's
/// retired count chained out of blocks that retired `chained` instructions
/// and ran `before` of the reader's block, if it completed.
fn native(machine: &mut Machine<FakeHost, NoDevices>, chained: u64, before: u32) -> Option<u64> {
    let thunk = register_read::instruction_counter_thunk(
        GuestAddress(BASE),
        10,
        cell_place(COUNTER_MIRRORS[1]),
        cell_place(INSTRUCTIONS_PLAIN),
        INSTRUCTION_COUNTER,
        before,
    )
    .unwrap();
    let control = machine.chain.address() as *mut chain::ChainControl;
    // SAFETY: the machine's own control, which no native code runs over
    // meanwhile.
    unsafe { core::ptr::addr_of_mut!((*control).retired).write(chained) };
    machine.cpu.state.x[10] = 0xdead;
    match run_native(machine, &thunk.code) {
        (pc, kind) if pc == BASE + 4 && kind == ExitKind::Fallthrough as u64 => {
            Some(machine.cpu.state.x[10])
        }
        (pc, kind) => {
            assert_eq!((pc, kind), (BASE, ExitKind::System as u64));
            assert_eq!(machine.cpu.state.x[10], 0xdead);
            None
        }
    }
}

/// What a native transition does to the counter's cell.
fn transition_natively(machine: &mut Machine<FakeHost, NoDevices>) {
    machine.mirrors[COUNTER_MIRRORS[1]].readable = 0;
}

/// After each native transition the thunk reads what a twin that takes
/// every read in the engine reads after the same instructions, the
/// engine's exit at the read retiring the chained blocks and the reader's
/// block up to it; native and engine reads interleave, and the thunk opens
/// the cell on that count as the engine does.
#[test]
fn native_instruction_reads_equal_the_engine_path_after_the_same_instructions() {
    let mut machine = counting_machine(true);
    let mut twin = counting_machine(true);
    assert_eq!(engine(&mut machine), 1_010);
    assert_eq!(engine(&mut twin), 1_010);
    assert_eq!(machine.mirrors[INSTRUCTIONS_PLAIN].readable, 1);
    let mut entered = 10;
    for (step, (chained, before)) in [(0, 0), (0, 3), (7, 0), (40, 2), (1 << 33, 9), (1, 1)]
        .into_iter()
        .enumerate()
    {
        transition_natively(&mut machine);
        transition_natively(&mut twin);
        let at_read = entered + chained + u64::from(before);
        twin.cpu.retired = at_read;
        let expected = engine(&mut twin);
        assert_eq!(expected, 1_000 + at_read);
        let read = if step % 3 == 2 {
            machine.cpu.retired = at_read;
            engine(&mut machine)
        } else {
            machine.cpu.retired = entered;
            native(&mut machine, chained, before).expect("plain, stale")
        };
        assert_eq!(read, expected, "after {chained} chained, {before} before");
        assert_eq!(
            machine.mirrors[COUNTER_MIRRORS[1]],
            register_read::Cell {
                readable: 1,
                value: read
            }
        );
        // The open cell keeps the count until the next transition, as for
        // the existing cell thunk.
        assert_eq!(native(&mut machine, chained + 5, before), Some(read));
        // The rest of the block and the next run's entry.
        entered = at_read + 3;
    }
    // A PMC1 that does not count reads its count as it stands.
    let mut idle = counting_machine(false);
    assert_eq!(engine(&mut idle), 1_000);
    idle.cpu.retired = 5_000;
    transition_natively(&mut idle);
    assert_eq!(native(&mut idle, 70, 4), Some(1_000));
}

/// The thunk exits where the read may not be plain: before the engine
/// completed one in the state, after derived state changed, with a degraded
/// answer, without a monitor, while the counter's PMI is armed or where the
/// PE counts otherwise than the monitor's context; a block ending in the
/// read gets the thunk with the instructions before it.
#[test]
fn native_instruction_reads_exit_unless_the_read_is_plain() {
    let mut machine = counting_machine(true);
    assert_eq!(native(&mut machine, 0, 0), None, "no plain read yet");
    let read = engine(&mut machine);
    transition_natively(&mut machine);
    assert_eq!(native(&mut machine, 4, 0), Some(read + 4));
    machine.derived_changed();
    assert_eq!(machine.mirrors[INSTRUCTIONS_PLAIN].readable, 0);
    assert_eq!(native(&mut machine, 4, 0), None, "derived state changed");
    engine(&mut machine);
    transition_natively(&mut machine);
    let selector = MIRRORED[COUNTER_MIRRORS[1]].selector();
    machine.cpu.degraded.insert(selector, 7);
    assert_eq!(native(&mut machine, 4, 0), None, "degraded");
    machine.cpu.degraded.remove(&selector);
    assert!(native(&mut machine, 4, 0).is_some());
    transition_natively(&mut machine);
    // PMCR0: PMC0 and PMC1 count, FIQ PMI armed on PMC1, then on PMC0.
    for (armed, cycles_open) in [(1 << 13, 1), (1 << 12, 0)] {
        let retired = machine.cpu.retired;
        let monitor = machine.cpu.pmu.as_mut().unwrap();
        monitor
            .write(pmu::PMCR0, 0b11 | 4 << 8 | armed, 0, retired, CONTEXT)
            .unwrap();
        let read = native(&mut machine, 4, 0);
        assert_eq!(read.is_none(), armed == 1 << 13, "PMI armed");
        assert_eq!(machine.mirrors.cycles.open, cycles_open);
        transition_natively(&mut machine);
    }
    let retired = machine.cpu.retired;
    let monitor = machine.cpu.pmu.as_mut().unwrap();
    monitor.write(pmu::PMCR0, 2, 0, retired, CONTEXT).unwrap();
    assert!(native(&mut machine, 4, 0).is_some());
    transition_natively(&mut machine);
    // EL0, which PMCR1_EL12 does not count, without the monitor's sync.
    machine.cpu.pstate.el = 0;
    assert_eq!(native(&mut machine, 4, 0), None, "context differs");
    machine.cpu.pstate.el = 1;
    machine.cpu.pmu = None;
    assert_eq!(native(&mut machine, 4, 0), None, "no monitor");

    let mut machine = super::machine(&[ADD, ADD, MRS_X10_PMC1, ADD]);
    let base = machine.cpu.pc;
    machine.host.exits = vec![NativeExit::Exit {
        next_pc: base + 12,
        kind: ExitKind::Fallthrough as u64,
    }];
    machine.step().unwrap();
    let thunk = register_read::instruction_counter_thunk(
        GuestAddress(base + 8),
        10,
        cell_place(COUNTER_MIRRORS[1]),
        cell_place(INSTRUCTIONS_PLAIN),
        INSTRUCTION_COUNTER,
        2,
    )
    .unwrap();
    assert_eq!(machine.host.installed.last(), Some(&thunk.code));
}
