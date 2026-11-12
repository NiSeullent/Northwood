//! Native PMC0 reads (`x86_64::register_read::cycle_counter_thunk`) run on
//! the machine's own frame and cells against the engine path: where a
//! native transition closed the counter's cell but the read stays plain,
//! the thunk computes the count the engine's read would answer at the same
//! system count and opens the cell on it.

use super::super::transitions::tests::run_native;
use super::*;
use crate::machine::pmu::{self, Context, Filter, Pmu};
use core::sync::atomic::Ordering::Relaxed;
use stela_compiler::x86_64::register_read::{Count, CountState};

const MRS_X10_PMC0: u32 = 0xd53a_f00a;
const ADD: u32 = 0x9100_0400; // add x0, x0, #1

/// A PE at EL1 whose PMC0 counts cycles of a 36 000 007 Hz core on a
/// 24 MHz system counter at EL1 (or not at all with `counting` false), and
/// whose native count reader reads `clock`.
fn counting_machine(clock: &CountState, counting: bool) -> Machine<FakeHost, NoDevices> {
    let mut machine = machine(&[]);
    machine.cpu.pstate = Pstate::reset(1);
    let context = Context {
        el: 1,
        filter: Filter::Guest,
        waiting: false,
    };
    let mut monitor = Pmu::new(36_000_007, 24_000_000);
    monitor.write(pmu::PMCR1, 1 << 16, 0, 0, context).unwrap();
    monitor
        .write(pmu::PMCR1_EL12, 1 << 16, 0, 0, context)
        .unwrap();
    monitor
        .write(pmu::PMCR0, u64::from(counting), 0, 0, context)
        .unwrap();
    monitor.write(pmu::PMC[0], 1_000, 0, 0, context).unwrap();
    machine.cpu.pmu = Some(monitor);
    machine.derived_changed();
    let reader = machine.clock.reader() as *mut Count;
    // SAFETY: the machine's own reader, which no native code runs over
    // meanwhile; `clock` outlives every run of the test.
    unsafe {
        (*reader).per_tsc = 0;
        (*reader).last = 0;
        (*reader).state = clock as *const CountState as u64;
    }
    machine
}

fn count_state() -> alloc::boxed::Box<CountState> {
    let clock = alloc::boxed::Box::new(CountState::default());
    clock.ceiling.store(u64::MAX, Relaxed);
    clock
}

/// The engine's count and the native one at `now`.
fn at(machine: &mut Machine<FakeHost, NoDevices>, clock: &CountState, now: u64) {
    machine.host.now = now;
    clock.offset.store(now, Relaxed);
}

/// PMC0 as the engine reads it.
fn engine(machine: &mut Machine<FakeHost, NoDevices>) -> u64 {
    machine.cpu.pc = BASE;
    machine.system_instruction(MRS_X10_PMC0).unwrap();
    assert_eq!(machine.cpu.pc, BASE + 4, "completed plainly");
    machine.cpu.state.x[10]
}

/// PMC0 as the thunk reads it, if it completed.
fn native(machine: &mut Machine<FakeHost, NoDevices>) -> Option<u64> {
    let thunk = register_read::cycle_counter_thunk(
        GuestAddress(BASE),
        10,
        cell_place(COUNTER_MIRRORS[0]),
        cell_place(CYCLES_PLAIN),
        CYCLES,
        Place::new(Anchor::Count, 0),
    );
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
    machine.mirrors[COUNTER_MIRRORS[0]].readable = 0;
}

/// After each native transition the thunk reads what a twin that takes
/// every read in the engine reads at the same count, with native and
/// engine reads interleaved, the remainder carried, and never backwards;
/// it opens the cell on that count as the engine does.
#[test]
fn native_cycle_reads_equal_the_engine_path_at_the_same_count() {
    let clock = count_state();
    let mut machine = counting_machine(&clock, true);
    let twin_clock = count_state();
    let mut twin = counting_machine(&twin_clock, true);
    at(&mut machine, &clock, 100);
    at(&mut twin, &twin_clock, 100);
    assert_eq!(engine(&mut machine), engine(&mut twin));
    assert_eq!(machine.mirrors[CYCLES_PLAIN].readable, 1);
    let mut last = 0;
    for (step, now) in [101, 103, 104, 1_000, 1_001, 77_777, 77_778, 1 << 33]
        .into_iter()
        .enumerate()
    {
        at(&mut machine, &clock, now);
        at(&mut twin, &twin_clock, now);
        transition_natively(&mut machine);
        transition_natively(&mut twin);
        let expected = engine(&mut twin);
        let read = if step % 3 == 2 {
            engine(&mut machine)
        } else {
            native(&mut machine).expect("plain, stale")
        };
        assert_eq!(read, expected, "at {now}");
        assert!(read >= last, "monotonic at {now}");
        last = read;
        assert_eq!(
            machine.mirrors[COUNTER_MIRRORS[0]],
            register_read::Cell {
                readable: 1,
                value: read
            }
        );
        // The open cell keeps the count until the next transition, as for
        // the existing cell thunk; an engine read is always live.
        at(&mut machine, &clock, now + 50);
        assert_eq!(native(&mut machine), Some(read));
    }
    // A PMC0 that does not count reads its count as it stands.
    let mut idle = counting_machine(&clock, false);
    at(&mut idle, &clock, 100);
    assert_eq!(engine(&mut idle), 1_000);
    at(&mut idle, &clock, 5_000);
    transition_natively(&mut idle);
    assert_eq!(native(&mut idle), Some(1_000));
}

/// The thunk exits where the read may not be plain: before the engine
/// completed one in the state, after derived state changed, without a
/// monitor or with a degraded answer; and the block's exit gets the thunk
/// only where the host counts its TSC.
#[test]
fn native_cycle_reads_exit_unless_the_read_is_plain() {
    let clock = count_state();
    let mut machine = counting_machine(&clock, true);
    at(&mut machine, &clock, 100);
    assert_eq!(native(&mut machine), None, "no plain read yet");
    let read = engine(&mut machine);
    transition_natively(&mut machine);
    at(&mut machine, &clock, 200);
    assert!(native(&mut machine).is_some_and(|value| value > read));
    machine.derived_changed();
    assert_eq!(machine.mirrors[CYCLES_PLAIN].readable, 0);
    assert_eq!(native(&mut machine), None, "derived state changed");
    engine(&mut machine);
    transition_natively(&mut machine);
    let selector = MIRRORED[COUNTER_MIRRORS[0]].selector();
    machine.cpu.degraded.insert(selector, 7);
    assert_eq!(native(&mut machine), None, "degraded");
    machine.cpu.degraded.remove(&selector);
    machine.cpu.pmu = None;
    assert_eq!(native(&mut machine), None, "no monitor");

    for tsc in [None, Some((5, 7))] {
        let mut machine = super::machine(&[ADD, MRS_X10_PMC0, ADD]);
        machine.host.tsc_counter = tsc;
        let base = machine.cpu.pc;
        machine.host.exits = vec![NativeExit::Exit {
            next_pc: base + 8,
            kind: ExitKind::Fallthrough as u64,
        }];
        machine.step().unwrap();
        let thunk = register_read::cycle_counter_thunk(
            GuestAddress(base + 4),
            10,
            cell_place(COUNTER_MIRRORS[0]),
            cell_place(CYCLES_PLAIN),
            CYCLES,
            Place::new(Anchor::Count, 0),
        );
        assert_eq!(
            machine.host.installed.last() == Some(&thunk.code),
            tsc.is_some()
        );
    }
}
