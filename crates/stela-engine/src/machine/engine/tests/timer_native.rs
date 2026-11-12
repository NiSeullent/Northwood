//! Native EL2 virtual timer accesses (`x86_64::register_read::timer_thunk`)
//! run on the machine's own frame and cells against the engine path: a
//! write completes natively only when it moves the next rise no earlier
//! than the engine looks again, and the engine then takes the timer from
//! storage, never from what it cached before the write.

use super::super::transitions::tests::run_native;
use super::*;
use core::sync::atomic::Ordering::Relaxed;
use stela_compiler::x86_64::register_read::{Count, CountState, TimerAccess};

const HCR_E2H: u64 = 1 << 34;
const TVAL: u32 = 0xd51b_e313; // msr cntv_tval_el0, x19
const CVAL: u32 = 0xd51b_e353; // msr cntv_cval_el0, x19
const CTL_WRITE: u32 = 0xd51b_e333; // msr cntv_ctl_el0, x19
const CTL_READ: u32 = 0xd53b_e32a; // mrs x10, cntv_ctl_el0
const ENABLE: u64 = 1;
const IMASK: u64 = 2;
const FIQ: u32 = 0b01;

/// A PE at EL2 with E2H on a chip with one timer bank, whose native count
/// reader reads `clock`.
fn host(clock: &CountState) -> Machine<FakeHost, NoDevices> {
    let mut machine = machine_on(one_timer_bank(), &[CVAL]);
    machine.cpu.pstate = Pstate::reset(2);
    let hcr = machine.cpu.read(SystemRegister::HCR_EL2);
    machine
        .cpu
        .registers
        .set(SystemRegister::HCR_EL2, 0, hcr | HCR_E2H);
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

/// The engine's count and the native one at `now`.
fn at(machine: &mut Machine<FakeHost, NoDevices>, clock: &CountState, now: u64) {
    machine.host.now = now;
    clock.offset.store(now, Relaxed);
}

/// Complete `word` with X19 `value` in the engine.
fn engine(machine: &mut Machine<FakeHost, NoDevices>, word: u32, value: u64) {
    machine.cpu.state.x[19] = value;
    machine.cpu.pc = BASE;
    machine.system_instruction(word).unwrap();
    assert_eq!(machine.cpu.pc, BASE + 4, "{word:#x} completed plainly");
}

/// Run `access` natively with X19 `value`: whether it completed.
fn native(machine: &mut Machine<FakeHost, NoDevices>, access: TimerAccess, value: u64) -> bool {
    let rt = if access == TimerAccess::ReadControl {
        10
    } else {
        19
    };
    machine.cpu.state.x[19] = value;
    let thunk = register_read::timer_thunk(GuestAddress(BASE), rt, access, TIMER);
    match run_native(machine, &thunk.code) {
        (pc, kind) if pc == BASE + 4 && kind == ExitKind::Fallthrough as u64 => true,
        (pc, kind) => {
            assert_eq!((pc, kind), (BASE, ExitKind::System as u64));
            false
        }
    }
}

fn stored(machine: &Machine<FakeHost, NoDevices>) -> (u64, u64) {
    (
        machine.cpu.registers.get(SystemRegister::CNTHV_CVAL_EL2, 0),
        machine.cpu.registers.get(SystemRegister::CNTHV_CTL_EL2, 0),
    )
}

/// The FIQ the engine finds waiting at `now` (every interrupt masked).
fn fiq_at(machine: &mut Machine<FakeHost, NoDevices>, clock: &CountState, now: u64) -> bool {
    at(machine, clock, now);
    assert!(!machine.deliver_interrupts(), "every interrupt is masked");
    machine.cpu.state.pending & FIQ != 0
}

/// The engine armed the timer for count 150 at count 100.
fn armed(clock: &CountState) -> Machine<FakeHost, NoDevices> {
    let mut machine = host(clock);
    at(&mut machine, clock, 100);
    engine(&mut machine, CVAL, 150);
    engine(&mut machine, CTL_WRITE, ENABLE);
    assert!(!fiq_at(&mut machine, clock, 100));
    assert_eq!(machine.mirrors.lines_until, 150);
    machine
}

fn count_state() -> alloc::boxed::Box<CountState> {
    let clock = alloc::boxed::Box::new(CountState::default());
    clock.ceiling.store(u64::MAX, Relaxed);
    clock
}

/// A plain access opens both cells on CNTHV_*'s storage, only at EL2 with
/// E2H outside a guarded level on a chip with one bank; the block's exit
/// gets the timer thunk where the host counts its TSC.
#[test]
fn plain_timer_accesses_open_the_cells_where_they_reach_the_el2_virtual_timer() {
    let clock = count_state();
    let machine = armed(&clock);
    let address = |register| machine.cpu.registers.address(register, 0).unwrap();
    assert_eq!(
        (
            machine.mirrors[TIMER_COMPARE],
            machine.mirrors[TIMER_CONTROL]
        ),
        (
            register_read::Cell {
                readable: 1,
                value: address(SystemRegister::CNTHV_CVAL_EL2),
            },
            register_read::Cell {
                readable: 1,
                value: address(SystemRegister::CNTHV_CTL_EL2),
            }
        )
    );
    for word in [TVAL, CVAL, CTL_WRITE, CTL_READ] {
        let mut machine = host(&clock);
        machine.cpu.pstate = Pstate::reset(1);
        machine.derived_changed();
        at(&mut machine, &clock, 100);
        engine(&mut machine, word, 0);
        assert_eq!(machine.mirrors[TIMER_COMPARE].readable, 0, "EL1");
        let mut machine = host(&clock);
        at(&mut machine, &clock, 100);
        engine(&mut machine, word, 0);
        assert_eq!(machine.mirrors[TIMER_COMPARE].readable, 1, "{word:#x}");
    }
    let chip = stela_contracts::Chip::AppleM3;
    let mut machine = machine_on(chip, &[CVAL]);
    machine.cpu.pstate = Pstate::reset(2);
    machine.derived_changed();
    engine(&mut machine, CVAL, 150);
    assert_eq!(machine.mirrors[TIMER_COMPARE].readable, 0, "two banks");

    const ADD: u32 = 0x9100_0400; // add x0, x0, #1
    for tsc in [None, Some((5, 7))] {
        let mut machine = machine_on(one_timer_bank(), &[ADD, TVAL, ADD]);
        machine.host.tsc_counter = tsc;
        let base = machine.cpu.pc;
        machine.host.exits = vec![NativeExit::Exit {
            next_pc: base + 8,
            kind: ExitKind::Fallthrough as u64,
        }];
        machine.step().unwrap();
        let thunk =
            register_read::timer_thunk(GuestAddress(base + 4), 19, TimerAccess::WriteValue, TIMER);
        assert_eq!(
            machine.host.installed.last() == Some(&thunk.code),
            tsc.is_some()
        );
    }
}

/// A later deadline completes natively; the engine, which looked again at
/// its old deadline, raises nothing there and the FIQ at the new one. An
/// earlier one, or one due at once, exits and changes nothing.
#[test]
fn later_deadlines_complete_natively_and_fire_at_the_new_count() {
    let clock = count_state();
    let mut machine = armed(&clock);
    assert!(!native(&mut machine, TimerAccess::WriteValue, 20));
    assert!(!native(&mut machine, TimerAccess::WriteValue, 0));
    assert!(!native(
        &mut machine,
        TimerAccess::WriteValue,
        (-5i32) as u32 as u64
    ));
    assert!(!native(&mut machine, TimerAccess::WriteCompare, 149));
    assert_eq!(stored(&machine), (150, ENABLE));
    assert!(native(&mut machine, TimerAccess::WriteValue, 100));
    assert_eq!(stored(&machine), (200, ENABLE));
    assert!(
        !fiq_at(&mut machine, &clock, 150),
        "no FIQ at the stale deadline"
    );
    assert_eq!(machine.mirrors.lines_until, 200);
    assert!(!fiq_at(&mut machine, &clock, 199));
    assert!(fiq_at(&mut machine, &clock, 200));
    // The line is high now: every write is the engine's.
    machine.mirrors.lines_until = 300;
    assert!(!native(&mut machine, TimerAccess::WriteCompare, 400));
    let written = fixed(&machine);
    assert!(!native(&mut machine, TimerAccess::WriteControl(written), 0));
}

fn fixed(machine: &Machine<FakeHost, NoDevices>) -> register_read::Written {
    machine.register_fixed_bits(SystemRegister::CNTHV_CTL_EL2)
}

/// Disabling and masking complete natively, and no FIQ comes at the
/// deadline the engine had; unmasking an expired compare exits, and the
/// engine's write then raises the FIQ.
#[test]
fn disable_and_mask_stay_native_and_unmasking_an_expired_compare_exits() {
    let clock = count_state();
    let mut machine = armed(&clock);
    let written = fixed(&machine);
    assert!(native(&mut machine, TimerAccess::WriteControl(written), 0));
    assert_eq!(stored(&machine), (150, 0));
    assert!(!fiq_at(&mut machine, &clock, 150));
    assert!(!fiq_at(&mut machine, &clock, 10_000));
    assert_eq!(machine.mirrors.lines_until, u64::MAX);

    let mut machine = armed(&clock);
    let written = fixed(&machine);
    assert!(native(
        &mut machine,
        TimerAccess::WriteControl(written),
        ENABLE | IMASK
    ));
    assert!(!fiq_at(&mut machine, &clock, 160));
    assert!(!native(
        &mut machine,
        TimerAccess::WriteControl(written),
        ENABLE
    ));
    assert_eq!(stored(&machine).1, ENABLE | IMASK);
    engine(&mut machine, CTL_WRITE, ENABLE);
    assert!(fiq_at(&mut machine, &clock, 160));
}

/// The count passes the new compare right after the thunk read it: the
/// engine, which looks again at its own deadline (150) or before, finds
/// the FIQ at the new compare (180) as it would after its own write; a
/// native ISR read stops at the old deadline.
#[test]
fn a_count_crossing_the_new_compare_after_the_thunk_still_raises_the_fiq() {
    let clock = count_state();
    let mut machine = armed(&clock);
    assert!(native(&mut machine, TimerAccess::WriteValue, 80));
    at(&mut machine, &clock, 180);
    let epoch = Place::new(
        Anchor::Control,
        core::mem::offset_of!(ChainControl, interrupt_epoch),
    );
    let count = Place::new(Anchor::Count, 0);
    let isr = register_read::interrupt_status_thunk(
        GuestAddress(BASE),
        9,
        INTERRUPT_STATUS,
        epoch,
        count,
    );
    assert_eq!(
        run_native(&mut machine, &isr.code),
        (BASE, ExitKind::System as u64),
        "the ISR read stops at the old deadline"
    );
    assert!(fiq_at(&mut machine, &clock, 180));
    // The engine path at the same counts.
    let mut engine_path = armed(&clock);
    engine(&mut engine_path, CVAL, 180);
    assert!(fiq_at(&mut engine_path, &clock, 180));
}

/// CTL reads natively give what the engine's read gives, in all eight
/// ENABLE, IMASK and ISTATUS cases.
#[test]
fn control_reads_match_the_engine_in_every_case() {
    let clock = count_state();
    let mut machine = armed(&clock);
    let mut now = 200;
    for control in [0, ENABLE, IMASK, ENABLE | IMASK] {
        for past in [false, true] {
            now += 10;
            at(&mut machine, &clock, now);
            let compare = if past { now } else { now + 1 };
            machine
                .cpu
                .registers
                .set(SystemRegister::CNTHV_CVAL_EL2, 0, compare);
            machine
                .cpu
                .registers
                .set(SystemRegister::CNTHV_CTL_EL2, 0, control);
            machine.cpu.state.x[10] = 0xdead;
            assert!(native(&mut machine, TimerAccess::ReadControl, 0));
            let read = machine.cpu.state.x[10];
            machine.cpu.state.x[10] = 0xdead;
            machine.cpu.pc = BASE;
            machine.system_instruction(CTL_READ).unwrap();
            assert_eq!(read, machine.cpu.state.x[10], "{control} {past}");
            assert_eq!(read & 4 != 0, control & ENABLE != 0 && past);
        }
    }
}
