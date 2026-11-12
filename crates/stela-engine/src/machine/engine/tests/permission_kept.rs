//! Native MSR SPRR_PERM_EL0 and SPRR_PERM_EL1 of the value the register
//! holds (`register_read::unchanged_write_thunk` on [`PERMISSION_KEPT`])
//! against the engine path: the engine's write of the held value changes
//! nothing derived, so the native one, which changes nothing, leaves the
//! machine as the engine's does; another value exits to the engine.

use super::*;

const MSR_EL0: u32 = 0xd51e_f1a8; // msr sprr_perm_el0, x8
const MSR_EL1: u32 = 0xd51e_f1c8; // msr sprr_perm_el1, x8

/// A machine at `el` with SPRR enabled whose `word` cell a first engine
/// path write of `first` opened, on the storage the write reached.
fn opened(el: u8, word: u32, first: u64) -> Machine<FakeHost, NoDevices> {
    let mut machine = machine(&[word]);
    machine.cpu.pstate = Pstate::reset(el);
    machine.cpu.apple.sprr_config |= mmu::SPRR_CONFIG_EN;
    machine.derived_changed();
    let base = machine.cpu.pc;
    machine.cpu.state.x[8] = first;
    machine.system_instruction(word).unwrap();
    assert_eq!(machine.cpu.pc, base + 4, "{word:#x} completed plainly");
    let (kept, rt) = permission_write_word(word).unwrap();
    assert_eq!(rt, 8);
    let cell = machine.mirrors[kept];
    assert_eq!(cell.readable, 1, "a plain write opens the cell");
    let encoding = decode_register(word).unwrap().encoding();
    assert_eq!(Some(cell.value), machine.cpu.apple_read_storage(encoding));
    machine
}

fn observed(machine: &Machine<FakeHost, NoDevices>) -> alloc::string::String {
    let open: Vec<Vec<u64>> = machine
        .mirrors
        .banks
        .iter()
        .map(|cells| cells.iter().map(|cell| cell.readable).collect())
        .collect();
    alloc::format!(
        "pc={:#x} perm={:x?} x={:?} open={open:?} regimes={} generation={} changes={}",
        machine.cpu.pc,
        machine.cpu.apple.sprr_perm,
        machine.cpu.state.x,
        machine.host.regimes.len(),
        machine.translation_generation,
        machine.statistics.translation_changes,
    )
}

#[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
#[test]
fn native_unchanged_permission_writes_match_the_engine_path() {
    use super::super::transitions::tests::run_native;
    const FIRST: u64 = 0x2010_0000_3300_0000;
    const OTHER: u64 = 0x2010_0000_7700_0000;
    for word in [MSR_EL0, MSR_EL1] {
        for el in [1, 2] {
            let (mut engine, mut native) = (opened(el, word, FIRST), opened(el, word, FIRST));
            // What native entry derives first, which closes cells.
            engine.native_system();
            native.native_system();
            for machine in [&mut engine, &mut native] {
                machine.cpu.state.x[8] = FIRST;
                machine.cpu.pc = machine.cpu.pc.wrapping_sub(4);
                machine.system_instruction(word).unwrap();
            }
            let pc = engine.cpu.pc;
            let (kept, _) = permission_write_word(word).unwrap();
            let all = register_read::Written {
                keep: u64::MAX,
                set: 0,
            };
            let thunk =
                register_read::unchanged_write_thunk(GuestAddress(pc), 8, cell_place(kept), all);
            assert_eq!(observed(&engine), observed(&native));
            let mut exits = 0;
            for value in [FIRST, FIRST, OTHER, OTHER, FIRST] {
                for machine in [&mut engine, &mut native] {
                    machine.cpu.pc = pc;
                    machine.cpu.state.x[8] = value;
                }
                let storage = native.mirrors[kept].value as *const u64;
                // SAFETY: the open cell's value is the address of the
                // register's storage in the machine, which lives and is not
                // written now.
                let held = unsafe { storage.read() };
                engine.system_instruction(word).unwrap();
                let (next, kind) = run_native(&mut native, &thunk.code);
                if value == held {
                    assert_eq!((next, kind), (pc + 4, ExitKind::Fallthrough as u64));
                    native.cpu.pc = next;
                } else {
                    assert_eq!((next, kind), (pc, ExitKind::System as u64), "{value:#x}");
                    native.system_instruction(word).unwrap();
                    exits += 1;
                }
                assert_eq!(
                    observed(&engine),
                    observed(&native),
                    "{word:#x} EL{el} {value:#x}"
                );
            }
            assert_eq!(exits, 2, "{word:#x} EL{el}: only the two changes exit");
        }
    }
}

/// A change still forgets what derived from the permissions; the held
/// value forgets nothing; a refused write closes the cells; the block's
/// exit gets the thunk.
#[test]
fn permission_writes_forget_derived_state_only_when_they_change() {
    let mut machine = opened(2, MSR_EL1, 5);
    let generation = machine.translation_generation;
    let pc = machine.cpu.pc - 4;
    machine.cpu.pc = pc;
    machine.system_instruction(MSR_EL1).unwrap();
    assert_eq!(machine.translation_generation, generation, "held value");
    machine.cpu.pc = pc;
    machine.cpu.state.x[8] = 6;
    machine.system_instruction(MSR_EL1).unwrap();
    assert_ne!(machine.translation_generation, generation, "a change");
    machine.cpu.pc = pc;
    machine
        .register_refused(MSR_EL1, decode_register(MSR_EL1), Refusal::Unimplemented)
        .unwrap();
    assert!(
        PERMISSION_KEPT
            .iter()
            .all(|&kept| machine.mirrors[kept].readable == 0)
    );

    let mut machine = machine_with_permission_block();
    let base = machine.cpu.pc;
    machine.host.exits = vec![NativeExit::Exit {
        next_pc: base + 8,
        kind: ExitKind::Fallthrough as u64,
    }];
    machine.step().unwrap();
    let all = register_read::Written {
        keep: u64::MAX,
        set: 0,
    };
    let thunk = register_read::unchanged_write_thunk(
        GuestAddress(base + 4),
        8,
        cell_place(PERMISSION_KEPT[0]),
        all,
    );
    assert_eq!(machine.host.installed.last(), Some(&thunk.code));
}

fn machine_with_permission_block() -> Machine<FakeHost, NoDevices> {
    const ADD: u32 = 0x9100_0400; // add x0, x0, #1
    machine(&[ADD, MSR_EL0, ADD])
}
