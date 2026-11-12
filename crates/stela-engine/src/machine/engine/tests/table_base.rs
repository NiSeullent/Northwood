//! Native MSR TTBR0_EL1 of the value the register holds
//! (`register_read::unchanged_write_thunk`) against the engine path: the
//! same value completes natively and leaves everything as the engine's MSR
//! does; another value exits to the engine.

use super::*;

const MSR: u32 = 0xd518_2003; // msr ttbr0_el1, x3

/// A machine at `el` whose table-base cell a first engine-path write of
/// `first` opened, on the storage of the register the write reached.
fn opened(el: u8, first: u64) -> Machine<FakeHost, NoDevices> {
    let mut machine = machine(&[MSR]);
    machine.cpu.pstate = Pstate::reset(el);
    machine.derived_changed();
    let base = machine.cpu.pc;
    machine.cpu.state.x[3] = first;
    machine.system_instruction(MSR).unwrap();
    let cell = machine.mirrors[TABLE_BASE_KEPT];
    assert_eq!(cell.readable, 1, "a plain write opens the cell");
    let reached = [SystemRegister::TTBR0_EL1, SystemRegister::TTBR0_EL2]
        .map(|register| machine.cpu.registers.address(register, 0));
    assert!(reached.contains(&Some(cell.value)));
    machine.cpu.pc = base + 4;
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
        "pc={:#x} el1={:#x} el2={:#x} x={:?} open={open:?} isolations={} regimes={} generation={} derived={}",
        machine.cpu.pc,
        machine.cpu.registers.get(SystemRegister::TTBR0_EL1, 0),
        machine.cpu.registers.get(SystemRegister::TTBR0_EL2, 0),
        machine.cpu.state.x,
        machine.host.isolations,
        machine.host.regimes.len(),
        machine.translation_generation,
        machine.native_system.is_some(),
    )
}

#[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
#[test]
fn native_unchanged_table_base_writes_match_the_engine_path() {
    use super::super::transitions::tests::run_native;
    const FIRST: u64 = 1 << 48 | 0x8000;
    const OTHER: u64 = 2 << 48 | 0xc000;
    for el in [1, 2] {
        let (mut engine, mut native) = (opened(el, FIRST), opened(el, FIRST));
        engine.native_system();
        native.native_system();
        let pc = engine.cpu.pc;
        assert_eq!(table_base_write(pc, MSR), Some(3));
        let cell = cell_place(TABLE_BASE_KEPT);
        let written = native.register_fixed_bits(SystemRegister::TTBR0_EL1);
        let thunk = register_read::unchanged_write_thunk(GuestAddress(pc), 3, cell, written);
        assert_eq!(observed(&engine), observed(&native));
        // The same value completes natively, again and again; another value
        // exits and the engine completes it; then the new one is kept.
        let mut exits = 0;
        for value in [FIRST, FIRST, OTHER, OTHER, FIRST] {
            for machine in [&mut engine, &mut native] {
                machine.cpu.pc = pc;
                machine.cpu.state.x[3] = value;
            }
            let storage = native.mirrors[TABLE_BASE_KEPT].value as *const u64;
            // SAFETY: the open cell's value is the address of the register's
            // storage in the machine, which lives and is not written now.
            let held = unsafe { storage.read() };
            engine.system_instruction(MSR).unwrap();
            let (next, kind) = run_native(&mut native, &thunk.code);
            if value == held {
                assert_eq!((next, kind), (pc + 4, ExitKind::Fallthrough as u64));
                native.cpu.pc = next;
            } else {
                assert_eq!((next, kind), (pc, ExitKind::System as u64), "{value:#x}");
                native.system_instruction(MSR).unwrap();
                exits += 1;
            }
            assert_eq!(observed(&engine), observed(&native), "EL{el} {value:#x}");
        }
        assert_eq!(exits, 2, "EL{el}: only the two changes exit");
    }
}
