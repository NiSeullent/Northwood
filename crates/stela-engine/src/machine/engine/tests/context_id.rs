//! Native MSR CONTEXTIDR_EL1 (`register_read::stored_write_thunk`) against
//! the engine path: the register the write reaches, the cells and the
//! host's regime and prediction barriers end up as the engine's MSR leaves
//! them.

use super::*;

const MSR: u32 = 0xd518_d021; // msr contextidr_el1, x1

fn index() -> usize {
    WRITTEN
        .iter()
        .position(|&(encoding, _)| encoding == CONTEXT_ID)
        .expect("CONTEXTIDR_EL1 is written natively")
}

/// A machine at `el` whose CONTEXTIDR write cell a first engine-path write
/// of `first` opened, on the storage of the register the write reached.
fn opened(el: u8, first: u64) -> Machine<FakeHost, NoDevices> {
    let mut machine = machine(&[MSR]);
    machine.cpu.pstate = Pstate::reset(el);
    machine.derived_changed();
    let base = machine.cpu.pc;
    machine.cpu.state.x[1] = first;
    machine.system_instruction(MSR).unwrap();
    let cell = machine.mirrors[written_cell(index())];
    assert_eq!(cell.readable, 1, "a plain write opens the cell");
    let reached = [
        SystemRegister::CONTEXTIDR_EL1,
        SystemRegister::CONTEXTIDR_EL2,
    ]
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
        "pc={:#x} el1={:#x} el2={:#x} x={:?} open={open:?} isolations={} regimes={} derived={}",
        machine.cpu.pc,
        machine.cpu.registers.get(SystemRegister::CONTEXTIDR_EL1, 0),
        machine.cpu.registers.get(SystemRegister::CONTEXTIDR_EL2, 0),
        machine.cpu.state.x,
        machine.host.isolations,
        machine.host.regimes.len(),
        machine.native_system.is_some(),
    )
}

#[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
#[test]
fn native_context_id_writes_match_the_engine_path() {
    use super::super::transitions::tests::run_native;
    for el in [1, 2] {
        let (mut engine, mut native) = (opened(el, 0x11), opened(el, 0x11));
        engine.native_system();
        native.native_system();
        let pc = engine.cpu.pc;
        assert_eq!(written_register(pc, MSR), Some((index(), 1)));
        let (_, register) = WRITTEN[index()];
        let cell = cell_place(written_cell(index()));
        let written = native.register_fixed_bits(register);
        let thunk = register_read::stored_write_thunk(GuestAddress(pc), 1, cell, written);
        assert_eq!(observed(&engine), observed(&native));
        // New, repeated and zero values, and one with bits CONTEXTIDR
        // reserves, which both paths drop alike.
        for value in [0x22, 0x22, 0, 0x7fff_ffff, 0xdead_beef_0000_0042, 0x11] {
            for machine in [&mut engine, &mut native] {
                machine.cpu.pc = pc;
                machine.cpu.state.x[1] = value;
            }
            engine.system_instruction(MSR).unwrap();
            let (next, kind) = run_native(&mut native, &thunk.code);
            assert_eq!((next, kind), (pc + 4, ExitKind::Fallthrough as u64));
            native.cpu.pc = next;
            assert_eq!(observed(&engine), observed(&native), "EL{el} {value:#x}");
        }
    }
}

/// A CONTEXTIDR write isolates no predictions and keeps the derived state
/// and every cell: native completion leaves out nothing the engine does.
#[test]
fn context_id_writes_keep_cells_regime_and_predictions() {
    let mut machine = opened(1, 0x11);
    machine.native_system();
    let cells = *machine.mirrors;
    let (isolations, regimes) = (machine.host.isolations, machine.host.regimes.len());
    machine.cpu.state.x[1] = 0x33;
    machine.system_instruction(MSR).unwrap();
    assert_eq!(
        machine.cpu.registers.get(SystemRegister::CONTEXTIDR_EL1, 0),
        0x33
    );
    assert_eq!(*machine.mirrors, cells);
    assert_eq!(machine.host.isolations, isolations);
    assert_eq!(machine.host.regimes.len(), regimes);
    assert!(machine.native_system.is_some());
}
