use super::super::tests::{BASE, board, installed_code, native_fault_site_in_entry};
use super::*;
use stela_compiler::block::RecoverySiteKind;

/// Registers that read their PA's low 16 bits with the top bit set,
/// recording every access in order.
#[derive(Default)]
struct Registers {
    accesses: Vec<(u64, u8, Option<u64>)>,
}

impl Devices for Registers {
    fn read(&mut self, _: usize, physical: u64, bytes: u8, _: u64) -> Result<u64, DeviceError> {
        self.accesses.push((physical, bytes, None));
        Ok(0x8000_0000 | (physical & 0xffff))
    }
    fn write(
        &mut self,
        _: usize,
        physical: u64,
        bytes: u8,
        value: u64,
        _: u64,
    ) -> Result<(), DeviceError> {
        self.accesses.push((physical, bytes, Some(value)));
        Ok(())
    }
    fn lines(&mut self, _: usize, _: u64) -> Lines {
        Lines::default()
    }
}

const DEVICE: u64 = 0x2_3b3c_0a00;

fn device_fault(offset: usize, address: u64, write: bool) -> NativeExit {
    NativeExit::Fault(NativeFault {
        origin: crate::machine::NativeFaultOrigin::Exception,
        trap: 14,
        error_code: 0,
        spills: stela_compiler::block::HostSpillSnapshot::new(),
        registers: stela_compiler::block::HostGprSnapshot::legacy([0; 16]),
        offset,
        address,
        write,
        cause: FaultCause::Device { physical: address },
    })
}

fn exit(next_pc: u64, kind: ExitKind) -> NativeExit {
    NativeExit::Exit {
        next_pc,
        kind: kind as u64,
    }
}

#[test]
fn nonresident_native_access_stops_at_the_operand_without_learning_mmio() {
    // add x3, x3, #1; ldr w0, [x1]; ret
    let words = [0x9100_0463, 0xb940_0020, 0xd65f_03c0];
    let mut machine = board(&words, Registers::default(), Vec::new());
    let frame = DEVICE & !(16 * 1024 - 1);
    machine.cpu.state.x[1] = DEVICE;
    let (_, offset) =
        native_fault_site_in_entry(&mut machine, BASE, BASE + 4, RecoverySiteKind::GuestAccess);
    let NativeExit::Fault(mut fault) = device_fault(offset, DEVICE, false) else {
        unreachable!()
    };
    fault.cause = FaultCause::NonresidentRam { frame };
    machine.host.exits.push(NativeExit::Fault(fault));
    let before = machine.cpu.retired;
    let Err(Stop::Unsupported(report)) = machine.step() else {
        panic!("nonresident access must stop for frame recovery")
    };
    assert_eq!(report.kind, RequirementKind::NonresidentRam { frame });
    assert_eq!(report.pc, BASE + 4);
    assert_eq!(machine.cpu.pc, BASE + 4);
    assert_eq!(machine.cpu.retired, before + 1);
    assert!(machine.devices.accesses.is_empty());
    assert!(!machine.page_has_device_sites(BASE));
}

#[test]
fn a_device_site_faults_once_then_completes_without_a_fault() {
    // add x3, x3, #1; ldr w0, [x1]; add x2, x2, #1; ret
    let words = [0x9100_0463, 0xb940_0020, 0x9100_0442, 0xd65f_03c0];
    let mut machine = board(&words, Registers::default(), Vec::new());
    machine.cpu.state.x[1] = DEVICE;
    let (_, offset) =
        native_fault_site_in_entry(&mut machine, BASE, BASE + 4, RecoverySiteKind::GuestAccess);
    // The first access faults, completes and is learned.
    machine.host.exits.push(device_fault(offset, DEVICE, false));
    machine.step().unwrap();
    assert_eq!(machine.devices.accesses, [(DEVICE, 4, None)]);
    assert_eq!(machine.cpu.state.x[0], 0x8000_0a00);
    assert_eq!(machine.cpu.pc, BASE + 8);
    assert!(machine.device_site(BASE + 4));
    assert!(machine.page_has_device_sites(BASE));
    // The page compiles again: the block at BASE ends before the site.
    machine.cpu.pc = BASE;
    let Ok(Fetched::Block(first)) = machine.fetch_block(BASE, false) else {
        panic!("a block");
    };
    assert_eq!(machine.code.blocks[first].instructions, 1);
    machine
        .host
        .exits
        .push(exit(BASE + 4, ExitKind::Fallthrough));
    machine.step().unwrap();
    assert_eq!(machine.cpu.pc, BASE + 4);
    // The site's block is a stub that exits to the engine, which completes
    // the access exactly, with no fault.
    let Ok(Fetched::Block(stub)) = machine.fetch_block(BASE + 4, false) else {
        panic!("a stub");
    };
    let code = installed_code(&mut machine, stub, 0, chain::DIRECT_EXIT_BYTES);
    assert_eq!(
        code,
        chain::direct_exit(GuestAddress(BASE + 4), ExitKind::System)
    );
    let retired = machine.cpu.retired;
    for round in 0..3 {
        machine.cpu.pc = BASE + 4;
        machine.host.exits.push(exit(BASE + 4, ExitKind::System));
        machine.step().unwrap();
        assert!(machine.host.exits.is_empty());
        assert_eq!(machine.devices.accesses.len(), 2 + round);
        assert_eq!(machine.devices.accesses[1 + round], (DEVICE, 4, None));
        assert_eq!(machine.cpu.pc, BASE + 8);
    }
    assert_eq!(machine.cpu.retired, retired + 3);
    // Blocks of the page are neither stored nor taken from a store.
    assert!(machine.before_device_site(BASE, 16) == Some(4));
}

#[test]
fn a_device_sites_exits_are_timed_apart_from_system_instructions() {
    // add x3, x3, #1; ldr w0, [x1]; add x2, x2, #1; ret
    let words = [0x9100_0463, 0xb940_0020, 0x9100_0442, 0xd65f_03c0];
    let mut machine = board(&words, Registers::default(), Vec::new());
    machine.cpu.state.x[1] = DEVICE;
    let (_, offset) =
        native_fault_site_in_entry(&mut machine, BASE, BASE + 4, RecoverySiteKind::GuestAccess);
    machine.host.exits.push(device_fault(offset, DEVICE, false));
    machine.step().unwrap();
    assert!(machine.device_site(BASE + 4));
    machine.set_step_timing(true);
    machine.cpu.pc = BASE + 4;
    machine.host.exits.push(exit(BASE + 4, ExitKind::System));
    machine.step().unwrap();
    assert_eq!(machine.cpu.pc, BASE + 8);
    let timing = machine.step_timing().unwrap();
    assert_eq!(timing.exits[step_timing::DEVICE_SITE_SLOT].0, 1);
    assert_eq!(timing.exits[ExitKind::System as usize].0, 0);
    let exits = machine.take_system_exits();
    let keys: Vec<_> = exits.counted().map(|exit| (exit.key, exit.count)).collect();
    assert_eq!(keys, [(step_timing::DEVICE_SITE_KEY, 1)]);
    // The access is charged to its device page too, with no lock to wait.
    let pages: Vec<_> = exits
        .devices
        .counted()
        .map(|exit| (exit.page, exit.count, exit.waited))
        .collect();
    assert_eq!(pages, [(step_timing::device_page(DEVICE), 1, 0)]);
    assert_eq!(exits.devices.forwarded, 0);
}

#[test]
fn a_device_site_that_reaches_ram_runs_natively_and_keeps_writeback_and_size() {
    // strb w0, [x1], #1; ret
    let words = [0x3800_1420, 0xd65f_03c0];
    let mut machine = board(&words, Registers::default(), Vec::new());
    machine.cpu.state.x[0] = 0x1234_5678;
    machine.cpu.state.x[1] = DEVICE;
    let (_, offset) =
        native_fault_site_in_entry(&mut machine, BASE, BASE, RecoverySiteKind::GuestAccess);
    machine.host.exits.push(device_fault(offset, DEVICE, true));
    machine.step().unwrap();
    assert_eq!(machine.devices.accesses, [(DEVICE, 1, Some(0x1234_5678))]);
    assert_eq!(
        (machine.cpu.state.x[1], machine.cpu.pc),
        (DEVICE + 1, BASE + 4)
    );
    // The stub completes the next store and its writeback, a byte wide (the
    // register takes the value's low byte, as from the fault path).
    machine.cpu.pc = BASE;
    machine.host.exits.push(exit(BASE, ExitKind::System));
    machine.step().unwrap();
    assert_eq!(
        machine.devices.accesses[1],
        (DEVICE + 1, 1, Some(0x1234_5678))
    );
    assert_eq!(
        (machine.cpu.state.x[1], machine.cpu.pc),
        (DEVICE + 2, BASE + 4)
    );
    // An access to RAM from the site runs as a block of its own, natively.
    machine.cpu.state.x[1] = BASE + 0x800;
    machine.cpu.pc = BASE;
    machine.host.exits.push(exit(BASE, ExitKind::System));
    machine
        .host
        .exits
        .push(exit(BASE + 4, ExitKind::Fallthrough));
    machine.step().unwrap();
    assert!(machine.host.exits.is_empty());
    assert_eq!(machine.devices.accesses.len(), 2);
    assert_eq!(machine.cpu.pc, BASE + 4);
    assert!(
        machine
            .code.blocks
            .iter()
            .any(|(_, block)| block.pc == BASE && machine.variants[block.variant.0 as usize].single)
    );
}

#[test]
fn promoted_device_site_uses_only_current_canonical_ram_admission() {
    use crate::memory::{ReservedSystemRamRange, system_ram::fixture};
    use core::{cell::UnsafeCell, ptr::NonNull};

    const FRAME_BYTES: usize = 16 * 1024;
    const FRAME: u64 = BASE + 0x1_0000;
    const RAM: u64 = FRAME + 0x800;
    const OLD_ONLY: u64 = BASE + 0x800;
    const HOLE: u64 = DEVICE + 0x1000;
    const VALUE: u64 = 0x78;

    #[repr(align(4096))]
    struct Storage(UnsafeCell<[u8; FRAME_BYTES]>);
    // SAFETY: the registered fixture retains the initialized storage and
    // accesses its bytes through the canonical synchronization protocol.
    unsafe impl Sync for Storage {}

    let storage = Arc::new(Storage(UnsafeCell::new([0x53; FRAME_BYTES])));
    let range = ReservedSystemRamRange {
        source_pa: FRAME,
        host_pa: 0x80_0000,
        host_base: NonNull::new(storage.0.get().cast()).unwrap(),
        bytes: FRAME_BYTES,
    };
    // SAFETY: the pinned storage remains with the fixture and has no other
    // writer. FakeHost's legacy bytes are a distinct allocation.
    let mut ram = unsafe { fixture(&[range], storage) };
    ram.enable_descriptor_frame_leases().unwrap();
    let ram = Arc::new(ram);
    let generation = ram.generation().unwrap();

    // strb w0, [x1], #1; ret. Learn the actual promotion through its fault.
    let mut machine = board(
        &[0x3800_1420, 0xd65f_03c0, 0xa900_2909, 0xd65f_03c0],
        Registers::default(),
        Vec::new(),
    );
    machine.cpu.state.x[0] = VALUE;
    machine.cpu.state.x[1] = DEVICE;
    let (_, offset) =
        native_fault_site_in_entry(&mut machine, BASE, BASE, RecoverySiteKind::GuestAccess);
    machine.host.exits.push(device_fault(offset, DEVICE, true));
    machine.step().unwrap();
    assert_eq!(machine.devices.accesses, [(DEVICE, 1, Some(VALUE))]);
    let site = machine.device_sites[&BASE];
    machine.cpu.pc = BASE;
    let Ok(Fetched::Block(stub)) = machine.fetch_block(BASE, false) else {
        panic!("an installed device stub");
    };
    assert!(machine.device_stub(stub, BASE));
    let retired = machine.cpu.retired;

    machine.host.ram[(OLD_ONLY - BASE) as usize] = 0x5a;
    ram.model_set_nonresident(FRAME);
    {
        let _execution = ram.cpu_execution_loan().unwrap();
        let mut tables = SiteTables {
            ram: &ram,
            descriptors: ram.descriptor_reader(),
            refusal: None,
        };
        assert_eq!(mmu::TableMemory::read_descriptor(&mut tables, RAM), None);
        assert_eq!(
            tables.refusal,
            Some(RequirementKind::NonresidentRam { frame: FRAME })
        );
    }
    for (address, expected_generation, missing) in [
        (DEVICE + 1, generation + 1, None),
        (OLD_ONLY, generation, None),
        (RAM, generation, Some(FRAME)),
    ] {
        machine.host.device_site_ram = Some(DeviceSiteRam {
            ram: Arc::clone(&ram),
            expected_generation: Some(expected_generation),
        });
        machine.cpu.state.x[1] = address;
        machine.host.exits.push(exit(BASE, ExitKind::System));
        let Err(Stop::Unsupported(report)) = machine.step() else {
            panic!("promoted access at {address:#x} must stop before an effect");
        };
        match missing {
            Some(frame) => assert_eq!(report.kind, RequirementKind::NonresidentRam { frame }),
            None => assert!(matches!(report.kind, RequirementKind::NativePreparation(_))),
        }
        assert_eq!(report.pc, BASE);
        assert_eq!(machine.cpu.pc, BASE);
        assert_eq!(machine.cpu.retired, retired);
        assert_eq!(machine.cpu.state.x[1], address);
        assert_eq!(machine.devices.accesses.len(), 1);
        assert_eq!(machine.device_sites[&BASE], site);
    }
    assert_eq!(machine.host.ram[(OLD_ONLY - BASE) as usize], 0x5a);

    // Neither RAM layout covers HOLE. Complete one bus operation, then
    // retry the original operand after restoration.
    machine.cpu.state.x[1] = HOLE;
    machine.host.exits.push(exit(BASE, ExitKind::System));
    machine.step().unwrap();
    assert_eq!(machine.cpu.pc, BASE + 4);
    assert_eq!(machine.cpu.retired, retired + 1);
    assert_eq!(machine.cpu.state.x[1], HOLE + 1);
    assert_eq!(
        machine.devices.accesses,
        [(DEVICE, 1, Some(VALUE)), (HOLE, 1, Some(VALUE))]
    );

    ram.model_set_resident(FRAME);
    machine.cpu.pc = BASE;
    machine.cpu.state.x[1] = RAM;
    machine.host.exits.push(exit(BASE, ExitKind::System));
    machine
        .host
        .exits
        .push(exit(BASE + 4, ExitKind::Fallthrough));
    machine.step().unwrap();
    assert!(machine.host.exits.is_empty());
    assert_eq!(machine.cpu.pc, BASE + 4);
    assert_eq!(machine.cpu.retired, retired + 2);
    assert_eq!(machine.devices.accesses.len(), 2);
    assert!(machine.code.blocks.iter().any(|(_, block)| {
        block.pc == BASE && machine.variants[block.variant.0 as usize].single
    }));

    // stp x9, x10, [x8]. A live service must stop both mailbox stores.
    let pc = BASE + 8;
    machine.learn_device_site(pc);
    machine.cpu.pc = pc;
    machine.cpu.state.x[8] = HOLE;
    machine.cpu.state.x[9] = VALUE;
    machine.cpu.state.x[10] = VALUE + 1;
    let site = machine.device_sites[&pc];
    let retired = machine.cpu.retired;
    let service = ram.try_service().unwrap().unwrap();
    machine.host.exits.push(exit(pc, ExitKind::System));
    let Err(Stop::Unsupported(report)) = machine.step() else {
        panic!("RAM service must prevent the promoted pair store");
    };
    assert_eq!(report.kind, RequirementKind::DeviceSiteRamServiceBusy);
    assert_eq!(report.pc, pc);
    assert_eq!(machine.cpu.pc, pc);
    assert_eq!(machine.cpu.retired, retired);
    assert_eq!(machine.cpu.state.x[8], HOLE);
    assert_eq!(machine.devices.accesses.len(), 2);
    assert_eq!(machine.device_sites[&pc], site);
    drop(service);

    machine.host.exits.push(exit(pc, ExitKind::System));
    machine.step().unwrap();
    assert_eq!(machine.cpu.pc, pc + 4);
    assert_eq!(machine.cpu.retired, retired + 1);
    assert_eq!(
        &machine.devices.accesses[2..],
        &[(HOLE, 8, Some(VALUE)), (HOLE + 8, 8, Some(VALUE + 1))]
    );
}
