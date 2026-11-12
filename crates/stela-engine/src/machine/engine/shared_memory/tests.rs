use super::*;
use crate::machine::{
    cpu::Reservation,
    engine::{
        NoDevices,
        tests::{FakeHost, board_on},
    },
    system::Register,
};
use crate::system::events::EventDomain;
use stela_compiler::context::memory::{
    MemoryOrder, MemoryRequest, MemoryRequirements,
    exclusive::{ExclusiveData, ExclusiveOperation, ExclusiveShape},
    flags,
};
const VA: u64 = 0x4000;
const PA: u64 = 0x14000;
pub(super) fn machine() -> (
    Machine<FakeHost, NoDevices>,
    Arc<RegisteredSystemRam>,
    EventDomain,
) {
    let (cpu, profile, missing, ram) = policy::tests::fixture();
    let mut machine = board_on(stela_contracts::Chip::AppleM1, &[], NoDevices, Vec::new());
    machine.cpu = cpu;
    machine.profile = profile;
    machine.profile_missing = missing;
    (machine, ram, EventDomain::new())
}
/// POSIX byte/native-scalar service fixture. Its prepared legacy
/// entry is not a production CPL0 proof and cannot enter NativeRun.
fn transfer(cpu: &mut crate::machine::Cpu, operation: ExclusiveOperation) -> ExclusiveData {
    let memory = cpu.canonical_memory.as_mut().unwrap();
    let mut payload = ExclusiveData {
        operation: operation as u32,
        order: MemoryOrder::Relaxed as u32,
        shape: ExclusiveShape::Scalar as u32,
        value: if operation == ExclusiveOperation::Store {
            [7, 0]
        } else {
            [0, 0]
        },
        ..ExclusiveData::default()
    };
    let request = MemoryRequest {
        address: VA,
        pc: cpu.pc,
        bytes: 8,
        flags: flags::EXCLUSIVE
            | if operation == ExclusiveOperation::Store {
                flags::WRITE
            } else {
                0
            },
        data: core::ptr::from_mut(&mut payload).cast(),
    };
    let entry = memory
        .memory
        .prepare(MemoryRequirements::default())
        .unwrap();
    // SAFETY: the admitted POSIX fixture owns initialized private
    // typed staging and keeps its source policy/native byte owner.
    let result = unsafe { entry.with_checked_view(|view| (view.callback)(view, &request)) };
    assert_eq!(result, Ok(MemoryStatus::Success as u32));
    payload
}
#[test]
fn linear_conversion_moves_the_event_register_and_refuses_live_legacy_reservations() {
    let (mut machine, ram, events) = machine();
    machine.cpu.event = true;
    machine.cpu.monitor = Some(Reservation {
        granule: PA,
        bytes: [0; 64],
    });
    assert_eq!(
        machine.install_shared_memory_for_processors(Arc::clone(&ram), &events),
        Err(MemoryStatus::Unsupported)
    );
    assert!(
        machine.cpu.event && machine.cpu.monitor.is_some() && !machine.cpu.has_canonical_memory()
    );
    machine.cpu.monitor = None;
    machine
        .install_shared_memory_for_processors(ram, &events)
        .unwrap();
    assert!(!machine.cpu.event && machine.cpu.event_is_set());
    assert!(machine.cpu.consume_event());
    assert!(!machine.cpu.consume_event());
    let producer = machine
        .cpu
        .canonical_memory
        .as_ref()
        .unwrap()
        .event_endpoint()
        .producer();
    assert!(producer.signal_local());
    assert!(machine.cpu.event_is_set());
    assert!(machine.cpu.consume_event());
    assert!(!machine.cpu.event_is_set());
    machine.cpu.signal_event();
    machine.cpu.signal_event();
    assert!(machine.cpu.consume_event());
    assert!(!machine.cpu.consume_event());
    // A producer cannot keep the PE endpoint alive after its owner.
    drop(machine);
    assert!(!producer.signal_local());
}
#[test]
fn clrex_clears_the_real_local_reservation_and_keeps_its_global_event_mark() {
    let (mut machine, ram, events) = machine();
    machine
        .install_shared_memory_for_processors(Arc::clone(&ram), &events)
        .unwrap();
    assert_eq!(
        transfer(&mut machine.cpu, ExclusiveOperation::Load).value,
        [0; 2]
    );
    machine.cpu.clear_exclusive().unwrap();
    assert!(!machine.cpu.event_is_set());
    // The independent global mark still opens and signals on canonical ABA.
    // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
    // registered.
    unsafe { ram.publish_physical(PA, &1u64.to_le_bytes()) }.unwrap();
    // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
    // registered.
    unsafe { ram.publish_physical(PA, &0u64.to_le_bytes()) }.unwrap();
    assert!(machine.cpu.consume_event());
    assert!(!machine.cpu.consume_event());
    assert_eq!(
        transfer(&mut machine.cpu, ExclusiveOperation::Store).status,
        1
    );
    let mut value = [0; 8];
    // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
    // registered.
    unsafe { ram.read_physical(PA, &mut value) }.unwrap();
    assert_eq!(value, [0; 8]);
}
#[test]
fn eret_clears_the_same_reservation_and_publishes_the_same_event_endpoint() {
    let (mut machine, ram, events) = machine();
    machine
        .install_shared_memory_for_processors(Arc::clone(&ram), &events)
        .unwrap();
    transfer(&mut machine.cpu, ExclusiveOperation::Load);
    machine.cpu.registers.set_raw(Register::SPSR_EL1, 0, 5);
    machine.cpu.exception_return(0x8000).unwrap();
    assert_eq!(machine.cpu.pc, 0x8000);
    assert!(machine.cpu.consume_event());
    assert!(!machine.cpu.event); // legacy scalar storage was never copied back
    assert_eq!(
        transfer(&mut machine.cpu, ExclusiveOperation::Store).status,
        1
    );
}
#[test]
fn failed_memory_owner_refuses_eret_before_any_cpu_or_event_effect() {
    let (mut machine, ram, events) = machine();
    machine
        .install_shared_memory_for_processors(ram, &events)
        .unwrap();
    let memory = machine.cpu.canonical_memory.as_mut().unwrap();
    // Corrupt finalization is a terminal domain failure, not a source fault.
    // SAFETY: the callback runs once on this thread and touches only the checked view.
    let result = unsafe { memory.memory.with_checked_view(|view| view.version = 0) };
    assert_eq!(result, Err(MemoryStatus::Domain));
    let before = (
        machine.cpu.state.clone(),
        machine.cpu.pc,
        machine.cpu.pstate,
        machine.cpu.retired,
    );
    assert_eq!(
        machine.cpu.exception_return(0x8000),
        Err(MemoryStatus::Domain)
    );
    assert_eq!(
        (
            machine.cpu.state.clone(),
            machine.cpu.pc,
            machine.cpu.pstate,
            machine.cpu.retired
        ),
        before
    );
    assert!(!machine.cpu.event_is_set());
}
#[test]
fn clrex_sevl_and_wfe_use_the_cpu_monitor() {
    let (mut machine, ram, events) = machine();
    machine
        .install_shared_memory_for_processors(ram, &events)
        .unwrap();
    transfer(&mut machine.cpu, ExclusiveOperation::Load);
    const CLREX: u32 = 0xd503_305f;
    const SEVL: u32 = 0xd503_20bf;
    const WFE: u32 = 0xd503_205f;
    let pc = machine.cpu.pc;
    machine.system_instruction(CLREX).unwrap();
    assert_eq!(machine.cpu.pc, pc + 4);
    assert_eq!(
        transfer(&mut machine.cpu, ExclusiveOperation::Store).status,
        1
    );
    machine.system_instruction(SEVL).unwrap();
    assert!(machine.cpu.event_is_set() && !machine.cpu.event);
    machine.system_instruction(WFE).unwrap();
    assert!(!machine.cpu.event_is_set());
    assert_eq!(machine.cpu.pc, pc + 12);
}
#[test]
fn dropped_monitor_preparation_does_not_clear_the_real_reservation() {
    let (mut machine, ram, events) = machine();
    machine
        .install_shared_memory_for_processors(ram, &events)
        .unwrap();
    transfer(&mut machine.cpu, ExclusiveOperation::Load);
    drop(machine.cpu.prepare_clear_exclusive().unwrap());
    assert_eq!(
        transfer(&mut machine.cpu, ExclusiveOperation::Store).status,
        0
    );
}
fn clock_domain(
    machine: &mut Machine<FakeHost, NoDevices>,
) -> stela_contracts::bus::LifecycleState {
    use super::super::ProcessorBinding;
    use stela_contracts::processor::ProcessorTarget;
    let active = stela_contracts::bus::LifecycleState {
        powered: true,
        clocked: true,
        quiesced: false,
        reset: false,
        retention: false,
    };
    machine
        .bind_processor_power(&[ProcessorBinding {
            domain: 1,
            target: ProcessorTarget::Core { pe: 0 },
            initial: active,
        }])
        .unwrap();
    active
}
#[test]
fn kept_pause_commits_the_same_monitor_before_controller_ack() {
    use super::super::ProcessorChange;
    let (mut machine, ram, events) = machine();
    machine
        .install_shared_memory_for_processors(ram, &events)
        .unwrap();
    let active = clock_domain(&mut machine);
    transfer(&mut machine.cpu, ExclusiveOperation::Load);
    let state = machine.cpu.state.clone();
    let request = [ProcessorChange {
        domain: 1,
        state: stela_contracts::bus::LifecycleState {
            clocked: false,
            ..active
        },
    }];
    let held = machine.processor_write().apply(&request).unwrap();
    assert_eq!(held.paused(0), Some(true));
    drop(held); // the controller ACK is inside the owner lease
    assert_eq!(machine.cpu.state, state);
    let resumed = machine
        .processor_write()
        .apply(&[ProcessorChange {
            domain: 1,
            state: active,
        }])
        .unwrap();
    drop(resumed);
    assert_eq!(
        transfer(&mut machine.cpu, ExclusiveOperation::Store).status,
        1
    );
}
#[test]
fn failed_pause_preparation_changes_no_cpu_or_controller_state() {
    use super::super::{ProcessorChange, ProcessorPowerError};
    let (mut machine, ram, events) = machine();
    machine
        .install_shared_memory_for_processors(ram, &events)
        .unwrap();
    let active = clock_domain(&mut machine);
    let memory = machine.cpu.canonical_memory.as_mut().unwrap();
    assert_eq!(
        // SAFETY: the callback runs once on this thread and touches only the checked view.
        unsafe { memory.memory.with_checked_view(|view| view.version = 0) },
        Err(MemoryStatus::Domain)
    );
    let state = (
        machine.cpu.state.clone(),
        machine.cpu.pc,
        machine.cpu.pstate,
        machine.cpu.retired,
    );
    let request = [ProcessorChange {
        domain: 1,
        state: stela_contracts::bus::LifecycleState {
            clocked: false,
            ..active
        },
    }];
    let result = machine.processor_write().apply(&request);
    assert!(matches!(
        result,
        Err(ProcessorPowerError::CanonicalMemory {
            pe: 0,
            status: MemoryStatus::Domain
        })
    ));
    assert_eq!(
        (
            machine.cpu.state.clone(),
            machine.cpu.pc,
            machine.cpu.pstate,
            machine.cpu.retired
        ),
        state
    );
    assert_eq!(machine.processors().paused(0), Some(false));
}
#[test]
fn converted_cpu_cannot_reenter_the_serial_direct_ram_scheduler() {
    let (mut machine, ram, events) = machine();
    machine
        .install_shared_memory_for_processors(ram, &events)
        .unwrap();
    let before = (
        machine.cpu.state.clone(),
        machine.cpu.pc,
        machine.cpu.pstate,
        machine.cpu.retired,
    );
    assert!(matches!(machine.run(1), super::super::Stop::Unsupported(_)));
    assert_eq!(
        (
            machine.cpu.state.clone(),
            machine.cpu.pc,
            machine.cpu.pstate,
            machine.cpu.retired
        ),
        before
    );
}
#[test]
fn unjoined_power_loss_and_reset_refuse_before_dropping_the_live_owner() {
    let (mut machine, ram, events) = machine();
    machine
        .install_shared_memory_for_processors(ram, &events)
        .unwrap();
    machine.cpu.apple.cyc_ovrd = 3 << 24;
    let before = (
        machine.cpu.state.clone(),
        machine.cpu.pc,
        machine.cpu.pstate,
        machine.cpu.retired,
    );
    assert!(machine.system_instruction(0xd503_207f).is_err());
    assert!(machine.waiting.is_none());
    assert!(machine.warm_reset().is_err());
    assert_eq!(
        (
            machine.cpu.state.clone(),
            machine.cpu.pc,
            machine.cpu.pstate,
            machine.cpu.retired
        ),
        before
    );
    assert!(machine.cpu.has_canonical_memory());
    let producer = machine
        .cpu
        .canonical_memory
        .as_ref()
        .unwrap()
        .event_endpoint()
        .producer();
    assert!(producer.signal_local());
    assert!(machine.cpu.consume_event());
}
#[test]
fn unregistered_native_inventory_is_never_a_stopped_conversion_receipt() {
    let (mut machine, ram, events) = machine();
    machine
        .bind_native_workers(&[super::super::publication::WorkerBinding {
            pe: 0,
            worker: super::super::publication::WorkerId(0),
        }])
        .unwrap();
    assert_eq!(
        machine.install_shared_memory_for_processors(ram, &events),
        Err(MemoryStatus::Unsupported)
    );
    assert!(!machine.cpu.has_canonical_memory());
}
/// The home PE of a parallel board holds canonical memory. Its SVCs from
/// EL0 of the host make records and complete natively, leaving the
/// canonical reservation and event endpoint alone as `Cpu::take` does; its
/// ERETs make none, since the reservation they clear lives in the shared
/// registry native code cannot reach, and a reservation taken before the
/// SVC still answers the store-exclusive after it.
#[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
#[test]
fn a_pe_with_canonical_memory_completes_svc_natively_and_eret_in_the_engine() {
    use super::super::mirrors::TRANSITIONS;
    use super::super::transitions::tests::run_native;
    use crate::machine::engine::{ExitKind, GuestAddress, Pstate};
    use crate::machine::exception::{Class, syndrome};
    use stela_compiler::x86_64::transition::{self, EXCEPTION_RETURN, SUPERVISOR_CALL};
    const VECTORS: u64 = 0x8000;
    const USER: u64 = 0x9000;
    const ERET: u32 = 0xd69f_03e0;
    const SVC: u32 = 0xd400_0001 | 0x80 << 5;
    let (mut machine, ram, events) = machine();
    machine
        .install_shared_memory_for_processors(Arc::clone(&ram), &events)
        .unwrap();
    machine.cpu.pstate = Pstate::reset(2);
    let registers = &mut machine.cpu.registers;
    registers.set(Register::HCR_EL2, 0, 1 << 34 | 1 << 27);
    registers.set(Register::VBAR_EL2, 0, VECTORS);
    machine.sync_regime();
    let eret = |machine: &mut Machine<FakeHost, NoDevices>| {
        machine.cpu.pc = 0x8800;
        let registers = &mut machine.cpu.registers;
        registers.set(Register::SPSR_EL2, 0, 0);
        registers.set(Register::ELR_EL2, 0, USER);
        machine.system_instruction(ERET).unwrap();
        assert_eq!((machine.cpu.pc, machine.cpu.pstate.el), (USER, 0));
    };
    for _ in 0..2 {
        eret(&mut machine);
        machine.finish_transition(1);
        machine.system_instruction(SVC).unwrap();
        machine.finish_transition(1);
        assert_eq!(
            (machine.cpu.pc, machine.cpu.pstate.el),
            (VECTORS + 0x400, 2)
        );
    }
    let table = &machine.mirrors.transitions;
    assert_eq!(
        table.records[0][SUPERVISOR_CALL].generation,
        table.generation
    );
    assert_eq!(table.records[2][EXCEPTION_RETURN].generation, 0);
    // Back at EL0, a reservation open across a native SVC.
    eret(&mut machine);
    assert!(machine.cpu.consume_event());
    transfer(&mut machine.cpu, ExclusiveOperation::Load);
    let svc = transition::supervisor_call_thunk(
        GuestAddress(USER),
        syndrome(Class::SupervisorCall, 0x80) as u32,
        &TRANSITIONS,
    );
    let exit = run_native(&mut machine, &svc.code);
    assert_eq!(exit, (VECTORS + 0x400, ExitKind::Return as u64));
    assert_eq!(machine.cpu.pstate.el, 2);
    assert_eq!(machine.statistics.native_transitions[SUPERVISOR_CALL], 1);
    assert!(!machine.cpu.event_is_set());
    assert_eq!(
        transfer(&mut machine.cpu, ExclusiveOperation::Store).status,
        0
    );
}
