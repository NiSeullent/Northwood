//! Engine boundary refusal and authorized Before settlement tests.
use super::super::tests::{BASE, FakeHost, installed_code, machine, make_pending};
use super::super::*;
use stela_compiler::block::{
    HostGprSnapshot, HostSpillSnapshot, RecoveryQualificationError, RecoverySiteKind,
};

const PROGRAM: [u32; 2] = [0xf9400043, 0xd4000001]; // ldr x3,[x2]; svc #0
/// The admitted input [`PROGRAM`]'s block compiles from.
fn input() -> Vec<u8> {
    PROGRAM.into_iter().flat_map(u32::to_le_bytes).collect()
}
fn fixture() -> (Machine<FakeHost, NoDevices>, usize, usize) {
    let mut machine = machine(&PROGRAM);
    let Ok(Fetched::Block(index)) = machine.fetch_block(BASE, false) else {
        panic!("actual compiled block");
    };
    let offset = machine
        .code
        .blocks
        .recovery()
        .get_mut(&index)
        .unwrap()
        .test_rows()
        .into_iter()
        .find(|row| row.site.kind() == RecoverySiteKind::GuestAccess)
        .unwrap()
        .start;
    (machine, index, offset)
}
fn captured(offset: usize) -> NativeFault {
    NativeFault {
        origin: crate::machine::NativeFaultOrigin::Exception,
        offset,
        address: 0x40,
        write: false,
        trap: 14,
        error_code: 0,
        cause: FaultCause::Guest(mmu::Fault::stage1(mmu::FaultKind::Translation, 3)),
        registers: HostGprSnapshot::legacy([0; 16]),
        spills: HostSpillSnapshot::new(),
    }
}
fn terminal(
    mut machine: Machine<FakeHost, NoDevices>,
    index: usize,
    fault: NativeFault,
    reason: NativeFailureReason,
) {
    let state = machine.cpu.state.clone();
    let (pc, retired, counter_reads) = (
        machine.cpu.pc,
        machine.cpu.retired,
        machine.host.counter_reads,
    );
    let stopped = machine.native_fault(index, fault, 7, Some(1)).unwrap_err();
    assert!(
        matches!(&stopped, Stop::NativeFailure(failure) if failure.reason == reason),
        "{stopped:?}"
    );
    assert_eq!(machine.cpu.state, state);
    assert_eq!((machine.cpu.pc, machine.cpu.retired), (pc, retired));
    assert_eq!(machine.run(1), stopped);
    assert_eq!(
        machine.host.counter_reads, counter_reads,
        "poisoned frame never reenters scheduling"
    );
    assert_eq!(machine.cpu.state, state);
}
#[test]
fn exact_guest_access_preserves_before_pc_fault_and_chained_retirement() {
    let (mut machine, index, offset) = fixture();
    machine
        .native_fault(index, captured(offset), 7, Some(1))
        .unwrap();
    assert_eq!(machine.cpu.read(SystemRegister::ELR_EL1), BASE);
    assert_eq!(machine.cpu.read(SystemRegister::FAR_EL1), 0x40);
    assert_eq!(machine.cpu.retired, 7, "faulting load does not retire");
}
#[test]
fn frame_entry_fault_is_terminal_without_source_map_projection() {
    let (machine, index, _) = fixture();
    terminal(
        machine,
        index,
        captured(0),
        NativeFailureReason::InternalSite,
    );
}
#[test]
fn operand_site_does_not_turn_illegal_host_opcode_into_guest_data_abort() {
    let (machine, index, offset) = fixture();
    let mut fault = captured(offset);
    fault.trap = 6;
    fault.cause = FaultCause::InvalidOpcode;
    terminal(machine, index, fault, NativeFailureReason::TrapSiteMismatch);
}
#[test]
fn host_fetch_page_table_and_protection_faults_never_project_as_guest_access() {
    for bit in 3..64 {
        let error_code = 1u64 << bit;
        let (machine, index, offset) = fixture();
        let mut fault = captured(offset);
        fault.error_code = error_code;
        terminal(machine, index, fault, NativeFailureReason::TrapSiteMismatch);
    }
}
#[test]
fn portable_guest_tag_on_internal_frame_cannot_mint_projection_permission() {
    let (mut machine, index, _) = fixture();
    make_pending(&mut machine, index, &input());
    let recovery = machine.code.blocks.recovery();
    let mut owner = recovery.get_mut(&index).unwrap();
    let mut forged = owner.test_pending_rows_mut()[0].clone();
    forged.start = 0;
    forged.end = 1;
    forged.site = RecoverySiteKind::GuestAccess;
    owner.test_push_pending(forged);
    drop(owner);
    terminal(
        machine,
        index,
        captured(0),
        NativeFailureReason::Qualification(RecoveryQualificationError::RecoveryPlan),
    );
}
#[test]
fn portable_missing_guest_tag_is_refused_instead_of_falling_back_to_source_map() {
    let (mut machine, index, offset) = fixture();
    make_pending(&mut machine, index, &input());
    let recovery = machine.code.blocks.recovery();
    let mut owner = recovery.get_mut(&index).unwrap();
    owner
        .test_pending_rows_mut()
        .iter_mut()
        .find(|row| row.start == offset)
        .unwrap()
        .site = RecoverySiteKind::NativeInternal;
    drop(owner);
    terminal(
        machine,
        index,
        captured(offset),
        NativeFailureReason::Qualification(RecoveryQualificationError::RecoveryPlan),
    );
}

/// Keep impossible ADD/BTI injection cases separate from an emitted memory
/// operand reached by a real chain contract. FakeHost simulates the transfer;
/// the operand RIP and retirement come from the compiler-minted plan.
fn chained_load_case(branch: u32, btype: u8, landing_pad: bool) {
    use stela_compiler::x86_64::chain;
    let mut words = vec![branch, 0, 0];
    if landing_pad {
        words.push(0xd503245f);
    } // BTI C must retire before LDR.
    words.extend([0xf9400043, 0xd4000001]);
    let mut machine = machine(&words);
    let Ok(Fetched::Block(mut entry)) = machine.fetch_block(BASE, false) else {
        panic!("caller");
    };
    let target_pc = BASE + 12;
    let Ok(Fetched::Block(mut target)) = machine.fetch_block(target_pc, false) else {
        panic!("callee");
    };
    if landing_pad {
        // A page becomes guarded through a TLBI, which forgets its
        // translation.
        machine.forget_lookups(mmu::tlbi::Reach::Containing(BASE), None);
        // The remap is the fetch cache's alone: no refill walks the tables.
        machine.dropped_refills.clear();
        let slot = (target_pc / FETCH_PAGE) as usize % FETCH_CACHE;
        machine.fetch[slot].as_mut().unwrap().guarded = true;
        let Ok(Fetched::Block(current_entry)) = machine.fetch_block(BASE, false) else {
            panic!("guarded caller");
        };
        let Ok(Fetched::Block(current_target)) = machine.fetch_block(target_pc, false) else {
            panic!("guarded target");
        };
        entry = current_entry;
        target = current_target;
        let key = machine.current_key(target_pc).expect("a translated page");
        assert_eq!(
            machine
                .code
                .cache()
                .targets
                .entry(machine.code.table.slot(target_pc, key))
                .targets,
            0b0111
        );
    }
    let operand_pc = target_pc + u64::from(landing_pad) * INSTRUCTION;
    let row = machine
        .code
        .blocks
        .recovery()
        .get_mut(&target)
        .unwrap()
        .test_rows()
        .into_iter()
        .find(|row| {
            row.site.kind() == RecoverySiteKind::GuestAccess
                && row.retirement.instruction_pc().0 == operand_pc
        })
        .unwrap();
    assert_eq!(row.retirement.retired_before(), u16::from(landing_pad));
    let operand = row.start;
    if btype == 0 {
        machine.link(entry, target_pc, ExitKind::Direct);
        assert_eq!(machine.statistics.links, 1, "actual direct thunk installed");
    } else {
        machine.link_dynamic(entry, btype, target_pc);
        assert_eq!(
            machine.statistics.lookups, 1,
            "actual lookup thunk installed"
        );
    }
    let current = machine.chain.snapshot();
    machine.host.chained = Some((
        machine.chain.address() as *mut ChainControl,
        ChainControl {
            budget: CHAIN_LIMIT - 1,
            retired: 1,
            block: (u64::from(btype) << chain::BLOCK_BTYPE_SHIFT) | target as u64,
            linked: 1,
            ..current
        },
    ));
    let offset = machine.code.blocks[target]
        .code
        .0
        .wrapping_add(operand)
        .wrapping_sub(machine.code.blocks[entry].code.0);
    machine.host.exits.push(NativeExit::Fault(captured(offset)));
    machine.cpu.pc = BASE;
    let before = machine.cpu.retired;
    machine.step().unwrap();
    assert_eq!(machine.cpu.read(SystemRegister::ELR_EL1), operand_pc);
    assert_eq!(
        machine.cpu.read(SystemRegister::SPSR_EL1) >> 10 & 3,
        if landing_pad { 0 } else { u64::from(btype) }
    );
    assert_eq!(machine.cpu.retired, before + 1 + u64::from(landing_pad));
}
#[test]
fn direct_chain_to_first_load_keeps_completed_prefix_and_zero_btype() {
    chained_load_case(0x14000003, 0, false);
}
#[test]
fn unguarded_indirect_chain_fault_on_first_load_preserves_transfer_btype() {
    chained_load_case(0xd63f0100, 0b10, false);
}
#[test]
fn guarded_landing_pad_retires_and_clears_btype_before_later_load_fault() {
    chained_load_case(0xd63f0100, 0b10, true);
}

#[test]
fn load_before_native_terminal_adapters_keeps_checked_fault_plan() {
    for terminal_word in [0xd538_d088, 0xdac1_47f1] {
        let mut machine = machine(&[0xf9400043, terminal_word]);
        let Ok(Fetched::Block(index)) = machine.fetch_block(BASE, false) else {
            panic!("actual native adapter emission");
        };
        let offset = machine
            .code
            .blocks
            .recovery()
            .get_mut(&index)
            .unwrap()
            .test_rows()
            .into_iter()
            .find(|row| row.site.kind() == RecoverySiteKind::GuestAccess)
            .unwrap()
            .start;
        machine
            .native_fault(index, captured(offset), 0, None)
            .unwrap();
        assert_eq!(machine.cpu.read(SystemRegister::ELR_EL1), BASE);
        assert_eq!(machine.cpu.retired, 0);
    }
}

fn translated_fixture() -> (Machine<FakeHost, NoDevices>, usize, usize) {
    let mut machine = super::super::tests::translated_machine(&PROGRAM);
    let Ok(Fetched::Block(index)) = machine.fetch_block(BASE, false) else {
        panic!("translated emission");
    };
    let recovery = machine.code.blocks.recovery();
    let owner = recovery.get_mut(&index).unwrap();
    let row = owner
        .test_rows()
        .into_iter()
        .find(|row| row.site.kind() == RecoverySiteKind::TranslatedHelper)
        .unwrap();
    drop(owner);
    let offset = row.start;
    let access = row.site.translated_access().unwrap();
    assert_eq!(
        (
            access.bytes(),
            access.write(),
            access.unprivileged(),
            access.atomic()
        ),
        (8, false, false, false)
    );
    assert_eq!(row.end, offset + 1, "only exact CALL return token");
    assert_eq!(installed_code(&mut machine, index, offset, 2), [8, 0]);
    make_pending(&mut machine, index, &input());
    (machine, index, offset)
}
fn helper_capture(offset: usize) -> NativeFault {
    NativeFault {
        origin: NativeFaultOrigin::TranslatedHelper,
        trap: 0,
        error_code: 0,
        cause: FaultCause::NonCanonical,
        address: 0x0040_0000_0000_0000,
        ..captured(offset)
    }
}
#[test]
fn deliberate_translation_helper_return_checks_exact_descriptor_before_projection() {
    let (mut machine, index, offset) = translated_fixture();
    machine.cpu.state.x[2] = 0x0040_0000_0000_0000;
    machine
        .native_fault(index, helper_capture(offset), 7, None)
        .unwrap();
    assert_eq!(machine.cpu.read(SystemRegister::ELR_EL1), BASE);
    assert_eq!(
        machine.cpu.read(SystemRegister::FAR_EL1),
        0x0040_0000_0000_0000
    );
    assert_eq!(machine.cpu.retired, 7);
}
#[test]
fn helper_origin_vector_direction_and_return_token_are_independent_permissions() {
    for mutation in 0..6 {
        let (machine, index, offset) = translated_fixture();
        let mut fault = helper_capture(offset);
        let reason = match mutation {
            0 => {
                fault.origin = NativeFaultOrigin::Exception;
                fault.trap = 14;
                NativeFailureReason::TrapSiteMismatch
            }
            1 => {
                fault.trap = 6;
                NativeFailureReason::TrapSiteMismatch
            }
            2 => {
                fault.write = true;
                NativeFailureReason::TrapSiteMismatch
            }
            3 => {
                fault.offset += 1;
                NativeFailureReason::InternalSite
            }
            4 => {
                fault.cause = FaultCause::InvalidOpcode;
                NativeFailureReason::TrapSiteMismatch
            }
            _ => {
                fault.offset -= 1;
                NativeFailureReason::InternalSite
            }
        };
        terminal(machine, index, fault, reason);
    }
    let (machine, index, offset) = fixture();
    terminal(
        machine,
        index,
        helper_capture(offset),
        NativeFailureReason::TrapSiteMismatch,
    );
}
