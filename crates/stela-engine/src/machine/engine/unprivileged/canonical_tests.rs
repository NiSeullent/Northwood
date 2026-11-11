// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::{
    DeviceError, DeviceSiteRam, Devices, Lines, Machine, RequirementKind, Stop,
    tests::{BASE, FakeHost, board},
};
use crate::{
    machine::system::Register as SystemRegister,
    memory::{RegisteredSystemRam, ReservedSystemRamRange, system_ram::fixture},
};
use alloc::sync::Arc;
use core::{cell::UnsafeCell, ptr::NonNull};

const LDTR: u32 = 0xf840_0841; // ldtr x1, [x2]
const STTR: u32 = 0xf800_0841; // sttr x1, [x2]
const FRAME_BYTES: usize = 16 * 1024;
const SECOND_FRAME: u64 = BASE + FRAME_BYTES as u64;
const FULL_BYTES: usize = 2 * FRAME_BYTES;
const PARTIAL: u64 = BASE + FULL_BYTES as u64;
const TABLE: u64 = BASE + 0x1000;

#[derive(Default)]
struct RecordingDevices {
    accesses: usize,
}

impl Devices for RecordingDevices {
    fn read(&mut self, _: usize, _: u64, _: u8, _: u64) -> Result<u64, DeviceError> {
        self.accesses += 1;
        Ok(0)
    }

    fn write(&mut self, _: usize, _: u64, _: u8, _: u64, _: u64) -> Result<(), DeviceError> {
        self.accesses += 1;
        Ok(())
    }

    fn lines(&mut self, _: usize, _: u64) -> Lines {
        Lines::default()
    }
}

fn registered_machine() -> (
    Machine<FakeHost, RecordingDevices>,
    Arc<RegisteredSystemRam>,
) {
    #[repr(align(4096))]
    struct Storage(UnsafeCell<[u8; FULL_BYTES + 16]>);
    // SAFETY: only the registered owner accesses the pinned, initialized bytes.
    unsafe impl Sync for Storage {}

    let storage = Arc::new(Storage(UnsafeCell::new([0x53; FULL_BYTES + 16])));
    let host = storage.0.get().cast::<u8>();
    let ranges = [
        ReservedSystemRamRange {
            source_pa: BASE,
            host_pa: 0x80_0000,
            host_base: NonNull::new(host).unwrap(),
            bytes: FULL_BYTES,
        },
        ReservedSystemRamRange {
            source_pa: PARTIAL,
            host_pa: 0x90_0000,
            // SAFETY: the final 16 bytes are part of this pinned allocation.
            host_base: NonNull::new(unsafe { host.add(FULL_BYTES) }).unwrap(),
            bytes: 16,
        },
    ];
    // SAFETY: the canonical owner keeps both disjoint ranges alive and is
    // their only writer. The FakeHost's legacy bytes are a separate allocation.
    let mut ram = unsafe { fixture(&ranges, storage) };
    ram.enable_descriptor_frame_leases().unwrap();
    let ram = Arc::new(ram);

    // A 39-bit VA and 4 KiB granule. An L1 block maps VA 0–1 GiB to BASE
    // as Normal memory with EL0 read/write. Only canonical RAM has the PTE.
    let descriptor = BASE | 1 | 1 << 10 | 1 << 6;
    // SAFETY: the descriptor's staging bytes are private and disjoint.
    unsafe { ram.publish_physical(TABLE, &descriptor.to_le_bytes()) }.unwrap();
    let mut machine = board(&[], RecordingDevices::default(), Vec::new());
    machine.host.device_site_ram = Some(DeviceSiteRam {
        ram: Arc::clone(&ram),
        expected_generation: Some(ram.generation().unwrap()),
    });
    let registers = &mut machine.cpu.registers;
    registers.set(SystemRegister::TCR_EL1, 0, 25 | 1 << 23 | 5 << 32);
    registers.set(SystemRegister::TTBR0_EL1, 0, TABLE);
    registers.set(SystemRegister::MAIR_EL1, 0, 0xff);
    registers.set(SystemRegister::SCTLR_EL1, 0, 1);
    machine.derived_changed();
    (machine, ram)
}

fn read<const N: usize>(ram: &RegisteredSystemRam, physical: u64) -> [u8; N] {
    let mut bytes = [0; N];
    // SAFETY: the output is private and disjoint from live registered RAM.
    unsafe { ram.read_physical(physical, &mut bytes) }.unwrap();
    bytes
}

#[test]
fn ldtr_and_sttr_use_current_registered_ram() {
    const OFFSET: usize = 0x800;
    const INITIAL: u64 = 0x10ad_20ad_30ad_40ad;
    const STORED: u64 = 0x519a_62ab_73bc_84cd;
    let (mut machine, ram) = registered_machine();
    let physical = BASE + OFFSET as u64;
    machine.host.ram[OFFSET..OFFSET + 8].fill(0x6b);
    // SAFETY: the input bytes are private and disjoint from registered RAM.
    unsafe { ram.publish_physical(physical, &INITIAL.to_le_bytes()) }.unwrap();
    machine.cpu.state.x[2] = OFFSET as u64;
    let pc = machine.cpu.pc;

    machine.system_instruction(LDTR).unwrap();
    assert_eq!(machine.cpu.state.x[1], INITIAL);
    assert_eq!(machine.cpu.pc, pc + 4);

    machine.cpu.state.x[1] = STORED;
    machine.system_instruction(STTR).unwrap();
    assert_eq!(read::<8>(&ram, physical), STORED.to_le_bytes());
    assert_eq!(machine.host.ram[OFFSET..OFFSET + 8], [0x6b; 8]);
    assert_eq!(machine.devices.accesses, 0);
    assert_eq!(machine.cpu.pc, pc + 8);
}

#[test]
fn ldtr_and_sttr_refuse_absent_stale_and_partial_registered_operands() {
    let (mut machine, ram) = registered_machine();
    let generation = ram.generation().unwrap();
    let pc = machine.cpu.pc;
    let retired = machine.cpu.retired;
    ram.model_set_nonresident(SECOND_FRAME);

    for (physical, expected_generation, absent) in [
        (SECOND_FRAME + 0x800, generation, Some(SECOND_FRAME)),
        (BASE + 0x800, generation + 1, None),
        (PARTIAL + 12, generation, None),
        (PARTIAL + 0x1000, generation, None),
    ] {
        let offset = (physical - BASE) as usize;
        machine.host.ram[offset..offset + 8].fill(0x6b);
        machine
            .host
            .device_site_ram
            .as_mut()
            .unwrap()
            .expected_generation = Some(expected_generation);
        machine.cpu.state.x[2] = offset as u64;
        for instruction in [LDTR, STTR] {
            machine.cpu.state.x[1] = 0x1032_5476_98ba_dcfe;
            let before = machine.cpu.state.clone();
            let Err(Stop::Unsupported(report)) = machine.system_instruction(instruction) else {
                panic!("the access at {physical:#x} must stop before an effect");
            };
            match absent {
                Some(frame) => assert_eq!(report.kind, RequirementKind::NonresidentRam { frame }),
                None => assert!(matches!(report.kind, RequirementKind::NativePreparation(_))),
            }
            assert_eq!(report.pc, pc);
            assert_eq!(machine.cpu.pc, pc);
            assert_eq!(machine.cpu.retired, retired);
            assert_eq!(machine.cpu.state, before);
            assert_eq!(machine.host.ram[offset..offset + 8], [0x6b; 8]);
            assert_eq!(machine.devices.accesses, 0);
        }
    }

    ram.model_set_resident(SECOND_FRAME);
    assert_eq!(read::<8>(&ram, SECOND_FRAME + 0x800), [0x53; 8]);
    assert_eq!(read::<8>(&ram, BASE + 0x800), [0x53; 8]);
    assert_eq!(read::<4>(&ram, PARTIAL + 12), [0x53; 4]);
}

#[test]
fn sttr_preflights_the_second_page_before_writing_the_first() {
    let (mut machine, ram) = registered_machine();
    let address = SECOND_FRAME - 4;
    let offset = (address - BASE) as usize;
    let value = 0x1c2d_3e4f_5061_7283u64;
    machine.host.ram[offset..offset + 8].fill(0x6b);
    machine.cpu.state.x[1] = value;
    machine.cpu.state.x[2] = offset as u64;
    let pc = machine.cpu.pc;
    let retired = machine.cpu.retired;
    ram.model_set_nonresident(SECOND_FRAME);

    let Err(Stop::Unsupported(report)) = machine.system_instruction(STTR) else {
        panic!("the second page must stop the whole store before writing");
    };
    assert_eq!(
        report.kind,
        RequirementKind::NonresidentRam {
            frame: SECOND_FRAME
        }
    );
    assert_eq!(report.pc, pc);
    assert_eq!(machine.cpu.pc, pc);
    assert_eq!(machine.cpu.retired, retired);
    assert_eq!(read::<4>(&ram, address), [0x53; 4]);
    assert_eq!(machine.host.ram[offset..offset + 8], [0x6b; 8]);
    assert_eq!(machine.devices.accesses, 0);

    ram.model_set_resident(SECOND_FRAME);
    assert_eq!(read::<8>(&ram, address), [0x53; 8]);
    machine.system_instruction(STTR).unwrap();
    assert_eq!(read::<8>(&ram, address), value.to_le_bytes());
    assert_eq!(machine.host.ram[offset..offset + 8], [0x6b; 8]);
    assert_eq!(machine.devices.accesses, 0);
    assert_eq!(machine.cpu.pc, pc + 4);
}
