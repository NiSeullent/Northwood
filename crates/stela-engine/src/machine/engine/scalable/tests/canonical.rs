// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    BASE, DeviceError, Devices, FakeHost, Machine, PTRUE_P0_B, Register, RequirementKind,
    STREAMING_VECTOR_BYTES, Stop, enable_streaming,
};
use crate::{
    machine::{
        engine::{DeviceSiteRam, tests::board_on},
        registers::Lines,
    },
    memory::{RegisteredSystemRam, ReservedSystemRamRange, system_ram::fixture},
};
use alloc::sync::Arc;
use core::{cell::UnsafeCell, ptr::NonNull};
use stela_compiler::context::memory::MemoryStatus;
use stela_contracts::Chip;

const PTRUE_P0_B_VL1: u32 = 0x2518_e020; // ptrue p0.b, vl1
const PTRUE_P0_B_VL2: u32 = 0x2518_e040; // ptrue p0.b, vl2
const ST1B_Z1_X5: u32 = 0xe400_e0a1; // st1b {z1.b}, p0, [x5]
const FRAME_BYTES: usize = 16 * 1024;
const FRAME: u64 = BASE + FRAME_BYTES as u64;
const FIRST: u64 = FRAME - 2;
const FULL_BYTES: usize = 2 * FRAME_BYTES;
const PARTIAL: u64 = BASE + FULL_BYTES as u64;
const TABLE: u64 = BASE + 0x1000;
const INITIAL: u8 = 0x53;
const LEGACY: u8 = 0x6b;

#[derive(Default)]
struct RecordingDevices {
    accesses: Vec<(u64, u8, Option<u64>)>,
}

impl Devices for RecordingDevices {
    fn read(&mut self, _: usize, physical: u64, bytes: u8, _: u64) -> Result<u64, DeviceError> {
        self.accesses.push((physical, bytes, None));
        Ok(0)
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

fn registered_machine() -> (
    Machine<FakeHost, RecordingDevices>,
    Arc<RegisteredSystemRam>,
) {
    #[repr(align(4096))]
    struct Storage(UnsafeCell<[u8; FULL_BYTES + 16]>);
    // SAFETY: only the canonical owner accesses the initialized pinned bytes.
    unsafe impl Sync for Storage {}

    let storage = Arc::new(Storage(UnsafeCell::new([INITIAL; FULL_BYTES + 16])));
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
            // SAFETY: the final 16 bytes belong to the same pinned allocation.
            host_base: NonNull::new(unsafe { host.add(FULL_BYTES) }).unwrap(),
            bytes: 16,
        },
    ];
    // SAFETY: the registration keeps both disjoint ranges alive and owns
    // their only access. FakeHost's legacy bytes are in a separate allocation.
    let mut ram = unsafe { fixture(&ranges, storage) };
    ram.enable_descriptor_frame_leases().unwrap();
    let ram = Arc::new(ram);

    // Normal memory at VA 0–1 GiB, backed from BASE. The descriptor exists
    // only in registered RAM. The tests invoke actual context leaves directly.
    let descriptor = BASE | 1 | 1 << 10;
    // SAFETY: the input bytes are private and disjoint from registered RAM.
    unsafe { ram.publish_physical(TABLE, &descriptor.to_le_bytes()) }.unwrap();
    let mut machine = board_on(Chip::AppleM6, &[], RecordingDevices::default(), Vec::new());
    enable_streaming(&mut machine);
    let registers = &mut machine.cpu.registers;
    registers.set(Register::TCR_EL1, 0, 25 | 1 << 23 | 5 << 32);
    registers.set(Register::TTBR0_EL1, 0, TABLE);
    registers.set(Register::MAIR_EL1, 0, 0xff);
    registers.set(Register::SCTLR_EL1, 0, 1);
    machine.derived_changed();
    machine.host.device_site_ram = Some(DeviceSiteRam {
        ram: Arc::clone(&ram),
        expected_generation: Some(ram.generation().unwrap()),
    });
    let lanes = core::array::from_fn::<_, STREAMING_VECTOR_BYTES, _>(|lane| 0x20 + lane as u8);
    machine
        .cpu
        .scalable
        .as_mut()
        .unwrap()
        .write_z(&mut machine.cpu.state, 1, &lanes)
        .unwrap();
    (machine, ram)
}

fn read<const N: usize>(ram: &RegisteredSystemRam, physical: u64) -> [u8; N] {
    let mut bytes = [0; N];
    // SAFETY: the output is private and disjoint from registered RAM.
    unsafe { ram.read_physical(physical, &mut bytes) }.unwrap();
    bytes
}

#[test]
fn st1b_preserves_the_resident_prefix_when_a_later_active_lane_is_nonresident() {
    let (mut machine, ram) = registered_machine();
    let offset = (FIRST - BASE) as usize;
    machine.host.ram[offset..offset + STREAMING_VECTOR_BYTES].fill(LEGACY);
    machine.cpu.state.x[5] = offset as u64;
    ram.model_set_nonresident(FRAME);

    machine.scalable_instruction(BASE, PTRUE_P0_B).unwrap();
    let pc = machine.cpu.pc;
    let retired = machine.cpu.retired;
    let Err(Stop::Unsupported(report)) = machine.scalable_instruction(pc, ST1B_Z1_X5) else {
        panic!("the first active lane on a nonresident frame must stop the store");
    };
    assert_eq!(
        report.kind,
        RequirementKind::RamTransfer {
            physical: FRAME,
            write: true,
            completed: 2,
            status: MemoryStatus::Nonresident,
        }
    );
    assert_eq!(report.pc, pc);
    assert_eq!(machine.cpu.pc, pc);
    assert_eq!(machine.cpu.retired, retired);
    assert!(machine.devices.accesses.is_empty());
    assert_eq!(read::<2>(&ram, FIRST), [0x20, 0x21]);
    assert_eq!(
        machine.host.ram[offset..offset + STREAMING_VECTOR_BYTES],
        [LEGACY; STREAMING_VECTOR_BYTES]
    );
    ram.model_set_resident(FRAME);
    assert_eq!(
        read::<{ STREAMING_VECTOR_BYTES - 2 }>(&ram, FRAME),
        [INITIAL; STREAMING_VECTOR_BYTES - 2]
    );
}

#[test]
fn st1b_ignores_inactive_lanes_on_a_nonresident_frame() {
    let (mut machine, ram) = registered_machine();
    let offset = (FIRST - BASE) as usize;
    machine.host.ram[offset..offset + STREAMING_VECTOR_BYTES].fill(LEGACY);
    machine.cpu.state.x[5] = offset as u64;
    ram.model_set_nonresident(FRAME);

    machine.scalable_instruction(BASE, PTRUE_P0_B_VL2).unwrap();
    let pc = machine.cpu.pc;
    let retired = machine.cpu.retired;
    machine.scalable_instruction(pc, ST1B_Z1_X5).unwrap();
    assert_eq!(machine.cpu.pc, pc + 4);
    assert_eq!(machine.cpu.retired, retired + 1);
    assert!(machine.devices.accesses.is_empty());
    assert_eq!(ram.nonresident_frame(FRAME, 1), Some(FRAME));
    assert_eq!(read::<2>(&ram, FIRST), [0x20, 0x21]);
    assert_eq!(
        machine.host.ram[offset..offset + STREAMING_VECTOR_BYTES],
        [LEGACY; STREAMING_VECTOR_BYTES]
    );
    ram.model_set_resident(FRAME);
    assert_eq!(
        read::<{ STREAMING_VECTOR_BYTES - 2 }>(&ram, FRAME),
        [INITIAL; STREAMING_VECTOR_BYTES - 2]
    );
}

#[test]
fn st1b_refuses_absent_or_partial_registered_frames_before_the_first_lane() {
    let (mut machine, ram) = registered_machine();
    ram.model_set_nonresident(FRAME);
    machine.scalable_instruction(BASE, PTRUE_P0_B_VL2).unwrap();
    let pc = machine.cpu.pc;
    let retired = machine.cpu.retired;
    for physical in [FRAME, PARTIAL] {
        let offset = (physical - BASE) as usize;
        machine.host.ram[offset..offset + 16].fill(LEGACY);
        machine.cpu.state.x[5] = offset as u64;
        let Err(Stop::Unsupported(report)) = machine.scalable_instruction(pc, ST1B_Z1_X5) else {
            panic!("the frame at {physical:#x} must stop before the first lane");
        };
        if physical == FRAME {
            assert_eq!(
                report.kind,
                RequirementKind::NonresidentRam { frame: FRAME }
            );
        } else {
            assert!(matches!(report.kind, RequirementKind::NativePreparation(_)));
        }
        assert_eq!(report.pc, pc);
        assert_eq!(machine.cpu.pc, pc);
        assert_eq!(machine.cpu.retired, retired);
        assert!(machine.devices.accesses.is_empty());
        assert_eq!(machine.host.ram[offset..offset + 16], [LEGACY; 16]);
    }
    ram.model_set_resident(FRAME);
    assert_eq!(read::<16>(&ram, FRAME), [INITIAL; 16]);
    assert_eq!(read::<16>(&ram, PARTIAL), [INITIAL; 16]);
}

#[test]
fn st1b_single_active_lane_writes_once_to_a_true_mmio_address() {
    const OFFSET: u64 = 0x2_0000;
    let (mut machine, ram) = registered_machine();
    // The existing L1 block maps this VA with Device-nGnRnE attributes.
    // Its GPA is outside both registered and retained RAM.
    machine.cpu.registers.set(Register::MAIR_EL1, 0, 0);
    machine.derived_changed();
    machine.cpu.state.x[5] = OFFSET;
    let legacy_before = machine.host.ram.clone();
    let registered_before = read::<{ FULL_BYTES + 16 }>(&ram, BASE);

    machine.scalable_instruction(BASE, PTRUE_P0_B_VL1).unwrap();
    let pc = machine.cpu.pc;
    let retired = machine.cpu.retired;
    machine.scalable_instruction(pc, ST1B_Z1_X5).unwrap();

    assert_eq!(machine.devices.accesses, [(BASE + OFFSET, 1, Some(0x20))]);
    assert_eq!(machine.cpu.pc, pc + 4);
    assert_eq!(machine.cpu.retired, retired + 1);
    assert_eq!(machine.host.ram, legacy_before);
    assert_eq!(read::<{ FULL_BYTES + 16 }>(&ram, BASE), registered_before);
}
