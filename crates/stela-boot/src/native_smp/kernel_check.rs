//! A bounded real translated ARM kernel cohort, never original-macOS SMP credit.
mod cold;
mod source;
use super::{executor::Executors, firmware::Capture};
use crate::Result;
use crate::{apic::Apic, engine::host, handoff::Handoff};
use alloc::{sync::Arc, vec::Vec};
use stela_compiler::{GuestState, TargetCpu};
use stela_contracts::Chip;
use stela_contracts::identity::Identity;
use stela_contracts::refuse;
use stela_engine::machine::Cpu;
use stela_engine::machine::CpuProfile;
use stela_engine::machine::DeviceError;
use stela_engine::machine::Devices;
use stela_engine::machine::Host;
use stela_engine::machine::Lines;
use stela_engine::machine::Machine;
use stela_engine::machine::PeerState;
use stela_engine::machine::Stop;
use stela_engine::machine::engine;
use stela_engine::machine::engine::publication::WorkerBinding;
use stela_engine::machine::engine::publication::WorkerId;
use stela_engine::machine::exception::{DAIF_I, Pstate};
use stela_engine::memory::RegisteredSystemRam;
use stela_soc::machine;

struct TestDevices {
    starts: Vec<engine::Start>,
    ready: Vec<bool>,
    canonical_writes: usize,
    cpu_commits: usize,
}
impl Devices for TestDevices {
    fn read(&mut self, _: usize, _: u64, _: u8, _: u64) -> Result<u64, DeviceError> {
        Err(DeviceError::NoDevice)
    }
    fn write(
        &mut self,
        pe: usize,
        address: u64,
        bytes: u8,
        value: u64,
        _: u64,
    ) -> Result<(), DeviceError> {
        if address != source::READY || bytes != 8 || value != 1 {
            return Err(DeviceError::NoDevice);
        }
        *self.ready.get_mut(pe).ok_or(DeviceError::NoDevice)? = true;
        Ok(())
    }
    fn write_canonical(
        &mut self,
        pe: usize,
        address: u64,
        bytes: u8,
        value: u64,
        _: u64,
        owners: &mut dyn engine::CanonicalDeviceWrite,
    ) -> Result<engine::ConsumedWrite, DeviceError> {
        if address != source::READY || bytes != 8 || value != 1 {
            return Err(DeviceError::NoDevice);
        }
        self.consume_write(pe, owners)
    }
    fn write_pair_canonical(
        &mut self,
        pe: usize,
        address: u64,
        bytes: u8,
        values: [u64; 2],
        _: u64,
        owners: &mut dyn engine::CanonicalDeviceWrite,
    ) -> Result<engine::ConsumedWrite, DeviceError> {
        if address != source::READY || bytes != 8 || values != [1, 1] {
            return Err(DeviceError::NoDevice);
        }
        self.consume_write(pe, owners)
    }
    fn lines(&mut self, _: usize, _: u64) -> Lines {
        Lines {
            irq: self.ready.iter().all(|ready| *ready),
            ..Lines::default()
        }
    }
    fn take_start(&mut self) -> Option<engine::Start> {
        self.starts.pop()
    }
}
impl TestDevices {
    fn consume_write(
        &mut self,
        pe: usize,
        owners: &mut dyn engine::CanonicalDeviceWrite,
    ) -> Result<engine::ConsumedWrite, DeviceError> {
        use engine::CanonicalDeviceMemory;
        {
            let mut memory = owners.memory()?;
            memory.memory().validate_current().map_err(|e| {
                DeviceError::TransferRefused(refuse!("fixture canonical memory {e}", ?e))
            })?;
            memory.fence_cpu_memory();
        }
        // Authored controller: both transactions commit while the
        // recovered cohort remains held, including the OTHER physical PE.
        let other = (pe + 1) % self.ready.len();
        let active = machine::memory_resources::LifecycleState {
            powered: true,
            clocked: true,
            quiesced: false,
            reset: false,
            retention: false,
        };
        let paused = owners
            .processors()?
            .apply(&[engine::ProcessorChange {
                domain: 1 + other as u64,
                state: machine::memory_resources::LifecycleState {
                    clocked: false,
                    ..active
                },
            }])
            .map_err(|e| DeviceError::TransferRefused(refuse!("fixture CPU preflight {e}", ?e)))?;
        assert_eq!(paused.paused(other), Some(true));
        self.cpu_commits += 1; // prepared-controller ACK inside lease
        drop(paused);
        let settlement = (|| {
            let resumed = owners
                .processors()?
                .apply(&[engine::ProcessorChange {
                    domain: 1 + other as u64,
                    state: active,
                }])
                .map_err(|e| DeviceError::TransferRefused(refuse!("fixture CPU resume {e}", ?e)))?;
            assert_eq!(resumed.paused(other), Some(false));
            self.cpu_commits += 1;
            drop(resumed);
            let mut memory = owners.memory()?;
            memory.memory().validate_current().map_err(|e| {
                DeviceError::TransferRefused(refuse!("fixture post-CPU canonical memory {e}", ?e))
            })?;
            memory.fence_cpu_memory();
            Ok(())
        })();
        if let Err(error) = settlement {
            return Ok(engine::ConsumedWrite::Unknown { error });
        }
        self.ready[pe] = true;
        self.canonical_writes += 1;
        Ok(engine::ConsumedWrite::Complete { settlement: Ok(()) })
    }
}
/// # Safety
/// `loader_ram` is the complete loader backing, reserved exclusively for
/// the guest. No guest, direct publisher or DMA accesses it during the check.
unsafe fn reservation(
    handoff: &'static Handoff,
    loader_ram: &[host::RamRange],
) -> Result<Arc<RegisteredSystemRam>> {
    const UNUSED_GUEST_BASE: u64 = 0x8_0000_0000;
    const { assert!(source::BASE + source::BYTES as u64 <= source::READY) };
    let bytes = source::BYTES as u64;
    let mut ranges = loader_ram.to_vec();
    let index = ranges
        .iter()
        .position(|range| range.bytes >= bytes)
        .ok_or(refuse!(
            "native kernel has no contiguous loader RAM for its fixture"
        ))?;
    let mut backing = ranges.remove(index);
    let fixture = host::RamRange {
        guest: source::BASE,
        host: backing.host,
        bytes,
    };
    if backing.bytes > bytes {
        backing.host = backing
            .host
            .checked_add(bytes)
            .ok_or(refuse!("native kernel loader RAM extent wraps"))?;
        backing.bytes -= bytes;
        ranges.insert(index, backing);
    }
    // Keep the retained physical union while leaving the fixture's MMIO
    // addresses unbacked. The authored guest only touches its low extent.
    let mut guest = UNUSED_GUEST_BASE;
    for range in &mut ranges {
        range.guest = guest;
        guest = guest
            .checked_add(range.bytes)
            .ok_or(refuse!("native kernel guest RAM extent wraps"))?;
    }
    ranges.insert(0, fixture);
    // SAFETY: only the guest addresses changed. The loader's complete
    // physical reservation and the closed writer contract remain held.
    let ram = unsafe { super::canonical::reserve(handoff, &ranges) }?;
    let zeros = [0; 4096];
    for offset in (0..source::BYTES).step_by(zeros.len()) {
        let length = (source::BYTES - offset).min(zeros.len());
        // SAFETY: the validated loader RAM has no other writer, and this
        // source slice is disjoint from it. Clear only the fixture.
        unsafe { ram.publish_physical(source::BASE + offset as u64, &zeros[..length]) }
            .map_err(|_| refuse!("native kernel fixture zeroing failed"))?;
    }
    Ok(ram)
}

fn publish_scheduler(ram: &RegisteredSystemRam) -> Result<()> {
    for (address, descriptor) in source::tables() {
        // SAFETY: the source bytes are private locals, disjoint from the registered RAM, and `ram`
        // keeps its registration live.
        unsafe { ram.publish_physical(address, &descriptor.to_le_bytes()) }
            .map_err(|_| refuse!("native kernel table publication failed"))?;
    }
    let (program, _) = source::warmup();
    // SAFETY: the source bytes are private locals, disjoint from the registered RAM, and `ram`
    // keeps its registration live.
    unsafe { ram.publish_physical(source::CODE, &program) }
        .map_err(|_| refuse!("native kernel source publication failed"))?;
    Ok(())
}
fn profile(cores: usize) -> Result<CpuProfile> {
    let chip = Chip::AppleM1;
    let fixed = stela_engine::machine::identification::derive(
        chip,
        TargetCpu::Baseline,
        true,
        stela_engine::machine::identification::Speculation::NONE,
    );
    // The diagnostic declares the existing t8103 P-core interface. Variant1 /
    // revision0 have the same explicitly inferred status as soc::cpus; this is
    // not a physical x86 P/E classification or an Apple SKU/core-count claim.
    let part = stela_soc_data::board::midr_part(0x8103, b'P')
        .ok_or(refuse!("diagnostic M1 interface has no sourced MIDR part"))?;
    let midr = (0x61 << 24) | (1 << 20) | (0xf << 16) | (u64::from(part) << 4);
    Ok(CpuProfile {
        name: "bounded-native-shared-kernel-m1-isa-v1",
        chip,
        counter_frequency: 24_000_000,
        fixed,
        processors: (0..cores)
            .map(|pe| Identity {
                midr,
                mpidr: (1 << 31) | pe as u64,
                clock: None,
                caches: None,
            })
            .collect(),
    })
}
fn ready<H: Host>(machine: &Machine<H, TestDevices>, entry: u64) -> bool {
    machine.peer_states().iter().all(|state| match state {
        PeerState::Running => machine.cpu.pc == entry,
        PeerState::Parked { cpu, wfi: None } => cpu.pc == entry,
        _ => false,
    })
}
fn states<H: Host>(machine: &Machine<H, TestDevices>) -> Result<(Vec<GuestState>, u64)> {
    let mut retired = 0u64;
    let states = machine
        .peer_states()
        .iter()
        .map(|state| {
            let cpu = match state {
                PeerState::Running => &machine.cpu,
                PeerState::Parked { cpu, wfi: None } => cpu,
                _ => {
                    return Err(refuse!(
                        "native kernel PE not at an active instruction boundary"
                    ));
                }
            };
            retired = retired.checked_add(cpu.retired).ok_or(refuse!(
                "native scheduler retired instruction total overflow"
            ))?;
            Ok(cpu.state.clone())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((states, retired))
}
/// # Safety
/// Real selected BSP/AP ownership/reservations/descriptors remain pinned. No
/// original guest or DMA runs; caller terminates the diagnostic after return.
pub(crate) unsafe fn check(
    handoff: &'static Handoff,
    capture: &'static Capture,
    executors: &Executors,
    host_owner: &mut host::HostOwner,
    sender: &Apic,
    loader_ram: &[host::RamRange],
) -> Result<()> {
    let cores = handoff.native_resources.cores as usize;
    // SAFETY: this diagnostic owns the complete loader backing and will stop
    // before the normal guest or any asynchronous producer can use it.
    let ram = unsafe { reservation(handoff, loader_ram) }?;
    if cores == 1 {
        // SAFETY: same bootstrap/physical-owner fixture contract.
        return unsafe { cold::check(handoff, capture, executors, host_owner, sender, ram) };
    }
    publish_scheduler(&ram)?;
    // Preserve a nonzero guest boot PE while the physical BSP
    // remains worker0. Other identities and their source affinity stay ordered.
    let boot_processor = cores - 1;
    let mut cpu = Cpu::of(Chip::AppleM1, source::CODE, Pstate::reset(2));
    for (index, vector) in cpu.state.v.iter_mut().enumerate() {
        *vector = [0x10203040 + index as u64, !(index as u64)];
    }
    let spsr = cpu.pstate.to_spsr(cpu.state.nzcv);
    let devices = TestDevices {
        starts: (0..cores)
            .filter(|&pe| pe != boot_processor)
            .map(|cpu| engine::Start {
                cpu,
                address: source::CODE,
            })
            .collect(),
        ready: alloc::vec![false; cores],
        canonical_writes: 0,
        cpu_commits: 0,
    };
    let (_, entry) = source::warmup();
    let startup = source::cold_startup(entry);
    // SAFETY: before the first guest instruction; every assigned physical
    // executor is still stopped and shares this exact registered reservation.
    unsafe { ram.publish_physical(source::CODE, &startup) }
        .map_err(|_| refuse!("native cold startup publication refused"))?;
    let bindings: Vec<_> = (0..cores)
        .map(|pe| WorkerBinding {
            pe,
            worker: WorkerId(if pe == boot_processor { 0 } else { pe + 1 }),
        })
        .collect();
    // SAFETY: every source byte belongs to this canonical fixture,
    // all assigned BSP/AP Hosts are stopped, and no guest or DMA has
    // executed. This exercises the same cold constructor as original adoption.
    let mut machine = unsafe {
        super::canonical::prepare(super::canonical::Preparation {
            handoff,
            capture,
            executors,
            sender,
            owner: host_owner,
            ram: Arc::clone(&ram),
            cpu,
            devices,
            profile: profile(cores)?,
            boot_processor,
            assignments: &bindings,
            processor_power: &(0..cores)
                .map(|pe| engine::ProcessorBinding {
                    domain: 1 + pe as u64,
                    target: stela_contracts::processor::ProcessorTarget::Core { pe },
                    initial: machine::memory_resources::LifecycleState {
                        powered: true,
                        clocked: true,
                        quiesced: false,
                        reset: false,
                        retention: false,
                    },
                })
                .collect::<Vec<_>>(),
        })
    }?;
    // In an oversubscribed hypervisor, one round needs a separate host
    // timeslice for every vCPU. Preserve the bare-metal deadline and the
    // round limit while scaling only the virtualized fixture's time budget.
    let deadline_seconds = if crate::output::hypervisor_exit_port() {
        10 * (cores as u64).div_ceil(8)
    } else {
        10
    };
    let deadline_ticks = handoff.tsc_hz.saturating_mul(deadline_seconds);
    let started = crate::apic::rdtsc();
    while !ready(&machine, entry) {
        // SAFETY: source-started Cpu loans and owner-local AP Hosts;
        // no original guest, external publisher or raw guest execution exists.
        if let Err(stop) = unsafe { machine.native_round() } {
            crate::output::arguments(format_args!("NATIVE SCHEDULER cold startup {stop:?}\n"));
            return Err(refuse!(
                "native scheduler checked cold control/CPU-start path stopped"
            ));
        }
        if crate::apic::rdtsc().wrapping_sub(started) >= deadline_ticks {
            crate::output::arguments(format_args!(
                "NATIVE SCHEDULER exhausted cold-start time budget seconds={deadline_seconds} workers={cores}\n"
            ));
            return Err(refuse!("native scheduler checked cold startup deadline"));
        }
    }
    let (before, retired_before) = states(&machine)?;
    let source::KernelProgram {
        code: program,
        done,
        fault_pc,
        vectors: handlers,
    } = source::scheduler_body(entry, spsr);
    crate::output::arguments(format_args!(
        "NATIVE SCHEDULER source pc={entry:#x} bytes={program:02x?}\n"
    ));
    // SAFETY: all frames are stopped before shared conversion; source
    // and vector bytes have exact new artifact identities and no racing writer.
    unsafe { ram.publish_physical(entry, &program) }
        .map_err(|_| refuse!("native scheduler source publication refused"))?;
    for (address, code) in handlers {
        crate::output::arguments(format_args!(
            "NATIVE SCHEDULER source pc={address:#x} bytes={code:02x?}\n"
        ));
        // SAFETY: the source bytes are private locals, disjoint from the registered RAM, and `ram`
        // keeps its registration live.
        unsafe { ram.publish_physical(address, &code) }
            .map_err(|_| refuse!("native scheduler vector publication refused"))?;
    }
    let (alias, descriptor) = source::read_only_alias();
    // SAFETY: the source bytes are private locals, disjoint from the registered RAM, and `ram`
    // keeps its registration live.
    unsafe { ram.publish_physical(alias, &descriptor.to_le_bytes()) }
        .map_err(|_| refuse!("native scheduler source alias publication refused"))?;
    let mut rounds = 0;
    let mut max_pending = 0;
    let scheduler_started = crate::apic::rdtsc();
    loop {
        // SAFETY: the real fixed BSP/AP owners, canonical RAM and source bytes
        // stay pinned. The diagnostic admits no external DMA/debug/AMX writer;
        // every frame and stopped Host effect is synchronously joined.
        match unsafe { machine.native_round() } {
            Ok(_) => {}
            Err(stop) if machine.pending_native_device_access().is_some() => {
                let _ = stop;
            }
            Err(stop) => {
                crate::output::arguments(format_args!("NATIVE SCHEDULER stopped {stop:?}\n"));
                return Err(refuse!(
                    "native scheduler reached an unqualified source boundary"
                ));
            }
        }
        max_pending = max_pending.max(machine.pending_native_device_access_count());
        while machine.pending_native_device_access().is_some() {
            // SAFETY: closed synchronous fixture, same recovered cohort;
            // no external producer or direct writer can run between phases.
            match unsafe { machine.complete_native_device_access() } {
                Ok(()) => {}
                Err(Stop::DeviceRead(_) | Stop::DeviceWrite(_)) => {}
                Err(stop) => {
                    crate::output::arguments(format_args!("NATIVE SCHEDULER device {stop:?}\n"));
                    return Err(refuse!(
                        "native scheduler canonical device settlement failed"
                    ));
                }
            }
        }
        rounds += 1;
        if machine.peer_states().iter().all(|state| match state {
            PeerState::Running => machine.cpu.pc == done,
            PeerState::Parked { cpu, wfi: None } => cpu.pc == done,
            _ => false,
        }) {
            break;
        }
        if rounds >= 256 {
            return Err(refuse!("native scheduler source program/IRQ round limit"));
        }
        if crate::apic::rdtsc().wrapping_sub(started) >= deadline_ticks {
            crate::output::arguments(format_args!(
                "NATIVE SCHEDULER exhausted time budget seconds={deadline_seconds} workers={cores} completed_rounds={rounds}\n"
            ));
            return Err(refuse!("native scheduler source program/IRQ deadline"));
        }
    }
    let scheduler_ticks = crate::apic::rdtsc().wrapping_sub(scheduler_started);
    if max_pending < 2
        || machine.devices.canonical_writes != cores * 2
        || machine.devices.cpu_commits != cores * 4
    {
        return Err(refuse!(
            "native scheduler did not consume whole actual pending cohort/CPU leases"
        ));
    }
    let (after, retired_after) = states(&machine)?;
    let scheduled_instructions = retired_after
        .checked_sub(retired_before)
        .filter(|&count| count != 0)
        .ok_or(refuse!(
            "native scheduler did not retire fixture instructions"
        ))?;
    for (old, actual) in before.iter().zip(&after) {
        let mut expected = old.clone();
        // Both READY writes leave the fixture's IRQ asserted. The timer FIQ
        // was disabled by its handler, so only IRQ remains in the frame.
        expected.pending = u32::from(DAIF_I);
        for register in [0, 1, 2, 3, 4, 5, 6, 7, 10, 19, 20, 22, 23, 24] {
            expected.x[register] = actual.x[register];
        }
        if *actual != expected
            || actual.x[0] != source::ALIAS
            || actual.x[1] != 0x1122334455667788
            || actual.x[2] >= (cores * source::INCREMENTS) as u64
            || actual.x[3] != old.x[3] + 2
            || actual.x[4] != 0
            || actual.x[5] != 0
            || actual.x[7] != 1
            || actual.x[6] != source::DATA + 64
            || actual.x[10] != done - 6 * 4
            || actual.x[19] != 0x112233
            || actual.x[20] != 1
            || actual.x[24] != 1
            || actual.x[22] != 0x9600004f
            || actual.x[23] != source::ALIAS
        {
            crate::output::arguments(format_args!("NATIVE SCHEDULER frame {actual:?}\n"));
            return Err(refuse!(
                "native scheduler complete frame/system/ERET/IRQ/abort mismatch"
            ));
        }
    }
    for state in machine.peer_states() {
        let cpu = match state {
            PeerState::Running => &machine.cpu,
            PeerState::Parked { cpu, .. } => cpu,
            _ => return Err(refuse!("native scheduler lost actual Cpu owner")),
        };
        crate::output::arguments(format_args!(
            "NATIVE SCHEDULER frame pc={:#x} retired={} pstate={:?} state={:?} ESR={:#x} FAR={:#x} ELR={:#x}\n",
            cpu.pc,
            cpu.retired,
            cpu.pstate,
            cpu.state,
            cpu.read(stela_engine::machine::system::Register::ESR_EL2),
            cpu.read(stela_engine::machine::system::Register::FAR_EL2),
            cpu.read(stela_engine::machine::system::Register::ELR_EL2)
        ));
        if cpu.event
            || !cpu.event_is_set()
            || cpu.read(stela_engine::machine::system::Register::ELR_EL2) != fault_pc
        {
            return Err(refuse!(
                "native scheduler lost canonical event or precise abort source PC"
            ));
        }
    }
    let mut value = [0; 8];
    // SAFETY: `value` is a private local, disjoint from the registered RAM, and `ram` keeps its
    // registration live.
    unsafe { ram.read_physical(source::DATA, &mut value) }
        .map_err(|_| refuse!("native scheduler canonical read refused"))?;
    if u64::from_le_bytes(value) != (cores * source::INCREMENTS) as u64 {
        return Err(refuse!(
            "native scheduler lost actual concurrent LSE updates or permission fault changed RAM"
        ));
    }
    // SAFETY: `value` is a private local, disjoint from the registered RAM, and `ram` keeps its
    // registration live.
    unsafe { ram.read_physical(source::DATA + 128, &mut value) }
        .map_err(|_| refuse!("native scheduler canonical store read refused"))?;
    if u64::from_le_bytes(value) != 0x1122334455667788 {
        return Err(refuse!(
            "native scheduler ordinary translated store did not use canonical RAM"
        ));
    }
    let unmodeled = machine.take_unmodeled();
    if !unmodeled.is_empty() {
        for requirement in unmodeled {
            crate::output::arguments(format_args!("NATIVE SCHEDULER UNMODELED {requirement}\n"));
        }
        return Err(refuse!(
            "native scheduler answered an unsupported operation degraded"
        ));
    }
    crate::output::arguments(format_args!(
        "CHECK native-shared-kernel PASS physical_cores={cores} boot_PE={boot_processor} BSP_worker=0 actual_frames={cores} lse_updates={} full_GPR_SIMD_FP_state=preserved canonical_RAM=shared cold_M0_to_M1=checked CPU_starts={} consumed_device_writes={} CPU_controller_commits={} max_pending={max_pending} legacy_warmup=none scheduler_rounds={rounds} scheduler_ticks={scheduler_ticks} tsc_hz={} scheduled_instructions={scheduled_instructions} source_ERET_STXP=failed source_WFE=completed source_timer_FIQ=delivered source_permission_abort=precise actual_Host_effects=receipted; original macOS SMP unqualified\n",
        cores * source::INCREMENTS,
        cores - 1,
        machine.devices.canonical_writes,
        machine.devices.cpu_commits,
        handoff.tsc_hz
    ));
    Ok(())
}
