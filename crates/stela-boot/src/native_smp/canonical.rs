//! Cold fixed-physical-core adapter over the boot's existing physical reservations.
//! Final Apple topology/device/DMA/reset admission belongs to the mount. This
//! constructs the checked Machine; it never certifies legacy raw V1.
use crate::Result;
use stela_boot_core::worker_bindings as bindings;
use stela_contracts::refuse;

use super::{executor::Executors, firmware::Capture, transport::FixedWorkers};
use crate::{apic::Apic, engine::host, handoff::Handoff, layout};
use alloc::{sync::Arc, vec::Vec};
use core::ptr::NonNull;
use stela_engine::machine::Cpu;
use stela_engine::machine::CpuProfile;
use stela_engine::machine::Devices;
use stela_engine::machine::Machine;
use stela_engine::machine::Stop;
use stela_engine::machine::engine::ServiceHost;
use stela_engine::machine::engine::publication::WorkerBinding;
use stela_engine::machine::engine::publication::WorkerId;
use stela_engine::memory::RegisteredSystemRam;
use stela_engine::memory::ReservedSystemRamRange;
use stela_engine::system::events::EventDomain;

struct Reservation {
    _handoff: &'static Handoff,
    _ranges: Vec<host::RamRange>,
}
pub(crate) type CanonicalMachine<'a, D> =
    Machine<ServiceHost<host::BareHost<'a>, FixedWorkers<'a>>, D>;

/// Register the SAME loader-owned bytes, without a second allocation or copy.
///
/// # Safety
/// The loader has exclusively reserved every listed guest physical range and
/// its shared-window WB CPU aliases for the entire boot. All earlier input
/// copies have completed; no raw V1 guest or asynchronous producer can access
/// these bytes after registration. The physical map and controls stay pinned.
#[cfg(feature = "diagnostics")]
pub(crate) unsafe fn reserve(
    handoff: &'static Handoff,
    ranges: &[host::RamRange],
) -> Result<Arc<RegisteredSystemRam>> {
    // SAFETY: forwarded unchanged loader ownership and closed-writer contract.
    unsafe { reserve_with_limit(handoff, ranges, None) }
}

/// Production assignment adds a conservative ceiling from installed
/// workers. Authored single-BSP diagnostics keep `reserve` without inventing
/// AP executors or treating copied firmware observations as owned workers.
///
/// # Safety
/// `reserve`'s complete reservation contract applies. `capture` is the validated
/// assigned roster; the started executors remain parked and kept.
pub(crate) unsafe fn reserve_for_workers(
    handoff: &'static Handoff,
    ranges: &[host::RamRange],
    capture: &Capture,
    executors: &Executors,
) -> Result<Arc<RegisteredSystemRam>> {
    let reserved = handoff.native_resources.cores as usize;
    // The started executors, fewer than reserved when an AP failed to start.
    let cores = executors.started();
    let count = capture.processor_count as usize;
    if cores > reserved
        || reserved > capture.core_count as usize
        || reserved > super::topology::MAX_CORES
        || count != capture.processors.len()
    {
        return Err(refuse!(
            "canonical target has an invalid assigned physical roster"
        ));
    }
    let selected = |slot: usize| {
        capture.processors[..count]
            .get(capture.selected[slot] as usize)
            .copied()
            .ok_or(refuse!("canonical target selected worker is absent"))
    };
    // SAFETY: current installed BSP with IF clear, before any guest entry.
    // The fresh observation is only a capability ceiling; registration below
    // captures its own current-processor permission independently.
    let bsp = unsafe { super::firmware::observe() }
        .map_err(|_| refuse!("canonical target BSP observation refused"))?;
    if !bsp.matches(selected(0)?) {
        return Err(refuse!("canonical target BSP differs from assigned worker"));
    }
    let five_level = matches!(
        handoff.placement,
        stela_engine::machine::mmu::placement::Placement::FiveLevel
    );
    bsp.qualify_native(five_level)
        .map_err(|refusal| refusal.refusal("canonical target BSP floor refused"))?;
    let mut observations = Vec::with_capacity(cores);
    observations.push(&bsp);
    for slot in 1..cores {
        let expected = selected(slot)?;
        let actual = executors
            .processors()
            .iter()
            .find(|processor| processor.slot() == slot)
            .ok_or(refuse!("canonical target assigned AP is absent"))?;
        let observed = actual.observation();
        if actual.apic_id() != expected.apic_id || !observed.matches(expected) {
            return Err(refuse!("canonical target AP differs from assigned worker"));
        }
        observed
            .qualify_native(five_level)
            .map_err(|refusal| refusal.refusal("canonical target AP floor refused"))?;
        observations.push(observed);
    }
    let limit = super::topology::common_native_target(&observations)
        .map_err(|_| refuse!("canonical target assigned observations refused"))?;
    // SAFETY: same loader owner and closed writers, with a policy-only limit.
    unsafe { reserve_with_limit(handoff, ranges, Some(limit)) }
}

unsafe fn reserve_with_limit(
    handoff: &'static Handoff,
    ranges: &[host::RamRange],
    limit: Option<stela_compiler::TargetCpu>,
) -> Result<Arc<RegisteredSystemRam>> {
    let (cr0, cr3, cr4, flags, efer): (u64, u64, u64, u64, u64);
    // SAFETY: the same cold CPL0 owner required by registration. Read only
    // live BSP controls, without changing firmware MTRR visibility or using
    // the firmware snapshot/an AP's startup controls as current evidence.
    unsafe {
        core::arch::asm!("mov {}, cr0", out(reg) cr0, options(nostack, preserves_flags));
        core::arch::asm!("mov {}, cr3", out(reg) cr3, options(nostack, preserves_flags));
        core::arch::asm!("mov {}, cr4", out(reg) cr4, options(nostack, preserves_flags));
        core::arch::asm!("pushfq", "pop {}", out(reg) flags, options(preserves_flags));
        efer = crate::cpu::msr::EFER.read();
    }
    crate::output::arguments(format_args!(
        "NATIVE RAM BSP cr0={cr0:#x} cr3={cr3:#x} cr4={cr4:#x} efer={efer:#x} rflags={flags:#x}\n"
    ));
    if ranges.is_empty() {
        return Err(refuse!("canonical guest RAM has no loader reservations"));
    }
    let mut registered = Vec::with_capacity(ranges.len());
    for range in ranges {
        let end = range
            .host
            .checked_add(range.bytes)
            .ok_or(refuse!("canonical guest RAM extent wraps"))?;
        if range.bytes == 0
            || !handoff.guest[..handoff.guest_count]
                .iter()
                .any(|reservation| {
                    reservation.physical <= range.host
                        && reservation
                            .physical
                            .checked_add(reservation.bytes)
                            .is_some_and(|top| end <= top)
                })
        {
            return Err(refuse!(
                "canonical guest RAM is outside actual loader ownership"
            ));
        }
        let bytes = usize::try_from(range.bytes)
            .map_err(|_| refuse!("canonical guest RAM exceeds host size"))?;
        let cpu = layout::physical(range.host);
        // SAFETY: exact loader reservation; no root/control migration
        // or native/host producer can race this pre-entry validation.
        unsafe { crate::native_memory::qualify_reserved_writeback(cpu, range.host, bytes) }
            .map_err(|_| refuse!("canonical guest RAM mapping/effective WB type is unqualified"))?;
        registered.push(ReservedSystemRamRange {
            source_pa: range.guest,
            host_pa: range.host,
            host_base: NonNull::new(cpu as *mut u8)
                .ok_or(refuse!("canonical guest RAM has null mapping"))?,
            bytes,
        });
    }
    let owner = Arc::new(Reservation {
        _handoff: handoff,
        _ranges: ranges.to_vec(),
    });
    // SAFETY: same exclusively reserved bytes and checked WB aliases; kept
    // owner survives frames, source fetch/walk, copyout and stopped settlement.
    unsafe {
        match limit {
            Some(limit) => RegisteredSystemRam::register_with_target_limit(
                &registered,
                owner,
                stela_engine::machine::cpu::RESERVATION_GRANULE,
                64,
                limit,
            ),
            None => RegisteredSystemRam::register(
                &registered,
                owner,
                stela_engine::machine::cpu::RESERVATION_GRANULE,
                64,
            ),
        }
    }
    .and_then(|mut ram| {
        if handoff.memory_profile == stela_boot_core::host_memory::Profile::Paged {
            ram.enable_descriptor_frame_leases()?;
            crate::output::arguments(format_args!(
                "NATIVE RAM canonical frame leases enabled (16 KiB)\n"
            ));
        }
        Ok(Arc::new(ram))
    })
    .map_err(|error| {
        crate::output::arguments(format_args!(
            "NATIVE RAM registration refused cause={error:?}\n"
        ));
        refuse!("canonical boot RAM registration refused: {error}", ?error)
    })
}

/// The host owners and guest state consumed by cold machine construction.
/// Host/executor borrows live with the machine; assignment slices are read
/// only while construction validates and installs their bindings.
pub(crate) struct Preparation<'host, 'plan, D> {
    pub handoff: &'static Handoff,
    pub capture: &'static Capture,
    pub executors: &'host Executors,
    pub sender: &'host Apic,
    pub owner: &'host mut host::HostOwner,
    pub ram: Arc<RegisteredSystemRam>,
    pub cpu: Cpu,
    pub devices: D,
    pub profile: CpuProfile,
    pub boot_processor: usize,
    pub assignments: &'plan [WorkerBinding],
    pub processor_power: &'plan [stela_engine::machine::engine::ProcessorBinding],
}

/// Construct a cold machine with its selected boot PE on the BSP worker.
///
/// # Safety
/// [`reserve`]'s reservation contract holds. The source PE/resource/AIC plan
/// has been projected to this core roster. The capture, executors, host state,
/// mappings and synchronous device producers stay pinned for the machine's
/// lifetime. Guest execution and asynchronous writers begin only after this
/// constructor returns and their complete platform state has been installed.
#[cfg(feature = "diagnostics")]
pub(crate) unsafe fn prepare<'host, D: Devices>(
    preparation: Preparation<'host, '_, D>,
) -> Result<CanonicalMachine<'host, D>> {
    // SAFETY: forwarded cold construction contract; diagnostic callers
    // keep any already prepared CPU lock state.
    unsafe { prepare_with_locks(preparation, false, None) }
}

/// Cold construction with the iBoot core-lock handoff. Initial CTRR,
/// lock state and source translation controls settle before native owners or
/// any frame loans are published. `None` preserves an already prepared Cpu.
///
/// # Safety
/// The complete cold ownership and producer contract of `prepare` applies.
pub(crate) unsafe fn prepare_with_locks<'host, D: Devices>(
    preparation: Preparation<'host, '_, D>,
    trace_artifacts: bool,
    core_locks: Option<stela_engine::machine::ctrr::CoreLocks>,
) -> Result<CanonicalMachine<'host, D>> {
    let Preparation {
        handoff,
        capture,
        executors,
        sender,
        owner,
        ram,
        cpu,
        devices,
        profile,
        boot_processor,
        assignments,
        processor_power,
    } = preparation;
    let reserved = handoff.native_resources.cores as usize;
    let started = executors.started();
    let cores = profile.processors.len().max(1);
    if cores == 0 || cores > started || started > reserved || reserved > capture.core_count as usize
    {
        return Err(refuse!(
            "canonical active PE roster exceeds the retained physical workers"
        ));
    }
    bindings::validate(cores, started, boot_processor, assignments)?;
    for binding in assignments.iter().filter(|binding| binding.worker.0 != 0) {
        let slot = binding.worker.0;
        let selected = capture.processors[capture.selected[slot] as usize];
        let Some(processor) = executors
            .processors()
            .iter()
            .find(|processor| processor.slot() == slot)
        else {
            return Err(refuse!("assigned AP executor is absent"));
        };
        if processor.apic_id() != selected.apic_id || !processor.observation().matches(selected) {
            return Err(refuse!(
                "actual AP executor differs from its assigned physical binding"
            ));
        }
    }
    let bsp = capture.processors[capture.selected[0] as usize];
    let apic = u32::try_from(bsp.apic_id)
        .map_err(|_| refuse!("canonical BSP APIC identity exceeds transport"))?;
    owner.bind_worker(WorkerId(0), apic)?;
    // SAFETY: registered owner, pristine GS-local dispatch; every later
    // access goes through this same canonical RAM, never old raw V1 stores.
    unsafe { owner.register_system_ram(Arc::clone(&ram)) }?;
    let transport = FixedWorkers::new(
        executors,
        sender,
        handoff.tsc_hz,
        Arc::clone(&ram),
        trace_artifacts,
    );
    let host = ServiceHost::new(owner.host(), transport, Arc::clone(&ram));
    let mut machine = Machine::new_at(cpu, host, devices, profile, ram.target(), boot_processor)?;
    if let Some(core_locks) = core_locks {
        machine.hand_off_core_locks(core_locks);
    }
    machine
        .install_shared_memory_for_active_processors(ram, &EventDomain::new())
        .map_err(|error| {
            crate::output::arguments(format_args!(
                "NATIVE CANONICAL cold installation refused: {error:?}\n"
            ));
            refuse!("cold source controls/canonical CPU ownership refused before entry")
        })?;
    // Initial handoff bindings must settle before any physical worker owner
    // publishes. Later changes require the complete held native cohort.
    if !processor_power.is_empty() {
        machine
            .bind_processor_power(processor_power)
            .map_err(|_| refuse!("canonical initial processor power bindings refused"))?;
    }
    let owners = machine
        .bind_assigned_native_workers(assignments)
        .map_err(|_| refuse!("canonical assigned physical roster binding refused"))?;
    machine.initialize_native_transport(owners);
    Ok(machine)
}

/// Execute checked native rounds and consuming canonical reads. The
/// ordinary boot reporting loop can call this instead of raw Machine::run.
/// Unsupported source/device families stop unexecuted, with no V1 fallback.
///
/// # Safety
/// `prepare`'s complete closed producer/mount contract remains valid. This is
/// admitted only after the final platform owner joins its complete graph.
#[cfg(feature = "diagnostics")]
pub(crate) unsafe fn run_slice<D: Devices>(
    machine: &mut CanonicalMachine<'_, D>,
    limit: u64,
) -> Stop {
    // SAFETY: same complete mount/producer contract as this entry point.
    unsafe { run_slice_while(machine, limit, None, || true) }
}

/// Return to the outer host-service loop after `service_ticks` of the checked
/// BSP counter, even when the block limit has not been reached. The caller
/// chooses the wall-time service interval from its calibrated counter rate.
/// One recovered round or synchronous device completion can exceed the interval;
/// a live frame, kept device continuation or Host session is never preempted.
///
/// # Safety
/// The complete closed producer/mount contract of `run_slice` remains valid.
pub(crate) unsafe fn run_slice_for<D: Devices>(
    machine: &mut CanonicalMachine<'_, D>,
    limit: u64,
    service_ticks: u64,
) -> Stop {
    let started = host::rdtsc();
    // SAFETY: same complete mount/producer contract as this entry point.
    unsafe {
        run_slice_while(
            machine,
            limit,
            Some(host::counter_deadline(
                started.saturating_add(service_ticks.max(1)),
            )),
            || host::rdtsc().wrapping_sub(started) < service_ticks.max(1),
        )
    }
}

/// Check the host-service deadline only at the same settled boundary as the
/// block budget. In particular, finish the entire recovered device cohort first:
/// its next access can keep the stopped workers and Host sessions.
unsafe fn run_slice_while<D: Devices>(
    machine: &mut CanonicalMachine<'_, D>,
    limit: u64,
    deadline: Option<stela_engine::machine::engine::NativeEpochDeadline>,
    mut continue_running: impl FnMut() -> bool,
) -> Stop {
    loop {
        if machine.pending_native_device_access().is_some() {
            // SAFETY: same cohort, controller and closed producers.
            match unsafe { machine.complete_native_device_access() } {
                Ok(()) | Err(Stop::DeviceRead(_) | Stop::DeviceWrite(_)) => continue,
                Err(stop) => return stop,
            }
        }
        if machine.statistics.blocks >= limit || !continue_running() {
            return Stop::Budget;
        }
        // SAFETY: same registered owner/fixed executor and closed providers.
        match unsafe { machine.native_round_bounded(limit - machine.statistics.blocks, deadline) } {
            Ok(_) | Err(Stop::DeviceRead(_) | Stop::DeviceWrite(_)) => {}
            Err(stop) => return stop,
        }
    }
}
