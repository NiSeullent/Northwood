//! Native host processors for system execution. Firmware identity/capability
//! observations are copied before EBS; persistent APs use owned Stela startup
//! and processor-local resources afterward. Production shared-memory admission
//! is distinct from the direct `SYSTEM_MODE` profile's single-agent translation contract.
use crate::Result;
use stela_contracts::refuse;
pub(crate) mod bootstrap;
pub(crate) mod canonical;
#[cfg(feature = "diagnostics")]
mod context_check;
pub(crate) mod executor;
pub(crate) mod firmware;
#[cfg(feature = "diagnostics")]
pub(crate) mod kernel_check;
pub(crate) use stela_boot_core::memory_types;
pub(crate) mod resources;
pub(crate) use stela_boot_core::topology;
pub(crate) mod transport;
mod tsc_sync;

/// Check the exact loader-owned reservation against the requested, source
/// identified logical processors before any INIT/SIPI or guest-state loan.
///
/// # Safety
/// `handoff` is the live immutable loader record and all reserved regions are
/// in the shared physmap. This does not admit shared guest execution.
pub(crate) unsafe fn validate(
    handoff: &crate::handoff::Handoff,
    requested: Option<usize>,
) -> Result<Option<&'static firmware::Capture>> {
    let cores = handoff.native_resources.cores as usize;
    let runners = handoff.native_resources.runners as usize;
    if runners == 0 || runners > cores || cores > topology::MAX_CORES {
        return Err(refuse!("native processor count is outside capacity"));
    }
    let resources = &handoff.native_resources;
    // The loader reserves fewer than requested only when firmware left no
    // AP startup page.
    if requested.is_some_and(|requested| cores > requested)
        || resources.stacks.bytes != resources::STACK_BYTES * cores as u64
        || resources.stacks.physical == 0
    {
        return Err(refuse!(
            "native executor reservations differ from requested cores"
        ));
    }
    let invocations = resources::invocation_stacks(cores)
        .map_err(|_| refuse!("native invocation-stack geometry exceeds capacity"))?;
    for (region, bytes) in [
        (resources.invocations, invocations.physical_bytes()),
        (
            resources.tasks,
            resources::task_stacks(crate::layout::task_stacks())
                .map_err(|_| refuse!("native task-stack geometry exceeds capacity"))?
                .raw_allocation_bytes(),
        ),
    ] {
        let end = region
            .physical
            .checked_add(region.bytes)
            .ok_or(refuse!("native guarded stack physical range wraps"))?;
        if region.physical == 0
            || !region.physical.is_multiple_of(crate::tables::PAGE)
            || region.bytes != bytes
        {
            return Err(refuse!(
                "native guarded stack reservation differs from loader geometry"
            ));
        }
        for other in [
            handoff.heap,
            handoff.code,
            handoff.pool,
            resources.stacks,
            handoff.storage.dma,
            if region.physical == resources.invocations.physical {
                resources.tasks
            } else {
                resources.invocations
            },
        ] {
            let other_end = other
                .physical
                .checked_add(other.bytes)
                .ok_or(refuse!("native retained reservation physical range wraps"))?;
            if region.physical < other_end && other.physical < end {
                return Err(refuse!(
                    "native guarded stacks alias another retained reservation"
                ));
            }
        }
        if handoff.guest_count > handoff.guest.len() {
            return Err(refuse!("native guest inventory exceeds handoff capacity"));
        }
        for guest in &handoff.guest[..handoff.guest_count] {
            let guest_end = guest
                .physical
                .checked_add(guest.bytes)
                .ok_or(refuse!("native guest backing range wraps"))?;
            if region.physical < guest_end && guest.physical < end {
                return Err(refuse!("native guarded stacks alias guest backing"));
            }
        }
    }
    for slot in [runners - 1, cores - 1] {
        let part =
            resources::partition(cores, runners, slot, handoff.code.bytes, handoff.pool.bytes)?;
        resources::code_part_bytes(part.code_bytes, resources.code_parts)?;
    }
    if (cores > 1 || resources.sipi != 0)
        && (!(crate::tables::PAGE..1 << 20).contains(&resources.sipi)
            || !resources.sipi.is_multiple_of(crate::tables::PAGE))
    {
        return Err(refuse!("native SIPI page is not owned low memory"));
    }
    // SAFETY: the caller keeps the immutable loader allocation.
    let capture = unsafe { firmware::captured(handoff.native_processors) }
        .map_err(firmware::Error::message)?;
    if cores > capture.core_count as usize {
        return Err(refuse!(
            "requested native cores exceed the source processor inventory"
        ));
    }
    let five_level = matches!(
        handoff.placement,
        stela_engine::machine::mmu::placement::Placement::FiveLevel
    );
    let assignable = (capture.logical_count as usize).min(topology::MAX_CORES);
    crate::output::arguments(format_args!(
        "NATIVE INVENTORY firmware_processors={} enabled={} physical_cores={} logical={} assignable={} retired={} requested={cores} runners={runners}\n",
        capture.processor_count,
        capture.enabled_count,
        topology::physical_cores(&capture.processors),
        capture.logical_count,
        assignable,
        capture.logical_count as usize - assignable,
    ));
    crate::host_inventory::number(
        "HostCores",
        topology::physical_cores(&capture.processors) as u64,
    );
    crate::host_inventory::number("HostThreads", capture.logical_count.into());
    let memory_types = memory_types::Image::from_snapshot(&capture.snapshots[0])?;
    for (slot, snapshot) in capture.snapshots[..cores].iter().enumerate() {
        let processor = capture.processors[capture.selected[slot] as usize];
        crate::output::arguments(format_args!(
            "NATIVE SOURCE slot={slot} apic={:#x} firmware_cpu={} flags={:#x} topology={} package={} module={} tile={} die={} core={} thread={} core_type={:?} cpuid_records={} msr_records={} pat_valid={} pat={:#x} cr0={:#x} cr3={:#x} cr4={:#x} efer={:#x}\n",
            processor.apic_id,
            processor.processor_number,
            processor.flags,
            processor.topology_version,
            processor.location.package,
            processor.location.module,
            processor.location.tile,
            processor.location.die,
            processor.location.core,
            processor.thread,
            snapshot.core_class(),
            snapshot.count,
            snapshot.msr_count,
            snapshot.pat_valid,
            snapshot.pat,
            snapshot.cr0,
            snapshot.cr3,
            snapshot.cr4,
            snapshot.efer,
        ));
        snapshot.qualify_native(five_level).map_err(|refusal| {
            refusal.refusal("a selected processor lacks the native executor floor")
        })?;
        memory_types.compatible(snapshot)?;
        if snapshot.same_memory_types(&capture.snapshots[0]).is_err() {
            let reference = &capture.snapshots[0];
            // Snapshot bounds were validated above. Record both source values
            // for each differing register; never normalize the admission input.
            for expected in &reference.msrs[..reference.msr_count as usize] {
                let actual = snapshot.msrs[..snapshot.msr_count as usize]
                    .iter()
                    .find(|actual| actual.index == expected.index);
                if actual.is_none_or(|actual| actual.value != expected.value) {
                    crate::output::arguments(format_args!(
                        "NATIVE SOURCE MISMATCH slot={slot} msr={:#x} bsp={:#x} ap={:?}\n",
                        expected.index,
                        expected.value,
                        actual.map(|value| value.value),
                    ));
                }
            }
            crate::output::arguments(format_args!(
                "NATIVE SOURCE slot={slot} requires initial offline-AP memory-type installation\n"
            ));
        }
    }
    Ok(Some(capture))
}
