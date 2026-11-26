//! Copy processor identity/capabilities while UEFI MP Services still owns APs.
//! Every callback returns before ExitBootServices; persistent executors use
//! Stela's own INIT/SIPI entry, stacks and control memory afterward.
pub(crate) use record::Capture;
use stela_boot_core::record;

use super::{
    memory_types::amd,
    topology::{self, Cpuid, Location, Msr, Processor, Snapshot},
};
use crate::{cpu::msr, handoff::Region, tables::PAGE};
use alloc::{boxed::Box, vec::Vec};
use core::{
    arch::{
        asm,
        x86_64::{__cpuid, __cpuid_count},
    },
    cell::UnsafeCell,
    ffi::c_void,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use uefi::{
    Handle,
    boot::{self, AllocateType, MemoryType},
    proto::pi::mp::{CPU_V2_EXTENDED_TOPOLOGY, MpServices},
};

const MAGIC: u64 = 0x5354_4c53_4d50_3031;
const VERSION: u32 = 5;
const MAX_VARIABLE_MTRRS: usize = 16;
const CR4_OSXSAVE: u64 = 1 << 18;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Error {
    NoMpServices,
    ProcessorInventory,
    Topology(topology::Refusal),
    CoreCount,
    WrongBootstrapProcessor,
    SnapshotCapacity,
    SnapshotCallback,
    SnapshotTimeout,
    ProcessorIdentity,
    Memory,
    AmdMemoryControls,
    /// A memory-encryption setting Stela cannot run under, and what to change.
    AmdEncryption(&'static str),
}

impl Error {
    pub(crate) fn message(self) -> &'static str {
        match self {
            Self::NoMpServices => "native cores need UEFI MP Services source identity",
            Self::ProcessorInventory => "native processor inventory is invalid",
            Self::Topology(_) => "native processor topology is ambiguous",
            Self::CoreCount => {
                "requested native count exceeds the available processors or the executor's capacity"
            }
            Self::WrongBootstrapProcessor => {
                "native BSP differs from the firmware processor identity"
            }
            Self::SnapshotCapacity => "native per-processor source observations exceed capacity",
            Self::SnapshotCallback => "firmware AP callback did not publish its source observation",
            Self::SnapshotTimeout => "firmware AP source observation did not finish before timeout",
            Self::ProcessorIdentity => {
                "native observed CPUID identity differs from firmware identity"
            }
            Self::Memory => "native source record allocation or extent is invalid",
            Self::AmdMemoryControls => "AMD fixed-attribute visibility controls are unqualified",
            Self::AmdEncryption(reason) => reason,
        }
    }
}

/// Opens only fixed-attribute read/write visibility, without changing routing,
/// type, encryption or latent attributes. The original control/IF is restored
/// before the joined firmware callback returns, including early errors.
struct FixedVisibility {
    syscfg: u64,
    interrupts: bool,
    changed: bool,
}
impl FixedVisibility {
    /// # Safety
    /// CPL0 AMD64 with observed MSR/MTRR/PAT controls, no concurrent writer on
    /// this processor. SME alone leaves the visibility control writable.
    unsafe fn open(fixed: bool) -> Result<Self, Error> {
        let flags: u64;
        // SAFETY: saves RFLAGS and masks interrupts on this CPU; the guard's drop restores IF.
        unsafe {
            asm!("pushfq", "pop {}", "cli", out(reg) flags);
        }
        // SAFETY: the caller guarantees CPL0 on AMD64, where SYSCFG exists.
        let original = unsafe { msr::AMD_SYSCFG.read() };
        let mut guard = Self {
            syscfg: original,
            interrupts: flags & (1 << 9) != 0,
            changed: false,
        };
        if let Some(reason) = amd::encryption_refusal(original) {
            return Err(Error::AmdEncryption(reason));
        }
        // With MtrrFixDramEn clear the fixed MTRRs' RdMem and WrMem bits have
        // no effect (AMD APM Vol. 2 7.7.4), so the type bits read without the
        // modify enable are the whole policy and SYSCFG stays untouched; a
        // hypervisor may refuse the write (KVM raises #GP).
        if fixed && original & amd::FIXED_DRAM != 0 && original & amd::FIXED_MODIFY == 0 {
            guard.changed = true;
            // SAFETY: the caller guarantees CPL0 on AMD64; only the fixed-MTRR modify enable
            // changes, and the guard restores it.
            unsafe {
                msr::AMD_SYSCFG.write(original | amd::FIXED_MODIFY);
            }
            core::sync::atomic::compiler_fence(Ordering::SeqCst);
            // SAFETY: the caller guarantees CPL0 on AMD64, where SYSCFG exists.
            if unsafe { msr::AMD_SYSCFG.read() } != original | amd::FIXED_MODIFY {
                return Err(Error::AmdMemoryControls);
            }
        }
        Ok(guard)
    }
}
impl Drop for FixedVisibility {
    fn drop(&mut self) {
        // SAFETY: this guard cannot migrate or escape the synchronous CPL0
        // observation. Only the original known control value is restored.
        unsafe {
            if self.changed {
                core::sync::atomic::compiler_fence(Ordering::SeqCst);
                msr::AMD_SYSCFG.write(self.syscfg);
            }
            if self.interrupts {
                asm!("sti", options(nomem, nostack));
            }
        }
    }
}

fn cpuid(leaf: u32, subleaf: u32) -> Cpuid {
    let result = __cpuid_count(leaf, subleaf);
    Cpuid {
        leaf,
        subleaf,
        eax: result.eax,
        ebx: result.ebx,
        ecx: result.ecx,
        edx: result.edx,
    }
}

fn push(snapshot: &mut Snapshot, record: Cpuid) -> Result<(), Error> {
    let slot = snapshot
        .records
        .get_mut(snapshot.count as usize)
        .ok_or(Error::SnapshotCapacity)?;
    *slot = record;
    snapshot.count += 1;
    Ok(())
}

fn push_msr(snapshot: &mut Snapshot, index: u32, value: u64) -> Result<(), Error> {
    let slot = snapshot
        .msrs
        .get_mut(snapshot.msr_count as usize)
        .ok_or(Error::SnapshotCapacity)?;
    *slot = Msr { index, value };
    snapshot.msr_count += 1;
    Ok(())
}

/// # Safety
/// Runs at CPL0 on this processor, either in a joined firmware
/// callback or under its owned native executor. No CPU feature is cached from
/// another processor, and unsupported leaves/MSRs are not probed.
pub(crate) unsafe fn observe() -> Result<Snapshot, Error> {
    let mut snapshot = Snapshot::default();
    // SAFETY: forwarded original CPU/control capture contract.
    unsafe { observe_controls(&mut snapshot, true) }?;
    Ok(snapshot)
}

/// [`observe`] into the heap, for callers on small stacks: Linux-compatible
/// work threads verify cache attributes, and two snapshots on their stack
/// overflowed it on AMD.
/// # Safety
/// The same contract as [`observe`].
pub(crate) unsafe fn observe_boxed() -> Result<Box<Snapshot>, Error> {
    let mut snapshot = Box::<Snapshot>::default();
    // SAFETY: forwarded original CPU/control capture contract.
    unsafe { observe_controls(&mut snapshot, true) }?;
    Ok(snapshot)
}

/// Observe the same installed CPU without changing its retained PAT.
/// # Safety
/// The same original CPU/MSR context as observe, with any required PAT
/// configuration already complete before its owner sealed the controls.
#[cfg(feature = "diagnostics")]
pub(crate) unsafe fn observe_with_original_pat() -> Result<Snapshot, Error> {
    let mut snapshot = Snapshot::default();
    // SAFETY: forwarded original CPU/control capture contract.
    unsafe { observe_controls(&mut snapshot, false) }?;
    Ok(snapshot)
}

/// # Safety
/// Current non-migrating CPL0 CPU, with exclusive control of its memory types.
/// Fills `snapshot`, which the caller passes as `Snapshot::default()`, in
/// place: a copy of its 2 KiB would cost the stack it exists to spare.
unsafe fn observe_controls(snapshot: &mut Snapshot, configure_pat: bool) -> Result<(), Error> {
    debug_assert_eq!((snapshot.count, snapshot.msr_count), (0, 0));
    let maximum = __cpuid(0).eax;
    let extended = __cpuid(0x8000_0000).eax;
    for leaf in [
        0,
        1,
        5,
        6,
        7,
        0x15,
        0x16,
        0x1a,
        0x8000_0000,
        0x8000_0001,
        0x8000_0007,
        0x8000_0008,
        0x8000_001e,
    ] {
        if leaf
            <= if leaf & 0x8000_0000 == 0 {
                maximum
            } else {
                extended
            }
        {
            push(snapshot, cpuid(leaf, 0))?;
        }
    }
    for leaf in [0xb, 0x1f, 0x8000_0026] {
        if leaf
            > if leaf & 0x8000_0000 == 0 {
                maximum
            } else {
                extended
            }
        {
            continue;
        }
        for subleaf in 0..8 {
            let record = cpuid(leaf, subleaf);
            push(snapshot, record)?;
            if record.ebx == 0 || record.ecx >> 8 & 0xff == 0 {
                break;
            }
            if subleaf == 7 {
                return Err(Error::SnapshotCapacity);
            }
        }
    }
    let vendor = snapshot.get(0, 0).ok_or(Error::ProcessorIdentity)?;
    let basic = snapshot.get(1, 0).ok_or(Error::ProcessorIdentity)?;
    if maximum >= 0xd && basic.ecx & (1 << 26) != 0 {
        let xstate = cpuid(0xd, 0);
        push(snapshot, xstate)?;
        push(snapshot, cpuid(0xd, 1))?;
        let components = u64::from(xstate.eax) | (u64::from(xstate.edx) << 32);
        for subleaf in [2, 5, 6, 7, 9, 17, 18] {
            if components & (1u64 << subleaf) != 0 {
                push(snapshot, cpuid(0xd, subleaf))?;
            }
        }
    }
    // SAFETY: CPL0 observation; these controls and EFER exist in x86-64.
    unsafe {
        asm!("mov {}, cr0",out(reg) snapshot.cr0,options(nomem,nostack,preserves_flags));
        asm!("mov {}, cr3",out(reg) snapshot.cr3,options(nomem,nostack,preserves_flags));
        asm!("mov {}, cr4",out(reg) snapshot.cr4,options(nomem,nostack,preserves_flags));
        snapshot.efer = msr::EFER.read();
        if basic.edx & (1 << 9) != 0 {
            snapshot.apic_base = msr::APIC_BASE.read();
        }
        if basic.edx & (1 << 16) != 0 {
            if configure_pat {
                // Before anything records PAT: every processor's stays the same.
                msr::PAT.write(stela_boot_core::page_window::with_write_combining(
                    msr::PAT.read(),
                ));
            }
            snapshot.pat = msr::PAT.read();
            snapshot.pat_valid = true;
        }
        if basic.edx & (1 << 12) != 0 {
            let cap = msr::MTRR_CAP.read();
            let variables = (cap & 0xff) as usize;
            if variables > MAX_VARIABLE_MTRRS {
                return Err(Error::SnapshotCapacity);
            }
            push_msr(snapshot, msr::MTRR_CAP.index, cap)?;
            push_msr(snapshot, msr::MTRR_DEFAULT.index, msr::MTRR_DEFAULT.read())?;
            let amd_controls = amd::architectural(snapshot);
            let visibility = if amd_controls {
                Some(FixedVisibility::open(cap & (1 << 8) != 0)?)
            } else {
                None
            };
            if cap & (1 << 8) != 0 {
                for index in [
                    0x250, 0x258, 0x259, 0x268, 0x269, 0x26a, 0x26b, 0x26c, 0x26d, 0x26e, 0x26f,
                ] {
                    push_msr(snapshot, index, msr::rdmsr(index))?;
                }
            }
            if let Some(visibility) = visibility {
                let original = visibility.syscfg;
                drop(visibility);
                if msr::AMD_SYSCFG.read() != original {
                    return Err(Error::AmdMemoryControls);
                }
                // Do not report masked fixed zeros as physical source state.
                snapshot.amd_fixed_raw = true;
                push_msr(snapshot, amd::SYSCFG, original)?;
                for index in amd::enabled(original) {
                    push_msr(snapshot, index, msr::rdmsr(index))?;
                }
            }
            for offset in 0..variables as u32 * 2 {
                let index = msr::MTRR_PHYS_BASE.at(offset).index;
                push_msr(snapshot, index, msr::rdmsr(index))?;
            }
        }
        // Pinned Linux hpet.c mwait_pc10_supported/hpet_is_pc10_damaged:
        // Intel advertises extended MWAIT+interrupt-break and PC10 substates
        // before the package C-state limit MSR is observed. Neither ordinary
        // leaf5 support nor a table alone authorizes this conditional RDMSR.
        if [vendor.ebx, vendor.edx, vendor.ecx]
            == [
                u32::from_le_bytes(*b"Genu"),
                u32::from_le_bytes(*b"ineI"),
                u32::from_le_bytes(*b"ntel"),
            ]
            && basic.ecx & (1 << 3) != 0
            && snapshot
                .get(5, 0)
                .is_some_and(|record| record.ecx & 3 == 3 && record.edx & (0xf << 28) != 0)
        {
            // SAFETY: Intel CPU's PC10 MWAIT enumeration is the pinned
            // Linux gate for MSR_PKG_CST_CONFIG_CONTROL, on this same CPL0 CPU.
            push_msr(snapshot, 0xe2, msr::rdmsr(0xe2))?;
        }
        if snapshot
            .get(7, 0)
            .is_some_and(|record| record.ebx & (1 << 1) != 0)
        {
            push_msr(snapshot, msr::TSC_ADJUST.index, msr::TSC_ADJUST.read())?;
        }
        if basic.ecx & ((1 << 26) | (1 << 27)) == (1 << 26) | (1 << 27)
            && snapshot.cr4 & CR4_OSXSAVE != 0
        {
            let (low, high): (u32, u32);
            asm!("xgetbv",in("ecx") 0u32,out("eax") low,out("edx") high,options(nomem,nostack,preserves_flags));
            snapshot.xcr0 = u64::from(low) | (u64::from(high) << 32);
            snapshot.xcr0_valid = true;
        }
    }
    Ok(())
}

struct Callback {
    observation: UnsafeCell<Option<Result<Snapshot, Error>>>,
    complete: AtomicBool,
}

extern "efiapi" fn observe_ap(argument: *mut c_void) {
    // SAFETY: blocking StartupThisAP keeps this one callback record until
    // it returns or the firmware terminates the timed-out callback.
    unsafe {
        let callback = &*argument.cast::<Callback>();
        *callback.observation.get() = Some(observe());
        callback.complete.store(true, Ordering::Release);
    }
}

/// Find the SMT and package boundary shifts in the BSP's CPUID topology.
/// Prefer leaf 1FH, whose final level also separates dies and tiles.
fn apic_id_shifts(maximum: u32) -> Option<(u32, u32)> {
    for leaf in [0x1f, 0xb] {
        if leaf > maximum {
            continue;
        }
        let mut thread = None;
        let mut package = None;
        for subleaf in 0..8 {
            let level = __cpuid_count(leaf, subleaf);
            let kind = level.ecx >> 8 & 0xff;
            if kind == 0 || level.ebx & 0xffff == 0 {
                break;
            }
            let shift = level.eax & 0x1f;
            if kind == 1 {
                thread = Some(shift);
            } else {
                package = Some(shift);
            }
        }
        if let (Some(thread), Some(package)) = (thread, package)
            && thread <= package
        {
            return Some((thread, package));
        }
    }
    None
}

pub(crate) struct Inventory {
    handle: Handle,
    processors: Vec<Processor>,
    logical: Vec<usize>,
    pub(crate) enabled: usize,
}

impl Inventory {
    pub(crate) fn logical_count(&self) -> usize {
        self.logical.len()
    }
}

/// Discover the complete firmware roster without running callbacks on APs.
pub(crate) fn inventory() -> Result<Inventory, Error> {
    let handle = boot::get_handle_for_protocol::<MpServices>().map_err(|_| Error::NoMpServices)?;
    let mp =
        boot::open_protocol_exclusive::<MpServices>(handle).map_err(|_| Error::NoMpServices)?;
    let count = mp
        .get_number_of_processors()
        .map_err(|_| Error::ProcessorInventory)?;
    if count.total == 0 || count.total > u32::MAX as usize || count.enabled > count.total {
        return Err(Error::ProcessorInventory);
    }
    let mut processors = Vec::with_capacity(count.total);
    // Either the complete inventory uses V2 coordinates or it uses legacy
    // coordinates; mixing flattened and hierarchical core keys refuses.
    let extended = mp.get_processor_info(CPU_V2_EXTENDED_TOPOLOGY).is_ok();
    for number in 0..count.total {
        let info = mp
            .get_processor_info(
                number
                    | if extended {
                        CPU_V2_EXTENDED_TOPOLOGY
                    } else {
                        0
                    },
            )
            .map_err(|_| Error::ProcessorInventory)?;
        let mut flags = 0;
        if info.is_bsp() {
            flags |= topology::BSP;
        }
        if info.is_enabled() {
            flags |= topology::ENABLED;
        }
        if info.is_healthy() {
            flags |= topology::HEALTHY;
        }
        let (location, thread) = if extended {
            let value = info.extended_information;
            (
                Location {
                    package: value.package,
                    module: value.module,
                    tile: value.tile,
                    die: value.die,
                    core: value.core,
                },
                value.thread,
            )
        } else {
            let value = info.location;
            (
                Location {
                    package: value.package,
                    core: value.core,
                    ..Location::default()
                },
                value.thread,
            )
        };
        processors.push(Processor {
            processor_number: number as u32,
            apic_id: info.processor_id,
            flags,
            topology_version: if extended { 2 } else { 1 },
            location,
            thread,
        });
    }
    if count.enabled
        != processors
            .iter()
            .filter(|processor| processor.flags & topology::ENABLED != 0)
            .count()
    {
        return Err(Error::ProcessorInventory);
    }
    let vendor = __cpuid(0);
    let shifts = apic_id_shifts(vendor.eax);
    // Intel's topology leaves describe the package boundary even when the
    // firmware calls cores from two sockets one package with global core IDs.
    let intel = (vendor.ebx, vendor.edx, vendor.ecx) == (0x756e_6547, 0x4965_6e69, 0x6c65_746e);
    let mismatched_packages = intel
        && shifts.is_some_and(|(_, shift)| !topology::packages_match_apic_ids(&processors, shift));
    let logical = match topology::select(&processors) {
        Ok(logical) if !mismatched_packages => logical,
        Ok(_) | Err(topology::Refusal::DuplicateThreadLocation) => {
            let (thread, package) =
                shifts.ok_or(Error::Topology(topology::Refusal::DuplicateThreadLocation))?;
            topology::place_by_apic_id(&mut processors, thread, package)
                .map_err(Error::Topology)?;
            let logical = topology::select(&processors).map_err(Error::Topology)?;
            crate::output::arguments(format_args!(
                "HOST PROCESSORS firmware coordinates conflict; reconstructed from APIC IDs \
                 (SMT shift={thread}, package shift={package})\n"
            ));
            logical
        }
        Err(error) => return Err(Error::Topology(error)),
    };
    if mp.who_am_i().map_err(|_| Error::WrongBootstrapProcessor)?
        != processors[logical[0]].processor_number as usize
    {
        return Err(Error::WrongBootstrapProcessor);
    }
    Ok(Inventory {
        handle,
        processors,
        logical,
        enabled: count.enabled,
    })
}

/// Capture the selected guest runners and spare native workers. If an AP
/// cannot be observed during an ordinary boot, keep the verified prefix and
/// retain the complete inventory for the post-EBS firmware handoff.
pub(crate) fn capture(
    inventory: Inventory,
    requested: usize,
    exact: bool,
) -> Result<(Region, usize), Error> {
    let Inventory {
        handle,
        processors,
        logical,
        enabled,
    } = inventory;
    let mp =
        boot::open_protocol_exclusive::<MpServices>(handle).map_err(|_| Error::NoMpServices)?;
    topology::assignment_count(Some(requested), logical.len()).map_err(|_| Error::CoreCount)?;
    let selected = &logical[..requested];
    let mut snapshots = Vec::with_capacity(selected.len());
    for (slot, &index) in selected.iter().enumerate() {
        let processor = processors[index];
        let observed = if slot == 0 {
            // SAFETY: UEFI executes the loader at CPL0 on the recorded BSP.
            unsafe { observe() }
        } else {
            observe_one(&mp, processor.processor_number as usize)
        };
        let observed = observed.and_then(|snapshot| {
            snapshot
                .matches(processor)
                .then_some(snapshot)
                .ok_or(Error::ProcessorIdentity)
        });
        match observed {
            Ok(snapshot) => snapshots.push(snapshot),
            Err(error) if slot != 0 && !exact => {
                crate::output::arguments(format_args!(
                    "NATIVE FIRMWARE SOURCE refused slot={slot} apic={:#x} firmware_cpu={} error={error:?}; keeping {slot} workers\n",
                    processor.apic_id, processor.processor_number,
                ));
                break;
            }
            Err(error) => return Err(error),
        }
    }
    let selected = &selected[..snapshots.len()];
    let pages = record_pages(processors.len()).ok_or(Error::Memory)?;
    let memory = boot::allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, pages)
        .map_err(|_| Error::Memory)?;
    // SAFETY: fresh owned firmware pages; all-zero is valid for this closed
    // record of integers/booleans/arrays. No large intermediate sits on a UEFI
    // stack. All protocol/callback/Vec borrows end before ExitBootServices.
    unsafe {
        memory.as_ptr().write_bytes(0, pages * PAGE as usize);
        let capture = Capture::from_raw_mut(memory.as_ptr(), processors.len());
        capture.magic = MAGIC;
        capture.version = VERSION;
        capture.processor_count = processors.len() as u32;
        capture.enabled_count = enabled as u32;
        capture.core_count = selected.len() as u32;
        capture.logical_count = logical.len() as u32;
        capture.processors.copy_from_slice(&processors);
        for (slot, &index) in selected.iter().enumerate() {
            capture.selected[slot] = index as u32;
            capture.snapshots[slot] = snapshots[slot];
        }
    }
    Ok((
        Region {
            physical: memory.as_ptr() as u64,
            bytes: (pages * PAGE as usize) as u64,
        },
        selected.len(),
    ))
}

fn observe_one(mp: &MpServices, processor_number: usize) -> Result<Snapshot, Error> {
    let mut callback = Callback {
        observation: UnsafeCell::new(None),
        complete: AtomicBool::new(false),
    };
    mp.startup_this_ap(
        processor_number,
        observe_ap,
        (&raw mut callback).cast(),
        None,
        Some(Duration::from_millis(500)),
    )
    .map_err(|_| Error::SnapshotTimeout)?;
    if !callback.complete.load(Ordering::Acquire) {
        return Err(Error::SnapshotCallback);
    }
    callback
        .observation
        .into_inner()
        .ok_or(Error::SnapshotCallback)?
}

/// Whole pages holding a record of `processors` inventory entries.
fn record_pages(processors: usize) -> Option<usize> {
    Some(Capture::layout(processors)?.size().div_ceil(PAGE as usize))
}

/// # Safety
/// This is the loader's immutable owned record, mapped in the engine physmap.
pub(crate) unsafe fn captured(region: Region) -> Result<&'static Capture, Error> {
    let header_pages = record_pages(0).ok_or(Error::Memory)?;
    if region.bytes < (header_pages as u64) * PAGE
        || region.physical == 0
        || !region.physical.is_multiple_of(PAGE)
    {
        return Err(Error::Memory);
    }
    let base = crate::layout::physical(region.physical) as *const u8;
    // SAFETY: caller keeps the exact loader allocation and its physmap; the
    // header-only view lies within the checked region.
    let count = unsafe { Capture::from_raw(base, 0) }.processor_count as usize;
    if record_pages(count).map(|pages| pages as u64 * PAGE) != Some(region.bytes) {
        return Err(Error::Memory);
    }
    // SAFETY: the region holds exactly the record for `count` entries.
    let capture = unsafe { Capture::from_raw(base, count) };
    let logical = capture.logical_count as usize;
    if capture.magic != MAGIC
        || capture.version != VERSION
        || topology::assignment_count(Some(capture.core_count as usize), logical).is_err()
    {
        return Err(Error::ProcessorInventory);
    }
    let processors = &capture.processors;
    let all = topology::select(processors).map_err(Error::Topology)?;
    let selected = all
        .get(..capture.core_count as usize)
        .ok_or(Error::ProcessorInventory)?;
    if all.len() != logical
        || capture.enabled_count as usize
            != processors
                .iter()
                .filter(|processor| processor.flags & topology::ENABLED != 0)
                .count()
    {
        return Err(Error::ProcessorInventory);
    }
    for (slot, &selected_index) in selected.iter().enumerate() {
        let index = capture.selected[slot] as usize;
        if selected_index != index
            || capture.snapshots[slot].msr_count as usize > topology::MAX_MSRS
            || !capture.snapshots[slot].matches(capture.processors[index])
        {
            return Err(Error::ProcessorIdentity);
        }
    }
    Ok(capture)
}
