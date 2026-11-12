//! Persistent storage service across the engine's shadow-page-table roots.
//!
//! PC drivers currently use identity addresses for their ECAM/MMIO and DMA
//! buffers. Keep their accesses in the loader's root, which also
//! maps the engine window, and restore the current engine root afterwards.
//! This boundary never maps host identity addresses into a guest root. All
//! buffers passed here belong to the engine window, not guest virtual RAM.
//!
//! The drivers keep each controller's state in `Rc` and `RefCell`, for one
//! owner at a time, while the service and every PE reach host devices from
//! cores of their own: a USB disk the guest drives and the USB keyboard the
//! service reads share one xHCI controller. [`RestoreRoot`] is the only way
//! into a driver, so it also holds the host drivers' lock, which serializes
//! them and orders each holder's writes before the next holder's reads.
//!
//! While the engine runs ([`TaggedRoots`]), the loader's root has a TLB
//! context of its own, so a driver call keeps the engine's translations.
use crate::Result;
use crate::output;
use alloc::{boxed::Box, format, string::String, vec::Vec};
use core::{
    arch::asm,
    cell::Cell,
    panic::Location,
    ptr,
    sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, Ordering},
};
use stela_boot_core::staged_disk::{self, DiskWorker};
use stela_contracts::refuse;
use stela_drivers::{
    Handoff, PlatformCaptures,
    block::{
        BlockDevice, BlockError, CallbackStats, Command, Direction, Finished, Identity, Shared,
        Window,
    },
    bus::dma::RegionAllocator,
    input::InputDevice,
    probe::{NativeServices, native::usb4, probe_with_policy, registry::DriverPolicy},
};
use stela_runtime::pc::{
    failure_record,
    gpt::{Guid, PartitionDevice},
    machine_state,
    storage::{self, Named, Target},
};

/// CR4.PGE: entries of [`tables::GLOBAL`] pages outlive CR3 writes.
const PGE: u64 = 1 << 7;
/// CR4.PCIDE: TLB entries tagged by their root's context.
const PCIDE: u64 = 1 << 17;
/// CR4.SMAP: supervisor data accesses stay out of user pages unless
/// RFLAGS.AC is set, which the host's views use for PAN.
const SMAP: u64 = 1 << 21;
/// CR3's context, which holds PWT and PCD without PCIDE.
const CONTEXT: u64 = 0xfff;
/// The loader root's context while tagged; the host root keeps 0 and the
/// guest views' roots have their own (`host::view_context`).
const LOADER_CONTEXT: u64 = 1;
/// A CR3 write that keeps the entering context's TLB entries.
pub(super) const KEEP_ENTRIES: u64 = 1 << 63;

/// Keeps this core's TLB entries across root changes while the engine runs:
/// the window's in global pages, on every processor, and with PCIDs the
/// others by root, and with SMAP spares the changes PAN would make. Entering the loader's root ([`RestoreRoot`]) then flushes
/// only its own context and leaving it flushes nothing, so the engine's
/// translations survive a call into the host drivers; no engine mapping
/// changes inside one. Dropped, CR4 is restored for the native validations,
/// which require PCIDE clear.
pub(super) struct TaggedRoots {
    cr4: u64,
    tagged: bool,
}
impl TaggedRoots {
    /// Global pages, and tagged roots unless the processor has no PCIDs or
    /// the current root's context bits hold cache controls.
    pub(super) fn enable() -> Option<Self> {
        let (cr3, cr4): (u64, u64);
        // SAFETY: CPL0 control register reads.
        unsafe {
            asm!("mov {}, cr3", out(reg) cr3, options(nomem, nostack, preserves_flags));
            asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags));
        }
        if cr4 & PCIDE != 0 {
            return None;
        }
        let tagged = core::arch::x86_64::__cpuid(1).ecx & 1 << 17 != 0 && cr3 & CONTEXT == 0;
        let smap = super::host::views_installed()
            && core::arch::x86_64::__get_cpuid_max(0).0 >= 7
            && core::arch::x86_64::__cpuid_count(7, 0).ebx & 1 << 20 != 0;
        let enabled = cr4 | PGE | if tagged { PCIDE } else { 0 } | if smap { SMAP } else { 0 };
        // SAFETY: every processor that runs this has PGE, and PCIDE is set
        // only with PCIDs and the current root context 0. With SMAP, the
        // host leaves no page it touches a user page, and the loader root's
        // calls run with AC set ([`RestoreRoot`]).
        unsafe { asm!("mov cr4, {}", in(reg) enabled, options(nostack, preserves_flags)) };
        super::host::tag_views(tagged, smap);
        Some(Self { cr4, tagged })
    }
}
impl Drop for TaggedRoots {
    fn drop(&mut self) {
        super::host::tag_views(false, false);
        let cr3: u64;
        // SAFETY: outside a driver call; PCIDE clears only under context 0,
        // and clearing it flushes every context, as clearing PGE does.
        unsafe {
            if self.tagged {
                asm!("mov {}, cr3", out(reg) cr3, options(nomem, nostack, preserves_flags));
                asm!("mov cr3, {}", in(reg) cr3 & !CONTEXT, options(nostack, preserves_flags));
            }
            asm!("mov cr4, {}", in(reg) self.cr4, options(nostack, preserves_flags));
        }
    }
}

/// Held while a core is inside the host drivers ([`RestoreRoot`]).
static HOST_DRIVERS: AtomicBool = AtomicBool::new(false);

/// The parts of a host disk call timed for [`HostDisks::report`].
#[derive(Clone, Copy)]
enum DiskPart {
    /// Waiting for the host drivers' lock (any caller).
    Wait,
    /// Changing to the loader's root (any caller).
    Root,
    /// A disk's start, finish and completion look, inside the drivers.
    Start,
    Finish,
    Posted,
    /// A staged disk's whole step.
    Step,
}

const DISK_PARTS: [&str; 6] = ["wait", "root", "start", "finish", "posted", "step"];

/// The longest of each [`DiskPart`] since the last report, in TSC ticks.
static DISK_TIMES: [AtomicU64; DISK_PARTS.len()] = [const { AtomicU64::new(0) }; DISK_PARTS.len()];

/// Each [`DiskPart`]'s calls and TSC ticks in all since the last report:
/// with the longest, what a call costs on average.
static DISK_TOTALS: [[AtomicU64; 2]; DISK_PARTS.len()] =
    [const { [const { AtomicU64::new(0) }; 2] }; DISK_PARTS.len()];

/// Count a call of `part` that took `ticks`.
fn record(part: DiskPart, ticks: u64) {
    DISK_TIMES[part as usize].fetch_max(ticks, Ordering::Relaxed);
    DISK_TOTALS[part as usize][0].fetch_add(1, Ordering::Relaxed);
    DISK_TOTALS[part as usize][1].fetch_add(ticks, Ordering::Relaxed);
}

/// Run `call`, counting what it took as `part`.
fn timed<T>(part: DiskPart, call: impl FnOnce() -> T) -> T {
    let started = super::host::rdtsc();
    let result = call();
    record(part, super::host::rdtsc().wrapping_sub(started));
    result
}

/// Whether a core is inside the host drivers now.
pub(super) fn host_drivers_taken() -> bool {
    HOST_DRIVERS.load(Ordering::Relaxed)
}
static HOST_DRIVERS_GS: AtomicU64 = AtomicU64::new(0);

/// Validate the existing manager's same-CPU entered loader context.
/// Allocation and CPU permission remain with the caller's owner.
pub(super) fn entered_loader_root(owner: &crate::paging::loader::LoaderTables) -> bool {
    if !HOST_DRIVERS.load(Ordering::Acquire) {
        return false;
    }
    let root: u64;
    // SAFETY: this internal query follows RootIdentity's CPL0/CPU admission.
    unsafe {
        asm!("mov {}, cr3", out(reg) root, options(nomem, nostack, preserves_flags));
    }
    // RootIdentity's admission is of an activated processor.
    let gs = super::host::activated_base();
    // The context is 0 only without PCIDs, whose CR3 holds none.
    let context = root & CONTEXT;
    gs != 0
        && HOST_DRIVERS_GS.load(Ordering::Acquire) == gs
        && (context == LOADER_CONTEXT || context == 0)
        && root & !CONTEXT == owner.root()
}
/// Whether this processor is inside the host drivers: a fault stopped it
/// there, as no driver calls back into Stela's output.
pub(super) fn host_drivers_held_here() -> bool {
    if !HOST_DRIVERS.load(Ordering::Acquire) {
        return false;
    }
    // SAFETY: CPL0 read of this processor's own GS base.
    let gs = unsafe { crate::cpu::msr::GS_BASE.read() };
    gs != 0 && HOST_DRIVERS_GS.load(Ordering::Acquire) == gs
}
/// Counter ticks cores waited for [`HOST_DRIVERS`] since the last report,
/// and the longest wait.
static HOST_DRIVERS_WAITED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

fn try_lock_host_drivers() -> bool {
    HOST_DRIVERS
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_ok()
}

/// Cores spinning for [`HOST_DRIVERS`] now ([`host_drivers_wanted`]).
static HOST_DRIVERS_WAITERS: AtomicU32 = AtomicU32::new(0);

/// Whether a core waits for the host drivers' lock now: a holder that runs
/// work in steps (the GPU pump) ends its turn at the next step boundary
/// instead of at its budget, leaving the rest pending.
pub(super) fn host_drivers_wanted() -> bool {
    HOST_DRIVERS_WAITERS.load(Ordering::Relaxed) != 0
}

/// Where the current holder of [`HOST_DRIVERS`] took it, for the report
/// of a wait that overran.
static HOST_DRIVERS_SITE: AtomicPtr<Location<'static>> = AtomicPtr::new(ptr::null_mut());

/// The longest wait for [`HOST_DRIVERS`], in seconds: longer than the
/// longest single timeout of a host driver (a controller's 60 s ready
/// wait), so only a holder that never lets go reaches it.
const HOST_DRIVERS_WAIT_SECONDS: u64 = 90;

/// Take the host drivers' lock, counting the wait. A wait past
/// [`HOST_DRIVERS_WAIT_SECONDS`] stops the mount with a report naming the
/// holder instead of spinning forever.
fn lock_host_drivers() -> u64 {
    if try_lock_host_drivers() {
        return 0;
    }
    // An unknown rate is taken as a fast one, so the bound only grows.
    let hz = match crate::firmware::tsc_hz() {
        0 => 10_000_000_000,
        hz => hz,
    };
    HOST_DRIVERS_WAITERS.fetch_add(1, Ordering::Relaxed);
    let spun = stela_boot_core::bounded_spin::spin(
        || {
            HOST_DRIVERS
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        },
        super::host::rdtsc,
        hz.saturating_mul(HOST_DRIVERS_WAIT_SECONDS),
        || {
            // A driver holding this lock can be retiring an aperture mapping.
            // An interrupt cannot reach this waiter if it entered with IF clear.
            crate::native_mmio::shootdown::service();
            core::hint::spin_loop();
        },
    );
    HOST_DRIVERS_WAITERS.fetch_sub(1, Ordering::Relaxed);
    let waited = match spun {
        stela_boot_core::bounded_spin::Spin::Taken(waited) => waited,
        stela_boot_core::bounded_spin::Spin::Overrun(waited) => {
            let site = HOST_DRIVERS_SITE.load(Ordering::Relaxed);
            // SAFETY: only `enter_locked` stores, a `&'static Location`.
            let site = unsafe { site.as_ref() };
            output::fatal(format_args!(
                "host drivers lock not released after {} ms: held from {} on the processor \
                 with gs {:#x}",
                u128::from(waited) * 1000 / u128::from(hz),
                site.map_or_else(|| String::from("unknown"), |site| format!("{site}")),
                HOST_DRIVERS_GS.load(Ordering::Relaxed),
            ))
        }
    };
    HOST_DRIVERS_WAITED[0].fetch_add(waited, Ordering::Relaxed);
    HOST_DRIVERS_WAITED[1].fetch_max(waited, Ordering::Relaxed);
    waited
}

/// The longest hold of [`HOST_DRIVERS`] since the last report, in TSC
/// ticks, and where it was taken: a holder that waits for a slow device
/// holds every other core's driver call, the input service's included.
static HOST_DRIVERS_HELD: AtomicU64 = AtomicU64::new(0);
static HOST_DRIVERS_HOLDER: AtomicPtr<Location<'static>> = AtomicPtr::new(ptr::null_mut());

/// The time cores waited for the host drivers since the last report, in
/// total and at most, in microseconds, and the longest hold, with where
/// it was taken.
pub(super) fn host_drivers_waited(
    tsc_hz: u64,
) -> (u128, u128, u128, Option<&'static Location<'static>>) {
    let micros = |slot: &AtomicU64| {
        u128::from(slot.swap(0, Ordering::Relaxed)) * 1_000_000 / u128::from(tsc_hz.max(1))
    };
    let holder = HOST_DRIVERS_HOLDER.swap(ptr::null_mut(), Ordering::Relaxed);
    (
        micros(&HOST_DRIVERS_WAITED[0]),
        micros(&HOST_DRIVERS_WAITED[1]),
        micros(&HOST_DRIVERS_HELD),
        // SAFETY: only `RestoreRoot::drop` stores, a `&'static Location`.
        unsafe { holder.as_ref() },
    )
}

/// Inside the host drivers: the loader's root, with interrupts masked and
/// the host drivers' lock held, until dropped.
pub(super) struct RestoreRoot {
    root: u64,
    flags: u64,
    tagged: bool,
    /// When this core took the lock, and where.
    entered: u64,
    site: &'static Location<'static>,
    _local: core::marker::PhantomData<*mut ()>,
}
impl RestoreRoot {
    /// Wait for the host drivers' lock, which no holder takes again: a
    /// driver never calls back into the engine's host interfaces.
    ///
    /// # Safety
    /// `root` is the live loader root, keeping the engine window. The
    /// processor's pinned local record is active.
    #[track_caller]
    pub(super) unsafe fn enter(root: u64) -> Self {
        let waited = lock_host_drivers();
        record(DiskPart::Wait, waited);
        // SAFETY: the caller supplies the live root, and this core holds the lock.
        unsafe { Self::enter_locked(root) }
    }

    /// Enter for input service, attributing this acquisition's actual
    /// contention to the current phase in a diagnostics build.
    ///
    /// # Safety
    /// As for [`Self::enter`].
    #[inline]
    #[track_caller]
    pub(super) unsafe fn enter_input(
        root: u64,
        #[cfg(feature = "diagnostics")] wait: &mut u64,
    ) -> Self {
        #[cfg(feature = "diagnostics")]
        {
            let waited = lock_host_drivers();
            // SAFETY: the caller supplies the live root, and this core holds the lock.
            let guard = unsafe { Self::enter_locked(root) };
            *wait = wait.saturating_add(waited);
            guard
        }
        #[cfg(not(feature = "diagnostics"))]
        // SAFETY: same loader root and pinned-core requirement as enter.
        unsafe {
            Self::enter(root)
        }
    }

    /// Enter for work that can wait until another service turn.
    ///
    /// # Safety
    /// As for [`Self::enter`].
    #[track_caller]
    pub(super) unsafe fn try_enter(root: u64) -> Option<Self> {
        if !try_lock_host_drivers() {
            // The holder may need this processor to acknowledge a shootdown.
            crate::native_mmio::shootdown::service();
            return None;
        }
        // SAFETY: the caller supplies the live root, and this core holds the lock.
        Some(unsafe { Self::enter_locked(root) })
    }

    /// # Safety
    /// The caller holds the host drivers' lock and meets [`Self::enter`]'s
    /// requirements.
    #[track_caller]
    unsafe fn enter_locked(root: u64) -> Self {
        let (previous, flags): (u64, u64);
        // SAFETY: CPL0 control register read; interrupts remain off until
        // the original root has been restored.
        unsafe {
            asm!("pushfq","pop {flags}","cli","mov {previous}, cr3",flags=out(reg)flags,previous=out(reg)previous);
        }
        // SAFETY: native CPU; this records the same concrete manager
        // owner while IF0 excludes its own IRQ/reentrant root transition.
        let gs = unsafe { crate::cpu::msr::GS_BASE.read() };
        HOST_DRIVERS_GS.store(gs, Ordering::Release);
        HOST_DRIVERS_SITE.store(
            ptr::from_ref(Location::caller()).cast_mut(),
            Ordering::Relaxed,
        );
        // A context or the host's state says so without a read of CR4, which
        // a hypervisor may intercept: disks are polled here thousands of times
        // a second.
        let tagged = previous & CONTEXT != 0
            || super::host::views_tagged().unwrap_or_else(|| {
                let cr4: u64;
                // SAFETY: CPL0 control register read.
                unsafe {
                    asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags))
                };
                cr4 & PCIDE != 0
            });
        super::host::allow_user_pages();
        // Tagged, entering flushes the loader's context alone.
        let root = if tagged {
            root & !CONTEXT | LOADER_CONTEXT
        } else {
            root
        };
        let switching = super::host::rdtsc();
        // SAFETY: live roots with the same engine code/stack/window.
        unsafe { asm!("mov cr3, {}", in(reg) root, options(nostack, preserves_flags)) };
        let entered = super::host::rdtsc();
        record(DiskPart::Root, entered.wrapping_sub(switching));
        Self {
            root: previous,
            flags,
            tagged,
            entered,
            site: Location::caller(),
            _local: core::marker::PhantomData,
        }
    }
}
impl Drop for RestoreRoot {
    fn drop(&mut self) {
        // Tagged, the engine root's entries are still its own: nothing in
        // a driver call changes an engine mapping.
        let root = if self.tagged {
            self.root | KEEP_ENTRIES
        } else {
            self.root
        };
        // SAFETY: the original live root; no borrowed identity address
        // survives a BlockDevice call.
        unsafe {
            asm!("mov cr3, {}",in(reg)root,options(nostack,preserves_flags));
        }
        HOST_DRIVERS_GS.store(0, Ordering::Release);
        let held = super::host::rdtsc().wrapping_sub(self.entered);
        if HOST_DRIVERS_HELD.fetch_max(held, Ordering::Relaxed) < held {
            HOST_DRIVERS_HOLDER.store(ptr::from_ref(self.site).cast_mut(), Ordering::Relaxed);
        }
        HOST_DRIVERS.store(false, Ordering::Release);
        // SAFETY: only IF was changed by enter().
        unsafe {
            if self.flags & (1 << 9) != 0 {
                asm!("sti", options(nomem, nostack));
            }
        }
    }
}

/// Enter the real host-driver lock for the native contention diagnostic.
///
/// # Safety
/// `root` is the kept loader root, with the current processor installed.
#[cfg(feature = "diagnostics")]
pub(crate) unsafe fn with_driver_root<T>(root: u64, operation: impl FnOnce() -> T) -> T {
    // SAFETY: the caller supplies the live root and local owner.
    let _guard = unsafe { RestoreRoot::enter(root) };
    operation()
}

struct Service {
    inner: Box<dyn BlockDevice>,
    root: u64,
    /// Commands started and not yet finished. With none no completion can be
    /// posted, and a look at the disk need not change roots, which without
    /// PCIDs flushes this core's TLB.
    running: u32,
    /// Whether the disk signals its completions, asked once.
    signals: Cell<Option<bool>>,
    /// The last look that found nothing ([`staged_disk::LookGate`]): it
    /// excuses looks for [`LOOK_WINDOW_MICROS`] while no doorbell moves, as
    /// a busy core takes the disk's signal only at its next interrupt window.
    looked: Cell<staged_disk::LookGate>,
    /// Host commands that failed, which the guest sees only as its
    /// controller's error status.
    failures: u64,
}

impl Service {
    /// Logs a failed host command: the first few, then each power-of-two
    /// count, so a failing disk stays visible without flooding the log.
    fn failed<T>(
        &mut self,
        operation: &str,
        at: u64,
        result: Result<T, BlockError>,
    ) -> Result<T, BlockError> {
        if let Err(error) = &result {
            self.failures += 1;
            if self.failures <= 16 || self.failures.is_power_of_two() {
                output::arguments(format_args!(
                    "HOST DISK ERROR serial={} {operation}={at}: {error} failures={}\n",
                    self.inner.identity().serial,
                    self.failures
                ));
            }
        }
        result
    }
}

/// The first block `command` reaches, for a report of its failure.
fn command_lba(command: &Command) -> u64 {
    match command {
        Command::Read { lba, .. }
        | Command::Write { lba, .. }
        | Command::WriteZeroes { lba, .. } => *lba,
        Command::Discard { extents } => extents.first().map_or(0, |extent| extent.lba),
        Command::Flush => 0,
    }
}

impl BlockDevice for Service {
    fn identity(&self) -> &Identity {
        self.inner.identity()
    }
    fn read(&mut self, lba: u64, buffer: &mut [u8]) -> Result<(), BlockError> {
        let read = {
            // SAFETY: open() captured the loader root; callers supply window RAM.
            let _root = unsafe { RestoreRoot::enter(self.root) };
            self.inner.read(lba, buffer)
        };
        // Reported with the loader's root and interrupts restored.
        self.failed("read lba", lba, read)
    }
    fn write(&mut self, lba: u64, data: &[u8]) -> Result<(), BlockError> {
        let written = {
            // SAFETY: as in read().
            let _root = unsafe { RestoreRoot::enter(self.root) };
            self.inner.write(lba, data)
        };
        // Reported with the loader's root and interrupts restored.
        self.failed("write lba", lba, written)
    }
    fn flush_cache(&mut self) -> Result<(), BlockError> {
        let flushed = {
            // SAFETY: as in read().
            let _root = unsafe { RestoreRoot::enter(self.root) };
            self.inner.flush_cache()
        };
        // Reported with the loader's root and interrupts restored.
        self.failed("flush lba", 0, flushed)
    }
    fn write_fua(&mut self, lba: u64, data: &[u8]) -> Result<(), BlockError> {
        let written = {
            // SAFETY: as in read().
            let _root = unsafe { RestoreRoot::enter(self.root) };
            self.inner.write_fua(lba, data)
        };
        // Reported with the loader's root and interrupts restored.
        self.failed("write_fua lba", lba, written)
    }
    fn discard(&mut self, lba: u64, blocks: u64) -> Result<(), BlockError> {
        let discarded = {
            // SAFETY: as in read().
            let _root = unsafe { RestoreRoot::enter(self.root) };
            self.inner.discard(lba, blocks)
        };
        // Reported with the loader's root and interrupts restored.
        self.failed("discard lba", lba, discarded)
    }
    fn write_zeroes(&mut self, lba: u64, blocks: u64, unmap: bool) -> Result<(), BlockError> {
        let zeroed = {
            // SAFETY: as in read().
            let _root = unsafe { RestoreRoot::enter(self.root) };
            self.inner.write_zeroes(lba, blocks, unmap)
        };
        // Reported with the loader's root and interrupts restored.
        self.failed("write_zeroes lba", lba, zeroed)
    }
    fn depth(&self) -> usize {
        self.inner.depth()
    }
    fn room(&self) -> bool {
        self.inner.room()
    }
    fn start(&mut self, command: Command) -> Result<u32, BlockError> {
        let lba = command_lba(&command);
        let started = {
            // SAFETY: as in read().
            let _root = unsafe { RestoreRoot::enter(self.root) };
            timed(DiskPart::Start, || self.inner.start(command))
        };
        self.running += u32::from(started.is_ok());
        self.looked.set(staged_disk::LookGate::default());
        // Reported with the loader's root and interrupts restored.
        self.failed("start lba", lba, started)
    }
    fn finish(&mut self, tag: u32) -> Option<Finished> {
        let finished = {
            // SAFETY: as in read().
            let _root = unsafe { RestoreRoot::enter(self.root) };
            timed(DiskPart::Finish, || self.inner.finish(tag))
        };
        self.running = self.running.saturating_sub(u32::from(finished.is_some()));
        self.looked.set(staged_disk::LookGate::default());
        // Reported with the loader's root and interrupts restored.
        finished.map(|result| self.failed("finished tag", u64::from(tag), result))
    }
    fn reaps_together(&self) -> bool {
        self.inner.reaps_together()
    }
    /// What an earlier finish collected is in memory the engine's root
    /// maps: no driver call and no root change.
    fn finished(&mut self, tag: u32) -> Option<Finished> {
        let finished = self.inner.finished(tag)?;
        self.running = self.running.saturating_sub(1);
        Some(self.failed("finished tag", u64::from(tag), finished))
    }
    fn posted(&self) -> bool {
        if self.running == 0 {
            return false;
        }
        // Read before the look: a signal after it moves the count.
        let doorbells = super::host::doorbells();
        let now = super::host::fresh_counter();
        let window = super::host::COUNTER_HZ / 1_000_000 * LOOK_WINDOW_MICROS;
        let mut gate = self.looked.get();
        if gate.skip(self.signals(), doorbells, now, window) {
            return false;
        }
        // SAFETY: as in read().
        let _root = unsafe { RestoreRoot::enter(self.root) };
        let posted = timed(DiskPart::Posted, || self.inner.posted());
        gate.looked(posted, doorbells, now);
        self.looked.set(gate);
        posted
    }
    fn signals(&self) -> bool {
        if let Some(signals) = self.signals.get() {
            return signals;
        }
        // SAFETY: as in read(); this query reaches shared controller state.
        let _root = unsafe { RestoreRoot::enter(self.root) };
        let signals = self.inner.signals();
        self.signals.set(Some(signals));
        signals
    }
}

/// The host sides of the guest's staged disks ([`stage`]).
static STAGED: [AtomicPtr<DiskWorker>; 4] = [const { AtomicPtr::new(ptr::null_mut()) }; 4];
/// A device call left a staged disk work: the processor that holds the
/// devices' lock, the only one that can have set it, steps the disk as it
/// releases the lock ([`released`]).
static STEP_DUE: AtomicBool = AtomicBool::new(false);
/// When a step was last taken, on the host's counter: the boot PE's slices
/// step only a disk nobody stepped for [`FALLBACK_MILLIS`].
static STEPPED: AtomicU64 = AtomicU64::new(0);
const FALLBACK_MILLIS: u64 = 1;
/// A completion the disk has not shown is looked for after this long.
const LOST_MILLIS: u64 = 1;
/// The time one step takes before it leaves the rest to the next, unless
/// a single call of the host disk takes longer.
const STEP_MICROS: u64 = 1000;
/// How long a fruitless look at a signalling disk excuses the next ones
/// while no core took a doorbell: a few of an NVMe read's latency.
const LOOK_WINDOW_MICROS: u64 = 20;
/// The doorbell count at the last step: a later one is a completion signal
/// no step has answered ([`staged_disk::step_due`]).
static SEEN_DOORBELLS: AtomicU64 = AtomicU64::new(0);

/// Called under the devices' lock; it only marks the step due.
fn wake_staged() {
    STEP_DUE.store(true, Ordering::Release);
}

/// Steps taken, and those given up because another core was inside the
/// host drivers, since the last report.
static STEPS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// The steps [`released`] took, and their TSC ticks in all: what a board
/// call (a guest's device access, a doorbell's mostly, or the home
/// machine's transfer) spent after it giving the host disk its commands and
/// taking its completions.
static RELEASED_STEPS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether a device call left a staged disk a step ([`released`]), without
/// taking it: a PE in a guest exit hands it to a sleeping PE's service.
pub(super) fn release_due() -> bool {
    STEP_DUE.load(Ordering::Relaxed)
}

/// After a device call released the devices' lock, outside it: the step its
/// staged disk commands left, on the processor whose guest made them, as
/// the disk's own I/O ran there before staging; whether it collected a
/// completion, which the devices then look at. A processor inside an
/// interrupt or the host drivers, or one that finds another core in them,
/// leaves the step to the next release: disk work never waits there.
pub(super) fn released() -> bool {
    if !STEP_DUE.load(Ordering::Relaxed) || !STEP_DUE.swap(false, Ordering::AcqRel) {
        return false;
    }
    if super::host::in_interrupt() || host_drivers_held_here() {
        STEP_DUE.store(true, Ordering::Release);
        return false;
    }
    let mut finished = false;
    for worker in staged_workers() {
        if worker.work() {
            let started = super::host::rdtsc();
            match step(worker) {
                Some(step) => {
                    finished |= step.finished != 0;
                    RELEASED_STEPS[0].fetch_add(1, Ordering::Relaxed);
                    RELEASED_STEPS[1].fetch_add(
                        super::host::rdtsc().wrapping_sub(started),
                        Ordering::Relaxed,
                    );
                }
                None => STEP_DUE.store(true, Ordering::Release),
            }
        }
    }
    finished
}

/// One step of `worker`, unless another core is inside the host drivers:
/// a step would wait for it there, and the work keeps for the next.
fn step(worker: &DiskWorker) -> Option<staged_disk::Step> {
    if host_drivers_taken() {
        STEPS[1].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    STEPS[0].fetch_add(1, Ordering::Relaxed);
    // Read before the step's look: a signal after it is still unanswered.
    SEEN_DOORBELLS.store(super::host::doorbells(), Ordering::Relaxed);
    let now = super::host::fresh_counter();
    STEPPED.store(now, Ordering::Relaxed);
    // A step ends once it has had its time: the rest waits for the next.
    let budget = super::host::COUNTER_HZ / 1_000_000 * STEP_MICROS;
    let mut over = || super::host::fresh_counter().wrapping_sub(now) >= budget;
    Some(timed(DiskPart::Step, || {
        worker.step_within(now, super::host::COUNTER_HZ / 1000 * LOST_MILLIS, &mut over)
    }))
}

/// Stage the host disk behind a guest storage model
/// ([`stela_boot_core::staged_disk`]): the model, under the devices' lock,
/// only queues commands and takes results, and [`serve_staged`] runs the host
/// I/O outside it. With every slot taken the disk is left as it is.
pub(super) fn stage(device: Box<dyn BlockDevice>) -> Box<dyn BlockDevice> {
    try_stage(device).unwrap_or_else(|device| device)
}

/// [`stage`], answering the disk as it was (`Err`) when every slot is
/// taken, for a caller that may only queue on a staged one.
pub(super) fn try_stage(
    device: Box<dyn BlockDevice>,
) -> core::result::Result<Box<dyn BlockDevice>, Box<dyn BlockDevice>> {
    if STAGED
        .iter()
        .all(|slot| !slot.load(Ordering::Acquire).is_null())
    {
        output::arguments(format_args!(
            "HOST DISK serial={} not staged: every slot is taken; its I/O runs under the devices' lock\n",
            device.identity().serial
        ));
        return Err(device);
    }
    let (disk, worker) = staged_disk::stage(device, wake_staged);
    let worker: *mut DiskWorker = Box::leak(Box::new(worker));
    // Staged once at the start, before any PE serves; a free slot remains.
    let stored = STAGED.iter().any(|slot| {
        slot.compare_exchange(ptr::null_mut(), worker, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    });
    assert!(stored, "a staged disk slot was taken meanwhile");
    Ok(Box::new(disk))
}

/// Give each staged disk's host side its queued commands and take its
/// completions, outside the devices' lock. A PE that waits (`whole`) steps
/// while each step makes progress, until an interrupt comes for it; one
/// with guest code to run takes one bounded step, and only for a disk no
/// processor stepped for [`FALLBACK_MILLIS`] or one whose completion signal
/// came since the last step. What a step took reaches the guest at the next
/// release of the devices' lock, whose published deadline then calls the
/// home machine at once. Nothing here waits for a disk. Whether a step
/// collected a completion, which the caller tells the home machine of
/// (`SharedDevices::call_home`): no release may follow soon on an idle board.
pub(super) fn serve_staged(whole: bool) -> bool {
    if !staged_disk::step_due(
        whole,
        super::host::fresh_counter(),
        STEPPED.load(Ordering::Relaxed),
        super::host::COUNTER_HZ / 1000 * FALLBACK_MILLIS,
        super::host::doorbells(),
        SEEN_DOORBELLS.load(Ordering::Relaxed),
    ) {
        return false;
    }
    let mut finished = false;
    for worker in staged_workers() {
        while worker.work() {
            let step = step(worker);
            finished |= step.as_ref().is_some_and(|step| step.finished != 0);
            let progressed = step.is_some_and(|step| step.progressed());
            if !whole || !progressed || super::host::interrupted() {
                break;
            }
        }
    }
    finished
}

fn staged_workers() -> impl Iterator<Item = &'static DiskWorker> {
    // SAFETY: `stage` stores only leaked workers, which live for the boot.
    STAGED
        .iter()
        .filter_map(|slot| unsafe { slot.load(Ordering::Acquire).as_ref() })
}

/// While held, the processors step the staged disks ([`released`],
/// [`serve_staged`]) and their models only queue; before it and after it,
/// a model's own polls step its disk, as nothing else would.
pub(super) struct Stepping(());

impl Stepping {
    pub(super) fn start() -> Self {
        staged_workers().for_each(|worker| worker.set_staged(true));
        Self(())
    }
}

impl Drop for Stepping {
    fn drop(&mut self) {
        staged_workers().for_each(|worker| worker.set_staged(false));
    }
}

/// The host disks [`open`] took, each once with its serial, to report
/// their I/O and to shut down before the PC restarts or turns off.
pub struct HostDisks {
    disks: Vec<(String, Shared)>,
    root: u64,
}

impl HostDisks {
    pub fn none() -> Self {
        Self {
            disks: Vec::new(),
            root: 0,
        }
    }

    fn add(&mut self, disk: &Shared) {
        if !self.disks.iter().any(|(_, taken)| taken.same(disk)) {
            let serial = disk.whole().identity().serial.clone();
            self.disks.push((serial, disk.clone()));
        }
    }

    pub fn profile_callbacks(&self, enabled: bool) {
        for (_, disk) in &self.disks {
            disk.set_callback_profile(enabled);
        }
    }

    /// Each disk's I/O since it was opened, as Linux's `/proc/diskstats`:
    /// per kind its commands, KiB, mean microseconds from submission to
    /// completion and requests merged; the device's busy milliseconds and
    /// the most commands it ran at once.
    pub fn report(&self, tsc_hz: u64) {
        let micros = |ticks: u64| u128::from(ticks) * 1_000_000 / u128::from(tsc_hz.max(1));
        for (serial, disk) in &self.disks {
            let stats = disk.stats();
            let mut line = format!("HOST DISK IO serial={serial}");
            for direction in Direction::ALL {
                let counts = stats.directions[direction as usize];
                if counts.ios != 0 {
                    line.push_str(&format!(
                        " {}={}/{}KiB/{}us/{}",
                        direction.name(),
                        counts.ios,
                        counts.bytes >> 10,
                        micros(counts.ticks) / u128::from(counts.ios),
                        counts.merges,
                    ));
                }
            }
            output::arguments(format_args!(
                "{line} busy_ms={} deepest={}\n",
                micros(stats.busy) / 1000,
                stats.deepest
            ));
            if let Some(callbacks) = stats.callbacks {
                Self::report_callback(serial, callbacks, tsc_hz, "periodic");
            }
        }
        // The longest of each part of a host disk call: the drivers' lock
        // and root change for any caller, a disk's calls inside them, and a
        // staged disk's step.
        let mut line = format!(
            "HOST DISK STAGED steps={} gave_up={} longest_us",
            STEPS[0].swap(0, Ordering::Relaxed),
            STEPS[1].swap(0, Ordering::Relaxed)
        );
        for (name, longest) in DISK_PARTS.iter().zip(&DISK_TIMES) {
            line.push_str(&format!(
                " {name}={}",
                micros(longest.swap(0, Ordering::Relaxed))
            ));
        }
        // Then each part's calls and microseconds in all, and the steps
        // board calls took after releasing the devices' lock.
        line.push_str(" calls/total_us");
        for (name, [calls, ticks]) in DISK_PARTS.iter().zip(&DISK_TOTALS) {
            line.push_str(&format!(
                " {name}={}/{}",
                calls.swap(0, Ordering::Relaxed),
                micros(ticks.swap(0, Ordering::Relaxed))
            ));
        }
        line.push_str(&format!(
            " after_calls={}/{}",
            RELEASED_STEPS[0].swap(0, Ordering::Relaxed),
            micros(RELEASED_STEPS[1].swap(0, Ordering::Relaxed))
        ));
        output::arguments(format_args!("{line}\n"));
    }

    /// One bounded snapshot before stop-time persistence; never waits for a
    /// disk owner, changes roots or accesses an active driver's registers.
    pub fn report_callbacks(&self, tsc_hz: u64) {
        for (serial, disk) in &self.disks {
            if let Some(activity) = disk.callback_activity() {
                output::arguments(format_args!(
                    "HOST DISK ACTIVITY serial={serial} counter_hz={tsc_hz} observed_before={} observed_after={} advance={:?} direct={:?} none=idle-or-raced coherent=false authority=none\n",
                    activity.observed_before_ticks,
                    activity.observed_after_ticks,
                    activity.advance,
                    activity.direct,
                ));
            }
            match disk.callback_stats() {
                Some(callbacks) => Self::report_callback(serial, callbacks, tsc_hz, "guest-stop"),
                None => output::arguments(format_args!(
                    "HOST DISK CALLBACK serial={serial} phase=guest-stop unavailable=disabled-unpublished-or-held\n"
                )),
            }
        }
    }

    fn report_callback(serial: &str, callbacks: CallbackStats, tsc_hz: u64, phase: &str) {
        let micros = |ticks: u64| u128::from(ticks) * 1_000_000 / u128::from(tsc_hz.max(1));
        let advance = callbacks.advance;
        let direct = callbacks.direct;
        output::arguments(format_args!(
            "HOST DISK CALLBACK serial={serial} phase={phase} nested-not-additive=true advance_calls={} advance_total_us={} advance_max_us={} starts={} finishes={} max_starts={} max_finishes={} direct_calls={} direct_total_us={} direct_max_us={}\n",
            advance.calls,
            micros(advance.elapsed_ticks),
            micros(advance.maximum_ticks),
            advance.starts,
            advance.finishes,
            advance.maximum_starts,
            advance.maximum_finishes,
            direct.calls,
            micros(direct.elapsed_ticks),
            micros(direct.maximum_ticks)
        ));
    }

    /// Give every disk its shutdown notice; a failure is reported and the
    /// transition still goes ahead, as it would for a disk that is gone.
    pub fn shutdown(&self, power_off: bool) {
        if self.disks.is_empty() {
            return;
        }
        // SAFETY: open() captured the loader root, as for every service call.
        let _root = unsafe { RestoreRoot::enter(self.root) };
        for (_, disk) in &self.disks {
            if let Err(error) = disk.shutdown(power_off) {
                output::arguments(format_args!("HOST DISK shutdown failed: {error}\n"));
            }
        }
    }
}

/// What [`open`] opens and brings up, each named explicitly.
#[derive(Clone, Copy)]
pub struct Selection<'a> {
    /// The guest's disks, in order: one service each.
    pub targets: &'a [&'a Target],
    /// The install medium, by its GUID.
    pub medium: Option<Guid>,
    /// The installed system `--preboot` names.
    pub preboot: Option<&'a Target>,
    /// The translation store, by its GUID; left out when it cannot open.
    pub store: Option<Guid>,
    /// The USB's boot log, by its GUID; left out when it cannot open.
    pub log: Option<Guid>,
    /// The data partition holding the current readable boot log.
    pub volume_log: Option<Guid>,
    /// Read the swap metadata while the disk probe still owns its resources.
    pub swap: Option<&'a super::swap::Probe>,
    /// Required host-owned durable FDR file volume, when explicitly staged.
    pub fdr: Option<Guid>,
    /// Whole USB disks for the guest, by serial number.
    pub usb_disks: &'a [String],
    /// The Wi-Fi network to join; without one a Wi-Fi adapter stays down.
    pub wifi: Option<&'a stela_drivers::net::wifi::network::Network>,
    pub policy: DriverPolicy<'a>,
    /// With `--external-disks`, the guest hub's free ports, which every disk
    /// left over fills in probe order.
    pub external: Option<usize>,
    /// Where the firmware found the disk Stela started from
    /// (`Handoff::boot_disk`), for the account of a failed search.
    pub boot_disk: Option<&'a str>,
    /// The target whose partition keeps the machine state in its tail,
    /// where a stop the USB's log cannot keep leaves its record
    /// (`failure_record`).
    pub state: Option<&'a Target>,
}

/// Open the explicitly selected partitions in the loader's identity root
/// `root` ([`RestoreRoot`]), and bring up every host input device and
/// network adapter. Ownership lasts with the returned services. On refusal,
/// the optional failure record uses only the chosen target's machine-state
/// tail. Probe once for all consumers;
/// do not reset a controller or reuse its DMA pages for a second probe.
///
/// # Safety
/// The handoff describes this running machine, only this service uses its
/// reserved DMA pages, `root` is the loader's root, and it and the window
/// remain live for the boot. The caller already holds the
/// RestoreRoot/host-driver guard: the owned work dispatcher in a real admitted
/// continuation, or the boot processor on its own stack when no owned stack
/// exists; acquiring that non-reentrant guard again is forbidden.
pub unsafe fn open(
    handoff: &Handoff,
    captures: PlatformCaptures,
    root: u64,
    services: Option<&dyn NativeServices>,
    selection: &Selection<'_>,
) -> Result<Opened, Refused> {
    let Selection {
        targets,
        medium,
        preboot,
        store,
        log,
        volume_log,
        swap,
        fdr,
        usb_disks,
        wifi,
        policy,
        external,
        boot_disk,
        state,
    } = *selection;
    if let Some(serial) = usb_disks
        .iter()
        .find(|serial| targets.iter().any(|target| &target.serial == *serial))
    {
        return Err(refuse!(
            "host disk {serial} is both a USB storage disk and a partition target",
            serial
        )
        .into());
    }
    // The drivers reach their DMA memory through the physmap, which every
    // engine root keeps in its window, never through the identity map only
    // the loader root has: a driver call outside the loader root, such as a
    // transfer timeout's cancel, then reads its context instead of faulting.
    // SAFETY: the caller owns the handoff's DMA pages and controllers; the
    // loader refused a DMA range its physmap does not cover (`system::memory`).
    let mut dma = unsafe {
        RegionAllocator::with_cpu_offset(handoff.dma.clone(), crate::layout::physical(0))
    };
    let firmware =
        super::firmware_requests::Requests::new(stela_drivers::probe::handoff_provider(handoff));
    // SAFETY: the loader mapped these ranges uncached and identity-mapped.
    let mut devices = unsafe {
        probe_with_policy(
            handoff, captures, &mut dma, firmware, services, wifi, &policy,
        )
    };
    firmware.report();
    // The owners native attachments keep, and those of failed ones, stay
    // for the boot: their mappings and DMA domains outlive the devices.
    Vec::leak(core::mem::take(&mut devices.owners));
    // EHCI controllers run for the boot; their roots' storage shares them.
    Vec::leak(core::mem::take(&mut devices.ehci_hosts));
    for host in &devices.hda_hosts {
        let location = &host.location;
        for codec in &host.codecs {
            let graph = &codec.graph;
            match &codec.setup {
                Ok(setup) => output::arguments(format_args!(
                    "HOST HD AUDIO {location} codec {}: {:08x} revision {:08x}, {setup}\n",
                    graph.address, graph.vendor_id, graph.revision_id
                )),
                Err(error) => output::arguments(format_args!(
                    "HOST HD AUDIO {location} codec {}: {:08x} setup failed: {error:?}\n",
                    graph.address, graph.vendor_id
                )),
            }
        }
        for failure in &host.codec_failures {
            output::arguments(format_args!(
                "HOST HD AUDIO {location} codec {} failed discovery attempt {}: {:?}\n",
                failure.address, failure.attempt, failure.error
            ));
        }
        if let Some(error) = &host.discovery_error {
            output::arguments(format_args!(
                "HOST HD AUDIO {location}: no usable codec ({error:?})\n"
            ));
        }
    }
    // HD Audio controllers stay attached for the boot; no guest audio device
    // consumes them yet.
    Vec::leak(core::mem::take(&mut devices.hda_hosts));
    // A USB device can connect, train its link and answer its first
    // commands after its controller's start looked at the ports: the
    // partitions this boot needs are waited for while the USB hosts report
    // news, as Linux finds such a device later through a port change.
    let required: Vec<Named> = targets
        .iter()
        .map(|target| Named::Disk((*target).clone()))
        .chain(medium.map(Named::Unique))
        .chain(preboot.map(|target| Named::Disk(target.clone())))
        .chain(fdr.map(Named::Host))
        .collect();
    let waited = {
        let clock = handoff.clock;
        let mut late = Vec::new();
        let usb_hosts = &mut devices.usb_hosts;
        let dma = &mut dma;
        let mut announced = false;
        let waited = storage::wait_for_named(
            &mut devices.disks,
            &required,
            storage::PATIENCE,
            &mut || clock.now_us(),
            &mut || {
                if !announced {
                    announced = true;
                    output::arguments(format_args!(
                        "HOST DISKS waiting for a partition this boot needs\n"
                    ));
                    crate::status::progress("waiting for the USB devices");
                }
                let mut arrival = storage::Arrival::default();
                for host in usb_hosts.iter_mut() {
                    let mut changes = host.poll(&mut *dma, firmware);
                    firmware.report();
                    for device in &changes.removed {
                        output::arguments(format_args!("USB removed {}\n", device.location));
                    }
                    for device in &changes.added.devices {
                        output::arguments(format_args!("USB added {}\n", device.location));
                    }
                    arrival.news |= !changes.removed.is_empty()
                        || !changes.added.devices.is_empty()
                        || !changes.added.failures.is_empty();
                    arrival.disks.extend(
                        core::mem::take(&mut changes.added.disks)
                            .into_iter()
                            .map(|disk| Box::new(disk) as Box<dyn BlockDevice>),
                    );
                    late.push(changes.added);
                }
                arrival
            },
            &mut |us| clock.delay(us),
        );
        for found in late {
            devices.absorb_usb(None, found, firmware);
        }
        waited
    };
    if waited.looks != 0 {
        output::arguments(format_args!(
            "HOST DISKS waited {} ms, {} looks, {} disks arrived, {}\n",
            waited.waited_us / 1000,
            waited.looks,
            waited.arrived,
            if waited.missing.is_none() {
                "all found"
            } else {
                "still missing"
            }
        ));
    }
    // The log only: the serials stay off the screen.
    for (named, reported) in storage::serial_fallbacks(&mut devices.disks, &required) {
        output::arguments(format_args!(
            "HOST DISK no disk reports serial {named:?}; its partition GUID is on the disk reporting {reported:?}, which is taken\n"
        ));
    }
    for host in &devices.usb4_hosts {
        let usb4::Connection::Firmware { icm, ready, early } = host.connection() else {
            continue;
        };
        output::arguments(format_args!(
            "HOST USB4 {}: firmware connection manager ready ({:?}), security level {}\n",
            host.location,
            ready.family,
            ready.security.map_or(
                alloc::string::String::from("unreported"),
                |level| alloc::format!("{level:?}")
            )
        ));
        for event in early {
            super::usb::report_usb4(&host.location, icm.family(), event);
        }
    }
    for (address, error) in &devices.failures {
        crate::storage::report_failure(*address, error);
    }
    for (address, expiry) in &devices.abandoned {
        let at = expiry
            .at
            .map_or(alloc::string::String::from("unknown"), |at| {
                alloc::format!("{}:{}", at.file(), at.line())
            });
        output::arguments(format_args!(
            "HOST DEVICE pci={address} abandoned after {} ms at step {at}\n",
            expiry.budget_us / 1000
        ));
    }
    for (location, error) in &devices.platform_failures {
        output::arguments(format_args!("PLATFORM FAILED {location}: {error}\n"));
    }
    match &devices.regulatory {
        Some(Ok(_)) => output::arguments(format_args!("HOST Wi-Fi regulatory database loaded\n")),
        Some(Err(error)) => output::arguments(format_args!(
            "HOST Wi-Fi regulatory database missing ({error:?}): the Wi-Fi adapters stay off\n"
        )),
        None => {}
    }
    for (address, warning) in &devices.warnings {
        output::arguments(format_args!("HOST WARNING pci={address}: {warning}\n"));
    }
    for (location, error) in devices.bluetooth_failures() {
        output::arguments(format_args!("BLUETOOTH FAILED {location}: {error}\n"));
    }
    output::arguments(format_args!(
        "BLUETOOTH PLATFORM {:?}\n",
        devices.bluetooth_platform.observation
    ));
    for (location, error) in &devices.usb_vendor_failures {
        output::arguments(format_args!("USB VENDOR FAILED {location}: {error}\n"));
    }
    for (address, class, why) in &devices.unsupported {
        output::arguments(format_args!(
            "HOST UNSUPPORTED pci={address} class={:02x}{:02x}{:02x}: {why}\n",
            class[0], class[1], class[2]
        ));
    }
    for adapter in &devices.gpus {
        match adapter.glk_memory {
            Some(Ok(memory)) => {
                let (base, bytes) = memory
                    .stolen
                    .map_or((0, 0), |stolen| (stolen.base, stolen.bytes));
                output::arguments(format_args!(
                    "HOST GPU GLK pci={} device={:04x} ggtt_bytes={} stolen_base={base:#x} stolen_bytes={bytes}\n",
                    adapter.address, adapter.device, memory.ggtt_bytes
                ));
            }
            Some(Err(error)) => output::arguments(format_args!(
                "HOST GPU GLK pci={} device={:04x} memory unavailable: {error:?}\n",
                adapter.address, adapter.device
            )),
            None => {}
        }
    }
    for (location, error) in &devices.device_failures {
        output::arguments(format_args!("HOST DEVICE FAILED {location}: {error}\n"));
    }
    for (location, error) in &devices.usb_failures {
        output::arguments(format_args!(
            "USB FAILED {location}: {}\n",
            stela_drivers::usb::xhci::explain(error)
        ));
    }
    for (location, why) in &devices.usb_unused {
        output::arguments(format_args!("USB UNUSED {location}: {why}\n"));
    }
    let inputs = Inputs {
        devices: core::mem::take(&mut devices.inputs),
        root,
    };
    let links = super::network::Links {
        devices: core::mem::take(&mut devices.nics),
        root,
    };
    let usb = super::usb::HostUsb::new(
        core::mem::take(&mut devices.usb_hosts),
        core::mem::take(&mut devices.usb4_hosts),
        core::mem::take(&mut devices.usb_storage),
        dma,
        firmware,
        root,
        handoff.clock.ticks_per_microsecond * 1_000_000,
    );
    // One disk may hold several of them: the stick holds the Mac's internal
    // SSD, the install medium, the translation store and the boot log, and
    // the internal SSD the installed system the Preboot volume names. The
    // store and the log are optional: one that cannot be opened is reported
    // and left out.
    let mut named = required;
    let required = named.len();
    let mut optional: Vec<(Named, Optional)> = store
        .map(|guid| (Named::Unique(guid), Optional::Store))
        .into_iter()
        .chain(log.map(|guid| (Named::Host(guid), Optional::Log)))
        .chain(volume_log.map(|guid| (Named::Unique(guid), Optional::VolumeLog)))
        .collect();
    if let Some(probe) = swap
        && !probe.inspect_partition(&mut devices.disks)
    {
        probe.inspect_volume(&mut devices.disks, handoff.clock.ticks_per_microsecond);
    }
    // The controllers that could hold a missing disk and were not driven.
    let not_driven = devices
        .unsupported
        .iter()
        .filter(|(_, class, _)| class[0] == 0x01)
        .map(|(address, _, why)| format!("{address} {why}"))
        .chain(
            devices
                .failures
                .iter()
                .filter(|(_, error)| crate::storage::may_hold_disks(error))
                .map(|(address, error)| format!("{address} failed: {error}")),
        )
        .collect::<Vec<_>>()
        .join("; ");
    let describe = |named: &Named, error: storage::StorageError| match named {
        Named::Disk(target) => {
            let hint = match &error {
                storage::StorageError::NoDisk { .. } if !not_driven.is_empty() => {
                    format!("; not driven: {not_driven}")
                }
                _ => String::new(),
            };
            refuse!(
                "{serial}:{partition}: {error}{hint}",
                serial = target.serial.as_str(),
                partition = %target.partition,
                %error,
                %hint
            )
        }
        Named::Unique(guid) | Named::Host(guid) => {
            refuse!("partition {guid}: {error}", %guid, %error)
        }
    };
    let service = |partition: PartitionDevice<Window>, role: &str| -> Box<dyn BlockDevice> {
        output::arguments(format_args!(
            "HOST PARTITION serial={} partition={} blocks={} block_size={}{role}\n",
            partition.disk().serial,
            partition.partition().unique_guid,
            partition.identity().blocks,
            partition.identity().block_size
        ));
        Box::new(Service {
            inner: Box::new(partition),
            root,
            running: 0,
            signals: Cell::new(None),
            looked: Cell::new(staged_disk::LookGate::default()),
            failures: 0,
        })
    };
    let mut opened = loop {
        named.truncate(required);
        named.extend(optional.iter().map(|(named, _)| named.clone()));
        match storage::open_shared(&mut devices.disks, &named) {
            Ok(opened) => break opened,
            Err((at, error)) if at >= required => {
                let (named, which) = optional.remove(at - required);
                output::arguments(format_args!(
                    "HOST PARTITION {} left out: {}\n",
                    which.role(),
                    describe(&named, error)
                ));
            }
            Err((at, error)) => {
                let role = role_of(at, targets.len(), medium.is_some(), preboot.is_some());
                let refusal = describe(&named[at], error);
                // Report all disks before taking the USB's disk for its log.
                let mut lines = account(
                    &mut devices,
                    role,
                    waited.waited_us,
                    boot_disk,
                    targets.first().copied(),
                );
                // The USB's log, when its disk is here though a required
                // partition is not, keeps this step's lines and the account.
                let log = log
                    .and_then(|guid| {
                        storage::open_shared(&mut devices.disks, &[Named::Host(guid)]).ok()
                    })
                    .and_then(|mut opened| opened.pop())
                    .map(|partition| service(partition, " (boot log)"));
                let refusal = refuse!("{role}: {refusal}", role, %refusal);
                // Without it, the chosen partition's machine state keeps the
                // step's record for the next boot to copy into the log.
                // The targets come first, so every one was found when a
                // later partition is refused.
                let kept = match (&log, state) {
                    (Some(_), _) => Keeping::Log,
                    (None, Some(target)) if at >= targets.len() => Keeping::Record(target),
                    (None, _) => Keeping::Screen,
                };
                lines.push(String::from(match kept {
                    Keeping::Log => "The boot log on the USB stick keeps these lines.",
                    Keeping::Record(_) => {
                        "The boot log on the USB stick is out of reach; the chosen disk keeps these lines, \
                         and the next start copies them into it."
                    }
                    Keeping::Screen => {
                        "The boot log on the USB stick is out of reach; this screen is this part's only record."
                    }
                }));
                for line in &lines {
                    output::arguments(format_args!("HOST DISKS {line}\n"));
                    crate::status::info(line);
                }
                if let Keeping::Record(target) = kept {
                    let mut text = String::from(
                        "STEP Starting the PC's disks and USB devices\n\
                         STELA FATAL host storage could not be opened: ",
                    );
                    text.push_str(&format!("{refusal}\n"));
                    for line in &lines {
                        text.push_str(&format!("HOST DISKS {line}\n"));
                    }
                    match leave_record(&mut devices.disks, target, &text) {
                        Ok(()) => output::arguments(format_args!(
                            "HOST DISKS record left on the chosen partition for the next boot's log\n"
                        )),
                        Err(why) => output::arguments(format_args!(
                            "HOST DISKS no record left on the chosen partition: {why}\n"
                        )),
                    }
                }
                return Err(Refused { refusal, log });
            }
        }
    };
    let mut host_disks = HostDisks {
        disks: Vec::new(),
        root,
    };
    for partition in &opened {
        host_disks.add(partition.device().shared());
    }
    let (mut store_partition, mut log_partition, mut volume_log_partition) = (None, None, None);
    for (_, which) in optional.iter().rev() {
        let role = format!(" ({})", which.role());
        let partition = Some(service(opened.pop().expect("an optional partition"), &role));
        match which {
            Optional::Store => store_partition = partition,
            Optional::Log => log_partition = partition,
            Optional::VolumeLog => volume_log_partition = partition,
        }
    }
    let fdr = fdr.map(|_| service(opened.pop().expect("FDR storage"), " (FDR storage)"));
    let preboot = preboot.map(|_| {
        service(
            opened.pop().expect("the installed system"),
            " (installed system)",
        )
    });
    let medium = medium.map(|_| service(opened.pop().expect("the medium"), " (install medium)"));
    let services = opened
        .into_iter()
        .map(|partition| service(partition, ""))
        .collect();
    let mut taken = usb_disks
        .iter()
        .map(|serial| {
            storage::take(&mut devices.disks, serial)
                .map_err(|error| refuse!("USB storage {serial}: {error}", serial, %error))
        })
        .collect::<Result<Vec<_>>>()?;
    // With `--external-disks`, every disk left, in probe order, up to the
    // hub's free ports; the rest are reported.
    if let Some(ports) = external {
        let rest = core::mem::take(&mut devices.disks);
        for (index, disk) in rest.into_iter().enumerate() {
            if index < ports {
                taken.push(disk);
            } else {
                output::arguments(format_args!(
                    "HOST DISK serial={} left out: the guest's hub has no free port\n",
                    disk.identity().serial
                ));
            }
        }
    }
    let usb_disks = taken
        .into_iter()
        .map(|disk| {
            let disk = Shared::new(disk);
            host_disks.add(&disk);
            let disk = disk.whole();
            output::arguments(format_args!(
                "HOST USB DISK serial={} model={:?} blocks={} block_size={} read_only={}\n",
                disk.identity().serial,
                disk.identity().model,
                disk.identity().blocks,
                disk.identity().block_size,
                disk.identity().read_only
            ));
            Box::new(Service {
                inner: Box::new(disk),
                root,
                running: 0,
                signals: Cell::new(None),
                looked: Cell::new(staged_disk::LookGate::default()),
                failures: 0,
            }) as Box<dyn BlockDevice>
        })
        .collect();
    Ok(Opened {
        services,
        medium,
        preboot,
        store: store_partition,
        log: log_partition,
        volume_log: volume_log_partition,
        fdr,
        inputs,
        links,
        bluetooth: super::bluetooth::HostBluetooth::first(
            core::mem::take(&mut devices.bluetooth),
            core::mem::take(&mut devices.pcie_bluetooth),
            root,
        ),
        gpus: core::mem::take(&mut devices.gpus),
        usb,
        usb_disks,
        disks: host_disks,
        power: core::mem::take(&mut devices.power),
    })
}

/// What the required partition at `at` of `open`'s list is for.
fn role_of(at: usize, targets: usize, medium: bool, preboot: bool) -> &'static str {
    let medium_at = targets;
    let preboot_at = medium_at + usize::from(medium);
    let fdr_at = preboot_at + usize::from(preboot);
    match at {
        at if at < targets => "the partition chosen for the Mac's SSD",
        at if medium && at == medium_at => "the USB stick's data partition (install files)",
        at if preboot && at == preboot_at => "the installed macOS system",
        at if at == fdr_at => "the stick's FDR storage partition",
        _ => "a partition this boot needs",
    }
}

/// Describe a failed disk search and the controllers, disks and GPT
/// partitions visible before taking any disk for the failure log. The
/// inventory includes only the suffixes of disk serial numbers.
fn account(
    devices: &mut stela_drivers::probe::Devices,
    role: &str,
    waited_us: u64,
    boot_disk: Option<&str>,
    chosen: Option<&Target>,
) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(match waited_us {
        0 => format!("Not usable: {role}."),
        us => format!("Not usable after {} s: {role}.", us.div_ceil(1_000_000)),
    });
    match boot_disk {
        Some(at) => lines.push(format!("The firmware started Stela from {at}.")),
        None => lines.push(String::from(
            "Where the firmware found the stick is unknown.",
        )),
    }
    let mut controllers: Vec<(stela_drivers::bus::pci::Address, Vec<u8>, usize)> = Vec::new();
    for (address, probe) in &devices.usb_port_probes {
        let at = match controllers.iter().position(|(a, ..)| a == address) {
            Some(at) => at,
            None => {
                controllers.push((*address, Vec::new(), 0));
                controllers.len() - 1
            }
        };
        if probe.connected {
            controllers[at].1.push(probe.port);
        } else {
            controllers[at].2 += 1;
        }
    }
    for (address, connected, empty) in &controllers {
        let ports: Vec<String> = connected.iter().map(|port| format!("{port}")).collect();
        lines.push(format!(
            "USB {address}: devices on ports [{}], {empty} empty.",
            ports.join(" ")
        ));
    }
    for (address, error) in &devices.failures {
        if crate::storage::may_hold_disks(error) {
            lines.push(format!("{address} failed: {error}."));
        }
    }
    for (address, class, why) in &devices.unsupported {
        if class[0] == 0x01 || class[..2] == [0x0c, 0x03] {
            lines.push(format!("{address} not driven: {why}."));
        }
    }
    if let Some((location, error)) = devices.usb_failures.first() {
        lines.push(format!(
            "USB failures: {}, first {location}: {}.",
            devices.usb_failures.len(),
            stela_drivers::usb::xhci::explain(error)
        ));
    }
    let firmware_block = boot_disk
        .and_then(|at| at.rsplit_once(" block="))
        .and_then(|(_, block)| block.parse::<u32>().ok());
    let usb: Vec<u32> = devices
        .disks
        .iter()
        .map(|disk| disk.identity())
        .filter(|identity| matches!(identity.transport, stela_drivers::block::Transport::Usb))
        .map(|identity| identity.block_size)
        .collect();
    if let Some(block) = firmware_block
        && !usb.is_empty()
        && !usb.contains(&block)
    {
        lines.push(format!(
            "The firmware read the stick with {block}-byte blocks; no USB disk found here uses them."
        ));
    }
    let disks = storage::disk_account(&mut devices.disks, chosen);
    if disks.is_empty() {
        lines.push(String::from("Disks found: none."));
    }
    lines.extend(disks.into_iter().map(|disk| format!("Disk {disk}.")));
    lines
}

/// Where a failed disk search's account is kept beyond the screen.
#[derive(Clone, Copy)]
enum Keeping<'a> {
    /// The USB's own log, whose disk was found.
    Log,
    /// The machine state of this chosen partition (`failure_record`).
    Record(&'a Target),
    Screen,
}

/// Leave `text` in the machine state of `target`'s partition
/// (`failure_record`), for the next boot to copy into the USB's log.
fn leave_record(
    disks: &mut Vec<Box<dyn BlockDevice>>,
    target: &Target,
    text: &str,
) -> core::result::Result<(), String> {
    let mut partition = storage::open_shared(disks, &[Named::Disk(target.clone())])
        .map_err(|(_, error)| format!("{error}"))?
        .pop()
        .expect("one partition was named");
    let block = u64::from(partition.identity().block_size);
    let split =
        machine_state::split(partition.identity().bytes()).map_err(|error| format!("{error}"))?;
    let at = split.state_offset() + failure_record::OFFSET;
    if block == 0 || !at.is_multiple_of(block) {
        return Err(format!("{block}-byte blocks do not align the record"));
    }
    partition
        .write(at / block, &failure_record::encode(text))
        .and_then(|()| partition.flush())
        .map_err(|error| format!("{error}"))
}

/// Why [`open`] stopped, and the USB's boot log when its disk was found, so
/// the caller keeps the step's lines before it stops.
pub struct Refused {
    pub refusal: crate::Refusal,
    pub log: Option<Box<dyn BlockDevice>>,
}

impl From<crate::Refusal> for Refused {
    fn from(refusal: crate::Refusal) -> Self {
        Self { refusal, log: None }
    }
}

/// A partition [`open`] leaves out when it cannot be opened.
#[derive(Clone, Copy)]
enum Optional {
    Store,
    Log,
    VolumeLog,
}

impl Optional {
    fn role(self) -> &'static str {
        match self {
            Optional::Store => "translation store",
            Optional::Log => "boot log",
            Optional::VolumeLog => "data-volume boot log",
        }
    }
}

/// What [`open`] brought up: a service per target, in order, and the
/// host's input devices and network adapters.
pub struct Opened {
    pub power: stela_drivers::probe::Power,
    pub services: alloc::vec::Vec<Box<dyn BlockDevice>>,
    /// The install medium's partition, found by its GUID.
    pub medium: Option<Box<dyn BlockDevice>>,
    /// The partition `--preboot` names, which the boot bundle is built from.
    pub preboot: Option<Box<dyn BlockDevice>>,
    /// The translation store's partition, found by its GUID.
    pub store: Option<Box<dyn BlockDevice>>,
    /// The partition holding the USB's boot log, found by its GUID.
    pub log: Option<Box<dyn BlockDevice>>,
    pub volume_log: Option<Box<dyn BlockDevice>>,
    pub fdr: Option<Box<dyn BlockDevice>>,
    pub inputs: Inputs,
    pub links: super::network::Links,
    /// The host's Bluetooth controller, for the guest's.
    pub bluetooth: Option<super::bluetooth::HostBluetooth>,
    /// The display controllers probe left for their own drivers.
    pub gpus: alloc::vec::Vec<stela_drivers::gpu::Adapter>,
    /// The xHCI controllers' topology owners, polled for hotplug.
    pub usb: super::usb::HostUsb,
    /// Whole disks for the guest's USB mass storage, in option order.
    pub usb_disks: alloc::vec::Vec<Box<dyn BlockDevice>>,
    /// Every disk taken above, for its shutdown notice.
    pub disks: HostDisks,
}

/// The host's input devices and the loader root their registers and DMA
/// pages are reached through.
pub struct Inputs {
    pub devices: alloc::vec::Vec<InputDevice>,
    pub root: u64,
}
