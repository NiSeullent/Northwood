//! The Metal service over NVK, which the guest reaches through StelaGPU's gate,
//!
//! It opens on the boot CPU once nouveau has bound the GPUs, before the guest's
//! DeviceTree is final, so the tree has a gate only with a device behind it,
//! and one for each GPU that opened.
//!
//! Each service runs on a stack of its own, deep enough for a compile. Its
//! lock means one is enough; opening comes before any caller. A second
//! stack takes the compilations the calls defer, one at a time under the
//! compiler's own lock ([`run_job`]): a call that needs one is answered
//! not ready, and its guest thread sleeps and calls again.
use super::{
    BOOT_CPU, Bound, Installed, PAGE, client, installed, look_again,
    stack::{self, Stack},
};
use crate::{engine::host, output};
use alloc::{boxed::Box, format, string::String, sync::Arc, vec::Vec};
use core::{
    ffi::{c_char, c_void},
    num::NonZeroU64,
    ptr::{self, NonNull},
    sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, Ordering},
};
use stela_contracts::gate::{BundleReport, GateService, MemoryRange, window};
use stela_drivers::gpu::metal::{
    compiler::{self, AirCompiler},
    device::{lifecycle::QuiesceStatus, pass_times},
    error::Error,
    native::{ExportedMemory, Loader, MemoryExporter, NativeBackend, Refusals, pipeline_store, vk},
    nvidia::{
        Attachment,
        open::{Gpu, OpenError, open, reopen},
    },
    service::{Served, ServiceHost, SharedService, identity::DeviceEpoch, wire::Status},
    timing::Totals,
    vulkan::physical::PciBusInfo,
};
use stela_engine::memory::system_ram::ExternalDmaLoan;
use stela_linux::{
    client::Run,
    function::{dma_quarantine_status, quarantined_functions},
};
use stela_loader::{apple_device_tree::DeviceTree, stela_gpu};

type Shared = SharedService<NativeBackend, AirCompiler>;

/// A process's main stack, as big as the deepest compile LLVM bounds.
const STACK_BYTES: usize = 16 << 20;

/// How long a GPU's accepted work may go without progress as the PC stops:
/// with the guest stopped, an event it waits on will never signal.
const STOP_STALL_SECONDS: u64 = 5;
/// How long the stop waits for a GSP to unload.
const STOP_UNLOAD_SECONDS: u64 = 10;

/// The services for the kernel's interrupts, once all are open.
static SERVICES: AtomicPtr<Vec<Arc<Shared>>> = AtomicPtr::new(ptr::null_mut());

unsafe extern "C" {
    fn stela_nvk_nir_options(physical: vk::PhysicalDevice) -> *const c_void;
    /// `compat/mesa/submit_thread.c`.
    fn stela_nvk_submit_thread(queue: vk::Queue) -> vk::Result;
    fn stela_nvk_submit_failure(
        queue: vk::Queue,
        semaphore: vk::Semaphore,
        value: u64,
    ) -> vk::Result;
    fn stela_nvk_forget_failures(queue: vk::Queue) -> u32;
    fn stela_nvk_count_failures(queue: vk::Queue, outstanding: *mut u32);
}

/// The compile stacks ([`open_gpu`]), whose jobs [`run`] runs with the
/// shootdown vector open; a GPU past these runs its jobs masked.
static COMPILE_STACKS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

fn run(stack: u64, body: &mut dyn FnMut()) {
    let job = COMPILE_STACKS
        .iter()
        .any(|compile| compile.load(Ordering::Relaxed) == stack);
    // SAFETY: a service's stack lives for the boot, and only the opener, a
    // turn's holder or the compile lock's holder is here.
    let mut on_stack = || unsafe { stack::on_stack(stack, &mut *body) };
    if job {
        // SAFETY: a job runs where long host work is allowed, from a run
        // loop or a guest call, holding no lock the doorbell's or the
        // shootdown's handler takes; it leaves IF and CR8 as they were.
        unsafe { crate::native_mmio::shootdown::job(on_stack) }
    } else {
        on_stack()
    }
}

pub(super) fn relax() {
    client::relax();
}

/// A table of the texts a log line has shown, and a count of the distinct
/// ones it had no room left to hold.
struct Seen<const N: usize> {
    hashes: [AtomicU64; N],
    unheld: AtomicU64,
}

impl<const N: usize> Seen<N> {
    const fn new() -> Self {
        Self {
            hashes: [const { AtomicU64::new(0) }; N],
            unheld: AtomicU64::new(0),
        }
    }

    /// Whether to show `text`: the first time, by its FNV-1a hash, since a
    /// text is built per call and only its bytes identify it. Once the table
    /// fills, a text it cannot hold is shown only at each power-of-two count
    /// of such texts, which the [`Unheld`] it answers names, so a flood of distinct
    /// texts stays bounded.
    fn first_time(&self, text: &str) -> Option<Unheld> {
        let hash = text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
        });
        for slot in &self.hashes {
            match slot.compare_exchange(0, hash, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return Some(Unheld(0)),
                Err(known) if known == hash => return None,
                Err(_) => {}
            }
        }
        let unheld = self.unheld.fetch_add(1, Ordering::Relaxed) + 1;
        unheld.is_power_of_two().then_some(Unheld(unheld))
    }
}

/// How many distinct texts a full [`Seen`] has counted, shown after a line.
struct Unheld(u64);

impl core::fmt::Display for Unheld {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            0 => Ok(()),
            count => write!(formatter, " ({count} distinct past the log's table)"),
        }
    }
}

/// How long a deferred compilation waits for a processor other than the boot
/// one before the boot one takes it: with every other busy or none (one PE),
/// the boot PE's Linux work waits for the job instead.
const BOOT_JOB_MILLIS: u64 = 100;

/// How long a deferred compilation waits for a processor whose guest waits
/// before the processor of the call that needs it runs it, busy or not
/// (`SharedService::set_force_after`): with every guest CPU busy, as in a
/// benchmark, no call waits forever.
const FORCE_JOB_MILLIS: u64 = 30;

/// How long a frontend call waits for the turn before it is answered
/// `Pending` and made again (`SharedService::set_wait_bound`): its PE takes
/// no guest interrupt while it waits, and XNU panics when a thread cannot be
/// switched off its CPU within half a second.
const WAIT_BOUND_MILLIS: u64 = 2;

/// About how long each part of the boot CPU's look at a service takes at
/// most (`SharedService::set_look_budget`): the destruction of what completed
/// work released, and the upkeep of the frees and semaphores the turns left,
/// done where a host kernel call needs no posting. The boot PE's guest waits
/// meanwhile; one look destroying a burst of releases stalled it 99-300 ms.
const LOOK_MICROS: u64 = 1000;

/// A single gate exit this long or longer is reported once a boot
/// (`GPU LONG EXIT`): 100 ms.
const LONG_EXIT_MILLIS: u64 = 100;

/// The longest single exit since the last report, in ticks, and its
/// operation; the frontend calls answered `Pending` at the wait bound.
static LONGEST_EXIT: AtomicU64 = AtomicU64::new(0);
static LONGEST_EXIT_OPERATION: AtomicU32 = AtomicU32::new(0);
static GAVE_UP: AtomicU64 = AtomicU64::new(0);

/// A turn held this long or longer is reported once for each operation and
/// phase: no turn should come near it.
const LONG_TURN_MILLIS: u64 = 50;

/// Runs one deferred compilation, on a processor whose guest waits for an
/// interrupt: a job runs whole, as the compilers have no point to stop at,
/// so only a processor with no guest work to do takes it, and its guest's
/// next interrupt waits for the job (a few to hundreds of milliseconds, a
/// very large kernel seconds). Other processors' guests run meanwhile, and
/// the guest clock does not wait for a processor whose guest waits. Off the
/// boot processor the job runs with interrupts on; on it, and in a call
/// that forces its job, with only the doorbell and the TLB shootdown open
/// ([`run`]), so a shootdown another processor sends is acknowledged at once
/// rather than after the job. The boot processor, whose Linux work the job
/// would hold up, takes one only after [`BOOT_JOB_MILLIS`]. Whether one ran.
pub(in crate::engine) fn run_job(boot: bool) -> bool {
    let Some(gpu) = installed() else {
        return false;
    };
    let older = if boot {
        gpu.tsc_hz / 1000 * BOOT_JOB_MILLIS
    } else {
        0
    };
    shared().iter().any(|service| {
        if !service.jobs_waiting() {
            return false;
        }
        let run = || {
            if boot {
                service.run_job(older)
            } else {
                // SAFETY: an AP's host work between slices, which holds nothing
                // its interrupts' handlers take; they save the vector state.
                unsafe { with_interrupts(|| service.run_job(older)) }
            }
        };
        measured_job(run)
    })
}

/// The most the heap's carved bytes rose during one job since the last
/// `GPU JOBS` line, and the carved bytes that job started from.
static JOB_RISE: AtomicU64 = AtomicU64::new(0);
static JOB_RISE_FROM: AtomicU64 = AtomicU64::new(0);
/// Set while a processor measures the job it runs.
static MEASURING: AtomicBool = AtomicBool::new(false);

/// Runs `run`, and when it ran a job, keeps how far the heap's carved
/// bytes rose above where they were: a compile's own peak, as jobs run one
/// at a time, with what other processors carved meanwhile. One processor
/// measures at a time, as the heap keeps one mark.
fn measured_job(run: impl FnOnce() -> bool) -> bool {
    if MEASURING
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return run();
    }
    let from = crate::ALLOCATOR.mark();
    let ran = run();
    if ran {
        let rise = crate::ALLOCATOR.marked_peak().saturating_sub(from) as u64;
        if rise > JOB_RISE.load(Ordering::Relaxed) {
            JOB_RISE.store(rise, Ordering::Relaxed);
            JOB_RISE_FROM.store(from as u64, Ordering::Relaxed);
        }
    }
    MEASURING.store(false, Ordering::Release);
    ran
}

/// Runs `body` with this processor's interrupts on.
///
/// # Safety
/// Nothing `body`'s caller holds is taken by an interrupt handler.
unsafe fn with_interrupts<R>(body: impl FnOnce() -> R) -> R {
    // SAFETY: the caller's.
    unsafe { core::arch::asm!("sti", options(nomem, nostack)) };
    let result = body();
    // SAFETY: back to how the caller ran.
    unsafe { core::arch::asm!("cli", options(nomem, nostack)) };
    result
}

/// Logs why the service refused a guest's call, once for each reason.
fn report(reason: &str) {
    static SEEN: Seen<512> = Seen::new();
    if let Some(unheld) = SEEN.first_time(reason) {
        output::arguments(format_args!("GPU REFUSED {reason}{unheld}\n"));
    }
}

/// Logs each distinct error or warning NVK reports, such as why a call failed
/// with only a generic result.
fn driver_message(text: &str) {
    static SEEN: Seen<128> = Seen::new();
    if let Some(unheld) = SEEN.first_time(text) {
        output::arguments(format_args!("GPU DRIVER {text}{unheld}\n"));
        report_peaks();
    }
}

/// Logs what a guest call made or met, such as a render pipeline, once for
/// each distinct note, with the heap's live bytes and high-water mark after
/// it: a compile that raised the mark shows it against the note before.
fn note(text: &str) {
    static SEEN: Seen<512> = Seen::new();
    if let Some(unheld) = SEEN.first_time(text) {
        let heap = crate::ALLOCATOR.usage();
        output::arguments(format_args!(
            "GPU {text} heap={}M peak={}M{unheld}\n",
            heap.live >> 20,
            heap.peak >> 20
        ));
        report_peaks();
    }
}

/// Stacks of the allocations that raised the heap's high-water mark, each
/// the carved bytes then and the image offsets of code on the stack.
const PEAK_SLOTS: usize = 16;
const PEAK_WORDS: usize = 24;
static PEAKS: [[AtomicU64; PEAK_WORDS + 1]; PEAK_SLOTS] =
    [const { [const { AtomicU64::new(0) }; PEAK_WORDS + 1] }; PEAK_SLOTS];
static PEAKS_TAKEN: AtomicU32 = AtomicU32::new(0);
static PEAKS_SHOWN: AtomicU32 = AtomicU32::new(0);

/// The heap's peak hook: keeps the allocating stack's code offsets. Runs
/// under the heap's lock, so it only reads the stack and stores words.
pub(super) fn peak_passed(carved: usize) {
    unsafe extern "C" {
        static __ImageBase: u8;
    }
    let slot = PEAKS_TAKEN.load(Ordering::Relaxed) as usize;
    if slot >= PEAK_SLOTS {
        return;
    }
    let base = &raw const __ImageBase as u64;
    // SAFETY: the mapped PE headers of this image: `e_lfanew`, then the
    // optional header's SizeOfCode and BaseOfCode.
    let (code, size) = unsafe {
        let header = u64::from((base as *const u32).byte_add(0x3c).read_unaligned());
        let pe = (base + header) as *const u32;
        (
            u64::from(pe.byte_add(0x2c).read_unaligned()),
            u64::from(pe.byte_add(0x1c).read_unaligned()),
        )
    };
    let stack: *const u64;
    // SAFETY: reads the stack pointer.
    unsafe {
        core::arch::asm!("mov {}, rsp", out(reg) stack, options(nomem, nostack, preserves_flags));
    }
    let words = &PEAKS[slot];
    words[0].store(carved as u64, Ordering::Relaxed);
    let mut kept = 0;
    for word in 0..512 {
        // SAFETY: as for a panic, an allocation runs 512 words from its stack's end.
        let offset = unsafe { stack.add(word).read() }.wrapping_sub(base);
        if offset.wrapping_sub(code) < size {
            kept += 1;
            words[kept].store(offset, Ordering::Relaxed);
            if kept == PEAK_WORDS {
                break;
            }
        }
    }
    PEAKS_TAKEN.store(slot as u32 + 1, Ordering::Release);
}

/// Logs the peak stacks taken since the last call, and once that the
/// slots filled, after which higher peaks show only in `BEAT peak=`.
fn report_peaks() {
    use core::fmt::Write;
    let taken = PEAKS_TAKEN.load(Ordering::Acquire);
    let shown = PEAKS_SHOWN.swap(taken, Ordering::AcqRel);
    if shown < taken && taken as usize == PEAK_SLOTS {
        output::arguments(format_args!(
            "HEAP PEAK the {PEAK_SLOTS} stack slots are full; later peaks are not traced\n"
        ));
    }
    for words in &PEAKS[shown as usize..taken as usize] {
        let mut line = format!(
            "HEAP PEAK carved={}M stack",
            words[0].load(Ordering::Relaxed) >> 20
        );
        for word in &words[1..] {
            let offset = word.load(Ordering::Relaxed);
            if offset != 0 {
                let _ = write!(line, " {offset:#x}");
            }
        }
        output::arguments(format_args!("{line}\n"));
    }
}

/// What the kext found of the Metal plugin, the bundle whose code this boot's
/// trust cache lists. A multi-user boot's Metal clients cannot start without
/// it. A missing plugin, whose directories a kext cannot create, is repaired
/// once by a single-user boot the engine drives (`bundle_repair`): this boot
/// restarts into it. Otherwise the boot stops with the remedy rather than let
/// WindowServer fail. The stop and the restart are orderly, not fatal: the
/// GPUs' firmware is running by now, and only the boot CPU's service,
/// outside Linux's callbacks, unloads it before the reset that would
/// otherwise take down the card's link.
fn bundle(report: BundleReport) {
    use super::super::bundle_repair;
    use stela_boot_core::bundle_repair::{Decision, Report};
    let decided = bundle_repair::decide(match report {
        BundleReport::Current => Report::Current,
        BundleReport::Updated => Report::Updated,
        BundleReport::Missing => Report::Missing,
        BundleReport::Failed => Report::Failed,
    });
    match (decided, report) {
        (Decision::Continue(_), BundleReport::Updated) => {
            output::arguments(format_args!(
                "GPU StelaGPU.bundle updated to this stela.efi's\n"
            ));
        }
        (Decision::Continue(_), _) => {
            output::arguments(format_args!("GPU StelaGPU.bundle is this stela.efi's\n"));
        }
        (Decision::Log, _) => {
            output::arguments(format_args!(
                "GPU StelaGPU.bundle is not installed ({report:?}); this single-user boot may install it\n"
            ));
        }
        (Decision::Repair, _) => {
            output::arguments(format_args!(
                "GPU StelaGPU.bundle is not installed and a kext cannot create its \
                 directories; restarting once into a single-user boot that creates them\n"
            ));
            super::super::service::request_restart();
        }
        (Decision::StopMissing { attempted: true }, _) => stop(
            "StelaGPU.bundle is still not installed in /Library/GPUBundles after the automatic \
             single-user repair (its BUNDLE REPAIR lines say how it went): boot macOS once with \
             --single-user and install it there by hand",
        ),
        (Decision::StopMissing { attempted: false }, _) => stop(
            "StelaGPU.bundle is not installed in /Library/GPUBundles, a kext cannot create \
             its directories, and without the machine state's NVRAM no automatic repair can be \
             recorded: boot macOS once with --single-user and install it there by hand",
        ),
        (Decision::StopFailed, _) => stop(
            "the StelaGPU kext could not install StelaGPU.bundle; its StelaGPU lines above say why",
        ),
    }

    fn stop(reason: &str) {
        output::stopping(reason);
        super::super::service::request_shutdown(true);
    }
}

/// The guest calls one operation's gate served since the last report.
struct Calls {
    /// The operation plus one; zero while the slot is free.
    key: AtomicU32,
    count: AtomicU64,
    /// TSC ticks holding the service, the longest one call held it, and
    /// waiting for another call's turn.
    held: AtomicU64,
    longest: AtomicU64,
    waited: AtomicU64,
    /// Calls answered `Unmapped`, which the frontend repeats.
    unmapped: AtomicU64,
    /// Requests the calls posted to the host's Linux environment, and the
    /// ticks they waited for the boot CPU to serve them.
    posted: AtomicU64,
    posted_waited: AtomicU64,
    /// The longest one wait for the turn, the ticks compiling outside the
    /// lock, and the compilations that were.
    longest_wait: AtomicU64,
    compiled: AtomicU64,
    deferrals: AtomicU64,
}

/// More slots than the protocol has operations.
static CALLS: [Calls; 96] = [const {
    Calls {
        key: AtomicU32::new(0),
        count: AtomicU64::new(0),
        held: AtomicU64::new(0),
        longest: AtomicU64::new(0),
        waited: AtomicU64::new(0),
        unmapped: AtomicU64::new(0),
        posted: AtomicU64::new(0),
        posted_waited: AtomicU64::new(0),
        longest_wait: AtomicU64::new(0),
        compiled: AtomicU64::new(0),
        deferrals: AtomicU64::new(0),
    }
}; 96];

/// The phases of the calls timed by phase since the last report.
static PHASES: Totals = Totals::new();

/// A timed call of a second over this or longer shows a line of its own:
/// 100 ms.
const SLOW_CALL_DIVISOR: u64 = 10;

/// Calls whose record the service could not read, answered by the gate's
/// fault alone (`GPU CALLS unreached=`): the frontend makes each again.
static UNREACHED: AtomicU64 = AtomicU64::new(0);

/// Counts a call the gate served: the PE that rang runs no guest code until
/// it returns, so a call that holds the service long is guest time lost.
fn served(served: Served) {
    report_long_turn(&served);
    note_exit(&served);
    // A record out of reach names no operation and leaves the call closed.
    if served.operation == 0 && served.status == Status::Closed {
        UNREACHED.fetch_add(1, Ordering::Relaxed);
    }
    if let Some(call) = served.phases {
        let tsc_hz = installed().map_or(0, |gpu| gpu.tsc_hz);
        if PHASES.add(&call, tsc_hz / SLOW_CALL_DIVISOR) {
            output::arguments(format_args!("{}\n", call.slow_line(tsc_hz)));
        }
    }
    // The display compares each swap with the GPU's progress.
    if let Some((submitted, completed)) = served.progress {
        stela_soc::dcp::gpu::note(submitted, completed);
    }
    // The PE ran nothing but this call since the last one it served.
    let (posted, posted_waited) = client::take_mine();
    let key = served.operation.wrapping_add(1);
    // Slots fill in order and stay claimed: the first that is free or
    // already this operation's is its own.
    let Some(calls) = CALLS.iter().find(|calls| {
        match calls
            .key
            .compare_exchange(0, key, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => true,
            Err(claimed) => claimed == key,
        }
    }) else {
        return;
    };
    calls.count.fetch_add(1, Ordering::Relaxed);
    calls.held.fetch_add(served.held, Ordering::Relaxed);
    calls.longest.fetch_max(served.held, Ordering::Relaxed);
    calls.waited.fetch_add(served.waited, Ordering::Relaxed);
    if served.status == Status::Unmapped {
        calls.unmapped.fetch_add(1, Ordering::Relaxed);
    }
    calls.posted.fetch_add(posted, Ordering::Relaxed);
    calls
        .posted_waited
        .fetch_add(posted_waited, Ordering::Relaxed);
    calls
        .longest_wait
        .fetch_max(served.longest_wait, Ordering::Relaxed);
    calls.compiled.fetch_add(served.compiled, Ordering::Relaxed);
    calls
        .deferrals
        .fetch_add(u64::from(served.deferrals), Ordering::Relaxed);
}

/// Keeps the longest single exit, and says once a boot when one reached
/// [`LONG_EXIT_MILLIS`]: `GPU LONG EXIT operation=… status=… waited_us=…
/// held_us=…`.
fn note_exit(served: &Served) {
    static REPORTED: AtomicBool = AtomicBool::new(false);
    if served.gave_up {
        GAVE_UP.fetch_add(1, Ordering::Relaxed);
    }
    let exit = served.waited.wrapping_add(served.held);
    if LONGEST_EXIT.fetch_max(exit, Ordering::Relaxed) < exit {
        LONGEST_EXIT_OPERATION.store(served.operation, Ordering::Relaxed);
    }
    let tsc_hz = installed().map_or(0, |gpu| gpu.tsc_hz);
    if tsc_hz == 0
        || exit < tsc_hz / 1000 * LONG_EXIT_MILLIS
        || REPORTED.swap(true, Ordering::Relaxed)
    {
        return;
    }
    let micros = |ticks: u64| u128::from(ticks) * 1_000_000 / u128::from(tsc_hz);
    output::arguments(format_args!(
        "GPU LONG EXIT operation={:#x} status={:?} waited_us={} held_us={} (once a boot)\n",
        served.operation,
        served.status,
        micros(served.waited),
        micros(served.held),
    ));
}

/// `GPU LONG TURN operation=… phase=… held_us=…`, once for each operation
/// and phase whose turn reached [`LONG_TURN_MILLIS`]: the phase is the timed
/// operation's longest, `untimed` for an operation timed by no phase.
fn report_long_turn(served: &Served) {
    static SEEN: Seen<128> = Seen::new();
    let tsc_hz = installed().map_or(0, |gpu| gpu.tsc_hz);
    if tsc_hz == 0 || served.held < tsc_hz / 1000 * LONG_TURN_MILLIS {
        return;
    }
    let phase = served
        .phases
        .as_ref()
        .map_or("untimed", |phases| phases.longest_phase().0);
    let key = format!("{:#x} {phase}", served.operation);
    if let Some(unheld) = SEEN.first_time(&key) {
        let detail = served
            .phases
            .as_ref()
            .map(|phases| format!("{}", phases.detail))
            .unwrap_or_default();
        output::arguments(format_args!(
            "GPU LONG TURN operation={:#x} phase={phase} held_us={}{detail}{unheld}\n",
            served.operation,
            u128::from(served.held) * 1_000_000 / u128::from(tsc_hz),
        ));
    }
}

/// What a device holds: its allocation with the part waiting for submitted
/// work and the part of that only held work keeps (now and its peak), that
/// work, the resources automatic heaps answered nil for (no
/// room / room only in pieces), its memory blocks with the resources placed in them and
/// those in memory of their own, and the connections counting the most
/// bytes, each as
/// id:objects/MiB. Nothing while a call holds the service.
fn report_memory(service: &Shared) {
    const SHOWN: usize = 4;
    let Some(mut memory) = service.memory() else {
        return;
    };
    let mib = |bytes: u64| bytes >> 20;
    let device = memory.device;
    output::arguments(format_args!(
        "GPU MEMORY allocated={}M deferred={}/{}M pinned={}/{}M/{}M in_flight={} completed={}/{} recycled={}/{}M reused={}/{} heap_nils={}/{} blocks={}/{}M placed={}/{}M own={} connections={}",
        mib(device.allocated),
        device.deferred,
        mib(device.deferred_bytes),
        device.pinned,
        mib(device.pinned_bytes),
        mib(device.peak_pinned_bytes),
        device.in_flight,
        device.completed,
        device.submitted,
        device.recycled.textures,
        mib(device.recycled.bytes),
        device.recycled.hits,
        device.recycled.hits + device.recycled.misses,
        device.heap_refusals.full,
        device.heap_refusals.fragmented,
        device.blocks.blocks,
        mib(device.blocks.block_bytes),
        device.blocks.placed,
        mib(device.blocks.placed_bytes),
        device.blocks.own,
        memory.connections.len(),
    ));
    memory
        .connections
        .sort_unstable_by_key(|connection| core::cmp::Reverse(connection.allocated));
    for connection in memory.connections.iter().take(SHOWN) {
        output::arguments(format_args!(
            " {}:{}/{}M",
            connection.id,
            connection.objects,
            mib(connection.allocated)
        ));
    }
    output::arguments(format_args!("\n"));
}

/// The operations whose calls held the service longest since the last
/// report: each as count/held/longest/waited/unmapped/posted/posted_waited,
/// microseconds between.
pub(in crate::engine) fn report_calls(tsc_hz: u64) {
    const SHOWN: usize = 8;
    // Peaks a compile or allocation raised with no GPU line after it.
    report_peaks();
    shared().iter().for_each(|service| report_memory(service));
    let micros = |ticks: u64| u128::from(ticks) * 1_000_000 / u128::from(tsc_hz.max(1));
    // How long submitted work took to be seen complete: the GPU's time plus
    // the wait for its interrupt and a look at the device.
    for service in shared().iter() {
        let (count, total, longest) = service.take_latency();
        if let Some(average) = total.checked_div(count) {
            output::arguments(format_args!(
                "GPU LATENCY completed={count} average_us={} longest_us={}\n",
                micros(average),
                micros(longest),
            ));
        }
    }
    // Which part of that latency was waiting for the GPU to start, the GPU's
    // own time, and waiting for the look after it finished.
    for service in shared().iter() {
        let Some(times) = service.gpu_times() else {
            continue;
        };
        if times.timed == 0 && times.waiting == 0 {
            continue;
        }
        let average = |total: u64, count: u64| total.checked_div(count).unwrap_or(0);
        output::arguments(format_args!(
            "GPU TIMES timed={} busy_us={} average_us={} longest_us={}@{}:{} idle_us={} \
             overlap_us={} placed={} queued_us={}/{} noticed_us={}/{} waiting={} \
             span_us={}@{}:{}/{}:{}\n",
            times.timed,
            times.busy_ns / 1000,
            average(times.busy_ns, times.timed) / 1000,
            times.longest_ns / 1000,
            times.longest_serial,
            times.longest_entries,
            times.idle_ns / 1000,
            times.overlap_ns / 1000,
            times.placed,
            micros(average(times.queued_ticks, times.placed)),
            micros(times.longest_queued_ticks),
            micros(average(times.noticed_ticks, times.placed)),
            micros(times.longest_noticed_ticks),
            times.waiting,
            times.span_ns / 1000,
            times.span_serial,
            times.span_index,
            times.span_count,
            times.span_entries,
        ));
    }
    // With `--profile`: the GPU's own time per pass, conversion and copy.
    if pass_times::pass_profile() {
        static UNTIMED: AtomicBool = AtomicBool::new(false);
        for service in shared().iter() {
            if service.has_timestamps() == Some(false) && !UNTIMED.swap(true, Ordering::Relaxed) {
                output::arguments(format_args!(
                    "GPU PASSES unavailable: the device has no timestamp queries\n"
                ));
            }
            if let Some(passes) = service.pass_times()
                && passes.command_buffers != 0
            {
                output::arguments(format_args!("{passes}\n"));
            }
        }
    }
    // The display's presents, and how long they held the boot CPU while a
    // posted call waited for it.
    let [presents, total, longest, on_boot, blocked] = client::take_presents();
    if presents != 0 {
        output::arguments(format_args!(
            "GPU PRESENTS count={presents} on_boot={on_boot} total_us={} longest_us={} \
             blocking_us={}\n",
            micros(total),
            micros(longest),
            micros(blocked),
        ));
    }
    // Pipelines created through the pipeline store's cache: found there, or
    // compiled by the driver, and the driver's time for each.
    let store = pipeline_store::take_lookups();
    if store.hits != 0 || store.misses != 0 {
        output::arguments(format_args!(
            "GPU STORE hits={} misses={} hit_us={} miss_us={} miss_longest_us={}\n",
            store.hits,
            store.misses,
            store.hit_ns / 1000,
            store.miss_ns / 1000,
            store.miss_longest_ns / 1000,
        ));
    }
    // Where the time of the slowest operations went, phase by phase.
    for taken in PHASES.take() {
        output::arguments(format_args!("{}\n", taken.line(tsc_hz)));
    }
    for service in shared().iter() {
        if let Some(line) = service.compiler_report() {
            output::arguments(format_args!("{line}\n"));
        }
        // The compilations calls deferred, which ran outside every turn.
        let jobs = service.take_job_report();
        if jobs.ran != 0 || jobs.deferred != 0 {
            output::arguments(format_args!(
                "GPU JOBS ran={} total_us={} longest_us={} longest_queued_us={} deferred={} \
                 repeated={} forced={} waiting={} heap_rise={}M from={}M\n",
                jobs.ran,
                micros(jobs.ticks),
                micros(jobs.longest),
                micros(jobs.longest_queued),
                jobs.deferred,
                jobs.repeated,
                jobs.forced,
                service.jobs_waiting(),
                JOB_RISE.swap(0, Ordering::Relaxed) >> 20,
                JOB_RISE_FROM.load(Ordering::Relaxed) >> 20,
            ));
        }
    }
    report_lock(tsc_hz);
    let longest = LONGEST_EXIT.swap(0, Ordering::Relaxed);
    let gave_up = GAVE_UP.swap(0, Ordering::Relaxed);
    if longest != 0 || gave_up != 0 {
        output::arguments(format_args!(
            "GPU EXIT longest_us={}@{:#x} gave_up={gave_up}\n",
            micros(longest),
            LONGEST_EXIT_OPERATION.load(Ordering::Relaxed),
        ));
    }
    let mut taken: Vec<_> = CALLS
        .iter()
        .filter_map(|calls| {
            let key = calls.key.load(Ordering::Relaxed);
            let count = calls.count.swap(0, Ordering::Relaxed);
            (key != 0 && count != 0).then(|| {
                (
                    key - 1,
                    count,
                    calls.held.swap(0, Ordering::Relaxed),
                    calls.longest.swap(0, Ordering::Relaxed),
                    calls.waited.swap(0, Ordering::Relaxed),
                    calls.unmapped.swap(0, Ordering::Relaxed),
                    calls.posted.swap(0, Ordering::Relaxed),
                    calls.posted_waited.swap(0, Ordering::Relaxed),
                )
            })
        })
        .collect();
    if let Some(pool) = client::pool_report(tsc_hz) {
        output::arguments(format_args!("{pool}\n"));
    }
    let posted = client::take_posted();
    if taken.is_empty() && posted.is_empty() {
        return;
    }
    taken.sort_unstable_by_key(|&(_, _, held, ..)| core::cmp::Reverse(held));
    let held: u64 = taken.iter().map(|&(_, _, held, ..)| held).sum();
    output::arguments(format_args!(
        "GPU CALLS held_us={} unreached={} \
         operation:count/held_us/longest_us/waited_us/unmapped/posted/posted_us",
        micros(held),
        UNREACHED.swap(0, Ordering::Relaxed),
    ));
    for &(operation, count, held, longest, waited, unmapped, posted, posted_waited) in
        taken.iter().take(SHOWN)
    {
        output::arguments(format_args!(
            " {operation:#x}:{count}/{}/{}/{}/{unmapped}/{posted}/{}",
            micros(held),
            micros(longest),
            micros(waited),
            micros(posted_waited),
        ));
    }
    // The host's Linux work the calls asked the boot CPU for, by ioctl.
    let (shootdowns, shot, ranged) = crate::native_mmio::shootdown::take_waits();
    output::arguments(format_args!(
        "\nGPU POSTED shootdowns={shootdowns}/{}us ranged={ranged} request:count/waited_us",
        micros(shot)
    ));
    for &(request, count, waited) in posted.iter().take(SHOWN) {
        output::arguments(format_args!(" {request:#x}:{count}/{}", micros(waited)));
    }
    // How long they waited for the boot CPU to take them.
    output::arguments(format_args!("\nGPU TAKEN request:count/us"));
    for &(request, count, taken) in client::take_taken().iter().take(SHOWN) {
        output::arguments(format_args!(" {request:#x}:{count}/{}", micros(taken)));
    }
    // How long they waited for a worker to take them.
    output::arguments(format_args!("\nGPU QUEUED request:count/us"));
    for &(request, count, queued) in client::take_queued().iter().take(SHOWN) {
        output::arguments(format_args!(" {request:#x}:{count}/{}", micros(queued)));
    }
    // What performing them took the service task, waits for the GPU included.
    output::arguments(format_args!("\nGPU SERVED request:count/us"));
    for &(request, count, took) in client::take_served().iter().take(SHOWN) {
        output::arguments(format_args!(" {request:#x}:{count}/{}", micros(took)));
    }
    output::arguments(format_args!("\n"));
}

/// `GPU LOCK operation:held_us/waited_us/longest_wait_us/compiled_us/compiles`:
/// for each operation that held or waited for the service's turn since the
/// last report, the turn's time, the waits for it, the longest one wait,
/// and the compilations it ran outside the turn. Reads before
/// [`report_calls`]'s `GPU CALLS`, which takes the counts.
fn report_lock(tsc_hz: u64) {
    const SHOWN: usize = 12;
    let micros = |ticks: u64| u128::from(ticks) * 1_000_000 / u128::from(tsc_hz.max(1));
    let mut rows: Vec<_> = CALLS
        .iter()
        .filter_map(|calls| {
            let key = calls.key.load(Ordering::Relaxed);
            let longest_wait = calls.longest_wait.swap(0, Ordering::Relaxed);
            let compiled = calls.compiled.swap(0, Ordering::Relaxed);
            let deferrals = calls.deferrals.swap(0, Ordering::Relaxed);
            let held = calls.held.load(Ordering::Relaxed);
            let waited = calls.waited.load(Ordering::Relaxed);
            (key != 0 && (held | waited | compiled) != 0).then_some((
                key - 1,
                held,
                waited,
                longest_wait,
                compiled,
                deferrals,
            ))
        })
        .collect();
    if rows.is_empty() {
        return;
    }
    rows.sort_unstable_by_key(|&(_, held, waited, ..)| core::cmp::Reverse(held + waited));
    let mut line =
        String::from("GPU LOCK operation:held_us/waited_us/longest_wait_us/compiled_us/compiles");
    for &(operation, held, waited, longest, compiled, deferrals) in rows.iter().take(SHOWN) {
        line.push_str(&format!(
            " {operation:#x}:{}/{}/{}/{}/{deferrals}",
            micros(held),
            micros(waited),
            micros(longest),
            micros(compiled),
        ));
    }
    output::arguments(format_args!("{line}\n"));
}

/// NVK's entry point, as Vulkan types it.
unsafe extern "system" fn entry(
    instance: vk::Instance,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    use vk::Handle;
    // SAFETY: the loader's call, and both ends are one function pointer.
    unsafe {
        core::mem::transmute(stela_mesa::vk_icdGetInstanceProcAddr(
            instance.as_raw() as *mut c_void,
            name,
        ))
    }
}

/// The host pages behind whole pages of guest RAM.
fn guest_pages(
    ranges: &[MemoryRange],
    loan: Option<&ExternalDmaLoan<'_>>,
) -> Result<(Vec<Run>, u64), Error> {
    const NOT_RAM: Error = Error::Invalid("imported memory is whole pages of guest RAM");
    // Linux stores each run's length in a 32-bit scatterlist entry.
    const MAX_RUN: u64 = u32::MAX as u64 / PAGE * PAGE;
    let mut runs: Vec<Run> = Vec::new();
    let mut total = 0u64;
    for range in ranges {
        if !(range.address | range.bytes).is_multiple_of(PAGE)
            || range.address.checked_add(range.bytes).is_none()
        {
            return Err(NOT_RAM);
        }
        total = total.checked_add(range.bytes).ok_or(NOT_RAM)?;
    }
    let mut valid = true;
    let mut append = |mut physical: u64, mut bytes: u64| {
        if !(physical | bytes).is_multiple_of(PAGE) || physical.checked_add(bytes).is_none() {
            valid = false;
            return;
        }
        while bytes != 0 {
            let mut count = bytes.min(MAX_RUN);
            if let Some(last) = runs.last_mut()
                && last.physical + last.bytes == physical
                && last.bytes < MAX_RUN
            {
                count = count.min(MAX_RUN - last.bytes);
                last.bytes += count;
            } else {
                runs.push(Run {
                    physical,
                    bytes: count,
                });
            }
            physical += count;
            bytes -= count;
        }
    };
    let whole = match loan {
        Some(loan) => loan.visit_physical_runs(&mut append).is_ok(),
        None => ranges
            .iter()
            .all(|range| host::guest_runs(range.address, range.bytes, &mut append)),
    };
    if !whole || !valid || total == 0 || u32::try_from(runs.len()).is_err() {
        return Err(NOT_RAM);
    }
    Ok((runs, total))
}

/// Guest RAM the kext wired, for NVK to import. The GPU reaches it through
/// the host pages behind it.
struct Exporter;

impl MemoryExporter for Exporter {
    fn export(&self, ranges: &[MemoryRange]) -> Result<ExportedMemory, Error> {
        let owner = host::guest_dma_owner();
        let mut loan = owner
            .as_ref()
            .map(|ram| ram.prepare_external_dma(ranges))
            .transpose()
            .map_err(|_| Error::Invalid("imported guest RAM is unavailable"))?;
        let (runs, bytes) = guest_pages(ranges, loan.as_ref())?;
        let result = client::export(&runs, bytes, || {
            if let Some(loan) = &mut loan {
                loan.begin_publication();
            }
        });
        let (fd, pointer) = match result {
            Ok(export) => export,
            Err(error) => {
                if let (client::ExportFailure::Unpublished, Some(loan)) = (error, &mut loan) {
                    // SAFETY: Linux received no runs. Either no CPU alias
                    // was installed or every CPU acknowledged retirement.
                    unsafe { loan.cancel_publication() };
                }
                return Err(Error::OutOfMemory);
            }
        };
        Ok(ExportedMemory { fd, pointer, bytes })
    }

    fn close(&self, exported: ExportedMemory) {
        client::close_descriptor(exported.fd);
        self.unmap(exported);
    }

    fn unmap(&self, exported: ExportedMemory) {
        client::unmap_export(exported.pointer.as_ptr() as u64, exported.bytes);
    }
}

fn service(gpu: &Gpu, stack: u64) -> Result<Arc<Shared>, OpenError> {
    let compiler = |physical| {
        // SAFETY: NVK's own physical device, whose options last as long.
        let options = unsafe { stela_nvk_nir_options(physical) };
        compiler::Service::linked(NonNull::new(options.cast_mut()).expect("NVK has NIR options"))
    };
    let loader = Loader {
        get_instance_proc_addr: entry,
        log: driver_message,
        // A commit's submission runs where Linux does, not on its PE.
        submit_thread: Some(stela_nvk_submit_thread),
        // A refused submission fails its command buffer, not the device.
        refusals: Some(Refusals {
            take: stela_nvk_submit_failure,
            forget: stela_nvk_forget_failures,
            count: stela_nvk_count_failures,
        }),
        // An import's driver work after its turn may wait for Linux here.
        relax,
    };
    // SAFETY: the linked NVK and compiler, which last for the boot.
    let service = unsafe {
        open(
            loader,
            Box::new(Exporter),
            compiler,
            gpu,
            DeviceEpoch::from_registered(NonZeroU64::MIN),
        )
    }?;
    let gpu = *gpu;
    // A lost device's channels are gone, so the next client gets a new one,
    // opened on the service's stack as the first was.
    // SAFETY: as above.
    let service = service.with_reopen(Box::new(move || unsafe {
        reopen(loader, Box::new(Exporter), compiler, &gpu)
    }));
    Ok(Arc::new(SharedService::new(
        service,
        ServiceHost {
            relax,
            stack,
            run,
            report,
            note,
            ticks: host::rdtsc,
            served,
            wake: look_again,
            bundle,
        },
    )))
}

/// Open a service over every bound GPU. What stops one is reported, and the
/// Mac boots without Metal on that GPU.
pub(in crate::engine) fn open_service() {
    let Some(gpu) = installed() else {
        return;
    };
    let mut constructed = false;
    let services: Vec<_> = gpu
        .start
        .bound
        .iter()
        .enumerate()
        .filter_map(|(index, bound)| open_gpu(gpu, index, bound, &mut constructed))
        .collect();
    SERVICES.store(Box::into_raw(Box::new(services)), Ordering::Release);
}

/// Runs the image's C constructors once, on the stack of whoever opens first.
fn construct(constructed: &mut bool) {
    if !core::mem::replace(constructed, true) {
        // SAFETY: the one call, before any other of the C's.
        unsafe { stela_mesa::stela_constructors() };
    }
}

/// The service of `bound`, at DRM minor `index`, whose failure is reported.
fn open_gpu(
    gpu: &Installed,
    index: usize,
    bound: &Bound,
    constructed: &mut bool,
) -> Option<Arc<Shared>> {
    let refuse = |reason: &str| {
        crate::output::arguments(format_args!("GPU {}: {reason}\n", bound.name));
        None
    };
    let Some(link) = bound.link else {
        return refuse("no PCIe link");
    };
    let Some(chipset) = client::chipset(index as u32) else {
        return refuse("no chipset reported");
    };
    let Some(stack) = Stack::map(gpu.window, STACK_BYTES) else {
        return refuse("no memory for its service");
    };
    let top = stack.top();
    // It serves for the boot.
    core::mem::forget(stack);
    let target = Gpu {
        pci: PciBusInfo {
            domain: bound.drm.segment.into(),
            bus: bound.drm.bus.into(),
            device: bound.drm.slot.into(),
            function: bound.drm.function.into(),
        },
        chipset,
        // A GPU in a Thunderbolt enclosure reaches Stela as the function
        // firmware enumerated, like any other in a slot.
        attachment: Attachment {
            link,
            external: false,
            location_number: index as u64,
        },
    };
    let mut opened = None;
    run(top, &mut || {
        construct(constructed);
        opened = Some(service(&target, top));
    });
    match opened? {
        Ok(service) => {
            // Without a second stack, every compilation stays in its turn.
            service.set_wait_bound(gpu.tsc_hz / 1000 * WAIT_BOUND_MILLIS);
            service.set_look_budget(gpu.tsc_hz / 1_000_000 * LOOK_MICROS);
            if let Some(compiling) = Stack::map(gpu.window, STACK_BYTES) {
                service.set_force_after(gpu.tsc_hz / 1000 * FORCE_JOB_MILLIS);
                if let Some(free) = COMPILE_STACKS
                    .iter()
                    .find(|compile| compile.load(Ordering::Relaxed) == 0)
                {
                    free.store(compiling.top(), Ordering::Relaxed);
                }
                service.set_compile_stack(compiling.top());
                core::mem::forget(compiling);
                // Imports' exports and driver imports run after their turn
                // on a stack of their own; without one they stay in it. So
                // do the frees and semaphores the boot CPU's look keeps up.
                if let Some(outside) = Stack::map(gpu.window, STACK_BYTES) {
                    service.set_outside_stack(outside.top());
                    core::mem::forget(outside);
                }
            } else {
                output::arguments(format_args!(
                    "GPU {}: no memory for a compile stack; compilations hold the service\n",
                    bound.name
                ));
            }
            Some(service)
        }
        Err(error) => refuse(&format!("Metal could not open: {error:?}")),
    }
}

/// `--profile`: every device times its command buffers' passes apart and
/// the report shows `GPU PASSES`.
pub(in crate::engine) fn profile_passes() {
    pass_times::set_pass_profile(true);
}

pub(super) fn shared() -> &'static [Arc<Shared>] {
    // SAFETY: the published services are leaked, so they live for the boot.
    unsafe { SERVICES.load(Ordering::Acquire).as_ref() }.map_or(&[], Vec::as_slice)
}

/// Tell the services a GPU interrupted: its MSI says nothing of which.
pub(super) fn interrupted() {
    shared().iter().for_each(|service| service.interrupted());
}

/// Waits on the completion timeline of each device that has not yet given
/// it, as a newly opened or reopened device has: completed work is then
/// looked for at once.
pub(super) fn watch_completions() {
    for (index, service) in shared().iter().enumerate() {
        if let Some((fd, timeline)) = service.take_completion() {
            super::waiter::start(fd, index, timeline);
        }
    }
}

/// Whether service `index` has submitted work not yet published complete.
pub(super) fn awaits_completion(index: usize) -> bool {
    shared()
        .get(index)
        .is_some_and(|service| service.awaits_completion())
}

/// Service `index`'s completion timeline `timeline` reached `value`.
pub(super) fn completed_through(index: usize, timeline: u64, value: u64) {
    if let Some(service) = shared().get(index) {
        service.completed_through(timeline, value);
    }
}

/// Bring every GPU to rest before the PC resets or turns off, on the boot CPU
/// outside Linux's callbacks. Admission closes for good and accepted work
/// finishes; `close_gates` then releases what the guest held of each GPU that
/// proved idle, answering the imports still in reach. Whatever could not be
/// proven stays alive and is reported: the transition itself ends the DMA.
pub(in crate::engine) fn stop(close_gates: impl FnOnce() -> u64) {
    let Some(gpu) = installed() else {
        return;
    };
    let services = shared();
    let ms = |since: u64| host::rdtsc().wrapping_sub(since) / (gpu.tsc_hz / 1000).max(1);
    let began = host::rdtsc();
    services.iter().for_each(|service| service.begin_quiesce());
    for (index, service) in services.iter().enumerate() {
        let status = settle(service, gpu.tsc_hz);
        output::arguments(format_args!("GPU {index} at the stop: {status:?}\n"));
    }
    let quiesce_ms = ms(began);
    let closing = host::rdtsc();
    let reachable = close_gates();
    let close_ms = ms(closing);
    let quarantine = dma_quarantine_status();
    output::arguments(format_args!(
        "GPU stopped: {reachable} imports in reach, {} DMA mappings of {} bytes and {} functions quarantined\n",
        quarantine.mappings,
        quarantine.bytes,
        quarantined_functions(),
    ));
    super::waiter::stop();
    let unloading = host::rdtsc();
    let interrupts = super::interrupts();
    let done = client::shutdown(STOP_UNLOAD_SECONDS);
    // Whether the GPU still interrupted while its driver unloaded, and
    // whether Linux's timers still ran, which its message waits sleep on.
    output::arguments(format_args!(
        "GPU UNLOAD waited_ms={} done={done} quiesce_ms={quiesce_ms} close_ms={close_ms} \
         interrupts={} timer_failure={:?}\n",
        ms(unloading),
        super::interrupts().wrapping_sub(interrupts),
        stela_linux::service::timer_delivery_failure(),
    ));
    if done {
        output::arguments(format_args!("GPU drivers stopped\n"));
    } else {
        output::arguments(format_args!(
            "GPU drivers still stopping after {STOP_UNLOAD_SECONDS} s\n"
        ));
        // The calls still running, the stop among them, and for how long.
        if let Some(line) = client::pool_report(gpu.tsc_hz) {
            output::arguments(format_args!("{line}\n"));
        }
    }
}

/// Unload the GPUs' firmware before a fatal stop ends the machine: the reset
/// that follows takes down the link of a card whose GSP still runs, until a
/// power cycle. The Linux environment that unloads it runs on the boot CPU,
/// so a stop raised there leaves the firmware running and says so. Once.
pub(in crate::engine) fn stop_for_fatal() {
    static STOPPING: AtomicBool = AtomicBool::new(false);
    if installed().is_none() || STOPPING.swap(true, Ordering::AcqRel) {
        return;
    }
    if host::processor() == BOOT_CPU.load(Ordering::Relaxed) {
        output::arguments(format_args!(
            "GPU firmware left running: the stop is on the CPU that unloads it\n"
        ));
    } else if client::shutdown_for_fatal(STOP_UNLOAD_SECONDS) {
        output::arguments(format_args!("GPU drivers stopped before the fatal stop\n"));
    } else {
        output::arguments(format_args!(
            "GPU drivers still stopping after {STOP_UNLOAD_SECONDS} s; the firmware may still run\n"
        ));
    }
}

/// Step `service` until accepted work ends or stops moving.
fn settle(service: &Shared, tsc_hz: u64) -> QuiesceStatus {
    let (mut last, mut moved) = (None, host::rdtsc());
    loop {
        let status = service.quiesce_step();
        let QuiesceStatus::Pending { completed, .. } = status else {
            return status;
        };
        if last != Some(completed) {
            (last, moved) = (Some(completed), host::rdtsc());
        } else if host::rdtsc() - moved >= STOP_STALL_SECONDS * tsc_hz {
            return status;
        }
        relax();
    }
}

/// Add each service's gate node to `tree`, for what the board mounts there.
pub(in crate::engine) fn place(
    tree: &mut DeviceTree,
) -> Vec<(stela_gpu::Placement, Arc<dyn GateService>)> {
    let mut gates = Vec::new();
    for (index, service) in shared().iter().enumerate() {
        match stela_gpu::place_gpu_gate(tree, index, window::BYTES) {
            Ok(placement) => gates.push((placement, service.clone() as Arc<dyn GateService>)),
            Err(error) => {
                crate::status::warn(&format!("GPU {index}'s gate has no place: {error}"));
            }
        }
    }
    gates
}
