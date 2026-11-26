//! `stela.efi`'s persistent translation store: the pack `stela setup` makes room for in
//! `stela/store` on the boot USB's FAT32 data partition.
//!
//! Before boot services end, [`read`] finds the volume with `store.cfg`,
//! gives this translator the preallocated spare file when it has no pack
//! yet, reads as much of the pack as this PC's RAM budget holds and notes
//! the volume's GPT partition. The engine reads blocks from the pack
//! ([`install`]) and, when the guest restarts or powers off, proves the
//! blocks it compiled anew and rewrites the pack in place over the file's
//! clusters, within the same budget ([`write`]), found by Stela's own FAT32
//! reader on the partition the storage drivers opened. The pack is a cache:
//! every failure leaves the boot running without a store, or without
//! writing it.
//!
//! When every PE runs on its own host core (the default), their machines
//! share the store ([`Shared`]), split by page key into parts behind locks
//! of their own, and it keeps the blocks any of them compiled for the others
//! to install ([`share`]). Idle host CPUs without guest PEs prove batches
//! independently. With no spare CPU a PE at EL0 proves a few waiting blocks.
use super::bundle::Bytes;
use super::host;
use crate::{
    apic::Apic, firmware, handoff::Region, native_smp::executor::Executors, output, paging::PAGE,
};
use alloc::{boxed::Box, format, string::String, vec};
use core::{
    cell::UnsafeCell,
    ptr::NonNull,
    sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering},
};
use stela_boot_core::host_memory::PACK_FLOOR;
use stela_drivers::block::BlockDevice;
use stela_engine::machine::Offer;
use stela_engine::machine::Translations;
use stela_engine::machine::engine::StoreEvent;
use stela_engine::machine::engine::publication::WorkerBinding;
use stela_engine::translations;
use stela_engine::translations::Digest;
use stela_engine::translations::Record;
use stela_engine::translations::pack;
use stela_engine::translations::pack::{DecodedRecord, Pack, StagedOffer, Store};
#[cfg(feature = "nvk")]
use stela_formats::payload::pieces::Extent;
use stela_loader::mount_interface::layout::{DATA_STORE, Layout};
use stela_runtime::pc::fat32::Fat32;
use stela_runtime::pc::gpt::Guid;
use stela_runtime::translation_store;
use uefi::{
    CString16, boot,
    proto::{
        device_path::{DevicePath, DevicePathNodeEnum, media::PartitionSignature},
        media::{
            file::{Directory, File as _, FileAttribute, FileInfo, FileMode, RegularFile},
            fs::SimpleFileSystem,
        },
    },
};

/// The handoff file holding the store's placement and its pack.
pub const FILE: &str = "translations";
const CONFIG: &str = "store.cfg";
const BYTES_KEY: &str = "bytes";
const CONFIG_LIMIT: usize = 256;
/// A pack's file name: the first bytes of its translator id in hex, an 8.3
/// name the FAT32 reader finds.
const NAME_BYTES: usize = 4;
const PACK_EXTENSION: &str = "pak";
/// A blank, preallocated pack file `stela setup` keeps beside the packs: a
/// new translator's pack takes it by rename. The boot never grows a file,
/// which the firmware does by writing every byte.
const SPARE: &str = "spare.bin";
/// The pack is read in steps of this many bytes, each shown as progress.
const PROGRESS_STEP: usize = 16 << 20;
/// The placement record at the start of the handoff file: magic, bound,
/// partition GUID, flags, pack bytes; the pack follows.
const PLACEMENT_MAGIC: [u8; 8] = *b"STLTSTOR";
const PLACEMENT_BYTES: usize = 64;
/// The pack file holds the bound's bytes and its volume's partition is
/// known: the engine may rewrite it.
const WRITABLE: u32 = 1;
const NORTHWOOD_LAYOUT: u32 = 2;
/// The words of the buffer a renamed file's `FileInfo` is built in: its
/// fixed fields and an 8.3 name.
const FILE_INFO_WORDS: usize = 64;

/// The pack's 8.3 name, independent of its store directory.
fn pack_name(translator: &Digest) -> String {
    format!("{}.{PACK_EXTENSION}", hex(&translator.0[..NAME_BYTES]))
}

pub(super) fn store_path(layout: Layout, name: &str) -> String {
    layout.data_path(&format!("{DATA_STORE}/{name}"))
}

pub(super) fn uefi_store_path(layout: Layout, name: &str) -> String {
    store_path(layout, name).replace('/', "\\")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(super) fn open(root: &mut Directory, path: &str, mode: FileMode) -> Option<RegularFile> {
    let path = CString16::try_from(path).ok()?;
    root.open(&path, mode, FileAttribute::empty())
        .ok()
        .and_then(|handle| crate::file_kind::regular(handle).ok())
}

/// Read `buffer.len()` bytes from the current file position.
pub(super) fn read_exact(file: &mut RegularFile, buffer: &mut [u8]) -> Option<()> {
    let mut done = 0;
    while done < buffer.len() {
        let count = file.read(&mut buffer[done..]).ok()?;
        if count == 0 {
            return None;
        }
        done += count;
    }
    Some(())
}

/// `store.cfg`'s bound.
fn bound(root: &mut Directory, layout: Layout) -> Option<u64> {
    let mut file = open(root, &uefi_store_path(layout, CONFIG), FileMode::Read)?;
    let mut text = vec![0; CONFIG_LIMIT];
    let mut done = 0;
    while done < text.len() {
        let count = file.read(&mut text[done..]).ok()?;
        if count == 0 {
            break;
        }
        done += count;
    }
    core::str::from_utf8(&text[..done])
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix(BYTES_KEY)?.trim().parse().ok())
}

/// The GPT partition GUID of the volume on `handle`.
pub(crate) fn partition(handle: uefi::Handle) -> Option<[u8; 16]> {
    // SAFETY: the path is only read, while no driver changes this handle.
    let path = unsafe {
        boot::open_protocol::<DevicePath>(
            boot::OpenProtocolParams {
                handle,
                agent: boot::image_handle(),
                controller: None,
            },
            boot::OpenProtocolAttributes::GetProtocol,
        )
    }
    .ok()?;
    path.node_iter().find_map(|node| match node.as_enum() {
        Ok(DevicePathNodeEnum::MediaHardDrive(drive)) => match drive.partition_signature() {
            PartitionSignature::Guid(guid) => Some(guid.to_bytes()),
            _ => None,
        },
        _ => None,
    })
}

/// What the loader read of the store's pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackRead {
    /// No pack yet, or one the engine would discard.
    None,
    /// All of the pack's `0` bytes.
    Whole(u64),
    /// The pack's first `read` of its `held` bytes: the records that end
    /// within this PC's budget.
    Part { read: u64, held: u64 },
    /// None of it: this PC's budget holds less than its index, or none at
    /// all. The pack stays on the volume, unwritten, for a PC with more RAM.
    Skipped,
}

/// The store the loader found: the handoff file's pages, whether the engine
/// can rewrite its pack, and what of the pack it read.
pub struct Opened {
    pub region: Region,
    pub writable: bool,
    pub pack: PackRead,
}

/// The store on the first volume with `store.cfg`, as the handoff file's
/// pages: run before boot services end, with no volume opened exclusively.
/// `budget` is the most of the pack this PC's RAM holds
/// (`host_memory::pack_budget`); the pack is read within it and the engine
/// rewrites it within it, so that it never grows past what the next boot on
/// this PC can read. `None` when no volume has a store.
pub fn read(budget: u64) -> Option<Opened> {
    let translator = translations::translator_id();
    with_configured(|handle, root, layout, bound| {
        let limit = bound.min(budget);
        let path = uefi_store_path(layout, &pack_name(&translator));
        if open(root, &path, FileMode::Read).is_none() {
            claim_spare(root, layout, &translator);
        }
        // A read-only volume's pack is read and never written; a missing
        // one leaves the boot's translations in memory.
        let opened = open(root, &path, FileMode::ReadWrite)
            .map(|file| (file, true))
            .or_else(|| open(root, &path, FileMode::Read).map(|file| (file, false)));
        let partition = partition(handle);
        let reading = output::now();
        let (region, pack, allocated) = match opened {
            Some((mut file, writable)) => {
                let allocated = writable
                    && file
                        .get_boxed_info::<FileInfo>()
                        .is_ok_and(|info| info.file_size() >= bound);
                let (region, pack) = read_pack(&mut file, &translator, limit);
                (region, pack, allocated)
            }
            None => (None, PackRead::None, false),
        };
        let (read, held) = match pack {
            PackRead::None | PackRead::Skipped => (0, 0),
            PackRead::Whole(bytes) => (bytes, bytes),
            PackRead::Part { read, held } => (read, held),
        };
        output::arguments(format_args!(
            "TRANSLATIONS {path} bound={bound} budget={budget} pack={read} held={held} \
             skipped={} allocated={allocated} partition={} read_ms={}\n",
            pack == PackRead::Skipped,
            partition.is_some(),
            output::millis_since(reading)
        ));
        let region = match region {
            Some(region) => region,
            None => Region {
                physical: firmware::pages(1).ok()? as u64,
                bytes: PLACEMENT_BYTES as u64,
            },
        };
        // Below the floor, or with its pack left for a larger PC, the boot
        // keeps its translations in memory.
        let writable =
            if allocated && partition.is_some() && limit >= PACK_FLOOR && pack != PackRead::Skipped
            {
                WRITABLE
            } else {
                0
            };
        // SAFETY: the exclusive loader allocation starts with space for
        // the placement header. Initialize it before taking a mutable slice.
        let placement = unsafe {
            let start = region.physical as *mut u8;
            start.write_bytes(0, PLACEMENT_BYTES);
            core::slice::from_raw_parts_mut(start, PLACEMENT_BYTES)
        };
        placement[..8].copy_from_slice(&PLACEMENT_MAGIC);
        // The engine lays out the next pack within this boot's limit.
        placement[8..16].copy_from_slice(&limit.to_le_bytes());
        placement[16..32].copy_from_slice(&partition.unwrap_or_default());
        let pack_bytes = region.bytes - PLACEMENT_BYTES as u64;
        placement[40..48].copy_from_slice(&pack_bytes.to_le_bytes());
        let flags = writable
            | if layout == Layout::Northwood {
                NORTHWOOD_LAYOUT
            } else {
                0
            };
        placement[32..36].copy_from_slice(&flags.to_le_bytes());
        Some(Opened {
            region,
            writable: flags & WRITABLE != 0,
            pack,
        })
    })
}

/// Use the first configured store, searching Northwood on all volumes first.
/// Keep the volume and the selected layout together for the callback.
pub(super) fn with_configured<T>(
    read: impl FnOnce(uefi::Handle, &mut Directory, Layout, u64) -> Option<T>,
) -> Option<T> {
    let handles = boot::find_handles::<SimpleFileSystem>().ok()?;
    for layout in Layout::PREFERRED {
        for &handle in &handles {
            let Ok(mut volume) = boot::open_protocol_exclusive::<SimpleFileSystem>(handle) else {
                continue;
            };
            let Ok(mut root) = volume.open_volume() else {
                continue;
            };
            if let Some(bound) = bound(&mut root, layout) {
                return read(handle, &mut root, layout, bound);
            }
        }
    }
    None
}

/// Give `translator` the spare pack file by renaming it, when the store
/// has one: the file keeps its preallocated clusters.
fn claim_spare(root: &mut Directory, layout: Layout, translator: &Digest) {
    let Some(mut file) = open(root, &uefi_store_path(layout, SPARE), FileMode::ReadWrite) else {
        return;
    };
    let Ok(info) = file.get_boxed_info::<FileInfo>() else {
        return;
    };
    let Ok(name) = CString16::try_from(pack_name(translator).as_str()) else {
        return;
    };
    let mut storage = vec![0u64; FILE_INFO_WORDS];
    // SAFETY: the words' bytes, 8-byte aligned as FileInfo requires.
    let bytes = unsafe {
        core::slice::from_raw_parts_mut(storage.as_mut_ptr().cast::<u8>(), storage.len() * 8)
    };
    let Ok(renamed) = FileInfo::new(
        bytes,
        info.file_size(),
        info.physical_size(),
        *info.create_time(),
        *info.last_access_time(),
        *info.modification_time(),
        info.attribute(),
        &name,
    ) else {
        return;
    };
    let claimed = file.set_info(renamed).is_ok() && file.flush().is_ok();
    output::arguments(format_args!(
        "TRANSLATIONS spare {} as {name}\n",
        if claimed { "taken" } else { "refused" }
    ));
}

/// Read a checked pack directly into its final handoff region: all of it,
/// or, when it passes `limit`, its first bytes holding the records that end
/// within it, its index trimmed to name only them
/// (`translation_store::within`).
fn read_pack(
    file: &mut RegularFile,
    translator: &Digest,
    limit: u64,
) -> (Option<Region>, PackRead) {
    let mut header = [0; pack::HEADER_BYTES];
    if read_exact(file, &mut header).is_none() {
        return (None, PackRead::None);
    }
    let Ok(prefix_len) = pack::prefix_bytes(&header, translator) else {
        return (None, PackRead::None);
    };
    if prefix_len as u64 > limit {
        return (None, PackRead::Skipped);
    }
    let mut prefix = vec![0; prefix_len];
    prefix[..header.len()].copy_from_slice(&header);
    if read_exact(file, &mut prefix[header.len()..]).is_none() {
        return (None, PackRead::None);
    }
    let Ok(within) = translation_store::within(&prefix, translator, limit) else {
        return (None, PackRead::None);
    };
    if within.records == 0 && within.omitted != 0 {
        return (None, PackRead::Skipped);
    }
    let length = within.length as usize;
    let bytes = PLACEMENT_BYTES + length;
    let pages = bytes.div_ceil(PAGE);
    let Ok(physical) = firmware::pages(pages) else {
        return (None, PackRead::None);
    };
    // SAFETY: firmware returned this exclusive allocation and `bytes` fits
    // in it. Initialize the pack range before borrowing it as a slice.
    let pack = unsafe {
        let start = (physical as *mut u8).add(PLACEMENT_BYTES);
        start.write_bytes(0, length);
        core::slice::from_raw_parts_mut(start, length)
    };
    pack[..prefix_len].copy_from_slice(&prefix);
    drop(prefix);
    let total = length >> 20;
    for (index, part) in pack[prefix_len..].chunks_mut(PROGRESS_STEP).enumerate() {
        if read_exact(file, part).is_none() {
            let start = NonNull::new(physical as *mut u8).expect("firmware allocation");
            // SAFETY: the pack slice is no longer used and these are the
            // original base and page count of the loader allocation.
            let _ = unsafe { boot::free_pages(start, pages) };
            return (None, PackRead::None);
        }
        let done = (prefix_len + index * PROGRESS_STEP + part.len()) >> 20;
        crate::status::progress(&format!("{done} of {total} MB"));
    }
    let read = if within.omitted == 0 {
        PackRead::Whole(length as u64)
    } else {
        translation_store::trim(pack);
        PackRead::Part {
            read: length as u64,
            held: within.pack_length,
        }
    };
    let region = Region {
        physical: physical as u64,
        bytes: bytes as u64,
    };
    (Some(region), read)
}

/// Where the boot's store is written: its bound, the pack's path and the
/// partition. Its parts ([`PARTS`]) hold the store itself.
struct Boot {
    translator: Digest,
    bound: u64,
    path: String,
    partition: Option<Box<dyn BlockDevice>>,
}

/// The parts a boot's store is split into by page key, each behind a lock
/// of its own, so that PEs looking up, sharing and offering blocks of
/// different pages do not wait for each other.
const PARTS: usize = stela_boot_core::topology::MAX_CORES;
const BASE_PARTS: usize = 16;
const _: () = assert!(PARTS.is_power_of_two() && PARTS <= u64::BITS as usize);
/// Retain the existing shard width for small hosts and read-only boots.
static SHARDS: AtomicUsize = AtomicUsize::new(BASE_PARTS);

/// A spin lock that counts how often it was found held and how long its
/// waiters spun.
#[repr(align(64))]
struct Lock<T> {
    held: AtomicBool,
    value: UnsafeCell<T>,
    acquired: AtomicU64,
    contended: AtomicU64,
    spun: AtomicU64,
}
// SAFETY: `with` holds `held` for every access to `value`.
unsafe impl<T> Sync for Lock<T> {}

impl<T> Lock<T> {
    const fn new(value: T) -> Self {
        Self {
            held: AtomicBool::new(false),
            value: UnsafeCell::new(value),
            acquired: AtomicU64::new(0),
            contended: AtomicU64::new(0),
            spun: AtomicU64::new(0),
        }
    }

    /// Run `f` on the value, alone. No interrupt handler reaches it.
    fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        if self
            .held
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            let started = output::now();
            while self
                .held
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                while self.held.load(Ordering::Relaxed) {
                    core::hint::spin_loop();
                }
            }
            self.contended.fetch_add(1, Ordering::Relaxed);
            self.spun
                .fetch_add(output::now().wrapping_sub(started), Ordering::Relaxed);
        }
        self.acquired.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `held` excludes every other access until it is released.
        let result = f(unsafe { &mut *self.value.get() });
        self.held.store(false, Ordering::Release);
        result
    }
}

static BOOT: Lock<Option<Boot>> = Lock::new(None);
static STORES: [Lock<Option<Store<'static>>>; PARTS] = [const { Lock::new(None) }; PARTS];

/// The part of the store that holds the page whose bytes hash to `key`.
fn part(key: &Digest) -> &'static Lock<Option<Store<'static>>> {
    &STORES[part_index(key)]
}

fn part_index(key: &Digest) -> usize {
    usize::from(key.0[2]) & (SHARDS.load(Ordering::Relaxed) - 1)
}

fn stores() -> &'static [Lock<Option<Store<'static>>>] {
    &STORES[..SHARDS.load(Ordering::Relaxed)]
}

/// Keep the store's capabilities off the locks' cache lines so a PE can
/// skip operations that have no backing without contending with its users.
#[repr(align(64))]
struct Available {
    capabilities: AtomicU8,
    /// A preflight limit for staging. Admission checks the live store bound.
    shared_bound: AtomicUsize,
    /// The fetch pages the machines keep in all ([`Translations::kept_pages`]).
    kept_pages: AtomicUsize,
}

static AVAILABLE: Available = Available {
    capabilities: AtomicU8::new(0),
    shared_bound: AtomicUsize::new(0),
    kept_pages: AtomicUsize::new(0),
};
const READ_PACK: u8 = 1 << 0;
const WRITE_PACK: u8 = 1 << 1;
const SHARE_BLOCKS: u8 = 1 << 2;

fn available(capabilities: u8) -> bool {
    AVAILABLE.capabilities.load(Ordering::Acquire) & capabilities != 0
}

/// What the machines' use of the store cost, beyond the parts' counters:
/// fetch pages read and hashed for their records and the engine counter
/// ticks that took, shared blocks found but not installed, and the TSC
/// ticks record lookups took through the store and decoding records.
struct Costs {
    page_reads: AtomicU64,
    page_ticks: AtomicU64,
    shared_rejected: AtomicU64,
    lookups: AtomicU64,
    lookup_ticks: AtomicU64,
    decode_ticks: AtomicU64,
    /// Proof batches run during the boot, and their TSC ticks.
    batches: AtomicU64,
    batch_ticks: AtomicU64,
}

static COSTS: Costs = Costs {
    page_reads: AtomicU64::new(0),
    page_ticks: AtomicU64::new(0),
    shared_rejected: AtomicU64::new(0),
    lookups: AtomicU64::new(0),
    lookup_ticks: AtomicU64::new(0),
    decode_ticks: AtomicU64::new(0),
    batches: AtomicU64::new(0),
    batch_ticks: AtomicU64::new(0),
};

/// How many blocks a shard accumulates before waking its proof worker.
const PROOF_BATCH_BLOCKS: usize = 32;
/// With no spare AP, space batches so that EL0 proof uses only a fraction
/// of a guest PE and never runs while it holds an XNU kernel lock.
const PROOF_SPACING: u64 = 4;
/// Counter ticks after a look that found nothing to prove before the next:
/// about a millisecond at the clocks Stela runs on.
const IDLE_LOOK_TICKS: u64 = 1 << 22;
static PROVING: AtomicBool = AtomicBool::new(false);
static NEXT_PROOF: AtomicU64 = AtomicU64::new(0);
static PROOF_PART: AtomicUsize = AtomicUsize::new(0);
static PROOF_WORKERS: [ProofWorker; PARTS] = [const { ProofWorker::new() }; PARTS];
static PROOF_WORKER_COUNT: AtomicUsize = AtomicUsize::new(0);
/// The most a stop spends proving what the boot left: the rest is not kept.
const STOP_PROOF_MILLIS: u64 = 3000;

/// One idle AP owns proof of its assigned shards. The bitmap coalesces
/// their offers without copying guest state, queuing unbounded jobs or
/// taking a lock on the sender's path to the AP.
#[repr(align(64))]
struct ProofWorker {
    apic: AtomicU32,
    pending: AtomicU64,
    active: AtomicBool,
    batches: AtomicU64,
}

impl ProofWorker {
    const fn new() -> Self {
        Self {
            apic: AtomicU32::new(0),
            pending: AtomicU64::new(0),
            active: AtomicBool::new(false),
            batches: AtomicU64::new(0),
        }
    }

    fn schedule(&self, shard: usize) {
        if self.pending.fetch_or(1 << shard, Ordering::Release) == 0 {
            host::ring(self.apic.load(Ordering::Relaxed));
        }
    }

    fn run(&self) {
        loop {
            crate::native_mmio::shootdown::service();
            let mut pending = self.pending.swap(0, Ordering::Acquire);
            while pending != 0 {
                let shard = pending.trailing_zeros() as usize;
                pending &= pending - 1;
                self.active.store(true, Ordering::SeqCst);
                // Paired with write's capability close and active scan.
                // A producer may leave a signal after closing, but it must
                // not start proof with the stores already taken by write.
                let writable = AVAILABLE.capabilities.load(Ordering::SeqCst) & WRITE_PACK != 0;
                if writable && !crate::ALLOCATOR.low() {
                    let lock = &STORES[shard];
                    let took = prove_batch(lock, true);
                    if took {
                        self.batches.fetch_add(1, Ordering::Relaxed);
                        if lock.with(|store| {
                            store
                                .as_ref()
                                .is_some_and(|store| store.proof_ready(PROOF_BATCH_BLOCKS))
                        }) {
                            self.pending.fetch_or(1 << shard, Ordering::Release);
                        }
                    }
                }
                self.active.store(false, Ordering::SeqCst);
            }
            if self.pending.load(Ordering::Acquire) == 0 {
                // SAFETY: this AP enters with interrupts masked. A directed
                // interrupt signaled before the wait is pending and wakes it;
                // the pinned AP IDT and shootdown state remain installed.
                unsafe { crate::native_idle::wait(|| None) };
            }
        }
    }
}

/// Reserve only workers that have no guest PE binding. A read-only or absent
/// pack has no durable proof to do, so its extra APs remain in their mailbox.
pub(super) fn start_proof_workers(
    executors: &Executors,
    bindings: &[WorkerBinding],
    sender: &Apic,
    tsc_hz: u64,
) -> usize {
    if !available(WRITE_PACK) {
        return 0;
    }
    let mut count = 0;
    for processor in executors.processors() {
        if bindings
            .iter()
            .any(|binding| binding.worker.0 == processor.slot())
        {
            continue;
        }
        let worker = &PROOF_WORKERS[count];
        let apic = u32::try_from(processor.apic_id()).expect("validated APIC ID");
        worker.apic.store(apic, Ordering::Relaxed);
        match processor.spawn(Box::new(move |_| worker.run()), sender, tsc_hz) {
            Ok(()) => count += 1,
            Err(error) => output::arguments(format_args!(
                "TRANSLATIONS APIC {apic:#x} proof worker unavailable: {error}\n"
            )),
        }
    }
    PROOF_WORKER_COUNT.store(count, Ordering::Release);
    count
}

/// Feed one owned relocation-proof batch to each eligible AP, using an
/// authored AArch64 source page and the actual offer/wake/fold path. The
/// diagnostic terminates here without a guest or a disk write.
#[cfg(feature = "diagnostics")]
pub(super) fn check_proof_workers(executors: &Executors, sender: &Apic, tsc_hz: u64) {
    use alloc::sync::Arc;
    use stela_compiler::{
        TargetCpu,
        x86_64::{MemoryChecks, NativeSystem},
    };
    use stela_engine::machine::engine::{Description, Preparation, publication::WorkerId};

    let logical = executors.started();
    if logical == 1 {
        output::arguments(format_args!(
            "CHECK native-translation-proof SKIP reason=no-remote-processor\n"
        ));
        return;
    }
    let guest_reserved = if logical == PARTS { 8 } else { 1 };
    let bindings: alloc::vec::Vec<_> = (0..guest_reserved)
        .map(|pe| WorkerBinding {
            pe,
            worker: WorkerId(pe),
        })
        .collect();
    SHARDS.store(PARTS, Ordering::Relaxed);
    for lock in stores() {
        drop(lock.with(Option::take));
    }
    install_parts(translations::translator_id(), None, 0);
    AVAILABLE.capabilities.store(WRITE_PACK, Ordering::SeqCst);
    let workers = start_proof_workers(executors, &bindings, sender, tsc_hz);
    if workers != logical - guest_reserved {
        output::fatal("translation proof diagnostic could not start every unassigned AP");
    }

    let mut keys = [None; PARTS];
    let mut source = vec![0; translations::PAGE_BYTES];
    let block = [0x00, 0x04, 0x00, 0x91, 0xc0, 0x03, 0x5f, 0xd6]; // add x0, x0, #1; ret
    for bytes in source
        .chunks_exact_mut(block.len())
        .take(PROOF_BATCH_BLOCKS)
    {
        bytes.copy_from_slice(&block);
    }
    let mut found = 0;
    let salt_range = source.len() - 8..;
    for salt in 0..4096u64 {
        source[salt_range.clone()].copy_from_slice(&salt.to_le_bytes());
        let key = Digest::of(&source);
        let shard = part_index(&key);
        if shard < workers && keys[shard].replace(key).is_none() {
            found += 1;
            if found == workers {
                break;
            }
        }
    }
    let preparation = Arc::new(
        Preparation::from_description(&Description {
            chip: stela_contracts::chip::Chip::AppleM1,
            target: TargetCpu::Baseline,
            fp: true,
            checks: MemoryChecks::DEFAULT,
            scalable: None,
            system: NativeSystem::NONE,
            withheld: 0,
        })
        .unwrap_or_else(|error| output::fatal(error)),
    );
    let mut shared = Shared;
    let started = output::now();
    for (shard, key) in keys[..workers].iter().enumerate() {
        let key =
            key.unwrap_or_else(|| output::fatal("translation proof diagnostic page key absent"));
        let page = (shard + 1) as u64 * translations::PAGE_BYTES as u64;
        for index in 0..PROOF_BATCH_BLOCKS {
            let pc = page + (index * block.len()) as u64;
            let offer = Offer::compile(
                key,
                page,
                pc,
                preparation.context(page),
                block.to_vec(),
                Arc::clone(&preparation),
            )
            .unwrap_or_else(|| output::fatal("translation proof diagnostic block refused"));
            shared.offer(offer);
        }
    }
    while PROOF_WORKERS[..workers]
        .iter()
        .any(|worker| worker.batches.load(Ordering::Relaxed) == 0)
    {
        if output::now().wrapping_sub(started) >= tsc_hz * 30 {
            output::fatal("translation proof diagnostic worker deadline");
        }
        crate::native_mmio::shootdown::service();
        core::hint::spin_loop();
    }
    finish_proofs();
    let (proven, unkept) = stores()
        .iter()
        .map(|lock| {
            lock.with(|store| {
                store.as_ref().map_or((0, 0), |store| {
                    (store.counters.proven, store.counters.unkept)
                })
            })
        })
        .fold((0, 0), |(proven, unkept), (add, lost)| {
            (proven + add, unkept + lost)
        });
    if proven != (workers * PROOF_BATCH_BLOCKS) as u64 || unkept != 0 {
        output::fatal("translation proof diagnostic entries missing");
    }
    output::arguments(format_args!(
        "CHECK native-translation-proof PASS logical={logical} guest_reserved={guest_reserved} \
         workers={workers} completed={workers} proven={proven} ms={}\n",
        output::millis_since(started)
    ));
}

/// What the store keeps beyond its pack, for an engine heap of `heap` bytes:
/// blocks waiting for proof, entries proven during the boot, decoded
/// records, and the fetch pages the machines keep (a page and its bookkeeping
/// take about 4.5 KiB).
fn bounds(heap: usize) -> (pack::Bounds, usize) {
    const MIB: usize = 1 << 20;
    let defaults = pack::Bounds::default();
    let bounds = pack::Bounds {
        pending: (heap / 64).clamp(defaults.pending, 256 * MIB),
        proven: (heap / 32).clamp(defaults.proven, 512 * MIB),
        decoded: (heap / 64).clamp(defaults.decoded, 128 * MIB),
    };
    (bounds, (heap / 64 / (4608)).max(4096))
}

/// Make `parts` the store's, over the pack read if any, sharing up to
/// `shared` bytes of blocks across PEs.
fn install_parts(translator: Digest, pack: Option<Pack<'static>>, shared: usize) {
    let (bounds, kept_pages) = bounds(crate::ALLOCATOR.usage().total);
    AVAILABLE.kept_pages.store(kept_pages, Ordering::Release);
    let stores = stores();
    for lock in stores {
        let evicted = lock.with(|store| {
            let store = store.get_or_insert_with(|| Store::over(translator, pack));
            store.share(shared / stores.len());
            store.set_bounds(bounds.share(stores.len()))
        });
        drop(evicted);
    }
}

/// The engine's view of the boot's store, from any PE's machine.
pub struct Shared;

impl Shared {
    fn publish_shared(offer: Offer) {
        let parts = SHARDS.load(Ordering::Relaxed);
        let bound = AVAILABLE.shared_bound.load(Ordering::Acquire) / parts;
        let lock = part(&offer.key());
        let Some(mut staged) = StagedOffer::new(offer, bound) else {
            return;
        };
        let retired = lock.with(|store| store.as_mut()?.accept_shared(&mut staged));
        drop(retired);
    }

    fn publish_writable(offer: Offer) {
        let parts = SHARDS.load(Ordering::Relaxed);
        let bound = AVAILABLE.shared_bound.load(Ordering::Acquire) / parts;
        let mut staged = StagedOffer::for_writable(&offer, bound);
        let workers = PROOF_WORKER_COUNT.load(Ordering::Acquire);
        let shard = part_index(&offer.key());
        let (retired, ready) = STORES[shard].with(|store| {
            let Some(store) = store.as_mut() else {
                return (None, false);
            };
            store.keep_pending(&offer);
            (
                staged
                    .as_mut()
                    .and_then(|staged| store.accept_shared(staged)),
                workers != 0 && store.proof_ready(PROOF_BATCH_BLOCKS),
            )
        });
        drop(retired);
        if ready {
            PROOF_WORKERS[shard % workers].schedule(shard);
        } else if workers == 0 && offer.quiet() {
            prove_some();
        }
    }
}

/// A proof only uses its owned input and immutable preparation. Holding the
/// shard lock for take and fold only leaves guest fetches free to use it.
fn prove_batch(lock: &Lock<Option<Store<'static>>>, on_worker: bool) -> bool {
    let started = output::now();
    let Some(batch) = lock.with(|store| store.as_mut()?.take_batch(PROOF_BATCH_BLOCKS)) else {
        return false;
    };
    let proven = if on_worker {
        // SAFETY: the AP is outside guest execution, holds no shard/driver
        // lock and has interrupts disabled. Its shootdown and doorbell do
        // not access the compiler. The previous interrupt mask is restored.
        unsafe { crate::native_mmio::shootdown::job(|| batch.prove()) }
    } else {
        batch.prove()
    };
    lock.with(|store| {
        if let Some(store) = store {
            store.fold(proven);
        }
    });
    COSTS.batches.fetch_add(1, Ordering::Relaxed);
    COSTS
        .batch_ticks
        .fetch_add(output::now().wrapping_sub(started), Ordering::Relaxed);
    true
}

/// Prove one batch of the blocks waiting for proof outside the store's
/// locks, when no other PE does and the last batch is long enough past.
fn prove_some() {
    let now = output::now();
    if now < NEXT_PROOF.load(Ordering::Relaxed) || PROVING.swap(true, Ordering::SeqCst) {
        return;
    }
    // A stop clears the capability before it waits for `PROVING`.
    let stores = stores();
    let proved = AVAILABLE.capabilities.load(Ordering::SeqCst) & WRITE_PACK != 0
        && !crate::ALLOCATOR.low()
        && (0..stores.len()).any(|_| {
            let lock = &stores[PROOF_PART.fetch_add(1, Ordering::Relaxed) & (stores.len() - 1)];
            prove_batch(lock, false)
        });
    let spent = output::now().wrapping_sub(now);
    // Only batches proven count: a look that found every part empty or
    // full is not one, and the next look waits a while, so that EL0
    // compiles do not each look through every part.
    let wait = if proved {
        spent.saturating_mul(PROOF_SPACING)
    } else {
        IDLE_LOOK_TICKS
    };
    NEXT_PROOF.store(now.wrapping_add(wait), Ordering::Relaxed);
    PROVING.store(false, Ordering::SeqCst);
}

impl Translations for Shared {
    /// The pack bytes outlive the lock, so other PEs can reach the store
    /// while a record is decoded. A writer also tracks absent keys.
    fn record(&mut self, key: &Digest) -> Option<alloc::sync::Arc<Record>> {
        if !available(READ_PACK | WRITE_PACK) {
            return None;
        }
        let started = output::now();
        COSTS.lookups.fetch_add(1, Ordering::Relaxed);
        let lock = part(key);
        let lookup = lock.with(|store| {
            let store = store.as_mut()?;
            if let Some(record) = store.cached_record(key) {
                return Some(Ok(record));
            }
            match store.stored(key) {
                Some(stored) => Some(Err((
                    stored,
                    *store.translator(),
                    store.take_decode_scratch(stored),
                ))),
                None => {
                    store.looked_up(key, false, false);
                    None
                }
            }
        });
        let (stored, translator, mut scratch) = match lookup {
            Some(Ok(record)) => {
                COSTS
                    .lookup_ticks
                    .fetch_add(output::now().wrapping_sub(started), Ordering::Relaxed);
                return Some(record);
            }
            Some(Err(source)) => source,
            None => {
                COSTS
                    .lookup_ticks
                    .fetch_add(output::now().wrapping_sub(started), Ordering::Relaxed);
                return None;
            }
        };
        let decoding = output::now();
        let decoded = DecodedRecord::decode(stored, &translator, key, &mut scratch);
        COSTS
            .decode_ticks
            .fetch_add(output::now().wrapping_sub(decoding), Ordering::Relaxed);
        let evicted = lock.with(|store| {
            let store = store.as_mut()?;
            Some(store.finish_decode(key, stored, &decoded, &mut scratch))
        });
        drop(evicted);
        COSTS
            .lookup_ticks
            .fetch_add(output::now().wrapping_sub(started), Ordering::Relaxed);
        decoded.ok().map(DecodedRecord::into_record)
    }

    fn prepared(
        &mut self,
        key: &Digest,
        pc: u64,
        context: &Digest,
    ) -> Option<alloc::sync::Arc<stela_engine::machine::engine::SharedTranslation>> {
        if !available(SHARE_BLOCKS) {
            return None;
        }
        part(key).with(|store| store.as_mut()?.prepared(key, pc, context))
    }

    fn offer(&mut self, offer: Offer) {
        let capabilities = AVAILABLE.capabilities.load(Ordering::Acquire);
        if capabilities & WRITE_PACK != 0 {
            Self::publish_writable(offer);
        } else if capabilities & SHARE_BLOCKS != 0 {
            Self::publish_shared(offer);
        }
    }

    fn lend(&mut self, offer: Offer) {
        if available(SHARE_BLOCKS) {
            Self::publish_shared(offer);
        }
    }

    fn note(&mut self, event: StoreEvent) {
        match event {
            StoreEvent::PageRead { ticks } => {
                COSTS.page_reads.fetch_add(1, Ordering::Relaxed);
                COSTS.page_ticks.fetch_add(ticks, Ordering::Relaxed);
            }
            StoreEvent::SharedRejected => {
                COSTS.shared_rejected.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn kept_pages(&mut self) -> Option<usize> {
        match AVAILABLE.kept_pages.load(Ordering::Acquire) {
            0 => None,
            pages => Some(pages),
        }
    }
}

/// The store the engine uses, if the loader read one or the PEs share one.
pub fn store() -> Option<&'static mut Shared> {
    available(READ_PACK | WRITE_PACK | SHARE_BLOCKS).then(|| Box::leak(Box::new(Shared)))
}

/// Share compiled blocks between PEs, keeping up to `bound` bytes of buffers
/// and cache metadata in the loader's store or a store for this boot. Allocator
/// overhead and shared immutable preparation storage are outside this budget.
pub fn share(bound: usize) {
    BOOT.with(|boot| {
        let translator = boot
            .get_or_insert_with(|| Boot {
                translator: translations::translator_id(),
                bound: 0,
                path: String::new(),
                partition: None,
            })
            .translator;
        install_parts(translator, None, bound);
        AVAILABLE.shared_bound.store(bound, Ordering::Release);
        if bound != 0 {
            AVAILABLE
                .capabilities
                .fetch_or(SHARE_BLOCKS, Ordering::Release);
        } else {
            AVAILABLE
                .capabilities
                .fetch_and(!SHARE_BLOCKS, Ordering::Release);
        }
    });
}

/// Make the store of the handoff file `contents` the engine's: the GPT
/// partition to open for writing it, when it can be written.
pub fn install(contents: &'static [u8], logical: usize) -> Option<Guid> {
    if contents.len() < PLACEMENT_BYTES || contents[..8] != PLACEMENT_MAGIC {
        return None;
    }
    let field = |at: usize| u64::from_le_bytes(contents[at..at + 8].try_into().expect("a field"));
    let bound = field(8);
    let flags = u32::from_le_bytes(contents[32..36].try_into().expect("a word"));
    let layout = if flags & NORTHWOOD_LAYOUT == 0 {
        Layout::Legacy
    } else {
        Layout::Northwood
    };
    let pack_bytes = usize::try_from(field(40)).ok()?;
    let pack = contents.get(PLACEMENT_BYTES..PLACEMENT_BYTES + pack_bytes)?;
    let translator = translations::translator_id();
    let parsed = (!pack.is_empty()).then(|| Pack::parse(pack, &translator));
    let pack = match parsed {
        Some(Ok(pack)) => {
            output::arguments(format_args!(
                "TRANSLATIONS pack records={} bytes={}\n",
                pack.records(),
                pack.bytes()
            ));
            Some(pack)
        }
        Some(Err(error)) => {
            output::arguments(format_args!("TRANSLATIONS pack discarded: {error:?}\n"));
            None
        }
        None => {
            output::arguments(format_args!("TRANSLATIONS no pack yet\n"));
            None
        }
    };
    BOOT.with(|boot| {
        for lock in stores() {
            let old = lock.with(Option::take);
            drop(old);
        }
        let partitions = if flags & WRITABLE != 0 {
            logical.clamp(BASE_PARTS, PARTS).next_power_of_two()
        } else {
            BASE_PARTS
        };
        SHARDS.store(partitions, Ordering::Relaxed);
        install_parts(translator, pack, 0);
        *boot = Some(Boot {
            translator,
            bound,
            path: store_path(layout, &pack_name(&translator)),
            partition: None,
        });
        let readable = if pack.is_some() { READ_PACK } else { 0 };
        AVAILABLE.shared_bound.store(0, Ordering::Release);
        AVAILABLE.capabilities.store(readable, Ordering::Release);
    });
    let partition = Guid(contents[16..32].try_into().expect("a GUID"));
    (flags & WRITABLE != 0).then_some(partition)
}

/// Keep `partition`, the store's volume, for [`write`].
pub fn set_partition(partition: Box<dyn BlockDevice>) {
    BOOT.with(|boot| {
        if let Some(boot) = boot {
            boot.partition = Some(partition);
            AVAILABLE
                .capabilities
                .fetch_or(WRITE_PACK, Ordering::Release);
        }
    });
}

/// Run `f` on the store's partition with the device extents of `name`, a
/// file beside the pack, for a writer that rewrites it in place as [`write`]
/// does the pack. Only while the engine may write the store.
#[cfg(feature = "nvk")]
pub fn with_store_file<R>(
    name: &str,
    f: impl FnOnce(&mut dyn BlockDevice, &[Extent]) -> R,
) -> Result<R, String> {
    BOOT.with(|boot| {
        let boot = boot.as_mut().ok_or("no translation store")?;
        let partition = boot
            .partition
            .take()
            .ok_or("the store's partition is not open")?;
        let mut volume = Fat32::open(Bytes::new(partition)).map_err(|error| format!("{error}"))?;
        let directory = boot
            .path
            .rsplit_once('/')
            .map_or("", |(directory, _)| directory);
        let file = volume.file(&format!("{directory}/{name}"));
        let mut device = volume.into_inner().into_inner();
        let result = file
            .map(|file| f(&mut *device, &file.extents))
            .map_err(|error| format!("{error}"));
        boot.partition = Some(device);
        result
    })
}

/// Milliseconds of `ticks` TSC ticks.
fn millis(ticks: u64) -> u64 {
    output::millis_since(output::now().wrapping_sub(ticks))
}

/// Report what the store did and cost this boot: lookups and their causes,
/// shared misses by reason, proof during the boot and lock waits.
fn report(counters: &pack::Counters) {
    let costs = &COSTS;
    let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
    output::arguments(format_args!(
        "TRANSLATIONS lookups={} repeated={} cached={} decoded={} lookup_ms={} decode_ms={} \
         pages={} page_ms={}\n",
        load(&costs.lookups),
        counters.repeated,
        counters.cached,
        counters.decoded,
        millis(load(&costs.lookup_ticks)),
        millis(load(&costs.decode_ticks)),
        load(&costs.page_reads),
        load(&costs.page_ticks) / (super::host::COUNTER_HZ / 1000),
    ));
    output::arguments(format_args!(
        "TRANSLATIONS shared hits={} rejected={} absent={} other_context={} evicted={} \
         in_flight={}\n",
        counters.shared_hits,
        load(&costs.shared_rejected),
        counters.shared_absent,
        counters.shared_other_context,
        counters.shared_evicted,
        counters.shared_in_flight,
    ));
    let (mut acquired, mut contended, mut spun) = (0, 0, 0);
    for lock in stores() {
        acquired += load(&lock.acquired);
        contended += load(&lock.contended);
        spun += load(&lock.spun);
    }
    let workers = &PROOF_WORKERS[..PROOF_WORKER_COUNT.load(Ordering::Acquire)];
    let busy = workers
        .iter()
        .filter(|worker| load(&worker.batches) != 0)
        .count();
    let offloaded = workers
        .iter()
        .map(|worker| load(&worker.batches))
        .sum::<u64>();
    output::arguments(format_args!(
        "TRANSLATIONS early proven={} unprovable={} unkept={} batches={} batch_ms={} \
         proof_workers={} working={busy} offloaded={offloaded} \
         locks acquired={acquired} contended={contended} spin_ms={}\n",
        counters.proven,
        counters.unprovable,
        counters.unkept,
        load(&costs.batches),
        millis(load(&costs.batch_ticks)),
        workers.len(),
        millis(spun),
    ));
}

fn finish_proofs() {
    AVAILABLE
        .capabilities
        .fetch_and(!WRITE_PACK, Ordering::SeqCst);
    // Each owner folds its batch before the stores can be taken. A late
    // producer can leave a signal, but cannot start another proof.
    while PROVING.load(Ordering::SeqCst) {
        core::hint::spin_loop();
    }
    for worker in &PROOF_WORKERS[..PROOF_WORKER_COUNT.load(Ordering::Acquire)] {
        while worker.active.load(Ordering::SeqCst) {
            crate::native_mmio::shootdown::service();
            core::hint::spin_loop();
        }
    }
}

/// At a quiescent point (the guest restarts or powers off): prove the
/// blocks compiled anew and rewrite the pack in place.
pub fn write() {
    let Some(boot) = BOOT.with(|boot| {
        AVAILABLE.capabilities.store(0, Ordering::SeqCst);
        AVAILABLE.shared_bound.store(0, Ordering::Release);
        boot.take()
    }) else {
        return;
    };
    finish_proofs();
    let mut parts = stores().iter().filter_map(|lock| lock.with(Option::take));
    let Some(mut store) = parts.next() else {
        return;
    };
    for part in parts {
        store.absorb(part);
    }
    let counters = store.counters;
    report(&counters);
    let Boot {
        bound,
        path,
        partition: Some(partition),
        ..
    } = boot
    else {
        return;
    };
    output::arguments(format_args!(
        "TRANSLATIONS proving {} blocks (loaded={} absent={} invalid={} dropped={})\n",
        store.pending(),
        counters.loaded,
        counters.absent,
        counters.invalid,
        counters.dropped
    ));
    // Proof recompiles the original and both relocation probes; stop while
    // the heap still has room for the layout and the write, and before the
    // stop waits long.
    let started = output::now();
    let planned = store.plan(bound, || {
        !crate::ALLOCATOR.low() && output::millis_since(started) < STOP_PROOF_MILLIS
    });
    let heap = crate::ALLOCATOR.usage();
    let written: u64 = match planned.layout.changed() {
        None => planned.layout.bytes(),
        Some(ranges) => ranges.iter().map(|range| range.end - range.start).sum(),
    };
    output::arguments(format_args!(
        "TRANSLATIONS planned records={} bytes={} changed={written} ms={} live={} carved={} \
         total={}\n",
        planned.layout.records(),
        planned.layout.bytes(),
        output::millis_since(started),
        heap.live,
        heap.carved,
        heap.total
    ));
    let started = output::now();
    let mut volume = match Fat32::open(Bytes::new(partition)) {
        Ok(volume) => volume,
        Err(error) => {
            output::arguments(format_args!("TRANSLATIONS not written: {error}\n"));
            return;
        }
    };
    let file = match volume.file(&path) {
        Ok(file) => file,
        Err(error) => {
            output::arguments(format_args!("TRANSLATIONS not written: {error}\n"));
            return;
        }
    };
    let mut device = volume.into_inner().into_inner();
    output::arguments(format_args!(
        "TRANSLATIONS writing extents={}\n",
        file.extents.len()
    ));
    match stela_runtime::translation_store::write_in_place(
        &mut *device,
        &file.extents,
        &planned.layout,
    ) {
        Ok(()) => output::arguments(format_args!(
            "TRANSLATIONS written records={} bytes={} changed={written} proven={} unprovable={} \
             unproven={} evicted={} ms={}\n",
            planned.layout.records(),
            planned.layout.bytes(),
            planned.proven,
            planned.unprovable,
            planned.unproven,
            planned.evicted,
            output::millis_since(started)
        )),
        Err(error) => output::arguments(format_args!("TRANSLATIONS not written: {error:?}\n")),
    }
}
