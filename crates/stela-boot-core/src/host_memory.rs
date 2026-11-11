//! The parts of Stela's memory sized from the host RAM free at boot (the
//! engine heap, native code arena and shadow page-table pool), the share
//! left for the guest, and what is left of it for the translation pack.
//!
//! The standard profile gives the heap its measured
//! desktop floor ([`HEAP_FLOOR`]) and the guest at least [`GUEST_FLOOR`]; a
//! PC of about 7 GiB free and up boots with it. The compact one is for
//! smaller PCs, a 4 GB one above all: the guest gets at least
//! [`GUEST_COMPACT_FLOOR`], the heap at least [`HEAP_COMPACT_FLOOR`] beside
//! its block records, and what is left is split evenly between them. A
//! compact boot runs slower: its code window and shadow pools are small, it
//! reads no translation pack, and macOS pages more with less DRAM. The paged
//! profile also reduces the heap and limits native workers on tighter hosts.
//! For now its guest DRAM is entirely resident, like the other profiles.
use crate::Result;
use crate::native_resources::{
    COMPACT_WINDOW_PER_RUNNER, CodeReservation, SMALL_WINDOW_PER_RUNNER, WINDOW_PER_RUNNER,
    code_reservation_per_runner, parked_reserve,
};
use alloc::string::String;
use stela_contracts::refuse;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
/// Each part's alignment: the large page the engine maps it with.
const ALIGNMENT: u64 = 2 * MIB;
/// What the heap needs does not shrink with the RAM, which bounds it from
/// below: a 26A434 desktop boot holds about 1.5 GiB live when WindowServer
/// compiles Core Animation's generic Vfx shader (`fixed_frag_lph_cpf` with
/// `fc_generic`, 42 529 NAK instructions in about 4800 blocks), whose
/// per-block liveness takes NAK another 450 MiB for the compile. That
/// peaked at 2150 MiB, and a boot with a fuller translation pack ran out of
/// a 2304 MiB heap there.
pub const HEAP_FLOOR: u64 = 2816 * MIB;
/// Above the floor, the heap grows with the host to this: a larger host
/// runs more and larger pipelines, apps and translation packs.
pub const HEAP_CEILING: u64 = 16 * GIB;
/// Free host RAM bytes per byte of the engine heap's base share. A 16 GiB
/// guest's first Metal shader compiles carved all of a sixteenth (1816 MiB)
/// and failed.
const HEAP_SHARE: u64 = 12;
/// Heap bytes per byte of the board's code window: the shared block records,
/// successors, recovery rows, index and lookup of the blocks it holds,
/// once for the board since the block store is shared. An 8-PE 26A434
/// boot with a full 448 MiB window (1.56 million of 1.69 million slots
/// live) held 437 MiB of them (`ENGINE MEMORY blocks=231M successors=53M
/// rows=14M index=19M lookup=120M`), 0.98 bytes per window byte. Before
/// the shared store each core kept its own, 2.9 bytes per byte.
const HEAP_PER_WINDOW: u64 = 1;
/// The code window [`HEAP_FLOOR`] was measured with. The floor moves by
/// [`HEAP_PER_WINDOW`] per byte the window differs from it: its block
/// records come on top of the rest of the peak, which does not depend on
/// the window. A 12 GiB VM with a 704 MiB window peaked at 2815 MiB carved,
/// 2623 MiB less the 192 MiB of window past this one.
const FLOOR_WINDOW: u64 = 512 * MIB;
/// Heap each running PE keeps for itself: its machine (about 1 MiB at
/// start), ITLB (512 KiB), fetch and jump caches (about 0.8 MiB) and
/// successor scratch (under 1 MiB); four times that, for its own compiles.
const PE_HEAP: u64 = 16 * MIB;
/// Guest DRAM bytes per byte of a running executor's shadow page-table
/// pool: room for its 8 roots to map all of guest DRAM once in 4 KiB pages
/// (8 bytes per page). An exhausted half resets and refills everything; a
/// 26A434 restore's userland filled 8 MiB halves 20 to 80 times a beat.
const POOL_SHARE: u64 = 64;
/// Runners whose pools together map the guest's DRAM once each. Past them
/// the runners divide that total: each PE's tables are a cache of what it
/// touches, and more PEs divide the guest's working set between them.
const POOL_SHARERS: u64 = 8;
const POOL_FLOOR: u64 = 64 * MIB;
/// The least pool a running executor gets, a 16 MiB half, at any PE count.
const POOL_RUNNER_FLOOR: u64 = 32 * MIB;
/// Each sharer's pool maps at most 64 GiB of guest DRAM once; past that its
/// halves reset as a cache of the guest's working set, rather than hold a
/// tenth of a large host's RAM.
const POOL_PER_RUNNER: u64 = GIB;
/// Past small guests a runner's pool grows by a byte per [`POOL_GROWTH`]
/// bytes of guest DRAM above [`POOL_BASE`], not by a sixty-fourth. Its upper
/// half holds the kernel's mappings, which grow with the DRAM it touches:
/// PE0's peaked at 23 MiB of a 52 MiB half with 6.7 GiB of guest DRAM
/// (`BEAT shadow_peak=52M/23M/52M`), about a 290th. Its lower half is a
/// cache of recent address spaces: it fills whatever it has (52 MiB there),
/// and ran out 74 times in a 20-minute run that reset its shadow 33 000
/// times for TLB maintenance (`SHADOW exhausted= maintained=`). Each half,
/// 32 MiB and a 320th of the guest, holds more than both measured peaks on
/// every host from 12 GiB (whose pools lose 6 MiB of 864) up; below that
/// the sixty-fourth is less.
const POOL_BASE: u64 = 64 * MIB;
const POOL_GROWTH: u64 = 160;
/// The least guest DRAM the standard profile boots macOS with. Below it
/// macOS spends its time paging; 26A434 reaches the desktop with 3.1 GiB.
pub const GUEST_FLOOR: u64 = 3 * GIB;
/// The least guest DRAM a compact boot gives macOS: what the restore and a
/// desktop need to run at all, paging as they go. A Mac mini (M2) ships
/// with no less than 8 GiB, so macOS sizes nothing for less than this;
/// Apple's own virtual Macs run from 2 GiB.
pub const GUEST_COMPACT_FLOOR: u64 = 2 * GIB;
/// The least engine heap a compact boot keeps beside its block records and
/// each PE's own [`PE_HEAP`]: the drivers, the pipelines a restore and a
/// desktop boot create, and one shader compile at a time. It is under half
/// the measured desktop peak, so a compact boot may run out of heap where a
/// standard one does not, at the largest shader compiles above all.
pub const HEAP_COMPACT_FLOOR: u64 = 1024 * MIB;
/// The least pool a running executor gets on a compact host: two 8 MiB
/// halves, which a 26A434 restore's userland filled 20 to 80 times a beat
/// and ran with.
const POOL_COMPACT_RUNNER_FLOOR: u64 = 16 * MIB;
/// The least usable host RAM at which a low-memory reservation is attempted.
pub const MIN_PHYSICAL: u64 = 2 * GIB;
/// The paged profile's provisional engine heap floor beside the code-window
/// records and each PE's own allocations.
const HEAP_PAGED_FLOOR: u64 = 512 * MIB;
/// The paged profile reserves up to four native workers; the other processors
/// stay parked and need no engine arenas.
const PAGED_MAX_CORES: usize = 4;
/// Allow for the loader's remaining stack, table and handoff allocations when
/// estimating resident guest DRAM. The final firmware map decides its size.
const PAGED_HANDOFF_RESERVE: u64 = 16 * MIB;
/// Free host RAM bytes per byte of the persistent translation pack the
/// loader may hold: a cache, it takes no more than this share and only what
/// the parts and the guest's floor leave.
pub const PACK_SHARE: u64 = 16;
/// The least pack worth reading. Below it the loader leaves the pack on its
/// volume, unread and unwritten, for a PC with more RAM.
pub const PACK_FLOOR: u64 = 64 * MIB;
/// What the loader takes between measuring the free RAM for the pack and
/// reserving the parts, beyond the pack itself: the pack's placement page,
/// the storage drivers' rings and the firmware ranges' rounding to 2 MiB.
const PACK_MARGIN: u64 = 16 * MIB;

/// Which split of the host RAM [`sizes`] made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// The heap at its measured desktop floor or more, the guest at
    /// [`GUEST_FLOOR`] or more.
    Standard,
    /// A small host: the guest at [`GUEST_COMPACT_FLOOR`] or more, the heap
    /// at [`HEAP_COMPACT_FLOOR`] or more, the rest split between them.
    Compact,
    /// The small heap and limited worker set intended for future host paging.
    /// The guest uses only its physical backing until a pager is available.
    Paged,
}

impl Profile {
    /// The profile's name, as the boot log prints it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Compact => "compact",
            Self::Paged => "paged",
        }
    }
}

/// The parts' sizes and the guest DRAM left after them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sizes {
    /// Native executors to reserve, which can be fewer than requested.
    pub cores: usize,
    pub heap: u64,
    pub code: CodeReservation,
    pub pool: u64,
    /// Estimated resident guest DRAM. The final map also accounts for the
    /// loader's stacks, tables and other handoff allocations.
    pub guest: u64,
    pub profile: Profile,
}

/// The DRAM the guest Mac reports and the host RAM it does not see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuestDram {
    /// `/chosen dram-size` and `boot_args.memSizeActual`.
    pub reported: u64,
    /// Installed host RAM that Stela and the firmware keep.
    pub kept: u64,
}

/// The guest reports the DRAM that backs it, `backed` bytes, not the
/// host's `installed` RAM.
///
/// XNU sets `max_mem_actual` from `memSizeActual` and `max_mem` and
/// `sane_size` from `memSize` (the managed range). `hw.memsize` reads
/// `max_mem_actual`; `hw.memsize_usable` reads `max_mem`. User space
/// (`NSProcessInfo.physicalMemory`, Metal's working-set size, jetsam
/// limits, cache sizes) uses `hw.memsize`. A reported size above what
/// exists makes it plan for memory that is not there: a 26A434 boot with
/// 3.9 GB backed and 8 GB reported ran its compressor and pager at full
/// load after login. On a Mac the gap is iBoot's carveouts, a few percent;
/// here those carveouts lie inside the backed range (`memSize` excludes
/// them), so the backed size is the truthful `max_mem_actual`.
pub fn guest_dram(installed: u64, backed: u64) -> GuestDram {
    GuestDram {
        reported: backed,
        kept: installed.saturating_sub(backed),
    }
}

/// `bytes` in GB to a tenth, as the boot's messages print RAM: "4.0".
pub fn gigabytes(bytes: u64) -> String {
    alloc::format!("{}.{}", bytes / GIB, bytes % GIB * 10 / GIB)
}

/// Too little usable RAM for the smallest profile: `free` bytes, of `needed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shortfall {
    pub free: u64,
    pub needed: u64,
}

impl Shortfall {
    /// The refusal the boot stops with, in whole GB rounded up, as RAM is
    /// sold, and the free RAM to a tenth.
    pub fn refusal(self) -> crate::Refusal {
        let needed = self.needed.div_ceil(GIB);
        let free = gigabytes(self.free);
        refuse!(
            "this PC has {free} GB of RAM free; Stela needs at least {needed} GB free to start the guest",
            free = %free,
            needed = %needed
        )
    }
}

/// The sizes for `free` host bytes and `runners` of the `cores` executors
/// running PEs; the others add only their parked reserves.
///
/// In the standard profile each part takes a share of `free`, between its
/// bounds:
/// - the heap a twelfth, [`HEAP_PER_WINDOW`] byte per byte of the board's
///   code window and [`PE_HEAP`] per runner, within
///   [`HEAP_FLOOR`]..=[`HEAP_CEILING`], the floor moved by the window's
///   difference from [`FLOOR_WINDOW`];
/// - the code arena what `code_reservation` gives: [`WINDOW_PER_RUNNER`]
///   per runner within free RAM / 16, or [`SMALL_WINDOW_PER_RUNNER`] where
///   the larger window would leave the guest under its floor;
/// - the pool, for each of up to [`POOL_SHARERS`] runners, 1/[`POOL_SHARE`]
///   of the guest's DRAM, which is what the other parts leave, or
///   [`POOL_BASE`] and 1/[`POOL_GROWTH`] of it if that is less: `pool =
///   sharers * (free - heap - code - pool) / 64`, solved for `pool`; at least
///   [`POOL_RUNNER_FLOOR`] per runner, at most [`POOL_PER_RUNNER`] per sharer.
///
/// A guest share under [`GUEST_FLOOR`] falls back to the [compact
/// profile](Profile::Compact). Smaller hosts use the [paged profile](Profile::Paged)
/// from [`MIN_PHYSICAL`] up, with fewer workers and a smaller heap as needed.
pub fn sizes(free: u64, cores: usize, runners: usize) -> Result<Sizes> {
    let estimate = estimate(free, cores, runners)?;
    if estimate.guest >= GUEST_FLOOR {
        return Ok(estimate);
    }
    if let Some(compact) = compact(free, cores, runners)? {
        return Ok(compact);
    }
    if free >= MIN_PHYSICAL {
        return paged(free, cores, runners, true);
    }
    Err(Shortfall {
        free,
        needed: MIN_PHYSICAL,
    }
    .refusal())
}

/// The least contiguous heap a paged boot can use. The loader retries with
/// this split when UEFI cannot allocate the preferred heap in one extent.
pub fn fragmented_sizes(free: u64, cores: usize, runners: usize) -> Result<Sizes> {
    if free < MIN_PHYSICAL {
        return Err(Shortfall {
            free,
            needed: MIN_PHYSICAL,
        }
        .refusal());
    }
    if runners > cores {
        return Err(refuse!("invalid native runner count"));
    }
    paged(free, cores, runners, false)
}

/// The bytes of persistent translation pack the loader may read for
/// `free` host bytes, measured just before it reads the pack, with
/// `runners` of `cores` executors running PEs: at most a
/// 1/[`PACK_SHARE`] of `free`, and only as much as still leaves the
/// standard profile the guest's floor, after the pack and a margin for the
/// loader's other allocations. Zero, below [`PACK_FLOOR`]: the parts and
/// the guest come first, and the pack never makes a boot fail. A compact
/// or paged host, where the standard profile never fits, reads no pack: its RAM is
/// the guest's and the heap's.
pub fn pack_budget(free: u64, cores: usize, runners: usize) -> u64 {
    let mut budget = free / PACK_SHARE / ALIGNMENT * ALIGNMENT;
    while budget >= PACK_FLOOR {
        let left = free.saturating_sub(budget + PACK_MARGIN);
        if estimate_guest(left, cores, runners).is_ok_and(|guest| guest >= GUEST_FLOOR) {
            return budget;
        }
        budget -= ALIGNMENT;
    }
    0
}

fn estimate_guest(free: u64, cores: usize, runners: usize) -> Result<u64> {
    estimate(free, cores, runners).map(|sizes| sizes.guest)
}

/// The standard profile's sizes with the larger code window per runner, or
/// with the smaller one where the larger would leave the guest under its
/// floor.
fn estimate(free: u64, cores: usize, runners: usize) -> Result<Sizes> {
    let large = estimate_with(free, cores, runners, WINDOW_PER_RUNNER)?;
    if large.guest >= GUEST_FLOOR {
        return Ok(large);
    }
    estimate_with(free, cores, runners, SMALL_WINDOW_PER_RUNNER)
}

fn estimate_with(free: u64, cores: usize, runners: usize, per_runner: u64) -> Result<Sizes> {
    let code = code_reservation_per_runner(free, cores, runners, per_runner)?;
    let running = runners as u64;
    let heap = free / HEAP_SHARE + HEAP_PER_WINDOW * code.window + running * PE_HEAP;
    let floor = (HEAP_FLOOR + HEAP_PER_WINDOW * code.window)
        .saturating_sub(HEAP_PER_WINDOW * FLOOR_WINDOW)
        / ALIGNMENT
        * ALIGNMENT;
    let heap = (heap / ALIGNMENT * ALIGNMENT).clamp(floor, HEAP_CEILING.max(floor));
    let rest = free.saturating_sub(heap).saturating_sub(code.bytes);
    let sharers = running.min(POOL_SHARERS);
    let pool = (pool_share(rest, running) / ALIGNMENT * ALIGNMENT).clamp(
        POOL_FLOOR.max(running * POOL_RUNNER_FLOOR),
        sharers * POOL_PER_RUNNER,
    );
    let (_, parked) = parked_reserve(cores, runners);
    let pool = pool + parked;
    Ok(Sizes {
        cores,
        heap,
        code,
        pool,
        guest: rest.saturating_sub(pool),
        profile: Profile::Standard,
    })
}

/// The compact profile's sizes, or `None` when `free` does not hold its
/// floors: the code arena at [`COMPACT_WINDOW_PER_RUNNER`] per runner, the
/// pool's and the heap's floors ([`POOL_COMPACT_RUNNER_FLOOR`] per runner;
/// [`HEAP_COMPACT_FLOOR`] with the window's block records and [`PE_HEAP`]
/// per runner) and the guest's [`GUEST_COMPACT_FLOOR`]. What is left above
/// them goes half to the heap and half to the guest's side, where the pool
/// takes its usual share of the guest, never more than that half, so the
/// guest keeps its floor.
fn compact(free: u64, cores: usize, runners: usize) -> Result<Option<Sizes>> {
    let code = code_reservation_per_runner(free, cores, runners, COMPACT_WINDOW_PER_RUNNER)?;
    let running = runners as u64;
    let (_, parked) = parked_reserve(cores, runners);
    let pool_floor = POOL_FLOOR.max(running * POOL_COMPACT_RUNNER_FLOOR);
    let heap_floor = (HEAP_COMPACT_FLOOR + HEAP_PER_WINDOW * code.window + running * PE_HEAP)
        .next_multiple_of(ALIGNMENT);
    let floors = code.bytes + pool_floor + parked + heap_floor + GUEST_COMPACT_FLOOR;
    let Some(surplus) = free.checked_sub(floors) else {
        return Ok(None);
    };
    let half = surplus / 2 / ALIGNMENT * ALIGNMENT;
    let heap = heap_floor + half;
    let rest = free - heap - code.bytes;
    let sharers = running.min(POOL_SHARERS);
    let cap = (sharers * POOL_PER_RUNNER)
        .min(pool_floor + half)
        .max(pool_floor);
    let pool = (pool_share(rest, running) / ALIGNMENT * ALIGNMENT).clamp(pool_floor, cap) + parked;
    Ok(Some(Sizes {
        cores,
        heap,
        code,
        pool,
        guest: rest - pool,
        profile: Profile::Compact,
    }))
}

fn paged(free: u64, cores: usize, runners: usize, grow_heap: bool) -> Result<Sizes> {
    let cores = cores.min(PAGED_MAX_CORES);
    let runners = runners.min(cores);
    let code = code_reservation_per_runner(free, cores, runners, COMPACT_WINDOW_PER_RUNNER)?;
    let running = runners as u64;
    let (_, parked) = parked_reserve(cores, runners);
    let pool = POOL_FLOOR.max(running * POOL_COMPACT_RUNNER_FLOOR) + parked;
    let heap_overhead = HEAP_PER_WINDOW * code.window + running * PE_HEAP;
    let available = free - code.bytes - pool - PAGED_HANDOFF_RESERVE;
    // Grow the heap once the guest has 2 GiB, up to the compact floor for
    // this worker count. Hosts that entered this profile because they
    // requested many workers can give the guest more of their surplus.
    let floor = HEAP_PAGED_FLOOR + heap_overhead;
    let heap = if grow_heap {
        (available.saturating_sub(GUEST_COMPACT_FLOOR) / ALIGNMENT * ALIGNMENT)
            .clamp(floor, HEAP_COMPACT_FLOOR + heap_overhead)
    } else {
        floor
    };
    Ok(Sizes {
        cores,
        heap,
        code,
        pool,
        guest: available - heap,
        profile: Profile::Paged,
    })
}

/// The running executors' pool for `rest` bytes of guest and pool together:
/// a sixty-fourth of the guest for each of up to [`POOL_SHARERS`] sharers,
/// or [`POOL_BASE`] and a [`POOL_GROWTH`]th of it if that is less.
fn pool_share(rest: u64, running: u64) -> u64 {
    let sharers = u128::from(running.min(POOL_SHARERS));
    let rest = u128::from(rest);
    let share = sharers * rest / (u128::from(POOL_SHARE) + sharers);
    let growth = u128::from(POOL_GROWTH);
    let grown = sharers * (growth * u128::from(POOL_BASE) + rest) / (growth + sharers);
    share.min(grown) as u64
}

#[cfg(test)]
mod tests;
