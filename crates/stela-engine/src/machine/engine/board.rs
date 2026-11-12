//! What the board's PEs share of compiled code, and the rules for taking
//! some of it away while other PEs may still hold it:
//!
//! - [`Grace`]: a word per PE that says whether it is inside a section that
//!   may hold what is being taken away, and how often it left one. What a
//!   PE unpublishes is reused only once every other PE was outside at that
//!   moment or has left since. No PE ever waits inside a section, so two
//!   PEs that wait never wait on each other.
//! - [`Ring`]: the code parts every PE installs into, a bump cursor for
//!   each of two generations (young and tenured code), each part chosen to
//!   go one ahead of its reuse.
//! - [`Slots`]: block slot numbers, reused through a limbo list once their
//!   grace passed; until then new numbers are given instead.
//! - [`Records`]: a record per slot that every PE reads without a lock.
//!
//! - [`Board`]: what `parallel::Hub` keeps of these for its PEs.
//!
//! - [`Code`]: the blocks, block index, block table and, when the host
//!   gives its whole code window, the one [`Arena`] of a board's machines,
//!   or of one machine alone.
//!
//! The grace is entered and left with the hub's run-loop flags and passed
//! once per run-loop iteration.

use super::{
    Block, CodePartitions, CodeWindow, Padded, VariantId,
    block_index::{BlockIndex, Finder},
    code_slots::CodeSlots,
    recovery::CodeRecovery,
};
use crate::platform::sync::{Mutex, MutexGuard};
use alloc::{boxed::Box, collections::VecDeque, sync::Arc, vec::Vec};
use core::cell::UnsafeCell;
use core::sync::atomic::{
    AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering, fence,
};
use stela_compiler::x86_64::chain::{self, LookupTable, TableReader};

mod ring;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(super) use ring::Why;
pub(super) use ring::{AGE_TURNS, Advance, MIN_PARTS, Place, Ring, TENURED, YOUNG};

/// One PE's word, on a cache line of its own: its PE writes it often.
#[repr(align(64))]
#[derive(Default)]
struct Word(AtomicU64);

/// Each PE's count of section entries and exits: odd while it is inside a
/// section, where it may hold and use what the guarded state names (code
/// addresses, slot numbers); even outside, where it holds none. A PE that
/// waits for a grace period first leaves its section, so it holds no other
/// waiter up.
pub(super) struct Grace {
    words: Box<[Word]>,
}

/// The PEs' words when something was unpublished ([`Grace::snapshot`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Snapshot(Box<[u64]>);

impl Grace {
    /// Words for `pes` PEs, every one outside.
    pub(super) fn new(pes: usize) -> Self {
        Self {
            words: (0..pes).map(|_| Word::default()).collect(),
        }
    }

    /// PE `pe` enters a section. A full barrier: its later reads of
    /// shared state come after a remover that saw it outside.
    pub(super) fn enter(&self, pe: usize) {
        let before = self.words[pe].0.fetch_add(1, Ordering::SeqCst);
        debug_assert!(before & 1 == 0, "a PE enters one section at a time");
    }

    /// PE `pe` leaves its section: what it read inside is no longer used.
    pub(super) fn leave(&self, pe: usize) {
        let before = self.words[pe].0.fetch_add(1, Ordering::Release);
        debug_assert!(before & 1 == 1, "a PE leaves the section it is in");
    }

    /// Leave and enter again at once: a point of a PE that stays inside
    /// where it holds nothing (its run loop's top).
    pub(super) fn pass(&self, pe: usize) {
        let before = self.words[pe].0.fetch_add(2, Ordering::SeqCst);
        debug_assert!(before & 1 == 1, "a PE passes a point of its section");
    }

    /// Whether PE `pe` is inside a section.
    pub(super) fn inside(&self, pe: usize) -> bool {
        self.words[pe].0.load(Ordering::Relaxed) & 1 != 0
    }

    /// The words now. Taken after the stores that unpublished something:
    /// a PE outside now finds it gone once it enters.
    pub(super) fn snapshot(&self) -> Snapshot {
        fence(Ordering::SeqCst);
        Snapshot(
            self.words
                .iter()
                .map(|word| word.0.load(Ordering::SeqCst))
                .collect(),
        )
    }

    /// Whether every PE but `skip` was outside at `snapshot` or has left
    /// since: what was unpublished before it is no longer used. `skip` is
    /// a caller inside a section that holds nothing it read before the
    /// snapshot without checking it again.
    pub(super) fn passed(&self, snapshot: &Snapshot, skip: Option<usize>) -> bool {
        self.words
            .iter()
            .zip(&snapshot.0)
            .enumerate()
            .all(|(pe, (word, &then))| {
                then & 1 == 0 || Some(pe) == skip || word.0.load(Ordering::Acquire) != then
            })
    }

    /// Wait outside its section, as PE `pe`, until `snapshot` has passed or
    /// `give_up` says to stop, then enter again: whether it passed. The
    /// caller checks what it held before again, as a remover may have
    /// taken it meanwhile.
    pub(super) fn wait(
        &self,
        pe: usize,
        snapshot: &Snapshot,
        mut give_up: impl FnMut() -> bool,
    ) -> bool {
        self.leave(pe);
        let passed = loop {
            if self.passed(snapshot, None) {
                break true;
            }
            if give_up() {
                break false;
            }
            core::hint::spin_loop();
        };
        self.enter(pe);
        passed
    }
}

/// What the board's PEs share of compiled code, kept by `parallel::Hub`.
/// Its one [`Grace`] covers code bytes and block slots alike: a PE is
/// inside while it is in its run loop and neither parked nor outside it
/// (`Hub::set_outside`, `Hub::set_parked`), and passes the loop's top once
/// per iteration, where it holds no code address or slot it does not check
/// again (the jump cache by incarnation, a held exit by its part).
pub(super) struct Board {
    pub(super) grace: Grace,
    /// A wait ran out once: reported then, not again.
    overdue: AtomicBool,
}

/// The compiled blocks a board's machines share, or a machine keeps alone
/// (a board of one PE): every PE runs any block any PE installed.
pub(super) struct Code {
    /// The blocks, which any PE reads without a lock.
    pub(super) blocks: Blocks,
    /// What changes with installs, links and evictions, under one lock.
    pub(super) cache: Mutex<Cache>,
    /// The block table's entries, for reads without the lock.
    pub(super) table: TableReader,
    /// The block index's blocks, for reads without the lock.
    pub(super) finder: Arc<Finder>,
    /// The code arena, when the engine allocates code itself.
    pub(super) arena: Option<Arena>,
    /// Advances when the board's machines forget every block at once
    /// (`Machine::forget_shared_code`): what jump cache entries name.
    /// On a line of its own, as every fetch reads it: which also puts the
    /// `Arc`'s counts, which clones write, on a line of their own.
    pub(super) epoch: Padded<AtomicU64>,
    pub(super) events: Events,
    /// The ghosts of blocks that left the index ([`super::churn`]): written
    /// under the lock, read by compiles without it.
    pub(super) ghosts: super::churn::Churn,
    /// The part being taken away, in steps any PE's install takes
    /// ([`Retirement`]), and whether there is one, read without its lock.
    pub(super) retiring: Mutex<Option<Retirement>>,
    pub(super) retire_pending: AtomicBool,
    /// Units of work a retirement step takes under the lock
    /// ([`super::code_cache::STEP_VISITS`]; tests take fewer).
    pub(super) step_visits: u64,
    /// The fetch pages its PEs compile in the tagged variant.
    pub(super) tagged: super::page_sets::PageSet,
    /// The fetch pages its PEs compile with the two-sided range check at
    /// EL0.
    pub(super) two_sided: super::page_sets::PageSet,
    /// Proven entries of blocks compiled at more than one address
    /// ([`super::rebase`]).
    pub(super) rebases: super::rebase::Rebases,
}

/// A part's blocks being taken away page by page, in steps of bounded
/// work, each step one hold of the code lock: the ring's
/// [`Advance::Retire`] leaves a whole part's installs before the part is
/// needed, so PEs take the steps as they install
/// ([`super::Machine::retire_steps`]) and none stalls for all of it. Until
/// a page's blocks retire, those not yet taken stay whole and reachable;
/// their code stays until the part's grace.
pub(super) struct Retirement {
    /// The ring part, or none for a host's reclaim.
    pub(super) part: Option<u32>,
    /// Why its blocks go, for their ghosts ([`super::churn`]).
    pub(super) cause: super::churn::Gone,
    pub(super) range: core::ops::Range<usize>,
    /// The blocks, by page then slot: a page's blocks go together, so
    /// each page's index list is walked once.
    pub(super) slots: Vec<(u64, u32)>,
    /// The first of `slots` not taken yet: past the page under way.
    pub(super) next: usize,
    /// The page under way, if one is.
    pub(super) page: Option<PageRetirement>,
    /// The PCs of the page's blocks, kept for the next page's capacity.
    pub(super) pcs: Vec<u64>,
    /// Where the release of the range's cells goes on, once the blocks went.
    pub(super) cells: Option<u64>,
    /// TSC ticks of its steps, and how many.
    pub(super) ticks: u64,
    pub(super) steps: u64,
    /// Blocks and index entries its steps visited, and the most in a step.
    pub(super) visits: u64,
    pub(super) step_visits: u64,
}

/// One page's blocks of a [`Retirement`], `slots[start..end]`, taken in
/// three phases, each in steps of bounded work: they leave the index, the
/// block table and their links out ([`PagePhase::Unindex`]); the page's
/// indexed blocks cut their links into them ([`PagePhase::Cut`]); they
/// retire ([`PagePhase::Retire`]). Between steps a block out of the index
/// is still live and its code whole: a link not yet cut or a jump cache
/// entry runs it, and it then exits to the engine, as it links nothing
/// and takes no table entry.
pub(super) struct PageRetirement {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) phase: PagePhase,
}

/// Where a [`PageRetirement`] goes on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PagePhase {
    /// The first of its blocks still in the index.
    Unindex(usize),
    /// The position in the page's index list the cut goes on from.
    Cut(usize),
    /// The first of its blocks not retired yet.
    Retire(usize),
}

/// What [`Block::part`] holds for code outside the board's arena.
pub(super) const NO_PART: u8 = u8::MAX;

/// The parts an [`Arena`] is cut into: what one retirement takes away.
const ARENA_PARTS: usize = 32;

/// The board's code arena: a host's whole [`CodeWindow`] cut into [`Ring`]
/// parts, written through its writable view and run through its executable
/// one.
pub(super) struct Arena {
    pub(super) window: CodeWindow,
    pub(super) part_bytes: usize,
    pub(super) ring: Ring,
}

impl Arena {
    fn new(window: CodeWindow) -> Self {
        let parts = ARENA_PARTS.min(window.bytes / 4096).max(MIN_PARTS);
        let part_bytes = (window.bytes / parts) & !4095;
        assert!(part_bytes >= 4096, "a code window of at least three pages");
        Self {
            window,
            part_bytes,
            ring: Ring::new(parts, part_bytes),
        }
    }

    /// The parts, as a host would give them.
    pub(super) fn partitions(&self) -> CodePartitions {
        CodePartitions {
            base: self.window.exec,
            bytes_per_part: self.part_bytes,
            parts: self.ring.parts as usize,
        }
    }

    /// The executable address of `place`.
    pub(super) fn address(&self, place: Place) -> usize {
        self.window.exec + place.part as usize * self.part_bytes + place.offset as usize
    }

    /// The part holding executable `address`, if the window does.
    pub(super) fn part_of(&self, address: usize) -> Option<usize> {
        let offset = address.checked_sub(self.window.exec)?;
        (offset < self.window.bytes)
            .then(|| (offset / self.part_bytes).min(self.ring.parts as usize - 1))
    }

    /// The executable range of `part`.
    pub(super) fn range(&self, part: u32) -> core::ops::Range<usize> {
        let start = self.window.exec + part as usize * self.part_bytes;
        start..start + self.part_bytes
    }

    /// The writable alias of executable `address`, checked to lie with
    /// `bytes` in the window.
    fn alias(&self, address: usize, bytes: usize) -> *mut u8 {
        let offset = address
            .checked_sub(self.window.exec)
            .filter(|offset| offset + bytes <= self.window.bytes)
            .expect("code inside the arena");
        (self.window.write + offset) as *mut u8
    }

    /// Copy `bytes` to executable `address`, which no core runs yet or
    /// which the caller's lock makes its own to rewrite.
    pub(super) fn write(&self, address: usize, bytes: &[u8]) {
        let alias = self.alias(address, bytes.len());
        // SAFETY: the host keeps the window writable for the board's life;
        // the caller owns these bytes (a fresh place, or a patch under the
        // board's lock).
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), alias, bytes.len()) };
    }

    /// Replace the instruction at patch point `point` with one aligned
    /// 8-byte store, which a core running it meets whole
    /// (`chain::patch_word`). Under the board's lock.
    pub(super) fn patch_point(&self, point: u64, instruction: [u8; chain::PATCH_BYTES]) {
        let word = self.alias((point & !7) as usize, 8) as *mut u64;
        // SAFETY: an aligned word of the window; the board's lock makes the
        // caller its only writer.
        let word = unsafe { AtomicU64::from_ptr(word) };
        let current = word.load(Ordering::Relaxed);
        word.store(
            chain::patch_word(current, point, instruction),
            Ordering::Release,
        );
    }
}

/// What a board's code changes under its lock: the block index (whose
/// [`Finder`] fetches read without it), the block table (whose entries
/// native probes read without it) and the blocks of each arena part. A
/// block's fault recovery is its record's ([`Record::recovery`]).
pub(super) struct Cache {
    pub(super) index: BlockIndex,
    pub(super) targets: LookupTable,
    pub(super) code_slots: CodeSlots,
    /// The blocks of a page being detached, kept for its capacity.
    pub(super) page_blocks: Vec<usize>,
    /// A bit per block slot for what a detach detaches.
    pub(super) marked: Vec<u64>,
    /// The lookup dispatchers in each arena part that has them, by ring
    /// generation and BTYPE: a tail calls the probes of its own part.
    pub(super) dispatchers: Vec<(u32, [Option<u64>; 4])>,
    /// How the lock was taken, counted by its holder.
    pub(super) lock: LockCounts,
    /// Block table entries taken and refused under the lock
    /// ([`TargetCounts`]).
    pub(super) targets_inserted: u64,
    pub(super) targets_raced: u64,
    /// Why blocks were installed on a shared board, with `--profile`:
    /// one census for the board, as its blocks are.
    pub(super) census: Option<Box<super::install_census::Census>>,
    /// Why blocks the board held before are installed again
    /// ([`super::churn`]).
    pub(super) churn: super::ChurnCounts,
}

/// How a board's code lock was taken, counted under it: no atomic beyond
/// the lock's own, and a timestamp only when another PE held it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LockCounts {
    pub acquired: u64,
    /// Acquisitions that found it held, the TSC ticks they waited and the
    /// longest wait since the last report ([`Code::report`]).
    pub contended: u64,
    pub waited: u64,
    pub longest: u64,
    /// By [`LockSite`]: what took it, how long it waited and held it.
    pub sites: [SiteCounts; LockSite::ALL.len()],
}

/// One [`LockSite`]'s share of the code lock: acquisitions, and the TSC
/// ticks waited for it and held, and the longest hold since the last report.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SiteCounts {
    pub acquired: u64,
    pub waited: u64,
    pub held: u64,
    pub longest_held: u64,
}

/// What takes the board's code lock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LockSite {
    /// A fetch's index lookup.
    Fetch,
    /// A block's publication.
    Install,
    /// A block table entry.
    Target,
    /// A direct link or cell.
    Link,
    /// A dynamic exit's lookup tail.
    Dynamic,
    /// A part's or a reclaim's eviction, one bounded step at a time.
    Evict,
    /// An IC IVAU's forget.
    Invalidate,
    /// A discard of every block, or its census, in bounded steps.
    Discard,
    /// The heartbeat's figures and the stale code sweep, in bounded steps.
    Report,
    /// Everything else: rare (a hub's start, a failure report).
    Other,
}

impl LockSite {
    pub const ALL: [Self; 10] = [
        Self::Fetch,
        Self::Install,
        Self::Target,
        Self::Link,
        Self::Dynamic,
        Self::Evict,
        Self::Invalidate,
        Self::Discard,
        Self::Report,
        Self::Other,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Fetch => "fetch",
            Self::Install => "install",
            Self::Target => "target",
            Self::Link => "link",
            Self::Dynamic => "dynamic",
            Self::Evict => "evict",
            Self::Invalidate => "invalidate",
            Self::Discard => "discard",
            Self::Report => "report",
            Self::Other => "other",
        }
    }
}

/// The board's code lock held, charging its hold to its [`LockSite`].
pub(super) struct CacheGuard<'a> {
    guard: MutexGuard<'a, Cache>,
    site: usize,
    since: u64,
}

impl core::ops::Deref for CacheGuard<'_> {
    type Target = Cache;
    fn deref(&self) -> &Cache {
        &self.guard
    }
}

impl core::ops::DerefMut for CacheGuard<'_> {
    fn deref_mut(&mut self) -> &mut Cache {
        &mut self.guard
    }
}

impl Drop for CacheGuard<'_> {
    fn drop(&mut self) {
        let held = timestamp().wrapping_sub(self.since);
        let site = &mut self.guard.lock.sites[self.site];
        site.held += held;
        site.longest_held = site.longest_held.max(held);
    }
}

/// What a board's code came to, for the heartbeat ([`super::Hub::code_report`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CodeReport {
    /// The window's bytes, its ring's parts and the current generation.
    pub window: usize,
    pub parts: usize,
    pub generation: u32,
    /// Parts taken away, of them for want of a block slot, and installs
    /// that lost to another PE's of the same block.
    pub retired: u64,
    pub forced: u64,
    pub lost: u64,
    /// Installs done again because a retirement chose their part before
    /// they published, and links refused because their thunk's or tail's
    /// part was chosen before the reference was noted.
    pub redone: u64,
    pub refused: u64,
    /// Waits for a part's grace, and those that ran out.
    pub grace_waits: u64,
    pub overdue: u64,
    /// The longest whole retirement since the last report: TSC ticks of
    /// its steps, and how many.
    pub retire_ticks: u64,
    pub retire_steps: u64,
    /// Block table entries a fetch's jump cache hit would add: those the
    /// table took, those refused under the lock (the block left the index
    /// meanwhile), those refused without it (the index no longer holds the
    /// block) and those put off (the block's entry keeps being displaced).
    pub targets: TargetCounts,
    pub lock: LockCounts,
    /// Installs of blocks the board held before, by why they went
    /// ([`super::ChurnCounts`]).
    pub churn: super::ChurnCounts,
    /// The parts young and tenured code hold, installs that went to the
    /// tenured generation, and parts taken away by why: the oldest young
    /// one, the tenured one entered least, a tenured one past its age, one
    /// whose blocks mostly left the index.
    pub generations: [u32; 2],
    pub promoted: u64,
    pub outs: [u64; 4],
    /// Installs relocated from an entry proven during the boot, entries
    /// proven and refused, and the bytes kept ([`super::rebase`]).
    pub rebases: super::RebaseCounts,
}

/// What [`CodeReport::targets`] counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TargetCounts {
    pub inserted: u64,
    pub raced: u64,
    pub unindexed: u64,
    pub deferred: u64,
}

/// Board events counted outside the lock: rare, so a shared atomic each,
/// on lines of their own apart from the fields every fetch reads.
#[derive(Default)]
#[repr(align(64))]
pub(super) struct Events {
    pub(super) retired: AtomicU64,
    pub(super) forced: AtomicU64,
    pub(super) lost: AtomicU64,
    /// Installs whose part a retirement chose before they published, done
    /// again; references into a part already chosen, refused.
    pub(super) redone: AtomicU64,
    pub(super) refused: AtomicU64,
    pub(super) grace_waits: AtomicU64,
    pub(super) overdue: AtomicU64,
    /// The longest whole retirement since the last report: the TSC ticks
    /// of its steps, and how many.
    pub(super) retire_ticks: AtomicU64,
    pub(super) retire_steps: AtomicU64,
    /// The last retirement's visits and most in one step, for tests.
    pub(super) retire_visits: AtomicU64,
    pub(super) retire_step_visits: AtomicU64,
    /// Jump cache hits whose block table entry was refused without the
    /// lock, or put off ([`TargetCounts`]).
    pub(super) targets_unindexed: AtomicU64,
    pub(super) targets_deferred: AtomicU64,
    /// Installs that went to the tenured generation.
    pub(super) promoted: AtomicU64,
}

impl Events {
    pub(super) fn count(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// A retirement ended.
    pub(super) fn retired(&self, retirement: &Retirement) {
        if self
            .retire_ticks
            .fetch_max(retirement.ticks, Ordering::Relaxed)
            < retirement.ticks
        {
            self.retire_steps.store(retirement.steps, Ordering::Relaxed);
        }
        self.retire_visits
            .store(retirement.visits, Ordering::Relaxed);
        self.retire_step_visits
            .store(retirement.step_visits, Ordering::Relaxed);
    }
}

/// A timestamp for lock waits: the TSC, which every PE of a board shares.
pub(super) fn timestamp() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: RDTSC has no effect beyond reading the counter.
        #[allow(unused_unsafe)]
        unsafe {
            core::arch::x86_64::_rdtsc()
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

/// The most block slots a board gives: a block takes at least a few dozen
/// bytes of code, so an arena's bytes over this bound its blocks.
const CODE_BYTES_PER_SLOT: usize = 32;
const MIN_SLOTS: usize = 1 << 16;
const MAX_SLOTS: usize = 1 << 24;

/// The block slots for an arena of `code_bytes`, or the most for a host
/// that does not say.
pub(super) fn slot_capacity(code_bytes: Option<usize>) -> u32 {
    code_bytes.map_or(MAX_SLOTS, |bytes| {
        (bytes / CODE_BYTES_PER_SLOT).clamp(MIN_SLOTS, MAX_SLOTS)
    }) as u32
}

impl Cache {
    fn new(targets: LookupTable, finder: Arc<Finder>) -> Self {
        Self {
            index: BlockIndex::with_finder(finder),
            targets,
            code_slots: CodeSlots::default(),
            page_blocks: Vec::new(),
            marked: Vec::new(),
            dispatchers: Vec::new(),
            lock: LockCounts::default(),
            targets_inserted: 0,
            targets_raced: 0,
            census: None,
            churn: super::ChurnCounts::default(),
        }
    }
}

impl Code {
    /// Room for `slots` blocks and a block table of `entries`
    /// (`chain::lookup_entries`).
    pub(super) fn new(slots: u32, entries: usize, window: Option<CodeWindow>) -> Self {
        let targets = LookupTable::with_entries(entries);
        let finder = Arc::new(Finder::new(slots));
        Self {
            blocks: Blocks::new(slots),
            table: targets.reader(),
            finder: Arc::clone(&finder),
            cache: Mutex::new(Cache::new(targets, finder)),
            ghosts: super::churn::Churn::new(slots),
            arena: window.map(Arena::new),
            epoch: Padded(AtomicU64::new(0)),
            events: Events::default(),
            retiring: Mutex::new(None),
            retire_pending: AtomicBool::new(false),
            step_visits: super::code_cache::STEP_VISITS,
            tagged: Default::default(),
            two_sided: Default::default(),
            rebases: super::rebase::Rebases::new(super::rebase::BOUND),
        }
    }

    /// The board's code lock, counted ([`LockCounts`]) as [`LockSite::Other`].
    #[cfg(test)]
    pub(super) fn cache(&self) -> CacheGuard<'_> {
        self.cache_at(LockSite::Other)
    }

    /// The board's code lock, counted ([`LockCounts`]) for `site`.
    pub(super) fn cache_at(&self, site: LockSite) -> CacheGuard<'_> {
        let site = site as usize;
        if let Ok(mut cache) = self.cache.try_lock() {
            cache.lock.acquired += 1;
            cache.lock.sites[site].acquired += 1;
            return CacheGuard {
                guard: cache,
                site,
                since: timestamp(),
            };
        }
        let start = timestamp();
        let mut cache = self.cache.lock().unwrap_or_else(|error| error.into_inner());
        let since = timestamp();
        let waited = since.wrapping_sub(start);
        let lock = &mut cache.lock;
        lock.acquired += 1;
        lock.contended += 1;
        lock.waited += waited;
        lock.longest = lock.longest.max(waited);
        lock.sites[site].acquired += 1;
        lock.sites[site].waited += waited;
        CacheGuard {
            guard: cache,
            site,
            since,
        }
    }

    /// The indexed block at `pc`, `physical` and `variant`: from the
    /// [`Finder`] without the lock, which a caller inside its section may
    /// read, or under it when the finder cannot say.
    pub(super) fn find(&self, pc: u64, physical: u64, variant: VariantId) -> Option<usize> {
        match self.finder.find(&self.blocks, pc, physical, variant) {
            Some(found) => found,
            None => self
                .cache_at(LockSite::Fetch)
                .index
                .get(&self.blocks, pc, physical, variant),
        }
    }

    /// The pages the block index holds, without the lock.
    pub(super) fn indexed_pages(&self) -> usize {
        self.finder.pages()
    }

    /// What the board's code came to; the lock's longest wait starts over.
    pub(super) fn report(&self) -> CodeReport {
        let (lock, inserted, raced, churn) = {
            let mut cache = self.cache_at(LockSite::Report);
            let lock = cache.lock;
            let counts = (cache.targets_inserted, cache.targets_raced);
            let churn = cache.churn;
            cache.lock.longest = 0;
            for site in &mut cache.lock.sites {
                site.longest_held = 0;
            }
            (lock, counts.0, counts.1, churn)
        };
        let count = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        CodeReport {
            window: self.arena.as_ref().map_or(0, |arena| arena.window.bytes),
            parts: self
                .arena
                .as_ref()
                .map_or(0, |arena| arena.ring.parts as usize),
            generation: self
                .arena
                .as_ref()
                .map_or(0, |arena| arena.ring.generation()),
            retired: count(&self.events.retired),
            forced: count(&self.events.forced),
            lost: count(&self.events.lost),
            redone: count(&self.events.redone),
            refused: count(&self.events.refused),
            grace_waits: count(&self.events.grace_waits),
            overdue: count(&self.events.overdue),
            retire_ticks: self.events.retire_ticks.swap(0, Ordering::Relaxed),
            retire_steps: self.events.retire_steps.swap(0, Ordering::Relaxed),
            targets: TargetCounts {
                inserted,
                raced,
                unindexed: count(&self.events.targets_unindexed),
                deferred: count(&self.events.targets_deferred),
            },
            lock,
            churn,
            generations: self
                .arena
                .as_ref()
                .map_or([0; 2], |arena| arena.ring.owned()),
            promoted: count(&self.events.promoted),
            outs: self
                .arena
                .as_ref()
                .map_or([0; 4], |arena| arena.ring.outs()),
            rebases: self.rebases.counts(),
        }
    }

    /// Forget every block: for a board no other machine holds.
    pub(super) fn clear(&mut self, free: bool) {
        self.blocks.clear(free);
        *self
            .retiring
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = None;
        *self.retire_pending.get_mut() = false;
        // The ring starts over: no other machine runs this board's code.
        if let Some(arena) = &mut self.arena {
            *arena = Arena::new(arena.window);
        }
        let mut cache = self.cache_at(LockSite::Discard);
        cache.targets.release_cells(0..=u64::MAX);
        cache.targets.clear();
        cache.index.clear();
        cache.code_slots = CodeSlots::default();
        cache.dispatchers.clear();
    }
}

/// A block slot's record. The block is written only by the PE that took
/// the slot from [`Slots`], before it publishes the slot (an index or table
/// store under the board's lock, or a release store of `live`), and stays
/// until the slot is taken again, which is only after every PE that could
/// have read the slot's number has left its section ([`Grace`]): so any PE
/// inside its section reads it without a lock. The rest are atomics any PE
/// changes under the board's lock and reads without it.
pub(super) struct Record {
    block: UnsafeCell<Option<Block>>,
    /// Advances each time the slot takes a block: a slot number held with
    /// it names that block while they match.
    incarnation: AtomicU32,
    /// Whether the slot holds an installed block.
    live: AtomicBool,
    /// Exits to the engine at the block's terminal, up to its limit.
    pub(super) taken: AtomicU32,
    /// A bit per successor: chained into a block of its page.
    linked: AtomicU64,
    /// A bit per successor: served by a lookup table cell, never linked.
    settled: AtomicU64,
    /// A bit per successor: its link slot's body was written, so a core
    /// may still run it after an unlink (`chain::link`).
    used: AtomicU64,
    /// Its dynamic exit is served by a lookup dispatcher.
    dynamic: AtomicBool,
    /// Its install lost to another PE's of the same block: never reachable.
    lost: AtomicBool,
    /// Jump cache hits that found no block table entry of it and went for
    /// the lock to add one ([`Self::remember`]).
    remembers: AtomicU32,
    /// Its fault recovery rows, under a lock of their own: a PE faulting
    /// in the block reads them, its first fault may qualify them. They stay
    /// after the block retires until the slot takes its next block, as a
    /// PE still inside its section may fault in it.
    recovery: Mutex<Option<CodeRecovery>>,
}

impl Default for Record {
    fn default() -> Self {
        Self {
            block: UnsafeCell::new(None),
            incarnation: AtomicU32::new(0),
            live: AtomicBool::new(false),
            taken: AtomicU32::new(0),
            linked: AtomicU64::new(0),
            settled: AtomicU64::new(0),
            used: AtomicU64::new(0),
            dynamic: AtomicBool::new(false),
            lost: AtomicBool::new(false),
            remembers: AtomicU32::new(0),
            recovery: Mutex::new(None),
        }
    }
}

/// A block slot's fault recovery that a caller found there.
pub(super) struct Owner<'a>(MutexGuard<'a, Option<CodeRecovery>>);

impl core::ops::Deref for Owner<'_> {
    type Target = CodeRecovery;
    fn deref(&self) -> &CodeRecovery {
        self.0.as_ref().expect("an owner found present")
    }
}

impl core::ops::DerefMut for Owner<'_> {
    fn deref_mut(&mut self) -> &mut CodeRecovery {
        self.0.as_mut().expect("an owner found present")
    }
}

/// The blocks' fault recovery, by slot, each under its record's lock.
pub(super) struct Recoveries<'a>(&'a Blocks);

impl Recoveries<'_> {
    fn lock(&self, slot: usize) -> Option<MutexGuard<'_, Option<CodeRecovery>>> {
        let record = self.0.records.get(u32::try_from(slot).ok()?)?;
        Some(
            record
                .recovery
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
        )
    }

    /// `slot`'s rows, held, if it has them.
    pub(super) fn get(&self, slot: &usize) -> Option<Owner<'_>> {
        self.lock(*slot).filter(|held| held.is_some()).map(Owner)
    }

    /// As [`Self::get`]: the guard lets the caller change them.
    pub(super) fn get_mut(&self, slot: &usize) -> Option<Owner<'_>> {
        self.get(slot)
    }

    /// `slot`'s rows, which its record holds once it was taken.
    pub(super) fn insert(&self, slot: usize, owner: CodeRecovery) {
        *self.lock(slot).expect("a slot that was given") = Some(owner);
    }

    pub(super) fn remove(&self, slot: &usize) -> Option<CodeRecovery> {
        self.lock(*slot)?.take()
    }

    /// The heap bytes the rows hold beyond their records.
    pub(super) fn heap_bytes(&self) -> usize {
        (0..self.0.len())
            .filter_map(|slot| self.get(&slot).map(|owner| owner.heap_bytes()))
            .sum()
    }

    /// Slots holding rows.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        (0..self.0.len())
            .filter(|slot| self.get(slot).is_some())
            .count()
    }

    /// The slots whose records have room for rows.
    #[cfg(test)]
    pub(super) fn capacity(&self) -> usize {
        self.0.capacity()
    }
}

// SAFETY: the block is written only by its slot's taker before the slot is
// published, and read only after (see above); the rest are atomics.
unsafe impl Sync for Record {}

/// Block table inserts a block takes at every attempt, and how often it
/// takes one after ([`Record::remember`]).
pub(super) const REMEMBERS_FREE: u32 = 4;
pub(super) const REMEMBER_EVERY: u32 = 32;

/// The most successors a block links: bits of [`Record::linked`].
pub(super) const MAX_SUCCESSORS: usize = u64::BITS as usize;

impl Record {
    pub(super) fn incarnation(&self) -> u32 {
        self.incarnation.load(Ordering::Acquire)
    }

    pub(super) fn live(&self) -> bool {
        self.live.load(Ordering::Acquire)
    }

    /// Whether successor `position` is linked or settled: not to link.
    pub(super) fn linked(&self, position: usize) -> bool {
        (self.linked.load(Ordering::Acquire) | self.settled.load(Ordering::Acquire)) & 1 << position
            != 0
    }

    /// Whether successor `position` is chained into a block of its page,
    /// which a detach of that block undoes.
    pub(super) fn chained(&self, position: usize) -> bool {
        self.linked.load(Ordering::Acquire) & 1 << position != 0
    }

    /// Successor `position` is served by a cell from now on: under the
    /// board's lock.
    pub(super) fn settle(&self, position: usize) {
        self.settled.fetch_or(1 << position, Ordering::Release);
    }

    /// Whether successor `position`'s link slot body was written.
    pub(super) fn used(&self, position: usize) -> bool {
        self.used.load(Ordering::Acquire) & 1 << position != 0
    }

    /// Mark successor `position` linked (or unlinked), and its slot body
    /// used if `used`: under the board's lock.
    pub(super) fn set_linked(&self, position: usize, linked: bool, used: bool) {
        let bit = 1 << position;
        if linked {
            self.linked.fetch_or(bit, Ordering::Release);
        } else {
            self.linked.fetch_and(!bit, Ordering::Release);
        }
        if used {
            self.used.fetch_or(bit, Ordering::Release);
        }
    }

    /// Whether its dynamic exit is still open for a lookup dispatcher.
    pub(super) fn dynamic_open(&self) -> bool {
        !self.dynamic.load(Ordering::Acquire)
    }

    pub(super) fn close_dynamic(&self) {
        self.dynamic.store(true, Ordering::Release);
    }

    /// Whether its install lost to another PE's ([`Self::mark_lost`]).
    pub(super) fn lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    /// Its install lost: under the board's lock, before its retirement.
    pub(super) fn mark_lost(&self) {
        self.lost.store(true, Ordering::Release);
    }

    /// A fetch found the block without a block table entry and would add
    /// one: whether this attempt is due. The first [`REMEMBERS_FREE`] are;
    /// after that one in [`REMEMBER_EVERY`], so an entry other inserts keep
    /// displacing from its set costs the lock a bounded share of its hits.
    pub(super) fn remember(&self) -> bool {
        let tries = self.remembers.fetch_add(1, Ordering::Relaxed);
        tries < REMEMBERS_FREE || tries.is_multiple_of(REMEMBER_EVERY)
    }
}

/// The board's blocks: slot numbers from [`Slots`] and a [`Record`] per
/// slot. Indexing gives a slot's block, as a table would.
pub(super) struct Blocks {
    slots: Slots,
    records: Records<Record>,
    /// Slots holding an installed block.
    live: AtomicUsize,
}

impl Blocks {
    pub(super) fn new(capacity: u32) -> Self {
        Self {
            slots: Slots::new(capacity),
            records: Records::new(capacity),
            live: AtomicUsize::new(0),
        }
    }

    /// One more than the highest slot given so far.
    pub(super) fn len(&self) -> usize {
        self.slots.issued() as usize
    }

    /// Slots holding a block.
    pub(super) fn live(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }

    pub(super) fn record(&self, slot: usize) -> &Record {
        self.records
            .get(slot as u32)
            .expect("a slot that was given")
    }

    /// `slot`'s record, if it has one yet: for a walk of every slot below
    /// [`Self::len`]. [`Self::take`] counts a slot as given before it
    /// allocates the record chunk of a slot that begins one, and gives the
    /// slot back without one when the heap has no room, so a slot below
    /// `len` may have no record (build 69: a heartbeat's exit walk met one
    /// 20 s into boot and stopped the machine).
    pub(super) fn try_record(&self, slot: usize) -> Option<&Record> {
        self.records.get(u32::try_from(slot).ok()?)
    }

    /// A slot given as [`Self::take`] gives it, without its record yet:
    /// what another PE sees between the two.
    #[cfg(test)]
    pub(super) fn take_without_record(&self, grace: &Grace) -> Option<usize> {
        self.slots.take(grace, None).map(|slot| slot as usize)
    }

    /// The blocks' fault recovery, each under its slot's lock.
    pub(super) fn recovery(&self) -> Recoveries<'_> {
        Recoveries(self)
    }

    /// `slot`'s block, if it ever held one.
    pub(super) fn get(&self, slot: usize) -> Option<&Block> {
        let record = self.records.get(u32::try_from(slot).ok()?)?;
        // SAFETY: see `Record`: written only before the slot was published.
        unsafe { (*record.block.get()).as_ref() }
    }

    /// `slot`'s block while it holds an installed one.
    pub(super) fn installed(&self, slot: usize) -> Option<&Block> {
        let record = self.records.get(u32::try_from(slot).ok()?)?;
        record.live().then(|| self.get(slot)).flatten()
    }

    /// A slot for a new block, with its record allocated, if one is left
    /// and the heap has room. `grace` and `skip` are as for [`Slots::take`].
    pub(super) fn take(&self, grace: &Grace, skip: Option<usize>) -> Option<usize> {
        let slot = self.slots.take(grace, skip)?;
        if self.records.try_reserve(slot).is_none() {
            self.slots.release(alloc::vec![slot], grace.snapshot());
            return None;
        }
        Some(slot as usize)
    }

    /// The block `slot` held before [`Self::take`] gave it to this
    /// caller, which nothing names any more.
    pub(super) fn vacate(&self, slot: usize) -> Option<Block> {
        let record = self.record(slot);
        debug_assert!(!record.live(), "a slot taken for a new block");
        // SAFETY: as for `fill`.
        unsafe { (*record.block.get()).take() }
    }

    /// Put `block` in `slot`, which [`Self::take`] gave and nothing names
    /// yet, and make it live.
    pub(super) fn fill(&self, slot: usize, block: Block) {
        let record = self.record(slot);
        debug_assert!(!record.live(), "a slot taken for a new block");
        // SAFETY: the slot was taken by this caller and no PE reads it: its
        // last release's grace passed.
        unsafe { *record.block.get() = Some(block) };
        record.taken.store(0, Ordering::Relaxed);
        record.linked.store(0, Ordering::Relaxed);
        record.settled.store(0, Ordering::Relaxed);
        record.used.store(0, Ordering::Relaxed);
        record.dynamic.store(false, Ordering::Relaxed);
        record.lost.store(false, Ordering::Relaxed);
        record.remembers.store(0, Ordering::Relaxed);
        record.incarnation.fetch_add(1, Ordering::Release);
        record.live.store(true, Ordering::Release);
        self.live.fetch_add(1, Ordering::Relaxed);
    }

    /// Make `slot` no longer live, as its block's code goes: under the
    /// board's lock, before the release of [`Self::release`].
    pub(super) fn retire(&self, slot: usize) {
        let record = self.record(slot);
        if record.live.swap(false, Ordering::AcqRel) {
            self.live.fetch_sub(1, Ordering::Relaxed);
        }
        record.incarnation.fetch_add(1, Ordering::Release);
    }

    /// `slots`, retired, were unpublished before `snapshot`.
    pub(super) fn release(&self, slots: Vec<u32>, snapshot: Snapshot) {
        self.slots.release(slots, snapshot);
    }

    /// Forget every block, for a board no other machine holds; `free`
    /// gives the records' memory back.
    pub(super) fn clear(&mut self, free: bool) {
        let capacity = self.slots.capacity;
        if free {
            *self = Self::new(capacity);
        } else {
            self.slots = Slots::new(capacity);
            self.records.reset();
            *self.live.get_mut() = 0;
        }
    }

    /// The heap bytes the records hold.
    pub(super) fn heap_bytes(&self) -> usize {
        self.records.heap_bytes()
    }

    /// The slots its allocated records cover.
    #[cfg(test)]
    pub(super) fn capacity(&self) -> usize {
        self.records
            .chunks
            .iter()
            .filter(|chunk| !chunk.load(Ordering::Relaxed).is_null())
            .count()
            * RECORD_CHUNK
    }

    /// Every slot that ever held a block, and its block.
    pub(super) fn iter(&self) -> impl Iterator<Item = (usize, &Block)> {
        (0..self.len()).filter_map(|slot| Some((slot, self.get(slot)?)))
    }
}

impl core::ops::Index<usize> for Blocks {
    type Output = Block;
    fn index(&self, slot: usize) -> &Block {
        self.get(slot).expect("a slot holding a block")
    }
}

/// How [`Board::wait`] ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Waited {
    Passed,
    /// The deadline passed first. Nothing is reused: the caller goes back
    /// to its run loop (where it reads its mail, which may be what the
    /// other PE waits on) and tries again later. `first` is true for the
    /// board's first overrun, which the caller reports.
    Overdue {
        first: bool,
    },
}

impl Board {
    pub(super) fn new(pes: usize) -> Self {
        Self {
            grace: Grace::new(pes),
            overdue: AtomicBool::new(false),
        }
    }

    /// Whether this is the board's first overrun of a wait: the one
    /// reported.
    pub(super) fn overdue(&self) -> bool {
        !self.overdue.swap(true, Ordering::Relaxed)
    }

    /// As PE `pe`, wait outside its section for `snapshot` for at most
    /// `ticks` of `counter`, the host's counter, then enter again. Never
    /// hangs: a PE inside that waits on this one (a broadcast, a stop)
    /// gets its answer once this one is back in its run loop.
    pub(super) fn wait(
        &self,
        pe: usize,
        snapshot: &Snapshot,
        ticks: u64,
        mut counter: impl FnMut() -> u64,
    ) -> Waited {
        let start = counter();
        if self
            .grace
            .wait(pe, snapshot, || counter().wrapping_sub(start) >= ticks)
        {
            return Waited::Passed;
        }
        Waited::Overdue {
            first: !self.overdue.swap(true, Ordering::Relaxed),
        }
    }
}

/// Block slot numbers below a capacity. A released slot goes to limbo with
/// the grace snapshot taken after its block was unpublished, and is given
/// again once that passed; until then a new number is given, and only a
/// full capacity refuses.
pub(super) struct Slots {
    capacity: u32,
    list: Mutex<SlotList>,
}

struct SlotList {
    next: u32,
    free: Vec<u32>,
    limbo: VecDeque<(Snapshot, Vec<u32>)>,
}

impl Slots {
    pub(super) fn new(capacity: u32) -> Self {
        Self {
            capacity,
            list: Mutex::new(SlotList {
                next: 0,
                free: Vec::new(),
                limbo: VecDeque::new(),
            }),
        }
    }

    /// A slot no PE holds, if one is left. `skip` is as for
    /// [`Grace::passed`].
    pub(super) fn take(&self, grace: &Grace, skip: Option<usize>) -> Option<u32> {
        let mut list = self.list.lock().unwrap_or_else(|error| error.into_inner());
        if list.free.is_empty() {
            while list
                .limbo
                .front()
                .is_some_and(|(snapshot, _)| grace.passed(snapshot, skip))
            {
                let (_, slots) = list.limbo.pop_front().expect("a front batch");
                list.free.extend(slots);
            }
        }
        if let Some(slot) = list.free.pop() {
            return Some(slot);
        }
        (list.next < self.capacity).then(|| {
            list.next += 1;
            list.next - 1
        })
    }

    /// `slots` were unpublished before `snapshot` was taken.
    pub(super) fn release(&self, slots: Vec<u32>, snapshot: Snapshot) {
        if slots.is_empty() {
            return;
        }
        self.list
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .limbo
            .push_back((snapshot, slots));
    }

    /// The most slots given at once so far: what block records need room for.
    pub(super) fn issued(&self) -> u32 {
        self.list
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .next
    }
}

/// Records in one allocation of [`Records`], which never moves.
const RECORD_CHUNK: usize = 1024;

/// A record per slot, in chunks allocated as slots reach them, which never
/// move: any PE reads a record with no lock. Records hold their fields in
/// atomics, so a writer and a reader never race on plain memory; what a
/// field means is settled by the slot discipline: the PE that took a slot
/// from [`Slots`] fills its record before it publishes the block (a table
/// or index store with release), and nobody reads a released slot's record
/// once its grace passed.
pub(super) struct Records<T> {
    chunks: Box<[AtomicPtr<T>]>,
    growing: Mutex<()>,
}

// SAFETY: a chunk is published once and freed only on drop; `T` is shared
// across PEs only through `&T`.
unsafe impl<T: Send + Sync> Send for Records<T> {}
// SAFETY: as above.
unsafe impl<T: Send + Sync> Sync for Records<T> {}

impl<T: Default> Records<T> {
    /// Room for `capacity` slots, none allocated yet.
    pub(super) fn new(capacity: u32) -> Self {
        Self {
            chunks: (0..(capacity as usize).div_ceil(RECORD_CHUNK))
                .map(|_| AtomicPtr::new(core::ptr::null_mut()))
                .collect(),
            growing: Mutex::new(()),
        }
    }

    /// `slot`'s record, if its chunk was allocated.
    pub(super) fn get(&self, slot: u32) -> Option<&T> {
        let chunk = self
            .chunks
            .get(slot as usize / RECORD_CHUNK)?
            .load(Ordering::Acquire);
        // SAFETY: a published chunk of RECORD_CHUNK records, live until drop.
        (!chunk.is_null()).then(|| unsafe { &*chunk.add(slot as usize % RECORD_CHUNK) })
    }

    /// `slot`'s record, allocating its chunk if needed: for the PE that
    /// took the slot, before it publishes it.
    #[cfg(test)]
    pub(super) fn reserve(&self, slot: u32) -> &T {
        if let Some(record) = self.get(slot) {
            return record;
        }
        let index = slot as usize / RECORD_CHUNK;
        let _growing = self
            .growing
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if self.chunks[index].load(Ordering::Acquire).is_null() {
            let chunk: Box<[T]> = (0..RECORD_CHUNK).map(|_| T::default()).collect();
            let chunk = Box::into_raw(chunk) as *mut T;
            self.chunks[index].store(chunk, Ordering::Release);
        }
        self.get(slot).expect("a published chunk")
    }

    /// [`Self::reserve`], or nothing when the heap has no room for the
    /// chunk or `slot` is past the capacity.
    pub(super) fn try_reserve(&self, slot: u32) -> Option<&T> {
        if let Some(record) = self.get(slot) {
            return Some(record);
        }
        let index = slot as usize / RECORD_CHUNK;
        let _growing = self
            .growing
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let chunk = self.chunks.get(index)?;
        if chunk.load(Ordering::Acquire).is_null() {
            let mut records = Vec::new();
            records.try_reserve_exact(RECORD_CHUNK).ok()?;
            records.resize_with(RECORD_CHUNK, T::default);
            let records = Box::into_raw(records.into_boxed_slice()) as *mut T;
            chunk.store(records, Ordering::Release);
        }
        self.get(slot)
    }

    /// Every record back to its default, the chunks kept: for an owner no
    /// other PE reads through.
    pub(super) fn reset(&mut self) {
        for chunk in &mut self.chunks {
            let chunk = *chunk.get_mut();
            if !chunk.is_null() {
                // SAFETY: a chunk of RECORD_CHUNK records, owned exclusively.
                let records = unsafe { core::slice::from_raw_parts_mut(chunk, RECORD_CHUNK) };
                records.fill_with(T::default);
            }
        }
    }

    /// The bytes its chunks hold.
    pub(super) fn heap_bytes(&self) -> usize {
        let chunks = self
            .chunks
            .iter()
            .filter(|chunk| !chunk.load(Ordering::Relaxed).is_null())
            .count();
        chunks * RECORD_CHUNK * size_of::<T>() + self.chunks.len() * size_of::<AtomicPtr<T>>()
    }
}

impl<T> Drop for Records<T> {
    fn drop(&mut self) {
        for chunk in &mut self.chunks {
            let chunk = *chunk.get_mut();
            if !chunk.is_null() {
                let slice = core::ptr::slice_from_raw_parts_mut(chunk, RECORD_CHUNK);
                // SAFETY: allocated in `reserve` as a boxed slice of this length.
                drop(unsafe { Box::from_raw(slice) });
            }
        }
    }
}
