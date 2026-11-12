//! Invalidation and reclamation of compiled blocks when translation or code
//! changes.

use super::*;

/// Blocks [`Machine::stale_code`] copies under the board's lock at a time.
const STALE_STEP: usize = 128;

/// Slots one step of an eviction or a discard's census takes under the
/// board's lock: tens of microseconds.
const EVICT_STEP: usize = 32;

/// Units of work one step of a retirement takes under the board's lock
/// ([`Machine::retire_step`]): a block or an index entry visited, about a
/// cache miss or a few each. A launch measured 1024 at 116 us a step: each
/// of the other PEs' fetches, links and installs, one every ~50 us per PE,
/// that comes during a step waits for the rest of it, so the waits a
/// retirement causes grow with the square of its step (build of run
/// 20261111-025825: 1.2 s of `evict` holds, 4-7 s of waits per beat). At
/// 256 a sign-in measured 43 us a step against one acquisition every 37 us
/// per PE (run 20261111-100711: 2.1 M a beat over 8 PEs); an eighth of the
/// first keeps a step under that gap.
pub(super) const STEP_VISITS: u64 = 128;

/// Cells one step of an eviction releases under the board's lock.
const CELL_STEP: usize = 256;
/// The fetches a PE counts per ring part before it adds them to the ring's
/// shared counts ([`Machine::entered`]).
const ENTERED_BATCH: u32 = 64;

/// What [`Machine::stale_code`] checks of a block, copied under the lock.
#[derive(Clone, Copy)]
struct StaleBlock {
    pc: u64,
    physical: u64,
    code: usize,
    source_bytes: usize,
    source_hash: u64,
    installer: u8,
}

/// The word and bit of the two positions of the leaf of `size` bytes at
/// `base` in a filter of [`FETCHED_LEAVES`] bits. A base's low bits are clear,
/// so the size's exponent keeps leaves of one base apart.
fn fetched_leaf_bits(base: u64, size: u64) -> [(usize, u64); 2] {
    let hash = (base | u64::from(size.trailing_zeros())).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    [hash >> 40, hash >> 16].map(|bits| {
        let bit = bits as usize & (FETCHED_LEAVES - 1);
        (bit / 64, 1 << (bit % 64))
    })
}

/// Whether `slot`'s bit is set in `bits`.
fn bit_set(bits: &[u64], slot: u32) -> bool {
    bits.get(slot as usize / 64)
        .is_some_and(|word| word & 1 << (slot % 64) != 0)
}

/// `marked`, a bit per slot below `len`, with the bits of `slots` set, the
/// rest clear: what [`unmark`] takes back.
fn mark(marked: &mut Vec<u64>, len: usize, slots: &[u32]) -> Vec<u64> {
    let mut marked = core::mem::take(marked);
    let words = len.div_ceil(64);
    if marked.len() < words {
        marked.resize(words, 0);
    }
    for &slot in slots {
        marked[slot as usize / 64] |= 1 << (slot % 64);
    }
    marked
}

/// Clear the bits of `slots` in `marked` and keep it in `kept` for the next.
fn unmark(kept: &mut Vec<u64>, mut marked: Vec<u64>, slots: &[u32]) {
    for &slot in slots {
        marked[slot as usize / 64] &= !(1 << (slot % 64));
    }
    *kept = marked;
}

/// The block keys `block` may have entries under ([`Machine::block_key`]):
/// its frame and variant, with the guarded page's BT bit clear and set.
fn block_keys(block: &Block) -> [u64; 2] {
    let state = (block.variant.0 + 1) << 1;
    [state, state | 1].map(|state| chain::block_key(block.physical, state))
}

/// The frame number of the 4 KiB frame holding `physical`, as block keys
/// name it.
fn frame_number(physical: u64) -> u64 {
    chain::key_frame(chain::block_key(physical, 1))
}

impl<H: Host, D: Devices> Machine<H, D> {
    /// Check the next [`STALE_CHECKS`] reachable compiled blocks against
    /// the guest bytes they were compiled from, a sweep that goes on where
    /// the last call stopped: a diagnostic of code changed without the IC
    /// maintenance that would discard its blocks, bounded per call.
    pub fn stale_code(&mut self) -> StaleCode {
        let mut report = StaleCode::default();
        let mut bytes = Vec::new();
        let mut last = None;
        // What each block's check needs, copied in bounded steps under the
        // lock: the RAM reads and hashes run without it.
        let mut checks = Vec::with_capacity(STALE_STEP);
        while report.blocks < STALE_CHECKS {
            checks.clear();
            {
                let cache = self.code.cache_at(board::LockSite::Report);
                let step = STALE_STEP.min(STALE_CHECKS - report.blocks);
                for (cursor, index) in cache.index.sweep(last.or(self.stale_cursor)).take(step) {
                    let block = &self.code.blocks[index];
                    checks.push((
                        cursor,
                        StaleBlock {
                            pc: block.pc,
                            physical: block.physical,
                            code: block.code.0,
                            source_bytes: block.source_bytes(),
                            source_hash: block.source_hash,
                            installer: block.installer,
                        },
                    ));
                }
            }
            if checks.is_empty() {
                break;
            }
            for &(cursor, block) in &checks {
                self.check_stale(&mut report, &mut bytes, block);
                last = Some(cursor);
            }
        }
        // A sweep that reached the last block starts over next time.
        self.stale_cursor = last.filter(|_| report.blocks == STALE_CHECKS);
        report
    }

    /// Check one block of [`Self::stale_code`] against its guest bytes.
    fn check_stale(&mut self, report: &mut StaleCode, bytes: &mut Vec<u8>, block: StaleBlock) {
        bytes.resize(block.source_bytes, 0);
        report.blocks += 1;
        if !self.host.read_ram(block.physical, bytes) {
            return;
        }
        let raw = source_hash(bytes);
        // A block's hash is of the bytes its compile admitted.
        for word in bytes.as_chunks_mut::<4>().0 {
            *word = self.admit(u32::from_le_bytes(*word)).to_le_bytes();
        }
        let admitted = source_hash(bytes);
        if admitted == block.source_hash && raw != admitted {
            report.admitted += 1;
        }
        if admitted == block.source_hash {
            return;
        }
        report.stale += 1;
        if report.first.is_none() {
            report.first = Some((block.pc, block.physical));
            let part = self.code.arena.as_ref().and_then(|arena| {
                let offset = block.code.checked_sub(arena.window.exec)?;
                Some((offset / arena.part_bytes) as u32)
            });
            report.origin = Some(StaleOrigin {
                pe: block.installer,
                part,
                noted: self
                    .parallel
                    .as_ref()
                    .map(|hub| hub.has_code(block.physical)),
            });
        }
    }

    /// An architectural cache epoch must never wrap back to an old permission.
    pub(super) fn advance_translation_generation(&mut self) {
        self.mirrors.close_code_page();
        self.mirrors.transitions.contexts_moved();
        self.translation_generation = self
            .translation_generation
            .checked_add(1)
            .expect("architectural translation generation exhausted");
    }

    /// What TTBR0 or its ASID selects may have changed.
    fn advance_table_generation(&mut self) {
        self.mirrors.close_code_page();
        self.mirrors.transitions.contexts_moved();
        self.table_generation = self
            .table_generation
            .checked_add(1)
            .expect("architectural table generation exhausted");
    }

    /// What `half`'s cached fetches and contexts last as long as: TTBR0
    /// selects the lower half's, and the upper half's once a fetch of it was
    /// non-global, by its ASID ([`Self::half_inputs`]).
    pub(super) fn fetch_epoch(&self, half: usize) -> FetchEpoch {
        let table = half == 0 || self.private_fetches[1];
        (
            self.translation_generation,
            if table { self.table_generation } else { 0 },
        )
    }

    /// The translations of the halves `halves` may have changed with no
    /// control value showing it: their lookup contexts go, so no lookup
    /// translates through their ITLB entries, nor does a cached fetch
    /// outlive what [`Self::fetched_through`] learnt of it. The block
    /// table's entries stay: they hold for a frame, whatever maps it.
    pub(super) fn forget_spaces(&mut self, halves: core::ops::RangeInclusive<usize>, cause: usize) {
        self.forget_contexts(halves, cause);
        self.advance_translation_generation();
    }

    /// Forget the contexts of `halves` that translate `asid`: what a TLBI of
    /// one process leaves of the others', which their ITLB entries stay good
    /// for.
    pub(super) fn forget_asid(&mut self, halves: core::ops::RangeInclusive<usize>, asid: u16) {
        for half in halves {
            self.statistics.lookup_forgets[0] += 1;
            self.spaces[half].retain(|space| !same_asid(space, asid));
            self.recent[half] = [None; RECENT];
        }
        self.advance_table_generation();
    }

    /// A TLBI of everything of some regimes reached nothing of `regime`'s
    /// ([`mmu::tlbi::Scope::reach`] answered [`mmu::tlbi::Reach::Nothing`]
    /// for the context the PE translates in): the contexts of other
    /// regimes go, which it may have reached, and `regime`'s stay.
    fn forget_other_regimes(&mut self, regime: mmu::Regime) {
        let mut forgot = false;
        for half in 0..2 {
            if self.spaces[half].len() == 0 {
                continue;
            }
            if self.spaces[half].iter().all(|space| space.regime != regime) {
                self.forget_contexts(half..=half, 1);
                forgot = true;
                continue;
            }
            if self.spaces[half].retain(|space| space.regime == regime) != 0 {
                self.statistics.lookup_forgets[1] += 1;
                self.recent[half] = [None; RECENT];
                forgot = true;
            }
        }
        let source = if forgot {
            ItlbSource::AllForgot
        } else {
            ItlbSource::AllKept
        };
        self.statistics.itlb_sources[source as usize] += 1;
        self.advance_translation_generation();
    }

    /// [`Self::forget_spaces`] for a caller that advances the generation.
    fn forget_contexts(&mut self, halves: core::ops::RangeInclusive<usize>, cause: usize) {
        for half in halves {
            self.refill_after(half);
            self.statistics.lookup_forgets[cause] += 1;
            self.spaces[half].clear();
            self.recent[half] = [None; RECENT];
            // Bits are inserted with a fetched size and only cleared here.
            if core::mem::take(&mut self.fetched_sizes[half]) != 0 {
                let filter = &mut self.fetched_leaves[half];
                match &mut self.fetched_words[half] {
                    Some(words) => {
                        for word in words.drain(..) {
                            filter[word as usize] = 0;
                        }
                    }
                    words @ None => {
                        filter.fill(0);
                        *words = Some(Vec::new());
                    }
                }
            }
            self.private_fetches[half] = false;
            if half == 0 {
                self.lower_globals = NO_GLOBALS;
            }
        }
    }

    /// A fetch in `half` translated through a leaf of `size` bytes. The
    /// upper half's contexts leave out the ASID until a fetch of it is
    /// non-global ([`Self::half_inputs`]): the first such fetch forgets
    /// them, as their entries may name blocks of another ASID's page.
    pub(super) fn fetched_through(&mut self, half: usize, pc: u64, size: u64, global: bool) {
        if half == 1 && !global && !self.private_fetches[1] {
            self.forget_spaces(1..=1, 3);
        }
        let leaf = size.next_power_of_two();
        let filter = &mut self.fetched_leaves[half];
        if filter.is_empty() {
            *filter = vec![0; FETCHED_LEAVES / 64];
        }
        let words = &mut self.fetched_words[half];
        for (word, bit) in fetched_leaf_bits(pc & !(leaf - 1), leaf) {
            if filter[word] == 0 {
                match words {
                    Some(set) if set.len() < FETCHED_WORDS => set.push(word as u32),
                    _ => *words = None,
                }
            }
            filter[word] |= bit;
        }
        self.fetched_sizes[half] |= leaf;
        self.private_fetches[half] |= !global;
        if half == 0 && global {
            let (first, last) = self.lower_globals;
            self.lower_globals = (first.min(pc), last.max(pc.saturating_add(size)));
        }
    }

    /// Whether a fetch of `half` translated through the leaf of `size` bytes
    /// at `base`, or through one that shares its filter bits.
    fn leaf_fetched(&self, half: usize, base: u64, size: u64) -> bool {
        fetched_leaf_bits(base, size)
            .iter()
            .all(|&(word, bit)| self.fetched_leaves[half][word] & bit != 0)
    }

    /// The span of `half` a TLBI of `first..=last` can reach of what it
    /// fetched: the leaves around them that fetches translated through, at
    /// each size such leaves had.
    fn fetched_span(&self, half: usize, first: u64, last: u64) -> Option<(u64, u64)> {
        const UPPER: u64 = !0 << 55;
        let (low, high) = [(0, !UPPER), (UPPER, u64::MAX)][half];
        if last < low || first > high {
            return None;
        }
        let (first, last) = (first.max(low), last.min(high));
        let mut span: Option<(u64, u64)> = None;
        let mut sizes = self.fetched_sizes[half];
        while sizes != 0 {
            let size = sizes.isolate_lowest_one();
            sizes &= sizes - 1;
            let (start, end) = (first & !(size - 1), last | (size - 1));
            let leaves = (end - start) / size;
            let fetched = leaves >= SCANNED_LEAVES
                || (0..=leaves).any(|leaf| self.leaf_fetched(half, start + leaf * size, size));
            if fetched {
                span = Some(span.map_or((start, end), |(lowest, highest)| {
                    (lowest.min(start), highest.max(end))
                }));
            }
        }
        span
    }

    /// Drop the cached fetches of the pages a TLBI reaches in the current
    /// translation: all of them if it reaches more than the cache holds.
    fn forget_fetches(&mut self, reach: mmu::tlbi::Reach) {
        let (first, last) = match reach {
            mmu::tlbi::Reach::Nothing => return,
            mmu::tlbi::Reach::All => return self.advance_translation_generation(),
            mmu::tlbi::Reach::Containing(address) => (address, address),
            mmu::tlbi::Reach::Overlapping { first, last } => (first, last),
        };
        for half in 0..2 {
            let Some((first, last)) = self.fetched_span(half, first, last) else {
                continue;
            };
            if (last - first) / FETCH_PAGE >= FETCH_CACHE as u64 {
                return self.advance_translation_generation();
            }
            for page in (first..=last).step_by(FETCH_PAGE as usize) {
                let slot = (page / FETCH_PAGE) as usize % FETCH_CACHE;
                if self.fetch[slot].is_some_and(|entry| (first..=last).contains(&entry.page)) {
                    self.fetch[slot] = None;
                }
            }
        }
    }

    /// Drop the ITLB entries of the pages a TLBI by VA reaches
    /// ([`Self::fetched_span`]), as Arm `TLBIMatch` selects TLB entries:
    /// - without an ASID (VAAE1, VAALE1, RVAAE1, RVAALE1 and their IS/OS
    ///   forms, and every upper-half span): those of every context;
    /// - with `asid` (VAE1, VALE1, RVAE1, RVALE1 and their IS/OS forms) in
    ///   the lower half: those of `asid`'s contexts, and in any context
    ///   those whose walk ended in a global leaf
    ///   ([`chain::LookupEntry::is_global_page`]), which match every ASID.
    ///   A non-global entry of another ASID stays. A context's ASID is the
    ///   one its regime's TCR.A1 selects from TTBR0 or TTBR1
    ///   ([`mmu::current_asid`]), both of which its key holds, with the
    ///   8 high bits ignored as for TCR.AS=0 (more entries go, never
    ///   fewer).
    ///
    /// A process's exit reaches its own pages this way and leaves the other
    /// processes' private pages. The ITLB finds the entries of the span's
    /// pages in every context from its index of where each granule's went
    /// (`chain::LookupTable`). Contexts stay: the hot pages of one that went
    /// would each cost an exit on every PE that runs them. The block table's
    /// entries stay too: a page translated anew finds them again by its
    /// frame.
    pub(super) fn forget_lookups(&mut self, reach: mmu::tlbi::Reach, asid: Option<u16>) {
        let (first, last) = match reach {
            mmu::tlbi::Reach::Nothing => return,
            mmu::tlbi::Reach::All => return self.forget_spaces(0..=1, 1),
            mmu::tlbi::Reach::Containing(address) => (address, address),
            mmu::tlbi::Reach::Overlapping { first, last } => (first, last),
        };
        for half in 0..2 {
            let Some((first, last)) = self.fetched_span(half, first, last) else {
                continue;
            };
            let scoped = asid.filter(|_| half == 0);
            let every = &self.spaces;
            let spaces = &every[half];
            let (ghosts, statistics) = (&mut self.itlb_ghosts, &mut self.statistics);
            let dropped = &mut self.dropped_refills;
            let forget = |entry: &chain::LookupEntry| {
                let source = match scoped {
                    None => Some(ItlbSource::DropEvery),
                    Some(_) if entry.is_global_page() => Some(ItlbSource::DropEvery),
                    Some(asid) => spaces
                        .live(entry.key)
                        .is_some_and(|space| same_asid(space, asid))
                        .then_some(ItlbSource::DropAsid),
                };
                let Some(source) = source else {
                    return false;
                };
                statistics.itlb_drops[1] += 1;
                statistics.itlb_sources[source as usize] += 1;
                super::compilation::note_itlb_ghost(ghosts, entry, ItlbMiss::Forgotten);
                if spaces.live(entry.key).is_some() {
                    Self::keep_dropped(
                        dropped,
                        |number| every.iter().any(|spaces| spaces.live(number).is_some()),
                        entry.key,
                        entry.pc,
                    );
                }
                true
            };
            if self.itlb.forget_range_where(first..=last, forget) {
                self.statistics.lookup_forgets[2] += 1;
            }
        }
    }

    /// A translation control changed: cached fetch translations go, and
    /// of the host's shadow what the new controls translate differently
    /// ([`Host::set_regime`]). Lookup contexts are keyed by the controls'
    /// values ([`contexts::Key`]), so a return to earlier ones finds theirs
    /// again.
    pub(super) fn controls_changed(&mut self) {
        self.derived_changed();
        self.permissions_changed();
    }

    /// Translation changed, but nothing derived from System registers:
    /// an SPRR_PERM write, which XNU makes around its writes to protected
    /// data and at thread switches, thousands of times a second.
    pub(super) fn permissions_changed(&mut self) {
        self.mirrors.close(SPRR_PERM_MIRROR);
        self.advance_translation_generation();
        self.statistics.translation_changes += 1;
        self.sync_host_regime();
    }

    /// Anything about translation may have changed: every lookup context
    /// goes too, and the whole shadow.
    pub(super) fn translations_changed(&mut self) {
        self.forget_contexts(0..=1, 3);
        self.host.invalidate_translations();
        self.controls_changed();
    }

    /// The host's answers to [`Host::fetch_permitted`] changed: every later
    /// fetch translates and asks again, and no cached, looked-up or chained
    /// transfer enters a block of another page before it has.
    pub fn fetch_permissions_changed(&mut self) {
        self.forget_spaces(0..=1, 3);
    }

    /// A TLBI: the fetch translations `reach` names go, and the host drops
    /// what it names of its shadow. Of the lookup table, which
    /// holds every context's entries, it drops what `everywhere` names: the
    /// entries at its VAs, or the contexts of the halves non-global entries
    /// were fetched through, or every context. No control changed, so the
    /// regime and what derives from System registers stay.
    pub(super) fn tlb_maintained(
        &mut self,
        reach: mmu::tlbi::Reach,
        everywhere: mmu::tlbi::Everywhere,
        asid: Option<u16>,
    ) {
        self.mirrors.close_code_page();
        match everywhere {
            mmu::tlbi::Everywhere::Addresses(addresses) => {
                // Canonical memory's kept TLBs refresh on the generation.
                if self.canonical_memory_serves() {
                    self.advance_translation_generation();
                } else {
                    self.forget_fetches(reach);
                }
                self.forget_lookups(addresses, asid);
            }
            mmu::tlbi::Everywhere::NonGlobal(_) => {
                let last = usize::from(self.private_fetches[1]);
                match asid {
                    Some(asid) => self.forget_asid(0..=last, asid),
                    None => self.forget_spaces(0..=last, 0),
                }
            }
            // The forms without an address. One that reaches nothing of the
            // regime the PE translates in reaches no entry of it
            // (`Scope::reach`, `Scope::covers`).
            mmu::tlbi::Everywhere::All if reach == mmu::tlbi::Reach::Nothing => {
                self.forget_other_regimes(self.cpu.tlbi_current().regime);
            }
            mmu::tlbi::Everywhere::All => {
                // Mail's whole invalidations count as received too
                // (`itlb_drops`' mail_all tells them apart).
                let source = match self.applying_mail {
                    true => ItlbSource::AllReceived,
                    false => ItlbSource::AllLocal,
                };
                self.statistics.itlb_sources[source as usize] += 1;
                self.forget_spaces(0..=1, 1);
            }
        }
        self.statistics.tlb_maintenance += 1;
        if reach != mmu::tlbi::Reach::All {
            self.statistics.scoped_maintenance += 1;
        }
        self.host.invalidate_reach(reach, everywhere);
    }

    /// A TTBR0 write, which changes the translation of the lower half
    /// alone: the regime's new controls tell the host so
    /// ([`Host::set_regime`]), which keeps the upper half's shadow. Lookup
    /// contexts are keyed by its value, as by every control's.
    pub(super) fn table_base_changed(&mut self) {
        self.advance_table_generation();
        self.statistics.table_base_changes += 1;
        self.sync_host_regime();
    }

    /// Forget the blocks compiled from the page holding `physical`, which an
    /// IC IVAU made stale, and their lookup entries, which every address
    /// space mapping the frame shares: a later fetch compiles the page
    /// again. Their code stays installed until its part is reclaimed.
    pub(super) fn invalidate_code_page(&mut self, physical: u64) {
        let frame = frame_number(physical);
        let code = Arc::clone(&self.code);
        let mut cache = code.cache_at(board::LockSite::Invalidate);
        // The page's blocks leave the index, so no detach finds them as
        // sources again: their links out are cut now.
        let page: Vec<usize> = cache.index.page(physical).collect();
        for &slot in &page {
            self.note_gone(&mut cache, slot, churn::Gone::Invalidated);
        }
        self.unlink_sources(&page);
        cache.targets.forget_frames(frame..=frame, |_| true);
        let forgot = cache.index.forget_page(physical & !(FETCH_PAGE - 1));
        drop(cache);
        // Every PE's jump cache on a shared board still names its blocks.
        if forgot || self.code_hub.is_some() {
            self.forget_code_page_here(physical);
        } else {
            self.forget_page_caches(physical);
        }
    }

    /// Another PE's IC IVAU of the page holding `physical`. On a shared
    /// board the poster forgot the page already, but this PE may have read
    /// the old bytes before the write and published their block after that
    /// forget: a compile runs outside, so the post did not wait for it. It
    /// reads its mail only after it published, so forgetting the page again
    /// here takes such a block away.
    pub(super) fn invalidate_posted_code_page(&mut self, physical: u64) {
        self.invalidate_code_page(physical);
    }

    /// Forget this PE's caches of the page holding `physical`, and its jump
    /// cache.
    fn forget_code_page_here(&mut self, physical: u64) {
        self.forget_page_caches(physical);
        // The jump cache holds no generation per page.
        self.code_generation += 1;
        self.statistics.code_invalidations += 1;
    }

    /// Forget this PE's stored page, waiting direct exit and known key of
    /// the page holding `physical`.
    fn forget_page_caches(&mut self, physical: u64) {
        self.forget_stored_page(physical & !(FETCH_PAGE - 1));
        // A forgotten block's exit waits for no block of its page.
        if self.direct_exit.is_some_and(|(block, ..)| {
            (self.code.blocks[block].physical ^ physical) & !(FETCH_PAGE - 1) == 0
        }) {
            self.direct_exit = None;
        }
        self.forget_known_key(physical & !(FETCH_PAGE - 1));
    }

    /// Make room in the code arena: reclaim its oldest part, or discard
    /// every block on a host whose arena is one part.
    pub(super) fn reclaim_code(&mut self) {
        // A ring reclaims its parts as installs fill it.
        if self.code.arena.is_some() {
            return;
        }
        self.lookup_dispatchers = Default::default();
        match self.host.reclaim_code() {
            Some(range) => self.evict_code(range),
            None => self.discard_code(Discard::Arena),
        }
    }

    /// Make room in the host's own arena for a leaf. With the board's
    /// arena holding the blocks, the host's holds only leaves: its oldest
    /// part goes with the leaves in it.
    pub(super) fn reclaim_leaves(&mut self) {
        if self.code.arena.is_none() {
            return self.reclaim_code();
        }
        match self.host.reclaim_code() {
            Some(range) => self
                .scalable_leaves
                .retain(|_, leaf| !range.contains(&leaf.code.0)),
            None => {
                self.host.discard_code();
                self.scalable_leaves.clear();
            }
        }
    }

    /// Release local code owners under pressure before allocating compilation
    /// input and recovery metadata: the oldest arena part's blocks, or every
    /// block on a host whose arena is one part.
    ///
    /// Memory another machine or the shared store holds stays low however
    /// much this one releases, so within one stretch of pressure a machine
    /// reclaims again only once it holds as many blocks as it did at its
    /// last reclaim: its growth stays bounded, and it does not release code
    /// before every block it installs. A machine holding fewer than
    /// [`MEMORY_RELEASE_BLOCKS`] keeps them: what they free would not
    /// relieve the heap.
    pub(super) fn reclaim_for_memory(&mut self) {
        // A ring's blocks are bounded by its bytes.
        if self.code.arena.is_some() || !self.host.memory_low() {
            self.reclaimed_at = None;
            return;
        }
        let live = self.code.blocks.live();
        if live < self.memory_release_blocks.max(1)
            || self.reclaimed_at.is_some_and(|held| live < held)
        {
            return;
        }
        self.reclaimed_at = Some(live);
        self.statistics.code_releases += 1;
        match self.host.release_oldest_code() {
            Some(range) => self.evict_code(range),
            None => self.discard_code(Discard::Memory),
        }
    }

    /// The grace slots this machine releases go to limbo under, and the PE
    /// it is there: a machine that keeps its code alone has a grace of no
    /// PE, so its slots are free again at once.
    pub(super) fn grace(&self) -> (&board::Grace, Option<usize>) {
        match &self.code_hub {
            Some(hub) => (&hub.board.grace, Some(self.current)),
            None => (&self.alone, None),
        }
    }

    /// Install native `bytes`: in the board's arena when it has one, else
    /// in the host's. `retire` lets a full ring take away its oldest part's
    /// code to make room, for a caller outside the board's lock: a block's
    /// install ([`Self::place`]), which keeps its places until it publishes
    /// ([`Self::release_places`]), so that their parts are not opened again
    /// meanwhile. An install it makes for the block (a thunk) holds the
    /// block's place, so it never waits for a grace or a part another
    /// install holds: it finds the arena full instead. Without `retire` the
    /// caller holds the board's lock from the take to its last write, and
    /// the lock keeps the part, as no retirement ends without it.
    pub(super) fn install_native(
        &mut self,
        bytes: &[u8],
        retire: bool,
    ) -> Result<NativeCode, InstallError> {
        let stream = core::mem::replace(&mut self.install_stream, board::YOUNG);
        let code = Arc::clone(&self.code);
        let Some(arena) = &code.arena else {
            return self.host.install(bytes);
        };
        if bytes.len() > arena.part_bytes {
            return Err(InstallError::Full);
        }
        #[cfg(debug_assertions)]
        self.assert_portable(bytes);
        // A part being retired takes a bounded step or a few at each install:
        // not for a link's thunk, whose caller holds the code lock.
        if retire {
            self.retire_steps(self.retire_budget());
        }
        let holding = !self.held_places.is_empty();
        loop {
            let full = match arena.ring.take(stream, bytes.len(), 16) {
                Ok(place) => {
                    let address = arena.address(place);
                    arena.write(address, bytes);
                    self.host.serialize_code();
                    if retire {
                        self.held_places.push(place);
                    } else {
                        arena.ring.release(place);
                    }
                    return Ok(NativeCode(address));
                }
                Err(full) => full,
            };
            if !retire {
                return Err(InstallError::Full);
            }
            let (grace, skip) = self.grace();
            match arena.ring.advance(stream, full, grace, skip) {
                board::Advance::Moved => {}
                // A PE that waits on another's install while holding a
                // place could be the one that install waits on.
                board::Advance::Busy | board::Advance::Wait(_) if holding => {
                    return Err(InstallError::Full);
                }
                // The next part is still being taken away or written: help.
                board::Advance::Busy => {
                    self.retire_steps(self.retire_budget());
                    self.host.relax();
                }
                board::Advance::Retire(part, _) => {
                    self.retire_part(part, churn::Gone::Ring);
                }
                board::Advance::Wait(snapshot) => self.wait_grace(&snapshot),
            }
        }
    }

    /// Host counter ticks of retirement steps one install takes: half a
    /// millisecond, so no PE stalls the guest count for a whole part.
    fn retire_budget(&self) -> u64 {
        (self.profile.counter_frequency / 2000).max(1)
    }

    /// Code in the board's arena runs on every PE: it names none of this
    /// PE's own data by address, only through its anchors, whose values
    /// differ by PE ([`Self::anchors`]).
    #[cfg(debug_assertions)]
    fn assert_portable(&self, bytes: &[u8]) {
        let anchors = self.anchors();
        let own = [
            anchors[Anchor::Control],
            anchors[Anchor::Mirrors],
            anchors[Anchor::Cells],
            anchors[Anchor::Pointer],
            anchors[Anchor::Memo],
            anchors[Anchor::Count],
            anchors[Anchor::StackZero],
            anchors[Anchor::Amx],
            &self.cpu.state as *const _ as u64,
        ];
        for window in bytes.windows(8) {
            let value = u64::from_le_bytes(window.try_into().expect("eight bytes"));
            assert!(
                value == 0 || !own.contains(&value),
                "board code names PE {}'s own address {value:#x}",
                self.current
            );
        }
    }

    /// Overwrite installed code at `code` with `bytes`.
    pub(super) fn patch_native(&mut self, code: NativeCode, bytes: &[u8]) {
        match &self.code.arena {
            Some(arena) => {
                arena.write(code.0, bytes);
                self.host.serialize_code();
            }
            None => self.host.patch(code, bytes),
        }
    }

    /// Replace the instruction at patch point `code`, which a core may be
    /// running, in one store (`Host::patch_instruction`).
    pub(super) fn patch_point(&mut self, code: NativeCode, instruction: [u8; chain::PATCH_BYTES]) {
        match &self.code.arena {
            Some(arena) => {
                assert!(chain::patchable(code.0 as u64), "a patch point");
                arena.patch_point(code.0 as u64, instruction);
                self.host.serialize_code();
            }
            None => self.host.patch_instruction(code, instruction),
        }
    }

    /// Unlink the exits whose patch points and instructions are `unlinks`,
    /// each in one store ([`Self::patch_point`]). In the board's arena the
    /// writer serializes once after the last: no other core's fetch depends
    /// on it, and this core runs none of them before it returns.
    pub(super) fn unlink_points(&mut self, unlinks: Vec<(u64, [u8; chain::PATCH_BYTES])>) {
        self.statistics.unlinks += unlinks.len() as u64;
        let Some(arena) = &self.code.arena else {
            for (point, instruction) in unlinks {
                self.host
                    .patch_instruction(NativeCode(point as usize), instruction);
            }
            return;
        };
        if unlinks.is_empty() {
            return;
        }
        for (point, instruction) in unlinks {
            assert!(chain::patchable(point), "a patch point");
            arena.patch_point(point, instruction);
        }
        self.host.serialize_code();
    }

    /// The generation of the arena part holding executable `address`, for
    /// code that part's code jumps to: a thunk or a tail goes with the
    /// block it serves ([`board::Ring::refer`]).
    pub(super) fn stream_at(&self, address: u64) -> usize {
        let arena = self.code.arena.as_ref();
        arena
            .and_then(|arena| Some(arena.ring.kind(arena.part_of(address as usize)?)))
            .unwrap_or(board::YOUNG)
    }

    /// Code at `from` is about to jump to `to`, which the caller just
    /// installed and still holds: `to`'s part stays while `from`'s does.
    /// False when `to`'s part was chosen to go meanwhile: the caller makes
    /// no jump, and its exit stays an exit to the engine. Under the board's
    /// lock, or before `from`'s block is published.
    #[must_use]
    pub(super) fn note_reference(&self, from: u64, to: u64) -> bool {
        if let Some(arena) = &self.code.arena
            && let (Some(from), Some(to)) =
                (arena.part_of(from as usize), arena.part_of(to as usize))
            && !arena.ring.refer(from, to)
        {
            board::Events::count(&self.code.events.refused);
            return false;
        }
        true
    }

    /// The places this PE's install took ([`Self::install_native`]) are
    /// published or given up: their parts may be opened again.
    pub(super) fn release_places(&mut self) {
        if let Some(arena) = &self.code.arena {
            for place in self.held_places.drain(..) {
                arena.ring.release(place);
            }
        }
    }

    /// A fetch found block `slot`: its part counts an entry, for the choice
    /// of the tenured part to go ([`board::Ring::entered`]). The PE keeps
    /// its counts and adds them to the ring's every [`ENTERED_BATCH`]
    /// fetches, before each install (where the ring chooses) and when it
    /// leaves its run loop: the hottest part's count is one line every PE's
    /// fetches would write. The choice needs only the parts' order by
    /// heat, which a batch of at most that many fetches per PE leaves.
    pub(super) fn entered(&mut self, slot: usize) {
        let part = self.code.blocks[slot].part;
        if part == board::NO_PART {
            return;
        }
        let part = usize::from(part);
        if self.entered_pending.len() <= part {
            self.entered_pending.resize(part + 1, 0);
        }
        self.entered_pending[part] += 1;
        self.entered_total += 1;
        if self.entered_total >= ENTERED_BATCH {
            self.flush_entered();
        }
    }

    /// Add this PE's kept part entries to the ring's counts.
    pub(super) fn flush_entered(&mut self) {
        if self.entered_total == 0 {
            return;
        }
        self.entered_total = 0;
        let Some(arena) = &self.code.arena else {
            return;
        };
        for (part, count) in self.entered_pending.iter_mut().enumerate() {
            if *count != 0 {
                arena.ring.entered_by(part, core::mem::take(count));
            }
        }
    }

    /// The generation a block of `pc`, `physical` and `variant` goes to:
    /// tenured when the ring took it away within a few turns
    /// ([`churn::Churn::promotes`]).
    pub(super) fn stream_for(&self, pc: u64, physical: u64, variant: VariantId) -> usize {
        let Some(arena) = &self.code.arena else {
            return board::YOUNG;
        };
        let identity = churn::identity(pc, physical, variant.0);
        let ring = &arena.ring;
        if ring.young_parts != 0
            && self
                .code
                .ghosts
                .promotes(identity, ring.generation(), ring.parts)
        {
            board::Events::count(&self.code.events.promoted);
            board::TENURED
        } else {
            board::YOUNG
        }
    }

    /// The ring's current generation and parts, or zeros without a ring.
    fn ring_position(&self) -> (u32, u32) {
        self.code
            .arena
            .as_ref()
            .map_or((0, 0), |arena| (arena.ring.generation(), arena.ring.parts))
    }

    /// Note that block `slot` leaves the index for `cause`
    /// ([`churn::Churn::gone`]). Under the board's lock, `cache`.
    pub(super) fn note_gone(&self, cache: &mut board::Cache, slot: usize, cause: churn::Gone) {
        let block = &self.code.blocks[slot];
        let identity = churn::identity(block.pc, block.physical, block.variant.0);
        let (generation, _) = self.ring_position();
        self.code.ghosts.gone(
            &mut cache.churn,
            identity,
            block.source_hash,
            generation,
            cause,
        );
        // A hole in a part that stays: its part may go early.
        if cause == churn::Gone::Invalidated
            && let Some(arena) = &self.code.arena
            && let Some(part) = arena.part_of(block.code.0)
        {
            arena.ring.dropped(part);
        }
    }

    /// Note the install of block `slot`, before it enters the index
    /// ([`churn::Churn::installed`]). Under the board's lock, `cache`.
    pub(super) fn note_churn(&self, cache: &mut board::Cache, slot: usize) {
        let blocks = &self.code.blocks;
        let block = &blocks[slot];
        let shared = cache
            .index
            .sharing(blocks, block.pc, block.physical, block.variant);
        let identity = churn::identity(block.pc, block.physical, block.variant.0);
        let (generation, parts) = self.ring_position();
        self.code.ghosts.installed(
            &mut cache.churn,
            identity,
            block.source_hash,
            generation,
            shared,
            parts,
        );
        if let Some(arena) = &self.code.arena
            && let Some(part) = arena.part_of(block.code.0)
        {
            arena.ring.installed(part);
        }
    }

    /// With every block slot given, move the board's ring on before its
    /// parts are full, taking the oldest parts' blocks away and waiting for
    /// their grace, until a slot is free. The caller installs its code again
    /// after: its part may be one that went. A slot, if that freed one.
    pub(super) fn retire_for_slot(&mut self) -> Option<usize> {
        let code = Arc::clone(&self.code);
        let arena = code.arena.as_ref()?;
        for _ in 0..2 * arena.ring.parts {
            if let Some(slot) = self.take_slot() {
                return Some(slot);
            }
            let (grace, skip) = self.grace();
            let young = arena.ring.head(board::YOUNG);
            match arena.ring.advance(board::YOUNG, young, grace, skip) {
                board::Advance::Moved => {}
                board::Advance::Busy => {
                    self.retire_steps(u64::MAX);
                    self.host.relax();
                }
                board::Advance::Retire(part, _) => {
                    self.retire_part(part, churn::Gone::Slot);
                    // Short of slots: all of it now.
                    self.retire_steps(u64::MAX);
                    board::Events::count(&code.events.forced);
                    let snapshot = self.grace().0.snapshot();
                    self.wait_grace(&snapshot);
                }
                board::Advance::Wait(snapshot) => self.wait_grace(&snapshot),
            }
        }
        self.take_slot()
    }

    /// Take away the code of the ring's `part`, which this
    /// PE moved the ring past ([`board::Advance::Retire`]): its blocks go
    /// as in a reclaim, and its probes, in steps the board's installs take
    /// ([`Self::retire_steps`]); the part is reused once the last step is
    /// done and every PE passed the grace taken after.
    fn retire_part(&mut self, part: u32, cause: churn::Gone) {
        let code = Arc::clone(&self.code);
        let range = code
            .arena
            .as_ref()
            .expect("a ring retires its parts")
            .range(part);
        let mut job = self.retirement(range, Some(part));
        job.cause = cause;
        let mut held = code
            .retiring
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // The ring moves on only once the last retirement ended, so none is
        // left; one would be finished here first.
        if let Some(mut left) = held.take() {
            while !self.retire_step(&mut left) {}
            self.finish_retirement(left);
        }
        *held = Some(job);
        code.retire_pending.store(true, Ordering::Release);
        drop(held);
        self.retire_steps(self.retire_budget());
    }

    /// Take steps of the board's retirement under way, if no other PE is,
    /// until it ends or `budget` host counter ticks passed.
    pub(super) fn retire_steps(&mut self, budget: u64) {
        if !self.code.retire_pending.load(Ordering::Acquire) {
            return;
        }
        let code = Arc::clone(&self.code);
        let Ok(mut held) = code.retiring.try_lock() else {
            return;
        };
        let started = self.host.counter();
        loop {
            let Some(job) = held.as_mut() else {
                return;
            };
            if self.retire_step(job) {
                break;
            }
            if self.host.counter().wrapping_sub(started) >= budget {
                self.install_causes.evict_ticks += self.host.counter().wrapping_sub(started);
                return;
            }
        }
        let job = held.take().expect("a retirement that ended");
        code.retire_pending.store(false, Ordering::Release);
        drop(held);
        self.finish_retirement(job);
        self.install_causes.evict_ticks += self.host.counter().wrapping_sub(started);
    }

    /// A retirement of the blocks whose code lies in `range`.
    fn retirement(
        &mut self,
        range: core::ops::Range<usize>,
        part: Option<u32>,
    ) -> board::Retirement {
        let code = Arc::clone(&self.code);
        let listed = code
            .cache_at(board::LockSite::Evict)
            .code_slots
            .take(&range);
        // Without a list, every slot's record, read without the lock: the
        // range takes no new block.
        let evicted = listed.unwrap_or_else(|| {
            code.blocks
                .iter()
                .filter(|&(slot, block)| {
                    code.blocks.record(slot).live() && range.contains(&block.code.0)
                })
                .map(|(slot, _)| slot as u32)
                .collect()
        });
        let mut slots: Vec<(u64, u32)> = evicted
            .iter()
            .map(|&slot| {
                (
                    code.blocks[slot as usize].physical & !(FETCH_PAGE - 1),
                    slot,
                )
            })
            .collect();
        slots.sort_unstable();
        code.cache_at(board::LockSite::Evict)
            .code_slots
            .give_back(evicted);
        board::Retirement {
            part,
            cause: match part {
                Some(_) => churn::Gone::Ring,
                None => churn::Gone::Other,
            },
            cells: Some(range.start as u64),
            range,
            slots,
            next: 0,
            page: None,
            pcs: Vec::new(),
            ticks: 0,
            steps: 0,
            visits: 0,
            step_visits: 0,
        }
    }

    /// One step of `job`, one hold of the code lock, of about
    /// [`STEP_VISITS`] units of work: the phases of its pages in turn
    /// ([`board::PageRetirement`]), and then its range's cells. Between
    /// steps no lookup, link or index read reaches a block that left the
    /// index, a block retires only once no link into it is left, and the
    /// others are still whole. Whether the job is done.
    pub(super) fn retire_step(&mut self, job: &mut board::Retirement) -> bool {
        let started = board::timestamp();
        let code = Arc::clone(&self.code);
        let mut cache = code.cache_at(board::LockSite::Evict);
        let mut visits = 0;
        if job.next < job.slots.len() || job.page.is_some() {
            while visits < code.step_visits {
                let left = code.step_visits - visits;
                if job.page.is_none() {
                    if job.next >= job.slots.len() {
                        break;
                    }
                    let first = job.slots[job.next].0;
                    let end = job.next
                        + job.slots[job.next..].partition_point(|&(page, _)| page == first);
                    job.pcs.clear();
                    job.page = Some(board::PageRetirement {
                        start: job.next,
                        end,
                        phase: board::PagePhase::Unindex(job.next),
                    });
                }
                let page = job.page.as_mut().expect("a page under way");
                let frame = job.slots[page.start].0;
                match page.phase {
                    board::PagePhase::Unindex(from) => {
                        let to = page.end.min(from + EVICT_STEP.min(left as usize));
                        visits += self.unindex(
                            &mut cache,
                            frame,
                            &job.slots[from..to],
                            &mut job.pcs,
                            job.cause,
                        );
                        page.phase = if to == page.end {
                            job.pcs.sort_unstable();
                            job.pcs.dedup();
                            board::PagePhase::Cut(0)
                        } else {
                            board::PagePhase::Unindex(to)
                        };
                    }
                    board::PagePhase::Cut(from) => {
                        let (next, cut) =
                            self.cut_links_into(&mut cache, frame, &job.pcs, from, left as usize);
                        visits += cut;
                        page.phase = match next {
                            Some(next) => board::PagePhase::Cut(next),
                            None => board::PagePhase::Retire(page.start),
                        };
                    }
                    board::PagePhase::Retire(from) => {
                        let to = page.end.min(from + EVICT_STEP);
                        self.retire_blocks(&mut cache, &job.slots[from..to]);
                        visits += (to - from) as u64;
                        if to == page.end {
                            job.next = page.end;
                            job.page = None;
                        } else {
                            page.phase = board::PagePhase::Retire(to);
                        }
                    }
                }
            }
        } else if let Some(from) = job.cells {
            // The exits in the range go, and with them their cells: no
            // block in it is live now, so none links through a cell again.
            let most = CELL_STEP.min(code.step_visits as usize);
            job.cells = cache
                .targets
                .release_some_cells(from..=job.range.end as u64 - 1, most);
            visits = most as u64;
        }
        drop(cache);
        job.visits += visits;
        job.step_visits = job.step_visits.max(visits);
        job.ticks += board::timestamp().wrapping_sub(started);
        job.steps += 1;
        job.next >= job.slots.len() && job.page.is_none() && job.cells.is_none()
    }

    /// The blocks `step` of the page at `frame` leave the index, the block
    /// table and their links out, and this PE's jump cache and waiting
    /// exit; their PCs go to `pcs`. They stay live: links into them are
    /// cut next ([`Self::cut_links_into`]). Under the board's lock,
    /// `cache`. The work it took, in [`STEP_VISITS`] units: a block each,
    /// a sixteenth of the page's index list, and the cells looked at.
    fn unindex(
        &mut self,
        cache: &mut board::Cache,
        frame: u64,
        step: &[(u64, u32)],
        pcs: &mut Vec<u64>,
        cause: churn::Gone,
    ) -> u64 {
        let code = Arc::clone(&self.code);
        let blocks = &code.blocks;
        let slots: Vec<u32> = step.iter().map(|&(_, slot)| slot).collect();
        // Only blocks still in the index leave a ghost: one an IC IVAU
        // forgot has its own, and a lost install's key is the winner's.
        for &slot in &slots {
            let block = &blocks[slot as usize];
            if cache
                .index
                .get(blocks, block.pc, block.physical, block.variant)
                == Some(slot as usize)
            {
                self.note_gone(cache, slot as usize, cause);
            }
        }
        let (_, listed) = cache.index.page_part(frame, 0, 0);
        let marked = mark(&mut cache.marked, blocks.len(), &slots);
        cache
            .index
            .forget_marked(frame, |slot| bit_set(&marked, slot));
        unmark(&mut cache.marked, marked, &slots);
        let mut visits = (slots.len() + listed / 16) as u64;
        let mut unlinks = Vec::new();
        for &slot in &slots {
            let block = &blocks[slot as usize];
            pcs.push(block.pc);
            for key in block_keys(block) {
                visits += cache.targets.forget_block(block.pc, key, slot) as u64;
            }
            // No detach visits it once it left the index: its links out
            // go now, and it links nothing again.
            let record = blocks.record(slot as usize);
            for (position, exit) in block.successors.iter().enumerate() {
                if record.chained(position) {
                    record.set_linked(position, false, false);
                    unlinks.push(
                        chain::unlinked(exit.site, exit.slot.into())
                            .expect("a linked exit's patch point"),
                    );
                }
            }
            let entry = &mut self.jump[(block.pc / INSTRUCTION) as usize % JUMP_CACHE];
            if entry.is_some_and(|entry| entry.block == slot as usize) {
                *entry = None;
            }
            if self
                .direct_exit
                .is_some_and(|(waiting, ..)| waiting == slot as usize)
            {
                self.direct_exit = None;
            }
        }
        self.unlink_points(unlinks);
        visits
    }

    /// Cut the links of the indexed blocks of the page at `frame`, from
    /// position `from` of its list and at most `most` of them, into the
    /// sorted `pcs`: those of blocks that left the index. A link names the
    /// block a PC and variant have, so a link to any of the PCs goes (one
    /// to a block compiled again there links again). Blocks indexed after
    /// they left link to none of them. Under the board's lock, `cache`.
    /// Where the cut goes on, if blocks are left, and the blocks visited.
    fn cut_links_into(
        &mut self,
        cache: &mut board::Cache,
        frame: u64,
        pcs: &[u64],
        from: usize,
        most: usize,
    ) -> (Option<usize>, u64) {
        let code = Arc::clone(&self.code);
        let blocks = &code.blocks;
        let (sources, listed) = cache.index.page_part(frame, from, most);
        let mut visited = 0;
        let mut unlinks = Vec::new();
        for source in sources {
            visited += 1;
            let record = blocks.record(source);
            for (position, exit) in blocks[source].successors.iter().enumerate() {
                if record.chained(position) && pcs.binary_search(&exit.target.0).is_ok() {
                    record.set_linked(position, false, false);
                    // Linked, so its point is patchable and in reach.
                    unlinks.push(
                        chain::unlinked(exit.site, exit.slot.into())
                            .expect("a linked exit's patch point"),
                    );
                }
            }
        }
        // One store each, which a core running the exit may meet.
        self.unlink_points(unlinks);
        let next = from + visited;
        ((next < listed).then_some(next), visited as u64)
    }

    /// Retire the blocks of `step`, which nothing reaches but a core
    /// already in one: their slots go once the retirement ends and its
    /// grace passed. Under the board's lock, `cache`.
    fn retire_blocks(&mut self, cache: &mut board::Cache, step: &[(u64, u32)]) {
        let code = Arc::clone(&self.code);
        let shared = self.code_hub.is_some();
        let now = self.host.counter();
        for slot in step.iter().map(|&(_, slot)| slot as usize) {
            // On a shared board its rows stay until the slot takes its
            // next block: another PE still inside its section may be
            // running the block and fault in it. A machine alone runs
            // none of it now.
            let recovery = code.blocks.recovery();
            let bytes = match shared {
                true => recovery.get(&slot).map(|owner| owner.code_bytes()),
                false => recovery.remove(&slot).map(|owner| owner.code_bytes()),
            }
            .unwrap_or(0);
            self.census_evicted(cache, slot, bytes, now);
            // Retired first: a PE adding its kept exits checks the block's
            // incarnation and liveness, so it adds them here before the
            // swap or by word after (but for one that checked just before).
            code.blocks.retire(slot);
            let taken = code.blocks.record(slot).taken.swap(0, Ordering::Relaxed);
            if taken != 0 {
                *self
                    .exit_words
                    .entry(code.blocks[slot].terminal)
                    .or_default() += u64::from(taken);
            }
        }
    }

    /// The end of `job`, whose steps are done: its slots go once their
    /// grace passed, and a ring part's probes and the part itself.
    fn finish_retirement(&mut self, job: board::Retirement) {
        let code = Arc::clone(&self.code);
        let arena = job.part.and_then(|_| code.arena.as_ref());
        if let (Some(_), Some(part)) = (arena, job.part) {
            code.cache_at(board::LockSite::Evict)
                .dispatchers
                .retain(|&(held, _)| held != part);
        }
        let slots: Vec<u32> = job.slots.iter().map(|&(_, slot)| slot).collect();
        let (grace, _) = self.grace();
        code.blocks.release(slots, grace.snapshot());
        if let (Some(arena), Some(part)) = (arena, job.part) {
            arena.ring.retired(part, grace.snapshot());
            board::Events::count(&code.events.retired);
        }
        if !self.scalable_leaves.is_empty() {
            self.scalable_leaves
                .retain(|_, leaf| !job.range.contains(&leaf.code.0));
        }
        if !job.slots.is_empty() {
            self.statistics.code_evictions += 1;
        }
        code.events.retired(&job);
    }

    /// Wait until `snapshot`'s grace passed, in rounds bounded by the host
    /// counter; the board's first overrun is reported once. A PE inside
    /// its section leaves it to wait ([`board::Board::wait`]); outside (a
    /// compile), it holds nothing anyone waits on.
    fn wait_grace(&mut self, snapshot: &board::Snapshot) {
        let Some(hub) = self.code_hub.clone() else {
            return;
        };
        let events = &self.code.events;
        board::Events::count(&events.grace_waits);
        let ticks = (self.profile.counter_frequency / 10).max(1);
        let pe = self.current;
        let waited = if hub.board.grace.inside(pe) {
            hub.board.wait(pe, snapshot, ticks, || self.host.counter())
        } else {
            let start = self.host.counter();
            loop {
                if hub.board.grace.passed(snapshot, None) {
                    break board::Waited::Passed;
                }
                if self.host.counter().wrapping_sub(start) >= ticks {
                    break board::Waited::Overdue {
                        first: hub.board.overdue(),
                    };
                }
                self.host.relax();
            }
        };
        if let board::Waited::Overdue { .. } = waited {
            board::Events::count(&self.code.events.overdue);
        }
        if let board::Waited::Overdue { first: true } = waited {
            let requirement = self.requirement(RequirementKind::Native(
                "a code part's grace ran past its deadline: its reuse waits",
            ));
            self.degrade(requirement);
        }
    }

    /// What the board's machines share: every index entry and block table
    /// entry goes, so no fetch, lookup or jump cache entry reaches a block
    /// installed before; their code goes as the ring reaches its parts.
    fn forget_shared_code(&mut self) {
        let mut cache = self.code.cache_at(board::LockSite::Discard);
        cache.index.clear();
        cache.targets.clear();
        drop(cache);
        self.code.epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// What jump cache entries are valid in: this machine's code
    /// generation and the board's epoch ([`Self::forget_shared_code`]).
    pub(super) fn jump_generation(&self) -> u64 {
        self.code_generation
            .wrapping_add(self.code.epoch.load(Ordering::Acquire) << 40)
    }

    /// Forget the blocks and leaves whose code lies in `range`, which the
    /// host reclaimed. The other blocks of their pages stay: they chain into
    /// them no more ([`Self::detach`]), and take an exit to the engine
    /// there until it links them to the blocks compiled again. Only indexed
    /// blocks are reached: one replaced in the index was detached then,
    /// and the blocks of a page an IC IVAU forgot whole reach only each
    /// other, with no lookup entry and in a code generation no jump cache
    /// entry names since. Their slots are given again once the grace
    /// passed ([`board::Blocks::release`]).
    pub(super) fn evict_code(&mut self, range: core::ops::Range<usize>) {
        // A host with nothing to release says so with an empty range, and
        // while memory stays low that comes before every compile.
        if range.is_empty() {
            return;
        }
        let started = self.host.counter();
        let mut job = self.retirement(range, None);
        while !self.retire_step(&mut job) {}
        self.finish_retirement(job);
        self.install_causes.evict_ticks += self.host.counter().wrapping_sub(started);
    }

    /// Cut every chained exit of the blocks in `slots`, which leave the
    /// index: under the board's lock.
    fn unlink_sources(&mut self, slots: &[usize]) {
        let code = Arc::clone(&self.code);
        for &source in slots {
            let record = code.blocks.record(source);
            for (position, exit) in code.blocks[source].successors.iter().enumerate() {
                if record.chained(position) {
                    record.set_linked(position, false, false);
                    // Linked, so its point is patchable and in reach.
                    let (point, instruction) = chain::unlinked(exit.site, exit.slot.into())
                        .expect("a linked exit's patch point");
                    self.patch_point(NativeCode(point as usize), instruction);
                    self.statistics.unlinks += 1;
                }
            }
        }
    }

    /// Make the blocks in the sorted `slots` unreachable: the indexed blocks
    /// of their pages undo their links to their PCs (a link names the
    /// block a PC and variant have, so any of the page's blocks of the PC
    /// may be the one it reached), and their lookup and jump cache
    /// entries go. Under the board's lock, `cache`.
    pub(super) fn detach(&mut self, cache: &mut board::Cache, slots: &[u32]) {
        let blocks = &self.code.blocks;
        let marked = mark(&mut cache.marked, blocks.len(), slots);
        let detached = |slot: usize| bit_set(&marked, slot as u32);
        let mut gone: Vec<(u64, u64)> = slots
            .iter()
            .map(|&slot| {
                let block = &blocks[slot as usize];
                (block.physical & !(FETCH_PAGE - 1), block.pc)
            })
            .collect();
        gone.sort_unstable();
        gone.dedup();
        let mut sources = core::mem::take(&mut cache.page_blocks);
        let mut unlinks = Vec::new();
        for page in gone.chunk_by(|a, b| a.0 == b.0) {
            sources.extend(cache.index.page(page[0].0));
            for &source in &sources {
                if detached(source) {
                    continue;
                }
                let record = blocks.record(source);
                for (position, exit) in blocks[source].successors.iter().enumerate() {
                    if record.chained(position)
                        && page.binary_search(&(page[0].0, exit.target.0)).is_ok()
                    {
                        record.set_linked(position, false, false);
                        // Linked, so its point is patchable and in reach.
                        let unlink = chain::unlinked(exit.site, exit.slot.into())
                            .expect("a linked exit's patch point");
                        unlinks.push(unlink);
                    }
                }
            }
            sources.clear();
        }
        cache.page_blocks = sources;
        // One store each, which a core running the exit may meet.
        self.unlink_points(unlinks);
        // Lookup entries go by the frames their keys name, in runs.
        let mut frames: Vec<u64> = gone.iter().map(|&(page, _)| frame_number(page)).collect();
        frames.dedup();
        for run in frames.chunk_by(|a, b| a + 1 == *b) {
            cache
                .targets
                .forget_frames(run[0]..=run[run.len() - 1], |entry| {
                    detached(entry.block as usize)
                });
        }
        // A jump cache entry sits in its PC's slot alone.
        for &(_, pc) in &gone {
            let entry = &mut self.jump[(pc / INSTRUCTION) as usize % JUMP_CACHE];
            if entry.as_ref().is_some_and(|entry| detached(entry.block)) {
                *entry = None;
            }
        }
        unmark(&mut cache.marked, marked, slots);
    }

    pub(super) fn discard_code(&mut self, cause: Discard) {
        self.flush_exits();
        // A machine alone clears its records below: their exits are kept
        // by word. A board's records stay, and their blocks' retirement
        // adds them: kept here too they would count twice.
        let shared = self.code_hub.is_some() || Arc::get_mut(&mut self.code).is_none();
        if !shared {
            self.exit_words = self.exit_words();
        }
        let code = Arc::clone(&self.code);
        let counted =
            self.census.is_some() || code.cache_at(board::LockSite::Discard).census.is_some();
        if counted {
            // In bounded steps of slots, each under the lock.
            let now = self.host.counter();
            let slots = code.blocks.len();
            for step in (0..slots).step_by(EVICT_STEP) {
                let mut cache = code.cache_at(board::LockSite::Discard);
                for slot in step..slots.min(step + EVICT_STEP) {
                    if code
                        .blocks
                        .try_record(slot)
                        .is_some_and(|record| record.live())
                    {
                        let bytes = code
                            .blocks
                            .recovery()
                            .get(&slot)
                            .map_or(0, |owner| owner.code_bytes());
                        self.census_evicted(&mut cache, slot, bytes, now);
                    }
                }
            }
        }
        drop(code);
        self.host.discard_code();
        self.lookup_dispatchers = Default::default();
        self.direct_exit = None;
        // A worker built on the hub's code holds it before it joins.
        if self.code_hub.is_some() || Arc::get_mut(&mut self.code).is_none() {
            // Other PEs run the board's code: it becomes unreachable now
            // and goes as the ring reaches it.
            self.forget_shared_code();
            self.pages.clear();
            self.known_keys.fill(None);
            self.scalable_leaves.clear();
            self.code_generation += 1;
            self.statistics.code_discards[cause as usize] += 1;
            return;
        }
        // The block table goes with the code; the ITLB, which names no
        // block, keeps its translations. A running guest normally fills the
        // chunks again after a discard: their storage goes back when
        // freeing memory is the reason to discard.
        let free = matches!(cause, Discard::Memory | Discard::Finished);
        Arc::get_mut(&mut self.code)
            .expect("a machine that keeps its code alone discards it")
            .clear(free);
        if free {
            self.spare_pages = Vec::new();
        }
        self.pages.clear();
        self.known_keys.fill(None);
        self.scalable_leaves.clear();
        self.code_generation += 1;
        self.statistics.code_discards[cause as usize] += 1;
    }

    /// Discard every installed block, freeing its metadata, as a run that
    /// has ended does before work that needs the memory.
    pub fn discard_blocks(&mut self) {
        self.discard_code(Discard::Finished);
    }
}
