//! Block translations from and to the persistent translation store
//! (`crate::translations`): a fetch page is read and hashed once, its
//! record looked up once, and each block of it compiled in a context the
//! record has is rebased instead of compiled. A block the store's process
//! already prepared at the same PC (another thread's, or ahead-of-time
//! translation's) is installed as it is. A block compiled anew is offered
//! back, with what the host needs to compile it again at the probe
//! addresses (`translations::prove`).
use super::recovery::StoredOrigin;
use super::{Devices, FETCH_PAGE, Host, Machine, Preparation, Variant};
use crate::translations::{Digest, PAGE_BYTES, Record};
use alloc::{boxed::Box, sync::Arc, vec::Vec};
use stela_compiler::block::CompiledBlock;

/// A block as compiled, with its complete admitted input and the
/// preparation that compiled it: immutable once made, so the store and every
/// PE that installs it share one. Its fault recovery rows are qualified.
#[derive(Debug, Clone)]
pub(super) struct Emission {
    pub(super) input: Vec<u8>,
    pub(super) block: CompiledBlock,
    pub(super) preparation: Arc<Preparation>,
}
impl Emission {
    pub(super) fn new(
        mut input: Vec<u8>,
        mut block: CompiledBlock,
        preparation: Arc<Preparation>,
    ) -> Self {
        // The store keeps an emission until it proves it, sharing its bound
        // by capacity: none of it is slack.
        input.shrink_to_fit();
        block.native.x86_64_code.shrink_to_fit();
        block.native.source_map.shrink_to_fit();
        block.direct_exits.shrink_to_fit();
        block.fault_recovery.shrink_to_fit();
        Self {
            input,
            block,
            preparation,
        }
    }
}

/// A same-process prepared block together with its complete admitted input.
/// The page key and context remain independent lookup bindings.
#[derive(Debug)]
pub struct SharedTranslation {
    pub(super) emission: Arc<Emission>,
    key: Digest,
    page: u64,
    pc: u64,
    context: Digest,
}
impl SharedTranslation {
    pub fn input(&self) -> &[u8] {
        &self.emission.input
    }
    pub fn block(&self) -> &CompiledBlock {
        &self.emission.block
    }
    pub fn preparation(&self) -> &Arc<Preparation> {
        &self.emission.preparation
    }
    pub(super) fn matches(&self, key: &Digest, pc: u64, context: &Digest, source: &[u8]) -> bool {
        self.key == *key
            && self.pc == pc
            && self.context == *context
            && self.page == pc & !(PAGE_BYTES as u64 - 1)
            && self.preparation().context(self.page) == *context
            && source.get(..self.input().len()) == Some(self.input())
            && self.block().source_bytes <= self.input().len()
            && self
                .block()
                .native
                .source_map
                .first()
                .is_some_and(|(at, _)| at.0 == pc)
    }
}

/// The Arc payloads/headers and every privately owned vector allocation. This
/// excludes allocator bookkeeping and shared immutable preparation storage.
fn translation_bytes(emission: &Emission) -> usize {
    use core::mem::size_of;
    let block = &emission.block;
    let bytes = [
        size_of::<SharedTranslation>() + size_of::<Emission>() + 4 * size_of::<usize>(),
        emission.input.capacity(),
        block.native.x86_64_code.capacity(),
        block.native.source_map.capacity() * size_of::<(stela_compiler::GuestAddress, usize)>(),
        block.direct_exits.capacity() * size_of::<stela_compiler::x86_64::chain::DirectExit>(),
        block.fault_recovery.capacity() * size_of::<stela_compiler::block::FaultRecovery>(),
    ];
    block.fault_recovery.iter().fold(
        bytes.into_iter().fold(0usize, usize::saturating_add),
        |total, row| {
            total.saturating_add(
                row.bindings.capacity() * size_of::<stela_compiler::block::RecoveryBinding>(),
            )
        },
    )
}

/// The store a host keeps for the engine.
pub trait Translations {
    /// The record of the page whose bytes hash to `key`, if the store has
    /// a valid one.
    fn record(&mut self, key: &Digest) -> Option<Arc<Record>>;
    /// The block at `pc` of the page whose bytes hash to `key`, prepared
    /// in `context` at that very PC since the record was read, if the
    /// store has one: shared, so a store other machines reach copies none
    /// of it while they wait. A store may index blocks by part of `key` and
    /// `context`; the caller [`SharedTranslation::matches`] what it gets.
    fn prepared(
        &mut self,
        _key: &Digest,
        _pc: u64,
        _context: &Digest,
    ) -> Option<Arc<SharedTranslation>> {
        None
    }
    /// A block the engine compiled from a page the store had no entry for.
    fn offer(&mut self, offer: Offer);
    /// A block for the other PEs alone, which no pack keeps.
    fn lend(&mut self, _offer: Offer) {}
    /// What a machine's use of the store cost or found, for the store's
    /// counters.
    fn note(&mut self, _event: StoreEvent) {}
    /// The fetch pages the machines sharing this store keep in all, when the
    /// store sizes them ([`KEPT_PAGES`] otherwise).
    fn kept_pages(&mut self) -> Option<usize> {
        None
    }
}

/// A machine's use of its store, reported to it ([`Translations::note`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreEvent {
    /// A fetch page was read and hashed for its record, and its record
    /// looked up, in `ticks` of [`Host::counter`].
    PageRead { ticks: u64 },
    /// A shared block was found but not installed: its admitted bytes or
    /// source extent differ from the fetch's.
    SharedRejected,
}

/// A block compiled anew, and how it was compiled. Emission fields are private;
/// only read-only provenance and the sealed same-process handoff are exposed.
///
/// ```compile_fail
/// fn replace_source(offer: &mut stela_engine::machine::Offer) {
///     offer.emission.input.clear();
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Offer {
    pub(super) key: Digest,
    /// The guest address of its fetch page.
    pub(super) page: u64,
    pub(super) pc: u64,
    pub(super) context: Digest,
    /// The block, the admitted words the compiler was given and how it
    /// compiled them, shared with the PE that installed it.
    pub(super) emission: Arc<Emission>,
    /// The entry PCs of its static successors, for translation ahead of
    /// execution.
    pub(super) successors: Vec<u64>,
    /// Compiled for EL0, where the PE holds no lock of the guest's kernel:
    /// a store may take time there for work of its own.
    pub(super) quiet: bool,
}

impl Offer {
    pub(crate) fn retained_bytes(&self) -> usize {
        translation_bytes(&self.emission)
    }
    /// Off-path preparation from canonical immutable input. No native template
    /// or copied site can be supplied; this owner emits the block itself.
    #[cfg(test)]
    pub(crate) fn compile(
        key: Digest,
        page: u64,
        pc: u64,
        context: Digest,
        input: Vec<u8>,
        preparation: Arc<Preparation>,
    ) -> Option<Self> {
        if page != pc & !(PAGE_BYTES as u64 - 1) || preparation.context(page) != context {
            return None;
        }
        let block = preparation.compile_input(&input, pc).ok()?;
        Some(Self {
            key,
            page,
            pc,
            context,
            emission: Arc::new(Emission::new(input, block, preparation)),
            successors: Vec::new(),
            quiet: false,
        })
    }
    pub fn key(&self) -> Digest {
        self.key
    }
    pub fn page(&self) -> u64 {
        self.page
    }
    pub fn pc(&self) -> u64 {
        self.pc
    }
    pub fn context(&self) -> Digest {
        self.context
    }
    pub fn input(&self) -> &[u8] {
        &self.emission.input
    }
    pub fn preparation(&self) -> &Arc<Preparation> {
        &self.emission.preparation
    }
    pub fn successors(&self) -> &[u64] {
        &self.successors
    }
    /// Whether it was compiled at EL0, where the PE holds no kernel lock.
    pub fn quiet(&self) -> bool {
        self.quiet
    }
    pub fn block(&self) -> &CompiledBlock {
        &self.emission.block
    }

    /// Keep this same-process emission and its complete private
    /// source/profile/template binding. No factory accepts portable site tags.
    ///
    /// ```compile_fail
    /// fn substitute(shared: &mut stela_engine::machine::engine::SharedTranslation) {
    ///     shared.emission.block.fault_recovery.clear();
    /// }
    /// ```
    pub fn shared(self) -> Arc<SharedTranslation> {
        Arc::new(SharedTranslation {
            emission: self.emission,
            key: self.key,
            page: self.page,
            pc: self.pc,
            context: self.context,
        })
    }

    /// Writable stores keep the original offer for proof admission. The
    /// shared cache does not need the offer's separate successor list.
    pub(crate) fn shared_for_proof(&self) -> Arc<SharedTranslation> {
        Arc::new(SharedTranslation {
            emission: Arc::clone(&self.emission),
            key: self.key,
            page: self.page,
            pc: self.pc,
            context: self.context,
        })
    }

    /// Its block's stored form: the block compiled again at both probe
    /// addresses of its page, proven to move only by them
    /// ([`translations::prove`](crate::translations::prove)).
    pub fn prove(&self) -> Option<crate::translations::Entry> {
        Self::prove_block(
            self.block(),
            self.input(),
            self.page,
            self.pc,
            self.context,
            self.preparation(),
        )
    }

    /// A pending source window owns no installed emission. Compile it for
    /// proof without building an owned offer for other PEs.
    pub(crate) fn prove_input(
        input: &[u8],
        page: u64,
        pc: u64,
        context: Digest,
        preparation: &Preparation,
    ) -> Option<crate::translations::Entry> {
        if page != pc & !(PAGE_BYTES as u64 - 1) || preparation.context(page) != context {
            return None;
        }
        let block = preparation.compile_input(input, pc).ok()?;
        Self::prove_block(&block, input, page, pc, context, preparation)
    }

    pub(super) fn prove_block(
        block: &CompiledBlock,
        input: &[u8],
        page: u64,
        pc: u64,
        context: Digest,
        preparation: &Preparation,
    ) -> Option<crate::translations::Entry> {
        let probes = crate::translations::probe_displacements(page).map(|displacement| {
            preparation
                .compile_input(input, pc.wrapping_add(displacement))
                .ok()
        });
        let [Some(first), Some(second)] = &probes else {
            return None;
        };
        crate::translations::prove(block, input, page, pc, context, [first, second])
    }
}

/// The fetch pages kept for their blocks at most; when full, the least
/// recently used quarter is let go. A page let go is read, hashed and looked
/// up again for its next block, which only a block not yet installed needs.
/// Low memory leaves them: the heap rests at the mark it calls low, and these
/// few MiB are worth more than the time hashing every compile's page costs.
#[cfg(not(test))]
pub(super) const KEPT_PAGES: usize = 4096;
#[cfg(test)]
pub(super) const KEPT_PAGES: usize = 4;

/// Known page keys per kept page: a page let go keeps its key this much
/// longer, at a few dozen bytes instead of its 4 KiB.
const KNOWN_PER_KEPT: usize = 8;

/// The page key of a fetch page read before, and a check of the bytes it
/// was the hash of ([`page_check`]): a page read again whose bytes give the
/// same check takes this key instead of hashing them.
#[derive(Clone, Copy)]
pub(super) struct KnownKey {
    frame: u64,
    check: u64,
    key: Digest,
}

/// A 64-bit check of a fetch page: four lanes, each step of which is a
/// bijection of the lane, so a page that differs in one word always
/// checks differently; other changes collide with a chance of 2^-64. A
/// collision costs no correctness: it names the old page's record, whose
/// entries are installed only when their input equals the fetch's bytes,
/// and an offer under that key is a record entry no fetch of the old page
/// matches.
pub(super) fn page_check(bytes: &[u8; PAGE_BYTES]) -> u64 {
    const MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut lanes: [u64; 4] = [1, 2, 3, 4];
    for chunk in bytes.as_chunks::<32>().0 {
        for (lane, word) in lanes.iter_mut().zip(chunk.as_chunks::<8>().0) {
            let word = u64::from_le_bytes(*word);
            *lane = (*lane ^ word).wrapping_mul(MULTIPLIER).rotate_left(29);
        }
    }
    lanes.iter().enumerate().fold(0, |check, (at, &lane)| {
        let mixed = (lane ^ lane >> 31).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        check ^ mixed.rotate_left(at as u32 * 16)
    })
}

/// A fetch page's bytes, as read for its first block, and its record.
pub(super) struct StoredPage {
    pub(super) key: Digest,
    pub(super) bytes: Box<[u8; PAGE_BYTES]>,
    pub(super) record: Option<Arc<Record>>,
    /// The latest installed block's, which the next of this page shares.
    origin: Option<Arc<StoredOrigin>>,
    /// `Machine::page_uses` at its last use.
    used: u64,
}

impl<H: Host, D: Devices> Machine<H, D> {
    /// The stored page holding `pc` at `physical`, read and looked up on
    /// its first block; `None` without a store, when the page cannot be
    /// read whole or when a host breakpoint or trace point lies in it,
    /// which shortens blocks in ways no context names.
    ///
    /// `source` is the block's window as just read: a page whose kept
    /// bytes differ from it (guest memory written without the IC
    /// maintenance that invalidates the page) is read and hashed again, so
    /// a stored or offered block always has the bytes the guest holds now.
    pub(super) fn stored_page(
        &mut self,
        pc: u64,
        physical: u64,
        source: &[u8],
    ) -> Option<(Digest, Option<Arc<Record>>)> {
        let frame = physical & !(FETCH_PAGE - 1);
        let offset = (physical - frame) as usize;
        let matches = |bytes: &[u8]| bytes.get(offset..offset + source.len()) == Some(source);
        self.page_uses += 1;
        if let Some(stored) = self.pages.get_mut(&frame) {
            if matches(stored.bytes.as_slice()) {
                stored.used = self.page_uses;
                return Some((stored.key, stored.record.clone()));
            }
            self.forget_stored_page(frame);
        }
        self.host.translations()?;
        let page = pc & !(FETCH_PAGE - 1);
        if self
            .breakpoints
            .iter()
            .chain(&self.traces)
            .any(|&stop| stop & !(FETCH_PAGE - 1) == page)
        {
            return None;
        }
        let reading = self.host.counter();
        // A read fills the whole buffer or fails.
        let mut bytes = self
            .spare_pages
            .pop()
            .unwrap_or_else(|| Box::new([0u8; PAGE_BYTES]));
        if !self.host.read_ram(frame, bytes.as_mut_slice()) || !matches(bytes.as_slice()) {
            self.spare_pages.push(bytes);
            return None;
        }
        let key = self.page_key(frame, &bytes);
        let record = self.host.translations()?.record(&key);
        let ticks = self.host.counter().wrapping_sub(reading);
        if let Some(translations) = self.host.translations() {
            translations.note(StoreEvent::PageRead { ticks });
        }
        if self.pages.len() >= self.kept_pages {
            self.forget_least_used_pages();
        }
        self.pages.insert(
            frame,
            StoredPage {
                key,
                bytes,
                record: record.clone(),
                origin: None,
                used: self.page_uses,
            },
        );
        #[cfg(feature = "native-profile")]
        {
            self.stored_pages_peak = self.stored_pages_peak.max(self.pages.len());
        }
        Some((key, record))
    }

    /// The page key of `bytes`, the fetch page at `frame`: the key it had
    /// when read last, while its bytes give the same check, else their
    /// SHA-256, which is kept for the next read.
    fn page_key(&mut self, frame: u64, bytes: &[u8; PAGE_BYTES]) -> Digest {
        let check = page_check(bytes);
        let known = (self.kept_pages * KNOWN_PER_KEPT).next_power_of_two();
        if self.known_keys.len() != known {
            self.known_keys = alloc::vec![None; known];
        }
        let slot = &mut self.known_keys[(frame / FETCH_PAGE) as usize % known];
        if let Some(held) = slot
            && held.frame == frame
            && held.check == check
        {
            self.statistics.known_keys += 1;
            return held.key;
        }
        let key = Digest::of(bytes.as_slice());
        *slot = Some(KnownKey { frame, check, key });
        key
    }

    /// Forget the known key of the page at `frame`, whose code an IC IVAU
    /// made stale.
    pub(super) fn forget_known_key(&mut self, frame: u64) {
        let known = self.known_keys.len();
        if known != 0 {
            let slot = &mut self.known_keys[(frame / FETCH_PAGE) as usize % known];
            if slot.is_some_and(|held| held.frame == frame) {
                *slot = None;
            }
        }
    }

    /// What a block of the record `key`, in the fetch page at `physical`
    /// and virtual `page`, keeps of it.
    pub(super) fn stored_origin(
        &mut self,
        physical: u64,
        page: u64,
        key: Digest,
        context: Digest,
        preparation: &Arc<Preparation>,
    ) -> Arc<StoredOrigin> {
        let kept = self.pages.get_mut(&(physical & !(FETCH_PAGE - 1)));
        match kept {
            Some(StoredPage {
                origin: Some(origin),
                ..
            }) if origin.page == page
                && origin.kept.is_none()
                && origin.context == context
                && Arc::ptr_eq(&origin.preparation, preparation) =>
            {
                origin.clone()
            }
            kept => {
                let origin = Arc::new(StoredOrigin {
                    key,
                    context,
                    page,
                    preparation: preparation.clone(),
                    kept: None,
                });
                if let Some(kept) = kept {
                    kept.origin = Some(origin.clone());
                }
                origin
            }
        }
    }

    /// The record `key`, from the page at `physical` if it is still kept,
    /// else decoded again from the pack.
    pub(super) fn stored_record(&mut self, physical: u64, key: &Digest) -> Option<Arc<Record>> {
        let frame = physical & !(FETCH_PAGE - 1);
        if let Some(StoredPage {
            key: kept,
            record: Some(record),
            ..
        }) = self.pages.get(&frame)
            && kept == key
        {
            return Some(record.clone());
        }
        self.host.translations()?.record(key)
    }

    /// Current, configured and lifetime peak retained source pages of this owner.
    #[cfg(feature = "native-profile")]
    pub fn stored_page_occupancy(&self) -> (usize, usize, usize) {
        (self.pages.len(), self.kept_pages, self.stored_pages_peak)
    }

    /// Keep the store's kept pages ([`Translations::kept_pages`], else
    /// `KEPT_PAGES`) across `machines` machines, which each keep their own:
    /// an equal share each, but no fewer than a sixteenth of `KEPT_PAGES`.
    pub(super) fn share_kept_pages(&mut self, machines: usize) {
        let total = self
            .host
            .translations()
            .and_then(|translations| translations.kept_pages())
            .unwrap_or(KEPT_PAGES);
        self.kept_pages = (total / machines.max(1)).max(KEPT_PAGES.div_ceil(16));
    }

    /// Let the least recently used quarter of the kept pages go, at least
    /// one. Their buffers stay for the pages read next
    /// ([`Self::forget_stored_page`]).
    fn forget_least_used_pages(&mut self) {
        let mut uses: Vec<u64> = self.pages.values().map(|page| page.used).collect();
        let quarter = (uses.len() / 4).max(1);
        let least_kept = if quarter == uses.len() {
            u64::MAX
        } else {
            *uses.select_nth_unstable(quarter).1
        };
        let gone: Vec<u64> = self
            .pages
            .iter()
            .filter(|(_, page)| page.used < least_kept)
            .map(|(&frame, _)| frame)
            .collect();
        for frame in gone {
            self.forget_stored_page(frame);
        }
    }

    /// Forget the kept page at `frame`, whose buffer the next page read
    /// takes: as many stay spare as [`Self::forget_least_used_pages`] lets
    /// go at once.
    pub(super) fn forget_stored_page(&mut self, frame: u64) {
        if let Some(page) = self.pages.remove(&frame)
            && self.spare_pages.len() < self.kept_pages.div_ceil(4)
        {
            self.spare_pages.push(page.bytes);
        }
    }

    /// The entry context of blocks compiled for `variant` through
    /// `preparation` in the page at `page` ([`Preparation::context`]).
    pub(super) fn stored_context(
        &mut self,
        variant: Variant,
        preparation: &Preparation,
        page: u64,
    ) -> Digest {
        let class = crate::translations::address_class(page);
        let variant = Variant {
            guarded_page: false,
            ..variant
        };
        *self
            .stored_contexts
            .entry((variant, class))
            .or_insert_with(|| preparation.context(page))
    }
}

#[cfg(test)]
pub(super) mod tests;
