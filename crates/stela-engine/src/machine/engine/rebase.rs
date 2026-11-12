//! In-boot reuse of a block at another virtual address: the same page
//! content mapped at a new address on each exec (a main program, the loader)
//! is rebased from a proven entry instead of compiled again.
//!
//! The board keeps, by (page key, entry offset, entry context), what its
//! PEs compiled there: first only the PC it was compiled at (`Seen`). A
//! compile of the same key, offset and context at another PC proves the
//! block ([`crate::translations::prove`], the pack's proof: the block
//! compiled again at both probe addresses) and keeps the proven entry; later
//! fetches at any page of that content relocate it
//! ([`Entry::relocate`]), with the checks the pack's entries pass: the
//! admitted input equal to the fetch's bytes, a page-aligned rebase and
//! recovery rows that rebase.
//!
//! Keys are content keys (the SHA-256 of the whole fetch page) and the
//! context digest names every compile input, so a rewritten frame hashes to
//! another key and never reaches an entry of its old bytes; entries of
//! content no frame holds any more leave with the bound. Memory is bounded:
//! when full, the oldest keys go first.
use crate::platform::sync::Mutex;
use crate::translations::{Digest, Entry};
use alloc::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
};
use core::sync::atomic::{AtomicU64, Ordering};

/// The bytes all parts keep at most.
#[cfg(not(test))]
pub(super) const BOUND: usize = 48 << 20;
#[cfg(test)]
pub(super) const BOUND: usize = 1 << 20;
/// Parts by the page key's first byte, each under its own lock.
const PARTS: usize = 8;

type Key = (Digest, u16, Digest);

/// What a key, its slot and its place in the order cost, apart from a
/// proven entry's buffers.
const KEY_BYTES: usize = 2 * core::mem::size_of::<(Key, Slot, u64)>()
    + core::mem::size_of::<(Key, u64)>()
    + 4 * core::mem::size_of::<usize>();

#[derive(Clone)]
enum Slot {
    /// Compiled at this PC only so far.
    Seen(u64),
    Proven(Arc<Entry>),
    /// Its proof failed: it is compiled at every address.
    Unprovable,
}

impl Slot {
    fn bytes(&self) -> usize {
        KEY_BYTES
            + match self {
                Self::Proven(entry) => entry.retained_bytes(),
                _ => 0,
            }
    }
}

#[derive(Default)]
struct Part {
    slots: BTreeMap<Key, (Slot, u64)>,
    /// Keys in the order they were kept, with the stamp they were kept at:
    /// a key kept again later has a newer stamp, and its older place is
    /// skipped.
    order: VecDeque<(Key, u64)>,
    bytes: usize,
    stamp: u64,
}

impl Part {
    fn put(&mut self, key: Key, slot: Slot, bound: usize) {
        let bytes = slot.bytes();
        if bytes > bound {
            return;
        }
        if let Some((old, _)) = self.slots.remove(&key) {
            self.bytes -= old.bytes();
        }
        while self.bytes + bytes > bound {
            let Some((oldest, stamp)) = self.order.pop_front() else {
                break;
            };
            if self
                .slots
                .get(&oldest)
                .is_some_and(|(_, kept)| *kept == stamp)
                && let Some((gone, _)) = self.slots.remove(&oldest)
            {
                self.bytes -= gone.bytes();
            }
        }
        self.compact();
        self.stamp += 1;
        self.order.push_back((key, self.stamp));
        self.slots.insert(key, (slot, self.stamp));
        self.bytes += bytes;
    }

    /// Move `key`, which is kept, to the newest end of the order: an entry
    /// that keeps serving stays while those not used go.
    fn touch(&mut self, key: &Key) {
        self.compact();
        self.stamp += 1;
        let stamp = self.stamp;
        if let Some((_, kept)) = self.slots.get_mut(key) {
            *kept = stamp;
            self.order.push_back((*key, stamp));
        }
    }

    /// Drop the places of keys kept or used again since: they stay until
    /// they come up, at most about one more per key, which its charge
    /// covers.
    fn compact(&mut self) {
        if self.order.len() > 2 * self.slots.len() + 16 {
            let slots = &self.slots;
            self.order
                .retain(|(key, stamp)| slots.get(key).is_some_and(|(_, kept)| kept == stamp));
        }
    }
}

/// What a fetch does with a block the store keys.
pub(super) enum Found {
    /// Relocate this entry.
    Entry(Arc<Entry>),
    /// Compile it, then prove it ([`Rebases::keep`]): it was compiled at
    /// another PC before.
    Prove,
    /// Compile it.
    Compile,
}

/// Reuse hits and proofs, for `PES CODE`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RebaseCounts {
    /// Installs relocated from a proven entry.
    pub hits: u64,
    /// Entries proven, and proofs that failed.
    pub proven: u64,
    pub unprovable: u64,
    /// The bytes kept now.
    pub bytes: usize,
}

/// The board's proven entries, by page key, offset and context.
pub(super) struct Rebases {
    parts: [Mutex<Part>; PARTS],
    bound: usize,
    hits: AtomicU64,
    proven: AtomicU64,
    unprovable: AtomicU64,
}

impl Rebases {
    pub(super) fn new(bound: usize) -> Self {
        Self {
            parts: core::array::from_fn(|_| Mutex::new(Part::default())),
            bound: bound / PARTS,
            hits: AtomicU64::new(0),
            proven: AtomicU64::new(0),
            unprovable: AtomicU64::new(0),
        }
    }

    fn part(&self, key: &Digest) -> crate::platform::sync::MutexGuard<'_, Part> {
        self.parts[usize::from(key.0[0]) % PARTS]
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    /// What a fetch at `pc` of the block at `offset` of the page whose
    /// bytes hash to `key`, compiled in `context`, does. A key not known
    /// yet is noted as compiled at `pc`.
    pub(super) fn look(&self, key: &Digest, offset: u16, context: &Digest, pc: u64) -> Found {
        let slot = (*key, offset, *context);
        let mut part = self.part(key);
        match part.slots.get(&slot) {
            Some((Slot::Proven(entry), _)) => {
                let entry = Arc::clone(entry);
                part.touch(&slot);
                Found::Entry(entry)
            }
            Some((Slot::Seen(seen), _)) if *seen != pc => Found::Prove,
            Some(_) => Found::Compile,
            None => {
                part.put(slot, Slot::Seen(pc), self.bound);
                Found::Compile
            }
        }
    }

    /// Keep the proof of the block at `offset` of `key` in `context`: its
    /// entry, or `None` when it moves by other than its address.
    pub(super) fn keep(&self, key: &Digest, offset: u16, context: &Digest, entry: Option<Entry>) {
        let slot = match entry {
            Some(entry) => {
                self.proven.fetch_add(1, Ordering::Relaxed);
                Slot::Proven(Arc::new(entry))
            }
            None => {
                self.unprovable.fetch_add(1, Ordering::Relaxed);
                Slot::Unprovable
            }
        };
        self.part(key)
            .put((*key, offset, *context), slot, self.bound);
    }

    /// Count an install relocated from a proven entry.
    pub(super) fn hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn counts(&self) -> RebaseCounts {
        RebaseCounts {
            hits: self.hits.load(Ordering::Relaxed),
            proven: self.proven.load(Ordering::Relaxed),
            unprovable: self.unprovable.load(Ordering::Relaxed),
            bytes: (0..PARTS)
                .map(|at| {
                    self.parts[at]
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .bytes
                })
                .sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(offset: u16, code: usize) -> Entry {
        Entry {
            context: Digest::default(),
            offset,
            source_bytes: 4,
            input: alloc::vec![0; 4],
            code: alloc::vec![0; code],
            relocations: alloc::vec![],
            source_map: alloc::vec![(offset, 0)],
            exits: alloc::vec![],
            recovery: alloc::vec![],
        }
    }

    #[test]
    fn a_second_pc_proves_and_a_proven_entry_serves_every_pc() {
        let rebases = Rebases::new(BOUND);
        let (key, context) = (Digest([1; 32]), Digest([2; 32]));
        assert!(matches!(
            rebases.look(&key, 8, &context, 0x1008),
            Found::Compile
        ));
        // The same PC again (its block went) compiles; another proves.
        assert!(matches!(
            rebases.look(&key, 8, &context, 0x1008),
            Found::Compile
        ));
        assert!(matches!(
            rebases.look(&key, 8, &context, 0x5008),
            Found::Prove
        ));
        rebases.keep(&key, 8, &context, Some(entry(8, 64)));
        for pc in [0x1008, 0x5008, 0x9008] {
            assert!(matches!(
                rebases.look(&key, 8, &context, pc),
                Found::Entry(_)
            ));
        }
        // Another context or offset is another block.
        assert!(matches!(
            rebases.look(&key, 8, &Digest([3; 32]), 0x9008),
            Found::Compile
        ));
        assert!(matches!(
            rebases.look(&key, 12, &context, 0x900c),
            Found::Compile
        ));
        rebases.keep(&key, 12, &context, None);
        assert!(matches!(
            rebases.look(&key, 12, &context, 0xd00c),
            Found::Compile
        ));
        let counts = rebases.counts();
        assert_eq!((counts.proven, counts.unprovable), (1, 1));
    }

    #[test]
    fn the_oldest_keys_go_when_the_bound_is_full() {
        // One part's share holds about ten keys with a small entry each.
        let one = Slot::Proven(Arc::new(entry(0, 256))).bytes();
        let rebases = Rebases::new(PARTS * one * 10);
        let context = Digest::default();
        // Keys of one part (the same first byte).
        let key = |n: u8| {
            let mut bytes = [0; 32];
            bytes[1] = n;
            Digest(bytes)
        };
        for n in 0..40 {
            rebases.keep(&key(n), 0, &context, Some(entry(0, 256)));
            assert!(rebases.counts().bytes <= PARTS * one * 10 / PARTS);
        }
        assert!(matches!(
            rebases.look(&key(39), 0, &context, 0),
            Found::Entry(_)
        ));
        assert!(matches!(
            rebases.look(&key(0), 0, &context, 0),
            Found::Compile
        ));
        // An entry in use stays while newer keys come.
        for n in 40..80 {
            assert!(matches!(
                rebases.look(&key(38), 0, &context, 0),
                Found::Entry(_)
            ));
            rebases.keep(&key(n), 0, &context, Some(entry(0, 256)));
        }
        assert!(matches!(
            rebases.look(&key(39), 0, &context, 0),
            Found::Compile
        ));
        // Keys kept or used again leave no unbounded order behind.
        for _ in 0..1000 {
            rebases.keep(&key(79), 0, &context, Some(entry(0, 256)));
            let _ = rebases.look(&key(79), 0, &context, 0);
        }
        let part = rebases.part(&key(0));
        assert!(part.order.len() <= 2 * part.slots.len() + 17);
    }
}
