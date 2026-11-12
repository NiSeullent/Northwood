//! Why blocks are installed, for `--profile`'s heartbeat: a block this
//! machine evicted within the last ten seconds (thrash), one another
//! machine has installed at that moment (elsewhere), or one this machine
//! never installed before (first). The tables are bounded and keyed by a
//! hash of the block's PC, frame address and variant, so the counts are
//! close, not exact: a slot keeps the latest block hashed to it. They exist
//! only while step timing is on: 1 MiB per machine and 2 MiB per hub, the
//! same on every host.

use alloc::{boxed::Box, vec};
use core::hash::{Hash, Hasher};
use core::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

/// Evictions a machine remembers, by hash slot.
const EVICTED: usize = 1 << 16;
/// Bits of a machine's filter of the blocks it installed (two per block).
const SEEN_BITS: usize = 1 << 22;
/// Installed blocks of every machine, by hash slot.
const LIVE: usize = 1 << 18;
/// Counter ticks per unit of a remembered eviction's time.
const TIME_SHIFT: u32 = 10;

/// What a machine's installs and evictions came to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InstallCauses {
    /// Blocks installed, and of them those this machine evicted within
    /// ten seconds, those installed on another machine at the time, and
    /// those it never installed before. The last three count only while
    /// step timing (`--profile`) is on.
    pub installs: u64,
    pub thrash: u64,
    pub elsewhere: u64,
    pub first: u64,
    /// Code bytes of the blocks installed and of the blocks evicted.
    pub installed_bytes: u64,
    pub evicted_bytes: u64,
    /// Guest instructions of the blocks installed: with `installed_bytes`,
    /// host code bytes per guest instruction. And of those, the code bytes
    /// and guest instructions of blocks in the upper half of the address
    /// space (TTBR1: the kernel and the guarded monitors).
    pub guest_instructions: u64,
    pub upper_bytes: u64,
    pub upper_instructions: u64,
    /// Counter ticks spent evicting ([`super::Machine::evict_code`]).
    pub evict_ticks: u64,
}

/// FNV-1a, for a block's identity.
struct Fnv(u64);

impl Hasher for Fnv {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
    }
}

/// The hash of a block's identity, the same on every machine: its PC, the
/// address its code was read from and what it was compiled for.
pub(super) fn identity(pc: u64, physical: u64, variant: &impl Hash) -> u64 {
    let mut hasher = Fnv(0xcbf2_9ce4_8422_2325);
    pc.hash(&mut hasher);
    physical.hash(&mut hasher);
    variant.hash(&mut hasher);
    // Spread the low bits FNV leaves weak.
    let hash = hasher.finish();
    (hash ^ hash >> 29).wrapping_mul(0xbf58_476d_1ce4_e5b9)
}

/// A nonzero tag of `identity`, apart from the bits that pick its slot.
fn tag(identity: u64) -> u32 {
    (identity >> 32) as u32 | 1
}

/// One machine's memory of what it installed and evicted.
pub(super) struct Census {
    /// Per slot: the tag of the block evicted last and when, in
    /// [`TIME_SHIFT`] units.
    evicted: Box<[u64]>,
    seen: Box<[u64]>,
}

impl Census {
    pub(super) fn new() -> Self {
        Self {
            evicted: vec![0; EVICTED].into_boxed_slice(),
            seen: vec![0; SEEN_BITS / 64].into_boxed_slice(),
        }
    }

    /// Note an install of `identity` at counter `now`: whether this
    /// machine evicted it within `window` ticks, and whether it is the
    /// first install of it here.
    pub(super) fn installed(&mut self, identity: u64, now: u64, window: u64) -> (bool, bool) {
        let slot = &mut self.evicted[identity as usize % EVICTED];
        let held = *slot;
        let thrash = (held >> 32) as u32 == tag(identity)
            && (now >> TIME_SHIFT).wrapping_sub(held & 0xffff_ffff) & 0xffff_ffff
                <= window >> TIME_SHIFT;
        if thrash {
            *slot = 0;
        }
        let mut first = false;
        for bits in [identity, identity.rotate_left(23)] {
            let bit = (bits >> 11) as usize % SEEN_BITS;
            let word = &mut self.seen[bit / 64];
            first |= *word & 1 << (bit % 64) == 0;
            *word |= 1 << (bit % 64);
        }
        (thrash, first)
    }

    /// Note an eviction of `identity` at counter `now`.
    pub(super) fn evicted(&mut self, identity: u64, now: u64) {
        self.evicted[identity as usize % EVICTED] =
            u64::from(tag(identity)) << 32 | (now >> TIME_SHIFT) & 0xffff_ffff;
    }
}

/// Which machines have each block installed, a bit per PE (modulo 32) in
/// a slot shared by every machine of a [`super::Hub`], allocated at the
/// first use.
#[derive(Default)]
pub struct Live {
    table: AtomicPtr<AtomicU64>,
}

impl Live {
    fn table(&self) -> &[AtomicU64] {
        let mut table = self.table.load(Ordering::Acquire);
        if table.is_null() {
            let made: Box<[AtomicU64]> = (0..LIVE).map(|_| AtomicU64::new(0)).collect();
            let made = Box::into_raw(made).cast::<AtomicU64>();
            table = match self.table.compare_exchange(
                core::ptr::null_mut(),
                made,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => made,
                Err(other) => {
                    // SAFETY: `made` came from `Box::into_raw` of `LIVE`
                    // entries above and was never shared.
                    drop(unsafe { Box::from_raw(core::ptr::slice_from_raw_parts_mut(made, LIVE)) });
                    other
                }
            };
        }
        // SAFETY: a non-null table is `LIVE` entries this owner frees only
        // when dropped.
        unsafe { core::slice::from_raw_parts(table, LIVE) }
    }

    /// Note that PE `pe` installed `identity`: whether another PE had it
    /// installed.
    pub(super) fn install(&self, identity: u64, pe: usize) -> bool {
        let slot = &self.table()[(identity >> 7) as usize % LIVE];
        let (tag, bit) = (u64::from(tag(identity)) << 32, 1u64 << (pe % 32));
        let mut held = slot.load(Ordering::Relaxed);
        loop {
            let (elsewhere, new) = if held & !0xffff_ffff == tag {
                (held & 0xffff_ffff & !bit != 0, held | bit)
            } else {
                (false, tag | bit)
            };
            match slot.compare_exchange_weak(held, new, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return elsewhere,
                Err(now) => held = now,
            }
        }
    }

    /// Note that PE `pe` evicted `identity`.
    pub(super) fn evict(&self, identity: u64, pe: usize) {
        let slot = &self.table()[(identity >> 7) as usize % LIVE];
        let (tag, bit) = (u64::from(tag(identity)) << 32, 1u64 << (pe % 32));
        let mut held = slot.load(Ordering::Relaxed);
        while held & !0xffff_ffff == tag && held & bit != 0 {
            match slot.compare_exchange_weak(
                held,
                held & !bit,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(now) => held = now,
            }
        }
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        let table = *self.table.get_mut();
        if !table.is_null() {
            // SAFETY: as in `Live::table`; no reference outlives `&mut self`.
            drop(unsafe { Box::from_raw(core::ptr::slice_from_raw_parts_mut(table, LIVE)) });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_eviction_is_thrash_for_a_window_and_a_block_is_first_once() {
        let mut census = Census::new();
        let block = identity(0x1000, 0x8000_1000, &3u8);
        let window = 10 << TIME_SHIFT;
        assert_eq!(census.installed(block, 0, window), (false, true));
        census.evicted(block, 100 << TIME_SHIFT);
        assert_eq!(
            census.installed(block, 105 << TIME_SHIFT, window),
            (true, false)
        );
        census.evicted(block, 200 << TIME_SHIFT);
        assert_eq!(
            census.installed(block, 300 << TIME_SHIFT, window),
            (false, false)
        );
        // Another variant of the same PC and frame is another block.
        let other = identity(0x1000, 0x8000_1000, &4u8);
        assert_ne!(other, block);
        assert!(census.installed(other, 300 << TIME_SHIFT, window).1);
    }

    #[test]
    fn a_block_another_pe_holds_counts_as_elsewhere_until_it_evicts_it() {
        let live = Live::default();
        let block = identity(0x2000, 0x9000_2000, &0u8);
        assert!(!live.install(block, 0));
        assert!(live.install(block, 1));
        live.evict(block, 0);
        assert!(!live.install(block, 1));
        live.evict(block, 1);
        assert!(!live.install(block, 2));
    }
}
