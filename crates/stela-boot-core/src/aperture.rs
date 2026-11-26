//! The rules the native MMIO aperture's owned mappings rely on: the processor
//! controls that make its uncached leaves uncached on Intel and AMD alike,
//! which host processors and roots may use it, how a retired translation
//! leaves every processor's TLB, and where a PCI BAR sits in the CPU's
//! physical space. Pure functions over observed register values and firmware
//! descriptions, and the shootdown's generation bookkeeping.
use core::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use stela_contracts::bytes::HOST_PAGE_BYTES as PAGE;

const CR0_WRITE_PROTECT: u64 = 1 << 16;
const CR4_LA57: u64 = 1 << 12;
const CR4_PCIDE: u64 = 1 << 17;
const EFER_NXE: u64 = 1 << 11;
/// The IA32_PAT memory type encodings (Intel SDM Vol. 3A 12.12.2, AMD APM
/// Vol. 2 7.8.1 give the same table).
const WRITE_BACK: u8 = 6;
const UNCACHED: u8 = 0;
/// The PAT entry a PTE with PWT and PCD set and PAT clear selects: the
/// aperture's leaves.
pub const LEAF_PAT_INDEX: u32 = 3;

/// One host processor's controls, read in the host's root.
#[derive(Clone, Copy, Debug)]
pub struct Controls {
    pub cr0: u64,
    pub cr3: u64,
    pub cr4: u64,
    pub efer: u64,
    pub pat: u64,
    /// The paging levels the loader built the window for, 4 or 5.
    pub levels: u32,
}

/// Why the controls cannot carry the aperture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// PAT entry 0 is not write-back, so the page tables' own
    /// write-back aliases are not.
    TablesNotWriteBack,
    /// The leaves' PAT entry is not uncached.
    LeavesNotUncached,
    /// Supervisor writes ignore read-only leaves without CR0.WP.
    WriteProtectOff,
    /// CR3 carries cache controls for the root without PCIDs.
    RootCacheControls,
    /// CR4.LA57 does not match the window's paging levels.
    PagingLevels,
    /// The leaves set NX, reserved without EFER.NXE.
    NoExecuteOff,
}

/// How a retired aperture translation leaves this processor's TLB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flush {
    /// INVLPG each page: without PCIDs it covers every root.
    Pages,
    /// Toggle CR4.PGE, which drops every PCID's entries (Intel SDM Vol. 3A
    /// 4.10.4.1, AMD APM Vol. 2 5.5.3): the loader root has a PCID of its own
    /// while the engine runs.
    AllContexts,
}

/// Check the controls the aperture needs and choose its TLB flush.
pub fn qualify(controls: &Controls) -> Result<Flush, Refusal> {
    let pat = |index: u32| (controls.pat >> (8 * index)) as u8;
    if pat(0) != WRITE_BACK {
        return Err(Refusal::TablesNotWriteBack);
    }
    if pat(LEAF_PAT_INDEX) != UNCACHED {
        return Err(Refusal::LeavesNotUncached);
    }
    if controls.cr0 & CR0_WRITE_PROTECT == 0 {
        return Err(Refusal::WriteProtectOff);
    }
    let tagged = controls.cr4 & CR4_PCIDE != 0;
    if !tagged && controls.cr3 & 0xfff != 0 {
        return Err(Refusal::RootCacheControls);
    }
    if (controls.cr4 & CR4_LA57 != 0) != (controls.levels == 5) {
        return Err(Refusal::PagingLevels);
    }
    if controls.efer & EFER_NXE == 0 {
        return Err(Refusal::NoExecuteOff);
    }
    Ok(if tagged {
        Flush::AllContexts
    } else {
        Flush::Pages
    })
}

/// Whether a root whose top-level entry for the aperture is `current`
/// reaches the aperture's leaves, given the loader root's `loader` entry:
/// every root copies the loader's window entries, so equal present entries
/// lead to the same tables.
pub fn shares_aperture(current: u64, loader: u64) -> bool {
    current & 1 != 0 && current == loader
}

/// The most processors that may share the aperture, MADT's 8-bit APIC IDs
/// and x2APIC's beyond them alike: the boot's processor cap.
pub const MAX_PROCESSORS: usize = 256;

/// How many recent requests keep the range they retired ([`Processors::plan`]).
pub const RECORDED: usize = 16;
/// The most pages a processor flushes one by one when it acknowledges; more
/// flush every translation.
pub const RANGE_PAGES: u64 = 16;
/// A recorded slot's generation while a request writes it.
const WRITING: u64 = u64::MAX;

/// What a processor does, for the report of a shootdown it holds up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Doing {
    /// Engine, guest or host work with interrupts off: it acknowledges at its
    /// next safe point.
    Masked = 0,
    /// A compile job with the shootdown vector open: it acknowledges in the
    /// vector's handler.
    Job = 1,
    /// A compile job with interrupts off.
    MaskedJob = 2,
    /// Waiting for its own shootdown, acknowledging others' as it waits.
    Retiring = 3,
}

impl Doing {
    fn of(bits: u64) -> Self {
        match bits & 3 {
            1 => Self::Job,
            2 => Self::MaskedJob,
            3 => Self::Retiring,
            _ => Self::Masked,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Masked => "masked",
            Self::Job => "job",
            Self::MaskedJob => "masked job",
            Self::Retiring => "retiring",
        }
    }
}

/// One joined processor, on a cache line of its own: retirements read every
/// one and each processor writes only its own.
#[derive(Debug)]
#[repr(align(64))]
struct Sharer {
    apic: AtomicU32,
    acknowledged: AtomicU64,
    /// [`Doing`] in the low two bits, the TSC it began at above them.
    doing: AtomicU64,
}

/// The range one request retired, by generation; read like a sequence
/// lock: valid only when its generation reads the same before and after.
#[derive(Debug)]
struct Recorded {
    generation: AtomicU64,
    start: AtomicU64,
    end: AtomicU64,
}

/// Up to [`RECORDED`] retired address ranges: what a processor flushes to
/// acknowledge a request ([`Processors::plan`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ranges {
    ranges: [(u64, u64); RECORDED],
    count: usize,
}

impl Ranges {
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = core::ops::Range<u64>> + '_ {
        self.ranges[..self.count]
            .iter()
            .map(|&(start, end)| start..end)
    }
}

/// Which host processors may hold aperture translations, and how far each
/// has flushed. Until an AP joins, only the boot processor runs and a
/// retirement's local flush is complete. After that a retirement
/// [`request`]s a generation, interrupts every [`sharer`], and waits until
/// each has [`acknowledge`]d it: a processor acknowledges a generation once
/// it has flushed every aperture translation the requests up to it retired
/// ([`plan`]). Installation needs no remote flush, since processors do not
/// cache not-present entries, and the permission's lock orders installations
/// and retirements.
///
/// [`request`]: Self::request
/// [`sharer`]: Self::sharers
/// [`acknowledge`]: Self::acknowledge
/// [`plan`]: Self::plan
#[derive(Debug)]
pub struct Processors {
    joined: AtomicUsize,
    requested: AtomicU64,
    sharers: [Sharer; MAX_PROCESSORS],
    recorded: [Recorded; RECORDED],
}

impl Default for Processors {
    fn default() -> Self {
        Self::new()
    }
}

impl Processors {
    pub const fn new() -> Self {
        Self {
            joined: AtomicUsize::new(0),
            requested: AtomicU64::new(0),
            sharers: [const {
                Sharer {
                    apic: AtomicU32::new(0),
                    acknowledged: AtomicU64::new(0),
                    doing: AtomicU64::new(0),
                }
            }; MAX_PROCESSORS],
            recorded: [const {
                Recorded {
                    generation: AtomicU64::new(0),
                    start: AtomicU64::new(0),
                    end: AtomicU64::new(0),
                }
            }; RECORDED],
        }
    }

    /// Register the installed BSP before admitting any AP.
    pub fn join_bsp(&self, apic: u32) -> Option<usize> {
        if self.joined.load(Ordering::Relaxed) != 0 {
            return None;
        }
        self.join(apic)
    }

    /// The BSP registers each AP before it can translate through the aperture.
    pub fn join_ap(&self, apic: u32) -> Option<usize> {
        if self.joined.load(Ordering::Relaxed) == 0 || self.slot(apic).is_some() {
            return None;
        }
        self.join(apic)
    }

    fn join(&self, apic: u32) -> Option<usize> {
        let slot = self.joined.load(Ordering::Relaxed);
        if slot == MAX_PROCESSORS {
            return None;
        }
        let sharer = &self.sharers[slot];
        sharer.apic.store(apic, Ordering::Relaxed);
        sharer
            .acknowledged
            .store(self.requested.load(Ordering::Acquire), Ordering::Relaxed);
        self.joined.store(slot + 1, Ordering::Release);
        Some(slot)
    }

    /// Count the joined processor with `apic` ID, an AP that failed to
    /// start and halted or never ran, as having acknowledged every flush:
    /// it takes no further aperture translation, so no retirement waits on
    /// it. It stays a sharer; an interrupt to it is harmless.
    pub fn exclude(&self, apic: u32) {
        if let Some(slot) = self.slot(apic) {
            self.sharers[slot]
                .acknowledged
                .store(u64::MAX, Ordering::Release);
        }
    }

    /// Whether only the boot processor may hold aperture translations.
    pub fn alone(&self) -> bool {
        self.joined.load(Ordering::Acquire) <= 1
    }

    /// The slot of the processor with `apic` ID.
    pub fn slot(&self, apic: u32) -> Option<usize> {
        (0..self.joined.load(Ordering::Acquire))
            .find(|&slot| self.sharers[slot].apic.load(Ordering::Relaxed) == apic)
    }

    /// Each joined processor's slot and APIC ID.
    pub fn sharers(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        (0..self.joined.load(Ordering::Acquire))
            .map(|slot| (slot, self.sharers[slot].apic.load(Ordering::Relaxed)))
    }

    /// Publish a flush request after the retired entries are cleared; the
    /// generation each sharer must acknowledge. `retired` is the address
    /// range whose entries were cleared, or `None` when they are not one
    /// range. Interrupts are sent after this returns, so a processor woken by
    /// one sees the request and, unless it looked before the range was
    /// recorded, the range.
    pub fn request(&self, retired: Option<core::ops::Range<u64>>) -> u64 {
        let generation = self.requested.fetch_add(1, Ordering::AcqRel) + 1;
        if let Some(range) = retired {
            self.record(generation, range);
        }
        generation
    }

    /// Keep `range` for `generation` in its slot, unless a request for a
    /// later generation holds or writes the slot: a reader that finds no
    /// matching generation flushes everything.
    fn record(&self, generation: u64, range: core::ops::Range<u64>) {
        let recorded = &self.recorded[generation as usize % RECORDED];
        let current = recorded.generation.load(Ordering::Relaxed);
        if current == WRITING
            || current > generation
            || recorded
                .generation
                .compare_exchange(current, WRITING, Ordering::Relaxed, Ordering::Relaxed)
                .is_err()
        {
            return;
        }
        // A reader that sees either value below sees WRITING or a later
        // generation on its second look.
        core::sync::atomic::fence(Ordering::Release);
        recorded.start.store(range.start, Ordering::Relaxed);
        recorded.end.store(range.end, Ordering::Relaxed);
        recorded.generation.store(generation, Ordering::Release);
    }

    /// The range `generation` retired, when it is still recorded.
    fn recorded(&self, generation: u64) -> Option<(u64, u64)> {
        let recorded = &self.recorded[generation as usize % RECORDED];
        if recorded.generation.load(Ordering::Acquire) != generation {
            return None;
        }
        let start = recorded.start.load(Ordering::Relaxed);
        let end = recorded.end.load(Ordering::Relaxed);
        core::sync::atomic::fence(Ordering::Acquire);
        (recorded.generation.load(Ordering::Relaxed) == generation).then_some((start, end))
    }

    /// What a processor that flushed through `flushed` flushes to
    /// acknowledge `requested`: the ranges of every request after `flushed`
    /// (none once it is there) when each is recorded and they hold
    /// [`RANGE_PAGES`] pages or fewer; `None`, everything, when it is too
    /// far behind, a request has no range or they hold more. Generations
    /// count up from 1 and never wrap: a 64-bit count of requests would take
    /// centuries to.
    pub fn plan(&self, flushed: u64, requested: u64) -> Option<Ranges> {
        let mut ranges = Ranges {
            ranges: [(0, 0); RECORDED],
            count: 0,
        };
        if requested <= flushed {
            return Some(ranges);
        }
        if requested - flushed > RECORDED as u64 {
            return None;
        }
        let mut pages = 0u64;
        for generation in flushed + 1..=requested {
            let (start, end) = self.recorded(generation)?;
            pages = pages.saturating_add(end.saturating_sub(start).div_ceil(PAGE));
            if pages > RANGE_PAGES {
                return None;
            }
            ranges.ranges[ranges.count] = (start, end);
            ranges.count += 1;
        }
        Some(ranges)
    }

    /// Note what slot `slot`'s processor does from TSC `now` on; what it did.
    pub fn set_doing(&self, slot: usize, doing: Doing, now: u64) -> Doing {
        self.sharers.get(slot).map_or(Doing::Masked, |sharer| {
            Doing::of(
                sharer
                    .doing
                    .swap(now << 2 | doing as u64, Ordering::Relaxed),
            )
        })
    }

    /// What slot `slot`'s processor does, and the TSC it began at.
    pub fn doing(&self, slot: usize) -> (Doing, u64) {
        self.sharers.get(slot).map_or((Doing::Masked, 0), |sharer| {
            let bits = sharer.doing.load(Ordering::Relaxed);
            (Doing::of(bits), bits >> 2)
        })
    }

    /// The latest requested generation.
    pub fn requested(&self) -> u64 {
        self.requested.load(Ordering::Acquire)
    }

    /// Slot `slot`'s processor flushed every translation older than
    /// `generation`'s request.
    pub fn acknowledge(&self, slot: usize, generation: u64) {
        if let Some(sharer) = self.sharers.get(slot) {
            sharer.acknowledged.fetch_max(generation, Ordering::Release);
        }
    }

    /// The generation slot `slot`'s processor last acknowledged.
    pub fn acknowledged(&self, slot: usize) -> u64 {
        self.sharers
            .get(slot)
            .map_or(0, |sharer| sharer.acknowledged.load(Ordering::Acquire))
    }

    /// Whether every sharer but `except` (the retiring processor's slot)
    /// acknowledged `generation`.
    pub fn flushed(&self, generation: u64, except: Option<usize>) -> bool {
        self.sharers()
            .all(|(slot, _)| Some(slot) == except || self.acknowledged(slot) >= generation)
    }
}

/// A root bridge's memory window as firmware describes it: bus addresses
/// `minimum..=maximum`, which the CPU reaches at `translation` above them.
#[derive(Clone, Copy, Debug)]
pub struct BusWindow {
    pub segment: u32,
    pub minimum: u64,
    pub maximum: u64,
    pub translation: u64,
}

/// The CPU physical address of a memory BAR at bus address `address` for
/// `bytes` on `segment`, when one root bridge window holds all of it.
pub fn bar_physical(
    windows: impl IntoIterator<Item = BusWindow>,
    segment: u32,
    address: u64,
    bytes: u64,
) -> Option<u64> {
    let last = address.checked_add(bytes.checked_sub(1)?)?;
    windows
        .into_iter()
        .find(|window| {
            window.segment == segment && window.minimum <= address && last <= window.maximum
        })
        .and_then(|window| address.checked_add(window.translation))
}

/// Translate a small BAR and the one complete CPU page that contains it.
/// Both the BAR and the entire page must lie inside the same bridge window.
pub fn subpage_bar(
    windows: impl IntoIterator<Item = BusWindow>,
    segment: u32,
    address: u64,
    bytes: u64,
) -> Option<(u64, usize)> {
    if !(1..PAGE).contains(&bytes) {
        return None;
    }
    for window in windows {
        let Some(physical) = bar_physical([window], segment, address, bytes) else {
            continue;
        };
        let offset = physical % PAGE;
        if offset + bytes > PAGE {
            return None;
        }
        let bus_page = address.checked_sub(offset)?;
        let physical_page = bar_physical([window], segment, bus_page, PAGE)?;
        return Some((physical_page, offset as usize));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAT_DEFAULT: u64 = 0x0007_0406_0007_0406;

    #[test]
    fn subpage_bar_requires_one_complete_translated_window_page() {
        let window = BusWindow {
            segment: 2,
            minimum: 0x1000,
            maximum: 0x2fff,
            translation: 0xfc00_0000,
        };
        assert_eq!(
            subpage_bar([window], 2, 0x1c00, 0x400),
            Some((0xfc00_1000, 0xc00))
        );
        assert_eq!(subpage_bar([window], 2, 0x1c00, 0x800), None);
        assert_eq!(subpage_bar([window], 0, 0x1c00, 0x400), None);
        let partial = BusWindow {
            minimum: 0x1c00,
            ..window
        };
        assert_eq!(subpage_bar([partial], 2, 0x1c00, 0x400), None);
    }

    fn controls() -> Controls {
        Controls {
            cr0: 0x8001_0033,
            cr3: 0x1000,
            cr4: 0x20,
            efer: 0xd01,
            pat: PAT_DEFAULT,
            levels: 4,
        }
    }

    #[test]
    fn the_power_on_pat_carries_the_aperture_on_either_vendor() {
        assert_eq!(qualify(&controls()), Ok(Flush::Pages));
        let five = Controls {
            cr4: 0x20 | CR4_LA57,
            levels: 5,
            ..controls()
        };
        assert_eq!(qualify(&five), Ok(Flush::Pages));
    }

    #[test]
    fn pcids_flush_every_context_and_allow_tagged_roots() {
        let tagged = Controls {
            cr3: 0x1001,
            cr4: 0x20 | CR4_PCIDE,
            ..controls()
        };
        assert_eq!(qualify(&tagged), Ok(Flush::AllContexts));
        let untagged = Controls {
            cr3: 0x1018,
            ..controls()
        };
        assert_eq!(qualify(&untagged), Err(Refusal::RootCacheControls));
    }

    #[test]
    fn controls_the_leaves_depend_on_refuse() {
        for (changed, refusal) in [
            (
                Controls {
                    pat: PAT_DEFAULT & !0xff | 1,
                    ..controls()
                },
                Refusal::TablesNotWriteBack,
            ),
            (
                Controls {
                    pat: PAT_DEFAULT | 7 << 24,
                    ..controls()
                },
                Refusal::LeavesNotUncached,
            ),
            (
                Controls {
                    cr0: 0x8000_0033,
                    ..controls()
                },
                Refusal::WriteProtectOff,
            ),
            (
                Controls {
                    levels: 5,
                    ..controls()
                },
                Refusal::PagingLevels,
            ),
            (
                Controls {
                    efer: 0x501,
                    ..controls()
                },
                Refusal::NoExecuteOff,
            ),
        ] {
            assert_eq!(qualify(&changed), Err(refusal));
        }
    }

    #[test]
    fn only_the_loader_roots_present_entry_reaches_the_aperture() {
        assert!(shares_aperture(0x5003, 0x5003));
        assert!(!shares_aperture(0x5003, 0x6003));
        assert!(!shares_aperture(0, 0));
    }

    #[test]
    fn a_retirement_waits_for_every_sharer_but_the_retiring_one() {
        let processors = Processors::new();
        assert!(processors.alone());
        assert!(processors.flushed(processors.request(None), None));
        let a = processors.join_bsp(4).unwrap();
        assert!(processors.alone());
        let b = processors.join_ap(9).unwrap();
        assert!(!processors.alone());
        assert_eq!(processors.slot(9), Some(b));
        assert_eq!(processors.slot(5), None);
        // A processor that joins has seen every earlier request.
        assert!(processors.flushed(processors.requested(), None));
        let generation = processors.request(None);
        assert!(!processors.flushed(generation, Some(b)));
        processors.acknowledge(a, generation);
        assert!(processors.flushed(generation, Some(b)));
        assert!(!processors.flushed(generation, None));
        processors.acknowledge(b, generation);
        processors.acknowledge(b, generation - 1);
        assert_eq!(processors.acknowledged(b), generation);
        assert!(processors.flushed(generation, None));
        assert_eq!(
            processors.sharers().collect::<alloc::vec::Vec<_>>(),
            [(a, 4), (b, 9)]
        );
    }

    #[test]
    fn an_excluded_ap_never_holds_up_a_retirement() {
        let processors = Processors::new();
        let bsp = processors.join_bsp(1).unwrap();
        let ready = processors.join_ap(2).unwrap();
        processors.join_ap(3).unwrap();
        processors.exclude(3);
        // An APIC ID that never joined changes nothing.
        processors.exclude(9);
        let generation = processors.request(None);
        processors.acknowledge(bsp, generation);
        assert!(!processors.flushed(generation, None));
        processors.acknowledge(ready, generation);
        assert!(processors.flushed(generation, None));
        assert_eq!(processors.sharers().count(), 3);
    }

    #[test]
    fn bsp_registration_precedes_aps_and_rejects_duplicate_identities() {
        let processors = Processors::new();
        assert_eq!(processors.join_ap(2), None);
        assert_eq!(processors.join_bsp(0), Some(0));
        assert!(processors.alone());
        assert_eq!(processors.join_bsp(1), None);
        assert_eq!(processors.join_ap(0), None);
        assert_eq!(processors.join_ap(2), Some(1));
        assert!(!processors.alone());
        assert_eq!(processors.join_ap(2), None);
    }

    #[test]
    fn bsp_and_ap_retirements_require_the_other_origin_to_acknowledge() {
        let processors = Processors::new();
        let bsp = processors.join_bsp(0).unwrap();
        let ap = processors.join_ap(2).unwrap();
        let from_bsp = processors.request(None);
        let from_ap = processors.request(None);
        assert!(!processors.flushed(from_bsp, Some(bsp)));
        assert!(!processors.flushed(from_ap, Some(ap)));
        processors.acknowledge(ap, from_ap);
        assert!(processors.flushed(from_bsp, Some(bsp)));
        assert!(!processors.flushed(from_ap, Some(ap)));
        processors.acknowledge(bsp, from_bsp);
        assert!(!processors.flushed(from_ap, Some(ap)));
        processors.acknowledge(bsp, from_ap);
        assert!(processors.flushed(from_ap, Some(ap)));
    }

    /// The planned ranges as (start, end) pairs.
    fn ranges(plan: Option<Ranges>) -> alloc::vec::Vec<(u64, u64)> {
        plan.expect("a plan of ranges")
            .iter()
            .map(|range| (range.start, range.end))
            .collect()
    }

    fn nothing(plan: Option<Ranges>) -> bool {
        plan.is_some_and(|ranges| ranges.is_empty())
    }

    /// Every joined processor but the retiring one is waited for, whatever
    /// it does; it acknowledges once it flushed the planned ranges.
    #[test]
    fn a_retirement_waits_for_every_other_sharer_whatever_it_does() {
        let processors = Processors::new();
        let bsp = processors.join_bsp(0).unwrap();
        let ap = processors.join_ap(2).unwrap();
        assert_eq!(processors.set_doing(ap, Doing::Job, 77), Doing::Masked);
        assert_eq!(processors.doing(ap), (Doing::Job, 77));
        let generation = processors.request(Some(0x4000..0x6000));
        // A processor in a job holds translations like any other.
        assert!(!processors.flushed(generation, Some(bsp)));
        assert_eq!(
            ranges(processors.plan(processors.acknowledged(ap), generation)),
            [(0x4000, 0x6000)]
        );
        processors.acknowledge(ap, generation);
        assert!(processors.flushed(generation, Some(bsp)));
        assert_eq!(processors.set_doing(ap, Doing::Masked, 90), Doing::Job);
        assert!(nothing(processors.plan(generation, generation)));
    }

    /// A processor behind several requests flushes each one's range, and
    /// everything once they hold more than RANGE_PAGES pages.
    #[test]
    fn plans_cover_every_request_since_the_last_flush() {
        let processors = Processors::new();
        let first = processors.request(Some(0x1000..0x3000));
        let second = processors.request(Some(0x9000..0xa000));
        assert_eq!(
            ranges(processors.plan(0, second)),
            [(0x1000, 0x3000), (0x9000, 0xa000)]
        );
        assert_eq!(ranges(processors.plan(first, second)), [(0x9000, 0xa000)]);
        let large = processors.request(Some(0..(RANGE_PAGES + 1) * PAGE));
        assert_eq!(processors.plan(second, large), None);
        // Small ones that add up past the bound flush everything too.
        let start = processors.requested();
        let mut last = start;
        for index in 0..RANGE_PAGES {
            last = processors.request(Some(index * 2 * PAGE..(index * 2 + 1) * PAGE));
        }
        assert_eq!(
            ranges(processors.plan(start, last)).len(),
            RANGE_PAGES as usize
        );
        let more = processors.request(Some(0x100_0000..0x100_1000));
        assert_eq!(processors.plan(start, more), None);
        assert_eq!(
            ranges(processors.plan(last, more)),
            [(0x100_0000, 0x100_1000)]
        );
    }

    /// A request with no range, and one whose slot another request is
    /// writing, flush everything.
    #[test]
    fn unrecorded_requests_flush_everything() {
        let processors = Processors::new();
        let ranged = processors.request(Some(0x1000..0x2000));
        let unranged = processors.request(None);
        assert_eq!(processors.plan(0, unranged), None);
        assert_eq!(ranges(processors.plan(0, ranged)), [(0x1000, 0x2000)]);
        let next = processors.request(Some(0x5000..0x6000));
        processors.recorded[next as usize % RECORDED]
            .generation
            .store(WRITING, Ordering::Relaxed);
        assert_eq!(processors.plan(unranged, next), None);
        // A request that finds its slot being written leaves it alone.
        let later = next + RECORDED as u64;
        processors.requested.store(later - 1, Ordering::Relaxed);
        assert_eq!(processors.request(Some(0x7000..0x8000)), later);
        assert_eq!(processors.plan(later - 1, later), None);
    }

    /// The record wraps after RECORDED requests: an old generation's slot
    /// holds a newer one, so a processor that far behind flushes everything,
    /// and a late writer never overwrites a newer record.
    #[test]
    fn recorded_slots_wrap_without_reuse() {
        let processors = Processors::new();
        let mut generations = alloc::vec::Vec::new();
        for index in 0..RECORDED as u64 * 3 {
            generations.push(processors.request(Some(index * PAGE..(index + 1) * PAGE)));
        }
        let last = *generations.last().unwrap();
        assert_eq!(processors.plan(0, last), None);
        assert_eq!(processors.plan(last - RECORDED as u64 - 1, last), None);
        let recent = ranges(processors.plan(last - 4, last));
        let expected: alloc::vec::Vec<_> = (last - 3..=last)
            .map(|generation| ((generation - 1) * PAGE, generation * PAGE))
            .collect();
        assert_eq!(recent, expected);
        // A request for an older generation that records late is ignored.
        processors.record(last - RECORDED as u64, 0xdead_0000..0xdead_1000);
        assert_eq!(
            ranges(processors.plan(last - 1, last)),
            [((last - 1) * PAGE, last * PAGE)]
        );
        assert!(nothing(processors.plan(last, last)));
    }

    #[test]
    fn each_sharer_has_a_cache_line_of_its_own() {
        assert_eq!(core::mem::align_of::<Sharer>(), 64);
        assert_eq!(core::mem::size_of::<Sharer>(), 64);
    }

    #[test]
    fn joins_stop_at_the_processor_cap() {
        let processors = Processors::new();
        assert_eq!(processors.join_bsp(0), Some(0));
        for apic in 1..MAX_PROCESSORS as u32 {
            assert!(processors.join_ap(apic).is_some());
        }
        assert_eq!(processors.join_ap(1000), None);
    }

    #[test]
    fn bars_map_through_the_window_that_holds_them() {
        let windows = [
            BusWindow {
                segment: 0,
                minimum: 0x8000_0000,
                maximum: 0x8fff_ffff,
                translation: 0,
            },
            BusWindow {
                segment: 1,
                minimum: 0x8000_0000,
                maximum: 0x8fff_ffff,
                translation: 0x10_0000_0000,
            },
        ];
        assert_eq!(
            bar_physical(windows, 0, 0x8000_4000, 0x4000),
            Some(0x8000_4000)
        );
        assert_eq!(
            bar_physical(windows, 1, 0x8000_4000, 0x4000),
            Some(0x10_8000_4000)
        );
        assert_eq!(bar_physical(windows, 0, 0x8fff_f000, 0x2000), None);
        assert_eq!(bar_physical(windows, 2, 0x8000_0000, 0x1000), None);
        assert_eq!(bar_physical(windows, 0, 0x8000_0000, 0), None);
    }
}
