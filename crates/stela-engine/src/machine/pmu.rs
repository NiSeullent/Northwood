//! Apple's per-PE CPMU fixed cycle and retired-instruction counters, with
//! separately checked M1–M6 layouts (`layout`). XNU xnu-12377.121.6 H13
//! supplies M1's 48-bit width, H14–H17 supply M2–M5's 64-bit width, and
//! original 26A434 mac18g supplies M6's distinct 15-counter mapping.
//!
//! - PMC0 counts cycles and PMC1 retired instructions. M1/M2 also count
//!   source-checked programmable events (`events`): NONE and active
//!   cycles on PMC2–9, retired instructions on PMC7. Other enabled pairs
//!   refuse; raw counter writes are independent of event selection.
//!   Counters wrap at their configured width, with bit47 (M1) or bit63
//!   (M2–M6) as overflow; PMSR uses the selected layout's slot mapping.
//! - PMCR0 enables counting (bits 0-7, 32-33 for PMC8 and PMC9), selects the
//!   PMI (bits 8-10, 4 for FIQ), reports it (bit 11, PMAI), enables each
//!   counter's PMI (bits 12-19, 44-45), PC capture for PMC5-7 (bits 24-26)
//!   and EL0 access (bit 30).
//! - PMCR1 filters counting by mode: bits 8-15 and 40-41 at EL0, bits 16-23
//!   and 48-49 in the kernel. XNU, which runs at EL2 on the M2, calls bits
//!   16 and 17 the EL2 enables, so in the EL2&0 host they count at EL2
//!   (inference). Which PMCR1 applies follows where the PE runs
//!   ([`Filter`]): `S3_1_C15_C7_2` is PMCR1_EL12 (Linux
//!   `arch/arm64/include/asm/apple_m1_pmu.h`), the EL1&0 regime's copy, which
//!   XNU writes with PMCR1's value before PMCR0 under `CONFIG_EXCLAVES ||
//!   HYPERVISOR` (`osfmk/arm64/cpc_arm64.c`) and which, like the VHE `_EL12`
//!   registers, is UNDEFINED below EL2 and is what PMCR1 reaches in the EL1&0
//!   regime (inference from the name); a guarded
//!   level counts by PMCR1_GL1 (`S3_1_C15_C8_2`, m1n1 `apple_regs.json`),
//!   whose kernel field applies, which SPTM 26A434 writes once per PE at
//!   boot with 0x30300 or a full enable pattern and the kernel saves and
//!   restores around a VM (its storage is `machine::impdef`'s entry, which
//!   `Pmu::set_guarded_filter` mirrors).
//!
//! This machine counts cycles at the PE's core clock while it is not waiting
//! in WFI or WFE, and instructions as the engine retires them: each
//! instruction that completes, and SVC, HVC and SMC, which complete as they
//! take their exception; an instruction that faults or traps is not
//! counted. The core clock is the PE's cluster's highest operating point
//! (`stela_soc_data::board::soc`): the engine runs a PE no slower for a lower
//! DVFS state, so it counts as a core held at its top state. Each PE has its
//! own monitor, so a thread that moves between PEs is charged, as on
//! hardware, by the counters of the PE it ran on. A PMI is asserted, as FIQ, while PMCR0 selects FIQ and a
//! counter with its PMI enabled has its overflow bit set; PMAI reads that
//! assertion (inference: XNU's FIQ handler reads PMAI to recognize a PMI,
//! then disables the monitor and clears every overflow by rewriting the
//! counters). The reset values are not documented; they are taken as 0,
//! under which nothing counts before software programs the monitor.
//!
//! Unqualified programmable events, PC capture, EL0 access, other PMI modes,
//! instruction matching (OPMAT, OPMSK) and the PMCR2-4 controls are not
//! modelled: a write that enables one stops. PMESR0 and PMESR1 store
//! selectors within the layout's field masks. An enabled event/slot must
//! have checked semantics; unsupported reconfiguration refuses atomically. M6's new controls refuse every access until their effects
//! are checked.
use stela_contracts::Chip;

mod events;
mod layout;
use events::Event;
use layout::Layout;
use stela_compiler::aarch64::system::registers::Encoding;

const fn monitor(op1: u8, crm: u8, op2: u8) -> Encoding {
    Encoding::new(3, op1, 15, crm, op2)
}

pub const PMCR0: Encoding = monitor(1, 0, 0);
pub const PMCR1: Encoding = monitor(1, 1, 0);
pub const PMCR2: Encoding = monitor(1, 2, 0);
pub const PMCR3: Encoding = monitor(1, 3, 0);
pub const PMCR4: Encoding = monitor(1, 4, 0);
pub const PMESR0: Encoding = monitor(1, 5, 0);
pub const PMESR1: Encoding = monitor(1, 6, 0);
pub const OPMAT0: Encoding = monitor(1, 7, 0);
pub const OPMAT1: Encoding = monitor(1, 8, 0);
pub const OPMSK0: Encoding = monitor(1, 9, 0);
pub const OPMSK1: Encoding = monitor(1, 10, 0);
/// The two controls 26A434 XNU's kpc saves after OPMSK1 (`kpc_get_config`,
/// 0xfffffe000bd41050) and restores with the others (0xfffffe000bd8db14):
/// XNU's `kpc_get_rawpmu_config` saves PM_MEMFLT_CTL23_EL1 and
/// PM_MEMFLT_CTL45_EL1 there, in that order (`xnu-12377.121.6`
/// `osfmk/arm64/kpc.c`), by symbolic name, so the encodings' match is an
/// inference. They configure nothing at 0.
pub const PM_MEMFLT_CTL23: Encoding = monitor(1, 0, 4);
pub const PM_MEMFLT_CTL45: Encoding = monitor(1, 1, 4);
/// The M5 kpc's tenth raw configuration register: its `kpc_get_config`
/// saves it after PM_MEMFLT_CTL45 and its restore writes it back
/// (kernelcache.release.mac17g 0xfffffe000be64104, 0xfffffe000beb56b0), a
/// slot public XNU does not have. No public name; it configures nothing at
/// 0 (inference, as for the others). The M6 layout leaves it unchecked.
pub const KPC_C5_1: Encoding = monitor(1, 5, 1);
/// PMCR1's EL1&0 copy ([`Filter::Guest`]).
pub const PMCR1_EL12: Encoding = monitor(1, 7, 2);
pub const PMSR: Encoding = monitor(1, 13, 0);
/// The PC captured for a PMC5-7 PMI (`S3_1_C15_C14_1`).
pub const PC_CAPTURE: Encoding = monitor(1, 14, 1);
/// PMC0 to PMC9: CRm 0-7, then 9 and 10 (H14 lacks
/// `CPMU_S3_2_C15_C9_09_ADDRESS`).
pub const PMC: [Encoding; COUNTERS] = [
    monitor(2, 0, 0),
    monitor(2, 1, 0),
    monitor(2, 2, 0),
    monitor(2, 3, 0),
    monitor(2, 4, 0),
    monitor(2, 5, 0),
    monitor(2, 6, 0),
    monitor(2, 7, 0),
    monitor(2, 9, 0),
    monitor(2, 10, 0),
];
/// The registers that store a configuration and accept only 0, which
/// configures nothing, in [`Pmu::extension`] order.
const EXTENSION: [Encoding; EXTENSIONS] = [
    PMCR2,
    PMCR3,
    PMCR4,
    OPMAT0,
    OPMAT1,
    OPMSK0,
    OPMSK1,
    PM_MEMFLT_CTL23,
    PM_MEMFLT_CTL45,
];
const EXTENSIONS: usize = 9;

pub const COUNTERS: usize = 10;
const CYCLES: usize = 0;
const INSTRUCTIONS: usize = 1;
#[cfg(test)]
const OVERFLOW: u64 = 1 << 63;

/// Counter geometry, separately checked from event selectors and clocks.
#[derive(Debug, Clone, Copy)]
enum Width {
    Bits48,
    Bits64,
}

impl Width {
    fn bits(self) -> u32 {
        match self {
            Self::Bits48 => 48,
            Self::Bits64 => 64,
        }
    }

    fn mask(self) -> u64 {
        u64::MAX >> (64 - self.bits())
    }

    fn overflow(self) -> u64 {
        1 << (self.bits() - 1)
    }
}

const PMCR0_COUNT: u64 = 0xff | 0b11 << 32;
const PMCR0_INTGEN: u64 = 0b111 << 8;
const PMCR0_INTGEN_FIQ: u64 = 4 << 8;
const PMCR0_PMAI: u64 = 1 << 11;
const PMCR0_PMI: u64 = 0xff << 12 | 0b11 << 44;
const PMCR0_PC_CAPTURE: u64 = 0b111 << 24;
const PMCR0_USER_ACCESS: u64 = 1 << 30;
const PMCR0_KNOWN: u64 =
    PMCR0_COUNT | PMCR0_INTGEN | PMCR0_PMAI | PMCR0_PMI | PMCR0_PC_CAPTURE | PMCR0_USER_ACCESS;
/// The counters this machine counts for: PMC0 and PMC1.
const PMCR1_KNOWN: u64 = 0xff << 8 | 0xff << 16 | 0b11 << 40 | 0b11 << 48;

/// Bit `n` of a PMCR0 or PMCR1 field whose PMC8 and PMC9 bits sit at
/// `high` instead of following PMC7's.
const fn counter_bit(low: u32, high: u32, n: usize) -> u64 {
    if n < 8 {
        1 << (low + n as u32)
    } else {
        1 << (high + n as u32 - 8)
    }
}

/// Where the PE runs, for counting: its EL, the PMCR1 that filters there
/// and whether it waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Context {
    pub el: u8,
    pub filter: Filter,
    pub waiting: bool,
}

/// Which PMCR1 filters counting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    /// PMCR1, in the EL2&0 regime.
    Host,
    /// PMCR1_EL12, in the EL1&0 regime.
    Guest,
    /// PMCR1_GL1's kernel field, at a guarded level.
    Guarded,
}

/// The register an access to `encoding` reaches: PMCR1 in the EL1&0
/// regime is that regime's copy, PMCR1_EL12, as the VHE `_EL1` encodings
/// reach the EL1 registers below EL2 (inference from the name).
fn regime_copy(encoding: Encoding, context: Context) -> Encoding {
    if encoding == PMCR1 && context.filter == Filter::Guest {
        PMCR1_EL12
    } else {
        encoding
    }
}

/// One PE's CPMU.
#[derive(Debug, Clone)]
pub struct Pmu {
    layout: Layout,
    /// The core clock, in Hz, and the system counter frequency.
    clock: u64,
    frequency: u64,
    pmcr0: u64,
    pmcr1: u64,
    pmcr1_el12: u64,
    /// PMCR1_GL1, mirrored from its `machine::impdef` entry.
    pmcr1_gl1: u64,
    pmesr: [Option<u64>; 2],
    extension: [Option<u64>; EXTENSIONS],
    /// [`KPC_C5_1`], on a layout that has it.
    kpc_c5_1: Option<u64>,
    /// The counts as of `synced_count` and `synced_retired`.
    counters: [u64; layout::MAX_COUNTERS],
    synced_count: u64,
    synced_retired: u64,
    /// Each cycle counter's fraction, in 1/`frequency` cycle units.
    remainder: [u64; layout::MAX_COUNTERS],
    context: Context,
    /// The system count and retired instruction count at which the PMI
    /// asserts, while the context and controls hold.
    cycle_deadline: Option<u64>,
    instruction_deadline: Option<u64>,
    /// The PMI is asserted as of the last synchronization.
    asserted: bool,
    /// Every counter counts alike in every context but a wait, as of the
    /// last update: a change of EL or guarded level changes no count.
    alike: bool,
}

impl Pmu {
    /// Preserve the original M2 monitor API.
    pub fn new(clock: u64, frequency: u64) -> Self {
        Self::with_layout(Layout::on(Chip::AppleM2), clock, frequency)
    }

    /// Construct only a source-checked core monitor, with a core clock
    /// supplied by its PE profile. A generation name supplies no clock or
    /// event layout. A zero clock or system frequency cannot construct a
    /// monitor.
    pub fn for_chip(chip: Chip, clock: u64, frequency: u64) -> Option<Self> {
        if clock == 0 || frequency == 0 {
            return None;
        }
        Some(Self::with_layout(Layout::on(chip), clock, frequency))
    }

    fn with_layout(layout: Layout, clock: u64, frequency: u64) -> Self {
        Self {
            layout,
            clock,
            frequency,
            pmcr0: 0,
            pmcr1: 0,
            pmcr1_el12: 0,
            pmcr1_gl1: 0,
            pmesr: [None; 2],
            extension: [None; EXTENSIONS],
            kpc_c5_1: None,
            counters: [0; layout::MAX_COUNTERS],
            synced_count: 0,
            synced_retired: 0,
            remainder: [0; layout::MAX_COUNTERS],
            context: Context {
                el: 2,
                filter: Filter::Host,
                waiting: false,
            },
            cycle_deadline: None,
            instruction_deadline: None,
            asserted: false,
            alike: true,
        }
    }

    /// All declared chips have a separately checked fixed CPMU layout.
    /// This says nothing about events, extra controls or a board's clocks.
    pub fn on(chip: Chip) -> bool {
        let _ = Layout::on(chip);
        true
    }

    /// Preserve the original ten-counter allocation query (M1/M2).
    pub fn owns(encoding: Encoding) -> bool {
        Self::owns_on(Chip::AppleM2, encoding)
    }

    /// The selected chip's allocation, including explicitly unchecked
    /// controls whose effects must refuse, and its exact counter aliases.
    pub fn owns_on(chip: Chip, encoding: Encoding) -> bool {
        let layout = Layout::on(chip);
        Self::common(encoding)
            || layout.counters.contains(&encoding)
            || layout.unqualified_controls.contains(&encoding)
            || layout.kpc_c5_1 && encoding == KPC_C5_1
    }

    pub fn counter_count(&self) -> usize {
        self.layout.counters.len()
    }

    pub fn counter_encoding(&self, slot: usize) -> Option<Encoding> {
        self.layout.counters.get(slot).copied()
    }

    fn common(encoding: Encoding) -> bool {
        matches!(
            encoding,
            PMCR0 | PMCR1 | PMESR0 | PMESR1 | PMCR1_EL12 | PMSR | PC_CAPTURE
        ) || EXTENSION.contains(&encoding)
    }

    fn check_access(&self, encoding: Encoding) -> Result<(), &'static str> {
        if self.layout.unqualified_controls.contains(&encoding) {
            Err("M6 CPMU control effects are not source-qualified")
        } else if Self::common(encoding)
            || self.layout.counters.contains(&encoding)
            || self.layout.kpc_c5_1 && encoding == KPC_C5_1
        {
            Ok(())
        } else {
            Err("register is not allocated in the selected CPMU layout")
        }
    }

    fn event_with(&self, n: usize, selectors: &[Option<u64>; 2]) -> Option<Event> {
        match n {
            CYCLES => Some(Event::Cycles),
            INSTRUCTIONS => Some(Event::Instructions),
            _ => self.layout.events.programmable(n, selectors),
        }
    }

    fn event(&self, n: usize) -> Option<Event> {
        self.event_with(n, &self.pmesr)
    }

    fn validate_events(
        &self,
        control: u64,
        selectors: &[Option<u64>; 2],
    ) -> Result<(), &'static str> {
        for n in 2..COUNTERS {
            if control & counter_bit(0, 32, n) != 0 && self.event_with(n, selectors).is_none() {
                return Err("programmable CPMU event/slot pair is not source-qualified");
            }
        }
        Ok(())
    }

    /// Whether counter `n` counts in `context`.
    fn counts(&self, n: usize, context: Context) -> bool {
        let (pmcr1, el) = match context.filter {
            Filter::Host => (self.pmcr1, context.el),
            Filter::Guest => (self.pmcr1_el12, context.el),
            Filter::Guarded => (self.pmcr1_gl1, 1),
        };
        let filter = if el == 0 {
            counter_bit(8, 40, n)
        } else {
            counter_bit(16, 48, n)
        };
        self.pmcr0 & counter_bit(0, 32, n) != 0
            && pmcr1 & filter != 0
            && !(self.event(n) == Some(Event::Cycles) && context.waiting)
    }

    /// The index of the counter `encoding` names, if it is one.
    pub fn counter(&self, encoding: Encoding) -> Option<usize> {
        self.layout.counters.iter().position(|&pmc| pmc == encoding)
    }

    /// The address of counter `encoding` while PMCR0 leaves it disabled,
    /// where a read finds what the last write or count left until a write
    /// of the monitor changes it.
    pub fn idle_storage(&self, encoding: Encoding) -> Option<u64> {
        let n = self.counter(encoding)?;
        (self.pmcr0 & counter_bit(0, 32, n) == 0).then(|| &self.counters[n] as *const u64 as u64)
    }

    /// PMC0 as a read at a later system count finds it: [`Self::read`]
    /// counts from `synced_count` in the context of the last
    /// synchronization, which no read changes, so the count at `now` is
    /// `counters[0]` plus the floor of the ticks since, scaled with the
    /// carried remainder, and nothing while PMC0 does not count there.
    /// Closed while PMC0's PMI is armed: a read then leaves the crossing of
    /// the overflow bit to the engine.
    pub fn native_cycles(&self) -> stela_compiler::x86_64::register_read::CycleCounter {
        let counting = self.counts(CYCLES, self.context);
        stela_compiler::x86_64::register_read::CycleCounter {
            open: u64::from(!self.armed(CYCLES)),
            counter: self.counters[CYCLES],
            synced: self.synced_count,
            remainder: self.remainder[CYCLES],
            clock: if counting { self.clock } else { 0 },
            frequency: self.frequency,
            mask: self.layout.width.mask(),
        }
    }

    /// PMC1 as a read finds it after `retired` instructions retired in
    /// `context`, the PE's context as native code is entered, plus the
    /// instructions native code retires since: [`Self::read`] counts the
    /// instructions since `synced_retired` in the context of the last
    /// synchronization, which every change of context brings up to date
    /// (`Machine::sync_monitor`), so the count is `counters[1]` plus them
    /// while PMC1 counts there. It is closed while PMC1's PMI is armed, so
    /// that a read never decides on the crossing of the overflow bit, and
    /// where `context` counts otherwise than the synchronized one.
    pub fn native_instructions(
        &self,
        retired: u64,
        context: Context,
    ) -> stela_compiler::x86_64::register_read::InstructionCounter {
        let counting = self.counts(INSTRUCTIONS, self.context);
        if self.armed(INSTRUCTIONS) || counting != self.counts(INSTRUCTIONS, context) {
            return Default::default();
        }
        let since = if counting {
            retired.wrapping_sub(self.synced_retired)
        } else {
            0
        };
        stela_compiler::x86_64::register_read::InstructionCounter {
            open: 1,
            base: self.counters[INSTRUCTIONS].wrapping_add(since),
            step: if counting { u64::MAX } else { 0 },
            mask: self.layout.width.mask(),
        }
    }

    /// Whether counter `n`'s PMI is armed: PMCR0 selects FIQ and enables it.
    fn armed(&self, n: usize) -> bool {
        self.pmcr0 & PMCR0_INTGEN == PMCR0_INTGEN_FIQ && self.pmcr0 & counter_bit(12, 44, n) != 0
    }

    /// Whether an enabled event depends on the core clock.
    pub fn timed(&self) -> bool {
        (0..COUNTERS).any(|n| {
            self.pmcr0 & counter_bit(0, 32, n) != 0 && self.event(n) == Some(Event::Cycles)
        })
    }

    /// Count up to the system count `now` and `retired` instructions in the
    /// context since the last synchronization.
    fn advance(&mut self, now: u64, retired: u64) {
        let ticks = now.saturating_sub(self.synced_count);
        let instructions = retired.wrapping_sub(self.synced_retired);
        for n in 0..COUNTERS {
            if !self.counts(n, self.context) {
                continue;
            }
            let increment = match self.event(n) {
                Some(Event::Cycles) => {
                    let scaled =
                        u128::from(ticks) * u128::from(self.clock) + u128::from(self.remainder[n]);
                    let frequency = u128::from(self.frequency);
                    self.remainder[n] = (scaled % frequency) as u64;
                    (scaled / frequency) as u64
                }
                Some(Event::Instructions) => instructions,
                Some(Event::None) | None => 0,
            };
            self.counters[n] = self.counters[n].wrapping_add(increment) & self.layout.width.mask();
        }
        self.synced_count = self.synced_count.max(now);
        self.synced_retired = retired;
    }

    /// Count up to now, then continue in `context`.
    pub fn sync(&mut self, now: u64, retired: u64, context: Context) {
        self.advance(now, retired);
        self.context = context;
        self.update();
    }

    /// The PMI of the current counts, and when the counting reaches one.
    fn update(&mut self) {
        let fiq = self.pmcr0 & PMCR0_INTGEN == PMCR0_INTGEN_FIQ;
        let armed = |n: usize| n < COUNTERS && fiq && self.pmcr0 & counter_bit(12, 44, n) != 0;
        self.asserted = (0..self.counter_count())
            .any(|n| armed(n) && self.counters[n] & self.layout.width.overflow() != 0);
        self.cycle_deadline = None;
        self.instruction_deadline = None;
        for n in 0..COUNTERS {
            if !armed(n) || !self.counts(n, self.context) {
                continue;
            }
            let needed = self.layout.width.overflow().wrapping_sub(self.counters[n])
                & self.layout.width.mask();
            match self.event(n) {
                Some(Event::Cycles) => {
                    let scaled = (u128::from(needed) * u128::from(self.frequency))
                        .saturating_sub(u128::from(self.remainder[n]));
                    let ticks = scaled.div_ceil(u128::from(self.clock));
                    let deadline = u64::try_from(ticks)
                        .ok()
                        .and_then(|ticks| self.synced_count.checked_add(ticks));
                    if let Some(deadline) = deadline {
                        self.cycle_deadline = Some(
                            self.cycle_deadline
                                .map_or(deadline, |old| old.min(deadline)),
                        );
                    }
                }
                Some(Event::Instructions) => {
                    if let Some(deadline) = self.synced_retired.checked_add(needed) {
                        self.instruction_deadline = Some(
                            self.instruction_deadline
                                .map_or(deadline, |old| old.min(deadline)),
                        );
                    }
                }
                Some(Event::None) | None => {}
            }
        }
    }

    /// Whether the PMI is asserted at system count `now` with `retired`
    /// instructions retired, the context unchanged since the last
    /// synchronization.
    pub fn asserted(&self, now: u64, retired: u64) -> bool {
        if !(self.asserted
            || self.cycle_deadline.is_some_and(|deadline| now >= deadline)
            || self
                .instruction_deadline
                .is_some_and(|deadline| retired >= deadline))
        {
            return false;
        }
        // A cached deadline identifies the first crossing, but the high
        // counter bit can clear again after a complete-width wrap. Do not
        // turn that deadline into an unchecked sticky pending latch.
        (0..self.counter_count()).any(|n| {
            let armed = n < COUNTERS
                && self.pmcr0 & PMCR0_INTGEN == PMCR0_INTGEN_FIQ
                && self.pmcr0 & counter_bit(12, 44, n) != 0;
            if !armed {
                return false;
            }
            let increment = if self.counts(n, self.context) {
                match self.event(n) {
                    Some(Event::Cycles) => {
                        let ticks = now.saturating_sub(self.synced_count);
                        ((u128::from(ticks) * u128::from(self.clock)
                            + u128::from(self.remainder[n]))
                            / u128::from(self.frequency)) as u64
                    }
                    Some(Event::Instructions) => retired.wrapping_sub(self.synced_retired),
                    Some(Event::None) | None => 0,
                }
            } else {
                0
            };
            self.counters[n].wrapping_add(increment) & self.layout.width.overflow() != 0
        })
    }

    /// Instruction PMI needs a scheduler observation after each native body.
    /// A disabled/reset monitor imposes no instruction deadline.
    pub(crate) fn has_instruction_deadline(&self) -> bool {
        self.asserted || self.instruction_deadline.is_some()
    }

    /// Whether a change of EL or guarded level leaves every count as it
    /// was: no enabled counter's filters tell the levels apart.
    pub fn alike(&self) -> bool {
        self.alike
    }

    /// The retired instruction count at which the PMI can first assert by
    /// instructions alone.
    pub fn instruction_deadline(&self) -> Option<u64> {
        self.instruction_deadline
    }

    /// The earliest system count at which the PMI can assert without
    /// another instruction retiring.
    pub fn deadline(&self) -> Option<u64> {
        if self.asserted {
            Some(self.synced_count)
        } else {
            self.cycle_deadline
        }
    }

    /// PMCR1_GL1 took `value` at system count `now` with `retired`
    /// instructions retired.
    pub fn set_guarded_filter(&mut self, value: u64, now: u64, retired: u64, context: Context) {
        self.advance(now, retired);
        self.pmcr1_gl1 = value;
        self.context = context;
        self.refresh_alike();
        self.update();
    }

    /// [`Self::alike`] for the controls now, which only writes change.
    fn refresh_alike(&mut self) {
        let contexts = [Filter::Host, Filter::Guest, Filter::Guarded]
            .into_iter()
            .flat_map(|filter| {
                (0..=2).map(move |el| Context {
                    el,
                    filter,
                    waiting: false,
                })
            });
        self.alike = (0..COUNTERS).all(|n| {
            let mut counts = contexts.clone().map(|context| self.counts(n, context));
            let first = counts.next();
            counts.all(|count| Some(count) == first)
        });
    }

    pub fn read(
        &mut self,
        encoding: Encoding,
        now: u64,
        retired: u64,
        context: Context,
    ) -> Result<u64, &'static str> {
        self.check_access(encoding)?;
        let encoding = regime_copy(encoding, context);
        self.sync(now, retired, context);
        let stored = |value: Option<u64>| {
            value.ok_or("read of a performance monitor control before its first write")
        };
        Ok(match encoding {
            PMCR0 => self.pmcr0 | if self.asserted { PMCR0_PMAI } else { 0 },
            PMCR1 => self.pmcr1,
            PMCR1_EL12 => self.pmcr1_el12,
            PMESR0 => stored(self.pmesr[0])?,
            PMESR1 => stored(self.pmesr[1])?,
            PMSR => (0..self.counter_count())
                .filter(|&n| self.counters[n] & self.layout.width.overflow() != 0)
                .fold(0, |status, n| status | self.layout.status_bit(n)),
            PC_CAPTURE => return Err("PMI PC capture is not modelled"),
            KPC_C5_1 => stored(self.kpc_c5_1)?,
            _ => match self.counter(encoding) {
                Some(n) => self.counters[n],
                None => {
                    let index = EXTENSION.iter().position(|&register| register == encoding);
                    stored(index.and_then(|index| self.extension[index]))?
                }
            },
        })
    }

    pub fn write(
        &mut self,
        encoding: Encoding,
        value: u64,
        now: u64,
        retired: u64,
        context: Context,
    ) -> Result<(), &'static str> {
        self.check_access(encoding)?;
        self.advance(now, retired);
        let written = self.store(regime_copy(encoding, context), value);
        self.context = context;
        self.refresh_alike();
        self.update();
        written
    }

    fn store(&mut self, encoding: Encoding, value: u64) -> Result<(), &'static str> {
        match encoding {
            PMCR0 => {
                if value & !PMCR0_KNOWN != 0 {
                    return Err("PMCR0 bit without public description");
                }
                self.validate_events(value, &self.pmesr)?;
                if !matches!(value & PMCR0_INTGEN, 0 | PMCR0_INTGEN_FIQ) {
                    return Err("PMCR0 PMI generation other than FIQ");
                }
                if value & PMCR0_PC_CAPTURE != 0 {
                    return Err("PMCR0 PMI PC capture is not modelled");
                }
                if value & PMCR0_USER_ACCESS != 0 {
                    return Err("PMCR0 EL0 access to the monitor is not modelled");
                }
                // PMAI reports the PMI; writing it changes nothing.
                self.pmcr0 = value & !PMCR0_PMAI;
            }
            PMCR1 | PMCR1_EL12 => {
                if value & !PMCR1_KNOWN != 0 {
                    return Err("PMCR1 bit without public description");
                }
                if encoding == PMCR1 {
                    self.pmcr1 = value;
                } else {
                    self.pmcr1_el12 = value;
                }
            }
            PMESR0 | PMESR1 => {
                if value & !self.layout.selectors != 0 {
                    return Err(
                        "CPMU event selector field is not source-qualified for this layout",
                    );
                }
                let mut selectors = self.pmesr;
                selectors[usize::from(encoding == PMESR1)] = Some(value);
                self.validate_events(self.pmcr0, &selectors)?;
                self.pmesr = selectors;
            }
            PMSR | PC_CAPTURE => return Err("write of a performance monitor status register"),
            KPC_C5_1 if value != 0 => {
                return Err("M5 kpc configuration S3_1_C15_C5_1 is not modelled");
            }
            KPC_C5_1 => self.kpc_c5_1 = Some(0),
            _ => match self.counter(encoding) {
                Some(n) => self.counters[n] = value & self.layout.width.mask(),
                None => {
                    let index = EXTENSION
                        .iter()
                        .position(|&register| register == encoding)
                        .expect("owned");
                    if value != 0 {
                        return Err(
                            "PMCR2-4, OPMAT, OPMSK and PM_MEMFLT_CTL settings are not modelled",
                        );
                    }
                    self.extension[index] = Some(0);
                }
            },
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
