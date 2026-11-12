//! What the engine reports: events, stops, unsupported requirements, hits,
//! surveys and statistics.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// An exception taken from `from_el`, a guarded level when
    /// `from_guarded`.
    Exception {
        vector: Vector,
        from_el: u8,
        from_guarded: bool,
        syndrome: u64,
        return_address: u64,
        fault_address: Option<u64>,
    },
    /// An exception return, to a guarded level when `guarded`.
    Return { to_el: u8, guarded: bool, pc: u64 },
    /// An opt-in guarded boundary trace, emitted synchronously without a queue.
    Monitor(MonitorEvent),
    /// GENTER (`enter`) or GEXIT, and the level the PE is now at.
    Guarded {
        enter: bool,
        el: u8,
        guarded: bool,
        pc: u64,
    },
}

/// Where a PE is, for a report of a wait on it
/// ([`RequirementKind::MaintenanceOverrun`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeState {
    /// In its run loop: an exit, a native slice or a wait for the devices'
    /// lock.
    Running,
    /// Waiting in the engine, on another PE's broadcast or a stop.
    Parked,
    /// Outside its run loop or asleep.
    Outside,
    Unknown,
}

impl fmt::Display for PeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Running => "running",
            Self::Parked => "parked",
            Self::Outside => "outside",
            Self::Unknown => "unknown",
        })
    }
}

/// A requirement this engine does not implement, with where it was met.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Requirement {
    pub pc: u64,
    pub el: u8,
    /// The link register then, which names the caller of an accessor; 0
    /// where no PE state is at hand.
    pub link: u64,
    pub kind: RequirementKind,
    /// The ESR_EL1 value the access would report if it were a guest exception.
    pub syndrome: Option<u64>,
    pub address: Option<u64>,
    /// The value a refused device write carried, which its report shows.
    pub written: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequirementKind {
    Instruction {
        word: u32,
        detail: String,
    },
    SystemRegister {
        word: u32,
        refusal: Refusal,
    },
    SystemInstruction {
        word: u32,
        detail: &'static str,
    },
    Translation(mmu::Unsupported),
    NonresidentRam {
        frame: u64,
    },
    RamTransfer {
        physical: u64,
        write: bool,
        completed: usize,
        status: stela_compiler::context::memory::MemoryStatus,
    },
    Fetch {
        physical: u64,
    },
    Device {
        physical: u64,
        write: bool,
        detail: String,
    },
    DeviceAdvance(String),
    IllegalReturn(IllegalReturn),
    /// Entering an EL1&0 guest of the EL2 host while the named Apple VM
    /// register (`impdef::Class::Guest`) holds state the machine does not
    /// apply to a guest.
    VmState(&'static str),
    Native(&'static str),
    ParallelAdmission(ParallelAdmissionError),
    /// Register residency cannot be recovered from the host capture.
    FaultRecovery(stela_compiler::block::FaultRecoveryError),
    /// Detailed native admission refusal before the source instruction executes.
    NativePreparation(String),
    /// A registered RAM service excluded entry before native code executed.
    NativeRamServiceBusy,
    /// A registered RAM service excluded a promoted device site's walk.
    DeviceSiteRamServiceBusy,
    /// Other PEs had not applied posted TLB or code maintenance after the
    /// reported bound: `pe`, one of the `unapplied` PEs, in `state`
    /// ([`super::parallel::Hub::waiting_state`]), was still short of mail
    /// `serial` after `ticks` of the system count. The wait goes on.
    MaintenanceOverrun {
        pe: usize,
        serial: u64,
        unapplied: u32,
        state: PeState,
        ticks: u64,
    },
}

impl fmt::Display for Requirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pc={:#x} el={} ", self.pc, self.el)?;
        if self.link != 0 {
            write!(f, "lr={:#x} ", self.link)?;
        }
        match &self.kind {
            RequirementKind::Instruction { word, detail } => {
                write!(f, "instruction {word:#010x}: {detail}")?
            }
            RequirementKind::SystemRegister { word, refusal } => {
                let access = stela_compiler::aarch64::system::registers::decode(*word);
                match access {
                    Some(access) => {
                        let e = access.encoding();
                        write!(
                            f,
                            "system register {} S{}_{}_C{}_C{}_{} ({word:#010x}): {refusal:?}",
                            if access.direction() == Direction::Read {
                                "read"
                            } else {
                                "write"
                            },
                            e.op0(),
                            e.op1(),
                            e.crn(),
                            e.crm(),
                            e.op2()
                        )?
                    }
                    None => write!(f, "system register {word:#010x}: {refusal:?}")?,
                }
            }
            RequirementKind::SystemInstruction { word, detail } => {
                write!(f, "system instruction {word:#010x}: {detail}")?
            }
            RequirementKind::Translation(unsupported) => {
                write!(f, "translation configuration {unsupported:?}")?
            }
            RequirementKind::NonresidentRam { frame } => {
                write!(f, "RAM frame at PA {frame:#x} is nonresident")?
            }
            RequirementKind::RamTransfer {
                physical,
                write,
                completed,
                status,
            } => write!(
                f,
                "registered RAM {} at PA {physical:#x} stopped after {completed} bytes: {status:?}",
                if *write { "write" } else { "read" }
            )?,
            RequirementKind::Fetch { physical } => {
                write!(f, "instruction fetch from non-RAM PA {physical:#x}")?
            }
            RequirementKind::Device {
                physical,
                write,
                detail,
            } => {
                write!(
                    f,
                    "device {} at PA {physical:#x}: {detail}",
                    if *write { "write" } else { "read" }
                )?;
                if let Some(value) = self.written {
                    write!(f, " (value {value:#x})")?;
                }
            }
            RequirementKind::DeviceAdvance(detail) => write!(f, "device advance: {detail}")?,
            RequirementKind::IllegalReturn(reason) => write!(f, "exception return {reason:?}")?,
            RequirementKind::VmState(register) => write!(
                f,
                "entering an EL1&0 guest with nonzero Apple VM state in {register}"
            )?,
            RequirementKind::Native(detail) => write!(f, "native execution: {detail}")?,
            RequirementKind::ParallelAdmission(error) => write!(f, "parallel admission: {error}")?,
            RequirementKind::FaultRecovery(cause) => write!(f, "native fault recovery: {cause}")?,
            RequirementKind::NativePreparation(detail) => {
                write!(f, "native preparation: {detail}")?
            }
            RequirementKind::NativeRamServiceBusy => write!(
                f,
                "native preparation: {}",
                super::NativeEntryError::RamServiceBusy
            )?,
            RequirementKind::DeviceSiteRamServiceBusy => {
                f.write_str("promoted device site: registered RAM memory service busy")?
            }
            RequirementKind::MaintenanceOverrun {
                pe,
                serial,
                unapplied,
                state,
                ticks,
            } => write!(
                f,
                "maintenance overrun: PE {pe} ({state}) has not applied mail {serial} after \
                 {ticks} counter ticks ({unapplied} PEs waited on); still waiting"
            )?,
        }
        if let Some(syndrome) = self.syndrome {
            write!(f, " esr={syndrome:#x}")?;
        }
        if let Some(address) = self.address {
            write!(f, " far={address:#x}")?;
        }
        Ok(())
    }
}

/// Trace hits kept between [`Machine::take_trace_hits`] calls.
pub const TRACE_HITS: usize = 4096;

/// Watch hits kept between [`Machine::take_watch_hits`] calls.
pub const WATCH_HITS: usize = 4096;

/// A device access inside a host watch range (`Machine::set_watches`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WatchHit {
    pub pc: u64,
    pub pe: usize,
    pub physical: u64,
    pub bytes: u8,
    /// The value read or written; zero for a refused access.
    pub value: u64,
    pub write: bool,
    /// Whether the device completed the access.
    pub completed: bool,
    /// Further hits dropped after this one, beyond [`WATCH_HITS`].
    pub dropped: u64,
}

/// A device register's accesses ([`Machine::device_registers`]) and, under
/// step timing, the profile-clock ticks of the native faults that reached it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeviceRegister {
    pub accesses: u64,
    pub ticks: u64,
}

/// A PE reaching a host trace point (`Machine::set_traces`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceHit {
    pub pc: u64,
    pub pe: usize,
    pub el: u8,
    /// X0 to X3 and the link register.
    pub arguments: [u64; 4],
    pub lr: u64,
    /// Further hits dropped after this one, beyond [`TRACE_HITS`].
    pub dropped: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stop {
    /// The block budget ran out.
    Budget,
    /// Every PE is off or waits in WFI or WFE with nothing able to wake it.
    Idle,
    /// Every PE is off or waits, and the run's waits reached the wait limit
    /// (`Machine::set_wait_limit`), a device has bytes for the host
    /// (`Devices::take_host_service`), or only something outside the board
    /// can end the wait (`Devices::waits_on_host`) and no limit is set. The
    /// host takes its inputs and runs the machine again to go on waiting.
    Waiting,
    /// A device carried out the guest's reset or power-off request.
    Reset(Reset),
    Unsupported(Requirement),
    /// The native implementation failed. Its owned code/capture remain held;
    /// the current frame is not a resumable architectural guest state.
    NativeFailure(Arc<NativeFailure>),
    /// An unexecuted device read kept by the canonical scheduler.
    DeviceRead(DeviceReadRequest),
    /// An unexecuted write, kept by the same canonical cohort owner.
    DeviceWrite(DeviceWriteRequest),
    /// The running PE reached a host breakpoint (`Machine::set_breakpoints`)
    /// at this PC, before executing its instruction.
    Breakpoint(u64),
}

/// A machine-wide transition a device carried out for the guest: what the
/// host must do next and the device's account of why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reset {
    pub transition: Transition,
    pub reason: String,
}

impl fmt::Display for Reset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.transition, self.reason)
    }
}

/// Where the running regime's TTBR1 tables map ([`Machine::upper_occupancy`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Occupancy {
    /// The level-1 table's physical address.
    pub root: u64,
    /// First and last VA of each run of valid level-1 entries.
    pub ranges: Vec<(u64, u64)>,
    /// The first and last byte of the outermost mapped 32 MiB, where the
    /// outermost entries are tables; their whole entry otherwise.
    pub lowest: Option<u64>,
    pub highest: Option<u64>,
}

/// Where a guest thread stopped running: the PE's PC, frame pointer and
/// link register, and the physical count, when software replaced the thread
/// pointer naming it (TPIDR_EL1). While the thread stays suspended its stack
/// still holds the frame records `frame` starts, so diagnostics can walk
/// where it waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Suspension {
    pub at: u64,
    pub pc: u64,
    pub frame: u64,
    pub link: u64,
}

/// The suspended threads [`Machine`] remembers at most.
pub(super) const SUSPENSIONS: usize = 1 << 14;

/// One device register a survey answered in place of its model: where it
/// was first reached, how often, the last value written to it, and the
/// device layer's reason for refusing it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Surveyed {
    pub physical: u64,
    pub bytes: u8,
    pub write: bool,
    pub pc: u64,
    /// The link register then, which names the caller of an accessor.
    pub link: u64,
    pub count: u64,
    pub value: u64,
    pub detail: String,
}

/// A survey's records by register and direction, and those not yet taken.
#[derive(Default)]
pub(super) struct Survey {
    pub(super) records: BTreeMap<(u64, bool), Surveyed>,
    pub(super) fresh: Vec<(u64, bool)>,
}

/// Why the engine discarded every installed block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Discard {
    /// Guest IC maintenance of every page, on this PE or broadcast by
    /// another.
    Maintenance,
    /// Low memory that releasing older code could not relieve.
    Memory,
    /// The block table could not grow.
    Table,
    /// A full code arena of one part.
    Arena,
    /// Breakpoints or traces changed, which every block embeds.
    Diagnostics,
    /// The run ended.
    Finished,
}

impl Discard {
    pub const ALL: [Self; 6] = [
        Self::Maintenance,
        Self::Memory,
        Self::Table,
        Self::Arena,
        Self::Diagnostics,
        Self::Finished,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Maintenance => "maintenance",
            Self::Memory => "memory",
            Self::Table => "table",
            Self::Arena => "arena",
            Self::Diagnostics => "diagnostics",
            Self::Finished => "finished",
        }
    }
}

/// The branch-kind exits [`Statistics::branch_misses`] counts: fallthrough,
/// direct, indirect, call and return, `ExitKind`'s first five.
pub const BRANCH_KINDS: usize = 5;
/// [`BranchMiss`]'s reasons and the count of runs with a transition.
pub const BRANCH_REASONS: usize = BranchMiss::COUNT + 1;

/// Why a run ended at a branch-kind exit instead of transferring
/// ([`Statistics::branch_misses`]), checked in this order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BranchMiss {
    /// The chain budget was spent (or the slice left none).
    Spent,
    /// A notification moved the interrupt epoch: a gate took the exit.
    Interrupt,
    /// The exit had no link or dispatcher yet.
    Unlinked,
    /// The ITLB held no translation of the target's page in the current
    /// context.
    Itlb,
    /// The block table held no entry of the target with that key.
    Table,
    /// The entry refused the exit's BTYPE.
    Btype,
    /// The table would have given the target: a stale cell, stamp or
    /// context in the probe.
    Found,
}

impl BranchMiss {
    pub(super) const COUNT: usize = Self::Found as usize + 1;
}

/// How a fetch began ([`Statistics::fetch_paths`]): through the ITLB and the
/// block table without a walk, with BTYPE zero or with an entry that admits
/// the PE's BTYPE; or, leaving that path, because the ITLB lacked the page
/// in the current context, the block table lacked the PC with the page's
/// key, the entry refused the BTYPE, or the fetch was of one instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FetchPath {
    Fast,
    FastBtype,
    ItlbMiss,
    TableMiss,
    BtypeRefused,
    Single,
}

impl FetchPath {
    pub(super) const COUNT: usize = 6;
}

const _: () = assert!(FetchPath::Single as usize + 1 == FetchPath::COUNT);

/// Where an installed block came from, for [`Statistics::store_sources`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StoreSource {
    /// Compiled in a page the store keys: offered to it.
    Compiled,
    /// Compiled where the store keys no page: a one-instruction block, a
    /// page with a device site or a host breakpoint, or no store at all.
    Unstorable,
    /// Relocated from a record of the pack the boot read.
    Pack,
    /// Compiled by another PE of this boot and installed here.
    Shared,
    /// Relocated from an entry proven during this boot, compiled at
    /// another address of the same page content ([`super::rebase`]).
    Rebased,
}

impl StoreSource {
    pub(super) const COUNT: usize = 5;
}

const _: () = assert!(StoreSource::Rebased as usize + 1 == StoreSource::COUNT);

/// Why the ITLB lacked a page it then took ([`Statistics::itlb_misses`]): a
/// fill of another page took its entry's place, a TLBI by VA dropped it,
/// its context was forgotten (an entry of the page in a forgotten context
/// of the same salt was found), or none of these was seen (the first fetch
/// of the page in the context, or one whose earlier entry left no trace).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ItlbMiss {
    Evicted,
    Forgotten,
    Context,
    Cold,
}

impl ItlbMiss {
    pub(super) const COUNT: usize = 4;
}

const _: () = assert!(ItlbMiss::Cold as usize + 1 == ItlbMiss::COUNT);

/// Where the ITLB's fills, drops and forgotten contexts came from, past
/// [`ItlbMiss`] ([`Statistics::itlb_sources`]):
/// - fills of the lower (user) and upper (kernel) half;
/// - cold fills of a page the ITLB took before in the same address space,
///   whatever the context (a refill whose earlier entry left no trace);
/// - entries TLBIs by VA dropped in one ASID's contexts, and in every
///   context (no ASID, a global page in the span, or the upper half);
/// - TLBIs of everything that forgot every context, this PE's own and
///   another PE's (a broadcast, or the whole invalidation its mail kept);
/// - such TLBIs that reached nothing of the regime the PE translates in,
///   which forgot only other regimes' contexts, or none;
/// - contexts made;
/// - TLBIs another PE sent that the PE's mail merged into a range;
/// - pages a context made anew walked again at once, which its forgotten
///   predecessor of the same key had taken (`Machine::refill_pages`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ItlbSource {
    FillsLower,
    FillsUpper,
    ColdSeen,
    DropAsid,
    DropEvery,
    AllLocal,
    AllReceived,
    AllForgot,
    AllKept,
    Made,
    MailMerged,
    Refilled,
}

impl ItlbSource {
    pub(super) const COUNT: usize = 12;
}

const _: () = assert!(ItlbSource::Refilled as usize + 1 == ItlbSource::COUNT);
// A fill counts at `FillsLower` plus its half.
const _: () = assert!(ItlbSource::FillsUpper as usize == ItlbSource::FillsLower as usize + 1);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(
    all(feature = "std", feature = "native-profile"),
    derive(serde::Serialize)
)]
pub struct Statistics {
    pub blocks: u64,
    /// Retirements from CPU incarnations replaced by a real PE restart.
    /// Add live/parked CPU counters through `retired_instructions`.
    pub replaced_retired: u128,
    /// Blocks entered through a chain from another, without the engine.
    pub chained: u64,
    /// Direct exits redirected into their successors.
    pub links: u64,
    /// Exits out of their page given a stamped tail
    /// ([`stela_compiler::x86_64::chain::stamped_tail`]).
    pub stamped_tails: u64,
    /// Exits of RETA*, BRA* and BLRA* thunks given a predicted tail
    /// ([`stela_compiler::x86_64::chain::predicted_tail`]).
    pub predicted_tails: u64,
    /// How many exits had a stamped tail when this PE last looked (every
    /// promotion step): a level, which adding keeps the larger of.
    pub stamped_in_use: u64,
    /// Links undone, as the blocks they chained into went.
    pub unlinks: u64,
    /// Exits redirected through a lookup of their target (returns and
    /// direct exits out of their page), and the transfers those made.
    pub lookups: u64,
    pub lookup_hits: u64,
    /// Native runs that ended in an exit to the engine, by `ExitKind`
    /// (Fallthrough, Direct, Indirect, Call, Return, Trap, PAC, Clock,
    /// ThreadRegister, FloatingPoint, ScalableNumerical, System, AMX).
    pub exit_kinds: [u64; 13],
    /// Exits of returns, indirect branches, calls and direct exits out of
    /// their page, by why no lookup dispatcher transferred them: none was
    /// installed yet, the ITLB held no translation of the target's page in
    /// the current context, the block table no entry of the target with
    /// that page's key, the chain budget was spent, or the entry refused
    /// the BTYPE.
    pub lookup_misses: [u64; 5],
    /// Forgotten lookup contexts and ITLB entries: halves whose contexts
    /// went for an ASIDE1 or a TLBI of everything, halves a TLBI by VA
    /// reached ITLB entries of, and halves whose contexts went for any
    /// other change (an unscoped translation change, the host's fetch
    /// permissions, the upper half's first non-global fetch).
    pub lookup_forgets: [u64; 4],
    /// Fetches the engine made, by where they found their block: the block
    /// table through the ITLB's translation, there after the fetch
    /// translated the page anew, the jump cache, the block index, or
    /// nowhere, so that it was compiled or installed.
    pub fetches: [u64; 5],
    /// Pages the ITLB lacked in the context it took their translation in,
    /// and pages it held there that took a new block key (a remap, or the
    /// frame's blocks taking another variant).
    pub itlb: [u64; 2],
    /// Fetches by whether they found their block through the ITLB and the
    /// block table without a walk, in `FetchPath`'s order.
    pub fetch_paths: [u64; 6],
    /// Pages the ITLB lacked in the context it took their translation in
    /// (`itlb[0]`), by why, in `ItlbMiss`'s order.
    pub itlb_misses: [u64; 4],
    /// ITLB entries of live contexts a fill took the place of, ITLB
    /// entries a TLBI by VA dropped, contexts that left a half's
    /// `contexts::LOWER_CONTEXTS` or `UPPER_CONTEXTS` most recent (their
    /// entries with them), lookup salts a new
    /// owner took from another (`Machine::context_salt`), and TLB
    /// invalidations of everything that stood for more scoped TLBIs than a
    /// PE's mail keeps (`MAIL_SCOPES`).
    pub itlb_drops: [u64; 5],
    /// The ITLB's fills, drops and forgotten contexts by where they came
    /// from, in `ItlbSource`'s order.
    pub itlb_sources: [u64; 12],
    pub compiled: u64,
    /// Of `compiled`, the blocks that run their iterations inside
    /// themselves (`CompiledBlock::self_loop`).
    pub self_loops: u64,
    /// Counter ticks spent analyzing and compiling `compiled`.
    pub compile_ticks: u64,
    /// Blocks taken from the persistent translation store instead of
    /// compiled, and compiled blocks offered to it.
    pub stored: u64,
    pub offered: u64,
    /// Blocks another thread of the process (or its ahead-of-time
    /// translation) had prepared, installed instead of compiled.
    pub shared: u64,
    /// Installs by where their block came from, at EL0 (`[0]`) and above
    /// (`[1]`), in [`StoreSource`]'s order: compiled with a store page
    /// (offered or lent to the store), compiled without one (one
    /// instruction, a device-site page, a host breakpoint, no store),
    /// taken from the pack, taken from another PE.
    pub store_sources: [[u64; StoreSource::COUNT]; 2],
    /// Counter ticks spent, for blocks not yet installed, reading, hashing
    /// and looking up their fetch pages; finding `shared` blocks; and
    /// installing every block.
    pub page_ticks: u64,
    pub shared_ticks: u64,
    pub install_ticks: u64,
    /// Fetch pages read again whose key a check of their bytes gave
    /// without hashing them ([`stored::page_check`]).
    pub known_keys: u64,
    /// Device transfers run because their deadline came, and the devices'
    /// interrupt lines and deadline asked anew because a cached answer
    /// expired, with the profile clock ticks each took under step timing.
    pub due_transfers: u64,
    pub transfer_ticks: u64,
    pub line_queries: u64,
    pub line_ticks: u64,
    /// Guest instructions in `compiled`.
    pub compiled_instructions: u64,
    pub exceptions: u64,
    pub interrupts: u64,
    pub system_exits: u64,
    /// SVC, ERET, GENTER and GEXIT that native code completed from a
    /// transition record (`x86_64::transition`), by kind.
    pub native_transitions: [u64; transition::KINDS],
    /// Those the engine completed, by the same kinds.
    pub engine_transitions: [u64; transition::KINDS],
    /// Why GENTER, GEXIT, SVC and ERET took the engine path, by kind
    /// (`x86_64::transition`) and `transitions::miss` reason.
    pub transition_misses: [[u64; super::transitions::miss::REASONS]; transition::KINDS],
    /// Why runs ended at a branch-kind exit, by kind (fallthrough, direct,
    /// indirect, call, return) and [`BranchMiss`] reason; the last slot
    /// counts those of each kind whose run made a native transition.
    pub branch_misses: [[u64; BRANCH_REASONS]; BRANCH_KINDS],
    /// AMX moves completed inside blocks (`x86_64::amx_transfer`) and by
    /// the engine's service.
    pub amx_moves: [u64; 2],
    /// AMX register operations completed inside blocks
    /// (`x86_64::amx_routine`) and by the engine's service.
    pub amx_operations: [u64; 2],
    /// Those operations by number and path (host kernel, model), and the
    /// ticks and count of the sampled calls of each
    /// (`x86_64::amx_transfer::Cell`), added every 2^16 operations.
    pub amx_kernels: [[u64; 2]; stela_compiler::x86_64::amx_transfer::KINDS],
    pub amx_sampled: [[[u64; 2]; 2]; stela_compiler::x86_64::amx_transfer::KINDS],
    pub device_accesses: u64,
    /// Discards of every installed block, by [`Discard`] cause.
    pub code_discards: [u64; Discard::ALL.len()],
    /// Parts of the code arena reclaimed for new code, oldest first.
    pub code_evictions: u64,
    /// Parts of the code arena released for heap pressure.
    pub code_releases: u64,
    /// Blocks another PE of the board installed first, whose own install
    /// was dropped.
    pub lost_installs: u64,
    /// Pages of compiled code an IC IVAU made stale.
    pub code_invalidations: u64,
    pub waits: u64,
    /// Changes of the running PE.
    pub switches: u64,
    /// Translation control writes that dropped every cached translation.
    pub translation_changes: u64,
    /// TLBIs, and those of them whose reach in the shadow was less than
    /// everything.
    pub tlb_maintenance: u64,
    pub scoped_maintenance: u64,
    /// TTBR0 writes, which drop the lower half's.
    pub table_base_changes: u64,
    /// Guest context changes at which the host isolated its branch
    /// predictions ([`PredictionContext`]).
    pub prediction_isolations: u64,
    /// Fetch pages whose blocks run in the translated variant, because one
    /// of their accesses reached a guest address outside the host's native
    /// ranges.
    pub translated_pages: u64,
    /// Fetch pages that went to the tagged variant: an access of theirs
    /// had a top byte the regime ignores ([`Machine::learn_tagged_page`]).
    pub tagged_pages: u64,
    /// Fetch pages that went to the two-sided range check at EL0: an EL0
    /// access of theirs reached a mapped upper-range address
    /// ([`Machine::learn_two_sided_page`]).
    pub two_sided_pages: u64,
    /// Operations answered with a degraded response; each distinct one is
    /// also reported once ([`Machine::take_unmodeled`]).
    pub degraded: u64,
    /// Counter ticks the host core halted while the PE waited, and spun
    /// while it waited for an event another core's store to its monitored
    /// granule sends.
    pub sleep_ticks: u64,
    pub spin_ticks: u64,
    /// Of `sleep_ticks`, those from the timer's arming to the wake in the
    /// host halt itself, past the halt-poll (`poll_for_wake`): the rest of
    /// `sleep_ticks` the host core spun.
    pub halt_ticks: u64,
    /// Of `waits`, WFEs completed inside their exit by a short bounded spin
    /// (`processors::Machine::short_event_wait`), without a scheduler pass.
    pub short_waits: u64,
    /// With every PE on a core of its own: maintenance this PE broadcast,
    /// device accesses it forwarded to the home machine, and the counter
    /// ticks it spent waiting on other PEs for those or parked while the
    /// home machine served another's.
    pub broadcasts: u64,
    pub forwards: u64,
    pub sync_ticks: u64,
}

impl core::ops::AddAssign for Statistics {
    fn add_assign(&mut self, other: Self) {
        let Self {
            blocks,
            replaced_retired,
            chained,
            links,
            stamped_tails,
            predicted_tails,
            stamped_in_use,
            unlinks,
            lookups,
            lookup_hits,
            exit_kinds,
            lookup_misses,
            lookup_forgets,
            fetches,
            itlb,
            fetch_paths,
            itlb_misses,
            itlb_drops,
            itlb_sources,
            compiled,
            self_loops,
            compile_ticks,
            stored,
            offered,
            shared,
            page_ticks,
            known_keys,
            shared_ticks,
            install_ticks,
            due_transfers,
            transfer_ticks,
            native_transitions,
            engine_transitions,
            transition_misses,
            branch_misses,
            line_queries,
            line_ticks,
            compiled_instructions,
            exceptions,
            interrupts,
            system_exits,
            amx_moves,
            amx_operations,
            amx_kernels,
            amx_sampled,
            device_accesses,
            code_discards,
            code_evictions,
            code_releases,
            lost_installs,
            code_invalidations,
            waits,
            switches,
            translation_changes,
            tlb_maintenance,
            scoped_maintenance,
            table_base_changes,
            prediction_isolations,
            translated_pages,
            tagged_pages,
            two_sided_pages,
            degraded,
            sleep_ticks,
            spin_ticks,
            halt_ticks,
            short_waits,
            broadcasts,
            forwards,
            sync_ticks,
            store_sources,
        } = other;
        self.replaced_retired = self.replaced_retired.saturating_add(replaced_retired);
        for (total, count) in self.exit_kinds.iter_mut().zip(exit_kinds) {
            *total += count;
        }
        for (total, count) in self.amx_moves.iter_mut().zip(amx_moves) {
            *total += count;
        }
        for (total, count) in self.amx_operations.iter_mut().zip(amx_operations) {
            *total += count;
        }
        for (total, count) in self
            .amx_kernels
            .as_flattened_mut()
            .iter_mut()
            .zip(amx_kernels.as_flattened())
        {
            *total += count;
        }
        for (total, count) in self
            .amx_sampled
            .as_flattened_mut()
            .as_flattened_mut()
            .iter_mut()
            .zip(amx_sampled.as_flattened().as_flattened())
        {
            *total += count;
        }
        for (total, count) in self.code_discards.iter_mut().zip(code_discards) {
            *total += count;
        }
        for (total, count) in self.native_transitions.iter_mut().zip(native_transitions) {
            *total += count;
        }
        for (total, count) in self.engine_transitions.iter_mut().zip(engine_transitions) {
            *total += count;
        }
        for (total, count) in self
            .transition_misses
            .as_flattened_mut()
            .iter_mut()
            .zip(transition_misses.as_flattened())
        {
            *total += count;
        }
        for (total, count) in self
            .branch_misses
            .as_flattened_mut()
            .iter_mut()
            .zip(branch_misses.as_flattened())
        {
            *total += count;
        }
        for (total, count) in self.lookup_misses.iter_mut().zip(lookup_misses) {
            *total += count;
        }
        for (total, count) in self.lookup_forgets.iter_mut().zip(lookup_forgets) {
            *total += count;
        }
        for (total, count) in self.fetches.iter_mut().zip(fetches) {
            *total += count;
        }
        for (total, count) in self.itlb.iter_mut().zip(itlb) {
            *total += count;
        }
        for (total, count) in self.fetch_paths.iter_mut().zip(fetch_paths) {
            *total += count;
        }
        for (total, count) in self.itlb_misses.iter_mut().zip(itlb_misses) {
            *total += count;
        }
        for (total, count) in self.itlb_drops.iter_mut().zip(itlb_drops) {
            *total += count;
        }
        for (total, count) in self.itlb_sources.iter_mut().zip(itlb_sources) {
            *total += count;
        }
        self.stamped_in_use = self.stamped_in_use.max(stamped_in_use);
        for (total, count) in self
            .store_sources
            .as_flattened_mut()
            .iter_mut()
            .zip(store_sources.as_flattened())
        {
            *total += count;
        }
        for (total, count) in [
            (&mut self.blocks, blocks),
            (&mut self.chained, chained),
            (&mut self.links, links),
            (&mut self.stamped_tails, stamped_tails),
            (&mut self.predicted_tails, predicted_tails),
            (&mut self.unlinks, unlinks),
            (&mut self.lookups, lookups),
            (&mut self.lookup_hits, lookup_hits),
            (&mut self.compiled, compiled),
            (&mut self.self_loops, self_loops),
            (&mut self.compile_ticks, compile_ticks),
            (&mut self.stored, stored),
            (&mut self.offered, offered),
            (&mut self.shared, shared),
            (&mut self.page_ticks, page_ticks),
            (&mut self.known_keys, known_keys),
            (&mut self.shared_ticks, shared_ticks),
            (&mut self.install_ticks, install_ticks),
            (&mut self.due_transfers, due_transfers),
            (&mut self.transfer_ticks, transfer_ticks),
            (&mut self.line_queries, line_queries),
            (&mut self.line_ticks, line_ticks),
            (&mut self.compiled_instructions, compiled_instructions),
            (&mut self.exceptions, exceptions),
            (&mut self.interrupts, interrupts),
            (&mut self.system_exits, system_exits),
            (&mut self.device_accesses, device_accesses),
            (&mut self.code_evictions, code_evictions),
            (&mut self.code_releases, code_releases),
            (&mut self.lost_installs, lost_installs),
            (&mut self.code_invalidations, code_invalidations),
            (&mut self.waits, waits),
            (&mut self.switches, switches),
            (&mut self.translation_changes, translation_changes),
            (&mut self.tlb_maintenance, tlb_maintenance),
            (&mut self.scoped_maintenance, scoped_maintenance),
            (&mut self.table_base_changes, table_base_changes),
            (&mut self.prediction_isolations, prediction_isolations),
            (&mut self.translated_pages, translated_pages),
            (&mut self.tagged_pages, tagged_pages),
            (&mut self.two_sided_pages, two_sided_pages),
            (&mut self.degraded, degraded),
            (&mut self.sleep_ticks, sleep_ticks),
            (&mut self.spin_ticks, spin_ticks),
            (&mut self.halt_ticks, halt_ticks),
            (&mut self.short_waits, short_waits),
            (&mut self.broadcasts, broadcasts),
            (&mut self.forwards, forwards),
            (&mut self.sync_ticks, sync_ticks),
        ] {
            *total += count;
        }
    }
}

/// The heap bytes of a machine's code bookkeeping, by structure.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Footprint {
    /// The block records, which hold their recovery owners; `owners` is 0
    /// since they moved there.
    pub blocks: usize,
    pub owners: usize,
    /// The live blocks' direct exit lists and recovery rows.
    pub successors: usize,
    pub rows: usize,
    /// The block index and the lookup table's index of granules.
    pub index: usize,
    pub lookup: usize,
    /// The fetch pages kept for the translation store.
    pub pages: usize,
}

/// The block slots a report walk visits between two calls of its `more`.
const WALK_CHUNK: usize = 1024;

/// A walk of the block slots in steps, for a report that would otherwise
/// visit every block at once: the next slot to visit.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SlotWalk {
    next: usize,
}

impl SlotWalk {
    /// Visit slots from where the last step stopped, `WALK_CHUNK` at a time,
    /// while `more` allows; whether every slot given so far was visited.
    fn step(
        &mut self,
        blocks: &super::board::Blocks,
        mut more: impl FnMut() -> bool,
        mut visit: impl FnMut(usize),
    ) -> bool {
        loop {
            let end = blocks.len();
            if self.next >= end {
                return true;
            }
            let chunk = end.min(self.next + WALK_CHUNK);
            for slot in self.next..chunk {
                visit(slot);
            }
            self.next = chunk;
            if !more() {
                return self.next >= blocks.len();
            }
        }
    }
}

/// The exits by instruction word ([`Machine::exit_words`]) gathered in
/// steps: the live blocks' counts so far.
#[derive(Debug, Default, Clone)]
pub struct ExitWordsWalk {
    slots: SlotWalk,
    words: BTreeMap<u32, u64>,
}

/// [`Machine::footprint`] gathered in steps: the rows and exit lists of the
/// blocks visited so far.
#[derive(Debug, Default, Clone, Copy)]
pub struct FootprintWalk {
    slots: SlotWalk,
    rows: usize,
    successors: usize,
}

impl<H: Host, D: Devices> Machine<H, D> {
    /// What its code bookkeeping holds now: a walk of every block, for
    /// reports.
    pub fn footprint(&self) -> Footprint {
        // The walks of every block take no board lock: rows are under
        // their records' locks; only the index and table sizes need it.
        let rows = self.code.blocks.recovery().heap_bytes();
        let successors = self
            .code
            .blocks
            .iter()
            .map(|(_, block)| block.successors.len() * size_of::<super::Successor>())
            .sum();
        let walk = FootprintWalk {
            slots: SlotWalk::default(),
            rows,
            successors,
        };
        self.footprint_finish(&walk)
    }

    /// One step of [`Self::footprint`]'s walk, which goes on while `more`
    /// allows, asked every `WALK_CHUNK` slots; whether it visited the last
    /// slot. It takes no board lock: rows are under their records' locks.
    /// Blocks installed or retired between steps make the sums those of the
    /// slots as each step found them.
    pub fn footprint_step(&self, walk: &mut FootprintWalk, more: impl FnMut() -> bool) -> bool {
        let blocks = &self.code.blocks;
        let recovery = blocks.recovery();
        let (mut rows, mut successors) = (walk.rows, walk.successors);
        let done = walk.slots.step(blocks, more, |slot| {
            if let Some(owner) = recovery.get(&slot) {
                rows += owner.heap_bytes();
            }
            if let Some(block) = blocks.get(slot) {
                successors += block.successors.len() * size_of::<super::Successor>();
            }
        });
        (walk.rows, walk.successors) = (rows, successors);
        done
    }

    /// The footprint from a finished walk: the parts its walk summed and the
    /// index and table sizes, which take the board's lock once.
    pub fn footprint_finish(&self, walk: &FootprintWalk) -> Footprint {
        let cache = self.code.cache_at(super::board::LockSite::Report);
        Footprint {
            blocks: self.code.blocks.heap_bytes(),
            // Owners live in the block records, which `blocks` counts.
            owners: 0,
            successors: walk.successors,
            rows: walk.rows,
            index: cache.index.heap_bytes(),
            lookup: cache.targets.index_bytes() + self.itlb.index_bytes(),
            pages: (self.pages.len() + self.spare_pages.len()) * crate::translations::PAGE_BYTES,
        }
    }

    /// One step of [`Self::exit_words`]' walk of the live blocks' exit
    /// counts, as [`Self::footprint_step`] walks.
    pub fn exit_words_step(&self, walk: &mut ExitWordsWalk, more: impl FnMut() -> bool) -> bool {
        let blocks = &self.code.blocks;
        let words = &mut walk.words;
        walk.slots.step(blocks, more, |slot| {
            // A slot just given may have no record yet.
            let Some(record) = blocks.try_record(slot) else {
                return;
            };
            let taken = record.taken.load(Ordering::Relaxed);
            if taken != 0
                && record.live()
                && let Some(block) = blocks.get(slot)
            {
                *words.entry(block.terminal).or_default() += u64::from(taken);
            }
        })
    }

    /// The exits by word from a finished walk, with those of the blocks
    /// retired so far. A block retired after its slot was visited counts
    /// twice until the next walk; one installed in a visited slot counts
    /// at the next walk.
    pub fn exit_words_finish(&self, walk: ExitWordsWalk) -> BTreeMap<u32, u64> {
        let mut words = walk.words;
        for (&word, &count) in &self.exit_words {
            *words.entry(word).or_default() += count;
        }
        // Exits this PE kept apart from its blocks' records.
        for held in &self.exit_pending {
            if held.count != 0 {
                *words.entry(held.word).or_default() += u64::from(held.count);
            }
        }
        words
    }
}

#[cfg(test)]
mod walk_tests {
    use super::super::tests::{BASE, machine};
    use super::*;

    /// A slot another PE was just given, whose record chunk it has not
    /// allocated yet, is counted among the given slots: the walks pass it
    /// by (build 69 stopped 20 s into boot at `record()` in the heartbeat's
    /// exit walk).
    #[test]
    fn the_walks_pass_a_given_slot_that_has_no_record_yet() {
        const RET: u32 = 0xd65f_03c0;
        let mut machine = machine(&[RET]);
        assert!(matches!(
            machine.fetch_block(BASE, false),
            Ok(Fetched::Block(_))
        ));
        let (grace, _) = machine.grace();
        // Slots of the first record chunk have records once one of them
        // took its block; the first slot of the next has none.
        let bare = loop {
            let slot = machine
                .code
                .blocks
                .take_without_record(grace)
                .expect("a slot");
            if machine.code.blocks.try_record(slot).is_none() {
                break slot;
            }
        };
        assert!(bare < machine.code.blocks.len());
        let mut walk = ExitWordsWalk::default();
        while !machine.exit_words_step(&mut walk, || true) {}
        let mut walk = FootprintWalk::default();
        while !machine.footprint_step(&mut walk, || true) {}
        assert!(!machine.board_slot(bare).live);
    }

    /// The walks in steps of one chunk each give what the walks of every
    /// block at once give.
    #[test]
    fn report_walks_in_steps_match_the_whole_walks() {
        const RET: u32 = 0xd65f_03c0;
        const BLOCKS: usize = WALK_CHUNK + WALK_CHUNK / 2;
        let mut machine = machine(&vec![RET; BLOCKS]);
        for block in 0..BLOCKS as u64 {
            let Ok(Fetched::Block(slot)) = machine.fetch_block(BASE + 4 * block, false) else {
                panic!("a block")
            };
            let taken = &machine.code.blocks.record(slot).taken;
            taken.store(1 + block as u32 % 3, Ordering::Relaxed);
        }
        // An exit counted by word outside a block's last instruction.
        machine.exit_words.insert(0x1234_5678, 5);
        let mut walk = ExitWordsWalk::default();
        let mut steps = 1;
        while !machine.exit_words_step(&mut walk, || false) {
            steps += 1;
        }
        assert_eq!(steps, 2, "one chunk a step");
        let words = machine.exit_words_finish(walk);
        assert_eq!(words, machine.exit_words());
        assert_eq!(words.get(&0x1234_5678), Some(&5));
        let mut walk = FootprintWalk::default();
        while !machine.footprint_step(&mut walk, || false) {}
        assert_eq!(machine.footprint_finish(&walk), machine.footprint());
    }
}
