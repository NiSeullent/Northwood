//! The x86 host and the board's device bus the engine runs against.

use super::*;
use crate::memory::RegisteredSystemRam;
use stela_compiler::context::memory::MemoryStatus;
use stela_contracts::{
    gate::{GateCall, GateOutcome},
    refuse,
};

/// A system block that the host refused before executing any native code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeEntryError {
    /// An active registered RAM service excluded the execution loan.
    RamServiceBusy,
    Refused(&'static str),
}

impl core::fmt::Display for NativeEntryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::RamServiceBusy => "registered RAM native execution refused: memory service busy",
            Self::Refused(reason) => reason,
        })
    }
}

/// Registered RAM owner for a promoted device site. A native host that
/// retains aliases supplies the generation those aliases may still use.
pub struct DeviceSiteRam {
    pub ram: Arc<RegisteredSystemRam>,
    pub expected_generation: Option<u64>,
}

/// One PE's System registers that take none of the board's locks, as
/// [`Devices::system_register`] serves them.
pub trait LocalRegisters: Send + Sync {
    /// An MRS (`write` `None`) or MSR of `encoding`: `None` for an encoding
    /// this does not hold, else the value read (0 for a write).
    fn register(&self, encoding: Encoding, write: Option<u64>) -> Option<u64>;
    /// The address of the 64-bit word `encoding` reads, for the board's
    /// life, if a plain read of it reads just that.
    fn storage(&self, encoding: Encoding) -> Option<u64>;
    /// An MSR of `encoding` that signals another PE, completed without the
    /// devices' lock where it can be: `None` when it must take
    /// [`Devices::system_register`]; `Some(Some(pe))` when it raised PE
    /// `pe`'s [`Self::fiq`], whose PE must look at its lines again;
    /// `Some(None)` when it changed no line.
    fn send(&self, _encoding: Encoding, _value: u64) -> Option<Option<usize>> {
        None
    }
    /// The FIQ this PE's handle drives without the devices' lock (Apple's
    /// fast IPIs), which its lines include beside the latched pins
    /// ([`Devices::latched_pins`]).
    fn fiq(&self) -> bool {
        false
    }
}

/// Device registers every PE reads without the devices' lock: words whose
/// value only software's writes change and whose read has no effect, such
/// as performance counters a power manager samples thousands of times a
/// second. Writes still take the lock; the model publishes each value where
/// this reads it.
pub trait PassiveRegisters: Send + Sync {
    /// The value a `bytes`-byte read at `physical` returns, if this serves
    /// it; `None` sends the read to [`Devices::read_scalar`].
    fn read(&self, physical: u64, bytes: u8) -> Option<u64>;
}

/// A data copy a device's DMA left to run outside the devices' lock
/// ([`Devices::take_unlocked`]): it reaches only guest RAM and state of its
/// own, which the device reads back under the lock once the copy is done.
pub trait UnlockedDma {
    /// Carry the copy out on guest RAM `memory`.
    fn run(self: Box<Self>, memory: &mut dyn Memory);
}

/// Equal, consecutive native code ranges a host reclaims as whole parts.
#[derive(Clone, Copy)]
pub struct CodePartitions {
    pub base: usize,
    pub bytes_per_part: usize,
    pub parts: usize,
}

/// The whole native code reservation, which the engine allocates itself
/// ([`Host::code_window`]): executable at `exec` on every PE that shares
/// it, and writable at `write`, the same memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodeWindow {
    pub exec: usize,
    pub write: usize,
    pub bytes: usize,
}

/// The x86 machine beneath the engine: guest RAM, native code, shadow
/// translation, time and waiting.
pub trait Host {
    /// Optional observational timing on the real local owner. Regular builds
    /// contain neither checkpoints nor their counters.
    #[cfg(feature = "native-profile")]
    fn native_timing(&mut self, _phase: NativeTimingPhase) {}
    #[cfg(feature = "native-profile")]
    fn native_instruction_sample(&mut self, _sample: NativeInstructionSample<'_>) {}
    #[cfg(feature = "native-profile")]
    fn native_timing_work_start(&mut self, _work: NativeTimingWork) -> Option<u64> {
        None
    }
    #[cfg(feature = "native-profile")]
    fn native_timing_work_end(&mut self, _work: NativeTimingWork, _started: u64) {}

    /// Select the source PE whose stopped services this Host can settle through
    /// its physical owner. Ordinary serial Hosts do not admit this path.
    fn select_native_service(&mut self, _worker: publication::WorkerId) -> bool {
        false
    }

    /// Complete this host worker's coherent CPU-memory accesses. This
    /// is not a device/DMA/controller drain or MCC completion permission.
    fn memory_barrier(&mut self) {
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
    }

    /// Retire this held worker's guest-facing direct aliases and cached
    /// translations for the exact registered RAM. Return true only after
    /// the aliases are removed and old translations cannot run again.
    fn revoke_ram_cpu_mappings(&mut self, _request: &publication::RamCpuMappingRevocation) -> bool {
        false
    }

    /// Source-checked identity kept by this physical worker's local
    /// Host/GS owner. Serial hosts have no native-worker permission.
    fn native_worker(&self) -> Option<publication::WorkerId> {
        None
    }

    /// Optional destination-local stable declaration owner. Absence keeps full
    /// live enumeration at every native boundary. A supplied invalid/revoked
    /// lease refuses; it never falls back to disguise stale ownership.
    fn native_cpu_lease(&self) -> Option<crate::NativeCpuLease> {
        None
    }

    /// Read guest RAM. False when any byte is not RAM.
    fn read_ram(&mut self, physical: u64, buffer: &mut [u8]) -> bool;

    /// Return the canonical owner if installed. Failure must stop the
    /// promoted access before trying a legacy probe or a device.
    fn device_site_ram(&self) -> Result<Option<DeviceSiteRam>, MemoryStatus> {
        Ok(None)
    }

    /// Whether the host's retained CPU map still names this GPA page.
    /// Used only to refuse a registered hole that contradicts that map.
    fn device_site_has_retained_ram(&self, _page: u64) -> bool {
        false
    }

    /// Read an aligned, little-endian guest page-table descriptor.
    fn read_ram_descriptor(&mut self, physical: u64) -> Option<u64> {
        let mut bytes = [0; 8];
        self.read_ram(physical, &mut bytes)
            .then(|| u64::from_le_bytes(bytes))
    }

    /// Write guest RAM. False when any byte is not RAM.
    fn write_ram(&mut self, physical: u64, data: &[u8]) -> bool;
    /// Store `new` at `physical` if guest RAM there holds `expected`, in one
    /// single-copy atomic step against every other core; whether it stored.
    /// When not, `expected` receives what RAM held. Accesses of up to 16
    /// bytes within one aligned 16-byte block check; `None` when the host
    /// cannot perform this one.
    fn compare_exchange_ram(
        &mut self,
        _physical: u64,
        _expected: &mut [u8],
        _new: &[u8],
    ) -> Option<bool> {
        None
    }
    /// Descriptor admission needs a real RAM/permission owner. A legacy host
    /// without that service cannot check a native coprocessor mapping.
    fn check_ram_range(
        &mut self,
        _physical: u64,
        _bytes: usize,
        _write: bool,
    ) -> Result<(), stela_contracts::dma::DmaError> {
        Err(stela_contracts::dma::DmaError::Unsupported(refuse!(
            "host DMA range validation is unavailable"
        )))
    }
    /// Stateful DMA must keep partial effects across later owner failures.
    /// The default is for hosts whose boolean failure commits no bytes.
    fn read_ram_progress(
        &mut self,
        physical: u64,
        buffer: &mut [u8],
    ) -> Result<(), stela_contracts::dma::TransferFault> {
        self.read_ram(physical, buffer)
            .then_some(())
            .ok_or(stela_contracts::dma::TransferFault {
                error: stela_contracts::dma::DmaError::Memory,
                completed: 0,
            })
    }
    fn write_ram_progress(
        &mut self,
        physical: u64,
        data: &[u8],
    ) -> Result<(), stela_contracts::dma::TransferFault> {
        self.write_ram(physical, data)
            .then_some(())
            .ok_or(stela_contracts::dma::TransferFault {
                error: stela_contracts::dma::DmaError::Memory,
                completed: 0,
            })
    }

    fn install(&mut self, code: &[u8]) -> Result<NativeCode, InstallError>;
    /// Install immutable context code, optionally reusing this same live Arc's
    /// installation on this Host. Ordinary patchable installs stay
    /// independent. Implementations must invalidate reuse before every arena
    /// patch, discard or reclaim; replacement Hosts start with no entries.
    /// The default preserves the existing one-install-per-call behavior.
    fn install_immutable(&mut self, code: &Arc<Vec<u8>>) -> Result<NativeCode, InstallError> {
        self.install(code.as_slice())
    }
    /// Overwrite installed code at `code` with `bytes`, which the next
    /// native run executes.
    fn patch(&mut self, code: NativeCode, bytes: &[u8]);
    /// Replace the instruction at the patch point `code`
    /// (`chain::patchable`), which a core may be running, with
    /// `instruction`: one aligned 8-byte store of the word holding it
    /// (`chain::patch_word`), so that a core runs the old instruction or the
    /// new one, each whole. What the new one reaches is installed or patched
    /// first. The default is [`Self::patch`], which serves a host on which
    /// no other core runs this arena's code.
    fn patch_instruction(
        &mut self,
        code: NativeCode,
        instruction: [u8; stela_compiler::x86_64::chain::PATCH_BYTES],
    ) {
        self.patch(code, &instruction);
    }
    /// Copy an idle installed native extent from this Host's kept
    /// executable owner. Refuse unknown/reclaimed ranges; numeric addresses
    /// alone never authorize reading native memory. Used in dispatcher fault
    /// validation, never in an interrupt handler.
    fn read_native(&mut self, _code: NativeCode, _bytes: &mut [u8]) -> bool {
        false
    }
    /// Free every installed block.
    fn discard_code(&mut self);
    /// The whole code reservation, when the engine allocates code itself:
    /// every PE whose host gives the same window shares one arena, and the
    /// engine then never calls [`Self::install`], [`Self::patch`],
    /// [`Self::patch_instruction`] or the reclaims for blocks. The host keeps
    /// it mapped, executable and writable as described, for the board's
    /// life, and runs [`Self::execute`] at any address in it.
    fn code_window(&self) -> Option<CodeWindow> {
        None
    }
    /// Make code written through the window's writable view visible to
    /// this core's instruction fetches (on x86, a serializing instruction).
    fn serialize_code(&mut self) {
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
    }
    /// Optional layout for indexing reclaimable block slots. It must
    /// stay fixed from the first block installation until discard, and
    /// reclamation must return one whole part or an empty range.
    fn code_partitions(&self) -> Option<CodePartitions> {
        None
    }
    /// Free the oldest part of the code arena for new code, as a ring
    /// does, and return the host addresses it held: the engine forgets
    /// every block and leaf there before installing more. `None` from a
    /// host whose arena is one part, for which the engine discards every
    /// block instead.
    fn reclaim_code(&mut self) -> Option<core::ops::Range<usize>> {
        None
    }
    /// Free the oldest part of the code arena that holds code, other than
    /// the part new code fills, and return the host addresses it held: low
    /// memory releases the oldest blocks' metadata and keeps the newer
    /// code. An empty range when no other part holds code; `None` for a
    /// host whose arena is one part.
    fn release_oldest_code(&mut self) -> Option<core::ops::Range<usize>> {
        None
    }
    /// Whether the memory the engine's own data takes runs low. Installed
    /// blocks' metadata is most of it, so the engine then reclaims code as
    /// when the code arena fills.
    fn memory_low(&self) -> bool {
        false
    }
    /// Run one native block on `state`, and the blocks it chains into. A
    /// fault's offset is from `code`.
    ///
    /// # Safety
    /// `code` was returned by [`Host::install`] since the last discard and
    /// was compiled for the system-mode block ABI.
    unsafe fn execute(&mut self, code: NativeCode, state: &mut GuestState) -> NativeExit;
    /// Like `execute`, with a refusal before any native effects.
    ///
    /// # Safety
    /// The contract of `execute` applies.
    unsafe fn try_execute(
        &mut self,
        code: NativeCode,
        state: &mut GuestState,
    ) -> Result<NativeExit, NativeEntryError> {
        // SAFETY: forwarded under the same installed system block contract.
        Ok(unsafe { self.execute(code, state) })
    }
    /// Run an installed context-ABI block through this physical owner's native
    /// recovery frame. `None` refuses context entry before any native effects.
    ///
    /// # Safety
    /// `code` was installed by this same Host since its last discard, uses the
    /// compiler's context-block ABI and matches the source-checked
    /// target/profile. The caller exclusively keeps the real guest frame,
    /// valid context and every reachable service/mapping/code owner through
    /// return or fault, complete recovery and source-PC settlement.
    /// The Host stays pinned to its physical owner; callbacks cannot reenter it.
    unsafe fn execute_context(
        &mut self,
        _code: NativeCode,
        _state: &mut GuestState,
        _context: &mut stela_compiler::context::ExecutionContext,
    ) -> Option<NativeExit> {
        None
    }

    /// The address to call installed context-leaf code at, with
    /// `stela_compiler::scalable_context`'s two-argument ABI (the SVE and SME
    /// instructions run so), or `None` on a host that cannot run it. The
    /// engine calls it outside any borrow of the host, whose RAM the leaf's
    /// memory service then reads and writes.
    fn context_entry(&mut self, _code: NativeCode) -> Option<usize> {
        None
    }
    /// Whether the PE may execute the page holding `virtual`, which its
    /// translation lets it execute: false where the host's view of memory
    /// refuses the fetch (an Apple SPRR view in which JIT pages are rw-),
    /// which takes a permission Instruction Abort. Asked when a fetch's
    /// translation is not cached; [`Machine::fetch_permissions_changed`]
    /// makes later fetches ask again.
    fn fetch_permitted(&mut self, _virtual: u64) -> bool {
        true
    }
    /// Where the host places the guest's address space beside its own,
    /// which decides the addresses native code accesses directly
    /// ([`mmu::placement`]).
    fn placement(&self) -> mmu::placement::Placement {
        mmu::placement::Placement::FiveLevel
    }
    /// The address of the 8 bytes that hold the host's translation helper,
    /// which blocks of the translated variant call for each access outside
    /// [`Host::placement`]'s native ranges ([`mmu::translated`]); `None`
    /// on a host without one, which stops explicitly where a guest maps
    /// such an address.
    fn translation_helper(&self) -> Option<u64> {
        None
    }
    /// Make the shadow translation follow `regime`: drop what a change of
    /// its controls translates differently (a TTBR0 write arrives only
    /// here, not through [`Host::invalidate_translations`]).
    fn set_regime(&mut self, regime: &Regime);
    /// From now on, take every lower-half global translation into `span`
    /// before the shadow uses one, walking again when that widens it. What
    /// the shadow holds from before is not in it.
    fn share_lower_globals(&mut self, _span: alloc::sync::Arc<super::GlobalSpan>) {}
    /// Keep the branch predictions trained before this call from steering
    /// speculation after it (FEAT_CSV2): the engine calls it whenever the
    /// guest's context for prediction changes ([`PredictionContext`]). A host
    /// that cannot isolate them leaves this empty and identifies no FEAT_CSV2
    /// (`identification::Speculation::contexts`).
    fn isolate_predictions(&mut self) {}
    /// Keep the reads after this call from executing before the branches
    /// before it resolve (FEAT_CSV3): the engine calls it between its
    /// software translation of an access and the device or tag read it
    /// permits, and [`Host::read_ram`] keeps the same order itself. A host
    /// that cannot order them leaves both empty and identifies no FEAT_CSV3
    /// (`identification::Speculation::faulting_data`).
    fn speculation_barrier(&self) {}
    /// Whether native code changes PSTATE.PAN itself: RFLAGS.AC stands for
    /// it on this host, which follows what native code left there.
    fn pan_in_flags(&self) -> bool {
        false
    }
    /// How native code moves the host from `from`, the regime it follows
    /// now, to `to`, which differs in the requester alone, so that native
    /// code may continue in `to` (`x86_64::transition`): `None` when this
    /// host cannot, or not now. The host follows the view word before it
    /// next resolves a fault of native code or takes a regime.
    fn native_view(&mut self, _from: &Regime, _to: &Regime) -> Option<NativeView> {
        None
    }
    /// The PE invalidated instruction caches (IC IVAU of `Some(page)`, a
    /// physical page; IC IALLU or IALLUIS for `None`), which this machine
    /// already applied to its own compiled code. A host that runs other
    /// PEs in machines of their own makes them drop theirs before they
    /// next execute: IC IVAU reaches every PE of the shareability domain.
    fn code_invalidated(&mut self, _page: Option<u64>) {}
    /// Drop every cached translation.
    fn invalidate_translations(&mut self);
    /// Drop the cached translations a TLBI reaches in the current context,
    /// as [`Host::set_regime`] last gave it, and, of those a host keeps for
    /// other contexts, what it removes from every context (`everywhere`,
    /// [`mmu::tlbi::Scope::everywhere`]); a superset is always correct.
    fn invalidate_reach(&mut self, reach: mmu::tlbi::Reach, _everywhere: mmu::tlbi::Everywhere) {
        if reach != mmu::tlbi::Reach::Nothing {
            self.invalidate_translations();
        }
    }
    /// The guest's system counter.
    fn counter(&mut self) -> u64;
    /// One turn of a wait for other processors: a safe point where this
    /// host answers what they may be waiting for from this processor, such
    /// as a TLB shootdown, so waits that cross do not deadlock. The default
    /// only eases the spin.
    fn relax(&mut self) {
        core::hint::spin_loop();
    }
    /// The TSC base and 64.64 fixed-point scale of a host whose
    /// [`Host::counter`] is `((tsc - base) * scale) >> 64`, which native
    /// code then reads itself ([`register_read::count_thunk`]).
    fn tsc_counter(&self) -> Option<(u64, u64)> {
        None
    }
    /// A cycle count for `--profile`'s phase timing ([`step_timing`]); a
    /// host without one answers 0.
    fn profile_ticks(&mut self) -> u64 {
        0
    }
    /// Under `--profile`, the step loop charged the profile clock's interval
    /// ending at `now` to `phase` ([`step_timing`]). A host with hardware
    /// performance counters attributes them to the interval since its last
    /// call; the default ignores it.
    fn profile_boundary(&mut self, _phase: step_timing::Phase, _now: u64) {}
    /// Interrupt the host at this counter value, or never.
    fn arm_timer(&mut self, deadline: Option<u64>);
    /// Block until a host interrupt arrives.
    fn wait_for_interrupt(&mut self);
    /// Take a pending host interrupt without blocking; whether one came
    /// since the last look.
    fn poll_interrupt(&mut self) -> bool {
        false
    }
    /// Progress events for the diagnostic log.
    fn report(&mut self, _event: &Event) {}
    /// The persistent translation store blocks are taken from and offered
    /// to, if the host keeps one ([`Translations`]).
    fn translations(&mut self) -> Option<&mut dyn Translations> {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceError {
    /// No device model covers this physical address.
    NoDevice,
    /// The device refused this register, width or state.
    Refused(stela_contracts::Refusal),
    /// A DMA service could not execute. Survey mode may substitute unknown
    /// register reads/writes, but cannot invent a completed DMA operation.
    TransferRefused(stela_contracts::Refusal),
}

/// A complete device pair read has consumed its input. A subsequent board
/// service failure cannot turn it into a replayable register refusal: the
/// engine retires the returned data and then stops at the failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsumedPair {
    pub values: [u64; 2],
    pub settlement: Result<(), DeviceError>,
}

/// How [`Devices::read_scalar`] answered a scalar read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalarRead {
    /// Only [`Devices::read_canonical`] serves it; nothing was read.
    Canonical,
    /// It triggers a debug PC sample of this PE; nothing was read. The
    /// engine supplies the PC ([`Devices::observe_pc_sample`]) and reads.
    Sample(usize),
    /// Read in the same call.
    Served(Result<u64, DeviceError>),
}

/// The board's devices on the guest physical bus. `cpu` is the number of the
/// PE making the access, as [`CpuProfile::processors`] numbers it, and `now`
/// is the system counter value (24 MHz ticks) of the access.
pub trait Devices {
    /// The PE a read triggers a debug PC sample of, when this is a sampler.
    fn pc_sample_target(&self, _physical: u64, _bytes: u8) -> Option<usize> {
        None
    }
    /// Supply that PE's execution PC immediately before its sample trigger.
    fn observe_pc_sample(&mut self, _physical: u64, _pc: u64) {}
    fn read(&mut self, cpu: usize, physical: u64, bytes: u8, now: u64) -> Result<u64, DeviceError>;
    /// Classify a scalar read and serve it when it is plain, in one call: a
    /// board shared between PEs takes its lock once for the three.
    fn read_scalar(&mut self, cpu: usize, physical: u64, bytes: u8, now: u64) -> ScalarRead {
        if self.requires_canonical_read(cpu, physical, DeviceReadShape::Scalar { bytes }) {
            ScalarRead::Canonical
        } else if let Some(sampled) = self.pc_sample_target(physical, bytes) {
            ScalarRead::Sample(sampled)
        } else {
            ScalarRead::Served(self.read(cpu, physical, bytes, now))
        }
    }
    /// Effect-free classification only; this grants no memory permission. A
    /// canonical-required read must never execute through the raw/serial hook.
    fn requires_canonical_read(
        &self,
        _cpu: usize,
        _physical: u64,
        _shape: DeviceReadShape,
    ) -> bool {
        false
    }
    /// Effect-free classification of a write only the canonical path
    /// ([`Devices::write_canonical`]) serves: a machine without canonical
    /// memory forwards it to the one with ([`parallel`]).
    fn requires_canonical_write(&self, _cpu: usize, _physical: u64, _bytes: u8) -> bool {
        false
    }
    /// Refuse before consumption, or return the consumed scalar and any
    /// later settlement failure while the same live canonical guard is held.
    fn read_canonical(
        &mut self,
        _cpu: usize,
        _physical: u64,
        _bytes: u8,
        _now: u64,
        _memory: &mut dyn CanonicalDeviceMemory,
    ) -> Result<ConsumedScalar, DeviceError> {
        Err(DeviceError::TransferRefused(refuse!(
            "canonical device scalar read has no live backend"
        )))
    }
    /// The consuming pair counterpart; no ordinary-read fallback is permitted.
    fn read_pair_canonical(
        &mut self,
        _cpu: usize,
        _physical: u64,
        _bytes: u8,
        _now: u64,
        _memory: &mut dyn CanonicalDeviceMemory,
    ) -> Result<ConsumedPair, DeviceError> {
        Err(DeviceError::TransferRefused(refuse!(
            "canonical device pair read has no live backend"
        )))
    }
    /// Outer Err is proved effect-free. Partial or unknown effects must be
    /// returned explicitly; no raw write fallback or invented ACK is allowed.
    fn write_canonical(
        &mut self,
        _cpu: usize,
        _physical: u64,
        _bytes: u8,
        _value: u64,
        _now: u64,
        _owners: &mut dyn CanonicalDeviceWrite,
    ) -> Result<ConsumedWrite, DeviceError> {
        Err(DeviceError::TransferRefused(refuse!(
            "canonical device write has no live backend"
        )))
    }
    /// One checked whole-pair owner; adjacent scalar registers are never
    /// combined by this default boundary.
    fn write_pair_canonical(
        &mut self,
        _cpu: usize,
        _physical: u64,
        _bytes: u8,
        _values: [u64; 2],
        _now: u64,
        _owners: &mut dyn CanonicalDeviceWrite,
    ) -> Result<ConsumedWrite, DeviceError> {
        Err(DeviceError::TransferRefused(refuse!(
            "canonical device pair write has no live backend"
        )))
    }
    /// Optional whole-pair boundary for a device with a checked consuming
    /// read protocol. `Ok(None)` has no effects and requests ordinary scalar
    /// reads. `Err` refuses before consumption; `Ok(Some(..))` must keep the
    /// data even if forwarding its resulting board effects later fails.
    fn read_pair(
        &mut self,
        _cpu: usize,
        _physical: u64,
        _bytes: u8,
        _now: u64,
    ) -> Result<Option<ConsumedPair>, DeviceError> {
        Ok(None)
    }
    fn write(
        &mut self,
        cpu: usize,
        physical: u64,
        bytes: u8,
        value: u64,
        now: u64,
    ) -> Result<(), DeviceError>;
    /// Complete a device write with exclusive access to the CPU and
    /// native owners. CPU-affecting controllers must preflight every owner,
    /// consume one whole processor batch, then commit infallibly before ACK.
    /// A CPU/lifecycle refusal is a non-substitutable native service failure.
    fn write_with_processors(
        &mut self,
        cpu: usize,
        physical: u64,
        bytes: u8,
        value: u64,
        now: u64,
        _processors: ProcessorWrite<'_>,
    ) -> Result<(), DeviceError> {
        self.write(cpu, physical, bytes, value, now)
    }
    /// [`Self::write_with_processors`] by a machine without canonical
    /// memory, in one call with its classification: `None`, with nothing
    /// written, for a write [`Self::requires_canonical_write`] holds for.
    fn write_plain(
        &mut self,
        cpu: usize,
        physical: u64,
        bytes: u8,
        value: u64,
        now: u64,
        processors: ProcessorWrite<'_>,
    ) -> Option<Result<(), DeviceError>> {
        (!self.requires_canonical_write(cpu, physical, bytes))
            .then(|| self.write_with_processors(cpu, physical, bytes, value, now, processors))
    }
    /// Bring time-driven state up to `now` and report the interrupt lines
    /// the devices drive into PE `cpu`.
    fn lines(&mut self, cpu: usize, now: u64) -> Lines;
    /// The lines the devices drive into PE `cpu` as the last [`Self::lines`]
    /// left them, with no time brought forward: `None` when only `lines`
    /// can tell. The engine asks after a PE switch, before the deadline.
    fn pins(&mut self, _cpu: usize) -> Option<Lines> {
        None
    }
    /// [`Self::pins`] without the FIQ PE `cpu`'s
    /// [`LocalRegisters::fiq`] drives, which that PE reads itself: what a
    /// shared board publishes for it.
    fn latched_pins(&mut self, cpu: usize) -> Option<Lines> {
        self.pins(cpu)
    }
    /// Whether a call under the devices' lock raised PE `cpu`'s
    /// [`LocalRegisters::fiq`] since the last call for it: it must look at
    /// its lines again. Kept for every PE the board has.
    fn take_signalled(&mut self, _cpu: usize) -> bool {
        false
    }
    /// The next counter value at which a device changes state by itself.
    fn deadline(&mut self, _now: u64) -> Option<u64> {
        None
    }
    /// [`Self::lines`], then [`Self::deadline`] at the same `now`: a board
    /// answers both from one advance.
    fn lines_and_deadline(&mut self, cpu: usize, now: u64) -> (Lines, Option<u64>) {
        (self.lines(cpu, now), self.deadline(now))
    }
    /// An MRS (`write` `None`) or MSR by PE `cpu` of a System register the
    /// board implements for all its PEs rather than one core, such as
    /// Apple's fast-IPI registers: `None` for an encoding it does not
    /// implement, else the value read (0 for a write) or why it is refused.
    fn system_register(
        &mut self,
        _cpu: usize,
        _encoding: Encoding,
        _write: Option<u64>,
        _now: u64,
    ) -> Option<Result<u64, &'static str>> {
        None
    }
    /// PE `cpu`'s handle on the registers [`Self::system_register`] serves
    /// that no lock need guard, fixed for the board's life.
    fn local_registers(&self, _cpu: usize) -> Option<Arc<dyn LocalRegisters>> {
        None
    }
    /// The registers any PE reads without the devices' lock, fixed for the
    /// board's life once its PEs run.
    fn passive_registers(&self) -> Option<Arc<dyn PassiveRegisters>> {
        None
    }
    /// Leave the data copies of the devices' DMA to [`Self::take_unlocked`]
    /// from now on: a board shared between PEs runs each outside the
    /// devices' lock. Without this call every copy runs in its transfer.
    fn defer_dma(&mut self) {}
    /// The copies the transfers since the last call left
    /// ([`Self::defer_dma`]), which the caller runs before its next device
    /// call; what they come to reaches the devices at their next transfer.
    fn take_unlocked(&mut self) -> Vec<Box<dyn UnlockedDma>> {
        Vec::new()
    }
    /// [`LocalRegisters::storage`] of PE `cpu`'s `encoding`.
    fn register_storage(&mut self, cpu: usize, encoding: Encoding) -> Option<u64> {
        self.local_registers(cpu)?.storage(encoding)
    }
    /// Whether [`Self::system_register`] may serve `encoding`, fixed for the
    /// board's life: devices shared behind a lock are asked only for these.
    fn serves_register(&self, _encoding: Encoding) -> bool {
        true
    }
    /// A reset or power-off the guest requested from a device, once.
    fn take_reset(&mut self) -> Option<Reset> {
        None
    }
    /// A PE the guest released from reset through a device, once.
    fn take_start(&mut self) -> Option<Start> {
        None
    }
    /// The reset address PE `cpu` leaves reset at, where the board holds it
    /// in a device register (RVBAR_EL2 reads it).
    fn reset_vector(&self, _cpu: usize) -> Option<u64> {
        None
    }
    /// PE `cpu` powered down (`false`: a WFI in CYC_OVRD's power-down
    /// mode) or left reset again (`true`: a start or a wake from power-down).
    fn processor_power(&mut self, _cpu: usize, _on: bool) {}
    /// PE `cpu`'s SIQ_CFG_EL1 now refuses (`true`) or accepts the interrupt
    /// controller's IRQs (`registers::SIQ_CFG_NO_IRQ`).
    fn irq_steering(&mut self, _cpu: usize, _refuses: bool, _now: u64) {}
    /// Whether an input from outside the board, which the host delivers
    /// between runs, can still wake a PE when no device deadline will:
    /// a sleeping SoC's power button. Then a wait of every PE with no
    /// deadline is [`Stop::Waiting`] rather than [`Stop::Idle`].
    fn waits_on_host(&self) -> bool {
        false
    }
    /// Whether a device moved bytes the host feeds or drains since the last
    /// call, such as a bulk transfer of a USB session the host writes and
    /// reads. Then a wait of every PE is [`Stop::Waiting`] at once, so the
    /// host refills or drains it before the board waits on.
    fn take_host_service(&mut self) -> bool {
        false
    }
    /// The physical regions the memory controllers have locked against writes,
    /// first and last byte, by CTRR lock group.
    fn memory_lock(&self) -> super::super::mmu::MemoryLocks {
        [None; 4]
    }
    /// A native bus peer's next bounded DMA step. None adds no polling work.
    fn dma_deadline(&self) -> Option<u64> {
        None
    }
    /// For a machine that advances devices shared with other PEs' machines:
    /// a count of the locked calls of any of them, which moves at every one
    /// that can change a device's deadline. While it moved by the machine's
    /// own last [`Self::lines_and_deadline`] alone, the deadline that call
    /// gave still holds. None: no such count, ask every time.
    fn generation(&self) -> Option<u64> {
        None
    }
    /// Run the DMA a device write by PE `cpu` started, on guest RAM
    /// `memory`, before the next guest access: device DMA completes in the
    /// write's own step.
    fn transfer(
        &mut self,
        _cpu: usize,
        _memory: &mut dyn Memory,
        _now: u64,
    ) -> Result<(), DeviceError> {
        Ok(())
    }
    /// The call a device access by PE `cpu` made through a call gate, after
    /// [`Self::transfer`]. It completes in the access's own step, on the
    /// address space of the thread that made it, outside the devices' lock.
    fn take_call(&mut self, _cpu: usize) -> Option<GateCall> {
        None
    }
    /// Record what a [`Self::take_call`] call came to.
    fn finish_call(&mut self, _outcome: GateOutcome) {}
    /// What follows a device write by PE `cpu`, in one call: the memory
    /// controllers' write lock ([`Self::memory_lock`]), the DMA the write
    /// started ([`Self::transfer`]) on `memory` under that lock, the call it
    /// made ([`Self::take_call`]), and the lock the DMA left. A board shared
    /// between PEs takes its lock once for all of them.
    fn settle_write(&mut self, cpu: usize, now: u64, memory: &mut dyn LockedMemory) -> Settled {
        let lock = self.memory_lock();
        let transfer = self.transfer(cpu, memory.under(lock), now);
        Settled {
            transfer,
            lock,
            call: self.take_call(cpu),
            after: self.memory_lock(),
        }
    }
    /// The model mapped at `physical`, for reports.
    fn device_name(&self, _physical: u64) -> Option<&'static str> {
        None
    }
    /// Host counter ticks this handle waited for a shared board's lock
    /// since the last call.
    fn take_lock_wait(&mut self) -> u64 {
        0
    }
}

/// Guest RAM as device DMA reaches it, under the memory controllers' write
/// lock [`Devices::settle_write`] gives it.
pub trait LockedMemory {
    fn under(&mut self, lock: super::super::mmu::MemoryLocks) -> &mut dyn Memory;
}

/// [`Devices::settle_write`]'s answer.
pub struct Settled {
    pub transfer: Result<(), DeviceError>,
    /// The write lock the DMA ran under, which the call runs under too.
    pub lock: super::super::mmu::MemoryLocks,
    pub call: Option<GateCall>,
    /// The write lock after the DMA.
    pub after: super::super::mmu::MemoryLocks,
}

/// Guest RAM through the host, as device DMA reaches it.
pub(super) struct HostMemory<'a, H: Host> {
    pub(super) host: &'a mut H,
    pub(super) lock: super::super::mmu::MemoryLocks,
}

impl<H: Host> Memory for HostMemory<'_, H> {
    fn read(&mut self, physical: u64, buffer: &mut [u8]) -> bool {
        self.read_progress(physical, buffer).is_ok()
    }
    fn write(&mut self, physical: u64, data: &[u8]) -> bool {
        self.write_progress(physical, data).is_ok()
    }
    fn check_range(
        &mut self,
        physical: u64,
        bytes: usize,
        write: bool,
    ) -> Result<(), stela_contracts::dma::DmaError> {
        if write {
            self.unlocked(physical, bytes)?;
        }
        self.host.check_ram_range(physical, bytes, write)
    }
    fn read_progress(
        &mut self,
        physical: u64,
        buffer: &mut [u8],
    ) -> Result<(), stela_contracts::dma::TransferFault> {
        self.host.read_ram_progress(physical, buffer)
    }
    fn write_progress(
        &mut self,
        physical: u64,
        data: &[u8],
    ) -> Result<(), stela_contracts::dma::TransferFault> {
        self.unlocked(physical, data.len()).map_err(|error| {
            stela_contracts::dma::TransferFault {
                error,
                completed: 0,
            }
        })?;
        self.host.write_ram_progress(physical, data)
    }
}

impl<H: Host> LockedMemory for HostMemory<'_, H> {
    fn under(&mut self, lock: super::super::mmu::MemoryLocks) -> &mut dyn Memory {
        self.lock = lock;
        self
    }
}

impl<H: Host> HostMemory<'_, H> {
    /// Whether a write of `bytes` at `physical` stays clear of every range
    /// the memory controller has locked. Its response to a write inside one
    /// is not modeled, so such a write refuses, naming the range.
    fn unlocked(&self, physical: u64, bytes: usize) -> Result<(), stela_contracts::dma::DmaError> {
        if bytes == 0 {
            return Ok(());
        }
        let last = u64::try_from(bytes - 1)
            .ok()
            .and_then(|n| physical.checked_add(n))
            .ok_or(stela_contracts::dma::DmaError::Memory)?;
        match self
            .lock
            .into_iter()
            .flatten()
            .find(|&(first, end)| physical <= end && first <= last)
        {
            Some((first, end)) => Err(stela_contracts::dma::DmaError::Unsupported(refuse!(
                "DMA write into a range the memory controller locked against writes",
                physical,
                bytes,
                first,
                end
            ))),
            None => Ok(()),
        }
    }
}

/// A powered-off PE leaves reset at `address` (its RVBAR) in EL2h.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Start {
    pub cpu: usize,
    pub address: u64,
}

/// A bus with no devices: every access is [`DeviceError::NoDevice`].
pub struct NoDevices;

impl Devices for NoDevices {
    fn read(&mut self, _: usize, _: u64, _: u8, _: u64) -> Result<u64, DeviceError> {
        Err(DeviceError::NoDevice)
    }
    fn write(&mut self, _: usize, _: u64, _: u8, _: u64, _: u64) -> Result<(), DeviceError> {
        Err(DeviceError::NoDevice)
    }
    fn lines(&mut self, _: usize, _: u64) -> Lines {
        Lines::default()
    }
}

/// What native code does to move a host to another view of one regime
/// ([`Host::native_view`]), as `x86_64::transition::Record` keeps it: load
/// `cr3` while the word at `root_at` is still `root` and the byte at
/// `stale_at` clear, write `view` to the word at `view_at`, and leave
/// RFLAGS.AC as `ac` says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeView {
    pub cr3: u64,
    pub root_at: u64,
    pub root: u64,
    pub stale_at: u64,
    pub view_at: u64,
    pub view: u64,
    pub ac: u64,
}
