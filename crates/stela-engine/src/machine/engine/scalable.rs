//! SVE and SME instructions that passed their access checks
//! (`machine::streaming`), run one at a time as context leaves of the shared
//! compiler (a `Features::LEAF | Features::CONTEXT` plan) on the PE's scalable
//! state (`crate::scalable`) at the streaming vector length. The leaf's
//! profile is the PE's mode (PSTATE.SM and ZA) and the chip's scalable
//! features the engine implements; leaves are cached by PC, word and profile
//! and discarded with the blocks. A leaf's memory accesses come back to the
//! engine one element at a time through the context's memory service
//! ([`serve`]), which translates and checks each as a plain data access,
//! with its watchpoints and, for a tag-checked form, its Tag Check
//! (`engine::tags`), and raises its exception after the leaf returns. A word that Arm's decode
//! makes UNDEFINED on the chip takes that, before its access checks
//! ([`Machine::scalable_undefined`]). What a leaf cannot run stops
//! explicitly: PSTATE.DIT set, whose timing the leaves do not keep. Numerical floating point, which
//! needs Arm's FP semantics, is completed in software
//! ([`Machine::scalable_numerical`]).
use super::{
    DeviceError, Devices, FETCH_PAGE, Fetched, Host, InstallError, Machine, NativeCode, RamTables,
    RequirementKind, Step, Stop, Translated, Variant, device_access::ram_refused,
    scheduler::native_instruction_incomplete, tags::TagCheck,
};
use crate::scalable::ScalableState;
use alloc::string::ToString;
use alloc::{format, string::String, vec::Vec};
use stela_compiler::{
    GuestAddress, GuestState, Options, TargetCpu,
    aarch64::{
        floating::features::NumericalFeatures,
        scalable::{
            ScalableMode, VectorLength,
            features::{ScalableFeature, ScalableFeatures},
            profile::ScalableControls,
        },
        sme::movement,
    },
    block::{
        BlockProfile, CompilationPlan, Features,
        scalable::{
            AdmittedScalableNumericalInstruction, admit_scalable_numerical_instruction,
            decode_scalable_numerical_instruction,
        },
    },
    context::{
        ExecutionContext,
        memory::{
            MEMORY_CONTEXT_BYTES, MEMORY_CONTEXT_VERSION, MemoryContextView, MemoryRequest,
            MemoryRequirements, MemoryStatus, SpAlignmentRequirement, flags, profile_requirements,
        },
    },
    scalable_context::ScalableProfile,
};
use {
    crate::machine::cpu::Exception, crate::machine::cpu::STREAMING_VECTOR_BYTES,
    crate::machine::debug, crate::machine::exception::Class, crate::machine::mmu,
    crate::machine::mmu::Access, crate::machine::mmu::MemoryType, crate::machine::mmu::alignment,
    crate::machine::streaming, crate::machine::streaming::VectorDestinations,
    crate::machine::system::File, stela_contracts::Feature,
};

/// The numerical features of every Apple chip's identification: no
/// FEAT_SME_FA64 (FullA64 off), FEAT_EBF16 or FEAT_RPRES.
const NUMERICAL_FEATURES: NumericalFeatures = NumericalFeatures::new(false, false, false);
/// RET X30, which ends a context leaf.
const RET_X30: u32 = 0xd65f_03c0;

/// The chip features the compiler's scalable profiles name.
const FEATURES: [(Feature, ScalableFeature); 17] = [
    (Feature::FEAT_SME, ScalableFeature::Sme),
    (Feature::FEAT_SME2, ScalableFeature::Sme2),
    (Feature::FEAT_SME2p1, ScalableFeature::Sme2p1),
    (Feature::FEAT_SME_F64F64, ScalableFeature::SmeF64F64),
    (Feature::FEAT_SME_I16I64, ScalableFeature::SmeI16I64),
    (Feature::FEAT_SME_B16B16, ScalableFeature::SmeB16B16),
    (Feature::FEAT_SME_F16F16, ScalableFeature::SmeF16F16),
    (Feature::FEAT_SME_F8F16, ScalableFeature::SmeF8F16),
    (Feature::FEAT_SME_F8F32, ScalableFeature::SmeF8F32),
    (Feature::FEAT_SME_LUTv2, ScalableFeature::SmeLutV2),
    (Feature::FEAT_SVE_B16B16, ScalableFeature::SveB16B16),
    (Feature::FEAT_BF16, ScalableFeature::Bf16),
    (Feature::FEAT_I8MM, ScalableFeature::I8mm),
    (Feature::FEAT_FAMINMAX, ScalableFeature::FaMinMax),
    (Feature::FEAT_LUT, ScalableFeature::Lut),
    (Feature::FEAT_FP8, ScalableFeature::Fp8),
    (Feature::FEAT_FPMR, ScalableFeature::Fpmr),
];

/// The scalable features of the PE whose register file is `registers`
/// that the engine implements: an instruction of a feature it has but the
/// engine lacks fails to compile, and stops.
pub(crate) fn features(registers: &File) -> ScalableFeatures {
    features_where(|feature| registers.implemented(feature) == Ok(true))
}

/// The scalable features `keep` selects of [`FEATURES`].
fn features_where(keep: impl Fn(Feature) -> bool) -> ScalableFeatures {
    FEATURES
        .iter()
        .filter(|(feature, _)| keep(*feature))
        .fold(ScalableFeatures::empty(), |features, (_, scalable)| {
            features.with(*scalable)
        })
}

/// A scalable leaf cannot be prepared or installed for this PE.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PreparationError {
    MissingState,
    InvalidVectorLength,
    Compile(stela_compiler::CompileError),
    MissingMemoryRequirements,
    CodeArenaFull,
}
impl From<stela_compiler::CompileError> for PreparationError {
    fn from(error: stela_compiler::CompileError) -> Self {
        Self::Compile(error)
    }
}
impl core::fmt::Display for PreparationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingState => f.write_str("a PE without SME state"),
            Self::InvalidVectorLength => f.write_str("the streaming vector length"),
            Self::Compile(error) => error.fmt(f),
            Self::MissingMemoryRequirements => {
                f.write_str("a context leaf without memory requirements")
            }
            Self::CodeArenaFull => f.write_str("the leaf exceeds the code arena"),
        }
    }
}
impl core::error::Error for PreparationError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Compile(error) => Some(error),
            _ => None,
        }
    }
}

/// What the engine runs an SVE or SME word as.
enum Prepared {
    Movprfx(Movprfx),
    Numerical(AdmittedScalableNumericalInstruction),
    Leaf(Leaf),
    /// UNDEFINED: the word needs a feature the chip lacks, or Debug state.
    Undefined,
}

/// MOVPRFX (unpredicated, and predicated with merging or zeroing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Movprfx {
    destination: u8,
    source: u8,
    /// The governing predicate, element bytes and merging.
    predicated: Option<(u8, usize, bool)>,
}

impl Movprfx {
    const UNPREDICATED_MASK: u32 = 0xffff_fc00;
    const UNPREDICATED: u32 = 0x0420_bc00;
    const PREDICATED_MASK: u32 = 0xff3e_e000;
    const PREDICATED: u32 = 0x0410_2000;

    fn decode(word: u32) -> Option<Self> {
        let (destination, source) = ((word & 0x1f) as u8, ((word >> 5) & 0x1f) as u8);
        let predicated = if word & Self::UNPREDICATED_MASK == Self::UNPREDICATED {
            None
        } else if word & Self::PREDICATED_MASK == Self::PREDICATED {
            let predicate = ((word >> 10) & 0b111) as u8;
            Some((predicate, 1 << ((word >> 22) & 0b11), word & 1 << 16 != 0))
        } else {
            return None;
        };
        Some(Self {
            destination,
            source,
            predicated,
        })
    }

    /// Arm's MOVPRFX: Zd = Zn, or for each element its active value,
    /// keeping (merging) or zeroing the inactive ones.
    fn apply(self, scalable: &mut ScalableState, state: &mut GuestState) -> crate::Result<()> {
        let mut result = [0u8; STREAMING_VECTOR_BYTES];
        scalable.read_z(state, self.source, &mut result)?;
        if let Some((predicate, element, merging)) = self.predicated {
            let mut mask = [0u8; STREAMING_VECTOR_BYTES / 8];
            scalable.read_predicate(predicate, &mut mask)?;
            let mut old = [0u8; STREAMING_VECTOR_BYTES];
            scalable.read_z(state, self.destination, &mut old)?;
            for (index, (new, old)) in result
                .chunks_mut(element)
                .zip(old.chunks(element))
                .enumerate()
            {
                let bit = index * element;
                if mask[bit / 8] & 1 << (bit % 8) == 0 {
                    if merging {
                        new.copy_from_slice(old);
                    } else {
                        new.fill(0);
                    }
                }
            }
        }
        scalable.write_z(state, self.destination, &result)
    }
}

/// What a PE's context leaf of `profile` compiles for.
fn leaf_options(profile: ScalableProfile, target: TargetCpu) -> Options {
    Options::new(
        BlockProfile::new(Features::LEAF | Features::CONTEXT),
        target,
    )
    .with_scalable(profile)
}

/// A context leaf's entry: the two-argument context ABI.
#[cfg(target_arch = "x86_64")]
type Entry = unsafe extern "sysv64" fn(*mut GuestState, *mut ExecutionContext);
#[cfg(not(target_arch = "x86_64"))]
type Entry = unsafe extern "C" fn(*mut GuestState, *mut ExecutionContext);

/// An installed leaf and what its memory accesses require of the PE.
#[derive(Debug, Clone, Copy)]
pub(super) struct Leaf {
    pub(super) code: NativeCode,
    memory: MemoryRequirements,
}

/// Why a memory request failed: the exception its element takes, or why the
/// engine stops. Either is reported to the leaf as `status`.
enum Failure {
    Exception(MemoryStatus, Exception),
    Stop(MemoryStatus, Stop),
}

fn ram_stop_status(stop: &Stop) -> Option<MemoryStatus> {
    let Stop::Unsupported(requirement) = stop else {
        return None;
    };
    match &requirement.kind {
        RequirementKind::NonresidentRam { .. } => Some(MemoryStatus::Nonresident),
        RequirementKind::RamTransfer { status, .. } => Some(*status),
        RequirementKind::NativePreparation(_) => Some(MemoryStatus::Domain),
        _ => None,
    }
}

/// The memory service of one leaf call: its requests reach `machine`, whose
/// GuestState and scalable state are lent to the leaf meanwhile.
struct Service<H: Host, D: Devices> {
    machine: *mut Machine<H, D>,
    word: u32,
    failure: Option<Failure>,
}

translated_callee! {
/// The `MemoryCallback` of [`Service`]: one architectural element, in
/// order, which the engine translates, checks and transfers.
unsafe fn serve<H: Host, D: Devices>(
    view: *mut MemoryContextView,
    request: *const MemoryRequest,
) -> u32 {
    if view.is_null() || request.is_null() {
        return MemoryStatus::InvalidRequest as u32;
    }
    // SAFETY: the leaf passes the view the engine built and a request in its
    // own frame, for this call.
    let (view, request) = unsafe { (&mut *view, &*request) };
    // SAFETY: `opaque` is the Service of this call, and its machine is not
    // otherwise borrowed while the leaf runs.
    let service = unsafe { &mut *view.opaque.cast::<Service<H, D>>() };
    // SAFETY: `machine` belongs to this call's service and is not otherwise borrowed while the leaf
    // runs.
    let machine = unsafe { &mut *service.machine };
    let word = service.word;
    let result = crate::platform::unwind::invoke(core::panic::AssertUnwindSafe(|| {
        machine.scalable_access(word, request)
    }));
    let mut failure = match result {
        Ok(Ok(())) => {
            view.completed_bytes += u64::from(request.bytes);
            return MemoryStatus::Success as u32;
        }
        Ok(Err(failure)) => failure,
        #[allow(unreachable_patterns)]
        Err(_) => Failure::Stop(
            MemoryStatus::Domain,
            machine.unsupported(RequirementKind::Instruction {
                word,
                detail: "the SVE and SME memory service panicked".into(),
            }),
        ),
    };
    let status = match &failure {
        Failure::Exception(status, _) | Failure::Stop(status, _) => *status,
    };
    if let Failure::Stop(_, stop) = &mut failure
        && ram_stop_status(stop).is_some()
        && let Stop::Unsupported(requirement) = stop
    {
        requirement.address = Some(request.address);
        match &mut requirement.kind {
            RequirementKind::RamTransfer { completed, .. } => {
                view.completed_bytes += *completed as u64;
                *completed = view.completed_bytes as usize;
            }
            RequirementKind::NonresidentRam { frame } if view.completed_bytes != 0 => {
                // The prefix may already have written RAM or consumed a device.
                // Keep it distinct from a missing frame before any callback completed.
                requirement.kind = RequirementKind::RamTransfer {
                    physical: *frame,
                    write: request.flags & flags::WRITE != 0,
                    completed: view.completed_bytes as usize,
                    status,
                };
            }
            _ => {}
        }
    }
    service.failure = Some(failure);
    view.record(status, request.address, request.pc, view.completed_bytes);
    status as u32
}
}

impl<H: Host, D: Devices> Machine<H, D> {
    /// Run the SVE or SME instruction `word` at `pc`, whose access checks
    /// passed.
    pub(super) fn scalable_instruction(&mut self, pc: u64, word: u32) -> Step {
        let refuse = |machine: &Self, detail: &str| {
            Err(machine.unsupported(RequirementKind::Instruction {
                word,
                detail: detail.into(),
            }))
        };
        // Version 2 profiles require PSTATE.DIT clear: the leaves are not
        // compiled for data-independent timing.
        if self.cpu.pstate.dit {
            return refuse(
                self,
                "SVE and SME instructions with PSTATE.DIT set are not run by the system engine",
            );
        }
        let profile = match self.scalable_profile() {
            Ok(profile) => profile,
            Err(error) => return refuse(self, &error.to_string()),
        };
        let leaf = match self.prepare_scalable(pc, word, profile) {
            Ok(Prepared::Movprfx(prefix)) => return self.movprfx(pc, word, prefix),
            Ok(Prepared::Numerical(admitted)) => return self.scalable_numerical(admitted),
            Ok(Prepared::Undefined) => {
                self.cpu.step_advance = false;
                return self.undefined();
            }
            Ok(Prepared::Leaf(leaf)) => leaf,
            Err(error) => return refuse(self, &error.to_string()),
        };
        if let Err(detail) = self.admit_memory(leaf.memory) {
            return refuse(self, detail);
        }
        let Some(entry) = self.host.context_entry(leaf.code) else {
            return refuse(
                self,
                "this host does not run the context leaves of SVE and SME instructions",
            );
        };
        // SAFETY: the host gives the address of the installed leaf, compiled
        // for the two-argument context ABI.
        let entry: Entry = unsafe { core::mem::transmute::<usize, Entry>(entry) };
        let retired = self.cpu.retired;
        let step = self.completing(pc, |machine| machine.call_leaf(entry, profile, word));
        if native_instruction_incomplete(&step) {
            self.cpu.retired = retired;
        }
        step
    }

    /// MOVPRFX on its own, as Arm permits it to run: the move it is,
    /// whatever follows (the follower then runs as the next instruction).
    fn movprfx(&mut self, pc: u64, word: u32, prefix: Movprfx) -> Step {
        self.completing(pc, |machine| {
            let Some(scalable) = machine.cpu.scalable.as_mut() else {
                return Err(machine.unsupported(RequirementKind::Instruction {
                    word,
                    detail: "a PE without SME state".into(),
                }));
            };
            if prefix.apply(scalable, &mut machine.cpu.state).is_err() {
                return Err(machine.unsupported(RequirementKind::Instruction {
                    word,
                    detail: "the streaming Z and P registers".into(),
                }));
            }
            machine.statistics.blocks += 1;
            machine.next()
        })
    }

    /// Run the scalar FP or SIMD&FP load or store `word` at `pc` in
    /// streaming mode: as a one-instruction block of the system-mode
    /// compiler, compiled apart from the blocks that end before it, after
    /// which the Z bits above 128 of each register it wrote are zero, as
    /// Arm's `V[]` makes them. An instruction that took an exception wrote
    /// none.
    pub(super) fn streaming_scalar(&mut self, pc: u64, word: u32) -> Step {
        let Some(VectorDestinations(destinations)) = streaming::vector_destinations(word) else {
            return Err(self.unsupported(RequirementKind::Instruction {
                word,
                detail: "scalar FP in streaming mode whose vector destinations are not known"
                    .into(),
            }));
        };
        let physical = match self.fetch_physical(pc) {
            Ok(physical) => physical,
            Err(Ok(exception)) => {
                self.cpu.step_advance = false;
                self.take(exception);
                return Ok(());
            }
            Err(Err(stop)) => return Err(stop),
        };
        let variant = Variant {
            fp: self.floating_point_enabled(),
            checks: self.memory_checks(),
            single: true,
            scalable: None,
            system: self.native_system(),
            guarded_page: self.fetch_guarded(pc),
            timing: self.cpu.pstate.dit && self.timing_target().is_some(),
            translated: false,
        };
        let variant = self.page_variant(variant, physical);
        let indexed = self
            .known_variant(variant)
            .and_then(|id| self.code.find(pc, physical, id));
        let index = match indexed {
            Some(index) => index,
            None => match self.compile(pc, physical, variant)? {
                Fetched::Block(index) => index,
                Fetched::Exception(exception) => {
                    self.cpu.step_advance = false;
                    self.take(exception);
                    return Ok(());
                }
                Fetched::Checked(_) => {
                    return Err(self.unsupported(RequirementKind::Instruction {
                        word,
                        detail: "a streaming scalar FP block that ends before itself".into(),
                    }));
                }
            },
        };
        let exceptions = self.statistics.exceptions;
        self.execute(index, 0)?;
        if self.statistics.exceptions != exceptions {
            return Ok(());
        }
        let Some(scalable) = self.cpu.scalable.as_mut() else {
            return Ok(());
        };
        let zeroed = destinations.into_iter().flatten().try_for_each(|register| {
            let value = self.cpu.state.v[usize::from(register)];
            scalable.write_v(&mut self.cpu.state, register, value)
        });
        if zeroed.is_err() {
            return Err(self.unsupported(RequirementKind::Instruction {
                word,
                detail: "the streaming Z registers".into(),
            }));
        }
        Ok(())
    }

    /// Complete the numerical floating-point instruction `instruction` in
    /// software (`floating::scalable`), with FPCR's controls and FPSR's
    /// cumulative flags; in streaming mode without FEAT_SME_FA64 Arm's
    /// `FPProcessException` takes no FP trap, so none is.
    fn scalable_numerical(&mut self, admitted: AdmittedScalableNumericalInstruction) -> Step {
        let (pc, word) = (admitted.instruction_pc().0, admitted.word());
        let refuse = |machine: &Self, detail: String| {
            Err(machine.unsupported(RequirementKind::Instruction { word, detail }))
        };
        self.completing(pc, |machine| {
            let Some(scalable) = machine.cpu.scalable.as_mut() else {
                return refuse(machine, "a PE without SME state".into());
            };
            if let Err(error) =
                crate::floating::scalable::complete(&mut machine.cpu.state, scalable, &admitted)
            {
                return refuse(machine, format!("{error}"));
            }
            machine.statistics.blocks += 1;
            machine.next()
        })
    }

    /// Call `entry` on the PE's state, lent with its scalable registers, and
    /// complete the instruction or take the exception its memory access
    /// raised.
    fn call_leaf(&mut self, entry: Entry, profile: ScalableProfile, word: u32) -> Step {
        let mut state = core::mem::take(&mut self.cpu.state);
        let Some(mut scalable) = self.cpu.scalable.take() else {
            self.cpu.state = state;
            return Err(self.unsupported(RequirementKind::Instruction {
                word,
                detail: "a PE without SME state".into(),
            }));
        };
        let mut service = Service {
            machine: self as *mut Self,
            word,
            failure: None,
        };
        let mut memory = MemoryContextView {
            version: MEMORY_CONTEXT_VERSION,
            bytes: MEMORY_CONTEXT_BYTES,
            callback: serve::<H, D>,
            opaque: (&raw mut service).cast(),
            fault_kind: MemoryStatus::Success as u32,
            fault_access: 0,
            fault_address: 0,
            fault_pc: 0,
            completed_bytes: 0,
            reserved_tail: 0,
        };
        // SAFETY: the view lends the PE's register buffers for this one
        // call, which the leaf compiled for `profile` reaches only through
        // it, and keeps no pointer to. Its memory service reaches the machine
        // through `service`, while `state` and `scalable` are out of it.
        let called = unsafe {
            scalable.with_view(|view| {
                let mut context = ExecutionContext {
                    memory: (&raw mut memory).cast(),
                    scalable: view,
                    ..ExecutionContext::default()
                };
                profile.validate_view(&context, view)?;
                entry(&mut state, &mut context);
                Ok::<_, stela_compiler::CompileError>(())
            })
        };
        self.cpu.state = state;
        self.cpu.scalable = Some(scalable);
        if let Err(error) = called {
            return Err(self.unsupported(RequirementKind::Instruction {
                word,
                detail: error.to_string(),
            }));
        }
        match (MemoryStatus::from_code(memory.fault_kind), service.failure) {
            (Some(MemoryStatus::Success), None) => {}
            (_, Some(Failure::Exception(_, exception))) => {
                self.take(exception);
                return Ok(());
            }
            (_, Some(Failure::Stop(_, stop))) => return Err(stop),
            // The leaf checks SP alignment itself, where the access needs it
            // checked (`admit_memory`).
            (Some(MemoryStatus::StackAlignment), None) => {
                self.take(Exception::synchronous(Class::SpAlignment, 0, self.cpu.pc));
                return Ok(());
            }
            (status, None) => {
                return Err(self.unsupported(RequirementKind::Instruction {
                    word,
                    detail: format!("the leaf reported memory status {status:?} by itself"),
                }));
            }
        }
        self.statistics.blocks += 1;
        self.next()?;
        match self.devices.take_reset() {
            Some(reason) => Err(Stop::Reset(reason)),
            None => Ok(()),
        }
    }

    /// Whether the PE's controls are those the leaf's memory accesses were
    /// compiled for: SP alignment checked as the access requires, and
    /// nothing the engine's memory service does not provide.
    fn admit_memory(&mut self, memory: MemoryRequirements) -> Result<(), &'static str> {
        let known = profile_requirements::NATIVE_ACCESS
            | profile_requirements::TAG_GRANULES
            | profile_requirements::SCALABLE_TAG_CHECKS;
        if memory.required_profile & !known != 0
            || memory.required_flags & (flags::ATOMIC | flags::ORDERED | flags::EXCLUSIVE) != 0
            || !memory.required_address_checks.is_empty()
            || memory.required_pointer_arithmetic.is_some()
        {
            return Err("an SVE or SME memory access the engine's memory service does not provide");
        }
        match (memory.required_sp_alignment, self.stack_alignment_checked()) {
            (SpAlignmentRequirement::Enabled, false) => Err(
                "an SP-based SVE or SME access with SCTLR_ELx.SA clear, whose leaf checks SP alignment",
            ),
            (SpAlignmentRequirement::Disabled, true) => Err(
                "an SP-based SVE or SME access with SCTLR_ELx.SA set, whose leaf does not check SP alignment",
            ),
            _ => Ok(()),
        }
    }

    /// One element of an SVE or SME access: translated and checked in
    /// Arm's order (alignment, each page's translation and permission, the
    /// memory type's alignment, watchpoints), then read or written whole.
    fn scalable_access(&mut self, word: u32, request: &MemoryRequest) -> Result<(), Failure> {
        let stop = |machine: &Self, detail: &str| {
            Failure::Stop(
                MemoryStatus::Unsupported,
                machine.unsupported(RequirementKind::Instruction {
                    word,
                    detail: detail.into(),
                }),
            )
        };
        let bytes = request.bytes as usize;
        if !request.has_supported_ordinary_shape()
            || request.flags & (flags::ASIMD | flags::PAIR) != 0
            || request.data.is_null()
            || request.pc != self.cpu.pc
        {
            return Err(stop(self, "a malformed SVE or SME memory request"));
        }
        let write = request.flags & flags::WRITE != 0;
        let address = request.address;
        let shape = alignment::Shape {
            kind: alignment::Kind::Plain,
            bytes: bytes as u64,
            pair: false,
        };
        let aligned = shape.aligned(address);
        let access = if write { Access::Write } else { Access::Read };
        let controls = self.cpu.controls(self.cpu.regime());
        let requester = self.cpu.requester();
        let registered = self
            .operand_ram()
            .map_err(|reason| self.scalable_ram_failure(reason))?;
        let mut tables = registered.as_ref().map(|ram| ram.tables());
        let mut pieces: Vec<(u64, usize, MemoryType)> = Vec::with_capacity(2);
        let mut done = 0;
        while done < bytes {
            let virtual_address = address.wrapping_add(done as u64);
            let length =
                ((FETCH_PAGE - (virtual_address & (FETCH_PAGE - 1))) as usize).min(bytes - done);
            let outcome = match tables.as_mut() {
                Some(tables) => {
                    let outcome = mmu::translate_checked(
                        &controls,
                        virtual_address,
                        access,
                        requester,
                        shape,
                        tables,
                    );
                    tables
                        .outcome(outcome)
                        .map_err(|reason| self.scalable_ram_failure(reason))?
                }
                None => mmu::translate_checked(
                    &controls,
                    virtual_address,
                    access,
                    requester,
                    shape,
                    &mut RamTables(&mut self.host),
                ),
            };
            match outcome {
                // An unaligned element crossing into Device memory faults
                // there too.
                Translated::Mapped(translation)
                    if done > 0 && !aligned && translation.memory == MemoryType::Device =>
                {
                    let fault = mmu::Fault {
                        kind: mmu::FaultKind::Alignment,
                        level: 0,
                        second_stage: None,
                    };
                    let exception = self.data_abort_exception(address, fault, write, false);
                    return Err(Failure::Exception(MemoryStatus::Alignment, exception));
                }
                Translated::Mapped(translation) => {
                    pieces.push((translation.physical, length, translation.memory));
                }
                Translated::Fault(fault) => {
                    let status = match fault.kind {
                        mmu::FaultKind::Alignment => MemoryStatus::Alignment,
                        mmu::FaultKind::Permission => MemoryStatus::Permission,
                        _ => MemoryStatus::Translation,
                    };
                    let exception = self.data_abort_exception(virtual_address, fault, write, false);
                    return Err(Failure::Exception(status, exception));
                }
                Translated::Unsupported(unsupported) => {
                    return Err(Failure::Stop(
                        MemoryStatus::Unsupported,
                        self.unsupported(RequirementKind::Translation(unsupported)),
                    ));
                }
            }
            done += length;
        }
        let watched = debug::Access {
            address,
            size: bytes as u64,
            read: !write,
            write,
            maintenance: false,
            el: self.cpu.pstate.el,
        };
        if let Some(exception) = self.watchpoint_exception(&watched, self.cpu.pc) {
            return Err(Failure::Exception(MemoryStatus::Permission, exception));
        }
        if request.flags & flags::TAG_CHECKED != 0 {
            match self.tag_check_with_tables(&watched, requester, None, tables.as_mut()) {
                Ok(TagCheck::Failed(exception)) => {
                    return Err(Failure::Exception(MemoryStatus::TagCheck, exception));
                }
                Ok(TagCheck::Passed | TagCheck::Untranslated) => {}
                Err(stop) => {
                    return Err(Failure::Stop(
                        ram_stop_status(&stop).unwrap_or(MemoryStatus::Unsupported),
                        stop,
                    ));
                }
            }
        }
        let physical_pieces = || {
            pieces
                .iter()
                .map(|&(physical, length, _)| (physical, length))
        };
        let ram = if let Some(registered) = &registered {
            registered
                .resident(&self.host, physical_pieces())
                .map_err(|reason| self.scalable_ram_failure(reason))?
        } else {
            physical_pieces().all(|(physical, length)| {
                let mut probe = [0u8; 1];
                self.host.read_ram(physical, &mut probe)
                    && self.host.read_ram(physical + length as u64 - 1, &mut probe)
            })
        };
        // SAFETY: the leaf's request names `bytes` of private staging.
        let data = unsafe { core::slice::from_raw_parts_mut(request.data, bytes) };
        if ram {
            if let Some(registered) = &registered {
                // SAFETY: this is the same loan that admitted every piece.
                return unsafe {
                    registered.transfer(&mut self.host, physical_pieces(), data, write)
                }
                .map_err(|reason| self.scalable_ram_failure(reason));
            }
            let mut offset = 0;
            for (physical, length, _) in pieces {
                let part = &mut data[offset..offset + length];
                if write {
                    self.host.write_ram(physical, part);
                } else {
                    self.host.read_ram(physical, part);
                }
                offset += length;
            }
            return Ok(());
        }
        let [(physical, _, _)] = pieces[..] else {
            return Err(stop(self, "an SVE or SME element crossing out of RAM"));
        };
        if bytes > size_of::<u64>() {
            return Err(stop(self, "a 16-byte SVE or SME element of a device"));
        }
        let canonical = registered.is_some();
        // A real device may need to acquire the exclusive RAM service.
        drop(registered);
        if canonical
            && write
            && self.cpu.canonical_memory.is_some()
            && self.serial_hold.is_none()
            && self
                .devices
                .requires_canonical_write(self.current, physical, bytes as u8)
        {
            return Err(self.scalable_ram_failure(ram_refused()));
        }
        let now = self.now();
        let mut value = [0u8; size_of::<u64>()];
        let result = if write {
            value[..bytes].copy_from_slice(data);
            self.device_write(physical, bytes as u8, u64::from_le_bytes(value), now)
                .map(|()| 0)
        } else {
            self.device_read(physical, bytes as u8, now)
        };
        match result {
            Ok(read) => {
                if !write {
                    data.copy_from_slice(&read.to_le_bytes()[..bytes]);
                }
                Ok(())
            }
            Err(DeviceError::NoDevice) => Err(Failure::Stop(
                MemoryStatus::Unsupported,
                self.scalable_device(address, physical, write, "no device model".into()),
            )),
            Err(DeviceError::Refused(detail) | DeviceError::TransferRefused(detail)) => {
                Err(Failure::Stop(
                    MemoryStatus::Unsupported,
                    self.scalable_device(address, physical, write, detail.to_string()),
                ))
            }
        }
    }

    fn scalable_ram_failure(&self, reason: RequirementKind) -> Failure {
        let stop = self.unsupported(reason);
        Failure::Stop(ram_stop_status(&stop).unwrap_or(MemoryStatus::Domain), stop)
    }

    fn scalable_device(&self, address: u64, physical: u64, write: bool, detail: String) -> Stop {
        let mut requirement = self.requirement(RequirementKind::Device {
            physical,
            write,
            detail,
        });
        requirement.address = Some(address);
        Stop::Unsupported(requirement)
    }

    /// What `word` at `pc` runs as under `profile`, compiling its leaf.
    fn prepare_scalable(
        &mut self,
        pc: u64,
        word: u32,
        profile: ScalableProfile,
    ) -> Result<Prepared, PreparationError> {
        if let Some(prefix) = Movprfx::decode(word) {
            return Ok(Prepared::Movprfx(prefix));
        }
        let result = match decode_scalable_numerical_instruction(GuestAddress(pc), word)
            .map_err(PreparationError::from)?
        {
            Some(instruction) => {
                admit_scalable_numerical_instruction(instruction, profile, NUMERICAL_FEATURES)
                    .map(Prepared::Numerical)
                    .map_err(PreparationError::from)
            }
            None => self.scalable_leaf(pc, word, profile).map(Prepared::Leaf),
        };
        match result {
            Err(_) if self.scalable_undefined(pc, word) => Ok(Prepared::Undefined),
            result => result,
        }
    }

    /// Whether the SVE or SME word `word` at `pc` decodes as UNDEFINED on
    /// this chip, which Arm's decode raises before any access check, in any
    /// mode: MOVT between ZT0 and a general-purpose register, defined only
    /// in Debug state, which this engine never enters (as the System
    /// instructions of Debug state), and a word that compiles with every
    /// scalable feature but not with those the chip has, engine-implemented
    /// or not, so that `IsFeatureImplemented` fails, such as BFADD inside
    /// the M4's FADD encoding.
    pub(super) fn scalable_undefined(&self, pc: u64, word: u32) -> bool {
        if movement::decode(word).is_some_and(|operand| operand.requirements.halted) {
            return true;
        }
        let Some(length) = VectorLength::from_bytes(STREAMING_VECTOR_BYTES) else {
            return false;
        };
        let chip = self.profile.chip;
        let compiles = |features: ScalableFeatures| {
            let controls = ScalableControls::new(length, features.contains(ScalableFeature::Sme2));
            let Ok(profile) = ScalableProfile::new_with_features(
                length,
                ScalableMode::new(true, true),
                features,
                controls,
            ) else {
                return false;
            };
            match decode_scalable_numerical_instruction(GuestAddress(pc), word) {
                Ok(Some(instruction)) => {
                    admit_scalable_numerical_instruction(instruction, profile, NUMERICAL_FEATURES)
                        .is_ok()
                }
                Ok(None) => {
                    let source: Vec<u8> = [word, RET_X30]
                        .iter()
                        .flat_map(|word| word.to_le_bytes())
                        .collect();
                    let options = leaf_options(profile, self.target);
                    stela_compiler::compile(&source, GuestAddress(pc), &options).is_ok()
                }
                Err(_) => false,
            }
        };
        compiles(features_where(|_| true)) && !compiles(features_where(|f| f.on(chip)))
    }

    /// The profile of the PE's current scalable state.
    fn scalable_profile(&self) -> Result<ScalableProfile, PreparationError> {
        let scalable = self
            .cpu
            .scalable
            .as_ref()
            .ok_or(PreparationError::MissingState)?;
        let features = features(&self.cpu.registers);
        let length = VectorLength::from_bytes(STREAMING_VECTOR_BYTES)
            .ok_or(PreparationError::InvalidVectorLength)?;
        // The FP8 forms read FPMR, which the PE reaches once their
        // CheckFPMREnabled passes.
        let controls = ScalableControls::new(length, features.contains(ScalableFeature::Sme2))
            .with_fpmr_enabled(streaming::fpmr_enabled(&self.cpu));
        ScalableProfile::new_with_features(length, scalable.mode(), features, controls)
            .map_err(PreparationError::from)
    }

    /// The installed leaf of `word` at `pc` for `profile`, compiling it on
    /// first use.
    fn scalable_leaf(
        &mut self,
        pc: u64,
        word: u32,
        profile: ScalableProfile,
    ) -> Result<Leaf, PreparationError> {
        if let Some(leaf) = self.scalable_leaves.get(&(pc, word, profile)) {
            return Ok(*leaf);
        }
        let source: Vec<u8> = [word, RET_X30]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let plan = CompilationPlan::new(
            &source,
            GuestAddress(pc),
            &leaf_options(profile, self.target),
        )
        .map_err(PreparationError::from)?;
        let memory = plan
            .memory_requirements()
            .ok_or(PreparationError::MissingMemoryRequirements)?;
        let compiled = plan.compile().map_err(PreparationError::from)?;
        let native = &compiled.native.x86_64_code;
        let code = match self.host.install(native) {
            Ok(code) => code,
            Err(InstallError::Full) => {
                self.reclaim_leaves();
                self.host
                    .install(native)
                    .map_err(|_| PreparationError::CodeArenaFull)?
            }
        };
        self.statistics.compiled += 1;
        let leaf = Leaf { code, memory };
        self.scalable_leaves.insert((pc, word, profile), leaf);
        Ok(leaf)
    }
}

#[cfg(test)]
mod tests;
