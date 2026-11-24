//! Native block dispatch.
//!
//! Guest code is fetched from registered regions and translated a block at a
//! time; data accesses use direct RAM. Replacing code invalidates the dependent
//! native blocks before returning.

mod aot;
mod discovery;
mod outcome;
pub(crate) mod scalable;
pub(crate) mod services;
pub use services::PauthFault;
mod validate;

pub use discovery::{
    DeferredDynamicEdge, DiscoveryIssue, DiscoveryProblem, DiscoveryReport, PreparedDiscoveryBlock,
};
pub use outcome::{
    ClockEvent, PointerAuthenticationEvent, RunOutcome, StopReason, ThreadRegisterEvent,
    ThreadState,
};
use stela_contracts::{refuse, require};

use crate::Result;
use crate::{execution::platform, floating, memory, pauth, system};
#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, string::ToString, vec::Vec};
use alloc::{collections::BTreeMap, rc::Rc};
use aot::ArtifactPreparation;
use core::marker::PhantomData;
use services::{
    RuntimeServices, complete_pauth_with_memory, runtime_pauth_key, serialize_instruction_execution,
};
#[cfg(feature = "std")]
use std::path::PathBuf;
use stela_compiler::{
    CompileError, GuestAddress,
    block::{
        BlockExit, BlockProfile, ExitKind, Features, MAX_BLOCK_INSTRUCTIONS,
        PointerAuthenticationOperation, analyze_with_profile,
    },
};
use validate::{
    validate_amx_exit, validate_clock_exit, validate_interior_word, validate_numerical_exit,
    validate_pauth_exit, validate_source_map, validate_system_exit, validate_terminal_word,
    validate_thread_register_exit, validate_trap_exit,
};

const INSTRUCTION_BYTES: usize = 4;
const ZERO_REGISTER_INDEX: u8 = 31;
const REGISTER_COUNT: u8 = ZERO_REGISTER_INDEX + 1;

#[derive(Debug)]
struct CodeRegion {
    base: u64,
    source: Box<[u8]>,
    end: u64,
    generation: u64,
}

#[derive(Debug)]
struct LoadedBlock {
    profile: BlockProfile,
    region: Rc<CodeRegion>,
    source_bytes: usize,
    prepared: crate::artifact::LoadedArtifact,
    mapping: platform::Mapping,
}

/// Source registration, resolution and execution are serialized through &mut.
/// No function pointer, mutable code slice or executable mapping escapes.
#[derive(Debug)]
pub struct BlockRuntime {
    address_space: u64,
    profile: BlockProfile,
    target: stela_compiler::TargetCpu,
    preparation: ArtifactPreparation,
    next_generation: u64,
    // Both source mutation methods evict every block referencing the old
    // snapshot. An exact-PC hit can use the retained source without a second
    // lookup in the ordered region map.
    regions: BTreeMap<u64, Rc<CodeRegion>>,
    loaded: BTreeMap<u64, LoadedBlock>,
    _serialized: PhantomData<Rc<()>>,
}

impl BlockRuntime {
    #[cfg(feature = "std")]
    pub fn new(address_space: u64, cache_root: PathBuf) -> Self {
        Self::new_with_profile(address_space, cache_root, BlockProfile::ORDINARY)
    }

    /// Explicit helper profile is part of every persistent translation identity.
    /// Preparing AOT needs no keys; execution receives a separately bound context.
    #[cfg(feature = "std")]
    pub fn new_with_profile(
        address_space: u64,
        cache_root: PathBuf,
        profile: BlockProfile,
    ) -> Self {
        Self::new_for_target(
            address_space,
            cache_root,
            profile,
            stela_compiler::TargetCpu::detect(),
        )
    }
    #[cfg(feature = "std")]
    pub fn new_for_target(
        address_space: u64,
        cache_root: PathBuf,
        profile: BlockProfile,
        target: stela_compiler::TargetCpu,
    ) -> Self {
        Self::with_preparation(
            address_space,
            profile,
            target,
            ArtifactPreparation::FileCache(cache_root),
        )
    }

    pub fn target_cpu(&self) -> stela_compiler::TargetCpu {
        self.target
    }

    pub fn address_space(&self) -> u64 {
        self.address_space
    }

    pub fn profile(&self) -> BlockProfile {
        self.profile
    }

    /// A permission service must not bypass immutable-source visibility or
    /// invalidation by presenting registered code as an ordinary data mapping.
    pub fn source_overlaps(&self, address: u64, bytes: u64) -> bool {
        let Some(end) = address.checked_add(bytes) else {
            return true;
        };
        bytes != 0
            && self
                .regions
                .range(..end)
                .next_back()
                .is_some_and(|(&base, r)| base < end && r.end > address)
    }

    /// Own an immutable file/code snapshot. Partial trailing bytes are held
    /// but can never be padded into a fetchable instruction. Same-base replacement
    /// always advances generation, even when the source bytes are identical.
    pub fn register_code(&mut self, base: GuestAddress, source: Vec<u8>) -> Result<u64> {
        require!(
            base.0.is_multiple_of(4) && source.len() >= 4,
            "code region must have an aligned base and at least one instruction"
        );
        let end = base
            .0
            .checked_add(source.len() as u64)
            .ok_or(refuse!("code region overflow"))?;
        for (&start, region) in &self.regions {
            require!(
                start == base.0 || end <= start || base.0 >= region.end,
                "code registration overlaps an existing region"
            );
        }
        let generation = self
            .next_generation
            .checked_add(1)
            .ok_or(refuse!("code generation exhausted"))?;
        self.loaded.retain(|_, block| block.region.base != base.0);
        self.regions.insert(
            base.0,
            Rc::new(CodeRegion {
                base: base.0,
                source: source.into_boxed_slice(),
                end,
                generation,
            }),
        );
        self.next_generation = generation;
        Ok(generation)
    }

    pub fn unregister_code(&mut self, base: GuestAddress) -> bool {
        let removed = self.regions.remove(&base.0).is_some();
        if removed {
            self.loaded.retain(|_, block| block.region.base != base.0);
        }
        removed
    }

    fn region(&self, pc: GuestAddress) -> Option<(u64, &Rc<CodeRegion>)> {
        self.regions
            .range(..=pc.0)
            .next_back()
            .filter(|(_, region)| pc.0 < region.end)
            .map(|(&base, region)| (base, region))
    }

    fn load(&mut self, pc: GuestAddress, outcome: &mut RunOutcome) -> Result<Option<CompileError>> {
        let (block, native) = match self.prepare_snapshot_attempt(pc)? {
            Ok(snapshot) => snapshot,
            Err(error) => return Ok(Some(error.into())),
        };
        let prepared = block.prepared;
        let region = self
            .regions
            .get(&block.region_base.0)
            .filter(|region| region.generation == block.generation)
            .ok_or_else(|| refuse!("prepared block source changed", pc = pc.0))?;
        let loaded = LoadedBlock::from_artifact(
            self.target,
            block.artifact_profile,
            Rc::clone(region),
            block.analysis.source_bytes,
            crate::artifact::LoadedArtifact {
                manifest: prepared.manifest,
            },
            &native,
        )?;
        if prepared.cache_hit {
            outcome.disk_cache_hits += 1;
        } else {
            outcome.compiled_blocks += 1;
        }
        self.loaded.insert(pc.0, loaded);

        Ok(None)
    }

    /// Run native blocks until the stop address, an unsatisfiable dispatch, a
    /// synchronous trap or the block budget. A trap leaves PC at the trapping
    /// instruction and reports the preferred return address separately; only
    /// the caller resumes. No host syscall, guest exception entry or EL change
    /// happens here. Unsupported code is returned as a stop, not a guest
    /// exception. A zero budget runs nothing. PC is the next guest instruction.
    ///
    /// # Safety
    /// The registered code, and the code generated from it, must be trusted.
    /// Every guest data access must satisfy the direct-RAM profile: in live
    /// bounds with suitable permissions and alignment, untagged or
    /// authenticated pointers, guest addresses equal to host addresses,
    /// exclusive writes, and no aliasing of Rust references, state, native
    /// code, runtime metadata or the host stack. Source snapshots are
    /// instruction storage, not guest data. The caller excludes concurrent
    /// mutation. Instruction pages are assumed unguarded with GCS disabled;
    /// PAC/BTI/GCS exceptions are not modeled. A native fault can terminate the
    /// host process.
    pub unsafe fn run(
        &mut self,
        state: &mut ThreadState,
        stop_pc: Option<GuestAddress>,
        block_budget: usize,
    ) -> Result<RunOutcome> {
        // SAFETY: the caller upholds `run`'s direct-RAM contract, which is `run_inner`'s.
        unsafe { self.run_inner(state, stop_pc, block_budget, RuntimeServices::direct(None)) }
    }

    /// Run with software PAuth, using the process's keys. Keys are borrowed;
    /// they never enter artifacts or guest RAM.
    ///
    /// # Safety
    /// As for [`Self::run`]. The caller keeps the process keys consistent for
    /// the duration.
    pub unsafe fn run_with_pauth(
        &mut self,
        state: &mut ThreadState,
        stop_pc: Option<GuestAddress>,
        block_budget: usize,
        context: &pauth::Context,
    ) -> Result<RunOutcome> {
        require!(
            self.profile.has(Features::POINTER_AUTHENTICATION),
            "PAuth execution requires the explicit software PAuth block profile"
        );
        require!(
            context.address_space() == self.address_space,
            "PAuth context belongs to another guest address space"
        );
        // SAFETY: the caller upholds this method's direct-RAM contract, which is `run_inner`'s.
        unsafe {
            self.run_inner(
                state,
                stop_pc,
                block_budget,
                RuntimeServices::direct(Some(context)),
            )
        }
    }

    /// Run PAC helpers with the address space's checked data reader.
    /// Authenticated loads keep tags through the reader's translation policy,
    /// and their base and destination are written only after the full
    /// eight-byte read.
    ///
    /// # Safety
    /// Ordinary native code has the requirements of [`Self::run`]. The reader
    /// must own this guest address space, and its mappings, policy and backing
    /// must stay valid for the call. `controls` is the effective SP-alignment
    /// policy of the same CPU.
    pub unsafe fn run_with_pauth_memory(
        &mut self,
        state: &mut ThreadState,
        stop_pc: Option<GuestAddress>,
        block_budget: usize,
        context: &pauth::Context,
        reader: &mut dyn memory::domain::MemoryReader,
        controls: pauth::memory::Controls,
    ) -> Result<RunOutcome> {
        require!(
            self.profile.has(Features::POINTER_AUTHENTICATION),
            "PAuth execution requires the explicit software PAuth block profile"
        );
        require!(
            context.address_space() == self.address_space,
            "PAuth context belongs to another guest address space"
        );
        require!(
            reader.address_space() == Some(self.address_space),
            "memory reader belongs to another guest address space"
        );
        reader
            .validate_privilege(controls.privileged)
            .map_err(|status| {
                refuse!(
                    "LDRA memory privilege profile unavailable",
                    status = status.name()
                )
            })?;
        // SAFETY: the caller upholds this method's direct-RAM contract, which is `run_inner`'s.
        unsafe {
            self.run_inner(
                state,
                stop_pc,
                block_budget,
                RuntimeServices::pauth_memory(context, reader, controls),
            )
        }
    }

    /// Run with the system-instruction service. Registered prefixes have no
    /// direct-RAM effects; memory services use their shared domain. A failure
    /// keeps its PC and completed prefix.
    ///
    /// # Safety
    /// As for [`Self::run`]. The service backend owns its architectural effects
    /// and memory synchronization.
    pub unsafe fn run_with_system(
        &mut self,
        state: &mut ThreadState,
        stop_pc: Option<GuestAddress>,
        block_budget: usize,
        dispatcher: &mut dyn system::execution::Dispatcher,
    ) -> Result<RunOutcome> {
        require!(
            self.profile.has(Features::SYSTEM),
            "System execution requires the explicit System block profile"
        );
        require!(
            dispatcher.address_space() == self.address_space,
            "System dispatcher belongs to another guest address space"
        );
        // SAFETY: the caller upholds this method's direct-RAM contract, which is `run_inner`'s.
        unsafe {
            self.run_inner(
                state,
                stop_pc,
                block_budget,
                RuntimeServices::system(dispatcher),
            )
        }
    }

    unsafe fn run_inner(
        &mut self,
        state: &mut ThreadState,
        stop_pc: Option<GuestAddress>,
        block_budget: usize,
        services: RuntimeServices<'_>,
    ) -> Result<RunOutcome> {
        let RuntimeServices {
            pointer_authentication: mut pauth_execution,
            system: mut system_dispatcher,
        } = services;
        platform::ensure_supported()?;
        require!(
            self.profile.checked_memory().is_none(),
            "checked artifact cannot execute through direct entry"
        );
        let mut outcome = RunOutcome {
            reason: StopReason::BudgetExhausted,
            blocks_executed: 0,
            compiled_blocks: 0,
            disk_cache_hits: 0,
            memory_cache_hits: 0,
            last_exit: None,
            floating_point_helpers: 0,
            pointer_authentication_events: Vec::new(),
            clock_events: Vec::new(),
            thread_register_events: Vec::new(),
            amx_operations: 0,
            amx_native_operations: 0,
        };
        loop {
            let pc = state.pc;
            // A zero budget requests no instruction fetch or execution.
            if self.profile.has(Features::POINTER_AUTHENTICATION) && block_budget == 0 {
                break;
            }
            // Instruction TBI is disabled in the fixed VA47 software profile.
            // This also prevents a resumed poisoned RETAB target from becoming
            // executable through a registered snapshot or matching stop sentinel.
            let extension = pc.0 >> 47;
            if self.profile.has(Features::POINTER_AUTHENTICATION)
                && extension != 0
                && extension != u64::MAX >> 47
            {
                outcome.reason = StopReason::NoncanonicalInstructionTarget { address: pc };
                break;
            }
            if stop_pc == Some(pc) {
                outcome.reason = StopReason::StopAddress;
                break;
            }
            if outcome.blocks_executed == block_budget {
                break;
            }
            if !pc.0.is_multiple_of(4) {
                outcome.reason = StopReason::UnalignedTarget;
                break;
            }
            let cached = self.loaded.get(&pc.0);
            let source = match cached {
                Some(block) => Some((block.region.base, &block.region)),
                None => self.region(pc),
            };
            let Some((base, region)) = source else {
                outcome.reason = StopReason::UnmappedTarget;
                break;
            };
            if region.end - pc.0 < 4 {
                outcome.reason = StopReason::TruncatedInstruction;
                break;
            }
            // Current PSTATE and CPU controls take precedence over decoding an
            // available word. This guard performs no architectural effect.
            let system_entry = if self.profile.has(Features::SYSTEM)
                || self.profile.checked_memory().is_some()
            {
                let Some(dispatcher) = system_dispatcher.as_deref_mut() else {
                    outcome.reason = StopReason::SystemContextRequired { instruction_pc: pc };
                    break;
                };
                let word = validate_terminal_word(pc, pc, INSTRUCTION_BYTES, base, &region.source)?;
                let boundary = system::execution::EntryBoundary::new(
                    self.address_space,
                    region.generation,
                    pc,
                    GuestAddress(base),
                    region.source.len(),
                    word,
                );
                let offset = (pc.0 - base) as usize;
                let available = ((region.source.len() - offset) / INSTRUCTION_BYTES)
                    .min(MAX_BLOCK_INSTRUCTIONS)
                    * INSTRUCTION_BYTES;
                let source = &region.source[offset..offset + available];
                if let Err(failure) = dispatcher.validate_entry(&boundary, source, &state.registers)
                {
                    outcome.reason = StopReason::SystemEntryFailure {
                        instruction_pc: pc,
                        word,
                        failure,
                    };
                    break;
                }
                Some(boundary)
            } else {
                None
            };
            let block = if let Some(block) = cached {
                outcome.memory_cache_hits += 1;
                block
            } else {
                // Strict AOT misses retain their source refusal ordering
                // without constructing a plan. Other misses use the parse
                // that prepares the artifact, including on a disk hit.
                let preflight = if matches!(self.preparation, ArtifactPreparation::AheadOfTime) {
                    let offset = (pc.0 - base) as usize;
                    let words = ((region.source.len() - offset) / 4).min(MAX_BLOCK_INSTRUCTIONS);
                    stela_compiler::block::source_bytes_with_profile(
                        &region.source[offset..offset + words * 4],
                        pc,
                        self.profile,
                    )
                    .err()
                } else {
                    None
                };
                let unsupported = match preflight {
                    Some(error) => Some(error),
                    None => self.load(pc, &mut outcome)?,
                };
                if let Some(error) = unsupported {
                    outcome.reason = StopReason::Unsupported {
                        detail: error.to_string(),
                    };
                    break;
                }
                self.loaded
                    .get(&pc.0)
                    .ok_or_else(|| refuse!("loaded block disappeared", pc = pc.0))?
            };
            let current = &block.region;
            debug_assert!(
                self.regions
                    .get(&current.base)
                    .is_some_and(|registered| Rc::ptr_eq(registered, current))
            );
            let mut completed_blocks = 1;
            let native_exit = if block.profile.has(Features::RESIDENT_LOOPS) {
                let available = block_budget - outcome.blocks_executed;
                let mut budget = stela_compiler::block::loop_budget::LoopBudget::new(available);
                // SAFETY: the distinct immutable profile/ABI binds this source
                // and mapping to the same bank and invocation-local counter.
                let returned = unsafe {
                    block
                        .mapping
                        .call_budgeted_block(&mut state.registers, &mut budget)?
                };
                completed_blocks = budget
                    .consumed(available)
                    .filter(|&consumed| consumed != 0)
                    .ok_or(refuse!("invalid resident-block budget accounting"))?;
                returned
            } else {
                // SAFETY: source replacement and unregistration evict cached
                // mappings. The caller's direct-RAM contract covers the call.
                unsafe { block.mapping.call_block(&mut state.registers)? }
            };
            let BlockExit { next_pc, kind } = native_exit;
            let exit_kind = ExitKind::try_from(kind)
                .map_err(|_| refuse!("invalid native block exit kind", kind))?;
            if matches!(
                exit_kind,
                ExitKind::Fallthrough
                    | ExitKind::Direct
                    | ExitKind::Indirect
                    | ExitKind::Call
                    | ExitKind::Return
            ) {
                state.pc = GuestAddress(next_pc);
                outcome.last_exit = Some((next_pc, kind));
                outcome.blocks_executed += completed_blocks;
                continue;
            }
            require!(
                exit_kind != ExitKind::ScalableNumerical,
                "context-block numerical exit requires owned scalable entry"
            );
            let trap = if exit_kind == ExitKind::Trap {
                Some(validate_trap_exit(
                    pc,
                    GuestAddress(next_pc),
                    block.source_bytes,
                    current.base,
                    &current.source,
                )?)
            } else {
                None
            };
            let pointer_authentication = if exit_kind == ExitKind::PointerAuthentication {
                require!(
                    self.profile.has(Features::POINTER_AUTHENTICATION),
                    "ordinary block produced an unrequested PAuth helper exit"
                );
                Some(validate_pauth_exit(
                    pc,
                    GuestAddress(next_pc),
                    block.source_bytes,
                    current.base,
                    &current.source,
                )?)
            } else {
                None
            };
            let clock_instruction = if exit_kind == ExitKind::Clock {
                require!(
                    self.profile.has(Features::CLOCK),
                    "block produced an unrequested clock helper exit"
                );
                Some(validate_clock_exit(
                    pc,
                    GuestAddress(next_pc),
                    block.source_bytes,
                    current.base,
                    &current.source,
                )?)
            } else {
                None
            };
            let thread_register_instruction = if exit_kind == ExitKind::ThreadRegister {
                require!(
                    self.profile.has(Features::THREAD_REGISTERS),
                    "block produced an unrequested thread-register helper exit"
                );
                Some(validate_thread_register_exit(
                    pc,
                    GuestAddress(next_pc),
                    block.source_bytes,
                    current.base,
                    &current.source,
                )?)
            } else {
                None
            };
            let amx_instruction = if exit_kind == ExitKind::Amx {
                require!(
                    self.profile.has(Features::AMX),
                    "block produced an unrequested AMX exit"
                );
                Some(validate_amx_exit(
                    pc,
                    GuestAddress(next_pc),
                    block.source_bytes,
                    current.base,
                    &current.source,
                )?)
            } else {
                None
            };
            let system_boundary = if exit_kind == ExitKind::System {
                require!(
                    self.profile.has(Features::SYSTEM),
                    "block produced an unrequested System helper exit"
                );
                let instruction = validate_system_exit(
                    pc,
                    GuestAddress(next_pc),
                    block.source_bytes,
                    current.base,
                    &current.source,
                )?;
                Some(system::execution::Boundary::new(
                    system_entry.ok_or(refuse!("System exit lacks validated entry"))?,
                    block.source_bytes,
                    instruction,
                ))
            } else {
                None
            };
            let numerical = if exit_kind == ExitKind::FloatingPoint {
                Some(validate_numerical_exit(
                    pc,
                    GuestAddress(next_pc),
                    block.source_bytes,
                    current.base,
                    &current.source,
                )?)
            } else {
                None
            };
            state.pc = GuestAddress(next_pc);
            outcome.last_exit = Some((next_pc, kind));
            outcome.blocks_executed += completed_blocks;
            if let Some(boundary) = system_boundary {
                let instruction = boundary.instruction();
                let dispatcher = system_dispatcher
                    .as_deref_mut()
                    .ok_or(refuse!("System exit lacks validated dispatcher"))?;
                match dispatcher.complete(boundary, &mut state.registers) {
                    Ok(next) => state.pc = next,
                    Err(failure) => {
                        outcome.reason = StopReason::SystemFailure {
                            instruction,
                            failure,
                        };
                        break;
                    }
                }
            }
            if let Some((operand, continuation)) = numerical {
                if let Err(trapped) = floating::execute(&mut state.registers, &operand) {
                    outcome.reason = StopReason::FloatingPointTrapped {
                        instruction_pc: state.pc,
                        fpcr: state.registers.fpcr,
                        exceptions: trapped.exceptions,
                    };
                    break;
                }
                state.pc = continuation;
                outcome.floating_point_helpers += 1;
            }
            if let Some(instruction) = thread_register_instruction {
                outcome.reason = StopReason::ThreadRegisterRequired { instruction };
                break;
            }
            if let Some(instruction) = amx_instruction {
                outcome.reason = StopReason::AmxRequired { instruction };
                break;
            }
            if let Some(instruction) = clock_instruction {
                outcome.reason = StopReason::ClockRequired { instruction };
                break;
            }
            if let Some(instruction) = pointer_authentication {
                // The implementation's permitted UNDEF constraint precedes key
                // availability checks and cannot invoke data translation.
                if let PointerAuthenticationOperation::Load(load) = instruction.operation
                    && load.undefined_overlap()
                {
                    outcome.reason = StopReason::PointerAuthenticationLoadUndefined { instruction };
                    break;
                }
                let Some(execution) = pauth_execution.as_mut() else {
                    outcome.reason = StopReason::PointerAuthenticationRequired { instruction };
                    break;
                };
                let context = execution.context;
                if let Some(key) = instruction.operation.required_key()
                    && !context.has_key(runtime_pauth_key(key))
                {
                    outcome.reason =
                        StopReason::PointerAuthenticationKeyUnavailable { instruction, key };
                    break;
                }
                if matches!(
                    instruction.operation,
                    PointerAuthenticationOperation::GenericCode(_)
                ) && !context.has_generic_key()
                {
                    outcome.reason =
                        StopReason::PointerAuthenticationGenericKeyUnavailable { instruction };
                    break;
                }
                let result = complete_pauth_with_memory(
                    context,
                    &instruction,
                    &mut state.registers,
                    execution.memory.as_mut(),
                );
                let (next, authenticated) = match result {
                    Ok(value) => value,
                    Err(fault) => {
                        outcome.reason = match fault {
                            PauthFault::Authentication(pauth::AuthenticationError::Failure {
                                code,
                            }) => StopReason::PointerAuthenticationFailure { instruction, code },
                            PauthFault::UndefinedOverlap => {
                                StopReason::PointerAuthenticationLoadUndefined { instruction }
                            }
                            PauthFault::StackAlignment { stack_pointer } => {
                                StopReason::PointerAuthenticationStackAlignment {
                                    instruction,
                                    stack_pointer,
                                }
                            }
                            PauthFault::MemoryRequired => {
                                StopReason::PointerAuthenticationMemoryRequired { instruction }
                            }
                            PauthFault::Read(fault) => {
                                StopReason::PointerAuthenticationMemoryFault { instruction, fault }
                            }
                            fault => StopReason::PointerAuthenticationUnsupported {
                                instruction,
                                detail: fault.to_string(),
                            },
                        };
                        break;
                    }
                };
                state.pc = next;
                let poisoned_transfer =
                    instruction.operation.is_transfer() && authenticated == Some(false);
                outcome
                    .pointer_authentication_events
                    .push(PointerAuthenticationEvent {
                        instruction,
                        next_pc: next,
                        authenticated,
                    });
                if poisoned_transfer {
                    outcome.reason = if instruction.operation.is_return() {
                        StopReason::PointerAuthenticationPoisonedReturn {
                            instruction,
                            poisoned_target: next,
                        }
                    } else {
                        StopReason::PointerAuthenticationPoisonedBranch {
                            instruction,
                            poisoned_target: next,
                        }
                    };
                    break;
                }
            }
            if let Some(trap) = trap {
                // The already executed instruction takes precedence over a
                // coincident stop address or exhausted block budget. Do not
                // advance to its return PC or execute another source block.
                outcome.reason = StopReason::SynchronousTrap { trap };
                break;
            }
        }
        Ok(outcome)
    }

    /// Inspect source/native identity and reverse maps without exposing RX pointers.
    pub fn loaded_manifest(&self, pc: GuestAddress) -> Option<&crate::Manifest> {
        self.loaded.get(&pc.0).map(|block| &block.prepared.manifest)
    }
}

#[cfg(test)]
mod tests;
