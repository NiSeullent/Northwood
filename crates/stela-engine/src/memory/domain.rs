//! The memory service behind context-backed native code.
//!
//! Instruction families submit each reached element; the domain resolves its
//! atomic parts, checks tags and publishes writes through the shared reservation
//! domain.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
mod scalar;
pub use scalar::{MemoryFault, MemoryReader};

mod publication;
pub use publication::PublicationFault;
pub(in crate::memory) mod source_mutation;
pub(crate) use source_mutation::SourceWriteReceipt;
mod profile;
mod tag_transfer;
pub use profile::{CheckedMemoryEntry, MemoryProfile};
mod lifecycle;
mod session;
pub(crate) use session::MemoryExecutionSession;
mod service;
pub(in crate::memory) use lifecycle::MemoryServiceLoan;
pub(in crate::memory) use lifecycle::ServiceLoanAttempt;
use lifecycle::{DomainSnapshot, DomainState};
pub use lifecycle::{ExecutionLoanAttempt, MemoryExecutionLoan, MemoryObservationLoan};
pub(in crate::memory) use service::AllWriterService;

mod exclusive;
pub(in crate::memory) mod native_target;

mod read;
mod shared;

use crate::platform::sync::Mutex;
use crate::system::events::{EventEndpoint, EventProducer};
use alloc::collections::BTreeMap;
use alloc::sync::{Arc, Weak};
use core::sync::atomic::{AtomicBool, Ordering};

use stela_compiler::{
    TargetCpu,
    context::memory::{
        MEMORY_CONTEXT_BYTES, MEMORY_CONTEXT_VERSION, MemoryContextView,
        MemoryImplementationProfile, MemoryOperationClass, MemoryRequest, MemoryRequirements,
        MemoryStatus, SpAlignmentRequirement, flags, profile_requirements,
    },
};

use super::{
    access,
    backing::{AccessKind, MappingId, MemoryBacking, MemoryView, ResolvedAccess},
    monitor,
};

pub(crate) const SCA_BYTES: usize = 16;
const HALF_SCA_BYTES: usize = SCA_BYTES / 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappingAttributes {
    pub tagged: bool,
    pub shareable: bool,
}

/// Architectural service category, kept even when typed staging is adapted
/// to the ordinary all-writer transfer path. This is not a native wire value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataOperation {
    Ordinary,
    Exclusive,
    Atomic,
    Ordered,
    AllocationTag,
}

/// Original reached request. Dynamic source policies need its full quantity
/// and source PC before checking a smaller atomic part's translation/rights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataRequest {
    pub address: u64,
    pub bytes: usize,
    pub flags: u32,
    pub pc: Option<u64>,
    pub operation: DataOperation,
}

impl DataRequest {
    fn native(request: &MemoryRequest, operation: DataOperation) -> Self {
        Self {
            address: request.address,
            bytes: request.bytes as usize,
            flags: request.flags,
            pc: Some(request.pc),
            operation,
        }
    }
}

/// One reached physical atomic quantity; no later part is authorized by it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataPart {
    pub address: u64,
    pub bytes: usize,
    pub kind: AccessKind,
}

/// Effective CPU controls belong to the per-thread policy, not each ISA family.
/// Policy callbacks must not reenter or retire this domain while an access is active.
pub trait DataPolicy {
    /// Request the machine's Device-RAM serializer and native full
    /// fences around each callback. This strengthens ordering only; it grants
    /// no translation, physical memory, instruction or MMIO admission.
    fn device_ram_ordering(&self) -> bool {
        false
    }
    fn validate_instructions(
        &self,
        required: stela_compiler::context::memory::InstructionRequirements,
    ) -> Result<(), MemoryStatus> {
        if required.is_empty() {
            Ok(())
        } else {
            Err(MemoryStatus::Unsupported)
        }
    }
    /// Typed service classes must be admitted by the CPU owner.
    fn validate_operation_execution(
        &self,
        _class: MemoryOperationClass,
    ) -> Result<(), MemoryStatus> {
        Err(MemoryStatus::Unsupported)
    }
    /// Reject unavailable effective-control profiles before any guest effect.
    fn validate_execution(&self, required_flags: u32) -> Result<(), MemoryStatus>;
    /// Check the body's SP specialization against the effective controls.
    fn validate_sp_alignment(&self, required: SpAlignmentRequirement) -> Result<(), MemoryStatus> {
        if required == SpAlignmentRequirement::Any {
            Ok(())
        } else {
            Err(MemoryStatus::Unsupported)
        }
    }
    /// Check the emitted FEAT_CPA address specialization against the CPU owner.
    fn validate_address_checks(
        &self,
        required: stela_compiler::context::memory::address::AddressCheckRequirements,
    ) -> Result<(), MemoryStatus> {
        if required.is_empty() {
            Ok(())
        } else {
            Err(MemoryStatus::Unsupported)
        }
    }
    /// FEAT_CPA arithmetic must be admitted by the CPU even when its result is
    /// discarded.
    fn validate_pointer_arithmetic(
        &self,
        required: Option<stela_compiler::context::pointer::PointerArithmeticRequirement>,
    ) -> Result<(), MemoryStatus> {
        if required.is_none() {
            Ok(())
        } else {
            Err(MemoryStatus::Unsupported)
        }
    }
    /// Check the source-bound implementation identity selected by the body.
    /// An unproven CPU implementation must decline rather than assume a choice.
    fn validate_implementation_profile(
        &self,
        _required: MemoryImplementationProfile,
    ) -> Result<(), MemoryStatus> {
        Err(MemoryStatus::Unsupported)
    }
    /// Address normalization for access decomposition only. Dynamic policies
    /// must not perform table walks, permission checks or descriptor updates here.
    fn translate(&self, raw_address: u64) -> Result<u64, MemoryStatus>;
    /// Check original-access constraints before any reached translation/effect.
    /// The legacy fixed policy has no additional request-dependent controls.
    fn validate_data_request(&mut self, _request: DataRequest) -> Result<(), MemoryStatus> {
        Ok(())
    }
    /// Translate precisely this reached part with its read/write intent.
    /// A dynamic policy may walk once when the entire part is inside the same
    /// source leaf; it must establish every byte's rights and physical identity.
    /// The default preserves legacy per-byte translation and exact fault address.
    fn translate_part(
        &mut self,
        _request: DataRequest,
        part: DataPart,
    ) -> Result<u64, (MemoryStatus, u64)> {
        let address = self
            .translate(part.address)
            .map_err(|status| (status, part.address))?;
        for offset in 1..part.bytes {
            let raw = part.address.wrapping_add(offset as u64);
            let translated = self.translate(raw).map_err(|status| (status, raw))?;
            if address.checked_add(offset as u64) != Some(translated) {
                return Err((MemoryStatus::Translation, raw));
            }
        }
        Ok(address)
    }
    fn check(&mut self, access: DataAccess<'_>) -> Result<(), MemoryStatus>;
}

/// Effective dedicated allocation-tag access policy for the current privilege.
/// The result includes effective allocation-tag enable and mapping attributes.
/// False still requires the dedicated instruction's translation/rights checks.
pub trait AllocationTagPolicy: DataPolicy {
    fn validate_privilege(&self, privileged: bool) -> Result<(), MemoryStatus>;
    fn allocation_tag_access(
        &self,
        attributes: MappingAttributes,
        privileged: bool,
    ) -> Result<bool, MemoryStatus>;
}

pub struct DataAccess<'a> {
    pub raw_address: u64,
    pub translated_address: u64,
    pub backing: &'a MemoryBacking,
    pub backing_offset: usize,
    pub bytes: usize,
    pub attributes: MappingAttributes,
    pub flags: u32,
}

type ReservationDomain = monitor::MonitorDomain<MemoryBacking, MemoryView, MappingAttributes>;
type ReservationThread = monitor::ThreadMonitor<MemoryBacking, MemoryView, MappingAttributes>;
type ReservationMapping = monitor::Mapping<MemoryView, MappingAttributes>;
type MappingRegistry = BTreeMap<MappingId, Weak<ReservationMapping>>;

#[derive(Clone)]
pub(super) struct RegisteredView {
    pub(super) view: Arc<MemoryView>,
    pub(super) monitor: Arc<ReservationMapping>,
}

/// One guest machine's physical coherence and reservation owner. Shared backing
/// across address spaces must bind to this same owner; COW creates new backing.
pub struct MemorySystem {
    identity: Arc<()>,
    reservations: Arc<ReservationDomain>,
    mappings: Mutex<MappingRegistry>,
    failed: AtomicBool,
    // Serializes Device RAM transactions on this machine.
    device_ram_order: Mutex<()>,
}

impl MemorySystem {
    pub fn new(erg_bytes: u64, stripe_count: usize) -> Result<Arc<Self>, MemoryStatus> {
        let reservations = monitor::MonitorDomain::new(erg_bytes, stripe_count)
            .map_err(|_| MemoryStatus::InvalidRequest)?;
        Ok(Arc::new(Self {
            identity: Arc::new(()),
            reservations,
            mappings: Mutex::new(BTreeMap::new()),
            failed: AtomicBool::new(false),
            device_ram_order: Mutex::new(()),
        }))
    }

    fn register_view(
        &self,
        view: &Arc<MemoryView>,
        attributes: MappingAttributes,
    ) -> Result<Arc<ReservationMapping>, MemoryStatus> {
        view.backing()
            .bind_coherence(&self.identity)
            .map_err(|_| MemoryStatus::Domain)?;
        let mut mappings = self.mappings.lock().map_err(|_| MemoryStatus::Domain)?;
        // Keep the registry bounded by live generations rather than mapping
        // history.
        mappings.retain(|_, mapping| mapping.strong_count() != 0);
        if let Some(mapping) = mappings.get(&view.id()).and_then(Weak::upgrade) {
            if *mapping.attributes() != attributes {
                return Err(MemoryStatus::InvalidRequest);
            }
            return Ok(mapping);
        }
        let mapping = monitor::Mapping::new(Arc::clone(view), attributes);
        mappings.insert(view.id(), Arc::downgrade(&mapping));
        Ok(mapping)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DomainRole {
    NormalWb,
    SourcePhysicalRam,
    /// Only CPL0 registration constructs this role. Native entry must
    /// keep a per-worker target proof through checked finalization.
    NativeSystemRam,
}

/// Canonical current mappings and invocation/service exclusion. The immutable
/// role survives every snapshot publication; physical RAM never acquires the
/// generic Normal-WB callback contract from its host layout.
pub struct MemoryDomain {
    role: DomainRole,
    state: Mutex<DomainState>,
    /// Snapshot generation; written only under `state` by an exclusive service.
    generation: core::sync::atomic::AtomicU64,
    target: TargetCpu,
    system: Arc<MemorySystem>,
    address_space: Option<u64>,
}

impl MemoryDomain {
    pub fn new(
        system: Arc<MemorySystem>,
        views: Vec<(Arc<MemoryView>, MappingAttributes)>,
        target: TargetCpu,
    ) -> Result<Arc<Self>, MemoryStatus> {
        Self::construct(None, system, views, target, DomainRole::NormalWb)
    }

    /// Bind this snapshot to its real process address-space identity.
    pub fn new_for_address_space(
        address_space: u64,
        system: Arc<MemorySystem>,
        views: Vec<(Arc<MemoryView>, MappingAttributes)>,
        target: TargetCpu,
    ) -> Result<Arc<Self>, MemoryStatus> {
        Self::construct(
            Some(address_space),
            system,
            views,
            target,
            DomainRole::NormalWb,
        )
    }

    pub(in crate::memory) fn is_source_physical_ram(&self) -> bool {
        self.role == DomainRole::SourcePhysicalRam
    }

    fn check_data_execution(&self) -> Result<(), MemoryStatus> {
        self.check_execution()?;
        if self.is_source_physical_ram() {
            return Err(MemoryStatus::Unsupported);
        }
        Ok(())
    }

    fn construct(
        address_space: Option<u64>,
        system: Arc<MemorySystem>,
        views: Vec<(Arc<MemoryView>, MappingAttributes)>,
        target: TargetCpu,
        role: DomainRole,
    ) -> Result<Arc<Self>, MemoryStatus> {
        if !crate::platform::cpu::admitted(target) {
            return Err(MemoryStatus::Unsupported);
        }
        Self::construct_admitted(address_space, system, views, target, role)
    }

    /// Reserved system RAM has a current-CPL0 target permission and
    /// mapping owner, rather than the legacy identity-frame allocator.
    pub(in crate::memory) fn new_native_system_ram(
        system: Arc<MemorySystem>,
        views: Vec<(Arc<MemoryView>, MappingAttributes)>,
        authority: &crate::platform::cpu::system::NativeSystemTarget,
        target: TargetCpu,
    ) -> Result<Arc<Self>, MemoryStatus> {
        if target.data_independent_timing() || !authority.target().supports(target) {
            return Err(MemoryStatus::Unsupported);
        }
        authority
            .revalidate()
            .map_err(|_| MemoryStatus::Unsupported)?;
        Self::construct_admitted(None, system, views, target, DomainRole::NativeSystemRam)
    }

    fn construct_admitted(
        address_space: Option<u64>,
        system: Arc<MemorySystem>,
        mut views: Vec<(Arc<MemoryView>, MappingAttributes)>,
        target: TargetCpu,
        role: DomainRole,
    ) -> Result<Arc<Self>, MemoryStatus> {
        views.sort_unstable_by_key(|(view, _)| view.guest_base());
        for pair in views.windows(2) {
            let end = pair[0].0.guest_base() + pair[0].0.bytes() as u64;
            if end > pair[1].0.guest_base() {
                return Err(MemoryStatus::InvalidRequest);
            }
        }
        let profile = MemoryProfile::inspect(&views);
        let views = views
            .into_iter()
            .map(|(view, attributes)| {
                Ok(RegisteredView {
                    monitor: system.register_view(&view, attributes)?,
                    view,
                })
            })
            .collect::<Result<_, MemoryStatus>>()?;
        Ok(Arc::new(Self {
            role,
            state: Mutex::new(DomainState {
                snapshot: DomainSnapshot {
                    views,
                    profile,
                    generation: 0,
                },
                native_loans: 0,
                servicing: false,
            }),
            generation: core::sync::atomic::AtomicU64::new(0),
            target,
            system,
            address_space,
        }))
    }

    #[cfg(test)]
    pub(super) fn registered_views(&self) -> Vec<RegisteredView> {
        self.state.lock().unwrap().snapshot.views.clone()
    }

    pub fn address_space(&self) -> Option<u64> {
        self.address_space
    }

    pub fn profile(&self) -> Result<MemoryProfile, MemoryStatus> {
        self.check_data_execution()?;
        Ok(self
            .state
            .lock()
            .map_err(|_| MemoryStatus::Domain)?
            .snapshot
            .profile
            .clone())
    }

    pub fn thread<P: DataPolicy>(self: &Arc<Self>, policy: P) -> MemoryThread<P> {
        MemoryThread {
            domain: Arc::clone(self),
            reservation: self.system.reservations.thread(),
            policy,
        }
    }

    pub fn check_execution(&self) -> Result<(), MemoryStatus> {
        if self.role != DomainRole::NativeSystemRam && !crate::platform::cpu::admitted(self.target)
        {
            return Err(MemoryStatus::Unsupported);
        }
        if self.system.failed.load(Ordering::Acquire) {
            return Err(MemoryStatus::Domain);
        }
        self.system
            .reservations
            .check_execution()
            .map_err(|_| MemoryStatus::Domain)
    }

    /// Retire all matching generations before releasing/changing native rights.
    /// The lifecycle owner serializes this shared mapping generation across all
    /// snapshots/address spaces keeping its Arc and excludes in-flight native
    /// accesses before native rights changes or unmapping.
    pub fn retire(self: &Arc<Self>, view: &Arc<MemoryView>) -> Result<(), MemoryStatus> {
        let _loan = self.service_loan()?;
        // Refuse on generation exhaustion before invalidating any mapping or
        // reservation; the exclusive service borrow keeps this value stable.
        let next_generation = self
            .generation()?
            .checked_add(1)
            .ok_or(MemoryStatus::Domain)?;
        self.retire_view(view)?;
        let mut state = self.state.lock().map_err(|_| MemoryStatus::Domain)?;
        self.check_execution()?;
        state.snapshot.generation = next_generation;
        self.generation.store(next_generation, Ordering::Release);
        Ok(())
    }

    fn retire_view(&self, view: &Arc<MemoryView>) -> Result<(), MemoryStatus> {
        let state = self.state.lock().map_err(|_| MemoryStatus::Domain)?;
        let registered = state
            .snapshot
            .views
            .iter()
            .find(|registered| Arc::ptr_eq(&registered.view, view))
            .ok_or(MemoryStatus::InvalidRequest)?;
        // Lifecycle transitions for this shared generation are serialized by its owner.
        // Repeating a retired generation must not clear fresh alias reservations.
        if !registered.view.is_current() {
            return Err(MemoryStatus::InvalidRequest);
        }
        let range = monitor::WriteRange::new(
            Arc::clone(registered.view.backing()),
            registered.view.backing_offset() as u64,
            registered.view.bytes() as u64,
        )
        .map_err(|_| MemoryStatus::Domain)?;
        self.system
            .reservations
            .revoke_mapping(&registered.monitor)
            .map_err(|_| MemoryStatus::Domain)?;
        registered.view.retire();
        // Finite lifecycle invalidation, not a committed byte store. It also
        // reaches reservations whose first mapping is another joined view.
        self.system
            .reservations
            .with_writer(&range, monitor::Invalidation::BeforeAttempt, || {
                Ok::<_, MemoryStatus>(())
            })
            .map_err(|_| MemoryStatus::Domain)?;
        self.check_execution()
    }

    fn resolve(
        &self,
        address: u64,
        bytes: usize,
        kind: AccessKind,
    ) -> Result<PhysicalAccess, MemoryStatus> {
        self.check_data_execution()?;
        self.resolve_spans(address, bytes, kind)
    }

    fn resolve_spans(
        &self,
        address: u64,
        bytes: usize,
        kind: AccessKind,
    ) -> Result<PhysicalAccess, MemoryStatus> {
        let state = self.state.lock().map_err(|_| MemoryStatus::Domain)?;
        let views = &state.snapshot.views;
        let mut first_span: Option<ResolvedAccess> = None;
        let mut remaining_spans = Vec::new();
        let mut current = address;
        let mut remaining = bytes;
        let mut attributes = None;
        while remaining != 0 {
            let index = views.partition_point(|view| view.view.guest_base() <= current);
            let registered = index
                .checked_sub(1)
                .and_then(|index| views.get(index))
                .ok_or(MemoryStatus::Translation)?;
            let offset = current
                .checked_sub(registered.view.guest_base())
                .and_then(|offset| usize::try_from(offset).ok())
                .filter(|offset| *offset < registered.view.bytes())
                .ok_or(MemoryStatus::Translation)?;
            let reached = remaining.min(registered.view.bytes() - offset);
            let span =
                registered
                    .view
                    .resolve(current, reached, kind)
                    .map_err(|error| match error {
                        super::backing::AccessError::PermissionDenied(_) => {
                            MemoryStatus::Permission
                        }
                        _ => MemoryStatus::Translation,
                    })?;
            let next_attributes = *registered.monitor.attributes();
            if attributes.is_some_and(|attributes| attributes != next_attributes) {
                return Err(MemoryStatus::Unsupported);
            }
            attributes = Some(next_attributes);
            if let Some(first) = &first_span {
                let progressed = bytes - remaining;
                if !Arc::ptr_eq(first.view().backing(), span.view().backing())
                    || first.backing_offset() + progressed != span.backing_offset()
                {
                    return Err(MemoryStatus::Unsupported);
                }
            }
            if first_span.is_none() {
                first_span = Some(span);
            } else {
                remaining_spans.push(span);
            }
            remaining -= reached;
            current = current.wrapping_add(reached as u64);
        }
        Ok(PhysicalAccess {
            first: first_span.ok_or(MemoryStatus::InvalidRequest)?,
            _remaining: remaining_spans,
            attributes: attributes.ok_or(MemoryStatus::InvalidRequest)?,
        })
    }
}

struct PhysicalAccess {
    // Keep every generation intersecting one atomic quantity.
    first: ResolvedAccess,
    _remaining: Vec<ResolvedAccess>,
    attributes: MappingAttributes,
}

impl PhysicalAccess {
    fn first(&self) -> &ResolvedAccess {
        &self.first
    }

    /// Invoked under the real monitor gate before reserved native access.
    fn check_current(&self) -> Result<(), MemoryStatus> {
        if self.first.view().is_current()
            && self._remaining.iter().all(|span| span.view().is_current())
        {
            Ok(())
        } else {
            Err(MemoryStatus::Domain)
        }
    }

    fn data(&self, raw_address: u64, bytes: usize, flags: u32) -> DataAccess<'_> {
        let first = self.first();
        DataAccess {
            raw_address,
            translated_address: first.guest_address(),
            backing: first.view().backing(),
            backing_offset: first.backing_offset(),
            bytes,
            attributes: self.attributes,
            flags,
        }
    }
}

/// One executing guest thread; the reservation survives ordinary block exits.
pub struct MemoryThread<P> {
    domain: Arc<MemoryDomain>,
    reservation: ReservationThread,
    policy: P,
}

/// The real local monitor plus its mapping/service exclusion loan, kept
/// through the final infallible CPU/controller settlement.
pub(crate) struct PreparedLocalClear<'a> {
    clear: monitor::PreparedClear<'a, MemoryBacking, MemoryView, MappingAttributes>,
    _loan: lifecycle::BorrowedExecutionLoan<'a>,
}
impl PreparedLocalClear<'_> {
    pub(crate) fn commit(self) {
        self.clear.commit();
    }
}

impl<P: DataPolicy> MemoryThread<P> {
    /// Match the exact immutable snapshot kept by a source attribute witness.
    /// Identity alone does not establish effective ARM memory attributes.
    pub(crate) fn has_domain(&self, domain: &Arc<MemoryDomain>) -> bool {
        Arc::ptr_eq(&self.domain, domain)
    }

    pub fn address_space(&self) -> Option<u64> {
        self.domain.address_space()
    }

    pub fn profile(&self) -> Result<MemoryProfile, MemoryStatus> {
        self.domain.profile()
    }

    pub fn execution_loan(&self) -> Result<MemoryExecutionLoan, MemoryStatus> {
        self.domain.execution_loan()
    }

    /// Like `execution_loan`, but report a busy host service instead of failing.
    pub fn try_execution_loan(&self) -> Result<ExecutionLoanAttempt, MemoryStatus> {
        self.domain.try_execution_loan()
    }

    /// Observe the domain under an exclusive read-only service borrow, which
    /// provides physical serialization, the current views and private staging.
    pub fn observation_loan(&self) -> Result<MemoryObservationLoan, MemoryStatus> {
        self.domain.observation_loan()
    }

    /// Admit the complete body's platform requirements before any guest effect.
    pub fn prepare(
        &mut self,
        requirements: MemoryRequirements,
    ) -> Result<CheckedMemoryEntry<'_, P>, MemoryStatus> {
        if self.domain.role == DomainRole::NativeSystemRam {
            return Err(MemoryStatus::Unsupported);
        }
        self.prepare_body(requirements)
    }

    /// The engine must borrow current-owner admission, rather than
    /// laundering a system domain through the legacy identity-frame mount.
    pub(crate) fn prepare_native_system<'a>(
        &'a mut self,
        requirements: MemoryRequirements,
        authority: &'a crate::platform::cpu::system::NativeSystemTarget,
    ) -> Result<CheckedMemoryEntry<'a, P>, MemoryStatus> {
        if self.domain.role != DomainRole::NativeSystemRam
            || !authority.target().supports(self.domain.target)
        {
            return Err(MemoryStatus::Unsupported);
        }
        authority
            .revalidate()
            .map_err(|_| MemoryStatus::Unsupported)?;
        let mut entry = self.prepare_body(requirements)?;
        entry.native_system = Some(authority);
        Ok(entry)
    }

    fn prepare_body(
        &mut self,
        requirements: MemoryRequirements,
    ) -> Result<CheckedMemoryEntry<'_, P>, MemoryStatus> {
        self.prepare_body_with_profile(requirements, None)
    }

    fn prepare_body_with_profile(
        &mut self,
        requirements: MemoryRequirements,
        pinned: Option<&MemoryProfile>,
    ) -> Result<CheckedMemoryEntry<'_, P>, MemoryStatus> {
        self.prepare_with(|owner| {
            owner.domain.check_execution()?;
            if requirements.required_profile & profile_requirements::SYSTEM_ATOMIC_READ_WRITEBACK
                != 0
                && owner.domain.role != DomainRole::NativeSystemRam
            {
                return Err(MemoryStatus::Unsupported);
            }
            if let Some(profile) = pinned {
                profile.validate(requirements, owner.domain.target)?;
            } else {
                owner
                    .domain
                    .profile()?
                    .validate(requirements, owner.domain.target)?;
            }
            owner.validate_execution(requirements.required_flags)?;
            owner
                .policy
                .validate_instructions(requirements.required_instructions)?;
            owner
                .policy
                .validate_sp_alignment(requirements.required_sp_alignment)?;
            owner
                .policy
                .validate_address_checks(requirements.required_address_checks)?;
            owner
                .policy
                .validate_pointer_arithmetic(requirements.required_pointer_arithmetic)?;
            if requirements.required_profile & profile_requirements::SCALABLE_TAG_CHECKS != 0 {
                owner.policy.validate_implementation_profile(
                    MemoryImplementationProfile::ScalableTagChecks,
                )?;
            }
            Ok(())
        })
    }
    /// Native entry must validate effective controls before executing an instruction.
    pub fn validate_execution(&self, required_flags: u32) -> Result<(), MemoryStatus> {
        self.domain.check_data_execution()?;
        self.policy.validate_execution(required_flags)?;
        for (flag, class) in [
            (flags::ATOMIC, MemoryOperationClass::Atomic),
            (flags::ORDERED, MemoryOperationClass::Ordered),
            (flags::EXCLUSIVE, MemoryOperationClass::Exclusive),
        ] {
            if required_flags & flag != 0 {
                self.policy.validate_operation_execution(class)?;
            }
        }
        Ok(())
    }

    pub fn policy_mut(&mut self) -> &mut P {
        &mut self.policy
    }

    /// Bind this reservation context to one PE before any mark. The binding
    /// lasts until the context is torn down.
    pub fn bind_events(&mut self, producer: &EventProducer) -> Result<(), MemoryStatus> {
        self.domain.check_execution()?;
        self.domain
            .system
            .reservations
            .bind_events(&mut self.reservation, producer)
            .map_err(|_| MemoryStatus::Domain)
    }

    /// Check the monitor binding against a borrowed live PE owner.
    /// Perform this preflight before architectural effects and keep both
    /// borrows for the complete entry. This check never changes a reservation.
    pub fn check_event_endpoint(&self, endpoint: &EventEndpoint) -> Result<(), MemoryStatus> {
        self.domain.check_execution()?;
        self.domain
            .system
            .reservations
            .check_event_endpoint(&self.reservation, &endpoint.producer())
            .map_err(|_| MemoryStatus::Domain)
    }

    /// Where the word nonzero while the local reservation is held lives
    /// (`ThreadMonitor::held_address`).
    pub(crate) fn reservation_word(&self) -> u64 {
        self.reservation.held_address()
    }

    /// Whether a local reservation is held.
    pub(crate) fn holds_reservation(&self) -> bool {
        self.reservation.holds_reservation()
    }

    pub(crate) fn prepare_clear_reservation(
        &mut self,
    ) -> Result<PreparedLocalClear<'_>, MemoryStatus> {
        let loan = self.domain.borrow_execution()?;
        let clear = self
            .domain
            .system
            .reservations
            .prepare_clear(&mut self.reservation)
            .map_err(|_| MemoryStatus::Domain)?;
        Ok(PreparedLocalClear { clear, _loan: loan })
    }

    /// Clear only the local reservation. The independent global mark survives.
    /// ERET separately signals its prechecked borrowed endpoint in source order.
    pub fn clear_reservation(&mut self) -> Result<(), MemoryStatus> {
        self.domain.check_execution()?;
        self.domain
            .system
            .reservations
            .clear(&mut self.reservation)
            .map_err(|_| MemoryStatus::Domain)?;
        // Cleanup can wait behind another PE; a failure while waiting is a
        // backend failure, never a successful ERET.
        self.domain.check_execution()
    }

    /// Apply the implementation's policy after an architectural memory fault.
    /// Platform and ABI declines keep the local reservation. The fault's
    /// location and completed prefix are unchanged.
    pub fn settle_fault(&mut self, status: MemoryStatus) -> Result<(), MemoryStatus> {
        if is_guest_fault(status as u32) {
            self.domain
                .system
                .reservations
                .clear(&mut self.reservation)
                .map_err(|_| MemoryStatus::Domain)?;
        }
        Ok(())
    }

    /// Borrow a service view for one synchronous context-backed call. This
    /// older API returns `R` even when finalization fails; checked execution
    /// uses `with_checked_view`.
    ///
    /// # Safety
    /// Native code may call only the supplied callback, with valid requests and
    /// disjoint private staging of the declared size. It must keep the header,
    /// opaque pointer and callback intact, keep no pointers, not re-enter this
    /// owner, and respect the mapping and staging lifetimes. Guest and native
    /// addresses must agree in their low four bits for 16-byte single-copy
    /// atomicity; unsupported footprints fail.
    pub unsafe fn with_view<R>(&mut self, callback: impl FnOnce(&mut MemoryContextView) -> R) -> R {
        // Legacy low-level contract: callers cannot infer successful finalization
        // from R. Owned checked execution must use with_checked_view instead.
        self.domain
            .check_data_execution()
            .expect("supported legacy memory role");
        let _loan = self
            .execution_loan()
            .expect("serialized legacy memory invocation");
        // SAFETY: The execution loan stays live through callback finalization. The
        // caller keeps staging/policy borrows valid and disjoint from guest RAM.
        let (result, _finalization) = unsafe { self.invoke_memory_view(callback) };
        result
    }

    /// Run and finalize one checked memory body. A finalization failure
    /// discards `R` even when native code returned; callers needing the fault
    /// details must copy them during the callback.
    ///
    /// # Safety
    /// As for `with_view`. The caller keeps the policy and storage borrows
    /// throughout.
    pub unsafe fn with_checked_view<R>(
        &mut self,
        callback: impl FnOnce(&mut MemoryContextView) -> R,
    ) -> Result<R, MemoryStatus> {
        self.domain.check_data_execution()?;
        let _loan = self.execution_loan()?;
        // SAFETY: The execution loan stays live through callback finalization; the
        // caller keeps the view callback and its disjoint staging/policy borrows.
        let (result, finalization) = unsafe { self.invoke_memory_view(callback) };
        finalization.map(|()| result)
    }

    /// One common finalization for callback faults and locally emitted alignment faults.
    unsafe fn invoke_memory_view<R>(
        &mut self,
        callback: impl FnOnce(&mut MemoryContextView) -> R,
    ) -> (R, Result<(), MemoryStatus>) {
        let mut view = MemoryContextView {
            version: MEMORY_CONTEXT_VERSION,
            bytes: MEMORY_CONTEXT_BYTES,
            callback: dispatch::<P>,
            opaque: core::ptr::from_mut(self).cast(),
            fault_kind: MemoryStatus::Success as u32,
            fault_access: 0,
            fault_address: 0,
            fault_pc: 0,
            completed_bytes: 0,
            reserved_tail: 0,
        };
        let result = callback(&mut view);
        let finalization = view
            .has_supported_header()
            .then_some(())
            .ok_or(MemoryStatus::Domain)
            .and_then(|()| MemoryStatus::from_code(view.fault_kind).ok_or(MemoryStatus::Domain))
            .and_then(|status| self.settle_fault(status))
            .and_then(|()| self.domain.check_execution());
        if finalization == Err(MemoryStatus::Domain) {
            self.domain.system.failed.store(true, Ordering::Release);
        }
        (result, finalization)
    }

    unsafe fn transfer(
        &mut self,
        view: &mut MemoryContextView,
        request: &MemoryRequest,
    ) -> Result<(), (MemoryStatus, u64)> {
        if self.policy.device_ram_ordering() {
            if !cfg!(target_arch = "x86_64") {
                return Err((MemoryStatus::Unsupported, request.address));
            }
            self.domain
                .check_execution()
                .map_err(|status| (status, request.address))?;
            // Same physical machine owner as every existing staged Device-RAM
            // transaction. Keep it across transfer/fault prefix; never
            // copy an ordering receipt or replace the checked policy/ABI.
            let system = Arc::clone(&self.domain.system);
            let _order = system
                .device_ram_order
                .try_lock()
                .map_err(|_| (MemoryStatus::Domain, request.address))?;
            access::device_ram_fence();
            // SAFETY: unchanged exact private staging and kept invocation.
            let result = unsafe { self.transfer_unordered(view, request) };
            access::device_ram_fence();
            return result;
        }
        // SAFETY: unchanged invocation; normal policies do not take Device locks.
        unsafe { self.transfer_unordered(view, request) }
    }

    unsafe fn transfer_unordered(
        &mut self,
        view: &mut MemoryContextView,
        request: &MemoryRequest,
    ) -> Result<(), (MemoryStatus, u64)> {
        if request.flags & (flags::ATOMIC | flags::ORDERED) != 0 {
            // SAFETY: dedicated typed payload staging is inherited from dispatch.
            return unsafe { self.transfer_shared(view, request) };
        }
        if request.flags & flags::EXCLUSIVE != 0 {
            // SAFETY: category-specific staging contract is inherited from dispatch.
            return unsafe { self.transfer_exclusive(view, request) };
        }
        // Preserve the source category through ordered payload adaptation.
        // SAFETY: Earlier branches excluded shared/exclusive payloads. The caller's
        // ordinary request and private staging stay valid for this invocation.
        unsafe { self.transfer_ordinary(view, request, DataOperation::Ordinary) }
    }

    /// # Safety
    /// Same private ordinary staging and kept invocation as `transfer`.
    pub(super) unsafe fn transfer_ordinary(
        &mut self,
        view: &mut MemoryContextView,
        request: &MemoryRequest,
        operation: DataOperation,
    ) -> Result<(), (MemoryStatus, u64)> {
        self.validate_execution(request.flags)
            .map_err(|status| (status, request.address))?;
        if request.data.is_null() || !request.has_supported_ordinary_shape() {
            return Err((MemoryStatus::InvalidRequest, request.address));
        }
        let intent = DataRequest::native(request, operation);
        self.policy
            .validate_data_request(intent)
            .map_err(|status| (status, request.address))?;
        let translated = self
            .policy
            .translate(request.address)
            .map_err(|status| (status, request.address))?;
        for part in AccessParts::new(translated, request.bytes as usize, request.flags) {
            let raw = request.address.wrapping_add(part.offset as u64);
            let write = request.flags & flags::WRITE != 0;
            let kind = if write {
                AccessKind::Write
            } else {
                AccessKind::Read
            };
            let address = self.policy.translate_part(
                intent,
                DataPart {
                    address: raw,
                    bytes: part.bytes,
                    kind,
                },
            )?;
            let physical = self
                .domain
                .resolve(address, part.bytes, kind)
                .map_err(|status| (status, raw))?;
            let first = physical.first();
            let pointer = first.host_ptr().as_ptr();
            let completed = view
                .completed_bytes
                .checked_add(part.bytes as u64)
                .ok_or((MemoryStatus::InvalidRequest, raw))?;
            if write {
                if part.bytes == SCA_BYTES && !self.domain.target.supports(TargetCpu::Cx16) {
                    return Err((MemoryStatus::Unsupported, raw));
                }
                first
                    .view()
                    .backing()
                    .host_pointer(first.backing_offset(), part.bytes, AccessKind::Read)
                    .map_err(|_| (MemoryStatus::Unsupported, raw))?;
                // Native staging is private and belongs exclusively to this call.
                let mut bytes = [0_u8; SCA_BYTES];
                // SAFETY: the invocation contract grants exact private staging.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        request.data.add(part.offset),
                        bytes.as_mut_ptr(),
                        part.bytes,
                    );
                }
                let value = u128::from_le_bytes(bytes);
                let range = monitor::WriteRange::new(
                    Arc::clone(first.view().backing()),
                    first.backing_offset() as u64,
                    part.bytes as u64,
                )
                .map_err(|_| (MemoryStatus::Domain, raw))?;
                let policy = if physical.attributes.shareable {
                    monitor::Invalidation::BeforeAttempt
                } else {
                    monitor::Invalidation::OnCommit
                };
                let target = self.domain.target;
                let mut mutation = self
                    .domain
                    .system
                    .prepare_source_write(
                        first.view().backing(),
                        first.backing_offset(),
                        part.bytes,
                    )
                    .map_err(|status| (status, raw))?;
                self.domain
                    .system
                    .reservations
                    .with_writer(&range, policy, || {
                        self.policy
                            .check(physical.data(raw, part.bytes, part.flags))?;
                        mutation.attempting();
                        // SAFETY: Resolved spans pin the complete writable footprint and its target
                        // requirements. Private staging is disjoint and the physical writer gate is held.
                        unsafe { access::write_part(pointer, part.bytes, value, target) }
                            .map_err(|_| MemoryStatus::Unsupported)?;
                        let _receipt = mutation.complete();
                        Ok::<_, MemoryStatus>(())
                    })
                    .map_err(|error| {
                        (
                            match error {
                                monitor::AccessError::Monitor(_) => MemoryStatus::Domain,
                                monitor::AccessError::Memory(status) => status,
                            },
                            raw,
                        )
                    })?;
            } else {
                self.policy
                    .check(physical.data(raw, part.bytes, part.flags))
                    .map_err(|status| (status, raw))?;
                let value = self
                    .domain
                    .read_part(first, part.bytes)
                    .map_err(|status| (status, raw))?;
                let bytes = value.to_le_bytes();
                // SAFETY: destination is private staging, not architectural RAM.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        request.data.add(part.offset),
                        part.bytes,
                    );
                }
            }
            view.completed_bytes = completed;
        }
        Ok(())
    }
}

fn is_guest_fault(status: u32) -> bool {
    [
        MemoryStatus::Translation,
        MemoryStatus::Permission,
        MemoryStatus::Alignment,
        MemoryStatus::StackAlignment,
        MemoryStatus::TagCheck,
    ]
    .into_iter()
    .any(|fault| status == fault as u32)
}

translated_callee! {
unsafe fn dispatch<P: DataPolicy>(
    view: *mut MemoryContextView,
    request: *const MemoryRequest,
) -> u32 {
    if view.is_null() || request.is_null() {
        return MemoryStatus::InvalidRequest as u32;
    }
    // SAFETY: the native invocation grants valid, disjoint ABI objects.
    let (view, request) = unsafe { (&mut *view, &*request) };
    if !view.has_supported_header() || view.opaque.is_null() {
        return MemoryStatus::InvalidRequest as u32;
    }
    // SAFETY: with_view binds this exact monomorphized owner for the invocation.
    let owner = unsafe { &mut *view.opaque.cast::<MemoryThread<P>>() };
    let result = crate::platform::unwind::invoke(core::panic::AssertUnwindSafe(|| {
        // SAFETY: dispatch inherits the valid private staging contract.
        unsafe { owner.transfer(view, request) }
    }));
    match result {
        Ok(Ok(())) => {
            if stela_compiler::context::memory::FaultAccess::from_code(view.fault_access).is_none()
            {
                owner.domain.system.failed.store(true, Ordering::Release);
                view.fault_kind = MemoryStatus::Domain as u32;
                return MemoryStatus::Domain as u32;
            }
            view.fault_access = 0;
            MemoryStatus::Success as u32
        }
        Ok(Err((status, address))) => {
            // Architectural fault cleanup is common at with_view return; a
            // platform/ABI decline must preserve the existing local mark.
            let status =
                if stela_compiler::context::memory::FaultAccess::from_code(view.fault_access)
                    .is_some()
                {
                    status
                } else {
                    owner.domain.system.failed.store(true, Ordering::Release);
                    MemoryStatus::Domain
                };
            // Keep the raw access code even on an ABI violation. Native-call
            // diagnostics must not normalize an unknown direction into success.
            view.fault_kind = status as u32;
            view.fault_address = address;
            view.fault_pc = request.pc;
            status as u32
        }
        Err(_) => {
            owner.domain.system.failed.store(true, Ordering::Release);
            let _ = owner
                .domain
                .system
                .reservations
                .clear(&mut owner.reservation);
            view.fault_kind = MemoryStatus::Domain as u32;
            view.fault_address = request.address;
            view.fault_pc = request.pc;
            MemoryStatus::Domain as u32
        }
    }
}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AccessPart {
    address: u64,
    offset: usize,
    bytes: usize,
    flags: u32,
}

struct AccessParts {
    address: u64,
    bytes: usize,
    offset: usize,
    part_bytes: usize,
    flags: u32,
}

impl AccessParts {
    fn new(address: u64, bytes: usize, descriptor: u32) -> Self {
        let residue = address as usize & (SCA_BYTES - 1);
        let asimd = descriptor & flags::ASIMD != 0 && descriptor & flags::SME == 0;
        let pair = descriptor & flags::PAIR != 0;
        let half = bytes / 2;
        // Exact Mem order for Normal-WB/LSE2/SCA16, LS64WB disabled and no
        // highest-address-first access. ASIMD wide cases precede LSE2; the
        // SVE-predicated 16B case follows it. Pair alignment uses half-width.
        let (part_bytes, clear_pair) = if asimd
            && bytes == SCA_BYTES
            && address.is_multiple_of(HALF_SCA_BYTES as u64)
        {
            (HALF_SCA_BYTES, false)
        } else if asimd && bytes == SCA_BYTES * 2 && address.is_multiple_of(HALF_SCA_BYTES as u64) {
            (HALF_SCA_BYTES, true)
        } else if residue + bytes <= SCA_BYTES {
            (bytes, false)
        } else if pair && half != 0 && address.is_multiple_of(half as u64)
            || descriptor & flags::SME == 0
                && descriptor & flags::SVE_PREDICATED != 0
                && bytes == SCA_BYTES
                && address.is_multiple_of(HALF_SCA_BYTES as u64)
        {
            (half, true)
        } else {
            (1, false)
        };
        let flags = if clear_pair {
            descriptor & !flags::PAIR
        } else {
            descriptor
        };
        Self {
            address,
            bytes,
            offset: 0,
            part_bytes,
            flags,
        }
    }
}

impl Iterator for AccessParts {
    type Item = AccessPart;
    fn next(&mut self) -> Option<Self::Item> {
        if self.offset == self.bytes {
            return None;
        }
        let part = AccessPart {
            address: self.address.wrapping_add(self.offset as u64),
            offset: self.offset,
            bytes: self.part_bytes,
            flags: self.flags,
        };
        self.offset += self.part_bytes;
        Some(part)
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests;
