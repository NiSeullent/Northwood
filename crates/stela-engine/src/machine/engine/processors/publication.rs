//! Linear ownership at native worker boundaries. This is the control-plane
//! protocol, not an AP transport or a replacement for recovering a native exit.
//! Only Machine preparation can create a native loan, and only Machine recovery
//! can consume its returned payload. A stop ACK alone has no such power.
use super::Host;
use alloc::{sync::Arc, vec::Vec};

mod service;
pub use service::{
    HostEffect, HostEffectCommand, HostEffectReceipt, RamCpuMappingRevocation, WorkerServices,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerBinding {
    /// Index in the unchanged guest CPU/resource profile, independent of the
    /// physical worker and the ordering of the publication roster.
    pub pe: usize,
    /// Fixed physical owner. A nonzero guest boot PE may belong to BSP worker 0;
    /// every owner must still publish and execute on its exact checked Host.
    pub worker: WorkerId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicationError {
    EmptyCohort,
    DuplicatePe(usize),
    ProcessorCount { expected: usize, actual: usize },
    DuplicateWorker(WorkerId),
    UnknownPe(usize),
    WrongInventory,
    WrongOwner,
    UnqualifiedHost,
    NotInitialized,
    AlreadyInitialized,
    InFlight,
    StaleRun,
    StopBusy,
    StaleStop,
    MissingInvalidation,
    GenerationExhausted,
    ServiceBusy,
    StaleService,
}

#[derive(Debug)]
struct Identity;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Frame {
    Unregistered,
    Recovered(Option<u64>),
    InFlight(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StopPhase {
    Requested,
    CommandIssued,
    Invalidated,
    Held,
    Releasing,
}

#[derive(Debug)]
struct Slot {
    binding: WorkerBinding,
    frame: Frame,
    stop: Option<(u64, StopPhase)>,
    services: Option<Arc<service::ServiceIdentity>>,
    // At most one closed allocation for this fixed physical owner. It grants
    // no service/stop permission and is reusable only after unique ownership.
    closed_services: Option<Arc<service::ServiceIdentity>>,
}

/// BSP-local publication inventory. Creating it does not assert that an AP is
/// stopped: every slot requires its unique owner's initial publication first.
/// This type contains no Cpu, GS-local Host, guest state or copied execution view.
#[derive(Debug)]
pub struct WorkerPublications {
    identity: Arc<Identity>,
    generation: u64,
    slots: Vec<Slot>,
}

#[derive(Debug)]
enum OwnerState {
    Unregistered,
    Returned(Option<u64>),
    Held { generation: u64, run: Option<u64> },
}

/// One permission per executor. The board transfers it to the assigned
/// physical worker and keeps it there. It cannot be cloned or substituted by
/// a numeric worker ID. Obtaining that worker's Host must still enforce its
/// physical-owner/GS check; this protocol never makes Host or Cpu Send.
#[derive(Debug)]
pub struct WorkerOwner {
    identity: Arc<Identity>,
    binding: WorkerBinding,
    state: OwnerState,
}

#[derive(Debug)]
#[must_use = "publish the actual owner's initialization before dispatch"]
pub struct InitialPublication {
    identity: Arc<Identity>,
    binding: WorkerBinding,
}

#[derive(Debug)]
struct Run {
    identity: Arc<Identity>,
    binding: WorkerBinding,
    generation: u64,
}

// Only the engine's concrete prepared-frame implementation can open this
// boundary. Boot transport cannot manufacture a return with a no-op callback.
pub(in crate::machine::engine) mod sealed {
    pub trait NativeFrame {}
}

/// A complete prepared native frame. Implementations are sealed to the engine;
/// they must keep every native-reachable service/code pin and perform
/// Host execution and exit capture. Cpu and Host are never payload members.
pub trait NativeFrame: sealed::NativeFrame + Send {
    type Returned: Send;
    fn execute(self, host: &mut dyn Host) -> Self::Returned;
}

/// The native frame and all reachable services/code pins travel with
/// this loan. Dropping a loan or its return never marks a worker recovered.
#[derive(Debug)]
#[must_use = "return the actual native frame to Machine recovery"]
pub struct NativeLoan<T> {
    run: Run,
    payload: T,
}

/// Release-publish this object through the AP transport, then Acquire-consume
/// it on the BSP. No payload accessor or public recovery constructor exists.
#[derive(Debug)]
#[must_use = "Machine must recover this actual native exit before settlement"]
pub struct NativeReturned<T> {
    run: Run,
    payload: T,
}

/// A refusal returns the still-owned payload/token, rather than losing the
/// frame or letting a caller replace it with a parked snapshot.
#[derive(Debug)]
pub struct WithPayload<T, E = PublicationError> {
    pub error: E,
    pub payload: T,
}

#[derive(Debug)]
pub enum RecoveryError<E> {
    Publication(PublicationError),
    Recovery(E),
}

/// Immutable affected membership and generation, used by the transport to
/// request real exits. Dropping it leaves the stop outstanding and dispatch
/// refused; cancellation cannot silently release an AP still processing it.
#[derive(Debug)]
#[must_use = "complete every actual return and retain the stop through ACK"]
pub struct StopRequest {
    identity: Arc<Identity>,
    generation: u64,
    members: Vec<WorkerBinding>,
}

#[derive(Debug)]
#[must_use = "execute on the assigned actual native owner"]
pub struct NativeStopCommand {
    identity: Arc<Identity>,
    binding: WorkerBinding,
    stop: u64,
    last_run: Option<u64>,
}

#[derive(Debug)]
#[must_use = "publish the owner-local invalidation receipt"]
pub struct NativeStopReceipt {
    command: NativeStopCommand,
}

/// A whole cohort is held only after every payload has been recovered
/// and every native owner has invalidated its dispatch state. Keep
/// this linear release permission until the controller's final ACK commit.
#[derive(Debug)]
#[must_use = "release the same stopped cohort only after controller ACK"]
pub struct StoppedWorkers {
    request: StopRequest,
}

/// Linear permission to rejoin one selectively released execution round. Every
/// omitted owner remains physically held in the original stopped generation.
#[derive(Debug)]
#[must_use = "recover and invalidate every released owner before rejoining"]
pub struct RunningCohort {
    request: StopRequest,
    released: Vec<(WorkerBinding, Option<u64>)>,
}

#[derive(Debug)]
#[must_use = "release on the actual owner; dispatch remains refused until receipt"]
pub struct WorkerRelease {
    identity: Arc<Identity>,
    binding: WorkerBinding,
    stop: u64,
}

#[derive(Debug)]
#[must_use = "return the same-generation actual-owner release receipt"]
pub struct ReleaseReceipt {
    release: WorkerRelease,
}

impl WorkerPublications {
    pub fn new(bindings: &[WorkerBinding]) -> Result<(Self, Vec<WorkerOwner>), PublicationError> {
        if bindings.is_empty() {
            return Err(PublicationError::EmptyCohort);
        }
        for (i, binding) in bindings.iter().enumerate() {
            if bindings[..i].iter().any(|other| other.pe == binding.pe) {
                return Err(PublicationError::DuplicatePe(binding.pe));
            }
            if bindings[..i]
                .iter()
                .any(|other| other.worker == binding.worker)
            {
                return Err(PublicationError::DuplicateWorker(binding.worker));
            }
        }
        let identity = Arc::new(Identity);
        let owners = bindings
            .iter()
            .map(|&binding| WorkerOwner {
                identity: identity.clone(),
                binding,
                state: OwnerState::Unregistered,
            })
            .collect();
        Ok((
            Self {
                identity,
                generation: 0,
                slots: bindings
                    .iter()
                    .map(|&binding| Slot {
                        binding,
                        frame: Frame::Unregistered,
                        stop: None,
                        services: None,
                        closed_services: None,
                    })
                    .collect(),
            },
            owners,
        ))
    }

    fn slot(&self, pe: usize) -> Result<usize, PublicationError> {
        self.slots
            .iter()
            .position(|slot| slot.binding.pe == pe)
            .ok_or(PublicationError::UnknownPe(pe))
    }

    fn identity(&self, identity: &Arc<Identity>) -> Result<(), PublicationError> {
        if Arc::ptr_eq(&self.identity, identity) {
            Ok(())
        } else {
            Err(PublicationError::WrongInventory)
        }
    }

    fn next(&self) -> Result<u64, PublicationError> {
        self.generation
            .checked_add(1)
            .ok_or(PublicationError::GenerationExhausted)
    }

    pub fn accept_initial(
        &mut self,
        publication: InitialPublication,
    ) -> Result<(), WithPayload<InitialPublication>> {
        let check = (|| {
            self.identity(&publication.identity)?;
            let index = self.slot(publication.binding.pe)?;
            let slot = &self.slots[index];
            if slot.binding != publication.binding {
                return Err(PublicationError::WrongOwner);
            }
            if slot.frame != Frame::Unregistered {
                return Err(PublicationError::AlreadyInitialized);
            }
            Ok(index)
        })();
        match check {
            Ok(index) => {
                self.slots[index].frame = Frame::Recovered(None);
                Ok(())
            }
            Err(error) => Err(WithPayload {
                error,
                payload: publication,
            }),
        }
    }

    /// Machine alone supplies the complete frame/service/code loan.
    /// No run can start until the previous return and any stop release finished.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "complete native-frame preparation is not yet joined"
        )
    )]
    pub(in crate::machine::engine) fn dispatch<T>(
        &mut self,
        pe: usize,
        payload: T,
    ) -> Result<NativeLoan<T>, WithPayload<T>> {
        let check = (|| {
            let index = self.slot(pe)?;
            let slot = &self.slots[index];
            if slot.stop.is_some() {
                return Err(PublicationError::StopBusy);
            }
            match slot.frame {
                Frame::Unregistered => return Err(PublicationError::NotInitialized),
                Frame::InFlight(_) => return Err(PublicationError::InFlight),
                Frame::Recovered(_) => {}
            }
            Ok((index, self.next()?))
        })();
        match check {
            Err(error) => Err(WithPayload { error, payload }),
            Ok((index, generation)) => {
                self.generation = generation;
                self.slots[index].frame = Frame::InFlight(generation);
                Ok(NativeLoan {
                    run: Run {
                        identity: self.identity.clone(),
                        binding: self.slots[index].binding,
                        generation,
                    },
                    payload,
                })
            }
        }
    }

    /// Called only after Acquire of the AP publication. The trusted
    /// Machine callback must complete precise fault/retirement/frame recovery.
    /// A failed callback keeps the real return and blocks all stop credit.
    pub(in crate::machine::engine) fn recover<T, E>(
        &mut self,
        mut returned: NativeReturned<T>,
        recover: impl FnOnce(&mut T) -> Result<(), E>,
    ) -> Result<(), WithPayload<NativeReturned<T>, RecoveryError<E>>> {
        let check = (|| {
            self.identity(&returned.run.identity)?;
            let index = self.slot(returned.run.binding.pe)?;
            let slot = &self.slots[index];
            if slot.binding != returned.run.binding {
                return Err(PublicationError::WrongOwner);
            }
            if slot.frame != Frame::InFlight(returned.run.generation) {
                return Err(PublicationError::StaleRun);
            }
            Ok(index)
        })();
        let index = match check {
            Ok(index) => index,
            Err(error) => {
                return Err(WithPayload {
                    error: RecoveryError::Publication(error),
                    payload: returned,
                });
            }
        };
        if let Err(error) = recover(&mut returned.payload) {
            return Err(WithPayload {
                error: RecoveryError::Recovery(error),
                payload: returned,
            });
        }
        self.slots[index].frame = Frame::Recovered(Some(returned.run.generation));
        // Dropping the consumed real payload releases its GuestState/service
        // loans. No borrow of a prior native frame survives as stopped credit.
        Ok(())
    }

    pub fn request_stop(&mut self, members: &[usize]) -> Result<StopRequest, PublicationError> {
        if members.is_empty() {
            return Err(PublicationError::EmptyCohort);
        }
        let mut indices = Vec::with_capacity(members.len());
        for (i, &pe) in members.iter().enumerate() {
            if members[..i].contains(&pe) {
                return Err(PublicationError::DuplicatePe(pe));
            }
            let index = self.slot(pe)?;
            if self.slots[index].frame == Frame::Unregistered {
                return Err(PublicationError::NotInitialized);
            }
            if self.slots[index].stop.is_some() {
                return Err(PublicationError::StopBusy);
            }
            indices.push(index);
        }
        let generation = self.next()?;
        self.generation = generation;
        for &index in &indices {
            self.slots[index].stop = Some((generation, StopPhase::Requested));
        }
        Ok(StopRequest {
            identity: self.identity.clone(),
            generation,
            members: indices
                .iter()
                .map(|&index| self.slots[index].binding)
                .collect(),
        })
    }

    fn check_stop(&self, request: &StopRequest, phase: StopPhase) -> Result<(), PublicationError> {
        self.identity(&request.identity)?;
        for member in &request.members {
            let slot = &self.slots[self.slot(member.pe)?];
            if slot.binding != *member || slot.stop != Some((request.generation, phase)) {
                return Err(PublicationError::StaleStop);
            }
            if !matches!(slot.frame, Frame::Recovered(_)) {
                return Err(PublicationError::InFlight);
            }
        }
        Ok(())
    }

    pub fn native_stop_commands(
        &mut self,
        request: &StopRequest,
    ) -> Result<Vec<NativeStopCommand>, PublicationError> {
        self.check_stop(request, StopPhase::Requested)?;
        let mut commands = Vec::with_capacity(request.members.len());
        for member in &request.members {
            let slot = &mut self
                .slots
                .iter_mut()
                .find(|slot| slot.binding == *member)
                .unwrap();
            let Frame::Recovered(last_run) = slot.frame else {
                unreachable!()
            };
            slot.stop = Some((request.generation, StopPhase::CommandIssued));
            commands.push(NativeStopCommand {
                identity: self.identity.clone(),
                binding: *member,
                stop: request.generation,
                last_run,
            });
        }
        Ok(commands)
    }

    pub fn accept_invalidation(
        &mut self,
        receipt: NativeStopReceipt,
    ) -> Result<(), WithPayload<NativeStopReceipt>> {
        let command = &receipt.command;
        let check = (|| {
            self.identity(&command.identity)?;
            let index = self.slot(command.binding.pe)?;
            let slot = &self.slots[index];
            if slot.binding != command.binding
                || slot.frame != Frame::Recovered(command.last_run)
                || slot.stop != Some((command.stop, StopPhase::CommandIssued))
            {
                return Err(PublicationError::StaleStop);
            }
            Ok(index)
        })();
        match check {
            Ok(index) => {
                self.slots[index].stop = Some((command.stop, StopPhase::Invalidated));
                Ok(())
            }
            Err(error) => Err(WithPayload {
                error,
                payload: receipt,
            }),
        }
    }

    pub fn hold_stop(
        &mut self,
        request: StopRequest,
    ) -> Result<StoppedWorkers, WithPayload<StopRequest>> {
        if let Err(error) = self.check_stop(&request, StopPhase::Invalidated) {
            return Err(WithPayload {
                error,
                payload: request,
            });
        }
        for slot in &mut self.slots {
            if slot.stop == Some((request.generation, StopPhase::Invalidated)) {
                slot.stop = Some((request.generation, StopPhase::Held));
            }
        }
        Ok(StoppedWorkers { request })
    }

    /// The first join borrows the complete Cpu/peers view, so all workers
    /// must be held in the same generation, including unaffected PEs. A later
    /// targeted-only join requires disjoint per-PE owner borrows.
    pub(in crate::machine::engine) fn require_stopped(
        &self,
        count: usize,
        members: &[usize],
    ) -> Result<(), PublicationError> {
        if self.slots.len() != count {
            return Err(PublicationError::ProcessorCount {
                expected: count,
                actual: self.slots.len(),
            });
        }
        for slot in &self.slots {
            if slot.binding.pe >= count {
                return Err(PublicationError::UnknownPe(slot.binding.pe));
            }
        }
        if members.is_empty() {
            return Err(PublicationError::EmptyCohort);
        }
        for &pe in members {
            self.slot(pe)?;
        }
        let mut generation = None;
        for slot in &self.slots {
            if !matches!(slot.frame, Frame::Recovered(_)) {
                return Err(PublicationError::InFlight);
            }
            let Some((stop, StopPhase::Held)) = slot.stop else {
                return Err(PublicationError::MissingInvalidation);
            };
            if generation.is_some_and(|previous| previous != stop) {
                return Err(PublicationError::StaleStop);
            }
            generation = Some(stop);
        }
        Ok(())
    }

    /// Call after dropping the successful ProcessorWrite lease, after ACK.
    /// Dispatch remains blocked until each owner consumes its release.
    pub fn release(
        &mut self,
        held: StoppedWorkers,
    ) -> Result<Vec<WorkerRelease>, WithPayload<StoppedWorkers>> {
        let check = self
            .check_stop(&held.request, StopPhase::Held)
            .and_then(|()| {
                if self.slots.iter().any(|slot| slot.services.is_some()) {
                    Err(PublicationError::ServiceBusy)
                } else {
                    Ok(())
                }
            });
        if let Err(error) = check {
            return Err(WithPayload {
                error,
                payload: held,
            });
        }
        let mut releases = Vec::with_capacity(held.request.members.len());
        for slot in &mut self.slots {
            if slot.stop == Some((held.request.generation, StopPhase::Held)) {
                slot.stop = Some((held.request.generation, StopPhase::Releasing));
                releases.push(WorkerRelease {
                    identity: self.identity.clone(),
                    binding: slot.binding,
                    stop: held.request.generation,
                });
            }
        }
        Ok(releases)
    }

    /// Release only owners with prepared native frames. The complete
    /// stopped request remains held; omitted owners never lose their hold.
    pub fn release_for_run(
        &mut self,
        held: StoppedWorkers,
        members: &[usize],
    ) -> Result<(RunningCohort, Vec<WorkerRelease>), WithPayload<StoppedWorkers>> {
        let check = (|| {
            self.check_stop(&held.request, StopPhase::Held)?;
            if self.slots.iter().any(|slot| slot.services.is_some()) {
                return Err(PublicationError::ServiceBusy);
            }
            if members.is_empty() {
                return Err(PublicationError::EmptyCohort);
            }
            let mut released = Vec::with_capacity(members.len());
            for (i, &pe) in members.iter().enumerate() {
                if members[..i].contains(&pe) {
                    return Err(PublicationError::DuplicatePe(pe));
                }
                let slot = &self.slots[self.slot(pe)?];
                if !held.request.members.contains(&slot.binding) {
                    return Err(PublicationError::UnknownPe(pe));
                }
                let Frame::Recovered(last_run) = slot.frame else {
                    return Err(PublicationError::InFlight);
                };
                released.push((slot.binding, last_run));
            }
            Ok(released)
        })();
        let released = match check {
            Ok(released) => released,
            Err(error) => {
                return Err(WithPayload {
                    error,
                    payload: held,
                });
            }
        };
        let releases = released
            .iter()
            .map(|(binding, _)| WorkerRelease {
                identity: self.identity.clone(),
                binding: *binding,
                stop: held.request.generation,
            })
            .collect();
        for (binding, _) in &released {
            let index = self.slot(binding.pe).unwrap();
            self.slots[index].stop = Some((held.request.generation, StopPhase::Releasing));
        }
        Ok((
            RunningCohort {
                request: held.request,
                released,
            },
            releases,
        ))
    }

    /// Every released owner must have supplied a new, recovered native
    /// return. Idle owners keep their original physical Held receipt;
    /// only the released owners receive fresh fence/invalidation commands.
    pub fn request_rejoin(
        &mut self,
        running: &RunningCohort,
    ) -> Result<Vec<NativeStopCommand>, PublicationError> {
        self.identity(&running.request.identity)?;
        for member in &running.request.members {
            let slot = &self.slots[self.slot(member.pe)?];
            if slot.binding != *member {
                return Err(PublicationError::WrongOwner);
            }
            let Frame::Recovered(last_run) = slot.frame else {
                return Err(PublicationError::InFlight);
            };
            if let Some((_, previous)) = running
                .released
                .iter()
                .find(|(binding, _)| binding == member)
            {
                if slot.stop.is_some() {
                    return Err(PublicationError::StaleStop);
                }
                if !last_run.is_some_and(|run| previous.is_none_or(|previous| run > previous)) {
                    return Err(PublicationError::StaleRun);
                }
            } else if slot.stop != Some((running.request.generation, StopPhase::Held)) {
                return Err(PublicationError::MissingInvalidation);
            }
            if slot.services.is_some() {
                return Err(PublicationError::ServiceBusy);
            }
        }
        let mut commands = Vec::with_capacity(running.released.len());
        for (binding, _) in &running.released {
            let index = self.slot(binding.pe)?;
            let Frame::Recovered(last_run) = self.slots[index].frame else {
                unreachable!()
            };
            self.slots[index].stop = Some((running.request.generation, StopPhase::CommandIssued));
            commands.push(NativeStopCommand {
                identity: self.identity.clone(),
                binding: *binding,
                stop: running.request.generation,
                last_run,
            });
        }
        Ok(commands)
    }

    /// Reconstitute the original complete all-writer hold only after all new
    /// owner invalidation receipts were accepted. No receipt is rebased
    /// and no guest power/park state substitutes for physical owner permission.
    pub fn rejoin(
        &mut self,
        running: RunningCohort,
    ) -> Result<StoppedWorkers, WithPayload<RunningCohort>> {
        let check = (|| {
            self.identity(&running.request.identity)?;
            for member in &running.request.members {
                let slot = &self.slots[self.slot(member.pe)?];
                if slot.binding != *member || !matches!(slot.frame, Frame::Recovered(_)) {
                    return Err(PublicationError::InFlight);
                }
                let phase = if running
                    .released
                    .iter()
                    .any(|(binding, _)| binding == member)
                {
                    StopPhase::Invalidated
                } else {
                    StopPhase::Held
                };
                if slot.stop != Some((running.request.generation, phase)) {
                    return Err(PublicationError::MissingInvalidation);
                }
                if slot.services.is_some() {
                    return Err(PublicationError::ServiceBusy);
                }
            }
            Ok(())
        })();
        if let Err(error) = check {
            return Err(WithPayload {
                error,
                payload: running,
            });
        }
        for (binding, _) in &running.released {
            let index = self.slot(binding.pe).unwrap();
            self.slots[index].stop = Some((running.request.generation, StopPhase::Held));
        }
        Ok(StoppedWorkers {
            request: running.request,
        })
    }

    pub fn accept_release(
        &mut self,
        receipt: ReleaseReceipt,
    ) -> Result<(), WithPayload<ReleaseReceipt>> {
        let release = &receipt.release;
        let check = (|| {
            self.identity(&release.identity)?;
            let index = self.slot(release.binding.pe)?;
            let slot = &self.slots[index];
            if slot.binding != release.binding
                || slot.stop != Some((release.stop, StopPhase::Releasing))
            {
                return Err(PublicationError::StaleStop);
            }
            Ok(index)
        })();
        match check {
            Ok(index) => {
                self.slots[index].stop = None;
                Ok(())
            }
            Err(error) => Err(WithPayload {
                error,
                payload: receipt,
            }),
        }
    }
}

impl WorkerOwner {
    pub fn binding(&self) -> WorkerBinding {
        self.binding
    }

    pub fn publish_initial(
        &mut self,
        host: &mut dyn Host,
    ) -> Result<InitialPublication, PublicationError> {
        if !matches!(self.state, OwnerState::Unregistered) {
            return Err(PublicationError::AlreadyInitialized);
        }
        self.check_host(host)?;
        host.invalidate_translations();
        self.state = OwnerState::Returned(None);
        Ok(InitialPublication {
            identity: self.identity.clone(),
            binding: self.binding,
        })
    }

    fn check(
        &self,
        identity: &Arc<Identity>,
        binding: WorkerBinding,
    ) -> Result<(), PublicationError> {
        if !Arc::ptr_eq(&self.identity, identity) {
            return Err(PublicationError::WrongInventory);
        }
        if self.binding != binding {
            return Err(PublicationError::WrongOwner);
        }
        Ok(())
    }

    fn check_host(&self, host: &dyn Host) -> Result<(), PublicationError> {
        match host.native_worker() {
            None => Err(PublicationError::UnqualifiedHost),
            Some(worker) if worker == self.binding.worker => Ok(()),
            Some(_) => Err(PublicationError::WrongOwner),
        }
    }

    pub fn execute<T: NativeFrame>(
        &mut self,
        loan: NativeLoan<T>,
        host: &mut dyn Host,
    ) -> Result<NativeReturned<T::Returned>, WithPayload<NativeLoan<T>>> {
        let check = (|| {
            self.check(&loan.run.identity, loan.run.binding)?;
            self.check_host(host)?;
            match self.state {
                OwnerState::Unregistered => return Err(PublicationError::NotInitialized),
                OwnerState::Held { .. } => return Err(PublicationError::StopBusy),
                OwnerState::Returned(last)
                    if last.is_some_and(|last| last >= loan.run.generation) =>
                {
                    return Err(PublicationError::StaleRun);
                }
                OwnerState::Returned(_) => {}
            }
            Ok(())
        })();
        if let Err(error) = check {
            return Err(WithPayload {
                error,
                payload: loan,
            });
        }
        let payload = loan.payload.execute(host);
        self.state = OwnerState::Returned(Some(loan.run.generation));
        Ok(NativeReturned {
            run: loan.run,
            payload,
        })
    }

    /// This effect runs on the executor's Host after its frame returned.
    /// It checks dispatch invalidation only; it does not replace complete
    /// Machine cache/service/regime settlement or imply a hardware power state.
    pub fn invalidate(
        &mut self,
        command: NativeStopCommand,
        host: &mut dyn Host,
    ) -> Result<NativeStopReceipt, WithPayload<NativeStopCommand>> {
        let check = self
            .check(&command.identity, command.binding)
            .and_then(|()| match self.state {
                OwnerState::Returned(last) if last == command.last_run => Ok(()),
                _ => Err(PublicationError::StaleStop),
            });
        if let Err(error) = check {
            return Err(WithPayload {
                error,
                payload: command,
            });
        }
        if let Err(error) = self.check_host(host) {
            return Err(WithPayload {
                error,
                payload: command,
            });
        }
        host.memory_barrier();
        host.invalidate_translations();
        self.state = OwnerState::Held {
            generation: command.stop,
            run: command.last_run,
        };
        Ok(NativeStopReceipt { command })
    }

    pub fn release(
        &mut self,
        release: WorkerRelease,
    ) -> Result<ReleaseReceipt, WithPayload<WorkerRelease>> {
        let check = self
            .check(&release.identity, release.binding)
            .and(match self.state {
                OwnerState::Held { generation, .. } if generation == release.stop => Ok(()),
                _ => Err(PublicationError::StaleStop),
            });
        if let Err(error) = check {
            return Err(WithPayload {
                error,
                payload: release,
            });
        }
        let OwnerState::Held { run, .. } = self.state else {
            unreachable!()
        };
        self.state = OwnerState::Returned(run);
        Ok(ReleaseReceipt { release })
    }
}

impl<T> NativeLoan<T> {
    pub fn binding(&self) -> WorkerBinding {
        self.run.binding
    }
    pub fn generation(&self) -> u64 {
        self.run.generation
    }
}
impl<T> NativeReturned<T> {
    pub fn binding(&self) -> WorkerBinding {
        self.run.binding
    }
    pub fn generation(&self) -> u64 {
        self.run.generation
    }
}
impl StopRequest {
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn members(&self) -> &[WorkerBinding] {
        &self.members
    }
}
impl NativeStopCommand {
    pub fn binding(&self) -> WorkerBinding {
        self.binding
    }
    pub fn generation(&self) -> u64 {
        self.stop
    }
}
impl WorkerRelease {
    pub fn binding(&self) -> WorkerBinding {
        self.binding
    }
    pub fn generation(&self) -> u64 {
        self.stop
    }
}

#[cfg(test)]
mod tests;
