//! Closed Host effects on a recovered, held physical worker. No CPU, Machine,
//! raw callback or borrowed Host is transported to another core.
use super::*;
use crate::machine::{
    Regime,
    mmu::tlbi::{Everywhere, Reach},
};
use crate::memory::{RegisteredSystemRam, system_ram::ClosedDescriptorFrame};
use alloc::boxed::Box;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// The owner-local part of stopped source-instruction settlement. RAM access
/// stays with the canonical all-writer owner; native entry uses NativeLoan.
#[derive(Debug)]
pub enum HostEffect {
    Regime(Box<Regime>),
    InvalidateTranslations,
    InvalidateReach(Reach, Everywhere),
    IsolatePredictions,
    CodeInvalidated(Option<u64>),
    ArmTimer(Option<u64>),
    DiscardCode,
    MemoryBarrier,
    RevokeRamCpuMappings(RamCpuMappingRevocation),
}

/// Issued only while the exact frame and its registered owner are held.
/// A worker may retire a wider set of local CPU translations to cover it.
#[derive(Debug)]
pub struct RamCpuMappingRevocation {
    ram: usize,
    generation: u64,
    physical: u64,
}

impl RamCpuMappingRevocation {
    pub(in crate::machine::engine) fn new(frame: &ClosedDescriptorFrame<'_>) -> Self {
        let service = frame.service();
        Self {
            ram: core::ptr::from_ref(service.registration()).addr(),
            generation: service.generation(),
            physical: frame.physical(),
        }
    }

    pub fn physical(&self) -> u64 {
        self.physical
    }

    pub fn belongs_to(&self, ram: &RegisteredSystemRam) -> bool {
        core::ptr::from_ref(ram).addr() == self.ram && ram.generation() == Ok(self.generation)
    }
}

#[derive(Debug)]
pub(super) struct ServiceIdentity {
    open: AtomicBool,
    completed: AtomicU64,
}
impl ServiceIdentity {
    fn reopen(mut closed: Arc<Self>) -> Option<Arc<Self>> {
        // A stale command/receipt or even a Weak observer must keep its old
        // identity permanently separate. Only an unaliased closed allocation
        // can become the next session; pointer equality alone is insufficient.
        let identity = Arc::get_mut(&mut closed)?;
        if *identity.open.get_mut() {
            return None;
        }
        *identity.completed.get_mut() = 0;
        *identity.open.get_mut() = true;
        Some(closed)
    }
}

/// A linear service session for one worker in one stopped cohort.
/// Dropping it cannot release that cohort. Only one effect may be outstanding;
/// its owner's receipt is required before another effect or close.
#[derive(Debug)]
#[must_use = "close the actual service session before releasing its cohort"]
pub struct WorkerServices {
    identity: Arc<Identity>,
    session: Arc<ServiceIdentity>,
    binding: WorkerBinding,
    stop: u64,
    issued: u64,
    pending: bool,
}

#[derive(Debug)]
#[must_use = "apply on the assigned held physical Host"]
pub struct HostEffectCommand {
    identity: Arc<Identity>,
    session: Arc<ServiceIdentity>,
    binding: WorkerBinding,
    stop: u64,
    sequence: u64,
    effect: HostEffect,
}

#[derive(Debug)]
#[must_use = "consume the actual Host effect receipt in its service session"]
pub struct HostEffectReceipt {
    command: HostEffectCommand,
}

impl WorkerPublications {
    /// Exact explicit roster for leaf services which borrow only instantiated
    /// Cpu owners. Whole-ProcessorWrite still requires its full modeled roster.
    pub(in crate::machine::engine) fn require_held_members(
        &self,
        members: &[usize],
    ) -> Result<(), PublicationError> {
        self.require_held_roster(members.iter().copied())
    }
    /// Borrow either source PE numbers or the existing held binding slice;
    /// membership validation never needs a temporary owning allocation.
    fn require_held_roster(
        &self,
        members: impl ExactSizeIterator<Item = usize> + Clone,
    ) -> Result<(), PublicationError> {
        let count = members.len();
        if count == 0
            || self.slots.len() != count
            || self
                .slots
                .iter()
                .any(|slot| !members.clone().any(|pe| pe == slot.binding.pe))
            || members
                .clone()
                .enumerate()
                .any(|(i, pe)| members.clone().take(i).any(|before| before == pe))
        {
            return Err(PublicationError::ProcessorCount {
                expected: count,
                actual: self.slots.len(),
            });
        }
        let mut generation = None;
        for slot in &self.slots {
            if !matches!(slot.frame, Frame::Recovered(_)) {
                return Err(PublicationError::InFlight);
            }
            let Some((stop, StopPhase::Held)) = slot.stop else {
                return Err(PublicationError::MissingInvalidation);
            };
            if generation.is_some_and(|before| before != stop) {
                return Err(PublicationError::StaleStop);
            }
            generation = Some(stop);
        }
        Ok(())
    }
    pub(in crate::machine::engine) fn require_all_held(
        &self,
        held: &StoppedWorkers,
    ) -> Result<(), PublicationError> {
        self.check_stop(&held.request, StopPhase::Held)?;
        self.require_held_roster(held.request.members.iter().map(|binding| binding.pe))
    }
    pub(in crate::machine::engine) fn require_recovered(
        &self,
        binding: WorkerBinding,
        generation: u64,
    ) -> Result<(), PublicationError> {
        let slot = &self.slots[self.slot(binding.pe)?];
        if slot.binding != binding || slot.frame != Frame::Recovered(Some(generation)) {
            return Err(PublicationError::StaleRun);
        }
        Ok(())
    }

    pub fn bindings(&self) -> impl Iterator<Item = WorkerBinding> + '_ {
        self.slots.iter().map(|slot| slot.binding)
    }

    pub fn open_services(
        &mut self,
        held: &StoppedWorkers,
        pe: usize,
    ) -> Result<WorkerServices, PublicationError> {
        self.check_stop(&held.request, StopPhase::Held)?;
        let index = self.slot(pe)?;
        let slot = &mut self.slots[index];
        if !held.request.members.contains(&slot.binding) {
            return Err(PublicationError::UnknownPe(pe));
        }
        if slot.services.is_some() {
            return Err(PublicationError::ServiceBusy);
        }
        let session = slot
            .closed_services
            .take()
            .and_then(ServiceIdentity::reopen)
            .unwrap_or_else(|| {
                Arc::new(ServiceIdentity {
                    open: AtomicBool::new(true),
                    completed: AtomicU64::new(0),
                })
            });
        slot.services = Some(Arc::clone(&session));
        Ok(WorkerServices {
            identity: Arc::clone(&self.identity),
            session,
            binding: slot.binding,
            stop: held.request.generation,
            issued: 0,
            pending: false,
        })
    }

    pub fn close_services(
        &mut self,
        services: WorkerServices,
    ) -> Result<(), WithPayload<WorkerServices>> {
        let check = (|| {
            self.identity(&services.identity)?;
            let index = self.slot(services.binding.pe)?;
            let slot = &self.slots[index];
            if slot.binding != services.binding
                || slot.stop != Some((services.stop, StopPhase::Held))
                || !slot
                    .services
                    .as_ref()
                    .is_some_and(|session| Arc::ptr_eq(session, &services.session))
            {
                return Err(PublicationError::StaleService);
            }
            if services.pending
                || services.session.completed.load(Ordering::Acquire) != services.issued
            {
                return Err(PublicationError::ServiceBusy);
            }
            Ok(index)
        })();
        match check {
            Ok(index) => {
                services.session.open.store(false, Ordering::Release);
                self.slots[index].services = None;
                // Consume the completed session, after all existing
                // generation/pending/receipt checks. The next open must prove
                // that no reference to this closed identity survives.
                self.slots[index].closed_services = Some(services.session);
                Ok(())
            }
            Err(error) => Err(WithPayload {
                error,
                payload: services,
            }),
        }
    }
}

impl WorkerServices {
    pub fn binding(&self) -> WorkerBinding {
        self.binding
    }

    pub fn command(&mut self, effect: HostEffect) -> Result<HostEffectCommand, PublicationError> {
        if self.pending {
            return Err(PublicationError::ServiceBusy);
        }
        if !self.session.open.load(Ordering::Acquire) {
            return Err(PublicationError::StaleService);
        }
        let sequence = self
            .issued
            .checked_add(1)
            .ok_or(PublicationError::GenerationExhausted)?;
        self.issued = sequence;
        self.pending = true;
        Ok(HostEffectCommand {
            identity: Arc::clone(&self.identity),
            session: Arc::clone(&self.session),
            binding: self.binding,
            stop: self.stop,
            sequence,
            effect,
        })
    }

    pub fn accept(
        &mut self,
        receipt: HostEffectReceipt,
    ) -> Result<(), WithPayload<HostEffectReceipt>> {
        let command = &receipt.command;
        if !self.pending
            || !Arc::ptr_eq(&self.identity, &command.identity)
            || !Arc::ptr_eq(&self.session, &command.session)
            || self.binding != command.binding
            || self.stop != command.stop
            || self.issued != command.sequence
            || self.session.completed.load(Ordering::Acquire) != command.sequence
        {
            return Err(WithPayload {
                error: PublicationError::StaleService,
                payload: receipt,
            });
        }
        self.pending = false;
        Ok(())
    }
}

impl HostEffectCommand {
    pub fn binding(&self) -> WorkerBinding {
        self.binding
    }
    pub fn generation(&self) -> u64 {
        self.stop
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn effect_name(&self) -> &'static str {
        match self.effect {
            HostEffect::Regime(_) => "regime",
            HostEffect::InvalidateTranslations => "invalidate",
            HostEffect::InvalidateReach(..) => "reach",
            HostEffect::IsolatePredictions => "isolate",
            HostEffect::CodeInvalidated(_) => "code",
            HostEffect::ArmTimer(_) => "timer",
            HostEffect::DiscardCode => "discard",
            HostEffect::MemoryBarrier => "cpu-memory-fence",
            HostEffect::RevokeRamCpuMappings(_) => "revoke-ram-cpu-mappings",
        }
    }
}

impl WorkerOwner {
    pub fn apply_host_effect(
        &mut self,
        command: HostEffectCommand,
        host: &mut dyn Host,
    ) -> Result<HostEffectReceipt, WithPayload<HostEffectCommand>> {
        let check = self.check(&command.identity, command.binding)
            .and_then(|()| self.check_host(host))
            .and_then(|()| {
                if !matches!(self.state, OwnerState::Held { generation, .. } if generation == command.stop)
                    || !command.session.open.load(Ordering::Acquire)
                    || command.session.completed.load(Ordering::Acquire).checked_add(1) != Some(command.sequence)
                {
                    Err(PublicationError::StaleService)
                } else {
                    Ok(())
                }
            });
        if let Err(error) = check {
            return Err(WithPayload {
                error,
                payload: command,
            });
        }
        match &command.effect {
            HostEffect::Regime(regime) => host.set_regime(regime),
            HostEffect::InvalidateTranslations => host.invalidate_translations(),
            HostEffect::InvalidateReach(reach, everywhere) => {
                host.invalidate_reach(*reach, *everywhere)
            }
            HostEffect::IsolatePredictions => host.isolate_predictions(),
            HostEffect::CodeInvalidated(page) => host.code_invalidated(*page),
            HostEffect::ArmTimer(deadline) => host.arm_timer(*deadline),
            HostEffect::DiscardCode => host.discard_code(),
            HostEffect::MemoryBarrier => host.memory_barrier(),
            HostEffect::RevokeRamCpuMappings(request) => {
                if !host.revoke_ram_cpu_mappings(request) {
                    return Err(WithPayload {
                        error: PublicationError::UnqualifiedHost,
                        payload: command,
                    });
                }
            }
        }
        command
            .session
            .completed
            .store(command.sequence, Ordering::Release);
        Ok(HostEffectReceipt { command })
    }
}

#[cfg(test)]
mod tests;
