//! Borrow the same canonical RAM while every native owner remains held.
use super::*;
use crate::machine::{
    Stop,
    engine::{RequirementKind, processors::Peer},
};
use crate::memory::{
    CanonicalRamService,
    system_ram::{ClosedDescriptorFrame, NonresidentRamFrame},
};
use publication::RamCpuMappingRevocation;
use stela_compiler::context::memory::MemoryStatus;

#[cfg(all(feature = "std", feature = "test-support"))]
mod reference;

/// The exact frame remains closed to registered I/O and the held workers
/// have revoked their guest-facing CPU mappings for the held interval.
#[must_use]
pub struct CpuMappingsRevokedFrame<'a>(ClosedDescriptorFrame<'a>);

impl CpuMappingsRevokedFrame<'_> {
    pub fn physical(&self) -> u64 {
        self.0.physical()
    }

    /// # Safety
    /// The output must be private and disjoint from live RAM/native frames.
    pub unsafe fn copy_into(&self, output: &mut [u8; 16 * 1024]) -> Result<(), MemoryStatus> {
        // SAFETY: forwarded unchanged private-output contract.
        unsafe { self.0.copy_into(output) }
    }
}

impl<'a> CpuMappingsRevokedFrame<'a> {
    /// Hide this frame while retaining the original backing and worker hold.
    pub fn mark_nonresident(self) -> Result<NonresidentRamFrame<'a>, MemoryStatus> {
        // SAFETY: construction required the exact held workers' retirement
        // receipts. The token still borrows the service and release is gated.
        unsafe { self.0.mark_nonresident() }
    }
}

/// Canonical RAM and the exact all-worker stop remain borrowed through a
/// controller's own prepared effect and ACK. This does not implement Apple DRAM,
/// cache-power or external DMA effects. No numeric completion can construct it.
/// Dropping without explicit settlement fails the canonical owner and keeps
/// worker release blocked, so lost/unknown controller effects cannot resume.
#[must_use = "retain through the real prepared effect and ACK, then finish"]
pub struct HeldRamService<'a, H: Host, T: NativeTransport> {
    ram: Option<CanonicalRamService<'a>>,
    local: &'a mut H,
    transport: &'a mut T,
    inventory: &'a mut publication::WorkerPublications,
    sessions: &'a mut Vec<WorkerServices>,
    selected: &'a mut Option<usize>,
    held: Option<publication::StoppedWorkers>,
}
impl<H: Host, T: NativeTransport> HeldRamService<'_, H, T> {
    pub fn memory(&self) -> &CanonicalRamService<'_> {
        self.ram.as_ref().expect("live canonical service")
    }
    /// No caller can change inventory/generation while this exclusive borrow
    /// and the non-Clone stopped token remain live.
    pub fn validate_current(&self) -> Result<(), stela_compiler::context::memory::MemoryStatus> {
        self.memory().validate_current()?;
        self.inventory
            .require_all_held(self.held.as_ref().expect("actual held cohort"))
            .map_err(|_| stela_compiler::context::memory::MemoryStatus::Domain)
    }

    /// Close admission before issuing any owner-local CPU revocation. A
    /// busy or wired frame returns `None` without issuing a revocation.
    /// Finish the held service before waiting for a busy writer to return.
    /// Once issued, missing completion fails the RAM and keeps the cohort
    /// stopped. The returned borrow must end before releasing the workers.
    pub fn try_revoke_frame_cpu_mappings(
        &mut self,
        physical: u64,
    ) -> Result<Option<CpuMappingsRevokedFrame<'_>>, MemoryStatus> {
        self.validate_current()?;
        if !self
            .sessions
            .iter()
            .map(WorkerServices::binding)
            .eq(self.inventory.bindings())
        {
            return Err(MemoryStatus::Domain);
        }
        let ram = self.ram.as_ref().expect("live canonical service");
        let Some(frame) = ram.try_close_frame(physical)? else {
            return Ok(None);
        };

        for session in self.sessions.iter_mut() {
            let effect = HostEffect::RevokeRamCpuMappings(RamCpuMappingRevocation::new(&frame));
            let result = session
                .command(effect)
                .map_err(|_| ())
                .and_then(|command| self.transport.apply(self.local, command).map_err(|_| ()))
                .and_then(|receipt| session.accept(receipt).map_err(|_| ()));
            if result.is_err() {
                ram.fail();
                self.transport
                    .fail_stopped("held worker RAM CPU mapping revocation did not complete");
            }
        }
        Ok(Some(CpuMappingsRevokedFrame(frame)))
    }
    /// Complete coherent CPU memory on the BSP helper and every held
    /// source worker. Each owner must apply its real fence and return the exact
    /// session receipt before continuing. No DMA/DRAM/cache-power completion or
    /// architectural DSB retirement is implied by this CPU-only operation.
    pub fn fence_cpu_memory(&mut self) {
        if self.validate_current().is_err() {
            self.transport
                .fail_stopped("canonical RAM fence lost its actual hold");
        }
        self.local.memory_barrier();
        for session in self.sessions.iter_mut() {
            let command = match session.command(HostEffect::MemoryBarrier) {
                Ok(command) => command,
                Err(_) => self
                    .transport
                    .fail_stopped("canonical RAM fence command refused"),
            };
            let receipt = match self.transport.apply(self.local, command) {
                Ok(receipt) => receipt,
                Err(_) => self
                    .transport
                    .fail_stopped("actual RAM worker fence refused"),
            };
            if session.accept(receipt).is_err() {
                self.transport
                    .fail_stopped("actual RAM worker fence receipt mismatched");
            }
        }
    }

    fn require_resident_for_release(&mut self) {
        if self.memory().has_nonresident_frames() {
            self.memory().fail();
            self.transport
                .fail_stopped("nonresident RAM frame remains at worker release");
        }
    }
    /// End only RAM exclusion, keeping the SAME worker stop and open
    /// owner-local sessions for a synchronous controller/CPU transaction.
    /// Only engine continuation code can consume the returned stop permission.
    pub(in crate::machine::engine) fn finish_memory(mut self) -> publication::StoppedWorkers {
        self.require_resident_for_release();
        if self.validate_current().is_err() {
            self.transport
                .fail_stopped("canonical write lost its actual RAM hold");
        }
        self.fence_cpu_memory();
        drop(self.ram.take());
        self.held.take().expect("actual retained worker hold")
    }

    /// Call only after the real controller owner has settled its prepared
    /// request, including its ACK or a proved effect-free refusal. The source
    /// instruction's own architectural retirement remains the engine's job.
    /// No RAM/controller effect or ACK is manufactured by this method.
    pub fn finish(mut self) {
        self.require_resident_for_release();
        if self.validate_current().is_err() {
            self.transport
                .fail_stopped("canonical RAM service lost its live owner");
        }
        self.fence_cpu_memory();
        // End byte/mapping exclusion only after controller settlement, before
        // allowing the owners to leave their same-generation hold.
        drop(self.ram.take());
        *self.selected = None;
        for session in self.sessions.drain(..) {
            if self.inventory.close_services(session).is_err() {
                self.transport
                    .fail_stopped("canonical RAM service has unfinished Host effects");
            }
        }
        let held = self.held.take().expect("actual held cohort");
        self.transport.release(self.local, self.inventory, held);
    }
}
impl<H: Host, T: NativeTransport> Drop for HeldRamService<'_, H, T> {
    fn drop(&mut self) {
        if let Some(ram) = &self.ram {
            ram.fail();
        }
        // Sessions stay open in the inventory on abandoned settlement. Neither
        // Drop nor a copied CPU/receipt may authorize redispatch.
    }
}
impl<H: Host, T: NativeTransport, D: Devices> Machine<ServiceHost<H, T>, D> {
    /// Recover/hold every assigned native worker, complete synchronous modeled
    /// copyout through the SAME Host/RAM, fence each physical owner, then
    /// exclude native loans and every canonical physical writer. No raw serial
    /// RAM lane or second memory owner is admitted.
    ///
    /// # Safety
    /// The registered reservation and full shared native execution contract
    /// remain valid. All device work is synchronous through Devices::transfer;
    /// asynchronous/direct DMA, firmware/SMI and every other producer outside
    /// the held workers and completed synchronous transfer are excluded. This
    /// includes independently advancing queued/background jobs which know this
    /// registered RAM's Arc. Paused synchronous device requests may remain
    /// only when the exclusive Devices owner excludes their
    /// advancement through the whole service. Writer gates or transfer(Ok) do
    /// not establish an empty I/O command queue. There may be no reentrant
    /// monitor/RAM write, CPU settlement or
    /// interrupt/migration while the returned all-writer gates are held. The
    /// prepared controller/request must be held through its effect,
    /// faults and ACK before finish; unavailable DRAM/gating effects still
    /// refuse. CPU visibility is not a generic cache/DRAM/DMA drain.
    pub unsafe fn prepare_native_ram_service(&mut self) -> Result<HeldRamService<'_, H, T>, Stop> {
        // SAFETY: same owner and provider contract; the caller does not
        // need the disjoint Devices borrow for this service.
        unsafe { self.prepare_native_ram_service_parts() }.map(|(_, service)| service)
    }

    /// The same live service, split from its Devices receiver so the
    /// prepared controller/request can remain borrowed through commit/ACK.
    ///
    /// # Safety
    /// All prepare_native_ram_service requirements apply. The Devices borrow
    /// may not initiate another RAM/DMA/monitor access while the gates are held.
    pub unsafe fn prepare_native_ram_service_parts(
        &mut self,
    ) -> Result<(&mut D, HeldRamService<'_, H, T>), Stop> {
        let bindings: Vec<_> = self
            .native_publication
            .as_ref()
            .map(|inventory| inventory.bindings().collect())
            .unwrap_or_default();
        let owner_matches = matches!(self.peers.get(self.current), Some(Peer::Running))
            && !bindings.is_empty()
            && bindings.iter().all(|binding| {
                matches!(
                    self.peers.get(binding.pe),
                    Some(Peer::Running | Peer::Parked(_) | Peer::Off)
                )
            })
            && self.peers.iter().enumerate().all(|(pe, peer)| {
                let cpu = match peer {
                    Peer::Running if pe == self.current => &self.cpu,
                    Peer::Parked(parked) => parked.cpu(),
                    Peer::Off => return true,
                    _ => return false,
                };
                if !bindings.iter().any(|binding| binding.pe == pe) {
                    return false;
                }
                cpu.canonical_memory.as_ref().is_some_and(|memory| {
                    memory.pe == pe
                        && Arc::ptr_eq(&memory.ram, &self.host.ram)
                        && Arc::ptr_eq(&memory.machine, &self.native_memory_identity)
                })
            });
        if !owner_matches {
            return Err(self.unsupported(RequirementKind::Native(
                "RAM service requires actual canonical PEs and their assigned worker inventory",
            )));
        }
        let held = self.hold_native_services()?;
        let now = self.now();
        self.device_transfer(now).map_err(|error| {
            self.unsupported(RequirementKind::DeviceAdvance(alloc::format!("{error:?}")))
        })?;
        let bindings: Vec<_> = self
            .native_publication
            .as_ref()
            .expect("held inventory")
            .bindings()
            .collect();
        for binding in bindings {
            if !self.host.select_native_service(binding.worker) {
                self.host
                    .transport
                    .fail_stopped("canonical RAM service lacks actual Host owner");
            }
            self.host.memory_barrier();
        }
        let unavailable = self.unsupported(RequirementKind::Native(
            "canonical RAM service failed before controller effect",
        ));
        let service = self
            .host
            .borrow_ram_service(
                self.native_publication.as_mut().expect("held inventory"),
                held,
            )
            .map_err(|_| unavailable)?;
        Ok((&mut self.devices, service))
    }
}

impl<H: Host, T: NativeTransport> ServiceHost<H, T> {
    /// Consume the SAME stopped token; never close/reopen a generation
    /// between recovering a source frame and performing its device service.
    pub(in crate::machine::engine) fn borrow_ram_service<'a>(
        &'a mut self,
        inventory: &'a mut publication::WorkerPublications,
        held: publication::StoppedWorkers,
    ) -> Result<HeldRamService<'a, H, T>, stela_compiler::context::memory::MemoryStatus> {
        use stela_compiler::context::memory::MemoryStatus;
        inventory
            .require_all_held(&held)
            .map_err(|_| MemoryStatus::Domain)?;
        let memory = self.ram.service()?;
        let mut service = HeldRamService {
            ram: Some(memory),
            local: &mut self.local,
            transport: &mut self.transport,
            inventory,
            sessions: &mut self.services,
            selected: &mut self.selected,
            held: Some(held),
        };
        service.fence_cpu_memory();
        Ok(service)
    }
}

impl<H: Host, T: NativeTransport> crate::machine::engine::device_read::sealed::Sealed
    for HeldRamService<'_, H, T>
{
}
impl<H: Host, T: NativeTransport> crate::machine::engine::CanonicalDeviceMemory
    for HeldRamService<'_, H, T>
{
    fn memory(&self) -> &CanonicalRamService<'_> {
        HeldRamService::memory(self)
    }
    fn fence_cpu_memory(&mut self) {
        HeldRamService::fence_cpu_memory(self);
    }
}
