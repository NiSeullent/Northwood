//! Source-bound system accesses over the canonical physical RAM owner.
//! This is a checked execution service, not an additional VM or RAM image.
mod controls;
mod exclusive;
mod host;
mod lifecycle;
mod native_run;
mod policy;
mod scheduler;
mod terminal;
pub(super) use terminal::ExecutionDomain;
#[cfg(feature = "native-profile")]
mod source_profile;
pub use host::{
    CpuMappingsRevokedFrame, HeldRamService, NativeTransport, ServiceHost, ServiceTransport,
};
#[cfg(feature = "native-profile")]
pub use source_profile::Statistics as NativeSourceStatistics;

use super::{Devices, Host, Machine, mmu};
use crate::memory::{RegisteredSystemRam, domain::MemoryThread};
use alloc::{sync::Arc, vec::Vec};
use controls::Controls;
pub use native_run::{
    ArtifactCacheStats, Cohort as NativeCohort, Identity as NativeRunIdentity,
    MAX_NATIVE_EPOCH_BODIES, NativeEpochDeadline, NativeEpochEnd, NativeEpochStatistics, NativeRun,
    Observation as NativeRunObservation, ReturnedRun, dispatch as dispatch_native_run,
    recover as recover_native_run,
};
pub(in crate::machine::engine) use native_run::{
    RecoveredDeviceAccess, RecoveredRead, RecoveredWrite,
};
#[cfg(feature = "native-profile")]
pub use policy::DataTlbStatistics as NativeDataTlbStatistics;
use policy::Policy;
pub use policy::{Failure as SharedMemoryFailure, Refusal as SharedMemoryRefusal};
use stela_compiler::context::memory::MemoryStatus;

/// One PE's persistent global reservation and effective source-memory service.
/// Machine refreshes controls while it still owns that PE; no caller can
/// substitute raw registers, a physical-address identity map or a ready bit.
pub struct SharedMemory {
    memory: MemoryThread<Policy>,
    ram: Arc<RegisteredSystemRam>,
    pe: usize,
    machine: Arc<ExecutionDomain>,
    event: crate::system::events::EventEndpoint,
    artifacts: native_run::ArtifactCache,
    #[cfg(feature = "native-profile")]
    epochs: NativeEpochStatistics,
    #[cfg(feature = "native-profile")]
    source: source_profile::Profile,
}
impl core::fmt::Debug for SharedMemory {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SharedMemory")
            .field("pe", &self.pe)
            .finish_non_exhaustive()
    }
}
impl SharedMemory {
    pub(crate) fn prepare_clear_reservation(
        &mut self,
    ) -> Result<crate::memory::domain::PreparedLocalClear<'_>, MemoryStatus> {
        self.memory.prepare_clear_reservation()
    }
    /// Tighten the per-PE kept compiler-artifact limit. This cannot raise
    /// the named default or a previously tightened bound; zero disables reuse.
    pub fn tighten_artifact_cache_budget(&mut self, bytes: usize) {
        self.artifacts.tighten(bytes);
    }
    pub fn native_artifact_stats(&self) -> ArtifactCacheStats {
        self.artifacts.stats()
    }
    pub fn processor(&self) -> usize {
        self.pe
    }
    pub(in crate::machine::engine) fn ram_owner(&self) -> &Arc<RegisteredSystemRam> {
        &self.ram
    }
    pub(super) fn prepare_reset_peer(
        &self,
        cpu: &crate::machine::Cpu,
        pe: usize,
        name: &'static str,
        missing: &[crate::machine::system::Register],
    ) -> Result<Self, MemoryStatus> {
        if cpu.canonical_memory.is_some() || cpu.monitor.is_some() {
            return Err(MemoryStatus::Unsupported);
        }
        let controls = Controls::capture(cpu, name, missing)?;
        let mut memory = self
            .ram
            .domain()
            .thread(Policy::new(controls, Arc::clone(&self.ram)));
        let event = self
            .event
            .domain()
            .register(cpu.event, None)
            .map_err(|_| MemoryStatus::Domain)?;
        memory.bind_events(&event.producer())?;
        Ok(Self {
            memory,
            ram: Arc::clone(&self.ram),
            pe,
            machine: Arc::clone(&self.machine),
            event,
            artifacts: self.artifacts.empty_like(),
            #[cfg(feature = "native-profile")]
            epochs: NativeEpochStatistics::default(),
            #[cfg(feature = "native-profile")]
            source: source_profile::Profile::default(),
        })
    }
    pub fn event_endpoint(&self) -> &crate::system::events::EventEndpoint {
        &self.event
    }
    pub(in crate::machine::engine) fn current_access_source<const WRITE: bool>(
        &self,
        cpu: &crate::machine::Cpu,
        profile: &crate::machine::CpuProfile,
        missing: &[crate::machine::system::Register],
        recovered: &native_run::RecoveredAccess<WRITE>,
        held: &crate::memory::CanonicalRamService<'_>,
    ) -> Result<(), MemoryStatus> {
        let (binding, _, pc, _) = recovered.source();
        let (word, _, _, _, _) = recovered.access().ok_or(MemoryStatus::Unsupported)?;
        let controls_hash = recovered.controls().ok_or(MemoryStatus::Unsupported)?;
        if binding.pe != self.pe
            || !recovered.machine_matches(&self.machine)
            || !held.belongs_to(&self.ram)
        {
            return Err(MemoryStatus::Domain);
        }
        held.validate_current()?;
        let controls = Controls::capture(cpu, profile.name, missing)?;
        if controls.identity() != controls_hash {
            return Err(MemoryStatus::Unsupported);
        }
        let mut actual = [0; 4];
        self.fetch(&controls, pc, &mut actual)?;
        if u32::from_le_bytes(actual) != word {
            return Err(MemoryStatus::Unsupported);
        }
        Ok(())
    }

    pub(in crate::machine::engine) fn current_device_access(
        &self,
        cpu: &crate::machine::Cpu,
        profile: &crate::machine::CpuProfile,
        missing: &[crate::machine::system::Register],
        request: crate::memory::domain::DataRequest,
        quantity_bytes: u8,
        physical: u64,
    ) -> Result<(), MemoryStatus> {
        use crate::memory::{
            backing::AccessKind,
            domain::{DataPart, DataPolicy},
        };
        let controls = Controls::capture(cpu, profile.name, missing)?;
        let mut policy = Policy::new(controls, Arc::clone(&self.ram));
        policy.validate_data_request(request)?;
        let width = usize::from(quantity_bytes);
        for offset in (0..request.bytes).step_by(width) {
            policy.failure = None;
            let address = request.address + offset as u64;
            let result = policy.translate_part(
                request,
                DataPart {
                    address,
                    bytes: width,
                    kind: if request.flags & stela_compiler::context::memory::flags::WRITE != 0 {
                        AccessKind::Write
                    } else {
                        AccessKind::Read
                    },
                },
            );
            let write = request.flags & stela_compiler::context::memory::flags::WRITE != 0;
            let reached = match policy.failure {
                Some(policy::Failure::DeviceRead {
                    physical, bytes, ..
                }) if !write => Some((physical, bytes)),
                Some(policy::Failure::DeviceWrite {
                    physical, bytes, ..
                }) if write => Some((physical, bytes)),
                _ => None,
            };
            if result != Err((MemoryStatus::Unsupported, address))
                || reached != Some((physical + offset as u64, width))
            {
                return Err(MemoryStatus::Unsupported);
            }
        }
        Ok(())
    }

    fn fetch(&self, controls: &Controls, pc: u64, output: &mut [u8]) -> Result<(), MemoryStatus> {
        let mut tables = policy::Tables::new(&self.ram);
        let result = self.fetch_with_tables(controls, pc, output, &mut tables);
        #[cfg(feature = "native-profile")]
        self.source.record(
            source_profile::Fetch::FullSource,
            1,
            tables.reads,
            output.len(),
            result.is_ok(),
        );
        result
    }

    fn fetch_with_tables(
        &self,
        controls: &Controls,
        pc: u64,
        output: &mut [u8],
        tables: &mut policy::Tables<'_>,
    ) -> Result<(), MemoryStatus> {
        let outcome = mmu::translate(
            &controls.translation,
            pc,
            mmu::Access::Execute,
            controls.requester,
            tables,
        );
        if let Some(status) = tables.failure {
            return Err(status);
        }
        let mmu::Outcome::Mapped(mapped) = outcome else {
            return Err(MemoryStatus::Unsupported);
        };
        if mapped.memory != mmu::MemoryType::Normal
            || mapped.attributes == mmu::attributes::TAGGED
            || output.len() as u64 > mapped.size - (mapped.physical & (mapped.size - 1))
        {
            return Err(MemoryStatus::Unsupported);
        }
        // SAFETY: private source staging, disjoint from RAM/frames. Code
        // publication is held stopped by the outer scheduler through preparation.
        unsafe { self.ram.read_physical(mapped.physical, output) }
    }
}

impl<H: Host, D: Devices> Machine<H, D> {
    /// Execute-source observations for the current PE. This includes
    /// entry checks and artifact byte validation; data counters are separate.
    #[cfg(feature = "native-profile")]
    pub fn native_source_statistics(&self) -> Option<NativeSourceStatistics> {
        self.cpu
            .canonical_memory
            .as_ref()
            .map(|memory| memory.source.statistics())
    }

    /// Cumulative epoch entries for the current canonical PE.
    #[cfg(feature = "native-profile")]
    pub fn native_epoch_statistics(&self) -> Option<NativeEpochStatistics> {
        self.cpu
            .canonical_memory
            .as_ref()
            .map(|memory| memory.epochs)
    }

    /// Cumulative observations for this current PE, not the full cohort.
    #[cfg(feature = "native-profile")]
    pub fn native_data_tlb_statistics(&mut self) -> Option<NativeDataTlbStatistics> {
        Some(
            self.cpu
                .canonical_memory
                .as_mut()?
                .memory
                .policy_mut()
                .data_tlb_statistics(),
        )
    }

    /// Bound every currently instantiated canonical PE's compiler artifacts.
    /// Reset/start owners inherit the tightened bound from their real source.
    /// No executing code or architectural state is cached or invalidated here.
    pub fn tighten_native_artifact_cache_budget(
        &mut self,
        per_pe_bytes: usize,
    ) -> Result<(), MemoryStatus> {
        for (pe, peer) in self.peers.iter().enumerate() {
            let cpu = match peer {
                super::processors::Peer::Running if pe == self.current => &self.cpu,
                super::processors::Peer::Parked(parked) => parked.cpu(),
                super::processors::Peer::Off => continue,
                _ => return Err(MemoryStatus::Domain),
            };
            let memory = cpu
                .canonical_memory
                .as_ref()
                .ok_or(MemoryStatus::Unsupported)?;
            if memory.pe != pe || !Arc::ptr_eq(&memory.machine, &self.native_memory_identity) {
                return Err(MemoryStatus::Domain);
            }
        }
        for (pe, peer) in self.peers.iter_mut().enumerate() {
            let cpu = match peer {
                super::processors::Peer::Running if pe == self.current => &mut self.cpu,
                super::processors::Peer::Parked(parked) => parked.cpu_mut(),
                super::processors::Peer::Off => continue,
                _ => unreachable!("validated actual PE roster"),
            };
            cpu.canonical_memory
                .as_mut()
                .expect("validated canonical owner")
                .tighten_artifact_cache_budget(per_pe_bytes);
        }
        Ok(())
    }

    /// Move every active PE's monitor/event permission into its Cpu.
    /// Before worker binding, exclusive Machine ownership excludes native loans;
    /// afterwards the complete roster must be recovered/invalidated and held in
    /// one stop generation. No prior monitor is reconstructed from copied bytes.
    pub fn install_shared_memory_for_processors(
        &mut self,
        ram: Arc<RegisteredSystemRam>,
        events: &crate::system::events::EventDomain,
    ) -> Result<(), MemoryStatus> {
        self.install_shared_memory(ram, events, false)
    }

    /// Convert exactly the instantiated Running/Parked Cpu owners. Peer::Off
    /// has no Cpu/frame permission and remains unbound; starts/reset/rebind must
    /// refuse until a real canonical owner and physical worker can be joined.
    pub fn install_shared_memory_for_active_processors(
        &mut self,
        ram: Arc<RegisteredSystemRam>,
        events: &crate::system::events::EventDomain,
    ) -> Result<(), MemoryStatus> {
        self.install_shared_memory(ram, events, true)
    }

    fn install_shared_memory(
        &mut self,
        ram: Arc<RegisteredSystemRam>,
        events: &crate::system::events::EventDomain,
        active_only: bool,
    ) -> Result<(), MemoryStatus> {
        if !matches!(
            self.peers.get(self.current),
            Some(super::processors::Peer::Running)
        ) {
            return Err(MemoryStatus::Unsupported);
        }
        if ram.reservation_granule() != crate::machine::cpu::RESERVATION_GRANULE {
            return Err(MemoryStatus::Unsupported);
        }
        if let Some(publication) = &self.native_publication {
            let members: Vec<_> = self
                .peers
                .iter()
                .enumerate()
                .filter_map(|(pe, peer)| {
                    (!active_only || !matches!(peer, super::processors::Peer::Off)).then_some(pe)
                })
                .collect();
            if active_only {
                publication.require_held_members(&members)
            } else {
                publication.require_stopped(self.peers.len(), &members)
            }
            .map_err(|_| MemoryStatus::Unsupported)?;
        }
        let mut prepared = Vec::with_capacity(self.peers.len());
        for (pe, peer) in self.peers.iter().enumerate() {
            let cpu = match peer {
                super::processors::Peer::Running
                    if pe == self.current && self.waiting.is_none() =>
                {
                    &self.cpu
                }
                super::processors::Peer::Parked(parked) if parked.native_ready() => parked.cpu(),
                super::processors::Peer::Off if active_only => continue,
                _ => return Err(MemoryStatus::Unsupported),
            };
            if cpu.canonical_memory.is_some() || cpu.monitor.is_some() {
                return Err(MemoryStatus::Unsupported);
            }
            let controls = Controls::capture(cpu, self.profile.name, &self.profile_missing)?;
            let mut memory = ram.domain().thread(Policy::new(controls, Arc::clone(&ram)));
            let event = events
                .register(cpu.event, None)
                .map_err(|_| MemoryStatus::Domain)?;
            memory.bind_events(&event.producer())?;
            prepared.push((
                pe,
                SharedMemory {
                    memory,
                    ram: Arc::clone(&ram),
                    pe,
                    event,
                    machine: Arc::clone(&self.native_memory_identity),
                    artifacts: native_run::ArtifactCache::default(),
                    #[cfg(feature = "native-profile")]
                    epochs: NativeEpochStatistics::default(),
                    #[cfg(feature = "native-profile")]
                    source: source_profile::Profile::default(),
                },
            ));
        }
        // No fallible step after the first move. The old scalar event storage
        // ceases to be authoritative; all services use the moved endpoint.
        for (pe, memory) in prepared {
            let cpu = match &mut self.peers[pe] {
                super::processors::Peer::Running => &mut self.cpu,
                super::processors::Peer::Parked(parked) => parked.cpu_mut(),
                super::processors::Peer::Off | super::processors::Peer::Remote => {
                    unreachable!("preflight admitted active PE")
                }
            };
            cpu.event = false;
            cpu.canonical_memory = Some(memory);
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests;
