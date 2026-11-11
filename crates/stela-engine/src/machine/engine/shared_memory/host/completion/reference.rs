// SPDX-License-Identifier: GPL-2.0-or-later
//! Hosted fixture for transactions that require a real held RAM capability.
use super::CpuMappingsRevokedFrame;
use crate::{
    machine::{
        Host, InstallError, NativeCode, NativeExit, Regime,
        engine::{
            NativeRun, NativeRunObservation, NativeTransport, ServiceHost, ServiceTransport,
            publication::{
                HostEffectCommand, HostEffectReceipt, RamCpuMappingRevocation, StoppedWorkers,
                WithPayload, WorkerBinding, WorkerId, WorkerOwner, WorkerPublications,
            },
        },
    },
    memory::RegisteredSystemRam,
};
use alloc::{sync::Arc, vec, vec::Vec};
use stela_compiler::{GuestState, context::memory::MemoryStatus};

const BINDING: WorkerBinding = WorkerBinding {
    pe: 0,
    worker: WorkerId(0),
};

impl CpuMappingsRevokedFrame<'_> {
    /// Run a transaction on hosted fixture RAM through one model worker's
    /// real publication, invalidation, service and release protocol. The
    /// model has no native execution or external DMA.
    ///
    /// Release requires the callback to restore any frame it made
    /// nonresident. Abandonment fails the fixture RAM as in the real service.
    ///
    /// # Safety
    /// The caller must provide fixture RAM with no real worker, direct alias
    /// or device access outside its registered memory interface.
    pub unsafe fn with_reference_revoked_frame<R>(
        ram: Arc<RegisteredSystemRam>,
        physical: u64,
        run: impl for<'a> FnOnce(CpuMappingsRevokedFrame<'a>) -> R,
    ) -> Result<R, MemoryStatus> {
        let (mut inventory, owners) = WorkerPublications::new(&[BINDING]).unwrap();
        let [mut owner] = owners.try_into().expect("one reference owner");
        let mut local = ReferenceHost {
            ram: Arc::clone(&ram),
            alias: Some(physical),
        };
        inventory
            .accept_initial(owner.publish_initial(&mut local).unwrap())
            .unwrap();

        let mut host = ServiceHost::new(local, ReferenceTransport(owner), ram);
        let held = host.transport.hold_all(&mut host.local, &mut inventory);
        let session = inventory.open_services(&held, BINDING.pe).unwrap();
        host.begin_services(vec![session]).unwrap();
        let mut service = host.borrow_ram_service(&mut inventory, held)?;
        let result = service
            .try_revoke_frame_cpu_mappings(physical)
            .and_then(|frame| frame.ok_or(MemoryStatus::Domain))
            .map(run);
        service.finish();
        result
    }
}

struct ReferenceHost {
    ram: Arc<RegisteredSystemRam>,
    alias: Option<u64>,
}

impl Host for ReferenceHost {
    fn native_worker(&self) -> Option<WorkerId> {
        Some(BINDING.worker)
    }

    fn revoke_ram_cpu_mappings(&mut self, request: &RamCpuMappingRevocation) -> bool {
        if !request.belongs_to(&self.ram) || self.alias != Some(request.physical()) {
            return false;
        }
        self.alias = None;
        true
    }

    fn read_ram(&mut self, _: u64, _: &mut [u8]) -> bool {
        unreachable!()
    }
    fn write_ram(&mut self, _: u64, _: &[u8]) -> bool {
        unreachable!()
    }
    fn install(&mut self, _: &[u8]) -> Result<NativeCode, InstallError> {
        unreachable!()
    }
    fn patch(&mut self, _: NativeCode, _: &[u8]) {
        unreachable!()
    }
    fn discard_code(&mut self) {
        unreachable!()
    }
    unsafe fn execute(&mut self, _: NativeCode, _: &mut GuestState) -> NativeExit {
        unreachable!()
    }
    fn set_regime(&mut self, _: &Regime) {
        unreachable!()
    }
    fn invalidate_translations(&mut self) {}
    fn counter(&mut self) -> u64 {
        unreachable!()
    }
    fn arm_timer(&mut self, _: Option<u64>) {
        unreachable!()
    }
    fn wait_for_interrupt(&mut self) {
        unreachable!()
    }
}

struct ReferenceTransport(WorkerOwner);

impl ServiceTransport for ReferenceTransport {
    fn apply(
        &mut self,
        local: &mut dyn Host,
        command: HostEffectCommand,
    ) -> Result<HostEffectReceipt, WithPayload<HostEffectCommand>> {
        self.0.apply_host_effect(command, local)
    }

    fn fail_stopped(&mut self, reason: &'static str) -> ! {
        panic!("{reason}");
    }
}

// SAFETY: the sole model worker starts initialized and cannot dispatch.
// Its actual owner handles every stop, effect and release for this fixture.
unsafe impl NativeTransport for ReferenceTransport {
    fn initialize(&mut self, _: &mut dyn Host, _: &mut WorkerPublications, _: Vec<WorkerOwner>) {
        unreachable!()
    }

    fn execute<'a>(
        &mut self,
        _: &mut dyn Host,
        _: &mut WorkerPublications,
        _: Vec<NativeRun<'a>>,
    ) -> Vec<NativeRunObservation> {
        unreachable!()
    }

    fn hold_all(
        &mut self,
        local: &mut dyn Host,
        inventory: &mut WorkerPublications,
    ) -> StoppedWorkers {
        let request = inventory.request_stop(&[BINDING.pe]).unwrap();
        let [command] = inventory
            .native_stop_commands(&request)
            .unwrap()
            .try_into()
            .expect("one reference stop");
        inventory
            .accept_invalidation(self.0.invalidate(command, local).unwrap())
            .unwrap();
        inventory.hold_stop(request).unwrap()
    }

    fn release(
        &mut self,
        _: &mut dyn Host,
        inventory: &mut WorkerPublications,
        held: StoppedWorkers,
    ) {
        let [command] = inventory
            .release(held)
            .unwrap()
            .try_into()
            .expect("one reference release");
        inventory
            .accept_release(self.0.release(command).unwrap())
            .unwrap();
    }
}
