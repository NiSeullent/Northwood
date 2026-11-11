//! BSP control-plane Host routing during a true all-worker stop. Only the
//! closed native effects cross cores; the Machine and Cpu never do.
mod completion;
pub use completion::{CpuMappingsRevokedFrame, HeldRamService};

use super::super::publication;
use super::*;
use crate::machine::{
    InstallError, NativeCode, NativeExit, Regime,
    engine::{DeviceSiteRam, NativeEntryError},
    mmu,
};
use publication::{
    HostEffect, HostEffectCommand, HostEffectReceipt, WithPayload, WorkerId, WorkerServices,
};
use stela_compiler::{
    GuestState,
    context::{ExecutionContext, memory::MemoryStatus},
};

/// Synchronous transport to the already-held native owner. Failure must
/// stop before any later source/controller effect or ACK can occur.
pub trait ServiceTransport {
    fn apply(
        &mut self,
        local: &mut dyn Host,
        command: HostEffectCommand,
    ) -> Result<HostEffectReceipt, WithPayload<HostEffectCommand>>;
    fn fail_stopped(&mut self, reason: &'static str) -> !;
}

/// Empty allocation storage only; entries are rebuilt from inventory
/// and recovered frames within each round. No entry carries owner permission.
#[derive(Default)]
pub(super) struct RoundScratch {
    pub(super) bindings: Vec<publication::WorkerBinding>,
    pub(super) expected: Vec<usize>,
    pub(super) seen: Vec<usize>,
}
impl RoundScratch {
    pub(super) fn clear(&mut self) {
        self.bindings.clear();
        self.expected.clear();
        self.seen.clear();
    }
}

/// The one existing Machine Host interface, with scoped physical-owner routing.
/// Canonical RAM access stays on its all-writer owner. The local Host
/// remains the BSP's Host, including native entry and diagnostic reports.
pub struct ServiceHost<H, T> {
    pub(super) local: H,
    pub(super) transport: T,
    ram: Arc<RegisteredSystemRam>,
    services: Vec<WorkerServices>,
    selected: Option<usize>,
    pub(super) round_scratch: RoundScratch,
    pub(super) retained_hold: Option<publication::StoppedWorkers>,
}
/// Existing fixed-owner execution/stop transport, keeping the frame
/// loans and generation-bound owner tokens. No metadata-only completion exists.
///
/// # Safety
/// Every submitted frame must be joined and recovered before return,
/// including error paths. Stop/release operate on all assigned owners;
/// a missing native return or effect must fail stopped without freeing loans.
/// `hold_all` must consume each owner's `WorkerOwner::invalidate`
/// receipt, including its CPU memory barrier, before returning the held cohort.
/// Opening source-service sessions does not issue another owner fence.
pub unsafe trait NativeTransport: ServiceTransport {
    fn initialize(
        &mut self,
        local: &mut dyn Host,
        inventory: &mut publication::WorkerPublications,
        owners: Vec<publication::WorkerOwner>,
    );
    fn execute<'a>(
        &mut self,
        local: &mut dyn Host,
        inventory: &mut publication::WorkerPublications,
        frames: Vec<super::NativeRun<'a>>,
    ) -> Vec<super::NativeRunObservation>;
    /// Execute prepared frames while keeping the idle-owner holds.
    /// The fallback preserves the complete original stop/release protocol;
    /// physical transports may selectively release the exact frame roster and
    /// rejoin their real returns under the stopped request.
    fn execute_held<'a>(
        &mut self,
        local: &mut dyn Host,
        inventory: &mut publication::WorkerPublications,
        frames: Vec<super::NativeRun<'a>>,
        held: publication::StoppedWorkers,
    ) -> (
        Vec<super::NativeRunObservation>,
        publication::StoppedWorkers,
    ) {
        self.release(local, inventory, held);
        let reports = self.execute(local, inventory, frames);
        let held = self.hold_all(local, inventory);
        (reports, held)
    }
    fn hold_all(
        &mut self,
        local: &mut dyn Host,
        inventory: &mut publication::WorkerPublications,
    ) -> publication::StoppedWorkers;
    fn release(
        &mut self,
        local: &mut dyn Host,
        inventory: &mut publication::WorkerPublications,
        held: publication::StoppedWorkers,
    );
}
impl<H: Host, T: ServiceTransport> ServiceHost<H, T> {
    pub fn new(local: H, transport: T, ram: Arc<RegisteredSystemRam>) -> Self {
        Self {
            local,
            transport,
            ram,
            services: Vec::new(),
            selected: None,
            round_scratch: RoundScratch::default(),
            retained_hold: None,
        }
    }
    /// Each linear session was opened from the exact currently-held cohort.
    /// A failed or dropped session leaves publication release refused.
    pub fn begin_services(
        &mut self,
        services: Vec<WorkerServices>,
    ) -> Result<(), Vec<WorkerServices>> {
        if !self.services.is_empty() || self.selected.is_some() || services.is_empty() {
            return Err(services);
        }
        self.services = services;
        Ok(())
    }
    /// Return every session for publication close before cohort release.
    pub fn end_services(&mut self) -> Vec<WorkerServices> {
        self.selected = None;
        core::mem::take(&mut self.services)
    }
    /// Transfer only empty storage. Every next session still comes from the
    /// freshly validated held inventory, never from kept permission.
    pub(super) fn take_empty_service_buffer(&mut self) -> Vec<WorkerServices> {
        if !self.services.is_empty() || self.selected.is_some() {
            self.transport
                .fail_stopped("native service buffer still has active sessions");
        }
        core::mem::take(&mut self.services)
    }

    /// Call only after each drained session passed publication close. Keeping
    /// allocation capacity carries no session, receipt, frame or stop token.
    pub(super) fn retain_empty_service_buffer(&mut self, services: Vec<WorkerServices>) {
        if !services.is_empty() || !self.services.is_empty() || self.selected.is_some() {
            self.transport
                .fail_stopped("native service buffer is not empty after close");
        }
        self.services = services;
    }
    fn effect(&mut self, effect: HostEffect) {
        let Some(index) = self.selected else {
            self.transport
                .fail_stopped("native Host effect has no selected stopped worker");
        };
        let command = match self.services[index].command(effect) {
            Ok(command) => command,
            Err(_) => self
                .transport
                .fail_stopped("native Host effect session refused command"),
        };
        let receipt = match self.transport.apply(&mut self.local, command) {
            Ok(receipt) => receipt,
            Err(_) => self
                .transport
                .fail_stopped("actual held native Host refused its effect"),
        };
        if self.services[index].accept(receipt).is_err() {
            self.transport
                .fail_stopped("actual native Host effect receipt mismatched session");
        }
    }
    fn source_service(&self) -> bool {
        !self.services.is_empty()
    }
    fn native_entry(&mut self) {
        if self.source_service() {
            self.transport
                .fail_stopped("native code entry/patch during stopped source service");
        }
    }
}
impl<H: Host, T: ServiceTransport> Host for ServiceHost<H, T> {
    fn memory_barrier(&mut self) {
        // Source helpers may have performed canonical copyout on the BSP;
        // complete those stores as well as the source PE's native ones.
        self.local.memory_barrier();
        if self.source_service() {
            self.effect(HostEffect::MemoryBarrier);
        }
    }
    fn select_native_service(&mut self, worker: WorkerId) -> bool {
        let selected = self
            .services
            .iter()
            .position(|session| session.binding().worker == worker);
        self.selected = selected;
        selected.is_some()
    }
    fn native_worker(&self) -> Option<WorkerId> {
        self.local.native_worker()
    }
    fn native_cpu_lease(&self) -> Option<crate::NativeCpuLease> {
        self.local.native_cpu_lease()
    }
    #[cfg(feature = "native-profile")]
    fn native_timing(&mut self, phase: super::super::NativeTimingPhase) {
        self.local.native_timing(phase);
    }
    #[cfg(feature = "native-profile")]
    fn native_instruction_sample(&mut self, sample: super::super::NativeInstructionSample<'_>) {
        self.local.native_instruction_sample(sample);
    }
    #[cfg(feature = "native-profile")]
    fn native_timing_work_start(&mut self, work: super::super::NativeTimingWork) -> Option<u64> {
        self.local.native_timing_work_start(work)
    }
    #[cfg(feature = "native-profile")]
    fn native_timing_work_end(&mut self, work: super::super::NativeTimingWork, started: u64) {
        self.local.native_timing_work_end(work, started);
    }
    fn read_ram(&mut self, physical: u64, buffer: &mut [u8]) -> bool {
        // SAFETY: the canonical owner supplies mapping and all-reader
        // exclusion. No identity/raw-RAM fallback is admitted by this adapter.
        unsafe { self.ram.read_physical(physical, buffer).is_ok() }
    }
    fn device_site_ram(&self) -> Result<Option<DeviceSiteRam>, MemoryStatus> {
        let local = self.local.device_site_ram()?;
        if local
            .as_ref()
            .is_some_and(|site| !Arc::ptr_eq(&site.ram, &self.ram))
        {
            return Err(MemoryStatus::Domain);
        }
        Ok(Some(DeviceSiteRam {
            ram: Arc::clone(&self.ram),
            expected_generation: local.and_then(|site| site.expected_generation),
        }))
    }
    fn device_site_has_retained_ram(&self, page: u64) -> bool {
        self.local.device_site_has_retained_ram(page)
    }
    fn write_ram(&mut self, physical: u64, data: &[u8]) -> bool {
        // SAFETY: every source/host/alias writer uses this same physical domain.
        unsafe { self.ram.publish_physical(physical, data).is_ok() }
    }
    fn compare_exchange_ram(
        &mut self,
        physical: u64,
        expected: &mut [u8],
        new: &[u8],
    ) -> Option<bool> {
        self.local.compare_exchange_ram(physical, expected, new)
    }
    fn check_ram_range(
        &mut self,
        physical: u64,
        bytes: usize,
        write: bool,
    ) -> Result<(), stela_contracts::dma::DmaError> {
        self.ram
            .check_dma_range(physical, bytes, write)
            .map_err(|_| stela_contracts::dma::DmaError::Memory)
    }
    fn read_ram_progress(
        &mut self,
        physical: u64,
        buffer: &mut [u8],
    ) -> Result<(), stela_contracts::dma::TransferFault> {
        // SAFETY: private DMA staging, same canonical reader/owner contract.
        unsafe { self.ram.read_physical_progress(physical, buffer) }.map_err(|fault| {
            stela_contracts::dma::TransferFault {
                error: stela_contracts::dma::DmaError::Memory,
                completed: fault.completed,
            }
        })
    }
    fn write_ram_progress(
        &mut self,
        physical: u64,
        data: &[u8],
    ) -> Result<(), stela_contracts::dma::TransferFault> {
        // SAFETY: private staging, every writer participates in this same domain.
        unsafe { self.ram.publish_physical(physical, data) }.map_err(|fault| {
            stela_contracts::dma::TransferFault {
                error: stela_contracts::dma::DmaError::Memory,
                completed: fault.completed,
            }
        })
    }
    fn install(&mut self, code: &[u8]) -> Result<NativeCode, InstallError> {
        self.native_entry();
        self.local.install(code)
    }
    fn install_immutable(&mut self, code: &Arc<Vec<u8>>) -> Result<NativeCode, InstallError> {
        self.native_entry();
        self.local.install_immutable(code)
    }
    fn read_native(&mut self, code: NativeCode, bytes: &mut [u8]) -> bool {
        if self.source_service() {
            false
        } else {
            self.local.read_native(code, bytes)
        }
    }
    fn patch(&mut self, code: NativeCode, bytes: &[u8]) {
        self.native_entry();
        self.local.patch(code, bytes);
    }
    fn patch_instruction(
        &mut self,
        code: NativeCode,
        instruction: [u8; stela_compiler::x86_64::chain::PATCH_BYTES],
    ) {
        self.native_entry();
        self.local.patch_instruction(code, instruction);
    }
    fn discard_code(&mut self) {
        if self.source_service() {
            self.effect(HostEffect::DiscardCode);
        } else {
            self.local.discard_code();
        }
    }
    fn reclaim_code(&mut self) -> Option<core::ops::Range<usize>> {
        self.native_entry();
        self.local.reclaim_code()
    }
    fn code_window(&self) -> Option<super::super::CodeWindow> {
        self.local.code_window()
    }
    fn serialize_code(&mut self) {
        // Every install or patch in the board's window ends here.
        self.native_entry();
        self.local.serialize_code();
    }
    fn release_oldest_code(&mut self) -> Option<core::ops::Range<usize>> {
        self.native_entry();
        self.local.release_oldest_code()
    }
    fn memory_low(&self) -> bool {
        self.local.memory_low()
    }
    unsafe fn execute(&mut self, code: NativeCode, state: &mut GuestState) -> NativeExit {
        self.native_entry();
        // SAFETY: caller keeps the existing installed system block contract;
        // stopped source services are excluded above.
        unsafe { self.local.execute(code, state) }
    }
    unsafe fn try_execute(
        &mut self,
        code: NativeCode,
        state: &mut GuestState,
    ) -> Result<NativeExit, NativeEntryError> {
        self.native_entry();
        // SAFETY: caller keeps the installed block and stopped services.
        unsafe { self.local.try_execute(code, state) }
    }
    unsafe fn execute_context(
        &mut self,
        code: NativeCode,
        state: &mut GuestState,
        context: &mut ExecutionContext,
    ) -> Option<NativeExit> {
        self.native_entry();
        // SAFETY: caller keeps context/code/frame owners on this BSP.
        unsafe { self.local.execute_context(code, state, context) }
    }
    fn context_entry(&mut self, code: NativeCode) -> Option<usize> {
        if self.source_service() {
            None
        } else {
            self.local.context_entry(code)
        }
    }
    fn fetch_permitted(&mut self, virtual_address: u64) -> bool {
        self.local.fetch_permitted(virtual_address)
    }
    fn placement(&self) -> mmu::placement::Placement {
        self.local.placement()
    }
    fn translation_helper(&self) -> Option<u64> {
        None
    }
    fn set_regime(&mut self, regime: &Regime) {
        if self.source_service() {
            self.effect(HostEffect::Regime(alloc::boxed::Box::new(*regime)));
        } else {
            self.local.set_regime(regime);
        }
    }
    fn isolate_predictions(&mut self) {
        if self.source_service() {
            self.effect(HostEffect::IsolatePredictions);
        } else {
            self.local.isolate_predictions();
        }
    }
    fn speculation_barrier(&self) {
        self.local.speculation_barrier();
    }
    fn pan_in_flags(&self) -> bool {
        self.local.pan_in_flags()
    }
    fn code_invalidated(&mut self, page: Option<u64>) {
        if self.source_service() {
            self.effect(HostEffect::CodeInvalidated(page));
        } else {
            self.local.code_invalidated(page);
        }
    }
    fn invalidate_translations(&mut self) {
        if self.source_service() {
            self.effect(HostEffect::InvalidateTranslations);
        } else {
            self.local.invalidate_translations();
        }
    }
    fn invalidate_reach(&mut self, reach: mmu::tlbi::Reach, everywhere: mmu::tlbi::Everywhere) {
        if self.source_service() {
            self.effect(HostEffect::InvalidateReach(reach, everywhere));
        } else {
            self.local.invalidate_reach(reach, everywhere);
        }
    }
    fn counter(&mut self) -> u64 {
        self.local.counter()
    }
    fn relax(&mut self) {
        self.local.relax();
    }
    fn tsc_counter(&self) -> Option<(u64, u64)> {
        self.local.tsc_counter()
    }
    fn profile_ticks(&mut self) -> u64 {
        self.local.profile_ticks()
    }
    fn profile_boundary(&mut self, phase: crate::machine::engine::step_timing::Phase, now: u64) {
        self.local.profile_boundary(phase, now);
    }
    fn translations(&mut self) -> Option<&mut dyn crate::machine::Translations> {
        self.local.translations()
    }
    fn arm_timer(&mut self, deadline: Option<u64>) {
        if self.source_service() {
            self.effect(HostEffect::ArmTimer(deadline));
        } else {
            self.local.arm_timer(deadline);
        }
    }
    fn wait_for_interrupt(&mut self) {
        if self.source_service() {
            self.transport
                .fail_stopped("waiting inside stopped native source service");
        }
        self.local.wait_for_interrupt();
    }
    fn poll_interrupt(&mut self) -> bool {
        if self.source_service() {
            self.transport
                .fail_stopped("waiting inside stopped native source service");
        }
        self.local.poll_interrupt()
    }
    fn report(&mut self, event: &crate::machine::Event) {
        self.local.report(event);
    }
}

#[cfg(test)]
pub(in crate::machine::engine::shared_memory) mod tests {
    use super::*;
    use crate::machine::{
        Stop,
        engine::NoDevices,
        engine::publication::{WorkerBinding, WorkerOwner, WorkerPublications},
    };
    use anyhow::Result;
    use stela_contracts::refuse;

    #[cfg(feature = "std")]
    struct RamRevocationModel {
        ram: Arc<RegisteredSystemRam>,
        aliases: Vec<(u64, u64)>,
        requests: Vec<u64>,
        acknowledge: bool,
    }

    #[cfg(feature = "std")]
    std::thread_local! {
        static RAM_REVOCATION: core::cell::RefCell<Option<RamRevocationModel>> = const {
            core::cell::RefCell::new(None)
        };
    }

    #[cfg(feature = "std")]
    fn model_ram_revocation(
        ram: &Arc<RegisteredSystemRam>,
        aliases: &[(u64, u64)],
        acknowledge: bool,
    ) {
        RAM_REVOCATION.with(|model| {
            assert!(
                model
                    .replace(Some(RamRevocationModel {
                        ram: Arc::clone(ram),
                        aliases: aliases.to_vec(),
                        requests: Vec::new(),
                        acknowledge,
                    }))
                    .is_none()
            );
        });
    }

    pub(in crate::machine::engine::shared_memory) struct Local;
    impl Host for Local {
        fn read_native(&mut self, code: NativeCode, output: &mut [u8]) -> bool {
            // Initialized permanent fixture bytes owned by this local Host;
            // this model reader does not assert a physical native arena lease.
            const CODE: [u8; 5] = [0x90, 0x48, 0x89, 0xc0, 0xc3];
            let Some(offset) = code.0.checked_sub(0x8000) else {
                return false;
            };
            let Some(bytes) = offset
                .checked_add(output.len())
                .and_then(|end| CODE.get(offset..end))
            else {
                return false;
            };
            output.copy_from_slice(bytes);
            true
        }
        fn native_worker(&self) -> Option<WorkerId> {
            Some(WorkerId(17))
        }
        fn read_ram(&mut self, _: u64, _: &mut [u8]) -> bool {
            panic!("raw RAM fallback");
        }
        fn write_ram(&mut self, _: u64, _: &[u8]) -> bool {
            panic!("raw RAM fallback");
        }
        fn install(&mut self, _: &[u8]) -> Result<NativeCode, InstallError> {
            panic!("CPL0 refusal must precede native install");
        }
        fn patch(&mut self, _: NativeCode, _: &[u8]) {
            unreachable!()
        }
        fn discard_code(&mut self) {}
        unsafe fn execute(&mut self, _: NativeCode, _: &mut GuestState) -> NativeExit {
            unreachable!()
        }
        fn set_regime(&mut self, _: &Regime) {}
        fn invalidate_translations(&mut self) {}
        #[cfg(feature = "std")]
        fn revoke_ram_cpu_mappings(
            &mut self,
            request: &publication::RamCpuMappingRevocation,
        ) -> bool {
            use stela_compiler::context::memory::MemoryStatus;

            RAM_REVOCATION.with(|model| {
                let mut model = model.borrow_mut();
                let Some(model) = model.as_mut() else {
                    return false;
                };
                assert!(request.belongs_to(&model.ram));
                let physical = request.physical();
                model.requests.push(physical);
                let mut output = [0xa5; 8];
                // SAFETY: the sentinel output is private and the model retains the RAM owner.
                let read = unsafe { model.ram.read_physical(physical, &mut output) };
                assert_eq!(read, Err(MemoryStatus::Domain));
                assert_eq!(output, [0xa5; 8]);
                let range = stela_contracts::gate::MemoryRange {
                    address: physical + 4096,
                    bytes: 4096,
                };
                assert!(matches!(
                    model.ram.prepare_external_dma(&[range]),
                    Err(MemoryStatus::Domain)
                ));
                if model.acknowledge {
                    model
                        .aliases
                        .retain(|&(_, mapped)| !(physical..physical + 16 * 1024).contains(&mapped));
                }
                model.acknowledge
            })
        }
        fn counter(&mut self) -> u64 {
            1
        }
        fn profile_ticks(&mut self) -> u64 {
            7
        }
        fn arm_timer(&mut self, _: Option<u64>) {}
        fn wait_for_interrupt(&mut self) {
            unreachable!()
        }
    }

    #[test]
    fn the_wrapper_keeps_the_local_profile_clock() {
        let (.., ram) = super::super::policy::tests::fixture();
        let mut host = ServiceHost::new(Local, OneWorker(None), ram);
        assert_eq!(host.profile_ticks(), 7);
    }
    pub(in crate::machine::engine::shared_memory) struct OneWorker(
        pub(in crate::machine::engine::shared_memory) Option<WorkerOwner>,
    );
    impl OneWorker {
        fn owner(&mut self) -> &mut WorkerOwner {
            self.0.as_mut().unwrap()
        }
    }
    impl ServiceTransport for OneWorker {
        fn apply(
            &mut self,
            local: &mut dyn Host,
            command: HostEffectCommand,
        ) -> Result<HostEffectReceipt, WithPayload<HostEffectCommand>> {
            self.owner().apply_host_effect(command, local)
        }
        fn fail_stopped(&mut self, reason: &'static str) -> ! {
            panic!("{reason}");
        }
    }
    // SAFETY: this one-worker test transport consumes/recover every
    // sealed frame and uses the real publication stop/effect/release protocol.
    // It does not forge CPL0 admission or claim native code execution.
    unsafe impl NativeTransport for OneWorker {
        fn initialize(
            &mut self,
            local: &mut dyn Host,
            inventory: &mut WorkerPublications,
            mut owners: Vec<WorkerOwner>,
        ) {
            assert_eq!(owners.len(), 1);
            let mut owner = owners.pop().unwrap();
            inventory
                .accept_initial(owner.publish_initial(local).unwrap())
                .unwrap();
            self.0 = Some(owner);
        }
        fn execute<'a>(
            &mut self,
            local: &mut dyn Host,
            inventory: &mut WorkerPublications,
            frames: Vec<NativeRun<'a>>,
        ) -> Vec<NativeRunObservation> {
            frames
                .into_iter()
                .map(|frame| {
                    let loan = super::super::dispatch_native_run(inventory, frame)
                        .unwrap_or_else(|_| panic!("frame dispatch"));
                    let returned = self
                        .owner()
                        .execute(loan, local)
                        .unwrap_or_else(|_| panic!("frame return"));
                    super::super::recover_native_run(inventory, returned)
                        .unwrap_or_else(|_| panic!("actual recovery"))
                })
                .collect()
        }
        fn hold_all(
            &mut self,
            local: &mut dyn Host,
            inventory: &mut WorkerPublications,
        ) -> publication::StoppedWorkers {
            let request = inventory.request_stop(&[0]).unwrap();
            let command = inventory
                .native_stop_commands(&request)
                .unwrap()
                .pop()
                .unwrap();
            inventory
                .accept_invalidation(self.owner().invalidate(command, local).unwrap())
                .unwrap();
            inventory.hold_stop(request).unwrap()
        }
        fn release(
            &mut self,
            _: &mut dyn Host,
            inventory: &mut WorkerPublications,
            held: publication::StoppedWorkers,
        ) {
            let release = inventory.release(held).unwrap().pop().unwrap();
            inventory
                .accept_release(self.owner().release(release).unwrap())
                .unwrap();
        }
    }
    struct Copyout {
        calls: usize,
        fail: bool,
    }
    impl Devices for Copyout {
        fn read(
            &mut self,
            _: usize,
            _: u64,
            _: u8,
            _: u64,
        ) -> Result<u64, crate::machine::DeviceError> {
            Err(crate::machine::DeviceError::NoDevice)
        }
        fn write(
            &mut self,
            _: usize,
            _: u64,
            _: u8,
            _: u64,
            _: u64,
        ) -> Result<(), crate::machine::DeviceError> {
            Err(crate::machine::DeviceError::NoDevice)
        }
        fn lines(&mut self, _: usize, _: u64) -> crate::machine::engine::Lines {
            crate::machine::engine::Lines::default()
        }
        fn transfer(
            &mut self,
            _: usize,
            memory: &mut dyn crate::machine::engine::Memory,
            _: u64,
        ) -> Result<(), crate::machine::DeviceError> {
            self.calls += 1;
            assert!(memory.write(0x14080, &0x12345678u64.to_le_bytes()));
            if self.fail {
                Err(crate::machine::DeviceError::TransferRefused(refuse!(
                    "retained modeled prefix"
                )))
            } else {
                Ok(())
            }
        }
    }
    fn memory_service_machine(
        fail: bool,
    ) -> (
        Machine<ServiceHost<Local, OneWorker>, Copyout>,
        Arc<RegisteredSystemRam>,
    ) {
        memory_service_machine_slots(fail, 1, false)
    }

    fn memory_service_machine_slots(
        fail: bool,
        count: usize,
        track_frames: bool,
    ) -> (
        Machine<ServiceHost<Local, OneWorker>, Copyout>,
        Arc<RegisteredSystemRam>,
    ) {
        let (cpu, mut profile, missing, mut ram) = super::super::policy::tests::fixture();
        if track_frames {
            Arc::get_mut(&mut ram)
                .unwrap()
                .enable_descriptor_frame_leases()
                .unwrap();
        }
        if count > 1 {
            // Synthetic modeled slots, not a claim about a physical Apple SKU.
            profile.processors = (0..count)
                .map(|pe| stela_contracts::identity::Identity {
                    midr: 0x611f0220,
                    mpidr: 0x80000000 | pe as u64,
                    clock: None,
                    caches: None,
                })
                .collect();
        }
        let host = ServiceHost::new(Local, OneWorker(None), Arc::clone(&ram));
        assert!(Local.native_cpu_lease().is_none());
        assert!(host.native_cpu_lease().is_none());
        let mut machine =
            Machine::new(cpu, host, Copyout { calls: 0, fail }, profile, ram.target());
        machine.profile_missing = missing;
        machine
            .install_shared_memory_for_active_processors(
                Arc::clone(&ram),
                &crate::system::events::EventDomain::new(),
            )
            .unwrap();
        let owners = machine
            .bind_active_native_workers(&[WorkerBinding {
                pe: 0,
                worker: WorkerId(17),
            }])
            .unwrap();
        machine.initialize_native_transport(owners);
        (machine, ram)
    }

    #[test]
    fn local_code_readback_forwards_only_outside_held_source_services() {
        let (mut machine, _) = memory_service_machine(false);
        let mut bytes = [0; 3];
        assert!(machine.host.read_native(NativeCode(0x8001), &mut bytes));
        assert_eq!(bytes, [0x48, 0x89, 0xc0]);
        assert!(!machine.host.read_native(NativeCode(0x8004), &mut bytes));
        assert_eq!(
            bytes,
            [0x48, 0x89, 0xc0],
            "owner rejects incomplete extents"
        );
        let held = machine.hold_native_services().unwrap();
        assert!(machine.host.source_service());
        // An open stopped-worker session supplies no local or remote
        // executable read permission, even before selecting its source worker.
        bytes.fill(0xa5);
        assert!(!machine.host.read_native(NativeCode(0x8001), &mut bytes));
        assert_eq!(bytes, [0xa5; 3]);
        assert!(machine.host.select_native_service(WorkerId(17)));
        assert!(!machine.host.read_native(NativeCode(0x8001), &mut bytes));
        assert_eq!(bytes, [0xa5; 3]);
        machine.release_native_services(held);
        assert!(!machine.host.source_service());
        assert!(machine.host.read_native(NativeCode(0x8001), &mut bytes));
        assert_eq!(bytes, [0x48, 0x89, 0xc0]);
    }

    #[test]
    fn native_services_reuse_empty_storage_after_close_and_release() {
        let (mut machine, _) = memory_service_machine(false);
        let mut allocation = None;
        let mut previous_generation = 0;
        for _ in 0..8 {
            let held = machine.hold_native_services().unwrap();
            let current = (
                machine.host.services.as_ptr(),
                machine.host.services.capacity(),
            );
            assert_eq!(machine.host.services.len(), 1);
            if let Some(previous) = allocation {
                assert_eq!(current, previous);
            }
            allocation = Some(current);
            let command = machine.host.services[0]
                .command(HostEffect::MemoryBarrier)
                .unwrap();
            assert!(command.generation() > previous_generation);
            previous_generation = command.generation();
            let receipt = machine
                .host
                .transport
                .apply(&mut machine.host.local, command)
                .unwrap();
            machine.host.services[0].accept(receipt).unwrap();
            assert!(machine.host.select_native_service(WorkerId(17)));
            machine.host.memory_barrier(); // owner command and receipt
            machine.release_native_services(held);
            assert!(machine.host.services.is_empty());
            assert!(machine.host.selected.is_none());
            assert_eq!(
                (
                    machine.host.services.as_ptr(),
                    machine.host.services.capacity()
                ),
                current
            );
            assert!(
                machine
                    .native_publication
                    .as_ref()
                    .unwrap()
                    .require_stopped(1, &[0])
                    .is_err()
            );
        }
    }

    #[test]
    fn canonical_ram_close_capacity_is_reused_by_the_next_scheduler_open() {
        let (mut machine, _) = memory_service_machine(false);
        // SAFETY: closed model with the canonical owner, no external
        // writer/native entry. Explicit finish consumes every receipt.
        unsafe { machine.prepare_native_ram_service() }
            .unwrap()
            .finish();
        assert!(machine.host.services.is_empty());
        let allocation = (
            machine.host.services.as_ptr(),
            machine.host.services.capacity(),
        );
        assert!(allocation.1 > 0);
        let held = machine.hold_native_services().unwrap();
        assert_eq!(
            (
                machine.host.services.as_ptr(),
                machine.host.services.capacity()
            ),
            allocation
        );
        machine.release_native_services(held);
    }

    #[test]
    fn public_begin_end_keep_sessions_and_clear_selection() {
        let (mut machine, _) = memory_service_machine(false);
        let held = machine.hold_native_services().unwrap();
        assert!(machine.host.select_native_service(WorkerId(17)));
        assert!(machine.host.begin_services(Vec::new()).is_err());
        let sessions = machine.host.end_services();
        assert_eq!(sessions.len(), 1);
        assert!(machine.host.selected.is_none());
        assert!(machine.host.services.is_empty());
        let allocation = (sessions.as_ptr(), sessions.capacity());
        machine.host.begin_services(sessions).unwrap();
        assert_eq!(
            (
                machine.host.services.as_ptr(),
                machine.host.services.capacity()
            ),
            allocation
        );
        machine.release_native_services(held);
    }

    #[test]
    #[should_panic(expected = "native service buffer still has active sessions")]
    fn service_buffer_cannot_be_taken_with_live_sessions() {
        let (mut machine, _) = memory_service_machine(false);
        let _held = machine.hold_native_services().unwrap();
        let _ = machine.host.take_empty_service_buffer();
    }

    #[test]
    #[should_panic(expected = "native service buffer is not empty after close")]
    fn service_buffer_cannot_keep_unclosed_sessions() {
        let (mut machine, _) = memory_service_machine(false);
        let _held = machine.hold_native_services().unwrap();
        let sessions = machine.host.end_services();
        machine.host.retain_empty_service_buffer(sessions);
    }

    #[test]
    #[should_panic(expected = "native service buffer still has active sessions")]
    fn empty_service_buffer_with_stale_selection_is_refused() {
        let (mut machine, _) = memory_service_machine(false);
        machine.host.selected = Some(0);
        let _ = machine.host.take_empty_service_buffer();
    }

    #[cfg(feature = "std")]
    #[test]
    fn pending_effect_close_fails_stopped_without_recycling_capacity() {
        let (mut machine, _) = memory_service_machine(false);
        let held = machine.hold_native_services().unwrap();
        let _pending = machine.host.services[0]
            .command(HostEffect::MemoryBarrier)
            .unwrap();
        let stopped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            machine.release_native_services(held);
        }));
        assert!(stopped.is_err());
        // Publication keeps the open session; no release occurred and
        // no emptied allocation was installed after the refused close.
        assert_eq!(machine.host.services.capacity(), 0);
        let inventory = machine.native_publication.as_mut().unwrap();
        assert!(inventory.require_stopped(1, &[0]).is_ok());
        assert!(inventory.request_stop(&[0]).is_err());
    }

    struct RosterLocal(WorkerId);
    impl Host for RosterLocal {
        fn native_worker(&self) -> Option<WorkerId> {
            Some(self.0)
        }
        fn read_ram(&mut self, _: u64, _: &mut [u8]) -> bool {
            panic!("roster model has no RAM lane")
        }
        fn write_ram(&mut self, _: u64, _: &[u8]) -> bool {
            panic!("roster model has no RAM lane")
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
            0
        }
        fn arm_timer(&mut self, _: Option<u64>) {
            unreachable!()
        }
        fn wait_for_interrupt(&mut self) {
            unreachable!()
        }
    }
    struct RosterTransport(Vec<(WorkerOwner, RosterLocal)>);
    impl ServiceTransport for RosterTransport {
        fn apply(
            &mut self,
            _: &mut dyn Host,
            command: HostEffectCommand,
        ) -> Result<HostEffectReceipt, WithPayload<HostEffectCommand>> {
            let (owner, host) = self
                .0
                .iter_mut()
                .find(|(owner, _)| owner.binding() == command.binding())
                .expect("actual assigned owner");
            owner.apply_host_effect(command, host)
        }
        fn fail_stopped(&mut self, reason: &'static str) -> ! {
            panic!("{reason}")
        }
    }

    #[test]
    fn empty_storage_reopens_fresh_sparse_permuted_rosters_with_nonzero_bsp() {
        let (_, _, _, ram) = super::super::policy::tests::fixture();
        let mut host = ServiceHost::new(Local, RosterTransport(Vec::new()), ram);
        let mut allocation = None;
        let mut metadata_allocation = None;
        // Change both source membership and owner order between fully
        // closed inventories. The BSP's worker 17 belongs to nonzero PE 7.
        for members in [[7, 2, 11], [11, 7, 2], [9, 3, 7]] {
            let bindings: Vec<_> = members
                .into_iter()
                .map(|pe| WorkerBinding {
                    pe,
                    worker: WorkerId(if pe == 7 { 17 } else { pe + 100 }),
                })
                .collect();
            let (mut inventory, owners) = WorkerPublications::new(&bindings).unwrap();
            host.transport.0 = owners
                .into_iter()
                .map(|owner| {
                    let local = RosterLocal(owner.binding().worker);
                    (owner, local)
                })
                .collect();
            for (owner, local) in &mut host.transport.0 {
                inventory
                    .accept_initial(owner.publish_initial(local).unwrap())
                    .unwrap();
            }
            let request = inventory.request_stop(&members).unwrap();
            for command in inventory.native_stop_commands(&request).unwrap() {
                let (owner, local) = host
                    .transport
                    .0
                    .iter_mut()
                    .find(|(owner, _)| owner.binding() == command.binding())
                    .unwrap();
                inventory
                    .accept_invalidation(owner.invalidate(command, local).unwrap())
                    .unwrap();
            }
            let held = inventory.hold_stop(request).unwrap();
            inventory.require_all_held(&held).unwrap();
            let mut scratch = core::mem::take(&mut host.round_scratch);
            assert_empty_scratch(&scratch);
            scratch.bindings.extend(inventory.bindings());
            assert_eq!(scratch.bindings, bindings);
            // These plain lists model only exact roster metadata. The
            // owner commands and receipts below remain linear and unchanged.
            scratch
                .expected
                .extend(scratch.bindings.iter().map(|binding| binding.pe));
            let mut services = host.take_empty_service_buffer();
            services.reserve(bindings.len());
            for binding in inventory.bindings().collect::<Vec<_>>() {
                services.push(inventory.open_services(&held, binding.pe).unwrap());
            }
            host.begin_services(services).unwrap();
            assert_eq!(
                host.services
                    .iter()
                    .map(WorkerServices::binding)
                    .collect::<Vec<_>>(),
                bindings
            );
            let current = (host.services.as_ptr(), host.services.capacity());
            if let Some(previous) = allocation {
                assert_eq!(current, previous);
            }
            allocation = Some(current);
            for binding in &bindings {
                assert!(host.select_native_service(binding.worker));
                host.memory_barrier();
                scratch.seen.push(binding.pe);
            }
            assert_eq!(scratch.seen, scratch.expected);
            let current_metadata = scratch_storage(&scratch);
            if let Some(previous) = metadata_allocation {
                assert_eq!(current_metadata, previous);
            }
            metadata_allocation = Some(current_metadata);
            scratch.clear();
            host.round_scratch = scratch;
            assert_empty_scratch(&host.round_scratch);
            let mut services = host.end_services();
            for session in services.drain(..) {
                inventory.close_services(session).unwrap();
            }
            host.retain_empty_service_buffer(services);
            assert!(host.services.is_empty());
            for release in inventory.release(held).unwrap() {
                let (owner, _) = host
                    .transport
                    .0
                    .iter_mut()
                    .find(|(owner, _)| owner.binding() == release.binding())
                    .unwrap();
                inventory
                    .accept_release(owner.release(release).unwrap())
                    .unwrap();
            }
        }
    }
    #[test]
    fn active_bootstrap_keeps_off_slots_unbound_and_refuses_start_or_reset() {
        let (mut machine, ram) = memory_service_machine_slots(false, 10, false);
        assert_eq!(machine.peers.len(), 10);
        assert!(
            machine.peers[1..]
                .iter()
                .all(|peer| matches!(peer, crate::machine::engine::processors::Peer::Off))
        );
        assert_eq!(
            machine
                .native_publication
                .as_ref()
                .unwrap()
                .bindings()
                .count(),
            1
        );
        let before = (
            machine.cpu.state.clone(),
            machine.cpu.pc,
            machine.cpu.retired,
        );
        // SAFETY: this single-threaded test machine owns its registered RAM, with no DMA, device
        // worker or other writer admitted.
        let (devices, mut service) = unsafe { machine.prepare_native_ram_service_parts() }.unwrap();
        assert_eq!(devices.calls, 1); // disjoint receiver, not a clone
        assert!(service.memory().belongs_to(&ram));
        service.validate_current().unwrap();
        service.fence_cpu_memory();
        service.validate_current().unwrap();
        service.finish();
        assert!(
            machine
                .start(crate::machine::engine::Start {
                    cpu: 1,
                    address: 0x1000
                })
                .is_err()
        );
        assert!(machine.warm_reset().is_err());
        let waiting = crate::machine::engine::Wait {
            kind: stela_compiler::aarch64::system::synchronization::WaitKind::Interrupt,
            timeout: None,
            down: true,
        };
        machine.waiting = Some(waiting);
        assert!(machine.wake().is_err());
        assert_eq!(machine.waiting, Some(waiting));
        machine.waiting = None;
        assert!(matches!(
            machine.peers[1],
            crate::machine::engine::processors::Peer::Off
        ));
        assert_eq!(
            (
                machine.cpu.state.clone(),
                machine.cpu.pc,
                machine.cpu.retired
            ),
            before
        );
        assert!(matches!(
            machine.run(1),
            crate::machine::Stop::Unsupported(_)
        ));
    }

    #[test]
    fn held_ram_service_uses_copyout_owner_and_explicit_settlement() {
        let (mut machine, ram) = memory_service_machine(false);
        let before = (
            machine.cpu.state.clone(),
            machine.cpu.pc,
            machine.cpu.retired,
        );
        // SAFETY: one real model owner, no native entry or external writer. The
        // same guard/protocol is exercised; this is not CPL0/native SMP proof.
        let service = unsafe { machine.prepare_native_ram_service() }.unwrap();
        service.validate_current().unwrap();
        assert!(service.memory().belongs_to(&ram));
        assert!(matches!(
            ram.domain().try_execution_loan().unwrap(),
            crate::memory::domain::ExecutionLoanAttempt::ServiceBusy
        ));
        let mut bytes = [0; 8];
        // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
        // registered.
        unsafe { ram.read_physical(0x14080, &mut bytes) }.unwrap();
        assert_eq!(bytes, 0x12345678u64.to_le_bytes());
        service.finish(); // model has no pending controller effect or ACK
        assert_eq!(machine.devices.calls, 1);
        assert_eq!(
            (
                machine.cpu.state.clone(),
                machine.cpu.pc,
                machine.cpu.retired
            ),
            before
        );
        assert!(machine.host.services.is_empty());
        assert!(
            machine
                .native_publication
                .as_ref()
                .unwrap()
                .require_stopped(1, &[0])
                .is_err()
        );
        drop(ram.domain().execution_loan().unwrap());
    }

    #[test]
    fn held_ram_service_closes_and_snapshots_a_frame_without_reacquiring_writers() {
        const FRAME: u64 = 0x10000;
        const BYTES: usize = 16 * 1024;
        let (mut machine, ram) = memory_service_machine_slots(false, 1, true);
        let mut expected = [0; BYTES];
        let tail = [1, 2, 3, 4, 5, 6, 7, 8];
        // SAFETY: private staging and the fixture keeps the registered mapping.
        unsafe {
            ram.publish_physical(FRAME + BYTES as u64 - 8, &tail)
                .unwrap();
            ram.read_physical(FRAME, &mut expected).unwrap();
        }
        assert_eq!(expected[BYTES - 8..], tail);

        // SAFETY: one actual modeled held worker, with no external DMA or writer.
        let service = unsafe { machine.prepare_native_ram_service() }.unwrap();
        assert!(ram.try_service().unwrap().is_none());
        let closed = service.memory().try_close_frame(FRAME).unwrap().unwrap();
        let mut snapshot = [0xa5; BYTES];
        // SAFETY: the output is private and the same held service owns the frame.
        unsafe {
            assert!(ram.read_physical(FRAME, &mut snapshot).is_err());
            assert_eq!(snapshot, [0xa5; BYTES]);
            closed.copy_into(&mut snapshot).unwrap();
        }
        assert_eq!(snapshot, expected);

        drop(closed);
        snapshot.fill(0xa5);
        // SAFETY: private staging and all modeled writers remain held.
        unsafe { ram.read_physical(FRAME, &mut snapshot) }.unwrap();
        assert_eq!(snapshot, expected);
        assert!(ram.try_service().unwrap().is_none());
        service.finish();
        assert!(machine.host.services.is_empty());
        assert_eq!(machine.devices.calls, 1);
        assert!(
            machine
                .native_publication
                .as_ref()
                .unwrap()
                .require_stopped(1, &[0])
                .is_err()
        );
        drop(ram.domain().execution_loan().unwrap());
    }

    #[cfg(feature = "std")]
    #[test]
    fn held_worker_revokes_a_closed_frame_only_after_its_io_loan_returns() {
        const FRAME: u64 = 0x10000;
        const BYTES: usize = 16 * 1024;
        let (mut machine, ram) = memory_service_machine_slots(false, 1, true);
        let aliases = [
            (0x4000, FRAME),
            (0x8000, FRAME + 3 * 4096),
            (0xc000, FRAME + BYTES as u64),
        ];
        model_ram_revocation(&ram, &aliases, true);
        let mut expected = [0; BYTES];
        // SAFETY: the fixture retains the registration and owns these private buffers.
        unsafe {
            ram.publish_physical(FRAME + BYTES as u64 - 8, b"revoked!")
                .unwrap();
            ram.read_physical(FRAME, &mut expected).unwrap();
        }
        let ranges = [stela_contracts::gate::MemoryRange {
            address: FRAME + 4096,
            bytes: 4096,
        }];
        let loan = ram.prepare_external_dma(&ranges).unwrap();

        // SAFETY: the modeled worker is the only CPU owner and the I/O loan
        // only reserves admission. It has published no CPU mapping or DMA.
        let mut service = unsafe { machine.prepare_native_ram_service() }.unwrap();
        assert!(
            service
                .try_revoke_frame_cpu_mappings(FRAME)
                .unwrap()
                .is_none()
        );
        RAM_REVOCATION.with(|model| {
            let model = model.borrow();
            let model = model.as_ref().unwrap();
            assert!(model.requests.is_empty());
            assert_eq!(model.aliases, aliases);
        });

        drop(loan);
        let revoked = service
            .try_revoke_frame_cpu_mappings(FRAME)
            .unwrap()
            .unwrap();
        assert_eq!(revoked.physical(), FRAME);
        let model = RAM_REVOCATION.with(|model| model.take().unwrap());
        assert_eq!(model.requests, [FRAME]);
        assert_eq!(model.aliases, [aliases[2]]);
        let mut output = [0xa5; BYTES];
        // SAFETY: this private output is disjoint from every modeled mapping.
        unsafe { revoked.copy_into(&mut output) }.unwrap();
        assert_eq!(output, expected);
        assert_eq!(&output[BYTES - 8..], b"revoked!");
        drop(revoked);
        drop(ram.prepare_external_dma(&ranges).unwrap());

        service.finish();
        assert_eq!(machine.devices.calls, 1);
        assert!(machine.host.services.is_empty());
        assert!(
            machine
                .native_publication
                .as_ref()
                .unwrap()
                .require_stopped(1, &[0])
                .is_err()
        );
        drop(ram.domain().execution_loan().unwrap());
    }

    #[cfg(feature = "std")]
    #[test]
    fn missing_worker_revocation_ack_keeps_the_original_cohort_failed_and_held() {
        const FRAME: u64 = 0x10000;
        let (mut machine, ram) = memory_service_machine_slots(false, 1, true);
        model_ram_revocation(&ram, &[(0x4000, FRAME)], false);
        let stopped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: one actual modeled worker and no external device writers.
            let mut service = unsafe { machine.prepare_native_ram_service() }.unwrap();
            let _ = service.try_revoke_frame_cpu_mappings(FRAME);
        }));
        let model = RAM_REVOCATION.with(|model| model.take().unwrap());
        assert_eq!(model.requests, [FRAME]);
        assert_eq!(model.aliases, [(0x4000, FRAME)]);
        let reason = stopped.expect_err("a refused revocation must fail stopped");
        assert_eq!(
            reason
                .downcast_ref::<alloc::string::String>()
                .map(alloc::string::String::as_str)
                .or_else(|| reason.downcast_ref::<&str>().copied()),
            Some("held worker RAM CPU mapping revocation did not complete")
        );
        assert!(ram.domain().check_execution().is_err());
        let inventory = machine.native_publication.as_mut().unwrap();
        assert!(inventory.require_stopped(1, &[0]).is_ok());
        assert!(inventory.request_stop(&[0]).is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn revoked_frame_remains_nonresident_until_the_original_snapshot_is_verified() {
        use stela_compiler::context::memory::MemoryStatus;
        use stela_contracts::gate::MemoryRange;

        const FRAME: u64 = 0x10000;
        const BYTES: usize = 16 * 1024;
        const NEXT: u64 = FRAME + BYTES as u64;
        let (mut machine, ram) = memory_service_machine_slots(false, 1, true);
        model_ram_revocation(&ram, &[(0x4000, FRAME)], true);
        let mut descriptors = ram.descriptor_reader();
        let descriptor = descriptors.read_descriptor(FRAME).unwrap();
        // SAFETY: the private source and the exact registered backing stay live.
        unsafe { ram.publish_physical(NEXT - 8, b"retained") }.unwrap();

        // SAFETY: this modeled worker and its synchronous Copyout own the fixture.
        let mut service = unsafe { machine.prepare_native_ram_service() }.unwrap();
        let revoked = service
            .try_revoke_frame_cpu_mappings(FRAME)
            .unwrap()
            .unwrap();
        let mut original = [0; BYTES];
        // SAFETY: this snapshot buffer is private and disjoint from live RAM.
        unsafe { revoked.copy_into(&mut original) }.unwrap();
        assert_eq!(&original[BYTES - 8..], b"retained");
        let mut missing = revoked.mark_nonresident().unwrap();
        assert_eq!(missing.physical(), FRAME);
        let model = RAM_REVOCATION.with(|model| model.take().unwrap());
        assert_eq!(model.requests, [FRAME]);
        assert!(model.aliases.is_empty());

        assert_eq!(ram.nonresident_frame(FRAME, 8), Some(FRAME));
        assert_eq!(ram.nonresident_frame(NEXT - 8, 16), Some(FRAME));
        assert_eq!(ram.nonresident_frame(NEXT, 8), None);
        assert_eq!(
            descriptors.read_descriptor(FRAME),
            Err(MemoryStatus::Nonresident)
        );
        assert_eq!(
            ram.update_descriptor(FRAME, descriptor, !descriptor),
            Err(MemoryStatus::Nonresident)
        );
        let mut output = [0xa5; 16];
        // SAFETY: the output is private, and the fixture retains the RAM registration.
        let read = unsafe { ram.read_physical_progress(NEXT - 8, &mut output) }.unwrap_err();
        assert_eq!(
            (read.status, read.completed),
            (MemoryStatus::Nonresident, 0)
        );
        assert_eq!(output, [0xa5; 16]);
        // SAFETY: the source is private and disjoint from the same registered RAM.
        let write = unsafe { ram.publish_physical(NEXT - 8, &[0xee; 16]) }.unwrap_err();
        assert_eq!(
            (write.status, write.completed),
            (MemoryStatus::Nonresident, 0)
        );
        let ranges = [MemoryRange {
            address: FRAME + 4096,
            bytes: 4096,
        }];
        assert!(matches!(
            ram.prepare_external_dma(&ranges),
            Err(MemoryStatus::Nonresident)
        ));
        let mut neighbor = [0; 8];
        // SAFETY: the output is private. The Copyout wrote the adjacent resident frame.
        unsafe { ram.read_physical(NEXT + 0x80, &mut neighbor) }.unwrap();
        assert_eq!(neighbor, 0x12345678u64.to_le_bytes());

        let mut wrong = original;
        wrong[BYTES - 1] ^= 0x80;
        assert_eq!(
            missing.try_restore(&wrong),
            Err(MemoryStatus::InvalidRequest)
        );
        assert_eq!(ram.nonresident_frame(FRAME, BYTES), Some(FRAME));
        assert_eq!(
            descriptors.read_descriptor(FRAME),
            Err(MemoryStatus::Nonresident)
        );

        assert_eq!(missing.try_restore(&original), Ok(true));
        drop(missing);
        assert_eq!(ram.nonresident_frame(FRAME, BYTES), None);
        assert_eq!(descriptors.read_descriptor(FRAME), Ok(descriptor));
        drop(ram.prepare_external_dma(&ranges).unwrap());
        service.finish();

        let mut actual = [0; BYTES];
        // SAFETY: the output is private and the fixture keeps the same backing.
        unsafe { ram.read_physical(FRAME, &mut actual) }.unwrap();
        assert_eq!(actual, original);
        assert_eq!(machine.devices.calls, 1);
        assert!(machine.host.services.is_empty());
        drop(ram.domain().execution_loan().unwrap());
    }

    #[cfg(feature = "std")]
    #[test]
    fn dropped_or_forgotten_nonresident_frame_prevents_worker_release() {
        use stela_compiler::context::memory::MemoryStatus;

        const FRAME: u64 = 0x10000;
        for forget in [false, true] {
            let (mut machine, ram) = memory_service_machine_slots(false, 1, true);
            model_ram_revocation(&ram, &[(0x4000, FRAME)], true);
            // SAFETY: the modeled worker and its synchronous Copyout own the fixture.
            let mut service = unsafe { machine.prepare_native_ram_service() }.unwrap();
            let missing = service
                .try_revoke_frame_cpu_mappings(FRAME)
                .unwrap()
                .unwrap()
                .mark_nonresident()
                .unwrap();
            assert_eq!(ram.nonresident_frame(FRAME, 8), Some(FRAME));
            let model = RAM_REVOCATION.with(|model| model.take().unwrap());
            assert_eq!(model.requests, [FRAME]);
            if forget {
                core::mem::forget(missing);
                assert_eq!(ram.domain().check_execution(), Ok(()));
            } else {
                drop(missing);
                assert_eq!(ram.domain().check_execution(), Err(MemoryStatus::Domain));
            }

            let stopped =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| service.finish()));
            assert!(stopped.is_err());
            assert_eq!(ram.domain().check_execution(), Err(MemoryStatus::Domain));
            let inventory = machine.native_publication.as_mut().unwrap();
            assert!(inventory.require_stopped(1, &[0]).is_ok());
            assert!(inventory.request_stop(&[0]).is_err());
            assert_eq!(machine.host.services.len(), 1);
        }
    }

    #[test]
    fn abandoned_ram_service_fails_owner_and_never_releases_workers() {
        let (mut machine, ram) = memory_service_machine(false);
        // SAFETY: this single-threaded test machine owns its registered RAM, with no DMA, device
        // worker or other writer admitted.
        let service = unsafe { machine.prepare_native_ram_service() }.unwrap();
        drop(service);
        assert_eq!(
            ram.domain().check_execution(),
            Err(stela_compiler::context::memory::MemoryStatus::Domain)
        );
        assert!(
            machine
                .native_publication
                .as_ref()
                .unwrap()
                .require_stopped(1, &[0])
                .is_ok()
        );
        assert_eq!(machine.host.services.len(), 1);
        let mut bytes = [0xa5; 8];
        // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
        // registered.
        assert!(unsafe { ram.read_physical(0x14080, &mut bytes) }.is_err());
        assert_eq!(bytes, [0xa5; 8]);
    }
    #[test]
    fn refused_copyout_keeps_committed_prefix_and_stopped_owners() {
        let (mut machine, ram) = memory_service_machine(true);
        let before = (
            machine.cpu.state.clone(),
            machine.cpu.pc,
            machine.cpu.retired,
        );
        // SAFETY: this single-threaded test machine owns its registered RAM, with no DMA, device
        // worker or other writer admitted.
        assert!(unsafe { machine.prepare_native_ram_service() }.is_err());
        assert_eq!(machine.devices.calls, 1);
        assert_eq!(
            (
                machine.cpu.state.clone(),
                machine.cpu.pc,
                machine.cpu.retired
            ),
            before
        );
        assert!(
            machine
                .native_publication
                .as_ref()
                .unwrap()
                .require_stopped(1, &[0])
                .is_ok()
        );
        assert_eq!(machine.host.services.len(), 1);
        let mut bytes = [0; 8];
        // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
        // registered.
        unsafe { ram.read_physical(0x14080, &mut bytes) }.unwrap();
        assert_eq!(bytes, 0x12345678u64.to_le_bytes());
    }

    fn branch_source_machine() -> Machine<ServiceHost<Local, OneWorker>, NoDevices> {
        let (mut cpu, profile, missing, ram) = super::super::policy::tests::fixture();
        cpu.pc = 0x4000;
        const SOURCE_PA: u64 = 0x14000;
        const B_SELF: u32 = 0x14000000;
        // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
        // registered.
        unsafe { ram.publish_physical(SOURCE_PA, &B_SELF.to_le_bytes()) }.unwrap();
        let host = ServiceHost::new(Local, OneWorker(None), Arc::clone(&ram));
        let mut machine = Machine::new(cpu, host, NoDevices, profile, ram.target());
        machine.profile_missing = missing;
        let events = crate::system::events::EventDomain::new();
        machine
            .install_shared_memory_for_processors(ram, &events)
            .unwrap();
        let binding = WorkerBinding {
            pe: 0,
            worker: WorkerId(17),
        };
        let owners = machine.bind_native_workers(&[binding]).unwrap();
        machine.initialize_native_transport(owners);
        machine
    }

    fn scratch_storage(scratch: &RoundScratch) -> [(usize, usize); 3] {
        [
            (
                scratch.bindings.as_ptr() as usize,
                scratch.bindings.capacity(),
            ),
            (
                scratch.expected.as_ptr() as usize,
                scratch.expected.capacity(),
            ),
            (scratch.seen.as_ptr() as usize, scratch.seen.capacity()),
        ]
    }

    fn assert_empty_scratch(scratch: &RoundScratch) {
        assert!(scratch.bindings.is_empty());
        assert!(scratch.expected.is_empty());
        assert!(scratch.seen.is_empty());
    }

    #[test]
    fn native_round_scratch_survives_breakpoint_and_ordinary_early_refusals() {
        let mut machine = branch_source_machine();
        let pc = machine.cpu.pc;
        machine.set_breakpoints(&[pc]);
        // SAFETY: the closed one-worker model excludes external writers.
        assert_eq!(unsafe { machine.native_round() }, Err(Stop::Breakpoint(pc)));
        let roster_len = machine
            .native_publication
            .as_ref()
            .unwrap()
            .bindings()
            .count();
        machine.host.round_scratch.expected.reserve(roster_len);
        machine.host.round_scratch.seen.reserve(roster_len);
        let allocation = scratch_storage(&machine.host.round_scratch);
        assert!(
            allocation
                .iter()
                .all(|(_, capacity)| *capacity >= roster_len)
        );
        for _ in 0..8 {
            // Every call opens/closes generation-bound sessions, then
            // releases the owner before the next breakpoint entry.
            // SAFETY: this single-threaded test machine owns its registered RAM, with no DMA,
            // device worker or other writer admitted.
            assert_eq!(unsafe { machine.native_round() }, Err(Stop::Breakpoint(pc)));
            assert_empty_scratch(&machine.host.round_scratch);
            assert_eq!(scratch_storage(&machine.host.round_scratch), allocation);
            assert!(machine.host.services.is_empty());
        }
        // A prior roster is never a substitute for the current binding.
        let inventory = machine.native_publication.take().unwrap();
        assert!(matches!(
            // SAFETY: this single-threaded test machine owns its registered RAM, with no DMA,
            // device worker or other writer admitted.
            unsafe { machine.native_round() },
            Err(Stop::Unsupported(_))
        ));
        assert_empty_scratch(&machine.host.round_scratch);
        assert_eq!(scratch_storage(&machine.host.round_scratch), allocation);
        machine.native_publication = Some(inventory);
        // Refuse before obtaining a stop when canonical ownership is
        // absent; keep capacity even though this round did collect bindings.
        let memory = machine.cpu.canonical_memory.take().unwrap();
        assert!(matches!(
            // SAFETY: this single-threaded test machine owns its registered RAM, with no DMA,
            // device worker or other writer admitted.
            unsafe { machine.native_round() },
            Err(Stop::Unsupported(_))
        ));
        assert_empty_scratch(&machine.host.round_scratch);
        assert_eq!(scratch_storage(&machine.host.round_scratch), allocation);
        machine.cpu.canonical_memory = Some(memory);
        machine.set_breakpoints(&[]);
        let before = (
            machine.cpu.state.clone(),
            machine.cpu.pc,
            machine.cpu.retired,
        );
        // The native admission refusal keeps the held owner/session.
        // It still restores empty metadata instead of caching that permission.
        assert!(matches!(
            // SAFETY: this single-threaded test machine owns its registered RAM, with no DMA,
            // device worker or other writer admitted.
            unsafe { machine.native_round() },
            Err(Stop::Unsupported(_))
        ));
        assert_empty_scratch(&machine.host.round_scratch);
        assert_eq!(scratch_storage(&machine.host.round_scratch), allocation);
        assert_eq!(
            (
                machine.cpu.state.clone(),
                machine.cpu.pc,
                machine.cpu.retired
            ),
            before
        );
        assert_eq!(machine.host.services.len(), 1);
        assert!(
            machine
                .native_publication
                .as_ref()
                .unwrap()
                .require_stopped(1, &[0])
                .is_ok()
        );
    }

    #[test]
    fn native_breakpoint_releases_stopped_services_and_trace_captures_frame() {
        let mut machine = branch_source_machine();
        let pc = machine.cpu.pc;
        machine.cpu.state.x[..4].copy_from_slice(&[0x10, 0x20, 0x30, 0x40]);
        let before = (machine.cpu.state.clone(), pc, machine.cpu.retired);
        machine.set_breakpoints(&[pc]);
        machine.set_traces(&[pc]);
        // SAFETY: closed one-worker model with no external writers/native entry.
        assert_eq!(unsafe { machine.native_round() }, Err(Stop::Breakpoint(pc)));
        assert!(machine.take_trace_hits().is_empty());
        assert!(machine.host.services.is_empty());
        machine.set_breakpoints(&[]);
        // Reenter the released owner. The host's CPL0 refusal follows
        // the trace, and cannot stand in for successful native execution.
        assert!(matches!(
            // SAFETY: this single-threaded test machine owns its registered RAM, with no DMA,
            // device worker or other writer admitted.
            unsafe { machine.native_round() },
            Err(Stop::Unsupported(_))
        ));
        let hits = machine.take_trace_hits();
        assert_eq!(hits.len(), 1);
        assert_eq!((hits[0].pc, hits[0].pe), (pc, 0));
        assert_eq!(hits[0].arguments, [0x10, 0x20, 0x30, 0x40]);
        assert_eq!(
            (
                machine.cpu.state.clone(),
                machine.cpu.pc,
                machine.cpu.retired
            ),
            before
        );
    }

    #[test]
    fn unprivileged_round_recovers_refusal_without_source_execution_or_raw_ram_fallback() {
        let mut machine = branch_source_machine();
        let before = (
            machine.cpu.state.clone(),
            machine.cpu.pc,
            machine.cpu.retired,
        );
        // SAFETY: closed one-worker model fixture, canonical byte owner,
        // no concurrent writer. The real CPL0 permission must refuse on this host.
        let stop = unsafe { machine.native_round() }.unwrap_err();
        assert!(matches!(stop, crate::machine::Stop::Unsupported(_)));
        assert_eq!(
            (
                machine.cpu.state.clone(),
                machine.cpu.pc,
                machine.cpu.retired
            ),
            before
        );
        assert_eq!(machine.statistics.blocks, 0);
        assert!(
            machine
                .native_publication
                .as_ref()
                .unwrap()
                .require_stopped(1, &[0])
                .is_ok()
        );
        assert_eq!(machine.host.services.len(), 1); // refusal does not release
    }
}
