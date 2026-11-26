//! StelaGPU's call gate: the window through which the StelaGPU kext opens a
//! connection for each task and the task enters the native GPU service
//! with no kernel crossing per call.
//!
//! The window's layout is [`window`]'s. Holding a gate page's mapping is
//! its task's authority; pages are the guest's 16 KiB, so no task maps
//! another's gate.
//!
//! A 64-bit store to a gate's `CALL` names a call record in the storing
//! thread's memory. The service runs it in the store's own step, on that
//! thread's PE and address space, and completes the record in place; the
//! store retires once the call has. Only when the record itself is out of
//! the thread's reach does the gate's `FAULT` report why: read on the PE
//! whose store made the call, it answers for that call alone, so another
//! thread's call through the same connection cannot clear it.
//!
//! Each store's request leaves the device ([`Gate::take`]) to run outside the
//! board's lock, so a long shader compile holds up no other PE's device
//! accesses, and comes back to be recorded ([`Gate::finish`]). Opening and
//! closing a connection go the same way, since they wait for the service too.
//!
//! Calls never block on the GPU: when the service reports completed work,
//! the gate raises its interrupt until the kext acknowledges it, so threads
//! waiting for command buffers sleep in the kernel and ask again on wake.
use super::bus::{Access, Direction, MmioDevice, MmioError, Width, refuse};
use alloc::{collections::BTreeMap, sync::Arc, vec::Vec};
use core::num::NonZeroU64;
use stela_contracts::{
    bus::AdvancePolicy,
    gate::{
        BundleReport, CallerFault, GateAnswer, GateCall, GateRequest, GateService, MemoryRange,
        window::{
            ACK, BUNDLE, CALL, CLOSE, COUNT, FAULT, FAULT_DENIED, FAULT_NOT_RAM, FAULT_UNSUPPORTED,
            IMPORT, IMPORT_ADDRESS, IMPORT_BYTES, IMPORT_TOKEN, IMPORTED, OPEN, OPENED, PAGE,
            RETIRE, RETIRED, STATUS,
        },
    },
    interrupt::{AicInput, Wire},
};

const CONNECTIONS: usize = stela_contracts::gate::window::CONNECTIONS as usize;

const NAME: &str = "stela-gpu-gate";
/// How soon the gate advances again while GPU work is outstanding: 250 us
/// of the 24 MHz counter, which bounds how late a completion is noticed.
const COMPLETION_POLL_TICKS: u64 = 6_000;

/// A gate page's state.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    Free,
    /// Held for the open in flight.
    Opening,
    Open(NonZeroU64),
}

pub struct Gate {
    /// Which of the board's gates this is.
    window: usize,
    service: Arc<dyn GateService>,
    irq: AicInput,
    /// Completed work the kext has not acknowledged.
    status: bool,
    /// Each gate page's connection, by page - 1.
    slots: [Slot; CONNECTIONS],
    /// Each PE's last call through a gate: the gate and its record.
    issued: BTreeMap<usize, (usize, u64)>,
    /// The fault of each call whose record was out of reach, by gate and
    /// record, until its PE's next call.
    faults: BTreeMap<(usize, u64), u64>,
    opened: u64,
    /// The address of the range the next `IMPORT_BYTES` adds.
    range_address: u64,
    ranges: Vec<MemoryRange>,
    token: u64,
    imported: u64,
    retired: u64,
    /// Imports the service holds, whose tokens have not come back.
    reachable: u64,
    /// What each PE's store asked, until that store's step takes it.
    pending: BTreeMap<usize, GateRequest>,
}

impl Gate {
    pub fn new(window: usize, service: Arc<dyn GateService>, irq: AicInput) -> Self {
        Self {
            window,
            service,
            irq,
            status: false,
            slots: [Slot::Free; CONNECTIONS],
            issued: BTreeMap::new(),
            faults: BTreeMap::new(),
            opened: 0,
            range_address: 0,
            ranges: Vec::new(),
            token: 0,
            imported: 0,
            retired: 0,
            reachable: 0,
            pending: BTreeMap::new(),
        }
    }

    /// The request PE `cpu`'s store made, to run in that store's own step.
    pub fn take(&mut self, cpu: usize) -> Option<GateCall> {
        let request = self.pending.remove(&cpu)?;
        Some(GateCall::new(self.window, self.service.clone(), request))
    }

    /// Records what a taken request came to.
    pub fn finish(&mut self, answer: GateAnswer) {
        match answer {
            GateAnswer::Opened(connection) => {
                self.opened = 0;
                let Some(gate) = self.slots.iter().position(|&slot| slot == Slot::Opening) else {
                    return;
                };
                self.slots[gate] = connection.map_or(Slot::Free, Slot::Open);
                if connection.is_some() {
                    self.faults.retain(|&(other, _), _| other != gate);
                    self.opened = gate as u64 + 1;
                }
            }
            GateAnswer::Closed => {}
            GateAnswer::Served {
                gate,
                connection,
                record,
                fault,
            } => {
                // The gate may have closed, and its page opened again, since.
                if self.slots[gate] == Slot::Open(connection) {
                    let code = match fault {
                        None => 0,
                        Some(CallerFault::Denied { .. }) => FAULT_DENIED,
                        Some(CallerFault::NotRam { .. }) => FAULT_NOT_RAM,
                        Some(CallerFault::Unsupported(_)) => FAULT_UNSUPPORTED,
                    };
                    if code == 0 {
                        self.faults.remove(&(gate, record));
                    } else {
                        self.faults.insert((gate, record), code);
                    }
                }
            }
            GateAnswer::Imported(handle) => {
                self.imported = handle.map_or(0, NonZeroU64::get);
                self.reachable += u64::from(handle.is_some());
            }
            GateAnswer::Retired(token) => {
                self.retired = token.map_or(0, NonZeroU64::get);
                self.reachable -= u64::from(token.is_some());
            }
            GateAnswer::Reported => {}
        }
    }

    /// Release what the guest left here for good, as the PC stops: every
    /// connection, then each token that frees. A service without a proof of
    /// idle keeps all of it. Answers the imports the GPU may still reach.
    pub fn stop(&mut self) -> u64 {
        if self.service.quiesced() {
            self.pending.clear();
            for slot in &mut self.slots {
                if let Slot::Open(connection) = core::mem::replace(slot, Slot::Free) {
                    self.service.disconnect(connection);
                }
            }
            while self.service.retired().is_some() {
                self.reachable -= 1;
            }
        }
        self.reachable
    }

    /// Holds `request` for PE `cpu`'s step. A request the PE's last store
    /// left, which its step never took, is lost: the service is told.
    fn hold(&mut self, cpu: usize, request: GateRequest) {
        if self.pending.insert(cpu, request).is_some() {
            self.service
                .dropped("a gate store replaced one its PE's step never ran");
        }
    }

    fn open(&mut self, cpu: usize) {
        self.opened = 0;
        if let Some(gate) = self.slots.iter().position(|&slot| slot == Slot::Free) {
            self.slots[gate] = Slot::Opening;
            self.hold(cpu, GateRequest::Open);
        }
    }

    fn close(&mut self, cpu: usize, page: u64) -> Result<(), &'static str> {
        let gate = gate_index(page).ok_or("no gate page")?;
        let Slot::Open(connection) = self.slots[gate] else {
            return Err("a closed gate");
        };
        self.slots[gate] = Slot::Free;
        self.faults.retain(|&(other, _), _| other != gate);
        let before = self.pending.len();
        self.pending.retain(
            |_, request| !matches!(request, GateRequest::Serve { gate: other, .. } if *other == gate),
        );
        if self.pending.len() != before {
            self.service
                .dropped("a gate's close dropped a call another PE's step never ran");
        }
        self.hold(cpu, GateRequest::Close(connection));
        Ok(())
    }

    fn import(&mut self, cpu: usize, page: u64) -> Result<(), &'static str> {
        self.imported = 0;
        let gate = gate_index(page).ok_or("no gate page")?;
        let Slot::Open(connection) = self.slots[gate] else {
            return Err("an import for a closed gate");
        };
        let token = NonZeroU64::new(self.token).ok_or("an import without a token")?;
        let ranges = core::mem::take(&mut self.ranges);
        let request = GateRequest::Import {
            connection,
            ranges,
            token,
        };
        self.hold(cpu, request);
        Ok(())
    }
}

/// The gate index of gate page `page`.
fn gate_index(page: u64) -> Option<usize> {
    let gate = usize::try_from(page).ok()?.checked_sub(1)?;
    (gate < CONNECTIONS).then_some(gate)
}

impl MmioDevice for Gate {
    fn advance_policy(&self) -> AdvancePolicy {
        AdvancePolicy::Always
    }

    fn advance(&mut self, access: &mut Access<'_>) -> Result<Option<u64>, MmioError> {
        if self.service.signaled() {
            self.status = true;
            access.interrupts.set_level(Wire::Aic(self.irq), true);
        }
        // While work is outstanding, look again soon: the service reads the
        // completion the GPU writes, with no host interrupt or Linux work.
        Ok(self
            .service
            .awaits_completion()
            .then(|| access.now + COMPLETION_POLL_TICKS))
    }

    fn name(&self) -> &'static str {
        NAME
    }

    fn read(
        &mut self,
        offset: u64,
        width: Width,
        access: &mut Access<'_>,
    ) -> Result<u64, MmioError> {
        let refused = |reason| refuse(NAME, offset, width, Direction::Read, reason);
        if width != Width::Double {
            return Err(refused("registers are 64-bit"));
        }
        let (page, register) = (offset / PAGE, offset % PAGE);
        match (page, register) {
            (0, OPENED) => Ok(self.opened),
            (0, COUNT) => Ok(CONNECTIONS as u64),
            (0, STATUS) => Ok(self.status.into()),
            (0, IMPORTED) => Ok(self.imported),
            (0, RETIRED) => Ok(self.retired),
            (_, FAULT) => {
                let gate = gate_index(page).ok_or_else(|| refused("no such register"))?;
                // The frontend reads the fault after a call its store's step
                // did not serve: a request still held for this PE was never
                // taken there, and runs late if at all.
                if matches!(
                    self.pending.get(&access.cpu),
                    Some(GateRequest::Serve { .. })
                ) {
                    self.service
                        .dropped("a gate call was still pending at its PE's next gate access");
                }
                // The fault of this PE's own last call through this gate.
                Ok(match self.issued.get(&access.cpu) {
                    Some(&(issued, record)) if issued == gate => {
                        self.faults.get(&(gate, record)).copied().unwrap_or(0)
                    }
                    _ => 0,
                })
            }
            _ => Err(refused("no such register")),
        }
    }

    fn write(
        &mut self,
        offset: u64,
        width: Width,
        value: u64,
        access: &mut Access<'_>,
    ) -> Result<(), MmioError> {
        let refused = |reason| refuse(NAME, offset, width, Direction::Write, reason);
        if width != Width::Double {
            return Err(refused("registers are 64-bit"));
        }
        let (page, register) = (offset / PAGE, offset % PAGE);
        match (page, register) {
            (0, OPEN) => self.open(access.cpu),
            (0, CLOSE) => self.close(access.cpu, value).map_err(refused)?,
            (0, ACK) => {
                self.status = false;
                access.interrupts.set_level(Wire::Aic(self.irq), false);
            }
            (0, IMPORT_ADDRESS) => self.range_address = value,
            (0, IMPORT_BYTES) => self.ranges.push(MemoryRange {
                address: self.range_address,
                bytes: value,
            }),
            (0, IMPORT_TOKEN) => self.token = value,
            (0, IMPORT) => self.import(access.cpu, value).map_err(refused)?,
            (0, RETIRE) => {
                self.retired = 0;
                self.hold(access.cpu, GateRequest::Retire);
            }
            (0, BUNDLE) => {
                let report =
                    BundleReport::from_raw(value).ok_or_else(|| refused("no such report"))?;
                self.hold(access.cpu, GateRequest::Bundle(report));
            }
            (_, CALL) => {
                let gate = gate_index(page).ok_or_else(|| refused("no such register"))?;
                let Slot::Open(connection) = self.slots[gate] else {
                    return Err(refused("a call through a closed gate"));
                };
                let request = GateRequest::Serve {
                    gate,
                    connection,
                    record: value,
                };
                // The PE's previous call is answered: its fault goes.
                if let Some(previous) = self.issued.insert(access.cpu, (gate, value)) {
                    self.faults.remove(&previous);
                }
                self.hold(access.cpu, request);
            }
            _ => return Err(refused("no such register")),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use alloc::vec::Vec;
    use std::sync::Mutex;
    use stela_contracts::{
        gate::{CallerMemory, window::IMPORT_TOKEN_WRITABLE},
        interrupt::{InterruptSink, Wire},
    };

    #[derive(Default)]
    struct Log {
        signaled: bool,
        awaiting: bool,
        next: u64,
        disconnected: Vec<u64>,
        served: Vec<(u64, u64, u8)>,
        imported: Vec<(u64, Vec<MemoryRange>, u64)>,
        retired: Vec<u64>,
        quiesced: bool,
        bundles: Vec<BundleReport>,
        dropped: Vec<&'static str>,
    }

    struct Service(Mutex<Log>);

    impl GateService for Service {
        fn connect(&self) -> Result<NonZeroU64, &'static str> {
            let mut log = self.0.lock().unwrap();
            log.next += 1;
            NonZeroU64::new(log.next).ok_or("exhausted")
        }

        fn disconnect(&self, connection: NonZeroU64) {
            self.0.lock().unwrap().disconnected.push(connection.get());
        }

        fn serve(
            &self,
            connection: NonZeroU64,
            record: u64,
            caller: &mut dyn CallerMemory,
        ) -> Result<(), CallerFault> {
            let mut byte = [0];
            caller.read(record, &mut byte)?;
            self.0
                .lock()
                .unwrap()
                .served
                .push((connection.get(), record, byte[0]));
            Ok(())
        }

        fn import(
            &self,
            connection: NonZeroU64,
            ranges: &[MemoryRange],
            token: NonZeroU64,
        ) -> Result<NonZeroU64, &'static str> {
            let mut log = self.0.lock().unwrap();
            log.imported
                .push((connection.get(), ranges.to_vec(), token.get()));
            NonZeroU64::new(0x1000 + log.imported.len() as u64).ok_or("exhausted")
        }

        fn retired(&self) -> Option<NonZeroU64> {
            NonZeroU64::new(self.0.lock().unwrap().retired.pop()?)
        }

        fn quiesced(&self) -> bool {
            self.0.lock().unwrap().quiesced
        }

        fn signaled(&self) -> bool {
            core::mem::take(&mut self.0.lock().unwrap().signaled)
        }

        fn awaits_completion(&self) -> bool {
            self.0.lock().unwrap().awaiting
        }

        fn bundle(&self, report: BundleReport) {
            self.0.lock().unwrap().bundles.push(report);
        }

        fn dropped(&self, reason: &'static str) {
            self.0.lock().unwrap().dropped.push(reason);
        }
    }

    /// One byte per address below 0x100.
    struct Caller;

    impl CallerMemory for Caller {
        fn read(&mut self, address: u64, buffer: &mut [u8]) -> Result<(), CallerFault> {
            let byte = u8::try_from(address).map_err(|_| CallerFault::Denied { address })?;
            buffer.fill(byte);
            Ok(())
        }

        fn write(&mut self, address: u64, _data: &[u8]) -> Result<(), CallerFault> {
            Err(CallerFault::Denied { address })
        }
    }

    const IRQ: AicInput = AicInput::new(0, 700);

    struct NoInterrupts;

    impl InterruptSink for NoInterrupts {
        fn set_level(&mut self, _wire: Wire, _asserted: bool) {}
    }

    #[derive(Default)]
    struct Line(Option<bool>);

    impl InterruptSink for Line {
        fn set_level(&mut self, wire: Wire, asserted: bool) {
            assert_eq!(wire, Wire::Aic(IRQ));
            self.0 = Some(asserted);
        }
    }

    fn store(gate: &mut Gate, cpu: usize, offset: u64, value: u64) -> Result<(), MmioError> {
        gate.write(
            offset,
            Width::Double,
            value,
            &mut Access::new(cpu, 0, &mut NoInterrupts),
        )
    }

    fn load(gate: &mut Gate, offset: u64) -> u64 {
        load_on(gate, 0, offset)
    }

    fn load_on(gate: &mut Gate, cpu: usize, offset: u64) -> u64 {
        gate.read(
            offset,
            Width::Double,
            &mut Access::new(cpu, 0, &mut NoInterrupts),
        )
        .unwrap()
    }

    #[test]
    fn the_kernel_names_the_memory_a_connection_imports() {
        const DESCRIPTOR: u64 = 8;
        const WRITABLE: u64 = DESCRIPTOR | IMPORT_TOKEN_WRITABLE;
        let service = Arc::new(Service(Mutex::new(Log::default())));
        let mut gate = Gate::new(0, service.clone(), IRQ);
        store(&mut gate, 0, OPEN, 1).unwrap();
        serve(&mut gate, 0);
        let page = load(&mut gate, OPENED);
        for (address, bytes) in [(0x8000, 0x4000), (0x2_0000, 0x8000)] {
            store(&mut gate, 0, IMPORT_ADDRESS, address).unwrap();
            store(&mut gate, 0, IMPORT_BYTES, bytes).unwrap();
        }
        assert!(store(&mut gate, 0, IMPORT, page).is_err(), "no token yet");
        store(&mut gate, 0, IMPORT_TOKEN, WRITABLE).unwrap();
        store(&mut gate, 0, IMPORT, page).unwrap();
        serve(&mut gate, 0);
        assert_eq!(load(&mut gate, IMPORTED), 0x1001);
        assert_eq!(
            service.0.lock().unwrap().imported,
            [(
                1,
                [
                    MemoryRange {
                        address: 0x8000,
                        bytes: 0x4000
                    },
                    MemoryRange {
                        address: 0x2_0000,
                        bytes: 0x8000
                    }
                ]
                .to_vec(),
                WRITABLE
            )]
        );
        // Ranges go with their import.
        store(&mut gate, 0, IMPORT_TOKEN, DESCRIPTOR).unwrap();
        store(&mut gate, 0, IMPORT, page).unwrap();
        serve(&mut gate, 0);
        assert_eq!(
            service.0.lock().unwrap().imported[1],
            (1, Vec::new(), DESCRIPTOR)
        );

        assert!(store(&mut gate, 0, IMPORT, page + 1).is_err());
        service
            .0
            .lock()
            .unwrap()
            .retired
            .extend([DESCRIPTOR, WRITABLE]);
        for token in [WRITABLE, DESCRIPTOR, 0] {
            store(&mut gate, 0, RETIRE, 1).unwrap();
            serve(&mut gate, 0);
            assert_eq!(load(&mut gate, RETIRED), token);
        }
    }

    /// Runs what PE `cpu`'s store left, as its step does.
    fn serve(gate: &mut Gate, cpu: usize) {
        if let Some(call) = gate.take(cpu) {
            let outcome = call.run(&mut Caller);
            gate.finish(outcome.answer);
        }
    }

    fn opened(gate: &mut Gate) -> u64 {
        store(gate, 0, OPEN, 0).unwrap();
        serve(gate, 0);
        load(gate, OPENED)
    }

    fn gate(log: &Arc<Service>) -> Gate {
        Gate::new(0, log.clone(), IRQ)
    }

    #[test]
    fn a_request_its_step_never_took_is_reported_when_dropped() {
        let log = Arc::new(Service(Mutex::default()));
        let mut gate = gate(&log);
        let (first, second) = (opened(&mut gate), opened(&mut gate));
        let call = |page| page * PAGE + CALL;
        // Each step takes its store's call: nothing is lost.
        store(&mut gate, 0, call(first), 0x10).unwrap();
        serve(&mut gate, 0);
        assert!(log.0.lock().unwrap().dropped.is_empty());
        // A fault read while the PE's call is held finds it untaken.
        store(&mut gate, 0, call(first), 0x18).unwrap();
        gate.read(
            first * PAGE + FAULT,
            Width::Double,
            &mut Access::new(0, 0, &mut NoInterrupts),
        )
        .unwrap();
        assert_eq!(log.0.lock().unwrap().dropped.len(), 1);
        serve(&mut gate, 0);
        log.0.lock().unwrap().dropped.clear();
        // A second store before the first one's step ran replaces it.
        store(&mut gate, 0, call(first), 0x20).unwrap();
        store(&mut gate, 0, call(first), 0x30).unwrap();
        assert_eq!(log.0.lock().unwrap().dropped.len(), 1);
        serve(&mut gate, 0);
        // Closing a gate drops another PE's call through it.
        store(&mut gate, 1, call(second), 0x40).unwrap();
        store(&mut gate, 0, CLOSE, second).unwrap();
        assert_eq!(log.0.lock().unwrap().dropped.len(), 2);
        assert_eq!(
            log.0
                .lock()
                .unwrap()
                .served
                .iter()
                .map(|s| s.1)
                .collect::<Vec<_>>(),
            [0x10, 0x18, 0x30]
        );
    }

    #[test]
    fn each_pe_serves_the_call_its_own_store_named() {
        let log = Arc::new(Service(Mutex::default()));
        let mut gate = gate(&log);
        assert_eq!(load(&mut gate, COUNT), CONNECTIONS as u64);
        let (first, second) = (opened(&mut gate), opened(&mut gate));
        assert_eq!((first, second), (1, 2));
        store(&mut gate, 3, first * PAGE + CALL, 0x11).unwrap();
        store(&mut gate, 5, second * PAGE + CALL, 0x22).unwrap();
        serve(&mut gate, 5);
        serve(&mut gate, 3);
        serve(&mut gate, 3);
        assert_eq!(
            log.0.lock().unwrap().served,
            [(2, 0x22, 0x22), (1, 0x11, 0x11)]
        );
        store(&mut gate, 3, first * PAGE + CALL, 0x1000).unwrap();
        serve(&mut gate, 3);
        assert_eq!(load_on(&mut gate, 3, first * PAGE + FAULT), FAULT_DENIED);
    }

    /// Two threads of one task call through its one gate on two PEs: the
    /// one whose record was out of reach still reads its fault after the
    /// other's call succeeded, and the other reads none.
    #[test]
    fn a_fault_belongs_to_the_call_that_met_it() {
        let log = Arc::new(Service(Mutex::default()));
        let mut gate = gate(&log);
        let page = opened(&mut gate);
        let (call, fault) = (page * PAGE + CALL, page * PAGE + FAULT);
        store(&mut gate, 3, call, 0x1000).unwrap();
        serve(&mut gate, 3);
        store(&mut gate, 5, call, 0x40).unwrap();
        serve(&mut gate, 5);
        assert_eq!(load_on(&mut gate, 3, fault), FAULT_DENIED);
        assert_eq!(load_on(&mut gate, 5, fault), 0);
        // A PE that never called through the gate reads no fault.
        assert_eq!(load_on(&mut gate, 7, fault), 0);
        // The PE's next call answers for itself.
        store(&mut gate, 3, call, 0x41).unwrap();
        serve(&mut gate, 3);
        assert_eq!(load_on(&mut gate, 3, fault), 0);
        assert!(gate.faults.is_empty());
        // A closed gate keeps no fault for the page's next connection.
        store(&mut gate, 3, call, 0x2000).unwrap();
        serve(&mut gate, 3);
        store(&mut gate, 0, CLOSE, page).unwrap();
        serve(&mut gate, 0);
        assert_eq!(opened(&mut gate), page);
        assert_eq!(load_on(&mut gate, 3, fault), 0);
    }

    #[test]
    fn a_call_runs_with_the_devices_free_and_reports_to_a_gate_still_open() {
        let log = Arc::new(Service(Mutex::default()));
        let mut gate = gate(&log);
        let page = opened(&mut gate);
        store(&mut gate, 0, page * PAGE + CALL, 0x1000).unwrap();
        let call = gate.take(0).unwrap();
        // The page closes and reopens under the call, which is another's now.
        store(&mut gate, 1, CLOSE, page).unwrap();
        serve(&mut gate, 1);
        assert_eq!(opened(&mut gate), page);
        let outcome = call.run(&mut Caller);
        gate.finish(outcome.answer);
        assert_eq!(load(&mut gate, page * PAGE + FAULT), 0);
    }

    #[test]
    fn a_full_window_opens_nothing_and_an_in_flight_open_holds_its_page() {
        let log = Arc::new(Service(Mutex::default()));
        let mut gate = gate(&log);
        for page in 1..CONNECTIONS as u64 {
            assert_eq!(opened(&mut gate), page);
        }
        store(&mut gate, 0, OPEN, 0).unwrap();
        store(&mut gate, 1, OPEN, 0).unwrap();
        assert!(gate.take(1).is_none());
        serve(&mut gate, 0);
        assert_eq!(load(&mut gate, OPENED), CONNECTIONS as u64);
        store(&mut gate, 0, OPEN, 0).unwrap();
        assert!(gate.take(0).is_none());
        assert_eq!(load(&mut gate, OPENED), 0);
    }

    #[test]
    fn stopping_releases_connections_only_after_the_service_proves_idle() {
        let log = Arc::new(Service(Mutex::default()));
        let mut gate = gate(&log);
        let page = opened(&mut gate);
        store(&mut gate, 0, IMPORT_TOKEN, 7).unwrap();
        store(&mut gate, 0, IMPORT, page).unwrap();
        serve(&mut gate, 0);
        store(&mut gate, 0, IMPORT_TOKEN, 8).unwrap();
        store(&mut gate, 0, IMPORT, page).unwrap();
        serve(&mut gate, 0);

        assert_eq!(gate.stop(), 2);
        assert!(log.0.lock().unwrap().disconnected.is_empty());

        log.0.lock().unwrap().quiesced = true;
        log.0.lock().unwrap().retired.push(7);
        assert_eq!(gate.stop(), 1);
        assert_eq!(log.0.lock().unwrap().disconnected, [1]);
        assert!(store(&mut gate, 0, page * PAGE + CALL, 0x11).is_err());

        log.0.lock().unwrap().retired.push(8);
        assert_eq!(gate.stop(), 0);
        assert_eq!(log.0.lock().unwrap().disconnected, [1]);
    }

    #[test]
    fn closed_gates_refuse_calls_and_drop_pending_ones() {
        let log = Arc::new(Service(Mutex::default()));
        let mut gate = gate(&log);
        let page = opened(&mut gate);
        store(&mut gate, 1, page * PAGE + CALL, 0x11).unwrap();
        store(&mut gate, 0, CLOSE, page).unwrap();
        serve(&mut gate, 1);
        serve(&mut gate, 0);
        assert!(log.0.lock().unwrap().served.is_empty());
        assert_eq!(log.0.lock().unwrap().disconnected, [1]);
        assert!(store(&mut gate, 0, page * PAGE + CALL, 0x11).is_err());
        assert!(store(&mut gate, 0, CLOSE, page).is_err());
        assert!(store(&mut gate, 0, (CONNECTIONS as u64 + 1) * PAGE + CALL, 0).is_err());
        assert!(
            gate.write(
                OPEN,
                Width::Word,
                0,
                &mut Access::new(0, 0, &mut NoInterrupts)
            )
            .is_err()
        );
    }

    #[test]
    fn completed_work_holds_the_interrupt_until_acknowledged() {
        let log = Arc::new(Service(Mutex::default()));
        let mut gate = gate(&log);
        let mut line = Line::default();
        gate.advance(&mut Access::new(0, 0, &mut line)).unwrap();
        assert_eq!((line.0, load(&mut gate, STATUS)), (None, 0));
        log.0.lock().unwrap().signaled = true;
        gate.advance(&mut Access::new(0, 0, &mut line)).unwrap();
        gate.advance(&mut Access::new(0, 0, &mut line)).unwrap();
        assert_eq!((line.0, load(&mut gate, STATUS)), (Some(true), 1));
        gate.write(ACK, Width::Double, 0, &mut Access::new(0, 0, &mut line))
            .unwrap();
        assert_eq!((line.0, load(&mut gate, STATUS)), (Some(false), 0));
    }

    /// While the service awaits a completion, the gate asks to advance
    /// again within the poll interval; idle, it asks for nothing.
    #[test]
    fn outstanding_work_brings_the_gate_back_soon() {
        let log = Arc::new(Service(Mutex::default()));
        let mut gate = gate(&log);
        let mut line = Line::default();
        let next = gate.advance(&mut Access::new(0, 100, &mut line)).unwrap();
        assert_eq!(next, None);
        log.0.lock().unwrap().awaiting = true;
        let next = gate.advance(&mut Access::new(0, 100, &mut line)).unwrap();
        assert_eq!(next, Some(100 + COMPLETION_POLL_TICKS));
    }

    #[test]
    fn the_kext_reports_its_plugin_through_the_service() {
        let log = Arc::new(Service(Mutex::default()));
        let mut gate = gate(&log);
        use stela_contracts::gate::window::{BUNDLE_MISSING, BUNDLE_UPDATED};
        for value in [BUNDLE_UPDATED, BUNDLE_MISSING] {
            store(&mut gate, 0, BUNDLE, value).unwrap();
            serve(&mut gate, 0);
        }
        assert!(store(&mut gate, 0, BUNDLE, 0).is_err());
        assert!(store(&mut gate, 0, BUNDLE, 5).is_err());
        assert_eq!(
            log.0.lock().unwrap().bundles,
            [BundleReport::Updated, BundleReport::Missing]
        );
    }
}
