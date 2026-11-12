//! Scalar guest accesses that reach devices, and the aborts they raise.

use super::*;
use stela_contracts::refuse;

impl<H: Host, D: Devices> Machine<H, D> {
    /// A device read by the running PE, watched.
    pub(super) fn device_read(
        &mut self,
        physical: u64,
        bytes: u8,
        now: u64,
    ) -> Result<u64, DeviceError> {
        self.host.speculation_barrier();
        let result = match self.devices.read_scalar(self.current, physical, bytes, now) {
            ScalarRead::Served(result) => result,
            ScalarRead::Canonical => self
                .serial_read_canonical(physical, DeviceReadShape::Scalar { bytes }, now)
                .and_then(|consumed| {
                    consumed.settlement?;
                    Ok(consumed.values[0])
                }),
            ScalarRead::Sample(cpu) => {
                let pc = self.processor_pc(cpu).ok_or_else(|| {
                    let reason = if matches!(self.peers.get(cpu), Some(processors::Peer::Remote)) {
                        // Keep the refusal stable for deduplication. The
                        // optional PE SOURCE report carries owner snapshots.
                        refuse!(
                            "debug PC sample: PE {cpu} belongs to a remote execution slot",
                            cpu
                        )
                    } else {
                        refuse!(
                            "debug PC sample: PE {cpu} has no powered execution state",
                            cpu
                        )
                    };
                    DeviceError::Refused(reason)
                })?;
                self.devices.observe_pc_sample(physical, pc);
                self.devices.read(self.current, physical, bytes, now)
            }
        };
        self.watch(physical, bytes, result.as_ref().ok().copied(), false);
        result
    }

    /// A device write by the running PE, watched.
    pub(super) fn device_write(
        &mut self,
        physical: u64,
        bytes: u8,
        value: u64,
        now: u64,
    ) -> Result<(), DeviceError> {
        let started = self
            .step_timing
            .is_some()
            .then(|| self.host.profile_ticks());
        let result = if self.serial_hold.is_some() {
            self.serial_write_canonical(physical, bytes, value, now)
        } else if self.parallel.is_some() {
            let current = self.current;
            let (devices, processors) = self.processor_write_parts();
            match devices.write_plain(current, physical, bytes, value, now, processors) {
                Some(result) => result,
                None => self.forward_write(physical, bytes, value),
            }
        } else {
            let current = self.current;
            let (devices, processors) = self.processor_write_parts();
            devices.write_with_processors(current, physical, bytes, value, now, processors)
        };
        if let Some(started) = started {
            let spent = self.host.profile_ticks().wrapping_sub(started);
            self.timed_write = Some([spent, 0, 0]);
        }
        self.watch(physical, bytes, result.is_ok().then_some(value), true);
        result
    }

    /// Add the ticks since `started` to part `part` of the device write
    /// being timed ([`step_timing::DeviceWrites`]).
    fn time_write(&mut self, part: usize, started: Option<u64>) {
        if let Some(started) = started {
            let spent = self.host.profile_ticks().wrapping_sub(started);
            if let Some(parts) = &mut self.timed_write {
                parts[part] += spent;
            }
        }
    }

    /// Native DMA/session service errors cannot become unknown-register
    /// substitutions. Enforce that boundary even
    /// when a provider returns an ordinary register error from its service.
    pub(super) fn device_transfer(&mut self, now: u64) -> Result<(), DeviceError> {
        self.device_settle(now).0
    }

    /// [`Self::device_transfer`], and the memory controllers' write lock
    /// the DMA left, from one call to the devices ([`Devices::settle_write`]).
    fn device_settle(
        &mut self,
        now: u64,
    ) -> (Result<(), DeviceError>, super::super::mmu::MemoryLocks) {
        self.observe_parallel_source(SourcePhase::DeviceAdvanceEnter);
        let work = self.work();
        // A just-accepted lock write precedes follow_memory_lock(), so the
        // devices give its owner's lock rather than the CPU's snapshot.
        let mut memory = HostMemory {
            host: &mut self.host,
            lock: self.cpu.memory_lock,
        };
        let timed = self
            .timed_write
            .is_some()
            .then(|| memory.host.profile_ticks());
        let settled = self.devices.settle_write(self.current, now, &mut memory);
        self.time_write(1, timed);
        // The call is this step's store's whatever other devices' DMA came
        // to: the caller reads its answer at the next instruction, and a call
        // left pending would run at a later step, on a record its stack has
        // since reused.
        if let Some(call) = settled.call {
            let timed = self
                .timed_write
                .is_some()
                .then(|| self.host.profile_ticks());
            self.run_gate_call(call, work, settled.lock);
            self.time_write(2, timed);
        }
        let result = match settled.transfer {
            Ok(()) => Ok(()),
            Err(DeviceError::Refused(reason) | DeviceError::TransferRefused(reason)) => {
                Err(DeviceError::TransferRefused(reason))
            }
            Err(DeviceError::NoDevice) => Err(DeviceError::TransferRefused(refuse!(
                "device DMA has no service model"
            ))),
        };
        self.observe_parallel_source(SourcePhase::DeviceAdvanceExit);
        (result, settled.after)
    }

    /// Run the call PE `current`'s store made through a call gate, if any,
    /// on the address space of the thread that made it.
    fn run_call(&mut self, work: u64, lock: super::super::mmu::MemoryLocks) {
        if let Some(call) = self.devices.take_call(self.current) {
            self.run_gate_call(call, work, lock);
        }
    }

    /// Run `call`, which PE `current`'s store made through a call gate.
    fn run_gate_call(&mut self, call: GateCall, work: u64, lock: super::super::mmu::MemoryLocks) {
        // A call from EL0 holds no lock the kernel times, so the count
        // need not wait for it: a compile or a GPU wait can take
        // milliseconds. Any other may hold one: the count waits for it
        // once it has been stopped `clock::STALL_MILLIS`.
        let at_el0 = self.cpu.pstate.el == 0;
        let released = self.clock.leave(|| self.host.counter(), work, at_el0);
        let mut caller = caller::CallerSpace {
            controls: self.cpu.controls(self.cpu.regime()),
            requester: mmu::Requester {
                unprivileged: true,
                ..self.cpu.requester()
            },
            lock,
            refused: None,
            host: &mut self.host,
        };
        let outcome = call.run(&mut caller);
        let refused = caller.refused;
        if released {
            self.clock.resume(|| self.host.counter(), work);
        }
        self.devices.finish_call(outcome);
        // The tables refuse what a kept translation still reaches: XNU
        // ages pages by clearing the access flag without a TLBI. The
        // frontend's retry touches every page of the call's spans, so
        // give up the caller's translations, as a TLB may at any time
        // and as TLBI ASIDE1 does, for each aged page's touch to take its
        // fault. Giving up only the refused page's would fix one page
        // per retry, and every retry repeats the call's work.
        if refused.is_some() {
            let current = self.cpu.tlbi_current();
            // A regime without ASIDs gives up everything.
            let everywhere = current
                .asid
                .map_or(mmu::tlbi::Everywhere::All, mmu::tlbi::Everywhere::NonGlobal);
            self.tlb_maintained(mmu::tlbi::Reach::All, everywhere, current.asid);
        }
    }

    /// The call a store whose write reported a failure made: what failed
    /// may have followed the register write, which queued the call.
    pub(super) fn run_refused_store_call(&mut self) {
        let work = self.work();
        let lock = self.devices.memory_lock();
        self.run_call(work, lock);
    }

    /// Carry out the devices' due work around an access. What a device
    /// cannot carry out is dropped, reported once, and leaves the access
    /// that found it due alone.
    pub(super) fn settle_devices(&mut self, now: u64) {
        self.settle_store(now);
    }

    /// [`Self::settle_devices`] after a store the devices took: the memory
    /// controllers' write lock it left, for [`Self::adopt_memory_lock`].
    pub(super) fn settle_store(&mut self, now: u64) -> super::super::mmu::MemoryLocks {
        let (result, lock) = self.device_settle(now);
        if let Err(error) = result {
            let advance = RequirementKind::DeviceAdvance(alloc::format!("{error:?}"));
            self.degrade(self.requirement(advance));
        }
        lock
    }

    pub(super) fn read_register(&self, register: Register) -> u64 {
        match register {
            Register::X(index) => self.cpu.state.x[usize::from(index)],
            Register::Sp => self.cpu.state.sp,
            Register::Zero => 0,
        }
    }

    pub(super) fn write_register(&mut self, register: Register, value: u64) {
        match register {
            Register::X(index) => self.cpu.state.x[usize::from(index)] = value,
            Register::Sp => self.cpu.state.sp = value,
            Register::Zero => {}
        }
    }

    /// (access address, writeback) for a scalar transfer.
    pub(super) fn scalar_address(
        &self,
        address: ScalarLoadAddress,
    ) -> (u64, Option<(Register, u64)>) {
        match address {
            ScalarLoadAddress::Literal { address } => (address, None),
            ScalarLoadAddress::Immediate {
                base,
                displacement,
                mode,
            } => {
                let base_value = self.read_register(base);
                let updated = base_value.wrapping_add(displacement as i64 as u64);
                match mode {
                    AddressMode::Offset => (updated, None),
                    AddressMode::PreIndex => (updated, Some((base, updated))),
                    AddressMode::PostIndex => (base_value, Some((base, updated))),
                }
            }
            ScalarLoadAddress::RegisterOffset {
                base,
                index,
                extend,
                shift,
            } => {
                let raw = self.read_register(index);
                let extended = match extend {
                    IndexExtend::Uxtw => raw & 0xffff_ffff,
                    IndexExtend::Sxtw => raw as u32 as i32 as i64 as u64,
                    IndexExtend::Lsl | IndexExtend::Sxtx => raw,
                };
                (
                    self.read_register(base).wrapping_add(extended << shift),
                    None,
                )
            }
        }
    }

    /// Complete one single-register scalar access to a device.
    pub(super) fn device_access(
        &mut self,
        word: u32,
        address: u64,
        physical: u64,
        write: bool,
    ) -> Step {
        let pc = self.cpu.pc;
        let mut settled_failure = None;
        let refuse = |machine: &mut Self, detail: &str| {
            machine.device_unexecuted(address, physical, write, detail.into())
        };
        let Ok(instruction) = Decoder.lift(GuestAddress(pc), word) else {
            return refuse(self, "device access by an undecodable instruction");
        };
        self.count_device_access(physical);
        if self.step_timing.is_some() {
            // The lock waits from here on are this access's.
            self.devices.take_lock_wait();
        }
        // A load the devices answer without their lock (a passive register)
        // changes no device, so the lines cached before it still hold: the
        // home PE would otherwise advance every device and tell every PE
        // again at its next interrupt check. Every call that can change a
        // device moves their generation ([`Devices::generation`]), so one
        // that did not move across the load left them as they were.
        let kept = self.device_lines.take();
        let generation = self.devices.generation();
        if let Some(load) = ScalarLoad::from_instruction(&instruction) {
            let (accessed, writeback) = self.scalar_address(load.address());
            if accessed != address {
                return refuse(self, "device load address differs from the fault");
            }
            let bytes = load.bytes();
            if !address.is_multiple_of(u64::from(bytes)) {
                return self.alignment_fault(address, false);
            }
            let now = self.now();
            let raw = match self.device_read(physical, bytes, now) {
                Ok(value) => {
                    if generation.is_some() && self.devices.generation() == generation {
                        self.device_lines = kept;
                    }
                    value
                }
                Err(error) => match self.device_refused(error, address, physical, bytes, None) {
                    None => 0,
                    Some(step) => return step,
                },
            };
            let bits = u32::from(bytes) * 8;
            let mut value = if bits == 64 {
                raw
            } else {
                raw & ((1u64 << bits) - 1)
            };
            if load.signed() && bits < 64 {
                let shift = 64 - bits;
                value = (((value << shift) as i64) >> shift) as u64;
            }
            if load.result_width() == Width::W32 {
                value &= 0xffff_ffff;
            }
            self.write_register(load.destination(), value);
            if let Some((base, updated)) = writeback {
                self.write_register(base, updated);
            }
        } else if let Some(store) = ScalarStore::from_instruction(&instruction) {
            let (accessed, writeback) = self.scalar_address(store.address());
            if accessed != address {
                return refuse(self, "device store address differs from the fault");
            }
            let bytes = store.bytes();
            if !address.is_multiple_of(u64::from(bytes)) {
                return self.alignment_fault(address, true);
            }
            let value = self.read_register(store.source_register());
            let now = self.now();
            let written = self.device_write(physical, bytes, value, now);
            // The write lock the store's DMA left, read in the same call.
            let lock = if written.is_ok() {
                Some(self.settle_store(now))
            } else {
                self.run_refused_store_call();
                None
            };
            if let Err(error) = written
                && let Some(step) =
                    self.device_refused(error, address, physical, bytes, Some(value))
            {
                return step;
            }
            if let Some((base, updated)) = writeback {
                self.write_register(base, updated);
            }
            match lock {
                Some(lock) => self.adopt_memory_lock(lock),
                None => self.follow_memory_lock(),
            }
        } else if let Operation::MemoryPair {
            transfer,
            bytes,
            data,
            base,
            displacement,
            mode,
        } = instruction.operation
        {
            // A pair is two accesses of `bytes` each, at ascending addresses
            // and in register order; Device memory faults on either
            // misaligned (Arm DDI 0487 B2.5.2).
            let first = self.immediate_address(base, displacement, mode);
            let bytes_wide = u64::from(bytes);
            if first != address {
                return refuse(self, "device pair address differs from the fault");
            }
            if !address.is_multiple_of(bytes_wide) {
                return self.alignment_fault(address, write);
            }
            // The smallest translation granule: the second element's PA
            // follows the first's only within one.
            const GRANULE: u64 = 0x1000;
            if physical % GRANULE + 2 * bytes_wide > GRANULE {
                return refuse(self, "device pair across a page");
            }
            let writeback = (mode != AddressMode::Offset).then(|| {
                let updated = self
                    .read_register(base)
                    .wrapping_add(displacement as i64 as u64);
                (base, updated)
            });
            let elements = [physical, physical + bytes_wide];
            let now = self.now();
            match transfer {
                MemoryTransfer::Load { signed, width } => {
                    let pair = if self.devices.requires_canonical_read(
                        self.current,
                        physical,
                        DeviceReadShape::Pair { bytes },
                    ) {
                        self.host.speculation_barrier();
                        self.serial_read_canonical(physical, DeviceReadShape::Pair { bytes }, now)
                            .map(Some)
                    } else {
                        self.host.speculation_barrier();
                        self.devices.read_pair(self.current, physical, bytes, now)
                    };
                    let raw_values = match pair {
                        Err(error) => {
                            self.watch(physical, bytes, None, false);
                            match self.device_refused(error, address, physical, bytes, None) {
                                None => [0; 2],
                                Some(step) => return step,
                            }
                        }
                        Ok(Some(pair)) => {
                            for (element, raw) in elements.into_iter().zip(pair.values) {
                                self.watch(element, bytes, Some(raw), false);
                            }
                            if let Err(error) = pair.settlement {
                                // Capture the original access PC before committing
                                // its data, writeback and instruction retirement.
                                let mut requirement = self.requirement(RequirementKind::Device {
                                    physical,
                                    write,
                                    detail: format!(
                                        "consumed device pair: board settlement failed: {error:?}"
                                    ),
                                });
                                requirement.address = Some(address);
                                settled_failure = Some(requirement);
                            }
                            pair.values
                        }
                        Ok(None) => {
                            let mut raw_values = [0; 2];
                            for (raw, element) in raw_values.iter_mut().zip(elements) {
                                *raw = match self.device_read(element, bytes, now) {
                                    Ok(raw) => raw,
                                    Err(error) => {
                                        let at = address + (element - physical);
                                        match self.device_refused(error, at, element, bytes, None) {
                                            None => 0,
                                            Some(step) => return step,
                                        }
                                    }
                                };
                            }
                            raw_values
                        }
                    };
                    let values = raw_values.map(|raw| {
                        let mut value = match (bytes, signed) {
                            (8, _) => raw,
                            (4, true) => raw as u32 as i32 as i64 as u64,
                            _ => raw & 0xffff_ffff,
                        };
                        if width == Width::W32 {
                            value &= 0xffff_ffff;
                        }
                        value
                    });
                    for (register, value) in data.into_iter().zip(values) {
                        self.write_register(register, value);
                    }
                }
                MemoryTransfer::Store => {
                    let values = data.map(|register| self.read_register(register));
                    let mut lock = None;
                    for (element, value) in elements.into_iter().zip(values) {
                        let written = self.device_write(element, bytes, value, now);
                        lock = if written.is_ok() {
                            Some(self.settle_store(now))
                        } else {
                            self.run_refused_store_call();
                            None
                        };
                        if let Err(error) = written {
                            let at = address + (element - physical);
                            if let Some(step) =
                                self.device_refused(error, at, element, bytes, Some(value))
                            {
                                return step;
                            }
                        }
                    }
                    match lock {
                        Some(lock) => self.adopt_memory_lock(lock),
                        None => self.follow_memory_lock(),
                    }
                }
            }
            if let Some((base, updated)) = writeback {
                self.write_register(base, updated);
            }
        } else {
            return refuse(
                self,
                "device access by other than a scalar or pair integer transfer",
            );
        }
        self.cpu.pc = pc;
        self.next()?;
        // The consumed data is retired; a failed board settlement after it
        // is reported once.
        if let Some(requirement) = settled_failure {
            self.degrade(requirement);
        }
        match self.devices.take_reset() {
            Some(reason) => Err(Stop::Reset(reason)),
            None => Ok(()),
        }
    }

    /// Give every PE the memory controller's write lock after a device
    /// write changed it.
    pub(super) fn follow_memory_lock(&mut self) {
        let lock = self.devices.memory_lock();
        self.adopt_memory_lock(lock);
    }

    /// [`Self::follow_memory_lock`] with the lock the devices gave.
    pub(super) fn adopt_memory_lock(&mut self, lock: super::super::mmu::MemoryLocks) {
        if lock == self.cpu.memory_lock {
            return;
        }
        self.cpu.memory_lock = lock;
        for peer in &mut self.peers {
            if let processors::Peer::Parked(parked) = peer {
                parked.cpu_mut().memory_lock = lock;
            }
        }
        self.translations_changed();
    }

    pub(super) fn alignment_fault(&mut self, address: u64, write: bool) -> Step {
        let class = if self.cpu.route() > self.cpu.pstate.el {
            Class::DataAbortLower
        } else {
            Class::DataAbortSame
        };
        let mut exception =
            Exception::synchronous(class, data_abort_iss(0b10_0001, write), self.cpu.pc);
        exception.fault_address = Some(address);
        self.take(exception);
        Ok(())
    }

    /// A device access the devices refused or no device answers: `None` when it
    /// completes degraded, reading 0 or dropping the write, else the step
    /// the instruction ends with. A survey answers every register access so
    /// and records it ([`Self::begin_survey`]); otherwise [`Self::degrade`]
    /// decides: an address no device covers takes the synchronous
    /// external abort the fabric raises there, and a register, setting or
    /// effect a device does not carry out completes degraded: the device
    /// keeps what it accepted and nothing is invented for the rest.
    pub(super) fn device_refused(
        &mut self,
        error: DeviceError,
        address: u64,
        physical: u64,
        bytes: u8,
        written: Option<u64>,
    ) -> Option<Step> {
        if self.surveyed(physical, bytes, written, &error) {
            return None;
        }
        let write = written.is_some();
        let unmapped = error == DeviceError::NoDevice;
        let detail = match error {
            DeviceError::NoDevice => "no device model".into(),
            DeviceError::Refused(detail) | DeviceError::TransferRefused(detail) => detail,
        };
        let mut requirement = self.requirement(RequirementKind::Device {
            physical,
            write,
            detail: detail.to_string(),
        });
        requirement.address = Some(address);
        // Each written value is the same refusal (`degrade`).
        requirement.written = written;
        self.degrade(requirement);
        unmapped.then(|| self.external_abort(address, write))
    }

    /// A device access in a form the engine cannot carry out (`detail`),
    /// reported once per register, direction and refusal: it takes the
    /// synchronous external abort without reaching the device.
    pub(super) fn device_unexecuted(
        &mut self,
        address: u64,
        physical: u64,
        write: bool,
        detail: String,
    ) -> Step {
        let mut requirement = self.requirement(RequirementKind::Device {
            physical,
            write,
            detail: detail.to_string(),
        });
        requirement.address = Some(address);
        self.degrade(requirement);
        self.external_abort(address, write)
    }

    /// The synchronous external abort (DFSC 0b010000) a data access takes;
    /// from EL0 or EL1 it goes to EL2 under HCR_EL2.TEA (Arm
    /// SyncExternalAbortTarget).
    pub(super) fn external_abort(&mut self, address: u64, write: bool) -> Step {
        let iss = data_abort_iss(SYNCHRONOUS_EXTERNAL_ABORT, write);
        let exception = self.external_abort_exception(
            [Class::DataAbortLower, Class::DataAbortSame],
            iss,
            address,
        );
        self.take(exception);
        Ok(())
    }

    /// The synchronous external abort of the access at `address` by the
    /// instruction at the PC, of the lower or same-level `classes`.
    pub(super) fn external_abort_exception(
        &self,
        classes: [Class; 2],
        iss: u32,
        address: u64,
    ) -> Exception {
        let target = if self.cpu.pstate.el < 2 && self.cpu.hcr(crate::machine::cpu::HCR_TEA) {
            2
        } else {
            self.cpu.route()
        };
        let class = if target > self.cpu.pstate.el {
            classes[0]
        } else {
            classes[1]
        };
        let mut exception = Exception::synchronous(class, iss, self.cpu.pc).to(target);
        exception.fault_address = Some(address);
        exception
    }

    /// Translate a data address for the current EL and regime.
    pub(super) fn translate_data(&mut self, address: u64, access: Access) -> Translated {
        let controls = self.cpu.controls(self.cpu.regime());
        let requester = self.cpu.requester();
        mmu::translate(
            &controls,
            address,
            access,
            requester,
            &mut RamTables(&mut self.host),
        )
    }

    /// The data abort a stage-1 fault raises at the current EL.
    pub(super) fn data_abort_exception(
        &mut self,
        address: u64,
        fault: mmu::Fault,
        write: bool,
        cache: bool,
    ) -> Exception {
        // A stage-2 fault of the instruction's own access, not of a stage-1
        // walk, carries its instruction syndrome where Arm defines one.
        let instruction = match fault.second_stage {
            Some(mmu::SecondStage { walk: false, .. }) if !cache => self
                .instruction_word()
                .and_then(crate::machine::exception::load_store_syndrome)
                .unwrap_or(0),
            _ => 0,
        };
        self.abort(
            fault,
            [Class::DataAbortLower, Class::DataAbortSame],
            fault.data_syndrome(write, cache) | instruction,
            address,
            self.cpu.pc,
        )
    }

    /// The word at the PC, through an instruction fetch of the current
    /// regime: `None` where it no longer translates to RAM.
    fn instruction_word(&mut self) -> Option<u32> {
        let controls = self.cpu.controls(self.cpu.regime());
        let requester = self.cpu.requester();
        let Translated::Mapped(translation) = mmu::translate(
            &controls,
            self.cpu.pc,
            Access::Execute,
            requester,
            &mut RamTables(&mut self.host),
        ) else {
            return None;
        };
        let mut bytes = [0; 4];
        self.host
            .read_ram(translation.physical, &mut bytes)
            .then(|| self.admit(u32::from_le_bytes(bytes)))
    }

    /// The abort for a translation fault (Arm `AArch64_Abort` and
    /// `AArch64_ReportException`): a stage-2 fault goes to EL2 whatever the
    /// routing, and records HPFAR_EL2 when its IPA is valid; `classes` are
    /// the lower-EL and same-EL classes, chosen by the target.
    pub(super) fn abort(
        &mut self,
        fault: mmu::Fault,
        classes: [Class; 2],
        iss: u32,
        address: u64,
        return_address: u64,
    ) -> Exception {
        // A synchronous external abort from EL0 or EL1 goes to EL2 under
        // HCR_EL2.TEA (Arm SyncExternalAbortTarget).
        let external = fault.kind == mmu::FaultKind::ExternalOnWalk;
        let target = if fault.to_el2()
            || external && self.cpu.pstate.el < 2 && self.cpu.hcr(crate::machine::cpu::HCR_TEA)
        {
            2
        } else {
            self.cpu.route()
        };
        let class = if target > self.cpu.pstate.el {
            classes[0]
        } else {
            classes[1]
        };
        if let Some(hpfar) = fault.hpfar() {
            self.cpu
                .registers
                .set_raw(SystemRegister::HPFAR_EL2, 0, hpfar);
        }
        let mut exception = Exception::synchronous(class, iss, return_address).to(target);
        exception.fault_address = Some(address);
        exception
    }

    pub(super) fn data_abort(
        &mut self,
        address: u64,
        fault: mmu::Fault,
        write: bool,
        cache: bool,
    ) -> Step {
        let exception = self.data_abort_exception(address, fault, write, cache);
        self.take(exception);
        Ok(())
    }

    /// Read `bytes.len()` bytes of RAM at `address` in the running PE's
    /// current regime, for diagnostics after a stop: a read that does not
    /// translate or reaches no RAM is `false`, and nothing is faulted or
    /// changed. The bytes must lie in one fetch page.
    pub fn read_virtual(&mut self, address: u64, bytes: &mut [u8]) -> bool {
        let controls = self.cpu.controls(self.cpu.regime());
        let requester = self.cpu.requester();
        let last = address.wrapping_add(bytes.len() as u64).wrapping_sub(1);
        if bytes.is_empty() || address & !(FETCH_PAGE - 1) != last & !(FETCH_PAGE - 1) {
            return false;
        }
        match mmu::translate(
            &controls,
            address,
            Access::Read,
            requester,
            &mut RamTables(&mut self.host),
        ) {
            Translated::Mapped(translation) => self.host.read_ram(translation.physical, bytes),
            _ => false,
        }
    }

    /// Read `bytes.len()` bytes of guest RAM at physical `address`, for
    /// diagnostics after a stop (the panic region): `false` when it reaches
    /// no RAM; nothing is faulted or changed.
    pub fn read_physical(&mut self, address: u64, bytes: &mut [u8]) -> bool {
        self.host.read_ram(address, bytes)
    }
}
