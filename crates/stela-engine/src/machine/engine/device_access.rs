//! One recovered cohort, one stop, and ordered consuming bus accesses.
use super::*;
use crate::memory::{
    domain::MemoryExecutionLoan,
    system_ram::{DescriptorReader, REGISTERED_FRAME_BYTES, RamPageKind, RegisteredSystemRam},
};
use shared_memory::RecoveredDeviceAccess;
use stela_compiler::context::memory::MemoryStatus;

/// The most instructions a machine completes as device sites: one past
/// them takes the fault path, as before it learned any.
const DEVICE_SITES: usize = 4096;
/// A device site whose accesses reach RAM this many times, and eight times
/// as often as a device, is left to native code again for good.
const RAM_SITE_ACCESSES: u32 = 64;

/// An instruction that reached a device register from native code. The
/// engine completes it, as the fault would have, without the fault: blocks
/// end before it, and the block at it is a stub that exits to the engine
/// (`Machine::complete_device_site`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct DeviceSite {
    /// The engine completed a device access there, and its access reached
    /// RAM instead (run natively, one instruction).
    device: u32,
    ram: u32,
    /// Left to native code again: it reaches RAM far more often.
    retired: bool,
}

/// A load or store the engine completes at a device site: its address,
/// direction and the bytes from it it reaches. Others run natively.
struct SiteAccess {
    address: u64,
    write: bool,
    bytes: u64,
    alignment: u64,
}

pub(super) fn ram_refused() -> RequirementKind {
    RequirementKind::NativePreparation("could not verify registered RAM binding".into())
}

/// One registered scalar or vector element, pinned across its walk and copy.
pub(super) struct OperandRam {
    ram: Arc<RegisteredSystemRam>,
    execution: MemoryExecutionLoan,
}

impl OperandRam {
    pub(super) fn tables(&self) -> SiteTables<'_> {
        SiteTables::new(&self.ram)
    }

    pub(super) fn resident(
        &self,
        host: &impl Host,
        pieces: impl Iterator<Item = (u64, usize)>,
    ) -> Result<bool, RequirementKind> {
        let binding = self
            .ram
            .cpu_binding(&self.execution)
            .map_err(|_| ram_refused())?;
        let mut resident = true;
        let frame_bytes = REGISTERED_FRAME_BYTES as u64;
        for (physical, length) in pieces {
            let frame = physical & !(frame_bytes - 1);
            if physical - frame + length as u64 > frame_bytes {
                return Err(ram_refused());
            }
            match binding
                .classify_page(frame, REGISTERED_FRAME_BYTES)
                .map_err(|_| ram_refused())?
            {
                RamPageKind::Resident => {}
                RamPageKind::Nonresident { frame } => {
                    return Err(RequirementKind::NonresidentRam { frame });
                }
                RamPageKind::Unmapped => {
                    if (0..frame_bytes)
                        .step_by(FETCH_PAGE as usize)
                        .any(|offset| host.device_site_has_retained_ram(frame + offset))
                    {
                        return Err(ram_refused());
                    }
                    resident = false;
                }
            }
        }
        Ok(resident)
    }

    /// Copy previously admitted pieces, retaining the completed prefix on failure.
    ///
    /// # Safety
    /// The staging slice is private and disjoint from registered RAM and all
    /// live native frames. The pieces were admitted under this same loan.
    pub(super) unsafe fn transfer(
        &self,
        host: &mut impl Host,
        pieces: impl Iterator<Item = (u64, usize)>,
        staging: &mut [u8],
        write: bool,
    ) -> Result<(), RequirementKind> {
        host.speculation_barrier();
        let mut offset = 0;
        for (physical, length) in pieces {
            let part = &mut staging[offset..offset + length];
            // SAFETY: the canonical owner remains pinned and part is private staging.
            let result = unsafe {
                if write {
                    self.ram.publish_physical(physical, part)
                } else {
                    self.ram.read_physical_progress(physical, part)
                }
            };
            if let Err(fault) = result {
                return Err(RequirementKind::RamTransfer {
                    physical,
                    write,
                    completed: offset + fault.completed,
                    status: fault.status,
                });
            }
            offset += length;
        }
        Ok(())
    }
}

/// Keep descriptor failures that the architectural walker cannot express.
pub(super) struct SiteTables<'a> {
    ram: &'a RegisteredSystemRam,
    descriptors: DescriptorReader<'a>,
    refusal: Option<RequirementKind>,
}

impl<'a> SiteTables<'a> {
    pub(super) fn new(ram: &'a RegisteredSystemRam) -> Self {
        Self {
            ram,
            descriptors: ram.descriptor_reader(),
            refusal: None,
        }
    }

    pub(super) fn translate(
        &mut self,
        controls: &mmu::Controls,
        address: u64,
        access: mmu::Access,
        requester: mmu::Requester,
    ) -> Result<Translated, RequirementKind> {
        let outcome = mmu::translate(controls, address, access, requester, self);
        self.outcome(outcome)
    }

    pub(super) fn outcome(&mut self, outcome: Translated) -> Result<Translated, RequirementKind> {
        match self.refusal.take() {
            Some(reason) => Err(reason),
            None => Ok(outcome),
        }
    }

    fn descriptor<T>(&mut self, physical: u64, result: Result<T, MemoryStatus>) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(MemoryStatus::Translation) => None,
            Err(status) => {
                if self.refusal.is_none() {
                    self.refusal = Some(if status == MemoryStatus::Nonresident {
                        self.ram
                            .nonresident_frame(physical, 8)
                            .map(|frame| RequirementKind::NonresidentRam { frame })
                            .unwrap_or_else(ram_refused)
                    } else {
                        ram_refused()
                    });
                }
                None
            }
        }
    }
}

impl mmu::TableMemory for SiteTables<'_> {
    fn read_descriptor(&mut self, physical: u64) -> Option<u64> {
        let result = self.descriptors.read_descriptor(physical);
        self.descriptor(physical, result)
    }

    fn update_descriptor(&mut self, physical: u64, old: u64, new: u64) -> Option<bool> {
        let result = self.descriptors.update_descriptor(physical, old, new);
        self.descriptor(physical, result)
    }
}

impl<H: Host, D: Devices> Machine<H, D> {
    pub(super) fn operand_ram(&self) -> Result<Option<OperandRam>, RequirementKind> {
        let registered = self.host.device_site_ram().map_err(|_| ram_refused())?;
        if self.cpu.canonical_memory.as_ref().is_some_and(|memory| {
            !registered
                .as_ref()
                .is_some_and(|owner| Arc::ptr_eq(&owner.ram, memory.ram_owner()))
        }) {
            return Err(ram_refused());
        }
        let Some(DeviceSiteRam {
            ram,
            expected_generation,
        }) = registered
        else {
            return Ok(None);
        };
        let execution = ram.cpu_execution_loan().map_err(|_| ram_refused())?;
        let binding = ram.cpu_binding(&execution).map_err(|_| ram_refused())?;
        if expected_generation.is_some_and(|expected| binding.generation() != expected) {
            return Err(ram_refused());
        }
        Ok(Some(OperandRam { ram, execution }))
    }

    /// The physical address of the instruction at `pc` of block `index`,
    /// unless the block is a single instruction's.
    pub(super) fn device_site_of(&self, index: usize, pc: u64) -> Option<u64> {
        let block = &self.code.blocks[index];
        (!self.variants[block.variant.0 as usize].single)
            .then(|| block.physical.wrapping_add(pc.wrapping_sub(block.pc)))
    }

    /// Learn that the instruction at `physical` reached a device register,
    /// after the engine completed it: its page's blocks compile again,
    /// ending before it.
    pub(super) fn learn_device_site(&mut self, physical: u64) {
        if self.device_sites.contains_key(&physical) || self.device_sites.len() >= DEVICE_SITES {
            return;
        }
        self.device_sites.insert(physical, DeviceSite::default());
        self.invalidate_code_page(physical);
    }

    /// Whether the instruction at `physical` is a device site.
    pub(super) fn device_site(&self, physical: u64) -> bool {
        !self.device_sites.is_empty()
            && self
                .device_sites
                .get(&physical)
                .is_some_and(|site| !site.retired)
    }

    /// The bytes of the window at `physical` before its first device site,
    /// when one lies in its first `bytes` past its first instruction.
    pub(super) fn before_device_site(&self, physical: u64, bytes: usize) -> Option<usize> {
        if self.device_sites.is_empty() {
            return None;
        }
        self.device_sites
            .range(physical + INSTRUCTION..physical + bytes as u64)
            .find(|(_, site)| !site.retired)
            .map(|(&site, _)| (site - physical) as usize)
    }

    /// Whether the fetch page holding `physical` has a device site: its
    /// blocks end in ways no entry context names, so the store neither gives
    /// nor takes them.
    pub(super) fn page_has_device_sites(&self, physical: u64) -> bool {
        let frame = physical & !(FETCH_PAGE - 1);
        !self.device_sites.is_empty()
            && self
                .device_sites
                .range(frame..frame + FETCH_PAGE)
                .any(|(_, site)| !site.retired)
    }

    /// Whether block `index`, exiting with a System exit at `pc`, is a device
    /// site's stub. The block says so, not this PE's sites: on a shared
    /// board a PE runs the stubs of sites other PEs learnt.
    pub(super) fn device_stub(&self, index: usize, pc: u64) -> bool {
        let block = &self.code.blocks[index];
        block.device_stub && block.pc == pc
    }

    /// Install the stub of the device site at `pc` and `physical`: native
    /// code that exits to the engine at once, with the site's word, admitted,
    /// as its one instruction.
    pub(super) fn install_device_stub(
        &mut self,
        pc: u64,
        physical: u64,
        variant: Variant,
        source: &[u8],
    ) -> Result<Fetched, Stop> {
        let block = stela_compiler::block::CompiledBlock {
            native: stela_compiler::NativeBlock {
                x86_64_code: chain::direct_exit(GuestAddress(pc), ExitKind::System).to_vec(),
                source_map: alloc::vec![(GuestAddress(pc), 0)],
            },
            source_bytes: INSTRUCTION as usize,
            direct_exits: Vec::new(),
            fault_recovery: Vec::new(),
            self_loop: false,
        };
        let preparation = self.preparation(variant);
        let emission = Arc::new(super::stored::Emission::new(
            source[..INSTRUCTION as usize].to_vec(),
            block,
            preparation,
        ));
        self.installing_stub = true;
        let installed = self.install(
            pc,
            physical,
            variant,
            source,
            recovery::Template::compiled(emission),
        );
        self.installing_stub = false;
        installed
    }

    /// The access the instruction `word` at `pc` makes, when the engine can
    /// complete it at a device as the fault path would: a scalar or pair
    /// integer load or store through a register other than SP, which is not
    /// unprivileged (LDTR, STTR).
    fn site_access(&self, pc: u64, word: u32) -> Option<SiteAccess> {
        // LDTR*, STTR*: their permission checks are EL0's.
        if word & 0x3b20_0c00 == 0x3800_0800 {
            return None;
        }
        let instruction = Decoder.lift(GuestAddress(pc), word).ok()?;
        let base = |address: &ScalarLoadAddress| match *address {
            ScalarLoadAddress::Immediate { base, .. }
            | ScalarLoadAddress::RegisterOffset { base, .. } => Some(base),
            ScalarLoadAddress::Literal { .. } => None,
        };
        let (address, write, bytes, alignment, base) =
            if let Some(load) = ScalarLoad::from_instruction(&instruction) {
                let address = load.address();
                (
                    self.scalar_address(address).0,
                    false,
                    u64::from(load.bytes()),
                    u64::from(load.bytes()),
                    base(&address)?,
                )
            } else if let Some(store) = ScalarStore::from_instruction(&instruction) {
                let address = store.address();
                (
                    self.scalar_address(address).0,
                    true,
                    u64::from(store.bytes()),
                    u64::from(store.bytes()),
                    base(&address)?,
                )
            } else if let Operation::MemoryPair {
                transfer,
                bytes,
                base,
                displacement,
                mode,
                ..
            } = instruction.operation
            {
                let write = matches!(transfer, MemoryTransfer::Store);
                let address = self.immediate_address(base, displacement, mode);
                (address, write, 2 * u64::from(bytes), u64::from(bytes), base)
            } else {
                return None;
            };
        (base != Register::Sp).then_some(SiteAccess {
            address,
            write,
            bytes,
            alignment,
        })
    }

    fn device_site_target(&mut self, access: &SiteAccess) -> Result<Option<u64>, RequirementKind> {
        let kind = if access.write {
            mmu::Access::Write
        } else {
            mmu::Access::Read
        };
        let registered = self.host.device_site_ram().map_err(|_| ram_refused())?;
        let canonical = self
            .cpu
            .canonical_memory
            .as_ref()
            .map(|memory| memory.ram_owner());
        if canonical.is_some_and(|ram| {
            self.serial_hold.is_none()
                || !registered
                    .as_ref()
                    .is_some_and(|site| Arc::ptr_eq(&site.ram, ram))
        }) {
            return Err(ram_refused());
        }
        let Some(DeviceSiteRam {
            ram,
            expected_generation,
        }) = registered
        else {
            let Translated::Mapped(translation) = self.translate_data(access.address, kind) else {
                return Ok(None);
            };
            let last = access.address.wrapping_add(access.bytes - 1);
            let page = translation.physical & !(FETCH_PAGE - 1);
            return Ok((!self.host.read_ram(page, &mut [0])
                && !self.watch.overlaps(access.address, last))
            .then_some(translation.physical));
        };
        // The promoted path consumes only a one-page operand. Other shapes
        // take the original native checks, including architectural alignment.
        let Some(last) = access.address.checked_add(access.bytes - 1) else {
            return Ok(None);
        };
        if !access.address.is_multiple_of(access.alignment) || (access.address ^ last) >= FETCH_PAGE
        {
            return Ok(None);
        }
        let execution = ram.cpu_execution_loan().map_err(|_| ram_refused())?;
        let binding = ram.cpu_binding(&execution).map_err(|_| ram_refused())?;
        if expected_generation.is_some_and(|expected| binding.generation() != expected) {
            return Err(ram_refused());
        }
        // The site remembers only the instruction, so translate its current
        // VA and classify the resulting GPA under this same execution loan.
        let translation = SiteTables::new(&ram).translate(
            &self.cpu.controls(self.cpu.regime()),
            access.address,
            kind,
            self.cpu.requester(),
        )?;
        let Translated::Mapped(translation) = translation else {
            return Ok(None);
        };
        let page = translation.physical & !(FETCH_PAGE - 1);
        if translation.physical - page + access.bytes > FETCH_PAGE {
            return Ok(None);
        }
        match binding
            .classify_page(page, FETCH_PAGE as usize)
            .map_err(|_| ram_refused())?
        {
            RamPageKind::Nonresident { frame } => Err(RequirementKind::NonresidentRam { frame }),
            RamPageKind::Resident => Ok(None),
            RamPageKind::Unmapped if self.host.device_site_has_retained_ram(page) => {
                Err(ram_refused())
            }
            RamPageKind::Unmapped => {
                Ok((!self.watch.overlaps(access.address, last)).then_some(translation.physical))
            }
        }
        // The loan ends before synchronous device work, which may borrow
        // the RAM service. The stopped serialized cohort stays held, and
        // the registered GPA geometry remains fixed for this boot.
    }

    /// Complete the instruction at the device site `pc`, block `index`'s
    /// only one, as its native fault would have: when its access reaches a
    /// device, through [`Self::device_access`]; else (RAM, a fault, a
    /// watchpoint, or a form the engine does not complete) by running it
    /// natively as a block of its own, which takes that path itself.
    pub(super) fn complete_device_site(&mut self, index: usize, pc: u64) -> Step {
        let physical = self.code.blocks[index].physical;
        let word = self.code.blocks[index].terminal;
        let device = match self.site_access(pc, word) {
            Some(access) => match self.device_site_target(&access) {
                Ok(target) => target.map(|target| (access, target)),
                Err(reason) => return Err(self.unsupported(reason)),
            },
            None => None,
        };
        let site = self.device_sites.entry(physical).or_default();
        match device {
            Some((access, target)) => {
                site.device = site.device.saturating_add(1);
                self.outside_run(|machine| {
                    machine.completing(pc, |machine| {
                        machine.device_access(word, access.address, target, access.write)
                    })
                })
            }
            None => {
                site.ram = site.ram.saturating_add(1);
                if site.ram >= RAM_SITE_ACCESSES && site.ram / 8 > site.device {
                    site.retired = true;
                    self.invalidate_code_page(physical);
                }
                self.run_one_natively(pc)
            }
        }
    }

    /// Run the instruction at `pc` as a block of its own, as the step of a
    /// single instruction does.
    fn run_one_natively(&mut self, pc: u64) -> Step {
        match self.fetch_block(pc, true)? {
            Fetched::Block(index) => self.execute(index, 0),
            Fetched::Exception(exception) => {
                self.take(exception);
                Ok(())
            }
            Fetched::Checked(word) => self.checked_instruction(pc, word),
        }
    }

    pub fn pending_native_device_access_count(&self) -> usize {
        usize::from(self.pending_device_read.is_some() || self.pending_device_write.is_some())
            + self.pending_device_tail.len()
    }
    /// Diagnostic only; the frame/generation remains private.
    pub fn pending_native_device_access(&self) -> Option<Stop> {
        self.pending_device_read
            .as_ref()
            .map(device_read::Pending::stop)
            .or_else(|| {
                self.pending_device_write
                    .as_ref()
                    .map(device_write::Pending::stop)
            })
    }
}
impl<H: Host, T: NativeTransport, D: Devices> Machine<ServiceHost<H, T>, D> {
    pub(super) fn queue_device_accesses(
        &mut self,
        accesses: Vec<RecoveredDeviceAccess>,
        held: publication::StoppedWorkers,
    ) -> Stop {
        self.pending_device_tail = accesses.into();
        self.queue_next_device_access(held)
    }
    fn queue_next_device_access(&mut self, held: publication::StoppedWorkers) -> Stop {
        match self
            .pending_device_tail
            .pop_front()
            .expect("actual pending cohort")
        {
            RecoveredDeviceAccess::Read(read) => self.queue_device_read(read, held),
            RecoveredDeviceAccess::Write(write) => self.queue_device_write(write, held),
        }
    }
    pub(super) fn finish_device_access(&mut self, held: publication::StoppedWorkers) -> Step {
        if self.pending_device_tail.is_empty() {
            self.release_native_services(held);
            Ok(())
        } else {
            Err(self.queue_next_device_access(held))
        }
    }
    /// Consume only the current front; no caller-supplied request,
    /// recovered frame, data or worker generation is accepted.
    ///
    /// # Safety
    /// The complete closed synchronous native producer contract remains valid
    /// through every pending request. Each backend keeps its prepared effect
    /// and ACK inside the same real worker generation.
    pub unsafe fn complete_native_device_access(&mut self) -> Step {
        if self.pending_device_read.is_some() {
            // SAFETY: same pending source and producer contract.
            unsafe { self.complete_device_read() }
        } else if self.pending_device_write.is_some() {
            // SAFETY: same pending source and producer contract.
            unsafe { self.complete_device_write() }
        } else {
            Err(self.unsupported(RequirementKind::Native(
                "no recovered device access is pending",
            )))
        }
    }
}

#[cfg(test)]
mod tests;
