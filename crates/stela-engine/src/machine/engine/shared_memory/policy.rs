//! Source-checked ARM data translation with architectural TLB maintenance.
use super::*;
use crate::memory::{
    backing::AccessKind,
    domain::{DataAccess, DataOperation, DataPart, DataPolicy, DataRequest},
    system_ram::DescriptorReader,
};
use stela_compiler::context::memory::{
    InstructionRequirements, MemoryOperationClass, SpAlignmentRequirement, flags,
};
mod tlb;
#[cfg(feature = "native-profile")]
pub use tlb::Statistics as DataTlbStatistics;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Failure {
    /// A reached, aligned Device read translated successfully but has no RAM
    /// backing. No device callback/effect has run; only a true recovered frame
    /// may turn this diagnostic into a stopped consuming continuation.
    DeviceRead {
        request: DataRequest,
        address: u64,
        physical: u64,
        bytes: usize,
    },
    DeviceWrite {
        request: DataRequest,
        address: u64,
        physical: u64,
        bytes: usize,
    },
    Fault {
        request: DataRequest,
        address: u64,
        access: AccessKind,
        fault: mmu::Fault,
    },
    Unsupported {
        request: DataRequest,
        address: u64,
        reason: Refusal,
    },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    UnalignedQuantity,
    Translation(mmu::Unsupported),
    /// Exact effective output of the reached walk, kept only on refusal.
    /// The diagnostic must not require another walk after the source stopped.
    MemoryAttributes {
        physical: u64,
        memory: mmu::MemoryType,
        attributes: u8,
        shareability: u8,
    },
    PhysicalGeometry,
}

pub(super) struct Policy {
    controls: Controls,
    ram: Arc<RegisteredSystemRam>,
    pub(super) failure: Option<Failure>,
    tlb: tlb::DataTlb,
}
impl Policy {
    pub(super) fn new(controls: Controls, ram: Arc<RegisteredSystemRam>) -> Self {
        Self {
            controls,
            ram,
            failure: None,
            tlb: tlb::DataTlb::default(),
        }
    }
    /// Only the stopped owning Machine supplies its architectural epoch.
    pub(super) fn refresh(&mut self, controls: Controls, generation: u64) {
        let changed = self.controls.translation != controls.translation
            || self.controls.requester != controls.requester;
        self.tlb.refresh(generation, changed);
        self.controls = controls;
        self.failure = None;
    }
    /// A stopped helper without a Machine epoch must use fresh walks.
    pub(super) fn reset_controls(&mut self, controls: Controls) {
        self.tlb.disable();
        self.controls = controls;
        self.failure = None;
    }
    #[cfg(feature = "native-profile")]
    pub(super) fn data_tlb_statistics(&self) -> DataTlbStatistics {
        self.tlb.statistics()
    }
    fn unsupported(&mut self, request: DataRequest, address: u64, reason: Refusal) -> MemoryStatus {
        self.failure.get_or_insert(Failure::Unsupported {
            request,
            address,
            reason,
        });
        MemoryStatus::Unsupported
    }
    fn check_physical_result(
        &mut self,
        request: DataRequest,
        address: u64,
        result: Result<(), MemoryStatus>,
    ) -> Result<(), (MemoryStatus, u64)> {
        result.map_err(|status| {
            let status = match status {
                MemoryStatus::Domain | MemoryStatus::Nonresident => status,
                _ => self.unsupported(request, address, Refusal::PhysicalGeometry),
            };
            (status, address)
        })
    }

    fn fault(
        &mut self,
        request: DataRequest,
        address: u64,
        access: AccessKind,
        fault: mmu::Fault,
    ) -> MemoryStatus {
        self.failure.get_or_insert(Failure::Fault {
            request,
            address,
            access,
            fault,
        });
        match fault.kind {
            mmu::FaultKind::Permission => MemoryStatus::Permission,
            mmu::FaultKind::Alignment => MemoryStatus::Alignment,
            mmu::FaultKind::TagCheck => MemoryStatus::TagCheck,
            _ => MemoryStatus::Translation,
        }
    }
}
fn shape(request: DataRequest) -> mmu::alignment::Shape {
    use mmu::alignment::Kind;
    mmu::alignment::Shape {
        bytes: request.bytes as u64,
        pair: request.flags & flags::PAIR != 0,
        kind: match request.operation {
            DataOperation::Ordinary => Kind::Plain,
            DataOperation::Exclusive => Kind::Exclusive,
            DataOperation::Atomic => Kind::Atomic,
            DataOperation::Ordered => Kind::Ordered,
            DataOperation::AllocationTag => Kind::Plain,
        },
    }
}
impl DataPolicy for Policy {
    fn device_ram_ordering(&self) -> bool {
        self.controls.cold_m0()
    }
    fn validate_instructions(&self, required: InstructionRequirements) -> Result<(), MemoryStatus> {
        self.controls.instructions(required)
    }
    fn validate_execution(&self, required: u32) -> Result<(), MemoryStatus> {
        if required & !flags::KNOWN != 0 {
            return Err(MemoryStatus::InvalidRequest);
        }
        if required & (flags::SME | flags::SVE_PREDICATED) != 0
            || required & flags::ASIMD != 0 && !self.controls.asimd
        {
            return Err(MemoryStatus::Unsupported);
        }
        Ok(())
    }
    fn validate_operation_execution(
        &self,
        class: MemoryOperationClass,
    ) -> Result<(), MemoryStatus> {
        // Ordered scalar transfers use the same checked physical RAM path as
        // ordinary quantities. The cold callback keeps the machine's Device
        // serializer and MFENCE pair; typed ordered staging adds acquire/release
        // semantics. Arm 2026-06 LDAPRB/LDAR/STLR use Mem with acquire/release
        // descriptors; aligned Device memory is not an instruction refusal.
        // This does not check LSE or exclusive Device transfers.
        if self.controls.cold_m0() && class == MemoryOperationClass::Atomic {
            Err(MemoryStatus::Unsupported)
        } else {
            // CLREX has no data-memory type. Exclusive transfers still refuse
            // in validate_data_request before any Device-RAM access.
            Ok(())
        }
    }
    fn validate_sp_alignment(&self, required: SpAlignmentRequirement) -> Result<(), MemoryStatus> {
        if required == SpAlignmentRequirement::Any || required == self.controls.sp.sp_requirement()
        {
            Ok(())
        } else {
            Err(MemoryStatus::Unsupported)
        }
    }
    fn validate_address_checks(
        &self,
        required: stela_compiler::context::memory::address::AddressCheckRequirements,
    ) -> Result<(), MemoryStatus> {
        use stela_compiler::context::memory::address::IncrementChoice;
        if required.is_empty()
            || self
                .controls
                .address_checks
                .is_some_and(|checks| required.matches(checks, IncrementChoice::Unchecked))
        {
            Ok(())
        } else {
            Err(MemoryStatus::Unsupported)
        }
    }
    fn translate(&self, address: u64) -> Result<u64, MemoryStatus> {
        // TBI/address-size decisions belong to the source walker. Its low four
        // address bits are unchanged, so decomposition needs no early walk.
        Ok(address)
    }
    fn validate_data_request(&mut self, request: DataRequest) -> Result<(), MemoryStatus> {
        self.validate_execution(request.flags)?;
        if request.bytes == 0
            || request.bytes > 16
                && !(request.bytes == 32
                    && request.operation == DataOperation::Ordinary
                    && request.flags & (flags::ASIMD | flags::PAIR) == (flags::ASIMD | flags::PAIR))
            || request.operation == DataOperation::AllocationTag
        {
            return Err(MemoryStatus::Unsupported);
        }
        if self.controls.cold_m0()
            && (!matches!(
                request.operation,
                DataOperation::Ordinary | DataOperation::Ordered
            ) || request.operation == DataOperation::Ordered
                && !matches!(request.bytes, 1 | 2 | 4 | 8))
        {
            return Err(MemoryStatus::Unsupported);
        }
        let shape = shape(request);
        if mmu::alignment::before_translation(
            self.controls.translation.sctlr,
            request.address,
            shape,
        ) {
            let access = if request.flags & flags::WRITE != 0 {
                AccessKind::Write
            } else {
                AccessKind::Read
            };
            return Err(self.fault(
                request,
                request.address,
                access,
                mmu::Fault {
                    kind: mmu::FaultKind::Alignment,
                    level: 0,
                    second_stage: None,
                },
            ));
        }
        // Ordinary Normal transfers use Mem's exact SCA/byte decomposition.
        // Keep the separate cold Device and nonordinary alignment cohorts
        // explicit; their admission is not inferred from ordinary RAM support.
        if !shape.aligned(request.address)
            && (request.operation != DataOperation::Ordinary || self.controls.cold_m0())
        {
            return Err(self.unsupported(request, request.address, Refusal::UnalignedQuantity));
        }
        Ok(())
    }
    fn translate_part(
        &mut self,
        request: DataRequest,
        part: DataPart,
    ) -> Result<u64, (MemoryStatus, u64)> {
        let eligible = request.operation == DataOperation::Ordinary && !self.controls.cold_m0();
        if let Some(physical) = self.tlb.lookup(part, eligible) {
            // The cache admits only contained Normal, untagged parts from a
            // checked walk. RAM ownership still needs a check on every hit.
            let ram = self.ram.check_physical(physical, part.bytes);
            self.check_physical_result(request, part.address, ram)?;
            return Ok(physical);
        }
        let access = match part.kind {
            AccessKind::Read => mmu::Access::Read,
            AccessKind::Write => mmu::Access::Write,
            AccessKind::Execute => return Err((MemoryStatus::Unsupported, part.address)),
        };
        // Match Arm Mem, including its ASIMD-pair alignment override. All
        // other pieces keep the original instruction's alignment result;
        // DEVPAGE2 takes FAULT when a later byte reaches Device memory.
        let alignment = if request.operation == DataOperation::Ordinary
            && request.bytes == 32
            && request.flags & (flags::ASIMD | flags::PAIR) == (flags::ASIMD | flags::PAIR)
            && request.address.is_multiple_of(8)
        {
            mmu::alignment::MemoryCheck::aligned()
        } else {
            mmu::alignment::MemoryCheck::new(shape(request), request.address)
        };
        let mut tables = Tables::new(&self.ram);
        let outcome = mmu::translate_checked_part(
            &self.controls.translation,
            part.address,
            access,
            self.controls.requester,
            alignment,
            &mut tables,
        );
        self.tlb.record_walk(
            #[cfg(feature = "native-profile")]
            tables.reads,
        );
        if let Some(status) = tables.failure {
            return Err((status, part.address));
        }
        let translated = match outcome {
            mmu::Outcome::Mapped(mapped) => mapped,
            mmu::Outcome::Fault(fault) => {
                return Err((
                    self.fault(request, part.address, part.kind, fault),
                    part.address,
                ));
            }
            mmu::Outcome::Unsupported(reason) => {
                return Err((
                    self.unsupported(request, part.address, Refusal::Translation(reason)),
                    part.address,
                ));
            }
        };
        let remaining = translated.size - (translated.physical & (translated.size - 1));
        if part.bytes == 0 || part.bytes as u64 > remaining {
            return Err((
                self.unsupported(request, part.address, Refusal::PhysicalGeometry),
                part.address,
            ));
        }
        let ram = self.ram.check_physical(translated.physical, part.bytes);
        if ram == Err(MemoryStatus::Nonresident) {
            return Err((MemoryStatus::Nonresident, part.address));
        }
        if translated.memory == mmu::MemoryType::Device
            && request.operation == DataOperation::Ordinary
            && ram == Err(MemoryStatus::Translation)
        {
            let reached = if request.flags & flags::WRITE == 0 {
                Failure::DeviceRead {
                    request,
                    address: part.address,
                    physical: translated.physical,
                    bytes: part.bytes,
                }
            } else {
                Failure::DeviceWrite {
                    request,
                    address: part.address,
                    physical: translated.physical,
                    bytes: part.bytes,
                }
            };
            self.failure.get_or_insert(reached);
            return Err((MemoryStatus::Unsupported, part.address));
        }
        let cold_device = self.controls.cold_m0()
            && matches!(
                request.operation,
                DataOperation::Ordinary | DataOperation::Ordered
            )
            && translated.memory == mmu::MemoryType::Device
            && translated.attributes == mmu::attributes::DEVICE_NGNRNE;
        if !cold_device
            && (translated.memory != mmu::MemoryType::Normal
                || translated.attributes == mmu::attributes::TAGGED)
            || !matches!(
                translated.shareability,
                mmu::attributes::INNER_SHAREABLE | mmu::attributes::OUTER_SHAREABLE
            )
        {
            return Err((
                self.unsupported(
                    request,
                    part.address,
                    Refusal::MemoryAttributes {
                        physical: translated.physical,
                        memory: translated.memory,
                        attributes: translated.attributes,
                        shareability: translated.shareability,
                    },
                ),
                part.address,
            ));
        }
        // Normal cacheability, allocation and transience describe the guest
        // translation, not a new host mapping. Every observer still reaches
        // the same checked coherent WB reservation below. Existing SCA,
        // monitor, publication and barrier services supply its memory effects;
        // Normal Non-cacheable is not Device memory and needs no Device lock.
        // Mapping the data quantity to an unowned device/aperture cannot acquire
        // canonical-RAM permission just because its ARM descriptor was Normal.
        self.check_physical_result(request, part.address, ram)?;
        if eligible {
            self.tlb.fill(part, translated);
        }
        Ok(translated.physical)
    }
    fn check(&mut self, access: DataAccess<'_>) -> Result<(), MemoryStatus> {
        if access.attributes.tagged || !access.attributes.shareable {
            Err(MemoryStatus::Unsupported)
        } else {
            Ok(())
        }
    }
}
pub(super) struct Tables<'a> {
    descriptors: DescriptorReader<'a>,
    pub(super) failure: Option<MemoryStatus>,
    #[cfg(feature = "native-profile")]
    pub(super) reads: u64,
}
impl<'a> Tables<'a> {
    pub(super) fn new(ram: &'a RegisteredSystemRam) -> Self {
        Self {
            descriptors: ram.descriptor_reader(),
            failure: None,
            #[cfg(feature = "native-profile")]
            reads: 0,
        }
    }
}
impl mmu::TableMemory for Tables<'_> {
    fn read_descriptor(&mut self, physical: u64) -> Option<u64> {
        #[cfg(feature = "native-profile")]
        {
            self.reads = self.reads.saturating_add(1);
        }
        match self.descriptors.read_descriptor(physical) {
            Ok(value) => Some(value),
            Err(MemoryStatus::Translation) => None,
            Err(status) => {
                self.failure = Some(status);
                None
            }
        }
    }
    fn update_descriptor(&mut self, physical: u64, old: u64, new: u64) -> Option<bool> {
        match self.descriptors.update_descriptor(physical, old, new) {
            Ok(changed) => Some(changed),
            Err(MemoryStatus::Translation) => None,
            Err(status) => {
                self.failure = Some(status);
                None
            }
        }
    }
}

#[cfg(test)]
pub(super) mod tests;
