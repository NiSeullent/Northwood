//! FEAT_MTE and FEAT_MTE2 in the engine: the tag instructions IRG, ADDG,
//! SUBG, LDG, STG, STZG, ST2G, STZ2G, STGP, LDGM, STGM and STZGM, DC GVA and
//! DC GZVA, and the Tag Check of tag-checked data accesses (Arm
//! `AArch64_MemTag`, `AArch64_ChooseTagOrZero`, `AArch64_MemZero` and
//! `AArch64_CheckTag`). Allocation Tags live in
//! [`crate::machine::tags::Store`] by physical address. A tag access
//! translates its address, with its faults and watchpoints, and reaches the
//! tags only where Allocation Tag access is enabled (`Cpu::allocation_tags`)
//! and the memory is Tagged (`attributes::TAGGED`): elsewhere a tag reads 0
//! and a tag write has no effect.
use super::{Devices, Host, Machine, RamTables, RequirementKind, Step, Stop, Translated};
use alloc::vec::Vec;
use stela_compiler::aarch64::{
    AddressMode, Operation, Register,
    tag_arithmetic::{self, Kind as ArithmeticKind},
    tag_memory::{BulkKind, Operand, StoreKind},
};
use {
    crate::machine::cpu::Exception, crate::machine::debug, crate::machine::exception::Class,
    crate::machine::mmu, crate::machine::mmu::Access, crate::machine::mmu::Translation,
    crate::machine::mmu::alignment, crate::machine::mmu::attributes::TAGGED,
    crate::machine::system::Register as SystemRegister, crate::machine::tags::GRANULE,
    crate::memory::tags::AllocationTag, crate::memory::tags::TagGenerationControls,
    crate::memory::tags::TagGenerationState, crate::memory::tags::TagRandomization,
    crate::memory::tags::choose_non_excluded_tag, stela_contracts::Feature,
};

/// GCR_EL1.Exclude and RRND.
const GCR_EXCLUDE: u64 = 0xffff;
const GCR_RRND: u64 = 1 << 16;
/// RGSR_EL1.TAG and SEED[15:0] (bits 23:8), the LFSR IRG steps.
const RGSR_TAG: u64 = 0xf;
const RGSR_SEED_SHIFT: u32 = 8;
const RGSR_SEED: u64 = 0xffff << RGSR_SEED_SHIFT;
/// GMID_EL1.BS and DCZID_EL0.BS: log2 of a block in words.
const BLOCK_SIZE: u64 = 0xf;
/// The largest block Arm allows (`MAX_ZERO_BLOCK_SIZE`, 2 KiB).
const MAX_BLOCK: u64 = 2048;
static ZEROES: [u8; MAX_BLOCK as usize] = [0; MAX_BLOCK as usize];

/// Whether the instruction's data accesses are tag checked (the
/// pseudocode's `tagchecked`): every form with a base register, except an
/// SP base without writeback; register-offset forms always; literal loads
/// never.
pub(super) fn tag_checked_form(operation: &Operation) -> bool {
    use stela_compiler::aarch64::{non_temporal, ordered, simd};
    let based =
        |base: Register, mode: AddressMode| mode != AddressMode::Offset || base != Register::Sp;
    match operation {
        Operation::Memory { base, mode, .. } | Operation::MemoryPair { base, mode, .. } => {
            based(*base, *mode)
        }
        Operation::MemoryRegisterOffset(_) => true,
        Operation::MemoryUnprivileged(operand) => operand.base != Register::Sp,
        Operation::MemoryNonTemporal(non_temporal::Operand { base, .. }) => *base != Register::Sp,
        Operation::Atomic(operand) => operand.base != Register::Sp,
        Operation::Ordered(ordered::Operand::Transfer { base, .. }) => *base != Register::Sp,
        Operation::Simd(
            simd::Operand::Memory { base, mode, .. } | simd::Operand::MemoryPair { base, mode, .. },
        ) => based(*base, *mode),
        Operation::Simd(simd::Operand::MemoryRegisterOffset(_)) => true,
        Operation::Simd(simd::Operand::StructureMemory(operand)) => {
            operand.post_index != simd::structure_memory::PostIndex::None
                || operand.base != Register::Sp
        }
        _ => false,
    }
}

/// One granule's translation for a tag access and whether its tags are
/// reachable.
#[derive(Clone, Copy)]
struct Granule {
    translation: Translation,
    tagged: bool,
}

/// The outcome of a Tag Check ([`Machine::tag_check`]).
pub(super) enum TagCheck {
    /// No granule mismatches, or the access is not checked.
    Passed,
    /// A granule's translation faults: the access takes that fault, and
    /// later accesses of the instruction are not reached.
    Untranslated,
    /// The Tag Check fault to take.
    Failed(Exception),
}

/// Why a tag access did not complete: an exception to take, or what the
/// engine does not implement.
pub(super) enum Failure {
    Exception(Exception),
    Translation(mmu::Unsupported),
    Refused(&'static str),
}

impl<H: Host, D: Devices> Machine<H, D> {
    /// A tag instruction (system-mode `SystemOperation::TagMemory`).
    pub(super) fn tag_memory(&mut self, word: u32, operand: Operand) -> Step {
        let feature = if operand.requires_privilege() {
            Feature::FEAT_MTE2
        } else {
            Feature::FEAT_MTE
        };
        match self.cpu.registers.implemented(feature) {
            Ok(true) => {}
            Ok(false) => return self.undefined(),
            Err(error) => {
                return Err(self.unsupported(RequirementKind::Instruction {
                    word,
                    detail: error.reason().into(),
                }));
            }
        }
        let (result, address) = match operand {
            Operand::RandomTag {
                destination,
                address,
                mask,
            } => return self.random_tag(destination, address, mask),
            Operand::LoadTag {
                destination,
                base,
                displacement,
            } => self.load_tag(destination, base, displacement),
            Operand::StoreTags {
                kind,
                source,
                base,
                displacement,
                mode,
            } => self.store_tags(kind, source, base, displacement, mode),
            Operand::StorePair {
                data,
                base,
                displacement,
                mode,
            } => self.store_pair_tag(data, base, displacement, mode),
            Operand::BulkTags { kind, data, base } => self.bulk_tags(kind, data, base),
        };
        match result {
            Ok(()) => self.next(),
            Err(Failure::Exception(exception)) => {
                self.take(exception);
                Ok(())
            }
            Err(Failure::Translation(unsupported)) => {
                Err(self.unsupported(RequirementKind::Translation(unsupported)))
            }
            Err(Failure::Refused(reason)) => {
                let mut requirement = self.requirement(RequirementKind::Instruction {
                    word,
                    detail: reason.into(),
                });
                requirement.address = address;
                Err(Stop::Unsupported(requirement))
            }
        }
    }

    /// ADDG and SUBG (system-mode `SystemOperation::TagOffset`): the
    /// address moves by the offset and takes the tag `tag_offset` steps on
    /// from the source's over the tags GCR_EL1.Exclude leaves, or 0 without
    /// Allocation Tag access (Arm `AArch64_ChooseNonExcludedTagOrZero`).
    pub(super) fn tag_offset(&mut self, word: u32, operand: tag_arithmetic::Operand) -> Step {
        match self.cpu.registers.implemented(Feature::FEAT_MTE) {
            Ok(true) => {}
            Ok(false) => return self.undefined(),
            Err(error) => {
                return Err(self.unsupported(RequirementKind::Instruction {
                    word,
                    detail: error.reason().into(),
                }));
            }
        }
        let ArithmeticKind::Offset {
            source,
            subtract,
            address_offset,
            tag_offset,
        } = operand.kind
        else {
            return Err(self.unsupported(RequirementKind::Instruction {
                word,
                detail: "tag arithmetic other than ADDG and SUBG".into(),
            }));
        };
        let value = self.read_register(source);
        let tag = if self.cpu.allocation_tags(self.cpu.pstate.el) {
            let exclude = (self.cpu.read(SystemRegister::GCR_EL1) & GCR_EXCLUDE) as u16;
            choose_non_excluded_tag(AllocationTag::from_address(value), tag_offset, exclude)
        } else {
            AllocationTag::ZERO
        };
        let offset = u64::from(address_offset);
        let result = if subtract {
            value.wrapping_sub(offset)
        } else {
            value.wrapping_add(offset)
        };
        self.write_register(operand.destination, tag.apply_to_address(result));
        self.next()
    }

    /// IRG (Arm `AArch64_ChooseTagOrZero`): Xm's low 16 bits and
    /// GCR_EL1.Exclude exclude tags. With GCR_EL1.RRND clear the next tag
    /// comes from RGSR_EL1's LFSR, which it updates; with RRND set Arm's
    /// `ChooseRandomNonExcludedTag` is IMPLEMENTATION DEFINED, and Stela
    /// takes the same LFSR, except that an all-excluded mask gives 0 with
    /// RGSR_EL1 unchanged.
    fn random_tag(&mut self, destination: Register, address: Register, mask: Register) -> Step {
        let value = self.read_register(address);
        let register_exclude = self.read_register(mask);
        let gcr = self.cpu.read(SystemRegister::GCR_EL1);
        let excluded = (gcr & GCR_EXCLUDE) as u16 | register_exclude as u16;
        let tag = if !self.cpu.allocation_tags(self.cpu.pstate.el)
            || gcr & GCR_RRND != 0 && excluded == u16::MAX
        {
            AllocationTag::ZERO
        } else {
            let rgsr = self.cpu.read(SystemRegister::RGSR_EL1);
            let mut state = TagGenerationState {
                seed: ((rgsr & RGSR_SEED) >> RGSR_SEED_SHIFT) as u16,
                tag: AllocationTag::new((rgsr & RGSR_TAG) as u8).expect("a four-bit tag"),
            };
            let controls = TagGenerationControls {
                allocation_access_enabled: true,
                excluded_tags: (gcr & GCR_EXCLUDE) as u16,
                randomization: TagRandomization::ArchitecturalLfsr,
            };
            let tag = state
                .generate_tag(controls, register_exclude, None)
                .expect("the LFSR needs no entropy");
            let rgsr = rgsr & !(RGSR_SEED | RGSR_TAG)
                | u64::from(state.seed) << RGSR_SEED_SHIFT
                | u64::from(state.tag.value());
            self.cpu.registers.set(SystemRegister::RGSR_EL1, 0, rgsr);
            tag
        };
        self.write_register(destination, tag.apply_to_address(value));
        self.next()
    }

    /// The base register's value, or SP's alignment fault (Arm
    /// `CheckSPAlignment`).
    fn tag_base(&mut self, base: Register) -> Result<u64, Failure> {
        let value = self.read_register(base);
        if base == Register::Sp && self.stack_alignment_checked() && !value.is_multiple_of(16) {
            return Err(Failure::Exception(Exception::synchronous(
                Class::SpAlignment,
                0,
                self.cpu.pc,
            )));
        }
        Ok(value)
    }

    /// The Alignment fault of a tag store whose address is not aligned to
    /// 16 bytes, which comes before translation (Arm `AArch64_MemTagWrite`).
    fn tag_alignment(&mut self, address: u64) -> Result<(), Failure> {
        if address.is_multiple_of(GRANULE) {
            return Ok(());
        }
        let fault = mmu::Fault::stage1(mmu::FaultKind::Alignment, 0);
        Err(Failure::Exception(
            self.data_abort_exception(address, fault, true, false),
        ))
    }

    /// Translate `address` for a tag access of `size` bytes, a write when
    /// `write`, as the data access at the same address would be (Arm
    /// `AArch64_TranslateTagAddress`), then check the watchpoints. `shape`
    /// makes Device memory take an Alignment fault (DC GVA, DC GZVA and
    /// STZGM: the FAULT choice of Unpredictable_DEVICETAGSTORE). A fault
    /// reports `reported` in FAR.
    fn tag_granule(
        &mut self,
        address: u64,
        size: u64,
        write: bool,
        shape: Option<alignment::Shape>,
        reported: u64,
    ) -> Result<Granule, Failure> {
        let controls = self.cpu.controls(self.cpu.regime());
        let requester = self.cpu.requester();
        let access = if write { Access::Write } else { Access::Read };
        let tables = &mut RamTables(&mut self.host);
        let outcome = match shape {
            Some(shape) => {
                mmu::translate_checked(&controls, address, access, requester, shape, tables)
            }
            None => mmu::translate(&controls, address, access, requester, tables),
        };
        let translation = match outcome {
            Translated::Mapped(translation) => translation,
            Translated::Fault(fault) => {
                return Err(Failure::Exception(
                    self.data_abort_exception(reported, fault, write, false),
                ));
            }
            Translated::Unsupported(unsupported) => return Err(Failure::Translation(unsupported)),
        };
        let watched = debug::Access {
            address,
            size,
            read: !write,
            write,
            maintenance: false,
            el: self.cpu.pstate.el,
        };
        if let Some(exception) = self.watchpoint_exception(&watched, self.cpu.pc) {
            return Err(Failure::Exception(exception));
        }
        let tagged =
            translation.attributes == TAGGED && self.cpu.allocation_tags(self.cpu.pstate.el);
        if tagged && !self.host.read_ram(translation.physical, &mut [0]) {
            return Err(Failure::Refused(
                "Allocation Tags of Tagged memory outside RAM",
            ));
        }
        Ok(Granule {
            translation,
            tagged,
        })
    }

    /// The tag of the granule at `offset` bytes into `granule`, 0 where
    /// its tags are unreachable.
    fn read_tag(&self, granule: Granule, offset: u64) -> u8 {
        if granule.tagged {
            self.host.speculation_barrier();
            self.tags.get(granule.translation.physical + offset)
        } else {
            0
        }
    }

    fn write_tag(&mut self, granule: Granule, offset: u64, tag: u8) {
        if granule.tagged {
            self.tags.set(granule.translation.physical + offset, tag);
        }
    }

    /// Write `bytes` of data at `granule`'s translation.
    fn write_data(&mut self, granule: Granule, bytes: &[u8]) -> Result<(), Failure> {
        if self.host.write_ram(granule.translation.physical, bytes) {
            Ok(())
        } else {
            Err(Failure::Refused("a tag store's data write outside RAM"))
        }
    }

    /// LDG: Xt takes the tag of the granule at the aligned address.
    fn load_tag(
        &mut self,
        destination: Register,
        base: Register,
        displacement: i32,
    ) -> (Result<(), Failure>, Option<u64>) {
        let base = match self.tag_base(base) {
            Ok(base) => base,
            Err(failure) => return (Err(failure), None),
        };
        let address = base.wrapping_add(displacement as i64 as u64) & !(GRANULE - 1);
        let result = self
            .tag_granule(address, GRANULE, false, None, address)
            .map(|granule| {
                let tag = AllocationTag::new(self.read_tag(granule, 0)).expect("a four-bit tag");
                let value = self.read_register(destination);
                self.write_register(destination, tag.apply_to_address(value));
            });
        (result, Some(address))
    }

    /// The access address of an immediate form and the base it writes back.
    fn tag_address(base: u64, displacement: i32, mode: AddressMode) -> (u64, Option<u64>) {
        let moved = base.wrapping_add(displacement as i64 as u64);
        match mode {
            AddressMode::Offset => (moved, None),
            AddressMode::PreIndex => (moved, Some(moved)),
            AddressMode::PostIndex => (base, Some(moved)),
        }
    }

    /// STG, STZG, ST2G and STZ2G: the tag of Xt (SP for register 31) for
    /// one or two granules at an address aligned to 16 bytes, whose data
    /// STZG and STZ2G zero. Every granule translates before any is written.
    fn store_tags(
        &mut self,
        kind: StoreKind,
        source: Register,
        base: Register,
        displacement: i32,
        mode: AddressMode,
    ) -> (Result<(), Failure>, Option<u64>) {
        let tag = AllocationTag::from_address(self.read_register(source)).value();
        let base_value = match self.tag_base(base) {
            Ok(base) => base,
            Err(failure) => return (Err(failure), None),
        };
        let (address, writeback) = Self::tag_address(base_value, displacement, mode);
        let (count, zero) = match kind {
            StoreKind::Tag => (1, false),
            StoreKind::TagPair => (2, false),
            StoreKind::ZeroTag => (1, true),
            StoreKind::ZeroTagPair => (2, true),
        };
        let result = (|| {
            self.tag_alignment(address)?;
            let mut granules = Vec::new();
            for index in 0..count {
                let address = address.wrapping_add(index * GRANULE);
                granules.push(self.tag_granule(address, GRANULE, true, None, address)?);
            }
            if zero {
                for &granule in &granules {
                    self.write_data(granule, &ZEROES[..GRANULE as usize])?;
                }
            }
            for &granule in &granules {
                self.write_tag(granule, 0, tag);
            }
            if let Some(value) = writeback {
                self.write_register(base, value);
            }
            Ok(())
        })();
        (result, Some(address))
    }

    /// STGP: Xt and Xt2 to the 16 bytes at an aligned address, whose tag
    /// becomes the address's.
    fn store_pair_tag(
        &mut self,
        data: [Register; 2],
        base: Register,
        displacement: i32,
        mode: AddressMode,
    ) -> (Result<(), Failure>, Option<u64>) {
        let base_value = match self.tag_base(base) {
            Ok(base) => base,
            Err(failure) => return (Err(failure), None),
        };
        let (address, writeback) = Self::tag_address(base_value, displacement, mode);
        let result = (|| {
            self.tag_alignment(address)?;
            let granule = self.tag_granule(address, GRANULE, true, None, address)?;
            let mut bytes = [0; GRANULE as usize];
            bytes[..8].copy_from_slice(&self.read_register(data[0]).to_le_bytes());
            bytes[8..].copy_from_slice(&self.read_register(data[1]).to_le_bytes());
            self.write_data(granule, &bytes)?;
            self.write_tag(granule, 0, AllocationTag::from_address(address).value());
            if let Some(value) = writeback {
                self.write_register(base, value);
            }
            Ok(())
        })();
        (result, Some(address))
    }

    /// The block in bytes of `register`'s BS field, which LDGM and STGM
    /// (GMID_EL1) and STZGM and DC GVA and GZVA (DCZID_EL0) work on.
    fn tag_block(&self, register: SystemRegister) -> Result<u64, Failure> {
        let bytes = 4 << (self.cpu.read(register) & BLOCK_SIZE);
        if self.profile_missing.contains(&register) || !(GRANULE..=MAX_BLOCK).contains(&bytes) {
            return Err(Failure::Refused(
                "a tag block without a profile size of 16 bytes to 2 KiB",
            ));
        }
        Ok(bytes)
    }

    /// LDGM, STGM and STZGM, UNDEFINED at EL0: the tags of an aligned block,
    /// a nibble each in Xt at the granule's index (address bits 7:4), or
    /// for STZGM Xt's low four bits and zeroed data.
    fn bulk_tags(
        &mut self,
        kind: BulkKind,
        data: Register,
        base: Register,
    ) -> (Result<(), Failure>, Option<u64>) {
        if self.cpu.pstate.el == 0 {
            let exception = Exception::undefined(self.cpu.pc);
            return (Err(Failure::Exception(exception)), None);
        }
        let value = self.read_register(data);
        let base_value = match self.tag_base(base) {
            Ok(base) => base,
            Err(failure) => return (Err(failure), None),
        };
        let result = (|| {
            let size = self.tag_block(match kind {
                BulkKind::StoreZero => SystemRegister::DCZID_EL0,
                _ => SystemRegister::GMID_EL1,
            })?;
            let address = base_value & !(size - 1);
            let shape = (kind == BulkKind::StoreZero).then_some(alignment::Shape {
                kind: alignment::Kind::Zero,
                bytes: size,
                pair: false,
            });
            let write = kind != BulkKind::Load;
            let granule = self.tag_granule(address, size, write, shape, address)?;
            let first = (address >> 4) & 0xf;
            let count = size / GRANULE;
            match kind {
                BulkKind::Load => {
                    let mut tags = 0;
                    for index in 0..count {
                        let tag = u64::from(self.read_tag(granule, index * GRANULE));
                        tags |= tag << (4 * ((first + index) % 16));
                    }
                    self.write_register(data, tags);
                }
                BulkKind::Store => {
                    for index in 0..count {
                        let tag = (value >> (4 * ((first + index) % 16))) as u8;
                        self.write_tag(granule, index * GRANULE, tag);
                    }
                }
                BulkKind::StoreZero => {
                    for index in 0..count {
                        self.write_tag(granule, index * GRANULE, value as u8);
                    }
                    self.write_data(granule, &ZEROES[..size as usize])?;
                }
            }
            Ok(())
        })();
        (result, Some(base_value))
    }

    /// DC GVA and DC GZVA (Arm `AArch64_MemZero` and `AArch64_WriteTagMem`):
    /// the DC ZVA block holding `address` takes its logical tag, and with
    /// `zero` its data is zeroed. Device memory takes an Alignment fault; a
    /// fault reports `address` in FAR.
    pub(super) fn set_block_tags(&mut self, address: u64, zero: bool) -> Result<(), Failure> {
        let size = self.tag_block(SystemRegister::DCZID_EL0)?;
        let start = address & !(size - 1);
        let shape = alignment::Shape {
            kind: alignment::Kind::Zero,
            bytes: size,
            pair: false,
        };
        let granule = self.tag_granule(start, size, true, Some(shape), address)?;
        let tag = AllocationTag::from_address(start).value();
        for index in 0..size / GRANULE {
            self.write_tag(granule, index * GRANULE, tag);
        }
        if zero {
            self.write_data(granule, &ZEROES[..size as usize])?;
        }
        Ok(())
    }

    /// The Tag Check of a tag-checked data access (Arm `AArch64_CheckTag`)
    /// translated for `requester`: where the access's EL checks tags at its
    /// address (`Cpu::tag_checked`), each 16-byte granule it touches that
    /// is Tagged memory must hold the address's logical tag. The Tag Check
    /// fault of the first mismatch, a Data Abort with DFSC 0b010001 whose
    /// FAR is `reported`, or else the access address in the first granule
    /// and the granule's in later ones (Arm leaves FAR bits 63:60 UNKNOWN;
    /// Stela keeps the address's). Checking stops at a granule whose
    /// translation faults, which the access then takes.
    pub(super) fn tag_check(
        &mut self,
        access: &debug::Access,
        requester: mmu::Requester,
        reported: Option<u64>,
    ) -> Result<TagCheck, Stop> {
        self.tag_check_with_tables(access, requester, reported, None)
    }

    pub(super) fn tag_check_with_tables(
        &mut self,
        access: &debug::Access,
        requester: mmu::Requester,
        reported: Option<u64>,
        mut registered: Option<&mut super::device_access::SiteTables<'_>>,
    ) -> Result<TagCheck, Stop> {
        if !self.cpu.tag_checked(access.el, access.address) {
            return Ok(TagCheck::Passed);
        }
        let tag = AllocationTag::from_address(access.address).value();
        let first = access.address & (GRANULE - 1);
        let count = (first + access.size.max(1) - 1) / GRANULE + 1;
        let kind = if access.write {
            Access::Write
        } else {
            Access::Read
        };
        let controls = self.cpu.controls(self.cpu.regime());
        for index in 0..count {
            let address = if index == 0 {
                access.address
            } else {
                (access.address & !(GRANULE - 1)).wrapping_add(index * GRANULE)
            };
            let outcome = match registered.as_deref_mut() {
                Some(tables) => tables
                    .translate(&controls, address, kind, requester)
                    .map_err(|reason| self.unsupported(reason))?,
                None => mmu::translate(
                    &controls,
                    address,
                    kind,
                    requester,
                    &mut RamTables(&mut self.host),
                ),
            };
            match outcome {
                Translated::Mapped(translation)
                    if translation.attributes == TAGGED && {
                        self.host.speculation_barrier();
                        self.tags.get(translation.physical) != tag
                    } =>
                {
                    let fault = mmu::Fault::stage1(mmu::FaultKind::TagCheck, 0);
                    let write = access.write && !access.read;
                    let reported = reported.unwrap_or(address);
                    return Ok(TagCheck::Failed(
                        self.data_abort_exception(reported, fault, write, false),
                    ));
                }
                Translated::Mapped(_) => {}
                Translated::Fault(_) => return Ok(TagCheck::Untranslated),
                Translated::Unsupported(unsupported) => {
                    return Err(self.unsupported(RequirementKind::Translation(unsupported)));
                }
            }
        }
        Ok(TagCheck::Passed)
    }
}

#[cfg(test)]
mod tests;
