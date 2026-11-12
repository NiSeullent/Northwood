//! Faults and terminal failures of native code, and translated-access fallback
//! runs.

use super::page_sets::Inserted;
use super::*;
use stela_compiler::x86_64::TopByte;

impl<H: Host, D: Devices> Machine<H, D> {
    pub(super) fn latch_canonical_native_failure(&mut self) {
        if self.native_failure.is_some() {
            return;
        }
        if let Some(capture) = self.native_memory_identity.failure() {
            let summary = Arc::new(NativeFailure {
                pc: capture.pc,
                code: Some(capture.code),
                native_offset: capture.fault.offset,
                address: capture.fault.address,
                origin: capture.fault.origin,
                trap: capture.fault.trap,
                error_code: capture.fault.error_code,
                unsettled_chain: None,
                reason: NativeFailureReason::CanonicalDomain {
                    pe: capture.pe,
                    retired_before: capture.retired,
                },
                board: None,
                owner: None,
            });
            self.native_failure = Some(recovery::CapturedFailure {
                summary,
                capture: capture.fault,
            });
        }
    }

    /// The original fixed capture kept after a terminal native failure.
    /// Its frame is diagnostic only; subsequent run calls refuse execution.
    pub fn native_failure_capture(&self) -> Option<&NativeFault> {
        self.native_failure.as_ref().map(|failure| &failure.capture)
    }

    pub(super) fn fail_native(
        &mut self,
        index: usize,
        fault: NativeFault,
        reason: NativeFailureReason,
    ) -> Stop {
        let block = self.code.blocks.get(index);
        let (last_block, reported_retired, lookup_hits, direct_links) = self.chain.completed();
        let board = matches!(
            reason,
            NativeFailureReason::UnknownCode | NativeFailureReason::CodeExtent
        );
        // The faulting host address is relative to the block `index`.
        let owner = (reason == NativeFailureReason::CodeExtent)
            .then(|| block.map(|block| block.code.0.wrapping_add(fault.offset)))
            .flatten()
            .and_then(|address| self.code_owner(address))
            .map(|slot| self.board_slot(slot));
        let failure = Arc::new(NativeFailure {
            pc: block.map_or(self.cpu.pc, |block| block.pc),
            code: block.map(|block| block.code),
            native_offset: fault.offset,
            address: fault.address,
            origin: fault.origin,
            trap: fault.trap,
            error_code: fault.error_code,
            unsettled_chain: Some(UnsettledChain {
                reported_retired,
                last_block,
                lookup_hits,
                direct_links,
            }),
            board: board.then(|| self.board_slot(last_block as u32 as usize)),
            owner,
            reason,
        });
        self.native_failure = Some(recovery::CapturedFailure {
            summary: failure.clone(),
            capture: fault,
        });
        Stop::NativeFailure(failure)
    }

    /// The slot whose block's code holds host `address`, a live one first:
    /// a walk of every slot, for a failure report.
    fn code_owner(&self, address: usize) -> Option<usize> {
        let blocks = &self.code.blocks;
        let recovery = blocks.recovery();
        let holds = |slot: usize| {
            let block = blocks.get(slot)?;
            let bytes = recovery.get(&slot).map_or(1, |owner| owner.code_bytes());
            (block.code.0..block.code.0 + bytes.max(1))
                .contains(&address)
                .then_some(slot)
        };
        let mut found = None;
        for slot in 0..blocks.len() {
            if let Some(slot) = holds(slot) {
                if blocks.record(slot).live() {
                    return Some(slot);
                }
                found.get_or_insert(slot);
            }
        }
        found
    }

    /// Where `slot` stands on the board, for a failure report.
    pub(super) fn board_slot(&self, slot: usize) -> recovery::BoardSlot {
        let code = &self.code;
        let held = slot < code.blocks.len();
        let block = code.blocks.get(slot);
        let record = held.then(|| code.blocks.record(slot));
        let part = code.arena.as_ref().zip(block).and_then(|(arena, block)| {
            let offset = block.code.0.checked_sub(arena.window.exec)?;
            Some(offset / arena.part_bytes)
        });
        let ring = code.arena.as_ref().map(|arena| &arena.ring);
        // Read once: a second guard of the slot's rows in one expression
        // would wait on the first.
        let rows = code
            .blocks
            .recovery()
            .get(&slot)
            .map(|owner| owner.code_bytes());
        recovery::BoardSlot {
            slot: slot as u64,
            held: block.is_some(),
            live: record.is_some_and(board::Record::live),
            incarnation: record.map_or(0, board::Record::incarnation),
            lost: record.is_some_and(board::Record::lost),
            installer: block.map_or(u8::MAX, |block| block.installer),
            rows: rows.is_some(),
            code: block.map_or(0, |block| block.code.0 as u64),
            code_bytes: rows.unwrap_or(0) as u64,
            part,
            generation: ring.map_or(0, board::Ring::generation),
            retired_through: ring
                .zip(part)
                .map_or(0, |(ring, part)| ring.retired_through(part)),
        }
    }

    pub(super) fn native_fault(
        &mut self,
        index: usize,
        fault: NativeFault,
        chained_retired: u64,
        chained_btype: Option<u8>,
    ) -> Step {
        use stela_compiler::block::{
            FaultRecovery, HostFaultSnapshot, RecoverySiteKind, RetirementPhase,
        };
        if matches!(fault.cause, FaultCause::RamBinding) {
            return Err(self.fail_native(index, fault, NativeFailureReason::RamBinding));
        }
        // Source maps locate diagnostics; only a checked exact
        // operand/check site can authorize architectural fault projection.
        let Some(block) = self.code.blocks.get(index) else {
            return Err(self.fail_native(index, fault, NativeFailureReason::UnknownCode));
        };
        let (entry_pc, physical, source_bytes) = (block.pc, block.physical, block.source_bytes());
        // The rows are the block's, under its record's lock: no board lock.
        // Borrowed from the board without a clone of its `Arc`, whose count
        // every PE's would write at each fault.
        let checked = match self.code.blocks.recovery().get(&index) {
            None => Err(NativeFailureReason::UnknownCode),
            Some(owner) if fault.offset >= owner.code_bytes() => {
                Err(NativeFailureReason::CodeExtent)
            }
            Some(owner) => Ok(owner.stored_key()),
        };
        let key = match checked {
            Ok(key) => key,
            Err(reason) => return Err(self.fail_native(index, fault, reason)),
        };
        // The stored record is read outside the rows' lock.
        let record = key.and_then(|key| self.stored_record(physical, &key));
        let qualified = {
            let recovery = self.code.blocks.recovery();
            let mut owner = recovery.get_mut(&index);
            match owner.as_deref_mut() {
                Some(owner) => owner
                    .qualify(entry_pc, record.as_deref())
                    .map(|rows| rows.rows_at_reusing(fault.offset, &mut self.recovery_scratch)),
                None => Err(NativeFailureReason::UnknownCode),
            }
        };
        let table = match qualified {
            Ok(table) => table,
            Err(reason) => return Err(self.fail_native(index, fault, reason)),
        };
        let recovery_site = match FaultRecovery::site_at(&table, fault.offset) {
            Ok(site) => site,
            Err(error) => {
                return Err(self.fail_native(index, fault, NativeFailureReason::Recovery(error)));
            }
        };
        let site = recovery_site.kind();
        if site == RecoverySiteKind::NativeInternal {
            return Err(self.fail_native(index, fault, NativeFailureReason::InternalSite));
        }
        let compatible = match (fault.origin, site) {
            (NativeFaultOrigin::Exception, RecoverySiteKind::GuestAccess) => match fault.trap {
                14 => {
                    fault.error_code & !0b111 == 0
                        && !matches!(fault.cause, FaultCause::InvalidOpcode)
                }
                13 => fault.error_code == 0 && matches!(fault.cause, FaultCause::NonCanonical),
                _ => false,
            },
            (NativeFaultOrigin::Exception, RecoverySiteKind::GuestCheck) => {
                fault.trap == 6
                    && fault.error_code == 0
                    && matches!(fault.cause, FaultCause::InvalidOpcode)
            }
            (NativeFaultOrigin::TranslatedHelper, RecoverySiteKind::TranslatedHelper) => {
                fault.trap == 0
                    && fault.error_code == 0
                    && recovery_site.translated_access().is_some_and(|access| {
                        access.bytes() != 0
                            && access.write() == fault.write
                            && match fault.cause {
                                FaultCause::InvalidOpcode => access.atomic(),
                                FaultCause::NonCanonical
                                | FaultCause::Device { .. }
                                | FaultCause::NonresidentRam { .. }
                                | FaultCause::Translation(_)
                                | FaultCause::Watched
                                | FaultCause::TagChecked => true,
                                FaultCause::Guest(_) | FaultCause::RamBinding => false,
                            }
                    })
            }
            _ => false,
        };
        if !compatible {
            return Err(self.fail_native(index, fault, NativeFailureReason::TrapSiteMismatch));
        }
        let mut staged = self.cpu.state.clone();
        stela_compiler::x86_64::pinned::settle(&fault.registers, &mut staged);
        let settlement = match FaultRecovery::apply(
            &table,
            fault.offset,
            HostFaultSnapshot {
                gprs: &fault.registers,
                spills: &fault.spills,
            },
            &mut staged,
        ) {
            Ok(Some(settlement)) => settlement,
            Ok(None) => {
                return Err(self.fail_native(index, fault, NativeFailureReason::InternalSite));
            }
            Err(error) => {
                return Err(self.fail_native(index, fault, NativeFailureReason::Recovery(error)));
            }
        };
        let pc = settlement.position.pc.0;
        let count = u64::from(settlement.position.retired_instructions);
        if settlement.phase != RetirementPhase::Before
            || settlement.site.kind() != site
            || pc.checked_sub(entry_pc) != count.checked_mul(INSTRUCTION)
            || pc
                .checked_sub(entry_pc)
                .is_none_or(|bytes| bytes >= source_bytes as u64)
        {
            return Err(self.fail_native(index, fault, NativeFailureReason::Position));
        }
        // Publish the complete authorized boundary once, including chained and
        // current-block retirement. Internal/After windows never reach replay.
        self.cpu.state = staged;
        self.cpu.retired = self
            .cpu
            .retired
            .wrapping_add(chained_retired)
            .wrapping_add(count);
        self.cpu.pc = pc;
        if let Some(btype) = chained_btype {
            self.cpu.pstate.btype = btype;
        }
        if pc != entry_pc {
            self.cpu.pstate.btype = 0;
        }
        // Keep the settled source PC and committed prefix. A nonresident
        // operand or descriptor needs page recovery before executing it again.
        if let FaultCause::NonresidentRam { frame } = fault.cause {
            return Err(self.unsupported(RequirementKind::NonresidentRam { frame }));
        }
        // A native DC ZVA's first store, a native DC clean's read of Xt, or
        // a native exclusive faulted with nothing written (a DC ZVA block is
        // in one page): the engine completes the instruction, reporting Xt
        // as the fault address with a cache maintenance syndrome and watching
        // the whole DC ZVA block (Arm `AArch64_DataMemZero`), or adopting the
        // monitor the frame holds, whatever stopped the access.
        let (terminal, word) = (
            self.code.blocks[index].terminal_pc(),
            self.code.blocks[index].terminal,
        );
        if pc == terminal && completes_natively(pc, word, self.block_variant(index).system) {
            self.statistics.system_exits += 1;
            let word = self.exit_word(index, pc)?;
            return self.completing(pc, |machine| machine.system_instruction(word));
        }
        // A DC ZVA inside the block (`Operation::ZeroBlock`) likewise.
        let system = self.block_variant(index).system;
        if pc != terminal && system.zero_block.is_some() {
            let word = self.word(index, pc)?;
            if zeroes_natively(pc, word, system) {
                self.statistics.system_exits += 1;
                return self.completing(pc, |machine| machine.system_instruction(word));
            }
        }
        // A native AMX move faulted, or its range check failed, with nothing
        // changed: the engine's service completes it, delivering the fault
        // the guest's translation gives it with the AMX syndrome
        // (`x86_64::amx_transfer`).
        if pc != terminal && self.block_variant(index).system.amx {
            let word = self.exit_word(index, pc)?;
            if stela_compiler::amx::decode(GuestAddress(pc), word)
                .ok()
                .flatten()
                .is_some_and(|amx| stela_compiler::x86_64::amx_transfer::lowers(amx.operation))
            {
                return self.completing(pc, |machine| machine.amx(word));
            }
        }
        match fault.cause {
            FaultCause::Guest(guest) => {
                let address = self.tagged_fault_address(index, pc, fault.address);
                self.data_abort(address, guest, fault.write, false)
            }
            FaultCause::NonresidentRam { .. } => unreachable!("returned before emulation"),
            FaultCause::RamBinding => unreachable!("returned before recovery"),
            FaultCause::Device { physical } => {
                let word = self.word(index, pc)?;
                let site = self.device_site_of(index, pc);
                let address = self.tagged_fault_address(index, pc, fault.address);
                let step = self.outside_run(|machine| {
                    machine.completing(pc, |machine| {
                        machine.device_access(word, address, physical, fault.write)
                    })
                });
                // Its next run ends its block before it and completes it
                // without this fault.
                if let Some(site) = site {
                    self.learn_device_site(site);
                }
                step
            }
            FaultCause::Translation(unsupported) => {
                Err(self.unsupported(RequirementKind::Translation(unsupported)))
            }
            FaultCause::Watched | FaultCause::TagChecked => self.watched_access(index, pc),
            // Native blocks fault, by #UD or at the access with a #GP, for a
            // misaligned SP base when compiled with the check (SCTLR_ELx.SA
            // or SA0 set), which comes first, and for an access that has an
            // Alignment fault: unaligned under SCTLR_ELx.A, or an atomic, or
            // an ordered access with nAA clear, crossing its 16-byte granule;
            // and for an address outside the guest's two ranges, which may
            // be past a check an earlier one through its base stood for.
            FaultCause::NonCanonical | FaultCause::InvalidOpcode => {
                let word = self.word(index, pc)?;
                if self.block_variant(index).checks.stack_alignment
                    && stack_based(word)
                    && !self.cpu.state.sp.is_multiple_of(16)
                {
                    self.timed_check(step_timing::Check::Alignment);
                    self.take(Exception::synchronous(Class::SpAlignment, 0, pc));
                    Ok(())
                } else {
                    self.native_check_fault(index, pc, word, &fault)
                }
            }
        }
    }

    /// Run the instruction at `pc` of block `index`, whose access to the
    /// guest's mapped `address` lies outside the host's native ranges, again in the translated
    /// variant: its fetch page's blocks are compiled again so that every
    /// access calls the host's translation helper outside those ranges,
    /// which translates it, delivers its fault or completes it. `GuestState`
    /// is the instruction's own (`native_fault`). A translated block that
    /// still leaves the ranges, or a host without the helper, stops
    /// explicitly.
    pub(super) fn run_translated(&mut self, index: usize, pc: u64, address: u64) -> Step {
        let block = &self.code.blocks[index];
        if self.variants[block.variant.0 as usize].translated
            || self.host.translation_helper().is_none()
        {
            let mut requirement = self.requirement(RequirementKind::Native(
                "a mapped guest address outside the host's native ranges",
            ));
            requirement.address = Some(address);
            return Err(Stop::Unsupported(requirement));
        }
        let page = block.physical.wrapping_add(pc.wrapping_sub(block.pc)) & !(FETCH_PAGE - 1);
        self.translated_pages.insert(page);
        self.first_translated.get_or_insert((pc, address));
        self.statistics.translated_pages += 1;
        // The frame's entries go: an ITLB key of the old variant then
        // misses, and the next fetch through it gives its page the new one.
        self.invalidate_code_page(page);
        Ok(())
    }

    /// The guest PC and data address that first needed the translated
    /// variant, once one has: a guest that maps an address outside the
    /// host's native ranges, which runs correctly but slower.
    pub fn first_translated(&self) -> Option<(u64, u64)> {
        self.first_translated
    }

    /// `variant` for a block of the fetch page holding `physical`: in the
    /// translated variant when the page needs it, through the host's
    /// translation helper.
    pub(super) fn page_variant(&self, mut variant: Variant, physical: u64) -> Variant {
        if self.page_translated(physical)
            && let Some(helper) = self.host.translation_helper()
        {
            variant.translated = true;
            variant.checks = variant.checks.with_translation(helper);
        }
        if self.page_tagged(physical) {
            variant.checks = variant.checks.with_top_byte(self.regime_top_byte());
        }
        if variant.checks.range.one_sided && self.code.two_sided.contains(physical) {
            variant.checks.range = variant.checks.range.one_sided(false);
        }
        variant
    }

    /// An EL0 access of the fetch page holding `physical`, by a block of
    /// the one-sided check if `one_sided`, reached the upper range, which
    /// that check refuses and its translation maps: the page goes in the
    /// board's two-sided set and its blocks go, so its next fetch compiles
    /// them with the two-sided check (`RangeCheck::one_sided`). A page the
    /// set holds already loses its blocks again only for a one-sided block.
    pub(super) fn learn_two_sided_page(&mut self, physical: u64, one_sided: bool) {
        let page = physical & !(FETCH_PAGE - 1);
        match self.code.two_sided.insert(page) {
            Inserted::New => self.statistics.two_sided_pages += 1,
            Inserted::Present if one_sided => {}
            Inserted::Present | Inserted::Full => return,
        }
        self.invalidate_code_page(page);
    }

    /// The halves whose data addresses the current translation regime
    /// ignores the top byte of: TCR_ELx.TBI0 and TBI1 (Arm `AddrTop`; every
    /// EL here uses two VA ranges, and TBID concerns only instruction
    /// addresses).
    pub(super) fn regime_top_byte(&self) -> TopByte {
        const TCR_TBI0: u64 = 1 << 37;
        const TCR_TBI1: u64 = 1 << 38;
        let tcr = self.cpu.regime_tcr();
        TopByte::of(tcr & TCR_TBI0 != 0, tcr & TCR_TBI1 != 0)
    }

    /// An access of the fetch page holding `physical`, by a block of the
    /// plain variant if `plain`, had a top byte the regime ignores, which
    /// the shadow does not map: the page goes in the board's tagged set and
    /// its blocks go, so every PE's next fetch compiles them for the tagged
    /// variant, whose accesses clear that byte before their check
    /// (`stela_compiler::x86_64::TopByte`) and so stay native. A page the
    /// set holds already loses its blocks again only for a plain block,
    /// which a PE compiled from a fetch that came before the set took the
    /// page. Past the set's capacity (`page_sets::CAPACITY`) a tagged
    /// access keeps its retry.
    pub(super) fn learn_tagged_page(&mut self, physical: u64, plain: bool) {
        let page = physical & !(FETCH_PAGE - 1);
        match self.code.tagged.insert(page) {
            Inserted::New => self.statistics.tagged_pages += 1,
            Inserted::Present if plain => {}
            Inserted::Present | Inserted::Full => return,
        }
        self.invalidate_code_page(page);
    }

    /// Whether blocks of the fetch page holding `physical` run in the
    /// tagged variant.
    pub(super) fn page_tagged(&self, physical: u64) -> bool {
        self.code.tagged.contains(physical)
    }

    /// The data address a guest fault or device access at host `address`
    /// names for the instruction at `pc` of block `index`. A tagged
    /// variant's access ran with its top byte cleared; the guest's address
    /// keeps it, as the engine's own walk of the access reports it. A host
    /// address inside the stripped access keeps its offset; one below it
    /// in its first page (a translation helper names the page it missed)
    /// is the access's own address; one in its last page, that page's
    /// address with the tag.
    pub(super) fn tagged_fault_address(&mut self, index: usize, pc: u64, address: u64) -> u64 {
        const LOW: u64 = (1 << 56) - 1;
        let top_byte = self.block_variant(index).checks.top_byte;
        if top_byte == TopByte::Kept {
            return address;
        }
        let Ok(word) = self.word(index, pc) else {
            return address;
        };
        let Ok(accesses) = self.data_accesses(pc, word) else {
            return address;
        };
        let page = |address: u64| address & !(FETCH_PAGE - 1);
        for (access, _) in &accesses {
            let start = top_byte.strip(access.address);
            let last = start.wrapping_add(access.size.max(1) - 1);
            if address.wrapping_sub(start) <= last.wrapping_sub(start) {
                return (address & LOW) | (access.address & !LOW);
            }
            if page(address) == page(start) {
                return access.address;
            }
            if page(address) == page(last) {
                return (address & LOW) | (access.address & !LOW);
            }
        }
        address
    }

    /// Whether blocks of the fetch page holding `physical` run in the
    /// translated variant.
    pub(super) fn page_translated(&self, physical: u64) -> bool {
        !self.translated_pages.is_empty()
            && self
                .translated_pages
                .contains(&(physical & !(FETCH_PAGE - 1)))
    }
}
