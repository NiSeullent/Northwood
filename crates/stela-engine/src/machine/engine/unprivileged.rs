//! LDTR*/STTR*, which the engine performs because native code accesses
//! memory with the current EL's permissions. Arm `AArch64_IsUnprivAccessPriv`
//! makes them unprivileged at EL1, and at EL2 when EL0 is in the host, unless
//! PSTATE.UAO is set; an unprivileged access is checked with EL0 permissions
//! and ignores PAN, in the current translation regime. With FEAT_MTE2 an
//! access whose base is not SP is tag checked as one from the EL it is
//! made at, EL0 when unprivileged.
use super::{
    DeviceError, Devices, FETCH_PAGE, Host, Machine, RamTables, RequirementKind, Step, Stop,
    Translated, device_access::ram_refused, tags::TagCheck,
};
use crate::machine::{
    cpu::Exception,
    debug,
    exception::Class,
    mmu::{self, Access, MemoryType},
};
use alloc::string::ToString;
use alloc::{string::String, vec::Vec};
use stela_compiler::aarch64::{MemoryTransfer, Register, Width, unprivileged::Operand};

const SCTLR_A: u64 = 1 << 1;

/// Arm `AArch64_IsUnprivAccessPriv` without FEAT_NV and EL3.
fn privileged(el: u8, uao: bool, host: bool) -> bool {
    if uao { el != 0 } else { el == 2 && !host }
}

impl<H: Host, D: Devices> Machine<H, D> {
    pub(super) fn unprivileged(&mut self, operand: Operand) -> Step {
        let privileged = privileged(self.cpu.pstate.el, self.cpu.pstate.uao, self.cpu.in_host());
        let base = self.read_register(operand.base);
        if operand.base == Register::Sp
            && self.stack_alignment_checked()
            && !base.is_multiple_of(16)
        {
            self.take(Exception::synchronous(Class::SpAlignment, 0, self.cpu.pc));
            return Ok(());
        }
        let address = base.wrapping_add(operand.displacement as i64 as u64);
        let bytes = usize::from(operand.bytes);
        let write = operand.transfer == MemoryTransfer::Store;
        let aligned = address.is_multiple_of(u64::from(operand.bytes));
        let sctlr = self.cpu.regime_sctlr();
        if !aligned && sctlr & SCTLR_A != 0 {
            return self.alignment_fault(address, write);
        }
        let mut requester = self.cpu.requester();
        requester.unprivileged = !privileged;
        let access = if write { Access::Write } else { Access::Read };
        let registered = self
            .operand_ram()
            .map_err(|reason| self.unsupported(reason))?;
        let mut tables = registered.as_ref().map(|ram| ram.tables());
        // Translate every page piece before touching memory.
        let mut pieces = Vec::new();
        let mut done = 0;
        while done < bytes {
            let virtual_address = address.wrapping_add(done as u64);
            let length =
                ((FETCH_PAGE - (virtual_address & (FETCH_PAGE - 1))) as usize).min(bytes - done);
            let controls = self.cpu.controls(self.cpu.regime());
            let outcome = match tables.as_mut() {
                Some(tables) => tables
                    .translate(&controls, virtual_address, access, requester)
                    .map_err(|reason| self.unsupported(reason))?,
                None => mmu::translate(
                    &controls,
                    virtual_address,
                    access,
                    requester,
                    &mut RamTables(&mut self.host),
                ),
            };
            match outcome {
                Translated::Mapped(translation) => pieces.push((translation, length)),
                Translated::Fault(fault) => {
                    let exception = self.data_abort_exception(virtual_address, fault, write, false);
                    self.take(exception);
                    return Ok(());
                }
                Translated::Unsupported(unsupported) => {
                    return Err(self.unsupported(RequirementKind::Translation(unsupported)));
                }
            }
            done += length;
        }
        if !aligned
            && pieces
                .iter()
                .any(|(translation, _)| translation.memory == MemoryType::Device)
        {
            return self.alignment_fault(address, write);
        }
        // Watchpoints match an unprivileged access as one from EL0.
        let watched = debug::Access {
            address,
            size: u64::from(operand.bytes),
            read: !write,
            write,
            maintenance: false,
            el: if privileged { self.cpu.pstate.el } else { 0 },
        };
        if let Some(exception) = self.watchpoint_exception(&watched, self.cpu.pc) {
            self.take(exception);
            return Ok(());
        }
        if operand.base != Register::Sp
            && let TagCheck::Failed(exception) =
                self.tag_check_with_tables(&watched, requester, None, tables.as_mut())?
        {
            self.take(exception);
            return Ok(());
        }
        let physical_pieces = || {
            pieces
                .iter()
                .map(|&(translation, length)| (translation.physical, length))
        };
        let mut data;
        let ram = if let Some(registered) = &registered {
            registered
                .resident(&self.host, physical_pieces())
                .map_err(|reason| self.unsupported(reason))?
        } else {
            pieces.iter().all(|&(translation, length)| {
                let mut probe = [0u8; 1];
                self.host.read_ram(translation.physical, &mut probe)
                    && self
                        .host
                        .read_ram(translation.physical + length as u64 - 1, &mut probe)
            })
        };
        let value = self.read_register(operand.data);
        if ram {
            data = value.to_le_bytes();
            if let Some(registered) = &registered {
                // SAFETY: the pieces passed preflight under this loan and data is private staging.
                unsafe { registered.transfer(&mut self.host, physical_pieces(), &mut data, write) }
                    .map_err(|reason| self.unsupported(reason))?;
            } else {
                let mut offset = 0;
                for (physical, length) in physical_pieces() {
                    let range = offset..offset + length;
                    if write {
                        self.host.write_ram(physical, &data[range]);
                    } else {
                        self.host.read_ram(physical, &mut data[range]);
                    }
                    offset += length;
                }
            }
        } else {
            // A real device may borrow the exclusive RAM service.
            let canonical = registered.is_some();
            drop(registered);
            let [(translation, _)] = pieces[..] else {
                return Err(self.unprivileged_device(
                    address,
                    pieces[0].0.physical,
                    write,
                    "unprivileged access crossing out of RAM".into(),
                ));
            };
            if canonical
                && write
                && self.cpu.canonical_memory.is_some()
                && self.serial_hold.is_none()
                && self.devices.requires_canonical_write(
                    self.current,
                    translation.physical,
                    operand.bytes,
                )
            {
                return Err(self.unsupported(ram_refused()));
            }
            let now = self.now();
            let result = if write {
                self.device_write(translation.physical, operand.bytes, value, now)
                    .map(|()| 0)
            } else {
                self.device_read(translation.physical, operand.bytes, now)
            };
            match result {
                Ok(read) => data = read.to_le_bytes(),
                Err(DeviceError::NoDevice) => {
                    return Err(self.unprivileged_device(
                        address,
                        translation.physical,
                        write,
                        "no device model".into(),
                    ));
                }
                Err(DeviceError::Refused(detail) | DeviceError::TransferRefused(detail)) => {
                    return Err(self.unprivileged_device(
                        address,
                        translation.physical,
                        write,
                        detail.to_string(),
                    ));
                }
            }
        }
        if let MemoryTransfer::Load { signed, width } = operand.transfer {
            let bits = bytes as u32 * 8;
            let mut loaded = u64::from_le_bytes(data);
            if bits < 64 {
                loaded &= (1 << bits) - 1;
                if signed {
                    loaded = (((loaded << (64 - bits)) as i64) >> (64 - bits)) as u64;
                }
            }
            if width == Width::W32 {
                loaded &= 0xffff_ffff;
            }
            self.write_register(operand.data, loaded);
        }
        self.next()?;
        match self.devices.take_reset() {
            Some(reason) => Err(Stop::Reset(reason)),
            None => Ok(()),
        }
    }

    fn unprivileged_device(
        &self,
        address: u64,
        physical: u64,
        write: bool,
        detail: String,
    ) -> Stop {
        let mut requirement = self.requirement(RequirementKind::Device {
            physical,
            write,
            detail,
        });
        requirement.address = Some(address);
        Stop::Unsupported(requirement)
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod canonical_tests;

#[cfg(test)]
mod tests {
    use super::super::tests::{BASE, machine};
    use super::privileged;
    use crate::machine::system::Register as SystemRegister;

    const LDTR_X1_X2_8: u32 = 0xf840_8841;
    const STTR_X1_X2: u32 = 0xf800_0841;
    const LDTRSB_W3_X2: u32 = 0x38c0_0843;
    const LDTRSB_X3_X2: u32 = 0x3880_0843;

    #[test]
    fn privilege_follows_el_host_and_uao() {
        assert!(!privileged(0, false, false) && !privileged(0, true, true));
        assert!(!privileged(1, false, false) && privileged(1, true, false));
        assert!(!privileged(2, false, true) && privileged(2, false, false));
        assert!(privileged(2, true, true));
        // At EL1 Arm privileges them only for EffectiveHCR_EL2_NVx<1:0>
        // '11', which no chip reaches: HCR_EL2.NV1 reads 0 where FEAT_NV is
        // implemented (no FEAT_E2H0), and NVx is '000' without it. E2H is
        // RES1, so TGE alone puts EL0 in the host.
        let mut machine = machine(&[]);
        machine
            .cpu
            .registers
            .set(SystemRegister::HCR_EL2, 0, 1 << 42 | 1 << 43);
        assert_ne!(machine.cpu.nested() & 0b11, 0b11);
        machine
            .cpu
            .registers
            .set(SystemRegister::HCR_EL2, 0, 1 << 27);
        assert!(machine.cpu.in_host());
    }

    #[test]
    fn pstate_uao_is_written_by_msr_immediate_and_read_at_bit_23() {
        const MSR_UAO_1: u32 = 0xd500_417f;
        const MSR_UAO_0: u32 = 0xd500_407f;
        const MRS_X0_UAO: u32 = 0xd538_4280;
        let mut machine = machine(&[]);
        machine.system_instruction(MSR_UAO_1).unwrap();
        machine.system_instruction(MRS_X0_UAO).unwrap();
        assert_eq!(
            (machine.cpu.pstate.uao, machine.cpu.state.x[0]),
            (true, 1 << 23)
        );
        machine.system_instruction(MSR_UAO_0).unwrap();
        machine.system_instruction(MRS_X0_UAO).unwrap();
        assert_eq!((machine.cpu.pstate.uao, machine.cpu.state.x[0]), (false, 0));
    }

    #[test]
    fn el1_accesses_use_el0_permissions_and_ignore_pan() {
        let mut machine = machine(&[]);
        // 4 KiB, 39-bit VA from level 1 at BASE + 0x4000: VA 0-1 GiB maps
        // BASE as Normal memory, EL0 read/write (AP 0b01).
        let block = BASE | 1 | 1 << 10 | 1 << 6;
        machine.host.ram[0x4000..0x4008].copy_from_slice(&block.to_le_bytes());
        let registers = &mut machine.cpu.registers;
        registers.set(SystemRegister::TCR_EL1, 0, 25 | 1 << 23 | 5 << 32);
        registers.set(SystemRegister::TTBR0_EL1, 0, BASE + 0x4000);
        registers.set(SystemRegister::MAIR_EL1, 0, 0xff);
        registers.set(SystemRegister::SCTLR_EL1, 0, 1);
        registers.set(SystemRegister::VBAR_EL1, 0, 0x2_0000);
        machine.derived_changed();
        machine.cpu.pstate.pan = true;
        machine.cpu.state.x[2] = 0x100;
        machine.host.ram[0x108..0x110].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
        let pc = machine.cpu.pc;
        machine.system_instruction(LDTR_X1_X2_8).unwrap();
        assert_eq!(
            (machine.cpu.state.x[1], machine.cpu.pc),
            (0x1122_3344_5566_7788, pc + 4)
        );
        machine.system_instruction(STTR_X1_X2).unwrap();
        assert_eq!(
            machine.host.ram[0x100..0x108],
            0x1122_3344_5566_7788u64.to_le_bytes()
        );
        machine.host.ram[0x100] = 0x80;
        machine.system_instruction(LDTRSB_W3_X2).unwrap();
        assert_eq!(machine.cpu.state.x[3], 0xffff_ff80);
        machine.system_instruction(LDTRSB_X3_X2).unwrap();
        assert_eq!(machine.cpu.state.x[3], 0xffff_ffff_ffff_ff80);

        // A kernel-only page (AP 0b00): a permission fault at level 1 at EL1,
        // unless PSTATE.UAO makes the access privileged.
        let block = BASE | 1 | 1 << 10;
        machine.host.ram[0x4000..0x4008].copy_from_slice(&block.to_le_bytes());
        machine.cpu.pstate.uao = true;
        machine.system_instruction(LDTR_X1_X2_8).unwrap();
        assert_eq!(machine.cpu.state.x[1], 0x1122_3344_5566_7788);
        machine.cpu.pstate.uao = false;
        let pc = machine.cpu.pc;
        machine.system_instruction(LDTR_X1_X2_8).unwrap();
        assert_eq!(machine.cpu.pc, 0x2_0200);
        let esr = machine.cpu.read(SystemRegister::ESR_EL1);
        assert_eq!((esr >> 26, esr & 0x3f), (0x25, 0b00_1101));
        assert_eq!(
            (
                machine.cpu.read(SystemRegister::FAR_EL1),
                machine.cpu.read(SystemRegister::ELR_EL1)
            ),
            (0x108, pc)
        );

        // A watchpoint on EL0 accesses only (PAC 0b10) catches the
        // unprivileged load at EL1, but not the one UAO makes privileged.
        let block = BASE | 1 | 1 << 10 | 1 << 6;
        machine.host.ram[0x4000..0x4008].copy_from_slice(&block.to_le_bytes());
        let registers = &mut machine.cpu.registers;
        registers.set(SystemRegister::OSLSR_EL1, 0, 0b1000);
        registers.set(SystemRegister::MDSCR_EL1, 0, 1 << 15 | 1 << 13);
        registers.set(SystemRegister::DBGWVR_EL1, 0, 0x108);
        registers.set(
            SystemRegister::DBGWCR_EL1,
            0,
            0xff << 5 | 0b11 << 3 | 0b10 << 1 | 1,
        );
        machine.derived_changed();
        machine.cpu.pstate.daif = 0;
        machine.cpu.pstate.pan = false;
        machine.cpu.pstate.uao = true;
        machine.cpu.state.x[1] = 0;
        machine.system_instruction(LDTR_X1_X2_8).unwrap();
        assert_eq!(machine.cpu.state.x[1], 0x1122_3344_5566_7788);
        machine.cpu.pstate.uao = false;
        let pc = machine.cpu.pc;
        machine.system_instruction(LDTR_X1_X2_8).unwrap();
        let esr = machine.cpu.read(SystemRegister::ESR_EL1);
        assert_eq!(
            (machine.cpu.pc, esr >> 26, esr & 0x3f),
            (0x2_0200, 0x35, 0x22)
        );
        assert_eq!(
            (
                machine.cpu.read(SystemRegister::FAR_EL1),
                machine.cpu.read(SystemRegister::ELR_EL1)
            ),
            (0x108, pc)
        );
    }
}
