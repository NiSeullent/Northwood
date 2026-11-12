// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2002 ShuChen <shuchen@realtek.com.tw>
// Copyright (c) 2003-2007 Francois Romieu <romieu@fr.zoreil.com>
// Copyright (c) a lot of people too. Please respect their work.
// Copyright (c) 2026 Stela contributors
//! Fallible, bounded register protocols adapted from Linux r8169_main.c.
use super::{Registers, Revision, Timer, firmware::ScriptIo, registers::*};
use crate::block::BlockError;
use core::cell::Cell;
mod dp;
mod ep;
mod ocp;

pub(super) struct Hardware<'a, R, T> {
    pub registers: &'a R,
    pub timer: &'a T,
    pub revision: Revision,
    pub function: u8,
    ocp_base: Cell<u16>,
    dp_mdio_failed: Cell<bool>,
}
impl<'a, R: Registers, T: Timer> Hardware<'a, R, T> {
    pub fn new(registers: &'a R, timer: &'a T, revision: Revision, function: u8) -> Self {
        Self {
            registers,
            timer,
            revision,
            function,
            ocp_base: Cell::new(OCP_PHY_BASE),
            dp_mdio_failed: Cell::new(false),
        }
    }
    pub fn poll32(
        &self,
        reg: usize,
        mask: u32,
        high: bool,
        delay: u64,
        tries: usize,
    ) -> Result<(), BlockError> {
        for _ in 0..tries {
            self.timer.delay(delay)?;
            let value = self.registers.read32(reg);
            if value == u32::MAX {
                return Err(BlockError::Device { status: value });
            }
            if (value & mask != 0) == high {
                return Ok(());
            }
        }
        Err(BlockError::Timeout)
    }
    pub fn reset(&self) -> Result<(), BlockError> {
        self.interrupts_off();
        self.registers.write8(COMMAND, RESET);
        for _ in 0..100 {
            self.timer.delay(100)?;
            let cmd = self.registers.read8(COMMAND);
            if cmd == u8::MAX {
                return Err(BlockError::Device { status: 255 });
            }
            if cmd & RESET == 0 {
                return Ok(());
            }
        }
        Err(BlockError::Timeout)
    }
    pub fn interrupts_off(&self) {
        if self.revision.nbase() {
            self.registers.write32(NBASE_INTERRUPT_MASK, 0);
            self.registers.write32(NBASE_INTERRUPT_STATUS, u32::MAX);
        } else {
            self.registers.write16(INTERRUPT_MASK, 0);
            self.registers.write16(INTERRUPT_STATUS, u16::MAX);
        }
    }
    /// rtl_irq_enable: `events` send the function's interrupt.
    pub fn interrupts_on(&self, events: u16) {
        if self.revision.nbase() {
            self.registers
                .write32(NBASE_INTERRUPT_MASK, u32::from(events));
        } else {
            self.registers.write16(INTERRUPT_MASK, events);
        }
    }
    /// rtl_get_events and rtl_ack_events of the status read: every pending
    /// event, cleared at the register's own width, so a later one raises
    /// the interrupt again. Returns the status's low word, which holds the
    /// events every chip has. A low word of all ones is a function that has
    /// left the bus, which rtl8169_interrupt leaves alone.
    pub fn acknowledge(&self) -> u16 {
        let pending = if self.revision.nbase() {
            self.registers.read32(NBASE_INTERRUPT_STATUS)
        } else {
            u32::from(self.registers.read16(INTERRUPT_STATUS))
        };
        if pending == 0 || pending as u16 == u16::MAX {
            return 0;
        }
        if self.revision.nbase() {
            self.registers.write32(NBASE_INTERRUPT_STATUS, pending);
        } else {
            self.registers.write16(INTERRUPT_STATUS, pending as u16);
        }
        pending as u16
    }
    pub fn efuse_read(&self, address: u16) -> Result<u8, BlockError> {
        if address > 0x03ff {
            return Err(BlockError::Unsupported(
                "Realtek efuse address outside hardware range",
            ));
        }
        self.registers
            .write32(EFUSE_ACCESS, u32::from(address) << 8);
        self.poll32(EFUSE_ACCESS, ACCESS, true, 100, 300)?;
        let value = self.registers.read32(EFUSE_ACCESS);
        if value == u32::MAX {
            return Err(BlockError::Device { status: value });
        }
        Ok(value as u8)
    }
    pub fn phy_read(&self, reg: u16) -> Result<u16, BlockError> {
        if reg > 0x1f {
            return Err(BlockError::Unsupported(
                "Realtek MDIO register outside clause 22",
            ));
        }
        if self.revision.modern_phy() {
            if reg == PHY_PAGE {
                return Ok(if self.ocp_base.get() == OCP_PHY_BASE {
                    0
                } else {
                    self.ocp_base.get() >> 4
                });
            }
            let address = self.phy_address(reg)?;
            self.phy_ocp_read(address)
        } else {
            self.dp_mdio(|| {
                self.registers.write32(PHY_ACCESS, u32::from(reg) << 16);
                let done = self.poll32(PHY_ACCESS, ACCESS, true, 25, 20);
                self.timer.delay(20)?; // Required after completion, including failure.
                done?;
                Ok(self.registers.read32(PHY_ACCESS) as u16)
            })
        }
    }
    pub fn phy_write(&self, reg: u16, value: u16) -> Result<(), BlockError> {
        if reg > 0x1f {
            return Err(BlockError::Unsupported(
                "Realtek MDIO register outside clause 22",
            ));
        }
        if self.revision.modern_phy() {
            if reg == PHY_PAGE {
                self.ocp_base.set(if value == 0 {
                    OCP_PHY_BASE
                } else {
                    value
                        .checked_mul(16)
                        .ok_or(BlockError::Unsupported("Realtek OCP page overflow"))?
                });
                return Ok(());
            }
            let address = self.phy_address(reg)?;
            if self.revision.version == 40 && address == OCP_PHY_BASE {
                // rtl8168g_phy_suspend_quirk: MCU IRQs must be gated before
                // powerdown and restored before reset or leaving powerdown.
                if value & PHY_RESET != 0 || value & PHY_POWER_DOWN == 0 {
                    self.eri_modify(0x1a8, 0, 0xfc00_0000)?;
                } else {
                    self.eri_modify(0x1a8, 0xfc00_0000, 0)?;
                }
            }
            self.phy_ocp_write(address, value)
        } else {
            self.dp_mdio(|| {
                self.registers.write32(
                    PHY_ACCESS,
                    ACCESS | (u32::from(reg) << 16) | u32::from(value),
                );
                let done = self.poll32(PHY_ACCESS, ACCESS, false, 25, 20);
                self.timer.delay(20)?;
                done
            })
        }
    }
    fn phy_address(&self, reg: u16) -> Result<u16, BlockError> {
        let offset = i32::from(reg)
            - if self.ocp_base.get() == OCP_PHY_BASE {
                0
            } else {
                16
            };
        u16::try_from(i32::from(self.ocp_base.get()) + offset * 2)
            .map_err(|_| BlockError::Unsupported("Realtek OCP address overflow"))
    }
    pub fn phy_modify(&self, reg: u16, clear: u16, set: u16) -> Result<(), BlockError> {
        self.phy_write(reg, (self.phy_read(reg)? & !clear) | set)
    }
    pub fn paged(&self, page: u16, reg: u16, clear: u16, set: u16) -> Result<(), BlockError> {
        let old = self.phy_read(PHY_PAGE)?;
        self.phy_write(PHY_PAGE, page)?;
        let result = self.phy_modify(reg, clear, set);
        let restored = self.phy_write(PHY_PAGE, old);
        result.and(restored)
    }
    pub fn paged_read(&self, page: u16, reg: u16) -> Result<u16, BlockError> {
        let old = self.phy_read(PHY_PAGE)?;
        self.phy_write(PHY_PAGE, page)?;
        let value = self.phy_read(reg);
        let restored = self.phy_write(PHY_PAGE, old);
        let value = value?;
        restored?;
        Ok(value)
    }
    pub fn phy_reset(&self) -> Result<(), BlockError> {
        self.phy_write(PHY_PAGE, 0)?;
        self.phy_modify(PHY_CONTROL, 0, PHY_RESET)?;
        self.timer.delay(50_000)?;
        for _ in 0..12 {
            if self.phy_read(PHY_CONTROL)? & PHY_RESET == 0 {
                return Ok(());
            }
            self.timer.delay(50_000)?;
        }
        Err(BlockError::Timeout)
    }
    pub fn ephy(&self, reg: u8, clear: u16, set: u16) -> Result<(), BlockError> {
        if reg > 31 {
            return Err(BlockError::Unsupported(
                "Realtek EPHY register outside range",
            ));
        }
        self.registers.write32(EPHY_ACCESS, u32::from(reg) << 16);
        self.poll32(EPHY_ACCESS, ACCESS, true, 10, 100)?;
        let value = (self.registers.read32(EPHY_ACCESS) as u16 & !clear) | set;
        self.timer.delay(10)?;
        self.registers.write32(
            EPHY_ACCESS,
            ACCESS | (u32::from(reg) << 16) | u32::from(value),
        );
        let result = self.poll32(EPHY_ACCESS, ACCESS, false, 10, 100);
        self.timer.delay(10)?;
        result
    }
    pub fn eri_read(&self, reg: u16) -> Result<u32, BlockError> {
        if reg & 3 != 0 {
            return Err(BlockError::Unsupported("unaligned Realtek ERI register"));
        }
        self.registers.write32(ERI_ADDRESS, 0xf000 | u32::from(reg));
        self.poll32(ERI_ADDRESS, ACCESS, true, 100, 100)?;
        Ok(self.registers.read32(ERI_DATA))
    }
    pub fn eri_write(&self, reg: u16, mask: u8, value: u32) -> Result<(), BlockError> {
        if reg & 3 != 0 || mask == 0 || mask > 15 {
            return Err(BlockError::Unsupported(
                "invalid Realtek ERI address or byte mask",
            ));
        }
        self.registers.write32(ERI_DATA, value);
        self.registers.write32(
            ERI_ADDRESS,
            ACCESS | (u32::from(mask) << 12) | u32::from(reg),
        );
        self.poll32(ERI_ADDRESS, ACCESS, false, 100, 100)
    }
    pub fn eri_modify(&self, reg: u16, clear: u32, set: u32) -> Result<(), BlockError> {
        self.eri_write(reg, 15, (self.eri_read(reg)? & !clear) | set)
    }
    /// The CSI fallback supported by Linux for config dword 0x70c. Stela uses
    /// this documented controller protocol directly; it does not guess ECAM size.
    pub fn latency(&self) -> Result<(), BlockError> {
        self.csi_modify(0x70c, 0xff00_0000, 0x2700_0000)
    }
    pub(super) fn csi_modify(&self, address: u16, clear: u32, set: u32) -> Result<(), BlockError> {
        if address & 3 != 0 || address > 0xfff || self.function > 7 {
            return Err(BlockError::Unsupported(
                "invalid Realtek CSI address or PCI function",
            ));
        }
        let cmd = u32::from(address) | 0xf000 | (u32::from(self.function) << 16);
        self.registers.write32(CSI_ADDRESS, cmd);
        self.poll32(CSI_ADDRESS, ACCESS, true, 10, 100)?;
        let value = (self.mmio32(CSI_DATA)? & !clear) | set;
        self.registers.write32(CSI_DATA, value);
        self.registers.write32(CSI_ADDRESS, ACCESS | cmd);
        self.poll32(CSI_ADDRESS, ACCESS, false, 10, 100)
    }
    fn mac_address(&self, reg: u16) -> Result<u16, BlockError> {
        let address = self
            .ocp_base
            .get()
            .checked_add(reg)
            .ok_or(BlockError::Unsupported("Realtek MCU OCP address overflow"))?;
        if address & 1 != 0 {
            return Err(BlockError::Unsupported("unaligned Realtek MCU OCP address"));
        }
        Ok(address)
    }
}
impl<R: Registers, T: Timer> ScriptIo for Hardware<'_, R, T> {
    fn read(&mut self, mac: bool, reg: u16) -> Result<u16, BlockError> {
        if !mac {
            return self.phy_read(reg);
        }
        let address = self.mac_address(reg)?;
        self.mac_ocp_read(address)
    }
    fn write(&mut self, mac: bool, reg: u16, value: u16) -> Result<(), BlockError> {
        if !mac {
            return self.phy_write(reg, value);
        }
        if reg == PHY_PAGE {
            self.ocp_base.set(
                value
                    .checked_mul(16)
                    .ok_or(BlockError::Unsupported("Realtek MCU page overflow"))?,
            );
            return Ok(());
        }
        let address = self.mac_address(reg)?;
        self.mac_ocp_write(address, value)
    }
    fn restore(&mut self) {
        self.ocp_base.set(OCP_PHY_BASE);
    }
}
