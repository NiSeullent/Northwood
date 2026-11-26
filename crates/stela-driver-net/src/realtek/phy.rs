// SPDX-License-Identifier: GPL-2.0-only AND GPL-2.0-or-later
// PHY identity/resume portions from Linux realtek_main.c (GPL-2.0-or-later):
// Copyright (c) 2004 Freescale Semiconductor, Inc.
// Author: Johnson Leung <r58129@freescale.com>
// Copyright (c) 2002 ShuChen <shuchen@realtek.com.tw>
// Copyright (c) 2003-2007 Francois Romieu <romieu@fr.zoreil.com>
// Copyright (c) a lot of people too. Please respect their work.
// Copyright (c) 2026 Stela contributors
//! Revision-specific analogue PHY programming from the pinned Linux source.
use super::{Error, Registers, Timer, Warning, firmware::Script, hardware::Hardware, registers::*};
use crate::block::BlockError;
mod d;
mod e;
mod ep;
mod f;
mod g;
mod h;
mod nbase;

const S: &[(u16, u16)] = &[
    (0x1f, 0x0001),
    (0x06, 0x006e),
    (0x08, 0x0708),
    (0x15, 0x4000),
    (0x18, 0x65c7),
    (0x1f, 0x0001),
    (0x03, 0x00a1),
    (0x02, 0x0008),
    (0x01, 0x0120),
    (0x00, 0x1000),
    (0x04, 0x0800),
    (0x04, 0x0000),
    (0x03, 0xff41),
    (0x02, 0xdf60),
    (0x01, 0x0140),
    (0x00, 0x0077),
    (0x04, 0x7800),
    (0x04, 0x7000),
    (0x03, 0x802f),
    (0x02, 0x4f02),
    (0x01, 0x0409),
    (0x00, 0xf0f9),
    (0x04, 0x9800),
    (0x04, 0x9000),
    (0x03, 0xdf01),
    (0x02, 0xdf20),
    (0x01, 0xff95),
    (0x00, 0xba00),
    (0x04, 0xa800),
    (0x04, 0xa000),
    (0x03, 0xff41),
    (0x02, 0xdf20),
    (0x01, 0x0140),
    (0x00, 0x00bb),
    (0x04, 0xb800),
    (0x04, 0xb000),
    (0x03, 0xdf41),
    (0x02, 0xdc60),
    (0x01, 0x6340),
    (0x00, 0x007d),
    (0x04, 0xd800),
    (0x04, 0xd000),
    (0x03, 0xdf01),
    (0x02, 0xdf20),
    (0x01, 0x100a),
    (0x00, 0xa0ff),
    (0x04, 0xf800),
    (0x04, 0xf000),
    (0x1f, 0x0000),
    (0x0b, 0x0000),
    (0x00, 0x9200),
];
const SCD: &[(u16, u16)] = &[
    (0x1f, 0x0001),
    (0x04, 0x0000),
    (0x03, 0x00a1),
    (0x02, 0x0008),
    (0x01, 0x0120),
    (0x00, 0x1000),
    (0x04, 0x0800),
    (0x04, 0x9000),
    (0x03, 0x802f),
    (0x02, 0x4f02),
    (0x01, 0x0409),
    (0x00, 0xf099),
    (0x04, 0x9800),
    (0x04, 0xa000),
    (0x03, 0xdf01),
    (0x02, 0xdf20),
    (0x01, 0xff95),
    (0x00, 0xba00),
    (0x04, 0xa800),
    (0x04, 0xf000),
    (0x03, 0xdf01),
    (0x02, 0xdf20),
    (0x01, 0x101a),
    (0x00, 0xa0ff),
    (0x04, 0xf800),
    (0x04, 0x0000),
    (0x1f, 0x0000),
    (0x1f, 0x0001),
    (0x10, 0xf41b),
    (0x14, 0xfb54),
    (0x18, 0xf5c7),
    (0x1f, 0x0000),
    (0x1f, 0x0001),
    (0x17, 0x0cc0),
    (0x1f, 0x0000),
];
const SCE: &[(u16, u16)] = &[
    (0x1f, 0x0001),
    (0x04, 0x0000),
    (0x03, 0x00a1),
    (0x02, 0x0008),
    (0x01, 0x0120),
    (0x00, 0x1000),
    (0x04, 0x0800),
    (0x04, 0x9000),
    (0x03, 0x802f),
    (0x02, 0x4f02),
    (0x01, 0x0409),
    (0x00, 0xf099),
    (0x04, 0x9800),
    (0x04, 0xa000),
    (0x03, 0xdf01),
    (0x02, 0xdf20),
    (0x01, 0xff95),
    (0x00, 0xba00),
    (0x04, 0xa800),
    (0x04, 0xf000),
    (0x03, 0xdf01),
    (0x02, 0xdf20),
    (0x01, 0x101a),
    (0x00, 0xa0ff),
    (0x04, 0xf800),
    (0x04, 0x0000),
    (0x1f, 0x0000),
    (0x1f, 0x0001),
    (0x0b, 0x8480),
    (0x1f, 0x0000),
    (0x1f, 0x0001),
    (0x18, 0x67c7),
    (0x04, 0x2000),
    (0x03, 0x002f),
    (0x02, 0x4360),
    (0x01, 0x0109),
    (0x00, 0x3022),
    (0x04, 0x2800),
    (0x1f, 0x0000),
    (0x1f, 0x0001),
    (0x17, 0x0cc0),
    (0x1f, 0x0000),
];
const C1: &[(u16, u16)] = &[
    (0x1f, 0x0001),
    (0x12, 0x2300),
    (0x1f, 0x0002),
    (0x00, 0x88d4),
    (0x01, 0x82b1),
    (0x03, 0x7002),
    (0x08, 0x9e30),
    (0x09, 0x01f0),
    (0x0a, 0x5500),
    (0x0c, 0x00c8),
    (0x1f, 0x0003),
    (0x12, 0xc096),
    (0x16, 0x000a),
    (0x1f, 0x0000),
    (0x1f, 0x0000),
    (0x09, 0x2000),
    (0x09, 0x0000),
];
const C2: &[(u16, u16)] = &[
    (0x1f, 0x0001),
    (0x12, 0x2300),
    (0x03, 0x802f),
    (0x02, 0x4f02),
    (0x01, 0x0409),
    (0x00, 0xf099),
    (0x04, 0x9800),
    (0x04, 0x9000),
    (0x1d, 0x3d98),
    (0x1f, 0x0002),
    (0x0c, 0x7eb8),
    (0x06, 0x0761),
    (0x1f, 0x0003),
    (0x16, 0x0f0a),
    (0x1f, 0x0000),
];
const C3: &[(u16, u16)] = &[
    (0x1f, 0x0001),
    (0x12, 0x2300),
    (0x1d, 0x3d98),
    (0x1f, 0x0002),
    (0x0c, 0x7eb8),
    (0x06, 0x5461),
    (0x1f, 0x0003),
    (0x16, 0x0f0a),
    (0x1f, 0x0000),
];

impl<R: Registers, T: Timer> Hardware<'_, R, T> {
    fn batch(&self, regs: &[(u16, u16)]) -> Result<(), BlockError> {
        for &(reg, value) in regs {
            self.phy_write(reg, value)?;
        }
        Ok(())
    }
    fn parameter(&self, parameter: u16, clear: u16, set: u16) -> Result<(), BlockError> {
        let old = self.phy_read(PHY_PAGE)?;
        self.phy_write(PHY_PAGE, 5)?;
        let result = (|| {
            self.phy_write(5, parameter)?;
            self.phy_modify(6, clear, set)
        })();
        let restore = self.phy_write(PHY_PAGE, old);
        result.and(restore)
    }
    fn extended_page(&self, page: u16, reg: u16, clear: u16, set: u16) -> Result<(), BlockError> {
        let old = self.phy_read(PHY_PAGE)?;
        self.phy_write(PHY_PAGE, 7)?;
        let result = (|| {
            self.phy_write(0x1e, page)?;
            self.phy_modify(reg, clear, set)
        })();
        let restore = self.phy_write(PHY_PAGE, old);
        result.and(restore)
    }
    pub(super) fn apply_firmware(&mut self, script: &Script<'_>) -> Result<(), Error> {
        let timer = self.timer;
        script.execute(self, timer)?;
        // Linux r8169_apply_firmware restores the shared OCP base and then
        // waits up to 600ms for a reset that the script may have started.
        self.phy_write(PHY_PAGE, 0)?;
        let mut ready = false;
        for _ in 0..12 {
            self.timer.delay(50_000)?;
            if self.phy_read(PHY_CONTROL)? & PHY_RESET == 0 {
                ready = true;
                break;
            }
        }
        if !ready {
            return Err(BlockError::Timeout.into());
        }
        Ok(())
    }
    fn evl(&mut self, script: &Script<'_>) -> Result<(), Error> {
        self.apply_firmware(script)?;
        self.extended_page(0xac, 0x18, 0xffff, 6)?;
        self.paged(3, 9, 0xffff, 0xa20f)?;
        self.parameter(0x8b5b, 0xffff, 0x9222)?;
        self.parameter(0x8b6d, 0xffff, 0x8000)?;
        self.parameter(0x8b76, 0xffff, 0x8000)?;
        self.phy_write(PHY_PAGE, 5)?;
        self.phy_write(5, 0x8b80)?;
        self.phy_modify(0x17, 0, 6)?;
        self.phy_write(PHY_PAGE, 0)?;
        self.extended_page(0x2d, 0x18, 0, 0x10)?;
        self.phy_modify(0x14, 0, 1 << 15)?;
        self.parameter(0x8b86, 0, 1)?;
        self.parameter(0x8b85, 0, 0x4000)?;
        // Linux rtl8168f_config_eee_phy analogue tuning. This does not
        // advertise EEE; the MAC TX-LPI mode stays off.
        self.extended_page(0x20, 0x15, 0, 1 << 8)?;
        self.parameter(0x8b85, 0, 1 << 13)?;
        self.phy_write(PHY_PAGE, 3)?;
        self.phy_modify(0x19, 0, 1)?;
        self.phy_modify(0x10, 0, 1 << 10)?;
        self.phy_write(PHY_PAGE, 0)?;
        self.paged(5, 1, 0, 1 << 8)?;
        Ok(())
    }
    pub(super) fn configure_phy(
        &mut self,
        script: Option<&Script<'_>>,
        gigabyte_sc: bool,
    ) -> Result<Option<Warning>, Error> {
        self.phy_write(PHY_PAGE, 0)?;
        let reported = (u32::from(self.phy_read(2)?) << 16) | u32::from(self.phy_read(3)?);
        // Upstream erratum 62bdc8fd1c21: these exact MACs integrate RTL8211B
        // but can report the generic FE/GE ID. Preserve reads and make
        // the narrower, documented classification correction observable.
        let (id, mut warning) =
            if matches!(self.revision.version, 28 | 31) && reported == 0x001c_c800 {
                (
                    0x001c_c912,
                    Some(Warning::DpPhyIdentityCorrected {
                        reported,
                        selected: 0x001c_c912,
                    }),
                )
            } else {
                (reported, None)
            };
        // This slice implements the internal FE/GE path and RTL8211B's
        // ID-specific resume quirk, followed by generic reset/autonegotiation.
        // Other external PHY IDs need their own realtek_main.c quirks.
        let supported = if self.revision.nbase() {
            matches!(id, 0x001c_c800 | 0x001c_c840)
                || (matches!(self.revision.version, 64..=66 | 70)
                    && matches!(id, 0x001c_c841 | 0x001c_c862 | 0x001c_c890))
        } else {
            matches!(id, 0x001c_c800 | 0x001c_c912)
        };
        if !supported {
            return Err(Error::UnsupportedPhy(id));
        }
        if matches!(
            self.revision.version,
            32 | 33 | 35 | 36 | 38 | 40 | 42 | 44 | 46 | 51 | 52
        ) && id != 0x001c_c800
        {
            // These E/F/8411 paths implement the internal FE/GE analogue path;
            // external PHY initialization remains outside these new revisions.
            return Err(Error::UnsupportedPhy(id));
        }
        if id == 0x001c_c912 {
            // rtl8211b_resume clears MII_MMD_DATA before genphy_resume:
            // rtl8211b_suspend may have left its bit 9 set.
            self.phy_write(14, 0)?;
        }
        self.phy_modify(PHY_CONTROL, 1 << 11, 0)?; // genphy_resume
        self.timer.delay(20_000)?;
        let mut nbase_capability = 0;
        if id == 0x001c_c800 || self.revision.nbase() {
            // realtek_main.c distinguishes the generic FE/GE PHY from the
            // identical-ID 2.5G PHY by vendor page a61's capability register.
            self.phy_write(PHY_PAGE, 0x0a61)?;
            let capability = self.phy_read(0x13);
            let restored = self.phy_write(PHY_PAGE, 0);
            let capability = capability?;
            restored?;
            if (capability & 0x2000 != 0) != self.revision.nbase() {
                return Err(Error::UnsupportedPhy(id));
            }
            nbase_capability = capability;
            if self.revision.nbase() && !self.nbase_internal_phy()? {
                return Err(Error::UnsupportedPhy(id));
            }
        }
        let status = self.phy_read(1)?;
        if status & (1 << 8) == 0 || self.phy_read(15)? & 0x3000 == 0 {
            return Err(Error::Script("Realtek PHY lacks copper gigabit capability"));
        }
        let result = (|| {
            match self.revision.version {
                2 | 3 => self.batch(S)?,
                4 => self.paged(2, 1, 0xffff, 0x90d0)?,
                5 => self.batch(SCD)?,
                6 => self.batch(SCE)?,
                17 => self.paged(1, 0x10, 0xffff, 0xf41b)?,
                18 => {
                    self.phy_write(0x1d, 0x0f00)?;
                    self.paged(2, 0x0c, 0xffff, 0x1ec8)?;
                }
                19 => {
                    self.batch(C1)?;
                    self.phy_modify(0x14, 0, 1 << 5)?;
                    self.phy_modify(0x0d, 0, 1 << 5)?;
                }
                20 => {
                    self.batch(C2)?;
                    self.c_tail()?;
                }
                21 | 22 => {
                    self.batch(C3)?;
                    self.c_tail()?;
                }
                23 | 24 => {
                    self.phy_modify(0x14, 0, 1 << 5)?;
                    self.phy_modify(0x0d, 0, 1 << 5)?;
                    self.paged(1, 0x1d, 0xffff, 0x3d98)?;
                }
                25 | 26 => self.d(script.ok_or(Error::Script("RTL8168D requires firmware"))?)?,
                28 => {
                    self.paged(1, 0x17, 0xffff, 0x0cc0)?;
                    self.extended_page(0x2d, 0x18, 0xffff, 0x0040)?;
                    self.phy_modify(0x0d, 0, 1 << 5)?;
                }
                31 => {} // Pinned donor's MAC31 PHY dispatch is NULL: generic resume/reset/aneg only.
                32 | 33 => {
                    warning =
                        Some(self.e(script.ok_or(Error::Script("RTL8168E requires firmware"))?)?)
                }
                35 | 36 | 38 => {
                    self.f(script.ok_or(Error::Script("RTL8168F/8411 requires firmware"))?)?
                }
                40 | 42 | 44 => {
                    self.g(script.ok_or(Error::Script("RTL8168G/8411B requires firmware"))?)?
                }
                46 => self.h(script.ok_or(Error::Script("RTL8168H/M requires firmware"))?)?,
                51 => self.ep_phy()?,
                52 => self.fp_phy()?, // XID54a's MAC firmware executes after MAC setup.
                61 | 63..=66 | 70 => self.nbase_phy(
                    script.ok_or(Error::Script("Realtek NBASE requires revision firmware"))?,
                )?,
                34 => self.evl(script.ok_or(Error::Script("RTL8168E-VL requires firmware"))?)?,
                _ => return Err(Error::UnsupportedRevision(self.revision)),
            }
            if gigabyte_sc && self.revision.version == 5 {
                self.paged(1, 0x10, 0xffff, 0xf01b)?;
            }
            self.phy_reset()?;
            if matches!(self.revision.version, 40 | 42 | 44 | 46 | 51 | 52) {
                // The generic PHY's AN EEE_ADV mapping is VEND2 a5d0.
                // MAC40+ has direct PHY-OCP/C45 access. Clear it after
                // reset, before restarting autonegotiation; no EEE advertised.
                self.phy_ocp_write(PHY_OCP_EEE_ADV, 0)?;
            }
            if self.revision.nbase() {
                self.phy_ocp_write(PHY_OCP_EEE_ADV, 0)?;
                self.phy_ocp_write(NBASE_EEE_ADV2, 0)?;
                let advertised = self.phy_ocp_read(NBASE_AN_CONTROL)?;
                self.phy_ocp_write(
                    NBASE_AN_CONTROL,
                    (advertised & !(NBASE_ADV_2500 | NBASE_ADV_5000 | NBASE_ADV_10000))
                        | NBASE_ADV_2500
                        | if self.revision.version == 70 && nbase_capability & 0x4000 != 0 {
                            NBASE_ADV_5000
                        } else {
                            0
                        },
                )?;
            }
            // IEEE selector, 10/100 half/full, and symmetric and asymmetric
            // pause, as r8169 advertises (`phy_support_asym_pause`): the MAC
            // runs flow control from the negotiated result in hardware, with
            // the FIFO pause thresholds its start set.
            self.phy_write(4, 0x0de1)?;
            self.phy_modify(9, (3 << 8) | (3 << 11) | (7 << 13), 1 << 9)?; // 1000 full, auto master.
            self.phy_write(PHY_CONTROL, (1 << 12) | (1 << 9))?; // restart autonegotiation.
            Ok(warning)
        })();
        let restored = self.phy_write(PHY_PAGE, 0).map_err(Error::from);
        let warning = result?;
        restored?;
        Ok(warning)
    }
    fn c_tail(&self) -> Result<(), BlockError> {
        self.phy_modify(0x16, 0, 1)?;
        self.phy_modify(0x14, 0, 1 << 5)?;
        self.phy_modify(0x0d, 0, 1 << 5)
    }
}
