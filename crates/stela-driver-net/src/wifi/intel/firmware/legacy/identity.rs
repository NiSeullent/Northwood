// SPDX-License-Identifier: GPL-2.0-or-later
//! Direct, read-only 31dc identity and clock observations.
//! Masks follow iwlwifi iwl-csr.h, iwl-config.h and iwl-drv.c at
//! cc9de5d4b5ab6459f5774946aa3195ad51236fa8.

use crate::wifi::intel::power::{AccessError, Csr};

pub const GP_CONTROL: u32 = 0x24;
pub const HW_REVISION: u32 = 0x28;
pub const RF_ID: u32 = 0x9c;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub hw_revision: u32,
    pub rf_id: u32,
    pub gp_control: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Read(AccessError),
    Unavailable,
    MacType(u32),
    BlankRadio,
    RadioType(u32),
    DualBandRadio,
    ClockUnavailable,
    NicAccessBusy,
    Changed,
}

impl Snapshot {
    pub fn read(io: &mut impl Csr) -> Result<Self, Error> {
        Ok(Self {
            hw_revision: io.read32(HW_REVISION).map_err(Error::Read)?,
            rf_id: io.read32(RF_ID).map_err(Error::Read)?,
            gp_control: io.read32(GP_CONTROL).map_err(Error::Read)?,
        })
    }

    pub fn read_awake(io: &mut impl Csr) -> Result<Self, Error> {
        let first = Self::read(io)?;
        first.qualify_awake()?;
        let second = Self::read(io)?;
        second.qualify_awake()?;
        if (first.hw_revision, first.rf_id) != (second.hw_revision, second.rf_id) {
            return Err(Error::Changed);
        }
        Ok(second)
    }

    /// Qualify the direct PU + JF identity for the 31dc shared-clock SKU.
    /// A blank RF type needs a separate, actual CRF observation.
    pub fn qualify_31dc(self) -> Result<(), Error> {
        if [self.hw_revision, self.rf_id, self.gp_control].contains(&u32::MAX) {
            return Err(Error::Unavailable);
        }
        let mac = (self.hw_revision & 0xfff0) >> 4;
        if mac != 0x31 {
            return Err(Error::MacType(mac));
        }
        let radio = (self.rf_id & 0x00ff_f000) >> 12;
        match radio {
            0 => return Err(Error::BlankRadio),
            0x105 | 0x108 => {}
            other => return Err(Error::RadioType(other)),
        }
        if self.rf_id & (1 << 28) != 0 {
            return Err(Error::DualBandRadio);
        }
        Ok(())
    }

    /// Confirm the current MAC clock and lack of an inherited MAC-access request.
    pub fn qualify_awake(self) -> Result<(), Error> {
        self.qualify_31dc()?;
        const READY: u32 = 1 | 1 << 2;
        if self.gp_control & (READY | 1 << 4) != READY {
            return Err(Error::ClockUnavailable);
        }
        if self.gp_control & (1 << 3) != 0 {
            return Err(Error::NicAccessBusy);
        }
        Ok(())
    }

    pub fn rfkill_asserted(self) -> bool {
        self.gp_control & (1 << 27) == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_31dc_identity_distinguishes_blank_or_unavailable_radio_from_valid_jf() {
        let valid = Snapshot {
            hw_revision: 0x31c,
            rf_id: 0x0010_8007,
            gp_control: 5,
        };
        assert_eq!(valid.qualify_awake(), Ok(()));
        assert_eq!(
            Snapshot {
                rf_id: 0x0010_5002,
                ..valid
            }
            .qualify_awake(),
            Ok(())
        );
        for (input, expected) in [
            (Snapshot { rf_id: 0, ..valid }, Error::BlankRadio),
            (
                Snapshot {
                    rf_id: u32::MAX,
                    ..valid
                },
                Error::Unavailable,
            ),
            (
                Snapshot {
                    hw_revision: 0x332,
                    ..valid
                },
                Error::MacType(0x33),
            ),
            (
                Snapshot {
                    rf_id: 0x0010_d000,
                    ..valid
                },
                Error::RadioType(0x10d),
            ),
            (
                Snapshot {
                    rf_id: valid.rf_id | (1 << 28),
                    ..valid
                },
                Error::DualBandRadio,
            ),
            (
                Snapshot {
                    gp_control: 0x15,
                    ..valid
                },
                Error::ClockUnavailable,
            ),
            (
                Snapshot {
                    gp_control: 0x0d,
                    ..valid
                },
                Error::NicAccessBusy,
            ),
        ] {
            assert_eq!(input.qualify_awake(), Err(expected));
        }
    }
}
