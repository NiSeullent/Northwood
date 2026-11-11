// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2005-2015, 2018-2026 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2015-2017 Intel Deutschland GmbH
//! Read-only NVM access and identity for separate Intel 9000 INIT firmware.
//! The ABI and extended NVM fields follow Linux iwlwifi at
//! cc9de5d4b5ab6459f5774946aa3195ad51236fa8.

use super::Firmware;
use crate::wifi::intel::{command::Id, nvm::HardwareAddress};

pub const CHUNK_BYTES: u16 = 2 * 1024;
/// The 9000's total NVM budget across all sections.
pub const CAPACITY: usize = 32 * 1024;
pub const MAC_STRAP: [u32; 2] = [0x388, 0x38c];
pub const MAC_OTP: [u32; 2] = [0x380, 0x384];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Section(u16);

impl Section {
    pub const COUNT: u16 = 13;
    pub const SW: Self = Self(1);
    pub const REGULATORY: Self = Self(3);
    pub const HARDWARE: Self = Self(10);
    pub const MAC_OVERRIDE: Self = Self(11);
    pub const PHY_SKU: Self = Self(12);

    pub const fn new(id: u16) -> Result<Self, Error> {
        if id >= Self::COUNT {
            return Err(Error::Section(id));
        }
        Ok(Self(id))
    }

    pub const fn id(self) -> u16 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    MissingApi,
    UnsupportedApi {
        command: u8,
        notification: u8,
    },
    Section(u16),
    Range {
        offset: u16,
        length: u16,
    },
    Length {
        expected: usize,
        actual: usize,
    },
    ReplyRange {
        requested: u16,
        actual: u16,
    },
    ReplySection {
        expected: Section,
        actual: u16,
    },
    ReplyOffset {
        expected: u16,
        actual: u16,
    },
    Status(u16),
    IdentitySection {
        section: Section,
        minimum: usize,
        actual: usize,
    },
    HardwareAddress,
}

pub(super) fn validate_firmware(firmware: &Firmware<'_>) -> Result<(), Error> {
    let version = firmware.command_version(1, 0x88).ok_or(Error::MissingApi)?;
    // The pinned API 46 images declare this ABI even though MVM submits the
    // bare command ID, whose wide header carries version zero.
    if (version.command_version, version.notification_version) != (3, 0) {
        return Err(Error::UnsupportedApi {
            command: version.command_version,
            notification: version.notification_version,
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    section: Section,
    offset: u16,
    length: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Read<'a> {
    /// A successful reply. A payload shorter than the request ends the section.
    Data(&'a [u8]),
    /// Address zero is unavailable for this section.
    Absent,
    /// An unavailable nonzero address marks the end of a section.
    End,
}

impl Request {
    pub const fn new(section: Section, offset: u16, length: u16) -> Result<Self, Error> {
        if length == 0 || length > CHUNK_BYTES || offset as usize + length as usize > CAPACITY {
            return Err(Error::Range { offset, length });
        }
        Ok(Self {
            section,
            offset,
            length,
        })
    }

    pub const fn section(self) -> Section {
        self.section
    }

    pub const fn offset(self) -> u16 {
        self.offset
    }

    /// MVM passes bare NVM_ACCESS_CMD through its wide-header group conversion.
    pub const fn id(self) -> Id {
        Id {
            group: 0,
            opcode: 0x88,
            version: 0,
        }
    }

    pub const fn payload(self) -> [u8; 8] {
        let [section_low, section_high] = self.section.id().to_le_bytes();
        let [offset_low, offset_high] = self.offset.to_le_bytes();
        let [length_low, length_high] = self.length.to_le_bytes();
        [
            0, // read
            0, // cache
            section_low,
            section_high,
            offset_low,
            offset_high,
            length_low,
            length_high,
        ]
    }

    /// Decode the exact NVM payload from a transport-matched command reply.
    pub fn decode(self, bytes: &[u8]) -> Result<Read<'_>, Error> {
        let (header, data) = bytes.split_at_checked(8).ok_or(Error::Length {
            expected: 8,
            actual: bytes.len(),
        })?;
        let offset = u16::from_le_bytes([header[0], header[1]]);
        let length = u16::from_le_bytes([header[2], header[3]]);
        let section = u16::from_le_bytes([header[4], header[5]]);
        let status = u16::from_le_bytes([header[6], header[7]]);
        if section != self.section.id() {
            return Err(Error::ReplySection {
                expected: self.section,
                actual: section,
            });
        }
        if offset != self.offset {
            return Err(Error::ReplyOffset {
                expected: self.offset,
                actual: offset,
            });
        }
        if length > self.length {
            return Err(Error::ReplyRange {
                requested: self.length,
                actual: length,
            });
        }
        let expected = match status {
            0 => usize::from(length),
            1 if length == 0 => 0,
            _ => return Err(Error::Status(status)),
        };
        if data.len() != expected {
            return Err(Error::Length {
                expected: 8 + expected,
                actual: bytes.len(),
            });
        }
        Ok(match (status, self.offset) {
            (0, _) => Read::Data(data),
            (1, 0) => Read::Absent,
            _ => Read::End,
        })
    }
}

/// Raw extended NVM identity fields and the base MAC selected by the 9000
/// CSR strap/OTP rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Identity {
    pub address: HardwareAddress,
    pub nvm_version: u32,
    pub hardware_addresses: u8,
    pub radio_configuration: u32,
    pub radio_type: u16,
    pub radio_step: u8,
    pub radio_dash: u8,
    pub radio_flavor: u8,
    pub tx_antennas: u8,
    pub rx_antennas: u8,
    pub sku_flags: u32,
}

impl Identity {
    /// Decode the SW and PHY_SKU identity fields using checked MAC CSR words.
    /// Supply OTP words when the strap lacks a usable unicast address.
    pub fn decode(
        software: &[u8],
        phy_sku: &[u8],
        strap: [u32; 2],
        otp: Option<[u32; 2]>,
    ) -> Result<Self, Error> {
        if software.len() < 10 {
            return Err(Error::IdentitySection {
                section: Section::SW,
                minimum: 10,
                actual: software.len(),
            });
        }
        if phy_sku.len() < 8 {
            return Err(Error::IdentitySection {
                section: Section::PHY_SKU,
                minimum: 8,
                actual: phy_sku.len(),
            });
        }
        let address = HardwareAddress::select(strap, otp).map_err(|_| Error::HardwareAddress)?;
        let radio_configuration = u32::from_le_bytes(phy_sku[..4].try_into().unwrap());
        Ok(Self {
            address,
            nvm_version: u32::from_le_bytes(software[..4].try_into().unwrap()),
            hardware_addresses: software[6] & 0xf,
            radio_configuration,
            radio_type: ((radio_configuration >> 12) & 0xfff) as u16,
            radio_step: ((radio_configuration >> 8) & 0xf) as u8,
            radio_dash: ((radio_configuration >> 4) & 0xf) as u8,
            radio_flavor: (radio_configuration & 0xf) as u8,
            tx_antennas: ((radio_configuration >> 24) & 0xf) as u8,
            rx_antennas: (radio_configuration >> 28) as u8,
            sku_flags: u32::from_le_bytes(phy_sku[4..8].try_into().unwrap()),
        })
    }
}

#[cfg(test)]
mod tests;
