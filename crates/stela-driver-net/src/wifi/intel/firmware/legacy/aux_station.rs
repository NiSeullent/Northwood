// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2005-2015, 2018-2026 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2015-2017 Intel Deutschland GmbH
//! First auxiliary station on a fresh Intel 9000 API 46 REGULAR station table.
//! Wire formats and ordering follow Linux iwlwifi mvm/sta.c and fw/api/{tx,sta}.h
//! at cc9de5d4b5ab6459f5774946aa3195ad51236fa8.

use super::{API, Firmware};
use crate::wifi::intel::{command::Id, firmware::CommandVersion};

const SCD_QUEUE_CFG: Id = Id {
    group: 1,
    opcode: 0x1d,
    version: 0,
};
const ADD_STA: Id = Id {
    group: 1,
    opcode: 0x18,
    version: 0,
};
const AUX_QUEUE: u8 = 1;
const AUX_STATION: u8 = 1; // The first internal station skips the AP's reserved ID 0.
const AUX_MAC_INDEX: u32 = 4; // MAC index 4, color 0.
const AUX_STATION_TYPE: u8 = 4;
const MANAGEMENT_TID: u8 = 8;
const MULTICAST_FIFO: u8 = 5;
const FRAME_LIMIT: u8 = 64;
const ENABLE_QUEUE: u8 = 1;
const DQA_CAPABILITY: u32 = 12;
const STATION_TYPE_API: u32 = 30;
const MLD_CAPABILITY: u32 = 110;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Profile,
    Version {
        command: u8,
        declared: Option<CommandVersion>,
    },
    DuplicateVersion(u8),
    StatusLength(usize),
    Status(u32),
}

/// Wire data for the first AUX allocation, before any other station exists.
/// A caller must own the queue and keep the actual DQA and LARI witnesses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    queue: [u8; 12],
    station: [u8; 48],
}

impl Plan {
    pub fn for_fresh_api46(firmware: &Firmware<'_>) -> Result<Self, Error> {
        if firmware.api_version() != API
            || !firmware.supports_capability(DQA_CAPABILITY)
            || !firmware.supports_api(STATION_TYPE_API)
            || firmware.supports_capability(MLD_CAPABILITY)
        {
            return Err(Error::Profile);
        }
        require_version(firmware, SCD_QUEUE_CFG, 1, 1)?;
        require_version(firmware, ADD_STA, 10, 2)?;

        // Token 0, no aggregation, SSN 0 and reserved 0.
        let queue = [
            0,
            AUX_STATION,
            MANAGEMENT_TID,
            AUX_QUEUE,
            ENABLE_QUEUE,
            0,
            MULTICAST_FIFO,
            FRAME_LIMIT,
            0,
            0,
            0,
            0,
        ];
        let mut station = [0; 48];
        station[2..4].copy_from_slice(&u16::MAX.to_le_bytes()); // tid_disable_tx
        station[4..8].copy_from_slice(&AUX_MAC_INDEX.to_le_bytes());
        station[16] = AUX_STATION;
        station[35] = AUX_STATION_TYPE;
        station[40..44].copy_from_slice(&(1u32 << AUX_QUEUE).to_le_bytes());
        Ok(Self { queue, station })
    }

    /// Value for HBUS_TARG_WRPTR to initialize AUX queue 1 to slot 0 before
    /// submitting its scheduler configuration on the host command queue.
    pub const fn initial_write_pointer(&self) -> u32 {
        (AUX_QUEUE as u32) << 8
    }

    pub fn configure_queue(&self) -> (Id, &[u8; 12]) {
        (SCD_QUEUE_CFG, &self.queue)
    }

    /// Follows the scheduler configuration's matching command completion.
    pub fn add_station(&self) -> (Id, &[u8; 48]) {
        (ADD_STA, &self.station)
    }
}

fn require_version(
    firmware: &Firmware<'_>,
    id: Id,
    command_version: u8,
    notification_version: u8,
) -> Result<(), Error> {
    let mut rows = firmware
        .command_versions()
        .unwrap_or_default()
        .iter()
        .filter(|row| (row.group, row.command) == (id.group, id.opcode));
    let declared = rows.next().copied();
    if rows.next().is_some() {
        return Err(Error::DuplicateVersion(id.opcode));
    }
    if !declared.is_some_and(|row| {
        (row.command_version, row.notification_version) == (command_version, notification_version)
    }) {
        return Err(Error::Version {
            command: id.opcode,
            declared,
        });
    }
    Ok(())
}

/// Decode the payload of an already matched solicited ADD_STA response.
pub fn check_station_status(payload: &[u8]) -> Result<(), Error> {
    let status = u32::from_le_bytes(
        payload
            .try_into()
            .map_err(|_| Error::StatusLength(payload.len()))?,
    );
    if status & 0xff == 1 {
        Ok(())
    } else {
        Err(Error::Status(status))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::wifi::intel::{command::Command, firmware::legacy::Sku};

    #[test]
    #[ignore = "oracle: STELA_IWLWIFI_IMAGES"]
    fn pinned_api46_aux_queue_and_station_match_the_legacy_wire_formats() {
        let root = std::env::var("STELA_IWLWIFI_IMAGES").unwrap();
        for sku in [Sku::Cnvi9000, Sku::WirelessAc9260] {
            let bytes =
                std::fs::read(std::path::Path::new(&root).join(sku.firmware_name())).unwrap();
            let fw = Firmware::parse(&bytes).unwrap();
            let plan = Plan::for_fresh_api46(&fw).unwrap();
            assert_eq!(plan.initial_write_pointer(), 0x100);

            let (id, payload) = plan.configure_queue();
            let command = Command::encode(id, 0, 12, payload).unwrap();
            assert_eq!(
                command.first_tb(),
                [
                    0x1d, 1, 12, 0, 12, 0, 0, 0, 0, 1, 8, 1, 1, 0, 5, 64, 0, 0, 0, 0
                ]
            );
            assert!(command.remainder().is_empty());

            let (id, payload) = plan.add_station();
            let command = Command::encode(id, 0, 13, payload).unwrap();
            assert_eq!(
                command.first_tb(),
                [
                    0x18, 1, 13, 0, 48, 0, 0, 0, 0, 0, 0xff, 0xff, 4, 0, 0, 0, 0, 0, 0, 0
                ]
            );
            assert_eq!(
                command.remainder(),
                [
                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0,
                    0, 0, 2, 0, 0, 0, 0, 0, 0, 0
                ]
            );
        }
    }

    #[test]
    fn add_station_requires_exactly_one_status_word_and_the_success_code() {
        assert_eq!(check_station_status(&[1, 2, 3, 4]), Ok(()));
        for status in [0u32, 2, 4, 8, 0x103] {
            assert_eq!(
                check_station_status(&status.to_le_bytes()),
                Err(Error::Status(status))
            );
        }
        for length in [0, 3, 5] {
            assert_eq!(
                check_station_status(&[1, 0, 0, 0, 0][..length]),
                Err(Error::StatusLength(length))
            );
        }
    }
}
