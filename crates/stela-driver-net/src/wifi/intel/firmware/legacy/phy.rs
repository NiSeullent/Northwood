// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2005-2015, 2018-2026 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2015-2017 Intel Deutschland GmbH
//! Intel 9000 INIT and REGULAR calibration command ABI from pinned iwlwifi `iwl-drv.c`,
//! `mvm/fw.c`, `mvm/mvm.h`, and `fw/api/config.h` at 347246472df8.

use super::{API, Firmware, alive::Image};
use crate::{bytes::le32, wifi::intel::command::Id};

const DEFAULT_CALIBRATION: u32 = 22;
const REGULAR_IMAGE: u32 = 0;
const INIT_IMAGE: u32 = 1;
const LAST_IMAGE: u32 = 3;
const LONG_GROUP: u8 = 1;
const TX_ANTENNA: u8 = 0x98;
const PHY_CONFIGURATION: u8 = 0x6a;
const ANTENNA_MASK: u8 = 0xf;
const TX_SHIFT: u32 = 16;
const RX_SHIFT: u32 = 20;
const SHARED_CLOCK: u32 = 1 << 31;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    FirmwareApi(u32),
    CommandApi {
        opcode: u8,
        found: Option<(u8, u8)>,
    },
    DuplicateCommandApi(u8),
    MissingPhySku,
    CalibrationLength(usize),
    CalibrationImage(u32),
    MissingInitCalibration,
    DuplicateInitCalibration,
    MissingRegularCalibration,
    DuplicateRegularCalibration,
    Antennas {
        firmware_tx: u8,
        firmware_rx: u8,
        nvm_tx: u8,
        nvm_rx: u8,
    },
}

/// The firmware PHY word and default calibration controls for one image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    phy: u32,
    flow: u32,
    event: u32,
}

impl Config {
    /// Select the defaults for the INIT image.
    pub fn from_firmware(firmware: &Firmware<'_>) -> Result<Self, Error> {
        Self::for_image(firmware, Image::Init)
    }

    /// Select the defaults for the REGULAR image.
    pub fn from_regular_firmware(firmware: &Firmware<'_>) -> Result<Self, Error> {
        Self::for_image(firmware, Image::Regular)
    }

    fn for_image(firmware: &Firmware<'_>, image: Image) -> Result<Self, Error> {
        let api = firmware.api_version();
        if api != API {
            return Err(Error::FirmwareApi(api));
        }
        require_abi(firmware, TX_ANTENNA, 1)?;
        require_abi(firmware, PHY_CONFIGURATION, 2)?;
        let phy = firmware.phy_sku().ok_or(Error::MissingPhySku)?;
        let (image, missing, duplicate) = match image {
            Image::Init => (
                INIT_IMAGE,
                Error::MissingInitCalibration,
                Error::DuplicateInitCalibration,
            ),
            Image::Regular => (
                REGULAR_IMAGE,
                Error::MissingRegularCalibration,
                Error::DuplicateRegularCalibration,
            ),
        };
        let mut calibration = None;
        for tlv in firmware
            .tlvs()
            .iter()
            .filter(|tlv| tlv.kind == DEFAULT_CALIBRATION)
        {
            if tlv.data.len() != 12 {
                return Err(Error::CalibrationLength(tlv.data.len()));
            }
            let tlv_image = le32(tlv.data, 0);
            if tlv_image > LAST_IMAGE {
                return Err(Error::CalibrationImage(tlv_image));
            }
            if tlv_image == image
                && calibration
                    .replace((le32(tlv.data, 4), le32(tlv.data, 8)))
                    .is_some()
            {
                return Err(duplicate);
            }
        }
        let (flow, event) = calibration.ok_or(missing)?;
        Ok(Self { phy, flow, event })
    }

    /// Use only antennas present in both the image and this adapter's NVM
    /// masks, optionally narrowed by its platform policy. The caller supplies
    /// shared-clock only for an independently identified board profile.
    pub fn commands(self, tx_mask: u8, rx_mask: u8, shared_clock: bool) -> Result<Commands, Error> {
        let fw_tx = (self.phy >> TX_SHIFT) as u8 & ANTENNA_MASK;
        let fw_rx = (self.phy >> RX_SHIFT) as u8 & ANTENNA_MASK;
        let tx = fw_tx & tx_mask;
        let rx = fw_rx & rx_mask;
        if tx_mask & !ANTENNA_MASK != 0 || rx_mask & !ANTENNA_MASK != 0 || tx == 0 || rx == 0 {
            return Err(Error::Antennas {
                firmware_tx: fw_tx,
                firmware_rx: fw_rx,
                nvm_tx: tx_mask,
                nvm_rx: rx_mask,
            });
        }
        let chain_mask =
            (u32::from(ANTENNA_MASK) << TX_SHIFT) | (u32::from(ANTENNA_MASK) << RX_SHIFT);
        let mut word = (self.phy & !(chain_mask | SHARED_CLOCK))
            | (u32::from(tx) << TX_SHIFT)
            | (u32::from(rx) << RX_SHIFT);
        if shared_clock {
            word |= SHARED_CLOCK;
        }
        let mut configuration = [0; 12];
        configuration[..4].copy_from_slice(&word.to_le_bytes());
        self.set_calibration(&mut configuration);
        Ok(Commands {
            antenna: u32::from(tx).to_le_bytes(),
            phy: configuration,
        })
    }

    /// Use the INIT-validated PHY word with the REGULAR calibration controls.
    pub(super) fn regular_request(self, init: Commands) -> Request {
        let mut configuration = init.phy;
        self.set_calibration(&mut configuration);
        Request(Payload::Phy(configuration))
    }

    fn set_calibration(self, configuration: &mut [u8; 12]) {
        configuration[4..8].copy_from_slice(&self.flow.to_le_bytes());
        configuration[8..].copy_from_slice(&self.event.to_le_bytes());
    }
}

fn require_abi(firmware: &Firmware<'_>, opcode: u8, version: u8) -> Result<(), Error> {
    let mut entries = firmware
        .command_versions()
        .unwrap_or_default()
        .iter()
        .filter(|entry| entry.group == LONG_GROUP && entry.command == opcode);
    let found = entries
        .next()
        .map(|entry| (entry.command_version, entry.notification_version));
    if entries.next().is_some() {
        return Err(Error::DuplicateCommandApi(opcode));
    }
    if found == Some((version, 0)) {
        Ok(())
    } else {
        Err(Error::CommandApi { opcode, found })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Commands {
    antenna: [u8; 4],
    phy: [u8; 12],
}

impl Commands {
    pub fn antenna(self) -> Request {
        Request(Payload::Antenna(self.antenna))
    }

    pub fn phy(self) -> Request {
        Request(Payload::Phy(self.phy))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Payload {
    Antenna([u8; 4]),
    Phy([u8; 12]),
}

/// A typed PHY request produced from the verified input masks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request(Payload);

impl Request {
    /// Linux MVM supplies a bare ID. The common header encoder rewrites its
    /// group to ALWAYS_LONG and retains the donor's zero wire version.
    pub fn id(self) -> Id {
        let opcode = match self.0 {
            Payload::Antenna(_) => TX_ANTENNA,
            Payload::Phy(_) => PHY_CONFIGURATION,
        };
        Id {
            group: 0,
            opcode,
            version: 0,
        }
    }

    pub fn payload(&self) -> &[u8] {
        match &self.0 {
            Payload::Antenna(bytes) => bytes,
            Payload::Phy(bytes) => bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    fn tlv(image: &mut Vec<u8>, kind: u32, data: &[u8]) {
        image.extend(kind.to_le_bytes());
        image.extend((data.len() as u32).to_le_bytes());
        image.extend(data);
        image.resize(image.len().next_multiple_of(4), 0);
    }

    fn fixture(phy_version: u8) -> Vec<u8> {
        let mut image = vec![0; 88];
        image[4..8].copy_from_slice(&0x0a4c_5749u32.to_le_bytes());
        image[72..76].copy_from_slice(&(API << 8).to_le_bytes());
        tlv(&mut image, 23, &0x0033_0018u32.to_le_bytes());
        tlv(
            &mut image,
            48,
            &[
                TX_ANTENNA,
                LONG_GROUP,
                1,
                0,
                PHY_CONFIGURATION,
                LONG_GROUP,
                phy_version,
                0,
            ],
        );
        image
    }

    #[test]
    fn init_payload_keeps_flow_then_event_and_narrows_both_antennas() {
        let mut image = fixture(2);
        let regular = [REGULAR_IMAGE, 0x0f06_15db, 0x0f02_958b].map(u32::to_le_bytes);
        tlv(&mut image, DEFAULT_CALIBRATION, regular.as_flattened());
        let init = [INIT_IMAGE, 0x1f8e_51db, 1].map(u32::to_le_bytes);
        tlv(&mut image, DEFAULT_CALIBRATION, init.as_flattened());
        let config = Config::from_firmware(&Firmware::parse(&image).unwrap()).unwrap();
        let commands = config.commands(1, 2, true).unwrap();
        assert_eq!(
            commands.antenna().id().wire(),
            Id {
                group: 1,
                opcode: 0x98,
                version: 0
            }
        );
        assert_eq!(commands.antenna().payload(), [1, 0, 0, 0]);
        assert_eq!(
            commands.phy().id().wire(),
            Id {
                group: 1,
                opcode: 0x6a,
                version: 0
            }
        );
        let payload = commands.phy();
        assert_eq!(le32(payload.payload(), 0), 0x8021_0018);
        assert_eq!(le32(payload.payload(), 4), 0x1f8e_51db);
        assert_eq!(le32(payload.payload(), 8), 1);
        let discrete = config.commands(1, 2, false).unwrap().phy();
        assert_eq!(le32(discrete.payload(), 0), 0x0021_0018);
        assert!(matches!(
            config.commands(4, 2, true),
            Err(Error::Antennas { .. })
        ));
    }

    #[test]
    fn missing_duplicate_or_wrong_abi_calibration_is_refused() {
        let mut image = fixture(2);
        assert_eq!(
            Config::from_firmware(&Firmware::parse(&image).unwrap()),
            Err(Error::MissingInitCalibration)
        );
        let init = [INIT_IMAGE, 1, 1].map(u32::to_le_bytes);
        tlv(&mut image, DEFAULT_CALIBRATION, init.as_flattened());
        tlv(&mut image, DEFAULT_CALIBRATION, init.as_flattened());
        assert_eq!(
            Config::from_firmware(&Firmware::parse(&image).unwrap()),
            Err(Error::DuplicateInitCalibration)
        );
        assert_eq!(
            Config::from_firmware(&Firmware::parse(&fixture(3)).unwrap()),
            Err(Error::CommandApi {
                opcode: PHY_CONFIGURATION,
                found: Some((3, 0))
            })
        );
    }

    #[test]
    fn regular_requires_one_type_zero_calibration_and_ignores_init_defaults() {
        let mut image = fixture(2);
        let init = [INIT_IMAGE, 1, 2].map(u32::to_le_bytes);
        tlv(&mut image, DEFAULT_CALIBRATION, init.as_flattened());
        assert_eq!(
            Config::from_regular_firmware(&Firmware::parse(&image).unwrap()),
            Err(Error::MissingRegularCalibration)
        );

        let regular = [REGULAR_IMAGE, 0x0f06_15db, 0x0f02_958b].map(u32::to_le_bytes);
        tlv(&mut image, DEFAULT_CALIBRATION, regular.as_flattened());
        let firmware = Firmware::parse(&image).unwrap();
        let init = Config::from_firmware(&firmware)
            .unwrap()
            .commands(1, 2, true)
            .unwrap();
        let regular_request = Config::from_regular_firmware(&firmware)
            .unwrap()
            .regular_request(init);
        assert_eq!(regular_request.id().wire(), init.phy().id().wire());
        assert_eq!(&regular_request.payload()[..4], &init.phy().payload()[..4]);
        assert_eq!(
            regular_request.payload(),
            [0x8021_0018, 0x0f06_15db, 0x0f02_958b]
                .map(u32::to_le_bytes)
                .as_flattened()
        );

        tlv(&mut image, DEFAULT_CALIBRATION, regular.as_flattened());
        let firmware = Firmware::parse(&image).unwrap();
        assert_eq!(
            Config::from_regular_firmware(&firmware),
            Err(Error::DuplicateRegularCalibration)
        );
        assert!(Config::from_firmware(&firmware).is_ok());
    }

    #[test]
    #[ignore = "oracle: STELA_IWLWIFI_IMAGES"]
    fn pinned_api_46_images_encode_regular_phy_calibration() {
        extern crate std;
        use super::super::Sku;

        let directory = std::env::var("STELA_IWLWIFI_IMAGES").expect("firmware directory");
        for sku in [Sku::Cnvi9000, Sku::WirelessAc9260] {
            let bytes =
                std::fs::read(std::path::Path::new(&directory).join(sku.firmware_name())).unwrap();
            let firmware = Firmware::parse(&bytes).unwrap();
            let shared_clock = sku == Sku::Cnvi9000;
            let init = Config::from_firmware(&firmware)
                .unwrap()
                .commands(3, 3, shared_clock)
                .unwrap();
            let regular_request = Config::from_regular_firmware(&firmware)
                .unwrap()
                .regular_request(init);
            assert_eq!(init.antenna().payload(), &3u32.to_le_bytes());
            assert_eq!(
                regular_request.payload(),
                [
                    0x0033_0018 | u32::from(shared_clock) << 31,
                    0x0f06_15db,
                    0x0f02_958b,
                ]
                .map(u32::to_le_bytes)
                .as_flattened()
            );
        }
    }
}
