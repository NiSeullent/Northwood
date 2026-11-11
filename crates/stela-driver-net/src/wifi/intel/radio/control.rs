// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2012-2014, 2018-2026 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2016-2017 Intel Deutschland GmbH
//! Pure source DQA/thermal/LTR/device-power command contracts. Preparation is
//! not initialization, thermal handoff, D3 transition or RF permission.
use super::{
    super::{command::Id, firmware::Firmware},
    lari::Observation,
};
use alloc::vec::Vec;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Dqa,
    TemperatureThresholds,
    Ctdp,
    Ltr,
    DevicePower,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    MissingApi(Kind),
    Version { kind: Kind, declared: u8 },
    HostThermalRequired,
    ThresholdCount(usize),
    CoolingState(u8),
    BiosClockNotObserved,
    BiosBudgetNotObserved,
    LtrNotObserved,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Control {
    kind: Kind,
    id: Id,
    bytes: Vec<u8>,
}
impl Control {
    fn make(
        fw: &Firmware<'_>,
        kind: Kind,
        id: Id,
        versions: &[u8],
        fixed_without_row: bool,
        bytes: Vec<u8>,
    ) -> Result<Self, Error> {
        let group = if id.group == 0 { 1 } else { id.group };
        if let Some(version) = fw.command_version(group, id.opcode) {
            if !versions.contains(&version.command_version) {
                return Err(Error::Version {
                    kind,
                    declared: version.command_version,
                });
            }
        } else if !fixed_without_row {
            return Err(Error::MissingApi(kind));
        }
        Ok(Self { kind, id, bytes })
    }
    pub fn kind(&self) -> Kind {
        self.kind
    }
    pub fn id(&self) -> Id {
        self.id
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn dqa(fw: &Firmware<'_>) -> Result<Option<Self>, Error> {
        if !fw.supports_capability(12) {
            return Ok(None);
        }
        // Source capability12 uses its fixed v1 queue0 structure even when
        // firmware has no DQA CMD_VERSIONS row. No guessed header ABI.
        Self::make(
            fw,
            Kind::Dqa,
            Id {
                group: 5,
                opcode: 0,
                version: 0,
            },
            &[1],
            true,
            0u32.to_le_bytes().to_vec(),
        )
        .map(Some)
    }
    pub fn temperature_thresholds(
        fw: &Firmware<'_>,
        thresholds_celsius: &[i16],
    ) -> Result<Self, Error> {
        if !fw.supports_capability(74) || !fw.supports_capability(75) {
            return Err(Error::HostThermalRequired);
        }
        if thresholds_celsius.len() > 8 {
            return Err(Error::ThresholdCount(thresholds_celsius.len()));
        }
        let mut thresholds = thresholds_celsius.to_vec();
        thresholds.sort_unstable();
        let mut bytes = (thresholds.len() as u32).to_le_bytes().to_vec();
        for threshold in thresholds {
            bytes.extend(threshold.to_le_bytes());
        }
        bytes.resize(20, 0);
        Self::make(
            fw,
            Kind::TemperatureThresholds,
            Id {
                group: 4,
                opcode: 4,
                version: 0,
            },
            &[1],
            false,
            bytes,
        )
    }
    pub fn ctdp(
        fw: &Firmware<'_>,
        radio_id: u32,
        bios: PowerBudget,
        operation: CtdpOperation,
        state: u8,
    ) -> Result<Option<Self>, Error> {
        if !fw.supports_capability(76) {
            return Ok(None);
        }
        if state >= 20 {
            return Err(Error::CoolingState(state));
        }
        let maximum = maximum_budget(radio_id, bios)?;
        let budget = (maximum - 150) * u32::from(19 - state) / 19 + 150;
        let mut bytes = (match operation {
            CtdpOperation::Start => 1u32,
            CtdpOperation::Stop => 2,
            CtdpOperation::Report => 4,
        })
        .to_le_bytes()
        .to_vec();
        bytes.extend(budget.to_le_bytes());
        bytes.extend([0; 4]);
        Self::make(
            fw,
            Kind::Ctdp,
            Id {
                group: 4,
                opcode: 3,
                version: 0,
            },
            &[1],
            false,
            bytes,
        )
        .map(Some)
    }
    pub fn ltr(fw: &Firmware<'_>, ltr: Ltr) -> Result<Option<Self>, Error> {
        if fw.supports_capability(50) {
            return Ok(None);
        }
        match ltr {
            Ltr::NotObserved => return Err(Error::LtrNotObserved),
            Ltr::Disabled => return Ok(None),
            Ltr::Enabled => (),
        }
        let mut bytes = alloc::vec![0;32];
        bytes[..4].copy_from_slice(&1u32.to_le_bytes());
        Self::make(
            fw,
            Kind::Ltr,
            Id {
                group: 0,
                opcode: 0xee,
                version: 0,
            },
            &[3],
            false,
            bytes,
        )
        .map(Some)
    }
    pub fn device_power(
        fw: &Firmware<'_>,
        mode: PowerMode,
        external_clock: Observation,
        in_d3: bool,
    ) -> Result<Self, Error> {
        let clock = match external_clock {
            Observation::NotObserved => return Err(Error::BiosClockNotObserved),
            Observation::Unavailable => false,
            Observation::Observed(value) => value.value != 0,
        };
        let version = fw
            .command_version(1, 0x77)
            .ok_or(Error::MissingApi(Kind::DevicePower))?
            .command_version;
        let mut flags = if mode == PowerMode::SavePower {
            1u16
        } else {
            0
        };
        if clock {
            flags |= 1 << 12;
        }
        if version >= 7 && in_d3 {
            flags |= 1 << 7;
        }
        let mut bytes = flags.to_le_bytes().to_vec();
        bytes.extend([0; 2]);
        Self::make(
            fw,
            Kind::DevicePower,
            Id {
                group: 0,
                opcode: 0x77,
                version: 0,
            },
            &[6, 7],
            false,
            bytes,
        )
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerMode {
    ContinuouslyAwake,
    SavePower,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ltr {
    NotObserved,
    Disabled,
    Enabled,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerBudget {
    NotObserved,
    Unavailable,
    Observed(u64),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CtdpOperation {
    Start,
    Stop,
    Report,
}
/// Total device consumption budget in mW, distinct from RF transmit dBm/EIRP.
/// Unavailable is checked BIOS-source exhaustion, not missing capture/AML.
pub fn maximum_budget(radio_id: u32, bios: PowerBudget) -> Result<u32, Error> {
    let default = match (radio_id >> 12) & 0xfff {
        0x105 | 0x108 => 2000,
        0x10a | 0x10c => 2400,
        0x10d => {
            if radio_id & (1 << 28) != 0 {
                5200
            } else {
                2880
            }
        }
        0x112 => 3450,
        _ => 5550,
    };
    let value = match bios {
        PowerBudget::NotObserved => return Err(Error::BiosBudgetNotObserved),
        PowerBudget::Unavailable => return Ok(default),
        PowerBudget::Observed(value) => value,
    };
    Ok(
        if value != 0xffff && value != 0xffffffff && (150..=u64::from(default)).contains(&value) {
            value as u32
        } else {
            default
        },
    )
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::wifi::intel::{
        command::Command,
        firmware::Family,
        radio::lari::{Source, Value},
    };
    fn firmware(rows: &[[u8; 4]], caps: &[u32]) -> Vec<u8> {
        let mut b = alloc::vec![0; 88];
        b[4..8].copy_from_slice(&0x0a4c5749u32.to_le_bytes());
        let mut tlv = |kind: u32, bytes: &[u8]| {
            b.extend(kind.to_le_bytes());
            b.extend((bytes.len() as u32).to_le_bytes());
            b.extend(bytes);
            b.resize(b.len().next_multiple_of(4), 0);
        };
        if !rows.is_empty() {
            tlv(48, &rows.iter().flatten().copied().collect::<Vec<_>>());
        }
        for word in 0..5 {
            let flags = caps
                .iter()
                .filter(|cap| **cap / 32 == word)
                .fold(0u32, |v, cap| v | (1 << (cap % 32)));
            if flags != 0 {
                let mut bytes = word.to_le_bytes().to_vec();
                bytes.extend(flags.to_le_bytes());
                tlv(30, &bytes);
            }
        }
        b
    }
    fn full(version: u8) -> Vec<u8> {
        firmware(
            &[
                [0x77, 1, version, 0],
                [0xee, 1, 3, 0],
                [4, 4, 1, 0],
                [3, 4, 1, 0],
            ],
            &[12, 74, 75, 76],
        )
    }
    fn clock(value: u32) -> Observation {
        Observation::Observed(Value {
            value,
            source: Source::Uefi,
            revision: 0,
        })
    }
    #[test]
    fn conditional_inputs_unknown_abis_and_host_thermal_are_explicit() {
        let b = firmware(&[], &[]);
        let fw = Firmware::parse(&b).unwrap();
        assert_eq!(Control::dqa(&fw), Ok(None));
        assert_eq!(
            Control::ctdp(&fw, 0, PowerBudget::NotObserved, CtdpOperation::Start, 255),
            Ok(None)
        );
        assert_eq!(Control::ltr(&fw, Ltr::Disabled), Ok(None));
        assert_eq!(
            Control::ltr(&fw, Ltr::NotObserved),
            Err(Error::LtrNotObserved)
        );
        assert_eq!(
            Control::ltr(&fw, Ltr::Enabled),
            Err(Error::MissingApi(Kind::Ltr))
        );
        assert_eq!(
            Control::temperature_thresholds(&fw, &[]),
            Err(Error::HostThermalRequired)
        );
        for caps in [&[74][..], &[75][..]] {
            let b = firmware(&[[4, 4, 1, 0]], caps);
            assert_eq!(
                Control::temperature_thresholds(&Firmware::parse(&b).unwrap(), &[]),
                Err(Error::HostThermalRequired)
            );
        }
        let b = firmware(&[], &[50]);
        assert_eq!(
            Control::ltr(&Firmware::parse(&b).unwrap(), Ltr::NotObserved),
            Ok(None)
        );
        let b = full(7);
        let fw = Firmware::parse(&b).unwrap();
        assert_eq!(
            Control::temperature_thresholds(&fw, &[0; 9]),
            Err(Error::ThresholdCount(9))
        );
        assert_eq!(
            Control::temperature_thresholds(&fw, &[50, -20, 50])
                .unwrap()
                .bytes(),
            &[
                3, 0, 0, 0, 236, 255, 50, 0, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );
        assert_eq!(
            Control::ctdp(&fw, 0, PowerBudget::Unavailable, CtdpOperation::Start, 20),
            Err(Error::CoolingState(20))
        );
        assert_eq!(
            maximum_budget(0, PowerBudget::NotObserved),
            Err(Error::BiosBudgetNotObserved)
        );
        assert_eq!(
            Control::device_power(&fw, PowerMode::SavePower, Observation::NotObserved, false),
            Err(Error::BiosClockNotObserved)
        );
        for (row, cap, kind) in [
            ([0, 5, 99, 0], 12, Kind::Dqa),
            ([4, 4, 99, 0], 74, Kind::TemperatureThresholds),
            ([3, 4, 99, 0], 76, Kind::Ctdp),
            ([0xee, 1, 99, 0], 0, Kind::Ltr),
            ([0x77, 1, 99, 0], 0, Kind::DevicePower),
        ] {
            let b = firmware(&[row], &[cap, 75]);
            let fw = Firmware::parse(&b).unwrap();
            let result = match kind {
                Kind::Dqa => Control::dqa(&fw).map(|v| v.unwrap()),
                Kind::TemperatureThresholds => Control::temperature_thresholds(&fw, &[]),
                Kind::Ctdp => {
                    Control::ctdp(&fw, 0, PowerBudget::Unavailable, CtdpOperation::Start, 0)
                        .map(|v| v.unwrap())
                }
                Kind::Ltr => Control::ltr(&fw, Ltr::Enabled).map(|v| v.unwrap()),
                Kind::DevicePower => Control::device_power(
                    &fw,
                    PowerMode::SavePower,
                    Observation::Unavailable,
                    false,
                ),
            };
            assert_eq!(result, Err(Error::Version { kind, declared: 99 }));
        }
    }
    #[test]
    #[ignore = "oracle: STELA_IWLWIFI_IMAGES"]
    fn pinned_api_46_images_encode_queue_zero_dqa_without_a_command_version_row() {
        use crate::wifi::intel::firmware::legacy::Sku;

        let root = std::env::var("STELA_IWLWIFI_IMAGES").expect("firmware directory");
        for sku in [Sku::Cnvi9000, Sku::WirelessAc9260] {
            let bytes =
                std::fs::read(std::path::Path::new(&root).join(sku.firmware_name())).unwrap();
            let firmware = Firmware::parse(&bytes).unwrap();
            assert_eq!(firmware.api_version(), 46);
            assert!(firmware.supports_capability(12));
            assert!(firmware.command_version(5, 0).is_none());

            let request = Control::dqa(&firmware).unwrap().unwrap();
            assert_eq!(request.kind(), Kind::Dqa);
            assert_eq!(
                request.id(),
                Id {
                    group: 5,
                    opcode: 0,
                    version: 0,
                }
            );
            assert_eq!(request.bytes(), &0u32.to_le_bytes());
            let command = Command::encode(request.id(), 0, 11, request.bytes()).unwrap();
            assert_eq!(command.sequence(), 11);
            assert_eq!(command.bytes(), &[0, 5, 11, 0, 4, 0, 0, 0, 0, 0, 0, 0]);
            assert!(command.remainder().is_empty());
        }
    }

    #[test]
    #[ignore = "oracle: STELA_IWLWIFI_IMAGES"]
    fn images_preserve_capability_skips_and_real_control_schemas() {
        let root = std::env::var("STELA_IWLWIFI_IMAGES").unwrap();
        for family in [Family::Ax200, Family::Ax210] {
            let b =
                std::fs::read(std::path::Path::new(&root).join(family.firmware_name())).unwrap();
            let fw = Firmware::parse(&b).unwrap();
            assert!(fw.command_version(5, 0).is_none());
            assert_eq!(Control::dqa(&fw).unwrap(), None);
            assert!(Control::temperature_thresholds(&fw, &[]).is_ok());
            assert_eq!(
                Control::ctdp(&fw, 0, PowerBudget::NotObserved, CtdpOperation::Start, 0).unwrap(),
                None
            );
            assert!(
                Control::device_power(&fw, PowerMode::SavePower, Observation::Unavailable, false)
                    .is_ok()
            );
            let result = Control::ltr(&fw, Ltr::Enabled).unwrap();
            assert_eq!(result.is_none(), fw.supports_capability(50));
            assert_eq!(
                maximum_budget(
                    if family == Family::Ax200 {
                        0x10a000
                    } else {
                        0x10d000
                    },
                    PowerBudget::Unavailable
                )
                .unwrap(),
                if family == Family::Ax200 { 2400 } else { 2880 }
            );
        }
    }
    #[test]
    #[ignore = "oracle: STELA_IWLWIFI_CONTROL_ORACLE"]
    fn unchanged_linux_control_helpers_match_every_complete_wire_command() {
        use sha2::{Digest, Sha256};
        let bytes = std::fs::read(std::env::var("STELA_IWLWIFI_CONTROL_ORACLE").unwrap()).unwrap();
        assert_eq!(
            std::format!("{:x}", Sha256::digest(&bytes)),
            "bfb687c04f8b6c9d76630e18c756d2c869da4716d8ae23bc2dc81ed007894a62"
        );
        let mut counts = [0; 6];
        for line in core::str::from_utf8(&bytes).unwrap().lines() {
            let f: Vec<_> = line.split_whitespace().collect();
            let u = |i: usize| f[i].parse::<u32>().unwrap();
            let b = full(if f[0] == "power" { u(1) as u8 } else { 7 });
            let fw = Firmware::parse(&b).unwrap();
            let (command, at) = match f[0] {
                "dqa" => {
                    counts[0] += 1;
                    (Control::dqa(&fw).unwrap().unwrap(), 1)
                }
                "ltr" => {
                    counts[1] += 1;
                    let command = Control::ltr(
                        &fw,
                        if u(1) == 0 {
                            Ltr::Disabled
                        } else {
                            Ltr::Enabled
                        },
                    )
                    .unwrap();
                    assert_eq!(u(2), u32::from(command.is_some()));
                    if command.is_none() {
                        continue;
                    }
                    (command.unwrap(), 3)
                }
                "power" => {
                    counts[2] += 1;
                    (
                        Control::device_power(
                            &fw,
                            if u(2) != 0 || u(3) != 0 {
                                PowerMode::ContinuouslyAwake
                            } else {
                                PowerMode::SavePower
                            },
                            clock(u(4)),
                            u(5) != 0,
                        )
                        .unwrap(),
                        6,
                    )
                }
                "thresholds" => {
                    counts[3] += 1;
                    let thresholds: Vec<_> = (0..u(1))
                        .map(|i| ((u(2) * 13 + i * 29) % 171) as i16 - 40)
                        .collect();
                    (
                        Control::temperature_thresholds(&fw, &thresholds).unwrap(),
                        3,
                    )
                }
                "budget" => {
                    counts[4] += 1;
                    assert_eq!(
                        maximum_budget(u(1), PowerBudget::Observed(f[2].parse().unwrap())).unwrap(),
                        u(3)
                    );
                    continue;
                }
                "ctdp" => {
                    counts[5] += 1;
                    let operation = match u(3) {
                        1 => CtdpOperation::Start,
                        2 => CtdpOperation::Stop,
                        4 => CtdpOperation::Report,
                        _ => panic!("operation"),
                    };
                    assert_eq!(
                        u(5),
                        if operation == CtdpOperation::Report {
                            33
                        } else {
                            0
                        }
                    );
                    (
                        Control::ctdp(
                            &fw,
                            u(1),
                            PowerBudget::Observed(f[2].parse().unwrap()),
                            operation,
                            u(4) as u8,
                        )
                        .unwrap()
                        .unwrap(),
                        6,
                    )
                }
                _ => panic!("unknown oracle row"),
            };
            assert_eq!(
                u32::from(command.id.group) * 256 + u32::from(command.id.opcode),
                u(at),
                "{line}"
            );
            let hex: std::string::String = command
                .bytes
                .iter()
                .map(|v| std::format!("{v:02x}"))
                .collect();
            assert_eq!(hex, f[at + 1], "{line}");
            assert_eq!(
                Command::encode(command.id, 0, 0, command.bytes())
                    .unwrap()
                    .bytes()[7],
                0
            );
        }
        assert_eq!(counts, [1, 2, 64, 144, 128, 7680]);
    }
}
