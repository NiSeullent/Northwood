// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2008-2014, 2018-2026 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2016-2017 Intel Deutschland GmbH
//! Bounded, borrowed Intel TLV firmware and self-load section planning.
//!
//! Adapted from iwl-drv.c, fw/file.h and pcie/ctxt-info.c at Linux
//! a804c5742bff224d4cb4d25dd383554d598be32b. Every TLV remains observable,
//! including metadata the loader does not consume. A parse is not an upload,
//! signature verification, ALIVE completion or permission to operate a radio.
use crate::bytes::le32;
use alloc::{collections::BTreeMap, vec::Vec};
use core::ops::Range;

pub mod legacy;

pub(super) const HEADER_BYTES: usize = 88;
pub(super) const TLV_HEADER_BYTES: usize = 8;
const MAGIC: u32 = 0x0a4c_5749;
const MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;
const MAX_TLVS: usize = 4096;
const MAX_SECTIONS: usize = 256;
const MAX_DRAM_ENTRIES: usize = 64;
const MAX_SECTION_BYTES: usize = 32 * 1024;
const CPU_SEPARATOR: u32 = 0xffff_cccc;
const PAGING_SEPARATOR: u32 = 0xaaaa_bbbb;
const API_NEW_VERSION: u32 = 20;

const SEC_RT: u32 = 19;
const SEC_INIT: u32 = 20;
const SEC_WOWLAN: u32 = 21;
const SECURE_RT: u32 = 24;
const SECURE_INIT: u32 = 25;
const SECURE_WOWLAN: u32 = 26;
const SEC_USNIFFER: u32 = 34;
const FLAGS: u32 = 18;
const PHY_SKU: u32 = 23;
const CPUS: u32 = 27;
const API: u32 = 29;
const CAPABILITIES: u32 = 30;
const PAGING: u32 = 32;
const FW_VERSION: u32 = 36;
const CMD_VERSIONS: u32 = 48;
const CMD_BIOS_TABLES: u32 = 73;
const IML: u32 = 52;

/// `IWL_DEVICE_FAMILY`, ordered as Linux compares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Family {
    Ax200,
    Ax210,
    Bz,
    Sc,
}

impl Family {
    /// The standalone device's image, for tests over a released firmware.
    #[cfg(test)]
    pub fn firmware_name(self) -> &'static str {
        match self {
            Self::Ax200 => "iwlwifi-cc-a0-77.ucode",
            Self::Ax210 => "iwlwifi-ty-a0-gf-a0-89.ucode",
            Self::Bz => "iwlwifi-gl-c0-fm-c0-c106.ucode",
            Self::Sc => "iwlwifi-sc-a0-fm-c0-c106.ucode",
        }
    }
    pub fn firmware_api(self) -> u32 {
        match self {
            Self::Ax200 => 77,
            Self::Ax210 => 89,
            Self::Bz | Self::Sc => 106,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    ImageSize,
    Header,
    Truncated { offset: usize },
    ResourceLimit,
    TlvLength { kind: u32, bytes: usize },
    Duplicate { kind: u32 },
    CpuCount(u32),
    SectionAddress,
    FirmwareApi { expected: u32, actual: u32 },
    Missing(&'static str),
    SectionLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageKind {
    Runtime,
    Init,
    Wowlan,
    Sniffer,
}

/// Exact raw metadata, with input coordinates and padding kept.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tlv<'a> {
    pub kind: u32,
    pub data: &'a [u8],
    pub padding: &'a [u8],
    pub range: Range<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Section<'a> {
    pub image: ImageKind,
    pub secure: bool,
    pub offset: u32,
    pub data: &'a [u8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandVersion {
    pub command: u8,
    pub group: u8,
    pub command_version: u8,
    pub notification_version: u8,
}

/// LMAC, UMAC and persistent paging payloads, in input order. Separator
/// sections never become DMA payloads. The transport must keep paging until
/// the device has stopped; an ALIVE notification alone does not release it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelfLoadPlan<'a> {
    firmware_api: u32,
    lmac: Vec<Section<'a>>,
    umac: Vec<Section<'a>>,
    paging: Vec<Section<'a>>,
}

#[derive(Debug)]
pub struct Firmware<'a> {
    bytes: &'a [u8],
    human_readable: &'a [u8; 64],
    version: u32,
    build: u32,
    ignored_header: u64,
    tlvs: Vec<Tlv<'a>>,
    sections: Vec<Section<'a>>,
    command_versions: Option<Vec<CommandVersion>>,
    flags: Option<u32>,
    phy_sku: Option<u32>,
    cpus: Option<u32>,
    paging_bytes: Option<u32>,
    version_components: Option<[u32; 3]>,
    iml: Option<&'a [u8]>,
    api_words: BTreeMap<u32, u32>,
    capability_words: BTreeMap<u32, u32>,
}

fn singleton<T>(slot: &mut Option<T>, value: T, kind: u32) -> Result<(), Error> {
    if slot.is_some() {
        return Err(Error::Duplicate { kind });
    }
    *slot = Some(value);
    Ok(())
}

impl<'a> SelfLoadPlan<'a> {
    pub fn firmware_api(&self) -> u32 {
        self.firmware_api
    }
    pub fn lmac(&self) -> &[Section<'a>] {
        &self.lmac
    }
    pub fn umac(&self) -> &[Section<'a>] {
        &self.umac
    }
    pub fn paging(&self) -> &[Section<'a>] {
        &self.paging
    }
}

impl<'a> Firmware<'a> {
    pub fn human_readable(&self) -> &[u8; 64] {
        self.human_readable
    }
    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn build(&self) -> u32 {
        self.build
    }
    pub fn ignored_header(&self) -> u64 {
        self.ignored_header
    }
    pub fn tlvs(&self) -> &[Tlv<'a>] {
        &self.tlvs
    }

    /// Exact borrowed initial microcode loader bytes (TLV52). Presence does
    /// not authenticate the loader or prove ROM execution.
    pub fn iml(&self) -> Option<&'a [u8]> {
        self.iml
    }
    pub fn sections(&self) -> &[Section<'a>] {
        &self.sections
    }
    pub fn flags(&self) -> Option<u32> {
        self.flags
    }
    pub fn phy_sku(&self) -> Option<u32> {
        self.phy_sku
    }
    pub fn cpus(&self) -> Option<u32> {
        self.cpus
    }
    pub fn paging_bytes(&self) -> Option<u32> {
        self.paging_bytes
    }
    pub fn version_components(&self) -> Option<[u32; 3]> {
        self.version_components
    }
    pub fn command_versions(&self) -> Option<&[CommandVersion]> {
        self.command_versions.as_deref()
    }

    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        if !(HEADER_BYTES..=MAX_IMAGE_BYTES).contains(&bytes.len()) {
            return Err(Error::ImageSize);
        }
        if le32(bytes, 0) != 0 || le32(bytes, 4) != MAGIC {
            return Err(Error::Header);
        }
        let mut image = Self {
            bytes,
            human_readable: bytes[8..72].try_into().expect("header checked"),
            version: le32(bytes, 72),
            build: le32(bytes, 76),
            ignored_header: u64::from_le_bytes(bytes[80..88].try_into().expect("header checked")),
            tlvs: Vec::new(),
            sections: Vec::new(),
            command_versions: None,
            flags: None,
            phy_sku: None,
            cpus: None,
            paging_bytes: None,
            version_components: None,
            iml: None,
            api_words: BTreeMap::new(),
            capability_words: BTreeMap::new(),
        };
        let mut at = HEADER_BYTES;
        while at < bytes.len() {
            if image.tlvs.len() == MAX_TLVS {
                return Err(Error::ResourceLimit);
            }
            let header = bytes
                .get(at..at + TLV_HEADER_BYTES)
                .ok_or(Error::Truncated { offset: at })?;
            let kind = le32(header, 0);
            let count = le32(header, 4) as usize;
            let aligned = count
                .checked_next_multiple_of(4)
                .ok_or(Error::Truncated { offset: at })?;
            let start = at + TLV_HEADER_BYTES;
            let end = start
                .checked_add(aligned)
                .filter(|end| *end <= bytes.len())
                .ok_or(Error::Truncated { offset: at })?;
            let data = &bytes[start..start + count];
            match kind {
                SEC_RT | SEC_INIT | SEC_WOWLAN | SECURE_RT | SECURE_INIT | SECURE_WOWLAN
                | SEC_USNIFFER => {
                    if data.len() < 4 {
                        return Err(Error::TlvLength {
                            kind,
                            bytes: data.len(),
                        });
                    }
                    if image.sections.len() == MAX_SECTIONS {
                        return Err(Error::ResourceLimit);
                    }
                    let offset = le32(data, 0);
                    let payload = &data[4..];
                    if !matches!(offset, CPU_SEPARATOR | PAGING_SEPARATOR)
                        && offset.checked_add(payload.len() as u32).is_none()
                    {
                        return Err(Error::SectionAddress);
                    }
                    image.sections.push(Section {
                        image: match kind {
                            SEC_RT | SECURE_RT => ImageKind::Runtime,
                            SEC_INIT | SECURE_INIT => ImageKind::Init,
                            SEC_WOWLAN | SECURE_WOWLAN => ImageKind::Wowlan,
                            _ => ImageKind::Sniffer,
                        },
                        secure: matches!(kind, SECURE_RT | SECURE_INIT | SECURE_WOWLAN),
                        offset,
                        data: payload,
                    });
                }
                FLAGS | PHY_SKU | CPUS | PAGING => {
                    if data.len() != 4 {
                        return Err(Error::TlvLength {
                            kind,
                            bytes: data.len(),
                        });
                    }
                    let value = le32(data, 0);
                    match kind {
                        FLAGS => singleton(&mut image.flags, value, kind)?,
                        PHY_SKU => singleton(&mut image.phy_sku, value, kind)?,
                        CPUS => {
                            if !(1..=2).contains(&value) {
                                return Err(Error::CpuCount(value));
                            }
                            singleton(&mut image.cpus, value, kind)?;
                        }
                        _ => {
                            if value > 1024 * 1024 || value & 4095 != 0 {
                                return Err(Error::TlvLength {
                                    kind,
                                    bytes: data.len(),
                                });
                            }
                            singleton(&mut image.paging_bytes, value, kind)?;
                        }
                    }
                }
                API | CAPABILITIES => {
                    if data.len() != 8 {
                        return Err(Error::TlvLength {
                            kind,
                            bytes: data.len(),
                        });
                    }
                    let words = if kind == API {
                        &mut image.api_words
                    } else {
                        &mut image.capability_words
                    };
                    *words.entry(le32(data, 0)).or_default() |= le32(data, 4);
                }
                FW_VERSION => {
                    if data.len() != 12 {
                        return Err(Error::TlvLength {
                            kind,
                            bytes: data.len(),
                        });
                    }
                    singleton(
                        &mut image.version_components,
                        [le32(data, 0), le32(data, 4), le32(data, 8)],
                        kind,
                    )?;
                }
                CMD_VERSIONS => {
                    if !data.len().is_multiple_of(4) {
                        return Err(Error::TlvLength {
                            kind,
                            bytes: data.len(),
                        });
                    }
                    let versions = data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|v| CommandVersion {
                            command: v[0],
                            group: v[1],
                            command_version: v[2],
                            notification_version: v[3],
                        })
                        .collect();
                    singleton(&mut image.command_versions, versions, kind)?;
                }
                CMD_BIOS_TABLES if !count.is_multiple_of(4) => {
                    return Err(Error::TlvLength { kind, bytes: count });
                }
                IML => {
                    if data.is_empty() {
                        return Err(Error::TlvLength { kind, bytes: 0 });
                    }
                    singleton(&mut image.iml, data, kind)?;
                }
                _ => {} // Below, never interpreted as a ready device.
            }
            image.tlvs.push(Tlv {
                kind,
                data,
                padding: &bytes[start + count..end],
                range: at..end,
            });
            at = end;
        }
        Ok(image)
    }

    pub fn supports_api(&self, bit: u32) -> bool {
        self.api_words
            .get(&(bit / 32))
            .is_some_and(|word| word & (1 << (bit % 32)) != 0)
    }

    pub fn supports_capability(&self, bit: u32) -> bool {
        self.capability_words
            .get(&(bit / 32))
            .is_some_and(|word| word & (1 << (bit % 32)) != 0)
    }

    pub fn api_version(&self) -> u32 {
        if self.supports_api(API_NEW_VERSION) {
            self.version
        } else {
            (self.version >> 8) & 0xff
        }
    }

    pub fn require_family(&self, family: Family) -> Result<(), Error> {
        let actual = self.api_version();
        if actual != family.firmware_api() {
            return Err(Error::FirmwareApi {
                expected: family.firmware_api(),
                actual,
            });
        }
        Ok(())
    }

    pub fn command_version(&self, group: u8, command: u8) -> Option<CommandVersion> {
        self.command_versions
            .as_deref()?
            .iter()
            .copied()
            .find(|v| v.group == group && v.command == command)
    }

    /// `iwl_fw_lookup_cmd_bios_supported_revision`: the newest BIOS table a
    /// command takes from UEFI or from ACPI, where the image says.
    pub fn bios_table_revision(&self, group: u8, command: u8, uefi: bool) -> Option<u8> {
        let table = self.tlvs.iter().find(|tlv| tlv.kind == CMD_BIOS_TABLES)?;
        let row = table
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .find(|row| row[0] == command && row[1] == group)?;
        Some(row[if uefi { 3 } else { 2 }])
    }

    /// Gen2 self-load layout only. AX210 has additional gen3 IML/context/PNVM
    /// requirements; this shared payload order does not satisfy that protocol.
    pub fn self_load_plan(&self) -> Result<SelfLoadPlan<'a>, Error> {
        if self.cpus != Some(2) {
            return Err(Error::Missing("dual CPU declaration"));
        }
        let mut plan = SelfLoadPlan {
            firmware_api: self.api_version(),
            lmac: Vec::new(),
            umac: Vec::new(),
            paging: Vec::new(),
        };
        let mut stage = 0;
        for section in self
            .sections
            .iter()
            .filter(|s| s.image == ImageKind::Runtime)
        {
            if section.offset == CPU_SEPARATOR {
                if stage != 0 || plan.lmac.is_empty() {
                    return Err(Error::SectionLayout);
                }
                stage = 1;
                continue;
            }
            if section.offset == PAGING_SEPARATOR {
                if stage != 1 || plan.umac.is_empty() {
                    return Err(Error::SectionLayout);
                }
                stage = 2;
                continue;
            }
            if section.data.is_empty() || section.data.len() > MAX_SECTION_BYTES {
                return Err(Error::SectionLayout);
            }
            let dest = match stage {
                0 => &mut plan.lmac,
                1 => &mut plan.umac,
                _ => &mut plan.paging,
            };
            if dest.len() == MAX_DRAM_ENTRIES {
                return Err(Error::ResourceLimit);
            }
            dest.push(*section);
        }
        if stage != 2 || plan.paging.len() < 2 {
            return Err(Error::SectionLayout);
        }
        // The first paging entry is the signed CSS image, separate from the
        // declared page data. A prefix ending at a section boundary must not
        // silently become a complete, smaller paging image.
        let paging = plan
            .paging
            .iter()
            .skip(1)
            .try_fold(0usize, |total, section| {
                total.checked_add(section.data.len())
            })
            .ok_or(Error::SectionLayout)?;
        if plan.paging[0].offset != 0 || self.paging_bytes.map(|n| n as usize) != Some(paging) {
            return Err(Error::SectionLayout);
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image() -> Vec<u8> {
        let mut bytes = vec![0; HEADER_BYTES];
        bytes[4..8].copy_from_slice(&MAGIC.to_le_bytes());
        bytes[72..76].copy_from_slice(&77u32.to_le_bytes());
        bytes
    }

    fn tlv(bytes: &mut Vec<u8>, kind: u32, data: &[u8]) {
        bytes.extend(kind.to_le_bytes());
        bytes.extend((data.len() as u32).to_le_bytes());
        bytes.extend(data);
        bytes.resize(bytes.len().next_multiple_of(4), 0);
    }
    fn word(bytes: &mut Vec<u8>, kind: u32, value: u32) {
        tlv(bytes, kind, &value.to_le_bytes());
    }
    fn section(bytes: &mut Vec<u8>, offset: u32, payload: &[u8]) {
        let mut data = offset.to_le_bytes().to_vec();
        data.extend(payload);
        tlv(bytes, SEC_RT, &data);
    }
    fn boot_image() -> Vec<u8> {
        let mut bytes = image();
        word(&mut bytes, CPUS, 2);
        let mut api = 0u32.to_le_bytes().to_vec();
        api.extend((1u32 << API_NEW_VERSION).to_le_bytes());
        tlv(&mut bytes, API, &api);
        section(&mut bytes, 0x0043_3000, &[0x11; 8]);
        section(&mut bytes, CPU_SEPARATOR, &[0; 4]);
        section(&mut bytes, 0x8043_3000, &[0x22; 8]);
        section(&mut bytes, PAGING_SEPARATOR, &[0; 4]);
        section(&mut bytes, 0, &[0x33; 8]);
        section(&mut bytes, 0x0100_0000, &[0x44; 4096]);
        word(&mut bytes, PAGING, 4096);
        bytes
    }

    #[test]
    fn iml_keeps_exact_borrowed_payload_and_refuses_empty_or_ambiguous_loader() {
        let mut bytes = image();
        assert!(Firmware::parse(&bytes).unwrap().iml().is_none());
        let at = bytes.len() + TLV_HEADER_BYTES;
        tlv(&mut bytes, IML, &[0xa5; 17]);
        let fw = Firmware::parse(&bytes).unwrap();
        assert_eq!(fw.iml().unwrap(), &[0xa5; 17]);
        assert_eq!(fw.iml().unwrap().as_ptr(), bytes[at..].as_ptr());
        assert_eq!(fw.tlvs().last().unwrap().data, fw.iml().unwrap());
        tlv(&mut bytes, IML, &[0x55; 4]);
        assert_eq!(
            Firmware::parse(&bytes).unwrap_err(),
            Error::Duplicate { kind: IML }
        );
        let mut bytes = image();
        tlv(&mut bytes, IML, &[]);
        assert_eq!(
            Firmware::parse(&bytes).unwrap_err(),
            Error::TlvLength {
                kind: IML,
                bytes: 0
            }
        );
    }

    #[test]
    fn borrowed_input_sections_and_unknown_metadata_keep_exact_coordinates() {
        let mut bytes = boot_image();
        let unknown_at = bytes.len();
        tlv(&mut bytes, 0x8765_4321, &[5, 6, 7]);
        *bytes.last_mut().unwrap() = 0xa5;
        let firmware = Firmware::parse(&bytes).unwrap();
        firmware.require_family(Family::Ax200).unwrap();
        assert_eq!(
            firmware.require_family(Family::Ax210),
            Err(Error::FirmwareApi {
                expected: 89,
                actual: 77
            })
        );
        let unknown = firmware.tlvs().last().unwrap();
        assert_eq!(
            (
                unknown.kind,
                unknown.data,
                unknown.padding,
                unknown.range.clone()
            ),
            (
                0x8765_4321,
                &[5, 6, 7][..],
                &[0xa5][..],
                unknown_at..bytes.len()
            )
        );
        assert_eq!(unknown.data.as_ptr(), bytes[unknown_at + 8..].as_ptr());
        let plan = firmware.self_load_plan().unwrap();
        assert_eq!(
            (plan.lmac().len(), plan.umac().len(), plan.paging().len()),
            (1, 1, 2)
        );
        assert_eq!(plan.lmac()[0].data, &[0x11; 8]);
        assert_eq!(plan.umac()[0].offset, 0x8043_3000);
        assert_eq!(plan.paging()[1].data, &[0x44; 4096]);
        assert_eq!(
            plan.lmac()[0].data.as_ptr(),
            firmware.sections()[0].data.as_ptr()
        );
    }

    #[test]
    fn truncated_lengths_padding_header_and_resource_budgets_refuse() {
        for size in 0..HEADER_BYTES {
            assert_eq!(
                Firmware::parse(&vec![0; size]).unwrap_err(),
                Error::ImageSize
            );
        }
        for offset in [0, 4] {
            let mut bytes = image();
            bytes[offset] ^= 1;
            assert_eq!(Firmware::parse(&bytes).unwrap_err(), Error::Header);
        }
        for count in [1usize, 2, 3, 5, 7, 9, 33, 257] {
            let mut bytes = image();
            tlv(&mut bytes, 0x9999, &vec![0x5a; count]);
            for length in HEADER_BYTES + 1..bytes.len() {
                assert!(
                    Firmware::parse(&bytes[..length]).is_err(),
                    "{count} {length}"
                );
            }
            Firmware::parse(&bytes).unwrap();
        }
        let mut bytes = image();
        bytes.extend(19u32.to_le_bytes());
        bytes.extend(u32::MAX.to_le_bytes());
        assert!(matches!(
            Firmware::parse(&bytes),
            Err(Error::Truncated { .. })
        ));
        let mut bytes = image();
        for _ in 0..=MAX_TLVS {
            tlv(&mut bytes, 0x9999, &[]);
        }
        assert_eq!(Firmware::parse(&bytes).unwrap_err(), Error::ResourceLimit);
        assert_eq!(
            Firmware::parse(&vec![0; MAX_IMAGE_BYTES + 1]).unwrap_err(),
            Error::ImageSize
        );
    }

    #[test]
    fn scalar_command_and_api_shapes_are_validated_without_invented_versions() {
        for kind in [
            FLAGS,
            PHY_SKU,
            CPUS,
            PAGING,
            API,
            CAPABILITIES,
            FW_VERSION,
            CMD_VERSIONS,
            SEC_RT,
        ] {
            let mut bytes = image();
            tlv(&mut bytes, kind, &[1, 2, 3]);
            assert!(matches!(
                Firmware::parse(&bytes),
                Err(Error::TlvLength { .. })
            ));
        }
        for cpu in [0, 3, u32::MAX] {
            let mut bytes = image();
            word(&mut bytes, CPUS, cpu);
            assert_eq!(Firmware::parse(&bytes).unwrap_err(), Error::CpuCount(cpu));
        }
        for kind in [FLAGS, PHY_SKU, CPUS, PAGING] {
            let value = match kind {
                CPUS => 2,
                PAGING => 4096,
                _ => 0,
            };
            let mut bytes = image();
            word(&mut bytes, kind, value);
            word(&mut bytes, kind, value);
            assert_eq!(
                Firmware::parse(&bytes).unwrap_err(),
                Error::Duplicate { kind }
            );
        }
        let mut bytes = image();
        tlv(&mut bytes, CMD_VERSIONS, &[0x10, 3, 2, 7, 0x11, 3, 1, 5]);
        let fw = Firmware::parse(&bytes).unwrap();
        assert_eq!(
            fw.command_version(3, 0x10),
            Some(CommandVersion {
                command: 0x10,
                group: 3,
                command_version: 2,
                notification_version: 7
            })
        );
        assert_eq!(fw.command_version(9, 0x10), None);
        tlv(&mut bytes, CMD_VERSIONS, &[]);
        assert_eq!(
            Firmware::parse(&bytes).unwrap_err(),
            Error::Duplicate { kind: CMD_VERSIONS }
        );
        let mut bytes = image();
        bytes[72..76].copy_from_slice(&0x1234_4d56u32.to_le_bytes());
        assert_eq!(Firmware::parse(&bytes).unwrap().api_version(), 77);
        tlv(&mut bytes, API, &[0, 0, 0, 0, 0, 0, 0x10, 0]);
        assert_eq!(Firmware::parse(&bytes).unwrap().api_version(), 0x1234_4d56);
    }

    #[test]
    fn malformed_section_order_sizes_and_incomplete_paging_cannot_form_a_plan() {
        let mut bytes = boot_image();
        bytes.truncate(bytes.len() - 12);
        assert_eq!(
            Firmware::parse(&bytes)
                .unwrap()
                .self_load_plan()
                .unwrap_err(),
            Error::SectionLayout
        );
        let mut bytes = image();
        word(&mut bytes, CPUS, 2);
        section(&mut bytes, CPU_SEPARATOR, &[]);
        assert_eq!(
            Firmware::parse(&bytes)
                .unwrap()
                .self_load_plan()
                .unwrap_err(),
            Error::SectionLayout
        );
        let mut bytes = boot_image();
        section(&mut bytes, CPU_SEPARATOR, &[0; 4]);
        assert_eq!(
            Firmware::parse(&bytes)
                .unwrap()
                .self_load_plan()
                .unwrap_err(),
            Error::SectionLayout
        );
        let mut bytes = image();
        section(&mut bytes, u32::MAX - 1, &[0; 4]);
        assert_eq!(Firmware::parse(&bytes).unwrap_err(), Error::SectionAddress);
        let mut bytes = image();
        word(&mut bytes, CPUS, 2);
        for _ in 0..=MAX_DRAM_ENTRIES {
            section(&mut bytes, 0, &[0]);
        }
        assert_eq!(
            Firmware::parse(&bytes)
                .unwrap()
                .self_load_plan()
                .unwrap_err(),
            Error::ResourceLimit
        );
    }

    #[test]
    #[ignore = "oracle: STELA_IWLWIFI_IMAGES"]
    fn real_ax200_ax210_images_keep_every_section_and_paging_byte() {
        extern crate std;
        let root = std::env::var("STELA_IWLWIFI_IMAGES").expect("actual pinned image directory");
        for (family, sections, lmac, umac, paging, size) in [
            (Family::Ax200, 50, 14, 15, 19, 585728),
            (Family::Ax210, 58, 15, 16, 25, 761856),
        ] {
            let bytes =
                std::fs::read(std::path::Path::new(&root).join(family.firmware_name())).unwrap();
            let firmware = Firmware::parse(&bytes).unwrap();
            firmware.require_family(family).unwrap();
            assert_eq!(firmware.sections().len(), sections);
            assert_eq!(firmware.paging_bytes(), Some(size));
            let plan = firmware.self_load_plan().unwrap();
            assert_eq!(
                (plan.lmac().len(), plan.umac().len(), plan.paging().len()),
                (lmac, umac, paging)
            );
            assert_eq!(
                plan.paging()
                    .iter()
                    .skip(1)
                    .map(|s| s.data.len())
                    .sum::<usize>(),
                size as usize
            );
            assert!(firmware.command_versions().is_some_and(|v| !v.is_empty()));
        }
    }
}
