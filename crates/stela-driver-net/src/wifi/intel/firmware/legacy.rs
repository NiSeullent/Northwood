// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2005-2015, 2018-2026 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2015-2017 Intel Deutschland GmbH
//! Firmware selection and section planning for the Intel 9000-series FH loader.
//!
//! The PCI table and image layouts follow Linux iwlwifi at
//! 347246472df854673f5f743e6895ab07302fbbb1. The PCI ID selects a candidate
//! firmware file. Attaching a device also requires MAC/RF identity checks and
//! a gen1 transport.

use super::{CPU_SEPARATOR, Error, Firmware, ImageKind, PAGING_SEPARATOR, Section};
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use stela_crypto::{Sha256, digest::Digest};

pub mod adapter;
pub mod alive;
#[path = "legacy/aux_station.rs"]
pub mod aux;
pub mod hcmd;
pub mod identity;
pub mod nvm;
pub mod paging;
pub mod phy;
pub mod phy_db;
pub mod rfh;
pub mod rx;
pub mod service;

const API: u32 = 46;
// FH_UCODE_LOAD_STATUS has one 16-bit progress field per CPU.
const MAX_CPU_SECTIONS: usize = 16;
const FH_MAX_TRANSFER: usize = 0x20000;
const FH_DMA_LIMIT: u64 = 1 << 36;
const EXTENDED_START: u32 = 0x40000;
const EXTENDED_END: u32 = 0x58000;

/// These exact API 46 images omit the LARI command version. Their pins match
/// the Intel Wi-Fi firmware provisioning recipes.
pub(in crate::wifi::intel) fn has_pinned_lari_v1(firmware: &Firmware<'_>) -> bool {
    const CNVI_9000: [u8; 32] = [
        0x9a, 0x0c, 0x5d, 0x93, 0x86, 0x46, 0x94, 0x2f, 0x97, 0x98, 0xe6, 0x21, 0x25, 0x79, 0xc2,
        0x57, 0x71, 0xe4, 0xbb, 0x44, 0x30, 0x1c, 0xd4, 0x09, 0xb8, 0x2a, 0xbe, 0x36, 0x1c, 0xa2,
        0xc3, 0xe0,
    ];
    const WIRELESS_AC_9260: [u8; 32] = [
        0x05, 0xf4, 0xc1, 0x25, 0x14, 0xfd, 0xdf, 0xd5, 0x6f, 0xea, 0x66, 0x0b, 0x3d, 0x2e, 0xb6,
        0xb6, 0xe6, 0xf4, 0xa8, 0x7a, 0x32, 0x4d, 0x5d, 0x40, 0x46, 0x40, 0x50, 0x31, 0x2d, 0xdc,
        0xa9, 0x11,
    ];
    firmware.api_version() == API
        && matches!(
            <[u8; 32]>::from(Sha256::digest(firmware.bytes)),
            CNVI_9000 | WIRELESS_AC_9260
        )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sku {
    WirelessAc9260,
    Cnvi9000,
}

impl Sku {
    pub const fn from_pci(vendor: u16, device: u16) -> Option<Self> {
        match (vendor, device) {
            (0x8086, 0x2526 | 0x271b | 0x271c) => Some(Self::WirelessAc9260),
            (0x8086, 0x30dc | 0x31dc | 0x9df0 | 0xa370) => Some(Self::Cnvi9000),
            _ => None,
        }
    }

    pub const fn firmware_name(self) -> &'static str {
        match self {
            Self::WirelessAc9260 => "iwlwifi-9260-th-b0-jf-b0-46.ucode",
            Self::Cnvi9000 => "iwlwifi-9000-pu-b0-jf-b0-46.ucode",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Upload<'a> {
    cpu1: Vec<Section<'a>>,
    cpu2: Vec<Section<'a>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cpu {
    First,
    Second,
}

/// A borrowed FH service-channel DMA transfer and its SRAM addressing mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transfer<'a> {
    destination: u32,
    data: &'a [u8],
}

impl<'a> Transfer<'a> {
    pub fn data(self) -> &'a [u8] {
        self.data
    }

    pub fn extended_sram(self) -> bool {
        (EXTENDED_START..EXTENDED_END).contains(&self.destination)
    }

    /// Ordered writes to the FH service channel's CSR registers. The caller
    /// supplies an IOVA for the full staging buffer and waits for the FH TX
    /// completion interrupt before submitting another transfer.
    pub fn register_writes(self, iova: u64) -> Option<[(u32, u32); 6]> {
        if iova.checked_add(self.data.len() as u64)? > FH_DMA_LIMIT {
            return None;
        }
        const TX_CONFIG: u32 = 0x1e20;
        const SRAM_ADDRESS: u32 = 0x19c8;
        const TFD_ADDRESS_LOW: u32 = 0x1948;
        const TFD_ADDRESS_HIGH_SIZE: u32 = 0x194c;
        const TX_BUFFER_STATUS: u32 = 0x1e28;
        const TX_BUFFER_READY: u32 = (1 << 20) | (1 << 12) | 3;
        const ENABLE_AND_INTERRUPT: u32 = 0x8010_0000;
        Some([
            (TX_CONFIG, 0),
            (SRAM_ADDRESS, self.destination),
            (TFD_ADDRESS_LOW, iova as u32),
            (
                TFD_ADDRESS_HIGH_SIZE,
                ((iova >> 32) as u32) << 28 | self.data.len() as u32,
            ),
            (TX_BUFFER_STATUS, TX_BUFFER_READY),
            (TX_CONFIG, ENABLE_AND_INTERRUPT),
        ])
    }
}

impl<'a> Upload<'a> {
    pub fn sections(&self, cpu: Cpu) -> &[Section<'a>] {
        match cpu {
            Cpu::First => &self.cpu1,
            Cpu::Second => &self.cpu2,
        }
    }

    /// Split a resident section by staging capacity and SRAM mode. Paging
    /// payloads are deliberately absent from this API.
    pub fn transfers(
        &self,
        cpu: Cpu,
        section: usize,
        staging_capacity: NonZeroUsize,
    ) -> Option<impl Iterator<Item = Transfer<'a>>> {
        let section = self.sections(cpu).get(section)?;
        let mut destination = section.offset;
        let mut remaining = section.data;
        let max_length = staging_capacity.get().min(FH_MAX_TRANSFER);
        Some(core::iter::from_fn(move || {
            if remaining.is_empty() {
                return None;
            }
            let boundary = if destination < EXTENDED_START {
                EXTENDED_START
            } else if destination < EXTENDED_END {
                EXTENDED_END
            } else {
                u32::MAX
            };
            let length = remaining
                .len()
                .min(max_length)
                .min((boundary - destination) as usize);
            let (data, rest) = remaining.split_at(length);
            let transfer = Transfer { destination, data };
            remaining = rest;
            destination += length as u32;
            Some(transfer)
        }))
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct LoadPlan<'a> {
    pub init: Upload<'a>,
    pub runtime: Upload<'a>,
    pub paging_css: Section<'a>,
    pub paging: Vec<Section<'a>>,
}

/// A load-status transition in one fresh gen1 resident upload epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InitStatus {
    pub before: u32,
    pub after: u32,
}

/// Work the prepared transport must acknowledge while loading either image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitStep<'a> {
    /// Check that this is a fresh, prepared epoch with clear status.
    CheckFresh,
    /// Submit one chunk and wait for its FH-TX completion.
    Transfer {
        cpu: Cpu,
        section: usize,
        transfer: Transfer<'a>,
    },
    /// Publish cumulative section progress after every chunk has completed.
    SectionLoaded {
        cpu: Cpu,
        section: usize,
        status: InitStatus,
    },
    /// Prepare the post-load interrupt path before publishing CPU completion.
    /// The first completion must preserve FH TX for any resident CPU2 data.
    EnableInterrupts(Cpu),
    /// Publish a CPU completion sentinel. The all-ones CPU2 value alone
    /// cannot distinguish a live requester from a failed MMIO read.
    CpuComplete { cpu: Cpu, status: InitStatus },
}

#[derive(Debug, PartialEq, Eq)]
pub enum InitError<E> {
    Plan(Error),
    Step(E),
}

fn delimiter(section: &Section<'_>, offset: u32) -> bool {
    section.offset == offset && matches!(section.data, [] | [0, 0, 0, 0])
}

fn cpu_sections<'a>(sections: &[Section<'a>]) -> Result<Vec<Section<'a>>, Error> {
    if sections.len() > MAX_CPU_SECTIONS {
        return Err(Error::ResourceLimit);
    }
    if sections.iter().any(|section| {
        section.data.is_empty() || matches!(section.offset, CPU_SEPARATOR | PAGING_SEPARATOR)
    }) {
        return Err(Error::SectionLayout);
    }
    Ok(sections.to_vec())
}

fn upload<'a>(sections: &[Section<'a>], kind: ImageKind) -> Result<Upload<'a>, Error> {
    let (cpu1, cpu2) = match sections
        .iter()
        .position(|section| delimiter(section, CPU_SEPARATOR))
    {
        Some(at) => {
            let (cpu1, tail) = sections.split_at(at);
            let cpu2 = &tail[1..];
            if cpu2.is_empty() {
                return Err(Error::SectionLayout);
            }
            (cpu1, cpu2)
        }
        // API 46's released INIT images omit a separator. Linux uploads the
        // whole INIT image in CPU1's pass and completes CPU2 with no uploads.
        None if kind == ImageKind::Init => (sections, &[][..]),
        None => return Err(Error::SectionLayout),
    };
    if cpu1.is_empty() {
        return Err(Error::SectionLayout);
    }
    Ok(Upload {
        cpu1: cpu_sections(cpu1)?,
        cpu2: cpu_sections(cpu2)?,
    })
}

impl<'a> LoadPlan<'a> {
    /// Validate both uploads and the runtime paging payload before any DMA.
    pub fn from_firmware(firmware: &Firmware<'a>) -> Result<Self, Error> {
        let actual = firmware.api_version();
        if actual != API {
            return Err(Error::FirmwareApi {
                expected: API,
                actual,
            });
        }
        if firmware.cpus() != Some(2) {
            return Err(Error::Missing("dual CPU declaration"));
        }
        let for_image = |kind| {
            firmware
                .sections()
                .iter()
                .copied()
                .filter(move |section| section.image == kind)
        };
        let init: Vec<_> = for_image(ImageKind::Init).collect();
        let runtime: Vec<_> = for_image(ImageKind::Runtime).collect();
        let paging_at = runtime
            .iter()
            .position(|section| delimiter(section, PAGING_SEPARATOR))
            .ok_or(Error::SectionLayout)?;
        let (resident, tail) = runtime.split_at(paging_at);
        let (paging_css, paging) = tail[1..].split_first().ok_or(Error::SectionLayout)?;
        // Both released images have a final CPU marker after the page data.
        let paging = match paging.split_last() {
            Some((last, data)) if delimiter(last, CPU_SEPARATOR) => data,
            _ => paging,
        };
        let bytes = paging.iter().try_fold(0usize, |total, section| {
            if section.data.is_empty() || matches!(section.offset, CPU_SEPARATOR | PAGING_SEPARATOR)
            {
                return Err(Error::SectionLayout);
            }
            total
                .checked_add(section.data.len())
                .ok_or(Error::SectionLayout)
        })?;
        if paging_css.offset != 0
            || !(1..=4096).contains(&paging_css.data.len())
            || bytes == 0
            || firmware.paging_bytes().map(|size| size as usize) != Some(bytes)
        {
            return Err(Error::SectionLayout);
        }
        Ok(Self {
            init: upload(&init, ImageKind::Init)?,
            runtime: upload(resident, ImageKind::Runtime)?,
            paging_css: *paging_css,
            paging: paging.to_vec(),
        })
    }

    /// Run the gen1 INIT sequence over an already prepared transport. The
    /// transport acknowledges a transfer only after FH-TX completion, and a
    /// status step only after verifying the corresponding publication. A
    /// failed sequence requires a new, independently prepared epoch.
    ///
    /// The interrupt step is part of the Linux ordering. Its actual handoff
    /// is transport-specific: `service::ServiceChannel` alone polls under
    /// the FH-only mask. CPU completion is distinct from firmware ALIVE.
    pub fn upload_init<E>(
        &self,
        staging_capacity: NonZeroUsize,
        execute: impl FnMut(InitStep<'a>) -> Result<(), E>,
    ) -> Result<(), InitError<E>> {
        self.init
            .upload_resident(ImageKind::Init, staging_capacity, execute)
    }

    /// Run the resident REGULAR sequence in a fresh, prepared epoch after INIT
    /// has stopped. At the CPU1 handoff the transport keeps FH TX observable
    /// until every CPU2 transfer completes. ALIVE and paging are separate.
    pub fn upload_regular<E>(
        &self,
        staging_capacity: NonZeroUsize,
        execute: impl FnMut(InitStep<'a>) -> Result<(), E>,
    ) -> Result<(), InitError<E>> {
        self.runtime
            .upload_resident(ImageKind::Runtime, staging_capacity, execute)
    }
}

impl<'a> Upload<'a> {
    fn upload_resident<E>(
        &self,
        image: ImageKind,
        staging_capacity: NonZeroUsize,
        mut execute: impl FnMut(InitStep<'a>) -> Result<(), E>,
    ) -> Result<(), InitError<E>> {
        if image == ImageKind::Runtime && (self.cpu1.is_empty() || self.cpu2.is_empty()) {
            return Err(InitError::Plan(Error::SectionLayout));
        }
        let cpus = [Cpu::First, Cpu::Second];
        for cpu in cpus {
            let sections = self.sections(cpu);
            // Sixteen cumulative bits equal the CPU completion sentinel.
            // Keep section progress distinct before touching the transport.
            if sections.len() >= MAX_CPU_SECTIONS {
                return Err(InitError::Plan(Error::ResourceLimit));
            }
            if sections.iter().any(|section| section.image != image) {
                return Err(InitError::Plan(Error::SectionLayout));
            }
        }

        let mut step = |action| execute(action).map_err(InitError::Step);
        step(InitStep::CheckFresh)?;
        let mut status = 0;
        for cpu in cpus {
            let (shift, complete) = match cpu {
                Cpu::First => (0, u32::from(u16::MAX)),
                Cpu::Second => (16, u32::MAX),
            };
            for section in 0..self.sections(cpu).len() {
                let transfers = self
                    .transfers(cpu, section, staging_capacity)
                    .expect("section index comes from the upload");
                for transfer in transfers {
                    step(InitStep::Transfer {
                        cpu,
                        section,
                        transfer,
                    })?;
                }
                let cumulative = ((1u32 << (section + 1)) - 1) << shift;
                let after = status | cumulative;
                step(InitStep::SectionLoaded {
                    cpu,
                    section,
                    status: InitStatus {
                        before: status,
                        after,
                    },
                })?;
                status = after;
            }
            step(InitStep::EnableInterrupts(cpu))?;
            step(InitStep::CpuComplete {
                cpu,
                status: InitStatus {
                    before: status,
                    after: complete,
                },
            })?;
            status = complete;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
