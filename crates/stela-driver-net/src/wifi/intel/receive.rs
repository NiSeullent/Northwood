// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2003-2014, 2018-2026 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2016-2017 Intel Deutschland GmbH
//! Bounded wire decoders from pcie/gen1_2/rx.c and fw/api/alive.h.
//! These inspect already CPU-owned bytes. They establish no DMA completion,
//! current reset epoch, key state, regulatory permission or radio readiness.
use super::firmware::{Family, Firmware};
use crate::bytes::{le16, le32, le64};
use core::ops::Range;
pub mod mpdu;
pub mod reorder;

pub(super) const ALIVE_COMMAND: u8 = 1;
pub(super) const ALIVE_GROUP: u8 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    BufferSize,
    Length { offset: usize, raw: u32 },
    Queue { expected: u8, actual: u8 },
    DescriptorSize,
    DescriptorAddress,
    DescriptorId,
    NotAlive,
    AliveVersion(Option<u8>),
    AliveLength { minimum: usize, actual: usize },
    AliveRejected { status: u16, flags: u16 },
}

#[derive(Clone, Debug)]
pub struct Packet<'a> {
    pub raw_flags: u32,
    pub command: u8,
    pub group: u8,
    pub sequence: u16,
    pub range: Range<usize>,
    raw: &'a [u8],
}
impl<'a> Packet<'a> {
    pub fn raw(&self) -> &'a [u8] {
        self.raw
    }
    pub fn payload(&self) -> &'a [u8] {
        &self.raw[8..]
    }
    pub fn unsolicited(&self) -> bool {
        self.sequence & 0x8000 != 0
    }
    /// No-reclaim exceptions are separately negotiated by the command layer;
    /// an unset unsolicited bit alone never authorizes reclaiming a command.
    pub fn command_queue(&self) -> u8 {
        ((self.sequence >> 8) & 0x1f) as u8
    }
    pub fn command_index(&self) -> u8 {
        self.sequence as u8
    }
}
/// Source frames start at 64-byte boundaries. AX200 permits several per RB;
/// AX210 consumes one and leaves the rest uninterpreted. Errors fuse iteration.
#[derive(Clone)]
pub struct Packets<'a> {
    bytes: &'a [u8],
    one_packet: bool,
    queue: u8,
    offset: usize,
    ended: bool,
}
impl<'a> Packets<'a> {
    pub fn new(family: Family, queue: u8, bytes: &'a [u8]) -> Result<Self, Error> {
        Self::with_layout(queue, bytes, family >= Family::Ax210)
    }

    pub(super) fn pre_ax210_mq(queue: u8, bytes: &'a [u8]) -> Result<Self, Error> {
        Self::with_layout(queue, bytes, false)
    }

    fn with_layout(queue: u8, bytes: &'a [u8], one_packet: bool) -> Result<Self, Error> {
        if bytes.is_empty() || bytes.len() > 16384 || !bytes.len().is_multiple_of(64) || queue > 63
        {
            return Err(Error::BufferSize);
        }
        Ok(Self {
            bytes,
            one_packet,
            queue,
            offset: 0,
            ended: false,
        })
    }
}
impl<'a> Iterator for Packets<'a> {
    type Item = Result<Packet<'a>, Error>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.ended || self.offset >= self.bytes.len() {
            return None;
        }
        let start = self.offset;
        let flags = le32(self.bytes, start);
        if flags == 0x55550000 {
            self.ended = true;
            return None;
        }
        let length = (flags & 0x3fff) as usize + 4;
        let next = start + length.next_multiple_of(64);
        self.ended = true;
        if length < 8 || next > self.bytes.len() {
            return Some(Err(Error::Length {
                offset: start,
                raw: flags,
            }));
        }
        let actual = ((flags >> 16) & 0x3f) as u8;
        if actual != self.queue {
            return Some(Err(Error::Queue {
                expected: self.queue,
                actual,
            }));
        }
        let raw = &self.bytes[start..start + length];
        self.offset = next;
        self.ended = self.one_packet;
        Some(Ok(Packet {
            raw_flags: flags,
            command: raw[4],
            group: raw[5],
            sequence: le16(raw, 6),
            range: start..start + length,
            raw,
        }))
    }
}
impl core::iter::FusedIterator for Packets<'_> {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreeDescriptor {
    Ax200([u8; 8]),
    Ax210([u8; 16]),
}

pub(super) fn pre_ax210_mq_free(iova: u64, id: u16) -> Result<[u8; 8], Error> {
    if iova == 0 || !iova.is_multiple_of(4096) {
        return Err(Error::DescriptorAddress);
    }
    if !(1..=0xfff).contains(&id) {
        return Err(Error::DescriptorId);
    }
    Ok((iova | u64::from(id)).to_le_bytes())
}

pub(super) fn pre_ax210_mq_vid(raw: u32) -> u16 {
    (raw & 0xfff) as u16
}

/// Exact donor layouts: AX200 IOVA|12-bitVID; AX210 VID at0 and IOVA at8.
/// This serialization neither publishes an address nor transfers DMA ownership.
pub fn free_descriptor(family: Family, iova: u64, id: u16) -> Result<FreeDescriptor, Error> {
    if family == Family::Ax200 {
        return pre_ax210_mq_free(iova, id).map(FreeDescriptor::Ax200);
    }
    if iova == 0 || !iova.is_multiple_of(2048) {
        return Err(Error::DescriptorAddress);
    }
    if id == 0 {
        return Err(Error::DescriptorId);
    }
    let mut bytes = [0; 16];
    bytes[..2].copy_from_slice(&id.to_le_bytes());
    bytes[8..].copy_from_slice(&iova.to_le_bytes());
    Ok(FreeDescriptor::Ax210(bytes))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UsedDescriptor<'a> {
    pub id: u16,
    pub fragmented: bool,
    pub raw: &'a [u8],
}
/// `iwl_pcie_used_bd_size`.
pub(super) const fn used_stride(family: Family) -> usize {
    match family {
        Family::Ax200 | Family::Bz | Family::Sc => 4,
        Family::Ax210 => 32,
    }
}
pub fn used_descriptor(family: Family, raw: &[u8]) -> Result<UsedDescriptor<'_>, Error> {
    let expected = used_stride(family);
    if raw.len() != expected {
        return Err(Error::DescriptorSize);
    }
    let (id, fragmented) = match family {
        Family::Ax200 => (pre_ax210_mq_vid(le32(raw, 0)), false),
        Family::Ax210 => (le16(raw, 4), raw[6] & 1 != 0),
        Family::Bz | Family::Sc => (le16(raw, 0), raw[2] & 1 != 0),
    };
    if id == 0 || (family == Family::Ax200 && id > 0xfff) {
        return Err(Error::DescriptorId);
    }
    Ok(UsedDescriptor {
        id,
        fragmented,
        raw,
    })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lmac {
    pub major: u32,
    pub minor: u32,
    pub subtype: u8,
    pub kind: u8,
    pub mac: u8,
    pub options: u8,
    pub timestamp: u32,
    pub debug_addresses: [u32; 8],
}
impl Lmac {
    pub(super) fn decode(bytes: &[u8]) -> Self {
        Self {
            major: le32(bytes, 0),
            minor: le32(bytes, 4),
            subtype: bytes[8],
            kind: bytes[9],
            mac: bytes[10],
            options: bytes[11],
            timestamp: le32(bytes, 12),
            debug_addresses: core::array::from_fn(|i| le32(bytes, 16 + i * 4)),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Umac {
    pub major: u32,
    pub minor: u32,
    pub debug_addresses: [u32; 2],
}
impl Umac {
    pub(super) fn decode(bytes: &[u8]) -> Self {
        Self {
            major: le32(bytes, 0),
            minor: le32(bytes, 4),
            debug_addresses: [le32(bytes, 8), le32(bytes, 12)],
        }
    }
}

pub(super) fn alive_flags(payload: &[u8]) -> Result<u16, Error> {
    let status = le16(payload, 0);
    let flags = le16(payload, 2);
    if status != 0xcafe {
        return Err(Error::AliveRejected { status, flags });
    }
    Ok(flags)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Imr {
    pub base: u64,
    pub bytes: u32,
    pub enabled: u32,
}
#[derive(Clone, Copy, Debug)]
pub struct Alive<'a> {
    pub notification_version: u8,
    pub flags: u16,
    pub lmac: [Lmac; 2],
    pub umac: Umac,
    pub sku: [u32; 3],
    pub imr: Imr,
    pub platform_id: Option<u64>,
    raw: &'a [u8],
}
impl<'a> Alive<'a> {
    /// Decode the image's negotiated layout, never a guessed default.
    /// The result is plain metadata; only a transport receiving this
    /// packet via its completed current-epoch RX may release ROM DMA.
    pub fn decode(packet: &Packet<'a>, firmware: &Firmware<'_>) -> Result<Self, Error> {
        if packet.command != ALIVE_COMMAND || packet.group != ALIVE_GROUP {
            return Err(Error::NotAlive);
        }
        let version = firmware
            .command_version(ALIVE_GROUP, ALIVE_COMMAND)
            .map(|v| v.notification_version);
        let minimum = match version {
            Some(6 | 7) => 144,
            Some(8) => 152,
            _ => return Err(Error::AliveVersion(version)),
        };
        let raw = packet.payload();
        if raw.len() < minimum {
            return Err(Error::AliveLength {
                minimum,
                actual: raw.len(),
            });
        }
        Ok(Self {
            notification_version: version.unwrap(),
            flags: alive_flags(raw)?,
            lmac: [Lmac::decode(&raw[4..52]), Lmac::decode(&raw[52..100])],
            umac: Umac::decode(&raw[100..116]),
            sku: core::array::from_fn(|i| le32(raw, 116 + i * 4)),
            imr: Imr {
                base: le64(raw, 128),
                bytes: le32(raw, 136),
                enabled: le32(raw, 140),
            },
            platform_id: (minimum == 152).then(|| le64(raw, 144)),
            raw,
        })
    }
    pub fn raw(self) -> &'a [u8] {
        self.raw
    }
}

#[cfg(test)]
mod tests;
