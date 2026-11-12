// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2005-2014, 2020-2021, 2025 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2016-2017 Intel Deutschland GmbH
//! Intel 9000 INIT PHY DB notifications and REGULAR replay. The layout and
//! ordering follow Linux iwlwifi's iwl-phy-db.c and fw/api/cmdhdr.h at
//! cc9de5d4b5ab6459f5774946aa3195ad51236fa8.

use super::rx::{BUFFER_BYTES, ENTRIES};
use crate::wifi::intel::command::Id;
use alloc::{collections::BTreeMap, vec::Vec};
use core::ops::Bound;

const RX_HEADER_BYTES: usize = 8;
const RX_PACKET_ALIGNMENT: usize = 64;
const SECTION_HEADER_BYTES: usize = 4;
const MAX_SECTION_BYTES: usize = BUFFER_BYTES - RX_HEADER_BYTES - SECTION_HEADER_BYTES;
const MAX_DATABASE_BYTES: usize = BUFFER_BYTES * ENTRIES as usize;
// The donor passes the highest received group plus one as a u8 count.
const MAX_REPLAY_GROUP: u16 = u8::MAX as u16 - 1;

pub(super) const ID: Id = Id {
    group: 1,
    opcode: 0x6c,
    version: 0,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Configuration,
    NonChannel,
    Papd,
    TxPower,
}

impl Kind {
    pub const fn id(self) -> u16 {
        match self {
            Self::Configuration => 1,
            Self::NonChannel => 2,
            Self::Papd => 4,
            Self::TxPower => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Section<'a> {
    pub kind: Kind,
    pub group: Option<u16>,
    pub data: &'a [u8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Length {
        expected: usize,
        actual: usize,
    },
    SectionSize(usize),
    UnknownType(u16),
    GroupLength {
        kind: Kind,
        minimum: usize,
        actual: usize,
    },
    GroupOrder {
        kind: Kind,
        highest: u16,
        actual: u16,
    },
    DatabaseFull,
    MissingSection(Kind),
    ReplayGroup {
        kind: Kind,
        group: u16,
    },
}

pub(super) type Key = (Kind, Option<u16>);

#[derive(Debug, Default)]
pub struct Database {
    sections: BTreeMap<Key, Vec<u8>>,
    highest_papd: Option<u16>,
    highest_tx_power: Option<u16>,
    bytes: usize,
}

impl Database {
    /// Record a PHY DB notification payload from the current INIT RX queue.
    /// The caller matches its notification ID and RX generation.
    pub fn record(&mut self, payload: &[u8]) -> Result<(), Error> {
        let (header, data) =
            payload
                .split_at_checked(SECTION_HEADER_BYTES)
                .ok_or(Error::Length {
                    expected: SECTION_HEADER_BYTES,
                    actual: payload.len(),
                })?;
        let raw_kind = u16::from_le_bytes([header[0], header[1]]);
        let length = usize::from(u16::from_le_bytes([header[2], header[3]]));
        if length > MAX_SECTION_BYTES {
            return Err(Error::SectionSize(length));
        }
        if data.len() != length {
            return Err(Error::Length {
                expected: SECTION_HEADER_BYTES + length,
                actual: payload.len(),
            });
        }
        let kind = match raw_kind {
            1 => Kind::Configuration,
            2 => Kind::NonChannel,
            4 => Kind::Papd,
            5 => Kind::TxPower,
            _ => return Err(Error::UnknownType(raw_kind)),
        };
        let grouped = match kind {
            Kind::Papd => Some((&mut self.highest_papd, 2)),
            Kind::TxPower => Some((&mut self.highest_tx_power, 6)),
            _ => None,
        };
        let group = match grouped.as_ref() {
            Some((highest, minimum)) => {
                if length < *minimum {
                    return Err(Error::GroupLength {
                        kind,
                        minimum: *minimum,
                        actual: length,
                    });
                }
                let actual = u16::from_le_bytes([data[0], data[1]]);
                if let Some(highest) = **highest
                    && actual > highest
                {
                    return Err(Error::GroupOrder {
                        kind,
                        highest,
                        actual,
                    });
                }
                Some(actual)
            }
            None => None,
        };
        let key = (kind, group);
        // Charge one aligned RFH packet per retained section. The first
        // group ID bounds later IDs without preallocating absent groups.
        let old = self
            .sections
            .get(&key)
            .map_or(0, |data| packet_bytes(data.len()));
        let bytes = self.bytes - old + packet_bytes(length);
        if bytes > MAX_DATABASE_BYTES {
            return Err(Error::DatabaseFull);
        }
        self.sections.insert(key, data.to_vec());
        self.bytes = bytes;
        if let Some(((highest, _), group)) = grouped.zip(group) {
            highest.get_or_insert(group);
        }
        Ok(())
    }

    /// Borrow the received sections in the order used by REGULAR replay.
    pub fn sections(&self) -> impl ExactSizeIterator<Item = Section<'_>> {
        self.sections
            .iter()
            .map(|(&(kind, group), data)| Section { kind, group, data })
    }

    /// Only replay data observed during this INIT. Group holes are valid,
    /// while an entire missing family or singleton leaves the DB incomplete.
    pub(super) fn replay_count(&self) -> Result<usize, Error> {
        for kind in [Kind::Configuration, Kind::NonChannel] {
            if !self.sections.contains_key(&(kind, None)) {
                return Err(Error::MissingSection(kind));
            }
        }
        for (kind, highest) in [
            (Kind::Papd, self.highest_papd),
            (Kind::TxPower, self.highest_tx_power),
        ] {
            let group = highest.ok_or(Error::MissingSection(kind))?;
            if group > MAX_REPLAY_GROUP {
                return Err(Error::ReplayGroup { kind, group });
            }
        }
        Ok(self.sections.len())
    }

    pub(super) fn next_replay(&self, after: Option<Key>) -> Option<(Key, Request<'_>)> {
        let start = after.map_or(Bound::Unbounded, Bound::Excluded);
        let (&key, data) = self.sections.range((start, Bound::Unbounded)).next()?;
        Some((key, Request { kind: key.0, data }))
    }
}

/// A section from the stopped INIT's retained data, without its RX header.
pub(super) struct Request<'a> {
    kind: Kind,
    data: &'a [u8],
}

impl Request<'_> {
    pub(super) fn payload(&self) -> Vec<u8> {
        let mut payload = Vec::with_capacity(SECTION_HEADER_BYTES + self.data.len());
        payload.extend(self.kind.id().to_le_bytes());
        payload.extend((self.data.len() as u16).to_le_bytes());
        payload.extend_from_slice(self.data);
        payload
    }
}

fn packet_bytes(section_bytes: usize) -> usize {
    (RX_HEADER_BYTES + SECTION_HEADER_BYTES + section_bytes).next_multiple_of(RX_PACKET_ALIGNMENT)
}

#[cfg(test)]
mod tests;
