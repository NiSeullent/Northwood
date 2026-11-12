// SPDX-License-Identifier: GPL-2.0-or-later
//! Northwood's volatile host swap format. The header sits at offset zero in
//! a dedicated partition or the first extent of a preallocated volume store.
//! Integers are little-endian and the UUID uses GPT's byte order.
//!
//! | Offset | Bytes | Field |
//! | --- | --- | --- |
//! | 0 | 8 | Magic `NWSWAP01` |
//! | 8 | 4 | Version, 1 |
//! | 12 | 4 | Header bytes, 4096 |
//! | 16 | 4 | Slot bytes, 16384 |
//! | 20 | 4 | Reserved, zero |
//! | 24 | 8 | Slot count |
//! | 32 | 8 | Bitmap offset, 1 MiB |
//! | 40 | 8 | Data offset, rounded to 1 MiB |
//! | 48 | 16 | Store UUID |
//! | 64 | 8 | Boot generation |
//! | 72 | 4 | State, idle (0) or in use (1) |
//! | 76 | 4 | Flags, bit 0 signals encryption |
//! | 80 | 4012 | Reserved, zero |
//! | 4092 | 4 | IEEE CRC-32 over the preceding 4092 bytes |
//!
//! V1 reserves the bitmap area on disk and keeps slot ownership in RAM.
//! A new boot chooses its own generation and starts with all slots free.

use super::gpt::{Guid, NORTHWOOD_SWAP_TYPE, Partition};
use core::fmt;
use stela_contracts::{
    bytes::{array, le32, le64},
    crc::crc32_ieee,
};

pub mod partition;
pub mod session;
pub mod volume;

const MAGIC: &[u8; 8] = b"NWSWAP01";
const VERSION: u32 = 1;
const ALIGNMENT: u64 = 1 << 20;
const SLOTS_PER_BITMAP_BLOCK: u64 = ALIGNMENT * 8;
const ENCRYPTED: u32 = 1;
const CHECKSUM_OFFSET: usize = HEADER_BYTES - 4;

pub const HEADER_BYTES: usize = 4096;
pub const SLOT_BYTES: u64 = 16 << 10;
pub const BITMAP_OFFSET: u64 = ALIGNMENT;
pub const MIN_STORE_BYTES: u64 = BITMAP_OFFSET + ALIGNMENT + SLOT_BYTES;
pub const DEFAULT_STORE_BYTES: u64 = 32 << 30;
pub const EXTENT_BYTES: u64 = 1 << 30;
pub const DEFAULT_VOLUME_MAX_BYTES: u64 = 8 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    slot_count: u64,
    data_offset: u64,
    used_bytes: u64,
}

impl Layout {
    /// Fill `store_bytes` with as many complete slots as fit after metadata.
    pub fn for_store(store_bytes: u64) -> Result<Self, Error> {
        if store_bytes < MIN_STORE_BYTES {
            return Err(Error::ShortStore {
                available: store_bytes,
                required: MIN_STORE_BYTES,
            });
        }
        // Each complete bitmap block describes this many bytes of slot data.
        // Account for the bitmap bytes first, so a slot at the boundary is
        // included only when its new bitmap block also fits.
        let bitmap_and_slots = ALIGNMENT + SLOTS_PER_BITMAP_BLOCK * SLOT_BYTES;
        let available = store_bytes - BITMAP_OFFSET;
        let complete = available / bitmap_and_slots;
        let trailing = (available % bitmap_and_slots).saturating_sub(ALIGNMENT) / SLOT_BYTES;
        Self::for_slots(complete * SLOTS_PER_BITMAP_BLOCK + trailing)
    }

    fn for_slots(slot_count: u64) -> Result<Self, Error> {
        if slot_count == 0 {
            return Err(Error::Layout);
        }
        let bitmap_bytes = slot_count.div_ceil(SLOTS_PER_BITMAP_BLOCK) * ALIGNMENT;
        let data_offset = BITMAP_OFFSET + bitmap_bytes;
        let used_bytes = slot_count
            .checked_mul(SLOT_BYTES)
            .and_then(|slots| data_offset.checked_add(slots))
            .ok_or(Error::Layout)?;
        Ok(Self {
            slot_count,
            data_offset,
            used_bytes,
        })
    }

    pub fn slot_count(self) -> u64 {
        self.slot_count
    }

    pub fn bitmap_bytes(self) -> u64 {
        self.data_offset - BITMAP_OFFSET
    }

    pub fn data_offset(self) -> u64 {
        self.data_offset
    }

    pub fn used_bytes(self) -> u64 {
        self.used_bytes
    }

    /// Byte offset relative to the start of the store.
    pub fn slot_offset(self, slot: u64) -> Option<u64> {
        (slot < self.slot_count).then(|| self.data_offset + slot * SLOT_BYTES)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum State {
    Idle = 0,
    InUse = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    layout: Layout,
    store_uuid: Guid,
    generation: u64,
    state: State,
    encrypted: bool,
}

impl Header {
    /// Create an idle header. A partition's store UUID is its unique GPT GUID.
    pub fn new(store_bytes: u64, store_uuid: Guid) -> Result<Self, Error> {
        if store_uuid.is_zero() {
            return Err(Error::StoreUuid);
        }
        Ok(Self {
            layout: Layout::for_store(store_bytes)?,
            store_uuid,
            generation: 0,
            state: State::Idle,
            encrypted: false,
        })
    }

    /// Start a fresh boot generation. The caller supplies fresh entropy and
    /// discards all slot ownership from any prior session.
    pub fn begin_session(self, generation: u64, encrypted: bool) -> Result<Self, Error> {
        if generation == 0 || generation == self.generation {
            return Err(Error::Generation);
        }
        Ok(Self {
            generation,
            state: State::InUse,
            encrypted,
            ..self
        })
    }

    pub fn layout(self) -> Layout {
        self.layout
    }

    pub fn store_uuid(self) -> Guid {
        self.store_uuid
    }

    pub fn generation(self) -> u64 {
        self.generation
    }

    pub fn state(self) -> State {
        self.state
    }

    pub fn encrypted(self) -> bool {
        self.encrypted
    }

    pub fn encode(self) -> [u8; HEADER_BYTES] {
        let mut bytes = [0; HEADER_BYTES];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..12].copy_from_slice(&VERSION.to_le_bytes());
        bytes[12..16].copy_from_slice(&(HEADER_BYTES as u32).to_le_bytes());
        bytes[16..20].copy_from_slice(&(SLOT_BYTES as u32).to_le_bytes());
        bytes[24..32].copy_from_slice(&self.layout.slot_count.to_le_bytes());
        bytes[32..40].copy_from_slice(&BITMAP_OFFSET.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.layout.data_offset.to_le_bytes());
        bytes[48..64].copy_from_slice(&self.store_uuid.0);
        bytes[64..72].copy_from_slice(&self.generation.to_le_bytes());
        bytes[72..76].copy_from_slice(&(self.state as u32).to_le_bytes());
        let flags = if self.encrypted { ENCRYPTED } else { 0 };
        bytes[76..80].copy_from_slice(&flags.to_le_bytes());
        let checksum = crc32_ieee(&bytes[..CHECKSUM_OFFSET]);
        bytes[CHECKSUM_OFFSET..].copy_from_slice(&checksum.to_le_bytes());
        bytes
    }

    /// Validate a header read from a bounded store of `store_bytes` bytes.
    pub fn decode(bytes: &[u8], store_bytes: u64) -> Result<Self, Error> {
        let bytes = bytes.get(..HEADER_BYTES).ok_or(Error::ShortHeader)?;
        if &bytes[..8] != MAGIC {
            return Err(Error::Magic);
        }
        let version = le32(bytes, 8);
        if version != VERSION {
            return Err(Error::Version(version));
        }
        if le32(bytes, CHECKSUM_OFFSET) != crc32_ieee(&bytes[..CHECKSUM_OFFSET]) {
            return Err(Error::Checksum);
        }
        if le32(bytes, 20) != 0 || bytes[80..CHECKSUM_OFFSET].iter().any(|&byte| byte != 0) {
            return Err(Error::Reserved);
        }
        let layout = Layout::for_slots(le64(bytes, 24))?;
        if le32(bytes, 12) != HEADER_BYTES as u32
            || le32(bytes, 16) != SLOT_BYTES as u32
            || le64(bytes, 32) != BITMAP_OFFSET
            || le64(bytes, 40) != layout.data_offset
        {
            return Err(Error::Layout);
        }
        if layout.used_bytes > store_bytes {
            return Err(Error::ShortStore {
                available: store_bytes,
                required: layout.used_bytes,
            });
        }
        let store_uuid = Guid(array(bytes, 48));
        if store_uuid.is_zero() {
            return Err(Error::StoreUuid);
        }
        let generation = le64(bytes, 64);
        let state = match le32(bytes, 72) {
            0 => State::Idle,
            1 if generation != 0 => State::InUse,
            1 => return Err(Error::Generation),
            state => return Err(Error::State(state)),
        };
        let flags = le32(bytes, 76);
        if flags & !ENCRYPTED != 0 {
            return Err(Error::Flags(flags));
        }
        Ok(Self {
            layout,
            store_uuid,
            generation,
            state,
            encrypted: flags & ENCRYPTED != 0,
        })
    }

    /// Check the partition type, extent, and identity as well as the header.
    /// `bytes` must have been read at the start of `partition`.
    pub fn decode_partition(
        bytes: &[u8],
        partition: &Partition,
        block_bytes: u64,
    ) -> Result<Self, Error> {
        if partition.type_guid != NORTHWOOD_SWAP_TYPE {
            return Err(Error::PartitionType);
        }
        if !matches!(block_bytes, 512 | 1024 | 2048 | 4096) {
            return Err(Error::BlockSize(block_bytes));
        }
        let store_bytes = partition
            .last_lba
            .checked_sub(partition.first_lba)
            .and_then(|span| span.checked_add(1))
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or(Error::Layout)?;
        let header = Self::decode(bytes, store_bytes)?;
        if header.store_uuid != partition.unique_guid {
            return Err(Error::StoreUuid);
        }
        Ok(header)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    ShortHeader,
    Magic,
    Version(u32),
    Checksum,
    Reserved,
    Layout,
    ShortStore { available: u64, required: u64 },
    StoreUuid,
    Generation,
    State(u32),
    Flags(u32),
    PartitionType,
    BlockSize(u64),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShortHeader => f.write_str("truncated Northwood Swap header"),
            Self::Magic => f.write_str("missing Northwood Swap header magic"),
            Self::Version(version) => write!(f, "unsupported Northwood Swap version {version}"),
            Self::Checksum => f.write_str("Northwood Swap header CRC mismatch"),
            Self::Reserved => f.write_str("nonzero reserved bytes in Northwood Swap header"),
            Self::Layout => f.write_str("invalid Northwood Swap store layout"),
            Self::ShortStore {
                available,
                required,
            } => write!(
                f,
                "Northwood Swap needs {required} bytes but the store has {available}"
            ),
            Self::StoreUuid => f.write_str("invalid or mismatched Northwood Swap store UUID"),
            Self::Generation => f.write_str("invalid or repeated Northwood Swap generation"),
            Self::State(state) => write!(f, "unknown Northwood Swap state {state}"),
            Self::Flags(flags) => write!(f, "unknown Northwood Swap flags {flags:#x}"),
            Self::PartitionType => f.write_str("wrong GPT partition type for Northwood Swap"),
            Self::BlockSize(bytes) => write!(f, "unsupported Northwood Swap block size {bytes}"),
        }
    }
}

impl core::error::Error for Error {}

#[cfg(test)]
mod tests;
