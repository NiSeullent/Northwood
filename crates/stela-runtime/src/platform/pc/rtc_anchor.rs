//! The PMU upcount's anchor, on the machine-state partition.
//!
//! A Mac's PMIC keeps its RTC upcount running while the machine is off, and
//! macOS keeps its own clock offset against it. Stela keeps the upcount at
//! one moment and the host's Unix seconds then
//! ([`RtcAnchor`](stela_soc::machine::RtcAnchor)), so the next boot's upcount
//! goes on from it by the wall time between
//! ([`RtcEpoch::anchored`](stela_soc::machine::RtcEpoch::anchored)).
//!
//! The region starts at [`OFFSET`] of the partition, after the board flash,
//! and holds two [`SLOT_BYTES`] slots written in turn, so a torn write
//! leaves the other slot's anchor. Each slot begins with a record,
//! little-endian; the rest of the slot is zero:
//!
//! | Offset | Bytes | Field |
//! | --- | --- | --- |
//! | 0 | 8 | magic `STELARTC` |
//! | 8 | 4 | version, 1 |
//! | 12 | 4 | `length`: the hashed fields' bytes, 40 for version 1 |
//! | 16 | 8 | sequence, one more than the other slot's at the write |
//! | 24 | 8 | upcount ticks at 32768 Hz |
//! | 32 | 8 | the host's Unix seconds then, 0 when unknown |
//! | `length` | 32 | SHA-256 of bytes 0..`length` |
//!
//! A later version keeps these fields where they are and adds its own
//! after them, raising `length`; a reader checks the hash over `length`
//! bytes and reads the fields it knows, so builds of either version read
//! each other's anchors. Builds before the anchor never read or write the
//! region. A blank, foreign or torn slot holds no anchor.
use alloc::vec;
use alloc::vec::Vec;
use stela_contracts::bytes::{le32, le64};
use stela_crypto::Sha256;
use stela_crypto::digest::Digest;
use stela_drivers::block::{BlockDevice, BlockError, Request};
use stela_soc::machine::RtcAnchor;

/// Where the region starts in the machine-state partition.
pub const OFFSET: u64 = super::board_flash::PARTITION_BYTES;
/// One slot: one block on 4096-byte devices, eight on 512-byte ones.
pub const SLOT_BYTES: usize = 4096;
/// The region's bytes: two slots.
pub const BYTES: u64 = 2 * SLOT_BYTES as u64;
/// The smallest machine-state partition that holds the region.
pub const PARTITION_BYTES: u64 = OFFSET + BYTES;
const MAGIC: &[u8; 8] = b"STELARTC";
const VERSION: u32 = 1;
/// The version 1 fields, which every version's record begins with.
const FIELDS: usize = 40;
const HASH: usize = 32;

/// The slot the next anchor goes to, and its sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Anchors {
    slot: usize,
    sequence: u64,
}

/// A slot's anchor and sequence, when it holds a valid record.
pub fn decode(slot: &[u8]) -> Option<(u64, RtcAnchor)> {
    if slot.len() < FIELDS + HASH || &slot[..8] != MAGIC || le32(slot, 8) == 0 {
        return None;
    }
    let length = le32(slot, 12) as usize;
    if length < FIELDS || length.checked_add(HASH)? > slot.len() {
        return None;
    }
    if Sha256::digest(&slot[..length])[..] != slot[length..length + HASH] {
        return None;
    }
    let unix_seconds = le64(slot, 32);
    Some((
        le64(slot, 16),
        RtcAnchor {
            ticks: le64(slot, 24),
            unix_seconds: (unix_seconds != 0).then_some(unix_seconds),
        },
    ))
}

/// A slot holding `anchor` with `sequence`.
pub fn encode(sequence: u64, anchor: RtcAnchor) -> Vec<u8> {
    let mut slot = vec![0; SLOT_BYTES];
    slot[..8].copy_from_slice(MAGIC);
    slot[8..12].copy_from_slice(&VERSION.to_le_bytes());
    slot[12..16].copy_from_slice(&(FIELDS as u32).to_le_bytes());
    slot[16..24].copy_from_slice(&sequence.to_le_bytes());
    slot[24..32].copy_from_slice(&anchor.ticks.to_le_bytes());
    slot[32..40].copy_from_slice(&anchor.unix_seconds.unwrap_or(0).to_le_bytes());
    let digest = Sha256::digest(&slot[..FIELDS]);
    slot[FIELDS..FIELDS + HASH].copy_from_slice(&digest);
    slot
}

fn block_size(device: &dyn BlockDevice) -> Result<u64, BlockError> {
    let block = device.identity().block_size as usize;
    if block == 0 || !SLOT_BYTES.is_multiple_of(block) {
        return Err(BlockError::Protocol(
            "the device's blocks do not divide an RTC anchor slot",
        ));
    }
    Ok(block as u64)
}

/// The newest anchor on `device`, the region alone, and where the next one
/// goes: the other slot.
pub fn read(device: &mut dyn BlockDevice) -> Result<(Option<RtcAnchor>, Anchors), BlockError> {
    block_size(device)?;
    let mut bytes = vec![0; BYTES as usize];
    device.read(0, &mut bytes)?;
    let newest = bytes
        .chunks(SLOT_BYTES)
        .enumerate()
        .filter_map(|(slot, bytes)| Some((slot, decode(bytes)?)))
        .max_by_key(|(_, (sequence, _))| *sequence);
    Ok(match newest {
        Some((slot, (sequence, anchor))) => (
            Some(anchor),
            Anchors {
                slot: 1 - slot,
                sequence: sequence.wrapping_add(1),
            },
        ),
        None => (
            None,
            Anchors {
                slot: 0,
                sequence: 1,
            },
        ),
    })
}

impl Anchors {
    /// The durable write that keeps `anchor` on a device of `block_size`
    /// blocks, the region alone; the one after it goes to the other slot.
    pub fn request(&mut self, anchor: RtcAnchor, block_size: u32) -> Request {
        let lba = (self.slot * SLOT_BYTES) as u64 / u64::from(block_size.max(1));
        let data = encode(self.sequence, anchor);
        self.slot = 1 - self.slot;
        self.sequence = self.sequence.wrapping_add(1);
        Request::Write {
            lba,
            data,
            durable: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stela_drivers::block::RamDisk;

    fn disk(block_size: u32) -> RamDisk {
        RamDisk::new("state", BYTES / u64::from(block_size), block_size)
    }

    fn anchor(ticks: u64, unix_seconds: Option<u64>) -> RtcAnchor {
        RtcAnchor {
            ticks,
            unix_seconds,
        }
    }

    fn keep(disk: &mut RamDisk, anchors: &mut Anchors, kept: RtcAnchor) {
        let block = disk.identity().block_size;
        anchors.request(kept, block).run(disk).unwrap();
    }

    #[test]
    fn a_blank_region_holds_no_anchor_and_the_first_goes_to_slot_zero() {
        let mut disk = disk(512);
        let (none, mut anchors) = read(&mut disk).unwrap();
        assert_eq!(none, None);
        keep(&mut disk, &mut anchors, anchor(5, Some(1_800_000_000)));
        assert!(disk.bytes[SLOT_BYTES..].iter().all(|&byte| byte == 0));
        assert_eq!(
            read(&mut disk).unwrap().0,
            Some(anchor(5, Some(1_800_000_000)))
        );
    }

    #[test]
    fn anchors_round_trip_in_turn_and_a_torn_slot_leaves_the_other() {
        for block in [512, 4096] {
            let mut disk = disk(block);
            let (_, mut anchors) = read(&mut disk).unwrap();
            for ticks in 1..=5 {
                keep(&mut disk, &mut anchors, anchor(ticks << 20, Some(ticks)));
                let (newest, next) = read(&mut disk).unwrap();
                assert_eq!(newest, Some(anchor(ticks << 20, Some(ticks))));
                assert_eq!(next, anchors);
            }
            // Unknown wall seconds round trip as unknown.
            keep(&mut disk, &mut anchors, anchor(7 << 20, None));
            assert_eq!(read(&mut disk).unwrap().0, Some(anchor(7 << 20, None)));
            // The newest (slot 1 after six writes) torn: the older one reads.
            disk.bytes[SLOT_BYTES + 24] ^= 1;
            assert_eq!(read(&mut disk).unwrap().0, Some(anchor(5 << 20, Some(5))));
        }
    }

    #[test]
    fn a_newer_version_with_more_fields_reads_and_foreign_bytes_do_not() {
        let mut slot = encode(9, anchor(123, Some(456)));
        // Version 2 adds eight bytes after the version 1 fields.
        slot[8..12].copy_from_slice(&2u32.to_le_bytes());
        slot[12..16].copy_from_slice(&48u32.to_le_bytes());
        slot[40..48].copy_from_slice(&0xabu64.to_le_bytes());
        let digest = Sha256::digest(&slot[..48]);
        slot[48..80].copy_from_slice(&digest);
        assert_eq!(decode(&slot), Some((9, anchor(123, Some(456)))));
        let mut foreign = slot.clone();
        foreign[0] = b'X';
        assert_eq!(decode(&foreign), None);
        let mut short = slot.clone();
        short[12..16].copy_from_slice(&8u32.to_le_bytes());
        assert_eq!(decode(&short), None);
        let mut long = slot;
        long[12..16].copy_from_slice(&(SLOT_BYTES as u32).to_le_bytes());
        assert_eq!(decode(&long), None);
    }

    #[test]
    fn the_region_follows_the_board_flash_inside_the_machine_state() {
        assert_eq!(OFFSET, 0xe1_0000);
        const { assert!(PARTITION_BYTES <= super::super::machine_state::MACHINE_STATE_BYTES) };
    }
}
