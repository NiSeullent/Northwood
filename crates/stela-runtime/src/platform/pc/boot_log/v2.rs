// SPDX-License-Identifier: GPL-2.0-or-later

//! Read-only compatibility with Stambha's checksummed ring and boot records.
//! Based on Stambha 0.6.1's `platform/pc/boot_log.rs` format.

use super::{BootStart, Digest, HeaderError, Log, MAGIC, Sha256, marker, read_stream};
use alloc::vec::Vec;

pub(super) const VERSION: u32 = 2;
pub(super) const COPY_AT: usize = 3584;
const FIELDS: usize = 56;
const RECORD_BYTES: usize = 24;
const MAX_RECORDS: usize = 16;

pub(super) fn read(header: &[u8], region: &[u8]) -> Result<Log, HeaderError> {
    if header.len() < FIELDS || !header.starts_with(MAGIC) {
        return Err(HeaderError::NotALog);
    }
    let u32_at = |at| u32::from_le_bytes(header[at..at + 4].try_into().unwrap());
    let u64_at = |at| u64::from_le_bytes(header[at..at + 8].try_into().unwrap());
    let version = u32_at(8);
    if version != VERSION {
        return Err(HeaderError::Version(version));
    }
    let count = u32_at(40) as usize;
    if count > MAX_RECORDS {
        return Err(HeaderError::Records(count as u32));
    }
    let fields = FIELDS + count * RECORD_BYTES;
    let Some(stored_digest) = header.get(fields..fields + 32) else {
        return Err(HeaderError::Checksum);
    };
    if stored_digest != Sha256::digest(&header[..fields]).as_slice() {
        return Err(HeaderError::Checksum);
    }
    if u32_at(12) != 0 || u32_at(52) != 0 {
        return Err(HeaderError::Reserved);
    }

    let ring_bytes = u64_at(32);
    let first_bytes = u32_at(44) as u64;
    let last_bytes = u32_at(48) as u64;
    let record_bytes = first_bytes + last_bytes;
    let used = ring_bytes.checked_add(count as u64 * record_bytes);
    if ring_bytes == 0 || used != Some(region.len() as u64) {
        return Err(HeaderError::RingSize {
            recorded: used.unwrap_or(u64::MAX),
            region: region.len() as u64,
        });
    }
    let (ring, records) = region.split_at(ring_bytes as usize);
    let head = u64_at(16);
    let boots = u64_at(24);
    let mut stream = read_stream(head, ring);
    // A larger ring may have unwritten zero bytes before its retained text.
    let unwritten = stream.iter().take_while(|&&byte| byte == 0).count();
    stream.drain(..unwritten);
    let mut starts = Vec::new();
    for index in 0..count {
        let field = FIELDS + index * RECORD_BYTES;
        let boot = u64_at(field);
        let start = u64_at(field + 8);
        let first = u32_at(field + 16) as u64;
        let last = u32_at(field + 20) as u64;
        if boot == 0 || boot > boots || start > head || first > first_bytes || last > last_bytes {
            continue;
        }
        let offset = index * record_bytes as usize;
        let before = &records[offset..offset + first as usize];
        let after = offset + first_bytes as usize;
        // An interrupted record update may leave another boot's old bytes.
        if before.is_empty() || marker(before) == Some(boot) {
            starts.push(BootStart {
                boot,
                start,
                first: before.to_vec(),
                last: records[after..after + last as usize].to_vec(),
            });
        }
    }
    starts.sort_unstable_by_key(|start| start.start);
    Ok(Log {
        boots,
        head,
        ring_bytes,
        stream,
        starts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn reads_wrapped_ring_and_preserved_boot_records_from_backup() {
        const PAGE: usize = super::super::HEADER_BYTES as usize;
        const RING: usize = 512;
        const SLOT: usize = 512;
        let fields = FIELDS + 2 * RECORD_BYTES;
        let mut bytes = vec![0; PAGE + RING + 4 * SLOT];
        let (header, data) = bytes.split_at_mut(PAGE);
        header[..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(&VERSION.to_le_bytes());
        header[16..24].copy_from_slice(&((RING + 8) as u64).to_le_bytes());
        header[24..32].copy_from_slice(&2u64.to_le_bytes());
        header[32..40].copy_from_slice(&(RING as u64).to_le_bytes());
        header[40..44].copy_from_slice(&2u32.to_le_bytes());
        header[44..48].copy_from_slice(&(SLOT as u32).to_le_bytes());
        header[48..52].copy_from_slice(&(SLOT as u32).to_le_bytes());
        let first = b"=== STELA BOOT 1 === version=0.6.1\nstartup\n";
        let last = b"stopped\n";
        header[56..64].copy_from_slice(&1u64.to_le_bytes());
        header[72..76].copy_from_slice(&(first.len() as u32).to_le_bytes());
        header[76..80].copy_from_slice(&(last.len() as u32).to_le_bytes());
        header[80..88].copy_from_slice(&2u64.to_le_bytes());
        header[88..96].copy_from_slice(&8u64.to_le_bytes());
        let digest = Sha256::digest(&header[..fields]);
        header[fields..fields + 32].copy_from_slice(&digest);
        header.copy_within(..PAGE - COPY_AT, COPY_AT);
        data[..8].copy_from_slice(b"last log");
        data[8..RING].fill(b'a');
        data[RING..RING + first.len()].copy_from_slice(first);
        data[RING + SLOT..RING + SLOT + last.len()].copy_from_slice(last);

        let log = super::super::read(&bytes).unwrap();
        assert_eq!(log.boots, 2);
        assert_eq!(log.overwritten(), 8);
        assert_eq!(&log.stream[..RING - 8], vec![b'a'; RING - 8]);
        assert_eq!(&log.stream[RING - 8..], b"last log");
        assert_eq!(log.starts[0].first, first);
        assert_eq!(log.starts[0].last, last);
        assert_eq!(log.starts[1].boot, 2);

        bytes[0] = 0;
        assert!(super::super::has_magic(&bytes[..PAGE]));
        assert_eq!(super::super::read(&bytes).unwrap(), log);
        bytes[COPY_AT + fields] ^= 1;
        assert_eq!(super::super::read(&bytes), Err(HeaderError::NotALog));
    }
}
