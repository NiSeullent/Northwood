//! Durable translation-pack writes through an owned block-device adapter,
//! and the part of a pack a boot reads when its RAM holds less than all of
//! it.
#[cfg(any(test, target_arch = "x86_64"))]
use alloc::vec::Vec;
use sha2::{Digest as _, Sha256};
#[cfg(any(test, target_arch = "x86_64"))]
use stela_engine::translations::pack::Layout;
use stela_engine::translations::{
    Digest,
    pack::{HEADER_BYTES, PackError, PackIndex},
};

/// Why a pack was not written in place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteError {
    /// The file holds fewer bytes than the pack.
    Small { file: u64, pack: u64 },
    /// An extent does not start on a device block.
    Misaligned,
    #[cfg(target_arch = "x86_64")]
    Device(stela_drivers::block::BlockError),
}

/// The bytes written to the device at a time.
#[cfg(target_arch = "x86_64")]
const TRANSFER_BYTES: usize = 1 << 20;

/// Write `layout` over the file whose device extents (its cluster chain,
/// cut to its size) are `extents`, whole device blocks at a time, with
/// zeros past the pack inside the file's last block; then make it durable.
/// A layout planned over the pack the file holds ([`Layout::changed`])
/// writes only the blocks of its changed ranges: its records first, made
/// durable before the index that names them, and nothing at all when
/// nothing changed. The file is never grown, and a pack larger than it is
/// refused before anything is written.
#[cfg(target_arch = "x86_64")]
pub fn write_in_place(
    device: &mut dyn stela_drivers::block::BlockDevice,
    extents: &[stela_formats::payload::pieces::Extent],
    layout: &Layout,
) -> Result<(), WriteError> {
    let block = u64::from(device.identity().block_size);
    let file: u64 = extents.iter().map(|extent| extent.bytes).sum();
    let pack = layout.bytes();
    if pack > file {
        return Err(WriteError::Small { file, pack });
    }
    if extents.iter().any(|extent| extent.offset % block != 0) {
        return Err(WriteError::Misaligned);
    }
    let mut buffer = Vec::with_capacity(TRANSFER_BYTES);
    let Some(changed) = layout.changed() else {
        write_range(device, extents, layout, 0..pack, block, &mut buffer)?;
        return device.flush().map_err(WriteError::Device);
    };
    let Some((index, records)) = changed.split_first() else {
        return Ok(());
    };
    for range in records {
        write_range(device, extents, layout, range.clone(), block, &mut buffer)?;
    }
    if !records.is_empty() {
        device.flush().map_err(WriteError::Device)?;
    }
    write_range(device, extents, layout, index.clone(), block, &mut buffer)?;
    device.flush().map_err(WriteError::Device)
}

/// Write the device blocks holding the pack's bytes `range`, each whole,
/// from `layout`; the file's extents map pack offsets to the device.
#[cfg(target_arch = "x86_64")]
fn write_range(
    device: &mut dyn stela_drivers::block::BlockDevice,
    extents: &[stela_formats::payload::pieces::Extent],
    layout: &Layout,
    range: core::ops::Range<u64>,
    block: u64,
    buffer: &mut Vec<u8>,
) -> Result<(), WriteError> {
    let file: u64 = extents.iter().map(|extent| extent.bytes).sum();
    let mut at = range.start - range.start % block;
    let end = range
        .end
        .next_multiple_of(block)
        .min(file.next_multiple_of(block));
    let mut extent_start = 0;
    for extent in extents {
        let extent_end = extent_start + extent.bytes;
        while at < end && at < extent_end {
            if at < extent_start {
                at = extent_start;
                continue;
            }
            let take =
                (extent_end.next_multiple_of(block).min(end) - at).min(TRANSFER_BYTES as u64);
            buffer.clear();
            buffer.resize(take as usize, 0);
            layout.read_at(at, buffer);
            // Bytes past the pack and inside the file's last block are
            // zeros; those of the next extent's first block do not belong
            // to this one.
            let inside = (extent_end - at).min(take) as usize;
            buffer[inside..].fill(0);
            buffer.resize(buffer.len().next_multiple_of(block as usize), 0);
            device
                .write((extent.offset + (at - extent_start)) / block, buffer)
                .map_err(WriteError::Device)?;
            at += take;
        }
        extent_start = extent_end;
    }
    Ok(())
}

/// A pack index entry's bytes: page key, offset from the pack's start and
/// length.
const INDEX_ENTRY_BYTES: usize = 32 + 8 + 8;
/// The SHA-256 of the header and index that follows the index.
const CHECKSUM_BYTES: usize = 32;
/// Where the header holds the record count: after the magic and schema.
const COUNT_AT: usize = 12;

/// What a reader that holds at most a budget's bytes reads of a pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Within {
    /// The pack's first bytes to read: its index and every record that ends
    /// within the budget.
    pub length: u64,
    /// The full pack's size as declared by the verified index.
    pub pack_length: u64,
    /// The records that end within the budget, and the others.
    pub records: usize,
    pub omitted: usize,
}

/// What to read of the pack whose header, index and index checksum are
/// `prefix` (`pack::prefix_bytes`) to hold at most `limit` bytes of it. A
/// pack's records lie anywhere after its index, so its first bytes up to the
/// last end within `limit` hold every record that ends within it: a usable
/// pack once [`trim`] names only those. `Err` when the index fails the
/// checks [`PackIndex::parse`] makes of it; a [`Within::length`] past `limit` when
/// the index alone passes it.
pub fn within(prefix: &[u8], translator: &Digest, limit: u64) -> Result<Within, PackError> {
    let index = PackIndex::parse(prefix, translator)?;
    let mut within = Within {
        length: index.prefix_bytes() as u64,
        pack_length: index.length(),
        records: 0,
        omitted: 0,
    };
    for (_, range) in index.iter() {
        if range.end <= limit {
            within.records += 1;
            within.length = within.length.max(range.end);
        } else {
            within.omitted += 1;
        }
    }
    Ok(within)
}

/// Make the index at the start of `pack`, a pack's first bytes whose index
/// [`within`] checked, name only the records that lie inside them, with its
/// count and checksum to match and zeros for the rest of its old room. The
/// records keep their offsets, so a layout planned over the result keeps
/// them where the file holds them and writes only what changed. Returns
/// the records it names.
pub fn trim(pack: &mut [u8]) -> usize {
    let count = u32::from_le_bytes(
        pack[COUNT_AT..COUNT_AT + 4]
            .try_into()
            .expect("a count word"),
    ) as usize;
    let old_end = HEADER_BYTES + count * INDEX_ENTRY_BYTES + CHECKSUM_BYTES;
    assert!(old_end <= pack.len(), "a checked index inside the pack");
    let mut kept = 0;
    for index in 0..count {
        let (offset, bytes) = entry(pack, index);
        if offset
            .checked_add(bytes)
            .is_some_and(|end| end <= pack.len() as u64)
        {
            let from = HEADER_BYTES + index * INDEX_ENTRY_BYTES;
            pack.copy_within(
                from..from + INDEX_ENTRY_BYTES,
                HEADER_BYTES + kept * INDEX_ENTRY_BYTES,
            );
            kept += 1;
        }
    }
    pack[COUNT_AT..COUNT_AT + 4].copy_from_slice(&(kept as u32).to_le_bytes());
    let index_end = HEADER_BYTES + kept * INDEX_ENTRY_BYTES;
    let checksum = Sha256::digest(&pack[..index_end]);
    pack[index_end..index_end + CHECKSUM_BYTES].copy_from_slice(&checksum);
    pack[index_end + CHECKSUM_BYTES..old_end].fill(0);
    kept
}

/// Index entry `index`'s record offset and length.
fn entry(pack: &[u8], index: usize) -> (u64, u64) {
    let at = HEADER_BYTES + index * INDEX_ENTRY_BYTES + 32;
    let field = |at: usize| u64::from_le_bytes(pack[at..at + 8].try_into().expect("a field"));
    (field(at), field(at + 8))
}

#[cfg(test)]
mod tests {
    use super::*;
    use stela_engine::translations::pack::{self, Pack, Piece};
    #[test]
    fn a_pack_is_written_over_its_files_extents_and_never_past_them() {
        use stela_drivers::block::RamDisk;
        use stela_formats::payload::pieces::Extent;
        const BLOCK: u32 = 512;
        let translator = Digest::of(b"translator");
        let records: Vec<(Digest, Piece)> = (0u8..40)
            .map(|index| (Digest::of(&[index]), Piece::New(vec![index; 300])))
            .collect();
        let layout = Layout::new(&translator, records);
        let pack = layout.bytes();
        let mut disk = RamDisk::new("pack", 128, BLOCK);
        disk.identity.features.write_cache = true;
        disk.bytes.fill(0xee);
        // Two clusters of 8 blocks, out of order, the file cut to its size.
        let cluster = 8 * u64::from(BLOCK);
        let extents = [
            Extent {
                offset: 64 * u64::from(BLOCK),
                bytes: cluster,
            },
            Extent {
                offset: 16 * u64::from(BLOCK),
                bytes: pack - cluster + 100,
            },
        ];
        write_in_place(&mut disk, &extents, &layout).unwrap();
        let mut read = disk.bytes[64 * BLOCK as usize..][..cluster as usize].to_vec();
        read.extend(&disk.bytes[16 * BLOCK as usize..][..(pack - cluster) as usize]);
        assert_eq!(read, layout.to_vec());
        assert!(Pack::parse(&read, &translator).is_ok());
        let after = 16 * BLOCK as usize + (pack - cluster) as usize;
        let padded = after.next_multiple_of(BLOCK as usize);
        assert!(disk.bytes[after..padded].iter().all(|&byte| byte == 0));
        assert!(
            disk.bytes[padded..64 * BLOCK as usize]
                .iter()
                .all(|&byte| byte == 0xee)
        );
        assert_eq!(disk.flushes, 1);
        let small = [Extent {
            offset: 0,
            bytes: pack - 1,
        }];
        assert_eq!(
            write_in_place(&mut disk, &small, &layout),
            Err(WriteError::Small {
                file: pack - 1,
                pack
            })
        );
        let misaligned = [Extent {
            offset: 1,
            bytes: pack,
        }];
        assert_eq!(
            write_in_place(&mut disk, &misaligned, &layout),
            Err(WriteError::Misaligned)
        );
    }

    #[test]
    fn an_unchanged_pack_is_not_written_and_a_range_writes_only_its_blocks() {
        use stela_drivers::block::RamDisk;
        use stela_engine::translations::{Record, pack::Store};
        use stela_formats::payload::pieces::Extent;
        const BLOCK: u32 = 512;
        let translator = Digest::of(b"translator");
        let records = || -> Vec<(Digest, Piece)> {
            (0u8..40)
                .map(|index| {
                    let key = Digest::of(&[index]);
                    (key, Piece::New(Record::new(translator, key).encode()))
                })
                .collect()
        };
        let bytes = Layout::new(&translator, records()).to_vec();
        assert!(bytes.len() as u64 > 8 * 512 + 512);
        let mut disk = RamDisk::new("pack", 128, BLOCK);
        disk.identity.features.write_cache = true;
        disk.bytes.fill(0xee);
        let cluster = 8 * u64::from(BLOCK);
        let extents = [
            Extent {
                offset: 64 * u64::from(BLOCK),
                bytes: cluster,
            },
            Extent {
                offset: 16 * u64::from(BLOCK),
                bytes: 4 * cluster,
            },
        ];
        // A boot that changed nothing plans the pack it read, unwritten.
        let planned = Store::new(translator, Some(&bytes))
            .0
            .plan(u64::MAX, || true);
        assert_eq!(planned.layout.changed(), Some(&[][..]));
        write_in_place(&mut disk, &extents, &planned.layout).unwrap();
        assert_eq!(disk.flushes, 0);
        assert!(disk.bytes.iter().all(|&byte| byte == 0xee));
        // A range across the extents' seam writes its blocks alone.
        let layout = Layout::new(&translator, records());
        let mut buffer = Vec::new();
        let start = cluster - 100;
        write_range(
            &mut disk,
            &extents,
            &layout,
            start..cluster + 10,
            u64::from(BLOCK),
            &mut buffer,
        )
        .unwrap();
        let block = BLOCK as usize;
        let first = 64 * block + (cluster as usize - block);
        let second = 16 * block;
        let written: Vec<usize> = (0..disk.bytes.len())
            .filter(|&at| disk.bytes[at] != 0xee)
            .collect();
        assert!(
            written
                .iter()
                .all(|&at| (first..first + block).contains(&at)
                    || (second..second + block).contains(&at))
        );
        let mut expected = vec![0; 2 * block];
        layout.read_at(cluster - u64::from(BLOCK), &mut expected);
        assert_eq!(disk.bytes[first..first + block], expected[..block]);
        assert_eq!(disk.bytes[second..second + block], expected[block..]);
    }

    /// A pack of 40 valid, empty records of `translator`, in key order.
    fn sample(translator: Digest) -> Vec<u8> {
        use stela_engine::translations::Record;
        let records = (0u8..40)
            .map(|index| {
                let key = Digest::of(&[index]);
                (key, Piece::New(Record::new(translator, key).encode()))
            })
            .collect();
        Layout::new(&translator, records).to_vec()
    }

    #[test]
    fn a_pack_larger_than_the_budget_is_read_as_its_records_within_it() {
        use stela_engine::translations::pack::Store;
        let translator = Digest::of(b"translator");
        let bytes = sample(translator);
        let full = Pack::parse(&bytes, &translator).unwrap();
        let header = &bytes[..HEADER_BYTES];
        let prefix = &bytes[..pack::prefix_bytes(header, &translator).unwrap()];
        // The whole pack within a budget its size or more.
        let whole = within(prefix, &translator, bytes.len() as u64).unwrap();
        assert_eq!(
            whole,
            Within {
                length: bytes.len() as u64,
                pack_length: bytes.len() as u64,
                records: 40,
                omitted: 0
            }
        );
        // A budget that cuts the records: the bytes up to the last record
        // ending within it, and the rest left out.
        let limit = (prefix.len() + (bytes.len() - prefix.len()) / 2) as u64;
        let part = within(prefix, &translator, limit).unwrap();
        assert!(part.length <= limit && part.length > prefix.len() as u64);
        assert_eq!(part.pack_length, bytes.len() as u64);
        assert!(part.records > 0 && part.omitted > 0);
        assert_eq!(part.records + part.omitted, 40);
        let mut read = bytes[..part.length as usize].to_vec();
        // Untrimmed, the index names records past the bytes read.
        assert!(Pack::parse(&read, &translator).is_err());
        assert_eq!(trim(&mut read), part.records);
        let trimmed = Pack::parse(&read, &translator).unwrap();
        assert_eq!(trimmed.records(), part.records);
        assert_eq!(trimmed.bytes() as u64, part.length);
        // Each record read is the pack's, at its offset; the others are
        // absent, as a key the pack never held.
        for (key, record) in full.iter() {
            match trimmed.get(&key) {
                Some(kept) => {
                    assert_eq!(kept, record);
                    let at = |slice: &[u8], base: &[u8]| {
                        slice.as_ptr() as usize - base.as_ptr() as usize
                    };
                    assert_eq!(at(kept, &read), at(record, &bytes));
                }
                None => assert!(at_least_past(record, &bytes, part.length)),
            }
        }
        // A boot over the part that changed nothing writes nothing: the
        // file keeps the whole pack for a PC with more RAM.
        let planned = Store::new(translator, Some(&read)).0.plan(limit, || true);
        assert_eq!(planned.layout.changed(), Some(&[][..]));
        assert_eq!(planned.layout.records(), part.records);
        // A budget the index alone passes reads nothing.
        let none = within(prefix, &translator, prefix.len() as u64 - 1).unwrap();
        assert!(none.length > prefix.len() as u64 - 1);
        assert_eq!(none.records, 0);
        // A damaged index is refused as the engine would.
        let mut damaged = prefix.to_vec();
        damaged[HEADER_BYTES] ^= 1;
        assert_eq!(
            within(&damaged, &translator, limit),
            Err(PackError::Checksum)
        );
    }

    /// Whether `record`, a slice of `pack`, ends past `length`.
    fn at_least_past(record: &[u8], pack: &[u8], length: u64) -> bool {
        let start = record.as_ptr() as usize - pack.as_ptr() as usize;
        (start + record.len()) as u64 > length
    }
}
