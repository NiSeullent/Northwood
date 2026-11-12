// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::pc::gpt::NORTHWOOD_SWAP_NAME;

fn uuid() -> Guid {
    "E3B6C15B-1369-4C2D-9121-51D581FD14C3".parse().unwrap()
}

fn recalculate_checksum(bytes: &mut [u8; HEADER_BYTES]) {
    let checksum = crc32_ieee(&bytes[..CHECKSUM_OFFSET]);
    bytes[CHECKSUM_OFFSET..].copy_from_slice(&checksum.to_le_bytes());
}

#[test]
fn v1_header_uses_the_documented_wire_format() {
    let header = Header::new(DEFAULT_STORE_BYTES, uuid()).unwrap();
    assert_eq!(header.state(), State::Idle);
    assert_eq!(header.generation(), 0);
    assert!(!header.encrypted());
    assert_eq!(header.store_uuid(), uuid());
    assert_eq!(
        Header::decode(&header.encode(), DEFAULT_STORE_BYTES),
        Ok(header)
    );

    let active = header.begin_session(0x1122_3344_5566_7788, true).unwrap();
    let bytes = active.encode();
    assert_eq!(&bytes[..8], b"NWSWAP01");
    assert_eq!(&bytes[48..64], &uuid().0);
    // Independent IEEE CRC-32 vector for the documented byte offsets.
    assert_eq!(&bytes[CHECKSUM_OFFSET..], &[0xc8, 0xca, 0xef, 0xec]);
    assert_eq!(Header::decode(&bytes, DEFAULT_STORE_BYTES), Ok(active));
    assert_eq!(active.state(), State::InUse);
    assert!(active.encrypted());
    assert_eq!(
        active.begin_session(active.generation(), true),
        Err(Error::Generation)
    );
    assert_eq!(active.begin_session(0, false), Err(Error::Generation));
    let reboot = active.begin_session(33, false).unwrap();
    assert_eq!(reboot.generation(), 33);
    assert!(!reboot.encrypted());
    assert_eq!(reboot.layout(), header.layout());
}

#[test]
fn layout_accounts_for_metadata_and_bitmap_growth() {
    assert!(matches!(
        Layout::for_store(MIN_STORE_BYTES - 1),
        Err(Error::ShortStore { .. })
    ));
    let minimum = Layout::for_store(MIN_STORE_BYTES).unwrap();
    assert_eq!(minimum.slot_count(), 1);
    assert_eq!(minimum.used_bytes(), MIN_STORE_BYTES);
    assert_eq!(minimum.slot_offset(0), Some(2 * ALIGNMENT));
    assert_eq!(minimum.slot_offset(1), None);

    let default = Layout::for_store(DEFAULT_STORE_BYTES).unwrap();
    assert_eq!(default.bitmap_bytes(), ALIGNMENT);
    assert_eq!(default.data_offset(), 2 * ALIGNMENT);
    assert_eq!(default.slot_count(), 2_097_024);
    assert_eq!(default.used_bytes(), DEFAULT_STORE_BYTES);
    assert_eq!(
        default.slot_offset(default.slot_count() - 1),
        Some(DEFAULT_STORE_BYTES - SLOT_BYTES)
    );
    assert_eq!(default.slot_offset(u64::MAX), None);

    let boundary = 2 * ALIGNMENT + SLOTS_PER_BITMAP_BLOCK * SLOT_BYTES;
    for bytes in [boundary, boundary + SLOT_BYTES, boundary + ALIGNMENT] {
        let layout = Layout::for_store(bytes).unwrap();
        assert_eq!(layout.slot_count(), SLOTS_PER_BITMAP_BLOCK);
        assert_eq!(layout.used_bytes(), boundary);
    }
    let next = Layout::for_store(boundary + ALIGNMENT + SLOT_BYTES).unwrap();
    assert_eq!(next.slot_count(), SLOTS_PER_BITMAP_BLOCK + 1);
    assert_eq!(next.bitmap_bytes(), 2 * ALIGNMENT);
    let largest = Layout::for_store(u64::MAX).unwrap();
    assert!(u64::MAX - largest.used_bytes() < SLOT_BYTES);
}

#[test]
fn rejects_truncated_or_corrupted_headers_and_stores() {
    let header = Header::new(DEFAULT_STORE_BYTES, uuid()).unwrap();
    let bytes = header.encode();
    assert_eq!(
        Header::decode(&bytes[..HEADER_BYTES - 1], DEFAULT_STORE_BYTES),
        Err(Error::ShortHeader)
    );
    assert_eq!(
        Header::decode(&[0; HEADER_BYTES], DEFAULT_STORE_BYTES),
        Err(Error::Magic)
    );
    assert_eq!(
        Header::decode(&bytes, DEFAULT_STORE_BYTES - 1),
        Err(Error::ShortStore {
            available: DEFAULT_STORE_BYTES - 1,
            required: DEFAULT_STORE_BYTES,
        })
    );
    let mut damaged = bytes;
    damaged[48] ^= 1;
    assert_eq!(
        Header::decode(&damaged, DEFAULT_STORE_BYTES),
        Err(Error::Checksum)
    );
}

#[test]
fn valid_crc_cannot_hide_invalid_fields() {
    let bytes = Header::new(DEFAULT_STORE_BYTES, uuid()).unwrap().encode();
    let fields: &[(usize, &[u8], Error)] = &[
        (8, &2u32.to_le_bytes(), Error::Version(2)),
        (12, &512u32.to_le_bytes(), Error::Layout),
        (16, &4096u32.to_le_bytes(), Error::Layout),
        (20, &[1], Error::Reserved),
        (24, &0u64.to_le_bytes(), Error::Layout),
        (24, &u64::MAX.to_le_bytes(), Error::Layout),
        (32, &4096u64.to_le_bytes(), Error::Layout),
        (40, &(2 * ALIGNMENT - 1).to_le_bytes(), Error::Layout),
        (40, &ALIGNMENT.to_le_bytes(), Error::Layout),
        (48, &[0; 16], Error::StoreUuid),
        (72, &1u32.to_le_bytes(), Error::Generation),
        (72, &2u32.to_le_bytes(), Error::State(2)),
        (76, &3u32.to_le_bytes(), Error::Flags(3)),
        (CHECKSUM_OFFSET - 1, &[1], Error::Reserved),
    ];
    for &(offset, replacement, expected) in fields {
        let mut modified = bytes;
        modified[offset..offset + replacement.len()].copy_from_slice(replacement);
        recalculate_checksum(&mut modified);
        assert_eq!(
            Header::decode(&modified, DEFAULT_STORE_BYTES),
            Err(expected)
        );
    }
}

#[test]
fn only_the_dedicated_gpt_partition_matches_its_header() {
    assert_eq!(
        NORTHWOOD_SWAP_TYPE,
        "E9AFCDBE-E70E-40C5-8A5F-AF97B70BD75D".parse().unwrap()
    );
    let header = Header::new(DEFAULT_STORE_BYTES, uuid()).unwrap();
    let bytes = header.encode();
    for block_bytes in [512, 4096] {
        let first_lba = ALIGNMENT / block_bytes;
        let partition = Partition {
            type_guid: NORTHWOOD_SWAP_TYPE,
            unique_guid: uuid(),
            first_lba,
            last_lba: first_lba + DEFAULT_STORE_BYTES / block_bytes - 1,
            attributes: 0,
            name: NORTHWOOD_SWAP_NAME.into(),
        };
        assert_eq!(
            Header::decode_partition(&bytes, &partition, block_bytes),
            Ok(header)
        );
        assert_eq!(
            Header::decode_partition(&bytes, &partition, 520),
            Err(Error::BlockSize(520))
        );

        let mut wrong = partition.clone();
        wrong.type_guid = Guid::ESP;
        assert_eq!(
            Header::decode_partition(&bytes, &wrong, block_bytes),
            Err(Error::PartitionType)
        );
        let mut wrong = partition.clone();
        wrong.unique_guid = Guid::from_entropy([0x18; 16]);
        assert_eq!(
            Header::decode_partition(&bytes, &wrong, block_bytes),
            Err(Error::StoreUuid)
        );
        let mut wrong = partition.clone();
        wrong.last_lba -= 1;
        assert!(matches!(
            Header::decode_partition(&bytes, &wrong, block_bytes),
            Err(Error::ShortStore { .. })
        ));
        let mut wrong = partition;
        wrong.last_lba = u64::MAX;
        assert_eq!(
            Header::decode_partition(&bytes, &wrong, block_bytes),
            Err(Error::Layout)
        );
    }
}
