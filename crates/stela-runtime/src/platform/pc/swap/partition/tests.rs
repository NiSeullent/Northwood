// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::pc::swap::{MIN_STORE_BYTES, State};
use alloc::vec;
use stela_drivers::block::RamDisk;
use stela_formats::payload::gpt::encode;

const MIB: u64 = 1 << 20;

fn guid(value: u8) -> Guid {
    Guid::from_entropy([value; 16])
}

fn existing(first_lba: u64, last_lba: u64, id: u8) -> Partition {
    Partition {
        type_guid: Guid::ESP,
        unique_guid: guid(id),
        first_lba,
        last_lba,
        attributes: 0,
        name: "existing".into(),
    }
}

#[test]
fn fits_an_aligned_gap_on_512_and_4096_byte_devices() {
    for block in [512, 4096] {
        let per_mib = MIB / block;
        let table = Table {
            disk_guid: guid(1),
            partitions: vec![
                existing(10 * per_mib, 13 * per_mib - 1, 2),
                existing(per_mib, 3 * per_mib - 1, 3),
            ],
        };
        let plan = plan(&table, 34..=15 * per_mib - 1, block, 4 * MIB, guid(4)).unwrap();
        assert_eq!(plan.partition.first_lba, 3 * per_mib);
        assert_eq!(plan.partition.last_lba, 7 * per_mib - 1);
        assert_eq!(plan.partition.type_guid, NORTHWOOD_SWAP_TYPE);
        assert_eq!(plan.header.store_uuid(), guid(4));
        assert_eq!(plan.header.state(), State::Idle);
    }
}

#[test]
fn refuses_unusable_gaps_and_overlapping_tables() {
    let mut table = Table {
        disk_guid: guid(1),
        partitions: vec![existing(256, 1023, 2)],
    };
    assert!(matches!(
        plan(&table, 34..=1535, 4096, 4 * MIB, guid(3)),
        Err(Error::NoSpace)
    ));
    assert!(matches!(
        plan(&table, 34..=4095, 4096, 4 * MIB, guid(2)),
        Err(Error::Existing)
    ));
    table.partitions.push(existing(800, 1100, 3));
    assert!(matches!(
        plan(&table, 34..=4095, 4096, 4 * MIB, guid(4)),
        Err(Error::Geometry)
    ));
}

#[test]
fn inspects_only_an_exact_typed_and_formatted_partition() {
    let mut disk = RamDisk::new("swap test", 2048, 4096);
    let geometry = gpt::geometry(disk.identity());
    let mut table = Table {
        disk_guid: guid(1),
        partitions: vec![existing(10, 255, 2)],
    };
    let plan = plan(
        &table,
        geometry.first_usable()..=geometry.last_usable(),
        geometry.block_bytes,
        MIN_STORE_BYTES,
        guid(3),
    )
    .unwrap();
    let first = plan.partition.first_lba;
    table.partitions.push(plan.partition.clone());
    for write in encode(geometry, &table).unwrap() {
        disk.write(write.lba, &write.bytes).unwrap();
    }
    assert!(matches!(
        inspect(&mut disk, Some(guid(2))),
        Err(Error::WrongType)
    ));
    assert!(matches!(
        inspect(&mut disk, Some(guid(4))),
        Err(Error::Missing)
    ));
    assert!(matches!(
        inspect(&mut disk, None),
        Err(Error::Format(FormatError::Magic))
    ));

    let wrong_uuid = Header::new(MIN_STORE_BYTES, guid(4)).unwrap();
    disk.write(first, &wrong_uuid.encode()).unwrap();
    assert!(matches!(
        inspect(&mut disk, None),
        Err(Error::Format(FormatError::StoreUuid))
    ));
    disk.write(first, &plan.header.encode()).unwrap();
    let before = disk.bytes.clone();
    assert_eq!(inspect(&mut disk, None).unwrap().0, plan.partition);
    assert_eq!(inspect(&mut disk, Some(guid(3))).unwrap().1, plan.header);
    assert!(disk.bytes == before);
}
