// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::pc::gpt::Table;
use alloc::{vec, vec::Vec};
use stela_drivers::block::RamDisk;
use stela_formats::payload::gpt::encode;

fn guid(value: u8) -> Guid {
    Guid::from_entropy([value; 16])
}

fn partition(type_guid: Guid, id: u8, first_lba: u64, last_lba: u64) -> Partition {
    Partition {
        type_guid,
        unique_guid: guid(id),
        first_lba,
        last_lba,
        attributes: 0,
        name: "fixture".into(),
    }
}

fn esp() -> Partition {
    partition(Guid::ESP, 2, 64, 95)
}

fn data() -> Partition {
    partition(BASIC_DATA, 3, 128, 255)
}

fn boot_esp() -> BootEsp {
    let esp = esp();
    BootEsp::new(esp.unique_guid, esp.first_lba, esp.blocks()).unwrap()
}

fn disk(id: u8, block_size: u32, partitions: Vec<Partition>) -> Box<dyn BlockDevice> {
    let mut disk = RamDisk::new("volume boot fixture", 512, block_size);
    let table = Table {
        disk_guid: guid(id),
        partitions,
    };
    for write in encode(gpt::geometry(disk.identity()), &table).unwrap() {
        disk.write(write.lba, &write.bytes).unwrap();
    }
    Box::new(disk)
}

#[test]
fn data_guid_on_an_unrelated_disk_does_not_redirect_selection() {
    for block_size in [512, 4096] {
        let data = data();
        let same_guid_elsewhere = partition(BASIC_DATA, 3, 320, 383);
        assert_eq!(same_guid_elsewhere.unique_guid, data.unique_guid);
        let mut disks = [
            disk(10, block_size, vec![same_guid_elsewhere]),
            disk(11, block_size, vec![esp(), data.clone()]),
        ];
        assert_eq!(boot_data(&mut disks, boot_esp()), Ok((1, data)));
    }
}

#[test]
fn duplicate_esp_guid_refuses_across_disks_and_within_one_disk() {
    let mut disks = [
        disk(10, 512, vec![esp()]),
        disk(11, 512, vec![esp(), data()]),
    ];
    assert_eq!(
        boot_data(&mut disks, boot_esp()),
        Err(BootDataError::AmbiguousEsp)
    );

    let duplicate = partition(Guid::ESP, 2, 320, 383);
    let mut disks = [disk(10, 512, vec![esp(), data(), duplicate])];
    assert_eq!(
        boot_data(&mut disks, boot_esp()),
        Err(BootDataError::AmbiguousEsp)
    );
}

#[test]
fn firmware_esp_type_start_and_length_must_match() {
    for wrong_esp in [
        partition(guid(7), 2, 64, 95),
        partition(Guid::ESP, 2, 65, 96),
        partition(Guid::ESP, 2, 64, 96),
    ] {
        let mut disks = [disk(10, 512, vec![wrong_esp, data()])];
        assert_eq!(
            boot_data(&mut disks, boot_esp()),
            Err(BootDataError::InvalidEsp)
        );
    }
}

#[test]
fn data_requires_one_basic_partition_on_the_boot_disk() {
    let mut disks = [disk(10, 512, vec![esp()]), disk(11, 512, vec![data()])];
    assert_eq!(
        boot_data(&mut disks, boot_esp()),
        Err(BootDataError::MissingData)
    );

    let second_data = partition(BASIC_DATA, 4, 320, 383);
    let mut disks = [disk(10, 512, vec![esp(), data(), second_data])];
    assert_eq!(
        boot_data(&mut disks, boot_esp()),
        Err(BootDataError::AmbiguousData)
    );
}

#[test]
fn data_requires_a_disjoint_extent_and_nonzero_unique_guid() {
    let overlapping = partition(guid(7), 4, data().last_lba, 383);
    let duplicate_guid = partition(guid(7), 3, 320, 383);
    for invalid_neighbor in [overlapping, duplicate_guid] {
        let mut disks = [disk(10, 512, vec![esp(), data(), invalid_neighbor])];
        assert_eq!(
            boot_data(&mut disks, boot_esp()),
            Err(BootDataError::InvalidData)
        );
    }

    let mut zero_guid = data();
    zero_guid.unique_guid = Guid::default();
    let mut disks = [disk(10, 512, vec![esp(), zero_guid])];
    assert_eq!(
        boot_data(&mut disks, boot_esp()),
        Err(BootDataError::InvalidData)
    );
}
