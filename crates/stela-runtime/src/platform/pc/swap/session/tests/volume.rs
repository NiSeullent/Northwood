// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::pc::{
    fat32::tests::Volume,
    gpt::Partition,
    swap::{DEFAULT_VOLUME_MAX_BYTES, HEADER_BYTES, MIN_STORE_BYTES, volume::BASIC_DATA},
};
use core::num::NonZeroUsize;
use stela_contracts::bytes::{le16, le32};

const BLOCK_BYTES: usize = 512;
const DATA_LBA: u64 = 2048;

struct FragmentedStore {
    partition: Guid,
    header: [u64; 2],
    slot: [u64; 2],
    outside: u64,
}

fn fragmented_disk(fua: bool) -> (DiskFixture, FragmentedStore) {
    let bytes = MIN_STORE_BYTES + SLOT_BYTES;
    let header = Header::new(bytes, uuid()).unwrap();
    let mut volume = Volume::new();
    volume.disk.write(21, &[0xf8]);
    volume.set(0, 0x0fff_fff8);
    volume.set(1, 0x0fff_ffff);

    let stela = volume.directory(2, 0, b"STELA      ");
    let swap = volume.directory(stela, 2, b"SWAP       ");
    for (directory, parent) in [(stela, 0), (swap, stela)] {
        volume.entry(directory, 0, b".          ", 0x10, directory, 0);
        volume.entry(directory, 1, b"..         ", 0x10, parent, 0);
    }
    volume.file(
        swap,
        2,
        b"SWAP    CFG",
        format!("bytes {bytes}\n").as_bytes(),
    );
    let mut clusters = volume.allocate(bytes);
    let header_blocks = HEADER_BYTES / BLOCK_BYTES;
    clusters[..header_blocks].rotate_left(header_blocks / 2);
    let slot_start = header.layout().data_offset() as usize / BLOCK_BYTES;
    let slot_blocks = PAGE_BYTES / BLOCK_BYTES;
    let short_run = 4096 / BLOCK_BYTES;
    clusters[slot_start..slot_start + slot_blocks].rotate_right(short_run);
    volume.file_in(swap, 3, b"0000    SWP", &header.encode(), &clusters);
    volume.entry(swap, 3, b"0000    SWP", 0x20, clusters[0], bytes as u32);

    let boot = &volume.disk.sectors[&0];
    let first_cluster = u64::from(le16(boot, 14)) + u64::from(boot[16]) * u64::from(le32(boot, 36));
    let lba = |cluster| DATA_LBA + first_cluster + u64::from(cluster - 2);
    let store = FragmentedStore {
        partition: Guid::from_entropy([0x43; 16]),
        header: [lba(clusters[0]), lba(clusters[header_blocks / 2])],
        slot: [
            lba(clusters[slot_start]),
            lba(clusters[slot_start + short_run]),
        ],
        outside: DATA_LBA - 1,
    };

    let blocks = volume.disk.bytes / BLOCK_BYTES as u64;
    let mut disk = RamDisk::new(
        "private fragmented FAT32 fixture",
        DATA_LBA + blocks + 64,
        BLOCK_BYTES as u32,
    );
    let table = Table {
        disk_guid: Guid::from_entropy([0x61; 16]),
        partitions: vec![Partition {
            type_guid: BASIC_DATA,
            unique_guid: store.partition,
            first_lba: DATA_LBA,
            last_lba: DATA_LBA + blocks - 1,
            attributes: 0,
            name: "STELA".into(),
        }],
    };
    for write in encode(geometry(disk.identity()), &table).unwrap() {
        disk.write(write.lba, &write.bytes).unwrap();
    }
    for (offset, contents) in volume.disk.sectors {
        disk.write(DATA_LBA + offset, &contents).unwrap();
    }
    disk.write(store.outside, &[0xc3; BLOCK_BYTES]).unwrap();
    disk.identity.features.max_transfer = 8192;
    disk.identity.features.max_read_transfer = NonZeroUsize::new(1024);
    disk.identity.features.write_cache = true;
    disk.identity.features.fua = fua;
    let disk = DiskFixture {
        changed: disk.identity.clone(),
        disk,
        trace: Rc::new(RefCell::new(Trace::default())),
    };
    (disk, store)
}

fn open_volume(disk: DiskFixture, store: &FragmentedStore) -> Session<DiskFixture> {
    Session::open_volume(
        disk,
        store.partition,
        uuid(),
        DEFAULT_VOLUME_MAX_BYTES,
        &mut Generator::from_seed([0x81; 32]),
    )
    .unwrap()
}

fn physical(disk: &[u8], lba: u64, bytes: usize) -> &[u8] {
    let offset = lba as usize * BLOCK_BYTES;
    &disk[offset..offset + bytes]
}

#[test]
fn fragmented_fat32_header_and_slot_follow_logical_order_and_durability() {
    for fua in [false, true] {
        let (disk, store) = fragmented_disk(fua);
        let trace = disk.trace.clone();
        let mut session = open_volume(disk, &store);
        assert_eq!(session.header().layout().slot_count(), 2);

        let page = core::array::from_fn(|offset| (offset % 251) as u8);
        let proof = session.write_slot(0, 7, &page).unwrap();
        let mut output = [0; PAGE_BYTES];
        session.read_slot(0, 7, &proof, &mut output).unwrap();
        assert_eq!(output, page);

        let disk = &session.disk.device().disk.bytes;
        let header = session.header().encode();
        let half = HEADER_BYTES / 2;
        assert_eq!(physical(disk, store.header[0], half), &header[..half]);
        assert_eq!(physical(disk, store.header[1], half), &header[half..]);
        assert_ne!(physical(disk, store.slot[0], 4096), &page[..4096]);
        assert_ne!(physical(disk, store.slot[1], 12_288), &page[4096..]);
        assert_eq!(
            physical(disk, store.outside, BLOCK_BYTES),
            &[0xc3; BLOCK_BYTES]
        );

        let trace = trace.borrow();
        assert_eq!(
            trace.writes,
            [
                (store.header[0], half, fua),
                (store.header[1], half, fua),
                (store.slot[0], 4096, fua),
                (store.slot[1], 8192, fua),
                (store.slot[1] + 16, 4096, fua),
            ]
        );
        assert_eq!(trace.flushes, if fua { 0 } else { 2 });
    }
}

#[test]
fn fragmented_fat32_failures_keep_slot_proofs_and_read_output_private() {
    let (disk, store) = fragmented_disk(false);
    let trace = disk.trace.clone();
    let mut session = open_volume(disk, &store);
    let page = [0x47; PAGE_BYTES];
    trace.borrow_mut().writes.clear();
    trace.borrow_mut().write_before_failure = Some(1);
    assert!(matches!(
        session.write_slot(0, 7, &page),
        Err(Error::Device(BlockError::Timeout))
    ));
    assert_eq!(
        trace.borrow().writes,
        [(store.slot[0], 4096, false), (store.slot[1], 8192, false)]
    );
    assert_eq!(trace.borrow().flushes, 1);

    trace.borrow_mut().writes.clear();
    trace.borrow_mut().fail_flush = true;
    assert!(matches!(
        session.write_slot(0, 7, &page),
        Err(Error::Device(BlockError::Timeout))
    ));
    assert_eq!(trace.borrow().writes.len(), 3);
    assert_eq!(trace.borrow().flushes, 2);

    trace.borrow_mut().fail_flush = false;
    let proof = session.write_slot(0, 7, &page).unwrap();
    assert_eq!(proof.counter, 3);
    // Four 1 KiB reads complete the first physical extent. Fail the next one.
    trace.borrow_mut().read_before_failure = Some(4);
    let mut output = [0xa3; PAGE_BYTES];
    assert!(matches!(
        session.read_slot(0, 7, &proof, &mut output),
        Err(Error::Device(BlockError::Timeout))
    ));
    assert_eq!(output, [0xa3; PAGE_BYTES]);
    session.read_slot(0, 7, &proof, &mut output).unwrap();
    assert_eq!(output, page);
}
