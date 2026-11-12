// SPDX-License-Identifier: GPL-2.0-or-later
use crate::pc::fat32::{Fat32, FatError, tests::Volume};
use stela_contracts::bytes::{le16, le32};

const ROOT: u32 = 2;
const DIRECTORY: u8 = 0x10;
const ARCHIVE: u8 = 0x20;

fn volume() -> Volume {
    let mut volume = Volume::new();
    volume.disk.write(21, &[0xf8]);
    volume.set(0, 0x0fff_fff8);
    volume.set(1, 0x0fff_ffff);
    volume
}

fn directory(volume: &mut Volume, parent: u32, slot: u64, name: &[u8; 11]) -> u32 {
    let cluster = volume.directory(parent, slot, name);
    let dotdot = if parent == ROOT { 0 } else { parent };
    volume.entry(cluster, 0, b".          ", DIRECTORY, cluster, 0);
    volume.entry(cluster, 1, b"..         ", DIRECTORY, dotdot, 0);
    cluster
}

fn verify(volume: &mut Volume) -> Result<(), FatError> {
    Fat32::open(&mut volume.disk).unwrap().verify_allocation()
}

#[test]
fn allocation_proof_accepts_nested_and_fragmented_owners() {
    let mut volume = volume();
    let tree = directory(&mut volume, ROOT, 0, b"TREE       ");
    let leaf = directory(&mut volume, tree, 2, b"LEAF       ");
    let clusters = volume.allocate(3 * 512);
    volume.file_in(
        leaf,
        2,
        b"OWNED   BIN",
        &[0xa5; 700],
        &[clusters[0], clusters[2]],
    );
    volume.file_in(tree, 3, b"OTHER   BIN", b"other", &clusters[1..2]);
    volume.file(leaf, 3, b"EMPTY   BIN", b"");

    let mut fat = Fat32::open(&mut volume.disk).unwrap();
    assert_eq!(fat.verify_allocation(), Ok(()));
    let file = fat.file("TREE/LEAF/OWNED.BIN").unwrap();
    assert_eq!(file.bytes, 700);
    assert_eq!(file.extents.len(), 2);
}

#[test]
fn allocation_proof_rejects_crosslinked_files_in_different_directories() {
    let mut volume = volume();
    let store = directory(&mut volume, ROOT, 0, b"STORE      ");
    let unrelated = directory(&mut volume, ROOT, 1, b"UNRELATE   ");
    let clusters = volume.allocate(700);
    volume.file_in(store, 2, b"SLOT    SWP", &[0xa5; 700], &clusters);
    assert_eq!(verify(&mut volume), Ok(()));

    // Each file still has exactly the number of clusters its size requires.
    volume.entry(unrelated, 2, b"ALIEN   BIN", ARCHIVE, clusters[1], 1);
    assert!(matches!(verify(&mut volume), Err(FatError::NotFat32(_))));
}

#[test]
fn allocation_proof_rejects_a_directory_owned_as_file_data() {
    let mut volume = volume();
    let owned = directory(&mut volume, ROOT, 0, b"OWNED      ");
    let unrelated = directory(&mut volume, ROOT, 1, b"UNRELATE   ");
    assert_eq!(verify(&mut volume), Ok(()));

    volume.entry(unrelated, 2, b"ALIAS   BIN", ARCHIVE, owned, 1);
    assert!(matches!(verify(&mut volume), Err(FatError::NotFat32(_))));
}

#[test]
fn allocation_proof_rejects_a_different_fat_mirror() {
    let mut volume = volume();
    let cluster = volume.allocate(1)[0];
    volume.file_in(ROOT, 0, b"OWNED   BIN", b"x", &[cluster]);
    assert_eq!(verify(&mut volume), Ok(()));

    let boot = &volume.disk.sectors[&0];
    let sector_bytes = u64::from(le16(boot, 11));
    let backup = (u64::from(le16(boot, 14)) + u64::from(le32(boot, 36))) * sector_bytes;
    volume
        .disk
        .write(backup + u64::from(cluster) * 4, &0u32.to_le_bytes());
    assert!(matches!(verify(&mut volume), Err(FatError::NotFat32(_))));
}

#[test]
fn allocation_proof_rejects_an_unreferenced_allocated_chain() {
    let mut volume = volume();
    volume.file(ROOT, 0, b"OWNED   BIN", b"x");
    assert_eq!(verify(&mut volume), Ok(()));

    let orphan = volume.allocate(2 * 512);
    volume.link(&orphan);
    assert!(matches!(verify(&mut volume), Err(FatError::NotFat32(_))));
}
