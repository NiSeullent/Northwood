//! The bytes that erase a whole disk into the USB layout setup expects, as
//! macOS's `diskutil eraseDisk FAT32 NAME GPT` lays it out: a GPT with a
//! 200 MiB EFI system partition and a FAT32 volume named NAME on the rest,
//! both formatted. Hosts without such a tool write these bytes to the raw
//! disk; Windows's own formatter refuses FAT32 volumes over 32 GB.
//!
//! The GPT is the shared encoder's ([`stela_formats::payload::gpt`]), and
//! the FAT32 volumes follow the shared layout of Microsoft's FAT
//! specification ([`stela_runtime::pc::fat32::format`]), with a root
//! directory holding only the volume label.
// macOS erases a USB with diskutil, so only the other hosts write this
// layout themselves.
#![cfg_attr(
    target_os = "macos",
    allow(
        dead_code,
        reason = "macOS uses diskutil; only ESP_BYTES is used outside the shared FAT32 encoder tests"
    )
)]
use anyhow::{Result, bail, ensure};
use stela_formats::payload::gpt::{self, Disk, Guid, Partition, Table};
use stela_runtime::pc::fat32::format::{
    ATTRIBUTE_VOLUME_ID, BACKUP_BOOT_SECTOR, END_OF_CHAIN, FATS, FSINFO_SECTOR, Geometry,
    MEDIA_ENTRY, ROOT_CLUSTER, volume_label,
};

/// The EFI system partition diskutil makes.
pub(super) const ESP_BYTES: u64 = 200 << 20;
/// Partitions start on 1 MiB boundaries.
const ALIGNMENT: u64 = 1 << 20;
/// The Microsoft basic data partition type of a FAT volume.
const BASIC_DATA: &str = "EBD0A0A2-B9E5-4433-87C0-68B6B72699C7";

/// One write of the layout: bytes, or zeros, at a byte offset of the disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Write {
    Bytes(u64, Vec<u8>),
    Zeros(u64, u64),
}

/// The writes that lay the disk out, in the order they go to it.
#[derive(Debug)]
pub(crate) struct Layout {
    pub(crate) writes: Vec<Write>,
}

/// Erase a `bytes`-long disk of `block_size`-byte logical blocks into an EFI
/// partition and a FAT32 volume labeled `name`. `random` supplies the disk
/// and partition GUIDs and the volume serial numbers.
pub(crate) fn layout(
    bytes: u64,
    block_size: u32,
    name: &str,
    random: &mut dyn FnMut(&mut [u8]) -> Result<()>,
) -> Result<Layout> {
    ensure!(
        matches!(block_size, 512 | 4096),
        "{block_size}-byte blocks are not a disk setup can erase"
    );
    let block = u64::from(block_size);
    let disk = Disk {
        block_bytes: block,
        blocks: bytes / block,
    };
    let align = ALIGNMENT / block;
    let esp_first = align;
    let esp_last = esp_first + ESP_BYTES / block - 1;
    let data_first = (esp_last + 1).next_multiple_of(align);
    // The backup GPT takes the last 33 blocks (512-byte) or 5 (4096-byte).
    let Some(data_last) = disk
        .blocks
        .checked_sub(disk.first_usable())
        .filter(|last| *last > data_first)
    else {
        bail!(
            "the disk holds {:.1} GB, too little for setup's layout",
            bytes as f64 / 1e9
        )
    };
    let mut guid = || -> Result<Guid> {
        let mut bytes = [0u8; 16];
        random(&mut bytes)?;
        Ok(Guid::from_entropy(bytes))
    };
    let esp = Partition {
        type_guid: Guid::ESP,
        unique_guid: guid()?,
        first_lba: esp_first,
        last_lba: esp_last,
        attributes: 0,
        name: "EFI System Partition".into(),
    };
    let data = Partition {
        type_guid: BASIC_DATA.parse().expect("a valid GUID"),
        unique_guid: guid()?,
        first_lba: data_first,
        last_lba: data_last,
        attributes: 0,
        name: name.into(),
    };
    let table = Table {
        disk_guid: guid()?,
        partitions: vec![esp.clone(), data.clone()],
    };
    let table = gpt::encode(disk, &table)
        .map_err(|error| anyhow::anyhow!("cannot lay out the GPT: {error}"))?;
    // The gaps outside the partitions first, so no signature of the disk's
    // old contents (an ISO 9660 image's, say) is left for a host to find.
    let mut writes = vec![
        Write::Zeros(0, esp_first * block),
        Write::Zeros(
            (data_last + 1) * block,
            (disk.blocks - data_last - 1) * block,
        ),
    ];
    writes.extend(
        table
            .into_iter()
            .map(|write| Write::Bytes(write.lba * block, write.bytes)),
    );
    let mut serial = || -> Result<u32> {
        let mut bytes = [0u8; 4];
        random(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    };
    writes.extend(fat32(&esp, block_size, "EFI", serial()?)?);
    writes.extend(fat32(&data, block_size, name, serial()?)?);
    Ok(Layout { writes })
}

/// The writes that format `partition` as an empty FAT32 volume, whose root
/// directory holds only the volume label.
fn fat32(partition: &Partition, block_size: u32, label: &str, serial: u32) -> Result<Vec<Write>> {
    let geometry = Geometry::new(partition.blocks(), block_size)?;
    let label = volume_label(label)?;
    let sector = geometry.sector_bytes;
    let base = partition.first_lba * sector;
    let boot = geometry.boot_sector(partition.first_lba, &label, serial)?;
    // Every cluster but the root directory's is free; the next is 3.
    let fsinfo = geometry.fsinfo(u32::try_from(geometry.clusters - 1)?, ROOT_CLUSTER + 1);

    // FAT entries 0 and 1 (the media byte and end of chain), and the root
    // directory's one-cluster chain.
    let mut fat = vec![0u8; usize::try_from(sector)?];
    for (slot, entry) in
        fat.as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip([MEDIA_ENTRY, END_OF_CHAIN, END_OF_CHAIN])
    {
        *slot = entry.to_le_bytes();
    }

    let mut root = vec![0u8; usize::try_from(geometry.cluster_bytes())?];
    root[..11].copy_from_slice(&label);
    root[11] = ATTRIBUTE_VOLUME_ID;

    let at = |sector_index: u64| base + sector_index * sector;
    let mut writes = vec![
        Write::Zeros(base, geometry.data_sector() * sector),
        Write::Bytes(at(0), boot.clone()),
        Write::Bytes(at(FSINFO_SECTOR), fsinfo.clone()),
        Write::Bytes(at(BACKUP_BOOT_SECTOR), boot),
        Write::Bytes(at(BACKUP_BOOT_SECTOR + FSINFO_SECTOR), fsinfo),
    ];
    for copy in 0..FATS {
        writes.push(Write::Bytes(at(geometry.fat_sector(copy)), fat.clone()));
    }
    writes.push(Write::Bytes(
        base + geometry.cluster_offset(ROOT_CLUSTER),
        root,
    ));
    Ok(writes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range;
    use stela_drivers::block::{BlockDevice, BlockError, Identity, Transport};
    use stela_runtime::pc::gpt::read as read_gpt;
    use {stela_formats::payload::ReadAt, stela_runtime::pc::fat32::Fat32};

    /// A dirty disk with the layout's ordered writes applied only to reads.
    struct Image<'a> {
        identity: Identity,
        writes: Vec<(Range<u64>, Option<&'a [u8]>)>,
    }

    impl<'a> Image<'a> {
        fn new(bytes: u64, block_size: u32, layout: &'a Layout) -> Self {
            let block = u64::from(block_size);
            assert!(bytes.is_multiple_of(block));
            let writes = layout
                .writes
                .iter()
                .map(|write| {
                    let (start, length, data) = match write {
                        Write::Bytes(at, data) => (*at, data.len() as u64, Some(data.as_slice())),
                        Write::Zeros(at, length) => (*at, *length, None),
                    };
                    let end = start.checked_add(length).expect("layout write overflows");
                    assert!(end <= bytes, "layout write exceeds disk capacity");
                    (start..end, data)
                })
                .collect();
            Self {
                identity: Identity {
                    transport: Transport::Ram,
                    location: "test".into(),
                    model: "layout".into(),
                    serial: "USB".into(),
                    blocks: bytes / block,
                    block_size,
                    read_only: true,
                    features: Default::default(),
                },
                writes,
            }
        }

        fn read_bytes(&self, offset: u64, buffer: &mut [u8]) {
            let end = offset + buffer.len() as u64;
            assert!(end <= self.identity.bytes());
            buffer.fill(0xa5);
            for (write, data) in &self.writes {
                let start = offset.max(write.start);
                let end = end.min(write.end);
                if start >= end {
                    continue;
                }
                let output = &mut buffer[(start - offset) as usize..(end - offset) as usize];
                if let Some(data) = data {
                    output.copy_from_slice(
                        &data[(start - write.start) as usize..(end - write.start) as usize],
                    );
                } else {
                    output.fill(0);
                }
            }
        }
    }

    impl BlockDevice for Image<'_> {
        fn identity(&self) -> &Identity {
            &self.identity
        }

        fn read(&mut self, lba: u64, buffer: &mut [u8]) -> Result<(), BlockError> {
            self.identity.check(lba, buffer.len())?;
            self.read_bytes(lba * u64::from(self.identity.block_size), buffer);
            Ok(())
        }

        fn write(&mut self, _lba: u64, _data: &[u8]) -> Result<(), BlockError> {
            Err(BlockError::ReadOnly)
        }
    }

    /// One partition's bytes, as the FAT reader reads them.
    struct Volume<'a> {
        disk: &'a Image<'a>,
        start: u64,
        bytes: u64,
    }

    impl ReadAt for Volume<'_> {
        type Error = &'static str;
        fn len(&self) -> u64 {
            self.bytes
        }
        fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> Result<(), Self::Error> {
            if offset
                .checked_add(buffer.len() as u64)
                .is_none_or(|end| end > self.bytes)
            {
                return Err("past the end");
            }
            self.disk.read_bytes(self.start + offset, buffer);
            Ok(())
        }
    }

    fn counter() -> impl FnMut(&mut [u8]) -> Result<()> {
        let mut next = 0u8;
        move |bytes: &mut [u8]| {
            for byte in bytes {
                next = next.wrapping_add(37);
                *byte = next;
            }
            Ok(())
        }
    }

    #[test]
    fn erase_creates_bootable_fat32_layout() {
        let bytes = 2u64 << 30;
        let layout = layout(bytes, 512, "STELA", &mut counter()).unwrap();
        let mut disk = Image::new(bytes, 512, &layout);
        let table = read_gpt(&mut disk).unwrap();
        let [esp, data] = &table.partitions[..] else {
            panic!("{:?}", table.partitions)
        };
        assert_eq!(esp.type_guid, Guid::ESP);
        assert_eq!(data.type_guid, BASIC_DATA.parse().unwrap());
        assert_eq!(data.name, "STELA");
        assert_eq!(esp.first_lba * 512, ALIGNMENT);
        assert_eq!(esp.blocks() * 512, ESP_BYTES);
        assert_eq!(data.first_lba * 512 % ALIGNMENT, 0);
        // The backup GPT ends the disk.
        assert!((data.last_lba + 1) * 512 <= bytes - 33 * 512);
        let mut sector = [0; 512];
        disk.read_bytes(bytes - 512, &mut sector);
        assert_eq!(&sector[..8], b"EFI PART");
        assert_ne!(esp.unique_guid, data.unique_guid);
        // The gap before the ESP is zeroed but for the GPT.
        let mut buffer = [0; 4096];
        for start in (34 * 512..ALIGNMENT).step_by(buffer.len()) {
            let len = (ALIGNMENT - start).min(buffer.len() as u64) as usize;
            disk.read_bytes(start, &mut buffer[..len]);
            assert!(buffer[..len].iter().all(|&byte| byte == 0));
        }
        disk.read_bytes(data.last_lba * 512, &mut sector);
        assert_eq!(sector, [0xa5; 512]);
        for (partition, label) in [(esp, b"EFI        "), (data, b"STELA      ")] {
            let first = partition.first_lba * 512;
            disk.read_bytes(first, &mut sector);
            assert_eq!(&sector[71..82], label);
            assert_eq!(&sector[82..90], b"FAT32   ");
            // The backup boot sector, and an empty root directory.
            let mut backup = [0; 512];
            disk.read_bytes(first + 6 * 512, &mut backup);
            assert_eq!(sector, backup);
            let mut fat = Fat32::open(Volume {
                disk: &disk,
                start: first,
                bytes: partition.blocks() * 512,
            })
            .unwrap();
            assert!(fat.file("EFI/BOOT/BOOTX64.EFI").is_err());
        }
    }

    #[test]
    fn northwood_label_matches_the_gpt_and_fat32_volume_metadata() {
        use stela_loader::mount_interface::layout::Layout as MediaLayout;

        let bytes = 2u64 << 30;
        let writes = layout(
            bytes,
            512,
            MediaLayout::Northwood.volume_label(),
            &mut counter(),
        )
        .unwrap();
        let mut disk = Image::new(bytes, 512, &writes);
        let table = read_gpt(&mut disk).unwrap();
        let [esp, data] = &table.partitions[..] else {
            panic!("{:?}", table.partitions)
        };
        assert_eq!(esp.type_guid, Guid::ESP);
        assert_eq!(data.type_guid, BASIC_DATA.parse().unwrap());
        assert_eq!(esp.name, "EFI System Partition");
        assert_eq!(data.name, "NORTHWOOD");
        for (partition, label) in [(esp, b"EFI        "), (data, b"NORTHWOOD  ")] {
            let base = partition.first_lba * 512;
            let geometry = Geometry::new(partition.blocks(), 512).unwrap();
            let mut sector = [0; 512];
            disk.read_bytes(base, &mut sector);
            assert_eq!(&sector[71..82], label);
            let mut backup = [0; 512];
            disk.read_bytes(base + BACKUP_BOOT_SECTOR * 512, &mut backup);
            assert_eq!(backup, sector);
            let mut root = [0; 32];
            disk.read_bytes(base + geometry.data_sector() * 512, &mut root);
            assert_eq!(&root[..11], label);
            assert_eq!(root[11], ATTRIBUTE_VOLUME_ID);
        }
    }

    #[test]
    fn four_kib_sector_esps_are_refused_rather_than_misformatted() {
        // 200 MiB of 4 KiB sectors has too few clusters for FAT32.
        let error = layout(2 << 30, 4096, "STELA", &mut counter()).unwrap_err();
        assert!(error.to_string().contains("too few clusters"), "{error}");
    }

    #[test]
    fn cluster_sizes_and_fat_sizes_follow_the_specification() {
        // A 64 GB stick: 32 KiB clusters, and a FAT that maps every one.
        let partition = Partition {
            type_guid: Guid::ESP,
            unique_guid: Guid([1; 16]),
            first_lba: 2048,
            last_lba: 2048 + (64_000_000_000 / 512) - 1,
            attributes: 0,
            name: String::new(),
        };
        let writes = fat32(&partition, 512, "STELA", 7).unwrap();
        let Write::Bytes(_, boot) = &writes[1] else {
            panic!("no boot sector")
        };
        assert_eq!(boot[13], 64);
        let fat_sectors = u32::from_le_bytes(boot[36..40].try_into().unwrap());
        let total = u32::from_le_bytes(boot[32..36].try_into().unwrap());
        let clusters = (total - 32 - 2 * fat_sectors) / 64;
        assert!(fat_sectors * 128 >= clusters + 2);
        assert!(volume_label("STELA").is_ok());
        assert!(volume_label("TOO LONG NAME").is_err());
        assert!(volume_label("bad/name").is_err());
    }
}
