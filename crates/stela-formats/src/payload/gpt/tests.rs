use super::*;

const DISK: Disk = Disk {
    block_bytes: 512,
    blocks: 200 * 1024 * 1024 * 1024 / 512,
};

fn counter() -> impl FnMut() -> [u8; 16] {
    let mut n = 0u8;
    move || {
        n += 1;
        [n; 16]
    }
}

fn request() -> Request {
    Request {
        staging_bytes: 3 * MIB + 5,
        machine_state_bytes: MACHINE_STATE_BYTES,
        cache_bytes: DEFAULT_CACHE_BYTES,
        min_apple_bytes: DEFAULT_MIN_APPLE_BYTES,
    }
}

fn read_back(disk: Disk, writes: &[Write]) -> Table {
    let at = |lba: u64| &writes.iter().find(|w| w.lba == lba).unwrap().bytes;
    let info = parse_header(disk, 1, at(1)).unwrap().unwrap();
    parse_entries(&info, at(info.entries_lba)).unwrap()
}

#[test]
fn crc32_matches_the_ieee_check_value() {
    assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    assert_eq!(crc32(b""), 0);
}

#[test]
fn guids_round_trip_in_mixed_endian_form() {
    let text = "C12A7328-F81F-11D2-BA4B-00A0C93EC93B";
    let guid = Guid::parse(text).unwrap();
    assert_eq!(guid.0[..4], [0x28, 0x73, 0x2a, 0xc1]);
    assert_eq!(alloc::format!("{guid}"), text);
    assert!(Guid::parse("C12A7328F81F11D2BA4B00A0C93EC93B").is_none());
    assert!(Guid::parse("Z12A7328-F81F-11D2-BA4B-00A0C93EC93B").is_none());
    let random = Guid::from_entropy([0xff; 16]);
    assert_eq!(random.0[7] >> 4, 4);
    assert_eq!(random.0[8] >> 6, 2);
}

#[test]
fn a_blank_disk_gets_aligned_stela_partitions_after_a_reserved_esp_gap() {
    let plan = plan(DISK, None, request(), counter()).unwrap();
    assert!(plan.write);
    let roles: Vec<_> = plan.table.partitions.iter().map(|p| p.role()).collect();
    assert_eq!(roles, Role::STELA.map(Some).to_vec());
    let alignment = ALIGNMENT / DISK.block_bytes;
    let first = &plan.table.partitions[0];
    assert!(first.first_lba * DISK.block_bytes >= RESERVED_ESP_BYTES);
    for pair in plan.table.partitions.windows(2) {
        assert!(pair[0].last_lba < pair[1].first_lba);
    }
    for partition in &plan.table.partitions {
        assert_eq!(partition.first_lba % alignment, 0);
        assert_eq!((partition.last_lba + 1) % alignment, 0);
    }
    let staging = plan.partition(Role::Staging).unwrap();
    assert_eq!(staging.blocks() * DISK.block_bytes, 4 * MIB);
    assert!(
        plan.table
            .partitions
            .iter()
            .all(|p| !p.unique_guid.is_zero())
    );
    assert_ne!(plan.table.disk_guid, plan.table.partitions[0].unique_guid);
}

#[test]
fn a_fresh_swap_reservation_preserves_the_apple_minimum_and_resumes() {
    let bytes = core::num::NonZeroU64::new(32 << 30);
    let original = plan(DISK, None, request(), counter()).unwrap();
    let planned = plan_with_swap(DISK, None, request(), bytes, counter()).unwrap();
    let swap = planned.partition(Role::NorthwoodSwap).unwrap();
    assert_eq!(swap.blocks() * DISK.block_bytes, bytes.unwrap().get());
    let apple = planned.partition(Role::AppleRegion).unwrap();
    assert!(apple.first_lba > swap.last_lba);
    assert!(apple.blocks() * DISK.block_bytes >= DEFAULT_MIN_APPLE_BYTES);
    assert!(plan_with_swap(DISK, Some(&original.table), request(), bytes, counter()).is_err());
    let decoded = read_back(DISK, &encode(DISK, &planned.table).unwrap());
    assert_eq!(decoded, planned.table);
    assert!(
        !plan(DISK, Some(&decoded), request(), counter())
            .unwrap()
            .write
    );
}

#[test]
fn encoded_tables_parse_back_with_valid_crcs_at_both_block_sizes() {
    for block_bytes in [512, 4096] {
        let disk = Disk {
            block_bytes,
            blocks: DISK.blocks * 512 / block_bytes,
        };
        let plan = plan(disk, None, request(), counter()).unwrap();
        let writes = encode(disk, &plan.table).unwrap();
        assert_eq!(writes.last().unwrap().lba, 0);
        assert_eq!(writes.last().unwrap().bytes[450], 0xee);
        assert_eq!(read_back(disk, &writes), plan.table);
        let backup = &writes[1];
        assert_eq!(backup.lba, disk.blocks - 1);
        let backup = parse_header(disk, disk.blocks - 1, &backup.bytes)
            .unwrap()
            .unwrap();
        assert_eq!(backup.entries_lba, disk.blocks - 1 - 16384 / block_bytes);
    }
}

#[test]
fn an_existing_esp_is_kept_and_foreign_partitions_are_refused() {
    let esp = Partition {
        type_guid: Role::Esp.type_guid(),
        unique_guid: Guid([9; 16]),
        first_lba: 2048,
        last_lba: 2048 + 1024 * 1024 * 2 - 1,
        attributes: 0,
        name: "EFI".into(),
    };
    let table = Table {
        disk_guid: Guid([7; 16]),
        partitions: vec![esp.clone()],
    };
    let planned = plan(DISK, Some(&table), request(), counter()).unwrap();
    assert_eq!(planned.table.disk_guid, table.disk_guid);
    assert_eq!(planned.table.partitions[0], esp);
    assert!(planned.table.partitions[1].first_lba > esp.last_lba);
    // The same layout again is reused without a write.
    let again = plan(DISK, Some(&planned.table), request(), counter()).unwrap();
    assert!(!again.write);
    let bigger = Request {
        staging_bytes: 1024 * MIB,
        ..request()
    };
    assert!(plan(DISK, Some(&planned.table), bigger, counter()).is_err());
    let mut foreign = table.clone();
    foreign.partitions.push(Partition {
        type_guid: Guid([1; 16]),
        ..esp
    });
    assert!(plan(DISK, Some(&foreign), request(), counter()).is_err());
    let mut partial = planned.table.clone();
    partial.partitions.pop();
    assert!(plan(DISK, Some(&partial), request(), counter()).is_err());
}

/// A sparse disk: absent blocks read as zero.
#[derive(Clone, Default)]
struct Sparse {
    disk: Option<Disk>,
    blocks: alloc::collections::BTreeMap<u64, Vec<u8>>,
}

impl Sparse {
    fn new(disk: Disk) -> Self {
        Self {
            disk: Some(disk),
            ..Self::default()
        }
    }

    fn block_bytes(&self) -> usize {
        self.disk.unwrap().block_bytes as usize
    }

    /// Write the first `blocks` blocks of `write` (all with `None`).
    fn apply(&mut self, write: &Write, blocks: Option<usize>) {
        let size = self.block_bytes();
        for (n, block) in write
            .bytes
            .chunks(size)
            .enumerate()
            .take(blocks.unwrap_or(usize::MAX))
        {
            self.blocks.insert(write.lba + n as u64, block.to_vec());
        }
    }

    fn read(&self) -> Result<Found, PayloadError> {
        let disk = self.disk.unwrap();
        read(disk, |lba, count| {
            let mut out = Vec::new();
            for at in lba..lba + count {
                match self.blocks.get(&at) {
                    Some(block) => out.extend_from_slice(block),
                    None => out.resize(out.len() + disk.block_bytes as usize, 0),
                }
            }
            Ok(out)
        })
    }

    /// What setup does next: plan on what it read, and write the table
    /// again when the plan changes it or the read asks for a repair.
    fn rerun(&mut self) -> Table {
        let disk = self.disk.unwrap();
        let (existing, repair) = match self.read().unwrap() {
            Found::Blank | Found::Interrupted => (None, None),
            Found::Table { table, repair } => (Some(table), repair),
        };
        let plan = plan(disk, existing.as_ref(), request(), counter()).unwrap();
        if plan.write || repair.is_some() {
            for write in encode(disk, &plan.table).unwrap() {
                self.apply(&write, None);
            }
        }
        match self.read().unwrap() {
            Found::Table {
                table,
                repair: None,
            } => table,
            other => panic!("not repaired: {other:?}"),
        }
    }
}

#[test]
fn every_interrupted_write_to_a_blank_disk_is_resumed() {
    for block_bytes in [512, 4096] {
        let disk = Disk {
            block_bytes,
            blocks: DISK.blocks * 512 / block_bytes,
        };
        let planned = plan(disk, None, request(), counter()).unwrap().table;
        let writes = encode(disk, &planned).unwrap();
        assert_eq!(Sparse::new(disk).read().unwrap(), Found::Blank);
        for done in 0..=writes.len() {
            // Each write cut off after its first block, then each whole.
            for torn in [true, false] {
                if torn
                    && writes
                        .get(done)
                        .is_none_or(|w| w.bytes.len() <= block_bytes as usize)
                {
                    continue;
                }
                let mut image = Sparse::new(disk);
                for write in &writes[..done] {
                    image.apply(write, None);
                }
                if torn {
                    image.apply(&writes[done], Some(1));
                }
                let found = image.read().unwrap();
                let expected = match (done, torn) {
                    (0, false) => Found::Blank,
                    (0 | 1, _) => Found::Interrupted,
                    (2 | 3, _) => Found::Table {
                        table: planned.clone(),
                        repair: Some("the primary GPT is missing"),
                    },
                    (4, _) => Found::Table {
                        table: planned.clone(),
                        repair: Some("the protective MBR is missing"),
                    },
                    _ => Found::Table {
                        table: planned.clone(),
                        repair: None,
                    },
                };
                // Primary entries without their header read as missing too.
                assert_eq!(found, expected, "{block_bytes} {done} {torn}");
                let table = image.rerun();
                if done >= 2 {
                    // Committed to the backup: the same table, whole.
                    assert_eq!(table, planned);
                }
                assert_eq!(table.partitions.len(), 4);
            }
        }
    }
}

#[test]
fn an_interrupted_rewrite_keeps_the_existing_esp_and_its_bytes() {
    let esp = Partition {
        type_guid: Role::Esp.type_guid(),
        unique_guid: Guid([9; 16]),
        first_lba: 2048,
        last_lba: 2048 + 1024 * 1024 * 2 - 1,
        attributes: 0,
        name: "EFI".into(),
    };
    let old = Table {
        disk_guid: Guid([7; 16]),
        partitions: vec![esp.clone()],
    };
    let mut base = Sparse::new(DISK);
    for write in encode(DISK, &old).unwrap() {
        base.apply(&write, None);
    }
    // The ESP's own blocks, which no GPT write may touch.
    let marker = vec![0x5a; 512];
    base.blocks.insert(esp.first_lba, marker.clone());
    base.blocks.insert(esp.last_lba, marker.clone());
    let new = plan(DISK, Some(&old), request(), counter()).unwrap().table;
    let writes = encode(DISK, &new).unwrap();
    for write in &writes {
        let end = write.lba + write.bytes.len() as u64 / 512;
        assert!(
            new.partitions
                .iter()
                .all(|p| end <= p.first_lba || write.lba > p.last_lba)
        );
    }
    for done in 0..=writes.len() {
        let mut image = base.clone();
        for write in &writes[..done] {
            image.apply(write, None);
        }
        let Found::Table { table, repair } = image.read().unwrap() else {
            panic!("no table after {done} writes");
        };
        let (want, why) = match done {
            0 => (&old, None),
            1 => (&old, Some("the backup GPT is damaged")),
            2 => (&old, Some("the backup GPT differs from the primary")),
            3 => (&new, Some("the primary GPT is damaged")),
            _ => (&new, None),
        };
        assert_eq!((&table, repair), (want, why), "{done}");
        let table = image.rerun();
        assert_eq!(table.partitions[0], esp);
        assert_eq!(table.disk_guid, old.disk_guid);
        assert_eq!(image.blocks[&esp.first_lba], marker);
        assert_eq!(image.blocks[&esp.last_lba], marker);
    }
}

#[test]
fn a_disk_without_a_gpt_that_is_not_blank_is_refused() {
    let mut mbr = Sparse::new(DISK);
    mbr.blocks.insert(0, vec![0x33; 512]);
    assert!(mbr.read().is_err());
    // Foreign bytes where a backup entry array would be.
    let mut tail = Sparse::new(DISK);
    tail.blocks.insert(DISK.blocks - 33, vec![0x44; 512]);
    assert!(tail.read().is_err());
    // A Stela entry array, but another block written too.
    let planned = plan(DISK, None, request(), counter()).unwrap().table;
    let mut mixed = Sparse::new(DISK);
    mixed.apply(&encode(DISK, &planned).unwrap()[0], None);
    mixed.blocks.insert(DISK.blocks - 34, vec![1; 512]);
    assert!(mixed.read().is_err());
    // A damaged primary with no backup is refused, never blank.
    let mut damaged = Sparse::new(DISK);
    let mut header = encode(DISK, &planned).unwrap()[3].clone();
    header.bytes[60] ^= 1;
    damaged.apply(&header, None);
    assert!(damaged.read().is_err());
    // A backup that names another system's partition is read, and the
    // plan refuses to write beside it.
    let mut other = planned.clone();
    other.partitions[0].type_guid = Guid([1; 16]);
    let mut foreign = Sparse::new(DISK);
    for write in &encode(DISK, &other).unwrap()[..2] {
        foreign.apply(write, None);
    }
    let Found::Table { table, repair } = foreign.read().unwrap() else {
        panic!("the backup is not read");
    };
    assert_eq!(repair, Some("the primary GPT is missing"));
    assert!(plan(DISK, Some(&table), request(), counter()).is_err());
}

#[test]
fn small_disks_damaged_headers_and_blank_blocks_are_told_apart() {
    let small = Disk {
        block_bytes: 512,
        blocks: 32 * 1024 * 1024 * 1024 / 512,
    };
    assert!(plan(small, None, request(), counter()).is_err());
    let plan = plan(DISK, None, request(), counter()).unwrap();
    let writes = encode(DISK, &plan.table).unwrap();
    let mut header = writes[3].bytes.clone();
    assert!(parse_header(DISK, 1, &vec![0; 512]).unwrap().is_none());
    header[60] ^= 1;
    assert!(parse_header(DISK, 1, &header).is_err());
    assert!(parse_header(DISK, 2, &writes[3].bytes).is_err());
    let info = parse_header(DISK, 1, &writes[3].bytes).unwrap().unwrap();
    let mut entries = writes[2].bytes.clone();
    entries[40] ^= 1;
    assert!(parse_entries(&info, &entries).is_err());
}

#[test]
fn guid_text_is_hex_digits_only_and_parses_as_from_str() {
    let esp: Guid = "C12A7328-F81F-11D2-BA4B-00A0C93EC93B".parse().unwrap();
    assert_eq!(esp, Guid::ESP);
    assert_eq!(Role::Esp.type_guid(), Guid::ESP);
    // A sign is not a hex digit, though `u8::from_str_radix` takes one.
    assert!(Guid::parse("+12A7328-F81F-11D2-BA4B-00A0C93EC93B").is_none());
    assert!(
        "C12A7328-F81F-11D2-BA4B-00A0C93EC93"
            .parse::<Guid>()
            .is_err()
    );
}

#[test]
fn entries_outside_the_header_usable_range_are_refused() {
    let plan = plan(DISK, None, request(), counter()).unwrap();
    let writes = encode(DISK, &plan.table).unwrap();
    let mut info = parse_header(DISK, 1, &writes[3].bytes).unwrap().unwrap();
    assert_eq!(info.first_usable, 34);
    assert_eq!(info.last_usable, DISK.blocks - 34);
    // A partition over the primary entry array.
    let mut entries = writes[2].bytes.clone();
    entries[32..40].copy_from_slice(&2u64.to_le_bytes());
    info.entries_crc = crc32(&entries[..info.entry_array_bytes() as usize]);
    assert_eq!(
        parse_entries(&info, &entries),
        Err(PayloadError::Disk("GPT entry outside the usable range"))
    );
    let mut header = writes[3].bytes.clone();
    header[40..48].copy_from_slice(&(DISK.blocks).to_le_bytes());
    header[16..20].fill(0);
    let crc = crc32(&header[..HEADER_BYTES as usize]);
    header[16..20].copy_from_slice(&crc.to_le_bytes());
    assert!(parse_header(DISK, 1, &header).is_err());
}

#[test]
fn a_usable_range_past_the_disk_reads_the_backup() {
    let plan = plan(DISK, None, request(), counter()).unwrap();
    let mut disk = Sparse::new(DISK);
    for write in encode(DISK, &plan.table).unwrap() {
        disk.apply(&write, None);
    }
    // The primary claims one more block than the disk has.
    let header = disk.blocks.get_mut(&1).unwrap();
    header[48..56].copy_from_slice(&DISK.blocks.to_le_bytes());
    header[16..20].fill(0);
    let crc = crc32(&header[..HEADER_BYTES as usize]);
    header[16..20].copy_from_slice(&crc.to_le_bytes());
    assert_eq!(
        disk.read(),
        Ok(Found::Table {
            table: plan.table,
            repair: Some("the primary GPT is damaged"),
        })
    );
}

#[test]
fn encoding_refuses_what_it_cannot_write_and_terminates_names() {
    let mut table = plan(DISK, None, request(), counter()).unwrap().table;
    table.partitions[0].name = "N".repeat(36);
    let writes = encode(DISK, &table).unwrap();
    assert_eq!(read_back(DISK, &writes).partitions[0].name, "N".repeat(35));
    assert_eq!(writes[2].bytes[56 + 70..56 + 72], [0, 0]);

    let mut reversed = table.clone();
    let partition = &mut reversed.partitions[0];
    core::mem::swap(&mut partition.first_lba, &mut partition.last_lba);
    assert!(encode(DISK, &reversed).is_err());
    let mut crowded = table.clone();
    let extra = crowded.partitions[0].clone();
    crowded.partitions.resize(ENTRY_COUNT as usize + 1, extra);
    assert_eq!(
        encode(DISK, &crowded),
        Err(PayloadError::Disk("too many partitions"))
    );
}
