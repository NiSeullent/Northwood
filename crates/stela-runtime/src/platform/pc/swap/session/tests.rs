// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use alloc::{rc::Rc, vec, vec::Vec};
use core::cell::RefCell;
use stela_drivers::block::{Identity, RamDisk};
use stela_formats::payload::gpt::{Table, encode};

#[cfg(all(feature = "std", target_arch = "x86_64"))]
mod held_frame;
mod volume;

#[derive(Default)]
struct Trace {
    writes: Vec<(u64, usize, bool)>,
    reads: usize,
    flushes: usize,
    write_before_failure: Option<usize>,
    read_before_failure: Option<usize>,
    fail_flush: bool,
    corrupt_read: bool,
    flip_byte_on_read: Option<usize>,
    observe_flush: Option<(
        alloc::sync::Arc<stela_engine::memory::RegisteredSystemRam>,
        u64,
    )>,
    resident_at_flush: Vec<bool>,
    changed_identity: bool,
}

struct DiskFixture {
    disk: RamDisk,
    changed: Identity,
    trace: Rc<RefCell<Trace>>,
}

// This is a single in-memory device. Geometry changes are visible to the
// caller before the first command. Reads and writes do not retain buffers.
impl ExclusiveDisk for DiskFixture {}

impl DiskFixture {
    fn write_inner(&mut self, lba: u64, bytes: &[u8], fua: bool) -> Result<(), BlockError> {
        let mut trace = self.trace.borrow_mut();
        trace.writes.push((lba, bytes.len(), fua));
        if fail_next(&mut trace.write_before_failure) {
            return Err(BlockError::Timeout);
        }
        self.disk.write(lba, bytes)
    }
}

fn fail_next(after: &mut Option<usize>) -> bool {
    match *after {
        Some(0) => {
            *after = None;
            true
        }
        Some(count) => {
            *after = Some(count - 1);
            false
        }
        None => false,
    }
}

impl BlockDevice for DiskFixture {
    fn identity(&self) -> &Identity {
        if self.trace.borrow().changed_identity {
            &self.changed
        } else {
            self.disk.identity()
        }
    }

    fn read(&mut self, lba: u64, bytes: &mut [u8]) -> Result<(), BlockError> {
        let mut trace = self.trace.borrow_mut();
        trace.reads += 1;
        if fail_next(&mut trace.read_before_failure) {
            return Err(BlockError::Timeout);
        }
        if let Some(offset) = trace.flip_byte_on_read.take() {
            self.disk.bytes[offset] ^= 1;
        }
        self.disk.read(lba, bytes)?;
        if trace.corrupt_read {
            bytes[0] ^= 1;
        }
        Ok(())
    }

    fn write(&mut self, lba: u64, bytes: &[u8]) -> Result<(), BlockError> {
        self.write_inner(lba, bytes, false)
    }

    fn write_fua(&mut self, lba: u64, bytes: &[u8]) -> Result<(), BlockError> {
        self.write_inner(lba, bytes, true)
    }

    fn flush_cache(&mut self) -> Result<(), BlockError> {
        let mut trace = self.trace.borrow_mut();
        trace.flushes += 1;
        if let Some((ram, physical)) = &trace.observe_flush {
            let resident = ram.nonresident_frame(*physical, PAGE_BYTES).is_none();
            trace.resident_at_flush.push(resident);
        }
        if trace.fail_flush {
            return Err(BlockError::Timeout);
        }
        Ok(())
    }
}

fn uuid() -> Guid {
    Guid::from_entropy([0x52; 16])
}

fn fixture(block: u32) -> (DiskFixture, Table, partition::Plan) {
    let mut disk = RamDisk::new("private swap fixture", (8 << 20) / u64::from(block), block);
    let geometry = geometry(disk.identity());
    let mut table = Table {
        disk_guid: Guid::from_entropy([0x61; 16]),
        partitions: vec![],
    };
    let planned = partition::plan(
        &table,
        geometry.first_usable()..=geometry.last_usable(),
        geometry.block_bytes,
        super::super::MIN_STORE_BYTES + SLOT_BYTES,
        uuid(),
    )
    .unwrap();
    table.partitions.push(planned.partition.clone());
    for write in encode(geometry, &table).unwrap() {
        disk.write(write.lba, &write.bytes).unwrap();
    }
    let entry = &planned.partition;
    disk.write(entry.first_lba, &planned.header.encode())
        .unwrap();
    disk.write(entry.first_lba - 1, &vec![0xc3; block as usize])
        .unwrap();
    disk.write(entry.last_lba + 1, &vec![0x35; block as usize])
        .unwrap();
    disk.identity.features.max_transfer = 4096;
    disk.identity.features.write_cache = true;
    let mut changed = disk.identity.clone();
    changed.block_size = if block == 512 { 4096 } else { 512 };
    (
        DiskFixture {
            disk,
            changed,
            trace: Rc::new(RefCell::new(Trace::default())),
        },
        table,
        planned,
    )
}

fn open(disk: DiskFixture, seed: u8) -> Result<Session<DiskFixture>, Error> {
    Session::open(disk, uuid(), &mut Generator::from_seed([seed; 32]))
}

#[test]
fn encrypted_slots_roundtrip_at_both_edges_of_512_and_4096_byte_stores() {
    for block in [512, 4096] {
        let (mut disk, _, planned) = fixture(block);
        disk.disk.identity.features.fua = block == 4096;
        let trace = disk.trace.clone();
        let mut session = open(disk, 1).unwrap();
        let header = session.header();
        assert!(header.encrypted());
        assert_eq!(header.state(), super::super::State::InUse);
        assert_ne!(header.generation(), 0);
        assert_eq!(header.layout().slot_count(), 2);
        let mut output = [0; PAGE_BYTES];
        let first = [0x41; PAGE_BYTES];
        let last = [0xe2; PAGE_BYTES];
        let proof = session.write_slot(0, 10, &first).unwrap();
        let last_proof = session.write_slot(1, 11, &last).unwrap();
        session.read_slot(0, 10, &proof, &mut output).unwrap();
        assert_eq!(output, first);
        session.read_slot(1, 11, &last_proof, &mut output).unwrap();
        assert_eq!(output, last);
        let entry = planned.partition;
        let disk_bytes = &session.disk.device().disk.bytes;
        let start = entry.first_lba as usize * block as usize;
        let end = (entry.last_lba + 1) as usize * block as usize;
        let data = start + header.layout().data_offset() as usize;
        assert_eq!(
            Header::decode_partition(&disk_bytes[start..], &entry, block.into()),
            Ok(header)
        );
        assert!(
            disk_bytes[start - block as usize..start]
                .iter()
                .all(|&b| b == 0xc3)
        );
        assert!(
            disk_bytes[end..end + block as usize]
                .iter()
                .all(|&b| b == 0x35)
        );
        assert_ne!(&disk_bytes[data..data + PAGE_BYTES], &first);
        assert_ne!(&disk_bytes[end - PAGE_BYTES..end], &last);
        let trace = trace.borrow();
        assert_eq!(trace.writes.len(), 9);
        assert!(
            trace
                .writes
                .iter()
                .all(|&(_, bytes, fua)| bytes == 4096 && fua == (block == 4096))
        );
        assert_eq!(trace.flushes, if block == 4096 { 0 } else { 3 });
    }
}

#[test]
fn range_geometry_and_counter_errors_issue_no_slot_commands() {
    let (disk, _, _) = fixture(512);
    let trace = disk.trace.clone();
    let mut session = open(disk, 2).unwrap();
    let writes = trace.borrow().writes.len();
    let plaintext = [0; PAGE_BYTES];
    for slot in [2, u64::MAX] {
        assert!(
            matches!(session.write_slot(slot, 0, &plaintext), Err(Error::Slot(s)) if s == slot)
        );
    }
    trace.borrow_mut().changed_identity = true;
    assert!(matches!(
        session.write_slot(0, 0, &plaintext),
        Err(Error::GeometryChanged)
    ));
    trace.borrow_mut().changed_identity = false;
    session.counter = u64::MAX;
    assert!(matches!(
        session.write_slot(0, 0, &plaintext),
        Err(Error::NonceExhausted)
    ));
    assert_eq!(trace.borrow().writes.len(), writes);
}

#[test]
fn newly_enabled_write_cache_refuses_writes_using_stale_features() {
    let (mut disk, _, _) = fixture(512);
    disk.disk.identity.features.write_cache = false;
    disk.changed = disk.disk.identity.clone();
    disk.changed.features.write_cache = true;
    let trace = disk.trace.clone();
    let mut session = open(disk, 8).unwrap();
    trace.borrow_mut().changed_identity = true;
    assert!(matches!(
        session.write_slot(0, 8, &[0xd3; PAGE_BYTES]),
        Err(Error::DeviceChanged)
    ));
    assert_eq!(trace.borrow().writes.len(), 1);
    assert_eq!(trace.borrow().flushes, 0);
}

#[test]
fn torn_writes_and_failed_flushes_never_return_proof_and_consume_nonces() {
    let (disk, _, _) = fixture(512);
    let trace = disk.trace.clone();
    let mut session = open(disk, 3).unwrap();
    let previous = session.write_slot(0, 3, &[0x66; PAGE_BYTES]).unwrap();
    trace.borrow_mut().write_before_failure = Some(1);
    assert!(matches!(
        session.write_slot(0, 3, &[0x77; PAGE_BYTES]),
        Err(Error::Device(BlockError::Timeout))
    ));
    let mut output = [0x18; PAGE_BYTES];
    assert!(matches!(
        session.read_slot(0, 3, &previous, &mut output),
        Err(Error::Authentication)
    ));
    assert_eq!(output, [0x18; PAGE_BYTES]);
    trace.borrow_mut().fail_flush = true;
    assert!(matches!(
        session.write_slot(0, 3, &[0x77; PAGE_BYTES]),
        Err(Error::Device(BlockError::Timeout))
    ));
    trace.borrow_mut().fail_flush = false;
    let retry = session.write_slot(0, 3, &[0x77; PAGE_BYTES]).unwrap();
    assert_eq!(retry.counter, previous.counter + 3);
    session.read_slot(0, 3, &retry, &mut output).unwrap();
    assert_eq!(output, [0x77; PAGE_BYTES]);
}

#[test]
fn failed_reads_and_wrong_receipts_never_publish_plaintext() {
    let (disk, _, _) = fixture(4096);
    let trace = disk.trace.clone();
    let mut session = open(disk, 4).unwrap();
    let proof = session.write_slot(0, 4, &[0xaf; PAGE_BYTES]).unwrap();
    let mut output = [0x21; PAGE_BYTES];
    let reads = trace.borrow().reads;
    assert!(matches!(
        session.read_slot(1, 4, &proof, &mut output),
        Err(Error::Authentication)
    ));
    assert!(matches!(
        session.read_slot(0, 5, &proof, &mut output),
        Err(Error::Authentication)
    ));
    assert_eq!(trace.borrow().reads, reads);
    let relabeled = SlotProof { page: 5, ..proof };
    assert!(matches!(
        session.read_slot(0, 5, &relabeled, &mut output),
        Err(Error::Authentication)
    ));
    assert_eq!(output, [0x21; PAGE_BYTES]);
    trace.borrow_mut().read_before_failure = Some(1);
    assert!(matches!(
        session.read_slot(0, 4, &proof, &mut output),
        Err(Error::Device(BlockError::Timeout))
    ));
    trace.borrow_mut().corrupt_read = true;
    assert!(matches!(
        session.read_slot(0, 4, &proof, &mut output),
        Err(Error::Authentication)
    ));
    assert_eq!(output, [0x21; PAGE_BYTES]);
    let (other, _, _) = fixture(4096);
    let mut other = open(other, 5).unwrap();
    assert!(matches!(
        other.read_slot(0, 4, &proof, &mut output),
        Err(Error::Authentication)
    ));
    assert_eq!(output, [0x21; PAGE_BYTES]);
}

#[test]
fn questionable_gpt_headers_and_read_only_disks_refuse_before_any_write() {
    for case in 0..11 {
        let (mut disk, mut table, planned) = fixture(512);
        let trace = disk.trace.clone();
        let geometry = geometry(disk.identity());
        match case {
            0 => disk.disk.bytes[512] ^= 1, // backup fallback
            1 => {
                table.partitions[0].name.push_str(" different backup");
                for write in encode(geometry, &table).unwrap().into_iter().take(2) {
                    disk.disk.write(write.lba, &write.bytes).unwrap();
                }
            }
            2 => disk.disk.bytes[446 + 16 + 4] = 0x07, // hybrid MBR
            3 => disk.disk.bytes[planned.partition.first_lba as usize * 512] ^= 1,
            4 => disk.disk.identity.read_only = true,
            5 => {
                let wrong = Header::new(super::super::MIN_STORE_BYTES, table.disk_guid).unwrap();
                disk.disk
                    .write(planned.partition.first_lba, &wrong.encode())
                    .unwrap();
            }
            6..=9 => {
                let at = (geometry.blocks as usize - 1) * 512;
                let backup = &mut disk.disk.bytes[at..at + 512];
                match case {
                    6 => backup[72..80].copy_from_slice(&2u64.to_le_bytes()),
                    7 => backup[32..40].copy_from_slice(&2u64.to_le_bytes()),
                    8 => backup[8..12].copy_from_slice(&0x0002_0000u32.to_le_bytes()),
                    9 => backup[20] = 1,
                    _ => unreachable!(),
                }
                update_gpt_header_crc(backup);
                // These edits leave the partition list readable.
                assert!(partition::inspect(&mut disk, Some(uuid())).is_ok());
            }
            10 => {
                let entries_crc = stela_formats::payload::gpt::crc32(&disk.disk.bytes[1024..1152]);
                for lba in [1, geometry.blocks - 1] {
                    let at = lba as usize * 512;
                    let header = &mut disk.disk.bytes[at..at + 512];
                    header[40..48].copy_from_slice(&3u64.to_le_bytes());
                    header[80..84].copy_from_slice(&1u32.to_le_bytes());
                    header[88..92].copy_from_slice(&entries_crc.to_le_bytes());
                    update_gpt_header_crc(header);
                }
                assert!(partition::inspect(&mut disk, Some(uuid())).is_ok());
            }
            _ => unreachable!(),
        }
        assert!(open(disk, 5).is_err(), "case {case}");
        assert!(trace.borrow().writes.is_empty(), "case {case}");
    }
}

fn update_gpt_header_crc(header: &mut [u8]) {
    let size = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    header[16..20].fill(0);
    let crc = stela_formats::payload::gpt::crc32(&header[..size]);
    header[16..20].copy_from_slice(&crc.to_le_bytes());
}

#[test]
fn failed_header_durability_does_not_open_a_session() {
    let (disk, _, _) = fixture(4096);
    let trace = disk.trace.clone();
    trace.borrow_mut().fail_flush = true;
    assert!(matches!(
        open(disk, 6),
        Err(Error::Device(BlockError::Timeout))
    ));
    assert_eq!(trace.borrow().writes.len(), 1);
    assert_eq!(trace.borrow().flushes, 1);
}

#[test]
fn split_fua_failure_never_returns_proof_or_falls_back_to_flush() {
    let (mut disk, _, _) = fixture(4096);
    disk.disk.identity.features.fua = true;
    let trace = disk.trace.clone();
    let mut session = open(disk, 7).unwrap();
    trace.borrow_mut().write_before_failure = Some(1);
    assert!(matches!(
        session.write_slot(0, 7, &[0xd9; PAGE_BYTES]),
        Err(Error::Device(BlockError::Timeout))
    ));
    assert_eq!(session.counter, 1);
    let trace = trace.borrow();
    assert_eq!(trace.writes.len(), 3);
    assert!(trace.writes.iter().all(|&(_, _, fua)| fua));
    assert_eq!(trace.flushes, 0);
}
