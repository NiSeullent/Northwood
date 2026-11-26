use super::*;
use alloc::{string::String, vec::Vec};
use stela_drivers::block::{Identity, RamDisk};

/// A RAM disk that records its transfers and can fail from a given
/// write on.
struct Recorder {
    disk: RamDisk,
    /// Each write's block and each flush (`None`), in order.
    events: Vec<Option<u64>>,
    reads: usize,
    writes_left: Option<usize>,
    /// Accept writes and keep none, as a device that loses them.
    dropping: bool,
}

impl Recorder {
    /// A disk with a write cache, so each durable write is followed
    /// by a flush.
    fn new(region: u64, block: u32) -> Self {
        let mut disk = RamDisk::new("log", region / u64::from(block), block);
        disk.identity.features.write_cache = true;
        Self {
            disk,
            events: Vec::new(),
            reads: 0,
            writes_left: None,
            dropping: false,
        }
    }

    fn log(&self) -> Result<Log, HeaderError> {
        read(&self.disk.bytes)
    }
}

impl BlockDevice for Recorder {
    fn identity(&self) -> &Identity {
        self.disk.identity()
    }

    fn read(&mut self, lba: u64, buffer: &mut [u8]) -> Result<(), BlockError> {
        self.reads += 1;
        self.disk.read(lba, buffer)
    }

    fn write(&mut self, lba: u64, data: &[u8]) -> Result<(), BlockError> {
        if let Some(left) = self.writes_left.as_mut() {
            if *left == 0 {
                return Err(BlockError::Timeout);
            }
            *left -= 1;
        }
        self.events.push(Some(lba));
        if self.dropping {
            return Ok(());
        }
        self.disk.write(lba, data)
    }

    fn flush_cache(&mut self) -> Result<(), BlockError> {
        self.events.push(None);
        self.disk.flush_cache()
    }
}

fn text(log: &Log) -> String {
    String::from_utf8(log.stream.clone()).unwrap()
}

#[test]
fn a_device_that_keeps_no_writes_fails_the_check() {
    let mut disk = Recorder::new(HEADER_BYTES + 4096, 512);
    let mut writer = Writer::open(&mut disk).unwrap();
    writer.flush(&mut disk).unwrap();
    assert_eq!(writer.verify(&mut disk), Ok(()));
    writer.append(b"more\n");
    disk.dropping = true;
    writer.flush(&mut disk).unwrap();
    assert_eq!(
        writer.verify(&mut disk),
        Err(Unkept::Head {
            stored: 0x15,
            written: 0x1a
        })
    );
    let mut blank = Recorder::new(HEADER_BYTES + 4096, 512);
    blank.dropping = true;
    let mut writer = Writer::open(&mut blank).unwrap();
    writer.flush(&mut blank).unwrap();
    assert_eq!(
        writer.verify(&mut blank),
        Err(Unkept::Header(HeaderError::Blank))
    );
}

#[test]
fn a_resumed_log_continues_the_same_boot() {
    let mut disk = Recorder::new(HEADER_BYTES + 4 * 4096, 512);
    let mut loader = Writer::open(&mut disk).unwrap();
    loader.append(b"before the takeover\n");
    loader.flush(&mut disk).unwrap();
    let mut engine = Writer::resume(&mut disk).unwrap();
    assert_eq!(engine.boots(), 1);
    engine.append(b"after it\n");
    engine.flush(&mut disk).unwrap();
    let log = disk.log().unwrap();
    assert_eq!(log.boots, 1);
    assert_eq!(
        text(&log),
        "=== STELA BOOT 1 ===\nbefore the takeover\nafter it\n"
    );
}

#[test]
fn boots_append_markers_on_512_and_4096_byte_blocks() {
    for block in [512, 4096] {
        let mut disk = Recorder::new(HEADER_BYTES + 4 * 4096, block);
        let mut writer = Writer::open(&mut disk).unwrap();
        assert_eq!(writer.fresh(), Some(&HeaderError::Blank));
        assert_eq!(writer.boots(), 1);
        writer.append(b"first line\n");
        writer.flush(&mut disk).unwrap();
        assert_eq!(
            text(&disk.log().unwrap()),
            "=== STELA BOOT 1 ===\nfirst line\n"
        );
        // Small flushes rewrite the tail block from memory, never
        // reading it back.
        let reads = disk.reads;
        for _ in 0..20 {
            writer.append(b"0123456789abcdef");
            writer.flush(&mut disk).unwrap();
        }
        assert_eq!(disk.reads, reads, "{block}-byte blocks");
        // A flush with nothing new writes nothing.
        let events = disk.events.len();
        writer.flush(&mut disk).unwrap();
        assert_eq!(disk.events.len(), events);
        // The next boot counts itself and ends the unfinished line.
        writer.append(b"cut off");
        writer.flush(&mut disk).unwrap();
        let mut again = Writer::open(&mut disk).unwrap();
        assert_eq!(again.fresh(), None);
        assert_eq!(again.boots(), 2);
        again.flush(&mut disk).unwrap();
        let log = disk.log().unwrap();
        assert_eq!(log.boots, 2);
        assert_eq!(log.head, log.stream.len() as u64);
        assert!(
            text(&log).ends_with("cut off\n=== STELA BOOT 2 ===\n"),
            "{}",
            text(&log)
        );
        assert_eq!(log.stream.len(), 21 + 11 + 320 + 7 + 1 + 21);
    }
}

#[test]
fn the_ring_wraps_and_the_reader_keeps_order() {
    for block in [512, 4096] {
        // A two-block ring.
        let ring = 2 * u64::from(block);
        let mut disk = Recorder::new(HEADER_BYTES + ring, block);
        let mut writer = Writer::open(&mut disk).unwrap();
        let mut stream: Vec<u8> = b"=== STELA BOOT 1 ===\n".to_vec();
        for line in 0..(3 * ring / 10) {
            let bytes = format!("line {line:04}\n");
            stream.extend_from_slice(bytes.as_bytes());
            writer.append(bytes.as_bytes());
            if line % 7 == 0 {
                writer.flush(&mut disk).unwrap();
            }
        }
        writer.flush(&mut disk).unwrap();
        let log = disk.log().unwrap();
        assert_eq!(log.head, stream.len() as u64);
        assert_eq!(log.overwritten(), stream.len() as u64 - ring);
        assert_eq!(log.stream, stream[stream.len() - ring as usize..]);
        // A boot opening a wrapped log keeps the oldest bytes of its
        // tail block.
        let mut again = Writer::open(&mut disk).unwrap();
        again.flush(&mut disk).unwrap();
        stream.extend_from_slice(b"=== STELA BOOT 2 ===\n");
        assert_eq!(
            disk.log().unwrap().stream,
            stream[stream.len() - ring as usize..]
        );
    }
}

#[test]
fn the_reader_orders_a_wrapped_region() {
    let ring = 16u64;
    let mut region = vec![0u8; (HEADER_BYTES + ring) as usize];
    Header {
        head: ring + 5,
        boots: 3,
        ring_bytes: ring,
    }
    .encode(&mut region[..HEADER_BYTES as usize]);
    // Stream bytes 5..21: 16..21 wrapped to ring 0..5.
    for at in 5..ring + 5 {
        region[(HEADER_BYTES + at % ring) as usize] = b'a' + at as u8;
    }
    let log = read(&region).unwrap();
    assert_eq!(log.boots, 3);
    assert_eq!(log.overwritten(), 5);
    let expected: Vec<u8> = (5..ring + 5).map(|at| b'a' + at as u8).collect();
    assert_eq!(log.stream, expected);
}

#[test]
fn the_header_is_written_last_so_an_interrupted_flush_keeps_the_old_stream() {
    for block in [512, 4096] {
        let mut disk = Recorder::new(HEADER_BYTES + 8 * 4096, block);
        let mut writer = Writer::open(&mut disk).unwrap();
        writer.append(b"kept\n");
        writer.flush(&mut disk).unwrap();
        let before = disk.log().unwrap();
        // Each flush ends with: data, flush, header, flush.
        let header = disk.events.len() - 2;
        assert_eq!(disk.events[header], Some(0));
        assert_eq!(disk.events[header - 1], None);
        assert_eq!(&disk.events[header + 1..], &[None]);
        assert!(disk.events[..header - 1].iter().all(|e| *e != Some(0)));
        // Data fills three blocks, one write; the device fails before
        // the header.
        writer.append(&vec![b'x'; 3 * block as usize]);
        disk.writes_left = Some(1);
        assert_eq!(writer.flush(&mut disk), Err(BlockError::Timeout));
        assert_eq!(disk.log().unwrap(), before);
    }
}

#[test]
fn a_pending_overflow_is_reported_in_the_stream() {
    let mut pending = Pending::new([0u8; 8]);
    pending.push(b"abc");
    pending.push(b"defgh");
    pending.push(b"ij");
    assert_eq!(pending.bytes(), 8 + "[boot log dropped 2 bytes]\n".len());
    let mut out = Vec::new();
    pending
        .drain(|bytes| {
            out.extend_from_slice(bytes);
            Ok::<_, ()>(())
        })
        .unwrap();
    assert_eq!(out, b"[boot log dropped 2 bytes]\ncdefghij");
    assert!(pending.is_empty());
    pending.push(b"0123456789AB");
    let mut out = Vec::new();
    pending
        .drain(|bytes| {
            out.extend_from_slice(bytes);
            Ok::<_, ()>(())
        })
        .unwrap();
    assert_eq!(out, b"[boot log dropped 4 bytes]\n456789AB");

    // Logs that read the same bytes from their own positions.
    let mut pending = Pending::new([0u8; 8]);
    let copy = |pending: &Pending<[u8; 8]>, from| {
        let mut out = Vec::new();
        let at = pending
            .copy_since(from, |bytes| {
                out.extend_from_slice(bytes);
                Ok::<_, ()>(())
            })
            .unwrap();
        (at, out)
    };
    pending.push(b"abcdef");
    assert_eq!(copy(&pending, 0), (6, b"abcdef".to_vec()));
    pending.push(b"ghij");
    assert_eq!(copy(&pending, 6), (10, b"ghij".to_vec()));
    assert_eq!(
        copy(&pending, 0),
        (10, b"[boot log dropped 2 bytes]\ncdefghij".to_vec())
    );
    assert_eq!(copy(&pending, 10), (10, Vec::new()));

    // The writer's own buffer between flushes.
    let mut disk = Recorder::new(HEADER_BYTES + 4 * 4096, 512);
    let mut writer = Writer::with_pending(&mut disk, 64, true).unwrap();
    for line in 0..10 {
        writer.append(format!("line {line:02}\n").as_bytes());
    }
    writer.flush(&mut disk).unwrap();
    // The marker (21 bytes) and 10 lines of 8 bytes; 64 are kept.
    assert_eq!(
        text(&disk.log().unwrap()),
        "[boot log dropped 37 bytes]\n\
             line 02\nline 03\nline 04\nline 05\nline 06\nline 07\nline 08\nline 09\n"
    );
}

#[test]
fn a_blank_or_invalid_header_starts_a_fresh_log_and_the_reader_refuses_it() {
    let mut disk = Recorder::new(HEADER_BYTES + 4096, 512);
    assert_eq!(disk.log(), Err(HeaderError::Blank));
    let mut writer = Writer::open(&mut disk).unwrap();
    writer.append(b"old boot\n");
    writer.flush(&mut disk).unwrap();
    let good = disk.disk.bytes.clone();

    let damage = |offset: usize| {
        let mut bytes = good.clone();
        bytes[offset] ^= 0x5a;
        bytes
    };
    let mut wrong_version = good.clone();
    wrong_version[8] = 3;
    let digest = Sha256::digest(&wrong_version[..FIELDS]);
    wrong_version[FIELDS..RECORD].copy_from_slice(&digest);
    let mut other_ring = good.clone();
    other_ring[32..40].copy_from_slice(&8192u64.to_le_bytes());
    let digest = Sha256::digest(&other_ring[..FIELDS]);
    other_ring[FIELDS..RECORD].copy_from_slice(&digest);
    let cases = [
        (damage(0), HeaderError::NotALog),
        (damage(16), HeaderError::Checksum),
        (damage(FIELDS + 3), HeaderError::Checksum),
        (wrong_version, HeaderError::Version(3)),
        (
            other_ring,
            HeaderError::RingSize {
                recorded: 8192,
                region: 4096,
            },
        ),
    ];
    for (bytes, error) in cases {
        disk.disk.bytes = bytes;
        assert_eq!(disk.log(), Err(error.clone()));
        let mut writer = Writer::open(&mut disk).unwrap();
        assert_eq!(writer.fresh(), Some(&error));
        assert_eq!(writer.boots(), 1);
        writer.flush(&mut disk).unwrap();
        // The old ring bytes are not taken as log data.
        assert_eq!(text(&disk.log().unwrap()), "=== STELA BOOT 1 ===\n");
    }
    assert_eq!(
        read(&good[..HEADER_BYTES as usize]),
        Err(HeaderError::Short {
            bytes: HEADER_BYTES
        })
    );
}

#[test]
fn reads_version_one_at_the_start_of_a_larger_usb_log() {
    let mut disk = Recorder::new(LOG_BYTES, 512);
    let mut writer = Writer::open(&mut disk).unwrap();
    writer.append(b"boot recorded in the first four MiB\n");
    writer.flush(&mut disk).unwrap();
    let original = disk.log().unwrap();
    disk.disk.bytes.resize(LOG_BYTES as usize * 2, 0);
    assert_eq!(disk.log().unwrap(), original);
}

#[test]
fn unsupported_geometry_is_refused() {
    let mut big = RamDisk::new("log", 4, 8192);
    assert!(matches!(
        Writer::open(&mut big),
        Err(BlockError::Unsupported(_))
    ));
    let mut tiny = RamDisk::new("log", 8, 512);
    assert!(matches!(
        Writer::open(&mut tiny),
        Err(BlockError::Unsupported(_))
    ));
    assert_eq!(PARTITION_BYTES, 0xc1_0000);
    assert_eq!(OFFSET, 0x81_0000);
}

#[test]
fn a_file_gives_the_block_runs_of_its_clusters_that_hold_the_layout() {
    use stela_formats::payload::pieces::Extent;
    let file = |extents: &[(u64, u64)]| fat32::File {
        bytes: extents.iter().map(|e| e.1).sum(),
        extents: extents
            .iter()
            .map(|&(offset, bytes)| Extent { offset, bytes })
            .collect(),
    };
    let size = 16 << 20;
    assert_eq!(
        file_runs(&file(&[(0x10_0000, size)]), 512, PARTITION_BYTES),
        Ok(vec![(0x800, PARTITION_BYTES / 512)])
    );
    // Scattered clusters give their runs in file order, adjacent ones
    // joined, up to the bytes the layout needs.
    assert_eq!(
        file_runs(
            &file(&[
                (0x20_0000, 0x1000),
                (0x20_1000, 0x1000),
                (0x10_0000, 0x1000),
                (0x40_0000, size)
            ]),
            512,
            PARTITION_BYTES
        ),
        Ok(vec![
            (0x1000, 16),
            (0x800, 8),
            (0x2000, (PARTITION_BYTES - 0x3000) / 512)
        ])
    );
    assert_eq!(
        file_runs(&file(&[]), 512, PARTITION_BYTES),
        Err(FileError::Short {
            bytes: 0,
            needed: PARTITION_BYTES
        })
    );
    assert_eq!(
        file_runs(&file(&[(0x10_0200, size)]), 4096, PARTITION_BYTES),
        Err(FileError::Misaligned {
            offset: 0x10_0200,
            block: 4096
        })
    );
    // A run may not end inside a block when another follows it.
    assert_eq!(
        file_runs(&file(&[(0, 0x1200), (0x8000, size)]), 4096, PARTITION_BYTES),
        Err(FileError::Misaligned {
            offset: 0x1200,
            block: 4096
        })
    );
    // A partial last block does not count.
    assert_eq!(
        file_runs(
            &file(&[(0, PARTITION_BYTES + 0x800)]),
            4096,
            PARTITION_BYTES
        ),
        Ok(vec![(0, PARTITION_BYTES / 4096)])
    );
    assert_eq!(
        file_runs(&file(&[(0, PARTITION_BYTES - 1)]), 512, PARTITION_BYTES),
        Err(FileError::Short {
            bytes: PARTITION_BYTES - 512,
            needed: PARTITION_BYTES
        })
    );
    // A log file needs only the region.
    assert_eq!(
        file_runs(&file(&[(0, LOG_BYTES)]), 512, LOG_BYTES),
        Ok(vec![(0, LOG_BYTES / 512)])
    );
}
