// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::pc::fat32::tests::Volume;
use crate::pc::swap::{DEFAULT_VOLUME_MAX_BYTES, SLOT_BYTES};

fn header(bytes: u64) -> Header {
    let uuid = "E3B6C15B-1369-4C2D-9121-51D581FD14C3".parse().unwrap();
    Header::new(bytes, uuid).unwrap()
}

#[test]
fn config_is_bounded_and_defines_exact_file_sizes() {
    let config = Config::parse(b"bytes 1073745920\r\n", 4096, DEFAULT_VOLUME_MAX_BYTES).unwrap();
    assert_eq!(config.bytes(), EXTENT_BYTES + 4096);
    let files: Vec<_> = config.files().collect();
    assert_eq!(files.len(), 2);
    assert_eq!(files[0].path(), "stela/swap/0000.swp");
    assert_eq!(files[0].bytes(), EXTENT_BYTES);
    assert_eq!(files[1].path(), "stela/swap/0001.swp");
    assert_eq!(files[1].bytes(), 4096);
    assert_eq!(
        Config::new(MAX_STORE_BYTES, 512, u64::MAX)
            .unwrap()
            .files()
            .last()
            .unwrap()
            .path(),
        "stela/swap/9999.swp"
    );
    assert!(matches!(
        Config::new(MAX_STORE_BYTES + 512, 512, u64::MAX),
        Err(Error::Capacity { .. })
    ));

    for text in [
        &b""[..],
        b"bytes ",
        b"bytes +1073745920",
        b"bytes  1073745920",
        b"bytes 1073745920 ",
        b"bytes 1073745920\nbytes 1073745920\n",
        b"bytes 18446744073709551616",
        b"bytes \xff",
    ] {
        assert_eq!(
            Config::parse(text, 512, DEFAULT_VOLUME_MAX_BYTES),
            Err(Error::Config)
        );
    }
    assert!(matches!(
        Config::new(MIN_STORE_BYTES + 1, 512, DEFAULT_VOLUME_MAX_BYTES),
        Err(Error::Capacity { .. })
    ));
    assert!(matches!(
        Config::new(MIN_STORE_BYTES, 512, MIN_STORE_BYTES - 1),
        Err(Error::Capacity { .. })
    ));
    assert_eq!(
        Config::new(MIN_STORE_BYTES, 0, DEFAULT_VOLUME_MAX_BYTES),
        Err(Error::BlockSize(0))
    );
}

fn medium(declared: u64, file_bytes: u64, header: Header, share_config_cluster: bool) -> Volume {
    let mut volume = Volume::new();
    let stela = volume.directory(2, 0, b"STELA      ");
    let swap = volume.directory(stela, 0, b"SWAP       ");
    let clusters = volume.allocate(file_bytes);
    volume.file_in(swap, 1, b"0000    SWP", &header.encode(), &clusters);
    volume.entry(
        swap,
        1,
        b"0000    SWP",
        0x20,
        clusters[0],
        file_bytes as u32,
    );
    let config = format!("bytes {declared}\n");
    if share_config_cluster {
        volume.file_in(
            swap,
            0,
            b"SWAP    CFG",
            config.as_bytes(),
            &clusters[clusters.len() - 1..],
        );
    } else {
        volume.file(swap, 0, b"SWAP    CFG", config.as_bytes());
    }
    volume
}

#[test]
fn fat_volume_inspection_checks_the_real_file_chain_and_header() {
    let bytes = MIN_STORE_BYTES;
    let expected = header(bytes);
    let mut volume = medium(bytes, bytes, expected, false);
    let metadata = inspect(&mut volume.disk, 512, DEFAULT_VOLUME_MAX_BYTES).unwrap();
    assert_eq!(metadata.header, expected);
    assert_eq!(metadata.config.bytes(), bytes);

    // These runs are aligned to their 512-byte FAT sectors, not 4 KiB
    // logical device blocks.
    assert_eq!(
        inspect(&mut volume.disk, 4096, DEFAULT_VOLUME_MAX_BYTES),
        Err(Error::Run)
    );
    let volume = medium(bytes, bytes + SLOT_BYTES, expected, false);
    assert_eq!(
        inspect(volume.disk, 512, DEFAULT_VOLUME_MAX_BYTES),
        Err(Error::FileSize {
            index: 0,
            expected: bytes,
            actual: bytes + SLOT_BYTES
        })
    );
    let volume = medium(bytes + SLOT_BYTES, bytes + SLOT_BYTES, expected, false);
    assert_eq!(
        inspect(volume.disk, 512, DEFAULT_VOLUME_MAX_BYTES),
        Err(Error::HeaderCapacity)
    );
    let volume = medium(bytes, bytes, header(bytes + SLOT_BYTES), false);
    assert_eq!(
        inspect(volume.disk, 512, DEFAULT_VOLUME_MAX_BYTES),
        Err(Error::Header(super::super::Error::ShortStore {
            available: bytes,
            required: bytes + SLOT_BYTES,
        }))
    );
}

#[test]
fn branded_store_never_borrows_extents_from_the_legacy_directory() {
    let bytes = MIN_STORE_BYTES;
    let expected = header(bytes);
    let mut volume = medium(bytes, bytes, expected, false);
    let northwood = volume.long_directory(2, 1, "Northwood", b"NORTHW~1   ");
    let swap = volume.directory(northwood, 0, b"SWAP       ");
    let config = format!("bytes {bytes}\n");
    volume.file(swap, 0, b"SWAP    CFG", config.as_bytes());
    assert!(
        matches!(inspect(&mut volume.disk, 512, DEFAULT_VOLUME_MAX_BYTES), Err(Error::Fat(FatError::Missing(path))) if path == "/northwood/swap/0000.swp")
    );

    let clusters = volume.allocate(bytes);
    volume.file_in(swap, 1, b"0000    SWP", &expected.encode(), &clusters);
    volume.entry(swap, 1, b"0000    SWP", 0x20, clusters[0], bytes as u32);
    assert_eq!(
        inspect(&mut volume.disk, 512, DEFAULT_VOLUME_MAX_BYTES)
            .unwrap()
            .header,
        expected
    );
}

#[test]
fn distinct_valid_fat_chains_cannot_share_a_candidate_cluster() {
    let bytes = MIN_STORE_BYTES;
    let volume = medium(bytes, bytes, header(bytes), true);
    assert!(matches!(
        inspect(volume.disk, 512, DEFAULT_VOLUME_MAX_BYTES),
        Err(Error::Overlap(_))
    ));
}

#[test]
fn layout_refuses_a_short_name_and_long_alias_for_the_same_path() {
    let bytes = MIN_STORE_BYTES;
    let expected = header(bytes);
    let mut volume = medium(bytes, bytes, expected, false);
    volume.long_directory(2, 1, "stela", b"STELA2     ");
    assert_eq!(
        inspect(&mut volume.disk, 512, DEFAULT_VOLUME_MAX_BYTES)
            .unwrap()
            .header,
        expected
    );
    assert!(matches!(
        verify_layout(volume.disk, 512, DEFAULT_VOLUME_MAX_BYTES),
        Err(Error::Fat(FatError::Chain(path, _))) if path == "/stela"
    ));
}

#[test]
fn unsorted_runs_can_be_adjacent_but_cannot_overlap() {
    let mut runs = [
        Extent {
            offset: 1536,
            bytes: 512,
        },
        Extent {
            offset: 512,
            bytes: 1024,
        },
        Extent {
            offset: 2048,
            bytes: 1024,
        },
    ];
    assert_eq!(validate_no_overlap(&mut runs), Ok(()));
    runs[0].bytes += 512;
    assert_eq!(validate_no_overlap(&mut runs), Err(Error::Overlap(1536)));
}

#[test]
fn slot_mapping_keeps_fragmented_runs_in_store_order() {
    let config = Config::new(MIN_STORE_BYTES + SLOT_BYTES, 4096, DEFAULT_VOLUME_MAX_BYTES).unwrap();
    let header = header(config.bytes());
    let data = header.layout().data_offset();
    let split = data + 4096;
    let first = 16 << 20;
    let second = 8 << 20;
    let layout = VerifiedLayout {
        metadata: Metadata { config, header },
        runs: vec![
            (
                0,
                Extent {
                    offset: first,
                    bytes: split,
                },
            ),
            (
                split,
                Extent {
                    offset: second,
                    bytes: config.bytes() - split,
                },
            ),
        ],
    };
    assert_eq!(
        layout.header_runs().collect::<Vec<_>>(),
        [Extent {
            offset: first,
            bytes: HEADER_BYTES as u64,
        }]
    );
    assert_eq!(
        layout.slot_runs(0).unwrap().collect::<Vec<_>>(),
        [
            Extent {
                offset: first + data,
                bytes: 4096,
            },
            Extent {
                offset: second,
                bytes: SLOT_BYTES - 4096,
            },
        ]
    );
    let last = header.layout().slot_count() - 1;
    assert_eq!(
        layout.slot_runs(last).unwrap().collect::<Vec<_>>(),
        [Extent {
            offset: second + header.layout().slot_offset(last).unwrap() - split,
            bytes: SLOT_BYTES,
        }]
    );
    assert!(layout.slot_runs(last + 1).is_none());
}
