use super::*;
use alloc::collections::BTreeMap;
use stela_contracts::refuse;

const BASE: u64 = 0x8_0000_0000;

const PATCHES: &str = r#"
    {"source": "dram-size", "required": false, "region": "DeviceTree", "file": "devicetree.adt", "file_offset": 0, "bytes": 8, "encoding": "u64-le", "field": "/chosen dram-size", "prepared_hex": ""},
    {"source": "entropy", "required": true, "region": "DeviceTree", "file": "devicetree.adt", "file_offset": 8, "bytes": 4, "encoding": "bytes", "field": "/chosen random-seed", "prepared_hex": ""},
    {"source": "framebuffer", "required": false, "region": "DeviceTree", "file": "devicetree.adt", "file_offset": 12, "bytes": 8, "encoding": "u64-le", "field": "boot_args.Video.v_display", "prepared_hex": ""}"#;

fn manifest_json(kernel: &[u8], tree: &[u8]) -> String {
    manifest_with(kernel, tree, PATCHES)
}

fn manifest_with(kernel: &[u8], tree: &[u8], patches: &str) -> String {
    let sha = crate::sha256;
    format!(
        r#"{{
  "schema_version": 4, "unknown_field": true,
  "options": {{"dram_base": {BASE}, "dram_size": 1073741824, "phys_base": {BASE}}},
  "files": [
    {{"name": "kernel.img", "bytes": {kb}, "sha256": "{ks}"}},
    {{"name": "devicetree.adt", "bytes": {tb}, "sha256": "{ts}"}}
  ],
  "regions": [
    {{"name": "BootKC", "file": "kernel.img", "file_offset": 0, "bytes": {kb}, "guest_pa": {BASE}, "memory_bytes": 16384, "sha256": "{ks}"}},
    {{"name": "DeviceTree", "file": "devicetree.adt", "file_offset": 0, "bytes": {tb}, "guest_pa": {dt}, "memory_bytes": 16384, "sha256": "{ts}"}}
  ],
  "entry": {{"image": "SPTM", "virtual_address": 18446744005645091720, "physical_address": {BASE}}},
  "entry_x0": {dt},
  "boot_args": {{"command_line": "-v"}},
  "top_of_kernel_data": {top},
  "patches": [{patches}
  ]
}}"#,
        kb = kernel.len(),
        ks = sha(kernel),
        tb = tree.len(),
        ts = sha(tree),
        dt = BASE + 0x4000,
        top = BASE + 0x8000,
    )
}

fn write_into(ram: &mut [u8], pa: u64, bytes: &[u8]) -> bool {
    let Some(offset) = pa.checked_sub(BASE).map(|o| o as usize) else {
        return false;
    };
    match ram.get_mut(offset..offset + bytes.len()) {
        Some(slot) => {
            slot.copy_from_slice(bytes);
            true
        }
        None => false,
    }
}

fn machine() -> Machine {
    Machine {
        dram_base: BASE,
        ram_bytes: FRAME_BYTES,
        dram_bytes: FRAME_BYTES,
        timebase_hz: 24_000_000,
        framebuffer: None,
        boot_args: None,
        nvram: None,
    }
}

fn tiled_kernel_manifest(first: &[u8], second: &[u8], tree: &[u8]) -> Manifest {
    let file = [first, second].concat();
    let json = manifest_json(&file, tree);
    let mut manifest = Manifest::parse(json.as_bytes()).unwrap();
    manifest.regions[0].bytes = first.len() as u64;
    manifest.regions[0].sha256 = crate::sha256(first);
    let second = Region {
        name: "BootKC-later".into(),
        file: "kernel.img".into(),
        file_offset: first.len() as u64,
        bytes: second.len() as u64,
        memory_bytes: 0x4000,
        guest_pa: BASE + 0x8000,
        sha256: crate::sha256(second),
        virtual_address: None,
    };
    // The source parts are ordered by file offset, independently of the
    // manifest's guest load order.
    manifest.regions.insert(0, second);
    manifest.top_of_kernel_data = BASE + 0xc000;
    manifest
}

fn verify_kernel_parts<'a>(
    manifest: &'a Manifest,
    parts: &'a [FilePart<'a>],
    tree: &'a [u8],
) -> Result<Verified<'a>, BundleError> {
    manifest.verify_files(&|name| match name {
        "kernel.img" => Some(FileData::Parts(parts)),
        "devicetree.adt" => Some(FileData::Whole(tree)),
        _ => None,
    })
}

#[test]
fn loads_discontiguous_verified_parts_in_manifest_order() -> Result<(), BundleError> {
    let staged = [0x12, 0x34, 0xee, 0x56, 0x78, 0x9a];
    let first = &staged[..2];
    let second = &staged[3..];
    let tree = [0x11; 20];
    let manifest = tiled_kernel_manifest(first, second, &tree);
    let parts = [
        FilePart {
            offset: 0,
            bytes: first,
        },
        FilePart {
            offset: 2,
            bytes: second,
        },
    ];
    let verified = verify_kernel_parts(&manifest, &parts, &tree)?;
    let mut actual = vec![0xff; 0xc000];
    let mut writes = Vec::new();
    let image = verified.load(
        &machine(),
        &mut |bytes| {
            bytes.fill(0x42);
            Ok(())
        },
        &mut |pa, bytes| {
            if pa == BASE || pa == BASE + 0x8000 {
                writes.push((pa, bytes.as_ptr()));
            }
            write_into(&mut actual, pa, bytes)
        },
    )?;
    assert_eq!(
        writes,
        [(BASE + 0x8000, second.as_ptr()), (BASE, first.as_ptr())]
    );
    assert_eq!(image.regions[0].name, "BootKC-later");
    assert_eq!(&actual[..2], first);
    assert_eq!(&actual[0x8000..0x8003], second);

    let whole = [first, second].concat();
    let verified = manifest.verify(&|name| match name {
        "kernel.img" => Some(whole.as_slice()),
        "devicetree.adt" => Some(tree.as_slice()),
        _ => None,
    })?;
    let mut expected = vec![0xff; 0xc000];
    verified.load(
        &machine(),
        &mut |bytes| {
            bytes.fill(0x42);
            Ok(())
        },
        &mut |pa, bytes| write_into(&mut expected, pa, bytes),
    )?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn part_backing_rejects_gaps_aliases_short_parts_and_bad_hashes() {
    let first = FilePart {
        offset: 0,
        bytes: b"ab",
    };
    let second = FilePart {
        offset: 2,
        bytes: b"cde",
    };
    let tree = [0x11; 20];
    let mut manifest = tiled_kernel_manifest(first.bytes, second.bytes, &tree);
    for parts in [
        &[][..],
        &[first][..],
        &[second, first],
        &[
            first,
            FilePart {
                offset: 1,
                ..second
            },
        ],
        &[
            first,
            FilePart {
                offset: 3,
                ..second
            },
        ],
        &[
            first,
            FilePart {
                bytes: b"",
                ..second
            },
        ],
        &[
            first,
            FilePart {
                bytes: b"cd",
                ..second
            },
        ],
        &[FilePart {
            bytes: b"abcde",
            ..first
        }],
    ] {
        assert!(matches!(
            verify_kernel_parts(&manifest, parts, &tree),
            Err(BundleError::File { name, .. }) if name == "kernel.img"
        ));
    }
    let tampered = [
        first,
        FilePart {
            bytes: b"cdf",
            ..second
        },
    ];
    assert!(matches!(
        verify_kernel_parts(&manifest, &tampered, &tree),
        Err(BundleError::Region { name, .. }) if name == "BootKC-later"
    ));

    let parts = [first, second];
    manifest.files[0].bytes += 1;
    assert!(matches!(
        verify_kernel_parts(&manifest, &parts, &tree),
        Err(BundleError::File { .. })
    ));
    manifest.files[0].bytes -= 1;
    for offset in [1, 3, u64::MAX] {
        manifest.regions[0].file_offset = offset;
        assert!(matches!(
            verify_kernel_parts(&manifest, &parts, &tree),
            Err(BundleError::File { .. })
        ));
    }
    manifest.regions[0].file_offset = 2;
    manifest.regions[0].memory_bytes = 2;
    assert!(matches!(
        verify_kernel_parts(&manifest, &parts, &tree),
        Err(BundleError::Region { .. })
    ));
    manifest.regions[0].memory_bytes = 0x4000;
    manifest.files.push(manifest.files[0].clone());
    assert!(matches!(
        verify_kernel_parts(&manifest, &parts, &tree),
        Err(BundleError::File { .. })
    ));
}

#[test]
fn whole_backing_still_hashes_bytes_outside_manifest_regions() {
    let tree = [0x11; 20];
    let mut manifest = tiled_kernel_manifest(b"ab", b"cde", &tree);
    let whole = b"abcdeX";
    manifest.files[0].bytes = whole.len() as u64;
    manifest.files[0].sha256 = crate::sha256(whole);
    for (bytes, valid) in [(whole, true), (b"abcdeY", false)] {
        let result = manifest.verify(&|name| match name {
            "kernel.img" => Some(bytes.as_slice()),
            "devicetree.adt" => Some(tree.as_slice()),
            _ => None,
        });
        if valid {
            assert!(result.is_ok());
        } else {
            assert!(matches!(
                result,
                Err(BundleError::File { name, .. }) if name == "kernel.img"
            ));
        }
    }
}

#[test]
fn admission_ledger_defaults_conservatively_and_preserves_exact_evidence() {
    let json = manifest_json(&[0xaa; 100], &[0x11; 20]);
    assert!(
        Manifest::parse(json.as_bytes())
            .unwrap()
            .hidden_devices
            .is_empty()
    );
    let supplied = json.replacen(
        "\"schema_version\": 4",
        "\"hidden_devices\": [\"/arm-io/ave\", \"/ secure-root-prefix\"], \"schema_version\": 4",
        1,
    );
    assert_eq!(
        Manifest::parse(supplied.as_bytes()).unwrap().hidden_devices,
        ["/arm-io/ave", "/ secure-root-prefix"]
    );
    let malformed = json.replacen(
        "\"schema_version\": 4",
        "\"hidden_devices\": [17], \"schema_version\": 4",
        1,
    );
    assert!(Manifest::parse(malformed.as_bytes()).is_err());
}

#[test]
fn selected_source_metadata_is_typed_and_duplicate_fields_refuse() {
    let json = manifest_json(&[1; 32], &[2; 32]);
    let authored = Manifest::parse(json.as_bytes()).unwrap();
    assert!(authored.target.is_none());
    assert!(authored.sources.is_empty());
    let target = r#"{"product_type":"test","device_class":"test","product_version":"test","product_build":"test","variant":"test","chip_id":24576,"board_id":8}"#;
    let source = r#"{"component":"RestoreDeviceTree","member":"Firmware/test.im4p","container_sha256":"container retained verbatim","payload_type":"dtre","description":"","decoded_bytes":32,"decoded_sha256":"decoded retained verbatim"}"#;
    let metadata = format!(r#""target":{target},"sources":[{source}],"#);
    let typed = json.replacen(
        "\"schema_version\"",
        &format!("{metadata}\"schema_version\""),
        1,
    );
    let parsed = Manifest::parse(typed.as_bytes()).unwrap();
    assert_eq!(parsed.target.as_ref().unwrap().board_id, Some(8));
    assert_eq!(parsed.sources[0].member, "Firmware/test.im4p");
    assert_eq!(
        parsed.sources[0].decoded_sha256,
        "decoded retained verbatim"
    );
    // Typed parsing preserves metadata; the Apple Entry intake separately
    // validates digests, selected IDs and original file contents.
    for repeated in [
        format!(r#""target":{target},"#),
        format!(r#""sources":[{source}],"#),
    ] {
        let wrong = typed.replacen(
            "\"schema_version\"",
            &format!("{repeated}\"schema_version\""),
            1,
        );
        assert!(Manifest::parse(wrong.as_bytes()).is_err());
    }
    for wrong in [
        typed.replace("\"chip_id\":24576", "\"chip_id\":24576,\"chip_id\":24576"),
        typed.replace("\"board_id\":8", "\"board_id\":4294967296"),
        typed.replace("\"decoded_bytes\":32", "\"decoded_bytes\":-1"),
    ] {
        assert!(Manifest::parse(wrong.as_bytes()).is_err());
    }
}

#[test]
fn boot_cpu_schema4_preserves_typed_source_identity_and_rejects_ambiguity() {
    let json = manifest_json(&[1; 32], &[2; 32]);
    assert!(Manifest::parse(json.as_bytes()).unwrap().boot_cpu.is_none());
    let record = r#"{"path":"/cpus/cpu4","affinity":256,"cluster_type":80,"original_index":4,"selection":"first-performance"}"#;
    let with_cpu = json.replacen(
        "\"schema_version\"",
        &format!("\"boot_cpu\":{record},\"schema_version\""),
        1,
    );
    let selected = Manifest::parse(with_cpu.as_bytes())
        .unwrap()
        .boot_cpu
        .unwrap();
    assert_eq!(selected.path, "/cpus/cpu4");
    assert_eq!(selected.affinity, 256);
    assert_eq!(selected.cluster_type, b'P');
    assert_eq!(selected.original_index, 4);
    assert_eq!(selected.selection, BootCpuSelection::FirstPerformance);
    let first = with_cpu.replace("first-performance", "first");
    assert_eq!(
        Manifest::parse(first.as_bytes())
            .unwrap()
            .boot_cpu
            .unwrap()
            .selection,
        BootCpuSelection::First
    );
    let explicit = with_cpu.replace("first-performance", "explicit");
    assert_eq!(
        Manifest::parse(explicit.as_bytes())
            .unwrap()
            .boot_cpu
            .unwrap()
            .selection,
        BootCpuSelection::Explicit
    );
    for wrong in [
        with_cpu.replace("first-performance", "cpu0"),
        with_cpu.replace("\"affinity\":256", "\"affinity\":4294967296"),
        with_cpu.replace("\"cluster_type\":80", "\"cluster_type\":256"),
        with_cpu.replace("\"original_index\":4", "\"original_index\":-1"),
        with_cpu.replace(
            "\"original_index\":4",
            "\"original_index\":4,\"original_index\":4",
        ),
        with_cpu.replacen(
            "\"schema_version\"",
            &format!("\"boot_cpu\":{record},\"schema_version\""),
            1,
        ),
    ] {
        assert!(Manifest::parse(wrong.as_bytes()).is_err());
    }
    for old in 0..4 {
        assert!(
            Manifest::parse(
                json.replace(
                    "\"schema_version\": 4",
                    &format!("\"schema_version\": {old}")
                )
                .as_bytes()
            )
            .is_err()
        );
    }
}

#[test]
fn the_mac_reports_the_installed_dram_above_the_backed_range() -> Result<(), BundleError> {
    let kernel = [0xaa; 100];
    let tree = [0x11; 20];
    let json = manifest_json(&kernel, &tree);
    let manifest = Manifest::parse(json.as_bytes())?;
    let files: BTreeMap<&str, &[u8]> =
        BTreeMap::from([("kernel.img", &kernel[..]), ("devicetree.adt", &tree[..])]);
    let verified = manifest.verify(&|name| files.get(name).copied())?;
    let mut ram = vec![0xffu8; FRAME_BYTES as usize];
    let mut entropy = |buffer: &mut [u8]| {
        buffer.fill(0x42);
        Ok(())
    };
    // Host memory Stela keeps shows as DRAM the kernel does not manage.
    let kept = Machine {
        dram_bytes: 3 * FRAME_BYTES,
        ..machine()
    };
    verified.load(&kept, &mut entropy, &mut |pa, bytes| {
        write_into(&mut ram, pa, bytes)
    })?;
    assert_eq!(&ram[0x4000..0x4008], &(3 * FRAME_BYTES).to_le_bytes());
    let short = Machine {
        dram_bytes: FRAME_BYTES - 0x4000,
        ..machine()
    };
    assert!(matches!(
        verified.load(&short, &mut entropy, &mut |_, _| true),
        Err(BundleError::Machine(_))
    ));
    Ok(())
}

#[test]
fn verifies_loads_and_patches_into_guest_ram() -> Result<(), BundleError> {
    let kernel = [0xaa; 100];
    let tree = [0x11; 20];
    let json = manifest_json(&kernel, &tree);
    let mut manifest = Manifest::parse(json.as_bytes())?;
    // Leave more than two zero chunks after the prepared tree bytes.
    manifest.regions[1].memory_bytes = 0x9000;
    manifest.top_of_kernel_data = BASE + 0xd000;
    let files: BTreeMap<&str, &[u8]> =
        BTreeMap::from([("kernel.img", &kernel[..]), ("devicetree.adt", &tree[..])]);
    let verified = manifest.verify(&|name| files.get(name).copied())?;
    let mut ram = vec![0xffu8; FRAME_BYTES as usize];
    let mut entropy = |buffer: &mut [u8]| {
        buffer.fill(0x42);
        Ok(())
    };
    let image = verified.load(&machine(), &mut entropy, &mut |pa, bytes| {
        write_into(&mut ram, pa, bytes)
    })?;
    assert_eq!(image.x0, BASE + 0x4000);
    assert_eq!(image.kept, vec!["boot_args.Video.v_display".to_string()]);
    assert_eq!(&ram[..100], &kernel);
    assert!(ram[100..0x4000].iter().all(|&b| b == 0));
    let dt = 0x4000;
    assert_eq!(&ram[dt..dt + 8], &FRAME_BYTES.to_le_bytes());
    assert_eq!(&ram[dt + 8..dt + 12], &[0x42; 4]);
    assert_eq!(&ram[dt + 12..dt + 20], &[0x11; 8]);
    assert!(ram[dt + 20..0xd000].iter().all(|&b| b == 0));
    assert_eq!(ram[0xd000], 0xff);

    // With a framebuffer and verbose boot-args, v_display is 0.
    let screen = Machine {
        framebuffer: Some(Framebuffer {
            guest_pa: 0xc000_0000,
            bytes: 0x80_0000,
            row_bytes: 7680,
            width: 1920,
            height: 1080,
            bits_per_pixel: 32,
        }),
        ..machine()
    };
    let image = verified.load(&screen, &mut entropy, &mut |pa, bytes| {
        write_into(&mut ram, pa, bytes)
    })?;
    assert!(image.kept.is_empty());
    assert_eq!(&ram[dt + 12..dt + 20], &0u64.to_le_bytes());
    Ok(())
}

#[test]
fn external_pipe_geometry_uses_the_same_framebuffer() -> Result<(), BundleError> {
    let kernel = [0xaa; 100];
    let tree = [0x11; 28];
    let patches = format!(
        r#"{PATCHES},
        {{"source":"framebuffer","required":true,"region":"DeviceTree","file":"devicetree.adt","file_offset":20,"bytes":4,"encoding":"bytes","field":"/arm-io/dispext0 framebuffer-width","prepared_hex":""}},
        {{"source":"framebuffer","required":true,"region":"DeviceTree","file":"devicetree.adt","file_offset":24,"bytes":4,"encoding":"bytes","field":"/arm-io/dispext0 framebuffer-height","prepared_hex":""}}"#
    );
    let manifest = Manifest::parse(manifest_with(&kernel, &tree, &patches).as_bytes())?;
    let files: BTreeMap<&str, &[u8]> =
        BTreeMap::from([("kernel.img", &kernel[..]), ("devicetree.adt", &tree[..])]);
    let verified = manifest.verify(&|name| files.get(name).copied())?;
    let mut ram = vec![0xff; FRAME_BYTES as usize];
    let mut entropy = |bytes: &mut [u8]| {
        bytes.fill(0x42);
        Ok(())
    };
    // A visible display cannot keep physical-board geometry without a
    // real backing resource when the native GOP is unavailable.
    assert!(
        verified
            .load(&machine(), &mut entropy, &mut |pa, bytes| write_into(
                &mut ram, pa, bytes
            ))
            .is_err()
    );
    let screen = Machine {
        framebuffer: Some(Framebuffer {
            guest_pa: 0xc000_0000,
            bytes: 0x80_0000,
            row_bytes: 7680,
            width: 1920,
            height: 1080,
            bits_per_pixel: 32,
        }),
        ..machine()
    };
    verified.load(&screen, &mut entropy, &mut |pa, bytes| {
        write_into(&mut ram, pa, bytes)
    })?;
    assert_eq!(&ram[0x4014..0x4018], &1920u32.to_le_bytes());
    assert_eq!(&ram[0x4018..0x401c], &1080u32.to_le_bytes());
    Ok(())
}

#[test]
fn display_scale_and_rotation_follow_the_video_depth() -> Result<(), BundleError> {
    let kernel = [0xaa; 100];
    let tree = [0x11; 32];
    let patches = format!(
        r#"{PATCHES},
        {{"source":"framebuffer","required":false,"region":"DeviceTree","file":"devicetree.adt","file_offset":20,"bytes":4,"encoding":"bytes","field":"/chosen display-rotation","prepared_hex":""}},
        {{"source":"framebuffer","required":false,"region":"DeviceTree","file":"devicetree.adt","file_offset":24,"bytes":4,"encoding":"bytes","field":"/chosen display-scale","prepared_hex":""}},
        {{"source":"framebuffer","required":false,"region":"DeviceTree","file":"devicetree.adt","file_offset":28,"bytes":4,"encoding":"bytes","field":"/chosen display-boot-rotation","prepared_hex":""}}"#
    );
    let manifest = Manifest::parse(manifest_with(&kernel, &tree, &patches).as_bytes())?;
    let files: BTreeMap<&str, &[u8]> =
        BTreeMap::from([("kernel.img", &kernel[..]), ("devicetree.adt", &tree[..])]);
    let verified = manifest.verify(&|name| files.get(name).copied())?;
    let mut ram = vec![0xff; FRAME_BYTES as usize];
    let mut entropy = |bytes: &mut [u8]| {
        bytes.fill(0x42);
        Ok(())
    };
    // Without a display the template stands, as iBoot leaves it.
    verified.load(&machine(), &mut entropy, &mut |pa, bytes| {
        write_into(&mut ram, pa, bytes)
    })?;
    assert_eq!(&ram[0x4014..0x4020], &[0x11; 12]);
    let screen = Machine {
        framebuffer: Some(Framebuffer {
            guest_pa: 0xc000_0000,
            bytes: 0x80_0000,
            row_bytes: 5120,
            width: 1280,
            height: 800,
            bits_per_pixel: 32,
        }),
        ..machine()
    };
    verified.load(&screen, &mut entropy, &mut |pa, bytes| {
        write_into(&mut ram, pa, bytes)
    })?;
    assert_eq!(&ram[0x4014..0x4018], &0u32.to_le_bytes());
    assert_eq!(&ram[0x4018..0x401c], &1u32.to_le_bytes());
    assert_eq!(&ram[0x401c..0x4020], &0u32.to_le_bytes());
    Ok(())
}

#[test]
fn a_persistent_nor_hands_off_its_current_bank() -> Result<(), BundleError> {
    let kernel = [0xaa; 100];
    let tree = [0x11; 20];
    let json = manifest_with(
        &kernel,
        &tree,
        r#"
    {"source": "nvram", "required": false, "region": "DeviceTree", "file": "devicetree.adt", "file_offset": 0, "bytes": 16, "encoding": "bytes", "field": "/chosen nvram-proxy-data", "prepared_hex": ""},
    {"source": "nvram", "required": false, "region": "DeviceTree", "file": "devicetree.adt", "file_offset": 16, "bytes": 4, "encoding": "bytes", "field": "/chosen nvram-current-bank", "prepared_hex": ""}"#,
    );
    let manifest = Manifest::parse(json.as_bytes())?;
    let files: BTreeMap<&str, &[u8]> =
        BTreeMap::from([("kernel.img", &kernel[..]), ("devicetree.adt", &tree[..])]);
    let verified = manifest.verify(&|name| files.get(name).copied())?;
    let mut ram = vec![0u8; FRAME_BYTES as usize];
    let mut entropy = |_: &mut [u8]| Err(refuse!("unused"));
    // Without a persistent NOR the prepared bank stands.
    let image = verified.load(&machine(), &mut entropy, &mut |pa, bytes| {
        write_into(&mut ram, pa, bytes)
    })?;
    assert_eq!(image.kept.len(), 2);
    assert_eq!(&ram[0x4000..0x4014], &[0x11; 20]);
    let persistent = Machine {
        nvram: Some(Nvram {
            bank: 1,
            proxy: vec![7; 16],
        }),
        ..machine()
    };
    verified.load(&persistent, &mut entropy, &mut |pa, bytes| {
        write_into(&mut ram, pa, bytes)
    })?;
    assert_eq!(&ram[0x4000..0x4010], &[7; 16]);
    assert_eq!(&ram[0x4010..0x4014], &1u32.to_le_bytes());
    let short = Machine {
        nvram: Some(Nvram {
            bank: 0,
            proxy: vec![7; 4],
        }),
        ..machine()
    };
    assert!(
        verified
            .load(&short, &mut entropy, &mut |_, _| true)
            .is_err()
    );
    Ok(())
}

#[test]
fn the_framebuffer_ends_the_managed_range() {
    let screen = |guest_pa| Machine {
        framebuffer: Some(Framebuffer {
            guest_pa,
            bytes: 0x4000,
            row_bytes: 64,
            width: 16,
            height: 16,
            bits_per_pixel: 32,
        }),
        ..machine()
    };
    let end = BASE + FRAME_BYTES;
    assert_eq!(machine().managed_end(), Ok(end));
    assert_eq!(screen(end - 0x4000).managed_end(), Ok(end - 0x4000));
    assert!(screen(BASE).managed_end().is_err());
}

#[test]
fn refuses_tampered_files_small_ram_and_missing_entropy() -> Result<(), BundleError> {
    let kernel = [0xaa; 100];
    let tree = [0x11; 20];
    let json = manifest_json(&kernel, &tree);
    let manifest = Manifest::parse(json.as_bytes())?;
    let tampered = [0xab; 100];
    let files: BTreeMap<&str, &[u8]> =
        BTreeMap::from([("kernel.img", &tampered[..]), ("devicetree.adt", &tree[..])]);
    // The kernel is tiled by its region, whose hash catches the change.
    assert!(matches!(
        manifest.verify(&|name| files.get(name).copied()),
        Err(BundleError::Region { .. })
    ));
    let files: BTreeMap<&str, &[u8]> =
        BTreeMap::from([("kernel.img", &kernel[..]), ("devicetree.adt", &tree[..])]);
    let verified = manifest.verify(&|name| files.get(name).copied())?;
    let mut write = |_: u64, _: &[u8]| true;
    let mut no_entropy = |_: &mut [u8]| Err(refuse!("no RDSEED or RDRAND"));
    assert_eq!(
        verified.load(&machine(), &mut no_entropy, &mut write).err(),
        Some(BundleError::Entropy(refuse!("no RDSEED or RDRAND")))
    );
    let small = Machine {
        ram_bytes: 0x4000,
        ..machine()
    };
    let mut entropy = |_: &mut [u8]| Ok(());
    assert!(matches!(
        verified.load(&small, &mut entropy, &mut write),
        Err(BundleError::Machine(_))
    ));
    // A file no region covers is checked by its own hash.
    let extra = json.replace(
        r#"  ],
  "regions""#,
        &format!(
            r#",    {{"name": "extra.bin", "bytes": 3, "sha256": "{}"}}
  ],
  "regions""#,
            crate::sha256(b"abc")
        ),
    );
    let extra = Manifest::parse(extra.as_bytes())?;
    let files: BTreeMap<&str, &[u8]> = BTreeMap::from([
        ("kernel.img", &kernel[..]),
        ("devicetree.adt", &tree[..]),
        ("extra.bin", &b"abd"[..]),
    ]);
    assert!(matches!(
        extra.verify(&|name| files.get(name).copied()),
        Err(BundleError::File { .. })
    ));
    let unsafe_name = json.replace("\"kernel.img\"", "\"../kernel.img\"");
    assert!(Manifest::parse(unsafe_name.as_bytes()).is_err());
    assert!(
        Manifest::parse(
            json.replace("\"schema_version\": 4", "\"schema_version\": 2")
                .as_bytes()
        )
        .is_err()
    );
    Ok(())
}
