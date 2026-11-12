use super::*;
use alloc::{boxed::Box, vec};
use serde_json::{Value, json};
use stela_driver_core::firmware::{Provider, Requirement};

fn image() -> Value {
    json!({"name":"Vendor/FW.bin", "path":"vendor/fw.bin",
            "target":{"driver":"realtek","hardware":"10ec:8168","revision":"xid-509","board":null},
            "provenance":{"source":"synthetic-fixture","revision":"v1","member":"fixtures/abc.bin","license":"CC0-1.0"},
            "length":3,"sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"})
}
fn manifest(images: Vec<Value>) -> Vec<u8> {
    serde_json::to_vec(&json!({"schema":1,"images":images})).unwrap()
}

#[test]
fn mount_revision_checks_each_image_without_downgrading_unknown_contracts() {
    let mut legacy = MOUNT_CAPABILITY;
    legacy[20] = 1;
    let mut unknown = MOUNT_CAPABILITY;
    unknown[20] = 5;
    let mut small = MOUNT_CAPABILITY;
    small[20] = 2;
    let mut no_bundle = MOUNT_CAPABILITY;
    no_bundle[20] = 3;
    assert!(reads_bundle(&MOUNT_CAPABILITY));
    assert!(!reads_bundle(&no_bundle) && supports_mount(&no_bundle));
    let current = MountLimits::from_image(&MOUNT_CAPABILITY).unwrap();
    let old = MountLimits::from_image(&legacy).unwrap();
    let small = MountLimits::from_image(&small).unwrap();
    assert!(old.check_lengths([16 * 1024 * 1024]).is_ok());
    assert!(old.check_lengths([16 * 1024 * 1024 + 1]).is_err());
    assert!(current.check_lengths([MAX_IMAGE_BYTES]).is_ok());
    assert!(small.check_lengths([MAX_IMAGE_BYTES, 1]).is_err());
    assert!(current.check_lengths([MAX_IMAGE_BYTES, 1]).is_ok());
    assert!(current.check_lengths([MAX_IMAGE_BYTES; 4]).is_ok());
    assert!(current.check_lengths([MAX_IMAGE_BYTES; 5]).is_err());
    assert!(current.check_lengths([0]).is_err());
    assert!(current.check_lengths([1; MAX_IMAGES + 1]).is_err());
    // Exact measured GB203 lengths, metadata only; no firmware is embedded.
    let lengths = [63_571_696, 200_812, 199_820];
    assert!(current.check_lengths(lengths).is_ok());
    assert!(old.check_lengths(lengths).is_err());
    assert!(
        small
            .check_lengths([63_571_696, 200_812, 199_820, 3_136_536])
            .is_ok()
    );
    assert!(
        small
            .check_lengths([63_571_696, 200_812, 199_820, 3_136_537])
            .is_err()
    );
    for length in [16 * 1024 * 1024, 16 * 1024 * 1024 + 1, MAX_IMAGE_BYTES] {
        let mut item = image();
        item["length"] = json!(length);
        let bytes = manifest(vec![item]);
        let parsed = Manifest::parse(&bytes).unwrap();
        assert!(parsed.check_mount(&MOUNT_CAPABILITY).is_ok());
        assert_eq!(
            parsed.check_mount(&legacy).is_ok(),
            length <= 16 * 1024 * 1024
        );
    }
    assert!(!supports_mount(b""));
    assert!(!supports_mount(&MOUNT_CAPABILITY[..20]));
    assert!(!supports_mount(&unknown));
    let mut mixed = legacy.to_vec();
    mixed.extend_from_slice(&MOUNT_CAPABILITY);
    assert!(
        MountLimits::from_image(&mixed)
            .unwrap()
            .check_lengths(lengths)
            .is_ok()
    );
    mixed.extend_from_slice(&unknown);
    assert!(!supports_mount(&mixed));
}

#[cfg(feature = "std")]
#[test]
fn installed_mount_checks_bounded_metadata_before_efi_replacement() {
    let root = tempfile::tempdir().unwrap();
    let directory = root.path().join("firmware");
    assert!(check_installed_mount(&directory, b"legacy-no-mount").is_ok());
    std::fs::create_dir(&directory).unwrap();
    assert!(check_installed_mount(&directory, b"legacy-no-mount").is_ok());
    let mut item = image();
    item["length"] = json!(63_571_696);
    std::fs::write(directory.join("manifest.json"), manifest(vec![item])).unwrap();
    let mut legacy = MOUNT_CAPABILITY;
    legacy[20] = 1;
    assert!(check_installed_mount(&directory, &legacy).is_err());
    let error = Package::read_for_mount(&directory, &legacy).err().unwrap();
    assert!(error.to_string().contains("image limit"));
    // A compatible mount proceeds to the absent payload and refuses it.
    let error = Package::read_for_mount(&directory, &MOUNT_CAPABILITY)
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("missing host firmware package file")
    );
    assert!(check_installed_mount(&directory, &MOUNT_CAPABILITY).is_ok());
    assert!(check_installed_mount(&directory, b"legacy-no-mount").is_err());
    for bytes in [b"{}".to_vec(), vec![b' '; MAX_MANIFEST_BYTES + 1]] {
        std::fs::write(directory.join("manifest.json"), bytes).unwrap();
        assert!(check_installed_mount(&directory, &MOUNT_CAPABILITY).is_err());
    }
    #[cfg(unix)]
    {
        let path = directory.join("manifest.json");
        std::fs::remove_file(&path).unwrap();
        let external = root.path().join("external.json");
        std::fs::write(&external, manifest(vec![image()])).unwrap();
        std::os::unix::fs::symlink(&external, &path).unwrap();
        assert!(check_installed_mount(&directory, &MOUNT_CAPABILITY).is_err());
    }
}

#[test]
fn additional_permanent_resources_share_one_validation_pass_without_copying() {
    static RAW: [u8; 3] = *b"abc";
    let input = manifest(vec![image()]);
    let parsed = Manifest::parse(&input).unwrap();
    let extra = || Resource {
        name: "platform/raw.bin",
        target: Target {
            driver: "platform",
            hardware: "uefi:test",
            revision: "observed",
            board: None,
        },
        provenance: Provenance {
            source: "synthetic",
            revision: "v1",
            member: "raw.bin",
            license: "CC0-1.0",
        },
        expected_length: 3,
        expected_sha256: parsed.images[0].digest().unwrap(),
        bytes: &RAW[..],
    };
    let catalog = parsed.catalog_with(&[&RAW], vec![extra()]).unwrap();
    let loaded = catalog
        .load(Requirement {
            name: "platform/raw.bin",
            target: extra().target,
            minimum_bytes: 3,
        })
        .unwrap();
    assert_eq!(catalog.inventory(), (2, 6));
    assert_eq!(loaded.bytes().as_ptr(), RAW.as_ptr());
    assert!(
        parsed
            .catalog_with(&[&RAW], vec![extra(), extra()])
            .is_err()
    );
    assert!(
        parsed
            .catalog_with(&[&RAW], (0..MAX_IMAGES).map(|_| extra()).collect())
            .is_err()
    );
    let mut corrupt = extra();
    corrupt.expected_sha256 = [0; 32];
    assert!(parsed.catalog_with(&[&RAW], vec![corrupt]).is_err());
    let mut collision = extra();
    collision.name = parsed.images[0].name;
    collision.target = Target {
        driver: "realtek",
        hardware: "10ec:8168",
        revision: "xid-509",
        board: None,
    };
    assert!(parsed.catalog_with(&[&RAW], vec![collision]).is_err());
}

#[test]
fn permanent_bytes_keep_pointer_identity_across_metadata_reconstruction_and_arc_clones() {
    let bytes = manifest(vec![image()]);
    let parsed = Manifest::parse(&bytes).unwrap();
    let persistent: &'static [u8] = Box::leak(b"abc".to_vec().into_boxed_slice());
    let before = parsed.catalog(&[persistent]).unwrap();
    assert_eq!(before.inventory(), (1, 3));
    drop(before);
    let catalog = alloc::sync::Arc::new(parsed.catalog(&[persistent]).unwrap());
    let copy = catalog.clone();
    assert!(alloc::sync::Arc::ptr_eq(&catalog, &copy));
    let requirement = Requirement {
        name: "Vendor/FW.bin",
        target: Target {
            driver: "realtek",
            hardware: "10ec:8168",
            revision: "xid-509",
            board: None,
        },
        minimum_bytes: 3,
    };
    let loaded = copy.load(requirement).unwrap();
    assert_eq!(loaded.bytes().as_ptr(), persistent.as_ptr());
    assert_eq!(loaded.provenance().license, "CC0-1.0");
    assert!(
        copy.load(Requirement {
            name: "vendor/fw.bin",
            ..requirement
        })
        .is_err()
    );
}

#[test]
fn malformed_unknown_and_duplicate_fields_never_publish_a_manifest() {
    for bytes in [
        b"".as_slice(),
        b"{}",
        b"[]",
        b"{\"schema\":1,\"schema\":1,\"images\":[]}",
        b"{\"schema\":1,\"images\":[],\"fallback\":true}",
        b"{\"schema\":2,\"images\":[]}",
    ] {
        assert!(Manifest::parse(bytes).is_err(), "{bytes:?}");
    }
    let mut input = image();
    input["target"]["fallback"] = json!(true);
    assert!(Manifest::parse(&manifest(vec![input])).is_err());
    let good = String::from_utf8(manifest(vec![image()])).unwrap();
    let duplicate = good.replace("\"length\":3", "\"length\":3,\"length\":3");
    assert!(Manifest::parse(duplicate.as_bytes()).is_err());
}

#[test]
fn all_budgets_and_duplicate_bindings_are_checked_without_image_reads() {
    assert!(Manifest::parse(&vec![b' '; MAX_MANIFEST_BYTES + 1]).is_err());
    let mut input = image();
    input["name"] = json!("x".repeat(MAX_LABEL_BYTES + 1));
    assert!(Manifest::parse(&manifest(vec![input])).is_err());
    let mut input = image();
    input["target"]["revision"] = json!("x".repeat(MAX_LABEL_BYTES + 1));
    assert!(Manifest::parse(&manifest(vec![input])).is_err());
    for length in [0, MAX_IMAGE_BYTES + 1] {
        let mut input = image();
        input["length"] = json!(length);
        assert!(Manifest::parse(&manifest(vec![input])).is_err());
    }
    let mut images = Vec::new();
    for n in 0..=MAX_IMAGES {
        let mut input = image();
        input["path"] = json!(format!("file{n}.bin"));
        input["target"]["revision"] = json!(format!("rev{n}"));
        images.push(input);
    }
    assert!(Manifest::parse(&manifest(images)).is_err());
    let mut images = Vec::new();
    for n in 0..5 {
        let mut input = image();
        input["length"] = json!(MAX_IMAGE_BYTES);
        input["path"] = json!(format!("file{n}.bin"));
        input["target"]["revision"] = json!(format!("rev{n}"));
        images.push(input);
    }
    assert!(Manifest::parse(&manifest(images)).is_err());
    assert!(Manifest::parse(&manifest(vec![image(), image()])).is_err());
    let mut duplicate = image();
    duplicate["path"] = json!("other.bin");
    assert!(Manifest::parse(&manifest(vec![image(), duplicate])).is_err());
    let mut duplicate = image();
    duplicate["target"]["revision"] = json!("other");
    assert!(Manifest::parse(&manifest(vec![image(), duplicate])).is_err());
}

#[test]
fn fat_aliases_and_uefi_path_escapes_are_rejected() {
    for path in [
        "",
        "/x",
        "x/",
        "x//y",
        ".",
        "..",
        "a/../b",
        "./x",
        "a\\b",
        "c:x",
        "a:b",
        "a b",
        "x.",
        "x ",
        "Vendor/fw.bin",
        "vendor/FW.bin",
        "x~1.bin",
        "con",
        "con.bin",
        "aux.bin",
        "lpt1.bin",
        "com9",
        "a\0b",
        "a\nb",
        "é.bin",
        "manifest.json",
    ] {
        let mut input = image();
        input["path"] = json!(path);
        assert!(
            Manifest::parse(&manifest(vec![input])).is_err(),
            "accepted {path:?}"
        );
    }
    assert!(storage_path("vendor/fw-1.2_test.bin"));
}

#[test]
fn corruption_truncation_wrong_counts_and_bad_digests_fail() {
    let bytes = manifest(vec![image()]);
    let parsed = Manifest::parse(&bytes).unwrap();
    assert!(parsed.catalog(&[]).is_err());
    assert!(parsed.catalog(&[b"ab"]).is_err());
    assert!(parsed.catalog(&[b"abcd"]).is_err());
    assert!(parsed.catalog(&[b"abd"]).is_err());
    for hash in [
        "f",
        "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD",
        "zz7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    ] {
        let mut input = image();
        input["sha256"] = json!(hash);
        assert!(Manifest::parse(&manifest(vec![input])).is_err());
    }
}

#[cfg(feature = "std")]
fn package(root: &std::path::Path) -> std::path::PathBuf {
    let source = root.join("source");
    std::fs::create_dir(&source).unwrap();
    std::fs::create_dir(source.join("vendor")).unwrap();
    std::fs::write(source.join("vendor/fw.bin"), b"abc").unwrap();
    std::fs::write(source.join("manifest.json"), manifest(vec![image()])).unwrap();
    source
}

#[cfg(feature = "std")]
#[test]
fn local_package_is_bounded_and_published_with_immutable_objects_before_manifest() {
    let root = tempfile::tempdir().unwrap();
    let source = package(root.path());
    let loaded = Package::read(&source).unwrap();
    assert_eq!(loaded.inventory(), (1, 3));
    let destination = root.path().join("firmware");
    loaded.publish(&destination).unwrap();
    let active = std::fs::read(destination.join("manifest.json")).unwrap();
    let parsed = Manifest::parse(&active).unwrap();
    assert!(parsed.images[0].path.starts_with("objects/"));
    assert_eq!(
        std::fs::read(destination.join(parsed.images[0].path)).unwrap(),
        b"abc"
    );
    assert_eq!(parsed.images[0].name, "Vendor/FW.bin");
    loaded.publish(&destination).unwrap();
    assert_eq!(
        std::fs::read(destination.join("manifest.json")).unwrap(),
        active
    );
    std::fs::write(destination.join(parsed.images[0].path), b"abd").unwrap();
    assert!(loaded.publish(&destination).is_err());
    assert_eq!(
        std::fs::read(destination.join("manifest.json")).unwrap(),
        active
    );
}

#[cfg(feature = "std")]
#[test]
fn local_loader_rejects_file_length_corruption_and_aliases_without_publication() {
    let root = tempfile::tempdir().unwrap();
    let source = package(root.path());
    for contents in [b"ab".as_slice(), b"abcd", b"abd"] {
        std::fs::write(source.join("vendor/fw.bin"), contents).unwrap();
        assert!(Package::read(&source).is_err());
    }
    std::fs::write(source.join("vendor/fw.bin"), b"abc").unwrap();
    std::fs::write(source.join("vendor/FW.BIN"), b"abc").unwrap();
    // A case-insensitive source filesystem has one entry; a case-sensitive
    // source with both spellings is ambiguous and must be rejected.
    if std::fs::read_dir(source.join("vendor")).unwrap().count() > 1 {
        assert!(Package::read(&source).is_err());
    }
    std::fs::write(
        source.join("manifest.json"),
        vec![b' '; MAX_MANIFEST_BYTES + 1],
    )
    .unwrap();
    assert!(Package::read(&source).is_err());
}

#[cfg(all(feature = "std", unix))]
#[test]
fn local_loader_and_publisher_refuse_symlink_escapes() {
    let root = tempfile::tempdir().unwrap();
    let source = package(root.path());
    let external = root.path().join("outside.bin");
    std::fs::write(&external, b"abc").unwrap();
    std::fs::remove_file(source.join("vendor/fw.bin")).unwrap();
    std::os::unix::fs::symlink(&external, source.join("vendor/fw.bin")).unwrap();
    assert!(Package::read(&source).is_err());
    std::fs::remove_file(source.join("vendor/fw.bin")).unwrap();
    std::fs::write(source.join("vendor/fw.bin"), b"abc").unwrap();
    let loaded = Package::read(&source).unwrap();
    let destination = root.path().join("firmware");
    std::fs::create_dir(&destination).unwrap();
    std::os::unix::fs::symlink(root.path(), destination.join("objects")).unwrap();
    assert!(loaded.publish(&destination).is_err());
    assert!(!destination.join("manifest.json").exists());
}

#[cfg(all(feature = "std", unix))]
#[test]
fn interruptions_before_manifest_commit_preserve_old_resources_and_allow_retry() {
    for stop in 0..5 {
        let root = tempfile::tempdir().unwrap();
        let source = package(root.path());
        let old = Package::read(&source).unwrap();
        let destination = root.path().join("firmware");
        old.publish(&destination).unwrap();
        let before = std::fs::read(destination.join("manifest.json")).unwrap();
        let parsed = Manifest::parse(&before).unwrap();
        let previous_object = destination.join(parsed.images[0].path);
        let mut changed = image();
        changed["target"]["revision"] = json!("v2");
        std::fs::write(source.join("manifest.json"), manifest(vec![changed])).unwrap();
        let next = Package::read(&source).unwrap();
        let mut checkpoints = 0;
        assert!(
            next.publish_checkpointed(&destination, |_| {
                let at = checkpoints;
                checkpoints += 1;
                if at == stop {
                    Err(fail("injected interruption"))
                } else {
                    Ok(())
                }
            })
            .is_err()
        );
        assert_eq!(checkpoints, stop + 1);
        assert_eq!(
            std::fs::read(destination.join("manifest.json")).unwrap(),
            before
        );
        assert_eq!(std::fs::read(&previous_object).unwrap(), b"abc");
        assert_eq!(Package::read(&destination).unwrap().inventory(), (1, 3));
        next.publish(&destination).unwrap();
        let after = std::fs::read(destination.join("manifest.json")).unwrap();
        assert_eq!(
            Manifest::parse(&after).unwrap().images[0].target.revision,
            "v2"
        );
        // Committed, the previous package is unreferenced and removed.
        assert!(!previous_object.parent().unwrap().exists());
        assert_eq!(Package::read(&destination).unwrap().inventory(), (1, 3));
    }
}

#[cfg(all(feature = "std", unix))]
#[test]
fn publication_removes_unreferenced_packages_but_keeps_other_entries() {
    let root = tempfile::tempdir().unwrap();
    let source = package(root.path());
    let destination = root.path().join("firmware");
    Package::read(&source)
        .unwrap()
        .publish(&destination)
        .unwrap();
    let objects = destination.join("objects");
    // An orphan from an older publication, an interrupted one's partial
    // directory, and entries publish never creates.
    let orphan = objects.join("8".repeat(64));
    let partial = objects.join("0".repeat(64));
    for directory in [&orphan, &partial] {
        std::fs::create_dir(directory).unwrap();
        std::fs::write(directory.join("0.bin"), vec![0; 4096]).unwrap();
    }
    std::fs::create_dir(objects.join("notes")).unwrap();
    let outside = root.path().join("outside");
    std::fs::create_dir(&outside).unwrap();
    std::fs::write(outside.join("keep.bin"), b"keep").unwrap();
    std::os::unix::fs::symlink(&outside, objects.join("f".repeat(64))).unwrap();
    let mut changed = image();
    changed["target"]["revision"] = json!("v2");
    std::fs::write(source.join("manifest.json"), manifest(vec![changed])).unwrap();
    Package::read(&source)
        .unwrap()
        .publish(&destination)
        .unwrap();
    let active = Manifest::parse(&std::fs::read(destination.join("manifest.json")).unwrap())
        .unwrap()
        .images[0]
        .path
        .split('/')
        .nth(1)
        .unwrap()
        .to_owned();
    let mut left: Vec<_> = std::fs::read_dir(&objects)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    left.sort();
    let mut expected = vec![active, "f".repeat(64), "notes".into()];
    expected.sort();
    assert_eq!(left, expected);
    assert_eq!(std::fs::read(outside.join("keep.bin")).unwrap(), b"keep");
}
