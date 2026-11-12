use super::*;
use crate::aux_kc::tests::{InputSection, auth, defined, object_file_of, undefined, words};
use crate::chained_fixups::{DecodedFixups, decode_authenticated};
use crate::load_plan::plan_bundle;
use object::read::macho::{MachHeader, Section as _, Segment as _};
use stela_contracts::pauth::Key;

const ARM64: u32 = 0;
const ARM64E: u32 = 0x8000_0002;
const CODE: u32 = macho::S_ATTR_PURE_INSTRUCTIONS | macho::S_ATTR_SOME_INSTRUCTIONS;
const IDENTIFIER: &str = "com.stela.Test";
/// `_main`, `_local`, `_printf`, `_imported`, `_class`.
const LOCAL: u32 = 1;
const PRINTF: u32 = 2;
const IMPORTED: u32 = 3;
const CLASS: u32 = 4;
/// Data key A, as an object's authenticated pointer spells it.
const KEY_DA: u64 = 2;

fn code() -> InputSection {
    InputSection::new(
        "__TEXT",
        "__text",
        CODE,
        words(&[
            0x9400_0000, // bl _printf
            0x9000_0000, // adrp x0, _imported@GOTPAGE
            0xf940_0000, // ldr x0, [x0, _imported@GOTPAGEOFF]
            0x9000_0001, // adrp x1, _local@PAGE
            0x9100_0021, // add x1, x1, _local@PAGEOFF
            0xd65f_03c0, // ret
        ]),
    )
    .relocate(0, PRINTF, macho::ARM64_RELOC_BRANCH26, 2)
    .relocate(4, IMPORTED, macho::ARM64_RELOC_GOT_LOAD_PAGE21, 2)
    .relocate(8, IMPORTED, macho::ARM64_RELOC_GOT_LOAD_PAGEOFF12, 2)
    .relocate(12, LOCAL, macho::ARM64_RELOC_PAGE21, 2)
    .relocate(16, LOCAL, macho::ARM64_RELOC_PAGEOFF12, 2)
}

/// `_local` and a pointer to the class, signed in an arm64e slice.
fn data(arm64e: bool) -> InputSection {
    let mut bytes = vec![0; 16];
    let section = if arm64e {
        bytes[8..].copy_from_slice(&auth(KEY_DA, 0x1234));
        InputSection::new("__DATA", "__data", 0, bytes).relocate(
            8,
            CLASS,
            macho::ARM64_RELOC_AUTHENTICATED_POINTER,
            3,
        )
    } else {
        InputSection::new("__DATA", "__data", 0, bytes).relocate(
            8,
            CLASS,
            macho::ARM64_RELOC_UNSIGNED,
            3,
        )
    };
    section.relocate(0, LOCAL, macho::ARM64_RELOC_UNSIGNED, 3)
}

fn object(subtype: u32) -> Vec<u8> {
    object_file_of(
        subtype,
        &[
            code(),
            data(subtype == ARM64E),
            InputSection::new("__DATA", "__objc_imageinfo", 0, words(&[0, 0x40])),
            InputSection::new("__LD", "__compact_unwind", 0, vec![0; 32]),
            InputSection::zerofill("__DATA", "__bss", 0x6000),
        ],
        &[
            defined("_main", true, 0, 0),
            defined("_local", false, 1, 0),
            undefined("_printf", false),
            undefined("_imported", true),
            undefined("_class", false),
        ],
    )
}

fn thin(file: &[u8], index: usize) -> &[u8] {
    let entry = 8 + 20 * index;
    let word = |at: usize| u32::from_be_bytes(file[at..at + 4].try_into().unwrap()) as usize;
    assert_eq!(word(entry + 16), PAGE.trailing_zeros() as usize);
    assert!(word(entry + 8).is_multiple_of(PAGE as usize));
    &file[word(entry + 8)..word(entry + 8) + word(entry + 12)]
}

fn fixups(slice: &[u8]) -> DecodedFixups {
    decode_authenticated(slice, &plan_bundle(slice).unwrap(), 0).unwrap()
}

/// Where each section of `slice` lies: (address, size, file offset).
fn section(slice: &[u8], name: &str) -> (u64, u64, u32) {
    let header = macho::MachHeader64::<LE>::parse(slice, 0).unwrap();
    let mut commands = header.load_commands(LE, slice, 0).unwrap();
    while let Some(command) = commands.next().unwrap() {
        let Some((segment, data)) = command.segment_64().unwrap() else {
            continue;
        };
        for section in segment.sections(LE, data).unwrap() {
            if section.name().starts_with(name.as_bytes()) {
                return (section.addr(LE), section.size(LE), section.offset(LE));
            }
        }
    }
    panic!("no section {name}");
}

fn instruction(slice: &[u8], address: u64) -> u32 {
    let at = address as usize;
    u32::from_le_bytes(slice[at..at + 4].try_into().unwrap())
}

/// What an `adrp` and the 12-bit offset of the instruction after it reach.
fn paged(adrp: u32, place: u64, following: u32, scale: u32) -> u64 {
    let immediate = (adrp >> 29 & 3) | (adrp >> 5 & 0x7_ffff) << 2;
    let page = (place & !0xfff) + (u64::from(immediate) << 12);
    page + (u64::from(following >> 10 & 0xfff) << scale)
}

#[test]
fn a_fat_plugin_holds_each_slice_bound_flat() {
    let (arm64, arm64e) = (object(ARM64), object(ARM64E));
    let file = link(IDENTIFIER, &[vec![&arm64], vec![&arm64e]]).unwrap();
    assert_eq!(file[..4], macho::FAT_MAGIC.to_be_bytes());
    assert_eq!(file[4..8], 2u32.to_be_bytes());

    for (index, subtype) in [ARM64, ARM64E].into_iter().enumerate() {
        let slice = thin(&file, index);
        let header = macho::MachHeader64::<LE>::parse(slice, 0).unwrap();
        assert_eq!(header.cpusubtype(LE), subtype);
        assert_eq!(header.filetype(LE), macho::MH_BUNDLE);
        let fixups = fixups(slice);
        let names: Vec<_> = fixups.imports.iter().map(|import| &import.name).collect();
        assert_eq!(names, ["_class", "_imported", "_printf"]);
        assert!(
            fixups
                .imports
                .iter()
                .all(|import| import.library_ordinal == -2)
        );
        assert!(fixups.imports[1].weak && !fixups.imports[2].weak);
        let (local, ..) = section(slice, "__data");
        assert_eq!(fixups.rebases.len(), 1);
        assert_eq!(fixups.rebases[0].guest_address, local);
        assert_eq!(fixups.rebases[0].value, local);
        // `__objc_imageinfo` and the zero fill stay; the unwind data does not.
        section(slice, "__objc_imageinfo");
        let (bss, size, offset) = section(slice, "__bss");
        assert_eq!((size, offset), (0x6000, 0));
        assert!(bss >= local);
        assert!(
            !slice
                .windows(16)
                .take(0x1000)
                .any(|window| window.starts_with(b"__compact_unwind"))
        );
    }
}

#[test]
fn linkedit_tables_are_pointer_aligned_after_variable_length_imports() {
    let mut residues = std::collections::BTreeSet::new();
    for subtype in [ARM64, ARM64E] {
        for import in [
            "_a",
            "_ab",
            "_abc",
            "_abcd",
            "_abcde",
            "_abcdef",
            "_abcdefg",
            "_abcdefgh",
        ] {
            let object = object_file_of(
                subtype,
                &[
                    InputSection::new("__TEXT", "__text", CODE, words(&[0x9400_0000, 0xd65f_03c0]))
                        .relocate(0, 1, macho::ARM64_RELOC_BRANCH26, 2),
                ],
                &[defined("_main", true, 0, 0), undefined(import, false)],
            );
            let file = link(IDENTIFIER, &[vec![&object]]).unwrap();
            let slice = thin(&file, 0);
            let header = macho::MachHeader64::<LE>::parse(slice, 0).unwrap();
            let mut commands = header.load_commands(LE, slice, 0).unwrap();
            let mut fixups_end = None;
            let mut strings = None;
            while let Some(command) = commands.next().unwrap() {
                if command.cmd() == macho::LC_DYLD_CHAINED_FIXUPS {
                    let data: &macho::LinkeditDataCommand<LE> = command.data().unwrap();
                    fixups_end = Some(data.dataoff.get(LE) + data.datasize.get(LE));
                }
                if let Some(table) = command.symtab().unwrap() {
                    assert_eq!(table.nsyms.get(LE), 0);
                    assert!(table.symoff.get(LE).is_multiple_of(8));
                    assert!(table.stroff.get(LE).is_multiple_of(8));
                    assert_eq!(table.strsize.get(LE), 1);
                    strings = Some(table.stroff.get(LE));
                }
            }
            let end = fixups_end.unwrap();
            residues.insert(end % 8);
            let strings = strings.unwrap();
            assert!(strings >= end && strings - end < 8);
            assert!(
                slice[end as usize..=strings as usize]
                    .iter()
                    .all(|&byte| byte == 0)
            );
            code_hashes(&file).unwrap();
        }
    }
    assert_eq!(residues, (0..8).collect());
}

#[test]
fn arm64_reaches_imports_through_plain_stubs_and_the_got() {
    let object = object(ARM64);
    let file = link(IDENTIFIER, &[vec![&object]]).unwrap();
    let slice = thin(&file, 0);
    let fixups = fixups(slice);
    assert!(fixups.authenticated_binds.is_empty());

    let (text, ..) = section(slice, "__text");
    let (stub, size, _) = section(slice, "__stubs");
    assert_eq!(size, STUB_BYTES);
    let branch = instruction(slice, text);
    assert_eq!(text + u64::from(branch & 0x03ff_ffff) * 4, stub);
    let slot = paged(
        instruction(slice, stub),
        stub,
        instruction(slice, stub + 4),
        3,
    );
    assert_eq!(instruction(slice, stub + 8), BR_X16);
    let bind = |address| {
        let bind = fixups
            .binds
            .iter()
            .find(|bind| bind.guest_address == address);
        fixups.imports[bind.unwrap().import_index as usize]
            .name
            .as_str()
    };
    assert_eq!(bind(slot), "_printf");
    let slot = paged(
        instruction(slice, text + 4),
        text + 4,
        instruction(slice, text + 8),
        3,
    );
    assert_eq!(bind(slot), "_imported");
    let local = paged(
        instruction(slice, text + 12),
        text + 12,
        instruction(slice, text + 16),
        0,
    );
    assert_eq!(local, section(slice, "__data").0);
}

#[test]
fn trusted_hashes_name_each_slices_code_directory() {
    let (arm64, arm64e) = (object(ARM64), object(ARM64E));
    let file = link(IDENTIFIER, &[vec![&arm64], vec![&arm64e]]).unwrap();
    let hashes = code_hashes(&file).unwrap();
    assert_eq!(hashes.len(), 2);
    for (index, hash) in hashes.iter().enumerate() {
        let slice = thin(&file, index);
        let header = macho::MachHeader64::<LE>::parse(slice, 0).unwrap();
        let mut commands = header.load_commands(LE, slice, 0).unwrap();
        let mut checked = false;
        while let Some(command) = commands.next().unwrap() {
            if command.cmd() == macho::LC_CODE_SIGNATURE {
                let data: &macho::LinkeditDataCommand<LE> = command.data().unwrap();
                let directory = &slice[data.dataoff.get(LE) as usize + 20..];
                assert_eq!(*hash, Sha256::digest(directory)[..20]);
                checked = true;
            }
        }
        assert!(checked);
    }
}

#[test]
fn bundle_hashes_seal_both_external_files_on_every_slice() {
    let (arm64, arm64e) = (object(ARM64), object(ARM64E));
    let slices = [vec![arm64.as_slice()], vec![arm64e.as_slice()]];
    let seal = BundleSeal::new(b"Info.plist", b"CodeResources");
    let raw = link(IDENTIFIER, &slices).unwrap();
    let bundle = link_bundle(IDENTIFIER, &slices, seal).unwrap();
    let hashes = bundle_code_hashes(&bundle, seal).unwrap();

    assert_eq!(bundle, link_bundle(IDENTIFIER, &slices, seal).unwrap());
    assert_eq!(hashes.len(), 2);
    assert_ne!(hashes, code_hashes(&raw).unwrap());
    assert!(code_hashes(&bundle).is_err());
    assert!(bundle_code_hashes(&raw, seal).is_err());
    assert!(bundle_code_hashes(&bundle, BundleSeal::new(b"different", b"CodeResources")).is_err());
    assert!(bundle_code_hashes(&bundle, BundleSeal::new(b"Info.plist", b"different")).is_err());
    for (index, hash) in hashes.iter().enumerate() {
        let slice = thin(&bundle, index);
        let header = macho::MachHeader64::<LE>::parse(slice, 0).unwrap();
        let mut commands = header.load_commands(LE, slice, 0).unwrap();
        while let Some(command) = commands.next().unwrap() {
            if command.cmd() == macho::LC_CODE_SIGNATURE {
                let data: &macho::LinkeditDataCommand<LE> = command.data().unwrap();
                let directory = &slice[data.dataoff.get(LE) as usize + 20..];
                assert_eq!(*hash, Sha256::digest(directory)[..20]);
                assert_eq!(&directory[24..28], &3u32.to_be_bytes());
                break;
            }
        }
    }
}

#[test]
fn trusted_hashes_refuse_truncation_and_changed_signed_pages() {
    let arm64 = object(ARM64);
    let file = link(IDENTIFIER, &[vec![&arm64]]).unwrap();
    for length in [0, 4, 8, 27, file.len() - 1] {
        assert!(code_hashes(&file[..length]).is_err(), "length {length}");
    }
    let slice = thin(&file, 0);
    let start = slice.as_ptr() as usize - file.as_ptr() as usize;
    let mut changed = file.clone();
    changed[start + 0x1000] ^= 1;
    let error = code_hashes(&changed).unwrap_err();
    assert!(error.to_string().contains("signed page"), "{error}");
    // The slice may not borrow a following slice's signature.
    changed = file.clone();
    changed[20..24].copy_from_slice(&32u32.to_be_bytes());
    assert!(code_hashes(&changed).is_err());
}

#[test]
fn trusted_hashes_refuse_invalid_signature_extents_and_hash_tables() {
    let signed = vec![0; HASH_PAGE as usize + 1];
    let blob = sign(&signed, IDENTIFIER, 0, None);
    code_hash(&signed, &blob, None).unwrap();
    for (at, value) in [
        (8, 2),
        (16, u32::MAX),
        (24, u32::MAX),
        (36, 0),
        (44, 1),
        (48, u32::MAX),
        (52, 1),
    ] {
        let mut changed = blob.clone();
        changed[at..at + 4].copy_from_slice(&value.to_be_bytes());
        assert!(code_hash(&signed, &changed, None).is_err(), "offset {at}");
    }
    let mut changed = blob.clone();
    changed[57] = 1; // SHA-1 cannot be labeled as a SHA-256 trust entry.
    assert!(code_hash(&signed, &changed, None).is_err());
}

#[test]
fn arm64e_signs_its_stubs_and_keeps_each_pointers_signature() {
    let object = object(ARM64E);
    let file = link(IDENTIFIER, &[vec![&object]]).unwrap();
    let slice = thin(&file, 0);
    let fixups = fixups(slice);

    let (text, ..) = section(slice, "__text");
    let (stub, size, _) = section(slice, "__auth_stubs");
    assert_eq!(size, AUTH_STUB_BYTES);
    let branch = instruction(slice, text);
    assert_eq!(text + u64::from(branch & 0x03ff_ffff) * 4, stub);
    let slot = paged(
        instruction(slice, stub),
        stub,
        instruction(slice, stub + 4),
        0,
    );
    assert_eq!(slot, section(slice, "__auth_got").0);
    assert_eq!(instruction(slice, stub + 12), BRAA_X16_X17);
    let signed = |address| {
        fixups
            .authenticated_binds
            .iter()
            .find(|bind| bind.guest_address == address)
            .unwrap()
    };
    let stub_slot = signed(slot);
    assert_eq!(
        fixups.imports[stub_slot.import_index as usize].name,
        "_printf"
    );
    assert_eq!(stub_slot.authentication.key, Key::Ia);
    assert!(stub_slot.authentication.address_diversity);
    assert_eq!(stub_slot.authentication.diversity, 0);

    let class = signed(section(slice, "__data").0 + 8);
    assert_eq!(fixups.imports[class.import_index as usize].name, "_class");
    assert_eq!(class.authentication.key, Key::Da);
    assert_eq!(class.authentication.diversity, 0x1234);
    assert!(class.authentication.address_diversity);
}

#[test]
fn chains_link_each_page_and_only_that_page() {
    let mut pointers = InputSection::new("__DATA", "__data", 0, vec![0; 0x8000]);
    for offset in [0u32, 0x10, 0x3ff8, 0x4000, 0x7ff8] {
        pointers = pointers.relocate(offset, 0, macho::ARM64_RELOC_UNSIGNED, 3);
    }
    for subtype in [ARM64, ARM64E] {
        let object = object_file_of(
            subtype,
            std::slice::from_ref(&pointers),
            &[defined("_target", false, 0, 0x100)],
        );
        let file = link(IDENTIFIER, &[vec![&object]]).unwrap();
        let slice = thin(&file, 0);
        let (data, ..) = section(slice, "__data");
        let fixups = fixups(slice);
        let found: Vec<_> = fixups
            .rebases
            .iter()
            .map(|rebase| rebase.guest_address - data)
            .collect();
        assert_eq!(found, [0, 0x10, 0x3ff8, 0x4000, 0x7ff8]);
        assert!(
            fixups
                .rebases
                .iter()
                .all(|rebase| rebase.value == data + 0x100)
        );
    }
}

#[test]
fn the_signature_is_ad_hoc_and_hashes_the_file() {
    let object = object(ARM64E);
    let file = link(IDENTIFIER, &[vec![&object]]).unwrap();
    let slice = thin(&file, 0);
    let signature = stela_formats::code_signature::read(slice).unwrap().unwrap();
    assert_eq!(signature.identifier.as_deref(), Some(IDENTIFIER));
    assert_eq!(signature.hash_type, CS_HASHTYPE_SHA256);
    assert_eq!(signature.flags, CS_ADHOC | CS_LINKER_SIGNED);

    let header = macho::MachHeader64::<LE>::parse(slice, 0).unwrap();
    let mut commands = header.load_commands(LE, slice, 0).unwrap();
    let mut blob = None;
    while let Some(command) = commands.next().unwrap() {
        if command.cmd() == macho::LC_CODE_SIGNATURE {
            let raw: &macho::LinkeditDataCommand<LE> = command.data().unwrap();
            blob = Some((raw.dataoff.get(LE) as usize, raw.datasize.get(LE) as usize));
        }
    }
    let (offset, bytes) = blob.unwrap();
    assert_eq!(offset + bytes, slice.len());
    let hashes = &slice[offset + 20 + CODE_DIRECTORY_BYTES + IDENTIFIER.len() + 1..];
    for (index, page) in slice[..offset].chunks(HASH_PAGE as usize).enumerate() {
        let hash = &hashes[index * HASH_BYTES..(index + 1) * HASH_BYTES];
        assert_eq!(hash, Sha256::digest(page).as_slice());
    }
}

#[test]
fn a_weak_definition_yields_to_the_strong_one_and_objects_share_one_image_info() {
    let info = || InputSection::new("__DATA", "__objc_imageinfo", 0, words(&[0, 0x40]));
    let pointer = |symbol| {
        InputSection::new("__DATA", "__data", 0, vec![0; 8]).relocate(
            0,
            symbol,
            macho::ARM64_RELOC_UNSIGNED,
            3,
        )
    };
    let mut weak = defined("_shared", true, 0, 0);
    weak.weak = true;
    let first = object_file_of(ARM64, &[pointer(0), info()], &[weak]);
    let second = object_file_of(
        ARM64,
        &[pointer(0), info()],
        &[defined("_shared", true, 0, 0)],
    );
    let file = link(IDENTIFIER, &[vec![&first, &second]]).unwrap();
    let slice = thin(&file, 0);
    let (data, ..) = section(slice, "__data");
    assert_eq!(section(slice, "__objc_imageinfo").1, 8);
    let fixups = fixups(slice);
    assert_eq!(fixups.rebases.len(), 2);
    assert!(fixups.rebases.iter().all(|rebase| rebase.value == data + 8));

    let twice = link(IDENTIFIER, &[vec![&second, &second]]);
    assert!(twice.unwrap_err().to_string().contains("defined twice"));
}

#[test]
fn refuses_what_a_plugin_cannot_hold() {
    // A pointer in the read-only segment.
    let readonly = object_file_of(
        ARM64,
        &[
            InputSection::new("__TEXT", "__const", 0, vec![0; 8]).relocate(
                0,
                0,
                macho::ARM64_RELOC_UNSIGNED,
                3,
            ),
        ],
        &[defined("_self", false, 0, 0)],
    );
    let error = link(IDENTIFIER, &[vec![&readonly]]).unwrap_err();
    assert!(error.to_string().contains("read-only"));

    // Objects of two architectures in a slice.
    let (arm64, arm64e) = (object(ARM64), object(ARM64E));
    let error = link(IDENTIFIER, &[vec![&arm64, &arm64e]]).unwrap_err();
    assert!(error.to_string().contains("architecture"));

    // Two image infos that disagree.
    let info = |flags| {
        object_file_of(
            ARM64,
            &[InputSection::new(
                "__DATA",
                "__objc_imageinfo",
                0,
                words(&[0, flags]),
            )],
            &[],
        )
    };
    let (left, right) = (info(0x40), info(0x42));
    let error = link(IDENTIFIER, &[vec![&left, &right]]).unwrap_err();
    assert!(error.to_string().contains("image info"));

    assert!(link(IDENTIFIER, &[vec![]]).is_err());
}
