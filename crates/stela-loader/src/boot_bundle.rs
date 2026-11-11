//! The kernel boot bundle: Apple's boot images laid out in guest physical
//! memory, with every value a loader computes already applied, so the UEFI
//! mount only copies regions and patches a few fixed-size fields.
//!
//! The region order, names and auxiliary `/chosen/memory-map` entries follow
//! Asahi m1n1 `proxyclient/m1n1/hv/__init__.py` `load_macos_sptm` (commit
//! `c6a89b908ac4bf1f7b495f20531f3ea029148adb`), which boots unchanged SPTM
//! kernels: the monitors and the kernel collection are split by segment and
//! placed in the order SPTM requires, and execution starts at SPTM's entry
//! with x0 holding the `boot_args` address. `boot_args` values follow m1n1
//! `_write_boot_args`. The format is `stela_formats::boot_bundle`'s.
use crate::Result;
use crate::aic;
use crate::apple_device_tree::DeviceTree;
use crate::boot_args::{self, BootArgs, Video};
use crate::boot_manifests;
use crate::boot_policy::{self, Decision, SealedVolume};
use crate::carveouts::{self, Placed};
use crate::darts;
use crate::devices;
use crate::dvfs;
use crate::gpu_dvfs;
use crate::handoff;
use crate::im4m::{self, Value};
use crate::kernel_collection::{KernelCollection, PAGE_BYTES};
use crate::lock_regs;
use crate::nvram;
use crate::root_selection;
use crate::soc_variant;
use crate::trust_cache;
use stela_contracts::{ctrr, refuse, require, tag_storage};

use stela_formats::sha256;
use stela_soc_data::board;

pub use {
    stela_formats::boot_bundle::schema::BootCpu,
    stela_formats::boot_bundle::schema::BootCpuSelection,
    stela_formats::boot_bundle::schema::Encoding, stela_formats::boot_bundle::schema::Entry,
    stela_formats::boot_bundle::schema::File, stela_formats::boot_bundle::schema::Image,
    stela_formats::boot_bundle::schema::Manifest, stela_formats::boot_bundle::schema::Options,
    stela_formats::boot_bundle::schema::OverlayProperty, stela_formats::boot_bundle::schema::Patch,
    stela_formats::boot_bundle::schema::PatchSource,
    stela_formats::boot_bundle::schema::PlaceholderDecision,
    stela_formats::boot_bundle::schema::Region, stela_formats::boot_bundle::schema::SCHEMA_VERSION,
    stela_formats::boot_bundle::schema::Source, stela_formats::boot_bundle::schema::Target,
};
mod device_tree;
mod layout;
mod patches;
use device_tree::*;
use layout::*;
use patches::*;

/// The revision of the content [`prepare`] writes within its schema; a
/// bundle of another revision is prepared again (`stela setup update stela`).
pub const REVISION: u32 = 39;
/// m1n1 `_write_boot_args`: the physical aperture base XNU expects, offset
/// by `phys_base` modulo 32 MiB.
const VIRT_BASE: u64 = 0xfffffe0010000000;
const VIRT_BASE_ALIGNMENT: u64 = 32 << 20;
/// m1n1 reserves 16 KiB for `boot_args`.
const BOOT_ARGS_REGION_BYTES: u64 = 0x4000;
const MEMORY_MAP: &str = "/chosen/memory-map";
const RESERVED_ENTRY_PREFIX: &str = "MemoryMapReserved-";
/// iBoot fill-in recipe sources: `macaddr/NAME`, `syscfg/KEY[/LEN]`,
/// `syscfg-mismatch/KEY[/LEN]`, `zeroes/LEN` and `hex/BYTES[/LEN]`, tried in
/// order within a comma-separated recipe (26A434 `iBoot.j473` resolver
/// 0xd41cc..0xd42bc, names at 0x1cce79; its `string/` source no template
/// uses).
const MACADDR_RECIPE: &str = "macaddr/";
const SYSCFG_RECIPE: &str = "syscfg/";
/// The same key read from the syscfg keybag providers instead of the syscfg
/// store (`iBoot.j473` 0xd68e4 with its mismatch flag set calls 0x10991c,
/// which asks each provider in turn, instead of 0x1096bc). No keybag provider
/// is modelled, so the source yields nothing.
const SYSCFG_MISMATCH_RECIPE: &str = "syscfg-mismatch/";
const ZERO_FILL_RECIPE: &str = "zeroes/";
/// Literal bytes as hex digits, zero-extended to LEN when given
/// (`iBoot.j473` 0xd442c..0xd4570; an odd digit count or a LEN below the
/// byte count panics, 0xd46ec and 0xd46e0).
const HEX_RECIPE: &str = "hex/";
const MAC_ADDRESS_BYTES: usize = 6;
/// Clock properties XNU `pexpert/arm/pe_identify_machine.c` reads from the
/// running CPU node; absent ones fall back to its defaults. iBoot writes all
/// six to the boot CPU from its clock getter, which returns the 24 MHz
/// reference on every SoC (`stela_soc_data::board::BOOT_CLOCK_HZ`); the
/// timebase is the bundle's configured counter rate.
const BOOT_CPU_CLOCK_PROPERTIES: [&str; 5] = [
    "clock-frequency",
    "memory-frequency",
    "bus-frequency",
    "peripheral-frequency",
    "fixed-frequency",
];
const TIMEBASE_PROPERTY: &str = "timebase-frequency";
/// The rows every other CPU node also carries (Stela's choice: whether
/// iBoot's CPU pass writes them beyond the boot CPU is untraced).
const CPU_FREQUENCY_PROPERTIES: [&str; 3] =
    [TIMEBASE_PROPERTY, "fixed-frequency", "peripheral-frequency"];
const MEMORY_MAP_ENTRY_BYTES: usize = 16;
const HIBERNATION_KEY_BYTES: usize = 0x30;

pub struct Input<'a> {
    pub bytes: &'a [u8],
    pub source: Source,
}

/// The original kernel collection and its source, optionally owned so
/// an installed boot can move the decoded buffer into the bundle.
pub struct KernelInput<'a> {
    pub bytes: Cow<'a, [u8]>,
    pub source: Source,
}

impl<'a> From<Input<'a>> for KernelInput<'a> {
    fn from(input: Input<'a>) -> Self {
        Self {
            bytes: Cow::Borrowed(input.bytes),
            source: input.source,
        }
    }
}

pub struct Inputs<'a> {
    pub target: Target,
    pub kernel_collection: KernelInput<'a>,
    /// Kexts linked against `kernel_collection` (`aux_kc`), which SPTM maps
    /// beside it.
    pub auxiliary_collection: Option<Input<'a>>,
    pub sptm: Input<'a>,
    pub txm: Input<'a>,
    pub device_tree: Input<'a>,
    pub static_trust_cache: Input<'a>,
    /// Copied unchanged: SEP firmware stays encrypted, an `IM4P` or, read
    /// from an installed system, the `IMG4` `seputil` delivers.
    pub sep_firmware: Option<Input<'a>>,
    /// The identity's global (non-personalized) IM4M for the boot objects.
    pub boot_manifest: Option<Input<'a>>,
    /// The identity's global Cryptex1 IM4M.
    pub cryptex1_manifest: Option<Input<'a>>,
    /// Trust cache modules placed after the static one in `TrustCache`.
    pub extra_trust_caches: Vec<Input<'a>>,
    /// A ramdisk, placed as the `RAMDisk` region: a restore ramdisk the
    /// kernel roots from as `md0`.
    pub ramdisk: Option<Input<'a>>,
    /// The sealed volume the boot roots from, and its root-hash `IM4P` when
    /// one is available; `None` when it roots from no sealed volume.
    pub root_hash: Option<RootHash<'a>>,
    /// The BaseSystem's root-hash `IM4P` when the root is another volume. The
    /// 26A434 kernel's `bsd_init` pivots an arm64 volume-group root through
    /// the Recovery volume's BaseSystem image unless `-s` or `-nobsdmgroot`
    /// is given (0xc158334..0xc15836c), and APFS's
    /// `handle_basesystem_rootmount` panics without this record.
    pub pivot_root_hash: Option<Input<'a>>,
    /// The build the boot set's iBoot image names, for `/chosen`'s version
    /// rows; `None` where the boot's objects hold no iBoot.
    pub firmware_version: Option<&'a str>,
    /// Firmware images the board's own devices run, copied unchanged as
    /// bundle files named by their guest file names (such as
    /// `USB-C_HPM,47.bin`); device models read their identity from them.
    pub board_firmware: Vec<(String, Input<'a>)>,
}

impl Inputs<'_> {
    /// The guest DRAM base of the target board, from its DeviceTree or its
    /// SoC (`board::board_dram_base`); `None` when the target names no chip.
    pub fn dram_base(&self) -> Result<Option<u64>> {
        let Some(chip_id) = self.target.chip_id else {
            return Ok(None);
        };
        Ok(Some(stela_soc_data::board::board_dram_base(
            self.device_tree.bytes,
            chip_id,
        )?))
    }
}

/// A sealed root volume, its root-hash `IM4P` and the snapshot it roots from.
pub struct RootHash<'a> {
    pub volume: SealedVolume,
    pub im4p: Option<Input<'a>>,
    pub snapshot: Option<String>,
}

pub struct Bundle {
    /// Output files in manifest order, as `(name, bytes)`.
    pub files: Vec<(String, Vec<u8>)>,
    pub manifest: Manifest,
}

impl Bundle {
    /// Append firmware named by the prepared board's DeviceTree. These
    /// files have source receipts but are not guest memory regions.
    pub fn append_board_firmware(
        &mut self,
        name: String,
        bytes: Vec<u8>,
        source: Source,
    ) -> Result<()> {
        self.append_file(name, bytes, source)
    }

    /// Append a file the engine reads by name (such as the staged signing
    /// answers of an install, `tss-answers.plist`), with its source receipt;
    /// it is not a guest memory region.
    pub fn append_file(&mut self, name: String, bytes: Vec<u8>, source: Source) -> Result<()> {
        require!(
            name.len() <= MAX_FILE_NAME && !self.files.iter().any(|(file, _)| file == &name),
            "bundle file name {name} is too long or already a bundle file",
            ?name
        );
        self.manifest.files.push(File {
            name: name.clone(),
            bytes: bytes.len() as u64,
            sha256: sha256(&bytes),
        });
        self.manifest.sources.push(source);
        self.files.push((name, bytes));
        Ok(())
    }
}

/// Where a split region begins or ends within an image.
#[derive(Clone, Copy)]
enum Mark {
    ImageStart,
    ImageEnd,
    SegmentStart(&'static str),
    SegmentEnd(&'static str),
}

struct Split {
    suffix: &'static str,
    start: Mark,
    end: Mark,
}

const fn split(suffix: &'static str, start: Mark, end: Mark) -> Split {
    Split { suffix, start, end }
}

use Mark::{ImageEnd, ImageStart, SegmentEnd, SegmentStart};
use alloc::{
    borrow::{Cow, ToOwned},
    format,
    string::String,
    vec,
    vec::Vec,
};

/// m1n1 `load_macos_sptm` image splits.
const KERNEL_SPLITS: &[Split] = &[
    split("ro", ImageStart, SegmentStart("__DATA_SPTM")),
    split("rs", SegmentStart("__DATA_SPTM"), SegmentEnd("__DATA_SPTM")),
    split("rx", SegmentStart("__TEXT_EXEC"), SegmentEnd("__TEXT_EXEC")),
    split(
        "bx",
        SegmentStart("__TEXT_BOOT_EXEC"),
        SegmentEnd("__TEXT_BOOT_EXEC"),
    ),
    split("rw", SegmentStart("__PRELINK_INFO"), SegmentEnd("__DATA")),
    split("le", SegmentStart("__LINKEDIT"), ImageEnd),
];
/// An auxiliary collection's writable segments come first, then its
/// read-only ones with the executable `__TEXT_EXEC` last (`aux_kc`).
const AUXILIARY_SPLITS: &[Split] = &[
    split("rw", ImageStart, SegmentStart("__TEXT")),
    split("ro", SegmentStart("__TEXT"), SegmentStart("__TEXT_EXEC")),
    split("rx", SegmentStart("__TEXT_EXEC"), SegmentEnd("__TEXT_EXEC")),
    split("le", SegmentStart("__LINKEDIT"), ImageEnd),
];
const SPTM_SPLITS: &[Split] = &[
    split("ro", ImageStart, SegmentStart("__TEXT_EXEC")),
    split("rx", SegmentStart("__TEXT_EXEC"), SegmentEnd("__TEXT_EXEC")),
    split("rw", SegmentEnd("__TEXT_EXEC"), SegmentStart("__LINKEDIT")),
    split("le", SegmentStart("__LINKEDIT"), ImageEnd),
];
const TXM_SPLITS: &[Split] = &[
    split("ro", ImageStart, SegmentStart("__TEXT_EXEC")),
    split("rx", SegmentStart("__TEXT_EXEC"), SegmentEnd("__TEXT_EXEC")),
    split(
        "bx",
        SegmentStart("__TEXT_BOOT_EXEC"),
        SegmentEnd("__TEXT_BOOT_EXEC"),
    ),
    split(
        "rw",
        SegmentEnd("__TEXT_BOOT_EXEC"),
        SegmentStart("__LINKEDIT"),
    ),
    split("le", SegmentStart("__LINKEDIT"), ImageEnd),
];

/// The physical order SPTM requires (m1n1: "Required by SPTM in this order"),
/// as `(image, split)` or a standalone region name.
const REGION_ORDER: &[&str] = &[
    "TXM-ro",
    "TXM-rx",
    "TXM-bx",
    "TrustCache",
    "AuxKC-ro",
    "AuxKC-rx",
    "BootKC-rx",
    "BootKC-bx",
    "BootKC-ro",
    "BootKC-rs",
    "DeviceTree",
    "SPTM-ro",
    "SPTM-rx",
    "SPTM-rw",
    "SPTM-le",
    "TXM-rw",
    "TXM-le",
    "BootKC-rw",
    "BootKC-le",
    // M4 and later, without a CL4 image (`lock_regs::CL4_DUMMY_PAGE`).
    // SPTM keeps the page out of its physical aperture (as CTXR-B's) and
    // takes the aperture address of AuxKC-le's end (26A434 sptm.t6050
    // file 0xbc7c8..0xbc834; `io_range_get_papt` panics "IO PADDR could not
    // be found" otherwise), so the auxiliary collection cannot end on it.
    lock_regs::CL4_DUMMY_PAGE,
    "AuxKC-rw",
    "AuxKC-le",
    "RAMDisk",
    "SEPFW",
    "BootArgs",
];

/// Independent handoff contract: SPTM's validate_region_order table, raw
/// 0x18728 (16 records), consumed at 0xb9a80..0xb9b74. The boolean marks an
/// optional image. Standalone payloads must not interrupt this sequence.
const SPTM_REGION_ORDER: &[(&str, bool)] = &[
    ("TXM-ro", false),
    ("TXM-rx", false),
    ("TXM-bx", false),
    ("TrustCache", false),
    ("AuxKC-ro", true),
    ("AuxKC-rx", true),
    ("BootKC-rx", false),
    ("BootKC-bx", false),
    ("BootKC-ro", false),
    ("BootKC-rs", false),
    ("CL4-rx", true),
    ("CL4-ro", true),
    ("DeviceTree", false),
    ("SPTM-ro", false),
    ("SPTM-rx", false),
    ("SPTM-rw", false),
];

fn validate_sptm_region_order(regions: &[Region]) -> Result<()> {
    let mut previous: Option<(&str, u64)> = None;
    for &(name, optional) in SPTM_REGION_ORDER {
        let Some(region) = regions.iter().find(|region| region.name == name) else {
            require!(
                optional,
                "SPTM region order: missing required region {name}",
                name
            );
            continue;
        };
        let start = region.guest_pa;
        let end = start
            .checked_add(region.memory_bytes)
            .ok_or_else(|| refuse!("SPTM region order: {name} range overflows", name))?;
        // The original CMN/CCMN checks reject null/sentinel endpoints.
        require!(
            start > 0 && start < u64::MAX - 1 && end > 0 && end < u64::MAX,
            "SPTM region order: {name} has invalid range {start:#x}..{end:#x}",
            name,
            start,
            end
        );
        if let Some((previous_name, previous_end)) = previous {
            require!(
                start == previous_end,
                "SPTM region order: {name} starts at {start:#x}, not immediately after \
                 {previous_name} ending at {previous_end:#x}",
                name,
                start,
                previous_name,
                previous_end
            );
        }
        previous = Some((name, end));
    }
    Ok(())
}

/// SPTM gives XNU the physical slide, physBase up to the first image, as
/// free memory (`PHYS_SLIDE`, registered at raw 0xba7dc of 26A434's SPTM):
/// no region may lie in it.
fn validate_physical_slide(regions: &[Region], phys_base: u64) -> Result<()> {
    let first = SPTM_REGION_ORDER[0].0;
    let start = regions
        .iter()
        .find(|region| region.name == first)
        .ok_or_else(|| refuse!("physical slide: missing region {first}", first))?
        .guest_pa;
    for region in regions {
        let end = region.guest_pa.saturating_add(region.memory_bytes);
        require!(
            end <= phys_base || region.guest_pa >= start,
            "{name} at {guest_pa:#x} lies in the physical slide {phys_base:#x}..{start:#x}, \
             which SPTM gives XNU as free memory",
            name = region.name,
            guest_pa = region.guest_pa,
            phys_base,
            start
        );
    }
    Ok(())
}

/// `SPTM-rm` describes one complete loaded image, including its link-edit
/// split beyond the ordered-region table. SPTM consumes its start/end at
/// raw 0xe076c..0xe0790 and 0xe09f8..0xe0a2c during memory setup.
fn sptm_image_span(regions: &[Region]) -> Result<core::ops::Range<u64>> {
    let mut span = None;
    for name in ["SPTM-ro", "SPTM-rx", "SPTM-rw", "SPTM-le"] {
        let region = regions
            .iter()
            .find(|region| region.name == name)
            .ok_or_else(|| refuse!("SPTM image span: missing region {name}", name))?;
        let end = region
            .guest_pa
            .checked_add(region.memory_bytes)
            .ok_or_else(|| refuse!("SPTM image span: {name} range overflows", name))?;
        match &mut span {
            None => span = Some(region.guest_pa..end),
            Some(span) => {
                require!(
                    region.guest_pa == span.end,
                    "SPTM image span: {name} starts at {guest_pa:#x}, expected {end:#x}",
                    name,
                    guest_pa = region.guest_pa,
                    end = span.end
                );
                span.end = end;
            }
        }
    }
    span.ok_or(refuse!("SPTM image span is empty").into())
}

/// The secure kernel's entries, unset: the board has no exclave image
/// (its identity lists no CL4 component). SPTM 26A434 requires `CL4-entry`,
/// `CL4-rx` and `CL4-virt` (`init_get_image_region`) and starts the secure
/// kernel only when `rx + entry - virt` is not `~0`; with every value the
/// template's unset `(~0, ~0)` it is `~0`, and the `CL4-rx` range it types
/// rounds to no pages.
const ABSENT_SECURE_KERNEL: [&str; 3] = ["CL4-entry", "CL4-rx", "CL4-virt"];

pub const KERNEL_FILE: &str = "kernelcache.img";
const AUXILIARY_FILE: &str = "auxkc.img";
const SPTM_FILE: &str = "sptm.img";
const TXM_FILE: &str = "txm.img";
/// The bundle file holding the prepared DeviceTree.
pub const DEVICE_TREE_FILE: &str = "devicetree.adt";
/// Immutable source metadata, verified as a whole file and never loaded as a region.
pub const ORIGINAL_DEVICE_TREE_FILE: &str = "original-devicetree.adt";
const TRUST_CACHE_FILE: &str = "trustcache.bin";
const BOOT_MANIFEST_FILE: &str = "boot-manifest.im4m";
/// The region of the original boot manifest, in iBoot's loaded-object
/// carveout rather than the memory map (`layout::boot_manifest_region`).
const BOOT_MANIFEST_REGION: &str = "BootManifest";
const SEP_FILE: &str = "sepfw.im4p";
pub const RAMDISK_FILE: &str = "ramdisk.img";
/// The engine handoff's file-name capacity (`stela-boot` `handoff::MAX_NAME`).
const MAX_FILE_NAME: usize = 48;
const BOOT_ARGS_FILE: &str = "bootargs.bin";
const CL4_DUMMY_PAGE_FILE: &str = "cl4-dummypage.bin";

struct LoadedImage {
    name: &'static str,
    file: &'static str,
    collection: KernelCollection,
    memory: Vec<u8>,
    /// The validated file already had this layout and was moved unchanged.
    kept_input: bool,
    /// `(region name, image offset range)` in split order.
    pieces: Vec<(String, core::ops::Range<u64>)>,
}

fn resolve(collection: &KernelCollection, mark: Mark) -> Result<u64> {
    let segment = |name: &str| {
        collection
            .segment(name)
            .ok_or_else(|| refuse!("image has no segment {name}", name))
    };
    Ok(match mark {
        ImageStart => 0,
        ImageEnd => collection.image_bytes(),
        SegmentStart(name) => segment(name)?.vm_address - collection.vm_base,
        SegmentEnd(name) => segment(name)?.vm_end() - collection.vm_base,
    })
}

impl LoadedImage {
    fn new(
        name: &'static str,
        file: &'static str,
        collection: KernelCollection,
        bytes: Cow<'_, [u8]>,
        splits: &[Split],
    ) -> Result<Self> {
        let (memory, kept_input) = match bytes {
            Cow::Owned(bytes) if collection.file_is_memory_image(bytes.len()) => (bytes, true),
            bytes => (collection.memory_image(&bytes)?, false),
        };
        let mut pieces = vec![];
        for split in splits {
            let start = resolve(&collection, split.start)?;
            let end = resolve(&collection, split.end)?;
            require!(
                start < end && start.is_multiple_of(PAGE_BYTES) && end.is_multiple_of(PAGE_BYTES),
                "{name}-{suffix} is not a non-empty page-aligned range",
                name,
                suffix = split.suffix
            );
            pieces.push((format!("{name}-{}", split.suffix), start..end));
        }
        // The pieces must tile the image so that nothing is dropped.
        let mut sorted: Vec<_> = pieces.iter().map(|(_, r)| r.clone()).collect();
        sorted.sort_by_key(|r| r.start);
        let mut cursor = 0;
        for range in sorted {
            require!(
                range.start == cursor,
                "{name} regions leave a gap or overlap at image offset {cursor:#x}",
                name,
                cursor
            );
            cursor = range.end;
        }
        require!(
            cursor == collection.image_bytes(),
            "{name} regions do not reach the image end",
            name
        );
        Ok(Self {
            name,
            file,
            collection,
            memory,
            kept_input,
            pieces,
        })
    }
}

/// Pending region content before placement.
struct Content {
    name: String,
    file: String,
    file_offset: u64,
    bytes: u64,
    memory_bytes: u64,
    virtual_address: Option<u64>,
}

fn page_round(bytes: u64) -> u64 {
    bytes.next_multiple_of(PAGE_BYTES)
}

/// The SEPFW staging bytes for a SEP image: its `IMG4` container, the form
/// the restore host delivers to `seputil` and the largest form `seputil`
/// can pass to `AppleSEPManager::setFirmwareBytes`.
fn sep_firmware_region_bytes(image: &[u8]) -> u64 {
    stela_formats::img4::container_bytes(image.len()) as u64
}

fn range_value(base: u64, length: u64) -> Vec<u8> {
    [base.to_le_bytes(), length.to_le_bytes()].concat()
}

fn parse_size(text: &str, recipe: &str) -> Result<usize> {
    let digits = text.trim_start_matches("0x");
    usize::from_str_radix(digits, 16)
        .map_err(|_| refuse!("bad size {text} in {recipe}", text, recipe).into())
}

/// The value one recipe alternative yields from the supplied board data.
fn alternative_value(
    alternative: &str,
    recipe: &str,
    options: &Options,
) -> Result<Option<Vec<u8>>> {
    if let Some(name) = alternative.strip_prefix(MACADDR_RECIPE) {
        let Some(value) = options.mac_addresses.get(name) else {
            return Ok(None);
        };
        require!(
            value.len() == MAC_ADDRESS_BYTES,
            "MAC address {name} must be {MAC_ADDRESS_BYTES} bytes",
            name,
            MAC_ADDRESS_BYTES
        );
        return Ok(Some(value.clone()));
    }
    if let Some(spec) = alternative.strip_prefix(SYSCFG_RECIPE) {
        let (key, size) = match spec.split_once('/') {
            Some((key, size)) => (key, Some(parse_size(size, recipe)?)),
            None => (spec, None),
        };
        let Some(value) = options.syscfg.get(key) else {
            return Ok(None);
        };
        let mut value = value.clone();
        if let Some(size) = size {
            // iBoot's syscfg resolver copies min(source bytes, recipe size)
            // and zero-fills the remainder (26A434 J473 0xd6a10..0xd6a54).
            value.resize(size, 0);
        }
        return Ok(Some(value));
    }
    if alternative.starts_with(SYSCFG_MISMATCH_RECIPE) {
        return Ok(None);
    }
    if let Some(size) = alternative.strip_prefix(ZERO_FILL_RECIPE) {
        return Ok(Some(vec![0; parse_size(size, recipe)?]));
    }
    if let Some(spec) = alternative.strip_prefix(HEX_RECIPE) {
        let (digits, size) = match spec.split_once('/') {
            Some((digits, size)) => (digits, Some(parse_size(size, recipe)?)),
            None => (spec, None),
        };
        require!(
            digits.len().is_multiple_of(2) && digits.bytes().all(|b| b.is_ascii_hexdigit()),
            "{recipe} does not hold whole hex bytes",
            recipe
        );
        let mut value: Vec<u8> = digits
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[high, low]| {
                let digit = |b: u8| (b as char).to_digit(16).unwrap_or(0) as u8;
                digit(high) << 4 | digit(low)
            })
            .collect();
        if let Some(size) = size {
            require!(
                value.len() <= size,
                "{recipe} holds {value_len} bytes; it allows {size}",
                recipe,
                value_len = value.len(),
                size
            );
            value.resize(size, 0);
        }
        return Ok(Some(value));
    }
    Err(refuse!(
        "unknown fill-in source {alternative} in {recipe}",
        alternative,
        recipe
    )
    .into())
}

/// Resolve every template placeholder as iBoot does: the first recipe
/// alternative with a value wins, from the supplied MAC addresses and syscfg
/// values or a `zeroes/N` fill. A property with no source is removed. Each
/// decision is recorded in the manifest.
fn resolve_placeholders(
    tree: &mut DeviceTree,
    options: &Options,
) -> Result<Vec<PlaceholderDecision>> {
    let mut decisions = vec![];
    for placeholder in tree.placeholders() {
        let mut chosen = None;
        for alternative in placeholder.recipe.split(',') {
            if let Some(value) = alternative_value(alternative, &placeholder.recipe, options)? {
                chosen = Some((alternative.to_owned(), value));
                break;
            }
        }
        let node = tree
            .node_mut(&placeholder.path)
            .ok_or(refuse!("placeholder node disappeared"))?;
        let resolution = match chosen {
            Some((alternative, value)) => {
                let bytes = value.len();
                node.set_property(&placeholder.property, value)?;
                format!("{alternative}: {bytes} bytes")
            }
            None if placeholder.recipe.contains(SYSCFG_MISMATCH_RECIPE) => {
                node.remove_property(&placeholder.property);
                "removed: no listed source was supplied, and no syscfg keybag provider is \
                 modelled for syscfg-mismatch"
                    .into()
            }
            None => {
                node.remove_property(&placeholder.property);
                "removed: no listed source was supplied".into()
            }
        };
        decisions.push(PlaceholderDecision {
            placeholder,
            resolution,
        });
    }
    Ok(decisions)
}

fn source_of(input: &Input) -> Source {
    input.source.clone()
}

/// A ramdisk the bundle takes ownership of, so its file is the caller's
/// buffer rather than a copy: an update ramdisk of hundreds of megabytes
/// then needs no second copy in `stela.efi`'s heap.
pub struct OwnedRamdisk {
    pub bytes: Vec<u8>,
    pub source: Source,
}

pub fn prepare(inputs: &Inputs, options: &Options) -> Result<Bundle> {
    assemble(
        inputs,
        options,
        Cow::Borrowed(&inputs.kernel_collection.bytes),
        None,
    )
}

/// Build an installed bundle from the decoder's owned kernel and source
/// receipt. When the kernel already has the required layout, move its
/// bytes unchanged and reuse the decoder's SHA-256 for the file receipt.
pub(crate) fn prepare_moving_kernel(
    installed: crate::installed::BundleInputs<'_>,
    options: &Options,
) -> Result<Bundle> {
    let mut inputs = installed.into_inputs();
    let kernel = core::mem::take(&mut inputs.kernel_collection.bytes);
    assemble(&inputs, options, kernel, None)
}

/// [`prepare`], with `ramdisk` moved into the bundle; `inputs` must name no
/// ramdisk of its own.
pub fn prepare_owned(inputs: &Inputs, ramdisk: OwnedRamdisk, options: &Options) -> Result<Bundle> {
    require!(
        inputs.ramdisk.is_none(),
        "a bundle has one ramdisk, given both borrowed and owned"
    );
    assemble(
        inputs,
        options,
        Cow::Borrowed(&inputs.kernel_collection.bytes),
        Some(ramdisk),
    )
}

fn assemble(
    inputs: &Inputs,
    options: &Options,
    kernel: Cow<'_, [u8]>,
    owned: Option<OwnedRamdisk>,
) -> Result<Bundle> {
    stela_formats::builds::find(&inputs.target.product_build)?;
    let ramdisk_bytes = owned
        .as_ref()
        .map(|ramdisk| ramdisk.bytes.len())
        .or_else(|| inputs.ramdisk.as_ref().map(|ramdisk| ramdisk.bytes.len()));
    let Placement {
        options,
        carveouts,
        dram_end,
    } = Placement::new(inputs, options)?;
    let options = &options;
    // The M4 and later lock their CTRR-C/D and CTXR-A..D copies at handoff.
    let chip = inputs.target.chip_id.and_then(board::chip);
    let core_locks_handed_off =
        chip.is_some_and(|chip| ctrr::generation(chip) == Some(ctrr::Generation::Ctrr3));
    let images = Images::load(inputs, kernel)?;
    let extra: Vec<&[u8]> = inputs
        .extra_trust_caches
        .iter()
        .map(|input| input.bytes)
        .collect();
    let trust_cache = match extra.as_slice() {
        [] => trust_cache::compose(&[inputs.static_trust_cache.bytes])?,
        [one] => trust_cache::compose(&[inputs.static_trust_cache.bytes, one])?,
        many => {
            trust_cache::compose(&[inputs.static_trust_cache.bytes, &trust_cache::merge(many)?])?
        }
    };
    let mut shaped = Shaped::new(inputs, options, &carveouts)?;
    let mut contents = contents(
        inputs,
        ramdisk_bytes,
        &trust_cache,
        &images,
        core_locks_handed_off,
        &mut shaped.unresolved,
    )?;
    let device_tree_bytes = reserve_memory_map(&mut shaped.tree, &mut contents)?;

    // Place every region back to back from the load base.
    let load_base = options
        .phys_base
        .checked_add(options.load_offset)
        .ok_or(refuse!("load base overflows"))?;
    let (mut regions, top_of_kernel_data) = place(contents, load_base)?;
    validate_sptm_region_order(&regions)?;
    if let (Some(manifest), Some(_)) = (&inputs.boot_manifest, &shaped.boot_manifest_object) {
        regions.insert(
            0,
            boot_manifest_region(&carveouts, manifest.bytes.len() as u64)?,
        );
    }
    validate_physical_slide(&regions, options.phys_base)?;
    require!(
        top_of_kernel_data <= dram_end,
        "regions end at {top_of_kernel_data:#x}, past DRAM end {dram_end:#x}",
        top_of_kernel_data,
        dram_end
    );
    let region = |name: &str| {
        regions
            .iter()
            .find(|r| r.name == name)
            .ok_or_else(|| refuse!("no region {name}", name))
    };
    let read_only_region = shaped
        .memory_controller
        .map(|controller| {
            let sptm_rx = region("SPTM-rx")?;
            lock_regs::ctrr_a(
                controller,
                options.dram_base,
                region("TXM-ro")?.guest_pa,
                sptm_rx.guest_pa + page_round(sptm_rx.memory_bytes),
            )
        })
        .transpose()?;
    fill_memory_map(&mut shaped.tree, &regions, &images, options, load_base)?;
    let core_locks = core_locks_handed_off
        .then(|| {
            // SPTM programs CTRR-D unless the property is a present 0.
            let kernel_ctrr = shaped
                .tree
                .property("/chosen", "kernel-ctrr-to-be-enabled")
                .is_none_or(|p| p.value.iter().any(|&byte| byte != 0));
            lock_regs::core_handoff(
                |name| {
                    let region = regions.iter().find(|r| r.name == name)?;
                    Some(region.guest_pa..region.guest_pa + region.memory_bytes)
                },
                kernel_ctrr,
                tag_offset(chip, options, dram_end, top_of_kernel_data)?,
            )
        })
        .transpose()?;
    if let Some(object) = &shaped.boot_manifest_object {
        let manifest = region(BOOT_MANIFEST_REGION)?;
        shaped
            .security
            .extend(object.place(&mut shaped.tree, manifest.guest_pa)?);
    }
    let tree = &shaped.tree;
    let device_tree = tree.serialize();
    require!(
        device_tree.len() as u64 == device_tree_bytes,
        "DeviceTree size changed after placement"
    );

    let device_tree_region = region("DeviceTree")?;
    let boot_args_region = region("BootArgs")?;
    let boot_args = boot_args_for(
        options,
        dram_end,
        top_of_kernel_data,
        device_tree_region,
        device_tree.len(),
    )?;
    let boot_args_bytes = boot_args::encode(&boot_args)?.to_vec();
    let boot_args_pa = boot_args_region.guest_pa;

    let mut patches = device_tree_patches(tree, device_tree_region)?;
    patches.extend(cpu_frequency_patches(tree, device_tree_region)?);
    if shaped.nvram_published {
        patches.extend(nvram_patches(tree, device_tree_region)?);
    }
    patches.extend(boot_args_patches(boot_args_region, &boot_args_bytes));

    let manifest_images = images.manifest_images(&regions)?;
    let entry = manifest_images[0].entry.clone();
    let kernel_source = &inputs.kernel_collection.source;
    let kernel_digest = (images.kernel.kept_input
        && images.kernel.memory.len() as u64 == kernel_source.decoded_bytes)
        .then_some(&kernel_source.decoded_sha256);
    let (files, ramdisk_source) = files(
        inputs,
        images,
        device_tree,
        trust_cache,
        boot_args_bytes,
        core_locks_handed_off,
        owned,
    );
    hash_regions(&mut regions, &files)?;
    let manifest = Manifest {
        schema_version: SCHEMA_VERSION,
        revision: REVISION,
        target: Some(inputs.target.clone()),
        options: options.clone(),
        boot_cpu: Some(shaped.boot_cpu),
        processors: shaped.processors,
        sources: sources(inputs, ramdisk_source),
        files: files
            .iter()
            .map(|(name, contents)| {
                let bytes = contents.len() as u64;
                // A whole-file region (such as a restore ramdisk) already
                // has the digest of these exact bytes.
                let whole = || {
                    regions
                        .iter()
                        .find(|region| {
                            region.file == *name && region.file_offset == 0 && region.bytes == bytes
                        })
                        .map(|region| &region.sha256)
                };
                let digest = if name == KERNEL_FILE {
                    kernel_digest.or_else(whole)
                } else {
                    whole()
                };
                File {
                    name: name.clone(),
                    bytes,
                    sha256: digest.cloned().unwrap_or_else(|| sha256(contents)),
                }
            })
            .collect(),
        regions,
        entry,
        entry_x0: boot_args_pa,
        images: manifest_images,
        boot_args,
        device_tree_bytes,
        top_of_kernel_data,
        carveouts,
        read_only_region,
        core_locks,
        hidden_devices: shaped.hidden_devices,
        patches,
        placeholders: shaped.placeholders,
        security: shaped.security,
        unresolved: shaped.unresolved,
    };
    let mut bundle = Bundle { files, manifest };
    for (name, image) in &inputs.board_firmware {
        bundle.append_board_firmware(name.clone(), image.bytes.to_vec(), source_of(image))?;
    }
    Ok(bundle)
}

/// A global manifest must name the selected board.
fn check_board(manifest: &im4m::Manifest, target: &Target) -> Result<()> {
    for (name, expected) in [("CHIP", target.chip_id), ("BORD", target.board_id)] {
        if let (Some(Value::Integer(found)), Some(expected)) = (manifest.property(name), expected) {
            require!(
                *found == u64::from(expected),
                "manifest {name} {found:#x} differs from the identity's {expected:#x}",
                name,
                found = *found,
                expected
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
