//! The kernel boot bundle: `bundle.json` and the files it lists, as prepared
//! on the host by `stela-dev boot prepare` (the iBoot side of the hand-off).
//! `stela_formats::boot_bundle` checks every region's SHA-256, writes the
//! regions and their zero tails into guest DRAM, and fills the patch slots
//! only the running machine knows; this supplies the machine's side: guest
//! RAM through the engine's host, seeded host entropy and the timebase.
//! The format is `stela_formats::boot_bundle`'s.
//!
//! [`read`] runs before boot services end and reads the files into their own
//! `LOADER_DATA` pages, so a restore ramdisk does not have to fit the heap.
//! [`from_disk`] runs after them instead: Stela's own storage drivers read
//! the installed system on a host disk's APFS container, and
//! `stela_loader::installed` builds the same bundle `stela-dev boot prepare
//! --disk` writes, in the heap. [`upgrade_from_disk`] and
//! [`recovery_from_disk`] build a staged update's and a recoveryOS's the
//! same way (`stela_loader::upgrade`, `stela_loader::recovery`).
use crate::Result;
use crate::{
    handoff::{BundleBacking, BundleLayout, BundlePart, File, MAX_FILES, Region},
    layout, output,
    paging::PAGE,
};
use alloc::string::ToString;
use alloc::{boxed::Box, format, string::String, vec, vec::Vec};
use core::{arch::asm, ops::RangeInclusive, ptr::NonNull};
use stela_boot_core::bundle::placement::{Group, plan_file};
pub(crate) use stela_boot_core::bundle::{live_tree, source};
use stela_contracts::refuse;
use stela_drivers::block::{BlockDevice, BlockError};
use stela_engine::machine::Host;
use stela_formats::boot_bundle::Framebuffer;
use stela_formats::boot_bundle::Machine;
use stela_formats::boot_bundle::Manifest;
use stela_formats::boot_bundle::Nvram;
use stela_formats::boot_bundle::{FileData, FilePart, MANIFEST_FILE, MAX_MANIFEST_BYTES};
use stela_formats::payload;
use stela_formats::payload::PayloadManifest;
use stela_formats::payload::ReadAt;
use stela_loader::boot_bundle::{KERNEL_FILE, RAMDISK_FILE};
use stela_loader::{
    apple_device_tree::DeviceTree, boot_bundle::Options, boot_selection::BootVolume, boot_trace,
    installed, recovery, upgrade,
};
use stela_runtime::pc::gpt::Guid;
use uefi::{
    CString16, boot,
    proto::media::{
        file::{Directory, File as _, FileAttribute, FileInfo, FileMode, RegularFile},
        fs::SimpleFileSystem,
    },
};

/// The counter frequency the machine presents in CNTFRQ_EL0.
const TIMEBASE_HZ: u64 = 24_000_000;
pub struct Entry {
    pub pc: u64,
    pub x0: u64,
    /// Verified selected PE in prepared/live ADT order, never source index.
    pub boot_processor: Option<usize>,
    /// Original CLPC class paths with verified hidden admission.
    pub hidden_clpc: Vec<String>,
    pub dram_base: u64,
    pub regions: usize,
    pub patches: usize,
    pub kept: Vec<String>,
    pub hidden_devices: Vec<String>,
    /// Guest RAM after all fixed-offset runtime patches, not its input file.
    pub live_tree: live_tree::LiveTree,
    /// Verified immutable original input and exact selected manifest metadata.
    pub source: source::Intake,
    /// The AMCC CTRR-A region iBoot programs, `(base, bytes)`.
    pub read_only_region: Option<(u64, u64)>,
    /// What iBoot leaves in the cores' lock registers (M4 and later).
    pub core_locks: stela_engine::machine::ctrr::CoreLocks,
}

pub(super) fn contents(file: &File) -> &[u8] {
    source_contents(&file.contents)
}

fn source_contents(region: &Region) -> &[u8] {
    // SAFETY: the loader left every bundle file allocated and mapped. The
    // adopted ranges are borrowed only during bundle verification and load,
    // which skip those writes and finish before guest execution.
    unsafe {
        core::slice::from_raw_parts(
            layout::physical(region.physical) as *const u8,
            region.bytes as usize,
        )
    }
}

fn open(root: &mut Directory, path: &str) -> Result<RegularFile> {
    let path = CString16::try_from(path).map_err(|_| refuse!("bundle path is not UCS-2"))?;
    root.open(&path, FileMode::Read, FileAttribute::empty())
        .map_err(|_| refuse!("cannot open a bundle file"))?
        .into_regular_file()
        .ok_or(refuse!("a bundle path is a directory"))
}

fn size(file: &mut RegularFile) -> Result<u64> {
    Ok(file
        .get_boxed_info::<FileInfo>()
        .map_err(|_| refuse!("cannot read a bundle file's size"))?
        .file_size())
}

/// The part of a large file read between progress updates.
const PROGRESS_STEP: usize = 16 << 20;
const RAMDISK: &str = "RAMDisk";
const LARGE: u64 = crate::tables::LARGE;

#[derive(Clone, Copy)]
struct Alignment {
    guest_offset: u64,
    name: &'static str,
}

/// Firmware pages held privately until their caller commits the read.
struct ReadPages {
    allocation: NonNull<u8>,
    start: NonNull<u8>,
    count: usize,
    length: usize,
    alignment: Option<Alignment>,
}

impl ReadPages {
    fn bytes(&self) -> &[u8] {
        // SAFETY: only a complete read leaves `read_sized`, and these pages
        // remain owned for the duration of the borrow.
        unsafe { core::slice::from_raw_parts(self.start.as_ptr(), self.length) }
    }

    fn region(&self) -> Region {
        Region {
            physical: self.start.as_ptr() as u64,
            bytes: self.length as u64,
        }
    }

    fn retain(self) -> Region {
        let region = self.region();
        core::mem::forget(self);
        region
    }
}

impl Drop for ReadPages {
    fn drop(&mut self) {
        // SAFETY: this is the exact, unpublished allocation made before EBS.
        let released = unsafe { boot::free_pages(self.allocation, self.count) };
        if let Some(alignment) = self.alignment {
            let status = released.map_or_else(|error| error.status(), |()| uefi::Status::SUCCESS);
            output::arguments(format_args!(
                "MEM02 {} release raw={:#x} bytes={:#x} status={status:?}\n",
                alignment.name,
                self.allocation.as_ptr() as usize,
                self.count * PAGE,
            ));
        }
    }
}

fn read_staged(
    root: &mut Directory,
    path: &str,
    sizes: RangeInclusive<u64>,
    alignment: Option<Alignment>,
) -> Result<ReadPages> {
    let mut file = open(root, path)?;
    let bytes = size(&mut file)?;
    if !sizes.contains(&bytes) {
        return Err(refuse!("file size outside configured bound", %path, bytes));
    }
    let pages = read_sized(&mut file, path, bytes, alignment)?;
    if pages.alignment.is_some() {
        check_file_end(&mut file)?;
    }
    Ok(pages)
}

fn check_file_end(file: &mut RegularFile) -> Result<()> {
    if file
        .read(&mut [0])
        .map_err(|_| refuse!("cannot check the bundle file end"))?
        != 0
    {
        return Err(refuse!("the bundle file grew during the read"));
    }
    Ok(())
}

/// Preserve the restore inventory's per-file retention contract.
fn read_bounded(root: &mut Directory, path: &str, sizes: RangeInclusive<u64>) -> Result<Region> {
    read_staged(root, path, sizes, None).map(ReadPages::retain)
}

/// Read a size-checked file into private `LOADER_DATA` pages.
fn read_sized(
    file: &mut RegularFile,
    path: &str,
    bytes: u64,
    alignment: Option<Alignment>,
) -> Result<ReadPages> {
    let length = usize::try_from(bytes)
        .ok()
        .filter(|&length| length <= isize::MAX as usize)
        .ok_or(refuse!("bundle file is too large"))?;
    let ordinary_count = length.div_ceil(PAGE).max(1);
    let allocate = |count| {
        boot::allocate_pages(
            boot::AllocateType::AnyPages,
            boot::MemoryType::LOADER_DATA,
            count,
        )
    };
    let aligned = alignment.and_then(|alignment| {
        // Both addresses are page-aligned. At most one large page less one
        // ordinary page is needed to make their low bits match.
        let count = ordinary_count + LARGE as usize / PAGE - 1;
        let allocation = match allocate(count) {
            Ok(allocation) => allocation,
            Err(_) => {
                output::arguments(format_args!(
                    "MEM02 {} staged fallback=alignment-allocation\n",
                    alignment.name,
                ));
                return None;
            }
        };
        let raw = allocation.as_ptr() as u64;
        let skip = (alignment.guest_offset % LARGE + LARGE - raw % LARGE) % LARGE;
        // SAFETY: the extra pages cover every possible page-aligned skip.
        let start = unsafe { allocation.add(skip as usize) };
        Some(ReadPages {
            allocation,
            start,
            count,
            length,
            alignment: Some(alignment),
        })
    });
    let pages = match aligned {
        Some(pages) => pages,
        None => {
            let allocation = allocate(ordinary_count)
                .map_err(|_| refuse!("firmware has no memory for Stela"))?;
            ReadPages {
                allocation,
                start: allocation,
                count: ordinary_count,
                length,
                alignment: None,
            }
        }
    };
    // SAFETY: the fresh, exclusively owned allocation covers length bytes
    // after the alignment prefix. Initialize them before creating the slice.
    let buffer = unsafe {
        pages.start.as_ptr().write_bytes(0, length);
        core::slice::from_raw_parts_mut(pages.start.as_ptr(), length)
    };
    // Large files are read in parts, with their progress on the screen.
    let name = path.rsplit('\\').next().unwrap_or(path);
    if length > PROGRESS_STEP {
        crate::status::progress_fraction(&format!("{name}  0 of {} MB", length >> 20), 0, bytes);
    } else {
        // Do not leave the preceding large file's completed bar on a new file.
        crate::status::progress(name);
    }
    let mut done = 0;
    while done < length {
        let end = length.min(done + PROGRESS_STEP);
        let count = file
            .read(&mut buffer[done..end])
            .map_err(|_| refuse!("cannot read a bundle file"))?;
        if count == 0 {
            return Err(refuse!("a bundle file ended early"));
        }
        done += count;
        if length > PROGRESS_STEP {
            crate::status::progress_fraction(
                &format!("{name}  {} of {} MB", done >> 20, length >> 20),
                done as u64,
                bytes,
            );
        }
    }
    Ok(pages)
}

/// Holds split and aligned source allocations until the EBS handoff.
#[derive(Default)]
pub(crate) struct StagedBundle {
    layout: BundleLayout,
    owners: Vec<ReadPages>,
}

impl StagedBundle {
    pub(crate) fn kernel_bytes(&self) -> u64 {
        self.layout
            .kernel_parts()
            .iter()
            .map(|part| part.contents.bytes)
            .sum()
    }

    fn retain_group(&mut self, pages: ReadPages, group: &Group<'_>, kernel_collection: bool) {
        if pages.alignment.is_some() {
            let slot = self.layout.backing_count;
            self.layout.backing[slot] = BundleBacking {
                guest_offset: group.guest_offset + group.head,
                region: Region {
                    physical: pages.region().physical + group.head,
                    bytes: group.interior,
                },
                kernel_collection,
            };
            self.layout.backing_count += 1;
        }
        self.owners.push(pages);
    }

    pub(crate) fn commit(self) -> BundleLayout {
        for pages in self.owners {
            pages.retain();
        }
        self.layout
    }
}

/// Read the exact source tile through one handle, in file-offset order.
fn read_kernel_parts(
    root: &mut Directory,
    path: &str,
    bytes: u64,
    groups: &[Group<'_>],
    staging: &mut StagedBundle,
) -> Result<()> {
    let mut file = open(root, path)?;
    if size(&mut file)? != bytes {
        return Err(refuse!("file size outside configured bound", %path, bytes));
    }
    for group in groups {
        let alignment = (group.interior != 0).then_some(Alignment {
            guest_offset: group.guest_offset,
            name: "BOOTKC",
        });
        let pages = read_sized(&mut file, path, group.bytes, alignment)?;
        for region in &group.regions {
            let offset = (region.file_offset - group.file_offset) as usize;
            let length = region.bytes as usize;
            let contents = &pages.bytes()[offset..offset + length];
            if stela_formats::sha256(contents) != region.sha256 {
                let name = region.name.as_str();
                return Err(refuse!("a BootKC region SHA-256 differs from bundle.json", %name));
            }
            let slot = staging.layout.kernel_count;
            staging.layout.kernel_parts[slot] = BundlePart {
                offset: region.file_offset,
                contents: Region {
                    physical: pages.region().physical + offset as u64,
                    bytes: region.bytes,
                },
            };
            staging.layout.kernel_count += 1;
        }
        staging.retain_group(pages, group, true);
    }
    check_file_end(&mut file)
}

pub(crate) struct Retained {
    pub files: Vec<(String, Region)>,
    pub processors: Option<usize>,
    pub staging: StagedBundle,
}

/// Read `bundle.json` from `directory` on the first volume that has it, then
/// every file it lists, unchanged; `bundle.json` comes first. Run before boot
/// services end, with no other exclusive opening of the volume.
pub(crate) fn read(
    directory: &str,
    verify_topology: bool,
    allow_adoption: bool,
) -> Result<Retained> {
    let directory = directory.trim_end_matches('\\');
    let manifest_path = format!("{directory}\\{MANIFEST_FILE}");
    let handles =
        boot::find_handles::<SimpleFileSystem>().map_err(|_| refuse!("no file systems"))?;
    for handle in handles {
        let Ok(mut volume) = boot::open_protocol_exclusive::<SimpleFileSystem>(handle) else {
            continue;
        };
        let Ok(mut root) = volume.open_volume() else {
            continue;
        };
        let Ok(mut manifest_file) = open(&mut root, &manifest_path) else {
            continue;
        };
        let Ok(bytes) = size(&mut manifest_file) else {
            continue;
        };
        if bytes > MAX_MANIFEST_BYTES as u64 {
            return Err(refuse!("bundle.json exceeds the manifest size limit"));
        }
        let Ok(json) = read_sized(&mut manifest_file, &manifest_path, bytes, None) else {
            continue;
        };
        let manifest = Manifest::parse(json.bytes()).map_err(|error| {
            output::arguments(format_args!("bundle refusal: {error}\n"));
            "bundle.json is not a boot bundle"
        })?;
        if manifest.files.len() >= MAX_FILES {
            return Err(refuse!("bundle lists too many files"));
        }
        let plan = |name| {
            allow_adoption
                .then(|| plan_file(&manifest, name, PAGE as u64, LARGE))
                .flatten()
        };
        let ramdisk = plan(RAMDISK_FILE).filter(|groups| {
            matches!(groups.as_slice(), [group] if group.interior != 0
                && matches!(group.regions.as_slice(), [region] if region.name == RAMDISK))
        });
        // Leave one slot for the RAMDisk if all the kernel regions are adopted.
        let kernel = plan(KERNEL_FILE).filter(|groups| {
            groups.iter().any(|group| group.interior != 0)
                && groups
                    .iter()
                    .map(|group| group.regions.len())
                    .sum::<usize>()
                    < MAX_FILES
        });
        let mut staging = StagedBundle::default();
        let mut files = Vec::with_capacity(manifest.files.len() + 1);
        files.push((String::from(MANIFEST_FILE), json));
        crate::status::begin("Reading the boot files");
        for file in &manifest.files {
            let path = format!("{directory}\\{}", file.name);
            output::arguments(format_args!("READ {path}\n"));
            if let Some(groups) = &kernel
                && file.name == KERNEL_FILE
            {
                read_kernel_parts(&mut root, &path, file.bytes, groups, &mut staging)
                    .inspect_err(|error| crate::status::fail(&error.to_string()))?;
                continue;
            }
            let candidate = ramdisk.as_ref().filter(|_| file.name == RAMDISK_FILE);
            let contents = read_staged(
                &mut root,
                &path,
                file.bytes..=file.bytes,
                candidate.map(|groups| Alignment {
                    guest_offset: groups[0].guest_offset,
                    name: "RAMDISK",
                }),
            )
            .inspect_err(|error| crate::status::fail(&error.to_string()))?;
            if let Some(groups) = candidate
                && contents.alignment.is_some()
                && stela_formats::sha256(contents.bytes()) != groups[0].regions[0].sha256
            {
                return Err(refuse!(
                    "the RAMDisk region SHA-256 differs from bundle.json"
                ));
            }
            files.push((file.name.clone(), contents));
        }
        let processors = verify_topology
            .then(|| {
                source::processor_count(&manifest, &|name| {
                    files
                        .iter()
                        .find(|(file, _)| file == name)
                        .map(|(_, pages)| pages.bytes())
                })
            })
            .transpose()?;
        let bytes = files
            .iter()
            .map(|(_, pages)| pages.length as u64)
            .sum::<u64>()
            + staging.kernel_bytes();
        crate::status::done(&format!("{} MB", bytes >> 20));
        // Reserve the handoff list before relinquishing any page owner.
        let mut retained = Vec::with_capacity(files.len());
        for (name, pages) in files {
            if let Some(groups) = &ramdisk
                && pages.alignment.is_some()
            {
                retained.push((name, pages.region()));
                staging.retain_group(pages, &groups[0], false);
            } else {
                retained.push((name, pages.retain()));
            }
        }
        return Ok(Retained {
            files: retained,
            processors,
            staging,
        });
    }
    Err(refuse!("no volume has the bundle's bundle.json"))
}

/// The restore inventory as it was read, and what it names beside itself.
pub struct Inventory {
    pub manifest: Region,
    /// Explicit opt-in experiment beside the inventory; never factory trust.
    pub fdr_diagnostic: Option<Region>,
    /// Its members' precomputed block hashes (`payload::BlockHashes`), when
    /// it names them.
    pub blocks: Option<Region>,
    /// An install medium's piece block hashes (`Layout::Pieces`).
    pub pieces: Option<Region>,
}

/// The most bytes a hash file may hold: a million blocks' hashes.
const MAX_HASH_FILE_BYTES: u64 = payload::MAX_HASH_BLOCKS * 32;

/// Bounded restore inventory, read before ExitBootServices, with the hash
/// files it names from the same directory of the same volume. Reject
/// ambiguous paths across firmware volumes rather than choosing a different
/// inventory.
pub fn restore_inventory(path: &str) -> Result<Inventory> {
    with_inventory(path, None, |_, _, _| Ok(()))
}

/// [`restore_inventory`], also running `check` on the inventory's volume
/// while it is open, with the inventory's directory. A local install binds
/// this handle to the same GPT partition the native reader will open.
pub fn with_inventory(
    path: &str,
    source: Option<Guid>,
    mut check: impl FnMut(&mut Directory, &str, &PayloadManifest) -> Result<()>,
) -> Result<Inventory> {
    const LIMIT: u64 = 16 * 1024 * 1024;
    let mut found = None;
    for handle in
        boot::find_handles::<SimpleFileSystem>().map_err(|_| refuse!("no file systems"))?
    {
        let Ok(mut volume) = boot::open_protocol_exclusive::<SimpleFileSystem>(handle) else {
            continue;
        };
        let Ok(mut root) = volume.open_volume() else {
            continue;
        };
        if open(&mut root, path).is_err() {
            continue;
        }
        if found.is_some() {
            return Err(refuse!(
                "restore inventory path appears on multiple volumes"
            ));
        }
        if let Some(expected) = source {
            let (actual, _, _) = crate::firmware::esp::gpt_partition(handle).ok_or(refuse!(
                "the install inventory volume has no GPT partition identity"
            ))?;
            if actual != expected {
                output::arguments(format_args!(
                    "INSTALL MEDIUM inventory partition={actual} expected={expected}\n"
                ));
                return Err(refuse!(
                    "the install inventory volume does not match --install-source"
                ));
            }
            output::arguments(format_args!("INSTALL MEDIUM source partition={actual}\n"));
        }
        let manifest = read_bounded(&mut root, path, 0..=LIMIT)?;
        // SAFETY: just read into Stela's own pages.
        let bytes = unsafe {
            core::slice::from_raw_parts(manifest.physical as *const u8, manifest.bytes as usize)
        };
        let parsed = PayloadManifest::parse(bytes).map_err(|error| {
            output::arguments(format_args!("restore inventory refusal: {error}\n"));
            "the restore inventory is not a payload manifest"
        })?;
        let directory = path
            .rsplit_once('\\')
            .map_or("", |(directory, _)| directory);
        check(&mut root, directory, &parsed)?;
        let mut hash_file = |entry: Option<&payload::BlockHashes>| {
            entry
                .map(|entry| {
                    let path = format!("{directory}\\{}", entry.file);
                    output::arguments(format_args!("READ {path}\n"));
                    read_bounded(&mut root, &path, 0..=MAX_HASH_FILE_BYTES).map_err(|_| {
                        refuse!("a hash file the restore inventory names cannot be read")
                    })
                })
                .transpose()
        };
        let blocks = hash_file(parsed.block_hashes.as_ref())?;
        let pieces = hash_file(match &parsed.layout {
            payload::Layout::Pieces { block_hashes, .. } => Some(block_hashes),
            _ => None,
        })?;
        let diagnostic_path = format!("{directory}\\{}", stela_runtime::restore::fdr::FILE);
        let diagnostic_name = CString16::try_from(diagnostic_path.as_str())
            .map_err(|_| refuse!("FDR diagnostic path is not UCS-2"))?;
        let fdr_diagnostic =
            match root.open(&diagnostic_name, FileMode::Read, FileAttribute::empty()) {
                Ok(file) => {
                    drop(file);
                    let seed = read_bounded(
                        &mut root,
                        &diagnostic_path,
                        0..=stela_runtime::restore::fdr::MAX_BYTES as u64,
                    )?;
                    let state_path = format!("{directory}\\{}", super::fdr::STATE_FILE);
                    let image = read_bounded(
                        &mut root,
                        &state_path,
                        0..=stela_runtime::pc::fdr_store::FILE_BYTES as u64,
                    )
                    .map_err(|_| refuse!("FDR diagnostic needs its preallocated fdrstate.bin"))?;
                    let partition = source
                        .map(|guid| guid.0)
                        .or_else(|| super::translations::partition(handle))
                        .ok_or(refuse!("FDR storage volume needs a GPT partition identity"))?;
                    // SAFETY: both regions were just read into kept loader pages.
                    let bytes = |region: Region| unsafe {
                        core::slice::from_raw_parts(
                            region.physical as *const u8,
                            region.bytes as usize,
                        )
                    };
                    let path = state_path.trim_start_matches('\\').replace('\\', "/");
                    Some(
                        super::fdr::prepare(partition, &path, bytes(seed), bytes(image)).map_err(
                            |error| {
                                output::arguments(format_args!("FDR state refusal: {error}\n"));
                                "FDR persisted state could not be loaded"
                            },
                        )?,
                    )
                }
                Err(error) if error.status() == uefi::Status::NOT_FOUND => None,
                Err(_) => return Err(refuse!("cannot open the FDR diagnostic")),
            };
        found = Some(Inventory {
            manifest,
            fdr_diagnostic,
            blocks,
            pieces,
        });
    }
    found.ok_or(refuse!("restore inventory file was not found"))
}

/// The size of the file at `path` of `root`, when it is one.
pub fn file_size(root: &mut Directory, path: &str) -> Result<u64> {
    let mut file = open(root, path)?;
    Ok(file
        .get_boxed_info::<FileInfo>()
        .map_err(|_| refuse!("cannot read a file's size"))?
        .file_size())
}

/// The largest transfer [`Bytes`] asks a device for.
const TRANSFER_BYTES: usize = 1 << 20;

/// Byte-addressed reads over a block device, for the APFS reader.
pub(crate) struct Bytes<D> {
    device: D,
    block: Vec<u8>,
}

impl<D: BlockDevice> Bytes<D> {
    pub(crate) fn new(device: D) -> Self {
        let block = vec![0; device.identity().block_size as usize];
        Self { device, block }
    }

    pub(crate) fn into_inner(self) -> D {
        self.device
    }
}

impl<D: BlockDevice> ReadAt for Bytes<D> {
    type Error = BlockError;

    fn len(&self) -> u64 {
        self.device.identity().bytes()
    }

    fn read_at(&mut self, mut offset: u64, mut buffer: &mut [u8]) -> Result<(), BlockError> {
        let size = self.block.len() as u64;
        while !buffer.is_empty() {
            let lba = offset / size;
            let within = (offset % size) as usize;
            let whole = buffer.len().min(TRANSFER_BYTES) / size as usize * size as usize;
            let done = if within == 0 && whole > 0 {
                self.device.read(lba, &mut buffer[..whole])?;
                whole
            } else {
                self.device.read(lba, &mut self.block)?;
                let count = buffer.len().min(self.block.len() - within);
                buffer[..count].copy_from_slice(&self.block[within..within + count]);
                count
            };
            offset += done as u64;
            buffer = &mut buffer[done..];
        }
        Ok(())
    }
}

/// Build the boot bundle of the installed system on `partition`, an APFS
/// container or a disk image with a GUID partition table, as `request`
/// names it, with `options` (`stela-dev boot prepare`'s defaults and the boot's
/// own boot-args); returns `bundle.json` first and then every file it
/// lists. Runs after boot services end, through Stela's own drivers.
pub fn from_disk(
    partition: Box<dyn BlockDevice>,
    request: &installed::Request,
    options: &Options,
) -> Result<Vec<(String, Vec<u8>)>> {
    let (offset, installed) =
        installed::prepare_disk(&mut Bytes::new(partition), request, options)?;
    output::arguments(format_args!(
        "PREBOOT {} volume={:?} container={offset:#x} objects={} named-by={}\n",
        installed.set.directory(),
        installed.preboot.name,
        installed.set.objects.len(),
        installed
            .evidence
            .named_by
            .as_deref()
            .unwrap_or("evidence alone")
    ));
    let manifest = serde_json::to_vec(&installed.bundle.manifest)
        .map_err(|error| refuse!("bundle.json: {error}", %error))?;
    let mut files = vec![(String::from(MANIFEST_FILE), manifest)];
    files.extend(installed.bundle.files);
    Ok(files)
}

/// Build the boot bundle of the macOS update staged on `partition`'s Update
/// volume (`stela_loader::upgrade`), with `options` plus the update's own
/// boot-args and OS environment; returns `bundle.json` first. Runs after
/// boot services end.
pub fn upgrade_from_disk(
    partition: Box<dyn BlockDevice>,
    volume: Option<BootVolume>,
    options: &Options,
) -> Result<Vec<(String, Vec<u8>)>> {
    let mut options = options.clone();
    options.boot_args = upgrade_boot_args(&options.boot_args);
    options.osenvironment = Some(upgrade::ENVIRONMENT);
    let (offset, staged) =
        upgrade::prepare_disk(&mut Bytes::new(partition), volume, None, &options)?;
    output::arguments(format_args!(
        "UPGRADE {} container={offset:#x} objects={} firmware-set-aside={}\n",
        staged.volume.name,
        staged.update.root.len(),
        staged.update.other_firmware.len()
    ));
    let manifest = serde_json::to_vec(&staged.bundle.manifest)
        .map_err(|error| refuse!("bundle.json: {error}", %error))?;
    let mut files = vec![(String::from(MANIFEST_FILE), manifest)];
    files.extend(staged.bundle.files);
    Ok(files)
}

/// Build the boot bundle of the recoveryOS of `kind` on `partition`
/// (`stela_loader::recovery`): the paired recoveryOS of the boot container's
/// Recovery volume, or System Recovery of the RecoveryOSContainer, as
/// NVRAM's boot `volume` names them; returns `bundle.json` first. Runs
/// after boot services end.
pub fn recovery_from_disk(
    partition: Box<dyn BlockDevice>,
    kind: recovery::Kind,
    volume: Option<BootVolume>,
    options: &Options,
) -> Result<Vec<(String, Vec<u8>)>> {
    let (offset, recovered) = recovery::prepare_disk(
        &mut Bytes::new(partition),
        kind,
        volume.as_ref(),
        None,
        options,
    )?;
    output::arguments(format_args!(
        "RECOVERY {} {} volume={:?} container={offset:#x} objects={} named-by={}\n",
        kind.label(),
        recovered.set.directory(),
        recovered.volume.name,
        recovered.set.objects.len(),
        recovered
            .evidence
            .named_by
            .as_deref()
            .unwrap_or("evidence alone")
    ));
    let manifest = serde_json::to_vec(&recovered.bundle.manifest)
        .map_err(|error| refuse!("bundle.json: {error}", %error))?;
    let mut files = vec![(String::from(MANIFEST_FILE), manifest)];
    files.extend(recovered.bundle.files);
    Ok(files)
}

/// The update ramdisk's boot-args: the console args kept, plus iBoot's
/// `rd=md0`, `-progress` and `-restore` that the restore kexts read.
fn upgrade_boot_args(console: &str) -> String {
    let mut args: Vec<&str> = console.split_ascii_whitespace().collect();
    for extra in upgrade::BOOT_ARGS {
        let key = extra.split('=').next().unwrap_or(extra);
        if !args
            .iter()
            .any(|a| *a == key || a.split('=').next() == Some(key))
        {
            args.push(extra);
        }
    }
    args.join(" ")
}

/// Fill from the boot's checked, seeded cryptographic stream.
pub(super) fn entropy(buffer: &mut [u8]) -> Result<()> {
    crate::random::fill(buffer).map_err(|_| refuse!("host cryptographic entropy is unavailable"))
}

/// The bundle's `bundle.json`, from the files the loader read or a bundle
/// built in the heap.
pub(super) fn manifest(files: &[File], built: Option<&[(String, Vec<u8>)]>) -> Result<Manifest> {
    let json = match built {
        Some(built) => built
            .iter()
            .find(|(name, _)| name == MANIFEST_FILE)
            .map(|(_, bytes)| bytes.as_slice()),
        None => files
            .iter()
            .find(|file| file.name() == MANIFEST_FILE)
            .map(contents),
    }
    .ok_or(refuse!("bundle.json was not read"))?;
    Manifest::parse(json).map_err(|error| refuse!("bundle manifest: {error}", %error))
}

/// The guest's DRAM: `backed` bytes of host RAM from `base`, within the
/// `installed` bytes the Mac reports.
#[derive(Debug, Clone, Copy)]
pub struct Dram {
    pub base: u64,
    pub backed: u64,
    pub installed: u64,
}

/// Load the bundle the loader read into guest DRAM `dram`, with the guest
/// framebuffer when the machine shows one.
pub fn load(
    host: &mut impl Host,
    files: &[File],
    layout: &BundleLayout,
    dram: Dram,
    framebuffer: Option<Framebuffer>,
    nvram: Option<Nvram>,
) -> Result<Entry> {
    let named = |name: &str| files.iter().find(|file| file.name() == name).map(contents);
    let kernel: Vec<_> = layout
        .kernel_parts()
        .iter()
        .map(|part| FilePart {
            offset: part.offset,
            bytes: source_contents(&part.contents),
        })
        .collect();
    load_named(
        host,
        &named,
        &kernel,
        layout.backing(),
        dram,
        framebuffer,
        nvram,
    )
}

/// Load a bundle built in the heap, as [`from_disk`] returns it.
pub fn load_built(
    host: &mut impl Host,
    files: &[(String, Vec<u8>)],
    dram: Dram,
    framebuffer: Option<Framebuffer>,
    nvram: Option<Nvram>,
) -> Result<Entry> {
    let named = |name: &str| {
        files
            .iter()
            .find(|(file, _)| file == name)
            .map(|(_, bytes)| bytes.as_slice())
    };
    load_named(host, &named, &[], &[], dram, framebuffer, nvram)
}

fn load_named<'a>(
    host: &mut impl Host,
    named: &impl Fn(&str) -> Option<&'a [u8]>,
    kernel: &'a [FilePart<'a>],
    backing: &[BundleBacking],
    dram: Dram,
    framebuffer: Option<Framebuffer>,
    nvram: Option<Nvram>,
) -> Result<Entry> {
    let mut trace = boot_trace::Trace::new(TIMEBASE_HZ as u32)
        .map_err(|error| refuse!("boot trace: {error}", %error))?;
    trace_record(host, &mut trace, *b"SVFY")?;
    let json = named(MANIFEST_FILE).ok_or(refuse!("bundle.json was not read"))?;
    let manifest =
        Manifest::parse(json).map_err(|error| refuse!("bundle manifest: {error}", %error))?;
    let verified = manifest
        .verify_files(&|name| {
            if name == KERNEL_FILE && !kernel.is_empty() {
                Some(FileData::Parts(kernel))
            } else {
                named(name).map(FileData::Whole)
            }
        })
        .map_err(|error| refuse!("bundle verification: {error}", %error))?;
    // The new reader contract is checked before any guest RAM load/patch.
    let source = source::Intake::from_verified(json, &verified, named)?;
    trace_record(host, &mut trace, *b"EVFY")?;
    // Use the verified tree region, including its file offset. The trace
    // must stay in the original reserved range below managed memory.
    let trace_region = manifest
        .regions
        .iter()
        .find(|region| region.name == "DeviceTree")
        .map(|region| -> Result<_> {
            let file =
                named(&region.file).ok_or(refuse!("verified DeviceTree file disappeared"))?;
            let start = usize::try_from(region.file_offset)
                .map_err(|_| refuse!("DeviceTree offset overflows"))?;
            let length = usize::try_from(region.bytes)
                .map_err(|_| refuse!("DeviceTree length overflows"))?;
            let end = start
                .checked_add(length)
                .ok_or(refuse!("DeviceTree range wraps"))?;
            let bytes = file
                .get(start..end)
                .ok_or(refuse!("DeviceTree range is out of bounds"))?;
            let tree = DeviceTree::parse(bytes)
                .map_err(|error| refuse!("boot trace tree: {error}", %error))?;
            super::remember_panic_region(&tree);
            let trace = boot_trace::Region::from_tree(&tree)
                .map_err(|error| refuse!("boot trace region: {error}", %error))?;
            if let Some(trace) = trace {
                trace
                    .validate(
                        dram.base,
                        manifest.options.phys_base,
                        manifest
                            .regions
                            .iter()
                            .map(|region| (region.guest_pa, region.memory_bytes)),
                    )
                    .map_err(|error| refuse!("boot trace placement: {error}", %error))?;
            }
            Ok(trace)
        })
        .transpose()?
        .flatten();
    let machine = Machine {
        dram_base: dram.base,
        ram_bytes: dram.backed,
        dram_bytes: dram.installed,
        timebase_hz: TIMEBASE_HZ,
        framebuffer,
        boot_args: None,
        nvram,
    };
    trace_record(host, &mut trace, *b"SLOD")?;
    let mut adopted: Vec<_> = backing
        .iter()
        .map(|part| {
            let start = dram.base + part.guest_offset;
            (start..start + part.region.bytes, part.region.physical, 0u64)
        })
        .collect();
    let image = verified
        .load(&machine, &mut entropy, &mut |pa, data| {
            let Some(top) = pa.checked_add(data.len() as u64) else {
                return false;
            };
            let mut overlaps = adopted
                .iter_mut()
                .filter(|(range, _, _)| !data.is_empty() && pa < range.end && range.start < top);
            let Some((range, physical, used)) = overlaps.next() else {
                return host.write_ram(pa, data);
            };
            if overlaps.next().is_some() {
                return false;
            }
            let start = pa.max(range.start);
            let end = top.min(range.end);
            let head = (start - pa) as usize;
            let tail = (end - pa) as usize;
            let covered = *used + (end - start);
            if covered > range.end - range.start
                || data[head..tail].as_ptr() as u64
                    != layout::physical(*physical + (start - range.start))
            {
                return false;
            }
            // The verified interior already occupies its guest pages.
            if !host.write_ram(pa, &data[..head]) || !host.write_ram(end, &data[tail..]) {
                return false;
            }
            *used = covered;
            true
        })
        .map_err(|error| refuse!("kernel load: {error}", %error))?;
    if adopted
        .iter()
        .any(|(range, _, used)| *used != range.end - range.start)
    {
        return Err(refuse!("verified bundle did not use all adopted bytes"));
    }
    let core_locks = manifest
        .core_locks(&machine)
        .map_err(|error| refuse!("core locks: {error}", %error))?
        .unwrap_or_default();
    trace_record(host, &mut trace, *b"ELOD")?;
    if let Some(region) = trace_region {
        if !host.write_ram(region.base, trace.bytes()) {
            return Err(refuse!("boot trace guest RAM write failed"));
        }
        output::arguments(format_args!(
            "BOOTTRACE format=BTRD origin=stela-native stages=verify,load records=4 hz={TIMEBASE_HZ} pa={:#x} bytes={}\n",
            region.base,
            trace.bytes().len()
        ));
    }
    // The load blanked them before it wrote the regions.
    output::arguments(format_args!(
        "CARVEOUTS blank count={} bytes={:#x}\n",
        manifest.carveouts.len(),
        manifest
            .carveouts
            .iter()
            .map(|carveout| carveout.bytes)
            .sum::<u64>()
    ));
    for region in &image.regions {
        output::arguments(format_args!(
            "REGION {} pa={:#x} bytes={:#x} memory={:#x}\n",
            region.name, region.guest_pa, region.bytes, region.memory_bytes
        ));
    }
    let live_tree = live_tree::LiveTree::capture(host, &image.regions, image.x0)?;
    source.validate_live(live_tree.bytes())?;
    Ok(Entry {
        pc: image.entry_pa,
        x0: image.x0,
        boot_processor: Some(source.boot_processor()),
        hidden_clpc: source.hidden_clpc().to_vec(),
        dram_base: dram.base,
        regions: image.regions.len(),
        patches: image.patches.len(),
        kept: image.kept,
        hidden_devices: manifest.hidden_devices,
        live_tree,
        source,
        read_only_region: manifest
            .read_only_region
            .map(|region| (region.base, region.bytes)),
        core_locks,
    })
}

/// Observe the same 24 MHz counter the first guest slice reads. No guest
/// slice has run yet, so no execution lag or idle leap has been charged.
/// The inline native RIP explicitly identifies Stela as the record source.
#[inline(always)]
fn trace_record(host: &mut impl Host, trace: &mut boot_trace::Trace, tag: [u8; 4]) -> Result<()> {
    let pc: u64;
    // SAFETY: obtains this native instruction address without accessing memory.
    unsafe {
        asm!("lea {pc}, [rip + 2f]", "2:", pc = out(reg) pc, options(nomem, nostack, preserves_flags));
    }
    trace
        .record(host.counter(), pc, tag, 0)
        .map_err(|error| refuse!("boot trace record: {error}", %error))
}
