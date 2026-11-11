//! System mode, loader phase: everything that needs the firmware.
//!
//! It runs identity-mapped under the firmware. It keeps a pristine copy of
//! this image (taken before any Rust state exists), chooses the host
//! placement, places a relocated copy in the Stela window, allocates the
//! engine's memory, builds its page tables (five-level under LA57, else
//! four-level), releases the pristine copy, and leaves boot services. It
//! loads the tables through the LA57 trampoline when five-level and calls
//! the engine phase in the window.
use crate::Result;
use crate::{
    console::Framebuffer,
    firmware,
    handoff::{File, Handoff, MAGIC, MAX_FILES, MAX_GUEST_RANGES, MAX_NAME, Region, Storage},
    la57, layout, output, paging,
    selection::Refusal,
    storage,
    tables::{self, Frames, LARGE, NX, PAGE, PRESENT, UNCACHED, WRITABLE, WRITE_COMBINING},
};
use alloc::{format, string::String, vec::Vec};
use core::{arch::asm, ops::Range, ptr::NonNull};
use stela_boot_core::options::{DEFAULT_WIFI_KEY, System};
use stela_contracts::refuse;
use stela_engine::machine::mmu::placement::Placement;
use stela_runtime::pc;
use stela_runtime::pc::gpt::Guid;
use uefi::{
    boot::{self, AllocateType, MemoryType},
    fs::FileSystem,
    mem::memory_map::{MemoryMap, MemoryMapOwned},
    proto::loaded_image::LoadedImage,
};

use crate::native_smp::resources::{Resources as NativeResources, STACK_BYTES, STACK_STRIDE};
mod handoff_phase;
mod memory;
pub(crate) mod memory_map;
const MIB: u64 = 1024 * 1024;
/// The loader heap's free bytes kept past boot services, for what still runs
/// before the engine does; the rest of it goes to the guest.
const LOADER_HEAP_KEPT: usize = 16 * MIB as usize;
const BELOW_4G: u64 = 0xffff_f000;
const RELOCATION_DIRECTORY: usize = 5;

/// The unique successful exit used to seal the native aperture capture.
pub(crate) struct EbsOwnership {
    _private: (),
}

/// Keeps the final memory map and its receipt together on the loader side.
pub(crate) struct SuccessfulExit {
    map: core::mem::ManuallyDrop<MemoryMapOwned>,
    ownership: Option<EbsOwnership>,
}

impl SuccessfulExit {
    pub(crate) fn map(&self) -> &MemoryMapOwned {
        &self.map
    }

    pub(crate) fn take_ownership(&mut self) -> Option<EbsOwnership> {
        self.ownership.take()
    }
}
const IMAGE_REL_BASED_ABSOLUTE: u16 = 0;
const IMAGE_REL_BASED_DIR64: u16 = 10;

/// Private firmware pages owned only while boot services are active.
pub(crate) struct Snapshot {
    copy: NonNull<u8>,
    start: usize,
    bytes: usize,
}

impl Drop for Snapshot {
    fn drop(&mut self) {
        let physical = self.copy.as_ptr() as usize;
        // SAFETY: this owner holds exactly the pages allocated in take_snapshot.
        // The entry and system loader both drop it before exiting boot services.
        let result = unsafe { boot::free_pages(self.copy, self.bytes / PAGE as usize) };
        output::arguments(format_args!(
            "BOOT IMAGE snapshot physical={physical:#x} bytes={:#x} release={}\n",
            self.bytes,
            Released(result),
        ));
    }
}

/// Copy this image as the firmware loaded it, before Rust state changes it.
/// Only the firmware entry glue has run; its statics are not used later.
pub(crate) fn take_snapshot() -> Option<Snapshot> {
    let image = boot::open_protocol_exclusive::<LoadedImage>(boot::image_handle()).ok()?;
    let (base, bytes) = image.info();
    let extent = firmware::loaded_extent(base as usize, bytes).ok()?;
    let bytes = extent.mapped_bytes;
    let copy = boot::allocate_pages(
        AllocateType::AnyPages,
        MemoryType::LOADER_DATA,
        bytes / PAGE as usize,
    )
    .ok()?;
    // SAFETY: exact checked firmware source extent and distinct fresh owned
    // pages. Zero only the destination tail, without reading source padding.
    unsafe {
        core::ptr::copy_nonoverlapping(base.cast::<u8>(), copy.as_ptr(), extent.bytes);
        copy.as_ptr()
            .add(extent.bytes)
            .write_bytes(0, bytes - extent.bytes);
    }
    Some(Snapshot {
        copy,
        start: base as usize,
        bytes,
    })
}

fn low_page(kind: MemoryType) -> Result<u64> {
    let page = boot::allocate_pages(AllocateType::MaxAddress(BELOW_4G), kind, 1)
        .map_err(|_| refuse!("no page below 4 GiB for the page-table root"))?;
    // SAFETY: a fresh page owned by Stela.
    unsafe { page.as_ptr().write_bytes(0, PAGE as usize) };
    Ok(page.as_ptr() as u64)
}

fn allocate(bytes: u64) -> Result<Region> {
    Ok(Region {
        physical: firmware::pages(bytes.div_ceil(PAGE) as usize)? as u64,
        bytes,
    })
}

/// The usable RAM in the current firmware map, measured as in the handoff.
fn memory_map(stage: Option<&str>) -> Result<memory_map::Map> {
    let map = boot::memory_map(MemoryType::LOADER_DATA).map_err(|_| refuse!("no memory map"))?;
    Ok(memory_map::summarize(&map, stage))
}

pub(crate) fn report_initial_map(map: &impl MemoryMap) {
    memory_map::summarize(map, Some("entry"));
}

/// Host-check measures the live map before loading any guest bundle. The
/// profile is an estimate for one worker at that point.
#[cfg(feature = "diagnostics")]
pub(crate) fn diagnostic_memory() -> Result<(Option<stela_boot_core::host_memory::Profile>, String)>
{
    use stela_boot_core::host_memory;
    let map = memory_map(Some("diagnostic"))?.ledger;
    let profile = host_memory::sizes(map.usable, 1, 1);
    let detail = format!(
        "usable={} MiB, minimum={} MiB, one-worker estimate={}",
        map.usable >> 20,
        host_memory::MIN_PHYSICAL >> 20,
        profile
            .as_ref()
            .map_or("refused", |sizes| sizes.profile.name()),
    );
    Ok((profile.ok().map(|sizes| sizes.profile), detail))
}

/// Return the loader heap's unused end to the firmware, whose free ranges
/// the guest's are found from: the heap is sized before anything is known of
/// what the boot will load. A refusal only leaves those pages to Stela.
fn release_loader_heap() -> u64 {
    let Some(released) = crate::ALLOCATOR.release_tail(LOADER_HEAP_KEPT) else {
        return 0;
    };
    let pages = released.len() / PAGE as usize;
    let Some(start) = core::ptr::NonNull::new(released.start as *mut u8) else {
        return 0;
    };
    // SAFETY: LOADER_DATA pages of the heap's firmware allocation, which the
    // heap no longer manages and nothing else refers to.
    let result = unsafe { boot::free_pages(start, pages) };
    let bytes = if result.is_ok() {
        released.len() as u64
    } else {
        0
    };
    output::arguments(format_args!(
        "BOOT HEAP released {}M: {}\n",
        released.len() >> 20,
        Released(result)
    ));
    bytes
}

/// A page release's result in the log: `ok`, or the firmware's status.
struct Released(uefi::Result);

impl core::fmt::Display for Released {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.0 {
            Ok(()) => f.write_str("ok"),
            Err(error) => write!(f, "{:?}", error.status()),
        }
    }
}

/// Whether a firmware memory map entry is RAM: every type but the reserved,
/// unusable, MMIO and PAL code ones.
fn is_ram(ty: MemoryType) -> bool {
    !matches!(
        ty,
        MemoryType::RESERVED
            | MemoryType::UNUSABLE
            | MemoryType::MMIO
            | MemoryType::MMIO_PORT_SPACE
            | MemoryType::PAL_CODE
    )
}

fn read32(base: usize, offset: usize) -> usize {
    // SAFETY: callers stay inside the image copy.
    unsafe { ((base + offset) as *const u32).read_unaligned() as usize }
}

/// Apply the image's DIR64 base relocations for a move by `delta`.
fn relocate(copy: usize, bytes: usize, delta: u64) -> Result<()> {
    let header = read32(copy, 0x3c);
    if read32(copy, header) != 0x4550 {
        return Err(refuse!("image copy is not PE"));
    }
    let optional = header + 24;
    let directory = optional + 112 + RELOCATION_DIRECTORY * 8;
    let (mut block, end) = (
        read32(copy, directory),
        read32(copy, directory) + read32(copy, directory + 4),
    );
    if end > bytes {
        return Err(refuse!("relocation directory outside the image"));
    }
    while block < end {
        let page = read32(copy, block);
        let size = read32(copy, block + 4);
        if size < 8 || block + size > end {
            return Err(refuse!("malformed relocation block"));
        }
        for entry in (block + 8..block + size).step_by(2) {
            // SAFETY: inside the relocation directory checked above.
            let value = unsafe { ((copy + entry) as *const u16).read_unaligned() };
            let offset = page + usize::from(value & 0xfff);
            match value >> 12 {
                IMAGE_REL_BASED_ABSOLUTE => {}
                IMAGE_REL_BASED_DIR64 if offset + 8 <= bytes => {
                    let slot = (copy + offset) as *mut u64;
                    // SAFETY: a relocated pointer inside the copy.
                    unsafe { slot.write_unaligned(slot.read_unaligned().wrapping_add(delta)) };
                }
                _ => return Err(refuse!("unsupported base relocation")),
            }
        }
        block += size;
    }
    Ok(())
}

fn rdtsc() -> u64 {
    let (low, high): (u32, u32);
    // SAFETY: RDTSC has no side effects.
    unsafe { asm!("rdtsc", out("eax") low, out("edx") high, options(nomem, nostack)) };
    (u64::from(high) << 32) | u64::from(low)
}

/// Identity-map RAM RW+NX in 2 MiB pages where whole chunks are RAM, with the
/// listed pages given their own rights.
fn identity(
    frames: &mut dyn Frames,
    root: u64,
    ram: &[(usize, usize)],
    special: &[(u64, u64, u64)],
) -> Result<()> {
    let covered =
        |start: u64, end: u64| ram.iter().any(|r| r.0 as u64 <= start && end <= r.1 as u64);
    let top = ram.iter().map(|r| r.1 as u64).max().unwrap_or(0);
    for chunk in (0..top).step_by(LARGE as usize) {
        let end = chunk + LARGE;
        let has_special = special.iter().any(|s| s.0 < end && chunk < s.1);
        if covered(chunk, end) && !has_special {
            tables::map_large(frames, root, chunk, chunk, PRESENT | WRITABLE | NX)?;
            continue;
        }
        for page in (chunk..end).step_by(PAGE as usize) {
            let rights = if let Some(s) = special.iter().find(|s| s.0 <= page && page < s.1) {
                s.2
            } else if covered(page, page + PAGE) {
                PRESENT | WRITABLE | NX
            } else {
                continue;
            };
            // Page zero stays unmapped.
            if page != 0 {
                tables::map_page(frames, root, page, page, rights)?;
            }
        }
    }
    Ok(())
}

/// Bundle files by name, `bundle.json` first, each in its own pages.
type BundleFiles = Vec<(String, Region)>;

/// The names the restore inventory and its block hashes take among the
/// bundle files the engine receives.
pub const RESTORE_INVENTORY_FILE: &str = "restore-payload.json";
pub const RESTORE_BLOCKS_FILE: &str = "restore-blocks.bin";
pub const INSTALL_PIECES_FILE: &str = "install-pieces.bin";

/// The guest inputs: `source.bin`, or with `--bundle DIR` the directory's
/// `bundle.json` and every file it lists, read unchanged into their own
/// pages. The bundle reader opens the volumes itself, so the configured
/// volume is closed first. With `--preboot SERIAL:GUID` nothing is read: the
/// engine builds the bundle from that disk with the host storage drivers,
/// whose ranges are found here, as they are for every other option that
/// opens a host controller: disks, the NOR store, input and the networks.
fn inputs(options: &[u8], mut volume: FileSystem) -> Result<Inputs> {
    let text = core::str::from_utf8(options).map_err(|_| refuse!("stela.cfg is not UTF-8"))?;
    let system = System::parse(text).map_err(stela_boot_core::options::Error::message)?;
    #[cfg(not(feature = "diagnostics"))]
    if system.diagnostic_option().is_some() {
        return Err(refuse!("diagnostic options require a diagnostics build"));
    }
    // Read while the firmware's FAT driver runs; the engine reports a file
    // it cannot use.
    let wifi_key = system
        .ssid
        .as_ref()
        .and_then(|_| {
            let path = system.wifi_key.as_deref().unwrap_or(DEFAULT_WIFI_KEY);
            crate::firmware::esp::wifi_key(&mut volume, path)
        })
        .unwrap_or_default();
    let host_devices = system.storage.is_some()
        || system.preboot.is_some()
        || system.restore_source.is_some()
        || system.install_source.is_some()
        || system.nor.is_some()
        || system.network_check.is_some()
        || system.network_check_get.is_some()
        || !system.network_check_ranges.is_empty()
        || system.audio_check.is_some()
        || !system.usb_storage.is_empty()
        || system.input
        || system.install.is_some()
        || system.guest_network;
    let storage = if host_devices {
        crate::status::begin("Finding the PCI devices Stela drives itself");
        let before = memory_map(None).ok();
        let found = storage::discover()?;
        if let (Some(before), Ok(after)) = (before, memory_map(None)) {
            output::arguments(format_args!(
                "HOST STORAGE usable_before={:#x} usable_after={:#x}\n",
                before.ledger.usable, after.ledger.usable,
            ));
        }
        crate::status::done("");
        Some(found)
    } else {
        None
    };
    let (source, mut files, bundle_pes, staging) = if system.preboot.is_some() {
        drop(volume);
        (Vec::new(), Vec::new(), None, Default::default())
    } else if let Some(directory) = system.bundle_directory.as_deref() {
        drop(volume);
        let verify_topology = !(system.native_bootstrap_check || system.native_shared_check);
        let bundle = crate::engine::bundle::read(
            directory,
            verify_topology,
            verify_topology && !system.dram_explicit,
        )?;
        (Vec::new(), bundle.files, bundle.processors, bundle.staging)
    } else {
        let source = firmware::source(&mut volume)?;
        drop(volume);
        (source, Vec::new(), None, Default::default())
    };
    let inventory = match (
        system.restore_manifest_path.as_deref(),
        system.install.as_deref(),
    ) {
        (Some(_), Some(_)) => {
            return Err(refuse!(
                "--install and --restore-manifest name one inventory"
            ));
        }
        (Some(path), None) => Some(crate::engine::bundle::restore_inventory(path)?),
        // The install medium's pieces are checked through the firmware's
        // FAT driver now, while it is there; the engine reads them through
        // its own after boot services end.
        (None, Some(path)) => {
            crate::status::begin("Checking the macOS install files on the USB");
            let source = system.install_source;
            let inventory = crate::engine::bundle::with_inventory(
                path,
                source,
                |root, directory, manifest| {
                    crate::engine::install::check_pieces(
                        root,
                        directory,
                        manifest,
                        source.is_some(),
                    )
                },
            )?;
            crate::status::done("");
            Some(inventory)
        }
        (None, None) => None,
    };
    if let Some(inventory) = inventory {
        let reserved = [
            RESTORE_INVENTORY_FILE,
            RESTORE_BLOCKS_FILE,
            INSTALL_PIECES_FILE,
            stela_runtime::restore::fdr::FILE,
        ];
        if files
            .iter()
            .any(|(name, _)| reserved.contains(&name.as_str()))
        {
            return Err(refuse!(
                "a bundle file takes a name the restore inventory reserves"
            ));
        }
        files.push((String::from(RESTORE_INVENTORY_FILE), inventory.manifest));
        if let Some(diagnostic) = inventory.fdr_diagnostic {
            files.push((String::from(stela_runtime::restore::fdr::FILE), diagnostic));
        }
        if let Some(blocks) = inventory.blocks {
            files.push((String::from(RESTORE_BLOCKS_FILE), blocks));
        }
        if let Some(pieces) = inventory.pieces {
            files.push((String::from(INSTALL_PIECES_FILE), pieces));
        }
    }
    if files.len() > MAX_FILES {
        return Err(refuse!("bundle lists too many files"));
    }
    // The Metal device's pipelines, rewritten over the translation store's
    // partition.
    #[cfg(feature = "nvk")]
    if files.len() < MAX_FILES
        && let Some(store) = crate::engine::pipelines::read()
    {
        files.push((String::from(crate::engine::pipelines::FILE), store));
    }
    Ok(Inputs {
        source,
        files,
        bundle_pes,
        staging,
        storage,
        wifi_key,
    })
}

/// Loader-owned source files, verified bundle cardinality and host inputs.
struct Inputs {
    source: Vec<u8>,
    files: BundleFiles,
    bundle_pes: Option<usize>,
    staging: crate::engine::bundle::StagedBundle,
    storage: Option<stela_drivers::Handoff>,
    wifi_key: Vec<u8>,
}

impl Inputs {
    fn report_staged(&self) -> u64 {
        let kernel = self.staging.kernel_bytes();
        let (bundle, files, inventory, pipelines) = self.files.iter().fold(
            (kernel, usize::from(kernel != 0), 0, 0),
            |(bundle, files, inventory, pipelines), (name, region)| match name.as_str() {
                RESTORE_INVENTORY_FILE
                | RESTORE_BLOCKS_FILE
                | INSTALL_PIECES_FILE
                | stela_runtime::restore::fdr::FILE => {
                    (bundle, files, inventory + region.bytes, pipelines)
                }
                #[cfg(feature = "nvk")]
                crate::engine::pipelines::FILE => (bundle, files, inventory, region.bytes),
                _ => (bundle + region.bytes, files + 1, inventory, pipelines),
            },
        );
        let (largest_name, largest_bytes) = self
            .files
            .iter()
            .max_by_key(|(_, region)| region.bytes)
            .map_or(("none", 0), |(name, region)| (name.as_str(), region.bytes));
        let (largest_name, largest_bytes) = if kernel > largest_bytes {
            (stela_loader::boot_bundle::KERNEL_FILE, kernel)
        } else {
            (largest_name, largest_bytes)
        };
        let source = self.source.len() as u64;
        output::arguments(format_args!(
            "HOST STAGED bundle={bundle:#x} files={files} largest={largest_name}:{largest_bytes:#x} \
             inventory={inventory:#x} pipelines={pipelines:#x} source={source:#x} wifi_key={:#x}\n",
            self.wifi_key.len(),
        ));
        bundle + inventory + pipelines + source
    }
}

/// `EFI_MEMORY_RUNTIME`: the region stays the firmware's after boot
/// services, for its runtime services.
const RUNTIME_ATTRIBUTE: u64 = 1 << 63;

/// The firmware's runtime regions, and its `ResetSystem` service: zero when
/// the system table has none.
struct Runtime {
    regions: Vec<(u64, u64, MemoryType)>,
    reset: u64,
}

impl Runtime {
    /// Read before boot services end. Without SetVirtualAddressMap the
    /// firmware's runtime services run at their physical addresses.
    fn capture() -> Result<Self> {
        let map =
            boot::memory_map(MemoryType::LOADER_DATA).map_err(|_| refuse!("no memory map"))?;
        let regions = map
            .entries()
            .filter(|entry| entry.att.bits() & RUNTIME_ATTRIBUTE != 0)
            .map(|entry| {
                let start = entry.phys_start;
                (start, start + entry.page_count * PAGE, entry.ty)
            })
            .collect();
        let reset = uefi::table::system_table_raw().map_or(0, |table| {
            // SAFETY: the firmware's live system table, before EBS.
            let runtime = unsafe { table.as_ref().runtime_services };
            if runtime.is_null() {
                0
            } else {
                // SAFETY: as above; its runtime services table.
                unsafe { (*runtime).reset_system as usize as u64 }
            }
        });
        Ok(Self { regions, reset })
    }

    /// Runtime code, which the identity map makes writable and executable,
    /// as Linux maps it without a memory attributes table.
    fn code(&self) -> impl Iterator<Item = (u64, u64, u64)> + '_ {
        self.regions
            .iter()
            .filter(|&&(_, _, kind)| runtime_code(kind))
            .map(|&(start, end, _)| (start, end, PRESENT | WRITABLE))
    }

    /// Map the runtime regions outside RAM by identity: device registers
    /// uncached, anything else write-back, code RWX and the rest RW+NX. A
    /// region overlapping RAM, which the identity map holds, or the storage
    /// registers, mapped already, is left as it is.
    fn map(
        &self,
        frames: &mut dyn Frames,
        root: u64,
        ram: &[(usize, usize)],
        devices: &[Range<u64>],
    ) -> Result<()> {
        for &(start, end, kind) in &self.regions {
            let overlaps = ram.iter().any(|r| (r.0 as u64) < end && start < r.1 as u64)
                || devices
                    .iter()
                    .any(|r| r.start & !(PAGE - 1) < end && start < r.end.next_multiple_of(PAGE));
            if overlaps {
                continue;
            }
            let attributes = match kind {
                MemoryType::MMIO | MemoryType::MMIO_PORT_SPACE => NX | UNCACHED,
                _ if runtime_code(kind) => 0,
                _ => NX,
            };
            for page in (start..end).step_by(PAGE as usize) {
                tables::map_page(frames, root, page, page, PRESENT | WRITABLE | attributes)?;
            }
        }
        Ok(())
    }
}

fn runtime_code(kind: MemoryType) -> bool {
    matches!(
        kind,
        MemoryType::RUNTIME_SERVICES_CODE | MemoryType::PAL_CODE
    )
}

/// Identity-map the storage controllers' registers uncached; none may
/// overlap RAM.
fn map_devices(
    frames: &mut dyn Frames,
    root: u64,
    ram: &[(usize, usize)],
    ranges: &[Range<u64>],
) -> Result<()> {
    for range in ranges {
        let start = range.start & !(PAGE - 1);
        let end = range.end.next_multiple_of(PAGE);
        if ram.iter().any(|r| (r.0 as u64) < end && start < r.1 as u64) {
            return Err(refuse!("storage registers overlap RAM"));
        }
        tables::map_range(
            frames,
            root,
            start,
            start,
            end - start,
            PRESENT | WRITABLE | NX | UNCACHED,
        )?;
    }
    Ok(())
}

/// What the menu says when the machine can only be restored.
fn held(refusal: Refusal) -> &'static str {
    match refusal {
        Refusal::BlankDisk => "The Mac's SSD is not prepared yet: choose Install macOS first.",
        Refusal::NoRecovery { installed: false } => {
            "No macOS is installed and the Mac's SSD holds no System Recovery: choose Install \
             macOS."
        }
        Refusal::NoRecovery { installed: true } => {
            "NVRAM asks for a recoveryOS the SSD lacks. Install macOS erases the SSD."
        }
        _ => "The Mac's last install did not finish: choose Install macOS again.",
    }
}

/// Why the default boot, `options`, can only wait for a restore, checked
/// before the startup menu opens so that the menu starts there.
pub fn held_at_start(options: &[u8]) -> Option<&'static str> {
    if !options.starts_with(b"--system") {
        return None;
    }
    if crate::choice::unchosen(options) {
        return Some("No disk is chosen for macOS yet: choose Install macOS first.");
    }
    crate::status::begin("Checking the internal SSD and NVRAM");
    let (held, found) = match crate::selection::peek(options.to_vec()) {
        Err(refusal @ Refusal::BlankDisk) => (Some(held(refusal)), "no macOS installed yet"),
        Err(refusal @ Refusal::RecoveryMode) => (Some(held(refusal)), "an install did not finish"),
        Err(refusal @ Refusal::NoRecovery { .. }) => (Some(held(refusal)), "no recoveryOS to boot"),
        _ => (None, ""),
    };
    crate::status::done(found);
    held
}

/// Run the engine; never returns.
/// `log` is the partition holding the USB's boot log, when the loader keeps
/// one there (`log_file`).
pub(crate) fn boot(
    mut selected: crate::choice::Selection,
    mut volume: FileSystem,
    log: Option<Guid>,
    host_firmware: crate::host_firmware::Resources,
    snapshot: Option<Snapshot>,
) -> ! {
    // NVRAM chooses macOS or recoveryOS before any input is read. A machine
    // NVRAM holds in recovery mode waits at the startup choices, as LLB
    // waits for a restore host, until one that boots is chosen.
    let (options, northwood_config) = loop {
        if let Some(system) = core::str::from_utf8(&selected.options)
            .ok()
            .and_then(|text| System::parse(text).ok())
        {
            crate::log_file::set_volume_log(!system.no_volume_log);
        }
        crate::status::begin("Reading the startup request in NVRAM");
        let northwood = crate::firmware::selected_northwood_config(&mut volume, &selected);
        let routed = crate::selection::route(selected.options);
        crate::status::done("");
        match routed {
            Ok(routed) => break (routed, northwood.unwrap_or_default()),
            Err(Refusal::Other(error)) => output::fatal(error),
            Err(refusal) => {
                selected = crate::choice::hold(&mut volume, held(refusal)).unwrap_or_else(|| {
                    output::fatal(
                        "the machine can only be restored and this volume has no startup \
                         choices to restore it from",
                    )
                });
                if !selected.options.starts_with(b"--system") {
                    output::fatal("the chosen startup option is not a system-mode boot");
                }
                crate::status::take_screen(Framebuffer::from_firmware());
            }
        }
    };
    // `inputs` closes the volume: protocols must be closed before boot
    // services end.
    let inputs = inputs(&options, volume);
    let result = inputs.and_then(|inputs| {
        output::boot_stage("firmware-inputs-ready");
        load(
            options,
            northwood_config,
            inputs,
            log,
            host_firmware,
            snapshot.ok_or(refuse!("no pristine image copy"))?,
        )
    });
    match result {
        Ok(never) => match never {},
        Err(error) => output::fatal(error),
    }
}

/// Read the translation store, when a volume has one and a file slot is
/// left, within what this PC's free RAM leaves after Stela's parts and the
/// guest's floor (`host_memory::pack_budget`), less `pending` bytes the
/// loader allocates before it reserves the parts: a cache, it never stops
/// the boot. Rewriting its pack needs the storage drivers.
fn translation_store(
    mut files: BundleFiles,
    mut storage: Option<stela_drivers::Handoff>,
    cores: usize,
    pending: u64,
) -> (BundleFiles, Option<stela_drivers::Handoff>) {
    use crate::engine::translations::PackRead;
    crate::status::begin("Opening the translation cache");
    // Without a memory map the pack gets nothing; the parts' sizing
    // reports the map's absence.
    let free = memory_map(None).map_or(0, |map| map.ledger.separate_usable.saturating_sub(pending));
    let budget = stela_boot_core::host_memory::pack_budget(free, cores, cores);
    let Some(store) = crate::engine::translations::read(budget) else {
        crate::status::done("none on this USB");
        return (files, storage);
    };
    let mb = |bytes: u64| bytes / 1_000_000;
    let stays = "new translations stay in memory this boot";
    match store.pack {
        PackRead::Skipped => crate::status::warn(&format!(
            "left on the USB: this PC's memory has no room for it; {stays}"
        )),
        PackRead::Part { read, held } => crate::status::warn(&format!(
            "{} of {} MB, as much as this PC's memory holds",
            mb(read),
            mb(held)
        )),
        PackRead::Whole(bytes) if store.writable => {
            crate::status::done(&format!("{} MB", mb(bytes)));
        }
        PackRead::None if store.writable => crate::status::done("empty"),
        PackRead::Whole(_) | PackRead::None if budget == 0 => {
            crate::status::warn(&format!("this PC's memory has no room for it; {stays}"))
        }
        PackRead::Whole(bytes) => crate::status::warn(&format!(
            "{} MB; {stays}: `stela setup repair` prepares the file",
            mb(bytes)
        )),
        PackRead::None => {
            crate::status::warn(&format!("{stays}: `stela setup repair` prepares the file"))
        }
    }
    if files.len() < MAX_FILES {
        files.push((
            String::from(crate::engine::translations::FILE),
            store.region,
        ));
        if store.writable && storage.is_none() {
            storage = storage::discover().ok();
        }
    }
    (files, storage)
}

fn load(
    options: Vec<u8>,
    northwood_config: Vec<u8>,
    inputs: Inputs,
    log: Option<Guid>,
    host_firmware: crate::host_firmware::Resources,
    snapshot: Snapshot,
) -> Result<core::convert::Infallible> {
    let staged = inputs.report_staged();
    let Inputs {
        source,
        files: bundle,
        bundle_pes,
        staging,
        storage,
        wifi_key,
    } = inputs;
    let text = core::str::from_utf8(&options).map_err(|_| refuse!("stela.cfg is not UTF-8"))?;
    let system = System::parse(text).map_err(stela_boot_core::options::Error::message)?;
    let (native_processors, available_cores) = crate::native_smp::firmware::capture_inventory()
        .map_err(crate::native_smp::firmware::Error::message)?;
    let cores = system.assigned_native_cores(available_cores, bundle_pes)?;
    // The translation cache takes only what the parts and the guest leave,
    // so it is read last before they are reserved: only the engine image's
    // copy comes between.
    let image_bytes = crate::native_image::ResidentImage::allocation_bytes(snapshot.bytes);
    let (bundle, storage) = translation_store(bundle, storage, cores, image_bytes);
    output::boot_stage("translation-store-ready");
    let pack = bundle
        .iter()
        .find(|(name, _)| name == crate::engine::translations::FILE)
        .map_or(0, |(_, region)| region.bytes);
    output::arguments(format_args!("HOST STAGED pack={pack:#x}\n"));
    crate::status::begin("Setting up Stela's memory");
    let image = memory::prepare_image(&snapshot)?;
    let (cores, sipi) = memory::prepare_sipi(cores);
    let before_reclaim = memory_map(None)?.ledger.separate_usable;
    let snapshot_bytes = snapshot.bytes;
    let before_release = crate::ALLOCATOR.usage();
    // The relocated image owns its bytes. Return the pristine copy and the
    // unused heap tail before sizing, so the firmware map is the authority.
    drop(snapshot);
    let released = release_loader_heap();
    output::arguments(format_args!(
        "HOST LOADER heap_total={:#x} heap_live={:#x} heap_released={released:#x} \
         snapshot={snapshot_bytes:#x} resident_image={image_bytes:#x}\n",
        before_release.total, before_release.live,
    ));
    let mut memory = memory::reserve(
        (cores, sipi),
        &image.ram,
        staged + pack,
        released,
        before_reclaim,
    )?;
    let tsc_hz = crate::firmware::tsc_hz();
    if !crate::cpu::INVARIANT_TSC.present() {
        output::arguments(format_args!(
            "UNSUPPORTED host TSC is not invariant: guest time and every timeout follow the cores' clock changes\n"
        ));
    }
    let wall = read_rtc();
    let power = host_power();
    let random_seed = match crate::entropy::retain_seed() {
        Ok((region, source)) => {
            output::arguments(format_args!("HOST entropy seeded source={source}\n"));
            region
        }
        Err(error) => {
            output::arguments(format_args!("UNSUPPORTED host entropy: {error}\n"));
            Region::default()
        }
    };

    let remapping = crate::native_dma::firmware_units()?;
    let amd_tables = crate::native_dma::reserve_amd_tables(&remapping);

    let mut native_mmio = match crate::native_mmio::Prepared::capture() {
        Ok(prepared) => Some(prepared),
        Err(error) => {
            output::arguments(format_args!("NATIVE MMIO unavailable capture={error}\n"));
            None
        }
    };

    let runtime = Runtime::capture()?;
    let storage = memory::map(
        &image,
        &mut memory,
        &runtime,
        storage.as_ref(),
        native_mmio.as_mut(),
        &host_firmware,
    )?;
    let memory::EngineImage {
        image,
        placement,
        screen,
        resident,
        ..
    } = image;
    let memory::EngineMemory {
        profile: memory_profile,
        heap,
        native_resources,
        native_task_allocations,
        stack,
        code,
        pool,
        handoff_page,
        trampoline,
        root,
        loader_tables,
    } = memory;
    let loader_tables = loader_tables
        .retain()
        .map_err(|_| refuse!("loader table ownership validation failed"))?;
    let bounce = native_resources.bounce[0];
    let handoff_phase::RetainedFiles {
        options: options_region,
        source: source_region,
        files,
        file_count,
    } = handoff_phase::retain_files(options, source, bundle)?;
    let northwood_config = handoff_phase::retain_config(northwood_config);
    let handoff = handoff_page.physical as *mut Handoff;
    let engine_entry =
        layout::image() + (crate::engine::entry as *const () as u64 - image.start as u64);

    crate::status::done("");
    let boot_heap = crate::ALLOCATOR.usage();
    output::arguments(format_args!(
        "BOOT HEAP total={}M peak={}M live={}M\n",
        boot_heap.total >> 20,
        boot_heap.peak >> 20,
        boot_heap.live >> 20,
    ));
    // SAFETY: the fresh immutable MP observation is held LOADER_DATA. Identity
    // mappings and the calibrated local APIC remain valid across EBS.
    let capture = unsafe {
        crate::native_smp::firmware::Capture::read(native_processors.physical as *const u8)
    };
    crate::status::begin("Taking control of the other processors");
    let boot_esp = (storage.present
        && memory_profile == stela_boot_core::host_memory::Profile::Paged)
        .then(firmware::esp::boot_esp)
        .flatten();
    let volume_log = crate::output::leave_boot_services().filter(|_| storage.present);
    let mut bundle_layout = staging.commit();
    // SAFETY: no protocol or pool allocation outlives this call; every page
    // Stela keeps is LOADER_DATA or LOADER_CODE.
    let map = unsafe { boot::exit_boot_services(None) };
    output::boot_stage("exit-boot-services");
    let mut exit = SuccessfulExit {
        map: core::mem::ManuallyDrop::new(map),
        ownership: Some(EbsOwnership { _private: () }),
    };
    // SAFETY: the firmware's identity map is still loaded, the APs idle in
    // firmware code, and no Stela driver or memory reuse has started.
    let inactive_firmware_units =
        unsafe { crate::native_dma::release_firmware_units(remapping, tsc_hz) };
    let native_mmio_capture = match native_mmio {
        Some(prepared) => match prepared.finish(&mut exit) {
            Ok(sealed) => crate::native_mmio::CaptureHandoff::new(Some(sealed)),
            Err(error) => {
                output::arguments(format_args!("NATIVE MMIO unavailable final-map={error}\n"));
                crate::native_mmio::CaptureHandoff::new(None)
            }
        },
        None => crate::native_mmio::CaptureHandoff::new(None),
    };
    // SAFETY: Stop every enabled firmware AP before offering its former code/data to
    // guest RAM, including workers that will later receive their owned SIPI.
    // The BSP alone sends ICR commands, with firmware IRQ handlers excluded.
    unsafe { asm!("cli", options(nomem, nostack)) };
    // x2APIC whenever CPUID offers it, from here on: the firmware no longer
    // runs on this processor, and every later APIC handle is made from the
    // base returned here (the handoff's) or read afresh. The APs take the
    // same mode as they enter Stela (`native_smp::executor`).
    // SAFETY: CPL0, interrupts masked, no APIC handle of this processor made.
    let apic_base = unsafe { crate::apic::prefer_x2apic() };
    let sender = crate::apic::Apic::new(apic_base, apic_base & crate::apic::BASE_ADDRESS);
    output::arguments(format_args!(
        "HOST APIC {}\n",
        if sender.is_x2apic() {
            "x2APIC"
        } else {
            "xAPIC (no x2APIC offered)"
        }
    ));
    // SAFETY: interrupts are masked, the BSP alone sends ICR commands, and `capture` lists the
    // enabled firmware APs.
    let deliveries = match unsafe {
        crate::native_smp::executor::quiesce_firmware_aps(capture, &sender, tsc_hz)
    } {
        Ok(deliveries) => deliveries,
        Err(error) => {
            output::arguments(format_args!("NATIVE FIRMWARE INIT refused: {error:?}\n"));
            output::fatal("firmware processor handoff refused before memory reuse");
        }
    };
    output::arguments(format_args!(
        "NATIVE FIRMWARE INIT deliveries={deliveries}; delivery only, no execution acknowledgement\n"
    ));
    crate::status::done("");
    // Keep this post-EBS boundary when the relocated engine replays status.
    let mut history = [0; crate::status::HISTORY_BYTES];
    let history_bytes = crate::status::history(&mut history);
    let mut host = [0; crate::host_inventory::HANDOFF_BYTES];
    let host_bytes = crate::host_inventory::save(&mut host);
    let handoff_phase::GuestMemory {
        ranges: guest,
        count,
        ledger: final_memory_map,
    } = handoff_phase::guest_memory(exit.map(), &mut bundle_layout);
    let adopted = bundle_layout
        .backing()
        .iter()
        .map(|part| part.region.bytes)
        .sum::<u64>();
    if adopted != 0 {
        output::arguments(format_args!(
            "HOST BUNDLE direct={adopted:#x} remaining_staged={:#x}\n",
            staged.saturating_sub(adopted),
        ));
    }
    // SAFETY: fresh LOADER_DATA pages sized for the handoff.
    unsafe {
        handoff.write(Handoff {
            magic: MAGIC,
            options: options_region,
            random_seed,
            source: source_region,
            files,
            file_count,
            heap,
            stack,
            code,
            pool,
            placement,
            root,
            bounce,
            screen,
            history,
            history_bytes,
            host,
            host_bytes,
            log_partition: log.map(|guid| guid.0),
            volume_log,
            boot_esp,
            tsc_hz,
            tsc_source: crate::firmware::tsc_source(),
            tsc_tolerance_ppm: crate::firmware::tsc_tolerance_ppm(),
            apic_base,
            native_processors,
            native_resources,
            native_mmio: native_mmio_capture.region().unwrap_or_default(),
            wall,
            guest,
            guest_count: count,
            installed: final_memory_map.installed,
            final_memory_map,
            memory_profile,
            storage,
            host_firmware,
            power,
            efi_reset: runtime.reset,
            rsdp: storage::rsdp().unwrap_or(0),
            wifi_key: Region {
                physical: wifi_key.as_ptr() as u64,
                bytes: wifi_key.len() as u64,
            },
            native_task_allocations,
            native_loader_tables: loader_tables,
            native_image: resident,
            native_mmio_capture,
            inactive_firmware_units,
            amd_tables,
            northwood_config,
            bundle_layout,
        });
    }
    // SAFETY: boot services ended; the completed root maps this code and
    // trampoline, and all engine-window reservations remain allocated.
    unsafe {
        handoff_phase::Transfer {
            placement,
            root,
            trampoline,
            engine_entry,
            handoff,
        }
        .enter()
    }
}

/// The host's ACPI power hardware, reported while the firmware's tables
/// are still mapped; the engine gets it without the report.
pub(crate) fn host_power() -> Option<pc::power::HostPower> {
    let rsdp = storage::rsdp()?;
    // SAFETY: the firmware's RSDP, its tables mapped while boot services last.
    match unsafe { pc::power::discover(rsdp) } {
        Ok(power) => {
            if let Some(reason) = power.refused {
                output::arguments(format_args!("POWER host refusal: {reason}\n"));
            }
            Some(pc::power::HostPower {
                refused: None,
                ..power
            })
        }
        Err(error) => {
            output::arguments(format_args!("POWER host has no usable FADT: {error:?}\n"));
            None
        }
    }
}

/// The firmware's real-time clock, read between two TSC values. An
/// unspecified time zone is read as UTC, as Linux reads the RTC and as QEMU
/// and Macs keep it; a time the firmware cannot give is reported.
fn read_rtc() -> Option<crate::handoff::RtcReading> {
    use stela_boot_core::firmware_time::{self, UnspecifiedZone};
    let tsc_before = rdtsc();
    let read = uefi::runtime::get_time_and_caps();
    let tsc_after = rdtsc();
    let (time, capabilities) = read
        .inspect_err(|error| {
            output::arguments(format_args!(
                "HOST RTC unavailable: GetTime failed: {:?}\n",
                error.status()
            ))
        })
        .ok()?;
    let fields = firmware_time::Time {
        year: time.year(),
        month: time.month(),
        day: time.day(),
        hour: time.hour(),
        minute: time.minute(),
        second: time.second(),
        nanosecond: time.nanosecond(),
        timezone_minutes: time.time_zone().unwrap_or(firmware_time::UNSPECIFIED_ZONE),
        daylight: time.daylight().bits(),
    };
    let utc = firmware_time::utc(fields, UnspecifiedZone::RtcIsUtc)
        .inspect_err(|error| output::arguments(format_args!("HOST RTC unusable: {error}\n")))
        .ok()?;
    Some(crate::handoff::RtcReading {
        utc,
        resolution: capabilities.resolution,
        tsc_before,
        tsc_after,
    })
}
