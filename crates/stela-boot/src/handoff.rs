//! Plain data the loader phase passes to the engine phase. Addresses are host
//! physical; the engine reaches them through its physmap.
use crate::console::Framebuffer;
use stela_engine::machine::mmu::placement::Placement;
use stela_runtime::pc;

pub const MAGIC: u64 = 0x5354_454c_4153_5953; // "SYSALETS"
pub const MAX_GUEST_RANGES: usize = 128;
/// A restore bundle's files (13 for a MacBookPro18,3 with its USB-C port
/// controllers' firmware) and the install inventory's four, with room.
pub const MAX_FILES: usize = 32;
pub use stela_boot_core::handoff_records::{File, MAX_NAME, Region, Storage};

#[repr(C)]
pub struct Handoff {
    pub magic: u64,
    /// `stela.cfg`.
    pub options: Region,
    /// Exclusive 256-bit pre-EBS seed, erased by the engine on consumption.
    /// An empty region means no checked entropy source was available.
    pub random_seed: Region,
    /// `source.bin`: the guest image, when no boot bundle is configured.
    pub source: Region,
    /// The boot bundle's `bundle.json` and every file it lists.
    pub files: [File; MAX_FILES],
    pub file_count: usize,
    pub heap: Region,
    pub stack: Region,
    pub code: Region,
    /// Page-table frames: engine roots and shadow tables.
    pub pool: Region,
    /// Where the guest's address space lies beside the window.
    pub placement: Placement,
    /// The loader's root, whose window entries every engine root shares.
    pub root: u64,
    /// The physical addresses of the bounce slot's two leaf entries.
    pub bounce: [u64; 2],
    pub screen: Option<Framebuffer>,
    /// The loader's finished status lines, which the engine's screen starts
    /// with (`status::replay`).
    pub history: [u8; crate::status::HISTORY_BYTES],
    pub history_bytes: usize,
    /// The loader's facts about the PC, as a host record, which the engine
    /// takes up (`host_inventory::restore`).
    pub host: [u8; crate::host_inventory::HANDOFF_BYTES],
    pub host_bytes: usize,
    /// The partition holding the USB's boot log the loader keeps
    /// (`log_file`), which the engine writes on to.
    pub log_partition: Option<[u8; 16]>,
    /// TSC ticks per second (`firmware::tsc_hz`).
    pub tsc_hz: u64,
    /// How `tsc_hz` was found.
    pub tsc_source: crate::firmware::TscSource,
    /// The most `tsc_hz` may be off by, in parts per million.
    pub tsc_tolerance_ppm: u32,
    /// IA32_APIC_BASE as the firmware left it.
    pub apic_base: u64,
    /// Immutable copied UEFI processor identities and per-physical-core raw
    /// CPUID/control observations; absent unless native cores were requested.
    pub native_processors: Region,
    pub native_resources: crate::native_smp::resources::Resources,
    /// Sealed owned UEFI/ACPI resource inventory and empty MMIO aperture.
    /// Zero when capture is unavailable; this alone grants no device owner.
    pub native_mmio: Region,
    /// The firmware's real-time clock at boot; `None` when the firmware has
    /// no valid time.
    pub wall: Option<RtcReading>,
    /// Guest backing in placement order, in 2 MiB-aligned ranges.
    pub guest: [Region; MAX_GUEST_RANGES],
    pub guest_count: usize,
    /// The host's RAM in the firmware's memory map, every RAM type
    /// included: the DRAM the guest Mac reports.
    pub installed: u64,
    /// The host storage drivers' ranges, for a bundle built from a disk.
    pub storage: Storage,
    /// Host-device firmware, separate from guest boot-bundle files.
    pub host_firmware: crate::host_firmware::Resources,
    /// The host's ACPI power hardware, when its FADT is usable.
    pub power: Option<pc::power::HostPower>,
    /// The firmware's `ResetSystem` runtime service, whose regions the
    /// loader keeps mapped by identity; zero without one.
    pub efi_reset: u64,
    /// The firmware's ACPI RSDP; zero without one.
    pub rsdp: u64,
    /// The `--wifi-key` file's bytes, read from the ESP; empty when
    /// `--wifi` names no network or the file could not be read. The engine
    /// parses and wipes them.
    pub wifi_key: Region,
    /// Firmware-created task allocation owners; the preceding wire fields keep
    /// their offsets. The engine can claim each owner once, after mapping checks.
    pub(crate) native_task_allocations: crate::native_smp::resources::TaskAllocations,
    /// Original loader table allocations; root is only their copied descriptor.
    pub(crate) native_loader_tables: crate::paging::loader::LoaderTables,
    /// Moved original allocation of the relocated static Rust/C engine image.
    /// Observed PE/header ranges cannot recreate this non-Clone owner.
    pub(crate) native_image: crate::native_image::ResidentImage,
    /// Original aperture capture sealed with this boot's successful exit.
    pub(crate) native_mmio_capture: crate::native_mmio::CaptureHandoff,
    /// Original remapping inventory qualified at firmware exit.
    pub(crate) inactive_firmware_units: crate::native_dma::InactiveFirmwareUnits,
    /// Exclusive firmware RAM below 4 GiB for AMD-Vi device tables.
    pub(crate) amd_tables: Region,
    /// The prepared data-volume slot the engine can extend through native I/O.
    pub(crate) volume_log: Option<crate::log_file::VolumeLog>,
    pub(crate) final_memory_map: crate::system::memory_map::Ledger,
    pub(crate) memory_profile: stela_boot_core::host_memory::Profile,
    /// Validated ESP configuration and boot entry. Kept outside the guest bundle.
    pub(crate) northwood_config: Region,
    /// The split BootKC source and aligned bundle extents included in `guest`.
    pub(crate) bundle_layout: BundleLayout,
    /// GPT identity of the image's boot ESP, independent of other DATA users.
    pub(crate) boot_esp: Option<pc::swap::volume::BootEsp>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BundlePart {
    pub(crate) offset: u64,
    pub(crate) contents: Region,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BundleBacking {
    pub(crate) guest_offset: u64,
    pub(crate) region: Region,
    pub(crate) kernel_collection: bool,
}

#[derive(Default)]
pub(crate) struct BundleLayout {
    pub(crate) kernel_parts: [BundlePart; MAX_FILES],
    pub(crate) kernel_count: usize,
    pub(crate) backing: [BundleBacking; MAX_FILES],
    pub(crate) backing_count: usize,
}

impl BundleLayout {
    pub(crate) fn kernel_parts(&self) -> &[BundlePart] {
        &self.kernel_parts[..self.kernel_count]
    }

    pub(crate) fn backing(&self) -> &[BundleBacking] {
        &self.backing[..self.backing_count]
    }
}

/// A read of the firmware's real-time clock: its time as UTC and its
/// resolution in counts per second, read between TSC values `tsc_before`
/// and `tsc_after`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RtcReading {
    pub utc: stela_boot_core::firmware_time::Timestamp,
    pub resolution: u32,
    pub tsc_before: u64,
    pub tsc_after: u64,
}

impl RtcReading {
    /// The Unix seconds at TSC `tsc`, carried forward at `tsc_hz`.
    pub fn unix_seconds_at(&self, tsc: u64, tsc_hz: u64) -> u64 {
        self.utc.seconds + tsc.saturating_sub(self.tsc_after) / tsc_hz
    }
}
