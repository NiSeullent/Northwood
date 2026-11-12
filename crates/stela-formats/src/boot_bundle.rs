//! The kernel boot bundle as the UEFI mount consumes it.
//!
//! `stela-dev boot prepare` writes `bundle.json` and the files it lists
//! ([`schema`] is the format). Before boot
//! services end the mount reads every file and [`Manifest::verify`]s each
//! file and region hash. [`Verified::load`] then writes the regions into
//! guest DRAM through the caller's writer, fills the patch slots only the
//! running machine knows, and returns a plain [`GuestImage`] for the engine.
//! Nothing here parses Mach-O, IM4P or the DeviceTree: every value the host
//! could compute is already in the files.
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::fmt;
use stela_contracts::Refusal;

pub mod schema;
pub use schema::{
    BootCpu, BootCpuSelection, CoreLocks, Encoding, Manifest, Options, Patch, PatchSource,
    SCHEMA_VERSION, Source, Target,
};
use schema::{File, Region};
pub const MANIFEST_FILE: &str = "bundle.json";
/// Maximum encoded size of a boot bundle manifest.
pub const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
/// Guest DRAM is backed by 2 MiB host frames.
pub const FRAME_BYTES: u64 = 2 * 1024 * 1024;
const U64_BYTES: usize = 8;
/// The verbose-boot flag, under which `Boot_Video.v_display` is 0 (m1n1
/// `set_bootargs`).
const VERBOSE_FLAG: &str = "-v";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BundleError {
    Manifest(String),
    File {
        name: String,
        reason: String,
    },
    Region {
        name: String,
        reason: String,
    },
    Patch {
        field: String,
        reason: String,
    },
    /// The running machine cannot host the bundle.
    Machine(String),
    /// No entropy source; the bundle requires fresh random bytes.
    Entropy(Refusal),
}

impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(what) => write!(f, "bundle.json: {what}"),
            Self::File { name, reason } => write!(f, "bundle file {name}: {reason}"),
            Self::Region { name, reason } => write!(f, "region {name}: {reason}"),
            Self::Patch { field, reason } => write!(f, "patch {field}: {reason}"),
            Self::Machine(what) => write!(f, "machine: {what}"),
            Self::Entropy(what) => write!(f, "entropy: {what}"),
        }
    }
}

impl core::error::Error for BundleError {}

/// The firmware framebuffer as the guest sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Framebuffer {
    /// Guest physical address the machine maps the framebuffer at.
    pub guest_pa: u64,
    pub bytes: u64,
    pub row_bytes: u64,
    pub width: u64,
    pub height: u64,
    pub bits_per_pixel: u64,
}

/// What the running machine provides for the patch slots.
#[derive(Debug, Clone)]
pub struct Machine {
    pub dram_base: u64,
    /// Guest DRAM bytes backed from `dram_base`.
    pub ram_bytes: u64,
    /// The DRAM the Mac reports (`/chosen dram-size`, `memSizeActual`):
    /// the backed bytes and, above them, the host memory Stela and the
    /// firmware keep, which the kernel never manages, as with iBoot's
    /// carveouts.
    pub dram_bytes: u64,
    /// The counter rate `CNTFRQ_EL0` reports.
    pub timebase_hz: u64,
    pub framebuffer: Option<Framebuffer>,
    /// Replaces the prepared kernel command line.
    pub boot_args: Option<String>,
    /// The persistent NOR's current NVRAM bank, replacing the prepared
    /// empty one.
    pub nvram: Option<Nvram>,
}

/// The NVRAM bank iBoot hands the kernel in `/chosen`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nvram {
    /// `nvram-current-bank`.
    pub bank: u32,
    /// `nvram-proxy-data`: the bank's bytes.
    pub proxy: Vec<u8>,
}

impl Machine {
    /// The end of the range the kernel manages: the DRAM end, or the
    /// framebuffer's base, which must then end the backed RAM. iBoot keeps
    /// `/vram` out of the kernel's memory; where it puts it is not
    /// established, so the machine places it at the top (inference).
    fn managed_end(&self) -> Result<u64, &'static str> {
        let end = self.dram_base + self.ram_bytes;
        match self.framebuffer {
            None => Ok(end),
            Some(screen) if screen.guest_pa.checked_add(screen.bytes) == Some(end) => {
                Ok(screen.guest_pa)
            }
            Some(_) => Err("the framebuffer does not end the backed RAM"),
        }
    }
}

impl Manifest {
    /// The cores' lock registers iBoot leaves on `machine`. Where the chip
    /// keeps memory tags, SPTM requires TAG_OFFSET_EL2 at the top of the
    /// managed range, sized for `/chosen dram-size`
    /// ([`stela_contracts::tag_storage`]), so its value follows the machine's
    /// DRAM rather than the prepared one.
    pub fn core_locks(&self, machine: &Machine) -> Result<Option<CoreLocks>, BundleError> {
        let Some(mut locks) = self.core_locks.clone() else {
            return Ok(None);
        };
        if locks.tag_offset != 0 {
            let fail = |reason: &str| BundleError::Machine(format!("tag storage: {reason}"));
            let phys_base = self.options.phys_base;
            let managed = machine.managed_end().map_err(fail)? - phys_base;
            locks.tag_offset =
                stela_contracts::tag_storage::hand_off(phys_base, managed, machine.dram_bytes)
                    .ok_or_else(|| fail("its base does not fit TAG_OFFSET_EL2"))?;
            if stela_contracts::tag_storage::base(locks.tag_offset) < self.top_of_kernel_data {
                return Err(fail("it overlaps the loaded regions"));
            }
        }
        Ok(Some(locks))
    }
}

/// A contiguous guest RAM range the engine must back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RamExtent {
    pub guest_pa: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedRegion {
    pub name: String,
    pub guest_pa: u64,
    pub bytes: u64,
    pub memory_bytes: u64,
}

/// A patch slot and the bytes now in guest RAM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedPatch {
    pub field: String,
    pub source: PatchSource,
    pub guest_pa: u64,
    pub value: Vec<u8>,
}

/// Plain data the engine starts from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestImage {
    /// The image entered first: SPTM.
    pub entry_image: String,
    pub entry_pa: u64,
    pub entry_va: u64,
    /// x0 on entry: the `boot_args` physical address.
    pub x0: u64,
    pub ram: Vec<RamExtent>,
    pub regions: Vec<LoadedRegion>,
    pub patches: Vec<AppliedPatch>,
    /// Slots left with their prepared bytes, such as the framebuffer when
    /// the machine has none.
    pub kept: Vec<String>,
}

fn manifest_error(what: impl ToString) -> BundleError {
    BundleError::Manifest(what.to_string())
}

/// One complete manifest region's bytes at their offset in the source file.
#[derive(Debug, Clone, Copy)]
pub struct FilePart<'a> {
    pub offset: u64,
    pub bytes: &'a [u8],
}

/// Borrowed file contents as one slice or as nonempty regions in file order.
#[derive(Debug, Clone, Copy)]
pub enum FileData<'a> {
    Whole(&'a [u8]),
    Parts(&'a [FilePart<'a>]),
}

impl<'a> FileData<'a> {
    fn region_bytes(self, region: &Region) -> Option<&'a [u8]> {
        if region.bytes > region.memory_bytes {
            return None;
        }
        let length = usize::try_from(region.bytes).ok()?;
        match self {
            Self::Whole(file) => {
                let start = usize::try_from(region.file_offset).ok()?;
                file.get(start..start.checked_add(length)?)
            }
            Self::Parts(parts) => {
                let index = parts
                    .binary_search_by_key(&region.file_offset, |part| part.offset)
                    .ok()?;
                let bytes = parts[index].bytes;
                (bytes.len() == length).then_some(bytes)
            }
        }
    }
}

impl Manifest {
    pub fn parse(json: &[u8]) -> Result<Self, BundleError> {
        if json.len() > MAX_MANIFEST_BYTES {
            return Err(manifest_error("exceeds 16 MiB"));
        }
        let manifest: Self = serde_json::from_slice(json).map_err(manifest_error)?;
        if manifest.schema_version != SCHEMA_VERSION {
            return Err(manifest_error(format!(
                "schema {} is not {SCHEMA_VERSION}",
                manifest.schema_version
            )));
        }
        for file in &manifest.files {
            let simple = !file.name.is_empty()
                && file
                    .name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b".-_,".contains(&b));
            if !simple {
                return Err(manifest_error(format!("unsafe file name {:?}", file.name)));
            }
        }
        Ok(manifest)
    }

    /// Whether the regions cover `file` exactly once, end to end.
    fn tiled(&self, file: &File) -> bool {
        let mut ranges: Vec<(u64, u64)> = self
            .regions
            .iter()
            .filter(|r| r.file == file.name)
            .map(|r| (r.file_offset, r.file_offset.saturating_add(r.bytes)))
            .collect();
        ranges.sort_unstable();
        let mut cursor = 0;
        for (start, end) in ranges {
            if start != cursor {
                return false;
            }
            cursor = end;
        }
        cursor == file.bytes
    }

    /// A part-backed file must have a nonempty, exact tile with one part per
    /// region. Unlike whole files, there is no contiguous buffer to hash for
    /// bytes outside the tile.
    fn parts_match(&self, file: &File, parts: &[FilePart<'_>]) -> bool {
        let mut regions: Vec<_> = self
            .regions
            .iter()
            .filter(|region| region.file == file.name)
            .collect();
        if parts.is_empty() || parts.len() != regions.len() {
            return false;
        }
        regions.sort_unstable_by_key(|region| region.file_offset);
        let mut offset = 0;
        for (region, part) in regions.into_iter().zip(parts) {
            if region.file_offset != offset
                || part.offset != offset
                || part.bytes.is_empty()
                || u64::try_from(part.bytes.len()).ok() != Some(region.bytes)
            {
                return false;
            }
            let Some(end) = offset.checked_add(region.bytes) else {
                return false;
            };
            offset = end;
        }
        offset == file.bytes
    }

    /// Check every file's size and every region's extent and SHA-256. A
    /// file the regions do not tile exactly is also checked against its own
    /// SHA-256; a tiled file's region hashes already cover every byte.
    /// `contents` returns a file's bytes by name.
    pub fn verify<'a>(
        &'a self,
        contents: &dyn Fn(&str) -> Option<&'a [u8]>,
    ) -> Result<Verified<'a>, BundleError> {
        self.verify_files(&|name| contents(name).map(FileData::Whole))
    }

    /// Verify the same file and region identities as [`Self::verify`]. A
    /// file supplied in parts must match its complete manifest region tile.
    /// The result borrows each region's original bytes for loading.
    pub fn verify_files<'a>(
        &'a self,
        contents: &dyn Fn(&str) -> Option<FileData<'a>>,
    ) -> Result<Verified<'a>, BundleError> {
        let mut files = Vec::with_capacity(self.files.len());
        for file in &self.files {
            let fail = |reason| BundleError::File {
                name: file.name.clone(),
                reason,
            };
            let data = contents(&file.name).ok_or_else(|| fail("was not read".into()))?;
            match data {
                FileData::Whole(bytes) => {
                    if bytes.len() as u64 != file.bytes {
                        return Err(fail(format!(
                            "is {} bytes, not {}",
                            bytes.len(),
                            file.bytes
                        )));
                    }
                    if !self.tiled(file) && crate::sha256(bytes) != file.sha256 {
                        return Err(fail("SHA-256 differs from bundle.json".into()));
                    }
                }
                FileData::Parts(parts) => {
                    if self
                        .files
                        .iter()
                        .filter(|other| other.name == file.name)
                        .take(2)
                        .count()
                        != 1
                        || !self.parts_match(file, parts)
                    {
                        return Err(fail(
                            "parts do not match an exact manifest region tile".into(),
                        ));
                    }
                }
            }
            files.push((file.name.as_str(), data));
        }
        let mut regions = Vec::with_capacity(self.regions.len());
        for region in &self.regions {
            let (_, file) = files
                .iter()
                .find(|(name, _)| *name == region.file)
                .ok_or_else(|| manifest_error(format!("no file {}", region.file)))?;
            let bytes = file
                .region_bytes(region)
                .ok_or_else(|| BundleError::Region {
                    name: region.name.clone(),
                    reason: format!("lies outside {}", region.file),
                })?;
            if crate::sha256(bytes) != region.sha256 {
                return Err(BundleError::Region {
                    name: region.name.clone(),
                    reason: "SHA-256 differs from bundle.json".into(),
                });
            }
            regions.push(bytes);
        }
        Ok(Verified {
            manifest: self,
            regions,
        })
    }
}

/// A manifest whose files and regions matched their hashes.
pub struct Verified<'a> {
    pub manifest: &'a Manifest,
    regions: Vec<&'a [u8]>,
}

fn u64_value(value: u64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

/// The low 32 bits of `value`, for a u32 DeviceTree property.
fn u32_value(value: u64) -> Vec<u8> {
    (value as u32).to_le_bytes().to_vec()
}

impl<'a> Verified<'a> {
    /// The bytes a patch slot takes on `machine`; `None` keeps the prepared
    /// bytes.
    fn patch_value(
        &self,
        patch: &Patch,
        machine: &Machine,
        entropy: &mut dyn FnMut(&mut [u8]) -> Result<(), Refusal>,
    ) -> Result<Option<Vec<u8>>, BundleError> {
        let error = |reason: &str| BundleError::Patch {
            field: patch.field.clone(),
            reason: reason.into(),
        };
        let bytes = usize::try_from(patch.bytes).map_err(|_| error("size overflows"))?;
        let options = &self.manifest.options;
        let value = match (patch.source, patch.field.as_str()) {
            (PatchSource::DramSize, "/chosen dram-size" | "boot_args.memSizeActual") => {
                Some(u64_value(machine.dram_bytes))
            }
            (PatchSource::DramSize, "boot_args.memSize") => Some(u64_value(
                machine.managed_end().map_err(error)? - options.phys_base,
            )),
            (PatchSource::Entropy, _) => {
                let mut random = vec![0; bytes];
                entropy(&mut random).map_err(BundleError::Entropy)?;
                Some(random)
            }
            (PatchSource::Timebase, _) => Some(u64_value(machine.timebase_hz)),
            (PatchSource::Framebuffer, field) => match machine.framebuffer {
                None => None,
                Some(screen) => {
                    let command_line = machine
                        .boot_args
                        .as_deref()
                        .unwrap_or(&self.manifest.boot_args.command_line);
                    let verbose = command_line
                        .split_ascii_whitespace()
                        .any(|w| w == VERBOSE_FLAG);
                    Some(match field {
                        "/vram reg" => {
                            [screen.guest_pa.to_le_bytes(), screen.bytes.to_le_bytes()].concat()
                        }
                        "boot_args.Video.v_baseAddr" => u64_value(screen.guest_pa),
                        "boot_args.Video.v_display" => u64_value(u64::from(!verbose)),
                        "boot_args.Video.v_rowBytes" => u64_value(screen.row_bytes),
                        "boot_args.Video.v_width" => u64_value(screen.width),
                        "boot_args.Video.v_height" => u64_value(screen.height),
                        "boot_args.Video.v_depth" => u64_value(screen.bits_per_pixel),
                        // iBoot derives these from v_depth (file 0x24148..
                        // 0x2425c): bits 8..15 rotation and 24..31 boot
                        // rotation in quarter turns, bits 16..23 the scale
                        // less one. Stela's depth has neither, so the scale
                        // is 1, never the template's 0.
                        "/chosen display-rotation" => {
                            u32_value((screen.bits_per_pixel >> 8 & 0xff) * 90)
                        }
                        "/chosen display-scale" => {
                            u32_value((screen.bits_per_pixel >> 16 & 0xff) + 1)
                        }
                        "/chosen display-boot-rotation" => {
                            u32_value((screen.bits_per_pixel >> 24 & 0xff) * 90)
                        }
                        "/arm-io/dispext0 framebuffer-width" => u32::try_from(screen.width)
                            .map_err(|_| error("external framebuffer width exceeds u32"))?
                            .to_le_bytes()
                            .to_vec(),
                        "/arm-io/dispext0 framebuffer-height" => u32::try_from(screen.height)
                            .map_err(|_| error("external framebuffer height exceeds u32"))?
                            .to_le_bytes()
                            .to_vec(),
                        _ => return Err(error("unknown framebuffer field")),
                    })
                }
            },
            (PatchSource::Nvram, "/chosen nvram-proxy-data") => {
                machine.nvram.as_ref().map(|nvram| nvram.proxy.clone())
            }
            (PatchSource::Nvram, "/chosen nvram-current-bank") => machine
                .nvram
                .as_ref()
                .map(|nvram| nvram.bank.to_le_bytes().to_vec()),
            (PatchSource::BootArgs, _) => match &machine.boot_args {
                None => None,
                Some(line) => {
                    if line.len() >= bytes || line.contains('\0') {
                        return Err(error("boot-args do not fit the command line"));
                    }
                    let mut value = line.as_bytes().to_vec();
                    value.resize(bytes, 0);
                    Some(value)
                }
            },
            _ => return Err(error("unknown field for its source")),
        };
        if let Some(value) = &value {
            let width = match patch.encoding {
                Encoding::U64Le => U64_BYTES,
                Encoding::Bytes | Encoding::CString => value.len(),
            };
            if value.len() != bytes || value.len() != width {
                return Err(error("value does not fill the slot"));
            }
        }
        Ok(value)
    }

    /// Write every region into guest DRAM and fill the patch slots.
    ///
    /// `write(guest_pa, bytes)` stores into guest RAM and returns false when
    /// any byte is not backed; regions are written in manifest order, then
    /// their zero tails, then the patches. `entropy` fills a buffer with
    /// fresh random bytes or refuses.
    pub fn load(
        &self,
        machine: &Machine,
        entropy: &mut dyn FnMut(&mut [u8]) -> Result<(), Refusal>,
        write: &mut dyn FnMut(u64, &[u8]) -> bool,
    ) -> Result<GuestImage, BundleError> {
        let options = &self.manifest.options;
        if options.dram_base != machine.dram_base {
            return Err(BundleError::Machine(format!(
                "bundle DRAM base {:#x} differs from the machine's {:#x}",
                options.dram_base, machine.dram_base
            )));
        }
        let ram_end = machine
            .dram_base
            .checked_add(machine.ram_bytes)
            .ok_or_else(|| BundleError::Machine("DRAM range overflows".into()))?;
        if machine.dram_bytes < machine.ram_bytes {
            return Err(BundleError::Machine(format!(
                "the Mac's {:#x} bytes of DRAM hold fewer than the {:#x} backed",
                machine.dram_bytes, machine.ram_bytes
            )));
        }
        if self.manifest.top_of_kernel_data > ram_end || options.phys_base < machine.dram_base {
            return Err(BundleError::Machine(format!(
                "the bundle needs DRAM up to {:#x}; {:#x} bytes are backed",
                self.manifest.top_of_kernel_data, machine.ram_bytes
            )));
        }
        static ZEROES: [u8; 16 * 1024] = [0; 16 * 1024];
        // iBoot hands its carveouts over blank, and their readers take
        // fresh state from them: SPTM's `sptm_uat_set_ctx_id` panics
        // (`VIOLATION_UAT_ILLEGAL_SET_CTX_ID`) when the GPU's TTBAT
        // (`gpu-region`) shows a free slot's valid bit set. Guest DRAM
        // holds what its backing held before, the last system's data after
        // a warm restart or memory training's patterns, so blank each
        // carveout before the regions load: the boot manifest's lies in
        // the loaded-object carveout.
        for carveout in &self.manifest.carveouts {
            let fail = |reason: &str| {
                BundleError::Machine(format!(
                    "carveout {:#x} at {:#x} {reason}",
                    carveout.id, carveout.base
                ))
            };
            let end = carveout
                .base
                .checked_add(carveout.bytes)
                .ok_or_else(|| fail("overflows"))?;
            if carveout.base < machine.dram_base || end > ram_end {
                return Err(fail("lies outside backed guest DRAM"));
            }
            let mut at = carveout.base;
            while at < end {
                let chunk = (end - at).min(ZEROES.len() as u64) as usize;
                if !write(at, &ZEROES[..chunk]) {
                    return Err(fail("could not be blanked: guest RAM write failed"));
                }
                at += chunk as u64;
            }
        }
        let mut regions = Vec::with_capacity(self.manifest.regions.len());
        for (region, &bytes) in self.manifest.regions.iter().zip(&self.regions) {
            let fail = |reason: &str| BundleError::Region {
                name: region.name.clone(),
                reason: reason.into(),
            };
            let end = region
                .guest_pa
                .checked_add(region.memory_bytes)
                .ok_or_else(|| fail("guest range overflows"))?;
            if region.guest_pa < machine.dram_base || end > ram_end {
                return Err(fail("lies outside backed guest DRAM"));
            }
            if !write(region.guest_pa, bytes) {
                return Err(fail("guest RAM write failed"));
            }
            // Reserved and zero-fill regions can be much larger than
            // their file bytes. Stream the zeros without allocating a tail.
            let mut tail = region.guest_pa + region.bytes;
            while tail < end {
                let chunk = (end - tail).min(ZEROES.len() as u64) as usize;
                if !write(tail, &ZEROES[..chunk]) {
                    return Err(fail("guest RAM write failed"));
                }
                tail += chunk as u64;
            }
            regions.push(LoadedRegion {
                name: region.name.clone(),
                guest_pa: region.guest_pa,
                bytes: region.bytes,
                memory_bytes: region.memory_bytes,
            });
        }
        let mut patches = Vec::new();
        let mut kept = Vec::new();
        for patch in &self.manifest.patches {
            let Some(value) = self.patch_value(patch, machine, entropy)? else {
                if patch.required {
                    return Err(BundleError::Patch {
                        field: patch.field.clone(),
                        reason: "is required and has no value".into(),
                    });
                }
                kept.push(patch.field.clone());
                continue;
            };
            let region = self
                .manifest
                .regions
                .iter()
                .find(|r| r.name == patch.region)
                .ok_or_else(|| BundleError::Patch {
                    field: patch.field.clone(),
                    reason: format!("names missing region {}", patch.region),
                })?;
            let inside = region.file == patch.file
                && patch.file_offset >= region.file_offset
                && patch.file_offset + patch.bytes <= region.file_offset + region.bytes;
            if !inside {
                return Err(BundleError::Patch {
                    field: patch.field.clone(),
                    reason: format!("lies outside region {}", region.name),
                });
            }
            let guest_pa = region.guest_pa + (patch.file_offset - region.file_offset);
            if !write(guest_pa, &value) {
                return Err(BundleError::Patch {
                    field: patch.field.clone(),
                    reason: "guest RAM write failed".into(),
                });
            }
            patches.push(AppliedPatch {
                field: patch.field.clone(),
                source: patch.source,
                guest_pa,
                value,
            });
        }
        Ok(GuestImage {
            entry_image: self.manifest.entry.image.clone(),
            entry_pa: self.manifest.entry.physical_address,
            entry_va: self.manifest.entry.virtual_address,
            x0: self.manifest.entry_x0,
            ram: vec![RamExtent {
                guest_pa: machine.dram_base,
                bytes: machine.ram_bytes,
            }],
            regions,
            patches,
            kept,
        })
    }
}

#[cfg(test)]
mod tests;
