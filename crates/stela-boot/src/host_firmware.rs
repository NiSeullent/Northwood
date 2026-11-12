//! Host-device resources from the selected boot volume and native UEFI before EBS. Unlike
//! a Vec/Arc in the loader heap, this fixed wire representation survives engine
//! relocation. Only immutable bytes in reserved LOADER_DATA pages cross it.
use crate::Result;
use crate::{firmware, handoff::Region, layout, output, paging::PAGE};
use alloc::{format, string::ToString, sync::Arc};
pub use stela_boot_core::host_resources::Resources;
use stela_boot_core::host_resources::report;
use stela_contracts::refuse;
use stela_drivers::firmware::Catalog;
use stela_loader::{
    host_firmware::{self as manifest, MAX_MANIFEST_BYTES, Manifest},
    mount_interface::layout::{FIRMWARE, FIRMWARE_MANIFEST, Layout},
};
use uefi::{
    CString16,
    proto::media::file::{Directory, File as _, FileAttribute, FileInfo, FileMode, RegularFile},
};
mod bundle;
mod smbios;
mod wifi;
use stela_drivers::net::wifi::intel::platform::VARIABLE_DESCRIPTORS;

/// At most 4096 siblings per path component; no unbounded directory scans or
/// variable-length firmware metadata allocations.
const MAX_DIRECTORY_ENTRIES: usize = 4096;
#[repr(align(8))]
struct InfoBuffer([u8; 1024]);

/// # Safety
/// This loader still owns every immutable reserved page and its identity
/// alias remains mapped for the resulting catalog's lifetime.
/// The standalone caller runs before EBS, so this emits no phase receipt.
pub unsafe fn identity_catalog(resources: &Resources) -> Result<Option<Arc<Catalog>>> {
    // SAFETY: caller supplied the permanent identity mapping contract.
    unsafe { resources.catalog(|physical| physical, None, output::arguments) }
}

/// # Safety
/// The loader supplied this validated wire handoff. Its LOADER_DATA pages
/// are reserved, and the engine maps them in its permanent shared physmap.
pub unsafe fn after_ebs(resources: &Resources) -> Result<Option<Arc<Catalog>>> {
    let started = output::now();
    // SAFETY: the caller's validated loader handoff and permanent physmap.
    let catalog =
        unsafe { resources.catalog(layout::physical, Some("after-ebs"), output::arguments) }?;
    if let Some(catalog) = &catalog {
        let (images, bytes) = catalog.inventory();
        output::arguments(format_args!(
            "HOST FIRMWARE phase=after-ebs images={images} bytes={bytes} catalog_ms={}\n",
            output::millis_since(started)
        ));
    }
    Ok(catalog)
}

/// Resolve each component by enumerating its directory once. UEFI/FAT
/// case-insensitive matching is accepted only when exactly one entry matches;
/// a second case alias is refused even on firmware with a case-sensitive driver.
fn open(root: &mut Directory, relative: &str) -> Result<Option<RegularFile>> {
    let path = CString16::try_from("\\").map_err(|_| refuse!("invalid firmware root"))?;
    let mut directory = root
        .open(&path, FileMode::Read, FileAttribute::empty())
        .map_err(|_| refuse!("cannot open host firmware root"))?
        .into_directory()
        .ok_or(refuse!("host firmware root is not a directory"))?;
    let mut parts = relative.split('/').peekable();
    while let Some(part) = parts.next() {
        let mut found = None;
        let mut buffer = InfoBuffer([0; 1024]);
        let mut end = false;
        for index in 0..=MAX_DIRECTORY_ENTRIES {
            let Some(info) = directory
                .read_entry(&mut buffer.0)
                .map_err(|_| refuse!("cannot enumerate bounded host firmware path"))?
            else {
                end = true;
                break;
            };
            if index == MAX_DIRECTORY_ENTRIES {
                return Err(refuse!("host firmware directory has too many entries"));
            }
            let name = info.file_name().to_string();
            if name.eq_ignore_ascii_case(part) {
                if found.is_some() {
                    return Err(refuse!("ambiguous host firmware path case aliases"));
                }
                found = Some(name);
            }
        }
        if !end {
            return Err(refuse!("host firmware directory has too many entries"));
        }
        let Some(name) = found else {
            return Ok(None);
        };
        let name = CString16::try_from(name.as_str())
            .map_err(|_| refuse!("host firmware path is not UCS-2"))?;
        let handle = directory
            .open(&name, FileMode::Read, FileAttribute::empty())
            .map_err(|_| refuse!("cannot open present host firmware path"))?;
        if parts.peek().is_none() {
            return handle
                .into_regular_file()
                .map(Some)
                .ok_or(refuse!("host firmware file is a directory"));
        }
        directory = handle
            .into_directory()
            .ok_or(refuse!("host firmware parent is not a directory"))?;
    }
    Err(refuse!("empty host firmware path"))
}

fn read_file(mut file: RegularFile, expected: Option<usize>, limit: usize) -> Result<Region> {
    let mut info = InfoBuffer([0; 1024]);
    let size = file
        .get_info::<FileInfo>(&mut info.0)
        .map_err(|_| refuse!("cannot read bounded host firmware file metadata"))?
        .file_size();
    let length = usize::try_from(size).map_err(|_| refuse!("host firmware file size overflow"))?;
    if length == 0 || length > limit || expected.is_some_and(|n| n != length) {
        return Err(refuse!(
            "host firmware file length is outside its manifest/resource budget"
        ));
    }
    // Size and manifest-wide budgets are checked before this allocation.
    let physical = firmware::pages(length.div_ceil(PAGE))?;
    // SAFETY: fresh reserved LOADER_DATA pages, exclusively written here before
    // any immutable view is published. They are never freed or given to a guest.
    let buffer = unsafe { core::slice::from_raw_parts_mut(physical as *mut u8, length) };
    let mut done = 0;
    while done < length {
        let count = file
            .read(&mut buffer[done..])
            .map_err(|_| refuse!("cannot read host firmware file"))?;
        if count == 0 {
            return Err(refuse!("host firmware file truncated during read"));
        }
        done += count;
    }
    let mut extra = [0];
    if file
        .read(&mut extra)
        .map_err(|_| refuse!("cannot finish host firmware file"))?
        != 0
    {
        return Err(refuse!("host firmware file grew during read"));
    }
    Ok(Region {
        physical: physical as u64,
        bytes: size,
    })
}

fn open_manifest(root: &mut Directory) -> Result<Option<(Layout, RegularFile)>> {
    for layout in Layout::PREFERRED {
        if let Some(file) = open(root, &layout.esp_path(FIRMWARE_MANIFEST))? {
            return Ok(Some((layout, file)));
        }
    }
    Ok(None)
}

/// Absence alone is optional. Any unreadable/malformed present manifest or
/// missing/corrupt resource is a boot refusal. Images belong to the same
/// selected firmware directory and volume as their manifest.
pub fn read(root: &mut Directory) -> Result<Resources> {
    let smbios = smbios::capture();
    let mut wifi =
        [stela_boot_core::host_resources::wifi::Capture::default(); VARIABLE_DESCRIPTORS.len()];
    for (slot, descriptor) in wifi.iter_mut().zip(&VARIABLE_DESCRIPTORS) {
        *slot = wifi::capture(descriptor.variable)?;
    }
    let exact = match open_manifest(root)? {
        Some((layout, file)) => Some((layout, read_file(file, None, MAX_MANIFEST_BYTES)?)),
        None => None,
    };
    let bundled = bundle::read(root);
    if exact.is_none() && bundled.is_none() {
        output::arguments(format_args!("HOST FIRMWARE manifest=absent\n"));
        let resources = Resources {
            wifi,
            smbios,
            ..Resources::absent()
        };
        // SAFETY: freshly captured immutable permanent pages, identity mapped.
        let _ = unsafe {
            resources.catalog(|physical| physical, Some("before-ebs"), output::arguments)
        }?;
        return Ok(resources);
    }
    let exact_bytes = exact.as_ref().map(|(_, region)| {
        // SAFETY: read_file reserved these pages and completed all writes.
        unsafe { core::slice::from_raw_parts(region.physical as *const u8, region.bytes as usize) }
    });
    let parsed = exact_bytes
        .map(Manifest::parse)
        .transpose()
        .map_err(|error| report(error, output::arguments))?;
    // Parse/identity checks and the total budget precede every image allocation.
    let started = output::now();
    let mut regions = alloc::vec::Vec::new();
    if let Some(((layout, _), parsed)) = exact.as_ref().zip(parsed.as_ref()) {
        for image in &parsed.images {
            let path = layout.esp_path(&format!("{FIRMWARE}/{}", image.path));
            let file = open(root, &path)?
                .ok_or(refuse!("host firmware manifest names a missing resource"))?;
            regions.push(read_file(
                file,
                Some(image.length),
                manifest::MAX_IMAGE_BYTES,
            )?);
        }
    }
    let read_ms = output::millis_since(started);
    // Without a bundle, the older manifest is handed off as it was read.
    let region = match (&bundled, exact) {
        (None, Some((_, region))) => region,
        _ => {
            let (objects, bundle_regions) = bundled
                .as_ref()
                .map(|bundled| (bundled.objects(), bundled.regions.as_slice()))
                .unwrap_or_default();
            regions.extend_from_slice(bundle_regions);
            let handoff = manifest::bundle::handoff_manifest(parsed.as_ref(), &objects)
                .map_err(|error| report(error, output::arguments))?;
            let physical = firmware::pages(handoff.len().div_ceil(PAGE))?;
            // SAFETY: fresh reserved LOADER_DATA pages, written once here
            // before any immutable view is published.
            unsafe {
                core::ptr::copy_nonoverlapping(handoff.as_ptr(), physical as *mut u8, handoff.len())
            };
            Region {
                physical: physical as u64,
                bytes: handoff.len() as u64,
            }
        }
    };
    if regions.len() > manifest::MAX_IMAGES {
        return Err(refuse!(
            "host firmware resources exceed the handoff's slots"
        ));
    }
    let mut resources = Resources {
        present: true,
        manifest: region,
        count: regions.len(),
        wifi,
        smbios,
        ..Resources::absent()
    };
    resources.images[..regions.len()].copy_from_slice(&regions);
    let started = output::now();
    // SAFETY: only this loader owns these permanent pages; identity alias is
    // currently mapped by firmware. Drop metadata before engine relocation.
    let catalog =
        unsafe { resources.catalog(|physical| physical, Some("before-ebs"), output::arguments) }?
            .ok_or(refuse!("missing host firmware catalog"))?;
    let (images, total) = catalog.inventory();
    output::arguments(format_args!(
        "HOST FIRMWARE phase=before-ebs images={images} bytes={total} read_ms={read_ms} \
         catalog_ms={}\n",
        output::millis_since(started)
    ));
    Ok(resources)
}
