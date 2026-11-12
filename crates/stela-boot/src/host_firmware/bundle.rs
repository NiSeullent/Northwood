//! The complete host firmware bundle `stela setup` puts on the USB's data
//! volume (`stela_loader::host_firmware::bundle`). Before boot services end
//! this reads only the files the PC's PCI functions need, logs each one
//! and the files the bundle lacks, and leaves the bundle as it is. A bundle
//! that cannot be read leaves the boot without it, as the log says; the
//! devices that needed its files stay off and their drivers say so.
use super::{open, read_file};
use crate::{handoff::Region, output, pci_io};
use alloc::{format, string::String, vec::Vec};
use stela_loader::host_firmware::{
    MAX_IMAGE_BYTES, MAX_MANIFEST_BYTES,
    bundle::{self, Bundle, Object, Pci},
};
use uefi::{
    boot,
    proto::media::{file::Directory, fs::SimpleFileSystem},
};

/// The files read from the bundle, in the order of their regions.
pub struct Bundled {
    bundle: Bundle<'static>,
    selected: Vec<usize>,
    pub regions: Vec<Region>,
}

impl Bundled {
    pub fn objects(&self) -> Vec<&Object<'static>> {
        self.selected
            .iter()
            .map(|&index| &self.bundle.objects[index])
            .collect()
    }
}

/// The bundle on the boot volume `root`, else on the first other volume
/// that has one.
pub fn read(root: &mut Directory) -> Option<Bundled> {
    if let Some(found) = load(root, "boot volume") {
        return report(found);
    }
    let handles = boot::find_handles::<SimpleFileSystem>().ok()?;
    for (index, handle) in handles.into_iter().enumerate() {
        // The boot volume itself is held already and refuses this open.
        let Ok(mut volume) = boot::open_protocol_exclusive::<SimpleFileSystem>(handle) else {
            continue;
        };
        let Ok(mut root) = volume.open_volume() else {
            continue;
        };
        if let Some(found) = load(&mut root, &format!("volume {index}")) {
            return report(found);
        }
    }
    None
}

fn report(found: Result<Bundled, String>) -> Option<Bundled> {
    match found {
        Ok(bundled) => Some(bundled),
        Err(why) => {
            output::arguments(format_args!(
                "HOST FIRMWARE BUNDLE unreadable, the boot goes on without it: {why}\n"
            ));
            None
        }
    }
}

/// The PCI functions the firmware's drivers found.
fn functions() -> Vec<Pci> {
    match pci_io::capture() {
        Ok(census) => census
            .functions
            .iter()
            .map(|function| {
                let identity = function.identity();
                Pci {
                    vendor: identity.vendor,
                    device: identity.device,
                    class: identity.class[0],
                }
            })
            .collect(),
        Err(error) => {
            output::arguments(format_args!(
                "HOST FIRMWARE BUNDLE no PCI census ({error:?}): USB Bluetooth files only\n"
            ));
            Vec::new()
        }
    }
}

/// The bundle in `root`, if it has one.
fn load(root: &mut Directory, place: &str) -> Option<Result<Bundled, String>> {
    let file = match open(root, bundle::MANIFEST) {
        Ok(Some(file)) => file,
        Ok(None) => return None,
        Err(error) => return Some(Err(format!("{error}"))),
    };
    Some(load_from(root, file, place))
}

fn load_from(
    root: &mut Directory,
    file: uefi::proto::media::file::RegularFile,
    place: &str,
) -> Result<Bundled, String> {
    let started = output::now();
    let region = read_file(file, None, MAX_MANIFEST_BYTES).map_err(|error| format!("{error}"))?;
    // SAFETY: read_file reserved these pages and completed all writes; they
    // are never freed or written again.
    let bytes: &'static [u8] =
        unsafe { core::slice::from_raw_parts(region.physical as *const u8, region.bytes as usize) };
    let parsed = Bundle::parse(bytes).map_err(|error| error.0)?;
    let functions = functions();
    let selection = bundle::select(&functions);
    for why in &selection.unknown {
        output::arguments(format_args!(
            "HOST FIRMWARE BUNDLE NO RECIPE for the NVIDIA display device at {why}: its \
             firmware is not in Stela's catalog\n"
        ));
    }
    let mut bundled = Bundled {
        selected: Vec::new(),
        regions: Vec::new(),
        bundle: parsed,
    };
    let mut lacking = 0;
    for (artifact, why) in &selection.wanted {
        let Some(index) = bundled
            .bundle
            .objects
            .iter()
            .position(|object| object.sha256 == artifact.sha256)
        else {
            lacking += 1;
            output::arguments(format_args!(
                "HOST FIRMWARE BUNDLE LACKS {} for {why}: `stela setup repair` adds it\n",
                artifact.member
            ));
            continue;
        };
        let object = &bundled.bundle.objects[index];
        let path = format!("{}/{}", bundle::DIRECTORY, object.path);
        let read = match open(root, &path) {
            Ok(Some(file)) => read_file(file, Some(object.length), MAX_IMAGE_BYTES),
            Ok(None) => Err(stela_contracts::refuse!("the bundle names a missing file")),
            Err(error) => Err(error),
        };
        match read {
            Ok(region) => {
                output::arguments(format_args!(
                    "HOST FIRMWARE BUNDLE USE {} bytes={} for {why}\n",
                    object.member, object.length
                ));
                bundled.selected.push(index);
                bundled.regions.push(region);
            }
            Err(error) => {
                lacking += 1;
                output::arguments(format_args!(
                    "HOST FIRMWARE BUNDLE LACKS {} for {why}: {error}\n",
                    object.member
                ));
            }
        }
    }
    let read_bytes: usize = bundled.objects().iter().map(|object| object.length).sum();
    output::arguments(format_args!(
        "HOST FIRMWARE BUNDLE {place} files={} pci={} used={} lacking={lacking} bytes={read_bytes} \
         read_ms={}\n",
        bundled.bundle.objects.len(),
        functions.len(),
        bundled.selected.len(),
        output::millis_since(started)
    ));
    Ok(bundled)
}
