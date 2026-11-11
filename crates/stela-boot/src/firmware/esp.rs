// SPDX-License-Identifier: GPL-2.0-or-later
//! Compatibility lookups on the boot volume's EFI system partition.
use crate::Result;
use alloc::vec::Vec;
use stela_boot_core::options::DEFAULT_WIFI_KEY;
use stela_contracts::refuse;
use stela_loader::mount_interface::layout::{
    CHOICES, CONFIG, Layout, OPTIONS, RECORD, SELECTED_DISK, WIFI_KEY,
};
use stela_runtime::pc::{gpt::Guid, swap::volume::BootEsp};
use uefi::{
    CString16, Handle,
    boot::{self, OpenProtocolAttributes, OpenProtocolParams},
    fs::{FileSystem, PathBuf},
    proto::{
        device_path::{
            DevicePath, DevicePathNodeEnum,
            media::{PartitionFormat, PartitionSignature},
        },
        loaded_image::LoadedImage,
    },
};

static LAYOUT_CAPABILITY: [u8; Layout::MOUNT_CAPABILITY.len()] = Layout::MOUNT_CAPABILITY;

/// Capture the loaded image's terminal GPT partition node before firmware exit.
pub(crate) fn boot_esp() -> Option<BootEsp> {
    let image = boot::image_handle();
    let device = boot::open_protocol_exclusive::<LoadedImage>(image)
        .ok()?
        .device()?;
    let (guid, start, blocks) = gpt_partition(device)?;
    BootEsp::new(guid, start, blocks)
}

/// The GUID and extent of the terminal GPT partition on a firmware handle.
pub(crate) fn gpt_partition(device: Handle) -> Option<(Guid, u64, u64)> {
    // SAFETY: read the device path without changing the firmware handle.
    let path = unsafe {
        boot::open_protocol::<DevicePath>(
            OpenProtocolParams {
                handle: device,
                agent: boot::image_handle(),
                controller: None,
            },
            OpenProtocolAttributes::GetProtocol,
        )
    }
    .ok()?;
    if path.instance_iter().count() != 1 {
        return None;
    }
    let DevicePathNodeEnum::MediaHardDrive(drive) = path.node_iter().last()?.as_enum().ok()? else {
        return None;
    };
    if drive.partition_format() != PartitionFormat::GPT {
        return None;
    }
    let PartitionSignature::Guid(guid) = drive.partition_signature() else {
        return None;
    };
    let guid = Guid(guid.to_bytes());
    (!guid.is_zero()).then(|| (guid, drive.partition_start(), drive.partition_size()))
}

fn path(text: &str) -> Result<PathBuf> {
    CString16::try_from(text)
        .map(PathBuf::from)
        .map_err(|_| refuse!("boot volume path is not UCS-2"))
}

/// Openable path in the preferred layout. A read failure on an existing file
/// belongs to that file and must reach the caller.
pub(crate) fn find(volume: &mut FileSystem, name: &str) -> Result<Option<PathBuf>> {
    core::hint::black_box(&LAYOUT_CAPABILITY);
    let layouts = if name == SELECTED_DISK && recorded_layout(volume)? == Some(Layout::Legacy) {
        [Layout::Legacy, Layout::Northwood]
    } else {
        Layout::PREFERRED
    };
    for layout in layouts {
        let candidate = path(&layout.uefi_path(name))?;
        if volume
            .try_exists(&candidate)
            .map_err(|_| refuse!("cannot locate {name} on the boot volume", name))?
        {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

fn recorded_layout(volume: &mut FileSystem) -> Result<Option<Layout>> {
    for layout in Layout::PREFERRED {
        if volume
            .try_exists(path(&layout.uefi_path(RECORD))?)
            .map_err(|_| refuse!("cannot locate {RECORD} on the boot volume", RECORD))?
        {
            return Ok(Some(layout));
        }
    }
    Ok(None)
}

/// Keep a setup's selected disk beside its record. Standalone media
/// without a record keep the location selected by their settings.
pub(crate) fn destination(volume: &mut FileSystem, name: &str) -> Result<PathBuf> {
    if let Some(layout) = recorded_layout(volume)? {
        return path(&layout.uefi_path(name));
    }
    for setting in [name, CONFIG, OPTIONS, CHOICES] {
        let candidate = path(&Layout::Northwood.uefi_path(setting))?;
        if volume
            .try_exists(candidate)
            .map_err(|_| refuse!("cannot locate {setting} on the boot volume", setting))?
        {
            return path(&Layout::Northwood.uefi_path(name));
        }
    }
    path(&Layout::Legacy.uefi_path(name))
}

/// An explicit key path wins. The two default locations can find each
/// other on an installer in the middle of the path transition.
pub(crate) fn wifi_key(volume: &mut FileSystem, configured: &str) -> Option<Vec<u8>> {
    let primary = path(configured).ok()?;
    if volume.try_exists(&primary).ok()? {
        return volume.read(primary).ok();
    }
    let other = if configured.eq_ignore_ascii_case(northwood_config::DEFAULT_WIFI_KEY) {
        Layout::Legacy
    } else if configured.eq_ignore_ascii_case(DEFAULT_WIFI_KEY) {
        Layout::Northwood
    } else {
        return None;
    };
    volume.read(path(&other.uefi_path(WIFI_KEY)).ok()?).ok()
}
