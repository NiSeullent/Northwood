// SPDX-License-Identifier: GPL-2.0-or-later
//! Primary ACPI IDs published in the firmware's device paths.

use uefi::{
    boot::{self, OpenProtocolAttributes, OpenProtocolParams},
    proto::device_path::{DevicePath, DeviceSubType, DeviceType},
};

/// The Expanded ACPI node contains numeric HID, UID and CID followed by
/// their three null-terminated ASCII strings. A match uses the primary HID.
fn expanded_primary_hid(data: &[u8]) -> Option<&[u8]> {
    let mut strings = data.get(12..)?.splitn(4, |byte| *byte == 0);
    let hid = strings.next()?;
    match (strings.next(), strings.next(), strings.next()) {
        (Some(_), Some(_), Some([])) if !hid.is_empty() => Some(hid),
        _ => None,
    }
}

/// Observe an exact primary HID while boot services are live. Device paths
/// are a positive witness only: firmware need not publish every ACPI device.
pub(crate) fn firmware_primary_hid(hid: &str) -> Option<bool> {
    if hid.is_empty() {
        return None;
    }
    for handle in boot::find_handles::<DevicePath>().ok()? {
        // SAFETY: GetProtocol borrows the firmware's published device path
        // for this loop iteration. The program only reads that protocol.
        let Ok(path) = (unsafe {
            boot::open_protocol::<DevicePath>(
                OpenProtocolParams {
                    handle,
                    agent: boot::image_handle(),
                    controller: None,
                },
                OpenProtocolAttributes::GetProtocol,
            )
        }) else {
            continue;
        };
        if path.node_iter().any(|node| {
            node.full_type() == (DeviceType::ACPI, DeviceSubType::ACPI_EXPANDED)
                && expanded_primary_hid(node.data()) == Some(hid.as_bytes())
        }) {
            return Some(true);
        }
    }
    None
}
