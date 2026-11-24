// SPDX-License-Identifier: GPL-2.0-or-later
//! Pre-EBS physical-host SMBIOS source acquisition only. UEFI 2.10 §2.3.4
//! grants identity mappings only for described memory; §4.6 config GUIDs and
//! DSP0134 §5.2 authorize reading their published tables. Never use this after
//! root replacement/EBS, or for undescribed/reserved/MMIO/protected memory.
use crate::paging::PAGE;
use stela_boot_core::host_resources::smbios::{
    Capture, Failure, Span, Stage, capture_with, failure, grant, select,
};
use stela_drivers::firmware::smbios::{Error, Format};
use uefi::{
    boot::{self, MemoryType},
    mem::memory_map::MemoryMap,
    table::cfg::ConfigTableEntry,
};

fn read(address: u64, output: &mut [u8], stage: Stage) -> Result<(), Failure> {
    let map = boot::memory_map(MemoryType::LOADER_DATA).map_err(|error| Failure {
        status: Some(error.status().0 as u64),
        ..failure(stage, Error::MemoryMapUnavailable, address, output.len())
    })?;
    if map.entries().len() > 4096 {
        return Err(failure(
            stage,
            Error::MemoryMapUnavailable,
            address,
            output.len(),
        ));
    }
    let cr4: u64;
    // SAFETY: read-only x64 control-register observation during UEFI.
    unsafe {
        core::arch::asm!("mov {}, cr4",out(reg)cr4,options(nomem,nostack,preserves_flags));
    }
    grant(
        map.entries().map(|entry| Span {
            start: entry.phys_start,
            pages: entry.page_count,
            kind: entry.ty,
            attributes: entry.att.bits(),
        }),
        address,
        output.len(),
        if cr4 & (1 << 12) != 0 { 56 } else { 47 },
    )
    .map_err(|error| failure(stage, error, address, output.len()))?;
    // SAFETY: published configuration table, boot services still live,
    // checked complete readable data-RAM extent and canonical pointer under the
    // UEFI x64 identity-map contract above. No Stela root has been installed.
    unsafe {
        core::ptr::copy_nonoverlapping(address as *const u8, output.as_mut_ptr(), output.len());
    }
    Ok(())
}
pub(super) fn capture() -> Capture {
    let selected = uefi::system::with_config_table(|entries| {
        select(entries.iter().filter_map(|entry| {
            let format = if entry.guid == ConfigTableEntry::SMBIOS3_GUID {
                Format::Smbios3
            } else if entry.guid == ConfigTableEntry::SMBIOS_GUID {
                Format::Smbios2
            } else {
                return None;
            };
            Some((format, entry.address as u64))
        }))
    });
    let capture = capture_with(selected, read, |bytes| {
        let page = boot::allocate_pages(
            boot::AllocateType::AnyPages,
            MemoryType::LOADER_DATA,
            bytes.div_ceil(PAGE),
        )
        .map_err(|error| Failure {
            status: Some(error.status().0 as u64),
            ..failure(Stage::Table, Error::AllocationFailed, 0, bytes)
        })?;
        // SAFETY: fresh exclusive allocation, immutable after publication and
        // excluded from guest allocation by the existing LOADER_DATA pipeline.
        Ok(unsafe { core::slice::from_raw_parts_mut(page.as_ptr(), bytes) })
    });
    record_board(&capture);
    capture
}

/// Records the system's vendor and product name, model names the user may
/// show, for the guest's `StelaHost` (`crate::host_inventory`). The serial
/// number, UUID and asset tag are never read.
fn record_board(capture: &Capture) {
    use stela_drivers::firmware::smbios::{Observation, entry, entry_length, identity};
    let (Some(format), true) = (
        capture.format,
        capture.state == stela_boot_core::host_resources::smbios::State::Present,
    ) else {
        return;
    };
    // SAFETY: the capture's own immutable LOADER_DATA pages, identity mapped.
    let bytes = unsafe {
        core::slice::from_raw_parts(
            capture.region.physical as *const u8,
            capture.region.bytes as usize,
        )
    };
    let Ok(length) = entry_length(bytes, format) else {
        return;
    };
    let Some(Ok(parsed)) = bytes.get(..length).map(|raw| entry(raw, format)) else {
        return;
    };
    let Ok(system) = identity(parsed, &bytes[length..]) else {
        return;
    };
    let table = &bytes[length..];
    let observed = |value| match value {
        Observation::Value(value) => Some(value),
        _ => None,
    };
    use stela_boot_core::host_identity::{intel_macbook_pro, smbios_string, smbios_text};
    for (key, value) in [
        ("SystemVendor", observed(system.system_vendor)),
        ("SystemModel", observed(system.product_name)),
        ("BoardVendor", observed(system.board_vendor)),
        ("BoardModel", smbios_string(table, 2, 5)),
        ("FirmwareVersion", smbios_string(table, 0, 5)),
    ] {
        if let Some(value) = value.and_then(smbios_text) {
            crate::host_inventory::text(key, alloc::string::String::from_utf8_lossy(value));
        }
    }
    if let Some(model) = observed(system.system_vendor)
        .zip(observed(system.product_name))
        .and_then(|(vendor, product)| intel_macbook_pro(vendor, product))
    {
        let bridge = model.expected_bridge.label();
        crate::host_inventory::text("AppleExpectedBridge", bridge);
        crate::output::arguments(format_args!(
            "HOST APPLE model={} expected_bridge={bridge}\n",
            model.model
        ));
    }
}
