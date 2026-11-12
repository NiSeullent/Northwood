// SPDX-License-Identifier: GPL-2.0-or-later
//! Read-only observations from the original, parked PCI fabric.

use super::{Address, ConfigInventory, Error, Fabric, PAGE, Slot, Source, mapping_error, topology};
use core::cell::RefCell;
use stela_boot_core::{ecam::RootWindow, firmware_pci::glk::RootBarError};
use stela_drivers::bus::pci::{
    CAPABILITY_EXPRESS, CAPABILITY_MSI, CAPABILITY_MSIX, CAPABILITY_POWER, Capabilities,
    CapabilityError, Layout, capability::Transport,
};

pub(super) fn command_and_status(
    reader: &mut ConfigInventory<'_>,
    slot: &Slot,
) -> Result<u32, Error> {
    let address = slot.address;
    for offset in [0, 8, 0x0c] {
        if slot.header.word(offset) != Some(reader.read32(address, offset)?) {
            return Err(Error::InvalidPlatformReply);
        }
    }
    let word = reader.read32(address, 4)?;
    if word == u32::MAX {
        return Err(Error::InvalidPlatformReply);
    }
    Ok(word)
}

pub(super) fn original_root<'a, 'fabric>(
    roots: &'a [RootWindow],
    fabric: &'fabric Fabric,
    mut slot: &'fabric Slot,
) -> Option<&'a RootWindow> {
    loop {
        match slot.route {
            topology::Parent::Root(index) => {
                return roots.get(index).filter(|root| root.space == 2);
            }
            topology::Parent::Bridge(address) => {
                let index = fabric
                    .slots
                    .binary_search_by_key(&address, |slot| slot.address)
                    .ok()?;
                let parent = &fabric.slots[index];
                if parent.address.segment != slot.address.segment
                    || parent.address.bus >= slot.address.bus
                {
                    return None;
                }
                slot = parent;
            }
        }
    }
}

pub(super) fn root_bar_error(error: RootBarError) -> &'static str {
    match error {
        RootBarError::NoWindow => "no-covering-root-window",
        RootBarError::Ambiguous => "ambiguous-root-window",
        RootBarError::Conflicting => "conflicting-root-windows",
        RootBarError::InvalidTranslation => "invalid-root-translation",
    }
}

pub(super) fn host_bar_state(source: &Source, bus: u64, host: u64, bytes: u64) -> &'static str {
    let Some(end) = host.checked_add(bytes) else {
        return "invalid-host-range";
    };
    if !host.is_multiple_of(PAGE) {
        "invalid-host-range"
    } else if !source.platform.wire().excludes_memory(host, end) {
        "host-memory-conflict"
    } else if host != bus {
        "translated"
    } else {
        "identity"
    }
}

#[derive(Default)]
pub(super) struct CapabilityState {
    pub(super) pm_state: Option<u16>,
    pub(super) msi_enabled: Option<u16>,
    pub(super) msix_enabled: Option<u16>,
    pub(super) pcie_pending: Option<u16>,
}

pub(super) fn read_capability_state(
    reader: &mut ConfigInventory<'_>,
    address: Address,
) -> Result<CapabilityState, CapabilityError> {
    let reader = RefCell::new(reader);
    let read = |offset, width| {
        reader
            .borrow_mut()
            .read(address, offset, width)
            .map_err(|error| CapabilityError::Config {
                offset,
                error: mapping_error(error),
            })
    };
    let mut state = CapabilityState::default();
    for capability in Capabilities::new(Transport(read)) {
        let capability = capability?;
        let (field, relative, shift, mask) = match capability.id {
            CAPABILITY_POWER => (&mut state.pm_state, 4, 0, 3),
            CAPABILITY_MSI => (&mut state.msi_enabled, 2, 0, 1),
            CAPABILITY_MSIX => (&mut state.msix_enabled, 2, 15, 1),
            CAPABILITY_EXPRESS => (&mut state.pcie_pending, 10, 5, 1),
            _ => continue,
        };
        let offset = capability.offset + relative;
        if capability.layout != Layout::Sized || capability.bytes < relative + 2 || field.is_some()
        {
            return Err(CapabilityError::Malformed { offset });
        }
        let value = read(offset, 2)? as u16;
        if value == u16::MAX {
            return Err(CapabilityError::FailedRead { offset });
        }
        *field = Some(value >> shift & mask);
    }
    Ok(state)
}
