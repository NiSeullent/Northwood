// SPDX-License-Identifier: GPL-2.0-or-later
//! Read-only Gemini Lake LPSS PCI observations before native driver publication.

use super::{
    Cohort, ConfigInventory, ContainedPci, Error, Fabric, Slot, Source, mapping_error,
    observations::{
        command_and_status, host_bar_state, original_root, read_capability_state, root_bar_error,
    },
};
use crate::output;
use stela_boot_core::firmware_pci::lpss::{self, BAR0_PROBE_BYTES, Bar0};
use stela_drivers::bus::pci::regs;

impl ContainedPci<'_> {
    /// # Safety
    /// The original parked fabric excludes competing configuration writers.
    pub(crate) unsafe fn report_lpss_attachment(&self) {
        let Ok(cohort) = self.source.pci_slots.0.try_borrow() else {
            return;
        };
        let Cohort::Parked(fabric) = &*cohort else {
            return;
        };
        let mut slots = fabric
            .slots
            .iter()
            .filter(|slot| lpss::matches(slot.header.identity))
            .peekable();
        if slots.peek().is_none() {
            return;
        }
        // SAFETY: the parked original fabric still excludes configuration
        // writers. Inventory reads no controller registers or BAR contents.
        let result = unsafe {
            self.source.with_config_inventory(|_, reader| {
                for slot in slots {
                    if report(self.source, fabric, slot, reader).is_err() {
                        let address = slot.address;
                        let identity = slot.header.identity;
                        output::arguments(format_args!(
                            "HOST LPSS-I2C BAR0 pci={address} id={:04x}:{:04x} state=config-unavailable\n",
                            identity as u16, identity >> 16,
                        ));
                    }
                }
                Ok::<_, Error>(())
            })
        };
        if let Err(error) = result {
            output::arguments(format_args!(
                "HOST LPSS-I2C inventory=unavailable error={error:?}\n"
            ));
        }
    }
}

fn report(
    source: &Source,
    fabric: &Fabric,
    slot: &Slot,
    reader: &mut ConfigInventory<'_>,
) -> Result<(), Error> {
    let address = slot.address;
    let vendor = slot.header.identity as u16;
    let device = slot.header.identity >> 16;
    let word = command_and_status(reader, slot)?;
    let command = word as u16;
    let mse = command & regs::COMMAND_MEMORY != 0;
    let bme = command & regs::COMMAND_MASTER != 0;
    let intx_disabled = command & regs::COMMAND_INTX_DISABLE != 0;
    let intx_asserted = word >> 16 & (1 << 3) != 0;
    let contained = slot
        .check_stopped(|offset, width| reader.read(address, offset, width).map_err(mapping_error))
        .is_ok();
    output::arguments(format_args!(
        "HOST LPSS-I2C PCI pci={address} id={vendor:04x}:{device:04x} mse={mse} bme={bme} intx-disabled={intx_disabled} intx-asserted={intx_asserted} contained={contained} stop-evidence=unavailable\n"
    ));
    match read_capability_state(reader, address) {
        Ok(state) => output::arguments(format_args!(
            "HOST LPSS-I2C CAPS pci={address} id={vendor:04x}:{device:04x} pm-state={:?} msi-enabled={:?} msix-enabled={:?} pcie-pending={:?}\n",
            state.pm_state, state.msi_enabled, state.msix_enabled, state.pcie_pending,
        )),
        Err(_) => output::arguments(format_args!(
            "HOST LPSS-I2C CAPS pci={address} id={vendor:04x}:{device:04x} state=unavailable\n"
        )),
    }

    let low = reader.read32(address, 0x10)?;
    let high = reader.read32(address, 0x14)?;
    let windows = source.platform.wire().roots();
    let translated = Bar0::observe(slot.header.layout, low, high)
        .ok_or("raw-bar0-refused")
        .and_then(|bar| {
            let root = original_root(windows, fabric, slot).ok_or("original-root-unavailable")?;
            let host = bar
                .original_root_host(root, windows)
                .map_err(root_bar_error)?;
            Ok((bar.pci_base, host))
        });
    match translated {
        Ok((bus, host)) => {
            let state = host_bar_state(source, bus, host, BAR0_PROBE_BYTES);
            output::arguments(format_args!(
                "HOST LPSS-I2C BAR0 pci={address} id={vendor:04x}:{device:04x} bus={bus:#x} host={host:#x} bytes={BAR0_PROBE_BYTES} state={state}\n"
            ));
        }
        Err(state) => output::arguments(format_args!(
            "HOST LPSS-I2C BAR0 pci={address} id={vendor:04x}:{device:04x} state={state}\n"
        )),
    }
    Ok(())
}
