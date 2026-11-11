// SPDX-License-Identifier: GPL-2.0-or-later
//! Read-only post-EBS observations for a future native GLK attachment.

use super::{
    Address, Cohort, ConfigInventory, ContainedPci, Error, Fabric, Slot, Source, mapping_error,
    observations::{
        command_and_status, host_bar_state, original_root, read_capability_state, root_bar_error,
    },
};
use crate::{console::Framebuffer, output};
use stela_boot_core::{
    ecam::RootWindow,
    firmware_pci::glk::{self, BAR0_BYTES, Bar0},
};
use stela_drivers::bus::pci::regs;

impl ContainedPci<'_> {
    /// # Safety
    /// The parked PCI fabric excludes competing configuration writers.
    pub(crate) unsafe fn report_glk_attachment(&self, screen: Option<Framebuffer>) {
        let Ok(cohort) = self.source.pci_slots.0.try_borrow() else {
            return;
        };
        let Cohort::Parked(fabric) = &*cohort else {
            return;
        };
        let mut slots = fabric
            .slots
            .iter()
            .filter(|slot| glk::matches(slot.header.identity, slot.header.class_revision))
            .peekable();
        if slots.peek().is_none() {
            return;
        }
        // SAFETY: this is the same original read-only inventory used during
        // containment. No BAR access or device configuration write occurs.
        let result = unsafe {
            self.source.with_config_inventory(|_, reader| {
                for slot in slots {
                    if let Err(error) = report(self.source, fabric, slot, reader, screen) {
                        let address = slot.address;
                        output::arguments(format_args!(
                            "HOST GLK ATTACH pci={address} config=unavailable error={error:?} attach=refused-no-engine-stop-receipt\n"
                        ));
                    }
                }
                Ok::<_, Error>(())
            })
        };
        if let Err(error) = result {
            output::arguments(format_args!(
                "HOST GLK ATTACH inventory=unavailable error={error:?}\n"
            ));
        }
    }
}

fn report(
    source: &Source,
    fabric: &Fabric,
    slot: &Slot,
    reader: &mut ConfigInventory<'_>,
    screen: Option<Framebuffer>,
) -> Result<(), Error> {
    let address = slot.address;
    let word = command_and_status(reader, slot)?;
    let contained = slot
        .check_stopped(|offset, width| reader.read(address, offset, width).map_err(mapping_error));
    let command = word as u16;
    let intx_asserted = word >> 16 & (1 << 3) != 0;
    let mse = command & regs::COMMAND_MEMORY != 0;
    let bme = command & regs::COMMAND_MASTER != 0;
    let intx_disabled = command & regs::COMMAND_INTX_DISABLE != 0;
    let gop = match screen {
        None => "absent",
        Some(_)
            if source
                .verified_framebuffer()
                .is_some_and(|gop| gop.address == address) =>
        {
            "same-function-verified"
        }
        Some(screen) => match screen.pci_source {
            Some(gop) if gop.address == address => "same-function-unverified",
            Some(_) => "other-function",
            None => "source-unavailable",
        },
    };
    let device = slot.header.identity >> 16;
    output::arguments(format_args!(
        "HOST GLK ATTACH pci={address} device={device:04x} mse={mse} bme={bme} intx_disabled={intx_disabled} intx_asserted={intx_asserted} pci_controls={contained:?} gop={gop} firmware_exclusion=unproven irq_drain=unproven attach=refused-no-engine-stop-receipt\n"
    ));
    match read_capability_state(reader, address) {
        Ok(state) => output::arguments(format_args!(
            "HOST GLK ATTACH pci={address} pm_state={:?} msi_enabled={:?} msix_enabled={:?} pcie_pending={:?}\n",
            state.pm_state, state.msi_enabled, state.msix_enabled, state.pcie_pending,
        )),
        Err(error) => output::arguments(format_args!(
            "HOST GLK ATTACH pci={address} capabilities=unavailable error={error:?}\n"
        )),
    }
    let low = reader.read32(address, 0x10)?;
    let high = reader.read32(address, 0x14)?;
    report_bar(
        source,
        address,
        original_root(source.platform.wire().roots(), fabric, slot),
        Bar0::observe(slot.header.layout, low, high),
    );
    Ok(())
}

fn report_bar(source: &Source, address: Address, root: Option<&RootWindow>, bar: Option<Bar0>) {
    let Some(bar) = bar else {
        output::arguments(format_args!(
            "HOST GLK ATTACH pci={address} bar0=raw-bar0-refused\n"
        ));
        return;
    };
    let bus = bar.pci_base;
    let host = match root.map(|root| bar.original_root_host(root, source.platform.wire().roots())) {
        Some(Ok(host)) => Ok(host),
        Some(Err(error)) => Err(root_bar_error(error)),
        None => Err("original-root-unavailable"),
    };
    let host = match host {
        Ok(host) => host,
        Err(reason) => {
            output::arguments(format_args!(
                "HOST GLK ATTACH pci={address} bar0_bus={bus:#x} bar0={reason}\n"
            ));
            return;
        }
    };
    let reason = host_bar_state(source, bus, host, BAR0_BYTES);
    output::arguments(format_args!(
        "HOST GLK ATTACH pci={address} bar0_bus={bus:#x} bar0_host={host:#x} bar0_expected_bytes={BAR0_BYTES} bar0_source=original-root-window bar0={reason}\n"
    ));
}
