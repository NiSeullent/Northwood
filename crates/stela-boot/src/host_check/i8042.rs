// SPDX-License-Identifier: GPL-2.0-or-later
//! Firmware table observations for the i8042 port probe.
use crate::{cpu, output, storage};
use stela_boot_core::madt::Routing;
use stela_drivers::probe::I8042Fadt;
use stela_runtime::pc::acpi::{self, AcpiError};

pub(super) fn observe_firmware() {
    let Some(rsdp) = storage::rsdp() else {
        output::arguments(format_args!("HOST I8042 FIRMWARE rsdp=unavailable\n"));
        return;
    };
    // SAFETY: Diagnostics still runs with the firmware's ACPI tables mapped.
    let fadt = match unsafe { acpi::find(rsdp, b"FACP") } {
        Ok(table) => Ok(I8042Fadt::observe(Some(table))),
        Err(AcpiError::Missing(_)) => Ok(I8042Fadt::observe(None)),
        Err(error) => Err(error),
    };
    output::arguments(format_args!("HOST I8042 FIRMWARE fadt={fadt:?}\n"));

    // SAFETY: same firmware ACPI mapping as the FADT observation above.
    let table = match unsafe { acpi::find(rsdp, b"APIC") } {
        Ok(table) => table,
        Err(error) => {
            output::arguments(format_args!("HOST I8042 FIRMWARE madt_lookup={error:?}\n"));
            return;
        }
    };
    match Routing::parse(table, cpu::physical_bits()) {
        Ok(routing) => output::arguments(format_args!(
            "HOST I8042 FIRMWARE madt_legacy_pic={} irq1={:?} irq12={:?}\n",
            routing.has_legacy_pic(),
            routing.isa(1),
            routing.isa(12),
        )),
        Err(error) => output::arguments(format_args!("HOST I8042 FIRMWARE madt_parse={error:?}\n")),
    }
}
