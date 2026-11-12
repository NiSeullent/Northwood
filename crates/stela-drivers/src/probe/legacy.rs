// SPDX-License-Identifier: GPL-2.0-or-later
//! The probe's existing attach paths for legacy PCI ownership.
use super::{Adapter, Devices, Error, Warning};
use crate::{
    Handoff,
    block::{BlockDevice, BlockError},
    bus::{
        dma::DmaAllocator,
        pci::{Address, Function, power::Timer},
    },
    firmware::Provider,
    input::{InputDevice, virtio::VirtioInput},
    net::{
        alx::Alx,
        atl1c::Atl1c,
        e1000e::{E1000e, pch::cold::attach as pch},
        igb::Igb,
        igc::Igc,
        realtek::Realtek,
        virtio::VirtioNet,
    },
    storage::{ahci, nvme, rtsx, sdhci, virtio},
    usb::xhci,
};
use alloc::{boxed::Box, rc::Rc, vec::Vec};
use stela_driver_base::budget::Budget;

pub(super) struct Context<'a> {
    pub handoff: &'a Handoff,
    pub dma: &'a mut dyn DmaAllocator,
    pub firmware: &'static dyn Provider,
    pub minimum: &'a Rc<dyn Timer>,
    pub budget: Option<&'a Budget>,
    pub devices: &'a mut Devices,
}

/// The scan has granted exclusive post-ExitBootServices PCI and DMA ownership.
pub(super) type Attach = unsafe fn(Function, &mut Context<'_>);

/// A failed attach may have enabled bus mastering or interrupts before the
/// budget elapsed. Quiesce only while the probe still holds the function.
fn quiesce_if_expired(budget: Option<&Budget>, function: &mut Function) {
    if budget.is_some_and(|budget| budget.expired().is_some()) {
        let _ = function.abandon();
    }
}

fn disks(
    devices: &mut Devices,
    address: Address,
    result: Result<Vec<Box<dyn BlockDevice>>, BlockError>,
) {
    match result {
        Ok(found) => devices.disks.extend(found),
        Err(error) => devices.failures.push((address, error.into())),
    }
}

fn link<T, E: Into<Error>>(
    devices: &mut Devices,
    address: Address,
    result: Result<T, E>,
    variant: impl FnOnce(T) -> Adapter,
) {
    match result {
        Ok(device) => devices.nics.push(variant(device)),
        Err(error) => devices.failures.push((address, error.into())),
    }
}

pub(super) unsafe fn virtio_block(mut function: Function, context: &mut Context<'_>) {
    let address = function.address;
    let devices = &mut *context.devices;
    let handoff = context.handoff;
    let minimum = context.minimum;
    let dma = &mut *context.dma;
    let attached = virtio::VirtioBlk::new(&mut function, handoff, dma, minimum.as_ref())
        .map(|device| vec![Box::new(device) as Box<dyn BlockDevice>]);
    if attached.is_err() {
        quiesce_if_expired(context.budget, &mut function);
    }
    disks(devices, address, attached);
}

pub(super) unsafe fn nvme(mut function: Function, context: &mut Context<'_>) {
    let address = function.address;
    let devices = &mut *context.devices;
    let handoff = context.handoff;
    let minimum = context.minimum;
    let dma = &mut *context.dma;
    let attached = nvme::attach(&mut function, handoff, dma, minimum.as_ref()).map(|attached| {
        devices.device_failures.extend(
            attached.failures.into_iter().map(|(namespace, error)| {
                (alloc::format!("{address} namespace {namespace}"), error)
            }),
        );
        attached
            .namespaces
            .into_iter()
            .map(|device| Box::new(device) as Box<dyn BlockDevice>)
            .collect()
    });
    if attached.is_err() {
        quiesce_if_expired(context.budget, &mut function);
    }
    disks(devices, address, attached);
}

pub(super) unsafe fn xhci(mut function: Function, context: &mut Context<'_>) {
    let address = function.address;
    let devices = &mut *context.devices;
    let handoff = context.handoff;
    let minimum = context.minimum;
    let dma = &mut *context.dma;
    let firmware = context.firmware;
    match xhci::attach_with_provider(&mut function, handoff, dma, firmware, minimum.clone()) {
        Ok(found) => devices.absorb_usb(Some(address), found, firmware),
        Err(error) => {
            quiesce_if_expired(context.budget, &mut function);
            devices.failures.push((address, error.into()));
        }
    }
}

pub(super) unsafe fn virtio_net(mut function: Function, context: &mut Context<'_>) {
    let address = function.address;
    let devices = &mut *context.devices;
    let handoff = context.handoff;
    let minimum = context.minimum;
    let dma = &mut *context.dma;
    let attached = VirtioNet::new(&mut function, handoff, dma, minimum.as_ref());
    if attached.is_err() {
        quiesce_if_expired(context.budget, &mut function);
    }
    link(devices, address, attached, Adapter::Virtio);
}

pub(super) unsafe fn igc(mut function: Function, context: &mut Context<'_>) {
    let address = function.address;
    let devices = &mut *context.devices;
    let handoff = context.handoff;
    let minimum = context.minimum;
    let dma = &mut *context.dma;
    // SAFETY: the caller grants exclusive post-EBS controller,
    // segment configuration and firmware/ASPM policy ownership.
    let attached = unsafe { Igc::new(&mut function, handoff, dma, minimum.clone()) };
    if attached.is_err() {
        quiesce_if_expired(context.budget, &mut function);
    }
    link(devices, address, attached, Adapter::Igc);
}

pub(super) unsafe fn e1000e(mut function: Function, context: &mut Context<'_>) {
    let address = function.address;
    let devices = &mut *context.devices;
    let handoff = context.handoff;
    let minimum = context.minimum;
    let dma = &mut *context.dma;
    // SAFETY: the caller granted exclusive ownership of all mapped PCI
    // functions and controllers after ExitBootServices.
    let attached = unsafe { E1000e::new(&mut function, handoff, dma, minimum.clone()) };
    if attached.is_err() {
        quiesce_if_expired(context.budget, &mut function);
    }
    link(devices, address, attached, Adapter::E1000e);
}

pub(super) unsafe fn pch(function: Function, context: &mut Context<'_>) {
    let address = function.address;
    let devices = &mut *context.devices;
    let handoff = context.handoff;
    let minimum = context.minimum;
    let dma = &mut *context.dma;
    let firmware = context.firmware;
    // The MTP/PTP cohorts read Linux's K1 board quirk from the
    // SMBIOS system fields; without them those cohorts refuse.
    let snapshot = crate::firmware::smbios::load(firmware).ok();
    let host = snapshot
        .as_ref()
        .map(|snapshot| pch::host_identity(&snapshot.identity))
        .unwrap_or_default();
    // SAFETY: as for e1000e; the adapter keeps the function, its
    // configuration and both register windows to the end of boot.
    let attached = unsafe { pch::Pch::attach(function, handoff, dma, minimum.clone(), host) };
    link(devices, address, attached.map(Box::new), Adapter::Pch);
}

pub(super) unsafe fn igb(mut function: Function, context: &mut Context<'_>) {
    let address = function.address;
    let devices = &mut *context.devices;
    let handoff = context.handoff;
    let minimum = context.minimum;
    let dma = &mut *context.dma;
    // SAFETY: as for e1000e, the caller granted exclusive ownership
    // of the mapped PCI functions and their controllers.
    let attached = unsafe { Igb::new(&mut function, handoff, dma, minimum.clone()) };
    if let Ok(device) = &attached {
        if device.warnings.nvm_autoload_timeout {
            devices.warnings.push((
                address,
                Warning::Message("igb NVM autoload completion timed out; NVM contents validated"),
            ));
        }
        if device.warnings.phy_configuration_timeout {
            devices.warnings.push((
                address,
                Warning::Message(
                    "igb PHY configuration completion timed out; PHY transactions validated",
                ),
            ));
        }
    }
    if attached.is_err() {
        quiesce_if_expired(context.budget, &mut function);
    }
    link(devices, address, attached, Adapter::Igb);
}

pub(super) unsafe fn alx(mut function: Function, context: &mut Context<'_>) {
    let address = function.address;
    let devices = &mut *context.devices;
    let handoff = context.handoff;
    let minimum = context.minimum;
    let dma = &mut *context.dma;
    // SAFETY: as for e1000e, the caller granted exclusive ownership
    // of the mapped PCI functions and their controllers.
    let attached = unsafe { Alx::new(&mut function, handoff, dma, minimum.clone()) };
    if attached.is_err() {
        quiesce_if_expired(context.budget, &mut function);
    }
    link(devices, address, attached, Adapter::Alx);
}

pub(super) unsafe fn atl1c(mut function: Function, context: &mut Context<'_>) {
    let address = function.address;
    let devices = &mut *context.devices;
    let handoff = context.handoff;
    let minimum = context.minimum;
    let dma = &mut *context.dma;
    // SAFETY: as for e1000e, the caller granted exclusive ownership
    // of the mapped PCI functions and their controllers.
    let attached = unsafe { Atl1c::new(&mut function, handoff, dma, minimum.clone()) };
    if attached.is_err() {
        quiesce_if_expired(context.budget, &mut function);
    }
    link(devices, address, attached, Adapter::Atl1c);
}

pub(super) unsafe fn realtek(mut function: Function, context: &mut Context<'_>) {
    let address = function.address;
    let devices = &mut *context.devices;
    let handoff = context.handoff;
    let minimum = context.minimum;
    let dma = &mut *context.dma;
    let firmware = context.firmware;
    // SAFETY: the caller granted exclusive post-EBS ownership of
    // mapped PCI/MMIO and DMA, including upstream ASPM configuration.
    let attached = unsafe { Realtek::new(&mut function, handoff, dma, firmware, minimum.clone()) };
    if let Ok(Some(warning)) = attached.as_ref().map(Realtek::warning) {
        devices.warnings.push((address, Warning::Realtek(warning)));
    }
    if attached.is_err() {
        quiesce_if_expired(context.budget, &mut function);
    }
    link(devices, address, attached, Adapter::Realtek);
}

pub(super) unsafe fn virtio_input(mut function: Function, context: &mut Context<'_>) {
    let address = function.address;
    let devices = &mut *context.devices;
    let handoff = context.handoff;
    let minimum = context.minimum;
    let dma = &mut *context.dma;
    match VirtioInput::new(&mut function, handoff, dma, minimum.as_ref()) {
        Ok(device) => devices.inputs.push(InputDevice::Virtio(Box::new(device))),
        Err(error) => devices.failures.push((address, error.into())),
    }
}

pub(super) unsafe fn rtsx(function: Function, context: &mut Context<'_>) {
    let address = function.address;
    let devices = &mut *context.devices;
    let handoff = context.handoff;
    let minimum = context.minimum;
    let attached = match rtsx::attach(function, handoff, minimum.clone()) {
        Ok(attached) => attached,
        Err(failure) => {
            devices
                .failures
                .push((address, Error::Device(failure.error.clone())));
            devices.owners.push(Box::new(failure));
            return;
        }
    };
    devices.rtsx_hosts.push(attached.host);
    if let Some(error) = attached.failure {
        devices
            .device_failures
            .push((alloc::format!("{address} RTSX SD slot"), error));
    }
    devices.disks.extend(
        attached
            .card
            .into_iter()
            .map(|card| Box::new(card) as Box<dyn BlockDevice>),
    );
}

pub(super) unsafe fn sdhci(function: Function, context: &mut Context<'_>) {
    let address = function.address;
    let devices = &mut *context.devices;
    let handoff = context.handoff;
    let minimum = context.minimum;
    let dma = &mut *context.dma;
    let attached = match sdhci::attach(function, handoff, dma, minimum.clone()) {
        Ok(attached) => attached,
        Err(failure) => {
            devices
                .failures
                .push((address, Error::Device(failure.error.clone())));
            devices.owners.push(Box::new(failure));
            return;
        }
    };
    for host in attached.failed_hosts {
        devices.owners.push(Box::new(host));
    }
    for slot in attached.failed_slots {
        devices.owners.push(Box::new(slot));
    }
    devices.sd_hosts.extend(attached.hosts);
    devices.device_failures.extend(
        attached
            .failures
            .into_iter()
            .map(|(slot, error)| (alloc::format!("{address} SD slot {slot}"), error)),
    );
    devices.disks.extend(
        attached
            .cards
            .into_iter()
            .map(|card| Box::new(card) as Box<dyn BlockDevice>),
    );
}

pub(super) unsafe fn ahci(function: Function, context: &mut Context<'_>) {
    let address = function.address;
    let devices = &mut *context.devices;
    let handoff = context.handoff;
    let minimum = context.minimum;
    let dma = &mut *context.dma;
    let attached = match ahci::attach(function, handoff, dma, minimum.clone()) {
        Ok(attached) => attached,
        Err(failure) => {
            devices
                .failures
                .push((address, Error::Device(failure.error.clone())));
            devices.owners.push(Box::new(failure));
            return;
        }
    };
    if attached.remapped_nvme != 0 {
        devices
            .warnings
            .push((address, Warning::Message(ahci::REMAPPED_NVME)));
    }
    devices.ahci_hosts.push(attached.host);
    devices.device_failures.extend(
        attached
            .failures
            .into_iter()
            .map(|(port, error)| (alloc::format!("{address} port {port}"), error)),
    );
    devices.disks.extend(
        attached
            .disks
            .into_iter()
            .map(|device| Box::new(device) as Box<dyn BlockDevice>),
    );
}
