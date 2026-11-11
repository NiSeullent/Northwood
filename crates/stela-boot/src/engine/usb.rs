//! The host's xHCI controllers after probe: each controller's topology
//! owner ([`xhci::Host`]), looked at between engine slices so hotplug and
//! the resets class drivers queue (`Controller::queue_reset`) are serviced,
//! as Linux's `hub_event` services them. A controller that signals its
//! events wakes the engine and is polled once it has work
//! ([`xhci::Host::due`]); another is polled on a timer. A keyboard or
//! pointer that arrives joins the host input. Network adapters can be handed
//! to the guest network service. Other class owners are kept until their
//! devices are removed. The USB storage interfaces probe configured stay owned here
//! for the boot, with the disks the guest reaches through them.
//!
//! A USB4 host whose firmware runs the connection manager is looked at on the
//! same timer: each firmware event is reported, as Linux's
//! `icm_handle_notification` receives it. A software connection manager is
//! polled every slice, since its work waits on the host's control channel;
//! neither takes the host's interrupts. What it tunnels after probe works
//! for USB3 and DisplayPort, and a PCIe device stays unused until the next
//! boot's scan, as a USB device no consumer takes does.
//!
//! The controllers' registers and DMA pages are reached through the
//! loader's identity root, so each poll runs in it (`storage::RestoreRoot`).
use super::firmware_requests::Requests;
use super::storage::RestoreRoot;
use crate::output;
use alloc::{string::String, vec::Vec};
use stela_drivers::{
    bus::{
        dma::RegionAllocator,
        thunderbolt::{
            icm,
            manager::{Admission, Platform, Report, Uuid},
        },
    },
    input::InputDevice,
    probe::{Adapter, native::usb4},
    usb::{
        storage::UsbStorage,
        xhci::{self, Found},
    },
};

/// Firmware events serviced per USB4 host per poll.
const USB4_EVENTS: usize = 16;

/// What a USB4 host's software connection manager asks of the mount.
pub(super) struct Usb4Platform {
    location: String,
    /// Whether an IOMMU confines the host and its tunnels' bridges.
    isolated: bool,
    /// The devices `--usb4-approve` lists.
    approved: Vec<Uuid>,
    tsc_per_ms: u64,
}

impl Usb4Platform {
    pub(super) fn new(location: &str, isolated: bool, approved: Vec<Uuid>, tsc_hz: u64) -> Self {
        Self {
            location: String::from(location),
            isolated,
            approved,
            tsc_per_ms: (tsc_hz / 1000).max(1),
        }
    }
}

impl Platform for Usb4Platform {
    fn now_ms(&self) -> u64 {
        super::host::rdtsc() / self.tsc_per_ms
    }

    fn report(&mut self, report: Report) {
        let location = &self.location;
        output::arguments(format_args!("HOST USB4 {location}: {report}\n"));
    }

    fn admit_pcie(&self, uuid: Uuid) -> Admission {
        if self.isolated {
            Admission::Isolated
        } else if self.approved.contains(&uuid) {
            Admission::Listed
        } else {
            Admission::Refused
        }
    }
}

/// Report one firmware event of the host at `location`.
pub(super) fn report_usb4(location: &str, family: icm::Family, event: &icm::Event) {
    match event {
        icm::Event::Notification(message) => match icm::Notification::decode(family, message) {
            Ok(notification) => {
                output::arguments(format_args!("HOST USB4 {location}: {notification}\n"))
            }
            Err(error) => output::arguments(format_args!(
                "HOST USB4 {location}: undecodable firmware event {:#x}: {error:?}\n",
                message.code
            )),
        },
        icm::Event::Discarded(error) => output::arguments(format_args!(
            "HOST USB4 {location}: firmware message discarded: {error:?}\n"
        )),
        icm::Event::DriverReady { request, .. } => output::arguments(format_args!(
            "HOST USB4 {location}: unrequested DRIVER_READY answer {request}\n"
        )),
    }
}

/// How often controllers without a signal are polled: Linux's root hub
/// status timer runs every 250 ms; a hub's status endpoint interval is at
/// most 255 ms.
const POLL_HZ: u64 = 16;

/// The owners a poll found that no consumer takes, with the locations of
/// the devices they came with that are still attached.
struct Kept {
    attached: Vec<String>,
    _found: Found,
}

#[derive(Default)]
pub(super) struct Arrivals {
    pub(super) inputs: Vec<InputDevice>,
    pub(super) links: Vec<Adapter>,
    pub(super) removed: Vec<xhci::SlotId>,
}

/// A USB4 host and what the polls last saw of it.
struct Usb4 {
    host: usb4::Host,
    /// Whether its connection manager still runs.
    running: bool,
    /// A software connection manager's PCIe tunnels.
    pcie_tunnels: usize,
}

pub struct HostUsb {
    hosts: Vec<xhci::Host>,
    usb4: Vec<Usb4>,
    tsc_per_ms: u64,
    /// The probe's DMA pages, which enumeration of a new device draws on.
    dma: RegionAllocator,
    /// The boot's firmware, through the probe's request record.
    firmware: &'static Requests,
    root: u64,
    /// New devices' class owners that no consumer takes, kept while one of
    /// their devices is attached.
    kept: Vec<Kept>,
    /// The storage interfaces probe configured, whose disks the mount
    /// opened: their class owners outlive `open`.
    _storage: Vec<UsbStorage>,
    interval: u64,
    last: u64,
}

impl HostUsb {
    pub fn new(
        hosts: Vec<xhci::Host>,
        usb4: Vec<usb4::Host>,
        storage: Vec<UsbStorage>,
        dma: RegionAllocator,
        firmware: &'static Requests,
        root: u64,
        tsc_hz: u64,
    ) -> Self {
        Self {
            hosts,
            usb4: usb4
                .into_iter()
                .map(|host| Usb4 {
                    pcie_tunnels: match host.connection() {
                        usb4::Connection::Software { manager } => manager.pcie_tunnels(),
                        usb4::Connection::Firmware { .. } => 0,
                    },
                    host,
                    running: true,
                })
                .collect(),
            tsc_per_ms: (tsc_hz / 1000).max(1),
            dma,
            firmware,
            root,
            kept: Vec::new(),
            _storage: storage,
            interval: tsc_hz / POLL_HZ,
            last: super::host::rdtsc(),
        }
    }

    /// Whether a host driver found firmware the USB lacks.
    pub fn firmware_missing(&self) -> bool {
        self.firmware.missing()
    }

    /// Poll controllers with work or a due attachment timer, and check
    /// non-signalling controllers on the fallback interval. Hand off network
    /// owners and their exact removal IDs when the guest reserved a port.
    pub(super) fn poll(
        &mut self,
        guest_network: bool,
        #[cfg(feature = "diagnostics")] wait: &mut u64,
    ) -> Arrivals {
        let mut arrived = Arrivals::default();
        let now = super::host::rdtsc();
        let timed = now.wrapping_sub(self.last) >= self.interval;
        if timed {
            self.last = now;
        }
        self.poll_usb4(
            now,
            timed,
            #[cfg(feature = "diagnostics")]
            wait,
        );
        let firmware = self.firmware;
        for host in &mut self.hosts {
            let mut changes = {
                // The software attachment timer is owned by this HostUsb.
                // Before it or the regular timer is due, a busy driver can
                // leave topology work queued for another service turn.
                let root = if timed || host.attachment_due() {
                    // SAFETY: the loader root captured by `storage::open`
                    // keeps the engine window and maps the controller and DMA pages.
                    // This service runs with the processor's pinned record active.
                    let root = unsafe {
                        RestoreRoot::enter_input(
                            self.root,
                            #[cfg(feature = "diagnostics")]
                            wait,
                        )
                    };
                    Some(root)
                } else {
                    // SAFETY: the same root and active pinned record apply here.
                    // A busy owner leaves the driver state untouched.
                    unsafe { RestoreRoot::try_enter(self.root) }
                };
                let Some(_root) = root else {
                    continue;
                };
                if !(if host.signals() {
                    host.due()
                } else {
                    timed || host.attachment_due()
                }) {
                    continue;
                }
                let changes = host.poll(&mut self.dma, firmware);
                // A new device's driver may have found its firmware missing;
                // the record is reached under the same lock.
                firmware.report();
                changes
            };
            for device in &changes.removed {
                output::arguments(format_args!("USB removed {}\n", device.location));
                if guest_network {
                    arrived.removed.push(device.id);
                }
                for kept in &mut self.kept {
                    kept.attached
                        .retain(|location| *location != device.location);
                }
            }
            if self.kept.iter().any(|kept| kept.attached.is_empty()) {
                // Dropping class owners can stop their transfers.
                // SAFETY: as for the poll above.
                let _root = unsafe { RestoreRoot::enter(self.root) };
                self.kept.retain(|kept| !kept.attached.is_empty());
            }
            let added = &mut changes.added;
            for (location, error) in &added.failures {
                output::arguments(format_args!(
                    "USB FAILED {location}: {}\n",
                    xhci::explain(error)
                ));
            }
            for (location, why) in &added.unused {
                output::arguments(format_args!("USB UNUSED {location}: {why}\n"));
            }
            arrived.inputs.extend(
                added
                    .inputs
                    .drain(..)
                    .map(|input| InputDevice::Usb(alloc::boxed::Box::new(input))),
            );
            for device in &added.devices {
                output::arguments(format_args!("USB added {}\n", device.location));
            }
            if guest_network {
                for (location, error) in &added.vendor_failures {
                    output::arguments(format_args!("USB VENDOR FAILED {location}: {error}\n"));
                }
                arrived
                    .links
                    .extend(added.links.drain(..).map(Adapter::Usb));
                arrived
                    .links
                    .extend(added.vendor_links.drain(..).map(Adapter::UsbVendor));
            }
            if !added.disks.is_empty() || !added.storage.is_empty() {
                output::arguments(format_args!(
                    "USB a disk added after boot is kept unused until the next boot\n"
                ));
            }
            if !added.links.is_empty() || !added.vendor_links.is_empty() {
                output::arguments(format_args!(
                    "USB a network adapter added after boot is kept unused until the next boot\n"
                ));
            }
            if !changes.added.devices.is_empty() {
                self.kept.push(Kept {
                    attached: changes
                        .added
                        .devices
                        .iter()
                        .map(|device| device.location.clone())
                        .collect(),
                    _found: changes.added,
                });
            }
        }
        arrived
    }

    /// Run each running USB4 host's software connection manager for up to a
    /// millisecond, and, when `timed`, service each firmware's events. A host whose
    /// transport faults is reported once and left stopped.
    fn poll_usb4(&mut self, now: u64, timed: bool, #[cfg(feature = "diagnostics")] wait: &mut u64) {
        let now_ms = now / self.tsc_per_ms;
        for usb4 in &mut self.usb4 {
            if !usb4.running {
                continue;
            }
            let (location, protocol) = usb4.host.protocol();
            match protocol {
                usb4::Connection::Firmware { .. } if !timed => {}
                usb4::Connection::Firmware { icm, .. } => {
                    for _ in 0..USB4_EVENTS {
                        let event = {
                            // SAFETY: as for the xHCI controllers below.
                            let _root = unsafe {
                                RestoreRoot::enter_input(
                                    self.root,
                                    #[cfg(feature = "diagnostics")]
                                    wait,
                                )
                            };
                            icm.poll(now_ms)
                        };
                        match event {
                            Ok(Some(event)) => report_usb4(location, icm.family(), &event),
                            Ok(None) => break,
                            Err(error) => {
                                output::arguments(format_args!(
                                    "HOST USB4 {location}: connection manager transport \
                                     stopped: {error:?}\n",
                                ));
                                usb4.running = false;
                                break;
                            }
                        }
                    }
                }
                usb4::Connection::Software { manager } => {
                    {
                        // SAFETY: as for the xHCI controllers below.
                        let _root = unsafe {
                            RestoreRoot::enter_input(
                                self.root,
                                #[cfg(feature = "diagnostics")]
                                wait,
                            )
                        };
                        loop {
                            usb4.running = manager.poll();
                            if !usb4.running
                                || manager.idle()
                                || super::host::rdtsc().wrapping_sub(now) >= self.tsc_per_ms
                            {
                                break;
                            }
                        }
                    }
                    let tunnels = manager.pcie_tunnels();
                    if tunnels > usb4.pcie_tunnels {
                        output::arguments(format_args!(
                            "HOST USB4 {location}: PCIe device added after the scan: \
                             unused until the next boot\n"
                        ));
                    }
                    usb4.pcie_tunnels = tunnels;
                }
            }
        }
    }
}
