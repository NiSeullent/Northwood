//! The host's Bluetooth controller behind the guest's USB one
//! (`stela_soc::usb::hci`): HCI commands and ACL data the guest queued go to
//! the first controller probe left healthy, and its events and ACL data come
//! back while the guest's queues have room. The rest wait in the host
//! controller, as they would for a host that reads slowly.
//!
//! A host with no healthy controller gets the controller model
//! (`stela_soc::usb::hci::model`) behind the same function: a controller
//! with no radio, reported once as `BLUETOOTH PLATFORM Modeled`. bluetoothd
//! then brings it up and idles instead of restarting for a missing chip.
//!
//! One controller serves the guest; others are reported and left idle. A
//! controller that fails is reported once, and the guest's commands then get
//! no answer, as from a controller that stopped responding.
//!
//! The controllers' registers and DMA pages are reached through the loader's
//! identity root, so each exchange runs in it (`storage::RestoreRoot`).
use super::{output, storage::RestoreRoot};
use alloc::{boxed::Box, collections::VecDeque, string::String, vec::Vec};
use stela_drivers::{
    bluetooth::{bcm4377, intel_pcie},
    probe::native::{Bcm4377, IntelBluetooth, PcieController, PcieHci},
    usb::bluetooth::{Packet, UsbHci},
};
use stela_soc::machine::{AppleDevices, Outbound, UsbBluetoothCounters};

enum Controller {
    Usb(Box<UsbHci>),
    Intel(Box<IntelBluetooth>),
    Broadcom(Box<Bcm4377>),
    /// The guest's function answers itself; `installed` once it does.
    Modeled {
        installed: bool,
    },
}

/// Whether a controller took a packet, or holds it back for now.
enum Sent {
    Taken,
    Busy,
}

pub struct HostBluetooth {
    controller: Controller,
    location: String,
    root: u64,
    /// What the guest sent that the controller has no room for yet.
    waiting: VecDeque<Outbound>,
    up: Option<bool>,
    failed: bool,
    /// Commands the model answered Unknown HCI Command, to report.
    unknown: Vec<u16>,
}

/// The controller is dropped in the loader's root. An install runs without
/// host input and drops the controller it found right after the device
/// start; a USB controller's drop closes its endpoints through the xHCI,
/// whose registers only that root maps.
impl Drop for HostBluetooth {
    fn drop(&mut self) {
        let controller = core::mem::replace(
            &mut self.controller,
            Controller::Modeled { installed: false },
        );
        super::network::in_loader_root(self.root, || drop(controller));
    }
}

impl HostBluetooth {
    /// The first controller `usb` and `pcie` hold that initialized, or the
    /// model without one.
    pub fn first(usb: Vec<UsbHci>, pcie: Vec<PcieHci>, root: u64) -> Option<Self> {
        let usb = usb.into_iter().filter_map(|hci| {
            let healthy = matches!(hci.initialization(), Some(Ok(_)) | None);
            let location = hci.binding().device().location.clone();
            healthy.then_some((Controller::Usb(Box::new(hci)), location))
        });
        let pcie = pcie.into_iter().filter_map(|hci| match hci.controller {
            PcieController::Intel {
                transport,
                initialization: Ok(_),
            } => Some((Controller::Intel(Box::new(transport)), hci.location)),
            PcieController::Broadcom {
                transport,
                initialization: Ok(_),
            } => Some((Controller::Broadcom(Box::new(transport)), hci.location)),
            _ => None,
        });
        let mut healthy = usb.chain(pcie);
        let Some((controller, location)) = healthy.next() else {
            output::arguments(format_args!(
                "BLUETOOTH PLATFORM Modeled: no host controller; the guest's controller \
                 has no radio and finds nothing in range\n"
            ));
            return Some(Self {
                controller: Controller::Modeled { installed: false },
                location: String::from("model"),
                root,
                waiting: VecDeque::new(),
                up: None,
                failed: false,
                unknown: Vec::new(),
            });
        };
        output::arguments(format_args!("BLUETOOTH guest uses {location}\n"));
        for (_, unused) in healthy {
            output::arguments(format_args!("BLUETOOTH unused {unused}\n"));
        }
        Some(Self {
            controller,
            location,
            root,
            waiting: VecDeque::new(),
            up: None,
            failed: false,
            unknown: Vec::new(),
        })
    }

    /// What the guest queued, whether it can take more, and its status,
    /// taken from its devices for the exchange to run without them. The
    /// model is installed at the first exchange.
    pub fn queued(
        &mut self,
        devices: &mut AppleDevices,
    ) -> (Vec<Outbound>, bool, Option<(bool, UsbBluetoothCounters)>) {
        if let Controller::Modeled { installed } = &mut self.controller {
            if !*installed && let Some(address) = devices.bluetooth_model() {
                *installed = true;
                let [a, b, c, d, e, f] = address;
                output::arguments(format_args!(
                    "BLUETOOTH model BD_ADDR {f:02x}:{e:02x}:{d:02x}:{c:02x}:{b:02x}:{a:02x}\n"
                ));
            }
            self.unknown.extend(devices.bluetooth_unknown());
        }
        let packets = core::iter::from_fn(|| devices.bluetooth_take()).collect();
        (
            packets,
            devices.bluetooth_room(),
            devices.bluetooth_status(),
        )
    }

    fn refuse(&mut self, what: &str, error: impl core::fmt::Display) {
        output::arguments(format_args!(
            "BLUETOOTH FAILED {} {what}: {error}\n",
            self.location
        ));
        self.failed = true;
        self.waiting.clear();
    }

    /// Reports the guest's controller going up or down.
    pub fn status(&mut self, status: Option<(bool, UsbBluetoothCounters)>) {
        for opcode in self.unknown.drain(..) {
            output::arguments(format_args!(
                "BLUETOOTH model: unknown command {opcode:#06x} (OGF {:#04x} OCF {:#05x}) \
                 answered Unknown HCI Command\n",
                opcode >> 10,
                opcode & 0x3ff
            ));
        }
        let (up, counters) = status.unwrap_or_default();
        if self.up != Some(up) {
            self.up = Some(up);
            output::arguments(format_args!(
                "GUEST BLUETOOTH {} {counters:?}\n",
                if up { "up" } else { "down" }
            ));
        }
    }

    pub fn transmit(&mut self, packets: Vec<Outbound>) {
        // The model answers inside the guest's function.
        if self.failed || matches!(self.controller, Controller::Modeled { .. }) {
            return;
        }
        self.waiting.extend(packets);
        // SAFETY: the loader root `storage::open` captured, which maps the
        // controllers' registers and DMA pages for the whole run.
        let _root = unsafe { RestoreRoot::enter(self.root) };
        while let Some(packet) = self.waiting.front() {
            let sent = match (&mut self.controller, packet) {
                (Controller::Usb(hci), Outbound::Command(bytes)) => hci
                    .submit_command(bytes)
                    .map(|()| Sent::Taken)
                    .map_err(|error| alloc::format!("{error}")),
                (Controller::Usb(hci), Outbound::Acl(bytes)) => hci
                    .submit_acl(bytes)
                    .map(|()| Sent::Taken)
                    .map_err(|error| alloc::format!("{error}")),
                (Controller::Intel(transport), packet) => {
                    let (kind, bytes) = match packet {
                        Outbound::Command(bytes) => (intel_pcie::Kind::Command, bytes),
                        Outbound::Acl(bytes) => (intel_pcie::Kind::Acl, bytes),
                    };
                    if transport.ready() {
                        transport
                            .send(kind, bytes)
                            .map(|_| Sent::Taken)
                            .map_err(|error| alloc::format!("{error:?}"))
                    } else {
                        Ok(Sent::Busy)
                    }
                }
                (Controller::Modeled { .. }, _) => Ok(Sent::Taken),
                (Controller::Broadcom(transport), packet) => {
                    let (kind, bytes) = match packet {
                        Outbound::Command(bytes) => (bcm4377::Kind::Command, bytes),
                        Outbound::Acl(bytes) => (bcm4377::Kind::Acl, bytes),
                    };
                    match transport.send(kind, bytes) {
                        Ok(_) => Ok(Sent::Taken),
                        Err(bcm4377::Error::QueueFull(_)) => Ok(Sent::Busy),
                        Err(error) => Err(alloc::format!("{error:?}")),
                    }
                }
            };
            match sent {
                Ok(Sent::Taken) => {
                    self.waiting.pop_front();
                }
                Ok(Sent::Busy) => return,
                Err(error) => return self.refuse("sending", error),
            }
        }
    }

    /// What the controller has for the guest.
    pub fn receive(&mut self) -> Vec<Packet> {
        if self.failed || matches!(self.controller, Controller::Modeled { .. }) {
            return Vec::new();
        }
        // SAFETY: as in `transmit`.
        let _root = unsafe { RestoreRoot::enter(self.root) };
        let received = match &mut self.controller {
            Controller::Modeled { .. } => Ok(Vec::new()),
            Controller::Usb(hci) => hci.poll().map_err(|error| alloc::format!("{error}")),
            Controller::Intel(transport) => transport
                .poll()
                .map(|packets| {
                    let useful = packets.into_iter().filter(|packet| !packet.diagnostic);
                    useful
                        .filter_map(|packet| match packet.kind {
                            intel_pcie::Kind::Event => Some(Packet::Event(packet.bytes)),
                            intel_pcie::Kind::Acl => Some(Packet::Acl(packet.bytes)),
                            _ => None,
                        })
                        .collect()
                })
                .map_err(|error| alloc::format!("{error:?}")),
            Controller::Broadcom(transport) => transport
                .poll()
                .map(|observations| {
                    let packets = observations.packets.into_iter();
                    packets
                        .filter_map(|packet| match packet.kind {
                            bcm4377::Kind::Event => Some(Packet::Event(packet.bytes)),
                            bcm4377::Kind::Acl => Some(Packet::Acl(packet.bytes)),
                            _ => None,
                        })
                        .collect()
                })
                .map_err(|error| alloc::format!("{error:?}")),
        };
        received.unwrap_or_else(|error| {
            self.refuse("receiving", error);
            Vec::new()
        })
    }

    /// Hands `packets` to the guest's controller. A vendor diagnostic frame
    /// is not ACL data and stays here.
    pub fn deliver(devices: &mut AppleDevices, packets: Vec<Packet>) {
        for packet in packets {
            match packet {
                Packet::Event(bytes) => devices.bluetooth_event(&bytes),
                Packet::Acl(bytes) => devices.bluetooth_acl(&bytes),
                Packet::Diagnostic(_) => {}
            }
        }
    }
}
