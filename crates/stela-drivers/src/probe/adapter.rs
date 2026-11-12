// SPDX-License-Identifier: GPL-2.0-or-later
//! Static dispatch for the host's network links.
use super::native;
use crate::{
    net::{
        self, Link, LinkError, Mac,
        alx::Alx,
        atl1c::Atl1c,
        e1000e::{E1000e, pch::cold::attach as pch},
        igb::Igb,
        igc::Igc,
        realtek::Realtek,
        virtio::VirtioNet,
    },
    usb::{cdc::UsbNet, ethernet::UsbEthernet},
};
use alloc::{boxed::Box, vec::Vec};

macro_rules! driver_name {
    (device, $device:ident) => {
        $device.driver()
    };
    ($name:literal, $device:ident) => {{
        let _ = $device;
        $name
    }};
}

macro_rules! next_poll {
    (scheduled, $device:ident) => {
        $device.due_us()
    };
    (unscheduled, $device:ident) => {{
        let _ = $device;
        None
    }};
}

macro_rules! adapters {
    ($($(#[cfg($feature:meta)])? $variant:ident($ty:ty), $name:tt, $poll:ident;)*) => {
        /// A host network adapter, as a raw Ethernet link.
        pub enum Adapter {
            $($(#[cfg($feature)])? $variant($ty),)*
        }

        impl Adapter {
            /// Stable implementation name for diagnostics and driver verification.
            pub fn driver(&self) -> &'static str {
                match self {
                    $($(#[cfg($feature)])? Self::$variant(device) => driver_name!($name, device),)*
                }
            }

            /// Where the adapter is, for logs.
            pub fn location(&self) -> &str {
                match self {
                    $($(#[cfg($feature)])? Self::$variant(device) => &device.location,)*
                }
            }
        }

        impl Link for Adapter {
            fn mac(&self) -> Mac {
                match self {
                    $($(#[cfg($feature)])? Self::$variant(device) => device.mac(),)*
                }
            }

            fn carrier(&mut self) -> bool {
                match self {
                    $($(#[cfg($feature)])? Self::$variant(device) => device.carrier(),)*
                }
            }

            fn signals(&self) -> bool {
                match self {
                    $($(#[cfg($feature)])? Self::$variant(device) => device.signals(),)*
                }
            }

            fn due_us(&self) -> Option<u64> {
                match self {
                    $($(#[cfg($feature)])? Self::$variant(device) => next_poll!($poll, device),)*
                }
            }

            fn transmit(&mut self, frame: &[u8]) -> Result<(), LinkError> {
                match self {
                    $($(#[cfg($feature)])? Self::$variant(device) => device.transmit(frame),)*
                }
            }

            fn receive(&mut self) -> Result<Option<Vec<u8>>, LinkError> {
                match self {
                    $($(#[cfg($feature)])? Self::$variant(device) => device.receive(),)*
                }
            }

            fn receive_reusing(&mut self, frame: &mut Vec<u8>) -> Result<bool, LinkError> {
                match self {
                    $($(#[cfg($feature)])? Self::$variant(device) => device.receive_reusing(frame),)*
                }
            }
        }
    };
}

adapters! {
    Virtio(VirtioNet), "virtio-net", unscheduled;
    Igc(Igc), "igc", unscheduled;
    E1000e(E1000e), "e1000e", unscheduled;
    Pch(Box<pch::Pch>), "e1000e", unscheduled;
    Igb(Igb), "igb", unscheduled;
    Alx(Alx), "alx", unscheduled;
    Atl1c(Atl1c), "atl1c", unscheduled;
    Realtek(Realtek), "realtek", unscheduled;
    Usb(UsbNet), "usb-cdc", unscheduled;
    UsbVendor(UsbEthernet), device, unscheduled;
    Tg3(Box<native::Located<native::Tg3>>), "tg3", unscheduled;
    Aqtion(Box<native::Located<net::aqtion::native::Attached>>), "atlantic", unscheduled;
    #[cfg(feature = "native-acpi")]
    IntelWifi(Box<native::Located<native::IntelWifi>>), "iwlwifi", scheduled;
    MediatekWifi(Box<native::Located<native::MediatekWifi>>), "mt7925e", scheduled;
    RealtekWifi(Box<native::Located<native::RealtekWifi>>), "rtw88_8822ce", scheduled;
}
