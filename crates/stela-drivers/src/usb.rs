//! USB devices behind the host's xHCI controllers ([`xhci`]), each
//! interface bound to its class driver when the controller attaches, and
//! Bulk-Only storage on EHCI controllers' high-speed roots ([`ehci`]): HID
//! keyboards and pointers ([`hid`]), Bulk-Only and UAS mass storage
//! ([`storage`]), CDC ECM/NCM and vendor Ethernet adapters ([`cdc`],
//! [`ethernet`]), Bluetooth HCI transports ([`bluetooth`]) and UAC2 audio
//! ([`audio`]). Every class driver reads its device through [`descriptor`],
//! the controller-neutral standard requests and configuration model.
//! [`typec`] holds the UCSI and Type-C mux library; probe
//! attaches ACPI UCSI managers from it.
pub mod audio;
pub mod bluetooth;
pub mod cdc;
pub mod descriptor;
pub mod ehci;
pub mod ethernet;
pub mod forwarding;
pub mod handoff;
pub mod hid;
pub mod rndis;
pub mod storage;
pub mod typec;
pub mod xhci;
