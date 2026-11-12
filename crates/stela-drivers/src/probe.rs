//! The PCI scan that brings up every driver: each function of every mapped
//! ECAM window goes to the driver of its class, and what each driver found
//! or refused is collected in [`Devices`].
use crate::{
    Clock, Handoff, PlatformCaptures,
    block::{BlockDevice, BlockError},
    bus::{
        dma::DmaAllocator,
        pci::{Address, EcamWindow, Function},
    },
    gpu,
    input::{self, InputDevice},
    net::{self, e1000e::pch::cold::attach as pch, realtek},
    storage::{ahci, rtsx, sdhci},
    usb::{self, bluetooth::UsbHci, xhci},
};
use alloc::{boxed::Box, rc::Rc, string::String, sync::Arc, vec::Vec};
use core::fmt;
use stela_driver_base::budget;

mod adapter;
pub use adapter::Adapter;

#[cfg(feature = "native-acpi")]
mod companion;
#[cfg(feature = "native-acpi")]
mod i2c;
#[cfg(feature = "native-acpi")]
pub use i2c::{HidMatch, HidQuirk, HidSetting};
#[cfg(feature = "native-acpi")]
mod power;
#[cfg(feature = "native-acpi")]
pub use power::{EmbeddedControllerInstallation, Power, install_embedded_controller};
mod legacy;
pub mod native;
#[cfg(any(test, feature = "native-acpi"))]
mod platform;
#[cfg(feature = "native-acpi")]
pub use platform::I8042Fadt;
pub mod registry;
pub mod services;
#[cfg(feature = "native-acpi")]
pub mod smbus;
#[cfg(feature = "native-acpi")]
pub mod typec;
use crate::net::wifi::network::Network;
pub use services::{NativeServices, ServiceError};

/// A controller refusal keeps its original typed details for boot diagnostics.
/// In particular, firmware names and integrity/target errors are never replaced
/// by a generic unsupported-device message.
#[derive(Debug)]
pub enum Error {
    Enumeration(crate::bus::pci::EnumerationError),
    Device(BlockError),
    Realtek(realtek::Error),
    Pch(pch::AttachError),
    /// The mount's [`NativeServices`] refused or lack a resource.
    Native(ServiceError),
    Vmd(crate::bus::vmd::Error),
    Tg3(net::tg3::Error),
    Aqtion(net::aqtion::native::Error),
    Sfh(input::sfh::Error),
    SfhCold(input::sfh::cold::native::NativeError),
    Ish(input::ish::Error),
    IshCold(input::ish::cold::native::Error),
    IntelWifi(net::wifi::intel::attach::AttachError),
    MediatekWifi(net::wifi::mediatek::attach::AttachError),
    RealtekWifi(net::wifi::realtek::attach::AttachError),
    IntelBluetooth(crate::bluetooth::intel_pcie::Error),
    Bcm4377(crate::bluetooth::bcm4377::Error),
    Ehci(usb::ehci::Error),
    HdaPci(crate::audio::hda::pci::AttachError),
    Hda(crate::audio::Error),
    HdaNative(native::hda::RetirementError),
    Usb4(native::usb4::Refusal),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device(error) => error.fmt(f),
            Self::Enumeration(error) => write!(f, "PCI enumeration: {error:?}"),
            Self::Realtek(error) => error.fmt(f),
            Self::Pch(error) => error.fmt(f),
            Self::Native(error) => error.fmt(f),
            Self::Vmd(error) => write!(f, "VMD: {error}"),
            Self::Tg3(error) => write!(f, "tg3: {error:?}"),
            Self::Aqtion(error) => write!(f, "atlantic: {error:?}"),
            Self::Sfh(error) => write!(f, "AMD SFH: {error:?}"),
            Self::SfhCold(error) => write!(f, "AMD SFH original-owner preparation: {error:?}"),
            Self::Ish(error) => write!(f, "Intel ISH: {error:?}"),
            Self::IshCold(error) => write!(f, "Intel ISH original-owner preparation: {error:?}"),
            Self::IntelWifi(error) => write!(f, "iwlwifi: {error}"),
            Self::MediatekWifi(error) => write!(f, "mt76: {error}"),
            Self::RealtekWifi(error) => write!(f, "rtw88: {error}"),
            Self::IntelBluetooth(error) => write!(f, "btintel_pcie: {error}"),
            Self::Bcm4377(error) => write!(f, "hci_bcm4377: {error}"),
            Self::Ehci(error) => write!(f, "EHCI: {error:?}"),
            Self::HdaPci(error) => write!(f, "HD Audio PCI: {error:?}"),
            Self::Hda(error) => write!(f, "HD Audio: {error:?}"),
            Self::HdaNative(error) => write!(f, "HD Audio native: {error:?}"),
            Self::Usb4(refusal) => write!(f, "USB4 host: {refusal}"),
        }
    }
}

impl From<BlockError> for Error {
    fn from(error: BlockError) -> Self {
        Self::Device(error)
    }
}

impl From<crate::bus::pci::ConfigError> for Error {
    fn from(error: crate::bus::pci::ConfigError) -> Self {
        Self::Device(BlockError::Config(error))
    }
}
impl From<crate::bus::pci::CapabilityError> for Error {
    fn from(error: crate::bus::pci::CapabilityError) -> Self {
        Self::Device(BlockError::PciCapability(error))
    }
}

impl From<pch::AttachError> for Error {
    fn from(error: pch::AttachError) -> Self {
        Self::Pch(error)
    }
}

impl From<realtek::Error> for Error {
    fn from(error: realtek::Error) -> Self {
        Self::Realtek(error)
    }
}

/// Why a device or controller outside the PCI drivers' attach was not brought
/// up or handed over.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlatformError {
    Config(crate::bus::pci::ConfigError),
    PciCapability(crate::bus::pci::CapabilityError),
    /// A USB host controller could not be taken from firmware.
    Handoff(usb::handoff::Error),
    /// Walking the ACPI namespace or decoding a resource template failed.
    #[cfg(feature = "native-acpi")]
    Acpi(crate::platform::acpi::resources::Error),
    /// The description or the machine state rules attachment out.
    Refused(&'static str),
    Ps2(input::ps2::Error),
    Native(ServiceError),
    I2c(crate::bus::i2c::Error),
    I2cPciMismatch(crate::bus::pci::Identity),
    I2cPciStop(crate::bus::pci::Identity, ServiceError),
    Lpss(crate::bus::lpss::Error),
    LpssAttach(crate::bus::lpss::AttachError),
    HidI2c(input::hid_i2c::Error),
    /// The HID report descriptor could not be interpreted as input.
    HidDescriptor(usb::hid::ParseError),
    Touchpad(input::touchpad::Error),
    #[cfg(feature = "native-acpi")]
    Gpio(crate::bus::gpio::native::Error),
    /// The embedded controller's operation regions could not be installed.
    #[cfg(feature = "native-acpi")]
    Ec(crate::platform::ec::binding::Failure),
    /// The embedded controller's description could not be read.
    #[cfg(feature = "native-acpi")]
    EcDiscovery(crate::platform::ec::discovery::native::DiscoveryError),
    #[cfg(feature = "native-acpi")]
    Battery(crate::platform::battery::Error),
    Smbus(crate::bus::smbus::Error),
    #[cfg(feature = "native-acpi")]
    Ucsi(crate::usb::typec::ucsi::acpi::attachment::Error),
    /// Finding a device or evaluating its method failed.
    #[cfg(feature = "native-acpi")]
    AcpiMethod(crate::platform::acpi::Error),
}

impl fmt::Display for PlatformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Handoff(error) => error.fmt(f),
            Self::Config(error) => error.fmt(f),
            Self::PciCapability(error) => write!(f, "PCI capability: {error:?}"),
            #[cfg(feature = "native-acpi")]
            Self::Acpi(error) => write!(f, "ACPI: {error:?}"),
            Self::Refused(why) => f.write_str(why),
            Self::Ps2(error) => error.fmt(f),
            Self::Native(error) => error.fmt(f),
            Self::I2c(error) => write!(f, "I2C controller: {error:?}"),
            Self::I2cPciMismatch(id) => write!(
                f,
                "I2C ACPI controller {} {:04x}:{:04x} is not a supported LPSS I2C function",
                id.address, id.vendor, id.device,
            ),
            Self::I2cPciStop(id, error) => write!(
                f,
                "LPSS I2C {} {:04x}:{:04x} handoff: {error}",
                id.address, id.vendor, id.device,
            ),
            Self::Lpss(error) => write!(f, "LPSS: {error:?}"),
            Self::LpssAttach(error) => write!(f, "LPSS I2C attach: {error:?}"),
            Self::HidI2c(error) => error.fmt(f),
            Self::HidDescriptor(error) => error.fmt(f),
            Self::Touchpad(error) => error.fmt(f),
            #[cfg(feature = "native-acpi")]
            Self::Gpio(error) => write!(f, "GPIO controller: {error:?}"),
            #[cfg(feature = "native-acpi")]
            Self::Ec(error) => write!(f, "embedded controller: {error:?}"),
            #[cfg(feature = "native-acpi")]
            Self::EcDiscovery(error) => write!(f, "embedded controller description: {error:?}"),
            #[cfg(feature = "native-acpi")]
            Self::Battery(error) => write!(f, "battery: {error:?}"),
            Self::Smbus(error) => error.fmt(f),
            #[cfg(feature = "native-acpi")]
            Self::Ucsi(error) => write!(f, "UCSI: {error:?}"),
            #[cfg(feature = "native-acpi")]
            Self::AcpiMethod(error) => write!(f, "ACPI method: {error:?}"),
        }
    }
}

/// Nonfatal diagnostics keep controller-specific values through discovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Warning {
    Message(&'static str),
    Realtek(realtek::Warning),
    /// An Intel xHCI took its switchable ports from EHCI.
    IntelPortSwitch(usb::handoff::Switched),
}

impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(message) => f.write_str(message),
            Self::Realtek(warning) => write!(f, "Realtek {warning}"),
            Self::IntelPortSwitch(switched) => write!(
                f,
                "Intel xHCI took its ports from EHCI: SuperSpeed {:#x}, USB 2 {:#x}",
                switched.superspeed, switched.usb2
            ),
        }
    }
}

/// Whether a driver here may attach `function`, or probe hands it over from
/// firmware (every USB host controller), including the HD Audio check: its
/// memory BARs must be mapped uncached before ExitBootServices.
pub fn claims(function: &Function) -> bool {
    claims_identity(&function.identity())
}

/// Candidate selection from copied firmware/native header DATA only. This
/// grants neither resource ownership nor permission to size or attach BARs.
pub fn claims_identity(function: &crate::bus::pci::Identity) -> bool {
    registry::pre_exit_claim(function)
}

/// Why the disks behind storage controller `function` are out of reach of
/// Stela's drivers, for the disk chooser to refuse them before boot
/// services end rather than the boot failing after; `None` when a driver
/// here takes the controller, or it is no storage controller.
pub fn disk_refusal(function: &crate::bus::pci::Identity) -> Option<&'static str> {
    if function.class[0] != 0x01
        || claims_identity(function)
        || native::Kind::of_identity(function).is_some()
    {
        return None;
    }
    Some(match function.class[1] {
        0x04 => "Its controller is in RAID mode: set SATA to AHCI in firmware setup",
        _ => "Stela has no driver for its controller",
    })
}

/// What [`probe_with_provider`] will do with `function`, for a boot log
/// line written before boot services end: the default candidate, or why
/// none applies. Later admission also checks the boot's driver policy.
/// `displays` are the slots whose function 0 is a display
/// controller. Options decided after the census (a Wi-Fi network, the
/// mount's native services) are named where they apply.
pub fn plan(
    function: &crate::bus::pci::Identity,
    displays: &[Address],
) -> alloc::borrow::Cow<'static, str> {
    use alloc::borrow::Cow;
    let selection = registry::select_pci(function, &registry::DriverPolicy::default());
    if let Some(why) = selection.refusal() {
        return Cow::Owned(alloc::format!("unsupported: {why}"));
    }
    if let Some(registry::Driver::Native(kind)) = selection.driver() {
        return Cow::Borrowed(kind.driver());
    }
    let driver = match selection {
        registry::Selection::Entry(entry) => match entry.manifest.implementation {
            "pci-display" | "nouveau-nvk" => "GPU",
            "i801" => "i801 SMBus",
            "ehci" => "EHCI USB",
            "xhci" | "xhci-handoff" => "xHCI USB",
            "uhci-handoff" | "ohci-handoff" => {
                "taken from firmware, not driven: UHCI or OHCI USB host controller"
            }
            "nvme" => "NVMe",
            "igc" => "igc Ethernet",
            "e1000e" => "e1000e Ethernet",
            "e1000e-pch" => "Intel PCH Ethernet",
            "igb" => "igb Ethernet",
            "alx" => "alx Ethernet",
            "atl1c" => "atl1c Ethernet",
            "r8169" => "r8169 Ethernet",
            "rtsx_pci" => "Realtek card reader",
            "sdhci" => "SDHCI card reader",
            "ahci" => "AHCI SATA",
            implementation => implementation,
        },
        _ if function.class[..2] == [6, 4] => "PCI bridge",
        _ => {
            if let Some(why) = unclaimed(function.class, function.address, displays) {
                return Cow::Owned(alloc::format!("unsupported: {why}"));
            }
            "not driven"
        }
    };
    Cow::Borrowed(driver)
}

/// An HD Audio function beside a display controller's function 0 is that
/// GPU's HDMI/DP audio, as Linux `get_bound_vga` finds it. No HDMI/DP codec
/// profile here serves one outside the admitted identities.
const DISPLAY_AUDIO: &str = "HDMI/DP audio controller of a GPU outside the supported HD Audio identities (its outputs carry no audio)";

/// Why no driver here takes `class` at `address`, given the slots whose
/// function 0 is a display controller.
fn unclaimed(class: [u8; 3], address: Address, displays: &[Address]) -> Option<&'static str> {
    let beside_display = displays.iter().any(|display| {
        (display.segment, display.bus, display.device)
            == (address.segment, address.bus, address.device)
    });
    if class[..2] == [0x04, 0x03] && address.function != 0 && beside_display {
        return Some(DISPLAY_AUDIO);
    }
    unsupported(class)
}

/// Why no driver here takes a function of a device class the host needs,
/// or none for classes outside this crate's scope.
fn unsupported(class: [u8; 3]) -> Option<&'static str> {
    match class {
        [0x01, 0x01, _] => Some("IDE storage controller"),
        [0x01, 0x04, _] => Some("RAID storage controller (Intel RST/VMD mode)"),
        [0x01, 0x06, _] => Some("SATA controller without the AHCI interface"),
        [0x01, 0x08, _] => Some("non-NVMe non-volatile memory controller"),
        [0x01, ..] => Some("storage controller"),
        [0x02, 0x00, _] => Some("Ethernet controller"),
        [0x02, 0x80, _] => {
            Some("Wi-Fi controller other than Intel's, MediaTek's MT7925 and Realtek's RTL8822C")
        }
        [0x02, ..] => Some("network controller"),
        // Admitted identities attach through the mount's native services.
        [0x04, 0x03, _] => Some("HD Audio controller outside the supported identities"),
        // EHCI attaches through the mount's native services.
        [0x0c, 0x03, 0x20] => None,
        [0x0c, 0x03, _] => {
            Some("UHCI or OHCI USB host controller (taken from firmware, not driven)")
        }
        // Listed NHIs attach through the mount's native services.
        [0x0c, 0x0a, _] => Some("USB4/Thunderbolt host outside the supported NHI identities"),
        [0x0c, 0x05, _] => Some(
            "SMBus host other than an Intel i801 under native ACPI (no Host Notify for touchpad companions)",
        ),
        [0x08, 0x05, _] => Some("SD host controller without standard SDHCI interface"),
        [0x0d, 0x11, _] => Some("PCIe Bluetooth controller other than Intel's"),
        _ => None,
    }
}

/// Every block and input device the drivers brought up, and every
/// controller they could not.
#[derive(Default)]
pub struct Devices {
    /// Native UAC2 playback/capture owners, awaiting explicit runtime PCM use.
    pub audio: Vec<crate::usb::audio::UsbAudio>,
    pub disks: Vec<Box<dyn BlockDevice>>,
    /// Display controllers, inventoried without changing firmware scanout.
    /// These are not initialized accelerators; each records its missing port.
    pub gpus: Vec<gpu::Adapter>,
    /// Host keyboards and pointers: virtio-input devices, USB HID devices
    /// behind xHCI controllers, and the i8042 ports the ACPI namespace
    /// lists, each polled through [`InputDevice::poll`].
    pub inputs: Vec<InputDevice>,
    /// Complete descriptor inventories, including hubs and unbound devices.
    /// Retention does not grant hardware ownership or guest forwarding.
    pub usb_devices: Vec<xhci::Device>,
    /// Explicitly polled USB topology owners, including empty controllers.
    pub usb_hosts: Vec<xhci::Host>,
    /// Attached EHCI controllers, which their roots' storage transports share.
    pub ehci_hosts: Vec<usb::ehci::host::Shared>,
    /// HD Audio controllers and their bound codecs.
    pub hda_hosts: Vec<native::hda::Host>,
    /// USB4/Thunderbolt hosts whose firmware connection manager answered.
    pub usb4_hosts: Vec<native::usb4::Host>,
    /// Native SDHCI socket owners, including empty/removable card sockets.
    pub sd_hosts: Vec<sdhci::Host>,
    /// Native Realtek socket owners, kept for empty/removable SD/MMC cards.
    pub rtsx_hosts: Vec<rtsx::Host>,
    /// Actual AHCI owners retained for lifecycle and failure retirement.
    pub ahci_hosts: Vec<ahci::Host>,
    /// Storage class owners remain available for explicit media rescan.
    pub usb_storage: Vec<crate::usb::storage::UsbStorage>,
    /// USB HCI transports with kept device-specific initialization results.
    /// Historical evidence does not establish radio or guest readiness.
    pub bluetooth: Vec<UsbHci>,
    /// PCIe HCI transports with their kept firmware initialization result.
    pub pcie_bluetooth: Vec<native::PcieHci>,
    /// Raw firmware uploaders awaiting USB removal/fresh enumeration.
    pub bluetooth_bootloaders: Vec<crate::usb::bluetooth::ath3k::UsbBootloader>,
    /// Exact pre-EBS board-variable observation, including unavailable diagnostics.
    pub bluetooth_platform: crate::usb::bluetooth::intel::DsbrCapture,
    /// Network adapters, as raw Ethernet links.
    pub nics: Vec<Adapter>,
    /// The regulatory database, once probe found a Wi-Fi adapter (a PCI
    /// network controller of class 02/80) while a network is configured:
    /// `Err` reports once that it is missing, and no Wi-Fi adapter then
    /// scans or transmits.
    pub regulatory: Option<
        Result<
            crate::net::wifi::regulatory::database::Database,
            crate::net::wifi::regulatory::database::LoadError,
        >,
    >,
    pub failures: Vec<(Address, Error)>,
    /// Devices behind an attached controller that failed on their own, such
    /// as AHCI ports and NVMe namespaces, by location.
    pub device_failures: Vec<(String, BlockError)>,
    /// Nonfatal initialization diagnostics, including adapters not selected.
    pub warnings: Vec<(Address, Warning)>,
    /// USB devices that failed to enumerate or bind, by location.
    pub usb_failures: Vec<(String, BlockError)>,
    /// Vendor initialization failures, preserving exact firmware requests.
    pub usb_vendor_failures: Vec<(String, crate::usb::ethernet::Error)>,
    /// USB interfaces no driver here takes, and why.
    pub usb_unused: Vec<(String, &'static str)>,
    /// How long each xHCI root port took at probe, for the boot log's
    /// `XHCI PROBE <pci> <probe>` lines.
    pub usb_port_probes: Vec<(Address, xhci::PortProbe)>,
    /// Devices and controllers outside the PCI drivers' attach that were not
    /// brought up or handed over, by location, such as
    /// `0000:00:1d.0 EHCI handoff`.
    pub platform_failures: Vec<(String, PlatformError)>,
    /// Objects that must stay alive while the devices they set up exist,
    /// including attaches that failed, such as the Intel VMD hierarchy behind
    /// its NVMe controller and a failed controller's mappings and DMA domain.
    pub owners: Vec<Box<dyn core::any::Any>>,
    /// The batteries, AC adapter and lid the ACPI namespace lists.
    #[cfg(feature = "native-acpi")]
    pub power: Power,
    /// Intel i801 SMBus hosts, shared with the touchpad companions on them.
    #[cfg(feature = "native-acpi")]
    pub smbus_hosts: Vec<smbus::Host>,
    /// Devices outside PCI that the platform describes but no driver here
    /// attaches, by location, and why; reported once, not as failures.
    pub platform_unsupported: Vec<(String, &'static str)>,
    /// ECAM windows that were not mapped, and so not scanned.
    pub unmapped: Vec<EcamWindow>,
    /// Whether the mount supplied [`NativeServices`]. Drivers that attach
    /// through owned mappings, requester DMA domains or the ACPI namespace
    /// need them.
    pub native_services: bool,
    /// Storage, network, audio and USB functions no driver here takes, with
    /// their class and why.
    pub unsupported: Vec<(Address, [u8; 3], &'static str)>,
    /// Functions whose start ran past [`DEVICE_START_US`]: their waits then
    /// ended at once, the driver gave up, and the function was left with
    /// bus mastering and interrupts off where the probe still held it.
    pub abandoned: Vec<(Address, budget::Expiry)>,
}

/// The longest one PCI function's start may take. Linux's own per-step
/// bounds are smaller: xHCI commands 5 s (`XHCI_CMD_DEFAULT_TIMEOUT`), USB
/// control transfers 5 s (`USB_CTRL_GET_TIMEOUT`), AHCI port reset 1 s,
/// the NVMe admin queue 60 s but CC.RDY by CAP.TO, which drives already
/// started by the firmware meet in a few seconds. A device that answers
/// starts far inside this; one that stops answering is given up here even
/// when a driver's per-step bounds would add up to more.
pub const DEVICE_START_US: u64 = 30_000_000;

/// One function's start under its [`DEVICE_START_US`] budget; a passed
/// budget is recorded as the function's abandonment when the start ends.
struct DeviceStart<'a> {
    address: Address,
    budget: Option<budget::Budget>,
    abandoned: &'a mut Vec<(Address, budget::Expiry)>,
}

impl<'a> DeviceStart<'a> {
    fn begin(
        address: Address,
        clock: Clock,
        abandoned: &'a mut Vec<(Address, budget::Expiry)>,
    ) -> Self {
        Self {
            address,
            budget: budget::Budget::start(clock, DEVICE_START_US).ok(),
            abandoned,
        }
    }
}

impl Drop for DeviceStart<'_> {
    fn drop(&mut self) {
        if let Some(expiry) = self.budget.as_ref().and_then(budget::Budget::expired) {
            self.abandoned.push((self.address, expiry));
        }
    }
}

impl Devices {
    /// Take up what an xHCI controller brought, at its start or later
    /// through a port change: disks, inputs, links, audio and Bluetooth,
    /// failures and the owners that keep them.
    pub fn absorb_usb(
        &mut self,
        address: Option<Address>,
        found: xhci::Found,
        firmware: &dyn crate::firmware::Provider,
    ) {
        self.disks.extend(
            found
                .disks
                .into_iter()
                .map(|d| Box::new(d) as Box<dyn BlockDevice>),
        );
        self.inputs.extend(
            found
                .inputs
                .into_iter()
                .map(|input| InputDevice::Usb(Box::new(input))),
        );
        self.usb_devices.extend(found.devices);
        self.usb_hosts.extend(found.hosts);
        self.usb_storage.extend(found.storage);
        self.audio.extend(found.audio);
        self.bluetooth_bootloaders.extend(found.bootloaders);
        for mut device in found.bluetooth {
            // The kept result owns its evidence/errors;
            // no catalog borrow or firmware bytes escape.
            let _ = device
                .initialize_with_provider(firmware, self.bluetooth_platform.platform_inputs());
            self.bluetooth.push(device);
        }
        self.nics.extend(found.links.into_iter().map(Adapter::Usb));
        self.nics
            .extend(found.vendor_links.into_iter().map(Adapter::UsbVendor));
        self.usb_failures.extend(found.failures);
        self.usb_vendor_failures.extend(found.vendor_failures);
        self.usb_unused.extend(found.unused);
        if let Some(address) = address {
            self.usb_port_probes
                .extend(found.probes.into_iter().map(|probe| (address, probe)));
        }
    }

    /// Failures from attempted native vendor initialization, USB and PCIe,
    /// and USB controllers no vendor setup here takes, each named once: no
    /// consumer runs them, so they stay off.
    pub fn bluetooth_failures(&self) -> impl Iterator<Item = (&str, &dyn fmt::Display)> {
        let usb = self.bluetooth.iter().filter_map(|device| {
            let Err(error) = device.initialization()? else {
                return None;
            };
            Some((
                device.binding().device().location.as_str(),
                error as &dyn fmt::Display,
            ))
        });
        let pcie = self.pcie_bluetooth.iter().filter_map(|hci| {
            let error: &dyn fmt::Display = match &hci.controller {
                native::PcieController::Intel { initialization, .. } => {
                    initialization.as_ref().err()?
                }
                native::PcieController::Broadcom { initialization, .. } => {
                    initialization.as_ref().err()?
                }
            };
            Some((hci.location.as_str(), error))
        });
        usb.chain(pcie)
    }
}

/// Enumerate PCI and bring up every virtio-blk, NVMe and AHCI controller,
/// every virtio-input device, every virtio-net and supported Intel/Realtek Ethernet adapter,
/// and every xHCI
/// controller with the USB disks, keyboards, pointers and Ethernet
/// adapters on it.
///
/// # Safety
///
/// `handoff` must describe the running machine after ExitBootServices,
/// with its ranges mapped as it says, and nothing else may drive these
/// controllers afterwards.
pub unsafe fn probe(
    handoff: &Handoff,
    captures: PlatformCaptures,
    dma: &mut dyn DmaAllocator,
) -> Devices {
    let firmware = handoff_provider(handoff);
    // SAFETY: the caller granted the complete machine/controller contract.
    unsafe { probe_with_provider(handoff, captures, dma, firmware, None, None) }
}

/// Attach legacy drivers with a standalone mount's owned minimum-delay clock.
/// Attachments that need a native resource broker still report unavailable.
///
/// # Safety
/// The requirements of [`probe`] apply. `minimum` must cover the actual
/// physical minimum delays of the mounted machine and outlive its users.
pub unsafe fn probe_with_minimum(
    handoff: &Handoff,
    captures: PlatformCaptures,
    dma: &mut dyn DmaAllocator,
    minimum: Rc<dyn crate::bus::pci::power::Timer>,
) -> Devices {
    // SAFETY: the caller owns the same handoff, mappings and DMA as probe().
    unsafe {
        probe_with_resources(
            handoff,
            captures,
            dma,
            handoff_provider(handoff),
            ProbeResources::Standalone(minimum),
            None,
            &registry::DriverPolicy::default(),
        )
    }
}

/// The handoff's firmware catalog as the provider [`probe_with_provider`]
/// takes. Attached devices keep borrowing their firmware for as long as
/// they run, so the catalog lives for the rest of the boot.
pub fn handoff_provider(handoff: &Handoff) -> &'static crate::firmware::HandoffProvider<'static> {
    let catalog = handoff
        .firmware
        .clone()
        // SAFETY: the reference count is never given back, so the catalog
        // stays allocated for every later borrow.
        .map(|catalog| unsafe { &*Arc::into_raw(catalog) });
    Box::leak(Box::new(crate::firmware::HandoffProvider::new(catalog)))
}

/// What every Intel Wi-Fi attach in one probe shares.
struct WifiInputs<'a> {
    clock: Clock,
    observations: net::wifi::intel::platform::Observations,
    firmware: &'static dyn crate::firmware::Provider,
    network: &'a Network,
}

/// Attach the Intel Wi-Fi function at `address` for `wifi.network`, reading
/// its ACPI companion, once the regulatory database loaded.
///
/// # Safety
/// [`NativeServices`]' contract holds for `services`, and nothing else
/// reaches the function.
#[allow(
    clippy::result_large_err,
    reason = "a failed attach hands back the owners that must outlive the device"
)]
unsafe fn intel_wifi(
    services: &dyn NativeServices,
    address: Address,
    completion: Option<crate::bus::irq::MsiMessage>,
    wifi: &WifiInputs<'_>,
    regulatory: &Option<
        Result<
            crate::net::wifi::regulatory::database::Database,
            crate::net::wifi::regulatory::database::LoadError,
        >,
    >,
    bridges: &[native::Bridge],
) -> Result<Adapter, native::Failure> {
    use net::wifi::intel::attach::AttachError;
    let Some(Ok(database)) = regulatory else {
        return Err((Error::IntelWifi(AttachError::NoRegulatoryDatabase), None));
    };
    let WifiInputs {
        clock,
        observations,
        firmware,
        network,
    } = *wifi;
    #[cfg(feature = "native-acpi")]
    {
        let session = services.acpi();
        let companion = session.and_then(|session| native::companion(session, address, bridges));
        // SAFETY: `companion` is this function's own, and the handoff's
        // captured variables are this boot's.
        let bios = unsafe {
            net::wifi::intel::radio::bios::service::native::NativeBios::new(
                session,
                companion,
                firmware,
                observations,
            )
        };
        let inputs = net::wifi::intel::link::Inputs {
            firmware,
            observations,
            pages: services.pages(),
            clock,
            bios,
            database: database.clone(),
            network: network.clone(),
            entropy: services
                .entropy()
                .map_err(|error| (Error::Native(error), None))?,
        };
        // SAFETY: forwarded from the caller.
        unsafe { native::intel_wifi(services, address, completion, inputs) }
            .map(|link| Adapter::IntelWifi(Box::new(link)))
    }
    #[cfg(not(feature = "native-acpi"))]
    {
        let _ = (
            services,
            address,
            completion,
            clock,
            observations,
            firmware,
            network,
            database,
            bridges,
        );
        Err((Error::IntelWifi(AttachError::NoNamespace), None))
    }
}

/// Attach the MediaTek Wi-Fi function at `address` for `wifi.network`, its
/// SAR tables read from its ACPI companion, once the regulatory database
/// loaded. `aspm` is whether firmware enabled ASPM on its link.
///
/// # Safety
/// As for [`intel_wifi`].
#[allow(
    clippy::result_large_err,
    reason = "a failed attach hands back the owners that must outlive the device"
)]
unsafe fn mediatek_wifi(
    services: &dyn NativeServices,
    address: Address,
    completion: Option<crate::bus::irq::MsiMessage>,
    wifi: &WifiInputs<'_>,
    regulatory: &Option<
        Result<
            crate::net::wifi::regulatory::database::Database,
            crate::net::wifi::regulatory::database::LoadError,
        >,
    >,
    bridges: &[native::Bridge],
    aspm: bool,
) -> Result<
    native::Located<net::wifi::mediatek::attach::Attached<native::MediatekOwner>>,
    native::Failure,
> {
    use net::wifi::mediatek::attach::AttachError;
    let Some(Ok(database)) = regulatory else {
        return Err((Error::MediatekWifi(AttachError::NoRegulatoryDatabase), None));
    };
    let entropy = services
        .entropy()
        .map_err(|error| (Error::Native(error), None))?;
    #[cfg(feature = "native-acpi")]
    let sar = net::wifi::mediatek::acpi::read(
        services
            .acpi()
            .and_then(|session| native::companion(session, address, bridges)),
    );
    #[cfg(not(feature = "native-acpi"))]
    let sar = {
        let _ = bridges;
        net::wifi::mediatek::acpi::Sar::default()
    };
    let inputs = net::wifi::mediatek::attach::Inputs {
        firmware: wifi.firmware,
        pages: services.pages(),
        clock: wifi.clock,
        vector: false,
        aspm,
        database: database.clone(),
        network: wifi.network.clone(),
        sar,
        entropy,
    };
    // SAFETY: forwarded from the caller.
    unsafe { native::mediatek_wifi(services, address, completion, inputs) }
}

/// Attach the RTL8822C at `address` for `wifi.network`, once the regulatory
/// database loaded.
///
/// # Safety
/// As for [`intel_wifi`].
#[allow(
    clippy::result_large_err,
    reason = "a failed attach hands back the owners that must outlive the device"
)]
unsafe fn realtek_wifi(
    services: &dyn NativeServices,
    address: Address,
    completion: Option<crate::bus::irq::MsiMessage>,
    wifi: &WifiInputs<'_>,
    regulatory: &Option<
        Result<
            crate::net::wifi::regulatory::database::Database,
            crate::net::wifi::regulatory::database::LoadError,
        >,
    >,
) -> Result<native::Located<native::RealtekWifi>, native::Failure> {
    use net::wifi::realtek::attach::{AttachError, Inputs};
    let Some(Ok(database)) = regulatory else {
        return Err((Error::RealtekWifi(AttachError::NoRegulatoryDatabase), None));
    };
    let inputs = Inputs {
        firmware: wifi.firmware,
        pages: services.pages(),
        clock: wifi.clock,
        database: database.clone(),
        network: wifi.network.clone(),
        entropy: services
            .entropy()
            .map_err(|error| (Error::Native(error), None))?,
        vector: false,
        link_control: 0,
    };
    // SAFETY: forwarded from the caller.
    unsafe { native::realtek_wifi(services, address, completion, inputs) }
}

/// Attach the Broadcom PCIe Bluetooth function at `address` with its
/// segment's bridges, for the board SMBIOS names, as `bcm4377_probe_dmi`
/// selects it.
///
/// # Safety
/// As for [`intel_wifi`]; `bridges` lists every bridge of the segment.
#[allow(
    clippy::result_large_err,
    reason = "a failed attach hands back the owners that must outlive the device"
)]
unsafe fn bcm4377(
    services: &dyn NativeServices,
    address: Address,
    bridges: &[native::Bridge],
    firmware: &'static dyn crate::firmware::Provider,
    clock: Clock,
) -> Result<native::Bcm4377, native::Failure> {
    use crate::bluetooth::bcm4377::{Board, Error as Bcm4377Error};
    use crate::firmware::smbios::Observation;
    let refused = |error: ServiceError| (Error::Native(error), None);
    let text = |observation| match observation {
        Observation::Value(bytes) => core::str::from_utf8(bytes).ok(),
        _ => None,
    };
    let board = crate::firmware::smbios::load(firmware)
        .ok()
        .and_then(|snapshot| {
            let identity = snapshot.identity;
            Board::from_dmi(text(identity.system_vendor)?, text(identity.product_name)?).ok()
        })
        .ok_or((
            Error::Bcm4377(Bcm4377Error::Unsupported(
                "an SMBIOS system the source's DMI board table lists",
            )),
            None,
        ))?;
    #[cfg(feature = "native-acpi")]
    let roots = services
        .acpi()
        .map(|session| native::root_buses(session, address.segment))
        .unwrap_or_default();
    #[cfg(not(feature = "native-acpi"))]
    let roots = Vec::new();
    if roots.is_empty() {
        return Err(refused(ServiceError::Refused(
            "no ACPI host bridge declares the segment's root buses",
        )));
    }
    let mut owned = Vec::new();
    for bridge in bridges
        .iter()
        .filter(|bridge| bridge.address.segment == address.segment)
    {
        owned.push(services.bridge(bridge.address).map_err(refused)?);
    }
    // SAFETY: these are the mount's owners of every bridge the scan found
    // in the segment and the root buses its ACPI host bridges declare; no
    // other driver sets their link policy.
    let topology = unsafe { crate::bus::pci::native_aspm::Topology::new(owned, roots) };
    // SAFETY: forwarded from the caller.
    unsafe { native::bcm4377(services, address, topology, board, clock) }
}

/// Enumerate and attach devices using an explicit exact firmware provider.
/// A loader can observe consumer requirements while delegating the original
/// lookup result, without reconstructing device identities outside drivers.
/// This performs the same hardware operations and reports the same failures as
/// [`probe`]; a missing resource still refuses that controller's initialization.
/// `services` supplies the native owners some drivers attach through.
/// `wifi` is the network the mount's configuration names; without one a
/// Wi-Fi adapter is unsupported and needs no Wi-Fi firmware or regulatory
/// data.
///
/// # Safety
/// The machine, mappings, DMA and exclusive post-ExitBootServices ownership
/// requirements of [`probe`] apply. The provider must preserve its documented
/// exact identity and integrity contract for every resource it returns. It
/// lives for the rest of the boot because attached devices keep borrowing
/// the firmware it returned.
pub unsafe fn probe_with_provider(
    handoff: &Handoff,
    captures: PlatformCaptures,
    dma: &mut dyn DmaAllocator,
    firmware: &'static dyn crate::firmware::Provider,
    services: Option<&dyn NativeServices>,
    wifi: Option<&Network>,
) -> Devices {
    // SAFETY: forwarded from the caller.
    unsafe {
        probe_with_policy(
            handoff,
            captures,
            dma,
            firmware,
            services,
            wifi,
            &registry::DriverPolicy::default(),
        )
    }
}

/// Apply the selected policy to PCI attachment. USB firmware handoff and
/// downstream USB class admission still use the mount's existing rules.
/// Display inventory remains available. The post-scan NVIDIA consumer
/// applies the same policy through [`registry::DriverPolicy::allows_nvmodern`].
///
/// # Safety
/// The requirements of [`probe_with_provider`] apply.
pub unsafe fn probe_with_policy(
    handoff: &Handoff,
    captures: PlatformCaptures,
    dma: &mut dyn DmaAllocator,
    firmware: &'static dyn crate::firmware::Provider,
    services: Option<&dyn NativeServices>,
    wifi: Option<&Network>,
    policy: &registry::DriverPolicy<'_>,
) -> Devices {
    // SAFETY: forwarded from the caller's native probe contract.
    unsafe {
        probe_with_resources(
            handoff,
            captures,
            dma,
            firmware,
            ProbeResources::Services(services),
            wifi,
            policy,
        )
    }
}

enum ProbeResources<'a> {
    Services(Option<&'a dyn NativeServices>),
    Standalone(Rc<dyn crate::bus::pci::power::Timer>),
}

unsafe fn probe_with_resources(
    handoff: &Handoff,
    captures: PlatformCaptures,
    dma: &mut dyn DmaAllocator,
    firmware: &'static dyn crate::firmware::Provider,
    resources: ProbeResources<'_>,
    wifi: Option<&Network>,
    policy: &registry::DriverPolicy<'_>,
) -> Devices {
    let (services, standalone_minimum) = match resources {
        ProbeResources::Services(services) => (services, None),
        ProbeResources::Standalone(minimum) => (None, Some(minimum)),
    };
    let mut devices = Devices {
        bluetooth_platform: captures.bluetooth,
        native_services: services.is_some(),
        ..Devices::default()
    };
    let wifi_inputs = wifi.map(|network| WifiInputs {
        clock: handoff.clock,
        observations: captures.wifi,
        firmware,
        network,
    });
    // Broadcom PCIe Bluetooth functions, attached once the scan has found
    // every bridge of their segment.
    let mut broadcom_bluetooth = Vec::new();
    let legacy_pci_writes = services.is_none_or(NativeServices::legacy_pci_writes);
    let mut ehci_hosts = Vec::new();
    let mut usb_handoffs = UsbHandoffStatus {
        companions_released: true,
        legacy_released: true,
    };
    // USB4 hosts with a software connection manager come up first: the scan
    // below then finds what they tunnel, as it finds any other function.
    let software_usb4 = services.map_or_else(Vec::new, |services| {
        // SAFETY: the caller owns the machine, and no driver has started.
        unsafe { native::usb4::bring_up(services, handoff, policy, &mut devices) }
    });
    let mut usb4_hosts = Vec::new();
    // Each PCI bridge and its secondary bus, for ACPI companion lookups.
    let mut bridges = Vec::new();
    // i801 SMBus hosts, attached under ACPI arbitration once the embedded
    // controller is.
    #[cfg(feature = "native-acpi")]
    let mut smbus_functions = Vec::new();
    let minimum_result = match (services, standalone_minimum) {
        (Some(services), _) => services.minimum_delay(),
        (None, Some(minimum)) => Ok(minimum),
        (None, None) => Err(ServiceError::Unavailable),
    };
    // Every window is enumerated before any driver runs, as Linux finds
    // every device before its fixups: the Intel xHCI port switch asks
    // whether an Intel EHCI exists anywhere.
    let (scanned, pci_scan_complete) = scan(handoff, &mut devices);
    #[cfg(feature = "native-acpi")]
    let pci_identities = if services.is_some() {
        scanned.iter().flatten().map(Function::identity).collect()
    } else {
        Vec::new()
    };
    let mut abandoned = Vec::new();
    let intel_ehci = scanned.iter().flatten().any(|function| {
        function.vendor == 0x8086
            && usb::handoff::Kind::of(function.class) == Some(usb::handoff::Kind::Ehci)
    });
    let displays: Vec<Address> = scanned
        .iter()
        .flatten()
        .filter(|function| function.class[0] == 0x03 && function.address.function == 0)
        .map(|function| function.address)
        .collect();
    for found in scanned {
        for mut function in found {
            let address = function.address;
            let identity = function.identity();
            if function.class[..2] == [6, 4] {
                use crate::bus::pci::ConfigSpace;
                let checked = (|| -> Result<_, BlockError> {
                    Ok(native::Bridge {
                        address,
                        secondary: function.config().read8(0x19)?,
                        aspm: crate::bus::pci::aspm_control(&function)?,
                    })
                })();
                match checked {
                    Ok(bridge) => bridges.push(bridge),
                    Err(error) => {
                        devices.failures.push((address, error.into()));
                        devices.owners.push(Box::new(function));
                        continue;
                    }
                }
            }
            let early = registry::early(&identity);
            let selection = registry::select_pci(&identity, policy);
            let refusal = selection.refusal();
            if let Some(why) = refusal {
                devices.unsupported.push((address, function.class, why));
                if !matches!(
                    early,
                    Some(registry::Early::Display | registry::Early::Usb(_))
                ) {
                    continue;
                }
            }
            let driver = selection.driver();
            if function.class[..2] == [2, 0x80] {
                if wifi.is_none()
                    && matches!(
                        driver,
                        None | Some(registry::Driver::Legacy(_))
                            | Some(registry::Driver::Native(
                                native::Kind::IntelWifi
                                    | native::Kind::MediatekWifi
                                    | native::Kind::RealtekWifi
                            ))
                    )
                {
                    // No Wi-Fi hardware is touched without a network. Other
                    // functions of this class, such as Broadcom Bluetooth,
                    // still attach.
                    devices.unsupported.push((
                        address,
                        function.class,
                        "Wi-Fi adapter stays off: no --wifi network",
                    ));
                    continue;
                }
                if wifi.is_some() && devices.regulatory.is_none() {
                    devices.regulatory =
                        Some(crate::net::wifi::regulatory::database::load(firmware));
                }
            }
            if let Some(registry::Early::Display) = early {
                devices.gpus.extend(gpu::Adapter::inspect(&mut function));
                continue;
            }
            crate::working(&alloc::format!("PCI {address}"));
            let start = DeviceStart::begin(address, handoff.clock, &mut abandoned);
            #[cfg(feature = "native-acpi")]
            if let Some(registry::Early::Smbus) = early {
                smbus_functions.push(address);
                continue;
            }
            if let Some(registry::Early::Usb(kind)) = early {
                let inputs = UsbHandoff {
                    handoff,
                    minimum: &minimum_result,
                    legacy_pci_writes,
                    intel_ehci,
                    attach_ehci: selection.attaches_ehci(),
                    route_ports: refusal.is_none(),
                };
                let Some(handed) = usb_handoff(
                    function,
                    kind,
                    &inputs,
                    &mut ehci_hosts,
                    &mut usb_handoffs,
                    &mut devices,
                ) else {
                    continue;
                };
                function = handed;
                if refusal.is_some() {
                    continue;
                }
            }
            if let Some(registry::Driver::Native(kind)) = driver {
                let Some(services) = services else {
                    devices
                        .failures
                        .push((address, Error::Native(ServiceError::Unavailable)));
                    continue;
                };
                match kind {
                    // Attached once the scan has found every bridge of the
                    // segment.
                    native::Kind::Bcm4377 => broadcom_bluetooth.push(address),
                    native::Kind::Usb4 => {
                        if !software_usb4.contains(&address) {
                            usb4_hosts.push(address);
                        }
                    }
                    // SAFETY: the caller grants `services`' contract, and
                    // this loop leaves `function` alone once it is claimed.
                    _ => unsafe {
                        attach_native(
                            kind,
                            function,
                            services,
                            &NativeInputs {
                                handoff,
                                firmware,
                                minimum: &minimum_result,
                                wifi: wifi_inputs.as_ref(),
                                bridges: &bridges,
                            },
                            &mut devices,
                        )
                    },
                }
                continue;
            }
            if !legacy_pci_writes {
                if function.class[..2] != [6, 4] {
                    devices.unsupported.push((
                        address,
                        function.class,
                        "legacy PCI writes are excluded by the blocked requester fabric",
                    ));
                }
                continue;
            }
            if let Some(registry::Driver::Legacy(attach)) = driver {
                let minimum = match &minimum_result {
                    Ok(minimum) => minimum,
                    Err(error) => {
                        devices.failures.push((address, Error::Native(*error)));
                        devices.owners.push(Box::new(function));
                        continue;
                    }
                };
                // SAFETY: the caller owns the mapped functions, and the
                // mount allows legacy PCI writes.
                unsafe {
                    attach(
                        function,
                        &mut legacy::Context {
                            handoff,
                            dma,
                            firmware,
                            minimum,
                            budget: start.budget.as_ref(),
                            devices: &mut devices,
                        },
                    );
                }
            } else if let Some(why) = unclaimed(function.class, address, &displays) {
                devices.unsupported.push((address, function.class, why));
            }
        }
    }
    devices.abandoned = abandoned;
    let companions_released = pci_scan_complete && usb_handoffs.companions_released;
    #[cfg_attr(not(feature = "native-acpi"), allow(unused_variables))]
    let ehci_legacy_released = attach_ehci_hosts(
        ehci_hosts,
        companions_released,
        services,
        handoff,
        &mut devices,
    );
    attach_usb4_hosts(usb4_hosts, services, &bridges, handoff, &mut devices);
    attach_broadcom_bluetooth(
        broadcom_bluetooth,
        services,
        &bridges,
        firmware,
        handoff,
        &mut devices,
    );
    #[cfg(feature = "native-acpi")]
    attach_platform(
        services,
        smbus_functions,
        &bridges,
        &pci_identities,
        handoff,
        usb_handoffs.permits_i8042(
            pci_scan_complete && !handoff.ecam.is_empty(),
            ehci_legacy_released,
        ),
        &mut devices,
    );
    devices
}

/// What the USB hosts' firmware handoffs in one scan share.
struct UsbHandoff<'a> {
    handoff: &'a Handoff,
    minimum: &'a Result<Rc<dyn crate::bus::pci::power::Timer>, ServiceError>,
    /// Whether the mount allows the legacy PCI writes the handoff makes.
    legacy_pci_writes: bool,
    /// Whether the scan found an Intel EHCI anywhere.
    intel_ehci: bool,
    /// Whether policy allows the separate native EHCI attach.
    attach_ehci: bool,
    /// Whether policy allows the xHCI driver that uses the routed ports.
    route_ports: bool,
}

struct PendingEhci {
    address: Address,
    legacy_released: bool,
}

struct UsbHandoffStatus {
    companions_released: bool,
    legacy_released: bool,
}

impl UsbHandoffStatus {
    fn failed(&mut self, kind: usb::handoff::Kind) {
        if matches!(kind, usb::handoff::Kind::Uhci | usb::handoff::Kind::Ohci) {
            self.companions_released = false;
        }
        self.legacy_released = false;
    }

    #[cfg(any(feature = "native-acpi", test))]
    fn permits_i8042(&self, pci_enumerated: bool, ehci_legacy_released: bool) -> bool {
        pci_enumerated && self.legacy_released && ehci_legacy_released
    }
}

/// Take a USB host of `kind` from firmware before any driver binds, as
/// Linux's early PCI fixup does, and return it for its driver. EHCI hosts
/// go to `ehci_hosts` for their native attach. `None` when the function
/// goes no further in this scan.
fn usb_handoff(
    mut function: Function,
    kind: usb::handoff::Kind,
    inputs: &UsbHandoff<'_>,
    ehci_hosts: &mut Vec<PendingEhci>,
    status: &mut UsbHandoffStatus,
    devices: &mut Devices,
) -> Option<Function> {
    let address = function.address;
    // EHCI's native attach takes the controller through its own
    // owners when legacy writes are excluded.
    if kind == usb::handoff::Kind::Ehci && !inputs.legacy_pci_writes {
        if inputs.attach_ehci {
            ehci_hosts.push(PendingEhci {
                address,
                legacy_released: false,
            });
        } else {
            status.failed(kind);
        }
        return None;
    }
    if !inputs.legacy_pci_writes {
        status.failed(kind);
        devices.failures.push((
            address,
            Error::Native(ServiceError::Refused("legacy USB PCI handoff unavailable")),
        ));
        return None;
    }
    let minimum = match inputs.minimum {
        Ok(minimum) => minimum,
        Err(error) => {
            status.failed(kind);
            devices.failures.push((address, Error::Native(*error)));
            devices.owners.push(Box::new(function));
            return None;
        }
    };
    // SAFETY: the caller owns the machine and no driver has bound
    // this controller yet, as Linux's early PCI fixup runs.
    let legacy_released =
        match unsafe { usb::handoff::early(&mut function, inputs.handoff, minimum.as_ref()) } {
            Some(Ok(handed)) => {
                let warning = if !handed.legacy_released {
                    Some("USB host firmware handoff did not verify that legacy SMIs are off")
                } else if handed.forced {
                    Some("USB host firmware handoff timed out; ownership taken with its SMIs off")
                } else {
                    None
                };
                if let Some(warning) = warning {
                    devices.warnings.push((address, Warning::Message(warning)));
                }
                handed.legacy_released
            }
            Some(Err(error)) => {
                devices.platform_failures.push((
                    alloc::format!("{address} {} handoff", kind.name()),
                    PlatformError::Handoff(error),
                ));
                if kind != usb::handoff::Kind::Ehci {
                    status.failed(kind);
                    devices.owners.push(Box::new(function));
                    return None;
                }
                false
            }
            None => false,
        };
    if kind == usb::handoff::Kind::Ehci {
        if inputs.attach_ehci {
            // The native attach retries through its own requester and BAR.
            ehci_hosts.push(PendingEhci {
                address,
                legacy_released,
            });
        } else if !legacy_released {
            status.failed(kind);
            devices.owners.push(Box::new(function));
        }
        return None;
    }
    status.legacy_released &= legacy_released;
    if !inputs.route_ports {
        return Some(function);
    }
    // `quirk_usb_handoff_xhci` ends with the Intel port switch,
    // before the xHCI driver binds.
    match usb::handoff::switch_intel_ports(&mut function, inputs.intel_ehci) {
        Ok(Some(switched)) => devices
            .warnings
            .push((address, Warning::IntelPortSwitch(switched))),
        Ok(None) => {}
        Err(error) => devices.platform_failures.push((
            alloc::format!("{address} xHCI port switch"),
            PlatformError::Handoff(error),
        )),
    }
    Some(function)
}

/// What every native attach in one scan shares.
struct NativeInputs<'a> {
    handoff: &'a Handoff,
    firmware: &'static dyn crate::firmware::Provider,
    minimum: &'a Result<Rc<dyn crate::bus::pci::power::Timer>, ServiceError>,
    wifi: Option<&'a WifiInputs<'a>>,
    /// Each PCI bridge the scan has found so far, for ACPI companion lookups.
    bridges: &'a [native::Bridge],
}

/// Attach `function` through the native driver of `kind`, recording the
/// device, or the failure and the owners it keeps, in `devices`.
///
/// # Safety
/// [`NativeServices`]' contract holds for `services`, and nothing else
/// reaches the function.
unsafe fn attach_native(
    kind: native::Kind,
    function: Function,
    services: &dyn NativeServices,
    inputs: &NativeInputs<'_>,
    devices: &mut Devices,
) {
    let address = function.address;
    let attached = match kind {
        // Both hubs need a native notification owner that does
        // not exist yet, so their engines are left as firmware
        // left them.
        native::Kind::Sfh => {
            devices.unsupported.push((
                address,
                function.class,
                "AMD Sensor Fusion Hub: no native notification and mode owner yet",
            ));
            return;
        }
        native::Kind::Ish => {
            devices.unsupported.push((
                    address,
                    function.class,
                    "Intel Integrated Sensor Hub: no native notification owner or firmware DMA address reset yet",
                ));
            return;
        }
        native::Kind::IntelBluetooth => {
            // SAFETY: as for tg3 below.
            match unsafe { native::intel_bluetooth(services, address, inputs.handoff.clock) } {
                Ok(mut transport) => {
                    let initialization = transport.initialize_firmware(
                        inputs.firmware,
                        devices.bluetooth_platform.platform_inputs(),
                    );
                    devices.pcie_bluetooth.push(native::PcieHci {
                        location: alloc::format!("{address}"),
                        controller: native::PcieController::Intel {
                            transport,
                            initialization,
                        },
                    });
                }
                Err((error, kept)) => {
                    devices.failures.push((address, error));
                    devices.owners.extend(kept);
                }
            }
            return;
        }
        // The scan defers these until it has found every bridge of the
        // segment (`attach_broadcom_bluetooth`, `attach_usb4_hosts`).
        native::Kind::Bcm4377 | native::Kind::Usb4 => return,
        native::Kind::Hda => {
            let minimum = match inputs.minimum {
                Ok(minimum) => minimum,
                Err(error) => {
                    devices.failures.push((address, Error::Native(*error)));
                    devices.owners.push(Box::new(function));
                    return;
                }
            };
            // SAFETY: as for tg3 below; only conventional
            // controllers, which no firmware agent drives, match.
            match unsafe {
                native::hda::attach(services, address, inputs.handoff.clock, minimum.clone())
            } {
                Ok(host) => devices.hda_hosts.push(host),
                Err((error, kept)) => {
                    devices.failures.push((address, error));
                    devices.owners.extend(kept);
                }
            }
            return;
        }
        native::Kind::IntelWifi => {
            let Some(wifi) = inputs.wifi else {
                return;
            };
            // SAFETY: as for tg3 below; the companion is this
            // function's own.
            match unsafe {
                intel_wifi(
                    services,
                    address,
                    inputs.handoff.completion,
                    wifi,
                    &devices.regulatory,
                    inputs.bridges,
                )
            } {
                Ok(link) => devices.nics.push(link),
                Err((error, kept)) => {
                    devices.failures.push((address, error));
                    devices.owners.extend(kept);
                }
            }
            return;
        }
        native::Kind::MediatekWifi => {
            let Some(wifi) = inputs.wifi else {
                return;
            };
            let aspm = match native::aspm_supported(&function, inputs.bridges) {
                Ok(aspm) => aspm,
                Err(error) => {
                    devices.failures.push((address, error.into()));
                    devices.owners.push(Box::new(function));
                    return;
                }
            };
            // SAFETY: as for tg3 below; the companion is this
            // function's own.
            match unsafe {
                mediatek_wifi(
                    services,
                    address,
                    inputs.handoff.completion,
                    wifi,
                    &devices.regulatory,
                    inputs.bridges,
                    aspm,
                )
            } {
                Ok(native::Located { location, link }) => {
                    if link.firmware_domain_unused {
                        devices.warnings.push((
                                address,
                                Warning::Message(
                                    "MT7925 firmware regulatory domain left unused; regulatory.db's country is programmed",
                                ),
                            ));
                    }
                    devices
                        .nics
                        .push(Adapter::MediatekWifi(Box::new(native::Located {
                            location,
                            link: link.station,
                        })));
                }
                Err((error, kept)) => {
                    devices.failures.push((address, error));
                    devices.owners.extend(kept);
                }
            }
            return;
        }
        native::Kind::RealtekWifi => {
            let Some(wifi) = inputs.wifi else {
                return;
            };
            // SAFETY: as for tg3 below.
            match unsafe {
                realtek_wifi(
                    services,
                    address,
                    inputs.handoff.completion,
                    wifi,
                    &devices.regulatory,
                )
            } {
                Ok(link) => {
                    devices.nics.push(Adapter::RealtekWifi(Box::new(link)));
                }
                Err((error, kept)) => {
                    devices.failures.push((address, error));
                    devices.owners.extend(kept);
                }
            }
            return;
        }
        native::Kind::Vmd => {
            // SAFETY: as for tg3 below; the host's hierarchy is
            // stopped with it.
            match unsafe { native::vmd(services, address, inputs.handoff.clock) } {
                Ok(found) => {
                    for child in &found.others {
                        let location = alloc::format!(
                            "{address} VMD {:02x}:{:02x}.{}",
                            child.address.bus,
                            child.address.device,
                            child.address.function
                        );
                        devices.device_failures.push((
                            location,
                            BlockError::Unsupported(
                                "device behind VMD other than its first NVMe controller",
                            ),
                        ));
                    }
                    if let Some(attached) = found.nvme {
                        devices
                            .device_failures
                            .extend(attached.failures.into_iter().map(|(namespace, error)| {
                                (alloc::format!("{address} VMD namespace {namespace}"), error)
                            }));
                        devices.disks.extend(
                            attached
                                .namespaces
                                .into_iter()
                                .map(|disk| Box::new(disk) as Box<dyn BlockDevice>),
                        );
                    }
                    devices.owners.push(found.owner);
                }
                Err((error, kept)) => {
                    devices.failures.push((address, error));
                    devices.owners.extend(kept);
                }
            }
            return;
        }
        native::Kind::Tg3 => {
            // SAFETY: the caller grants `services`' contract, and
            // this loop leaves `function` alone once it is claimed.
            let link = unsafe { native::tg3(services, address) };
            link.map(|link| Adapter::Tg3(Box::new(link)))
        }
        native::Kind::Aqtion => {
            // SAFETY: as for tg3.
            let link = unsafe { native::aqtion(services, address) };
            link.map(|link| Adapter::Aqtion(Box::new(link)))
        }
    };
    match attached {
        Ok(adapter) => devices.nics.push(adapter),
        Err((error, kept)) => {
            devices.failures.push((address, error));
            devices.owners.extend(kept);
        }
    }
}

/// Enumerate every mapped ECAM window, and whether every window was
/// scanned. Unmapped windows and failed enumerations go to `devices`.
fn scan(handoff: &Handoff, devices: &mut Devices) -> (Vec<Vec<Function>>, bool) {
    let mut pci_scan_complete = true;
    let mut scanned = Vec::new();
    for window in &handoff.ecam {
        if !handoff.is_mapped(&window.range()) {
            devices.unmapped.push(*window);
            pci_scan_complete = false;
            continue;
        }
        // SAFETY: mapped as checked; the caller owns the machine.
        match unsafe { Function::enumerate(window) } {
            Ok(found) => scanned.push(found),
            Err(failure) => {
                pci_scan_complete = false;
                devices
                    .failures
                    .push((failure.address(), Error::Enumeration(failure.error)));
                devices.owners.push(Box::new(failure));
            }
        }
    }
    (scanned, pci_scan_complete)
}

/// Attach each EHCI host the scan found, once every UHCI/OHCI companion
/// has left firmware, with the USB devices on it.
fn attach_ehci_hosts(
    ehci_hosts: Vec<PendingEhci>,
    companions_released: bool,
    services: Option<&dyn NativeServices>,
    handoff: &Handoff,
    devices: &mut Devices,
) -> bool {
    let mut all_legacy_released = true;
    for PendingEhci {
        address,
        mut legacy_released,
    } in ehci_hosts
    {
        let attached = match services {
            Some(services) if companions_released => {
                // SAFETY: the completed scan has run the UHCI/OHCI handoff
                // for every discovered companion. The native attach owns
                // EHCI's BIOS handoff through its original requester and BAR.
                unsafe { native::ehci(services, address, handoff.clock, &mut legacy_released) }
            }
            Some(_) => Err((
                Error::Native(ServiceError::Refused("EHCI companion handoff incomplete")),
                None,
            )),
            None => Err((Error::Native(ServiceError::Unavailable), None)),
        };
        all_legacy_released &= legacy_released;
        let controller = match attached {
            Ok(attached) => attached,
            Err((error, kept)) => {
                devices.failures.push((address, error));
                devices.owners.extend(kept);
                continue;
            }
        };
        let found = usb::ehci::host::enumerate(
            &controller,
            &alloc::format!("{address} EHCI"),
            handoff.clock,
        );
        devices.disks.extend(
            found
                .disks
                .into_iter()
                .map(|disk| Box::new(disk) as Box<dyn BlockDevice>),
        );
        devices.usb_storage.extend(found.storage);
        devices.usb_failures.extend(found.failures);
        devices.usb_unused.extend(found.unused);
        devices.inputs.extend(
            found
                .inputs
                .into_iter()
                .map(|input| InputDevice::Ehci(Box::new(input))),
        );
        devices.ehci_hosts.push(controller);
    }
    all_legacy_released
}

/// Attach each USB4 host left to the firmware connection manager, once the
/// scan has found every bridge of its segment.
fn attach_usb4_hosts(
    usb4_hosts: Vec<Address>,
    services: Option<&dyn NativeServices>,
    bridges: &[native::Bridge],
    handoff: &Handoff,
    devices: &mut Devices,
) {
    for address in usb4_hosts {
        let Some(services) = services else {
            devices
                .failures
                .push((address, Error::Native(ServiceError::Unavailable)));
            continue;
        };
        // SAFETY: as for tg3; the scan found every bridge of the segment.
        match unsafe { native::usb4::attach(services, address, bridges, handoff.clock, false) } {
            Ok(host) => devices.usb4_hosts.push(host),
            Err((error, kept)) => {
                devices.failures.push((address, error));
                devices.owners.extend(kept);
            }
        }
    }
}

/// Attach each Broadcom PCIe Bluetooth function, once the scan has found
/// every bridge of its segment, and load its firmware.
fn attach_broadcom_bluetooth(
    broadcom_bluetooth: Vec<Address>,
    services: Option<&dyn NativeServices>,
    bridges: &[native::Bridge],
    firmware: &'static dyn crate::firmware::Provider,
    handoff: &Handoff,
    devices: &mut Devices,
) {
    for address in broadcom_bluetooth {
        let Some(services) = services else {
            devices
                .failures
                .push((address, Error::Native(ServiceError::Unavailable)));
            continue;
        };
        // SAFETY: as for tg3; the scan found every bridge of the segment.
        match unsafe { bcm4377(services, address, bridges, firmware, handoff.clock) } {
            Ok(mut transport) => {
                let initialization = transport.initialize(firmware);
                devices.pcie_bluetooth.push(native::PcieHci {
                    location: alloc::format!("{address}"),
                    controller: native::PcieController::Broadcom {
                        transport,
                        initialization,
                    },
                });
            }
            Err((error, kept)) => {
                devices.failures.push((address, error));
                devices.owners.extend(kept);
            }
        }
    }
}

/// The devices the ACPI namespace lists: the embedded controller first,
/// whose fields other devices' methods may read, then the i801 SMBus hosts,
/// the i8042, I2C, power and Type-C. Without an ACPI session each SMBus host
/// is reported unavailable.
#[cfg(feature = "native-acpi")]
fn attach_platform(
    services: Option<&dyn NativeServices>,
    mut smbus_functions: Vec<Address>,
    bridges: &[native::Bridge],
    pci_identities: &[crate::bus::pci::Identity],
    handoff: &Handoff,
    usb_legacy_released: bool,
    devices: &mut Devices,
) {
    if let Some((services, session)) =
        services.and_then(|services| services.acpi().map(|session| (services, session)))
    {
        // Before any other device's methods, which may read EC fields.
        let ec = match services.embedded_controller() {
            Ok(Some(installed)) => {
                devices.owners.push(Box::new(installed.clone()));
                Some(installed)
            }
            Ok(None) => power::attach_ec(session, devices),
            // The mount installs no EC while ACPI hardware is not active,
            // and reports its own failed install. Say so once, and only
            // when the namespace lists an EC.
            Err(ServiceError::Unavailable) => {
                if session
                    .devices(services::ids::EMBEDDED_CONTROLLER[0])
                    .is_ok_and(|found| !found.is_empty())
                {
                    devices.platform_unsupported.push((
                        String::from("embedded controller"),
                        "not attached: ACPI hardware is not active, or the mount's install failed",
                    ));
                }
                None
            }
            Err(error) => {
                devices.platform_failures.push((
                    String::from("embedded controller"),
                    PlatformError::Native(error),
                ));
                None
            }
        };
        for address in core::mem::take(&mut smbus_functions) {
            // SAFETY: the mount grants each i801's I/O window to probe.
            match unsafe { smbus::attach(services, session, address, bridges, handoff.clock) } {
                Ok(host) => devices.smbus_hosts.push(host),
                Err(error) => devices
                    .platform_failures
                    .push((alloc::format!("{address} SMBus"), error)),
            }
        }
        let smbus_hosts = devices.smbus_hosts.clone();
        platform::attach_i8042(
            session,
            handoff.clock,
            &smbus_hosts,
            usb_legacy_released,
            devices,
        );
        i2c::attach(services, session, handoff.clock, devices, pci_identities);
        power::attach_power(session, devices);
        typec::attach(services, session, ec.as_ref(), handoff.clock, devices);
    }
    for address in smbus_functions {
        devices.platform_failures.push((
            alloc::format!("{address} SMBus"),
            PlatformError::Native(ServiceError::Unavailable),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A function whose device never answers spends its whole budget and
    /// is recorded abandoned, with the wait it stopped at; the next
    /// function then starts with a fresh budget and answers normally.
    #[test]
    fn a_dead_function_is_abandoned_and_the_next_one_still_starts() {
        let _clock = crate::test_clock::virtualize();
        let clock = Clock {
            ticks_per_microsecond: 1,
        };
        let at = |bus| Address {
            segment: 0,
            bus,
            device: 0,
            function: 0,
        };
        let mut abandoned = Vec::new();
        let started = Clock::ticks();
        {
            let _start = DeviceStart::begin(at(0x0b), clock, &mut abandoned);
            // A driver step whose own bound is a minute: the budget ends it.
            assert!(!clock.wait(60_000_000, || false));
        }
        assert!(Clock::ticks() - started < DEVICE_START_US + 1_000);
        {
            let _start = DeviceStart::begin(at(0x0c), clock, &mut abandoned);
            assert!(!clock.deadline(1_000_000).passed());
            assert!(clock.wait(1_000_000, || true));
        }
        assert_eq!(abandoned.len(), 1);
        let (address, expiry) = abandoned[0];
        assert_eq!(address, at(0x0b));
        assert_eq!(expiry.budget_us, DEVICE_START_US);
        assert!(expiry.at.unwrap().file().ends_with("probe.rs"));
    }
    use crate::firmware::{
        self, Catalog, HandoffProvider, Provenance, Provider, Requirement, Resource, Target,
    };
    use alloc::format;
    use sha2::{Digest, Sha256};

    fn identity(
        bus: u8,
        function: u8,
        vendor: u16,
        device: u16,
        class: u32,
    ) -> crate::bus::pci::Identity {
        crate::bus::pci::Identity {
            address: Address {
                segment: 0,
                bus,
                device: 0,
                function,
            },
            vendor,
            device,
            class: [(class >> 16) as u8, (class >> 8) as u8, class as u8],
        }
    }

    #[test]
    fn plan_names_the_driver_of_common_desktop_functions() {
        let gpu = identity(1, 0, 0x10de, 0x2882, 0x03_00_00);
        let displays = [gpu.address];
        for (function, expected) in [
            (identity(0x0b, 0, 0x1022, 0x149c, 0x0c_03_30), "xHCI USB"),
            (identity(0x0c, 0, 0x1022, 0x7901, 0x01_06_01), "AHCI SATA"),
            (identity(0x04, 0, 0x144d, 0xa808, 0x01_08_02), "NVMe"),
            (
                identity(0x05, 0, 0x1969, 0x1063, 0x02_00_00),
                "atl1c Ethernet",
            ),
            (gpu, "GPU"),
            (identity(0x00, 1, 0x1022, 0x1483, 0x06_04_00), "PCI bridge"),
            (identity(0x00, 0, 0x1022, 0x1480, 0x06_00_00), "not driven"),
            (
                identity(0x05, 0, 0x1106, 0x0571, 0x01_01_80),
                "unsupported: IDE storage controller",
            ),
        ] {
            assert_eq!(
                plan(&function, &displays),
                expected,
                "{:?}",
                function.address
            );
        }
        // The GPU's own HDMI audio function, beside its function 0.
        let hdmi = identity(1, 1, 0x1234, 0x0001, 0x04_03_00);
        assert!(plan(&hdmi, &displays).starts_with("unsupported: HDMI/DP audio"));
        assert!(plan(&hdmi, &[]).starts_with("unsupported: HD Audio controller"));
    }

    #[test]
    fn plan_names_a_driver_exactly_for_the_functions_claimed() {
        let functions = [
            identity(0x0b, 0, 0x1022, 0x149c, 0x0c_03_30),
            identity(0x0b, 1, 0x1022, 0x43d5, 0x0c_03_30),
            identity(0x0c, 0, 0x1022, 0x7901, 0x01_06_01),
            identity(0x04, 0, 0x144d, 0xa808, 0x01_08_02),
            identity(0x0d, 0, 0x10ec, 0x8168, 0x02_00_00),
            identity(0x05, 0, 0x1969, 0x1063, 0x02_00_00),
            identity(0x0e, 0, 0x8086, 0x15f3, 0x02_00_00),
            identity(0x0f, 0, 0x1022, 0x1487, 0x04_03_00),
            identity(0x10, 0, 0x1af4, 0x1042, 0x01_00_00),
            identity(0x11, 0, 0x8086, 0x1e2d, 0x0c_03_20),
            identity(0x00, 0, 0x1022, 0x1480, 0x06_00_00),
            identity(0x12, 0, 0x1234, 0x5678, 0x0c_03_10),
        ];
        for function in functions {
            let named = plan(&function, &[]);
            // A USB host taken from firmware has its registers mapped too.
            let driven = !named.starts_with("unsupported") && named != "not driven";
            assert_eq!(
                driven,
                claims_identity(&function),
                "{:?} {named}",
                function.address
            );
        }
    }

    #[test]
    fn i8042_requires_the_usb_legacy_result_of_a_complete_scan() {
        let mut status = UsbHandoffStatus {
            companions_released: true,
            legacy_released: true,
        };
        assert!(status.permits_i8042(true, true));
        assert!(!status.permits_i8042(false, true));
        assert!(!status.permits_i8042(true, false));
        status.failed(usb::handoff::Kind::Xhci);
        assert!(status.companions_released);
        assert!(!status.permits_i8042(true, true));
        status.failed(usb::handoff::Kind::Ohci);
        assert!(!status.companions_released);
    }

    #[test]
    fn display_probe_observes_glk_memory_without_changing_ecam() {
        let mut ecam = alloc::vec![u32::MAX; (1 << 20) / 4];
        for (slot, identity) in [(2, 0x3185_8086), (3, 0x46a6_8086), (4, 0x2684_10de)] {
            let start = slot * (1 << 15) / 4;
            let config = &mut ecam[start..start + 4096 / 4];
            config.fill(0);
            config[0] = identity;
            config[0x08 / 4] = 0x0300_0000;
            config[0x50 / 4] = 0x0140;
            config[0x5c / 4] = 0x9be0_0001;
        }
        let original = ecam.clone();
        let window = EcamWindow {
            base: ecam.as_mut_ptr() as u64,
            segment: 0,
            start_bus: 0,
            end_bus: 0,
        };
        let handoff = Handoff {
            ecam: alloc::vec![window],
            mapped: alloc::vec![window.range()],
            dma: 0..0,
            clock: Clock {
                ticks_per_microsecond: 1,
            },
            firmware: None,
            completion: None,
        };
        let services = services::tests::Refusing::minimum_only();
        // SAFETY: an empty DMA region cannot allocate physical pages.
        let mut dma = unsafe { crate::bus::dma::RegionAllocator::new(0..0) };
        // SAFETY: only the live synthetic ECAM buffer is mapped. The services
        // refuse native access, and none of the display BARs is assigned.
        let devices = unsafe {
            probe_with_provider(
                &handoff,
                PlatformCaptures::default(),
                &mut dma,
                handoff_provider(&handoff),
                Some(&services),
                None,
            )
        };
        assert_eq!(ecam, original);
        assert!(services.preparations.borrow().is_empty());
        let [glk, other_intel, nvidia] = devices.gpus.as_slice() else {
            panic!("expected all three display functions in the inventory");
        };
        assert_eq!(glk.device, 0x3185);
        assert_eq!(
            glk.glk_memory,
            Some(Ok(gpu::intel::glk::MemoryLayout {
                ggtt_bytes: 2 << 20,
                stolen: Some(gpu::intel::glk::StolenMemory {
                    base: 0x9be0_0000,
                    bytes: 32 << 20,
                }),
            }))
        );
        assert_eq!(other_intel.device, 0x46a6);
        assert_eq!(other_intel.glk_memory, None);
        assert_eq!(nvidia.device, 0x2684);
        assert_eq!(nvidia.glk_memory, None);
    }

    #[test]
    fn ehci_uses_native_preparation_only_after_companion_handoff() {
        for companion_refused in [false, true] {
            let mut ecam = alloc::vec![u32::MAX; (1 << 20) / 4];
            let mut function = |device: usize, class: u32| {
                let start = device * (1 << 15) / 4;
                let page = &mut ecam[start..start + 4096 / 4];
                page.fill(0);
                page[0] = 0x1234_4321;
                page[2] = class;
            };
            function(0, 0x0c03_2000);
            if companion_refused {
                // The companion's BAR is absent or unmapped. The scan
                // discovers it after EHCI, and it cannot reach register I/O.
                function(1, 0x0c03_1000);
            }
            let window = EcamWindow {
                base: ecam.as_mut_ptr() as u64,
                segment: 0,
                start_bus: 0,
                end_bus: 0,
            };
            let handoff = Handoff {
                ecam: alloc::vec![window],
                mapped: alloc::vec![window.range()],
                dma: 0..0,
                clock: Clock {
                    ticks_per_microsecond: 1,
                },
                firmware: None,
                completion: None,
            };
            let services = services::tests::Refusing::minimum_only();
            // SAFETY: the empty region can allocate no DMA pages. Only the
            // synthetic ECAM buffer is mapped, and native claims all refuse.
            let mut dma = unsafe { crate::bus::dma::RegionAllocator::new(0..0) };
            // SAFETY: ECAM is the live fixture above. Its legacy BARs are
            // unmapped and the services refuse all native device access.
            let devices = unsafe {
                probe_with_provider(
                    &handoff,
                    PlatformCaptures::default(),
                    &mut dma,
                    handoff_provider(&handoff),
                    Some(&services),
                    None,
                )
            };
            let address = Address {
                segment: 0,
                bus: 0,
                device: 0,
                function: 0,
            };
            assert!(devices.ehci_hosts.is_empty());
            assert_eq!(ecam[1], 0);
            assert_eq!(ecam[4], 0);
            // The early handoff ran first and found the BAR unmapped; the
            // native attach still tries its own mapping.
            assert!(
                matches!(
                    devices.platform_failures.as_slice(),
                    [(name, PlatformError::Handoff(usb::handoff::Error::Unmapped)), ..]
                        if name.ends_with("EHCI handoff")
                ),
                "{:?}",
                devices.platform_failures
            );
            if companion_refused {
                assert!(services.preparations.borrow().is_empty());
                assert!(matches!(
                    devices.failures.as_slice(),
                    [(found, Error::Native(ServiceError::Refused("EHCI companion handoff incomplete")))]
                        if *found == address
                ));
            } else {
                assert_eq!(*services.preparations.borrow(), [address]);
                assert!(matches!(
                    devices.failures.as_slice(),
                    [(found, Error::Native(ServiceError::Unavailable))] if *found == address
                ));
            }
        }
    }

    /// The chooser refuses a disk whose controller no driver here takes,
    /// naming RAID mode, and offers every disk a driver reaches.
    #[test]
    fn disks_behind_undriven_storage_controllers_are_refused_before_the_choice() {
        let identity = |vendor, device, class| crate::bus::pci::Identity {
            address: Address {
                segment: 0,
                bus: 0,
                device: 0x17,
                function: 0,
            },
            vendor,
            device,
            class,
        };
        // A RAID-mode controller outside the AHCI table, and an IDE one.
        assert_eq!(
            disk_refusal(&identity(0x8086, 0x1234, [0x01, 0x04, 0x00])),
            Some("Its controller is in RAID mode: set SATA to AHCI in firmware setup")
        );
        assert_eq!(
            disk_refusal(&identity(0x1b4b, 0x9230, [0x01, 0x04, 0x00])),
            Some("Its controller is in RAID mode: set SATA to AHCI in firmware setup")
        );
        assert_eq!(
            disk_refusal(&identity(0x8086, 0x7ae2, [0x01, 0x01, 0x8a])),
            Some("Stela has no driver for its controller")
        );
        // AHCI, Intel RAID-mode SATA the AHCI table lists, NVMe, VMD.
        assert_eq!(
            disk_refusal(&identity(0x8086, 0x7ae2, [0x01, 0x06, 0x01])),
            None
        );
        assert_eq!(
            disk_refusal(&identity(0x8086, 0x2822, [0x01, 0x04, 0x00])),
            None
        );
        assert_eq!(
            disk_refusal(&identity(0x144d, 0xa808, [0x01, 0x08, 0x02])),
            None
        );
        assert_eq!(
            disk_refusal(&identity(0x8086, 0x467f, [0x01, 0x04, 0x00])),
            None
        );
        // xHCI: USB disks are refused by their own checks, not here.
        assert_eq!(
            disk_refusal(&identity(0x8086, 0x7ae0, [0x0c, 0x03, 0x30])),
            None
        );
    }

    #[test]
    fn unclaimed_host_classes_are_reported_and_others_ignored() {
        assert_eq!(
            unsupported([0x02, 0x80, 0x00]),
            Some("Wi-Fi controller other than Intel's, MediaTek's MT7925 and Realtek's RTL8822C")
        );
        assert_eq!(
            unsupported([0x01, 0x04, 0x00]),
            Some("RAID storage controller (Intel RST/VMD mode)")
        );
        assert_eq!(
            unsupported([0x0c, 0x03, 0x10]),
            Some("UHCI or OHCI USB host controller (taken from firmware, not driven)")
        );
        assert_eq!(unsupported([0x0c, 0x03, 0x20]), None); // EHCI: native
        assert_eq!(
            unsupported([0x08, 0x05, 0x02]),
            Some("SD host controller without standard SDHCI interface")
        );
        assert!(unsupported([0x0c, 0x05, 0x00]).is_some_and(|why| why.contains("i801")));
        assert_eq!(unsupported([0x06, 0x04, 0x00]), None); // PCI bridge
        assert_eq!(unsupported([0x03, 0x00, 0x00]), None); // display: GPU inventory
    }

    #[test]
    fn hd_audio_beside_a_display_function_is_reported_as_display_audio() {
        let at = |bus, device, function| Address {
            segment: 0,
            bus,
            device,
            function,
        };
        let displays = [at(0x0d, 0, 0)];
        let audio = [0x04, 0x03, 0x00];
        assert_eq!(
            unclaimed(audio, at(0x0d, 0, 1), &displays),
            Some(DISPLAY_AUDIO)
        );
        // A chipset or board HD Audio controller keeps the general reason.
        for address in [at(0x0d, 1, 1), at(0x0e, 0, 1), at(0x0d, 0, 0)] {
            assert_eq!(
                unclaimed(audio, address, &displays),
                Some("HD Audio controller outside the supported identities")
            );
        }
        // Other classes beside a GPU are unaffected.
        assert_eq!(
            unclaimed([0x0c, 0x03, 0x10], at(0x0d, 0, 3), &displays),
            unsupported([0x0c, 0x03, 0x10])
        );
    }

    #[test]
    fn wifi_without_regulatory_data_never_claims_the_adapter() {
        let services = services::tests::Refusing::default();
        let handoff = Handoff {
            ecam: Vec::new(),
            mapped: Vec::new(),
            dma: 0..0,
            clock: crate::Clock {
                ticks_per_microsecond: 1,
            },
            firmware: None,
            completion: None,
        };
        let firmware = handoff_provider(&handoff);
        let network = Network {
            ssid: b"test".to_vec(),
            secret: crate::net::wifi::network::Secret::parse(b"test passphrase").unwrap(),
            country: None,
        };
        let address = Address {
            segment: 0,
            bus: 1,
            device: 0,
            function: 0,
        };
        // SAFETY: the refusing services cannot grant hardware access.
        let result = unsafe {
            intel_wifi(
                &services,
                address,
                None,
                &WifiInputs {
                    clock: handoff.clock,
                    observations: Default::default(),
                    firmware,
                    network: &network,
                },
                &None,
                &[],
            )
        };
        assert!(matches!(
            result,
            Err((
                Error::IntelWifi(net::wifi::intel::attach::AttachError::NoRegulatoryDatabase),
                None
            ))
        ));
        // SAFETY: as above.
        let result = unsafe {
            mediatek_wifi(
                &services,
                address,
                None,
                &WifiInputs {
                    clock: handoff.clock,
                    observations: Default::default(),
                    firmware,
                    network: &network,
                },
                &None,
                &[],
                false,
            )
        };
        assert!(matches!(
            result,
            Err((
                Error::MediatekWifi(net::wifi::mediatek::attach::AttachError::NoRegulatoryDatabase),
                None
            ))
        ));
        assert!(services.requests.borrow().is_empty());
    }

    #[test]
    fn a_board_smbios_does_not_name_attaches_no_broadcom_bluetooth() {
        let services = services::tests::Refusing::default();
        let handoff = Handoff {
            ecam: Vec::new(),
            mapped: Vec::new(),
            dma: 0..0,
            clock: crate::Clock {
                ticks_per_microsecond: 1,
            },
            firmware: None,
            completion: None,
        };
        let address = Address {
            segment: 0,
            bus: 2,
            device: 0,
            function: 0,
        };
        // SAFETY: the refusing services cannot grant hardware access.
        let result = unsafe {
            bcm4377(
                &services,
                address,
                &[],
                handoff_provider(&handoff),
                handoff.clock,
            )
        };
        assert!(matches!(
            result,
            Err((
                Error::Bcm4377(crate::bluetooth::bcm4377::Error::Unsupported(_)),
                None
            ))
        ));
        assert!(services.requests.borrow().is_empty());
    }

    #[test]
    fn probe_diagnostics_preserve_exact_firmware_refusals() {
        let requirement = Requirement {
            name: "rtl_nic/rtl8168e-3.fw",
            target: Target {
                driver: "realtek",
                hardware: "10ec:8168",
                revision: "xid-2c8",
                board: None,
            },
            minimum_bytes: 4,
        };
        let missing = HandoffProvider::new(None).load(requirement).unwrap_err();
        let catalog = Catalog::from_handoff(vec![Resource {
            name: requirement.name,
            target: Target {
                revision: "xid-2c9",
                ..requirement.target
            },
            provenance: Provenance {
                source: "synthetic-fixtures",
                revision: "v1",
                member: "fixtures/abcd.bin",
                license: "CC0-1.0",
            },
            expected_length: 4,
            expected_sha256: Sha256::digest(b"abcd").into(),
            bytes: b"abcd".to_vec(),
        }])
        .unwrap();
        let wrong_target = HandoffProvider::new(Some(&catalog))
            .load(requirement)
            .unwrap_err();
        assert_eq!(wrong_target.kind, firmware::ErrorKind::WrongTarget);
        let address = Address {
            segment: 0,
            bus: 1,
            device: 2,
            function: 0,
        };
        let mut devices = Devices::default();
        for original in [missing, wrong_target] {
            let error = realtek::Error::Firmware(original.clone());
            devices.failures.push((address, error.into()));
            let (location, error) = devices.failures.last().unwrap();
            assert_eq!(*location, address);
            assert!(
                matches!(error, Error::Realtek(realtek::Error::Firmware(kept)) if *kept == original)
            );
            assert_eq!(format!("{error}"), format!("{original}"));
            assert!(format!("{error}").contains(requirement.name));
        }
        let device: Error = BlockError::Device { status: 0x1234 }.into();
        assert_eq!(format!("{device}"), "device status 0x1234");
    }
}
