//! Native resource services the mount supplies to [`probe`](super::probe_with_provider)
//! for drivers that attach through owned configuration space, owned register
//! mappings, per-requester DMA domains and the ACPI namespace.
use crate::bus::irq::IrqController;
use crate::bus::{
    dma::{mapping::DmaDomain, memory::PageSource},
    mmio::owned::{self, Mapping},
    pci::{
        Address, Bar, Function,
        owned::OwnedConfig,
        requester::{
            ClaimError, RequesterBinding, StoppedRequester, preparation::PreparingRequester,
        },
    },
    thunderbolt::manager::Platform,
};
use alloc::{boxed::Box, rc::Rc, sync::Arc};
use core::fmt;

/// Complete configuration observations for one adopted PCI subtree.
/// The callback reaches only the kept bridge and the listed functions.
/// Data observations alone confer no reset, DMA or configuration ownership.
pub struct ArrivalInventory<'a> {
    pub bridge: Address,
    pub functions: &'a [Address],
    pub read32:
        &'a mut dyn FnMut(Address, u16) -> Result<Option<u32>, crate::bus::pci::ConfigError>,
}

/// The namespace connection to a matched LPSS function before claiming it.
/// The address and speed come from the ACPI resource. The descriptor register
/// includes any selected board override.
#[cfg(feature = "native-acpi")]
pub struct LpssI2cObservation<'a> {
    pub pci: crate::bus::pci::Identity,
    pub hid: &'a str,
    pub device: &'a str,
    pub controller: &'a str,
    pub address: crate::bus::i2c::Address,
    pub speed: u32,
    pub gpio_interrupt: bool,
    pub descriptor: u16,
}

#[cfg(feature = "native-acpi")]
pub type LpssI2cObserver = for<'a> fn(LpssI2cObservation<'a>);

/// Why a service refused a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServiceError {
    Config(crate::bus::pci::ConfigError),
    PciCapability(crate::bus::pci::CapabilityError),
    MinimumDelay(crate::platform::work::WorkError),
    /// The mount has no such service, such as no ACPI namespace.
    Unavailable,
    /// The mount refuses it, such as a requester its IOMMU cannot isolate.
    Refused(&'static str),
    Mapping(owned::Error),
    Claim(ClaimError),
}

impl fmt::Display for ServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MinimumDelay(error) => write!(f, "minimum delay provider: {error:?}"),
            Self::Unavailable => f.write_str("native resource service unavailable"),
            Self::Refused(why) => f.write_str(why),
            Self::Mapping(error) => write!(f, "native mapping: {error:?}"),
            Self::Config(error) => error.fmt(f),
            Self::PciCapability(error) => write!(f, "PCI capability: {error:?}"),
            Self::Claim(error) => write!(f, "stopped requester: {error:?}"),
        }
    }
}

/// The mount's native owners: configuration and BAR mappings from its native
/// MMIO platform, requester DMA domains from its IOMMU or the passthrough
/// domain ([`DomainKind`](crate::bus::dma::mapping::DomainKind)), kept DMA
/// pages and the ACPI namespace.
///
/// # Safety
/// Implementations transfer ownership: each [`claim_stopped`](Self::claim_stopped)
/// requester is the unique owner of its configuration, with engines, DMA and
/// interrupts stopped and drained, as [`StoppedRequester::from_stopped_pci`]
/// requires. Each bridge is the unique owner of its configuration space,
/// left forwarding. Each mapping covers exactly the requested BAR, uncached and
/// read/write, with no other accessor. Each domain meets the [`DmaDomain`]
/// contract for its kind. Every
/// returned owner stays valid in every access context `probe`'s drivers use.
/// A mount that supplies the ACPI namespace also grants `probe` port I/O to
/// the i8042 at ports 0x60 and 0x64 when the namespace lists one or the FADT
/// does not rule one out (Linux `acpi_parse_fadt`), with IRQ1
/// and IRQ12 masked at every interrupt controller and no other user of
/// those ports, and to the I/O window (BAR 4) of each Intel i801 SMBus host
/// it claims, with no other user than the AML its arbitration hands it to.
/// Each firmware memory mapping is the one AML reaches the same firmware-owned
/// RAM through: writable, live in every such context for the program, and
/// never aliased by a Rust reference.
pub unsafe trait NativeServices {
    /// Whether legacy probe may still use the raw loader ECAM view for
    /// device writes. A mount that has blocked the complete requester
    /// fabric keeps this view for read-only discovery.
    fn legacy_pci_writes(&self) -> bool {
        true
    }

    /// The original clock epoch's checked physical minimum-delay provider.
    /// Acquire before PCI PM/reset effects. Nominal rates or scheduling clocks
    /// cannot implement the hardware minimum; no fallback is supplied here.
    fn minimum_delay(&self) -> Result<Rc<dyn crate::bus::pci::power::Timer>, ServiceError>;
    /// Keep the original configuration and root path for the owning driver's
    /// cold vendor preparation. PCI controls are gated, but no engine stop,
    /// posted-write completion or private domain is granted. The first attach
    /// has no Stela IRQ/work users; the platform excludes prior firmware and
    /// conflicting reset/configuration/routing owners throughout preparation.
    fn prepare_requester(&self, address: Address) -> Result<PreparingRequester, ServiceError>;

    /// Claim the function at `address` as a stopped requester through an owned
    /// mapping of its configuration space.
    fn claim_stopped(&self, address: Address) -> Result<StoppedRequester, ServiceError>;

    /// The PCI-to-PCI bridge at `address` through an owned mapping of its
    /// configuration space, left forwarding, for a driver that sets its
    /// link's upstream end as Linux's ASPM core does through `bus->self`.
    fn bridge(&self, address: Address) -> Result<Function<OwnedConfig>, ServiceError>;

    /// Map memory BAR `bar` of the function at `address` for its driver.
    fn map_bar(&self, address: Address, bar: Bar) -> Result<Mapping, ServiceError>;

    /// The mount's reservation of memory BAR `bar` of the function at
    /// `address`, for drivers that map parts of it or route child resources
    /// inside it, as Intel VMD does.
    fn bar_resource(&self, address: Address, bar: Bar) -> Result<owned::Resource, ServiceError>;

    /// The mapping permission that issues [`Self::map_bar`]'s mappings.
    fn authority(&self) -> Result<owned::Authority, ServiceError>;

    /// The DMA domain for the requester `binding` names. The domain keeps the
    /// binding, and with it the configuration epoch it keeps.
    fn domain(&self, binding: RequesterBinding) -> Result<Arc<dyn DmaDomain>, ServiceError>;

    /// Whether the mount's IOMMU confines the DMA of the function at
    /// `address`.
    fn dma_translated(&self, _address: Address) -> bool {
        false
    }

    /// Whether every possible future child within this bridge's reserved bus
    /// range is confined by an active remapper. The bridge's own requester
    /// translation is insufficient: firmware can route a child to another unit.
    fn dma_subtree_translated(&self, _bridge: Address) -> bool {
        false
    }

    /// Record the complete adopted subtree before any new native claim, then
    /// admit each requester through its firmware-selected remapping unit.
    ///
    /// # Safety
    /// The caller owns the original bridge, the complete adopted subtree and
    /// their configuration/reset/routing epoch throughout this call. The
    /// callback reads the same kept configuration resources, including the
    /// bridge's updated bus numbers. No driver has claimed a listed function.
    /// Previous PF/VF assignments remain unchanged; every new PF and enabled
    /// VF is included. No PCI, SR-IOV or routing writer runs during observation
    /// and subsequent native claims. The observations are not an ownership grant.
    unsafe fn requesters_arrived(
        &self,
        _arrivals: ArrivalInventory<'_>,
    ) -> Result<(), ServiceError> {
        Ok(())
    }

    /// What the software connection manager of the USB4 host at `location`
    /// needs of the mount. `isolated` requires [`Self::dma_translated`] for
    /// the host and [`Self::dma_subtree_translated`] for every hot-plug bridge
    /// its tunnels can come up on.
    fn usb4_platform(
        &self,
        _location: &str,
        _isolated: bool,
    ) -> Result<Box<dyn Platform + Send>, ServiceError> {
        Err(ServiceError::Unavailable)
    }

    /// Pages for driver DMA rings and buffers, for the rest of the boot.
    fn pages(&self) -> &'static dyn PageSource;

    /// An owned cryptographic source for authentication nonces. The handle
    /// outlives probe; a platform without trusted entropy reports unavailable.
    fn entropy(&self) -> Result<Arc<dyn crate::entropy::Entropy>, ServiceError> {
        Err(ServiceError::Unavailable)
    }

    /// The interrupt controller that routes MSIs to the probing CPU, for
    /// drivers that need a delivered interrupt, such as Intel's ISH.
    fn interrupts(&self) -> Result<&'static dyn IrqController, ServiceError>;

    /// The first EC installed and connected during boot's staged ACPI
    /// initialization. Probe keeps the receipt instead of installing a
    /// second operation-region handler.
    #[cfg(feature = "native-acpi")]
    fn embedded_controller(
        &self,
    ) -> Result<Option<crate::platform::ec::capability::Installed<'static>>, ServiceError> {
        Ok(None)
    }

    /// The ACPI namespace, for devices described there rather than on PCI:
    /// i8042 keyboards and mice, embedded controllers, I²C-HID devices and
    /// batteries ([`platform::acpi::Session::devices`](crate::platform::acpi::Session::devices)
    /// matches `_HID`/`_CID` and filters by `_STA`; `_CRS` decodes through
    /// [`platform::acpi::resources`](crate::platform::acpi::resources)).
    ///
    /// The session is the mount's one ACPICA instance, alive for the
    /// program. Its provider serves the ports, decode, power and global lock
    /// of the first embedded controller the ECDT or namespace names, and no
    /// other owner reaches that controller, as
    /// [`Binding::install`](crate::platform::ec::binding::Binding::install)
    /// requires.
    #[cfg(feature = "native-acpi")]
    fn acpi(&self) -> Option<&'static crate::platform::acpi::Session>;

    /// Validated I²C-HID rules for this host. Probe borrows them only while
    /// it attaches the ACPI devices.
    #[cfg(feature = "native-acpi")]
    fn hid_quirks(&self) -> &[super::HidQuirk<'_>] {
        &[]
    }

    /// An optional reporter for ACPI and PCI observations. The callback
    /// receives facts only and cannot authorize requester or BAR ownership.
    #[cfg(feature = "native-acpi")]
    fn lpss_i2c_observer(&self) -> Option<LpssI2cObserver> {
        None
    }

    /// Reserve memory resource `ordinal` of `device`'s `_CRS` from the
    /// mount's native broker, for a controller the namespace describes, as
    /// [`MemoryBroker::reserve`](crate::platform::acpi::resources::MemoryBroker::reserve)
    /// reserves it.
    #[cfg(feature = "native-acpi")]
    fn acpi_memory(
        &self,
        device: &crate::platform::acpi::resources::DeviceResources<'_>,
        ordinal: usize,
    ) -> Result<owned::Resource, ServiceError>;

    /// The mapping AML's SystemMemory regions reach `bytes` of firmware RAM
    /// at `physical` through, for a mailbox the firmware shares with its
    /// methods, such as a UCSI `_CRS`, which Linux `ucsi_acpi_probe` maps
    /// write-back. It lives for the program, as the namespace does; the
    /// mount refuses a range that overlaps RAM it allocates.
    #[cfg(feature = "native-acpi")]
    fn firmware_memory(
        &self,
        physical: u64,
        bytes: usize,
    ) -> Result<core::ptr::NonNull<u8>, ServiceError>;
}

/// ACPI hardware IDs of the classes attached through [`NativeServices`].
pub mod ids {
    /// i8042 keyboards: IBM Enhanced and the compatible IDs Linux's
    /// `i8042_pnp_kbd_devids` lists.
    pub const I8042_KEYBOARD: &[&str] = &[
        "PNP0300", "PNP0301", "PNP0302", "PNP0303", "PNP0304", "PNP0305", "PNP0306", "PNP0309",
        "PNP030A", "PNP030B", "PNP0320", "PNP0343", "PNP0344", "PNP0345", "CPQA0D7",
    ];
    /// i8042 auxiliary (PS/2 mouse) ports, as Linux's `i8042_pnp_aux_devids`.
    pub const I8042_MOUSE: &[&str] = &[
        "AUI0200", "FJC6000", "FJC6001", "PNP0F03", "PNP0F0B", "PNP0F0E", "PNP0F12", "PNP0F13",
        "PNP0F19", "PNP0F1C", "SYN0801",
    ];
    /// ACPI embedded controller.
    pub const EMBEDDED_CONTROLLER: &[&str] = &["PNP0C09"];
    /// HID over I²C, by its `_CID`.
    pub const I2C_HID: &[&str] = &["PNP0C50", "ACPI0C50"];
    /// Control-method battery.
    pub const BATTERY: &[&str] = &["PNP0C0A"];
    /// Control-method lid.
    pub const LID: &[&str] = &["PNP0C0D"];
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::bus::{
        dma::memory::{MemoryError, PinnedPages},
        mmio::owned::CacheAttribute,
    };
    use alloc::vec::Vec;
    use core::cell::RefCell;

    /// Services that record every request and refuse them all, as a mount
    /// without native owners would.
    #[derive(Default)]
    pub(crate) struct Refusing {
        pub(crate) requests: RefCell<Vec<Address>>,
        pub(crate) preparations: RefCell<Vec<Address>>,
        pub(crate) stopped: RefCell<Vec<Address>>,
        model_minimum: bool,
    }
    impl Refusing {
        /// Isolated timing-only fixture. All physical grants still refuse.
        pub(crate) fn minimum_only() -> Self {
            Self {
                model_minimum: true,
                ..Self::default()
            }
        }
    }

    struct NoPages;
    impl PageSource for NoPages {
        fn allocate(
            &self,
            _: usize,
            _: usize,
            _: CacheAttribute,
        ) -> Result<PinnedPages, MemoryError> {
            Err(MemoryError::Unavailable)
        }
    }

    // SAFETY: every physical request is refused; no config, BAR, DMA or IRQ
    // owner is transferred. The explicit private timing fixture cannot reach
    // hardware because no physical service can succeed.
    unsafe impl NativeServices for Refusing {
        fn minimum_delay(&self) -> Result<Rc<dyn crate::bus::pci::power::Timer>, ServiceError> {
            if self.model_minimum {
                Ok(crate::minimum_model::owned(crate::Clock {
                    ticks_per_microsecond: 1000,
                }))
            } else {
                Err(ServiceError::Unavailable)
            }
        }
        fn prepare_requester(&self, address: Address) -> Result<PreparingRequester, ServiceError> {
            self.requests.borrow_mut().push(address);
            self.preparations.borrow_mut().push(address);
            Err(ServiceError::Unavailable)
        }
        fn claim_stopped(&self, address: Address) -> Result<StoppedRequester, ServiceError> {
            self.requests.borrow_mut().push(address);
            self.stopped.borrow_mut().push(address);
            Err(ServiceError::Unavailable)
        }
        fn bridge(&self, address: Address) -> Result<Function<OwnedConfig>, ServiceError> {
            self.requests.borrow_mut().push(address);
            Err(ServiceError::Unavailable)
        }
        fn map_bar(&self, address: Address, _: Bar) -> Result<Mapping, ServiceError> {
            self.requests.borrow_mut().push(address);
            Err(ServiceError::Unavailable)
        }
        fn bar_resource(&self, address: Address, _: Bar) -> Result<owned::Resource, ServiceError> {
            self.requests.borrow_mut().push(address);
            Err(ServiceError::Unavailable)
        }
        fn authority(&self) -> Result<owned::Authority, ServiceError> {
            Err(ServiceError::Unavailable)
        }
        fn domain(&self, _: RequesterBinding) -> Result<Arc<dyn DmaDomain>, ServiceError> {
            Err(ServiceError::Refused("requester shares an IOMMU group"))
        }
        fn interrupts(&self) -> Result<&'static dyn IrqController, ServiceError> {
            Err(ServiceError::Unavailable)
        }
        fn pages(&self) -> &'static dyn PageSource {
            &NoPages
        }
        #[cfg(feature = "native-acpi")]
        fn acpi_memory(
            &self,
            _: &crate::platform::acpi::resources::DeviceResources<'_>,
            _: usize,
        ) -> Result<owned::Resource, ServiceError> {
            Err(ServiceError::Unavailable)
        }
        #[cfg(feature = "native-acpi")]
        fn acpi(&self) -> Option<&'static crate::platform::acpi::Session> {
            None
        }
        #[cfg(feature = "native-acpi")]
        fn firmware_memory(
            &self,
            _: u64,
            _: usize,
        ) -> Result<core::ptr::NonNull<u8>, ServiceError> {
            Err(ServiceError::Unavailable)
        }
    }

    #[test]
    fn refusals_name_their_reason() {
        let services = Refusing::default();
        let address = Address {
            segment: 0,
            bus: 1,
            device: 0,
            function: 0,
        };
        assert_eq!(
            services.claim_stopped(address).err(),
            Some(ServiceError::Unavailable)
        );
        assert_eq!(*services.requests.borrow(), [address]);
        assert!(matches!(services.entropy(), Err(ServiceError::Unavailable)));

        assert_eq!(
            alloc::format!(
                "{}",
                ServiceError::Refused("requester shares an IOMMU group")
            ),
            "requester shares an IOMMU group"
        );
    }

    #[test]
    fn hardware_ids_are_unique_and_well_formed() {
        let all: Vec<&str> = [
            ids::I8042_KEYBOARD,
            ids::I8042_MOUSE,
            ids::EMBEDDED_CONTROLLER,
            ids::I2C_HID,
            ids::BATTERY,
            ids::LID,
        ]
        .concat();
        for (index, id) in all.iter().enumerate() {
            assert!(matches!(id.len(), 7 | 8), "{id}");
            assert!(
                id.chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
            );
            assert!(!all[index + 1..].contains(id), "{id}");
        }
    }
}
