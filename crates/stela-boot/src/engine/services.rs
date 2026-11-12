//! The host drivers' native services ([`NativeServices`]): configuration
//! space and BAR mappings through the owned MMIO aperture
//! ([`native_mmio::Source`]), DMA pages from the native allocator, DMA
//! domains, and the firmware's ACPI namespace.
//!
//! Remapping units start each discovered requester in the drivers' DMA
//! window or blocked, by what the drivers behind each unit need
//! ([`super::remapping`]), before probe starts devices. A claimed requester
//! receives its own domain. A requester no taken unit routes uses a
//! passthrough domain, and unrestricted DMA is reported once.
use super::aml::HostAcpi;
use crate::{native_memory, native_mmio, output};
use alloc::{boxed::Box, rc::Rc, sync::Arc, vec::Vec};
use core::sync::atomic::{AtomicBool, Ordering};
use stela_boot_core::{passthrough::Passthrough, remapping::initial::Access};
#[cfg(feature = "diagnostics")]
use stela_drivers::probe::services::{LpssI2cObservation, LpssI2cObserver};
use stela_drivers::{
    bus::{
        dma::{
            mapping::{DmaDomain, DmaError},
            memory::PageSource,
        },
        mmio::owned::{self, CacheAttribute, Mapping, PhysicalSpan, Protection},
        pci::{
            Address, Bar, Function, FunctionError,
            owned::OwnedConfig,
            power::Timer,
            requester::{
                ClaimError, FunctionRole, RequesterBinding, StoppedRequester,
                preparation::{AttachmentBackend, DetachError, Root},
            },
        },
        thunderbolt::manager::{Platform, Uuid},
    },
    platform::acpi::resources::{self, DeviceResources, MemoryBroker, Resource, Space},
    probe::{HidQuirk, NativeServices, ServiceError},
};

/// Whether a passthrough domain was handed out, which is reported once.
static PASSTHROUGH_REPORTED: AtomicBool = AtomicBool::new(false);

pub(super) struct Services<'a> {
    mmio: &'a native_mmio::Source,
    remappers: &'a super::remapping::Remappers,
    acpi: Option<&'static HostAcpi>,
    hid_quirks: Vec<HidQuirk<'a>>,
    heap: core::ops::Range<u64>,
    guest_ram: Vec<core::ops::Range<u64>>,
    tsc_hz: u64,
    /// The USB4 devices `--usb4-approve` lists.
    usb4_approved: Vec<Uuid>,
    /// A broker for each register window [`NativeServices::acpi_memory`]
    /// reserved, with the window: one device's `_CRS` memory rounded to
    /// pages, which only that broker suballocates.
    brokers: core::cell::RefCell<alloc::vec::Vec<(PhysicalSpan, MemoryBroker)>>,
}

const PAGE: u64 = 4096;

/// The pages memory descriptor `ordinal` of `device` covers, rounded as
/// [`MemoryBroker::reserve`] rounds it.
fn acpi_pages(device: &DeviceResources<'_>, ordinal: usize) -> Option<PhysicalSpan> {
    let Some(Resource::Address(address)) = device
        .descriptors()
        .get(ordinal)
        .map(|descriptor| &descriptor.resource)
    else {
        return None;
    };
    if address.space != Space::Memory || address.length == 0 {
        return None;
    }
    let start = address.minimum & !(PAGE - 1);
    let end = address
        .minimum
        .checked_add(address.length)?
        .checked_add(PAGE - 1)?
        & !(PAGE - 1);
    PhysicalSpan::new(start, end - start).ok()
}

/// A broker refusal as a service refusal.
fn broker_refusal(error: resources::Error) -> ServiceError {
    ServiceError::Refused(match error {
        resources::Error::Conflict => "ACPI memory overlaps a reservation",
        resources::Error::OutsideReservation => "ACPI memory outside its reserved window",
        resources::Error::Unassigned => "ACPI memory resource is not a fixed consumer range",
        resources::Error::ResourceChanged => "ACPI resources changed",
        _ => "ACPI resources could not be read",
    })
}

/// The i8042's interrupts (ISA IRQ 1 and 12), masked while probe owns the
/// controller through the ACPI namespace.
const I8042_IRQS: [u8; 2] = [1, 12];

#[cfg(feature = "diagnostics")]
fn report_lpss_i2c(observed: LpssI2cObservation<'_>) {
    use stela_drivers::bus::i2c::Address as I2cAddress;

    let identity = u32::from(observed.pci.vendor) | u32::from(observed.pci.device) << 16;
    if !stela_boot_core::firmware_pci::lpss::matches(identity) {
        return;
    }
    let (address, bits) = match observed.address {
        I2cAddress::SevenBit(address) => (u16::from(address), 7),
        I2cAddress::TenBit(address) => (address, 10),
    };
    let irq = if observed.gpio_interrupt {
        "gpio"
    } else {
        "apic-polled"
    };
    output::arguments(format_args!(
        "HOST LPSS-I2C ACPI pci={} id={:04x}:{:04x} hid={} device={} controller={} \
         addr={address:#04x} addr-bits={bits} speed={} irq={irq} descriptor={:#04x}\n",
        observed.pci.address,
        observed.pci.vendor,
        observed.pci.device,
        observed.hid,
        observed.device,
        observed.controller,
        observed.speed,
        observed.descriptor,
    ));
}

impl<'a> Services<'a> {
    pub(super) fn with_hid_quirks(mut self, quirks: Vec<HidQuirk<'a>>) -> Self {
        self.hid_quirks = quirks;
        self
    }

    /// These services' DMA domains also admit the host pages behind guest
    /// RAM, which the guest's GPU work reaches.
    pub(super) fn with_guest_ram(mut self, guest_ram: Vec<core::ops::Range<u64>>) -> Self {
        self.guest_ram = guest_ram;
        self
    }

    /// The RAM device DMA may reach untranslated: the drivers' heap and the
    /// guest's memory.
    fn dma_regions(&self) -> Vec<core::ops::Range<u64>> {
        let mut regions = alloc::vec![self.heap.clone()];
        regions.extend_from_slice(&self.guest_ram);
        regions
    }

    /// Services over `mmio` and `acpi`, whose DMA domains admit the native
    /// allocator's pages at `heap`. The ACPI namespace is offered only once
    /// the i8042's interrupts are masked, as its port grant requires.
    pub(super) fn new(
        mmio: &'a native_mmio::Source,
        remappers: &'a super::remapping::Remappers,
        acpi: Option<&'static HostAcpi>,
        heap: core::ops::Range<u64>,
        tsc_hz: u64,
        usb4_approved: Vec<Uuid>,
    ) -> Self {
        let acpi = acpi.filter(|_| match mmio.require_isa_masked(&I8042_IRQS) {
            Ok(()) => true,
            Err(error) => {
                output::arguments(format_args!(
                    "HOST ACPI devices left out: the i8042 interrupt masks could not be verified ({error:?})\n"
                ));
                false
            }
        });
        Self {
            guest_ram: Vec::new(),
            mmio,
            remappers,
            acpi,
            hid_quirks: Vec::new(),
            heap,
            tsc_hz,
            usb4_approved,
            brokers: core::cell::RefCell::new(alloc::vec::Vec::new()),
        }
    }
}

// SAFETY: cold PCI preparation consumes the checked original configuration
// owner once through the owned aperture and leaves its bus-master seal in
// place until the driver joins stop and domain binding. Each BAR belongs to
// the requesting driver. The ACPI namespace is
// offered only with the i8042's interrupts masked; ring 0 reaches the i8042's
// and each claimed i801's ports, which nothing else here uses. Firmware
// memory is the AML provider's own mapping of RAM the map leaves to firmware.
unsafe impl NativeServices for Services<'_> {
    fn hid_quirks(&self) -> &[HidQuirk<'_>] {
        &self.hid_quirks
    }

    #[cfg(feature = "diagnostics")]
    fn lpss_i2c_observer(&self) -> Option<LpssI2cObserver> {
        Some(report_lpss_i2c)
    }

    fn embedded_controller(
        &self,
    ) -> Result<Option<stela_drivers::platform::ec::capability::Installed<'static>>, ServiceError>
    {
        self.acpi.map_or(Ok(None), HostAcpi::embedded_controller)
    }
    /// The invariant TSC, which every supported host has and reads fastest,
    /// else the HPET, which a boot without an invariant TSC retains.
    fn minimum_delay(&self) -> Result<Rc<dyn Timer>, ServiceError> {
        let invariant = crate::cpu::INVARIANT_TSC.present();
        crate::native_clock::tsc::Minimum::timer(self.tsc_hz, invariant)
            .map_or_else(|| self.mmio.pci_minimum(), Ok)
            .map_err(ServiceError::MinimumDelay)
    }

    fn entropy(&self) -> Result<Arc<dyn stela_drivers::entropy::Entropy>, ServiceError> {
        crate::random::source()
    }

    /// The remapping policy blocks the whole fabric only when no device
    /// needs the drivers' pool, so legacy probe then has nothing to drive or
    /// hand over. Every other requester a legacy driver takes starts in a
    /// window over that pool or untranslated.
    fn legacy_pci_writes(&self) -> bool {
        !self.remappers.has_blocked_fabric()
    }

    fn prepare_requester(
        &self,
        address: Address,
    ) -> Result<stela_drivers::bus::pci::requester::preparation::PreparingRequester, ServiceError>
    {
        use stela_drivers::bus::pci::requester::preparation::PreparingRequester;
        let role = self.remappers.function_role(address);
        if role != FunctionRole::Physical {
            return Err(ServiceError::Claim(ClaimError::UnsupportedRole(role)));
        }
        let prepared = match self.remappers.unremapped_preparation(self.mmio, address) {
            Some(prepared) => prepared.map(|(config, lines)| {
                let root: Box<dyn Root> = Box::new(UnremappedRoot {
                    address,
                    regions: self.dma_regions(),
                    _lines: lines,
                });
                (config, root)
            }),
            None => self.remappers.preparation(self.mmio, address),
        };
        let (config, root) = prepared.map_err(|error| {
            use super::remapping::PreparationError;
            match error {
                PreparationError::Dma(error) => {
                    let why = match self.remappers.initial_access(address) {
                        Access::Window => "its unit starts in the drivers' DMA window for the pool drivers behind it",
                        Access::Untranslated => "no unit routes it, but other units are taken, so the machine's interrupt and DMA state is not proven",
                        Access::Blocked => "its unit's blocked fabric did not qualify",
                    };
                    output::arguments(format_args!("UNSUPPORTED Stela cold requester {address}: original DMA drain unavailable ({error:?}): {why}\n"));
                    ServiceError::Refused("cold requester original DMA drain unavailable")
                }
                PreparationError::Interrupt(error) => {
                    output::arguments(format_args!("UNSUPPORTED Stela cold requester {address}: original interrupt containment unavailable ({error:?})\n"));
                    ServiceError::Mapping(error)
                }
                PreparationError::Configuration(error) => {
                    output::arguments(format_args!("UNSUPPORTED Stela cold requester {address}: original configuration handoff refused ({error})\n"));
                    ServiceError::Refused("cold requester original configuration handoff refused")
                }
                PreparationError::AmdConfiguration => ServiceError::Refused("AMD-Vi original PCI configuration handoff unavailable"),
            }
        })?;
        let function = match Function::new(address, config) {
            Ok(function) => function,
            Err(refusal) => {
                let _root = core::mem::ManuallyDrop::new(root);
                return Err(match refusal.reason {
                    FunctionError::Absent => {
                        ServiceError::Refused("no function answers at this address")
                    }
                    FunctionError::ConfigurationRetry => {
                        ServiceError::Refused("PCI function requests configuration retry")
                    }
                    FunctionError::Access(error) => ServiceError::Config(error),
                });
            }
        };
        // SAFETY: first serialized post-EBS attach through the original owned
        // configuration and selected root. The platform keeps the initial DMA
        // backing, excludes prior firmware/IRQ/work and competing PCI writers,
        // and admits snooped WB DMA. Complete stable inventory supplies role.
        // PCI containment here grants no endpoint engine stop or payload drain;
        // the owning driver must produce that receipt with BARs still kept.
        unsafe { PreparingRequester::new(function, role, root) }.map_err(|rejected| {
            let _root = core::mem::ManuallyDrop::new(rejected.root);
            drop(rejected.function);
            ServiceError::Claim(rejected.error)
        })
    }

    fn claim_stopped(&self, _: Address) -> Result<StoppedRequester, ServiceError> {
        Err(ServiceError::Refused(
            "native endpoint engine stop receipt unavailable",
        ))
    }

    fn bridge(&self, _: Address) -> Result<Function<OwnedConfig>, ServiceError> {
        Err(ServiceError::Refused(
            "native bridge forwarding handoff unavailable",
        ))
    }

    fn map_bar(&self, address: Address, bar: Bar) -> Result<Mapping, ServiceError> {
        let resource = self.bar_resource(address, bar)?;
        let bytes = usize::try_from(resource.physical_span().bytes())
            .map_err(|_| ServiceError::Refused("BAR too large"))?;
        self.mmio
            .authority()
            .map(
                resource,
                0,
                bytes,
                CacheAttribute::Uncached,
                Protection::ReadWrite,
            )
            .map_err(ServiceError::Mapping)
    }

    fn bar_resource(&self, address: Address, bar: Bar) -> Result<owned::Resource, ServiceError> {
        let Bar::Memory {
            address: base,
            size,
            prefetchable,
            ..
        } = bar
        else {
            return Err(ServiceError::Refused("an I/O BAR has no mapping"));
        };
        // SAFETY: the function's driver owns the BAR it asks for.
        unsafe {
            self.mmio
                .claim_bar(address.segment, base, size, prefetchable)
        }
        .map_err(ServiceError::Mapping)
    }

    fn authority(&self) -> Result<owned::Authority, ServiceError> {
        Ok(self.mmio.authority().clone())
    }

    fn domain(&self, binding: RequesterBinding) -> Result<Arc<dyn DmaDomain>, ServiceError> {
        let address = binding.address();
        let binding = match self.remappers.domain(self.mmio, binding) {
            Ok(domain) => {
                return domain.map_err(|error| {
                    output::arguments(format_args!(
                        "HOST IOMMU domain for {address:?} refused: {error:?}\n"
                    ));
                    ServiceError::Refused("the IOMMU refused the requester's domain")
                });
            }
            Err(binding) => binding,
        };
        if !PASSTHROUGH_REPORTED.swap(true, Ordering::Relaxed) {
            output::arguments(format_args!("no IOMMU: device DMA unrestricted\n"));
        }
        Ok(Arc::new(Passthrough::retaining(
            self.dma_regions(),
            binding,
        )))
    }

    fn dma_translated(&self, address: Address) -> bool {
        self.remappers.translates(address)
    }

    fn dma_subtree_translated(&self, address: Address) -> bool {
        self.remappers.translates_subtree(address)
    }

    unsafe fn requesters_arrived(
        &self,
        arrivals: stela_drivers::probe::services::ArrivalInventory<'_>,
    ) -> Result<(), ServiceError> {
        if !self.legacy_pci_writes() {
            return Err(ServiceError::Refused(
                "initial blocked PCI fabric does not admit changed topology",
            ));
        }
        if arrivals
            .functions
            .iter()
            .any(|address| address.segment != arrivals.bridge.segment)
            || !self
                .remappers
                .observe_arrivals(arrivals.bridge, arrivals.functions, arrivals.read32)
                .map_err(ServiceError::Config)?
        {
            return Err(ServiceError::Refused(
                "PCI arrival has no stable complete PF/VF inventory",
            ));
        }
        for &address in arrivals.functions {
            self.remappers.admit(address).map_err(|error| {
                output::arguments(format_args!(
                    "HOST IOMMU requester {address} refused: {error:?}\n"
                ));
                ServiceError::Refused("the IOMMU refused the requester that arrived")
            })?;
        }
        Ok(())
    }

    fn usb4_platform(
        &self,
        location: &str,
        isolated: bool,
    ) -> Result<Box<dyn Platform + Send>, ServiceError> {
        Ok(Box::new(super::usb::Usb4Platform::new(
            location,
            isolated,
            self.usb4_approved.clone(),
            self.tsc_hz,
        )))
    }

    fn pages(&self) -> &'static dyn PageSource {
        &native_memory::SOURCE
    }

    fn interrupts(
        &self,
    ) -> Result<&'static dyn stela_drivers::bus::irq::IrqController, ServiceError> {
        crate::native_irq::service().map_err(|_| ServiceError::Unavailable)
    }

    fn acpi_memory(
        &self,
        device: &DeviceResources<'_>,
        ordinal: usize,
    ) -> Result<owned::Resource, ServiceError> {
        if self.acpi.is_none() {
            return Err(ServiceError::Unavailable);
        }
        let pages = acpi_pages(device, ordinal)
            .ok_or(ServiceError::Refused("no memory resource at that ordinal"))?;
        let mut brokers = self.brokers.borrow_mut();
        let at = match brokers
            .iter()
            .position(|(window, _)| window.start() <= pages.start() && pages.end() <= window.end())
        {
            Some(at) => at,
            None => {
                // SAFETY: probe owns the device whose `_CRS` names the span.
                let root =
                    unsafe { self.mmio.claim_platform(pages) }.map_err(ServiceError::Mapping)?;
                // SAFETY: the broker is the only suballocator of the window,
                // which the aperture reserved outside RAM after EBS.
                let broker =
                    unsafe { MemoryBroker::new(alloc::vec![root]) }.map_err(broker_refusal)?;
                brokers.push((pages, broker));
                brokers.len() - 1
            }
        };
        brokers[at]
            .1
            .reserve(device, ordinal)
            .map(|reservation| reservation.resource().clone())
            .map_err(broker_refusal)
    }

    fn acpi(&self) -> Option<&'static stela_drivers::platform::acpi::Session> {
        self.acpi.map(HostAcpi::session)
    }

    fn firmware_memory(
        &self,
        physical: u64,
        bytes: usize,
    ) -> Result<core::ptr::NonNull<u8>, ServiceError> {
        let acpi = self.acpi.ok_or(ServiceError::Unavailable)?;
        let end = physical
            .checked_add(bytes as u64)
            .filter(|_| bytes != 0)
            .ok_or(ServiceError::Refused(
                "firmware memory range is empty or wraps",
            ))?;
        if !self.mmio.firmware_owned(physical, end) {
            return Err(ServiceError::Refused(
                "firmware memory overlaps RAM the host allocates",
            ));
        }
        acpi.firmware_memory(physical, bytes as u64)
            .ok_or(ServiceError::Refused(
                "firmware memory lies beyond the identity map",
            ))
    }
}

/// A requester's original root on a platform with no remapping hardware.
struct UnremappedRoot {
    address: Address,
    regions: Vec<core::ops::Range<u64>>,
    _lines: native_mmio::InitialPciLines,
}

// SAFETY: with no remapping hardware, no translation path, IOTLB or blocked
// context sits between the endpoint and memory: its writes reach RAM in PCIe
// order, ahead of the completions the receipt producer's register reads
// return, and the parked master seal has held since containment. Binding
// hands out the same untranslated physical view every requester already has.
unsafe impl Root for UnremappedRoot {
    fn drain(&self) -> Result<(), DmaError> {
        Ok(())
    }
    unsafe fn bind(
        &self,
        binding: RequesterBinding,
    ) -> Result<Box<dyn AttachmentBackend>, DmaError> {
        if binding.address() != self.address {
            return Err(DmaError::IommuUnavailable);
        }
        if !PASSTHROUGH_REPORTED.swap(true, Ordering::Relaxed) {
            output::arguments(format_args!("no IOMMU: device DMA unrestricted\n"));
        }
        Ok(Box::new(UnremappedAttachment(Arc::new(
            Passthrough::retaining(self.regions.clone(), binding),
        ))))
    }
}

struct UnremappedAttachment(Arc<Passthrough>);

// SAFETY: the domain is this requester's untranslated physical view, kept
// with its binding, and its messages reach the APIC untranslated. With no root to block and no lease ledger to prove the
// requester unmapped, detach refuses before any effect and keeps them all.
unsafe impl AttachmentBackend for UnremappedAttachment {
    fn domain(&self) -> Arc<dyn DmaDomain> {
        self.0.clone()
    }
    fn irq_owner(
        &mut self,
        _requester: &stela_drivers::bus::pci::requester::RequesterId,
        _config: Option<&stela_drivers::bus::pci::owned::ConfigRetention>,
    ) -> Result<Option<Box<dyn stela_drivers::bus::irq::posted_msi::RouteOwner>>, DmaError> {
        Ok(Some(crate::native_irq::direct::route_owner()))
    }
    unsafe fn detach(&mut self) -> Result<(), DetachError> {
        Err(DetachError::Unexposed(DmaError::IommuUnavailable))
    }
}
