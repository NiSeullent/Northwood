//! USB4/Thunderbolt hosts, attached as Linux's `nhi_probe` and `nhi_select_cm`
//! attach them when ACPI grants the OS no native USB4 control, which this
//! mount never requests. A host whose firmware runs the connection manager
//! (ICM) keeps it; as Linux does with no authorizing agent, a device that
//! firmware leaves awaiting approval gets no PCIe tunnel. Any other host gets
//! the software connection manager ([`Manager`]).
//!
//! Software hosts come up before the PCI scan ([`bring_up`]), so the scan
//! finds what they tunnel as it finds any other function, and a USB device
//! behind a USB3 tunnel is there when the xHCI controllers attach.
use super::{Bridge, Failure, Kind, NativeServices, ServiceError, map};
use crate::{
    Clock,
    bus::{
        pci::{Address, ConfigSpace, EcamWindow, Function, hotplug, regs},
        thunderbolt::{
            channel::{self, Channel},
            cold::{self, MappedRegisters},
            controller::{self, Controller},
            icm,
            manager::Manager,
            ring,
        },
    },
    probe::{Devices, Error, Handoff, Warning, registry},
};
use alloc::{boxed::Box, format, string::String, vec::Vec};
use core::{fmt, mem::ManuallyDrop};
mod ownership;
mod transport;
use ownership::Owners;
pub use ownership::{Error as RetirementError, Failure as RetirementFailure, RetiredTransport};

/// How long a domain's routers stay silent before it counts as found, and
/// the longest the finding takes.
const QUIET_MS: u64 = 700;
const SETTLE_MS: u64 = 3000;
/// How long the PCIe links of new tunnels get to train and their functions
/// to answer.
const LINK_MS: u64 = 1000;
/// The wait after link training before configuration requests
/// (PCI Express Base 6.0 §6.6.1).
const LINK_RECOVERY_MS: u64 = 100;
/// Between looks at the hosts.
const POLL_US: u64 = 1000;

/// What runs a host's domain.
pub enum Connection {
    /// Its firmware: the ICM transport and the firmware's protocol
    /// parameters, with what arrived before DRIVER_READY was answered.
    Firmware {
        icm: Box<icm::Host<MappedRegisters>>,
        ready: icm::DriverReady,
        early: Vec<icm::Event>,
    },
    /// This mount.
    Software { manager: Manager },
}

/// An attached host, with the claimed function and BAR0 mapping it runs on.
pub struct Host {
    pub location: String,
    connection: ManuallyDrop<Box<Connection>>,
    _owners: ManuallyDrop<Box<Owners>>,
}

impl Host {
    /// Borrow the active protocol while its actual original native owner stays
    /// retained. Abandoned Host Drop never destroys the CM/router future.
    pub fn connection(&self) -> &Connection {
        &self.connection
    }
    /// Borrow protocol and diagnostic location together without splitting any
    /// actual native owner or creating a second CM/transport capability.
    pub fn protocol(&mut self) -> (&str, &mut Connection) {
        (&self.location, &mut self.connection)
    }
    pub fn connection_mut(&mut self) -> &mut Connection {
        &mut self.connection
    }
}
/// Why a host was not attached.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The `IMR_VALID` property says the firmware image is invalid, which
    /// `nhi_imr_valid` refuses.
    InvalidFirmwareImage,
    Clock,
    Minimum(crate::bus::pci::power::Error),
    Config(crate::bus::pci::ConfigError),
    Ring(ring::Error),
    Controller(controller::Error),
    Cold(cold::Error),
    Icm(icm::Error),
    Channel(channel::Error),
    Platform(ServiceError),
    /// What arrived below a hot-plug bridge has no place.
    Arrival(hotplug::Error),
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFirmwareImage => f.write_str("firmware image not valid"),
            Self::Clock => f.write_str("uncalibrated clock"),
            Self::Config(error) => error.fmt(f),
            Self::Minimum(error) => write!(f, "USB4 minimum delay: {error:?}"),
            Self::Ring(error) => write!(f, "NHI ring: {error:?}"),
            Self::Controller(error) => write!(f, "NHI preparation: {error:?}"),
            Self::Cold(error) => write!(f, "NHI cold ownership: {error:?}"),
            Self::Icm(error) => write!(f, "ICM: {error:?}"),
            Self::Channel(error) => write!(f, "control channel: {error:?}"),
            Self::Platform(error) => write!(f, "connection manager: {error}"),
            Self::Arrival(hotplug::Error::Config(error)) => error.fmt(f),
            Self::Arrival(hotplug::Error::PciCapability(error)) => {
                write!(f, "tunnelled PCIe capability: {error:?}")
            }
            Self::Arrival(hotplug::Error::Buses) => {
                f.write_str("tunnelled PCIe device: the bridge has no bus numbers left for it")
            }
            Self::Arrival(hotplug::Error::Windows) => {
                f.write_str("tunnelled PCIe device: the bridge's memory windows cannot hold it")
            }
        }
    }
}

/// Whether the function's ACPI companion says its firmware image is
/// invalid, as `nhi_imr_valid` reads `IMR_VALID`: absent means valid.
#[cfg(feature = "native-acpi")]
fn invalid_image(services: &dyn NativeServices, address: Address, bridges: &[Bridge]) -> bool {
    use crate::platform::acpi::Value;
    services
        .acpi()
        .and_then(|session| super::companion(session, address, bridges))
        .is_some_and(|companion| {
            matches!(companion.property("IMR_VALID"), Ok(Some(Value::Integer(0))))
        })
}

#[cfg(not(feature = "native-acpi"))]
fn invalid_image(_: &dyn NativeServices, _: Address, _: &[Bridge]) -> bool {
    false
}

/// Attach the USB4 host at `address`: prepare its NHI as `nhi_probe` does,
/// enable its bus mastering, then start its connection manager. A firmware
/// one has answered DRIVER_READY on return; a software one is yet to run.
/// `isolated` is [`NativeServices::usb4_platform`]'s.
///
/// # Safety
/// [`NativeServices`]' contract holds for `services`, `clock` is the
/// handoff's calibrated clock, and `bridges` lists the segment's bridges.
#[expect(
    clippy::result_large_err,
    reason = "a failed attach hands back the owners that must outlive the device"
)]
pub(in crate::probe) unsafe fn attach(
    services: &dyn NativeServices,
    address: Address,
    bridges: &[Bridge],
    clock: Clock,
    isolated: bool,
) -> Result<Host, Failure> {
    let minimum = services
        .minimum_delay()
        .map_err(|error| (Error::Native(error), None))?;
    let refused =
        |refusal: Refusal, kept: Option<Box<dyn core::any::Any>>| (Error::Usb4(refusal), kept);
    if invalid_image(services, address, bridges) {
        return Err(refused(Refusal::InvalidFirmwareImage, None));
    }
    if clock.check_duration(60_000_000).is_err() {
        return Err(refused(Refusal::Clock, None));
    }
    // Keep the actual original preparation. Config containment alone never
    // supplies the old engine/root/IRQ receipt or a private domain.
    let mut requester = ManuallyDrop::new(
        services
            .prepare_requester(address)
            .map_err(|error| (Error::Native(error), None))?,
    );
    let function = requester.function_mut();
    if function.address != address {
        return Err(refused(
            Refusal::Cold(cold::Error::Identity),
            Some(Box::new((requester, minimum))),
        ));
    }
    let family = icm::Family::of(function.vendor, function.device);
    let location = format!("{address}");
    if let Err(error) = cold::check_flr(function, minimum.as_ref()) {
        return Err(refused(
            Refusal::Cold(error),
            Some(Box::new((requester, minimum))),
        ));
    }
    let bars = match function.bars() {
        Ok(bars) => bars,
        Err(error) => return Err((error.into(), Some(Box::new((requester, minimum))))),
    };
    let mapping = match map(services, address, &bars, 0) {
        Ok(mapping) => mapping,
        Err(error) => return Err((Error::Native(error), Some(Box::new((requester, minimum))))),
    };
    let command = match function.config().read16(regs::COMMAND) {
        Ok(command) => command,
        Err(error) => {
            return Err((
                error.into(),
                Some(Box::new((requester, ManuallyDrop::new(mapping), minimum))),
            ));
        }
    };
    if command == u16::MAX {
        return Err(refused(
            Refusal::Cold(cold::Error::Identity),
            Some(Box::new((requester, ManuallyDrop::new(mapping), minimum))),
        ));
    }
    // Memory decode only, with Bus Master still contained in the original
    // global/configuration custody. No ring data allocation/publication exists.
    if let Err(error) = function
        .config()
        .write16(regs::COMMAND, command | regs::COMMAND_MEMORY)
    {
        return Err((
            error.into(),
            Some(Box::new((requester, ManuallyDrop::new(mapping), minimum))),
        ));
    }
    let inputs = cold::Inputs {
        requester: ManuallyDrop::into_inner(requester),
        mapping,
    };
    // SAFETY: actual first original requester and its same owned whole BAR0;
    // platform retains original/consumed global DMA custody, excluded firmware
    // IRQ/work/config owners, D0/shared power, no Stela CM/ring/IRQ publisher.
    // This cold reset is function-local; HRR is not used as inherited stop.
    let cold = unsafe { cold::Cold::new(inputs) };
    let ready = match cold.quiesce_and_bind(minimum.as_ref()) {
        Ok(ready) => ready,
        Err(failed) => {
            return Err(refused(
                Refusal::Cold(failed.error.clone()),
                Some(Box::new((failed, minimum))),
            ));
        }
    };
    // SAFETY: immediate paired transfer into one retained native owner graph,
    // before any bus-master/ring/CM publication. Actual Attachment is kept.
    let (function, attachment, registers) = unsafe { ready.into_parts() };
    let mut owners = ManuallyDrop::new(Box::new(Owners {
        function,
        attachment,
        registers,
        software: None,
        minimum,
    }));
    let now = || {
        clock
            .try_now_us()
            .map(|us| us / 1000)
            .map_err(|_| Refusal::Clock)
    };
    let started = (|| {
        let owners = &mut **owners;
        let config = owners.function.config();
        let command = config.read16(regs::COMMAND).map_err(Refusal::Config)?;
        if command & regs::COMMAND_MEMORY == 0 {
            config
                .write16(regs::COMMAND, command | regs::COMMAND_MEMORY)
                .map_err(Refusal::Config)?;
        }
        {
            // SAFETY: the configuration and registers are the claimed
            // function's and its BAR0, and the claimed requester's DMA and
            // interrupts are stopped.
            let mut controller = unsafe {
                Controller::new_with_router_reset(
                    &mut *config,
                    owners.registers.clone(),
                    controller::RouterReset::Preserve,
                    owners.minimum.clone(),
                )
            }
            .map_err(Refusal::Controller)?;
            while controller
                .poll(now()?)
                .map_err(Refusal::Controller)?
                .is_none()
            {
                core::hint::spin_loop();
            }
        }
        // `nhi_probe`'s pci_set_master, now that the domain is attached.
        let command = config.read16(regs::COMMAND).map_err(Refusal::Config)?;
        config
            .write16(regs::COMMAND, command | regs::COMMAND_MASTER)
            .map_err(Refusal::Config)?;
        if config.read16(regs::COMMAND).map_err(Refusal::Config)?
            != (command | regs::COMMAND_MASTER)
        {
            return Err(Refusal::Cold(cold::Error::Registers));
        }
        if family.is_none() {
            let platform = services
                .usb4_platform(&location, isolated)
                .map_err(Refusal::Platform)?;
            // SAFETY: as above; the domain is the claimed requester's own,
            // the prepared NHI has its interrupts masked and rings idle, and
            // no firmware runs a connection manager on it.
            let mut channel = unsafe {
                Channel::new(
                    owners.registers.clone(),
                    services.pages(),
                    owners.attachment.domain(),
                    u64::MAX,
                )
            }
            .map_err(Refusal::Channel)?;
            channel.start(now()?).map_err(Refusal::Channel)?;
            let shared = transport::SharedChannel::new(channel);
            owners.software = Some(shared.clone());
            let manager = Manager::new(Box::new(transport::Sender(shared)), platform);
            return Ok(Connection::Software { manager });
        }
        // SAFETY: as above; this mount negotiates no native USB4 control, so
        // its firmware's ICM is the connection manager.
        let mut host = unsafe {
            icm::Host::new(
                &*config,
                owners.registers.clone(),
                services.pages(),
                owners.attachment.domain(),
                u64::MAX,
            )
        }
        .map_err(Refusal::Icm)?;
        host.start(now()?).map_err(Refusal::Icm)?;
        let request = host.driver_ready(now()?).map_err(Refusal::Icm)?;
        let mut early = Vec::new();
        loop {
            match host.poll(now()?).map_err(Refusal::Icm)? {
                Some(icm::Event::DriverReady {
                    request: answered,
                    parameters,
                }) if answered == request => {
                    return Ok(Connection::Firmware {
                        icm: Box::new(host),
                        ready: parameters,
                        early,
                    });
                }
                Some(event) => early.push(event),
                None => core::hint::spin_loop(),
            }
        }
    })();
    match started {
        Ok(connection) => Ok(Host {
            location,
            connection: ManuallyDrop::new(Box::new(connection)),
            _owners: owners,
        }),
        Err(refusal) => Err(refused(refusal, Some(Box::new(owners)))),
    }
}

/// A hot-plug bridge, and how to reach what hangs from it.
struct Slot {
    window: EcamWindow,
    address: Address,
}

impl Slot {
    /// Whether something has arrived below it for [`hotplug::adopt`].
    ///
    /// # Safety
    /// As for [`bring_up`].
    unsafe fn arrived(&self) -> Result<bool, hotplug::Error> {
        let Address {
            bus,
            device,
            function,
            ..
        } = self.address;
        // SAFETY: forwarded; the window is mapped, and only probe reaches it.
        let config = unsafe { self.window.config(bus, device, function) };
        // SAFETY: as above.
        hotplug::arrived(&config, |bus, device, function| unsafe {
            self.window.config(bus, device, function)
        })
    }
}

/// What a look over the mapped windows found.
#[derive(Default)]
struct Survey {
    bridges: Vec<Bridge>,
    slots: Vec<Slot>,
    hosts: Vec<Address>,
}

/// # Safety
/// As for [`bring_up`].
#[expect(
    clippy::result_large_err,
    reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
)]
unsafe fn survey(
    handoff: &Handoff,
    policy: &registry::DriverPolicy<'_>,
    devices: &mut Devices,
) -> Survey {
    let mut found = Survey::default();
    for window in &handoff.ecam {
        if !handoff.is_mapped(&window.range()) {
            continue;
        }
        // SAFETY: mapped as checked; the caller owns the machine.
        let functions = match unsafe { Function::enumerate(window) } {
            Ok(functions) => functions,
            Err(failure) => {
                devices
                    .failures
                    .push((failure.address(), Error::Enumeration(failure.error)));
                devices.owners.push(Box::new(failure));
                continue;
            }
        };
        for mut function in functions {
            let address = function.address;
            if function.class[..2] == [6, 4] {
                let checked = (|| -> Result<_, Error> {
                    let bridge = Bridge {
                        address,
                        secondary: function.config().read8(0x19)?,
                        aspm: crate::bus::pci::aspm_control(&function)?,
                    };
                    let hotplug = hotplug::hot_plug_capable(&*function.config())
                        .map_err(|error| Error::Usb4(Refusal::Arrival(error)))?;
                    Ok((bridge, hotplug))
                })();
                match checked {
                    Ok((bridge, hotplug)) => {
                        found.bridges.push(bridge);
                        if hotplug {
                            found.slots.push(Slot {
                                window: *window,
                                address,
                            });
                        }
                    }
                    Err(error) => {
                        devices.failures.push((address, error));
                        devices.owners.push(Box::new(function));
                        continue;
                    }
                }
            }
            if matches!(
                registry::select_pci(&function.identity(), policy).driver(),
                Some(registry::Driver::Native(Kind::Usb4))
            ) && icm::Family::of(function.vendor, function.device).is_none()
            {
                found.hosts.push(address);
            }
        }
    }
    found
}

/// Poll the software connection managers for up to `ms`, until `done` holds.
/// `done` hears whether any of them has work in progress.
fn pump(hosts: &mut [Host], clock: Clock, ms: u64, mut done: impl FnMut(bool) -> bool) {
    let Ok(start) = clock.try_now_us() else {
        return;
    };
    loop {
        let mut busy = false;
        for host in hosts.iter_mut() {
            if let Connection::Software { manager } = host.connection_mut() {
                busy |= manager.poll() && !manager.idle();
            }
        }
        let Ok(now) = clock.try_now_us() else {
            return;
        };
        if done(busy) || now.saturating_sub(start) >= ms * 1000 || clock.try_delay(POLL_US).is_err()
        {
            return;
        }
    }
}

/// Start the USB4 hosts no firmware runs a connection manager on, let their
/// domains come up, and number and place the PCIe functions they tunnel. The
/// addresses returned are the hosts dealt with here, attached or not.
///
/// PCIe tunnels of later plug events keep their functions unused: there is no
/// scan to find them, as Linux's pciehp would.
///
/// # Safety
/// [`NativeServices`]' contract holds for `services`. The caller owns the
/// machine, and no driver has started.
pub(in crate::probe) unsafe fn bring_up(
    services: &dyn NativeServices,
    handoff: &Handoff,
    policy: &registry::DriverPolicy<'_>,
    devices: &mut Devices,
) -> Vec<Address> {
    // SAFETY: forwarded.
    let Survey {
        bridges,
        slots,
        hosts,
    } = unsafe { survey(handoff, policy, devices) };
    let clock = handoff.clock;
    let isolated = hosts
        .iter()
        .all(|address| services.dma_translated(*address))
        && slots
            .iter()
            .all(|slot| services.dma_subtree_translated(slot.address));
    for &address in &hosts {
        // SAFETY: as for tg3 in probe; the look found every bridge.
        match unsafe { attach(services, address, &bridges, clock, isolated) } {
            Ok(host) => devices.usb4_hosts.push(host),
            Err((error, kept)) => {
                devices.failures.push((address, error));
                devices.owners.extend(kept);
            }
        }
    }
    let attached = &mut devices.usb4_hosts;
    let Ok(start) = clock.try_now_us() else {
        return hosts;
    };
    let mut quiet_since = start;
    pump(attached, clock, SETTLE_MS, |busy| {
        let now = clock.try_now_us().unwrap_or(start);
        if busy {
            quiet_since = now;
        }
        now.saturating_sub(quiet_since) >= QUIET_MS * 1000
    });
    let tunnels: usize = attached
        .iter()
        .map(|host| match host.connection() {
            Connection::Software { manager } => manager.pcie_tunnels(),
            Connection::Firmware { .. } => 0,
        })
        .sum();
    if tunnels == 0 {
        return hosts;
    }
    let (adopting, failed_arrivals) = {
        let mut failed = Vec::new();
        let mut failures = Vec::new();
        let ready = {
            let mut waiting = || {
                let mut ready = Vec::new();
                for (index, slot) in slots.iter().enumerate() {
                    if failed.contains(&index) {
                        continue;
                    }
                    // SAFETY: original mapped slot remains probe-owned.
                    match unsafe { slot.arrived() } {
                        Ok(true) => ready.push(index),
                        Ok(false) => {}
                        Err(error) => {
                            failed.push(index);
                            failures.push((slot.address, error));
                        }
                    }
                }
                ready
            };
            pump(attached, clock, LINK_MS, |_| waiting().len() >= tunnels);
            // PCIe requires this physical recovery interval before the next
            // configuration request. Nominal polling timeout data is not a
            // lower-bound timing capability. Keep all actual owners on failure
            // and remove every protocol from future live service.
            let host = attached
                .first()
                .expect("a live software tunnel has its host");
            let address = host._owners.function.address;
            let minimum = host._owners.minimum.clone();
            if let Err(error) = minimum.wait_us(LINK_RECOVERY_MS * 1000) {
                devices
                    .failures
                    .push((address, Error::Usb4(Refusal::Minimum(error))));
                devices.owners.extend(
                    attached
                        .drain(..)
                        .map(|host| Box::new(host) as Box<dyn core::any::Any>),
                );
                return hosts;
            }
            waiting()
        };
        (ready, failures)
    };
    for (address, error) in failed_arrivals {
        devices
            .failures
            .push((address, Error::Usb4(Refusal::Arrival(error))));
    }
    'adoption: for index in adopting {
        let slot = &slots[index];
        let bridge = slot.address;
        // SAFETY: forwarded; the functions below a bridge are still probe's.
        let adopted = hotplug::adopt(
            bridge.segment,
            (bridge.bus, bridge.device, bridge.function),
            |bus, device, function| unsafe { slot.window.config(bus, device, function) },
        );
        let arrivals = match adopted {
            Ok(arrivals) => arrivals,
            Err(error) => {
                devices
                    .failures
                    .push((bridge, Error::Usb4(Refusal::Arrival(error))));
                continue;
            }
        };
        // Observe every present RID in the owned reserved subtree, including
        // functions adoption cannot place (such as ARI functions above 7).
        // Omitting such a PF could misclassify another function as physical.
        // SAFETY: this is the kept original slot bridge, still exclusively owned.
        let configuration = unsafe {
            slot.window
                .config(bridge.bus, bridge.device, bridge.function)
        };
        let buses = (|| -> Result<_, crate::bus::pci::ConfigError> {
            Ok((configuration.read8(0x19)?, configuration.read8(0x1a)?))
        })();
        let (first, last) = match buses {
            Ok(buses) => buses,
            Err(error) => {
                devices.failures.push((bridge, error.into()));
                devices.owners.push(Box::new(configuration));
                continue;
            }
        };
        if first <= bridge.bus
            || last < first
            || first < slot.window.start_bus
            || last > slot.window.end_bus
        {
            devices.failures.push((
                bridge,
                Error::Native(ServiceError::Refused(
                    "PCI arrival has no owned downstream bus range",
                )),
            ));
            continue;
        }
        let mut functions = Vec::new();
        for bus in first..=last {
            for number in 0..=255u16 {
                let address = Address {
                    segment: bridge.segment,
                    bus,
                    device: (number >> 3) as u8,
                    function: (number & 7) as u8,
                };
                // SAFETY: the original slot owns this complete reserved range;
                // no child driver has started or configuration writer overlaps.
                let configuration =
                    unsafe { slot.window.config(bus, address.device, address.function) };
                match configuration.read16(0) {
                    Ok(u16::MAX) => {}
                    Ok(_) => functions.push(address),
                    Err(error) => {
                        devices.failures.push((address, error.into()));
                        devices.owners.push(Box::new(configuration));
                        continue 'adoption;
                    }
                }
            }
        }
        let mut read32 = |address: Address, offset: u16| {
            if address.segment != slot.window.segment
                || !(slot.window.start_bus..=slot.window.end_bus).contains(&address.bus)
                || (address != bridge && !functions.contains(&address))
                || address.device >= 32
                || address.function >= 8
                || offset > 4092
                || !offset.is_multiple_of(4)
            {
                return Ok(None);
            }
            // SAFETY: bring_up owns this exact complete adopted subtree and
            // its original bridge under the same kept mapped ECAM window.
            let config = unsafe {
                slot.window
                    .config(address.bus, address.device, address.function)
            };
            config.read32(offset).map(Some)
        };
        // SAFETY: adopt just completed the entire subtree under this owned
        // slot. No child driver has started; all prior PF/VF assignments and
        // the bridge's updated routing remain owned throughout observation.
        if let Err(error) = unsafe {
            services.requesters_arrived(crate::probe::services::ArrivalInventory {
                bridge,
                functions: &functions,
                read32: &mut read32,
            })
        } {
            devices.failures.push((bridge, Error::Native(error)));
        }
        for address in arrivals.payload_mismatches {
            devices.warnings.push((
                address,
                Warning::Message("largest payload below its link's: expect malformed packets"),
            ));
        }
    }
    hosts
}
