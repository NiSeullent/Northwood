// SPDX-License-Identifier: GPL-2.0-or-later

//! Diagnostics derived from observed identities and already attached devices.
use super::{Machine, Verdict};
use crate::{host_firmware::Resources, output};
use alloc::{collections::BTreeSet, format, string::String, vec::Vec};
use northwood_config::SwapLocation;
use stela_drivers::{
    block::{BlockDevice, BlockError, Identity, Transport},
    bus::{lpss, pci},
    gpu::Vendor,
    input::InputDevice,
    net::{
        Link,
        wifi::intel::platform::{Observation, Variable},
    },
    probe::{
        Adapter, Devices, PlatformError,
        registry::{self, Class, Firmware, Status},
    },
};
use stela_loader::host_firmware::{
    MAX_MANIFEST_BYTES, Manifest, recipes::require_15igl7_wifi, requirements::Inventory,
};
use stela_runtime::pc::{
    fat32::FatError,
    gpt::{Guid, NORTHWOOD_SWAP_TYPE, Partition, PartitionDevice, Table},
    swap,
};

/// The manifest and payloads were checked by the loader before it passed
/// these resources to the diagnostic choice. The PCI census reads only
/// firmware-published configuration and does not attach devices.
pub(super) fn firmware(machine: &mut Machine, resources: &Resources) -> Vec<(u16, u16)> {
    let census = crate::pci_io::capture_host_check();
    let mut identities = Vec::new();
    let mut required = BTreeSet::new();
    let mut profile_selected = BTreeSet::new();
    let mut cnvi_addresses = BTreeSet::new();
    let observed = match census {
        Ok(census) => {
            let mut matched = 0usize;
            let mut unmatched = 0usize;
            let mut lpss_functions = 0usize;
            for function in &census.functions {
                let id = function.identity();
                let cnvi = (id.vendor, id.device, id.class) == (0x8086, 0x31dc, [2, 0x80, 0]);
                identities.push((id.vendor, id.device));
                let word = |offset: usize| {
                    u16::from_le_bytes([function.header[offset], function.header[offset + 1]])
                };
                lpss_functions += usize::from(
                    lpss::Profile::select(lpss::Identity {
                        vendor: id.vendor,
                        device: id.device,
                        subsystem_vendor: word(0x2c),
                        subsystem_device: word(0x2e),
                    })
                    .is_ok(),
                );
                machine.wifi_observed |= id.class[..2] == [0x02, 0x80];
                let mut found = false;
                for candidate in registry::pci_candidates(&id) {
                    found = true;
                    let implemented = !cnvi && candidate.status == Status::Implemented;
                    if candidate.class == Class::Wifi && implemented {
                        machine.wifi_candidate.get_or_insert(candidate.id);
                    }
                    if !cnvi {
                        match candidate.firmware {
                            Firmware::None => {}
                            Firmware::Required(names) => required.extend(names.iter().copied()),
                            Firmware::ProfileSelected(profile) => {
                                profile_selected.insert((candidate.id, profile));
                            }
                        }
                    }
                    let status = if implemented {
                        "candidate"
                    } else {
                        "inventory"
                    };
                    output::arguments(format_args!(
                        "DEVICE PCI {} {:04x}:{:04x} class={:02x}{:02x}{:02x} {status}={}.nwdrv ({})\n",
                        id.address,
                        id.vendor,
                        id.device,
                        id.class[0],
                        id.class[1],
                        id.class[2],
                        candidate.id,
                        candidate.implementation,
                    ));
                }
                if found {
                    matched += 1;
                } else {
                    unmatched += 1;
                    output::arguments(format_args!(
                        "DEVICE PCI {} {:04x}:{:04x} class={:02x}{:02x}{:02x} candidate=none\n",
                        id.address, id.vendor, id.device, id.class[0], id.class[1], id.class[2],
                    ));
                }
                if cnvi
                    && !cnvi_addresses.contains(&id.address)
                    && census
                        .functions
                        .iter()
                        .filter(|function| function.address == id.address)
                        .all(|function| function.identity() == id)
                {
                    cnvi_addresses.insert(id.address);
                    // The BDF supplies the PCI context for this boot. The
                    // UEFI variable observations belong to the platform.
                    let observations = resources.wifi_observations();
                    let general_cfg =
                        wifi_observation_state(observations.get(Variable::GeneralConfiguration));
                    let wbem = wifi_observation_state(observations.get(Variable::Wbem));
                    output::arguments(format_args!(
                        "HOST WIFI LARI RECORD pci={} UefiCnvWlanGeneralCfg={general_cfg} UefiCnvWlanWBEM={wbem}\n",
                        id.address,
                    ));
                }
            }
            let complete = census.failures.is_empty() && matched + unmatched != 0;
            for error in &census.failures {
                output::arguments(format_args!("DEVICE PCI observation: {error}\n"));
            }
            machine.record(
                "devices-pci",
                if complete {
                    Verdict::Pass
                } else {
                    Verdict::Skip
                },
                format!(
                    "UEFI census matched={matched} unmatched={unmatched} incomplete={}",
                    census.failures.len(),
                ),
            );
            if lpss_functions != 0 {
                machine.record(
                    "lpss-pci",
                    Verdict::Skip,
                    format!("LPSS I2C functions observed={lpss_functions}, firmware census supplies no stop evidence"),
                );
            }
            complete
        }
        Err(error) => {
            machine.record(
                "devices-pci",
                Verdict::Skip,
                format!("UEFI PCI census: {error}"),
            );
            false
        }
    };
    cnvi_dmar(&cnvi_addresses);
    let manifest = if resources.present {
        let region = resources.manifest;
        if region.physical == 0 || region.bytes == 0 || region.bytes > MAX_MANIFEST_BYTES as u64 {
            machine.record(
                "firmware",
                Verdict::Fail,
                String::from("invalid manifest range"),
            );
            return identities;
        }
        // SAFETY: the loader allocated and validated these immutable bytes
        // before entering the diagnostics choice. Firmware still identity maps them.
        let bytes = unsafe {
            core::slice::from_raw_parts(region.physical as *const u8, region.bytes as usize)
        };
        match Manifest::parse(bytes) {
            Ok(manifest) => Some(manifest),
            Err(error) => {
                machine.record(
                    "firmware",
                    Verdict::Fail,
                    format!("invalid manifest: {error}"),
                );
                return identities;
            }
        }
    } else {
        None
    };
    let mut missing = cnvi_firmware(machine, &cnvi_addresses, manifest.as_ref());
    for name in required {
        if !manifest
            .as_ref()
            .is_some_and(|manifest| manifest.images.iter().any(|image| image.name == name))
        {
            missing += 1;
            output::arguments(format_args!("FIRMWARE candidate resource missing={name}\n"));
        }
    }
    for &(module, selection) in &profile_selected {
        output::arguments(format_args!(
            "FIRMWARE candidate module={module}.nwdrv selection={selection}\n"
        ));
    }
    machine.record(
        "firmware",
        if missing != 0 {
            Verdict::Fail
        } else if !observed || !profile_selected.is_empty() {
            Verdict::Skip
        } else {
            Verdict::Pass
        },
        format!(
            "USB images={} fixed candidate names missing={missing} device-selected modules={}",
            resources.count,
            profile_selected.len(),
        ),
    );
    identities
}

fn cnvi_dmar(addresses: &BTreeSet<pci::Address>) {
    if addresses.is_empty() {
        return;
    }
    let Some(rsdp) = crate::storage::rsdp() else {
        output::arguments(format_args!("HOST WIFI CNVI DMAR rsdp=unavailable\n"));
        return;
    };
    // SAFETY: the firmware's RSDP and its tables are mapped before EBS.
    let table = match unsafe { stela_runtime::pc::acpi::find(rsdp, b"DMAR") } {
        Ok(table) => table,
        Err(error) => {
            output::arguments(format_args!("HOST WIFI CNVI DMAR table={error:?}\n"));
            return;
        }
    };
    let table = match stela_boot_core::dmar::parse(table) {
        Ok(table) => table,
        Err(error) => {
            output::arguments(format_args!("HOST WIFI CNVI DMAR parse={error:?}\n"));
            return;
        }
    };
    for address in addresses {
        let direct = |scope: &stela_boot_core::remapping::Scope| {
            !scope.bridge
                && scope.bus == address.bus
                && scope.path.as_slice() == [(address.device, address.function)]
        };
        let units = || {
            table
                .units
                .iter()
                .filter(|unit| unit.segment == address.segment)
        };
        let direct_units: Vec<_> = units()
            .filter(|unit| unit.scopes.iter().any(direct))
            .map(|unit| unit.base)
            .collect();
        let include_all: Vec<_> = units()
            .filter(|unit| unit.include_all)
            .map(|unit| unit.base)
            .collect();
        let direct_reserved = table
            .reserved
            .iter()
            .filter(|range| range.segment == address.segment && range.scopes.iter().any(direct))
            .count();
        output::arguments(format_args!(
            "HOST WIFI CNVI DMAR pci={address} direct_units={direct_units:x?} include_all_units={include_all:x?} direct_reserved={direct_reserved} scope=firmware-listing\n"
        ));
    }
}

fn cnvi_firmware(
    machine: &mut Machine,
    addresses: &BTreeSet<pci::Address>,
    manifest: Option<&Manifest<'_>>,
) -> usize {
    if addresses.is_empty() {
        return 0;
    }
    let mut requirements = Inventory::default();
    if let Err(error) = require_15igl7_wifi(&mut requirements) {
        machine.record("wifi-cnvi", Verdict::Fail, format!("{error}"));
        return 1;
    }
    let missing: Vec<_> = requirements
        .requests()
        .iter()
        .filter(|request| {
            let request = request.view();
            !manifest.is_some_and(|manifest| {
                manifest.images.iter().any(|image| {
                    image.name == request.name
                        && image.target.driver == request.target.driver
                        && image.target.hardware == request.target.hardware
                        && image.target.revision == request.target.revision
                        && image.target.board == request.target.board
                        && image.length >= request.minimum_bytes
                })
            })
        })
        .map(|request| request.name.as_str())
        .collect();
    let firmware = if missing.is_empty() {
        String::from("firmware package present")
    } else {
        format!("firmware missing={}", missing.join(","))
    };
    for address in addresses {
        machine.record(
            "wifi-cnvi",
            Verdict::Skip,
            format!("pci={address} 8086:31dc driver attachment unavailable; {firmware}"),
        );
    }
    missing.len()
}

fn wifi_observation_state(observation: Observation) -> &'static str {
    match observation {
        Observation::NotCaptured => "not-captured",
        Observation::Absent { .. } => "absent",
        Observation::Failed { .. } => "failed",
        Observation::BoundExceeded { .. } => "bound-exceeded",
        Observation::Present { .. } => "present",
    }
}

/// Check usable boot I/O and physical carrier after the existing host probe.
/// No network packets or credentials are supplied by this diagnostic.
pub(super) fn attached(machine: &mut Machine, devices: &mut Devices) {
    let display = machine.screen.is_some();
    let storage = devices.disks.len();
    let input = devices.inputs.len();
    let missing: Vec<_> = [
        (storage == 0, "storage", "Check native storage"),
        (input == 0, "input", "Connect USB input"),
        (!display, "GOP", "Select firmware display"),
    ]
    .into_iter()
    .filter_map(|(missing, kind, action)| missing.then_some((kind, action)))
    .collect();
    let storage_transports: BTreeSet<_> = devices
        .disks
        .iter()
        .map(|disk| match disk.identity().transport {
            Transport::VirtioBlk => "VirtioBlk",
            Transport::Nvme => "NVMe",
            Transport::Ahci => "AHCI",
            Transport::Usb => "USB",
            Transport::Ram => "RAM",
            Transport::Firmware => "Firmware",
            Transport::SdMmc => "SD/MMC",
        })
        .collect();
    let storage_transports = storage_transports.into_iter().collect::<Vec<_>>().join(",");
    let wifi_drivers: BTreeSet<_> = devices
        .nics
        .iter()
        .filter_map(|adapter| match adapter {
            Adapter::IntelWifi(_) | Adapter::MediatekWifi(_) | Adapter::RealtekWifi(_) => {
                Some(adapter.driver())
            }
            _ => None,
        })
        .collect();
    let wifi = if !wifi_drivers.is_empty() {
        format!(
            "{}:attached",
            wifi_drivers.into_iter().collect::<Vec<_>>().join(",")
        )
    } else if let Some(candidate) = machine.wifi_candidate {
        format!("{candidate}:candidate")
    } else if machine.wifi_observed {
        String::from("no registered candidate")
    } else {
        String::from("unobserved")
    };
    let mut detail = format!("disks={storage} input={input} GOP={display} Wi-Fi={wifi}");
    if !storage_transports.is_empty() {
        detail.push_str(&format!(" storage={storage_transports}"));
    }
    if !missing.is_empty() {
        detail.push_str(&format!(
            " missing={} | {}",
            missing
                .iter()
                .map(|(kind, _)| *kind)
                .collect::<Vec<_>>()
                .join(","),
            missing
                .iter()
                .map(|(_, action)| *action)
                .collect::<Vec<_>>()
                .join(". "),
        ));
    }
    machine.record(
        "devices",
        if missing.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        detail,
    );
    graphics(machine, devices, display);
    input_status(machine, devices);
    let mut carriers = 0;
    for adapter in &mut devices.nics {
        let carrier = adapter.carrier();
        carriers += usize::from(carrier);
        let transport = match adapter {
            Adapter::Usb(_) | Adapter::UsbVendor(_) => "USB",
            Adapter::IntelWifi(_) | Adapter::MediatekWifi(_) | Adapter::RealtekWifi(_) => "Wi-Fi",
            _ => "Ethernet",
        };
        output::arguments(format_args!(
            "NETWORK link={} driver={} transport={transport} carrier={carrier}\n",
            adapter.location(),
            adapter.driver(),
        ));
    }
    machine.record(
        "network",
        if carriers > 0 {
            Verdict::Pass
        } else {
            Verdict::Skip
        },
        if carriers > 0 {
            format!(
                "attached={} carrier={carriers} (carrier only)",
                devices.nics.len()
            )
        } else {
            format!(
                "attached={} carrier=0. Connect Ethernet or enable USB tethering.",
                devices.nics.len()
            )
        },
    );
}

fn graphics(machine: &mut Machine, devices: &Devices, display: bool) {
    let mut glk = None;
    for adapter in &devices.gpus {
        let vendor = match adapter.vendor {
            Vendor::Intel => 0x8086,
            Vendor::Nvidia => 0x10de,
            Vendor::Amd => 0x1002,
            Vendor::Other(vendor) => vendor,
        };
        let id = pci::Identity {
            address: adapter.address,
            vendor,
            device: adapter.device,
            class: adapter.class,
        };
        let module = registry::pci_candidates(&id)
            .find(|manifest| manifest.class == Class::Gpu && manifest.status == Status::Implemented)
            .map_or("none", |manifest| manifest.id);
        let mut detail = format!(
            "pci={} id={vendor:04x}:{:04x} accelerator-candidate={module}, {}",
            adapter.address,
            adapter.device,
            adapter.unavailable_reason(),
        );
        if let Some(memory) = adapter.glk_memory {
            let observation = match memory {
                Ok(memory) => format!(
                    "GLK PCI inventory GGTT={} MiB stolen={} MiB",
                    memory.ggtt_bytes >> 20,
                    memory.stolen.map_or(0, |stolen| stolen.bytes >> 20),
                ),
                Err(error) => format!("GLK PCI memory observation unavailable: {error:?}"),
            };
            detail.push_str(&format!(", {observation}"));
            glk.get_or_insert(observation);
        }
        machine.record("graphics-pci", Verdict::Skip, detail);
    }
    let detail = if display {
        "firmware GOP available"
    } else {
        "no usable firmware GOP, select firmware display"
    };
    let mut detail = format!("{detail}, PCI GPUs={} (inventory only)", devices.gpus.len());
    if let Some(glk) = glk {
        detail.push_str(&format!(", {glk}"));
    }
    machine.record(
        "graphics",
        if display {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        detail,
    );
}

fn input_status(machine: &mut Machine, devices: &Devices) {
    let drivers: BTreeSet<_> = devices
        .inputs
        .iter()
        .map(|input| match input {
            InputDevice::Virtio(_) => "virtio-input",
            InputDevice::Usb(_) => "USB HID",
            InputDevice::Ehci(_) => "EHCI HID",
            InputDevice::Ps2(_) => "PS/2",
            InputDevice::I2cHid(_) => "I2C-HID",
            InputDevice::Touchpad(_) => "touchpad",
            InputDevice::Ish(_) => "Intel ISH",
        })
        .collect();
    let mut detail = if drivers.is_empty() {
        String::from("no input attached, connect USB input")
    } else {
        format!(
            "attached={} drivers={}",
            devices.inputs.len(),
            drivers.into_iter().collect::<Vec<_>>().join(",")
        )
    };
    for (_, error) in &devices.platform_failures {
        if matches!(
            error,
            PlatformError::I2cPciMismatch(_) | PlatformError::I2cPciStop(..)
        ) {
            detail.push_str(&format!(", {error}"));
            break;
        }
    }
    if !devices.native_services {
        detail.push_str(", ACPI I2C attach unavailable in standalone probe");
    }
    machine.record(
        "input",
        if devices.inputs.is_empty() {
            Verdict::Fail
        } else {
            Verdict::Pass
        },
        detail,
    );
}

/// Only the active boot stick's FAT32 data sibling may supply the volume
/// candidate. The GPT partition list is read through the existing native
/// storage check. This observer has no write entry point.
pub(super) struct Swap {
    location: SwapLocation,
    partition_guid: Option<Guid>,
    volume_max_bytes: u64,
    partition_inventory: usize,
    partition: Option<(Guid, u64)>,
    volume: Option<(Guid, u64)>,
    boot_matches: usize,
    boot_usb: bool,
    partition_refusal: Option<String>,
    volume_refusal: Option<String>,
}

impl Swap {
    pub(super) fn new(config: &northwood_config::Swap) -> Self {
        let location = if config.enabled {
            config.location
        } else {
            SwapLocation::None
        };
        let partition_requested = matches!(location, SwapLocation::Auto | SwapLocation::Partition)
            && !config.partition.is_empty();
        let partition_guid = if partition_requested {
            Guid::parse(&config.partition).filter(|guid| !guid.is_zero())
        } else {
            None
        };
        Self {
            location,
            partition_guid,
            volume_max_bytes: config.volume_max_bytes,
            partition_inventory: 0,
            partition: None,
            volume: None,
            boot_matches: 0,
            boot_usb: false,
            partition_refusal: (partition_requested && partition_guid.is_none())
                .then(|| String::from("invalid selected GPT swap UUID")),
            volume_refusal: None,
        }
    }

    pub(super) fn observe(
        &mut self,
        disk: &mut dyn BlockDevice,
        table: &Table,
        boot: Option<Guid>,
    ) {
        if matches!(self.location, SwapLocation::Auto | SwapLocation::Partition) {
            self.observe_partition(disk, table);
        }
        if matches!(self.location, SwapLocation::Auto | SwapLocation::VolumeFile)
            && let Some(boot) = boot
            && !boot.is_zero()
        {
            self.observe_volume(disk, table, boot);
        }
    }

    fn observe_partition(&mut self, disk: &mut dyn BlockDevice, table: &Table) {
        let Some(id) = self.partition_guid else {
            self.partition_inventory += table
                .partitions
                .iter()
                .filter(|partition| partition.type_guid == NORTHWOOD_SWAP_TYPE)
                .count();
            return;
        };
        if self.partition_refusal.is_some()
            || !table.partitions.iter().any(|entry| entry.unique_guid == id)
        {
            return;
        }
        if self.partition.is_some() {
            self.partition_refusal = Some(format!("duplicate GPT swap UUID {id}"));
            return;
        }
        match swap::partition::inspect(&mut ReadOnlyDisk(disk), Some(id)) {
            Ok((_, header)) => {
                self.partition = Some((id, header.layout().slot_count() * swap::SLOT_BYTES));
            }
            Err(error) => self.partition_refusal = Some(format!("partition={id} refused: {error}")),
        }
    }

    fn observe_volume(&mut self, disk: &mut dyn BlockDevice, table: &Table, boot: Guid) {
        let mut matches = table
            .partitions
            .iter()
            .filter(|entry| entry.unique_guid == boot);
        let Some(esp) = matches.next() else {
            return;
        };
        self.boot_matches += 1;
        if self.boot_matches != 1 || matches.next().is_some() {
            self.volume_refusal = Some(String::from("duplicate boot ESP GUID"));
            return;
        }
        if esp.type_guid != Guid::ESP {
            self.volume_refusal = Some(String::from("boot GPT entry is not an ESP"));
            return;
        }
        if disk.identity().transport != Transport::Usb {
            return;
        }
        self.boot_usb = true;
        let basic_data = Guid::parse("EBD0A0A2-B9E5-4433-87C0-68B6B72699C7");
        let mut data = table
            .partitions
            .iter()
            .filter(|entry| Some(entry.type_guid) == basic_data);
        let Some(partition) = data.next() else {
            return;
        };
        if data.next().is_some() {
            self.volume_refusal = Some(String::from("boot disk has several basic data partitions"));
            return;
        }
        let id = partition.unique_guid;
        if id.is_zero()
            || table.partitions.iter().any(|entry| {
                !core::ptr::eq(entry, partition)
                    && (entry.unique_guid == id
                        || (entry.first_lba <= partition.last_lba
                            && partition.first_lba <= entry.last_lba))
            })
        {
            self.volume_refusal = Some(String::from(
                "boot data partition has an invalid GPT identity or extent",
            ));
            return;
        }
        let block = u64::from(disk.identity().block_size);
        match volume(disk, partition, block, self.volume_max_bytes) {
            Ok(metadata) => self.volume = Some((id, metadata.config.bytes())),
            Err(swap::volume::Error::Fat(FatError::NotFat32(_))) => {}
            Err(swap::volume::Error::Fat(FatError::Missing(path)))
                if missing_volume_config(&path) => {}
            Err(error) => self.volume_refusal = Some(format!("data volume={id} refused: {error}")),
        }
    }

    pub(super) fn report(self, machine: &mut Machine, required: bool) {
        let (verdict, detail) = self.summary(required);
        machine.record("swap", verdict, detail);
    }

    fn summary(self, required: bool) -> (Verdict, String) {
        if self.location == SwapLocation::None {
            return (
                Verdict::Skip,
                String::from("swap disabled by configuration"),
            );
        }
        if self.partition_refusal.is_none()
            && let Some((id, bytes)) = self.partition
        {
            return (
                Verdict::Pass,
                format!(
                    "partition={id} usable={} MiB (read-only metadata, paging inactive)",
                    bytes >> 20
                ),
            );
        }
        if self.volume_refusal.is_none()
            && let Some((id, bytes)) = self.volume
        {
            let mut detail = format!(
                "boot data volume={id} capacity={} MiB (read-only metadata, paging inactive)",
                bytes >> 20,
            );
            if let Some(reason) = self.partition_refusal {
                detail.push_str(&format!(", {reason}"));
            }
            return (Verdict::Pass, detail);
        }

        let refused = self.partition_refusal.is_some() || self.volume_refusal.is_some();
        let mut reasons = Vec::new();
        if let Some(reason) = self.partition_refusal {
            reasons.push(reason);
        } else if let Some(id) = self.partition_guid {
            reasons.push(format!("selected GPT swap partition={id} not observed"));
        } else if self.partition_inventory > 0 || self.location == SwapLocation::Partition {
            reasons.push(format!(
                "typed GPT partitions={} (no swap UUID selected)",
                self.partition_inventory,
            ));
        }
        if let Some(reason) = self.volume_refusal {
            reasons.push(reason);
        } else if matches!(self.location, SwapLocation::Auto | SwapLocation::VolumeFile) {
            reasons.push(String::from(if !self.boot_usb {
                "no prepared swap store observed (boot USB data sibling not observed)"
            } else {
                "no prepared swap store on boot USB FAT32 data volume"
            }));
        }
        let mut detail = reasons.join(", ");
        if required || refused {
            detail.push_str(" | Verify the selected store or prepare swap in setup. Resident boot may continue.");
        }
        (
            if required || refused {
                Verdict::Fail
            } else {
                Verdict::Skip
            },
            detail,
        )
    }
}

fn missing_volume_config(path: &str) -> bool {
    swap::volume::CONFIG_PATH
        .strip_prefix(path.trim_start_matches('/'))
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

fn volume(
    disk: &mut dyn BlockDevice,
    partition: &Partition,
    block: u64,
    maximum: u64,
) -> Result<swap::volume::Metadata, swap::volume::Error> {
    let partition = PartitionDevice::new(ReadOnlyDisk(disk), partition.clone());
    swap::volume::inspect(crate::engine::bundle::Bytes::new(partition), block, maximum)
}

struct ReadOnlyDisk<'a>(&'a mut dyn BlockDevice);

impl BlockDevice for ReadOnlyDisk<'_> {
    fn identity(&self) -> &Identity {
        self.0.identity()
    }

    fn read(&mut self, lba: u64, buffer: &mut [u8]) -> Result<(), BlockError> {
        self.0.read(lba, buffer)
    }

    fn write(&mut self, _: u64, _: &[u8]) -> Result<(), BlockError> {
        Err(BlockError::ReadOnly)
    }
}
