// SPDX-License-Identifier: GPL-2.0-or-later
//! Lavapipe admission before opening the Metal service. It uses the same
//! physical-device and queue queries as the PCI path.

use super::{Extensions, NativeBackend, Target, application, failure};
use crate::metal::{
    error::{Error, Result},
    vulkan::{backend::requirements, physical::PhysicalDevice},
};
use ash::{ext, vk};
use core::num::NonZeroU64;

pub(super) const HOST_ALLOCATION: vk::ExternalMemoryHandleTypeFlags =
    vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;

/// The CPU device and the requirements it satisfies. The host-pointer
/// result applies to storage buffers with a shader device address.
#[derive(Debug)]
pub struct SoftwareDevice {
    pub physical: PhysicalDevice,
    pub queue_family: u32,
    pub host_storage_alignment: Option<NonZeroU64>,
    /// Ordinary AIR requests preservation of FP32 subnormals.
    pub fp32_denorm_preserve: bool,
}

impl SoftwareDevice {
    /// Reports the first unmet requirement for the current Metal backend.
    /// Shader and memory support are both taken from the driver.
    pub fn check(&self) -> Result<()> {
        check(
            &self.physical,
            self.host_storage_alignment,
            self.fp32_denorm_preserve,
        )
        .map(|_| ())
    }
}

pub(super) fn check(
    physical: &PhysicalDevice,
    alignment: Option<NonZeroU64>,
    fp32_denorm_preserve: bool,
) -> Result<NonZeroU64> {
    if let Some(missing) = requirements(&physical.features)
        .iter()
        .find(|requirement| !requirement.present)
    {
        return Err(Error::Unsupported(missing.name));
    }
    let alignment = alignment.ok_or(Error::Unsupported(
        "Vulkan host allocation for storage buffers",
    ))?;
    if !fp32_denorm_preserve {
        return Err(Error::Unsupported("shaderDenormPreserveFloat32"));
    }
    Ok(alignment)
}

#[cfg(any(test, all(feature = "metal-linked", not(target_os = "uefi"))))]
pub(super) fn check_hosted_lavapipe(
    physical: &PhysicalDevice,
    alignment: Option<NonZeroU64>,
) -> Result<NonZeroU64> {
    check_hosted_features(physical)?;
    alignment.ok_or(Error::Unsupported(
        "Vulkan host allocation for storage buffers",
    ))
}

#[cfg(any(test, feature = "metal-linked"))]
pub(super) fn check_hosted_features(physical: &PhysicalDevice) -> Result<()> {
    if let Some(missing) = requirements(&physical.features)
        .iter()
        .find(|requirement| !requirement.present && requirement.name != "shaderFloat16")
    {
        return Err(Error::Unsupported(missing.name));
    }
    if !physical.features.shader_int16 {
        return Err(Error::Unsupported("shaderInt16 for software half"));
    }
    Ok(())
}

/// Inspects the lavapipe CPU device through the supplied Vulkan driver or
/// loader. A temporary instance is destroyed before returning the snapshot.
///
/// # Safety
///
/// `get_instance_proc_addr` and the Vulkan functions it provides must be
/// valid for the duration of this call.
pub unsafe fn probe_software(
    get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    log: fn(&str),
) -> Result<SoftwareDevice> {
    let static_fn = ash::StaticFn {
        get_instance_proc_addr,
    };
    // SAFETY: the caller guarantees the driver entry point.
    let entry = unsafe { ash::Entry::from_static_fn(static_fn) };
    let application = application();
    let info = vk::InstanceCreateInfo::default().application_info(&application);
    // SAFETY: the application info and its strings outlive the call.
    let instance = unsafe { entry.create_instance(&info, None) }.map_err(failure)?;
    let selected = NativeBackend::select(&instance, Target::Software, log).map(
        |(device, extensions, physical, queue_family)| {
            let (fp32_denorm_preserve, host_storage_alignment) =
                probe_properties(&instance, device, &extensions);
            SoftwareDevice {
                physical,
                queue_family,
                host_storage_alignment,
                fp32_denorm_preserve,
            }
        },
    );
    // SAFETY: the probe created no device or other instance child, and
    // selected owns only data copied from the queries.
    unsafe { instance.destroy_instance(None) };
    selected.ok_or(Error::Unsupported("Vulkan lavapipe CPU device"))
}

pub(super) fn probe_properties(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    extensions: &Extensions,
) -> (bool, Option<NonZeroU64>) {
    let has_host = extensions.has(ext::external_memory_host::NAME);
    let mut float = vk::PhysicalDeviceFloatControlsProperties::default();
    let mut host = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut float);
    if has_host {
        properties = properties.push_next(&mut host);
    }
    // SAFETY: the device is Vulkan 1.3+, and only chains the host pointer
    // structure when the device offers the extension.
    unsafe { instance.get_physical_device_properties2(device, &mut properties) };
    let preserves = float.shader_denorm_preserve_float32 == vk::TRUE;
    if !has_host {
        return (preserves, None);
    }
    let info = vk::PhysicalDeviceExternalBufferInfo::default()
        .handle_type(HOST_ALLOCATION)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
    let mut external = vk::ExternalBufferProperties::default();
    // SAFETY: the device belongs to this instance and offers the host-memory
    // extension. The query's input and output outlive the call.
    unsafe {
        instance.get_physical_device_external_buffer_properties(device, &info, &mut external)
    };
    (
        preserves,
        storage_alignment(
            host.min_imported_host_pointer_alignment,
            external.external_memory_properties,
        ),
    )
}

fn storage_alignment(
    alignment: u64,
    properties: vk::ExternalMemoryProperties,
) -> Option<NonZeroU64> {
    if alignment.is_power_of_two() && host_importable(properties) {
        return NonZeroU64::new(alignment);
    }
    None
}

pub(super) fn host_importable(properties: vk::ExternalMemoryProperties) -> bool {
    let features = properties.external_memory_features;
    // Metal can bind multiple buffers into the same backing memory.
    properties.compatible_handle_types.contains(HOST_ALLOCATION)
        && features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
        && !features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::{
        nvidia::{Chip, fixture::nvk},
        vulkan::{DriverId, PhysicalDeviceType, physical::PciBusInfo},
    };

    #[test]
    fn the_cpu_and_pci_targets_select_their_own_device() {
        let mut gpu = nvk(Chip::find(0x1b3).unwrap(), 16 << 30);
        let pci = PciBusInfo {
            domain: 0,
            bus: 1,
            device: 0,
            function: 0,
        };
        gpu.identity.pci_bus = Some(pci);
        assert!(Target::Pci(pci).matches(&gpu));
        assert!(!Target::Software.matches(&gpu));
        assert_eq!(
            DriverId::MesaLlvmPipe.raw(),
            vk::DriverId::MESA_LLVMPIPE.as_raw() as u32
        );
        gpu.identity.driver_id = DriverId::MesaLlvmPipe;
        gpu.identity.device_type = PhysicalDeviceType::Cpu;
        gpu.identity.api_version = vk::API_VERSION_1_3;
        assert!(!Target::Software.matches(&gpu));
        assert!(!Target::Pci(pci).matches(&gpu));
        gpu.identity.pci_bus = None;
        assert!(Target::Software.matches(&gpu));
        gpu.identity.api_version = vk::API_VERSION_1_2;
        assert!(!Target::Software.matches(&gpu));
    }

    #[test]
    fn host_storage_must_accept_nondedicated_host_allocations() {
        use vk::ExternalMemoryFeatureFlags as F;
        let properties = |features, handles| {
            vk::ExternalMemoryProperties::default()
                .external_memory_features(features)
                .compatible_handle_types(handles)
        };
        let host = properties(F::IMPORTABLE, HOST_ALLOCATION);
        assert_eq!(storage_alignment(4096, host), NonZeroU64::new(4096));
        for invalid in [0, 4095, 4097] {
            assert_eq!(storage_alignment(invalid, host), None);
        }
        assert_eq!(
            storage_alignment(4096, properties(F::EXPORTABLE, HOST_ALLOCATION)),
            None
        );
        assert_eq!(
            storage_alignment(
                4096,
                properties(F::IMPORTABLE | F::DEDICATED_ONLY, HOST_ALLOCATION)
            ),
            None
        );
        assert_eq!(
            storage_alignment(
                4096,
                properties(
                    F::IMPORTABLE,
                    vk::ExternalMemoryHandleTypeFlags::HOST_MAPPED_FOREIGN_MEMORY_EXT
                )
            ),
            None
        );
    }

    #[test]
    fn host_import_does_not_override_the_shader_gate() {
        let mut physical = nvk(Chip::find(0x1b3).unwrap(), 16 << 30);
        physical.features.shader_float16 = false;
        let cpu = SoftwareDevice {
            physical,
            queue_family: 0,
            host_storage_alignment: NonZeroU64::new(4096),
            fp32_denorm_preserve: true,
        };
        assert_eq!(cpu.check(), Err(Error::Unsupported("shaderFloat16")));
    }

    #[test]
    fn software_admission_requires_fp32_subnormal_preservation() {
        let cpu = SoftwareDevice {
            physical: nvk(Chip::find(0x1b3).unwrap(), 16 << 30),
            queue_family: 0,
            host_storage_alignment: NonZeroU64::new(4096),
            fp32_denorm_preserve: false,
        };
        assert_eq!(
            cpu.check(),
            Err(Error::Unsupported("shaderDenormPreserveFloat32"))
        );
    }

    #[test]
    fn hosted_admission_keeps_other_vulkan_and_host_requirements() {
        let mut physical = nvk(Chip::find(0x1b3).unwrap(), 16 << 30);
        let alignment = NonZeroU64::new(4096);
        physical.features.shader_float16 = false;
        physical.features.shader_int16 = true;
        assert_eq!(
            check_hosted_lavapipe(&physical, alignment),
            Ok(alignment.unwrap())
        );
        assert_eq!(
            check(&physical, alignment, false),
            Err(Error::Unsupported("shaderFloat16"))
        );
        assert_eq!(
            check_hosted_lavapipe(&physical, None),
            Err(Error::Unsupported(
                "Vulkan host allocation for storage buffers"
            ))
        );
        physical.features.shader_int16 = false;
        assert_eq!(
            check_hosted_lavapipe(&physical, alignment),
            Err(Error::Unsupported("shaderInt16 for software half"))
        );
        physical.features.null_descriptor = false;
        assert_eq!(
            check_hosted_lavapipe(&physical, alignment),
            Err(Error::Unsupported("nullDescriptor"))
        );
    }
}
