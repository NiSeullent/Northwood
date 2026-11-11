// SPDX-License-Identifier: GPL-2.0-or-later
//! Explicit admission for Stela's patched lavapipe build.

#[cfg(not(target_os = "uefi"))]
use super::HostMemoryExporter;
use super::{GuestCopyProvider, InternalKernels, Loader, MemorySource, NativeBackend, Target, vk};
use crate::metal::{
    compiler::{self, AirCompiler},
    device::Device,
    error::{Error, Result},
    service::{Service, identity::DeviceEpoch},
    vulkan::backend::Backend,
};
use alloc::boxed::Box;
use core::{ffi::c_void, ptr::NonNull};

/// Opens the bounded Stela lavapipe service with leased host allocations.
///
/// # Safety
/// The ICD, linked compiler initialization and NIR options must meet the
/// requirements of [`open_copied_lavapipe`].
/// `exporter` must honor the contract of [`super::HostAllocation`].
#[cfg(not(target_os = "uefi"))]
pub unsafe fn open_hosted_lavapipe(
    loader: Loader,
    exporter: Box<dyn HostMemoryExporter>,
    nir_options: impl FnOnce(vk::PhysicalDevice) -> Result<NonNull<c_void>>,
    epoch: DeviceEpoch,
) -> Result<Service<NativeBackend, AirCompiler>> {
    // SAFETY: inherited from this public constructor.
    unsafe {
        open(
            loader,
            MemorySource::HostedLavapipe(exporter),
            nir_options,
            epoch,
        )
    }
}

/// Opens the bounded lavapipe service with canonical guest RAM copies.
/// The service offers buffer compute and single-sample color rendering.
/// Imported buffers can only back render targets through private storage.
///
/// # Safety
/// The caller must verify the provenance of the exact ICD supplied to
/// `loader`. It must implement Stela's per-shader FP32 mode for compute,
/// vertex and fragment execution. Its private NIR ABI and internal kernels
/// must match the linked compiler, whose Mesa runtime must already be
/// initialized. `nir_options` must return options for the selected physical
/// device and this compiler's supported stages. The ICD and options must
/// outlive every use through the service.
///
/// Every import must come from a trusted gate or an equivalent fixture and
/// keep the tagged bidirectional wire alive through retirement, as required
/// by [`GuestCopyProvider::lease`].
pub unsafe fn open_copied_lavapipe(
    loader: Loader,
    provider: Box<dyn GuestCopyProvider>,
    nir_options: impl FnOnce(vk::PhysicalDevice) -> Result<NonNull<c_void>>,
    epoch: DeviceEpoch,
) -> Result<Service<NativeBackend, AirCompiler>> {
    // SAFETY: inherited from this public constructor.
    unsafe {
        open(
            loader,
            MemorySource::CopiedLavapipe(provider),
            nir_options,
            epoch,
        )
    }
}

unsafe fn open(
    loader: Loader,
    source: MemorySource,
    nir_options: impl FnOnce(vk::PhysicalDevice) -> Result<NonNull<c_void>>,
    epoch: DeviceEpoch,
) -> Result<Service<NativeBackend, AirCompiler>> {
    let mut built = None;
    let kernels = |physical| {
        let options = nir_options(physical)?;
        // SAFETY: the caller supplies options matching the linked compiler.
        let code = unsafe { compiler::Service::linked(options).internal_kernels() }?;
        let [
            accumulate,
            select,
            resolve_draws,
            resolve_dispatches,
            store_texels,
            load_texels,
            average_samples,
            decode_etc,
            decode_pvrtc,
            reorder_blocks,
            decode_astc,
        ] = code.code();
        // The backend copies this NIR into its pipelines during open.
        built = Some((options, code));
        Ok(InternalKernels {
            accumulate,
            select,
            resolve_draws,
            resolve_dispatches,
            store_texels,
            load_texels,
            average_samples,
            decode_etc,
            decode_pvrtc,
            reorder_blocks,
            decode_astc,
        })
    };
    // SAFETY: the caller verifies this ICD's private shader behavior and ABI.
    let backend = unsafe { NativeBackend::open(loader, Target::Software, source, kernels) }?;
    let Some((options, code)) = built else {
        return Err(Error::Invalid("the backend asked for no kernels"));
    };
    drop(code);
    // SAFETY: the options belong to the live selected physical device.
    let compiler = unsafe { AirCompiler::new_linked_lavapipe(options, backend.physical_device()) }?;
    Ok(Service::new(
        Device::new_hosted_lavapipe(backend, compiler)?,
        epoch,
    ))
}
