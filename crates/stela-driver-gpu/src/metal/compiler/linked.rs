//! The [`Service`] of the compiler's C linked into this image.
use super::{AirCompiler, Lowering, Service, ffi};
use crate::metal::{
    error::{Error, Result},
    vulkan::{DriverId, PhysicalDeviceType, physical::PhysicalDevice},
};
use core::{
    ffi::{c_char, c_void},
    ptr::NonNull,
};

unsafe extern "C" {
    fn stela_metal_reflect(
        air: *const c_void,
        size: usize,
        entry: *const c_char,
        constants: *const ffi::ConstantValue,
        constant_count: u32,
        reflection: *mut ffi::Reflection,
        error: *mut ffi::AirError,
    ) -> bool;
    fn stela_metal_reflect_bindings(
        air: *const c_void,
        size: usize,
        entry: *const c_char,
        constants: *const ffi::ConstantValue,
        constant_count: u32,
        bindings: *mut ffi::Bindings,
        error: *mut ffi::AirError,
    ) -> bool;
    fn stela_metal_sampler_states(
        air: *const c_void,
        size: usize,
        states: *mut u64,
        count: *mut u32,
        error: *mut ffi::AirError,
    ) -> bool;
    fn stela_metal_functions(
        air: *const c_void,
        size: usize,
        visit: ffi::VisitFunction,
        data: *mut c_void,
        error: *mut ffi::AirError,
    ) -> bool;
    fn stela_metal_exports(
        air: *const c_void,
        size: usize,
        visit: ffi::VisitFunction,
        data: *mut c_void,
        error: *mut ffi::AirError,
    ) -> bool;
    fn stela_metal_stitch(
        sources: *const ffi::StitchSource,
        source_count: u32,
        graphs: *const ffi::StitchGraph,
        graph_count: u32,
        visit: ffi::VisitBytes,
        data: *mut c_void,
        error: *mut ffi::AirError,
    ) -> bool;
    fn stela_metal_compile_reflected(
        air: *const c_void,
        size: usize,
        entry: *const c_char,
        target: *const ffi::Target,
        specialization: *const ffi::Specialization,
        reflection: *const ffi::Reflection,
        error: *mut ffi::AirError,
    ) -> *mut c_void;
    fn stela_metal_compile_software_reflected(
        air: *const c_void,
        size: usize,
        entry: *const c_char,
        target: *const ffi::Target,
        specialization: *const ffi::Specialization,
        reflection: *const ffi::Reflection,
        error: *mut ffi::AirError,
    ) -> *mut c_void;
    fn stela_metal_compile_lavapipe_reflected(
        air: *const c_void,
        size: usize,
        entry: *const c_char,
        target: *const ffi::Target,
        specialization: *const ffi::Specialization,
        reflection: *const ffi::Reflection,
        error: *mut ffi::AirError,
    ) -> *mut c_void;
    fn stela_metal_tessellation_stages(
        target: *const ffi::Target,
        tessellation: *const ffi::Tessellation,
        stages: *mut *mut c_void,
    );
    fn stela_metal_accumulate_kernel(options: *const c_void) -> *mut c_void;
    fn stela_metal_select_kernel(options: *const c_void) -> *mut c_void;
    fn stela_metal_resolve_draws_kernel(options: *const c_void) -> *mut c_void;
    fn stela_metal_resolve_dispatches_kernel(options: *const c_void) -> *mut c_void;
    fn stela_metal_store_texels_kernel(options: *const c_void) -> *mut c_void;
    fn stela_metal_load_texels_kernel(options: *const c_void) -> *mut c_void;
    fn stela_metal_average_samples_kernel(options: *const c_void) -> *mut c_void;
    fn stela_metal_decode_etc_kernel(options: *const c_void) -> *mut c_void;
    fn stela_metal_decode_pvrtc_kernel(options: *const c_void) -> *mut c_void;
    fn stela_metal_reorder_blocks_kernel(options: *const c_void) -> *mut c_void;
    fn stela_metal_decode_astc_kernel(options: *const c_void) -> *mut c_void;
    fn stela_metal_release(shader: *mut c_void);
    fn stela_metal_buffer_reads(shader: *const c_void) -> u32;
}

impl AirCompiler {
    /// Opens the linked lavapipe compiler with software half arithmetic and
    /// a distinct compiled NIR cache for compute and supported render stages.
    ///
    /// # Safety
    ///
    /// `nir_options` must be the selected device's NIR options and must
    /// outlive this compiler and every shader it compiles.
    pub unsafe fn new_linked_software(
        nir_options: NonNull<c_void>,
        device: &PhysicalDevice,
    ) -> Result<Self> {
        let service = Service::linked_software(nir_options, device)?;
        // SAFETY: the caller guarantees the lifetime and origin of the
        // driver's options; the compiler functions are linked into the image.
        let mut compiler = unsafe { Self::new(service, device) };
        compiler.lowering = Lowering::SoftwareHalf;
        Ok(compiler)
    }

    /// Opens the bounded AIR profile for the linked Stela lavapipe build,
    /// preserving FP32 subnormals in the shader operations it admits.
    ///
    /// The selected CPU driver must include Stela's per-shader denormal mode
    /// support for compute, the vertex frontend and fragment rasterization.
    ///
    /// # Safety
    ///
    /// `nir_options` must belong to that device and outlive the compiler and
    /// all its compiled shaders.
    pub unsafe fn new_linked_lavapipe(
        nir_options: NonNull<c_void>,
        device: &PhysicalDevice,
    ) -> Result<Self> {
        let mut service = Service::linked_software(nir_options, device)?;
        service.compile = stela_metal_compile_lavapipe_reflected;
        // SAFETY: the caller guarantees the validity of the driver's options.
        let mut compiler = unsafe { Self::new(service, device) };
        compiler.lowering = Lowering::Lavapipe;
        Ok(compiler)
    }
}

impl Service {
    /// Selects bounded software half lowering. The mode uses 16-bit integer
    /// SSA and refuses unsupported operations and shader interfaces at compilation.
    fn linked_software(nir_options: NonNull<c_void>, device: &PhysicalDevice) -> Result<Self> {
        let identity = &device.identity;
        if identity.driver_id != DriverId::MesaLlvmPipe
            || identity.device_type != PhysicalDeviceType::Cpu
            || identity.pci_bus.is_some()
        {
            return Err(Error::Unsupported(
                "the software compiler requires lavapipe",
            ));
        }
        if !device.features.shader_int16 {
            return Err(Error::Unsupported("shaderInt16 for software half"));
        }
        let mut service = Self::linked(nir_options);
        service.compile = stela_metal_compile_software_reflected;
        Ok(service)
    }

    /// The linked query depends only on the verified module's bytes.
    #[cfg(target_os = "uefi")]
    pub(super) fn has_linked_sampler_states(&self) -> bool {
        let linked = Self::linked(self.nir_options);
        core::ptr::fn_addr_eq(self.sampler_states, linked.sampler_states)
    }

    #[cfg(target_os = "uefi")]
    pub(super) fn accepts_compilation_projection(&self) -> bool {
        let linked = Self::linked(self.nir_options);
        core::ptr::fn_addr_eq(self.reflect, linked.reflect)
            && core::ptr::fn_addr_eq(self.compile, linked.compile)
    }

    /// The linked service, compiling with the driver's `nir_options`.
    pub fn linked(nir_options: NonNull<c_void>) -> Self {
        Self {
            reflect: stela_metal_reflect,
            reflect_bindings: stela_metal_reflect_bindings,
            sampler_states: stela_metal_sampler_states,
            functions: stela_metal_functions,
            exports: stela_metal_exports,
            stitch: stela_metal_stitch,
            compile: stela_metal_compile_reflected,
            tessellation_stages: stela_metal_tessellation_stages,
            accumulate_kernel: stela_metal_accumulate_kernel,
            select_kernel: stela_metal_select_kernel,
            resolve_draws_kernel: stela_metal_resolve_draws_kernel,
            resolve_dispatches_kernel: stela_metal_resolve_dispatches_kernel,
            store_texels_kernel: stela_metal_store_texels_kernel,
            load_texels_kernel: stela_metal_load_texels_kernel,
            average_samples_kernel: stela_metal_average_samples_kernel,
            decode_etc_kernel: stela_metal_decode_etc_kernel,
            decode_pvrtc_kernel: stela_metal_decode_pvrtc_kernel,
            reorder_blocks_kernel: stela_metal_reorder_blocks_kernel,
            decode_astc_kernel: stela_metal_decode_astc_kernel,
            release: stela_metal_release,
            buffer_reads: stela_metal_buffer_reads,
            nir_options,
        }
    }
}
