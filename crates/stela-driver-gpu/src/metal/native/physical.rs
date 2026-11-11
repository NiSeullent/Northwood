//! The driver's answers for one `VkPhysicalDevice`: the device extensions
//! this backend uses, the feature chain it probes and enables, and the
//! [`PhysicalDevice`] snapshot Metal's capabilities derive from.
use super::device_generated;
use crate::metal::vulkan::{
    DriverId, Format, FormatFeatures, MemoryHeapFlags, MemoryProperties, PhysicalDeviceType,
    ResolveModes, SampleCounts, ShaderStages, SubgroupFeatures,
    physical::{
        Features, FormatSupport, FormatTable, Identity, Limits, Memory, MemoryHeap, MemoryType,
        MeshLimits, PciBusInfo, PhysicalDevice, SampleLocationLimits, SparseLimits,
    },
};
use alloc::{boxed::Box, string::String, vec::Vec};
use ash::{ext, khr, nv, vk};
use core::{
    ffi::{CStr, c_void},
    ptr,
};

/// `VK_KHR_unified_image_layouts`, newer than the bindings.
pub(super) const UNIFIED_IMAGE_LAYOUTS: &CStr = c"VK_KHR_unified_image_layouts";

/// `VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR`
/// (`VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR`
/// is 1000527000 in Mesa 26.2.5's `vulkan_core.h`).
#[repr(C)]
#[derive(Clone, Copy)]
struct UnifiedImageLayoutsFeatures {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    unified_image_layouts: vk::Bool32,
    unified_image_layouts_video: vk::Bool32,
}

/// `VK_KHR_maintenance8`, newer than the bindings: copies between depth or
/// stencil aspects and color images.
pub(super) const MAINTENANCE8: &CStr = c"VK_KHR_maintenance8";

/// `VkPhysicalDeviceMaintenance8FeaturesKHR`
/// (`VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR` is
/// 1000574000 in Mesa 26.2.5's `vulkan_core.h`).
#[repr(C)]
#[derive(Clone, Copy)]
struct Maintenance8Features {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    maintenance8: vk::Bool32,
}

impl Default for Maintenance8Features {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(1_000_574_000),
            p_next: ptr::null_mut(),
            maintenance8: vk::FALSE,
        }
    }
}

impl Default for UnifiedImageLayoutsFeatures {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(1_000_527_000),
            p_next: ptr::null_mut(),
            unified_image_layouts: vk::FALSE,
            unified_image_layouts_video: vk::FALSE,
        }
    }
}

/// Extensions the device enables when the driver offers them. The
/// requirements decide which absences refuse the device.
const ENABLED: [&CStr; 33] = [
    khr::pipeline_executable_properties::NAME,
    UNIFIED_IMAGE_LAYOUTS,
    MAINTENANCE8,
    device_generated::NAME,
    ext::robustness2::NAME,
    ext::dynamic_rendering_unused_attachments::NAME,
    ext::extended_dynamic_state3::NAME,
    khr::maintenance5::NAME,
    ext::conditional_rendering::NAME,
    ext::host_image_copy::NAME,
    ext::mesh_shader::NAME,
    ext::sample_locations::NAME,
    khr::calibrated_timestamps::NAME,
    ext::memory_budget::NAME,
    khr::external_memory_fd::NAME,
    khr::external_semaphore_fd::NAME,
    ext::external_memory_dma_buf::NAME,
    ext::shader_stencil_export::NAME,
    ext::pci_bus_info::NAME,
    ext::image_drm_format_modifier::NAME,
    khr::vertex_attribute_divisor::NAME,
    ext::depth_bias_control::NAME,
    ext::depth_clip_enable::NAME,
    khr::line_rasterization::NAME,
    ext::image_view_min_lod::NAME,
    ext::vertex_input_dynamic_state::NAME,
    ext::fragment_shader_interlock::NAME,
    khr::fragment_shader_barycentric::NAME,
    ext::shader_atomic_float::NAME,
    ext::shader_image_atomic_int64::NAME,
    khr::cooperative_matrix::NAME,
    khr::dynamic_rendering_local_read::NAME,
    nv::shader_atomic_float16_vector::NAME,
];

/// Extensions only probed: no Metal path uses them yet, so the device does
/// not enable them, but their features name what the driver could offer.
const PROBED: [&CStr; 4] = [
    ext::external_memory_host::NAME,
    khr::acceleration_structure::NAME,
    khr::ray_query::NAME,
    khr::ray_tracing_pipeline::NAME,
];

/// The wanted extensions the driver offers.
#[derive(Clone, Debug, Default)]
pub(super) struct Extensions {
    supported: Vec<&'static CStr>,
}

impl Extensions {
    pub(super) fn of(available: &[vk::ExtensionProperties]) -> Self {
        let offered = |wanted: &&CStr| {
            available
                .iter()
                .any(|extension| extension.extension_name_as_c_str() == Ok(*wanted))
        };
        Self {
            supported: ENABLED
                .iter()
                .chain(&PROBED)
                .copied()
                .filter(offered)
                .collect(),
        }
    }

    pub(super) fn has(&self, name: &CStr) -> bool {
        self.supported.contains(&name)
    }

    /// The names the device enables.
    pub(super) fn enabled(&self) -> Vec<*const core::ffi::c_char> {
        self.supported
            .iter()
            .filter(|name| !PROBED.contains(name))
            .map(|name| name.as_ptr())
            .collect()
    }
}

/// Every feature structure this backend reads, linked into one
/// `VkPhysicalDeviceFeatures2` chain of the supported extensions' structures.
/// It is boxed so the links stay valid.
#[derive(Default)]
pub(super) struct FeatureChain {
    pub(super) core: vk::PhysicalDeviceFeatures2<'static>,
    v11: vk::PhysicalDeviceVulkan11Features<'static>,
    v12: vk::PhysicalDeviceVulkan12Features<'static>,
    v13: vk::PhysicalDeviceVulkan13Features<'static>,
    maintenance5: vk::PhysicalDeviceMaintenance5FeaturesKHR<'static>,
    maintenance8: Maintenance8Features,
    unified: UnifiedImageLayoutsFeatures,
    robustness2: vk::PhysicalDeviceRobustness2FeaturesEXT<'static>,
    unused_attachments: vk::PhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT<'static>,
    dynamic_state3: vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT<'static>,
    conditional: vk::PhysicalDeviceConditionalRenderingFeaturesEXT<'static>,
    host_image_copy: vk::PhysicalDeviceHostImageCopyFeaturesEXT<'static>,
    mesh: vk::PhysicalDeviceMeshShaderFeaturesEXT<'static>,
    divisor: vk::PhysicalDeviceVertexAttributeDivisorFeaturesKHR<'static>,
    depth_bias: vk::PhysicalDeviceDepthBiasControlFeaturesEXT<'static>,
    depth_clip: vk::PhysicalDeviceDepthClipEnableFeaturesEXT<'static>,
    lines: vk::PhysicalDeviceLineRasterizationFeaturesKHR<'static>,
    view_min_lod: vk::PhysicalDeviceImageViewMinLodFeaturesEXT<'static>,
    vertex_input: vk::PhysicalDeviceVertexInputDynamicStateFeaturesEXT<'static>,
    interlock: vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT<'static>,
    barycentric: vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR<'static>,
    atomic_float: vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT<'static>,
    image_atomic_int64: vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT<'static>,
    cooperative_matrix: vk::PhysicalDeviceCooperativeMatrixFeaturesKHR<'static>,
    local_read: vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR<'static>,
    float16_vector: vk::PhysicalDeviceShaderAtomicFloat16VectorFeaturesNV<'static>,
    executable: vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR<'static>,
    generated: device_generated::Features,
    acceleration_structure: vk::PhysicalDeviceAccelerationStructureFeaturesKHR<'static>,
    ray_query: vk::PhysicalDeviceRayQueryFeaturesKHR<'static>,
    ray_tracing: vk::PhysicalDeviceRayTracingPipelineFeaturesKHR<'static>,
}

/// Appends `$field` to the chain ending at `$tail` when `$present`.
macro_rules! link {
    ($chain:ident, $tail:ident, $($field:ident if $present:expr;)*) => {$(
        if $present {
            let next = ptr::addr_of_mut!($chain.$field).cast::<c_void>();
            // SAFETY: `$tail` points at the `p_next` of a structure in the
            // same boxed chain, which outlives this call.
            unsafe { *$tail = next };
            $tail = ptr::addr_of_mut!($chain.$field.p_next);
        }
    )*};
}

impl FeatureChain {
    /// A chain of the structures the driver supports, all members false;
    /// `probe` also links the structures of probed-only extensions.
    pub(super) fn new(extensions: &Extensions, probe: bool) -> Box<Self> {
        let mut chain = Box::<Self>::default();
        let this = &mut *chain;
        let mut tail: *mut *mut c_void = ptr::addr_of_mut!(this.core.p_next);
        link!(this, tail,
            v11 if true;
            v12 if true;
            v13 if true;
            maintenance5 if extensions.has(khr::maintenance5::NAME);
            maintenance8 if extensions.has(MAINTENANCE8);
            unified if extensions.has(UNIFIED_IMAGE_LAYOUTS);
            robustness2 if extensions.has(ext::robustness2::NAME);
            unused_attachments if extensions.has(ext::dynamic_rendering_unused_attachments::NAME);
            dynamic_state3 if extensions.has(ext::extended_dynamic_state3::NAME);
            conditional if extensions.has(ext::conditional_rendering::NAME);
            host_image_copy if extensions.has(ext::host_image_copy::NAME);
            mesh if extensions.has(ext::mesh_shader::NAME);
            divisor if extensions.has(khr::vertex_attribute_divisor::NAME);
            depth_bias if extensions.has(ext::depth_bias_control::NAME);
            depth_clip if extensions.has(ext::depth_clip_enable::NAME);
            lines if extensions.has(khr::line_rasterization::NAME);
            view_min_lod if extensions.has(ext::image_view_min_lod::NAME);
            vertex_input if extensions.has(ext::vertex_input_dynamic_state::NAME);
            interlock if extensions.has(ext::fragment_shader_interlock::NAME);
            barycentric if extensions.has(khr::fragment_shader_barycentric::NAME);
            atomic_float if extensions.has(ext::shader_atomic_float::NAME);
            image_atomic_int64 if extensions.has(ext::shader_image_atomic_int64::NAME);
            cooperative_matrix if extensions.has(khr::cooperative_matrix::NAME);
            local_read if extensions.has(khr::dynamic_rendering_local_read::NAME);
            float16_vector if extensions.has(nv::shader_atomic_float16_vector::NAME);
            executable if extensions.has(khr::pipeline_executable_properties::NAME);
            generated if extensions.has(device_generated::NAME);
            acceleration_structure if probe && extensions.has(khr::acceleration_structure::NAME);
            ray_query if probe && extensions.has(khr::ray_query::NAME);
            ray_tracing if probe && extensions.has(khr::ray_tracing_pipeline::NAME);
        );
        let _ = tail;
        chain
    }

    /// Clears what the device must not enable: robust access, which Metal
    /// does not promise and which costs performance, capture-replay and
    /// multi-device addressing, and features whose prerequisites it lacks.
    pub(super) fn for_enabling(&mut self) {
        self.core.features.robust_buffer_access = vk::FALSE;
        self.v13.robust_image_access = vk::FALSE;
        self.robustness2.robust_buffer_access2 = vk::FALSE;
        self.robustness2.robust_image_access2 = vk::FALSE;
        self.v12.buffer_device_address_capture_replay = vk::FALSE;
        self.v12.buffer_device_address_multi_device = vk::FALSE;
        self.cooperative_matrix
            .cooperative_matrix_robust_buffer_access = vk::FALSE;
        // Needs `primitiveFragmentShadingRate`, which the device never enables.
        self.mesh.primitive_fragment_shading_rate_mesh_shader = vk::FALSE;
        // Only primary command buffers record.
        self.conditional.inherited_conditional_rendering = vk::FALSE;
        // Generated commands use the device's one pipeline layout.
        self.generated.dynamic_generated_pipeline_layout = vk::FALSE;
    }

    /// Whether the device enables device-generated commands.
    pub(super) fn generated_commands(&self) -> bool {
        self.generated.device_generated_commands == vk::TRUE
    }

    fn features(&self, extensions: &Extensions, api_version: u32) -> Features {
        let on = |value: vk::Bool32| value == vk::TRUE;
        let core = &self.core.features;
        let (v11, v12, v13) = (&self.v11, &self.v12, &self.v13);
        let dynamic_state3 = &self.dynamic_state3;
        let atomic_float = &self.atomic_float;
        let vulkan_1_3 = api_version >= vk::API_VERSION_1_3;
        Features {
            image_cube_array: on(core.image_cube_array),
            independent_blend: on(core.independent_blend),
            geometry_shader: on(core.geometry_shader),
            tessellation_shader: on(core.tessellation_shader),
            sample_rate_shading: on(core.sample_rate_shading),
            dual_src_blend: on(core.dual_src_blend),
            alpha_to_one: on(core.alpha_to_one),
            multi_draw_indirect: on(core.multi_draw_indirect),
            draw_indirect_first_instance: on(core.draw_indirect_first_instance),
            depth_clamp: on(core.depth_clamp),
            depth_bias_clamp: on(core.depth_bias_clamp),
            fill_mode_non_solid: on(core.fill_mode_non_solid),
            wide_lines: on(core.wide_lines),
            large_points: on(core.large_points),
            multi_viewport: on(core.multi_viewport),
            sampler_anisotropy: on(core.sampler_anisotropy),
            texture_compression_etc2: on(core.texture_compression_etc2),
            texture_compression_astc_ldr: on(core.texture_compression_astc_ldr),
            texture_compression_astc_hdr: on(v13.texture_compression_astc_hdr),
            texture_compression_bc: on(core.texture_compression_bc),
            // PVRTC is a format extension (`VK_IMG_format_pvrtc`), which no
            // Mesa driver for PC GPUs offers.
            texture_compression_pvrtc: false,
            occlusion_query_precise: on(core.occlusion_query_precise),
            pipeline_statistics_query: on(core.pipeline_statistics_query),
            vertex_pipeline_stores_and_atomics: on(core.vertex_pipeline_stores_and_atomics),
            fragment_stores_and_atomics: on(core.fragment_stores_and_atomics),
            shader_image_gather_extended: on(core.shader_image_gather_extended),
            shader_storage_image_multisample: on(core.shader_storage_image_multisample),
            shader_storage_image_read_without_format: on(
                core.shader_storage_image_read_without_format
            ),
            shader_storage_image_write_without_format: on(
                core.shader_storage_image_write_without_format
            ),
            shader_clip_distance: on(core.shader_clip_distance),
            shader_cull_distance: on(core.shader_cull_distance),
            shader_float64: on(core.shader_float64),
            shader_int64: on(core.shader_int64),
            shader_int16: on(core.shader_int16),
            shader_float16: on(v12.shader_float16),
            shader_int8: on(v12.shader_int8),
            shader_resource_residency: on(core.shader_resource_residency),
            shader_resource_min_lod: on(core.shader_resource_min_lod),
            sparse_binding: on(core.sparse_binding),
            sparse_residency_buffer: on(core.sparse_residency_buffer),
            sparse_residency_image_2d: on(core.sparse_residency_image2_d),
            sparse_residency_image_3d: on(core.sparse_residency_image3_d),
            sparse_residency_samples: SampleCounts::EMPTY
                .with(SampleCounts::C2, on(core.sparse_residency2_samples))
                .with(SampleCounts::C4, on(core.sparse_residency4_samples))
                .with(SampleCounts::C8, on(core.sparse_residency8_samples))
                .with(SampleCounts::C16, on(core.sparse_residency16_samples)),
            sparse_residency_aliased: on(core.sparse_residency_aliased),
            storage_buffer_8bit_access: on(v12.storage_buffer8_bit_access),
            storage_buffer_16bit_access: on(v11.storage_buffer16_bit_access),
            multiview: on(v11.multiview),
            sampler_mirror_clamp_to_edge: on(v12.sampler_mirror_clamp_to_edge),
            sampler_filter_minmax: on(v12.sampler_filter_minmax),
            descriptor_indexing: on(v12.descriptor_indexing),
            runtime_descriptor_array: on(v12.runtime_descriptor_array),
            descriptor_binding_partially_bound: on(v12.descriptor_binding_partially_bound),
            descriptor_binding_update_unused_while_pending: on(
                v12.descriptor_binding_update_unused_while_pending
            ),
            descriptor_binding_sampled_image_update_after_bind: on(
                v12.descriptor_binding_sampled_image_update_after_bind
            ),
            descriptor_binding_storage_image_update_after_bind: on(
                v12.descriptor_binding_storage_image_update_after_bind
            ),
            descriptor_binding_uniform_texel_buffer_update_after_bind: on(
                v12.descriptor_binding_uniform_texel_buffer_update_after_bind
            ),
            descriptor_binding_storage_texel_buffer_update_after_bind: on(
                v12.descriptor_binding_storage_texel_buffer_update_after_bind
            ),
            shader_sampled_image_array_non_uniform_indexing: on(
                v12.shader_sampled_image_array_non_uniform_indexing
            ),
            shader_storage_image_array_non_uniform_indexing: on(
                v12.shader_storage_image_array_non_uniform_indexing
            ),
            buffer_device_address: on(v12.buffer_device_address),
            timeline_semaphore: on(v12.timeline_semaphore),
            vulkan_memory_model: on(v12.vulkan_memory_model),
            vulkan_memory_model_device_scope: on(v12.vulkan_memory_model_device_scope),
            shader_buffer_int64_atomics: on(v12.shader_buffer_int64_atomics),
            shader_shared_int64_atomics: on(v12.shader_shared_int64_atomics),
            shader_image_int64_atomics: on(self.image_atomic_int64.shader_image_int64_atomics),
            shader_buffer_float32_atomic_add: on(atomic_float.shader_buffer_float32_atomic_add),
            shader_shared_float32_atomic_add: on(atomic_float.shader_shared_float32_atomic_add),
            shader_image_float32_atomics: on(atomic_float.shader_image_float32_atomics),
            shader_demote_to_helper_invocation: on(v13.shader_demote_to_helper_invocation),
            subgroup_size_control: on(v13.subgroup_size_control),
            compute_full_subgroups: on(v13.compute_full_subgroups),
            synchronization2: on(v13.synchronization2),
            dynamic_rendering: on(v13.dynamic_rendering),
            dynamic_rendering_local_read: on(self.local_read.dynamic_rendering_local_read),
            maintenance4: on(v13.maintenance4),
            maintenance5: on(self.maintenance5.maintenance5),
            maintenance8: on(self.maintenance8.maintenance8),
            host_image_copy: on(self.host_image_copy.host_image_copy),
            // Core, without feature bits, in Vulkan 1.3.
            extended_dynamic_state: vulkan_1_3,
            extended_dynamic_state2: vulkan_1_3,
            extended_dynamic_state3_polygon_mode: on(
                dynamic_state3.extended_dynamic_state3_polygon_mode
            ),
            extended_dynamic_state3_depth_clamp_enable: on(
                dynamic_state3.extended_dynamic_state3_depth_clamp_enable
            ),
            extended_dynamic_state3_depth_clip_enable: on(
                dynamic_state3.extended_dynamic_state3_depth_clip_enable
            ),
            vertex_input_dynamic_state: on(self.vertex_input.vertex_input_dynamic_state),
            vertex_attribute_instance_rate_divisor: on(self
                .divisor
                .vertex_attribute_instance_rate_divisor),
            depth_clip_enable: on(self.depth_clip.depth_clip_enable),
            bresenham_lines: on(self.lines.bresenham_lines),
            rectangular_lines: on(self.lines.rectangular_lines),
            image_view_min_lod: on(self.view_min_lod.min_lod),
            fragment_shader_barycentric: on(self.barycentric.fragment_shader_barycentric),
            task_shader: on(self.mesh.task_shader),
            mesh_shader: on(self.mesh.mesh_shader),
            fragment_shader_pixel_interlock: on(self.interlock.fragment_shader_pixel_interlock),
            unified_image_layouts: on(self.unified.unified_image_layouts),
            null_descriptor: on(self.robustness2.null_descriptor),
            dynamic_rendering_unused_attachments: on(self
                .unused_attachments
                .dynamic_rendering_unused_attachments),
            // A property; `probe` fills it.
            dynamic_primitive_topology_unrestricted: false,
            host_query_reset: on(v12.host_query_reset),
            conditional_rendering: on(self.conditional.conditional_rendering),
            image_drm_format_modifier: extensions.has(ext::image_drm_format_modifier::NAME),
            shader_float16_vector_atomics: on(self.float16_vector.shader_float16_vector_atomics),
            memory_budget: extensions.has(ext::memory_budget::NAME),
            calibrated_timestamps: extensions.has(khr::calibrated_timestamps::NAME),
            cooperative_matrix: on(self.cooperative_matrix.cooperative_matrix),
            acceleration_structure: on(self.acceleration_structure.acceleration_structure),
            ray_query: on(self.ray_query.ray_query),
            ray_tracing_pipeline: on(self.ray_tracing.ray_tracing_pipeline),
        }
    }
}

/// Keeps the bits an option set names; drivers may report newer ones.
macro_rules! known {
    ($set:ty, $bits:expr) => {
        <$set>::from_bits($bits & <$set>::KNOWN.bits()).unwrap_or(<$set>::EMPTY)
    };
}

/// The snapshot of `device`; `None` when its driver is not one Stela admits.
pub(super) fn probe(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    extensions: &Extensions,
) -> Option<PhysicalDevice> {
    let mut v11 = vk::PhysicalDeviceVulkan11Properties::default();
    let mut v12 = vk::PhysicalDeviceVulkan12Properties::default();
    let mut v13 = vk::PhysicalDeviceVulkan13Properties::default();
    let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
    let mut locations = vk::PhysicalDeviceSampleLocationsPropertiesEXT::default();
    let mut mesh = vk::PhysicalDeviceMeshShaderPropertiesEXT::default();
    let mut divisor = vk::PhysicalDeviceVertexAttributeDivisorPropertiesKHR::default();
    let mut dynamic_state3 = vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT::default();
    let mut generated = device_generated::Properties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default()
        .push_next(&mut v11)
        .push_next(&mut v12)
        .push_next(&mut v13);
    if extensions.has(ext::pci_bus_info::NAME) {
        properties = properties.push_next(&mut pci);
    }
    if extensions.has(ext::sample_locations::NAME) {
        properties = properties.push_next(&mut locations);
    }
    if extensions.has(ext::mesh_shader::NAME) {
        properties = properties.push_next(&mut mesh);
    }
    if extensions.has(khr::vertex_attribute_divisor::NAME) {
        properties = properties.push_next(&mut divisor);
    }
    if extensions.has(ext::extended_dynamic_state3::NAME) {
        properties = properties.push_next(&mut dynamic_state3);
    }
    if extensions.has(device_generated::NAME) {
        // The bindings cannot chain the structure, so it goes first by hand.
        generated.p_next = properties.p_next;
        properties.p_next = ptr::addr_of_mut!(generated).cast();
    }
    // SAFETY: `device` came from this instance and every chained structure
    // belongs to an extension the device supports.
    unsafe { instance.get_physical_device_properties2(device, &mut properties) };
    let core = properties.properties;

    let mut chain = FeatureChain::new(extensions, true);
    // SAFETY: as above; the boxed chain links only supported structures.
    unsafe { instance.get_physical_device_features2(device, &mut chain.core) };
    let mut features = chain.features(extensions, core.api_version);
    features.dynamic_primitive_topology_unrestricted =
        dynamic_state3.dynamic_primitive_topology_unrestricted == vk::TRUE;

    let driver_id = DriverId::from_raw(v12.driver_id.as_raw() as u32)?;
    let name = core.device_name_as_c_str().map_or_else(
        |_| String::new(),
        |name| name.to_string_lossy().into_owned(),
    );
    let conformance = v12.conformance_version;
    let identity = Identity {
        api_version: core.api_version,
        driver_id,
        driver_version: core.driver_version,
        vendor_id: core.vendor_id,
        device_id: core.device_id,
        pipeline_cache_uuid: core.pipeline_cache_uuid,
        device_type: PhysicalDeviceType::from_raw(core.device_type.as_raw() as u32)
            .unwrap_or(PhysicalDeviceType::Other),
        device_name: name,
        conformance_version: [
            conformance.major,
            conformance.minor,
            conformance.subminor,
            conformance.patch,
        ],
        pci_bus: extensions
            .has(ext::pci_bus_info::NAME)
            .then_some(PciBusInfo {
                domain: pci.pci_domain,
                bus: pci.pci_bus,
                device: pci.pci_device,
                function: pci.pci_function,
            }),
    };

    // Metal's sample positions are set per pass, which needs the dynamic
    // enable as well as the extension.
    let locations = (extensions.has(ext::sample_locations::NAME)
        && chain
            .dynamic_state3
            .extended_dynamic_state3_sample_locations_enable
            == vk::TRUE)
        .then_some(&locations);
    let divisor = extensions
        .has(khr::vertex_attribute_divisor::NAME)
        .then_some(divisor.max_vertex_attrib_divisor);
    let mut limits = limits(
        &core,
        (&v11, &v12, &v13),
        locations,
        features.mesh_shader.then_some(&mesh),
        divisor,
    );
    limits.generated_commands = chain
        .generated_commands()
        .then(|| device_generated::limits(&generated));
    limits.dma_buf_import = extensions.has(khr::external_memory_fd::NAME)
        && extensions.has(ext::external_memory_dma_buf::NAME);
    limits.stencil_export = extensions.has(ext::shader_stencil_export::NAME);
    Some(PhysicalDevice {
        identity,
        limits,
        memory: memory(instance, device, extensions),
        formats: formats(instance, device, &features),
        features,
    })
}

fn limits(
    core: &vk::PhysicalDeviceProperties,
    (v11, v12, v13): (
        &vk::PhysicalDeviceVulkan11Properties,
        &vk::PhysicalDeviceVulkan12Properties,
        &vk::PhysicalDeviceVulkan13Properties,
    ),
    locations: Option<&vk::PhysicalDeviceSampleLocationsPropertiesEXT>,
    mesh: Option<&vk::PhysicalDeviceMeshShaderPropertiesEXT>,
    max_vertex_attrib_divisor: Option<u32>,
) -> Limits {
    let limits = &core.limits;
    let samples = |flags: vk::SampleCountFlags| known!(SampleCounts, flags.as_raw());
    let resolve = |flags: vk::ResolveModeFlags| known!(ResolveModes, flags.as_raw());
    let on = |value: vk::Bool32| value == vk::TRUE;
    let sparse = &core.sparse_properties;
    Limits {
        max_image_dimension_1d: limits.max_image_dimension1_d,
        max_image_dimension_2d: limits.max_image_dimension2_d,
        max_image_dimension_3d: limits.max_image_dimension3_d,
        max_image_dimension_cube: limits.max_image_dimension_cube,
        max_image_array_layers: limits.max_image_array_layers,
        max_texel_buffer_elements: limits.max_texel_buffer_elements,
        max_uniform_buffer_range: limits.max_uniform_buffer_range,
        max_storage_buffer_range: limits.max_storage_buffer_range,
        max_push_constants_size: limits.max_push_constants_size,
        max_memory_allocation_count: limits.max_memory_allocation_count,
        max_sampler_allocation_count: limits.max_sampler_allocation_count,
        buffer_image_granularity: limits.buffer_image_granularity,
        sparse_address_space_size: limits.sparse_address_space_size,
        max_per_stage_update_after_bind_samplers: v12
            .max_per_stage_descriptor_update_after_bind_samplers,
        max_per_stage_update_after_bind_sampled_images: v12
            .max_per_stage_descriptor_update_after_bind_sampled_images,
        max_per_stage_update_after_bind_storage_images: v12
            .max_per_stage_descriptor_update_after_bind_storage_images,
        max_per_stage_update_after_bind_storage_buffers: v12
            .max_per_stage_descriptor_update_after_bind_storage_buffers,
        max_descriptor_set_update_after_bind_samplers: v12
            .max_descriptor_set_update_after_bind_samplers,
        max_descriptor_set_update_after_bind_sampled_images: v12
            .max_descriptor_set_update_after_bind_sampled_images,
        max_descriptor_set_update_after_bind_storage_images: v12
            .max_descriptor_set_update_after_bind_storage_images,
        max_vertex_input_attributes: limits.max_vertex_input_attributes,
        max_vertex_input_bindings: limits.max_vertex_input_bindings,
        max_vertex_input_attribute_offset: limits.max_vertex_input_attribute_offset,
        max_vertex_input_binding_stride: limits.max_vertex_input_binding_stride,
        max_vertex_output_components: limits.max_vertex_output_components,
        max_tessellation_generation_level: limits.max_tessellation_generation_level,
        max_tessellation_patch_size: limits.max_tessellation_patch_size,
        max_fragment_input_components: limits.max_fragment_input_components,
        max_fragment_output_attachments: limits.max_fragment_output_attachments,
        max_fragment_dual_src_attachments: limits.max_fragment_dual_src_attachments,
        max_compute_shared_memory_size: limits.max_compute_shared_memory_size,
        max_compute_work_group_count: limits.max_compute_work_group_count,
        max_compute_work_group_invocations: limits.max_compute_work_group_invocations,
        max_compute_work_group_size: limits.max_compute_work_group_size,
        max_sampler_lod_bias: limits.max_sampler_lod_bias,
        max_sampler_anisotropy: limits.max_sampler_anisotropy,
        max_viewports: limits.max_viewports,
        max_viewport_dimensions: limits.max_viewport_dimensions,
        min_texel_buffer_offset_alignment: limits.min_texel_buffer_offset_alignment,
        min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment,
        min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
        max_framebuffer_width: limits.max_framebuffer_width,
        max_framebuffer_height: limits.max_framebuffer_height,
        max_framebuffer_layers: limits.max_framebuffer_layers,
        framebuffer_color_sample_counts: samples(limits.framebuffer_color_sample_counts),
        framebuffer_depth_sample_counts: samples(limits.framebuffer_depth_sample_counts),
        framebuffer_stencil_sample_counts: samples(limits.framebuffer_stencil_sample_counts),
        framebuffer_no_attachments_sample_counts: samples(
            limits.framebuffer_no_attachments_sample_counts,
        ),
        max_color_attachments: limits.max_color_attachments,
        sampled_image_color_sample_counts: samples(limits.sampled_image_color_sample_counts),
        sampled_image_integer_sample_counts: samples(limits.sampled_image_integer_sample_counts),
        sampled_image_depth_sample_counts: samples(limits.sampled_image_depth_sample_counts),
        storage_image_sample_counts: samples(limits.storage_image_sample_counts),
        max_sample_mask_words: limits.max_sample_mask_words,
        timestamp_compute_and_graphics: on(limits.timestamp_compute_and_graphics),
        timestamp_period: limits.timestamp_period,
        max_clip_distances: limits.max_clip_distances,
        max_cull_distances: limits.max_cull_distances,
        point_size_range: limits.point_size_range,
        line_width_range: limits.line_width_range,
        non_coherent_atom_size: limits.non_coherent_atom_size,
        subgroup_size: v11.subgroup_size,
        min_subgroup_size: v13.min_subgroup_size,
        max_subgroup_size: v13.max_subgroup_size,
        subgroup_supported_stages: known!(ShaderStages, v11.subgroup_supported_stages.as_raw()),
        subgroup_supported_operations: known!(
            SubgroupFeatures,
            v11.subgroup_supported_operations.as_raw()
        ),
        subgroup_quad_operations_in_all_stages: on(v11.subgroup_quad_operations_in_all_stages),
        max_multiview_view_count: v11.max_multiview_view_count,
        max_memory_allocation_size: v11.max_memory_allocation_size,
        max_buffer_size: v13.max_buffer_size,
        storage_texel_buffer_offset_alignment_bytes: v13
            .storage_texel_buffer_offset_alignment_bytes,
        storage_texel_buffer_offset_single_texel_alignment: on(
            v13.storage_texel_buffer_offset_single_texel_alignment
        ),
        uniform_texel_buffer_offset_alignment_bytes: v13
            .uniform_texel_buffer_offset_alignment_bytes,
        uniform_texel_buffer_offset_single_texel_alignment: on(
            v13.uniform_texel_buffer_offset_single_texel_alignment
        ),
        supported_depth_resolve_modes: resolve(v12.supported_depth_resolve_modes),
        supported_stencil_resolve_modes: resolve(v12.supported_stencil_resolve_modes),
        independent_resolve_none: on(v12.independent_resolve_none),
        independent_resolve: on(v12.independent_resolve),
        // Without the extension, instance-rate bindings step every instance.
        max_vertex_attrib_divisor: max_vertex_attrib_divisor.unwrap_or(1),
        sample_locations: locations.map(|locations| SampleLocationLimits {
            sample_counts: samples(locations.sample_location_sample_counts),
            max_grid_size: [
                locations.max_sample_location_grid_size.width,
                locations.max_sample_location_grid_size.height,
            ],
            coordinate_range: locations.sample_location_coordinate_range,
            sub_pixel_bits: locations.sample_location_sub_pixel_bits,
            variable_sample_locations: on(locations.variable_sample_locations),
        }),
        mesh: mesh.map(|mesh| MeshLimits {
            max_task_work_group_total_count: mesh.max_task_work_group_total_count,
            max_task_work_group_count: mesh.max_task_work_group_count,
            max_task_work_group_invocations: mesh.max_task_work_group_invocations,
            max_task_payload_size: mesh.max_task_payload_size,
            max_mesh_work_group_total_count: mesh.max_mesh_work_group_total_count,
            max_mesh_work_group_count: mesh.max_mesh_work_group_count,
            max_mesh_work_group_invocations: mesh.max_mesh_work_group_invocations,
            max_mesh_output_vertices: mesh.max_mesh_output_vertices,
            max_mesh_output_primitives: mesh.max_mesh_output_primitives,
        }),
        generated_commands: None,
        dma_buf_import: false,
        stencil_export: false,
        sparse: SparseLimits {
            residency_standard_2d_block_shape: on(sparse.residency_standard2_d_block_shape),
            residency_standard_2d_multisample_block_shape: on(
                sparse.residency_standard2_d_multisample_block_shape
            ),
            residency_standard_3d_block_shape: on(sparse.residency_standard3_d_block_shape),
            residency_aligned_mip_size: on(sparse.residency_aligned_mip_size),
            residency_non_resident_strict: on(sparse.residency_non_resident_strict),
        },
    }
}

fn memory(instance: &ash::Instance, device: vk::PhysicalDevice, extensions: &Extensions) -> Memory {
    let has_budget = extensions.has(ext::memory_budget::NAME);
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceMemoryProperties2::default();
    if has_budget {
        properties = properties.push_next(&mut budget);
    }
    // SAFETY: `device` came from this instance; the budget structure is
    // chained only when its extension is supported.
    unsafe { instance.get_physical_device_memory_properties2(device, &mut properties) };
    let properties = properties.memory_properties;
    Memory {
        heaps: properties
            .memory_heaps_as_slice()
            .iter()
            .enumerate()
            .map(|(index, heap)| MemoryHeap {
                size: heap.size,
                flags: known!(MemoryHeapFlags, heap.flags.as_raw()),
                budget: has_budget.then_some(budget.heap_budget[index]),
            })
            .collect(),
        types: properties
            .memory_types_as_slice()
            .iter()
            .map(|memory| MemoryType {
                properties: known!(MemoryProperties, memory.property_flags.as_raw()),
                heap_index: memory.heap_index,
            })
            .collect(),
    }
}

/// The features of `format`'s images of the linear DRM format modifier.
fn linear_modifier_features(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    format: vk::Format,
) -> FormatFeatures {
    const LINEAR: u64 = 0;
    let count = {
        let mut list = vk::DrmFormatModifierPropertiesList2EXT::default();
        let mut properties = vk::FormatProperties2::default().push_next(&mut list);
        // SAFETY: `device` came from this instance, which enabled the
        // modifier extension when the device has it.
        unsafe { instance.get_physical_device_format_properties2(device, format, &mut properties) };
        list.drm_format_modifier_count
    };
    let mut modifiers = alloc::vec![vk::DrmFormatModifierProperties2EXT::default(); count as usize];
    {
        let mut list = vk::DrmFormatModifierPropertiesList2EXT::default()
            .drm_format_modifier_properties(&mut modifiers);
        let mut properties = vk::FormatProperties2::default().push_next(&mut list);
        // SAFETY: as above; the list holds the count the driver reported.
        unsafe { instance.get_physical_device_format_properties2(device, format, &mut properties) };
    }
    modifiers
        .iter()
        .find(|modifier| modifier.drm_format_modifier == LINEAR)
        .map_or(FormatFeatures::EMPTY, |modifier| {
            known!(
                FormatFeatures,
                modifier.drm_format_modifier_tiling_features.as_raw()
            )
        })
}

/// Support for every format the Metal layer can request.
fn formats(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    features: &Features,
) -> FormatTable {
    let mut table = FormatTable::default();
    for &format in Format::ALL {
        let vulkan = vk::Format::from_raw(format.raw() as i32);
        let mut flags = vk::FormatProperties3::default();
        let mut properties = vk::FormatProperties2::default().push_next(&mut flags);
        // SAFETY: `device` came from this instance; `VkFormatProperties3` is
        // core in Vulkan 1.3.
        unsafe { instance.get_physical_device_format_properties2(device, vulkan, &mut properties) };
        let optimal = known!(FormatFeatures, flags.optimal_tiling_features.as_raw());
        let usage = if optimal.contains(FormatFeatures::COLOR_ATTACHMENT) {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        } else if optimal.contains(FormatFeatures::DEPTH_STENCIL_ATTACHMENT) {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            vk::ImageUsageFlags::SAMPLED
        };
        let info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(vulkan)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage);
        let mut image = vk::ImageFormatProperties2::default();
        // SAFETY: as above.
        let queried = unsafe {
            instance.get_physical_device_image_format_properties2(device, &info, &mut image)
        };
        let sample_counts = match queried {
            Ok(()) => known!(
                SampleCounts,
                image.image_format_properties.sample_counts.as_raw()
            ),
            Err(_) => SampleCounts::EMPTY,
        };
        let sparse_counts = if features.sparse_residency_image_2d {
            SampleCounts::C1.union(features.sparse_residency_samples)
        } else {
            SampleCounts::EMPTY
        };
        let sparse = (0..u32::BITS)
            .filter_map(|bit| SampleCounts::single(1 << bit))
            .filter(|&count| sample_counts.intersection(sparse_counts).contains(count))
            .filter(|count| {
                let info = vk::PhysicalDeviceSparseImageFormatInfo2::default()
                    .format(vulkan)
                    .ty(vk::ImageType::TYPE_2D)
                    .samples(vk::SampleCountFlags::from_raw(count.bits()))
                    .usage(vk::ImageUsageFlags::SAMPLED)
                    .tiling(vk::ImageTiling::OPTIMAL);
                // SAFETY: as above; the device supports sparse residency at
                // this count.
                let properties = unsafe {
                    instance.get_physical_device_sparse_image_format_properties2_len(device, &info)
                };
                properties > 0
            })
            .fold(SampleCounts::EMPTY, SampleCounts::union);
        table.insert(
            format,
            FormatSupport {
                optimal,
                linear: known!(FormatFeatures, flags.linear_tiling_features.as_raw()),
                modifier_linear: if features.image_drm_format_modifier {
                    linear_modifier_features(instance, device, vulkan)
                } else {
                    FormatFeatures::EMPTY
                },
                buffer: known!(FormatFeatures, flags.buffer_features.as_raw()),
                sample_counts,
                sparse,
            },
        );
    }
    table
}
