//! Metal device capabilities derived from the Vulkan device, the shader
//! compiler and this Metal implementation.
//!
//! A capability is advertised only when all three provide it: the driver
//! reports the Vulkan feature, limit or format support; the AIR compiler
//! lowers the shader constructs; and this layer implements the API path.
//! Each refusal is recorded once as a [`Gap`] naming the layer that lacks it,
//! so negative answers stay explicit. A GPU family is the conjunction of its
//! requirements. Apple1 to Apple7 are claimed, with their formats,
//! programmable blending, imageblocks and tile functions, as macOS's
//! compositor requires Apple3 of its display surfaces and every Mac that runs
//! macOS 27 has an Apple7 GPU. Metal 3 is claimed without ray tracing, which
//! `supportsRaytracing` refuses; Metal 4 and Apple8 and later, which
//! presuppose Apple's architecture, are not.
use super::{
    abi::{
        ArgumentBuffersTier, CounterSamplingPoint, DeviceLocation, GpuFamily, ReadWriteTextureTier,
        SparsePageSize, TextureType,
    },
    pixel_format::{Compression, Description, Encoding, Kind, PixelFormat, Planes, Realization},
    resource::{raw_rendered, raw_texel_format},
    vertex::VertexFormat,
    vulkan::{
        FormatFeatures, PhysicalDeviceType, ResolveModes, SampleCounts,
        physical::{Features, FormatSupport, PhysicalDevice},
    },
};
use alloc::{boxed::Box, string::String, vec::Vec};

pub mod block;

option_set! {
    /// Shader constructs the AIR → NIR → vendor compiler path lowers with
    /// complete semantics. The compiler reports its own set.
    pub struct ShaderFeatures: u64 {
        const COMPUTE = 1 << 0;
        const VERTEX_FRAGMENT = 1 << 1;
        const TEXTURE_SAMPLE = 1 << 2;
        const TEXTURE_READ_WRITE = 1 << 3;
        const TEXTURE_ATOMICS = 1 << 4;
        const BUFFER_ATOMICS = 1 << 5;
        const THREADGROUP_MEMORY = 1 << 6;
        const SIMDGROUP = 1 << 7;
        const QUADGROUP = 1 << 8;
        const SIMDGROUP_MATRIX = 1 << 9;
        const ARGUMENT_BUFFERS = 1 << 10;
        const TESSELLATION = 1 << 11;
        const MESH = 1 << 12;
        const FUNCTION_POINTERS = 1 << 13;
        const DYNAMIC_LIBRARIES = 1 << 14;
        const RAY_TRACING = 1 << 15;
        const BARYCENTRICS = 1 << 16;
        const PULL_MODEL_INTERPOLATION = 1 << 17;
        const QUERY_LOD = 1 << 18;
        const RASTER_ORDER_GROUPS = 1 << 19;
        const LAYERED_RENDERING = 1 << 20;
        const VERTEX_AMPLIFICATION = 1 << 21;
        const DEPTH_COMPARE_SAMPLE = 1 << 22;
        /// Kernels encode indirect commands (`command_buffer`,
        /// `render_command`, `compute_command`).
        const INDIRECT_COMMAND_ENCODING = 1 << 23;
        /// Sparse samples, reads and gathers report whether their texels
        /// are resident (`sparse_color`).
        const SPARSE_FEEDBACK = 1 << 24;
        /// Imageblocks, tile functions and their threadgroup memory.
        const IMAGEBLOCKS = 1 << 25;
    }
}

option_set! {
    /// API paths a Metal implementation realizes on its Vulkan backend.
    pub struct ApiFeatures: u64 {
        const HEAPS = 1 << 0;
        const INDIRECT_DRAW_DISPATCH = 1 << 1;
        const INDIRECT_COMMAND_BUFFERS = 1 << 2;
        const SPARSE_TEXTURES = 1 << 3;
        const TIMESTAMP_COUNTERS = 1 << 4;
        const STATISTIC_COUNTERS = 1 << 5;
        const PROGRAMMABLE_SAMPLE_POSITIONS = 1 << 6;
        const SHARED_EVENTS = 1 << 7;
        const IO_COMMAND_QUEUES = 1 << 8;
        const METALFX = 1 << 9;
        /// Tessellation pipelines and `drawPatches`.
        const TESSELLATION = 1 << 10;
    }
}

impl ApiFeatures {
    /// The paths this build implements.
    pub const IMPLEMENTED: Self = Self::HEAPS
        .union(Self::INDIRECT_DRAW_DISPATCH)
        .union(Self::INDIRECT_COMMAND_BUFFERS)
        .union(Self::SPARSE_TEXTURES)
        .union(Self::TIMESTAMP_COUNTERS)
        .union(Self::STATISTIC_COUNTERS)
        .union(Self::PROGRAMMABLE_SAMPLE_POSITIONS)
        .union(Self::SHARED_EVENTS)
        .union(Self::IO_COMMAND_QUEUES)
        .union(Self::METALFX)
        .union(Self::TESSELLATION);
}

/// Device facts outside Vulkan: where the GPU sits and what its vendor
/// backend knows about memory layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Platform {
    /// `MTLArchitecture.name`: the GPU's canonical chip name.
    pub architecture: String,
    pub location: DeviceLocation,
    pub location_number: u64,
    /// Host link bandwidth in bytes per second; zero when built in.
    pub max_transfer_rate: u64,
    /// No display engine is attached to this GPU.
    pub headless: bool,
    pub removable: bool,
    /// Row-pitch alignment of explicitly laid out linear images.
    pub linear_row_alignment: u64,
    /// Memory offset alignment for binding a linear image inside a buffer.
    pub linear_offset_alignment: u64,
    /// The backend's sparse page size, which is also its tile size.
    pub sparse_page_bytes: u64,
}

/// One capability this device does not provide and the layer that lacks it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gap {
    pub capability: String,
    pub cause: Cause,
}

raw_enum! {
    pub enum Cause: u32 {
        /// The Vulkan driver does not report the feature, limit or format.
        Driver = 0,
        /// The shader compiler does not lower the required constructs.
        Compiler = 1,
        /// This Metal implementation has no path for the API.
        Api = 2,
        /// The capability presupposes an Apple GPU architecture.
        AppleArchitecture = 3,
    }
}

option_set! {
    /// Per-pixel-format capabilities in the terms of Apple's feature tables.
    pub struct FormatCapabilities: u32 {
        /// Sampled or read by shaders.
        const READ = 1 << 0;
        const FILTER = 1 << 1;
        const WRITE = 1 << 2;
        /// Read and written by one function (read-write tier 2).
        const READ_WRITE = 1 << 3;
        const COLOR = 1 << 4;
        const BLEND = 1 << 5;
        const DEPTH_STENCIL = 1 << 6;
        const MSAA = 1 << 7;
        const RESOLVE = 1 << 8;
        const SPARSE = 1 << 9;
        const ATOMIC = 1 << 10;
        const TEXTURE_BUFFER_READ = 1 << 11;
        const TEXTURE_BUFFER_WRITE = 1 << 12;
        const TEXTURE_BUFFER_READ_WRITE = 1 << 13;
        /// Stored linearly inside a buffer.
        const LINEAR = 1 << 14;
        const COMPARE_SAMPLE = 1 << 15;
        /// Source and destination of blit copies.
        const BLIT = 1 << 16;
        /// Copied between host memory and the texture without a GPU pass
        /// (`VK_EXT_host_image_copy`).
        const HOST_TRANSFER = 1 << 17;
        /// Not Apple's: attachments render through the image's raw texels
        /// ([`resource::raw_rendered`](super::resource::raw_rendered)), as
        /// the driver renders no view of its format.
        const RAW_RENDER = 1 << 18;
    }
}

#[derive(Clone, Debug, PartialEq)]
enum PixelFormats {
    /// The capabilities of every known format, in `PixelFormat::ALL` order.
    Indexed(Box<IndexedPixelFormats>),
    /// A decoded list with its original omissions, order and duplicates.
    Listed(Box<[(PixelFormat, FormatCapabilities)]>),
}

type IndexedPixelFormats = [FormatCapabilities; PixelFormat::ALL.len()];

fn indexed_format(formats: &IndexedPixelFormats, format: PixelFormat) -> FormatCapabilities {
    formats[format.index()]
}

impl PixelFormats {
    fn get(&self, format: PixelFormat) -> FormatCapabilities {
        match self {
            Self::Indexed(formats) => indexed_format(formats, format),
            Self::Listed(formats) => formats
                .iter()
                .find(|(known, _)| *known == format)
                .map_or(FormatCapabilities::EMPTY, |(_, capabilities)| *capabilities),
        }
    }
}

/// `MTLSize`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Size {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

/// Metal's per-stage binding table sizes on Mac GPUs.
pub const MAX_BUFFER_BINDINGS: u32 = 31;
pub const MAX_TEXTURE_BINDINGS: u32 = 128;
pub const MAX_SAMPLER_BINDINGS: u32 = 16;
pub const MAX_VERTEX_ATTRIBUTES: u32 = 31;
/// Argument buffer tier 2 resource and sampler counts on Mac GPUs.
const TIER2_ARGUMENT_RESOURCES: u32 = 500_000;
const TIER2_ARGUMENT_SAMPLERS: u32 = 2048;
/// The widest point an Apple GPU draws, Metal feature set tables.
const APPLE_MAX_POINT_SIZE: f32 = 511.0;
/// Vulkan's standard sparse block shapes describe 64 KiB tiles.
const STANDARD_SPARSE_TILE_BYTES: u64 = 64 << 10;

#[derive(Clone, Debug, PartialEq)]
pub struct Limits {
    pub max_threads_per_threadgroup: Size,
    pub max_total_threads_per_threadgroup: u64,
    pub thread_execution_width: u64,
    pub max_threadgroup_memory_length: u64,
    pub max_buffer_length: u64,
    pub max_texture_dimension_1d: u64,
    pub max_texture_dimension_2d: u64,
    pub max_texture_dimension_3d: u64,
    pub max_texture_dimension_cube: u64,
    pub max_texture_array_length: u64,
    pub max_texture_buffer_width: u64,
    pub max_color_attachments: u64,
    pub max_viewports: u64,
    pub max_vertex_amplification_count: u64,
    pub max_argument_buffer_sampler_count: u64,
    pub max_argument_buffer_resources: u64,
    pub max_point_size: f32,
    pub max_vertex_buffer_stride: u64,
    pub max_vertex_attribute_offset: u64,
    /// Largest per-instance step rate; 1 without instance-rate divisors.
    pub max_vertex_step_rate: u64,
    pub max_tessellation_factor: u64,
    pub max_mesh_threads_per_threadgroup: u64,
    pub max_object_payload_length: u64,
    pub buffer_alignment_for_uniforms: u64,
    pub sparse_tile_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TexelBufferAlignment {
    bytes: u64,
    single_texel: bool,
}

impl TexelBufferAlignment {
    fn for_texel(self, texel_bytes: u64) -> u64 {
        if self.single_texel && texel_bytes.is_power_of_two() {
            texel_bytes.min(self.bytes)
        } else {
            self.bytes
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Capabilities {
    pub name: String,
    pub architecture: String,
    pub location: DeviceLocation,
    pub location_number: u64,
    pub max_transfer_rate: u64,
    pub headless: bool,
    pub low_power: bool,
    pub removable: bool,
    pub has_unified_memory: bool,
    pub recommended_max_working_set_size: u64,
    pub limits: Limits,
    pub argument_buffers_tier: ArgumentBuffersTier,
    pub read_write_texture_tier: ReadWriteTextureTier,
    pub supports_32bit_float_filtering: bool,
    pub supports_32bit_msaa: bool,
    pub supports_bc_texture_compression: bool,
    pub depth24_stencil8_supported: bool,
    pub supports_query_texture_lod: bool,
    pub supports_pull_model_interpolation: bool,
    pub supports_shader_barycentric_coordinates: bool,
    pub programmable_sample_positions: bool,
    pub raster_order_groups: bool,
    pub supports_raytracing: bool,
    pub supports_raytracing_from_render: bool,
    pub supports_primitive_motion_blur: bool,
    pub supports_function_pointers: bool,
    pub supports_function_pointers_from_render: bool,
    pub supports_dynamic_libraries: bool,
    pub supports_render_dynamic_libraries: bool,
    /// `supportsPlacementSparse`: a Metal 4 capability, never claimed.
    pub supports_placement_sparse: bool,
    pub supports_mesh_shaders: bool,
    pub supports_tessellation: bool,
    /// Depth resolve filters beyond sample 0 (Metal's min and max).
    pub depth_resolve_modes: ResolveModes,
    pub stencil_resolve_modes: ResolveModes,
    pub texture_sample_counts: SampleCounts,
    pub max_sampler_anisotropy: f32,
    pub sampler_mirror_clamp_to_edge: bool,
    pub sampler_reduction_modes: bool,
    /// Blend factors reading a second fragment output.
    pub dual_source_blending: bool,
    pub alpha_to_one: bool,
    /// `MTLDepthClipModeClamp`.
    pub depth_clamp: bool,
    /// `MTLTriangleFillModeLines`.
    pub wireframe: bool,
    /// Lines and polygon edges rasterize as Apple GPUs': by the diamond
    /// exit rule at one sample, as rectangles under multisampling.
    pub line_modes: bool,
    /// `MTLTextureViewDescriptor.minLOD`.
    pub view_min_lod: bool,
    /// Timestamp ticks per second; zero without timestamp counters.
    pub timestamp_frequency: u64,
    /// `MTLCommonCounterSetStatistic` is among `counterSets`.
    pub statistic_counters: bool,
    /// `supportsMemorylessRenderTargets`: memoryless textures, backed by
    /// device memory that no pass loads or stores.
    pub memoryless_render_targets: bool,
    /// `isFramebufferReadSupported`: `[[color(n)]]` reads of any color
    /// attachment, ordered among the overlapping fragments of a draw, which
    /// read and write the attachment as storage in a fragment shader
    /// interlock section.
    pub framebuffer_reads: bool,
    /// Fragment functions can run the fragment tests themselves, on copies
    /// of the pass's depth and stencil attachments in their interlock
    /// section: those that write the stencil reference on a device without
    /// stencil export, and those accessing attachments as storage whose
    /// pipeline cannot test fragments before they run.
    pub fragment_tests: bool,
    /// Fragment shaders write the stencil reference
    /// (`VK_EXT_shader_stencil_export`).
    pub stencil_export: bool,
    /// A stencil resolve following a min or max depth resolve
    /// (`MTLMultisampleStencilResolveFilterDepthResolvedSample`), which
    /// Vulkan does not have: the average samples kernel resolves the
    /// stencil from copies of the attachments' samples where the pass ends,
    /// and Vulkan's rendering resolves the depth alone.
    pub stencil_follows_depth: bool,
    counter_sampling: [bool; 5],
    /// Sparse 3D textures, with Vulkan's 3D residency and block shapes.
    sparse_volumes: bool,
    /// The sample counts of each format's sparse 2D textures.
    sparse_samples: Vec<(PixelFormat, SampleCounts)>,
    vertex_formats: Vec<VertexFormat>,
    families: Vec<GpuFamily>,
    formats: PixelFormats,
    gaps: Vec<Gap>,
    linear_row_alignment: u64,
    linear_offset_alignment: u64,
    texel_buffer_alignment: TexelBufferAlignment,
}

impl Capabilities {
    pub fn derive(
        device: &PhysicalDevice,
        platform: &Platform,
        shaders: ShaderFeatures,
        api: ApiFeatures,
    ) -> Self {
        Derivation {
            device,
            platform,
            shaders,
            api,
            gaps: Vec::new(),
        }
        .run()
    }

    #[cfg(any(test, feature = "metal-linked"))]
    pub(super) fn hosted_lavapipe(device: &PhysicalDevice) -> super::error::Result<Self> {
        let platform = Platform {
            architecture: String::from("llvmpipe"),
            location: DeviceLocation::Unspecified,
            location_number: 0,
            max_transfer_rate: 0,
            headless: true,
            removable: false,
            linear_row_alignment: 4,
            linear_offset_alignment: 4,
            sparse_page_bytes: 0,
        };
        let mut capabilities = Self::derive(
            device,
            &platform,
            ShaderFeatures::COMPUTE | ShaderFeatures::VERTEX_FRAGMENT,
            ApiFeatures::EMPTY,
        );
        let qualified = FormatCapabilities::COLOR | FormatCapabilities::BLIT;
        let formats = [PixelFormat::Rgba8Unorm, PixelFormat::Bgra8Unorm];
        super::error::require(
            capabilities.supports_texture_sample_count(1)
                && formats.iter().all(|&format| {
                    let supported = capabilities.format(format);
                    supported.contains(qualified)
                        && !supported.contains(FormatCapabilities::RAW_RENDER)
                }),
            "single-sample RGBA8 and BGRA8 attachments for hosted lavapipe",
        )?;
        capabilities.formats =
            PixelFormats::Listed(Box::new(formats.map(|format| (format, qualified))));
        capabilities.families.clear();
        capabilities.vertex_formats.clear();
        capabilities.texture_sample_counts = SampleCounts::C1;
        capabilities.read_write_texture_tier = ReadWriteTextureTier::None;
        capabilities.supports_32bit_float_filtering = false;
        capabilities.supports_32bit_msaa = false;
        capabilities.supports_bc_texture_compression = false;
        capabilities.depth24_stencil8_supported = false;
        capabilities.depth_resolve_modes = ResolveModes::EMPTY;
        capabilities.stencil_resolve_modes = ResolveModes::EMPTY;
        capabilities.max_sampler_anisotropy = 1.0;
        capabilities.sampler_mirror_clamp_to_edge = false;
        capabilities.sampler_reduction_modes = false;
        capabilities.dual_source_blending = false;
        capabilities.alpha_to_one = false;
        capabilities.depth_clamp = false;
        capabilities.wireframe = false;
        capabilities.line_modes = false;
        capabilities.view_min_lod = false;
        capabilities.memoryless_render_targets = false;
        capabilities.stencil_export = false;
        capabilities.stencil_follows_depth = false;
        capabilities.sparse_volumes = false;
        capabilities.sparse_samples.clear();
        let limits = &mut capabilities.limits;
        limits.max_texture_dimension_1d = 0;
        limits.max_texture_dimension_3d = 0;
        limits.max_texture_dimension_cube = 0;
        limits.max_texture_array_length = 1;
        limits.max_texture_buffer_width = 0;
        limits.max_color_attachments = 1;
        limits.max_viewports = 1;
        limits.max_vertex_amplification_count = 1;
        // The wire has no argument buffer tier below Tier1. Zero limits
        // and refusal of argument encoders describe this profile's support.
        limits.max_argument_buffer_sampler_count = 0;
        limits.max_argument_buffer_resources = 0;
        limits.max_threadgroup_memory_length = 0;
        capabilities.gaps.push(Gap {
            capability: String::from(
                "hosted lavapipe exposes only buffer compute and single-sample color rendering",
            ),
            cause: Cause::Api,
        });
        Ok(capabilities)
    }

    pub fn supports_family(&self, family: GpuFamily) -> bool {
        self.families.contains(&family)
    }

    pub fn families(&self) -> &[GpuFamily] {
        &self.families
    }

    /// `supportsFeatureSet:` for the deprecated macOS feature sets; iOS and
    /// tvOS sets and unknown values are unsupported.
    pub fn supports_feature_set(&self, raw: u64) -> bool {
        const MACOS_GPU_FAMILY1_V1: u64 = 10000;
        const MACOS_GPU_FAMILY1_V4: u64 = 10004;
        const MACOS_READ_WRITE_TEXTURE_TIER2: u64 = 10002;
        const MACOS_GPU_FAMILY2_V1: u64 = 10005;
        match raw {
            MACOS_READ_WRITE_TEXTURE_TIER2 => {
                self.read_write_texture_tier == ReadWriteTextureTier::Tier2
            }
            MACOS_GPU_FAMILY1_V1..=MACOS_GPU_FAMILY1_V4 => self.supports_family(GpuFamily::Mac1),
            MACOS_GPU_FAMILY2_V1 => self.supports_family(GpuFamily::Mac2),
            _ => false,
        }
    }

    /// Capabilities of `format`; empty for formats this device cannot store.
    pub fn format(&self, format: PixelFormat) -> FormatCapabilities {
        self.formats.get(format)
    }

    /// Every refused capability, each listed once.
    pub fn gaps(&self) -> &[Gap] {
        &self.gaps
    }

    pub fn supports_texture_sample_count(&self, count: u64) -> bool {
        u32::try_from(count)
            .ok()
            .and_then(SampleCounts::single)
            .is_some_and(|flag| self.texture_sample_counts.contains(flag))
    }

    pub fn supports_vertex_format(&self, format: VertexFormat) -> bool {
        self.vertex_formats.contains(&format)
    }

    pub fn supports_vertex_amplification_count(&self, count: u64) -> bool {
        (1..=self.limits.max_vertex_amplification_count).contains(&count)
    }

    pub fn supports_counter_sampling(&self, point: CounterSamplingPoint) -> bool {
        self.counter_sampling[point.raw() as usize]
    }

    /// Rasterization rate maps of one or two layers, as Apple's: each renders
    /// every region at full quality, which Metal allows as it may round a
    /// layer's samples up, so a pass with one renders as without.
    pub fn supports_rasterization_rate_map(&self, layer_count: u64) -> bool {
        (1..=2).contains(&layer_count)
    }

    /// `sparseTileSizeInBytesForSparsePageSize:`; `None` for page sizes the
    /// backend does not use.
    pub fn sparse_tile_bytes(&self, page: SparsePageSize) -> Option<u64> {
        let bytes = self.limits.sparse_tile_bytes;
        (bytes != 0 && page.bytes() == bytes).then_some(bytes)
    }

    /// Texels per sparse tile, following Vulkan's standard block shapes:
    /// 2D textures and arrays at the sample counts the driver has residency
    /// for, and single-sample 3D textures where it has 3D residency.
    pub fn sparse_tile_size(
        &self,
        texture_type: TextureType,
        format: PixelFormat,
        samples: u64,
    ) -> Option<Size> {
        let count = u32::try_from(samples).ok().and_then(SampleCounts::single)?;
        let single = count == SampleCounts::C1;
        let shape = match texture_type {
            TextureType::D2 | TextureType::D2Array if single => Shape::Planar(count),
            TextureType::D2Multisample | TextureType::D2MultisampleArray if !single => {
                Shape::Planar(count)
            }
            TextureType::D3 if single && self.sparse_volumes => Shape::Volume,
            _ => return None,
        };
        let counts = self
            .sparse_samples
            .iter()
            .find(|(known, _)| *known == format)
            .map_or(SampleCounts::EMPTY, |(_, counts)| *counts);
        if !counts.contains(count) || self.limits.sparse_tile_bytes != STANDARD_SPARSE_TILE_BYTES {
            return None;
        }
        standard_tile_shape(shape, format.describe())
    }

    /// `minimumLinearTextureAlignmentForPixelFormat:`; zero when the format
    /// cannot be stored linearly.
    pub fn minimum_linear_texture_alignment(&self, format: PixelFormat) -> u64 {
        if self.format(format).contains(FormatCapabilities::LINEAR) {
            self.linear_row_alignment.max(self.linear_offset_alignment)
        } else {
            0
        }
    }

    /// Row-pitch and offset alignments of linear textures inside buffers.
    /// Metal reports their maximum; creation checks each separately.
    pub fn linear_texture_alignments(&self) -> (u64, u64) {
        (self.linear_row_alignment, self.linear_offset_alignment)
    }

    /// `minimumTextureBufferAlignmentForPixelFormat:`; zero when the format
    /// has no texture-buffer use.
    pub fn minimum_texture_buffer_alignment(&self, format: PixelFormat) -> u64 {
        let buffer =
            FormatCapabilities::TEXTURE_BUFFER_READ | FormatCapabilities::TEXTURE_BUFFER_WRITE;
        if !self.format(format).intersects(buffer) {
            return 0;
        }
        self.texel_buffer_alignment
            .for_texel(u64::from(format.describe().block.bytes))
    }
}

/// The standard sparse block shapes this device uses.
#[derive(Clone, Copy)]
enum Shape {
    /// 2D, at a sample count.
    Planar(SampleCounts),
    Volume,
}

/// Texels in one 64 KiB standard sparse block (Vulkan specification,
/// "Standard Sparse Image Block Shapes" and its MSAA table), scaled by the
/// format's block size. Each doubling of the sample count halves the
/// single-sample shape, width first.
fn standard_tile_shape(shape: Shape, description: &Description) -> Option<Size> {
    let bytes = description.block.bytes;
    let index = match bytes {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        16 => 4,
        _ => return None,
    };
    const SHAPES_2D: [(u64, u64); 5] = [(256, 256), (256, 128), (128, 128), (128, 64), (64, 64)];
    const SHAPES_3D: [(u64, u64, u64); 5] = [
        (64, 32, 32),
        (32, 32, 32),
        (32, 32, 16),
        (32, 16, 16),
        (16, 16, 16),
    ];
    let (width, height, depth) = match shape {
        Shape::Planar(samples) => {
            let (width, height) = SHAPES_2D[index];
            let halvings = samples.bits().trailing_zeros();
            (width >> halvings.div_ceil(2), height >> (halvings / 2), 1)
        }
        Shape::Volume => SHAPES_3D[index],
    };
    let block = description.block;
    Some(Size {
        width: width * u64::from(block.width),
        height: height * u64::from(block.height),
        depth,
    })
}

struct Derivation<'a> {
    device: &'a PhysicalDevice,
    platform: &'a Platform,
    shaders: ShaderFeatures,
    api: ApiFeatures,
    gaps: Vec<Gap>,
}

/// One requirement of a capability: the driver's part, the compiler's
/// constructs and this layer's API path.
struct Requirement {
    name: &'static str,
    driver: bool,
    shaders: ShaderFeatures,
    api: ApiFeatures,
}

const fn driver(name: &'static str, driver: bool) -> Requirement {
    Requirement {
        name,
        driver,
        shaders: ShaderFeatures::EMPTY,
        api: ApiFeatures::EMPTY,
    }
}

const fn shader(name: &'static str, driver: bool, shaders: ShaderFeatures) -> Requirement {
    Requirement {
        name,
        driver,
        shaders,
        api: ApiFeatures::EMPTY,
    }
}

const fn api(name: &'static str, driver: bool, api: ApiFeatures) -> Requirement {
    Requirement {
        name,
        driver,
        shaders: ShaderFeatures::EMPTY,
        api,
    }
}

/// Metal tessellation: factors from a compute pass feed a tessellator that
/// runs the post-tessellation vertex function.
fn tessellation(features: &Features) -> Requirement {
    Requirement {
        name: "tessellation",
        driver: features.tessellation_shader,
        shaders: ShaderFeatures::TESSELLATION
            .union(ShaderFeatures::COMPUTE)
            .union(ShaderFeatures::VERTEX_FRAGMENT),
        api: ApiFeatures::TESSELLATION,
    }
}

/// Whether shaders on `device` access storage copies of depth and stencil
/// attachments, multisampled ones included, which image copies between
/// depth or stencil aspects and color images fill.
pub fn depth_stencil_copies(device: &PhysicalDevice) -> bool {
    let features = &device.features;
    features.maintenance8
        && features.shader_storage_image_multisample
        && features.shader_storage_image_read_without_format
}

/// Whether a fragment function on `device` can run the fragment tests
/// itself: on storage copies of the attachments in its interlock section.
pub fn fragment_tests(device: &PhysicalDevice) -> bool {
    device.features.fragment_shader_pixel_interlock && depth_stencil_copies(device)
}

impl Derivation<'_> {
    fn cause(&self, requirement: &Requirement) -> Option<Cause> {
        if !requirement.driver {
            Some(Cause::Driver)
        } else if !self.shaders.contains(requirement.shaders) {
            Some(Cause::Compiler)
        } else if !self.api.contains(requirement.api) {
            Some(Cause::Api)
        } else {
            None
        }
    }

    /// Whether `requirement` holds; records it as a gap when it does not.
    fn check(&mut self, requirement: Requirement) -> bool {
        match self.cause(&requirement) {
            Some(cause) => {
                self.gap(requirement.name, cause);
                false
            }
            None => true,
        }
    }

    /// Whether every requirement holds; `capability` is recorded with the
    /// cause of the first one that does not.
    fn check_all(&mut self, capability: &'static str, requirements: &[Requirement]) -> bool {
        let first = requirements
            .iter()
            .find_map(|requirement| self.cause(requirement));
        for requirement in requirements {
            if let Some(cause) = self.cause(requirement) {
                self.gap(requirement.name, cause);
            }
        }
        match first {
            Some(cause) => {
                self.gap(capability, cause);
                false
            }
            None => true,
        }
    }

    fn gap(&mut self, capability: &'static str, cause: Cause) {
        if !self.gaps.iter().any(|gap| gap.capability == capability) {
            self.gaps.push(Gap {
                capability: String::from(capability),
                cause,
            });
        }
    }

    fn run(mut self) -> Capabilities {
        let device = self.device;
        let limits = &device.limits;
        let features = &device.features;

        let formats = self.formats();
        let caps = |format| indexed_format(&formats, format);
        let float32 = [
            PixelFormat::R32Float,
            PixelFormat::Rg32Float,
            PixelFormat::Rgba32Float,
        ];
        let filtering32 = float32
            .iter()
            .all(|format| caps(*format).contains(FormatCapabilities::FILTER));
        let msaa32 = float32
            .iter()
            .all(|format| caps(*format).contains(FormatCapabilities::MSAA));
        let bc = features.texture_compression_bc
            && PixelFormat::ALL
                .iter()
                .filter(|format| format.describe().compression == Compression::Bc)
                .all(|format| caps(*format).contains(FormatCapabilities::FILTER));
        let depth24 =
            caps(PixelFormat::Depth24UnormStencil8).contains(FormatCapabilities::DEPTH_STENCIL);

        let supports_32bit_float_filtering =
            self.check(driver("32-bit float filtering", filtering32));
        let supports_32bit_msaa = self.check(driver("32-bit float MSAA", msaa32));
        let supports_bc = self.check(driver("BC texture compression", bc));
        let depth24_stencil8 = self.check(driver("Depth24Unorm_Stencil8", depth24));

        let read_write_tier = self.read_write_tier(&formats);
        let argument_tier = self.argument_buffers_tier();

        let query_lod = self.check(shader(
            "texture LOD queries",
            true,
            ShaderFeatures::QUERY_LOD,
        ));
        let pull_model = self.check(shader(
            "pull-model interpolation",
            features.sample_rate_shading,
            ShaderFeatures::PULL_MODEL_INTERPOLATION,
        ));
        let barycentrics = self.check(shader(
            "shader barycentric coordinates",
            features.fragment_shader_barycentric,
            ShaderFeatures::BARYCENTRICS,
        ));
        let sample_positions = self.check(api(
            "programmable sample positions",
            limits.sample_locations.is_some_and(|locations| {
                locations.variable_sample_locations
                    && locations.sub_pixel_bits >= 4
                    && locations.sample_counts.contains(SampleCounts::C4)
            }),
            ApiFeatures::PROGRAMMABLE_SAMPLE_POSITIONS,
        ));
        let raster_order_groups = self.check(shader(
            "raster order groups",
            features.fragment_shader_pixel_interlock,
            ShaderFeatures::RASTER_ORDER_GROUPS,
        ));
        let framebuffer_reads = raster_order_groups
            && self.check(api(
                "framebuffer reads",
                features.dynamic_rendering_local_read && features.shader_storage_image_multisample,
                ApiFeatures::EMPTY,
            ));
        // A stencil output without stencil export, or storage attachments
        // whose pipeline cannot test fragments first: the function tests
        // fragments on storage copies of the attachments, which image copies
        // between depth or stencil aspects and color images fill.
        let fragment_tests = raster_order_groups && fragment_tests(self.device);
        // Vulkan resolves the depth while the stencil, which follows it,
        // resolves from copies of the samples.
        let stencil_follows_depth =
            depth_stencil_copies(self.device) && limits.independent_resolve_none;
        let raytracing = self.check(shader(
            "ray tracing",
            features.acceleration_structure && features.ray_query,
            ShaderFeatures::RAY_TRACING,
        ));
        let raytracing_from_render = raytracing
            && self.check(shader(
                "ray tracing from render",
                true,
                ShaderFeatures::RAY_TRACING | ShaderFeatures::VERTEX_FRAGMENT,
            ));
        // Vulkan acceleration structures have no motion instances.
        let motion_blur = self.check(driver("primitive motion blur", false));
        let function_pointers = self.check(shader(
            "function pointers",
            true,
            ShaderFeatures::FUNCTION_POINTERS,
        ));
        let function_pointers_from_render = function_pointers
            && self.check(shader(
                "function pointers from render",
                true,
                ShaderFeatures::FUNCTION_POINTERS | ShaderFeatures::VERTEX_FRAGMENT,
            ));
        let dynamic_libraries = self.check(shader(
            "dynamic libraries",
            true,
            ShaderFeatures::DYNAMIC_LIBRARIES,
        ));
        let render_dynamic_libraries = dynamic_libraries
            && self.check(shader(
                "render dynamic libraries",
                true,
                ShaderFeatures::DYNAMIC_LIBRARIES | ShaderFeatures::VERTEX_FRAGMENT,
            ));
        let mesh = self.check(shader(
            "mesh shaders",
            features.mesh_shader && features.task_shader && limits.mesh.is_some(),
            ShaderFeatures::MESH,
        ));
        let tessellation = self.check(tessellation(features));
        // Placement sparse resources map only through `MTL4CommandQueue`.
        self.gap("placement sparse resources", Cause::AppleArchitecture);

        let timestamps = self.check(api(
            "timestamp counters",
            limits.timestamp_compute_and_graphics && limits.timestamp_period > 0.0,
            ApiFeatures::TIMESTAMP_COUNTERS,
        ));
        let statistics = self.check(api(
            "statistic counters",
            features.pipeline_statistics_query,
            ApiFeatures::STATISTIC_COUNTERS,
        ));
        let counters = timestamps || statistics;
        if counters {
            // Tile dispatches exist only on Apple GPUs.
            self.gap("tile-dispatch counter sampling", Cause::AppleArchitecture);
        }
        let counter_sampling = [counters, counters, counters, false, counters];

        let texture_sample_counts = limits.framebuffer_color_sample_counts
            & limits.framebuffer_depth_sample_counts
            & limits.sampled_image_color_sample_counts
            & limits.sampled_image_depth_sample_counts;

        // Amplified draws repeat per view with the view in push constants,
        // which keeps Metal's arbitrary viewport and layer mappings.
        let amplification = self.check(shader(
            "vertex amplification",
            true,
            ShaderFeatures::VERTEX_AMPLIFICATION,
        ));
        let sparse = self.check_all(
            "sparse textures",
            &[
                api(
                    "sparse textures",
                    // Metal's tier 1 reads unmapped tiles as zero.
                    features.sparse_binding
                        && features.sparse_residency_image_2d
                        && limits.sparse.residency_standard_2d_block_shape
                        && limits.sparse.residency_non_resident_strict
                        && self.platform.sparse_page_bytes != 0,
                    ApiFeatures::SPARSE_TEXTURES,
                ),
                shader(
                    "sparse residency feedback",
                    features.shader_resource_residency,
                    ShaderFeatures::SPARSE_FEEDBACK,
                ),
            ],
        );
        let sparse_volumes = sparse
            && self.check(driver(
                "sparse 3D textures",
                features.sparse_residency_image_3d
                    && limits.sparse.residency_standard_3d_block_shape,
            ));
        let sparse_samples = if sparse {
            self.sparse_samples()
        } else {
            Vec::new()
        };
        let multisample = |(_, counts): &(PixelFormat, SampleCounts)| {
            !counts.difference(SampleCounts::C1).is_empty()
        };
        if sparse && !sparse_samples.iter().any(multisample) {
            self.gap("sparse multisample textures", Cause::Driver);
        }

        let step_rates = self.check(driver(
            "vertex step rates",
            features.vertex_attribute_instance_rate_divisor,
        ));
        let vertex_formats = self.vertex_formats();
        let dual_source_blending =
            self.check(driver("dual-source blending", features.dual_src_blend));
        let alpha_to_one = self.check(driver("alpha to one", features.alpha_to_one));
        let depth_clamp = self.check(driver(
            "depth clamping",
            features.depth_clamp
                && features.depth_clip_enable
                && features.extended_dynamic_state3_depth_clamp_enable
                && features.extended_dynamic_state3_depth_clip_enable,
        ));
        let view_min_lod = self.check(driver(
            "texture view minimum LODs",
            features.image_view_min_lod,
        ));
        let wireframe = self.check(driver(
            "triangle line fill",
            features.fill_mode_non_solid && features.extended_dynamic_state3_polygon_mode,
        ));
        let line_modes = self.check(driver(
            "line rasterization modes",
            features.bresenham_lines && features.rectangular_lines,
        ));

        let workgroup = limits.max_compute_work_group_size;
        let mesh_limits = limits.mesh.filter(|_| mesh);
        let limits_out = Limits {
            max_threads_per_threadgroup: Size {
                width: u64::from(workgroup[0]),
                height: u64::from(workgroup[1]),
                depth: u64::from(workgroup[2]),
            },
            max_total_threads_per_threadgroup: u64::from(limits.max_compute_work_group_invocations),
            thread_execution_width: u64::from(limits.max_subgroup_size),
            max_threadgroup_memory_length: u64::from(limits.max_compute_shared_memory_size),
            max_buffer_length: limits
                .max_buffer_size
                .min(limits.max_memory_allocation_size),
            max_texture_dimension_1d: u64::from(limits.max_image_dimension_1d),
            max_texture_dimension_2d: u64::from(
                limits
                    .max_image_dimension_2d
                    .min(limits.max_framebuffer_width)
                    .min(limits.max_framebuffer_height),
            ),
            max_texture_dimension_3d: u64::from(limits.max_image_dimension_3d),
            max_texture_dimension_cube: u64::from(limits.max_image_dimension_cube),
            max_texture_array_length: u64::from(limits.max_image_array_layers),
            max_texture_buffer_width: u64::from(limits.max_texel_buffer_elements),
            max_color_attachments: u64::from(
                limits
                    .max_color_attachments
                    .min(limits.max_fragment_output_attachments),
            ),
            max_viewports: u64::from(if features.multi_viewport {
                limits.max_viewports
            } else {
                1
            }),
            max_vertex_amplification_count: if amplification { 2 } else { 1 },
            max_argument_buffer_sampler_count: u64::from(match argument_tier {
                ArgumentBuffersTier::Tier2 => TIER2_ARGUMENT_SAMPLERS,
                ArgumentBuffersTier::Tier1 => MAX_SAMPLER_BINDINGS,
            }),
            max_argument_buffer_resources: u64::from(match argument_tier {
                ArgumentBuffersTier::Tier2 => TIER2_ARGUMENT_RESOURCES,
                ArgumentBuffersTier::Tier1 => MAX_TEXTURE_BINDINGS,
            }),
            // Apple GPUs draw points at most 511 pixels wide, and the
            // compiler clamps every point size to it (air_to_nir.c).
            max_point_size: limits.point_size_range[1].min(APPLE_MAX_POINT_SIZE),
            max_vertex_buffer_stride: u64::from(limits.max_vertex_input_binding_stride),
            max_vertex_attribute_offset: u64::from(limits.max_vertex_input_attribute_offset),
            max_vertex_step_rate: if step_rates {
                u64::from(limits.max_vertex_attrib_divisor)
            } else {
                1
            },
            max_tessellation_factor: if tessellation {
                u64::from(limits.max_tessellation_generation_level)
            } else {
                0
            },
            max_mesh_threads_per_threadgroup: mesh_limits
                .map_or(0, |mesh| u64::from(mesh.max_mesh_work_group_invocations)),
            max_object_payload_length: mesh_limits
                .map_or(0, |mesh| u64::from(mesh.max_task_payload_size)),
            buffer_alignment_for_uniforms: limits.min_uniform_buffer_offset_alignment,
            sparse_tile_bytes: if sparse {
                self.platform.sparse_page_bytes
            } else {
                0
            },
        };

        let integrated = device.identity.device_type == PhysicalDeviceType::IntegratedGpu;
        let mut capabilities = Capabilities {
            name: device.identity.device_name.clone(),
            architecture: self.platform.architecture.clone(),
            location: self.platform.location,
            location_number: self.platform.location_number,
            max_transfer_rate: self.platform.max_transfer_rate,
            headless: self.platform.headless,
            low_power: integrated,
            removable: self.platform.removable,
            has_unified_memory: integrated,
            recommended_max_working_set_size: device
                .memory
                .device_local_budget()
                .unwrap_or_else(|| device.memory.device_local_bytes()),
            limits: limits_out,
            argument_buffers_tier: argument_tier,
            read_write_texture_tier: read_write_tier,
            supports_32bit_float_filtering,
            supports_32bit_msaa,
            supports_bc_texture_compression: supports_bc,
            depth24_stencil8_supported: depth24_stencil8,
            supports_query_texture_lod: query_lod,
            supports_pull_model_interpolation: pull_model,
            supports_shader_barycentric_coordinates: barycentrics,
            programmable_sample_positions: sample_positions,
            raster_order_groups,
            supports_raytracing: raytracing,
            supports_raytracing_from_render: raytracing_from_render,
            supports_primitive_motion_blur: motion_blur,
            supports_function_pointers: function_pointers,
            supports_function_pointers_from_render: function_pointers_from_render,
            supports_dynamic_libraries: dynamic_libraries,
            supports_render_dynamic_libraries: render_dynamic_libraries,
            supports_placement_sparse: false,
            supports_mesh_shaders: mesh,
            supports_tessellation: tessellation,
            depth_resolve_modes: limits.supported_depth_resolve_modes,
            stencil_resolve_modes: limits.supported_stencil_resolve_modes,
            texture_sample_counts,
            max_sampler_anisotropy: if features.sampler_anisotropy {
                limits.max_sampler_anisotropy
            } else {
                1.0
            },
            sampler_mirror_clamp_to_edge: features.sampler_mirror_clamp_to_edge,
            sampler_reduction_modes: features.sampler_filter_minmax,
            dual_source_blending,
            alpha_to_one,
            depth_clamp,
            wireframe,
            line_modes,
            view_min_lod,
            timestamp_frequency: if timestamps {
                (1e9 / f64::from(limits.timestamp_period)) as u64
            } else {
                0
            },
            statistic_counters: statistics,
            memoryless_render_targets: true,
            framebuffer_reads,
            fragment_tests,
            stencil_export: limits.stencil_export,
            stencil_follows_depth,
            counter_sampling,
            sparse_volumes,
            sparse_samples,
            vertex_formats,
            families: Vec::new(),
            formats: PixelFormats::Indexed(formats),
            gaps: Vec::new(),
            linear_row_alignment: self.platform.linear_row_alignment,
            linear_offset_alignment: self.platform.linear_offset_alignment,
            texel_buffer_alignment: TexelBufferAlignment {
                bytes: limits
                    .uniform_texel_buffer_offset_alignment_bytes
                    .max(limits.storage_texel_buffer_offset_alignment_bytes),
                single_texel: limits.uniform_texel_buffer_offset_single_texel_alignment
                    && limits.storage_texel_buffer_offset_single_texel_alignment,
            },
        };
        capabilities.families = self.families(&capabilities);
        capabilities.gaps = self.gaps;
        capabilities
    }

    fn formats(&mut self) -> Box<IndexedPixelFormats> {
        let mut formats = Box::new([FormatCapabilities::EMPTY; PixelFormat::ALL.len()]);
        for (format, capabilities) in PixelFormat::ALL.iter().zip(formats.iter_mut()) {
            let description = format.describe();
            *capabilities = match description.realization {
                Ok(_) if format.planes().is_some() => self.two_plane_capabilities(*format),
                Ok(realization) => {
                    let support = self.device.formats.get(realization.format);
                    self.format_capabilities(description, support, &realization)
                }
                Err(_) => FormatCapabilities::EMPTY,
            };
        }
        self.compression_gaps(&formats);
        if PixelFormat::ALL
            .iter()
            .zip(formats.iter())
            .any(|(format, capabilities)| {
                format.describe().realization.is_err() && capabilities.is_empty()
            })
        {
            self.gap("4:2:2 pixel formats", Cause::Driver);
        }
        formats
    }

    /// The formats with sparse 2D textures and their sample counts.
    fn sparse_samples(&self) -> Vec<(PixelFormat, SampleCounts)> {
        PixelFormat::ALL
            .iter()
            .filter_map(|format| {
                let realization = format.describe().realization.ok()?;
                if !realization.encoding.is_exact() {
                    return None;
                }
                let counts = self.sparse_counts(self.device.formats.get(realization.format));
                (!counts.is_empty()).then_some((*format, counts))
            })
            .collect()
    }

    /// Whether fragment functions may write attachments as storage in their
    /// interlock section: framebuffer reads' requirements, unrecorded.
    fn storage_attachments(&self) -> bool {
        let features = &self.device.features;
        features.fragment_shader_pixel_interlock
            && features.dynamic_rendering_local_read
            && features.shader_storage_image_multisample
            && self.shaders.contains(ShaderFeatures::RASTER_ORDER_GROUPS)
    }

    /// A format's sparse 2D sample counts: multisample counts need the
    /// standard multisample block shapes.
    fn sparse_counts(&self, support: FormatSupport) -> SampleCounts {
        let features = &self.device.features;
        let sparse = &self.device.limits.sparse;
        let counts = SampleCounts::C1
            .with(
                features.sparse_residency_samples,
                sparse.residency_standard_2d_multisample_block_shape,
            )
            .intersection(support.sample_counts);
        if features.sparse_residency_image_2d {
            support.sparse.intersection(counts)
        } else {
            SampleCounts::EMPTY
        }
    }

    fn vertex_formats(&mut self) -> Vec<VertexFormat> {
        let (supported, refused): (Vec<VertexFormat>, Vec<VertexFormat>) =
            VertexFormat::ALL.iter().partition(|format| {
                self.device
                    .formats
                    .get(format.vulkan().0)
                    .buffer
                    .contains(FormatFeatures::VERTEX_BUFFER)
            });
        // Vertex functions decode the formats of whole words that the
        // pipeline fetches as words (`vertex::word_formats`).
        if refused
            .iter()
            .any(|format| !format.vulkan().1.is_multiple_of(4))
        {
            self.gap("vertex formats the driver cannot fetch", Cause::Driver);
        }
        supported
    }

    fn compression_gaps(&mut self, formats: &IndexedPixelFormats) {
        let groups = [
            (Compression::Bc, "BC pixel formats"),
            (Compression::Etc2Eac, "ETC2 and EAC pixel formats"),
            (Compression::AstcLdr, "ASTC LDR pixel formats"),
            (Compression::AstcHdr, "ASTC HDR pixel formats"),
            (Compression::Pvrtc, "PVRTC pixel formats"),
        ];
        for (compression, name) in groups {
            let unusable = PixelFormat::ALL
                .iter()
                .zip(formats)
                .filter(|(format, _)| format.describe().compression == compression)
                .all(|(_, capabilities)| !capabilities.contains(FormatCapabilities::READ));
            if unusable {
                self.gap(name, Cause::Driver);
            }
        }
    }

    fn format_capabilities(
        &self,
        description: &Description,
        support: FormatSupport,
        realization: &Realization,
    ) -> FormatCapabilities {
        use FormatCapabilities as C;
        use FormatFeatures as F;
        let features = &self.device.features;
        let optimal = support.optimal;
        let shaders = self.shaders;
        let color = description.is_color();
        let integer = matches!(description.kind, Kind::Color(numeric) if numeric.is_integer());
        let multisample = support.sample_counts.contains(SampleCounts::C4);
        let raster = shaders.contains(ShaderFeatures::VERTEX_FRAGMENT);
        // Shader writes would skip a stored encoding's clamping, and only
        // an exact encoding's texels lie in a buffer or sparse tile as Metal's.
        // The two-plane extended-range formats' writes hold their half
        // floats to the codes (`ResourceId::EXTENDED_STORAGE`).
        let exact = realization.encoding.is_exact();
        let held = matches!(realization.encoding, Encoding::ExtendedRange(codes) if codes.planes);
        let identity = realization.is_identity();
        let storage = color
            && identity
            && (exact || held)
            && optimal.contains(F::STORAGE_IMAGE | F::STORAGE_WRITE_WITHOUT_FORMAT)
            && features.shader_storage_image_write_without_format
            && shaders.contains(ShaderFeatures::TEXTURE_READ_WRITE);
        // An sRGB format's storage views are its linear twin's, through
        // which shaders encode and decode (`PixelFormat::srgb_storage`).
        let twin = description
            .format
            .srgb_storage()
            .and_then(|linear| linear.describe().realization.as_ref().ok())
            .map(|linear| self.device.formats.get(linear.format).optimal);
        let storage = storage
            || twin.is_some_and(|twin| {
                twin.contains(F::STORAGE_IMAGE | F::STORAGE_WRITE_WITHOUT_FORMAT)
                    && features.shader_storage_image_write_without_format
                    && shaders.contains(ShaderFeatures::TEXTURE_READ_WRITE)
            });
        let storage_read = twin
            .unwrap_or(optimal)
            .contains(F::STORAGE_READ_WITHOUT_FORMAT)
            && features.shader_storage_image_read_without_format;
        let sampled =
            optimal.contains(F::SAMPLED_IMAGE) && shaders.contains(ShaderFeatures::TEXTURE_SAMPLE);
        let attachment = color
            && realization.writes_through()
            && raster
            && optimal.contains(F::COLOR_ATTACHMENT);
        // A format the driver neither renders nor blends renders through
        // its raw texels: the fragment function blends and writes them in
        // its interlock section, and the average samples kernel resolves.
        let raw_support = raw_texel_format(description.format)
            .filter(|_| {
                color
                    && identity
                    && exact
                    && raster
                    && raw_rendered(description.format).is_some()
                    && !optimal.contains(F::COLOR_ATTACHMENT)
                    && self.storage_attachments()
            })
            .map(|raw| self.device.formats.get(raw))
            .filter(|raw| raw.optimal.contains(F::COLOR_ATTACHMENT | F::STORAGE_IMAGE));
        let raw_render = raw_support.is_some();
        let raw_multisample =
            raw_support.is_some_and(|raw| raw.sample_counts.contains(SampleCounts::C4));
        let depth_stencil = !color
            && !description.view_only
            && raster
            && optimal.contains(F::DEPTH_STENCIL_ATTACHMENT);

        let mut capabilities = C::EMPTY
            .with(C::READ, sampled)
            .with(
                C::FILTER,
                sampled && !integer && optimal.contains(F::SAMPLED_IMAGE_FILTER_LINEAR),
            )
            .with(C::WRITE, storage)
            .with(C::READ_WRITE, storage && storage_read)
            .with(C::COLOR, attachment || raw_render)
            .with(
                C::BLEND,
                attachment && optimal.contains(F::COLOR_ATTACHMENT_BLEND) || raw_render,
            )
            .with(C::RAW_RENDER, raw_render)
            .with(C::DEPTH_STENCIL, depth_stencil)
            .with(
                C::MSAA,
                (attachment || depth_stencil) && multisample || raw_multisample,
            )
            .with(
                C::SPARSE,
                exact && self.sparse_counts(support).contains(SampleCounts::C1),
            )
            .with(
                C::ATOMIC,
                storage
                    && optimal.contains(F::STORAGE_IMAGE_ATOMIC)
                    && shaders.contains(ShaderFeatures::TEXTURE_ATOMICS),
            )
            .with(
                C::COMPARE_SAMPLE,
                sampled
                    && description.has_depth()
                    && optimal.contains(F::SAMPLED_IMAGE_DEPTH_COMPARISON)
                    && shaders.contains(ShaderFeatures::DEPTH_COMPARE_SAMPLE),
            )
            .with(C::BLIT, optimal.contains(F::TRANSFER_SRC | F::TRANSFER_DST))
            // The CPU reaches a decoded format's blocks, not its image.
            .with(
                C::HOST_TRANSFER,
                features.host_image_copy
                    && optimal.contains(F::HOST_IMAGE_TRANSFER)
                    && !matches!(realization.encoding, Encoding::Decoded(_)),
            );

        // Vulkan averages only float and normalized color attachments.
        let color_resolve = attachment && multisample && !integer || raw_multisample;
        let depth_resolve = depth_stencil
            && multisample
            && self
                .device
                .limits
                .supported_depth_resolve_modes
                .contains(ResolveModes::SAMPLE_ZERO);
        capabilities = capabilities.with(C::RESOLVE, color_resolve || depth_resolve);

        if color && identity && exact && !description.is_compressed() {
            let buffer = support.buffer;
            let buffer_write = buffer.contains(F::STORAGE_TEXEL_BUFFER)
                && shaders.contains(ShaderFeatures::TEXTURE_READ_WRITE)
                && features.shader_storage_image_write_without_format;
            capabilities = capabilities
                .with(
                    C::TEXTURE_BUFFER_READ,
                    buffer.contains(F::UNIFORM_TEXEL_BUFFER)
                        && shaders.contains(ShaderFeatures::TEXTURE_SAMPLE),
                )
                .with(C::TEXTURE_BUFFER_WRITE, buffer_write)
                .with(
                    C::TEXTURE_BUFFER_READ_WRITE,
                    buffer_write && features.shader_storage_image_read_without_format,
                )
                .with(
                    C::LINEAR,
                    features.image_drm_format_modifier && support.linear.contains(F::SAMPLED_IMAGE),
                );
        }
        capabilities
    }

    /// A two-plane format's: an IOSurface's planes, linear images of one
    /// and two components ([`PixelFormat::plane_formats`]), which shaders
    /// sample and read together.
    /// A 4:4:4 format renders through both planes' raw texels: the
    /// fragment function converts, blends and writes them in its interlock
    /// section ([`FormatCapabilities::RAW_RENDER`]). Rendering to a
    /// subsampled format, which averages its second plane's texels, and
    /// shader writes are not implemented.
    fn two_plane_capabilities(&self, format: PixelFormat) -> FormatCapabilities {
        use crate::metal::vulkan::Format as V;
        use FormatCapabilities as C;
        use FormatFeatures as F;
        let plane = |format: PixelFormat| {
            let description = format.describe();
            description.realization.map_or(C::EMPTY, |realization| {
                let support = self.device.formats.get(realization.format);
                self.format_capabilities(description, support, &realization)
            })
        };
        let Some([first, second]) = format.plane_formats() else {
            return C::EMPTY;
        };
        let (first, second) = (plane(first), plane(second));
        let both = |flag| first.contains(flag) && second.contains(flag);
        // The planes are linear-modifier images (`resource::linear_texture`).
        let linear = |format: V, features: F| {
            self.device
                .formats
                .get(format)
                .modifier_linear
                .contains(features)
        };
        // Shaders convert raw texels of byte components only.
        let render = format.planes() == Some(Planes::Full)
            && !format.ten_bit_planes()
            && self.storage_attachments()
            && linear(V::R8Unorm, F::COLOR_ATTACHMENT)
            && linear(V::R8G8Unorm, F::COLOR_ATTACHMENT)
            && linear(V::R8Uint, F::COLOR_ATTACHMENT | F::STORAGE_IMAGE)
            && linear(V::R16Uint, F::STORAGE_IMAGE);
        C::EMPTY
            .with(C::READ, both(C::READ))
            .with(C::FILTER, both(C::FILTER))
            .with(C::LINEAR, both(C::LINEAR))
            .with(C::COLOR, render)
            .with(C::BLEND, render)
            .with(C::RAW_RENDER, render)
    }

    /// Tier 1 reads and writes the 32-bit single-channel formats in one
    /// function; tier 2 adds Apple's listed 8-, 16- and 32-bit formats.
    fn read_write_tier(&mut self, formats: &IndexedPixelFormats) -> ReadWriteTextureTier {
        use PixelFormat as P;
        const TIER1: [PixelFormat; 3] = [P::R32Float, P::R32Uint, P::R32Sint];
        const TIER2: [PixelFormat; 12] = [
            P::Rgba32Float,
            P::Rgba32Uint,
            P::Rgba32Sint,
            P::Rgba16Float,
            P::Rgba16Uint,
            P::Rgba16Sint,
            P::Rgba8Unorm,
            P::Rgba8Uint,
            P::Rgba8Sint,
            P::R16Float,
            P::R16Uint,
            P::R16Sint,
        ];
        const TIER2_8BIT: [PixelFormat; 3] = [P::R8Unorm, P::R8Uint, P::R8Sint];
        let read_write = |format: &PixelFormat| {
            indexed_format(formats, *format).contains(FormatCapabilities::READ_WRITE)
        };
        let tier1 = TIER1.iter().all(read_write);
        let tier2 = tier1 && TIER2.iter().chain(&TIER2_8BIT).all(read_write);
        let cause = if self.shaders.contains(ShaderFeatures::TEXTURE_READ_WRITE) {
            Cause::Driver
        } else {
            Cause::Compiler
        };
        if !tier1 {
            self.gap("read-write textures", cause);
            ReadWriteTextureTier::None
        } else if !tier2 {
            self.gap("read-write texture tier 2", cause);
            ReadWriteTextureTier::Tier1
        } else {
            ReadWriteTextureTier::Tier2
        }
    }

    /// Argument buffers address every resource through bindless descriptor
    /// heaps and buffer device addresses; tier 2 needs heaps large enough for
    /// Apple's Mac tier 2 counts.
    fn argument_buffers_tier(&mut self) -> ArgumentBuffersTier {
        let features = &self.device.features;
        let limits = &self.device.limits;
        let bindless = features.descriptor_indexing
            && features.runtime_descriptor_array
            && features.descriptor_binding_partially_bound
            && features.descriptor_binding_update_unused_while_pending
            && features.descriptor_binding_sampled_image_update_after_bind
            && features.descriptor_binding_storage_image_update_after_bind
            && features.shader_sampled_image_array_non_uniform_indexing
            && features.shader_storage_image_array_non_uniform_indexing
            && features.buffer_device_address;
        let capacity = limits.max_descriptor_set_update_after_bind_sampled_images
            >= TIER2_ARGUMENT_RESOURCES
            && limits.max_descriptor_set_update_after_bind_storage_images
                >= TIER2_ARGUMENT_RESOURCES
            && limits.max_descriptor_set_update_after_bind_samplers >= TIER2_ARGUMENT_SAMPLERS
            && limits.max_sampler_allocation_count >= TIER2_ARGUMENT_SAMPLERS;
        let tier1 = self.check(shader(
            "argument buffers",
            bindless,
            ShaderFeatures::ARGUMENT_BUFFERS,
        ));
        if tier1 && self.check(driver("argument buffers tier 2", capacity)) {
            ArgumentBuffersTier::Tier2
        } else {
            ArgumentBuffersTier::Tier1
        }
    }

    fn families(&mut self, capabilities: &Capabilities) -> Vec<GpuFamily> {
        let features = self.device.features;
        let limits = &capabilities.limits;
        let format = |format: PixelFormat, required: FormatCapabilities| {
            capabilities.format(format).contains(required)
        };
        let render =
            FormatCapabilities::COLOR | FormatCapabilities::BLEND | FormatCapabilities::MSAA;
        let base_formats = format(PixelFormat::Rgba8Unorm, render | FormatCapabilities::FILTER)
            && format(PixelFormat::Bgra8Unorm, render | FormatCapabilities::FILTER)
            && format(
                PixelFormat::Rgba16Float,
                render | FormatCapabilities::FILTER,
            )
            && format(
                PixelFormat::R32Float,
                FormatCapabilities::COLOR | FormatCapabilities::READ,
            )
            && format(
                PixelFormat::Depth32Float,
                FormatCapabilities::DEPTH_STENCIL | FormatCapabilities::READ,
            );

        let common1 = self.check_all(
            "GPU family Common1",
            &[
                shader(
                    "render and compute functions",
                    true,
                    ShaderFeatures::COMPUTE
                        | ShaderFeatures::VERTEX_FRAGMENT
                        | ShaderFeatures::TEXTURE_SAMPLE
                        | ShaderFeatures::THREADGROUP_MEMORY,
                ),
                driver(
                    "Common1 limits",
                    limits.max_color_attachments >= 4
                        && limits.max_texture_dimension_2d >= 8192
                        && limits.max_threadgroup_memory_length >= 16 << 10
                        && limits.max_total_threads_per_threadgroup >= 512
                        && capabilities.supports_texture_sample_count(4),
                ),
                driver("Common1 pixel formats", base_formats),
            ],
        );
        let common2 = common1
            && self.check_all(
                "GPU family Common2",
                &[
                    api(
                        "indirect draws and dispatches",
                        features.multi_draw_indirect && features.draw_indirect_first_instance,
                        ApiFeatures::INDIRECT_DRAW_DISPATCH,
                    ),
                    shader("simdgroup functions", true, ShaderFeatures::SIMDGROUP),
                    shader(
                        "buffer atomics",
                        features.fragment_stores_and_atomics,
                        ShaderFeatures::BUFFER_ATOMICS,
                    ),
                    shader("argument buffers", true, ShaderFeatures::ARGUMENT_BUFFERS),
                    driver(
                        "read-write textures",
                        capabilities.read_write_texture_tier != ReadWriteTextureTier::None,
                    ),
                    driver("cube texture arrays", features.image_cube_array),
                    shader(
                        "layered rendering",
                        limits.max_texture_array_length >= 2048,
                        ShaderFeatures::LAYERED_RENDERING,
                    ),
                    tessellation(&features),
                    driver("depth clamping", capabilities.depth_clamp),
                    driver("dual-source blending", capabilities.dual_source_blending),
                ],
            );
        let common3 = common2
            && self.check_all(
                "GPU family Common3",
                &[
                    api(
                        "indirect command buffers",
                        features.conditional_rendering,
                        ApiFeatures::INDIRECT_COMMAND_BUFFERS,
                    ),
                    shader(
                        "GPU-encoded indirect commands",
                        true,
                        ShaderFeatures::INDIRECT_COMMAND_ENCODING,
                    ),
                    shader("quadgroup functions", true, ShaderFeatures::QUADGROUP),
                    api("heaps", true, ApiFeatures::HEAPS),
                ],
            );
        let mac2 = common3
            && self.check_all(
                "GPU family Mac2",
                &[
                    driver(
                        "BC texture compression",
                        capabilities.supports_bc_texture_compression,
                    ),
                    driver(
                        "read-write texture tier 2",
                        capabilities.read_write_texture_tier == ReadWriteTextureTier::Tier2,
                    ),
                    driver(
                        "argument buffers tier 2",
                        capabilities.argument_buffers_tier == ArgumentBuffersTier::Tier2,
                    ),
                    driver("16 viewports", limits.max_viewports >= 16),
                    driver(
                        "32-bit float filtering",
                        capabilities.supports_32bit_float_filtering,
                    ),
                    driver(
                        "Mac2 limits",
                        limits.max_color_attachments >= 8
                            && limits.max_texture_dimension_2d >= 16384
                            && limits.max_threadgroup_memory_length >= 32 << 10
                            && limits.max_total_threads_per_threadgroup >= 1024,
                    ),
                ],
            );
        let metal3 = mac2
            && self.check_all(
                "GPU family Metal3",
                &[
                    driver("mesh shaders", capabilities.supports_mesh_shaders),
                    driver("function pointers", capabilities.supports_function_pointers),
                    driver("dynamic libraries", capabilities.supports_dynamic_libraries),
                    api(
                        "fast resource loading",
                        true,
                        ApiFeatures::IO_COMMAND_QUEUES,
                    ),
                    api("MetalFX upscaling", true, ApiFeatures::METALFX),
                ],
            );
        // Apple1 and Apple2, which Apple3 contains, add the formats of
        // Apple's first GPUs: ETC2, EAC and PVRTC blocks and the 16-bit
        // packed formats, then ASTC LDR blocks, which this device samples
        // decoded as Apple's do.
        let readable = |compression: Compression| {
            PixelFormat::ALL
                .iter()
                .filter(|format| format.describe().compression == compression)
                .all(|format| {
                    capabilities
                        .format(*format)
                        .contains(FormatCapabilities::READ | FormatCapabilities::FILTER)
                })
        };
        let packed16 = [
            PixelFormat::B5G6R5Unorm,
            PixelFormat::A1Bgr5Unorm,
            PixelFormat::Abgr4Unorm,
            PixelFormat::Bgr5A1Unorm,
        ]
        .into_iter()
        .all(|packed| format(packed, render | FormatCapabilities::FILTER));
        let apple3 = mac2 && capabilities.framebuffer_reads;
        let apple1 = apple3
            && self.check_all(
                "GPU family Apple1",
                &[
                    driver("ETC2 and EAC sampling", readable(Compression::Etc2Eac)),
                    driver("PVRTC sampling", readable(Compression::Pvrtc)),
                    driver("16-bit packed render targets", packed16),
                ],
            );
        let apple2 = apple1
            && self.check_all(
                "GPU family Apple2",
                &[driver("ASTC LDR sampling", readable(Compression::AstcLdr))],
            );
        // Apple4 to Apple7 add what the feature set tables give them below
        // ray tracing, which Metal leaves to `supportsRaytracing`.
        let apple4 = apple2
            && self.check_all(
                "GPU family Apple4",
                &[
                    shader(
                        "imageblocks and tile functions",
                        true,
                        ShaderFeatures::IMAGEBLOCKS,
                    ),
                    driver("raster order groups", capabilities.raster_order_groups),
                ],
            );
        // Apple5's RG11B10 and RGB9E5 vertex formats are single words, which
        // vertex functions decode when the driver cannot fetch them
        // (`vertex::word_formats`).
        let apple5 = apple4
            && self.check_all(
                "GPU family Apple5",
                &[
                    driver(
                        "stencil resolve",
                        !capabilities.stencil_resolve_modes.is_empty(),
                    ),
                    driver(
                        "stencil resolve following a min or max depth resolve",
                        capabilities.stencil_follows_depth,
                    ),
                ],
            );
        // NVIDIA's fragment shaders have no stencil output: functions that
        // write one run the fragment tests themselves.
        if !self.device.limits.stencil_export && !capabilities.fragment_tests {
            self.gap("stencil feedback", Cause::Driver);
        }
        let apple6 = apple5
            && self.check_all(
                "GPU family Apple6",
                &[
                    driver("sparse textures", limits.sparse_tile_bytes != 0),
                    driver(
                        "vertex amplification",
                        capabilities.supports_vertex_amplification_count(2),
                    ),
                    driver(
                        "function pointers in render pipelines",
                        capabilities.supports_function_pointers_from_render,
                    ),
                    driver(
                        "render dynamic libraries",
                        capabilities.supports_render_dynamic_libraries,
                    ),
                    shader("texture atomics", true, ShaderFeatures::TEXTURE_ATOMICS),
                    driver("ASTC HDR sampling", readable(Compression::AstcHdr)),
                ],
            );
        let apple7 = apple6
            && metal3
            && self.check_all(
                "GPU family Apple7",
                &[
                    shader("simdgroup matrices", true, ShaderFeatures::SIMDGROUP_MATRIX),
                    driver(
                        "barycentric coordinates",
                        capabilities.supports_shader_barycentric_coordinates,
                    ),
                ],
            );
        self.gap("Apple GPU families from Apple8", Cause::AppleArchitecture);
        self.gap("GPU family Metal4", Cause::AppleArchitecture);

        let mut families = Vec::new();
        for (family, held) in [
            (GpuFamily::Apple1, apple1),
            (GpuFamily::Apple2, apple2),
            // Core Animation maps WindowServer's two-plane framebuffers to a
            // Metal format only on a device in this family, which predates
            // tile shaders. Programmable blending is among its features.
            (GpuFamily::Apple3, apple3),
            (GpuFamily::Apple4, apple4),
            (GpuFamily::Apple5, apple5),
            (GpuFamily::Apple6, apple6),
            (GpuFamily::Apple7, apple7),
            (GpuFamily::Common1, common1),
            (GpuFamily::Common2, common2),
            (GpuFamily::Common3, common3),
            (GpuFamily::Mac1, mac2),
            (GpuFamily::Mac2, mac2),
            (GpuFamily::MacCatalyst1, mac2),
            (GpuFamily::MacCatalyst2, mac2),
            (GpuFamily::Metal3, metal3),
        ] {
            if held {
                families.push(family);
            }
        }
        families
    }
}
