//! The internal Vulkan boundary under Stela's Metal device.
//!
//! Metal objects and commands lower to the values and operations declared
//! here, which a backend realizes on a conformant Vulkan driver (NVK on
//! NVIDIA). Every raw value is the Vulkan value from
//! `vulkan_core.h` at the pinned Mesa release (`mesa-26.2.5`), so a backend
//! passes them through unchanged. Only values the Metal layer emits or reads
//! are declared; unknown driver values stay outside these closed types.
pub mod backend;
pub mod command;
pub mod object;
pub mod physical;

raw_enum! {
    /// `VkFormat` values for the Metal pixel and vertex formats.
    pub enum Format: u32 {
        R4G4B4A4UnormPack16 = 2,
        R5G6B5UnormPack16 = 4,
        R5G5B5A1UnormPack16 = 6,
        A1R5G5B5UnormPack16 = 8,
        R8Unorm = 9,
        R8Snorm = 10,
        R8Uint = 13,
        R8Sint = 14,
        R8Srgb = 15,
        R8G8Unorm = 16,
        R8G8Snorm = 17,
        R8G8Uint = 20,
        R8G8Sint = 21,
        R8G8Srgb = 22,
        R8G8B8Unorm = 23,
        R8G8B8Snorm = 24,
        R8G8B8Uint = 27,
        R8G8B8Sint = 28,
        R8G8B8A8Unorm = 37,
        R8G8B8A8Snorm = 38,
        R8G8B8A8Uint = 41,
        R8G8B8A8Sint = 42,
        R8G8B8A8Srgb = 43,
        B8G8R8A8Unorm = 44,
        B8G8R8A8Srgb = 50,
        A2R10G10B10UnormPack32 = 58,
        A2B10G10R10UnormPack32 = 64,
        A2B10G10R10SnormPack32 = 65,
        A2B10G10R10UintPack32 = 68,
        R16Unorm = 70,
        R16Snorm = 71,
        R16Uint = 74,
        R16Sint = 75,
        R16Sfloat = 76,
        R16G16Unorm = 77,
        R16G16Snorm = 78,
        R16G16Uint = 81,
        R16G16Sint = 82,
        R16G16Sfloat = 83,
        R16G16B16Unorm = 84,
        R16G16B16Snorm = 85,
        R16G16B16Uint = 88,
        R16G16B16Sint = 89,
        R16G16B16Sfloat = 90,
        R16G16B16A16Unorm = 91,
        R16G16B16A16Snorm = 92,
        R16G16B16A16Uint = 95,
        R16G16B16A16Sint = 96,
        R16G16B16A16Sfloat = 97,
        R32Uint = 98,
        R32Sint = 99,
        R32Sfloat = 100,
        R32G32Uint = 101,
        R32G32Sint = 102,
        R32G32Sfloat = 103,
        R32G32B32Uint = 104,
        R32G32B32Sint = 105,
        R32G32B32Sfloat = 106,
        R32G32B32A32Uint = 107,
        R32G32B32A32Sint = 108,
        R32G32B32A32Sfloat = 109,
        B10G11R11UfloatPack32 = 122,
        E5B9G9R9UfloatPack32 = 123,
        D16Unorm = 124,
        D32Sfloat = 126,
        S8Uint = 127,
        D24UnormS8Uint = 129,
        D32SfloatS8Uint = 130,
        Bc1RgbaUnormBlock = 133,
        Bc1RgbaSrgbBlock = 134,
        Bc2UnormBlock = 135,
        Bc2SrgbBlock = 136,
        Bc3UnormBlock = 137,
        Bc3SrgbBlock = 138,
        Bc4UnormBlock = 139,
        Bc4SnormBlock = 140,
        Bc5UnormBlock = 141,
        Bc5SnormBlock = 142,
        Bc6hUfloatBlock = 143,
        Bc6hSfloatBlock = 144,
        Bc7UnormBlock = 145,
        Bc7SrgbBlock = 146,
        Etc2R8G8B8UnormBlock = 147,
        Etc2R8G8B8SrgbBlock = 148,
        Etc2R8G8B8A1UnormBlock = 149,
        Etc2R8G8B8A1SrgbBlock = 150,
        Etc2R8G8B8A8UnormBlock = 151,
        Etc2R8G8B8A8SrgbBlock = 152,
        EacR11UnormBlock = 153,
        EacR11SnormBlock = 154,
        EacR11G11UnormBlock = 155,
        EacR11G11SnormBlock = 156,
        Astc4x4UnormBlock = 157,
        Astc4x4SrgbBlock = 158,
        Astc5x4UnormBlock = 159,
        Astc5x4SrgbBlock = 160,
        Astc5x5UnormBlock = 161,
        Astc5x5SrgbBlock = 162,
        Astc6x5UnormBlock = 163,
        Astc6x5SrgbBlock = 164,
        Astc6x6UnormBlock = 165,
        Astc6x6SrgbBlock = 166,
        Astc8x5UnormBlock = 167,
        Astc8x5SrgbBlock = 168,
        Astc8x6UnormBlock = 169,
        Astc8x6SrgbBlock = 170,
        Astc8x8UnormBlock = 171,
        Astc8x8SrgbBlock = 172,
        Astc10x5UnormBlock = 173,
        Astc10x5SrgbBlock = 174,
        Astc10x6UnormBlock = 175,
        Astc10x6SrgbBlock = 176,
        Astc10x8UnormBlock = 177,
        Astc10x8SrgbBlock = 178,
        Astc10x10UnormBlock = 179,
        Astc10x10SrgbBlock = 180,
        Astc12x10UnormBlock = 181,
        Astc12x10SrgbBlock = 182,
        Astc12x12UnormBlock = 183,
        Astc12x12SrgbBlock = 184,
        Pvrtc12bppUnormBlockImg = 1000054000,
        Pvrtc14bppUnormBlockImg = 1000054001,
        Pvrtc12bppSrgbBlockImg = 1000054004,
        Pvrtc14bppSrgbBlockImg = 1000054005,
        Astc4x4SfloatBlock = 1000066000,
        Astc5x4SfloatBlock = 1000066001,
        Astc5x5SfloatBlock = 1000066002,
        Astc6x5SfloatBlock = 1000066003,
        Astc6x6SfloatBlock = 1000066004,
        Astc8x5SfloatBlock = 1000066005,
        Astc8x6SfloatBlock = 1000066006,
        Astc8x8SfloatBlock = 1000066007,
        Astc10x5SfloatBlock = 1000066008,
        Astc10x6SfloatBlock = 1000066009,
        Astc10x8SfloatBlock = 1000066010,
        Astc10x10SfloatBlock = 1000066011,
        Astc12x10SfloatBlock = 1000066012,
        Astc12x12SfloatBlock = 1000066013,
        G8B8G8R8Unorm422 = 1000156000,
        B8G8R8G8Unorm422 = 1000156001,
        A8Unorm = 1000470001,
    }
}

impl Format {
    /// The aspects of an image of this format.
    pub const fn aspects(self) -> ImageAspects {
        match self {
            Self::D16Unorm | Self::D32Sfloat => ImageAspects::DEPTH,
            Self::S8Uint => ImageAspects::STENCIL,
            Self::D24UnormS8Uint | Self::D32SfloatS8Uint => {
                ImageAspects::DEPTH.union(ImageAspects::STENCIL)
            }
            _ => ImageAspects::COLOR,
        }
    }
}

option_set! {
    /// `VkFormatFeatureFlags2`.
    pub struct FormatFeatures: u64 {
        const SAMPLED_IMAGE = 0x1;
        const STORAGE_IMAGE = 0x2;
        const STORAGE_IMAGE_ATOMIC = 0x4;
        const UNIFORM_TEXEL_BUFFER = 0x8;
        const STORAGE_TEXEL_BUFFER = 0x10;
        const STORAGE_TEXEL_BUFFER_ATOMIC = 0x20;
        const VERTEX_BUFFER = 0x40;
        const COLOR_ATTACHMENT = 0x80;
        const COLOR_ATTACHMENT_BLEND = 0x100;
        const DEPTH_STENCIL_ATTACHMENT = 0x200;
        const BLIT_SRC = 0x400;
        const BLIT_DST = 0x800;
        const SAMPLED_IMAGE_FILTER_LINEAR = 0x1000;
        const TRANSFER_SRC = 0x4000;
        const TRANSFER_DST = 0x8000;
        const SAMPLED_IMAGE_FILTER_MINMAX = 0x10000;
        const STORAGE_READ_WITHOUT_FORMAT = 0x8000_0000;
        const STORAGE_WRITE_WITHOUT_FORMAT = 0x1_0000_0000;
        const SAMPLED_IMAGE_DEPTH_COMPARISON = 0x2_0000_0000;
        const HOST_IMAGE_TRANSFER = 0x4000_0000_0000;
    }
}

raw_enum! {
    /// `VkPhysicalDeviceType`.
    pub enum PhysicalDeviceType: u32 {
        Other = 0,
        IntegratedGpu = 1,
        DiscreteGpu = 2,
        VirtualGpu = 3,
        Cpu = 4,
    }
}

raw_enum! {
    /// The `VkDriverId` values Stela admits as backends.
    pub enum DriverId: u32 {
        MesaRadv = 3,
        IntelOpenSourceMesa = 6,
        MesaLlvmPipe = 13,
        MesaVenus = 22,
        MesaNvk = 24,
    }
}

raw_enum! {
    /// `VkImageType`.
    pub enum ImageType: u32 {
        D1 = 0,
        D2 = 1,
        D3 = 2,
    }
}

raw_enum! {
    /// `VkImageViewType`.
    pub enum ImageViewType: u32 {
        D1 = 0,
        D2 = 1,
        D3 = 2,
        Cube = 3,
        D1Array = 4,
        D2Array = 5,
        CubeArray = 6,
    }
}

raw_enum! {
    /// `VkImageTiling`. Linear images use explicit DRM linear layouts.
    pub enum ImageTiling: u32 {
        Optimal = 0,
        Linear = 1,
        DrmFormatModifier = 1000158000,
    }
}

option_set! {
    /// `VkImageAspectFlags`.
    pub struct ImageAspects: u32 {
        const COLOR = 0x1;
        const DEPTH = 0x2;
        const STENCIL = 0x4;
    }
}

option_set! {
    /// `VkSampleCountFlags`.
    pub struct SampleCounts: u32 {
        const C1 = 0x1;
        const C2 = 0x2;
        const C4 = 0x4;
        const C8 = 0x8;
        const C16 = 0x10;
        const C32 = 0x20;
        const C64 = 0x40;
    }
}

impl SampleCounts {
    /// The single flag for `count` samples, if `count` is a Vulkan count.
    pub const fn single(count: u32) -> Option<Self> {
        if count.is_power_of_two() && count <= 64 {
            Some(Self(count))
        } else {
            None
        }
    }
}

option_set! {
    /// `VkImageUsageFlags`.
    pub struct ImageUsage: u32 {
        const TRANSFER_SRC = 0x1;
        const TRANSFER_DST = 0x2;
        const SAMPLED = 0x4;
        const STORAGE = 0x8;
        const COLOR_ATTACHMENT = 0x10;
        const DEPTH_STENCIL_ATTACHMENT = 0x20;
        const INPUT_ATTACHMENT = 0x80;
        const HOST_TRANSFER = 0x40_0000;
    }
}

option_set! {
    /// `VkImageCreateFlags`.
    pub struct ImageCreate: u32 {
        const SPARSE_BINDING = 0x1;
        const SPARSE_RESIDENCY = 0x2;
        const SPARSE_ALIASED = 0x4;
        const MUTABLE_FORMAT = 0x8;
        const CUBE_COMPATIBLE = 0x10;
        const ARRAY_2D_COMPATIBLE = 0x20;
        const EXTENDED_USAGE = 0x100;
        const ALIAS = 0x400;
        const SAMPLE_LOCATIONS_COMPATIBLE_DEPTH = 0x1000;
    }
}

option_set! {
    /// `VkBufferUsageFlags2`.
    pub struct BufferUsage: u64 {
        const TRANSFER_SRC = 0x1;
        const TRANSFER_DST = 0x2;
        const UNIFORM_TEXEL_BUFFER = 0x4;
        const STORAGE_TEXEL_BUFFER = 0x8;
        const UNIFORM_BUFFER = 0x10;
        const STORAGE_BUFFER = 0x20;
        const INDEX_BUFFER = 0x40;
        const VERTEX_BUFFER = 0x80;
        const INDIRECT_BUFFER = 0x100;
        const CONDITIONAL_RENDERING = 0x200;
        const SHADER_DEVICE_ADDRESS = 0x2_0000;
        /// `VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT`.
        const PREPROCESS_BUFFER = 0x8000_0000;
    }
}

option_set! {
    /// `VkBufferCreateFlags`.
    pub struct BufferCreate: u32 {
        const SPARSE_BINDING = 0x1;
        const SPARSE_RESIDENCY = 0x2;
        const SPARSE_ALIASED = 0x4;
    }
}

option_set! {
    /// `VkMemoryPropertyFlags`.
    pub struct MemoryProperties: u32 {
        const DEVICE_LOCAL = 0x1;
        const HOST_VISIBLE = 0x2;
        const HOST_COHERENT = 0x4;
        const HOST_CACHED = 0x8;
        const LAZILY_ALLOCATED = 0x10;
    }
}

option_set! {
    /// `VkMemoryHeapFlags`.
    pub struct MemoryHeapFlags: u32 {
        const DEVICE_LOCAL = 0x1;
    }
}

raw_enum! {
    /// `VkComponentSwizzle`.
    pub enum ComponentSwizzle: u32 {
        Identity = 0,
        Zero = 1,
        One = 2,
        R = 3,
        G = 4,
        B = 5,
        A = 6,
    }
}

raw_enum! {
    /// `VkFilter`.
    pub enum Filter: u32 {
        Nearest = 0,
        Linear = 1,
    }
}

raw_enum! {
    /// `VkSamplerMipmapMode`.
    pub enum SamplerMipmapMode: u32 {
        Nearest = 0,
        Linear = 1,
    }
}

raw_enum! {
    /// `VkSamplerAddressMode`.
    pub enum SamplerAddressMode: u32 {
        Repeat = 0,
        MirroredRepeat = 1,
        ClampToEdge = 2,
        ClampToBorder = 3,
        MirrorClampToEdge = 4,
    }
}

raw_enum! {
    /// `VkBorderColor`.
    pub enum BorderColor: u32 {
        FloatTransparentBlack = 0,
        IntTransparentBlack = 1,
        FloatOpaqueBlack = 2,
        IntOpaqueBlack = 3,
        FloatOpaqueWhite = 4,
        IntOpaqueWhite = 5,
    }
}

raw_enum! {
    /// `VkSamplerReductionMode`.
    pub enum SamplerReductionMode: u32 {
        WeightedAverage = 0,
        Min = 1,
        Max = 2,
    }
}

raw_enum! {
    /// `VkCompareOp`.
    pub enum CompareOp: u32 {
        Never = 0,
        Less = 1,
        Equal = 2,
        LessOrEqual = 3,
        Greater = 4,
        NotEqual = 5,
        GreaterOrEqual = 6,
        Always = 7,
    }
}

raw_enum! {
    /// `VkStencilOp`.
    pub enum StencilOp: u32 {
        Keep = 0,
        Zero = 1,
        Replace = 2,
        IncrementAndClamp = 3,
        DecrementAndClamp = 4,
        Invert = 5,
        IncrementAndWrap = 6,
        DecrementAndWrap = 7,
    }
}

raw_enum! {
    /// `VkBlendFactor`.
    pub enum BlendFactor: u32 {
        Zero = 0,
        One = 1,
        SrcColor = 2,
        OneMinusSrcColor = 3,
        DstColor = 4,
        OneMinusDstColor = 5,
        SrcAlpha = 6,
        OneMinusSrcAlpha = 7,
        DstAlpha = 8,
        OneMinusDstAlpha = 9,
        ConstantColor = 10,
        OneMinusConstantColor = 11,
        ConstantAlpha = 12,
        OneMinusConstantAlpha = 13,
        SrcAlphaSaturate = 14,
        Src1Color = 15,
        OneMinusSrc1Color = 16,
        Src1Alpha = 17,
        OneMinusSrc1Alpha = 18,
    }
}

raw_enum! {
    /// `VkBlendOp`.
    pub enum BlendOp: u32 {
        Add = 0,
        Subtract = 1,
        ReverseSubtract = 2,
        Min = 3,
        Max = 4,
    }
}

option_set! {
    /// `VkColorComponentFlags`.
    pub struct ColorComponents: u32 {
        const R = 0x1;
        const G = 0x2;
        const B = 0x4;
        const A = 0x8;
    }
}

raw_enum! {
    /// `VkPrimitiveTopology`.
    pub enum PrimitiveTopology: u32 {
        PointList = 0,
        LineList = 1,
        LineStrip = 2,
        TriangleList = 3,
        TriangleStrip = 4,
        PatchList = 10,
    }
}

raw_enum! {
    /// `VkPipelineBindPoint`.
    pub enum PipelineBindPoint: u32 {
        Graphics = 0,
        Compute = 1,
    }
}

raw_enum! {
    /// `VkPolygonMode`.
    pub enum PolygonMode: u32 {
        Fill = 0,
        Line = 1,
    }
}

option_set! {
    /// `VkCullModeFlags`.
    pub struct CullMode: u32 {
        const FRONT = 0x1;
        const BACK = 0x2;
    }
}

raw_enum! {
    /// `VkFrontFace`.
    pub enum FrontFace: u32 {
        CounterClockwise = 0,
        Clockwise = 1,
    }
}

raw_enum! {
    /// `VkIndexType`.
    pub enum IndexType: u32 {
        Uint16 = 0,
        Uint32 = 1,
    }
}

raw_enum! {
    /// `VkVertexInputRate`.
    pub enum VertexInputRate: u32 {
        Vertex = 0,
        Instance = 1,
    }
}

raw_enum! {
    /// `VkAttachmentLoadOp`.
    pub enum AttachmentLoadOp: u32 {
        Load = 0,
        Clear = 1,
        DontCare = 2,
    }
}

raw_enum! {
    /// `VkAttachmentStoreOp`.
    pub enum AttachmentStoreOp: u32 {
        Store = 0,
        DontCare = 1,
    }
}

option_set! {
    /// `VkResolveModeFlags`.
    pub struct ResolveModes: u32 {
        const SAMPLE_ZERO = 0x1;
        const AVERAGE = 0x2;
        const MIN = 0x4;
        const MAX = 0x8;
    }
}

option_set! {
    /// `VkShaderStageFlags`.
    pub struct ShaderStages: u32 {
        const VERTEX = 0x1;
        const TESSELLATION_CONTROL = 0x2;
        const TESSELLATION_EVALUATION = 0x4;
        const GEOMETRY = 0x8;
        const FRAGMENT = 0x10;
        const COMPUTE = 0x20;
        const TASK = 0x40;
        const MESH = 0x80;
    }
}

option_set! {
    /// `VkSubgroupFeatureFlags`.
    pub struct SubgroupFeatures: u32 {
        const BASIC = 0x1;
        const VOTE = 0x2;
        const ARITHMETIC = 0x4;
        const BALLOT = 0x8;
        const SHUFFLE = 0x10;
        const SHUFFLE_RELATIVE = 0x20;
        const CLUSTERED = 0x40;
        const QUAD = 0x80;
        const ROTATE = 0x200;
        const ROTATE_CLUSTERED = 0x400;
    }
}

option_set! {
    /// `VkPipelineStageFlags2`.
    pub struct PipelineStages: u64 {
        const TOP_OF_PIPE = 0x1;
        const DRAW_INDIRECT = 0x2;
        const VERTEX_INPUT = 0x4;
        const VERTEX_SHADER = 0x8;
        const TESSELLATION_CONTROL_SHADER = 0x10;
        const TESSELLATION_EVALUATION_SHADER = 0x20;
        const FRAGMENT_SHADER = 0x80;
        const EARLY_FRAGMENT_TESTS = 0x100;
        const LATE_FRAGMENT_TESTS = 0x200;
        const COLOR_ATTACHMENT_OUTPUT = 0x400;
        const COMPUTE_SHADER = 0x800;
        const ALL_TRANSFER = 0x1000;
        const BOTTOM_OF_PIPE = 0x2000;
        const HOST = 0x4000;
        const ALL_GRAPHICS = 0x8000;
        const ALL_COMMANDS = 0x1_0000;
        /// `VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT`.
        const COMMAND_PREPROCESS = 0x2_0000;
        const CONDITIONAL_RENDERING = 0x4_0000;
        const TASK_SHADER = 0x8_0000;
        const MESH_SHADER = 0x10_0000;
    }
}

option_set! {
    /// `VkAccessFlags2`.
    pub struct Access: u64 {
        const INDIRECT_COMMAND_READ = 0x1;
        const INDEX_READ = 0x2;
        const VERTEX_ATTRIBUTE_READ = 0x4;
        const UNIFORM_READ = 0x8;
        const SHADER_READ = 0x20;
        const SHADER_WRITE = 0x40;
        const COLOR_ATTACHMENT_READ = 0x80;
        const COLOR_ATTACHMENT_WRITE = 0x100;
        const DEPTH_STENCIL_ATTACHMENT_READ = 0x200;
        const DEPTH_STENCIL_ATTACHMENT_WRITE = 0x400;
        const TRANSFER_READ = 0x800;
        const TRANSFER_WRITE = 0x1000;
        const HOST_READ = 0x2000;
        const HOST_WRITE = 0x4000;
        const MEMORY_READ = 0x8000;
        const MEMORY_WRITE = 0x1_0000;
        const COMMAND_PREPROCESS_READ = 0x2_0000;
        const COMMAND_PREPROCESS_WRITE = 0x4_0000;
        const CONDITIONAL_RENDERING_READ = 0x10_0000;
    }
}

impl Access {
    /// Accesses that write memory and therefore need availability operations.
    pub const WRITES: Self = Self::SHADER_WRITE
        .union(Self::COLOR_ATTACHMENT_WRITE)
        .union(Self::DEPTH_STENCIL_ATTACHMENT_WRITE)
        .union(Self::TRANSFER_WRITE)
        .union(Self::HOST_WRITE)
        .union(Self::MEMORY_WRITE)
        .union(Self::COMMAND_PREPROCESS_WRITE);
}

option_set! {
    /// `VkQueryPipelineStatisticFlags`.
    pub struct PipelineStatistics: u32 {
        const INPUT_ASSEMBLY_VERTICES = 0x1;
        const INPUT_ASSEMBLY_PRIMITIVES = 0x2;
        const VERTEX_SHADER_INVOCATIONS = 0x4;
        const CLIPPING_INVOCATIONS = 0x20;
        const CLIPPING_PRIMITIVES = 0x40;
        const FRAGMENT_SHADER_INVOCATIONS = 0x80;
        const TESSELLATION_CONTROL_SHADER_PATCHES = 0x100;
        const TESSELLATION_EVALUATION_SHADER_INVOCATIONS = 0x200;
        const COMPUTE_SHADER_INVOCATIONS = 0x400;
    }
}

raw_enum! {
    /// `VkDynamicState`, for the states Metal sets on its encoders.
    pub enum DynamicState: u32 {
        DepthBias = 3,
        BlendConstants = 4,
        StencilCompareMask = 6,
        StencilWriteMask = 7,
        StencilReference = 8,
        SampleLocations = 1000143000,
        CullMode = 1000267000,
        FrontFace = 1000267001,
        PrimitiveTopology = 1000267002,
        ViewportWithCount = 1000267003,
        ScissorWithCount = 1000267004,
        VertexInputBindingStride = 1000267005,
        DepthTestEnable = 1000267006,
        DepthWriteEnable = 1000267007,
        DepthCompareOp = 1000267008,
        StencilTestEnable = 1000267010,
        StencilOp = 1000267011,
        RasterizerDiscardEnable = 1000377001,
        DepthBiasEnable = 1000377002,
        PrimitiveRestartEnable = 1000377004,
        DepthClampEnable = 1000455003,
        PolygonMode = 1000455004,
        DepthClipEnable = 1000455016,
        SampleLocationsEnable = 1000455017,
    }
}
