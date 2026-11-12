use super::{
    blit::{BlitTexture, BufferLayout, Region},
    counters::{CounterRange, CounterSample},
    device_generated::{GeneratedLayoutKey, GeneratedWork, IndirectRenderClass},
    indirect::ExecutionRange,
    render::{PassBoundary, PatchSource, ScissorRect, Viewport},
    *,
};
use crate::metal::{
    abi::{
        BarrierScope, BlitOption, CpuCacheMode, CullMode as CullModeAbi, DispatchType,
        HazardTrackingMode, IndexType, IndirectCommandType, LoadAction, PrimitiveTopologyClass,
        PrimitiveType, RenderStages, ResourceOptions, StorageMode, StoreAction, StoreActionOptions,
        SwizzleChannels, TessellationControlPointIndexType, TextureType, TextureUsage,
        VisibilityResultMode, VisibilityResultType,
    },
    binding::{
        AccumulateArguments, ArgumentTable, BufferBinding, DispatchArguments, GeneratedSequence,
        PushConstants, ResolveArguments, TessellationArguments,
    },
    capabilities::Size,
    error::Error,
    hazard::FULL,
    indirect::{
        IndirectCommandBuffer, IndirectCommandBufferDescriptor, IndirectCommandBufferId,
        IndirectPipeline, RecordLayout,
    },
    nvidia::fixture::capabilities,
    pass::{AttachedTexture, AttachmentDescriptor, RenderPassDescriptor, Target, render_pass},
    pipeline::{
        ColorAttachmentDescriptor, ComputePipelineDescriptor, MAX_COLOR_ATTACHMENTS,
        MeshRenderPipelineDescriptor, RenderPipelineDescriptor, RenderTargets, StageThreads,
        TessellatedVertex, TessellationDescriptor, compute_pipeline, mesh_render_pipeline,
        render_pipeline, tessellation,
    },
    pixel_format::{BlockKernel, PixelFormat},
    resource::TextureDescriptor,
    shader::{
        Access as ShaderAccess, AttributeUse, BufferUse, CompiledShader, DYNAMIC_BICUBIC,
        NumericClass, Patch, PatchDomain, Provenance, tests::interface,
    },
    vertex::{
        Attribute, Layout, STRIDE_DYNAMIC, StageInLayout, StageInStep, StageInputDescriptor,
        StepFunction, VertexDescriptor, VertexFormat,
    },
    vulkan::{
        Access, AttachmentLoadOp, AttachmentStoreOp, CullMode, PipelineBindPoint, PipelineStages,
        ShaderStages,
        backend::Requirements,
        command::{Barrier, BufferCopy, BufferImageCopy, GeneratedCommands},
        object::{
            ExecutionSet, Image, ImageView, IndirectCommandsLayout, IndirectToken, QueryType,
            ShaderCode,
        },
    },
};
use alloc::{sync::Arc, vec::Vec};

#[derive(Default)]
struct MockResolver {
    chunks: u64,
    compiled: Vec<Specialization>,
    pools: u64,
    /// The object threadgroup memory of the last mesh draw.
    object_lengths: Vec<(u32, u64)>,
    /// Pipeline states by id, for indirect commands.
    render_states: Vec<RenderPipelineState>,
    compute_states: Vec<ComputePipelineState>,
    indirect: Vec<Arc<IndirectCommandBuffer>>,
    classes: Vec<IndirectRenderClass>,
    compute_set: Option<(ExecutionSet, Pipeline)>,
    layouts: Vec<GeneratedLayoutKey>,
    /// The bicubic samplers of each render variant asked for.
    bicubic: Vec<[u32; 2]>,
}

impl ChunkSource for MockResolver {
    fn acquire(&mut self, bytes: u64) -> Result<Chunk> {
        self.chunks += 1;
        Ok(Chunk {
            buffer: Buffer(1000 + self.chunks),
            address: self.chunks << 40,
            size: bytes,
        })
    }
}

impl Resolver for MockResolver {
    fn compute_pipeline(
        &mut self,
        _: &ComputePipelineState,
        specialization: &Specialization,
    ) -> Result<Pipeline> {
        let index = match self
            .compiled
            .iter()
            .position(|known| known == specialization)
        {
            Some(index) => index,
            None => {
                self.compiled.push(specialization.clone());
                self.compiled.len() - 1
            }
        };
        Ok(Pipeline(500 + index as u64))
    }

    fn clamped_sampler(&mut self, sampler: ResourceId, _: f32, _: f32) -> Result<ResourceId> {
        Ok(ResourceId(sampler.0 + 100))
    }

    fn bicubic_render_pipeline(
        &mut self,
        _: &RenderPipelineState,
        bicubic: [u32; 2],
    ) -> Result<Pipeline> {
        self.bicubic.push(bicubic);
        Ok(Pipeline(800 + u64::from(bicubic[0] | bicubic[1] << 16)))
    }

    fn mesh_pipeline(
        &mut self,
        _: &RenderPipelineState,
        _: [u32; 3],
        _: [u32; 3],
        object_lengths: &[(u32, u64)],
    ) -> Result<Pipeline> {
        self.object_lengths = object_lengths.to_vec();
        Ok(Pipeline(900))
    }

    fn query_pool(&mut self, _: QueryType) -> Result<(QueryPool, u32)> {
        self.pools += 1;
        Ok((QueryPool(self.pools), 2))
    }

    fn render_pipeline_state(&mut self, id: u64) -> Result<RenderPipelineState> {
        let known = self.render_states.iter().find(|state| state.id == id);
        known.cloned().ok_or(Error::Invalid("unknown pipeline"))
    }

    fn compute_pipeline_state(&mut self, id: u64) -> Result<ComputePipelineState> {
        let known = self.compute_states.iter().find(|state| state.id == id);
        known.cloned().ok_or(Error::Invalid("unknown pipeline"))
    }

    fn indirect_commands(
        &mut self,
        id: IndirectCommandBufferId,
    ) -> Result<Arc<IndirectCommandBuffer>> {
        let known = self.indirect.get(id.0 as usize);
        known.cloned().ok_or(Error::Invalid("unknown object"))
    }

    fn indirect_render_classes(&mut self) -> Vec<IndirectRenderClass> {
        self.classes.clone()
    }

    fn indirect_compute_set(&mut self) -> Option<(ExecutionSet, Pipeline)> {
        self.compute_set
    }

    /// Layout `n` is the `n`th key asked for; each sequence preprocesses
    /// into 96 bytes at 64-byte alignment.
    fn generated_layout(
        &mut self,
        key: &GeneratedLayoutKey,
        _: Option<ExecutionSet>,
        sequences: u32,
    ) -> Result<(IndirectCommandsLayout, Requirements)> {
        let index = match self.layouts.iter().position(|known| known == key) {
            Some(index) => index,
            None => {
                self.layouts.push(key.clone());
                self.layouts.len() - 1
            }
        };
        let requirements = Requirements {
            size: 96 * u64::from(sequences),
            alignment: 64,
            memory_types: u32::MAX,
        };
        Ok((IndirectCommandsLayout(index as u64), requirements))
    }

    /// Places records in buffer `2000 + id` at address `id << 44`.
    fn indirect_records(
        &mut self,
        id: IndirectCommandBufferId,
    ) -> Result<(RecordLayout, BufferRef)> {
        let known = self.indirect.get_mut(id.0 as usize);
        let contents = Arc::make_mut(known.ok_or(Error::Invalid("unknown object"))?);
        let layout = match contents.records() {
            Some(layout) => *layout,
            None => {
                let (layout, _) = contents.resident_image()?;
                contents.make_resident(layout);
                layout
            }
        };
        let records = BufferRef {
            buffer: Buffer(2000 + u64::from(id.0)),
            offset: 0,
            address: u64::from(id.0) << 44,
            length: layout.bytes(),
        };
        Ok((layout, records))
    }

    fn update_indirect_commands(
        &mut self,
        id: IndirectCommandBufferId,
        update: &mut dyn FnMut(&mut IndirectCommandBuffer) -> Result<()>,
    ) -> Result<()> {
        let known = self.indirect.get_mut(id.0 as usize);
        update(Arc::make_mut(
            known.ok_or(Error::Invalid("unknown object"))?,
        ))
    }
}

const FEATURES: DynamicFeatures = DynamicFeatures {
    depth_clamp: true,
    polygon_mode: true,
    sample_locations: false,
};

fn texture(pixel_format: PixelFormat) -> TextureDescriptor {
    TextureDescriptor {
        texture_type: TextureType::D2,
        pixel_format,
        width: 64,
        height: 32,
        depth: 1,
        mipmap_level_count: 3,
        sample_count: 1,
        array_length: 1,
        options: ResourceOptions {
            cpu_cache_mode: CpuCacheMode::DefaultCache,
            storage_mode: StorageMode::Private,
            hazard_tracking_mode: HazardTrackingMode::Default,
        },
        usage: TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        swizzle: SwizzleChannels::IDENTITY,
    }
}

fn pass(
    color: &TextureDescriptor,
    store: StoreAction,
    visibility: bool,
) -> crate::metal::pass::RenderPassPlan<ImageView> {
    let mut descriptor = RenderPassDescriptor {
        color_attachments: [None; MAX_COLOR_ATTACHMENTS],
        depth_attachment: None,
        depth_resolve_filter: crate::metal::abi::MultisampleDepthResolveFilter::Sample0,
        stencil_attachment: None,
        stencil_resolve_filter: crate::metal::abi::MultisampleStencilResolveFilter::Sample0,
        visibility_results: visibility,
        render_target_array_length: 0,
        render_target_width: 0,
        render_target_height: 0,
        default_raster_sample_count: 0,
        tile_memory: Default::default(),
        sample_positions: Vec::new(),
        rasterization_rate_map: false,
    };
    descriptor.color_attachments[0] = Some(AttachmentDescriptor {
        target: Target {
            texture: AttachedTexture {
                texture: ImageView(7),
                descriptor: color,
            },
            level: 0,
            slice: 0,
            depth_plane: 0,
        },
        resolve: None,
        load_action: LoadAction::Clear,
        store_action: store,
        store_action_options: StoreActionOptions::EMPTY,
        clear: [0.0, 0.0, 0.0, 1.0],
    });
    render_pass(&capabilities(), &descriptor).unwrap()
}

fn shader(stage: Stage, edit: impl FnOnce(&mut Interface)) -> CompiledShader {
    let mut interface = interface(stage);
    edit(&mut interface);
    CompiledShader {
        code: ShaderCode(stage.raw()),
        interface,
        provenance: Provenance::default(),
    }
}

pub(crate) fn render_state(amplification: u64) -> RenderPipelineState {
    let vertex = shader(Stage::Vertex, |_| {});
    let fragment = shader(Stage::Fragment, |_| {});
    let layout = VertexDescriptor {
        attributes: alloc::vec![Attribute {
            index: 0,
            format: VertexFormat::Float4,
            offset: 0,
            buffer_index: 2
        }],
        layouts: alloc::vec![Layout {
            buffer_index: 2,
            stride: 16,
            step_function: StepFunction::PerVertex,
            step_rate: 1
        }],
    };
    let mut targets = RenderTargets::default();
    targets.color_attachments[0] = ColorAttachmentDescriptor {
        pixel_format: Some(PixelFormat::Rgba8Unorm),
        ..ColorAttachmentDescriptor::default()
    };
    let descriptor = RenderPipelineDescriptor {
        vertex_function: &vertex,
        fragment_function: Some(&fragment),
        vertex_descriptor: Some(&layout),
        targets,
        input_primitive_topology: PrimitiveTopologyClass::Triangle,
        max_vertex_amplification_count: amplification,
        tessellation: None,
    };
    let plan = render_pipeline(&capabilities(), &descriptor).unwrap();
    RenderPipelineState {
        id: 0,
        pipeline: Some(Pipeline(42)),
        plan,
        interfaces: alloc::vec![vertex.interface, fragment.interface],
        indirect: None,
        bicubic: [0; 2],
        outdated: false,
    }
}

/// A pipeline of a quad post-tessellation function with four control
/// points, indexed by 16-bit control point indices, which fetches its
/// stage-in by `layout`.
fn patch_state(layout: Option<&VertexDescriptor>) -> RenderPipelineState {
    patch_state_of(layout, Some(4), false)
}

/// A pipeline of a quad patch function of `control_points`, whose
/// attributes are `layout`'s, each a control point's when
/// `reads_control_points`.
fn patch_state_of(
    layout: Option<&VertexDescriptor>,
    control_points: Option<u32>,
    reads_control_points: bool,
) -> RenderPipelineState {
    let evaluation = shader(Stage::Vertex, |interface| {
        interface.patch = Some(Patch {
            domain: PatchDomain::Quad,
            control_points,
        });
        if let Some(layout) = layout {
            interface.attributes = layout
                .attributes
                .iter()
                .map(|attribute| AttributeUse {
                    index: attribute.index,
                    numeric: NumericClass::Float,
                    components: 4,
                    patch_control_point: reads_control_points,
                })
                .collect();
        }
    });
    let properties = TessellationDescriptor {
        factor_scale_enabled: true,
        control_point_index_type: TessellationControlPointIndexType::UInt16,
        ..TessellationDescriptor::default()
    };
    let tessellation = tessellation(&capabilities(), &evaluation.interface, &properties, layout)
        .unwrap()
        .unwrap();
    let mut targets = RenderTargets::default();
    targets.color_attachments[0].pixel_format = Some(PixelFormat::Rgba8Unorm);
    let descriptor = RenderPipelineDescriptor {
        vertex_function: &evaluation,
        fragment_function: None,
        vertex_descriptor: None,
        targets,
        input_primitive_topology: PrimitiveTopologyClass::Unspecified,
        max_vertex_amplification_count: 1,
        tessellation: Some(TessellatedVertex {
            tessellation: &tessellation,
            vertex: ShaderCode(100),
            control: ShaderCode(200),
        }),
    };
    RenderPipelineState {
        id: 1,
        pipeline: Some(Pipeline(43)),
        plan: render_pipeline(&capabilities(), &descriptor).unwrap(),
        interfaces: alloc::vec![evaluation.interface],
        indirect: None,
        bicubic: [0; 2],
        outdated: false,
    }
}

#[test]
fn patch_draws_upload_their_tessellation_arguments() {
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    commands
        .set_render_pipeline_state(&patch_state(None))
        .unwrap();
    let source = PatchSource {
        control_points: 4,
        patch_indices: None,
        control_point_indices: Some((buffer(5, 4096), 64)),
    };
    assert_eq!(
        commands.draw_patches(&mut resolver, &source, 0, 1, 1, 0),
        Err(Error::Invalid("no tessellation factor buffer is set"))
    );
    commands
        .set_tessellation_factor_buffer(Some(buffer(4, 1024)), 96, 48)
        .unwrap();
    commands.set_tessellation_factor_scale(2.0).unwrap();
    assert!(
        commands
            .draw_primitives(&mut resolver, PrimitiveType::Triangle, 0, 3, 1, 0)
            .is_err()
    );
    let wrong_count = PatchSource {
        control_points: 3,
        ..source
    };
    assert!(
        commands
            .draw_patches(&mut resolver, &wrong_count, 0, 1, 1, 0)
            .is_err()
    );
    let unindexed = PatchSource {
        control_point_indices: None,
        ..source
    };
    assert!(
        commands
            .draw_patches(&mut resolver, &unindexed, 0, 1, 1, 0)
            .is_err()
    );
    commands
        .draw_patches(&mut resolver, &source, 8, 6, 2, 1)
        .unwrap();
    commands
        .draw_patches_indirect(&mut resolver, &source, buffer(6, 64), 16)
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;

    assert!(commands.contains(&Command::SetPrimitiveTopology(
        crate::metal::vulkan::PrimitiveTopology::PatchList
    )));
    assert!(commands.contains(&Command::Draw {
        vertex_count: 6,
        instance_count: 2,
        first_vertex: 8,
        first_instance: 1,
    }));
    assert!(commands.contains(&Command::DrawIndirect {
        buffer: Buffer(6),
        offset: 16,
    }));
    let constants = commands.iter().rev().find_map(|command| match command {
        Command::PushConstants(constants) => Some(*constants),
        _ => None,
    });
    let constants: PushConstants = constants.unwrap();
    let (chunk, bytes) = &recording.uploads[0];
    let arguments = &bytes[(constants.arguments - chunk.address) as usize..]
        [..TessellationArguments::BYTES as usize];
    let expected = TessellationArguments {
        factors: BufferBinding {
            address: (4 << 32) + 96,
            length: 1024 - 96,
        },
        patch_indices: BufferBinding::default(),
        control_point_indices: BufferBinding {
            address: (5 << 32) + 64,
            length: 4096 - 64,
        },
        instance_stride: 48,
        factor_scale: 2.0,
        control_points: 4,
        strides: [0; 31],
    };
    assert_eq!(arguments, expected.encode());
}

/// A patch function that reads no control points draws patches of none.
#[test]
fn patches_of_no_control_points_draw_when_none_are_read() {
    let layout = VertexDescriptor {
        attributes: alloc::vec![Attribute {
            index: 0,
            format: VertexFormat::Float4,
            offset: 0,
            buffer_index: 2,
        }],
        layouts: alloc::vec![Layout {
            buffer_index: 2,
            stride: 16,
            step_function: StepFunction::PerPatchControlPoint,
            step_rate: 1,
        }],
    };
    for reads in [false, true] {
        let color = texture(PixelFormat::Rgba8Unorm);
        let mut resolver = MockResolver::default();
        let mut commands = CommandBuffer::new(FEATURES);
        commands
            .begin_render(
                &mut resolver,
                pass(&color, StoreAction::Store, false),
                None,
                Vec::new(),
            )
            .unwrap();
        commands
            .set_render_pipeline_state(&patch_state_of(reads.then_some(&layout), None, reads))
            .unwrap();
        commands
            .set_tessellation_factor_buffer(Some(buffer(4, 1024)), 0, 48)
            .unwrap();
        let source = PatchSource {
            control_points: 0,
            patch_indices: None,
            control_point_indices: Some((buffer(5, 4096), 0)),
        };
        assert_eq!(
            commands
                .draw_patches(&mut resolver, &source, 0, 2, 1, 0)
                .is_ok(),
            !reads
        );
    }
}

/// The evaluation stage fetches stage-in from the bound vertex buffers, at
/// the strides bound with layouts of dynamic stride.
#[test]
fn patch_draws_bind_their_stage_in_strides() {
    let layout = VertexDescriptor {
        attributes: alloc::vec![Attribute {
            index: 0,
            format: VertexFormat::Float4,
            offset: 0,
            buffer_index: 2,
        }],
        layouts: alloc::vec![Layout {
            buffer_index: 2,
            stride: STRIDE_DYNAMIC,
            step_function: StepFunction::PerPatchControlPoint,
            step_rate: 1,
        }],
    };
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    commands
        .set_render_pipeline_state(&patch_state(Some(&layout)))
        .unwrap();
    commands
        .set_tessellation_factor_buffer(Some(buffer(4, 1024)), 0, 0)
        .unwrap();
    let source = PatchSource {
        control_points: 4,
        patch_indices: None,
        control_point_indices: Some((buffer(5, 4096), 0)),
    };
    let mut draw =
        |commands: &mut CommandBuffer| commands.draw_patches(&mut resolver, &source, 0, 1, 1, 0);
    assert_eq!(
        draw(&mut commands),
        Err(Error::Invalid(
            "a vertex buffer the pipeline reads is unbound"
        ))
    );
    commands
        .set_render_buffer(Stage::Vertex, Some(buffer(7, 256)), 0, 2)
        .unwrap();
    assert_eq!(
        draw(&mut commands),
        Err(Error::Invalid(
            "a dynamic-stride vertex buffer was bound without a stride"
        ))
    );
    commands
        .set_vertex_buffer_with_stride(buffer(7, 256), 2, 32, 2)
        .unwrap();
    assert!(
        draw(&mut commands).is_err(),
        "offsets are multiples of four"
    );
    assert_eq!(
        commands.set_vertex_buffer_offset_with_stride(16, 32, 3),
        Err(Error::Invalid("no vertex buffer is bound at the index"))
    );
    // The bound buffer moves to an offset and stride of its own.
    commands
        .set_vertex_buffer_offset_with_stride(16, 32, 2)
        .unwrap();
    draw(&mut commands).unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let constants = recording.segments[0]
        .commands
        .iter()
        .rev()
        .find_map(|command| match command {
            Command::PushConstants(constants) => Some(*constants),
            _ => None,
        })
        .unwrap();
    let (chunk, bytes) = &recording.uploads[0];
    let arguments = &bytes[(constants.arguments - chunk.address) as usize..];
    let stride = TessellationArguments::STRIDES_OFFSET as usize + 4 * 2;
    assert_eq!(arguments[stride..][..4], 32u32.to_le_bytes());
    let count = TessellationArguments::CONTROL_POINTS_OFFSET as usize;
    assert_eq!(arguments[count..][..4], 4u32.to_le_bytes());
}

/// The grid of `drawMeshThreads` is covered by threadgroups of the stage it
/// launches: the object function when there is one.
#[test]
fn mesh_thread_draws_cover_the_grid_with_the_first_stage() {
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut targets = RenderTargets::default();
    targets.color_attachments[0].pixel_format = Some(PixelFormat::Rgba8Unorm);
    for (object, groups) in [(true, [4, 1, 1]), (false, [13, 1, 1])] {
        let object_function = interface(Stage::Object);
        let mesh_function = interface(Stage::Mesh);
        let descriptor = MeshRenderPipelineDescriptor {
            object_function: object.then_some(&object_function),
            mesh_function: &mesh_function,
            fragment_function: None,
            targets,
            max_vertex_amplification_count: 1,
            object_threads: StageThreads::default(),
            mesh_threads: StageThreads::default(),
            payload_memory_length: 0,
        };
        let state = RenderPipelineState {
            id: 2,
            pipeline: None,
            plan: mesh_render_pipeline(&capabilities(), &descriptor).unwrap(),
            interfaces: object
                .then(|| object_function.clone())
                .into_iter()
                .chain([mesh_function.clone()])
                .collect(),
            indirect: None,
            bicubic: [0; 2],
            outdated: false,
        };
        let mut resolver = MockResolver::default();
        let mut commands = CommandBuffer::new(FEATURES);
        commands
            .begin_render(
                &mut resolver,
                pass(&color, StoreAction::Store, false),
                None,
                Vec::new(),
            )
            .unwrap();
        commands.set_render_pipeline_state(&state).unwrap();
        commands
            .set_object_threadgroup_memory_length(64, 2)
            .unwrap();
        commands
            .set_object_threadgroup_memory_length(16, 0)
            .unwrap();
        commands
            .set_object_threadgroup_memory_length(32, 2)
            .unwrap();
        commands
            .draw_mesh_threads(&mut resolver, [100, 1, 1], [32, 1, 1], [8, 1, 1])
            .unwrap();
        assert_eq!(resolver.object_lengths, [(0, 16), (2, 32)]);
        commands.end_encoding(&mut resolver).unwrap();
        let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
        assert!(
            recording.segments[0]
                .commands
                .contains(&Command::DrawMeshTasks(groups))
        );
    }
}

fn buffer(id: u64, length: u64) -> BufferRef {
    BufferRef {
        buffer: Buffer(id),
        offset: 0,
        address: id << 32,
        length,
    }
}

/// Finishes `commands` and takes their recording.
fn commit(
    mut commands: CommandBuffer,
    resolver: &mut MockResolver,
    completion: (Semaphore, u64),
) -> Recording {
    commands.finish(resolver, completion).unwrap();
    commands.into_recording()
}

fn count(commands: &[Command], matches: impl Fn(&Command) -> bool) -> usize {
    commands.iter().filter(|command| matches(command)).count()
}

#[test]
fn a_draw_binds_every_buffer_its_functions_access() {
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    let mut state = render_state(1);
    state.interfaces[1].buffer_reads = 1 << 1;
    commands.set_render_pipeline_state(&state).unwrap();
    commands
        .set_render_buffer(Stage::Vertex, Some(buffer(3, 256)), 0, 2)
        .unwrap();
    let draw = |commands: &mut CommandBuffer, resolver: &mut MockResolver| {
        commands.draw_primitives(resolver, PrimitiveType::Triangle, 0, 3, 1, 0)
    };
    // Unbound, the fragment function would read its buffer at address zero.
    assert_eq!(
        draw(&mut commands, &mut resolver),
        Err(Error::Invalid(
            "the fragment function accesses buffer 1, which is unbound"
        ))
    );
    // A vertex buffer at that index binds the other stage's table.
    commands
        .set_render_buffer(Stage::Vertex, Some(buffer(4, 256)), 0, 1)
        .unwrap();
    assert!(draw(&mut commands, &mut resolver).is_err());
    // Inline bytes bind it as a buffer does.
    commands
        .set_render_bytes(&mut resolver, Stage::Fragment, &[9; 12], 1)
        .unwrap();
    draw(&mut commands, &mut resolver).unwrap();
}

#[test]
fn draws_through_bicubic_samplers_bind_the_variant_filtering_them() {
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    let mut state = render_state(1);
    state.interfaces[1].samplers = alloc::vec![0, 2];
    commands.set_render_pipeline_state(&state).unwrap();
    commands
        .set_render_buffer(Stage::Vertex, Some(buffer(3, 256)), 0, 2)
        .unwrap();
    let linear = ResourceId::sampler(1, 2, 0);
    let cubic = ResourceId::sampler(3, 4, ResourceId::SAMPLER_CUBIC);
    let mut draw_with = |sampler: Option<(Stage, ResourceId, u32)>| {
        if let Some((stage, sampler, index)) = sampler {
            commands
                .set_render_sampler(&mut resolver, stage, sampler, None, index)
                .unwrap();
        }
        commands
            .draw_primitives(&mut resolver, PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    };
    // A linear sampler draws with the pipeline.
    draw_with(Some((Stage::Fragment, linear, 0)));
    // A bicubic sampler the fragment function samples binds the variant
    // filtering it bicubically, which the next draw keeps.
    draw_with(Some((Stage::Fragment, cubic, 2)));
    draw_with(None);
    // Bicubic samplers where no function samples change nothing.
    draw_with(Some((Stage::Fragment, cubic, 5)));
    draw_with(Some((Stage::Vertex, cubic, 0)));
    // Linear again, the draw binds the pipeline.
    draw_with(Some((Stage::Fragment, linear, 2)));
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    assert_eq!(resolver.bicubic, [[0, 1 << 2]; 4]);
    let bound: Vec<_> = recording.segments[0]
        .commands
        .iter()
        .filter_map(|command| match command {
            Command::BindPipeline {
                bind_point: PipelineBindPoint::Graphics,
                pipeline,
            } => Some(pipeline.0),
            _ => None,
        })
        .collect();
    assert_eq!(bound, [42, 800 + (1 << 2 << 16), 42]);
}

#[test]
fn draws_bind_no_variant_for_samplers_their_pipeline_filters_bicubically() {
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    let mut state = render_state(1);
    state.interfaces[1].samplers = alloc::vec![2];
    state.bicubic = [0, DYNAMIC_BICUBIC];
    commands.set_render_pipeline_state(&state).unwrap();
    commands
        .set_render_buffer(Stage::Vertex, Some(buffer(3, 256)), 0, 2)
        .unwrap();
    let cubic = ResourceId::sampler(3, 4, ResourceId::SAMPLER_CUBIC);
    commands
        .set_render_sampler(&mut resolver, Stage::Fragment, cubic, None, 2)
        .unwrap();
    commands
        .draw_primitives(&mut resolver, PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    assert!(resolver.bicubic.is_empty());
}

/// A pipeline made before its process's first bicubic or pixel-coordinate
/// sampler draws through a variant compiled since, whatever it binds.
#[test]
fn draws_through_outdated_pipelines_bind_a_variant() {
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    let mut state = render_state(1);
    state.outdated = true;
    commands.set_render_pipeline_state(&state).unwrap();
    commands
        .set_render_buffer(Stage::Vertex, Some(buffer(3, 256)), 0, 2)
        .unwrap();
    commands
        .draw_primitives(&mut resolver, PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    assert_eq!(resolver.bicubic, [[0; 2]]);
}

#[test]
fn dispatches_through_bicubic_samplers_run_the_variant_filtering_them() {
    let threads = Size {
        width: 64,
        height: 1,
        depth: 1,
    };
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands.begin_compute(DispatchType::Serial).unwrap();
    let mut state = kernel_state(false);
    state.interface.samplers = alloc::vec![1, 3];
    commands.set_compute_pipeline_state(&state).unwrap();
    commands
        .set_compute_buffer(Some(buffer(4, 1024)), 0, 0)
        .unwrap();
    let cubic = ResourceId::sampler(3, 4, ResourceId::SAMPLER_CUBIC);
    for (sampler, index) in [(ResourceId::sampler(1, 2, 0), 1), (cubic, 3), (cubic, 4)] {
        commands
            .set_compute_sampler(&mut resolver, sampler, None, index)
            .unwrap();
        commands
            .dispatch_threadgroups(&mut resolver, [4, 1, 1], threads)
            .unwrap();
    }
    let bicubic: Vec<_> = resolver
        .compiled
        .iter()
        .map(|specialization| specialization.bicubic_samplers)
        .collect();
    assert_eq!(bicubic, [0, 1 << 3]);
}

/// Shaders divide a pixel-coordinate sampler's coordinates themselves, so
/// dispatches binding one run the variant that reads its flags, as MPS's
/// blurs do with mipmapped sources.
#[test]
fn dispatches_through_pixel_coordinate_samplers_run_the_variant_reading_them() {
    let threads = Size {
        width: 64,
        height: 1,
        depth: 1,
    };
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands.begin_compute(DispatchType::Serial).unwrap();
    let mut state = kernel_state(false);
    state.interface.samplers = alloc::vec![0];
    commands.set_compute_pipeline_state(&state).unwrap();
    commands
        .set_compute_buffer(Some(buffer(4, 1024)), 0, 0)
        .unwrap();
    let pixels = ResourceId::sampler(3, 3, ResourceId::SAMPLER_PIXELS);
    commands
        .set_compute_sampler(&mut resolver, pixels, None, 0)
        .unwrap();
    commands
        .dispatch_threadgroups(&mut resolver, [4, 1, 1], threads)
        .unwrap();
    let variants: Vec<_> = resolver
        .compiled
        .iter()
        .map(|specialization| specialization.bicubic_samplers)
        .collect();
    assert_eq!(variants, [1 << 0]);
}

#[test]
fn a_draw_binds_state_buffers_and_tables() {
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    commands
        .set_render_pipeline_state(&render_state(1))
        .unwrap();
    commands
        .set_render_buffer(Stage::Vertex, Some(buffer(3, 256)), 64, 2)
        .unwrap();
    commands
        .set_render_bytes(&mut resolver, Stage::Fragment, &[9; 12], 0)
        .unwrap();
    commands
        .set_viewports(&[Viewport {
            origin_x: 0.0,
            origin_y: 0.0,
            width: 64.0,
            height: 32.0,
            znear: 0.0,
            zfar: 1.0,
        }])
        .unwrap();
    commands
        .draw_primitives(&mut resolver, PrimitiveType::TriangleStrip, 0, 4, 1, 0)
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 5));

    assert_eq!(recording.segments.len(), 1);
    let segment = &recording.segments[0];
    assert_eq!(segment.signals, [(Semaphore(1), 5)]);
    let commands = &segment.commands;
    assert_eq!(commands[0], Command::Barrier(FULL));
    let Command::BeginRendering(rendering) = &commands[1] else {
        panic!("{:?}", commands[1])
    };
    let attachment = rendering.color[0].unwrap();
    assert_eq!(
        (attachment.view, attachment.load_op, attachment.store_op),
        (
            ImageView(7),
            AttachmentLoadOp::Clear,
            AttachmentStoreOp::Store
        )
    );
    assert!(commands.contains(&Command::SetPrimitiveTopology(
        crate::metal::vulkan::PrimitiveTopology::TriangleStrip
    )));
    assert!(commands.contains(&Command::SetPrimitiveRestart(true)));
    assert!(commands.contains(&Command::BindPipeline {
        bind_point: PipelineBindPoint::Graphics,
        pipeline: Pipeline(42),
    }));
    let flipped = commands.iter().find_map(|command| match command {
        Command::SetViewports(viewports) if viewports[0].height < 0.0 => Some(viewports[0]),
        _ => None,
    });
    assert_eq!(
        flipped.map(|viewport| (viewport.y, viewport.height)),
        Some((32.0, -32.0))
    );
    let vertex = commands.iter().find_map(|command| match command {
        Command::BindVertexBuffers { first: 2, buffers } => Some(buffers[0]),
        _ => None,
    });
    assert_eq!(
        vertex.map(|vertex| (vertex.buffer, vertex.offset, vertex.size, vertex.stride)),
        Some((Buffer(3), 64, 192, Some(16)))
    );
    let draw = commands
        .iter()
        .position(|command| {
            matches!(
                command,
                Command::Draw {
                    vertex_count: 4,
                    ..
                }
            )
        })
        .unwrap();
    let push = commands
        .iter()
        .position(|command| matches!(command, Command::PushConstants(_)))
        .unwrap();
    assert!(push < draw);
    assert_eq!(commands.last(), Some(&Command::EndRendering));

    // Both stages' tables were uploaded after the inline bytes.
    let Command::PushConstants(constants) = commands[push] else {
        unreachable!()
    };
    let (chunk, bytes) = &recording.uploads[0];
    let table_at = |address: u64| {
        &bytes[(address - chunk.address) as usize..][..ArgumentTable::BYTES as usize]
    };
    let vertex_table = table_at(constants.tables[0]);
    assert_eq!(
        u64::from_le_bytes(vertex_table[16..24].try_into().unwrap()),
        (3 << 32) + 64
    );
    let fragment_table = table_at(constants.tables[1]);
    let bytes_address = u64::from_le_bytes(fragment_table[..8].try_into().unwrap());
    assert_eq!(
        &bytes[(bytes_address - chunk.address) as usize..][..12],
        &[9; 12]
    );
}

#[test]
fn render_barriers_split_the_pass_and_keep_its_contents() {
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::DontCare, false),
            None,
            Vec::new(),
        )
        .unwrap();
    commands
        .render_memory_barrier(
            &mut resolver,
            BarrierScope::TEXTURES,
            RenderStages::FRAGMENT,
            RenderStages::FRAGMENT,
        )
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let begins: Vec<_> = recording.segments[0]
        .commands
        .iter()
        .filter_map(|command| match command {
            Command::BeginRendering(rendering) => {
                rendering.color[0].map(|color| (color.load_op, color.store_op))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        begins,
        [
            (AttachmentLoadOp::Clear, AttachmentStoreOp::Store),
            (AttachmentLoadOp::Load, AttachmentStoreOp::DontCare)
        ]
    );
    assert_eq!(
        count(&recording.segments[0].commands, |command| *command
            == Command::EndRendering),
        2
    );
}

/// A pass with an `imageblockSampleLength` places its imageblock data in
/// whole tiles, which its fragment and tile tables name; a function whose
/// imageblock is larger is refused.
#[test]
fn passes_place_imageblock_data_in_whole_tiles() {
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    let mut plan = pass(&color, StoreAction::Store, false);
    plan.tile.size = [32, 16];
    plan.tile.imageblock_sample_length = 32;
    commands
        .begin_render(&mut resolver, plan, None, Vec::new())
        .unwrap();
    let mut state = kernel_state(true);
    state.interface.imageblock_bytes = 48;
    let mut formats = [None; MAX_COLOR_ATTACHMENTS];
    formats[0] = Some(PixelFormat::Rgba8Unorm);
    let target = crate::metal::pipeline::tile_target(&capabilities(), &formats, 1, false).unwrap();
    commands.set_tile_pipeline_state(&state, target).unwrap();
    let threads = Size {
        width: 32,
        height: 16,
        depth: 1,
    };
    assert!(
        commands
            .dispatch_threads_per_tile(&mut resolver, threads)
            .is_err()
    );
    state.interface.imageblock_bytes = 32;
    commands.set_tile_pipeline_state(&state, target).unwrap();
    commands
        .dispatch_threads_per_tile(&mut resolver, threads)
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let pushed = recording.segments[0]
        .commands
        .iter()
        .find_map(|command| match command {
            Command::PushConstants(constants) if constants.arguments != 0 => Some(*constants),
            _ => None,
        })
        .unwrap();
    let table = uploaded(&recording, pushed.tables[0]);
    let at = ArgumentTable::IMAGEBLOCK_OFFSET as usize;
    // 64 by 32 pixels are whole tiles of 32 by 16.
    assert_eq!(word(table, at), 32);
    assert_eq!(word(table, at + 4), 64);
    assert_ne!(double(table, at + 8), 0);
}

/// A tile dispatch runs between pass instances: a threadgroup of the tile's
/// threads per tile, its arguments carrying the tile and the pass's
/// persistent tile memory, and its tables the encoder's tile bindings.
#[test]
fn tile_dispatches_run_a_threadgroup_per_tile() {
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    let mut plan = pass(&color, StoreAction::Store, false);
    plan.tile.threadgroup_memory_length = 32;
    commands
        .begin_render(&mut resolver, plan, None, Vec::new())
        .unwrap();
    let mut state = kernel_state(true);
    state.interface.threadgroup = alloc::vec![crate::metal::shader::ThreadgroupUse {
        index: 2,
        declared_bytes: 16,
    }];
    let mut formats = [None; MAX_COLOR_ATTACHMENTS];
    formats[0] = Some(PixelFormat::Rgba8Unorm);
    let target = crate::metal::pipeline::tile_target(&capabilities(), &formats, 1, false).unwrap();
    commands.set_tile_pipeline_state(&state, target).unwrap();
    commands.set_tile_buffer(Some(buffer(3, 64)), 0, 0).unwrap();
    let threads = Size {
        width: 16,
        height: 8,
        depth: 1,
    };
    // Its threadgroup buffer needs a place in the tile's memory.
    assert!(
        commands
            .dispatch_threads_per_tile(&mut resolver, threads)
            .is_err()
    );
    commands.set_tile_threadgroup_memory(16, 32, 2).unwrap();
    assert!(
        commands
            .dispatch_threads_per_tile(&mut resolver, threads)
            .is_err()
    );
    commands.set_tile_threadgroup_memory(16, 16, 2).unwrap();
    commands
        .dispatch_threads_per_tile(&mut resolver, threads)
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let specialization = resolver.compiled.last().unwrap().clone();
    assert!(specialization.tile && specialization.dynamic_threadgroups);
    assert_eq!(specialization.workgroup_size, Some([128, 1, 1]));
    assert_eq!(specialization.storage_attachments, target.attachments);
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;
    // 64 by 32 pixels are two tiles of 32 by 32.
    assert_eq!(
        count(commands, |command| *command == Command::Dispatch([2, 1, 1])),
        1
    );
    assert_eq!(
        count(commands, |command| *command == Command::EndRendering),
        2
    );
    let pushed = commands
        .iter()
        .find_map(|command| match command {
            Command::PushConstants(constants) if constants.arguments != 0 => Some(*constants),
            _ => None,
        })
        .unwrap();
    let table = uploaded(&recording, pushed.tables[0]);
    assert_eq!(
        double(table, ArgumentTable::BUFFER_ADDRESSES_OFFSET as usize),
        3 << 32
    );
    let arguments = uploaded(&recording, pushed.arguments);
    assert_eq!(
        word(arguments, DispatchArguments::SHAPE_OFFSET as usize),
        16
    );
    assert_eq!(
        word(arguments, DispatchArguments::SHAPE_OFFSET as usize + 4),
        8
    );
    assert_eq!(
        word(arguments, DispatchArguments::OFFSETS_OFFSET as usize + 8),
        16
    );
    let tile = DispatchArguments::TILE_OFFSET as usize;
    assert_eq!(word(arguments, tile), 32 | 32 << 16);
    assert_eq!(word(arguments, tile + 4), 32);
    assert_ne!(double(arguments, tile + 8), 0);
}

#[test]
fn deferred_store_actions_must_be_set() {
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Unknown, false),
            None,
            Vec::new(),
        )
        .unwrap();
    assert_eq!(
        commands.end_encoding(&mut resolver),
        Err(Error::Invalid("a store action was never set"))
    );

    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Unknown, false),
            None,
            Vec::new(),
        )
        .unwrap();
    commands
        .set_store_action(
            crate::metal::pass::AttachmentSlot::Color(0),
            StoreAction::DontCare,
        )
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
}

#[test]
fn visibility_results_copy_or_sum_their_queries() {
    // An accumulating pass sums each offset's queries after what the
    // buffer holds there.
    for kind in [
        VisibilityResultType::Reset,
        VisibilityResultType::Accumulate,
    ] {
        visibility_results(kind);
    }
}

fn visibility_results(kind: VisibilityResultType) {
    let accumulate = kind == VisibilityResultType::Accumulate;
    let color = texture(PixelFormat::Rgba8Unorm);
    let visibility = buffer(8, 64);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, true),
            Some((visibility, kind)),
            Vec::new(),
        )
        .unwrap();
    commands
        .set_render_pipeline_state(&render_state(1))
        .unwrap();
    commands
        .set_render_buffer(Stage::Vertex, Some(buffer(3, 256)), 0, 2)
        .unwrap();
    for offset in [0, 8, 0] {
        commands
            .set_visibility_result_mode(VisibilityResultMode::Counting, offset)
            .unwrap();
        commands
            .draw_primitives(&mut resolver, PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;
    assert_eq!(
        count(commands, |command| matches!(
            command,
            Command::BeginQuery { precise: true, .. }
        )),
        3
    );
    assert_eq!(
        count(commands, |command| matches!(
            command,
            Command::EndQuery { .. }
        )),
        3
    );
    assert_eq!(
        recording.query_pools,
        [
            (QueryType::Occlusion, QueryPool(1)),
            (QueryType::Occlusion, QueryPool(2))
        ],
        "a second pool once the first is full"
    );
    let Some(&Command::AccumulateResults { arguments, runs }) = commands.last() else {
        panic!("{:?}", commands.last())
    };
    assert_eq!(runs, 2);
    let (chunk, bytes) = recording
        .uploads
        .iter()
        .find(|(chunk, _)| (chunk.address..chunk.address + chunk.size).contains(&arguments))
        .unwrap();
    let encoded = &bytes[(arguments - chunk.address) as usize..];
    let run = |index: usize| {
        let at = AccumulateArguments::RUNS_OFFSET as usize + 16 * index;
        let word =
            |offset: usize| u32::from_le_bytes(encoded[at + offset..][..4].try_into().unwrap());
        let destination = u64::from_le_bytes(encoded[at + 8..][..8].try_into().unwrap());
        (word(0), word(4), destination)
    };
    let held = u32::from(accumulate);
    assert_eq!(run(0), (0, 2 + held, 8 << 32));
    assert_eq!(run(1), (2 + held, 1 + held, (8 << 32) + 8));
    assert_eq!(
        count(commands, |command| matches!(
            command,
            Command::CopyBuffer { .. }
        )),
        2 * held as usize
    );
    let end = commands
        .iter()
        .position(|command| *command == Command::EndRendering)
        .unwrap();
    let copy = commands
        .iter()
        .position(|command| matches!(command, Command::CopyQueryResults { .. }))
        .unwrap();
    assert!(end < copy, "results are copied after the pass");
}

#[test]
fn amplified_draws_repeat_per_view() {
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    commands
        .set_render_pipeline_state(&render_state(2))
        .unwrap();
    commands
        .set_vertex_amplification(&[(0, 0), (1, 3)])
        .unwrap();
    commands
        .set_render_buffer(Stage::Vertex, Some(buffer(3, 256)), 0, 2)
        .unwrap();
    commands
        .draw_primitives(&mut resolver, PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let views: Vec<_> = recording.segments[0]
        .commands
        .iter()
        .filter_map(|command| match command {
            Command::PushConstants(constants) => {
                Some((constants.amplification_id, constants.view_mapping))
            }
            _ => None,
        })
        .collect();
    assert_eq!(views[views.len() - 2..], [(0, 0), (1, 3 << 16 | 1)]);
    assert_eq!(
        count(&recording.segments[0].commands, |command| matches!(
            command,
            Command::Draw { .. }
        )),
        2
    );
}

#[test]
fn draws_are_validated_against_the_pipeline_and_pass() {
    let color = texture(PixelFormat::Rgba8Unorm);
    let other = texture(PixelFormat::Bgra8Unorm);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    assert!(
        commands
            .draw_primitives(&mut resolver, PrimitiveType::Triangle, 0, 3, 1, 0)
            .is_err(),
        "no encoder"
    );
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    assert_eq!(
        commands.begin_blit(),
        Err(Error::Invalid("another encoder is still encoding"))
    );
    assert_eq!(
        commands.draw_primitives(&mut resolver, PrimitiveType::Triangle, 0, 3, 1, 0),
        Err(Error::Invalid("no render pipeline state is set"))
    );
    commands
        .set_render_pipeline_state(&render_state(1))
        .unwrap();
    assert_eq!(
        commands.draw_primitives(&mut resolver, PrimitiveType::Triangle, 0, 3, 1, 0),
        Err(Error::Invalid(
            "a vertex buffer the pipeline reads is unbound"
        ))
    );
    commands
        .set_render_buffer(Stage::Vertex, Some(buffer(3, 256)), 0, 2)
        .unwrap();
    assert!(
        commands
            .draw_primitives(&mut resolver, PrimitiveType::Line, 0, 2, 1, 0)
            .is_err(),
        "outside the class"
    );
    commands.end_encoding(&mut resolver).unwrap();

    commands
        .begin_render(
            &mut resolver,
            pass(&other, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    assert_eq!(
        commands.set_render_pipeline_state(&render_state(1)),
        Err(Error::Invalid(
            "pipeline color formats differ from the pass"
        ))
    );
}

fn kernel_state(writes: bool) -> ComputePipelineState {
    let kernel = shader(Stage::Kernel, |interface| {
        let access = if writes {
            ShaderAccess::READ | ShaderAccess::WRITE
        } else {
            ShaderAccess::READ
        };
        interface.buffers = alloc::vec![BufferUse {
            index: 0,
            access,
            argument_buffer: false
        }];
    });
    let descriptor = ComputePipelineDescriptor {
        compute_function: &kernel.interface,
        threadgroup_size_is_multiple_of_thread_execution_width: false,
        max_total_threads_per_threadgroup: 0,
        stage_input: None,
    };
    let plan = compute_pipeline(&capabilities(), &descriptor).unwrap();
    ComputePipelineState {
        id: 1,
        plan,
        interface: kernel.interface,
        indirect: None,
    }
}

#[test]
fn stage_in_dispatches_pass_their_region() {
    let kernel = shader(Stage::Kernel, |interface| {
        interface.attributes = alloc::vec![AttributeUse {
            index: 0,
            numeric: NumericClass::Float,
            components: 4,
            patch_control_point: false
        }];
    });
    let layout = StageInputDescriptor {
        attributes: alloc::vec![Attribute {
            index: 0,
            format: VertexFormat::Float4,
            offset: 0,
            buffer_index: 1,
        }],
        layouts: alloc::vec![StageInLayout {
            buffer_index: 1,
            stride: 16,
            step_function: StageInStep::ThreadPositionInGridX,
            step_rate: 1,
        }],
        index_type: IndexType::UInt16,
        index_buffer_index: 0,
    };
    let descriptor = ComputePipelineDescriptor {
        compute_function: &kernel.interface,
        threadgroup_size_is_multiple_of_thread_execution_width: false,
        max_total_threads_per_threadgroup: 0,
        stage_input: Some(&layout),
    };
    let state = ComputePipelineState {
        id: 1,
        plan: compute_pipeline(&capabilities(), &descriptor).unwrap(),
        interface: kernel.interface,
        indirect: None,
    };
    let threads = Size {
        width: 32,
        height: 1,
        depth: 1,
    };
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands.begin_compute(DispatchType::Serial).unwrap();
    commands.set_compute_pipeline_state(&state).unwrap();
    assert_eq!(
        commands.dispatch_threadgroups(&mut resolver, [1, 1, 1], threads),
        Err(Error::Invalid("no stage-in region is set"))
    );
    commands.set_stage_in_region([4, 0, 0], [60, 1, 1]).unwrap();
    commands
        .dispatch_threadgroups(&mut resolver, [2, 1, 1], threads)
        .unwrap();
    commands
        .set_stage_in_region_indirect(buffer(9, 64), 8)
        .unwrap();
    commands
        .dispatch_threadgroups(&mut resolver, [2, 1, 1], threads)
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    assert_eq!(
        resolver.compiled[0].stage_in.as_ref(),
        Some(&layout),
        "the compiler specializes on the layout"
    );
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let pushed: Vec<PushConstants> = recording.segments[0]
        .commands
        .iter()
        .filter_map(|command| match command {
            Command::PushConstants(constants) => Some(*constants),
            _ => None,
        })
        .collect();
    let (chunk, bytes) = &recording.uploads[0];
    let region = &bytes[(pushed[0].arguments - chunk.address) as usize..][..24];
    let expected: Vec<u8> = [4u32, 0, 0, 60, 1, 1]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    assert_eq!(region, &expected[..]);
    assert_eq!(pushed[1].arguments, (9 << 32) + 8);
}

#[test]
fn dynamic_stage_in_strides_follow_the_region() {
    let kernel = shader(Stage::Kernel, |interface| {
        interface.attributes = alloc::vec![AttributeUse {
            index: 0,
            numeric: NumericClass::Float,
            components: 4,
            patch_control_point: false
        }];
    });
    let layout = StageInputDescriptor {
        attributes: alloc::vec![Attribute {
            index: 0,
            format: VertexFormat::Float4,
            offset: 0,
            buffer_index: 3,
        }],
        layouts: alloc::vec![StageInLayout {
            buffer_index: 3,
            stride: STRIDE_DYNAMIC,
            step_function: StageInStep::ThreadPositionInGridX,
            step_rate: 1,
        }],
        index_type: IndexType::UInt16,
        index_buffer_index: 0,
    };
    let descriptor = ComputePipelineDescriptor {
        compute_function: &kernel.interface,
        threadgroup_size_is_multiple_of_thread_execution_width: false,
        max_total_threads_per_threadgroup: 0,
        stage_input: Some(&layout),
    };
    let state = ComputePipelineState {
        id: 1,
        plan: compute_pipeline(&capabilities(), &descriptor).unwrap(),
        interface: kernel.interface,
        indirect: None,
    };
    let threads = Size {
        width: 32,
        height: 1,
        depth: 1,
    };
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands.begin_compute(DispatchType::Serial).unwrap();
    commands.set_compute_pipeline_state(&state).unwrap();
    commands.set_stage_in_region([0, 0, 0], [8, 1, 1]).unwrap();
    commands
        .set_compute_buffer(Some(buffer(5, 256)), 0, 3)
        .unwrap();
    assert_eq!(
        commands.dispatch_threadgroups(&mut resolver, [1, 1, 1], threads),
        Err(Error::Invalid(
            "a dynamic-stride stage-in buffer was bound without an attribute stride"
        ))
    );
    assert_eq!(
        commands.set_compute_buffer_offset_with_stride(0, 6, 3),
        Err(Error::Invalid("attribute strides are multiples of four"))
    );
    commands
        .set_compute_buffer_offset_with_stride(16, 24, 3)
        .unwrap();
    commands
        .dispatch_threadgroups(&mut resolver, [1, 1, 1], threads)
        .unwrap();
    commands
        .set_stage_in_region_indirect(buffer(9, 64), 8)
        .unwrap();
    assert_eq!(
        commands.dispatch_threadgroups(&mut resolver, [1, 1, 1], threads),
        Err(Error::Unsupported(
            "GPU stage-in regions of kernels whose stage-in strides are dynamic"
        ))
    );
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let arguments = recording.segments[0]
        .commands
        .iter()
        .find_map(|command| match command {
            Command::PushConstants(constants) => Some(constants.arguments),
            _ => None,
        })
        .unwrap();
    let (chunk, bytes) = &recording.uploads[0];
    let block = &bytes[(arguments - chunk.address) as usize..][..DispatchArguments::BYTES as usize];
    assert_eq!(
        &block[..24],
        &[0u32, 0, 0, 8, 1, 1]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>()[..]
    );
    let strides = DispatchArguments::STRIDES_OFFSET as usize;
    assert_eq!(&block[strides + 12..strides + 16], &24u32.to_le_bytes());
    assert!(
        block[strides..strides + 12].iter().all(|&byte| byte == 0),
        "unbound strides are zero"
    );
}

fn barriers(recording: &Recording) -> usize {
    count(&recording.segments[0].commands[1..], |command| {
        matches!(command, Command::Barrier(_))
    })
}

#[test]
fn every_encoder_pushes_the_command_buffer_s_log_buffer() {
    let threads = Size {
        width: 64,
        height: 1,
        depth: 1,
    };
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    let log = buffer(4, 1024);
    commands.set_log_buffer(log);
    for _ in 0..2 {
        commands.begin_compute(DispatchType::Serial).unwrap();
        commands
            .set_compute_pipeline_state(&kernel_state(false))
            .unwrap();
        commands
            .dispatch_threadgroups(&mut resolver, [1, 1, 1], threads)
            .unwrap();
        commands.end_encoding(&mut resolver).unwrap();
    }
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let logs: Vec<_> = recording.segments[0]
        .commands
        .iter()
        .filter_map(|command| match command {
            Command::PushConstants(constants) => Some(constants.log),
            _ => None,
        })
        .collect();
    assert_eq!(logs, [log.address; 2]);
}

#[test]
fn serial_dispatches_order_writes_and_concurrent_ones_do_not() {
    let threads = Size {
        width: 64,
        height: 1,
        depth: 1,
    };
    let run = |dispatch_type, writes| {
        let mut resolver = MockResolver::default();
        let mut commands = CommandBuffer::new(FEATURES);
        commands.begin_compute(dispatch_type).unwrap();
        commands
            .set_compute_pipeline_state(&kernel_state(writes))
            .unwrap();
        commands
            .set_compute_buffer(Some(buffer(4, 1024)), 0, 0)
            .unwrap();
        commands
            .dispatch_threadgroups(&mut resolver, [4, 1, 1], threads)
            .unwrap();
        commands
            .dispatch_threads(&mut resolver, [100, 1, 1], threads)
            .unwrap();
        commands.end_encoding(&mut resolver).unwrap();
        (commit(commands, &mut resolver, (Semaphore(1), 1)), resolver)
    };
    let (serial, resolver) = run(DispatchType::Serial, true);
    assert_eq!(barriers(&serial), 1);
    assert_eq!(
        resolver.compiled.len(),
        1,
        "both dispatches share a variant"
    );
    assert_eq!(resolver.compiled[0].workgroup_size, Some([64, 1, 1]));
    let dispatches: Vec<_> = serial.segments[0]
        .commands
        .iter()
        .filter_map(|command| match command {
            Command::Dispatch(groups) => Some(*groups),
            _ => None,
        })
        .collect();
    assert_eq!(dispatches, [[4, 1, 1], [2, 1, 1]]);
    let grid = serial.segments[0]
        .commands
        .iter()
        .rev()
        .find_map(|command| match command {
            Command::PushConstants(constants) => Some(constants.grid),
            _ => None,
        });
    assert_eq!(grid, Some([100, 1, 1]));

    assert_eq!(
        barriers(&run(DispatchType::Serial, false).0),
        0,
        "read-only dispatches never wait"
    );
    assert_eq!(barriers(&run(DispatchType::Concurrent, true).0), 0);
}

#[test]
fn encoders_after_writes_wait_for_them() {
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands.begin_compute(DispatchType::Serial).unwrap();
    commands
        .set_compute_pipeline_state(&kernel_state(true))
        .unwrap();
    commands
        .dispatch_threadgroups(
            &mut resolver,
            [1, 1, 1],
            Size {
                width: 32,
                height: 1,
                depth: 1,
            },
        )
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    commands.begin_blit().unwrap();
    commands
        .copy_buffer(buffer(4, 64), 0, buffer(5, 64), 0, 64)
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let barrier = recording.segments[0]
        .commands
        .iter()
        .skip(1)
        .find_map(|command| match command {
            Command::Barrier(barrier) => Some(*barrier),
            _ => None,
        });
    assert_eq!(
        barrier,
        Some(Barrier {
            src_stages: PipelineStages::COMPUTE_SHADER,
            src_access: Access::SHADER_WRITE,
            dst_stages: PipelineStages::ALL_TRANSFER,
            dst_access: Access::TRANSFER_READ | Access::TRANSFER_WRITE,
        })
    );
}

#[test]
fn unaligned_fills_copy_their_edges() {
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands.begin_blit().unwrap();
    commands
        .fill_buffer(&mut resolver, buffer(6, 64), 3, 10, 0xab)
        .unwrap();
    commands
        .fill_buffer(&mut resolver, buffer(6, 64), 41, 2, 0xcd)
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;
    assert!(commands.contains(&Command::FillBuffer {
        buffer: Buffer(6),
        offset: 4,
        size: 8,
        data: 0xabab_abab
    }));
    let copies: Vec<_> = commands
        .iter()
        .filter_map(|command| match command {
            Command::CopyBuffer {
                dst: Buffer(6),
                region,
                ..
            } => Some((region.dst_offset, region.size)),
            _ => None,
        })
        .collect();
    assert_eq!(copies, [(3, 1), (12, 1), (41, 2)]);
}

#[test]
fn buffer_texture_copies_use_texel_strides() {
    let color = texture(PixelFormat::Rgba16Float);
    let depth = texture(PixelFormat::Depth32FloatStencil8);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands.begin_blit().unwrap();
    let source = BufferLayout {
        buffer: buffer(2, 1 << 16),
        offset: 256,
        bytes_per_row: 512,
        bytes_per_image: 0,
    };
    let region = Region {
        slice: 0,
        level: 1,
        origin: [8, 0, 0],
        size: [16, 16, 1],
    };
    let target = BlitTexture {
        image: Image(9),
        descriptor: &color,
        storage: ResourceId::NULL,
        blocks: None,
    };
    commands
        .copy_buffer_to_texture(&mut resolver, source, target, region, BlitOption::EMPTY)
        .unwrap();
    let outside = Region {
        level: 1,
        origin: [24, 0, 0],
        ..region
    };
    assert!(
        commands
            .copy_buffer_to_texture(&mut resolver, source, target, outside, BlitOption::EMPTY)
            .is_err()
    );
    let depth_target = BlitTexture {
        image: Image(10),
        descriptor: &depth,
        storage: ResourceId::NULL,
        blocks: None,
    };
    assert_eq!(
        commands.copy_buffer_to_texture(
            &mut resolver,
            source,
            depth_target,
            region,
            BlitOption::EMPTY
        ),
        Err(Error::Invalid("depth-stencil copies name one aspect"))
    );
    commands
        .copy_buffer_to_texture(
            &mut resolver,
            source,
            depth_target,
            region,
            BlitOption::STENCIL_FROM_DEPTH_STENCIL,
        )
        .unwrap();
    let destination = BufferLayout {
        buffer: buffer(3, 1 << 16),
        offset: 1024,
        ..source
    };
    commands
        .copy_texture_to_buffer(
            &mut resolver,
            target,
            region,
            destination,
            BlitOption::EMPTY,
        )
        .unwrap();
    commands.generate_mipmaps(target, true).unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let copies: Vec<_> = recording.segments[0]
        .commands
        .iter()
        .filter_map(|command| match command {
            Command::CopyBufferToImage { region, .. } => Some(*region),
            _ => None,
        })
        .collect();
    assert_eq!(
        (
            copies[0].buffer_offset,
            copies[0].buffer_row_length,
            copies[0].buffer_image_height
        ),
        (256, 64, 16)
    );
    assert_eq!(
        (copies[1].subresource.aspects, copies[1].buffer_row_length),
        (crate::metal::vulkan::ImageAspects::STENCIL, 512)
    );
    let (src, dst, readback) = recording.segments[0]
        .commands
        .iter()
        .find_map(|command| match command {
            Command::CopyImageToBuffer { src, dst, region } => Some((*src, *dst, *region)),
            _ => None,
        })
        .unwrap();
    assert_eq!((src, dst), (Image(9), Buffer(3)));
    assert_eq!(
        readback,
        BufferImageCopy {
            buffer_offset: 1024,
            ..copies[0]
        }
    );
    let blits = count(&recording.segments[0].commands, |command| {
        matches!(command, Command::BlitImage { .. })
    });
    assert_eq!(blits, 2, "levels 1 and 2 are generated");
    let _ = BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: 0,
    };
}

#[test]
fn events_split_command_buffers_into_submissions() {
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .encode_wait_for_event(&mut resolver, Semaphore(7), 3)
        .unwrap();
    commands.begin_blit().unwrap();
    commands
        .copy_buffer(buffer(4, 64), 0, buffer(5, 64), 0, 64)
        .unwrap();
    assert!(
        commands
            .encode_signal_event(&mut resolver, Semaphore(8), 1)
            .is_err(),
        "events wait for the encoder to end"
    );
    commands.end_encoding(&mut resolver).unwrap();
    commands
        .encode_signal_event(&mut resolver, Semaphore(8), 1)
        .unwrap();
    commands
        .encode_wait_for_event(&mut resolver, Semaphore(7), 4)
        .unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 9));
    let shape: Vec<_> = recording
        .segments
        .iter()
        .map(|segment| {
            (
                segment.waits.clone(),
                segment.commands.len(),
                segment.signals.clone(),
            )
        })
        .collect();
    assert_eq!(
        shape,
        [
            (
                alloc::vec![(Semaphore(7), 3)],
                2,
                alloc::vec![(Semaphore(8), 1)]
            ),
            (
                alloc::vec![(Semaphore(7), 4)],
                1,
                alloc::vec![(Semaphore(1), 9)]
            ),
        ]
    );
}

#[test]
fn waits_for_values_the_command_buffer_signals_take_no_submission() {
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .encode_signal_event(&mut resolver, Semaphore(8), 2)
        .unwrap();
    for (event, value) in [(Semaphore(8), 1), (Semaphore(8), 2)] {
        commands
            .encode_wait_for_event(&mut resolver, event, value)
            .unwrap();
    }
    commands
        .encode_wait_for_event(&mut resolver, Semaphore(8), 3)
        .unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 9));
    let waits: Vec<_> = recording
        .segments
        .iter()
        .map(|segment| segment.waits.clone())
        .collect();
    assert_eq!(
        waits,
        [alloc::vec![], alloc::vec![(Semaphore(8), 3)]],
        "only a value the command buffer does not reach waits"
    );
    assert!(
        recording
            .segments
            .iter()
            .all(|segment| segment.commands.first() == Some(&Command::Barrier(FULL))),
        "every segment opens with a full barrier"
    );
}

#[test]
fn signals_without_a_later_wait_take_no_submission() {
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    for value in [1, 2] {
        commands.begin_blit().unwrap();
        commands
            .copy_buffer(buffer(4, 64), 0, buffer(5, 64), 0, 64)
            .unwrap();
        commands.end_encoding(&mut resolver).unwrap();
        commands
            .encode_signal_event(&mut resolver, Semaphore(8), value)
            .unwrap();
    }
    let recording = commit(commands, &mut resolver, (Semaphore(1), 9));
    let [segment] = recording.segments.as_slice() else {
        panic!("one submission, not {}", recording.segments.len());
    };
    assert_eq!(
        segment.signals,
        [(Semaphore(8), 2), (Semaphore(1), 9)],
        "the event keeps its last value, signaled with completion"
    );
    let barriers = count(&segment.commands, |command| {
        *command == Command::Barrier(FULL)
    });
    assert_eq!(barriers, 3, "each joined segment keeps its opening barrier");
}

#[test]
fn bindings_validate_slots_and_offsets() {
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands.begin_compute(DispatchType::Serial).unwrap();
    assert!(
        commands
            .set_compute_buffer(Some(buffer(1, 16)), 32, 0)
            .is_err()
    );
    assert!(
        commands
            .set_compute_buffer(Some(buffer(1, 16)), 0, 31)
            .is_err()
    );
    assert!(
        commands.set_compute_buffer_offset(4, 3).is_err(),
        "nothing bound"
    );
    commands
        .set_compute_buffer(Some(buffer(1, 16)), 0, 3)
        .unwrap();
    commands.set_compute_buffer_offset(4, 3).unwrap();
    assert!(commands.set_compute_texture(ResourceId::NULL, 128).is_err());
    assert!(
        commands
            .set_compute_bytes(&mut resolver, &[0; 4097], 0)
            .is_err()
    );
    commands
        .set_compute_sampler(&mut resolver, ResourceId(5), Some((0.0, 2.0)), 0)
        .unwrap();
    assert!(
        commands
            .set_compute_sampler(&mut resolver, ResourceId(5), None, 16)
            .is_err()
    );
}

/// A pass that loads an IOSurface's extended-range planes decodes the
/// bounds of what its draws may have reached, and encodes each disjoint
/// part of it: their scissors, within their viewports for triangles.
#[test]
fn surface_plane_conversions_cover_what_the_pass_drew() {
    let color = texture(PixelFormat::Rgba8Unorm);
    let planes = crate::metal::pass::ExtendedPlanes {
        storage: ResourceId(0x55),
        encoding: 9,
        extent: [64, 32],
        layers: 1,
        color: 1 << 40,
        color_bytes_per_row: 256,
        alpha: 2 << 40,
        alpha_bytes_per_row: 64,
        loads: true,
        stores: true,
        narrows: true,
        decodes_whole: false,
        mirror: 0,
        stamp: 0,
    };
    let record = |draws: &[(ScissorRect, Option<Viewport>, PrimitiveType)]| {
        let mut resolver = MockResolver::default();
        let mut commands = CommandBuffer::new(FEATURES);
        let mut plan = pass(&color, StoreAction::Store, false);
        plan.inputs.extended_planes[0] = Some(planes);
        commands
            .begin_render(&mut resolver, plan, None, Vec::new())
            .unwrap();
        // Of any topology.
        let mut state = render_state(1);
        state.plan.topology_class = crate::metal::abi::PrimitiveTopologyClass::Unspecified;
        commands.set_render_pipeline_state(&state).unwrap();
        commands
            .set_render_buffer(Stage::Vertex, Some(buffer(3, 256)), 0, 2)
            .unwrap();
        for &(scissor, viewport, primitive) in draws {
            commands.set_scissor_rects(&[scissor]).unwrap();
            if let Some(viewport) = viewport {
                commands.set_viewports(&[viewport]).unwrap();
            }
            commands
                .draw_primitives(&mut resolver, primitive, 0, 3, 1, 0)
                .unwrap();
        }
        commands.end_encoding(&mut resolver).unwrap();
        let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
        recording.segments[0]
            .commands
            .iter()
            .filter_map(|command| match *command {
                Command::StoreTexels { arguments, count }
                | Command::LoadTexels { arguments, count } => {
                    let encoded = uploaded(&recording, arguments);
                    Some((
                        count,
                        double(encoded, 0),
                        [24, 28, 40, 44].map(|at| word(encoded, at)),
                        double(encoded, 56),
                    ))
                }
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let rect = |x, y, width, height| ScissorRect {
        x,
        y,
        width,
        height,
    };
    // The conversion of [x, y, width, height].
    let converted = |[x, y, width, height]: [u32; 4]| {
        (
            width * height,
            (1 << 40) + u64::from(y) * 256 + u64::from(x) * 4,
            [width, height, x, y],
            (2 << 40) + u64::from(y) * 64 + u64::from(x),
        )
    };
    let triangle = PrimitiveType::Triangle;
    // Two draws apart: the decode of their bounds, 8..40 by 4..20, and an
    // encode of each.
    let drawn = record(&[
        (rect(8, 4, 16, 8), None, triangle),
        (rect(24, 12, 16, 8), None, triangle),
    ]);
    assert_eq!(
        drawn,
        [
            converted([8, 4, 32, 16]),
            converted([8, 4, 16, 8]),
            converted([24, 12, 16, 8]),
        ]
    );
    // Overlapping draws encode their union once.
    let overlapping = record(&[
        (rect(8, 4, 16, 8), None, triangle),
        (rect(16, 8, 16, 8), None, triangle),
    ]);
    assert_eq!(
        overlapping,
        [converted([8, 4, 24, 12]), converted([8, 4, 24, 12])]
    );
    // A whole scissor bounded by a viewport of fractional edges: the pixels
    // it touches.
    let viewport = Viewport {
        origin_x: 10.5,
        origin_y: 3.25,
        width: 20.0,
        height: 6.5,
        znear: 0.0,
        zfar: 1.0,
    };
    let whole = rect(0, 0, 64, 32);
    let bounded = record(&[(whole, Some(viewport), triangle)]);
    assert_eq!(
        bounded,
        [converted([10, 3, 21, 7]), converted([10, 3, 21, 7])]
    );
    // Points reach past the viewport by their size: the scissor bounds them.
    let points = record(&[(whole, Some(viewport), PrimitiveType::Point)]);
    assert_eq!(
        points,
        [converted([0, 0, 64, 32]), converted([0, 0, 64, 32])]
    );
    // Nothing drawn: an empty decode, and nothing to encode.
    let none = record(&[]);
    assert_eq!(none.len(), 1);
    assert_eq!(none[0].0, 0);
}

/// A pass whose fragments mark an IOSurface working image's dirty tiles
/// (its attachment's raw texels are the working image) encodes, when only
/// draws reached all of it, a workgroup per tile with its stamp; a scissor
/// narrower than the target keeps its rectangle, and a pass whose attachment
/// is not the working image marks nothing.
#[test]
fn marked_passes_encode_their_dirty_tiles() {
    let mut color = texture(PixelFormat::Rgba8Unorm);
    color.width = 200;
    color.height = 70;
    let planes = crate::metal::pass::ExtendedPlanes {
        storage: ResourceId(0x55),
        encoding: 9,
        extent: [200, 70],
        layers: 1,
        color: 1 << 40,
        color_bytes_per_row: 800,
        alpha: 2 << 40,
        alpha_bytes_per_row: 200,
        loads: true,
        stores: true,
        narrows: true,
        decodes_whole: true,
        mirror: crate::metal::binding::TexelArguments::MIRROR_COMPARE,
        stamp: 7,
    };
    let record = |scissor: ScissorRect, attached: ResourceId| {
        let mut resolver = MockResolver::default();
        let mut commands = CommandBuffer::new(FEATURES);
        let mut plan = pass(&color, StoreAction::Store, false);
        plan.inputs.extended_planes[0] = Some(planes);
        plan.inputs.textures[0] = attached;
        commands
            .begin_render(&mut resolver, plan, None, Vec::new())
            .unwrap();
        let mut state = render_state(1);
        state.plan.topology_class = crate::metal::abi::PrimitiveTopologyClass::Unspecified;
        commands.set_render_pipeline_state(&state).unwrap();
        commands
            .set_render_buffer(Stage::Vertex, Some(buffer(3, 256)), 0, 2)
            .unwrap();
        commands.set_scissor_rects(&[scissor]).unwrap();
        commands
            .draw_primitives(&mut resolver, PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        commands.end_encoding(&mut resolver).unwrap();
        let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
        recording.segments[0]
            .commands
            .iter()
            .filter_map(|command| match *command {
                Command::LoadTexels { arguments, count } => {
                    let encoded = uploaded(&recording, arguments);
                    Some((
                        count,
                        word(encoded, 52),
                        [24, 28].map(|at| word(encoded, at)),
                    ))
                }
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let whole = ScissorRect {
        x: 0,
        y: 0,
        width: 200,
        height: 70,
    };
    // 4 x 2 tiles of 64, a workgroup of 64 threads each.
    assert_eq!(
        record(whole, ResourceId(0x55)),
        [(4 * 2 * 64, 7, [200, 70])]
    );
    let part = ScissorRect {
        x: 8,
        y: 4,
        width: 16,
        height: 8,
    };
    assert_eq!(record(part, ResourceId(0x55)), [(16 * 8, 0, [16, 8])]);
    assert_eq!(record(whole, ResourceId(0x66)), [(200 * 70, 0, [200, 70])]);
}

#[test]
fn extended_range_buffer_copies_run_the_texel_kernels() {
    let color = texture(PixelFormat::Bgra10XrSrgb);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands.begin_blit().unwrap();
    let layout = BufferLayout {
        buffer: buffer(2, 1 << 16),
        offset: 256,
        bytes_per_row: 512,
        bytes_per_image: 0,
    };
    let region = Region {
        slice: 0,
        level: 1,
        origin: [8, 4, 0],
        size: [16, 8, 1],
    };
    let target = BlitTexture {
        image: Image(9),
        descriptor: &color,
        storage: ResourceId(0x77),
        blocks: None,
    };
    commands
        .copy_buffer_to_texture(&mut resolver, layout, target, region, BlitOption::EMPTY)
        .unwrap();
    commands
        .copy_texture_to_buffer(&mut resolver, target, region, layout, BlitOption::EMPTY)
        .unwrap();
    let unviewed = BlitTexture {
        storage: ResourceId::NULL,
        ..target
    };
    assert_eq!(
        commands.copy_buffer_to_texture(&mut resolver, layout, unviewed, region, BlitOption::EMPTY),
        Err(Error::Unsupported(
            "buffer copies of 1D and 3D textures of a stored encoding"
        ))
    );
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;
    assert_eq!(
        count(commands, |command| matches!(
            command,
            Command::CopyBufferToImage { .. } | Command::CopyImageToBuffer { .. }
        )),
        0,
        "the image holds half floats, not the buffer's codes"
    );
    let kernels: Vec<_> = commands
        .iter()
        .filter_map(|command| match *command {
            Command::StoreTexels { arguments, count } => Some((true, arguments, count)),
            Command::LoadTexels { arguments, count } => Some((false, arguments, count)),
            _ => None,
        })
        .collect();
    assert_eq!(
        kernels
            .iter()
            .map(|&(store, _, count)| (store, count))
            .collect::<Vec<_>>(),
        [(true, 128), (false, 128)]
    );
    for &(_, arguments, _) in &kernels {
        let encoded = uploaded(&recording, arguments);
        assert_eq!(
            double(encoded, 0),
            (2 << 32) + 256,
            "buffer address and offset"
        );
        assert_eq!(word(encoded, 12), 4, "sRGB BGRA10_XR");
        assert_eq!(double(encoded, 16), 0x77);
        assert_eq!(
            [24, 28, 32, 40, 44, 48].map(|at| word(encoded, at)),
            [16, 8, 512, 8, 4, 0]
        );
    }
}

#[test]
fn decoded_buffer_copies_move_blocks_and_decode_them() {
    let etc = texture(PixelFormat::Etc2Rgb8);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands.begin_blit().unwrap();
    let layout = BufferLayout {
        buffer: buffer(2, 1 << 16),
        offset: 256,
        bytes_per_row: 512,
        bytes_per_image: 0,
    };
    // Level 1's 8x4 blocks of 8 bytes follow level 0's 1024; the region is
    // 4x2 blocks from block (2, 1).
    let region = Region {
        slice: 0,
        level: 1,
        origin: [8, 4, 0],
        size: [16, 8, 1],
    };
    let target = BlitTexture {
        image: Image(9),
        descriptor: &etc,
        storage: ResourceId(0x77),
        blocks: Some(buffer(3, etc.cpu_copy_bytes())),
    };
    commands
        .copy_buffer_to_texture(&mut resolver, layout, target, region, BlitOption::EMPTY)
        .unwrap();
    commands
        .copy_texture_to_buffer(&mut resolver, target, region, layout, BlitOption::EMPTY)
        .unwrap();
    let view = BlitTexture {
        blocks: None,
        ..target
    };
    assert_eq!(
        commands.copy_buffer_to_texture(&mut resolver, layout, view, region, BlitOption::EMPTY),
        Err(Error::Unsupported(
            "copies through views of compressed textures the device samples decoded"
        ))
    );
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;
    let rows = |src, dst, [a, b]: [u64; 2], [c, d]: [u64; 2]| Command::CopyBufferRegions {
        src: Buffer(src),
        dst: Buffer(dst),
        regions: alloc::vec![
            BufferCopy {
                src_offset: a,
                dst_offset: c,
                size: 32,
            },
            BufferCopy {
                src_offset: b,
                dst_offset: d,
                size: 32,
            },
        ],
    };
    let copies: Vec<_> = commands
        .iter()
        .filter(|command| matches!(command, Command::CopyBufferRegions { .. }))
        .cloned()
        .collect();
    assert_eq!(
        copies,
        [
            rows(2, 3, [256, 768], [1104, 1168]),
            rows(3, 2, [1104, 1168], [256, 768]),
        ]
    );
    let decodes: Vec<_> = commands
        .iter()
        .filter_map(|command| match *command {
            Command::DecodeBlocks {
                kernel,
                arguments,
                count,
            } => Some((kernel, arguments, count)),
            _ => None,
        })
        .collect();
    let [(kernel, arguments, 128)] = decodes[..] else {
        panic!("one decode of the region's texels: {decodes:?}");
    };
    assert_eq!(kernel, BlockKernel::Etc);
    let encoded = uploaded(&recording, arguments);
    assert_eq!(
        double(encoded, 0),
        (3 << 32) + 1104,
        "the region's first block"
    );
    assert_eq!(
        word(encoded, 12),
        1 | 4 << 8 | 4 << 16,
        "ETC2 RGB8 in 4x4 blocks"
    );
    assert_eq!(double(encoded, 16), 0x77);
    assert_eq!(
        [24, 28, 32, 36, 40, 44, 48].map(|at| word(encoded, at)),
        [16, 8, 64, 0, 8, 4, 0]
    );
    assert_eq!(
        count(commands, |command| matches!(
            command,
            Command::CopyBufferToImage { .. } | Command::CopyImageToBuffer { .. }
        )),
        0,
        "the image holds texels, not the buffer's blocks"
    );
}

#[test]
fn pvrtc_copies_reorder_blocks_and_decode_whole_levels() {
    let pvrtc = TextureDescriptor {
        width: 32,
        height: 32,
        mipmap_level_count: 1,
        ..texture(PixelFormat::PvrtcRgba4bpp)
    };
    // CPU transfers copy as many blocks as a region covers from the level's
    // first, whatever its origin and strides: 2x2 blocks are the start of
    // the first row of 8, and 8x2 its first two rows.
    let at = |origin: [u64; 2], size: [u64; 2]| Region {
        slice: 0,
        level: 0,
        origin: [origin[0], origin[1], 0],
        size: [size[0], size[1], 1],
    };
    assert_eq!(
        at([8, 8], [8, 8]).block_run(&pvrtc),
        Ok(at([0, 0], [16, 4]))
    );
    assert_eq!(
        at([0, 16], [32, 8]).block_run(&pvrtc),
        Ok(at([0, 0], [32, 8]))
    );
    assert_eq!(
        at([0, 0], [16, 12]).block_run(&pvrtc),
        Err(Error::Unsupported(
            "PVRTC regions whose blocks end within a later row of their level"
        ))
    );
    // Blits copy a region's rows of blocks, which the device reorders into
    // and out of the level's Morton order.
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands.begin_blit().unwrap();
    let layout = BufferLayout {
        buffer: buffer(2, 1 << 16),
        offset: 64,
        bytes_per_row: 16,
        bytes_per_image: 32,
    };
    let target = || BlitTexture {
        image: Image(9),
        descriptor: &pvrtc,
        storage: ResourceId(0x77),
        blocks: Some(buffer(3, pvrtc.cpu_copy_bytes())),
    };
    let region = at([8, 8], [8, 8]);
    commands
        .copy_buffer_to_texture(&mut resolver, layout, target(), region, BlitOption::EMPTY)
        .unwrap();
    commands
        .copy_texture_to_buffer(&mut resolver, target(), region, layout, BlitOption::EMPTY)
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;
    assert!(
        !commands
            .iter()
            .any(|command| matches!(command, Command::CopyBufferRegions { .. })),
        "{commands:?}"
    );
    let reorders = commands
        .iter()
        .filter_map(|command| match *command {
            Command::ReorderBlocks { arguments, count } => Some((arguments, count)),
            _ => None,
        })
        .collect::<alloc::vec::Vec<_>>();
    assert_eq!(reorders.len(), 2, "{commands:?}");
    for (&(arguments, count), to_rows) in reorders.iter().zip([0, 1]) {
        assert_eq!(count, 4, "2x2 blocks");
        let encoded = uploaded(&recording, arguments);
        assert_eq!(double(encoded, 0), (2 << 32) + 64, "the buffer's rows");
        assert_eq!(double(encoded, 8), 3 << 32, "the level's blocks");
        assert_eq!(
            [16, 20, 24, 28, 32, 36, 40, 44].map(|at| word(encoded, at)),
            [4, to_rows, 8, 8, 2, 2, 2, 16]
        );
    }
    let Some(&Command::DecodeBlocks {
        kernel: BlockKernel::Pvrtc,
        arguments,
        count: 1024,
    }) = commands
        .iter()
        .find(|command| matches!(command, Command::DecodeBlocks { .. }))
    else {
        panic!("a decode of the whole level: {commands:?}");
    };
    let encoded = uploaded(&recording, arguments);
    assert_eq!(double(encoded, 0), 3 << 32, "the level's first block");
    assert_eq!(word(encoded, 12), 8 | 4 << 8 | 4 << 16, "4-bpp PVRTC");
    assert_eq!(
        [24, 28, 40, 44, 48].map(|at| word(encoded, at)),
        [32, 32, 0, 0, 0]
    );
}

#[test]
fn raw_rendered_attachments_resolve_where_the_pass_ends() {
    let mut ms = texture(PixelFormat::Abgr4Unorm);
    ms.texture_type = TextureType::D2Multisample;
    ms.sample_count = 4;
    ms.mipmap_level_count = 1;
    let mut resolved = texture(PixelFormat::Abgr4Unorm);
    resolved.mipmap_level_count = 1;
    let target = |texture: ImageView, descriptor| Target {
        texture: AttachedTexture {
            texture,
            descriptor,
        },
        level: 0,
        slice: 0,
        depth_plane: 0,
    };
    let mut descriptor = RenderPassDescriptor {
        color_attachments: [None; MAX_COLOR_ATTACHMENTS],
        depth_attachment: None,
        depth_resolve_filter: crate::metal::abi::MultisampleDepthResolveFilter::Sample0,
        stencil_attachment: None,
        stencil_resolve_filter: crate::metal::abi::MultisampleStencilResolveFilter::Sample0,
        visibility_results: false,
        render_target_array_length: 0,
        render_target_width: 0,
        render_target_height: 0,
        default_raster_sample_count: 0,
        tile_memory: Default::default(),
        sample_positions: Vec::new(),
        rasterization_rate_map: false,
    };
    descriptor.color_attachments[0] = Some(AttachmentDescriptor {
        target: target(ImageView(7), &ms),
        resolve: Some(target(ImageView(8), &resolved)),
        load_action: LoadAction::Clear,
        store_action: StoreAction::MultisampleResolve,
        store_action_options: StoreActionOptions::EMPTY,
        clear: [1.0, 0.5, 0.0, 0.2],
    });
    let mut plan = render_pass(&capabilities(), &descriptor).unwrap();
    let color = plan.color[0].as_ref().unwrap();
    assert!(color.averaged);
    // R4G4B4A4: red 15, green 8, blue 0 and alpha 3 from the top bits.
    assert_eq!(
        color.clear,
        crate::metal::pass::ClearValue::Uint([0xf803, 0, 0, 0])
    );
    plan.inputs.textures[0] = ResourceId(0x55);
    plan.inputs.averages[0] = ResourceId(0x66);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(&mut resolver, plan, None, Vec::new())
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;
    let Some(Command::BeginRendering(rendering)) = commands
        .iter()
        .find(|command| matches!(command, Command::BeginRendering(_)))
    else {
        panic!("no rendering")
    };
    assert_eq!(
        rendering.color[0].as_ref().unwrap().resolve,
        None,
        "Vulkan averages no raw texels"
    );
    let Some(&Command::AverageSamples { arguments, count }) = commands
        .iter()
        .find(|command| matches!(command, Command::AverageSamples { .. }))
    else {
        panic!("no average")
    };
    assert_eq!(count, 64 * 32);
    let encoded = uploaded(&recording, arguments);
    assert_eq!((double(encoded, 0), double(encoded, 8)), (0x55, 0x66));
    assert_eq!(
        [16, 20, 24, 28, 32].map(|at| word(encoded, at)),
        [64 * 32, 2, 64, 32, 4]
    );
}

/// The bytes uploaded at `address`.
fn uploaded(recording: &Recording, address: u64) -> &[u8] {
    let (chunk, bytes) = recording
        .uploads
        .iter()
        .find(|(chunk, _)| (chunk.address..chunk.address + chunk.size).contains(&address))
        .expect("an upload holds the address");
    &bytes[(address - chunk.address) as usize..]
}

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn double(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

#[test]
fn statistic_samples_sum_intervals_no_query_spans() {
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    let first = 9 << 32;
    commands
        .sample_counters(&mut resolver, CounterSample::Statistic(first), false)
        .unwrap();
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    commands
        .set_render_pipeline_state(&patch_state(None))
        .unwrap();
    commands
        .set_tessellation_factor_buffer(Some(buffer(4, 1024)), 96, 48)
        .unwrap();
    let source = PatchSource {
        control_points: 4,
        patch_indices: None,
        control_point_indices: Some((buffer(5, 4096), 64)),
    };
    commands
        .draw_patches(&mut resolver, &source, 0, 1, 1, 0)
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    commands
        .sample_counters(&mut resolver, CounterSample::Statistic(first + 64), false)
        .unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;

    let mut active = false;
    for command in commands {
        match command {
            Command::BeginQuery { .. } => active = true,
            Command::EndQuery { .. } => active = false,
            Command::BeginRendering(_) | Command::EndRendering => {
                assert!(!active, "no statistics query spans a pass boundary");
            }
            _ => {}
        }
    }
    assert!(!active, "commit ends the last interval");
    // Before the pass, the pass up to the patch draw, the patch draw, the
    // rest of the pass, after the pass, and after the second sample.
    assert_eq!(
        count(commands, |c| matches!(c, Command::BeginQuery { .. })),
        6
    );
    assert!(
        recording
            .query_pools
            .iter()
            .all(|&(query_type, _)| query_type == QueryType::PipelineStatistics)
    );
    let copies: Vec<_> = commands
        .iter()
        .filter_map(|command| match command {
            &Command::CopyQueryResults {
                first,
                count,
                stride,
                wait,
                ..
            } => Some((first, count, stride, wait)),
            _ => None,
        })
        .collect();
    // Two queries per mock pool: the five ended intervals copy in three runs.
    assert_eq!(
        copies,
        [(0, 2, 56, true), (0, 2, 56, true), (0, 1, 56, true)]
    );
    let cleared: Vec<_> = commands
        .iter()
        .filter_map(|command| match command {
            &Command::FillBuffer { size, data, .. } => Some((size, data)),
            _ => None,
        })
        .collect();
    assert_eq!(cleared, [(8, 0)], "the patch draw's vertex invocations");

    let Some(&Command::AccumulateResults { arguments, runs }) = commands.last() else {
        panic!("{:?}", commands.last())
    };
    assert_eq!(runs, 14, "seven counted members per sample");
    let encoded = uploaded(&recording, arguments);
    assert_eq!(word(encoded, 12), 7, "one interval's results apart");
    let run = |index: usize| {
        let at = AccumulateArguments::RUNS_OFFSET as usize + 16 * index;
        (
            word(encoded, at),
            word(encoded, at + 4),
            double(encoded, at + 8),
        )
    };
    assert_eq!(
        run(0),
        (4, 0, first),
        "the first sample precedes every interval"
    );
    assert_eq!(run(7), (4, 5, first + 64), "tessellation input patches");
    assert_eq!(run(8), (0, 5, first + 72), "vertex invocations");
    assert_eq!(run(13), (6, 5, first + 120), "compute kernel invocations");
    assert!(
        (0..14).all(|index| run(index).2 != first + 64 + 48),
        "fragments passed keeps the error value"
    );
}

#[test]
fn render_pass_timestamps_reset_before_the_pass() {
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    let pool = QueryPool(30);
    let sample = |query| CounterSample::Timestamp { pool, query };
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            alloc::vec![
                (sample(0), PassBoundary::StartOfVertex),
                (sample(1), PassBoundary::EndOfFragment),
            ],
        )
        .unwrap();
    commands
        .sample_counters(&mut resolver, sample(2), true)
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;
    let begin = commands
        .iter()
        .position(|command| matches!(command, Command::BeginRendering(_)))
        .unwrap();
    for query in 0..3 {
        let reset = commands.iter().position(|command| {
            *command
                == Command::ResetQueryPool {
                    pool,
                    first: query,
                    count: 1,
                }
        });
        assert!(reset.is_some_and(|reset| reset < begin), "query {query}");
    }
    assert!(commands.contains(&Command::WriteTimestamp {
        pool,
        query: 2,
        stage: PipelineStages::ALL_COMMANDS,
    }));
    assert!(commands.contains(&Command::WriteTimestamp {
        pool,
        query: 1,
        stage: PipelineStages::FRAGMENT_SHADER,
    }));
}

#[test]
fn blits_resolve_counters_on_the_gpu() {
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    let samples = buffer(9, 128);
    commands
        .sample_counters(
            &mut resolver,
            CounterSample::Statistic(samples.address),
            false,
        )
        .unwrap();
    commands.begin_blit().unwrap();
    let timestamps = CounterRange::Timestamps {
        pool: QueryPool(30),
        first: 1,
        count: 2,
    };
    assert!(
        commands
            .resolve_counters(&mut resolver, timestamps, buffer(5, 64), 4)
            .is_err(),
        "results are 8-byte aligned"
    );
    commands
        .resolve_counters(&mut resolver, timestamps, buffer(5, 64), 8)
        .unwrap();
    commands
        .resolve_counters(
            &mut resolver,
            CounterRange::Statistics(samples),
            buffer(6, 256),
            0,
        )
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;

    let fill = commands
        .iter()
        .position(|command| {
            *command
                == Command::FillBuffer {
                    buffer: Buffer(5),
                    offset: 8,
                    size: 16,
                    data: u32::MAX,
                }
        })
        .expect("untaken samples hold the error value");
    let copy = commands
        .iter()
        .position(|command| {
            matches!(
                command,
                Command::CopyQueryResults {
                    first: 1,
                    count: 2,
                    wait: false,
                    ..
                }
            )
        })
        .unwrap();
    assert!(fill < copy);
    let accumulate = commands
        .iter()
        .position(|command| matches!(command, Command::AccumulateResults { .. }))
        .unwrap();
    let resolved = commands
        .iter()
        .position(|command| matches!(command, Command::CopyBuffer { src: Buffer(9), .. }))
        .unwrap();
    assert!(
        accumulate < resolved,
        "pending statistic samples resolve first"
    );
    assert_eq!(
        count(commands, |c| matches!(c, Command::AccumulateResults { .. })),
        1,
        "commit has nothing left to resolve"
    );
}

fn icb_descriptor(command_types: IndirectCommandType) -> IndirectCommandBufferDescriptor {
    IndirectCommandBufferDescriptor {
        command_types,
        inherit_pipeline_state: false,
        inherit_buffers: false,
        inherit_cull_mode: false,
        inherit_depth_stencil_state: true,
        inherit_depth_bias: true,
        inherit_depth_clip_mode: true,
        inherit_front_facing_winding: true,
        inherit_triangle_fill_mode: true,
        max_vertex_buffer_bind_count: 4,
        max_fragment_buffer_bind_count: 1,
        max_kernel_buffer_bind_count: 1,
        max_kernel_threadgroup_memory_bind_count: 0,
        max_object_buffer_bind_count: 0,
        max_mesh_buffer_bind_count: 0,
        support_ray_tracing: false,
        support_dynamic_attribute_stride: false,
    }
}

/// A resolver knowing the encoder's pipeline (id 0, `Pipeline(42)`), an
/// indirect pipeline (id 1, `Pipeline(43)`) and `buffer` as indirect
/// command buffer 0.
fn icb_resolver(buffer: IndirectCommandBuffer) -> MockResolver {
    let mut other = render_state(1);
    other.id = 1;
    other.pipeline = Some(Pipeline(43));
    MockResolver {
        render_states: alloc::vec![render_state(1), other],
        indirect: alloc::vec![Arc::new(buffer)],
        ..MockResolver::default()
    }
}

fn render_icb(count: u64) -> IndirectCommandBuffer {
    let types = IndirectCommandType::DRAW | IndirectCommandType::DRAW_INDEXED;
    let mut icb =
        IndirectCommandBuffer::new(&capabilities(), &icb_descriptor(types), count).unwrap();
    let mut other = render_state(1);
    other.id = 1;
    for index in 0..count as usize {
        let mut command = icb.render_command(index).unwrap();
        command.set_render_pipeline_state(&other).unwrap();
        command
            .set_buffer(Stage::Vertex, buffer(5, 256), 0, 2)
            .unwrap();
        command.set_cull_mode(CullModeAbi::Back).unwrap();
        command
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }
    icb
}

fn vertex_buffers(commands: &[Command]) -> Vec<Buffer> {
    commands
        .iter()
        .filter_map(|command| match command {
            Command::BindVertexBuffers { buffers, .. } => Some(buffers[0].buffer),
            _ => None,
        })
        .collect()
}

#[test]
fn indirect_draws_replay_and_restore_the_encoder() {
    let mut icb = render_icb(3);
    icb.render_command(1).unwrap().reset().unwrap();
    let mut indexed = icb.render_command(2).unwrap();
    indexed
        .draw_indexed_primitives(
            PrimitiveType::Triangle,
            6,
            IndexType::UInt16,
            buffer(6, 64),
            0,
            1,
            0,
            0,
        )
        .unwrap();
    let mut resolver = icb_resolver(icb);
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    commands
        .set_render_pipeline_state(&render_state(1))
        .unwrap();
    commands
        .set_render_buffer(Stage::Vertex, Some(buffer(3, 256)), 0, 2)
        .unwrap();
    commands
        .execute_commands(
            &mut resolver,
            IndirectCommandBufferId(0),
            ExecutionRange::Direct(0..3),
        )
        .unwrap();
    commands
        .draw_primitives(&mut resolver, PrimitiveType::Triangle, 0, 4, 1, 0)
        .unwrap();
    assert!(
        commands
            .execute_commands(
                &mut resolver,
                IndirectCommandBufferId(0),
                ExecutionRange::Direct(2..4)
            )
            .is_err()
    );
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;

    let pipelines: Vec<Pipeline> = commands
        .iter()
        .filter_map(|command| match command {
            Command::BindPipeline { pipeline, .. } => Some(*pipeline),
            _ => None,
        })
        .collect();
    assert_eq!(
        pipelines,
        [Pipeline(42), Pipeline(43), Pipeline(42)],
        "the commands share a pipeline, then the encoder's returns"
    );
    assert_eq!(
        vertex_buffers(commands),
        [Buffer(5), Buffer(3)],
        "the commands' buffers bind once"
    );
    let culls: Vec<&Command> = commands
        .iter()
        .filter(|command| matches!(command, Command::SetCullMode(_)))
        .collect();
    assert_eq!(
        culls[1..],
        [
            &Command::SetCullMode(CullMode::BACK),
            &Command::SetCullMode(CullMode::EMPTY)
        ]
    );
    let draws: Vec<u32> = commands
        .iter()
        .filter_map(|command| match command {
            Command::Draw { vertex_count, .. } => Some(*vertex_count),
            Command::DrawIndexed { index_count, .. } => Some(*index_count),
            _ => None,
        })
        .collect();
    assert_eq!(draws, [3, 6, 4], "the reset command draws nothing");
}

#[test]
fn gpu_ranges_select_commands_outside_the_pass() {
    let mut resolver = icb_resolver(render_icb(2));
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    let range = |offset| ExecutionRange::Indirect {
        buffer: buffer(9, 64),
        offset,
    };
    let id = IndirectCommandBufferId(0);
    assert!(
        commands
            .execute_commands(&mut resolver, id, range(2))
            .is_err()
    );
    assert!(
        commands
            .execute_commands(&mut resolver, id, range(60))
            .is_err()
    );
    commands
        .execute_commands(&mut resolver, id, range(8))
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;

    let select = commands
        .iter()
        .position(|command| matches!(command, Command::SelectCommands { count: 2, .. }))
        .unwrap();
    assert!(matches!(commands[select - 2], Command::EndRendering));
    let resume = select
        + commands[select..]
            .iter()
            .position(|command| matches!(command, Command::BeginRendering(_)))
            .unwrap();
    assert!(
        commands[select..resume]
            .iter()
            .any(|command| matches!(command, Command::Barrier(barrier)
                if barrier.dst_stages.contains(PipelineStages::CONDITIONAL_RENDERING))),
        "the flags are visible to conditional rendering"
    );
    let Command::SelectCommands { arguments, .. } = commands[select] else {
        unreachable!()
    };
    let arguments = uploaded(&recording, arguments);
    assert_eq!(double(arguments, 0), (9 << 32) + 8);
    assert_eq!(word(arguments, 16), 2);
    let flags = double(arguments, 8);

    let gated: Vec<(usize, u64)> = commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| match command {
            Command::BeginConditionalRendering { offset, .. } => Some((index, *offset)),
            _ => None,
        })
        .collect();
    assert_eq!(gated.len(), 2);
    let flag_offset = |address: u64| {
        let (chunk, _) = recording
            .uploads
            .iter()
            .find(|(chunk, _)| (chunk.address..chunk.address + chunk.size).contains(&address))
            .unwrap();
        address - chunk.address
    };
    assert_eq!(gated[0].1, flag_offset(flags));
    assert_eq!(gated[1].1, flag_offset(flags) + 4);
    for &(begin, _) in &gated {
        let end = begin
            + commands[begin..]
                .iter()
                .position(|command| matches!(command, Command::EndConditionalRendering))
                .unwrap();
        assert_eq!(
            count(&commands[begin..end], |command| matches!(
                command,
                Command::Draw { .. }
            )),
            1,
            "each command's draw is gated by its flag"
        );
    }
}

#[test]
fn indirect_dispatches_compile_their_variants() {
    let types =
        IndirectCommandType::CONCURRENT_DISPATCH | IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
    let mut icb = IndirectCommandBuffer::new(&capabilities(), &icb_descriptor(types), 2).unwrap();
    let state = kernel_state(true);
    let threads = |width| Size {
        width,
        height: 1,
        depth: 1,
    };
    let mut first = icb.compute_command(0).unwrap();
    first.set_compute_pipeline_state(&state).unwrap();
    first.set_kernel_buffer(buffer(4, 64), 0, 0).unwrap();
    first
        .concurrent_dispatch_threadgroups([2, 1, 1], threads(64))
        .unwrap();
    let mut second = icb.compute_command(1).unwrap();
    second.set_compute_pipeline_state(&state).unwrap();
    second.set_kernel_buffer(buffer(4, 64), 0, 0).unwrap();
    second.set_barrier().unwrap();
    second
        .concurrent_dispatch_threads([100, 1, 1], threads(32))
        .unwrap();
    let mut resolver = MockResolver {
        compute_states: alloc::vec![state],
        indirect: alloc::vec![Arc::new(icb)],
        ..MockResolver::default()
    };
    let mut commands = CommandBuffer::new(FEATURES);
    commands.begin_compute(DispatchType::Concurrent).unwrap();
    commands
        .execute_commands(
            &mut resolver,
            IndirectCommandBufferId(0),
            ExecutionRange::Direct(0..2),
        )
        .unwrap();
    assert_eq!(
        commands.dispatch_threadgroups(&mut resolver, [1, 1, 1], threads(1)),
        Err(Error::Invalid("no compute pipeline state is set")),
        "the encoder's own state returns"
    );
    commands.end_encoding(&mut resolver).unwrap();
    assert_eq!(
        resolver.compiled.len(),
        2,
        "one variant per threadgroup size"
    );
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;
    let dispatches: Vec<usize> = commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| matches!(command, Command::Dispatch(_)).then_some(index))
        .collect();
    assert_eq!(commands[dispatches[0]], Command::Dispatch([2, 1, 1]));
    assert_eq!(commands[dispatches[1]], Command::Dispatch([4, 1, 1]));
    assert!(
        commands[dispatches[0]..dispatches[1]].contains(&Command::Barrier(FULL)),
        "the second command's barrier"
    );
}

#[test]
fn blits_reset_and_copy_indirect_commands() {
    let types = IndirectCommandType::DRAW;
    let empty = IndirectCommandBuffer::new(&capabilities(), &icb_descriptor(types), 3).unwrap();
    let mut resolver = icb_resolver(render_icb(2));
    resolver.indirect.push(Arc::new(empty));
    let (source, destination) = (IndirectCommandBufferId(0), IndirectCommandBufferId(1));
    let mut commands = CommandBuffer::new(FEATURES);
    assert!(
        commands
            .reset_commands(&mut resolver, source, 0..1)
            .is_err(),
        "outside a blit encoder"
    );
    commands.begin_blit().unwrap();
    commands
        .copy_indirect_commands(&mut resolver, source, 0..2, destination, 1)
        .unwrap();
    assert!(
        commands
            .copy_indirect_commands(&mut resolver, source, 0..2, destination, 2)
            .is_err()
    );
    commands
        .reset_commands(&mut resolver, source, 1..2)
        .unwrap();
    commands
        .optimize_indirect_commands(&mut resolver, destination, 0..3)
        .unwrap();
    assert!(
        commands
            .optimize_indirect_commands(&mut resolver, destination, 2..4)
            .is_err()
    );
    let copied = resolver.indirect[1].commands(0..3).unwrap();
    assert!(copied[0].action.is_none());
    assert!(copied[1].action.is_some() && copied[2].action.is_some());
    let reset = resolver.indirect[0].commands(0..2).unwrap();
    assert!(reset[0].action.is_some() && reset[1].action.is_none());
}

#[test]
fn blits_copy_records_of_resident_buffers_in_order() {
    let descriptor = IndirectCommandBufferDescriptor {
        inherit_pipeline_state: true,
        inherit_cull_mode: true,
        ..icb_descriptor(IndirectCommandType::DRAW)
    };
    let new = || IndirectCommandBuffer::new(&capabilities(), &descriptor, 3).unwrap();
    let mut source = new();
    source
        .render_command(0)
        .unwrap()
        .draw_primitives(PrimitiveType::Point, 0, 3, 1, 0)
        .unwrap();
    let narrow = IndirectCommandBufferDescriptor {
        max_vertex_buffer_bind_count: 1,
        ..descriptor
    };
    let narrow = IndirectCommandBuffer::new(&capabilities(), &narrow, 3).unwrap();
    let mut resolver = MockResolver {
        indirect: [source, new(), new(), narrow].map(Arc::new).to_vec(),
        ..MockResolver::default()
    };
    let id = IndirectCommandBufferId;
    let (layout, records) = resolver.indirect_records(id(1)).unwrap();
    let mut commands = CommandBuffer::new(FEATURES);
    commands.begin_blit().unwrap();
    // Into a resident buffer, the CPU's commands land as records.
    commands
        .copy_indirect_commands(&mut resolver, id(0), 0..2, id(1), 1)
        .unwrap();
    assert!(resolver.indirect[0].records().is_none());
    // From one, the destination becomes resident.
    commands
        .copy_indirect_commands(&mut resolver, id(1), 0..3, id(2), 0)
        .unwrap();
    assert!(resolver.indirect[2].records().is_some());
    assert!(
        commands
            .copy_indirect_commands(&mut resolver, id(1), 0..1, id(3), 0)
            .is_err(),
        "the layouts differ"
    );
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let copies: Vec<_> = recording.segments[0]
        .commands
        .iter()
        .filter_map(|command| match command {
            Command::CopyBuffer { src, dst, region } => Some((*src, *dst, *region)),
            _ => None,
        })
        .collect();
    let stride = u64::from(layout.stride);
    assert_eq!(copies.len(), 2);
    assert_eq!(
        (copies[0].1, copies[0].2.dst_offset, copies[0].2.size),
        (records.buffer, layout.record(1), 2 * stride)
    );
    assert_eq!(
        (copies[1].0, copies[1].2.src_offset, copies[1].2.size),
        (records.buffer, layout.record(0), 3 * stride)
    );
}

/// A GPU-resident buffer of `count` triangle draws under `descriptor`, in
/// a resolver of its own.
fn resident_icb(descriptor: IndirectCommandBufferDescriptor, count: u64) -> MockResolver {
    let mut icb = IndirectCommandBuffer::new(&capabilities(), &descriptor, count).unwrap();
    for index in 0..count as usize {
        icb.render_command(index)
            .unwrap()
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }
    let mut resolver = MockResolver {
        indirect: alloc::vec![Arc::new(icb)],
        ..MockResolver::default()
    };
    resolver
        .indirect_records(IndirectCommandBufferId(0))
        .unwrap();
    resolver
}

fn generated(commands: &[Command]) -> Vec<GeneratedCommands> {
    commands
        .iter()
        .filter_map(|command| match command {
            Command::ExecuteGeneratedCommands(generated) => Some(*generated),
            _ => None,
        })
        .collect()
}

fn u64_at(bytes: &[u8], at: u32) -> u64 {
    u64::from_le_bytes(bytes[at as usize..at as usize + 8].try_into().unwrap())
}

#[test]
fn resident_draws_execute_one_generated_sequence_per_slot() {
    let descriptor = IndirectCommandBufferDescriptor {
        inherit_pipeline_state: true,
        inherit_cull_mode: true,
        ..icb_descriptor(IndirectCommandType::DRAW)
    };
    let mut resolver = resident_icb(descriptor, 3);
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    let id = IndirectCommandBufferId(0);
    assert_eq!(
        commands.execute_commands(&mut resolver, id, ExecutionRange::Direct(0..2)),
        Err(Error::Invalid("no render pipeline state is set"))
    );
    commands
        .set_render_pipeline_state(&render_state(1))
        .unwrap();
    commands
        .execute_commands(&mut resolver, id, ExecutionRange::Direct(1..3))
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    assert_eq!(
        resolver.layouts,
        [GeneratedLayoutKey {
            stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            execution_set: false,
            work: GeneratedWork::Draw {
                indexed: false,
                vertex_buffers: alloc::vec![2],
            },
        }]
    );
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;

    let resolve = commands
        .iter()
        .position(|command| matches!(command, Command::ResolveDraws { count: 2, .. }))
        .unwrap();
    let Command::ResolveDraws { arguments, .. } = commands[resolve] else {
        unreachable!()
    };
    let arguments = uploaded(&recording, arguments);
    assert_eq!(word(arguments, ResolveArguments::FIRST_OFFSET as usize), 1);
    assert_eq!(
        word(arguments, ResolveArguments::FLAGS_OFFSET as usize),
        ResolveArguments::INHERIT_PIPELINE
    );
    assert_eq!(
        word(arguments, ResolveArguments::SLOT_COUNT_OFFSET as usize),
        2,
        "the triangle primitive types"
    );
    let range = u64_at(arguments, ResolveArguments::RANGE_OFFSET);
    assert_eq!(&uploaded(&recording, range)[..8], [1, 0, 0, 0, 2, 0, 0, 0]);
    assert!(
        commands[resolve..]
            .iter()
            .any(|command| matches!(command, Command::Barrier(barrier)
            if barrier.dst_stages.contains(PipelineStages::COMMAND_PREPROCESS))),
        "the sequences are visible to preprocessing"
    );

    // Each command runs once per primitive type, in order.
    let executions = generated(commands);
    assert_eq!(executions.len(), 2 * 2);
    let counts: Vec<u64> = executions
        .iter()
        .map(|execution| execution.count.unwrap())
        .collect();
    assert!(counts.windows(2).all(|pair| pair[1] == pair[0] + 4));
    assert_eq!(
        executions[1].address + u64::from(GeneratedSequence::BYTES),
        executions[2].address
    );
    assert!(executions.iter().all(|execution| execution.set.is_none()
        && execution.sequences == 1
        && execution.preprocess % 64 == 0
        && execution.preprocess_size == 96));
    let topologies = commands
        .iter()
        .filter(|command| matches!(command, Command::SetPrimitiveTopology(_)))
        .count();
    assert_eq!(topologies, 2 * 2, "one per execution");
    let last = commands
        .iter()
        .rposition(|command| matches!(command, Command::ExecuteGeneratedCommands(_)))
        .unwrap();
    assert!(
        commands[last..].contains(&Command::BindPipeline {
            bind_point: PipelineBindPoint::Graphics,
            pipeline: Pipeline(42),
        }),
        "the encoder pipeline returns"
    );
}

#[test]
fn amplified_resident_draws_run_once_for_each_view() {
    let descriptor = IndirectCommandBufferDescriptor {
        inherit_pipeline_state: true,
        inherit_cull_mode: true,
        ..icb_descriptor(IndirectCommandType::DRAW)
    };
    let mut resolver = resident_icb(descriptor, 2);
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    commands
        .set_render_pipeline_state(&render_state(1))
        .unwrap();
    commands
        .set_vertex_amplification(&[(0, 0), (1, 3)])
        .unwrap();
    commands
        .execute_commands(
            &mut resolver,
            IndirectCommandBufferId(0),
            ExecutionRange::Direct(0..2),
        )
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;
    // Each view resolves its own sequences, whose push constants are the
    // view's, then runs every command's slots.
    let resolves: Vec<u64> = commands
        .iter()
        .filter_map(|command| match command {
            Command::ResolveDraws { arguments, .. } => Some(*arguments),
            _ => None,
        })
        .collect();
    assert_eq!(resolves.len(), 2);
    assert_eq!(generated(commands).len(), 2 * 2 * 2);
    for (view, &arguments) in resolves.iter().enumerate() {
        let arguments = uploaded(&recording, arguments);
        let push = ResolveArguments::PUSH_OFFSET as usize;
        let at = |offset: u32| word(arguments, push + offset as usize);
        assert_eq!(at(PushConstants::AMPLIFICATION_OFFSET), view as u32);
        assert_eq!(at(PushConstants::AMPLIFICATIONS_OFFSET), 2);
        assert_eq!(
            at(PushConstants::VIEW_MAPPING_OFFSET),
            [0, 3 << 16 | 1][view]
        );
    }
}

#[test]
fn resident_patch_draws_run_as_point_draws_of_the_encoders_pipeline() {
    let descriptor = IndirectCommandBufferDescriptor {
        inherit_pipeline_state: true,
        inherit_cull_mode: true,
        ..icb_descriptor(IndirectCommandType::DRAW_INDEXED_PATCHES)
    };
    let mut icb = IndirectCommandBuffer::new(&capabilities(), &descriptor, 2).unwrap();
    let source = PatchSource {
        control_points: 4,
        patch_indices: None,
        control_point_indices: Some((buffer(5, 4096), 64)),
    };
    for index in 0..2 {
        icb.render_command(index)
            .unwrap()
            .draw_patches(&source, 0, 2, 1, 0, (buffer(4, 1024), 96, 48))
            .unwrap();
    }
    let mut resolver = MockResolver {
        indirect: alloc::vec![Arc::new(icb)],
        ..MockResolver::default()
    };
    resolver
        .indirect_records(IndirectCommandBufferId(0))
        .unwrap();
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    let state = patch_state(None);
    commands.set_render_pipeline_state(&state).unwrap();
    commands.set_tessellation_factor_scale(2.0).unwrap();
    commands
        .execute_commands(
            &mut resolver,
            IndirectCommandBufferId(0),
            ExecutionRange::Direct(0..2),
        )
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    // No vertex buffer feeds the patches' points.
    assert_eq!(
        resolver
            .layouts
            .iter()
            .map(|key| &key.work)
            .collect::<Vec<_>>(),
        [&GeneratedWork::Draw {
            indexed: false,
            vertex_buffers: Vec::new(),
        }]
    );
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;
    let Some(&Command::ResolveDraws { arguments, .. }) = commands
        .iter()
        .find(|command| matches!(command, Command::ResolveDraws { .. }))
    else {
        panic!("no resolve");
    };
    let arguments = uploaded(&recording, arguments);
    assert_eq!(
        word(arguments, ResolveArguments::FLAGS_OFFSET as usize),
        ResolveArguments::INHERIT_PIPELINE | ResolveArguments::SCALES
    );
    assert_eq!(
        word(arguments, ResolveArguments::FACTOR_SCALE_OFFSET as usize),
        2.0f32.to_bits(),
        "the pipeline scales its factors"
    );
    assert_eq!(
        word(arguments, ResolveArguments::SLOT_COUNT_OFFSET as usize),
        1
    );
    let slot = ResolveArguments::SLOTS_OFFSET as usize;
    assert_eq!(
        [word(arguments, slot + 4), word(arguments, slot + 8)],
        [RecordLayout::DRAW_INDEXED_PATCHES, 0]
    );
    // One execution per command, each a patch list's.
    assert_eq!(generated(commands).len(), 2);
    let topologies: Vec<_> = commands
        .iter()
        .filter_map(|command| match command {
            Command::SetPrimitiveTopology(topology) => Some(*topology),
            _ => None,
        })
        .collect();
    assert_eq!(
        topologies,
        [crate::metal::vulkan::PrimitiveTopology::PatchList]
    );
    assert!(commands.contains(&Command::BindPipeline {
        bind_point: PipelineBindPoint::Graphics,
        pipeline: Pipeline(43),
    }));
}

#[test]
fn resident_draws_switch_pipelines_through_execution_sets() {
    let descriptor = IndirectCommandBufferDescriptor {
        inherit_cull_mode: true,
        support_dynamic_attribute_stride: true,
        ..icb_descriptor(IndirectCommandType::DRAW | IndirectCommandType::DRAW_INDEXED)
    };
    let mut resolver = resident_icb(descriptor, 1);
    let class = |index, pipeline| IndirectRenderClass {
        index,
        set: Some(ExecutionSet(10 + u64::from(index))),
        pipeline: Pipeline(pipeline),
        stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
        rasterization: true,
        topology: PrimitiveTopologyClass::Triangle,
        patches: None,
        vertex_buffers: alloc::vec![(0, Some(16)), (3, None)],
        storage_attachments: 0,
        blended_attachments: 1,
        tests_fragments: false,
    };
    resolver.classes = alloc::vec![class(0, 70), class(2, 71)];
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    commands
        .execute_commands(
            &mut resolver,
            IndirectCommandBufferId(0),
            ExecutionRange::Direct(0..1),
        )
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    assert_eq!(resolver.layouts.len(), 2, "draws and indexed draws");
    assert!(resolver.layouts.iter().all(|key| key.execution_set));
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;

    // Two classes, two draw kinds, two triangle primitive types.
    let executions = generated(commands);
    assert_eq!(executions.len(), 8);
    assert_eq!(executions[0].set, Some(ExecutionSet(10)));
    assert_eq!(executions[7].set, Some(ExecutionSet(12)));
    let binds = commands
        .iter()
        .filter(|command| matches!(command, Command::BindPipeline { .. }))
        .count();
    assert_eq!(binds, 8, "every execution binds a pipeline of its class");
    let arguments = commands
        .iter()
        .find_map(|command| match command {
            Command::ResolveDraws { arguments, .. } => Some(*arguments),
            _ => None,
        })
        .unwrap();
    let arguments = uploaded(&recording, arguments);
    assert_eq!(
        word(arguments, ResolveArguments::CLASS_COUNT_OFFSET as usize),
        3
    );
    // A pipeline the commands set scales factors in its own control stage.
    assert_eq!(
        word(arguments, ResolveArguments::FLAGS_OFFSET as usize),
        ResolveArguments::SCALES
    );
    let strides = uploaded(
        &recording,
        u64_at(arguments, ResolveArguments::STRIDES_OFFSET),
    );
    assert_eq!(word(strides, 0), 16, "buffer 0 of class 0");
    assert_eq!(word(strides, 4 * (2 * 31)), 16, "buffer 0 of class 2");
    assert_eq!(word(strides, 4 * (2 * 31 + 3)), 0, "a dynamic stride");
    let slot = ResolveArguments::SLOTS_OFFSET as usize + 7 * 16;
    assert_eq!(
        [
            word(arguments, slot),
            word(arguments, slot + 4),
            word(arguments, slot + 8)
        ],
        [
            2,
            RecordLayout::DRAW_INDEXED,
            PrimitiveType::TriangleStrip.raw() as u32
        ]
    );
}

#[test]
fn resident_barriers_order_every_command() {
    let descriptor = IndirectCommandBufferDescriptor {
        inherit_pipeline_state: true,
        inherit_cull_mode: true,
        ..icb_descriptor(IndirectCommandType::DRAW)
    };
    let mut resolver = resident_icb(descriptor, 2);
    let contents = Arc::make_mut(&mut resolver.indirect[0]);
    assert!(!contents.has_barriers());
    contents.render_command(1).unwrap().set_barrier().unwrap();
    assert!(contents.has_barriers());
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(
            &mut resolver,
            pass(&color, StoreAction::Store, false),
            None,
            Vec::new(),
        )
        .unwrap();
    commands
        .set_render_pipeline_state(&render_state(1))
        .unwrap();
    commands
        .execute_commands(
            &mut resolver,
            IndirectCommandBufferId(0),
            ExecutionRange::Direct(0..2),
        )
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    // The resolve and each command end the pass instance, as does the
    // encoder.
    let ends = recording.segments[0]
        .commands
        .iter()
        .filter(|command| matches!(command, Command::EndRendering))
        .count();
    assert_eq!(ends, 1 + 2 + 1);
}

#[test]
fn generated_layouts_read_the_sequence_at_token_offsets() {
    let key = GeneratedLayoutKey {
        stages: ShaderStages::VERTEX,
        execution_set: true,
        work: GeneratedWork::Draw {
            indexed: true,
            vertex_buffers: alloc::vec![1, 30],
        },
    };
    let info = key.info();
    assert_eq!(info.stride, GeneratedSequence::BYTES);
    assert_eq!(
        info.tokens,
        [
            (0, IndirectToken::ExecutionSet(ShaderStages::VERTEX)),
            (
                8,
                IndirectToken::PushConstants {
                    offset: 0,
                    size: 64
                }
            ),
            (72, IndirectToken::IndexBuffer),
            (104, IndirectToken::VertexBuffer(1)),
            (568, IndirectToken::VertexBuffer(30)),
            (600, IndirectToken::DrawIndexed),
        ]
    );
}

/// A compute pipeline state whose indirect variant is pipeline 77, at index
/// 3 of the compute execution set.
fn indirect_kernel_state() -> ComputePipelineState {
    ComputePipelineState {
        indirect: Some(IndirectCompute {
            pipeline: Pipeline(77),
            place: IndirectPipeline {
                class: IndirectPipeline::COMPUTE_CLASS,
                index: 3,
            },
        }),
        ..kernel_state(true)
    }
}

/// A GPU-resident buffer of `count` dispatches under `descriptor`, with
/// `state` unless the buffer inherits it, the last after a barrier when
/// `barrier`, in a resolver of its own.
fn resident_dispatches(
    descriptor: IndirectCommandBufferDescriptor,
    count: usize,
    state: Option<&ComputePipelineState>,
    barrier: bool,
) -> MockResolver {
    let mut icb = IndirectCommandBuffer::new(&capabilities(), &descriptor, count as u64).unwrap();
    for index in 0..count {
        let mut command = icb.compute_command(index).unwrap();
        if let Some(state) = state {
            command.set_compute_pipeline_state(state).unwrap();
        }
        if barrier && index + 1 == count {
            command.set_barrier().unwrap();
        }
        let threads = Size {
            width: 64,
            height: 1,
            depth: 1,
        };
        command
            .concurrent_dispatch_threadgroups([2, 1, 1], threads)
            .unwrap();
    }
    let mut resolver = MockResolver {
        indirect: alloc::vec![Arc::new(icb)],
        ..MockResolver::default()
    };
    resolver
        .indirect_records(IndirectCommandBufferId(0))
        .unwrap();
    resolver
}

#[test]
fn resident_dispatches_run_concurrently_in_one_execution() {
    let descriptor = icb_descriptor(IndirectCommandType::CONCURRENT_DISPATCH);
    let state = indirect_kernel_state();
    let mut resolver = resident_dispatches(descriptor, 3, Some(&state), false);
    let id = IndirectCommandBufferId(0);
    let mut commands = CommandBuffer::new(FEATURES);
    commands.begin_compute(DispatchType::Concurrent).unwrap();
    // Without an indirect-bindable compute pipeline, no command can run.
    commands
        .execute_commands(&mut resolver, id, ExecutionRange::Direct(0..3))
        .unwrap();
    resolver.compute_set = Some((ExecutionSet(20), Pipeline(78)));
    commands
        .execute_commands(&mut resolver, id, ExecutionRange::Direct(0..3))
        .unwrap();
    commands
        .set_compute_pipeline_state(&kernel_state(true))
        .unwrap();
    let threads = Size {
        width: 1,
        height: 1,
        depth: 1,
    };
    commands
        .dispatch_threadgroups(&mut resolver, [1, 1, 1], threads)
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    assert_eq!(
        resolver.layouts,
        [GeneratedLayoutKey {
            stages: ShaderStages::COMPUTE,
            execution_set: true,
            work: GeneratedWork::Dispatch,
        }]
    );
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;

    let resolves: Vec<usize> = commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| {
            matches!(command, Command::ResolveDispatches { count: 3, .. }).then_some(index)
        })
        .collect();
    assert_eq!(resolves.len(), 1);
    let Command::ResolveDispatches { arguments, .. } = commands[resolves[0]] else {
        unreachable!()
    };
    let arguments = uploaded(&recording, arguments);
    assert_eq!(word(arguments, ResolveArguments::FLAGS_OFFSET as usize), 0);
    assert_ne!(u64_at(arguments, ResolveArguments::TABLES_OFFSET), 0);
    assert!(
        commands[resolves[0]..]
            .iter()
            .any(|command| matches!(command, Command::Barrier(barrier)
            if barrier.dst_stages.contains(PipelineStages::COMMAND_PREPROCESS))),
        "the sequences are visible to preprocessing"
    );
    // One execution of every command, with a pipeline of the set bound.
    let executions = generated(commands);
    assert_eq!(executions.len(), 1);
    let execution = executions[0];
    assert_eq!(execution.stages, ShaderStages::COMPUTE);
    assert_eq!(execution.set, Some(ExecutionSet(20)));
    assert_eq!(
        (execution.sequences, execution.size, execution.count),
        (3, 3 * u64::from(GeneratedSequence::BYTES), None)
    );
    assert_eq!(execution.preprocess_size, 3 * 96);
    let at = commands
        .iter()
        .position(|command| matches!(command, Command::ExecuteGeneratedCommands(_)))
        .unwrap();
    assert_eq!(
        commands[at - 1],
        Command::BindPipeline {
            bind_point: PipelineBindPoint::Compute,
            pipeline: Pipeline(78),
        }
    );
    // The encoder's next dispatch binds its own pipeline again.
    assert!(commands[at..].iter().any(|command| matches!(
        command,
        Command::BindPipeline {
            bind_point: PipelineBindPoint::Compute,
            pipeline: Pipeline(500),
        }
    )));
}

#[test]
fn resident_dispatches_with_barriers_run_one_execution_each() {
    let descriptor = IndirectCommandBufferDescriptor {
        inherit_pipeline_state: true,
        inherit_buffers: true,
        ..icb_descriptor(IndirectCommandType::CONCURRENT_DISPATCH)
    };
    let mut resolver = resident_dispatches(descriptor, 2, None, true);
    let id = IndirectCommandBufferId(0);
    let mut commands = CommandBuffer::new(FEATURES);
    commands.begin_compute(DispatchType::Serial).unwrap();
    commands
        .set_compute_pipeline_state(&kernel_state(true))
        .unwrap();
    assert_eq!(
        commands.execute_commands(&mut resolver, id, ExecutionRange::Direct(0..2)),
        Err(Error::Invalid(
            "the compute pipeline state does not support indirect command buffers"
        ))
    );
    commands
        .set_compute_pipeline_state(&indirect_kernel_state())
        .unwrap();
    commands
        .execute_commands(&mut resolver, id, ExecutionRange::Direct(0..2))
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    assert_eq!(
        resolver.layouts,
        [GeneratedLayoutKey {
            stages: ShaderStages::COMPUTE,
            execution_set: false,
            work: GeneratedWork::Dispatch,
        }]
    );
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;
    let resolve = commands
        .iter()
        .position(|command| matches!(command, Command::ResolveDispatches { count: 2, .. }))
        .unwrap();
    let Command::ResolveDispatches { arguments, .. } = commands[resolve] else {
        unreachable!()
    };
    let arguments = uploaded(&recording, arguments);
    assert_eq!(
        word(arguments, ResolveArguments::FLAGS_OFFSET as usize),
        ResolveArguments::INHERIT_PIPELINE | ResolveArguments::INHERIT_BUFFERS
    );
    assert_eq!(u64_at(arguments, ResolveArguments::TABLES_OFFSET), 0);
    // Each command runs alone, after a full barrier, with the indirect
    // variant of the encoder's pipeline.
    let executions = generated(commands);
    assert_eq!(executions.len(), 2);
    assert!(executions.iter().all(|execution| execution.set.is_none()
        && execution.sequences == 1
        && execution.preprocess_size == 96));
    assert_eq!(
        executions[0].address + u64::from(GeneratedSequence::BYTES),
        executions[1].address
    );
    assert!(commands[resolve..].contains(&Command::BindPipeline {
        bind_point: PipelineBindPoint::Compute,
        pipeline: Pipeline(77),
    }));
    for (index, command) in commands.iter().enumerate() {
        if matches!(command, Command::ExecuteGeneratedCommands(_)) {
            assert_eq!(
                commands[index - 1],
                Command::Barrier(FULL),
                "a full barrier precedes each command"
            );
        }
    }
}

/// A pass with depth and stencil copies whose pipeline tests fragments
/// itself: its draws run on the copies, which the attachments fill before
/// the first such draw and which return before a draw testing in its
/// pipeline and where the pass ends.
#[test]
fn fragment_testing_draws_run_on_copies_of_the_depth_and_stencil() {
    use crate::metal::pass::TestedCopy;
    let color = texture(PixelFormat::Rgba8Unorm);
    let depth = texture(PixelFormat::Depth32FloatStencil8);
    let mut plan = pass(&color, StoreAction::Store, false);
    let mut descriptor = RenderPassDescriptor {
        color_attachments: [None; MAX_COLOR_ATTACHMENTS],
        depth_attachment: None,
        depth_resolve_filter: crate::metal::abi::MultisampleDepthResolveFilter::Sample0,
        stencil_attachment: None,
        stencil_resolve_filter: crate::metal::abi::MultisampleStencilResolveFilter::Sample0,
        visibility_results: false,
        render_target_array_length: 0,
        render_target_width: 0,
        render_target_height: 0,
        default_raster_sample_count: 0,
        tile_memory: Default::default(),
        sample_positions: Vec::new(),
        rasterization_rate_map: false,
    };
    let target = Target {
        texture: AttachedTexture {
            texture: ImageView(8),
            descriptor: &depth,
        },
        level: 0,
        slice: 0,
        depth_plane: 0,
    };
    descriptor.depth_attachment = Some(AttachmentDescriptor {
        target,
        resolve: None,
        load_action: LoadAction::Clear,
        store_action: StoreAction::Store,
        store_action_options: StoreActionOptions::EMPTY,
        clear: 1.0,
    });
    descriptor.stencil_attachment = Some(AttachmentDescriptor {
        target,
        resolve: None,
        load_action: LoadAction::Clear,
        store_action: StoreAction::Store,
        store_action_options: StoreActionOptions::EMPTY,
        clear: 0,
    });
    let with_depth = render_pass(&capabilities(), &descriptor).unwrap();
    plan.depth = with_depth.depth;
    plan.stencil = with_depth.stencil;
    plan.depth_format = with_depth.depth_format;
    plan.stencil_format = with_depth.stencil_format;
    let copy = |id: u32, aspect| TestedCopy {
        id: ResourceId::texture(0, id, 1),
        image: Image(100 + u64::from(id)),
        attachment: Image(8),
        aspect,
        level: 0,
        base_layer: 0,
        layers: 1,
        extent: [64, 32],
        encoding: 3,
        planes: false,
    };
    plan.inputs.depth_copy = Some(copy(1, crate::metal::vulkan::ImageAspects::DEPTH));
    plan.inputs.stencil_copy = Some(copy(2, crate::metal::vulkan::ImageAspects::STENCIL));
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(&mut resolver, plan, None, Vec::new())
        .unwrap();
    let mut testing = render_state(1);
    testing.plan.tests_fragments = true;
    let plain = render_state(1);
    commands
        .set_render_buffer(Stage::Vertex, Some(buffer(3, 256)), 0, 2)
        .unwrap();
    let draw = |commands: &mut CommandBuffer, resolver: &mut MockResolver| {
        commands
            .draw_primitives(resolver, PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap()
    };
    commands.set_render_pipeline_state(&testing).unwrap();
    draw(&mut commands, &mut resolver);
    draw(&mut commands, &mut resolver);
    // A draw that tests nothing in its pipeline leaves the copies in place.
    commands.set_render_pipeline_state(&plain).unwrap();
    draw(&mut commands, &mut resolver);
    let tested =
        crate::metal::pipeline::depth_stencil(&crate::metal::pipeline::DepthStencilDescriptor {
            depth_compare_function: crate::metal::abi::CompareFunction::Less,
            depth_write_enabled: true,
            ..Default::default()
        });
    commands.set_depth_stencil_state(Some(&tested)).unwrap();
    draw(&mut commands, &mut resolver);
    commands.set_render_pipeline_state(&testing).unwrap();
    draw(&mut commands, &mut resolver);
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let copies: Vec<(Image, Image)> = recording.segments[0]
        .commands
        .iter()
        .filter_map(|command| match command {
            Command::CopyImage { src, dst, .. } => Some((*src, *dst)),
            _ => None,
        })
        .collect();
    let (to, from) = (
        [(Image(8), Image(101)), (Image(8), Image(102))],
        [(Image(101), Image(8)), (Image(102), Image(8))],
    );
    // In, back before the testing draw, in again, back where the pass ends.
    assert_eq!(copies, [to, from, to, from].concat());
    assert_eq!(
        count(&recording.segments[0].commands, |command| *command
            == Command::EndRendering),
        5
    );
}

/// A stencil resolve following a min depth resolve: Vulkan resolves the
/// depth alone and stores both aspects' samples, which the average samples
/// kernel reads through copies where the pass ends, writing a copy of the
/// resolve texture's stencil that then fills it.
#[test]
fn stencil_resolves_following_min_depth_run_on_copies_of_the_samples() {
    use crate::metal::{
        pass::{FollowedStencil, TestedCopy},
        vulkan::{AttachmentStoreOp, ImageAspects, ResolveModes},
    };
    let mut ms = texture(PixelFormat::Depth32FloatStencil8);
    ms.texture_type = TextureType::D2Multisample;
    ms.sample_count = 4;
    ms.mipmap_level_count = 1;
    let mut resolved = texture(PixelFormat::Depth32FloatStencil8);
    resolved.mipmap_level_count = 1;
    fn attachment<'a, Clear>(
        ms: &'a TextureDescriptor,
        resolved: &'a TextureDescriptor,
        clear: Clear,
    ) -> AttachmentDescriptor<'a, ImageView, Clear> {
        let target = |texture: ImageView, descriptor| Target {
            texture: AttachedTexture {
                texture,
                descriptor,
            },
            level: 0,
            slice: 0,
            depth_plane: 0,
        };
        AttachmentDescriptor {
            target: target(ImageView(8), ms),
            resolve: Some(target(ImageView(9), resolved)),
            load_action: LoadAction::Clear,
            store_action: StoreAction::MultisampleResolve,
            store_action_options: StoreActionOptions::EMPTY,
            clear,
        }
    }
    let descriptor = RenderPassDescriptor {
        color_attachments: [None; MAX_COLOR_ATTACHMENTS],
        depth_attachment: Some(attachment(&ms, &resolved, 1.0)),
        depth_resolve_filter: crate::metal::abi::MultisampleDepthResolveFilter::Min,
        stencil_attachment: Some(attachment(&ms, &resolved, 0)),
        stencil_resolve_filter:
            crate::metal::abi::MultisampleStencilResolveFilter::DepthResolvedSample,
        visibility_results: false,
        render_target_array_length: 0,
        render_target_width: 0,
        render_target_height: 0,
        default_raster_sample_count: 0,
        tile_memory: Default::default(),
        sample_positions: Vec::new(),
        rasterization_rate_map: false,
    };
    let mut plan = render_pass(&capabilities(), &descriptor).unwrap();
    let stencil = plan.stencil.as_ref().unwrap();
    assert!(stencil.follows_depth);
    assert_eq!(stencil.resolve.unwrap().0, ResolveModes::MIN);
    let copy = |id: u32, image: u64, attachment: u64, aspect| TestedCopy {
        id: ResourceId::texture(0, id, 1),
        image: Image(image),
        attachment: Image(attachment),
        aspect,
        level: 0,
        base_layer: 0,
        layers: 1,
        extent: [64, 32],
        encoding: 3,
        planes: false,
    };
    plan.inputs.follows_depth = Some(FollowedStencil {
        depth: copy(1, 101, 8, ImageAspects::DEPTH),
        stencil: copy(2, 102, 8, ImageAspects::STENCIL),
        resolve: copy(3, 103, 9, ImageAspects::STENCIL),
    });
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(&mut resolver, plan, None, Vec::new())
        .unwrap();
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let commands = &recording.segments[0].commands;
    let Some(Command::BeginRendering(rendering)) = commands
        .iter()
        .find(|command| matches!(command, Command::BeginRendering(_)))
    else {
        panic!("no rendering")
    };
    let (depth, stencil) = (
        rendering.depth.as_ref().unwrap(),
        rendering.stencil.as_ref().unwrap(),
    );
    assert_eq!(depth.resolve.map(|(mode, _)| mode), Some(ResolveModes::MIN));
    assert_eq!(stencil.resolve, None, "Vulkan follows no depth resolve");
    assert_eq!(
        (depth.store_op, stencil.store_op),
        (AttachmentStoreOp::Store, AttachmentStoreOp::Store),
        "the kernel reads the samples a resolving store action discards"
    );
    let order: Vec<(Image, Image)> = commands
        .iter()
        .filter_map(|command| match command {
            Command::CopyImage { src, dst, .. } => Some((*src, *dst)),
            Command::AverageSamples { .. } => Some((Image(0), Image(0))),
            _ => None,
        })
        .collect();
    assert_eq!(
        order,
        [
            (Image(8), Image(101)),
            (Image(8), Image(102)),
            (Image(0), Image(0)),
            (Image(103), Image(9)),
        ]
    );
    let Some(&Command::AverageSamples { arguments, count }) = commands
        .iter()
        .find(|command| matches!(command, Command::AverageSamples { .. }))
    else {
        panic!("no average")
    };
    assert_eq!(count, 64 * 32);
    let encoded = uploaded(&recording, arguments);
    assert_eq!(
        [0, 8, 40].map(|at| double(encoded, at)),
        [2, 3, 1].map(|id| ResourceId::texture(0, id, 1).0)
    );
    assert_eq!(
        [16, 20, 24, 28, 32].map(|at| word(encoded, at)),
        [
            64 * 32,
            crate::metal::binding::AverageArguments::FOLLOW_DEPTH | 3,
            64,
            32,
            4
        ]
    );
}

#[test]
fn a_lost_device_names_the_widest_dispatch_and_its_threadgroup_memory() {
    let mut workload = Workload::default();
    let shape = |groups: [u32; 3], threads: [u32; 3]| {
        let mut shape = DispatchShape {
            threadgroups: groups,
            threads,
            ..DispatchShape::default()
        };
        shape.lengths[0] = (0, 4096);
        shape.lengths[1] = (1, 8192);
        shape.length_count = 2;
        shape
    };
    workload.dispatched(shape([3, 2, 1], [16, 16, 1]));
    workload.dispatched(shape([1, 1, 1], [32, 1, 1]));
    assert_eq!(workload.widest, Some(shape([3, 2, 1], [16, 16, 1])));
    assert_eq!(
        alloc::format!("{}", workload.widest.unwrap()),
        "3x2x1 threadgroups of 16x16x1, threadgroup 0 4096 bytes, threadgroup 1 8192 bytes"
    );
}

/// A pass whose color attachment is a 3D texture's depth plane, which no
/// storage view reads: draws accessing it as storage run on its copy, which
/// the plane fills before the first such draw and which returns before a
/// draw rendering it through the blend unit and where the pass ends.
#[test]
fn storage_draws_of_a_3d_plane_run_on_its_copy() {
    use crate::metal::{pass::TestedCopy, vulkan::command::ImageCopy};
    let color = texture(PixelFormat::Rgba8Unorm);
    let mut plan = pass(&color, StoreAction::Store, false);
    let copy = TestedCopy {
        id: ResourceId::texture(0, 5, 1),
        image: Image(105),
        attachment: Image(7),
        aspect: crate::metal::vulkan::ImageAspects::COLOR,
        level: 1,
        base_layer: 2,
        layers: 1,
        extent: [64, 32],
        encoding: 0,
        planes: true,
    };
    plan.inputs.textures[0] = copy.id;
    plan.inputs.color_copies[0] = Some(copy);
    let mut resolver = MockResolver::default();
    let mut commands = CommandBuffer::new(FEATURES);
    commands
        .begin_render(&mut resolver, plan, None, Vec::new())
        .unwrap();
    let mut storage = render_state(1);
    storage.interfaces[1].storage_attachments = 1;
    let blended = render_state(1);
    commands
        .set_render_buffer(Stage::Vertex, Some(buffer(3, 256)), 0, 2)
        .unwrap();
    let draw = |commands: &mut CommandBuffer, resolver: &mut MockResolver| {
        commands
            .draw_primitives(resolver, PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap()
    };
    commands.set_render_pipeline_state(&storage).unwrap();
    draw(&mut commands, &mut resolver);
    draw(&mut commands, &mut resolver);
    commands.set_render_pipeline_state(&blended).unwrap();
    draw(&mut commands, &mut resolver);
    commands.set_render_pipeline_state(&storage).unwrap();
    draw(&mut commands, &mut resolver);
    commands.end_encoding(&mut resolver).unwrap();
    let recording = commit(commands, &mut resolver, (Semaphore(1), 1));
    let copies: Vec<(Image, Image, ImageCopy)> = recording.segments[0]
        .commands
        .iter()
        .filter_map(|command| match command {
            Command::CopyImage { src, dst, regions } => Some((*src, *dst, regions[0])),
            _ => None,
        })
        .collect();
    let pairs: Vec<(Image, Image)> = copies.iter().map(|&(src, dst, _)| (src, dst)).collect();
    let (to, from) = ((Image(7), Image(105)), (Image(105), Image(7)));
    // In, back before the blended draw, in again, back where the pass ends.
    assert_eq!(pairs, [to, from, to, from]);
    // The plane is depth 2 of the attachment's level, the copy's layer 0.
    let into = copies[0].2;
    assert_eq!(
        (
            into.src.mip_level,
            into.src.base_array_layer,
            into.src.layer_count
        ),
        (1, 0, 1)
    );
    assert_eq!((into.src_offset, into.dst_offset), ([0, 0, 2], [0; 3]));
    assert_eq!(into.extent, [64, 32, 1]);
    let back = copies[1].2;
    assert_eq!((back.src_offset, back.dst_offset), ([0; 3], [0, 0, 2]));
}

/// A set call that binds the value already bound keeps the uploaded table;
/// any changed value, offset-only changes included, makes it upload again,
/// and a new encoder's empty bindings always upload.
#[test]
fn table_rebind_of_the_bound_value_keeps_the_upload() {
    let buffer = BufferRef {
        buffer: Buffer(7),
        offset: 0,
        address: 0x1_0000,
        length: 256,
    };
    let table = Slot::from(Stage::Vertex).index();
    let uploaded = |bindings: &mut Bindings| bindings.tables[table].address = Some(0x40);
    let clean = |bindings: &Bindings| bindings.tables[table].address.is_some();
    let mut bindings = Bindings::default();
    bindings
        .set_buffer(Stage::Vertex, Some(buffer), 16, 0)
        .unwrap();
    bindings
        .set_texture(Stage::Vertex, ResourceId(3), 1)
        .unwrap();
    bindings
        .set_sampler(Stage::Vertex, ResourceId(5), 2)
        .unwrap();
    assert!(!clean(&bindings));
    uploaded(&mut bindings);
    bindings
        .set_buffer(Stage::Vertex, Some(buffer), 16, 0)
        .unwrap();
    bindings.set_buffer_offset(Stage::Vertex, 16, 0).unwrap();
    bindings
        .set_texture(Stage::Vertex, ResourceId(3), 1)
        .unwrap();
    bindings
        .set_sampler(Stage::Vertex, ResourceId(5), 2)
        .unwrap();
    assert!(
        clean(&bindings),
        "rebinding the bound values keeps the table"
    );
    let changes: [fn(&mut Bindings); 4] = [
        |b| b.set_buffer_offset(Stage::Vertex, 32, 0).unwrap(),
        |b| b.set_buffer(Stage::Vertex, None, 0, 0).unwrap(),
        |b| b.set_texture(Stage::Vertex, ResourceId(4), 1).unwrap(),
        |b| b.set_sampler(Stage::Vertex, ResourceId(6), 2).unwrap(),
    ];
    for change in changes {
        uploaded(&mut bindings);
        change(&mut bindings);
        assert!(!clean(&bindings), "a changed value uploads the table again");
    }
    uploaded(&mut bindings);
    bindings.reset();
    bindings
        .set_texture(Stage::Vertex, ResourceId::default(), 1)
        .unwrap();
    assert!(!clean(&bindings), "a new encoder's table always uploads");
}
