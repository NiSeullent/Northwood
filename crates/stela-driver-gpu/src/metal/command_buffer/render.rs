//! `MTLRenderCommandEncoder`.
//!
//! A render encoder is one Vulkan render pass instance, split only by
//! Metal memory barriers: the instance ends, a barrier orders the stages,
//! and an instance that loads every attachment resumes. Store actions,
//! resolves and the first instance's clears are patched into the
//! `vkCmdBeginRendering` commands when the encoder ends, after deferred
//! store actions are known.
//!
//! Patch draws are point-patch draws of the pipeline's tessellation stages,
//! each with an upload of its [`TessellationArguments`].
//!
//! Metal's clip space has y up and Vulkan's y down, so viewports are flipped
//! with a negative height; winding is defined in window space by both APIs
//! and maps unchanged.
use super::{
    BoundPipeline, BufferRef, CommandBuffer, Encoder, Recorder, RenderPipelineState, Resolver,
    buffer_slot, counters::CounterSample, fragment_tests,
};
use crate::metal::{
    abi::{
        BarrierScope, CullMode as MtlCullMode, DepthClipMode, IndexType as MtlIndexType,
        PrimitiveTopologyClass, PrimitiveType, RenderStages, StoreAction, StoreActionOptions,
        TessellationControlPointIndexType, TriangleFillMode, VisibilityResultMode,
        VisibilityResultType, Winding,
    },
    binding::{
        AccumulateArguments, AverageArguments, BufferBinding, PassDirty, PassImageblock,
        ResourceId, SequenceVertexBuffer, TessellationArguments, TexelArguments,
    },
    capabilities::MAX_BUFFER_BINDINGS,
    error::{Error, Result, ensure},
    hazard::{RENDER_PASS, Use},
    pass::{AttachmentSlot, ExtendedPlanes, FollowedStencil, RenderPassPlan, RenderingAttachment},
    pipeline::{DepthStencilState, MAX_COLOR_ATTACHMENTS},
    resource::raw_rendered,
    shader::{Interface, MAX_PATCH_CONTROL_POINTS, Stage},
    vertex::{STRIDE_DYNAMIC, VertexDescriptor},
    vulkan::{
        Access, AttachmentLoadOp, AttachmentStoreOp, CullMode, FrontFace, IndexType,
        PipelineBindPoint, PipelineStages, PolygonMode, PrimitiveTopology, ResolveModes,
        SampleCounts,
        command::{
            Attachment, Barrier, BufferCopy, Command, DepthStencil, Rect, Rendering, VertexBuffer,
            Viewport as VkViewport,
        },
        object::{Buffer, ImageView, Pipeline, QueryPool, QueryType},
    },
};
use alloc::{boxed::Box, vec::Vec};

/// `MTLViewport`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    pub origin_x: f64,
    pub origin_y: f64,
    pub width: f64,
    pub height: f64,
    pub znear: f64,
    pub zfar: f64,
}

/// `MTLScissorRect`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScissorRect {
    pub x: u64,
    pub y: u64,
    pub width: u64,
    pub height: u64,
}

/// Where a render pass samples counters (`sampleBufferAttachments`).
/// Statistics count whole pass instances, so start boundaries sample before
/// the pass and end boundaries after it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PassBoundary {
    StartOfVertex,
    EndOfVertex,
    StartOfFragment,
    EndOfFragment,
}

impl PassBoundary {
    const fn starts(self) -> bool {
        matches!(self, Self::StartOfVertex | Self::StartOfFragment)
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct VertexBinding {
    buffer: BufferRef,
    offset: u64,
    /// `setVertexBuffer:offset:attributeStride:`.
    stride: Option<u64>,
}

/// `setTessellationFactorBuffer:offset:instanceStride:`.
#[derive(Clone, Copy, Debug)]
pub(super) struct FactorBuffer {
    buffer: BufferRef,
    offset: u64,
    instance_stride: u32,
}

/// The rasterizer state Metal sets per encoder and an indirect command may
/// set for itself: cull mode, winding, depth bias (constant, slope, clamp),
/// depth-stencil state, triangle fill mode and depth clip mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RasterState {
    pub cull_mode: MtlCullMode,
    pub winding: Winding,
    pub depth_bias: [f32; 3],
    pub depth_stencil: DepthStencilState,
    pub fill_mode: TriangleFillMode,
    pub clip_mode: DepthClipMode,
}

/// Metal's initial encoder values.
impl Default for RasterState {
    fn default() -> Self {
        Self {
            cull_mode: MtlCullMode::None,
            winding: Winding::Clockwise,
            depth_bias: [0.0; 3],
            depth_stencil: DepthStencilState::default(),
            fill_mode: TriangleFillMode::Fill,
            clip_mode: DepthClipMode::Clip,
        }
    }
}

/// What a patch draw reads besides the tessellation factors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatchSource {
    /// `numberOfPatchControlPoints`.
    pub control_points: u32,
    /// `patchIndexBuffer` and its offset: `uint` patch numbers.
    pub patch_indices: Option<(BufferRef, u64)>,
    /// `controlPointIndexBuffer` and its offset, for indexed patch draws.
    pub control_point_indices: Option<(BufferRef, u64)>,
}

/// The strides of the vertex buffers a patch draw's evaluation stage
/// fetches `function`'s stage-in from, by `layout`: each buffer is bound at
/// a multiple of four, and those of dynamic stride with a stride. Zero for
/// the buffers whose layout fixes it.
fn patch_strides(
    vertex_buffers: &[Option<VertexBinding>; 31],
    function: &Interface,
    layout: Option<&VertexDescriptor>,
) -> Result<[u32; 31]> {
    let mut strides = [0; 31];
    let Some(layout) = layout else {
        return Ok(strides);
    };
    for used in &function.attributes {
        let attribute = layout
            .attributes
            .iter()
            .find(|attribute| attribute.index == used.index)
            .ok_or(Error::Invalid(
                "stage-in attribute missing from the vertex descriptor",
            ))?;
        let index = attribute.buffer_index as usize;
        let bound = vertex_buffers
            .get(index)
            .copied()
            .flatten()
            .ok_or(Error::Invalid(
                "a vertex buffer the pipeline reads is unbound",
            ))?;
        ensure(
            bound.offset.is_multiple_of(4),
            "stage-in vertex buffers are bound at multiples of four",
        )?;
        let dynamic = layout.layouts.iter().any(|layout| {
            layout.buffer_index == attribute.buffer_index && layout.stride == STRIDE_DYNAMIC
        });
        if dynamic {
            let stride = bound.stride.ok_or(Error::Invalid(
                "a dynamic-stride vertex buffer was bound without a stride",
            ))?;
            ensure(
                stride.is_multiple_of(4),
                "vertex strides are multiples of four",
            )?;
            strides[index] =
                u32::try_from(stride).map_err(|_| Error::Invalid("vertex buffer stride"))?;
        }
    }
    Ok(strides)
}

/// Visibility queries of one pass: each mode change or pass restart ends
/// the active query, and offsets used by several queries are summed.
#[derive(Debug, Default)]
struct Visibility {
    buffer: Option<BufferRef>,
    /// The pass adds its results to what the buffer holds.
    accumulate: bool,
    mode: Option<(VisibilityResultMode, u64)>,
    pool: Option<(QueryPool, u32, u32)>,
    active: Option<(QueryPool, u32)>,
    /// (pool, query, offset), in query order.
    queries: Vec<(QueryPool, u32, u64)>,
}

/// Packed view mappings. A render encoder normally has one view.
#[derive(Clone)]
pub(super) enum Amplification {
    One(u32),
    Many(Vec<u32>),
}

impl Amplification {
    fn new(mut mappings: impl ExactSizeIterator<Item = u32>) -> Self {
        if mappings.len() == 1 {
            Self::One(mappings.next().expect("one view mapping"))
        } else {
            Self::Many(mappings.collect())
        }
    }

    pub(super) fn as_slice(&self) -> &[u32] {
        match self {
            Self::One(mapping) => core::slice::from_ref(mapping),
            Self::Many(mappings) => mappings,
        }
    }
}

pub(super) struct RenderState {
    pass: RenderPassPlan<ImageView>,
    /// Indices of this pass's `BeginRendering` commands in the segment.
    begins: Vec<usize>,
    pub(super) pipeline: Option<RenderPipelineState>,
    /// The Vulkan pipeline last bound: the state's, or a variant of it.
    bound: Option<Pipeline>,
    /// `setObjectThreadgroupMemoryLength:atIndex:` lengths, by index.
    pub(super) object_threadgroup_lengths: Vec<(u32, u64)>,
    pub(super) vertex_buffers: [Option<VertexBinding>; 31],
    bound_vertex_buffers: [Option<VertexBuffer>; 31],
    index_buffer: Option<(Buffer, u64, IndexType)>,
    topology: Option<PrimitiveTopology>,
    restart: Option<bool>,
    pub(super) amplification: Amplification,
    pub(super) factors: Option<FactorBuffer>,
    pub(super) factor_scale: f32,
    pub(super) raster: RasterState,
    visibility: Visibility,
    samples: Vec<(CounterSample, PassBoundary)>,
    /// Timestamp queries the pass writes, reset before it begins.
    pub(super) timestamp_resets: Vec<(QueryPool, u32)>,
    /// What last accessed each color attachment in the pass instance.
    attachment_paths: [AttachmentPath; MAX_COLOR_ATTACHMENTS],
    /// The encoder's tile pipeline and threadgroup memory, and the pass's
    /// tile memory.
    pub(super) tile: super::tile::TileState,
    /// Where the pass's depth and stencil are, for pipelines that run the
    /// fragment tests themselves.
    pub(super) tests: super::fragment_tests::TestState,
    /// The first viewport's near and far depths.
    pub(super) depth_range: [f32; 2],
    /// The front and back faces' stencil reference values.
    pub(super) stencil_references: [u32; 2],
    /// The decodes of IOSurface planes before the pass, as (command index,
    /// color attachment), which its end narrows to what it drew.
    plane_decodes: Vec<(usize, usize)>,
}

impl RenderState {
    pub(super) fn pass(&self) -> &RenderPassPlan<ImageView> {
        &self.pass
    }

    /// The encoder's vertex buffers as a generated command sequence binds
    /// them, with their own strides.
    pub(super) fn sequence_vertex_buffers(
        &self,
    ) -> Result<[SequenceVertexBuffer; MAX_BUFFER_BINDINGS as usize]> {
        let mut buffers = [SequenceVertexBuffer::default(); MAX_BUFFER_BINDINGS as usize];
        for (buffer, bound) in buffers.iter_mut().zip(&self.vertex_buffers) {
            if let Some(bound) = bound {
                let size = bound.buffer.length - bound.offset;
                *buffer = SequenceVertexBuffer {
                    address: bound.buffer.address + bound.offset,
                    size: size.min(u32::MAX.into()) as u32,
                    stride: u32::try_from(bound.stride.unwrap_or(0))
                        .map_err(|_| Error::Invalid("vertex buffer stride"))?,
                };
            }
        }
        Ok(buffers)
    }
}

/// The Vulkan topology of a Metal primitive type.
pub(super) const fn primitive_topology(primitive: PrimitiveType) -> PrimitiveTopology {
    match primitive {
        PrimitiveType::Point => PrimitiveTopology::PointList,
        PrimitiveType::Line => PrimitiveTopology::LineList,
        PrimitiveType::LineStrip => PrimitiveTopology::LineStrip,
        PrimitiveType::Triangle => PrimitiveTopology::TriangleList,
        PrimitiveType::TriangleStrip => PrimitiveTopology::TriangleStrip,
    }
}

/// The active render encoder, borrowing only the encoder field so the
/// recorder and bindings stay available.
pub(super) fn render(encoder: &mut Encoder) -> Result<&mut RenderState> {
    match encoder {
        Encoder::Render(state) => Ok(state),
        _ => Err(Error::Invalid("no render command encoder is encoding")),
    }
}

/// The color attachments `state`'s fragment function accesses as storage
/// ([`StorageAttachment`](crate::metal::shader::StorageAttachment)), one bit
/// each.
pub(super) fn storage_attachments(state: &RenderPipelineState) -> u32 {
    state
        .interfaces
        .iter()
        .fold(0, |mask, interface| mask | interface.storage_attachments)
}

/// The color attachments `state`'s blend unit writes, one bit each.
pub(super) fn blended_attachments(state: &RenderPipelineState) -> u32 {
    let info = &state.plan.info;
    (0..MAX_COLOR_ATTACHMENTS)
        .filter(|&index| {
            info.color_formats.get(index).is_some_and(Option::is_some)
                && info
                    .color_blend
                    .get(index)
                    .is_some_and(|blend| !blend.write_mask.is_empty())
        })
        .fold(0, |mask, index| mask | 1 << index)
}

/// What last accessed a color attachment in the pass instance, which a draw
/// accessing it by the other path waits for.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum AttachmentPath {
    /// Nothing since the last barrier.
    #[default]
    None,
    /// The blend unit, or the instance's load operation.
    Blend,
    /// A fragment function's storage accesses, which follow each other in
    /// their interlock.
    Storage,
}

/// Orders the blend unit's and the load and store operations' accesses of
/// attachments against fragment functions' storage accesses of them, each
/// pixel's after its earlier ones.
const ATTACHMENT_PATHS: Barrier = Barrier {
    src_stages: PipelineStages::FRAGMENT_SHADER.union(PipelineStages::COLOR_ATTACHMENT_OUTPUT),
    src_access: Access::SHADER_WRITE.union(Access::COLOR_ATTACHMENT_WRITE),
    dst_stages: PipelineStages::FRAGMENT_SHADER.union(PipelineStages::COLOR_ATTACHMENT_OUTPUT),
    dst_access: Access::SHADER_READ
        .union(Access::SHADER_WRITE)
        .union(Access::COLOR_ATTACHMENT_READ)
        .union(Access::COLOR_ATTACHMENT_WRITE),
};

/// The most bytes of imageblock data a pass allocates.
const MAX_IMAGEBLOCK_MEMORY: u64 = 1 << 30;

impl CommandBuffer {
    /// A pass's imageblock data, when it has an `imageblockSampleLength`:
    /// whole tiles of samples, whose contents are undefined until written.
    fn pass_imageblock(
        &mut self,
        resolver: &mut dyn Resolver,
        pass: &RenderPassPlan<ImageView>,
    ) -> Result<PassImageblock> {
        let tile = pass.tile;
        if tile.imageblock_sample_length == 0 {
            return Ok(PassImageblock::default());
        }
        let pitch = pass.extent[0].next_multiple_of(tile.size[0]);
        let rows = pass.extent[1].next_multiple_of(tile.size[1]);
        let bytes = u64::from(tile.imageblock_sample_length)
            * u64::from(pitch)
            * u64::from(rows)
            * u64::from(pass.sample_count.max(1));
        if bytes > MAX_IMAGEBLOCK_MEMORY {
            return Err(Error::Unsupported("imageblock data of a pass this large"));
        }
        let upload = self.recorder.uploads.scratch(resolver, bytes.max(16))?;
        Ok(PassImageblock {
            stride: tile.imageblock_sample_length,
            pitch,
            address: upload.address,
        })
    }

    /// `renderCommandEncoderWithDescriptor:` for a pass whose attachment
    /// views the device resolved.
    pub fn begin_render(
        &mut self,
        resolver: &mut dyn Resolver,
        mut pass: RenderPassPlan<ImageView>,
        visibility_buffer: Option<(BufferRef, VisibilityResultType)>,
        samples: Vec<(CounterSample, PassBoundary)>,
    ) -> Result<()> {
        self.idle()?;
        ensure(
            pass.visibility_results == visibility_buffer.is_some(),
            "the visibility result buffer was not resolved",
        )?;
        let pass_samples = pass.samples()?;
        let imageblock = self.pass_imageblock(resolver, &pass)?;
        self.bindings.reset();
        self.bindings
            .set_pass(pass.inputs.textures, pass_samples, imageblock);
        // The pass's fragments mark the dirty tiles of one IOSurface's
        // working image, which they reach as its attachment's raw texels;
        // another's encode goes without them.
        let mut dirty = PassDirty::default();
        for (index, planes) in pass.inputs.extended_planes.iter_mut().enumerate() {
            let Some(planes) = planes.as_mut().filter(|planes| planes.stamp != 0) else {
                continue;
            };
            if dirty.stamp == 0 && pass.inputs.textures[index] == planes.storage {
                dirty = PassDirty {
                    stamp: planes.stamp,
                    attachment: index as u32,
                };
            } else {
                planes.stamp = 0;
            }
        }
        self.bindings.set_dirty(dirty);
        for &(sample, boundary) in &samples {
            if let (CounterSample::Statistic(destination), true) = (sample, boundary.starts()) {
                self.recorder.sample_statistics(resolver, destination)?;
            }
        }
        // A two-plane attachment's second plane clears before the pass,
        // which writes it as storage.
        let clears = pass.inputs.second_plane_clears;
        if clears.iter().any(Option::is_some) {
            self.recorder.access(super::blit::TRANSFER);
            for &(image, value) in clears.iter().flatten() {
                self.recorder
                    .push(Command::ClearColorImage { image, value });
            }
        }
        let plane_decodes = self.extended_planes(resolver, &pass, true, None, false)?;
        for (color, staged) in pass.color.iter().zip(&pass.inputs.staged_colors) {
            if let Some(staged) = staged
                && color
                    .as_ref()
                    .is_some_and(|color| color.load_op == AttachmentLoadOp::Load)
            {
                if let Some(rows) = staged.guest_rows {
                    let segment = self.recorder.segment();
                    if segment.commands.len() != 1
                        || !segment.signals.is_empty()
                        || segment.copies_guest_rows()
                    {
                        self.recorder.pause_statistics();
                        self.recorder.start_segment(Vec::new());
                        self.recorder.resume_statistics(resolver, false)?;
                    }
                    self.recorder.segment().guest_load = Some(rows);
                }
                self.recorder.access(super::blit::TRANSFER);
                self.recorder.push(Command::CopyBufferToImage {
                    src: staged.buffer,
                    dst: staged.image,
                    region: staged.region,
                });
            }
        }
        self.recorder.access(RENDER_PASS);
        self.recorder.pause_statistics();
        let begin = self.recorder.commands().len();
        self.recorder
            .push(Command::BeginRendering(rendering(&pass, true, true)));
        self.recorder.resume_statistics(resolver, false)?;
        let mut state = RenderState {
            pass,
            begins: alloc::vec![begin],
            pipeline: None,
            bound: None,
            object_threadgroup_lengths: Vec::new(),
            vertex_buffers: [None; 31],
            bound_vertex_buffers: [None; 31],
            index_buffer: None,
            topology: None,
            restart: None,
            amplification: Amplification::One(0),
            factors: None,
            factor_scale: 1.0,
            raster: RasterState::default(),
            visibility: Visibility {
                buffer: visibility_buffer.map(|(buffer, _)| buffer),
                accumulate: visibility_buffer
                    .is_some_and(|(_, kind)| kind == VisibilityResultType::Accumulate),
                ..Visibility::default()
            },
            samples,
            timestamp_resets: Vec::new(),
            attachment_paths: [AttachmentPath::Blend; MAX_COLOR_ATTACHMENTS],
            tile: super::tile::TileState::default(),
            tests: super::fragment_tests::TestState::default(),
            depth_range: [0.0, 1.0],
            stencil_references: [0, 0],
            plane_decodes,
        };
        for &(sample, boundary) in &state.samples {
            if let (CounterSample::Timestamp { pool, query }, true) = (sample, boundary.starts()) {
                state.timestamp_resets.push((pool, query));
                self.recorder.push(Command::WriteTimestamp {
                    pool,
                    query,
                    stage: PipelineStages::TOP_OF_PIPE,
                });
            }
        }
        self.default_render_state(&mut state);
        self.encoder = Encoder::Render(Box::new(state));
        Ok(())
    }

    /// Metal's initial encoder state, as dynamic state.
    fn default_render_state(&mut self, state: &mut RenderState) {
        let [width, height] = state.pass.extent;
        let viewport = Viewport {
            origin_x: 0.0,
            origin_y: 0.0,
            width: f64::from(width),
            height: f64::from(height),
            znear: 0.0,
            zfar: 1.0,
        };
        let features = self.features;
        let recorder = &mut self.recorder;
        recorder.push(Command::SetViewports(alloc::vec![flip(viewport)]));
        recorder.push(Command::SetScissors(alloc::vec![Rect {
            offset: [0, 0],
            extent: [width, height]
        }]));
        recorder.push(Command::SetCullMode(CullMode::EMPTY));
        recorder.push(Command::SetFrontFace(FrontFace::Clockwise));
        recorder.push(Command::SetDepthBias {
            enable: false,
            constant: 0.0,
            clamp: 0.0,
            slope: 0.0,
        });
        recorder.push(Command::SetDepthStencil(depth_stencil(
            &DepthStencilState::default(),
        )));
        recorder.push(Command::SetStencilReference { front: 0, back: 0 });
        recorder.push(Command::SetBlendConstants([0.0; 4]));
        if features.polygon_mode {
            recorder.push(Command::SetPolygonMode(PolygonMode::Fill));
        }
        if features.depth_clamp {
            recorder.push(Command::SetDepthClamp(false));
        }
        if features.sample_locations {
            let samples = SampleCounts::single(state.pass.sample_count).unwrap_or(SampleCounts::C1);
            let locations = state.pass.sample_locations.clone();
            recorder.push(Command::SetSampleLocations { samples, locations });
        }
    }

    pub(super) fn end_render(
        &mut self,
        resolver: &mut dyn Resolver,
        mut state: RenderState,
    ) -> Result<()> {
        state.pass.finish()?;
        end_query(&mut self.recorder, &mut state.visibility);
        for &(sample, boundary) in &state.samples {
            let stage = match boundary {
                PassBoundary::EndOfVertex => PipelineStages::VERTEX_SHADER,
                PassBoundary::EndOfFragment => PipelineStages::FRAGMENT_SHADER,
                PassBoundary::StartOfVertex | PassBoundary::StartOfFragment => continue,
            };
            if let CounterSample::Timestamp { pool, query } = sample {
                state.timestamp_resets.push((pool, query));
                self.recorder
                    .push(Command::WriteTimestamp { pool, query, stage });
            }
        }
        self.recorder.pause_statistics();
        // The store operations follow the functions' storage writes.
        if state.attachment_paths.contains(&AttachmentPath::Storage) {
            self.recorder.push(Command::RegionBarrier(ATTACHMENT_PATHS));
        }
        self.recorder.push(Command::EndRendering);
        self.recorder.resume_statistics(resolver, false)?;
        for &(sample, boundary) in &state.samples {
            if let (CounterSample::Statistic(destination), false) = (sample, boundary.starts()) {
                self.recorder.sample_statistics(resolver, destination)?;
            }
        }

        let last = state.begins.len() - 1;
        let commands = self.recorder.commands();
        // What the pass's work may have written; the IOSurface planes'
        // conversions cover only that (`extended_planes`).
        let (drawn, draws_only) = drawn_area(&commands[state.begins[0]..], state.pass.extent);
        for (instance, &index) in state.begins.iter().enumerate() {
            commands[index] =
                Command::BeginRendering(rendering(&state.pass, instance == 0, instance == last));
        }
        if !state.timestamp_resets.is_empty() {
            let resets =
                state
                    .timestamp_resets
                    .iter()
                    .map(|&(pool, first)| Command::ResetQueryPool {
                        pool,
                        first,
                        count: 1,
                    });
            let at = state.begins[0];
            commands.splice(at..at, resets.collect::<Vec<_>>());
        }
        self.average_samples(resolver, &state.pass)?;
        self.narrow_plane_decodes(resolver, &state, DrawnArea::bounds(&drawn))?;
        self.extended_planes(resolver, &state.pass, false, Some(&drawn), draws_only)?;
        let mut guest_store = None;
        for (color, staged) in state
            .pass
            .color
            .iter()
            .zip(&state.pass.inputs.staged_colors)
        {
            if let Some(staged) = staged
                && color
                    .as_ref()
                    .is_some_and(|color| color.store_action == StoreAction::Store)
            {
                guest_store = staged.guest_rows;
                self.recorder.access(super::blit::TRANSFER);
                self.recorder.push(Command::CopyImageToBuffer {
                    src: staged.image,
                    dst: staged.buffer,
                    region: staged.region,
                });
                // A CPU may read after the next event signal, before the
                // command buffer completes. Keep the transfer write in
                // the hazard tracker for later GPU work, too.
                self.recorder.push(Command::Barrier(Barrier {
                    src_stages: PipelineStages::ALL_TRANSFER,
                    src_access: Access::TRANSFER_WRITE,
                    dst_stages: PipelineStages::HOST,
                    dst_access: Access::HOST_READ,
                }));
            }
        }
        self.follow_depth(resolver, &state.pass)?;
        self.resolve_visibility(resolver, state.visibility)?;
        if let Some(rows) = guest_store {
            self.recorder.pause_statistics();
            self.recorder.segment().guest_store = Some(rows);
            // Guest events and completion belong to the continuation.
            self.recorder.start_segment(Vec::new());
            self.recorder.resume_statistics(resolver, false)?;
        }
        Ok(())
    }

    /// Converts each two-plane extended-range attachment's half floats
    /// ([`ExtendedPlanes`](crate::metal::pass::ExtendedPlanes)) with the
    /// texel kernels: before the pass, from an IOSurface's planes it loads;
    /// once its last instance has stored them, into the planes, or for
    /// another texture in place to the values its texels read.
    ///
    /// The end of a pass that loaded an IOSurface's planes converts only
    /// `drawn`, the disjoint rectangles its work may have written
    /// ([`drawn_area`]), one conversion each: every code survives a decode
    /// and encode, so the rest of the planes already holds what a whole
    /// encode would write. Answers the decodes' command indices, which the
    /// end narrows to the rectangles' bounds. Where the drawn part is all of
    /// it and only draws wrote it (`draws_only`), an attachment whose
    /// fragments marked its dirty tiles ([`ExtendedPlanes::stamp`]) encodes
    /// only those.
    fn extended_planes(
        &mut self,
        resolver: &mut dyn Resolver,
        pass: &RenderPassPlan<ImageView>,
        begins: bool,
        drawn: Option<&[[u32; 4]]>,
        draws_only: bool,
    ) -> Result<Vec<(usize, usize)>> {
        let mut decodes = Vec::new();
        for (index, planes) in pass.inputs.extended_planes.iter().enumerate() {
            let Some(planes) = planes else { continue };
            // A store action set after the pass began decides its end.
            let stores = pass.color[index].as_ref().is_some_and(|color| {
                !color.memoryless && color.store_op == AttachmentStoreOp::Store
            });
            if begins && !planes.loads || !begins && !stores {
                continue;
            }
            // The rectangles are disjoint: their conversions need no order.
            let whole = [None];
            let narrowed: Vec<_>;
            let areas: &[Option<[u32; 4]>] = match drawn.filter(|_| narrows(planes)) {
                Some(rects) => {
                    narrowed = rects.iter().copied().map(Some).collect();
                    &narrowed
                }
                None => &whole,
            };
            let [width, height] = planes.extent;
            let stamp = match (begins, drawn) {
                (false, Some([[0, 0, w, h]])) if draws_only && [*w, *h] == [width, height] => {
                    planes.stamp
                }
                _ => 0,
            };
            let mut ordered = false;
            for &area in areas {
                let Some((arguments, count)) = plane_conversion(planes, area, stamp)? else {
                    continue;
                };
                let upload = self.recorder.upload(
                    resolver,
                    &arguments.encode(),
                    TexelArguments::ALIGNMENT,
                )?;
                if !ordered {
                    self.recorder.access(Use::new(
                        PipelineStages::COMPUTE_SHADER,
                        Access::SHADER_READ.union(Access::SHADER_WRITE),
                    ));
                    ordered = true;
                }
                if begins {
                    decodes.push((self.recorder.commands().len(), index));
                }
                self.recorder.push(if begins {
                    Command::StoreTexels {
                        arguments: upload.address,
                        count,
                    }
                } else {
                    Command::LoadTexels {
                        arguments: upload.address,
                        count,
                    }
                });
            }
        }
        Ok(decodes)
    }

    /// Narrows the decodes before a pass that loaded IOSurface planes to
    /// `drawn`, the part its work may have read or written: the rest is
    /// neither read nor encoded back. A pass that drew nothing decodes
    /// nothing.
    fn narrow_plane_decodes(
        &mut self,
        resolver: &mut dyn Resolver,
        state: &RenderState,
        drawn: [u32; 4],
    ) -> Result<()> {
        for &(at, index) in &state.plane_decodes {
            let Some(planes) = state.pass.inputs.extended_planes[index]
                .as_ref()
                .filter(|planes| narrows(planes) && !planes.decodes_whole)
            else {
                continue;
            };
            let [width, height] = planes.extent;
            if drawn == [0, 0, width, height] {
                continue;
            }
            let command = match plane_conversion(planes, Some(drawn), 0)? {
                Some((arguments, count)) => {
                    let upload = self.recorder.upload(
                        resolver,
                        &arguments.encode(),
                        TexelArguments::ALIGNMENT,
                    )?;
                    Command::StoreTexels {
                        arguments: upload.address,
                        count,
                    }
                }
                // Nothing to decode: an empty dispatch keeps the indices.
                None => match self.recorder.commands()[at] {
                    Command::StoreTexels { arguments, .. } => Command::StoreTexels {
                        arguments,
                        count: 0,
                    },
                    _ => return Err(Error::Invalid("a pass's plane decode moved")),
                },
            };
            let slot = &mut self.recorder.commands()[at];
            ensure(
                matches!(slot, Command::StoreTexels { .. }),
                "a pass's plane decode moved",
            )?;
            *slot = command;
        }
        Ok(())
    }

    /// Resolves each attachment rendering through its raw texels whose
    /// store action resolves, with the average samples kernel, once the
    /// pass's last instance has stored it.
    fn average_samples(
        &mut self,
        resolver: &mut dyn Resolver,
        pass: &RenderPassPlan<ImageView>,
    ) -> Result<()> {
        for (index, color) in pass.color.iter().enumerate() {
            let Some(_) = color.as_ref().filter(|color| {
                color.averaged && color.resolve.is_some() && color.store_action.resolves()
            }) else {
                continue;
            };
            let format = pass.color_formats[index]
                .and_then(raw_rendered)
                .ok_or(Error::Invalid(
                    "an averaged attachment renders a raw format",
                ))?;
            let (source, destination) = (pass.inputs.textures[index], pass.inputs.averages[index]);
            ensure(
                source != ResourceId::NULL && destination != ResourceId::NULL,
                "an averaged attachment and its resolve texture have raw texel views",
            )?;
            let [width, height] = pass.extent;
            let count = width
                .checked_mul(height)
                .and_then(|texels| texels.checked_mul(pass.layer_count))
                .ok_or(Error::Unsupported("resolves of 4 Gi texels"))?;
            if count == 0 {
                continue;
            }
            let arguments = AverageArguments {
                source: source.0,
                destination: destination.0,
                count,
                format,
                width,
                height,
                samples: pass.sample_count,
                depth: 0,
            };
            let upload =
                self.recorder
                    .upload(resolver, &arguments.encode(), AverageArguments::ALIGNMENT)?;
            self.recorder.access(Use::new(
                PipelineStages::COMPUTE_SHADER,
                Access::SHADER_READ.union(Access::SHADER_WRITE),
            ));
            self.recorder.push(Command::AverageSamples {
                arguments: upload.address,
                count,
            });
        }
        Ok(())
    }

    /// Resolves a stencil that follows a min or max depth resolve
    /// ([`RenderingAttachment::follows_depth`]) once the pass's last
    /// instance has stored its samples: the average samples kernel picks
    /// each texel's stencil from copies of the depth and stencil samples
    /// into a copy of the resolve texture's stencil, which then fills it.
    fn follow_depth(
        &mut self,
        resolver: &mut dyn Resolver,
        pass: &RenderPassPlan<ImageView>,
    ) -> Result<()> {
        let Some(followed) = follows_depth(pass) else {
            return Ok(());
        };
        let Some((mode, _)) = pass.stencil.as_ref().and_then(|stencil| stencil.resolve) else {
            return Ok(());
        };
        let [width, height] = pass.extent;
        let count = width
            .checked_mul(height)
            .and_then(|texels| texels.checked_mul(pass.layer_count))
            .ok_or(Error::Unsupported("resolves of 4 Gi texels"))?;
        if count == 0 {
            return Ok(());
        }
        let greatest = if mode == ResolveModes::MAX {
            AverageArguments::FOLLOW_MAX
        } else {
            0
        };
        let arguments = AverageArguments {
            source: followed.stencil.id.0,
            destination: followed.resolve.id.0,
            count,
            format: AverageArguments::FOLLOW_DEPTH | greatest | followed.depth.encoding,
            width,
            height,
            samples: pass.sample_count,
            depth: followed.depth.id.0,
        };
        let upload =
            self.recorder
                .upload(resolver, &arguments.encode(), AverageArguments::ALIGNMENT)?;
        let transfer = Use::new(
            PipelineStages::ALL_TRANSFER,
            Access::TRANSFER_READ.union(Access::TRANSFER_WRITE),
        );
        // The stores of the pass's last instance precede the copies.
        self.recorder.access(transfer);
        fragment_tests::barrier(
            &mut self.recorder,
            fragment_tests::ATTACHMENT,
            fragment_tests::TRANSFER,
        );
        fragment_tests::copy_images(
            &mut self.recorder,
            &[followed.depth, followed.stencil],
            false,
        );
        self.recorder.access(Use::new(
            PipelineStages::COMPUTE_SHADER,
            Access::SHADER_READ.union(Access::SHADER_WRITE),
        ));
        self.recorder.push(Command::AverageSamples {
            arguments: upload.address,
            count,
        });
        self.recorder.access(transfer);
        fragment_tests::copy_images(&mut self.recorder, &[followed.resolve], true);
        // Later passes may attach the resolve texture.
        self.recorder.full_barrier();
        // The kernel replaced the pipeline and push constants.
        self.bindings.reset();
        Ok(())
    }

    /// Writes each used offset's sample count (or nonzero flag) into the
    /// visibility result buffer.
    fn resolve_visibility(
        &mut self,
        resolver: &mut dyn Resolver,
        visibility: Visibility,
    ) -> Result<()> {
        let (Some(buffer), false) = (visibility.buffer, visibility.queries.is_empty()) else {
            return Ok(());
        };
        let mut offsets: Vec<u64> = visibility.queries.iter().map(|query| query.2).collect();
        offsets.sort_unstable();
        offsets.dedup();
        for &offset in &offsets {
            ensure(
                offset.is_multiple_of(8) && offset + 8 <= buffer.length,
                "visibility result offset",
            )?;
        }
        self.recorder.access(Use::new(
            PipelineStages::ALL_TRANSFER,
            Access::TRANSFER_READ | Access::TRANSFER_WRITE,
        ));
        if offsets.len() == visibility.queries.len() && !visibility.accumulate {
            for &(pool, query, offset) in &visibility.queries {
                let (dst, offset) = (buffer.buffer, buffer.at(offset, 8)?);
                self.recorder.push(Command::CopyQueryResults {
                    pool,
                    first: query,
                    count: 1,
                    dst,
                    offset,
                    stride: 8,
                    wait: true,
                });
            }
            return Ok(());
        }
        // Gather each offset's results contiguously, after what the buffer
        // holds there for an accumulating pass, then sum them.
        let held = if visibility.accumulate {
            offsets.len()
        } else {
            0
        };
        let scratch = self.recorder.upload(
            resolver,
            &alloc::vec![0; (visibility.queries.len() + held) * 8],
            8,
        )?;
        let mut runs = Vec::with_capacity(offsets.len());
        let mut next = 0u32;
        for &offset in &offsets {
            let first = next;
            if visibility.accumulate {
                self.recorder.push(Command::CopyBuffer {
                    src: buffer.buffer,
                    dst: scratch.buffer,
                    region: BufferCopy {
                        src_offset: buffer.at(offset, 8)?,
                        dst_offset: scratch.offset + u64::from(next) * 8,
                        size: 8,
                    },
                });
                next += 1;
            }
            for &(pool, query, _) in visibility.queries.iter().filter(|query| query.2 == offset) {
                let at = scratch.offset + u64::from(next) * 8;
                let dst = scratch.buffer;
                self.recorder.push(Command::CopyQueryResults {
                    pool,
                    first: query,
                    count: 1,
                    dst,
                    offset: at,
                    stride: 8,
                    wait: true,
                });
                next += 1;
            }
            runs.push((first, next - first, buffer.address + offset));
        }
        let count = runs.len() as u32;
        let arguments = AccumulateArguments {
            results: scratch.address,
            stride: 1,
            runs,
        };
        let arguments = self.recorder.upload(
            resolver,
            &arguments.encode(),
            AccumulateArguments::ALIGNMENT,
        )?;
        self.recorder.access(Use::new(
            PipelineStages::COMPUTE_SHADER,
            Access::SHADER_READ | Access::SHADER_WRITE,
        ));
        self.recorder.push(Command::AccumulateResults {
            arguments: arguments.address,
            runs: count,
        });
        // The kernel replaced the pipeline and push constants.
        self.bindings.reset();
        Ok(())
    }
}

fn end_query(recorder: &mut Recorder, visibility: &mut Visibility) {
    if let Some((pool, query)) = visibility.active.take() {
        recorder.push(Command::EndQuery { pool, query });
    }
}

/// The most rectangles a pass's drawn area keeps apart; past them the two
/// whose union adds the least merge.
const DRAWN_RECTS: usize = 8;

/// The part of a pass's `extent` its work may have read or written, as
/// disjoint `[x, y, width, height]` rectangles, from the pass's commands
/// from its first `BeginRendering` on: each draw's scissor rectangles, and
/// for filled triangles also its viewports, which bound every fragment,
/// blend and storage access of theirs (points, lines and mesh draws may
/// reach past the viewport by their width). Anything but a draw inside the
/// pass (a tile dispatch, a copy, generated commands) may reach any pixel,
/// so the whole extent; nothing drawn is none. Answers too whether only
/// draws wrote it.
fn drawn_area(commands: &[Command], extent: [u32; 2]) -> (Vec<[u32; 4]>, bool) {
    let full = (alloc::vec![[0, 0, extent[0], extent[1]]], false);
    let whole = Rect {
        offset: [0, 0],
        extent,
    };
    let mut scissors = alloc::vec![whole];
    // The viewports' bounds as [x0, y0, x1, y1], and whether draws keep
    // within them.
    let mut viewports = [0, 0, extent[0], extent[1]];
    let mut topology = None;
    let mut filled = true;
    let mut area = DrawnArea::default();
    for command in commands {
        let clipped = match command {
            Command::SetScissors(rects) => {
                scissors.clone_from(rects);
                continue;
            }
            Command::SetViewports(set) => {
                viewports = viewport_bounds(set, extent);
                continue;
            }
            Command::SetPrimitiveTopology(set) => {
                topology = Some(*set);
                continue;
            }
            Command::SetPolygonMode(mode) => {
                filled = *mode == PolygonMode::Fill;
                continue;
            }
            Command::Draw { .. }
            | Command::DrawIndexed { .. }
            | Command::DrawIndirect { .. }
            | Command::DrawIndexedIndirect { .. } => {
                filled
                    && matches!(
                        topology,
                        Some(
                            PrimitiveTopology::TriangleList
                                | PrimitiveTopology::TriangleStrip
                                | PrimitiveTopology::PatchList
                        )
                    )
            }
            Command::DrawMeshTasks(_) | Command::DrawMeshTasksIndirect { .. } => false,
            // Order, state and queries touch no attachment.
            Command::Barrier(_)
            | Command::RegionBarrier(_)
            | Command::BeginRendering(_)
            | Command::EndRendering
            | Command::BindPipeline { .. }
            | Command::PushConstants(_)
            | Command::SetDepthBias { .. }
            | Command::SetBlendConstants(_)
            | Command::SetStencilReference { .. }
            | Command::SetDepthStencil(_)
            | Command::SetCullMode(_)
            | Command::SetFrontFace(_)
            | Command::SetPrimitiveRestart(_)
            | Command::SetRasterizerDiscard(_)
            | Command::SetDepthClamp(_)
            | Command::SetSampleLocations { .. }
            | Command::BindVertexBuffers { .. }
            | Command::BindIndexBuffer { .. }
            | Command::ResetQueryPool { .. }
            | Command::BeginQuery { .. }
            | Command::EndQuery { .. }
            | Command::WriteTimestamp { .. }
            | Command::BeginConditionalRendering { .. }
            | Command::EndConditionalRendering => continue,
            _ => return full,
        };
        let bounds = if clipped {
            viewports
        } else {
            [0, 0, extent[0], extent[1]]
        };
        for rect in &scissors {
            let clip = |axis: usize, end: i64| {
                end.clamp(i64::from(bounds[axis]), i64::from(bounds[axis + 2])) as u32
            };
            let start = [0, 1].map(|axis| clip(axis, i64::from(rect.offset[axis])));
            let end = [0, 1].map(|axis| {
                clip(
                    axis,
                    i64::from(rect.offset[axis]) + i64::from(rect.extent[axis]),
                )
            });
            area.add([start[0], start[1], end[0], end[1]]);
        }
    }
    let rects = area
        .rects
        .iter()
        .map(|&[x0, y0, x1, y1]| [x0, y0, x1 - x0, y1 - y0])
        .collect();
    (rects, true)
}

/// The pixels the viewports `set` cover, within `extent`, as
/// `[x0, y0, x1, y1]`: every pixel a sample of theirs lies in.
fn viewport_bounds(set: &[VkViewport], extent: [u32; 2]) -> [u32; 4] {
    let mut low = [f32::INFINITY; 2];
    let mut high = [f32::NEG_INFINITY; 2];
    for viewport in set {
        // Flipped viewports have negative heights.
        let xs = [viewport.x, viewport.x + viewport.width];
        let ys = [viewport.y, viewport.y + viewport.height];
        for (axis, ends) in [xs, ys].into_iter().enumerate() {
            low[axis] = low[axis].min(ends[0].min(ends[1]));
            high[axis] = high[axis].max(ends[0].max(ends[1]));
        }
    }
    // Not a number reaches anywhere.
    let side = |axis: usize, value: f32, up: bool| {
        let limit = extent[axis] as f32;
        if value.is_nan() {
            return if up { extent[axis] } else { 0 };
        }
        // Clamped first, the truncation is the floor (core has no rounding).
        let clamped = value.clamp(0.0, limit);
        let floor = clamped as u32;
        if up && (floor as f32) < clamped {
            floor + 1
        } else {
            floor
        }
    };
    [
        side(0, low[0], false),
        side(1, low[1], false),
        side(0, high[0], true),
        side(1, high[1], true),
    ]
}

/// Disjoint rectangles, `[x0, y0, x1, y1]`, covering everything added.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct DrawnArea {
    rects: Vec<[u32; 4]>,
}

impl DrawnArea {
    fn add(&mut self, rect: [u32; 4]) {
        if rect[0] >= rect[2] || rect[1] >= rect[3] {
            return;
        }
        let mut rect = rect;
        // A rectangle overlapping kept ones takes them in.
        while let Some(at) = self.rects.iter().position(|&kept| overlap(kept, rect)) {
            rect = union(self.rects.swap_remove(at), rect);
        }
        self.rects.push(rect);
        while self.rects.len() > DRAWN_RECTS {
            let mut best = (u64::MAX, 0, 1);
            for a in 0..self.rects.len() {
                for b in a + 1..self.rects.len() {
                    let (x, y) = (self.rects[a], self.rects[b]);
                    let grown = area(union(x, y)) - area(x) - area(y);
                    if grown < best.0 {
                        best = (grown, a, b);
                    }
                }
            }
            let merged = union(self.rects[best.1], self.rects.swap_remove(best.2));
            self.rects.swap_remove(best.1);
            self.add(merged);
        }
    }

    /// Their bounds as `[x, y, width, height]`, zero for none.
    fn bounds(rects: &[[u32; 4]]) -> [u32; 4] {
        let Some(first) = rects.first() else {
            return [0; 4];
        };
        let to_corners = |&[x, y, width, height]: &[u32; 4]| [x, y, x + width, y + height];
        let [x0, y0, x1, y1] = rects.iter().map(to_corners).fold(to_corners(first), union);
        [x0, y0, x1 - x0, y1 - y0]
    }
}

fn overlap(a: [u32; 4], b: [u32; 4]) -> bool {
    a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
}

fn union(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

fn area(rect: [u32; 4]) -> u64 {
    u64::from(rect[2] - rect[0]) * u64::from(rect[3] - rect[1])
}

/// Whether a pass converts only what it drew of `planes`
/// ([`ExtendedPlanes::narrows`]): texels outside the drawn part keep their
/// codes.
fn narrows(planes: &ExtendedPlanes) -> bool {
    planes.narrows
}

/// The texel kernels' arguments converting `planes` over `area`
/// (`[x, y, width, height]`, all of the extent for `None`), and the threads;
/// `None` for none. With a pass's `stamp` the area is the whole extent and
/// a workgroup converts each tile its fragments marked
/// ([`TexelArguments::stamp`]).
fn plane_conversion(
    planes: &ExtendedPlanes,
    area: Option<[u32; 4]>,
    stamp: u32,
) -> Result<Option<(TexelArguments, u32)>> {
    let [x, y, width, height] = area.unwrap_or([0, 0, planes.extent[0], planes.extent[1]]);
    let tile = TexelArguments::DIRTY_TILE;
    let count = if stamp != 0 {
        width.div_ceil(tile) * height.div_ceil(tile) * TexelArguments::WORKGROUP_SIZE
    } else {
        width
            .checked_mul(height)
            .and_then(|texels| texels.checked_mul(planes.layers))
            .ok_or(Error::Unsupported("conversions of 4 Gi texels"))?
    };
    if count == 0 {
        return Ok(None);
    }
    // The planes' addresses are of the area's first texel.
    let at = |base: u64, row: u32, bytes: u64| {
        if base == 0 {
            0
        } else {
            base + u64::from(y) * u64::from(row) + u64::from(x) * bytes
        }
    };
    Ok(Some((
        TexelArguments {
            buffer: at(planes.color, planes.color_bytes_per_row, 4),
            count,
            encoding: planes.encoding,
            image: planes.storage.0,
            width,
            height,
            bytes_per_row: planes.color_bytes_per_row,
            origin: [x, y, 0],
            alpha: at(planes.alpha, planes.alpha_bytes_per_row, 1),
            alpha_bytes_per_row: planes.alpha_bytes_per_row,
            mirror: planes.mirror,
            stamp,
            ..TexelArguments::default()
        },
        count,
    )))
}

/// The copies through which the pass's stencil resolve follows its depth
/// resolve, when its store action resolves.
fn follows_depth(pass: &RenderPassPlan<ImageView>) -> Option<FollowedStencil> {
    pass.stencil
        .as_ref()
        .filter(|stencil| {
            stencil.follows_depth && stencil.resolve.is_some() && stencil.store_action.resolves()
        })
        .and(pass.inputs.follows_depth)
}

/// The `VkRenderingInfo` of one instance of a pass: only the first loads
/// as Metal asked and only the last stores and resolves as Metal asked.
fn rendering(pass: &RenderPassPlan<ImageView>, first: bool, last: bool) -> Rendering {
    // A stencil resolve following the depth resolve reads the stored depth
    // and stencil samples once the pass ends.
    let follows = follows_depth(pass).is_some();
    let attachment =
        |attachment: &RenderingAttachment<ImageView>, depth_stencil: bool| Attachment {
            view: attachment.view.texture,
            // The average samples kernel resolves an averaged attachment and a
            // stencil following the depth resolve.
            resolve: if last && !attachment.averaged && !attachment.follows_depth {
                attachment.resolve.map(|(mode, view)| (mode, view.texture))
            } else {
                None
            },
            load_op: if first {
                attachment.load_op
            } else {
                AttachmentLoadOp::Load
            },
            store_op: if last && !(follows && depth_stencil) {
                attachment.store_op
            } else {
                AttachmentStoreOp::Store
            },
            clear: attachment.clear,
        };
    Rendering {
        extent: pass.extent,
        layer_count: pass.layer_count,
        color: pass
            .color
            .iter()
            .map(|color| color.as_ref().map(|color| attachment(color, false)))
            .collect(),
        depth: pass.depth.as_ref().map(|depth| attachment(depth, true)),
        stencil: pass
            .stencil
            .as_ref()
            .map(|stencil| attachment(stencil, true)),
    }
}

fn flip(viewport: Viewport) -> VkViewport {
    VkViewport {
        x: viewport.origin_x as f32,
        y: (viewport.origin_y + viewport.height) as f32,
        width: viewport.width as f32,
        height: -viewport.height as f32,
        min_depth: viewport.znear as f32,
        max_depth: viewport.zfar as f32,
    }
}

pub(super) fn depth_stencil(state: &DepthStencilState) -> DepthStencil {
    let face = |face: &crate::metal::pipeline::StencilFace| {
        (
            face.fail_op,
            face.pass_op,
            face.depth_fail_op,
            face.compare_op,
            face.compare_mask,
            face.write_mask,
        )
    };
    DepthStencil {
        depth_test: state.depth_test_enable,
        depth_write: state.depth_write_enable,
        depth_compare: state.depth_compare_op,
        stencil_test: state.stencil_test_enable,
        faces: [face(&state.front), face(&state.back)],
    }
}

/// Render encoder operations.
impl CommandBuffer {
    pub fn set_render_pipeline_state(&mut self, state: &RenderPipelineState) -> Result<()> {
        let render = render(&mut self.encoder)?;
        let plan = &state.plan;
        let pass = &render.pass;
        ensure(
            plan.sample_count == pass.sample_count,
            "pipeline and pass sample counts differ",
        )?;
        ensure(
            state
                .interfaces
                .iter()
                .all(|interface| interface.imageblock_bytes <= pass.tile.imageblock_sample_length),
            "the pipeline's imageblock data exceeds the pass's imageblockSampleLength",
        )?;
        let matches = |pipeline: Option<_>, pass: Option<_>| pipeline.is_none() || pipeline == pass;
        ensure(
            (0..MAX_COLOR_ATTACHMENTS)
                .all(|index| matches(plan.color_formats[index], pass.color_formats[index])),
            "pipeline color formats differ from the pass",
        )?;
        ensure(
            matches(plan.depth_format, pass.depth_format),
            "pipeline depth format differs from the pass",
        )?;
        ensure(
            matches(plan.stencil_format, pass.stencil_format),
            "pipeline stencil format differs from the pass",
        )?;
        let storage = storage_attachments(state);
        if (0..MAX_COLOR_ATTACHMENTS).any(|index| {
            storage & 1 << index != 0 && pass.inputs.textures[index] == ResourceId::NULL
        }) {
            return Err(Error::Unsupported(
                "framebuffer reads of an attachment without a raw texel view or copy",
            ));
        }
        let rebind = render
            .pipeline
            .as_ref()
            .is_none_or(|bound| (bound.id, bound.pipeline) != (state.id, state.pipeline));
        let views = plan.amplification_count;
        if render.amplification.as_slice().len() != views as usize {
            // Each view maps to itself until the encoder sets mappings.
            render.amplification = Amplification::new((0..views).map(|view| view << 16 | view));
        }
        render.pipeline = Some(state.clone());
        if rebind {
            if let Some(pipeline) = state.pipeline {
                render.topology = None;
                render.restart = None;
                // An outdated pipeline's draws bind a variant instead.
                if !state.outdated {
                    render.bound = Some(pipeline);
                    self.recorder.push(Command::BindPipeline {
                        bind_point: PipelineBindPoint::Graphics,
                        pipeline,
                    });
                }
            }
            self.recorder
                .push(Command::SetRasterizerDiscard(!plan.rasterization_enabled));
        }
        self.bound(BoundPipeline::Render(state.id));
        Ok(())
    }

    pub fn set_render_buffer(
        &mut self,
        stage: Stage,
        buffer: Option<BufferRef>,
        offset: u64,
        index: u32,
    ) -> Result<()> {
        let render = render(&mut self.encoder)?;
        if stage == Stage::Vertex {
            render.vertex_buffers[buffer_slot(index)?] = buffer.map(|buffer| VertexBinding {
                buffer,
                offset,
                stride: None,
            });
        }
        self.bindings.set_buffer(stage, buffer, offset, index)
    }

    /// `setVertexBuffer:offset:attributeStride:atIndex:`, for layouts with a
    /// dynamic stride.
    pub fn set_vertex_buffer_with_stride(
        &mut self,
        buffer: BufferRef,
        offset: u64,
        stride: u64,
        index: u32,
    ) -> Result<()> {
        let render = render(&mut self.encoder)?;
        render.vertex_buffers[buffer_slot(index)?] = Some(VertexBinding {
            buffer,
            offset,
            stride: Some(stride),
        });
        self.bindings
            .set_buffer(Stage::Vertex, Some(buffer), offset, index)
    }

    /// `setVertexBufferOffset:attributeStride:atIndex:`: the bound buffer's
    /// offset and dynamic stride.
    pub fn set_vertex_buffer_offset_with_stride(
        &mut self,
        offset: u64,
        stride: u64,
        index: u32,
    ) -> Result<()> {
        let render = render(&mut self.encoder)?;
        let binding = render.vertex_buffers[buffer_slot(index)?]
            .as_mut()
            .ok_or(Error::Invalid("no vertex buffer is bound at the index"))?;
        binding.offset = offset;
        binding.stride = Some(stride);
        self.bindings
            .set_buffer_offset(Stage::Vertex, offset, index)
    }

    pub fn set_render_buffer_offset(
        &mut self,
        stage: Stage,
        offset: u64,
        index: u32,
    ) -> Result<()> {
        let render = render(&mut self.encoder)?;
        if stage == Stage::Vertex
            && let Some(binding) = &mut render.vertex_buffers[buffer_slot(index)?]
        {
            binding.offset = offset;
        }
        self.bindings.set_buffer_offset(stage, offset, index)
    }

    /// `set*Bytes:length:atIndex:`: the bytes are copied into upload memory.
    pub fn set_render_bytes(
        &mut self,
        resolver: &mut dyn Resolver,
        stage: Stage,
        bytes: &[u8],
        index: u32,
    ) -> Result<()> {
        render(&mut self.encoder)?;
        let buffer = self.upload_bytes(resolver, bytes)?;
        self.set_render_buffer(stage, Some(buffer), 0, index)
    }

    pub fn set_render_texture(
        &mut self,
        stage: Stage,
        texture: ResourceId,
        index: u32,
    ) -> Result<()> {
        render(&mut self.encoder)?;
        self.bindings.set_texture(stage, texture, index)
    }

    pub fn set_render_sampler(
        &mut self,
        resolver: &mut dyn Resolver,
        stage: Stage,
        sampler: ResourceId,
        clamp: Option<(f32, f32)>,
        index: u32,
    ) -> Result<()> {
        render(&mut self.encoder)?;
        let sampler = match clamp {
            Some((min, max)) => resolver.clamped_sampler(sampler, min, max)?,
            None => sampler,
        };
        self.bindings.set_sampler(stage, sampler, index)
    }

    pub fn set_viewports(&mut self, viewports: &[Viewport]) -> Result<()> {
        let render = render(&mut self.encoder)?;
        ensure(
            (1..=16).contains(&viewports.len()),
            "between 1 and 16 viewports",
        )?;
        render.depth_range = [viewports[0].znear as f32, viewports[0].zfar as f32];
        self.recorder.push(Command::SetViewports(
            viewports.iter().copied().map(flip).collect(),
        ));
        Ok(())
    }

    pub fn set_scissor_rects(&mut self, rects: &[ScissorRect]) -> Result<()> {
        let [width, height] = render(&mut self.encoder)?.pass.extent;
        ensure(
            (1..=16).contains(&rects.len()),
            "between 1 and 16 scissor rectangles",
        )?;
        let mut scissors = Vec::with_capacity(rects.len());
        for rect in rects {
            let inside = rect
                .x
                .checked_add(rect.width)
                .is_some_and(|right| right <= u64::from(width))
                && rect
                    .y
                    .checked_add(rect.height)
                    .is_some_and(|bottom| bottom <= u64::from(height));
            ensure(inside, "scissor rectangles lie within the render target")?;
            scissors.push(Rect {
                offset: [rect.x as i32, rect.y as i32],
                extent: [rect.width as u32, rect.height as u32],
            });
        }
        self.recorder.push(Command::SetScissors(scissors));
        Ok(())
    }

    pub fn set_cull_mode(&mut self, mode: MtlCullMode) -> Result<()> {
        render(&mut self.encoder)?.raster.cull_mode = mode;
        let mode = match mode {
            MtlCullMode::None => CullMode::EMPTY,
            MtlCullMode::Front => CullMode::FRONT,
            MtlCullMode::Back => CullMode::BACK,
        };
        self.recorder.push(Command::SetCullMode(mode));
        Ok(())
    }

    pub fn set_front_facing_winding(&mut self, winding: Winding) -> Result<()> {
        render(&mut self.encoder)?.raster.winding = winding;
        let face = match winding {
            Winding::Clockwise => FrontFace::Clockwise,
            Winding::CounterClockwise => FrontFace::CounterClockwise,
        };
        self.recorder.push(Command::SetFrontFace(face));
        Ok(())
    }

    pub fn set_depth_bias(&mut self, constant: f32, slope: f32, clamp: f32) -> Result<()> {
        render(&mut self.encoder)?.raster.depth_bias = [constant, slope, clamp];
        let enable = constant != 0.0 || slope != 0.0;
        self.recorder.push(Command::SetDepthBias {
            enable,
            constant,
            clamp,
            slope,
        });
        Ok(())
    }

    pub fn set_depth_stencil_state(&mut self, state: Option<&DepthStencilState>) -> Result<()> {
        let state = state.copied().unwrap_or_default();
        let render = render(&mut self.encoder)?;
        render.raster.depth_stencil = state;
        // A pipeline testing fragments itself keeps the tests off; its draws
        // read the state from their table.
        if !render.tests.untested() {
            self.recorder
                .push(Command::SetDepthStencil(depth_stencil(&state)));
        }
        Ok(())
    }

    /// `setObjectThreadgroupMemoryLength:atIndex:`.
    pub fn set_object_threadgroup_memory_length(&mut self, length: u64, index: u32) -> Result<()> {
        let lengths = &mut render(&mut self.encoder)?.object_threadgroup_lengths;
        lengths.retain(|&(slot, _)| slot != index);
        lengths.push((index, length));
        lengths.sort_unstable();
        Ok(())
    }

    pub fn set_stencil_reference_values(&mut self, front: u32, back: u32) -> Result<()> {
        render(&mut self.encoder)?.stencil_references = [front, back];
        self.recorder
            .push(Command::SetStencilReference { front, back });
        Ok(())
    }

    pub fn set_blend_color(&mut self, color: [f32; 4]) -> Result<()> {
        render(&mut self.encoder)?;
        self.recorder.push(Command::SetBlendConstants(color));
        self.bindings.set_blend_color(color);
        Ok(())
    }

    pub fn set_triangle_fill_mode(&mut self, mode: TriangleFillMode) -> Result<()> {
        let render = render(&mut self.encoder)?;
        match (mode, self.features.polygon_mode) {
            (TriangleFillMode::Fill, false) => {}
            (TriangleFillMode::Lines, false) => {
                return Err(Error::Unsupported("triangle line fill"));
            }
            (mode, true) => {
                let polygon = if mode == TriangleFillMode::Lines {
                    PolygonMode::Line
                } else {
                    PolygonMode::Fill
                };
                self.recorder.push(Command::SetPolygonMode(polygon));
            }
        }
        render.raster.fill_mode = mode;
        Ok(())
    }

    pub fn set_depth_clip_mode(&mut self, mode: DepthClipMode) -> Result<()> {
        let render = render(&mut self.encoder)?;
        match (mode, self.features.depth_clamp) {
            (DepthClipMode::Clip, false) => {}
            (DepthClipMode::Clamp, false) => return Err(Error::Unsupported("depth clamping")),
            (mode, true) => {
                self.recorder
                    .push(Command::SetDepthClamp(mode == DepthClipMode::Clamp));
            }
        }
        render.raster.clip_mode = mode;
        Ok(())
    }

    /// Sets each part of `raster` that differs from the encoder's.
    pub(super) fn set_raster_state(&mut self, raster: &RasterState) -> Result<()> {
        let current = render(&mut self.encoder)?.raster;
        if current.cull_mode != raster.cull_mode {
            self.set_cull_mode(raster.cull_mode)?;
        }
        if current.winding != raster.winding {
            self.set_front_facing_winding(raster.winding)?;
        }
        if current.depth_bias != raster.depth_bias {
            let [constant, slope, clamp] = raster.depth_bias;
            self.set_depth_bias(constant, slope, clamp)?;
        }
        if current.depth_stencil != raster.depth_stencil {
            self.set_depth_stencil_state(Some(&raster.depth_stencil))?;
        }
        if current.fill_mode != raster.fill_mode {
            self.set_triangle_fill_mode(raster.fill_mode)?;
        }
        if current.clip_mode != raster.clip_mode {
            self.set_depth_clip_mode(raster.clip_mode)?;
        }
        Ok(())
    }

    /// `setVertexAmplificationCount:viewMappings:`; each mapping is
    /// (viewport offset, render target array index offset).
    pub fn set_vertex_amplification(&mut self, mappings: &[(u32, u32)]) -> Result<()> {
        let render = render(&mut self.encoder)?;
        ensure(
            !mappings.is_empty(),
            "the amplification count is at least one",
        )?;
        ensure(
            mappings
                .iter()
                .all(|&(viewport, layer)| viewport <= 0xffff && layer <= 0xffff),
            "view mapping offsets",
        )?;
        render.amplification = Amplification::new(
            mappings
                .iter()
                .map(|&(viewport, layer)| layer << 16 | viewport),
        );
        Ok(())
    }

    pub fn set_visibility_result_mode(
        &mut self,
        mode: VisibilityResultMode,
        offset: u64,
    ) -> Result<()> {
        let render = render(&mut self.encoder)?;
        ensure(
            mode == VisibilityResultMode::Disabled || render.visibility.buffer.is_some(),
            "visibility results need a visibility result buffer",
        )?;
        end_query(&mut self.recorder, &mut render.visibility);
        render.visibility.mode = (mode != VisibilityResultMode::Disabled).then_some((mode, offset));
        Ok(())
    }

    /// `set{Color,Depth,Stencil}StoreAction:` for a deferred action.
    pub fn set_store_action(&mut self, slot: AttachmentSlot, action: StoreAction) -> Result<()> {
        render(&mut self.encoder)?
            .pass
            .set_store_action(slot, action)
    }

    /// `set{Color,Depth,Stencil}StoreActionOptions:`.
    pub fn set_store_action_options(
        &mut self,
        slot: AttachmentSlot,
        options: StoreActionOptions,
    ) -> Result<()> {
        render(&mut self.encoder)?
            .pass
            .set_store_action_options(slot, options)
    }

    /// `memoryBarrierWithScope:afterStages:beforeStages:` (and the
    /// resource form, which orders the same stages): the pass instance ends
    /// and resumes behind a barrier.
    pub fn render_memory_barrier(
        &mut self,
        resolver: &mut dyn Resolver,
        scope: BarrierScope,
        after: RenderStages,
        before: RenderStages,
    ) -> Result<()> {
        render(&mut self.encoder)?;
        ensure(!scope.is_empty(), "a barrier names a scope")?;
        let (src_stages, dst_stages) = (render_stages(after), render_stages(before));
        ensure(
            !src_stages.is_empty() && !dst_stages.is_empty(),
            "barriers name stages Vulkan has",
        )?;
        self.outside_pass(resolver, |recorder| {
            recorder.push(Command::Barrier(Barrier {
                src_stages,
                src_access: Access::MEMORY_WRITE,
                dst_stages,
                dst_access: Access::MEMORY_READ | Access::MEMORY_WRITE,
            }));
        })
    }

    /// Forgets the bindings generated commands left undefined: the bound
    /// pipeline, vertex and index buffers, and push constants. The
    /// encoder's pipeline, if any, is bound again.
    pub(super) fn after_generated_commands(&mut self) -> Result<()> {
        let render = render(&mut self.encoder)?;
        render.bound_vertex_buffers = [None; 31];
        render.index_buffer = None;
        render.topology = None;
        render.restart = None;
        self.bindings.pushed = None;
        render.bound = None;
        if let Some(state) = &render.pipeline {
            if let Some(pipeline) = state.pipeline {
                render.bound = Some(pipeline);
                self.recorder.push(Command::BindPipeline {
                    bind_point: PipelineBindPoint::Graphics,
                    pipeline,
                });
            }
            let discard = !state.plan.rasterization_enabled;
            self.recorder.push(Command::SetRasterizerDiscard(discard));
        }
        Ok(())
    }

    /// Ends the pass instance, records `between` outside it and resumes
    /// with an instance that loads every attachment.
    pub(super) fn outside_pass(
        &mut self,
        resolver: &mut dyn Resolver,
        between: impl FnOnce(&mut Recorder),
    ) -> Result<()> {
        let render = render(&mut self.encoder)?;
        end_query(&mut self.recorder, &mut render.visibility);
        self.recorder.pause_statistics();
        // The instance's store operations follow its storage writes, and the
        // next one's load operations are the blend unit's.
        if render.attachment_paths.contains(&AttachmentPath::Storage) {
            self.recorder.push(Command::RegionBarrier(ATTACHMENT_PATHS));
        }
        render.attachment_paths = [AttachmentPath::Blend; MAX_COLOR_ATTACHMENTS];
        self.recorder.push(Command::EndRendering);
        between(&mut self.recorder);
        let index = self.recorder.commands().len();
        // Patched with the final attachment operations at endEncoding.
        self.recorder.push(Command::BeginRendering(rendering(
            &render.pass,
            false,
            false,
        )));
        render.begins.push(index);
        self.recorder.resume_statistics(resolver, false)
    }

    pub fn draw_primitives(
        &mut self,
        resolver: &mut dyn Resolver,
        primitive: PrimitiveType,
        vertex_start: u32,
        vertex_count: u32,
        instance_count: u32,
        base_instance: u32,
    ) -> Result<()> {
        self.prepare_draw(resolver, primitive)?;
        if vertex_count == 0 || instance_count == 0 {
            return Ok(());
        }
        let first_vertex = vertex_start;
        let draw = Command::Draw {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance: base_instance,
        };
        self.amplified(resolver, draw)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives(
        &mut self,
        resolver: &mut dyn Resolver,
        primitive: PrimitiveType,
        index_count: u32,
        index_type: MtlIndexType,
        index_buffer: BufferRef,
        index_buffer_offset: u64,
        instance_count: u32,
        base_vertex: i32,
        base_instance: u32,
    ) -> Result<()> {
        index_buffer.at(
            index_buffer_offset,
            u64::from(index_count) * index_type.bytes(),
        )?;
        self.prepare_draw(resolver, primitive)?;
        self.bind_index_buffer(index_type, index_buffer, index_buffer_offset)?;
        if index_count == 0 || instance_count == 0 {
            return Ok(());
        }
        let draw = Command::DrawIndexed {
            index_count,
            instance_count,
            first_index: 0,
            vertex_offset: base_vertex,
            first_instance: base_instance,
        };
        self.amplified(resolver, draw)
    }

    /// `drawPrimitives:indirectBuffer:indirectBufferOffset:`; Metal's
    /// argument layout is `VkDrawIndirectCommand`.
    pub fn draw_primitives_indirect(
        &mut self,
        resolver: &mut dyn Resolver,
        primitive: PrimitiveType,
        indirect: BufferRef,
        offset: u64,
    ) -> Result<()> {
        ensure(
            offset.is_multiple_of(4),
            "indirect arguments are 4-byte aligned",
        )?;
        let offset = indirect.at(offset, 16)?;
        self.prepare_draw(resolver, primitive)?;
        self.amplified(
            resolver,
            Command::DrawIndirect {
                buffer: indirect.buffer,
                offset,
            },
        )
    }

    /// Metal's indexed argument layout is `VkDrawIndexedIndirectCommand`,
    /// with indices counted from the index buffer offset.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives_indirect(
        &mut self,
        resolver: &mut dyn Resolver,
        primitive: PrimitiveType,
        index_type: MtlIndexType,
        index_buffer: BufferRef,
        index_buffer_offset: u64,
        indirect: BufferRef,
        offset: u64,
    ) -> Result<()> {
        ensure(
            offset.is_multiple_of(4),
            "indirect arguments are 4-byte aligned",
        )?;
        let offset = indirect.at(offset, 20)?;
        self.prepare_draw(resolver, primitive)?;
        self.bind_index_buffer(index_type, index_buffer, index_buffer_offset)?;
        self.amplified(
            resolver,
            Command::DrawIndexedIndirect {
                buffer: indirect.buffer,
                offset,
            },
        )
    }

    pub fn set_tessellation_factor_buffer(
        &mut self,
        buffer: Option<BufferRef>,
        offset: u64,
        instance_stride: u64,
    ) -> Result<()> {
        let render = render(&mut self.encoder)?;
        ensure(
            offset.is_multiple_of(4),
            "tessellation factor offsets are 4-byte aligned",
        )?;
        let instance_stride = u32::try_from(instance_stride)
            .map_err(|_| Error::Invalid("tessellation factor instance stride"))?;
        render.factors = buffer.map(|buffer| FactorBuffer {
            buffer,
            offset,
            instance_stride,
        });
        Ok(())
    }

    pub fn set_tessellation_factor_scale(&mut self, scale: f32) -> Result<()> {
        let render = render(&mut self.encoder)?;
        ensure(
            scale.is_finite() && scale >= 0.0,
            "tessellation factor scales are finite and nonnegative",
        )?;
        render.factor_scale = scale;
        Ok(())
    }

    /// `drawPatches:…` and `drawIndexedPatches:…`: patch `n` of the draw is
    /// vertex `patch_start + n` of a single-point patch list.
    pub fn draw_patches(
        &mut self,
        resolver: &mut dyn Resolver,
        source: &PatchSource,
        patch_start: u32,
        patch_count: u32,
        instance_count: u32,
        base_instance: u32,
    ) -> Result<()> {
        self.prepare_patches(resolver, source)?;
        if patch_count == 0 || instance_count == 0 {
            return Ok(());
        }
        let draw = Command::Draw {
            vertex_count: patch_count,
            instance_count,
            first_vertex: patch_start,
            first_instance: base_instance,
        };
        self.patch_draw(resolver, draw)
    }

    /// The indirect patch draws: `MTLDrawPatchIndirectArguments` (patch
    /// count, instance count, patch start, base instance) is the
    /// `VkDrawIndirectCommand` of [`Self::draw_patches`].
    pub fn draw_patches_indirect(
        &mut self,
        resolver: &mut dyn Resolver,
        source: &PatchSource,
        indirect: BufferRef,
        offset: u64,
    ) -> Result<()> {
        ensure(
            offset.is_multiple_of(4),
            "indirect arguments are 4-byte aligned",
        )?;
        let offset = indirect.at(offset, 16)?;
        self.prepare_patches(resolver, source)?;
        self.patch_draw(
            resolver,
            Command::DrawIndirect {
                buffer: indirect.buffer,
                offset,
            },
        )
    }

    /// A patch draw counts in its own statistics interval: its synthesized
    /// vertex stage is not a Metal vertex function.
    fn patch_draw(&mut self, resolver: &mut dyn Resolver, draw: Command) -> Result<()> {
        self.recorder.split_statistics(resolver, true)?;
        self.amplified(resolver, draw)?;
        self.recorder.split_statistics(resolver, false)
    }

    /// `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
    pub fn draw_mesh_threadgroups(
        &mut self,
        resolver: &mut dyn Resolver,
        threadgroups: [u32; 3],
        object_threads: [u32; 3],
        mesh_threads: [u32; 3],
    ) -> Result<()> {
        self.prepare_mesh(resolver, object_threads, mesh_threads, [0; 3])?;
        if threadgroups.contains(&0) {
            return Ok(());
        }
        self.amplified(resolver, Command::DrawMeshTasks(threadgroups))
    }

    /// `drawMeshThreads:…`: whole threadgroups cover the grid, and the grid
    /// size bounds the threads that run.
    pub fn draw_mesh_threads(
        &mut self,
        resolver: &mut dyn Resolver,
        threads: [u32; 3],
        object_threads: [u32; 3],
        mesh_threads: [u32; 3],
    ) -> Result<()> {
        let pipeline = render(&mut self.encoder)?
            .pipeline
            .as_ref()
            .ok_or(Error::Invalid("no render pipeline state is set"))?;
        // Variants add the object and mesh stages, so the plan lacks them.
        let has_object = pipeline
            .interfaces
            .iter()
            .any(|interface| interface.stage == Stage::Object);
        let group = if has_object {
            object_threads
        } else {
            mesh_threads
        };
        ensure(!group.contains(&0), "threadgroups have at least one thread")?;
        let groups = [0, 1, 2].map(|axis| threads[axis].div_ceil(group[axis]));
        self.prepare_mesh(resolver, object_threads, mesh_threads, threads)?;
        if groups.contains(&0) {
            return Ok(());
        }
        self.amplified(resolver, Command::DrawMeshTasks(groups))
    }

    /// Metal's threadgroup arguments are `VkDrawMeshTasksIndirectCommandEXT`.
    pub fn draw_mesh_threadgroups_indirect(
        &mut self,
        resolver: &mut dyn Resolver,
        indirect: BufferRef,
        offset: u64,
        object_threads: [u32; 3],
        mesh_threads: [u32; 3],
    ) -> Result<()> {
        ensure(
            offset.is_multiple_of(4),
            "indirect arguments are 4-byte aligned",
        )?;
        let offset = indirect.at(offset, 12)?;
        self.prepare_mesh(resolver, object_threads, mesh_threads, [0; 3])?;
        self.amplified(
            resolver,
            Command::DrawMeshTasksIndirect {
                buffer: indirect.buffer,
                offset,
            },
        )
    }

    fn bind_index_buffer(
        &mut self,
        index_type: MtlIndexType,
        buffer: BufferRef,
        offset: u64,
    ) -> Result<()> {
        ensure(
            offset.is_multiple_of(index_type.bytes()),
            "index buffer offsets are aligned to the index size",
        )?;
        let offset = buffer.at(offset, 0)?;
        let index_type = match index_type {
            MtlIndexType::UInt16 => IndexType::Uint16,
            MtlIndexType::UInt32 => IndexType::Uint32,
        };
        let render = render(&mut self.encoder)?;
        let binding = (buffer.buffer, offset, index_type);
        if render.index_buffer != Some(binding) {
            render.index_buffer = Some(binding);
            self.recorder.push(Command::BindIndexBuffer {
                buffer: buffer.buffer,
                offset,
                index_type,
            });
        }
        Ok(())
    }

    /// Validates the pipeline for a vertex draw and brings topology, vertex
    /// buffers, the visibility query and the tables up to date.
    fn prepare_draw(
        &mut self,
        resolver: &mut dyn Resolver,
        primitive: PrimitiveType,
    ) -> Result<()> {
        let render = render(&mut self.encoder)?;
        let pipeline = render
            .pipeline
            .as_ref()
            .ok_or(Error::Invalid("no render pipeline state is set"))?;
        ensure(
            pipeline.plan.mesh.is_none(),
            "mesh pipelines draw with drawMeshThreadgroups",
        )?;
        ensure(
            pipeline.plan.tessellation.is_none(),
            "tessellated pipelines draw with drawPatches",
        )?;
        let class = pipeline.plan.topology_class;
        ensure(
            class == PrimitiveTopologyClass::Unspecified || class == primitive.class(),
            "the primitive type is outside the pipeline's topology class",
        )?;
        self.bindings.require_buffers(&pipeline.interfaces)?;
        self.bind_bicubic(resolver)?;
        let render = self::render(&mut self.encoder)?;
        let pipeline = render
            .pipeline
            .as_ref()
            .ok_or(Error::Invalid("no render pipeline state is set"))?;
        let topology = primitive_topology(primitive);
        let mut commands = Vec::new();
        if render.topology != Some(topology) {
            render.topology = Some(topology);
            commands.push(Command::SetPrimitiveTopology(topology));
        }
        // Metal restarts strips at the all-ones index, and never lists.
        let restart = primitive.is_strip();
        if render.restart != Some(restart) {
            render.restart = Some(restart);
            commands.push(Command::SetPrimitiveRestart(restart));
        }
        for binding in &pipeline.plan.info.vertex_input.bindings {
            let index = binding.binding as usize;
            let bound = render.vertex_buffers[index].ok_or(Error::Invalid(
                "a vertex buffer the pipeline reads is unbound",
            ))?;
            let stride = match (binding.stride, bound.stride) {
                (Some(stride), _) => u64::from(stride),
                (None, Some(stride)) => stride,
                (None, None) => {
                    return Err(Error::Invalid(
                        "a dynamic-stride vertex buffer was bound without a stride",
                    ));
                }
            };
            let offset = bound.buffer.at(bound.offset, 0)?;
            let size = bound.buffer.length - bound.offset;
            let buffer = VertexBuffer {
                buffer: bound.buffer.buffer,
                offset,
                size,
                stride: Some(stride),
            };
            if render.bound_vertex_buffers[index] != Some(buffer) {
                render.bound_vertex_buffers[index] = Some(buffer);
                commands.push(Command::BindVertexBuffers {
                    first: binding.binding,
                    buffers: alloc::vec![buffer],
                });
            }
        }
        commands
            .into_iter()
            .for_each(|command| self.recorder.push(command));
        self.prepare_fragment_tests(resolver)?;
        self.order_attachments(resolver)?;
        self.begin_query(resolver)?;
        self.bindings.flush(
            &mut self.recorder,
            resolver,
            &[Stage::Vertex, Stage::Fragment],
        )
    }

    /// A draw accessing a color attachment by one path follows the pass
    /// instance's accesses of it by the other: the blend unit's and its
    /// load operation's, or fragment functions' storage accesses, which
    /// follow each other through their interlock.
    fn order_attachments(&mut self, resolver: &mut dyn Resolver) -> Result<()> {
        let render = render(&mut self.encoder)?;
        let Some(pipeline) = render.pipeline.as_ref() else {
            return Ok(());
        };
        let storage = storage_attachments(pipeline);
        let blended = blended_attachments(pipeline);
        self.place_color_copies(resolver, storage)?;
        self.order_attachment_paths(storage, blended)
    }

    /// Orders a draw that accesses the `storage` attachments as storage
    /// and writes the `blended` ones through the blend unit.
    pub(super) fn order_attachment_paths(&mut self, storage: u32, blended: u32) -> Result<()> {
        let render = render(&mut self.encoder)?;
        let blended = blended & !storage;
        let waits = (0..MAX_COLOR_ATTACHMENTS).any(|index| {
            let path = render.attachment_paths[index];
            storage & 1 << index != 0 && path == AttachmentPath::Blend
                || blended & 1 << index != 0 && path == AttachmentPath::Storage
        });
        if waits {
            render.attachment_paths = [AttachmentPath::None; MAX_COLOR_ATTACHMENTS];
        }
        for (index, path) in render.attachment_paths.iter_mut().enumerate() {
            if storage & 1 << index != 0 {
                *path = AttachmentPath::Storage;
            } else if blended & 1 << index != 0 && *path == AttachmentPath::None {
                *path = AttachmentPath::Blend;
            }
        }
        if waits {
            self.recorder.push(Command::RegionBarrier(ATTACHMENT_PATHS));
        }
        Ok(())
    }

    /// Validates the pipeline and buffers of a patch draw, uploads its
    /// arguments and brings topology, the visibility query and the tables up
    /// to date.
    fn prepare_patches(&mut self, resolver: &mut dyn Resolver, source: &PatchSource) -> Result<()> {
        let render = render(&mut self.encoder)?;
        let pipeline = render
            .pipeline
            .as_ref()
            .ok_or(Error::Invalid("no render pipeline state is set"))?;
        let tessellation = pipeline.plan.tessellation.as_ref().ok_or(Error::Invalid(
            "patch draws need a post-tessellation vertex function",
        ))?;
        self.bindings.require_buffers(&pipeline.interfaces)?;
        // Each patch draws as one point the stage fetches its control points
        // for: a function that reads none takes patches of none.
        let reads_control_points = pipeline.interfaces.first().is_some_and(|interface| {
            interface
                .attributes
                .iter()
                .any(|attribute| attribute.patch_control_point)
        });
        ensure(
            source.control_points <= MAX_PATCH_CONTROL_POINTS
                && (source.control_points != 0 || !reads_control_points),
            "patch control point count",
        )?;
        ensure(
            tessellation
                .patch
                .control_points
                .is_none_or(|count| count == source.control_points),
            "the draw's control point count differs from the function's",
        )?;
        let index_bytes = match tessellation.control_point_index {
            TessellationControlPointIndexType::None => None,
            TessellationControlPointIndexType::UInt16 => Some(2),
            TessellationControlPointIndexType::UInt32 => Some(4),
        };
        ensure(
            index_bytes.is_some() == source.control_point_indices.is_some(),
            "indexed patch draws match the pipeline's control point index type",
        )?;
        self.bind_bicubic(resolver)?;
        let render = self::render(&mut self.encoder)?;
        let pipeline = render
            .pipeline
            .as_ref()
            .ok_or(Error::Invalid("no render pipeline state is set"))?;
        let tessellation = pipeline.plan.tessellation.as_ref().ok_or(Error::Invalid(
            "patch draws need a post-tessellation vertex function",
        ))?;
        let factors = render
            .factors
            .ok_or(Error::Invalid("no tessellation factor buffer is set"))?;
        let binding = |(buffer, offset): (BufferRef, u64), alignment: u64| {
            ensure(
                offset.is_multiple_of(alignment),
                "patch buffer offsets are aligned to their elements",
            )?;
            buffer.at(offset, 0)?;
            Ok::<_, Error>(BufferBinding {
                address: buffer.address + offset,
                length: buffer.length - offset,
            })
        };
        let arguments = TessellationArguments {
            factors: binding((factors.buffer, factors.offset), 4)?,
            patch_indices: match source.patch_indices {
                Some(indices) => binding(indices, 4)?,
                None => BufferBinding::default(),
            },
            control_point_indices: match (source.control_point_indices, index_bytes) {
                (Some(indices), Some(bytes)) => binding(indices, bytes)?,
                _ => BufferBinding::default(),
            },
            instance_stride: factors.instance_stride,
            factor_scale: if tessellation.factor_scale {
                render.factor_scale
            } else {
                1.0
            },
            control_points: source.control_points,
            strides: patch_strides(
                &render.vertex_buffers,
                pipeline.interfaces.first().ok_or(Error::Invalid(
                    "the pipeline has no post-tessellation vertex function",
                ))?,
                tessellation.vertex_descriptor.as_ref(),
            )?,
        };
        if render.topology != Some(PrimitiveTopology::PatchList) {
            render.topology = Some(PrimitiveTopology::PatchList);
            self.recorder
                .push(Command::SetPrimitiveTopology(PrimitiveTopology::PatchList));
        }
        if render.restart != Some(false) {
            render.restart = Some(false);
            self.recorder.push(Command::SetPrimitiveRestart(false));
        }
        let upload = self.recorder.upload(
            resolver,
            &arguments.encode(),
            TessellationArguments::ALIGNMENT,
        )?;
        self.bindings.constants.arguments = upload.address;
        self.prepare_fragment_tests(resolver)?;
        self.order_attachments(resolver)?;
        self.begin_query(resolver)?;
        self.bindings.flush(
            &mut self.recorder,
            resolver,
            &[Stage::Vertex, Stage::Fragment],
        )
    }

    /// Binds the variant of the vertex pipeline whose stages filter the
    /// bicubic samplers the draw binds them bicubically, or the pipeline
    /// itself when it does.
    fn bind_bicubic(&mut self, resolver: &mut dyn Resolver) -> Result<()> {
        let render = render(&mut self.encoder)?;
        let Some(state) = render.pipeline.as_ref() else {
            return Ok(());
        };
        let mut bicubic = [0; 2];
        for interface in &state.interfaces {
            let stage = match interface.stage {
                Stage::Fragment => 1,
                _ => 0,
            };
            bicubic[stage] |= self.bindings.bicubic_samplers(interface.stage, interface);
        }
        let pipeline =
            if !state.outdated && (0..2).all(|stage| bicubic[stage] & !state.bicubic[stage] == 0) {
                state.pipeline
            } else {
                Some(resolver.bicubic_render_pipeline(state, bicubic)?)
            };
        if let Some(pipeline) = pipeline.filter(|&pipeline| render.bound != Some(pipeline)) {
            render.bound = Some(pipeline);
            self.recorder.push(Command::BindPipeline {
                bind_point: PipelineBindPoint::Graphics,
                pipeline,
            });
        }
        Ok(())
    }

    fn prepare_mesh(
        &mut self,
        resolver: &mut dyn Resolver,
        object_threads: [u32; 3],
        mesh_threads: [u32; 3],
        grid: [u32; 3],
    ) -> Result<()> {
        let render = render(&mut self.encoder)?;
        let state = render
            .pipeline
            .as_ref()
            .ok_or(Error::Invalid("no render pipeline state is set"))?;
        ensure(
            state.plan.mesh.is_some(),
            "vertex pipelines draw with drawPrimitives",
        )?;
        self.bindings.require_buffers(&state.interfaces)?;
        ensure(
            !mesh_threads.contains(&0),
            "mesh threadgroups have at least one thread",
        )?;
        let pipeline = resolver.mesh_pipeline(
            state,
            object_threads,
            mesh_threads,
            &render.object_threadgroup_lengths,
        )?;
        render.bound = Some(pipeline);
        self.recorder.push(Command::BindPipeline {
            bind_point: PipelineBindPoint::Graphics,
            pipeline,
        });
        self.bindings.constants.grid = grid;
        self.prepare_fragment_tests(resolver)?;
        self.order_attachments(resolver)?;
        self.begin_query(resolver)?;
        self.bindings.flush(
            &mut self.recorder,
            resolver,
            &[Stage::Object, Stage::Mesh, Stage::Fragment],
        )
    }

    /// Begins an occlusion query for the current visibility mode if none is
    /// active.
    pub(super) fn begin_query(&mut self, resolver: &mut dyn Resolver) -> Result<()> {
        let render = render(&mut self.encoder)?;
        let visibility = &mut render.visibility;
        let (Some((mode, offset)), None) = (visibility.mode, visibility.active) else {
            return Ok(());
        };
        let (pool, query) = match visibility.pool {
            Some((pool, next, count)) if next < count => {
                visibility.pool = Some((pool, next + 1, count));
                (pool, next)
            }
            _ => {
                let (pool, count) = resolver.query_pool(QueryType::Occlusion)?;
                ensure(count > 0, "occlusion pools hold queries")?;
                self.recorder.query_pools.push((QueryType::Occlusion, pool));
                visibility.pool = Some((pool, 1, count));
                (pool, 0)
            }
        };
        visibility.active = Some((pool, query));
        visibility.queries.push((pool, query, offset));
        let precise = mode == VisibilityResultMode::Counting;
        self.recorder.push(Command::BeginQuery {
            pool,
            query,
            precise,
        });
        Ok(())
    }

    /// Records `draw` once per amplified view, each with its view's push
    /// constants.
    fn amplified(&mut self, resolver: &mut dyn Resolver, draw: Command) -> Result<()> {
        let views = render(&mut self.encoder)?.amplification.as_slice();
        let size = match draw {
            Command::Draw {
                vertex_count: count,
                instance_count,
                ..
            }
            | Command::DrawIndexed {
                index_count: count,
                instance_count,
                ..
            } => Some(u64::from(count) * u64::from(instance_count)),
            Command::DrawMeshTasks(groups) => {
                Some(groups.iter().map(|&count| u64::from(count)).product())
            }
            _ => None,
        };
        match size {
            Some(size) => {
                self.workload.direct += 1;
                self.workload.largest = self.workload.largest.max(size);
            }
            None => self.workload.indirect += 1,
        }
        self.bindings.constants.amplification_count = views.len() as u32;
        for (view, &mapping) in views.iter().enumerate() {
            self.bindings.constants.amplification_id = view as u32;
            self.bindings.constants.view_mapping = mapping;
            self.bindings.flush(&mut self.recorder, resolver, &[])?;
            self.recorder.push(draw.clone());
        }
        Ok(())
    }
}

fn render_stages(stages: RenderStages) -> PipelineStages {
    let fragment = PipelineStages::FRAGMENT_SHADER
        .union(PipelineStages::EARLY_FRAGMENT_TESTS)
        .union(PipelineStages::LATE_FRAGMENT_TESTS)
        .union(PipelineStages::COLOR_ATTACHMENT_OUTPUT);
    PipelineStages::EMPTY
        .with(
            PipelineStages::VERTEX_INPUT
                | PipelineStages::VERTEX_SHADER
                | PipelineStages::TESSELLATION_CONTROL_SHADER
                | PipelineStages::TESSELLATION_EVALUATION_SHADER,
            stages.contains(RenderStages::VERTEX),
        )
        .with(fragment, stages.contains(RenderStages::FRAGMENT))
        .with(
            PipelineStages::TASK_SHADER,
            stages.contains(RenderStages::OBJECT),
        )
        .with(
            PipelineStages::MESH_SHADER,
            stages.contains(RenderStages::MESH),
        )
}

#[cfg(test)]
mod drawn_tests {
    use super::DrawnArea;

    /// Drawn rectangles stay disjoint: one overlapping kept ones takes them
    /// in, and past the limit the two whose union adds the least merge.
    #[test]
    fn drawn_rectangles_stay_disjoint_and_few() {
        let mut area = DrawnArea::default();
        area.add([0, 0, 4, 4]);
        area.add([8, 0, 12, 4]);
        area.add([2, 2, 10, 3]);
        assert_eq!(area.rects, [[0, 0, 12, 4]]);
        area.add([5, 5, 5, 9]);
        assert_eq!(area.rects.len(), 1);
        let mut many = DrawnArea::default();
        for n in 0..9 {
            many.add([n * 10, 0, n * 10 + 2, 2]);
        }
        assert_eq!(many.rects.len(), super::DRAWN_RECTS);
        assert!(many.rects.contains(&[0, 0, 12, 2]));
        for (n, &a) in many.rects.iter().enumerate() {
            assert!(many.rects[n + 1..].iter().all(|&b| !super::overlap(a, b)));
        }
        assert_eq!(
            DrawnArea::bounds(&[[8, 4, 16, 8], [24, 12, 16, 8]]),
            [8, 4, 32, 16]
        );
        assert_eq!(DrawnArea::bounds(&[]), [0; 4]);
    }
}
