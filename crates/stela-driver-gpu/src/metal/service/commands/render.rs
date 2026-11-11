//! Commands of `MTLRenderCommandEncoder`.
use super::{Payload, binding, count, depth_stencil_state, grid, index_type, primitive, size};
use crate::metal::{
    abi::{
        BarrierScope, CullMode, DepthClipMode, LoadAction, MultisampleDepthResolveFilter,
        MultisampleStencilResolveFilter, PrimitiveType, RenderStages, StoreAction,
        StoreActionOptions, TriangleFillMode, VisibilityResultMode, VisibilityResultType, Winding,
    },
    command_buffer::{
        CommandBuffer,
        render::{PassBoundary, ScissorRect, Viewport},
    },
    device::{
        BufferId, ComputePipelineId, CounterBufferId, RenderPipelineId, TextureId,
        queue::{AttachmentRequest, RenderPassRequest},
    },
    pass::{AttachmentSlot, TileMemory},
    pipeline::MAX_COLOR_ATTACHMENTS,
    service::{
        Service, boolean, decode, decode_bits,
        pipelines::render_stage,
        wire::{
            self, Fault, Status,
            commands::{
                BeginRender, BlendColor, ColorStoreAction, ColorStoreActionOptions,
                CounterSampleRecord, DepthBias, DrawIndexed, DrawIndexedIndirect, DrawIndirect,
                DrawMesh, DrawMeshIndirect, DrawPatches, DrawPatchesIndirect, DrawPrimitives,
                FactorBuffer, FactorScale, Mode, PassAttachmentRecord, PassBoundaryCode,
                RenderBarrier, RenderCode, ScissorRecord, SetPipeline, SetThreadgroupMemory,
                StageBuffer, StageBufferOffset, StageBytes, StageSampler, StageTexture,
                StencilReference, TileDispatch, TileThreadgroupMemory, VertexBufferStride,
                VertexOffsetStride, ViewMappingRecord, ViewportRecord, VisibilityMode,
            },
        },
    },
    shader::Compiler,
    vulkan::backend::Backend,
};
use alloc::{format, string::String, vec::Vec};
use core::num::NonZeroU64;

/// `MTLRenderStageTile`, whose bindings tile dispatches read.
const TILE_STAGE: u64 = 4;

impl<B: Backend, C: Compiler> Service<B, C> {
    pub(super) fn render(
        &mut self,
        connection: NonZeroU64,
        commands: &mut CommandBuffer,
        code: RenderCode,
        payload: Payload,
    ) -> Result<(), Fault> {
        match code {
            RenderCode::Begin => {
                let (record, data) = payload.split::<BeginRender>()?;
                let request = self.render_pass(connection, &record, data.bytes())?;
                if let Err(error) = self.device.begin_render(commands, &request) {
                    self.noted = Some(self.describe_pass(&request));
                    return Err(error.into());
                }
            }
            RenderCode::SetPipeline => {
                let SetPipeline { pipeline } = payload.record()?;
                let pipeline = self.object::<RenderPipelineId>(connection, pipeline)?;
                commands.set_render_pipeline_state(self.device.render_pipeline_state(pipeline)?)?;
            }
            RenderCode::SetBuffer => {
                let StageBuffer {
                    stage,
                    buffer,
                    offset,
                    index,
                } = payload.record()?;
                let buffer = self.optional_buffer(connection, buffer)?;
                if stage == TILE_STAGE {
                    commands.set_tile_buffer(buffer, offset, binding(index)?)?;
                } else {
                    commands.set_render_buffer(
                        render_stage(stage)?,
                        buffer,
                        offset,
                        binding(index)?,
                    )?;
                }
            }
            RenderCode::SetBufferOffset => {
                let StageBufferOffset {
                    stage,
                    offset,
                    index,
                } = payload.record()?;
                if stage == TILE_STAGE {
                    commands.set_tile_buffer_offset(offset, binding(index)?)?;
                } else {
                    commands.set_render_buffer_offset(
                        render_stage(stage)?,
                        offset,
                        binding(index)?,
                    )?;
                }
            }
            RenderCode::SetBytes => {
                let (StageBytes { stage, index }, bytes) = payload.split()?;
                let resolver = &mut self.device.resolver();
                if stage == TILE_STAGE {
                    commands.set_tile_bytes(resolver, bytes.bytes(), binding(index)?)?;
                } else {
                    commands.set_render_bytes(
                        resolver,
                        render_stage(stage)?,
                        bytes.bytes(),
                        binding(index)?,
                    )?;
                }
            }
            RenderCode::SetTexture => {
                let StageTexture {
                    stage,
                    texture,
                    index,
                } = payload.record()?;
                let texture = self.texture_id(connection, texture)?;
                if stage == TILE_STAGE {
                    commands.set_tile_texture(texture, binding(index)?)?;
                } else {
                    commands.set_render_texture(render_stage(stage)?, texture, binding(index)?)?;
                }
            }
            RenderCode::SetSampler => {
                let request: StageSampler = payload.record()?;
                let sampler = self.sampler_id(connection, request.sampler)?;
                let clamp = boolean(request.clamped)?.then_some((request.lod_min, request.lod_max));
                let resolver = &mut self.device.resolver();
                if request.stage == TILE_STAGE {
                    commands.set_tile_sampler(resolver, sampler, clamp, binding(request.index)?)?;
                } else {
                    commands.set_render_sampler(
                        resolver,
                        render_stage(request.stage)?,
                        sampler,
                        clamp,
                        binding(request.index)?,
                    )?;
                }
            }
            RenderCode::SetVertexBufferStride => {
                let request: VertexBufferStride = payload.record()?;
                let buffer = self.buffer_ref(connection, request.buffer)?;
                commands.set_vertex_buffer_with_stride(
                    buffer,
                    request.offset,
                    request.stride,
                    binding(request.index)?,
                )?;
            }
            RenderCode::SetVertexBufferOffsetStride => {
                let request: VertexOffsetStride = payload.record()?;
                commands.set_vertex_buffer_offset_with_stride(
                    request.offset,
                    request.stride,
                    binding(request.index)?,
                )?;
            }
            RenderCode::SetViewports => {
                let viewports: Vec<Viewport> = payload
                    .list::<ViewportRecord>()?
                    .into_iter()
                    .map(|record| Viewport {
                        origin_x: record.origin_x,
                        origin_y: record.origin_y,
                        width: record.width,
                        height: record.height,
                        znear: record.znear,
                        zfar: record.zfar,
                    })
                    .collect();
                commands.set_viewports(&viewports)?;
            }
            RenderCode::SetScissorRects => {
                let rects: Vec<ScissorRect> = payload
                    .list::<ScissorRecord>()?
                    .into_iter()
                    .map(|record| ScissorRect {
                        x: record.x,
                        y: record.y,
                        width: record.width,
                        height: record.height,
                    })
                    .collect();
                commands.set_scissor_rects(&rects)?;
            }
            RenderCode::SetCullMode => {
                let Mode { mode } = payload.record()?;
                commands.set_cull_mode(decode(CullMode::from_raw, mode, "cull mode")?)?;
            }
            RenderCode::SetFrontFacingWinding => {
                let Mode { mode } = payload.record()?;
                commands.set_front_facing_winding(decode(Winding::from_raw, mode, "winding")?)?;
            }
            RenderCode::SetDepthBias => {
                let DepthBias {
                    constant,
                    slope,
                    clamp,
                    reserved: _,
                } = payload.record()?;
                commands.set_depth_bias(constant, slope, clamp)?;
            }
            RenderCode::SetDepthStencilState => {
                let state = depth_stencil_state(&payload.record()?)?;
                commands.set_depth_stencil_state(Some(&state))?;
            }
            RenderCode::SetTilePipeline => {
                let SetPipeline { pipeline } = payload.record()?;
                let pipeline = self.object::<ComputePipelineId>(connection, pipeline)?;
                let (state, target) = self.device.tile_pipeline_state(pipeline)?;
                commands.set_tile_pipeline_state(state, target)?;
            }
            RenderCode::SetTileThreadgroupMemory => {
                let TileThreadgroupMemory {
                    length,
                    offset,
                    index,
                } = payload.record()?;
                commands.set_tile_threadgroup_memory(length, offset, binding(index)?)?;
            }
            RenderCode::DispatchThreadsPerTile => {
                let TileDispatch { threads } = payload.record()?;
                commands.dispatch_threads_per_tile(&mut self.device.resolver(), size(threads))?;
            }
            RenderCode::SetObjectThreadgroupMemory => {
                let SetThreadgroupMemory { length, index } = payload.record()?;
                commands.set_object_threadgroup_memory_length(length, binding(index)?)?;
            }
            RenderCode::SetStencilReference => {
                let StencilReference { front, back } = payload.record()?;
                commands.set_stencil_reference_values(front, back)?;
            }
            RenderCode::SetBlendColor => {
                let BlendColor { color } = payload.record()?;
                commands.set_blend_color(color)?;
            }
            RenderCode::SetTriangleFillMode => {
                let Mode { mode } = payload.record()?;
                let mode = decode(TriangleFillMode::from_raw, mode, "triangle fill mode")?;
                commands.set_triangle_fill_mode(mode)?;
            }
            RenderCode::SetDepthClipMode => {
                let Mode { mode } = payload.record()?;
                commands.set_depth_clip_mode(decode(
                    DepthClipMode::from_raw,
                    mode,
                    "clip mode",
                )?)?;
            }
            RenderCode::SetVertexAmplification => {
                let mappings: Vec<(u32, u32)> = payload
                    .list::<ViewMappingRecord>()?
                    .into_iter()
                    .map(|mapping| {
                        (
                            mapping.viewport_array_index_offset,
                            mapping.render_target_array_index_offset,
                        )
                    })
                    .collect();
                commands.set_vertex_amplification(&mappings)?;
            }
            RenderCode::SetVisibilityResultMode => {
                let VisibilityMode { mode, offset } = payload.record()?;
                let mode = decode(VisibilityResultMode::from_raw, mode, "visibility mode")?;
                commands.set_visibility_result_mode(mode, offset)?;
            }
            RenderCode::SetColorStoreAction => {
                let ColorStoreAction { action, index } = payload.record()?;
                let index = usize::try_from(index)
                    .map_err(|_| Fault::malformed("color index").value(index))?;
                commands.set_store_action(AttachmentSlot::Color(index), store_action(action)?)?;
            }
            RenderCode::SetDepthStoreAction => {
                let Mode { mode } = payload.record()?;
                commands.set_store_action(AttachmentSlot::Depth, store_action(mode)?)?;
            }
            RenderCode::SetStencilStoreAction => {
                let Mode { mode } = payload.record()?;
                commands.set_store_action(AttachmentSlot::Stencil, store_action(mode)?)?;
            }
            RenderCode::SetColorStoreActionOptions => {
                let ColorStoreActionOptions { options, index } = payload.record()?;
                let index = usize::try_from(index)
                    .map_err(|_| Fault::malformed("color index").value(index))?;
                commands.set_store_action_options(
                    AttachmentSlot::Color(index),
                    store_options(options)?,
                )?;
            }
            RenderCode::SetDepthStoreActionOptions => {
                let Mode { mode } = payload.record()?;
                commands.set_store_action_options(AttachmentSlot::Depth, store_options(mode)?)?;
            }
            RenderCode::SetStencilStoreActionOptions => {
                let Mode { mode } = payload.record()?;
                commands.set_store_action_options(AttachmentSlot::Stencil, store_options(mode)?)?;
            }
            RenderCode::MemoryBarrier => {
                let RenderBarrier {
                    scope,
                    after,
                    before,
                } = payload.record()?;
                let scope = decode_bits(BarrierScope::from_bits, scope, "barrier scope")?;
                let stages = |raw| decode_bits(RenderStages::from_bits, raw, "render stages");
                commands.render_memory_barrier(
                    &mut self.device.resolver(),
                    scope,
                    stages(after)?,
                    stages(before)?,
                )?;
            }
            RenderCode::DrawPrimitives => {
                let draw: DrawPrimitives = payload.record()?;
                let primitive = primitive(draw.primitive)?;
                if self.device.is_hosted_lavapipe()
                    && !matches!(
                        primitive,
                        PrimitiveType::Triangle | PrimitiveType::TriangleStrip
                    )
                {
                    return Err(Fault::new(
                        Status::Unsupported,
                        "primitive on hosted lavapipe",
                    ));
                }
                commands.draw_primitives(
                    &mut self.device.resolver(),
                    primitive,
                    count(draw.vertex_start)?,
                    count(draw.vertex_count)?,
                    count(draw.instance_count)?,
                    count(draw.base_instance)?,
                )?;
            }
            RenderCode::DrawIndexed => {
                let draw: DrawIndexed = payload.record()?;
                let index_buffer = self.buffer_ref(connection, draw.index_buffer)?;
                let base_vertex = i32::try_from(draw.base_vertex).map_err(|_| {
                    Fault::new(Status::Invalid, "the base vertex exceeds 32 bits")
                        .signed(draw.base_vertex)
                })?;
                commands.draw_indexed_primitives(
                    &mut self.device.resolver(),
                    primitive(draw.primitive)?,
                    count(draw.index_count)?,
                    index_type(draw.index_type)?,
                    index_buffer,
                    draw.index_buffer_offset,
                    count(draw.instance_count)?,
                    base_vertex,
                    count(draw.base_instance)?,
                )?;
            }
            RenderCode::DrawIndirect => {
                let draw: DrawIndirect = payload.record()?;
                let indirect = self.buffer_ref(connection, draw.buffer)?;
                commands.draw_primitives_indirect(
                    &mut self.device.resolver(),
                    primitive(draw.primitive)?,
                    indirect,
                    draw.offset,
                )?;
            }
            RenderCode::DrawIndexedIndirect => {
                let draw: DrawIndexedIndirect = payload.record()?;
                let index_buffer = self.buffer_ref(connection, draw.index_buffer)?;
                let indirect = self.buffer_ref(connection, draw.buffer)?;
                commands.draw_indexed_primitives_indirect(
                    &mut self.device.resolver(),
                    primitive(draw.primitive)?,
                    index_type(draw.index_type)?,
                    index_buffer,
                    draw.index_buffer_offset,
                    indirect,
                    draw.offset,
                )?;
            }
            RenderCode::SetTessellationFactorBuffer => {
                let FactorBuffer {
                    buffer,
                    offset,
                    instance_stride,
                } = payload.record()?;
                let buffer = self.optional_buffer(connection, buffer)?;
                commands.set_tessellation_factor_buffer(buffer, offset, instance_stride)?;
            }
            RenderCode::SetTessellationFactorScale => {
                let FactorScale { scale, reserved: _ } = payload.record()?;
                commands.set_tessellation_factor_scale(scale)?;
            }
            RenderCode::DrawPatches => {
                let draw: DrawPatches = payload.record()?;
                let source = self.patch_source(connection, &draw.source)?;
                commands.draw_patches(
                    &mut self.device.resolver(),
                    &source,
                    count(draw.patch_start)?,
                    count(draw.patch_count)?,
                    count(draw.instance_count)?,
                    count(draw.base_instance)?,
                )?;
            }
            RenderCode::DrawPatchesIndirect => {
                let draw: DrawPatchesIndirect = payload.record()?;
                let source = self.patch_source(connection, &draw.source)?;
                let indirect = self.buffer_ref(connection, draw.buffer)?;
                commands.draw_patches_indirect(
                    &mut self.device.resolver(),
                    &source,
                    indirect,
                    draw.offset,
                )?;
            }
            RenderCode::DrawMeshThreadgroups => {
                let draw: DrawMesh = payload.record()?;
                commands.draw_mesh_threadgroups(
                    &mut self.device.resolver(),
                    grid(draw.size)?,
                    grid(draw.object_threads)?,
                    grid(draw.mesh_threads)?,
                )?;
            }
            RenderCode::DrawMeshThreads => {
                let draw: DrawMesh = payload.record()?;
                commands.draw_mesh_threads(
                    &mut self.device.resolver(),
                    grid(draw.size)?,
                    grid(draw.object_threads)?,
                    grid(draw.mesh_threads)?,
                )?;
            }
            RenderCode::DrawMeshIndirect => {
                let draw: DrawMeshIndirect = payload.record()?;
                let indirect = self.buffer_ref(connection, draw.buffer)?;
                commands.draw_mesh_threadgroups_indirect(
                    &mut self.device.resolver(),
                    indirect,
                    draw.offset,
                    grid(draw.object_threads)?,
                    grid(draw.mesh_threads)?,
                )?;
            }
        }
        Ok(())
    }

    /// The `MTLRenderPassDescriptor` of `record` and its data: sample
    /// positions, then counter samples.
    fn render_pass(
        &self,
        connection: NonZeroU64,
        record: &BeginRender,
        data: &[u8],
    ) -> Result<RenderPassRequest, Fault> {
        let positions = record.sample_position_count as usize * size_of::<[f32; 2]>();
        let (positions, samples) = data.split_at_checked(positions).ok_or(Fault::MALFORMED)?;
        let samples = wire::read_array::<CounterSampleRecord>(samples)?;
        if samples.len() != record.counter_sample_count as usize {
            return Err(Fault::MALFORMED);
        }
        let mut color_attachments = [None; MAX_COLOR_ATTACHMENTS];
        for (slot, color) in color_attachments.iter_mut().zip(&record.color_attachments) {
            *slot = self.attachment(connection, &color.attachment, color.clear_color)?;
        }
        let (depth, stencil) = (&record.depth_attachment, &record.stencil_attachment);
        Ok(RenderPassRequest {
            color_attachments,
            depth_attachment: self.attachment(connection, &depth.attachment, depth.clear_depth)?,
            depth_resolve_filter: decode(
                MultisampleDepthResolveFilter::from_raw,
                depth.resolve_filter,
                "depth resolve filter",
            )?,
            stencil_attachment: self.attachment(
                connection,
                &stencil.attachment,
                stencil.clear_stencil,
            )?,
            stencil_resolve_filter: decode(
                MultisampleStencilResolveFilter::from_raw,
                stencil.resolve_filter,
                "stencil resolve filter",
            )?,
            visibility_result_buffer: (record.visibility_result_buffer != 0)
                .then(|| self.object::<BufferId>(connection, record.visibility_result_buffer))
                .transpose()?,
            visibility_result_type: decode(
                VisibilityResultType::from_raw,
                record.visibility_result_type,
                "visibility result type",
            )?,
            render_target_array_length: record.render_target_array_length,
            render_target_width: record.render_target_width,
            render_target_height: record.render_target_height,
            default_raster_sample_count: record.default_raster_sample_count,
            tile_memory: TileMemory {
                width: record.tile_width,
                height: record.tile_height,
                threadgroup_memory_length: record.threadgroup_memory_length,
                imageblock_sample_length: record.imageblock_sample_length,
            },
            sample_positions: wire::read_array(positions)?,
            rasterization_rate_map: boolean(record.rasterization_rate_map)?,
            counter_samples: samples
                .iter()
                .map(|sample| self.counter_sample(connection, sample))
                .collect::<Result<_, _>>()?,
        })
    }

    /// The color attachments of a refused pass, for the platform's log:
    /// each texture and resolve texture as it was made, and the actions.
    fn describe_pass(&self, request: &RenderPassRequest) -> String {
        use core::fmt::Write;
        let texture = |id: TextureId| {
            self.device.texture_descriptor(id).map_or_else(
                |_| String::from("?"),
                |d| {
                    format!(
                        "{:?} {}x{}x{} samples={} storage={:?} usage={:?}",
                        d.pixel_format,
                        d.width,
                        d.height,
                        d.depth,
                        d.sample_count,
                        d.options.storage_mode,
                        d.usage
                    )
                },
            )
        };
        let mut text = String::from("PASS REFUSED");
        for (index, attachment) in request.color_attachments.iter().enumerate() {
            let Some(attachment) = attachment else {
                continue;
            };
            let _ = write!(
                text,
                " color{index}=[{} load={:?} store={:?}",
                texture(attachment.texture),
                attachment.load_action,
                attachment.store_action
            );
            if let Some((resolve, ..)) = attachment.resolve {
                let _ = write!(text, " resolve={}", texture(resolve));
            }
            text.push(']');
        }
        text
    }

    /// An attachment of a pass, or none without a texture.
    fn attachment<Clear>(
        &self,
        connection: NonZeroU64,
        record: &PassAttachmentRecord,
        clear: Clear,
    ) -> Result<Option<AttachmentRequest<Clear>>, Fault> {
        if record.texture == 0 {
            return Ok(None);
        }
        let resolve = (record.resolve_texture != 0)
            .then(|| {
                Ok::<_, Fault>((
                    self.object::<TextureId>(connection, record.resolve_texture)?,
                    record.resolve_level,
                    record.resolve_slice,
                    record.resolve_depth_plane,
                ))
            })
            .transpose()?;
        Ok(Some(AttachmentRequest {
            texture: self.object::<TextureId>(connection, record.texture)?,
            level: record.level,
            slice: record.slice,
            depth_plane: record.depth_plane,
            resolve,
            load_action: decode(LoadAction::from_raw, record.load_action, "load action")?,
            store_action: store_action(record.store_action)?,
            store_action_options: store_options(record.store_action_options)?,
            clear,
            surface: (record.surface != 0).then_some((record.surface, record.seed)),
        }))
    }

    fn counter_sample(
        &self,
        connection: NonZeroU64,
        record: &CounterSampleRecord,
    ) -> Result<(CounterBufferId, u32, PassBoundary), Fault> {
        let boundary = match decode(PassBoundaryCode::from_raw, record.boundary, "pass boundary")? {
            PassBoundaryCode::StartOfVertex => PassBoundary::StartOfVertex,
            PassBoundaryCode::EndOfVertex => PassBoundary::EndOfVertex,
            PassBoundaryCode::StartOfFragment => PassBoundary::StartOfFragment,
            PassBoundaryCode::EndOfFragment => PassBoundary::EndOfFragment,
        };
        let index = u32::try_from(record.index)
            .map_err(|_| Fault::malformed("sample index").value(record.index))?;
        let buffer = self.object::<CounterBufferId>(connection, record.buffer)?;
        Ok((buffer, index, boundary))
    }
}

fn store_action(raw: u64) -> Result<StoreAction, Fault> {
    decode(StoreAction::from_raw, raw, "store action")
}

fn store_options(raw: u64) -> Result<StoreActionOptions, Fault> {
    decode_bits(StoreActionOptions::from_bits, raw, "store action options")
}
