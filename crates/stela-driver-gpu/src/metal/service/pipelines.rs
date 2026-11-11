//! Render, mesh and compute pipelines, their function handles and visible
//! function tables.
use super::{
    Call, GuestMemory, Object, Service, boolean, decode, decode_bits,
    resources::pixel_format,
    wire::{
        self, AttributeRecord, ColorAttachmentRecord, ComputePipelineResponse, Empty, Fault,
        FunctionHandleRequest, FunctionHandleResponse, LayoutRecord, LinkedFunctionsRecord,
        NewComputePipelineRequest, NewFunctionTableRequest, NewMeshPipelineRequest,
        NewRenderPipelineRequest, NewTilePipelineRequest, PipelineArchivesRecord, PipelineFlags,
        RenderPipelineResponse, RenderTargetsRecord, SetFunctionsRequest, Status,
        TessellationRecord,
    },
};
use crate::metal::{
    abi::{
        BlendFactor, BlendOperation, ColorWriteMask, IndexType, PrimitiveTopologyClass,
        TessellationControlPointIndexType, TessellationFactorFormat,
        TessellationFactorStepFunction, TessellationPartitionMode, Winding,
    },
    device::{
        ArchiveId, BufferId, FunctionId, LibraryId, RenderPipelineId,
        pipelines::{
            ComputePipelineRequest, FunctionHandle, LinkedFunctions, MeshPipelineRequest,
            PipelineArchives, RenderPipelineRequest,
        },
    },
    pipeline::{
        self, ColorAttachmentDescriptor, MAX_COLOR_ATTACHMENTS, RenderTargets, StageThreads,
        TessellationDescriptor,
    },
    pixel_format::PixelFormat,
    shader::{Compiler, Stage},
    timing::Timed,
    vertex::{
        Attribute, Layout, StageInLayout, StageInStep, StageInputDescriptor, StepFunction,
        VertexDescriptor, VertexFormat,
    },
    vulkan::backend::Backend,
};
use alloc::{format, string::String, vec::Vec};
use core::num::NonZeroU64;

impl<B: Backend, C: Compiler> Service<B, C> {
    pub(super) fn new_render_pipeline(&mut self, call: &mut Call) -> Result<(), Fault> {
        self.device.timing().begin(Timed::NewRenderPipeline);
        let request: NewRenderPipelineRequest = call.request()?;
        let flags = flags(request.flags)?;
        if self.device.is_hosted_lavapipe()
            && (!flags.is_empty()
                || request.targets.raster_sample_count != 1
                || request.targets.alpha_to_coverage_enabled != 0
                || request.targets.alpha_to_one_enabled != 0
                || request.targets.depth_attachment_pixel_format != 0
                || request.targets.stencil_attachment_pixel_format != 0
                || !request.targets.color_attachments[1..]
                    .iter()
                    .all(|color| color.pixel_format == 0)
                || !hosted_pipeline_inputs(
                    &request.archives,
                    &[
                        request.vertex_linked_functions,
                        request.fragment_linked_functions,
                    ],
                ))
        {
            return Err(Fault::new(
                Status::Unsupported,
                "render pipeline on hosted lavapipe",
            ));
        }
        let connection = call.connection;
        let memory = &mut *call.memory;
        let pipeline = RenderPipelineRequest {
            vertex_function: self.object(connection, request.vertex_function)?,
            fragment_function: self.optional(connection, request.fragment_function)?,
            vertex_descriptor: flags
                .contains(PipelineFlags::DESCRIBED_INPUT)
                .then(|| vertex_descriptor(&request.vertex_descriptor))
                .transpose()?,
            targets: render_targets(&request.targets)?,
            input_primitive_topology: decode(
                PrimitiveTopologyClass::from_raw,
                request.input_primitive_topology,
                "primitive topology class",
            )?,
            max_vertex_amplification_count: request.max_vertex_amplification_count,
            tessellation: tessellation(&request.tessellation)?,
            vertex_linked_functions: self.linked(
                connection,
                &mut *memory,
                &request.vertex_linked_functions,
            )?,
            fragment_linked_functions: self.linked(
                connection,
                &mut *memory,
                &request.fragment_linked_functions,
            )?,
            support_indirect_command_buffers: flags
                .contains(PipelineFlags::INDIRECT_COMMAND_BUFFERS),
            archives: self.archives(connection, &mut *memory, &request.archives)?,
        };
        self.noted = Some(self.describe_render_pipeline(&pipeline));
        let pipeline = self.device.new_render_pipeline(&pipeline)?;
        self.adopt_render_pipeline(call, pipeline)
    }

    /// A render pipeline's functions and vertex fetch, for the platform's
    /// log: `PIPELINE vertex=… fragment=… [vertex_linked=[…]]
    /// [fragment_linked=[…]] attributes=[index:format@offset/buffer …]
    /// layouts=[buffer:stride/step/rate …]`. The linked lists name the
    /// visible functions a stage's function tables and references reach,
    /// such as a framework's custom shaders.
    fn describe_render_pipeline(&self, pipeline: &RenderPipelineRequest) -> String {
        use core::fmt::Write;
        let entry = |id: FunctionId| self.device.function_entry(id).unwrap_or("?");
        let mut text = format!("PIPELINE vertex={}", entry(pipeline.vertex_function));
        let fragment = pipeline.fragment_function.map_or("-", entry);
        let _ = write!(text, " fragment={fragment}");
        for (stage, linked) in [
            ("vertex", &pipeline.vertex_linked_functions),
            ("fragment", &pipeline.fragment_linked_functions),
        ] {
            let mut names = linked.functions.iter().chain(&linked.private_functions);
            if let Some(&first) = names.next() {
                let _ = write!(text, " {stage}_linked=[{}", entry(first));
                for &id in names {
                    let _ = write!(text, ",{}", entry(id));
                }
                text.push(']');
            }
        }
        if let Some(vertex) = &pipeline.vertex_descriptor {
            text.push_str(" attributes=[");
            for attribute in &vertex.attributes {
                let _ = write!(
                    text,
                    " {}:{:?}@{}/{}",
                    attribute.index, attribute.format, attribute.offset, attribute.buffer_index
                );
            }
            text.push_str(" ] layouts=[");
            for layout in &vertex.layouts {
                let _ = write!(
                    text,
                    " {}:{}/{:?}/{}",
                    layout.buffer_index, layout.stride, layout.step_function, layout.step_rate
                );
            }
            text.push_str(" ]");
        }
        text
    }

    pub(super) fn new_mesh_pipeline(&mut self, call: &mut Call) -> Result<(), Fault> {
        let request: NewMeshPipelineRequest = call.request()?;
        let flags = flags(request.flags)?;
        let connection = call.connection;
        let memory = &mut *call.memory;
        let threads = |max_total_threads, multiple| StageThreads {
            max_total_threads,
            multiple_of_width: flags.contains(multiple),
        };
        let pipeline = MeshPipelineRequest {
            object_function: self.optional(connection, request.object_function)?,
            mesh_function: self.object(connection, request.mesh_function)?,
            fragment_function: self.optional(connection, request.fragment_function)?,
            targets: render_targets(&request.targets)?,
            max_vertex_amplification_count: request.max_vertex_amplification_count,
            object_threads: threads(
                request.max_total_threads_per_object_threadgroup,
                PipelineFlags::OBJECT_THREADGROUP_SIZE_IS_MULTIPLE_OF_WIDTH,
            ),
            mesh_threads: threads(
                request.max_total_threads_per_mesh_threadgroup,
                PipelineFlags::MESH_THREADGROUP_SIZE_IS_MULTIPLE_OF_WIDTH,
            ),
            payload_memory_length: request.payload_memory_length,
            object_linked_functions: self.linked(
                connection,
                &mut *memory,
                &request.object_linked_functions,
            )?,
            mesh_linked_functions: self.linked(
                connection,
                &mut *memory,
                &request.mesh_linked_functions,
            )?,
            fragment_linked_functions: self.linked(
                connection,
                &mut *memory,
                &request.fragment_linked_functions,
            )?,
            archives: self.archives(connection, &mut *memory, &request.archives)?,
        };
        let pipeline = self.device.new_mesh_render_pipeline(&pipeline)?;
        self.adopt_render_pipeline(call, pipeline)
    }

    pub(super) fn new_compute_pipeline(&mut self, call: &mut Call) -> Result<(), Fault> {
        let request: NewComputePipelineRequest = call.request()?;
        let flags = flags(request.flags)?;
        if self.device.is_hosted_lavapipe()
            && (!flags.is_empty()
                || !hosted_pipeline_inputs(&request.archives, &[request.linked_functions]))
        {
            return Err(Fault::new(
                Status::Unsupported,
                "compute pipeline on hosted lavapipe",
            ));
        }
        let connection = call.connection;
        let pipeline = ComputePipelineRequest {
            compute_function: self.object(connection, request.compute_function)?,
            threadgroup_size_is_multiple_of_thread_execution_width: flags
                .contains(PipelineFlags::THREADGROUP_SIZE_IS_MULTIPLE_OF_WIDTH),
            max_total_threads_per_threadgroup: request.max_total_threads_per_threadgroup,
            stage_input: flags
                .contains(PipelineFlags::DESCRIBED_INPUT)
                .then(|| stage_input(&request))
                .transpose()?,
            linked_functions: self.linked(
                connection,
                &mut *call.memory,
                &request.linked_functions,
            )?,
            support_indirect_command_buffers: flags
                .contains(PipelineFlags::INDIRECT_COMMAND_BUFFERS),
            archives: self.archives(connection, &mut *call.memory, &request.archives)?,
        };
        let pipeline = self.device.new_compute_pipeline(&pipeline)?;
        let object = Object::ComputePipeline(pipeline);
        let (handle, response) = self.adopt(connection, object, |device| {
            let plan = &device.compute_pipeline_state(pipeline)?.plan;
            Ok(ComputePipelineResponse {
                handle: 0,
                resource_id: device.compute_pipeline_resource_id(pipeline)?,
                max_total_threads_per_threadgroup: plan.max_total_threads_per_threadgroup,
                thread_execution_width: plan.thread_execution_width,
                static_threadgroup_memory_length: plan.static_threadgroup_memory_length,
                imageblock_bytes: plan.imageblock_bytes.into(),
            })
        })?;
        call.reply(&ComputePipelineResponse {
            handle: handle.handle,
            ..response
        })
    }

    pub(super) fn new_tile_pipeline(&mut self, call: &mut Call) -> Result<(), Fault> {
        let request: NewTilePipelineRequest = call.request()?;
        let flags = flags(request.flags)?;
        let connection = call.connection;
        let mut formats = [None; MAX_COLOR_ATTACHMENTS];
        for (format, &raw) in formats.iter_mut().zip(&request.color_formats) {
            if raw != 0 {
                *format = Some(pixel_format(raw)?);
            }
        }
        let target = pipeline::tile_target(
            self.device.capabilities(),
            &formats,
            request.raster_sample_count,
            flags.contains(PipelineFlags::THREADGROUP_SIZE_MATCHES_TILE_SIZE),
        )?;
        let pipeline = ComputePipelineRequest {
            compute_function: self.object(connection, request.tile_function)?,
            threadgroup_size_is_multiple_of_thread_execution_width: false,
            max_total_threads_per_threadgroup: request.max_total_threads_per_threadgroup,
            stage_input: None,
            linked_functions: self.linked(
                connection,
                &mut *call.memory,
                &request.linked_functions,
            )?,
            support_indirect_command_buffers: false,
            archives: self.archives(connection, &mut *call.memory, &request.archives)?,
        };
        let function = pipeline.compute_function;
        let pipeline = match self.device.new_tile_pipeline(&pipeline, target) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                // The gate logs the fault; this names the pipeline it was.
                let name = self.device.function_name(function).unwrap_or("?");
                self.refused.push(format!(
                    "tile pipeline of {name}: maxTotalThreadsPerThreadgroup {}, colors {:?}, \
                     samples {}, threadgroupSizeMatchesTileSize {}",
                    request.max_total_threads_per_threadgroup,
                    request.color_formats,
                    request.raster_sample_count,
                    target.matches_tile_size,
                ));
                return Err(error.into());
            }
        };
        let object = Object::ComputePipeline(pipeline);
        let (handle, response) = self.adopt(connection, object, |device| {
            let plan = &device.compute_pipeline_state(pipeline)?.plan;
            Ok(ComputePipelineResponse {
                handle: 0,
                resource_id: 0,
                max_total_threads_per_threadgroup: plan.max_total_threads_per_threadgroup,
                thread_execution_width: plan.thread_execution_width,
                static_threadgroup_memory_length: plan.static_threadgroup_memory_length,
                imageblock_bytes: plan.imageblock_bytes.into(),
            })
        })?;
        call.reply(&ComputePipelineResponse {
            handle: handle.handle,
            ..response
        })
    }

    pub(super) fn function_handle(&mut self, call: &mut Call) -> Result<(), Fault> {
        let request: FunctionHandleRequest = call.request()?;
        let function = self.object::<FunctionId>(call.connection, request.function)?;
        let handle = match (
            self.entry(call.connection, request.pipeline)?,
            request.stage,
        ) {
            (Object::ComputePipeline(pipeline), 0) => {
                self.device.compute_function_handle(pipeline, function)?
            }
            (Object::RenderPipeline(pipeline), stage) => {
                let stage = render_stage(stage)?;
                self.device
                    .render_function_handle(pipeline, stage, function)?
            }
            (Object::ComputePipeline(_), _) => {
                return Err(Fault::malformed("render stage of a compute pipeline"));
            }
            _ => return Err(Fault::STALE),
        };
        call.reply(&FunctionHandleResponse {
            value: handle.map_or(0, |handle| handle.0),
        })
    }

    pub(super) fn new_function_table(&mut self, call: &mut Call) -> Result<(), Fault> {
        let request: NewFunctionTableRequest = call.request()?;
        let table = self
            .device
            .new_visible_function_table(request.function_count)?;
        let response = self.adopt_buffer(call.connection, table)?;
        call.reply(&response)
    }

    pub(super) fn set_functions(&mut self, call: &mut Call) -> Result<(), Fault> {
        let request: SetFunctionsRequest = call.request()?;
        let table = self.object::<BufferId>(call.connection, request.table)?;
        let handles: Vec<Option<FunctionHandle>> =
            wire::read_array::<u64>(&call.memory.read(request.handles)?)?
                .into_iter()
                .map(|value| (value != 0).then_some(FunctionHandle(value)))
                .collect();
        self.device
            .set_visible_functions(table, request.index, &handles)?;
        call.reply(&Empty {})
    }

    fn adopt_render_pipeline(
        &mut self,
        call: &mut Call,
        pipeline: RenderPipelineId,
    ) -> Result<(), Fault> {
        let object = Object::RenderPipeline(pipeline);
        let (handle, (resource_id, mesh, imageblock)) =
            self.adopt(call.connection, object, |device| {
                let state = device.render_pipeline_state(pipeline)?;
                let mesh = state.plan.mesh;
                let imageblock = state
                    .interfaces
                    .iter()
                    .map(|interface| interface.imageblock_bytes)
                    .max()
                    .unwrap_or(0);
                Ok((
                    device.render_pipeline_resource_id(pipeline)?,
                    mesh,
                    imageblock,
                ))
            })?;
        let (object, mesh, width) = mesh.map_or((None, None, 0), |threads| {
            (
                threads.object,
                Some(threads.mesh),
                threads.thread_execution_width,
            )
        });
        let threads =
            |stage: Option<StageThreads>| stage.map_or(0, |stage| stage.max_total_threads);
        let widths = |stage: Option<StageThreads>| stage.map_or(0, |_| width);
        call.reply(&RenderPipelineResponse {
            handle: handle.handle,
            resource_id,
            max_total_threads_per_object_threadgroup: threads(object),
            max_total_threads_per_mesh_threadgroup: threads(mesh),
            object_thread_execution_width: widths(object),
            mesh_thread_execution_width: widths(mesh),
            imageblock_sample_length: u64::from(imageblock),
        })
    }

    /// The function a nonzero handle names.
    fn optional(&self, connection: NonZeroU64, handle: u64) -> Result<Option<FunctionId>, Fault> {
        (handle != 0)
            .then(|| self.object(connection, handle))
            .transpose()
    }

    fn archives(
        &self,
        connection: NonZeroU64,
        memory: &mut dyn GuestMemory,
        record: &PipelineArchivesRecord,
    ) -> Result<PipelineArchives, Fault> {
        let searched: Vec<u64> = wire::read_array(&memory.read(record.searched)?)?;
        Ok(PipelineArchives {
            searched: searched
                .into_iter()
                .map(|handle| self.object::<ArchiveId>(connection, handle))
                .collect::<Result<_, _>>()?,
            required: record.required != 0,
            added_to: (record.added_to != 0)
                .then(|| self.object(connection, record.added_to))
                .transpose()?,
        })
    }

    fn linked(
        &self,
        connection: NonZeroU64,
        memory: &mut dyn GuestMemory,
        record: &LinkedFunctionsRecord,
    ) -> Result<LinkedFunctions, Fault> {
        let mut objects =
            |span| -> Result<Vec<u64>, Fault> { wire::read_array(&memory.read(span)?) };
        let (functions, private_functions, libraries) = (
            objects(record.functions)?,
            objects(record.private_functions)?,
            objects(record.libraries)?,
        );
        let functions_of = |handles: Vec<u64>| {
            handles
                .into_iter()
                .map(|handle| self.object::<FunctionId>(connection, handle))
                .collect::<Result<Vec<_>, Fault>>()
        };
        Ok(LinkedFunctions {
            functions: functions_of(functions)?,
            private_functions: functions_of(private_functions)?,
            libraries: libraries
                .into_iter()
                .map(|handle| self.object::<LibraryId>(connection, handle))
                .collect::<Result<_, _>>()?,
            max_call_stack_depth: u32::try_from(record.max_call_stack_depth).map_err(|_| {
                Fault::malformed("call stack depth").value(record.max_call_stack_depth)
            })?,
        })
    }
}

fn hosted_pipeline_inputs(
    archives: &PipelineArchivesRecord,
    stages: &[LinkedFunctionsRecord],
) -> bool {
    archives.searched.length == 0
        && archives.added_to == 0
        && archives.required == 0
        && stages.iter().all(|linked| {
            linked.functions.length == 0
                && linked.private_functions.length == 0
                && linked.libraries.length == 0
        })
}

fn flags(raw: u64) -> Result<PipelineFlags, Fault> {
    decode_bits(PipelineFlags::from_bits, raw, "pipeline flags")
}

/// A pixel format, none for zero.
fn optional_format(raw: u64) -> Result<Option<PixelFormat>, Fault> {
    (raw != 0).then(|| pixel_format(raw)).transpose()
}

/// One bit of `MTLRenderStages`.
/// The stage of one `MTLRenderStages` bit.
pub(super) fn render_stage(raw: u64) -> Result<Stage, Fault> {
    match raw {
        1 => Ok(Stage::Vertex),
        2 => Ok(Stage::Fragment),
        4 => Err(Fault::new(Status::Unsupported, "tile functions")),
        8 => Ok(Stage::Object),
        16 => Ok(Stage::Mesh),
        _ => Err(Fault::malformed("render stage").bits(raw)),
    }
}

fn render_targets(record: &RenderTargetsRecord) -> Result<RenderTargets, Fault> {
    let mut targets = RenderTargets::default();
    for (attachment, record) in targets
        .color_attachments
        .iter_mut()
        .zip(&record.color_attachments)
    {
        *attachment = color_attachment(record)?;
    }
    Ok(RenderTargets {
        depth_attachment_pixel_format: optional_format(record.depth_attachment_pixel_format)?,
        stencil_attachment_pixel_format: optional_format(record.stencil_attachment_pixel_format)?,
        raster_sample_count: record.raster_sample_count,
        alpha_to_coverage_enabled: boolean(record.alpha_to_coverage_enabled)?,
        alpha_to_one_enabled: boolean(record.alpha_to_one_enabled)?,
        rasterization_enabled: boolean(record.rasterization_enabled)?,
        ..targets
    })
}

fn color_attachment(record: &ColorAttachmentRecord) -> Result<ColorAttachmentDescriptor, Fault> {
    let factor = |raw| decode(BlendFactor::from_raw, raw, "blend factor");
    let operation = |raw| decode(BlendOperation::from_raw, raw, "blend operation");
    Ok(ColorAttachmentDescriptor {
        pixel_format: optional_format(record.pixel_format)?,
        blending_enabled: boolean(record.blending_enabled)?,
        source_rgb_blend_factor: factor(record.source_rgb_blend_factor)?,
        destination_rgb_blend_factor: factor(record.destination_rgb_blend_factor)?,
        rgb_blend_operation: operation(record.rgb_blend_operation)?,
        source_alpha_blend_factor: factor(record.source_alpha_blend_factor)?,
        destination_alpha_blend_factor: factor(record.destination_alpha_blend_factor)?,
        alpha_blend_operation: operation(record.alpha_blend_operation)?,
        write_mask: decode_bits(
            ColorWriteMask::from_bits,
            record.write_mask,
            "color write mask",
        )?,
    })
}

fn tessellation(record: &TessellationRecord) -> Result<TessellationDescriptor, Fault> {
    Ok(TessellationDescriptor {
        partition_mode: decode(
            TessellationPartitionMode::from_raw,
            record.partition_mode,
            "tessellation partition mode",
        )?,
        max_tessellation_factor: record.max_tessellation_factor,
        factor_scale_enabled: boolean(record.factor_scale_enabled)?,
        factor_format: decode(
            TessellationFactorFormat::from_raw,
            record.factor_format,
            "tessellation factor format",
        )?,
        control_point_index_type: decode(
            TessellationControlPointIndexType::from_raw,
            record.control_point_index_type,
            "control point index type",
        )?,
        factor_step_function: decode(
            TessellationFactorStepFunction::from_raw,
            record.factor_step_function,
            "tessellation factor step function",
        )?,
        output_winding_order: decode(Winding::from_raw, record.output_winding_order, "winding")?,
    })
}

/// The attributes a descriptor sets, and the indices of the buffers they
/// read, each once in order.
fn attributes(records: &[AttributeRecord; 31]) -> Result<(Vec<Attribute>, Vec<u32>), Fault> {
    let mut attributes = Vec::new();
    let mut buffers = Vec::new();
    for (index, record) in (0u32..).zip(records) {
        if record.format == 0 {
            continue;
        }
        let buffer_index = u32::try_from(record.buffer_index)
            .ok()
            .filter(|&buffer| buffer < 31)
            .ok_or(Fault::malformed("attribute buffer index").value(record.buffer_index))?;
        attributes.push(Attribute {
            index,
            format: decode(VertexFormat::from_raw, record.format, "vertex format")?,
            offset: record.offset,
            buffer_index,
        });
        if !buffers.contains(&buffer_index) {
            buffers.push(buffer_index);
        }
    }
    buffers.sort_unstable();
    Ok((attributes, buffers))
}

fn vertex_descriptor(record: &wire::VertexDescriptorRecord) -> Result<VertexDescriptor, Fault> {
    let (attributes, buffers) = attributes(&record.attributes)?;
    let layouts = buffers
        .into_iter()
        .map(|buffer_index| {
            let LayoutRecord {
                stride,
                step_function,
                step_rate,
            } = record.layouts[buffer_index as usize];
            Ok(Layout {
                buffer_index,
                stride,
                step_function: decode(StepFunction::from_raw, step_function, "step function")?,
                step_rate,
            })
        })
        .collect::<Result<_, Fault>>()?;
    Ok(VertexDescriptor {
        attributes,
        layouts,
    })
}

fn stage_input(request: &NewComputePipelineRequest) -> Result<StageInputDescriptor, Fault> {
    let record = &request.stage_input;
    let (attributes, buffers) = attributes(&record.attributes)?;
    let layouts = buffers
        .into_iter()
        .map(|buffer_index| {
            let layout = record.layouts[buffer_index as usize];
            Ok(StageInLayout {
                buffer_index,
                stride: layout.stride,
                step_function: decode(
                    StageInStep::from_raw,
                    layout.step_function,
                    "stage-in step function",
                )?,
                step_rate: layout.step_rate,
            })
        })
        .collect::<Result<_, Fault>>()?;
    Ok(StageInputDescriptor {
        attributes,
        layouts,
        index_type: decode(IndexType::from_raw, record.index_type, "index type")?,
        index_buffer_index: u32::try_from(record.index_buffer_index)
            .map_err(|_| Fault::malformed("index buffer index").value(record.index_buffer_index))?,
    })
}
