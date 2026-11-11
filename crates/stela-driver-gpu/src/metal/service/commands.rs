//! `Commit`: replays a connection's command stream into a command buffer
//! and submits it. A command the encoders refuse discards the command
//! buffer, as Metal's validation would, and the commit answers why.
use super::{
    Call, Service, boolean, decode,
    wire::{
        self, Empty, Fault, Handle, Record, Status, Value,
        commands::{
            BlitCode, BufferLayoutRecord, CommandBufferCode, CommandCode, CommandHeader,
            CommitRequest, ComputeCode, DepthStencilRecord, EventCommand, PatchSourceRecord,
            RenderCode, SampleCounters, StencilRecord,
        },
    },
};
use crate::metal::{
    abi::{CompareFunction, IndexType, PrimitiveType, StencilOperation},
    binding::ResourceId,
    capabilities::Size,
    command_buffer::{
        BufferRef, CommandBuffer,
        blit::{BlitTexture, BufferLayout},
        render::PatchSource,
    },
    device::{BufferId, SamplerId, TextureId},
    error::Error,
    pipeline::{DepthStencilDescriptor, DepthStencilState, StencilDescriptor, depth_stencil},
    shader::Compiler,
    timing::{Detail, Phase, Timed},
    vulkan::backend::Backend,
};
use alloc::vec::Vec;
use core::num::NonZeroU64;

mod blit;
mod compute;
mod encode;
mod indirect;
mod render;
mod resource_state;

/// The entries of a stream: of commands a command buffer replays, or of
/// encoding calls for an indirect command buffer. `decode` names their
/// codes.
pub(super) struct Stream<'a, Code> {
    bytes: &'a [u8],
    decode: fn(u32) -> Option<Code>,
}

impl<'a, Code> Stream<'a, Code> {
    pub(super) fn new(bytes: &'a [u8], decode: fn(u32) -> Option<Code>) -> Self {
        Self { bytes, decode }
    }

    fn entry(&mut self) -> Result<(Code, Payload<'a>), Fault> {
        const HEADER: usize = core::mem::size_of::<CommandHeader>();
        let left = self.bytes.len() as u64;
        let header: CommandHeader = wire::read(
            self.bytes
                .get(..HEADER)
                .ok_or(Fault::malformed("command header").length(left, HEADER as u64))?,
        )?;
        let end = HEADER + header.length as usize;
        let entry = self.bytes.get(..end.next_multiple_of(8)).ok_or(
            Fault::malformed("command entry").length(left, end.next_multiple_of(8) as u64),
        )?;
        self.bytes = &self.bytes[entry.len()..];
        let code = (self.decode)(header.code)
            .ok_or(Fault::new(Status::UnknownOperation, "unknown command").value(header.code))?;
        Ok((code, Payload(&entry[HEADER..end])))
    }
}

impl<'a, Code> Iterator for Stream<'a, Code> {
    type Item = Result<(Code, Payload<'a>), Fault>;

    fn next(&mut self) -> Option<Self::Item> {
        (!self.bytes.is_empty()).then(|| self.entry())
    }
}

/// An entry's record and the variable data after it.
#[derive(Clone, Copy)]
pub(super) struct Payload<'a>(&'a [u8]);

impl<'a> Payload<'a> {
    /// The entry's record, with no data after it.
    pub(super) fn record<T: Record>(self) -> Result<T, Fault> {
        wire::read(self.0)
    }

    /// The records filling the command's data, which has no record of its
    /// own.
    pub(super) fn list<T: Record>(self) -> Result<Vec<T>, Fault> {
        wire::read_array(self.0)
    }

    /// The entry's record and the data after it.
    pub(super) fn split<T: Record>(self) -> Result<(T, Payload<'a>), Fault> {
        let (record, data) = self
            .0
            .split_at_checked(core::mem::size_of::<T>())
            .ok_or(Fault::MALFORMED)?;
        Ok((wire::read(record)?, Payload(data)))
    }

    pub(super) fn bytes(self) -> &'a [u8] {
        self.0
    }
}

impl<B: Backend, C: Compiler> Service<B, C> {
    pub(super) fn commit(&mut self, call: &mut Call) -> Result<(), Fault> {
        self.device.timing().begin(Timed::Commit);
        let CommitRequest {
            commands: span,
            queue,
        } = call.request()?;
        let connection = call.connection;
        let mut commands = self.device.new_command_buffer();
        let start = self.device.timing().now();
        let stream_bytes = call.memory.read(span)?;
        let start = self.device.timing().since(Phase::CommitRead, start);
        self.device.timing().detail(
            Timed::Commit,
            Detail::Commit {
                stream: stream_bytes.len() as u64,
                uploads: 0,
            },
        );
        let mut stream = Stream::new(&stream_bytes, CommandCode::from_raw);
        let replayed = stream.try_for_each(|entry| {
            let (code, payload) = entry?;
            self.replay(connection, &mut commands, code, payload)
        });
        self.device.timing().since(Phase::CommitReplay, start);
        if let Err(fault) = replayed {
            self.device.discard(commands);
            // A deferred compilation runs the commit again, which submits
            // the command buffer with its events: signaling them now would
            // wake their waiters before its work.
            if !self.device.deferred() {
                self.commit_events_only(connection, queue, &stream_bytes);
            }
            return Err(fault);
        }
        let value = self.device.commit_on(connection.get(), queue, commands)?;
        call.reply(&Value { value })
    }

    /// A command buffer that fails to replay does none of its work, but
    /// still waits for and signals its events in order, as a Metal command
    /// buffer that errors does: work queued behind its events must not wait
    /// forever. What cannot be replayed even so is dropped.
    fn commit_events_only(&mut self, connection: NonZeroU64, queue: u64, stream_bytes: &[u8]) {
        let mut events = self.device.new_command_buffer();
        let mut count = 0;
        let mut stream = Stream::new(stream_bytes, CommandCode::from_raw);
        let replayed = stream.try_for_each(|entry| {
            let (code, payload) = entry?;
            match code {
                CommandCode::CommandBuffer(
                    code @ (CommandBufferCode::SignalEvent | CommandBufferCode::WaitForEvent),
                ) => {
                    count += 1;
                    self.command_buffer(connection, &mut events, code, payload)
                }
                _ => Ok(()),
            }
        });
        // One with no events submits nothing.
        match replayed {
            Ok(()) if count > 0 => {
                let _ = self.device.commit_on(connection.get(), queue, events);
            }
            _ => self.device.discard(events),
        }
    }

    fn replay(
        &mut self,
        connection: NonZeroU64,
        commands: &mut CommandBuffer,
        code: CommandCode,
        payload: Payload,
    ) -> Result<(), Fault> {
        if self.device.is_hosted_lavapipe() && !hosted_command(code) {
            return Err(Fault::new(
                Status::Unsupported,
                "command on hosted lavapipe",
            ));
        }
        match code {
            CommandCode::CommandBuffer(code) => {
                self.command_buffer(connection, commands, code, payload)
            }
            CommandCode::Compute(code) => self.compute(connection, commands, code, payload),
            CommandCode::Blit(code) => self.blit(connection, commands, code, payload),
            CommandCode::Render(code) => self.render(connection, commands, code, payload),
            CommandCode::ResourceState(code) => {
                self.resource_state(connection, commands, code, payload)
            }
            CommandCode::Indirect(code) => self.indirect(connection, commands, code, payload),
        }
    }

    fn command_buffer(
        &mut self,
        connection: NonZeroU64,
        commands: &mut CommandBuffer,
        code: CommandBufferCode,
        payload: Payload,
    ) -> Result<(), Fault> {
        match code {
            CommandBufferCode::EndEncoding => {
                payload.record::<Empty>()?;
                commands.end_encoding(&mut self.device.resolver())?;
            }
            CommandBufferCode::SignalEvent | CommandBufferCode::WaitForEvent => {
                let EventCommand { event, value } = payload.record()?;
                let event = self
                    .device
                    .event_semaphore(self.object(connection, event)?)?;
                let resolver = &mut self.device.resolver();
                match code {
                    CommandBufferCode::SignalEvent => {
                        commands.encode_signal_event(resolver, event, value)
                    }
                    _ => commands.encode_wait_for_event(resolver, event, value),
                }?;
            }
            CommandBufferCode::LogBuffer => {
                let Handle { handle } = payload.record()?;
                commands.set_log_buffer(self.buffer_ref(connection, handle)?);
            }
            CommandBufferCode::SampleCounters => {
                let SampleCounters {
                    buffer,
                    index,
                    barrier,
                    reserved: _,
                } = payload.record()?;
                let index = u32::try_from(index)
                    .map_err(|_| Fault::malformed("sample index").value(index))?;
                let sample = self
                    .device
                    .counter_sample(self.object(connection, buffer)?, index)?;
                let barrier = boolean(barrier)?;
                commands.sample_counters(&mut self.device.resolver(), sample, barrier)?;
            }
        }
        Ok(())
    }

    fn patch_source(
        &self,
        connection: NonZeroU64,
        record: &PatchSourceRecord,
    ) -> Result<PatchSource, Fault> {
        let indices = |buffer, offset| {
            Ok::<_, Fault>(
                self.optional_buffer(connection, buffer)?
                    .map(|buffer| (buffer, offset)),
            )
        };
        Ok(PatchSource {
            control_points: u32::try_from(record.control_points).map_err(|_| {
                Fault::malformed("patch control points").value(record.control_points)
            })?,
            patch_indices: indices(record.patch_index_buffer, record.patch_index_offset)?,
            control_point_indices: indices(
                record.control_point_index_buffer,
                record.control_point_index_offset,
            )?,
        })
    }

    fn buffer_ref(&self, connection: NonZeroU64, handle: u64) -> Result<BufferRef, Fault> {
        let buffer = self.object::<BufferId>(connection, handle)?;
        if self.device.is_guest_copy_buffer(buffer)? {
            return Err(Error::Unsupported("direct access to a copied guest buffer").into());
        }
        Ok(self.device.buffer(buffer)?)
    }

    /// The buffer `handle` names, or none for nil.
    fn optional_buffer(
        &self,
        connection: NonZeroU64,
        handle: u64,
    ) -> Result<Option<BufferRef>, Fault> {
        (handle != 0)
            .then(|| self.buffer_ref(connection, handle))
            .transpose()
    }

    fn buffer_layout(
        &self,
        connection: NonZeroU64,
        record: BufferLayoutRecord,
    ) -> Result<BufferLayout, Fault> {
        Ok(BufferLayout {
            buffer: self.buffer_ref(connection, record.buffer)?,
            offset: record.offset,
            bytes_per_row: record.bytes_per_row,
            bytes_per_image: record.bytes_per_image,
        })
    }

    /// The `gpuResourceID` of the texture `handle` names, null for nil.
    fn texture_id(&self, connection: NonZeroU64, handle: u64) -> Result<ResourceId, Fault> {
        if handle == 0 {
            return Ok(ResourceId::NULL);
        }
        let texture = self.object::<TextureId>(connection, handle)?;
        if self.device.is_staged_buffer_texture(texture)? {
            return Err(Error::Unsupported("shader bindings of host buffer render targets").into());
        }
        Ok(self.device.texture_resource_id(texture)?)
    }

    /// The `gpuResourceID` of the sampler `handle` names, null for nil.
    fn sampler_id(&self, connection: NonZeroU64, handle: u64) -> Result<ResourceId, Fault> {
        if handle == 0 {
            return Ok(ResourceId::NULL);
        }
        Ok(self
            .device
            .sampler_resource_id(self.object::<SamplerId>(connection, handle)?)?)
    }

    fn blit_texture(&self, connection: NonZeroU64, handle: u64) -> Result<BlitTexture<'_>, Fault> {
        let id = self.object::<TextureId>(connection, handle)?;
        Ok(BlitTexture {
            image: self.device.texture_image(id)?,
            descriptor: self.device.texture_descriptor(id)?,
            storage: ResourceId::NULL,
            blocks: self.device.texture_blocks(id)?,
        })
    }
}

fn hosted_command(code: CommandCode) -> bool {
    matches!(
        code,
        CommandCode::CommandBuffer(
            CommandBufferCode::EndEncoding
                | CommandBufferCode::SignalEvent
                | CommandBufferCode::WaitForEvent
        ) | CommandCode::Compute(
            ComputeCode::Begin
                | ComputeCode::SetPipeline
                | ComputeCode::SetBuffer
                | ComputeCode::SetBufferOffset
                | ComputeCode::SetBytes
                | ComputeCode::DispatchThreadgroups
                | ComputeCode::DispatchThreads
        ) | CommandCode::Blit(
            BlitCode::Begin
                | BlitCode::CopyBuffer
                | BlitCode::CopyBufferToTexture
                | BlitCode::CopyTextureToBuffer
                | BlitCode::FillBuffer
        ) | CommandCode::Render(
            RenderCode::Begin
                | RenderCode::SetPipeline
                | RenderCode::SetBuffer
                | RenderCode::SetBufferOffset
                | RenderCode::SetBytes
                | RenderCode::SetViewports
                | RenderCode::SetScissorRects
                | RenderCode::SetColorStoreAction
                | RenderCode::DrawPrimitives
        )
    )
}

/// An argument table index.
fn binding(index: u64) -> Result<u32, Fault> {
    u32::try_from(index).map_err(|_| Fault::malformed("binding index").value(index))
}

/// An `MTLSize` of threads in a threadgroup.
fn size([width, height, depth]: [u64; 3]) -> Size {
    Size {
        width,
        height,
        depth,
    }
}

/// An `MTLSize` of a grid, whose dimensions the GPU counts in 32 bits.
fn grid(size: [u64; 3]) -> Result<[u32; 3], Fault> {
    let mut grid = [0; 3];
    for (out, size) in grid.iter_mut().zip(size) {
        *out = u32::try_from(size).map_err(|_| {
            Fault::new(Status::Invalid, "a grid dimension exceeds 32 bits").value(size)
        })?;
    }
    Ok(grid)
}

/// A draw's count, start or base, which the GPU takes in 32 bits.
fn count(value: u64) -> Result<u32, Fault> {
    u32::try_from(value)
        .map_err(|_| Fault::new(Status::Invalid, "a draw argument exceeds 32 bits").value(value))
}

fn primitive(raw: u64) -> Result<PrimitiveType, Fault> {
    decode(PrimitiveType::from_raw, raw, "primitive type")
}

fn index_type(raw: u64) -> Result<IndexType, Fault> {
    decode(IndexType::from_raw, raw, "index type")
}

/// The state of a depth-stencil state object made from `record`.
fn depth_stencil_state(record: &DepthStencilRecord) -> Result<DepthStencilState, Fault> {
    Ok(depth_stencil(&DepthStencilDescriptor {
        depth_compare_function: compare_function(record.depth_compare_function)?,
        depth_write_enabled: boolean(record.depth_write_enabled)?,
        front_face_stencil: stencil(&record.front_face_stencil)?,
        back_face_stencil: stencil(&record.back_face_stencil)?,
    }))
}

fn stencil(record: &StencilRecord) -> Result<StencilDescriptor, Fault> {
    let operation = |raw| decode(StencilOperation::from_raw, raw, "stencil operation");
    Ok(StencilDescriptor {
        stencil_compare_function: compare_function(record.compare_function)?,
        stencil_failure_operation: operation(record.stencil_failure_operation)?,
        depth_failure_operation: operation(record.depth_failure_operation)?,
        depth_stencil_pass_operation: operation(record.depth_stencil_pass_operation)?,
        read_mask: record.read_mask,
        write_mask: record.write_mask,
    })
}

fn compare_function(raw: u64) -> Result<CompareFunction, Fault> {
    decode(CompareFunction::from_raw, raw, "compare function")
}
