//! `MTLCommandBuffer` recording: encoders lowered to Vulkan commands.
//!
//! A command buffer records into segments. Encoding a wait for an event
//! starts a new segment that waits on the event's timeline semaphore, and
//! encoding a signal ends the current one, so each segment is one
//! `VkSubmitInfo2` of one Vulkan command buffer. Every segment begins with a
//! full barrier, which orders it after everything submitted earlier on the
//! queue; the [`Hazards`] tracker adds barriers inside a segment.
//!
//! Encoder state is the command buffer's: one encoder is active at a time,
//! as in Metal. Its operations are in [`render`], [`compute`] and [`blit`].
use super::{
    binding::{
        ArgumentTable, BufferBinding, PassImageblock, PassSamples, PushConstants, ResourceId,
    },
    error::{Error, Result, ensure},
    hazard::{FULL, Hazards, Use},
    indirect::{IndirectCommandBuffer, IndirectCommandBufferId, IndirectPipeline, RecordLayout},
    pipeline::{ComputePipelinePlan, MAX_COLOR_ATTACHMENTS, RenderPipelinePlan},
    shader::{Interface, Specialization, Stage},
    sparse::Mapping,
    upload::{Chunk, ChunkSource, Upload, UploadArena},
    vulkan::{
        backend::Requirements,
        command::{Command, SparseBind},
        object::{
            Buffer, ExecutionSet, IndirectCommandsLayout, Memory, Pipeline, QueryPool, QueryType,
            Semaphore,
        },
    },
};
use alloc::{boxed::Box, sync::Arc, vec::Vec};

pub mod blit;
pub mod compute;
pub mod counters;
pub mod device_generated;
pub mod fragment_tests;
pub mod indirect;
pub mod render;
pub mod resource_state;
pub mod tile;

/// A Metal buffer as commands see it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferRef {
    pub buffer: Buffer,
    /// Where the Metal buffer starts in `buffer`: zero for buffers with
    /// their own `VkBuffer`, nonzero for upload suballocations.
    pub offset: u64,
    /// The device address of the Metal buffer's first byte.
    pub address: u64,
    pub length: u64,
}

/// The mapped bytes of a managed buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MappedRange {
    pub memory: Memory,
    pub offset: u64,
    pub length: u64,
}

impl BufferRef {
    /// `offset` within the Metal buffer as an offset within `buffer`, when
    /// `bytes` from there fit.
    pub fn at(&self, offset: u64, bytes: u64) -> Result<u64> {
        let end = offset.checked_add(bytes).filter(|&end| end <= self.length);
        end.map(|_| self.offset + offset)
            .ok_or(Error::Invalid("range extends past the buffer"))
    }
}

/// A render pipeline state object.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderPipelineState {
    /// The device's id for the state, which names its variants.
    pub id: u64,
    /// The Vulkan pipeline; `None` for mesh pipelines, which bind a variant
    /// per draw.
    pub pipeline: Option<Pipeline>,
    pub plan: RenderPipelinePlan,
    /// Interfaces of the pipeline's functions, by table slot.
    pub interfaces: Vec<Interface>,
    /// Its place in the execution sets when GPU-resident indirect commands
    /// can switch to it.
    pub indirect: Option<IndirectPipeline>,
    /// The samplers its vertex and fragment stages filter bicubically, by
    /// index ([`Specialization::bicubic_samplers`]): draws binding other
    /// bicubic samplers there run a variant
    /// ([`Resolver::bicubic_render_pipeline`]). Mesh pipelines' stages
    /// filter every sampler bicubically once their process has bicubic
    /// samplers.
    pub bicubic: [u32; 2],
    /// It was compiled before its process made bicubic or pixel-coordinate
    /// samplers, which its functions may take as arguments: draws bind a
    /// variant compiled since ([`Resolver::bicubic_render_pipeline`]).
    pub outdated: bool,
}

/// A compute pipeline state object; its Vulkan pipelines are variants.
#[derive(Clone, Debug, PartialEq)]
pub struct ComputePipelineState {
    pub id: u64,
    pub plan: ComputePipelinePlan,
    pub interface: Interface,
    /// Its variant for GPU-resident indirect commands, when they can
    /// switch to it.
    pub indirect: Option<IndirectCompute>,
}

/// A compute pipeline's variant whose dispatches shape its threadgroups
/// ([`Specialization::dynamic_threadgroups`]), which GPU-resident indirect
/// commands run, and its place in the compute execution set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndirectCompute {
    pub pipeline: Pipeline,
    pub place: IndirectPipeline,
}

/// Device services encoding needs.
pub trait Resolver: ChunkSource {
    /// The compute pipeline for one specialization, compiled on first use.
    fn compute_pipeline(
        &mut self,
        state: &ComputePipelineState,
        specialization: &Specialization,
    ) -> Result<Pipeline>;

    /// A sampler like `sampler` with its LOD range clamped
    /// (`setSamplerState:lodMinClamp:lodMaxClamp:`).
    fn clamped_sampler(
        &mut self,
        sampler: ResourceId,
        lod_min: f32,
        lod_max: f32,
    ) -> Result<ResourceId>;

    /// The mesh pipeline variant for the object and mesh threadgroup sizes
    /// and object threadgroup memory of a draw, compiled on first use.
    fn mesh_pipeline(
        &mut self,
        state: &RenderPipelineState,
        object: [u32; 3],
        mesh: [u32; 3],
        object_lengths: &[(u32, u64)],
    ) -> Result<Pipeline>;

    /// The variant of a vertex pipeline whose vertex and fragment stages
    /// filter the samplers at `bicubic` bicubically
    /// ([`RenderPipelineState::bicubic`]), compiled on first use.
    fn bicubic_render_pipeline(
        &mut self,
        state: &RenderPipelineState,
        bicubic: [u32; 2],
    ) -> Result<Pipeline>;

    /// A reset query pool of `query_type` and its query count; pipeline
    /// statistics pools count [`counters::STATISTICS`].
    fn query_pool(&mut self, query_type: QueryType) -> Result<(QueryPool, u32)>;

    /// The render pipeline state with device id `id`, which an indirect
    /// command names.
    fn render_pipeline_state(&mut self, id: u64) -> Result<RenderPipelineState>;

    /// The compute pipeline state with device id `id`.
    fn compute_pipeline_state(&mut self, id: u64) -> Result<ComputePipelineState>;

    /// The current contents of an indirect command buffer.
    fn indirect_commands(
        &mut self,
        id: IndirectCommandBufferId,
    ) -> Result<Arc<IndirectCommandBuffer>>;

    /// The live state classes of indirect-bindable render pipelines.
    fn indirect_render_classes(&mut self) -> Vec<device_generated::IndirectRenderClass>;

    /// The compute execution set and a live pipeline of it, while any
    /// indirect-bindable compute pipeline lives.
    fn indirect_compute_set(&mut self) -> Option<(ExecutionSet, Pipeline)>;

    /// The indirect commands layout of `key`, created on first use, and the
    /// preprocess memory `sequences` sequences of it need with `set` in
    /// upload memory.
    fn generated_layout(
        &mut self,
        key: &device_generated::GeneratedLayoutKey,
        set: Option<ExecutionSet>,
        sequences: u32,
    ) -> Result<(IndirectCommandsLayout, Requirements)>;

    /// The layout and buffer of an indirect command buffer's records,
    /// making it GPU-resident.
    fn indirect_records(
        &mut self,
        id: IndirectCommandBufferId,
    ) -> Result<(RecordLayout, BufferRef)>;

    /// Changes an indirect command buffer's contents, for blit resets and
    /// copies; a failed `update` leaves them unchanged.
    fn update_indirect_commands(
        &mut self,
        id: IndirectCommandBufferId,
        update: &mut dyn FnMut(&mut IndirectCommandBuffer) -> Result<()>,
    ) -> Result<()>;
}

/// Optional dynamic states the device enabled; see [`Capabilities`].
///
/// [`Capabilities`]: super::capabilities::Capabilities
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DynamicFeatures {
    pub depth_clamp: bool,
    pub polygon_mode: bool,
    pub sample_locations: bool,
}

impl DynamicFeatures {
    pub fn new(capabilities: &super::capabilities::Capabilities) -> Self {
        Self {
            depth_clamp: capabilities.depth_clamp,
            polygon_mode: capabilities.wireframe,
            sample_locations: capabilities.programmable_sample_positions,
        }
    }
}

/// One Vulkan submission of a command buffer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Segment {
    pub waits: Vec<(Semaphore, u64)>,
    /// Sparse bindings applied after the waits and everything submitted
    /// before, ahead of `commands`.
    pub binds: Vec<SparseBind>,
    pub commands: Vec<Command>,
    pub signals: Vec<(Semaphore, u64)>,
    pub(crate) guest_load: Option<super::vulkan::backend::GuestRows>,
    pub(crate) guest_store: Option<super::vulkan::backend::GuestRows>,
}

impl Segment {
    pub(crate) fn copies_guest_rows(&self) -> bool {
        self.guest_load.is_some() || self.guest_store.is_some()
    }
}

/// A committed command buffer, ready for the queue.
#[derive(Debug)]
pub struct Recording {
    pub segments: Vec<Segment>,
    /// Upload chunks and the bytes to copy into them before submission;
    /// they return to the pool on completion.
    pub uploads: Vec<(Chunk, Vec<u8>)>,
    /// Query pools to reset and recycle on completion.
    pub query_pools: Vec<(QueryType, QueryPool)>,
    /// Sparse mapping operations and the segment whose binds they become.
    pub mappings: Vec<(usize, Mapping)>,
    /// Managed buffers to invalidate on completion, for the host to read
    /// what the GPU wrote.
    pub invalidations: Vec<MappedRange>,
    /// The pipeline states its encoders set, in first-set order.
    pub pipelines: Vec<BoundPipeline>,
    /// What its commands do.
    pub workload: Workload,
}

/// What a command buffer recorded, which describes the work in flight when
/// the device is lost: its draws' sizes, and the commands that can make the
/// GPU wait or loop on what other work wrote.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Workload {
    pub direct: u32,
    pub indirect: u32,
    /// The most vertices, indices or mesh threadgroups times instances of one
    /// direct draw.
    pub largest: u64,
    pub passes: u32,
    pub dispatches: u32,
    /// Copies, blits and fills.
    pub transfers: u32,
    /// Query result copies that wait for their queries to be available.
    pub waited_queries: u32,
    pub conditional: u32,
    /// The device's own compute passes, which read counts the GPU wrote.
    pub internal: u32,
    /// The direct dispatch with the most threads, which names what a lost
    /// device was running.
    pub widest: Option<DispatchShape>,
}

/// A direct dispatch: its threadgroups, threads per threadgroup and the
/// first threadgroup memory lengths it bound, by slot.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DispatchShape {
    pub threadgroups: [u32; 3],
    pub threads: [u32; 3],
    pub lengths: [(u32, u32); DispatchShape::LENGTHS],
    pub length_count: u8,
}

impl DispatchShape {
    pub const LENGTHS: usize = 8;

    fn total(&self) -> u64 {
        self.threadgroups
            .iter()
            .chain(&self.threads)
            .map(|&size| u64::from(size))
            .product()
    }
}

impl core::fmt::Display for DispatchShape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let [x, y, z] = self.threadgroups;
        let [w, h, d] = self.threads;
        write!(f, "{x}x{y}x{z} threadgroups of {w}x{h}x{d}")?;
        for (slot, length) in &self.lengths[..usize::from(self.length_count)] {
            write!(f, ", threadgroup {slot} {length} bytes")?;
        }
        Ok(())
    }
}

impl Workload {
    /// Keeps `shape` when it runs the most threads so far.
    fn dispatched(&mut self, shape: DispatchShape) {
        if self
            .widest
            .is_none_or(|widest| shape.total() > widest.total())
        {
            self.widest = Some(shape);
        }
    }

    /// Counts the commands of `segments` besides the draws already counted.
    fn count(mut self, segments: &[Segment]) -> Self {
        for command in segments.iter().flat_map(|segment| &segment.commands) {
            match command {
                Command::BeginRendering(_) => self.passes += 1,
                // A pass resolving through its raw texels averages in a dispatch.
                Command::Dispatch(_)
                | Command::DispatchIndirect { .. }
                | Command::AverageSamples { .. } => self.dispatches += 1,
                Command::CopyBuffer { .. }
                | Command::CopyBufferRegions { .. }
                | Command::CopyBufferToImage { .. }
                | Command::CopyImageToBuffer { .. }
                | Command::CopyImage { .. }
                | Command::BlitImage { .. }
                | Command::ClearColorImage { .. }
                | Command::FillBuffer { .. } => self.transfers += 1,
                Command::CopyQueryResults { wait: true, .. } => self.waited_queries += 1,
                Command::BeginConditionalRendering { .. } => self.conditional += 1,
                Command::AccumulateResults { .. }
                | Command::SelectCommands { .. }
                | Command::ResolveDraws { .. }
                | Command::ResolveDispatches { .. }
                | Command::ExecuteGeneratedCommands(_) => self.internal += 1,
                Command::StoreTexels { .. }
                | Command::LoadTexels { .. }
                | Command::DecodeBlocks { .. }
                | Command::ReorderBlocks { .. } => self.transfers += 1,
                _ => {}
            }
        }
        self
    }
}

/// A pipeline state an encoder set, by its device id, which names the work
/// in flight when the device is lost. The pipelines an indirect command
/// buffer sets itself are not among them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundPipeline {
    Render(u64),
    Compute(u64),
}

/// One stage's argument table and whether the GPU copy is current.
#[derive(Clone, Debug, Default)]
struct Table {
    slots: ArgumentTable,
    buffers: [Option<BufferRef>; super::capabilities::MAX_BUFFER_BINDINGS as usize],
    address: Option<u64>,
}

/// The argument table a binding goes to: a stage's, or a render encoder's
/// tile functions', which tile dispatches read as a kernel's.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Slot {
    Stage(Stage),
    Tile,
}

impl Slot {
    const fn index(self) -> usize {
        match self {
            Self::Stage(stage) => stage.table().raw() as usize,
            Self::Tile => 3,
        }
    }
}

impl From<Stage> for Slot {
    fn from(stage: Stage) -> Self {
        Self::Stage(stage)
    }
}

/// Bindings shared by the stages of an encoder.
#[derive(Clone, Debug, Default)]
struct Bindings {
    tables: [Table; 4],
    constants: PushConstants,
    pushed: Option<PushConstants>,
}

/// Why a draw is refused when a buffer one of its functions accesses is
/// unbound, by stage and buffer index.
macro_rules! unbound {
    ($stage:literal: $($index:literal)*) => {
        [$(concat!("the ", $stage, " function accesses buffer ", stringify!($index),
                   ", which is unbound")),*]
    };
}
const UNBOUND: [[&str; super::capabilities::MAX_BUFFER_BINDINGS as usize]; 5] = [
    unbound!("vertex": 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30),
    unbound!("fragment": 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30),
    unbound!("object": 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30),
    unbound!("mesh": 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30),
    unbound!("tile": 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30),
];

impl Bindings {
    fn table(&mut self, slot: impl Into<Slot>) -> &mut Table {
        &mut self.tables[slot.into().index()]
    }

    /// Fails unless each buffer the functions access is bound. Metal
    /// requires a draw to bind every buffer its functions reference; one
    /// left unbound would be read at address zero, which faults the device
    /// for every process.
    fn require_buffers(&self, interfaces: &[Interface]) -> Result<()> {
        interfaces
            .iter()
            .try_for_each(|interface| self.require_buffers_of(interface.stage.into(), interface))
    }

    /// Fails unless each buffer `interface` accesses is bound in `slot`'s
    /// table.
    fn require_buffers_of(&self, slot: Slot, interface: &Interface) -> Result<()> {
        {
            let table = &self.tables[slot.index()];
            let unbound = table
                .slots
                .buffers
                .iter()
                .enumerate()
                .find(|&(index, buffer)| {
                    interface.buffer_reads & (1 << index) != 0 && buffer.address == 0
                });
            if let Some((index, _)) = unbound {
                let stage = match (slot, interface.stage) {
                    (Slot::Tile, _) => 4,
                    (_, Stage::Fragment) => 1,
                    (_, Stage::Object) => 2,
                    (_, Stage::Mesh) => 3,
                    _ => 0,
                };
                return Err(Error::Invalid(UNBOUND[stage][index]));
            }
        }
        Ok(())
    }

    fn set_buffer(
        &mut self,
        slot: impl Into<Slot>,
        buffer: Option<BufferRef>,
        offset: u64,
        index: u32,
    ) -> Result<()> {
        let target = slot.into();
        let slot = buffer_slot(index)?;
        let binding = match buffer {
            Some(buffer) => {
                ensure(offset <= buffer.length, "buffer offset beyond the buffer")?;
                BufferBinding {
                    address: buffer.address + offset,
                    length: buffer.length - offset,
                }
            }
            None => BufferBinding::default(),
        };
        let table = self.table(target);
        table.buffers[slot] = buffer;
        if table.slots.buffers[slot] != binding {
            table.slots.buffers[slot] = binding;
            table.address = None;
        }
        Ok(())
    }

    /// `set*BufferOffset:atIndex:` keeps the buffer and moves its offset.
    fn set_buffer_offset(&mut self, slot: impl Into<Slot>, offset: u64, index: u32) -> Result<()> {
        let slot = slot.into();
        let bound = self.table(slot).buffers[buffer_slot(index)?];
        let bound = bound.ok_or(Error::Invalid("no buffer is bound at that index"))?;
        self.set_buffer(slot, Some(bound), offset, index)
    }

    /// Unbinds the buffers of `stages`, keeping their textures and samplers.
    fn clear_buffers(&mut self, stages: &[Stage]) {
        for &stage in stages {
            let table = self.table(stage);
            if table.buffers.iter().any(Option::is_some) {
                table.slots.buffers = ArgumentTable::default().buffers;
                table.buffers = [None; super::capabilities::MAX_BUFFER_BINDINGS as usize];
                table.address = None;
            }
        }
    }

    fn set_texture(
        &mut self,
        slot: impl Into<Slot>,
        texture: ResourceId,
        index: u32,
    ) -> Result<()> {
        ensure(
            (index as usize) < self.tables[0].slots.textures.len(),
            "texture index exceeds the argument table",
        )?;
        let table = self.table(slot);
        if table.slots.textures[index as usize] != texture {
            table.slots.textures[index as usize] = texture;
            table.address = None;
        }
        Ok(())
    }

    /// The textures of the pass's color attachments, for `[[color(n)]]`,
    /// its samples and its imageblock data.
    fn set_pass(
        &mut self,
        textures: [ResourceId; MAX_COLOR_ATTACHMENTS],
        samples: PassSamples,
        imageblock: PassImageblock,
    ) {
        for slot in [Slot::Stage(Stage::Fragment), Slot::Tile] {
            let table = self.table(slot);
            table.slots.attachments = textures;
            table.slots.samples = samples;
            table.slots.imageblock = imageblock;
            table.address = None;
        }
    }

    /// The pass's dirty tiles, which its fragment functions mark.
    fn set_dirty(&mut self, dirty: crate::metal::binding::PassDirty) {
        let table = self.table(Stage::Fragment);
        if table.slots.dirty != dirty {
            table.slots.dirty = dirty;
            table.address = None;
        }
    }

    /// What a fragment function testing fragments itself reads.
    fn set_fragment_tests(&mut self, tests: crate::metal::binding::FragmentTests) {
        let table = self.table(Stage::Fragment);
        if table.slots.fragment_tests != tests {
            table.slots.fragment_tests = tests;
            table.address = None;
        }
    }

    fn set_blend_color(&mut self, color: [f32; 4]) {
        let color = color.map(f32::to_bits);
        let table = self.table(Stage::Fragment);
        if table.slots.blend_color != color {
            table.slots.blend_color = color;
            table.address = None;
        }
    }

    fn set_sampler(
        &mut self,
        slot: impl Into<Slot>,
        sampler: ResourceId,
        index: u32,
    ) -> Result<()> {
        ensure(
            (index as usize) < self.tables[0].slots.samplers.len(),
            "sampler index exceeds the argument table",
        )?;
        let table = self.table(slot);
        if table.slots.samplers[index as usize] != sampler {
            table.slots.samplers[index as usize] = sampler;
            table.address = None;
        }
        Ok(())
    }

    /// The samplers `interface` samples through that `slot`'s table binds
    /// with Metal's bicubic filter or pixel coordinates, by index.
    fn bicubic_samplers(&self, slot: impl Into<Slot>, interface: &Interface) -> u32 {
        let samplers = &self.tables[slot.into().index()].slots.samplers;
        interface
            .samplers
            .iter()
            .filter(|&&index| {
                samplers
                    .get(index as usize)
                    .is_some_and(|sampler| sampler.high() & ResourceId::SAMPLER_SHADED != 0)
            })
            .fold(0, |mask, &index| mask | 1 << index)
    }

    /// Uploads changed tables of `stages` and pushes constants when they
    /// changed.
    fn flush(
        &mut self,
        recorder: &mut Recorder,
        resolver: &mut dyn Resolver,
        stages: &[Stage],
    ) -> Result<()> {
        for &stage in stages {
            let address = self.upload_table(recorder, resolver, stage.into())?;
            self.constants.tables[stage.table().raw() as usize] = address;
        }
        if self.pushed != Some(self.constants) {
            recorder.push(Command::PushConstants(self.constants));
            self.pushed = Some(self.constants);
        }
        Ok(())
    }

    /// The address of `slot`'s table, uploaded when it changed.
    fn upload_table(
        &mut self,
        recorder: &mut Recorder,
        resolver: &mut dyn Resolver,
        slot: Slot,
    ) -> Result<u64> {
        let table = &mut self.tables[slot.index()];
        if let Some(address) = table.address {
            return Ok(address);
        }
        let (upload, bytes) = recorder.uploads.zeroed(
            resolver,
            ArgumentTable::BYTES as usize,
            ArgumentTable::ALIGNMENT,
        )?;
        table.slots.encode_into(bytes);
        table.address = Some(upload.address);
        Ok(upload.address)
    }

    /// Bindings carry over between encoders only through the table memory
    /// already uploaded; a new encoder starts empty, as Metal's do.
    /// Forgets an encoder's bindings; the log buffer is the command buffer's.
    fn reset(&mut self) {
        let log = self.constants.log;
        *self = Self::default();
        self.constants.log = log;
    }
}

fn buffer_slot(index: u32) -> Result<usize> {
    ensure(
        index < super::capabilities::MAX_BUFFER_BINDINGS,
        "buffer index exceeds the argument table",
    )?;
    Ok(index as usize)
}

/// `segments` with each one that waits for nothing joined to the one before
/// it, whose signals move to its end: a submission of its own would leave
/// the GPU idle while the next is submitted. A signal then comes later, past
/// commands only, never past a wait that could depend on it. The joined
/// segment's opening full barrier still orders its work after the signaled
/// work. A semaphore signaled twice keeps its higher value.
fn merge_unwaited(segments: Vec<Segment>) -> Vec<Segment> {
    let mut merged: Vec<Segment> = Vec::with_capacity(segments.len());
    for segment in segments {
        let Some(last) = merged.last_mut().filter(|last| {
            segment.waits.is_empty()
                && segment.binds.is_empty()
                && !segment.copies_guest_rows()
                && !last.copies_guest_rows()
        }) else {
            merged.push(segment);
            continue;
        };
        last.commands.extend(segment.commands);
        for (semaphore, value) in segment.signals {
            match last
                .signals
                .iter_mut()
                .find(|(signal, _)| *signal == semaphore)
            {
                Some((_, kept)) => *kept = (*kept).max(value),
                None => last.signals.push((semaphore, value)),
            }
        }
    }
    merged
}

/// The command stream of a command buffer, independent of encoder state.
#[derive(Debug)]
struct Recorder {
    segments: Vec<Segment>,
    hazards: Hazards,
    uploads: UploadArena,
    query_pools: Vec<(QueryType, QueryPool)>,
    statistics: Option<counters::Statistics>,
    mappings: Vec<(usize, Mapping)>,
    invalidations: Vec<MappedRange>,
}

impl Recorder {
    fn new() -> Self {
        let mut recorder = Self {
            segments: Vec::new(),
            hazards: Hazards::default(),
            uploads: UploadArena::default(),
            query_pools: Vec::new(),
            statistics: None,
            mappings: Vec::new(),
            invalidations: Vec::new(),
        };
        recorder.start_segment(Vec::new());
        recorder
    }

    fn start_segment(&mut self, waits: Vec<(Semaphore, u64)>) {
        self.segments.push(Segment {
            waits,
            commands: alloc::vec![Command::Barrier(FULL)],
            ..Segment::default()
        });
        self.hazards = Hazards::default();
    }

    fn segment(&mut self) -> &mut Segment {
        self.segments
            .last_mut()
            .expect("a command buffer always has a segment")
    }

    fn commands(&mut self) -> &mut Vec<Command> {
        &mut self
            .segments
            .last_mut()
            .expect("a command buffer always has a segment")
            .commands
    }

    fn push(&mut self, command: Command) {
        self.commands().push(command);
    }

    /// Orders `next` after the pending work it depends on.
    fn access(&mut self, next: Use) {
        if let Some(barrier) = self.hazards.access(next) {
            self.push(Command::Barrier(barrier));
        }
    }

    fn full_barrier(&mut self) {
        if let Some(barrier) = self.hazards.full() {
            self.push(Command::Barrier(barrier));
        }
    }

    fn upload(
        &mut self,
        resolver: &mut dyn Resolver,
        bytes: &[u8],
        alignment: u64,
    ) -> Result<Upload> {
        self.uploads.write(resolver, bytes, alignment)
    }
}

enum Encoder {
    None,
    Render(Box<render::RenderState>),
    Compute(Box<compute::ComputeState>),
    Blit,
    ResourceState,
}

/// A command buffer being encoded.
pub struct CommandBuffer {
    recorder: Recorder,
    bindings: Bindings,
    encoder: Encoder,
    features: DynamicFeatures,
    pipelines: Vec<BoundPipeline>,
    workload: Workload,
    /// What its passes change of the extended-range textures' planes,
    /// applied once it is committed.
    pub(crate) planes: super::device::resources::PlaneChanges,
}

impl CommandBuffer {
    pub fn new(features: DynamicFeatures) -> Self {
        Self {
            recorder: Recorder::new(),
            bindings: Bindings::default(),
            encoder: Encoder::None,
            features,
            pipelines: Vec::new(),
            workload: Workload::default(),
            planes: Default::default(),
        }
    }

    /// Records that an encoder set `pipeline`.
    fn bound(&mut self, pipeline: BoundPipeline) {
        if !self.pipelines.contains(&pipeline) {
            self.pipelines.push(pipeline);
        }
    }

    /// Gives the shaders of this command buffer the buffer `os_log` appends to.
    pub fn set_log_buffer(&mut self, buffer: BufferRef) {
        self.bindings.constants.log = buffer.address;
    }

    fn idle(&self) -> Result<()> {
        ensure(
            matches!(self.encoder, Encoder::None),
            "another encoder is still encoding",
        )
    }

    /// Copies `set*Bytes` data into upload memory as a buffer of its own.
    fn upload_bytes(&mut self, resolver: &mut dyn Resolver, bytes: &[u8]) -> Result<BufferRef> {
        ensure(bytes.len() <= 4096, "set*Bytes holds at most 4 KiB")?;
        let upload = self.recorder.upload(resolver, bytes, 16)?;
        Ok(BufferRef {
            buffer: upload.buffer,
            offset: upload.offset,
            address: upload.address,
            length: bytes.len() as u64,
        })
    }

    /// `endEncoding` of the active encoder.
    pub fn end_encoding(&mut self, resolver: &mut dyn Resolver) -> Result<()> {
        match core::mem::replace(&mut self.encoder, Encoder::None) {
            Encoder::None => Err(Error::Invalid("no encoder is encoding")),
            Encoder::Render(state) => {
                // The copies return to the attachments in a pass instance
                // before the last, whose store operations follow them.
                self.encoder = Encoder::Render(state);
                self.finish_fragment_tests(resolver)?;
                let Encoder::Render(state) = core::mem::replace(&mut self.encoder, Encoder::None)
                else {
                    unreachable!("the render encoder was just restored")
                };
                self.end_render(resolver, *state)
            }
            Encoder::Compute(_) | Encoder::Blit | Encoder::ResourceState => Ok(()),
        }?;
        self.bindings.reset();
        Ok(())
    }

    /// `encodeSignalEvent:value:`: work encoded so far signals the event.
    pub fn encode_signal_event(
        &mut self,
        resolver: &mut dyn Resolver,
        event: Semaphore,
        value: u64,
    ) -> Result<()> {
        self.idle()?;
        // Statistics queries end with their segment.
        self.recorder.pause_statistics();
        self.recorder
            .segments
            .last_mut()
            .expect("a segment")
            .signals
            .push((event, value));
        self.recorder.start_segment(Vec::new());
        self.recorder.resume_statistics(resolver, false)
    }

    /// `encodeWaitForEvent:value:`: later work waits for the event.
    pub fn encode_wait_for_event(
        &mut self,
        resolver: &mut dyn Resolver,
        event: Semaphore,
        value: u64,
    ) -> Result<()> {
        self.idle()?;
        // A value this command buffer signals earlier needs no wait: the
        // signal ends its segment, and every later segment opens with a full
        // barrier on the one queue, so later work already follows the
        // signaled work. A wait would split the submission instead.
        let signaled = self
            .recorder
            .segments
            .iter()
            .flat_map(|segment| &segment.signals)
            .any(|&(signal, reached)| signal == event && reached >= value);
        if signaled {
            return Ok(());
        }
        let current = self.recorder.segments.last().expect("a segment");
        // A segment holding only its opening barrier can take the wait.
        if current.commands.len() == 1 && current.signals.is_empty() {
            self.recorder
                .segments
                .last_mut()
                .expect("a segment")
                .waits
                .push((event, value));
            return Ok(());
        }
        self.recorder.pause_statistics();
        self.recorder.start_segment(alloc::vec![(event, value)]);
        self.recorder.resume_statistics(resolver, false)
    }

    /// `commit`: ends the commands; the last segment signals `completion`.
    /// Statistic samples not yet resolved are written last.
    pub fn finish(
        &mut self,
        resolver: &mut dyn Resolver,
        completion: (Semaphore, u64),
    ) -> Result<()> {
        self.idle()?;
        self.recorder.pause_statistics();
        self.recorder.resolve_statistics(resolver)?;
        let last = self.recorder.segments.last_mut().expect("a segment");
        last.signals.push(completion);
        // An empty trailing segment left by a final signal still carries the
        // completion signal; it orders after everything before it.
        if self.recorder.mappings.is_empty() {
            self.recorder.segments = merge_unwaited(core::mem::take(&mut self.recorder.segments));
        }
        Ok(())
    }

    /// The recording: once finished, for the queue to submit; otherwise
    /// for the device to take back what the commands acquired.
    pub fn into_recording(self) -> Recording {
        let workload = self.workload.count(&self.recorder.segments);
        Recording {
            segments: self.recorder.segments,
            uploads: self.recorder.uploads.finish(),
            query_pools: self.recorder.query_pools,
            mappings: self.recorder.mappings,
            invalidations: self.recorder.invalidations,
            pipelines: self.pipelines,
            workload,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
