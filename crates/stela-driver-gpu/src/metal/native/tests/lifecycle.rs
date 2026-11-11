//! Driver acknowledgements required before releasing possibly queued objects.
use super::super::*;
use crate::metal::nvidia::{Chip, fixture::nvk};
use alloc::sync::Arc;
use core::{
    cell::Cell,
    ffi::c_void,
    sync::atomic::{AtomicUsize, Ordering},
};
use vk::Handle;

struct Calls {
    result: vk::Result,
    waits: AtomicUsize,
    frees: AtomicUsize,
}

unsafe extern "system" fn wait_queue(queue: vk::Queue) -> vk::Result {
    // SAFETY: tests encode their kept Calls allocation in the queue handle.
    let calls = unsafe { &*(queue.as_raw() as *const Calls) };
    calls.waits.fetch_add(1, Ordering::Relaxed);
    calls.result
}

unsafe extern "system" fn wait_device(device: vk::Device) -> vk::Result {
    // SAFETY: tests encode their kept Calls allocation in the device handle.
    let calls = unsafe { &*(device.as_raw() as *const Calls) };
    calls.waits.fetch_add(1, Ordering::Relaxed);
    calls.result
}

unsafe extern "system" fn free_buffers(
    device: vk::Device,
    _: vk::CommandPool,
    count: u32,
    _: *const vk::CommandBuffer,
) {
    // SAFETY: as in wait_device; the callback never dereferences fake buffers.
    let calls = unsafe { &*(device.as_raw() as *const Calls) };
    assert_eq!(calls.waits.load(Ordering::Relaxed), 1);
    calls.frees.fetch_add(count as usize, Ordering::Relaxed);
}

fn device(calls: &Calls) -> ash::Device {
    // SAFETY: only the typed callbacks above are invoked; Calls outlives it.
    unsafe {
        ash::Device::load_with(
            |name| match name.to_bytes() {
                b"vkQueueWaitIdle" => wait_queue as *const c_void,
                b"vkDeviceWaitIdle" => wait_device as *const c_void,
                b"vkFreeCommandBuffers" => free_buffers as *const c_void,
                _ => core::ptr::null(),
            },
            vk::Device::from_raw(calls as *const Calls as u64),
        )
    }
}

#[test]
fn failed_idle_keeps_possibly_submitted_command_buffers() {
    for (result, freed) in [
        (vk::Result::SUCCESS, 2),
        (vk::Result::ERROR_DEVICE_LOST, 0),
        (vk::Result::ERROR_OUT_OF_HOST_MEMORY, 0),
    ] {
        let calls = Calls {
            result,
            waits: AtomicUsize::new(0),
            frees: AtomicUsize::new(0),
        };
        let device = device(&calls);
        let queue = vk::Queue::from_raw(&calls as *const Calls as u64);
        let buffers = [
            vk::CommandBuffer::from_raw(1),
            vk::CommandBuffer::from_raw(2),
        ];
        let outcome = free_after_idle(&device, queue, vk::CommandPool::null(), &buffers);
        assert_eq!(outcome.is_ok(), result == vk::Result::SUCCESS);
        assert_eq!(calls.waits.load(Ordering::Relaxed), 1);
        assert_eq!(calls.frees.load(Ordering::Relaxed), freed);
    }
}

struct Exporter(Arc<AtomicUsize>);

impl Drop for Exporter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

impl MemoryExporter for Exporter {
    fn export(&self, _: &[MemoryRange]) -> Result<ExportedMemory> {
        unreachable!()
    }
    fn close(&self, _: ExportedMemory) {
        unreachable!()
    }
    fn unmap(&self, _: ExportedMemory) {
        unreachable!()
    }
}

impl HostMemoryExporter for Exporter {
    fn lease(&self, _: &[MemoryRange]) -> Result<HostAllocation> {
        unreachable!()
    }
}

impl GuestCopyProvider for Exporter {
    unsafe fn lease(&self, _: &[MemoryRange]) -> Result<Box<dyn GuestCopyLease>> {
        unreachable!()
    }
}

impl GuestCopyLease for Exporter {
    fn read_into(&self, _: u64, _: &mut [u8]) -> Result<()> {
        unreachable!()
    }

    fn publish_from(&self, _: u64, _: &[u8]) -> Result<()> {
        unreachable!()
    }
}

unsafe extern "system" fn instance_proc(
    _: vk::Instance,
    _: *const core::ffi::c_char,
) -> vk::PFN_vkVoidFunction {
    None
}

unsafe extern "system" fn device_proc(
    _: vk::Device,
    _: *const core::ffi::c_char,
) -> vk::PFN_vkVoidFunction {
    None
}

fn backend(device: ash::Device, drops: &Arc<AtomicUsize>) -> NativeBackend {
    // SAFETY: these tables have no GPU; only wait_device runs during Drop.
    // Missing destruction callbacks fail the test if Drop tries to free state.
    let (entry, instance) = unsafe {
        (
            ash::Entry::from_static_fn(ash::StaticFn {
                get_instance_proc_addr: instance_proc,
            }),
            ash::Instance::load_with(
                |name| {
                    if name.to_bytes() == b"vkGetDeviceProcAddr" {
                        device_proc as *const c_void
                    } else {
                        core::ptr::null()
                    }
                },
                vk::Instance::null(),
            ),
        )
    };
    let conditional = ext::conditional_rendering::Device::new(&instance, &device);
    let exporter: Arc<dyn MemoryExporter> = Arc::new(Exporter(drops.clone()));
    let outside_device = device.clone();
    NativeBackend {
        _entry: ManuallyDrop::new(entry),
        instance,
        messenger: None,
        device,
        fns: ExtensionFns {
            host_image_copy: None,
            dynamic_state3: None,
            sample_locations: None,
            depth_bias: None,
            mesh: None,
            calibrated: None,
            external_memory_fd: None,
            external_semaphore_fd: None,
            conditional,
            generated: None,
            executables: None,
        },
        queue: vk::Queue::null(),
        snapshot: nvk(Chip::find(0x1b3).unwrap(), 16 << 30),
        heap: Heap {
            layout: vk::DescriptorSetLayout::null(),
            pool: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
        },
        null_sampler: vk::Sampler::null(),
        layout: vk::PipelineLayout::null(),
        command_pool: vk::CommandPool::null(),
        in_flight: VecDeque::new(),
        unsignaled: Vec::new(),
        mappings: BTreeMap::new(),
        memories: crate::metal::native::memory::Memories::new(1024),
        exporter: ManuallyDrop::new(Some(exporter.clone())),
        host_imports: ManuallyDrop::new(None),
        copy_imports: ManuallyDrop::new(None),
        outside: Arc::new(crate::metal::native::NativeOutside::new(
            outside_device,
            Some(exporter),
            false,
            || {},
        )),
        imports: BTreeMap::new(),
        import_cache: crate::metal::native::ImportCache::default(),
        retired: VecDeque::new(),
        refusals: None,
        outstanding: None,
        new_images: Vec::new(),
        zeroing: crate::metal::native::zero::Zeroing::new(1 << 32, |_| {}),
        bind_timeline: vk::Semaphore::null(),
        bind_value: 0,
        accumulate: vk::Pipeline::null(),
        select: vk::Pipeline::null(),
        resolve_draws: vk::Pipeline::null(),
        resolve_dispatches: vk::Pipeline::null(),
        store_texels: vk::Pipeline::null(),
        load_texels: vk::Pipeline::null(),
        average_samples: vk::Pipeline::null(),
        decode_etc: vk::Pipeline::null(),
        decode_pvrtc: vk::Pipeline::null(),
        reorder_blocks: vk::Pipeline::null(),
        decode_astc: vk::Pipeline::null(),
        store: crate::metal::native::pipeline_store::Persisted::none(|_| {}),
        deferral: Default::default(),
        ticks: crate::metal::timing::no_clock,
        split: [0; 3],
    }
}

#[test]
fn failed_device_idle_keeps_the_export_owner() {
    let calls = Calls {
        result: vk::Result::ERROR_DEVICE_LOST,
        waits: AtomicUsize::new(0),
        frees: AtomicUsize::new(0),
    };
    let drops = Arc::new(AtomicUsize::new(0));
    let host_drops = Arc::new(AtomicUsize::new(0));
    let copied_drops = Arc::new(AtomicUsize::new(0));
    let lease_drops = Arc::new(AtomicUsize::new(0));
    let mut backend = backend(device(&calls), &drops);
    backend.host_imports = ManuallyDrop::new(Some(host_memory::Imports::new(
        Box::new(Exporter(host_drops.clone())),
        &backend.instance,
        &backend.device,
        vk::PhysicalDevice::null(),
        NonZeroU64::MIN,
    )));
    let mut copies = copy_memory::Imports::new(Box::new(Exporter(copied_drops.clone())));
    copies.owned.insert(
        1,
        copy_memory::Import {
            lease: Box::new(Exporter(lease_drops.clone())),
            token: NonZeroU64::new(43).unwrap(),
            bytes: 4096,
        },
    );
    copies.checkpoint = Some((vk::Semaphore::from_raw(17), 1));
    backend.copy_imports = ManuallyDrop::new(Some(copies));
    drop(backend);
    assert_eq!(calls.waits.load(Ordering::Relaxed), 1);
    assert_eq!(calls.frees.load(Ordering::Relaxed), 0);
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    assert_eq!(host_drops.load(Ordering::Relaxed), 0);
    assert_eq!(copied_drops.load(Ordering::Relaxed), 0);
    assert_eq!(lease_drops.load(Ordering::Relaxed), 0);
}

#[test]
fn idle_acknowledgement_requires_native_success_and_keeps_roots_owned() {
    for (result, expected) in [
        (vk::Result::SUCCESS, Ok(())),
        (vk::Result::ERROR_DEVICE_LOST, Err(Error::DeviceLost)),
        (
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            Err(Error::OutOfMemory),
        ),
    ] {
        let calls = Calls {
            result,
            waits: AtomicUsize::new(0),
            frees: AtomicUsize::new(0),
        };
        let drops = Arc::new(AtomicUsize::new(0));
        let mut backend = ManuallyDrop::new(backend(device(&calls), &drops));
        assert_eq!(backend.acknowledge_idle(), expected);
        assert_eq!(calls.waits.load(Ordering::Relaxed), 1);
        assert_eq!(calls.frees.load(Ordering::Relaxed), 0);
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        // SAFETY: this fixture has no native GPU resources. Its two owned
        // Rust roots are dropped once without invoking fake Vulkan teardown.
        unsafe {
            ManuallyDrop::drop(&mut backend.exporter);
            ManuallyDrop::drop(&mut backend._entry);
        }
    }
}

struct BufferCalls {
    status: Cell<vk::Result>,
    value: Cell<u64>,
    allocated: Cell<u64>,
    queries: Cell<usize>,
    freed: Cell<Option<(u32, u64)>>,
}

unsafe extern "system" fn allocate(
    device: vk::Device,
    info: *const vk::CommandBufferAllocateInfo<'_>,
    output: *mut vk::CommandBuffer,
) -> vk::Result {
    // SAFETY: the fixture owns the kept BufferCalls and Vulkan arguments.
    let (calls, info) = unsafe { (&*(device.as_raw() as *const BufferCalls), &*info) };
    assert!(info.level == vk::CommandBufferLevel::PRIMARY);
    assert_eq!(info.command_pool.as_raw(), 2);
    if calls.status.get() == vk::Result::SUCCESS {
        for index in 0..info.command_buffer_count {
            let next = calls.allocated.get() + 1;
            // SAFETY: the caller provides one writable handle per requested buffer.
            unsafe {
                output
                    .add(index as usize)
                    .write(vk::CommandBuffer::from_raw(next))
            };
            calls.allocated.set(next);
        }
    }
    calls.status.get()
}

unsafe extern "system" fn counter(
    device: vk::Device,
    _: vk::Semaphore,
    value: *mut u64,
) -> vk::Result {
    // SAFETY: the fixture owns the kept BufferCalls.
    let calls = unsafe { &*(device.as_raw() as *const BufferCalls) };
    calls.queries.set(calls.queries.get() + 1);
    // SAFETY: the caller provides a writable counter.
    unsafe { value.write(calls.value.get()) };
    calls.status.get()
}

unsafe extern "system" fn free_completed(
    device: vk::Device,
    pool: vk::CommandPool,
    count: u32,
    buffers: *const vk::CommandBuffer,
) {
    // SAFETY: the fixture owns the kept BufferCalls and submitted handle array.
    let (calls, buffers) = unsafe {
        (
            &*(device.as_raw() as *const BufferCalls),
            core::slice::from_raw_parts(buffers, count as usize),
        )
    };
    assert_eq!(pool.as_raw(), 2);
    let ordered = buffers
        .iter()
        .fold(0, |sum, buffer| sum * 10 + buffer.as_raw());
    assert!(calls.freed.replace(Some((count, ordered))).is_none());
}

#[test]
fn inline_and_owned_buffers_retire_once_in_queue_order() {
    let calls = BufferCalls {
        status: Cell::new(vk::Result::SUCCESS),
        value: Cell::new(0),
        allocated: Cell::new(0),
        queries: Cell::new(0),
        freed: Cell::new(None),
    };
    // SAFETY: only these typed test callbacks are invoked and Calls stays live.
    let device = unsafe {
        ash::Device::load_with(
            |name| match name.to_bytes() {
                b"vkAllocateCommandBuffers" => allocate as *const c_void,
                b"vkGetSemaphoreCounterValue" => counter as *const c_void,
                b"vkFreeCommandBuffers" => free_completed as *const c_void,
                _ => core::ptr::null(),
            },
            vk::Device::from_raw(&calls as *const BufferCalls as u64),
        )
    };
    let drops = Arc::new(AtomicUsize::new(0));
    let mut backend = ManuallyDrop::new(backend(device, &drops));
    backend.command_pool = vk::CommandPool::from_raw(2);
    for (timeline, value, count) in [(7, 1, 1), (7, 2, 1), (7, 3, 2), (8, 4, 1)] {
        let buffers =
            CommandBuffers::allocate(&backend.device, backend.command_pool, count).unwrap();
        assert_eq!(buffers.as_slice().len(), count as usize);
        backend.in_flight.push_back(InFlight {
            buffers,
            timeline: vk::Semaphore::from_raw(timeline),
            value,
        });
    }
    calls.status.set(vk::Result::ERROR_DEVICE_LOST);
    for count in [1, 2] {
        assert!(matches!(
            CommandBuffers::allocate(&backend.device, backend.command_pool, count),
            Err(Error::DeviceLost),
        ));
    }
    assert_eq!(backend.reclaim(), Err(Error::DeviceLost));
    assert_eq!(
        backend.semaphore_value(Semaphore(7)),
        Err(Error::DeviceLost)
    );
    assert_eq!(backend.in_flight.len(), 4);
    assert_eq!(calls.allocated.get(), 5);
    assert_eq!(calls.freed.get(), None);

    calls.status.set(vk::Result::SUCCESS);
    backend.reclaim().unwrap();
    assert_eq!(calls.freed.get(), None);
    calls.value.set(4);
    assert_eq!(backend.semaphore_value(Semaphore(8)), Ok(4));
    assert_eq!(calls.freed.get(), None);
    assert_eq!(backend.in_flight.len(), 4);
    calls.value.set(1);
    backend.reclaim().unwrap();
    assert_eq!(calls.freed.replace(None), Some((1, 1)));
    assert_eq!(backend.in_flight.len(), 3);

    calls.value.set(4);
    assert_eq!(backend.semaphore_value(Semaphore(7)), Ok(4));
    assert_eq!(calls.freed.replace(None), Some((3, 234)));
    assert_eq!(backend.in_flight.len(), 1);
    assert_eq!(backend.semaphore_value(Semaphore(8)), Ok(4));
    assert_eq!(calls.freed.replace(None), Some((1, 5)));
    assert!(backend.in_flight.is_empty());
    let queries = calls.queries.get();
    backend.reclaim().unwrap();
    assert_eq!(calls.queries.get(), queries);
    // SAFETY: the fixture has no Vulkan resources and the list is empty.
    unsafe { ManuallyDrop::drop(&mut backend.exporter) };
}

#[test]
fn a_freed_import_is_taken_back_for_the_same_guest_ram_only() {
    use crate::metal::native::{ExportedMemory, ImportCache};
    use stela_contracts::gate::MemoryRange;
    let mut page = [0u8; 64];
    let pointer = core::ptr::NonNull::new(page.as_mut_ptr()).unwrap();
    let exported = |bytes| ExportedMemory {
        fd: -1,
        pointer,
        bytes,
    };
    let range = |address| MemoryRange {
        address,
        bytes: 4096,
    };
    let mut cache = ImportCache::default();
    assert!(
        cache
            .keep(
                alloc::vec![range(0x1000)],
                vk::DeviceMemory::from_raw(1),
                exported(4096)
            )
            .is_empty()
    );
    assert!(cache.take(&[range(0x2000)]).is_none(), "other guest RAM");
    assert!(
        cache.take(&[range(0x1000), range(0x3000)]).is_none(),
        "more guest RAM"
    );
    let kept = cache.take(&[range(0x1000)]).expect("the same guest RAM");
    assert_eq!(kept.memory, vk::DeviceMemory::from_raw(1));
    assert!(cache.take(&[range(0x1000)]).is_none(), "taken once");
    // Past the bounds the oldest go, for the caller to free.
    let mut evicted = 0;
    for index in 0..ImportCache::COUNT as u64 + 2 {
        evicted += cache
            .keep(
                alloc::vec![range(index << 12)],
                vk::DeviceMemory::from_raw(index + 2),
                exported(4096),
            )
            .len();
    }
    assert_eq!(evicted, 2);
    assert!(cache.take(&[range(0)]).is_none(), "the oldest went");
    let big = cache.keep(
        alloc::vec![range(1 << 40)],
        vk::DeviceMemory::from_raw(99),
        exported(ImportCache::BYTES),
    );
    assert_eq!(
        big.len(),
        ImportCache::COUNT,
        "every older one goes for the bytes"
    );
    assert_eq!(cache.drain().count(), 1);
}
