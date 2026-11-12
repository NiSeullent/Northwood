// SPDX-License-Identifier: GPL-2.0-or-later

extern crate std;

use super::*;
use crate::metal::{
    native::{Loader, NativeBackend, Target, application, probe_software},
    vulkan::{BufferCreate, BufferUsage},
};
use core::ffi::{c_char, c_int, c_void};

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(name: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(library: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(library: *mut c_void) -> c_int;
}

#[test]
#[ignore = "requires the pinned lavapipe ICD selected by VK_DRIVER_FILES"]
fn lavapipe_import_updates_the_original_leased_mapping() {
    assert!(std::env::var_os("VK_DRIVER_FILES").is_some());
    // SAFETY: the C strings outlive the loader calls. The library stays open
    // until all Vulkan objects created through it have been destroyed.
    let library = unsafe { dlopen(c"libvulkan.so.1".as_ptr(), 2) };
    assert!(!library.is_null(), "Vulkan loader");
    let symbol = unsafe { dlsym(library, c"vkGetInstanceProcAddr".as_ptr()) };
    assert!(!symbol.is_null(), "Vulkan entry point");
    let get_instance_proc_addr =
        unsafe { core::mem::transmute::<*mut c_void, vk::PFN_vkGetInstanceProcAddr>(symbol) };
    let log = |line: &str| std::println!("{line}");
    let entry = unsafe {
        ash::Entry::from_static_fn(ash::StaticFn {
            get_instance_proc_addr,
        })
    };
    let app = application();
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app),
            None,
        )
    }
    .unwrap();
    let (physical, _, snapshot, family) =
        NativeBackend::select(&instance, Target::Software, log).unwrap();
    let probed = unsafe { probe_software(get_instance_proc_addr, log) }.unwrap();
    assert_eq!(probed.physical.identity, snapshot.identity);
    let alignment = probed.host_storage_alignment.unwrap();
    assert_eq!(alignment.get(), PAGE);

    let owner = exporter();
    if let Err(expected) = probed.check() {
        let loader = Loader {
            get_instance_proc_addr,
            log,
            submit_thread: None,
            refusals: None,
            relax: || {},
        };
        let refused = unsafe {
            NativeBackend::new_software(loader, Box::new(Exporter(owner.0.clone())), |_| {
                panic!("shader builder after refusal")
            })
        };
        assert_eq!(refused.err(), Some(expected));
    }

    let priority = [1.0];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family)
        .queue_priorities(&priority)];
    let names = [
        ext::external_memory_host::NAME.as_ptr(),
        ash::khr::maintenance5::NAME.as_ptr(),
    ];
    let mut features = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true);
    let mut maintenance = vk::PhysicalDeviceMaintenance5FeaturesKHR::default().maintenance5(true);
    let info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues)
        .enabled_extension_names(&names)
        .push_next(&mut features)
        .push_next(&mut maintenance);
    let device = unsafe { instance.create_device(physical, &info, None) }.unwrap();
    let mut imports = Imports::new(
        Box::new(Exporter(owner.0.clone())),
        &instance,
        &device,
        physical,
        alignment,
    );
    let ranges = [MemoryRange {
        address: PAGE,
        bytes: PAGE,
    }];
    const DESCRIPTOR: u64 = 8;
    for token in [DESCRIPTOR, IMPORT_TOKEN_WRITABLE] {
        assert_eq!(
            imports
                .allocate(
                    &device,
                    &snapshot.memory,
                    &ranges,
                    NonZeroU64::new(token).unwrap(),
                )
                .err(),
            Some(Error::Unsupported(
                "host allocation requires a bidirectional guest wire"
            ))
        )
    }
    assert_eq!(alloc::sync::Arc::strong_count(&owner.0), 2);
    let writable = NonZeroU64::new(DESCRIPTOR | IMPORT_TOKEN_WRITABLE).unwrap();
    let (memory, import) = imports
        .allocate(&device, &snapshot.memory, &ranges, writable)
        .unwrap();
    assert_eq!(import.token, writable);
    assert_eq!(alloc::sync::Arc::strong_count(&owner.0), 3);
    std::println!("untagged and empty descriptor tokens refused; writable token kept intact");
    let data = owner.0.0.get().cast::<u8>();
    // SAFETY: no submission has started and the owner gives the original page.
    unsafe { data.write_bytes(0xa7, PAGE as usize) };
    const COUNT: u64 = 256;
    const OFFSET: u64 = PAGE / 2;
    let info = BufferInfo {
        size: PAGE - OFFSET,
        usage: BufferUsage::TRANSFER_SRC
            | BufferUsage::TRANSFER_DST
            | BufferUsage::STORAGE_BUFFER
            | BufferUsage::SHADER_DEVICE_ADDRESS,
        flags: BufferCreate::EMPTY,
    };
    imports.admit_buffer(&instance, &info).unwrap();
    let buffer = objects::imported_buffer(&info, HOST_ALLOCATION, |create| {
        unsafe { device.create_buffer(create, None) }.map_err(failure)
    })
    .unwrap();
    let needs = unsafe { device.get_buffer_memory_requirements(buffer) };
    import.check_binding(needs, OFFSET).unwrap();
    unsafe { device.bind_buffer_memory(buffer, memory, OFFSET) }.unwrap();
    imports.owned.insert(raw(memory), import);

    let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(family);
    let pool = unsafe { device.create_command_pool(&pool_info, None) }.unwrap();
    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .command_buffer_count(1);
    let cmd = unsafe { device.allocate_command_buffers(&alloc) }.unwrap()[0];
    let barrier = [vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ)];
    let queue = unsafe { device.get_device_queue(family, 0) };
    const PATTERN: u32 = 0x1357_9bdf;
    unsafe {
        device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .unwrap();
        device.cmd_fill_buffer(cmd, buffer, 0, COUNT, PATTERN);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &barrier,
            &[],
            &[],
        );
        device.end_command_buffer(cmd).unwrap();
        let submits = [vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&cmd))];
        device
            .queue_submit(queue, &submits, vk::Fence::null())
            .unwrap();
        device.queue_wait_idle(queue).unwrap();
    }
    // SAFETY: the host barrier and queue idle made the transfer available.
    let result = unsafe { core::slice::from_raw_parts(data, PAGE as usize) };
    let (start, end) = (OFFSET as usize, (OFFSET + COUNT) as usize);
    assert!(
        result[..start]
            .iter()
            .chain(&result[end..])
            .all(|byte| *byte == 0xa7)
    );
    assert!(
        result[start..end]
            .chunks_exact(4)
            .all(|bytes| bytes == PATTERN.to_le_bytes())
    );
    std::println!("leased host allocation: {COUNT} bytes at offset {OFFSET}; guard bytes verified");
    stage_bgra_rows(&device, &snapshot.memory, queue, pool, buffer, data, OFFSET);
    unsafe {
        device.destroy_command_pool(pool, None);
        device.destroy_buffer(buffer, None);
    }
    imports.release(&device);
    assert_eq!(alloc::sync::Arc::strong_count(&owner.0), 2);
    drop(imports);
    assert_eq!(alloc::sync::Arc::strong_count(&owner.0), 1);
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
        assert_eq!(dlclose(library), 0);
    }
    std::println!("original leased mapping survived until Vulkan memory was released");
}

/// A separate image makes its initial UNDEFINED transition before taking any
/// pixels from the external buffer. Copies touch visible pixels, leaving the
/// original row padding and unrelated allocations alone.
fn stage_bgra_rows(
    device: &ash::Device,
    memory: &Memory,
    queue: vk::Queue,
    pool: vk::CommandPool,
    buffer: vk::Buffer,
    data: *mut u8,
    buffer_offset: u64,
) {
    const WIDTH: u32 = 17;
    const HEIGHT: u32 = 3;
    const PITCH: usize = 128;
    const READBACK: u64 = 512;
    const IN: [u8; 4] = [0x43, 0x5d, 0x7b, 0xe1];
    const OUT: [u8; 4] = [0, 0, 255, 255];
    let extent = vk::Extent3D {
        width: WIDTH,
        height: HEIGHT,
        depth: 1,
    };
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::B8G8R8A8_UNORM)
        .extent(extent)
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST);
    // SAFETY: the device is alive and the queue is idle from the buffer test.
    let image = unsafe { device.create_image(&image_info, None) }.unwrap();
    let needs = unsafe { device.get_image_memory_requirements(image) };
    let request = MemoryRequest {
        required: MemoryProperties::DEVICE_LOCAL,
        preferred: MemoryProperties::EMPTY,
        mapped: false,
    };
    let image_memory = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(needs.size)
                .memory_type_index(
                    objects::memory_candidates(memory, needs.memory_type_bits, request)[0],
                ),
            None,
        )
    }
    .unwrap();
    unsafe { device.bind_image_memory(image, image_memory, 0) }.unwrap();
    let cmd = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .command_buffer_count(1),
        )
    }
    .unwrap()[0];
    let range = objects::whole(vk::ImageAspectFlags::COLOR);
    let initial = [vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)];
    let copy = vk::BufferImageCopy::default()
        .buffer_row_length((PITCH / 4) as u32)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1),
        )
        .image_extent(extent);
    let transfer = vk::PipelineStageFlags::TRANSFER;
    let barrier = |source, target| {
        [vk::MemoryBarrier::default()
            .src_access_mask(source)
            .dst_access_mask(target)]
    };
    let read = vk::AccessFlags::TRANSFER_READ;
    let write = vk::AccessFlags::TRANSFER_WRITE;
    // SAFETY: the leased original page is idle. Every device copy stays
    // within the buffer and the explicit barriers order conflicting accesses.
    unsafe {
        let start = data.add(buffer_offset as usize);
        start.write_bytes(0xa7, (PAGE - buffer_offset) as usize);
        for row in 0..HEIGHT as usize {
            for column in 0..WIDTH as usize {
                start
                    .add(row * PITCH + column * 4)
                    .cast::<[u8; 4]>()
                    .write(IN);
            }
        }
        device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .unwrap();
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            transfer,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &initial,
        );
        device.cmd_copy_buffer_to_image(cmd, buffer, image, vk::ImageLayout::GENERAL, &[copy]);
        device.cmd_pipeline_barrier(
            cmd,
            transfer,
            transfer,
            vk::DependencyFlags::empty(),
            &barrier(write, read),
            &[],
            &[],
        );
        device.cmd_copy_image_to_buffer(
            cmd,
            image,
            vk::ImageLayout::GENERAL,
            buffer,
            &[copy.buffer_offset(READBACK)],
        );
        device.cmd_pipeline_barrier(
            cmd,
            transfer,
            transfer,
            vk::DependencyFlags::empty(),
            &barrier(read, write),
            &[],
            &[],
        );
        device.cmd_clear_color_image(
            cmd,
            image,
            vk::ImageLayout::GENERAL,
            &vk::ClearColorValue {
                float32: [1.0, 0.0, 0.0, 1.0],
            },
            &[range],
        );
        device.cmd_pipeline_barrier(
            cmd,
            transfer,
            transfer,
            vk::DependencyFlags::empty(),
            &barrier(write, read),
            &[],
            &[],
        );
        device.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::GENERAL, buffer, &[copy]);
        device.cmd_pipeline_barrier(
            cmd,
            transfer,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &barrier(write, vk::AccessFlags::HOST_READ),
            &[],
            &[],
        );
        device.end_command_buffer(cmd).unwrap();
        let submits = [vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&cmd))];
        device
            .queue_submit(queue, &submits, vk::Fence::null())
            .unwrap();
        device.queue_wait_idle(queue).unwrap();
        let page = core::slice::from_raw_parts(data, PAGE as usize);
        for (offset, pixel) in [(0, OUT), (READBACK as usize, IN)] {
            let base = buffer_offset as usize + offset;
            for row in 0..HEIGHT as usize {
                let start = base + row * PITCH;
                let end = start + WIDTH as usize * 4;
                assert!(page[start..end].chunks_exact(4).all(|bytes| bytes == pixel));
                assert!(page[end..start + PITCH].iter().all(|byte| *byte == 0xa7));
            }
        }
        assert!(
            page[..buffer_offset as usize]
                .iter()
                .all(|byte| *byte == 0xa7)
        );
        assert!(
            page[buffer_offset as usize + READBACK as usize + HEIGHT as usize * PITCH..]
                .iter()
                .all(|byte| *byte == 0xa7)
        );
        device.destroy_image(image, None);
        device.free_memory(image_memory, None);
    }
    std::println!("BGRA 17×3 upload and download preserved 128-byte rows and guard bytes");
}
