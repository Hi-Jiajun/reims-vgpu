//! The rail's modules, composed on whatever GPU this machine has.
//!
//! Every module in this crate deliberately splits its fallible bookkeeping from
//! the Vulkan handles, so each one is unit-tested with no device present. That
//! leaves exactly one thing untested: whether the pieces are true *about a
//! driver*. A ring that recycles a slot when its timeline point is reached is
//! correct arithmetic; whether the point it was given is the one the GPU
//! actually signals is a fact about a submission.
//!
//! So this test does the smallest thing that can be wrong end to end. It fills
//! a buffer on the GPU, waits on the timeline the submission signalled, and
//! reads the bytes back. A rail that got the timeline value, the queue family,
//! the memory type or the coherence rule wrong fails here and passes every
//! unit test in the crate.
//!
//! A machine with no Vulkan prints that it had none and asserts nothing
//! further. It says which of the two happened, so a silently skipped run is
//! distinguishable from a passing one in the output.

use ash::vk;
use reims_vgpu_core::identity::DeviceEpoch as EpochId;
use reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM;
use reims_vgpu_core::texture_shape::{TextureKind, TextureShape, TextureUsage};
use reims_vgpu_vulkan::buffer;
use reims_vgpu_vulkan::device::DeviceEpoch;
use reims_vgpu_vulkan::host::VulkanHost;
use reims_vgpu_vulkan::image;
use reims_vgpu_vulkan::memory::{select_memory_type, MappedMemoryKind, MemoryClass};
use reims_vgpu_vulkan::placement::Route;
use reims_vgpu_vulkan::pools::WorkerPool;
use reims_vgpu_vulkan::timeline::Timeline;
use reims_vgpu_vulkan::view;

/// What the GPU writes, and what the CPU has to read back.
const PATTERN: u32 = 0xABCD_EF01;
const BYTES: u64 = 256;
const DEPTH: usize = 2;
/// Generous: this is a `vkCmdFillBuffer` of 256 bytes. A timeout here is a
/// failure and not a slow machine.
const TIMEOUT_NS: u64 = 5_000_000_000;

#[test]
fn a_filled_buffer_reads_back_after_the_timeline_this_rail_reserved() {
    let Ok(host) = VulkanHost::open("reims-vgpu-vulkan integration") else {
        println!("no real device: nothing to compose");
        return;
    };
    println!("composing on: {}", host.report_line());

    let census = host.census();
    let mut epoch = DeviceEpoch::create(
        host.instance(),
        host.physical_device(),
        census,
        EpochId::FIRST,
    )
    .expect("the driver refused a set its own census admitted");

    // One owner for one `VkQueue`, taken from this epoch's ledger.
    let family = census.queues().universal().index;
    let owner = epoch
        .queues()
        .claim_in(family, 0)
        .expect("the chosen family has queue zero");
    let device = epoch.device().clone();
    // SAFETY: the family and index came from an owner this epoch handed out.
    let queue = unsafe { device.get_device_queue(owner.family(), owner.index()) };

    // The timeline the whole test is about. Zero initial value: the rail never
    // hands out zero as a point, so a retirement queued against a fresh
    // timeline cannot be collected before its work was recorded.
    let mut type_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let semaphore = unsafe {
        device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut type_info),
            None,
        )
    }
    .expect("timeline semaphores are the support floor");
    let mut timeline = Timeline::adopt(semaphore);

    // One pool per worker, on the family the owner names — a command buffer
    // submitted to any other family is invalid usage, which is why the pool
    // carries it.
    let pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(owner.family())
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )
    }
    .expect("a command pool on the chosen family");
    let buffers = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(DEPTH as u32),
        )
    }
    .expect("command buffers");
    let mut worker = WorkerPool::adopt(pool, owner.family(), buffers);
    assert_eq!(worker.ring().depth(), DEPTH);

    // The buffer the GPU writes and the CPU reads. `Readback` is the class
    // whose whole point is that coherence is a preference and not a
    // requirement, so which type this lands on is a real host answer rather
    // than a constant — the run prints it. On the Intel ARL iGPU this was
    // developed against it is a `HOST_CACHED` non-coherent type, so the
    // invalidate below is the arm that executes there.
    let buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(BYTES)
                .usage(vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .expect("a transfer destination");
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let properties = unsafe {
        host.instance()
            .get_physical_device_memory_properties(host.physical_device())
    };

    let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::default();
    let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut maintenance3);
    unsafe {
        host.instance()
            .get_physical_device_properties2(host.physical_device(), &mut properties2);
    }

    let request = census.memory().topology.request(MemoryClass::Readback);
    let pick = select_memory_type(
        &properties,
        requirements.memory_type_bits,
        &request,
        requirements.size,
        maintenance3.max_memory_allocation_size,
    )
    .expect("a readback type exists on every Vulkan device");
    println!(
        "readback type={} heap={} coherent={}",
        pick.index,
        pick.heap_index,
        MappedMemoryKind::of(&properties, pick.index).coherent
    );

    let memory = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(pick.index),
            None,
        )
    }
    .expect("the selected type allocates");
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }.expect("bind");

    // Record and submit. The point comes from the rail's cursor, and the whole
    // question is whether the GPU signals that value.
    let lease = worker.ring_mut().begin().expect("a free slot");
    let command = worker.buffers()[lease.slot()];
    unsafe {
        device
            .begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("begin");
        device.cmd_fill_buffer(command, buffer, 0, vk::WHOLE_SIZE, PATTERN);
        device.end_command_buffer(command).expect("end");
    }

    let point = timeline.reserve();
    assert_ne!(point.0, 0, "zero is never handed out");
    let commands = [command];
    let signals = [semaphore];
    let values = [point.0];
    let mut timeline_info =
        vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
    let submit = vk::SubmitInfo::default()
        .command_buffers(&commands)
        .signal_semaphores(&signals)
        .push_next(&mut timeline_info);
    unsafe { device.queue_submit(queue, &[submit], vk::Fence::null()) }.expect("submit");
    worker.ring_mut().submitted(lease, point);

    // Exhaustion refuses rather than blocking or recycling a slot the GPU may
    // still be reading. One slot is in flight and one is taken; the third
    // request has nowhere to go.
    let held = worker.ring_mut().begin().expect("the second slot");
    let refused = worker
        .ring_mut()
        .begin()
        .expect_err("every slot is spoken for");
    assert_eq!(refused.depth, DEPTH);
    assert_eq!(refused.in_flight, 1, "the held lease is not in flight");
    worker.ring_mut().abandon(held);

    // Wait on the point the rail reserved, then ask the driver where the
    // timeline actually got to.
    // SAFETY: `device` is the device the semaphore was created on and it has
    // not been destroyed.
    unsafe { timeline.wait(&device, point, TIMEOUT_NS) }.expect("the GPU signalled the point");
    // SAFETY: as above.
    let reached = unsafe { timeline.poll(&device) }.expect("a forwards reading");
    assert!(
        reached.reached(point),
        "the driver signalled {reached:?}, below the reserved {point:?}"
    );

    // Which is what makes the slot recordable again.
    assert_eq!(worker.ring_mut().recycle(reached), 1);
    assert_eq!(worker.ring().in_flight(), 0);
    assert!(worker.ring().resettable());
    let after = worker.ring_mut().begin().expect("the recycled slot");
    worker.ring_mut().abandon(after);

    // And the bytes. A non-coherent mapping owes an invalidate first; the
    // memory module is what says whether this one does.
    let kind = MappedMemoryKind::of(&properties, pick.index);
    let mapped =
        unsafe { device.map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty()) }
            .expect("host visible");
    if !kind.coherent {
        let range = vk::MappedMemoryRange::default()
            .memory(memory)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        unsafe { device.invalidate_mapped_memory_ranges(&[range]) }.expect("invalidate");
    }
    // SAFETY: the mapping covers `requirements.size >= BYTES` bytes, the GPU is
    // finished with them, and `u32` has no alignment requirement the driver's
    // mapping does not already meet.
    let words = unsafe { std::slice::from_raw_parts(mapped.cast::<u32>(), (BYTES / 4) as usize) };
    assert!(
        words.iter().all(|w| *w == PATTERN),
        "the GPU's fill did not reach the CPU: {:#010x} at {:?}",
        words[0],
        words.iter().position(|w| *w != PATTERN)
    );
    unsafe { device.unmap_memory(memory) };

    println!(
        "filled and read back {} words at {PATTERN:#010x}",
        words.len()
    );

    // Teardown in creation order's reverse. The epoch's own `Drop` idles the
    // device, but these are this test's objects and it gives them back itself.
    unsafe {
        device.device_wait_idle().expect("idle before teardown");
        device.destroy_buffer(buffer, None);
        device.free_memory(memory, None);
        device.destroy_command_pool(worker.pool(), None);
        device.destroy_semaphore(timeline.semaphore(), None);
    }
    epoch.queues().release(owner);
    // `epoch`'s own `Drop` idles and destroys the device. `device` is a clone
    // of its loader table and owns nothing, so it needs no teardown of its own.
    drop(epoch);
}

/// The image half: a decoded texture declaration reaching a real `VkImage`
/// through the query that admitted it.
///
/// The plan is pure and the admission is pure, so both are unit-tested here
/// with invented `VkImageFormatProperties`. What no unit test can check is
/// whether the tuple this rail builds is one a driver answers at all — a
/// usage bit a format does not support turns
/// `vkGetPhysicalDeviceImageFormatProperties` into
/// `ERROR_FORMAT_NOT_SUPPORTED`, and every downstream number this rail then
/// validates against would be zero.
///
/// So this asks the driver about exactly the tuple the plan names, admits the
/// plan against exactly that answer, and creates the image from exactly the
/// admitted plan. A refusal anywhere is this rail's, with both numbers in it.
#[test]
fn a_decoded_texture_becomes_an_image_the_driver_admitted() {
    let Ok(host) = VulkanHost::open("reims-vgpu-vulkan image integration") else {
        println!("no real device: nothing to allocate");
        return;
    };
    let census = host.census();
    let epoch = DeviceEpoch::create(
        host.instance(),
        host.physical_device(),
        census,
        EpochId::FIRST,
    )
    .expect("the driver refused a set its own census admitted");
    let device = epoch.device().clone();

    // A cube array with a mip chain: the shape whose layer count is the one a
    // caller deriving it itself gets wrong, and whose mip count a device is
    // entitled to cap below the pyramid.
    let declaration = TextureShape {
        kind: TextureKind::CubeArray.ordinal(),
        width: 64,
        height: 64,
        depth: 1,
        mipmap_level_count: 7,
        sample_count: 1,
        array_length: 2,
        pixel_format: MTL_FORMAT_RGBA8_UNORM,
        usage: TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
    };
    let texture = declaration
        .checked()
        .expect("a declaration the guest API admits");
    assert_eq!(texture.layers(), 12);

    let plan = image::plan(
        texture,
        vk::Format::R8G8B8A8_UNORM,
        Route::HostStaging {
            working: MemoryClass::DeviceLocal,
        },
    )
    .expect("a plannable texture");
    let query = plan.query();

    // The one thing that needs a device: what this exact combination's limits
    // are. Not the general `maxImageDimension2D`, which is the ceiling over
    // every usage and would admit tuples this one does not.
    let reported = unsafe {
        host.instance().get_physical_device_image_format_properties(
            host.physical_device(),
            query.format,
            query.image_type,
            query.tiling,
            query.usage,
            query.flags,
        )
    }
    .expect("RGBA8 sampled and color-attachable is universal on Vulkan 1.2");
    println!(
        "image tuple extent={}x{}x{} mips={} layers={} samples=0x{:x}",
        reported.max_extent.width,
        reported.max_extent.height,
        reported.max_extent.depth,
        reported.max_mip_levels,
        reported.max_array_layers,
        reported.sample_counts.as_raw(),
    );

    let admitted = plan
        .admitted(reported)
        .unwrap_or_else(|refusal| panic!("{refusal}"));
    let info = admitted.create_info();
    assert_eq!(info.array_layers, 12);
    assert_eq!(info.mip_levels, 7);
    assert!(info.flags.contains(vk::ImageCreateFlags::CUBE_COMPATIBLE));

    let image = unsafe { device.create_image(&info, None) }
        .expect("the driver takes what its own reported properties admitted");

    // And it is allocatable: an image plan that cannot be backed is a plan
    // that passed every check and produces nothing.
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let properties = unsafe {
        host.instance()
            .get_physical_device_memory_properties(host.physical_device())
    };
    let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::default();
    let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut maintenance3);
    unsafe {
        host.instance()
            .get_physical_device_properties2(host.physical_device(), &mut properties2);
    }
    let pick = select_memory_type(
        &properties,
        requirements.memory_type_bits,
        &census.memory().topology.request(MemoryClass::DeviceLocal),
        requirements.size,
        maintenance3.max_memory_allocation_size,
    )
    .expect("a device-local type exists for a sampled colour image");
    let memory = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(pick.index),
            None,
        )
    }
    .expect("the selected type allocates");
    unsafe { device.bind_image_memory(image, memory, 0) }.expect("bind");
    println!(
        "image bytes={} type={} for {} subresources",
        requirements.size,
        pick.index,
        texture.subresources()
    );

    // A device that caps this tuple below the declaration refuses with both
    // numbers rather than clamping. Asked here against the driver's own answer
    // narrowed by one, so the assertion is about this rail's comparison and not
    // about any particular host's limits.
    let one_layer_short = vk::ImageFormatProperties {
        max_array_layers: 11,
        ..reported
    };
    assert_eq!(
        plan.admitted(one_layer_short),
        Err(image::Refusal::LayersBeyondDevice {
            declared: 12,
            max: 11,
        })
    );

    // Every view this texture is addressable through, created for real. The
    // expansion is pure arithmetic and unit-tested as such; what a driver has
    // to agree with is that eighty-four single-slice views over a
    // cube-compatible image are legal, which no arithmetic can establish.
    let whole = view::whole(texture, vk::Format::R8G8B8A8_UNORM).create_info(image);
    let sampled = unsafe { device.create_image_view(&whole, None) }
        .expect("a cube-array view over a cube-compatible image");
    let expansion = view::attachments(texture, vk::Format::R8G8B8A8_UNORM);
    assert_eq!(expansion.len() as u32, texture.subresources());
    let attachments: Vec<vk::ImageView> = expansion
        .iter()
        .map(|attachment| {
            unsafe { device.create_image_view(&attachment.plan.create_info(image), None) }
                .unwrap_or_else(|e| {
                    panic!(
                        "level {} slice {} refused: {e}",
                        attachment.level, attachment.slice
                    )
                })
        })
        .collect();
    println!("views whole=1 attachments={}", attachments.len());

    // The buffer half, through the same census. `maxBufferSize` is core in 1.3
    // and an extension below it, so whether this host reported one at all is a
    // real answer and the run prints it.
    let limits = census.buffers();
    println!("buffer max={:?}", limits.max_buffer_size);
    let buffer_plan = buffer::plan(
        1 << 16,
        Route::HostStaging {
            working: MemoryClass::DeviceLocal,
        },
        limits,
    )
    .unwrap_or_else(|refusal| panic!("{refusal}"));
    // Every operation class at once. A driver that rejected the combination
    // would fail here, which is the only thing about the wide usage set that a
    // unit test cannot check.
    let wide = unsafe { device.create_buffer(&buffer_plan.create_info(), None) }
        .expect("a buffer bindable as every class");
    unsafe { device.destroy_buffer(wide, None) };

    unsafe {
        device.device_wait_idle().expect("idle before teardown");
        for attachment in attachments {
            device.destroy_image_view(attachment, None);
        }
        device.destroy_image_view(sampled, None);
        device.destroy_image(image, None);
        device.free_memory(memory, None);
    }
    drop(epoch);
}
