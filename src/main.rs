// This code roughly follows https://vulkan-tutorial.com/ - Drawing a triangle

use ash::extensions::ext::DebugUtils as DebugUtilsLoader;
use ash::extensions::khr::{Surface as SurfaceLoader, Swapchain as SwapchainLoader};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::{util, vk, Device};
use byte_strings::c_str;
use std::ffi::CStr;
use ultraviolet::{Mat4, Vec3};
use winit::{
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, StartCause, VirtualKeyCode, WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const FRAMES_IN_FLIGHT: usize = 2;

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    let severity = format!("{:?}", message_severity).to_lowercase();
    let ty = format!("{:?}", message_type).to_lowercase();
    println!("[Debug Msg][{}][{}] {:?}", severity, ty, message);
    vk::FALSE
}

fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Vulkan examples")
        .build(&event_loop)?;

    let entry = unsafe { ash::Entry::new() }?;

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Instance
    let app_info = vk::ApplicationInfo::builder()
        .application_name(&c_str!("Vulkan examples"))
        .application_version(vk::make_version(1, 0, 0))
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    let mut instance_extensions = ash_window::enumerate_required_extensions(&window)?;
    instance_extensions.push(DebugUtilsLoader::name());

    let instance_extension_ptrs = instance_extensions
        .iter()
        .map(|ext| ext.as_ptr())
        .collect::<Vec<_>>();

    let instance_layers = [c_str!("VK_LAYER_KHRONOS_validation")];

    let instance_layer_ptrs = instance_layers
        .iter()
        .map(|layer| layer.as_ptr())
        .collect::<Vec<_>>();

    let device_extensions = [SwapchainLoader::name()];

    let device_extension_pointers = device_extensions
        .iter()
        .map(|ext| ext.as_ptr())
        .collect::<Vec<_>>();

    let device_layers = [c_str!("VK_LAYER_KHRONOS_validation")];

    let device_layer_pointers = device_layers
        .iter()
        .map(|layer| layer.as_ptr())
        .collect::<Vec<_>>();

    let mut debug_messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::all() ^ vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
        )
        .pfn_user_callback(Some(vulkan_debug_utils_callback));

    let instance_info = vk::InstanceCreateInfo::builder()
        .push_next(&mut debug_messenger_info)
        .application_info(&app_info)
        .enabled_extension_names(&instance_extension_ptrs)
        .enabled_layer_names(&instance_layer_ptrs);

    let instance = unsafe { entry.create_instance(&instance_info, None) }?;

    let debug_utils_loader = DebugUtilsLoader::new(&entry, &instance);
    let debug_messenger =
        unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_messenger_info, None) }?;

    let surface_loader = SurfaceLoader::new(&entry, &instance);

    let surface = unsafe { ash_window::create_surface(&entry, &instance, &window, None) }?;

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Physical_devices_and_queue_families
    let (physical_device, queue_family, format, device_properties) =
        unsafe { instance.enumerate_physical_devices() }?
            .into_iter()
            .filter_map(|physical_device| unsafe {
                let queue_family = match instance
                    .get_physical_device_queue_family_properties(physical_device)
                    .into_iter()
                    .enumerate()
                    .position(|(i, queue_family_properties)| {
                        queue_family_properties
                            .queue_flags
                            .contains(vk::QueueFlags::GRAPHICS)
                            && surface_loader
                                .get_physical_device_surface_support(
                                    physical_device,
                                    i as u32,
                                    surface,
                                )
                                .unwrap()
                    }) {
                    Some(queue_family) => queue_family as u32,
                    None => return None,
                };

                let formats = surface_loader
                    .get_physical_device_surface_formats(physical_device, surface)
                    .unwrap();
                let format = match formats
                    .iter()
                    .find(|surface_format| {
                        surface_format.format == vk::Format::B8G8R8A8_SRGB
                            && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                    })
                    .or_else(|| formats.get(0))
                {
                    Some(surface_format) => *surface_format,
                    None => return None,
                };

                let supported_device_extensions = instance
                    .enumerate_device_extension_properties(physical_device)
                    .unwrap();

                let has_needed_extensions = device_extensions.iter().all(|device_extension| {
                    supported_device_extensions.iter().any(|properties| {
                        &CStr::from_ptr(properties.extension_name.as_ptr()) == device_extension
                    })
                });

                if !has_needed_extensions {
                    return None;
                }

                let device_properties = instance.get_physical_device_properties(physical_device);
                Some((physical_device, queue_family, format, device_properties))
            })
            .max_by_key(|(.., properties)| match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 2,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 0,
            })
            .expect("No suitable physical device found");

    println!("Using physical device: {:?}", unsafe {
        CStr::from_ptr(device_properties.device_name.as_ptr())
    });

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Logical_device_and_queues
    let queue_info = [*vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family)
        .queue_priorities(&[1.0])];
    let features = vk::PhysicalDeviceFeatures::builder();

    let device_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_info)
        .enabled_features(&features)
        .enabled_extension_names(&device_extension_pointers)
        .enabled_layer_names(&device_layer_pointers);

    let device = unsafe { instance.create_device(physical_device, &device_info, None) }?;
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    let mut allocator = vk_mem::Allocator::new(&vk_mem::AllocatorCreateInfo {
        device: device.clone(),
        physical_device,
        instance: instance.clone(),
        // Defaults
        flags: vk_mem::AllocatorCreateFlags::NONE,
        preferred_large_heap_block_size: 0,
        frame_in_use_count: 0,
        heap_size_limits: None,
    })?;

    let (verts, indices) = cube_verts();

    let vertex_buffer = Buffer::new(
        bytemuck::cast_slice(&verts),
        vk::BufferUsageFlags::VERTEX_BUFFER,
        &allocator,
        queue_family,
    )?;

    let index_buffer = Buffer::new(
        bytemuck::cast_slice(&indices),
        vk::BufferUsageFlags::INDEX_BUFFER,
        &allocator,
        queue_family,
    )?;

    let num_indices = indices.len() as u32;

    // https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Render_passes
    let attachments = [
        *vk::AttachmentDescription::builder()
            .format(format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
        *vk::AttachmentDescription::builder()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];

    let color_attachment_refs = [*vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

    let depth_attachment_ref = *vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let subpasses = [*vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)
        .depth_stencil_attachment(&depth_attachment_ref)];
    let dependencies = [*vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

    let render_pass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    let render_pass = unsafe { device.create_render_pass(&render_pass_info, None) }?;

    let pipelines = Pipelines::new(&device, render_pass)?;

    let surface_caps = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
    }?;
    let mut image_count = surface_caps.min_image_count + 1;
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }

    let window_size = window.inner_size();

    let mut extent = vk::Extent2D {
        width: window_size.width,
        height: window_size.height,
    };

    let mut depth_buffer = DepthBuffer::new(extent.width, extent.height, &device, &allocator)?;

    let mut swapchain_info = *vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let swapchain_loader = SwapchainLoader::new(&instance, &device);
    let mut swapchain = Swapchain::new(
        swapchain_loader.clone(),
        &device,
        swapchain_info,
        render_pass,
        depth_buffer.view,
    )?;

    // https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Command_buffers
    let command_pool_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let command_pool = unsafe { device.create_command_pool(&command_pool_info, None) }?;

    let cmd_buf_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(swapchain.framebuffers.len() as _);
    let cmd_bufs = unsafe { device.allocate_command_buffers(&cmd_buf_allocate_info) }?;

    let mut syncronisation = Syncronisation::new(&device, swapchain.framebuffers.len());

    let view_matrix = Mat4::look_at(Vec3::new(-1.0, 0.0, 0.0), Vec3::zero(), Vec3::unit_y());

    let perspective_matrix = ultraviolet::projection::perspective_infinite_z_vk(
        59.0_f32.to_radians(),
        extent.width as f32 / extent.height as f32,
        0.1,
    );

    let mut perspective_view_matrix = perspective_matrix * view_matrix;
    let mut cube_rotation = 0.0;

    let mut frame = 0;
    event_loop.run(move |event, _, control_flow| match event {
        Event::NewEvents(StartCause::Init) => {
            *control_flow = ControlFlow::Poll;
        }
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(size) => {
                extent.width = size.width as u32;
                extent.height = size.height as u32;

                let perspective_matrix = ultraviolet::projection::perspective_infinite_z_vk(
                    59.0_f32.to_radians(),
                    extent.width as f32 / extent.height as f32,
                    0.1,
                );
                perspective_view_matrix = perspective_matrix * view_matrix;

                swapchain_info.image_extent = extent;

                unsafe {
                    device.device_wait_idle().unwrap();
                }

                depth_buffer.cleanup(&device, &allocator).unwrap();

                depth_buffer =
                    DepthBuffer::new(extent.width, extent.height, &device, &allocator).unwrap();

                unsafe {
                    swapchain.cleanup(&device);
                }
                swapchain = Swapchain::new(
                    swapchain_loader.clone(),
                    &device,
                    swapchain_info,
                    render_pass,
                    depth_buffer.view,
                )
                .unwrap();
            }
            _ => (),
        },
        Event::DeviceEvent { event, .. } => match event {
            DeviceEvent::Key(KeyboardInput {
                virtual_keycode: Some(keycode),
                state,
                ..
            }) => match (keycode, state) {
                (VirtualKeyCode::Escape, ElementState::Released) => {
                    *control_flow = ControlFlow::Exit
                }
                _ => (),
            },
            _ => (),
        },
        Event::MainEventsCleared => {
            cube_rotation += 0.01;

            unsafe {
                device
                    .wait_for_fences(&[syncronisation.in_flight_fences[frame]], true, u64::MAX)
                    .unwrap();
            }

            match unsafe {
                swapchain.swapchain_loader.acquire_next_image(
                    swapchain.swapchain,
                    u64::MAX,
                    syncronisation.image_available_semaphores[frame],
                    vk::Fence::null(),
                )
            } {
                Ok((image_index, _suboptimal)) => {
                    syncronisation.wait_fences(&device, image_index as usize, frame);

                    let command_buffer = cmd_bufs[image_index as usize];
                    let framebuffer = swapchain.framebuffers[image_index as usize];

                    let cmd_buf_begin_info = vk::CommandBufferBeginInfo::builder();
                    unsafe { device.begin_command_buffer(command_buffer, &cmd_buf_begin_info) }
                        .unwrap();

                    let clear_values = [
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                        },
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    ];
                    let area = vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
                    };
                    let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                        .render_pass(render_pass)
                        .framebuffer(framebuffer)
                        .render_area(area)
                        .clear_values(&clear_values);

                    let viewport = *vk::Viewport::builder()
                        .x(0.0)
                        .y(0.0)
                        .width(extent.width as f32)
                        .height(extent.height as f32)
                        .min_depth(0.0)
                        .max_depth(1.0);

                    unsafe {
                        device.cmd_set_viewport(command_buffer, 0, &[viewport]);
                        device.cmd_set_scissor(command_buffer, 0, &[area]);

                        device.cmd_begin_render_pass(
                            command_buffer,
                            &render_pass_begin_info,
                            vk::SubpassContents::INLINE,
                        );

                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipelines.triangle_pipeline,
                        );
                        device.cmd_draw(command_buffer, 3, 1, 0, 0);

                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipelines.cube_pipeline,
                        );
                        device.cmd_push_constants(
                            command_buffer,
                            pipelines.push_constants_pipeline_layout,
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            bytemuck::bytes_of(&CubePushConstants {
                                perspective_view_matrix,
                                cube_transform: Mat4::from_translation(Vec3::new(0.0, 0.0, 0.25))
                                    * Mat4::from_scale(0.2)
                                    * Mat4::from_rotation_y(cube_rotation),
                            }),
                        );

                        device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &[vertex_buffer.buffer],
                            &[0],
                        );
                        device.cmd_bind_index_buffer(
                            command_buffer,
                            index_buffer.buffer,
                            0,
                            vk::IndexType::UINT16,
                        );
                        device.cmd_draw_indexed(command_buffer, num_indices, 1, 0, 0, 0);

                        device.cmd_end_render_pass(command_buffer);

                        device.end_command_buffer(command_buffer).unwrap();
                    }

                    let wait_semaphores = [syncronisation.image_available_semaphores[frame]];
                    let command_buffers = [command_buffer];
                    let signal_semaphores = [syncronisation.render_finished_semaphores[frame]];
                    let submit_info = vk::SubmitInfo::builder()
                        .wait_semaphores(&wait_semaphores)
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                        .command_buffers(&command_buffers)
                        .signal_semaphores(&signal_semaphores);

                    unsafe {
                        let in_flight_fence = syncronisation.in_flight_fences[frame];
                        device.reset_fences(&[in_flight_fence]).unwrap();
                        device
                            .queue_submit(queue, &[*submit_info], in_flight_fence)
                            .unwrap()
                    }

                    let swapchains = [swapchain.swapchain];
                    let image_indices = [image_index];
                    let present_info = vk::PresentInfoKHR::builder()
                        .wait_semaphores(&signal_semaphores)
                        .swapchains(&swapchains)
                        .image_indices(&image_indices);

                    // Sometimes, resizes can happen between acquiring a frame and presenting.
                    // In this case, there isn't anything we can really do so we just print an error.
                    if let Err(err) = unsafe {
                        swapchain
                            .swapchain_loader
                            .queue_present(queue, &present_info)
                    } {
                        println!("Error while presenting: {:?}", err);
                    }

                    frame = (frame + 1) % FRAMES_IN_FLIGHT;
                }
                Err(error) => println!("Next frame error: {:?}", error),
            }
        }
        Event::LoopDestroyed => unsafe {
            device.device_wait_idle().unwrap();

            syncronisation.cleanup(&device);
            depth_buffer.cleanup(&device, &allocator).unwrap();
            vertex_buffer.cleanup(&allocator).unwrap();
            index_buffer.cleanup(&allocator).unwrap();

            device.destroy_command_pool(command_pool, None);

            swapchain.cleanup(&device);

            pipelines.cleanup(&device);

            device.destroy_render_pass(render_pass, None);

            surface_loader.destroy_surface(surface, None);

            debug_utils_loader.destroy_debug_utils_messenger(debug_messenger, None);

            allocator.destroy();

            device.destroy_device(None);

            instance.destroy_instance(None);
            println!("Exited cleanly");
        },
        _ => (),
    })
}

struct DepthBuffer {
    image: vk::Image,
    allocation: vk_mem::Allocation,
    view: vk::ImageView,
}

impl DepthBuffer {
    fn new(
        width: u32,
        height: u32,
        device: &Device,
        allocator: &vk_mem::Allocator,
    ) -> anyhow::Result<Self> {
        let (image, allocation, _allocation_info) = allocator
            .create_image(
                &vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::D32_SFLOAT)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::GpuOnly,
                    flags: vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
                    ..Default::default()
                },
            )
            .map_err(|err| anyhow::anyhow!("{:?}", err))?;

        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::D32_SFLOAT)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .level_count(1)
                            .layer_count(1)
                            .build(),
                    ),
                None,
            )
        }?;

        Ok(Self {
            image,
            allocation,
            view,
        })
    }

    fn cleanup(&self, device: &Device, allocator: &vk_mem::Allocator) -> anyhow::Result<()> {
        unsafe {
            device.destroy_image_view(self.view, None);
        }

        allocator
            .destroy_image(self.image, &self.allocation)
            .map_err(|err| err.into())
    }
}

struct Buffer {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
}

impl Buffer {
    fn new(
        bytes: &[u8],
        usage: vk::BufferUsageFlags,
        allocator: &vk_mem::Allocator,
        queue_family: u32,
    ) -> anyhow::Result<Self> {
        let (buffer, allocation, _allocation_info) = allocator.create_buffer(
            &vk::BufferCreateInfo::builder()
                .size(bytes.len() as u64)
                .usage(usage)
                .queue_family_indices(&[queue_family]),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::GpuOnly,
                ..Default::default()
            },
        )?;

        let pointer = allocator.map_memory(&allocation)?;

        let slice = unsafe { std::slice::from_raw_parts_mut(pointer, bytes.len()) };

        slice.copy_from_slice(bytes);

        allocator.unmap_memory(&allocation)?;

        Ok(Self { buffer, allocation })
    }

    fn cleanup(&self, allocator: &vk_mem::Allocator) -> anyhow::Result<()> {
        allocator
            .destroy_buffer(self.buffer, &self.allocation)
            .map_err(|err| err.into())
    }
}

struct Swapchain {
    swapchain_loader: SwapchainLoader,
    swapchain: vk::SwapchainKHR,
    framebuffers: Vec<vk::Framebuffer>,
    image_views: Vec<vk::ImageView>,
    _images: Vec<vk::Image>,
}

impl Swapchain {
    fn new(
        swapchain_loader: SwapchainLoader,
        device: &Device,
        info: vk::SwapchainCreateInfoKHR,
        render_pass: vk::RenderPass,
        depth_image_view: vk::ImageView,
    ) -> anyhow::Result<Self> {
        let swapchain = unsafe { swapchain_loader.create_swapchain(&info, None) }.unwrap();
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }.unwrap();

        let image_views: Vec<_> = images
            .iter()
            .map(|swapchain_image| {
                let image_view_info = vk::ImageViewCreateInfo::builder()
                    .image(*swapchain_image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(info.image_format)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1)
                            .build(),
                    );
                unsafe { device.create_image_view(&image_view_info, None) }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let framebuffers = image_views
            .iter()
            .map(|image_view| {
                let attachments = [*image_view, depth_image_view];
                let framebuffer_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(info.image_extent.width)
                    .height(info.image_extent.height)
                    .layers(1);

                unsafe { device.create_framebuffer(&framebuffer_info, None) }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            framebuffers,
            image_views,
            _images: images,
            swapchain,
            swapchain_loader,
        })
    }

    // note: we don't destroy the images here because they're still being used
    // when we resize
    unsafe fn cleanup(&self, device: &Device) {
        for &framebuffer in &self.framebuffers {
            device.destroy_framebuffer(framebuffer, None);
        }

        for &image_view in &self.image_views {
            device.destroy_image_view(image_view, None);
        }

        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
    }
}

struct Syncronisation {
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
}

impl Syncronisation {
    fn new(device: &Device, num_images: usize) -> Self {
        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

        Self {
            image_available_semaphores: (0..FRAMES_IN_FLIGHT)
                .map(|_| unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap())
                .collect(),
            render_finished_semaphores: (0..FRAMES_IN_FLIGHT)
                .map(|_| unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap())
                .collect(),
            in_flight_fences: (0..FRAMES_IN_FLIGHT)
                .map(|_| unsafe { device.create_fence(&fence_info, None) }.unwrap())
                .collect(),
            images_in_flight: (0..num_images).map(|_| vk::Fence::null()).collect(),
        }
    }

    fn wait_fences(&mut self, device: &Device, image_index: usize, frame: usize) {
        let image_in_flight = self.images_in_flight[image_index];
        if image_in_flight != vk::Fence::null() {
            unsafe { device.wait_for_fences(&[image_in_flight], true, u64::MAX) }.unwrap();
        }
        self.images_in_flight[image_index] = self.in_flight_fences[frame];
    }

    unsafe fn cleanup(&self, device: &Device) {
        for &semaphore in self
            .image_available_semaphores
            .iter()
            .chain(self.render_finished_semaphores.iter())
        {
            device.destroy_semaphore(semaphore, None);
        }

        for &fence in &self.in_flight_fences {
            device.destroy_fence(fence, None);
        }
    }
}

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
struct CubePushConstants {
    perspective_view_matrix: Mat4,
    cube_transform: Mat4,
}

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
struct Vertex {
    position: Vec3,
    colour: Vec3,
}

fn vertex(x: f32, z: f32, y: f32) -> Vertex {
    let corner = Vec3::new(x, y, z);

    Vertex {
        position: corner * 2.0 - Vec3::broadcast(1.0),
        colour: corner,
    }
}

fn cube_verts() -> ([Vertex; 8], [u16; 36]) {
    (
        [
            vertex(0.0, 0.0, 0.0),
            vertex(1.0, 0.0, 0.0),
            vertex(0.0, 1.0, 0.0),
            vertex(1.0, 1.0, 0.0),
            vertex(0.0, 0.0, 1.0),
            vertex(1.0, 0.0, 1.0),
            vertex(0.0, 1.0, 1.0),
            vertex(1.0, 1.0, 1.0),
        ],
        [
            // bottom
            0, 1, 2, 1, 2, 3, // front
            1, 3, 5, 3, 5, 7, // back
            0, 2, 4, 2, 4, 6, // left
            0, 1, 4, 1, 4, 5, // right
            2, 3, 6, 3, 6, 7, // top
            4, 5, 6, 5, 6, 7,
        ],
    )
}

struct Pipelines {
    triangle_pipeline: vk::Pipeline,
    cube_pipeline: vk::Pipeline,
    push_constants_pipeline_layout: vk::PipelineLayout,
}

impl Pipelines {
    fn new(device: &Device, render_pass: vk::RenderPass) -> anyhow::Result<Self> {
        let pipeline_cache = unsafe {
            device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::builder().build(), None)?
        };

        let push_constants_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder().push_constant_ranges(&[
                    *vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::VERTEX)
                        .size(std::mem::size_of::<CubePushConstants>() as u32),
                ]),
                None,
            )
        }
        .unwrap();

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        let write_all = [*vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false)];

        let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&write_all);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0);

        let triangle_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);

        let empty_vertex_input = vk::PipelineVertexInputStateCreateInfo::builder();

        let vertex_binding_desc = &[*vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)];

        let vertex_attr_desc = &[
            *vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            *vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::size_of::<Vec3>() as u32),
        ];

        let cube_vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(vertex_binding_desc)
            .vertex_attribute_descriptions(vertex_attr_desc);

        let vs_triangle = util::read_spv(&mut std::io::Cursor::new(include_bytes!(
            "../shaders/compiled/triangle.vert.spv"
        )))
        .unwrap();
        let vs_triangle = vk::ShaderModuleCreateInfo::builder().code(&vs_triangle);
        let vs_triangle = unsafe { device.create_shader_module(&vs_triangle, None) }.unwrap();

        let vs_cube = util::read_spv(&mut std::io::Cursor::new(include_bytes!(
            "../shaders/compiled/cube.vert.spv"
        )))
        .unwrap();
        let vs_cube = vk::ShaderModuleCreateInfo::builder().code(&vs_cube);
        let vs_cube = unsafe { device.create_shader_module(&vs_cube, None) }.unwrap();

        let fs_colour = util::read_spv(&mut std::io::Cursor::new(include_bytes!(
            "../shaders/compiled/colour.frag.spv"
        )))
        .unwrap();
        let fs_colour = vk::ShaderModuleCreateInfo::builder().code(&fs_colour);
        let fs_colour = unsafe { device.create_shader_module(&fs_colour, None) }.unwrap();

        let triangle_pipeline_stages = [
            *vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vs_triangle)
                .name(&c_str!("main")),
            *vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fs_colour)
                .name(&c_str!("main")),
        ];

        let triangle_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&triangle_pipeline_stages)
            .vertex_input_state(&empty_vertex_input)
            .input_assembly_state(&triangle_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .depth_stencil_state(&depth_stencil)
            .layout(push_constants_pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let cube_pipeline_stages = [
            *vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vs_cube)
                .name(&c_str!("main")),
            *vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fs_colour)
                .name(&c_str!("main")),
        ];

        let cube_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&cube_pipeline_stages)
            .vertex_input_state(&cube_vertex_input)
            .input_assembly_state(&triangle_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .depth_stencil_state(&depth_stencil)
            .layout(push_constants_pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipelines = unsafe {
            device.create_graphics_pipelines(
                pipeline_cache,
                &[*triangle_pipeline_info, *cube_pipeline_info],
                None,
            )
        }
        .unwrap();

        let pipelines = Self {
            triangle_pipeline: pipelines[0],
            cube_pipeline: pipelines[1],
            push_constants_pipeline_layout,
        };

        unsafe {
            device.destroy_pipeline_cache(pipeline_cache, None);
            device.destroy_shader_module(vs_triangle, None);
            device.destroy_shader_module(vs_cube, None);
            device.destroy_shader_module(fs_colour, None);
        }

        Ok(pipelines)
    }

    unsafe fn cleanup(&self, device: &Device) {
        device.destroy_pipeline_layout(self.push_constants_pipeline_layout, None);
        device.destroy_pipeline(self.triangle_pipeline, None);
        device.destroy_pipeline(self.cube_pipeline, None);
    }
}
