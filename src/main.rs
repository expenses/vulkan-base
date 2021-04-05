use ash::extensions::ext::DebugUtils as DebugUtilsLoader;
use ash::extensions::khr::{Surface as SurfaceLoader, Swapchain as SwapchainLoader};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::{util, vk, Device};
use ash_helpers::{Buffer, Image};
use byte_strings::c_str;
use std::ffi::CStr;
use ultraviolet::{Mat4, Vec2, Vec3};
use winit::{
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, StartCause, VirtualKeyCode, WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

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
        .with_title("Vulkan base")
        .build(&event_loop)?;

    let entry = unsafe { ash::Entry::new() }?;

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Instance
    let app_info = vk::ApplicationInfo::builder()
        .application_name(&c_str!("Vulkan base"))
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

    let cube = Model {
        vertices: Buffer::new(
            bytemuck::cast_slice(&verts),
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &allocator,
            queue_family,
        )?,
        indices: Buffer::new(
            bytemuck::cast_slice(&indices),
            vk::BufferUsageFlags::INDEX_BUFFER,
            &allocator,
            queue_family,
        )?,
        num_indices: indices.len() as u32,
    };

    let (verts, indices) = plane_verts();

    let plane = Model {
        vertices: Buffer::new(
            bytemuck::cast_slice(&verts),
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &allocator,
            queue_family,
        )?,
        indices: Buffer::new(
            bytemuck::cast_slice(&indices),
            vk::BufferUsageFlags::INDEX_BUFFER,
            &allocator,
            queue_family,
        )?,
        num_indices: indices.len() as u32,
    };

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

    let resources = Resources::new(&device)?;
    let pipelines = Pipelines::new(&device, render_pass, &resources)?;

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

    let mut depth_buffer =
        Image::new_depth_buffer(extent.width, extent.height, &device, &allocator)?;

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
    let command_pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family),
            None,
        )
    }?;

    let cmd_buf_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = unsafe { device.allocate_command_buffers(&cmd_buf_allocate_info) }?[0];

    let descriptor_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&[*vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)])
                .max_sets(1),
            None,
        )
    }?;

    let image = unsafe {
        device.begin_command_buffer(
            command_buffer,
            &vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        let decompressed_image = image::load_from_memory_with_format(
            include_bytes!("../explosion.png"),
            image::ImageFormat::Png,
        )?;

        let decompressed_image = match decompressed_image {
            image::DynamicImage::ImageRgba8(image) => image,
            _ => panic!(),
        };

        let (image, staging_buffer) = ash_helpers::load_rgba_image_from_bytes(
            &*decompressed_image,
            decompressed_image.width(),
            decompressed_image.height(),
            command_buffer,
            &device,
            &allocator,
            queue_family,
        )?;

        device.end_command_buffer(command_buffer)?;

        let image_written_fence = device.create_fence(&vk::FenceCreateInfo::builder(), None)?;

        device.queue_submit(
            queue,
            &[*vk::SubmitInfo::builder().command_buffers(&[command_buffer])],
            image_written_fence,
        )?;

        device.wait_for_fences(&[image_written_fence], true, u64::MAX)?;

        device.destroy_fence(image_written_fence, None);

        staging_buffer.cleanup(&allocator)?;

        image
    };

    let sampler = unsafe {
        device.create_sampler(
            &*vk::SamplerCreateInfo::builder().mag_filter(vk::Filter::LINEAR),
            None,
        )
    }?;

    let descriptor_sets = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::builder()
                .set_layouts(&[resources.single_texture_dsl])
                .descriptor_pool(descriptor_pool),
        )
    }?;

    let single_texture_set = descriptor_sets[0];

    unsafe {
        device.update_descriptor_sets(
            &[*vk::WriteDescriptorSet::builder()
                .dst_set(single_texture_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&[*vk::DescriptorImageInfo::builder()
                    .sampler(sampler)
                    .image_view(image.view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)])],
            &[],
        );
    }

    let view_matrix = Mat4::look_at(Vec3::new(-2.0, 1.0, 0.0), Vec3::zero(), Vec3::unit_y());

    let perspective_matrix = ultraviolet::projection::perspective_infinite_z_vk(
        59.0_f32.to_radians(),
        extent.width as f32 / extent.height as f32,
        0.1,
    );

    let mut perspective_view_matrix = perspective_matrix * view_matrix;
    let mut cube_rotation = 0.0;

    let syncronisation = Syncronisation::new(&device);

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
                    Image::new_depth_buffer(extent.width, extent.height, &device, &allocator)
                        .unwrap();

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

            match unsafe {
                swapchain.swapchain_loader.acquire_next_image(
                    swapchain.swapchain,
                    u64::MAX,
                    syncronisation.present_complete_semaphore,
                    vk::Fence::null(),
                )
            } {
                Ok((image_index, _suboptimal)) => {
                    let framebuffer = swapchain.framebuffers[image_index as usize];

                    let window_area = vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
                    };

                    let viewport = *vk::Viewport::builder()
                        .width(extent.width as f32)
                        .height(extent.height as f32)
                        .max_depth(1.0);

                    unsafe {
                        device
                            .wait_for_fences(&[syncronisation.draw_commands_fence], true, u64::MAX)
                            .unwrap();

                        device
                            .reset_fences(&[syncronisation.draw_commands_fence])
                            .unwrap();

                        device
                            .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())
                            .unwrap();

                        device
                            .begin_command_buffer(
                                command_buffer,
                                &vk::CommandBufferBeginInfo::builder()
                                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                            )
                            .unwrap();

                        device.cmd_set_viewport(command_buffer, 0, &[viewport]);
                        device.cmd_set_scissor(command_buffer, 0, &[window_area]);

                        device.cmd_begin_render_pass(
                            command_buffer,
                            &vk::RenderPassBeginInfo::builder()
                                .render_pass(render_pass)
                                .framebuffer(framebuffer)
                                .render_area(window_area)
                                .clear_values(&[
                                    vk::ClearValue {
                                        color: vk::ClearColorValue {
                                            float32: [0.0, 0.25, 0.5, 1.0],
                                        },
                                    },
                                    vk::ClearValue {
                                        depth_stencil: vk::ClearDepthStencilValue {
                                            depth: 1.0,
                                            stencil: 0,
                                        },
                                    },
                                ]),
                            vk::SubpassContents::INLINE,
                        );

                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipelines.cube_pipeline,
                        );
                        device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &[cube.vertices.buffer],
                            &[0],
                        );
                        device.cmd_bind_index_buffer(
                            command_buffer,
                            cube.indices.buffer,
                            0,
                            vk::IndexType::UINT16,
                        );
                        device.cmd_push_constants(
                            command_buffer,
                            pipelines.push_constants_pipeline_layout,
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            bytemuck::bytes_of(&PushConstants {
                                perspective_view_matrix,
                                cube_transform: Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0))
                                    * Mat4::from_rotation_y(cube_rotation)
                                    * Mat4::from_rotation_z(cube_rotation * 0.5)
                                    * Mat4::from_scale(0.2),
                            }),
                        );
                        device.cmd_draw_indexed(command_buffer, cube.num_indices, 1, 0, 0, 0);

                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipelines.plane_pipeline,
                        );
                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipelines.textured_pipeline_layout,
                            0,
                            &[single_texture_set],
                            &[],
                        );
                        device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &[plane.vertices.buffer],
                            &[0],
                        );
                        device.cmd_bind_index_buffer(
                            command_buffer,
                            plane.indices.buffer,
                            0,
                            vk::IndexType::UINT16,
                        );
                        device.cmd_push_constants(
                            command_buffer,
                            pipelines.push_constants_pipeline_layout,
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            bytemuck::bytes_of(&PushConstants {
                                perspective_view_matrix,
                                cube_transform: Mat4::from_scale(5.0),
                            }),
                        );
                        device.cmd_draw_indexed(command_buffer, plane.num_indices, 1, 0, 0, 0);

                        device.cmd_end_render_pass(command_buffer);

                        device.end_command_buffer(command_buffer).unwrap();

                        device
                            .queue_submit(
                                queue,
                                &[*vk::SubmitInfo::builder()
                                    .wait_semaphores(&[syncronisation.present_complete_semaphore])
                                    .wait_dst_stage_mask(&[
                                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                                    ])
                                    .command_buffers(&[command_buffer])
                                    .signal_semaphores(&[
                                        syncronisation.rendering_complete_semaphore
                                    ])],
                                syncronisation.draw_commands_fence,
                            )
                            .unwrap();

                        swapchain
                            .swapchain_loader
                            .queue_present(
                                queue,
                                &vk::PresentInfoKHR::builder()
                                    .wait_semaphores(&[syncronisation.rendering_complete_semaphore])
                                    .swapchains(&[swapchain.swapchain])
                                    .image_indices(&[image_index]),
                            )
                            .expect("Presenting failed. This is very unlikely to happen.");
                    }
                }
                Err(error) => println!("Next frame error: {:?}", error),
            }
        }
        Event::LoopDestroyed => unsafe {
            device.device_wait_idle().unwrap();

            resources.cleanup(&device);
            device.destroy_sampler(sampler, None);
            device.destroy_descriptor_pool(descriptor_pool, None);

            syncronisation.cleanup(&device);

            device.destroy_command_pool(command_pool, None);

            swapchain.cleanup(&device);
            pipelines.cleanup(&device);

            device.destroy_render_pass(render_pass, None);

            surface_loader.destroy_surface(surface, None);

            debug_utils_loader.destroy_debug_utils_messenger(debug_messenger, None);

            cube.cleanup(&allocator).unwrap();
            plane.cleanup(&allocator).unwrap();

            depth_buffer.cleanup(&device, &allocator).unwrap();
            image.cleanup(&device, &allocator).unwrap();

            allocator.destroy();

            device.destroy_device(None);

            instance.destroy_instance(None);
            println!("Exited cleanly");
        },
        _ => (),
    })
}

struct Model {
    vertices: Buffer,
    indices: Buffer,
    num_indices: u32,
}

impl Model {
    unsafe fn cleanup(&self, allocator: &vk_mem::Allocator) -> anyhow::Result<()> {
        self.vertices.cleanup(allocator)?;
        self.indices.cleanup(allocator)
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
    present_complete_semaphore: vk::Semaphore,
    rendering_complete_semaphore: vk::Semaphore,
    draw_commands_fence: vk::Fence,
}

impl Syncronisation {
    fn new(device: &Device) -> Self {
        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

        Self {
            present_complete_semaphore: unsafe { device.create_semaphore(&semaphore_info, None) }
                .unwrap(),
            rendering_complete_semaphore: unsafe { device.create_semaphore(&semaphore_info, None) }
                .unwrap(),
            draw_commands_fence: unsafe { device.create_fence(&fence_info, None) }.unwrap(),
        }
    }

    unsafe fn cleanup(&self, device: &Device) {
        device.destroy_semaphore(self.present_complete_semaphore, None);
        device.destroy_semaphore(self.rendering_complete_semaphore, None);
        device.destroy_fence(self.draw_commands_fence, None)
    }
}

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
struct PushConstants {
    perspective_view_matrix: Mat4,
    cube_transform: Mat4,
}

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
struct Vertex {
    position: Vec3,
    colour: Vec3,
}

impl Vertex {
    fn attribute_desc() -> [vk::VertexInputAttributeDescription; 2] {
        [
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
        ]
    }
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
            0, 1, 2, 2, 1, 3, // bottom
            3, 1, 5, 3, 5, 7, // front
            0, 2, 4, 4, 2, 6, // back
            1, 0, 4, 1, 4, 5, // left
            2, 3, 6, 6, 3, 7, // right
            5, 4, 6, 5, 6, 7, // top
        ],
    )
}

fn plane_verts() -> ([TexturedVertex; 4], [u16; 6]) {
    (
        [
            TexturedVertex {
                position: Vec3::new(-1.0, 0.0, -1.0),
                uv: Vec2::new(0.0, 0.0),
            },
            TexturedVertex {
                position: Vec3::new(1.0, 0.0, -1.0),
                uv: Vec2::new(1.0, 0.0),
            },
            TexturedVertex {
                position: Vec3::new(-1.0, 0.0, 1.0),
                uv: Vec2::new(0.0, 1.0),
            },
            TexturedVertex {
                position: Vec3::new(1.0, 0.0, 1.0),
                uv: Vec2::new(1.0, 1.0),
            },
        ],
        [1, 0, 2, 1, 2, 3],
    )
}

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
struct TexturedVertex {
    position: Vec3,
    uv: Vec2,
}

impl TexturedVertex {
    fn attribute_desc() -> [vk::VertexInputAttributeDescription; 2] {
        [
            *vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            *vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::size_of::<Vec3>() as u32),
        ]
    }
}

struct Pipelines {
    cube_pipeline: vk::Pipeline,
    plane_pipeline: vk::Pipeline,
    push_constants_pipeline_layout: vk::PipelineLayout,
    textured_pipeline_layout: vk::PipelineLayout,
}

impl Pipelines {
    fn new(
        device: &Device,
        render_pass: vk::RenderPass,
        resources: &Resources,
    ) -> anyhow::Result<Self> {
        let pipeline_cache = unsafe {
            device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::builder().build(), None)?
        };

        let push_constants_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder().push_constant_ranges(&[
                    *vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::VERTEX)
                        .size(std::mem::size_of::<PushConstants>() as u32),
                ]),
                None,
            )
        }?;

        let textured_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[resources.single_texture_dsl])
                    .push_constant_ranges(&[*vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::VERTEX)
                        .size(std::mem::size_of::<PushConstants>() as u32)]),
                None,
            )
        }?;

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

        let cull_rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .line_width(1.0);

        let triangle_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);

        let vertex_binding_desc = &[*vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)];

        let vertex_attr_desc = &Vertex::attribute_desc();

        let cube_vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(vertex_binding_desc)
            .vertex_attribute_descriptions(vertex_attr_desc);

        let textured_vertex_binding_desc = &[*vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<TexturedVertex>() as u32)];

        let textured_vertex_attr_desc = &TexturedVertex::attribute_desc();

        let textured_vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(textured_vertex_binding_desc)
            .vertex_attribute_descriptions(textured_vertex_attr_desc);

        let vs_cube =
            load_shader_module(include_bytes!("../shaders/compiled/cube.vert.spv"), &device)?;

        let fs_colour = load_shader_module(
            include_bytes!("../shaders/compiled/colour.frag.spv"),
            &device,
        )?;

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
            .rasterization_state(&cull_rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .depth_stencil_state(&depth_stencil)
            .layout(push_constants_pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let vs_plane = load_shader_module(
            include_bytes!("../shaders/compiled/plane.vert.spv"),
            &device,
        )?;

        let fs_textured = load_shader_module(
            include_bytes!("../shaders/compiled/textured.frag.spv"),
            &device,
        )?;

        let plane_pipeline_stages = [
            *vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vs_plane)
                .name(&c_str!("main")),
            *vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fs_textured)
                .name(&c_str!("main")),
        ];

        let plane_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&plane_pipeline_stages)
            .vertex_input_state(&textured_vertex_input)
            .input_assembly_state(&triangle_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&cull_rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .depth_stencil_state(&depth_stencil)
            .layout(textured_pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipelines = unsafe {
            device.create_graphics_pipelines(
                pipeline_cache,
                &[*cube_pipeline_info, *plane_pipeline_info],
                None,
            )
        }
        .unwrap();

        let pipelines = Self {
            cube_pipeline: pipelines[0],
            plane_pipeline: pipelines[1],
            push_constants_pipeline_layout,
            textured_pipeline_layout,
        };

        unsafe {
            device.destroy_pipeline_cache(pipeline_cache, None);
            device.destroy_shader_module(vs_cube, None);
            device.destroy_shader_module(fs_colour, None);
            device.destroy_shader_module(vs_plane, None);
            device.destroy_shader_module(fs_textured, None);
        }

        Ok(pipelines)
    }

    unsafe fn cleanup(&self, device: &Device) {
        device.destroy_pipeline_layout(self.textured_pipeline_layout, None);
        device.destroy_pipeline_layout(self.push_constants_pipeline_layout, None);
        device.destroy_pipeline(self.cube_pipeline, None);
        device.destroy_pipeline(self.plane_pipeline, None);
    }
}

fn load_shader_module(bytes: &[u8], device: &Device) -> anyhow::Result<vk::ShaderModule> {
    let spv = util::read_spv(&mut std::io::Cursor::new(bytes))?;
    unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&spv), None) }
        .map_err(|err| err.into())
}

struct Resources {
    single_texture_dsl: vk::DescriptorSetLayout,
}

impl Resources {
    fn new(device: &Device) -> anyhow::Result<Self> {
        Ok(Self {
            single_texture_dsl: unsafe {
                device.create_descriptor_set_layout(
                    &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                    ]),
                    None,
                )
            }?,
        })
    }

    unsafe fn cleanup(&self, device: &Device) {
        device.destroy_descriptor_set_layout(self.single_texture_dsl, None);
    }
}
