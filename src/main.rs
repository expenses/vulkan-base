// This code roughly follows https://vulkan-tutorial.com/ - Drawing a triangle

use erupt::{
    cstr,
    utils::{self, surface},
    vk, DeviceLoader, EntryLoader, InstanceLoader,
};
use std::{
    ffi::{c_void, CStr, CString},
    os::raw::c_char,
};
use winit::{
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, StartCause, VirtualKeyCode, WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const TITLE: &str = "erupt_examples: triangle";
const FRAMES_IN_FLIGHT: usize = 2;
const LAYER_KHRONOS_VALIDATION: *const c_char = cstr!("VK_LAYER_KHRONOS_validation");

unsafe extern "system" fn debug_callback(
    _message_severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    eprintln!(
        "{}",
        CStr::from_ptr((*p_callback_data).p_message).to_string_lossy()
    );

    vk::FALSE
}

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(TITLE)
        .build(&event_loop)
        .unwrap();

    let entry = EntryLoader::new().unwrap();
    println!(
        "{} - Vulkan Instance {}.{}.{}",
        TITLE,
        vk::version_major(entry.instance_version()),
        vk::version_minor(entry.instance_version()),
        vk::version_patch(entry.instance_version())
    );

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Instance
    let application_name = CString::new("Hello Triangle").unwrap();
    let engine_name = CString::new("No Engine").unwrap();
    let app_info = vk::ApplicationInfoBuilder::new()
        .application_name(&application_name)
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    let mut instance_extensions = surface::enumerate_required_extensions(&window).unwrap();
    instance_extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION_NAME);

    let mut instance_layers = Vec::new();
    instance_layers.push(LAYER_KHRONOS_VALIDATION);

    let device_extensions = [vk::KHR_SWAPCHAIN_EXTENSION_NAME];

    let mut device_layers = Vec::new();
    device_layers.push(LAYER_KHRONOS_VALIDATION);

    let instance_info = vk::InstanceCreateInfoBuilder::new()
        .application_info(&app_info)
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&instance_layers);

    let mut instance = InstanceLoader::new(&entry, &instance_info, None).unwrap();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Validation_layers
    let messenger = {
        let messenger_info = vk::DebugUtilsMessengerCreateInfoEXTBuilder::new()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE_EXT
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT,
            )
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
            .pfn_user_callback(Some(debug_callback));

        unsafe { instance.create_debug_utils_messenger_ext(&messenger_info, None, None) }.unwrap()
    };

    // https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Window_surface
    let surface = unsafe { surface::create_surface(&mut instance, &window, None) }.unwrap();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Physical_devices_and_queue_families
    let (physical_device, queue_family, format, present_mode, device_properties) =
        unsafe { instance.enumerate_physical_devices(None) }
            .unwrap()
            .into_iter()
            .filter_map(|physical_device| unsafe {
                let queue_family = match instance
                    .get_physical_device_queue_family_properties(physical_device, None)
                    .into_iter()
                    .enumerate()
                    .position(|(i, queue_family_properties)| {
                        queue_family_properties
                            .queue_flags
                            .contains(vk::QueueFlags::GRAPHICS)
                            && instance
                                .get_physical_device_surface_support_khr(
                                    physical_device,
                                    i as u32,
                                    surface,
                                    None,
                                )
                                .unwrap()
                    }) {
                    Some(queue_family) => queue_family as u32,
                    None => return None,
                };

                let formats = instance
                    .get_physical_device_surface_formats_khr(physical_device, surface, None)
                    .unwrap();
                let format = match formats
                    .iter()
                    .find(|surface_format| {
                        surface_format.format == vk::Format::B8G8R8A8_SRGB
                            && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR_KHR
                    })
                    .or_else(|| formats.get(0))
                {
                    Some(surface_format) => surface_format.clone(),
                    None => return None,
                };

                let present_mode = instance
                    .get_physical_device_surface_present_modes_khr(physical_device, surface, None)
                    .unwrap()
                    .into_iter()
                    .find(|present_mode| present_mode == &vk::PresentModeKHR::MAILBOX_KHR)
                    .unwrap_or(vk::PresentModeKHR::FIFO_KHR);

                let supported_device_extensions = instance
                    .enumerate_device_extension_properties(physical_device, None, None)
                    .unwrap();
                if !device_extensions.iter().all(|device_extension| {
                    let device_extension = CStr::from_ptr(*device_extension);

                    supported_device_extensions.iter().any(|properties| {
                        CStr::from_ptr(properties.extension_name.as_ptr()) == device_extension
                    })
                }) {
                    return None;
                }

                let device_properties =
                    instance.get_physical_device_properties(physical_device, None);
                Some((
                    physical_device,
                    queue_family,
                    format,
                    present_mode,
                    device_properties,
                ))
            })
            .max_by_key(|(_, _, _, _, properties)| match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 2,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 0,
            })
            .expect("No suitable physical device found");

    println!("Using physical device: {:?}", unsafe {
        CStr::from_ptr(device_properties.device_name.as_ptr())
    });

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Logical_device_and_queues
    let queue_info = [vk::DeviceQueueCreateInfoBuilder::new()
        .queue_family_index(queue_family)
        .queue_priorities(&[1.0])];
    let features = vk::PhysicalDeviceFeaturesBuilder::new();

    let device_info = vk::DeviceCreateInfoBuilder::new()
        .queue_create_infos(&queue_info)
        .enabled_features(&features)
        .enabled_extension_names(&device_extensions)
        .enabled_layer_names(&device_layers);

    let device = DeviceLoader::new(&instance, physical_device, &device_info, None).unwrap();
    let queue = unsafe { device.get_device_queue(queue_family, 0, None) };

    // https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Render_passes
    let attachments = [vk::AttachmentDescriptionBuilder::new()
        .format(format.format)
        .samples(vk::SampleCountFlagBits::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];

    let color_attachment_refs = [vk::AttachmentReferenceBuilder::new()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let subpasses = [vk::SubpassDescriptionBuilder::new()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)];
    let dependencies = [vk::SubpassDependencyBuilder::new()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

    let render_pass_info = vk::RenderPassCreateInfoBuilder::new()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    let render_pass = unsafe { device.create_render_pass(&render_pass_info, None, None) }.unwrap();

    let pipelines = Pipelines::new(&device, render_pass).unwrap();

    let surface_caps = unsafe {
        instance.get_physical_device_surface_capabilities_khr(physical_device, surface, None)
    }
    .unwrap();
    let mut image_count = surface_caps.min_image_count + 1;
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }

    let window_size = window.inner_size();

    let mut extent = vk::Extent2D {
        width: window_size.width,
        height: window_size.height,
    };

    let mut swapchain_info = vk::SwapchainCreateInfoKHRBuilder::new()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let mut swapchain_bits = SwapchainBits::new(&device, &swapchain_info, render_pass).unwrap();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Command_buffers
    let command_pool_info = vk::CommandPoolCreateInfoBuilder::new()
        .queue_family_index(queue_family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let command_pool =
        unsafe { device.create_command_pool(&command_pool_info, None, None) }.unwrap();

    let cmd_buf_allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(swapchain_bits.framebuffers.len() as _);
    let cmd_bufs = unsafe { device.allocate_command_buffers(&cmd_buf_allocate_info) }.unwrap();

    let mut syncronisation_bits =
        SyncronisationBits::new(&device, swapchain_bits.framebuffers.len());

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

                swapchain_info = swapchain_info.image_extent(extent);

                unsafe {
                    swapchain_bits.cleanup(&device);
                }
                swapchain_bits = SwapchainBits::new(&device, &swapchain_info, render_pass).unwrap();
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
            unsafe {
                device
                    .wait_for_fences(
                        &[syncronisation_bits.in_flight_fences[frame]],
                        true,
                        u64::MAX,
                    )
                    .unwrap();
            }

            match unsafe {
                device
                    .acquire_next_image_khr(
                        swapchain_bits.swapchain,
                        u64::MAX,
                        Some(syncronisation_bits.image_available_semaphores[frame]),
                        None,
                        None,
                    )
                    .result()
            } {
                Ok(image_index) => {
                    syncronisation_bits.wait_fences(&device, image_index as usize, frame);

                    let command_buffer = cmd_bufs[image_index as usize];
                    let framebuffer = swapchain_bits.framebuffers[image_index as usize];

                    let cmd_buf_begin_info = vk::CommandBufferBeginInfoBuilder::new();
                    unsafe { device.begin_command_buffer(command_buffer, &cmd_buf_begin_info) }
                        .unwrap();

                    let clear_values = [vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    }];
                    let area = vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
                    };
                    let render_pass_begin_info = vk::RenderPassBeginInfoBuilder::new()
                        .render_pass(render_pass)
                        .framebuffer(framebuffer)
                        .render_area(area)
                        .clear_values(&clear_values);

                    let viewport = vk::ViewportBuilder::new()
                        .x(0.0)
                        .y(0.0)
                        .width(extent.width as f32)
                        .height(extent.height as f32)
                        .min_depth(0.0)
                        .max_depth(1.0);

                    unsafe {
                        device.cmd_set_viewport(command_buffer, 0, &[viewport]);
                        device.cmd_set_scissor(command_buffer, 0, &[area.into_builder()]);

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
                        device.cmd_end_render_pass(command_buffer);

                        device.end_command_buffer(command_buffer).unwrap();
                    }

                    let wait_semaphores = [syncronisation_bits.image_available_semaphores[frame]];
                    let command_buffers = [command_buffer];
                    let signal_semaphores = [syncronisation_bits.render_finished_semaphores[frame]];
                    let submit_info = vk::SubmitInfoBuilder::new()
                        .wait_semaphores(&wait_semaphores)
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                        .command_buffers(&command_buffers)
                        .signal_semaphores(&signal_semaphores);

                    unsafe {
                        let in_flight_fence = syncronisation_bits.in_flight_fences[frame];
                        device.reset_fences(&[in_flight_fence]).unwrap();
                        device
                            .queue_submit(queue, &[submit_info], Some(in_flight_fence))
                            .unwrap()
                    }

                    let swapchains = [swapchain_bits.swapchain];
                    let image_indices = [image_index];
                    let present_info = vk::PresentInfoKHRBuilder::new()
                        .wait_semaphores(&signal_semaphores)
                        .swapchains(&swapchains)
                        .image_indices(&image_indices);

                    // Sometimes, resizes can happen between acquiring a frame and presenting.
                    // In this case, there isn't anything we can really do so we just print an error.
                    if let Err(err) =
                        unsafe { device.queue_present_khr(queue, &present_info) }.result()
                    {
                        println!("Error while presenting: {}", err);
                    }

                    frame = (frame + 1) % FRAMES_IN_FLIGHT;
                }
                Err(error) => println!("Next frame error: {}", error),
            }
        }
        Event::LoopDestroyed => unsafe {
            device.device_wait_idle().unwrap();

            syncronisation_bits.cleanup(&device);

            device.destroy_command_pool(Some(command_pool), None);

            swapchain_bits.cleanup(&device);

            pipelines.cleanup(&device);

            device.destroy_render_pass(Some(render_pass), None);

            device.destroy_device(None);

            instance.destroy_surface_khr(Some(surface), None);

            if !messenger.is_null() {
                instance.destroy_debug_utils_messenger_ext(Some(messenger), None);
            }

            instance.destroy_instance(None);
            println!("Exited cleanly");
        },
        _ => (),
    })
}

struct SwapchainBits {
    swapchain: vk::SwapchainKHR,
    framebuffers: Vec<vk::Framebuffer>,
    image_views: Vec<vk::ImageView>,
    images: Vec<vk::Image>,
}

impl SwapchainBits {
    fn new(
        device: &DeviceLoader,
        swapchain_info: &vk::SwapchainCreateInfoKHRBuilder,
        render_pass: vk::RenderPass,
    ) -> anyhow::Result<Self> {
        let info = swapchain_info.build();
        let swapchain = unsafe { device.create_swapchain_khr(swapchain_info, None, None) }.unwrap();
        let images = unsafe { device.get_swapchain_images_khr(swapchain, None) }.unwrap();

        let image_views: Vec<_> = images
            .iter()
            .map(|swapchain_image| {
                let image_view_info = vk::ImageViewCreateInfoBuilder::new()
                    .image(*swapchain_image)
                    .view_type(vk::ImageViewType::_2D)
                    .format(info.image_format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(
                        vk::ImageSubresourceRangeBuilder::new()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    );
                unsafe { device.create_image_view(&image_view_info, None, None) }.result()
            })
            .collect::<Result<Vec<_>, _>>()?;

        let framebuffers = image_views
            .iter()
            .map(|image_view| {
                let attachments = [*image_view];
                let framebuffer_info = vk::FramebufferCreateInfoBuilder::new()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(info.image_extent.width)
                    .height(info.image_extent.height)
                    .layers(1);

                unsafe { device.create_framebuffer(&framebuffer_info, None, None) }.result()
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            framebuffers,
            image_views,
            images,
            swapchain,
        })
    }

    unsafe fn cleanup(&self, device: &DeviceLoader) {
        device.device_wait_idle().unwrap();

        for &framebuffer in &self.framebuffers {
            device.destroy_framebuffer(Some(framebuffer), None);
        }

        for &image_view in &self.image_views {
            device.destroy_image_view(Some(image_view), None);
        }

        device.destroy_swapchain_khr(Some(self.swapchain), None);
    }
}

struct SyncronisationBits {
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
}

impl SyncronisationBits {
    fn new(device: &DeviceLoader, num_images: usize) -> Self {
        let semaphore_info = vk::SemaphoreCreateInfoBuilder::new();
        let fence_info = vk::FenceCreateInfoBuilder::new().flags(vk::FenceCreateFlags::SIGNALED);

        Self {
            image_available_semaphores: (0..FRAMES_IN_FLIGHT)
                .map(|_| unsafe { device.create_semaphore(&semaphore_info, None, None) }.unwrap())
                .collect(),
            render_finished_semaphores: (0..FRAMES_IN_FLIGHT)
                .map(|_| unsafe { device.create_semaphore(&semaphore_info, None, None) }.unwrap())
                .collect(),
            in_flight_fences: (0..FRAMES_IN_FLIGHT)
                .map(|_| unsafe { device.create_fence(&fence_info, None, None) }.unwrap())
                .collect(),
            images_in_flight: (0..num_images).map(|_| vk::Fence::null()).collect(),
        }
    }

    fn wait_fences(&mut self, device: &DeviceLoader, image_index: usize, frame: usize) {
        let image_in_flight = self.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            unsafe { device.wait_for_fences(&[image_in_flight], true, u64::MAX) }.unwrap();
        }
        self.images_in_flight[image_index] = self.in_flight_fences[frame];
    }

    unsafe fn cleanup(&self, device: &DeviceLoader) {
        for &semaphore in self
            .image_available_semaphores
            .iter()
            .chain(self.render_finished_semaphores.iter())
        {
            device.destroy_semaphore(Some(semaphore), None);
        }

        for &fence in &self.in_flight_fences {
            device.destroy_fence(Some(fence), None);
        }
    }
}

struct Pipelines {
    triangle_pipeline: vk::Pipeline,
}

impl Pipelines {
    fn new(device: &DeviceLoader, render_pass: vk::RenderPass) -> anyhow::Result<Self> {
        let pipeline_layout_info = vk::PipelineLayoutCreateInfoBuilder::new();
        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None, None) }.unwrap();

        let viewport_state = vk::PipelineViewportStateCreateInfoBuilder::new()
            .viewport_count(1)
            .scissor_count(1);

        let dynamic_state = vk::PipelineDynamicStateCreateInfoBuilder::new()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        let write_all = [vk::PipelineColorBlendAttachmentStateBuilder::new()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false)];

        let color_blending = vk::PipelineColorBlendStateCreateInfoBuilder::new()
            .logic_op_enable(false)
            .attachments(&write_all);

        let multisampling = vk::PipelineMultisampleStateCreateInfoBuilder::new()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlagBits::_1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfoBuilder::new()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0);

        let triangle_assembly = vk::PipelineInputAssemblyStateCreateInfoBuilder::new()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let empty_vertex_input = vk::PipelineVertexInputStateCreateInfoBuilder::new();

        let entry_point = unsafe { CStr::from_ptr(cstr!("main")) };

        let vert_decoded =
            utils::decode_spv(include_bytes!("../shaders/compiled/shader.vert.spv")).unwrap();
        let module_info = vk::ShaderModuleCreateInfoBuilder::new().code(&vert_decoded);
        let shader_vert = unsafe { device.create_shader_module(&module_info, None, None) }.unwrap();

        let frag_decoded =
            utils::decode_spv(include_bytes!("../shaders/compiled/shader.frag.spv")).unwrap();
        let module_info = vk::ShaderModuleCreateInfoBuilder::new().code(&frag_decoded);
        let shader_frag = unsafe { device.create_shader_module(&module_info, None, None) }.unwrap();

        let triangle_pipeline_stages = [
            vk::PipelineShaderStageCreateInfoBuilder::new()
                .stage(vk::ShaderStageFlagBits::VERTEX)
                .module(shader_vert)
                .name(&entry_point),
            vk::PipelineShaderStageCreateInfoBuilder::new()
                .stage(vk::ShaderStageFlagBits::FRAGMENT)
                .module(shader_frag)
                .name(&entry_point),
        ];

        let pipeline_info = vk::GraphicsPipelineCreateInfoBuilder::new()
            .stages(&triangle_pipeline_stages)
            .vertex_input_state(&empty_vertex_input)
            .input_assembly_state(&triangle_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipelines =
            unsafe { device.create_graphics_pipelines(None, &[pipeline_info], None) }.unwrap();

        let pipelines = Self {
            triangle_pipeline: pipelines[0],
        };

        unsafe {
            device.destroy_pipeline_layout(Some(pipeline_layout), None);
            device.destroy_shader_module(Some(shader_vert), None);
            device.destroy_shader_module(Some(shader_frag), None);
        }

        Ok(pipelines)
    }

    unsafe fn cleanup(&self, device: &DeviceLoader) {
        device.destroy_pipeline(Some(self.triangle_pipeline), None);
    }
}
