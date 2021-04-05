use ash::{version::DeviceV1_0, vk, Device};

pub struct Buffer {
    pub buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
}

impl Buffer {
    pub fn new(
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

    pub fn cleanup(&self, allocator: &vk_mem::Allocator) -> anyhow::Result<()> {
        allocator
            .destroy_buffer(self.buffer, &self.allocation)
            .map_err(|err| err.into())
    }
}

pub struct Image {
    image: vk::Image,
    allocation: vk_mem::Allocation,
    pub view: vk::ImageView,
}

impl Image {
    pub fn new_depth_buffer(
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

    pub fn cleanup(&self, device: &Device, allocator: &vk_mem::Allocator) -> anyhow::Result<()> {
        unsafe {
            device.destroy_image_view(self.view, None);
        }

        allocator
            .destroy_image(self.image, &self.allocation)
            .map_err(|err| err.into())
    }
}

pub fn load_rgba_image_from_bytes(
    bytes: &[u8],
    width: u32,
    height: u32,
    command_buffer: vk::CommandBuffer,
    device: &Device,
    allocator: &vk_mem::Allocator,
    queue_family: u32,
) -> anyhow::Result<(Image, Buffer)> {
    let staging_buffer = Buffer::new(
        bytes,
        vk::BufferUsageFlags::TRANSFER_SRC,
        &allocator,
        queue_family,
    )?;

    let extent = vk::Extent3D {
        width,
        height,
        depth: 1,
    };

    let (image, allocation, _allocation_info) = allocator.create_image(
        &vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_SRGB)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST),
        &vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        },
    )?;

    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);

    let view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_SRGB)
                .subresource_range(*subresource_range),
            None,
        )
    }?;

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            // See
            // https://www.khronos.org/registry/vulkan/specs/1.0/html/vkspec.html#synchronization-access-types-supported
            // https://vulkan-tutorial.com/Texture_mapping/Images
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[*vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .subresource_range(*subresource_range)],
        );

        device.cmd_copy_buffer_to_image(
            command_buffer,
            staging_buffer.buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[*vk::BufferImageCopy::builder()
                .buffer_row_length(width)
                .buffer_image_height(height)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(extent)],
        );

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[*vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(image)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .subresource_range(*subresource_range)],
        );
    }

    Ok((
        Image {
            image,
            view,
            allocation,
        },
        staging_buffer,
    ))
}
