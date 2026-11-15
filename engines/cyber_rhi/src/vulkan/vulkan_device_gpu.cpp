// The command half of the Vulkan RHI device (ADR 0029): bind groups, pipelines,
// command lists, submission, resource transitions and swapchains.
//
// Conventions that make D3D12-shaped renderer code run unchanged:
//   * bind group slot N is descriptor set N, and a layout entry's `binding` is the
//     Vulkan binding number, so a shader is compiled against `set = slot`;
//   * viewports are submitted with a negative height, so clip-space y points up as
//     it does in Direct3D and front-face winding keeps its Direct3D meaning;
//   * barriers are derived from the RHI `from` and `to` states (`vulkan_barriers.h`),
//     and swapchain images start in the `Present` state like Direct3D back buffers;
//   * a swapchain's acquire gates the graphics queue and its present waits for every
//     earlier graphics submit, so callers never see a binary semaphore.

#include "vulkan_device.h"

#if defined(CYBER_RHI_HAS_VULKAN)

#include "vulkan_barriers.h"
#include "vulkan_commands.h"
#include "vulkan_device_internal.h"

#include <cstring>
#include <new>

namespace cyber::rhi::detail
{

using cyber::core::ErrorCode;

// ---- Heap data behind the stable `internal` pointers ----------------------

struct CommandListData
{
    explicit CommandListData(memory::Allocator& allocator) noexcept : transientViews(allocator) {}

    VulkanRhiDevice* device = nullptr;
    vk::CommandPool pool = 0;
    vk::CommandBuffer commands = nullptr;
    QueueKind queue = QueueKind::Graphics;
    CommandListState state = CommandListState::Initial;
    CommandListStats stats{};
    RenderPassInfo pass{};
    bool passOpen = false;
    u32 labelDepth = 0;
    PipelineKind pipelineKind = PipelineKind::Graphics;
    PipelineLayoutHandle layout{};
    vk::PipelineLayout nativeLayout = 0;
    // Views made for `ClearColorTexture` and `ClearDepthStencilTexture`; they must
    // outlive execution, so they are released when the list is reset or destroyed.
    DynamicArray<vk::ImageView> transientViews;
    char name[kNameBytes] = {};
};

struct SwapChainData
{
    VulkanRhiDevice* device = nullptr;
    VulkanSwapchain swapchain;
    Format format = Format::BGRA8UnormSrgb;
    PresentMode presentMode = PresentMode::Fifo;
    TextureHandle textures[kMaxSwapChainImages] = {};
    TextureViewHandle views[kMaxSwapChainImages] = {};
    u32 imageCount = 0;
    bool acquired = false;
    u32 currentImage = 0;
    vk::Semaphore imageAvailable = 0;
    vk::Semaphore renderFinished = 0;
    SwapChainHandle handle{};
};

namespace
{

// ---- Small helpers ---------------------------------------------------------

[[nodiscard]] vk::Flags ToStageFlags(ShaderStageMask mask) noexcept
{
    vk::Flags flags = 0;
    if (HasStage(mask, ShaderStage::Vertex))
    {
        flags |= vk::ShaderStage::kVertex;
    }
    if (HasStage(mask, ShaderStage::Pixel))
    {
        flags |= vk::ShaderStage::kFragment;
    }
    if (HasStage(mask, ShaderStage::Compute))
    {
        flags |= vk::ShaderStage::kCompute;
    }
    return flags;
}

constexpr vk::DescriptorType kUniformBufferDynamic = static_cast<vk::DescriptorType>(8);
constexpr vk::DescriptorType kStorageBufferDynamic = static_cast<vk::DescriptorType>(9);

[[nodiscard]] vk::DescriptorType ToDescriptorType(BindingType type, bool dynamic) noexcept
{
    switch (type)
    {
    case BindingType::UniformBuffer:
        return dynamic ? kUniformBufferDynamic : vk::DescriptorType::UniformBuffer;
    case BindingType::StorageBuffer:
    case BindingType::ReadOnlyStorageBuffer:
        return dynamic ? kStorageBufferDynamic : vk::DescriptorType::StorageBuffer;
    case BindingType::SampledTexture:
        return vk::DescriptorType::SampledImage;
    case BindingType::StorageTexture:
        return vk::DescriptorType::StorageImage;
    case BindingType::Sampler:
    default:
        return vk::DescriptorType::Sampler;
    }
}

[[nodiscard]] bool IsBufferBinding(BindingType type) noexcept
{
    return type == BindingType::UniformBuffer || type == BindingType::StorageBuffer ||
           type == BindingType::ReadOnlyStorageBuffer;
}

// Aspect bits a texture's format implies.
[[nodiscard]] vk::Flags FormatAspects(Format format) noexcept
{
    const FormatInfo info = FormatInfoOf(format);
    vk::Flags aspects = 0;
    if (info.depth)
    {
        aspects |= vk::ImageAspect::kDepth;
    }
    if (info.stencil)
    {
        aspects |= vk::ImageAspect::kStencil;
    }
    return aspects != 0 ? aspects : vk::ImageAspect::kColor;
}

struct ResolvedRange
{
    u32 mip = 0;
    u32 mipCount = 1;
    u32 layer = 0;
    u32 layerCount = 1;
    vk::Flags aspect = vk::ImageAspect::kColor;
};

// Zero counts mean "to the end", the RHI's range convention.
[[nodiscard]] bool ResolveRange(const TextureSlot& texture, const SubresourceRange& range,
                                ResolvedRange* out) noexcept
{
    const u32 mips = texture.desc.mipLevels;
    const u32 layers = texture.desc.arrayLayers;
    if (range.baseMipLevel >= mips || range.baseArrayLayer >= layers)
    {
        return false;
    }
    out->mip = range.baseMipLevel;
    out->mipCount = range.mipLevelCount != 0 ? range.mipLevelCount : mips - range.baseMipLevel;
    out->layer = range.baseArrayLayer;
    out->layerCount = range.layerCount != 0 ? range.layerCount : layers - range.baseArrayLayer;
    if (out->mip + out->mipCount > mips || out->layer + out->layerCount > layers)
    {
        return false;
    }
    const vk::Flags formatAspects = FormatAspects(texture.desc.format);
    out->aspect = formatAspects;
    if (formatAspects != vk::ImageAspect::kColor)
    {
        if (range.aspect == TextureAspect::Depth)
        {
            out->aspect = vk::ImageAspect::kDepth;
        }
        else if (range.aspect == TextureAspect::Stencil)
        {
            out->aspect = vk::ImageAspect::kStencil;
        }
    }
    return true;
}

void SetTrackedState(TextureSlot& texture, const ResolvedRange& range, ResourceState state) noexcept
{
    const u32 total = texture.desc.mipLevels * texture.desc.arrayLayers;
    for (u32 layer = range.layer; layer < range.layer + range.layerCount; ++layer)
    {
        for (u32 mip = range.mip; mip < range.mip + range.mipCount; ++mip)
        {
            const u32 index = total <= kMaxTrackedSubresources ? layer * texture.desc.mipLevels + mip : 0;
            texture.states[index] = state;
        }
    }
}

[[nodiscard]] ResourceState TrackedState(const TextureSlot& texture, u32 mip, u32 layer) noexcept
{
    const u32 total = texture.desc.mipLevels * texture.desc.arrayLayers;
    const u32 index = total <= kMaxTrackedSubresources ? layer * texture.desc.mipLevels + mip : 0;
    return texture.states[index];
}

[[nodiscard]] vk::ImageSubresourceRange ToNativeRange(const ResolvedRange& range) noexcept
{
    return vk::ImageSubresourceRange{range.aspect, range.mip, range.mipCount, range.layer,
                                     range.layerCount};
}

[[nodiscard]] u32 MipExtent(u32 base, u32 mip) noexcept
{
    const u32 value = base >> mip;
    return value != 0 ? value : 1U;
}

// ---- Bind group layouts ----------------------------------------------------

Status<Error> VkCreateBindGroupLayout(void* internal, const BindGroupLayoutDesc& desc,
                                      BindGroupLayoutHandle* out) noexcept
{
    if (internal == nullptr || out == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: layout creation needs a device and an output handle");
    }
    if (desc.entries.Size() == 0 || desc.entries.Size() > kMaxLayoutEntries)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: a bind group layout has between one and 64 entries");
    }
    VulkanRhiDevice& device = Self(internal);
    VulkanBindingDesc bindings[kMaxLayoutEntries] = {};
    u32 descriptorCount = 0;
    u32 dynamicCount = 0;
    for (usize index = 0; index < desc.entries.Size(); ++index)
    {
        const BindGroupLayoutEntry& entry = desc.entries[index];
        if (entry.arrayCount == 0)
        {
            return Err(ErrorCode::InvalidArgument, "vulkan backend: a binding has zero elements");
        }
        bindings[index].binding = entry.binding;
        bindings[index].type = ToDescriptorType(entry.type, entry.dynamicOffset);
        bindings[index].count = entry.arrayCount;
        bindings[index].stages = ToStageFlags(entry.visibility);
        descriptorCount += entry.arrayCount;
        dynamicCount += entry.dynamicOffset ? 1U : 0U;
    }

    const std::lock_guard<std::mutex> lock(device.mutex);
    u32 slotIndex = 0;
    if (!device.layouts.Acquire(&slotIndex))
    {
        return Err(ErrorCode::Internal, "vulkan backend: layout table could not grow");
    }
    LayoutSlot& slot = device.layouts.At(slotIndex);
    const Status<Error> created = device.pipelines.CreateSetLayout(
        bindings, static_cast<u32>(desc.entries.Size()), &slot.layout);
    if (!created)
    {
        device.layouts.Release(slotIndex);
        return created;
    }
    for (usize index = 0; index < desc.entries.Size(); ++index)
    {
        slot.entries[index] = desc.entries[index];
    }
    slot.entryCount = static_cast<u32>(desc.entries.Size());
    slot.descriptorCount = descriptorCount;
    slot.dynamicOffsetCount = dynamicCount;
    CopyLabel(slot.name, desc.label);
    *out = device.layouts.HandleOf<BindGroupLayoutTag>(slotIndex);
    return Status<Error>::Ok();
}

Status<Error> VkDestroyBindGroupLayout(void* internal, BindGroupLayoutHandle handle) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    LayoutSlot* slot = device.layouts.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: layout handle is stale or invalid");
    }
    device.pipelines.DestroySetLayout(slot->layout);
    device.layouts.Release(handle.index);
    return Status<Error>::Ok();
}

// ---- Bind groups -----------------------------------------------------------

constexpr u32 kPoolMaxSets = 1024;
constexpr u32 kPoolDescriptorsPerType = 4096;

[[nodiscard]] Status<Error> AddDescriptorPool(VulkanRhiDevice& device) noexcept
{
    const vk::DescriptorPoolSize sizes[7] = {
        {vk::DescriptorType::UniformBuffer, kPoolDescriptorsPerType},
        {kUniformBufferDynamic, kPoolDescriptorsPerType},
        {vk::DescriptorType::StorageBuffer, kPoolDescriptorsPerType},
        {kStorageBufferDynamic, kPoolDescriptorsPerType},
        {vk::DescriptorType::SampledImage, kPoolDescriptorsPerType},
        {vk::DescriptorType::StorageImage, kPoolDescriptorsPerType},
        {vk::DescriptorType::Sampler, kPoolDescriptorsPerType},
    };
    vk::DescriptorPool pool = 0;
    const Status<Error> created =
        device.pipelines.CreateDescriptorPool(kPoolMaxSets, sizes, 7, false, &pool);
    if (!created)
    {
        return created;
    }
    if (!device.descriptorPools.TryPushBack(pool))
    {
        device.pipelines.DestroyDescriptorPool(pool);
        return Err(ErrorCode::Internal, "vulkan backend: descriptor pool list could not grow");
    }
    return Status<Error>::Ok();
}

// Writes every resource of a group into its descriptor set. The resources are
// flattened in binding order, as the RHI defines them.
[[nodiscard]] Status<Error> WriteGroup(VulkanRhiDevice& device, const LayoutSlot& layout,
                                       vk::DescriptorSet set,
                                       memory::Span<const BindingResource> resources) noexcept
{
    if (resources.Size() != layout.descriptorCount)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: the resource count does not match the layout's descriptors");
    }
    DynamicArray<vk::DescriptorBufferInfo> bufferInfos(*device.allocator);
    DynamicArray<vk::DescriptorImageInfo> imageInfos(*device.allocator);
    DynamicArray<vk::WriteDescriptorSet> writes(*device.allocator);
    if (!bufferInfos.TryResize(layout.descriptorCount) ||
        !imageInfos.TryResize(layout.descriptorCount) || !writes.TryResize(layout.entryCount))
    {
        return Err(ErrorCode::Internal, "vulkan backend: descriptor write scratch failed to allocate");
    }

    u32 cursor = 0;
    for (u32 entryIndex = 0; entryIndex < layout.entryCount; ++entryIndex)
    {
        const BindGroupLayoutEntry& entry = layout.entries[entryIndex];
        vk::WriteDescriptorSet& write = writes[entryIndex];
        write = vk::WriteDescriptorSet{};
        write.dstSet = set;
        write.dstBinding = entry.binding;
        write.descriptorCount = entry.arrayCount;
        write.descriptorType = ToDescriptorType(entry.type, entry.dynamicOffset);
        for (u32 element = 0; element < entry.arrayCount; ++element)
        {
            const BindingResource& resource = resources[cursor + element];
            if (resource.type != entry.type)
            {
                return Err(ErrorCode::InvalidArgument,
                           "vulkan backend: a resource's type does not match its layout entry");
            }
            if (IsBufferBinding(entry.type))
            {
                const BufferSlot* buffer = device.buffers.Resolve(resource.buffer);
                if (buffer == nullptr)
                {
                    return Err(ErrorCode::InvalidArgument,
                               "vulkan backend: a bound buffer is stale or invalid");
                }
                if (resource.offset > buffer->desc.sizeBytes)
                {
                    return Err(ErrorCode::OutOfRange,
                               "vulkan backend: a buffer binding starts past the end of the buffer");
                }
                vk::DescriptorBufferInfo& info = bufferInfos[cursor + element];
                info.buffer = buffer->buffer.buffer;
                info.offset = resource.offset;
                info.range = resource.size != 0 ? resource.size : vk::kWholeSize;
            }
            else if (entry.type == BindingType::Sampler)
            {
                const SamplerSlot* sampler = device.samplers.Resolve(resource.sampler);
                if (sampler == nullptr)
                {
                    return Err(ErrorCode::InvalidArgument,
                               "vulkan backend: a bound sampler is stale or invalid");
                }
                imageInfos[cursor + element] =
                    vk::DescriptorImageInfo{sampler->sampler, 0, vk::ImageLayout::Undefined};
            }
            else
            {
                const ViewSlot* view = device.views.Resolve(resource.textureView);
                if (view == nullptr)
                {
                    return Err(ErrorCode::InvalidArgument,
                               "vulkan backend: a bound texture view is stale or invalid");
                }
                imageInfos[cursor + element] = vk::DescriptorImageInfo{
                    0, view->view,
                    entry.type == BindingType::StorageTexture ? vk::ImageLayout::General
                                                              : vk::ImageLayout::ShaderReadOnlyOptimal};
            }
        }
        if (IsBufferBinding(entry.type))
        {
            write.pBufferInfo = bufferInfos.Data() + cursor;
        }
        else
        {
            write.pImageInfo = imageInfos.Data() + cursor;
        }
        cursor += entry.arrayCount;
    }
    device.context.DeviceFunctionsExt().updateDescriptorSets(
        device.context.Device(), layout.entryCount, writes.Data(), 0, nullptr);
    return Status<Error>::Ok();
}

Status<Error> VkCreateBindGroup(void* internal, const BindGroupDesc& desc,
                                BindGroupHandle* out) noexcept
{
    if (internal == nullptr || out == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: bind group creation needs a device and an output handle");
    }
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    const LayoutSlot* layout = device.layouts.Resolve(desc.layout);
    if (layout == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: the bind group's layout is stale");
    }
    u32 slotIndex = 0;
    if (!device.groups.Acquire(&slotIndex))
    {
        return Err(ErrorCode::Internal, "vulkan backend: bind group table could not grow");
    }
    GroupSlot& slot = device.groups.At(slotIndex);

    // Allocate from the newest pool, opening another when it is exhausted.
    const vk::DeviceApiExt& ext = device.context.DeviceFunctionsExt();
    vk::DescriptorSetAllocateInfo info{};
    info.descriptorSetCount = 1;
    info.pSetLayouts = &layout->layout;
    vk::Result result = vk::Result::ErrorOutOfHostMemory;
    for (u32 attempt = 0; attempt < 2; ++attempt)
    {
        if (device.descriptorPools.Empty())
        {
            const Status<Error> added = AddDescriptorPool(device);
            if (!added)
            {
                device.groups.Release(slotIndex);
                return added;
            }
        }
        slot.pool = static_cast<u32>(device.descriptorPools.Size() - 1);
        info.descriptorPool = device.descriptorPools.Back();
        result = ext.allocateDescriptorSets(device.context.Device(), &info, &slot.set);
        if (result == vk::Result::Success)
        {
            break;
        }
        // Out of pool memory or fragmented: a fresh pool is the answer once.
        const Status<Error> added = AddDescriptorPool(device);
        if (!added)
        {
            device.groups.Release(slotIndex);
            return added;
        }
    }
    if (result != vk::Result::Success)
    {
        device.groups.Release(slotIndex);
        return Err(ErrorCode::Internal, "vulkan backend: vkAllocateDescriptorSets failed");
    }
    const Status<Error> written = WriteGroup(device, *layout, slot.set, desc.resources);
    if (!written)
    {
        (void)ext.freeDescriptorSets(device.context.Device(),
                                     device.descriptorPools[slot.pool], 1, &slot.set);
        device.groups.Release(slotIndex);
        return written;
    }
    slot.layout = desc.layout;
    slot.descriptorCount = layout->descriptorCount;
    CopyLabel(slot.name, desc.label);
    *out = device.groups.HandleOf<BindGroupTag>(slotIndex);
    return Status<Error>::Ok();
}

Status<Error> VkUpdateBindGroup(void* internal, BindGroupHandle handle,
                                memory::Span<const BindingResource> resources) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    GroupSlot* group = device.groups.Resolve(handle);
    if (group == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: bind group handle is stale");
    }
    const LayoutSlot* layout = device.layouts.Resolve(group->layout);
    if (layout == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: the group's layout was destroyed");
    }
    return WriteGroup(device, *layout, group->set, resources);
}

Status<Error> VkDestroyBindGroup(void* internal, BindGroupHandle handle) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    GroupSlot* group = device.groups.Resolve(handle);
    if (group == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: bind group handle is stale");
    }
    (void)device.context.DeviceFunctionsExt().freeDescriptorSets(
        device.context.Device(), device.descriptorPools[group->pool], 1, &group->set);
    device.groups.Release(handle.index);
    return Status<Error>::Ok();
}

BindGroupInfo VkBindGroupInfo(const void* internal, BindGroupHandle handle) noexcept
{
    const VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    BindGroupInfo info{};
    if (const GroupSlot* group = device.groups.Resolve(handle))
    {
        info.valid = true;
        info.layout = group->layout;
        info.resourceCount = group->descriptorCount;
    }
    return info;
}

// ---- Pipeline layouts ------------------------------------------------------

Status<Error> VkCreatePipelineLayout(void* internal, const PipelineLayoutDesc& desc,
                                     PipelineLayoutHandle* out) noexcept
{
    if (internal == nullptr || out == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: pipeline layout creation needs a device and an output handle");
    }
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);

    vk::DescriptorSetLayout sets[kBindGroupSlotCount] = {};
    for (u32 slot = 0; slot < kBindGroupSlotCount; ++slot)
    {
        if (desc.groups[slot].IsValid())
        {
            const LayoutSlot* layout = device.layouts.Resolve(desc.groups[slot]);
            if (layout == nullptr)
            {
                return Err(ErrorCode::InvalidArgument,
                           "vulkan backend: a pipeline layout group is stale or invalid");
            }
            sets[slot] = layout->layout;
        }
        else
        {
            sets[slot] = device.emptySetLayout;
        }
    }
    vk::PushConstantRange ranges[8] = {};
    if (desc.pushConstants.Size() > 8)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: too many push constant ranges");
    }
    u32 pushBytes = 0;
    for (usize index = 0; index < desc.pushConstants.Size(); ++index)
    {
        const PushConstantRange& range = desc.pushConstants[index];
        ranges[index] = vk::PushConstantRange{ToStageFlags(range.stages), range.offsetBytes,
                                              range.sizeBytes};
        pushBytes = range.offsetBytes + range.sizeBytes > pushBytes ? range.offsetBytes + range.sizeBytes
                                                                   : pushBytes;
    }

    u32 slotIndex = 0;
    if (!device.pipelineLayouts.Acquire(&slotIndex))
    {
        return Err(ErrorCode::Internal, "vulkan backend: pipeline layout table could not grow");
    }
    PipelineLayoutSlot& slot = device.pipelineLayouts.At(slotIndex);
    const Status<Error> created = device.pipelines.CreatePipelineLayout(
        sets, kBindGroupSlotCount, ranges, static_cast<u32>(desc.pushConstants.Size()), &slot.layout);
    if (!created)
    {
        device.pipelineLayouts.Release(slotIndex);
        return created;
    }
    for (u32 index = 0; index < kBindGroupSlotCount; ++index)
    {
        slot.groups[index] = desc.groups[index];
    }
    slot.pushConstantBytes = pushBytes;
    CopyLabel(slot.name, desc.label);
    *out = device.pipelineLayouts.HandleOf<PipelineLayoutTag>(slotIndex);
    return Status<Error>::Ok();
}

Status<Error> VkDestroyPipelineLayout(void* internal, PipelineLayoutHandle handle) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    PipelineLayoutSlot* slot = device.pipelineLayouts.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: pipeline layout handle is stale or invalid");
    }
    device.pipelines.DestroyPipelineLayout(slot->layout);
    device.pipelineLayouts.Release(handle.index);
    return Status<Error>::Ok();
}

// ---- Pipelines -------------------------------------------------------------

[[nodiscard]] u32 ToVulkanBlendFactor(BlendFactor factor) noexcept
{
    // Identical to Vulkan's numbering up to ConstantColor and OneMinusConstantColor;
    // the saturate factor sits after the two constant-alpha factors.
    return factor == BlendFactor::SourceAlphaSaturate ? 14U : static_cast<u32>(factor);
}

Status<Error> VkCreateGraphicsPipeline(void* internal, const GraphicsPipelineDesc& desc,
                                       PipelineHandle* out) noexcept
{
    if (internal == nullptr || out == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: pipeline creation needs a device and an output handle");
    }
    if (desc.rasterizer.polygonMode != PolygonMode::Fill || !desc.rasterizer.depthClipEnabled)
    {
        return Err(ErrorCode::Unsupported,
                   "vulkan backend: wireframe and disabled depth clip need device features that are not enabled");
    }
    if (desc.colorAttachments.Size() > kMaxColorAttachments ||
        desc.vertexLayout.bindings.Size() > kMaxVertexBindings)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: too many color attachments or vertex bindings");
    }
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    const PipelineLayoutSlot* layout = device.pipelineLayouts.Resolve(desc.layout);
    const ShaderSlot* vertex = device.shaders.Resolve(desc.vertexShader);
    if (layout == nullptr || vertex == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: pipeline needs a valid layout and vertex shader");
    }
    const ShaderSlot* pixel = nullptr;
    if (desc.pixelShader.IsValid())
    {
        pixel = device.shaders.Resolve(desc.pixelShader);
        if (pixel == nullptr)
        {
            return Err(ErrorCode::InvalidArgument, "vulkan backend: the pixel shader is stale");
        }
    }

    vk::PipelineShaderStageCreateInfo stages[2] = {};
    u32 stageCount = 0;
    stages[stageCount].stage = vk::ShaderStage::kVertex;
    stages[stageCount].module = vertex->module;
    stages[stageCount].pName = vertex->entry;
    ++stageCount;
    if (pixel != nullptr)
    {
        stages[stageCount].stage = vk::ShaderStage::kFragment;
        stages[stageCount].module = pixel->module;
        stages[stageCount].pName = pixel->entry;
        ++stageCount;
    }

    vk::VertexInputBindingDescription bindings[kMaxVertexBindings] = {};
    for (usize index = 0; index < desc.vertexLayout.bindings.Size(); ++index)
    {
        bindings[index].binding = static_cast<u32>(index);
        bindings[index].stride = desc.vertexLayout.bindings[index].strideBytes;
        bindings[index].inputRate =
            desc.vertexLayout.bindings[index].rate == VertexInputRate::PerInstance ? 1U : 0U;
    }
    DynamicArray<vk::VertexInputAttributeDescription> attributes(*device.allocator);
    if (!attributes.TryResize(desc.vertexLayout.attributes.Size()))
    {
        return Err(ErrorCode::Internal, "vulkan backend: vertex attribute scratch failed");
    }
    for (usize index = 0; index < desc.vertexLayout.attributes.Size(); ++index)
    {
        const VertexAttribute& attribute = desc.vertexLayout.attributes[index];
        const vk::Format format = ToVulkanFormat(attribute.format);
        if (format == vk::Format::Undefined)
        {
            return Err(ErrorCode::InvalidArgument, "vulkan backend: a vertex format is not valid");
        }
        attributes[index] = vk::VertexInputAttributeDescription{attribute.location, attribute.binding,
                                                                format, attribute.offsetBytes};
    }
    vk::PipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.vertexBindingDescriptionCount = static_cast<u32>(desc.vertexLayout.bindings.Size());
    vertexInput.pVertexBindingDescriptions = bindings;
    vertexInput.vertexAttributeDescriptionCount = static_cast<u32>(attributes.Size());
    vertexInput.pVertexAttributeDescriptions = attributes.Data();

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.topology = static_cast<u32>(desc.topology);
    vk::PipelineViewportStateCreateInfo viewport{};

    vk::PipelineRasterizationStateCreateInfo raster{};
    raster.cullMode = static_cast<vk::Flags>(desc.rasterizer.cullMode);  // none, front, back
    raster.frontFace =
        desc.rasterizer.frontFace == FrontFace::Clockwise ? 1U : 0U;
    const bool bias = desc.rasterizer.depthBiasConstant != 0.0F ||
                      desc.rasterizer.depthBiasSlope != 0.0F;
    raster.depthBiasEnable = bias ? vk::kTrue : vk::kFalse;
    raster.depthBiasConstantFactor = desc.rasterizer.depthBiasConstant;
    raster.depthBiasSlopeFactor = desc.rasterizer.depthBiasSlope;
    raster.depthBiasClamp = desc.rasterizer.depthBiasClamp;

    vk::PipelineMultisampleStateCreateInfo multisample{};
    multisample.rasterizationSamples = desc.multisample.sampleCount;
    multisample.pSampleMask = &desc.multisample.sampleMask;
    multisample.alphaToCoverageEnable = desc.multisample.alphaToCoverage ? vk::kTrue : vk::kFalse;

    const bool hasDepth = desc.depthStencilFormat != Format::Count;
    const auto stencilFace = [](const StencilFaceState& face) noexcept
    {
        vk::StencilOpState state{};
        state.failOp = static_cast<u32>(face.failOp);
        state.passOp = static_cast<u32>(face.passOp);
        state.depthFailOp = static_cast<u32>(face.depthFailOp);
        state.compareOp = static_cast<vk::CompareOp>(static_cast<u32>(face.compare));
        state.compareMask = face.readMask;
        state.writeMask = face.writeMask;
        return state;
    };
    vk::PipelineDepthStencilStateCreateInfo depth{};
    depth.depthTestEnable = hasDepth && desc.depthStencil.depthTestEnabled ? vk::kTrue : vk::kFalse;
    depth.depthWriteEnable = hasDepth && desc.depthStencil.depthWriteEnabled ? vk::kTrue : vk::kFalse;
    depth.depthCompareOp =
        static_cast<vk::CompareOp>(static_cast<u32>(desc.depthStencil.depthCompare));
    depth.stencilTestEnable = hasDepth && desc.depthStencil.stencilEnabled ? vk::kTrue : vk::kFalse;
    depth.front = stencilFace(desc.depthStencil.front);
    depth.back = stencilFace(desc.depthStencil.back);

    vk::PipelineColorBlendAttachmentState attachments[kMaxColorAttachments] = {};
    vk::Format colorFormats[kMaxColorAttachments] = {};
    for (usize index = 0; index < desc.colorAttachments.Size(); ++index)
    {
        const ColorAttachmentState& state = desc.colorAttachments[index];
        colorFormats[index] = ToVulkanFormat(state.format);
        if (colorFormats[index] == vk::Format::Undefined)
        {
            return Err(ErrorCode::InvalidArgument,
                       "vulkan backend: a color attachment format is not valid");
        }
        vk::PipelineColorBlendAttachmentState& target = attachments[index];
        target.blendEnable = state.blend.enabled ? vk::kTrue : vk::kFalse;
        target.srcColorBlendFactor = ToVulkanBlendFactor(state.blend.color.source);
        target.dstColorBlendFactor = ToVulkanBlendFactor(state.blend.color.destination);
        target.colorBlendOp = static_cast<u32>(state.blend.color.operation);
        target.srcAlphaBlendFactor = ToVulkanBlendFactor(state.blend.alpha.source);
        target.dstAlphaBlendFactor = ToVulkanBlendFactor(state.blend.alpha.destination);
        target.alphaBlendOp = static_cast<u32>(state.blend.alpha.operation);
        vk::Flags mask = 0;
        mask |= (state.blend.writeMask & kColorWriteRed) != 0 ? 0x1U : 0U;
        mask |= (state.blend.writeMask & kColorWriteGreen) != 0 ? 0x2U : 0U;
        mask |= (state.blend.writeMask & kColorWriteBlue) != 0 ? 0x4U : 0U;
        mask |= (state.blend.writeMask & kColorWriteAlpha) != 0 ? 0x8U : 0U;
        target.colorWriteMask = mask;
    }
    vk::PipelineColorBlendStateCreateInfo blend{};
    blend.attachmentCount = static_cast<u32>(desc.colorAttachments.Size());
    blend.pAttachments = attachments;

    const u32 dynamicStates[2] = {vk::kDynamicStateViewport, vk::kDynamicStateScissor};
    vk::PipelineDynamicStateCreateInfo dynamic{};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;

    vk::PipelineRenderingCreateInfo rendering{};
    rendering.colorAttachmentCount = static_cast<u32>(desc.colorAttachments.Size());
    rendering.pColorAttachmentFormats = colorFormats;
    if (hasDepth)
    {
        const vk::Format depthFormat = ToVulkanFormat(desc.depthStencilFormat);
        rendering.depthAttachmentFormat = depthFormat;
        if (FormatInfoOf(desc.depthStencilFormat).stencil)
        {
            rendering.stencilAttachmentFormat = depthFormat;
        }
    }

    vk::GraphicsPipelineCreateInfo info{};
    info.pNext = &rendering;
    info.stageCount = stageCount;
    info.pStages = stages;
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &inputAssembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = layout->layout;

    u32 slotIndex = 0;
    if (!device.pipelineTable.Acquire(&slotIndex))
    {
        return Err(ErrorCode::Internal, "vulkan backend: pipeline table could not grow");
    }
    PipelineSlot& slot = device.pipelineTable.At(slotIndex);
    if (device.context.DeviceFunctionsExt().createGraphicsPipelines(
            device.context.Device(), 0, 1, &info, nullptr, &slot.pipeline) != vk::Result::Success)
    {
        device.pipelineTable.Release(slotIndex);
        return Err(ErrorCode::Internal, "vulkan backend: vkCreateGraphicsPipelines failed");
    }
    slot.kind = PipelineKind::Graphics;
    slot.hash = HashGraphicsPipelineDesc(desc);
    slot.layout = desc.layout;
    CopyLabel(slot.name, desc.label);
    if (desc.label != nullptr)
    {
        (void)device.context.SetObjectName(vk::ObjectType::Pipeline, slot.pipeline, desc.label);
    }
    *out = device.pipelineTable.HandleOf<PipelineTag>(slotIndex);
    return Status<Error>::Ok();
}

Status<Error> VkCreateComputePipeline(void* internal, const ComputePipelineDesc& desc,
                                      PipelineHandle* out) noexcept
{
    if (internal == nullptr || out == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: pipeline creation needs a device and an output handle");
    }
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    const PipelineLayoutSlot* layout = device.pipelineLayouts.Resolve(desc.layout);
    const ShaderSlot* shader = device.shaders.Resolve(desc.computeShader);
    if (layout == nullptr || shader == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: pipeline needs a valid layout and compute shader");
    }
    vk::ComputePipelineCreateInfo info{};
    info.stage.stage = vk::ShaderStage::kCompute;
    info.stage.module = shader->module;
    info.stage.pName = shader->entry;
    info.layout = layout->layout;

    u32 slotIndex = 0;
    if (!device.pipelineTable.Acquire(&slotIndex))
    {
        return Err(ErrorCode::Internal, "vulkan backend: pipeline table could not grow");
    }
    PipelineSlot& slot = device.pipelineTable.At(slotIndex);
    if (device.context.DeviceFunctionsExt().createComputePipelines(
            device.context.Device(), 0, 1, &info, nullptr, &slot.pipeline) != vk::Result::Success)
    {
        device.pipelineTable.Release(slotIndex);
        return Err(ErrorCode::Internal, "vulkan backend: vkCreateComputePipelines failed");
    }
    slot.kind = PipelineKind::Compute;
    slot.hash = HashComputePipelineDesc(desc);
    slot.layout = desc.layout;
    CopyLabel(slot.name, desc.label);
    if (desc.label != nullptr)
    {
        (void)device.context.SetObjectName(vk::ObjectType::Pipeline, slot.pipeline, desc.label);
    }
    *out = device.pipelineTable.HandleOf<PipelineTag>(slotIndex);
    return Status<Error>::Ok();
}

Status<Error> VkDestroyPipeline(void* internal, PipelineHandle handle) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    PipelineSlot* slot = device.pipelineTable.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: pipeline handle is stale or invalid");
    }
    device.pipelines.DestroyPipeline(slot->pipeline);
    device.pipelineTable.Release(handle.index);
    return Status<Error>::Ok();
}

PipelineInfo VkPipelineInfo(const void* internal, PipelineHandle handle) noexcept
{
    const VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    PipelineInfo info{};
    if (const PipelineSlot* slot = device.pipelineTable.Resolve(handle))
    {
        info.valid = true;
        info.kind = slot->kind;
        info.hash = slot->hash;
    }
    return info;
}

// ---- Immediate work: transitions -------------------------------------------

// Records `record` into a one-shot command buffer on the graphics queue, submits it,
// and waits. Used for the device-level `Transition` and for swapchain setup, where
// there is no caller-owned command list.
template <typename Record>
[[nodiscard]] Status<Error> RunImmediate(VulkanRhiDevice& device, Record&& record) noexcept
{
    const vk::DeviceApiExt& ext = device.context.DeviceFunctionsExt();
    vk::CommandBufferAllocateInfo allocate{};
    allocate.commandPool = device.transitionPool;
    vk::CommandBuffer commands = nullptr;
    if (ext.allocateCommandBuffers(device.context.Device(), &allocate, &commands) !=
        vk::Result::Success)
    {
        return Err(ErrorCode::Internal, "vulkan backend: vkAllocateCommandBuffers failed");
    }
    vk::CommandBufferBeginInfo begin{};
    begin.flags = vk::kCommandBufferOneTimeSubmit;
    if (ext.beginCommandBuffer(commands, &begin) != vk::Result::Success)
    {
        ext.freeCommandBuffers(device.context.Device(), device.transitionPool, 1, &commands);
        return Err(ErrorCode::Internal, "vulkan backend: vkBeginCommandBuffer failed");
    }
    record(commands);
    Status<Error> result = Status<Error>::Ok();
    if (ext.endCommandBuffer(commands) != vk::Result::Success)
    {
        result = Err(ErrorCode::Internal, "vulkan backend: vkEndCommandBuffer failed");
    }
    else
    {
        vk::CommandBufferSubmitInfo commandInfo{};
        commandInfo.commandBuffer = commands;
        const u64 value = ++device.internalValue;
        vk::SemaphoreSubmitInfo signal{};
        signal.semaphore = device.internalTimeline;
        signal.value = value;
        signal.stageMask = vk::Stage2::kAllCommands;
        vk::SubmitInfo2 submit{};
        submit.commandBufferInfoCount = 1;
        submit.pCommandBufferInfos = &commandInfo;
        submit.signalSemaphoreInfoCount = 1;
        submit.pSignalSemaphoreInfos = &signal;
        if (ext.queueSubmit2(device.context.GraphicsQueue(), 1, &submit, 0) != vk::Result::Success)
        {
            result = Err(ErrorCode::Internal, "vulkan backend: vkQueueSubmit2 failed");
        }
        else
        {
            vk::SemaphoreWaitInfo wait{};
            wait.semaphoreCount = 1;
            wait.pSemaphores = &device.internalTimeline;
            wait.pValues = &value;
            if (ext.waitSemaphores(device.context.Device(), &wait, ~0ULL) != vk::Result::Success)
            {
                result = Err(ErrorCode::DeviceLost, "vulkan backend: wait after immediate submit failed");
            }
        }
    }
    ext.freeCommandBuffers(device.context.Device(), device.transitionPool, 1, &commands);
    return result;
}

// Builds the native barriers for the RHI barrier lists and updates tracked state.
// Returns an error for a stale handle or a bad range; nothing is recorded then.
struct BarrierBatch
{
    explicit BarrierBatch(memory::Allocator& allocator) noexcept : buffers(allocator), images(allocator)
    {
    }
    DynamicArray<vk::BufferMemoryBarrier2> buffers;
    DynamicArray<vk::ImageMemoryBarrier2> images;
};

[[nodiscard]] Status<Error> BuildBarriers(VulkanRhiDevice& device,
                                          memory::Span<const BufferBarrier> buffers,
                                          memory::Span<const TextureBarrier> textures,
                                          BarrierBatch& batch) noexcept
{
    if (!batch.buffers.TryReserve(buffers.Size()) || !batch.images.TryReserve(textures.Size()))
    {
        return Err(ErrorCode::Internal, "vulkan backend: barrier scratch failed to allocate");
    }
    for (usize index = 0; index < buffers.Size(); ++index)
    {
        BufferSlot* slot = device.buffers.Resolve(buffers[index].buffer);
        if (slot == nullptr)
        {
            return Err(ErrorCode::InvalidArgument, "vulkan backend: a barrier buffer is stale");
        }
        (void)batch.buffers.TryPushBack(MakeBufferBarrier(slot->buffer.buffer, 0, vk::kWholeSize,
                                                          buffers[index].from, buffers[index].to));
        slot->state = buffers[index].to;
    }
    for (usize index = 0; index < textures.Size(); ++index)
    {
        TextureSlot* slot = device.textures.Resolve(textures[index].texture);
        ResolvedRange range{};
        if (slot == nullptr || !ResolveRange(*slot, textures[index].range, &range))
        {
            return Err(ErrorCode::InvalidArgument,
                       "vulkan backend: a barrier texture is stale or its range is invalid");
        }
        (void)batch.images.TryPushBack(MakeImageBarrier(slot->image.image, ToNativeRange(range),
                                                        textures[index].from, textures[index].to));
        SetTrackedState(*slot, range, textures[index].to);
    }
    return Status<Error>::Ok();
}

Status<Error> VkTransition(void* internal, memory::Span<const BufferBarrier> buffers,
                           memory::Span<const TextureBarrier> textures) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    BarrierBatch batch(*device.allocator);
    const Status<Error> built = BuildBarriers(device, buffers, textures, batch);
    if (!built)
    {
        return built;
    }
    const vk::DeviceApiExt& ext = device.context.DeviceFunctionsExt();
    return RunImmediate(device,
                        [&](vk::CommandBuffer commands) noexcept
                        {
                            RecordBarriers(ext, commands, batch.buffers.Data(),
                                           static_cast<u32>(batch.buffers.Size()),
                                           batch.images.Data(),
                                           static_cast<u32>(batch.images.Size()));
                        });
}

// ---- Command lists ---------------------------------------------------------

[[nodiscard]] CommandListData& List(void* internal) noexcept
{
    return *static_cast<CommandListData*>(internal);
}
[[nodiscard]] const CommandListData& List(const void* internal) noexcept
{
    return *static_cast<const CommandListData*>(internal);
}

[[nodiscard]] Status<Error> NotRecording() noexcept
{
    return Err(ErrorCode::InvalidArgument, "vulkan backend: the command list is not recording");
}

void ReleaseTransientViews(CommandListData& list) noexcept
{
    VulkanRhiDevice& device = *list.device;
    for (usize index = 0; index < list.transientViews.Size(); ++index)
    {
        device.memory.DestroyImageView(list.transientViews[index]);
    }
    list.transientViews.Clear();
}

Status<Error> ListBegin(void* internal) noexcept
{
    CommandListData& list = List(internal);
    if (list.state == CommandListState::Recording)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: the command list is already recording");
    }
    vk::CommandBufferBeginInfo begin{};
    if (list.device->context.DeviceFunctionsExt().beginCommandBuffer(list.commands, &begin) !=
        vk::Result::Success)
    {
        return Err(ErrorCode::Internal, "vulkan backend: vkBeginCommandBuffer failed");
    }
    list.state = CommandListState::Recording;
    return Status<Error>::Ok();
}

Status<Error> ListEnd(void* internal) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    if (list.passOpen)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: a render pass is still open on the command list");
    }
    if (list.device->context.DeviceFunctionsExt().endCommandBuffer(list.commands) !=
        vk::Result::Success)
    {
        return Err(ErrorCode::Internal, "vulkan backend: vkEndCommandBuffer failed");
    }
    list.state = CommandListState::Ended;
    return Status<Error>::Ok();
}

void ListReset(void* internal) noexcept
{
    CommandListData& list = List(internal);
    VulkanRhiDevice& device = *list.device;
    (void)device.context.DeviceFunctionsExt().resetCommandPool(device.context.Device(), list.pool, 0);
    {
        const std::lock_guard<std::mutex> lock(device.mutex);
        ReleaseTransientViews(list);
    }
    list.state = CommandListState::Initial;
    list.stats = CommandListStats{};
    list.pass = RenderPassInfo{};
    list.passOpen = false;
    list.labelDepth = 0;
    list.layout = PipelineLayoutHandle{};
    list.nativeLayout = 0;
}

Status<Error> ListBeginRenderPass(void* internal, const RenderPassDesc& desc) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    if (list.passOpen)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: a render pass is already open");
    }
    VulkanRhiDevice& device = *list.device;
    const std::lock_guard<std::mutex> lock(device.mutex);

    vk::RenderingAttachmentInfo colors[kMaxColorAttachments] = {};
    RenderPassInfo pass{};
    pass.valid = true;
    u32 width = 0;
    u32 height = 0;
    const auto mapLoad = [](LoadOp op) noexcept
    {
        return op == LoadOp::Load ? vk::kAttachmentLoadOpLoad
                                  : (op == LoadOp::Clear ? vk::kAttachmentLoadOpClear
                                                         : vk::kAttachmentLoadOpDontCare);
    };
    const auto mapStore = [](StoreOp op) noexcept
    { return op == StoreOp::Store ? vk::kAttachmentStoreOpStore : vk::kAttachmentStoreOpDontCare; };
    const auto extentOf = [&](const ViewSlot& view, u32* w, u32* h, u32* samples) noexcept
    {
        const TextureSlot* texture = device.textures.Resolve(view.texture);
        if (texture == nullptr)
        {
            return false;
        }
        *w = MipExtent(texture->desc.width, view.desc.range.baseMipLevel);
        *h = MipExtent(texture->desc.height, view.desc.range.baseMipLevel);
        *samples = texture->desc.sampleCount;
        return true;
    };

    if (desc.colorAttachments.Size() > kMaxColorAttachments)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: too many color attachments");
    }
    for (usize index = 0; index < desc.colorAttachments.Size(); ++index)
    {
        const ColorAttachment& attachment = desc.colorAttachments[index];
        const ViewSlot* view = device.views.Resolve(attachment.view);
        u32 samples = 1;
        if (view == nullptr || !extentOf(*view, &width, &height, &samples))
        {
            return Err(ErrorCode::InvalidArgument,
                       "vulkan backend: a color attachment view is stale or invalid");
        }
        vk::RenderingAttachmentInfo& target = colors[index];
        target.imageView = view->view;
        target.imageLayout = vk::ImageLayout::ColorAttachmentOptimal;
        target.loadOp = mapLoad(attachment.loadOp);
        target.storeOp = mapStore(attachment.storeOp);
        target.clearValue.color[0] = attachment.clearValue.red;
        target.clearValue.color[1] = attachment.clearValue.green;
        target.clearValue.color[2] = attachment.clearValue.blue;
        target.clearValue.color[3] = attachment.clearValue.alpha;
        if (attachment.resolveView.IsValid())
        {
            const ViewSlot* resolve = device.views.Resolve(attachment.resolveView);
            if (resolve == nullptr)
            {
                return Err(ErrorCode::InvalidArgument,
                           "vulkan backend: a resolve view is stale or invalid");
            }
            target.resolveMode = 2;  // average
            target.resolveImageView = resolve->view;
            target.resolveImageLayout = vk::ImageLayout::ColorAttachmentOptimal;
        }
        pass.colorFormats[index] = view->desc.format;
        pass.sampleCount = samples;
    }
    pass.colorAttachmentCount = static_cast<u32>(desc.colorAttachments.Size());

    vk::RenderingAttachmentInfo depthAttachment{};
    vk::RenderingAttachmentInfo stencilAttachment{};
    bool hasStencil = false;
    if (desc.hasDepthStencil)
    {
        const ViewSlot* view = device.views.Resolve(desc.depthStencil.view);
        u32 samples = 1;
        if (view == nullptr || !extentOf(*view, &width, &height, &samples))
        {
            return Err(ErrorCode::InvalidArgument,
                       "vulkan backend: the depth attachment view is stale or invalid");
        }
        const vk::ImageLayout layout = desc.depthStencil.readOnly
                                           ? kVulkanDepthStencilReadOnly
                                           : vk::ImageLayout::DepthStencilAttachmentOptimal;
        depthAttachment.imageView = view->view;
        depthAttachment.imageLayout = layout;
        depthAttachment.loadOp = mapLoad(desc.depthStencil.depthLoadOp);
        depthAttachment.storeOp = mapStore(desc.depthStencil.depthStoreOp);
        depthAttachment.clearValue.depthStencil.depth = desc.depthStencil.clearValue.depth;
        depthAttachment.clearValue.depthStencil.stencil = desc.depthStencil.clearValue.stencil;
        hasStencil = FormatInfoOf(view->desc.format).stencil;
        if (hasStencil)
        {
            stencilAttachment = depthAttachment;
            stencilAttachment.loadOp = mapLoad(desc.depthStencil.stencilLoadOp);
            stencilAttachment.storeOp = mapStore(desc.depthStencil.stencilStoreOp);
        }
        pass.hasDepthStencil = true;
        pass.depthStencilFormat = view->desc.format;
        pass.sampleCount = samples;
    }
    if (width == 0 || height == 0)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: a render pass needs an attachment");
    }
    pass.width = width;
    pass.height = height;

    vk::RenderingInfo rendering{};
    rendering.renderArea = vk::Rect2D{{0, 0}, {width, height}};
    rendering.colorAttachmentCount = pass.colorAttachmentCount;
    rendering.pColorAttachments = colors;
    rendering.pDepthAttachment = desc.hasDepthStencil ? &depthAttachment : nullptr;
    rendering.pStencilAttachment = hasStencil ? &stencilAttachment : nullptr;
    device.context.DeviceFunctionsExt().cmdBeginRendering(list.commands, &rendering);
    list.pass = pass;
    list.passOpen = true;
    ++list.stats.renderPasses;
    return Status<Error>::Ok();
}

Status<Error> ListEndRenderPass(void* internal) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording || !list.passOpen)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: no render pass is open");
    }
    list.device->context.DeviceFunctionsExt().cmdEndRendering(list.commands);
    list.pass = RenderPassInfo{};
    list.passOpen = false;
    return Status<Error>::Ok();
}

Status<Error> ListSetPipeline(void* internal, PipelineHandle handle) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    VulkanRhiDevice& device = *list.device;
    const std::lock_guard<std::mutex> lock(device.mutex);
    const PipelineSlot* pipeline = device.pipelineTable.Resolve(handle);
    if (pipeline == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: pipeline handle is stale or invalid");
    }
    const PipelineLayoutSlot* layout = device.pipelineLayouts.Resolve(pipeline->layout);
    device.context.DeviceFunctionsExt().cmdBindPipeline(
        list.commands, pipeline->kind == PipelineKind::Compute ? 1U : 0U, pipeline->pipeline);
    list.pipelineKind = pipeline->kind;
    list.layout = pipeline->layout;
    list.nativeLayout = layout != nullptr ? layout->layout : 0;
    ++list.stats.pipelineChanges;
    return Status<Error>::Ok();
}

Status<Error> ListSetBindGroup(void* internal, BindGroupSlot slot, BindGroupHandle handle,
                               memory::Span<const u64> dynamicOffsets) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    if (list.nativeLayout == 0)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: set a pipeline before binding a group");
    }
    if (dynamicOffsets.Size() > kMaxLayoutEntries)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: too many dynamic offsets");
    }
    VulkanRhiDevice& device = *list.device;
    const std::lock_guard<std::mutex> lock(device.mutex);
    const GroupSlot* group = device.groups.Resolve(handle);
    if (group == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: bind group handle is stale");
    }
    u32 offsets[kMaxLayoutEntries] = {};
    for (usize index = 0; index < dynamicOffsets.Size(); ++index)
    {
        offsets[index] = static_cast<u32>(dynamicOffsets[index]);
    }
    device.context.DeviceFunctionsExt().cmdBindDescriptorSets(
        list.commands, list.pipelineKind == PipelineKind::Compute ? 1U : 0U, list.nativeLayout,
        static_cast<u32>(slot), 1, &group->set, static_cast<u32>(dynamicOffsets.Size()), offsets);
    ++list.stats.bindGroupChanges;
    return Status<Error>::Ok();
}

Status<Error> ListSetVertexBuffer(void* internal, u32 binding, BufferHandle handle,
                                  u64 offsetBytes) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    VulkanRhiDevice& device = *list.device;
    const std::lock_guard<std::mutex> lock(device.mutex);
    const BufferSlot* buffer = device.buffers.Resolve(handle);
    if (buffer == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: vertex buffer is stale or invalid");
    }
    const vk::Buffer native = buffer->buffer.buffer;
    device.context.DeviceFunctionsExt().cmdBindVertexBuffers(list.commands, binding, 1, &native,
                                                             &offsetBytes);
    return Status<Error>::Ok();
}

Status<Error> ListSetIndexBuffer(void* internal, BufferHandle handle, u64 offsetBytes,
                                 IndexFormat format) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    VulkanRhiDevice& device = *list.device;
    const std::lock_guard<std::mutex> lock(device.mutex);
    const BufferSlot* buffer = device.buffers.Resolve(handle);
    if (buffer == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: index buffer is stale or invalid");
    }
    device.context.DeviceFunctionsExt().cmdBindIndexBuffer(
        list.commands, buffer->buffer.buffer, offsetBytes, format == IndexFormat::Uint16 ? 0U : 1U);
    return Status<Error>::Ok();
}

Status<Error> ListSetViewport(void* internal, const Viewport& viewport) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    // A negative height starts the viewport at its bottom edge and flips it, so the
    // same clip-space coordinates land where Direct3D puts them.
    const vk::Viewport native{viewport.x,         viewport.y + viewport.height, viewport.width,
                              -viewport.height,   viewport.minDepth,            viewport.maxDepth};
    list.device->context.DeviceFunctionsExt().cmdSetViewport(list.commands, 0, 1, &native);
    return Status<Error>::Ok();
}

Status<Error> ListSetScissor(void* internal, const ScissorRect& scissor) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    const vk::Rect2D native{{static_cast<i32>(scissor.x), static_cast<i32>(scissor.y)},
                            {scissor.width, scissor.height}};
    list.device->context.DeviceFunctionsExt().cmdSetScissor(list.commands, 0, 1, &native);
    return Status<Error>::Ok();
}

Status<Error> ListDraw(void* internal, const DrawArguments& arguments) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    list.device->context.DeviceFunctionsExt().cmdDraw(list.commands, arguments.vertexCount,
                                                      arguments.instanceCount, arguments.firstVertex,
                                                      arguments.firstInstance);
    ++list.stats.draws;
    return Status<Error>::Ok();
}

Status<Error> ListDrawIndexed(void* internal, const DrawIndexedArguments& arguments) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    list.device->context.DeviceFunctionsExt().cmdDrawIndexed(
        list.commands, arguments.indexCount, arguments.instanceCount, arguments.firstIndex,
        arguments.vertexOffset, arguments.firstInstance);
    ++list.stats.draws;
    return Status<Error>::Ok();
}

[[nodiscard]] vk::Buffer NativeBuffer(VulkanRhiDevice& device, BufferHandle handle) noexcept
{
    const BufferSlot* slot = device.buffers.Resolve(handle);
    return slot != nullptr ? slot->buffer.buffer : 0;
}

Status<Error> ListDrawIndirect(void* internal, BufferHandle arguments, u64 offsetBytes,
                               u32 drawCount, u32 strideBytes) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    VulkanRhiDevice& device = *list.device;
    const std::lock_guard<std::mutex> lock(device.mutex);
    const vk::Buffer buffer = NativeBuffer(device, arguments);
    if (buffer == 0)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: indirect buffer is stale or invalid");
    }
    device.context.DeviceFunctionsExt().cmdDrawIndirect(list.commands, buffer, offsetBytes,
                                                        drawCount, strideBytes);
    ++list.stats.draws;
    return Status<Error>::Ok();
}

Status<Error> ListDrawIndexedIndirect(void* internal, BufferHandle arguments, u64 offsetBytes,
                                      u32 drawCount, u32 strideBytes) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    VulkanRhiDevice& device = *list.device;
    const std::lock_guard<std::mutex> lock(device.mutex);
    const vk::Buffer buffer = NativeBuffer(device, arguments);
    if (buffer == 0)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: indirect buffer is stale or invalid");
    }
    device.context.DeviceFunctionsExt().cmdDrawIndexedIndirect(list.commands, buffer, offsetBytes,
                                                               drawCount, strideBytes);
    ++list.stats.draws;
    return Status<Error>::Ok();
}

Status<Error> ListDrawIndirectCount(void* internal, const IndirectCountArguments& arguments) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    VulkanRhiDevice& device = *list.device;
    if (!device.drawIndirectCountSupported)
    {
        return Err(ErrorCode::Unsupported,
                   "vulkan backend: this device does not support indirect draws with a count");
    }
    const std::lock_guard<std::mutex> lock(device.mutex);
    const vk::Buffer args = NativeBuffer(device, arguments.argumentBuffer);
    const vk::Buffer count = NativeBuffer(device, arguments.countBuffer);
    if (args == 0 || count == 0)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: indirect buffer is stale or invalid");
    }
    device.context.DeviceFunctionsExt().cmdDrawIndirectCount(
        list.commands, args, arguments.argumentOffsetBytes, count, arguments.countOffsetBytes,
        arguments.maxDrawCount, arguments.strideBytes);
    ++list.stats.draws;
    return Status<Error>::Ok();
}

Status<Error> ListDrawIndexedIndirectCount(void* internal,
                                           const IndirectCountArguments& arguments) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    VulkanRhiDevice& device = *list.device;
    if (!device.drawIndirectCountSupported)
    {
        return Err(ErrorCode::Unsupported,
                   "vulkan backend: this device does not support indirect draws with a count");
    }
    const std::lock_guard<std::mutex> lock(device.mutex);
    const vk::Buffer args = NativeBuffer(device, arguments.argumentBuffer);
    const vk::Buffer count = NativeBuffer(device, arguments.countBuffer);
    if (args == 0 || count == 0)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: indirect buffer is stale or invalid");
    }
    device.context.DeviceFunctionsExt().cmdDrawIndexedIndirectCount(
        list.commands, args, arguments.argumentOffsetBytes, count, arguments.countOffsetBytes,
        arguments.maxDrawCount, arguments.strideBytes);
    ++list.stats.draws;
    return Status<Error>::Ok();
}

Status<Error> ListDispatch(void* internal, const DispatchArguments& arguments) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    list.device->context.DeviceFunctionsExt().cmdDispatch(
        list.commands, arguments.groupCountX, arguments.groupCountY, arguments.groupCountZ);
    ++list.stats.dispatches;
    return Status<Error>::Ok();
}

Status<Error> ListDispatchIndirect(void* internal, BufferHandle arguments, u64 offsetBytes) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    VulkanRhiDevice& device = *list.device;
    const std::lock_guard<std::mutex> lock(device.mutex);
    const vk::Buffer buffer = NativeBuffer(device, arguments);
    if (buffer == 0)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: indirect buffer is stale or invalid");
    }
    device.context.DeviceFunctionsExt().cmdDispatchIndirect(list.commands, buffer, offsetBytes);
    ++list.stats.dispatches;
    return Status<Error>::Ok();
}

Status<Error> ListCopyBuffer(void* internal, const BufferCopy& copy) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    VulkanRhiDevice& device = *list.device;
    const std::lock_guard<std::mutex> lock(device.mutex);
    const vk::Buffer source = NativeBuffer(device, copy.source);
    const vk::Buffer destination = NativeBuffer(device, copy.destination);
    if (source == 0 || destination == 0)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: a copy buffer is stale or invalid");
    }
    const vk::BufferCopy region{copy.sourceOffset, copy.destinationOffset, copy.sizeBytes};
    device.context.DeviceFunctionsExt().cmdCopyBuffer(list.commands, source, destination, 1, &region);
    ++list.stats.copies;
    return Status<Error>::Ok();
}

[[nodiscard]] Status<Error> CopyWithTexture(CommandListData& list, const BufferTextureCopy& copy,
                                            bool toTexture) noexcept
{
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    VulkanRhiDevice& device = *list.device;
    const std::lock_guard<std::mutex> lock(device.mutex);
    const BufferSlot* buffer = device.buffers.Resolve(copy.buffer);
    const TextureSlot* texture = device.textures.Resolve(copy.texture);
    if (buffer == nullptr || texture == nullptr || copy.mipLevel >= texture->desc.mipLevels ||
        copy.arrayLayer >= texture->desc.arrayLayers)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: a copy buffer or texture is stale, or its subresource is invalid");
    }
    const FormatInfo format = FormatInfoOf(texture->desc.format);
    vk::BufferImageCopy region{};
    region.bufferOffset = copy.bufferOffset;
    // Vulkan counts the row and image heights in texels, the RHI in bytes.
    if (copy.bufferRowPitch != 0 && format.bytesPerBlock != 0)
    {
        region.bufferRowLength = copy.bufferRowPitch / format.bytesPerBlock * format.blockWidth;
    }
    if (copy.bufferDepthPitch != 0 && copy.bufferRowPitch != 0)
    {
        region.bufferImageHeight = copy.bufferDepthPitch / copy.bufferRowPitch * format.blockHeight;
    }
    region.imageSubresource = vk::ImageSubresourceLayers{FormatAspects(texture->desc.format) &
                                                             (format.depth ? vk::ImageAspect::kDepth
                                                                           : vk::ImageAspect::kColor),
                                                         copy.mipLevel, copy.arrayLayer, 1};
    region.imageOffset = vk::Offset3D{static_cast<i32>(copy.x), static_cast<i32>(copy.y),
                                      static_cast<i32>(copy.z)};
    region.imageExtent = vk::Extent3D{copy.width, copy.height, copy.depth};
    const vk::DeviceApiExt& ext = device.context.DeviceFunctionsExt();
    if (toTexture)
    {
        ext.cmdCopyBufferToImage(list.commands, buffer->buffer.buffer, texture->image.image,
                                 vk::ImageLayout::TransferDstOptimal, 1, &region);
    }
    else
    {
        ext.cmdCopyImageToBuffer(list.commands, texture->image.image,
                                 vk::ImageLayout::TransferSrcOptimal, buffer->buffer.buffer, 1,
                                 &region);
    }
    ++list.stats.copies;
    return Status<Error>::Ok();
}

Status<Error> ListCopyBufferToTexture(void* internal, const BufferTextureCopy& copy) noexcept
{
    return CopyWithTexture(List(internal), copy, true);
}

Status<Error> ListCopyTextureToBuffer(void* internal, const BufferTextureCopy& copy) noexcept
{
    return CopyWithTexture(List(internal), copy, false);
}

Status<Error> ListCopyTexture(void* internal, const TextureCopy& copy) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    VulkanRhiDevice& device = *list.device;
    const std::lock_guard<std::mutex> lock(device.mutex);
    const TextureSlot* source = device.textures.Resolve(copy.source);
    const TextureSlot* destination = device.textures.Resolve(copy.destination);
    if (source == nullptr || destination == nullptr || copy.sourceMipLevel >= source->desc.mipLevels ||
        copy.destinationMipLevel >= destination->desc.mipLevels ||
        copy.sourceArrayLayer >= source->desc.arrayLayers ||
        copy.destinationArrayLayer >= destination->desc.arrayLayers)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: a copy texture is stale, or its subresource is invalid");
    }
    vk::ImageCopy region{};
    region.srcSubresource = vk::ImageSubresourceLayers{FormatAspects(source->desc.format),
                                                       copy.sourceMipLevel, copy.sourceArrayLayer, 1};
    region.dstSubresource = vk::ImageSubresourceLayers{
        FormatAspects(destination->desc.format), copy.destinationMipLevel,
        copy.destinationArrayLayer, 1};
    region.extent = vk::Extent3D{copy.width, copy.height, copy.depth};
    device.context.DeviceFunctionsExt().cmdCopyImage(
        list.commands, source->image.image, vk::ImageLayout::TransferSrcOptimal,
        destination->image.image, vk::ImageLayout::TransferDstOptimal, 1, &region);
    ++list.stats.copies;
    return Status<Error>::Ok();
}

// A clear is a render pass that loads with `clear` and stores, per subresource. That
// keeps the RHI's contract (a texture is cleared in its render-target or depth-write
// state) without a second image layout, and needs no transfer-destination usage.
[[nodiscard]] Status<Error> ClearViaRendering(CommandListData& list, TextureHandle handle,
                                              const SubresourceRange& requested, bool depth,
                                              const vk::ClearValue& clear) noexcept
{
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    if (list.passOpen)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: a clear cannot be recorded inside a render pass");
    }
    VulkanRhiDevice& device = *list.device;
    const std::lock_guard<std::mutex> lock(device.mutex);
    const TextureSlot* texture = device.textures.Resolve(handle);
    ResolvedRange range{};
    if (texture == nullptr || !ResolveRange(*texture, requested, &range))
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: the clear texture is stale or its range is invalid");
    }
    const vk::DeviceApiExt& ext = device.context.DeviceFunctionsExt();
    const bool stencil = depth && (range.aspect & vk::ImageAspect::kStencil) != 0;
    for (u32 layer = range.layer; layer < range.layer + range.layerCount; ++layer)
    {
        for (u32 mip = range.mip; mip < range.mip + range.mipCount; ++mip)
        {
            vk::ImageViewCreateInfo viewInfo{};
            viewInfo.image = texture->image.image;
            viewInfo.viewType = vk::ImageViewType::Type2D;
            viewInfo.format = ToVulkanFormat(texture->desc.format);
            viewInfo.subresourceRange = vk::ImageSubresourceRange{range.aspect, mip, 1, layer, 1};
            vk::ImageView view = 0;
            if (ext.createImageView(device.context.Device(), &viewInfo, nullptr, &view) !=
                vk::Result::Success)
            {
                return Err(ErrorCode::Internal, "vulkan backend: vkCreateImageView failed");
            }
            if (!list.transientViews.TryPushBack(view))
            {
                device.memory.DestroyImageView(view);
                return Err(ErrorCode::Internal, "vulkan backend: clear view list could not grow");
            }
            vk::RenderingAttachmentInfo attachment{};
            attachment.imageView = view;
            attachment.imageLayout = depth ? vk::ImageLayout::DepthStencilAttachmentOptimal
                                           : vk::ImageLayout::ColorAttachmentOptimal;
            attachment.loadOp = vk::kAttachmentLoadOpClear;
            attachment.storeOp = vk::kAttachmentStoreOpStore;
            attachment.clearValue = clear;
            vk::RenderingInfo rendering{};
            rendering.renderArea = vk::Rect2D{{0, 0},
                                              {MipExtent(texture->desc.width, mip),
                                               MipExtent(texture->desc.height, mip)}};
            if (depth)
            {
                rendering.pDepthAttachment = &attachment;
                rendering.pStencilAttachment = stencil ? &attachment : nullptr;
            }
            else
            {
                rendering.colorAttachmentCount = 1;
                rendering.pColorAttachments = &attachment;
            }
            ext.cmdBeginRendering(list.commands, &rendering);
            ext.cmdEndRendering(list.commands);
        }
    }
    ++list.stats.clears;
    return Status<Error>::Ok();
}

Status<Error> ListClearColorTexture(void* internal, TextureHandle texture,
                                    const ClearColorValue& value,
                                    const SubresourceRange& range) noexcept
{
    vk::ClearValue clear{};
    clear.color[0] = value.red;
    clear.color[1] = value.green;
    clear.color[2] = value.blue;
    clear.color[3] = value.alpha;
    return ClearViaRendering(List(internal), texture, range, false, clear);
}

Status<Error> ListClearDepthStencilTexture(void* internal, TextureHandle texture,
                                           const ClearDepthStencilValue& value,
                                           const SubresourceRange& range) noexcept
{
    vk::ClearValue clear{};
    clear.depthStencil.depth = value.depth;
    clear.depthStencil.stencil = value.stencil;
    return ClearViaRendering(List(internal), texture, range, true, clear);
}

Status<Error> ListBarrier(void* internal, memory::Span<const BufferBarrier> buffers,
                          memory::Span<const TextureBarrier> textures) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    if (list.passOpen)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: a barrier cannot be recorded inside a render pass");
    }
    VulkanRhiDevice& device = *list.device;
    const std::lock_guard<std::mutex> lock(device.mutex);
    BarrierBatch batch(*device.allocator);
    const Status<Error> built = BuildBarriers(device, buffers, textures, batch);
    if (!built)
    {
        return built;
    }
    RecordBarriers(device.context.DeviceFunctionsExt(), list.commands, batch.buffers.Data(),
                   static_cast<u32>(batch.buffers.Size()), batch.images.Data(),
                   static_cast<u32>(batch.images.Size()));
    list.stats.barriers += static_cast<u32>(buffers.Size() + textures.Size());
    return Status<Error>::Ok();
}

Status<Error> ListPushConstants(void* internal, ShaderStageMask stages, u32 offsetBytes,
                                memory::Span<const memory::Byte> data) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    if (list.nativeLayout == 0)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: set a pipeline before pushing constants");
    }
    list.device->context.DeviceFunctionsExt().cmdPushConstants(
        list.commands, list.nativeLayout, ToStageFlags(stages), offsetBytes,
        static_cast<u32>(data.Size()), data.Data());
    ++list.stats.pushConstants;
    return Status<Error>::Ok();
}

Status<Error> ListBeginDebugLabel(void* internal, const char* label) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    if (list.device->beginLabel != nullptr)
    {
        vk::DebugLabel native{};
        native.pLabelName = label;
        list.device->beginLabel(list.commands, &native);
    }
    ++list.labelDepth;
    ++list.stats.debugLabels;
    return Status<Error>::Ok();
}

Status<Error> ListEndDebugLabel(void* internal) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording || list.labelDepth == 0)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: no debug label is open");
    }
    if (list.device->endLabel != nullptr)
    {
        list.device->endLabel(list.commands);
    }
    --list.labelDepth;
    return Status<Error>::Ok();
}

Status<Error> ListSetDebugMarker(void* internal, const char* label) noexcept
{
    CommandListData& list = List(internal);
    if (list.state != CommandListState::Recording)
    {
        return NotRecording();
    }
    if (list.device->insertLabel != nullptr)
    {
        vk::DebugLabel native{};
        native.pLabelName = label;
        list.device->insertLabel(list.commands, &native);
    }
    ++list.stats.debugLabels;
    return Status<Error>::Ok();
}

Status<Error> ListWriteTimestamp(void*, TimestampQueryHeapHandle, u32) noexcept
{
    return Err(ErrorCode::Unsupported, "vulkan backend: timestamp queries are not implemented yet");
}

Status<Error> ListResolveTimestamps(void*, TimestampQueryHeapHandle, u32, u32) noexcept
{
    return Err(ErrorCode::Unsupported, "vulkan backend: timestamp queries are not implemented yet");
}

RenderPassInfo ListCurrentRenderPass(const void* internal) noexcept
{
    return List(internal).pass;
}

CommandListStats ListStats(const void* internal) noexcept
{
    return List(internal).stats;
}

[[nodiscard]] const CommandListInterface* ListInterface() noexcept
{
    static const CommandListInterface table = []() noexcept
    {
        CommandListInterface value{};
        value.begin = &ListBegin;
        value.end = &ListEnd;
        value.reset = &ListReset;
        value.beginRenderPass = &ListBeginRenderPass;
        value.endRenderPass = &ListEndRenderPass;
        value.setPipeline = &ListSetPipeline;
        value.setBindGroup = &ListSetBindGroup;
        value.setVertexBuffer = &ListSetVertexBuffer;
        value.setIndexBuffer = &ListSetIndexBuffer;
        value.setViewport = &ListSetViewport;
        value.setScissor = &ListSetScissor;
        value.draw = &ListDraw;
        value.drawIndexed = &ListDrawIndexed;
        value.drawIndirect = &ListDrawIndirect;
        value.drawIndexedIndirect = &ListDrawIndexedIndirect;
        value.dispatch = &ListDispatch;
        value.dispatchIndirect = &ListDispatchIndirect;
        value.copyBuffer = &ListCopyBuffer;
        value.copyBufferToTexture = &ListCopyBufferToTexture;
        value.copyTextureToBuffer = &ListCopyTextureToBuffer;
        value.copyTexture = &ListCopyTexture;
        value.clearColorTexture = &ListClearColorTexture;
        value.clearDepthStencilTexture = &ListClearDepthStencilTexture;
        value.barrier = &ListBarrier;
        value.pushConstants = &ListPushConstants;
        value.beginDebugLabel = &ListBeginDebugLabel;
        value.endDebugLabel = &ListEndDebugLabel;
        value.setDebugMarker = &ListSetDebugMarker;
        value.writeTimestamp = &ListWriteTimestamp;
        value.resolveTimestamps = &ListResolveTimestamps;
        value.currentRenderPass = &ListCurrentRenderPass;
        value.stats = &ListStats;
        value.drawIndirectCount = &ListDrawIndirectCount;
        value.drawIndexedIndirectCount = &ListDrawIndexedIndirectCount;
        return value;
    }();
    return &table;
}

Status<Error> VkCreateCommandList(void* internal, const CommandListDesc& desc,
                                  CommandListHandle* out, CommandListCreation* creation) noexcept
{
    if (internal == nullptr || out == nullptr || creation == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: command list creation needs a device and outputs");
    }
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);

    u32 family = device.context.Info().queues.graphics;
    if (desc.queue == QueueKind::Compute)
    {
        family = device.context.Info().queues.compute;
    }
    else if (desc.queue == QueueKind::Copy)
    {
        family = device.context.Info().queues.transfer;
    }

    void* storage = device.allocator->Allocate(sizeof(CommandListData), alignof(CommandListData));
    if (storage == nullptr)
    {
        return Err(ErrorCode::Internal, "vulkan backend: command list allocation failed");
    }
    auto* data = new (storage) CommandListData(*device.allocator);
    data->device = &device;
    data->queue = desc.queue;
    const vk::DeviceApiExt& ext = device.context.DeviceFunctionsExt();
    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.queueFamilyIndex = family;
    bool ok = ext.createCommandPool(device.context.Device(), &poolInfo, nullptr, &data->pool) ==
              vk::Result::Success;
    if (ok)
    {
        vk::CommandBufferAllocateInfo allocate{};
        allocate.commandPool = data->pool;
        ok = ext.allocateCommandBuffers(device.context.Device(), &allocate, &data->commands) ==
             vk::Result::Success;
    }
    u32 slotIndex = 0;
    if (ok && !device.commandLists.Acquire(&slotIndex))
    {
        ok = false;
    }
    if (!ok)
    {
        if (data->pool != 0)
        {
            ext.destroyCommandPool(device.context.Device(), data->pool, nullptr);
        }
        data->~CommandListData();
        device.allocator->Deallocate(data);
        return Err(ErrorCode::Internal, "vulkan backend: command list creation failed");
    }
    CopyLabel(data->name, desc.label);
    if (desc.label != nullptr)
    {
        (void)device.context.SetObjectName(vk::ObjectType::CommandBuffer,
                                           reinterpret_cast<u64>(data->commands), desc.label);
    }
    device.commandLists.At(slotIndex).data = data;
    *out = device.commandLists.HandleOf<CommandListTag>(slotIndex);
    creation->interface = ListInterface();
    creation->internal = data;
    return Status<Error>::Ok();
}

void DestroyListData(VulkanRhiDevice& device, CommandListData* data) noexcept
{
    ReleaseTransientViews(*data);
    device.context.DeviceFunctionsExt().destroyCommandPool(device.context.Device(), data->pool,
                                                           nullptr);
    data->~CommandListData();
    device.allocator->Deallocate(data);
}

Status<Error> VkDestroyCommandList(void* internal, CommandListHandle handle) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    CommandListSlot* slot = device.commandLists.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: command list handle is stale or invalid");
    }
    DestroyListData(device, slot->data);
    device.commandLists.Release(handle.index);
    return Status<Error>::Ok();
}

// ---- Submission ------------------------------------------------------------

Status<Error> VkSubmit(void* internal, const SubmitDesc& desc) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    const vk::DeviceApiExt& ext = device.context.DeviceFunctionsExt();

    DynamicArray<vk::CommandBufferSubmitInfo> commands(*device.allocator);
    DynamicArray<vk::SemaphoreSubmitInfo> waits(*device.allocator);
    DynamicArray<vk::SemaphoreSubmitInfo> signals(*device.allocator);
    if (!commands.TryReserve(desc.commandLists.Size()) ||
        !waits.TryReserve(desc.gpuWaits.Size() + 1) ||
        !signals.TryReserve(desc.signals.Size() + 1))
    {
        return Err(ErrorCode::Internal, "vulkan backend: submit scratch failed to allocate");
    }
    SubmissionInfo info{};
    info.valid = true;
    info.queue = desc.queue;
    for (usize index = 0; index < desc.commandLists.Size(); ++index)
    {
        const CommandListSlot* slot = device.commandLists.Resolve(desc.commandLists[index]);
        if (slot == nullptr || slot->data->state != CommandListState::Ended)
        {
            return Err(ErrorCode::InvalidArgument,
                       "vulkan backend: a submitted command list is stale or not ended");
        }
        vk::CommandBufferSubmitInfo commandInfo{};
        commandInfo.commandBuffer = slot->data->commands;
        (void)commands.TryPushBack(commandInfo);
        info.commandListCount += 1;
        info.barrierCount += slot->data->stats.barriers;
    }
    for (usize index = 0; index < desc.gpuWaits.Size(); ++index)
    {
        const FenceSlot* fence = device.fences.Resolve(desc.gpuWaits[index].fence);
        if (fence == nullptr)
        {
            return Err(ErrorCode::InvalidArgument, "vulkan backend: a wait fence is stale or invalid");
        }
        vk::SemaphoreSubmitInfo wait{};
        wait.semaphore = fence->semaphore;
        wait.value = desc.gpuWaits[index].value;
        wait.stageMask = vk::Stage2::kAllCommands;
        (void)waits.TryPushBack(wait);
    }
    // Work after a swapchain acquire waits for the image, once, through the gate.
    if (desc.queue == QueueKind::Graphics && device.gateValue != 0)
    {
        vk::SemaphoreSubmitInfo gate{};
        gate.semaphore = device.internalTimeline;
        gate.value = device.gateValue;
        gate.stageMask = vk::Stage2::kAllCommands;
        (void)waits.TryPushBack(gate);
    }
    for (usize index = 0; index < desc.signals.Size(); ++index)
    {
        const FenceSlot* fence = device.fences.Resolve(desc.signals[index].fence);
        if (fence == nullptr)
        {
            return Err(ErrorCode::InvalidArgument,
                       "vulkan backend: a signal fence is stale or invalid");
        }
        vk::SemaphoreSubmitInfo signal{};
        signal.semaphore = fence->semaphore;
        signal.value = desc.signals[index].value;
        signal.stageMask = vk::Stage2::kAllCommands;
        (void)signals.TryPushBack(signal);
        info.signalledValue = desc.signals[index].value;
    }
    // Present waits for every earlier graphics submit through this timeline.
    u64 timelineValue = 0;
    if (desc.queue == QueueKind::Graphics)
    {
        timelineValue = ++device.internalValue;
        vk::SemaphoreSubmitInfo order{};
        order.semaphore = device.internalTimeline;
        order.value = timelineValue;
        order.stageMask = vk::Stage2::kAllCommands;
        (void)signals.TryPushBack(order);
    }

    vk::SubmitInfo2 submit{};
    submit.waitSemaphoreInfoCount = static_cast<u32>(waits.Size());
    submit.pWaitSemaphoreInfos = waits.Data();
    submit.commandBufferInfoCount = static_cast<u32>(commands.Size());
    submit.pCommandBufferInfos = commands.Data();
    submit.signalSemaphoreInfoCount = static_cast<u32>(signals.Size());
    submit.pSignalSemaphoreInfos = signals.Data();
    vk::Queue queue = device.context.GraphicsQueue();
    if (desc.queue == QueueKind::Compute)
    {
        queue = device.context.ComputeQueue();
    }
    else if (desc.queue == QueueKind::Copy)
    {
        queue = device.context.TransferQueue();
    }
    const vk::Result result = ext.queueSubmit2(queue, 1, &submit, 0);
    if (result == vk::Result::ErrorDeviceLost)
    {
        return Err(ErrorCode::DeviceLost, "vulkan backend: device lost on submit");
    }
    if (result != vk::Result::Success)
    {
        return Err(ErrorCode::Internal, "vulkan backend: vkQueueSubmit2 failed");
    }
    device.lastSubmission[static_cast<u32>(desc.queue)] = info;
    return Status<Error>::Ok();
}

SubmissionInfo VkSubmissionInfo(const void* internal, QueueKind kind) noexcept
{
    const VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    if (static_cast<u32>(kind) >= kQueueKindCount)
    {
        return SubmissionInfo{};
    }
    return device.lastSubmission[static_cast<u32>(kind)];
}

// ---- Swapchains ------------------------------------------------------------

[[nodiscard]] Format ToRhiColorFormat(vk::Format format) noexcept
{
    return format == vk::Format::B8G8R8A8Unorm ? Format::BGRA8Unorm : Format::BGRA8UnormSrgb;
}

// Registers every swapchain image as an RHI texture and view, and moves the images to
// the `Present` state a Direct3D back buffer starts in.
[[nodiscard]] Status<Error> RegisterSwapchainImages(VulkanRhiDevice& device,
                                                    SwapChainData& data) noexcept
{
    const vk::DeviceApiExt& ext = device.context.DeviceFunctionsExt();
    data.imageCount = data.swapchain.ImageCount() < kMaxSwapChainImages
                          ? data.swapchain.ImageCount()
                          : kMaxSwapChainImages;
    data.format = ToRhiColorFormat(data.swapchain.Format());
    for (u32 index = 0; index < data.imageCount; ++index)
    {
        u32 textureIndex = 0;
        if (!device.textures.Acquire(&textureIndex))
        {
            return Err(ErrorCode::Internal, "vulkan backend: texture table could not grow");
        }
        TextureSlot& texture = device.textures.At(textureIndex);
        texture.external = true;
        texture.image.image = data.swapchain.Image(index);
        texture.desc.format = data.format;
        texture.desc.usage = TextureUsage::RenderTarget | TextureUsage::CopySource;
        texture.desc.width = data.swapchain.Width();
        texture.desc.height = data.swapchain.Height();
        texture.states[0] = ResourceState::Present;
        data.textures[index] = device.textures.HandleOf<TextureTag>(textureIndex);

        u32 viewIndex = 0;
        if (!device.views.Acquire(&viewIndex))
        {
            return Err(ErrorCode::Internal, "vulkan backend: view table could not grow");
        }
        ViewSlot& view = device.views.At(viewIndex);
        vk::ImageViewCreateInfo info{};
        info.image = texture.image.image;
        info.format = data.swapchain.Format();
        info.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspect::kColor, 0, 1, 0, 1};
        if (ext.createImageView(device.context.Device(), &info, nullptr, &view.view) !=
            vk::Result::Success)
        {
            device.views.Release(viewIndex);
            return Err(ErrorCode::Internal, "vulkan backend: vkCreateImageView failed");
        }
        view.texture = data.textures[index];
        view.desc.texture = data.textures[index];
        view.desc.format = data.format;
        view.desc.range.mipLevelCount = 1;
        view.desc.range.layerCount = 1;
        data.views[index] = device.views.HandleOf<TextureViewTag>(viewIndex);
    }

    // `Undefined` to `PresentSrc`, once, so the first frame's `Present` to
    // `RenderTarget` barrier names the layout the image really has.
    DynamicArray<vk::ImageMemoryBarrier2> barriers(*device.allocator);
    if (!barriers.TryReserve(data.imageCount))
    {
        return Err(ErrorCode::Internal, "vulkan backend: swapchain barrier scratch failed");
    }
    for (u32 index = 0; index < data.imageCount; ++index)
    {
        (void)barriers.TryPushBack(MakeImageBarrier(
            data.swapchain.Image(index), vk::ImageSubresourceRange{vk::ImageAspect::kColor, 0, 1, 0, 1},
            ResourceState::Undefined, ResourceState::Present));
    }
    return RunImmediate(device,
                        [&](vk::CommandBuffer commands) noexcept
                        {
                            RecordBarriers(ext, commands, nullptr, 0, barriers.Data(),
                                           static_cast<u32>(barriers.Size()));
                        });
}

void UnregisterSwapchainImages(VulkanRhiDevice& device, SwapChainData& data) noexcept
{
    for (u32 index = 0; index < data.imageCount; ++index)
    {
        if (ViewSlot* view = device.views.Resolve(data.views[index]))
        {
            device.memory.DestroyImageView(view->view);
            device.views.Release(data.views[index].index);
        }
        if (device.textures.Resolve(data.textures[index]) != nullptr)
        {
            device.textures.Release(data.textures[index].index);
        }
        data.views[index] = TextureViewHandle{};
        data.textures[index] = TextureHandle{};
    }
    data.imageCount = 0;
}

[[nodiscard]] SwapChainData& Chain(void* internal) noexcept
{
    return *static_cast<SwapChainData*>(internal);
}

// An empty submit that waits on `waitSemaphore` (binary) and signals a timeline value,
// used to gate the graphics queue on an acquired image.
[[nodiscard]] Status<Error> SubmitGate(VulkanRhiDevice& device, vk::Semaphore imageAvailable) noexcept
{
    vk::SemaphoreSubmitInfo wait{};
    wait.semaphore = imageAvailable;
    wait.stageMask = vk::Stage2::kAllCommands;
    const u64 value = ++device.internalValue;
    vk::SemaphoreSubmitInfo signal{};
    signal.semaphore = device.internalTimeline;
    signal.value = value;
    signal.stageMask = vk::Stage2::kAllCommands;
    vk::SubmitInfo2 submit{};
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &wait;
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos = &signal;
    if (device.context.DeviceFunctionsExt().queueSubmit2(device.context.GraphicsQueue(), 1, &submit,
                                                         0) != vk::Result::Success)
    {
        return Err(ErrorCode::Internal, "vulkan backend: swapchain gate submit failed");
    }
    device.gateValue = value;
    return Status<Error>::Ok();
}

Status<Error> ChainAcquire(void* internal, SwapChainFrame* out) noexcept
{
    SwapChainData& data = Chain(internal);
    VulkanRhiDevice& device = *data.device;
    if (out == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: acquire needs an output frame");
    }
    const std::lock_guard<std::mutex> lock(device.mutex);
    if (data.acquired)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: an image is already acquired; present it first");
    }
    u32 image = 0;
    vk::Semaphore available = 0;
    vk::Semaphore finished = 0;
    for (u32 attempt = 0; attempt < 2; ++attempt)
    {
        auto acquired = data.swapchain.Acquire(&image, &available, &finished);
        if (!acquired)
        {
            return Status<Error>::Err(acquired.Error());
        }
        if (acquired.Value() == VulkanPresentResult::Ok)
        {
            break;
        }
        // Out of date: rebuild at the surface's current size and try once more.
        UnregisterSwapchainImages(device, data);
        Status<Error> rebuilt = data.swapchain.Recreate(data.swapchain.Width(), data.swapchain.Height());
        if (!rebuilt)
        {
            return rebuilt;
        }
        rebuilt = RegisterSwapchainImages(device, data);
        if (!rebuilt)
        {
            return rebuilt;
        }
        if (attempt == 1)
        {
            return Err(ErrorCode::Busy, "vulkan backend: the swapchain stayed out of date");
        }
    }
    const Status<Error> gated = SubmitGate(device, available);
    if (!gated)
    {
        return gated;
    }
    data.acquired = true;
    data.currentImage = image;
    data.imageAvailable = available;
    data.renderFinished = finished;
    out->valid = true;
    out->imageIndex = image;
    out->texture = data.textures[image];
    out->view = data.views[image];
    return Status<Error>::Ok();
}

Status<Error> ChainPresent(void* internal) noexcept
{
    SwapChainData& data = Chain(internal);
    VulkanRhiDevice& device = *data.device;
    const std::lock_guard<std::mutex> lock(device.mutex);
    if (!data.acquired)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: no image is acquired to present");
    }
    // Present waits for every graphics submit so far.
    vk::SemaphoreSubmitInfo wait{};
    wait.semaphore = device.internalTimeline;
    wait.value = device.internalValue;
    wait.stageMask = vk::Stage2::kAllCommands;
    vk::SemaphoreSubmitInfo signal{};
    signal.semaphore = data.renderFinished;
    signal.stageMask = vk::Stage2::kAllCommands;
    vk::SubmitInfo2 submit{};
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &wait;
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos = &signal;
    if (device.context.DeviceFunctionsExt().queueSubmit2(device.context.GraphicsQueue(), 1, &submit,
                                                         0) != vk::Result::Success)
    {
        return Err(ErrorCode::Internal, "vulkan backend: present gate submit failed");
    }
    auto presented = data.swapchain.Present(data.currentImage, data.renderFinished);
    data.acquired = false;
    if (!presented)
    {
        return Status<Error>::Err(presented.Error());
    }
    if (presented.Value() == VulkanPresentResult::NeedsRecreate)
    {
        // The next acquire rebuilds; nothing to do now.
        return Status<Error>::Ok();
    }
    return Status<Error>::Ok();
}

Status<Error> ChainResize(void* internal, u32 width, u32 height) noexcept
{
    SwapChainData& data = Chain(internal);
    VulkanRhiDevice& device = *data.device;
    Status<Error> idle = device.context.WaitIdle();
    if (!idle)
    {
        return idle;
    }
    const std::lock_guard<std::mutex> lock(device.mutex);
    UnregisterSwapchainImages(device, data);
    data.acquired = false;
    Status<Error> rebuilt = data.swapchain.Recreate(width, height);
    if (!rebuilt)
    {
        return rebuilt;
    }
    return RegisterSwapchainImages(device, data);
}

SwapChainInfo ChainInfo(const void* internal) noexcept
{
    const SwapChainData& data = *static_cast<const SwapChainData*>(internal);
    SwapChainInfo info{};
    info.valid = true;
    info.format = data.format;
    info.width = data.swapchain.Width();
    info.height = data.swapchain.Height();
    info.imageCount = data.imageCount;
    info.currentImage = data.currentImage;
    info.presentMode = data.presentMode;
    info.acquired = data.acquired;
    return info;
}

[[nodiscard]] const SwapChainInterface* ChainInterface() noexcept
{
    static const SwapChainInterface table = []() noexcept
    {
        SwapChainInterface value{};
        value.acquireNextImage = &ChainAcquire;
        value.present = &ChainPresent;
        value.resize = &ChainResize;
        value.info = &ChainInfo;
        return value;
    }();
    return &table;
}

Status<Error> VkCreateSwapChain(void* internal, const SwapChainDesc& desc, SwapChainHandle* out,
                                SwapChainCreation* creation) noexcept
{
    if (internal == nullptr || out == nullptr || creation == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: swapchain creation needs a device and outputs");
    }
    VulkanRhiDevice& device = Self(internal);
    if (!device.context.Info().presentation)
    {
        return Err(ErrorCode::Unsupported,
                   "vulkan backend: this device was created without presentation support");
    }
    const std::lock_guard<std::mutex> lock(device.mutex);
    void* storage = device.allocator->Allocate(sizeof(SwapChainData), alignof(SwapChainData));
    if (storage == nullptr)
    {
        return Err(ErrorCode::Internal, "vulkan backend: swapchain allocation failed");
    }
    auto* data = new (storage) SwapChainData();
    data->device = &device;
    data->presentMode = desc.presentMode;

    const auto fail = [&](Status<Error> error) noexcept
    {
        UnregisterSwapchainImages(device, *data);
        data->~SwapChainData();
        device.allocator->Deallocate(data);
        return error;
    };
    auto created = VulkanSwapchain::Create(device.context, const_cast<void*>(desc.surface.nativeWindow),
                                           desc.surface.width, desc.surface.height);
    if (!created)
    {
        return fail(Status<Error>::Err(created.Error()));
    }
    data->swapchain = std::move(created.Value());
    const Status<Error> registered = RegisterSwapchainImages(device, *data);
    if (!registered)
    {
        return fail(registered);
    }
    u32 slotIndex = 0;
    if (!device.swapChains.Acquire(&slotIndex))
    {
        return fail(Err(ErrorCode::Internal, "vulkan backend: swapchain table could not grow"));
    }
    device.swapChains.At(slotIndex).data = data;
    data->handle = device.swapChains.HandleOf<SwapChainTag>(slotIndex);
    *out = data->handle;
    creation->interface = ChainInterface();
    creation->internal = data;
    return Status<Error>::Ok();
}

Status<Error> VkDestroySwapChain(void* internal, SwapChainHandle handle) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    // The GPU may still be reading the images.
    (void)device.context.WaitIdle();
    const std::lock_guard<std::mutex> lock(device.mutex);
    SwapChainSlot* slot = device.swapChains.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: swapchain handle is stale or invalid");
    }
    SwapChainData* data = slot->data;
    UnregisterSwapchainImages(device, *data);
    data->~SwapChainData();
    device.allocator->Deallocate(data);
    device.swapChains.Release(handle.index);
    return Status<Error>::Ok();
}

}  // namespace

// ---- Installation and teardown ---------------------------------------------

void InstallGpuEntries(DeviceInterface& table) noexcept
{
    table.transition = &VkTransition;
    table.createBindGroupLayout = &VkCreateBindGroupLayout;
    table.destroyBindGroupLayout = &VkDestroyBindGroupLayout;
    table.createBindGroup = &VkCreateBindGroup;
    table.updateBindGroup = &VkUpdateBindGroup;
    table.destroyBindGroup = &VkDestroyBindGroup;
    table.bindGroupInfo = &VkBindGroupInfo;
    table.createPipelineLayout = &VkCreatePipelineLayout;
    table.destroyPipelineLayout = &VkDestroyPipelineLayout;
    table.createGraphicsPipeline = &VkCreateGraphicsPipeline;
    table.destroyGraphicsPipeline = &VkDestroyPipeline;
    table.createComputePipeline = &VkCreateComputePipeline;
    table.destroyComputePipeline = &VkDestroyPipeline;
    table.pipelineInfo = &VkPipelineInfo;
    table.createCommandList = &VkCreateCommandList;
    table.destroyCommandList = &VkDestroyCommandList;
    table.submit = &VkSubmit;
    table.submissionInfo = &VkSubmissionInfo;
    table.createSwapChain = &VkCreateSwapChain;
    table.destroySwapChain = &VkDestroySwapChain;
}

Status<Error> InitializeGpuObjects(VulkanRhiDevice& device) noexcept
{
    const vk::DeviceApiExt& ext = device.context.DeviceFunctionsExt();
    const VulkanDeviceInfo& info = device.context.Info();

    // Queue families that will touch resources: more than one makes new resources
    // concurrent-shared, so cross-queue use needs no ownership transfer.
    u32 families[3] = {info.queues.graphics, info.queues.compute, info.queues.transfer};
    u32 distinct[3] = {};
    u32 distinctCount = 0;
    for (const u32 family : families)
    {
        bool known = false;
        for (u32 index = 0; index < distinctCount; ++index)
        {
            known = known || distinct[index] == family;
        }
        if (!known)
        {
            distinct[distinctCount++] = family;
        }
    }
    device.memory.SetQueueFamilies(distinct, distinctCount);

    vk::SemaphoreTypeCreateInfo type{};
    vk::SemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.pNext = &type;
    if (ext.createSemaphore(device.context.Device(), &semaphoreInfo, nullptr,
                            &device.internalTimeline) != vk::Result::Success)
    {
        device.internalTimeline = 0;
        return Err(ErrorCode::Internal, "vulkan backend: internal timeline creation failed");
    }
    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::CommandPoolCreate::kTransient | vk::CommandPoolCreate::kResetCommandBuffer;
    poolInfo.queueFamilyIndex = info.queues.graphics;
    if (ext.createCommandPool(device.context.Device(), &poolInfo, nullptr,
                              &device.transitionPool) != vk::Result::Success)
    {
        device.transitionPool = 0;
        return Err(ErrorCode::Internal, "vulkan backend: transition pool creation failed");
    }
    const Status<Error> empty = device.pipelines.CreateSetLayout(nullptr, 0, &device.emptySetLayout);
    if (!empty)
    {
        return empty;
    }
    device.drawIndirectCountSupported = info.drawIndirectCount;
    if (info.debugUtilsEnabled)
    {
        device.beginLabel = reinterpret_cast<vk::PFN_CmdBeginDebugUtilsLabelEXT>(
            device.context.ResolveDeviceFunction("vkCmdBeginDebugUtilsLabelEXT"));
        device.endLabel = reinterpret_cast<vk::PFN_CmdEndDebugUtilsLabelEXT>(
            device.context.ResolveDeviceFunction("vkCmdEndDebugUtilsLabelEXT"));
        device.insertLabel = reinterpret_cast<vk::PFN_CmdInsertDebugUtilsLabelEXT>(
            device.context.ResolveDeviceFunction("vkCmdInsertDebugUtilsLabelEXT"));
    }
    return Status<Error>::Ok();
}

void VulkanRhiDevice::ReleaseGpuObjects() noexcept
{
    const vk::DeviceApiExt& ext = context.DeviceFunctionsExt();
    for (usize index = 0; index < commandLists.Size(); ++index)
    {
        if (commandLists.At(index).alive)
        {
            DestroyListData(*this, commandLists.At(index).data);
        }
    }
    for (usize index = 0; index < swapChains.Size(); ++index)
    {
        if (swapChains.At(index).alive)
        {
            SwapChainData* data = swapChains.At(index).data;
            data->~SwapChainData();
            allocator->Deallocate(data);
        }
    }
    for (usize index = 0; index < pipelineTable.Size(); ++index)
    {
        if (pipelineTable.At(index).alive)
        {
            pipelines.DestroyPipeline(pipelineTable.At(index).pipeline);
        }
    }
    for (usize index = 0; index < pipelineLayouts.Size(); ++index)
    {
        if (pipelineLayouts.At(index).alive)
        {
            pipelines.DestroyPipelineLayout(pipelineLayouts.At(index).layout);
        }
    }
    // Destroying a pool frees every set allocated from it.
    for (usize index = 0; index < descriptorPools.Size(); ++index)
    {
        pipelines.DestroyDescriptorPool(descriptorPools[index]);
    }
    descriptorPools.Clear();
    for (usize index = 0; index < layouts.Size(); ++index)
    {
        if (layouts.At(index).alive)
        {
            pipelines.DestroySetLayout(layouts.At(index).layout);
        }
    }
    if (emptySetLayout != 0)
    {
        pipelines.DestroySetLayout(emptySetLayout);
        emptySetLayout = 0;
    }
    if (transitionPool != 0)
    {
        ext.destroyCommandPool(context.Device(), transitionPool, nullptr);
        transitionPool = 0;
    }
    if (internalTimeline != 0)
    {
        ext.destroySemaphore(context.Device(), internalTimeline, nullptr);
        internalTimeline = 0;
    }
}

}  // namespace cyber::rhi::detail

#endif
