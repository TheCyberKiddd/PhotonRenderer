#include "vulkan_device.h"

#if defined(CYBER_RHI_HAS_VULKAN)

#include "vulkan_device_internal.h"

#include <cyber/rhi/command_list.h>
#include <cyber/rhi/swapchain.h>

namespace cyber::rhi::detail
{
namespace
{


constexpr ResourceState kGenericRead = ResourceState::CopySource | ResourceState::VertexBuffer |
                                       ResourceState::IndexBuffer | ResourceState::ConstantBuffer |
                                       ResourceState::ShaderRead | ResourceState::IndirectArgument;

// ---- Device reporting ------------------------------------------------------

Capabilities VkCapabilities(const void* internal) noexcept
{
    return Self(internal).capabilities;
}

MemoryStats VkMemoryStats(const void* internal) noexcept
{
    const VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    return device.memory.Statistics();
}

QueueHandle VkQueue(const void* /*internal*/, QueueKind kind) noexcept
{
    // The context resolves a queue for every kind, aliasing graphics when the
    // device has no dedicated family, so every kind has a handle.
    QueueHandle handle;
    if (static_cast<u32>(kind) < kQueueKindCount)
    {
        handle.index = static_cast<u32>(kind);
        handle.generation = 1;
    }
    return handle;
}

u32 VkReportLeaks(const void* internal) noexcept
{
    const VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    return device.buffers.AliveCount() + device.textures.AliveCount() +
           device.views.AliveCount() + device.samplers.AliveCount() +
           device.shaders.AliveCount() + device.fences.AliveCount();
}

Status<Error> VkWaitIdle(void* internal) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    return device.context.WaitIdle();
}

// ---- Fences ----------------------------------------------------------------

Status<Error> VkCreateFence(void* internal, const FenceDesc& desc, FenceHandle* out) noexcept
{
    if (internal == nullptr || out == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: fence creation needs a device and an output");
    }
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    u32 index = 0;
    if (!device.fences.Acquire(&index))
    {
        return Err(ErrorCode::Internal, "vulkan backend: fence table could not grow");
    }
    vk::SemaphoreTypeCreateInfo type{};
    type.initialValue = desc.initialValue;
    vk::SemaphoreCreateInfo info{};
    info.pNext = &type;
    FenceSlot& slot = device.fences.At(index);
    if (device.context.DeviceFunctionsExt().createSemaphore(device.context.Device(), &info, nullptr,
                                                            &slot.semaphore) != vk::Result::Success)
    {
        device.fences.Release(index);
        return Err(ErrorCode::Internal, "vulkan backend: vkCreateSemaphore failed");
    }
    CopyLabel(slot.name, desc.label);
    if (desc.label != nullptr)
    {
        (void)device.context.SetObjectName(vk::ObjectType::Semaphore, slot.semaphore, desc.label);
    }
    *out = device.fences.HandleOf<FenceTag>(index);
    return Status<Error>::Ok();
}

Status<Error> VkDestroyFence(void* internal, FenceHandle handle) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    FenceSlot* slot = device.fences.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: fence handle is stale or invalid");
    }
    device.context.DeviceFunctionsExt().destroySemaphore(device.context.Device(), slot->semaphore,
                                                         nullptr);
    device.fences.Release(handle.index);
    return Status<Error>::Ok();
}

Status<Error> VkSignalFence(void* internal, FenceHandle handle, u64 value) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    FenceSlot* slot = device.fences.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: fence handle is stale or invalid");
    }
    const vk::DeviceApiExt& ext = device.context.DeviceFunctionsExt();
    u64 current = 0;
    (void)ext.getSemaphoreCounterValue(device.context.Device(), slot->semaphore, &current);
    // A timeline only moves forward; the RHI contract treats a lower signal as
    // already satisfied, so it is accepted and ignored.
    if (value <= current)
    {
        return Status<Error>::Ok();
    }
    vk::SemaphoreSignalInfo info{};
    info.semaphore = slot->semaphore;
    info.value = value;
    if (ext.signalSemaphore(device.context.Device(), &info) != vk::Result::Success)
    {
        return Err(ErrorCode::Internal, "vulkan backend: vkSignalSemaphore failed");
    }
    return Status<Error>::Ok();
}

Status<Error> VkWaitFence(void* internal, FenceHandle handle, u64 value,
                          u64 timeoutNanoseconds) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    vk::Semaphore semaphore = 0;
    {
        const std::lock_guard<std::mutex> lock(device.mutex);
        FenceSlot* slot = device.fences.Resolve(handle);
        if (slot == nullptr)
        {
            return Err(ErrorCode::InvalidArgument,
                       "vulkan backend: fence handle is stale or invalid");
        }
        semaphore = slot->semaphore;
    }
    // The wait runs outside the lock so another thread can signal or create
    // objects meanwhile.
    vk::SemaphoreWaitInfo wait{};
    wait.semaphoreCount = 1;
    wait.pSemaphores = &semaphore;
    wait.pValues = &value;
    const vk::Result result = device.context.DeviceFunctionsExt().waitSemaphores(
        device.context.Device(), &wait, timeoutNanoseconds);
    if (result == vk::Result::Success)
    {
        return Status<Error>::Ok();
    }
    if (result == vk::Result::Timeout)
    {
        // The same code the D3D12 backend uses for "not reached in time".
        return Err(ErrorCode::NotFound, "vulkan backend: fence did not reach the value in time");
    }
    if (result == vk::Result::ErrorDeviceLost)
    {
        return Err(ErrorCode::DeviceLost, "vulkan backend: device lost while waiting on a fence");
    }
    return Err(ErrorCode::Internal, "vulkan backend: vkWaitSemaphores failed");
}

u64 VkFenceValue(const void* internal, FenceHandle handle) noexcept
{
    const VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    const FenceSlot* slot = device.fences.Resolve(handle);
    if (slot == nullptr)
    {
        return 0;
    }
    u64 value = 0;
    (void)device.context.DeviceFunctionsExt().getSemaphoreCounterValue(
        device.context.Device(), slot->semaphore, &value);
    return value;
}

// ---- Buffers ---------------------------------------------------------------

[[nodiscard]] vk::Flags ToVulkanBufferUsage(BufferUsage usage) noexcept
{
    vk::Flags flags = 0;
    if (HasUsage(usage, BufferUsage::Vertex))
    {
        flags |= vk::BufferUsage::kVertexBuffer;
    }
    if (HasUsage(usage, BufferUsage::Index))
    {
        flags |= vk::BufferUsage::kIndexBuffer;
    }
    if (HasUsage(usage, BufferUsage::Uniform))
    {
        flags |= vk::BufferUsage::kUniformBuffer;
    }
    if (HasUsage(usage, BufferUsage::Storage))
    {
        flags |= vk::BufferUsage::kStorageBuffer;
    }
    if (HasUsage(usage, BufferUsage::Indirect))
    {
        flags |= vk::BufferUsage::kIndirectBuffer;
    }
    if (HasUsage(usage, BufferUsage::CopySource))
    {
        flags |= vk::BufferUsage::kTransferSrc;
    }
    if (HasUsage(usage, BufferUsage::CopyDestination))
    {
        flags |= vk::BufferUsage::kTransferDst;
    }
    // A buffer with only host-access usage still needs a Vulkan usage bit.
    return flags != 0 ? flags : vk::BufferUsage::kTransferSrc | vk::BufferUsage::kTransferDst;
}

Status<Error> VkCreateBuffer(void* internal, const BufferDesc& desc, BufferHandle* out) noexcept
{
    if (internal == nullptr || out == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: buffer creation needs a device and an output handle");
    }
    if (desc.sizeBytes == 0)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: buffer size is zero");
    }
    const bool hostVisible = HasUsage(desc.usage, BufferUsage::HostVisible);
    const bool readback = HasUsage(desc.usage, BufferUsage::Readback);
    if (hostVisible && readback)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: a buffer is HostVisible or Readback, not both");
    }
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    u32 index = 0;
    if (!device.buffers.Acquire(&index))
    {
        return Err(ErrorCode::Internal, "vulkan backend: buffer table could not grow");
    }
    VulkanBufferDesc native{};
    native.sizeBytes = desc.sizeBytes;
    native.usage = ToVulkanBufferUsage(desc.usage);
    native.memoryClass = hostVisible ? VulkanMemoryClass::Upload
                                     : (readback ? VulkanMemoryClass::Readback
                                                 : VulkanMemoryClass::DeviceLocal);
    BufferSlot& slot = device.buffers.At(index);
    const Status<Error> created = device.memory.CreateBuffer(native, &slot.buffer);
    if (!created)
    {
        device.buffers.Release(index);
        return created;
    }
    slot.desc = desc;
    slot.state = hostVisible ? kGenericRead
                             : (readback ? ResourceState::CopyDestination : ResourceState::Undefined);
    CopyLabel(slot.name, desc.label);
    if (desc.label != nullptr)
    {
        (void)device.context.SetObjectName(vk::ObjectType::Buffer, slot.buffer.buffer, desc.label);
    }
    *out = device.buffers.HandleOf<BufferTag>(index);
    return Status<Error>::Ok();
}

Status<Error> VkDestroyBuffer(void* internal, BufferHandle handle) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    BufferSlot* slot = device.buffers.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: buffer handle is stale or invalid");
    }
    device.memory.DestroyBuffer(slot->buffer);
    device.buffers.Release(handle.index);
    return Status<Error>::Ok();
}

BufferInfo VkBufferInfo(const void* internal, BufferHandle handle) noexcept
{
    const VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    BufferInfo info{};
    if (const BufferSlot* slot = device.buffers.Resolve(handle))
    {
        info.valid = true;
        info.sizeBytes = slot->desc.sizeBytes;
        info.elementStride = slot->desc.elementStride;
        info.usage = slot->desc.usage;
        info.lifetime = slot->desc.lifetime;
    }
    return info;
}

Status<Error> VkWriteBuffer(void* internal, BufferHandle handle, u64 offset,
                            memory::Span<const memory::Byte> bytes) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    BufferSlot* slot = device.buffers.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: buffer handle is stale or invalid");
    }
    if (!HasUsage(slot->desc.usage, BufferUsage::HostVisible) || slot->buffer.block.mapped == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: buffer is not HostVisible, so it cannot be written");
    }
    if (offset > slot->desc.sizeBytes || bytes.Size() > slot->desc.sizeBytes - offset)
    {
        return Err(ErrorCode::OutOfRange, "vulkan backend: write range is outside the buffer");
    }
    if (bytes.Size() > 0)
    {
        std::memcpy(static_cast<u8*>(slot->buffer.block.mapped) + offset, bytes.Data(),
                    bytes.Size());
    }
    return Status<Error>::Ok();
}

Status<Error> VkReadBuffer(void* internal, BufferHandle handle, u64 offset,
                           memory::Span<memory::Byte> bytes) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    BufferSlot* slot = device.buffers.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: buffer handle is stale or invalid");
    }
    if (!HasUsage(slot->desc.usage, BufferUsage::Readback) || slot->buffer.block.mapped == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: buffer is not Readback, so it cannot be read");
    }
    if (offset > slot->desc.sizeBytes || bytes.Size() > slot->desc.sizeBytes - offset)
    {
        return Err(ErrorCode::OutOfRange, "vulkan backend: read range is outside the buffer");
    }
    if (bytes.Size() > 0)
    {
        std::memcpy(bytes.Data(), static_cast<const u8*>(slot->buffer.block.mapped) + offset,
                    bytes.Size());
    }
    return Status<Error>::Ok();
}

ResourceState VkBufferState(const void* internal, BufferHandle handle) noexcept
{
    const VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    const BufferSlot* slot = device.buffers.Resolve(handle);
    return slot != nullptr ? slot->state : ResourceState::Undefined;
}

// ---- Textures --------------------------------------------------------------

Status<Error> VkCreateTexture(void* internal, const TextureDesc& desc, TextureHandle* out) noexcept
{
    if (internal == nullptr || out == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: texture creation needs a device and an output handle");
    }
    if (desc.reserved)
    {
        return Err(ErrorCode::Unsupported,
                   "vulkan backend: reserved (sparse) textures are not implemented yet");
    }
    if (desc.width == 0 || desc.height == 0 || desc.depth == 0 || desc.arrayLayers == 0 ||
        desc.mipLevels == 0 || desc.usage == TextureUsage::None)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: texture needs extent, layers, mips and a usage");
    }
    const vk::Format format = ToVulkanFormat(desc.format);
    if (format == vk::Format::Undefined)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: texture format is not valid");
    }
    VulkanRhiDevice& device = Self(internal);
    if (IsBlockCompressed(desc.format) && !device.context.Info().textureCompressionBc)
    {
        return Err(ErrorCode::Unsupported,
                   "vulkan backend: this device does not support BC texture compression");
    }

    VulkanImageDesc native{};
    native.width = desc.width;
    native.height = desc.height;
    native.depth = desc.dimension == TextureDimension::Dim3D ? desc.depth : 1;
    native.mipLevels = desc.mipLevels;
    native.arrayLayers = desc.dimension == TextureDimension::Dim3D ? 1 : desc.arrayLayers;
    native.format = format;
    switch (desc.dimension)
    {
    case TextureDimension::Dim1D:
        native.type = vk::ImageType::Type1D;
        break;
    case TextureDimension::Dim3D:
        native.type = vk::ImageType::Type3D;
        break;
    case TextureDimension::Dim2D:
    case TextureDimension::Cube:
    default:
        native.type = vk::ImageType::Type2D;
        break;
    }
    if (desc.dimension == TextureDimension::Cube && desc.arrayLayers % 6 != 0)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: a cube texture needs a multiple of six layers");
    }
    if (HasUsage(desc.usage, TextureUsage::Sampled))
    {
        native.usage |= vk::ImageUsage::kSampled;
    }
    if (HasUsage(desc.usage, TextureUsage::Storage))
    {
        native.usage |= vk::ImageUsage::kStorage;
    }
    if (HasUsage(desc.usage, TextureUsage::RenderTarget))
    {
        native.usage |= vk::ImageUsage::kColorAttachment;
    }
    if (HasUsage(desc.usage, TextureUsage::DepthStencil))
    {
        native.usage |= vk::ImageUsage::kDepthStencilAttachment;
    }
    if (HasUsage(desc.usage, TextureUsage::CopySource))
    {
        native.usage |= vk::ImageUsage::kTransferSrc;
    }
    if (HasUsage(desc.usage, TextureUsage::CopyDestination))
    {
        native.usage |= vk::ImageUsage::kTransferDst;
    }
    native.samples = desc.sampleCount;
    native.cubeCompatible = desc.dimension == TextureDimension::Cube;

    const std::lock_guard<std::mutex> lock(device.mutex);
    u32 index = 0;
    if (!device.textures.Acquire(&index))
    {
        return Err(ErrorCode::Internal, "vulkan backend: texture table could not grow");
    }
    TextureSlot& slot = device.textures.At(index);
    const Status<Error> created = device.memory.CreateImage(native, &slot.image);
    if (!created)
    {
        device.textures.Release(index);
        return created;
    }
    slot.desc = desc;
    slot.sizeBytes = slot.image.block.sizeBytes;
    *out = device.textures.HandleOf<TextureTag>(index);
    return Status<Error>::Ok();
}

Status<Error> VkDestroyTexture(void* internal, TextureHandle handle) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    TextureSlot* slot = device.textures.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: texture handle is stale or invalid");
    }
    if (slot->external)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: a swapchain image is destroyed with its swapchain");
    }
    device.memory.DestroyImage(slot->image);
    device.textures.Release(handle.index);
    return Status<Error>::Ok();
}

TextureInfo VkTextureInfo(const void* internal, TextureHandle handle) noexcept
{
    const VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    TextureInfo info{};
    if (const TextureSlot* slot = device.textures.Resolve(handle))
    {
        info.valid = true;
        info.dimension = slot->desc.dimension;
        info.format = slot->desc.format;
        info.usage = slot->desc.usage;
        info.width = slot->desc.width;
        info.height = slot->desc.height;
        info.depth = slot->desc.depth;
        info.arrayLayers = slot->desc.arrayLayers;
        info.mipLevels = slot->desc.mipLevels;
        info.sampleCount = slot->desc.sampleCount;
        info.lifetime = slot->desc.lifetime;
        info.sizeBytes = slot->sizeBytes;
        info.reserved = slot->desc.reserved;
    }
    return info;
}

ResourceState VkTextureState(const void* internal, TextureHandle handle, u32 mipLevel,
                             u32 arrayLayer) noexcept
{
    const VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    const TextureSlot* slot = device.textures.Resolve(handle);
    if (slot == nullptr || mipLevel >= slot->desc.mipLevels || arrayLayer >= slot->desc.arrayLayers)
    {
        return ResourceState::Undefined;
    }
    const u32 total = slot->desc.mipLevels * slot->desc.arrayLayers;
    // Beyond the tracked count the whole texture shares one state, as in the
    // D3D12 backend.
    const u32 index = total <= kMaxTrackedSubresources ? arrayLayer * slot->desc.mipLevels + mipLevel : 0;
    return slot->states[index];
}

// ---- Views -----------------------------------------------------------------

Status<Error> VkCreateTextureView(void* internal, const TextureViewDesc& desc,
                                  TextureViewHandle* out) noexcept
{
    if (internal == nullptr || out == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: view creation needs a device and an output handle");
    }
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    TextureSlot* texture = device.textures.Resolve(desc.texture);
    if (texture == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: view texture is stale or invalid");
    }
    const vk::Format format = ToVulkanFormat(desc.format);
    if (format == vk::Format::Undefined)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: view format is not valid");
    }
    // A count of zero means "to the end", the RHI's convention for a range.
    const u32 mipCount = desc.range.mipLevelCount != 0
                             ? desc.range.mipLevelCount
                             : (texture->desc.mipLevels > desc.range.baseMipLevel
                                    ? texture->desc.mipLevels - desc.range.baseMipLevel
                                    : 0);
    const u32 layerCount = desc.range.layerCount != 0
                               ? desc.range.layerCount
                               : (texture->desc.arrayLayers > desc.range.baseArrayLayer
                                      ? texture->desc.arrayLayers - desc.range.baseArrayLayer
                                      : 0);
    if (mipCount == 0 || layerCount == 0 ||
        desc.range.baseMipLevel + mipCount > texture->desc.mipLevels ||
        desc.range.baseArrayLayer + layerCount > texture->desc.arrayLayers)
    {
        return Err(ErrorCode::OutOfRange, "vulkan backend: view range is outside the texture");
    }

    vk::ImageViewType type = vk::ImageViewType::Type2D;
    switch (desc.dimension)
    {
    case TextureDimension::Dim1D:
        type = layerCount > 1 ? static_cast<vk::ImageViewType>(4) : vk::ImageViewType::Type1D;
        break;
    case TextureDimension::Dim3D:
        type = vk::ImageViewType::Type3D;
        break;
    case TextureDimension::Cube:
        type = layerCount > 6 ? static_cast<vk::ImageViewType>(6) : vk::ImageViewType::Cube;
        break;
    case TextureDimension::Dim2D:
    default:
        type = layerCount > 1 ? vk::ImageViewType::Type2DArray : vk::ImageViewType::Type2D;
        break;
    }
    vk::Flags aspect = vk::ImageAspect::kColor;
    switch (desc.range.aspect)
    {
    case TextureAspect::Color:
        aspect = vk::ImageAspect::kColor;
        break;
    case TextureAspect::Depth:
        aspect = vk::ImageAspect::kDepth;
        break;
    case TextureAspect::Stencil:
        aspect = vk::ImageAspect::kStencil;
        break;
    case TextureAspect::DepthStencil:
        aspect = vk::ImageAspect::kDepth | vk::ImageAspect::kStencil;
        break;
    default:
        break;
    }

    u32 index = 0;
    if (!device.views.Acquire(&index))
    {
        return Err(ErrorCode::Internal, "vulkan backend: view table could not grow");
    }
    ViewSlot& slot = device.views.At(index);
    vk::ImageViewCreateInfo info{};
    info.image = texture->image.image;
    info.viewType = type;
    info.format = format;
    info.subresourceRange = vk::ImageSubresourceRange{aspect, desc.range.baseMipLevel, mipCount,
                                                      desc.range.baseArrayLayer, layerCount};
    if (device.context.DeviceFunctionsExt().createImageView(device.context.Device(), &info, nullptr,
                                                            &slot.view) != vk::Result::Success)
    {
        device.views.Release(index);
        return Err(ErrorCode::Internal, "vulkan backend: vkCreateImageView failed");
    }
    slot.texture = desc.texture;
    slot.desc = desc;
    slot.desc.range.mipLevelCount = mipCount;
    slot.desc.range.layerCount = layerCount;
    slot.desc.label = nullptr;
    CopyLabel(slot.name, desc.label);
    if (desc.label != nullptr)
    {
        (void)device.context.SetObjectName(vk::ObjectType::ImageView, slot.view, desc.label);
    }
    *out = device.views.HandleOf<TextureViewTag>(index);
    return Status<Error>::Ok();
}

Status<Error> VkDestroyTextureView(void* internal, TextureViewHandle handle) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    ViewSlot* slot = device.views.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: view handle is stale or invalid");
    }
    device.memory.DestroyImageView(slot->view);
    device.views.Release(handle.index);
    return Status<Error>::Ok();
}

TextureViewInfo VkTextureViewInfo(const void* internal, TextureViewHandle handle) noexcept
{
    const VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    TextureViewInfo info{};
    if (const ViewSlot* slot = device.views.Resolve(handle))
    {
        info.valid = true;
        info.format = slot->desc.format;
        info.dimension = slot->desc.dimension;
        info.usage = slot->desc.usage;
        info.range = slot->desc.range;
    }
    return info;
}

// ---- Samplers --------------------------------------------------------------

Status<Error> VkCreateSampler(void* internal, const SamplerDesc& desc, SamplerHandle* out) noexcept
{
    if (internal == nullptr || out == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: sampler creation needs a device and an output handle");
    }
    VulkanRhiDevice& device = Self(internal);
    const auto filter = [](FilterMode mode) noexcept
    { return mode == FilterMode::Nearest ? vk::Filter::Nearest : vk::Filter::Linear; };
    const auto address = [](AddressMode mode) noexcept
    {
        switch (mode)
        {
        case AddressMode::Repeat:
            return vk::SamplerAddressMode::Repeat;
        case AddressMode::MirroredRepeat:
            return vk::SamplerAddressMode::MirroredRepeat;
        case AddressMode::ClampToBorder:
            return vk::SamplerAddressMode::ClampToBorder;
        case AddressMode::ClampToEdge:
        default:
            return vk::SamplerAddressMode::ClampToEdge;
        }
    };
    vk::SamplerCreateInfo info{};
    info.minFilter = filter(desc.minFilter);
    info.magFilter = filter(desc.magFilter);
    info.mipmapMode = desc.mipmapFilter == FilterMode::Nearest ? vk::SamplerMipmapMode::Nearest
                                                               : vk::SamplerMipmapMode::Linear;
    info.addressModeU = address(desc.addressU);
    info.addressModeV = address(desc.addressV);
    info.addressModeW = address(desc.addressW);
    info.mipLodBias = desc.mipLodBias;
    // Anisotropy needs the core feature; without it the request degrades to
    // trilinear rather than failing, as the capability report says.
    if (desc.maxAnisotropy > 1 && device.context.Info().samplerAnisotropy)
    {
        info.anisotropyEnable = vk::kTrue;
        info.maxAnisotropy = static_cast<float>(desc.maxAnisotropy);
    }
    info.compareEnable = desc.compareEnabled ? vk::kTrue : vk::kFalse;
    info.compareOp = static_cast<vk::CompareOp>(static_cast<u32>(desc.compareFunction));
    info.minLod = desc.minLod;
    info.maxLod = desc.maxLod;
    switch (desc.borderColor)
    {
    case BorderColor::OpaqueBlack:
        info.borderColor = vk::BorderColor::FloatOpaqueBlack;
        break;
    case BorderColor::OpaqueWhite:
        info.borderColor = vk::BorderColor::FloatOpaqueWhite;
        break;
    case BorderColor::TransparentBlack:
    default:
        info.borderColor = vk::BorderColor::FloatTransparentBlack;
        break;
    }

    const std::lock_guard<std::mutex> lock(device.mutex);
    u32 index = 0;
    if (!device.samplers.Acquire(&index))
    {
        return Err(ErrorCode::Internal, "vulkan backend: sampler table could not grow");
    }
    SamplerSlot& slot = device.samplers.At(index);
    const Status<Error> created = device.memory.CreateSampler(info, &slot.sampler);
    if (!created)
    {
        device.samplers.Release(index);
        return created;
    }
    slot.desc = desc;
    slot.desc.label = nullptr;
    CopyLabel(slot.name, desc.label);
    if (desc.label != nullptr)
    {
        (void)device.context.SetObjectName(vk::ObjectType::Sampler, slot.sampler, desc.label);
    }
    *out = device.samplers.HandleOf<SamplerTag>(index);
    return Status<Error>::Ok();
}

Status<Error> VkDestroySampler(void* internal, SamplerHandle handle) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    SamplerSlot* slot = device.samplers.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: sampler handle is stale or invalid");
    }
    device.memory.DestroySampler(slot->sampler);
    device.samplers.Release(handle.index);
    return Status<Error>::Ok();
}

// ---- Shaders ---------------------------------------------------------------

Status<Error> VkCreateShader(void* internal, const ShaderDesc& desc, ShaderHandle* out) noexcept
{
    if (internal == nullptr || out == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: shader creation needs a device and an output handle");
    }
    if (desc.format != ShaderBytecodeFormat::SpirV)
    {
        return Err(ErrorCode::Unsupported,
                   "vulkan backend: shaders must be SPIR-V; the shader compiler has no SPIR-V output yet");
    }
    if (desc.code.Size() < 20 || desc.code.Size() % 4 != 0)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: SPIR-V code must be a whole number of 32-bit words");
    }
    VulkanRhiDevice& device = Self(internal);
    // The caller's bytes may not be 4-aligned, so the words are copied.
    DynamicArray<u32> words(*device.allocator);
    if (!words.TryResize(desc.code.Size() / 4))
    {
        return Err(ErrorCode::Internal, "vulkan backend: shader copy failed to allocate");
    }
    std::memcpy(words.Data(), desc.code.Data(), desc.code.Size());

    const std::lock_guard<std::mutex> lock(device.mutex);
    u32 index = 0;
    if (!device.shaders.Acquire(&index))
    {
        return Err(ErrorCode::Internal, "vulkan backend: shader table could not grow");
    }
    ShaderSlot& slot = device.shaders.At(index);
    const Status<Error> created =
        device.pipelines.CreateShader(words.Data(), words.Size(), &slot.module);
    if (!created)
    {
        device.shaders.Release(index);
        return created;
    }
    slot.stage = desc.stage;
    slot.codeSize = desc.code.Size();
    {
        const char* entryName = desc.entryPoint != nullptr ? desc.entryPoint : "main";
        u32 length = 0;
        while (length + 1 < sizeof(slot.entry) && entryName[length] != 0)
        {
            slot.entry[length] = entryName[length];
            ++length;
        }
        slot.entry[length] = 0;
    }
    CopyLabel(slot.name, desc.label);
    if (desc.label != nullptr)
    {
        (void)device.context.SetObjectName(vk::ObjectType::ShaderModule, slot.module, desc.label);
    }
    *out = device.shaders.HandleOf<ShaderTag>(index);
    return Status<Error>::Ok();
}

Status<Error> VkDestroyShader(void* internal, ShaderHandle handle) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    ShaderSlot* slot = device.shaders.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument, "vulkan backend: shader handle is stale or invalid");
    }
    device.pipelines.DestroyShader(slot->module);
    device.shaders.Release(handle.index);
    return Status<Error>::Ok();
}

ShaderInfo VkShaderInfo(const void* internal, ShaderHandle handle) noexcept
{
    const VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    ShaderInfo info{};
    if (const ShaderSlot* slot = device.shaders.Resolve(handle))
    {
        info.valid = true;
        info.stage = slot->stage;
        info.format = ShaderBytecodeFormat::SpirV;
        info.codeSize = slot->codeSize;
    }
    return info;
}

// ---- Debug names -----------------------------------------------------------

ObjectNameInfo VkObjectName(const void* internal, const ObjectRef& object) noexcept
{
    const VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    const char* name = nullptr;
    switch (object.kind)
    {
    case ObjectKind::Buffer:
        if (const BufferSlot* slot = device.buffers.Resolve(UnpackHandle<BufferTag>(object.packed)))
        {
            name = slot->name;
        }
        break;
    case ObjectKind::Texture:
        if (const TextureSlot* slot = device.textures.Resolve(UnpackHandle<TextureTag>(object.packed)))
        {
            name = slot->name;
        }
        break;
    case ObjectKind::TextureView:
        if (const ViewSlot* slot = device.views.Resolve(UnpackHandle<TextureViewTag>(object.packed)))
        {
            name = slot->name;
        }
        break;
    case ObjectKind::Sampler:
        if (const SamplerSlot* slot = device.samplers.Resolve(UnpackHandle<SamplerTag>(object.packed)))
        {
            name = slot->name;
        }
        break;
    case ObjectKind::Shader:
        if (const ShaderSlot* slot = device.shaders.Resolve(UnpackHandle<ShaderTag>(object.packed)))
        {
            name = slot->name;
        }
        break;
    case ObjectKind::Fence:
        if (const FenceSlot* slot = device.fences.Resolve(UnpackHandle<FenceTag>(object.packed)))
        {
            name = slot->name;
        }
        break;
    default:
        break;
    }
    ObjectNameInfo info{};
    if (name != nullptr && name[0] != '\0')
    {
        info.named = true;
        u32 length = 0;
        while (length + 1 < sizeof(info.name) && name[length] != '\0')
        {
            info.name[length] = name[length];
            ++length;
        }
        info.name[length] = '\0';
        info.length = length;
    }
    return info;
}

// ---- Features that are not implemented ------------------------------------

Status<Error> VkCreateTimestampQueryHeap(void*, const TimestampQueryHeapDesc&,
                                         TimestampQueryHeapHandle*) noexcept
{
    return Err(ErrorCode::Unsupported, "vulkan backend: timestamp queries are not implemented yet");
}
Status<Error> VkDestroyTimestampQueryHeap(void*, TimestampQueryHeapHandle) noexcept
{
    return Err(ErrorCode::Unsupported, "vulkan backend: timestamp queries are not implemented yet");
}
Status<Error> VkReadTimestamps(void*, TimestampQueryHeapHandle, u32, u32, FenceHandle, u64,
                               memory::Span<u64>) noexcept
{
    return Err(ErrorCode::Unsupported, "vulkan backend: timestamp queries are not implemented yet");
}
u64 VkTimestampFrequency(const void*, QueueKind) noexcept
{
    return 0;
}
Status<Error> VkCreateTileHeap(void*, const TileHeapDesc&, TileHeapHandle*) noexcept
{
    return Err(ErrorCode::Unsupported, "vulkan backend: reserved textures are not implemented yet");
}
Status<Error> VkDestroyTileHeap(void*, TileHeapHandle) noexcept
{
    return Err(ErrorCode::Unsupported, "vulkan backend: reserved textures are not implemented yet");
}
TextureTiling VkTextureTiling(const void*, TextureHandle) noexcept
{
    return TextureTiling{};
}
Status<Error> VkUpdateTileMappings(void*, QueueKind, memory::Span<const TileMapping>) noexcept
{
    return Err(ErrorCode::Unsupported, "vulkan backend: reserved textures are not implemented yet");
}
TileMappingInfo VkTileMapping(const void*, const TileMapping&) noexcept
{
    return TileMappingInfo{};
}
Status<Error> VkCreatePipelineCache(void*, const PipelineCacheDesc&, PipelineCacheHandle*) noexcept
{
    return Err(ErrorCode::Unsupported, "vulkan backend: pipeline caches are not implemented yet");
}
Status<Error> VkDestroyPipelineCache(void*, PipelineCacheHandle) noexcept
{
    return Err(ErrorCode::Unsupported, "vulkan backend: pipeline caches are not implemented yet");
}
Status<Error> VkPipelineCacheBytes(const void*, PipelineCacheHandle, memory::Allocator&,
                                   memory::DynamicArray<memory::Byte>*) noexcept
{
    return Err(ErrorCode::Unsupported, "vulkan backend: pipeline caches are not implemented yet");
}

[[nodiscard]] DeviceInterface BuildInterface() noexcept
{
    DeviceInterface table{};
    table.capabilities = &VkCapabilities;
    table.memoryStats = &VkMemoryStats;
    table.queue = &VkQueue;
    table.createFence = &VkCreateFence;
    table.destroyFence = &VkDestroyFence;
    table.signalFence = &VkSignalFence;
    table.waitFence = &VkWaitFence;
    table.fenceValue = &VkFenceValue;
    table.createTimestampQueryHeap = &VkCreateTimestampQueryHeap;
    table.destroyTimestampQueryHeap = &VkDestroyTimestampQueryHeap;
    table.readTimestamps = &VkReadTimestamps;
    table.timestampFrequency = &VkTimestampFrequency;
    table.reportLeaks = &VkReportLeaks;
    table.waitIdle = &VkWaitIdle;
    table.createBuffer = &VkCreateBuffer;
    table.destroyBuffer = &VkDestroyBuffer;
    table.bufferInfo = &VkBufferInfo;
    table.writeBuffer = &VkWriteBuffer;
    table.readBuffer = &VkReadBuffer;
    table.bufferState = &VkBufferState;
    table.createTexture = &VkCreateTexture;
    table.destroyTexture = &VkDestroyTexture;
    table.textureInfo = &VkTextureInfo;
    table.createTextureView = &VkCreateTextureView;
    table.destroyTextureView = &VkDestroyTextureView;
    table.textureViewInfo = &VkTextureViewInfo;
    table.textureState = &VkTextureState;
    table.createTileHeap = &VkCreateTileHeap;
    table.destroyTileHeap = &VkDestroyTileHeap;
    table.textureTiling = &VkTextureTiling;
    table.updateTileMappings = &VkUpdateTileMappings;
    table.tileMapping = &VkTileMapping;
    table.createSampler = &VkCreateSampler;
    table.destroySampler = &VkDestroySampler;
    table.createShader = &VkCreateShader;
    table.destroyShader = &VkDestroyShader;
    table.shaderInfo = &VkShaderInfo;
    table.createPipelineCache = &VkCreatePipelineCache;
    table.destroyPipelineCache = &VkDestroyPipelineCache;
    table.pipelineCacheBytes = &VkPipelineCacheBytes;
    table.objectName = &VkObjectName;
    InstallGpuEntries(table);
    return table;
}

[[nodiscard]] Capabilities BuildCapabilities(const VulkanDeviceInfo& info) noexcept
{
    Capabilities capabilities{};
    AdapterInfo& adapter = capabilities.adapter;
    std::memcpy(adapter.name, info.name, sizeof(adapter.name));
    adapter.vendorId = info.vendorId;
    adapter.deviceId = info.deviceId;
    adapter.driverVersion = info.driverVersion;
    adapter.dedicatedVideoMemoryBytes = info.deviceLocalBytes;
    adapter.sharedSystemMemoryBytes = info.hostVisibleBytes;
    adapter.backend = BackendKind::Vulkan;
    adapter.apiVersion = vk::VersionMajor(info.apiVersion) * 100U + vk::VersionMinor(info.apiVersion);

    Limits& limits = capabilities.limits;
    limits.maxTexture1D = info.maxImageDimension1D;
    limits.maxTexture2D = info.maxImageDimension2D;
    limits.maxTexture3D = info.maxImageDimension3D;
    limits.maxTextureCube = info.maxImageDimensionCube;
    limits.maxTextureArrayLayers = info.maxImageArrayLayers;
    // The remaining limits are the values the specification guarantees every
    // device meets. Reading the device's real figures needs the full
    // `VkPhysicalDeviceLimits` layout; until it is declared the report is a safe
    // lower bound, never an overclaim.
    limits.maxColorAttachments = 4;
    limits.maxVertexBuffers = 16;
    limits.maxVertexAttributes = 16;
    limits.maxBindGroups = kBindGroupSlotCount;
    limits.maxBindingsPerGroup = 64;
    limits.maxBindingArrayElements = 1024;
    limits.maxSamplerArrayElements = 1024;
    limits.maxPushConstantBytes = info.maxPushConstantBytes;
    limits.maxSamplerAnisotropy = info.samplerAnisotropy ? 16 : 1;
    limits.minUniformBufferOffsetAlignment = 256;
    limits.minStorageBufferOffsetAlignment = 256;
    limits.maxBufferBytes = 1ULL << 30;

    // Only what this increment can do: the rest is added as each feature lands.
    Features& features = capabilities.features;
    features.Add(Feature::TimelineFences);
    features.Add(Feature::Compute);
    features.Add(Feature::IndirectDraw);
    if (info.multiDrawIndirect)
    {
        features.Add(Feature::MultiDrawIndirect);
    }
    if (info.drawIndirectCount)
    {
        features.Add(Feature::IndirectDrawCount);
    }
    if (info.debugUtilsEnabled)
    {
        features.Add(Feature::DebugMarkers);
    }
    features.Add(Feature::StorageTextures);
    features.Add(Feature::BindlessResources);
    if (info.samplerAnisotropy)
    {
        features.Add(Feature::AnisotropicFiltering);
    }
    return capabilities;
}

}  // namespace

const DeviceInterface* VulkanDeviceInterface() noexcept
{
    static const DeviceInterface table = BuildInterface();
    return &table;
}

Result<DeviceCreation, Error> CreateVulkanDevice(const DeviceDesc& desc,
                                                 memory::Allocator& allocator) noexcept
{
    using Created = Result<DeviceCreation, Error>;
    void* storage = allocator.Allocate(sizeof(VulkanRhiDevice), alignof(VulkanRhiDevice));
    if (storage == nullptr)
    {
        return Created::Err(Error{ErrorCode::Internal, "vulkan backend: device allocation failed"});
    }
    auto* device = new (storage) VulkanRhiDevice(allocator);

    VulkanContextDesc contextDesc{};
    contextDesc.adapterIndex = desc.adapterIndex;
    contextDesc.enableValidation = desc.validation != ValidationLevel::Off;
    auto context = VulkanContext::Create(contextDesc, allocator);
    if (!context)
    {
        const Error error = context.Error();
        device->~VulkanRhiDevice();
        allocator.Deallocate(storage);
        return Created::Err(error);
    }
    device->context = std::move(context.Value());

    const Status<Error> initialized = device->memory.Initialize(
        device->context.Device(), device->context.DeviceFunctions(),
        device->context.DeviceFunctionsExt(), device->context.MemoryProperties(),
        device->context.Info().bufferDeviceAddress);
    if (!initialized)
    {
        const Error error = initialized.Error();
        device->~VulkanRhiDevice();
        allocator.Deallocate(storage);
        return Created::Err(error);
    }
    if (const Status<Error> gpu = InitializeGpuObjects(*device); !gpu)
    {
        const Error error = gpu.Error();
        device->~VulkanRhiDevice();
        allocator.Deallocate(storage);
        return Created::Err(error);
    }
    device->capabilities = BuildCapabilities(device->context.Info());

    DeviceCreation creation{};
    creation.interface = VulkanDeviceInterface();
    creation.internal = device;
    return Created::Ok(creation);
}

void DestroyVulkanDevice(void* internal) noexcept
{
    if (internal == nullptr)
    {
        return;
    }
    auto* device = static_cast<VulkanRhiDevice*>(internal);
    memory::Allocator* allocator = device->allocator;
    device->~VulkanRhiDevice();
    allocator->Deallocate(device);
}

}  // namespace cyber::rhi::detail

#endif
