// The Vulkan RHI device's timestamp queries, pipeline caches and reserved
// (sparse-resident) textures: the three entry groups that answered Unsupported
// until Phase 34 closed. Each mirrors the D3D12 backend's contract, so renderer
// code that profiles, caches pipelines or streams tiles runs on both unchanged.

#include "vulkan_device.h"

#if defined(CYBER_RHI_HAS_VULKAN)

#include "../validation.h"
#include "vulkan_device_internal.h"

#include <cstring>

namespace cyber::rhi::detail
{

namespace
{

[[nodiscard]] u32 MipSize(u32 base, u32 mip) noexcept
{
    const u32 value = base >> mip;
    return value != 0 ? value : 1U;
}

// The semaphore and running value that order work on a queue kind: the graphics
// queue's internal timeline, or the kind's own.
void TimelineOf(VulkanRhiDevice& device, QueueKind kind, vk::Semaphore* semaphore,
                u64** value) noexcept
{
    if (kind == QueueKind::Graphics)
    {
        *semaphore = device.internalTimeline;
        *value = &device.internalValue;
        return;
    }
    const u32 index = static_cast<u32>(kind);
    *semaphore = device.queueTimeline[index];
    *value = &device.queueValue[index];
}

[[nodiscard]] vk::Queue QueueOf(VulkanRhiDevice& device, QueueKind kind) noexcept
{
    switch (kind)
    {
    case QueueKind::Compute:
        return device.context.ComputeQueue();
    case QueueKind::Copy:
        return device.context.TransferQueue();
    default:
        return device.context.GraphicsQueue();
    }
}

}  // namespace

// ---- Timestamps ----------------------------------------------------------------

Status<Error> VkCreateTimestampQueryHeap(void* internal, const TimestampQueryHeapDesc& desc,
                                         TimestampQueryHeapHandle* out) noexcept
{
    if (internal == nullptr || out == nullptr || desc.queryCount == 0)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: timestamp heap needs a device, count, and output");
    }
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    if (!device.capabilities.features.Has(Feature::TimestampQueries))
    {
        return Err(ErrorCode::Unsupported, "vulkan backend: timestamp queries are not supported");
    }
    const u64 readbackBytes = static_cast<u64>(desc.queryCount) * sizeof(u64);
    if (readbackBytes > device.capabilities.limits.maxBufferBytes)
    {
        return Err(ErrorCode::OutOfRange,
                   "vulkan backend: timestamp heap exceeds the readback limit");
    }
    const vk::DeviceApiExt& ext = device.context.DeviceFunctionsExt();
    vk::QueryPoolCreateInfo info{};
    info.queryType = vk::kQueryTypeTimestamp;
    info.queryCount = desc.queryCount;
    vk::QueryPool pool = 0;
    if (ext.createQueryPool(device.context.Device(), &info, nullptr, &pool) != vk::Result::Success)
    {
        return Err(ErrorCode::Internal, "vulkan backend: vkCreateQueryPool failed");
    }
    VulkanBufferDesc native{};
    native.sizeBytes = readbackBytes;
    native.usage = vk::BufferUsage::kTransferDst;
    native.memoryClass = VulkanMemoryClass::Readback;
    VulkanBuffer readback{};
    const Status<Error> created = device.memory.CreateBuffer(native, &readback);
    if (!created || readback.block.mapped == nullptr)
    {
        ext.destroyQueryPool(device.context.Device(), pool, nullptr);
        return created
                   ? Err(ErrorCode::Internal, "vulkan backend: timestamp readback is not mapped")
                   : created;
    }
    std::memset(readback.block.mapped, 0, static_cast<usize>(readbackBytes));
    // Vulkan queries must be reset before their first write; D3D12's need not be.
    // A host reset when the device allows it, otherwise ahead of the next submission.
    if (device.context.Info().hostQueryReset)
    {
        ext.resetQueryPool(device.context.Device(), pool, 0, desc.queryCount);
    }
    else if (!RecordQueryReset(device, pool, 0, desc.queryCount))
    {
        device.memory.DestroyBuffer(readback);
        ext.destroyQueryPool(device.context.Device(), pool, nullptr);
        return Err(ErrorCode::Internal,
                   "vulkan backend: timestamp heap reset could not be recorded");
    }
    u32 index = 0;
    if (!device.timestampHeaps.Acquire(&index))
    {
        device.memory.DestroyBuffer(readback);
        ext.destroyQueryPool(device.context.Device(), pool, nullptr);
        return Err(ErrorCode::Internal, "vulkan backend: timestamp heap table could not grow");
    }
    TimestampHeapSlot& slot = device.timestampHeaps.At(index);
    slot.pool = pool;
    slot.readback = readback;
    slot.queryCount = desc.queryCount;
    CopyLabel(slot.name, desc.label);
    *out = device.timestampHeaps.HandleOf<TimestampQueryHeapTag>(index);
    return Status<Error>::Ok();
}

Status<Error> VkDestroyTimestampQueryHeap(void* internal, TimestampQueryHeapHandle handle) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    TimestampHeapSlot* slot = device.timestampHeaps.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: timestamp heap handle is stale or invalid");
    }
    if (device.initCommands != nullptr)
    {
        (void)FlushResourceInitAndWait(device);  // the pending reset may name the pool
    }
    device.context.DeviceFunctionsExt().destroyQueryPool(device.context.Device(), slot->pool,
                                                         nullptr);
    device.memory.DestroyBuffer(slot->readback);
    device.timestampHeaps.Release(handle.index);
    return Status<Error>::Ok();
}

Status<Error> VkReadTimestamps(void* internal, TimestampQueryHeapHandle handle, u32 firstQuery,
                               u32 queryCount, FenceHandle completionFence, u64 completionValue,
                               memory::Span<u64> out) noexcept
{
    if (internal == nullptr || queryCount == 0 || out.Size() < queryCount)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: timestamp read needs a device and enough output slots");
    }
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    const TimestampHeapSlot* slot = device.timestampHeaps.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: timestamp heap handle is stale or invalid");
    }
    if (firstQuery >= slot->queryCount || queryCount > slot->queryCount - firstQuery)
    {
        return Err(ErrorCode::OutOfRange, "vulkan backend: timestamp read range exceeds the heap");
    }
    const FenceSlot* fence = device.fences.Resolve(completionFence);
    if (fence == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: timestamp read fence is stale or invalid");
    }
    u64 completed = 0;
    const vk::Result counted = device.context.DeviceFunctionsExt().getSemaphoreCounterValue(
        device.context.Device(), fence->semaphore, &completed);
    if (counted == vk::Result::ErrorDeviceLost)
    {
        return Err(ErrorCode::DeviceLost, "vulkan backend: device lost");
    }
    if (counted != vk::Result::Success || completed < completionValue)
    {
        return Err(ErrorCode::NotFound,
                   "vulkan backend: timestamp completion fence has not reached its value");
    }
    const u64* values = static_cast<const u64*>(slot->readback.block.mapped);
    for (u32 index = 0; index < queryCount; ++index)
    {
        out[index] = values[firstQuery + index];
    }
    return Status<Error>::Ok();
}

u64 VkTimestampFrequency(const void* internal, QueueKind kind) noexcept
{
    if (internal == nullptr || static_cast<u32>(kind) >= kQueueKindCount)
    {
        return 0;
    }
    const VulkanRhiDevice& device = Self(internal);
    const VulkanDeviceInfo& info = device.context.Info();
    if (info.timestampValidBits == 0 || info.timestampPeriodNs <= 0.0F || kind == QueueKind::Copy)
    {
        return 0;
    }
    return static_cast<u64>(1.0e9 / static_cast<double>(info.timestampPeriodNs) + 0.5);
}

// ---- Pipeline caches -------------------------------------------------------------

Status<Error> VkCreatePipelineCache(void* internal, const PipelineCacheDesc& desc,
                                    PipelineCacheHandle* out) noexcept
{
    if (internal == nullptr || out == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: pipeline cache creation needs a device and an output");
    }
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    // The driver checks the header of seeded bytes and ignores a blob from another
    // device or driver, which is the "stale cache" case: a valid, empty cache.
    vk::PipelineCacheCreateInfo info{};
    info.initialDataSize = desc.initialData.Size();
    info.pInitialData = desc.initialData.Size() != 0 ? desc.initialData.Data() : nullptr;
    vk::PipelineCache cache = 0;
    const vk::DeviceApiExt& ext = device.context.DeviceFunctionsExt();
    if (ext.createPipelineCache(device.context.Device(), &info, nullptr, &cache) !=
        vk::Result::Success)
    {
        info.initialDataSize = 0;
        info.pInitialData = nullptr;
        if (ext.createPipelineCache(device.context.Device(), &info, nullptr, &cache) !=
            vk::Result::Success)
        {
            return Err(ErrorCode::Internal, "vulkan backend: vkCreatePipelineCache failed");
        }
    }
    u32 index = 0;
    if (!device.pipelineCaches.Acquire(&index))
    {
        ext.destroyPipelineCache(device.context.Device(), cache, nullptr);
        return Err(ErrorCode::Internal, "vulkan backend: pipeline cache table could not grow");
    }
    PipelineCacheSlot& slot = device.pipelineCaches.At(index);
    slot.cache = cache;
    CopyLabel(slot.name, desc.label);
    *out = device.pipelineCaches.HandleOf<PipelineCacheTag>(index);
    return Status<Error>::Ok();
}

Status<Error> VkDestroyPipelineCache(void* internal, PipelineCacheHandle handle) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    PipelineCacheSlot* slot = device.pipelineCaches.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: pipeline cache handle is stale or invalid");
    }
    device.context.DeviceFunctionsExt().destroyPipelineCache(device.context.Device(), slot->cache,
                                                             nullptr);
    device.pipelineCaches.Release(handle.index);
    return Status<Error>::Ok();
}

Status<Error> VkPipelineCacheBytes(const void* internal, PipelineCacheHandle handle,
                                   memory::Allocator& allocator,
                                   memory::DynamicArray<memory::Byte>* out) noexcept
{
    if (internal == nullptr || out == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: pipeline cache bytes need an output");
    }
    const VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    const PipelineCacheSlot* slot = device.pipelineCaches.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: pipeline cache handle is stale or invalid");
    }
    const vk::DeviceApiExt& ext = device.context.DeviceFunctionsExt();
    usize size = 0;
    if (ext.getPipelineCacheData(device.context.Device(), slot->cache, &size, nullptr) !=
        vk::Result::Success)
    {
        return Err(ErrorCode::Internal, "vulkan backend: vkGetPipelineCacheData failed");
    }
    memory::DynamicArray<memory::Byte> bytes(allocator);
    if (size != 0)
    {
        if (!bytes.TryResize(size))
        {
            return Err(ErrorCode::Internal,
                       "vulkan backend: pipeline cache bytes failed to allocate");
        }
        const vk::Result written =
            ext.getPipelineCacheData(device.context.Device(), slot->cache, &size, bytes.Data());
        if (written != vk::Result::Success && written != vk::Result::Incomplete)
        {
            return Err(ErrorCode::Internal, "vulkan backend: vkGetPipelineCacheData failed");
        }
        (void)bytes.TryResize(size);
    }
    *out = static_cast<memory::DynamicArray<memory::Byte>&&>(bytes);
    return Status<Error>::Ok();
}

vk::PipelineCache NativePipelineCache(const VulkanRhiDevice& device,
                                      PipelineCacheHandle handle) noexcept
{
    if (!handle.IsValid())
    {
        return 0;
    }
    const PipelineCacheSlot* slot = device.pipelineCaches.Resolve(handle);
    return slot != nullptr ? slot->cache : 0;
}

// ---- Reserved textures -------------------------------------------------------------

Status<Error> WaitForTileBinds(VulkanRhiDevice& device) noexcept
{
    const vk::DeviceApiExt& ext = device.context.DeviceFunctionsExt();
    for (u32 kind = 0; kind < kQueueKindCount; ++kind)
    {
        if (device.lastBind[kind] == 0)
        {
            continue;
        }
        vk::Semaphore semaphore = 0;
        u64* value = nullptr;
        TimelineOf(device, static_cast<QueueKind>(kind), &semaphore, &value);
        vk::SemaphoreWaitInfo wait{};
        wait.semaphoreCount = 1;
        wait.pSemaphores = &semaphore;
        wait.pValues = &device.lastBind[kind];
        if (ext.waitSemaphores(device.context.Device(), &wait, ~0ULL) != vk::Result::Success)
        {
            return Err(ErrorCode::DeviceLost, "vulkan backend: waiting for tile binds failed");
        }
        device.lastBind[kind] = 0;
    }
    return Status<Error>::Ok();
}

namespace
{

// Binds `count` tiles. `memories[i] == 0` with no aliasing unbinds. Standard tiles
// use image binds, packed ones opaque binds into the mip tail. Ordered on `kind`
// after the work already submitted to it, and before the next submission there.
[[nodiscard]] Status<Error> BindTiles(VulkanRhiDevice& device, QueueKind kind,
                                      const TileMapping* mappings, const vk::DeviceMemory* memories,
                                      const u64* offsets, usize count) noexcept
{
    if (count == 0)
    {
        return Status<Error>::Ok();
    }
    DynamicArray<vk::SparseImageMemoryBind> imageBinds(*device.allocator);
    DynamicArray<vk::SparseMemoryBind> opaqueBinds(*device.allocator);
    DynamicArray<vk::SparseImageMemoryBindInfo> imageInfos(*device.allocator);
    DynamicArray<vk::SparseImageOpaqueMemoryBindInfo> opaqueInfos(*device.allocator);
    if (!imageBinds.TryReserve(count) || !opaqueBinds.TryReserve(count) ||
        !imageInfos.TryReserve(count) || !opaqueInfos.TryReserve(count))
    {
        return Err(ErrorCode::Internal, "vulkan backend: tile bind scratch failed to allocate");
    }
    // Pass one fills the bind arrays; pass two points one info at each run of
    // binds to the same image (the arrays no longer move by then).
    struct Run
    {
        vk::Image image;
        bool opaque;
        usize first;
        usize count;
    };
    DynamicArray<Run> runs(*device.allocator);
    for (usize index = 0; index < count; ++index)
    {
        const TileMapping& mapping = mappings[index];
        const TextureSlot* texture = device.textures.Resolve(mapping.texture);
        if (texture == nullptr)
        {
            return Err(ErrorCode::InvalidArgument, "vulkan backend: a tile's texture is stale");
        }
        const vk::Image image = texture->image.image;
        if (mapping.packed)
        {
            vk::SparseMemoryBind bind{};
            bind.resourceOffset =
                texture->mipTailOffset + static_cast<u64>(mapping.tileX) * kTileSizeBytes;
            bind.size = kTileSizeBytes;
            bind.memory = memories[index];
            bind.memoryOffset = offsets[index];
            if (runs.Size() == 0 || runs.Back().image != image || !runs.Back().opaque)
            {
                (void)runs.TryPushBack(Run{image, true, opaqueBinds.Size(), 0});
            }
            (void)opaqueBinds.TryPushBack(bind);
        }
        else
        {
            const TextureTiling& tiling = texture->tiling;
            const u32 mipWidth = MipSize(texture->desc.width, mapping.mipLevel);
            const u32 mipHeight = MipSize(texture->desc.height, mapping.mipLevel);
            const u32 x = mapping.tileX * tiling.tileWidth;
            const u32 y = mapping.tileY * tiling.tileHeight;
            vk::SparseImageMemoryBind bind{};
            bind.subresource.aspectMask = vk::ImageAspect::kColor;
            bind.subresource.mipLevel = mapping.mipLevel;
            bind.subresource.arrayLayer = 0;
            bind.offset = vk::Offset3D{static_cast<i32>(x), static_cast<i32>(y), 0};
            bind.extent = vk::Extent3D{
                mipWidth - x < tiling.tileWidth ? mipWidth - x : tiling.tileWidth,
                mipHeight - y < tiling.tileHeight ? mipHeight - y : tiling.tileHeight, 1};
            bind.memory = memories[index];
            bind.memoryOffset = offsets[index];
            if (runs.Size() == 0 || runs.Back().image != image || runs.Back().opaque)
            {
                (void)runs.TryPushBack(Run{image, false, imageBinds.Size(), 0});
            }
            (void)imageBinds.TryPushBack(bind);
        }
        runs.Back().count += 1;
    }
    for (usize index = 0; index < runs.Size(); ++index)
    {
        const Run& run = runs[index];
        if (run.opaque)
        {
            (void)opaqueInfos.TryPushBack(vk::SparseImageOpaqueMemoryBindInfo{
                run.image, static_cast<u32>(run.count), opaqueBinds.Data() + run.first});
        }
        else
        {
            (void)imageInfos.TryPushBack(vk::SparseImageMemoryBindInfo{
                run.image, static_cast<u32>(run.count), imageBinds.Data() + run.first});
        }
    }

    vk::Semaphore semaphore = 0;
    u64* value = nullptr;
    TimelineOf(device, kind, &semaphore, &value);
    const u64 waitValue = *value;
    const u64 signalValue = ++(*value);
    vk::TimelineSemaphoreSubmitInfo timeline{};
    timeline.waitSemaphoreValueCount = waitValue != 0 ? 1U : 0U;
    timeline.pWaitSemaphoreValues = &waitValue;
    timeline.signalSemaphoreValueCount = 1;
    timeline.pSignalSemaphoreValues = &signalValue;
    vk::BindSparseInfo info{};
    info.pNext = &timeline;
    info.waitSemaphoreCount = waitValue != 0 ? 1U : 0U;
    info.pWaitSemaphores = &semaphore;
    info.imageBindCount = static_cast<u32>(imageInfos.Size());
    info.pImageBinds = imageInfos.Data();
    info.imageOpaqueBindCount = static_cast<u32>(opaqueInfos.Size());
    info.pImageOpaqueBinds = opaqueInfos.Data();
    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &semaphore;
    const vk::Result result =
        device.context.DeviceFunctionsExt().queueBindSparse(QueueOf(device, kind), 1, &info, 0);
    if (result != vk::Result::Success)
    {
        // The signal never happens; take the value back so later waits stay valid.
        --(*value);
        return Err(result == vk::Result::ErrorDeviceLost ? ErrorCode::DeviceLost
                                                         : ErrorCode::Internal,
                   "vulkan backend: vkQueueBindSparse failed");
    }
    device.pendingBind[static_cast<u32>(kind)] = signalValue;
    device.lastBind[static_cast<u32>(kind)] = signalValue;
    return Status<Error>::Ok();
}

// The shared zeroed tile unmapped tiles alias, created once (sparse aliasing only).
[[nodiscard]] bool EnsureZeroTile(VulkanRhiDevice& device, u32 typeBits) noexcept
{
    if (!device.sparseAliasing)
    {
        return false;
    }
    if (device.sparseZeroTile != 0)
    {
        return (typeBits & (1U << device.sparseZeroTileType)) != 0;
    }
    vk::DeviceMemory memory = 0;
    u32 type = 0;
    if (!device.memory.AllocateRaw(kTileSizeBytes, typeBits, &memory, &type))
    {
        return false;
    }
    // A fresh driver allocation is zeroed by the operating system on every
    // platform this backend runs on (Windows and Linux DRM both clear new pages).
    device.sparseZeroTile = memory;
    device.sparseZeroTileType = type;
    return true;
}

}  // namespace

Status<Error> CreateReservedImage(VulkanRhiDevice& device, const TextureDesc& desc,
                                  const VulkanImageDesc& native, TextureSlot& slot) noexcept
{
    if (!device.capabilities.features.Has(Feature::ReservedTextures))
    {
        return Err(ErrorCode::Unsupported,
                   "vulkan backend: reserved textures need sparse 2D residency with the "
                   "standard block shape");
    }
    if (desc.dimension != TextureDimension::Dim2D || desc.arrayLayers != 1 ||
        desc.sampleCount != 1 || HasUsage(desc.usage, TextureUsage::RenderTarget) ||
        HasUsage(desc.usage, TextureUsage::DepthStencil) || desc.mipLevels > kMaxTiledMips)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: a reserved texture is a plain 2d texture");
    }
    vk::MemoryRequirements requirements{};
    const Status<Error> created =
        device.memory.CreateSparseImage(native, device.sparseAliasing, &slot.image, &requirements);
    if (!created)
    {
        return created;
    }
    const vk::DeviceApiExt& ext = device.context.DeviceFunctionsExt();
    const auto fail = [&](const char* message) noexcept
    {
        device.memory.DestroyImage(slot.image);
        return Err(ErrorCode::Unsupported, message);
    };
    if (requirements.alignment != kTileSizeBytes)
    {
        return fail("vulkan backend: this device's sparse block is not 64 KiB");
    }
    vk::SparseImageMemoryRequirements sparse[4] = {};
    u32 sparseCount = 4;
    ext.getImageSparseMemoryRequirements(device.context.Device(), slot.image.image, &sparseCount,
                                         sparse);
    const vk::SparseImageMemoryRequirements* color = nullptr;
    for (u32 index = 0; index < sparseCount && index < 4; ++index)
    {
        if ((sparse[index].formatProperties.aspectMask & vk::ImageAspect::kColor) != 0)
        {
            color = &sparse[index];
        }
    }
    if (color == nullptr || color->formatProperties.imageGranularity.width == 0 ||
        color->formatProperties.imageGranularity.height == 0)
    {
        return fail("vulkan backend: the driver reports no sparse layout for this format");
    }
    TextureTiling tiling{};
    tiling.tileWidth = color->formatProperties.imageGranularity.width;
    tiling.tileHeight = color->formatProperties.imageGranularity.height;
    tiling.standardMipCount =
        color->imageMipTailFirstLod < desc.mipLevels ? color->imageMipTailFirstLod : desc.mipLevels;
    tiling.packedMipCount = desc.mipLevels - tiling.standardMipCount;
    tiling.packedTileCount =
        tiling.packedMipCount != 0
            ? static_cast<u32>((color->imageMipTailSize + kTileSizeBytes - 1) / kTileSizeBytes)
            : 0U;
    u32 total = tiling.packedTileCount;
    for (u32 mip = 0; mip < tiling.standardMipCount; ++mip)
    {
        const u32 width = MipSize(desc.width, mip);
        const u32 height = MipSize(desc.height, mip);
        tiling.mipTilesX[mip] = (width + tiling.tileWidth - 1) / tiling.tileWidth;
        tiling.mipTilesY[mip] = (height + tiling.tileHeight - 1) / tiling.tileHeight;
        total += tiling.mipTilesX[mip] * tiling.mipTilesY[mip];
    }
    tiling.totalTileCount = total;
    tiling.valid = total != 0;
    if (!tiling.valid)
    {
        return fail("vulkan backend: a reserved texture has no tiles");
    }
    void* tiles =
        device.allocator->Allocate(sizeof(TileMappingInfo) * total, alignof(TileMappingInfo));
    if (tiles == nullptr)
    {
        device.memory.DestroyImage(slot.image);
        return Err(ErrorCode::Internal, "vulkan backend: tile table failed to allocate");
    }
    slot.tiles = static_cast<TileMappingInfo*>(tiles);
    for (u32 index = 0; index < total; ++index)
    {
        new (&slot.tiles[index]) TileMappingInfo{};
    }
    slot.reserved = true;
    slot.tiling = tiling;
    slot.mipTailOffset = color->imageMipTailOffset;
    slot.sparseTypeBits = requirements.memoryTypeBits;
    slot.sizeBytes = 0;
    return Status<Error>::Ok();
}

Status<Error> AliasUnmappedTiles(VulkanRhiDevice& device, TextureHandle handle) noexcept
{
    TextureSlot* texture = device.textures.Resolve(handle);
    if (texture == nullptr || !texture->reserved ||
        !EnsureZeroTile(device, texture->sparseTypeBits))
    {
        return Status<Error>::Ok();  // without aliasing, unmapped tiles stay unbound
    }
    const TextureTiling& tiling = texture->tiling;
    DynamicArray<TileMapping> mappings(*device.allocator);
    DynamicArray<vk::DeviceMemory> memories(*device.allocator);
    DynamicArray<u64> offsets(*device.allocator);
    if (!mappings.TryReserve(tiling.totalTileCount) ||
        !memories.TryReserve(tiling.totalTileCount) || !offsets.TryReserve(tiling.totalTileCount))
    {
        return Err(ErrorCode::Internal, "vulkan backend: tile alias scratch failed to allocate");
    }
    for (u32 mip = 0; mip < tiling.standardMipCount; ++mip)
    {
        for (u32 y = 0; y < tiling.mipTilesY[mip]; ++y)
        {
            for (u32 x = 0; x < tiling.mipTilesX[mip]; ++x)
            {
                TileMapping mapping{};
                mapping.texture = handle;
                mapping.mipLevel = mip;
                mapping.tileX = x;
                mapping.tileY = y;
                (void)mappings.TryPushBack(mapping);
            }
        }
    }
    for (u32 index = 0; index < tiling.packedTileCount; ++index)
    {
        TileMapping mapping{};
        mapping.texture = handle;
        mapping.packed = true;
        mapping.tileX = index;
        (void)mappings.TryPushBack(mapping);
    }
    for (usize index = 0; index < mappings.Size(); ++index)
    {
        (void)memories.TryPushBack(device.sparseZeroTile);
        (void)offsets.TryPushBack(0);
    }
    return BindTiles(device, QueueKind::Graphics, mappings.Data(), memories.Data(), offsets.Data(),
                     mappings.Size());
}

Status<Error> VkCreateTileHeap(void* internal, const TileHeapDesc& desc,
                               TileHeapHandle* out) noexcept
{
    if (internal == nullptr || out == nullptr || desc.tileCount == 0)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: a tile heap needs a device, an output and at least one tile");
    }
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    if (!device.capabilities.features.Has(Feature::ReservedTextures))
    {
        return Err(ErrorCode::Unsupported, "vulkan backend: reserved textures are not supported");
    }
    vk::DeviceMemory memory = 0;
    u32 type = 0;
    // Sparse images on one device share their memory types in practice; the
    // device-local type is checked against each texture's bits at bind time.
    const Status<Error> allocated = device.memory.AllocateRaw(
        static_cast<u64>(desc.tileCount) * kTileSizeBytes, ~0U, &memory, &type);
    if (!allocated)
    {
        return allocated;
    }
    u32 index = 0;
    if (!device.tileHeaps.Acquire(&index))
    {
        device.memory.FreeRaw(memory);
        return Err(ErrorCode::Internal, "vulkan backend: tile heap table could not grow");
    }
    TileHeapSlot& slot = device.tileHeaps.At(index);
    slot.memory = memory;
    slot.tileCount = desc.tileCount;
    slot.memoryType = type;
    CopyLabel(slot.name, desc.label);
    *out = device.tileHeaps.HandleOf<TileHeapTag>(index);
    return Status<Error>::Ok();
}

Status<Error> VkDestroyTileHeap(void* internal, TileHeapHandle handle) noexcept
{
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    TileHeapSlot* slot = device.tileHeaps.Resolve(handle);
    if (slot == nullptr)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: tile heap handle is stale or invalid");
    }
    // Binds that name the heap must have run. Work still reading its tiles is the
    // caller's to have finished (the contract: unmap first).
    const Status<Error> waited = WaitForTileBinds(device);
    if (!waited)
    {
        return waited;
    }
    // Tiles still pointing here now read as unmapped.
    for (usize index = 0; index < device.textures.Size(); ++index)
    {
        TextureSlot& texture = device.textures.At(index);
        if (!texture.alive || !texture.reserved || texture.tiles == nullptr)
        {
            continue;
        }
        for (u32 tile = 0; tile < texture.tiling.totalTileCount; ++tile)
        {
            if (texture.tiles[tile].mapped && texture.tiles[tile].heap == handle)
            {
                texture.tiles[tile] = TileMappingInfo{};
            }
        }
    }
    device.memory.FreeRaw(slot->memory);
    device.tileHeaps.Release(handle.index);
    return Status<Error>::Ok();
}

TextureTiling VkTextureTiling(const void* internal, TextureHandle handle) noexcept
{
    const VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    const TextureSlot* slot = device.textures.Resolve(handle);
    return slot != nullptr && slot->reserved ? slot->tiling : TextureTiling{};
}

Status<Error> VkUpdateTileMappings(void* internal, QueueKind queue,
                                   memory::Span<const TileMapping> mappings) noexcept
{
    if (internal == nullptr || static_cast<u32>(queue) >= kQueueKindCount)
    {
        return Err(ErrorCode::InvalidArgument,
                   "vulkan backend: tile mappings need a device and a queue kind");
    }
    VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    if (!device.capabilities.features.Has(Feature::ReservedTextures))
    {
        return Err(ErrorCode::Unsupported, "vulkan backend: reserved textures are not supported");
    }
    if (!device.context.Info().sparseQueue[static_cast<u32>(queue)])
    {
        return Err(ErrorCode::Unsupported, "vulkan backend: this queue does not take sparse binds");
    }
    // The whole batch is checked before any of it reaches the queue.
    for (usize index = 0; index < mappings.Size(); ++index)
    {
        const TileMapping& mapping = mappings[index];
        const TextureSlot* texture = device.textures.Resolve(mapping.texture);
        const TileHeapSlot* heap =
            mapping.heap.IsValid() ? device.tileHeaps.Resolve(mapping.heap) : nullptr;
        const TextureTiling tiling =
            texture != nullptr && texture->reserved ? texture->tiling : TextureTiling{};
        Error error{};
        if (!validation::ValidateTileMapping(tiling, mapping, heap != nullptr ? heap->tileCount : 0,
                                             heap != nullptr, &error))
        {
            return Status<Error>::Err(error);
        }
        if (heap != nullptr && (texture->sparseTypeBits & (1U << heap->memoryType)) == 0)
        {
            return Err(ErrorCode::InvalidArgument,
                       "vulkan backend: the tile heap's memory type cannot back this texture");
        }
    }
    DynamicArray<vk::DeviceMemory> memories(*device.allocator);
    DynamicArray<u64> offsets(*device.allocator);
    if (!memories.TryReserve(mappings.Size()) || !offsets.TryReserve(mappings.Size()))
    {
        return Err(ErrorCode::Internal, "vulkan backend: tile mapping scratch failed to allocate");
    }
    for (usize index = 0; index < mappings.Size(); ++index)
    {
        const TileMapping& mapping = mappings[index];
        const TileHeapSlot* heap =
            mapping.heap.IsValid() ? device.tileHeaps.Resolve(mapping.heap) : nullptr;
        const TextureSlot* texture = device.textures.Resolve(mapping.texture);
        vk::DeviceMemory memory = 0;
        u64 offset = 0;
        if (heap != nullptr)
        {
            memory = heap->memory;
            offset = static_cast<u64>(mapping.heapTile) * kTileSizeBytes;
        }
        else if (EnsureZeroTile(device, texture->sparseTypeBits))
        {
            memory = device.sparseZeroTile;  // unmapped tiles read as zero
        }
        (void)memories.TryPushBack(memory);
        (void)offsets.TryPushBack(offset);
    }
    const Status<Error> bound =
        BindTiles(device, queue, mappings.Data(), memories.Data(), offsets.Data(), mappings.Size());
    if (!bound)
    {
        return bound;
    }
    for (usize index = 0; index < mappings.Size(); ++index)
    {
        const TileMapping& mapping = mappings[index];
        TextureSlot* texture = device.textures.Resolve(mapping.texture);
        TileMappingInfo& entry =
            texture->tiles[validation::TileLinearIndex(texture->tiling, mapping)];
        entry.mapped = mapping.heap.IsValid();
        entry.heap = mapping.heap.IsValid() ? mapping.heap : TileHeapHandle{};
        entry.heapTile = mapping.heap.IsValid() ? mapping.heapTile : 0;
    }
    return Status<Error>::Ok();
}

TileMappingInfo VkTileMapping(const void* internal, const TileMapping& tile) noexcept
{
    TileMappingInfo info{};
    if (internal == nullptr)
    {
        return info;
    }
    const VulkanRhiDevice& device = Self(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    const TextureSlot* texture = device.textures.Resolve(tile.texture);
    if (texture == nullptr || !texture->reserved || texture->tiles == nullptr)
    {
        return info;
    }
    TileMapping probe = tile;
    probe.heap = TileHeapHandle{};
    if (!validation::ValidateTileMapping(texture->tiling, probe, 0, false, nullptr))
    {
        return info;
    }
    return texture->tiles[validation::TileLinearIndex(texture->tiling, probe)];
}

}  // namespace cyber::rhi::detail

#endif
