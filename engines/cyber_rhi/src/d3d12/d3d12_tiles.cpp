#include "d3d12_internal.h"

#if defined(CYBER_RHI_HAS_D3D12)

#include "../validation.h"

#include <cyber/core/log.h>

#include <new>

namespace cyber::rhi::detail
{
namespace
{

using cyber::core::Error;
using cyber::core::ErrorCode;
using cyber::core::Status;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::usize;
using cyber::memory::kInvalidHandleIndex;

[[nodiscard]] Status<Error> StatusError(ErrorCode code, const char* message) noexcept
{
    return Status<Error>::Err(Error{code, message});
}

[[nodiscard]] u32 NextGeneration(u32 current) noexcept
{
    const u32 next = current + 1;
    return next == 0 ? 1 : next;
}

// Tiles sent to the queue per UpdateTileMappings call. The arrays live on the
// stack; a longer run of one (texture, heap) pair is split into several calls,
// which the queue orders exactly as one.
constexpr usize kTilesPerCall = 64;

}  // namespace

TextureTiling QueryResourceTiling(ID3D12Device* device, ID3D12Resource* resource,
                                  const TextureDesc& desc) noexcept
{
    TextureTiling tiling;
    if (device == nullptr || resource == nullptr || desc.mipLevels > kMaxTiledMips)
    {
        return tiling;
    }
    UINT totalTiles = 0;
    D3D12_PACKED_MIP_INFO packed{};
    D3D12_TILE_SHAPE shape{};
    UINT subresources = desc.mipLevels;
    D3D12_SUBRESOURCE_TILING subresourceTilings[kMaxTiledMips]{};
    device->GetResourceTiling(resource, &totalTiles, &packed, &shape, &subresources, 0,
                              subresourceTilings);
    if (totalTiles == 0 || shape.WidthInTexels == 0 || shape.HeightInTexels == 0)
    {
        return tiling;
    }
    tiling.tileWidth = shape.WidthInTexels;
    tiling.tileHeight = shape.HeightInTexels;
    tiling.standardMipCount = packed.NumStandardMips;
    tiling.packedMipCount = packed.NumPackedMips;
    tiling.packedTileCount = packed.NumTilesForPackedMips;
    tiling.totalTileCount = totalTiles;
    for (u32 mip = 0; mip < tiling.standardMipCount && mip < kMaxTiledMips; ++mip)
    {
        tiling.mipTilesX[mip] = subresourceTilings[mip].WidthInTiles;
        tiling.mipTilesY[mip] = subresourceTilings[mip].HeightInTiles;
    }
    tiling.valid = true;
    return tiling;
}

Status<Error> D3D12CreateTileHeap(void* internal, const TileHeapDesc& desc,
                                  TileHeapHandle* out) noexcept
{
    if (internal == nullptr || out == nullptr || desc.tileCount == 0)
    {
        return StatusError(ErrorCode::InvalidArgument,
                           "d3d12 backend: tile heap needs a device, tiles, and an output");
    }
    auto& device = *static_cast<D3D12Device*>(internal);
    const Status<Error> healthy = DeviceLostStatus(device);
    if (!healthy)
    {
        return healthy;
    }
    const std::lock_guard<std::mutex> lock(device.mutex);
    if (!device.capabilities.features.Has(Feature::ReservedTextures))
    {
        return StatusError(ErrorCode::Unsupported,
                           "d3d12 backend: reserved textures need tiled resources tier 2");
    }

    D3D12_HEAP_DESC heapDesc{};
    heapDesc.SizeInBytes = static_cast<u64>(desc.tileCount) * kTileSizeBytes;
    heapDesc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapDesc.Properties.CreationNodeMask = 1;
    heapDesc.Properties.VisibleNodeMask = 1;
    heapDesc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    // Textures that are not attachments: the only kind a reserved texture is,
    // and the only heap tier-1 heaps may mix tiles into.
    heapDesc.Flags = D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES;
    ID3D12Heap* heap = nullptr;
    if (FAILED(device.device->CreateHeap(&heapDesc, __uuidof(ID3D12Heap),
                                         reinterpret_cast<void**>(&heap))) ||
        heap == nullptr)
    {
        return StatusError(ErrorCode::OutOfRange, "d3d12 backend: tile heap creation failed");
    }
    NameObject(heap, desc.label);

    u32 index = kInvalidHandleIndex;
    if (device.freeTileHeapSlots.Size() > 0)
    {
        index = device.freeTileHeapSlots[device.freeTileHeapSlots.Size() - 1];
        (void)device.freeTileHeapSlots.TryResize(device.freeTileHeapSlots.Size() - 1);
    }
    else
    {
        if (!device.tileHeaps.TryPushBack(D3D12TileHeapSlot{}))
        {
            heap->Release();
            return StatusError(ErrorCode::Internal, "d3d12 backend: tile heap table growth failed");
        }
        index = static_cast<u32>(device.tileHeaps.Size() - 1);
    }
    D3D12TileHeapSlot& slot = device.tileHeaps[index];
    slot.generation = NextGeneration(slot.generation);
    slot.alive = true;
    slot.heap = heap;
    slot.desc = desc;
    TileHeapHandle handle;
    handle.index = index;
    handle.generation = slot.generation;
    *out = handle;
    return Status<Error>::Ok();
}

Status<Error> D3D12DestroyTileHeap(void* internal, TileHeapHandle heap) noexcept
{
    if (internal == nullptr)
    {
        return StatusError(ErrorCode::InvalidArgument, "d3d12 backend: device is null");
    }
    auto& device = *static_cast<D3D12Device*>(internal);
    const std::lock_guard<std::mutex> lock(device.mutex);
    D3D12TileHeapSlot* slot = Resolve(device, heap);
    if (slot == nullptr)
    {
        return StatusError(ErrorCode::InvalidArgument,
                           "d3d12 backend: tile heap handle is stale or invalid");
    }
    // A mapping update still queued names this heap.
    D3D12WaitForTileMappings(device);
    slot->heap->Release();
    slot->heap = nullptr;
    slot->alive = false;
    slot->generation = NextGeneration(slot->generation);
    if (!device.freeTileHeapSlots.TryPushBack(heap.index))
    {
        return StatusError(ErrorCode::Internal, "d3d12 backend: tile heap free list growth failed");
    }
    return Status<Error>::Ok();
}

TextureTiling D3D12TextureTiling(const void* internal, TextureHandle texture) noexcept
{
    if (internal == nullptr)
    {
        return TextureTiling{};
    }
    auto& device = *static_cast<D3D12Device*>(const_cast<void*>(internal));
    const std::lock_guard<std::mutex> lock(device.mutex);
    const D3D12TextureSlot* slot = Resolve(device, texture);
    return slot != nullptr && slot->desc.reserved ? slot->tiling : TextureTiling{};
}

Status<Error> D3D12UpdateTileMappings(void* internal, QueueKind queue,
                                      memory::Span<const TileMapping> mappings) noexcept
{
    if (internal == nullptr || static_cast<u32>(queue) >= kQueueKindCount)
    {
        return StatusError(ErrorCode::InvalidArgument,
                           "d3d12 backend: tile mappings need a device and a queue kind");
    }
    auto& device = *static_cast<D3D12Device*>(internal);
    const Status<Error> healthy = DeviceLostStatus(device);
    if (!healthy)
    {
        return healthy;
    }
    const std::lock_guard<std::mutex> lock(device.mutex);
    ID3D12CommandQueue* commandQueue = device.queues[static_cast<u32>(queue)].queue;
    if (commandQueue == nullptr)
    {
        return StatusError(ErrorCode::Unsupported, "d3d12 backend: the device has no such queue");
    }

    // The whole batch is checked before any of it reaches the queue.
    for (usize index = 0; index < mappings.Size(); ++index)
    {
        const TileMapping& mapping = mappings[index];
        const D3D12TextureSlot* texture = Resolve(device, mapping.texture);
        const D3D12TileHeapSlot* heap =
            mapping.heap.IsValid() ? Resolve(device, mapping.heap) : nullptr;
        const TextureTiling tiling =
            texture != nullptr && texture->desc.reserved ? texture->tiling : TextureTiling{};
        Error error{};
        if (!validation::ValidateTileMapping(tiling, mapping,
                                             heap != nullptr ? heap->desc.tileCount : 0,
                                             heap != nullptr, &error))
        {
            return Status<Error>::Err(error);
        }
    }

    // Runs of one (texture, heap) pair become one call each; an invalid heap
    // is an unmap, sent with the NULL range flag and no heap.
    usize begin = 0;
    while (begin < mappings.Size())
    {
        const TileMapping& first = mappings[begin];
        usize end = begin + 1;
        while (end < mappings.Size() && end - begin < kTilesPerCall &&
               mappings[end].texture == first.texture && mappings[end].heap == first.heap)
        {
            ++end;
        }
        D3D12TextureSlot* texture = Resolve(device, first.texture);
        D3D12TileHeapSlot* heap = first.heap.IsValid() ? Resolve(device, first.heap) : nullptr;
        const UINT count = static_cast<UINT>(end - begin);

        D3D12_TILED_RESOURCE_COORDINATE coordinates[kTilesPerCall]{};
        D3D12_TILE_REGION_SIZE sizes[kTilesPerCall]{};
        D3D12_TILE_RANGE_FLAGS flags[kTilesPerCall]{};
        UINT heapOffsets[kTilesPerCall]{};
        UINT rangeCounts[kTilesPerCall]{};
        for (UINT i = 0; i < count; ++i)
        {
            const TileMapping& mapping = mappings[begin + i];
            coordinates[i].X = mapping.tileX;
            coordinates[i].Y = mapping.packed ? 0 : mapping.tileY;
            coordinates[i].Z = 0;
            coordinates[i].Subresource =
                mapping.packed ? texture->tiling.standardMipCount : mapping.mipLevel;
            sizes[i].NumTiles = 1;
            sizes[i].UseBox = FALSE;
            flags[i] = heap != nullptr ? D3D12_TILE_RANGE_FLAG_NONE : D3D12_TILE_RANGE_FLAG_NULL;
            heapOffsets[i] = heap != nullptr ? mapping.heapTile : 0;
            rangeCounts[i] = 1;
        }
        commandQueue->UpdateTileMappings(texture->resource, count, coordinates, sizes,
                                         heap != nullptr ? heap->heap : nullptr, count, flags,
                                         heapOffsets, rangeCounts, D3D12_TILE_MAPPING_FLAG_NONE);
        // Marks the update in the queue's order, so a later release of the
        // texture or the heap can wait until the queue is done with both.
        D3D12Queue& tileQueue = device.queues[static_cast<u32>(queue)];
        if (tileQueue.tileFence == nullptr)
        {
            (void)device.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence),
                                             reinterpret_cast<void**>(&tileQueue.tileFence));
        }
        if (tileQueue.tileFence != nullptr &&
            SUCCEEDED(commandQueue->Signal(tileQueue.tileFence, tileQueue.tileSignalled + 1)))
        {
            ++tileQueue.tileSignalled;
        }
        for (UINT i = 0; i < count; ++i)
        {
            const TileMapping& mapping = mappings[begin + i];
            D3D12TileEntry& entry =
                texture->tiles[validation::TileLinearIndex(texture->tiling, mapping)];
            entry.heap = heap != nullptr ? mapping.heap : TileHeapHandle{};
            entry.tile = heap != nullptr ? mapping.heapTile : 0;
        }
        begin = end;
    }
    return Status<Error>::Ok();
}

TileMappingInfo D3D12TileMapping(const void* internal, const TileMapping& tile) noexcept
{
    TileMappingInfo info;
    if (internal == nullptr)
    {
        return info;
    }
    auto& device = *static_cast<D3D12Device*>(const_cast<void*>(internal));
    const std::lock_guard<std::mutex> lock(device.mutex);
    const D3D12TextureSlot* texture = Resolve(device, tile.texture);
    if (texture == nullptr || !texture->desc.reserved || texture->tiles == nullptr)
    {
        return info;
    }
    TileMapping probe = tile;
    probe.heap = TileHeapHandle{};
    if (!validation::ValidateTileMapping(texture->tiling, probe, 0, false, nullptr))
    {
        return info;
    }
    const D3D12TileEntry& entry =
        texture->tiles[validation::TileLinearIndex(texture->tiling, probe)];
    if (entry.heap.IsValid() && Resolve(device, entry.heap) != nullptr)
    {
        info.mapped = true;
        info.heap = entry.heap;
        info.heapTile = entry.tile;
    }
    return info;
}

void D3D12WaitForTileMappings(D3D12Device& device) noexcept
{
    for (u32 index = 0; index < kQueueKindCount; ++index)
    {
        D3D12Queue& entry = device.queues[index];
        if (entry.tileFence == nullptr || entry.tileSignalled == 0 ||
            entry.tileFence->GetCompletedValue() >= entry.tileSignalled)
        {
            continue;
        }
        HANDLE event = CreateEventExW(nullptr, nullptr, 0, EVENT_MODIFY_STATE | SYNCHRONIZE);
        if (event == nullptr)
        {
            continue;
        }
        if (SUCCEEDED(entry.tileFence->SetEventOnCompletion(entry.tileSignalled, event)))
        {
            (void)WaitForSingleObject(event, INFINITE);
        }
        (void)CloseHandle(event);
    }
}

}  // namespace cyber::rhi::detail

#endif
