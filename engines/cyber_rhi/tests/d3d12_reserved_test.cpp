// Reserved textures on real Direct3D 12 adapters (Phase 33): the driver's
// tiling agrees with the standard shapes, data written through mapped tiles
// reads back exactly, and an unmapped tile reads zero -- the tier-2 guarantee
// the texture streamer is built on. Runs on every local adapter that reports
// Feature::ReservedTextures (the HD Graphics 530 and WARP here).

#include <cyber/core/types.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/rhi/backend.h>
#include <cyber/rhi/command_list.h>
#include <cyber/rhi/d3d12.h>
#include <cyber/rhi/device.h>
#include <cyber/test/test.h>

#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

using namespace cyber::rhi;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::u8;
using cyber::core::usize;
using cyber::memory::Byte;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;

namespace
{

constexpr u32 kSize = 256;     // mip 0: 2x2 tiles of 128x128 RGBA8
constexpr u32 kPackedMip = 2;  // 64x64 and below are packed

u32 Texel(u32 x, u32 y, u32 mip) noexcept
{
    return (x & 0xFFu) | ((y & 0xFFu) << 8u) | ((mip * 40u + 7u) << 16u) | 0xFF000000u;
}

bool RunList(Device& device, CommandList& list, FenceHandle fence, u64& value)
{
    if (!list.End().IsOk())
    {
        return false;
    }
    const CommandListHandle handle = list.Handle();
    const FenceSignal signal{fence, ++value};
    SubmitDesc submit{};
    submit.queue = QueueKind::Graphics;
    submit.commandLists = Span<const CommandListHandle>(&handle, 1);
    submit.signals = Span<const FenceSignal>(&signal, 1);
    return device.Submit(submit).IsOk() && device.WaitFence(fence, value).IsOk();
}

bool NewList(Device& device, CommandList& out)
{
    CommandListDesc desc{};
    desc.queue = QueueKind::Graphics;
    auto created = device.CreateCommandList(desc);
    if (!created)
    {
        return false;
    }
    out = std::move(created.Value());
    return out.Begin().IsOk();
}

// Copies one mip out to a readback buffer and returns its texels.
bool ReadMip(Device& device, FenceHandle fence, u64& value, TextureHandle texture, u32 mip,
             BufferHandle readback, std::vector<u32>& out)
{
    const u32 size = kSize >> mip;
    CommandList list;
    if (!NewList(device, list))
    {
        return false;
    }
    BufferTextureCopy copy{};
    copy.buffer = readback;
    copy.bufferRowPitch = size * 4u;
    copy.texture = texture;
    copy.mipLevel = mip;
    copy.width = size;
    copy.height = size;
    copy.depth = 1;
    if (!list.CopyTextureToBuffer(copy).IsOk() || !RunList(device, list, fence, value))
    {
        return false;
    }
    out.assign(static_cast<size_t>(size) * size, 0u);
    return device
        .ReadBuffer(readback, 0,
                    Span<Byte>(reinterpret_cast<Byte*>(out.data()), out.size() * sizeof(u32)))
        .IsOk();
}

}  // namespace

CG_TEST(d3d12_reserved_textures_round_trip_through_mapped_tiles)
{
    const Backend* backend = D3D12Backend();
    if (backend == nullptr)
    {
        CG_SKIP("no Direct3D 12 runtime");
    }
    AdapterInfo adapters[8]{};
    const u32 adapterCount = backend->enumerateAdapters(adapters, 8);
    u32 tested = 0;
    for (u32 adapterIndex = 0; adapterIndex < adapterCount; ++adapterIndex)
    {
        HeapAllocator heap;
        DeviceDesc desc{};
        desc.backend = backend;
        desc.adapterIndex = adapterIndex;
        auto createdDevice = Device::Create(desc, heap);
        if (!createdDevice)
        {
            continue;
        }
        Device device = std::move(createdDevice.Value());
        if (!device.Capabilities().features.Has(Feature::ReservedTextures))
        {
            std::printf("adapter \"%s\": no tiled resources tier 2, skipped\n",
                        adapters[adapterIndex].name);
            continue;
        }
        ++tested;

        TextureDesc textureDesc{};
        textureDesc.format = Format::RGBA8Unorm;
        textureDesc.usage =
            TextureUsage::Sampled | TextureUsage::CopyDestination | TextureUsage::CopySource;
        textureDesc.width = kSize;
        textureDesc.height = kSize;
        textureDesc.mipLevels = 9;
        textureDesc.reserved = true;
        textureDesc.label = "reserved_test";
        auto createdTexture = device.CreateTexture(textureDesc);
        CG_REQUIRE(createdTexture.IsOk());
        const TextureHandle texture = createdTexture.Value();

        // The driver's tiling against the standard shapes the streamer plans
        // with. The packed tile count is the driver's to choose.
        const TextureTiling driver = device.TextureTilingOf(texture);
        const TextureTiling standard = ComputeStandardTiling(textureDesc);
        CG_REQUIRE(driver.valid);
        std::printf("adapter \"%s\": tile %ux%u, %u standard mips, %u packed in %u tiles "
                    "(standard estimate %u)\n",
                    adapters[adapterIndex].name, driver.tileWidth, driver.tileHeight,
                    driver.standardMipCount, driver.packedMipCount, driver.packedTileCount,
                    standard.packedTileCount);
        // The tile shape and the grids of mips that cover whole tiles are fixed
        // by the standard swizzle. Whether mips smaller than a tile are packed
        // is the driver's choice: the HD Graphics 530 packs them, WARP gives
        // each its own partial tile. The standard tiling assumes packing, so
        // every mip it calls standard must be standard on the driver too.
        CG_EXPECT_EQ(driver.tileWidth, standard.tileWidth);
        CG_EXPECT_EQ(driver.tileHeight, standard.tileHeight);
        CG_EXPECT_TRUE(driver.standardMipCount >= standard.standardMipCount);
        for (u32 mip = 0; mip < standard.standardMipCount; ++mip)
        {
            CG_EXPECT_EQ(driver.mipTilesX[mip], standard.mipTilesX[mip]);
            CG_EXPECT_EQ(driver.mipTilesY[mip], standard.mipTilesY[mip]);
        }
        CG_REQUIRE(driver.standardMipCount >= kPackedMip);
        // Mip kPackedMip (64x64) is packed on one driver and a one-tile
        // standard mip on the other; the test maps whichever it is.
        const bool smallMipPacked = driver.standardMipCount == kPackedMip;
        const u32 smallTiles = smallMipPacked ? driver.packedTileCount : 1u;

        const u32 heapTiles = 4 + smallTiles;
        auto createdHeap = device.CreateTileHeap(TileHeapDesc{heapTiles, "reserved_test_heap"});
        CG_REQUIRE(createdHeap.IsOk());
        const TileHeapHandle tileHeap = createdHeap.Value();

        // Mip 0's four tiles onto heap tiles 3, 2, 1, 0 (deliberately not in
        // order), the small mip after them. Mip 1 stays unmapped.
        std::vector<TileMapping> mappings;
        for (u32 tile = 0; tile < 4; ++tile)
        {
            TileMapping mapping;
            mapping.texture = texture;
            mapping.mipLevel = 0;
            mapping.tileX = tile % 2;
            mapping.tileY = tile / 2;
            mapping.heap = tileHeap;
            mapping.heapTile = 3 - tile;
            mappings.push_back(mapping);
        }
        for (u32 tile = 0; tile < smallTiles; ++tile)
        {
            TileMapping mapping;
            mapping.texture = texture;
            mapping.packed = smallMipPacked;
            mapping.mipLevel = smallMipPacked ? 0u : kPackedMip;
            mapping.tileX = smallMipPacked ? tile : 0u;
            mapping.heap = tileHeap;
            mapping.heapTile = 4 + tile;
            mappings.push_back(mapping);
        }
        CG_REQUIRE(
            device
                .UpdateTileMappings(QueueKind::Graphics,
                                    Span<const TileMapping>(mappings.data(), mappings.size()))
                .IsOk());

        // Upload mip 0 and the first packed mip.
        const u32 packedSize = kSize >> kPackedMip;
        const u64 mip0Bytes = static_cast<u64>(kSize) * kSize * 4u;
        const u64 packedBytes = static_cast<u64>(packedSize) * packedSize * 4u;
        std::vector<u32> upload(static_cast<size_t>((mip0Bytes + packedBytes) / 4u));
        for (u32 y = 0; y < kSize; ++y)
        {
            for (u32 x = 0; x < kSize; ++x)
            {
                upload[y * kSize + x] = Texel(x, y, 0);
            }
        }
        for (u32 y = 0; y < packedSize; ++y)
        {
            for (u32 x = 0; x < packedSize; ++x)
            {
                upload[static_cast<size_t>(mip0Bytes / 4u) + y * packedSize + x] =
                    Texel(x, y, kPackedMip);
            }
        }
        BufferDesc uploadDesc{};
        uploadDesc.usage = BufferUsage::HostVisible | BufferUsage::CopySource;
        uploadDesc.sizeBytes = mip0Bytes + packedBytes;
        const BufferHandle uploadBuffer = device.CreateBuffer(uploadDesc).Value();
        CG_REQUIRE(device
                       .WriteBuffer(uploadBuffer, 0,
                                    Span<const Byte>(reinterpret_cast<const Byte*>(upload.data()),
                                                     upload.size() * sizeof(u32)))
                       .IsOk());
        BufferDesc readbackDesc{};
        readbackDesc.usage = BufferUsage::Readback | BufferUsage::CopyDestination;
        readbackDesc.sizeBytes = mip0Bytes;
        const BufferHandle readback = device.CreateBuffer(readbackDesc).Value();
        FenceDesc fenceDesc{};
        const FenceHandle fence = device.CreateFence(fenceDesc).Value();
        u64 fenceValue = 0;

        CommandList list;
        CG_REQUIRE(NewList(device, list));
        const TextureBarrier toCopy[] = {TextureBarrier{
            texture, SubresourceRange{}, ResourceState::Undefined, ResourceState::CopyDestination}};
        const BufferBarrier buffers[] = {
            {uploadBuffer, ResourceState::Undefined, ResourceState::CopySource},
            {readback, device.BufferStateOf(readback), ResourceState::CopyDestination}};
        const usize bufferCount =
            device.BufferStateOf(readback) == ResourceState::CopyDestination ? 1u : 2u;
        CG_REQUIRE(list.Barrier(Span<const BufferBarrier>(buffers, bufferCount),
                                Span<const TextureBarrier>(toCopy, 1))
                       .IsOk());
        BufferTextureCopy mip0{};
        mip0.buffer = uploadBuffer;
        mip0.bufferRowPitch = kSize * 4u;
        mip0.texture = texture;
        mip0.width = kSize;
        mip0.height = kSize;
        mip0.depth = 1;
        CG_REQUIRE(list.CopyBufferToTexture(mip0).IsOk());
        BufferTextureCopy packedCopy = mip0;
        packedCopy.bufferOffset = mip0Bytes;
        packedCopy.bufferRowPitch = packedSize * 4u;
        packedCopy.mipLevel = kPackedMip;
        packedCopy.width = packedSize;
        packedCopy.height = packedSize;
        CG_REQUIRE(list.CopyBufferToTexture(packedCopy).IsOk());
        const TextureBarrier toRead[] = {TextureBarrier{texture, SubresourceRange{},
                                                        ResourceState::CopyDestination,
                                                        ResourceState::CopySource}};
        CG_REQUIRE(list.Barrier(Span<const BufferBarrier>{}, Span<const TextureBarrier>(toRead, 1))
                       .IsOk());
        CG_REQUIRE(RunList(device, list, fence, fenceValue));

        std::vector<u32> read;
        CG_REQUIRE(ReadMip(device, fence, fenceValue, texture, 0, readback, read));
        u32 wrong = 0;
        for (u32 i = 0; i < kSize * kSize; ++i)
        {
            wrong += read[i] != upload[i] ? 1u : 0u;
        }
        CG_EXPECT_EQ(wrong, 0u);
        CG_REQUIRE(ReadMip(device, fence, fenceValue, texture, kPackedMip, readback, read));
        wrong = 0;
        for (u32 i = 0; i < packedSize * packedSize; ++i)
        {
            wrong += read[i] != upload[static_cast<size_t>(mip0Bytes / 4u) + i] ? 1u : 0u;
        }
        CG_EXPECT_EQ(wrong, 0u);

        // Unmap tile (1, 0) of mip 0: its 128x128 texels read zero, the other
        // three tiles are untouched. Ordered on the queue after the copies.
        TileMapping unmap = mappings[1];
        unmap.heap = TileHeapHandle{};
        CG_REQUIRE(
            device.UpdateTileMappings(QueueKind::Graphics, Span<const TileMapping>(&unmap, 1))
                .IsOk());
        CG_EXPECT_FALSE(device.TileMappingOf(unmap).mapped);
        CG_EXPECT_TRUE(device.TileMappingOf(mappings[0]).mapped);
        CG_REQUIRE(ReadMip(device, fence, fenceValue, texture, 0, readback, read));
        u32 notZero = 0;
        wrong = 0;
        for (u32 y = 0; y < kSize; ++y)
        {
            for (u32 x = 0; x < kSize; ++x)
            {
                const bool inUnmapped = x >= driver.tileWidth && y < driver.tileHeight;
                if (inUnmapped)
                {
                    notZero += read[y * kSize + x] != 0u ? 1u : 0u;
                }
                else
                {
                    wrong += read[y * kSize + x] != upload[y * kSize + x] ? 1u : 0u;
                }
            }
        }
        CG_EXPECT_EQ(notZero, 0u);
        CG_EXPECT_EQ(wrong, 0u);

        // Teardown waits for the GPU, as any release of memory the queue may
        // still touch must; tile heaps in particular cannot die under a mapping
        // an executing command list reads.
        CG_REQUIRE(device.WaitIdle().IsOk());
        CG_REQUIRE(device.DestroyBuffer(uploadBuffer).IsOk());
        CG_REQUIRE(device.DestroyBuffer(readback).IsOk());
        CG_REQUIRE(device.DestroyTexture(texture).IsOk());
        CG_REQUIRE(device.DestroyTileHeap(tileHeap).IsOk());
        // The fence goes last: the validating device proves a resource idle by
        // the fence values it was submitted under.
        CG_REQUIRE(device.DestroyFence(fence).IsOk());
        CG_EXPECT_EQ(device.ReportLeaks(), 0u);
    }
    if (tested == 0)
    {
        CG_SKIP("no adapter supports tiled resources tier 2");
    }
}

// A reserved texture, and the tile heap behind it, released straight after a
// mapping update with nothing waited on in between. The update is a queue
// operation that still names both, so the backend has to wait for it before it
// releases either; without that the debug layer ended the process.
CG_TEST(d3d12_reserved_texture_released_right_after_mapping_does_not_race_the_queue)
{
    const Backend* backend = D3D12Backend();
    if (backend == nullptr)
    {
        CG_SKIP("no Direct3D 12 runtime");
    }
    AdapterInfo adapters[8]{};
    const u32 adapterCount = backend->enumerateAdapters(adapters, 8);
    u32 tested = 0;
    for (u32 adapterIndex = 0; adapterIndex < adapterCount; ++adapterIndex)
    {
        HeapAllocator heap;
        DeviceDesc desc{};
        desc.backend = backend;
        desc.adapterIndex = adapterIndex;
        desc.validation = ValidationLevel::Standard;
        auto createdDevice = Device::Create(desc, heap);
        if (!createdDevice ||
            !createdDevice.Value().Capabilities().features.Has(Feature::ReservedTextures))
        {
            continue;
        }
        Device device = std::move(createdDevice.Value());
        ++tested;

        for (u32 round = 0; round < 8; ++round)
        {
            TextureDesc textureDesc{};
            textureDesc.format = Format::RGBA8Unorm;
            textureDesc.usage = TextureUsage::Sampled | TextureUsage::CopyDestination;
            textureDesc.width = kSize;
            textureDesc.height = kSize;
            textureDesc.mipLevels = 9;
            textureDesc.reserved = true;
            textureDesc.label = "reserved_race_test";
            auto texture = device.CreateTexture(textureDesc);
            CG_REQUIRE(texture.IsOk());
            auto tileHeap = device.CreateTileHeap(TileHeapDesc{4, "reserved_race_heap"});
            CG_REQUIRE(tileHeap.IsOk());

            TileMapping mappings[4];
            for (u32 tile = 0; tile < 4; ++tile)
            {
                mappings[tile].texture = texture.Value();
                mappings[tile].mipLevel = 0;
                mappings[tile].tileX = tile % 2;
                mappings[tile].tileY = tile / 2;
                mappings[tile].heap = tileHeap.Value();
                mappings[tile].heapTile = tile;
            }
            CG_REQUIRE(
                device.UpdateTileMappings(QueueKind::Graphics, Span<const TileMapping>(mappings, 4))
                    .IsOk());
            // Remap half of them to nothing, so both kinds of update are queued.
            TileMapping unmap[2] = {mappings[0], mappings[1]};
            unmap[0].heap = TileHeapHandle{};
            unmap[1].heap = TileHeapHandle{};
            CG_REQUIRE(
                device.UpdateTileMappings(QueueKind::Graphics, Span<const TileMapping>(unmap, 2))
                    .IsOk());
            CG_REQUIRE(device.DestroyTexture(texture.Value()).IsOk());
            CG_REQUIRE(device.DestroyTileHeap(tileHeap.Value()).IsOk());
        }
        CG_REQUIRE(device.WaitIdle().IsOk());
    }
    if (tested == 0)
    {
        CG_SKIP("no Direct3D 12 adapter with reserved textures");
    }
}
