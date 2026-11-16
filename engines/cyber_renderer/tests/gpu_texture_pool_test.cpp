// The GPU texture pool (Phase 33): the streamer's pages become tile mappings
// of reserved RHI textures in one tile heap.
//
// On the validation backend the tests check the mapping invariants after
// every frame of an oversubscribed run: each resident level is fully mapped,
// each non-resident standard level fully unmapped, no heap tile is mapped
// twice, and the number of mapped tiles equals the streamer's used pages. On
// every Direct3D 12 adapter with tiled resources the data is read back from
// the GPU and compared byte for byte, before and after evictions.

#include "test_backend.h"

#include <cyber/memory/heap_allocator.h>
#include <cyber/renderer/gpu_texture_pool.h>
#include <cyber/renderer/texture_streaming.h>
#include <cyber/rhi/backend.h>
#include <cyber/rhi/d3d12.h>
#include <cyber/rhi/device.h>
#include <cyber/test/test.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using cyber::core::Byte;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::usize;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using namespace cyber::renderer;
namespace rhi = cyber::rhi;

// A procedural texture in RGBA8 or BC1 whose every byte is a function of
// (texture, level, offset), so a readback can be checked exactly. BC1 blocks
// are arbitrary bytes, which the copy path moves unchanged like any other.
class PatternSource final : public TextureMipSource
{
public:
    PatternSource(u32 id, u32 size, rhi::Format format) noexcept
        : m_id(id), m_size(size), m_format(format)
    {
        for (u32 s = size; s >= 1; s /= 2)
        {
            ++m_levels;
            if (s == 1)
            {
                break;
            }
        }
    }
    u32 LevelCount() const noexcept override
    {
        return m_levels;
    }
    StreamingLevelInfo Level(u32 level) const noexcept override
    {
        const u32 size = std::max(m_size >> level, 1u);
        const u32 blocks = m_format == rhi::Format::BC1Unorm ? (size + 3u) / 4u : size;
        const u64 bytes =
            static_cast<u64>(blocks) * blocks * (m_format == rhi::Format::BC1Unorm ? 8u : 4u);
        return StreamingLevelInfo{size, size, bytes};
    }
    bool ReadLevel(u32 level, Span<Byte> destination) noexcept override
    {
        for (usize i = 0; i < destination.Size(); ++i)
        {
            destination[i] = Expected(level, i);
        }
        return true;
    }
    [[nodiscard]] Byte Expected(u32 level, usize offset) const noexcept
    {
        return static_cast<Byte>((offset * 131u + level * 29u + m_id * 7u + (offset >> 9u)) &
                                 0xFFu);
    }
    [[nodiscard]] rhi::TextureDesc Desc() const noexcept
    {
        rhi::TextureDesc desc{};
        desc.format = m_format;
        desc.usage = rhi::TextureUsage::Sampled | rhi::TextureUsage::CopySource;
        desc.width = m_size;
        desc.height = m_size;
        desc.mipLevels = m_levels;
        return desc;
    }

private:
    u32 m_id = 0;
    u32 m_size = 0;
    rhi::Format m_format = rhi::Format::RGBA8Unorm;
    u32 m_levels = 0;
};

TextureStreamerDesc StreamerDesc(u32 tiles)
{
    TextureStreamerDesc desc;
    desc.budgetBytes = static_cast<u64>(tiles) * rhi::kTileSizeBytes;
    desc.pageBytes = static_cast<u32>(rhi::kTileSizeBytes);
    desc.ioThreads = 2;
    desc.maxInFlight = 8;
    desc.maxUploadBytesPerFrame = 2ull << 20u;
    desc.unusedFrameLimit = 4;
    return desc;
}

// Checks every tile of every texture against the streamer's residency.
// Returns the number of violations; `mappedOut` gets the mapped tile count.
u32 CheckMappings(const rhi::Device& device, const GpuTexturePool& pool,
                  const TextureStreamer& streamer, const std::vector<TextureHandle>& handles,
                  usize& mappedOut)
{
    u32 violations = 0;
    std::set<u32> heapTiles;
    mappedOut = 0;
    for (const TextureHandle handle : handles)
    {
        const rhi::TextureHandle texture = pool.Texture(handle);
        const rhi::TextureTiling tiling = device.TextureTilingOf(texture);
        const u32 resident = streamer.ResidentLevel(handle);
        for (u32 mip = 0; mip < tiling.standardMipCount; ++mip)
        {
            for (u32 tile = 0; tile < tiling.MipTileCount(mip); ++tile)
            {
                rhi::TileMapping probe;
                probe.texture = texture;
                probe.mipLevel = mip;
                probe.tileX = tile % tiling.mipTilesX[mip];
                probe.tileY = tile / tiling.mipTilesX[mip];
                const rhi::TileMappingInfo where = device.TileMappingOf(probe);
                const bool shouldBe = mip >= resident;
                if (where.mapped != shouldBe)
                {
                    ++violations;
                }
                if (where.mapped)
                {
                    ++mappedOut;
                    violations += heapTiles.insert(where.heapTile).second ? 0u : 1u;
                }
            }
        }
        for (u32 tile = 0; tile < tiling.packedTileCount; ++tile)
        {
            rhi::TileMapping probe;
            probe.texture = texture;
            probe.packed = true;
            probe.tileX = tile;
            const rhi::TileMappingInfo where = device.TileMappingOf(probe);
            if (where.mapped != (resident < tiling.standardMipCount + tiling.packedMipCount))
            {
                ++violations;
            }
            if (where.mapped)
            {
                ++mappedOut;
                violations += heapTiles.insert(where.heapTile).second ? 0u : 1u;
            }
        }
    }
    return violations;
}

void RunFrame(TextureStreamer& streamer, GpuTexturePool& pool,
              const std::vector<TextureHandle>& handles, const std::vector<float>& wanted,
              const std::vector<float>& coverage)
{
    const auto begun = pool.BeginFrame();
    if (!begun)
    {
        std::printf("begin: %s\n", begun.Error().message);
    }
    for (usize i = 0; i < handles.size(); ++i)
    {
        streamer.ReportUsage(handles[i], wanted[i], coverage[i]);
    }
    (void)streamer.Update(pool);
    const auto ended = pool.EndFrame();
    if (!ended)
    {
        std::printf("end: %s\n", ended.Error().message);
    }
}

}  // namespace

CG_TEST(gpu_texture_pool_keeps_tile_mappings_equal_to_residency)
{
    HeapAllocator allocator;
    rhi::DeviceDesc deviceDesc{};
    deviceDesc.backend = rhi::NullBackend();
    auto createdDevice = rhi::Device::Create(deviceDesc, allocator);
    CG_REQUIRE(createdDevice.IsOk());
    rhi::Device device = std::move(createdDevice.Value());

    // Six textures, RGBA8 and BC1, about 4.7 MiB of full chains against a
    // 40-tile (2.5 MiB) pool: eviction is guaranteed.
    std::vector<std::unique_ptr<PatternSource>> sources;
    for (u32 i = 0; i < 6; ++i)
    {
        sources.push_back(std::make_unique<PatternSource>(i, i % 2 == 0 ? 512u : 1024u,
                                                          i % 2 == 0 ? rhi::Format::RGBA8Unorm
                                                                     : rhi::Format::BC1Unorm));
    }
    TextureStreamer streamer;
    CG_REQUIRE(streamer.Initialize(StreamerDesc(40), allocator).IsOk());
    GpuTexturePool pool;
    GpuTexturePoolDesc poolDesc;
    poolDesc.tileCount = 40;
    CG_REQUIRE(pool.Initialize(device, poolDesc, allocator).IsOk());
    std::vector<TextureHandle> handles;
    for (auto& source : sources)
    {
        auto added = pool.AddTexture(streamer, *source, source->Desc(), "pattern");
        CG_REQUIRE(added.IsOk());
        handles.push_back(added.Value());
    }

    u32 violations = 0;
    usize mapped = 0;
    // Phase one: all want full resolution with falling importance; phase
    // two: the order reverses, forcing the first textures out.
    for (u32 frame = 0; frame < 300; ++frame)
    {
        const bool reversed = frame >= 150;
        std::vector<float> coverage;
        for (u32 i = 0; i < 6; ++i)
        {
            coverage.push_back(0.05f * static_cast<float>(reversed ? i + 1 : 6 - i));
        }
        RunFrame(streamer, pool, handles, std::vector<float>(6, 0.0f), coverage);
        violations += CheckMappings(device, pool, streamer, handles, mapped);
        // Mapped tiles are exactly the resident pages; pages reserved for
        // reads in flight are not mapped until their upload.
        const StreamingStats& stats = streamer.Stats();
        violations += mapped == stats.usedPages - stats.reservedPages ? 0u : 1u;
        // Real IO threads feed the streamer; pacing the frames lets them keep up on
        // a loaded machine instead of the run finishing before the reads land.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // The reversed phase ends with the last texture wanted most; give the IO
    // threads time to finish it rather than asserting on a fixed frame count.
    for (u32 extra = 0; extra < 2000 && streamer.ResidentLevel(handles[5]) != 0u; ++extra)
    {
        std::vector<float> coverage;
        for (u32 i = 0; i < 6; ++i)
        {
            coverage.push_back(0.05f * static_cast<float>(i + 1));
        }
        RunFrame(streamer, pool, handles, std::vector<float>(6, 0.0f), coverage);
        violations += CheckMappings(device, pool, streamer, handles, mapped);
        const StreamingStats& stats = streamer.Stats();
        violations += mapped == stats.usedPages - stats.reservedPages ? 0u : 1u;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    streamer.Flush(pool);
    (void)pool.EndFrame();
    violations += CheckMappings(device, pool, streamer, handles, mapped);

    CG_EXPECT_EQ(violations, 0u);
    CG_EXPECT_TRUE(pool.Stats().evictions > 0u);
    CG_EXPECT_TRUE(pool.Stats().uploads > 0u);
    // The reversal leaves the last texture at full resolution.
    CG_EXPECT_EQ(streamer.ResidentLevel(handles[5]), 0u);

    for (const TextureHandle handle : handles)
    {
        pool.RemoveTexture(streamer, handle);
    }
    (void)streamer.Update(pool);
    CG_EXPECT_EQ(streamer.Stats().usedPages, 0u);
    streamer.Shutdown();
    pool.Shutdown();
    CG_EXPECT_EQ(device.ReportLeaks(), 0u);
}

CG_TEST(gpu_texture_pool_streams_correct_bytes_on_d3d12)
{
    const rhi::Backend* backend = cyber::rhi::test::TestBackend();
    if (backend == nullptr)
    {
        CG_SKIP("no Direct3D 12 runtime");
    }
    rhi::AdapterInfo adapters[8]{};
    const u32 adapterCount = backend->enumerateAdapters(adapters, 8);
    u32 tested = 0;
    for (u32 adapterIndex = 0; adapterIndex < adapterCount; ++adapterIndex)
    {
        HeapAllocator allocator;
        rhi::DeviceDesc deviceDesc{};
        deviceDesc.backend = backend;
        deviceDesc.adapterIndex = adapterIndex;
        deviceDesc.validation = rhi::ValidationLevel::Exhaustive;
        auto createdDevice = rhi::Device::Create(deviceDesc, allocator);
        if (!createdDevice ||
            !createdDevice.Value().Capabilities().features.Has(rhi::Feature::ReservedTextures))
        {
            continue;
        }
        rhi::Device device = std::move(createdDevice.Value());
        ++tested;

        PatternSource colour(1, 512, rhi::Format::RGBA8Unorm);
        PatternSource compressed(2, 1024, rhi::Format::BC1Unorm);
        PatternSource third(3, 1024, rhi::Format::BC1Unorm);
        // Drivers tile small mips differently, so the budget is sized from
        // this adapter's own tile counts: room for the first two textures in
        // full and the third one short by four tiles, so it can only fit by
        // evicting.
        const auto tilesOf = [&device](const PatternSource& source) -> u32
        {
            rhi::TextureDesc probe = source.Desc();
            probe.reserved = true;
            probe.usage = probe.usage | rhi::TextureUsage::CopyDestination;
            auto texture = device.CreateTexture(probe);
            if (!texture)
            {
                return 0u;
            }
            const u32 tiles = device.TextureTilingOf(texture.Value()).totalTileCount;
            (void)device.DestroyTexture(texture.Value());
            return tiles;
        };
        const u32 budgetTiles = tilesOf(colour) + tilesOf(compressed) + tilesOf(third) - 4u;
        TextureStreamer streamer;
        CG_REQUIRE(streamer.Initialize(StreamerDesc(budgetTiles), allocator).IsOk());
        GpuTexturePool pool;
        GpuTexturePoolDesc poolDesc;
        poolDesc.tileCount = budgetTiles;
        CG_REQUIRE(pool.Initialize(device, poolDesc, allocator).IsOk());
        const TextureHandle colourHandle =
            pool.AddTexture(streamer, colour, colour.Desc(), "colour").Value();
        const TextureHandle bcHandle =
            pool.AddTexture(streamer, compressed, compressed.Desc(), "bc1").Value();

        for (u32 frame = 0; frame < 200 && (streamer.ResidentLevel(colourHandle) != 0u ||
                                            streamer.ResidentLevel(bcHandle) != 0u);
             ++frame)
        {
            RunFrame(streamer, pool, {colourHandle, bcHandle}, {0.0f, 0.0f}, {0.3f, 0.3f});
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        streamer.Flush(pool);
        CG_REQUIRE(pool.EndFrame().IsOk());
        CG_REQUIRE(device.WaitIdle().IsOk());
        CG_REQUIRE(streamer.ResidentLevel(colourHandle) == 0u);
        CG_REQUIRE(streamer.ResidentLevel(bcHandle) == 0u);

        // Read every level of both textures back from the GPU.
        rhi::BufferDesc readbackDesc{};
        readbackDesc.usage = rhi::BufferUsage::Readback | rhi::BufferUsage::CopyDestination;
        readbackDesc.sizeBytes = 2u << 20u;
        const rhi::BufferHandle readback = device.CreateBuffer(readbackDesc).Value();
        const rhi::FenceHandle fence = device.CreateFence(rhi::FenceDesc{}).Value();
        u64 fenceValue = 0;
        const auto readLevel = [&](TextureHandle handle, const PatternSource& source, u32 level,
                                   bool expectZero) -> u32
        {
            const rhi::TextureHandle texture = pool.Texture(handle);
            const StreamingLevelInfo info = source.Level(level);
            const bool bc = source.Desc().format == rhi::Format::BC1Unorm;
            const u32 blocks = bc ? (info.width + 3u) / 4u : info.width;
            const u32 rowBytes = blocks * (bc ? 8u : 4u);
            const u32 pitch = (rowBytes + 255u) / 256u * 256u;
            rhi::CommandListDesc listDesc{};
            auto created = device.CreateCommandList(listDesc);
            if (!created || !created.Value().Begin().IsOk())
            {
                return 1u;
            }
            rhi::CommandList list = std::move(created.Value());
            rhi::SubresourceRange range{};
            range.baseMipLevel = level;
            range.mipLevelCount = 1;
            range.layerCount = 1;
            const rhi::TextureBarrier toCopy{texture, range,
                                             device.TextureStateOf(texture, level, 0),
                                             rhi::ResourceState::CopySource};
            const rhi::BufferBarrier bufferBarrier{readback, device.BufferStateOf(readback),
                                                   rhi::ResourceState::CopyDestination};
            const usize bufferCount =
                bufferBarrier.from == rhi::ResourceState::CopyDestination ? 0u : 1u;
            const auto barriered =
                list.Barrier(Span<const rhi::BufferBarrier>(&bufferBarrier, bufferCount),
                             Span<const rhi::TextureBarrier>(&toCopy, 1));
            if (!barriered)
            {
                std::printf("  readback barrier: %s\n", barriered.Error().message);
                return 1u;
            }
            rhi::BufferTextureCopy copy{};
            copy.buffer = readback;
            copy.bufferRowPitch = pitch;
            copy.texture = texture;
            copy.mipLevel = level;
            copy.width = bc ? blocks * 4u : info.width;
            copy.height = bc ? blocks * 4u : info.height;
            copy.depth = 1;
            const auto copied = list.CopyTextureToBuffer(copy);
            if (!copied)
            {
                std::printf("  readback copy: %s\n", copied.Error().message);
                return 1u;
            }
            const rhi::TextureBarrier back{texture, range, rhi::ResourceState::CopySource,
                                           rhi::ResourceState::ShaderRead};
            (void)list.Barrier(Span<const rhi::BufferBarrier>{},
                               Span<const rhi::TextureBarrier>(&back, 1));
            (void)list.End();
            const rhi::CommandListHandle listHandle = list.Handle();
            const rhi::FenceSignal signal{fence, ++fenceValue};
            rhi::SubmitDesc submit{};
            submit.commandLists = Span<const rhi::CommandListHandle>(&listHandle, 1);
            submit.signals = Span<const rhi::FenceSignal>(&signal, 1);
            if (!device.Submit(submit).IsOk() || !device.WaitFence(fence, fenceValue).IsOk())
            {
                return 1u;
            }
            const u32 rows = bc ? blocks : info.height;
            std::vector<Byte> bytes(static_cast<usize>(pitch) * rows);
            (void)device.ReadBuffer(readback, 0, Span<Byte>(bytes.data(), bytes.size()));
            u32 wrong = 0;
            for (u32 row = 0; row < rows; ++row)
            {
                for (u32 column = 0; column < rowBytes; ++column)
                {
                    const Byte got = bytes[static_cast<usize>(row) * pitch + column];
                    const Byte want =
                        expectZero
                            ? Byte{0}
                            : source.Expected(level, static_cast<usize>(row) * rowBytes + column);
                    wrong += got != want ? 1u : 0u;
                }
            }
            if (wrong != 0)
            {
                std::printf("  level %u (%ux%u%s): %u of %u bytes differ; got %02x %02x %02x %02x "
                            "want %02x %02x %02x %02x\n",
                            level, info.width, info.height, bc ? ", bc1" : "", wrong,
                            rows * rowBytes, static_cast<unsigned>(bytes[0]),
                            static_cast<unsigned>(bytes[1]), static_cast<unsigned>(bytes[2]),
                            static_cast<unsigned>(bytes[3]),
                            static_cast<unsigned>(source.Expected(level, 0)),
                            static_cast<unsigned>(source.Expected(level, 1)),
                            static_cast<unsigned>(source.Expected(level, 2)),
                            static_cast<unsigned>(source.Expected(level, 3)));
            }
            return wrong;
        };

        u32 wrong = 0;
        for (u32 level = 0; level < colour.LevelCount(); ++level)
        {
            wrong += readLevel(colourHandle, colour, level, false);
        }
        for (u32 level = 0; level < compressed.LevelCount(); ++level)
        {
            wrong += readLevel(bcHandle, compressed, level, false);
        }
        CG_EXPECT_EQ(wrong, 0u);

        // The colour texture leaves the view; the BC texture keeps wanting
        // everything and a third, larger texture arrives that needs more room
        // than is free. The colour texture's surplus levels are evicted
        // (unmapped) and read back as zero; its tail stays.
        const TextureHandle thirdHandle =
            pool.AddTexture(streamer, third, third.Desc(), "third").Value();
        for (u32 frame = 0; frame < 300 && streamer.ResidentLevel(thirdHandle) != 0u; ++frame)
        {
            (void)pool.BeginFrame();
            streamer.ReportUsage(bcHandle, 0.0f, 0.3f);
            streamer.ReportUsage(thirdHandle, 0.0f, 0.3f);
            (void)streamer.Update(pool);
            (void)pool.EndFrame();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CG_REQUIRE(device.WaitIdle().IsOk());
        const u32 colourResident = streamer.ResidentLevel(colourHandle);
        std::printf("adapter \"%s\": %u uploads, %u evictions, %u flushes; colour texture now "
                    "resident from level %u\n",
                    adapters[adapterIndex].name, pool.Stats().uploads, pool.Stats().evictions,
                    pool.Stats().uploadFlushes, colourResident);
        CG_EXPECT_EQ(streamer.ResidentLevel(thirdHandle), 0u);
        CG_EXPECT_TRUE(colourResident > 0u);
        u32 zeroWrong = 0;
        u32 keptWrong = 0;
        const rhi::TextureTiling colourTiling = device.TextureTilingOf(pool.Texture(colourHandle));
        for (u32 level = 0; level < colour.LevelCount(); ++level)
        {
            if (level < colourResident && level < colourTiling.standardMipCount)
            {
                zeroWrong += readLevel(colourHandle, colour, level, true);
            }
            else if (level >= colourResident)
            {
                keptWrong += readLevel(colourHandle, colour, level, false);
            }
        }
        for (u32 level = 0; level < third.LevelCount(); ++level)
        {
            keptWrong += readLevel(thirdHandle, third, level, false);
        }
        CG_EXPECT_EQ(zeroWrong, 0u);
        CG_EXPECT_EQ(keptWrong, 0u);

        CG_REQUIRE(device.DestroyBuffer(readback).IsOk());
        pool.RemoveTexture(streamer, colourHandle);
        pool.RemoveTexture(streamer, bcHandle);
        pool.RemoveTexture(streamer, thirdHandle);
        streamer.Shutdown();
        pool.Shutdown();
        CG_REQUIRE(device.DestroyFence(fence).IsOk());
        CG_EXPECT_EQ(device.ReportLeaks(), 0u);
    }
    if (tested == 0)
    {
        CG_SKIP("no adapter supports tiled resources tier 2");
    }
}
