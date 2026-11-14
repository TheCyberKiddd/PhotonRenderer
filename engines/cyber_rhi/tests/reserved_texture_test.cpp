// Reserved (tiled) textures and tile heaps on the validation backend
// (Phase 33): the standard tiling shapes, what may be reserved, and the
// mapping table's behaviour -- batches validated whole, unmapping, packed
// tiles, and heaps that die under their mappings.

#include "rhi_fixture.h"

#include <cyber/rhi/texture.h>
#include <cyber/test/test.h>

using namespace cyber::rhi;
using cyber::core::ErrorCode;
using cyber::core::u32;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using cyber::rhi::test::OpenDevice;

namespace
{

TextureDesc Reserved(Format format, u32 size, u32 mips)
{
    TextureDesc desc{};
    desc.format = format;
    desc.usage = TextureUsage::Sampled | TextureUsage::CopyDestination | TextureUsage::CopySource;
    desc.width = size;
    desc.height = size;
    desc.mipLevels = mips;
    desc.reserved = true;
    desc.label = "reserved";
    return desc;
}

}  // namespace

CG_TEST(standard_tiling_matches_the_d3d12_tile_shapes)
{
    // 4-byte texels: 128x128 tiles. 2048..128 are standard, 64..1 packed.
    const TextureTiling rgba = ComputeStandardTiling(Reserved(Format::RGBA8Unorm, 2048, 12));
    CG_REQUIRE(rgba.valid);
    CG_EXPECT_EQ(rgba.tileWidth, 128u);
    CG_EXPECT_EQ(rgba.tileHeight, 128u);
    CG_EXPECT_EQ(rgba.standardMipCount, 5u);
    CG_EXPECT_EQ(rgba.packedMipCount, 7u);
    CG_EXPECT_EQ(rgba.mipTilesX[0], 16u);
    CG_EXPECT_EQ(rgba.MipTileCount(4), 1u);
    // 64x64x4 + 32x32x4 + ... is under one tile.
    CG_EXPECT_EQ(rgba.packedTileCount, 1u);
    CG_EXPECT_EQ(rgba.totalTileCount, 256u + 64u + 16u + 4u + 1u + 1u);

    // BC1 (8-byte blocks): 128x64 blocks = 512x256 texels.
    const TextureTiling bc1 = ComputeStandardTiling(Reserved(Format::BC1Unorm, 2048, 12));
    CG_REQUIRE(bc1.valid);
    CG_EXPECT_EQ(bc1.tileWidth, 512u);
    CG_EXPECT_EQ(bc1.tileHeight, 256u);
    CG_EXPECT_EQ(bc1.standardMipCount, 3u);                  // 2048, 1024, 512
    CG_EXPECT_EQ(bc1.mipTilesX[0] * bc1.mipTilesY[0], 32u);  // 2 MiB in 64 KiB tiles

    // BC7 (16-byte blocks): 64x64 blocks = 256x256 texels.
    const TextureTiling bc7 = ComputeStandardTiling(Reserved(Format::BC7Unorm, 1024, 11));
    CG_REQUIRE(bc7.valid);
    CG_EXPECT_EQ(bc7.tileWidth, 256u);
    CG_EXPECT_EQ(bc7.standardMipCount, 3u);  // 1024, 512, 256

    // Not reservable: arrays and 3D.
    TextureDesc array = Reserved(Format::RGBA8Unorm, 256, 1);
    array.arrayLayers = 2;
    CG_EXPECT_FALSE(ComputeStandardTiling(array).valid);
}

CG_TEST(reserved_texture_creation_rules)
{
    HeapAllocator heap;
    Device device;
    CG_REQUIRE(OpenDevice(device, heap));
    CG_REQUIRE(device.Capabilities().features.Has(Feature::ReservedTextures));

    auto created = device.CreateTexture(Reserved(Format::RGBA8Unorm, 512, 10));
    CG_REQUIRE(created.IsOk());
    const TextureInfo info = device.TextureInfoOf(created.Value());
    CG_EXPECT_TRUE(info.reserved);
    CG_EXPECT_EQ(info.sizeBytes, 0u);  // no memory of its own
    const TextureTiling tiling = device.TextureTilingOf(created.Value());
    CG_EXPECT_TRUE(tiling.valid);
    CG_EXPECT_EQ(tiling.standardMipCount, 3u);  // 512, 256, 128
    CG_REQUIRE(device.DestroyTexture(created.Value()).IsOk());

    TextureDesc target = Reserved(Format::RGBA8Unorm, 512, 1);
    target.usage = target.usage | TextureUsage::RenderTarget;
    const auto targetResult = device.CreateTexture(target);
    CG_REQUIRE(targetResult.IsError());
    CG_EXPECT_EQ(targetResult.Error().code, ErrorCode::InvalidArgument);
    TextureDesc array = Reserved(Format::RGBA8Unorm, 512, 1);
    array.arrayLayers = 4;
    const auto arrayResult = device.CreateTexture(array);
    CG_REQUIRE(arrayResult.IsError());
    CG_EXPECT_EQ(arrayResult.Error().code, ErrorCode::InvalidArgument);

    // An ordinary texture has no tiling.
    TextureDesc plain = Reserved(Format::RGBA8Unorm, 64, 1);
    plain.reserved = false;
    auto ordinary = device.CreateTexture(plain);
    CG_REQUIRE(ordinary.IsOk());
    CG_EXPECT_FALSE(device.TextureTilingOf(ordinary.Value()).valid);
    CG_EXPECT_FALSE(device.TextureInfoOf(ordinary.Value()).reserved);
    CG_REQUIRE(device.DestroyTexture(ordinary.Value()).IsOk());

    const auto emptyHeap = device.CreateTileHeap(TileHeapDesc{0, "empty"});
    CG_REQUIRE(emptyHeap.IsError());
    CG_EXPECT_EQ(emptyHeap.Error().code, ErrorCode::InvalidArgument);
}

CG_TEST(tile_mappings_map_unmap_and_validate_whole_batches)
{
    HeapAllocator heap;
    Device device;
    CG_REQUIRE(OpenDevice(device, heap));
    const TextureHandle texture =
        device.CreateTexture(Reserved(Format::RGBA8Unorm, 256, 9)).Value();
    const TileHeapHandle pool = device.CreateTileHeap(TileHeapDesc{8, "pool"}).Value();
    const TextureTiling tiling = device.TextureTilingOf(texture);
    CG_REQUIRE(tiling.standardMipCount == 2u);  // 256 (2x2 tiles), 128 (1x1)
    CG_REQUIRE(tiling.packedTileCount == 1u);

    TileMapping mip0[4];
    for (u32 i = 0; i < 4; ++i)
    {
        mip0[i].texture = texture;
        mip0[i].mipLevel = 0;
        mip0[i].tileX = i % 2;
        mip0[i].tileY = i / 2;
        mip0[i].heap = pool;
        mip0[i].heapTile = 7 - i;  // pages need not be contiguous or ordered
    }
    CG_REQUIRE(
        device.UpdateTileMappings(QueueKind::Graphics, Span<const TileMapping>(mip0, 4)).IsOk());
    for (u32 i = 0; i < 4; ++i)
    {
        const TileMappingInfo where = device.TileMappingOf(mip0[i]);
        CG_EXPECT_TRUE(where.mapped);
        CG_EXPECT_TRUE(where.heap == pool);
        CG_EXPECT_EQ(where.heapTile, 7 - i);
    }

    // The packed tail maps as tile 0 of the packed set.
    TileMapping tail;
    tail.texture = texture;
    tail.packed = true;
    tail.heap = pool;
    tail.heapTile = 0;
    CG_REQUIRE(
        device.UpdateTileMappings(QueueKind::Graphics, Span<const TileMapping>(&tail, 1)).IsOk());
    CG_EXPECT_TRUE(device.TileMappingOf(tail).mapped);

    // A batch with one bad entry applies nothing: the first entry would have
    // unmapped tile (0,0), and it must still be mapped afterwards.
    TileMapping batch[2];
    batch[0] = mip0[0];
    batch[0].heap = TileHeapHandle{};
    batch[1] = mip0[1];
    batch[1].heapTile = 8;  // outside the 8-tile heap
    const auto badBatchResult =
        device.UpdateTileMappings(QueueKind::Graphics, Span<const TileMapping>(batch, 2));
    CG_REQUIRE(badBatchResult.IsError());
    CG_EXPECT_EQ(badBatchResult.Error().code, ErrorCode::InvalidArgument);
    CG_EXPECT_TRUE(device.TileMappingOf(mip0[0]).mapped);

    TileMapping outside = mip0[0];
    outside.tileX = 2;
    CG_EXPECT_FALSE(
        device.UpdateTileMappings(QueueKind::Graphics, Span<const TileMapping>(&outside, 1))
            .IsOk());
    TileMapping packedOutside = tail;
    packedOutside.tileX = 1;
    CG_EXPECT_FALSE(
        device.UpdateTileMappings(QueueKind::Graphics, Span<const TileMapping>(&packedOutside, 1))
            .IsOk());

    // Unmap with an invalid heap.
    TileMapping unmap = mip0[3];
    unmap.heap = TileHeapHandle{};
    CG_REQUIRE(
        device.UpdateTileMappings(QueueKind::Graphics, Span<const TileMapping>(&unmap, 1)).IsOk());
    CG_EXPECT_FALSE(device.TileMappingOf(mip0[3]).mapped);

    // Mappings into a destroyed heap read as unmapped; a stale heap is refused.
    CG_REQUIRE(device.DestroyTileHeap(pool).IsOk());
    CG_EXPECT_FALSE(device.TileMappingOf(mip0[0]).mapped);
    CG_EXPECT_FALSE(
        device.UpdateTileMappings(QueueKind::Graphics, Span<const TileMapping>(mip0, 1)).IsOk());
    CG_EXPECT_FALSE(device.DestroyTileHeap(pool).IsOk());

    // An ordinary texture cannot be mapped.
    TextureDesc plain = Reserved(Format::RGBA8Unorm, 256, 1);
    plain.reserved = false;
    const TextureHandle ordinary = device.CreateTexture(plain).Value();
    const TileHeapHandle second = device.CreateTileHeap(TileHeapDesc{1, "second"}).Value();
    TileMapping wrong;
    wrong.texture = ordinary;
    wrong.heap = second;
    CG_EXPECT_FALSE(
        device.UpdateTileMappings(QueueKind::Graphics, Span<const TileMapping>(&wrong, 1)).IsOk());
    CG_EXPECT_TRUE(
        device.UpdateTileMappings(QueueKind::Graphics, Span<const TileMapping>{}).IsOk());

    CG_REQUIRE(device.DestroyTexture(texture).IsOk());
    CG_REQUIRE(device.DestroyTexture(ordinary).IsOk());
    CG_REQUIRE(device.DestroyTileHeap(second).IsOk());
}
