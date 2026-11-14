#include <cyber/memory/allocator.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/renderer/clustered_lighting.h>
#include <cyber/test/test.h>

#include <array>
#include <cmath>
#include <limits>

using cyber::core::ErrorCode;
using cyber::core::u32;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using cyber::renderer::BuildClusterHeatMap;
using cyber::renderer::BuildClusterLightLists;
using cyber::renderer::ClusterGridDesc;
using cyber::renderer::ClusterLightBound;
using cyber::renderer::ClusterLightLists;

namespace
{

class SwitchableAllocator final : public cyber::memory::Allocator
{
public:
    void* Allocate(cyber::memory::usize size, cyber::memory::usize alignment) noexcept override
    {
        return fail ? nullptr : backing.Allocate(size, alignment);
    }

    void Deallocate(void* pointer) noexcept override
    {
        backing.Deallocate(pointer);
    }

    void Reset() noexcept override
    {
        backing.Reset();
    }

    [[nodiscard]] bool Owns(const void* pointer) const noexcept override
    {
        return backing.Owns(pointer);
    }

    [[nodiscard]] cyber::memory::AllocatorStats Stats() const noexcept override
    {
        return backing.Stats();
    }

    bool fail = false;

private:
    HeapAllocator backing;
};

[[nodiscard]] ClusterGridDesc SmallGrid() noexcept
{
    ClusterGridDesc grid{};
    grid.widthPixels = 48;
    grid.heightPixels = 48;
    grid.tileSizePixels = 16;
    grid.depthSlices = 2;
    grid.verticalFovRadians = 1.570796327f;
    grid.nearPlane = 1.0f;
    grid.farPlane = 9.0f;
    return grid;
}

}  // namespace

CG_TEST(renderer_clustered_lighting_builds_deterministic_lists_and_empty_clusters)
{
    HeapAllocator allocator;
    ClusterLightLists lists(allocator);
    const std::array<ClusterLightBound, 4> lights{{
        {0.0f, 0.0f, 2.0f, 0.2f},
        {0.0f, 0.0f, 6.0f, 0.2f},
        {100.0f, 0.0f, 2.0f, 0.2f},
        {0.0f, 0.0f, 2.0f, 0.2f},
    }};
    const auto built = BuildClusterLightLists(
        Span<const ClusterLightBound>(lights.data(), lights.size()), SmallGrid(), lists);
    CG_REQUIRE(built);
    CG_EXPECT_EQ(lists.TilesX(), u32{3});
    CG_EXPECT_EQ(lists.TilesY(), u32{3});
    CG_EXPECT_EQ(lists.ClusterCount(), u32{18});
    CG_EXPECT_EQ(lists.ClusterAtPixel(0.0f, 0.0f, 1.0f), u32{0});
    CG_EXPECT_EQ(lists.ClusterAtPixel(47.0f, 47.0f, 9.0f), u32{17});
    CG_EXPECT_EQ(lists.ClusterAtPixel(48.0f, 0.0f, 1.0f), cyber::renderer::kInvalidClusterIndex);

    bool nearLightFound = false;
    bool farLightFound = false;
    for (u32 cluster = 0; cluster < lists.ClusterCount(); ++cluster)
    {
        const auto indices = lists.Lights(cluster);
        for (cyber::core::usize lightIndex = 0; lightIndex < indices.Size(); ++lightIndex)
        {
            const u32 index = indices[lightIndex];
            CG_EXPECT_TRUE(index < lights.size());
            if (index == 0)
            {
                nearLightFound = true;
                CG_EXPECT_EQ(cluster / 9, u32{0});
            }
            if (index == 1)
            {
                farLightFound = true;
                CG_EXPECT_EQ(cluster / 9, u32{1});
            }
            if (index == 3)
            {
                CG_EXPECT_EQ(cluster / 9, u32{0});
            }
            CG_EXPECT_NE(index, u32{2});
        }
    }
    CG_EXPECT_TRUE(nearLightFound);
    CG_EXPECT_TRUE(farLightFound);
    CG_EXPECT_TRUE(lists.Lights(0).Empty());
    CG_EXPECT_TRUE(lists.Lights(8).Empty());
    CG_EXPECT_TRUE(lists.Lights(17).Empty());
    const auto nearIndices = lists.Lights(4);
    CG_REQUIRE(nearIndices.Size() == 2);
    CG_EXPECT_EQ(nearIndices[0], u32{0});
    CG_EXPECT_EQ(nearIndices[1], u32{3});
    const cyber::memory::usize allocationCount = allocator.Stats().allocations;
    CG_REQUIRE(BuildClusterLightLists(Span<const ClusterLightBound>(lights.data(), lights.size()),
                                      SmallGrid(), lists));
    CG_EXPECT_EQ(allocator.Stats().allocations, allocationCount);
    CG_EXPECT_EQ(lists.Lights(4)[0], u32{0});
    CG_EXPECT_EQ(lists.Lights(4)[1], u32{3});
}

CG_TEST(renderer_clustered_lighting_keeps_output_unchanged_on_invalid_input)
{
    HeapAllocator allocator;
    ClusterLightLists lists(allocator);
    const ClusterLightBound valid{0.0f, 0.0f, 3.0f, 0.5f};
    CG_REQUIRE(
        BuildClusterLightLists(Span<const ClusterLightBound>(&valid, 1), SmallGrid(), lists));
    const u32 originalClusters = lists.ClusterCount();
    const auto originalLights = lists.Lights(0);
    const u32 originalSize = static_cast<u32>(originalLights.Size());

    ClusterLightBound invalid = valid;
    invalid.range = std::numeric_limits<float>::quiet_NaN();
    const auto rejected =
        BuildClusterLightLists(Span<const ClusterLightBound>(&invalid, 1), SmallGrid(), lists);
    CG_REQUIRE(!rejected);
    CG_EXPECT_EQ(rejected.Error().code, ErrorCode::InvalidArgument);
    CG_EXPECT_EQ(lists.ClusterCount(), originalClusters);
    CG_EXPECT_EQ(lists.Lights(0).Size(), originalSize);

    ClusterGridDesc badGrid = SmallGrid();
    badGrid.farPlane = badGrid.nearPlane;
    const auto rejectedGrid =
        BuildClusterLightLists(Span<const ClusterLightBound>(&valid, 1), badGrid, lists);
    CG_REQUIRE(!rejectedGrid);
    CG_EXPECT_EQ(rejectedGrid.Error().code, ErrorCode::InvalidArgument);
    CG_EXPECT_EQ(lists.ClusterCount(), originalClusters);
}

CG_TEST(renderer_clustered_lighting_keeps_output_unchanged_on_allocation_failure)
{
    SwitchableAllocator allocator;
    ClusterLightLists lists(allocator);
    const ClusterLightBound light{0.0f, 0.0f, 3.0f, 0.5f};
    CG_REQUIRE(
        BuildClusterLightLists(Span<const ClusterLightBound>(&light, 1), SmallGrid(), lists));
    const u32 originalClusters = lists.ClusterCount();
    const u32 originalOccupancy = static_cast<u32>(lists.Lights(4).Size());

    allocator.fail = true;
    ClusterGridDesc largerGrid = SmallGrid();
    largerGrid.widthPixels *= 2;
    const auto rejected =
        BuildClusterLightLists(Span<const ClusterLightBound>(&light, 1), largerGrid, lists);
    CG_REQUIRE(!rejected);
    CG_EXPECT_EQ(rejected.Error().code, ErrorCode::Internal);
    CG_EXPECT_EQ(lists.ClusterCount(), originalClusters);
    CG_EXPECT_EQ(lists.Lights(4).Size(), originalOccupancy);
}

CG_TEST(renderer_clustered_lighting_builds_tile_heat_map)
{
    HeapAllocator allocator;
    ClusterLightLists lists(allocator);
    const std::array<ClusterLightBound, 2> lights{{
        {0.0f, 0.0f, 3.0f, 0.5f},
        {0.0f, 0.0f, 3.0f, 0.5f},
    }};
    CG_REQUIRE(BuildClusterLightLists(Span<const ClusterLightBound>(lights.data(), lights.size()),
                                      SmallGrid(), lists));
    std::array<u32, 9> heatMap{};
    CG_REQUIRE(BuildClusterHeatMap(lists, Span<u32>(heatMap.data(), heatMap.size())));
    CG_EXPECT_EQ(heatMap[4], u32{2});
    bool occupied = false;
    for (u32 occupancy : heatMap)
    {
        occupied = occupied || occupancy > 0;
    }
    CG_EXPECT_TRUE(occupied);

    std::array<u32, 8> wrongSize{};
    const auto rejected = BuildClusterHeatMap(lists, Span<u32>(wrongSize.data(), wrongSize.size()));
    CG_REQUIRE(!rejected);
    CG_EXPECT_EQ(rejected.Error().code, ErrorCode::InvalidArgument);
}

CG_TEST(renderer_clustered_lighting_rejects_grid_size_overflow)
{
    HeapAllocator allocator;
    ClusterLightLists lists(allocator);
    ClusterGridDesc grid{};
    grid.widthPixels = std::numeric_limits<u32>::max();
    grid.heightPixels = std::numeric_limits<u32>::max();
    grid.tileSizePixels = 1;
    grid.depthSlices = 24;
    const auto rejected = BuildClusterLightLists({}, grid, lists);
    CG_REQUIRE(!rejected);
    CG_EXPECT_EQ(rejected.Error().code, ErrorCode::OutOfRange);
    CG_EXPECT_EQ(lists.ClusterCount(), u32{0});
}

CG_TEST(renderer_clustered_lighting_matches_sampled_light_spheres)
{
    HeapAllocator allocator;
    ClusterLightLists lists(allocator);
    ClusterGridDesc grid{};
    grid.widthPixels = 320;
    grid.heightPixels = 180;
    grid.tileSizePixels = 16;
    grid.depthSlices = 12;
    grid.verticalFovRadians = 1.04719755f;
    grid.nearPlane = 0.5f;
    grid.farPlane = 128.0f;
    std::array<ClusterLightBound, 48> lights{};
    u32 random = 0xA341316CU;
    auto nextRandom = [&random]() noexcept
    {
        random ^= random << 13U;
        random ^= random >> 17U;
        random ^= random << 5U;
        return static_cast<float>(random & 0x00FFFFFFU) / 16777215.0f;
    };
    for (ClusterLightBound& light : lights)
    {
        light.depth = 1.0f + nextRandom() * 48.0f;
        light.x = (nextRandom() * 2.0f - 1.0f) * light.depth * 0.9f;
        light.y = (nextRandom() * 2.0f - 1.0f) * light.depth * 0.45f;
        light.range = 0.25f + nextRandom() * 7.0f;
    }
    CG_REQUIRE(BuildClusterLightLists(Span<const ClusterLightBound>(lights.data(), lights.size()),
                                      grid, lists));

    const float tanHalfFovY = std::tan(grid.verticalFovRadians * 0.5f);
    const float aspect = static_cast<float>(grid.widthPixels) / grid.heightPixels;
    for (u32 lightIndex = 0; lightIndex < lights.size(); ++lightIndex)
    {
        const ClusterLightBound& light = lights[lightIndex];
        for (u32 sample = 0; sample < 128; ++sample)
        {
            const float dx = nextRandom() * 2.0f - 1.0f;
            const float dy = nextRandom() * 2.0f - 1.0f;
            const float dz = nextRandom() * 2.0f - 1.0f;
            if (dx * dx + dy * dy + dz * dz > 1.0f)
            {
                continue;
            }
            const float x = light.x + dx * light.range;
            const float y = light.y + dy * light.range;
            const float depth = light.depth + dz * light.range;
            if (depth < grid.nearPlane || depth > grid.farPlane)
            {
                continue;
            }
            const float pixelX = (0.5f + 0.5f * x / (depth * tanHalfFovY * aspect)) *
                                 static_cast<float>(grid.widthPixels);
            const float pixelY =
                (0.5f - 0.5f * y / (depth * tanHalfFovY)) * static_cast<float>(grid.heightPixels);
            if (pixelX < 0.0f || pixelY < 0.0f || pixelX >= static_cast<float>(grid.widthPixels) ||
                pixelY >= static_cast<float>(grid.heightPixels))
            {
                continue;
            }
            const u32 cluster = lists.ClusterAtPixel(pixelX, pixelY, depth);
            CG_EXPECT_NE(cluster, cyber::renderer::kInvalidClusterIndex);
            const auto indices = lists.Lights(cluster);
            bool found = false;
            for (cyber::core::usize index = 0; index < indices.Size(); ++index)
            {
                found = found || indices[index] == lightIndex;
            }
            CG_EXPECT_TRUE(found);
        }
    }
}
