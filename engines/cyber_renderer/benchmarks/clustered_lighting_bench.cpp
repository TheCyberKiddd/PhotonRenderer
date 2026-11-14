#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/renderer/clustered_lighting.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>

using cyber::core::u32;
using cyber::core::u64;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using cyber::renderer::BuildClusterLightLists;
using cyber::renderer::ClusterGridDesc;
using cyber::renderer::ClusterLightBound;
using cyber::renderer::ClusterLightLists;

namespace
{

constexpr u32 kLightCount = 1000;
constexpr u32 kWarmupCount = 5;
constexpr u32 kSampleCount = 101;

[[nodiscard]] u32 RandomStep(u32& state) noexcept
{
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
}

[[nodiscard]] float RandomRange(u32& state, float minimum, float maximum) noexcept
{
    const float unit = static_cast<float>(RandomStep(state) & 0x00FFFFFFU) / 16777215.0f;
    return minimum + (maximum - minimum) * unit;
}

void AnimateLights(std::array<ClusterLightBound, kLightCount>& lights, u32& random) noexcept
{
    for (ClusterLightBound& light : lights)
    {
        light.x += RandomRange(random, -0.25f, 0.25f);
        light.y += RandomRange(random, -0.25f, 0.25f);
        light.depth += RandomRange(random, -0.1f, 0.1f);
    }
}

}  // namespace

int main()
{
    HeapAllocator allocator;
    std::array<ClusterLightBound, kLightCount> lights{};
    u32 random = 0xC0FFEEU;
    for (ClusterLightBound& light : lights)
    {
        light.depth = RandomRange(random, 20.0f, 250.0f);
        const float horizontalExtent = light.depth * 0.57735027f * 0.9f;
        const float verticalExtent = light.depth * 0.57735027f * 0.9f;
        light.x = RandomRange(random, -horizontalExtent, horizontalExtent);
        light.y = RandomRange(random, -verticalExtent, verticalExtent);
        light.range = RandomRange(random, 1.0f, 12.0f);
    }

    ClusterGridDesc grid{};
    grid.widthPixels = 1280;
    grid.heightPixels = 720;
    grid.tileSizePixels = 16;
    grid.depthSlices = 24;
    grid.verticalFovRadians = 1.04719755f;
    grid.nearPlane = 0.1f;
    grid.farPlane = 500.0f;
    ClusterLightLists lists(allocator);

    for (u32 iteration = 0; iteration < kWarmupCount; ++iteration)
    {
        AnimateLights(lights, random);
        if (!BuildClusterLightLists(Span<const ClusterLightBound>(lights.data(), lights.size()),
                                    grid, lists))
        {
            std::fprintf(stderr, "cluster build warmup failed\n");
            return 1;
        }
    }
    std::array<long long, kSampleCount> elapsedUs{};
    for (u32 iteration = 0; iteration < kSampleCount; ++iteration)
    {
        AnimateLights(lights, random);
        const auto start = std::chrono::steady_clock::now();
        const auto built = BuildClusterLightLists(
            Span<const ClusterLightBound>(lights.data(), lights.size()), grid, lists);
        const auto stop = std::chrono::steady_clock::now();
        if (!built)
        {
            std::fprintf(stderr, "cluster build failed: %s\n", built.Error().message);
            return 1;
        }
        elapsedUs[iteration] =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    }
    std::sort(elapsedUs.begin(), elapsedUs.end());

    u64 indexCount = 0;
    u32 maxOccupancy = 0;
    for (u32 cluster = 0; cluster < lists.ClusterCount(); ++cluster)
    {
        const u32 occupancy = static_cast<u32>(lists.Lights(cluster).Size());
        indexCount += occupancy;
        maxOccupancy = std::max(maxOccupancy, occupancy);
    }
    std::printf("backend=cpu config=%s logical_processors=%u lights=%u resolution=%ux%u tile=%u "
                "slices=%u clusters=%u "
                "indices=%llu max_occupancy=%u overflow=0 median_us=%lld p95_us=%lld\n",
#ifdef NDEBUG
                "release",
#else
                "debug",
#endif
                std::thread::hardware_concurrency(), kLightCount, grid.widthPixels,
                grid.heightPixels, grid.tileSizePixels, grid.depthSlices, lists.ClusterCount(),
                static_cast<unsigned long long>(indexCount), maxOccupancy,
                elapsedUs[kSampleCount / 2], elapsedUs[kSampleCount * 95 / 100]);
    return 0;
}
