#include <cyber/jobs/job_system.h>
#include <cyber/math/constants.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/renderer/clustered_lighting.h>
#include <cyber/software/clustered_pbr.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/raster.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <utility>

namespace
{

using cyber::core::u32;
using cyber::core::u64;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using cyber::software::ClusteredPbrLightingData;
using cyber::software::PbrLightingData;
using cyber::software::PbrPointLight;

constexpr u32 kWidth = 320;
constexpr u32 kHeight = 180;
constexpr u32 kLightCount = 1000;
constexpr u32 kWarmupFrames = 5;
constexpr u32 kMeasuredFrames = 101;
constexpr u32 kDepthSlices = 24;

[[nodiscard]] float SignedUnit(u32 value) noexcept
{
    return static_cast<float>(value & 0xFFFFU) / 32767.5f - 1.0f;
}

void SetVertex(cyber::software::ShadedRasterVertex& vertex, float x, float y) noexcept
{
    vertex.position = cyber::math::Vec2{x, y};
    vertex.varyings.worldPosition = cyber::math::Vec3{0.0f, 0.0f, -8.0f};
    vertex.varyings.normal = cyber::math::Vec3{0.0f, 1.0f, 0.0f};
    vertex.varyings.color = cyber::math::Color{1.0f, 1.0f, 1.0f, 1.0f};
    vertex.depth = 0.5f;
    vertex.inverseW = 1.0f;
}

}  // namespace

int main()
{
    using namespace cyber::renderer;
    using namespace cyber::software;

    HeapAllocator allocator;
    cyber::jobs::JobSystem::Desc jobDesc{};
    jobDesc.workerCount = 3;
    auto jobSystemResult = cyber::jobs::JobSystem::Create(jobDesc, allocator);
    if (!jobSystemResult)
    {
        std::fprintf(stderr, "clustered PBR benchmark: job system creation failed: %s\n",
                     jobSystemResult.Error().message);
        return 1;
    }
    auto jobSystem = std::move(jobSystemResult.Value());

    std::array<PbrPointLight, kLightCount> lights{};
    std::array<ClusterLightBound, kLightCount> bounds{};
    for (u32 index = 0; index < kLightCount; ++index)
    {
        const u32 hash = index * 747796405U + 2891336453U;
        lights[index].position =
            cyber::math::Vec3{SignedUnit(hash) * 12.0f, SignedUnit(hash >> 7) * 8.0f,
                              -2.0f - static_cast<float>((hash >> 16) % 46U)};
        lights[index].intensity = cyber::math::Vec3{8.0f, 7.0f, 6.0f};
        lights[index].range = 0.5f + static_cast<float>((hash >> 24) % 40U) / 10.0f;
    }

    ClusterGridDesc grid{};
    grid.widthPixels = kWidth;
    grid.heightPixels = kHeight;
    grid.tileSizePixels = 16;
    grid.depthSlices = kDepthSlices;
    grid.verticalFovRadians = cyber::math::kPi / 3.0f;
    grid.nearPlane = 0.1f;
    grid.farPlane = 64.0f;
    ClusterLightLists clusters(allocator);

    PbrLightingData lighting{};
    lighting.eyePosition = cyber::math::Vec3::Zero();
    lighting.ambientIrradiance = cyber::math::Vec3{0.01f, 0.01f, 0.01f};
    lighting.material.baseColor = cyber::math::Color{0.7f, 0.55f, 0.35f, 1.0f};
    lighting.material.roughness = 0.65f;
    lighting.pointLightSpan = Span<const PbrPointLight>(lights.data(), lights.size());

    ClusteredPbrLightingData clustered{};
    clustered.lighting = &lighting;
    clustered.pointLights = lighting.pointLightSpan;
    clustered.clusters = &clusters;
    clustered.viewFromWorld = cyber::math::Mat4::Identity();

    FramebufferDesc framebufferDesc{};
    framebufferDesc.width = static_cast<cyber::core::i32>(kWidth);
    framebufferDesc.height = static_cast<cyber::core::i32>(kHeight);
    framebufferDesc.format = PixelFormat::RGBA32F;
    auto framebufferResult = Framebuffer::Create(framebufferDesc, allocator);
    if (!framebufferResult)
    {
        std::fprintf(stderr, "clustered PBR benchmark: framebuffer creation failed: %s\n",
                     framebufferResult.Error().message);
        return 1;
    }
    auto framebuffer = std::move(framebufferResult.Value());

    DepthBufferDesc depthDesc{};
    depthDesc.width = static_cast<cyber::core::i32>(kWidth);
    depthDesc.height = static_cast<cyber::core::i32>(kHeight);
    auto depthResult = DepthBuffer::Create(depthDesc, allocator);
    if (!depthResult)
    {
        std::fprintf(stderr, "clustered PBR benchmark: depth creation failed: %s\n",
                     depthResult.Error().message);
        return 1;
    }
    auto depth = std::move(depthResult.Value());

    std::array<ShadedRasterTriangle, 2> triangles{};
    SetVertex(triangles[0].v0, 0.0f, 0.0f);
    SetVertex(triangles[0].v1, static_cast<float>(kWidth), 0.0f);
    SetVertex(triangles[0].v2, static_cast<float>(kWidth), static_cast<float>(kHeight));
    SetVertex(triangles[1].v0, 0.0f, 0.0f);
    SetVertex(triangles[1].v1, static_cast<float>(kWidth), static_cast<float>(kHeight));
    SetVertex(triangles[1].v2, 0.0f, static_cast<float>(kHeight));

    RasterizerDesc rasterDesc{};
    rasterDesc.useJobs = true;
    std::array<double, kMeasuredFrames> frameMilliseconds{};
    u64 indexCount = 0;
    u32 maxOccupancy = 0;
    std::array<u32, ((kWidth + 15U) / 16U) * ((kHeight + 15U) / 16U)> heatMap{};

    for (u32 frame = 0; frame < kWarmupFrames + kMeasuredFrames; ++frame)
    {
        const float shift = static_cast<float>(frame % 11U) * 0.015f;
        for (u32 light = 0; light < kLightCount; ++light)
        {
            lights[light].position.x += shift;
            const float originalX = SignedUnit(light * 747796405U + 2891336453U) * 12.0f;
            lights[light].position.x =
                std::clamp(lights[light].position.x, originalX - 0.1f, originalX + 0.1f);
        }

        const auto start = std::chrono::steady_clock::now();
        const auto boundsStatus =
            BuildClusterLightBounds(lighting.pointLightSpan, clustered.viewFromWorld,
                                    Span<ClusterLightBound>(bounds.data(), bounds.size()));
        if (!boundsStatus)
        {
            std::fprintf(stderr, "clustered PBR benchmark: light bounds failed: %s\n",
                         boundsStatus.Error().message);
            return 1;
        }
        const auto buildStatus = BuildClusterLightLists(
            Span<const ClusterLightBound>(bounds.data(), bounds.size()), grid, clusters);
        if (!buildStatus)
        {
            std::fprintf(stderr, "clustered PBR benchmark: cluster build failed: %s\n",
                         buildStatus.Error().message);
            return 1;
        }
        framebuffer.Clear(cyber::math::Color{0.02f, 0.025f, 0.035f, 1.0f});
        depth.Clear(0.0f);
        RasterStats stats{};
        const auto renderStatus = RasterizeShadedTriangles(
            framebuffer, depth,
            Span<const ShadedRasterTriangle>(triangles.data(), triangles.size()),
            ClusteredPbrPixelShader, &clustered, &jobSystem, allocator, rasterDesc, &stats);
        if (!renderStatus)
        {
            std::fprintf(stderr, "clustered PBR benchmark: rasterization failed: %s\n",
                         renderStatus.Error().message);
            return 1;
        }
        const auto end = std::chrono::steady_clock::now();

        if (stats.pixels != static_cast<cyber::core::u64>(kWidth) * kHeight)
        {
            std::fprintf(stderr, "clustered PBR benchmark: fullscreen scene coverage mismatch\n");
            return 1;
        }
        indexCount = 0;
        for (u32 cluster = 0; cluster < clusters.ClusterCount(); ++cluster)
        {
            indexCount += clusters.Lights(cluster).Size();
        }
        const auto heatStatus =
            BuildClusterHeatMap(clusters, Span<u32>(heatMap.data(), heatMap.size()));
        if (!heatStatus)
        {
            std::fprintf(stderr, "clustered PBR benchmark: heat map failed: %s\n",
                         heatStatus.Error().message);
            return 1;
        }
        maxOccupancy = *std::max_element(heatMap.begin(), heatMap.end());

        if (frame >= kWarmupFrames)
        {
            frameMilliseconds[frame - kWarmupFrames] =
                std::chrono::duration<double, std::milli>(end - start).count();
        }
    }

    std::sort(frameMilliseconds.begin(), frameMilliseconds.end());
    const double median = frameMilliseconds[frameMilliseconds.size() / 2];
    const double p95 = frameMilliseconds[(frameMilliseconds.size() * 95U + 99U) / 100U - 1U];
    const u32 tilesX = (kWidth + grid.tileSizePixels - 1U) / grid.tileSizePixels;
    const u32 tilesY = (kHeight + grid.tileSizePixels - 1U) / grid.tileSizePixels;
    std::printf(
        "backend=cpu config=%s logical_processors=%u workers=%u lights=%u resolution=%ux%u tile=%u "
        "slices=%u clusters=%u indices=%llu max_occupancy=%u warmup=%u samples=%u "
        "median_frame_ms=%.3f p95_frame_ms=%.3f rendered_pixels=%u\n",
#if defined(NDEBUG)
        "release",
#else
        "debug",
#endif
        std::thread::hardware_concurrency(), jobSystem.WorkerCount(), kLightCount, kWidth, kHeight,
        grid.tileSizePixels, grid.depthSlices, tilesX * tilesY * grid.depthSlices,
        static_cast<unsigned long long>(indexCount), maxOccupancy, kWarmupFrames, kMeasuredFrames,
        median, p95, kWidth * kHeight);
    return 0;
}
