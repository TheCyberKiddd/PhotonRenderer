#include <cyber/memory/heap_allocator.h>
#include <cyber/renderer/clustered_lighting.h>
#include <cyber/software/clustered_pbr.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/raster.h>
#include <cyber/test/test.h>

#include <array>
#include <cmath>
#include <utility>

using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using cyber::renderer::BuildClusterLightLists;
using cyber::renderer::ClusterGridDesc;
using cyber::renderer::ClusterLightBound;
using cyber::renderer::ClusterLightLists;
using cyber::software::BuildClusterLightBounds;
using cyber::software::ClusteredPbrLightingData;
using cyber::software::ClusteredPbrPixelShader;
using cyber::software::PbrLightingData;
using cyber::software::PbrPixelShader;
using cyber::software::PbrPointLight;
using cyber::software::Varyings;

CG_TEST(software_clustered_pbr_matches_brute_force_for_thousand_lights)
{
    constexpr cyber::core::usize kLightCount = 1000;
    std::array<PbrPointLight, kLightCount> pointLights{};
    std::array<ClusterLightBound, kLightCount> bounds{};
    for (cyber::core::usize index = 0; index < pointLights.size(); ++index)
    {
        const cyber::core::u32 hash =
            static_cast<cyber::core::u32>(index) * 747796405U + 2891336453U;
        const auto signedUnit = [](cyber::core::u32 value) noexcept
        { return static_cast<float>(value & 0xFFFFU) / 32767.5f - 1.0f; };
        PbrPointLight& light = pointLights[index];
        light.position = cyber::math::Vec3{signedUnit(hash) * 12.0f, signedUnit(hash >> 7) * 12.0f,
                                           -2.0f - static_cast<float>((hash >> 16) % 40U)};
        light.intensity = cyber::math::Vec3{12.0f, 10.0f, 8.0f};
        light.range = 0.5f + static_cast<float>((hash >> 24) % 40U) / 10.0f;
    }
    for (cyber::core::usize index = 0; index < 12; ++index)
    {
        pointLights[index].position =
            cyber::math::Vec3{static_cast<float>(index % 3) * 0.2f, 1.0f, -5.0f};
        pointLights[index].range = 2.0f;
    }

    ClusterGridDesc grid{};
    grid.widthPixels = 64;
    grid.heightPixels = 64;
    grid.tileSizePixels = 16;
    grid.depthSlices = 16;
    grid.verticalFovRadians = cyber::math::kPi * 0.5f;
    grid.nearPlane = 1.0f;
    grid.farPlane = 50.0f;

    const Span<const PbrPointLight> lightSpan(pointLights.data(), pointLights.size());
    const auto boundsResult =
        BuildClusterLightBounds(lightSpan, cyber::math::Mat4::Identity(),
                                Span<ClusterLightBound>(bounds.data(), bounds.size()));
    CG_REQUIRE(boundsResult);

    HeapAllocator allocator;
    ClusterLightLists clusters(allocator);
    CG_REQUIRE(BuildClusterLightLists(Span<const ClusterLightBound>(bounds.data(), bounds.size()),
                                      grid, clusters));

    PbrLightingData lighting{};
    lighting.eyePosition = cyber::math::Vec3::Zero();
    lighting.ambientIrradiance = cyber::math::Vec3::Zero();
    lighting.material.baseColor = cyber::math::Color{0.7f, 0.55f, 0.35f, 1.0f};
    lighting.material.roughness = 0.65f;
    lighting.pointLightSpan = lightSpan;

    Varyings surface{};
    surface.pixelPosition = cyber::math::Vec2{32.5f, 32.5f};
    surface.worldPosition = cyber::math::Vec3{0.0f, 0.0f, -5.0f};
    surface.normal = cyber::math::Vec3{0.0f, 1.0f, 0.0f};
    surface.color = cyber::math::Color{1.0f, 1.0f, 1.0f, 1.0f};

    ClusteredPbrLightingData clustered{};
    clustered.lighting = &lighting;
    clustered.pointLights = lightSpan;
    clustered.clusters = &clusters;
    clustered.viewFromWorld = cyber::math::Mat4::Identity();

    const cyber::core::u32 clusterIndex =
        clusters.ClusterAtPixel(surface.pixelPosition.x, surface.pixelPosition.y, 5.0f);
    CG_REQUIRE(clusterIndex != cyber::renderer::kInvalidClusterIndex);
    const auto clusterIndices = clusters.Lights(clusterIndex);
    CG_EXPECT_TRUE(clusterIndices.Size() < pointLights.size());
    bool hasIndexBeyondMismatchedSpan = false;
    for (cyber::core::usize index = 0; index < clusterIndices.Size(); ++index)
    {
        hasIndexBeyondMismatchedSpan = hasIndexBeyondMismatchedSpan || clusterIndices[index] >= 1;
    }
    CG_REQUIRE(hasIndexBeyondMismatchedSpan);

    const cyber::math::Color expected = PbrPixelShader(surface, &lighting);
    const cyber::math::Color actual = ClusteredPbrPixelShader(surface, &clustered);
    CG_EXPECT_NEAR(actual.r, expected.r, 1.0e-5f);
    CG_EXPECT_NEAR(actual.g, expected.g, 1.0e-5f);
    CG_EXPECT_NEAR(actual.b, expected.b, 1.0e-5f);

    auto framebufferResult = cyber::software::Framebuffer::Create(
        cyber::software::FramebufferDesc{64, 64, cyber::software::PixelFormat::RGBA32F}, allocator);
    CG_REQUIRE(framebufferResult);
    auto depthResult =
        cyber::software::DepthBuffer::Create(cyber::software::DepthBufferDesc{64, 64}, allocator);
    CG_REQUIRE(depthResult);
    auto framebuffer = std::move(framebufferResult.Value());
    auto depth = std::move(depthResult.Value());
    std::array<cyber::software::ShadedRasterTriangle, 1> triangles{};
    const auto setVertex = [&](cyber::software::ShadedRasterVertex& vertex, float x, float y)
    {
        vertex.position = cyber::math::Vec2{x, y};
        vertex.varyings = surface;
        vertex.depth = 0.5f;
        vertex.inverseW = 1.0f;
    };
    setVertex(triangles[0].v0, 0.0f, 0.0f);
    setVertex(triangles[0].v1, 128.0f, 0.0f);
    setVertex(triangles[0].v2, 0.0f, 128.0f);
    cyber::software::RasterizerDesc rasterDesc{};
    rasterDesc.useJobs = false;
    const auto rasterized = cyber::software::RasterizeShadedTriangles(
        framebuffer, depth,
        Span<const cyber::software::ShadedRasterTriangle>(triangles.data(), triangles.size()),
        ClusteredPbrPixelShader, &clustered, nullptr, allocator, rasterDesc);
    CG_REQUIRE(rasterized);
    constexpr std::array<std::array<cyber::core::i32, 2>, 8> kSamplePixels{{
        {4, 4},
        {20, 4},
        {36, 4},
        {52, 4},
        {4, 20},
        {20, 20},
        {36, 20},
        {20, 52},
    }};
    for (const auto& pixel : kSamplePixels)
    {
        Varyings sampleSurface = surface;
        sampleSurface.pixelPosition = cyber::math::Vec2{static_cast<float>(pixel[0]) + 0.5f,
                                                        static_cast<float>(pixel[1]) + 0.5f};
        const cyber::math::Color sampleExpected = PbrPixelShader(sampleSurface, &lighting);
        const cyber::math::Color sampleRendered = framebuffer.GetPixel(pixel[0], pixel[1]);
        CG_EXPECT_NEAR(sampleRendered.r, sampleExpected.r, 1.0e-5f);
        CG_EXPECT_NEAR(sampleRendered.g, sampleExpected.g, 1.0e-5f);
        CG_EXPECT_NEAR(sampleRendered.b, sampleExpected.b, 1.0e-5f);
    }

    clustered.pointLights = Span<const PbrPointLight>(pointLights.data(), 1);
    const cyber::math::Color incompatibleFallback = ClusteredPbrPixelShader(surface, &clustered);
    lighting.pointLightSpan = clustered.pointLights;
    const cyber::math::Color incompatibleBruteForce = PbrPixelShader(surface, &lighting);
    CG_EXPECT_NEAR(incompatibleFallback.r, incompatibleBruteForce.r, 1.0e-5f);
    CG_EXPECT_NEAR(incompatibleFallback.g, incompatibleBruteForce.g, 1.0e-5f);
    CG_EXPECT_NEAR(incompatibleFallback.b, incompatibleBruteForce.b, 1.0e-5f);
    clustered.pointLights = lightSpan;
    lighting.pointLightSpan = lightSpan;

    surface.pixelPosition.x = 64.5f;
    const cyber::math::Color fallback = ClusteredPbrPixelShader(surface, &clustered);
    const cyber::math::Color bruteForce = PbrPixelShader(surface, &lighting);
    CG_EXPECT_NEAR(fallback.r, bruteForce.r, 1.0e-5f);
    CG_EXPECT_NEAR(fallback.g, bruteForce.g, 1.0e-5f);
    CG_EXPECT_NEAR(fallback.b, bruteForce.b, 1.0e-5f);
}

CG_TEST(software_clustered_pbr_rejects_mismatched_bounds_output)
{
    const std::array<PbrPointLight, 1> lights{};
    std::array<ClusterLightBound, 2> bounds{};
    const auto result = BuildClusterLightBounds(
        Span<const PbrPointLight>(lights.data(), lights.size()), cyber::math::Mat4::Identity(),
        Span<ClusterLightBound>(bounds.data(), bounds.size()));
    CG_EXPECT_FALSE(result);
    CG_EXPECT_EQ(result.Error().code, cyber::core::ErrorCode::InvalidArgument);
}
