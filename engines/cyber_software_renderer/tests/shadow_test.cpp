#include <cyber/memory/heap_allocator.h>
#include <cyber/software/image_io.h>
#include <cyber/software/pbr.h>
#include <cyber/software/shader.h>
#include <cyber/test/test.h>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace cyber::memory;
using namespace cyber::software;
using cyber::core::f32;
using cyber::math::Color;
using cyber::math::Mat4;
using cyber::math::Vec3;

namespace
{

DepthBuffer MakeShadowMap(HeapAllocator& allocator, cyber::core::i32 width, cyber::core::i32 height)
{
    auto result = DepthBuffer::Create(DepthBufferDesc{width, height, 0}, allocator);
    if (!result)
    {
        return DepthBuffer{};
    }
    return std::move(result.Value());
}

Framebuffer MakeScratchColor(HeapAllocator& allocator, cyber::core::i32 width,
                             cyber::core::i32 height)
{
    auto result =
        Framebuffer::Create(FramebufferDesc{width, height, PixelFormat::RGBA32F, 0}, allocator);
    if (!result)
    {
        return Framebuffer{};
    }
    return std::move(result.Value());
}

bool MatrixNear(const cyber::math::Mat4& lhs, const cyber::math::Mat4& rhs, f32 tolerance)
{
    for (cyber::core::u32 column = 0; column < 4; ++column)
    {
        for (cyber::core::u32 row = 0; row < 4; ++row)
        {
            if (std::fabs(lhs[column][row] - rhs[column][row]) > tolerance)
            {
                return false;
            }
        }
    }
    return true;
}

constexpr cyber::core::i32 kShadowSceneWidth = 256;
constexpr cyber::core::i32 kShadowSceneHeight = 192;
constexpr cyber::core::i32 kShadowSceneSupersample = 15;
constexpr cyber::core::i32 kFogSceneSupersample = 8;
constexpr cyber::core::i32 kShadowRenderWidth = kShadowSceneWidth * kShadowSceneSupersample;
constexpr cyber::core::i32 kShadowRenderHeight = kShadowSceneHeight * kShadowSceneSupersample;
constexpr cyber::core::i32 kFogRenderWidth = kShadowSceneWidth * kFogSceneSupersample;
constexpr cyber::core::i32 kFogRenderHeight = kShadowSceneHeight * kFogSceneSupersample;

struct ShadowSceneShaderData
{
    TransformShaderData transform;
    PbrLightingData lighting;
};

void ShadowSceneVertex(const VertexInput& input, const void* userData,
                       VertexStageOutput& output) noexcept
{
    const auto& data = *static_cast<const ShadowSceneShaderData*>(userData);
    TransformVertexShader(input, &data.transform, output);
}

Color ShadowScenePixel(const Varyings& varyings, const void* userData) noexcept
{
    const auto& data = *static_cast<const ShadowSceneShaderData*>(userData);
    return PbrPixelShader(varyings, &data.lighting);
}

void AppendTriangle(VertexInput* vertices, cyber::core::u32& count, const Vec3& a, const Vec3& b,
                    const Vec3& c, const Vec3& normal, const Color& color)
{
    const Vec3 positions[3] = {a, b, c};
    for (const Vec3& position : positions)
    {
        VertexInput& vertex = vertices[count++];
        vertex.position = position;
        vertex.normal = normal;
        vertex.color = color;
    }
}

void AppendBox(VertexInput* vertices, cyber::core::u32& count, const Vec3& center,
               const Vec3& halfExtents, const Color& color)
{
    constexpr Vec3 kNormals[6] = {{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}, {-1.0f, 0.0f, 0.0f},
                                  {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},  {0.0f, -1.0f, 0.0f}};
    constexpr cyber::core::u32 kFaces[6][4] = {
        {4, 5, 6, 7}, {1, 0, 3, 2}, {0, 4, 7, 3}, {5, 1, 2, 6}, {7, 6, 2, 3}, {0, 1, 5, 4},
    };
    const Vec3 corners[8] = {
        center + Vec3{-halfExtents.x, -halfExtents.y, -halfExtents.z},
        center + Vec3{halfExtents.x, -halfExtents.y, -halfExtents.z},
        center + Vec3{halfExtents.x, halfExtents.y, -halfExtents.z},
        center + Vec3{-halfExtents.x, halfExtents.y, -halfExtents.z},
        center + Vec3{-halfExtents.x, -halfExtents.y, halfExtents.z},
        center + Vec3{halfExtents.x, -halfExtents.y, halfExtents.z},
        center + Vec3{halfExtents.x, halfExtents.y, halfExtents.z},
        center + Vec3{-halfExtents.x, halfExtents.y, halfExtents.z},
    };
    constexpr cyber::core::u32 kTriangles[2][3] = {{0, 1, 2}, {0, 2, 3}};
    for (cyber::core::u32 face = 0; face < 6; ++face)
    {
        for (const auto& triangle : kTriangles)
        {
            AppendTriangle(vertices, count, corners[kFaces[face][triangle[0]]],
                           corners[kFaces[face][triangle[1]]], corners[kFaces[face][triangle[2]]],
                           kNormals[face], color);
        }
    }
}

bool RenderShadowAcceptanceScene(HeapAllocator& heap, const VertexInput* vertices,
                                 cyber::core::u32 vertexCount, const Mat4& cameraViewProjection,
                                 const Vec3& eye, const Vec3& cameraForward,
                                 const Vec3& lightTravel, const ShadowMapView& shadow,
                                 Framebuffer& color, DepthBuffer& depth,
                                 bool enableFog = false,
                                 const DirectionalShadowCascadeSet* cascades = nullptr)
{
    ShadowSceneShaderData data{};
    data.transform.modelViewProjection = cameraViewProjection;
    data.transform.modelToWorld = Mat4::Identity();
    data.transform.normalToWorld = Mat4::Identity();
    data.lighting.eyePosition = eye;
    data.lighting.cameraForward = cameraForward;
    data.lighting.ambientIrradiance = Vec3{0.04f, 0.04f, 0.04f};
    data.lighting.material.baseColor = Color{1.0f, 1.0f, 1.0f, 1.0f};
    data.lighting.material.roughness = 0.72f;
    data.lighting.directionalCount = 1;
    data.lighting.directional[0].direction = lightTravel;
    data.lighting.directional[0].radiance = Vec3{3.0f, 2.85f, 2.65f};
    data.lighting.directional[0].shadow = shadow;
    if (cascades != nullptr)
    {
        data.lighting.directional[0].cascades = *cascades;
    }
    if (enableFog)
    {
        data.lighting.fog.density = 0.045f;
        data.lighting.fog.heightFalloff = 0.2f;
        data.lighting.fog.baseHeight = 0.0f;
        data.lighting.fog.maxDistance = 35.0f;
        data.lighting.fog.scatteringAlbedo = Vec3{0.72f, 0.8f, 0.92f};
        data.lighting.fog.scatteringIntensity = 3.0f;
        data.lighting.fog.anisotropy = 0.25f;
        data.lighting.fog.integrationSteps = 8;
    }
    color.Clear(Color{0.035f, 0.045f, 0.065f, 1.0f});
    depth.Clear();

    ShaderDesc shader{};
    shader.pipeline.viewport =
        Viewport{0.0f, 0.0f, static_cast<f32>(color.Width()), static_cast<f32>(color.Height())};
    shader.pipeline.cullMode = CullMode::None;
    shader.vertexShader = &ShadowSceneVertex;
    shader.pixelShader = &ShadowScenePixel;
    shader.userData = &data;
    return static_cast<bool>(
        DrawShadedTriangles(shader, cyber::memory::Span<const VertexInput>(vertices, vertexCount),
                            color, depth, nullptr, heap));
}

bool DownsampleShadowScene(const Framebuffer& source, Framebuffer& destination,
                           cyber::core::i32 factor = kShadowSceneSupersample)
{
    if (!source.IsValid() || !destination.IsValid() || factor < 1 ||
        source.Width() != destination.Width() * factor ||
        source.Height() != destination.Height() * factor)
    {
        return false;
    }

    const f32 sampleCount = static_cast<f32>(factor * factor);
    for (cyber::core::i32 y = 0; y < destination.Height(); ++y)
    {
        for (cyber::core::i32 x = 0; x < destination.Width(); ++x)
        {
            Color sum{0.0f, 0.0f, 0.0f, 0.0f};
            for (cyber::core::i32 sampleY = 0; sampleY < factor; ++sampleY)
            {
                for (cyber::core::i32 sampleX = 0; sampleX < factor; ++sampleX)
                {
                    const Color sample =
                        source.GetPixel(x * factor + sampleX, y * factor + sampleY);
                    sum.r += sample.r;
                    sum.g += sample.g;
                    sum.b += sample.b;
                    sum.a += sample.a;
                }
            }
            destination.SetPixel(
                x, y,
                ToneMapAces(Color{sum.r / sampleCount, sum.g / sampleCount,
                                  sum.b / sampleCount, sum.a / sampleCount}));
        }
    }
    return true;
}

}  // namespace

CG_TEST(software_shadow_scene_supersampling_resolves_linear_edge_coverage)
{
    HeapAllocator heap;
    Framebuffer source = MakeScratchColor(heap, kShadowSceneSupersample * 2,
                                          kShadowSceneSupersample);
    Framebuffer resolved = MakeScratchColor(heap, 2, 1);
    CG_REQUIRE(source.IsValid());
    CG_REQUIRE(resolved.IsValid());

    for (cyber::core::i32 y = 0; y < source.Height(); ++y)
    {
        for (cyber::core::i32 x = 0; x < source.Width(); ++x)
        {
            const bool black = x < kShadowSceneSupersample && x + y < 14;
            source.SetPixel(x, y, black ? Color{0.0f, 0.0f, 0.0f, 1.0f}
                                        : Color{1.0f, 1.0f, 1.0f, 1.0f});
        }
    }

    CG_REQUIRE(DownsampleShadowScene(source, resolved));
    const f32 whiteCoverage = 1.0f - (105.0f / 225.0f);
    const Color expectedEdge = ToneMapAces(Color{whiteCoverage, whiteCoverage, whiteCoverage, 1.0f});
    const Color resolvedEdge = resolved.GetPixel(0, 0);
    const Color resolvedInterior = resolved.GetPixel(1, 0);
    CG_EXPECT_NEAR(resolvedEdge.r, expectedEdge.r, 1.0e-6f);
    CG_EXPECT_NEAR(resolvedInterior.r, ToneMapAces(Color{1.0f, 1.0f, 1.0f, 1.0f}).r,
                   1.0e-6f);
}

CG_TEST(software_shadow_reverse_z_visibility_and_pcf)
{
    HeapAllocator heap;
    DepthBuffer depth = MakeShadowMap(heap, 5, 5);
    CG_REQUIRE(depth.IsValid());
    depth.Clear();
    depth.TestAndWrite(2, 2, 0.5f);

    ShadowMapView shadow{};
    shadow.depth = &depth;
    shadow.constantDepthBias = 0.0f;
    shadow.slopeDepthBias = 0.0f;
    shadow.filterRadius = 0;

    CG_EXPECT_NEAR(EvaluateShadowVisibility(shadow, Vec3{0.0f, 0.0f, 0.25f}, Vec3{0.0f, 0.0f, 1.0f},
                                            Vec3{0.0f, 0.0f, 1.0f}),
                   0.0f, 1.0e-6f);
    CG_EXPECT_NEAR(EvaluateShadowVisibility(shadow, Vec3{0.0f, 0.0f, 0.75f}, Vec3{0.0f, 0.0f, 1.0f},
                                            Vec3{0.0f, 0.0f, 1.0f}),
                   1.0f, 1.0e-6f);

    shadow.filterRadius = 1;
    CG_EXPECT_NEAR(EvaluateShadowVisibility(shadow, Vec3{0.0f, 0.0f, 0.25f}, Vec3{0.0f, 0.0f, 1.0f},
                                            Vec3{0.0f, 0.0f, 1.0f}),
                   8.0f / 9.0f, 1.0e-6f);
    shadow.softFilter = true;
    const f32 softVisibility = EvaluateShadowVisibility(
        shadow, Vec3{0.0f, 0.0f, 0.25f}, Vec3{0.0f, 0.0f, 1.0f}, Vec3{0.0f, 0.0f, 1.0f});
    CG_EXPECT_TRUE(std::isfinite(softVisibility));
    CG_EXPECT_TRUE(softVisibility >= 0.0f && softVisibility <= 1.0f);
}

CG_TEST(software_shadow_soft_filter_supports_a_bounded_wide_radius)
{
    HeapAllocator heap;
    DepthBuffer depth = MakeShadowMap(heap, 64, 64);
    CG_REQUIRE(depth.IsValid());
    depth.Clear(0.9f);
    depth.Row(20)[20] = 0.0f;

    ShadowMapView shadow{};
    shadow.depth = &depth;
    shadow.constantDepthBias = 0.0f;
    shadow.slopeDepthBias = 0.0f;
    shadow.softFilter = true;
    shadow.filterRadius = kMaxSoftShadowFilterRadius;
    CG_EXPECT_NEAR(EvaluateShadowVisibility(shadow, Vec3{0.0f, 0.0f, 0.5f}, Vec3{0.0f, 0.0f, 1.0f},
                                            Vec3{0.0f, 0.0f, 1.0f}),
                   1.0f / 16.0f, 1.0e-6f);

    shadow.filterRadius = 100;
    CG_EXPECT_NEAR(EvaluateShadowVisibility(shadow, Vec3{0.0f, 0.0f, 0.5f}, Vec3{0.0f, 0.0f, 1.0f},
                                            Vec3{0.0f, 0.0f, 1.0f}),
                   1.0f / 16.0f, 1.0e-6f);
    shadow.softFilter = false;
    CG_EXPECT_NEAR(EvaluateShadowVisibility(shadow, Vec3{0.0f, 0.0f, 0.5f}, Vec3{0.0f, 0.0f, 1.0f},
                                            Vec3{0.0f, 0.0f, 1.0f}),
                   0.0f, 1.0e-6f);
}

CG_TEST(software_shadow_pcss_contact_hardens_near_blockers)
{
    HeapAllocator heap;
    DepthBuffer nearBlocker = MakeShadowMap(heap, 64, 64);
    DepthBuffer farBlocker = MakeShadowMap(heap, 64, 64);
    CG_REQUIRE(nearBlocker.IsValid());
    CG_REQUIRE(farBlocker.IsValid());
    nearBlocker.Clear();
    farBlocker.Clear();
    for (cyber::core::i32 y = 28; y <= 35; ++y)
    {
        for (cyber::core::i32 x = 28; x <= 35; ++x)
        {
            nearBlocker.Row(y)[x] = 0.55f;
            farBlocker.Row(y)[x] = 0.95f;
        }
    }

    ShadowMapView shadow{};
    shadow.constantDepthBias = 0.0f;
    shadow.slopeDepthBias = 0.0f;
    shadow.filterRadius = kMaxSoftShadowFilterRadius;
    shadow.usePcss = true;
    shadow.perspectiveProjection = true;
    shadow.nearPlane = 1.0f;
    shadow.farPlane = 100.0f;
    shadow.verticalFovRadians = cyber::math::kHalfPi;
    shadow.sourceRadius = 0.5f;
    const Vec3 receiver{0.203125f, -0.203125f, 0.5f};
    const Vec3 normal{0.0f, 0.0f, 1.0f};
    const Vec3 toLight{0.0f, 0.0f, 1.0f};

    shadow.depth = &nearBlocker;
    const f32 nearVisibility = EvaluateShadowVisibility(shadow, receiver, normal, toLight);
    shadow.depth = &farBlocker;
    const f32 farVisibility = EvaluateShadowVisibility(shadow, receiver, normal, toLight);
    CG_EXPECT_NEAR(nearVisibility, 1.0f, 1.0e-6f);
    CG_EXPECT_TRUE(farVisibility >= 0.0f && farVisibility < nearVisibility);
}

CG_TEST(software_shadow_bias_clamping_and_out_of_frustum)
{
    HeapAllocator heap;
    DepthBuffer depth = MakeShadowMap(heap, 3, 3);
    CG_REQUIRE(depth.IsValid());
    depth.Clear();
    depth.TestAndWrite(1, 1, 0.5003f);

    ShadowMapView shadow{};
    shadow.depth = &depth;
    shadow.constantDepthBias = 0.0f;
    shadow.slopeDepthBias = 0.0f;
    shadow.filterRadius = 0;

    const Vec3 normal{0.0f, 0.0f, 1.0f};
    const Vec3 toLight{0.0f, 0.0f, 1.0f};
    CG_EXPECT_NEAR(EvaluateShadowVisibility(shadow, Vec3{0.0f, 0.0f, 0.5f}, normal, toLight), 0.0f,
                   1.0e-6f);

    shadow.constantDepthBias = 0.001f;
    CG_EXPECT_NEAR(EvaluateShadowVisibility(shadow, Vec3{0.0f, 0.0f, 0.5f}, normal, toLight), 1.0f,
                   1.0e-6f);

    shadow.constantDepthBias = 0.0f;
    shadow.slopeDepthBias = 0.002f;
    CG_EXPECT_NEAR(
        EvaluateShadowVisibility(shadow, Vec3{0.0f, 0.0f, 0.5f}, Vec3{1.0f, 0.0f, 0.0f}, toLight),
        1.0f, 1.0e-6f);

    shadow.slopeDepthBias = 0.0f;
    shadow.normalOffsetBias = 0.001f;
    CG_EXPECT_NEAR(EvaluateShadowVisibility(shadow, Vec3{0.0f, 0.0f, 0.5f}, normal, toLight), 1.0f,
                   1.0e-6f);

    shadow.normalOffsetBias = 0.0f;
    shadow.filterRadius = 100;
    const f32 clampedRadius =
        EvaluateShadowVisibility(shadow, Vec3{0.0f, 0.0f, 0.5f}, normal, toLight);
    CG_EXPECT_TRUE(std::isfinite(clampedRadius));
    CG_EXPECT_TRUE(clampedRadius >= 0.0f && clampedRadius <= 1.0f);
    CG_EXPECT_NEAR(EvaluateShadowVisibility(shadow, Vec3{2.0f, 0.0f, 0.5f}, normal, toLight), 1.0f,
                   1.0e-6f);
    CG_EXPECT_NEAR(EvaluateShadowVisibility(shadow,
                                            Vec3{std::numeric_limits<f32>::infinity(), 0.0f, 0.5f},
                                            normal, toLight),
                   1.0f, 1.0e-6f);
    CG_EXPECT_NEAR(EvaluateShadowVisibility(ShadowMapView{}, Vec3::Zero(), normal, toLight), 1.0f,
                   1.0e-6f);
}

CG_TEST(software_pbr_directional_shadow_only_attenuates_direct_light)
{
    HeapAllocator heap;
    DepthBuffer depth = MakeShadowMap(heap, 1, 1);
    CG_REQUIRE(depth.IsValid());
    depth.Clear(0.9f);

    PbrLightingData lighting{};
    lighting.eyePosition = Vec3{0.0f, 5.0f, 0.0f};
    lighting.ambientIrradiance = Vec3{0.2f, 0.2f, 0.2f};
    lighting.material.baseColor = Color{0.8f, 0.8f, 0.8f, 1.0f};
    lighting.directional[0].direction = Vec3{0.0f, -1.0f, 0.0f};
    lighting.directional[0].radiance = Vec3{2.0f, 2.0f, 2.0f};
    lighting.directionalCount = 1;

    Varyings surface{};
    surface.normal = Vec3{0.0f, 1.0f, 0.0f};
    surface.worldPosition = Vec3{0.0f, 0.0f, 0.2f};
    surface.color = Color{1.0f, 1.0f, 1.0f, 1.0f};
    const Color lit = PbrPixelShader(surface, &lighting);

    lighting.directional[0].shadow.depth = &depth;
    lighting.directional[0].shadow.constantDepthBias = 0.0f;
    lighting.directional[0].shadow.slopeDepthBias = 0.0f;
    lighting.directional[0].shadow.filterRadius = 0;
    const Color shadowed = PbrPixelShader(surface, &lighting);

    CG_EXPECT_TRUE(lit.r > shadowed.r);
    CG_EXPECT_TRUE(shadowed.r > 0.0f);
    CG_EXPECT_TRUE(shadowed.g > 0.0f);

    lighting.directional[0].shadow.depth = nullptr;
    lighting.directional[0].cascades.count = 1;
    lighting.directional[0].cascades.cascades[0].splitNear = 0.0f;
    lighting.directional[0].cascades.cascades[0].splitFar = 1.0f;
    lighting.directional[0].cascades.cascades[0].shadow = ShadowMapView{};
    lighting.directional[0].cascades.cascades[0].shadow.depth = &depth;
    lighting.directional[0].cascades.cascades[0].shadow.constantDepthBias = 0.0f;
    lighting.directional[0].cascades.cascades[0].shadow.slopeDepthBias = 0.0f;
    lighting.directional[0].cascades.cascades[0].shadow.filterRadius = 0;
    lighting.cameraForward = Vec3{0.0f, 0.0f, 1.0f};
    const Color cascadedShadow = PbrPixelShader(surface, &lighting);
    CG_EXPECT_NEAR(cascadedShadow.r, shadowed.r, 1.0e-6f);
}

CG_TEST(software_pbr_spot_shadow_only_attenuates_direct_light)
{
    HeapAllocator heap;
    DepthBuffer depth = MakeShadowMap(heap, 1, 1);
    CG_REQUIRE(depth.IsValid());
    depth.Clear(0.9f);

    PbrLightingData lighting{};
    lighting.eyePosition = Vec3{0.0f, 5.0f, 0.0f};
    lighting.ambientIrradiance = Vec3::Zero();
    lighting.material.baseColor = Color{0.8f, 0.8f, 0.8f, 1.0f};
    lighting.spots[0].position = Vec3{0.0f, 2.0f, 0.2f};
    lighting.spots[0].direction = Vec3{0.0f, -1.0f, 0.0f};
    lighting.spots[0].intensity = Vec3{100.0f, 100.0f, 100.0f};
    lighting.spotCount = 1;

    Varyings surface{};
    surface.normal = Vec3{0.0f, 1.0f, 0.0f};
    surface.worldPosition = Vec3{0.0f, 0.0f, 0.2f};
    surface.color = Color{1.0f, 1.0f, 1.0f, 1.0f};
    const Color lit = PbrPixelShader(surface, &lighting);

    lighting.spots[0].shadow.depth = &depth;
    lighting.spots[0].shadow.constantDepthBias = 0.0f;
    lighting.spots[0].shadow.slopeDepthBias = 0.0f;
    lighting.spots[0].shadow.filterRadius = 0;
    const Color shadowed = PbrPixelShader(surface, &lighting);
    CG_EXPECT_TRUE(lit.r > shadowed.r);
    CG_EXPECT_NEAR(shadowed.r, 0.0f, 1.0e-6f);
}

CG_TEST(software_pbr_point_shadow_uses_the_dominant_cube_face)
{
    HeapAllocator heap;
    DepthBuffer depth = MakeShadowMap(heap, 1, 1);
    CG_REQUIRE(depth.IsValid());
    depth.Clear(0.9f);

    PbrLightingData lighting{};
    lighting.eyePosition = Vec3{0.0f, 5.0f, 0.0f};
    lighting.ambientIrradiance = Vec3::Zero();
    lighting.material.baseColor = Color{0.8f, 0.8f, 0.8f, 1.0f};
    lighting.pointCount = 1;
    lighting.points[0].position = Vec3{0.0f, 2.0f, 0.2f};
    lighting.points[0].range = 10.0f;
    lighting.points[0].intensity = Vec3{100.0f, 100.0f, 100.0f};

    Varyings surface{};
    surface.normal = Vec3{0.0f, 1.0f, 0.0f};
    surface.worldPosition = Vec3{0.0f, 0.0f, 0.2f};
    surface.color = Color{1.0f, 1.0f, 1.0f, 1.0f};
    const Color lit = PbrPixelShader(surface, &lighting);

    lighting.points[0].shadow.faces[SelectShadowCubeFace(Vec3{0.0f, -2.0f, 0.0f})].depth = &depth;
    lighting.points[0]
        .shadow.faces[SelectShadowCubeFace(Vec3{0.0f, -2.0f, 0.0f})]
        .constantDepthBias = 0.0f;
    lighting.points[0].shadow.faces[SelectShadowCubeFace(Vec3{0.0f, -2.0f, 0.0f})].slopeDepthBias =
        0.0f;
    lighting.points[0].shadow.faces[SelectShadowCubeFace(Vec3{0.0f, -2.0f, 0.0f})].filterRadius = 0;
    const Color shadowed = PbrPixelShader(surface, &lighting);
    CG_EXPECT_TRUE(lit.r > 0.0f);
    CG_EXPECT_NEAR(shadowed.r, 0.0f, 1.0e-6f);
}

CG_TEST(software_point_shadow_blends_across_cube_face_seams)
{
    HeapAllocator heap;
    DepthBuffer blocked = MakeShadowMap(heap, 4, 4);
    DepthBuffer lit = MakeShadowMap(heap, 4, 4);
    CG_REQUIRE(blocked.IsValid());
    CG_REQUIRE(lit.IsValid());
    blocked.Clear(1.0f);
    lit.Clear(0.0f);

    PointShadowMapSet shadows{};
    shadows.faces[0].depth = &blocked;
    shadows.faces[2].depth = &lit;
    for (ShadowMapView& face : shadows.faces)
    {
        face.constantDepthBias = 0.0f;
        face.slopeDepthBias = 0.0f;
        face.filterRadius = 0;
    }

    const Vec3 position{0.0f, 0.0f, 0.5f};
    const Vec3 normal{0.0f, 0.0f, 1.0f};
    const Vec3 toLight{0.0f, 0.0f, 1.0f};
    const f32 towardX = EvaluatePointShadowVisibility(
        shadows, Vec3{1.0f, 0.99f, 0.0f}, position, normal, toLight);
    const f32 towardY = EvaluatePointShadowVisibility(
        shadows, Vec3{0.99f, 1.0f, 0.0f}, position, normal, toLight);
    const f32 awayFromSeam = EvaluatePointShadowVisibility(
        shadows, Vec3{1.0f, 0.5f, 0.0f}, position, normal, toLight);
    CG_EXPECT_NEAR(towardX, towardY, 0.01f);
    CG_EXPECT_TRUE(towardX > 0.45f && towardX < 0.55f);
    CG_EXPECT_NEAR(awayFromSeam, 0.0f, 1.0e-6f);
}

CG_TEST(software_directional_cascades_fit_frustum_and_stabilize_subtexel_motion)
{
    Camera camera{};
    camera.position = Vec3{2.0f, 3.0f, 4.0f};
    camera.yawRadians = 0.3f;
    camera.pitchRadians = -0.15f;
    camera.nearPlane = 0.1f;
    camera.farPlane = 400.0f;
    DirectionalShadowCascadeConfig config{};
    config.cascadeCount = 4;
    config.mapResolution = 1024;
    config.shadowDistance = 150.0f;
    config.depthPadding = 30.0f;

    DirectionalShadowCascadeSet cascades{};
    const Vec3 lightTravel = cyber::math::Normalize(Vec3{-0.4f, -1.0f, -0.2f});
    CG_REQUIRE(static_cast<bool>(
        BuildDirectionalShadowCascades(camera, 16.0f / 9.0f, lightTravel, config, cascades)));
    CG_EXPECT_EQ(cascades.count, cyber::core::u32{4});
    CG_EXPECT_NEAR(cascades.cascades[0].splitNear, camera.nearPlane, 1.0e-6f);

    const Vec3 forward = CameraForward(camera);
    const Vec3 right = CameraRight(camera);
    const Vec3 up = CameraUp(camera);
    const f32 tangentY = std::tan(camera.fovYRadians * 0.5f);
    const f32 aspect = 16.0f / 9.0f;
    for (cyber::core::u32 cascadeIndex = 0; cascadeIndex < cascades.count; ++cascadeIndex)
    {
        const DirectionalShadowCascade& cascade = cascades.cascades[cascadeIndex];
        CG_EXPECT_TRUE(cascade.splitFar > cascade.splitNear);
        const f32 halfHeightNear = cascade.splitNear * tangentY;
        const f32 halfHeightFar = cascade.splitFar * tangentY;
        for (cyber::core::u32 plane = 0; plane < 2; ++plane)
        {
            const f32 depth = plane == 0 ? cascade.splitNear : cascade.splitFar;
            const f32 halfHeight = plane == 0 ? halfHeightNear : halfHeightFar;
            const f32 halfWidth = halfHeight * aspect;
            const Vec3 center = camera.position + forward * depth;
            for (const f32 ySign : {-1.0f, 1.0f})
            {
                for (const f32 xSign : {-1.0f, 1.0f})
                {
                    const Vec3 corner =
                        center + right * (xSign * halfWidth) + up * (ySign * halfHeight);
                    const Vec3 ndc = cascade.shadow.clipFromWorld.TransformPoint(corner);
                    CG_EXPECT_TRUE(std::fabs(ndc.x) <= 1.0001f);
                    CG_EXPECT_TRUE(std::fabs(ndc.y) <= 1.0001f);
                    CG_EXPECT_TRUE(ndc.z >= -1.0e-4f && ndc.z <= 1.0001f);
                }
            }
        }
    }

    const f32 worldRadius = 1.0f / cascades.cascades[0].shadow.clipFromWorld[0].x;
    const f32 texelWorldSize = 2.0f * worldRadius / static_cast<f32>(config.mapResolution);
    camera.position.z += texelWorldSize * 0.01f;
    DirectionalShadowCascadeSet moved{};
    CG_REQUIRE(static_cast<bool>(
        BuildDirectionalShadowCascades(camera, aspect, lightTravel, config, moved)));
    CG_EXPECT_TRUE(MatrixNear(cascades.cascades[0].shadow.clipFromWorld,
                              moved.cascades[0].shadow.clipFromWorld, 2.0e-5f));
}

CG_TEST(software_directional_shadow_rendering_keeps_subtexel_camera_motion_stable)
{
    HeapAllocator heap;
    Camera camera{};
    camera.position = Vec3{0.0f, 3.0f, 6.0f};
    camera.pitchRadians = -0.22f;

    DirectionalShadowCascadeConfig config{};
    config.cascadeCount = 3;
    config.mapResolution = 64;
    config.shadowDistance = 32.0f;
    config.depthPadding = 24.0f;
    config.blendFraction = 0.15f;
    const Vec3 lightTravel =
        cyber::math::Normalize(Vec3{0.38f, -0.86f, 0.34f});
    DirectionalShadowCascadeSet first{};
    CG_REQUIRE(static_cast<bool>(
        BuildDirectionalShadowCascades(camera, 16.0f / 9.0f, lightTravel, config, first)));

    const Vec3 lightRight =
        cyber::math::Normalize(cyber::math::Cross(lightTravel, Vec3{0.0f, 1.0f, 0.0f}));
    const f32 worldRadius = 1.0f / first.cascades[0].shadow.clipFromWorld[0].x;
    const f32 texelWorldSize =
        2.0f * worldRadius / static_cast<f32>(config.mapResolution);
    camera.position += lightRight * (texelWorldSize * 0.001f);
    DirectionalShadowCascadeSet moved{};
    CG_REQUIRE(static_cast<bool>(
        BuildDirectionalShadowCascades(camera, 16.0f / 9.0f, lightTravel, config, moved)));

    constexpr cyber::core::i32 kResolution = 64;
    DepthBuffer firstDepth[kMaxDirectionalShadowCascades]{};
    DepthBuffer movedDepth[kMaxDirectionalShadowCascades]{};
    for (cyber::core::u32 index = 0; index < config.cascadeCount; ++index)
    {
        auto firstDepthResult =
            DepthBuffer::Create(DepthBufferDesc{kResolution, kResolution, 0}, heap);
        auto movedDepthResult =
            DepthBuffer::Create(DepthBufferDesc{kResolution, kResolution, 0}, heap);
        CG_REQUIRE(static_cast<bool>(firstDepthResult));
        CG_REQUIRE(static_cast<bool>(movedDepthResult));
        firstDepth[index] = std::move(firstDepthResult.Value());
        movedDepth[index] = std::move(movedDepthResult.Value());
    }

    const VertexInput scene[] = {
        VertexInput{Vec3{-30.0f, 0.0f, -30.0f}}, VertexInput{Vec3{30.0f, 0.0f, -30.0f}},
        VertexInput{Vec3{30.0f, 0.0f, 30.0f}},   VertexInput{Vec3{-30.0f, 0.0f, -30.0f}},
        VertexInput{Vec3{30.0f, 0.0f, 30.0f}},   VertexInput{Vec3{-30.0f, 0.0f, 30.0f}},
        VertexInput{Vec3{-1.0f, 0.0f, -2.0f}},    VertexInput{Vec3{1.0f, 0.0f, -2.0f}},
        VertexInput{Vec3{0.0f, 3.0f, -2.0f}},
    };
    PipelineStats firstStats[kMaxDirectionalShadowCascades]{};
    PipelineStats movedStats[kMaxDirectionalShadowCascades]{};
    const cyber::memory::Span<const VertexInput> sceneSpan(scene);
    PipelineStats invalidStats[1]{{}};
    invalidStats[0].pixels = 17;
    CG_EXPECT_FALSE(static_cast<bool>(RenderDirectionalShadowMaps(
        first, sceneSpan, cyber::memory::Span<DepthBuffer>(firstDepth, 1), heap, nullptr,
        CullMode::None, false,
        cyber::memory::Span<PipelineStats>(invalidStats, 1))));
    CG_EXPECT_EQ(invalidStats[0].pixels, cyber::core::u64{0});
    for (cyber::core::u32 index = 0; index < config.cascadeCount; ++index)
    {
        CG_EXPECT_TRUE(first.cascades[index].shadow.depth == nullptr);
    }

    CG_REQUIRE(static_cast<bool>(RenderDirectionalShadowMaps(
        first, sceneSpan,
        cyber::memory::Span<DepthBuffer>(firstDepth, kMaxDirectionalShadowCascades), heap,
        nullptr, CullMode::None, false,
        cyber::memory::Span<PipelineStats>(firstStats, kMaxDirectionalShadowCascades))));
    CG_REQUIRE(static_cast<bool>(RenderDirectionalShadowMaps(
        moved, sceneSpan,
        cyber::memory::Span<DepthBuffer>(movedDepth, kMaxDirectionalShadowCascades), heap,
        nullptr, CullMode::None, false,
        cyber::memory::Span<PipelineStats>(movedStats, kMaxDirectionalShadowCascades))));

    for (cyber::core::u32 cascade = 0; cascade < config.cascadeCount; ++cascade)
    {
        CG_EXPECT_TRUE(MatrixNear(first.cascades[cascade].shadow.clipFromWorld,
                                  moved.cascades[cascade].shadow.clipFromWorld, 2.0e-5f));
        CG_EXPECT_TRUE(first.cascades[cascade].shadow.depth == &firstDepth[cascade]);
        CG_EXPECT_TRUE(moved.cascades[cascade].shadow.depth == &movedDepth[cascade]);
        CG_EXPECT_TRUE(firstStats[cascade].pixels > 0);
        CG_EXPECT_EQ(firstStats[cascade].pixels, movedStats[cascade].pixels);
        for (cyber::core::i32 y = 0; y < kResolution; ++y)
        {
            for (cyber::core::i32 x = 0; x < kResolution; ++x)
            {
                CG_EXPECT_NEAR(firstDepth[cascade].Get(x, y), movedDepth[cascade].Get(x, y),
                               1.0e-6f);
            }
        }
    }
}

CG_TEST(software_directional_cascade_splits_blend_without_visibility_discontinuity)
{
    HeapAllocator heap;
    DepthBuffer nearDepth = MakeShadowMap(heap, 1, 1);
    DepthBuffer farDepth = MakeShadowMap(heap, 1, 1);
    CG_REQUIRE(nearDepth.IsValid());
    CG_REQUIRE(farDepth.IsValid());
    nearDepth.Clear(0.9f);
    farDepth.Clear(0.1f);

    DirectionalShadowCascadeSet cascades{};
    cascades.count = 2;
    cascades.blendFraction = 0.2f;
    cascades.cascades[0].splitNear = 1.0f;
    cascades.cascades[0].splitFar = 5.0f;
    cascades.cascades[0].shadow.depth = &nearDepth;
    cascades.cascades[0].shadow.constantDepthBias = 0.0f;
    cascades.cascades[0].shadow.slopeDepthBias = 0.0f;
    cascades.cascades[0].shadow.filterRadius = 0;
    cascades.cascades[1].splitNear = 5.0f;
    cascades.cascades[1].splitFar = 10.0f;
    cascades.cascades[1].shadow.depth = &farDepth;
    cascades.cascades[1].shadow.constantDepthBias = 0.0f;
    cascades.cascades[1].shadow.slopeDepthBias = 0.0f;
    cascades.cascades[1].shadow.filterRadius = 0;

    const Vec3 normal{0.0f, 0.0f, 1.0f};
    const Vec3 toLight{0.0f, 0.0f, 1.0f};
    const Vec3 receiver{0.0f, 0.0f, 0.5f};
    CG_EXPECT_NEAR(EvaluateDirectionalShadowVisibility(cascades, 3.9f, receiver, normal, toLight),
                   0.0f, 1.0e-6f);
    CG_EXPECT_NEAR(EvaluateDirectionalShadowVisibility(cascades, 4.4f, receiver, normal, toLight),
                   0.25f, 1.0e-6f);
    CG_EXPECT_NEAR(EvaluateDirectionalShadowVisibility(cascades, 4.9f, receiver, normal, toLight),
                   0.875f, 1.0e-6f);
    CG_EXPECT_NEAR(EvaluateDirectionalShadowVisibility(cascades, 10.1f, receiver, normal, toLight),
                   1.0f, 1.0e-6f);
}

CG_TEST(software_shadow_map_renderer_writes_reverse_z_occluder_depth)
{
    HeapAllocator heap;
    DepthBuffer depth = MakeShadowMap(heap, 32, 32);
    CG_REQUIRE(depth.IsValid());

    const VertexInput vertices[] = {
        VertexInput{Vec3{-0.8f, -0.8f, 0.5f}},
        VertexInput{Vec3{0.8f, -0.8f, 0.5f}},
        VertexInput{Vec3{0.0f, 0.8f, 0.5f}},
    };
    ShadowRenderDesc renderDesc{};
    renderDesc.cullMode = CullMode::None;
    PipelineStats stats{};
    CG_REQUIRE(static_cast<bool>(RenderShadowMap(renderDesc,
                                                 cyber::memory::Span<const VertexInput>(vertices),
                                                 depth, nullptr, heap, &stats)));
    CG_EXPECT_TRUE(stats.pixels > 0);
    CG_EXPECT_NEAR(depth.Get(16, 16), 0.5f, 1.0e-5f);

    ShadowMapView shadow{};
    shadow.depth = &depth;
    shadow.constantDepthBias = 0.0f;
    shadow.slopeDepthBias = 0.0f;
    shadow.filterRadius = 0;
    CG_EXPECT_NEAR(EvaluateShadowVisibility(shadow, Vec3{0.0f, 0.0f, 0.25f}, Vec3{0.0f, 0.0f, 1.0f},
                                            Vec3{0.0f, 0.0f, 1.0f}),
                   0.0f, 1.0e-6f);
    CG_EXPECT_NEAR(EvaluateShadowVisibility(shadow, Vec3{0.0f, 0.0f, 0.75f}, Vec3{0.0f, 0.0f, 1.0f},
                                            Vec3{0.0f, 0.0f, 1.0f}),
                   1.0f, 1.0e-6f);
}

CG_TEST(software_directional_cascade_builder_rejects_invalid_settings)
{
    Camera camera{};
    DirectionalShadowCascadeConfig config{};
    DirectionalShadowCascadeSet cascades{};
    config.cascadeCount = kMaxDirectionalShadowCascades + 1;
    CG_EXPECT_FALSE(static_cast<bool>(
        BuildDirectionalShadowCascades(camera, 1.0f, Vec3{0.0f, -1.0f, 0.0f}, config, cascades)));
    CG_EXPECT_EQ(cascades.count, cyber::core::u32{0});

    config = DirectionalShadowCascadeConfig{};
    CG_EXPECT_FALSE(static_cast<bool>(
        BuildDirectionalShadowCascades(camera, 1.0f, Vec3::Zero(), config, cascades)));
    CG_EXPECT_EQ(cascades.count, cyber::core::u32{0});
}

CG_TEST(software_spot_and_point_shadow_transforms_cover_light_directions)
{
    ShadowMapView spot{};
    const Vec3 spotPosition{2.0f, 3.0f, -1.0f};
    SpotShadowConfig spotConfig{};
    spotConfig.outerConeRadians = 0.4f;
    CG_REQUIRE(static_cast<bool>(
        BuildSpotShadowMap(spotPosition, Vec3{0.0f, -1.0f, 0.0f}, spotConfig, spot)));
    const Vec3 spotCenter =
        spot.clipFromWorld.TransformPoint(spotPosition + Vec3{0.0f, -5.0f, 0.0f});
    CG_EXPECT_NEAR(spotCenter.x, 0.0f, 1.0e-5f);
    CG_EXPECT_NEAR(spotCenter.y, 0.0f, 1.0e-5f);
    CG_EXPECT_TRUE(spotCenter.z >= 0.0f && spotCenter.z <= 1.0f);

    PointShadowMapSet point{};
    const Vec3 pointPosition{-3.0f, 2.0f, 1.0f};
    CG_REQUIRE(static_cast<bool>(BuildPointShadowMaps(pointPosition, PointShadowConfig{}, point)));
    const Vec3 directions[6] = {Vec3{1.0f, 0.0f, 0.0f}, Vec3{-1.0f, 0.0f, 0.0f},
                                Vec3{0.0f, 1.0f, 0.0f}, Vec3{0.0f, -1.0f, 0.0f},
                                Vec3{0.0f, 0.0f, 1.0f}, Vec3{0.0f, 0.0f, -1.0f}};
    for (cyber::core::u32 face = 0; face < 6; ++face)
    {
        const Vec3 center =
            point.faces[face].clipFromWorld.TransformPoint(pointPosition + directions[face] * 5.0f);
        CG_EXPECT_NEAR(center.x, 0.0f, 1.0e-5f);
        CG_EXPECT_NEAR(center.y, 0.0f, 1.0e-5f);
        CG_EXPECT_TRUE(center.z >= 0.0f && center.z <= 1.0f);
    }

    CG_EXPECT_EQ(SelectShadowCubeFace(Vec3{1.0f, 0.8f, 0.0f}), cyber::core::u32{0});
    CG_EXPECT_EQ(SelectShadowCubeFace(Vec3{-1.0f, 0.0f, 0.0f}), cyber::core::u32{1});
    CG_EXPECT_EQ(SelectShadowCubeFace(Vec3{0.0f, 2.0f, 0.0f}), cyber::core::u32{2});
    CG_EXPECT_EQ(SelectShadowCubeFace(Vec3{0.0f, -2.0f, 0.0f}), cyber::core::u32{3});
    CG_EXPECT_EQ(SelectShadowCubeFace(Vec3{0.0f, 0.0f, 2.0f}), cyber::core::u32{4});
    CG_EXPECT_EQ(SelectShadowCubeFace(Vec3{0.0f, 0.0f, -2.0f}), cyber::core::u32{5});
}

CG_TEST(software_shadow_atlas_allocates_guard_bands_and_reuses_released_regions)
{
    ShadowAtlas atlas{};
    CG_REQUIRE(static_cast<bool>(InitializeShadowAtlas(2048, 1024, atlas)));
    ShadowAtlasAllocation first{};
    ShadowAtlasAllocation second{};
    CG_REQUIRE(static_cast<bool>(AllocateShadowAtlas(atlas, 512, 4, first)));
    CG_REQUIRE(static_cast<bool>(AllocateShadowAtlas(atlas, 256, 4, second)));
    CG_EXPECT_EQ(first.size, cyber::core::u32{1024});
    CG_EXPECT_EQ(first.innerOffset, cyber::core::u32{4});
    CG_EXPECT_EQ(first.innerResolution, cyber::core::u32{512});
    CG_EXPECT_TRUE(first.x != second.x || first.y != second.y);
    CG_EXPECT_TRUE(ReleaseShadowAtlas(atlas, second));
    CG_EXPECT_FALSE(ReleaseShadowAtlas(atlas, second));
    ShadowAtlasAllocation reused{};
    CG_REQUIRE(static_cast<bool>(AllocateShadowAtlas(atlas, 256, 4, reused)));
    CG_EXPECT_EQ(reused.x, second.x);
    CG_EXPECT_EQ(reused.y, second.y);
    CG_EXPECT_FALSE(ReleaseShadowAtlas(atlas, second));
    CG_EXPECT_TRUE(ReleaseShadowAtlas(atlas, first));

    cyber::core::u32 resolution = 0;
    CG_REQUIRE(static_cast<bool>(SelectShadowResolution(0.0f, 128, 1024, resolution)));
    CG_EXPECT_EQ(resolution, cyber::core::u32{128});
    CG_REQUIRE(static_cast<bool>(SelectShadowResolution(0.5f, 128, 1024, resolution)));
    CG_EXPECT_EQ(resolution, cyber::core::u32{512});
    CG_REQUIRE(static_cast<bool>(SelectShadowResolution(1.0f, 128, 1024, resolution)));
    CG_EXPECT_EQ(resolution, cyber::core::u32{1024});
    CG_EXPECT_NEAR(ShadowDepthDebugColor(0.25f).r, 0.25f, 1.0e-6f);
}

CG_TEST(software_pbr_shadow_cascade_visualization_selects_stable_debug_color)
{
    PbrLightingData lighting{};
    lighting.eyePosition = Vec3::Zero();
    lighting.cameraForward = Vec3{0.0f, 0.0f, 1.0f};
    lighting.directionalCount = 1;
    lighting.visualizeShadowCascades = true;
    lighting.directional[0].cascades.count = 2;
    lighting.directional[0].cascades.cascades[0].splitNear = 0.0f;
    lighting.directional[0].cascades.cascades[0].splitFar = 5.0f;
    lighting.directional[0].cascades.cascades[1].splitNear = 5.0f;
    lighting.directional[0].cascades.cascades[1].splitFar = 10.0f;
    Varyings surface{};
    surface.worldPosition = Vec3{0.0f, 0.0f, 2.0f};
    surface.normal = Vec3{0.0f, 1.0f, 0.0f};
    const Color color = PbrPixelShader(surface, &lighting);
    CG_EXPECT_TRUE(cyber::math::ApproxEqual(Vec3{color.r, color.g, color.b},
                                            Vec3{DirectionalShadowCascadeDebugColor(0).r,
                                                 DirectionalShadowCascadeDebugColor(0).g,
                                                 DirectionalShadowCascadeDebugColor(0).b}));
}

CG_TEST(software_shadow_scene_renders_a_measurable_soft_shadow)
{
    HeapAllocator heap;
    constexpr cyber::core::u32 kVertexCount = 6 + 3 * 36;
    VertexInput vertices[kVertexCount]{};
    cyber::core::u32 vertexCount = 0;
    const Color groundColor{0.78f, 0.8f, 0.84f, 1.0f};
    const Vec3 ground[4] = {Vec3{-5.0f, 0.0f, -5.0f}, Vec3{-5.0f, 0.0f, 5.0f},
                            Vec3{5.0f, 0.0f, 5.0f}, Vec3{5.0f, 0.0f, -5.0f}};
    AppendTriangle(vertices, vertexCount, ground[0], ground[1], ground[2], Vec3{0.0f, 1.0f, 0.0f},
                   groundColor);
    AppendTriangle(vertices, vertexCount, ground[0], ground[2], ground[3], Vec3{0.0f, 1.0f, 0.0f},
                   groundColor);
    AppendBox(vertices, vertexCount, Vec3{0.0f, 0.72f, 0.0f}, Vec3{0.6f, 0.72f, 0.55f},
              Color{0.88f, 0.32f, 0.18f, 1.0f});
    AppendBox(vertices, vertexCount, Vec3{-1.45f, 0.42f, -0.55f}, Vec3{0.38f, 0.42f, 0.42f},
              Color{0.22f, 0.46f, 0.82f, 1.0f});
    AppendBox(vertices, vertexCount, Vec3{1.35f, 0.34f, 0.9f}, Vec3{0.34f, 0.34f, 0.34f},
              Color{0.3f, 0.72f, 0.42f, 1.0f});
    CG_EXPECT_EQ(vertexCount, kVertexCount);

    constexpr cyber::core::i32 kReferenceShadowResolution = 512;
    constexpr cyber::core::i32 kHighShadowResolution = 2048;
    DepthBuffer referenceShadowDepth =
        MakeShadowMap(heap, kReferenceShadowResolution, kReferenceShadowResolution);
    DepthBuffer highShadowDepth = MakeShadowMap(heap, kHighShadowResolution, kHighShadowResolution);
    Framebuffer referenceColor = MakeScratchColor(heap, kShadowSceneWidth, kShadowSceneHeight);
    Framebuffer softColor = MakeScratchColor(heap, kShadowSceneWidth, kShadowSceneHeight);
    Framebuffer unshadowedColor = MakeScratchColor(heap, kShadowSceneWidth, kShadowSceneHeight);
    Framebuffer shadowedHighRes = MakeScratchColor(heap, kShadowRenderWidth, kShadowRenderHeight);
    DepthBuffer sceneDepth = MakeShadowMap(heap, kShadowRenderWidth, kShadowRenderHeight);
    CG_REQUIRE(referenceShadowDepth.IsValid());
    CG_REQUIRE(highShadowDepth.IsValid());
    CG_REQUIRE(referenceColor.IsValid());
    CG_REQUIRE(softColor.IsValid());
    CG_REQUIRE(unshadowedColor.IsValid());
    CG_REQUIRE(shadowedHighRes.IsValid());
    CG_REQUIRE(sceneDepth.IsValid());

    const Vec3 lightTravel = cyber::math::Normalize(Vec3{-0.75f, -0.7f, -0.45f});
    const Vec3 sceneCenter{0.0f, 0.7f, 0.0f};
    const Vec3 lightEye = sceneCenter - lightTravel * 12.0f;
    const Mat4 lightView = Mat4::LookAt(lightEye, sceneCenter, Vec3{0.0f, 1.0f, 0.0f});
    const Mat4 lightProjection = Mat4::OrthographicReverseZ(-6.0f, 6.0f, -6.0f, 6.0f, 0.1f, 40.0f);

    ShadowRenderDesc shadowRender{};
    shadowRender.clipFromWorld = lightProjection * lightView;
    shadowRender.cullMode = CullMode::None;
    PipelineStats referenceShadowStats{};
    CG_REQUIRE(static_cast<bool>(RenderShadowMap(
        shadowRender, cyber::memory::Span<const VertexInput>(vertices, vertexCount),
        referenceShadowDepth, nullptr, heap, &referenceShadowStats)));
    CG_EXPECT_TRUE(referenceShadowStats.pixels > 0);
    PipelineStats highShadowStats{};
    CG_REQUIRE(static_cast<bool>(
        RenderShadowMap(shadowRender, cyber::memory::Span<const VertexInput>(vertices, vertexCount),
                        highShadowDepth, nullptr, heap, &highShadowStats)));
    CG_EXPECT_TRUE(highShadowStats.pixels > referenceShadowStats.pixels);

    const Vec3 eye{3.9f, 3.2f, 5.3f};
    const Vec3 target{0.0f, 0.55f, 0.0f};
    const Vec3 cameraForward = cyber::math::Normalize(target - eye);
    const Mat4 cameraView = Mat4::LookAt(eye, target, Vec3{0.0f, 1.0f, 0.0f});
    const Mat4 cameraProjection = Mat4::PerspectiveReverseZ(
        0.9f, static_cast<f32>(kShadowSceneWidth) / static_cast<f32>(kShadowSceneHeight), 0.1f,
        40.0f);

    ShadowMapView shadow{};
    shadow.depth = &referenceShadowDepth;
    shadow.clipFromWorld = shadowRender.clipFromWorld;
    shadow.constantDepthBias = 0.0005f;
    shadow.slopeDepthBias = 0.002f;
    shadow.normalOffsetBias = 0.001f;
    shadow.filterRadius = 8;
    shadow.softFilter = true;
    shadow.usePcss = true;
    shadow.nearPlane = 0.1f;
    shadow.farPlane = 40.0f;
    shadow.worldUnitsPerTexel = 12.0f / static_cast<f32>(kReferenceShadowResolution);
    shadow.sourceAngularRadiusRadians = 0.02f;
    const Vec3 normal{0.0f, 1.0f, 0.0f};
    const Vec3 toLight = -lightTravel;
    CG_EXPECT_TRUE(EvaluateShadowVisibility(shadow, Vec3{-3.0f, 0.0f, -3.0f}, normal, toLight) >
                   0.95f);
    const Mat4 viewProjection = cameraProjection * cameraView;
    CG_REQUIRE(RenderShadowAcceptanceScene(heap, vertices, vertexCount, viewProjection, eye,
                                           cameraForward, lightTravel, shadow, shadowedHighRes,
                                           sceneDepth));
    CG_REQUIRE(DownsampleShadowScene(shadowedHighRes, referenceColor));
    shadow.depth = nullptr;
    CG_REQUIRE(RenderShadowAcceptanceScene(heap, vertices, vertexCount, viewProjection, eye,
                                           cameraForward, lightTravel, shadow, shadowedHighRes,
                                           sceneDepth));
    CG_REQUIRE(DownsampleShadowScene(shadowedHighRes, unshadowedColor));

    shadow.depth = &highShadowDepth;
    shadow.filterRadius = 16;
    shadow.worldUnitsPerTexel = 12.0f / static_cast<f32>(kHighShadowResolution);
    shadow.softFilter = true;
    CG_REQUIRE(RenderShadowAcceptanceScene(heap, vertices, vertexCount, viewProjection, eye,
                                           cameraForward, lightTravel, shadow, shadowedHighRes,
                                           sceneDepth));
    CG_REQUIRE(DownsampleShadowScene(shadowedHighRes, softColor));

    cyber::core::u32 changedPixels = 0;
    f32 maximumDifference = 0.0f;
    for (cyber::core::i32 y = 0; y < kShadowSceneHeight; ++y)
    {
        for (cyber::core::i32 x = 0; x < kShadowSceneWidth; ++x)
        {
            const Color withShadow = softColor.GetPixel(x, y);
            const Color withoutShadow = unshadowedColor.GetPixel(x, y);
            const f32 difference = std::fabs(withShadow.r - withoutShadow.r) +
                                   std::fabs(withShadow.g - withoutShadow.g) +
                                   std::fabs(withShadow.b - withoutShadow.b);
            if (difference > 0.04f)
            {
                ++changedPixels;
            }
            maximumDifference = std::max(maximumDifference, difference);
        }
    }
    CG_EXPECT_TRUE(changedPixels > 100);
    CG_EXPECT_TRUE(maximumDifference > 0.2f);

    CG_REQUIRE(static_cast<bool>(cyber::software::WriteImage(
        "phase28_shadow_256_unshadowed.ppm", unshadowedColor, heap,
        cyber::software::ImageWriteDesc{cyber::software::ImageFormat::Ppm, true})));
    CG_REQUIRE(static_cast<bool>(cyber::software::WriteImage(
        "phase28_shadow_256_reference_512map.ppm", referenceColor, heap,
        cyber::software::ImageWriteDesc{cyber::software::ImageFormat::Ppm, true})));
    CG_REQUIRE(static_cast<bool>(cyber::software::WriteImage(
        "phase28_shadow_256_soft_2048map.ppm", softColor, heap,
        cyber::software::ImageWriteDesc{cyber::software::ImageFormat::Ppm, true})));
    Framebuffer shadowDifference = MakeScratchColor(heap, kShadowSceneWidth, kShadowSceneHeight);
    CG_REQUIRE(shadowDifference.IsValid());
    for (cyber::core::i32 y = 0; y < kShadowSceneHeight; ++y)
    {
        for (cyber::core::i32 x = 0; x < kShadowSceneWidth; ++x)
        {
            const Color withShadow = softColor.GetPixel(x, y);
            const Color withoutShadow = unshadowedColor.GetPixel(x, y);
            const f32 difference = std::fabs(withShadow.r - withoutShadow.r) +
                                   std::fabs(withShadow.g - withoutShadow.g) +
                                   std::fabs(withShadow.b - withoutShadow.b);
            const f32 amplified = std::min(difference * 4.0f, 1.0f);
            shadowDifference.SetPixel(x, y, Color{amplified, amplified, amplified, 1.0f});
        }
    }
    CG_REQUIRE(static_cast<bool>(cyber::software::WriteImage(
        "phase28_shadow_256_soft_difference.ppm", shadowDifference, heap,
        cyber::software::ImageWriteDesc{cyber::software::ImageFormat::Ppm, true})));

    Framebuffer fogShadowed = MakeScratchColor(heap, kShadowSceneWidth, kShadowSceneHeight);
    Framebuffer fogUnshadowed = MakeScratchColor(heap, kShadowSceneWidth, kShadowSceneHeight);
    Framebuffer fogHighRes = MakeScratchColor(heap, kFogRenderWidth, kFogRenderHeight);
    DepthBuffer fogDepth = MakeShadowMap(heap, kFogRenderWidth, kFogRenderHeight);
    CG_REQUIRE(fogShadowed.IsValid());
    CG_REQUIRE(fogUnshadowed.IsValid());
    CG_REQUIRE(fogHighRes.IsValid());
    CG_REQUIRE(fogDepth.IsValid());
    shadow.depth = &highShadowDepth;
    CG_REQUIRE(RenderShadowAcceptanceScene(heap, vertices, vertexCount, viewProjection, eye,
                                           cameraForward, lightTravel, shadow, fogHighRes,
                                           fogDepth, true));
    CG_REQUIRE(DownsampleShadowScene(fogHighRes, fogShadowed, kFogSceneSupersample));
    CG_REQUIRE(static_cast<bool>(cyber::software::WriteImage(
        "phase28_shadowed_volumetric_fog.ppm", fogShadowed, heap,
        cyber::software::ImageWriteDesc{cyber::software::ImageFormat::Ppm, true})));
    shadow.depth = nullptr;
    CG_REQUIRE(RenderShadowAcceptanceScene(heap, vertices, vertexCount, viewProjection, eye,
                                           cameraForward, lightTravel, shadow, fogHighRes,
                                           fogDepth, true));
    CG_REQUIRE(DownsampleShadowScene(fogHighRes, fogUnshadowed, kFogSceneSupersample));
    CG_REQUIRE(static_cast<bool>(cyber::software::WriteImage(
        "phase28_unshadowed_volumetric_fog.ppm", fogUnshadowed, heap,
        cyber::software::ImageWriteDesc{cyber::software::ImageFormat::Ppm, true})));
    cyber::core::u32 fogChangedPixels = 0;
    for (cyber::core::i32 y = 0; y < kShadowSceneHeight; ++y)
    {
        for (cyber::core::i32 x = 0; x < kShadowSceneWidth; ++x)
        {
            const Color withShadow = fogShadowed.GetPixel(x, y);
            const Color withoutShadow = fogUnshadowed.GetPixel(x, y);
            const f32 difference = std::fabs(withShadow.r - withoutShadow.r) +
                                   std::fabs(withShadow.g - withoutShadow.g) +
                                   std::fabs(withShadow.b - withoutShadow.b);
            if (difference > 0.04f)
            {
                ++fogChangedPixels;
            }
        }
    }
    CG_EXPECT_TRUE(fogChangedPixels > 50);

    Camera cascadeCamera{};
    cascadeCamera.position = eye;
    cascadeCamera.yawRadians = std::atan2(cameraForward.x, -cameraForward.z);
    cascadeCamera.pitchRadians = std::asin(std::clamp(cameraForward.y, -1.0f, 1.0f));
    cascadeCamera.fovYRadians = 0.9f;
    cascadeCamera.nearPlane = 0.1f;
    cascadeCamera.farPlane = 40.0f;
    DirectionalShadowCascadeConfig cascadeConfig{};
    cascadeConfig.cascadeCount = kMaxDirectionalShadowCascades;
    cascadeConfig.mapResolution = 512;
    cascadeConfig.shadowDistance = 40.0f;
    cascadeConfig.depthPadding = 24.0f;
    cascadeConfig.blendFraction = 0.12f;
    cascadeConfig.filterRadius = 4;
    cascadeConfig.sourceAngularRadiusRadians = 0.008f;
    DirectionalShadowCascadeSet cascades{};
    CG_REQUIRE(static_cast<bool>(BuildDirectionalShadowCascades(
        cascadeCamera, static_cast<f32>(kShadowSceneWidth) / kShadowSceneHeight, lightTravel,
        cascadeConfig, cascades)));
    DepthBuffer cascadeDepth[kMaxDirectionalShadowCascades]{};
    PipelineStats cascadeStats[kMaxDirectionalShadowCascades]{};
    for (cyber::core::u32 index = 0; index < cascades.count; ++index)
    {
        const auto mapResolution = static_cast<cyber::core::i32>(cascadeConfig.mapResolution);
        auto depth = DepthBuffer::Create(DepthBufferDesc{mapResolution, mapResolution, 0}, heap);
        CG_REQUIRE(static_cast<bool>(depth));
        cascadeDepth[index] = std::move(depth.Value());
    }
    CG_REQUIRE(static_cast<bool>(RenderDirectionalShadowMaps(
        cascades, cyber::memory::Span<const VertexInput>(vertices, vertexCount),
        cyber::memory::Span<DepthBuffer>(cascadeDepth, cascades.count), heap, nullptr,
        CullMode::None, false, cyber::memory::Span<PipelineStats>(cascadeStats, cascades.count))));
    for (cyber::core::u32 index = 0; index < cascades.count; ++index)
    {
        CG_EXPECT_TRUE(cascadeStats[index].pixels > 0);
    }

    Framebuffer cascadeColor = MakeScratchColor(heap, kShadowSceneWidth, kShadowSceneHeight);
    DepthBuffer cascadeSceneDepth = MakeShadowMap(heap, kShadowSceneWidth, kShadowSceneHeight);
    CG_REQUIRE(cascadeColor.IsValid());
    CG_REQUIRE(cascadeSceneDepth.IsValid());
    CG_REQUIRE(RenderShadowAcceptanceScene(
        heap, vertices, vertexCount, viewProjection, eye, cameraForward, lightTravel,
        ShadowMapView{}, cascadeColor, cascadeSceneDepth, false, &cascades));
    cyber::core::u32 cascadedShadowPixels = 0;
    f32 maximumCascadeShadowDifference = 0.0f;
    for (cyber::core::i32 y = 0; y < kShadowSceneHeight; ++y)
    {
        for (cyber::core::i32 x = 0; x < kShadowSceneWidth; ++x)
        {
            const Color linear = cascadeColor.GetPixel(x, y);
            const Color mapped = ToneMapAces(linear);
            cascadeColor.SetPixel(x, y, mapped);
            const Color unshadowed = unshadowedColor.GetPixel(x, y);
            const f32 difference = std::fabs(mapped.r - unshadowed.r) +
                                   std::fabs(mapped.g - unshadowed.g) +
                                   std::fabs(mapped.b - unshadowed.b);
            if (difference > 0.04f)
            {
                ++cascadedShadowPixels;
            }
            maximumCascadeShadowDifference =
                std::max(maximumCascadeShadowDifference, difference);
        }
    }
    CG_EXPECT_TRUE(cascadedShadowPixels > 100);
    CG_EXPECT_TRUE(maximumCascadeShadowDifference > 0.2f);
    CG_REQUIRE(static_cast<bool>(cyber::software::WriteImage(
        "phase28_cascaded_shadow_acceptance.ppm", cascadeColor, heap,
        cyber::software::ImageWriteDesc{cyber::software::ImageFormat::Ppm, true})));

    constexpr cyber::core::i32 kDepthPreviewResolution = 1024;
    Framebuffer depthDebug =
        MakeScratchColor(heap, kDepthPreviewResolution, kDepthPreviewResolution);
    CG_REQUIRE(depthDebug.IsValid());
    for (cyber::core::i32 y = 0; y < kDepthPreviewResolution; ++y)
    {
        for (cyber::core::i32 x = 0; x < kDepthPreviewResolution; ++x)
        {
            depthDebug.SetPixel(x, y, ShadowDepthDebugColor(highShadowDepth.Get(x * 2, y * 2)));
        }
    }
    CG_REQUIRE(static_cast<bool>(cyber::software::WriteImage(
        "phase28_shadow_2048_depth.ppm", depthDebug, heap,
        cyber::software::ImageWriteDesc{cyber::software::ImageFormat::Ppm, true})));
}
