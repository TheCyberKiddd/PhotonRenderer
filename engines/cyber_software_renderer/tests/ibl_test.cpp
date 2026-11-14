#include <cyber/jobs/job_system.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/platform/filesystem.h>
#include <cyber/software/ibl.h>
#include <cyber/software/pipeline.h>
#include <cyber/test/test.h>

#include <cmath>
#include <cstdio>
#include <utility>

using namespace cyber::software;
using namespace cyber::memory;
using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::math::Color;
using cyber::math::Vec2;
using cyber::math::Vec3;

namespace
{

constexpr f32 kPi = cyber::math::kPi;

// Returns an invalid cubemap on failure, which every caller then reports
// through CG_REQUIRE, because CG_REQUIRE itself cannot return a value.
Cubemap MakeCubemap(HeapAllocator& allocator, i32 faceSize, u32 mipLevels = 0)
{
    CubemapDesc desc{};
    desc.faceSize = faceSize;
    desc.mipLevels = mipLevels;
    auto created = Cubemap::Create(desc, allocator);
    if (created.IsError())
    {
        return Cubemap{};
    }
    return std::move(created).Value();
}

void FillConstant(Cubemap& cubemap, const Color& color)
{
    cubemap.Clear(color);
}

Vec3 SampleRgb(const Cubemap& cubemap, const Vec3& direction, f32 level)
{
    const Color texel = SampleCubemapLevel(cubemap, direction, level);
    return Vec3{texel.r, texel.g, texel.b};
}

Vec3 FibonacciDirection(u32 index, u32 count)
{
    const f32 y = 1.0f - 2.0f * (static_cast<f32>(index) + 0.5f) / static_cast<f32>(count);
    const f32 radius = std::sqrt(std::max(0.0f, 1.0f - y * y));
    const f32 phi = 2.0f * kPi * static_cast<f32>(index) * 0.61803398875f;
    return Vec3{radius * std::cos(phi), y, radius * std::sin(phi)};
}

struct ProbeCaptureScene
{
    Span<const Vertex> vertices;
    HeapAllocator* allocator = nullptr;
    u32 renderCount = 0;
    i32 captureWidth = 0;
    u32 failOnFace = kCubeFaceCount;
    u32 invalidateTargetOnFace = kCubeFaceCount;
};

cyber::core::Status<cyber::core::Error>
RenderProbeCaptureFace(const ReflectionProbeCaptureView& view, Framebuffer& color,
                       DepthBuffer& depth, void* userData) noexcept
{
    auto& scene = *static_cast<ProbeCaptureScene*>(userData);
    ++scene.renderCount;
    scene.captureWidth = color.Width();
    if (static_cast<u32>(view.face) == scene.failOnFace)
    {
        return cyber::core::Status<cyber::core::Error>::Err(
            cyber::core::Error{cyber::core::ErrorCode::Internal, "test capture callback failure"});
    }
    if (static_cast<u32>(view.face) == scene.invalidateTargetOnFace)
    {
        color = Framebuffer{};
        return cyber::core::Status<cyber::core::Error>::Ok();
    }

    PipelineDesc pipeline{};
    pipeline.view = view.view;
    pipeline.projection = view.projection;
    pipeline.viewport =
        Viewport{0.0f, 0.0f, static_cast<f32>(color.Width()), static_cast<f32>(color.Height())};
    pipeline.cullMode = CullMode::None;
    return DrawTriangles(pipeline, scene.vertices, color, depth, nullptr, *scene.allocator);
}

}  // namespace

CG_TEST(ibl_cubemap_create_rejects_invalid_descriptors)
{
    HeapAllocator allocator;

    CubemapDesc tooSmall{};
    tooSmall.faceSize = 0;
    CG_EXPECT_TRUE(Cubemap::Create(tooSmall, allocator).IsError());

    CubemapDesc tooLarge{};
    tooLarge.faceSize = kMaxCubemapFaceSize + 1;
    CG_EXPECT_TRUE(Cubemap::Create(tooLarge, allocator).IsError());

    CubemapDesc tooManyLevels{};
    tooManyLevels.faceSize = 4;
    tooManyLevels.mipLevels = 9;
    CG_EXPECT_TRUE(Cubemap::Create(tooManyLevels, allocator).IsError());

    Cubemap cubemap = MakeCubemap(allocator, 8);

    CG_REQUIRE(cubemap.IsValid());
    CG_EXPECT_TRUE(cubemap.IsValid());
    CG_EXPECT_EQ(cubemap.FaceSize(), i32{8});
    CG_EXPECT_EQ(cubemap.MipLevels(), u32{4});
    CG_EXPECT_EQ(cubemap.MipSize(0), i32{8});
    CG_EXPECT_EQ(cubemap.MipSize(3), i32{1});
    CG_EXPECT_EQ(cubemap.MipSize(4), i32{0});
}

CG_TEST(ibl_cubemap_texel_access_is_bounded)
{
    HeapAllocator allocator;
    Cubemap cubemap = MakeCubemap(allocator, 4);
    CG_REQUIRE(cubemap.IsValid());

    cubemap.SetTexel(2, 0, 1, 3, Color{0.25f, 0.5f, 0.75f, 1.0f});
    const Color read = cubemap.GetTexel(2, 0, 1, 3);
    CG_EXPECT_NEAR(read.r, 0.25f, 1.0e-6f);
    CG_EXPECT_NEAR(read.g, 0.5f, 1.0e-6f);
    CG_EXPECT_NEAR(read.b, 0.75f, 1.0e-6f);

    // Out-of-range writes are ignored and out-of-range reads are black.
    cubemap.SetTexel(6, 0, 0, 0, Color{1.0f, 1.0f, 1.0f, 1.0f});
    cubemap.SetTexel(0, 0, -1, 0, Color{1.0f, 1.0f, 1.0f, 1.0f});
    cubemap.SetTexel(0, 9, 0, 0, Color{1.0f, 1.0f, 1.0f, 1.0f});
    CG_EXPECT_NEAR(cubemap.GetTexel(6, 0, 0, 0).r, 0.0f, 1.0e-6f);
    CG_EXPECT_NEAR(cubemap.GetTexel(0, 0, -1, 0).r, 0.0f, 1.0e-6f);
    CG_EXPECT_NEAR(cubemap.GetTexel(0, 0, 4, 0).r, 0.0f, 1.0e-6f);
}

CG_TEST(ibl_cube_addressing_round_trips)
{
    for (u32 index = 0; index < 4096; ++index)
    {
        const Vec3 direction = FibonacciDirection(index, 4096);
        u32 face = 0;
        f32 u = 0.0f;
        f32 v = 0.0f;
        CubeFaceUvFromDirection(direction, face, u, v);
        CG_REQUIRE(face < kCubeFaceCount);
        CG_REQUIRE(u >= -1.0e-5f && u <= 1.0f + 1.0e-5f);
        CG_REQUIRE(v >= -1.0e-5f && v <= 1.0f + 1.0e-5f);
        const Vec3 rebuilt = CubeDirectionFromFaceUv(face, u, v);
        CG_REQUIRE_NEAR(rebuilt.x, direction.x, 1.0e-4f);
        CG_REQUIRE_NEAR(rebuilt.y, direction.y, 1.0e-4f);
        CG_REQUIRE_NEAR(rebuilt.z, direction.z, 1.0e-4f);
    }

    // A degenerate direction resolves to the center of +X rather than a NaN.
    u32 face = 7;
    f32 u = -1.0f;
    f32 v = -1.0f;
    CubeFaceUvFromDirection(Vec3::Zero(), face, u, v);
    CG_EXPECT_EQ(face, u32{0});
    CG_EXPECT_NEAR(u, 0.5f, 1.0e-6f);
    CG_EXPECT_NEAR(v, 0.5f, 1.0e-6f);
}

CG_TEST(ibl_cube_texel_solid_angles_cover_the_sphere)
{
    constexpr i32 kSize = 24;
    f32 total = 0.0f;
    for (u32 face = 0; face < kCubeFaceCount; ++face)
    {
        for (i32 y = 0; y < kSize; ++y)
        {
            for (i32 x = 0; x < kSize; ++x)
            {
                const f32 u = (static_cast<f32>(x) + 0.5f) / static_cast<f32>(kSize);
                const f32 v = (static_cast<f32>(y) + 0.5f) / static_cast<f32>(kSize);
                total += CubeTexelSolidAngle(kSize, u, v);
            }
        }
    }
    CG_EXPECT_NEAR(total, 4.0f * kPi, 1.0e-3f);
    CG_EXPECT_NEAR(CubeTexelSolidAngle(0, 0.5f, 0.5f), 0.0f, 1.0e-6f);
}

CG_TEST(ibl_cubemap_filtering_preserves_a_constant)
{
    HeapAllocator allocator;
    Cubemap cubemap = MakeCubemap(allocator, 16);
    CG_REQUIRE(cubemap.IsValid());
    FillConstant(cubemap, Color{0.4f, 0.6f, 0.8f, 1.0f});
    cubemap.GenerateMips();

    // Every level, every direction, including the cube edges and corners that
    // a clamped bilinear fetch would darken.
    for (u32 level = 0; level < cubemap.MipLevels(); ++level)
    {
        for (u32 index = 0; index < 512; ++index)
        {
            const Vec3 sampled =
                SampleRgb(cubemap, FibonacciDirection(index, 512), static_cast<f32>(level));
            CG_REQUIRE_NEAR(sampled.x, 0.4f, 1.0e-5f);
            CG_REQUIRE_NEAR(sampled.y, 0.6f, 1.0e-5f);
            CG_REQUIRE_NEAR(sampled.z, 0.8f, 1.0e-5f);
        }
    }

    const Vec3 corner = SampleRgb(cubemap, Vec3{1.0f, 1.0f, 1.0f}, 0.0f);
    CG_EXPECT_NEAR(corner.x, 0.4f, 1.0e-5f);
}

CG_TEST(ibl_equirectangular_projection_maps_poles_and_constants)
{
    HeapAllocator allocator;

    TextureDesc desc{};
    desc.width = 32;
    desc.height = 16;
    desc.format = TextureFormat::RGBA32F;
    desc.generateMips = false;
    auto created = Texture::Create(desc, allocator);
    CG_REQUIRE(created.IsOk());
    Texture source = std::move(created).Value();

    // Top row red, bottom row blue, middle green, so the projection can be
    // checked against the +Y, -Y, and horizon directions.
    for (i32 y = 0; y < desc.height; ++y)
    {
        const Color row = y == 0 ? Color{1.0f, 0.0f, 0.0f, 1.0f}
                                 : (y == desc.height - 1 ? Color{0.0f, 0.0f, 1.0f, 1.0f}
                                                         : Color{0.0f, 1.0f, 0.0f, 1.0f});
        for (i32 x = 0; x < desc.width; ++x)
        {
            source.SetTexel(0, x, y, row);
        }
    }

    Cubemap cubemap = MakeCubemap(allocator, 16);

    CG_REQUIRE(cubemap.IsValid());
    EquirectangularProjectionDesc projection{};
    projection.samplesPerAxis = 2;
    projection.useJobs = false;
    CG_REQUIRE(ProjectEquirectangularToCubemap(source, projection, cubemap).IsOk());

    const Vec3 up = SampleRgb(cubemap, Vec3{0.0f, 1.0f, 0.0f}, 0.0f);
    const Vec3 down = SampleRgb(cubemap, Vec3{0.0f, -1.0f, 0.0f}, 0.0f);
    const Vec3 horizon = SampleRgb(cubemap, Vec3{0.0f, 0.0f, -1.0f}, 0.0f);
    CG_EXPECT_TRUE(up.x > up.y && up.x > up.z);
    CG_EXPECT_TRUE(down.z > down.x && down.z > down.y);
    CG_EXPECT_TRUE(horizon.y > horizon.x && horizon.y > horizon.z);

    EquirectangularProjectionDesc invalid{};
    invalid.samplesPerAxis = 0;
    CG_EXPECT_TRUE(ProjectEquirectangularToCubemap(source, invalid, cubemap).IsError());
    Cubemap empty;
    CG_EXPECT_TRUE(ProjectEquirectangularToCubemap(source, projection, empty).IsError());
}

CG_TEST(ibl_environment_map_builds_an_owned_linear_lighting_set)
{
    HeapAllocator allocator;
    FramebufferDesc sourceDesc{};
    sourceDesc.width = 16;
    sourceDesc.height = 8;
    sourceDesc.format = PixelFormat::RGBA32F;
    auto sourceResult = Framebuffer::Create(sourceDesc, allocator);
    CG_REQUIRE(sourceResult.IsOk());
    Framebuffer source = std::move(sourceResult).Value();
    source.Clear(Color{1.5f, 1.5f, 1.5f, 1.0f});

    EnvironmentMapDesc desc{};
    desc.faceSize = 8;
    desc.diffuseFaceSize = 4;
    desc.brdfLutSize = 8;
    desc.equirectangularSamplesPerAxis = 1;
    desc.specularSampleCount = 16;
    desc.brdfSampleCount = 16;
    desc.useJobs = false;
    auto created = EnvironmentMap::Create(source, desc, allocator);
    CG_REQUIRE(created.IsOk());
    EnvironmentMap environment = std::move(created).Value();
    CG_EXPECT_TRUE(environment.IsValid());
    CG_EXPECT_EQ(environment.Source().FaceSize(), i32{8});
    CG_EXPECT_EQ(environment.Specular().MipLevels(), u32{4});
    CG_EXPECT_EQ(environment.DiffuseIrradiance().FaceSize(), i32{4});
    CG_EXPECT_EQ(environment.BrdfLut().Width(), i32{8});

    EnvironmentLighting lighting = environment.Lighting();
    IblSurface surface{};
    surface.normal = Vec3{0.0f, 0.0f, 1.0f};
    surface.viewDirection = Vec3{0.0f, 0.0f, 1.0f};
    surface.baseColor = Vec3::One();
    surface.metallic = 1.0f;
    surface.roughness = 0.0f;
    const Vec3 reflected = EvaluateImageBasedLighting(lighting, surface);
    CG_EXPECT_NEAR(reflected.x, 1.5f, 0.08f);
    CG_EXPECT_NEAR(reflected.y, 1.5f, 0.08f);
    CG_EXPECT_NEAR(reflected.z, 1.5f, 0.08f);

    EnvironmentMapDesc invalid = desc;
    invalid.faceSize = 0;
    CG_EXPECT_TRUE(EnvironmentMap::Create(source, invalid, allocator).IsError());
    CG_EXPECT_TRUE(
        EnvironmentMap::Load("cyber_phase29_missing_environment.hdr", desc, allocator).IsError());
}

CG_TEST(ibl_scene_capture_renders_all_faces_and_builds_environment_lighting)
{
    HeapAllocator allocator;
    Vertex panels[42]{};
    u32 vertexCount = 0;
    const auto appendPanel =
        [&](const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d, const Color& color)
    {
        panels[vertexCount++] = Vertex{a, color};
        panels[vertexCount++] = Vertex{b, color};
        panels[vertexCount++] = Vertex{c, color};
        panels[vertexCount++] = Vertex{a, color};
        panels[vertexCount++] = Vertex{c, color};
        panels[vertexCount++] = Vertex{d, color};
    };
    appendPanel(Vec3{2.0f, -1.0f, -1.0f}, Vec3{2.0f, 1.0f, -1.0f}, Vec3{2.0f, 1.0f, 1.0f},
                Vec3{2.0f, -1.0f, 1.0f}, Color{1.0f, 0.0f, 0.0f, 1.0f});
    appendPanel(Vec3{-2.0f, -1.0f, 1.0f}, Vec3{-2.0f, 1.0f, 1.0f}, Vec3{-2.0f, 1.0f, -1.0f},
                Vec3{-2.0f, -1.0f, -1.0f}, Color{0.0f, 0.0f, 1.0f, 1.0f});
    appendPanel(Vec3{-1.0f, 2.0f, -1.0f}, Vec3{1.0f, 2.0f, -1.0f}, Vec3{1.0f, 2.0f, 1.0f},
                Vec3{-1.0f, 2.0f, 1.0f}, Color{0.0f, 1.0f, 0.0f, 1.0f});
    appendPanel(Vec3{-1.0f, -2.0f, 1.0f}, Vec3{1.0f, -2.0f, 1.0f}, Vec3{1.0f, -2.0f, -1.0f},
                Vec3{-1.0f, -2.0f, -1.0f}, Color{1.0f, 1.0f, 0.0f, 1.0f});
    appendPanel(Vec3{-1.0f, -1.0f, 2.0f}, Vec3{1.0f, -1.0f, 2.0f}, Vec3{1.0f, 1.0f, 2.0f},
                Vec3{-1.0f, 1.0f, 2.0f}, Color{1.0f, 0.0f, 1.0f, 1.0f});
    appendPanel(Vec3{1.0f, -1.0f, -2.0f}, Vec3{-1.0f, -1.0f, -2.0f}, Vec3{-1.0f, 1.0f, -2.0f},
                Vec3{1.0f, 1.0f, -2.0f}, Color{0.0f, 1.0f, 1.0f, 1.0f});
    appendPanel(Vec3{4.0f, -2.0f, -2.0f}, Vec3{4.0f, 2.0f, -2.0f}, Vec3{4.0f, 2.0f, 2.0f},
                Vec3{4.0f, -2.0f, 2.0f}, Color{1.0f, 1.0f, 1.0f, 1.0f});

    ProbeCaptureScene scene{Span<const Vertex>{panels, vertexCount}, &allocator};
    ReflectionProbeCaptureDesc captureDesc{};
    captureDesc.faceSize = 16;
    captureDesc.supersampleFactor = 2;
    captureDesc.farPlane = 10.0f;
    auto captured = CaptureReflectionProbe(captureDesc, &RenderProbeCaptureFace, &scene, allocator);
    CG_REQUIRE(captured.IsOk());
    Cubemap cubemap = std::move(captured).Value();
    CG_EXPECT_EQ(scene.renderCount, kCubeFaceCount);
    CG_EXPECT_EQ(scene.captureWidth, 32);
    CG_EXPECT_TRUE(cubemap.IsValid());
    CG_EXPECT_EQ(cubemap.FaceSize(), captureDesc.faceSize);
    const Color positiveX = SampleCubemapLevel(cubemap, Vec3{1.0f, 0.0f, 0.0f}, 0.0f);
    const Color negativeX = SampleCubemapLevel(cubemap, Vec3{-1.0f, 0.0f, 0.0f}, 0.0f);
    const Color positiveY = SampleCubemapLevel(cubemap, Vec3{0.0f, 1.0f, 0.0f}, 0.0f);
    const Color negativeY = SampleCubemapLevel(cubemap, Vec3{0.0f, -1.0f, 0.0f}, 0.0f);
    const Color positiveZ = SampleCubemapLevel(cubemap, Vec3{0.0f, 0.0f, 1.0f}, 0.0f);
    const Color negativeZ = SampleCubemapLevel(cubemap, Vec3{0.0f, 0.0f, -1.0f}, 0.0f);
    CG_EXPECT_NEAR(positiveX.r, 1.0f, 1.0e-5f);
    CG_EXPECT_NEAR(positiveX.b, 0.0f, 1.0e-5f);
    CG_EXPECT_NEAR(negativeX.r, 0.0f, 1.0e-5f);
    CG_EXPECT_NEAR(negativeX.b, 1.0f, 1.0e-5f);
    CG_EXPECT_NEAR(positiveY.g, 1.0f, 1.0e-5f);
    CG_EXPECT_NEAR(negativeY.r, 1.0f, 1.0e-5f);
    CG_EXPECT_NEAR(negativeY.g, 1.0f, 1.0e-5f);
    CG_EXPECT_NEAR(positiveZ.r, 1.0f, 1.0e-5f);
    CG_EXPECT_NEAR(positiveZ.b, 1.0f, 1.0e-5f);
    CG_EXPECT_NEAR(negativeZ.g, 1.0f, 1.0e-5f);
    CG_EXPECT_NEAR(negativeZ.b, 1.0f, 1.0e-5f);

    EnvironmentMapDesc environmentDesc{};
    environmentDesc.faceSize = captureDesc.faceSize;
    environmentDesc.diffuseFaceSize = 4;
    environmentDesc.brdfLutSize = 8;
    environmentDesc.specularSampleCount = 16;
    environmentDesc.brdfSampleCount = 16;
    environmentDesc.equirectangularSamplesPerAxis = 0;
    environmentDesc.useJobs = false;
    auto environmentResult = EnvironmentMap::Create(cubemap, environmentDesc, allocator);
    CG_REQUIRE(environmentResult.IsOk());
    EnvironmentMap environment = std::move(environmentResult).Value();
    CG_EXPECT_TRUE(environment.IsValid());
    CG_EXPECT_NEAR(SampleCubemapLevel(environment.Source(), Vec3{1.0f, 0.0f, 0.0f}, 0.0f).r, 1.0f,
                   1.0e-5f);

    scene.renderCount = 0;
    scene.failOnFace = 2;
    const auto failed =
        CaptureReflectionProbe(captureDesc, &RenderProbeCaptureFace, &scene, allocator);
    CG_EXPECT_TRUE(failed.IsError());
    CG_EXPECT_EQ(scene.renderCount, u32{3});
    CG_EXPECT_TRUE(CaptureReflectionProbe(captureDesc, nullptr, &scene, allocator).IsError());
    ReflectionProbeCaptureDesc invalid = captureDesc;
    invalid.nearPlane = invalid.farPlane;
    CG_EXPECT_TRUE(
        CaptureReflectionProbe(invalid, &RenderProbeCaptureFace, &scene, allocator).IsError());
    invalid = captureDesc;
    invalid.supersampleFactor = kMaxSupersampleFactor + 1;
    CG_EXPECT_TRUE(
        CaptureReflectionProbe(invalid, &RenderProbeCaptureFace, &scene, allocator).IsError());
    invalid = captureDesc;
    invalid.faceSize = kMaxCubemapFaceSize;
    invalid.supersampleFactor = kMaxSupersampleFactor;
    CG_EXPECT_TRUE(
        CaptureReflectionProbe(invalid, &RenderProbeCaptureFace, &scene, allocator).IsError());
    scene.failOnFace = kCubeFaceCount;
    scene.invalidateTargetOnFace = 1;
    scene.renderCount = 0;
    CG_EXPECT_TRUE(
        CaptureReflectionProbe(captureDesc, &RenderProbeCaptureFace, &scene, allocator).IsError());
    CG_EXPECT_EQ(scene.renderCount, u32{2});
}

CG_TEST(ibl_environment_map_loads_a_file_into_precomputed_lighting)
{
    HeapAllocator allocator;
    constexpr char kPath[] = "cyber_phase29_environment_load_test.ppm";
    constexpr char kImage[] = "P3\n2 1\n255\n255 128 0 255 128 0\n";
    const Span<const cyber::memory::Byte> bytes{
        reinterpret_cast<const cyber::memory::Byte*>(kImage), sizeof(kImage) - 1};
    CG_REQUIRE(cyber::platform::WriteFile(kPath, bytes).IsOk());

    EnvironmentMapDesc desc{};
    desc.faceSize = 8;
    desc.diffuseFaceSize = 4;
    desc.brdfLutSize = 8;
    desc.equirectangularSamplesPerAxis = 1;
    desc.specularSampleCount = 16;
    desc.brdfSampleCount = 16;
    desc.useJobs = false;
    auto loaded = EnvironmentMap::Load(kPath, desc, allocator);
    CG_EXPECT_TRUE(cyber::platform::DeleteFile(kPath).IsOk());
    CG_REQUIRE(loaded.IsOk());
    EnvironmentMap environment = std::move(loaded).Value();
    CG_EXPECT_TRUE(environment.IsValid());
    const Color texel =
        SampleCubemapLevel(environment.Source(), CubeDirectionFromFaceUv(0, 0.5f, 0.5f), 0.0f);
    CG_EXPECT_NEAR(texel.r, 1.0f, 1.0e-5f);
    CG_EXPECT_NEAR(texel.g, cyber::math::SrgbToLinear(128.0f / 255.0f), 1.0e-5f);
    CG_EXPECT_NEAR(texel.b, 0.0f, 1.0e-5f);
}

CG_TEST(ibl_spherical_harmonics_reproduce_a_constant_environment)
{
    HeapAllocator allocator;
    Cubemap environment = MakeCubemap(allocator, 16);
    CG_REQUIRE(environment.IsValid());
    FillConstant(environment, Color{0.5f, 0.5f, 0.5f, 1.0f});

    SphericalHarmonicsL2 harmonics{};
    CG_REQUIRE(ProjectCubemapToSphericalHarmonics(environment, harmonics).IsOk());

    // A constant environment of radiance L irradiates every normal with pi * L.
    for (u32 index = 0; index < 256; ++index)
    {
        const Vec3 irradiance =
            EvaluateSphericalHarmonicsIrradiance(harmonics, FibonacciDirection(index, 256));
        CG_REQUIRE_NEAR(irradiance.x, kPi * 0.5f, 2.0e-3f);
        CG_REQUIRE_NEAR(irradiance.y, kPi * 0.5f, 2.0e-3f);
    }

    Cubemap empty;
    CG_EXPECT_TRUE(ProjectCubemapToSphericalHarmonics(empty, harmonics).IsError());
}

CG_TEST(ibl_spherical_harmonics_follow_a_bright_hemisphere)
{
    HeapAllocator allocator;
    Cubemap environment = MakeCubemap(allocator, 16);
    CG_REQUIRE(environment.IsValid());
    environment.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
    const i32 size = environment.FaceSize();
    for (u32 face = 0; face < kCubeFaceCount; ++face)
    {
        for (i32 y = 0; y < size; ++y)
        {
            for (i32 x = 0; x < size; ++x)
            {
                const f32 u = (static_cast<f32>(x) + 0.5f) / static_cast<f32>(size);
                const f32 v = (static_cast<f32>(y) + 0.5f) / static_cast<f32>(size);
                if (CubeDirectionFromFaceUv(face, u, v).y > 0.0f)
                {
                    environment.SetTexel(face, 0, x, y, Color{1.0f, 1.0f, 1.0f, 1.0f});
                }
            }
        }
    }

    SphericalHarmonicsL2 harmonics{};
    CG_REQUIRE(ProjectCubemapToSphericalHarmonics(environment, harmonics).IsOk());
    const Vec3 upward = EvaluateSphericalHarmonicsIrradiance(harmonics, Vec3{0.0f, 1.0f, 0.0f});
    const Vec3 downward = EvaluateSphericalHarmonicsIrradiance(harmonics, Vec3{0.0f, -1.0f, 0.0f});
    CG_EXPECT_TRUE(upward.x > 2.5f);
    CG_EXPECT_TRUE(downward.x < 0.4f);
}

CG_TEST(ibl_irradiance_cubemap_matches_the_analytic_constant)
{
    HeapAllocator allocator;
    Cubemap environment = MakeCubemap(allocator, 16);
    CG_REQUIRE(environment.IsValid());
    FillConstant(environment, Color{0.25f, 0.25f, 0.25f, 1.0f});

    Cubemap irradiance = MakeCubemap(allocator, 8);

    CG_REQUIRE(irradiance.IsValid());
    IrradianceFilterDesc filter{};
    filter.useJobs = false;
    CG_REQUIRE(BuildDiffuseIrradianceCubemap(environment, filter, irradiance).IsOk());

    for (u32 index = 0; index < 128; ++index)
    {
        const Vec3 sampled = SampleRgb(irradiance, FibonacciDirection(index, 128), 0.0f);
        CG_REQUIRE_NEAR(sampled.x, kPi * 0.25f, 0.02f);
    }

    Cubemap empty;
    CG_EXPECT_TRUE(BuildDiffuseIrradianceCubemap(empty, filter, irradiance).IsError());
    CG_EXPECT_TRUE(BuildDiffuseIrradianceCubemap(environment, filter, empty).IsError());
}

CG_TEST(ibl_specular_prefilter_preserves_a_constant_at_every_roughness)
{
    HeapAllocator allocator;
    Cubemap environment = MakeCubemap(allocator, 16);
    CG_REQUIRE(environment.IsValid());
    FillConstant(environment, Color{0.75f, 0.75f, 0.75f, 1.0f});
    environment.GenerateMips();

    Cubemap prefiltered = MakeCubemap(allocator, 16);

    CG_REQUIRE(prefiltered.IsValid());
    SpecularPrefilterDesc prefilter{};
    prefilter.sampleCount = 32;
    prefilter.useJobs = false;
    CG_REQUIRE(PrefilterSpecularCubemap(environment, prefilter, prefiltered).IsOk());

    for (u32 level = 0; level < prefiltered.MipLevels(); ++level)
    {
        for (u32 index = 0; index < 64; ++index)
        {
            const Vec3 sampled =
                SampleRgb(prefiltered, FibonacciDirection(index, 64), static_cast<f32>(level));
            CG_REQUIRE_NEAR(sampled.x, 0.75f, 1.0e-3f);
        }
    }

    CG_EXPECT_NEAR(SpecularMipRoughness(0, 5), 0.0f, 1.0e-6f);
    CG_EXPECT_NEAR(SpecularMipRoughness(4, 5), 1.0f, 1.0e-6f);
    CG_EXPECT_NEAR(SpecularMipRoughness(9, 5), 1.0f, 1.0e-6f);
    CG_EXPECT_NEAR(SpecularRoughnessMip(0.5f, 5), 2.0f, 1.0e-6f);
    CG_EXPECT_NEAR(SpecularRoughnessMip(-1.0f, 5), 0.0f, 1.0e-6f);
    CG_EXPECT_NEAR(SpecularMipRoughness(0, 1), 0.0f, 1.0e-6f);

    Cubemap singleLevel = MakeCubemap(allocator, 4, 1);

    CG_REQUIRE(singleLevel.IsValid());
    CG_EXPECT_TRUE(PrefilterSpecularCubemap(environment, prefilter, singleLevel).IsError());
    prefilter.sampleCount = 0;
    CG_EXPECT_TRUE(PrefilterSpecularCubemap(environment, prefilter, prefiltered).IsError());
}

CG_TEST(ibl_specular_prefilter_level_zero_is_the_environment)
{
    HeapAllocator allocator;
    Cubemap environment = MakeCubemap(allocator, 16);
    CG_REQUIRE(environment.IsValid());
    const i32 size = environment.FaceSize();
    for (u32 face = 0; face < kCubeFaceCount; ++face)
    {
        for (i32 y = 0; y < size; ++y)
        {
            for (i32 x = 0; x < size; ++x)
            {
                const f32 u = (static_cast<f32>(x) + 0.5f) / static_cast<f32>(size);
                const f32 v = (static_cast<f32>(y) + 0.5f) / static_cast<f32>(size);
                const Vec3 direction = CubeDirectionFromFaceUv(face, u, v);
                const f32 value = 0.5f + 0.5f * direction.y;
                environment.SetTexel(face, 0, x, y, Color{value, value * 0.5f, 0.1f, 1.0f});
            }
        }
    }
    environment.GenerateMips();

    Cubemap prefiltered = MakeCubemap(allocator, 16);

    CG_REQUIRE(prefiltered.IsValid());
    SpecularPrefilterDesc prefilter{};
    prefilter.sampleCount = 16;
    prefilter.useJobs = false;
    CG_REQUIRE(PrefilterSpecularCubemap(environment, prefilter, prefiltered).IsOk());

    f32 worst = 0.0f;
    for (u32 index = 0; index < 256; ++index)
    {
        const Vec3 direction = FibonacciDirection(index, 256);
        const Vec3 reference = SampleRgb(environment, direction, 0.0f);
        const Vec3 mirror = SampleRgb(prefiltered, direction, 0.0f);
        worst = std::max(worst, std::abs(reference.x - mirror.x));
    }
    CG_EXPECT_TRUE(worst < 2.0e-3f);
}

CG_TEST(ibl_brdf_table_is_bounded_and_matches_the_analytic_fit)
{
    HeapAllocator allocator;
    BrdfLutDesc desc{};
    desc.size = 32;
    desc.sampleCount = 256;
    desc.useJobs = false;
    Texture lut;
    CG_REQUIRE(BuildSplitSumBrdfLut(desc, lut, allocator).IsOk());
    CG_EXPECT_TRUE(lut.IsValid());
    CG_EXPECT_EQ(lut.Width(), i32{32});

    f32 worst = 0.0f;
    f32 worstCore = 0.0f;
    for (i32 index = 1; index < 10; ++index)
    {
        const f32 ndotv = static_cast<f32>(index) / 10.0f;
        for (i32 step = 0; step <= 10; ++step)
        {
            const f32 roughness = static_cast<f32>(step) / 10.0f;
            const Vec2 table = SampleSplitSumBrdf(&lut, ndotv, roughness);
            CG_REQUIRE(table.x >= 0.0f && table.x <= 1.01f);
            CG_REQUIRE(table.y >= 0.0f && table.y <= 1.01f);
            CG_REQUIRE(table.x + table.y <= 1.05f);
            const Vec2 analytic = SampleSplitSumBrdf(nullptr, ndotv, roughness);
            const f32 difference = std::abs(table.x + table.y - analytic.x - analytic.y);
            worst = std::max(worst, difference);
            if (ndotv >= 0.3f)
            {
                worstCore = std::max(worstCore, difference);
            }
        }
    }
    // The fallback is Karis's mobile fit of the same integral. It is an
    // approximation, and a known-loose one at high roughness, so it bounds the
    // table rather than verifying it.
    std::printf("  brdf table vs analytic fit: worst %.4f, worst at n.v >= 0.3 %.4f\n",
                static_cast<double>(worst), static_cast<double>(worstCore));
    CG_EXPECT_TRUE(worst < 0.25f);
    CG_EXPECT_TRUE(worstCore < 0.2f);

    // The table itself is verified against an independent estimator: uniform
    // hemisphere sampling of the same integral, which shares no code with the
    // importance-sampled build path.
    f32 worstReference = 0.0f;
    for (i32 index = 3; index <= 9; index += 2)
    {
        const f32 ndotv = static_cast<f32>(index) / 10.0f;
        const Vec3 view{std::sqrt(std::max(0.0f, 1.0f - ndotv * ndotv)), 0.0f, ndotv};
        for (i32 step = 3; step <= 10; ++step)
        {
            const f32 roughness = static_cast<f32>(step) / 10.0f;
            const f32 alpha = roughness * roughness;
            const f32 k = alpha * 0.5f;
            constexpr u32 kSamples = 20000;
            f32 scale = 0.0f;
            f32 bias = 0.0f;
            for (u32 sample = 0; sample < kSamples; ++sample)
            {
                // Uniform over the +Z hemisphere, which is where the normal
                // points in this parameterization.
                const f32 ndotl = (static_cast<f32>(sample) + 0.5f) / static_cast<f32>(kSamples);
                const f32 radial = std::sqrt(std::max(0.0f, 1.0f - ndotl * ndotl));
                const f32 phi = 2.0f * kPi * static_cast<f32>(sample) * 0.61803398875f;
                const Vec3 light{radial * std::cos(phi), radial * std::sin(phi), ndotl};
                const Vec3 half = cyber::math::Normalize(view + light);
                const f32 ndoth = half.z;
                const f32 vdoth = std::max(0.0f, cyber::math::Dot(view, half));
                const f32 denominator = ndoth * ndoth * (alpha * alpha - 1.0f) + 1.0f;
                const f32 distribution = alpha * alpha / (kPi * denominator * denominator);
                const f32 geometry =
                    (ndotv / (ndotv * (1.0f - k) + k)) * (ndotl / (ndotl * (1.0f - k) + k));
                const f32 weight = distribution * geometry / (4.0f * ndotv);
                const f32 fresnel = std::pow(1.0f - vdoth, 5.0f);
                scale += (1.0f - fresnel) * weight;
                bias += fresnel * weight;
            }
            // Uniform hemisphere sampling: the solid angle per sample is the
            // hemisphere's 2 pi divided by the sample count.
            const f32 measure = 2.0f * kPi / static_cast<f32>(kSamples);
            scale *= measure;
            bias *= measure;
            const Vec2 table = SampleSplitSumBrdf(&lut, ndotv, roughness);
            worstReference = std::max(worstReference, std::abs(table.x + table.y - scale - bias));
        }
    }
    std::printf("  brdf table vs uniform-sampled reference: worst %.4f\n",
                static_cast<double>(worstReference));
    CG_EXPECT_TRUE(worstReference < 0.03f);

    BrdfLutDesc invalidSize{};
    invalidSize.size = 4;
    CG_EXPECT_TRUE(BuildSplitSumBrdfLut(invalidSize, lut, allocator).IsError());
    BrdfLutDesc invalidSamples{};
    invalidSamples.sampleCount = 4;
    CG_EXPECT_TRUE(BuildSplitSumBrdfLut(invalidSamples, lut, allocator).IsError());
}

CG_TEST(ibl_white_furnace_returns_the_environment)
{
    HeapAllocator allocator;
    constexpr f32 kRadiance = 0.6f;

    Cubemap environment = MakeCubemap(allocator, 16);

    CG_REQUIRE(environment.IsValid());
    FillConstant(environment, Color{kRadiance, kRadiance, kRadiance, 1.0f});
    environment.GenerateMips();

    Cubemap prefiltered = MakeCubemap(allocator, 16);

    CG_REQUIRE(prefiltered.IsValid());
    SpecularPrefilterDesc prefilter{};
    prefilter.sampleCount = 64;
    prefilter.useJobs = false;
    CG_REQUIRE(PrefilterSpecularCubemap(environment, prefilter, prefiltered).IsOk());

    Cubemap irradiance = MakeCubemap(allocator, 8);

    CG_REQUIRE(irradiance.IsValid());
    IrradianceFilterDesc filter{};
    filter.useJobs = false;
    CG_REQUIRE(BuildDiffuseIrradianceCubemap(environment, filter, irradiance).IsOk());

    BrdfLutDesc lutDesc{};
    lutDesc.size = 64;
    lutDesc.sampleCount = 512;
    lutDesc.useJobs = false;
    Texture lut;
    CG_REQUIRE(BuildSplitSumBrdfLut(lutDesc, lut, allocator).IsOk());

    EnvironmentLighting lighting{};
    lighting.specular = &prefiltered;
    lighting.diffuseIrradiance = &irradiance;
    lighting.brdfLut = &lut;

    // A white, fully reflective surface must return the environment radiance
    // at every roughness and both metallic extremes; that is what the
    // multiple-scattering term in the split-sum form buys.
    for (i32 step = 0; step <= 10; ++step)
    {
        const f32 roughness = static_cast<f32>(step) / 10.0f;
        for (i32 metal = 0; metal <= 1; ++metal)
        {
            IblSurface surface{};
            surface.normal = Vec3{0.0f, 1.0f, 0.0f};
            surface.viewDirection = cyber::math::Normalize(Vec3{0.3f, 0.8f, 0.2f});
            surface.baseColor = Vec3::One();
            surface.metallic = static_cast<f32>(metal);
            surface.roughness = roughness;
            const Vec3 result = EvaluateImageBasedLighting(lighting, surface);
            CG_REQUIRE_NEAR(result.x, kRadiance, 0.03f);
        }
    }
}

CG_TEST(ibl_evaluation_rejects_degenerate_inputs)
{
    HeapAllocator allocator;
    Cubemap environment = MakeCubemap(allocator, 8);
    CG_REQUIRE(environment.IsValid());
    FillConstant(environment, Color{1.0f, 1.0f, 1.0f, 1.0f});

    EnvironmentLighting lighting{};
    lighting.diffuseIrradiance = &environment;
    lighting.intensity = 0.0f;

    IblSurface surface{};
    const Vec3 zeroIntensity = EvaluateImageBasedLighting(lighting, surface);
    CG_EXPECT_NEAR(zeroIntensity.x, 0.0f, 1.0e-6f);

    lighting.intensity = 1.0f;
    surface.normal = Vec3::Zero();
    surface.viewDirection = Vec3::Zero();
    const Vec3 degenerate = EvaluateImageBasedLighting(lighting, surface);
    CG_EXPECT_TRUE(std::isfinite(degenerate.x) && degenerate.x >= 0.0f);

    // No specular cubemap means no specular term, not a crash.
    const Vec3 specular = EvaluateIblSpecular(lighting, surface);
    CG_EXPECT_NEAR(specular.x, 0.0f, 1.0e-6f);
}

CG_TEST(ibl_specular_occlusion_narrows_with_roughness)
{
    CG_EXPECT_NEAR(ComputeSpecularOcclusion(1.0f, 1.0f, 0.5f), 1.0f, 1.0e-5f);
    CG_EXPECT_NEAR(ComputeSpecularOcclusion(1.0f, 0.0f, 0.5f), 0.0f, 1.0e-5f);

    // A mirror lobe escapes an occluded crevice; a rough one does not.
    const f32 smooth = ComputeSpecularOcclusion(0.8f, 0.35f, 0.0f);
    const f32 rough = ComputeSpecularOcclusion(0.8f, 0.35f, 1.0f);
    CG_EXPECT_TRUE(smooth > rough);
    CG_EXPECT_TRUE(rough <= 0.36f);
    CG_EXPECT_NEAR(ComputeSpecularOcclusion(0.5f, std::nanf(""), 0.5f), 0.0f, 1.0e-5f);
}

CG_TEST(ibl_precomputation_is_independent_of_the_job_system)
{
    HeapAllocator allocator;
    auto createdJobs =
        cyber::jobs::JobSystem::Create(cyber::jobs::JobSystem::Desc{4, nullptr}, allocator);
    CG_REQUIRE(createdJobs.IsOk());
    cyber::jobs::JobSystem jobSystem = std::move(createdJobs).Value();

    Cubemap environment = MakeCubemap(allocator, 16);
    CG_REQUIRE(environment.IsValid());
    const i32 size = environment.FaceSize();
    for (u32 face = 0; face < kCubeFaceCount; ++face)
    {
        for (i32 y = 0; y < size; ++y)
        {
            for (i32 x = 0; x < size; ++x)
            {
                const f32 value =
                    static_cast<f32>((x * 7 + y * 13 + static_cast<i32>(face) * 3) % 17) / 17.0f;
                environment.SetTexel(face, 0, x, y, Color{value, 1.0f - value, 0.5f, 1.0f});
            }
        }
    }
    environment.GenerateMips();

    Cubemap serial = MakeCubemap(allocator, 16);

    CG_REQUIRE(serial.IsValid());
    Cubemap parallel = MakeCubemap(allocator, 16);
    CG_REQUIRE(parallel.IsValid());
    SpecularPrefilterDesc prefilter{};
    prefilter.sampleCount = 32;
    prefilter.useJobs = false;
    CG_REQUIRE(PrefilterSpecularCubemap(environment, prefilter, serial).IsOk());
    prefilter.useJobs = true;
    CG_REQUIRE(PrefilterSpecularCubemap(environment, prefilter, parallel, &jobSystem).IsOk());

    for (u32 level = 0; level < serial.MipLevels(); ++level)
    {
        const i32 levelSize = serial.MipSize(level);
        for (u32 face = 0; face < kCubeFaceCount; ++face)
        {
            for (i32 y = 0; y < levelSize; ++y)
            {
                for (i32 x = 0; x < levelSize; ++x)
                {
                    const Color a = serial.GetTexel(face, level, x, y);
                    const Color b = parallel.GetTexel(face, level, x, y);
                    CG_REQUIRE(a.r == b.r && a.g == b.g && a.b == b.b);
                }
            }
        }
    }

    jobSystem.Shutdown();
}

CG_TEST(ibl_probe_weight_is_one_inside_and_zero_outside)
{
    ReflectionProbe probe{};
    probe.influenceMin = Vec3{-2.0f, -2.0f, -2.0f};
    probe.influenceMax = Vec3{2.0f, 2.0f, 2.0f};
    probe.blendDistance = 0.5f;

    CG_EXPECT_NEAR(EvaluateReflectionProbeWeight(probe, Vec3::Zero()), 1.0f, 1.0e-6f);
    CG_EXPECT_NEAR(EvaluateReflectionProbeWeight(probe, Vec3{3.0f, 0.0f, 0.0f}), 0.0f, 1.0e-6f);
    CG_EXPECT_NEAR(EvaluateReflectionProbeWeight(probe, Vec3{2.0f, 0.0f, 0.0f}), 0.0f, 1.0e-6f);
    CG_EXPECT_TRUE(EvaluateReflectionProbeWeight(probe, Vec3{1.75f, 0.0f, 0.0f}) > 0.0f);

    // The ramp uses the nearest face, so a corner is not brighter than a face.
    const f32 corner = EvaluateReflectionProbeWeight(probe, Vec3{1.75f, 1.75f, 1.75f});
    const f32 face = EvaluateReflectionProbeWeight(probe, Vec3{1.75f, 0.0f, 0.0f});
    CG_EXPECT_NEAR(corner, face, 1.0e-6f);

    ReflectionProbe empty{};
    CG_EXPECT_NEAR(EvaluateReflectionProbeWeight(empty, Vec3::Zero()), 0.0f, 1.0e-6f);

    ReflectionProbe hard = probe;
    hard.blendDistance = 0.0f;
    CG_EXPECT_NEAR(EvaluateReflectionProbeWeight(hard, Vec3{1.99f, 0.0f, 0.0f}), 1.0f, 1.0e-6f);
}

CG_TEST(ibl_box_parallax_redirects_toward_the_proxy_intersection)
{
    ReflectionProbe probe{};
    probe.position = Vec3::Zero();
    probe.influenceMin = Vec3{-5.0f, -5.0f, -5.0f};
    probe.influenceMax = Vec3{5.0f, 5.0f, 5.0f};
    probe.proxyMin = Vec3{-5.0f, -5.0f, -5.0f};
    probe.proxyMax = Vec3{5.0f, 5.0f, 5.0f};

    // From the capture position the correction is the identity.
    const Vec3 atCenter = ApplyBoxParallaxCorrection(probe, Vec3::Zero(), Vec3{1.0f, 0.0f, 0.0f});
    CG_EXPECT_NEAR(atCenter.x, 1.0f, 1.0e-5f);

    // Looking +X from a point near the -X wall hits the +X wall at y = 4, so
    // the corrected direction tilts upward relative to the mirror ray.
    const Vec3 offset =
        ApplyBoxParallaxCorrection(probe, Vec3{-4.0f, 4.0f, 0.0f}, Vec3{1.0f, 0.0f, 0.0f});
    CG_EXPECT_TRUE(offset.y > 0.3f);
    CG_EXPECT_TRUE(offset.x > 0.0f);

    ReflectionProbe noProxy = probe;
    noProxy.proxyMin = Vec3::Zero();
    noProxy.proxyMax = Vec3::Zero();
    const Vec3 unchanged =
        ApplyBoxParallaxCorrection(noProxy, Vec3{1.0f, 1.0f, 0.0f}, Vec3{0.0f, 0.0f, -1.0f});
    CG_EXPECT_NEAR(unchanged.z, -1.0f, 1.0e-6f);
}

CG_TEST(ibl_probe_blend_normalizes_overlapping_weights)
{
    HeapAllocator allocator;
    Cubemap first = MakeCubemap(allocator, 8);
    CG_REQUIRE(first.IsValid());
    Cubemap second = MakeCubemap(allocator, 8);
    CG_REQUIRE(second.IsValid());
    FillConstant(first, Color{kPi, 0.0f, 0.0f, 1.0f});
    FillConstant(second, Color{0.0f, kPi, 0.0f, 1.0f});

    ReflectionProbe probes[2]{};
    probes[0].influenceMin = Vec3{-2.0f, -2.0f, -2.0f};
    probes[0].influenceMax = Vec3{2.0f, 2.0f, 2.0f};
    probes[0].environment.diffuseIrradiance = &first;
    probes[1].influenceMin = Vec3{-2.0f, -2.0f, -2.0f};
    probes[1].influenceMax = Vec3{2.0f, 2.0f, 2.0f};
    probes[1].environment.diffuseIrradiance = &second;

    ReflectionProbeSet set{};
    set.probes = cyber::memory::Span<const ReflectionProbe>{probes, 2};

    IblSurface surface{};
    surface.normal = Vec3{0.0f, 1.0f, 0.0f};
    surface.viewDirection = Vec3{0.0f, 1.0f, 0.0f};
    surface.baseColor = Vec3::One();
    surface.metallic = 0.0f;
    surface.roughness = 1.0f;

    // Two probes of weight 1 each normalize to one half, so the result is the
    // average rather than the sum.
    const Vec3 blended = EvaluateReflectionProbes(set, Vec3::Zero(), surface);
    CG_EXPECT_NEAR(blended.x, blended.y, 1.0e-5f);
    CG_EXPECT_TRUE(blended.x > 0.3f && blended.x < 0.6f);

    // Outside both probes the fallback takes the whole weight.
    EnvironmentLighting fallback{};
    fallback.diffuseIrradiance = &first;
    set.fallback = &fallback;
    const Vec3 outside = EvaluateReflectionProbes(set, Vec3{9.0f, 0.0f, 0.0f}, surface);
    CG_EXPECT_TRUE(outside.x > 0.8f);
    CG_EXPECT_NEAR(outside.y, 0.0f, 1.0e-5f);
}

CG_TEST(ibl_hiz_pyramid_keeps_the_closest_depth)
{
    HeapAllocator allocator;
    DepthBufferDesc desc{};
    desc.width = 8;
    desc.height = 8;
    auto created = DepthBuffer::Create(desc, allocator);
    CG_REQUIRE(created.IsOk());
    DepthBuffer depth = std::move(created).Value();
    depth.Clear(0.0f);

    for (i32 y = 0; y < 8; ++y)
    {
        for (i32 x = 0; x < 8; ++x)
        {
            CG_REQUIRE(depth.TestAndWrite(x, y, static_cast<f32>(x + y * 8) / 64.0f));
        }
    }

    auto built = HiZPyramid::Build(depth, allocator);
    CG_REQUIRE(built.IsOk());
    HiZPyramid pyramid = std::move(built).Value();
    CG_EXPECT_TRUE(pyramid.IsValid());
    CG_EXPECT_EQ(pyramid.Width(), i32{8});
    CG_EXPECT_EQ(pyramid.LevelCount(), u32{4});
    CG_EXPECT_EQ(pyramid.LevelWidth(3), i32{1});

    for (i32 y = 0; y < 8; ++y)
    {
        for (i32 x = 0; x < 8; ++x)
        {
            CG_REQUIRE_NEAR(pyramid.Depth(0, x, y), depth.Get(x, y), 1.0e-6f);
        }
    }

    // Reverse-Z: the coarse level holds the maximum, which is the closest
    // surface in the footprint.
    for (u32 level = 1; level < pyramid.LevelCount(); ++level)
    {
        const i32 width = pyramid.LevelWidth(level);
        const i32 height = pyramid.LevelHeight(level);
        for (i32 y = 0; y < height; ++y)
        {
            for (i32 x = 0; x < width; ++x)
            {
                const f32 expected =
                    std::max(std::max(pyramid.Depth(level - 1, x * 2, y * 2),
                                      pyramid.Depth(level - 1, x * 2 + 1, y * 2)),
                             std::max(pyramid.Depth(level - 1, x * 2, y * 2 + 1),
                                      pyramid.Depth(level - 1, x * 2 + 1, y * 2 + 1)));
                CG_REQUIRE_NEAR(pyramid.Depth(level, x, y), expected, 1.0e-6f);
            }
        }
    }

    CG_EXPECT_NEAR(pyramid.Depth(0, -1, 0), 0.0f, 1.0e-6f);
    CG_EXPECT_NEAR(pyramid.Depth(9, 0, 0), 0.0f, 1.0e-6f);

    DepthBuffer invalid;
    CG_EXPECT_TRUE(HiZPyramid::Build(invalid, allocator).IsError());
}

CG_TEST(ibl_screen_space_trace_rejects_incomplete_descriptors)
{
    HeapAllocator allocator;
    ScreenSpaceReflectionDesc desc{};
    const ScreenSpaceReflectionSample sample =
        TraceScreenSpaceReflection(desc, Vec3::Zero(), Vec3{0.0f, 1.0f, 0.0f});
    CG_EXPECT_FALSE(sample.hit);
    CG_EXPECT_NEAR(sample.confidence, 0.0f, 1.0e-6f);

    FramebufferDesc colorDesc{};
    colorDesc.width = 16;
    colorDesc.height = 16;
    colorDesc.format = PixelFormat::RGBA32F;
    auto color = Framebuffer::Create(colorDesc, allocator);
    CG_REQUIRE(color.IsOk());
    Framebuffer target = std::move(color).Value();

    DepthBufferDesc depthDesc{};
    depthDesc.width = 8;
    depthDesc.height = 8;
    auto depth = DepthBuffer::Create(depthDesc, allocator);
    CG_REQUIRE(depth.IsOk());
    DepthBuffer buffer = std::move(depth).Value();

    desc.color = &target;
    desc.depth = &buffer;
    // Mismatched dimensions are refused rather than sampled out of range.
    const ScreenSpaceReflectionSample mismatched =
        TraceScreenSpaceReflection(desc, Vec3::Zero(), Vec3{0.0f, 1.0f, 0.0f});
    CG_EXPECT_FALSE(mismatched.hit);
}

CG_TEST(ibl_procedural_sky_is_blue_at_noon_and_dark_below_the_horizon)
{
    ProceduralSkyConfig config{};
    config.sunDirection = Vec3{0.0f, -1.0f, 0.0f};
    config.viewSampleCount = 16;
    config.lightSampleCount = 4;
    config.groundAlbedo = Vec3::Zero();
    // At noon the zenith is the sun, so the disk is excluded while the sky
    // colour itself is measured.
    config.includeSunDisk = false;

    const Vec3 zenith = EvaluateProceduralSky(config, Vec3{0.0f, 1.0f, 0.0f});
    CG_EXPECT_TRUE(zenith.z > zenith.y && zenith.y > zenith.x);
    CG_EXPECT_TRUE(zenith.z > 0.0f);

    // A horizon ray traverses far more air, so the long-wavelength share rises.
    const Vec3 horizon = EvaluateProceduralSky(config, Vec3{0.0f, 0.02f, -1.0f});
    CG_EXPECT_TRUE(horizon.x / horizon.z > zenith.x / zenith.z);

    // A black ground still returns the air between the viewer and it, which is
    // the aerial perspective, so the value is small rather than exactly zero.
    const Vec3 ground = EvaluateProceduralSky(config, Vec3{0.0f, -1.0f, 0.0f});
    CG_EXPECT_TRUE(ground.x >= 0.0f && ground.x < 0.01f);
    CG_EXPECT_TRUE(ground.z < zenith.z);

    // The sun disk carries far more radiance than the sky beside it.
    config.includeSunDisk = true;
    const Vec3 disk = EvaluateProceduralSky(config, Vec3{0.0f, 1.0f, 0.0f});
    const Vec3 offSun = EvaluateProceduralSky(config, Vec3{0.0f, 0.9f, 0.4f});
    CG_EXPECT_TRUE(disk.y > offSun.y * 10.0f);

    const Vec3 transmitted = EvaluateSunTransmittance(config);
    CG_EXPECT_TRUE(transmitted.x > 0.0f && transmitted.x < config.sunIrradiance.x);
    CG_EXPECT_TRUE(transmitted.z < transmitted.x);

    ProceduralSkyConfig invalid{};
    invalid.sunDirection = Vec3::Zero();
    CG_EXPECT_NEAR(EvaluateProceduralSky(invalid, Vec3{0.0f, 1.0f, 0.0f}).x, 0.0f, 1.0e-6f);
    CG_EXPECT_NEAR(EvaluateSunTransmittance(invalid).x, 0.0f, 1.0e-6f);
}

CG_TEST(ibl_procedural_sky_capture_fills_a_cubemap)
{
    HeapAllocator allocator;
    Cubemap sky = MakeCubemap(allocator, 16);
    CG_REQUIRE(sky.IsValid());

    ProceduralSkyConfig config{};
    config.sunDirection = cyber::math::Normalize(Vec3{0.2f, -0.6f, -0.4f});
    config.viewSampleCount = 8;
    config.lightSampleCount = 4;
    CG_REQUIRE(RenderProceduralSkyCubemap(config, sky).IsOk());

    const Vec3 up = SampleRgb(sky, Vec3{0.0f, 1.0f, 0.0f}, 0.0f);
    const Vec3 down = SampleRgb(sky, Vec3{0.0f, -1.0f, 0.0f}, 0.0f);
    CG_EXPECT_TRUE(up.z > 0.0f);
    // The sky is blue-dominant; the sunlit grey ground below is not.
    CG_EXPECT_TRUE(up.z > up.x);
    CG_EXPECT_TRUE(down.z <= down.x * 1.001f);

    // The chain is regenerated, so a coarse level is a filtered sky rather
    // than black.
    const Vec3 coarse =
        SampleRgb(sky, Vec3{0.0f, 1.0f, 0.0f}, static_cast<f32>(sky.MipLevels() - 1));
    CG_EXPECT_TRUE(coarse.z > 0.0f);

    Cubemap empty;
    CG_EXPECT_TRUE(RenderProceduralSkyCubemap(config, empty).IsError());
    ProceduralSkyConfig invalid{};
    invalid.viewSampleCount = 0;
    CG_EXPECT_TRUE(RenderProceduralSkyCubemap(invalid, sky).IsError());
}
