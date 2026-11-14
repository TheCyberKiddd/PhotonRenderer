#include <cyber/math/mat4.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/software/camera.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/shader.h>
#include <cyber/test/test.h>

#include <cmath>
#include <cstring>
#include <utility>

using namespace cyber::software;
using namespace cyber::memory;
using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u64;
using cyber::math::Color;
using cyber::math::Mat4;
using cyber::math::Vec2;
using cyber::math::Vec3;

namespace
{

struct SceneData
{
    TransformShaderData transform;
    Color color{1.0f, 1.0f, 1.0f, 1.0f};
};

void SceneVertex(const VertexInput& input, const void* userData, VertexStageOutput& output) noexcept
{
    const auto* scene = static_cast<const SceneData*>(userData);
    TransformVertexShader(input, &scene->transform, output);
}

Color ConstantPixel(const Varyings&, const void* userData) noexcept
{
    return static_cast<const SceneData*>(userData)->color;
}

Color UvPixel(const Varyings& varyings, const void*) noexcept
{
    return Color{varyings.uv.x, varyings.uv.y, 0.0f, 1.0f};
}

bool CreateTargets(HeapAllocator& heap, i32 size, Framebuffer& color, DepthBuffer& depth)
{
    FramebufferDesc colorDesc{};
    colorDesc.width = size;
    colorDesc.height = size;
    colorDesc.format = PixelFormat::RGBA32F;
    auto createdColor = Framebuffer::Create(colorDesc, heap);
    if (!createdColor)
    {
        return false;
    }
    color = std::move(createdColor.Value());

    DepthBufferDesc depthDesc{};
    depthDesc.width = size;
    depthDesc.height = size;
    auto createdDepth = DepthBuffer::Create(depthDesc, heap);
    if (!createdDepth)
    {
        return false;
    }
    depth = std::move(createdDepth.Value());
    return true;
}

void MakeQuad(f32 z, VertexInput (&vertices)[6])
{
    const Vec3 positions[4] = {
        Vec3{-1.0f, -1.0f, z},
        Vec3{1.0f, -1.0f, z},
        Vec3{1.0f, 1.0f, z},
        Vec3{-1.0f, 1.0f, z},
    };
    const Vec2 uvs[4] = {Vec2{0.0f, 0.0f}, Vec2{1.0f, 0.0f}, Vec2{1.0f, 1.0f}, Vec2{0.0f, 1.0f}};
    const i32 indices[6] = {0, 1, 2, 0, 2, 3};
    for (i32 index = 0; index < 6; ++index)
    {
        const i32 corner = indices[index];
        VertexInput vertex{};
        vertex.position = positions[corner];
        vertex.uv = uvs[corner];
        vertex.normal = Vec3{0.0f, 0.0f, 1.0f};
        vertex.color = Color{1.0f, 1.0f, 1.0f, 1.0f};
        vertices[index] = vertex;
    }
}

}  // namespace

CG_TEST(software_shader_constant_color)
{
    HeapAllocator heap;
    Framebuffer color;
    DepthBuffer depth;
    CG_REQUIRE(CreateTargets(heap, 8, color, depth));
    color.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
    depth.Clear(0.0f);

    SceneData scene;
    scene.color = Color{0.25f, 0.5f, 0.75f, 1.0f};

    VertexInput quad[6];
    MakeQuad(0.5f, quad);

    ShaderDesc desc{};
    desc.pipeline.viewport = Viewport{0.0f, 0.0f, 8.0f, 8.0f};
    desc.vertexShader = &SceneVertex;
    desc.pixelShader = &ConstantPixel;
    desc.userData = &scene;

    PipelineStats stats{};
    CG_REQUIRE(static_cast<bool>(DrawShadedTriangles(desc, Span<const VertexInput>(quad), color,
                                                     depth, nullptr, heap, &stats)));
    CG_EXPECT_EQ(stats.triangles, u64{2});
    CG_EXPECT_EQ(stats.trianglesSubmitted, u64{2});
    CG_EXPECT_EQ(stats.pixels, u64{64});
    CG_EXPECT_EQ(stats.pixelsRejected, u64{0});

    for (i32 y = 0; y < 8; ++y)
    {
        for (i32 x = 0; x < 8; ++x)
        {
            const Color pixel = color.GetPixel(x, y);
            CG_EXPECT_NEAR(pixel.r, 0.25f, 1.0e-5f);
            CG_EXPECT_NEAR(pixel.g, 0.5f, 1.0e-5f);
            CG_EXPECT_NEAR(pixel.b, 0.75f, 1.0e-5f);
        }
    }
    CG_EXPECT_NEAR(depth.Get(4, 4), 0.5f, 1.0e-5f);
}

CG_TEST(software_shader_varying_interpolation)
{
    HeapAllocator heap;
    Framebuffer color;
    DepthBuffer depth;
    CG_REQUIRE(CreateTargets(heap, 8, color, depth));
    color.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
    depth.Clear(0.0f);

    SceneData scene;
    VertexInput quad[6];
    MakeQuad(0.5f, quad);

    ShaderDesc desc{};
    desc.pipeline.viewport = Viewport{0.0f, 0.0f, 8.0f, 8.0f};
    desc.vertexShader = &SceneVertex;
    desc.pixelShader = &UvPixel;
    desc.userData = &scene;

    CG_REQUIRE(static_cast<bool>(
        DrawShadedTriangles(desc, Span<const VertexInput>(quad), color, depth, nullptr, heap)));

    // UVs increase with x and decrease with y because the viewport flips y.
    const Color topLeft = color.GetPixel(0, 0);
    const Color topRight = color.GetPixel(7, 0);
    const Color bottomLeft = color.GetPixel(0, 7);
    CG_EXPECT_TRUE(topRight.r > topLeft.r);
    CG_EXPECT_TRUE(bottomLeft.g < topLeft.g);
    CG_EXPECT_NEAR(topLeft.r, 0.5f / 8.0f, 2.0e-3f);
}

CG_TEST(software_shader_clipping_preserves_varyings)
{
    HeapAllocator heap;
    Framebuffer color;
    DepthBuffer depth;
    CG_REQUIRE(CreateTargets(heap, 32, color, depth));
    color.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
    depth.Clear(0.0f);

    Camera camera;
    camera.nearPlane = 0.1f;
    camera.farPlane = 100.0f;
    camera.fovYRadians = 60.0f * cyber::math::kDegreesToRadians;

    SceneData scene;
    scene.transform.modelViewProjection = CameraProjection(camera, 1.0f) * CameraView(camera);

    const VertexInput crossing[3] = {
        VertexInput{Vec3{0.0f, 0.5f, 0.5f}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{0.0f, 0.0f},
                    Color{1.0f, 1.0f, 1.0f, 1.0f}},
        VertexInput{Vec3{-2.0f, 0.0f, -1.0f}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{1.0f, 0.0f},
                    Color{1.0f, 1.0f, 1.0f, 1.0f}},
        VertexInput{Vec3{2.0f, 0.0f, -1.0f}, Vec3{0.0f, 0.0f, 1.0f}, Vec2{0.0f, 1.0f},
                    Color{1.0f, 1.0f, 1.0f, 1.0f}},
    };

    ShaderDesc desc{};
    desc.pipeline.viewport = Viewport{0.0f, 0.0f, 32.0f, 32.0f};
    desc.vertexShader = &SceneVertex;
    desc.pixelShader = &ConstantPixel;
    desc.userData = &scene;

    PipelineStats stats{};
    CG_REQUIRE(static_cast<bool>(DrawShadedTriangles(desc, Span<const VertexInput>(crossing), color,
                                                     depth, nullptr, heap, &stats)));
    CG_EXPECT_EQ(stats.trianglesClipped, u64{1});
    CG_EXPECT_TRUE(stats.pixels > u64{0});
    CG_EXPECT_TRUE(stats.trianglesInvalid == u64{0});
}

CG_TEST(software_shader_rejects_invalid_arguments)
{
    HeapAllocator heap;
    Framebuffer color;
    DepthBuffer depth;
    CG_REQUIRE(CreateTargets(heap, 8, color, depth));

    SceneData scene;
    VertexInput quad[6];
    MakeQuad(0.5f, quad);

    ShaderDesc desc{};
    desc.pipeline.viewport = Viewport{0.0f, 0.0f, 8.0f, 8.0f};
    desc.vertexShader = &SceneVertex;
    desc.pixelShader = &ConstantPixel;
    desc.userData = &scene;

    // A missing stage is rejected.
    ShaderDesc noVertex = desc;
    noVertex.vertexShader = nullptr;
    CG_EXPECT_FALSE(static_cast<bool>(
        DrawShadedTriangles(noVertex, Span<const VertexInput>(quad), color, depth, nullptr, heap)));

    ShaderDesc noPixel = desc;
    noPixel.pixelShader = nullptr;
    CG_EXPECT_FALSE(static_cast<bool>(
        DrawShadedTriangles(noPixel, Span<const VertexInput>(quad), color, depth, nullptr, heap)));

    // Vertex count must be a multiple of three.
    CG_EXPECT_FALSE(static_cast<bool>(
        DrawShadedTriangles(desc, Span<const VertexInput>(quad, 4), color, depth, nullptr, heap)));

    // An invalid target is rejected.
    Framebuffer invalid;
    CG_EXPECT_FALSE(static_cast<bool>(
        DrawShadedTriangles(desc, Span<const VertexInput>(quad), invalid, depth, nullptr, heap)));
}

CG_TEST(software_shader_depth_only_draw_matches_the_shaded_pipeline)
{
    HeapAllocator heap;
    constexpr i32 kSize = 96;
    auto colorResult =
        Framebuffer::Create(FramebufferDesc{kSize, kSize, PixelFormat::RGBA32F, 0}, heap);
    auto shadedDepthResult = DepthBuffer::Create(DepthBufferDesc{kSize, kSize, 0}, heap);
    auto depthOnlyResult = DepthBuffer::Create(DepthBufferDesc{kSize, kSize, 0}, heap);
    CG_REQUIRE(static_cast<bool>(colorResult));
    CG_REQUIRE(static_cast<bool>(shadedDepthResult));
    CG_REQUIRE(static_cast<bool>(depthOnlyResult));
    Framebuffer color = std::move(colorResult.Value());
    DepthBuffer shadedDepth = std::move(shadedDepthResult.Value());
    DepthBuffer depthOnly = std::move(depthOnlyResult.Value());

    // Geometry that crosses the near plane and spans depth, so clipping, the
    // perspective divide, winding, and the depth test all take part.
    const VertexInput vertices[] = {
        VertexInput{Vec3{-1.5f, -1.0f, -0.5f}},
        VertexInput{Vec3{1.5f, -1.0f, -3.0f}},
        VertexInput{Vec3{0.0f, 1.4f, -1.5f}},
        VertexInput{Vec3{-1.2f, 0.9f, -2.2f}},
        VertexInput{Vec3{1.4f, 0.4f, -0.8f}},
        VertexInput{Vec3{0.2f, -1.3f, -2.6f}},
        // This vertex sits behind the near plane at z = 1.8, so the triangle
        // is clipped and both pipelines must emit the same fan.
        VertexInput{Vec3{-0.9f, -0.2f, 2.6f}},
        VertexInput{Vec3{1.1f, 0.7f, -0.3f}},
        VertexInput{Vec3{0.1f, 1.2f, -1.1f}},
    };
    const Mat4 projection = Mat4::PerspectiveReverseZ(1.1f, 1.0f, 0.2f, 12.0f);
    const Mat4 view =
        Mat4::LookAt(Vec3{0.0f, 0.0f, 2.0f}, Vec3{0.0f, 0.0f, -1.0f}, Vec3{0.0f, 1.0f, 0.0f});
    const Mat4 clipFromObject = projection * view;

    TransformShaderData transform{};
    transform.modelViewProjection = clipFromObject;
    transform.modelToWorld = Mat4::Identity();
    transform.normalToWorld = Mat4::Identity();

    ShaderDesc shaded{};
    shaded.vertexShader = &TransformVertexShader;
    shaded.pixelShader = [](const Varyings&, const void*) noexcept
    { return Color{1.0f, 1.0f, 1.0f, 1.0f}; };
    shaded.userData = &transform;
    shaded.pipeline.viewport =
        Viewport{0.0f, 0.0f, static_cast<f32>(kSize), static_cast<f32>(kSize)};
    shaded.pipeline.cullMode = CullMode::None;
    shaded.pipeline.useJobs = false;

    color.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
    shadedDepth.Clear();
    PipelineStats shadedStats{};
    CG_REQUIRE(
        static_cast<bool>(DrawShadedTriangles(shaded, Span<const VertexInput>(vertices), color,
                                              shadedDepth, nullptr, heap, &shadedStats)));

    DepthOnlyShaderDesc depthDesc{};
    depthDesc.vertexShader = &TransformDepthOnlyVertexShader;
    depthDesc.userData = &clipFromObject;
    depthDesc.pipeline = shaded.pipeline;

    depthOnly.Clear();
    PipelineStats depthStats{};
    CG_REQUIRE(static_cast<bool>(DrawDepthOnlyTriangles(
        depthDesc, Span<const VertexInput>(vertices), depthOnly, nullptr, heap, &depthStats)));

    CG_EXPECT_EQ(depthStats.triangles, shadedStats.triangles);
    CG_EXPECT_EQ(depthStats.trianglesClipped, shadedStats.trianglesClipped);
    CG_EXPECT_EQ(depthStats.trianglesCulled, shadedStats.trianglesCulled);
    CG_EXPECT_EQ(depthStats.trianglesSubmitted, shadedStats.trianglesSubmitted);
    CG_EXPECT_EQ(depthStats.pixels, shadedStats.pixels);
    CG_EXPECT_EQ(depthStats.pixelsRejected, shadedStats.pixelsRejected);
    CG_EXPECT_TRUE(depthStats.trianglesClipped > u64{0});
    CG_EXPECT_TRUE(depthStats.pixels > u64{0});

    u64 mismatches = 0;
    for (i32 y = 0; y < kSize; ++y)
    {
        for (i32 x = 0; x < kSize; ++x)
        {
            if (depthOnly.Get(x, y) != shadedDepth.Get(x, y))
            {
                ++mismatches;
            }
        }
    }
    CG_EXPECT_EQ(mismatches, u64{0});
}

CG_TEST(software_shader_depth_only_draw_rejects_invalid_input)
{
    HeapAllocator heap;
    auto created = DepthBuffer::Create(DepthBufferDesc{16, 16, 0}, heap);
    CG_REQUIRE(static_cast<bool>(created));
    DepthBuffer depth = std::move(created.Value());
    const Mat4 identity = Mat4::Identity();
    const VertexInput vertices[3] = {};

    DepthOnlyShaderDesc desc{};
    desc.pipeline.viewport = Viewport{0.0f, 0.0f, 16.0f, 16.0f};
    desc.userData = &identity;
    PipelineStats stats{};
    stats.pixels = 9;

    // Null vertex stage.
    CG_EXPECT_FALSE(static_cast<bool>(DrawDepthOnlyTriangles(
        desc, Span<const VertexInput>(vertices), depth, nullptr, heap, &stats)));
    CG_EXPECT_EQ(stats.pixels, u64{0});

    desc.vertexShader = &TransformDepthOnlyVertexShader;
    DepthBuffer invalid{};
    CG_EXPECT_FALSE(static_cast<bool>(DrawDepthOnlyTriangles(
        desc, Span<const VertexInput>(vertices), invalid, nullptr, heap, &stats)));

    // A vertex count that is not a multiple of three.
    CG_EXPECT_FALSE(static_cast<bool>(DrawDepthOnlyTriangles(
        desc, Span<const VertexInput>(vertices, 2), depth, nullptr, heap, &stats)));

    DepthOnlyShaderDesc badViewport = desc;
    badViewport.pipeline.viewport = Viewport{0.0f, 0.0f, 0.0f, 16.0f};
    CG_EXPECT_FALSE(static_cast<bool>(DrawDepthOnlyTriangles(
        badViewport, Span<const VertexInput>(vertices), depth, nullptr, heap, &stats)));

    // An empty draw succeeds and writes nothing.
    CG_EXPECT_TRUE(static_cast<bool>(
        DrawDepthOnlyTriangles(desc, Span<const VertexInput>{}, depth, nullptr, heap, &stats)));
    CG_EXPECT_EQ(stats.pixels, u64{0});

    // A null user pointer makes the built-in shader emit origin vertices, which
    // the pipeline then rejects rather than rasterizing.
    desc.userData = nullptr;
    PipelineStats nullDataStats{};
    CG_REQUIRE(static_cast<bool>(DrawDepthOnlyTriangles(desc, Span<const VertexInput>(vertices),
                                                        depth, nullptr, heap, &nullDataStats)));
    CG_EXPECT_EQ(nullDataStats.pixels, u64{0});
}
