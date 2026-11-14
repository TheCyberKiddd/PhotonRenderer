#include "reference_renderer.h"

#include <cyber/math/mat4.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/software/antialias.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/image_io.h>
#include <cyber/software/pbr.h>
#include <cyber/software/shader.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

using namespace cyber::memory;
using namespace cyber::software;
using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::usize;
using cyber::math::Color;
using cyber::math::Mat4;
using cyber::math::Vec3;

namespace
{

constexpr i32 kWidth = 480;
constexpr i32 kHeight = 360;
constexpr u32 kGridSize = 7;
constexpr u32 kLongitudeSegments = 48;
constexpr u32 kLatitudeSegments = 32;
constexpr f32 kSphereRadius = 0.58f;
constexpr f32 kSphereSpacing = 1.6f;
constexpr f32 kCameraFov = 38.0f * cyber::math::kDegreesToRadians;
constexpr u32 kPathSamplesPerPixel = 64;
constexpr u32 kPathBounces = 4;
constexpr i32 kDefaultSupersample = 8;
constexpr i32 kMaximumSupersample = 15;
constexpr f32 kMaximumMeanSrgbError = 0.015f;
constexpr f32 kMaximumCoverageMismatch = 0.001f;

struct SceneUniforms
{
    TransformShaderData transform;
    PbrLightingData lighting;
};

struct Options
{
    const char* capturePath = nullptr;
    const char* referencePath = nullptr;
    f32 exposureEv = 0.0f;
    u32 samplesPerPixel = kPathSamplesPerPixel;
    i32 supersample = kDefaultSupersample;
};

void SceneVertex(const VertexInput& input, const void* userData, VertexStageOutput& output) noexcept
{
    TransformVertexShader(input, &static_cast<const SceneUniforms*>(userData)->transform, output);
}

Color ScenePixel(const Varyings& varyings, const void* userData) noexcept
{
    return PbrPixelShader(varyings, &static_cast<const SceneUniforms*>(userData)->lighting);
}

void PrintUsage()
{
    std::printf("usage: cyber_sample_lighting --capture <path.ppm|path.tga> "
                "[--reference <path.ppm|path.tga>] [--exposure <stops>] "
                "[--samples <count>] [--supersample <1-15>]\n");
    std::printf("  Renders a 7x7 metallic-roughness sphere grid and compares it with "
                "an independent CPU path-traced reference.\n");
    std::printf("  Supersampling defaults to 8x per axis and box-filters to 480x360.\n");
}

bool ParseOptions(int argc, char** argv, Options& options)
{
    for (int index = 1; index < argc; ++index)
    {
        if (std::strcmp(argv[index], "--capture") == 0)
        {
            if (index + 1 >= argc)
            {
                return false;
            }
            options.capturePath = argv[++index];
        }
        else if (std::strcmp(argv[index], "--reference") == 0)
        {
            if (index + 1 >= argc)
            {
                return false;
            }
            options.referencePath = argv[++index];
        }
        else if (std::strcmp(argv[index], "--exposure") == 0)
        {
            if (index + 1 >= argc)
            {
                return false;
            }
            char* end = nullptr;
            const f32 value = std::strtof(argv[++index], &end);
            if (end == argv[index] || *end != '\0' || !std::isfinite(value) || value < -16.0f ||
                value > 16.0f)
            {
                return false;
            }
            options.exposureEv = value;
        }
        else if (std::strcmp(argv[index], "--samples") == 0)
        {
            if (index + 1 >= argc)
            {
                return false;
            }
            char* end = nullptr;
            const unsigned long count = std::strtoul(argv[++index], &end, 10);
            if (end == argv[index] || *end != '\0' || count == 0 || count > 4096)
            {
                return false;
            }
            options.samplesPerPixel = static_cast<u32>(count);
        }
        else if (std::strcmp(argv[index], "--supersample") == 0)
        {
            if (index + 1 >= argc)
            {
                return false;
            }
            char* end = nullptr;
            const long factor = std::strtol(argv[++index], &end, 10);
            if (end == argv[index] || *end != '\0' || factor < 1 || factor > kMaximumSupersample)
            {
                return false;
            }
            options.supersample = static_cast<i32>(factor);
        }
        else
        {
            return false;
        }
    }
    const u32 supersampleArea = static_cast<u32>(options.supersample * options.supersample);
    return options.capturePath != nullptr && options.samplesPerPixel >= supersampleArea &&
           options.samplesPerPixel % supersampleArea == 0;
}

bool BuildSphereMesh(DynamicArray<VertexInput>& vertices)
{
    const auto makeVertex = [](u32 latitude, u32 longitude) noexcept
    {
        const f32 theta =
            cyber::math::kPi * static_cast<f32>(latitude) / static_cast<f32>(kLatitudeSegments);
        const f32 phi = 2.0f * cyber::math::kPi * static_cast<f32>(longitude) /
                        static_cast<f32>(kLongitudeSegments);
        const f32 sinTheta = std::sin(theta);
        const f32 cosTheta = std::cos(theta);
        const f32 sinPhi = std::sin(phi);
        const f32 cosPhi = std::cos(phi);

        VertexInput vertex{};
        vertex.position = Vec3{sinTheta * cosPhi, cosTheta, sinTheta * sinPhi};
        vertex.normal = vertex.position;
        vertex.tangent = Vec3{-sinPhi, 0.0f, cosPhi};
        vertex.tangentSign = 1.0f;
        vertex.uv =
            cyber::math::Vec2{static_cast<f32>(longitude) / static_cast<f32>(kLongitudeSegments),
                              static_cast<f32>(latitude) / static_cast<f32>(kLatitudeSegments)};
        vertex.color = Color{1.0f, 1.0f, 1.0f, 1.0f};
        return vertex;
    };

    for (u32 latitude = 0; latitude < kLatitudeSegments; ++latitude)
    {
        for (u32 longitude = 0; longitude < kLongitudeSegments; ++longitude)
        {
            const VertexInput a = makeVertex(latitude, longitude);
            const VertexInput b = makeVertex(latitude, longitude + 1);
            const VertexInput c = makeVertex(latitude + 1, longitude + 1);
            const VertexInput d = makeVertex(latitude + 1, longitude);
            if (!vertices.TryPushBack(a) || !vertices.TryPushBack(b) || !vertices.TryPushBack(c) ||
                !vertices.TryPushBack(a) || !vertices.TryPushBack(c) || !vertices.TryPushBack(d))
            {
                return false;
            }
        }
    }
    return true;
}

bool CreateTextures(HeapAllocator& allocator, Texture& baseColor, Texture& normal)
{
    TextureDesc baseDesc{};
    baseDesc.width = 1;
    baseDesc.height = 1;
    baseDesc.format = TextureFormat::RGBA32F;
    baseDesc.generateMips = false;
    auto baseResult = Texture::Create(baseDesc, allocator);
    if (!baseResult)
    {
        std::printf("failed to allocate base-color texture\n");
        return false;
    }
    baseColor = std::move(baseResult.Value());
    const Color authoredSrgb{0.88f, 0.82f, 0.70f, 1.0f};
    baseColor.SetTexel(0, 0, 0, cyber::math::FromSrgb(authoredSrgb));

    constexpr i32 kNormalSize = 64;
    TextureDesc normalDesc{};
    normalDesc.width = kNormalSize;
    normalDesc.height = kNormalSize;
    normalDesc.format = TextureFormat::RGBA32F;
    normalDesc.generateMips = true;
    auto normalResult = Texture::Create(normalDesc, allocator);
    if (!normalResult)
    {
        std::printf("failed to allocate tangent-space normal map\n");
        return false;
    }
    normal = std::move(normalResult.Value());
    for (i32 y = 0; y < kNormalSize; ++y)
    {
        for (i32 x = 0; x < kNormalSize; ++x)
        {
            const f32 u = (static_cast<f32>(x) + 0.5f) / kNormalSize;
            const f32 v = (static_cast<f32>(y) + 0.5f) / kNormalSize;
            const f32 tangentX = 0.12f * std::sin(2.0f * cyber::math::kPi * 5.0f * u) *
                                 std::cos(2.0f * cyber::math::kPi * 3.0f * v);
            const f32 tangentY = 0.12f * std::cos(2.0f * cyber::math::kPi * 4.0f * u) *
                                 std::sin(2.0f * cyber::math::kPi * 4.0f * v);
            const f32 tangentZ = std::sqrt(1.0f - tangentX * tangentX - tangentY * tangentY);
            normal.SetTexel(0, x, y,
                            Color{tangentX * 0.5f + 0.5f, tangentY * 0.5f + 0.5f,
                                  tangentZ * 0.5f + 0.5f, 1.0f});
        }
    }
    return true;
}

PbrLightingData MakeLighting(const Vec3& eyePosition, u32 column, u32 row, const Texture& baseColor,
                             const Texture& normal)
{
    PbrLightingData data{};
    data.eyePosition = eyePosition;
    data.ambientIrradiance = Vec3{0.12f, 0.13f, 0.16f};
    data.material.baseColor = Color{0.72f, 0.74f, 0.78f, 1.0f};
    data.material.metallic = static_cast<f32>(column) / static_cast<f32>(kGridSize - 1);
    data.material.roughness =
        0.12f + 0.88f * static_cast<f32>(row) / static_cast<f32>(kGridSize - 1);
    data.material.normalScale = 1.0f;
    data.baseColorTexture = &baseColor;
    data.normalTexture = &normal;

    data.directional[0].direction = Vec3{-0.45f, -0.75f, -0.48f};
    data.directional[0].radiance = Vec3{3.5f, 3.2f, 2.8f};
    data.directionalCount = 1;

    data.points[0].position = Vec3{-4.0f, 5.0f, 8.0f};
    data.points[0].intensity = Vec3{95.0f, 105.0f, 125.0f};
    data.points[0].range = 24.0f;
    data.pointCount = 1;

    data.spots[0].position = Vec3{4.0f, 4.5f, 8.0f};
    data.spots[0].direction = cyber::math::Normalize(Vec3{-0.3f, -0.3f, -1.0f});
    data.spots[0].intensity = Vec3{145.0f, 95.0f, 65.0f};
    data.spots[0].range = 24.0f;
    data.spots[0].innerConeRadians = 0.26f;
    data.spots[0].outerConeRadians = 0.55f;
    data.spotCount = 1;
    return data;
}

Vec3 SphereCenter(u32 column, u32 row) noexcept
{
    const f32 halfGrid = static_cast<f32>(kGridSize - 1) * 0.5f;
    return Vec3{(static_cast<f32>(column) - halfGrid) * kSphereSpacing,
                (halfGrid - static_cast<f32>(row)) * kSphereSpacing, 0.0f};
}

bool CreateTargets(HeapAllocator& allocator, Framebuffer& color, DepthBuffer& depth, i32 width,
                   i32 height)
{
    FramebufferDesc colorDesc{};
    colorDesc.width = width;
    colorDesc.height = height;
    colorDesc.format = PixelFormat::RGBA32F;
    auto colorResult = Framebuffer::Create(colorDesc, allocator);
    if (!colorResult)
    {
        return false;
    }
    color = std::move(colorResult.Value());

    DepthBufferDesc depthDesc{};
    depthDesc.width = width;
    depthDesc.height = height;
    auto depthResult = DepthBuffer::Create(depthDesc, allocator);
    if (!depthResult)
    {
        return false;
    }
    depth = std::move(depthResult.Value());
    return true;
}

// Resolves through the renderer-wide path in cyber/software/antialias.h.
//
// The filter is the box kernel rather than the renderer default: this sample
// measures the forward render against an independently path-traced reference,
// and a box resolve is the unbiased mean of the covered samples, which is the
// same quantity the path tracer estimates. A wider kernel would shift both
// images and make the reported error a property of the filter.
bool ResolveDownsample(const Framebuffer& source, Framebuffer& destination, i32 factor,
                       HeapAllocator& allocator) noexcept
{
    ResolveDesc desc{};
    desc.factor = factor;
    desc.filter = ResolveFilter::Box;
    desc.useJobs = false;
    return ResolveSupersampled(source, destination, desc, nullptr, allocator).IsOk();
}

void ApplyToneMapping(Framebuffer& framebuffer, f32 exposureEv) noexcept
{
    for (i32 y = 0; y < framebuffer.Height(); ++y)
    {
        for (i32 x = 0; x < framebuffer.Width(); ++x)
        {
            framebuffer.SetPixel(x, y, ToneMapAces(framebuffer.GetPixel(x, y), exposureEv));
        }
    }
}

bool WriteCapture(const char* path, const Framebuffer& framebuffer, HeapAllocator& allocator)
{
    ImageWriteDesc desc{};
    desc.encodeSrgb = true;
    const auto result = WriteImage(path, framebuffer, allocator, desc);
    if (!result)
    {
        std::printf("failed to write image: %s\n", path);
        return false;
    }
    return true;
}

bool RenderForward(const DynamicArray<VertexInput>& sphere, HeapAllocator& allocator,
                   Framebuffer& color, DepthBuffer& depth, const PbrLightingData* materials)
{
    const Vec3 eyePosition{0.0f, 0.0f, 20.0f};
    const Mat4 view = Mat4::LookAt(eyePosition, Vec3::Zero(), Vec3{0.0f, 1.0f, 0.0f});
    const f32 width = static_cast<f32>(color.Width());
    const f32 height = static_cast<f32>(color.Height());
    const Mat4 projection = Mat4::PerspectiveReverseZ(kCameraFov, width / height, 0.1f, 100.0f);
    color.Clear(Color{0.012f, 0.016f, 0.025f, 0.0f});
    depth.Clear();

    for (u32 row = 0; row < kGridSize; ++row)
    {
        for (u32 column = 0; column < kGridSize; ++column)
        {
            SceneUniforms uniforms{};
            const Mat4 model =
                Mat4::Translation(SphereCenter(column, row)) * Mat4::Scale(kSphereRadius);
            uniforms.transform.modelToWorld = model;
            uniforms.transform.normalToWorld = Mat4::Scale(1.0f / kSphereRadius);
            uniforms.transform.modelViewProjection = projection * view * model;
            uniforms.transform.eyePosition = eyePosition;
            uniforms.lighting = materials[static_cast<usize>(row) * kGridSize + column];

            ShaderDesc shader{};
            shader.pipeline.viewport = Viewport{0.0f, 0.0f, width, height};
            shader.pipeline.cullMode = CullMode::Back;
            shader.vertexShader = SceneVertex;
            shader.pixelShader = ScenePixel;
            shader.userData = &uniforms;
            const auto status =
                DrawShadedTriangles(shader, Span<const VertexInput>(sphere.Data(), sphere.Size()),
                                    color, depth, nullptr, allocator);
            if (!status)
            {
                std::printf("forward sphere rendering failed at row %u, column %u\n", row, column);
                return false;
            }
        }
    }
    return true;
}

bool ReportDifference(const Framebuffer& forward, const Framebuffer& reference)
{
    u64 channelError = 0;
    u64 coverageMismatch = 0;
    for (i32 y = 0; y < forward.Height(); ++y)
    {
        for (i32 x = 0; x < forward.Width(); ++x)
        {
            const Color a = forward.GetPixel(x, y);
            const Color b = reference.GetPixel(x, y);
            channelError += static_cast<u64>(
                std::abs(static_cast<i32>(EncodeSrgb8(a.r)) - static_cast<i32>(EncodeSrgb8(b.r))));
            channelError += static_cast<u64>(
                std::abs(static_cast<i32>(EncodeSrgb8(a.g)) - static_cast<i32>(EncodeSrgb8(b.g))));
            channelError += static_cast<u64>(
                std::abs(static_cast<i32>(EncodeSrgb8(a.b)) - static_cast<i32>(EncodeSrgb8(b.b))));
            coverageMismatch += (a.a > 0.5f) != (b.a > 0.5f) ? 1u : 0u;
        }
    }
    const f32 sampleCount = static_cast<f32>(forward.Width()) * forward.Height();
    const f32 meanAbsoluteError = static_cast<f32>(channelError) / (sampleCount * 3.0f * 255.0f);
    const f32 coverageError = static_cast<f32>(coverageMismatch) / sampleCount;
    std::printf("independent path-traced reference: mean sRGB absolute error %.5f, "
                "coverage mismatch %.3f%% (limits %.3f and %.3f%%)\n",
                static_cast<double>(meanAbsoluteError), static_cast<double>(coverageError * 100.0f),
                static_cast<double>(kMaximumMeanSrgbError),
                static_cast<double>(kMaximumCoverageMismatch * 100.0f));
    return meanAbsoluteError <= kMaximumMeanSrgbError && coverageError <= kMaximumCoverageMismatch;
}

}  // namespace

int main(int argc, char** argv)
{
    Options options{};
    if (!ParseOptions(argc, argv, options))
    {
        PrintUsage();
        return 2;
    }

    HeapAllocator allocator;
    Texture baseColor;
    Texture normal;
    if (!CreateTextures(allocator, baseColor, normal))
    {
        return 1;
    }

    const Vec3 eyePosition{0.0f, 0.0f, 20.0f};
    PbrLightingData materials[kGridSize * kGridSize]{};
    for (u32 row = 0; row < kGridSize; ++row)
    {
        for (u32 column = 0; column < kGridSize; ++column)
        {
            materials[static_cast<usize>(row) * kGridSize + column] =
                MakeLighting(eyePosition, column, row, baseColor, normal);
        }
    }

    DynamicArray<VertexInput> sphere(allocator);
    if (!BuildSphereMesh(sphere))
    {
        std::printf("failed to allocate sphere mesh\n");
        return 1;
    }

    const i32 renderWidth = kWidth * options.supersample;
    const i32 renderHeight = kHeight * options.supersample;
    Framebuffer forwardRender;
    Framebuffer forward;
    DepthBuffer depth;
    Framebuffer referenceRender;
    Framebuffer reference;
    if (!CreateTargets(allocator, forwardRender, depth, renderWidth, renderHeight))
    {
        std::printf("failed to allocate forward render targets\n");
        return 1;
    }
    FramebufferDesc outputDesc{};
    outputDesc.width = kWidth;
    outputDesc.height = kHeight;
    outputDesc.format = PixelFormat::RGBA32F;
    auto forwardResult = Framebuffer::Create(outputDesc, allocator);
    if (!forwardResult)
    {
        std::printf("failed to allocate resolved forward framebuffer\n");
        return 1;
    }
    forward = std::move(forwardResult.Value());
    auto referenceRenderResult = Framebuffer::Create(
        FramebufferDesc{renderWidth, renderHeight, PixelFormat::RGBA32F}, allocator);
    if (!referenceRenderResult)
    {
        std::printf("failed to allocate supersampled reference framebuffer\n");
        return 1;
    }
    referenceRender = std::move(referenceRenderResult.Value());
    FramebufferDesc referenceDesc{};
    referenceDesc.width = kWidth;
    referenceDesc.height = kHeight;
    referenceDesc.format = PixelFormat::RGBA32F;
    auto referenceResult = Framebuffer::Create(referenceDesc, allocator);
    if (!referenceResult)
    {
        std::printf("failed to allocate reference framebuffer\n");
        return 1;
    }
    reference = std::move(referenceResult.Value());

    const auto start = std::chrono::steady_clock::now();
    if (!RenderForward(sphere, allocator, forwardRender, depth, materials))
    {
        return 1;
    }
    SphereGridPathTraceDesc traceDesc{};
    traceDesc.eyePosition = eyePosition;
    traceDesc.verticalFovRadians = kCameraFov;
    traceDesc.sphereRadius = kSphereRadius;
    traceDesc.sphereSpacing = kSphereSpacing;
    traceDesc.gridSize = kGridSize;
    const u32 supersampleArea = static_cast<u32>(options.supersample * options.supersample);
    traceDesc.samplesPerPixel = std::max(1u, options.samplesPerPixel / supersampleArea);
    traceDesc.maxBounces = kPathBounces;
    traceDesc.seed = 0xC0FFEE26ull;
    if (!RenderSphereGridPathTrace(referenceRender,
                                   Span<const PbrLightingData>(materials, kGridSize * kGridSize),
                                   traceDesc))
    {
        std::printf("independent path-traced reference failed\n");
        return 1;
    }
    if (!ResolveDownsample(forwardRender, forward, options.supersample, allocator) ||
        !ResolveDownsample(referenceRender, reference, options.supersample, allocator))
    {
        std::printf("failed to resolve supersampled render targets\n");
        return 1;
    }
    ApplyToneMapping(forward, options.exposureEv);
    ApplyToneMapping(reference, options.exposureEv);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    std::printf("%u path samples per output pixel, %dx supersampling (%dx%d render), "
                "%u maximum bounces, seed 0x%llX, %lld ms\n",
                traceDesc.samplesPerPixel * supersampleArea, options.supersample, renderWidth,
                renderHeight, kPathBounces, static_cast<unsigned long long>(traceDesc.seed),
                static_cast<long long>(elapsed.count()));
    const bool withinThreshold = ReportDifference(forward, reference);
    if (!WriteCapture(options.capturePath, forward, allocator))
    {
        return 1;
    }
    if (options.referencePath != nullptr &&
        !WriteCapture(options.referencePath, reference, allocator))
    {
        return 1;
    }
    return withinThreshold ? 0 : 1;
}
