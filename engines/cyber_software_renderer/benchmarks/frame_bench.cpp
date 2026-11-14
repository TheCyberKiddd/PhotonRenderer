// Whole-frame throughput benchmark for the software renderer.
//
// Renders a representative game-like scene (a grid of lit, textured spheres on
// a ground plane) through the shaded pipeline and reports frame time and frames
// per second at several target resolutions. Three workloads isolate cost:
//
//   flat   - vertex-colored pipeline, no lighting and no texturing
//   pbr    - Cook-Torrance forward shading, one directional light
//   pbr+4  - the same plus four shadowless point lights
//
// Each workload also reports the present pack (linear -> sRGB8) cost, which a
// real game pays once per frame before blitting to the window.

#include <cyber/jobs/job_system.h>
#include <cyber/math/constants.h>
#include <cyber/math/mat4.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/software/camera.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/pbr.h>
#include <cyber/software/pipeline.h>
#include <cyber/software/present_target.h>
#include <cyber/software/shader.h>
#include <cyber/software/texture.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::math::Color;
using cyber::math::Mat4;
using cyber::math::Vec2;
using cyber::math::Vec3;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using namespace cyber::software;

constexpr u32 kWarmupFrames = 8;
constexpr u32 kMeasuredFrames = 40;
constexpr u32 kGridSide = 6;         // 36 spheres
constexpr u32 kSphereSegments = 24;  // longitude
constexpr u32 kSphereRings = 12;     // latitude
constexpr i32 kTextureSize = 256;

struct Resolution
{
    const char* name;
    i32 width;
    i32 height;
};

constexpr Resolution kResolutions[] = {
    {"320x180", 320, 180},   {"640x360", 640, 360},     {"1280x720", 1280, 720},
    {"1600x900", 1600, 900}, {"1920x1080", 1920, 1080},
};

void AppendSphere(std::vector<VertexInput>& out, u32 segments, u32 rings)
{
    const auto position = [&](u32 ring, u32 segment)
    {
        const f32 v = static_cast<f32>(ring) / static_cast<f32>(rings);
        const f32 u = static_cast<f32>(segment) / static_cast<f32>(segments);
        const f32 phi = v * cyber::math::kPi;
        const f32 theta = u * 2.0f * cyber::math::kPi;
        const Vec3 normal{std::sin(phi) * std::cos(theta), std::cos(phi),
                          std::sin(phi) * std::sin(theta)};
        VertexInput vertex{};
        vertex.position = normal;
        vertex.normal = normal;
        vertex.uv = Vec2{u, v};
        vertex.color = Color{1.0f, 1.0f, 1.0f, 1.0f};
        vertex.tangent = Vec3{-std::sin(theta), 0.0f, std::cos(theta)};
        vertex.tangentSign = 1.0f;
        return vertex;
    };

    for (u32 ring = 0; ring < rings; ++ring)
    {
        for (u32 segment = 0; segment < segments; ++segment)
        {
            const VertexInput a = position(ring, segment);
            const VertexInput b = position(ring + 1U, segment);
            const VertexInput c = position(ring + 1U, segment + 1U);
            const VertexInput d = position(ring, segment + 1U);
            out.push_back(a);
            out.push_back(b);
            out.push_back(c);
            out.push_back(a);
            out.push_back(c);
            out.push_back(d);
        }
    }
}

void AppendGround(std::vector<VertexInput>& out, f32 extent)
{
    const auto corner = [&](f32 x, f32 z, f32 u, f32 v)
    {
        VertexInput vertex{};
        vertex.position = Vec3{x * extent, 0.0f, z * extent};
        vertex.normal = Vec3{0.0f, 1.0f, 0.0f};
        vertex.uv = Vec2{u * 8.0f, v * 8.0f};
        vertex.color = Color{0.6f, 0.62f, 0.66f, 1.0f};
        vertex.tangent = Vec3{1.0f, 0.0f, 0.0f};
        return vertex;
    };
    const VertexInput a = corner(-1.0f, -1.0f, 0.0f, 0.0f);
    const VertexInput b = corner(-1.0f, 1.0f, 0.0f, 1.0f);
    const VertexInput c = corner(1.0f, 1.0f, 1.0f, 1.0f);
    const VertexInput d = corner(1.0f, -1.0f, 1.0f, 0.0f);
    out.push_back(a);
    out.push_back(c);
    out.push_back(b);
    out.push_back(a);
    out.push_back(d);
    out.push_back(c);
}

// One uniform block per draw, matching the sample convention: the vertex and
// pixel stages share a single user-data pointer.
struct SceneUniforms
{
    TransformShaderData transform{};
    PbrLightingData lighting{};
};

void SceneVertexShader(const VertexInput& input, const void* userData,
                       VertexStageOutput& output) noexcept
{
    TransformVertexShader(input, &static_cast<const SceneUniforms*>(userData)->transform, output);
}

[[nodiscard]] Color ScenePbrPixelShader(const Varyings& varyings, const void* userData) noexcept
{
    return PbrPixelShader(varyings, &static_cast<const SceneUniforms*>(userData)->lighting);
}

[[nodiscard]] Color FlatPixelShader(const Varyings& varyings, const void*) noexcept
{
    return varyings.color;
}

struct Timing
{
    double meanMs = 0.0;
    double minMs = 0.0;
    double p95Ms = 0.0;
};

[[nodiscard]] Timing Summarize(std::vector<double>& samples)
{
    Timing timing{};
    if (samples.empty())
    {
        return timing;
    }
    std::sort(samples.begin(), samples.end());
    double total = 0.0;
    for (const double sample : samples)
    {
        total += sample;
    }
    timing.meanMs = total / static_cast<double>(samples.size());
    timing.minMs = samples.front();
    const auto index = static_cast<std::size_t>(static_cast<double>(samples.size() - 1U) * 0.95);
    timing.p95Ms = samples[index];
    return timing;
}

enum class Workload
{
    Flat,
    Pbr,
    PbrFourPoints,
};

[[nodiscard]] const char* WorkloadName(Workload workload) noexcept
{
    switch (workload)
    {
    case Workload::Flat:
        return "flat";
    case Workload::Pbr:
        return "pbr (1 dir)";
    case Workload::PbrFourPoints:
        return "pbr (1 dir + 4 point)";
    }
    return "?";
}

}  // namespace

int main()
{
    HeapAllocator allocator;

    const u32 hardwareThreads = std::max(1U, std::thread::hardware_concurrency());
    cyber::jobs::JobSystem::Desc jobDesc{};
    jobDesc.workerCount = hardwareThreads > 1U ? hardwareThreads - 1U : 1U;
    auto jobSystemResult = cyber::jobs::JobSystem::Create(jobDesc, allocator);
    if (!jobSystemResult)
    {
        std::fprintf(stderr, "frame bench: job system creation failed: %s\n",
                     jobSystemResult.Error().message);
        return 1;
    }
    auto jobSystem = std::move(jobSystemResult.Value());

    // Geometry: one sphere mesh instanced over a grid, plus a ground quad.
    std::vector<VertexInput> sphere;
    AppendSphere(sphere, kSphereSegments, kSphereRings);
    std::vector<VertexInput> ground;
    AppendGround(ground, 30.0f);

    const u64 sphereTriangles = static_cast<u64>(sphere.size()) / 3U;
    const u64 sceneTriangles = sphereTriangles * kGridSide * kGridSide + 2U;

    // Base-color texture with a full mip chain, so texturing pays real cache
    // traffic rather than hitting one hot cache line.
    TextureDesc textureDesc{};
    textureDesc.width = kTextureSize;
    textureDesc.height = kTextureSize;
    textureDesc.format = TextureFormat::RGBA8Unorm;
    textureDesc.generateMips = true;
    auto textureResult = Texture::Create(textureDesc, allocator);
    if (!textureResult)
    {
        std::fprintf(stderr, "frame bench: texture creation failed: %s\n",
                     textureResult.Error().message);
        return 1;
    }
    auto texture = std::move(textureResult.Value());
    for (i32 y = 0; y < kTextureSize; ++y)
    {
        for (i32 x = 0; x < kTextureSize; ++x)
        {
            const bool checker = (((x >> 4) + (y >> 4)) & 1) != 0;
            const f32 noise = static_cast<f32>((x * 1664525 + y * 1013904223) & 0xFF) / 255.0f;
            const Color color =
                checker ? Color{0.78f, 0.72f, 0.60f, 1.0f} : Color{0.22f, 0.28f, 0.36f, 1.0f};
            texture.SetTexel(0, x, y,
                             Color{color.r * (0.85f + 0.15f * noise),
                                   color.g * (0.85f + 0.15f * noise),
                                   color.b * (0.85f + 0.15f * noise), 1.0f});
        }
    }
    texture.GenerateMips();

    std::printf("CyberGraphics software renderer - whole-frame benchmark\n");
    std::printf("hardware threads: %u (1 main + %u workers)\n", hardwareThreads,
                jobDesc.workerCount);
    std::printf("scene: %llu triangles (%u spheres of %llu tris + ground), 256x256 mipped "
                "base color\n",
                static_cast<unsigned long long>(sceneTriangles), kGridSide * kGridSide,
                static_cast<unsigned long long>(sphereTriangles));
    std::printf("frames: %u warmup + %u measured per case\n\n", kWarmupFrames, kMeasuredFrames);
    std::printf("%-22s %-10s %10s %10s %10s %10s %12s\n", "workload", "res", "mean ms", "min ms",
                "p95 ms", "fps", "present ms");

    const Workload workloads[] = {Workload::Flat, Workload::Pbr, Workload::PbrFourPoints};

    for (const Workload workload : workloads)
    {
        for (const Resolution& resolution : kResolutions)
        {
            FramebufferDesc colorDesc{};
            colorDesc.width = resolution.width;
            colorDesc.height = resolution.height;
            colorDesc.format = PixelFormat::RGBA32F;
            auto colorResult = Framebuffer::Create(colorDesc, allocator);
            if (!colorResult)
            {
                std::fprintf(stderr, "frame bench: framebuffer creation failed: %s\n",
                             colorResult.Error().message);
                return 1;
            }
            auto color = std::move(colorResult.Value());

            DepthBufferDesc depthDesc{};
            depthDesc.width = resolution.width;
            depthDesc.height = resolution.height;
            auto depthResult = DepthBuffer::Create(depthDesc, allocator);
            if (!depthResult)
            {
                std::fprintf(stderr, "frame bench: depth creation failed: %s\n",
                             depthResult.Error().message);
                return 1;
            }
            auto depth = std::move(depthResult.Value());

            std::vector<cyber::core::u8> presentPixels(
                PresentByteSize(resolution.width, resolution.height), 0U);

            Camera camera{};
            camera.position = Vec3{0.0f, 4.5f, 16.0f};
            camera.pitchRadians = -0.18f;
            camera.fovYRadians = 60.0f * cyber::math::kDegreesToRadians;
            camera.nearPlane = 0.1f;
            camera.farPlane = 200.0f;
            const f32 aspect =
                static_cast<f32>(resolution.width) / static_cast<f32>(resolution.height);
            const Mat4 view = CameraView(camera);
            const Mat4 projection = CameraProjection(camera, aspect);

            PbrLightingData lighting{};
            lighting.eyePosition = camera.position;
            lighting.cameraForward = CameraForward(camera);
            lighting.ambientIrradiance = Vec3{0.05f, 0.055f, 0.07f};
            lighting.material.baseColor = Color{0.9f, 0.85f, 0.8f, 1.0f};
            lighting.material.metallic = 0.25f;
            lighting.material.roughness = 0.4f;
            lighting.baseColorTexture = &texture;
            lighting.sampler = SamplerDesc{};
            lighting.directional[0].direction = Vec3{-0.4f, -1.0f, -0.3f};
            lighting.directional[0].radiance = Vec3{3.0f, 2.9f, 2.7f};
            lighting.directionalCount = 1;
            if (workload == Workload::PbrFourPoints)
            {
                for (u32 light = 0; light < 4U; ++light)
                {
                    const f32 angle = static_cast<f32>(light) * 0.5f * cyber::math::kPi;
                    lighting.points[light].position =
                        Vec3{std::cos(angle) * 9.0f, 3.0f, std::sin(angle) * 9.0f};
                    lighting.points[light].intensity = Vec3{18.0f, 14.0f, 10.0f};
                    lighting.points[light].range = 22.0f;
                }
                lighting.pointCount = 4;
            }

            PipelineDesc pipeline{};
            pipeline.view = view;
            pipeline.projection = projection;
            pipeline.viewport = Viewport{0.0f, 0.0f, static_cast<f32>(resolution.width),
                                         static_cast<f32>(resolution.height)};
            pipeline.cullMode = CullMode::Back;
            pipeline.tileSize = 64;
            pipeline.useJobs = true;

            std::vector<double> frameSamples;
            std::vector<double> presentSamples;
            frameSamples.reserve(kMeasuredFrames);
            presentSamples.reserve(kMeasuredFrames);
            u64 shadedPixels = 0;
            u64 submittedTriangles = 0;

            for (u32 frame = 0; frame < kWarmupFrames + kMeasuredFrames; ++frame)
            {
                const f32 spin = static_cast<f32>(frame) * 0.02f;
                const auto frameStart = std::chrono::steady_clock::now();

                color.Clear(Color{0.03f, 0.04f, 0.06f, 1.0f});
                depth.Clear(0.0f);

                u64 framePixels = 0;
                u64 frameTriangles = 0;

                const auto draw = [&](const std::vector<VertexInput>& mesh, const Mat4& model)
                {
                    SceneUniforms uniforms{};
                    uniforms.lighting = lighting;
                    uniforms.transform.modelViewProjection = projection * view * model;
                    uniforms.transform.modelToWorld = model;
                    Mat4 inverse = Mat4::Identity();
                    if (model.TryInverse(inverse))
                    {
                        uniforms.transform.normalToWorld = inverse.Transpose();
                    }
                    uniforms.transform.eyePosition = camera.position;

                    ShaderDesc shaderDesc{};
                    shaderDesc.pipeline = pipeline;
                    shaderDesc.pipeline.model = model;
                    shaderDesc.vertexShader = &SceneVertexShader;
                    shaderDesc.pixelShader =
                        workload == Workload::Flat ? &FlatPixelShader : &ScenePbrPixelShader;
                    shaderDesc.userData = &uniforms;
                    PipelineStats stats{};
                    const auto status = DrawShadedTriangles(
                        shaderDesc, Span<const VertexInput>(mesh.data(), mesh.size()), color, depth,
                        &jobSystem, allocator, &stats);
                    if (!status)
                    {
                        std::fprintf(stderr, "frame bench: draw failed: %s\n",
                                     status.Error().message);
                        std::exit(1);
                    }
                    framePixels += stats.pixels;
                    frameTriangles += stats.trianglesSubmitted;
                };

                draw(ground, Mat4::Identity());
                for (u32 z = 0; z < kGridSide; ++z)
                {
                    for (u32 x = 0; x < kGridSide; ++x)
                    {
                        const f32 fx = (static_cast<f32>(x) - 2.5f) * 3.2f;
                        const f32 fz = (static_cast<f32>(z) - 2.5f) * 3.2f;
                        const Mat4 model = Mat4::Translation(Vec3{fx, 1.3f, fz}) *
                                           Mat4::RotationY(spin + static_cast<f32>(x + z)) *
                                           Mat4::Scale(1.2f);
                        draw(sphere, model);
                    }
                }

                const auto drawEnd = std::chrono::steady_clock::now();
                const auto packStatus = PackPresentPixels(
                    color, Span<cyber::core::u8>(presentPixels.data(), presentPixels.size()));
                if (!packStatus)
                {
                    std::fprintf(stderr, "frame bench: present pack failed: %s\n",
                                 packStatus.Error().message);
                    return 1;
                }
                const auto frameEnd = std::chrono::steady_clock::now();

                if (frame >= kWarmupFrames)
                {
                    frameSamples.push_back(
                        std::chrono::duration<double, std::milli>(frameEnd - frameStart).count());
                    presentSamples.push_back(
                        std::chrono::duration<double, std::milli>(frameEnd - drawEnd).count());
                    shadedPixels = framePixels;
                    submittedTriangles = frameTriangles;
                }
            }

            const Timing frameTiming = Summarize(frameSamples);
            const Timing presentTiming = Summarize(presentSamples);
            std::printf("%-22s %-10s %10.2f %10.2f %10.2f %10.1f %12.2f\n", WorkloadName(workload),
                        resolution.name, frameTiming.meanMs, frameTiming.minMs, frameTiming.p95Ms,
                        frameTiming.meanMs > 0.0 ? 1000.0 / frameTiming.meanMs : 0.0,
                        presentTiming.meanMs);
            std::fflush(stdout);
            if (resolution.width == 1280)
            {
                std::printf("    (1280x720 detail: %llu shaded samples, %llu triangles "
                            "rasterized)\n",
                            static_cast<unsigned long long>(shadedPixels),
                            static_cast<unsigned long long>(submittedTriangles));
            }
        }
        std::printf("\n");
    }

    // Fixed-cost probe: isolates per-frame and per-draw-call overhead from
    // pixel cost by shrinking the target to 320x180 and varying only the
    // number of draw calls and whether the tile loop is dispatched to workers.
    {
        FramebufferDesc colorDesc{};
        colorDesc.width = 320;
        colorDesc.height = 180;
        colorDesc.format = PixelFormat::RGBA32F;
        auto colorResult = Framebuffer::Create(colorDesc, allocator);
        DepthBufferDesc depthDesc{};
        depthDesc.width = 320;
        depthDesc.height = 180;
        auto depthResult = DepthBuffer::Create(depthDesc, allocator);
        if (!colorResult || !depthResult)
        {
            std::fprintf(stderr, "frame bench: probe target creation failed\n");
            return 1;
        }
        auto color = std::move(colorResult.Value());
        auto depth = std::move(depthResult.Value());

        Camera camera{};
        camera.position = Vec3{0.0f, 4.5f, 16.0f};
        camera.pitchRadians = -0.18f;
        const Mat4 view = CameraView(camera);
        const Mat4 projection = CameraProjection(camera, 320.0f / 180.0f);

        const auto measure = [&](u32 drawCalls, bool useJobs)
        {
            std::vector<double> samples;
            samples.reserve(kMeasuredFrames);
            for (u32 frame = 0; frame < kWarmupFrames + kMeasuredFrames; ++frame)
            {
                const auto start = std::chrono::steady_clock::now();
                color.Clear(Color{0.03f, 0.04f, 0.06f, 1.0f});
                depth.Clear(0.0f);
                for (u32 call = 0; call < drawCalls; ++call)
                {
                    const Mat4 model =
                        Mat4::Translation(Vec3{static_cast<f32>(call % 6U) * 3.2f - 8.0f, 1.3f,
                                               static_cast<f32>(call / 6U) * 3.2f - 8.0f});
                    SceneUniforms uniforms{};
                    uniforms.transform.modelViewProjection = projection * view * model;
                    uniforms.transform.modelToWorld = model;
                    uniforms.transform.normalToWorld = Mat4::Identity();
                    ShaderDesc shaderDesc{};
                    shaderDesc.pipeline.view = view;
                    shaderDesc.pipeline.projection = projection;
                    shaderDesc.pipeline.model = model;
                    shaderDesc.pipeline.viewport = Viewport{0.0f, 0.0f, 320.0f, 180.0f};
                    shaderDesc.pipeline.useJobs = useJobs;
                    shaderDesc.vertexShader = &SceneVertexShader;
                    shaderDesc.pixelShader = &FlatPixelShader;
                    shaderDesc.userData = &uniforms;
                    const auto status = DrawShadedTriangles(
                        shaderDesc, Span<const VertexInput>(sphere.data(), sphere.size()), color,
                        depth, useJobs ? &jobSystem : nullptr, allocator, nullptr);
                    if (!status)
                    {
                        std::fprintf(stderr, "frame bench: probe draw failed: %s\n",
                                     status.Error().message);
                        std::exit(1);
                    }
                }
                const auto end = std::chrono::steady_clock::now();
                if (frame >= kWarmupFrames)
                {
                    samples.push_back(
                        std::chrono::duration<double, std::milli>(end - start).count());
                }
            }
            return Summarize(samples);
        };

        std::printf("fixed-cost probe at 320x180, flat shading, %llu tris per draw call\n",
                    static_cast<unsigned long long>(sphereTriangles));
        std::printf("%-14s %-10s %10s %10s\n", "draw calls", "jobs", "mean ms", "ms/call");
        const u32 callCounts[] = {1U, 4U, 12U, 36U};
        for (const bool useJobs : {true, false})
        {
            for (const u32 calls : callCounts)
            {
                const Timing timing = measure(calls, useJobs);
                std::printf("%-14u %-10s %10.3f %10.3f\n", calls, useJobs ? "on" : "off",
                            timing.meanMs, timing.meanMs / static_cast<double>(calls));
                std::fflush(stdout);
            }
        }
    }

    return 0;
}
