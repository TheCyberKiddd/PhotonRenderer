// Phase 31 sample: baked global illumination in the test scene, with
// ray-traced direct shadows.
//
// A Cornell-style room is baked into an irradiance volume by the CPU path
// tracer, then rendered twice through the software rasterizer: once with the
// best flat ambient term a constant can give, and once with the baked volume.
// Direct light from the lamp is shadowed by one ray per sample against the
// same acceleration structure the bake used -- the software-reference
// ray-traced shadow technique. The two renders are written side by side.
//
// The sample also prints, for every Direct3D 12 adapter it can open, which
// technique the ray-tracing policy selects for shadows, so the capability gate
// can be checked on any machine.

#include <cyber/core/result.h>
#include <cyber/jobs/job_system.h>
#include <cyber/math/mat4.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/renderer/raytracing_path.h>
#include <cyber/rhi/backend.h>
#include <cyber/rhi/d3d12.h>
#include <cyber/rhi/device.h>
#include <cyber/software/antialias.h>
#include <cyber/software/denoise.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/gi_quality.h>
#include <cyber/software/global_illumination.h>
#include <cyber/software/image_io.h>
#include <cyber/software/pbr.h>
#include <cyber/software/raytrace.h>
#include <cyber/software/shader.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>

using namespace cyber::memory;
using namespace cyber::software;
using cyber::core::f32;
using cyber::core::f64;
using cyber::core::i32;
using cyber::core::u32;
using cyber::core::usize;
using cyber::math::Color;
using cyber::math::Mat4;
using cyber::math::Vec3;

namespace
{

constexpr i32 kWidth = 640;
constexpr i32 kHeight = 480;
constexpr u32 kGroupCount = 5;
const Vec3 kEye{2.0f, 1.5f, 9.2f};
const Vec3 kTarget{2.0f, 1.4f, 1.8f};

// One material group: the rasterized geometry and the matching light-transport
// material. Index in the group array is the instance identifier.
struct Group
{
    explicit Group(HeapAllocator& allocator) noexcept : vertices(allocator) {}

    DynamicArray<VertexInput> vertices;
    GiSurfaceMaterial material{};
};

bool Push(DynamicArray<VertexInput>& vertices, const Vec3& position, const Vec3& normal)
{
    VertexInput vertex{};
    vertex.position = position;
    vertex.normal = normal;
    return vertices.TryPushBack(vertex);
}

bool AddQuad(DynamicArray<VertexInput>& vertices, const Vec3& center, const Vec3& halfU,
             const Vec3& halfV, const Vec3& facing)
{
    Vec3 a = center - halfU - halfV;
    Vec3 b = center + halfU - halfV;
    const Vec3 c = center + halfU + halfV;
    Vec3 d = center - halfU + halfV;
    if (cyber::math::Dot(cyber::math::Cross(b - a, c - a), facing) < 0.0f)
    {
        std::swap(b, d);
    }
    const Vec3 n = cyber::math::Normalize(facing);
    return Push(vertices, a, n) && Push(vertices, b, n) && Push(vertices, c, n) &&
           Push(vertices, a, n) && Push(vertices, c, n) && Push(vertices, d, n);
}

bool AddSolidBox(DynamicArray<VertexInput>& vertices, const Vec3& minimum, const Vec3& maximum)
{
    const Vec3 center = (minimum + maximum) * 0.5f;
    const Vec3 half = (maximum - minimum) * 0.5f;
    const Vec3 x{half.x, 0.0f, 0.0f};
    const Vec3 y{0.0f, half.y, 0.0f};
    const Vec3 z{0.0f, 0.0f, half.z};
    return AddQuad(vertices, center + x, y, z, Vec3{1.0f, 0.0f, 0.0f}) &&
           AddQuad(vertices, center - x, y, z, Vec3{-1.0f, 0.0f, 0.0f}) &&
           AddQuad(vertices, center + y, x, z, Vec3{0.0f, 1.0f, 0.0f}) &&
           AddQuad(vertices, center - y, x, z, Vec3{0.0f, -1.0f, 0.0f}) &&
           AddQuad(vertices, center + z, x, y, Vec3{0.0f, 0.0f, 1.0f}) &&
           AddQuad(vertices, center - z, x, y, Vec3{0.0f, 0.0f, -1.0f});
}

// A 4 x 3 x 4 room, open toward the camera at z = 4: white floor, ceiling and
// back wall, a red wall on the left, a green wall on the right, and two white
// blocks.
bool BuildRoom(Group* groups)
{
    groups[0].material.albedo = Vec3{0.76f, 0.75f, 0.72f};
    groups[1].material.albedo = Vec3{0.72f, 0.07f, 0.06f};
    groups[2].material.albedo = Vec3{0.10f, 0.62f, 0.12f};
    groups[3].material.albedo = Vec3{0.76f, 0.75f, 0.72f};
    groups[4].material.albedo = Vec3{0.76f, 0.75f, 0.72f};
    DynamicArray<VertexInput>& white = groups[0].vertices;
    return AddQuad(white, Vec3{2.0f, 0.0f, 2.0f}, Vec3{2.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, 2.0f},
                   Vec3{0.0f, 1.0f, 0.0f}) &&
           AddQuad(white, Vec3{2.0f, 3.0f, 2.0f}, Vec3{2.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, 2.0f},
                   Vec3{0.0f, -1.0f, 0.0f}) &&
           AddQuad(white, Vec3{2.0f, 1.5f, 0.0f}, Vec3{2.0f, 0.0f, 0.0f}, Vec3{0.0f, 1.5f, 0.0f},
                   Vec3{0.0f, 0.0f, 1.0f}) &&
           AddQuad(groups[1].vertices, Vec3{0.0f, 1.5f, 2.0f}, Vec3{0.0f, 1.5f, 0.0f},
                   Vec3{0.0f, 0.0f, 2.0f}, Vec3{1.0f, 0.0f, 0.0f}) &&
           AddQuad(groups[2].vertices, Vec3{4.0f, 1.5f, 2.0f}, Vec3{0.0f, 1.5f, 0.0f},
                   Vec3{0.0f, 0.0f, 2.0f}, Vec3{-1.0f, 0.0f, 0.0f}) &&
           AddSolidBox(groups[3].vertices, Vec3{0.6f, 0.0f, 0.7f}, Vec3{1.6f, 1.9f, 1.7f}) &&
           AddSolidBox(groups[4].vertices, Vec3{2.3f, 0.0f, 1.9f}, Vec3{3.4f, 0.9f, 3.0f});
}

f32 SmoothRangeFade(f32 distance, f32 range) noexcept
{
    if (!(distance >= 0.0f) || !(range > 0.0f) || distance >= range)
    {
        return 0.0f;
    }
    const f32 ratio = distance / range;
    const f32 fade = 1.0f - ratio * ratio;
    return fade * fade;
}

struct SceneUniforms
{
    TransformShaderData transform{};
    // Ambient and indirect only: the lamp is added by the pixel shader below,
    // with a ray-traced shadow.
    PbrLightingData lighting{};
    const TopLevelAccelerationStructure* geometry = nullptr;
    GiPointLight lamp{};
};

void SceneVertex(const VertexInput& input, const void* userData, VertexStageOutput& output) noexcept
{
    TransformVertexShader(input, &static_cast<const SceneUniforms*>(userData)->transform, output);
}

Color ScenePixel(const Varyings& varyings, const void* userData) noexcept
{
    const auto& uniforms = *static_cast<const SceneUniforms*>(userData);
    Color color = PbrPixelShader(varyings, &uniforms.lighting);

    const Vec3 normal = cyber::math::Normalize(varyings.normal);
    const Vec3 position = varyings.worldPosition;
    const Vec3 toLight = uniforms.lamp.position - position;
    const f32 distance = cyber::math::Length(toLight);
    if (distance <= 1.0e-4f)
    {
        return color;
    }
    const Vec3 lightDirection = toLight * (1.0f / distance);
    const f32 ndotl = cyber::math::Dot(normal, lightDirection);
    if (ndotl <= 0.0f)
    {
        return color;
    }
    // The software-reference ray-traced shadow: one any-hit ray toward the
    // lamp, from a point offset off the surface.
    if (!TraceVisibility(*uniforms.geometry, OffsetRayOrigin(position, normal),
                         uniforms.lamp.position))
    {
        return color;
    }
    const Vec3 view = cyber::math::Normalize(uniforms.lighting.eyePosition - position);
    const Color brdf = EvaluateGgxBrdf(normal, view, lightDirection, uniforms.lighting.material);
    const f32 scale =
        ndotl * SmoothRangeFade(distance, uniforms.lamp.range) / (distance * distance);
    color.r += brdf.r * uniforms.lamp.intensity.x * scale;
    color.g += brdf.g * uniforms.lamp.intensity.y * scale;
    color.b += brdf.b * uniforms.lamp.intensity.z * scale;
    return color;
}

bool Render(const Group* groups, const SceneUniforms& base, AntiAliasTarget& target,
            Framebuffer& resolved, HeapAllocator& allocator)
{
    const Mat4 view = Mat4::LookAt(kEye, kTarget, Vec3{0.0f, 1.0f, 0.0f});
    const Mat4 projection = Mat4::PerspectiveReverseZ(
        0.62f, static_cast<f32>(kWidth) / static_cast<f32>(kHeight), 0.1f, 40.0f);
    target.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
    for (u32 group = 0; group < kGroupCount; ++group)
    {
        SceneUniforms uniforms = base;
        uniforms.transform.modelViewProjection = projection * view;
        uniforms.transform.eyePosition = kEye;
        uniforms.lighting.eyePosition = kEye;
        uniforms.lighting.cameraForward = cyber::math::Normalize(kTarget - kEye);
        const Vec3 albedo = groups[group].material.albedo;
        uniforms.lighting.material.baseColor = Color{albedo.x, albedo.y, albedo.z, 1.0f};
        uniforms.lighting.material.metallic = 0.0f;
        uniforms.lighting.material.roughness = 0.85f;
        ShaderDesc shader{};
        shader.pipeline.view = view;
        shader.pipeline.projection = projection;
        shader.pipeline.viewport = Viewport{0.0f, 0.0f, static_cast<f32>(target.Color().Width()),
                                            static_cast<f32>(target.Color().Height())};
        shader.pipeline.cullMode = CullMode::None;
        shader.vertexShader = &SceneVertex;
        shader.pixelShader = &ScenePixel;
        shader.userData = &uniforms;
        const DynamicArray<VertexInput>& vertices = groups[group].vertices;
        if (DrawShadedTriangles(shader, Span<const VertexInput>{vertices.Data(), vertices.Size()},
                                target.Color(), target.Depth(), nullptr, allocator)
                .IsError())
        {
            return false;
        }
    }
    ResolveDesc resolve{};
    if (target.Resolve(resolved, nullptr, allocator, resolve).IsError())
    {
        return false;
    }
    for (i32 y = 0; y < resolved.Height(); ++y)
    {
        for (i32 x = 0; x < resolved.Width(); ++x)
        {
            resolved.SetPixel(x, y, ToneMapAces(resolved.GetPixel(x, y), -0.6f));
        }
    }
    return true;
}

// Counter-based uniform in [0, 1), so the noisy frames are the same on every
// run.
f32 HashUniform(u32 x, u32 y, u32 frame) noexcept
{
    u32 value = x * 0x8DA6B343u ^ y * 0xD8163841u ^ frame * 0xCB1AB31Fu;
    value ^= value >> 16u;
    value *= 0x7FEB352Du;
    value ^= value >> 15u;
    value *= 0x846CA68Bu;
    value ^= value >> 16u;
    return static_cast<f32>(value >> 8u) * (1.0f / 16777216.0f);
}

// Ray-traced soft shadows from a disk light in the same room, seen from
// straight above, at one shadow ray per pixel, then denoised. Writes four
// panels: the raw signal, one denoised frame, eight accumulated frames, and a
// 256-sample reference.
struct ShadowGBuffer
{
    explicit ShadowGBuffer(HeapAllocator& allocator) noexcept
        : positions(allocator), normals(allocator)
    {
    }

    DynamicArray<Vec3> positions;
    DynamicArray<Vec3> normals;
};

bool CopyPanel(const Framebuffer& source, Framebuffer& composite, i32 panel)
{
    for (i32 y = 0; y < source.Height(); ++y)
    {
        for (i32 x = 0; x < source.Width(); ++x)
        {
            composite.SetPixel(x + panel * source.Width(), y, source.GetPixel(x, y));
        }
    }
    return true;
}

bool RenderDenoiseComparison(const TopLevelAccelerationStructure& tlas, const Vec3& lightCenter,
                             const char* path, HeapAllocator& allocator,
                             cyber::jobs::JobSystem* jobs)
{
    constexpr i32 kPanel = 256;
    constexpr usize kPixels = static_cast<usize>(kPanel) * kPanel;
    constexpr f32 kLightRadius = 0.45f;
    constexpr f32 kNear = 0.1f;
    constexpr f32 kFar = 10.0f;
    constexpr f32 kCameraHeight = 2.95f;

    FramebufferDesc desc{};
    desc.width = kPanel;
    desc.height = kPanel;
    desc.format = PixelFormat::RGBA32F;
    auto noisyResult = Framebuffer::Create(desc, allocator);
    auto denoisedResult = Framebuffer::Create(desc, allocator);
    auto referenceResult = Framebuffer::Create(desc, allocator);
    auto normalImageResult = Framebuffer::Create(desc, allocator);
    desc.width = kPanel * 4;
    auto compositeResult = Framebuffer::Create(desc, allocator);
    DepthBufferDesc depthDesc{};
    depthDesc.width = kPanel;
    depthDesc.height = kPanel;
    auto depthResult = DepthBuffer::Create(depthDesc, allocator);
    auto denoiserResult = SpatioTemporalDenoiser::Create(kPanel, kPanel, allocator);
    ShadowGBuffer gbuffer(allocator);
    if (noisyResult.IsError() || denoisedResult.IsError() || referenceResult.IsError() ||
        normalImageResult.IsError() || compositeResult.IsError() || depthResult.IsError() ||
        denoiserResult.IsError() || !gbuffer.positions.TryResize(kPixels) ||
        !gbuffer.normals.TryResize(kPixels))
    {
        return false;
    }
    Framebuffer noisy = std::move(noisyResult).Value();
    Framebuffer denoised = std::move(denoisedResult).Value();
    Framebuffer reference = std::move(referenceResult).Value();
    Framebuffer normalImage = std::move(normalImageResult).Value();
    Framebuffer composite = std::move(compositeResult).Value();
    DepthBuffer depth = std::move(depthResult).Value();
    SpatioTemporalDenoiser denoiser = std::move(denoiserResult).Value();

    // G-buffer from primary rays: an orthographic camera just under the
    // ceiling, looking straight down at the 4 x 4 floor.
    for (i32 y = 0; y < kPanel; ++y)
    {
        f32* depthRow = depth.Row(y);
        for (i32 x = 0; x < kPanel; ++x)
        {
            RayDesc ray{};
            ray.origin = Vec3{(static_cast<f32>(x) + 0.5f) / kPanel * 4.0f, kCameraHeight,
                              (static_cast<f32>(y) + 0.5f) / kPanel * 4.0f};
            ray.direction = Vec3{0.0f, -1.0f, 0.0f};
            const RayHit hit = TraceRay(tlas, ray);
            const usize index = static_cast<usize>(y) * kPanel + static_cast<usize>(x);
            const f32 distance = hit.hit ? hit.distance : kCameraHeight;
            gbuffer.positions[index] = ray.origin + ray.direction * distance;
            gbuffer.normals[index] = hit.hit ? hit.shadingNormal : Vec3{0.0f, 1.0f, 0.0f};
            depthRow[x] = (kNear * kFar / distance - kNear) / (kFar - kNear);
            const Vec3 n = gbuffer.normals[index];
            normalImage.SetPixel(
                x, y, Color{n.x * 0.5f + 0.5f, n.y * 0.5f + 0.5f, n.z * 0.5f + 0.5f, 1.0f});
        }
    }

    // One shadow ray toward a point on the disk light chosen by two uniforms.
    const auto visibility = [&](usize index, f32 u1, f32 u2)
    {
        const f32 radius = kLightRadius * std::sqrt(u1);
        const f32 angle = 2.0f * cyber::math::kPi * u2;
        const Vec3 sample =
            lightCenter + Vec3{radius * std::cos(angle), 0.0f, radius * std::sin(angle)};
        const Vec3 n = gbuffer.normals[index];
        if (cyber::math::Dot(n, sample - gbuffer.positions[index]) <= 0.0f)
        {
            return 0.0f;
        }
        return TraceVisibility(tlas, OffsetRayOrigin(gbuffer.positions[index], n), sample) ? 1.0f
                                                                                           : 0.0f;
    };

    for (i32 y = 0; y < kPanel; ++y)
    {
        for (i32 x = 0; x < kPanel; ++x)
        {
            const usize index = static_cast<usize>(y) * kPanel + static_cast<usize>(x);
            f32 sum = 0.0f;
            for (u32 s = 0; s < 256; ++s)
            {
                sum += visibility(index, (static_cast<f32>(s % 16) + 0.5f) / 16.0f,
                                  (static_cast<f32>(s / 16) + 0.5f) / 16.0f);
            }
            const f32 value = sum / 256.0f;
            reference.SetPixel(x, y, Color{value, value, value, 1.0f});
        }
    }

    DenoiseGuides guides{};
    guides.depth = &depth;
    guides.normals = &normalImage;
    guides.nearPlane = kNear;
    guides.farPlane = kFar;
    NoiseMetrics raw{};
    NoiseMetrics single{};
    for (u32 frame = 0; frame < 8; ++frame)
    {
        for (i32 y = 0; y < kPanel; ++y)
        {
            for (i32 x = 0; x < kPanel; ++x)
            {
                const usize index = static_cast<usize>(y) * kPanel + static_cast<usize>(x);
                const u32 ux = static_cast<u32>(x);
                const u32 uy = static_cast<u32>(y);
                const f32 value = visibility(index, HashUniform(ux, uy, frame * 2u),
                                             HashUniform(ux, uy, frame * 2u + 1u));
                noisy.SetPixel(x, y, Color{value, value, value, 1.0f});
            }
        }
        if (denoiser.Denoise(noisy, guides, DenoiserDesc{}, denoised, nullptr, jobs).IsError())
        {
            return false;
        }
        if (frame == 0)
        {
            CopyPanel(noisy, composite, 0);
            CopyPanel(denoised, composite, 1);
            (void)MeasureNoise(noisy, reference, raw);
            (void)MeasureNoise(denoised, reference, single);
        }
    }
    NoiseMetrics eight{};
    (void)MeasureNoise(denoised, reference, eight);
    CopyPanel(denoised, composite, 2);
    CopyPanel(reference, composite, 3);
    std::printf("Ray-traced soft shadow, 1 spp, relative RMSE against 256 spp: raw %.3f, "
                "denoised 1 frame %.3f, 8 frames %.3f.\n",
                raw.relativeRmse, single.relativeRmse, eight.relativeRmse);

    ImageWriteDesc writeDesc{};
    writeDesc.format = std::strstr(path, ".ppm") != nullptr ? ImageFormat::Ppm : ImageFormat::Tga;
    writeDesc.encodeSrgb = true;
    return WriteImage(path, composite, allocator, writeDesc).IsOk();
}

void PrintRayTracingPolicy(HeapAllocator& allocator)
{
    using namespace cyber::renderer;
    const cyber::rhi::Backend* backend = cyber::rhi::D3D12Backend();
    if (backend == nullptr)
    {
        std::printf(
            "Ray tracing: this build has no D3D12 backend; shadows use the rasterized path.\n");
        return;
    }
    const u32 adapterCount = backend->enumerateAdapters(nullptr, 0);
    for (u32 adapter = 0; adapter < adapterCount; ++adapter)
    {
        cyber::rhi::DeviceDesc desc{};
        desc.backend = backend;
        desc.adapterIndex = adapter;
        auto created = cyber::rhi::Device::Create(desc, allocator);
        if (created.IsError())
        {
            continue;
        }
        const cyber::rhi::Device device = std::move(created).Value();
        const RayTracingPlan plan = SelectRayTracingPlan(device, RayTracingPolicyDesc{});
        const RayTracingSelection shadows = plan.For(RayTracedEffect::Shadows);
        std::printf("Ray tracing on adapter %u \"%s\": capable %s, inline %s -> shadows %s (%s)\n",
                    adapter, device.Capabilities().adapter.name,
                    plan.hardwareCapable ? "yes" : "no", plan.inlineCapable ? "yes" : "no",
                    RayTracingTechniqueName(shadows.technique),
                    RayTracingFallbackReasonName(shadows.reason));
    }
}

}  // namespace

int main(int argc, char** argv)
{
    const char* outputPath = argc > 1 ? argv[1] : "global_illumination.tga";
    if (argc > 2)
    {
        std::printf("usage: cyber_sample_global_illumination [output.tga]\n");
        return 2;
    }

    HeapAllocator allocator;
    Group groups[kGroupCount] = {Group(allocator), Group(allocator), Group(allocator),
                                 Group(allocator), Group(allocator)};
    if (!BuildRoom(groups))
    {
        std::printf("failed to build the room\n");
        return 1;
    }

    // Acceleration structures: one bottom level per material group.
    BottomLevelAccelerationStructure blases[kGroupCount];
    RayInstanceDesc instances[kGroupCount]{};
    GiSurfaceMaterial materials[kGroupCount]{};
    usize triangles = 0;
    for (u32 group = 0; group < kGroupCount; ++group)
    {
        AccelerationStructureBuildDesc desc{};
        desc.vertices =
            Span<const VertexInput>{groups[group].vertices.Data(), groups[group].vertices.Size()};
        auto built = BottomLevelAccelerationStructure::Build(desc, allocator);
        if (built.IsError())
        {
            std::printf("acceleration structure build failed: %s\n", built.Error().message);
            return 1;
        }
        blases[group] = std::move(built).Value();
        instances[group].blas = &blases[group];
        instances[group].instanceId = group;
        materials[group] = groups[group].material;
        triangles += groups[group].vertices.Size() / 3;
    }
    auto tlasResult = TopLevelAccelerationStructure::Build(
        Span<const RayInstanceDesc>{instances, kGroupCount}, allocator);
    if (tlasResult.IsError())
    {
        std::printf("top-level build failed: %s\n", tlasResult.Error().message);
        return 1;
    }
    const TopLevelAccelerationStructure tlas = std::move(tlasResult).Value();

    GiPointLight lamp{};
    lamp.position = Vec3{2.0f, 2.75f, 2.2f};
    lamp.intensity = Vec3{15.0f, 14.0f, 12.5f};
    lamp.range = 20.0f;
    GiScene scene{};
    scene.geometry = &tlas;
    scene.materials = Span<const GiSurfaceMaterial>{materials, kGroupCount};
    scene.pointLights = Span<const GiPointLight>{&lamp, 1};

    // Bake.
    IrradianceVolumeDesc volumeDesc{};
    volumeDesc.boundsMin = Vec3{0.2f, 0.2f, 0.2f};
    volumeDesc.boundsMax = Vec3{3.8f, 2.8f, 3.8f};
    volumeDesc.probeCountX = 9;
    volumeDesc.probeCountY = 7;
    volumeDesc.probeCountZ = 9;
    auto volumeResult = IrradianceVolume::Create(volumeDesc, allocator);
    if (volumeResult.IsError())
    {
        std::printf("irradiance volume allocation failed: %s\n", volumeResult.Error().message);
        return 1;
    }
    IrradianceVolume volume = std::move(volumeResult).Value();
    cyber::jobs::JobSystem::Desc jobDesc{};
    auto jobsResult = cyber::jobs::JobSystem::Create(jobDesc, allocator);
    cyber::jobs::JobSystem* jobs = jobsResult.IsOk() ? &jobsResult.Value() : nullptr;
    IrradianceBakeDesc bake{};
    bake.raysPerProbe = 2048;
    bake.path.maxBounces = 4;
    IrradianceBakeStats bakeStats{};
    const auto bakeStart = std::chrono::steady_clock::now();
    if (BakeIrradianceVolume(scene, bake, volume, &bakeStats, jobs).IsError())
    {
        std::printf("irradiance bake failed\n");
        return 1;
    }
    const f64 bakeMs =
        std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - bakeStart)
            .count();

    // The fairest constant: the mean indirect irradiance the volume reports
    // over the floor. The comparison is then about distribution, not level.
    Vec3 meanIndirect{};
    u32 floorSamples = 0;
    for (u32 ix = 0; ix < 8; ++ix)
    {
        for (u32 iz = 0; iz < 8; ++iz)
        {
            meanIndirect += SampleIrradianceVolume(volume,
                                                   Vec3{0.25f + 0.5f * static_cast<f32>(ix), 0.0f,
                                                        0.25f + 0.5f * static_cast<f32>(iz)},
                                                   Vec3{0.0f, 1.0f, 0.0f});
            ++floorSamples;
        }
    }
    meanIndirect = meanIndirect * (1.0f / static_cast<f32>(floorSamples));

    AntiAliasTargetDesc targetDesc{};
    targetDesc.width = kWidth;
    targetDesc.height = kHeight;
    targetDesc.quality = AntiAliasQuality::Balanced;
    targetDesc.format = PixelFormat::RGBA32F;
    auto targetResult = AntiAliasTarget::Create(targetDesc, allocator);
    FramebufferDesc imageDesc{};
    imageDesc.width = kWidth;
    imageDesc.height = kHeight;
    imageDesc.format = PixelFormat::RGBA32F;
    auto flatResult = Framebuffer::Create(imageDesc, allocator);
    auto bakedResult = Framebuffer::Create(imageDesc, allocator);
    imageDesc.width = kWidth * 2;
    auto compositeResult = Framebuffer::Create(imageDesc, allocator);
    if (targetResult.IsError() || flatResult.IsError() || bakedResult.IsError() ||
        compositeResult.IsError())
    {
        std::printf("failed to allocate render targets\n");
        return 1;
    }
    AntiAliasTarget target = std::move(targetResult).Value();
    Framebuffer flat = std::move(flatResult).Value();
    Framebuffer baked = std::move(bakedResult).Value();
    Framebuffer composite = std::move(compositeResult).Value();

    SceneUniforms uniforms{};
    uniforms.geometry = &tlas;
    uniforms.lamp = lamp;
    uniforms.lighting.ambientIrradiance = meanIndirect;
    const auto renderStart = std::chrono::steady_clock::now();
    if (!Render(groups, uniforms, target, flat, allocator))
    {
        std::printf("flat-ambient render failed\n");
        return 1;
    }
    IrradianceVolumeView view{};
    view.volume = &volume;
    uniforms.lighting.irradianceVolume = &view;
    if (!Render(groups, uniforms, target, baked, allocator))
    {
        std::printf("baked-GI render failed\n");
        return 1;
    }
    const f64 renderMs =
        std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - renderStart)
            .count();
    for (i32 y = 0; y < kHeight; ++y)
    {
        for (i32 x = 0; x < kWidth; ++x)
        {
            composite.SetPixel(x, y, flat.GetPixel(x, y));
            composite.SetPixel(x + kWidth, y, baked.GetPixel(x, y));
        }
    }
    ImageWriteDesc writeDesc{};
    writeDesc.format =
        std::strstr(outputPath, ".ppm") != nullptr ? ImageFormat::Ppm : ImageFormat::Tga;
    writeDesc.encodeSrgb = true;
    if (!WriteImage(outputPath, composite, allocator, writeDesc))
    {
        std::printf("failed to write %s\n", outputPath);
        return 1;
    }

    // The denoiser comparison goes next to the main image, with "_denoise"
    // before the extension.
    char denoisePath[512] = {};
    const char* extension = std::strrchr(outputPath, '.');
    const usize stem =
        extension != nullptr ? static_cast<usize>(extension - outputPath) : std::strlen(outputPath);
    if (stem + 16 >= sizeof(denoisePath))
    {
        std::printf("output path is too long\n");
        return 1;
    }
    std::memcpy(denoisePath, outputPath, stem);
    std::snprintf(denoisePath + stem, sizeof(denoisePath) - stem, "_denoise%s",
                  extension != nullptr ? extension : ".tga");
    if (!RenderDenoiseComparison(tlas, Vec3{2.0f, 2.75f, 2.2f}, denoisePath, allocator, jobs))
    {
        std::printf("denoiser comparison failed\n");
        return 1;
    }
    if (jobs != nullptr)
    {
        jobs->Shutdown();
    }

    std::printf("Room: %zu triangles in %u instances, BVH %zu bytes.\n", triangles, kGroupCount,
                tlas.DataSizeBytes());
    std::printf("Bake: %u probes (%u valid, %u inside geometry), %u rays per probe, %u bounces, "
                "%llu path rays, %.0f ms, %zu bytes.\n",
                bakeStats.probeCount, bakeStats.validProbes, bakeStats.insideProbes,
                bake.raysPerProbe, bake.path.maxBounces,
                static_cast<unsigned long long>(bakeStats.path.rays), bakeMs,
                volume.DataSizeBytes());
    std::printf("Render: two %dx%d frames, 2x2 supersampled, ray-traced lamp shadows, %.0f ms.\n",
                kWidth, kHeight, renderMs);
    PrintRayTracingPolicy(allocator);
    std::printf("Wrote %s (left: flat ambient; right: baked irradiance volume)\n", outputPath);
    std::printf("Wrote %s (raw 1 spp, denoised 1 frame, denoised 8 frames, 256 spp reference)\n",
                denoisePath);
    return 0;
}
