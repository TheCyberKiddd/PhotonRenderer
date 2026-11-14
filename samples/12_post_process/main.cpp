// Phase 30 sample: the post-processing chain and the anti-aliasing paths,
// priced effect by effect on the machine that runs it.
//
// The roadmap's bar for this phase is that every effect toggles at runtime
// with its time shown, and that temporal anti-aliasing shows no ghosting in a
// moving scene. This sample is where that is visible rather than asserted: it
// renders a scene with something for every pass to act on, runs the chain,
// prints what each pass cost and what switching it off changed, then resolves
// the same moving scene with no filter, with FXAA, and with temporal
// accumulation, and reports the measured difference between them.
//
// The renderer shades on the CPU, so the times printed here are CPU
// milliseconds for the whole effect, not a GPU queue timestamp.

#include <cyber/core/result.h>
#include <cyber/math/mat4.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/image_io.h>
#include <cyber/software/pbr.h>
#include <cyber/software/post_process.h>
#include <cyber/software/post_process_quality.h>
#include <cyber/software/shader.h>
#include <cyber/software/temporal_aa.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>

using namespace cyber::memory;
using namespace cyber::software;
using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::core::usize;
using cyber::math::Color;
using cyber::math::Mat4;
using cyber::math::Vec2;
using cyber::math::Vec3;
using cyber::math::Vec4;

namespace
{

constexpr i32 kWidth = 640;
constexpr i32 kHeight = 400;
constexpr f32 kNearPlane = 0.1f;
constexpr f32 kFarPlane = 80.0f;
constexpr f32 kFieldOfView = 0.9f;
// The camera pans across the scene, which is what gives motion blur and the
// temporal resolve something to follow.
constexpr f32 kPanPerFrame = 0.035f;
constexpr i32 kTemporalFrames = 24;

[[nodiscard]] Vec3 CameraEye(i32 frame) noexcept
{
    const f32 pan = static_cast<f32>(frame) * kPanPerFrame;
    return Vec3{-1.4f + pan, 2.2f, 7.0f};
}

[[nodiscard]] Vec3 CameraTarget(i32 frame) noexcept
{
    const f32 pan = static_cast<f32>(frame) * kPanPerFrame;
    return Vec3{0.1f + pan, 1.0f, 0.0f};
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

struct SceneUniforms
{
    TransformShaderData transform{};
    PbrLightingData lighting{};
};

// What the normal and velocity passes need beyond the transform.
struct AuxiliaryUniforms
{
    TransformShaderData transform{};
    // Clip-space transform of the same geometry one frame earlier, which is
    // what a screen-space velocity is the difference of.
    Mat4 previousViewProjection = Mat4::Identity();
    f32 width = 1.0f;
    f32 height = 1.0f;
};

void SceneVertex(const VertexInput& input, const void* userData, VertexStageOutput& output) noexcept
{
    TransformVertexShader(input, &static_cast<const SceneUniforms*>(userData)->transform, output);
}

Color ScenePixel(const Varyings& varyings, const void* userData) noexcept
{
    return PbrPixelShader(varyings, &static_cast<const SceneUniforms*>(userData)->lighting);
}

void AuxiliaryVertex(const VertexInput& input, const void* userData,
                     VertexStageOutput& output) noexcept
{
    TransformVertexShader(input, &static_cast<const AuxiliaryUniforms*>(userData)->transform,
                          output);
}

// World-space normal, encoded into the unit range the occlusion pass expects.
Color NormalPixel(const Varyings& varyings, const void* /*userData*/) noexcept
{
    const Vec3 normal = cyber::math::Normalize(varyings.normal);
    return Color{normal.x * 0.5f + 0.5f, normal.y * 0.5f + 0.5f, normal.z * 0.5f + 0.5f, 1.0f};
}

// Screen-space pixels per frame, from where this surface point is now and
// where it was. Writing it from the interpolated world position means the
// vector is exact for a moving camera over static geometry, which is the case
// this sample sets up.
Color VelocityPixel(const Varyings& varyings, const void* userData) noexcept
{
    const auto& uniforms = *static_cast<const AuxiliaryUniforms*>(userData);
    const Vec4 world{varyings.worldPosition.x, varyings.worldPosition.y, varyings.worldPosition.z,
                     1.0f};
    const Vec4 previousClip = uniforms.previousViewProjection * world;
    if (!(previousClip.w > 1.0e-6f))
    {
        return Color{0.0f, 0.0f, 0.0f, 1.0f};
    }
    const f32 previousX = (previousClip.x / previousClip.w * 0.5f + 0.5f) * uniforms.width;
    const f32 previousY = (0.5f - previousClip.y / previousClip.w * 0.5f) * uniforms.height;
    return Color{varyings.pixelPosition.x - previousX, varyings.pixelPosition.y - previousY, 0.0f,
                 1.0f};
}

[[nodiscard]] bool PushTriangle(DynamicArray<VertexInput>& mesh, const Vec3& a, const Vec3& b,
                                const Vec3& c, const Color& color)
{
    const Vec3 normal = cyber::math::Normalize(cyber::math::Cross(b - a, c - a));
    for (const Vec3& position : {a, b, c})
    {
        VertexInput vertex{};
        vertex.position = position;
        vertex.normal = normal;
        vertex.color = color;
        if (!mesh.TryPushBack(vertex))
        {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool PushQuad(DynamicArray<VertexInput>& mesh, const Vec3& a, const Vec3& b,
                            const Vec3& c, const Vec3& d, const Color& color)
{
    return PushTriangle(mesh, a, b, c, color) && PushTriangle(mesh, a, c, d, color);
}

[[nodiscard]] bool AddBox(DynamicArray<VertexInput>& mesh, const Vec3& center,
                          const Vec3& halfExtents, const Color& color)
{
    const f32 x0 = center.x - halfExtents.x;
    const f32 x1 = center.x + halfExtents.x;
    const f32 y0 = center.y - halfExtents.y;
    const f32 y1 = center.y + halfExtents.y;
    const f32 z0 = center.z - halfExtents.z;
    const f32 z1 = center.z + halfExtents.z;
    return PushQuad(mesh, Vec3{x0, y0, z1}, Vec3{x1, y0, z1}, Vec3{x1, y1, z1},
                    Vec3{x0, y1, z1}, color) &&
           PushQuad(mesh, Vec3{x1, y0, z0}, Vec3{x0, y0, z0}, Vec3{x0, y1, z0}, Vec3{x1, y1, z0},
                    color) &&
           PushQuad(mesh, Vec3{x0, y0, z0}, Vec3{x0, y0, z1}, Vec3{x0, y1, z1}, Vec3{x0, y1, z0},
                    color) &&
           PushQuad(mesh, Vec3{x1, y0, z1}, Vec3{x1, y0, z0}, Vec3{x1, y1, z0}, Vec3{x1, y1, z1},
                    color) &&
           PushQuad(mesh, Vec3{x0, y1, z1}, Vec3{x1, y1, z1}, Vec3{x1, y1, z0}, Vec3{x0, y1, z0},
                    color) &&
           PushQuad(mesh, Vec3{x0, y0, z0}, Vec3{x1, y0, z0}, Vec3{x1, y0, z1}, Vec3{x0, y0, z1},
                    color);
}

// A scene with something for every pass to act on: a floor and a back wall
// meeting in a crease for ambient occlusion, a row of posts receding into the
// distance for depth of field, fine vertical slats for the anti-aliasing to
// resolve, and one small surface driven far above the tone mapper's shoulder
// for bloom.
[[nodiscard]] bool BuildScene(DynamicArray<VertexInput>& mesh)
{
    const Color floorColor{0.42f, 0.44f, 0.47f, 1.0f};
    const Color wallColor{0.30f, 0.32f, 0.38f, 1.0f};
    if (!PushQuad(mesh, Vec3{-14.0f, 0.0f, 10.0f}, Vec3{14.0f, 0.0f, 10.0f},
                  Vec3{14.0f, 0.0f, -14.0f}, Vec3{-14.0f, 0.0f, -14.0f}, floorColor))
    {
        return false;
    }
    if (!PushQuad(mesh, Vec3{-14.0f, 0.0f, -8.0f}, Vec3{14.0f, 0.0f, -8.0f},
                  Vec3{14.0f, 7.0f, -8.0f}, Vec3{-14.0f, 7.0f, -8.0f}, wallColor))
    {
        return false;
    }

    // Posts at increasing distance. The near ones sit inside the focal plane
    // and the far ones well outside it.
    for (i32 index = 0; index < 6; ++index)
    {
        const f32 z = 1.5f - static_cast<f32>(index) * 1.9f;
        const f32 shade = 0.52f - static_cast<f32>(index) * 0.05f;
        if (!AddBox(mesh, Vec3{-2.6f + static_cast<f32>(index) * 1.05f, 0.65f, z},
                    Vec3{0.28f, 0.65f, 0.28f}, Color{shade, shade * 0.92f, shade * 0.80f, 1.0f}))
        {
            return false;
        }
    }

    // A slatted screen: thin, regularly spaced uprights, which is the feature
    // a one-sample-per-pixel render cannot hold onto and the anti-aliasing
    // paths are measured against.
    for (i32 index = 0; index < 22; ++index)
    {
        const f32 x = -5.4f + static_cast<f32>(index) * 0.5f;
        if (!AddBox(mesh, Vec3{x, 1.6f, -6.2f}, Vec3{0.035f, 1.6f, 0.06f},
                    Color{0.78f, 0.76f, 0.70f, 1.0f}))
        {
            return false;
        }
    }

    // The highlight. Its base color is far above 1, so the tone mapper has a
    // shoulder to roll off and bloom has energy to scatter.
    if (!AddBox(mesh, Vec3{2.45f, 1.15f, 0.4f}, Vec3{0.22f, 0.22f, 0.22f},
                Color{36.0f, 31.0f, 18.0f, 1.0f}))
    {
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

[[nodiscard]] Mat4 ViewProjection(i32 frame, const Vec2& jitterPixels) noexcept
{
    const Mat4 view = Mat4::LookAt(CameraEye(frame), CameraTarget(frame), Vec3{0.0f, 1.0f, 0.0f});
    const Mat4 projection = Mat4::PerspectiveReverseZ(
        kFieldOfView, static_cast<f32>(kWidth) / static_cast<f32>(kHeight), kNearPlane, kFarPlane);
    return ApplyJitterToProjection(projection, jitterPixels, kWidth, kHeight) * view;
}

[[nodiscard]] bool RenderColor(Span<const VertexInput> vertices, i32 frame,
                               const Vec2& jitterPixels, Framebuffer& color, DepthBuffer& depth,
                               HeapAllocator& allocator)
{
    const Vec3 eye = CameraEye(frame);
    const Vec3 target = CameraTarget(frame);

    SceneUniforms uniforms{};
    uniforms.transform.modelViewProjection = ViewProjection(frame, jitterPixels);
    uniforms.transform.modelToWorld = Mat4::Identity();
    uniforms.transform.normalToWorld = Mat4::Identity();
    uniforms.transform.eyePosition = eye;
    uniforms.lighting.eyePosition = eye;
    uniforms.lighting.cameraForward = cyber::math::Normalize(target - eye);
    uniforms.lighting.ambientIrradiance = Vec3{0.16f, 0.18f, 0.22f};
    uniforms.lighting.material.roughness = 0.55f;
    uniforms.lighting.material.metallic = 0.0f;
    uniforms.lighting.directional[0].direction =
        cyber::math::Normalize(Vec3{-0.45f, -0.80f, -0.40f});
    uniforms.lighting.directional[0].radiance = Vec3{3.2f, 3.0f, 2.7f};
    uniforms.lighting.directionalCount = 1;

    ShaderDesc shader{};
    shader.pipeline.viewport =
        Viewport{0.0f, 0.0f, static_cast<f32>(kWidth), static_cast<f32>(kHeight)};
    shader.pipeline.cullMode = CullMode::None;
    shader.pipeline.useJobs = false;
    shader.vertexShader = &SceneVertex;
    shader.pixelShader = &ScenePixel;
    shader.userData = &uniforms;

    color.Clear(Color{0.035f, 0.042f, 0.055f, 1.0f});
    depth.Clear();
    return DrawShadedTriangles(shader, vertices, color, depth, nullptr, allocator).IsOk();
}

[[nodiscard]] bool RenderAuxiliary(Span<const VertexInput> vertices, i32 frame,
                                   const Vec2& jitterPixels, PixelShaderFn pixelShader,
                                   const Color& clearColor, Framebuffer& target,
                                   DepthBuffer& depth, HeapAllocator& allocator)
{
    AuxiliaryUniforms uniforms{};
    uniforms.transform.modelViewProjection = ViewProjection(frame, jitterPixels);
    uniforms.transform.modelToWorld = Mat4::Identity();
    uniforms.transform.normalToWorld = Mat4::Identity();
    uniforms.transform.eyePosition = CameraEye(frame);
    uniforms.previousViewProjection = ViewProjection(frame - 1, jitterPixels);
    uniforms.width = static_cast<f32>(kWidth);
    uniforms.height = static_cast<f32>(kHeight);

    ShaderDesc shader{};
    shader.pipeline.viewport =
        Viewport{0.0f, 0.0f, static_cast<f32>(kWidth), static_cast<f32>(kHeight)};
    shader.pipeline.cullMode = CullMode::None;
    shader.pipeline.useJobs = false;
    shader.vertexShader = &AuxiliaryVertex;
    shader.pixelShader = pixelShader;
    shader.userData = &uniforms;

    target.Clear(clearColor);
    depth.Clear();
    return DrawShadedTriangles(shader, vertices, target, depth, nullptr, allocator).IsOk();
}

[[nodiscard]] bool MakeImage(HeapAllocator& heap, Framebuffer& out)
{
    FramebufferDesc desc{};
    desc.width = kWidth;
    desc.height = kHeight;
    desc.format = PixelFormat::RGBA32F;
    auto created = Framebuffer::Create(desc, heap);
    if (!created)
    {
        return false;
    }
    out = std::move(created).Value();
    return true;
}

void CopyInto(const Framebuffer& source, Framebuffer& destination) noexcept
{
    for (i32 y = 0; y < source.Height(); ++y)
    {
        for (i32 x = 0; x < source.Width(); ++x)
        {
            destination.SetPixel(x, y, source.GetPixel(x, y));
        }
    }
}

[[nodiscard]] bool Write(const char* directory, const char* name, const Framebuffer& image,
                         HeapAllocator& allocator)
{
    char path[512];
    std::snprintf(path, sizeof(path), "%s%s", directory, name);
    const auto status = WriteImage(path, image, allocator);
    if (!status.IsOk())
    {
        std::fprintf(stderr, "failed to write %s\n", path);
        return false;
    }
    std::printf("  wrote %s\n", path);
    return true;
}

[[nodiscard]] PostProcessView MakeView() noexcept
{
    PostProcessView view{};
    view.projection = Mat4::PerspectiveReverseZ(
        kFieldOfView, static_cast<f32>(kWidth) / static_cast<f32>(kHeight), kNearPlane, kFarPlane);
    view.view = Mat4::LookAt(CameraEye(0), CameraTarget(0), Vec3{0.0f, 1.0f, 0.0f});
    view.nearPlane = kNearPlane;
    view.farPlane = kFarPlane;
    return view;
}

[[nodiscard]] PostProcessSettings MakeSettings(const ColorGradeLut* lut) noexcept
{
    PostProcessSettings settings{};
    settings.enableAmbientOcclusion = true;
    settings.enableDepthOfField = true;
    settings.enableMotionBlur = true;
    settings.enableBloom = true;
    settings.enableExposure = true;
    settings.enableToneMap = true;
    settings.enableColorGrade = true;
    settings.colorGradeLut = lut;
    settings.ambientOcclusion.radius = 0.9f;
    settings.ambientOcclusion.intensity = 0.9f;
    settings.depthOfField.focusDistance = 7.2f;
    settings.depthOfField.apertureFStop = 2.0f;
    settings.bloom.intensity = 0.09f;
    return settings;
}

}  // namespace

int main(int argc, char** argv)
{
    const char* directory = argc > 1 ? argv[1] : "";

    HeapAllocator heap;

    DynamicArray<VertexInput> mesh{heap};
    if (!BuildScene(mesh))
    {
        std::fprintf(stderr, "failed to build the scene\n");
        return 1;
    }
    const Span<const VertexInput> vertices{mesh.Data(), mesh.Size()};

    Framebuffer color;
    Framebuffer normals;
    Framebuffer velocity;
    Framebuffer destination;
    Framebuffer reference;
    if (!MakeImage(heap, color) || !MakeImage(heap, normals) || !MakeImage(heap, velocity) ||
        !MakeImage(heap, destination) || !MakeImage(heap, reference))
    {
        std::fprintf(stderr, "failed to allocate the frame buffers\n");
        return 1;
    }

    DepthBufferDesc depthDesc{};
    depthDesc.width = kWidth;
    depthDesc.height = kHeight;
    auto depthCreated = DepthBuffer::Create(depthDesc, heap);
    if (!depthCreated)
    {
        std::fprintf(stderr, "failed to allocate the depth buffer\n");
        return 1;
    }
    DepthBuffer depth = std::move(depthCreated).Value();

    auto stackCreated = PostProcessStack::Create(kWidth, kHeight, heap);
    if (!stackCreated)
    {
        std::fprintf(stderr, "failed to create the post-processing stack\n");
        return 1;
    }
    PostProcessStack stack = std::move(stackCreated).Value();

    ColorGradeSettings gradeSettings{};
    gradeSettings.contrast = 1.08f;
    gradeSettings.saturation = 1.06f;
    gradeSettings.temperature = 0.08f;
    auto lutCreated = ColorGradeLut::Bake(gradeSettings, kDefaultColorGradeLutSize, heap);
    if (!lutCreated)
    {
        std::fprintf(stderr, "failed to bake the color grading table\n");
        return 1;
    }
    const ColorGradeLut lut = std::move(lutCreated).Value();

    // ---------------------------------------------------------------------
    // The frame, and what each pass of the chain cost on it.
    // ---------------------------------------------------------------------

    constexpr i32 kShownFrame = 12;
    if (!RenderColor(vertices, kShownFrame, Vec2{0.0f, 0.0f}, color, depth, heap))
    {
        std::fprintf(stderr, "failed to render the scene\n");
        return 1;
    }
    // The occlusion pass reads depth from the color render, so the auxiliary
    // passes get their own depth buffer rather than overwriting it.
    auto auxiliaryDepthCreated = DepthBuffer::Create(depthDesc, heap);
    if (!auxiliaryDepthCreated)
    {
        std::fprintf(stderr, "failed to allocate the auxiliary depth buffer\n");
        return 1;
    }
    DepthBuffer auxiliaryDepth = std::move(auxiliaryDepthCreated).Value();
    if (!RenderAuxiliary(vertices, kShownFrame, Vec2{0.0f, 0.0f}, &NormalPixel,
                         Color{0.5f, 0.5f, 0.0f, 1.0f}, normals, auxiliaryDepth, heap) ||
        !RenderAuxiliary(vertices, kShownFrame, Vec2{0.0f, 0.0f}, &VelocityPixel,
                         Color{0.0f, 0.0f, 0.0f, 1.0f}, velocity, auxiliaryDepth, heap))
    {
        std::fprintf(stderr, "failed to render the auxiliary buffers\n");
        return 1;
    }

    PostProcessInputs inputs{};
    inputs.color = &color;
    inputs.depth = &depth;
    inputs.normals = &normals;
    inputs.velocity = &velocity;

    const PostProcessSettings all = MakeSettings(&lut);
    const PostProcessView view = MakeView();

    PostProcessTimings timings{};
    if (!stack.Execute(inputs, reference, all, view, 1.0f / 60.0f, nullptr, &timings).IsOk())
    {
        std::fprintf(stderr, "the post-processing chain failed\n");
        return 1;
    }

    std::printf("post-processing chain at %dx%d, CPU time per pass\n\n", kWidth, kHeight);
    std::printf("  %-20s %10s  %s\n", "pass", "ms", "state");
    for (const PassTiming& timing : timings.passes)
    {
        std::printf("  %-20s %10.3f  %s\n", timing.name != nullptr ? timing.name : "?",
                    static_cast<double>(timing.nanoseconds) * 1.0e-6,
                    timing.executed ? "on" : "off");
    }
    std::printf("  %-20s %10.3f\n", "total", static_cast<double>(timings.totalNanoseconds) * 1.0e-6);
    std::printf("  metered %.4f, exposure %+.3f EV\n\n",
                static_cast<double>(timings.averageLuminance),
                static_cast<double>(timings.exposureEv));

    // Each effect off on its own: what it cost, and how much of the image it
    // was responsible for. A toggle that changes no pixel is a checkbox.
    struct Toggle
    {
        const char* name;
        bool PostProcessSettings::*flag;
    };
    const Toggle toggles[] = {
        {"ambient_occlusion", &PostProcessSettings::enableAmbientOcclusion},
        {"depth_of_field", &PostProcessSettings::enableDepthOfField},
        {"motion_blur", &PostProcessSettings::enableMotionBlur},
        {"bloom", &PostProcessSettings::enableBloom},
        {"exposure", &PostProcessSettings::enableExposure},
        {"tone_map", &PostProcessSettings::enableToneMap},
        {"color_grade", &PostProcessSettings::enableColorGrade},
    };

    std::printf("each effect switched off on its own\n\n");
    std::printf("  %-20s %10s %10s\n", "disabled", "total ms", "changed");
    for (const Toggle& toggle : toggles)
    {
        PostProcessSettings settings = all;
        settings.*(toggle.flag) = false;

        PostProcessTimings off{};
        if (!stack.Execute(inputs, destination, settings, view, 1.0f / 60.0f, nullptr, &off).IsOk())
        {
            std::fprintf(stderr, "the post-processing chain failed\n");
            return 1;
        }

        ImageDifferenceMetrics difference{};
        if (!MeasureImageDifference(destination, reference, 1.0f / 255.0f, difference).IsOk())
        {
            std::fprintf(stderr, "failed to compare the images\n");
            return 1;
        }
        std::printf("  %-20s %10.3f %9.2f%%\n", toggle.name,
                    static_cast<double>(off.totalNanoseconds) * 1.0e-6,
                    static_cast<double>(difference.differingFraction) * 100.0);
    }
    std::printf("\n");

    if (!Write(directory, "post_process.ppm", reference, heap))
    {
        return 1;
    }

    // ---------------------------------------------------------------------
    // The anti-aliasing paths on the same moving scene.
    // ---------------------------------------------------------------------

    Framebuffer unfiltered;
    Framebuffer fxaa;
    Framebuffer temporal;
    Framebuffer scratch;
    if (!MakeImage(heap, unfiltered) || !MakeImage(heap, fxaa) || !MakeImage(heap, temporal) ||
        !MakeImage(heap, scratch))
    {
        std::fprintf(stderr, "failed to allocate the anti-aliasing buffers\n");
        return 1;
    }

    auto resolverCreated = TemporalAaResolver::Create(kWidth, kHeight, heap);
    if (!resolverCreated)
    {
        std::fprintf(stderr, "failed to create the temporal resolver\n");
        return 1;
    }
    TemporalAaResolver resolver = std::move(resolverCreated).Value();

    TemporalAaSettings temporalSettings{};
    // The depth buffer is what lets the resolve tell a disocclusion from
    // motion. Without it, a pixel of floor that a post was covering reprojects
    // onto the post and reuses its color, and the slats leave trails across
    // the floor as the camera pans.
    temporalSettings.nearPlane = kNearPlane;
    temporalSettings.farPlane = kFarPlane;
    TemporalAaStats temporalStats{};
    for (i32 frame = 0; frame < kTemporalFrames; ++frame)
    {
        const Vec2 jitter = TemporalJitterOffset(static_cast<u32>(frame));
        if (!RenderColor(vertices, frame, jitter, color, depth, heap) ||
            !RenderAuxiliary(vertices, frame, jitter, &VelocityPixel, Color{0.0f, 0.0f, 0.0f, 1.0f},
                             velocity, auxiliaryDepth, heap))
        {
            std::fprintf(stderr, "failed to render frame %d\n", frame);
            return 1;
        }
        TemporalAaInputs temporalInputs{};
        temporalInputs.color = &color;
        temporalInputs.velocity = &velocity;
        temporalInputs.depth = &depth;
        if (!resolver.Resolve(temporalInputs, scratch, temporalSettings, nullptr, &temporalStats)
                 .IsOk())
        {
            std::fprintf(stderr, "the temporal resolve failed\n");
            return 1;
        }
    }
    CopyInto(scratch, temporal);
    // The unfiltered comparison is the last frame as rendered, with no jitter,
    // and FXAA is that same frame filtered.
    if (!RenderColor(vertices, kTemporalFrames - 1, Vec2{0.0f, 0.0f}, unfiltered, depth, heap))
    {
        std::fprintf(stderr, "failed to render the comparison frame\n");
        return 1;
    }
    if (!FxaaResolve(unfiltered, fxaa, FxaaSettings{}, nullptr).IsOk())
    {
        std::fprintf(stderr, "the FXAA resolve failed\n");
        return 1;
    }

    EdgeAliasingMetrics noneEdges{};
    EdgeAliasingMetrics fxaaEdges{};
    EdgeAliasingMetrics temporalEdges{};
    if (!MeasureEdgeAliasing(unfiltered, 0.25f, noneEdges).IsOk() ||
        !MeasureEdgeAliasing(fxaa, 0.25f, fxaaEdges).IsOk() ||
        !MeasureEdgeAliasing(temporal, 0.25f, temporalEdges).IsOk())
    {
        std::fprintf(stderr, "failed to measure the edges\n");
        return 1;
    }

    std::printf("anti-aliasing on the moving scene, %d accumulated frames\n\n", kTemporalFrames);
    std::printf("  %-14s %14s %14s\n", "method", "hard steps", "mean gradient");
    std::printf("  %-14s %13.3f%% %14.5f\n", AntiAliasMethodName(AntiAliasMethod::None),
                static_cast<double>(noneEdges.hardStepFraction) * 100.0,
                static_cast<double>(noneEdges.meanGradient));
    std::printf("  %-14s %13.3f%% %14.5f\n", AntiAliasMethodName(AntiAliasMethod::Fxaa),
                static_cast<double>(fxaaEdges.hardStepFraction) * 100.0,
                static_cast<double>(fxaaEdges.meanGradient));
    std::printf("  %-14s %13.3f%% %14.5f\n", AntiAliasMethodName(AntiAliasMethod::Temporal),
                static_cast<double>(temporalEdges.hardStepFraction) * 100.0,
                static_cast<double>(temporalEdges.meanGradient));
    std::printf("\n  temporal history: %u reused, %u rejected (%u on depth), %u rectified, "
                "%u relaxed\n\n",
                temporalStats.historyPixels, temporalStats.rejectedPixels,
                temporalStats.depthRejectedPixels, temporalStats.clampedPixels,
                temporalStats.relaxedPixels);

    // The images are written tone mapped, because an unbounded HDR buffer
    // encoded straight to bytes shows nothing but clipping.
    PostProcessSettings display{};
    display.enableAmbientOcclusion = false;
    display.enableDepthOfField = false;
    display.enableMotionBlur = false;
    display.enableBloom = false;
    display.enableExposure = true;
    display.enableToneMap = true;
    display.enableColorGrade = false;

    struct Variant
    {
        const char* name;
        const Framebuffer* image;
    };
    const Variant variants[] = {{"aa_none.ppm", &unfiltered},
                                {"aa_fxaa.ppm", &fxaa},
                                {"aa_temporal.ppm", &temporal}};
    for (const Variant& variant : variants)
    {
        PostProcessInputs displayInputs{};
        displayInputs.color = variant.image;
        if (!stack.Execute(displayInputs, destination, display, view, 1.0f / 60.0f, nullptr, nullptr)
                 .IsOk())
        {
            std::fprintf(stderr, "failed to encode %s\n", variant.name);
            return 1;
        }
        if (!Write(directory, variant.name, destination, heap))
        {
            return 1;
        }
    }

    return 0;
}
