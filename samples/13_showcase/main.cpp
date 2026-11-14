// 13_showcase: every renderer feature built so far, in one frame.
//
// This is not a feature test. It is the picture the phases add up to: a PBR
// interior lit by a sun with soft contact-hardening shadows, image-based
// lighting from a probe captured in the room itself, a metal roughness sweep
// that reflects that probe, 4x4 supersampled anti-aliasing, and the full
// Phase 30 post chain -- ambient occlusion, depth of field, bloom, auto
// exposure, ACES tone mapping and colour grading -- on top.
//
// It writes the same frame twice, once with the post chain and once with tone
// mapping alone, so the chain's contribution is visible side by side.

#include <cyber/core/result.h>
#include <cyber/math/mat4.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/software/antialias.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/ibl.h>
#include <cyber/software/image_io.h>
#include <cyber/software/pbr.h>
#include <cyber/software/post_process.h>
#include <cyber/software/shader.h>
#include <cyber/software/shadow.h>
#include <cyber/software/texture.h>

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

namespace
{

constexpr i32 kWidth = 960;
constexpr i32 kHeight = 540;
constexpr i32 kShadowMapSize = 2048;
constexpr i32 kProbeFaceSize = 128;
constexpr f32 kNearPlane = 0.1f;
constexpr f32 kFarPlane = 60.0f;
constexpr f32 kFieldOfView = 0.78f;
constexpr f32 kPi = cyber::math::kPi;

// The probe sits where the reflective row is, so what the metals mirror is the
// room they actually stand in.
constexpr Vec3 kProbeCenter{0.0f, 2.05f, 0.35f};
const Vec3 kEye{3.45f, 2.25f, 5.35f};
const Vec3 kTarget{-0.15f, 1.05f, -0.35f};

// -----------------------------------------------------------------------
// Shaders
// -----------------------------------------------------------------------

struct SceneUniforms
{
    TransformShaderData transform{};
    PbrLightingData lighting{};
};

void SceneVertex(const VertexInput& input, const void* userData, VertexStageOutput& output) noexcept
{
    TransformVertexShader(input, &static_cast<const SceneUniforms*>(userData)->transform, output);
}

Color ScenePixel(const Varyings& varyings, const void* userData) noexcept
{
    return PbrPixelShader(varyings, &static_cast<const SceneUniforms*>(userData)->lighting);
}

// The ambient-occlusion pass reads world normals encoded into the unit range.
Color NormalPixel(const Varyings& varyings, const void* /*userData*/) noexcept
{
    const Vec3 normal = cyber::math::Normalize(varyings.normal);
    return Color{normal.x * 0.5f + 0.5f, normal.y * 0.5f + 0.5f, normal.z * 0.5f + 0.5f, 1.0f};
}

// -----------------------------------------------------------------------
// Geometry helpers
// -----------------------------------------------------------------------

bool PushTriangle(DynamicArray<VertexInput>& mesh, const Vec3& a, const Vec3& b, const Vec3& c,
                  const Vec3& normal, const Color& color)
{
    VertexInput vertex{};
    vertex.normal = normal;
    vertex.color = color;
    vertex.position = a;
    if (!mesh.TryPushBack(vertex))
    {
        return false;
    }
    vertex.position = b;
    if (!mesh.TryPushBack(vertex))
    {
        return false;
    }
    vertex.position = c;
    return mesh.TryPushBack(vertex);
}

bool PushSmoothTriangle(DynamicArray<VertexInput>& mesh, const Vec3& a, const Vec3& na,
                        const Vec3& b, const Vec3& nb, const Vec3& c, const Vec3& nc,
                        const Color& color)
{
    VertexInput vertex{};
    vertex.color = color;
    vertex.position = a;
    vertex.normal = na;
    if (!mesh.TryPushBack(vertex))
    {
        return false;
    }
    vertex.position = b;
    vertex.normal = nb;
    if (!mesh.TryPushBack(vertex))
    {
        return false;
    }
    vertex.position = c;
    vertex.normal = nc;
    return mesh.TryPushBack(vertex);
}

bool PushQuad(DynamicArray<VertexInput>& mesh, const Vec3& a, const Vec3& b, const Vec3& c,
              const Vec3& d, const Vec3& normal, const Color& color)
{
    return PushTriangle(mesh, a, b, c, normal, color) && PushTriangle(mesh, a, c, d, normal, color);
}

bool PushTexturedQuad(DynamicArray<VertexInput>& mesh, const Vec3& a, const Vec2& uvA,
                      const Vec3& b, const Vec2& uvB, const Vec3& c, const Vec2& uvC, const Vec3& d,
                      const Vec2& uvD, const Vec3& normal)
{
    const auto push = [&](const Vec3& position, const Vec2& uv)
    {
        VertexInput vertex{};
        vertex.position = position;
        vertex.normal = normal;
        vertex.uv = uv;
        vertex.color = Color{1.0f, 1.0f, 1.0f, 1.0f};
        return mesh.TryPushBack(vertex);
    };
    return push(a, uvA) && push(b, uvB) && push(c, uvC) && push(a, uvA) && push(c, uvC) &&
           push(d, uvD);
}

bool AddBox(DynamicArray<VertexInput>& mesh, const Vec3& center, const Vec3& halfExtents,
            const Color& color, f32 rotationY = 0.0f)
{
    const Vec3 normals[6] = {Vec3{0.0f, 0.0f, 1.0f}, Vec3{0.0f, 0.0f, -1.0f},
                             Vec3{1.0f, 0.0f, 0.0f}, Vec3{-1.0f, 0.0f, 0.0f},
                             Vec3{0.0f, 1.0f, 0.0f}, Vec3{0.0f, -1.0f, 0.0f}};
    const Vec3 tangents[6] = {Vec3{1.0f, 0.0f, 0.0f},  Vec3{-1.0f, 0.0f, 0.0f},
                              Vec3{0.0f, 0.0f, -1.0f}, Vec3{0.0f, 0.0f, 1.0f},
                              Vec3{1.0f, 0.0f, 0.0f},  Vec3{1.0f, 0.0f, 0.0f}};
    const Vec3 bitangents[6] = {Vec3{0.0f, 1.0f, 0.0f},  Vec3{0.0f, 1.0f, 0.0f},
                                Vec3{0.0f, 1.0f, 0.0f},  Vec3{0.0f, 1.0f, 0.0f},
                                Vec3{0.0f, 0.0f, -1.0f}, Vec3{0.0f, 0.0f, 1.0f}};
    const f32 sine = std::sin(rotationY);
    const f32 cosine = std::cos(rotationY);
    const auto rotate = [&](const Vec3& value) {
        return Vec3{cosine * value.x + sine * value.z, value.y, -sine * value.x + cosine * value.z};
    };

    for (usize face = 0; face < 6; ++face)
    {
        const Vec3 normal = rotate(normals[face]);
        const f32 tangentExtent = face < 2 ? halfExtents.x : halfExtents.z;
        const f32 bitangentExtent = face < 4 ? halfExtents.y : halfExtents.z;
        const Vec3 faceCenter =
            center +
            rotate(normals[face] *
                   (face < 2 ? halfExtents.z : (face < 4 ? halfExtents.x : halfExtents.y)));
        const Vec3 right = rotate(tangents[face] * tangentExtent);
        const Vec3 up = rotate(bitangents[face] * bitangentExtent);
        if (!PushQuad(mesh, faceCenter - right - up, faceCenter + right - up,
                      faceCenter + right + up, faceCenter - right + up, normal, color))
        {
            return false;
        }
    }
    return true;
}

bool AddSphere(DynamicArray<VertexInput>& mesh, const Vec3& center, f32 radius, const Color& color,
               u32 segments = 48, u32 rings = 32)
{
    const auto normalAt = [](f32 theta, f32 phi) {
        return Vec3{std::sin(theta) * std::cos(phi), std::cos(theta),
                    std::sin(theta) * std::sin(phi)};
    };
    for (u32 ring = 0; ring < rings; ++ring)
    {
        const f32 theta0 = static_cast<f32>(ring) / static_cast<f32>(rings) * kPi;
        const f32 theta1 = static_cast<f32>(ring + 1) / static_cast<f32>(rings) * kPi;
        for (u32 segment = 0; segment < segments; ++segment)
        {
            const f32 phi0 = static_cast<f32>(segment) / static_cast<f32>(segments) * 2.0f * kPi;
            const f32 phi1 =
                static_cast<f32>(segment + 1) / static_cast<f32>(segments) * 2.0f * kPi;
            const Vec3 n00 = normalAt(theta0, phi0);
            const Vec3 n01 = normalAt(theta0, phi1);
            const Vec3 n10 = normalAt(theta1, phi0);
            const Vec3 n11 = normalAt(theta1, phi1);
            if (!PushSmoothTriangle(mesh, center + n00 * radius, n00, center + n10 * radius, n10,
                                    center + n11 * radius, n11, color) ||
                !PushSmoothTriangle(mesh, center + n00 * radius, n00, center + n11 * radius, n11,
                                    center + n01 * radius, n01, color))
            {
                return false;
            }
        }
    }
    return true;
}

// -----------------------------------------------------------------------
// The scene
// -----------------------------------------------------------------------

// A polished stone floor. Darker grout lines give the anti-aliasing fine
// high-contrast detail to resolve at a grazing angle, and the mip chain keeps
// it from shimmering into noise toward the back wall.
void BuildFloorTexture(Texture& texture)
{
    constexpr i32 size = 256;
    constexpr i32 tile = 128;
    for (i32 y = 0; y < size; ++y)
    {
        for (i32 x = 0; x < size; ++x)
        {
            const i32 localX = x % tile;
            const i32 localY = y % tile;
            const bool grout = localX < 3 || localY < 3;
            const i32 tileX = x / tile;
            const i32 tileY = y / tile;
            const i32 noise = (x * 31 + y * 17 + tileX * 131 + tileY * 71) % 23;
            Color texel{};
            if (grout)
            {
                texel = Color{0.022f, 0.026f, 0.032f, 1.0f};
            }
            else
            {
                const bool light = ((tileX + tileY) & 1) != 0;
                const f32 grain = static_cast<f32>(noise) * 0.0028f;
                const Color base = light ? Color{0.245f, 0.255f, 0.275f, 1.0f}
                                         : Color{0.088f, 0.098f, 0.118f, 1.0f};
                texel = Color{base.r + grain, base.g + grain, base.b + grain, 1.0f};
            }
            texture.SetTexel(0, x, y, texel);
        }
    }
    texture.GenerateMips();
}

// The floor: one quad, twelve metres on a side. Two of its triangles straddle
// the near plane behind the camera, which used to put a clipped corner outside
// the rasterizer's guard band and lose the triangle whole. The clipper handles
// that now, so the floor needs no tessellation.
bool BuildFloor(DynamicArray<VertexInput>& mesh)
{
    constexpr f32 e = 6.0f;
    // Twelve texture repeats across twelve metres; the image holds a 2 x 2
    // grid of stone, so the floor reads as half-metre slabs.
    constexpr f32 kRepeats = 12.0f;
    return PushTexturedQuad(mesh, Vec3{-e, 0.0f, -e}, Vec2{0.0f, 0.0f}, Vec3{-e, 0.0f, e},
                            Vec2{0.0f, kRepeats}, Vec3{e, 0.0f, e}, Vec2{kRepeats, kRepeats},
                            Vec3{e, 0.0f, -e}, Vec2{kRepeats, 0.0f}, Vec3{0.0f, 1.0f, 0.0f});
}

// Walls, ceiling, and the ceiling light panel. The panel's vertex colour is far
// above 1, which is what gives the tone mapper a shoulder to roll off and
// bloom something to scatter.
bool BuildRoom(DynamicArray<VertexInput>& mesh)
{
    const Color backWall{0.185f, 0.205f, 0.245f, 1.0f};
    const Color sideWall{0.145f, 0.160f, 0.200f, 1.0f};
    const Color ceiling{0.230f, 0.235f, 0.250f, 1.0f};
    constexpr f32 e = 6.0f;
    constexpr f32 h = 4.6f;
    if (!PushQuad(mesh, Vec3{-e, 0.0f, -e}, Vec3{e, 0.0f, -e}, Vec3{e, h, -e}, Vec3{-e, h, -e},
                  Vec3{0.0f, 0.0f, 1.0f}, backWall) ||
        !PushQuad(mesh, Vec3{-e, 0.0f, e}, Vec3{-e, h, e}, Vec3{e, h, e}, Vec3{e, 0.0f, e},
                  Vec3{0.0f, 0.0f, -1.0f}, sideWall) ||
        !PushQuad(mesh, Vec3{-e, 0.0f, -e}, Vec3{-e, h, -e}, Vec3{-e, h, e}, Vec3{-e, 0.0f, e},
                  Vec3{1.0f, 0.0f, 0.0f}, sideWall) ||
        !PushQuad(mesh, Vec3{e, 0.0f, e}, Vec3{e, h, e}, Vec3{e, h, -e}, Vec3{e, 0.0f, -e},
                  Vec3{-1.0f, 0.0f, 0.0f}, sideWall) ||
        !PushQuad(mesh, Vec3{-e, h, e}, Vec3{-e, h, -e}, Vec3{e, h, -e}, Vec3{e, h, e},
                  Vec3{0.0f, -1.0f, 0.0f}, ceiling))
    {
        return false;
    }

    // Two recessed light panels in the ceiling, and a warm strip high on the
    // back wall. All three are bloom sources and all three show up in the
    // metals as reflected highlights.
    return AddBox(mesh, Vec3{-1.9f, h - 0.06f, 0.9f}, Vec3{1.5f, 0.05f, 0.30f},
                  Color{26.0f, 24.0f, 20.0f, 1.0f}) &&
           AddBox(mesh, Vec3{2.1f, h - 0.06f, -1.4f}, Vec3{1.2f, 0.05f, 0.26f},
                  Color{18.0f, 19.0f, 24.0f, 1.0f}) &&
           AddBox(mesh, Vec3{0.0f, 3.25f, -5.9f}, Vec3{3.4f, 0.09f, 0.06f},
                  Color{22.0f, 13.0f, 6.0f, 1.0f});
}

// The pedestal the reflective row stands on, plus a few props at different
// depths so depth of field has near and far material to work with.
bool BuildProps(DynamicArray<VertexInput>& mesh)
{
    return AddBox(mesh, Vec3{0.0f, 0.195f, 0.2f}, Vec3{3.15f, 0.195f, 1.05f},
                  Color{0.125f, 0.140f, 0.165f, 1.0f}) &&
           AddBox(mesh, Vec3{-4.15f, 0.62f, -2.35f}, Vec3{0.60f, 0.62f, 0.60f},
                  Color{0.145f, 0.265f, 0.470f, 1.0f}, -0.26f) &&
           AddBox(mesh, Vec3{4.25f, 0.46f, -1.15f}, Vec3{0.72f, 0.46f, 0.58f},
                  Color{0.470f, 0.165f, 0.105f, 1.0f}, 0.21f) &&
           AddBox(mesh, Vec3{-2.55f, 0.30f, 3.35f}, Vec3{0.52f, 0.30f, 0.52f},
                  Color{0.155f, 0.330f, 0.215f, 1.0f}, 0.14f) &&
           AddSphere(mesh, Vec3{3.05f, 0.44f, 2.55f}, 0.44f, Color{0.330f, 0.145f, 0.285f, 1.0f},
                     32, 22) &&
           AddBox(mesh, Vec3{-4.6f, 1.55f, -5.86f}, Vec3{0.95f, 1.15f, 0.07f},
                  Color{0.330f, 0.175f, 0.075f, 1.0f}) &&
           AddBox(mesh, Vec3{3.6f, 1.75f, -5.86f}, Vec3{0.78f, 0.98f, 0.07f},
                  Color{0.085f, 0.215f, 0.300f, 1.0f});
}

// -----------------------------------------------------------------------
// Draw list
// -----------------------------------------------------------------------

// One material, one mesh, one choice of where its ambient light comes from.
// Metals read the local probe, which gives them a mirror of this room; the
// matte surfaces read the global environment, which is the same capture
// convolved down to irradiance.
struct Draw
{
    const DynamicArray<VertexInput>* mesh = nullptr;
    PbrMaterial material{};
    const Texture* texture = nullptr;
    bool useProbes = false;
};

struct SceneLighting
{
    const PbrDirectionalLight* sun = nullptr;
    const PbrDirectionalLight* fill = nullptr;
    const PbrPointLight* points = nullptr;
    u32 pointCount = 0;
    const ReflectionProbeSet* probes = nullptr;
    const EnvironmentLighting* environment = nullptr;
};

void FillLighting(PbrLightingData& lighting, const SceneLighting& scene, const Draw& draw,
                  const Vec3& eye, const Vec3& forward)
{
    lighting.eyePosition = eye;
    lighting.cameraForward = forward;
    lighting.ambientIrradiance = Vec3{0.040f, 0.046f, 0.060f};
    lighting.material = draw.material;
    lighting.baseColorTexture = draw.texture;
    lighting.sampler.addressU = AddressMode::Repeat;
    lighting.sampler.addressV = AddressMode::Repeat;
    if (draw.useProbes && scene.probes != nullptr)
    {
        lighting.reflectionProbes = *scene.probes;
    }
    else
    {
        lighting.environment = scene.environment;
    }
    lighting.directional[0] = *scene.sun;
    lighting.directional[1] = *scene.fill;
    lighting.directionalCount = 2;
    for (u32 index = 0; index < scene.pointCount && index < kMaxPbrPointLights; ++index)
    {
        lighting.points[index] = scene.points[index];
    }
    lighting.pointCount = scene.pointCount;
}

[[nodiscard]] bool DrawScene(Span<const Draw> draws, const SceneLighting& scene, const Vec3& eye,
                             const Vec3& target, Framebuffer& color, DepthBuffer& depth,
                             HeapAllocator& allocator)
{
    const Mat4 view = Mat4::LookAt(eye, target, Vec3{0.0f, 1.0f, 0.0f});
    const Mat4 projection = Mat4::PerspectiveReverseZ(
        kFieldOfView, static_cast<f32>(kWidth) / static_cast<f32>(kHeight), kNearPlane, kFarPlane);
    const Vec3 forward = cyber::math::Normalize(target - eye);

    for (usize index = 0; index < draws.Size(); ++index)
    {
        const Draw& draw = draws[index];
        SceneUniforms uniforms{};
        uniforms.transform.modelViewProjection = projection * view;
        uniforms.transform.modelToWorld = Mat4::Identity();
        uniforms.transform.normalToWorld = Mat4::Identity();
        uniforms.transform.eyePosition = eye;
        FillLighting(uniforms.lighting, scene, draw, eye, forward);

        ShaderDesc shader{};
        shader.pipeline.view = view;
        shader.pipeline.projection = projection;
        shader.pipeline.viewport =
            Viewport{0.0f, 0.0f, static_cast<f32>(color.Width()), static_cast<f32>(color.Height())};
        shader.pipeline.cullMode = CullMode::None;
        shader.pipeline.useJobs = true;
        shader.vertexShader = &SceneVertex;
        shader.pixelShader = &ScenePixel;
        shader.userData = &uniforms;
        if (!DrawShadedTriangles(shader,
                                 Span<const VertexInput>{draw.mesh->Data(), draw.mesh->Size()},
                                 color, depth, nullptr, allocator)
                 .IsOk())
        {
            return false;
        }
    }
    return true;
}

// The world-normal and depth buffers the post chain reads, at output
// resolution. The colour pass renders supersampled, so these cannot be shared
// with it.
[[nodiscard]] bool DrawNormalsAndDepth(Span<const Draw> draws, const Vec3& eye, const Vec3& target,
                                       Framebuffer& normals, DepthBuffer& depth,
                                       HeapAllocator& allocator)
{
    const Mat4 view = Mat4::LookAt(eye, target, Vec3{0.0f, 1.0f, 0.0f});
    const Mat4 projection = Mat4::PerspectiveReverseZ(
        kFieldOfView, static_cast<f32>(kWidth) / static_cast<f32>(kHeight), kNearPlane, kFarPlane);

    normals.Clear(Color{0.5f, 0.5f, 0.0f, 1.0f});
    depth.Clear();
    for (usize index = 0; index < draws.Size(); ++index)
    {
        SceneUniforms uniforms{};
        uniforms.transform.modelViewProjection = projection * view;
        uniforms.transform.modelToWorld = Mat4::Identity();
        uniforms.transform.normalToWorld = Mat4::Identity();
        uniforms.transform.eyePosition = eye;

        ShaderDesc shader{};
        shader.pipeline.view = view;
        shader.pipeline.projection = projection;
        shader.pipeline.viewport =
            Viewport{0.0f, 0.0f, static_cast<f32>(kWidth), static_cast<f32>(kHeight)};
        shader.pipeline.cullMode = CullMode::None;
        shader.pipeline.useJobs = true;
        shader.vertexShader = &SceneVertex;
        shader.pixelShader = &NormalPixel;
        shader.userData = &uniforms;
        if (!DrawShadedTriangles(
                 shader,
                 Span<const VertexInput>{draws[index].mesh->Data(), draws[index].mesh->Size()},
                 normals, depth, nullptr, allocator)
                 .IsOk())
        {
            return false;
        }
    }
    return true;
}

// -----------------------------------------------------------------------
// Probe capture
// -----------------------------------------------------------------------

struct CaptureContext
{
    Span<const Draw> draws{};
    const SceneLighting* scene = nullptr;
    HeapAllocator* allocator = nullptr;
};

cyber::core::Status<cyber::core::Error> CaptureFace(const ReflectionProbeCaptureView& view,
                                                    Framebuffer& color, DepthBuffer& depth,
                                                    void* userData) noexcept
{
    auto& context = *static_cast<CaptureContext*>(userData);
    const Vec3 forward = CubeDirectionFromFaceUv(static_cast<u32>(view.face), 0.5f, 0.5f);
    for (usize index = 0; index < context.draws.Size(); ++index)
    {
        Draw draw = context.draws[index];
        // The capture is what the metals will mirror. They are left out of it:
        // a probe cannot reflect itself, and the row stands close enough to the
        // probe that including it would fill the cube with its own dark backs.
        if (draw.useProbes)
        {
            continue;
        }
        SceneUniforms uniforms{};
        uniforms.transform.modelViewProjection = view.projection * view.view;
        uniforms.transform.modelToWorld = Mat4::Identity();
        uniforms.transform.normalToWorld = Mat4::Identity();
        uniforms.transform.eyePosition = view.position;
        FillLighting(uniforms.lighting, *context.scene, draw, view.position, forward);
        uniforms.lighting.environment = nullptr;
        uniforms.lighting.ambientIrradiance = Vec3{0.040f, 0.046f, 0.060f};

        ShaderDesc shader{};
        shader.pipeline.view = view.view;
        shader.pipeline.projection = view.projection;
        shader.pipeline.viewport =
            Viewport{0.0f, 0.0f, static_cast<f32>(color.Width()), static_cast<f32>(color.Height())};
        shader.pipeline.cullMode = CullMode::None;
        shader.pipeline.useJobs = true;
        shader.vertexShader = &SceneVertex;
        shader.pixelShader = &ScenePixel;
        shader.userData = &uniforms;
        if (!DrawShadedTriangles(shader,
                                 Span<const VertexInput>{draw.mesh->Data(), draw.mesh->Size()},
                                 color, depth, nullptr, *context.allocator)
                 .IsOk())
        {
            return cyber::core::Status<cyber::core::Error>::Err(cyber::core::Error{
                cyber::core::ErrorCode::Internal, "failed to render a reflection probe face"});
        }
    }
    return cyber::core::Status<cyber::core::Error>::Ok();
}

// -----------------------------------------------------------------------

[[nodiscard]] bool MakeImage(HeapAllocator& heap, i32 width, i32 height, Framebuffer& out)
{
    FramebufferDesc desc{};
    desc.width = width;
    desc.height = height;
    desc.format = PixelFormat::RGBA32F;
    auto created = Framebuffer::Create(desc, heap);
    if (!created)
    {
        return false;
    }
    out = std::move(created).Value();
    return true;
}

[[nodiscard]] bool Write(const char* directory, const char* name, const Framebuffer& image,
                         HeapAllocator& allocator)
{
    char path[512];
    std::snprintf(path, sizeof(path), "%s%s", directory, name);
    ImageWriteDesc desc{};
    desc.format = ImageFormat::Ppm;
    desc.encodeSrgb = true;
    if (!WriteImage(path, image, allocator, desc))
    {
        std::fprintf(stderr, "failed to write %s\n", path);
        return false;
    }
    std::printf("  wrote %s\n", path);
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    const char* directory = argc > 1 ? argv[1] : "";

    HeapAllocator heap;

    // ---------------------------------------------------------------- scene
    DynamicArray<VertexInput> room{heap};
    DynamicArray<VertexInput> floorMesh{heap};
    DynamicArray<VertexInput> props{heap};
    DynamicArray<VertexInput> metals{heap};
    DynamicArray<VertexInput> chrome{heap};
    DynamicArray<VertexInput> casters{heap};

    if (!BuildRoom(room) || !BuildProps(props))
    {
        std::fprintf(stderr, "failed to build the room\n");
        return 1;
    }
    if (!BuildFloor(floorMesh))
    {
        std::fprintf(stderr, "failed to build the floor\n");
        return 1;
    }

    // The roughness sweep. One mesh, because the row shares a material except
    // for roughness, which is varied per sphere below by splitting the draw.
    constexpr i32 kSphereCount = 5;
    constexpr f32 kSphereRadius = 0.43f;
    DynamicArray<VertexInput> spheres[kSphereCount] = {
        DynamicArray<VertexInput>{heap}, DynamicArray<VertexInput>{heap},
        DynamicArray<VertexInput>{heap}, DynamicArray<VertexInput>{heap},
        DynamicArray<VertexInput>{heap}};
    for (i32 index = 0; index < kSphereCount; ++index)
    {
        const f32 x = -2.32f + static_cast<f32>(index) * 1.16f;
        if (!AddSphere(spheres[index], Vec3{x, 0.39f + kSphereRadius, 0.2f}, kSphereRadius,
                       Color{0.955f, 0.930f, 0.880f, 1.0f}))
        {
            std::fprintf(stderr, "failed to build the roughness sweep\n");
            return 1;
        }
    }
    // A mirror-finish sphere behind the row, so there is one surface in frame
    // that is a pure reflection of the probe.
    if (!AddSphere(chrome, Vec3{-0.15f, 0.95f, -3.3f}, 0.95f, Color{0.97f, 0.98f, 1.0f, 1.0f}, 64,
                   44))
    {
        std::fprintf(stderr, "failed to build the mirror sphere\n");
        return 1;
    }

    // ------------------------------------------------------------- shadows
    // The sun comes from the upper left, slightly in front of the subject, so the
    // shadows fall to the right and back where the camera can see them. A key
    // light placed behind the camera hides its own shadows.
    const Vec3 lightTravel = cyber::math::Normalize(Vec3{0.62f, -0.78f, -0.33f});
    const Vec3 lightCenter{0.0f, 1.3f, -0.4f};
    const Mat4 lightView =
        Mat4::LookAt(lightCenter - lightTravel * 24.0f, lightCenter, Vec3{0.0f, 1.0f, 0.0f});
    // The map has to cover every *receiver*, not just the casters: a surface
    // that falls outside it has no depth to compare against and reads as
    // shadowed. The room's 12 x 12 x 4.6 box projects well past its diagonal
    // along the light's axes, so the half-extent is sized for that, not for
    // the handful of objects that actually cast.
    constexpr f32 kShadowExtent = 13.0f;
    const Mat4 lightProjection = Mat4::OrthographicReverseZ(
        -kShadowExtent, kShadowExtent, -kShadowExtent, kShadowExtent, 0.1f, 60.0f);

    // Only what stands in the room casts. The room itself is a closed box:
    // its ceiling would occlude the sun everywhere, and its flat floor and
    // walls only ever self-shadow into acne.
    const DynamicArray<VertexInput>* casterMeshes[] = {
        &props, &chrome, &spheres[0], &spheres[1], &spheres[2], &spheres[3], &spheres[4]};
    for (const DynamicArray<VertexInput>* mesh : casterMeshes)
    {
        for (usize index = 0; index < mesh->Size(); ++index)
        {
            if (!casters.TryPushBack((*mesh)[index]))
            {
                std::fprintf(stderr, "failed to gather the shadow casters\n");
                return 1;
            }
        }
    }

    DepthBufferDesc shadowDesc{};
    shadowDesc.width = kShadowMapSize;
    shadowDesc.height = kShadowMapSize;
    auto shadowCreated = DepthBuffer::Create(shadowDesc, heap);
    if (!shadowCreated)
    {
        std::fprintf(stderr, "failed to allocate the shadow map\n");
        return 1;
    }
    DepthBuffer shadowDepth = std::move(shadowCreated).Value();

    ShadowRenderDesc shadowRender{};
    shadowRender.clipFromWorld = lightProjection * lightView;
    shadowRender.cullMode = CullMode::None;
    shadowRender.useJobs = true;
    if (!RenderShadowMap(shadowRender, Span<const VertexInput>{casters.Data(), casters.Size()},
                         shadowDepth, nullptr, heap))
    {
        std::fprintf(stderr, "failed to render the shadow map\n");
        return 1;
    }

    PbrDirectionalLight sun{};
    sun.direction = lightTravel;
    sun.radiance = Vec3{5.20f, 4.75f, 4.05f};
    sun.shadow.depth = &shadowDepth;
    sun.shadow.clipFromWorld = shadowRender.clipFromWorld;
    sun.shadow.constantDepthBias = 0.0012f;
    sun.shadow.slopeDepthBias = 0.0035f;
    sun.shadow.normalOffsetBias = 0.012f;
    // Percentage-closer soft shadows: the penumbra widens with the distance
    // between the caster and what it lands on, so contact stays sharp.
    sun.shadow.nearPlane = 0.1f;
    sun.shadow.farPlane = 60.0f;
    sun.shadow.usePcss = true;
    sun.shadow.softFilter = true;
    sun.shadow.filterRadius = 8;
    sun.shadow.perspectiveProjection = false;
    sun.shadow.worldUnitsPerTexel = (2.0f * kShadowExtent) / static_cast<f32>(kShadowMapSize);
    sun.shadow.sourceAngularRadiusRadians = 0.013f;

    // The three emissive fixtures, as actual light sources. The boxes above
    // are what the eye and the metals see; these are what they emit.
    PbrPointLight panels[3]{};
    panels[0].position = Vec3{-1.9f, 4.35f, 0.9f};
    panels[0].intensity = Vec3{6.5f, 5.9f, 4.7f};
    panels[0].range = 14.0f;
    panels[1].position = Vec3{2.1f, 4.35f, -1.4f};
    panels[1].intensity = Vec3{3.6f, 4.0f, 5.0f};
    panels[1].range = 13.0f;
    panels[2].position = Vec3{0.0f, 3.25f, -5.6f};
    panels[2].intensity = Vec3{4.5f, 2.7f, 1.3f};
    panels[2].range = 10.0f;

    PbrDirectionalLight fill{};
    fill.direction = cyber::math::Normalize(Vec3{0.52f, -0.42f, 0.74f});
    fill.radiance = Vec3{0.14f, 0.18f, 0.27f};

    // ----------------------------------------------------------- materials
    PbrMaterial wallMaterial{};
    wallMaterial.baseColor = Color{1.0f, 1.0f, 1.0f, 1.0f};
    wallMaterial.metallic = 0.0f;
    wallMaterial.roughness = 0.62f;

    PbrMaterial floorMaterial{};
    floorMaterial.baseColor = Color{1.0f, 1.0f, 1.0f, 1.0f};
    floorMaterial.metallic = 0.05f;
    floorMaterial.roughness = 0.34f;

    PbrMaterial propMaterial{};
    propMaterial.baseColor = Color{1.0f, 1.0f, 1.0f, 1.0f};
    propMaterial.metallic = 0.04f;
    propMaterial.roughness = 0.48f;

    PbrMaterial chromeMaterial{};
    chromeMaterial.baseColor = Color{0.97f, 0.98f, 1.0f, 1.0f};
    chromeMaterial.metallic = 1.0f;
    chromeMaterial.roughness = 0.045f;

    TextureDesc floorTextureDesc{};
    floorTextureDesc.width = 256;
    floorTextureDesc.height = 256;
    floorTextureDesc.format = TextureFormat::RGBA32F;
    floorTextureDesc.generateMips = true;
    auto floorTextureCreated = Texture::Create(floorTextureDesc, heap);
    if (!floorTextureCreated)
    {
        std::fprintf(stderr, "failed to create the floor texture\n");
        return 1;
    }
    Texture floorTexture = std::move(floorTextureCreated).Value();
    BuildFloorTexture(floorTexture);

    // ------------------------------------------------------------ draw list
    DynamicArray<Draw> draws{heap};
    const auto pushDraw = [&](const DynamicArray<VertexInput>* mesh, const PbrMaterial& material,
                              const Texture* texture, bool useProbes)
    {
        Draw draw{};
        draw.mesh = mesh;
        draw.material = material;
        draw.texture = texture;
        draw.useProbes = useProbes;
        return draws.TryPushBack(draw);
    };
    bool built = pushDraw(&room, wallMaterial, nullptr, false) &&
                 pushDraw(&floorMesh, floorMaterial, &floorTexture, false) &&
                 pushDraw(&props, propMaterial, nullptr, false) &&
                 pushDraw(&chrome, chromeMaterial, nullptr, true);
    for (i32 index = 0; index < kSphereCount && built; ++index)
    {
        PbrMaterial metal{};
        metal.baseColor = Color{0.955f, 0.930f, 0.880f, 1.0f};
        metal.metallic = 1.0f;
        // 0.04 to 0.46: a mirror at one end, a brushed finish at the other.
        metal.roughness = 0.04f + static_cast<f32>(index) * 0.105f;
        built = pushDraw(&spheres[index], metal, nullptr, true);
    }
    if (!built)
    {
        std::fprintf(stderr, "failed to build the draw list\n");
        return 1;
    }
    const Span<const Draw> drawSpan{draws.Data(), draws.Size()};

    // ------------------------------------------------------------------ IBL
    SceneLighting lighting{};
    lighting.sun = &sun;
    lighting.fill = &fill;
    lighting.points = panels;
    lighting.pointCount = 3;

    std::printf("capturing the reflection probe (6 faces at %dx%d)\n", kProbeFaceSize,
                kProbeFaceSize);
    ReflectionProbeCaptureDesc captureDesc{};
    captureDesc.position = kProbeCenter;
    captureDesc.faceSize = kProbeFaceSize;
    captureDesc.supersampleFactor = kDefaultSupersampleFactor;
    captureDesc.resolveFilter = kDefaultResolveFilter;
    captureDesc.nearPlane = 0.06f;
    captureDesc.farPlane = 40.0f;
    CaptureContext captureContext{drawSpan, &lighting, &heap};
    auto captured = CaptureReflectionProbe(captureDesc, &CaptureFace, &captureContext, heap);
    if (!captured)
    {
        std::fprintf(stderr, "probe capture failed: %s\n", captured.Error().message);
        return 1;
    }

    EnvironmentMapDesc environmentDesc{};
    environmentDesc.faceSize = kProbeFaceSize;
    environmentDesc.diffuseFaceSize = 16;
    environmentDesc.brdfLutSize = 64;
    environmentDesc.specularSampleCount = 128;
    environmentDesc.brdfSampleCount = 256;
    environmentDesc.useJobs = true;
    auto environmentCreated = EnvironmentMap::Create(captured.Value(), environmentDesc, heap);
    if (!environmentCreated)
    {
        std::fprintf(stderr, "environment preprocessing failed: %s\n",
                     environmentCreated.Error().message);
        return 1;
    }
    EnvironmentMap environmentMap = std::move(environmentCreated).Value();
    EnvironmentLighting environment = environmentMap.Lighting();
    environment.intensity = 0.80f;

    ReflectionProbe probe{};
    probe.position = kProbeCenter;
    probe.influenceMin = Vec3{-5.9f, 0.0f, -5.9f};
    probe.influenceMax = Vec3{5.9f, 4.5f, 5.9f};
    probe.proxyMin = Vec3{-6.0f, 0.0f, -6.0f};
    probe.proxyMax = Vec3{6.0f, 4.6f, 6.0f};
    probe.blendDistance = 0.5f;
    probe.environment = environment;
    const ReflectionProbe probeArray[] = {probe};
    ReflectionProbeSet probes{};
    probes.probes = Span<const ReflectionProbe>{probeArray, 1};
    probes.fallback = &environment;

    lighting.probes = &probes;
    lighting.environment = &environment;

    // --------------------------------------------------------- the main pass
    AntiAliasTargetDesc antialiasDesc{};
    antialiasDesc.width = kWidth;
    antialiasDesc.height = kHeight;
    antialiasDesc.quality = AntiAliasQuality::High;
    antialiasDesc.filter = kDefaultResolveFilter;
    antialiasDesc.format = PixelFormat::RGBA32F;
    auto antialiasCreated = AntiAliasTarget::Create(antialiasDesc, heap);
    if (!antialiasCreated)
    {
        std::fprintf(stderr, "failed to allocate the supersampled target: %s\n",
                     antialiasCreated.Error().message);
        return 1;
    }
    AntiAliasTarget antialias = std::move(antialiasCreated).Value();
    antialias.Clear(Color{0.012f, 0.016f, 0.026f, 1.0f});

    std::printf("shading %dx%d at %dx%d samples per pixel\n", kWidth, kHeight, antialias.Factor(),
                antialias.Factor());
    if (!DrawScene(drawSpan, lighting, kEye, kTarget, antialias.Color(), antialias.Depth(), heap))
    {
        std::fprintf(stderr, "the main pass failed\n");
        return 1;
    }

    Framebuffer resolved;
    if (!MakeImage(heap, kWidth, kHeight, resolved))
    {
        std::fprintf(stderr, "failed to allocate the resolve target\n");
        return 1;
    }
    ResolveDesc resolveDesc{};
    resolveDesc.factor = antialias.Factor();
    resolveDesc.filter = kDefaultResolveFilter;
    resolveDesc.useJobs = true;
    if (!antialias.Resolve(resolved, nullptr, heap, resolveDesc))
    {
        std::fprintf(stderr, "the resolve failed\n");
        return 1;
    }

    // ------------------------------------------------- post-process inputs
    Framebuffer normals;
    if (!MakeImage(heap, kWidth, kHeight, normals))
    {
        std::fprintf(stderr, "failed to allocate the normal buffer\n");
        return 1;
    }
    DepthBufferDesc postDepthDesc{};
    postDepthDesc.width = kWidth;
    postDepthDesc.height = kHeight;
    auto postDepthCreated = DepthBuffer::Create(postDepthDesc, heap);
    if (!postDepthCreated)
    {
        std::fprintf(stderr, "failed to allocate the post depth buffer\n");
        return 1;
    }
    DepthBuffer postDepth = std::move(postDepthCreated).Value();
    if (!DrawNormalsAndDepth(drawSpan, kEye, kTarget, normals, postDepth, heap))
    {
        std::fprintf(stderr, "the normal and depth pass failed\n");
        return 1;
    }

    auto stackCreated = PostProcessStack::Create(kWidth, kHeight, heap);
    if (!stackCreated)
    {
        std::fprintf(stderr, "failed to create the post-processing stack\n");
        return 1;
    }
    PostProcessStack stack = std::move(stackCreated).Value();

    ColorGradeSettings gradeSettings{};
    gradeSettings.contrast = 1.07f;
    gradeSettings.saturation = 1.09f;
    gradeSettings.temperature = 0.06f;
    gradeSettings.tint = -0.015f;
    auto lutCreated = ColorGradeLut::Bake(gradeSettings, kDefaultColorGradeLutSize, heap);
    if (!lutCreated)
    {
        std::fprintf(stderr, "failed to bake the grading table\n");
        return 1;
    }
    const ColorGradeLut lut = std::move(lutCreated).Value();

    PostProcessView view{};
    view.projection = Mat4::PerspectiveReverseZ(
        kFieldOfView, static_cast<f32>(kWidth) / static_cast<f32>(kHeight), kNearPlane, kFarPlane);
    view.view = Mat4::LookAt(kEye, kTarget, Vec3{0.0f, 1.0f, 0.0f});
    view.nearPlane = kNearPlane;
    view.farPlane = kFarPlane;

    PostProcessSettings settings{};
    settings.enableAmbientOcclusion = true;
    settings.enableDepthOfField = true;
    settings.enableMotionBlur = false;  // a still frame has no velocity to follow
    settings.enableBloom = true;
    settings.enableExposure = true;
    settings.enableToneMap = true;
    settings.enableColorGrade = true;
    settings.exposure.compensationEv = -0.30f;
    settings.colorGradeLut = &lut;
    settings.useJobs = true;
    settings.ambientOcclusion.radius = 0.50f;
    settings.ambientOcclusion.intensity = 0.80f;
    settings.ambientOcclusion.sampleCount = 24;
    settings.ambientOcclusion.power = 1.15f;
    // Focus on the roughness sweep; the back wall and the near props fall off.
    settings.depthOfField.focusDistance = 6.4f;
    settings.depthOfField.focalLengthMm = 50.0f;
    settings.depthOfField.apertureFStop = 2.0f;
    settings.depthOfField.maxCocPixels = 9.0f;
    settings.depthOfField.sampleCount = 32;
    settings.bloom.threshold = 1.0f;
    settings.bloom.intensity = 0.075f;
    settings.bloom.scatter = 0.72f;
    settings.toneMap.whitePoint = 4.0f;

    PostProcessInputs inputs{};
    inputs.color = &resolved;
    inputs.depth = &postDepth;
    inputs.normals = &normals;

    Framebuffer finalImage;
    Framebuffer plainImage;
    if (!MakeImage(heap, kWidth, kHeight, finalImage) ||
        !MakeImage(heap, kWidth, kHeight, plainImage))
    {
        std::fprintf(stderr, "failed to allocate the output images\n");
        return 1;
    }

    PostProcessTimings timings{};
    if (!stack.Execute(inputs, finalImage, settings, view, 1.0f / 60.0f, nullptr, &timings).IsOk())
    {
        std::fprintf(stderr, "the post-processing chain failed\n");
        return 1;
    }

    // The same frame with tone mapping alone, as the comparison.
    PostProcessSettings plain{};
    plain.enableAmbientOcclusion = false;
    plain.enableDepthOfField = false;
    plain.enableMotionBlur = false;
    plain.enableBloom = false;
    plain.enableExposure = true;
    plain.enableToneMap = true;
    plain.enableColorGrade = false;
    plain.useJobs = true;
    if (!stack.Execute(inputs, plainImage, plain, view, 1.0f / 60.0f, nullptr, nullptr).IsOk())
    {
        std::fprintf(stderr, "the comparison pass failed\n");
        return 1;
    }

    // ------------------------------------------------------------- report
    usize triangles = 0;
    for (usize index = 0; index < draws.Size(); ++index)
    {
        triangles += draws[index].mesh->Size() / 3;
    }

    std::printf("\n%zu triangles, %zu draws, %d x %d shadow map, PCSS soft shadows\n", triangles,
                draws.Size(), kShadowMapSize, kShadowMapSize);
    std::printf("probe: 6 x %d cube, %d-sample specular prefilter, %d x %d BRDF table\n",
                kProbeFaceSize, environmentDesc.specularSampleCount, environmentDesc.brdfLutSize,
                environmentDesc.brdfLutSize);
    std::printf("exposure: metered %.4f, %+.2f EV\n\n",
                static_cast<double>(timings.averageLuminance),
                static_cast<double>(timings.exposureEv));
    std::printf("  %-20s %10s  %s\n", "post pass", "ms", "state");
    for (const PassTiming& timing : timings.passes)
    {
        std::printf("  %-20s %10.3f  %s\n", timing.name != nullptr ? timing.name : "?",
                    static_cast<double>(timing.nanoseconds) * 1.0e-6,
                    timing.executed ? "on" : "off");
    }
    std::printf("  %-20s %10.3f\n\n", "total",
                static_cast<double>(timings.totalNanoseconds) * 1.0e-6);

    if (!Write(directory, "showcase.ppm", finalImage, heap) ||
        !Write(directory, "showcase_no_post.ppm", plainImage, heap))
    {
        return 1;
    }
    return 0;
}
