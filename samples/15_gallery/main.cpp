// 15_gallery: the phase-31 renderer in one rich frame.
//
// A twelve-metre room full of material and shape variety -- a metal roughness
// sweep, a chrome sphere, a gold torus, a copper cylinder, a porcelain cone,
// plastic and rubber spheres, painted walls, a textured stone floor and three
// emissive fixtures -- rendered twice through the software path:
//
//   left   phase 29/30: image-based diffuse and specular, PCSS sun shadows,
//          lamps with no shadows, full post chain.
//   right  phase 31: the diffuse ambient comes from a path-traced irradiance
//          volume baked in this room, and every lamp is shadowed by a ray
//          against the same BVH the bake used.
//
// It writes the phase-31 frame on its own as the hero image, and the pair side
// by side as the comparison.

#include <cyber/core/result.h>
#include <cyber/jobs/job_system.h>
#include <cyber/math/mat4.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/software/antialias.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/global_illumination.h>
#include <cyber/software/ibl.h>
#include <cyber/software/image_io.h>
#include <cyber/software/pbr.h>
#include <cyber/software/post_process.h>
#include <cyber/software/raytrace.h>
#include <cyber/software/shader.h>
#include <cyber/software/shadow.h>
#include <cyber/software/texture.h>

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
using cyber::math::Vec2;
using cyber::math::Vec3;

namespace
{

constexpr i32 kWidth = 1024;
constexpr i32 kHeight = 576;
constexpr i32 kShadowMapSize = 2048;
constexpr i32 kProbeFaceSize = 128;
constexpr f32 kNearPlane = 0.1f;
constexpr f32 kFarPlane = 60.0f;
constexpr f32 kFieldOfView = 0.80f;
constexpr f32 kPi = cyber::math::kPi;

constexpr Vec3 kProbeCenter{0.0f, 2.05f, 0.35f};
const Vec3 kEye{3.80f, 3.20f, 11.00f};
const Vec3 kTarget{0.00f, 1.20f, -0.20f};

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
    return PushTriangle(mesh, a, b, c, normal, color) &&
           PushTriangle(mesh, a, c, d, normal, color);
}

bool PushTexturedQuad(DynamicArray<VertexInput>& mesh, const Vec3& a, const Vec2& uvA,
                      const Vec3& b, const Vec2& uvB, const Vec3& c, const Vec2& uvC,
                      const Vec3& d, const Vec2& uvD, const Vec3& normal)
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
    const Vec3 tangents[6] = {Vec3{1.0f, 0.0f, 0.0f}, Vec3{-1.0f, 0.0f, 0.0f},
                              Vec3{0.0f, 0.0f, -1.0f}, Vec3{0.0f, 0.0f, 1.0f},
                              Vec3{1.0f, 0.0f, 0.0f}, Vec3{1.0f, 0.0f, 0.0f}};
    const Vec3 bitangents[6] = {Vec3{0.0f, 1.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f},
                                Vec3{0.0f, 1.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f},
                                Vec3{0.0f, 0.0f, -1.0f}, Vec3{0.0f, 0.0f, 1.0f}};
    const f32 sine = std::sin(rotationY);
    const f32 cosine = std::cos(rotationY);
    const auto rotate = [&](const Vec3& value)
    {
        return Vec3{cosine * value.x + sine * value.z, value.y,
                    -sine * value.x + cosine * value.z};
    };

    for (usize face = 0; face < 6; ++face)
    {
        const Vec3 normal = rotate(normals[face]);
        const f32 tangentExtent = face < 2 ? halfExtents.x : halfExtents.z;
        const f32 bitangentExtent = face < 4 ? halfExtents.y : halfExtents.z;
        const Vec3 faceCenter =
            center + rotate(normals[face] * (face < 2 ? halfExtents.z
                                                      : (face < 4 ? halfExtents.x
                                                                  : halfExtents.y)));
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
    const auto normalAt = [](f32 theta, f32 phi)
    {
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

// A torus about the Y axis: curvature in two directions, which sweeps a
// highlight across a metal the way a sphere cannot.
bool AddTorus(DynamicArray<VertexInput>& mesh, const Vec3& center, f32 majorRadius,
              f32 minorRadius, const Color& color, u32 major = 64, u32 minor = 28)
{
    const auto point = [&](f32 u, f32 v, Vec3& position, Vec3& normal)
    {
        const f32 cu = std::cos(u);
        const f32 su = std::sin(u);
        const f32 cv = std::cos(v);
        const f32 sv = std::sin(v);
        normal = Vec3{cv * cu, sv, cv * su};
        position = center + Vec3{(majorRadius + minorRadius * cv) * cu, minorRadius * sv,
                                 (majorRadius + minorRadius * cv) * su};
    };
    for (u32 i = 0; i < major; ++i)
    {
        const f32 u0 = static_cast<f32>(i) / static_cast<f32>(major) * 2.0f * kPi;
        const f32 u1 = static_cast<f32>(i + 1) / static_cast<f32>(major) * 2.0f * kPi;
        for (u32 j = 0; j < minor; ++j)
        {
            const f32 v0 = static_cast<f32>(j) / static_cast<f32>(minor) * 2.0f * kPi;
            const f32 v1 = static_cast<f32>(j + 1) / static_cast<f32>(minor) * 2.0f * kPi;
            Vec3 p00{};
            Vec3 p01{};
            Vec3 p10{};
            Vec3 p11{};
            Vec3 n00{};
            Vec3 n01{};
            Vec3 n10{};
            Vec3 n11{};
            point(u0, v0, p00, n00);
            point(u0, v1, p01, n01);
            point(u1, v0, p10, n10);
            point(u1, v1, p11, n11);
            if (!PushSmoothTriangle(mesh, p00, n00, p10, n10, p11, n11, color) ||
                !PushSmoothTriangle(mesh, p00, n00, p11, n11, p01, n01, color))
            {
                return false;
            }
        }
    }
    return true;
}

// A closed cylinder standing on `baseCenter`.
bool AddCylinder(DynamicArray<VertexInput>& mesh, const Vec3& baseCenter, f32 radius, f32 height,
                 const Color& color, u32 segments = 56)
{
    const Vec3 up{0.0f, 1.0f, 0.0f};
    for (u32 segment = 0; segment < segments; ++segment)
    {
        const f32 a0 = static_cast<f32>(segment) / static_cast<f32>(segments) * 2.0f * kPi;
        const f32 a1 = static_cast<f32>(segment + 1) / static_cast<f32>(segments) * 2.0f * kPi;
        const Vec3 n0{std::cos(a0), 0.0f, std::sin(a0)};
        const Vec3 n1{std::cos(a1), 0.0f, std::sin(a1)};
        const Vec3 b0 = baseCenter + n0 * radius;
        const Vec3 b1 = baseCenter + n1 * radius;
        const Vec3 t0 = b0 + up * height;
        const Vec3 t1 = b1 + up * height;
        if (!PushSmoothTriangle(mesh, b0, n0, b1, n1, t1, n1, color) ||
            !PushSmoothTriangle(mesh, b0, n0, t1, n1, t0, n0, color) ||
            !PushTriangle(mesh, baseCenter + up * height, t0, t1, up, color) ||
            !PushTriangle(mesh, baseCenter, b1, b0, Vec3{0.0f, -1.0f, 0.0f}, color))
        {
            return false;
        }
    }
    return true;
}

// A cone standing on `baseCenter`, with the side normals of the true surface.
bool AddCone(DynamicArray<VertexInput>& mesh, const Vec3& baseCenter, f32 radius, f32 height,
             const Color& color, u32 segments = 56)
{
    const Vec3 apex = baseCenter + Vec3{0.0f, height, 0.0f};
    const f32 slant = std::sqrt(radius * radius + height * height);
    const f32 ny = radius / slant;
    const f32 nr = height / slant;
    for (u32 segment = 0; segment < segments; ++segment)
    {
        const f32 a0 = static_cast<f32>(segment) / static_cast<f32>(segments) * 2.0f * kPi;
        const f32 a1 = static_cast<f32>(segment + 1) / static_cast<f32>(segments) * 2.0f * kPi;
        const f32 am = (a0 + a1) * 0.5f;
        const Vec3 n0{nr * std::cos(a0), ny, nr * std::sin(a0)};
        const Vec3 n1{nr * std::cos(a1), ny, nr * std::sin(a1)};
        const Vec3 nm{nr * std::cos(am), ny, nr * std::sin(am)};
        const Vec3 b0 = baseCenter + Vec3{radius * std::cos(a0), 0.0f, radius * std::sin(a0)};
        const Vec3 b1 = baseCenter + Vec3{radius * std::cos(a1), 0.0f, radius * std::sin(a1)};
        if (!PushSmoothTriangle(mesh, b0, n0, b1, n1, apex, nm, color) ||
            !PushTriangle(mesh, baseCenter, b1, b0, Vec3{0.0f, -1.0f, 0.0f}, color))
        {
            return false;
        }
    }
    return true;
}

// -----------------------------------------------------------------------
// Textures and the room
// -----------------------------------------------------------------------

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

constexpr f32 kRoomExtent = 6.0f;
constexpr f32 kRoomHeight = 4.6f;

bool BuildFloor(DynamicArray<VertexInput>& mesh)
{
    constexpr f32 e = kRoomExtent;
    constexpr f32 kRepeats = 12.0f;
    return PushTexturedQuad(mesh, Vec3{-e, 0.0f, -e}, Vec2{0.0f, 0.0f}, Vec3{-e, 0.0f, e},
                            Vec2{0.0f, kRepeats}, Vec3{e, 0.0f, e}, Vec2{kRepeats, kRepeats},
                            Vec3{e, 0.0f, -e}, Vec2{kRepeats, 0.0f}, Vec3{0.0f, 1.0f, 0.0f});
}

// Back wall and ceiling frame the room while leaving the front open to camera.
bool BuildShell(DynamicArray<VertexInput>& mesh)
{
    const Color backWall{0.235f, 0.250f, 0.285f, 1.0f};
    const Color ceiling{0.300f, 0.300f, 0.310f, 1.0f};
    constexpr f32 e = kRoomExtent;
    constexpr f32 h = kRoomHeight;
    return PushQuad(mesh, Vec3{-e, 0.0f, -e}, Vec3{e, 0.0f, -e}, Vec3{e, h, -e}, Vec3{-e, h, -e},
                    Vec3{0.0f, 0.0f, 1.0f}, backWall) &&
           PushQuad(mesh, Vec3{-e, h, e}, Vec3{-e, h, -e}, Vec3{e, h, -e}, Vec3{e, h, e},
                    Vec3{0.0f, -1.0f, 0.0f}, ceiling);
}

// The two coloured walls, each its own instance so the bake bleeds their
// colour onto whatever stands near them -- the clearest read on whether the
// irradiance volume is doing its job.
bool BuildLeftWall(DynamicArray<VertexInput>& mesh)
{
    constexpr f32 e = kRoomExtent;
    constexpr f32 h = kRoomHeight;
    return PushQuad(mesh, Vec3{-e, 0.0f, -e}, Vec3{-e, h, -e}, Vec3{-e, h, e}, Vec3{-e, 0.0f, e},
                    Vec3{1.0f, 0.0f, 0.0f}, Color{0.520f, 0.085f, 0.075f, 1.0f});
}

bool BuildRightWall(DynamicArray<VertexInput>& mesh)
{
    constexpr f32 e = kRoomExtent;
    constexpr f32 h = kRoomHeight;
    return PushQuad(mesh, Vec3{e, 0.0f, e}, Vec3{e, h, e}, Vec3{e, h, -e}, Vec3{e, 0.0f, -e},
                    Vec3{-1.0f, 0.0f, 0.0f}, Color{0.075f, 0.430f, 0.320f, 1.0f});
}

// Three emissive fixtures. Their vertex colour is far above one, which gives
// the tone mapper a shoulder to roll off, bloom something to scatter and the
// metals a highlight to reflect.
bool BuildFixtures(DynamicArray<VertexInput>& mesh)
{
    constexpr f32 h = kRoomHeight;
    return AddBox(mesh, Vec3{-1.9f, h - 0.06f, 0.9f}, Vec3{1.5f, 0.05f, 0.30f},
                  Color{26.0f, 24.0f, 20.0f, 1.0f}) &&
           AddBox(mesh, Vec3{2.1f, h - 0.06f, -1.4f}, Vec3{1.2f, 0.05f, 0.26f},
                  Color{18.0f, 19.0f, 24.0f, 1.0f}) &&
           AddBox(mesh, Vec3{0.0f, 3.25f, -5.9f}, Vec3{3.4f, 0.09f, 0.06f},
                  Color{22.0f, 13.0f, 6.0f, 1.0f});
}

// -----------------------------------------------------------------------
// Scene description
// -----------------------------------------------------------------------

constexpr u32 kSphereCount = 5;
constexpr f32 kSphereRadius = 0.43f;
constexpr u32 kGroupCount = 20;

constexpr u32 kGroupFloor = 0;
constexpr u32 kGroupShell = 1;
constexpr u32 kGroupLeftWall = 2;
constexpr u32 kGroupRightWall = 3;
constexpr u32 kGroupFixtures = 4;
constexpr u32 kGroupPedestal = 5;
constexpr u32 kGroupSweep = 6;  // 6..10
constexpr u32 kGroupChrome = 11;
constexpr u32 kGroupTorus = 12;
constexpr u32 kGroupCylinder = 13;
constexpr u32 kGroupCone = 14;
constexpr u32 kGroupPlastic = 15;
constexpr u32 kGroupRubber = 16;
constexpr u32 kGroupBlueBox = 17;
constexpr u32 kGroupOrangeBox = 18;
constexpr u32 kGroupGreenBox = 19;

// One instance: its forward material, where its ambient comes from, and the
// Lambertian stand-in the bake uses for it.
struct Group
{
    PbrMaterial material{};
    const Texture* texture = nullptr;
    bool useProbes = false;
    bool castsShadow = true;
    GiSurfaceMaterial transport{};
};

struct SceneLighting
{
    const PbrDirectionalLight* sun = nullptr;
    const PbrDirectionalLight* fill = nullptr;
    const PbrPointLight* lamps = nullptr;
    u32 lampCount = 0;
    const ReflectionProbeSet* probes = nullptr;
    const EnvironmentLighting* environment = nullptr;
    // Phase 31. Null selects the image-based diffuse term instead.
    const IrradianceVolumeView* irradianceVolume = nullptr;
    // Ray-traced lamp shadows against the BVH the bake used.
    const TopLevelAccelerationStructure* geometry = nullptr;
};

struct SceneUniforms
{
    TransformShaderData transform{};
    PbrLightingData lighting{};
    const TopLevelAccelerationStructure* geometry = nullptr;
    const PbrPointLight* lamps = nullptr;
    u32 lampCount = 0;
};

void SceneVertex(const VertexInput& input, const void* userData, VertexStageOutput& output) noexcept
{
    TransformVertexShader(input, &static_cast<const SceneUniforms*>(userData)->transform, output);
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

// Cook-Torrance through the shared PBR shader, then the lamps on top. The
// lamps are added here rather than through PbrLightingData because each one
// needs a visibility ray, which is what makes their shadows ray-traced rather
// than shadow-mapped.
Color ScenePixel(const Varyings& varyings, const void* userData) noexcept
{
    const auto& uniforms = *static_cast<const SceneUniforms*>(userData);
    Color color = PbrPixelShader(varyings, &uniforms.lighting);
    if (uniforms.lampCount == 0 || uniforms.lamps == nullptr)
    {
        return color;
    }
    const Vec3 normal = cyber::math::Normalize(varyings.normal);
    const Vec3 position = varyings.worldPosition;
    const Vec3 view = cyber::math::Normalize(uniforms.lighting.eyePosition - position);
    for (u32 index = 0; index < uniforms.lampCount; ++index)
    {
        const PbrPointLight& lamp = uniforms.lamps[index];
        const Vec3 toLight = lamp.position - position;
        const f32 distance = cyber::math::Length(toLight);
        if (distance <= 1.0e-4f)
        {
            continue;
        }
        const Vec3 lightDirection = toLight * (1.0f / distance);
        const f32 ndotl = cyber::math::Dot(normal, lightDirection);
        const f32 fade = SmoothRangeFade(distance, lamp.range);
        if (ndotl <= 0.0f || fade <= 0.0f)
        {
            continue;
        }
        if (uniforms.geometry != nullptr &&
            !TraceVisibility(*uniforms.geometry, OffsetRayOrigin(position, normal), lamp.position))
        {
            continue;
        }
        const Color brdf =
            EvaluateGgxBrdf(normal, view, lightDirection, uniforms.lighting.material);
        const f32 scale = ndotl * fade / (distance * distance);
        color.r += brdf.r * lamp.intensity.x * scale;
        color.g += brdf.g * lamp.intensity.y * scale;
        color.b += brdf.b * lamp.intensity.z * scale;
    }
    return color;
}

Color NormalPixel(const Varyings& varyings, const void* /*userData*/) noexcept
{
    const Vec3 normal = cyber::math::Normalize(varyings.normal);
    return Color{normal.x * 0.5f + 0.5f, normal.y * 0.5f + 0.5f, normal.z * 0.5f + 0.5f, 1.0f};
}

void FillLighting(PbrLightingData& lighting, const SceneLighting& scene, const Group& group,
                  const Vec3& eye, const Vec3& forward)
{
    lighting.eyePosition = eye;
    lighting.cameraForward = forward;
    lighting.ambientIrradiance = Vec3{0.040f, 0.046f, 0.060f};
    lighting.material = group.material;
    lighting.baseColorTexture = group.texture;
    lighting.sampler.addressU = AddressMode::Repeat;
    lighting.sampler.addressV = AddressMode::Repeat;
    if (group.useProbes && scene.probes != nullptr)
    {
        lighting.reflectionProbes = *scene.probes;
    }
    else
    {
        lighting.environment = scene.environment;
    }
    lighting.irradianceVolume = scene.irradianceVolume;
    lighting.directional[0] = *scene.sun;
    lighting.directional[1] = *scene.fill;
    lighting.directionalCount = 2;
}

struct Scene
{
    const DynamicArray<VertexInput>* meshes = nullptr;
    const Group* groups = nullptr;
};

[[nodiscard]] bool DrawScene(const Scene& scene, const SceneLighting& lighting, const Vec3& eye,
                             const Vec3& target, Framebuffer& color, DepthBuffer& depth,
                             HeapAllocator& allocator)
{
    const Mat4 view = Mat4::LookAt(eye, target, Vec3{0.0f, 1.0f, 0.0f});
    const Mat4 projection = Mat4::PerspectiveReverseZ(
        kFieldOfView, static_cast<f32>(kWidth) / static_cast<f32>(kHeight), kNearPlane, kFarPlane);
    const Vec3 forward = cyber::math::Normalize(target - eye);

    for (u32 index = 0; index < kGroupCount; ++index)
    {
        SceneUniforms uniforms{};
        uniforms.transform.modelViewProjection = projection * view;
        uniforms.transform.modelToWorld = Mat4::Identity();
        uniforms.transform.normalToWorld = Mat4::Identity();
        uniforms.transform.eyePosition = eye;
        FillLighting(uniforms.lighting, lighting, scene.groups[index], eye, forward);
        uniforms.geometry = lighting.geometry;
        uniforms.lamps = lighting.lamps;
        uniforms.lampCount = lighting.lampCount;

        ShaderDesc shader{};
        shader.pipeline.view = view;
        shader.pipeline.projection = projection;
        shader.pipeline.viewport = Viewport{0.0f, 0.0f, static_cast<f32>(color.Width()),
                                            static_cast<f32>(color.Height())};
        shader.pipeline.cullMode = CullMode::None;
        shader.pipeline.useJobs = true;
        shader.vertexShader = &SceneVertex;
        shader.pixelShader = &ScenePixel;
        shader.userData = &uniforms;
        const DynamicArray<VertexInput>& mesh = scene.meshes[index];
        if (!DrawShadedTriangles(shader, Span<const VertexInput>{mesh.Data(), mesh.Size()}, color,
                                 depth, nullptr, allocator)
                 .IsOk())
        {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool DrawNormalsAndDepth(const Scene& scene, const Vec3& eye, const Vec3& target,
                                       Framebuffer& normals, DepthBuffer& depth,
                                       HeapAllocator& allocator)
{
    const Mat4 view = Mat4::LookAt(eye, target, Vec3{0.0f, 1.0f, 0.0f});
    const Mat4 projection = Mat4::PerspectiveReverseZ(
        kFieldOfView, static_cast<f32>(kWidth) / static_cast<f32>(kHeight), kNearPlane, kFarPlane);

    normals.Clear(Color{0.5f, 0.5f, 0.0f, 1.0f});
    depth.Clear();
    for (u32 index = 0; index < kGroupCount; ++index)
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
        const DynamicArray<VertexInput>& mesh = scene.meshes[index];
        if (!DrawShadedTriangles(shader, Span<const VertexInput>{mesh.Data(), mesh.Size()},
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
    Scene scene{};
    const SceneLighting* lighting = nullptr;
    HeapAllocator* allocator = nullptr;
};

cyber::core::Status<cyber::core::Error> CaptureFace(const ReflectionProbeCaptureView& view,
                                                    Framebuffer& color, DepthBuffer& depth,
                                                    void* userData) noexcept
{
    auto& context = *static_cast<CaptureContext*>(userData);
    const Vec3 forward = CubeDirectionFromFaceUv(static_cast<u32>(view.face), 0.5f, 0.5f);
    for (u32 index = 0; index < kGroupCount; ++index)
    {
        const Group& group = context.scene.groups[index];
        // A probe cannot reflect the surfaces that read it.
        if (group.useProbes)
        {
            continue;
        }
        SceneUniforms uniforms{};
        uniforms.transform.modelViewProjection = view.projection * view.view;
        uniforms.transform.modelToWorld = Mat4::Identity();
        uniforms.transform.normalToWorld = Mat4::Identity();
        uniforms.transform.eyePosition = view.position;
        FillLighting(uniforms.lighting, *context.lighting, group, view.position, forward);
        uniforms.lighting.environment = nullptr;
        uniforms.lighting.irradianceVolume = nullptr;
        uniforms.lighting.ambientIrradiance = Vec3{0.040f, 0.046f, 0.060f};
        // The capture carries the lamps as ordinary unshadowed point lights: a
        // cube face is not worth a visibility ray per texel.
        const u32 lampCount = context.lighting->lampCount < kMaxPbrPointLights
                                  ? context.lighting->lampCount
                                  : kMaxPbrPointLights;
        for (u32 lamp = 0; lamp < lampCount; ++lamp)
        {
            uniforms.lighting.points[lamp] = context.lighting->lamps[lamp];
        }
        uniforms.lighting.pointCount = lampCount;

        ShaderDesc shader{};
        shader.pipeline.view = view.view;
        shader.pipeline.projection = view.projection;
        shader.pipeline.viewport = Viewport{0.0f, 0.0f, static_cast<f32>(color.Width()),
                                            static_cast<f32>(color.Height())};
        shader.pipeline.cullMode = CullMode::None;
        shader.pipeline.useJobs = true;
        shader.vertexShader = &SceneVertex;
        shader.pixelShader = &ScenePixel;
        shader.userData = &uniforms;
        const DynamicArray<VertexInput>& mesh = context.scene.meshes[index];
        if (!DrawShadedTriangles(shader, Span<const VertexInput>{mesh.Data(), mesh.Size()}, color,
                                 depth, nullptr, *context.allocator)
                 .IsOk())
        {
            return cyber::core::Status<cyber::core::Error>::Err(
                cyber::core::Error{cyber::core::ErrorCode::Internal,
                                   "failed to render a reflection probe face"});
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

    // ---------------------------------------------------------------- meshes
    DynamicArray<VertexInput> meshes[kGroupCount] = {
        DynamicArray<VertexInput>{heap}, DynamicArray<VertexInput>{heap},
        DynamicArray<VertexInput>{heap}, DynamicArray<VertexInput>{heap},
        DynamicArray<VertexInput>{heap}, DynamicArray<VertexInput>{heap},
        DynamicArray<VertexInput>{heap}, DynamicArray<VertexInput>{heap},
        DynamicArray<VertexInput>{heap}, DynamicArray<VertexInput>{heap},
        DynamicArray<VertexInput>{heap}, DynamicArray<VertexInput>{heap},
        DynamicArray<VertexInput>{heap}, DynamicArray<VertexInput>{heap},
        DynamicArray<VertexInput>{heap}, DynamicArray<VertexInput>{heap},
        DynamicArray<VertexInput>{heap}, DynamicArray<VertexInput>{heap},
        DynamicArray<VertexInput>{heap}, DynamicArray<VertexInput>{heap}};
    Group groups[kGroupCount]{};

    const Color kBareMetal{0.955f, 0.930f, 0.880f, 1.0f};
    bool built = BuildFloor(meshes[kGroupFloor]) && BuildShell(meshes[kGroupShell]) &&
                 BuildLeftWall(meshes[kGroupLeftWall]) && BuildRightWall(meshes[kGroupRightWall]) &&
                 BuildFixtures(meshes[kGroupFixtures]) &&
                 AddBox(meshes[kGroupPedestal], Vec3{0.0f, 0.195f, 0.2f},
                        Vec3{3.15f, 0.195f, 1.05f}, Color{0.125f, 0.140f, 0.165f, 1.0f});
    // The roughness sweep: five bare-metal spheres on the pedestal.
    for (u32 index = 0; index < kSphereCount && built; ++index)
    {
        const f32 x = -2.32f + static_cast<f32>(index) * 1.16f;
        built = AddSphere(meshes[kGroupSweep + index], Vec3{x, 0.39f + kSphereRadius, 0.2f},
                          kSphereRadius, kBareMetal);
    }
    built = built &&
            // A mirror sphere, the one pure reflection of the room in frame.
            AddSphere(meshes[kGroupChrome], Vec3{-0.15f, 0.95f, -3.3f}, 0.95f,
                      Color{0.97f, 0.98f, 1.0f, 1.0f}, 64, 44) &&
            // Gold torus, copper cylinder, porcelain cone.
            AddTorus(meshes[kGroupTorus], Vec3{3.25f, 0.78f, 2.05f}, 0.72f, 0.26f,
                     Color{1.000f, 0.766f, 0.336f, 1.0f}) &&
            AddCylinder(meshes[kGroupCylinder], Vec3{-3.70f, 0.0f, 1.35f}, 0.46f, 1.85f,
                        Color{0.955f, 0.637f, 0.538f, 1.0f}) &&
            AddCone(meshes[kGroupCone], Vec3{-2.25f, 0.0f, 3.55f}, 0.62f, 1.45f,
                    Color{0.930f, 0.925f, 0.900f, 1.0f}) &&
            // A glossy plastic sphere and a rubber one.
            AddSphere(meshes[kGroupPlastic], Vec3{2.35f, 0.52f, 4.05f}, 0.52f,
                      Color{0.640f, 0.085f, 0.095f, 1.0f}, 40, 28) &&
            AddSphere(meshes[kGroupRubber], Vec3{4.55f, 0.46f, -0.45f}, 0.46f,
                      Color{0.075f, 0.110f, 0.260f, 1.0f}, 36, 24) &&
            // Painted boxes at three depths, for depth of field to work with.
            AddBox(meshes[kGroupBlueBox], Vec3{-4.35f, 0.62f, -2.35f}, Vec3{0.60f, 0.62f, 0.60f},
                   Color{0.145f, 0.265f, 0.470f, 1.0f}, -0.26f) &&
            AddBox(meshes[kGroupOrangeBox], Vec3{4.45f, 0.46f, -2.60f}, Vec3{0.72f, 0.46f, 0.58f},
                   Color{0.470f, 0.165f, 0.105f, 1.0f}, 0.21f) &&
            AddBox(meshes[kGroupGreenBox], Vec3{-1.15f, 0.30f, 4.65f}, Vec3{0.52f, 0.30f, 0.52f},
                   Color{0.155f, 0.330f, 0.215f, 1.0f}, 0.14f);
    if (!built)
    {
        std::fprintf(stderr, "failed to build the scene\n");
        return 1;
    }

    // ------------------------------------------------------------- materials
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

    const auto setMaterial = [&groups](u32 index, const Color& base, f32 metallic, f32 roughness,
                                       bool useProbes, const Vec3& albedo)
    {
        groups[index].material.baseColor = base;
        groups[index].material.metallic = metallic;
        groups[index].material.roughness = roughness;
        groups[index].useProbes = useProbes;
        groups[index].transport.albedo = albedo;
    };

    const Color kWhite{1.0f, 1.0f, 1.0f, 1.0f};
    setMaterial(kGroupFloor, kWhite, 0.05f, 0.32f, false, Vec3{0.165f, 0.175f, 0.195f});
    groups[kGroupFloor].texture = &floorTexture;
    groups[kGroupFloor].castsShadow = false;
    setMaterial(kGroupShell, kWhite, 0.0f, 0.62f, false, Vec3{0.270f, 0.280f, 0.305f});
    groups[kGroupShell].castsShadow = false;
    setMaterial(kGroupLeftWall, kWhite, 0.0f, 0.58f, false, Vec3{0.560f, 0.090f, 0.080f});
    groups[kGroupLeftWall].castsShadow = false;
    setMaterial(kGroupRightWall, kWhite, 0.0f, 0.58f, false, Vec3{0.080f, 0.470f, 0.350f});
    groups[kGroupRightWall].castsShadow = false;
    setMaterial(kGroupFixtures, kWhite, 0.0f, 0.70f, false, Vec3{0.700f, 0.700f, 0.700f});
    // The fixtures are the emitters the bake gathers light from directly.
    groups[kGroupFixtures].transport.emission = Vec3{11.0f, 10.0f, 8.4f};
    groups[kGroupFixtures].castsShadow = false;
    setMaterial(kGroupPedestal, kWhite, 0.04f, 0.48f, false, Vec3{0.125f, 0.140f, 0.165f});
    for (u32 index = 0; index < kSphereCount; ++index)
    {
        // 0.04 to 0.46: a mirror at one end, a brushed finish at the other.
        setMaterial(kGroupSweep + index, kBareMetal, 1.0f,
                    0.04f + static_cast<f32>(index) * 0.105f, true, Vec3{0.60f, 0.58f, 0.55f});
    }
    setMaterial(kGroupChrome, Color{0.97f, 0.98f, 1.0f, 1.0f}, 1.0f, 0.045f, true,
                Vec3{0.70f, 0.71f, 0.73f});
    setMaterial(kGroupTorus, Color{1.000f, 0.766f, 0.336f, 1.0f}, 1.0f, 0.165f, true,
                Vec3{0.75f, 0.58f, 0.26f});
    setMaterial(kGroupCylinder, Color{0.955f, 0.637f, 0.538f, 1.0f}, 1.0f, 0.295f, true,
                Vec3{0.72f, 0.48f, 0.41f});
    setMaterial(kGroupCone, Color{0.930f, 0.925f, 0.900f, 1.0f}, 0.0f, 0.115f, false,
                Vec3{0.780f, 0.775f, 0.755f});
    setMaterial(kGroupPlastic, Color{0.640f, 0.085f, 0.095f, 1.0f}, 0.0f, 0.245f, false,
                Vec3{0.520f, 0.070f, 0.078f});
    setMaterial(kGroupRubber, Color{0.075f, 0.110f, 0.260f, 1.0f}, 0.0f, 0.880f, false,
                Vec3{0.075f, 0.110f, 0.260f});
    setMaterial(kGroupBlueBox, kWhite, 0.04f, 0.50f, false, Vec3{0.145f, 0.265f, 0.470f});
    setMaterial(kGroupOrangeBox, kWhite, 0.04f, 0.44f, false, Vec3{0.470f, 0.165f, 0.105f});
    setMaterial(kGroupGreenBox, kWhite, 0.04f, 0.52f, false, Vec3{0.155f, 0.330f, 0.215f});

    Scene scene{};
    scene.meshes = meshes;
    scene.groups = groups;

    usize triangles = 0;
    for (u32 index = 0; index < kGroupCount; ++index)
    {
        triangles += meshes[index].Size() / 3;
    }

    // ------------------------------------------- acceleration structures
    BottomLevelAccelerationStructure blases[kGroupCount];
    RayInstanceDesc instances[kGroupCount]{};
    GiSurfaceMaterial transport[kGroupCount]{};
    for (u32 index = 0; index < kGroupCount; ++index)
    {
        AccelerationStructureBuildDesc desc{};
        desc.vertices = Span<const VertexInput>{meshes[index].Data(), meshes[index].Size()};
        auto level = BottomLevelAccelerationStructure::Build(desc, heap);
        if (!level)
        {
            std::fprintf(stderr, "BVH build failed for group %u: %s\n", index,
                         level.Error().message);
            return 1;
        }
        blases[index] = std::move(level).Value();
        instances[index].blas = &blases[index];
        instances[index].instanceId = index;
        transport[index] = groups[index].transport;
    }
    auto topLevel =
        TopLevelAccelerationStructure::Build(Span<const RayInstanceDesc>{instances, kGroupCount},
                                             heap);
    if (!topLevel)
    {
        std::fprintf(stderr, "top-level BVH build failed: %s\n", topLevel.Error().message);
        return 1;
    }
    const TopLevelAccelerationStructure tlas = std::move(topLevel).Value();

    // ------------------------------------------------------------- shadows
    const Vec3 lightTravel = cyber::math::Normalize(Vec3{0.62f, -0.78f, -0.33f});
    const Vec3 lightCenter{0.0f, 1.3f, -0.4f};
    const Mat4 lightView =
        Mat4::LookAt(lightCenter - lightTravel * 24.0f, lightCenter, Vec3{0.0f, 1.0f, 0.0f});
    constexpr f32 kShadowExtent = 13.0f;
    const Mat4 lightProjection = Mat4::OrthographicReverseZ(-kShadowExtent, kShadowExtent,
                                                            -kShadowExtent, kShadowExtent, 0.1f,
                                                            60.0f);

    DynamicArray<VertexInput> casters{heap};
    for (u32 index = 0; index < kGroupCount; ++index)
    {
        if (!groups[index].castsShadow)
        {
            continue;
        }
        for (usize vertex = 0; vertex < meshes[index].Size(); ++vertex)
        {
            if (!casters.TryPushBack(meshes[index][vertex]))
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
    sun.shadow.nearPlane = 0.1f;
    sun.shadow.farPlane = 60.0f;
    sun.shadow.usePcss = true;
    sun.shadow.softFilter = true;
    sun.shadow.filterRadius = 8;
    sun.shadow.perspectiveProjection = false;
    sun.shadow.worldUnitsPerTexel = (2.0f * kShadowExtent) / static_cast<f32>(kShadowMapSize);
    sun.shadow.sourceAngularRadiusRadians = 0.013f;

    PbrDirectionalLight fill{};
    fill.direction = cyber::math::Normalize(Vec3{0.52f, -0.42f, 0.74f});
    fill.radiance = Vec3{0.14f, 0.18f, 0.27f};

    // The three fixtures as actual sources, for the forward path and for the
    // bake: the same positions and the same numbers in both.
    constexpr u32 kLampCount = 3;
    PbrPointLight lamps[kLampCount]{};
    lamps[0].position = Vec3{-1.9f, 4.35f, 0.9f};
    lamps[0].intensity = Vec3{6.5f, 5.9f, 4.7f};
    lamps[0].range = 14.0f;
    lamps[1].position = Vec3{2.1f, 4.35f, -1.4f};
    lamps[1].intensity = Vec3{3.6f, 4.0f, 5.0f};
    lamps[1].range = 13.0f;
    lamps[2].position = Vec3{0.0f, 3.25f, -5.6f};
    lamps[2].intensity = Vec3{4.5f, 2.7f, 1.3f};
    lamps[2].range = 10.0f;

    cyber::jobs::JobSystem::Desc jobDesc{};
    auto jobsResult = cyber::jobs::JobSystem::Create(jobDesc, heap);
    cyber::jobs::JobSystem* jobs = jobsResult.IsOk() ? &jobsResult.Value() : nullptr;

    // ------------------------------------------------------------------ IBL
    SceneLighting lighting{};
    lighting.sun = &sun;
    lighting.fill = &fill;
    lighting.lamps = lamps;
    lighting.lampCount = kLampCount;

    std::printf("capturing the reflection probe (6 faces at %dx%d)\n", kProbeFaceSize,
                kProbeFaceSize);
    ReflectionProbeCaptureDesc captureDesc{};
    captureDesc.position = kProbeCenter;
    captureDesc.faceSize = kProbeFaceSize;
    captureDesc.supersampleFactor = kDefaultSupersampleFactor;
    captureDesc.resolveFilter = kDefaultResolveFilter;
    captureDesc.nearPlane = 0.06f;
    captureDesc.farPlane = 40.0f;
    CaptureContext captureContext{scene, &lighting, &heap};
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

    // ------------------------------------------------- phase 31: the bake
    GiDirectionalLight giSun{};
    giSun.direction = sun.direction;
    giSun.irradiance = sun.radiance;
    GiPointLight giLamps[kLampCount]{};
    for (u32 index = 0; index < kLampCount; ++index)
    {
        giLamps[index].position = lamps[index].position;
        giLamps[index].intensity = lamps[index].intensity;
        giLamps[index].range = lamps[index].range;
    }

    GiScene giScene{};
    giScene.geometry = &tlas;
    giScene.materials = Span<const GiSurfaceMaterial>{transport, kGroupCount};
    giScene.directionalLights = Span<const GiDirectionalLight>{&giSun, 1};
    giScene.pointLights = Span<const GiPointLight>{giLamps, kLampCount};
    giScene.skyRadiance = Vec3{0.012f, 0.016f, 0.026f};

    IrradianceVolumeDesc volumeDesc{};
    volumeDesc.boundsMin = Vec3{-5.6f, 0.15f, -5.6f};
    volumeDesc.boundsMax = Vec3{5.6f, 4.3f, 5.6f};
    volumeDesc.probeCountX = 13;
    volumeDesc.probeCountY = 7;
    volumeDesc.probeCountZ = 13;
    auto volumeCreated = IrradianceVolume::Create(volumeDesc, heap);
    if (!volumeCreated)
    {
        std::fprintf(stderr, "irradiance volume allocation failed: %s\n",
                     volumeCreated.Error().message);
        return 1;
    }
    IrradianceVolume volume = std::move(volumeCreated).Value();

    IrradianceBakeDesc bake{};
    bake.raysPerProbe = 768;
    bake.path.maxBounces = 3;
    IrradianceBakeStats bakeStats{};
    std::printf("baking the irradiance volume (%u x %u x %u probes, %u rays each)\n",
                volumeDesc.probeCountX, volumeDesc.probeCountY, volumeDesc.probeCountZ,
                bake.raysPerProbe);
    const auto bakeStart = std::chrono::steady_clock::now();
    if (!BakeIrradianceVolume(giScene, bake, volume, &bakeStats, jobs))
    {
        std::fprintf(stderr, "the irradiance bake failed\n");
        return 1;
    }
    const f64 bakeMs =
        std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - bakeStart)
            .count();

    IrradianceVolumeView volumeView{};
    volumeView.volume = &volume;
    volumeView.intensity = 1.0f;

    // ---------------------------------------------------------- the frames
    AntiAliasTargetDesc antialiasDesc{};
    antialiasDesc.width = kWidth;
    antialiasDesc.height = kHeight;
    antialiasDesc.quality = AntiAliasQuality::Balanced;
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

    Framebuffer normals;
    Framebuffer resolved;
    if (!MakeImage(heap, kWidth, kHeight, normals) || !MakeImage(heap, kWidth, kHeight, resolved))
    {
        std::fprintf(stderr, "failed to allocate the render targets\n");
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
    if (!DrawNormalsAndDepth(scene, kEye, kTarget, normals, postDepth, heap))
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
    settings.enableMotionBlur = false;
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
    settings.depthOfField.focusDistance = 11.5f;
    settings.depthOfField.focalLengthMm = 50.0f;
    settings.depthOfField.apertureFStop = 4.0f;
    settings.depthOfField.maxCocPixels = 8.0f;
    settings.depthOfField.sampleCount = 32;
    settings.bloom.threshold = 1.0f;
    settings.bloom.intensity = 0.075f;
    settings.bloom.scatter = 0.72f;
    settings.toneMap.whitePoint = 4.0f;

    PostProcessInputs inputs{};
    inputs.color = &resolved;
    inputs.depth = &postDepth;
    inputs.normals = &normals;

    Framebuffer iblFrame;
    Framebuffer giFrame;
    Framebuffer composite;
    if (!MakeImage(heap, kWidth, kHeight, iblFrame) || !MakeImage(heap, kWidth, kHeight, giFrame) ||
        !MakeImage(heap, kWidth * 2, kHeight, composite))
    {
        std::fprintf(stderr, "failed to allocate the output images\n");
        return 1;
    }

    ResolveDesc resolveDesc{};
    resolveDesc.factor = antialias.Factor();
    resolveDesc.filter = kDefaultResolveFilter;
    resolveDesc.useJobs = true;

    // Pass one: the phase 29/30 frame -- image-based diffuse, unshadowed lamps.
    std::printf("shading %dx%d at %dx%d samples per pixel (image-based ambient)\n", kWidth, kHeight,
                antialias.Factor(), antialias.Factor());
    lighting.irradianceVolume = nullptr;
    lighting.geometry = nullptr;
    antialias.Clear(Color{0.012f, 0.016f, 0.026f, 1.0f});
    const auto iblStart = std::chrono::steady_clock::now();
    if (!DrawScene(scene, lighting, kEye, kTarget, antialias.Color(), antialias.Depth(), heap))
    {
        std::fprintf(stderr, "the image-based pass failed\n");
        return 1;
    }
    const f64 iblMs =
        std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - iblStart).count();
    if (!antialias.Resolve(resolved, nullptr, heap, resolveDesc))
    {
        std::fprintf(stderr, "the resolve failed\n");
        return 1;
    }
    if (!stack.Execute(inputs, iblFrame, settings, view, 1.0f / 60.0f, nullptr, nullptr).IsOk())
    {
        std::fprintf(stderr, "the post chain failed\n");
        return 1;
    }

    // Pass two: phase 31 -- baked irradiance for the diffuse term, and a
    // visibility ray per lamp per sample.
    std::printf("shading %dx%d again (baked irradiance volume, ray-traced lamp shadows)\n", kWidth,
                kHeight);
    lighting.irradianceVolume = &volumeView;
    lighting.geometry = &tlas;
    antialias.Clear(Color{0.012f, 0.016f, 0.026f, 1.0f});
    const auto giStart = std::chrono::steady_clock::now();
    if (!DrawScene(scene, lighting, kEye, kTarget, antialias.Color(), antialias.Depth(), heap))
    {
        std::fprintf(stderr, "the phase 31 pass failed\n");
        return 1;
    }
    const f64 giMs =
        std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - giStart).count();
    if (!antialias.Resolve(resolved, nullptr, heap, resolveDesc))
    {
        std::fprintf(stderr, "the resolve failed\n");
        return 1;
    }
    PostProcessTimings timings{};
    if (!stack.Execute(inputs, giFrame, settings, view, 1.0f / 60.0f, nullptr, &timings).IsOk())
    {
        std::fprintf(stderr, "the post chain failed\n");
        return 1;
    }

    for (i32 y = 0; y < kHeight; ++y)
    {
        for (i32 x = 0; x < kWidth; ++x)
        {
            composite.SetPixel(x, y, iblFrame.GetPixel(x, y));
            composite.SetPixel(x + kWidth, y, giFrame.GetPixel(x, y));
        }
    }

    // -------------------------------------------------------------- report
    std::printf("\n%zu triangles in %u instances, BVH %zu bytes, %dx%d shadow map\n", triangles,
                kGroupCount, tlas.DataSizeBytes(), kShadowMapSize, kShadowMapSize);
    std::printf("bake: %u probes (%u valid, %u inside), %u rays each, %u bounces, %llu rays, "
                "%.0f ms, %zu bytes\n",
                bakeStats.probeCount, bakeStats.validProbes, bakeStats.insideProbes,
                bake.raysPerProbe, bake.path.maxBounces,
                static_cast<unsigned long long>(bakeStats.path.rays), bakeMs,
                volume.DataSizeBytes());
    std::printf("render: image-based %.0f ms, phase 31 %.0f ms, exposure %+.2f EV\n", iblMs, giMs,
                static_cast<double>(timings.exposureEv));

    const bool ok = Write(directory, "gallery.ppm", giFrame, heap) &&
                    Write(directory, "gallery_compare.ppm", composite, heap) &&
                    Write(directory, "gallery_ibl.ppm", iblFrame, heap);
    if (jobs != nullptr)
    {
        jobs->Shutdown();
    }
    return ok ? 0 : 1;
}
