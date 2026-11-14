// Anti-aliasing showcase.
//
// Renders one scene built out of the shapes that alias worst -- thin rods, a
// long thin railing, curved silhouettes, a receding textured floor, and sharp
// specular highlights -- and resolves it through the renderer-wide
// supersample-and-reconstruct path in cyber/software/antialias.h.
//
// The default run writes one capture per quality preset and one per
// reconstruction filter, and prints two measurements per image so the
// difference is a number and not only a judgement:
//
//   hard steps   fraction of neighboring pixel pairs whose sRGB luminance
//                jumps by 48/255 or more. A staircase is made of hard steps,
//                so this falls as anti-aliasing improves.
//   gradations   distinct sRGB luminance values in the image. A resolved edge
//                is a ramp, so this rises.

#include <cyber/jobs/job_system.h>
#include <cyber/math/constants.h>
#include <cyber/math/mat4.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/platform/filesystem.h>
#include <cyber/software/antialias.h>
#include <cyber/software/camera.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/image_io.h>
#include <cyber/software/pbr.h>
#include <cyber/software/shader.h>
#include <cyber/software/shadow.h>
#include <cyber/software/texture.h>

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
using cyber::core::f64;
using cyber::core::i32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::usize;
using cyber::math::Color;
using cyber::math::Mat4;
using cyber::math::Vec2;
using cyber::math::Vec3;

namespace
{

constexpr i32 kDefaultWidth = 960;
constexpr i32 kDefaultHeight = 540;
constexpr f32 kDefaultExposure = -0.55f;
constexpr u32 kCascadeCount = 3;
constexpr u32 kCascadeResolution = 1024;
constexpr i32 kHardStepThreshold = 48;

struct SceneUniforms
{
    TransformShaderData transform;
    PbrLightingData lighting;
};

void SceneVertex(const VertexInput& input, const void* userData, VertexStageOutput& output) noexcept
{
    TransformVertexShader(input, &static_cast<const SceneUniforms*>(userData)->transform, output);
}

Color ScenePixel(const Varyings& varyings, const void* userData) noexcept
{
    return PbrPixelShader(varyings, &static_cast<const SceneUniforms*>(userData)->lighting);
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

using MeshData = DynamicArray<VertexInput>;

bool PushTriangle(MeshData& mesh, const VertexInput& a, const VertexInput& b, const VertexInput& c)
{
    return mesh.TryPushBack(a) && mesh.TryPushBack(b) && mesh.TryPushBack(c);
}

bool PushQuad(MeshData& mesh, const VertexInput& a, const VertexInput& b, const VertexInput& c,
              const VertexInput& d)
{
    return PushTriangle(mesh, a, b, c) && PushTriangle(mesh, a, c, d);
}

VertexInput MakeVertex(const Vec3& position, const Vec3& normal, const Vec2& uv,
                       const Vec3& tangent)
{
    VertexInput vertex{};
    vertex.position = position;
    vertex.normal = normal;
    vertex.uv = uv;
    vertex.tangent = tangent;
    vertex.tangentSign = 1.0f;
    vertex.color = Color{1.0f, 1.0f, 1.0f, 1.0f};
    return vertex;
}

// Unit-radius UV sphere centered on the origin.
bool BuildSphere(MeshData& mesh, u32 longitudes, u32 latitudes)
{
    const auto at = [&](u32 latitude, u32 longitude) noexcept
    {
        const f32 v = static_cast<f32>(latitude) / static_cast<f32>(latitudes);
        const f32 u = static_cast<f32>(longitude) / static_cast<f32>(longitudes);
        const f32 theta = v * cyber::math::kPi;
        const f32 phi = u * cyber::math::kTwoPi;
        const Vec3 normal{std::sin(theta) * std::cos(phi), std::cos(theta),
                          std::sin(theta) * std::sin(phi)};
        const Vec3 tangent{-std::sin(phi), 0.0f, std::cos(phi)};
        return MakeVertex(normal, normal, Vec2{u, v}, tangent);
    };

    for (u32 latitude = 0; latitude < latitudes; ++latitude)
    {
        for (u32 longitude = 0; longitude < longitudes; ++longitude)
        {
            if (!PushQuad(mesh, at(latitude, longitude), at(latitude + 1, longitude),
                          at(latitude + 1, longitude + 1), at(latitude, longitude + 1)))
            {
                return false;
            }
        }
    }
    return true;
}

// Unit cube spanning [-0.5, 0.5] on every axis.
bool BuildBox(MeshData& mesh)
{
    struct Face
    {
        Vec3 normal;
        Vec3 tangent;
        Vec3 bitangent;
    };
    const Face faces[6] = {
        Face{Vec3{0.0f, 0.0f, 1.0f}, Vec3{1.0f, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f}},
        Face{Vec3{0.0f, 0.0f, -1.0f}, Vec3{-1.0f, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f}},
        Face{Vec3{1.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, -1.0f}, Vec3{0.0f, 1.0f, 0.0f}},
        Face{Vec3{-1.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, 1.0f}, Vec3{0.0f, 1.0f, 0.0f}},
        Face{Vec3{0.0f, 1.0f, 0.0f}, Vec3{1.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, -1.0f}},
        Face{Vec3{0.0f, -1.0f, 0.0f}, Vec3{1.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, 1.0f}},
    };
    for (const Face& face : faces)
    {
        const Vec3 center = face.normal * 0.5f;
        const Vec3 right = face.tangent * 0.5f;
        const Vec3 up = face.bitangent * 0.5f;
        const VertexInput a =
            MakeVertex(center - right - up, face.normal, Vec2{0.0f, 1.0f}, face.tangent);
        const VertexInput b =
            MakeVertex(center + right - up, face.normal, Vec2{1.0f, 1.0f}, face.tangent);
        const VertexInput c =
            MakeVertex(center + right + up, face.normal, Vec2{1.0f, 0.0f}, face.tangent);
        const VertexInput d =
            MakeVertex(center - right + up, face.normal, Vec2{0.0f, 0.0f}, face.tangent);
        if (!PushQuad(mesh, a, b, c, d))
        {
            return false;
        }
    }
    return true;
}

// Torus around the +Y axis.
bool BuildTorus(MeshData& mesh, f32 majorRadius, f32 minorRadius, u32 majorSegments,
                u32 minorSegments)
{
    const auto at = [&](u32 major, u32 minor) noexcept
    {
        const f32 u = static_cast<f32>(major) / static_cast<f32>(majorSegments);
        const f32 v = static_cast<f32>(minor) / static_cast<f32>(minorSegments);
        const f32 phi = u * cyber::math::kTwoPi;
        const f32 theta = v * cyber::math::kTwoPi;
        const Vec3 ringCenter{std::cos(phi) * majorRadius, 0.0f, std::sin(phi) * majorRadius};
        const Vec3 outward{std::cos(phi), 0.0f, std::sin(phi)};
        const Vec3 normal = outward * std::cos(theta) + Vec3{0.0f, std::sin(theta), 0.0f};
        const Vec3 position = ringCenter + normal * minorRadius;
        const Vec3 tangent{-std::sin(phi), 0.0f, std::cos(phi)};
        return MakeVertex(position, normal, Vec2{u * 6.0f, v}, tangent);
    };

    for (u32 major = 0; major < majorSegments; ++major)
    {
        for (u32 minor = 0; minor < minorSegments; ++minor)
        {
            if (!PushQuad(mesh, at(major, minor), at(major, minor + 1), at(major + 1, minor + 1),
                          at(major + 1, minor)))
            {
                return false;
            }
        }
    }
    return true;
}

// Capped cylinder of unit radius and unit height, centered on the origin and
// extending along +/-Y.
bool BuildCylinder(MeshData& mesh, u32 segments)
{
    for (u32 segment = 0; segment < segments; ++segment)
    {
        const f32 u0 = static_cast<f32>(segment) / static_cast<f32>(segments);
        const f32 u1 = static_cast<f32>(segment + 1) / static_cast<f32>(segments);
        const f32 phi0 = u0 * cyber::math::kTwoPi;
        const f32 phi1 = u1 * cyber::math::kTwoPi;
        const Vec3 n0{std::cos(phi0), 0.0f, std::sin(phi0)};
        const Vec3 n1{std::cos(phi1), 0.0f, std::sin(phi1)};
        const Vec3 t0{-std::sin(phi0), 0.0f, std::cos(phi0)};
        const Vec3 t1{-std::sin(phi1), 0.0f, std::cos(phi1)};

        const VertexInput a = MakeVertex(n0 + Vec3{0.0f, -0.5f, 0.0f}, n0, Vec2{u0, 1.0f}, t0);
        const VertexInput b = MakeVertex(n1 + Vec3{0.0f, -0.5f, 0.0f}, n1, Vec2{u1, 1.0f}, t1);
        const VertexInput c = MakeVertex(n1 + Vec3{0.0f, 0.5f, 0.0f}, n1, Vec2{u1, 0.0f}, t1);
        const VertexInput d = MakeVertex(n0 + Vec3{0.0f, 0.5f, 0.0f}, n0, Vec2{u0, 0.0f}, t0);
        if (!PushQuad(mesh, a, b, c, d))
        {
            return false;
        }

        const Vec3 up{0.0f, 1.0f, 0.0f};
        const Vec3 down{0.0f, -1.0f, 0.0f};
        const VertexInput topCenter =
            MakeVertex(Vec3{0.0f, 0.5f, 0.0f}, up, Vec2{0.5f, 0.5f}, Vec3{1.0f, 0.0f, 0.0f});
        const VertexInput bottomCenter =
            MakeVertex(Vec3{0.0f, -0.5f, 0.0f}, down, Vec2{0.5f, 0.5f}, Vec3{1.0f, 0.0f, 0.0f});
        if (!PushTriangle(
                mesh, topCenter,
                MakeVertex(n0 + Vec3{0.0f, 0.5f, 0.0f}, up, Vec2{u0, 0.0f}, Vec3{1.0f, 0.0f, 0.0f}),
                MakeVertex(n1 + Vec3{0.0f, 0.5f, 0.0f}, up, Vec2{u1, 0.0f},
                           Vec3{1.0f, 0.0f, 0.0f})))
        {
            return false;
        }
        if (!PushTriangle(mesh, bottomCenter,
                          MakeVertex(n1 + Vec3{0.0f, -0.5f, 0.0f}, down, Vec2{u1, 1.0f},
                                     Vec3{1.0f, 0.0f, 0.0f}),
                          MakeVertex(n0 + Vec3{0.0f, -0.5f, 0.0f}, down, Vec2{u0, 1.0f},
                                     Vec3{1.0f, 0.0f, 0.0f})))
        {
            return false;
        }
    }
    return true;
}

// Unit-size floor in the XZ plane at y = 0, with `tiles` UV repeats per edge.
//
// The plane is subdivided rather than drawn as one quad on purpose: a ground
// plane large enough to reach the horizon also extends behind the camera, and
// the near-plane clip of one huge quad lands a vertex far outside the
// rasterizer's screen guard band, which drops the whole triangle. Small cells
// clip to small screen triangles.
bool BuildFloor(MeshData& mesh, f32 tiles, u32 subdivisions)
{
    const Vec3 normal{0.0f, 1.0f, 0.0f};
    const Vec3 tangent{1.0f, 0.0f, 0.0f};
    const f32 step = 1.0f / static_cast<f32>(subdivisions);

    for (u32 row = 0; row < subdivisions; ++row)
    {
        for (u32 column = 0; column < subdivisions; ++column)
        {
            const f32 x0 = -0.5f + static_cast<f32>(column) * step;
            const f32 x1 = x0 + step;
            const f32 z0 = -0.5f + static_cast<f32>(row) * step;
            const f32 z1 = z0 + step;
            const f32 u0 = (x0 + 0.5f) * tiles;
            const f32 u1 = (x1 + 0.5f) * tiles;
            const f32 v0 = (z0 + 0.5f) * tiles;
            const f32 v1 = (z1 + 0.5f) * tiles;

            const VertexInput a = MakeVertex(Vec3{x0, 0.0f, z0}, normal, Vec2{u0, v0}, tangent);
            const VertexInput b = MakeVertex(Vec3{x0, 0.0f, z1}, normal, Vec2{u0, v1}, tangent);
            const VertexInput c = MakeVertex(Vec3{x1, 0.0f, z1}, normal, Vec2{u1, v1}, tangent);
            const VertexInput d = MakeVertex(Vec3{x1, 0.0f, z0}, normal, Vec2{u1, v0}, tangent);
            if (!PushQuad(mesh, a, b, c, d))
            {
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

bool MakeTexture(HeapAllocator& heap, Texture& texture, i32 size)
{
    TextureDesc desc{};
    desc.width = size;
    desc.height = size;
    desc.format = TextureFormat::RGBA32F;
    desc.generateMips = true;
    auto created = Texture::Create(desc, heap);
    if (!created)
    {
        return false;
    }
    texture = std::move(created).Value();
    return true;
}

// Tiled stone floor: a checker with a grout line and a little per-tile value
// variation, which is what makes minification aliasing visible at distance.
bool MakeFloorTexture(HeapAllocator& heap, Texture& texture)
{
    constexpr i32 kSize = 128;
    if (!MakeTexture(heap, texture, kSize))
    {
        return false;
    }
    for (i32 y = 0; y < kSize; ++y)
    {
        for (i32 x = 0; x < kSize; ++x)
        {
            const i32 tileX = x / 64;
            const i32 tileY = y / 64;
            const bool light = ((tileX + tileY) % 2) == 0;
            const i32 withinX = x % 64;
            const i32 withinY = y % 64;
            const bool grout = withinX < 3 || withinY < 3 || withinX > 60 || withinY > 60;

            Color texel =
                light ? Color{0.62f, 0.60f, 0.56f, 1.0f} : Color{0.11f, 0.12f, 0.14f, 1.0f};
            if (grout)
            {
                texel = Color{0.045f, 0.045f, 0.05f, 1.0f};
            }
            else
            {
                const f32 noise = 0.94f + 0.06f * std::sin(static_cast<f32>(x) * 0.7f +
                                                           static_cast<f32>(y) * 1.3f);
                texel.r *= noise;
                texel.g *= noise;
                texel.b *= noise;
            }
            (void)texture.SetTexel(0, x, y, texel);
        }
    }
    texture.GenerateMips();
    return true;
}

// Shallow dimples, so the floor has a tangent-space normal to light.
bool MakeFloorNormalTexture(HeapAllocator& heap, Texture& texture)
{
    constexpr i32 kSize = 64;
    if (!MakeTexture(heap, texture, kSize))
    {
        return false;
    }
    for (i32 y = 0; y < kSize; ++y)
    {
        for (i32 x = 0; x < kSize; ++x)
        {
            const f32 u = static_cast<f32>(x) / static_cast<f32>(kSize) * cyber::math::kTwoPi;
            const f32 v = static_cast<f32>(y) / static_cast<f32>(kSize) * cyber::math::kTwoPi;
            const f32 slopeX = 0.22f * std::cos(u * 2.0f) * std::sin(v * 2.0f);
            const f32 slopeY = 0.22f * std::sin(u * 2.0f) * std::cos(v * 2.0f);
            const Vec3 normal = cyber::math::Normalize(Vec3{slopeX, slopeY, 1.0f});
            (void)texture.SetTexel(0, x, y,
                                   Color{normal.x * 0.5f + 0.5f, normal.y * 0.5f + 0.5f,
                                         normal.z * 0.5f + 0.5f, 1.0f});
        }
    }
    texture.GenerateMips();
    return true;
}

// Flat 1x1 white base color for the untextured objects, so every draw takes
// the same shader path.
bool MakeWhiteTexture(HeapAllocator& heap, Texture& texture)
{
    if (!MakeTexture(heap, texture, 2))
    {
        return false;
    }
    for (i32 y = 0; y < 2; ++y)
    {
        for (i32 x = 0; x < 2; ++x)
        {
            (void)texture.SetTexel(0, x, y, Color{1.0f, 1.0f, 1.0f, 1.0f});
        }
    }
    texture.GenerateMips();
    return true;
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

struct Draw
{
    const MeshData* mesh = nullptr;
    Mat4 model = Mat4::Identity();
    PbrMaterial material{};
    const Texture* baseColor = nullptr;
    const Texture* normal = nullptr;
    f32 mipLevel = 0.0f;
    bool castsShadow = true;
};

struct Meshes
{
    MeshData sphere;
    MeshData box;
    MeshData torus;
    MeshData rod;
    MeshData floor;

    explicit Meshes(Allocator& allocator) noexcept
        : sphere(allocator), box(allocator), torus(allocator), rod(allocator), floor(allocator)
    {
    }
};

struct Textures
{
    Texture floorBaseColor;
    Texture floorNormal;
    Texture white;
};

[[nodiscard]] Mat4 NormalMatrix(const Mat4& model) noexcept
{
    Mat4 inverse = Mat4::Identity();
    if (!model.TryInverse(inverse))
    {
        return Mat4::Identity();
    }
    return inverse.Transpose();
}

bool BuildScene(const Meshes& meshes, const Textures& textures, DynamicArray<Draw>& draws)
{
    const auto push = [&](Draw draw) { return draws.TryPushBack(draw); };

    // Floor. A fixed mip level stands in for per-pixel texture LOD, which the
    // CPU pixel stage does not compute; supersampling is what keeps the
    // distant tiles from shimmering.
    {
        Draw floor{};
        floor.mesh = &meshes.floor;
        floor.model = Mat4::Scale(Vec3{60.0f, 1.0f, 60.0f});
        floor.material.baseColor = Color{1.0f, 1.0f, 1.0f, 1.0f};
        floor.material.metallic = 0.0f;
        floor.material.roughness = 0.42f;
        floor.material.normalScale = 0.8f;
        floor.baseColor = &textures.floorBaseColor;
        floor.normal = &textures.floorNormal;
        floor.mipLevel = 1.5f;
        floor.castsShadow = false;
        if (!push(floor))
        {
            return false;
        }
    }

    // Sphere row: roughness sweeps left to right, alternating dielectric and
    // metal, so both broad and razor-thin highlights are in frame.
    constexpr u32 kSphereCount = 5;
    for (u32 index = 0; index < kSphereCount; ++index)
    {
        const f32 t = static_cast<f32>(index) / static_cast<f32>(kSphereCount - 1);
        const f32 radius = 0.62f;
        Draw sphere{};
        sphere.mesh = &meshes.sphere;
        sphere.model =
            Mat4::Translation(Vec3{-3.3f + 1.65f * static_cast<f32>(index), radius, 0.1f}) *
            Mat4::Scale(radius);
        // Dielectrics on the left, metals on the right, roughness rising
        // across the row. The smoothest metal is deliberately not a mirror:
        // with no image-based lighting a perfect mirror has nothing to
        // reflect and resolves to black.
        const Color colors[kSphereCount] = {
            Color{0.72f, 0.21f, 0.17f, 1.0f}, Color{0.16f, 0.36f, 0.62f, 1.0f},
            Color{0.86f, 0.86f, 0.84f, 1.0f}, Color{0.95f, 0.78f, 0.38f, 1.0f},
            Color{0.90f, 0.55f, 0.37f, 1.0f},
        };
        sphere.material.metallic = index >= 3 ? 1.0f : 0.0f;
        sphere.material.roughness = 0.06f + 0.62f * t * t;
        sphere.material.baseColor = colors[index];
        sphere.baseColor = &textures.white;
        if (!push(sphere))
        {
            return false;
        }
    }

    // Rotated box, a hard-edged silhouette at an angle no pixel grid likes.
    {
        Draw box{};
        box.mesh = &meshes.box;
        box.model = Mat4::Translation(Vec3{-2.6f, 0.55f, -2.9f}) *
                    Mat4::RotationY(27.0f * cyber::math::kDegreesToRadians) * Mat4::Scale(1.1f);
        box.material.baseColor = Color{0.70f, 0.24f, 0.20f, 1.0f};
        box.material.metallic = 0.0f;
        box.material.roughness = 0.33f;
        box.baseColor = &textures.white;
        if (!push(box))
        {
            return false;
        }
    }

    // Tilted torus: a curved silhouette plus a thin inner rim.
    {
        Draw torus{};
        torus.mesh = &meshes.torus;
        torus.model = Mat4::Translation(Vec3{2.3f, 0.95f, -3.1f}) *
                      Mat4::RotationX(68.0f * cyber::math::kDegreesToRadians) *
                      Mat4::RotationY(18.0f * cyber::math::kDegreesToRadians);
        torus.material.baseColor = Color{0.88f, 0.89f, 0.92f, 1.0f};
        torus.material.metallic = 1.0f;
        torus.material.roughness = 0.34f;
        torus.baseColor = &textures.white;
        if (!push(torus))
        {
            return false;
        }
    }

    // Rod fence. Each rod is about one pixel wide at this resolution, which is
    // where a single-sample rasterizer drops geometry outright.
    constexpr u32 kRodCount = 17;
    for (u32 index = 0; index < kRodCount; ++index)
    {
        const f32 x = -4.8f + 0.6f * static_cast<f32>(index);
        Draw rod{};
        rod.mesh = &meshes.rod;
        rod.model =
            Mat4::Translation(Vec3{x, 0.42f, 3.9f}) * Mat4::Scale(Vec3{0.021f, 0.84f, 0.021f});
        rod.material.baseColor = Color{0.78f, 0.78f, 0.80f, 1.0f};
        rod.material.metallic = 1.0f;
        rod.material.roughness = 0.28f;
        rod.baseColor = &textures.white;
        if (!push(rod))
        {
            return false;
        }
    }

    // Railing across the rod tops: a long, nearly horizontal thin edge.
    {
        Draw rail{};
        rail.mesh = &meshes.box;
        rail.model =
            Mat4::Translation(Vec3{0.0f, 0.86f, 3.9f}) * Mat4::Scale(Vec3{10.4f, 0.046f, 0.046f});
        rail.material.baseColor = Color{0.80f, 0.80f, 0.82f, 1.0f};
        rail.material.metallic = 1.0f;
        rail.material.roughness = 0.3f;
        rail.baseColor = &textures.white;
        if (!push(rail))
        {
            return false;
        }
    }

    return true;
}

// World-space copy of every shadow caster, which is what
// RenderDirectionalShadowMaps consumes.
bool BuildCasters(const DynamicArray<Draw>& draws, MeshData& casters)
{
    for (usize index = 0; index < draws.Size(); ++index)
    {
        const Draw& draw = draws.Data()[index];
        if (!draw.castsShadow || draw.mesh == nullptr)
        {
            continue;
        }
        const Mat4 normalMatrix = NormalMatrix(draw.model);
        for (usize vertex = 0; vertex < draw.mesh->Size(); ++vertex)
        {
            VertexInput world = draw.mesh->Data()[vertex];
            world.position = draw.model.TransformPoint(world.position);
            world.normal = normalMatrix.TransformVector(world.normal);
            if (!casters.TryPushBack(world))
            {
                return false;
            }
        }
    }
    return true;
}

struct Lights
{
    PbrDirectionalLight sun{};
    // Unshadowed cool bounce from the opposite side. Metals have no diffuse
    // term and this renderer has no image-based lighting, so a second
    // directional light is what keeps a metal sphere from reading as a black
    // disc with one highlight on it.
    PbrDirectionalLight sky{};
    PbrPointLight fill{};
    PbrSpotLight spot{};
};

void BuildLights(Lights& lights)
{
    lights.sun.direction = cyber::math::Normalize(Vec3{-0.42f, -0.86f, -0.30f});
    lights.sun.radiance = Vec3{2.6f, 2.45f, 2.2f};

    lights.sky.direction = cyber::math::Normalize(Vec3{0.55f, -0.42f, 0.72f});
    lights.sky.radiance = Vec3{0.42f, 0.52f, 0.72f};

    lights.fill.position = Vec3{-3.1f, 1.9f, 3.2f};
    lights.fill.intensity = Vec3{7.0f, 3.4f, 1.6f};
    lights.fill.range = 12.0f;

    lights.spot.position = Vec3{2.6f, 4.2f, 1.4f};
    lights.spot.direction = cyber::math::Normalize(Vec3{-0.25f, -1.0f, -0.35f});
    lights.spot.intensity = Vec3{14.0f, 15.0f, 18.0f};
    lights.spot.range = 14.0f;
    lights.spot.innerConeRadians = 0.28f;
    lights.spot.outerConeRadians = 0.46f;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

// Vertical sky gradient in linear radiance, written straight into the
// supersampled target before the first draw. It stands in for an environment
// map: the renderer has none, so the background would otherwise be flat.
void FillSkyGradient(Framebuffer& color) noexcept
{
    const Vec3 zenith{0.030f, 0.052f, 0.105f};
    const Vec3 horizon{0.135f, 0.155f, 0.205f};
    const f32 inverseHeight = 1.0f / static_cast<f32>(std::max(color.Height() - 1, 1));

    for (i32 y = 0; y < color.Height(); ++y)
    {
        const f32 t = static_cast<f32>(y) * inverseHeight;
        const f32 blend = t * t;
        const Color row{zenith.x + (horizon.x - zenith.x) * blend,
                        zenith.y + (horizon.y - zenith.y) * blend,
                        zenith.z + (horizon.z - zenith.z) * blend, 1.0f};
        for (i32 x = 0; x < color.Width(); ++x)
        {
            color.SetPixel(x, y, row);
        }
    }
}

bool RenderScene(AntiAliasTarget& target, const DynamicArray<Draw>& draws, const MeshData& casters,
                 const Camera& camera, const Lights& lights, DirectionalShadowCascadeSet& cascades,
                 Span<DepthBuffer> shadowMaps, cyber::jobs::JobSystem* jobSystem,
                 HeapAllocator& heap)
{
    const f32 aspect = static_cast<f32>(target.Width()) / static_cast<f32>(target.Height());
    const Mat4 view = CameraView(camera);
    const Mat4 projection = CameraProjection(camera, aspect);
    const Vec3 forward = CameraForward(camera);

    DirectionalShadowCascadeConfig shadowConfig{};
    shadowConfig.cascadeCount = kCascadeCount;
    shadowConfig.mapResolution = kCascadeResolution;
    shadowConfig.shadowDistance = 42.0f;
    shadowConfig.splitLambda = 0.75f;
    shadowConfig.filterRadius = 6;
    shadowConfig.constantDepthBias = 0.0008f;
    shadowConfig.slopeDepthBias = 0.003f;
    shadowConfig.normalOffsetBias = 0.02f;
    if (!BuildDirectionalShadowCascades(camera, aspect, lights.sun.direction, shadowConfig,
                                        cascades))
    {
        std::printf("failed to fit directional shadow cascades\n");
        return false;
    }
    if (!RenderDirectionalShadowMaps(cascades,
                                     Span<const VertexInput>(casters.Data(), casters.Size()),
                                     shadowMaps, heap, jobSystem, CullMode::None, true))
    {
        std::printf("failed to render directional shadow maps\n");
        return false;
    }

    FillSkyGradient(target.Color());
    target.Depth().Clear();

    for (usize index = 0; index < draws.Size(); ++index)
    {
        const Draw& draw = draws.Data()[index];
        if (draw.mesh == nullptr || draw.mesh->Size() == 0)
        {
            continue;
        }

        SceneUniforms uniforms{};
        uniforms.transform.modelToWorld = draw.model;
        uniforms.transform.normalToWorld = NormalMatrix(draw.model);
        uniforms.transform.modelViewProjection = projection * view * draw.model;
        uniforms.transform.eyePosition = camera.position;

        uniforms.lighting.eyePosition = camera.position;
        uniforms.lighting.cameraForward = forward;
        uniforms.lighting.ambientIrradiance = Vec3{0.055f, 0.065f, 0.085f};
        uniforms.lighting.material = draw.material;
        uniforms.lighting.baseColorTexture = draw.baseColor;
        uniforms.lighting.normalTexture = draw.normal;
        uniforms.lighting.mipLevel = draw.mipLevel;
        uniforms.lighting.sampler.addressU = AddressMode::Repeat;
        uniforms.lighting.sampler.addressV = AddressMode::Repeat;
        uniforms.lighting.directional[0] = lights.sun;
        uniforms.lighting.directional[0].cascades = cascades;
        uniforms.lighting.directional[1] = lights.sky;
        uniforms.lighting.directionalCount = 2;
        uniforms.lighting.points[0] = lights.fill;
        uniforms.lighting.pointCount = 1;
        uniforms.lighting.spots[0] = lights.spot;
        uniforms.lighting.spotCount = 1;
        uniforms.lighting.fog.density = 0.004f;
        uniforms.lighting.fog.heightFalloff = 0.08f;
        uniforms.lighting.fog.startDistance = 6.0f;
        uniforms.lighting.fog.scatteringAlbedo = Vec3{0.52f, 0.60f, 0.76f};
        uniforms.lighting.fog.scatteringIntensity = 0.5f;
        uniforms.lighting.fog.integrationSteps = 4;

        ShaderDesc shader{};
        shader.pipeline.viewport = target.RenderViewport();
        shader.pipeline.cullMode = CullMode::None;
        shader.pipeline.useJobs = true;
        shader.vertexShader = SceneVertex;
        shader.pixelShader = ScenePixel;
        shader.userData = &uniforms;

        const auto status = DrawShadedTriangles(
            shader, Span<const VertexInput>(draw.mesh->Data(), draw.mesh->Size()), target.Color(),
            target.Depth(), jobSystem, heap);
        if (!status)
        {
            std::printf("draw %zu failed\n", index);
            return false;
        }
    }
    return true;
}

void ApplyToneMapping(Framebuffer& image, f32 exposureEv) noexcept
{
    for (i32 y = 0; y < image.Height(); ++y)
    {
        for (i32 x = 0; x < image.Width(); ++x)
        {
            image.SetPixel(x, y, ToneMapAces(image.GetPixel(x, y), exposureEv));
        }
    }
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

struct EdgeMetrics
{
    f32 hardStepFraction = 0.0f;
};

[[nodiscard]] i32 Luminance8(const Color& color) noexcept
{
    const f32 luminance = 0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b;
    return static_cast<i32>(EncodeSrgb8(luminance));
}

[[nodiscard]] EdgeMetrics MeasureEdges(const Framebuffer& image)
{
    EdgeMetrics metrics{};
    u64 pairs = 0;
    u64 hardSteps = 0;

    for (i32 y = 0; y < image.Height(); ++y)
    {
        for (i32 x = 0; x < image.Width(); ++x)
        {
            const i32 center = Luminance8(image.GetPixel(x, y));
            if (x + 1 < image.Width())
            {
                const i32 right = Luminance8(image.GetPixel(x + 1, y));
                ++pairs;
                hardSteps += std::abs(right - center) >= kHardStepThreshold ? 1u : 0u;
            }
            if (y + 1 < image.Height())
            {
                const i32 below = Luminance8(image.GetPixel(x, y + 1));
                ++pairs;
                hardSteps += std::abs(below - center) >= kHardStepThreshold ? 1u : 0u;
            }
        }
    }

    metrics.hardStepFraction =
        pairs == 0 ? 0.0f : static_cast<f32>(hardSteps) / static_cast<f32>(pairs);
    return metrics;
}

// Root-mean-square 8-bit sRGB channel difference against the 8x8 capture,
// which stands in for ground truth. It is the single number that says how far
// a setting is from a converged image, and unlike an edge count it also
// catches the thin geometry a low sample rate drops entirely.
[[nodiscard]] f32 MeasureErrorAgainst(const Framebuffer& image, const Framebuffer& reference)
{
    if (!reference.IsValid() || image.Width() != reference.Width() ||
        image.Height() != reference.Height())
    {
        return -1.0f;
    }
    u64 squaredError = 0;
    for (i32 y = 0; y < image.Height(); ++y)
    {
        for (i32 x = 0; x < image.Width(); ++x)
        {
            const Color a = image.GetPixel(x, y);
            const Color b = reference.GetPixel(x, y);
            const i32 differences[3] = {
                static_cast<i32>(EncodeSrgb8(a.r)) - static_cast<i32>(EncodeSrgb8(b.r)),
                static_cast<i32>(EncodeSrgb8(a.g)) - static_cast<i32>(EncodeSrgb8(b.g)),
                static_cast<i32>(EncodeSrgb8(a.b)) - static_cast<i32>(EncodeSrgb8(b.b)),
            };
            for (i32 difference : differences)
            {
                squaredError += static_cast<u64>(difference * difference);
            }
        }
    }
    const f64 samples = static_cast<f64>(image.Width()) * image.Height() * 3.0;
    return static_cast<f32>(std::sqrt(static_cast<f64>(squaredError) / samples));
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

struct Options
{
    const char* outputDirectory = "aa_samples";
    i32 width = kDefaultWidth;
    i32 height = kDefaultHeight;
    f32 exposure = kDefaultExposure;
    // Single-capture mode: render only this quality and filter.
    bool single = false;
    AntiAliasQuality quality = kDefaultAntiAliasQuality;
    ResolveFilter filter = kDefaultResolveFilter;
};

const char* QualityName(AntiAliasQuality quality) noexcept
{
    switch (quality)
    {
    case AntiAliasQuality::Off:
        return "off";
    case AntiAliasQuality::Balanced:
        return "balanced";
    case AntiAliasQuality::High:
        return "high";
    case AntiAliasQuality::Ultra:
        return "ultra";
    }
    return "unknown";
}

const char* FilterName(ResolveFilter filter) noexcept
{
    switch (filter)
    {
    case ResolveFilter::Box:
        return "box";
    case ResolveFilter::Tent:
        return "tent";
    case ResolveFilter::Mitchell:
        return "mitchell";
    case ResolveFilter::BlackmanHarris:
        return "blackman";
    }
    return "unknown";
}

bool ParseQuality(const char* text, AntiAliasQuality& out)
{
    if (std::strcmp(text, "off") == 0)
    {
        out = AntiAliasQuality::Off;
        return true;
    }
    if (std::strcmp(text, "balanced") == 0)
    {
        out = AntiAliasQuality::Balanced;
        return true;
    }
    if (std::strcmp(text, "high") == 0)
    {
        out = AntiAliasQuality::High;
        return true;
    }
    if (std::strcmp(text, "ultra") == 0)
    {
        out = AntiAliasQuality::Ultra;
        return true;
    }
    return false;
}

bool ParseFilter(const char* text, ResolveFilter& out)
{
    if (std::strcmp(text, "box") == 0)
    {
        out = ResolveFilter::Box;
        return true;
    }
    if (std::strcmp(text, "tent") == 0)
    {
        out = ResolveFilter::Tent;
        return true;
    }
    if (std::strcmp(text, "mitchell") == 0)
    {
        out = ResolveFilter::Mitchell;
        return true;
    }
    if (std::strcmp(text, "blackman") == 0)
    {
        out = ResolveFilter::BlackmanHarris;
        return true;
    }
    return false;
}

void PrintUsage()
{
    std::printf("usage: cyber_sample_antialiasing [--out <directory>] [--width <px>] "
                "[--height <px>]\n");
    std::printf("                                 [--exposure <stops>] "
                "[--quality off|balanced|high|ultra]\n");
    std::printf("                                 [--filter box|tent|mitchell|blackman]\n");
    std::printf("  With no --quality or --filter, writes the full comparison set: one capture\n");
    std::printf("  per quality preset and one per reconstruction filter at 4x4.\n");
    std::printf("  Supplying either renders that single configuration only.\n");
}

bool ParseOptions(int argc, char** argv, Options& options)
{
    for (int index = 1; index < argc; ++index)
    {
        const char* argument = argv[index];
        const auto next = [&](const char*& value)
        {
            if (index + 1 >= argc)
            {
                return false;
            }
            value = argv[++index];
            return true;
        };

        if (std::strcmp(argument, "--out") == 0)
        {
            if (!next(options.outputDirectory))
            {
                return false;
            }
        }
        else if (std::strcmp(argument, "--width") == 0 || std::strcmp(argument, "--height") == 0)
        {
            const char* value = nullptr;
            if (!next(value))
            {
                return false;
            }
            char* end = nullptr;
            const long pixels = std::strtol(value, &end, 10);
            if (end == value || *end != '\0' || pixels < 32 || pixels > 4096)
            {
                return false;
            }
            if (std::strcmp(argument, "--width") == 0)
            {
                options.width = static_cast<i32>(pixels);
            }
            else
            {
                options.height = static_cast<i32>(pixels);
            }
        }
        else if (std::strcmp(argument, "--exposure") == 0)
        {
            const char* value = nullptr;
            if (!next(value))
            {
                return false;
            }
            char* end = nullptr;
            const f32 stops = std::strtof(value, &end);
            if (end == value || *end != '\0' || !std::isfinite(stops) || stops < -16.0f ||
                stops > 16.0f)
            {
                return false;
            }
            options.exposure = stops;
        }
        else if (std::strcmp(argument, "--quality") == 0)
        {
            const char* value = nullptr;
            if (!next(value) || !ParseQuality(value, options.quality))
            {
                return false;
            }
            options.single = true;
        }
        else if (std::strcmp(argument, "--filter") == 0)
        {
            const char* value = nullptr;
            if (!next(value) || !ParseFilter(value, options.filter))
            {
                return false;
            }
            options.single = true;
        }
        else
        {
            return false;
        }
    }
    return true;
}

void BuildPath(char* out, usize capacity, const char* directory, const char* name)
{
    (void)std::snprintf(out, capacity, "%s/%s.ppm", directory, name);
}

[[nodiscard]] usize CountTriangles(const DynamicArray<Draw>& draws) noexcept
{
    usize total = 0;
    for (usize index = 0; index < draws.Size(); ++index)
    {
        const Draw& draw = draws.Data()[index];
        total += draw.mesh != nullptr ? draw.mesh->Size() / 3 : 0;
    }
    return total;
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

    HeapAllocator heap;

    auto jobResult = cyber::jobs::JobSystem::Create(cyber::jobs::JobSystem::Desc{0, nullptr}, heap);
    if (!jobResult)
    {
        std::printf("failed to create the job system\n");
        return 1;
    }
    cyber::jobs::JobSystem jobSystem = std::move(jobResult).Value();

    Meshes meshes(heap);
    if (!BuildSphere(meshes.sphere, 64, 40) || !BuildBox(meshes.box) ||
        !BuildTorus(meshes.torus, 0.72f, 0.21f, 72, 28) || !BuildCylinder(meshes.rod, 16) ||
        !BuildFloor(meshes.floor, 30.0f, 48))
    {
        std::printf("failed to build scene geometry\n");
        return 1;
    }

    Textures textures{};
    if (!MakeFloorTexture(heap, textures.floorBaseColor) ||
        !MakeFloorNormalTexture(heap, textures.floorNormal) ||
        !MakeWhiteTexture(heap, textures.white))
    {
        std::printf("failed to build scene textures\n");
        return 1;
    }

    DynamicArray<Draw> draws(heap);
    if (!BuildScene(meshes, textures, draws))
    {
        std::printf("failed to build the scene\n");
        return 1;
    }

    MeshData casters(heap);
    if (!BuildCasters(draws, casters))
    {
        std::printf("failed to build the shadow caster list\n");
        return 1;
    }

    Lights lights{};
    BuildLights(lights);

    Camera camera{};
    camera.position = Vec3{0.35f, 1.75f, 7.1f};
    camera.yawRadians = -0.045f;
    camera.pitchRadians = -0.135f;
    camera.fovYRadians = 38.0f * cyber::math::kDegreesToRadians;
    camera.nearPlane = 0.1f;
    camera.farPlane = 160.0f;

    DepthBuffer shadowMaps[kCascadeCount];
    for (u32 index = 0; index < kCascadeCount; ++index)
    {
        DepthBufferDesc desc{};
        desc.width = static_cast<i32>(kCascadeResolution);
        desc.height = static_cast<i32>(kCascadeResolution);
        auto created = DepthBuffer::Create(desc, heap);
        if (!created)
        {
            std::printf("failed to allocate shadow map %u\n", index);
            return 1;
        }
        shadowMaps[index] = std::move(created).Value();
    }

    if (!cyber::platform::CreateDirectories(options.outputDirectory))
    {
        std::printf("failed to create the output directory: %s\n", options.outputDirectory);
        return 1;
    }

    struct Capture
    {
        const char* name;
        AntiAliasQuality quality;
        ResolveFilter filter;
    };

    Capture captures[8]{};
    usize captureCount = 0;
    if (options.single)
    {
        static char singleName[128];
        (void)std::snprintf(singleName, sizeof(singleName), "aa_%s_%s",
                            QualityName(options.quality), FilterName(options.filter));
        captures[captureCount++] = Capture{singleName, options.quality, options.filter};
    }
    else
    {
        // The 8x8 capture is rendered first so the rest can be measured
        // against it.
        captures[captureCount++] =
            Capture{"aa_ultra_8x", AntiAliasQuality::Ultra, kDefaultResolveFilter};
        captures[captureCount++] = Capture{"aa_off", AntiAliasQuality::Off, ResolveFilter::Box};
        captures[captureCount++] =
            Capture{"aa_balanced_2x", AntiAliasQuality::Balanced, kDefaultResolveFilter};
        captures[captureCount++] =
            Capture{"aa_high_4x", AntiAliasQuality::High, kDefaultResolveFilter};
        captures[captureCount++] =
            Capture{"filter_box_4x", AntiAliasQuality::High, ResolveFilter::Box};
        captures[captureCount++] =
            Capture{"filter_tent_4x", AntiAliasQuality::High, ResolveFilter::Tent};
        captures[captureCount++] =
            Capture{"filter_mitchell_4x", AntiAliasQuality::High, ResolveFilter::Mitchell};
        captures[captureCount++] =
            Capture{"filter_blackman_4x", AntiAliasQuality::High, ResolveFilter::BlackmanHarris};
    }

    std::printf("CyberGraphics anti-aliasing showcase\n");
    std::printf("  scene     %zu draws, %zu triangles, %zu shadow-caster triangles\n", draws.Size(),
                CountTriangles(draws), casters.Size() / 3);
    std::printf("  output    %dx%d into %s\n", options.width, options.height,
                options.outputDirectory);
    std::printf("\n%-20s %7s %9s %10s %11s %9s\n", "capture", "samples", "filter", "seconds",
                "hard steps", "vs 8x");

    // Reference image for the error column; stays invalid in single-capture
    // mode, where there is nothing to compare against.
    Framebuffer reference;

    for (usize index = 0; index < captureCount; ++index)
    {
        const Capture& capture = captures[index];

        AntiAliasTargetDesc targetDesc{};
        targetDesc.width = options.width;
        targetDesc.height = options.height;
        targetDesc.quality = capture.quality;
        targetDesc.filter = capture.filter;
        targetDesc.format = PixelFormat::RGBA32F;
        auto createdTarget = AntiAliasTarget::Create(targetDesc, heap);
        if (!createdTarget)
        {
            std::printf("failed to create the %s target\n", capture.name);
            return 1;
        }
        AntiAliasTarget target = std::move(createdTarget).Value();

        FramebufferDesc resolvedDesc{};
        resolvedDesc.width = options.width;
        resolvedDesc.height = options.height;
        resolvedDesc.format = PixelFormat::RGBA32F;
        auto createdResolved = Framebuffer::Create(resolvedDesc, heap);
        if (!createdResolved)
        {
            std::printf("failed to create the %s resolve target\n", capture.name);
            return 1;
        }
        Framebuffer resolved = std::move(createdResolved).Value();

        const auto start = std::chrono::steady_clock::now();

        DirectionalShadowCascadeSet cascades{};
        if (!RenderScene(target, draws, casters, camera, lights, cascades,
                         Span<DepthBuffer>(shadowMaps, kCascadeCount), &jobSystem, heap))
        {
            return 1;
        }

        ResolveDesc resolveDesc{};
        resolveDesc.bandRows = 32;
        resolveDesc.useJobs = true;
        if (!target.Resolve(resolved, &jobSystem, heap, resolveDesc))
        {
            std::printf("failed to resolve %s\n", capture.name);
            return 1;
        }

        // Tone mapping runs after the resolve so the filter averages HDR
        // radiance, not clipped display values: a bright highlight next to a
        // dark edge then reconstructs at its real intensity.
        ApplyToneMapping(resolved, options.exposure);

        const auto finish = std::chrono::steady_clock::now();
        const double seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count();

        const EdgeMetrics metrics = MeasureEdges(resolved);
        const f32 error = MeasureErrorAgainst(resolved, reference);

        char path[512];
        BuildPath(path, sizeof(path), options.outputDirectory, capture.name);
        ImageWriteDesc writeDesc{};
        writeDesc.format = ImageFormat::Ppm;
        writeDesc.encodeSrgb = true;
        if (!WriteImage(path, resolved, heap, writeDesc))
        {
            std::printf("failed to write %s\n", path);
            return 1;
        }

        if (error < 0.0f)
        {
            std::printf("%-20s %7d %9s %10.2f %10.3f%% %9s\n", capture.name, target.SampleCount(),
                        FilterName(capture.filter), seconds,
                        static_cast<double>(metrics.hardStepFraction) * 100.0, "reference");
        }
        else
        {
            std::printf("%-20s %7d %9s %10.2f %10.3f%% %9.2f\n", capture.name, target.SampleCount(),
                        FilterName(capture.filter), seconds,
                        static_cast<double>(metrics.hardStepFraction) * 100.0,
                        static_cast<double>(error));
        }
        std::fflush(stdout);

        // The first sweep capture is the 8x8 reference every later row is
        // measured against.
        if (!options.single && !reference.IsValid())
        {
            reference = std::move(resolved);
        }
    }

    std::printf("\nwrote %zu capture(s) to %s\n", captureCount, options.outputDirectory);
    jobSystem.Shutdown();
    return 0;
}
