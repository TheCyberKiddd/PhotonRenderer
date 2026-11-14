#pragma once

// Scene construction shared by the Phase 31 global-illumination suites.
//
// Geometry is grouped by material: each group becomes one bottom-level
// structure and one instance whose identifier is the group index, which is how
// GiScene finds the material. Every quad is wound so its geometric normal
// points the way the caller asked, because emission and the inside/outside
// probe classification both depend on which side is the front.

#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/global_illumination.h>
#include <cyber/software/raytrace.h>

#include <utility>
#include <vector>

namespace cyber::software::test
{

class GiTestScene
{
public:
    explicit GiTestScene(memory::HeapAllocator& allocator) noexcept : m_allocator(&allocator) {}

    GiTestScene(const GiTestScene&) = delete;
    GiTestScene& operator=(const GiTestScene&) = delete;

    u32 AddGroup(const GiSurfaceMaterial& material)
    {
        m_groups.emplace_back();
        m_materials.push_back(material);
        return static_cast<u32>(m_groups.size() - 1);
    }

    // A rectangle centred on center, spanning +-halfU and +-halfV, whose front
    // face points along facing.
    void AddQuad(u32 group, const math::Vec3& center, const math::Vec3& halfU,
                 const math::Vec3& halfV, const math::Vec3& facing)
    {
        math::Vec3 a = center - halfU - halfV;
        math::Vec3 b = center + halfU - halfV;
        const math::Vec3 c = center + halfU + halfV;
        math::Vec3 d = center - halfU + halfV;
        if (math::Dot(math::Cross(b - a, c - a), facing) < 0.0f)
        {
            std::swap(b, d);
        }
        const math::Vec3 normal = math::Normalize(facing);
        Push(group, a, normal);
        Push(group, b, normal);
        Push(group, c, normal);
        Push(group, a, normal);
        Push(group, c, normal);
        Push(group, d, normal);
    }

    // Six faces of an axis-aligned box. inward faces the inside, which is what
    // a room is; outward is a solid object.
    void AddBox(u32 group, const math::Vec3& minimum, const math::Vec3& maximum, bool inward)
    {
        const math::Vec3 center = (minimum + maximum) * 0.5f;
        const math::Vec3 half = (maximum - minimum) * 0.5f;
        const f32 sign = inward ? -1.0f : 1.0f;
        const math::Vec3 x{half.x, 0.0f, 0.0f};
        const math::Vec3 y{0.0f, half.y, 0.0f};
        const math::Vec3 z{0.0f, 0.0f, half.z};
        AddQuad(group, center + x, y, z, math::Vec3{sign, 0.0f, 0.0f});
        AddQuad(group, center - x, y, z, math::Vec3{-sign, 0.0f, 0.0f});
        AddQuad(group, center + y, x, z, math::Vec3{0.0f, sign, 0.0f});
        AddQuad(group, center - y, x, z, math::Vec3{0.0f, -sign, 0.0f});
        AddQuad(group, center + z, x, y, math::Vec3{0.0f, 0.0f, sign});
        AddQuad(group, center - z, x, y, math::Vec3{0.0f, 0.0f, -sign});
    }

    [[nodiscard]] bool Build()
    {
        m_blases.clear();
        m_blases.reserve(m_groups.size());
        std::vector<RayInstanceDesc> instances;
        for (u32 group = 0; group < m_groups.size(); ++group)
        {
            if (m_groups[group].empty())
            {
                continue;
            }
            AccelerationStructureBuildDesc desc{};
            desc.vertices =
                memory::Span<const VertexInput>(m_groups[group].data(), m_groups[group].size());
            auto built = BottomLevelAccelerationStructure::Build(desc, *m_allocator);
            if (built.IsError())
            {
                return false;
            }
            m_blases.push_back(std::move(built).Value());
            RayInstanceDesc instance{};
            instance.blas = &m_blases.back();
            instance.instanceId = group;
            instances.push_back(instance);
        }
        auto tlas = TopLevelAccelerationStructure::Build(
            memory::Span<const RayInstanceDesc>(instances.data(), instances.size()), *m_allocator);
        if (tlas.IsError())
        {
            return false;
        }
        m_tlas = std::move(tlas).Value();
        return true;
    }

    [[nodiscard]] GiScene Scene() const noexcept
    {
        GiScene scene{};
        scene.geometry = &m_tlas;
        scene.materials =
            memory::Span<const GiSurfaceMaterial>(m_materials.data(), m_materials.size());
        return scene;
    }

    [[nodiscard]] const TopLevelAccelerationStructure& Geometry() const noexcept
    {
        return m_tlas;
    }

private:
    void Push(u32 group, const math::Vec3& position, const math::Vec3& normal)
    {
        VertexInput vertex{};
        vertex.position = position;
        vertex.normal = normal;
        m_groups[group].push_back(vertex);
    }

    memory::HeapAllocator* m_allocator = nullptr;
    std::vector<std::vector<VertexInput>> m_groups;
    std::vector<GiSurfaceMaterial> m_materials;
    std::vector<BottomLevelAccelerationStructure> m_blases;
    TopLevelAccelerationStructure m_tlas;
};

inline GiSurfaceMaterial Diffuse(f32 r, f32 g, f32 b) noexcept
{
    GiSurfaceMaterial material{};
    material.albedo = math::Vec3{r, g, b};
    return material;
}

inline GiSurfaceMaterial Emissive(f32 albedo, f32 radiance) noexcept
{
    GiSurfaceMaterial material{};
    material.albedo = math::Vec3{albedo, albedo, albedo};
    material.emission = math::Vec3{radiance, radiance, radiance};
    return material;
}

// Image helpers for the denoiser suites. Each returns an invalid object on
// failure, which the caller reports through CG_REQUIRE.
inline Framebuffer MakeImage(memory::HeapAllocator& allocator, i32 width, i32 height)
{
    FramebufferDesc desc{};
    desc.width = width;
    desc.height = height;
    desc.format = PixelFormat::RGBA32F;
    auto created = Framebuffer::Create(desc, allocator);
    if (created.IsError())
    {
        return Framebuffer{};
    }
    return std::move(created).Value();
}

inline DepthBuffer MakeDepth(memory::HeapAllocator& allocator, i32 width, i32 height)
{
    DepthBufferDesc desc{};
    desc.width = width;
    desc.height = height;
    auto created = DepthBuffer::Create(desc, allocator);
    if (created.IsError())
    {
        return DepthBuffer{};
    }
    return std::move(created).Value();
}

// Inverse of LinearizeReverseZ for a perspective projection.
inline f32 ReverseZFromDistance(f32 distance, f32 nearPlane, f32 farPlane) noexcept
{
    return (nearPlane * farPlane / distance - nearPlane) / (farPlane - nearPlane);
}

// World-space normal encoded to [0, 1], as the post chain stores it.
inline math::Color EncodeNormal(const math::Vec3& normal) noexcept
{
    const math::Vec3 n = math::Normalize(normal);
    return math::Color{n.x * 0.5f + 0.5f, n.y * 0.5f + 0.5f, n.z * 0.5f + 0.5f, 1.0f};
}

// Uniform in [0, 1) from three integers: the per-pixel, per-frame noise the
// denoiser suites feed in, identical on every run.
inline f32 HashUniform(u32 x, u32 y, u32 frame) noexcept
{
    u32 value = x * 0x8DA6B343u ^ y * 0xD8163841u ^ frame * 0xCB1AB31Fu;
    value ^= value >> 16u;
    value *= 0x7FEB352Du;
    value ^= value >> 15u;
    value *= 0x846CA68Bu;
    value ^= value >> 16u;
    return static_cast<f32>(value >> 8u) * (1.0f / 16777216.0f);
}

// The SH coefficients of a constant radiance field: only the DC term is
// non-zero, and it is the radiance times the integral of Y00 over the sphere.
inline SphericalHarmonicsL2 ConstantRadianceSh(const math::Vec3& radiance) noexcept
{
    SphericalHarmonicsL2 harmonics{};
    harmonics.coefficients[0] = radiance * (0.282095f * 4.0f * math::kPi);
    return harmonics;
}

}  // namespace cyber::software::test
