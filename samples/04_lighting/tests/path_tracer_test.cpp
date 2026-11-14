#include "../reference_renderer.h"

#include <cyber/memory/heap_allocator.h>
#include <cyber/test/test.h>

#include <cstring>
#include <utility>

using namespace cyber::memory;
using namespace cyber::software;
using cyber::math::Color;
using cyber::math::Vec3;

namespace
{

bool CreateTarget(HeapAllocator& allocator, Framebuffer& target)
{
    FramebufferDesc desc{};
    desc.width = 32;
    desc.height = 24;
    desc.format = PixelFormat::RGBA32F;
    auto result = Framebuffer::Create(desc, allocator);
    if (!result)
    {
        return false;
    }
    target = std::move(result.Value());
    return true;
}

SphereGridPathTraceDesc TestDesc()
{
    SphereGridPathTraceDesc desc{};
    desc.eyePosition = Vec3{0.0f, 0.0f, 5.0f};
    desc.verticalFovRadians = 45.0f * cyber::math::kDegreesToRadians;
    desc.sphereRadius = 0.75f;
    desc.sphereSpacing = 2.0f;
    desc.gridSize = 1;
    desc.samplesPerPixel = 8;
    desc.maxBounces = 3;
    desc.seed = 0xC0FFEE26ull;
    return desc;
}

}  // namespace

CG_TEST(software_path_tracer_is_deterministic_and_lit)
{
    HeapAllocator allocator;
    Framebuffer first;
    Framebuffer second;
    CG_REQUIRE(CreateTarget(allocator, first));
    CG_REQUIRE(CreateTarget(allocator, second));

    PbrLightingData material{};
    material.eyePosition = Vec3{0.0f, 0.0f, 5.0f};
    material.ambientIrradiance = Vec3{0.1f, 0.1f, 0.1f};
    material.material.baseColor = Color{0.7f, 0.6f, 0.5f, 1.0f};
    material.material.roughness = 0.4f;
    material.directionalCount = 1;
    material.directional[0].direction = Vec3{0.0f, 0.0f, -1.0f};
    material.directional[0].radiance = Vec3{2.0f, 2.0f, 2.0f};
    const PbrLightingData materials[] = {material};
    const SphereGridPathTraceDesc desc = TestDesc();
    const Span<const PbrLightingData> span{materials, 1};

    CG_EXPECT_TRUE(RenderSphereGridPathTrace(first, span, desc));
    CG_EXPECT_TRUE(RenderSphereGridPathTrace(second, span, desc));
    CG_EXPECT_EQ(first.DataSizeBytes(), second.DataSizeBytes());
    CG_EXPECT_TRUE(std::memcmp(first.Data(), second.Data(), first.DataSizeBytes()) == 0);
    CG_EXPECT_TRUE(first.GetPixel(16, 12).r > 0.0f);
    CG_EXPECT_NEAR(first.GetPixel(0, 0).a, 0.0f, 1.0e-7f);
}

CG_TEST(software_path_tracer_rejects_invalid_configuration)
{
    HeapAllocator allocator;
    Framebuffer target;
    CG_REQUIRE(CreateTarget(allocator, target));
    PbrLightingData material{};
    const PbrLightingData materials[] = {material};
    SphereGridPathTraceDesc desc = TestDesc();
    desc.samplesPerPixel = 0;
    CG_EXPECT_FALSE(
        RenderSphereGridPathTrace(target, Span<const PbrLightingData>(materials, 1), desc));

    desc = TestDesc();
    desc.sphereSpacing = desc.sphereRadius;
    CG_EXPECT_FALSE(
        RenderSphereGridPathTrace(target, Span<const PbrLightingData>(materials, 1), desc));
}
