#pragma once

#include <cyber/core/types.h>
#include <cyber/math/vec3.h>
#include <cyber/memory/span.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/pbr.h>

namespace cyber::software
{

struct SphereGridPathTraceDesc
{
    math::Vec3 eyePosition{0.0f, 0.0f, 20.0f};
    core::f32 verticalFovRadians = 0.0f;
    core::f32 sphereRadius = 0.0f;
    core::f32 sphereSpacing = 0.0f;
    core::u32 gridSize = 0;
    core::u32 samplesPerPixel = 0;
    core::u32 maxBounces = 0;
    core::u64 seed = 0;
};

// Independent deterministic CPU path tracer for a regular grid of spheres.
// The supplied records contain scene inputs only; this implementation never
// calls the production renderer's BRDF or pixel shader.
[[nodiscard]] bool RenderSphereGridPathTrace(Framebuffer& target,
                                             memory::Span<const PbrLightingData> materials,
                                             const SphereGridPathTraceDesc& desc) noexcept;

}  // namespace cyber::software
