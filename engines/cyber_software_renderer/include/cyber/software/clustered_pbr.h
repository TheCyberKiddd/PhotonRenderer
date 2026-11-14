#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/math/mat4.h>
#include <cyber/memory/span.h>
#include <cyber/renderer/clustered_lighting.h>
#include <cyber/software/pbr.h>

namespace cyber::software
{

struct ClusteredPbrLightingData
{
    const PbrLightingData* lighting = nullptr;
    cyber::memory::Span<const PbrPointLight> pointLights{};
    const cyber::renderer::ClusterLightLists* clusters = nullptr;
    // Transforms world space to a conventional right-handed view where forward
    // is -Z. Cluster bounds use positive view depth.
    math::Mat4 viewFromWorld{};
};

// Converts point lights into the conservative view-space bounds consumed by
// BuildClusterLightLists. Output must have one element per light.
[[nodiscard]] core::Status<core::Error>
BuildClusterLightBounds(cyber::memory::Span<const PbrPointLight> pointLights,
                        const math::Mat4& viewFromWorld,
                        cyber::memory::Span<cyber::renderer::ClusterLightBound> output) noexcept;

// CPU-only clustered-forward PBR pixel shader; it does not require RHI or GPU
// compute support. If the fragment is outside the supplied grid or its list is
// incompatible with the point-light span, it evaluates all local lights
// rather than risking missing illumination.
[[nodiscard]] math::Color ClusteredPbrPixelShader(const Varyings& varyings,
                                                  const void* userData) noexcept;

}  // namespace cyber::software
