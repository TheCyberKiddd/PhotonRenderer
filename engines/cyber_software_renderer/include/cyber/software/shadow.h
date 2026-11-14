#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/math/mat4.h>
#include <cyber/math/vec3.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/span.h>
#include <cyber/software/camera.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/pipeline.h>
#include <cyber/software/shading.h>

namespace cyber::software
{

using cyber::core::f32;
using cyber::core::u32;
using cyber::core::u64;

// Non-owning reverse-Z shadow-map view. `clipFromWorld` must use the same
// reverse-Z depth convention as DepthBuffer: near is 1 and far/clear is 0.
struct ShadowMapView
{
    const DepthBuffer* depth = nullptr;
    math::Mat4 clipFromWorld = math::Mat4::Identity();
    // Positive depth bias moves the receiver toward the light in reverse-Z.
    f32 constantDepthBias = 0.0005f;
    f32 slopeDepthBias = 0.002f;
    // World-space offset along the receiver normal.
    f32 normalOffsetBias = 0.0f;
    // Radius in texels: hard PCF is clamped to 4; soft PCF and PCSS to 16.
    u32 filterRadius = 1;
    // A fixed stratified 4x4 tap grid softens PCF without per-pixel rotation noise.
    bool softFilter = false;
    // PCSS blocker search and contact-hardening filter. Perspective maps use a
    // physical emitter radius; orthographic maps use an angular emitter radius.
    bool usePcss = false;
    bool perspectiveProjection = false;
    f32 nearPlane = 0.1f;
    f32 farPlane = 100.0f;
    f32 verticalFovRadians = 0.0f;
    f32 worldUnitsPerTexel = 0.0f;
    f32 sourceRadius = 0.0f;
    f32 sourceAngularRadiusRadians = 0.0f;
};

inline constexpr u32 kMaxShadowFilterRadius = 4;
inline constexpr u32 kMaxSoftShadowFilterRadius = 16;
inline constexpr u32 kMaxDirectionalShadowCascades = 4;
inline constexpr u32 kShadowAtlasTileSize = 64;
inline constexpr u32 kMaxShadowAtlasAllocations = 256;

// Returns the fraction of PCF taps lit in [0, 1]. Invalid or out-of-frustum
// lookups are treated as lit, matching the clear depth of a reverse-Z map.
// The function is allocation-free and suitable for pixel-shader use.
[[nodiscard]] f32 EvaluateShadowVisibility(const ShadowMapView& shadow,
                                           const math::Vec3& worldPosition,
                                           const math::Vec3& worldNormal,
                                           const math::Vec3& directionToLight) noexcept;

struct SpotShadowConfig
{
    f32 nearPlane = 0.05f;
    f32 farPlane = 25.0f;
    f32 outerConeRadians = 0.55f;
    f32 constantDepthBias = 0.0005f;
    f32 slopeDepthBias = 0.002f;
    f32 normalOffsetBias = 0.0f;
    u32 filterRadius = 8;
    f32 sourceRadius = 0.05f;
};

// Produces a reverse-Z perspective transform for a spot light. Direction
// points along the cone axis and outerConeRadians is its half-angle.
[[nodiscard]] core::Status<core::Error> BuildSpotShadowMap(const math::Vec3& position,
                                                           const math::Vec3& direction,
                                                           const SpotShadowConfig& config,
                                                           ShadowMapView& output) noexcept;

struct PointShadowConfig
{
    f32 nearPlane = 0.05f;
    f32 farPlane = 25.0f;
    f32 constantDepthBias = 0.0005f;
    f32 slopeDepthBias = 0.002f;
    f32 normalOffsetBias = 0.0f;
    u32 filterRadius = 8;
    f32 sourceRadius = 0.05f;
};

struct PointShadowMapSet
{
    ShadowMapView faces[6]{};
};

// Produces the six reverse-Z 90-degree face transforms in +X, -X, +Y, -Y,
// +Z, -Z order. The caller supplies and owns each face depth map.
[[nodiscard]] core::Status<core::Error> BuildPointShadowMaps(const math::Vec3& position,
                                                             const PointShadowConfig& config,
                                                             PointShadowMapSet& output) noexcept;

[[nodiscard]] u32 SelectShadowCubeFace(const math::Vec3& lightToReceiver) noexcept;
// Blends adjacent cube faces near point-shadow seams to avoid a dominant-face
// discontinuity. Out-of-face and missing-map samples remain lit.
[[nodiscard]] f32 EvaluatePointShadowVisibility(const PointShadowMapSet& shadows,
                                                const math::Vec3& lightToReceiver,
                                                const math::Vec3& worldPosition,
                                                const math::Vec3& worldNormal,
                                                const math::Vec3& directionToLight) noexcept;

struct ShadowAtlasAllocation
{
    u32 x = 0;
    u32 y = 0;
    u32 size = 0;
    u32 innerOffset = 0;
    u32 innerResolution = 0;
    u32 slot = 0xFFFFFFFFU;
    u32 generation = 0;
};

// Caller-owned deterministic square-tile atlas. Dimensions must be multiples
// of 64, from 64 through 4096. Allocations are power-of-two aligned, include
// guard-band texels, and are explicitly released before reuse.
struct ShadowAtlas
{
    u32 width = 0;
    u32 height = 0;
    u64 occupiedRows[64]{};
    struct AllocationRecord
    {
        u32 x = 0;
        u32 y = 0;
        u32 size = 0;
        u32 generation = 0;
        bool active = false;
    } allocations[kMaxShadowAtlasAllocations]{};
    u32 nextGeneration = 1;
};

[[nodiscard]] core::Status<core::Error> InitializeShadowAtlas(u32 width, u32 height,
                                                              ShadowAtlas& output) noexcept;
[[nodiscard]] core::Status<core::Error> SelectShadowResolution(f32 importance,
                                                               u32 minimumResolution,
                                                               u32 maximumResolution,
                                                               u32& output) noexcept;
[[nodiscard]] core::Status<core::Error> AllocateShadowAtlas(ShadowAtlas& atlas,
                                                            u32 requestedResolution,
                                                            u32 guardBandTexels,
                                                            ShadowAtlasAllocation& output) noexcept;
[[nodiscard]] bool ReleaseShadowAtlas(ShadowAtlas& atlas,
                                      const ShadowAtlasAllocation& allocation) noexcept;

// Maps a depth sample to a grayscale debug value without modifying the map.
[[nodiscard]] math::Color ShadowDepthDebugColor(f32 reverseZDepth) noexcept;

struct DirectionalShadowCascade
{
    f32 splitNear = 0.0f;
    f32 splitFar = 0.0f;
    ShadowMapView shadow{};
};

struct DirectionalShadowCascadeSet
{
    DirectionalShadowCascade cascades[kMaxDirectionalShadowCascades]{};
    u32 count = 0;
    f32 blendFraction = 0.1f;
};

struct DirectionalShadowCascadeConfig
{
    u32 cascadeCount = 4;
    u32 mapResolution = 1024;
    f32 shadowDistance = 150.0f;
    // 0 is uniform; 1 is logarithmic; intermediate values blend the partitions.
    f32 splitLambda = 0.7f;
    // Fraction of each cascade's far interval blended with its successor.
    f32 blendFraction = 0.1f;
    // Light-space depth padding for off-frustum casters.
    f32 depthPadding = 20.0f;
    f32 constantDepthBias = 0.0005f;
    f32 slopeDepthBias = 0.002f;
    f32 normalOffsetBias = 0.0f;
    u32 filterRadius = 8;
    f32 sourceAngularRadiusRadians = 0.00465f;
};

// Fits practical-split directional cascades to the camera frustum. XY bounds
// use a rounded bounding sphere and texel-snapped light-space center to avoid
// scale changes and sub-texel translation shimmer. `lightDirection` points in
// the direction light rays travel. Output shadow depths remain caller-owned.
[[nodiscard]] core::Status<core::Error>
BuildDirectionalShadowCascades(const Camera& camera, f32 aspect, const math::Vec3& lightDirection,
                               const DirectionalShadowCascadeConfig& config,
                               DirectionalShadowCascadeSet& output) noexcept;

// Selects the cascade by positive camera view depth and blends into the next
// cascade over the configured overlap. Positions beyond the final split are
// unshadowed.
[[nodiscard]] f32 EvaluateDirectionalShadowVisibility(const DirectionalShadowCascadeSet& cascades,
                                                      f32 viewDepth,
                                                      const math::Vec3& worldPosition,
                                                      const math::Vec3& worldNormal,
                                                      const math::Vec3& directionToLight) noexcept;
[[nodiscard]] u32 GetDirectionalShadowCascadeIndex(const DirectionalShadowCascadeSet& cascades,
                                                   f32 viewDepth) noexcept;
[[nodiscard]] math::Color DirectionalShadowCascadeDebugColor(u32 cascadeIndex) noexcept;

struct ShadowRenderDesc
{
    math::Mat4 modelToWorld = math::Mat4::Identity();
    math::Mat4 clipFromWorld = math::Mat4::Identity();
    CullMode cullMode = CullMode::Back;
    bool useJobs = true;
};
// Clears and renders a caller-owned depth map through the depth-only
// reverse-Z raster path. No color target is allocated, bound, or written, and
// no pixel shader runs, so a shadow map costs coverage, the depth test, and
// the depth store only. Alpha-tested occluders are therefore not supported.
[[nodiscard]] core::Status<core::Error>
RenderShadowMap(const ShadowRenderDesc& desc, cyber::memory::Span<const VertexInput> vertices,
                DepthBuffer& depth, jobs::JobSystem* jobSystem,
                cyber::memory::Allocator& allocator, PipelineStats* stats = nullptr) noexcept;

// Renders every configured directional cascade into caller-owned, equally sized
// depth maps. On success each cascade borrows its corresponding depth map. The
// depth-map array must outlive all lighting that samples the returned set.
[[nodiscard]] core::Status<core::Error>
RenderDirectionalShadowMaps(DirectionalShadowCascadeSet& cascades,
                            cyber::memory::Span<const VertexInput> vertices,
                            cyber::memory::Span<DepthBuffer> depthMaps,
                            cyber::memory::Allocator& allocator,
                            jobs::JobSystem* jobSystem = nullptr,
                            CullMode cullMode = CullMode::Back, bool useJobs = true,
                            cyber::memory::Span<PipelineStats> stats = {}) noexcept;

}  // namespace cyber::software
