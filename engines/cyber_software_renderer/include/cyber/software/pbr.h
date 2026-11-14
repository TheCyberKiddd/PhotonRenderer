#pragma once

#include <cyber/core/types.h>
#include <cyber/math/color.h>
#include <cyber/math/vec3.h>
#include <cyber/memory/span.h>
#include <cyber/software/ibl.h>
#include <cyber/software/shading.h>
#include <cyber/software/shadow.h>
#include <cyber/software/texture.h>

namespace cyber::software
{

using cyber::core::f32;
using cyber::core::u32;

// Defined in global_illumination.h, which a caller that sets
// PbrLightingData::irradianceVolume includes for the full type.
struct IrradianceVolumeView;

struct PbrMaterial
{
    math::Color baseColor{0.8f, 0.8f, 0.8f, 1.0f};
    f32 metallic = 0.0f;
    f32 roughness = 0.5f;
    f32 normalScale = 1.0f;
};

struct PbrDirectionalLight
{
    // Direction the light travels. Radiance is linear RGB in W/(m^2 sr).
    math::Vec3 direction{-0.5f, -1.0f, -0.3f};
    math::Vec3 radiance{1.0f, 1.0f, 1.0f};
    ShadowMapView shadow{};
    DirectionalShadowCascadeSet cascades{};
};

struct PbrPointLight
{
    math::Vec3 position;
    // Linear RGB radiant intensity in W/sr; incident radiance falls as 1/r^2.
    math::Vec3 intensity{1.0f, 1.0f, 1.0f};
    // Smoothly reaches zero at this distance, in meters.
    f32 range = 10.0f;
    PointShadowMapSet shadow{};
};

struct PbrSpotLight
{
    math::Vec3 position;
    // Direction the light travels. Angles are measured from this axis.
    math::Vec3 direction{0.0f, -1.0f, 0.0f};
    // Linear RGB radiant intensity in W/sr; incident radiance falls as 1/r^2.
    math::Vec3 intensity{1.0f, 1.0f, 1.0f};
    f32 range = 10.0f;
    f32 innerConeRadians = 0.35f;
    f32 outerConeRadians = 0.55f;
    ShadowMapView shadow{};
};

struct PbrHeightFog
{
    // Extinction per world unit at baseHeight. Zero disables fog entirely.
    f32 density = 0.0f;
    f32 heightFalloff = 0.0f;
    f32 baseHeight = 0.0f;
    f32 startDistance = 0.0f;
    f32 maxDistance = 10000.0f;
    math::Vec3 scatteringAlbedo{0.8f, 0.85f, 0.9f};
    f32 scatteringIntensity = 1.0f;
    // Henyey-Greenstein anisotropy, clamped to [-0.9, 0.9].
    f32 anisotropy = 0.0f;
    // Fixed midpoint integration steps; clamped to [1, 32].
    u32 integrationSteps = 8;
};

inline constexpr u32 kMaxPbrDirectionalLights = 2;
inline constexpr u32 kMaxPbrPointLights = 4;
inline constexpr u32 kMaxPbrSpotLights = 4;

struct PbrLightingData
{
    math::Vec3 eyePosition;
    // Used to select a directional shadow cascade; ignored when no cascades are
    // configured. Positive depth increases along this direction.
    math::Vec3 cameraForward{0.0f, 0.0f, -1.0f};
    bool visualizeShadowCascades = false;
    // Irradiance from a uniform diffuse environment, in W/m^2. Used only when
    // no image-based lighting is configured below.
    math::Vec3 ambientIrradiance{0.03f, 0.03f, 0.03f};
    // Image-based lighting replaces the flat ambient term. Exactly one source
    // answers, in this order: a probe set, then a global environment, then
    // ambientIrradiance. A probe set and a global environment are not summed,
    // because both carry the full diffuse and specular environment response.
    ReflectionProbeSet reflectionProbes{};
    const EnvironmentLighting* environment = nullptr;
    // Screen-space reflections, composited over the probe result by
    // confidence. Ignored unless a probe set is configured.
    const ScreenSpaceReflectionDesc* screenSpaceReflections = nullptr;
    // Baked global illumination. When set, the volume answers the diffuse
    // ambient term in place of whichever source above would have: inside the
    // image-based path it replaces only the environment's irradiance lookup,
    // so probes, screen-space reflections, and a global environment still
    // supply the specular term; without image-based lighting it replaces
    // ambientIrradiance. ambientOcclusion is applied to the result in both
    // cases, which is what supplies the contact shadowing a probe grid is too
    // coarse to represent.
    const IrradianceVolumeView* irradianceVolume = nullptr;
    // Hemispherical visibility, which also drives specular occlusion.
    f32 ambientOcclusion = 1.0f;
    PbrMaterial material{};
    // Texture RGB values are linear. Base-color images authored in sRGB must
    // be decoded on import; normal maps are sampled as linear data.
    const Texture* baseColorTexture = nullptr;
    const Texture* normalTexture = nullptr;
    SamplerDesc sampler{};
    f32 mipLevel = 0.0f;

    PbrDirectionalLight directional[kMaxPbrDirectionalLights]{};
    u32 directionalCount = 0;
    PbrPointLight points[kMaxPbrPointLights]{};
    u32 pointCount = 0;
    // Optional arbitrary-sized point-light data. When usePointLightIndices is
    // set, pointLightIndices selects the entries used for this pixel.
    cyber::memory::Span<const PbrPointLight> pointLightSpan{};
    cyber::memory::Span<const u32> pointLightIndices{};
    bool usePointLightIndices = false;
    PbrSpotLight spots[kMaxPbrSpotLights]{};
    u32 spotCount = 0;
    PbrHeightFog fog{};
};

// Evaluates the Cook-Torrance metallic-roughness BRDF. V and L point away
// from the surface; the returned RGB is a linear reflectance per steradian.
// Inputs are normalized internally. Invalid directions, back-facing
// configurations, and non-finite material values produce zero contribution.
[[nodiscard]] math::Color EvaluateGgxBrdf(const math::Vec3& normal, const math::Vec3& viewDirection,
                                          const math::Vec3& lightDirection,
                                          const PbrMaterial& material) noexcept;

// Decodes an RGB normal-map sample in [0, 1] and transforms it through an
// orthonormal tangent frame. The tangent is re-orthogonalized; a degenerate
// tangent selects a stable perpendicular axis.
[[nodiscard]] math::Vec3 ApplyTangentSpaceNormal(const math::Vec3& normal,
                                                 const math::Vec3& tangent, f32 tangentSign,
                                                 const math::Color& normalSample,
                                                 f32 normalScale = 1.0f) noexcept;

// Cook-Torrance forward pixel shader. It adds direct directional, inverse-
// square point, and smooth-cone spot lighting to the diffuse ambient term.
// Result RGB is unclamped linear HDR. A null user-data pointer returns the
// interpolated vertex color.
[[nodiscard]] math::Color PbrPixelShader(const Varyings& varyings, const void* userData) noexcept;

// ACES fitted tone mapper. Exposure is in photographic stops; output RGB is
// clamped to [0, 1] linear and should be sRGB-encoded by the output stage.
[[nodiscard]] math::Color ToneMapAces(const math::Color& hdrColor, f32 exposureEv = 0.0f) noexcept;

}  // namespace cyber::software
