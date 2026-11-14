#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/math/color.h>
#include <cyber/math/mat4.h>
#include <cyber/math/vec2.h>
#include <cyber/math/vec3.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/span.h>
#include <cyber/software/antialias.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/texture.h>

namespace cyber::jobs
{
class JobSystem;
}

namespace cyber::software
{

using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::core::u8;
using cyber::core::usize;

// Image-based lighting: environment maps, the split-sum specular chain, the
// diffuse irradiance term, reflection probes, screen-space reflections, and a
// procedural sky.
//
// Every radiometric quantity here is linear RGB. Radiance is in W/(m^2 sr) and
// irradiance is in W/m^2, the same convention the direct lights in pbr.h use,
// so an environment and a directional light of equal radiance light a surface
// by the same amount.
//
// The precomputation is deterministic: every integral uses a fixed Hammersley
// sequence and a fixed traversal order, so the same input produces identical
// output with or without a job system.

// ---------------------------------------------------------------------------
// Cubemaps
// ---------------------------------------------------------------------------

// Face order matches D3D and Vulkan cube arrays.
enum class CubeFace : u8
{
    PositiveX = 0,
    NegativeX,
    PositiveY,
    NegativeY,
    PositiveZ,
    NegativeZ,
};

inline constexpr u32 kCubeFaceCount = 6;
inline constexpr i32 kMaxCubemapFaceSize = 2048;
// 2048 down to 1 inclusive is 12 levels.
inline constexpr u32 kMaxCubemapMipLevels = 12;

struct CubemapDesc
{
    // Face edge length in texels; must be in [1, kMaxCubemapFaceSize].
    i32 faceSize = 0;
    // 0 requests the full chain down to 1x1.
    u32 mipLevels = 0;
};

// Move-only HDR cube texture with an allocator-owned mip chain.
//
// Storage is always four binary32 channels, because a prefiltered environment
// holds unbounded radiance that an 8-bit format cannot carry. Texels are
// stored level-major and then face-major inside a level.
class Cubemap
{
public:
    Cubemap() noexcept = default;
    ~Cubemap();

    Cubemap(const Cubemap&) = delete;
    Cubemap& operator=(const Cubemap&) = delete;

    Cubemap(Cubemap&& other) noexcept;
    Cubemap& operator=(Cubemap&& other) noexcept;

    // Allocates every level and clears it to transparent black. Returns
    // InvalidArgument for an out-of-range face size or level count, and
    // Internal when allocation fails.
    [[nodiscard]] static core::Result<Cubemap, core::Error>
    Create(const CubemapDesc& desc, memory::Allocator& allocator) noexcept;

    [[nodiscard]] bool IsValid() const noexcept;
    [[nodiscard]] i32 FaceSize() const noexcept;
    [[nodiscard]] u32 MipLevels() const noexcept;
    // Edge length of one level, clamped to 1. Zero for an invalid level.
    [[nodiscard]] i32 MipSize(u32 level) const noexcept;
    [[nodiscard]] usize DataSizeBytes() const noexcept;

    // Out-of-range faces, levels, or coordinates read transparent black and
    // ignore writes.
    [[nodiscard]] math::Color GetTexel(u32 face, u32 level, i32 x, i32 y) const noexcept;
    void SetTexel(u32 face, u32 level, i32 x, i32 y, const math::Color& color) noexcept;

    void Clear(const math::Color& color) noexcept;

    // Box-filters every level above zero from the four texels below it, inside
    // one face. Every tap of a 2x2 reduction stays on the face, so no cross-face
    // fetch is involved; a face size that is not a power of two drops the last
    // row and column of an odd level.
    void GenerateMips() noexcept;

private:
    void Release() noexcept;
    void MoveFrom(Cubemap& other) noexcept;
    [[nodiscard]] f32* TexelPointer(u32 face, u32 level, i32 x, i32 y) const noexcept;

    memory::Allocator* m_allocator = nullptr;
    f32* m_data = nullptr;
    usize m_sizeBytes = 0;
    usize m_levelOffsets[kMaxCubemapMipLevels] = {};
    i32 m_faceSize = 0;
    u32 m_mipLevels = 0;
};

// Maps a face-local coordinate to a direction. u and v are in [0, 1] with a
// top-left origin; values outside that range extend the face plane, which is
// how a bilinear tap that leaves a face is resolved. The result is normalized.
[[nodiscard]] math::Vec3 CubeDirectionFromFaceUv(u32 face, f32 u, f32 v) noexcept;

// Inverse of CubeDirectionFromFaceUv. A zero or non-finite direction selects
// CubeFace::PositiveX at the face center.
void CubeFaceUvFromDirection(const math::Vec3& direction, u32& face, f32& u, f32& v) noexcept;

// Solid angle of one texel of a faceSize face whose center is at (u, v) in
// [0, 1]. Summed over a whole cube this integrates to 4 pi.
[[nodiscard]] f32 CubeTexelSolidAngle(i32 faceSize, f32 u, f32 v) noexcept;

// Nearest-texel fetch along a direction.
[[nodiscard]] math::Color SampleCubemapNearest(const Cubemap& cubemap, const math::Vec3& direction,
                                               u32 level) noexcept;

// Seamless bilinear fetch inside one level. A tap that leaves the face is
// re-dispatched through the direction it represents, so there is no seam and
// no clamped-edge smear.
[[nodiscard]] math::Color SampleCubemapFace(const Cubemap& cubemap, const math::Vec3& direction,
                                            u32 level) noexcept;

// Trilinear fetch. The level is clamped to [0, MipLevels - 1] and a fractional
// level blends the two adjacent levels.
[[nodiscard]] math::Color SampleCubemapLevel(const Cubemap& cubemap, const math::Vec3& direction,
                                             f32 level) noexcept;

// ---------------------------------------------------------------------------
// Environment capture and precomputation
// ---------------------------------------------------------------------------

struct EquirectangularProjectionDesc
{
    // Stratified taps per axis inside each destination texel, in [1, 8]. The
    // default removes the aliasing a single tap leaves near the poles.
    u32 samplesPerAxis = 2;
    bool useJobs = true;
};

// Projects a latitude-longitude environment image into a cubemap. source holds
// linear radiance; u wraps around the horizon starting at -Z and v runs from +Y
// at the top to -Y at the bottom. Returns InvalidArgument for an invalid
// source, an invalid output, or an out-of-range sample count.
[[nodiscard]] core::Status<core::Error>
ProjectEquirectangularToCubemap(const Texture& source, const EquirectangularProjectionDesc& desc,
                                Cubemap& output, jobs::JobSystem* jobSystem = nullptr) noexcept;

// Projects a decoded image directly, including HDR images loaded by LoadImage.
// RGB values are treated as linear radiance; horizontal sampling wraps and
// vertical sampling clamps. The same sampling pattern and deterministic job
// behavior as the Texture overload are used.
[[nodiscard]] core::Status<core::Error>
ProjectEquirectangularToCubemap(const Framebuffer& source,
                                const EquirectangularProjectionDesc& desc, Cubemap& output,
                                jobs::JobSystem* jobSystem = nullptr) noexcept;

// Order-2 spherical harmonics, nine coefficients of linear RGB radiance.
struct SphericalHarmonicsL2
{
    math::Vec3 coefficients[9]{};
};

// Projects every texel of level 0, weighted by its solid angle, onto the
// order-2 basis. Returns InvalidArgument for an invalid cubemap.
[[nodiscard]] core::Status<core::Error>
ProjectCubemapToSphericalHarmonics(const Cubemap& environment,
                                   SphericalHarmonicsL2& output) noexcept;

// Cosine-convolved irradiance for a normal, in W/m^2. This is the Ramamoorthi
// and Hanrahan form, so a constant environment of radiance L returns pi * L and
// a Lambertian surface of albedo rho then reflects rho * L exactly.
[[nodiscard]] math::Vec3 EvaluateSphericalHarmonicsIrradiance(const SphericalHarmonicsL2& harmonics,
                                                              const math::Vec3& normal) noexcept;

struct IrradianceFilterDesc
{
    bool useJobs = true;
};

// Cosine-convolves the environment into output, which holds irradiance in
// W/m^2, not radiance. A small face size is enough because the result is
// band-limited; 32 is the usual choice. Only level 0 is written and the chain
// is then regenerated. Returns InvalidArgument for an invalid input or output.
[[nodiscard]] core::Status<core::Error>
BuildDiffuseIrradianceCubemap(const Cubemap& environment, const IrradianceFilterDesc& desc,
                              Cubemap& output, jobs::JobSystem* jobSystem = nullptr) noexcept;

struct SpecularPrefilterDesc
{
    // GGX importance taps per texel, in [1, 4096].
    u32 sampleCount = 64;
    bool useJobs = true;
};

// Roughness assigned to a prefiltered level: level / (mipLevels - 1), so level
// 0 is a mirror and the last level is fully rough.
[[nodiscard]] f32 SpecularMipRoughness(u32 level, u32 mipLevels) noexcept;
// Inverse mapping, clamped to [0, mipLevels - 1].
[[nodiscard]] f32 SpecularRoughnessMip(f32 roughness, u32 mipLevels) noexcept;

// Convolves the environment with the GGX distribution once per level. Level 0
// is copied, so a mirror reflection is exact rather than approximated. Higher
// levels importance-sample the half-vector distribution and fetch the source
// through a solid-angle-matched mip level, which is what keeps a low tap count
// free of fireflies. Returns InvalidArgument for an invalid input, an invalid
// output, a single-level output, or an out-of-range sample count.
[[nodiscard]] core::Status<core::Error>
PrefilterSpecularCubemap(const Cubemap& environment, const SpecularPrefilterDesc& desc,
                         Cubemap& output, jobs::JobSystem* jobSystem = nullptr) noexcept;

struct BrdfLutDesc
{
    // Edge length in texels, in [8, 512]. Both axes use the same size.
    i32 size = 64;
    // Taps per texel, in [16, 8192].
    u32 sampleCount = 512;
    bool useJobs = true;
};

// Builds the split-sum environment BRDF table. The red channel is the scale
// applied to F0 and the green channel is the bias added for grazing angles. U
// is N.V in (0, 1] and V is roughness in [0, 1]. output is replaced by a
// freshly created RGBA32F texture without mips. Returns InvalidArgument for an
// out-of-range descriptor and Internal when allocation fails.
[[nodiscard]] core::Status<core::Error>
BuildSplitSumBrdfLut(const BrdfLutDesc& desc, Texture& output, memory::Allocator& allocator,
                     jobs::JobSystem* jobSystem = nullptr) noexcept;

// Bilinear table lookup returning (scale, bias). A null or invalid table falls
// back to the Karis analytic fit, so lighting stays usable without
// precomputation.
[[nodiscard]] math::Vec2 SampleSplitSumBrdf(const Texture* lut, f32 ndotv, f32 roughness) noexcept;

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

// Surface inputs for an environment lookup. This mirrors the fields of
// PbrMaterial that image-based lighting needs, without depending on it.
struct IblSurface
{
    math::Vec3 normal{0.0f, 1.0f, 0.0f};
    // Points from the surface toward the eye.
    math::Vec3 viewDirection{0.0f, 1.0f, 0.0f};
    math::Vec3 baseColor{1.0f, 1.0f, 1.0f};
    f32 metallic = 0.0f;
    f32 roughness = 0.5f;
    // Reflectance at normal incidence for dielectrics; 0.04 is 4% and matches
    // the direct lighting path.
    f32 reflectance = 0.04f;
    // Visibility of the hemisphere, from an occlusion texture or a bake.
    f32 ambientOcclusion = 1.0f;
    // Irradiance, in W/m^2, that answers in place of the environment's diffuse
    // lookup when useIrradianceOverride is set. This is how baked global
    // illumination enters the image-based path: the volume supplies the
    // irradiance and the environment keeps the specular term, and the diffuse
    // lobe still receives only the energy the specular lobes leave. The value
    // is in scene units, so EnvironmentLighting::intensity is not applied to it.
    math::Vec3 irradianceOverride{};
    bool useIrradianceOverride = false;
};

struct EnvironmentLighting
{
    // Prefiltered GGX chain. Null disables the specular term.
    const Cubemap* specular = nullptr;
    // Cosine-convolved irradiance, in W/m^2. Preferred over diffuseSh.
    const Cubemap* diffuseIrradiance = nullptr;
    // Used when diffuseIrradiance is null.
    const SphericalHarmonicsL2* diffuseSh = nullptr;
    // Null falls back to the analytic environment-BRDF fit.
    const Texture* brdfLut = nullptr;
    // Linear multiplier applied to both terms.
    f32 intensity = 1.0f;
    // Derive specular occlusion from ambientOcclusion so a reflection does not
    // leak into a crevice the diffuse term already darkens.
    bool applySpecularOcclusion = true;
};

struct EnvironmentMapDesc
{
    // Edge length of the captured source and prefiltered specular cubemaps.
    i32 faceSize = 128;
    // Diffuse irradiance is band-limited; a small cube is sufficient.
    i32 diffuseFaceSize = 32;
    // Edge length of the split-sum BRDF lookup texture.
    i32 brdfLutSize = 64;
    u32 equirectangularSamplesPerAxis = 2;
    u32 specularSampleCount = 128;
    u32 brdfSampleCount = 512;
    bool useJobs = true;
};

// Owns all precomputed resources for an image-based lighting environment.
// The source, specular, irradiance and BRDF data share one allocator and stay
// alive for as long as Lighting()'s non-owning pointers are used.
class EnvironmentMap
{
public:
    EnvironmentMap() noexcept = default;
    EnvironmentMap(const EnvironmentMap&) = delete;
    EnvironmentMap& operator=(const EnvironmentMap&) = delete;
    EnvironmentMap(EnvironmentMap&&) noexcept = default;
    EnvironmentMap& operator=(EnvironmentMap&&) noexcept = default;

    // Builds every lighting resource from a linear-RGB equirectangular image.
    // The returned value is published only after projection and all
    // precomputation succeed.
    [[nodiscard]] static core::Result<EnvironmentMap, core::Error>
    Create(const Framebuffer& equirectangular, const EnvironmentMapDesc& desc,
           memory::Allocator& allocator, jobs::JobSystem* jobSystem = nullptr) noexcept;

    // Builds every lighting resource from a linear-radiance cubemap. Only the
    // base level is consumed; source mips are regenerated before filtering.
    [[nodiscard]] static core::Result<EnvironmentMap, core::Error>
    Create(const Cubemap& source, const EnvironmentMapDesc& desc, memory::Allocator& allocator,
           jobs::JobSystem* jobSystem = nullptr) noexcept;

    // Loads a supported image with LoadImage (including Radiance RGBE HDR),
    // then creates its precomputed environment resources.
    [[nodiscard]] static core::Result<EnvironmentMap, core::Error>
    Load(const char* utf8Path, const EnvironmentMapDesc& desc, memory::Allocator& allocator,
         jobs::JobSystem* jobSystem = nullptr) noexcept;

    [[nodiscard]] bool IsValid() const noexcept;
    // Borrowed lighting view. The EnvironmentMap must outlive every use of it.
    [[nodiscard]] EnvironmentLighting Lighting() const noexcept;
    [[nodiscard]] const Cubemap& Source() const noexcept;
    [[nodiscard]] const Cubemap& Specular() const noexcept;
    [[nodiscard]] const Cubemap& DiffuseIrradiance() const noexcept;
    [[nodiscard]] const Texture& BrdfLut() const noexcept;

private:
    [[nodiscard]] static core::Result<EnvironmentMap, core::Error>
    CreateFromSource(Cubemap&& source, const EnvironmentMapDesc& desc, memory::Allocator& allocator,
                     jobs::JobSystem* jobSystem) noexcept;

    Cubemap m_source;
    Cubemap m_specular;
    Cubemap m_diffuseIrradiance;
    Texture m_brdfLut;
};

// Specular visibility derived from hemispherical occlusion. A rough lobe covers
// most of the hemisphere and is occluded nearly as much as the diffuse term; a
// mirror lobe is narrow and is barely occluded. The result is in [0, 1].
[[nodiscard]] f32 ComputeSpecularOcclusion(f32 ndotv, f32 ambientOcclusion, f32 roughness) noexcept;

// Diffuse environment term, as outgoing radiance in W/(m^2 sr).
[[nodiscard]] math::Vec3 EvaluateIblDiffuse(const EnvironmentLighting& environment,
                                            const IblSurface& surface) noexcept;

// Specular environment term along the mirror direction.
[[nodiscard]] math::Vec3 EvaluateIblSpecular(const EnvironmentLighting& environment,
                                             const IblSurface& surface) noexcept;

// Specular environment term along a caller-supplied direction, which is how a
// parallax-corrected probe or a screen-space trace redirects the lookup.
[[nodiscard]] math::Vec3 EvaluateIblSpecularDirection(const EnvironmentLighting& environment,
                                                      const IblSurface& surface,
                                                      const math::Vec3& reflectDirection) noexcept;

// Sum of both terms.
[[nodiscard]] math::Vec3 EvaluateImageBasedLighting(const EnvironmentLighting& environment,
                                                    const IblSurface& surface) noexcept;

// Applies the single-scattering split-sum specular weight to a radiance value.
// This is how a radiance from another source, such as a screen-space trace,
// becomes a specular contribution on the same footing as the environment path.
// The environment's cubemaps are not read and its intensity is not applied,
// because the supplied radiance is already in scene units.
[[nodiscard]] math::Vec3 ApplySpecularEnvironmentWeight(const EnvironmentLighting& environment,
                                                        const IblSurface& surface,
                                                        const math::Vec3& radiance) noexcept;

// ---------------------------------------------------------------------------
// Reflection probes
// ---------------------------------------------------------------------------

inline constexpr u32 kMaxReflectionProbes = 16;

struct ReflectionProbe
{
    // Capture position, which is the origin of the parallax proxy.
    math::Vec3 position{};
    // Axis-aligned region the probe affects.
    math::Vec3 influenceMin{};
    math::Vec3 influenceMax{};
    // Box the environment is reprojected onto. An empty box disables parallax
    // correction and the probe behaves like an infinite environment.
    math::Vec3 proxyMin{};
    math::Vec3 proxyMax{};
    // Distance inside the influence box over which the weight ramps from 0 to
    // 1. Zero produces a hard boundary, which is visible as a seam.
    f32 blendDistance = 0.0f;
    EnvironmentLighting environment{};
};

struct ReflectionProbeSet
{
    memory::Span<const ReflectionProbe> probes{};
    // Takes whatever weight the probes leave, so a surface outside every probe
    // still reflects the sky.
    const EnvironmentLighting* fallback = nullptr;
};

struct ReflectionProbeCaptureDesc
{
    math::Vec3 position{};
    // Resolved output cubemap face edge length.
    i32 faceSize = 128;
    // Capture every face at faceSize * supersampleFactor, then resolve in
    // linear light before publishing the cubemap.
    i32 supersampleFactor = 1;
    ResolveFilter resolveFilter = kDefaultResolveFilter;
    f32 nearPlane = 0.1f;
    f32 farPlane = 1000.0f;
};

struct ReflectionProbeCaptureView
{
    CubeFace face = CubeFace::PositiveX;
    math::Vec3 position{};
    math::Mat4 view = math::Mat4::Identity();
    math::Mat4 projection = math::Mat4::Identity();
};

using ReflectionProbeCaptureFn =
    core::Status<core::Error> (*)(const ReflectionProbeCaptureView& view, Framebuffer& color,
                                  DepthBuffer& depth, void* userData) noexcept;

// Renders the caller's scene once for each cube face into linear RGBA32F and
// reverse-Z depth targets. The callback receives the face camera and is
// responsible for rendering the scene into the supplied, cleared targets.
// When supersampleFactor is greater than one, the callback targets are scaled
// by that factor and each face is resolved to faceSize before the cubemap is
// returned. The targets are reused synchronously and must not be retained by
// the callback. The cubemap is returned only if all six callbacks succeed.
[[nodiscard]] core::Result<Cubemap, core::Error>
CaptureReflectionProbe(const ReflectionProbeCaptureDesc& desc, ReflectionProbeCaptureFn renderFace,
                       void* userData, memory::Allocator& allocator) noexcept;

// Smooth influence weight in [0, 1]. It is 1 deeper than blendDistance inside
// the box, 0 outside it, and a smoothstep in between. The ramp uses the nearest
// box face, so the weight is continuous across an edge or a corner.
[[nodiscard]] f32 EvaluateReflectionProbeWeight(const ReflectionProbe& probe,
                                                const math::Vec3& worldPosition) noexcept;

// Reprojects a reflection ray onto the probe's proxy box and returns the
// direction from the capture position to that intersection, which is what makes
// a probe look correct away from its capture point. An empty proxy box or a
// miss returns reflectDirection unchanged.
[[nodiscard]] math::Vec3 ApplyBoxParallaxCorrection(const ReflectionProbe& probe,
                                                    const math::Vec3& worldPosition,
                                                    const math::Vec3& reflectDirection) noexcept;

// Blends every overlapping probe by normalized weight, with the fallback taking
// the remainder, and returns outgoing radiance. Weights that sum above 1 are
// normalized, so a surface inside two probes is never brighter than one inside
// a single probe.
[[nodiscard]] math::Vec3 EvaluateReflectionProbes(const ReflectionProbeSet& probes,
                                                  const math::Vec3& worldPosition,
                                                  const IblSurface& surface) noexcept;

// ---------------------------------------------------------------------------
// Screen-space reflections
// ---------------------------------------------------------------------------

inline constexpr u32 kMaxHiZLevels = 16;

// Reverse-Z maximum-depth pyramid. Level 0 copies the source depth buffer and
// each level above it holds the closest depth of the four texels below it,
// which is the conservative bound a ray march needs to skip empty space.
class HiZPyramid
{
public:
    HiZPyramid() noexcept = default;
    ~HiZPyramid();

    HiZPyramid(const HiZPyramid&) = delete;
    HiZPyramid& operator=(const HiZPyramid&) = delete;

    HiZPyramid(HiZPyramid&& other) noexcept;
    HiZPyramid& operator=(HiZPyramid&& other) noexcept;

    // Returns InvalidArgument for an invalid depth buffer and Internal when
    // allocation fails.
    [[nodiscard]] static core::Result<HiZPyramid, core::Error>
    Build(const DepthBuffer& depth, memory::Allocator& allocator) noexcept;

    [[nodiscard]] bool IsValid() const noexcept;
    [[nodiscard]] i32 Width() const noexcept;
    [[nodiscard]] i32 Height() const noexcept;
    [[nodiscard]] u32 LevelCount() const noexcept;
    [[nodiscard]] i32 LevelWidth(u32 level) const noexcept;
    [[nodiscard]] i32 LevelHeight(u32 level) const noexcept;
    // Closest reverse-Z depth in the covered footprint. Out-of-range reads
    // return 0, the cleared far value.
    [[nodiscard]] f32 Depth(u32 level, i32 x, i32 y) const noexcept;

private:
    void Release() noexcept;
    void MoveFrom(HiZPyramid& other) noexcept;

    memory::Allocator* m_allocator = nullptr;
    f32* m_data = nullptr;
    usize m_sizeBytes = 0;
    usize m_levelOffsets[kMaxHiZLevels] = {};
    i32 m_width = 0;
    i32 m_height = 0;
    u32 m_levelCount = 0;
};

struct ScreenSpaceReflectionDesc
{
    // Linear HDR scene color of the current frame.
    const Framebuffer* color = nullptr;
    const DepthBuffer* depth = nullptr;
    // Optional acceleration. Without it the march is a fixed-step linear walk.
    // Both paths accept the same hits; the pyramid only skips empty tiles.
    const HiZPyramid* hiZ = nullptr;
    // The trace assumes the depth buffer covers the whole viewport, which is
    // what a full-target render produces.
    math::Mat4 clipFromWorld = math::Mat4::Identity();
    // Reverse-Z depth linearization, matching the projection in clipFromWorld.
    bool perspectiveProjection = true;
    f32 nearPlane = 0.1f;
    f32 farPlane = 1000.0f;
    // Above this roughness the trace is skipped and probes answer alone.
    f32 maxRoughness = 0.6f;
    // World-space interval behind a depth sample that still counts as a hit.
    f32 thickness = 0.25f;
    f32 maxWorldDistance = 50.0f;
    u32 maxSteps = 64;
    // Fraction of the screen over which confidence fades at the border.
    f32 edgeFadeFraction = 0.1f;
};

struct ScreenSpaceReflectionSample
{
    math::Vec3 radiance{};
    // 0 means unusable and 1 means fully trusted. Partial values come from the
    // screen border, a grazing ray, or a roughness close to maxRoughness.
    f32 confidence = 0.0f;
    bool hit = false;
    // Depth samples the march consumed, which is what the Hi-Z pyramid lowers.
    u32 steps = 0;
};

// Marches the reflection ray through the depth buffer. The march runs in screen
// space, so it is resolution-bound rather than scene-bound. A hit is a
// crossing: the ray was in front of a surface and is now behind it by less than
// thickness, which is what stops a ray that leaves nearly parallel to its own
// surface from re-finding it. A ray that leaves the screen, exceeds
// maxWorldDistance, or never crosses a surface returns zero confidence.
[[nodiscard]] ScreenSpaceReflectionSample
TraceScreenSpaceReflection(const ScreenSpaceReflectionDesc& desc, const math::Vec3& worldPosition,
                           const math::Vec3& reflectDirection) noexcept;

// Screen-space reflections composited over the probe result by confidence. This
// is the function a pixel shader calls: it needs no knowledge of which source
// answered.
[[nodiscard]] math::Vec3 ResolveReflections(const ScreenSpaceReflectionDesc& desc,
                                            const ReflectionProbeSet& probes,
                                            const math::Vec3& worldPosition,
                                            const IblSurface& surface) noexcept;

// ---------------------------------------------------------------------------
// Procedural sky
// ---------------------------------------------------------------------------

// Single-scattering Rayleigh and Mie atmosphere. Lengths are in meters and the
// defaults are Earth at sea level, so the model produces a blue zenith, a warm
// horizon at sunset, and a sun disk of the correct angular size.
struct ProceduralSkyConfig
{
    // Direction the sunlight travels, so a downward vector is noon.
    math::Vec3 sunDirection{0.0f, -1.0f, 0.0f};
    // Top-of-atmosphere spectral irradiance, in W/m^2.
    math::Vec3 sunIrradiance{22.0f, 21.0f, 19.0f};
    f32 sunAngularRadiusRadians = 0.00465f;
    // Scattering coefficients at sea level, per meter.
    math::Vec3 rayleighScattering{5.802e-6f, 13.558e-6f, 33.1e-6f};
    f32 mieScattering = 3.996e-6f;
    f32 mieExtinction = 4.44e-6f;
    // Henyey-Greenstein anisotropy of the Mie phase function, in [-0.9, 0.9].
    f32 mieAnisotropy = 0.76f;
    f32 rayleighScaleHeightMeters = 8000.0f;
    f32 mieScaleHeightMeters = 1200.0f;
    f32 planetRadiusMeters = 6360000.0f;
    f32 atmosphereRadiusMeters = 6420000.0f;
    f32 viewAltitudeMeters = 500.0f;
    // Lambertian albedo of the ground below the horizon.
    math::Vec3 groundAlbedo{0.1f, 0.1f, 0.1f};
    // Ray-march steps along the view ray, in [2, 128].
    u32 viewSampleCount = 24;
    // Steps along each sun-transmittance ray, in [2, 64].
    u32 lightSampleCount = 8;
    // Include the sun disk itself. A capture used for lighting keeps it, and
    // the specular prefilter then spreads it correctly.
    bool includeSunDisk = true;
};

// Sky radiance along a view direction, in W/(m^2 sr). A direction below the
// horizon returns the ground, lit by the transmitted sun and the zenith sky.
[[nodiscard]] math::Vec3 EvaluateProceduralSky(const ProceduralSkyConfig& config,
                                               const math::Vec3& viewDirection) noexcept;

// Sun irradiance that reaches the view altitude, in W/m^2, which is what a
// matching PbrDirectionalLight should use so the sky and the direct light agree.
[[nodiscard]] math::Vec3 EvaluateSunTransmittance(const ProceduralSkyConfig& config) noexcept;

// Renders the sky into level 0 of output and regenerates its chain. Returns
// InvalidArgument for an invalid output or an out-of-range sample count.
[[nodiscard]] core::Status<core::Error>
RenderProceduralSkyCubemap(const ProceduralSkyConfig& config, Cubemap& output,
                           jobs::JobSystem* jobSystem = nullptr) noexcept;

}  // namespace cyber::software
