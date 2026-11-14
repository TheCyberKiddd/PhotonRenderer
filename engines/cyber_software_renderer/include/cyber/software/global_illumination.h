#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/math/vec2.h>
#include <cyber/math/vec3.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/span.h>
#include <cyber/software/ibl.h>
#include <cyber/software/raytrace.h>

namespace cyber::jobs
{
class JobSystem;
}

namespace cyber::software
{

using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::u8;
using cyber::core::usize;

// Baked global illumination: a CPU path tracer, and the irradiance volume it
// bakes into.
//
// ADR 0028 records why this is the method shipped first. In short: a regular
// grid of spherical-harmonic irradiance probes lights static and dynamic
// geometry by one mechanism, needs no lightmap UVs, runs on every adapter, and
// reuses the order-2 basis and irradiance convolution Phase 29 established.
//
// Units match the rest of the renderer: radiance in W/(m^2 sr), irradiance in
// W/m^2, all linear RGB. A volume baked from a set of lights and sampled by the
// forward path reproduces the indirect light those same lights produce.
//
// Everything is deterministic. Every random decision draws from a counter-based
// hash of (sequence, sample, dimension), so a bake depends only on its inputs
// and is byte-identical with and without a job system, whatever the worker
// count.

// ---------------------------------------------------------------------------
// Scene description for light transport
// ---------------------------------------------------------------------------

// The bake models every surface as Lambertian. Specular interreflection is a
// small fraction of indirect diffuse energy in the scenes this renderer
// targets, and modelling it would make the bake view-dependent, which a probe
// cannot represent anyway.
struct GiSurfaceMaterial
{
    // Diffuse reflectance in [0, 1] per channel. Values outside that range are
    // clamped, because an albedo above one creates energy and diverges.
    math::Vec3 albedo{0.8f, 0.8f, 0.8f};
    // Emitted radiance, in W/(m^2 sr), from the front face only.
    math::Vec3 emission{0.0f, 0.0f, 0.0f};
};

struct GiDirectionalLight
{
    // Direction the light travels.
    math::Vec3 direction{-0.5f, -1.0f, -0.3f};
    // Irradiance on a surface facing the light, in W/m^2. This is numerically
    // the value PbrDirectionalLight::radiance carries, so the same number in
    // both places lights the direct and indirect paths consistently.
    math::Vec3 irradiance{1.0f, 1.0f, 1.0f};
};

struct GiPointLight
{
    math::Vec3 position{};
    // Radiant intensity in W/sr, falling off with the inverse square of the
    // distance and the same smooth range window PbrPointLight uses.
    math::Vec3 intensity{1.0f, 1.0f, 1.0f};
    f32 range = 10.0f;
};

inline constexpr u32 kMaxGiDirectionalLights = 4;
inline constexpr u32 kMaxGiPointLights = 64;

// Everything the integrator reads. All members are borrowed and must outlive
// every call that receives the scene.
struct GiScene
{
    const TopLevelAccelerationStructure* geometry = nullptr;
    // Indexed by RayHit::instanceId. An identifier with no entry uses the
    // default material, so a missing table produces a grey world rather than a
    // crash.
    memory::Span<const GiSurfaceMaterial> materials{};
    memory::Span<const GiDirectionalLight> directionalLights{};
    memory::Span<const GiPointLight> pointLights{};
    // Radiance of rays that leave the scene. When skyCubemap is set it answers
    // and skyRadiance is ignored. The sky should not contain a sun disk that a
    // directional light also represents, or that light is counted twice.
    const Cubemap* skyCubemap = nullptr;
    math::Vec3 skyRadiance{0.0f, 0.0f, 0.0f};
    // Multiplies the default secondary-ray origin offset. Raise it for a scene
    // modelled at a large scale.
    f32 rayOffsetScale = 1.0f;
};

// Validates a scene description: geometry present and valid, light counts in
// range, finite light values, a non-negative offset scale. Returns
// InvalidArgument naming the first problem.
[[nodiscard]] core::Status<core::Error> ValidateGiScene(const GiScene& scene) noexcept;

// ---------------------------------------------------------------------------
// Path tracing
// ---------------------------------------------------------------------------

struct PathTraceSettings
{
    // Surface interactions after the first, in [0, 32]. Zero is direct light
    // only, which is the control configuration the acceptance suite uses to
    // prove the bake carries interreflection.
    u32 maxBounces = 3;
    // Russian roulette terminates low-throughput paths without bias by
    // reweighting the survivors. It starts after this many bounces.
    bool russianRoulette = true;
    u32 rouletteStartBounce = 2;
    // Clamp on the radiance one path may contribute, which trades a small,
    // bounded bias for the removal of fireflies. Zero disables the clamp.
    f32 maxPathRadiance = 0.0f;
};

struct PathTraceStats
{
    u64 rays = 0;
    u64 shadowRays = 0;
    u64 terminatedByRoulette = 0;
    RayTraversalStats traversal{};

    void Add(const PathTraceStats& other) noexcept
    {
        rays += other.rays;
        shadowRays += other.shadowRays;
        terminatedByRoulette += other.terminatedByRoulette;
        traversal.Add(other.traversal);
    }
};

// Radiance arriving at ray.origin from along the ray, in W/(m^2 sr): the light
// leaving the first surface the ray hits back toward the origin, or the sky on
// a miss.
//
// Direct light from directional and point lights is gathered at every surface
// interaction by next-event estimation with an any-hit shadow ray. Emission and
// the sky are gathered when a path hits them, and are not light-sampled, so
// neither is ever counted twice. Indirect bounces sample the cosine-weighted
// hemisphere, for which the Lambertian estimator reduces to albedo times the
// incoming radiance. `sequence` and `sample` select the random stream; the same
// pair always returns the same value.
[[nodiscard]] math::Vec3 TracePathRadiance(const GiScene& scene, const RayDesc& ray,
                                           const PathTraceSettings& settings, u32 sequence,
                                           u32 sample, PathTraceStats* stats = nullptr) noexcept;

// Irradiance at a surface point, in W/m^2, by cosine-weighted Monte Carlo over
// the hemisphere around `normal`: pi times the mean incoming radiance. With
// sampleCount around a thousand this is the ground truth the irradiance volume
// is judged against.
[[nodiscard]] math::Vec3 EstimateIrradiance(const GiScene& scene, const math::Vec3& position,
                                            const math::Vec3& normal,
                                            const PathTraceSettings& settings, u32 sampleCount,
                                            u32 sequence, PathTraceStats* stats = nullptr) noexcept;

// ---------------------------------------------------------------------------
// Irradiance volume
// ---------------------------------------------------------------------------

inline constexpr u32 kMaxIrradianceProbesPerAxis = 128;
inline constexpr u32 kMaxIrradianceProbes = 1u << 18;

// A probe keeps the moments of the distance to the nearest surface in an
// octahedral map of this many texels per side, so each texel covers about 12
// degrees of the sphere. A single moment per probe cannot tell a wall on one
// side from open space on the other, which is the whole of what leak
// suppression needs to know.
//
// The resolution was measured, not chosen: six axis lobes leaked through a
// shared wall at 7% mean, because the lobe toward the wall also averages the
// diagonal rays that run past it to the floor and ceiling; an 8x8 map cut that
// to 1.7% but still leaked at oblique angles, where its bilinear footprint
// spans 35 degrees and reaches a texel looking down the room. 16x16 is the
// resolution DDGI uses for the same moments, and the acceptance suite's leak
// test passes at it.
inline constexpr u32 kIrradianceVisibilityResolution = 16;
inline constexpr u32 kIrradianceVisibilityTexels =
    kIrradianceVisibilityResolution * kIrradianceVisibilityResolution;

// Octahedral mapping of a unit direction to [0, 1]^2 and back. The +Z
// hemisphere fills the inner diamond and the -Z hemisphere folds into the
// corners. Exposed so a debug view and the tests use the same mapping as the
// bake.
[[nodiscard]] math::Vec2 EncodeOctahedral(const math::Vec3& direction) noexcept;
[[nodiscard]] math::Vec3 DecodeOctahedral(const math::Vec2& uv) noexcept;

enum class IrradianceProbeState : u8
{
    // Created but not baked. Sampling ignores it.
    Unbaked = 0,
    // Baked and usable.
    Valid,
    // Baked, but it sits inside geometry: enough of its rays hit back faces
    // that what it saw is the inside of a wall. Sampling ignores it, because
    // its irradiance is black and its visibility is meaningless.
    Inside,
};

struct IrradianceProbe
{
    // Incident radiance projected onto order-2 spherical harmonics. Irradiance
    // for a normal comes from EvaluateSphericalHarmonicsIrradiance, the same
    // convolution an environment's SH uses.
    SphericalHarmonicsL2 radiance{};
    // Mean and mean square of the distance to the first surface, per
    // octahedral texel, in world units, clamped to the bake's maximum
    // visibility distance. Texels are row-major, u fastest.
    f32 meanDistance[kIrradianceVisibilityTexels] = {};
    f32 meanDistanceSquared[kIrradianceVisibilityTexels] = {};
    // Fraction of the probe's rays that hit a back face.
    f32 backfaceFraction = 0.0f;
    IrradianceProbeState state = IrradianceProbeState::Unbaked;
};

struct IrradianceVolumeDesc
{
    // World-space box the grid spans. Probes sit on its corners and on a
    // regular lattice between them.
    math::Vec3 boundsMin{-1.0f, -1.0f, -1.0f};
    math::Vec3 boundsMax{1.0f, 1.0f, 1.0f};
    // Probes per axis, each in [2, kMaxIrradianceProbesPerAxis], with the
    // total at most kMaxIrradianceProbes.
    u32 probeCountX = 4;
    u32 probeCountY = 4;
    u32 probeCountZ = 4;
};

// A grid of irradiance probes, allocator-owned and move-only.
class IrradianceVolume
{
public:
    IrradianceVolume() noexcept = default;
    ~IrradianceVolume();

    IrradianceVolume(const IrradianceVolume&) = delete;
    IrradianceVolume& operator=(const IrradianceVolume&) = delete;

    IrradianceVolume(IrradianceVolume&& other) noexcept;
    IrradianceVolume& operator=(IrradianceVolume&& other) noexcept;

    // Allocates every probe in the Unbaked state. Returns InvalidArgument for a
    // box that is empty or non-finite on any axis or a probe count out of
    // range, and Internal when allocation fails.
    [[nodiscard]] static core::Result<IrradianceVolume, core::Error>
    Create(const IrradianceVolumeDesc& desc, memory::Allocator& allocator) noexcept;

    [[nodiscard]] bool IsValid() const noexcept;
    [[nodiscard]] const IrradianceVolumeDesc& Desc() const noexcept;
    [[nodiscard]] u32 ProbeCount() const noexcept;
    // Distance between neighbouring probes along each axis.
    [[nodiscard]] math::Vec3 ProbeSpacing() const noexcept;
    [[nodiscard]] usize DataSizeBytes() const noexcept;
    // Allocator the probes live in, which is also where a bake takes its
    // scratch storage from. Null for an invalid volume.
    [[nodiscard]] memory::Allocator* GetAllocator() const noexcept;

    // Linear index of grid coordinate (x, y, z), x fastest. kInvalidRayIndex for
    // an out-of-range coordinate.
    [[nodiscard]] u32 ProbeIndex(u32 x, u32 y, u32 z) const noexcept;
    [[nodiscard]] math::Vec3 ProbePosition(u32 x, u32 y, u32 z) const noexcept;
    [[nodiscard]] math::Vec3 ProbePosition(u32 index) const noexcept;

    // Out-of-range indices return a shared Unbaked probe and ignore writes.
    [[nodiscard]] const IrradianceProbe& Probe(u32 index) const noexcept;
    void SetProbe(u32 index, const IrradianceProbe& probe) noexcept;

    // Marks every probe Unbaked and clears its data.
    void Clear() noexcept;

private:
    void Release() noexcept;
    void MoveFrom(IrradianceVolume& other) noexcept;

    memory::Allocator* m_allocator = nullptr;
    IrradianceProbe* m_probes = nullptr;
    usize m_sizeBytes = 0;
    u32 m_probeCount = 0;
    IrradianceVolumeDesc m_desc{};
};

struct IrradianceBakeDesc
{
    // Rays traced from each probe, in [16, 65536]. Directions follow a
    // spherical Fibonacci lattice, which covers the sphere evenly at any count,
    // so the SH projection has no clustering error to average away.
    u32 raysPerProbe = 256;
    PathTraceSettings path{};
    // A probe whose back-face fraction exceeds this is classified Inside.
    // A probe in open space sees essentially none; a probe buried in a wall
    // sees about half or more.
    f32 insideBackfaceThreshold = 0.25f;
    // Distances in the visibility moments are clamped here. Zero selects 1.5
    // times the largest probe spacing, which is the farthest a lookup ever
    // tests against a probe.
    f32 maxVisibilityDistance = 0.0f;
    // Rotates the ray lattice per probe by a deterministic amount, which
    // breaks up the structured aliasing an identical lattice at every probe
    // leaves in the interpolated result.
    bool rotatePerProbe = true;
    bool useJobs = true;
};

struct IrradianceBakeStats
{
    u32 probeCount = 0;
    u32 validProbes = 0;
    u32 insideProbes = 0;
    PathTraceStats path{};
};

// Bakes every probe in the volume. A probe's rays shade their hit points with
// the full path tracer, so the stored radiance includes the bounces the
// settings allow; projecting it onto SH and convolving with the cosine lobe at
// lookup then yields irradiance with one more bounce of light transport than
// the path carries.
//
// Returns InvalidArgument for an invalid scene, an invalid volume, or an
// out-of-range descriptor. The volume is modified only on success: probes are
// baked into scratch storage and copied in once every probe has finished.
[[nodiscard]] core::Status<core::Error>
BakeIrradianceVolume(const GiScene& scene, const IrradianceBakeDesc& desc, IrradianceVolume& volume,
                     IrradianceBakeStats* stats = nullptr,
                     jobs::JobSystem* jobSystem = nullptr) noexcept;

struct IrradianceVolumeSampleDesc
{
    // The lookup point is pushed this far along the surface normal before the
    // probe weights are computed. Without it a point on a wall tests visibility
    // from exactly on the wall and the Chebyshev test is ill-conditioned. Zero
    // selects a quarter of the smallest probe spacing.
    f32 normalBias = 0.0f;
    // Chebyshev visibility weighting from the stored distance moments. Turning
    // it off is the control configuration that shows light leaking through a
    // wall.
    bool useVisibility = true;
    // Down-weights probes behind the surface. Kept smooth rather than a hard
    // cut, because a hard cut makes the interpolation discontinuous as a
    // surface rotates past a probe.
    bool useNormalWeighting = true;
    // Irradiance returned where no usable probe contributes, in W/m^2: a point
    // outside the volume whose nearest cell holds only Inside probes, or an
    // unbaked volume.
    math::Vec3 fallbackIrradiance{0.0f, 0.0f, 0.0f};
};

// What a lookup did, for tests and for a debug view.
struct IrradianceVolumeSample
{
    math::Vec3 irradiance{};
    // Probes of the eight in the cell that contributed any weight.
    u32 contributingProbes = 0;
    // Sum of the final weights before normalization. Near zero means the
    // answer is barely supported and the caller may prefer another source.
    f32 totalWeight = 0.0f;
    bool usedFallback = false;
};

// Irradiance for a surface point and normal, in W/m^2. Points outside the box
// clamp to its nearest cell. The eight probes around the point are weighted by
// their trilinear weight, by the smooth normal term, and by the Chebyshev
// visibility of the point from the probe, read bilinearly from the probe's
// octahedral distance map in the direction of the point; Inside and
// Unbaked probes are skipped and the remaining weights renormalized.
[[nodiscard]] IrradianceVolumeSample
SampleIrradianceVolumeDetailed(const IrradianceVolume& volume, const math::Vec3& position,
                               const math::Vec3& normal,
                               const IrradianceVolumeSampleDesc& desc = {}) noexcept;

[[nodiscard]] math::Vec3
SampleIrradianceVolume(const IrradianceVolume& volume, const math::Vec3& position,
                       const math::Vec3& normal,
                       const IrradianceVolumeSampleDesc& desc = {}) noexcept;

// What the forward path holds to consume a baked volume. Borrowed: the volume
// must outlive every frame the view is used in.
struct IrradianceVolumeView
{
    const IrradianceVolume* volume = nullptr;
    IrradianceVolumeSampleDesc sampling{};
    // Linear multiplier on the sampled irradiance.
    f32 intensity = 1.0f;
};

// Evaluates a view. A null or invalid volume returns the view's fallback.
[[nodiscard]] math::Vec3 EvaluateIrradianceVolume(const IrradianceVolumeView& view,
                                                  const math::Vec3& position,
                                                  const math::Vec3& normal) noexcept;

}  // namespace cyber::software
