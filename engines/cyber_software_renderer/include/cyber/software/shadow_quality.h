#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/math/vec3.h>
#include <cyber/memory/span.h>
#include <cyber/software/shadow.h>

namespace cyber::software
{

using cyber::core::f32;
using cyber::core::u32;

// Scene-agnostic shadow-artifact measurements over the software shadow
// sampler. Each function evaluates caller-supplied probes through the same
// `EvaluateDirectionalShadowVisibility` path the PBR shader uses, so a metric
// reflects the shipped sampler rather than a parallel implementation. The
// functions allocate nothing, read only borrowed data, and are deterministic.
//
// The four metrics map onto the four artifacts a shadow implementation is
// judged by: self-shadowing acne, detached (peter-panning) contact shadows,
// cascade-boundary seams, and camera-motion shimmer. A caller turns a metric
// into an acceptance threshold; this header defines no scene and no limit.

// One receiver sample. `viewDepth` is the positive camera-space depth used for
// cascade selection, matching `PbrLightingData::cameraForward` projection.
struct ShadowProbe
{
    math::Vec3 worldPosition;
    math::Vec3 worldNormal{0.0f, 1.0f, 0.0f};
    // Unit vector from the receiver toward the light.
    math::Vec3 directionToLight{0.0f, 1.0f, 0.0f};
    f32 viewDepth = 0.0f;
};

// Acne is spurious self-shadowing: a receiver that no geometry occludes
// sampling as shadowed because its own depth loses the comparison. The caller
// supplies probes it knows are unoccluded.
struct ShadowAcneMetrics
{
    u32 sampleCount = 0;
    // Probes whose visibility fell below `visibilityThreshold`.
    u32 darkenedSamples = 0;
    // Fraction of probes below the threshold, in [0, 1].
    f32 darkenedFraction = 0.0f;
    // Lowest and mean visibility across all probes.
    f32 minVisibility = 1.0f;
    f32 meanVisibility = 1.0f;
    // Largest run of consecutive darkened probes in input order. Acne appears
    // as isolated speckle, so a long run indicates a genuine occluder in the
    // probe set rather than bias error.
    u32 longestDarkenedRun = 0;
};

// Measures false shadowing across probes the caller knows are lit. Returns
// InvalidArgument for an empty probe set, a non-finite probe, or a threshold
// outside (0, 1].
[[nodiscard]] core::Status<core::Error>
MeasureShadowAcne(const DirectionalShadowCascadeSet& cascades,
                  cyber::memory::Span<const ShadowProbe> litProbes, f32 visibilityThreshold,
                  ShadowAcneMetrics& output) noexcept;

// A detached shadow (peter-panning) is a lit gap between an occluder's contact
// point and where its shadow begins, caused by excessive depth bias or a
// normal offset that pushes the receiver past the occluder.
struct ShadowContactMetrics
{
    // Visibility sampled at the contact point itself.
    f32 contactVisibility = 1.0f;
    // Distance from the contact point to the first probe that sampled lit,
    // measured along the walk direction in world units. Equals the full walk
    // length when every probe stayed shadowed.
    f32 detachmentDistance = 0.0f;
    // True when the walk found a lit probe, so `detachmentDistance` is a
    // measured gap rather than the walk bound.
    bool foundLitSample = false;
    // Probe index of the first lit sample, or sampleCount when none was found.
    u32 firstLitIndex = 0;
    u32 sampleCount = 0;
};

// Walks `sampleCount` probes from the contact point along `direction` in
// `stepDistance` world-unit increments and reports where visibility first
// exceeds `litThreshold`. The walk direction should lie in the receiver
// surface and point from the contact point into the shadow, so a correctly
// grounded shadow stays dark for its whole penumbra-free extent.
//
// Returns InvalidArgument for a zero sample count, a non-positive step, a
// degenerate direction, a non-finite input, or a threshold outside (0, 1].
[[nodiscard]] core::Status<core::Error>
MeasureShadowContact(const DirectionalShadowCascadeSet& cascades, const ShadowProbe& contactProbe,
                     const math::Vec3& direction, f32 stepDistance, u32 sampleCount,
                     f32 litThreshold, ShadowContactMetrics& output) noexcept;

// A cascade seam is a visibility step where a receiver crosses a split
// boundary: the two cascades disagree and the blend fails to hide it. The
// probes form a transect ordered by increasing view depth.
struct ShadowTransectMetrics
{
    u32 sampleCount = 0;
    // Largest absolute visibility change between consecutive probes.
    f32 maxStep = 0.0f;
    // Index of the probe where `maxStep` was measured, so a caller can report
    // the view depth at which the worst discontinuity occurs.
    u32 maxStepIndex = 0;
    // Mean absolute change, which separates one sharp seam from a transect
    // that is noisy throughout.
    f32 meanStep = 0.0f;
    // Largest step measured at a probe pair that straddles a cascade split.
    f32 maxStepAcrossSplit = 0.0f;
    // Number of consecutive probe pairs that straddle a split.
    u32 splitCrossings = 0;
};

// Measures visibility continuity along a transect. Probes must be ordered by
// non-decreasing `viewDepth`; the transect should cross every cascade split so
// `maxStepAcrossSplit` is meaningful. Returns InvalidArgument for fewer than
// two probes, a non-finite probe, or probes that are not ordered by view
// depth.
[[nodiscard]] core::Status<core::Error>
MeasureShadowTransect(const DirectionalShadowCascadeSet& cascades,
                      cyber::memory::Span<const ShadowProbe> transect,
                      ShadowTransectMetrics& output) noexcept;

// Shimmer is temporal instability: a static receiver whose visibility flickers
// as the camera moves, because the shadow map re-snaps or the selected cascade
// changes. Each frame contributes one visibility value per probe.
struct ShadowStabilityMetrics
{
    u32 probeCount = 0;
    u32 frameCount = 0;
    // Largest absolute change in one probe between consecutive frames.
    f32 maxFrameDelta = 0.0f;
    u32 maxFrameDeltaProbe = 0;
    // Frame index whose change from its predecessor produced `maxFrameDelta`.
    u32 maxFrameDeltaFrame = 0;
    // Mean absolute frame-to-frame change over every probe and frame pair.
    f32 meanFrameDelta = 0.0f;
    // Largest total variation summed over all frames for a single probe. A
    // probe that settles has small total variation; one that flickers
    // accumulates it even when each step is small.
    f32 maxTotalVariation = 0.0f;
};

// Analyzes a visibility sequence laid out as `frameCount` consecutive rows of
// `probeCount` values, which is what a caller fills by sampling the same
// probes once per rendered frame. Returns InvalidArgument for a zero probe or
// frame count, a span whose size is not probeCount * frameCount, or a
// non-finite value.
[[nodiscard]] core::Status<core::Error>
MeasureShadowStability(cyber::memory::Span<const f32> visibilitySequence, u32 probeCount,
                       u32 frameCount, ShadowStabilityMetrics& output) noexcept;

}  // namespace cyber::software
