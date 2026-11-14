#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/math/vec3.h>
#include <cyber/memory/span.h>
#include <cyber/software/framebuffer.h>

namespace cyber::software
{

using cyber::core::f32;
using cyber::core::u32;

// Measurements for global illumination and ray-traced signals.
//
// The roadmap's bar for Phase 31 is that the chosen GI method "runs in the
// test scene". Running is not a test, so these functions turn the properties
// that make a GI result usable into numbers a suite can assert on, as
// shadow_quality.h and post_process_quality.h do for their phases:
//
//   accuracy  the baked irradiance agrees with a path-traced ground truth
//   leakage   light does not cross a wall the probe grid straddles
//   noise     a denoised ray-traced signal is smooth and still unbiased
//
// Every function reads borrowed data, allocates nothing, and is
// deterministic. The acceptance suite pairs each threshold with a control
// configuration that fails it.

// Agreement between estimated and reference irradiance at matching sample
// points.
struct IrradianceErrorMetrics
{
    u32 sampleCount = 0;
    // Relative luminance error |e - r| / max(r, floor), mean and worst.
    f32 meanRelativeError = 0.0f;
    f32 maxRelativeError = 0.0f;
    // Signed relative luminance error, mean. Negative means the estimate is
    // too dark, which is what a bake with too few bounces produces.
    f32 meanRelativeBias = 0.0f;
    // Ratio of summed estimated to summed reference luminance. One means the
    // estimate carries the right total energy, whatever its distribution.
    f32 energyRatio = 0.0f;
    // Samples whose relative error exceeds the supplied tolerance.
    u32 failingSamples = 0;
    f32 failingFraction = 0.0f;
};

// Compares irradiance sample by sample on Rec. 709 luminance. luminanceFloor
// keeps a near-black reference from turning a negligible absolute error into
// a huge relative one; it must be positive, as must tolerance. Returns
// InvalidArgument for empty or mismatched spans, a non-finite value, or a
// non-positive floor or tolerance.
[[nodiscard]] core::Status<core::Error>
MeasureIrradianceError(memory::Span<const math::Vec3> estimate,
                       memory::Span<const math::Vec3> reference, f32 luminanceFloor, f32 tolerance,
                       IrradianceErrorMetrics& out) noexcept;

// Light reaching places it should not.
//
// The samples are irradiance at points on the dark side of an occluder -- the
// inside of a sealed room next to a lit one, say -- and litReference is the
// irradiance on the lit side. A probe grid that interpolates across the wall
// without a visibility test bleeds the lit side into the dark side, and the
// leak ratio is how much of it got through.
struct LightLeakMetrics
{
    u32 sampleCount = 0;
    // Dark-side luminance over lit-side luminance, mean and worst.
    f32 meanLeakRatio = 0.0f;
    f32 maxLeakRatio = 0.0f;
    // Samples whose ratio exceeds leakThreshold.
    u32 leakingSamples = 0;
    f32 leakingFraction = 0.0f;
};

// Returns InvalidArgument for an empty span, a non-finite value, a lit
// reference with non-positive luminance, or a non-positive threshold.
[[nodiscard]] core::Status<core::Error> MeasureLightLeak(memory::Span<const math::Vec3> darkSide,
                                                         const math::Vec3& litReference,
                                                         f32 leakThreshold,
                                                         LightLeakMetrics& out) noexcept;

// Residual noise and bias of an image against a converged reference.
//
// A denoiser has to do two things at once: remove the variance and not move
// the mean. relativeRmse measures the first; relativeBias measures the second,
// and is the number that catches a filter that "denoises" by darkening or by
// blurring a bright region into a dark one. laplacianEnergy is reference-free:
// the mean magnitude of the luminance Laplacian, which is large for per-pixel
// noise and small for a smooth image, so the same scene can be compared before
// and after filtering without a reference at all.
struct NoiseMetrics
{
    u32 pixelCount = 0;
    // Root mean square luminance difference from the reference, absolute and
    // divided by the reference's mean luminance.
    f32 rmse = 0.0f;
    f32 relativeRmse = 0.0f;
    // Mean signed luminance difference, absolute and relative.
    f32 bias = 0.0f;
    f32 relativeBias = 0.0f;
    f32 laplacianEnergy = 0.0f;
    f32 referenceMeanLuminance = 0.0f;
};

// Measures image against reference. Both must be valid and the same size.
// Returns InvalidArgument otherwise, and when the reference's mean luminance
// is not positive.
[[nodiscard]] core::Status<core::Error>
MeasureNoise(const Framebuffer& image, const Framebuffer& reference, NoiseMetrics& out) noexcept;

// Reference-free Laplacian energy of one image, as in NoiseMetrics. Returns
// InvalidArgument for an invalid image or one smaller than 3x3.
[[nodiscard]] core::Status<core::Error> MeasureLaplacianEnergy(const Framebuffer& image,
                                                               f32& out) noexcept;

}  // namespace cyber::software
