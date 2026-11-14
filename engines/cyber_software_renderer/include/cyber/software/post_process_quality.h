#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/memory/span.h>
#include <cyber/software/framebuffer.h>

namespace cyber::software
{

using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;

// Image measurements for the post-processing and anti-aliasing phase.
//
// The roadmap's bar for this phase is that temporal anti-aliasing shows no
// visible ghosting in a moving scene. "Visible" is not a test, so these
// functions turn the artifacts into numbers a suite can assert a threshold on,
// the same way shadow_quality.h does for shadows. Every function reads
// borrowed images, allocates nothing, and is deterministic.
//
// Each metric is paired, in the acceptance suite, with a control configuration
// that fails it. A measurement that nothing can fail measures nothing.

// Per-pixel agreement between two images of equal size.
struct ImageDifferenceMetrics
{
    u32 pixelCount = 0;
    // Mean and worst absolute RGB difference, in linear units.
    f32 meanAbsoluteError = 0.0f;
    f32 maxAbsoluteError = 0.0f;
    // Root mean square difference, which weights a few large errors more
    // heavily than the mean does.
    f32 rootMeanSquareError = 0.0f;
    // Pixels whose worst channel differs by more than the supplied tolerance.
    u32 differingPixels = 0;
    f32 differingFraction = 0.0f;
};

// Compares two images channel by channel. Returns InvalidArgument for an
// invalid image, a size mismatch, or a negative tolerance.
[[nodiscard]] core::Status<core::Error> MeasureImageDifference(const Framebuffer& image,
                                                               const Framebuffer& reference,
                                                               f32 tolerance,
                                                               ImageDifferenceMetrics& out) noexcept;

// How much of the previous frame survived into this one where it should not
// have.
//
// Ghosting is stale history that reprojection failed to correct: a trail of the
// last frame's content dragging behind motion. The measurement needs three
// images of the same scene and the same size:
//
//   resolved          what the anti-aliasing produced for frame N
//   current           frame N with no temporal history, the correct answer
//   previous          frame N-1 with no temporal history, the stale answer
//
// For every pixel the two references disagree on -- the pixels motion actually
// changed -- the resolved color is projected onto the line from current to
// previous. The projection is 0 when the resolve landed on the correct answer
// and 1 when it reproduced the stale one, so it reads directly as the fraction
// of history that leaked through. Pixels the references agree on carry no
// information about ghosting and are excluded; changedPixels reports how many
// were measured, so a test can refuse a scene that did not move.
struct GhostingMetrics
{
    // Pixels where the two references differ by more than changeThreshold.
    u32 changedPixels = 0;
    f32 changedFraction = 0.0f;
    // Mean and worst history contamination over those pixels, in [0, 1].
    f32 meanContamination = 0.0f;
    f32 maxContamination = 0.0f;
    // The single fraction of the previous frame that best explains the whole
    // resolved image: the least-squares fit of resolved = current + a *
    // (previous - current) over the changed pixels, clamped to [0, 1].
    //
    // This is the number to assert on, and meanContamination is the number to
    // read alongside it. The per-pixel ratio divides by how much the pixel
    // changed, so a pixel that barely cleared changeThreshold turns a
    // negligible error into a large ratio, and the unweighted mean of those
    // ratios is dominated by pixels carrying almost no visible signal. The
    // least-squares fit weights each pixel by how much it actually moved,
    // which is what an eye does when it decides whether a trail is there.
    f32 weightedContamination = 0.0f;
    // Pixels whose contamination exceeds ghostThreshold, and their share of
    // the changed pixels. This is the number that corresponds to what an eye
    // calls a visible trail.
    u32 ghostedPixels = 0;
    f32 ghostedFraction = 0.0f;
};

// Measures history contamination. changeThreshold selects the moving pixels
// and ghostThreshold decides which of them count as ghosted; both are absolute
// linear color differences and must be positive. Returns InvalidArgument for an
// invalid or mismatched image or a non-positive threshold, and NotFound when no
// pixel changed, because a still scene cannot show ghosting.
[[nodiscard]] core::Status<core::Error> MeasureGhosting(const Framebuffer& resolved,
                                                        const Framebuffer& current,
                                                        const Framebuffer& previous,
                                                        f32 changeThreshold, f32 ghostThreshold,
                                                        GhostingMetrics& out) noexcept;

// Frame-to-frame flicker across a sequence of resolves of an unchanging scene.
//
// A temporal filter that is too eager to reject history trades ghosting for
// shimmer, so this is the metric that keeps the ghosting result honest: both
// have to pass at once.
struct TemporalStabilityMetrics
{
    u32 frameCount = 0;
    // Mean and worst absolute change between consecutive frames.
    f32 meanFrameDelta = 0.0f;
    f32 maxFrameDelta = 0.0f;
    // Largest change between the first and last frame, which catches a slow
    // drift that per-frame deltas hide.
    f32 totalDrift = 0.0f;
};

// Measures stability over at least two same-sized frames in capture order.
// Returns InvalidArgument for fewer than two frames or a mismatched image.
[[nodiscard]] core::Status<core::Error>
MeasureTemporalStability(memory::Span<const Framebuffer* const> frames,
                         TemporalStabilityMetrics& out) noexcept;

// Edge quality of a single image.
//
// A hard step is a neighboring pair of pixels whose luminance jumps by more
// than stepThreshold: the staircase an unfiltered edge leaves behind. An
// anti-aliased image of the same scene replaces those steps with ramps, so the
// fraction falls even though the scene is identical.
struct EdgeAliasingMetrics
{
    u32 pixelCount = 0;
    // Horizontal or vertical neighbor pairs exceeding the threshold.
    u32 hardStepCount = 0;
    f32 hardStepFraction = 0.0f;
    // Mean luminance gradient magnitude over the image, which separates a
    // genuinely softer image from one that merely lost contrast.
    f32 meanGradient = 0.0f;
};

// Measures edge steps. stepThreshold is an absolute luminance difference and
// must be positive. Returns InvalidArgument for an invalid image or a
// non-positive threshold.
[[nodiscard]] core::Status<core::Error> MeasureEdgeAliasing(const Framebuffer& image,
                                                            f32 stepThreshold,
                                                            EdgeAliasingMetrics& out) noexcept;

}  // namespace cyber::software
