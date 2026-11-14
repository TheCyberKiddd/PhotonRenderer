#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/math/color.h>
#include <cyber/math/mat4.h>
#include <cyber/math/vec2.h>
#include <cyber/memory/allocator.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>

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

// Temporal anti-aliasing, and the edge filter that stands in for it when a
// frame has no usable history.
//
// Supersampling, which the renderer uses everywhere else, removes aliasing by
// paying for every sample inside one frame. Temporal anti-aliasing instead
// spreads those samples across frames: the camera is jittered by a sub-pixel
// offset each frame, and the result is accumulated into a history buffer that
// is reprojected along the velocity buffer as the scene moves. One sample per
// pixel per frame converges on the same answer, as long as the history that is
// reused actually corresponds to the pixel reusing it.
//
// Everything hard about temporal anti-aliasing is that last clause. Where the
// reprojection is wrong -- a disocclusion, a shadow that moved, a reflection
// that is not attached to the surface under it -- stale color bleeds into the
// new frame and reads as a smear trailing the motion. This header constrains
// history three ways, and the acceptance suite measures the result rather than
// inspecting it:
//
//  - Neighborhood clamping. History is clipped into the color range of the
//    current frame's 3x3 neighborhood, in YCoCg, so it can never introduce a
//    color the new frame does not already contain nearby.
//  - Variance clipping. The clip box is the neighborhood mean plus or minus
//    its standard deviation scaled by varianceGamma, which is tighter than the
//    raw min and max and does not let one bright outlier widen the box.
//  - Velocity-based rejection. A pixel whose reprojected source lands outside
//    the frame, or whose motion exceeds the disocclusion threshold, drops its
//    history and restarts accumulation.
//
// Determinism is the same contract the rest of the renderer keeps: the jitter
// is a fixed Halton sequence, and the resolve writes identical bytes with and
// without a job system.

// ---------------------------------------------------------------------------
// Anti-aliasing method selection
// ---------------------------------------------------------------------------

// The renderer-level choice a caller makes once.
//
// Supersample is this renderer's coverage-based path and stands where MSAA
// stands on a GPU: it resolves geometry, shading, shadow, and texture aliasing
// by rendering more samples, at a cost that scales with the sample count. A
// software rasterizer has no fixed-function coverage mask, so there is no
// cheaper MSAA to implement underneath it; supersampling is that mechanism.
enum class AntiAliasMethod : u8
{
    // One sample per pixel, no filter. The control case.
    None = 0,
    // The renderer-wide N x N supersampled path in antialias.h.
    Supersample,
    // Single-frame luminance edge filter. Cheapest, and the fallback when
    // history is unavailable.
    Fxaa,
    // Jittered accumulation with reprojected, clamped history.
    Temporal,
};

[[nodiscard]] const char* AntiAliasMethodName(AntiAliasMethod method) noexcept;

// ---------------------------------------------------------------------------
// Camera jitter
// ---------------------------------------------------------------------------

// Default jitter period. Sixteen phases is long enough to cover a 4x4 sample
// grid and short enough that a static camera converges quickly.
inline constexpr i32 kDefaultJitterPhaseCount = 16;
inline constexpr i32 kMaxJitterPhaseCount = 256;

// The radical inverse of index in the given base. The base must be at least 2;
// anything smaller returns 0.
[[nodiscard]] f32 RadicalInverse(u32 index, u32 base) noexcept;

// Halton(2, 3) point for a sequence index, in [0, 1) on both axes. Index 0 is
// the first point of the sequence, not the degenerate origin.
[[nodiscard]] math::Vec2 HaltonPoint(u32 index) noexcept;

// Sub-pixel jitter offset for a frame, in pixels, centered on zero so the
// sequence averages to the pixel center. phaseCount is clamped to
// [1, 256]; a count of 1 disables jitter.
[[nodiscard]] math::Vec2 TemporalJitterOffset(u32 frameIndex,
                                              i32 phaseCount = kDefaultJitterPhaseCount) noexcept;

// Shifts a projection matrix by a sub-pixel offset in pixels, x right and y
// down, for a width x height target.
//
// The offset is applied as clip.xy += offsetNdc * clip.w, which is correct for
// perspective and orthographic projections alike and leaves depth untouched, so
// a jittered frame still depth-tests against an unjittered shadow map.
// A non-finite offset or a degenerate target returns the projection unchanged.
[[nodiscard]] math::Mat4 ApplyJitterToProjection(const math::Mat4& projection,
                                                 const math::Vec2& offsetPixels, i32 width,
                                                 i32 height) noexcept;

// ---------------------------------------------------------------------------
// Temporal resolve
// ---------------------------------------------------------------------------

struct TemporalAaSettings
{
    // Fraction of the previous result kept per frame, in [0, 1). 0.9 converges
    // in roughly ten frames and still reacts inside a quarter of a second at
    // 60 Hz. 1 would never converge and is rejected.
    f32 historyWeight = 0.9f;
    // Clip history into the current neighborhood. Switching it off is the
    // control case for the ghosting measurement: the same scene then smears.
    bool clampHistory = true;
    // Width of the variance clip box in standard deviations. Lower is tighter
    // and sharper but reintroduces flicker; 1.0 is the usual compromise.
    f32 varianceGamma = 1.0f;
    // How far the clip box is widened where the input is oscillating, in
    // multiples of the measured oscillation amplitude. 0 restores a strict
    // neighborhood clamp.
    //
    // A neighborhood clamp cannot preserve a feature narrower than a pixel.
    // Half the frames, a jittered sample misses the feature entirely, its
    // whole neighborhood agrees that nothing is there, and the converged
    // history -- which is the correct partial coverage -- is clamped away.
    // The accumulation then sits below the truth and the feature reads as a
    // dim flicker. This is the artifact that makes temporal anti-aliasing lose
    // thin geometry.
    //
    // The information that separates that case from a genuine cut is not in
    // the current frame; it is in how the pixel has been disagreeing. The
    // resolver tracks a running mean of the signed difference between the
    // input and its reprojected history, and a running mean of that
    // difference's magnitude. A cut disagrees hard and in one direction, so
    // both rise together. Sub-pixel flicker disagrees in alternating
    // directions, so the magnitude rises while the signed mean stays near
    // zero. Their difference is the oscillation amplitude, and the box is
    // widened by it: the feature survives, and the cut does not, because at
    // the moment a cut arrives the pixel's history of disagreement is still
    // empty.
    // 2 is the measured knee on the acceptance scene: it recovers essentially
    // all of the converged accuracy a neighborhood clamp costs, and the edge
    // statistics of the result land on those of the integrated reference
    // rather than below them, which is what a correctly reconstructed image
    // looks like. Widening further buys little and trusts history more.
    f32 flickerRelaxation = 2.0f;
    // Reproject history along the velocity buffer. With no velocity buffer the
    // resolve falls back to a static reprojection, which is correct only for a
    // still camera.
    bool useVelocity = true;
    // Motion beyond this, in pixels per frame, is treated as a disocclusion
    // and drops history for that pixel.
    f32 disocclusionPixels = 64.0f;
    // Weight each contribution by 1 / (1 + luma), the standard fix for a
    // single bright sample dominating the average and flickering as it moves
    // between pixels.
    //
    // Off by default, because it is not free: down-weighting bright samples
    // biases the accumulation darker wherever a pixel alternates between a
    // bright and a dark value, which is exactly what a sub-pixel highlight
    // under jitter does. On the acceptance scene it costs roughly an order of
    // magnitude of converged accuracy against the integrated reference. Turn
    // it on for scene-referred HDR input, where one stray firefly is the
    // larger problem, and leave it off for the display-referred and
    // moderate-range cases this renderer resolves by default.
    bool luminanceWeighting = false;

    // Depth-based disocclusion rejection, used only when a depth buffer is
    // supplied with the frame.
    //
    // Velocity alone cannot see a disocclusion. A pixel of floor that a post
    // was covering last frame has the floor's velocity, which reprojects it
    // onto the post, and the post's color is then reused for the floor. The
    // result is a trail of the occluder smeared along the motion, and it is
    // the artifact that survives neighborhood rectification most readily,
    // because an occluder edge is exactly where the neighborhood is widest.
    //
    // Comparing the reprojected depth against the depth actually rendered
    // closes it: a surface that is not at the distance its history says it was
    // is not the same surface, whatever the velocity buffer claims.
    // depthRejectRelative is a fraction of the view-space distance, so the
    // tolerance scales with distance the way depth precision does.
    f32 depthRejectRelative = 0.03f;
    // The projection the depth buffer was rendered with, needed to linearize
    // reverse-Z. Ignored when no depth buffer is supplied.
    f32 nearPlane = 0.1f;
    f32 farPlane = 1000.0f;

    // False forces the single-threaded path. Both paths write identical bytes.
    bool useJobs = true;
};

// The buffers a frame hands to the resolver. color is required; velocity and
// depth are each optional, and the resolve degrades in a defined way without
// them: no velocity means a static reprojection, correct only for a still
// camera, and no depth means no disocclusion test beyond the velocity
// threshold.
struct TemporalAaInputs
{
    const Framebuffer* color = nullptr;
    const Framebuffer* velocity = nullptr;
    // Reverse-Z depth of this frame, matching the resolver's size. The
    // resolver keeps the previous frame's copy itself.
    const DepthBuffer* depth = nullptr;
};

// What the resolve did with history this frame, for debugging and for the
// acceptance measurements.
struct TemporalAaStats
{
    // Pixels that reused history at all.
    u32 historyPixels = 0;
    // Pixels whose history was rejected: off-screen reprojection, motion past
    // the disocclusion threshold, or a reprojected depth that does not match
    // the surface actually rendered.
    u32 rejectedPixels = 0;
    // Of those, the ones rejected by the depth test alone. This is the count
    // that separates an occluder trail being prevented from a camera moving
    // fast, and it is zero when no depth buffer was supplied.
    u32 depthRejectedPixels = 0;
    // Pixels whose history was pulled back into the neighborhood box. A high
    // count is normal during motion and is the mechanism that prevents
    // trailing.
    u32 clampedPixels = 0;
    // Pixels whose box was widened by the measured oscillation amplitude,
    // which is where sub-pixel detail is being held onto rather than clamped
    // away. Zero when flickerRelaxation is zero.
    u32 relaxedPixels = 0;
    // True when this was the first frame after a reset and the input was
    // copied through without accumulation.
    bool historyPrimed = false;
};

// Accumulates jittered frames into an anti-aliased result.
//
// A caller renders frame N with the projection returned by
// ApplyJitterToProjection for phase N, then calls Resolve. The resolver owns
// the history buffer; the destination it writes is also what it keeps, so the
// caller may hand the same destination to the post chain afterwards.
class TemporalAaResolver
{
public:
    TemporalAaResolver() noexcept = default;
    ~TemporalAaResolver() = default;

    TemporalAaResolver(const TemporalAaResolver&) = delete;
    TemporalAaResolver& operator=(const TemporalAaResolver&) = delete;

    TemporalAaResolver(TemporalAaResolver&&) noexcept = default;
    TemporalAaResolver& operator=(TemporalAaResolver&&) noexcept = default;

    // Allocates the history buffer. Dimensions must be in [1, 16384]. Returns
    // InvalidArgument for a bad dimension and Internal when allocation fails.
    [[nodiscard]] static core::Result<TemporalAaResolver, core::Error>
    Create(i32 width, i32 height, memory::Allocator& allocator) noexcept;

    [[nodiscard]] bool IsValid() const noexcept;
    [[nodiscard]] i32 Width() const noexcept;
    [[nodiscard]] i32 Height() const noexcept;
    [[nodiscard]] bool HasHistory() const noexcept;

    // Drops history. Call it on a camera cut, a resolution change, or any
    // other discontinuity, because reprojection cannot describe one.
    void Reset() noexcept;

    // Resolves one jittered frame.
    //
    // color is the frame as rendered, velocity may be null, and destination
    // must be valid and exactly Width() x Height(). stats may be null. The
    // first frame after Create or Reset is copied through unchanged and
    // reported with historyPrimed set.
    //
    // Returns InvalidArgument for a mismatched buffer, a historyWeight outside
    // [0, 1), or a velocity buffer of the wrong size.
    [[nodiscard]] core::Status<core::Error> Resolve(const Framebuffer& color,
                                                    const Framebuffer* velocity,
                                                    Framebuffer& destination,
                                                    const TemporalAaSettings& settings,
                                                    jobs::JobSystem* jobSystem,
                                                    TemporalAaStats* stats) noexcept;

    // The same resolve, with the frame's depth available for the disocclusion
    // test. inputs.color is required and must not alias destination.
    [[nodiscard]] core::Status<core::Error> Resolve(const TemporalAaInputs& inputs,
                                                    Framebuffer& destination,
                                                    const TemporalAaSettings& settings,
                                                    jobs::JobSystem* jobSystem,
                                                    TemporalAaStats* stats) noexcept;

    [[nodiscard]] const Framebuffer& History() const noexcept;

private:
    Framebuffer m_history;
    // Running mean of the signed input-minus-history difference, and of that
    // difference's magnitude, in YCoCg, reprojected with the color history.
    // Two buffers each: one read this frame, one written for the next, so the
    // job-parallel path never reads a pixel another row has already written.
    Framebuffer m_signedDrift[2];
    Framebuffer m_absoluteDrift[2];
    // Linearized view-space distance of the frame the history describes, kept
    // as a float image so it can be reprojected with the same bilinear filter
    // the color history uses.
    Framebuffer m_depthHistory[2];
    bool m_hasDepthHistory = false;
    i32 m_width = 0;
    i32 m_height = 0;
    i32 m_driftIndex = 0;
    bool m_hasHistory = false;
};

// ---------------------------------------------------------------------------
// FXAA
// ---------------------------------------------------------------------------

struct FxaaSettings
{
    // Minimum local contrast, as a fraction of the brighter luma, before an
    // edge is filtered. Below this the pixel is passed through.
    f32 relativeThreshold = 0.125f;
    // Absolute contrast floor, which keeps the filter off dark gradients where
    // the relative test is noisy.
    f32 absoluteThreshold = 0.0312f;
    // Blend applied to the low-pass result, in [0, 1]. 1 is the full filter.
    f32 subpixelBlend = 0.75f;
    // False forces the single-threaded path.
    bool useJobs = true;
};

// Single-frame luminance edge filter, run on display-referred color.
//
// It finds a local contrast step, picks the dominant edge direction, then
// walks along the edge in both directions to find its ends. How far along the
// edge a pixel lies decides how far it moves toward the other side, up to half
// a pixel at the middle, which is what turns a staircase into a ramp; a
// sub-pixel contrast term catches the single-pixel features the walk cannot
// see. It touches only edges, so unlike supersampling it cannot fix specular
// or texture aliasing; it is here as the cheap alternative and as the fallback
// for a frame with no history.
//
// source and destination must be valid, the same size, and different objects.
[[nodiscard]] core::Status<core::Error> FxaaResolve(const Framebuffer& source,
                                                    Framebuffer& destination,
                                                    const FxaaSettings& settings,
                                                    jobs::JobSystem* jobSystem) noexcept;

}  // namespace cyber::software
