#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
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
using cyber::core::usize;

// Spatio-temporal denoising of a noisy ray-traced signal.
//
// One or a few rays per pixel produce an unbiased but very noisy estimate of
// shadowing, occlusion, or indirect light. This filter turns it into a usable
// image in two stages, after Schied et al., "Spatiotemporal Variance-Guided
// Filtering" (2017):
//
//  1. Temporal accumulation. Each pixel is reprojected into the previous frame
//     along the velocity buffer and blended with its history. Every reprojected
//     tap is tested against the previous frame's depth and normal, so a
//     disoccluded pixel restarts its history instead of inheriting a stranger's.
//     Alongside colour, the first two moments of luminance are accumulated,
//     which gives each pixel an estimate of its own variance over time. Where
//     history is too short for that estimate to mean anything, variance is
//     estimated spatially instead.
//
//  2. Edge-aware wavelet filtering. An a-trous filter with a 5x5 B3-spline
//     kernel runs several times at doubling step sizes, covering a wide
//     footprint at a fixed cost per pass. Each tap is weighted by depth and
//     normal similarity, which keeps the filter from crossing geometric edges,
//     and by luminance similarity scaled by the pixel's standard deviation,
//     which makes the filter strong where the signal is noisy and gentle where
//     it has already converged. Variance is filtered alongside colour, so each
//     pass refines the guide the next pass uses.
//
// Conventions follow the post-processing chain in post_process.h: colour is
// linear and unbounded, depth is reverse-Z, normals are world-space unit
// vectors remapped to [0, 1] in the red, green, and blue channels, and
// velocity is screen-space pixels per frame, x right and y down, in red and
// green. The filter is deterministic; job-parallel and single-threaded runs
// write identical bytes.

struct DenoiseGuides
{
    // Reverse-Z depth of the frame being denoised. Required.
    const DepthBuffer* depth = nullptr;
    // World-space normals encoded to [0, 1]. Required.
    const Framebuffer* normals = nullptr;
    // Optional. Without it history is reprojected statically, which is correct
    // only while the camera and the scene are still.
    const Framebuffer* velocity = nullptr;
    // Depth linearization, matching the projection the frame was rendered
    // with.
    f32 nearPlane = 0.1f;
    f32 farPlane = 1000.0f;
};

inline constexpr u32 kMaxDenoiserIterations = 6;

struct DenoiserDesc
{
    bool temporal = true;
    // Smallest weight the current frame receives once history is long. A
    // short history uses 1 / length instead, which is a plain running mean and
    // converges as fast as an average can. In (0, 1].
    f32 temporalAlpha = 0.1f;
    // Cap on the history length the 1 / length weight counts to, in [1, 255].
    u32 maxHistoryLength = 32;
    // A reprojected tap is rejected when its linear depth differs from the
    // pixel's by more than this fraction of the pixel's depth.
    f32 depthRejectRelative = 0.1f;
    // A reprojected tap is rejected when the cosine between its normal and the
    // pixel's is below this.
    f32 normalRejectCosine = 0.9f;
    // History shorter than this uses the spatial variance estimate.
    u32 spatialVarianceHistory = 4;

    bool spatial = true;
    // Wavelet passes, in [0, kMaxDenoiserIterations]. Pass i has a step of
    // 2^i pixels, so five passes reach a footprint of about 125 pixels.
    u32 iterations = 4;
    // Depth edge sensitivity: a tap is weighted down as its linear depth
    // departs from the expected plane by this fraction of the pixel's depth,
    // per pixel of step. Larger values blur more across depth changes.
    f32 depthSigma = 0.02f;
    // Exponent on the normal cosine. Larger values preserve creases more.
    f32 normalPower = 64.0f;
    // Luminance edge sensitivity, in standard deviations. Larger values blur
    // more.
    f32 luminanceSigma = 4.0f;
    bool useJobs = true;
};

struct DenoiserStats
{
    u32 pixelCount = 0;
    // Pixels that had history last frame but lost it to the depth, normal, or
    // screen-bounds tests this frame: disocclusions, and the camera moving
    // faster than history can follow.
    u32 historyRejectedPixels = 0;
    // Mean accumulated history length after this frame.
    f32 meanHistoryLength = 0.0f;
    // Pixels that used the spatial variance estimate.
    u32 spatialVariancePixels = 0;
    u32 iterations = 0;
};

// Move-only. Owns the history and scratch for one resolution.
class SpatioTemporalDenoiser
{
public:
    SpatioTemporalDenoiser() noexcept = default;
    ~SpatioTemporalDenoiser();

    SpatioTemporalDenoiser(const SpatioTemporalDenoiser&) = delete;
    SpatioTemporalDenoiser& operator=(const SpatioTemporalDenoiser&) = delete;

    SpatioTemporalDenoiser(SpatioTemporalDenoiser&& other) noexcept;
    SpatioTemporalDenoiser& operator=(SpatioTemporalDenoiser&& other) noexcept;

    // Returns InvalidArgument for a size outside [1, 16384] and Internal when
    // allocation fails.
    [[nodiscard]] static core::Result<SpatioTemporalDenoiser, core::Error>
    Create(i32 width, i32 height, memory::Allocator& allocator) noexcept;

    [[nodiscard]] bool IsValid() const noexcept;
    [[nodiscard]] i32 Width() const noexcept;
    [[nodiscard]] i32 Height() const noexcept;
    [[nodiscard]] usize DataSizeBytes() const noexcept;

    // Discards history. The next frame filters spatially only, as after a
    // camera cut.
    void Reset() noexcept;

    // Denoises one frame into output. noisy and every guide must match the
    // denoiser's size; output must match too and may be any valid format.
    // noisy and output must be different images. Returns InvalidArgument for a
    // missing or mismatched buffer or an out-of-range descriptor. History is
    // updated only on success.
    [[nodiscard]] core::Status<core::Error> Denoise(const Framebuffer& noisy,
                                                    const DenoiseGuides& guides,
                                                    const DenoiserDesc& desc, Framebuffer& output,
                                                    DenoiserStats* stats = nullptr,
                                                    jobs::JobSystem* jobSystem = nullptr) noexcept;

private:
    void Release() noexcept;
    void MoveFrom(SpatioTemporalDenoiser& other) noexcept;

    memory::Allocator* m_allocator = nullptr;
    f32* m_storage = nullptr;
    usize m_sizeBytes = 0;
    i32 m_width = 0;
    i32 m_height = 0;
    bool m_hasHistory = false;

    // Views into m_storage, per pixel unless noted.
    f32* m_historyColor = nullptr;    // 3
    f32* m_historyMoments = nullptr;  // 2
    f32* m_historyLength = nullptr;   // 1
    f32* m_previousDepth = nullptr;   // 1, linear
    f32* m_previousNormal = nullptr;  // 3, decoded
    f32* m_currentDepth = nullptr;    // 1, linear
    f32* m_currentNormal = nullptr;   // 3, decoded
    f32* m_accumColor = nullptr;      // 3
    f32* m_accumMoments = nullptr;    // 2
    f32* m_accumLength = nullptr;     // 1
    f32* m_filterA = nullptr;         // 4: colour and variance
    f32* m_filterB = nullptr;         // 4
};

}  // namespace cyber::software
