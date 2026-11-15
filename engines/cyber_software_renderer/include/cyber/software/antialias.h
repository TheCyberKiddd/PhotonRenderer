#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/math/color.h>
#include <cyber/memory/allocator.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/pipeline.h>

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

// Renderer-wide anti-aliasing.
//
// The software renderer resolves geometric, shading, shadow, and texture
// aliasing with one mechanism: render every sample of an N x N supersampled
// target, then reconstruct the final image with a windowed filter. Nothing in
// the raster, shading, or shadow paths changes, so every effect the renderer
// produces is anti-aliased by the same amount, including specular highlights,
// shadow edges, and texture minification that an edge-only post filter such as
// FXAA or an MSAA coverage mask cannot touch.
//
// Filtering happens in linear light, which is the reason a resolved edge reads
// as a straight ramp rather than the dark fringe a gamma-space average
// produces.

// Supersampling rate presets. The factor is per axis, so High costs 16 samples
// per final pixel and Ultra costs 64.
enum class AntiAliasQuality : u8
{
    // One sample per pixel. Aliased; kept so a caller can measure against it.
    Off = 0,
    // 2x2. Removes the worst staircase at four times the sample cost.
    Balanced,
    // 4x4. The renderer-wide default: edges read as smooth at normal viewing
    // distance and thin geometry stops dropping out.
    High,
    // 8x8. Reference quality for captures and golden images.
    Ultra,
};

// Reconstruction filters, ordered by increasing support. Radius is in final
// pixels; a radius above 0.5 reads samples from neighboring pixel footprints,
// which is what separates a reconstruction filter from a plain average.
enum class ResolveFilter : u8
{
    // Unweighted average of the samples inside one pixel. Radius 0.5.
    // Cheapest, and the only filter here that cannot ring.
    Box = 0,
    // Linear falloff over two pixels. Radius 1. Softer than box, no ringing.
    Tent,
    // Mitchell-Netravali B = C = 1/3. Radius 2. Negative lobes keep edges
    // crisp while removing the staircase, which is why it is the default.
    Mitchell,
    // Blackman-Harris window. Radius 1.5. Smoothest result, no visible
    // ringing; use it when a capture should look soft rather than sharp.
    BlackmanHarris,
};

inline constexpr i32 kMinSupersampleFactor = 1;
inline constexpr i32 kMaxSupersampleFactor = 16;
// What a caller gets by default anywhere in CyberGraphics: 4x4 samples
// reconstructed with Mitchell-Netravali.
inline constexpr i32 kDefaultSupersampleFactor = 4;
inline constexpr ResolveFilter kDefaultResolveFilter = ResolveFilter::Mitchell;
inline constexpr AntiAliasQuality kDefaultAntiAliasQuality = AntiAliasQuality::High;

// Samples per axis for a preset. An unrecognized value returns the default.
[[nodiscard]] constexpr i32 SupersampleFactorForQuality(AntiAliasQuality quality) noexcept
{
    switch (quality)
    {
    case AntiAliasQuality::Off:
        return 1;
    case AntiAliasQuality::Balanced:
        return 2;
    case AntiAliasQuality::High:
        return 4;
    case AntiAliasQuality::Ultra:
        return 8;
    }
    return kDefaultSupersampleFactor;
}

// Samples per final pixel for a preset: the factor squared.
[[nodiscard]] constexpr i32 SampleCountForQuality(AntiAliasQuality quality) noexcept
{
    const i32 factor = SupersampleFactorForQuality(quality);
    return factor * factor;
}

// Half-width of a filter's support in final pixels. An unrecognized value
// returns the default filter's radius.
[[nodiscard]] constexpr f32 ResolveFilterRadius(ResolveFilter filter) noexcept
{
    switch (filter)
    {
    case ResolveFilter::Box:
        return 0.5f;
    case ResolveFilter::Tent:
        return 1.0f;
    case ResolveFilter::Mitchell:
        return 2.0f;
    case ResolveFilter::BlackmanHarris:
        return 1.5f;
    }
    return 2.0f;
}

// Evaluates a filter at a signed distance from the pixel center, in final
// pixels. Returns 0 outside the support. Mitchell returns negative values
// inside its outer lobes, which is intended. The function is pure and
// allocation-free, so a test can integrate it directly.
[[nodiscard]] f32 EvaluateResolveFilter(ResolveFilter filter, f32 offsetPixels) noexcept;

struct ResolveDesc
{
    // Samples per axis in the source. Must be in [1, 16] and must relate the
    // source and destination dimensions exactly.
    i32 factor = kDefaultSupersampleFactor;
    ResolveFilter filter = kDefaultResolveFilter;
    // Clamps each resolved channel to at least 0. A filter with negative lobes
    // can undershoot at a high-contrast edge, which an sRGB encode would
    // otherwise flatten to black and show as a dark halo.
    bool clampNegative = true;
    // Retained for source compatibility and ignored. The resolve streams each
    // block of destination rows through a ring of `tapCount` filtered rows, so
    // its scratch is a few rows rather than the image and it needs a single
    // job-system dispatch instead of two per band.
    i32 bandRows = 32;
    // False forces the single-threaded path even when a job system is
    // supplied. Both paths write identical bytes.
    bool useJobs = true;
};

// Reconstructs destination from the supersampled source.
//
// The filter is separable and applied in two passes: horizontally into a
// temporary row band, then vertically into the destination. Source sample
// (sx, sy) sits at ((sx + 0.5) / factor, (sy + 0.5) / factor) in destination
// pixels, so the weights repeat for every destination pixel and are computed
// once. Weights are normalized to sum to one, and coordinates outside the
// source clamp to the edge, so a constant image resolves to that exact
// constant under every filter and no border darkening appears.
//
// Both framebuffers may be RGBA8Unorm or RGBA32F; filtering runs in binary32
// regardless. Alpha is filtered with the color channels. Output is
// deterministic and byte-identical with and without a job system and for any
// band size.
//
// Returns InvalidArgument for an invalid framebuffer, a factor outside
// [1, 16], or dimensions where source is not destination times factor, and
// Internal when the temporary allocation fails.
[[nodiscard]] core::Status<core::Error>
ResolveSupersampled(const Framebuffer& source, Framebuffer& destination, const ResolveDesc& desc,
                    jobs::JobSystem* jobSystem, memory::Allocator& allocator) noexcept;

struct AntiAliasTargetDesc
{
    // Final, resolved resolution. Both must be in [1, 16384], and the
    // supersampled dimensions must stay within 16384 as well.
    i32 width = 0;
    i32 height = 0;
    AntiAliasQuality quality = kDefaultAntiAliasQuality;
    // Overrides quality when non-zero. Must be in [1, 16].
    i32 factor = 0;
    ResolveFilter filter = kDefaultResolveFilter;
    // Format of the supersampled color target. RGBA32F keeps HDR shading
    // intact until the resolve, which is what a tone mapper needs; RGBA8Unorm
    // halves the footprint and clamps every sample to [0, 1] first.
    PixelFormat format = PixelFormat::RGBA32F;
};

// A supersampled color and depth target plus its resolve.
//
// A caller renders into Color() and Depth() through the normal pipeline, using
// RenderViewport() and RenderWidth()/RenderHeight() for anything that needs
// the raster resolution, then calls Resolve to produce the final image. Aspect
// ratio is unchanged by supersampling, so camera setup that uses the final
// dimensions stays correct.
class AntiAliasTarget
{
public:
    AntiAliasTarget() noexcept = default;
    ~AntiAliasTarget() = default;

    AntiAliasTarget(const AntiAliasTarget&) = delete;
    AntiAliasTarget& operator=(const AntiAliasTarget&) = delete;

    AntiAliasTarget(AntiAliasTarget&&) noexcept = default;
    AntiAliasTarget& operator=(AntiAliasTarget&&) noexcept = default;

    // Allocates the supersampled color and depth targets. Returns
    // InvalidArgument for an out-of-range dimension, factor, or format, and
    // Internal when allocation fails.
    [[nodiscard]] static core::Result<AntiAliasTarget, core::Error>
    Create(const AntiAliasTargetDesc& desc, memory::Allocator& allocator) noexcept;

    [[nodiscard]] bool IsValid() const noexcept;

    [[nodiscard]] Framebuffer& Color() noexcept
    {
        return m_color;
    }
    [[nodiscard]] const Framebuffer& Color() const noexcept
    {
        return m_color;
    }
    [[nodiscard]] DepthBuffer& Depth() noexcept
    {
        return m_depth;
    }
    [[nodiscard]] const DepthBuffer& Depth() const noexcept
    {
        return m_depth;
    }

    [[nodiscard]] i32 Factor() const noexcept
    {
        return m_factor;
    }
    [[nodiscard]] ResolveFilter Filter() const noexcept
    {
        return m_filter;
    }
    // Samples per final pixel: Factor() squared.
    [[nodiscard]] i32 SampleCount() const noexcept
    {
        return m_factor * m_factor;
    }

    // Resolved dimensions.
    [[nodiscard]] i32 Width() const noexcept
    {
        return m_width;
    }
    [[nodiscard]] i32 Height() const noexcept
    {
        return m_height;
    }

    // Supersampled dimensions: the resolved size times Factor().
    [[nodiscard]] i32 RenderWidth() const noexcept
    {
        return m_width * m_factor;
    }
    [[nodiscard]] i32 RenderHeight() const noexcept
    {
        return m_height * m_factor;
    }

    // Full-target viewport in supersampled pixels, for PipelineDesc.
    [[nodiscard]] Viewport RenderViewport() const noexcept;

    // Clears the color target to color and the depth target to the reverse-Z
    // far value, which is what a frame starts from.
    void Clear(const math::Color& color) noexcept;

    // Resolves into destination, which must be valid and exactly
    // Width() x Height(). clampNegative, bandRows, and useJobs are taken from
    // desc; its factor and filter are ignored, because the target owns them.
    [[nodiscard]] core::Status<core::Error> Resolve(Framebuffer& destination,
                                                    jobs::JobSystem* jobSystem,
                                                    memory::Allocator& allocator,
                                                    const ResolveDesc& desc = {}) const noexcept;

private:
    Framebuffer m_color;
    DepthBuffer m_depth;
    i32 m_width = 0;
    i32 m_height = 0;
    i32 m_factor = 0;
    ResolveFilter m_filter = kDefaultResolveFilter;
};

}  // namespace cyber::software
