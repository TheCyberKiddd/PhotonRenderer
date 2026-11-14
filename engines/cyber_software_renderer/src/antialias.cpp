#include <cyber/jobs/job_system.h>
#include <cyber/software/antialias.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace cyber::software
{

using cyber::core::usize;

namespace
{

constexpr i32 kMaxTargetDimension = 16384;
constexpr i32 kMinBandRows = 1;
constexpr i32 kMaxBandRows = 256;
// The widest window a filter can need: 2 * radius * factor + 1 taps with
// radius 2 and factor 16.
constexpr i32 kMaxFilterTaps = 2 * 2 * kMaxSupersampleFactor + 1;

// Mitchell-Netravali with B = C = 1/3, scaled so the support is +/-2 pixels.
[[nodiscard]] f32 MitchellWeight(f32 distance) noexcept
{
    constexpr f32 kB = 1.0f / 3.0f;
    constexpr f32 kC = 1.0f / 3.0f;
    const f32 x = distance;
    const f32 x2 = x * x;
    const f32 x3 = x2 * x;
    if (x < 1.0f)
    {
        const f32 p3 = 12.0f - 9.0f * kB - 6.0f * kC;
        const f32 p2 = -18.0f + 12.0f * kB + 6.0f * kC;
        const f32 p0 = 6.0f - 2.0f * kB;
        return (p3 * x3 + p2 * x2 + p0) * (1.0f / 6.0f);
    }
    if (x < 2.0f)
    {
        const f32 q3 = -kB - 6.0f * kC;
        const f32 q2 = 6.0f * kB + 30.0f * kC;
        const f32 q1 = -12.0f * kB - 48.0f * kC;
        const f32 q0 = 8.0f * kB + 24.0f * kC;
        return (q3 * x3 + q2 * x2 + q1 * x + q0) * (1.0f / 6.0f);
    }
    return 0.0f;
}

// Four-term Blackman-Harris window stretched over +/-1.5 pixels. The window
// peaks at 1 and reaches 0 at the support edge, so its mean over the span is
// its first coefficient; dividing by span * a0 gives the unit area a
// reconstruction filter needs.
[[nodiscard]] f32 BlackmanHarrisWeight(f32 distance) noexcept
{
    constexpr f32 kRadius = 1.5f;
    constexpr f32 kA0 = 0.35875f;
    constexpr f32 kA1 = 0.48829f;
    constexpr f32 kA2 = 0.14128f;
    constexpr f32 kA3 = 0.01168f;
    constexpr f32 kNormalization = 1.0f / (2.0f * kRadius * kA0);
    if (distance >= kRadius)
    {
        return 0.0f;
    }
    const f32 phase = cyber::math::kTwoPi * (distance / (2.0f * kRadius) + 0.5f);
    const f32 window =
        kA0 - kA1 * std::cos(phase) + kA2 * std::cos(2.0f * phase) - kA3 * std::cos(3.0f * phase);
    return window * kNormalization;
}

struct FilterKernel
{
    // Offset from destinationIndex * factor to the first tap.
    i32 firstOffset = 0;
    i32 tapCount = 0;
    f32 weights[kMaxFilterTaps]{};
};

// Builds the tap window and normalized weights shared by every destination
// pixel along one axis. Source sample s sits at (s + 0.5) / factor in
// destination pixels, so for destination pixel d the taps are the samples with
// |(s + 0.5) / factor - (d + 0.5)| <= radius, and the offsets s - d * factor
// are the same for every d.
[[nodiscard]] FilterKernel BuildFilterKernel(ResolveFilter filter, i32 factor) noexcept
{
    FilterKernel kernel{};
    const f32 radius = ResolveFilterRadius(filter);
    const f32 scale = static_cast<f32>(factor);
    const f32 lowest = (0.5f - radius) * scale - 0.5f;
    const f32 highest = (0.5f + radius) * scale - 0.5f;

    // A tap exactly on the support boundary evaluates to zero, so the bounds
    // are taken open and the window stays symmetric for any factor.
    i32 first = static_cast<i32>(std::floor(lowest)) + 1;
    i32 last = static_cast<i32>(std::ceil(highest)) - 1;
    if (last < first)
    {
        // Degenerate only for a factor so small that no sample lands inside
        // the support; the nearest sample still has to be used.
        first = 0;
        last = 0;
    }
    if (last - first + 1 > kMaxFilterTaps)
    {
        last = first + kMaxFilterTaps - 1;
    }

    kernel.firstOffset = first;
    kernel.tapCount = last - first + 1;

    f32 total = 0.0f;
    for (i32 tap = 0; tap < kernel.tapCount; ++tap)
    {
        const f32 sampleCenter = (static_cast<f32>(first + tap) + 0.5f) / scale;
        const f32 weight = EvaluateResolveFilter(filter, sampleCenter - 0.5f);
        kernel.weights[tap] = weight;
        total += weight;
    }
    if (!(std::abs(total) > 1e-8f))
    {
        // Cannot normalize; fall back to an average so the resolve still
        // preserves a constant image.
        const f32 uniform = 1.0f / static_cast<f32>(kernel.tapCount);
        for (i32 tap = 0; tap < kernel.tapCount; ++tap)
        {
            kernel.weights[tap] = uniform;
        }
        return kernel;
    }
    const f32 inverseTotal = 1.0f / total;
    for (i32 tap = 0; tap < kernel.tapCount; ++tap)
    {
        kernel.weights[tap] *= inverseTotal;
    }
    return kernel;
}

struct Rgba
{
    f32 r = 0.0f;
    f32 g = 0.0f;
    f32 b = 0.0f;
    f32 a = 0.0f;
};

// Horizontal pass context. One invocation filters one source row of the band
// into destinationWidth accumulated samples.
struct HorizontalContext
{
    const Framebuffer* source = nullptr;
    Rgba* band = nullptr;
    const FilterKernel* kernel = nullptr;
    i32 destinationWidth = 0;
    i32 factor = 0;
    i32 firstSourceRow = 0;
};

void FilterSourceRow(void* userData, u32 index) noexcept
{
    const HorizontalContext& context = *static_cast<const HorizontalContext*>(userData);
    const Framebuffer& source = *context.source;
    const FilterKernel& kernel = *context.kernel;
    const i32 sourceRow =
        std::clamp(context.firstSourceRow + static_cast<i32>(index), 0, source.Height() - 1);
    Rgba* output =
        context.band + static_cast<usize>(index) * static_cast<usize>(context.destinationWidth);

    // RGBA32F rows are read directly; every other format goes through
    // GetPixel, which converts.
    const f32* floatRow = nullptr;
    if (source.Format() == PixelFormat::RGBA32F)
    {
        floatRow = reinterpret_cast<const f32*>(source.Row(sourceRow));
    }
    const i32 lastColumn = source.Width() - 1;

    for (i32 x = 0; x < context.destinationWidth; ++x)
    {
        const i32 base = x * context.factor + kernel.firstOffset;
        Rgba sum{};
        for (i32 tap = 0; tap < kernel.tapCount; ++tap)
        {
            const i32 column = std::clamp(base + tap, 0, lastColumn);
            const f32 weight = kernel.weights[tap];
            if (floatRow != nullptr)
            {
                const f32* texel = floatRow + static_cast<usize>(column) * 4;
                sum.r += texel[0] * weight;
                sum.g += texel[1] * weight;
                sum.b += texel[2] * weight;
                sum.a += texel[3] * weight;
            }
            else
            {
                const math::Color texel = source.GetPixel(column, sourceRow);
                sum.r += texel.r * weight;
                sum.g += texel.g * weight;
                sum.b += texel.b * weight;
                sum.a += texel.a * weight;
            }
        }
        output[x] = sum;
    }
}

// Vertical pass context. One invocation produces one destination row from the
// band the horizontal pass filled.
struct VerticalContext
{
    Framebuffer* destination = nullptr;
    const Rgba* band = nullptr;
    const FilterKernel* kernel = nullptr;
    i32 destinationWidth = 0;
    i32 factor = 0;
    i32 bandRowCount = 0;
    i32 firstSourceRow = 0;
    i32 firstDestinationRow = 0;
    bool clampNegative = false;
};

void FilterDestinationRow(void* userData, u32 index) noexcept
{
    const VerticalContext& context = *static_cast<const VerticalContext*>(userData);
    const FilterKernel& kernel = *context.kernel;
    const i32 destinationRow = context.firstDestinationRow + static_cast<i32>(index);
    const i32 base = destinationRow * context.factor + kernel.firstOffset;

    // The horizontal pass already edge-clamped each band row, so a tap only
    // has to stay inside the band.
    const Rgba* tapRows[kMaxFilterTaps]{};
    for (i32 tap = 0; tap < kernel.tapCount; ++tap)
    {
        const i32 bandRow =
            std::clamp(base + tap - context.firstSourceRow, 0, context.bandRowCount - 1);
        tapRows[tap] = context.band +
                       static_cast<usize>(bandRow) * static_cast<usize>(context.destinationWidth);
    }

    for (i32 x = 0; x < context.destinationWidth; ++x)
    {
        Rgba sum{};
        for (i32 tap = 0; tap < kernel.tapCount; ++tap)
        {
            const Rgba& value = tapRows[tap][x];
            const f32 weight = kernel.weights[tap];
            sum.r += value.r * weight;
            sum.g += value.g * weight;
            sum.b += value.b * weight;
            sum.a += value.a * weight;
        }
        if (context.clampNegative)
        {
            sum.r = std::max(sum.r, 0.0f);
            sum.g = std::max(sum.g, 0.0f);
            sum.b = std::max(sum.b, 0.0f);
            sum.a = std::max(sum.a, 0.0f);
        }
        context.destination->SetPixel(x, destinationRow, math::Color{sum.r, sum.g, sum.b, sum.a});
    }
}

}  // namespace

f32 EvaluateResolveFilter(ResolveFilter filter, f32 offsetPixels) noexcept
{
    if (!std::isfinite(offsetPixels))
    {
        return 0.0f;
    }
    const f32 distance = std::abs(offsetPixels);
    switch (filter)
    {
    case ResolveFilter::Box:
        return distance < 0.5f ? 1.0f : 0.0f;
    case ResolveFilter::Tent:
        return distance < 1.0f ? 1.0f - distance : 0.0f;
    case ResolveFilter::Mitchell:
        return MitchellWeight(distance);
    case ResolveFilter::BlackmanHarris:
        return BlackmanHarrisWeight(distance);
    }
    return MitchellWeight(distance);
}

core::Status<core::Error> ResolveSupersampled(const Framebuffer& source, Framebuffer& destination,
                                              const ResolveDesc& desc, jobs::JobSystem* jobSystem,
                                              memory::Allocator& allocator) noexcept
{
    using Status = core::Status<core::Error>;

    if (!source.IsValid() || !destination.IsValid())
    {
        return Status::Err(core::Error{core::ErrorCode::InvalidArgument, "framebuffer is invalid"});
    }
    if (desc.factor < kMinSupersampleFactor || desc.factor > kMaxSupersampleFactor)
    {
        return Status::Err(
            core::Error{core::ErrorCode::InvalidArgument, "supersample factor out of range"});
    }
    if (source.Width() != destination.Width() * desc.factor ||
        source.Height() != destination.Height() * desc.factor)
    {
        return Status::Err(core::Error{core::ErrorCode::InvalidArgument,
                                       "source is not the destination scaled by the factor"});
    }
    if (&source == &destination)
    {
        return Status::Err(
            core::Error{core::ErrorCode::InvalidArgument, "resolve cannot be in place"});
    }

    const i32 destinationWidth = destination.Width();
    const i32 destinationHeight = destination.Height();
    const FilterKernel kernel = BuildFilterKernel(desc.filter, desc.factor);
    const i32 bandRows = std::clamp(desc.bandRows, kMinBandRows, kMaxBandRows);

    // Source rows one band needs: the band's own rows plus the vertical
    // window's reach on each side.
    const i32 bandSourceRows = bandRows * desc.factor + kernel.tapCount;
    const usize bandSamples =
        static_cast<usize>(bandSourceRows) * static_cast<usize>(destinationWidth);
    void* storage = allocator.Allocate(bandSamples * sizeof(Rgba), alignof(Rgba));
    if (storage == nullptr)
    {
        return Status::Err(
            core::Error{core::ErrorCode::Internal, "failed to allocate resolve band"});
    }
    Rgba* band = static_cast<Rgba*>(storage);

    const bool useJobs = jobSystem != nullptr && desc.useJobs;

    for (i32 bandStart = 0; bandStart < destinationHeight; bandStart += bandRows)
    {
        const i32 bandHeight = std::min(bandRows, destinationHeight - bandStart);
        const i32 firstSourceRow = bandStart * desc.factor + kernel.firstOffset;
        // Rows the band's first destination row through its last one reach:
        // the stride between them plus one full vertical window.
        const i32 sourceRowCount = (bandHeight - 1) * desc.factor + kernel.tapCount;

        HorizontalContext horizontal{};
        horizontal.source = &source;
        horizontal.band = band;
        horizontal.kernel = &kernel;
        horizontal.destinationWidth = destinationWidth;
        horizontal.factor = desc.factor;
        horizontal.firstSourceRow = firstSourceRow;

        const u32 rowCount = static_cast<u32>(sourceRowCount);
        if (useJobs && rowCount > 1)
        {
            jobSystem->ParallelFor(rowCount, 1, &FilterSourceRow, &horizontal);
        }
        else
        {
            for (u32 row = 0; row < rowCount; ++row)
            {
                FilterSourceRow(&horizontal, row);
            }
        }

        VerticalContext vertical{};
        vertical.destination = &destination;
        vertical.band = band;
        vertical.kernel = &kernel;
        vertical.destinationWidth = destinationWidth;
        vertical.factor = desc.factor;
        vertical.bandRowCount = sourceRowCount;
        vertical.firstSourceRow = firstSourceRow;
        vertical.firstDestinationRow = bandStart;
        vertical.clampNegative = desc.clampNegative;

        const u32 destinationRows = static_cast<u32>(bandHeight);
        if (useJobs && destinationRows > 1)
        {
            jobSystem->ParallelFor(destinationRows, 1, &FilterDestinationRow, &vertical);
        }
        else
        {
            for (u32 row = 0; row < destinationRows; ++row)
            {
                FilterDestinationRow(&vertical, row);
            }
        }
    }

    allocator.Deallocate(storage);
    return Status::Ok();
}

core::Result<AntiAliasTarget, core::Error>
AntiAliasTarget::Create(const AntiAliasTargetDesc& desc, memory::Allocator& allocator) noexcept
{
    using Created = core::Result<AntiAliasTarget, core::Error>;

    const i32 factor = desc.factor != 0 ? desc.factor : SupersampleFactorForQuality(desc.quality);
    if (factor < kMinSupersampleFactor || factor > kMaxSupersampleFactor)
    {
        return Created::Err(
            core::Error{core::ErrorCode::InvalidArgument, "supersample factor out of range"});
    }
    if (desc.width < 1 || desc.height < 1 || desc.width > kMaxTargetDimension ||
        desc.height > kMaxTargetDimension)
    {
        return Created::Err(
            core::Error{core::ErrorCode::InvalidArgument, "resolved dimensions out of range"});
    }
    if (desc.width > kMaxTargetDimension / factor || desc.height > kMaxTargetDimension / factor)
    {
        return Created::Err(core::Error{core::ErrorCode::InvalidArgument,
                                        "supersampled dimensions exceed the target limit"});
    }
    if (!IsValidPixelFormat(desc.format))
    {
        return Created::Err(core::Error{core::ErrorCode::InvalidArgument, "invalid pixel format"});
    }

    AntiAliasTarget target;
    target.m_width = desc.width;
    target.m_height = desc.height;
    target.m_factor = factor;
    target.m_filter = desc.filter;

    FramebufferDesc colorDesc{};
    colorDesc.width = desc.width * factor;
    colorDesc.height = desc.height * factor;
    colorDesc.format = desc.format;
    auto color = Framebuffer::Create(colorDesc, allocator);
    if (!color)
    {
        return Created::Err(std::move(color).Error());
    }
    target.m_color = std::move(color).Value();

    DepthBufferDesc depthDesc{};
    depthDesc.width = colorDesc.width;
    depthDesc.height = colorDesc.height;
    auto depth = DepthBuffer::Create(depthDesc, allocator);
    if (!depth)
    {
        return Created::Err(std::move(depth).Error());
    }
    target.m_depth = std::move(depth).Value();

    return Created::Ok(std::move(target));
}

bool AntiAliasTarget::IsValid() const noexcept
{
    return m_color.IsValid() && m_depth.IsValid() && m_factor >= kMinSupersampleFactor;
}

Viewport AntiAliasTarget::RenderViewport() const noexcept
{
    Viewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<f32>(RenderWidth());
    viewport.height = static_cast<f32>(RenderHeight());
    return viewport;
}

void AntiAliasTarget::Clear(const math::Color& color) noexcept
{
    if (!IsValid())
    {
        return;
    }
    m_color.Clear(color);
    m_depth.Clear();
}

core::Status<core::Error> AntiAliasTarget::Resolve(Framebuffer& destination,
                                                   jobs::JobSystem* jobSystem,
                                                   memory::Allocator& allocator,
                                                   const ResolveDesc& desc) const noexcept
{
    using Status = core::Status<core::Error>;
    if (!IsValid())
    {
        return Status::Err(
            core::Error{core::ErrorCode::InvalidArgument, "anti-alias target is invalid"});
    }
    ResolveDesc resolve = desc;
    resolve.factor = m_factor;
    resolve.filter = m_filter;
    return ResolveSupersampled(m_color, destination, resolve, jobSystem, allocator);
}

}  // namespace cyber::software
