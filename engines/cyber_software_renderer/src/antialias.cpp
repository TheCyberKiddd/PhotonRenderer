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

// Filters one source row horizontally into `destinationWidth` accumulated
// samples. RGBA32F rows are read directly and, away from the edges, without
// the per-tap clamp; every other format goes through GetPixel, which converts.
void FilterSourceRow(const Framebuffer& source, const FilterKernel& kernel, i32 factor,
                     i32 sourceRow, Rgba* output, i32 destinationWidth) noexcept
{
    const f32* floatRow = nullptr;
    if (source.Format() == PixelFormat::RGBA32F)
    {
        floatRow = reinterpret_cast<const f32*>(source.Row(sourceRow));
    }
    const i32 lastColumn = source.Width() - 1;
    const i32 taps = kernel.tapCount;

    for (i32 x = 0; x < destinationWidth; ++x)
    {
        const i32 base = x * factor + kernel.firstOffset;
        Rgba sum{};
        if (floatRow != nullptr && base >= 0 && base + taps - 1 <= lastColumn)
        {
            const f32* texel = floatRow + static_cast<usize>(base) * 4;
            for (i32 tap = 0; tap < taps; ++tap, texel += 4)
            {
                const f32 weight = kernel.weights[tap];
                sum.r += texel[0] * weight;
                sum.g += texel[1] * weight;
                sum.b += texel[2] * weight;
                sum.a += texel[3] * weight;
            }
        }
        else
        {
            for (i32 tap = 0; tap < taps; ++tap)
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
        }
        output[x] = sum;
    }
}

// One block of destination rows. The block streams down the image: each source
// row is filtered horizontally once into a ring of `tapCount` rows and every
// destination row that needs it reads it from there, so the scratch is a few
// rows rather than the image and the whole resolve is one job-system dispatch.
// Blocks overlap only by the vertical reach of the filter at their seams.
struct ResolveBlockContext
{
    const Framebuffer* source = nullptr;
    Framebuffer* destination = nullptr;
    const FilterKernel* kernel = nullptr;
    Rgba* scratch = nullptr;
    usize scratchSamplesPerBlock = 0;
    i32 destinationWidth = 0;
    i32 destinationHeight = 0;
    i32 factor = 0;
    i32 blockRows = 0;
    bool clampNegative = false;
};

void ResolveBlock(void* userData, u32 index) noexcept
{
    const ResolveBlockContext& context = *static_cast<const ResolveBlockContext*>(userData);
    const FilterKernel& kernel = *context.kernel;
    const i32 taps = kernel.tapCount;
    const i32 width = context.destinationWidth;
    const i32 sourceLastRow = context.source->Height() - 1;
    Rgba* ring = context.scratch + static_cast<usize>(index) * context.scratchSamplesPerBlock;

    const i32 firstRow = static_cast<i32>(index) * context.blockRows;
    const i32 endRow = std::min(context.destinationHeight, firstRow + context.blockRows);
    const auto slotOf = [taps](i32 unclampedRow) noexcept
    { return static_cast<usize>(((unclampedRow % taps) + taps) % taps); };

    // Highest unclamped source row already in the ring. Starts below any row a
    // block can need.
    i32 filteredUpTo = -(1 << 30);
    const Rgba* tapRows[kMaxFilterTaps]{};
    const bool floatDestination = context.destination->Format() == PixelFormat::RGBA32F;

    for (i32 row = firstRow; row < endRow; ++row)
    {
        const i32 firstTapRow = row * context.factor + kernel.firstOffset;
        const i32 lastTapRow = firstTapRow + taps - 1;
        for (i32 unclamped = std::max(filteredUpTo + 1, firstTapRow); unclamped <= lastTapRow;
             ++unclamped)
        {
            FilterSourceRow(*context.source, kernel, context.factor,
                            std::clamp(unclamped, 0, sourceLastRow),
                            ring + slotOf(unclamped) * static_cast<usize>(width), width);
        }
        filteredUpTo = lastTapRow;

        for (i32 tap = 0; tap < taps; ++tap)
        {
            tapRows[tap] = ring + slotOf(firstTapRow + tap) * static_cast<usize>(width);
        }
        // RGBA32F destinations are written in place; SetPixel would redo the
        // bounds check and the format switch for every pixel.
        f32* floatOut =
            floatDestination ? reinterpret_cast<f32*>(context.destination->Row(row)) : nullptr;
        for (i32 x = 0; x < width; ++x)
        {
            Rgba sum{};
            for (i32 tap = 0; tap < taps; ++tap)
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
            if (floatOut != nullptr)
            {
                f32* texel = floatOut + static_cast<usize>(x) * 4;
                texel[0] = sum.r;
                texel[1] = sum.g;
                texel[2] = sum.b;
                texel[3] = sum.a;
            }
            else
            {
                context.destination->SetPixel(x, row, math::Color{sum.r, sum.g, sum.b, sum.a});
            }
        }
    }
}

}  // namespace

f32 EvaluateResolveFilter(ResolveFilter filter, f32 offsetPixels) noexcept
{
    if (!math::IsFinite(offsetPixels))
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

    // A fixed block count: enough blocks to keep every worker busy and to
    // balance uneven rows, few enough that the seam overlap stays small. A
    // single block is the serial path.
    constexpr i32 kParallelBlocks = 16;
    const bool useJobs = jobSystem != nullptr && desc.useJobs && destinationHeight > 1;
    const i32 blockRows =
        useJobs ? std::max(1, (destinationHeight + kParallelBlocks - 1) / kParallelBlocks)
                : destinationHeight;
    const i32 blockCount = (destinationHeight + blockRows - 1) / blockRows;

    const usize samplesPerBlock =
        static_cast<usize>(kernel.tapCount) * static_cast<usize>(destinationWidth);
    void* storage = allocator.Allocate(
        samplesPerBlock * static_cast<usize>(blockCount) * sizeof(Rgba), alignof(Rgba));
    if (storage == nullptr)
    {
        return Status::Err(
            core::Error{core::ErrorCode::Internal, "failed to allocate resolve scratch"});
    }

    ResolveBlockContext context{};
    context.source = &source;
    context.destination = &destination;
    context.kernel = &kernel;
    context.scratch = static_cast<Rgba*>(storage);
    context.scratchSamplesPerBlock = samplesPerBlock;
    context.destinationWidth = destinationWidth;
    context.destinationHeight = destinationHeight;
    context.factor = desc.factor;
    context.blockRows = blockRows;
    context.clampNegative = desc.clampNegative;

    if (useJobs && blockCount > 1)
    {
        jobSystem->ParallelFor(static_cast<u32>(blockCount), 1, &ResolveBlock, &context);
    }
    else
    {
        for (i32 block = 0; block < blockCount; ++block)
        {
            ResolveBlock(&context, static_cast<u32>(block));
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
