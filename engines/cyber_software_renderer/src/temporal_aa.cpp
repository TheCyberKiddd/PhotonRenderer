#include <cyber/software/temporal_aa.h>

#include <cyber/jobs/job_system.h>
#include <cyber/software/post_process.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <utility>

namespace cyber::software
{
namespace
{

using core::Error;
using core::ErrorCode;
using math::Color;
using math::Vec2;

using Status = core::Status<Error>;

[[nodiscard]] Status InvalidArgument(const char* message) noexcept
{
    return Status::Err(Error{ErrorCode::InvalidArgument, message});
}

[[nodiscard]] f32 SafeFinite(f32 value, f32 fallback = 0.0f) noexcept
{
    return std::isfinite(value) ? value : fallback;
}

// YCoCg is the usual working space for a neighborhood clamp: it separates
// luminance from two chroma axes with three adds and two shifts worth of work,
// so the clip box is tight on brightness without dragging hue with it.
struct YCoCg
{
    f32 y = 0.0f;
    f32 co = 0.0f;
    f32 cg = 0.0f;
};

[[nodiscard]] YCoCg ToYCoCg(const Color& color) noexcept
{
    const f32 red = SafeFinite(color.r);
    const f32 green = SafeFinite(color.g);
    const f32 blue = SafeFinite(color.b);
    return YCoCg{0.25f * red + 0.5f * green + 0.25f * blue, 0.5f * (red - blue),
                 -0.25f * red + 0.5f * green - 0.25f * blue};
}

[[nodiscard]] Color FromYCoCg(const YCoCg& value, f32 alpha) noexcept
{
    const f32 temp = value.y - value.cg;
    return Color{temp + value.co, value.y + value.cg, temp - value.co, alpha};
}

[[nodiscard]] Color SampleBilinearClamped(const Framebuffer& image, f32 x, f32 y) noexcept
{
    const i32 width = image.Width();
    const i32 height = image.Height();
    const f32 clampedX = std::clamp(SafeFinite(x), 0.0f, static_cast<f32>(width - 1));
    const f32 clampedY = std::clamp(SafeFinite(y), 0.0f, static_cast<f32>(height - 1));
    const i32 x0 = static_cast<i32>(std::floor(clampedX));
    const i32 y0 = static_cast<i32>(std::floor(clampedY));
    const i32 x1 = std::min(x0 + 1, width - 1);
    const i32 y1 = std::min(y0 + 1, height - 1);
    const f32 fx = clampedX - static_cast<f32>(x0);
    const f32 fy = clampedY - static_cast<f32>(y0);

    const Color c00 = image.GetPixel(x0, y0);
    const Color c10 = image.GetPixel(x1, y0);
    const Color c01 = image.GetPixel(x0, y1);
    const Color c11 = image.GetPixel(x1, y1);
    const f32 w00 = (1.0f - fx) * (1.0f - fy);
    const f32 w10 = fx * (1.0f - fy);
    const f32 w01 = (1.0f - fx) * fy;
    const f32 w11 = fx * fy;
    return Color{c00.r * w00 + c10.r * w10 + c01.r * w01 + c11.r * w11,
                 c00.g * w00 + c10.g * w10 + c01.g * w01 + c11.g * w11,
                 c00.b * w00 + c10.b * w10 + c01.b * w01 + c11.b * w11,
                 c00.a * w00 + c10.a * w10 + c01.a * w01 + c11.a * w11};
}

struct TemporalContext
{
    const Framebuffer* color = nullptr;
    const Framebuffer* velocity = nullptr;
    const Framebuffer* history = nullptr;
    const DepthBuffer* depth = nullptr;
    const Framebuffer* depthHistory = nullptr;
    Framebuffer* nextDepthHistory = nullptr;
    const Framebuffer* signedDrift = nullptr;
    const Framebuffer* absoluteDrift = nullptr;
    Framebuffer* nextSignedDrift = nullptr;
    Framebuffer* nextAbsoluteDrift = nullptr;
    Framebuffer* destination = nullptr;
    f32 historyWeight = 0.9f;
    f32 varianceGamma = 1.0f;
    f32 flickerRelaxation = 1.0f;
    f32 disocclusionPixels = 64.0f;
    f32 depthRejectRelative = 0.03f;
    f32 nearPlane = 0.1f;
    f32 farPlane = 1000.0f;
    bool testDepth = false;
    bool clampHistory = true;
    bool useVelocity = true;
    bool luminanceWeighting = true;
    i32 width = 0;
    i32 height = 0;
    std::atomic<u32>* historyPixels = nullptr;
    std::atomic<u32>* rejectedPixels = nullptr;
    std::atomic<u32>* clampedPixels = nullptr;
    std::atomic<u32>* relaxedPixels = nullptr;
    std::atomic<u32>* depthRejectedPixels = nullptr;
};

void TemporalRow(void* userData, u32 index) noexcept
{
    auto& context = *static_cast<TemporalContext*>(userData);
    const i32 y = static_cast<i32>(index);

    u32 rowHistory = 0;
    u32 rowRejected = 0;
    u32 rowClamped = 0;
    u32 rowRelaxed = 0;
    u32 rowDepthRejected = 0;

    const f32 driftRate = 1.0f - context.historyWeight;
    const Color kNoDrift{0.0f, 0.0f, 0.0f, 0.0f};

    for (i32 x = 0; x < context.width; ++x)
    {
        const Color current = context.color->GetPixel(x, y);

        // Neighborhood statistics of the current frame. The mean and the
        // second moment give the variance interval, and the hard min and max
        // give the extent; the clip box below is their union.
        YCoCg sum{};
        YCoCg sumSquared{};
        YCoCg minimum{1.0e30f, 1.0e30f, 1.0e30f};
        YCoCg maximum{-1.0e30f, -1.0e30f, -1.0e30f};
        f32 sampleCount = 0.0f;
        for (i32 offsetY = -1; offsetY <= 1; ++offsetY)
        {
            const i32 sampleY = std::clamp(y + offsetY, 0, context.height - 1);
            for (i32 offsetX = -1; offsetX <= 1; ++offsetX)
            {
                const i32 sampleX = std::clamp(x + offsetX, 0, context.width - 1);
                const YCoCg sample = ToYCoCg(context.color->GetPixel(sampleX, sampleY));
                sum.y += sample.y;
                sum.co += sample.co;
                sum.cg += sample.cg;
                sumSquared.y += sample.y * sample.y;
                sumSquared.co += sample.co * sample.co;
                sumSquared.cg += sample.cg * sample.cg;
                minimum.y = std::min(minimum.y, sample.y);
                minimum.co = std::min(minimum.co, sample.co);
                minimum.cg = std::min(minimum.cg, sample.cg);
                maximum.y = std::max(maximum.y, sample.y);
                maximum.co = std::max(maximum.co, sample.co);
                maximum.cg = std::max(maximum.cg, sample.cg);
                sampleCount += 1.0f;
            }
        }

        const f32 inverseCount = 1.0f / sampleCount;
        const YCoCg mean{sum.y * inverseCount, sum.co * inverseCount, sum.cg * inverseCount};
        const YCoCg meanSquared{sumSquared.y * inverseCount, sumSquared.co * inverseCount,
                                sumSquared.cg * inverseCount};
        const YCoCg deviation{
            std::sqrt(std::max(0.0f, meanSquared.y - mean.y * mean.y)),
            std::sqrt(std::max(0.0f, meanSquared.co - mean.co * mean.co)),
            std::sqrt(std::max(0.0f, meanSquared.cg - mean.cg * mean.cg))};

        f32 velocityX = 0.0f;
        f32 velocityY = 0.0f;
        if (context.useVelocity && context.velocity != nullptr)
        {
            const Color sample = context.velocity->GetPixel(x, y);
            velocityX = SafeFinite(sample.r);
            velocityY = SafeFinite(sample.g);
        }

        const f32 motion = std::sqrt(velocityX * velocityX + velocityY * velocityY);
        const f32 sourceX = static_cast<f32>(x) - velocityX;
        const f32 sourceY = static_cast<f32>(y) - velocityY;
        const bool offScreen = sourceX < -0.5f || sourceY < -0.5f ||
                               sourceX > static_cast<f32>(context.width) - 0.5f ||
                               sourceY > static_cast<f32>(context.height) - 0.5f;
        // The surface this pixel shows now, and the one its history says it
        // showed. A velocity buffer cannot tell these apart: a pixel of floor
        // that an occluder was covering carries the floor's velocity, which
        // reprojects it onto the occluder, and the occluder's color is reused
        // for the floor. Comparing the distances is what catches it.
        f32 currentDistance = 0.0f;
        bool depthRejected = false;
        if (context.testDepth)
        {
            currentDistance = LinearizeReverseZ(context.depth->Get(x, y), context.nearPlane,
                                                context.farPlane);
        }
        if (context.testDepth && !offScreen)
        {
            const f32 historyDistance =
                SafeFinite(SampleBilinearClamped(*context.depthHistory, sourceX, sourceY).r,
                           context.farPlane);
            // The tolerance is a fraction of the distance, because depth
            // precision and the parallax a frame of motion produces both scale
            // with it.
            const f32 tolerance =
                context.depthRejectRelative * std::max(currentDistance, context.nearPlane);
            depthRejected = std::abs(historyDistance - currentDistance) > tolerance;
        }

        // A disocclusion is history that describes geometry this pixel no
        // longer shows. None of these cases can be corrected, only dropped.
        const bool reject = offScreen || motion > context.disocclusionPixels || depthRejected;

        if (reject)
        {
            ++rowRejected;
            if (depthRejected && !offScreen && motion <= context.disocclusionPixels)
            {
                ++rowDepthRejected;
            }
            if (context.testDepth)
            {
                context.nextDepthHistory->SetPixel(
                    x, y, Color{currentDistance, 0.0f, 0.0f, 0.0f});
            }
            context.destination->SetPixel(x, y, current);
            // History that was dropped says nothing about how this pixel has
            // been disagreeing, so the drift restarts with it.
            context.nextSignedDrift->SetPixel(x, y, kNoDrift);
            context.nextAbsoluteDrift->SetPixel(x, y, kNoDrift);
            continue;
        }

        if (context.testDepth)
        {
            context.nextDepthHistory->SetPixel(x, y, Color{currentDistance, 0.0f, 0.0f, 0.0f});
        }

        Color history = SampleBilinearClamped(*context.history, sourceX, sourceY);
        const YCoCg rawHistoryYCoCg = ToYCoCg(history);
        YCoCg historyYCoCg = rawHistoryYCoCg;

        // Drift measured up to the previous frame, reprojected alongside the
        // color it describes. It is read before this frame's difference is
        // folded in, so the frame a cut arrives on is judged by the pixel's
        // record before the cut, which is what makes the cut rejectable.
        const Color priorSigned = SampleBilinearClamped(*context.signedDrift, sourceX, sourceY);
        const Color priorAbsolute =
            SampleBilinearClamped(*context.absoluteDrift, sourceX, sourceY);

        if (context.clampHistory)
        {
            // The box is the union of the neighborhood's hard extent and its
            // variance interval, not their intersection. A flat neighborhood
            // has no variance, so the box collapses onto the hard extent and a
            // cut is still rejected outright; a neighborhood that disagrees
            // with itself widens past its own extent, which is where the
            // sub-pixel detail a single jittered sample missed lives.
            YCoCg low{std::min(minimum.y, mean.y - deviation.y * context.varianceGamma),
                      std::min(minimum.co, mean.co - deviation.co * context.varianceGamma),
                      std::min(minimum.cg, mean.cg - deviation.cg * context.varianceGamma)};
            YCoCg high{std::max(maximum.y, mean.y + deviation.y * context.varianceGamma),
                       std::max(maximum.co, mean.co + deviation.co * context.varianceGamma),
                       std::max(maximum.cg, mean.cg + deviation.cg * context.varianceGamma)};

            // Oscillation amplitude: how much this pixel has been disagreeing
            // with its history, less the part of that disagreement that
            // pointed consistently one way. A cut is all signed drift and
            // widens nothing; sub-pixel flicker is all amplitude.
            if (context.flickerRelaxation > 0.0f)
            {
                const f32 amplitude[3] = {
                    std::max(0.0f, std::abs(SafeFinite(priorAbsolute.r)) -
                                       std::abs(SafeFinite(priorSigned.r))) *
                        context.flickerRelaxation,
                    std::max(0.0f, std::abs(SafeFinite(priorAbsolute.g)) -
                                       std::abs(SafeFinite(priorSigned.g))) *
                        context.flickerRelaxation,
                    std::max(0.0f, std::abs(SafeFinite(priorAbsolute.b)) -
                                       std::abs(SafeFinite(priorSigned.b))) *
                        context.flickerRelaxation};
                if (amplitude[0] > 0.0f || amplitude[1] > 0.0f || amplitude[2] > 0.0f)
                {
                    low.y -= amplitude[0];
                    high.y += amplitude[0];
                    low.co -= amplitude[1];
                    high.co += amplitude[1];
                    low.cg -= amplitude[2];
                    high.cg += amplitude[2];
                    ++rowRelaxed;
                }
            }

            // The sample is pulled back per channel rather than clipped along
            // the segment to the box center. Clipping is the textbook form and
            // preserves hue, but the two chroma axes of a near-grey
            // neighborhood are nearly degenerate, so the smallest chroma
            // disagreement dominates the clip ratio and drags luminance to the
            // center of the box with it. Measured on the acceptance scene,
            // clipping costs about a third of the converged accuracy that the
            // per-channel form keeps.
            const YCoCg clamped{std::clamp(historyYCoCg.y, low.y, high.y),
                                std::clamp(historyYCoCg.co, low.co, high.co),
                                std::clamp(historyYCoCg.cg, low.cg, high.cg)};
            if (clamped.y != historyYCoCg.y || clamped.co != historyYCoCg.co ||
                clamped.cg != historyYCoCg.cg)
            {
                ++rowClamped;
            }
            historyYCoCg = clamped;
            history = FromYCoCg(historyYCoCg, history.a);
        }

        // Fold this frame's disagreement into the drift the next frame reads.
        // The difference is measured against the history as it arrived, before
        // rectification, because rectification is the thing being decided.
        const YCoCg currentYCoCg = ToYCoCg(current);
        const f32 difference[3] = {currentYCoCg.y - rawHistoryYCoCg.y,
                                   currentYCoCg.co - rawHistoryYCoCg.co,
                                   currentYCoCg.cg - rawHistoryYCoCg.cg};
        context.nextSignedDrift->SetPixel(
            x, y,
            Color{SafeFinite(priorSigned.r) + (difference[0] - SafeFinite(priorSigned.r)) * driftRate,
                  SafeFinite(priorSigned.g) + (difference[1] - SafeFinite(priorSigned.g)) * driftRate,
                  SafeFinite(priorSigned.b) + (difference[2] - SafeFinite(priorSigned.b)) * driftRate,
                  0.0f});
        context.nextAbsoluteDrift->SetPixel(
            x, y,
            Color{SafeFinite(priorAbsolute.r) +
                      (std::abs(difference[0]) - SafeFinite(priorAbsolute.r)) * driftRate,
                  SafeFinite(priorAbsolute.g) +
                      (std::abs(difference[1]) - SafeFinite(priorAbsolute.g)) * driftRate,
                  SafeFinite(priorAbsolute.b) +
                      (std::abs(difference[2]) - SafeFinite(priorAbsolute.b)) * driftRate,
                  0.0f});

        f32 currentWeight = 1.0f - context.historyWeight;
        f32 historyWeight = context.historyWeight;
        if (context.luminanceWeighting)
        {
            // Weighting by 1 / (1 + luma) averages in a perceptual space, so a
            // single bright sample cannot dominate the result and flicker as
            // it moves between pixels.
            currentWeight /= 1.0f + std::max(0.0f, Luminance(current));
            historyWeight /= 1.0f + std::max(0.0f, Luminance(history));
        }
        const f32 totalWeight = currentWeight + historyWeight;
        if (!(totalWeight > 0.0f))
        {
            context.destination->SetPixel(x, y, current);
            continue;
        }
        const f32 inverseWeight = 1.0f / totalWeight;
        const Color resolved{
            (current.r * currentWeight + history.r * historyWeight) * inverseWeight,
            (current.g * currentWeight + history.g * historyWeight) * inverseWeight,
            (current.b * currentWeight + history.b * historyWeight) * inverseWeight,
            (current.a * currentWeight + history.a * historyWeight) * inverseWeight};
        context.destination->SetPixel(x, y, resolved);
        ++rowHistory;
    }

    context.historyPixels->fetch_add(rowHistory, std::memory_order_relaxed);
    context.rejectedPixels->fetch_add(rowRejected, std::memory_order_relaxed);
    context.clampedPixels->fetch_add(rowClamped, std::memory_order_relaxed);
    context.relaxedPixels->fetch_add(rowRelaxed, std::memory_order_relaxed);
    context.depthRejectedPixels->fetch_add(rowDepthRejected, std::memory_order_relaxed);
}

struct FxaaContext
{
    const Framebuffer* source = nullptr;
    Framebuffer* destination = nullptr;
    f32 relativeThreshold = 0.125f;
    f32 absoluteThreshold = 0.0312f;
    f32 subpixelBlend = 0.75f;
    i32 width = 0;
    i32 height = 0;
};

[[nodiscard]] f32 SampleLuma(const Framebuffer& image, i32 x, i32 y, i32 width,
                             i32 height) noexcept
{
    const i32 clampedX = std::clamp(x, 0, width - 1);
    const i32 clampedY = std::clamp(y, 0, height - 1);
    return Luminance(image.GetPixel(clampedX, clampedY));
}

void FxaaRow(void* userData, u32 index) noexcept
{
    auto& context = *static_cast<FxaaContext*>(userData);
    const i32 y = static_cast<i32>(index);
    const i32 width = context.width;
    const i32 height = context.height;

    for (i32 x = 0; x < width; ++x)
    {
        const f32 center = SampleLuma(*context.source, x, y, width, height);
        const f32 north = SampleLuma(*context.source, x, y - 1, width, height);
        const f32 south = SampleLuma(*context.source, x, y + 1, width, height);
        const f32 west = SampleLuma(*context.source, x - 1, y, width, height);
        const f32 east = SampleLuma(*context.source, x + 1, y, width, height);

        const f32 highest = std::max({center, north, south, west, east});
        const f32 lowest = std::min({center, north, south, west, east});
        const f32 contrast = highest - lowest;
        // Below either threshold the pixel is flat enough that filtering it
        // would only soften detail.
        if (contrast < std::max(context.absoluteThreshold, context.relativeThreshold * highest))
        {
            context.destination->SetPixel(x, y, context.source->GetPixel(x, y));
            continue;
        }

        const f32 northWest = SampleLuma(*context.source, x - 1, y - 1, width, height);
        const f32 northEast = SampleLuma(*context.source, x + 1, y - 1, width, height);
        const f32 southWest = SampleLuma(*context.source, x - 1, y + 1, width, height);
        const f32 southEast = SampleLuma(*context.source, x + 1, y + 1, width, height);

        // Which way the edge runs: compare the vertical and horizontal second
        // derivatives of luminance across the 3x3 block.
        const f32 horizontal = std::abs(north + south - 2.0f * center) * 2.0f +
                               std::abs(northEast + southEast - 2.0f * east) +
                               std::abs(northWest + southWest - 2.0f * west);
        const f32 vertical = std::abs(west + east - 2.0f * center) * 2.0f +
                             std::abs(northEast + northWest - 2.0f * north) +
                             std::abs(southEast + southWest - 2.0f * south);
        const bool horizontalEdge = horizontal >= vertical;

        // Step toward the steeper side of the edge: that is the side the
        // center pixel should move toward to turn the step into a ramp.
        const f32 negative = horizontalEdge ? north : west;
        const f32 positive = horizontalEdge ? south : east;
        const f32 gradientNegative = std::abs(negative - center);
        const f32 gradientPositive = std::abs(positive - center);
        const bool steeperPositive = gradientPositive > gradientNegative;
        const i32 stepX = horizontalEdge ? 0 : (steeperPositive ? 1 : -1);
        const i32 stepY = horizontalEdge ? (steeperPositive ? 1 : -1) : 0;

        // Walk along the edge in both directions until the edge ends, which is
        // where the luminance stops matching the step this pixel sits on.
        // How far along the edge the pixel lies is what decides the blend: a
        // pixel near an end barely moves, one in the middle moves half a
        // pixel. Without this search the filter can only soften a step
        // slightly; with it, the staircase becomes a ramp.
        const f32 edgeLuma = 0.5f * (center + (steeperPositive ? positive : negative));
        const f32 gradientThreshold =
            0.25f * std::max(gradientNegative, gradientPositive);
        const i32 alongX = horizontalEdge ? 1 : 0;
        const i32 alongY = horizontalEdge ? 0 : 1;

        constexpr i32 kMaxEdgeSteps = 12;
        f32 distanceNegative = static_cast<f32>(kMaxEdgeSteps);
        f32 distancePositive = static_cast<f32>(kMaxEdgeSteps);
        bool foundNegative = false;
        bool foundPositive = false;
        for (i32 step = 1; step <= kMaxEdgeSteps; ++step)
        {
            if (!foundNegative)
            {
                // The luma of the edge itself, sampled between the two rows or
                // columns it separates.
                const f32 sampleCenter =
                    SampleLuma(*context.source, x - alongX * step, y - alongY * step, width, height);
                const f32 sampleSide =
                    SampleLuma(*context.source, x - alongX * step + stepX,
                               y - alongY * step + stepY, width, height);
                const f32 along = 0.5f * (sampleCenter + sampleSide);
                if (std::abs(along - edgeLuma) >= gradientThreshold)
                {
                    distanceNegative = static_cast<f32>(step);
                    foundNegative = true;
                }
            }
            if (!foundPositive)
            {
                const f32 sampleCenter =
                    SampleLuma(*context.source, x + alongX * step, y + alongY * step, width, height);
                const f32 sampleSide =
                    SampleLuma(*context.source, x + alongX * step + stepX,
                               y + alongY * step + stepY, width, height);
                const f32 along = 0.5f * (sampleCenter + sampleSide);
                if (std::abs(along - edgeLuma) >= gradientThreshold)
                {
                    distancePositive = static_cast<f32>(step);
                    foundPositive = true;
                }
            }
            if (foundNegative && foundPositive)
            {
                break;
            }
        }

        const f32 edgeLength = distanceNegative + distancePositive;
        const f32 nearestEnd = std::min(distanceNegative, distancePositive);
        const f32 edgeBlend = std::max(0.0f, 0.5f - nearestEnd / std::max(1.0e-5f, edgeLength));

        // Sub-pixel term: how far the center sits from its local average,
        // relative to the local contrast. It catches the single-pixel features
        // an edge walk cannot see.
        const f32 average = (2.0f * (north + south + west + east) + northWest + northEast +
                             southWest + southEast) /
                            12.0f;
        const f32 subpixel = std::clamp(std::abs(average - center) / std::max(1.0e-5f, contrast),
                                        0.0f, 1.0f);
        const f32 smoothed = subpixel * subpixel * (3.0f - 2.0f * subpixel);
        const f32 subpixelAmount = smoothed * smoothed * context.subpixelBlend;

        // The blend is at most half a pixel: beyond that the filter would pull
        // a pixel past its neighbor and soften the image instead of the edge.
        const f32 blend = std::min(0.5f, std::max(edgeBlend, subpixelAmount));

        const Color centerColor = context.source->GetPixel(x, y);
        const Color neighbor = context.source->GetPixel(std::clamp(x + stepX, 0, width - 1),
                                                        std::clamp(y + stepY, 0, height - 1));
        context.destination->SetPixel(
            x, y,
            Color{centerColor.r + (neighbor.r - centerColor.r) * blend,
                  centerColor.g + (neighbor.g - centerColor.g) * blend,
                  centerColor.b + (neighbor.b - centerColor.b) * blend, centerColor.a});
    }
}

// Linearizes a reverse-Z depth buffer into the red channel of a float image,
// which is the form the resolver reprojects it in.
void StoreLinearDepth(const DepthBuffer& depth, const TemporalAaSettings& settings,
                      Framebuffer& destination) noexcept
{
    for (i32 y = 0; y < destination.Height(); ++y)
    {
        for (i32 x = 0; x < destination.Width(); ++x)
        {
            const f32 distance =
                LinearizeReverseZ(depth.Get(x, y), settings.nearPlane, settings.farPlane);
            destination.SetPixel(x, y, Color{distance, 0.0f, 0.0f, 0.0f});
        }
    }
}

void CopyImage(const Framebuffer& source, Framebuffer& destination) noexcept
{
    for (i32 y = 0; y < source.Height(); ++y)
    {
        for (i32 x = 0; x < source.Width(); ++x)
        {
            destination.SetPixel(x, y, source.GetPixel(x, y));
        }
    }
}

}  // namespace

const char* AntiAliasMethodName(AntiAliasMethod method) noexcept
{
    switch (method)
    {
    case AntiAliasMethod::None:
        return "none";
    case AntiAliasMethod::Supersample:
        return "supersample";
    case AntiAliasMethod::Fxaa:
        return "fxaa";
    case AntiAliasMethod::Temporal:
        return "temporal";
    }
    return "unknown";
}

f32 RadicalInverse(u32 index, u32 base) noexcept
{
    if (base < 2u)
    {
        return 0.0f;
    }
    double result = 0.0;
    double fraction = 1.0 / static_cast<double>(base);
    u32 remaining = index;
    while (remaining > 0u)
    {
        result += static_cast<double>(remaining % base) * fraction;
        remaining /= base;
        fraction /= static_cast<double>(base);
    }
    return static_cast<f32>(result);
}

math::Vec2 HaltonPoint(u32 index) noexcept
{
    // Index 0 of the sequence is the origin, which would be an unjittered
    // frame; starting at 1 makes every phase a distinct sub-pixel position.
    const u32 sequenceIndex = index + 1u;
    return Vec2{RadicalInverse(sequenceIndex, 2u), RadicalInverse(sequenceIndex, 3u)};
}

math::Vec2 TemporalJitterOffset(u32 frameIndex, i32 phaseCount) noexcept
{
    const i32 phases = std::clamp(phaseCount, 1, kMaxJitterPhaseCount);
    if (phases == 1)
    {
        return Vec2{0.0f, 0.0f};
    }
    const Vec2 point = HaltonPoint(frameIndex % static_cast<u32>(phases));
    return Vec2{point.x - 0.5f, point.y - 0.5f};
}

math::Mat4 ApplyJitterToProjection(const math::Mat4& projection, const math::Vec2& offsetPixels,
                                   i32 width, i32 height) noexcept
{
    if (width <= 0 || height <= 0 || !std::isfinite(offsetPixels.x) ||
        !std::isfinite(offsetPixels.y))
    {
        return projection;
    }

    // Pixels to normalized device coordinates. Y is negated because pixel rows
    // run down while clip space runs up.
    const f32 offsetNdcX = 2.0f * offsetPixels.x / static_cast<f32>(width);
    const f32 offsetNdcY = -2.0f * offsetPixels.y / static_cast<f32>(height);

    // clip.xy += offset * clip.w, expressed by folding the w row into the x
    // and y rows. This holds for perspective and orthographic alike and leaves
    // clip.z untouched.
    math::Mat4 jittered = projection;
    for (i32 column = 0; column < 4; ++column)
    {
        const f32 w = projection.columns[column].w;
        jittered.columns[column].x += offsetNdcX * w;
        jittered.columns[column].y += offsetNdcY * w;
    }
    return jittered;
}

core::Result<TemporalAaResolver, Error> TemporalAaResolver::Create(
    i32 width, i32 height, memory::Allocator& allocator) noexcept
{
    using Created = core::Result<TemporalAaResolver, Error>;

    if (width < 1 || width > 16384 || height < 1 || height > 16384)
    {
        return Created::Err(Error{ErrorCode::InvalidArgument, "resolver dimensions out of range"});
    }

    FramebufferDesc desc{};
    desc.width = width;
    desc.height = height;
    desc.format = PixelFormat::RGBA32F;
    auto history = Framebuffer::Create(desc, allocator);
    if (!history)
    {
        return Created::Err(history.Error());
    }

    TemporalAaResolver resolver;
    resolver.m_history = std::move(history).Value();
    for (i32 index = 0; index < 2; ++index)
    {
        auto signedDrift = Framebuffer::Create(desc, allocator);
        if (!signedDrift)
        {
            return Created::Err(signedDrift.Error());
        }
        auto absoluteDrift = Framebuffer::Create(desc, allocator);
        if (!absoluteDrift)
        {
            return Created::Err(absoluteDrift.Error());
        }
        auto depthHistory = Framebuffer::Create(desc, allocator);
        if (!depthHistory)
        {
            return Created::Err(depthHistory.Error());
        }
        resolver.m_signedDrift[index] = std::move(signedDrift).Value();
        resolver.m_absoluteDrift[index] = std::move(absoluteDrift).Value();
        resolver.m_depthHistory[index] = std::move(depthHistory).Value();
    }
    resolver.m_width = width;
    resolver.m_height = height;
    resolver.m_hasHistory = false;
    return Created::Ok(std::move(resolver));
}

bool TemporalAaResolver::IsValid() const noexcept
{
    return m_history.IsValid();
}

i32 TemporalAaResolver::Width() const noexcept
{
    return m_width;
}

i32 TemporalAaResolver::Height() const noexcept
{
    return m_height;
}

bool TemporalAaResolver::HasHistory() const noexcept
{
    return m_hasHistory;
}

void TemporalAaResolver::Reset() noexcept
{
    m_hasHistory = false;
    m_hasDepthHistory = false;
}

const Framebuffer& TemporalAaResolver::History() const noexcept
{
    return m_history;
}

Status TemporalAaResolver::Resolve(const Framebuffer& color, const Framebuffer* velocity,
                                   Framebuffer& destination, const TemporalAaSettings& settings,
                                   jobs::JobSystem* jobSystem, TemporalAaStats* stats) noexcept
{
    TemporalAaInputs inputs{};
    inputs.color = &color;
    inputs.velocity = velocity;
    return Resolve(inputs, destination, settings, jobSystem, stats);
}

Status TemporalAaResolver::Resolve(const TemporalAaInputs& inputs, Framebuffer& destination,
                                   const TemporalAaSettings& settings, jobs::JobSystem* jobSystem,
                                   TemporalAaStats* stats) noexcept
{
    if (inputs.color == nullptr)
    {
        return InvalidArgument("temporal resolve needs a color buffer");
    }
    const Framebuffer& color = *inputs.color;
    const Framebuffer* velocity = inputs.velocity;

    if (!IsValid())
    {
        return InvalidArgument("resolver is invalid");
    }
    if (!color.IsValid() || color.Width() != m_width || color.Height() != m_height)
    {
        return InvalidArgument("temporal source does not match the resolver size");
    }
    if (!destination.IsValid() || destination.Width() != m_width ||
        destination.Height() != m_height)
    {
        return InvalidArgument("temporal destination does not match the resolver size");
    }
    if (&color == &destination)
    {
        return InvalidArgument("temporal resolve cannot be in place");
    }
    if (!std::isfinite(settings.historyWeight) || settings.historyWeight < 0.0f ||
        settings.historyWeight >= 1.0f)
    {
        return InvalidArgument("history weight must be in [0, 1)");
    }
    if (velocity != nullptr &&
        (!velocity->IsValid() || velocity->Width() != m_width || velocity->Height() != m_height))
    {
        return InvalidArgument("velocity buffer does not match the resolver size");
    }
    if (inputs.depth != nullptr && (!inputs.depth->IsValid() ||
                                    inputs.depth->Width() != m_width ||
                                    inputs.depth->Height() != m_height))
    {
        return InvalidArgument("depth buffer does not match the resolver size");
    }

    // The depth test needs the previous frame's depth as well as this one's,
    // so it starts working one frame after depth is first supplied rather than
    // on the frame it appears.
    const bool testDepth = inputs.depth != nullptr && m_hasDepthHistory;

    TemporalAaStats local{};

    if (!m_hasHistory)
    {
        // Nothing to reproject: the first frame is its own result and becomes
        // the history the next frame accumulates onto.
        CopyImage(color, destination);
        CopyImage(color, m_history);
        // No history means no record of disagreement either, so the first
        // frame after a reset relaxes nothing.
        const math::Color cleared{0.0f, 0.0f, 0.0f, 0.0f};
        m_signedDrift[0].Clear(cleared);
        m_signedDrift[1].Clear(cleared);
        m_absoluteDrift[0].Clear(cleared);
        m_absoluteDrift[1].Clear(cleared);
        m_driftIndex = 0;
        m_hasHistory = true;
        m_hasDepthHistory = false;
        if (inputs.depth != nullptr)
        {
            StoreLinearDepth(*inputs.depth, settings, m_depthHistory[0]);
            m_hasDepthHistory = true;
        }
        local.historyPrimed = true;
        if (stats != nullptr)
        {
            *stats = local;
        }
        return Status::Ok();
    }

    std::atomic<u32> historyPixels{0u};
    std::atomic<u32> rejectedPixels{0u};
    std::atomic<u32> clampedPixels{0u};
    std::atomic<u32> relaxedPixels{0u};
    std::atomic<u32> depthRejectedPixels{0u};

    const i32 nextDriftIndex = 1 - m_driftIndex;

    TemporalContext context{};
    context.color = &color;
    context.velocity = velocity;
    context.history = &m_history;
    context.signedDrift = &m_signedDrift[m_driftIndex];
    context.absoluteDrift = &m_absoluteDrift[m_driftIndex];
    context.nextSignedDrift = &m_signedDrift[nextDriftIndex];
    context.nextAbsoluteDrift = &m_absoluteDrift[nextDriftIndex];
    context.depth = inputs.depth;
    context.depthHistory = &m_depthHistory[m_driftIndex];
    context.nextDepthHistory = &m_depthHistory[nextDriftIndex];
    context.testDepth = testDepth;
    context.depthRejectRelative =
        std::max(0.0f, SafeFinite(settings.depthRejectRelative, 0.03f));
    context.nearPlane = settings.nearPlane;
    context.farPlane = settings.farPlane;
    context.destination = &destination;
    context.historyWeight = settings.historyWeight;
    context.varianceGamma = std::max(0.0f, SafeFinite(settings.varianceGamma, 1.0f));
    context.flickerRelaxation = std::max(0.0f, SafeFinite(settings.flickerRelaxation, 1.0f));
    context.disocclusionPixels = std::max(0.0f, SafeFinite(settings.disocclusionPixels, 64.0f));
    context.clampHistory = settings.clampHistory;
    context.useVelocity = settings.useVelocity;
    context.luminanceWeighting = settings.luminanceWeighting;
    context.width = m_width;
    context.height = m_height;
    context.historyPixels = &historyPixels;
    context.rejectedPixels = &rejectedPixels;
    context.clampedPixels = &clampedPixels;
    context.relaxedPixels = &relaxedPixels;
    context.depthRejectedPixels = &depthRejectedPixels;

    const u32 rows = static_cast<u32>(m_height);
    if (settings.useJobs && jobSystem != nullptr && rows > 1u)
    {
        jobSystem->ParallelFor(rows, 1u, &TemporalRow, &context);
    }
    else
    {
        for (u32 row = 0; row < rows; ++row)
        {
            TemporalRow(&context, row);
        }
    }

    // The resolve is also the history for the next frame, which is what makes
    // the accumulation recursive rather than a two-frame average.
    CopyImage(destination, m_history);
    if (inputs.depth != nullptr && !testDepth)
    {
        // Depth arrived this frame but there was nothing to compare it with.
        // Record it so the next frame can.
        StoreLinearDepth(*inputs.depth, settings, m_depthHistory[nextDriftIndex]);
    }
    m_hasDepthHistory = inputs.depth != nullptr;
    m_driftIndex = nextDriftIndex;

    local.historyPixels = historyPixels.load(std::memory_order_relaxed);
    local.rejectedPixels = rejectedPixels.load(std::memory_order_relaxed);
    local.clampedPixels = clampedPixels.load(std::memory_order_relaxed);
    local.relaxedPixels = relaxedPixels.load(std::memory_order_relaxed);
    local.depthRejectedPixels = depthRejectedPixels.load(std::memory_order_relaxed);
    if (stats != nullptr)
    {
        *stats = local;
    }
    return Status::Ok();
}

Status FxaaResolve(const Framebuffer& source, Framebuffer& destination,
                   const FxaaSettings& settings, jobs::JobSystem* jobSystem) noexcept
{
    if (!source.IsValid() || !destination.IsValid())
    {
        return InvalidArgument("fxaa framebuffer is invalid");
    }
    if (source.Width() != destination.Width() || source.Height() != destination.Height())
    {
        return InvalidArgument("fxaa source and destination differ in size");
    }
    if (&source == &destination)
    {
        return InvalidArgument("fxaa cannot run in place");
    }

    FxaaContext context{};
    context.source = &source;
    context.destination = &destination;
    context.relativeThreshold =
        std::clamp(SafeFinite(settings.relativeThreshold, 0.125f), 0.0f, 1.0f);
    context.absoluteThreshold =
        std::clamp(SafeFinite(settings.absoluteThreshold, 0.0312f), 0.0f, 1.0f);
    context.subpixelBlend = std::clamp(SafeFinite(settings.subpixelBlend, 0.75f), 0.0f, 1.0f);
    context.width = source.Width();
    context.height = source.Height();

    const u32 rows = static_cast<u32>(context.height);
    if (settings.useJobs && jobSystem != nullptr && rows > 1u)
    {
        jobSystem->ParallelFor(rows, 1u, &FxaaRow, &context);
    }
    else
    {
        for (u32 row = 0; row < rows; ++row)
        {
            FxaaRow(&context, row);
        }
    }
    return Status::Ok();
}

}  // namespace cyber::software
