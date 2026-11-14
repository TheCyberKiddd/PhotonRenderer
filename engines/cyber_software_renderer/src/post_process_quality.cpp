#include <cyber/software/post_process.h>
#include <cyber/software/post_process_quality.h>

#include <algorithm>
#include <cmath>

namespace cyber::software
{
namespace
{

using core::Error;
using core::ErrorCode;
using math::Color;

using core::usize;

using Status = core::Status<Error>;

[[nodiscard]] Status InvalidArgument(const char* message) noexcept
{
    return Status::Err(Error{ErrorCode::InvalidArgument, message});
}

[[nodiscard]] bool SameSize(const Framebuffer& lhs, const Framebuffer& rhs) noexcept
{
    return lhs.Width() == rhs.Width() && lhs.Height() == rhs.Height();
}

[[nodiscard]] f32 Finite(f32 value) noexcept
{
    return math::IsFinite(value) ? value : 0.0f;
}

}  // namespace

Status MeasureImageDifference(const Framebuffer& image, const Framebuffer& reference, f32 tolerance,
                              ImageDifferenceMetrics& out) noexcept
{
    if (!image.IsValid() || !reference.IsValid())
    {
        return InvalidArgument("comparison image is invalid");
    }
    if (!SameSize(image, reference))
    {
        return InvalidArgument("comparison images differ in size");
    }
    if (!math::IsFinite(tolerance) || tolerance < 0.0f)
    {
        return InvalidArgument("comparison tolerance must be zero or positive");
    }

    out = ImageDifferenceMetrics{};
    double absoluteSum = 0.0;
    double squaredSum = 0.0;
    f32 worst = 0.0f;
    u32 differing = 0;

    for (i32 y = 0; y < image.Height(); ++y)
    {
        for (i32 x = 0; x < image.Width(); ++x)
        {
            const Color actual = image.GetPixel(x, y);
            const Color expected = reference.GetPixel(x, y);
            const f32 channels[3] = {std::abs(Finite(actual.r) - Finite(expected.r)),
                                     std::abs(Finite(actual.g) - Finite(expected.g)),
                                     std::abs(Finite(actual.b) - Finite(expected.b))};
            f32 pixelWorst = 0.0f;
            for (const f32 difference : channels)
            {
                absoluteSum += static_cast<double>(difference);
                squaredSum += static_cast<double>(difference) * static_cast<double>(difference);
                pixelWorst = std::max(pixelWorst, difference);
            }
            worst = std::max(worst, pixelWorst);
            if (pixelWorst > tolerance)
            {
                ++differing;
            }
            out.pixelCount += 1u;
        }
    }

    if (out.pixelCount == 0u)
    {
        return InvalidArgument("comparison images are empty");
    }

    const double channelCount = static_cast<double>(out.pixelCount) * 3.0;
    out.meanAbsoluteError = static_cast<f32>(absoluteSum / channelCount);
    out.rootMeanSquareError = static_cast<f32>(std::sqrt(squaredSum / channelCount));
    out.maxAbsoluteError = worst;
    out.differingPixels = differing;
    out.differingFraction = static_cast<f32>(differing) / static_cast<f32>(out.pixelCount);
    return Status::Ok();
}

Status MeasureGhosting(const Framebuffer& resolved, const Framebuffer& current,
                       const Framebuffer& previous, f32 changeThreshold, f32 ghostThreshold,
                       GhostingMetrics& out) noexcept
{
    if (!resolved.IsValid() || !current.IsValid() || !previous.IsValid())
    {
        return InvalidArgument("ghosting input image is invalid");
    }
    if (!SameSize(resolved, current) || !SameSize(resolved, previous))
    {
        return InvalidArgument("ghosting input images differ in size");
    }
    if (!math::IsFinite(changeThreshold) || changeThreshold <= 0.0f ||
        !math::IsFinite(ghostThreshold) || ghostThreshold <= 0.0f)
    {
        return InvalidArgument("ghosting thresholds must be positive");
    }

    out = GhostingMetrics{};
    const u32 pixelCount = static_cast<u32>(resolved.Width()) * static_cast<u32>(resolved.Height());
    double contaminationSum = 0.0;
    double projectionSum = 0.0;
    double axisEnergySum = 0.0;
    f32 worst = 0.0f;
    u32 changed = 0;
    u32 ghosted = 0;

    for (i32 y = 0; y < resolved.Height(); ++y)
    {
        for (i32 x = 0; x < resolved.Width(); ++x)
        {
            const Color resolvedColor = resolved.GetPixel(x, y);
            const Color currentColor = current.GetPixel(x, y);
            const Color previousColor = previous.GetPixel(x, y);

            // The axis from the correct answer to the stale one.
            const f32 axis[3] = {Finite(previousColor.r) - Finite(currentColor.r),
                                 Finite(previousColor.g) - Finite(currentColor.g),
                                 Finite(previousColor.b) - Finite(currentColor.b)};
            const f32 axisLengthSquared = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
            const f32 axisLength = std::sqrt(axisLengthSquared);
            if (axisLength <= changeThreshold)
            {
                // The frames agree here, so this pixel says nothing about
                // ghosting either way.
                continue;
            }
            ++changed;

            const f32 offset[3] = {Finite(resolvedColor.r) - Finite(currentColor.r),
                                   Finite(resolvedColor.g) - Finite(currentColor.g),
                                   Finite(resolvedColor.b) - Finite(currentColor.b)};
            // Projection of the resolve onto that axis: 0 landed on the
            // correct answer, 1 reproduced the stale one.
            const f32 dot = offset[0] * axis[0] + offset[1] * axis[1] + offset[2] * axis[2];
            const f32 projection = dot / axisLengthSquared;
            const f32 contamination = std::clamp(projection, 0.0f, 1.0f);

            // The same projection, accumulated as a numerator and denominator
            // instead of a per-pixel ratio, which is the least-squares fit
            // over the whole image.
            projectionSum += static_cast<double>(dot);
            axisEnergySum += static_cast<double>(axisLengthSquared);

            contaminationSum += static_cast<double>(contamination);
            worst = std::max(worst, contamination);
            if (contamination > ghostThreshold)
            {
                ++ghosted;
            }
        }
    }

    if (changed == 0u)
    {
        return Status::Err(
            Error{ErrorCode::NotFound, "no pixel changed between the reference frames"});
    }

    out.changedPixels = changed;
    out.changedFraction =
        pixelCount > 0u ? static_cast<f32>(changed) / static_cast<f32>(pixelCount) : 0.0f;
    out.meanContamination = static_cast<f32>(contaminationSum / static_cast<double>(changed));
    out.maxContamination = worst;
    out.weightedContamination =
        axisEnergySum > 0.0 ? static_cast<f32>(std::clamp(projectionSum / axisEnergySum, 0.0, 1.0))
                            : 0.0f;
    out.ghostedPixels = ghosted;
    out.ghostedFraction = static_cast<f32>(ghosted) / static_cast<f32>(changed);
    return Status::Ok();
}

Status MeasureTemporalStability(memory::Span<const Framebuffer* const> frames,
                                TemporalStabilityMetrics& out) noexcept
{
    if (frames.Size() < 2u || frames.Data() == nullptr)
    {
        return InvalidArgument("stability needs at least two frames");
    }

    const Framebuffer* first = frames[0];
    if (first == nullptr || !first->IsValid())
    {
        return InvalidArgument("stability frame is invalid");
    }
    for (usize index = 1; index < frames.Size(); ++index)
    {
        const Framebuffer* frame = frames[index];
        if (frame == nullptr || !frame->IsValid() || !SameSize(*first, *frame))
        {
            return InvalidArgument("stability frames differ in size");
        }
    }

    out = TemporalStabilityMetrics{};
    out.frameCount = static_cast<u32>(frames.Size());

    double deltaSum = 0.0;
    f32 worstDelta = 0.0f;
    for (usize index = 1; index < frames.Size(); ++index)
    {
        ImageDifferenceMetrics difference{};
        const Status measured =
            MeasureImageDifference(*frames[index], *frames[index - 1], 0.0f, difference);
        if (!measured)
        {
            return measured;
        }
        deltaSum += static_cast<double>(difference.meanAbsoluteError);
        worstDelta = std::max(worstDelta, difference.maxAbsoluteError);
    }

    ImageDifferenceMetrics drift{};
    const Status measuredDrift =
        MeasureImageDifference(*frames[frames.Size() - 1], *first, 0.0f, drift);
    if (!measuredDrift)
    {
        return measuredDrift;
    }

    out.meanFrameDelta = static_cast<f32>(deltaSum / static_cast<double>(frames.Size() - 1));
    out.maxFrameDelta = worstDelta;
    out.totalDrift = drift.maxAbsoluteError;
    return Status::Ok();
}

Status MeasureEdgeAliasing(const Framebuffer& image, f32 stepThreshold,
                           EdgeAliasingMetrics& out) noexcept
{
    if (!image.IsValid())
    {
        return InvalidArgument("edge measurement image is invalid");
    }
    if (!math::IsFinite(stepThreshold) || stepThreshold <= 0.0f)
    {
        return InvalidArgument("edge step threshold must be positive");
    }

    out = EdgeAliasingMetrics{};
    const i32 width = image.Width();
    const i32 height = image.Height();
    double gradientSum = 0.0;
    u32 hardSteps = 0;
    u32 pairCount = 0;

    for (i32 y = 0; y < height; ++y)
    {
        for (i32 x = 0; x < width; ++x)
        {
            const f32 center = Luminance(image.GetPixel(x, y));
            if (x + 1 < width)
            {
                const f32 step = std::abs(Luminance(image.GetPixel(x + 1, y)) - center);
                gradientSum += static_cast<double>(step);
                ++pairCount;
                if (step > stepThreshold)
                {
                    ++hardSteps;
                }
            }
            if (y + 1 < height)
            {
                const f32 step = std::abs(Luminance(image.GetPixel(x, y + 1)) - center);
                gradientSum += static_cast<double>(step);
                ++pairCount;
                if (step > stepThreshold)
                {
                    ++hardSteps;
                }
            }
            out.pixelCount += 1u;
        }
    }

    out.hardStepCount = hardSteps;
    out.hardStepFraction =
        pairCount > 0u ? static_cast<f32>(hardSteps) / static_cast<f32>(pairCount) : 0.0f;
    out.meanGradient =
        pairCount > 0u ? static_cast<f32>(gradientSum / static_cast<double>(pairCount)) : 0.0f;
    return Status::Ok();
}

}  // namespace cyber::software
