#include <cyber/math/vec3.h>
#include <cyber/software/shadow_quality.h>

#include <cmath>

namespace cyber::software
{

using cyber::core::f32;
using cyber::core::u32;

namespace
{

bool IsFinite(const math::Vec3& value) noexcept
{
    return math::IsFinite(value.x) && math::IsFinite(value.y) && math::IsFinite(value.z);
}

bool IsFinite(const ShadowProbe& probe) noexcept
{
    return IsFinite(probe.worldPosition) && IsFinite(probe.worldNormal) &&
           IsFinite(probe.directionToLight) && math::IsFinite(probe.viewDepth);
}

bool IsValidThreshold(f32 threshold) noexcept
{
    return math::IsFinite(threshold) && threshold > 0.0f && threshold <= 1.0f;
}

core::Status<core::Error> Invalid(const char* message) noexcept
{
    return core::Status<core::Error>::Err(core::Error{core::ErrorCode::InvalidArgument, message});
}

f32 SampleProbe(const DirectionalShadowCascadeSet& cascades, const ShadowProbe& probe) noexcept
{
    return EvaluateDirectionalShadowVisibility(cascades, probe.viewDepth, probe.worldPosition,
                                               probe.worldNormal, probe.directionToLight);
}

}  // namespace

core::Status<core::Error> MeasureShadowAcne(const DirectionalShadowCascadeSet& cascades,
                                            cyber::memory::Span<const ShadowProbe> litProbes,
                                            f32 visibilityThreshold,
                                            ShadowAcneMetrics& output) noexcept
{
    output = ShadowAcneMetrics{};
    if (litProbes.Empty() || litProbes.Data() == nullptr)
    {
        return Invalid("shadow acne measurement needs at least one probe");
    }
    if (!IsValidThreshold(visibilityThreshold))
    {
        return Invalid("visibility threshold must be finite and within (0, 1]");
    }
    for (cyber::core::usize index = 0; index < litProbes.Size(); ++index)
    {
        if (!IsFinite(litProbes[index]))
        {
            return Invalid("shadow acne probe is not finite");
        }
    }

    f32 minVisibility = 1.0f;
    // Visibility is in [0, 1] and probe counts are bounded by the caller's
    // span, so a double accumulator keeps the mean independent of order.
    cyber::core::f64 total = 0.0;
    u32 darkened = 0;
    u32 run = 0;
    u32 longestRun = 0;

    for (cyber::core::usize index = 0; index < litProbes.Size(); ++index)
    {
        const f32 visibility = SampleProbe(cascades, litProbes[index]);
        total += static_cast<cyber::core::f64>(visibility);
        if (visibility < minVisibility)
        {
            minVisibility = visibility;
        }
        if (visibility < visibilityThreshold)
        {
            ++darkened;
            ++run;
            if (run > longestRun)
            {
                longestRun = run;
            }
        }
        else
        {
            run = 0;
        }
    }

    const auto sampleCount = static_cast<u32>(litProbes.Size());
    output.sampleCount = sampleCount;
    output.darkenedSamples = darkened;
    output.darkenedFraction = static_cast<f32>(darkened) / static_cast<f32>(sampleCount);
    output.minVisibility = minVisibility;
    output.meanVisibility = static_cast<f32>(total / static_cast<cyber::core::f64>(sampleCount));
    output.longestDarkenedRun = longestRun;
    return core::Status<core::Error>::Ok();
}

core::Status<core::Error> MeasureShadowContact(const DirectionalShadowCascadeSet& cascades,
                                               const ShadowProbe& contactProbe,
                                               const math::Vec3& direction, f32 stepDistance,
                                               u32 sampleCount, f32 litThreshold,
                                               ShadowContactMetrics& output) noexcept
{
    output = ShadowContactMetrics{};
    if (sampleCount == 0)
    {
        return Invalid("shadow contact measurement needs at least one sample");
    }
    if (!math::IsFinite(stepDistance) || !(stepDistance > 0.0f))
    {
        return Invalid("shadow contact step distance must be finite and positive");
    }
    if (!IsValidThreshold(litThreshold))
    {
        return Invalid("lit threshold must be finite and within (0, 1]");
    }
    if (!IsFinite(contactProbe) || !IsFinite(direction))
    {
        return Invalid("shadow contact probe or direction is not finite");
    }
    const f32 directionLength = math::Length(direction);
    if (!math::IsFinite(directionLength) || !(directionLength > 0.0f))
    {
        return Invalid("shadow contact direction is degenerate");
    }
    const math::Vec3 step = direction * (stepDistance / directionLength);

    output.sampleCount = sampleCount;
    output.contactVisibility = SampleProbe(cascades, contactProbe);
    output.firstLitIndex = sampleCount;
    output.detachmentDistance = stepDistance * static_cast<f32>(sampleCount);

    for (u32 index = 0; index < sampleCount; ++index)
    {
        ShadowProbe probe = contactProbe;
        probe.worldPosition = contactProbe.worldPosition + step * static_cast<f32>(index);
        if (SampleProbe(cascades, probe) >= litThreshold)
        {
            output.foundLitSample = true;
            output.firstLitIndex = index;
            output.detachmentDistance = stepDistance * static_cast<f32>(index);
            return core::Status<core::Error>::Ok();
        }
    }
    return core::Status<core::Error>::Ok();
}

core::Status<core::Error> MeasureShadowTransect(const DirectionalShadowCascadeSet& cascades,
                                                cyber::memory::Span<const ShadowProbe> transect,
                                                ShadowTransectMetrics& output) noexcept
{
    output = ShadowTransectMetrics{};
    if (transect.Size() < 2 || transect.Data() == nullptr)
    {
        return Invalid("shadow transect needs at least two probes");
    }
    for (cyber::core::usize index = 0; index < transect.Size(); ++index)
    {
        if (!IsFinite(transect[index]))
        {
            return Invalid("shadow transect probe is not finite");
        }
        if (index != 0 && transect[index].viewDepth < transect[index - 1].viewDepth)
        {
            return Invalid("shadow transect probes must be ordered by view depth");
        }
    }

    const auto SplitStraddled = [&cascades](f32 nearDepth, f32 farDepth) noexcept
    {
        for (u32 cascade = 0; cascade + 1 < cascades.count; ++cascade)
        {
            const f32 split = cascades.cascades[cascade].splitFar;
            if (nearDepth < split && farDepth >= split)
            {
                return true;
            }
        }
        return false;
    };

    cyber::core::f64 total = 0.0;
    f32 previous = SampleProbe(cascades, transect[0]);
    for (cyber::core::usize index = 1; index < transect.Size(); ++index)
    {
        const f32 current = SampleProbe(cascades, transect[index]);
        const f32 step = std::fabs(current - previous);
        total += static_cast<cyber::core::f64>(step);
        if (step > output.maxStep)
        {
            output.maxStep = step;
            output.maxStepIndex = static_cast<u32>(index);
        }
        if (SplitStraddled(transect[index - 1].viewDepth, transect[index].viewDepth))
        {
            ++output.splitCrossings;
            if (step > output.maxStepAcrossSplit)
            {
                output.maxStepAcrossSplit = step;
            }
        }
        previous = current;
    }

    const auto pairCount = static_cast<u32>(transect.Size() - 1);
    output.sampleCount = static_cast<u32>(transect.Size());
    output.meanStep = static_cast<f32>(total / static_cast<cyber::core::f64>(pairCount));
    return core::Status<core::Error>::Ok();
}

core::Status<core::Error> MeasureShadowStability(cyber::memory::Span<const f32> visibilitySequence,
                                                 u32 probeCount, u32 frameCount,
                                                 ShadowStabilityMetrics& output) noexcept
{
    output = ShadowStabilityMetrics{};
    if (probeCount == 0 || frameCount == 0)
    {
        return Invalid("shadow stability measurement needs a positive probe and frame count");
    }
    const auto expected =
        static_cast<cyber::core::u64>(probeCount) * static_cast<cyber::core::u64>(frameCount);
    if (static_cast<cyber::core::u64>(visibilitySequence.Size()) != expected ||
        visibilitySequence.Data() == nullptr)
    {
        return Invalid("shadow stability sequence size must equal probeCount * frameCount");
    }
    for (cyber::core::usize index = 0; index < visibilitySequence.Size(); ++index)
    {
        if (!math::IsFinite(visibilitySequence[index]))
        {
            return Invalid("shadow stability sample is not finite");
        }
    }

    output.probeCount = probeCount;
    output.frameCount = frameCount;
    if (frameCount < 2)
    {
        // One frame cannot show temporal change; every delta is zero.
        return core::Status<core::Error>::Ok();
    }

    cyber::core::f64 total = 0.0;
    for (u32 probe = 0; probe < probeCount; ++probe)
    {
        cyber::core::f64 variation = 0.0;
        for (u32 frame = 1; frame < frameCount; ++frame)
        {
            const cyber::core::usize current =
                static_cast<cyber::core::usize>(frame) * probeCount + probe;
            const cyber::core::usize previous =
                static_cast<cyber::core::usize>(frame - 1) * probeCount + probe;
            const f32 delta = std::fabs(visibilitySequence[current] - visibilitySequence[previous]);
            total += static_cast<cyber::core::f64>(delta);
            variation += static_cast<cyber::core::f64>(delta);
            if (delta > output.maxFrameDelta)
            {
                output.maxFrameDelta = delta;
                output.maxFrameDeltaProbe = probe;
                output.maxFrameDeltaFrame = frame;
            }
        }
        const auto variationValue = static_cast<f32>(variation);
        if (variationValue > output.maxTotalVariation)
        {
            output.maxTotalVariation = variationValue;
        }
    }

    const auto deltaCount =
        static_cast<cyber::core::f64>(probeCount) * static_cast<cyber::core::f64>(frameCount - 1);
    output.meanFrameDelta = static_cast<f32>(total / deltaCount);
    return core::Status<core::Error>::Ok();
}

}  // namespace cyber::software
