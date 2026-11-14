#include <cyber/software/gi_quality.h>

#include <algorithm>
#include <cmath>

namespace cyber::software
{

namespace
{

using cyber::core::ErrorCode;
using cyber::core::f64;
using cyber::core::i32;
using cyber::core::usize;

f32 Luminance(const math::Vec3& value) noexcept
{
    return 0.2126f * value.x + 0.7152f * value.y + 0.0722f * value.z;
}

f32 Luminance(const math::Color& value) noexcept
{
    return 0.2126f * value.r + 0.7152f * value.g + 0.0722f * value.b;
}

bool IsFinite(const math::Vec3& value) noexcept
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

f64 LaplacianEnergy(const Framebuffer& image) noexcept
{
    const i32 width = image.Width();
    const i32 height = image.Height();
    f64 sum = 0.0;
    u32 count = 0;
    for (i32 y = 1; y + 1 < height; ++y)
    {
        for (i32 x = 1; x + 1 < width; ++x)
        {
            const f32 center = Luminance(image.GetPixel(x, y));
            const f32 neighbours =
                Luminance(image.GetPixel(x - 1, y)) + Luminance(image.GetPixel(x + 1, y)) +
                Luminance(image.GetPixel(x, y - 1)) + Luminance(image.GetPixel(x, y + 1));
            const f32 laplacian = neighbours - 4.0f * center;
            if (std::isfinite(laplacian))
            {
                sum += std::fabs(laplacian);
                ++count;
            }
        }
    }
    return count == 0 ? 0.0 : sum / static_cast<f64>(count);
}

}  // namespace

core::Status<core::Error> MeasureIrradianceError(memory::Span<const math::Vec3> estimate,
                                                 memory::Span<const math::Vec3> reference,
                                                 f32 luminanceFloor, f32 tolerance,
                                                 IrradianceErrorMetrics& out) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (estimate.Empty() || estimate.Size() != reference.Size())
    {
        return StatusType::Err(ErrorCode::InvalidArgument,
                               "irradiance spans are empty or mismatched");
    }
    if (!(luminanceFloor > 0.0f) || !(tolerance > 0.0f) || !std::isfinite(luminanceFloor) ||
        !std::isfinite(tolerance))
    {
        return StatusType::Err(ErrorCode::InvalidArgument, "irradiance floor or tolerance invalid");
    }
    IrradianceErrorMetrics metrics{};
    f64 relativeSum = 0.0;
    f64 biasSum = 0.0;
    f64 estimateEnergy = 0.0;
    f64 referenceEnergy = 0.0;
    for (usize index = 0; index < estimate.Size(); ++index)
    {
        if (!IsFinite(estimate[index]) || !IsFinite(reference[index]))
        {
            return StatusType::Err(ErrorCode::InvalidArgument, "irradiance sample is not finite");
        }
        const f32 e = Luminance(estimate[index]);
        const f32 r = Luminance(reference[index]);
        const f32 denominator = std::max(r, luminanceFloor);
        const f32 signedError = (e - r) / denominator;
        const f32 error = std::fabs(signedError);
        relativeSum += error;
        biasSum += signedError;
        estimateEnergy += e;
        referenceEnergy += r;
        metrics.maxRelativeError = std::max(metrics.maxRelativeError, error);
        if (error > tolerance)
        {
            ++metrics.failingSamples;
        }
    }
    const f64 count = static_cast<f64>(estimate.Size());
    metrics.sampleCount = static_cast<u32>(estimate.Size());
    metrics.meanRelativeError = static_cast<f32>(relativeSum / count);
    metrics.meanRelativeBias = static_cast<f32>(biasSum / count);
    metrics.energyRatio =
        referenceEnergy > 0.0 ? static_cast<f32>(estimateEnergy / referenceEnergy) : 0.0f;
    metrics.failingFraction = static_cast<f32>(static_cast<f64>(metrics.failingSamples) / count);
    out = metrics;
    return StatusType::Ok();
}

core::Status<core::Error> MeasureLightLeak(memory::Span<const math::Vec3> darkSide,
                                           const math::Vec3& litReference, f32 leakThreshold,
                                           LightLeakMetrics& out) noexcept
{
    using StatusType = core::Status<core::Error>;
    const f32 lit = IsFinite(litReference) ? Luminance(litReference) : 0.0f;
    if (darkSide.Empty() || !(lit > 0.0f) || !(leakThreshold > 0.0f) ||
        !std::isfinite(leakThreshold))
    {
        return StatusType::Err(ErrorCode::InvalidArgument, "light leak inputs are invalid");
    }
    LightLeakMetrics metrics{};
    f64 sum = 0.0;
    for (usize index = 0; index < darkSide.Size(); ++index)
    {
        if (!IsFinite(darkSide[index]))
        {
            return StatusType::Err(ErrorCode::InvalidArgument, "light leak sample is not finite");
        }
        const f32 ratio = std::max(0.0f, Luminance(darkSide[index])) / lit;
        sum += ratio;
        metrics.maxLeakRatio = std::max(metrics.maxLeakRatio, ratio);
        if (ratio > leakThreshold)
        {
            ++metrics.leakingSamples;
        }
    }
    const f64 count = static_cast<f64>(darkSide.Size());
    metrics.sampleCount = static_cast<u32>(darkSide.Size());
    metrics.meanLeakRatio = static_cast<f32>(sum / count);
    metrics.leakingFraction = static_cast<f32>(static_cast<f64>(metrics.leakingSamples) / count);
    out = metrics;
    return StatusType::Ok();
}

core::Status<core::Error> MeasureNoise(const Framebuffer& image, const Framebuffer& reference,
                                       NoiseMetrics& out) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (!image.IsValid() || !reference.IsValid() || image.Width() != reference.Width() ||
        image.Height() != reference.Height())
    {
        return StatusType::Err(ErrorCode::InvalidArgument,
                               "noise images are invalid or mismatched");
    }
    const i32 width = image.Width();
    const i32 height = image.Height();
    f64 squared = 0.0;
    f64 signedSum = 0.0;
    f64 referenceSum = 0.0;
    u32 count = 0;
    for (i32 y = 0; y < height; ++y)
    {
        for (i32 x = 0; x < width; ++x)
        {
            const f32 value = Luminance(image.GetPixel(x, y));
            const f32 expected = Luminance(reference.GetPixel(x, y));
            if (!std::isfinite(value) || !std::isfinite(expected))
            {
                continue;
            }
            const f64 difference = static_cast<f64>(value) - static_cast<f64>(expected);
            squared += difference * difference;
            signedSum += difference;
            referenceSum += expected;
            ++count;
        }
    }
    if (count == 0 || !(referenceSum > 0.0))
    {
        return StatusType::Err(ErrorCode::InvalidArgument, "noise reference has no luminance");
    }
    NoiseMetrics metrics{};
    const f64 n = static_cast<f64>(count);
    const f64 meanReference = referenceSum / n;
    metrics.pixelCount = count;
    metrics.rmse = static_cast<f32>(std::sqrt(squared / n));
    metrics.relativeRmse = static_cast<f32>(std::sqrt(squared / n) / meanReference);
    metrics.bias = static_cast<f32>(signedSum / n);
    metrics.relativeBias = static_cast<f32>(signedSum / n / meanReference);
    metrics.referenceMeanLuminance = static_cast<f32>(meanReference);
    metrics.laplacianEnergy = static_cast<f32>(LaplacianEnergy(image));
    out = metrics;
    return StatusType::Ok();
}

core::Status<core::Error> MeasureLaplacianEnergy(const Framebuffer& image, f32& out) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (!image.IsValid() || image.Width() < 3 || image.Height() < 3)
    {
        return StatusType::Err(ErrorCode::InvalidArgument,
                               "laplacian image is invalid or too small");
    }
    out = static_cast<f32>(LaplacianEnergy(image));
    return StatusType::Ok();
}

}  // namespace cyber::software
