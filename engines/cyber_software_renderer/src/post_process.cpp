#include <cyber/software/post_process.h>

#include <cyber/jobs/job_system.h>
#include <cyber/platform/time.h>

#include <algorithm>
#include <cmath>
#include <new>
#include <utility>

namespace cyber::software
{
namespace
{

using core::Error;
using core::ErrorCode;
using math::Color;
using math::Vec3;

using Status = core::Status<Error>;

// Monotonic nanoseconds. Clock reports seconds; the passes report whole
// nanoseconds so a sub-microsecond pass is still legible in the readout.
[[nodiscard]] u64 NowNanoseconds() noexcept
{
    return static_cast<u64>(platform::Clock::NowSeconds() * 1.0e9);
}

[[nodiscard]] Status InvalidArgument(const char* message) noexcept
{
    return Status::Err(Error{ErrorCode::InvalidArgument, message});
}

[[nodiscard]] f32 Saturate(f32 value) noexcept
{
    if (!std::isfinite(value))
    {
        return 0.0f;
    }
    return std::clamp(value, 0.0f, 1.0f);
}

[[nodiscard]] f32 SafeFinite(f32 value, f32 fallback = 0.0f) noexcept
{
    return std::isfinite(value) ? value : fallback;
}

[[nodiscard]] Color Lerp(const Color& from, const Color& to, f32 t) noexcept
{
    return Color{from.r + (to.r - from.r) * t, from.g + (to.g - from.g) * t,
                 from.b + (to.b - from.b) * t, from.a + (to.a - from.a) * t};
}

// Bilinear read with edge clamping, in pixel coordinates where (0, 0) is the
// center of the first pixel.
[[nodiscard]] Color SampleBilinear(const Framebuffer& image, f32 x, f32 y) noexcept
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
    return Lerp(Lerp(c00, c10, fx), Lerp(c01, c11, fx), fy);
}

// A row-parallel pass over a destination image. Every pass in this file uses
// the same shape, so the job-parallel and single-threaded paths cannot drift.
template <typename Context>
void DispatchRows(i32 rowCount, bool useJobs, jobs::JobSystem* jobSystem,
                  jobs::ParallelFunction function, Context& context) noexcept
{
    if (rowCount <= 0)
    {
        return;
    }
    const u32 rows = static_cast<u32>(rowCount);
    if (useJobs && jobSystem != nullptr && rows > 1)
    {
        jobSystem->ParallelFor(rows, 1, function, &context);
        return;
    }
    for (u32 row = 0; row < rows; ++row)
    {
        function(&context, row);
    }
}

// ---------------------------------------------------------------------------
// Tone map operators
// ---------------------------------------------------------------------------

[[nodiscard]] f32 ReinhardChannel(f32 value, f32 whitePoint) noexcept
{
    const f32 white = whitePoint > 0.0f ? whitePoint : 1.0f;
    const f32 whiteSquared = white * white;
    return (value * (1.0f + value / whiteSquared)) / (1.0f + value);
}

// Hable's filmic curve, normalized so the white point maps to 1.
[[nodiscard]] f32 FilmicPartial(f32 value) noexcept
{
    constexpr f32 kShoulderStrength = 0.22f;
    constexpr f32 kLinearStrength = 0.30f;
    constexpr f32 kLinearAngle = 0.10f;
    constexpr f32 kToeStrength = 0.20f;
    constexpr f32 kToeNumerator = 0.01f;
    constexpr f32 kToeDenominator = 0.30f;

    return ((value * (kShoulderStrength * value + kLinearStrength * kLinearAngle) +
             kToeStrength * kToeNumerator) /
            (value * (kShoulderStrength * value + kLinearStrength) +
             kToeStrength * kToeDenominator)) -
           kToeNumerator / kToeDenominator;
}

[[nodiscard]] f32 FilmicChannel(f32 value) noexcept
{
    constexpr f32 kLinearWhite = 11.2f;
    static const f32 kWhiteScale = 1.0f / FilmicPartial(kLinearWhite);
    return FilmicPartial(value) * kWhiteScale;
}

// Narkowicz's fit of the ACES RRT and ODT, the same curve pbr.h exposes.
[[nodiscard]] f32 AcesChannel(f32 value) noexcept
{
    constexpr f32 kA = 2.51f;
    constexpr f32 kB = 0.03f;
    constexpr f32 kC = 2.43f;
    constexpr f32 kD = 0.59f;
    constexpr f32 kE = 0.14f;
    return (value * (kA * value + kB)) / (value * (kC * value + kD) + kE);
}

// ---------------------------------------------------------------------------
// Low-discrepancy sample patterns
// ---------------------------------------------------------------------------

[[nodiscard]] f32 RadicalInverseBase2(u32 index) noexcept
{
    u32 bits = index;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<f32>(static_cast<double>(bits) * 2.3283064365386963e-10);
}

// Golden-angle spiral. Successive points are maximally spread, which is what
// keeps a low tap count from showing structure in a gather.
void SpiralSample(i32 index, i32 count, f32& outX, f32& outY) noexcept
{
    constexpr f32 kGoldenAngle = 2.39996323f;
    const f32 fractionalIndex = (static_cast<f32>(index) + 0.5f) / static_cast<f32>(count);
    const f32 radius = std::sqrt(fractionalIndex);
    const f32 angle = static_cast<f32>(index) * kGoldenAngle;
    outX = radius * std::cos(angle);
    outY = radius * std::sin(angle);
}

// ---------------------------------------------------------------------------
// Pass contexts
// ---------------------------------------------------------------------------

struct OcclusionContext
{
    const DepthBuffer* depth = nullptr;
    const Framebuffer* normals = nullptr;
    Framebuffer* occlusion = nullptr;
    ViewGeometry geometry{};
    math::Mat4 view = math::Mat4::Identity();
    f32 radius = 0.5f;
    f32 bias = 0.02f;
    f32 intensity = 1.0f;
    f32 power = 1.0f;
    i32 sampleCount = 16;
    i32 width = 0;
    i32 height = 0;
};

void OcclusionRow(void* userData, u32 index) noexcept
{
    auto& context = *static_cast<OcclusionContext*>(userData);
    const i32 y = static_cast<i32>(index);
    const f32 farPlane = context.geometry.farPlane;

    for (i32 x = 0; x < context.width; ++x)
    {
        const f32 depth = context.depth->Get(x, y);
        const f32 linearDepth = LinearizeReverseZ(depth, context.geometry.nearPlane, farPlane);
        // Background texels have nothing in front of them to occlude.
        if (depth <= 0.0f || linearDepth >= farPlane * 0.999f)
        {
            context.occlusion->SetPixel(x, y, Color{1.0f, 1.0f, 1.0f, 1.0f});
            continue;
        }

        const Vec3 origin =
            ReconstructViewPosition(x, y, context.width, context.height, linearDepth,
                                    context.geometry);
        const Color encodedNormal = context.normals->GetPixel(x, y);
        Vec3 worldNormal{encodedNormal.r * 2.0f - 1.0f, encodedNormal.g * 2.0f - 1.0f,
                         encodedNormal.b * 2.0f - 1.0f};
        Vec3 normal = context.view.TransformVector(worldNormal);
        const f32 normalLength = math::Length(normal);
        if (!(normalLength > 1.0e-4f))
        {
            context.occlusion->SetPixel(x, y, Color{1.0f, 1.0f, 1.0f, 1.0f});
            continue;
        }
        normal = normal * (1.0f / normalLength);

        // An orthonormal basis around the normal. The tangent is built from a
        // per-pixel rotation so neighboring pixels do not share a pattern.
        const Vec3 axis = std::abs(normal.z) < 0.9f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f};
        Vec3 tangent = Cross(axis, normal);
        const f32 tangentLength = math::Length(tangent);
        if (!(tangentLength > 1.0e-4f))
        {
            context.occlusion->SetPixel(x, y, Color{1.0f, 1.0f, 1.0f, 1.0f});
            continue;
        }
        tangent = tangent * (1.0f / tangentLength);
        const Vec3 bitangent = Cross(normal, tangent);

        // Interleaved rotation: a fixed 4x4 pattern of angles, which is
        // deterministic and still breaks up the banding a shared pattern shows.
        const i32 patternIndex = (y & 3) * 4 + (x & 3);
        const f32 rotation = static_cast<f32>(patternIndex) * 0.39269908f;
        const f32 cosRotation = std::cos(rotation);
        const f32 sinRotation = std::sin(rotation);

        f32 occlusion = 0.0f;
        for (i32 sample = 0; sample < context.sampleCount; ++sample)
        {
            f32 spiralX = 0.0f;
            f32 spiralY = 0.0f;
            SpiralSample(sample, context.sampleCount, spiralX, spiralY);
            const f32 rotatedX = spiralX * cosRotation - spiralY * sinRotation;
            const f32 rotatedY = spiralX * sinRotation + spiralY * cosRotation;
            // Cosine-weighted hemisphere point: the disc sample lifted onto
            // the hemisphere, then pulled toward the origin so near samples
            // outnumber far ones.
            const f32 planarSquared = std::min(rotatedX * rotatedX + rotatedY * rotatedY, 1.0f);
            const f32 height = std::sqrt(std::max(0.0f, 1.0f - planarSquared));
            const f32 scale = 0.25f + 0.75f * RadicalInverseBase2(static_cast<u32>(sample) + 1u);

            const Vec3 offset = (tangent * rotatedX + bitangent * rotatedY + normal * height) *
                                (context.radius * scale);
            const Vec3 samplePosition = origin + offset;
            if (!(samplePosition.z > 1.0e-4f))
            {
                continue;
            }

            // Project back to the screen to read the depth actually stored
            // there.
            const f32 ndcX = samplePosition.x / (samplePosition.z * context.geometry.tanHalfFovX);
            const f32 ndcY = samplePosition.y / (samplePosition.z * context.geometry.tanHalfFovY);
            const f32 sampleX = (ndcX * 0.5f + 0.5f) * static_cast<f32>(context.width) - 0.5f;
            const f32 sampleY = (0.5f - ndcY * 0.5f) * static_cast<f32>(context.height) - 0.5f;
            const i32 texelX = static_cast<i32>(std::lround(sampleX));
            const i32 texelY = static_cast<i32>(std::lround(sampleY));
            if (texelX < 0 || texelX >= context.width || texelY < 0 || texelY >= context.height)
            {
                continue;
            }

            const f32 sampleDepth = context.depth->Get(texelX, texelY);
            if (sampleDepth <= 0.0f)
            {
                continue;
            }
            const f32 sceneDepth =
                LinearizeReverseZ(sampleDepth, context.geometry.nearPlane, farPlane);
            if (sceneDepth >= samplePosition.z - context.bias)
            {
                continue;
            }
            // Range check: an occluder far closer than the radius belongs to
            // another surface and must not darken this one.
            const f32 rangeFalloff =
                std::clamp(context.radius / std::max(1.0e-4f, origin.z - sceneDepth), 0.0f, 1.0f);
            occlusion += rangeFalloff;
        }

        const f32 normalized = occlusion / static_cast<f32>(context.sampleCount);
        f32 visibility = std::clamp(1.0f - normalized * context.intensity, 0.0f, 1.0f);
        if (context.power != 1.0f)
        {
            visibility = std::pow(visibility, context.power);
        }
        context.occlusion->SetPixel(x, y, Color{visibility, visibility, visibility, 1.0f});
    }
}

struct OcclusionBlurContext
{
    const Framebuffer* source = nullptr;
    Framebuffer* destination = nullptr;
    const DepthBuffer* depth = nullptr;
    i32 radius = 2;
    i32 width = 0;
    i32 height = 0;
    f32 nearPlane = 0.1f;
    f32 farPlane = 1000.0f;
    f32 depthTolerance = 0.05f;
};

void OcclusionBlurRow(void* userData, u32 index) noexcept
{
    auto& context = *static_cast<OcclusionBlurContext*>(userData);
    const i32 y = static_cast<i32>(index);
    for (i32 x = 0; x < context.width; ++x)
    {
        const f32 centerDepth =
            LinearizeReverseZ(context.depth->Get(x, y), context.nearPlane, context.farPlane);
        f32 total = 0.0f;
        f32 weight = 0.0f;
        for (i32 offsetY = -context.radius; offsetY <= context.radius; ++offsetY)
        {
            const i32 sampleY = std::clamp(y + offsetY, 0, context.height - 1);
            for (i32 offsetX = -context.radius; offsetX <= context.radius; ++offsetX)
            {
                const i32 sampleX = std::clamp(x + offsetX, 0, context.width - 1);
                const f32 sampleDepth = LinearizeReverseZ(context.depth->Get(sampleX, sampleY),
                                                          context.nearPlane, context.farPlane);
                // Depth-aware: a neighbor on another surface would blur the
                // occlusion across the silhouette.
                const f32 relative = std::abs(sampleDepth - centerDepth) /
                                     std::max(1.0e-4f, std::abs(centerDepth));
                if (relative > context.depthTolerance)
                {
                    continue;
                }
                total += context.source->GetPixel(sampleX, sampleY).r;
                weight += 1.0f;
            }
        }
        const f32 visibility = weight > 0.0f ? total / weight : context.source->GetPixel(x, y).r;
        context.destination->SetPixel(x, y, Color{visibility, visibility, visibility, 1.0f});
    }
}

struct ModulateContext
{
    const Framebuffer* source = nullptr;
    const Framebuffer* occlusion = nullptr;
    Framebuffer* destination = nullptr;
    i32 width = 0;
};

void ModulateRow(void* userData, u32 index) noexcept
{
    auto& context = *static_cast<ModulateContext*>(userData);
    const i32 y = static_cast<i32>(index);
    for (i32 x = 0; x < context.width; ++x)
    {
        const Color color = context.source->GetPixel(x, y);
        const f32 visibility = context.occlusion->GetPixel(x, y).r;
        context.destination->SetPixel(
            x, y, Color{color.r * visibility, color.g * visibility, color.b * visibility, color.a});
    }
}

struct DepthOfFieldContext
{
    const Framebuffer* source = nullptr;
    const DepthBuffer* depth = nullptr;
    Framebuffer* destination = nullptr;
    ViewGeometry geometry{};
    f32 focusDistance = 5.0f;
    f32 focalLengthMm = 50.0f;
    f32 apertureFStop = 2.8f;
    f32 maxCocPixels = 12.0f;
    f32 pixelsPerMm = 1.0f;
    i32 sampleCount = 24;
    i32 width = 0;
    i32 height = 0;
};

// Thin-lens circle of confusion, in pixels. The sign is dropped: a near and a
// far blur of the same magnitude gather the same way here, which is the
// simplification a single-layer gather makes.
[[nodiscard]] f32 CircleOfConfusionPixels(f32 viewDepth, const DepthOfFieldContext& context) noexcept
{
    const f32 focalLength = std::max(1.0e-3f, context.focalLengthMm);
    const f32 aperture = focalLength / std::max(1.0e-3f, context.apertureFStop);
    const f32 focus = std::max(focalLength * 1.001f, context.focusDistance * 1000.0f);
    const f32 distance = std::max(focalLength * 1.001f, viewDepth * 1000.0f);
    const f32 cocMm = std::abs(aperture * focalLength * (distance - focus) /
                               (distance * (focus - focalLength)));
    return std::min(cocMm * context.pixelsPerMm, context.maxCocPixels);
}

void DepthOfFieldRow(void* userData, u32 index) noexcept
{
    auto& context = *static_cast<DepthOfFieldContext*>(userData);
    const i32 y = static_cast<i32>(index);
    for (i32 x = 0; x < context.width; ++x)
    {
        const f32 centerDepth = LinearizeReverseZ(context.depth->Get(x, y),
                                                  context.geometry.nearPlane,
                                                  context.geometry.farPlane);
        const f32 centerCoc = CircleOfConfusionPixels(centerDepth, context);
        if (centerCoc < 0.5f)
        {
            context.destination->SetPixel(x, y, context.source->GetPixel(x, y));
            continue;
        }

        Color accumulated = context.source->GetPixel(x, y);
        f32 weight = 1.0f;
        for (i32 sample = 0; sample < context.sampleCount; ++sample)
        {
            f32 spiralX = 0.0f;
            f32 spiralY = 0.0f;
            SpiralSample(sample, context.sampleCount, spiralX, spiralY);
            const f32 sampleX = static_cast<f32>(x) + spiralX * centerCoc;
            const f32 sampleY = static_cast<f32>(y) + spiralY * centerCoc;
            const i32 texelX = std::clamp(static_cast<i32>(std::lround(sampleX)), 0,
                                          context.width - 1);
            const i32 texelY = std::clamp(static_cast<i32>(std::lround(sampleY)), 0,
                                          context.height - 1);

            const f32 sampleDepth = LinearizeReverseZ(context.depth->Get(texelX, texelY),
                                                      context.geometry.nearPlane,
                                                      context.geometry.farPlane);
            const f32 sampleCoc = CircleOfConfusionPixels(sampleDepth, context);
            // A sharp foreground sample must not bleed onto a blurred
            // background, so a tap only contributes when its own circle of
            // confusion reaches this pixel.
            const f32 distance = std::sqrt((sampleX - static_cast<f32>(x)) *
                                               (sampleX - static_cast<f32>(x)) +
                                           (sampleY - static_cast<f32>(y)) *
                                               (sampleY - static_cast<f32>(y)));
            const f32 reach = std::max(sampleCoc, sampleDepth > centerDepth ? 0.0f : centerCoc);
            if (distance > reach)
            {
                continue;
            }
            const Color tap = context.source->GetPixel(texelX, texelY);
            accumulated = Color{accumulated.r + tap.r, accumulated.g + tap.g,
                                accumulated.b + tap.b, accumulated.a + tap.a};
            weight += 1.0f;
        }

        const f32 inverse = 1.0f / weight;
        context.destination->SetPixel(x, y,
                                      Color{accumulated.r * inverse, accumulated.g * inverse,
                                            accumulated.b * inverse, accumulated.a * inverse});
    }
}

struct MotionBlurContext
{
    const Framebuffer* source = nullptr;
    const Framebuffer* velocity = nullptr;
    Framebuffer* destination = nullptr;
    f32 shutterScale = 0.5f;
    f32 maxVelocityPixels = 32.0f;
    i32 sampleCount = 8;
    i32 width = 0;
};

void MotionBlurRow(void* userData, u32 index) noexcept
{
    auto& context = *static_cast<MotionBlurContext*>(userData);
    const i32 y = static_cast<i32>(index);
    for (i32 x = 0; x < context.width; ++x)
    {
        const Color velocitySample = context.velocity->GetPixel(x, y);
        f32 velocityX = SafeFinite(velocitySample.r) * context.shutterScale;
        f32 velocityY = SafeFinite(velocitySample.g) * context.shutterScale;
        const f32 length = std::sqrt(velocityX * velocityX + velocityY * velocityY);
        if (length < 0.5f)
        {
            context.destination->SetPixel(x, y, context.source->GetPixel(x, y));
            continue;
        }
        if (length > context.maxVelocityPixels)
        {
            const f32 scale = context.maxVelocityPixels / length;
            velocityX *= scale;
            velocityY *= scale;
        }

        Color accumulated{0.0f, 0.0f, 0.0f, 0.0f};
        for (i32 sample = 0; sample < context.sampleCount; ++sample)
        {
            // Taps are centered on the pixel and spread over the open shutter,
            // so the blur straddles the sample rather than trailing it.
            const f32 t = (static_cast<f32>(sample) + 0.5f) /
                              static_cast<f32>(context.sampleCount) -
                          0.5f;
            const Color tap = SampleBilinear(*context.source, static_cast<f32>(x) + velocityX * t,
                                             static_cast<f32>(y) + velocityY * t);
            accumulated = Color{accumulated.r + tap.r, accumulated.g + tap.g,
                                accumulated.b + tap.b, accumulated.a + tap.a};
        }
        const f32 inverse = 1.0f / static_cast<f32>(context.sampleCount);
        context.destination->SetPixel(x, y,
                                      Color{accumulated.r * inverse, accumulated.g * inverse,
                                            accumulated.b * inverse, accumulated.a * inverse});
    }
}

struct BrightPassContext
{
    const Framebuffer* source = nullptr;
    Framebuffer* destination = nullptr;
    f32 threshold = 1.0f;
    f32 knee = 0.5f;
    i32 width = 0;
    i32 sourceWidth = 0;
    i32 sourceHeight = 0;
};

void BrightPassRow(void* userData, u32 index) noexcept
{
    auto& context = *static_cast<BrightPassContext*>(userData);
    const i32 y = static_cast<i32>(index);
    for (i32 x = 0; x < context.width; ++x)
    {
        // Box-average the 2x2 source footprint while thresholding, which is
        // the downsample and the bright pass in one read.
        Color sum{0.0f, 0.0f, 0.0f, 0.0f};
        for (i32 offsetY = 0; offsetY < 2; ++offsetY)
        {
            for (i32 offsetX = 0; offsetX < 2; ++offsetX)
            {
                const i32 sampleX = std::min(x * 2 + offsetX, context.sourceWidth - 1);
                const i32 sampleY = std::min(y * 2 + offsetY, context.sourceHeight - 1);
                const Color texel = context.source->GetPixel(sampleX, sampleY);
                sum = Color{sum.r + texel.r, sum.g + texel.g, sum.b + texel.b, sum.a + texel.a};
            }
        }
        const Color color{sum.r * 0.25f, sum.g * 0.25f, sum.b * 0.25f, 1.0f};

        const f32 luminance = Luminance(color);
        // Soft knee: a quadratic ramp over the knee width instead of a step,
        // so a highlight fades in as it brightens.
        const f32 knee = std::max(1.0e-4f, context.knee);
        const f32 soft = std::clamp(luminance - context.threshold + knee, 0.0f, 2.0f * knee);
        const f32 contribution =
            std::max(soft * soft / (4.0f * knee), luminance - context.threshold);
        const f32 scale = luminance > 1.0e-6f ? std::max(0.0f, contribution) / luminance : 0.0f;
        context.destination->SetPixel(x, y,
                                      Color{color.r * scale, color.g * scale, color.b * scale, 1.0f});
    }
}

struct DownsampleContext
{
    const Framebuffer* source = nullptr;
    Framebuffer* destination = nullptr;
    i32 width = 0;
    i32 sourceWidth = 0;
    i32 sourceHeight = 0;
};

void DownsampleRow(void* userData, u32 index) noexcept
{
    auto& context = *static_cast<DownsampleContext*>(userData);
    const i32 y = static_cast<i32>(index);
    for (i32 x = 0; x < context.width; ++x)
    {
        Color sum{0.0f, 0.0f, 0.0f, 0.0f};
        for (i32 offsetY = 0; offsetY < 2; ++offsetY)
        {
            for (i32 offsetX = 0; offsetX < 2; ++offsetX)
            {
                const i32 sampleX = std::min(x * 2 + offsetX, context.sourceWidth - 1);
                const i32 sampleY = std::min(y * 2 + offsetY, context.sourceHeight - 1);
                const Color texel = context.source->GetPixel(sampleX, sampleY);
                sum = Color{sum.r + texel.r, sum.g + texel.g, sum.b + texel.b, sum.a + texel.a};
            }
        }
        context.destination->SetPixel(x, y, Color{sum.r * 0.25f, sum.g * 0.25f, sum.b * 0.25f, 1.0f});
    }
}

struct UpsampleContext
{
    const Framebuffer* coarse = nullptr;
    const Framebuffer* fine = nullptr;
    Framebuffer* destination = nullptr;
    f32 scatter = 0.7f;
    i32 width = 0;
};

void UpsampleRow(void* userData, u32 index) noexcept
{
    auto& context = *static_cast<UpsampleContext*>(userData);
    const i32 y = static_cast<i32>(index);
    for (i32 x = 0; x < context.width; ++x)
    {
        // Bilinear magnification of the coarser level, blended over the finer
        // one. Repeating this up the chain is what gives bloom its wide,
        // smooth falloff without one enormous kernel.
        const f32 sampleX = (static_cast<f32>(x) + 0.5f) * 0.5f - 0.5f;
        const f32 sampleY = (static_cast<f32>(y) + 0.5f) * 0.5f - 0.5f;
        const Color coarse = SampleBilinear(*context.coarse, sampleX, sampleY);
        const Color fine = context.fine->GetPixel(x, y);
        context.destination->SetPixel(x, y, Lerp(fine, coarse, context.scatter));
    }
}

struct BloomCompositeContext
{
    const Framebuffer* source = nullptr;
    const Framebuffer* bloom = nullptr;
    Framebuffer* destination = nullptr;
    f32 intensity = 0.06f;
    i32 width = 0;
    i32 height = 0;
    i32 bloomWidth = 0;
    i32 bloomHeight = 0;
};

void BloomCompositeRow(void* userData, u32 index) noexcept
{
    auto& context = *static_cast<BloomCompositeContext*>(userData);
    const i32 y = static_cast<i32>(index);
    const f32 scaleX = static_cast<f32>(context.bloomWidth) / static_cast<f32>(context.width);
    const f32 scaleY = static_cast<f32>(context.bloomHeight) / static_cast<f32>(context.height);
    for (i32 x = 0; x < context.width; ++x)
    {
        const f32 sampleX = (static_cast<f32>(x) + 0.5f) * scaleX - 0.5f;
        const f32 sampleY = (static_cast<f32>(y) + 0.5f) * scaleY - 0.5f;
        const Color bloom = SampleBilinear(*context.bloom, sampleX, sampleY);
        const Color color = context.source->GetPixel(x, y);
        const Color blended = Lerp(color, bloom, context.intensity);
        context.destination->SetPixel(x, y, Color{blended.r, blended.g, blended.b, color.a});
    }
}

struct ResolveContext
{
    const Framebuffer* source = nullptr;
    Framebuffer* destination = nullptr;
    const ColorGradeLut* lut = nullptr;
    ToneMapSettings toneMap{};
    f32 exposureEv = 0.0f;
    bool applyToneMap = true;
    i32 width = 0;
};

void ResolveRow(void* userData, u32 index) noexcept
{
    auto& context = *static_cast<ResolveContext*>(userData);
    const i32 y = static_cast<i32>(index);
    for (i32 x = 0; x < context.width; ++x)
    {
        Color color = context.source->GetPixel(x, y);
        if (context.applyToneMap)
        {
            color = ApplyToneMap(color, context.exposureEv, context.toneMap);
        }
        else if (context.exposureEv != 0.0f)
        {
            const f32 scale = std::exp2(context.exposureEv);
            color = Color{color.r * scale, color.g * scale, color.b * scale, color.a};
        }
        if (context.lut != nullptr)
        {
            color = context.lut->Sample(color);
        }
        context.destination->SetPixel(x, y, color);
    }
}

[[nodiscard]] core::Result<Framebuffer, Error> MakeHdrImage(i32 width, i32 height,
                                                            memory::Allocator& allocator) noexcept
{
    FramebufferDesc desc{};
    desc.width = width;
    desc.height = height;
    desc.format = PixelFormat::RGBA32F;
    return Framebuffer::Create(desc, allocator);
}

class ScopedTimer
{
public:
    explicit ScopedTimer(PassTiming* timing) noexcept
        : m_timing(timing), m_start(NowNanoseconds())
    {
    }

    ~ScopedTimer()
    {
        if (m_timing != nullptr)
        {
            m_timing->nanoseconds = NowNanoseconds() - m_start;
            m_timing->executed = true;
        }
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    PassTiming* m_timing = nullptr;
    u64 m_start = 0;
};

}  // namespace

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

f32 Luminance(const Color& color) noexcept
{
    return 0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b;
}

f32 LinearizeReverseZ(f32 depth, f32 nearPlane, f32 farPlane) noexcept
{
    const f32 near = std::max(1.0e-6f, nearPlane);
    const f32 far = std::max(near * 1.000001f, farPlane);
    if (!std::isfinite(depth))
    {
        return far;
    }
    const f32 clamped = std::clamp(depth, 0.0f, 1.0f);
    const f32 denominator = clamped * (far - near) + near;
    if (denominator <= 0.0f)
    {
        return far;
    }
    return std::clamp((near * far) / denominator, near, far);
}

ViewGeometry MakeViewGeometry(const PostProcessView& view) noexcept
{
    ViewGeometry geometry{};
    geometry.nearPlane = std::max(1.0e-6f, SafeFinite(view.nearPlane, 0.1f));
    geometry.farPlane = std::max(geometry.nearPlane * 1.000001f, SafeFinite(view.farPlane, 1000.0f));

    // A perspective projection stores 1 / (aspect * tan(fovY / 2)) and
    // 1 / tan(fovY / 2) on its diagonal; recovering the tangents keeps the
    // unprojection consistent with whatever camera the frame used.
    const f32 scaleX = view.projection.columns[0].x;
    const f32 scaleY = view.projection.columns[1].y;
    geometry.tanHalfFovX =
        (std::isfinite(scaleX) && std::abs(scaleX) > 1.0e-6f) ? 1.0f / std::abs(scaleX) : 1.0f;
    geometry.tanHalfFovY =
        (std::isfinite(scaleY) && std::abs(scaleY) > 1.0e-6f) ? 1.0f / std::abs(scaleY) : 1.0f;
    return geometry;
}

Vec3 ReconstructViewPosition(i32 x, i32 y, i32 width, i32 height, f32 linearDepth,
                             const ViewGeometry& geometry) noexcept
{
    if (width <= 0 || height <= 0)
    {
        return Vec3{0.0f, 0.0f, std::max(0.0f, linearDepth)};
    }
    const f32 ndcX = (static_cast<f32>(x) + 0.5f) / static_cast<f32>(width) * 2.0f - 1.0f;
    const f32 ndcY = 1.0f - (static_cast<f32>(y) + 0.5f) / static_cast<f32>(height) * 2.0f;
    const f32 depth = std::max(0.0f, SafeFinite(linearDepth));
    return Vec3{ndcX * geometry.tanHalfFovX * depth, ndcY * geometry.tanHalfFovY * depth, depth};
}

// ---------------------------------------------------------------------------
// Exposure
// ---------------------------------------------------------------------------

Status BuildLuminanceHistogram(const Framebuffer& hdrColor, f32 minLogLuminance,
                               f32 maxLogLuminance, LuminanceHistogram& out) noexcept
{
    if (!hdrColor.IsValid())
    {
        return InvalidArgument("histogram source is invalid");
    }
    if (!std::isfinite(minLogLuminance) || !std::isfinite(maxLogLuminance) ||
        maxLogLuminance <= minLogLuminance)
    {
        return InvalidArgument("histogram log range is empty");
    }

    out = LuminanceHistogram{};
    out.minLogLuminance = minLogLuminance;
    out.maxLogLuminance = maxLogLuminance;

    const f32 range = maxLogLuminance - minLogLuminance;
    const f32 scale = static_cast<f32>(kLuminanceHistogramBins) / range;
    for (i32 y = 0; y < hdrColor.Height(); ++y)
    {
        for (i32 x = 0; x < hdrColor.Width(); ++x)
        {
            const f32 luminance = Luminance(hdrColor.GetPixel(x, y));
            i32 bin = 0;
            if (std::isfinite(luminance) && luminance > 0.0f)
            {
                const f32 logLuminance = std::log2(luminance);
                const f32 position = (logLuminance - minLogLuminance) * scale;
                bin = std::clamp(static_cast<i32>(position), 0, kLuminanceHistogramBins - 1);
            }
            out.bins[bin] += 1u;
            out.totalSamples += 1u;
        }
    }
    return Status::Ok();
}

f32 HistogramAverageLuminance(const LuminanceHistogram& histogram, f32 lowPercent,
                              f32 highPercent) noexcept
{
    if (histogram.totalSamples == 0u || !std::isfinite(lowPercent) || !std::isfinite(highPercent))
    {
        return kMiddleGrey;
    }
    const f32 low = std::clamp(lowPercent, 0.0f, 1.0f);
    const f32 high = std::clamp(highPercent, 0.0f, 1.0f);
    if (high <= low)
    {
        return kMiddleGrey;
    }

    const f32 total = static_cast<f32>(histogram.totalSamples);
    const f32 lowCount = low * total;
    const f32 highCount = high * total;
    const f32 range = histogram.maxLogLuminance - histogram.minLogLuminance;
    const f32 binWidth = range / static_cast<f32>(kLuminanceHistogramBins);

    f32 weightedLog = 0.0f;
    f32 weight = 0.0f;
    f32 walked = 0.0f;
    for (i32 bin = 0; bin < kLuminanceHistogramBins; ++bin)
    {
        const f32 count = static_cast<f32>(histogram.bins[bin]);
        if (count <= 0.0f)
        {
            continue;
        }
        // Partial bins at both ends, so a percentile window that falls inside
        // a bin does not jump as the frame changes.
        const f32 binStart = walked;
        const f32 binEnd = walked + count;
        walked = binEnd;
        const f32 covered = std::min(binEnd, highCount) - std::max(binStart, lowCount);
        if (covered <= 0.0f)
        {
            continue;
        }
        const f32 binCenterLog =
            histogram.minLogLuminance + (static_cast<f32>(bin) + 0.5f) * binWidth;
        weightedLog += binCenterLog * covered;
        weight += covered;
    }

    if (weight <= 0.0f)
    {
        return kMiddleGrey;
    }
    const f32 averageLog = weightedLog / weight;
    const f32 average = std::exp2(averageLog);
    return std::isfinite(average) && average > 0.0f ? average : kMiddleGrey;
}

f32 ExposureEvForAverageLuminance(f32 averageLuminance) noexcept
{
    if (!std::isfinite(averageLuminance) || averageLuminance <= 0.0f)
    {
        return 0.0f;
    }
    return -std::log2(averageLuminance / kMiddleGrey);
}

f32 AdaptExposure(ExposureState& state, f32 targetEv, f32 deltaSeconds,
                  const ExposureSettings& settings) noexcept
{
    const f32 target = SafeFinite(targetEv);
    if (!state.adapted || !std::isfinite(deltaSeconds) || deltaSeconds <= 0.0f)
    {
        state.currentEv = target;
        state.adapted = true;
        return state.currentEv;
    }

    const f32 difference = target - state.currentEv;
    const f32 rate = difference > 0.0f ? std::max(0.0f, settings.adaptUpEvPerSecond)
                                       : std::max(0.0f, settings.adaptDownEvPerSecond);
    const f32 step = rate * deltaSeconds;
    if (step >= std::abs(difference))
    {
        state.currentEv = target;
    }
    else
    {
        state.currentEv += std::copysign(step, difference);
    }
    return state.currentEv;
}

core::Result<f32, Error> ComputeExposureEv(const Framebuffer& hdrColor,
                                           const ExposureSettings& settings, ExposureState& state,
                                           f32 deltaSeconds) noexcept
{
    using Computed = core::Result<f32, Error>;

    if (settings.mode == ExposureMode::Manual)
    {
        state.currentEv = SafeFinite(settings.manualEv);
        state.adapted = true;
        return Computed::Ok(state.currentEv);
    }

    if (!hdrColor.IsValid())
    {
        return Computed::Err(Error{ErrorCode::InvalidArgument, "exposure metering source is invalid"});
    }

    LuminanceHistogram histogram{};
    const Status built = BuildLuminanceHistogram(hdrColor, settings.minLogLuminance,
                                                 settings.maxLogLuminance, histogram);
    if (!built)
    {
        return Computed::Err(built.Error());
    }

    const f32 average =
        HistogramAverageLuminance(histogram, settings.lowPercent, settings.highPercent);
    const f32 metered = ExposureEvForAverageLuminance(average);
    const f32 minEv = std::min(settings.minEv, settings.maxEv);
    const f32 maxEv = std::max(settings.minEv, settings.maxEv);
    const f32 clamped = std::clamp(metered, minEv, maxEv);
    const f32 adapted = AdaptExposure(state, clamped, deltaSeconds, settings);
    return Computed::Ok(adapted + SafeFinite(settings.compensationEv));
}

// ---------------------------------------------------------------------------
// Tone mapping and grading
// ---------------------------------------------------------------------------

Color ApplyToneMap(const Color& hdrColor, f32 exposureEv, const ToneMapSettings& settings) noexcept
{
    const f32 scale = std::exp2(SafeFinite(exposureEv));
    const f32 red = std::max(0.0f, SafeFinite(hdrColor.r) * scale);
    const f32 green = std::max(0.0f, SafeFinite(hdrColor.g) * scale);
    const f32 blue = std::max(0.0f, SafeFinite(hdrColor.b) * scale);

    switch (settings.op)
    {
    case ToneMapOperator::Clamp:
        return Color{Saturate(red), Saturate(green), Saturate(blue), hdrColor.a};
    case ToneMapOperator::Reinhard:
        return Color{Saturate(ReinhardChannel(red, settings.whitePoint)),
                     Saturate(ReinhardChannel(green, settings.whitePoint)),
                     Saturate(ReinhardChannel(blue, settings.whitePoint)), hdrColor.a};
    case ToneMapOperator::Filmic:
        return Color{Saturate(FilmicChannel(red)), Saturate(FilmicChannel(green)),
                     Saturate(FilmicChannel(blue)), hdrColor.a};
    case ToneMapOperator::AcesFitted:
        break;
    }
    return Color{Saturate(AcesChannel(red)), Saturate(AcesChannel(green)),
                 Saturate(AcesChannel(blue)), hdrColor.a};
}

Color ApplyColorGrade(const Color& color, const ColorGradeSettings& settings) noexcept
{
    f32 channels[3] = {Saturate(color.r), Saturate(color.g), Saturate(color.b)};

    // White balance first: it is a property of the capture, so it belongs
    // before any creative adjustment.
    const f32 temperature = std::clamp(SafeFinite(settings.temperature), -1.0f, 1.0f);
    const f32 tint = std::clamp(SafeFinite(settings.tint), -1.0f, 1.0f);
    channels[0] *= 1.0f + 0.2f * temperature;
    channels[2] *= 1.0f - 0.2f * temperature;
    channels[1] *= 1.0f - 0.2f * tint;

    const f32 lift[3] = {SafeFinite(settings.lift.r), SafeFinite(settings.lift.g),
                         SafeFinite(settings.lift.b)};
    const f32 gamma[3] = {SafeFinite(settings.gamma.r, 1.0f), SafeFinite(settings.gamma.g, 1.0f),
                          SafeFinite(settings.gamma.b, 1.0f)};
    const f32 gain[3] = {SafeFinite(settings.gain.r, 1.0f), SafeFinite(settings.gain.g, 1.0f),
                         SafeFinite(settings.gain.b, 1.0f)};

    for (i32 channel = 0; channel < 3; ++channel)
    {
        f32 value = Saturate(channels[channel]);
        value = lift[channel] + value * (gain[channel] - lift[channel]);
        value = Saturate(value);
        const f32 exponent = std::max(1.0e-3f, gamma[channel]);
        if (exponent != 1.0f)
        {
            value = std::pow(value, 1.0f / exponent);
        }
        channels[channel] = Saturate(value);
    }

    const f32 contrast = std::max(0.0f, SafeFinite(settings.contrast, 1.0f));
    if (contrast != 1.0f)
    {
        for (f32& value : channels)
        {
            value = Saturate(kMiddleGrey + (value - kMiddleGrey) * contrast);
        }
    }

    const f32 saturation = std::max(0.0f, SafeFinite(settings.saturation, 1.0f));
    if (saturation != 1.0f)
    {
        const f32 luminance =
            Luminance(Color{channels[0], channels[1], channels[2], 1.0f});
        for (f32& value : channels)
        {
            value = Saturate(luminance + (value - luminance) * saturation);
        }
    }

    return Color{channels[0], channels[1], channels[2], color.a};
}

ColorGradeLut::~ColorGradeLut()
{
    Release();
}

ColorGradeLut::ColorGradeLut(ColorGradeLut&& other) noexcept
{
    MoveFrom(other);
}

ColorGradeLut& ColorGradeLut::operator=(ColorGradeLut&& other) noexcept
{
    if (this != &other)
    {
        Release();
        MoveFrom(other);
    }
    return *this;
}

void ColorGradeLut::Release() noexcept
{
    if (m_allocator != nullptr && m_entries != nullptr)
    {
        m_allocator->Deallocate(m_entries);
    }
    m_allocator = nullptr;
    m_entries = nullptr;
    m_entryCount = 0;
    m_size = 0;
}

void ColorGradeLut::MoveFrom(ColorGradeLut& other) noexcept
{
    m_allocator = other.m_allocator;
    m_entries = other.m_entries;
    m_entryCount = other.m_entryCount;
    m_size = other.m_size;
    other.m_allocator = nullptr;
    other.m_entries = nullptr;
    other.m_entryCount = 0;
    other.m_size = 0;
}

core::Result<ColorGradeLut, Error> ColorGradeLut::Bake(const ColorGradeSettings& settings, i32 size,
                                                       memory::Allocator& allocator) noexcept
{
    using Created = core::Result<ColorGradeLut, Error>;

    if (size < kMinColorGradeLutSize || size > kMaxColorGradeLutSize)
    {
        return Created::Err(Error{ErrorCode::InvalidArgument, "lut size out of range"});
    }

    const usize entryCount =
        static_cast<usize>(size) * static_cast<usize>(size) * static_cast<usize>(size);
    void* storage = allocator.Allocate(entryCount * sizeof(Color), alignof(Color));
    if (storage == nullptr)
    {
        return Created::Err(Error{ErrorCode::Internal, "failed to allocate grading lut"});
    }

    ColorGradeLut lut;
    lut.m_allocator = &allocator;
    lut.m_entries = static_cast<Color*>(storage);
    lut.m_entryCount = entryCount;
    lut.m_size = size;

    const f32 step = 1.0f / static_cast<f32>(size - 1);
    usize entry = 0;
    for (i32 blue = 0; blue < size; ++blue)
    {
        for (i32 green = 0; green < size; ++green)
        {
            for (i32 red = 0; red < size; ++red)
            {
                const Color source{static_cast<f32>(red) * step, static_cast<f32>(green) * step,
                                   static_cast<f32>(blue) * step, 1.0f};
                lut.m_entries[entry] = ApplyColorGrade(source, settings);
                ++entry;
            }
        }
    }
    return Created::Ok(std::move(lut));
}

core::Result<ColorGradeLut, Error> ColorGradeLut::Identity(i32 size,
                                                           memory::Allocator& allocator) noexcept
{
    return Bake(ColorGradeSettings{}, size, allocator);
}

core::Result<ColorGradeLut, Error> ColorGradeLut::FromEntries(memory::Span<const Color> entries,
                                                              i32 size,
                                                              memory::Allocator& allocator) noexcept
{
    using Created = core::Result<ColorGradeLut, Error>;

    if (size < kMinColorGradeLutSize || size > kMaxColorGradeLutSize)
    {
        return Created::Err(Error{ErrorCode::InvalidArgument, "lut size out of range"});
    }
    const usize entryCount =
        static_cast<usize>(size) * static_cast<usize>(size) * static_cast<usize>(size);
    if (entries.Size() != entryCount || entries.Data() == nullptr)
    {
        return Created::Err(Error{ErrorCode::InvalidArgument, "lut entry count does not match size"});
    }

    void* storage = allocator.Allocate(entryCount * sizeof(Color), alignof(Color));
    if (storage == nullptr)
    {
        return Created::Err(Error{ErrorCode::Internal, "failed to allocate grading lut"});
    }

    ColorGradeLut lut;
    lut.m_allocator = &allocator;
    lut.m_entries = static_cast<Color*>(storage);
    lut.m_entryCount = entryCount;
    lut.m_size = size;
    for (usize index = 0; index < entryCount; ++index)
    {
        lut.m_entries[index] = entries[index];
    }
    return Created::Ok(std::move(lut));
}

bool ColorGradeLut::IsValid() const noexcept
{
    return m_entries != nullptr && m_size >= kMinColorGradeLutSize;
}

i32 ColorGradeLut::Size() const noexcept
{
    return m_size;
}

memory::Span<const Color> ColorGradeLut::Entries() const noexcept
{
    return memory::Span<const Color>{m_entries, m_entryCount};
}

Color ColorGradeLut::Sample(const Color& color) const noexcept
{
    if (!IsValid())
    {
        return color;
    }

    const f32 maxIndex = static_cast<f32>(m_size - 1);
    const f32 coordinates[3] = {Saturate(color.r) * maxIndex, Saturate(color.g) * maxIndex,
                                Saturate(color.b) * maxIndex};
    i32 base[3]{};
    f32 fraction[3]{};
    for (i32 axis = 0; axis < 3; ++axis)
    {
        const f32 floored = std::floor(coordinates[axis]);
        base[axis] = std::clamp(static_cast<i32>(floored), 0, m_size - 1);
        fraction[axis] = coordinates[axis] - floored;
    }

    f32 accumulated[3] = {0.0f, 0.0f, 0.0f};
    for (i32 corner = 0; corner < 8; ++corner)
    {
        const i32 offsetR = corner & 1;
        const i32 offsetG = (corner >> 1) & 1;
        const i32 offsetB = (corner >> 2) & 1;
        const f32 weight = (offsetR != 0 ? fraction[0] : 1.0f - fraction[0]) *
                           (offsetG != 0 ? fraction[1] : 1.0f - fraction[1]) *
                           (offsetB != 0 ? fraction[2] : 1.0f - fraction[2]);
        if (weight <= 0.0f)
        {
            continue;
        }
        const i32 indexR = std::min(base[0] + offsetR, m_size - 1);
        const i32 indexG = std::min(base[1] + offsetG, m_size - 1);
        const i32 indexB = std::min(base[2] + offsetB, m_size - 1);
        const usize entry = static_cast<usize>((indexB * m_size + indexG) * m_size + indexR);
        const Color& sample = m_entries[entry];
        accumulated[0] += sample.r * weight;
        accumulated[1] += sample.g * weight;
        accumulated[2] += sample.b * weight;
    }
    return Color{accumulated[0], accumulated[1], accumulated[2], color.a};
}

// ---------------------------------------------------------------------------
// The stack
// ---------------------------------------------------------------------------

const char* PostProcessPassName(PostProcessPass pass) noexcept
{
    switch (pass)
    {
    case PostProcessPass::AmbientOcclusion:
        return "ambient_occlusion";
    case PostProcessPass::DepthOfField:
        return "depth_of_field";
    case PostProcessPass::MotionBlur:
        return "motion_blur";
    case PostProcessPass::Bloom:
        return "bloom";
    case PostProcessPass::Exposure:
        return "exposure";
    case PostProcessPass::ToneMap:
        return "tone_map";
    case PostProcessPass::ColorGrade:
        return "color_grade";
    }
    return "unknown";
}

core::Result<PostProcessStack, Error> PostProcessStack::Create(i32 width, i32 height,
                                                               memory::Allocator& allocator) noexcept
{
    using Created = core::Result<PostProcessStack, Error>;

    if (width < 1 || width > 16384 || height < 1 || height > 16384)
    {
        return Created::Err(Error{ErrorCode::InvalidArgument, "post stack dimensions out of range"});
    }

    PostProcessStack stack;
    stack.m_allocator = &allocator;
    stack.m_width = width;
    stack.m_height = height;

    for (Framebuffer& image : stack.m_color)
    {
        auto created = MakeHdrImage(width, height, allocator);
        if (!created)
        {
            return Created::Err(created.Error());
        }
        image = std::move(created).Value();
    }

    auto occlusion = MakeHdrImage(width, height, allocator);
    if (!occlusion)
    {
        return Created::Err(occlusion.Error());
    }
    stack.m_occlusion = std::move(occlusion).Value();

    // The bloom chain starts at half resolution and halves until a level would
    // be smaller than two texels, which is the point where another downsample
    // stops carrying information.
    i32 mipWidth = std::max(1, width / 2);
    i32 mipHeight = std::max(1, height / 2);
    i32 mipCount = 0;
    while (mipCount < kMaxBloomMips && mipWidth >= 2 && mipHeight >= 2)
    {
        auto down = MakeHdrImage(mipWidth, mipHeight, allocator);
        if (!down)
        {
            return Created::Err(down.Error());
        }
        stack.m_bloomDown[mipCount] = std::move(down).Value();

        auto up = MakeHdrImage(mipWidth, mipHeight, allocator);
        if (!up)
        {
            return Created::Err(up.Error());
        }
        stack.m_bloomUp[mipCount] = std::move(up).Value();

        ++mipCount;
        mipWidth = std::max(1, mipWidth / 2);
        mipHeight = std::max(1, mipHeight / 2);
    }
    stack.m_bloomMipCount = mipCount;

    return Created::Ok(std::move(stack));
}

bool PostProcessStack::IsValid() const noexcept
{
    return m_color[0].IsValid() && m_color[1].IsValid() && m_occlusion.IsValid();
}

i32 PostProcessStack::Width() const noexcept
{
    return m_width;
}

i32 PostProcessStack::Height() const noexcept
{
    return m_height;
}

ExposureState& PostProcessStack::Exposure() noexcept
{
    return m_exposure;
}

const ExposureState& PostProcessStack::Exposure() const noexcept
{
    return m_exposure;
}

const Framebuffer& PostProcessStack::AmbientOcclusionBuffer() const noexcept
{
    return m_occlusion;
}

Framebuffer& PostProcessStack::Current() noexcept
{
    return m_color[m_currentIndex];
}

Framebuffer& PostProcessStack::Scratch() noexcept
{
    return m_color[1 - m_currentIndex];
}

void PostProcessStack::Swap() noexcept
{
    m_currentIndex = 1 - m_currentIndex;
}

Status PostProcessStack::RunAmbientOcclusion(const PostProcessInputs& inputs,
                                             const PostProcessSettings& settings,
                                             const PostProcessView& view,
                                             jobs::JobSystem* jobSystem) noexcept
{
    const AmbientOcclusionSettings& ao = settings.ambientOcclusion;

    OcclusionContext context{};
    context.depth = inputs.depth;
    context.normals = inputs.normals;
    context.occlusion = &m_occlusion;
    context.geometry = MakeViewGeometry(view);
    context.view = view.view;
    context.radius = std::max(1.0e-4f, SafeFinite(ao.radius, 0.5f));
    context.bias = std::max(0.0f, SafeFinite(ao.bias));
    context.intensity = std::clamp(SafeFinite(ao.intensity, 1.0f), 0.0f, 4.0f);
    context.power = std::clamp(SafeFinite(ao.power, 1.0f), 0.1f, 8.0f);
    context.sampleCount = std::clamp(ao.sampleCount, 4, 64);
    context.width = m_width;
    context.height = m_height;
    DispatchRows(m_height, settings.useJobs, jobSystem, &OcclusionRow, context);

    const i32 blurRadius = std::clamp(ao.blurRadius, 0, 8);
    if (blurRadius > 0)
    {
        // Blur into the scratch HDR image and swap it back, so the occlusion
        // buffer the caller can inspect is the filtered one.
        OcclusionBlurContext blur{};
        blur.source = &m_occlusion;
        blur.destination = &Scratch();
        blur.depth = inputs.depth;
        blur.radius = blurRadius;
        blur.width = m_width;
        blur.height = m_height;
        blur.nearPlane = context.geometry.nearPlane;
        blur.farPlane = context.geometry.farPlane;
        DispatchRows(m_height, settings.useJobs, jobSystem, &OcclusionBlurRow, blur);

        for (i32 y = 0; y < m_height; ++y)
        {
            for (i32 x = 0; x < m_width; ++x)
            {
                m_occlusion.SetPixel(x, y, Scratch().GetPixel(x, y));
            }
        }
    }

    ModulateContext modulate{};
    modulate.source = &Current();
    modulate.occlusion = &m_occlusion;
    modulate.destination = &Scratch();
    modulate.width = m_width;
    DispatchRows(m_height, settings.useJobs, jobSystem, &ModulateRow, modulate);
    Swap();

    m_occlusionValid = true;
    return Status::Ok();
}

Status PostProcessStack::RunDepthOfField(const PostProcessInputs& inputs,
                                         const PostProcessSettings& settings,
                                         const PostProcessView& view,
                                         jobs::JobSystem* jobSystem) noexcept
{
    const DepthOfFieldSettings& dof = settings.depthOfField;

    DepthOfFieldContext context{};
    context.source = &Current();
    context.depth = inputs.depth;
    context.destination = &Scratch();
    context.geometry = MakeViewGeometry(view);
    context.focusDistance = std::max(1.0e-3f, SafeFinite(dof.focusDistance, 5.0f));
    context.focalLengthMm = std::max(1.0e-3f, SafeFinite(dof.focalLengthMm, 50.0f));
    context.apertureFStop = std::max(0.5f, SafeFinite(dof.apertureFStop, 2.8f));
    context.maxCocPixels = std::clamp(SafeFinite(dof.maxCocPixels, 12.0f), 0.0f, 64.0f);
    // A circle of confusion is a length on the sensor; the sensor height maps
    // onto the frame height, which gives the pixels-per-millimetre factor.
    context.pixelsPerMm =
        static_cast<f32>(m_height) / std::max(1.0e-3f, SafeFinite(view.sensorHeightMm, 24.0f));
    context.sampleCount = std::clamp(dof.sampleCount, 4, 64);
    context.width = m_width;
    context.height = m_height;

    DispatchRows(m_height, settings.useJobs, jobSystem, &DepthOfFieldRow, context);
    Swap();
    return Status::Ok();
}

Status PostProcessStack::RunMotionBlur(const PostProcessInputs& inputs,
                                       const PostProcessSettings& settings,
                                       jobs::JobSystem* jobSystem) noexcept
{
    const MotionBlurSettings& blur = settings.motionBlur;

    MotionBlurContext context{};
    context.source = &Current();
    context.velocity = inputs.velocity;
    context.destination = &Scratch();
    context.shutterScale =
        std::clamp(SafeFinite(blur.shutterAngleDegrees, 180.0f) / 360.0f, 0.0f, 1.0f);
    context.maxVelocityPixels = std::max(0.0f, SafeFinite(blur.maxVelocityPixels, 32.0f));
    context.sampleCount = std::clamp(blur.sampleCount, 2, 32);
    context.width = m_width;

    DispatchRows(m_height, settings.useJobs, jobSystem, &MotionBlurRow, context);
    Swap();
    return Status::Ok();
}

Status PostProcessStack::RunBloom(const PostProcessSettings& settings,
                                  jobs::JobSystem* jobSystem) noexcept
{
    if (m_bloomMipCount <= 0)
    {
        return Status::Ok();
    }

    const BloomSettings& bloom = settings.bloom;
    const i32 mipCount = std::clamp(bloom.mipCount, 1, m_bloomMipCount);

    BrightPassContext bright{};
    bright.source = &Current();
    bright.destination = &m_bloomDown[0];
    bright.threshold = std::max(0.0f, SafeFinite(bloom.threshold, 1.0f));
    bright.knee = std::clamp(SafeFinite(bloom.softKnee, 0.5f), 1.0e-4f, 1.0f);
    bright.width = m_bloomDown[0].Width();
    bright.sourceWidth = m_width;
    bright.sourceHeight = m_height;
    DispatchRows(m_bloomDown[0].Height(), settings.useJobs, jobSystem, &BrightPassRow, bright);

    for (i32 mip = 1; mip < mipCount; ++mip)
    {
        DownsampleContext down{};
        down.source = &m_bloomDown[mip - 1];
        down.destination = &m_bloomDown[mip];
        down.width = m_bloomDown[mip].Width();
        down.sourceWidth = m_bloomDown[mip - 1].Width();
        down.sourceHeight = m_bloomDown[mip - 1].Height();
        DispatchRows(m_bloomDown[mip].Height(), settings.useJobs, jobSystem, &DownsampleRow, down);
    }

    // The coarsest level starts the upsample chain unchanged.
    const i32 coarsest = mipCount - 1;
    for (i32 y = 0; y < m_bloomUp[coarsest].Height(); ++y)
    {
        for (i32 x = 0; x < m_bloomUp[coarsest].Width(); ++x)
        {
            m_bloomUp[coarsest].SetPixel(x, y, m_bloomDown[coarsest].GetPixel(x, y));
        }
    }

    const f32 scatter = std::clamp(SafeFinite(bloom.scatter, 0.7f), 0.0f, 1.0f);
    for (i32 mip = coarsest - 1; mip >= 0; --mip)
    {
        UpsampleContext up{};
        up.coarse = &m_bloomUp[mip + 1];
        up.fine = &m_bloomDown[mip];
        up.destination = &m_bloomUp[mip];
        up.scatter = scatter;
        up.width = m_bloomUp[mip].Width();
        DispatchRows(m_bloomUp[mip].Height(), settings.useJobs, jobSystem, &UpsampleRow, up);
    }

    BloomCompositeContext composite{};
    composite.source = &Current();
    composite.bloom = &m_bloomUp[0];
    composite.destination = &Scratch();
    composite.intensity = std::clamp(SafeFinite(bloom.intensity, 0.06f), 0.0f, 1.0f);
    composite.width = m_width;
    composite.height = m_height;
    composite.bloomWidth = m_bloomUp[0].Width();
    composite.bloomHeight = m_bloomUp[0].Height();
    DispatchRows(m_height, settings.useJobs, jobSystem, &BloomCompositeRow, composite);
    Swap();
    return Status::Ok();
}

Status PostProcessStack::Execute(const PostProcessInputs& inputs, Framebuffer& destination,
                                 const PostProcessSettings& settings, const PostProcessView& view,
                                 f32 deltaSeconds, jobs::JobSystem* jobSystem,
                                 PostProcessTimings* timings) noexcept
{
    if (!IsValid())
    {
        return InvalidArgument("post stack is invalid");
    }
    if (inputs.color == nullptr || !inputs.color->IsValid())
    {
        return InvalidArgument("post input color is invalid");
    }
    if (inputs.color->Width() != m_width || inputs.color->Height() != m_height)
    {
        return InvalidArgument("post input color does not match the stack size");
    }
    if (!destination.IsValid() || destination.Width() != m_width ||
        destination.Height() != m_height)
    {
        return InvalidArgument("post destination does not match the stack size");
    }

    PostProcessTimings local{};
    for (i32 pass = 0; pass < kPostProcessPassCount; ++pass)
    {
        local.passes[pass].name = PostProcessPassName(static_cast<PostProcessPass>(pass));
    }
    const u64 frameStart = NowNanoseconds();

    m_currentIndex = 0;
    m_occlusionValid = false;
    for (i32 y = 0; y < m_height; ++y)
    {
        for (i32 x = 0; x < m_width; ++x)
        {
            Current().SetPixel(x, y, inputs.color->GetPixel(x, y));
        }
    }

    const bool depthReady = inputs.depth != nullptr && inputs.depth->IsValid() &&
                            inputs.depth->Width() == m_width && inputs.depth->Height() == m_height;
    const bool normalsReady = inputs.normals != nullptr && inputs.normals->IsValid() &&
                              inputs.normals->Width() == m_width &&
                              inputs.normals->Height() == m_height;
    const bool velocityReady = inputs.velocity != nullptr && inputs.velocity->IsValid() &&
                               inputs.velocity->Width() == m_width &&
                               inputs.velocity->Height() == m_height;

    if (settings.enableAmbientOcclusion && depthReady && normalsReady)
    {
        ScopedTimer timer(&local.passes[static_cast<i32>(PostProcessPass::AmbientOcclusion)]);
        const Status status = RunAmbientOcclusion(inputs, settings, view, jobSystem);
        if (!status)
        {
            return status;
        }
    }

    if (settings.enableDepthOfField && depthReady)
    {
        ScopedTimer timer(&local.passes[static_cast<i32>(PostProcessPass::DepthOfField)]);
        const Status status = RunDepthOfField(inputs, settings, view, jobSystem);
        if (!status)
        {
            return status;
        }
    }

    if (settings.enableMotionBlur && velocityReady)
    {
        ScopedTimer timer(&local.passes[static_cast<i32>(PostProcessPass::MotionBlur)]);
        const Status status = RunMotionBlur(inputs, settings, jobSystem);
        if (!status)
        {
            return status;
        }
    }

    if (settings.enableBloom)
    {
        ScopedTimer timer(&local.passes[static_cast<i32>(PostProcessPass::Bloom)]);
        const Status status = RunBloom(settings, jobSystem);
        if (!status)
        {
            return status;
        }
    }

    f32 exposureEv = 0.0f;
    f32 averageLuminance = 0.0f;
    if (settings.enableExposure)
    {
        ScopedTimer timer(&local.passes[static_cast<i32>(PostProcessPass::Exposure)]);
        auto computed = ComputeExposureEv(Current(), settings.exposure, m_exposure, deltaSeconds);
        if (!computed)
        {
            return Status::Err(computed.Error());
        }
        exposureEv = computed.Value();
        if (settings.exposure.mode == ExposureMode::Automatic)
        {
            LuminanceHistogram histogram{};
            if (BuildLuminanceHistogram(Current(), settings.exposure.minLogLuminance,
                                        settings.exposure.maxLogLuminance, histogram))
            {
                averageLuminance = HistogramAverageLuminance(
                    histogram, settings.exposure.lowPercent, settings.exposure.highPercent);
            }
        }
    }

    // Tone mapping and grading are one read-modify-write over the image, so
    // they share a resolve pass; each still reports its own time, measured by
    // running the shared pass under whichever of the two is enabled.
    const bool gradeReady = settings.enableColorGrade && settings.colorGradeLut != nullptr &&
                            settings.colorGradeLut->IsValid();

    ResolveContext resolve{};
    resolve.source = &Current();
    resolve.destination = &destination;
    resolve.lut = gradeReady ? settings.colorGradeLut : nullptr;
    resolve.toneMap = settings.toneMap;
    resolve.exposureEv = exposureEv;
    resolve.applyToneMap = settings.enableToneMap;
    resolve.width = m_width;

    {
        PassTiming* toneTiming =
            settings.enableToneMap ? &local.passes[static_cast<i32>(PostProcessPass::ToneMap)]
                                   : nullptr;
        PassTiming* gradeTiming =
            gradeReady ? &local.passes[static_cast<i32>(PostProcessPass::ColorGrade)] : nullptr;
        const u64 resolveStart = NowNanoseconds();
        DispatchRows(m_height, settings.useJobs, jobSystem, &ResolveRow, resolve);
        const u64 elapsed = NowNanoseconds() - resolveStart;

        // Split the shared cost between whichever of the two ran, so the
        // reported total still equals the measured frame.
        const i32 shareCount = (toneTiming != nullptr ? 1 : 0) + (gradeTiming != nullptr ? 1 : 0);
        if (shareCount > 0)
        {
            const u64 share = elapsed / static_cast<u64>(shareCount);
            if (toneTiming != nullptr)
            {
                toneTiming->nanoseconds = share;
                toneTiming->executed = true;
            }
            if (gradeTiming != nullptr)
            {
                gradeTiming->nanoseconds = elapsed - (toneTiming != nullptr ? share : 0u);
                gradeTiming->executed = true;
            }
        }
    }

    local.totalNanoseconds = NowNanoseconds() - frameStart;
    local.exposureEv = exposureEv;
    local.averageLuminance = averageLuminance;
    if (timings != nullptr)
    {
        *timings = local;
    }
    return Status::Ok();
}

}  // namespace cyber::software
