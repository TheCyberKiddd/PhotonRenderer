#include <cyber/jobs/job_system.h>
#include <cyber/software/denoise.h>
#include <cyber/software/post_process.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace cyber::software
{

namespace
{

using cyber::core::ErrorCode;
using cyber::core::f64;

// Floats per pixel across every buffer the denoiser keeps; see the member
// list in denoise.h for the breakdown.
constexpr usize kFloatsPerPixel = 3 + 2 + 1 + 1 + 3 + 1 + 3 + 3 + 2 + 1 + 4 + 4;
constexpr i32 kMaxDimension = 16384;

// B3-spline taps, the kernel the a-trous transform is defined with.
constexpr f32 kAtrousKernel[5] = {1.0f / 16.0f, 1.0f / 4.0f, 3.0f / 8.0f, 1.0f / 4.0f,
                                  1.0f / 16.0f};

f32 Luminance(f32 r, f32 g, f32 b) noexcept
{
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

f32 Finite(f32 value) noexcept
{
    return std::isfinite(value) ? value : 0.0f;
}

void RunRows(jobs::JobSystem* jobSystem, bool useJobs, i32 rows, jobs::ParallelFunction function,
             void* userData) noexcept
{
    if (rows <= 0)
    {
        return;
    }
    if (useJobs && jobSystem != nullptr && jobSystem->IsRunning() && rows > 1)
    {
        jobSystem->ParallelFor(static_cast<u32>(rows), 4, function, userData);
        return;
    }
    for (i32 row = 0; row < rows; ++row)
    {
        function(userData, static_cast<u32>(row));
    }
}

struct FrameContext
{
    i32 width = 0;
    i32 height = 0;
    const Framebuffer* noisy = nullptr;
    const DenoiseGuides* guides = nullptr;
    const DenoiserDesc* desc = nullptr;
    bool hasHistory = false;

    const f32* historyColor = nullptr;
    const f32* historyMoments = nullptr;
    const f32* historyLength = nullptr;
    const f32* previousDepth = nullptr;
    const f32* previousNormal = nullptr;

    f32* currentDepth = nullptr;
    f32* currentNormal = nullptr;
    f32* accumColor = nullptr;
    f32* accumMoments = nullptr;
    f32* accumLength = nullptr;
    f32* filterOut = nullptr;

    // A-trous pass state.
    const f32* source = nullptr;
    f32* destination = nullptr;
    i32 step = 1;
};

void DecodeGuidesRow(void* userData, u32 row) noexcept
{
    const FrameContext& context = *static_cast<const FrameContext*>(userData);
    const i32 y = static_cast<i32>(row);
    const DenoiseGuides& guides = *context.guides;
    const f32* depthRow = guides.depth->Row(y);
    for (i32 x = 0; x < context.width; ++x)
    {
        const usize pixel =
            static_cast<usize>(y) * static_cast<usize>(context.width) + static_cast<usize>(x);
        context.currentDepth[pixel] =
            LinearizeReverseZ(depthRow[x], guides.nearPlane, guides.farPlane);
        const math::Color encoded = guides.normals->GetPixel(x, y);
        math::Vec3 normal{encoded.r * 2.0f - 1.0f, encoded.g * 2.0f - 1.0f,
                          encoded.b * 2.0f - 1.0f};
        const f32 length = math::Length(normal);
        normal = length > 1.0e-6f && std::isfinite(length) ? normal * (1.0f / length)
                                                           : math::Vec3{0.0f, 0.0f, 1.0f};
        context.currentNormal[pixel * 3 + 0] = normal.x;
        context.currentNormal[pixel * 3 + 1] = normal.y;
        context.currentNormal[pixel * 3 + 2] = normal.z;
    }
}

math::Vec3 NormalAt(const f32* normals, usize pixel) noexcept
{
    return math::Vec3{normals[pixel * 3 + 0], normals[pixel * 3 + 1], normals[pixel * 3 + 2]};
}

void TemporalRow(void* userData, u32 row) noexcept
{
    const FrameContext& context = *static_cast<const FrameContext*>(userData);
    const DenoiserDesc& desc = *context.desc;
    const i32 y = static_cast<i32>(row);
    const i32 width = context.width;
    const i32 height = context.height;

    for (i32 x = 0; x < width; ++x)
    {
        const usize pixel =
            static_cast<usize>(y) * static_cast<usize>(width) + static_cast<usize>(x);
        const math::Color sample = context.noisy->GetPixel(x, y);
        const f32 r = Finite(sample.r);
        const f32 g = Finite(sample.g);
        const f32 b = Finite(sample.b);
        const f32 luminance = Luminance(r, g, b);

        bool reprojected = false;
        f32 history[3] = {};
        f32 moments[2] = {};
        f32 length = 0.0f;
        if (desc.temporal && context.hasHistory)
        {
            f32 vx = 0.0f;
            f32 vy = 0.0f;
            if (context.guides->velocity != nullptr)
            {
                const math::Color velocity = context.guides->velocity->GetPixel(x, y);
                vx = Finite(velocity.r);
                vy = Finite(velocity.g);
            }
            // Velocity takes a point from the previous frame to this one, so
            // the previous position is this one minus it. Taps are the four
            // texel centres around that point, bilinearly weighted, each
            // tested on its own so an edge keeps the side that matches.
            const f32 px = static_cast<f32>(x) + 0.5f - vx - 0.5f;
            const f32 py = static_cast<f32>(y) + 0.5f - vy - 0.5f;
            const f32 fx = std::floor(px);
            const f32 fy = std::floor(py);
            const f32 tx = px - fx;
            const f32 ty = py - fy;
            const f32 depth = context.currentDepth[pixel];
            const math::Vec3 normal = NormalAt(context.currentNormal, pixel);
            f32 totalWeight = 0.0f;
            for (u32 tap = 0; tap < 4; ++tap)
            {
                const f32 sx = fx + static_cast<f32>(tap & 1u);
                const f32 sy = fy + static_cast<f32>(tap >> 1u);
                if (!(sx >= 0.0f) || !(sy >= 0.0f) || sx >= static_cast<f32>(width) ||
                    sy >= static_cast<f32>(height))
                {
                    continue;
                }
                const usize source =
                    static_cast<usize>(sy) * static_cast<usize>(width) + static_cast<usize>(sx);
                const f32 previousDepth = context.previousDepth[source];
                if (std::fabs(previousDepth - depth) > desc.depthRejectRelative * depth)
                {
                    continue;
                }
                if (math::Dot(NormalAt(context.previousNormal, source), normal) <
                    desc.normalRejectCosine)
                {
                    continue;
                }
                const f32 weight =
                    ((tap & 1u) != 0 ? tx : 1.0f - tx) * ((tap >> 1u) != 0 ? ty : 1.0f - ty);
                if (weight <= 0.0f)
                {
                    continue;
                }
                history[0] += context.historyColor[source * 3 + 0] * weight;
                history[1] += context.historyColor[source * 3 + 1] * weight;
                history[2] += context.historyColor[source * 3 + 2] * weight;
                moments[0] += context.historyMoments[source * 2 + 0] * weight;
                moments[1] += context.historyMoments[source * 2 + 1] * weight;
                length += context.historyLength[source] * weight;
                totalWeight += weight;
            }
            if (totalWeight > 1.0e-4f)
            {
                const f32 inverse = 1.0f / totalWeight;
                for (f32& channel : history)
                {
                    channel *= inverse;
                }
                moments[0] *= inverse;
                moments[1] *= inverse;
                length *= inverse;
                reprojected = length >= 1.0f;
            }
        }

        f32* color = context.accumColor + pixel * 3;
        f32* accumMoments = context.accumMoments + pixel * 2;
        if (reprojected)
        {
            const f32 newLength =
                std::min(std::floor(length + 0.5f) + 1.0f, static_cast<f32>(desc.maxHistoryLength));
            // A plain running mean while history is short, then an exponential
            // average once 1 / length drops below the floor.
            const f32 alpha = std::max(desc.temporalAlpha, 1.0f / newLength);
            color[0] = history[0] + (r - history[0]) * alpha;
            color[1] = history[1] + (g - history[1]) * alpha;
            color[2] = history[2] + (b - history[2]) * alpha;
            accumMoments[0] = moments[0] + (luminance - moments[0]) * alpha;
            accumMoments[1] = moments[1] + (luminance * luminance - moments[1]) * alpha;
            context.accumLength[pixel] = newLength;
        }
        else
        {
            color[0] = r;
            color[1] = g;
            color[2] = b;
            accumMoments[0] = luminance;
            accumMoments[1] = luminance * luminance;
            context.accumLength[pixel] = 1.0f;
        }
    }
}

// Seeds the filter with colour and a variance estimate. Long history uses the
// temporal moments; short history uses a 5x5 bilateral spatial estimate, since
// two or three samples give a variance too noisy to steer anything.
void VarianceRow(void* userData, u32 row) noexcept
{
    const FrameContext& context = *static_cast<const FrameContext*>(userData);
    const DenoiserDesc& desc = *context.desc;
    const i32 y = static_cast<i32>(row);
    const i32 width = context.width;
    const i32 height = context.height;
    for (i32 x = 0; x < width; ++x)
    {
        const usize pixel =
            static_cast<usize>(y) * static_cast<usize>(width) + static_cast<usize>(x);
        const f32* color = context.accumColor + pixel * 3;
        f32 variance = 0.0f;
        if (context.accumLength[pixel] >= static_cast<f32>(desc.spatialVarianceHistory))
        {
            const f32 m1 = context.accumMoments[pixel * 2 + 0];
            const f32 m2 = context.accumMoments[pixel * 2 + 1];
            variance = std::max(0.0f, m2 - m1 * m1);
        }
        else
        {
            const f32 depth = context.currentDepth[pixel];
            const math::Vec3 normal = NormalAt(context.currentNormal, pixel);
            f32 sumWeight = 0.0f;
            f32 sumM1 = 0.0f;
            f32 sumM2 = 0.0f;
            for (i32 dy = -2; dy <= 2; ++dy)
            {
                const i32 sy = y + dy;
                if (sy < 0 || sy >= height)
                {
                    continue;
                }
                for (i32 dx = -2; dx <= 2; ++dx)
                {
                    const i32 sx = x + dx;
                    if (sx < 0 || sx >= width)
                    {
                        continue;
                    }
                    const usize source =
                        static_cast<usize>(sy) * static_cast<usize>(width) + static_cast<usize>(sx);
                    const f32 depthDelta = std::fabs(context.currentDepth[source] - depth);
                    if (depthDelta > desc.depthRejectRelative * depth)
                    {
                        continue;
                    }
                    const f32 facing =
                        std::max(0.0f, math::Dot(NormalAt(context.currentNormal, source), normal));
                    const f32 weight = std::pow(facing, desc.normalPower);
                    sumWeight += weight;
                    sumM1 += context.accumMoments[source * 2 + 0] * weight;
                    sumM2 += context.accumMoments[source * 2 + 1] * weight;
                }
            }
            if (sumWeight > 0.0f)
            {
                const f32 m1 = sumM1 / sumWeight;
                const f32 m2 = sumM2 / sumWeight;
                variance = std::max(0.0f, m2 - m1 * m1);
            }
            // Boost the spatial estimate for very short history, as SVGF does:
            // it under-represents the temporal variance the pixel will show.
            variance *= 4.0f / std::max(context.accumLength[pixel], 1.0f);
        }
        f32* out = context.filterOut + pixel * 4;
        out[0] = color[0];
        out[1] = color[1];
        out[2] = color[2];
        out[3] = variance;
    }
}

f32 PrefilteredVariance(const FrameContext& context, i32 x, i32 y) noexcept
{
    constexpr f32 kGaussian[3] = {0.25f, 0.5f, 0.25f};
    f32 sum = 0.0f;
    f32 weight = 0.0f;
    for (i32 dy = -1; dy <= 1; ++dy)
    {
        const i32 sy = y + dy;
        if (sy < 0 || sy >= context.height)
        {
            continue;
        }
        for (i32 dx = -1; dx <= 1; ++dx)
        {
            const i32 sx = x + dx;
            if (sx < 0 || sx >= context.width)
            {
                continue;
            }
            const f32 k = kGaussian[dx + 1] * kGaussian[dy + 1];
            const usize source =
                static_cast<usize>(sy) * static_cast<usize>(context.width) + static_cast<usize>(sx);
            sum += context.source[source * 4 + 3] * k;
            weight += k;
        }
    }
    return weight > 0.0f ? sum / weight : 0.0f;
}

void AtrousRow(void* userData, u32 row) noexcept
{
    const FrameContext& context = *static_cast<const FrameContext*>(userData);
    const DenoiserDesc& desc = *context.desc;
    const i32 y = static_cast<i32>(row);
    const i32 width = context.width;
    const i32 height = context.height;
    const i32 step = context.step;
    for (i32 x = 0; x < width; ++x)
    {
        const usize pixel =
            static_cast<usize>(y) * static_cast<usize>(width) + static_cast<usize>(x);
        const f32* center = context.source + pixel * 4;
        const f32 centerLuminance = Luminance(center[0], center[1], center[2]);
        const f32 depth = context.currentDepth[pixel];
        const math::Vec3 normal = NormalAt(context.currentNormal, pixel);
        const f32 sigmaLuminance =
            desc.luminanceSigma * std::sqrt(std::max(PrefilteredVariance(context, x, y), 0.0f)) +
            1.0e-4f;
        const f32 depthScale = desc.depthSigma * std::max(depth, 1.0e-4f);

        f32 sumColor[3] = {};
        f32 sumWeight = 0.0f;
        f32 sumVariance = 0.0f;
        for (i32 ky = -2; ky <= 2; ++ky)
        {
            const i32 sy = y + ky * step;
            if (sy < 0 || sy >= height)
            {
                continue;
            }
            for (i32 kx = -2; kx <= 2; ++kx)
            {
                const i32 sx = x + kx * step;
                if (sx < 0 || sx >= width)
                {
                    continue;
                }
                const usize source =
                    static_cast<usize>(sy) * static_cast<usize>(width) + static_cast<usize>(sx);
                const f32* tap = context.source + source * 4;
                const f32 kernel = kAtrousKernel[kx + 2] * kAtrousKernel[ky + 2];
                const f32 pixelDistance =
                    static_cast<f32>(step) * std::sqrt(static_cast<f32>(kx * kx + ky * ky));
                const f32 depthDelta = std::fabs(context.currentDepth[source] - depth);
                const f32 depthWeight = pixelDistance > 0.0f
                                            ? std::exp(-depthDelta / (depthScale * pixelDistance))
                                            : 1.0f;
                const f32 facing =
                    std::max(0.0f, math::Dot(NormalAt(context.currentNormal, source), normal));
                const f32 normalWeight = std::pow(facing, desc.normalPower);
                const f32 luminanceDelta =
                    std::fabs(Luminance(tap[0], tap[1], tap[2]) - centerLuminance);
                const f32 luminanceWeight = std::exp(-luminanceDelta / sigmaLuminance);
                const f32 weight = kernel * depthWeight * normalWeight * luminanceWeight;
                sumColor[0] += tap[0] * weight;
                sumColor[1] += tap[1] * weight;
                sumColor[2] += tap[2] * weight;
                sumVariance += tap[3] * weight * weight;
                sumWeight += weight;
            }
        }
        // The centre tap always has weight, so sumWeight is positive.
        f32* out = context.destination + pixel * 4;
        const f32 inverse = 1.0f / sumWeight;
        out[0] = sumColor[0] * inverse;
        out[1] = sumColor[1] * inverse;
        out[2] = sumColor[2] * inverse;
        out[3] = sumVariance * inverse * inverse;
    }
}

bool SameSize(const Framebuffer& image, i32 width, i32 height) noexcept
{
    return image.IsValid() && image.Width() == width && image.Height() == height;
}

}  // namespace

SpatioTemporalDenoiser::~SpatioTemporalDenoiser()
{
    Release();
}

SpatioTemporalDenoiser::SpatioTemporalDenoiser(SpatioTemporalDenoiser&& other) noexcept
{
    MoveFrom(other);
}

SpatioTemporalDenoiser& SpatioTemporalDenoiser::operator=(SpatioTemporalDenoiser&& other) noexcept
{
    if (this != &other)
    {
        Release();
        MoveFrom(other);
    }
    return *this;
}

void SpatioTemporalDenoiser::Release() noexcept
{
    if (m_allocator != nullptr && m_storage != nullptr)
    {
        m_allocator->Deallocate(m_storage);
    }
    m_allocator = nullptr;
    m_storage = nullptr;
    m_sizeBytes = 0;
    m_width = 0;
    m_height = 0;
    m_hasHistory = false;
    m_historyColor = nullptr;
    m_historyMoments = nullptr;
    m_historyLength = nullptr;
    m_previousDepth = nullptr;
    m_previousNormal = nullptr;
    m_currentDepth = nullptr;
    m_currentNormal = nullptr;
    m_accumColor = nullptr;
    m_accumMoments = nullptr;
    m_accumLength = nullptr;
    m_filterA = nullptr;
    m_filterB = nullptr;
}

void SpatioTemporalDenoiser::MoveFrom(SpatioTemporalDenoiser& other) noexcept
{
    m_allocator = std::exchange(other.m_allocator, nullptr);
    m_storage = std::exchange(other.m_storage, nullptr);
    m_sizeBytes = std::exchange(other.m_sizeBytes, 0);
    m_width = std::exchange(other.m_width, 0);
    m_height = std::exchange(other.m_height, 0);
    m_hasHistory = std::exchange(other.m_hasHistory, false);
    m_historyColor = std::exchange(other.m_historyColor, nullptr);
    m_historyMoments = std::exchange(other.m_historyMoments, nullptr);
    m_historyLength = std::exchange(other.m_historyLength, nullptr);
    m_previousDepth = std::exchange(other.m_previousDepth, nullptr);
    m_previousNormal = std::exchange(other.m_previousNormal, nullptr);
    m_currentDepth = std::exchange(other.m_currentDepth, nullptr);
    m_currentNormal = std::exchange(other.m_currentNormal, nullptr);
    m_accumColor = std::exchange(other.m_accumColor, nullptr);
    m_accumMoments = std::exchange(other.m_accumMoments, nullptr);
    m_accumLength = std::exchange(other.m_accumLength, nullptr);
    m_filterA = std::exchange(other.m_filterA, nullptr);
    m_filterB = std::exchange(other.m_filterB, nullptr);
}

core::Result<SpatioTemporalDenoiser, core::Error>
SpatioTemporalDenoiser::Create(i32 width, i32 height, memory::Allocator& allocator) noexcept
{
    using ResultType = core::Result<SpatioTemporalDenoiser, core::Error>;
    if (width < 1 || height < 1 || width > kMaxDimension || height > kMaxDimension)
    {
        return ResultType::Err(ErrorCode::InvalidArgument, "denoiser size out of range");
    }
    const usize pixels = static_cast<usize>(width) * static_cast<usize>(height);
    const usize floats = pixels * kFloatsPerPixel;
    void* memory = allocator.Allocate(floats * sizeof(f32), 16);
    if (memory == nullptr)
    {
        return ResultType::Err(ErrorCode::Internal, "denoiser allocation failed");
    }
    SpatioTemporalDenoiser denoiser;
    denoiser.m_allocator = &allocator;
    denoiser.m_storage = static_cast<f32*>(memory);
    denoiser.m_sizeBytes = floats * sizeof(f32);
    denoiser.m_width = width;
    denoiser.m_height = height;
    for (usize index = 0; index < floats; ++index)
    {
        denoiser.m_storage[index] = 0.0f;
    }
    f32* cursor = denoiser.m_storage;
    const auto take = [&cursor, pixels](usize channels) noexcept
    {
        f32* view = cursor;
        cursor += pixels * channels;
        return view;
    };
    denoiser.m_historyColor = take(3);
    denoiser.m_historyMoments = take(2);
    denoiser.m_historyLength = take(1);
    denoiser.m_previousDepth = take(1);
    denoiser.m_previousNormal = take(3);
    denoiser.m_currentDepth = take(1);
    denoiser.m_currentNormal = take(3);
    denoiser.m_accumColor = take(3);
    denoiser.m_accumMoments = take(2);
    denoiser.m_accumLength = take(1);
    denoiser.m_filterA = take(4);
    denoiser.m_filterB = take(4);
    return ResultType::Ok(std::move(denoiser));
}

bool SpatioTemporalDenoiser::IsValid() const noexcept
{
    return m_storage != nullptr && m_width > 0 && m_height > 0;
}

i32 SpatioTemporalDenoiser::Width() const noexcept
{
    return m_width;
}

i32 SpatioTemporalDenoiser::Height() const noexcept
{
    return m_height;
}

usize SpatioTemporalDenoiser::DataSizeBytes() const noexcept
{
    return m_sizeBytes;
}

void SpatioTemporalDenoiser::Reset() noexcept
{
    m_hasHistory = false;
}

core::Status<core::Error> SpatioTemporalDenoiser::Denoise(const Framebuffer& noisy,
                                                          const DenoiseGuides& guides,
                                                          const DenoiserDesc& desc,
                                                          Framebuffer& output, DenoiserStats* stats,
                                                          jobs::JobSystem* jobSystem) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (!IsValid())
    {
        return StatusType::Err(ErrorCode::InvalidArgument, "denoiser is not initialized");
    }
    if (!SameSize(noisy, m_width, m_height) || !SameSize(output, m_width, m_height) ||
        &noisy == &output)
    {
        return StatusType::Err(ErrorCode::InvalidArgument,
                               "denoiser input and output must be distinct images of its size");
    }
    if (guides.depth == nullptr || !guides.depth->IsValid() || guides.depth->Width() != m_width ||
        guides.depth->Height() != m_height || guides.normals == nullptr ||
        !SameSize(*guides.normals, m_width, m_height) ||
        (guides.velocity != nullptr && !SameSize(*guides.velocity, m_width, m_height)))
    {
        return StatusType::Err(ErrorCode::InvalidArgument,
                               "denoiser guide buffer missing or mismatched");
    }
    if (!(guides.nearPlane > 0.0f) || !(guides.farPlane > guides.nearPlane) ||
        !std::isfinite(guides.farPlane))
    {
        return StatusType::Err(ErrorCode::InvalidArgument, "denoiser depth range is invalid");
    }
    if (!(desc.temporalAlpha > 0.0f) || desc.temporalAlpha > 1.0f || desc.maxHistoryLength < 1 ||
        desc.maxHistoryLength > 255 || desc.iterations > kMaxDenoiserIterations ||
        !(desc.depthRejectRelative > 0.0f) || !std::isfinite(desc.normalRejectCosine) ||
        !(desc.depthSigma > 0.0f) || !(desc.normalPower >= 0.0f) || !(desc.luminanceSigma > 0.0f) ||
        !std::isfinite(desc.luminanceSigma) || !std::isfinite(desc.depthSigma) ||
        !std::isfinite(desc.normalPower))
    {
        return StatusType::Err(ErrorCode::InvalidArgument, "denoiser descriptor out of range");
    }

    FrameContext context{};
    context.width = m_width;
    context.height = m_height;
    context.noisy = &noisy;
    context.guides = &guides;
    context.desc = &desc;
    context.hasHistory = m_hasHistory;
    context.historyColor = m_historyColor;
    context.historyMoments = m_historyMoments;
    context.historyLength = m_historyLength;
    context.previousDepth = m_previousDepth;
    context.previousNormal = m_previousNormal;
    context.currentDepth = m_currentDepth;
    context.currentNormal = m_currentNormal;
    context.accumColor = m_accumColor;
    context.accumMoments = m_accumMoments;
    context.accumLength = m_accumLength;
    context.filterOut = m_filterA;

    RunRows(jobSystem, desc.useJobs, m_height, &DecodeGuidesRow, &context);
    RunRows(jobSystem, desc.useJobs, m_height, &TemporalRow, &context);
    RunRows(jobSystem, desc.useJobs, m_height, &VarianceRow, &context);

    const usize pixels = static_cast<usize>(m_width) * static_cast<usize>(m_height);
    // History colour is the first wavelet pass's output when one runs, which
    // is what SVGF feeds back: lightly filtered history converges faster and
    // carries less noise into the next frame than the raw accumulation.
    bool historyColorWritten = false;
    const f32* result = m_filterA;
    u32 iterations = 0;
    if (desc.spatial)
    {
        f32* source = m_filterA;
        f32* destination = m_filterB;
        for (u32 pass = 0; pass < desc.iterations; ++pass)
        {
            context.source = source;
            context.destination = destination;
            context.step = 1 << pass;
            RunRows(jobSystem, desc.useJobs, m_height, &AtrousRow, &context);
            if (pass == 0)
            {
                for (usize pixel = 0; pixel < pixels; ++pixel)
                {
                    m_historyColor[pixel * 3 + 0] = destination[pixel * 4 + 0];
                    m_historyColor[pixel * 3 + 1] = destination[pixel * 4 + 1];
                    m_historyColor[pixel * 3 + 2] = destination[pixel * 4 + 2];
                }
                historyColorWritten = true;
            }
            result = destination;
            std::swap(source, destination);
            ++iterations;
        }
    }

    for (i32 y = 0; y < m_height; ++y)
    {
        for (i32 x = 0; x < m_width; ++x)
        {
            const usize pixel =
                static_cast<usize>(y) * static_cast<usize>(m_width) + static_cast<usize>(x);
            const f32* value = result + pixel * 4;
            output.SetPixel(x, y, math::Color{value[0], value[1], value[2], 1.0f});
        }
    }

    DenoiserStats totals{};
    totals.pixelCount = static_cast<u32>(pixels);
    totals.iterations = iterations;
    f64 lengthSum = 0.0;
    for (usize pixel = 0; pixel < pixels; ++pixel)
    {
        const f32 length = m_accumLength[pixel];
        lengthSum += length;
        if (desc.temporal && m_hasHistory && length <= 1.0f)
        {
            ++totals.historyRejectedPixels;
        }
        if (length < static_cast<f32>(desc.spatialVarianceHistory))
        {
            ++totals.spatialVariancePixels;
        }
        if (!historyColorWritten)
        {
            m_historyColor[pixel * 3 + 0] = m_accumColor[pixel * 3 + 0];
            m_historyColor[pixel * 3 + 1] = m_accumColor[pixel * 3 + 1];
            m_historyColor[pixel * 3 + 2] = m_accumColor[pixel * 3 + 2];
        }
        m_historyMoments[pixel * 2 + 0] = m_accumMoments[pixel * 2 + 0];
        m_historyMoments[pixel * 2 + 1] = m_accumMoments[pixel * 2 + 1];
        m_historyLength[pixel] = length;
        m_previousDepth[pixel] = m_currentDepth[pixel];
        m_previousNormal[pixel * 3 + 0] = m_currentNormal[pixel * 3 + 0];
        m_previousNormal[pixel * 3 + 1] = m_currentNormal[pixel * 3 + 1];
        m_previousNormal[pixel * 3 + 2] = m_currentNormal[pixel * 3 + 2];
    }
    totals.meanHistoryLength = static_cast<f32>(lengthSum / static_cast<f64>(pixels));
    m_hasHistory = desc.temporal;
    if (stats != nullptr)
    {
        *stats = totals;
    }
    return StatusType::Ok();
}

}  // namespace cyber::software
