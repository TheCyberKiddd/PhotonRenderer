#include <cyber/software/ibl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace cyber::software
{

namespace
{

using cyber::core::usize;

f32 Saturate(f32 value) noexcept
{
    if (!math::IsFinite(value))
    {
        return 0.0f;
    }
    return std::clamp(value, 0.0f, 1.0f);
}

bool IsFinite(const math::Vec3& value) noexcept
{
    return math::IsFinite(value.x) && math::IsFinite(value.y) && math::IsFinite(value.z);
}

f32 SmoothStep(f32 value) noexcept
{
    const f32 t = Saturate(value);
    return t * t * (3.0f - 2.0f * t);
}

bool IsValidBox(const math::Vec3& minimum, const math::Vec3& maximum) noexcept
{
    return IsFinite(minimum) && IsFinite(maximum) && maximum.x > minimum.x &&
           maximum.y > minimum.y && maximum.z > minimum.z;
}

usize AlignUp(usize value, usize alignment) noexcept
{
    return (value + alignment - 1) / alignment * alignment;
}

f32 LinearViewDistance(const ScreenSpaceReflectionDesc& desc, f32 reverseZDepth) noexcept
{
    if (desc.perspectiveProjection)
    {
        const f32 denominator = desc.nearPlane + reverseZDepth * (desc.farPlane - desc.nearPlane);
        if (denominator <= 0.0f)
        {
            return desc.farPlane;
        }
        return desc.nearPlane * desc.farPlane / denominator;
    }
    return desc.farPlane - reverseZDepth * (desc.farPlane - desc.nearPlane);
}

}  // namespace

// ---------------------------------------------------------------------------
// Reflection probes
// ---------------------------------------------------------------------------

f32 EvaluateReflectionProbeWeight(const ReflectionProbe& probe,
                                  const math::Vec3& worldPosition) noexcept
{
    if (!IsValidBox(probe.influenceMin, probe.influenceMax) || !IsFinite(worldPosition))
    {
        return 0.0f;
    }
    if (worldPosition.x < probe.influenceMin.x || worldPosition.x > probe.influenceMax.x ||
        worldPosition.y < probe.influenceMin.y || worldPosition.y > probe.influenceMax.y ||
        worldPosition.z < probe.influenceMin.z || worldPosition.z > probe.influenceMax.z)
    {
        return 0.0f;
    }
    const f32 blendDistance = math::IsFinite(probe.blendDistance) && probe.blendDistance > 0.0f
                                  ? probe.blendDistance
                                  : 0.0f;
    if (blendDistance <= 0.0f)
    {
        return 1.0f;
    }

    // Distance to the nearest influence face, so the ramp is continuous across
    // an edge and a corner rather than per-axis.
    f32 distance =
        std::min(worldPosition.x - probe.influenceMin.x, probe.influenceMax.x - worldPosition.x);
    distance = std::min(distance, std::min(worldPosition.y - probe.influenceMin.y,
                                           probe.influenceMax.y - worldPosition.y));
    distance = std::min(distance, std::min(worldPosition.z - probe.influenceMin.z,
                                           probe.influenceMax.z - worldPosition.z));
    return SmoothStep(distance / blendDistance);
}

math::Vec3 ApplyBoxParallaxCorrection(const ReflectionProbe& probe, const math::Vec3& worldPosition,
                                      const math::Vec3& reflectDirection) noexcept
{
    if (!IsValidBox(probe.proxyMin, probe.proxyMax) || !IsFinite(worldPosition) ||
        !IsFinite(reflectDirection))
    {
        return reflectDirection;
    }
    const math::Vec3 direction = cyber::math::Normalize(reflectDirection);
    if (cyber::math::LengthSquared(direction) < 1.0e-12f)
    {
        return reflectDirection;
    }

    // Exit distance of the ray through the proxy box. Starting inside the box
    // the smallest positive slab crossing is the exit, which is the point the
    // environment is reprojected from.
    f32 exitDistance = std::numeric_limits<f32>::infinity();
    for (usize axis = 0; axis < 3; ++axis)
    {
        const f32 component = direction[axis];
        if (std::abs(component) < 1.0e-6f)
        {
            continue;
        }
        const f32 plane = component > 0.0f ? probe.proxyMax[axis] : probe.proxyMin[axis];
        const f32 distance = (plane - worldPosition[axis]) / component;
        if (distance > 0.0f)
        {
            exitDistance = std::min(exitDistance, distance);
        }
    }
    if (!math::IsFinite(exitDistance) || exitDistance <= 0.0f)
    {
        return reflectDirection;
    }

    const math::Vec3 intersection = worldPosition + direction * exitDistance;
    const math::Vec3 corrected = intersection - probe.position;
    if (cyber::math::LengthSquared(corrected) < 1.0e-12f)
    {
        return reflectDirection;
    }
    return cyber::math::Normalize(corrected);
}

math::Vec3 EvaluateReflectionProbes(const ReflectionProbeSet& probes,
                                    const math::Vec3& worldPosition,
                                    const IblSurface& surface) noexcept
{
    math::Vec3 normal = cyber::math::Normalize(surface.normal);
    if (cyber::math::LengthSquared(normal) < 1.0e-12f)
    {
        normal = math::Vec3{0.0f, 1.0f, 0.0f};
    }
    math::Vec3 view = cyber::math::Normalize(surface.viewDirection);
    if (cyber::math::LengthSquared(view) < 1.0e-12f)
    {
        view = normal;
    }
    const math::Vec3 mirror = cyber::math::Reflect(-view, normal);

    const usize count = std::min<usize>(probes.probes.Size(), kMaxReflectionProbes);
    f32 weights[kMaxReflectionProbes]{};
    f32 total = 0.0f;
    for (usize index = 0; index < count; ++index)
    {
        weights[index] = EvaluateReflectionProbeWeight(probes.probes[index], worldPosition);
        total += weights[index];
    }

    f32 remaining = 0.0f;
    if (total > 1.0f)
    {
        const f32 normalization = 1.0f / total;
        for (usize index = 0; index < count; ++index)
        {
            weights[index] *= normalization;
        }
    }
    else
    {
        remaining = 1.0f - total;
    }

    math::Vec3 result{};
    for (usize index = 0; index < count; ++index)
    {
        if (weights[index] <= 0.0f)
        {
            continue;
        }
        const ReflectionProbe& probe = probes.probes[index];
        const math::Vec3 direction = ApplyBoxParallaxCorrection(probe, worldPosition, mirror);
        const math::Vec3 radiance =
            EvaluateIblDiffuse(probe.environment, surface) +
            EvaluateIblSpecularDirection(probe.environment, surface, direction);
        result = result + radiance * weights[index];
    }
    if (remaining > 0.0f && probes.fallback != nullptr)
    {
        result = result + EvaluateImageBasedLighting(*probes.fallback, surface) * remaining;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Hierarchical depth
// ---------------------------------------------------------------------------

HiZPyramid::~HiZPyramid()
{
    Release();
}

HiZPyramid::HiZPyramid(HiZPyramid&& other) noexcept
{
    MoveFrom(other);
}

HiZPyramid& HiZPyramid::operator=(HiZPyramid&& other) noexcept
{
    if (this != &other)
    {
        Release();
        MoveFrom(other);
    }
    return *this;
}

void HiZPyramid::Release() noexcept
{
    if (m_allocator != nullptr && m_data != nullptr)
    {
        m_allocator->Deallocate(m_data);
    }
    m_allocator = nullptr;
    m_data = nullptr;
    m_sizeBytes = 0;
    m_width = 0;
    m_height = 0;
    m_levelCount = 0;
    for (usize index = 0; index < kMaxHiZLevels; ++index)
    {
        m_levelOffsets[index] = 0;
    }
}

void HiZPyramid::MoveFrom(HiZPyramid& other) noexcept
{
    m_allocator = other.m_allocator;
    m_data = other.m_data;
    m_sizeBytes = other.m_sizeBytes;
    m_width = other.m_width;
    m_height = other.m_height;
    m_levelCount = other.m_levelCount;
    for (usize index = 0; index < kMaxHiZLevels; ++index)
    {
        m_levelOffsets[index] = other.m_levelOffsets[index];
    }
    other.m_allocator = nullptr;
    other.m_data = nullptr;
    other.m_sizeBytes = 0;
    other.m_width = 0;
    other.m_height = 0;
    other.m_levelCount = 0;
}

core::Result<HiZPyramid, core::Error> HiZPyramid::Build(const DepthBuffer& depth,
                                                        memory::Allocator& allocator) noexcept
{
    using ResultType = core::Result<HiZPyramid, core::Error>;
    if (!depth.IsValid())
    {
        return ResultType::Err(core::ErrorCode::InvalidArgument, "depth buffer invalid");
    }

    HiZPyramid pyramid;
    pyramid.m_width = depth.Width();
    pyramid.m_height = depth.Height();

    u32 levels = 1;
    while (levels < kMaxHiZLevels && (std::max(1, pyramid.m_width >> (levels - 1)) > 1 ||
                                      std::max(1, pyramid.m_height >> (levels - 1)) > 1))
    {
        ++levels;
    }
    pyramid.m_levelCount = levels;

    usize offset = 0;
    for (u32 level = 0; level < levels; ++level)
    {
        pyramid.m_levelOffsets[level] = offset;
        offset += static_cast<usize>(std::max(1, pyramid.m_width >> level)) *
                  static_cast<usize>(std::max(1, pyramid.m_height >> level));
    }

    const usize sizeBytes = offset * sizeof(f32);
    void* memory = allocator.Allocate(AlignUp(sizeBytes, 16), 16);
    if (memory == nullptr)
    {
        return ResultType::Err(core::ErrorCode::Internal, "hi-z allocation failed");
    }
    pyramid.m_allocator = &allocator;
    pyramid.m_data = static_cast<f32*>(memory);
    pyramid.m_sizeBytes = sizeBytes;

    for (i32 y = 0; y < pyramid.m_height; ++y)
    {
        for (i32 x = 0; x < pyramid.m_width; ++x)
        {
            pyramid.m_data[static_cast<usize>(y) * static_cast<usize>(pyramid.m_width) +
                           static_cast<usize>(x)] = depth.Get(x, y);
        }
    }

    for (u32 level = 1; level < levels; ++level)
    {
        const i32 width = std::max(1, pyramid.m_width >> level);
        const i32 height = std::max(1, pyramid.m_height >> level);
        const i32 parentWidth = std::max(1, pyramid.m_width >> (level - 1));
        const i32 parentHeight = std::max(1, pyramid.m_height >> (level - 1));
        f32* target = pyramid.m_data + pyramid.m_levelOffsets[level];
        const f32* source = pyramid.m_data + pyramid.m_levelOffsets[level - 1];
        for (i32 y = 0; y < height; ++y)
        {
            for (i32 x = 0; x < width; ++x)
            {
                const i32 x0 = std::min(x * 2, parentWidth - 1);
                const i32 y0 = std::min(y * 2, parentHeight - 1);
                const i32 x1 = std::min(x0 + 1, parentWidth - 1);
                const i32 y1 = std::min(y0 + 1, parentHeight - 1);
                // Reverse-Z: the larger value is the closer surface, so the
                // maximum is the conservative near bound of the footprint.
                const f32 closest = std::max(
                    std::max(source[static_cast<usize>(y0) * static_cast<usize>(parentWidth) +
                                    static_cast<usize>(x0)],
                             source[static_cast<usize>(y0) * static_cast<usize>(parentWidth) +
                                    static_cast<usize>(x1)]),
                    std::max(source[static_cast<usize>(y1) * static_cast<usize>(parentWidth) +
                                    static_cast<usize>(x0)],
                             source[static_cast<usize>(y1) * static_cast<usize>(parentWidth) +
                                    static_cast<usize>(x1)]));
                target[static_cast<usize>(y) * static_cast<usize>(width) + static_cast<usize>(x)] =
                    closest;
            }
        }
    }
    return ResultType::Ok(std::move(pyramid));
}

bool HiZPyramid::IsValid() const noexcept
{
    return m_data != nullptr && m_width > 0 && m_height > 0 && m_levelCount > 0;
}

i32 HiZPyramid::Width() const noexcept
{
    return m_width;
}

i32 HiZPyramid::Height() const noexcept
{
    return m_height;
}

u32 HiZPyramid::LevelCount() const noexcept
{
    return m_levelCount;
}

i32 HiZPyramid::LevelWidth(u32 level) const noexcept
{
    if (!IsValid() || level >= m_levelCount)
    {
        return 0;
    }
    return std::max(1, m_width >> level);
}

i32 HiZPyramid::LevelHeight(u32 level) const noexcept
{
    if (!IsValid() || level >= m_levelCount)
    {
        return 0;
    }
    return std::max(1, m_height >> level);
}

f32 HiZPyramid::Depth(u32 level, i32 x, i32 y) const noexcept
{
    if (!IsValid() || level >= m_levelCount)
    {
        return 0.0f;
    }
    const i32 width = std::max(1, m_width >> level);
    const i32 height = std::max(1, m_height >> level);
    if (x < 0 || y < 0 || x >= width || y >= height)
    {
        return 0.0f;
    }
    return m_data[m_levelOffsets[level] + static_cast<usize>(y) * static_cast<usize>(width) +
                  static_cast<usize>(x)];
}

// ---------------------------------------------------------------------------
// Screen-space reflections
// ---------------------------------------------------------------------------

namespace
{

struct ScreenRay
{
    math::Vec2 start{};
    math::Vec2 end{};
    f32 startDepth = 0.0f;
    f32 endDepth = 0.0f;
    bool valid = false;
};

// Projects the segment into pixel space. Both the position and the reverse-Z
// depth are affine along the segment after the perspective divide, so the
// march can interpolate linearly.
ScreenRay ProjectRay(const ScreenSpaceReflectionDesc& desc, const math::Vec3& origin,
                     const math::Vec3& direction, f32 width, f32 height) noexcept
{
    ScreenRay ray{};
    const math::Vec4 startClip =
        desc.clipFromWorld * math::Vec4{origin.x, origin.y, origin.z, 1.0f};
    if (!math::IsFinite(startClip.w) || startClip.w <= 1.0e-6f)
    {
        return ray;
    }

    f32 distance = desc.maxWorldDistance;
    math::Vec4 endClip{};
    bool found = false;
    for (u32 attempt = 0; attempt < 8; ++attempt)
    {
        const math::Vec3 end = origin + direction * distance;
        endClip = desc.clipFromWorld * math::Vec4{end.x, end.y, end.z, 1.0f};
        if (math::IsFinite(endClip.w) && endClip.w > 1.0e-6f)
        {
            found = true;
            break;
        }
        distance *= 0.5f;
    }
    if (!found)
    {
        return ray;
    }

    const f32 startReciprocal = 1.0f / startClip.w;
    const f32 endReciprocal = 1.0f / endClip.w;
    ray.start = math::Vec2{(0.5f * startClip.x * startReciprocal + 0.5f) * width,
                           (0.5f - 0.5f * startClip.y * startReciprocal) * height};
    ray.end = math::Vec2{(0.5f * endClip.x * endReciprocal + 0.5f) * width,
                         (0.5f - 0.5f * endClip.y * endReciprocal) * height};
    ray.startDepth = startClip.z * startReciprocal;
    ray.endDepth = endClip.z * endReciprocal;
    ray.valid = math::IsFinite(ray.start.x) && math::IsFinite(ray.start.y) &&
                math::IsFinite(ray.end.x) && math::IsFinite(ray.end.y) &&
                math::IsFinite(ray.startDepth) && math::IsFinite(ray.endDepth);
    return ray;
}

f32 EdgeFade(const ScreenSpaceReflectionDesc& desc, f32 x, f32 y, f32 width, f32 height) noexcept
{
    const f32 fraction = Saturate(desc.edgeFadeFraction);
    if (fraction <= 0.0f)
    {
        return 1.0f;
    }
    const f32 marginX = fraction * width;
    const f32 marginY = fraction * height;
    const f32 fadeX = marginX > 0.0f ? Saturate(std::min(x, width - x) / marginX) : 1.0f;
    const f32 fadeY = marginY > 0.0f ? Saturate(std::min(y, height - y) / marginY) : 1.0f;
    return SmoothStep(std::min(fadeX, fadeY));
}

}  // namespace

ScreenSpaceReflectionSample TraceScreenSpaceReflection(const ScreenSpaceReflectionDesc& desc,
                                                       const math::Vec3& worldPosition,
                                                       const math::Vec3& reflectDirection) noexcept
{
    ScreenSpaceReflectionSample sample{};
    if (desc.color == nullptr || !desc.color->IsValid() || desc.depth == nullptr ||
        !desc.depth->IsValid())
    {
        return sample;
    }
    if (desc.color->Width() != desc.depth->Width() || desc.color->Height() != desc.depth->Height())
    {
        return sample;
    }
    if (!IsFinite(worldPosition) || !IsFinite(reflectDirection))
    {
        return sample;
    }
    if (!math::IsFinite(desc.nearPlane) || !math::IsFinite(desc.farPlane) ||
        desc.nearPlane <= 0.0f || desc.farPlane <= desc.nearPlane)
    {
        return sample;
    }
    if (!math::IsFinite(desc.maxWorldDistance) || desc.maxWorldDistance <= 0.0f)
    {
        return sample;
    }

    const math::Vec3 direction = cyber::math::Normalize(reflectDirection);
    if (cyber::math::LengthSquared(direction) < 1.0e-12f)
    {
        return sample;
    }

    const i32 pixelWidth = desc.depth->Width();
    const i32 pixelHeight = desc.depth->Height();
    const f32 width = static_cast<f32>(pixelWidth);
    const f32 height = static_cast<f32>(pixelHeight);

    const ScreenRay ray = ProjectRay(desc, worldPosition, direction, width, height);
    if (!ray.valid)
    {
        return sample;
    }

    const math::Vec2 delta{ray.end.x - ray.start.x, ray.end.y - ray.start.y};
    const f32 pixelLength = std::sqrt(delta.x * delta.x + delta.y * delta.y);
    if (pixelLength < 1.0e-3f)
    {
        return sample;
    }

    const u32 maximumSteps = std::clamp(desc.maxSteps, 1u, 1024u);
    const f32 thickness =
        math::IsFinite(desc.thickness) && desc.thickness > 0.0f ? desc.thickness : 0.0f;

    // Hi-Z traversal when a pyramid is supplied, a fixed-step walk otherwise.
    // Both accept the same hits; the pyramid only skips tiles whose closest
    // surface is still farther than the ray, which cannot contain a crossing.
    const bool useHiZ = desc.hiZ != nullptr && desc.hiZ->IsValid() &&
                        desc.hiZ->Width() == pixelWidth && desc.hiZ->Height() == pixelHeight;
    const u32 topLevel = useHiZ ? desc.hiZ->LevelCount() - 1u : 0u;

    const f32 stepLength = 1.0f / static_cast<f32>(maximumSteps);
    f32 previousT = 0.0f;
    f32 t = stepLength;
    u32 level = useHiZ ? std::min(topLevel, 3u) : 0u;
    u32 steps = 0;
    bool hit = false;
    bool wasInFront = false;
    f32 hitT = 0.0f;

    while (t <= 1.0f && steps < maximumSteps * 4u)
    {
        const f32 x = ray.start.x + delta.x * t;
        const f32 y = ray.start.y + delta.y * t;
        if (x < 0.0f || y < 0.0f || x >= width || y >= height)
        {
            break;
        }
        const i32 pixelX = static_cast<i32>(x);
        const i32 pixelY = static_cast<i32>(y);
        const f32 rayDepth = ray.startDepth + (ray.endDepth - ray.startDepth) * t;
        ++steps;

        if (useHiZ && level > 0)
        {
            const i32 tileX = pixelX >> level;
            const i32 tileY = pixelY >> level;
            const i32 tileSize = 1 << level;

            // Distance to the far edge of this tile, which is where the ray is
            // at its deepest inside it.
            f32 advance = std::numeric_limits<f32>::infinity();
            if (std::abs(delta.x) > 1.0e-6f)
            {
                const f32 boundary = delta.x > 0.0f ? static_cast<f32>((tileX + 1) * tileSize)
                                                    : static_cast<f32>(tileX * tileSize) - 1.0e-3f;
                advance = std::min(advance, (boundary - ray.start.x) / delta.x - t);
            }
            if (std::abs(delta.y) > 1.0e-6f)
            {
                const f32 boundary = delta.y > 0.0f ? static_cast<f32>((tileY + 1) * tileSize)
                                                    : static_cast<f32>(tileY * tileSize) - 1.0e-3f;
                advance = std::min(advance, (boundary - ray.start.y) / delta.y - t);
            }
            if (!math::IsFinite(advance) || advance <= 0.0f)
            {
                advance = stepLength;
            }

            // The skip is only sound when the ray stays in front of the tile's
            // closest surface for the whole crossing, so the test uses the
            // ray's depth where it exits rather than where it enters.
            const f32 exitT = std::min(1.0f, t + advance);
            const f32 exitDepth = ray.startDepth + (ray.endDepth - ray.startDepth) * exitT;
            const f32 closest = desc.hiZ->Depth(level, tileX, tileY);
            if (closest < std::min(rayDepth, exitDepth))
            {
                previousT = t;
                t += advance + 1.0e-4f;
                level = std::min(topLevel, level + 1u);
                continue;
            }
            --level;
            continue;
        }

        const f32 sceneDepth = desc.depth->Get(pixelX, pixelY);
        if (sceneDepth > 0.0f)
        {
            if (rayDepth < sceneDepth)
            {
                const f32 rayDistance = LinearViewDistance(desc, rayDepth);
                const f32 sceneDistance = LinearViewDistance(desc, sceneDepth);
                // A hit is a crossing: the ray was in front of the surface and
                // is now behind it by less than the surface's assumed
                // thickness. Without the crossing requirement a ray that
                // starts nearly parallel to its own surface re-finds it.
                if (wasInFront && rayDistance - sceneDistance <= thickness)
                {
                    hit = true;
                    hitT = t;
                    break;
                }
            }
            else
            {
                wasInFront = true;
            }
        }

        previousT = t;
        t += stepLength;
        if (useHiZ)
        {
            level = std::min(topLevel, level + 1u);
        }
    }

    sample.steps = steps;
    if (!hit)
    {
        return sample;
    }

    // Refine the crossing between the last miss and the hit so the fetched
    // pixel is the surface the ray met rather than the step that overshot it.
    f32 low = previousT;
    f32 high = hitT;
    for (u32 iteration = 0; iteration < 8; ++iteration)
    {
        const f32 middle = (low + high) * 0.5f;
        const f32 x = ray.start.x + delta.x * middle;
        const f32 y = ray.start.y + delta.y * middle;
        if (x < 0.0f || y < 0.0f || x >= width || y >= height)
        {
            high = middle;
            continue;
        }
        const f32 rayDepth = ray.startDepth + (ray.endDepth - ray.startDepth) * middle;
        const f32 sceneDepth = desc.depth->Get(static_cast<i32>(x), static_cast<i32>(y));
        if (sceneDepth > 0.0f && rayDepth < sceneDepth)
        {
            high = middle;
        }
        else
        {
            low = middle;
        }
        ++sample.steps;
    }

    const f32 finalT = high;
    const f32 hitX = ray.start.x + delta.x * finalT;
    const f32 hitY = ray.start.y + delta.y * finalT;
    const i32 pixelX = std::clamp(static_cast<i32>(hitX), 0, pixelWidth - 1);
    const i32 pixelY = std::clamp(static_cast<i32>(hitY), 0, pixelHeight - 1);
    const math::Color texel = desc.color->GetPixel(pixelX, pixelY);
    sample.radiance = math::Vec3{texel.r, texel.g, texel.b};
    sample.hit = true;
    sample.confidence =
        EdgeFade(desc, hitX, hitY, width, height) * SmoothStep(1.0f - finalT * finalT);
    return sample;
}

math::Vec3 ResolveReflections(const ScreenSpaceReflectionDesc& desc,
                              const ReflectionProbeSet& probes, const math::Vec3& worldPosition,
                              const IblSurface& surface) noexcept
{
    const math::Vec3 probeRadiance = EvaluateReflectionProbes(probes, worldPosition, surface);

    const f32 maxRoughness = Saturate(desc.maxRoughness);
    const f32 roughness = Saturate(surface.roughness);
    if (maxRoughness <= 0.0f || roughness > maxRoughness)
    {
        return probeRadiance;
    }

    math::Vec3 normal = cyber::math::Normalize(surface.normal);
    if (cyber::math::LengthSquared(normal) < 1.0e-12f)
    {
        normal = math::Vec3{0.0f, 1.0f, 0.0f};
    }
    math::Vec3 view = cyber::math::Normalize(surface.viewDirection);
    if (cyber::math::LengthSquared(view) < 1.0e-12f)
    {
        view = normal;
    }
    const math::Vec3 mirror = cyber::math::Reflect(-view, normal);
    const ScreenSpaceReflectionSample traced =
        TraceScreenSpaceReflection(desc, worldPosition, mirror);
    if (!traced.hit || traced.confidence <= 0.0f)
    {
        return probeRadiance;
    }

    // A rough surface trusts the trace less, because one screen-space ray
    // cannot stand in for a wide lobe.
    const f32 roughnessFade = SmoothStep(1.0f - roughness / maxRoughness);
    const f32 confidence = Saturate(traced.confidence * roughnessFade);
    if (confidence <= 0.0f)
    {
        return probeRadiance;
    }

    const EnvironmentLighting* weighting = probes.fallback;
    if (weighting == nullptr && !probes.probes.Empty())
    {
        weighting = &probes.probes[0].environment;
    }
    EnvironmentLighting weightSource{};
    if (weighting != nullptr)
    {
        weightSource.brdfLut = weighting->brdfLut;
        weightSource.applySpecularOcclusion = weighting->applySpecularOcclusion;
    }
    const math::Vec3 screenSpecular =
        ApplySpecularEnvironmentWeight(weightSource, surface, traced.radiance);
    return probeRadiance * (1.0f - confidence) + screenSpecular * confidence;
}

}  // namespace cyber::software
