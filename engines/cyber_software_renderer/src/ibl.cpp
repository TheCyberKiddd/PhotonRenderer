#include <cyber/jobs/job_system.h>
#include <cyber/software/ibl.h>
#include <cyber/software/image_io.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace cyber::software
{

namespace
{

using cyber::core::usize;

constexpr f32 kPi = cyber::math::kPi;
constexpr f32 kInversePi = 1.0f / kPi;

f32 Saturate(f32 value) noexcept
{
    if (!math::IsFinite(value))
    {
        return 0.0f;
    }
    return std::clamp(value, 0.0f, 1.0f);
}

f32 FiniteOrZero(f32 value) noexcept
{
    return math::IsFinite(value) ? value : 0.0f;
}

bool IsFinite(const math::Vec3& value) noexcept
{
    return math::IsFinite(value.x) && math::IsFinite(value.y) && math::IsFinite(value.z);
}

math::Vec3 FiniteNonNegative(const math::Vec3& value) noexcept
{
    return math::Vec3{math::IsFinite(value.x) && value.x > 0.0f ? value.x : 0.0f,
                      math::IsFinite(value.y) && value.y > 0.0f ? value.y : 0.0f,
                      math::IsFinite(value.z) && value.z > 0.0f ? value.z : 0.0f};
}

math::Vec3 Multiply(const math::Vec3& lhs, const math::Vec3& rhs) noexcept
{
    return math::Vec3{lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
}

math::Vec3 ToVec3(const math::Color& color) noexcept
{
    return math::Vec3{color.r, color.g, color.b};
}

math::Color ToColor(const math::Vec3& value) noexcept
{
    return math::Color{value.x, value.y, value.z, 1.0f};
}

// Van der Corput radical inverse in base 2, which with the index fraction
// forms the Hammersley set. The sequence is fixed, so every integral in this
// file is reproducible.
f32 RadicalInverseBase2(u32 bits) noexcept
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<f32>(bits) * 2.3283064365386963e-10f;
}

math::Vec2 Hammersley(u32 index, u32 count) noexcept
{
    const f32 u = count == 0 ? 0.0f : static_cast<f32>(index) / static_cast<f32>(count);
    return math::Vec2{u, RadicalInverseBase2(index)};
}

// GGX normal distribution with the Disney alpha = roughness^2 remap, which is
// the same remap EvaluateGgxBrdf uses, so a prefiltered level and a direct
// highlight of equal roughness have the same width.
f32 DistributionGgx(f32 ndoth, f32 roughness) noexcept
{
    const f32 alpha = roughness * roughness;
    const f32 alphaSquared = alpha * alpha;
    const f32 cosine = Saturate(ndoth);
    const f32 denominator = cosine * cosine * (alphaSquared - 1.0f) + 1.0f;
    if (denominator <= 0.0f)
    {
        return 0.0f;
    }
    return alphaSquared / (kPi * denominator * denominator);
}

// Importance-samples the GGX half vector around `normal`.
math::Vec3 ImportanceSampleGgx(const math::Vec2& random, const math::Vec3& normal,
                               f32 roughness) noexcept
{
    const f32 alpha = std::max(roughness * roughness, 1.0e-4f);
    const f32 phi = 2.0f * kPi * random.x;
    const f32 cosThetaSquaredDenominator = 1.0f + (alpha * alpha - 1.0f) * random.y;
    const f32 cosTheta = std::sqrt(
        std::max(0.0f, (1.0f - random.y) / std::max(cosThetaSquaredDenominator, 1.0e-8f)));
    const f32 sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));

    const math::Vec3 tangentSpace{sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta};
    const math::Vec3 up =
        std::abs(normal.z) < 0.999f ? math::Vec3{0.0f, 0.0f, 1.0f} : math::Vec3{1.0f, 0.0f, 0.0f};
    const math::Vec3 tangentX = cyber::math::Normalize(cyber::math::Cross(up, normal));
    const math::Vec3 tangentY = cyber::math::Cross(normal, tangentX);
    return cyber::math::Normalize(tangentX * tangentSpace.x + tangentY * tangentSpace.y +
                                  normal * tangentSpace.z);
}

// Smith height-correlated visibility in the IBL parameterization, where k is
// alpha / 2 rather than the direct-lighting (alpha + 1)^2 / 8.
f32 GeometrySmithIbl(f32 ndotv, f32 ndotl, f32 roughness) noexcept
{
    const f32 alpha = roughness * roughness;
    const f32 k = alpha * 0.5f;
    const f32 viewTerm = ndotv / (ndotv * (1.0f - k) + k);
    const f32 lightTerm = ndotl / (ndotl * (1.0f - k) + k);
    return viewTerm * lightTerm;
}

f32 AreaElement(f32 x, f32 y) noexcept
{
    return std::atan2(x * y, std::sqrt(x * x + y * y + 1.0f));
}

usize AlignUp(usize value, usize alignment) noexcept
{
    return (value + alignment - 1) / alignment * alignment;
}

void RunRange(jobs::JobSystem* jobSystem, bool useJobs, u32 count, jobs::ParallelFunction function,
              void* userData) noexcept
{
    if (count == 0)
    {
        return;
    }
    if (useJobs && jobSystem != nullptr && jobSystem->IsRunning() && count > 1)
    {
        jobSystem->ParallelFor(count, 1, function, userData);
        return;
    }
    for (u32 index = 0; index < count; ++index)
    {
        function(userData, index);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Cubemap
// ---------------------------------------------------------------------------

Cubemap::~Cubemap()
{
    Release();
}

Cubemap::Cubemap(Cubemap&& other) noexcept
{
    MoveFrom(other);
}

Cubemap& Cubemap::operator=(Cubemap&& other) noexcept
{
    if (this != &other)
    {
        Release();
        MoveFrom(other);
    }
    return *this;
}

void Cubemap::Release() noexcept
{
    if (m_allocator != nullptr && m_data != nullptr)
    {
        m_allocator->Deallocate(m_data);
    }
    m_allocator = nullptr;
    m_data = nullptr;
    m_sizeBytes = 0;
    m_faceSize = 0;
    m_mipLevels = 0;
    for (usize index = 0; index < kMaxCubemapMipLevels; ++index)
    {
        m_levelOffsets[index] = 0;
    }
}

void Cubemap::MoveFrom(Cubemap& other) noexcept
{
    m_allocator = other.m_allocator;
    m_data = other.m_data;
    m_sizeBytes = other.m_sizeBytes;
    m_faceSize = other.m_faceSize;
    m_mipLevels = other.m_mipLevels;
    for (usize index = 0; index < kMaxCubemapMipLevels; ++index)
    {
        m_levelOffsets[index] = other.m_levelOffsets[index];
    }
    other.m_allocator = nullptr;
    other.m_data = nullptr;
    other.m_sizeBytes = 0;
    other.m_faceSize = 0;
    other.m_mipLevels = 0;
}

core::Result<Cubemap, core::Error> Cubemap::Create(const CubemapDesc& desc,
                                                   memory::Allocator& allocator) noexcept
{
    using ResultType = core::Result<Cubemap, core::Error>;
    if (desc.faceSize <= 0 || desc.faceSize > kMaxCubemapFaceSize)
    {
        return ResultType::Err(core::ErrorCode::InvalidArgument, "cubemap face size out of range");
    }

    u32 fullChain = 1;
    for (i32 size = desc.faceSize; size > 1; size /= 2)
    {
        ++fullChain;
    }
    const u32 levels = desc.mipLevels == 0 ? fullChain : desc.mipLevels;
    if (levels == 0 || levels > fullChain || levels > kMaxCubemapMipLevels)
    {
        return ResultType::Err(core::ErrorCode::InvalidArgument, "cubemap mip count out of range");
    }

    Cubemap cubemap;
    usize offset = 0;
    for (u32 level = 0; level < levels; ++level)
    {
        const i32 size = std::max(1, desc.faceSize >> level);
        cubemap.m_levelOffsets[level] = offset;
        offset += static_cast<usize>(size) * static_cast<usize>(size) * kCubeFaceCount * 4u;
    }

    const usize sizeBytes = offset * sizeof(f32);
    void* memory = allocator.Allocate(AlignUp(sizeBytes, 16), 16);
    if (memory == nullptr)
    {
        return ResultType::Err(core::ErrorCode::Internal, "cubemap allocation failed");
    }

    cubemap.m_allocator = &allocator;
    cubemap.m_data = static_cast<f32*>(memory);
    cubemap.m_sizeBytes = sizeBytes;
    cubemap.m_faceSize = desc.faceSize;
    cubemap.m_mipLevels = levels;
    for (usize index = 0; index < offset; ++index)
    {
        cubemap.m_data[index] = 0.0f;
    }
    return ResultType::Ok(std::move(cubemap));
}

bool Cubemap::IsValid() const noexcept
{
    return m_data != nullptr && m_faceSize > 0 && m_mipLevels > 0;
}

i32 Cubemap::FaceSize() const noexcept
{
    return m_faceSize;
}

u32 Cubemap::MipLevels() const noexcept
{
    return m_mipLevels;
}

i32 Cubemap::MipSize(u32 level) const noexcept
{
    if (!IsValid() || level >= m_mipLevels)
    {
        return 0;
    }
    return std::max(1, m_faceSize >> level);
}

usize Cubemap::DataSizeBytes() const noexcept
{
    return m_sizeBytes;
}

f32* Cubemap::TexelPointer(u32 face, u32 level, i32 x, i32 y) const noexcept
{
    if (!IsValid() || face >= kCubeFaceCount || level >= m_mipLevels)
    {
        return nullptr;
    }
    const i32 size = std::max(1, m_faceSize >> level);
    if (x < 0 || y < 0 || x >= size || y >= size)
    {
        return nullptr;
    }
    const usize faceTexels = static_cast<usize>(size) * static_cast<usize>(size);
    const usize index = m_levelOffsets[level] +
                        (static_cast<usize>(face) * faceTexels +
                         static_cast<usize>(y) * static_cast<usize>(size) + static_cast<usize>(x)) *
                            4u;
    return m_data + index;
}

math::Color Cubemap::GetTexel(u32 face, u32 level, i32 x, i32 y) const noexcept
{
    const f32* texel = TexelPointer(face, level, x, y);
    if (texel == nullptr)
    {
        return math::Color{0.0f, 0.0f, 0.0f, 0.0f};
    }
    return math::Color{texel[0], texel[1], texel[2], texel[3]};
}

void Cubemap::SetTexel(u32 face, u32 level, i32 x, i32 y, const math::Color& color) noexcept
{
    f32* texel = TexelPointer(face, level, x, y);
    if (texel == nullptr)
    {
        return;
    }
    texel[0] = color.r;
    texel[1] = color.g;
    texel[2] = color.b;
    texel[3] = color.a;
}

void Cubemap::Clear(const math::Color& color) noexcept
{
    if (!IsValid())
    {
        return;
    }
    for (u32 level = 0; level < m_mipLevels; ++level)
    {
        const i32 size = std::max(1, m_faceSize >> level);
        for (u32 face = 0; face < kCubeFaceCount; ++face)
        {
            for (i32 y = 0; y < size; ++y)
            {
                for (i32 x = 0; x < size; ++x)
                {
                    SetTexel(face, level, x, y, color);
                }
            }
        }
    }
}

void Cubemap::GenerateMips() noexcept
{
    if (!IsValid())
    {
        return;
    }
    for (u32 level = 1; level < m_mipLevels; ++level)
    {
        const i32 size = std::max(1, m_faceSize >> level);
        const i32 parentSize = std::max(1, m_faceSize >> (level - 1));
        for (u32 face = 0; face < kCubeFaceCount; ++face)
        {
            for (i32 y = 0; y < size; ++y)
            {
                for (i32 x = 0; x < size; ++x)
                {
                    const i32 x0 = std::min(x * 2, parentSize - 1);
                    const i32 y0 = std::min(y * 2, parentSize - 1);
                    const i32 x1 = std::min(x0 + 1, parentSize - 1);
                    const i32 y1 = std::min(y0 + 1, parentSize - 1);
                    const math::Color a = GetTexel(face, level - 1, x0, y0);
                    const math::Color b = GetTexel(face, level - 1, x1, y0);
                    const math::Color c = GetTexel(face, level - 1, x0, y1);
                    const math::Color d = GetTexel(face, level - 1, x1, y1);
                    SetTexel(face, level, x, y,
                             math::Color{
                                 (a.r + b.r + c.r + d.r) * 0.25f, (a.g + b.g + c.g + d.g) * 0.25f,
                                 (a.b + b.b + c.b + d.b) * 0.25f, (a.a + b.a + c.a + d.a) * 0.25f});
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cube addressing
// ---------------------------------------------------------------------------

math::Vec3 CubeDirectionFromFaceUv(u32 face, f32 u, f32 v) noexcept
{
    if (!math::IsFinite(u) || !math::IsFinite(v))
    {
        u = 0.5f;
        v = 0.5f;
    }
    const f32 s = 2.0f * u - 1.0f;
    const f32 t = 1.0f - 2.0f * v;
    math::Vec3 direction{};
    switch (face)
    {
    case 0:
        direction = math::Vec3{1.0f, t, -s};
        break;
    case 1:
        direction = math::Vec3{-1.0f, t, s};
        break;
    case 2:
        direction = math::Vec3{s, 1.0f, -t};
        break;
    case 3:
        direction = math::Vec3{s, -1.0f, t};
        break;
    case 4:
        direction = math::Vec3{s, t, 1.0f};
        break;
    default:
        direction = math::Vec3{-s, t, -1.0f};
        break;
    }
    return cyber::math::Normalize(direction);
}

void CubeFaceUvFromDirection(const math::Vec3& direction, u32& face, f32& u, f32& v) noexcept
{
    face = 0;
    u = 0.5f;
    v = 0.5f;
    if (!IsFinite(direction))
    {
        return;
    }
    const f32 absX = std::abs(direction.x);
    const f32 absY = std::abs(direction.y);
    const f32 absZ = std::abs(direction.z);
    const f32 major = std::max(absX, std::max(absY, absZ));
    if (major <= 0.0f)
    {
        return;
    }

    f32 s = 0.0f;
    f32 t = 0.0f;
    if (major == absX)
    {
        if (direction.x > 0.0f)
        {
            face = 0;
            s = -direction.z / absX;
        }
        else
        {
            face = 1;
            s = direction.z / absX;
        }
        t = direction.y / absX;
    }
    else if (major == absY)
    {
        if (direction.y > 0.0f)
        {
            face = 2;
            t = -direction.z / absY;
        }
        else
        {
            face = 3;
            t = direction.z / absY;
        }
        s = direction.x / absY;
    }
    else
    {
        if (direction.z > 0.0f)
        {
            face = 4;
            s = direction.x / absZ;
        }
        else
        {
            face = 5;
            s = -direction.x / absZ;
        }
        t = direction.y / absZ;
    }

    u = (s + 1.0f) * 0.5f;
    v = (1.0f - t) * 0.5f;
}

f32 CubeTexelSolidAngle(i32 faceSize, f32 u, f32 v) noexcept
{
    if (faceSize <= 0 || !math::IsFinite(u) || !math::IsFinite(v))
    {
        return 0.0f;
    }
    const f32 inverseSize = 1.0f / static_cast<f32>(faceSize);
    const f32 u0 = 2.0f * (u - 0.5f * inverseSize) - 1.0f;
    const f32 u1 = 2.0f * (u + 0.5f * inverseSize) - 1.0f;
    const f32 v0 = 2.0f * (v - 0.5f * inverseSize) - 1.0f;
    const f32 v1 = 2.0f * (v + 0.5f * inverseSize) - 1.0f;
    return AreaElement(u0, v0) - AreaElement(u0, v1) - AreaElement(u1, v0) + AreaElement(u1, v1);
}

math::Color SampleCubemapNearest(const Cubemap& cubemap, const math::Vec3& direction,
                                 u32 level) noexcept
{
    if (!cubemap.IsValid())
    {
        return math::Color{0.0f, 0.0f, 0.0f, 0.0f};
    }
    const u32 clampedLevel = std::min(level, cubemap.MipLevels() - 1u);
    const i32 size = cubemap.MipSize(clampedLevel);
    u32 face = 0;
    f32 u = 0.0f;
    f32 v = 0.0f;
    CubeFaceUvFromDirection(direction, face, u, v);
    const i32 x = std::clamp(static_cast<i32>(u * static_cast<f32>(size)), 0, size - 1);
    const i32 y = std::clamp(static_cast<i32>(v * static_cast<f32>(size)), 0, size - 1);
    return cubemap.GetTexel(face, clampedLevel, x, y);
}

namespace
{

// Fetches one texel of a face, resolving a coordinate outside the face through
// the direction that coordinate represents. This is what makes bilinear
// filtering seamless across a cube edge.
math::Color FetchFaceTexel(const Cubemap& cubemap, u32 level, i32 size, u32 face, i32 x,
                           i32 y) noexcept
{
    if (x >= 0 && y >= 0 && x < size && y < size)
    {
        return cubemap.GetTexel(face, level, x, y);
    }
    const f32 inverseSize = 1.0f / static_cast<f32>(size);
    const f32 u = (static_cast<f32>(x) + 0.5f) * inverseSize;
    const f32 v = (static_cast<f32>(y) + 0.5f) * inverseSize;
    const math::Vec3 direction = CubeDirectionFromFaceUv(face, u, v);
    u32 neighbourFace = 0;
    f32 neighbourU = 0.0f;
    f32 neighbourV = 0.0f;
    CubeFaceUvFromDirection(direction, neighbourFace, neighbourU, neighbourV);
    const i32 neighbourX =
        std::clamp(static_cast<i32>(neighbourU * static_cast<f32>(size)), 0, size - 1);
    const i32 neighbourY =
        std::clamp(static_cast<i32>(neighbourV * static_cast<f32>(size)), 0, size - 1);
    return cubemap.GetTexel(neighbourFace, level, neighbourX, neighbourY);
}

}  // namespace

math::Color SampleCubemapFace(const Cubemap& cubemap, const math::Vec3& direction,
                              u32 level) noexcept
{
    if (!cubemap.IsValid())
    {
        return math::Color{0.0f, 0.0f, 0.0f, 0.0f};
    }
    const u32 clampedLevel = std::min(level, cubemap.MipLevels() - 1u);
    const i32 size = cubemap.MipSize(clampedLevel);
    if (size <= 1)
    {
        return SampleCubemapNearest(cubemap, direction, clampedLevel);
    }

    u32 face = 0;
    f32 u = 0.0f;
    f32 v = 0.0f;
    CubeFaceUvFromDirection(direction, face, u, v);

    const f32 x = u * static_cast<f32>(size) - 0.5f;
    const f32 y = v * static_cast<f32>(size) - 0.5f;
    const f32 flooredX = std::floor(x);
    const f32 flooredY = std::floor(y);
    const f32 fractionX = x - flooredX;
    const f32 fractionY = y - flooredY;
    const i32 x0 = static_cast<i32>(flooredX);
    const i32 y0 = static_cast<i32>(flooredY);

    const math::Color c00 = FetchFaceTexel(cubemap, clampedLevel, size, face, x0, y0);
    const math::Color c10 = FetchFaceTexel(cubemap, clampedLevel, size, face, x0 + 1, y0);
    const math::Color c01 = FetchFaceTexel(cubemap, clampedLevel, size, face, x0, y0 + 1);
    const math::Color c11 = FetchFaceTexel(cubemap, clampedLevel, size, face, x0 + 1, y0 + 1);

    const f32 w00 = (1.0f - fractionX) * (1.0f - fractionY);
    const f32 w10 = fractionX * (1.0f - fractionY);
    const f32 w01 = (1.0f - fractionX) * fractionY;
    const f32 w11 = fractionX * fractionY;
    return math::Color{c00.r * w00 + c10.r * w10 + c01.r * w01 + c11.r * w11,
                       c00.g * w00 + c10.g * w10 + c01.g * w01 + c11.g * w11,
                       c00.b * w00 + c10.b * w10 + c01.b * w01 + c11.b * w11,
                       c00.a * w00 + c10.a * w10 + c01.a * w01 + c11.a * w11};
}

math::Color SampleCubemapLevel(const Cubemap& cubemap, const math::Vec3& direction,
                               f32 level) noexcept
{
    if (!cubemap.IsValid())
    {
        return math::Color{0.0f, 0.0f, 0.0f, 0.0f};
    }
    const f32 maximumLevel = static_cast<f32>(cubemap.MipLevels() - 1u);
    const f32 clamped = math::IsFinite(level) ? std::clamp(level, 0.0f, maximumLevel) : 0.0f;
    const u32 lower = static_cast<u32>(std::floor(clamped));
    const f32 fraction = clamped - static_cast<f32>(lower);
    const math::Color low = SampleCubemapFace(cubemap, direction, lower);
    if (fraction <= 0.0f || lower + 1u >= cubemap.MipLevels())
    {
        return low;
    }
    const math::Color high = SampleCubemapFace(cubemap, direction, lower + 1u);
    return math::Color{low.r + (high.r - low.r) * fraction, low.g + (high.g - low.g) * fraction,
                       low.b + (high.b - low.b) * fraction, low.a + (high.a - low.a) * fraction};
}

// ---------------------------------------------------------------------------
// Equirectangular projection
// ---------------------------------------------------------------------------

namespace
{

struct EquirectContext
{
    const Texture* source = nullptr;
    const Framebuffer* framebuffer = nullptr;
    Cubemap* output = nullptr;
    i32 size = 0;
    u32 samplesPerAxis = 1;
    SamplerDesc sampler{};
};

math::Color SampleEquirectFramebuffer(const Framebuffer& source, f32 u, f32 v) noexcept
{
    u -= std::floor(u);
    v = std::clamp(v, 0.0f, 1.0f);
    const f32 x = u * static_cast<f32>(source.Width()) - 0.5f;
    const f32 y = v * static_cast<f32>(source.Height()) - 0.5f;
    const i32 x0Unwrapped = static_cast<i32>(std::floor(x));
    const i32 y0Unclamped = static_cast<i32>(std::floor(y));
    const f32 tx = x - static_cast<f32>(x0Unwrapped);
    const f32 ty = y - static_cast<f32>(y0Unclamped);
    const i32 width = source.Width();
    const i32 height = source.Height();
    const i32 x0 = (x0Unwrapped % width + width) % width;
    const i32 x1 = (x0 + 1) % width;
    const i32 y0 = std::clamp(y0Unclamped, 0, height - 1);
    const i32 y1 = std::clamp(y0Unclamped + 1, 0, height - 1);
    const math::Color c00 = source.GetPixel(x0, y0);
    const math::Color c10 = source.GetPixel(x1, y0);
    const math::Color c01 = source.GetPixel(x0, y1);
    const math::Color c11 = source.GetPixel(x1, y1);
    const math::Color top{c00.r + (c10.r - c00.r) * tx, c00.g + (c10.g - c00.g) * tx,
                          c00.b + (c10.b - c00.b) * tx, c00.a + (c10.a - c00.a) * tx};
    const math::Color bottom{c01.r + (c11.r - c01.r) * tx, c01.g + (c11.g - c01.g) * tx,
                             c01.b + (c11.b - c01.b) * tx, c01.a + (c11.a - c01.a) * tx};
    return math::Color{top.r + (bottom.r - top.r) * ty, top.g + (bottom.g - top.g) * ty,
                       top.b + (bottom.b - top.b) * ty, top.a + (bottom.a - top.a) * ty};
}

math::Vec2 EquirectangularUvFromDirection(const math::Vec3& direction) noexcept
{
    const math::Vec3 normalized = cyber::math::Normalize(direction);
    const f32 longitude = std::atan2(normalized.x, -normalized.z);
    const f32 latitude = std::asin(std::clamp(normalized.y, -1.0f, 1.0f));
    return math::Vec2{longitude / (2.0f * kPi) + 0.5f, 0.5f - latitude / kPi};
}

void ProjectEquirectRow(void* userData, u32 index) noexcept
{
    auto* context = static_cast<EquirectContext*>(userData);
    const i32 size = context->size;
    const u32 face = index / static_cast<u32>(size);
    const i32 y = static_cast<i32>(index % static_cast<u32>(size));
    const f32 inverseSize = 1.0f / static_cast<f32>(size);
    const u32 taps = context->samplesPerAxis;
    const f32 tapScale = 1.0f / static_cast<f32>(taps);

    for (i32 x = 0; x < size; ++x)
    {
        math::Vec3 sum{};
        for (u32 tapY = 0; tapY < taps; ++tapY)
        {
            for (u32 tapX = 0; tapX < taps; ++tapX)
            {
                const f32 u = (static_cast<f32>(x) + (static_cast<f32>(tapX) + 0.5f) * tapScale) *
                              inverseSize;
                const f32 v = (static_cast<f32>(y) + (static_cast<f32>(tapY) + 0.5f) * tapScale) *
                              inverseSize;
                const math::Vec3 direction = CubeDirectionFromFaceUv(face, u, v);
                const math::Vec2 sourceUv = EquirectangularUvFromDirection(direction);
                const math::Color texel =
                    context->source != nullptr
                        ? SampleTexture(*context->source, sourceUv.x, sourceUv.y, context->sampler)
                        : SampleEquirectFramebuffer(*context->framebuffer, sourceUv.x, sourceUv.y);
                sum = sum + ToVec3(texel);
            }
        }
        const f32 weight = tapScale * tapScale;
        context->output->SetTexel(face, 0, x, y, ToColor(sum * weight));
    }
}

}  // namespace

core::Status<core::Error> ProjectEquirectangularToCubemap(const Texture& source,
                                                          const EquirectangularProjectionDesc& desc,
                                                          Cubemap& output,
                                                          jobs::JobSystem* jobSystem) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (!source.IsValid())
    {
        return StatusType::Err(core::ErrorCode::InvalidArgument, "equirectangular source invalid");
    }
    if (!output.IsValid())
    {
        return StatusType::Err(core::ErrorCode::InvalidArgument, "cubemap output invalid");
    }
    if (desc.samplesPerAxis == 0 || desc.samplesPerAxis > 8)
    {
        return StatusType::Err(core::ErrorCode::InvalidArgument, "sample count out of range");
    }

    EquirectContext context{};
    context.source = &source;
    context.output = &output;
    context.size = output.FaceSize();
    context.samplesPerAxis = desc.samplesPerAxis;
    context.sampler.addressU = AddressMode::Repeat;
    context.sampler.addressV = AddressMode::ClampToEdge;

    RunRange(jobSystem, desc.useJobs, static_cast<u32>(context.size) * kCubeFaceCount,
             &ProjectEquirectRow, &context);
    output.GenerateMips();
    return StatusType::Ok();
}

core::Status<core::Error> ProjectEquirectangularToCubemap(const Framebuffer& source,
                                                          const EquirectangularProjectionDesc& desc,
                                                          Cubemap& output,
                                                          jobs::JobSystem* jobSystem) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (!source.IsValid())
    {
        return StatusType::Err(core::ErrorCode::InvalidArgument, "equirectangular source invalid");
    }
    if (!output.IsValid())
    {
        return StatusType::Err(core::ErrorCode::InvalidArgument, "cubemap output invalid");
    }
    if (desc.samplesPerAxis == 0 || desc.samplesPerAxis > 8)
    {
        return StatusType::Err(core::ErrorCode::InvalidArgument, "sample count out of range");
    }

    EquirectContext context{};
    context.framebuffer = &source;
    context.output = &output;
    context.size = output.FaceSize();
    context.samplesPerAxis = desc.samplesPerAxis;

    RunRange(jobSystem, desc.useJobs, static_cast<u32>(context.size) * kCubeFaceCount,
             &ProjectEquirectRow, &context);
    output.GenerateMips();
    return StatusType::Ok();
}

// ---------------------------------------------------------------------------
// Spherical harmonics
// ---------------------------------------------------------------------------

namespace
{

void EvaluateShBasis(const math::Vec3& direction, f32 basis[9]) noexcept
{
    const f32 x = direction.x;
    const f32 y = direction.y;
    const f32 z = direction.z;
    basis[0] = 0.282095f;
    basis[1] = 0.488603f * y;
    basis[2] = 0.488603f * z;
    basis[3] = 0.488603f * x;
    basis[4] = 1.092548f * x * y;
    basis[5] = 1.092548f * y * z;
    basis[6] = 0.315392f * (3.0f * z * z - 1.0f);
    basis[7] = 1.092548f * x * z;
    basis[8] = 0.546274f * (x * x - y * y);
}

}  // namespace

core::Status<core::Error> ProjectCubemapToSphericalHarmonics(const Cubemap& environment,
                                                             SphericalHarmonicsL2& output) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (!environment.IsValid())
    {
        return StatusType::Err(core::ErrorCode::InvalidArgument, "environment cubemap invalid");
    }

    for (usize index = 0; index < 9; ++index)
    {
        output.coefficients[index] = math::Vec3::Zero();
    }

    const i32 size = environment.FaceSize();
    const f32 inverseSize = 1.0f / static_cast<f32>(size);
    for (u32 face = 0; face < kCubeFaceCount; ++face)
    {
        for (i32 y = 0; y < size; ++y)
        {
            for (i32 x = 0; x < size; ++x)
            {
                const f32 u = (static_cast<f32>(x) + 0.5f) * inverseSize;
                const f32 v = (static_cast<f32>(y) + 0.5f) * inverseSize;
                const math::Vec3 direction = CubeDirectionFromFaceUv(face, u, v);
                const f32 solidAngle = CubeTexelSolidAngle(size, u, v);
                const math::Vec3 radiance =
                    FiniteNonNegative(ToVec3(environment.GetTexel(face, 0, x, y)));
                f32 basis[9]{};
                EvaluateShBasis(direction, basis);
                for (usize index = 0; index < 9; ++index)
                {
                    output.coefficients[index] =
                        output.coefficients[index] + radiance * (basis[index] * solidAngle);
                }
            }
        }
    }
    return StatusType::Ok();
}

math::Vec3 EvaluateSphericalHarmonicsIrradiance(const SphericalHarmonicsL2& harmonics,
                                                const math::Vec3& normal) noexcept
{
    const math::Vec3 n = cyber::math::Normalize(normal);
    if (!IsFinite(n))
    {
        return math::Vec3::Zero();
    }
    // Ramamoorthi and Hanrahan, with the cosine-lobe convolution constants
    // folded into the basis weights.
    constexpr f32 c1 = 0.429043f;
    constexpr f32 c2 = 0.511664f;
    constexpr f32 c3 = 0.743125f;
    constexpr f32 c4 = 0.886227f;
    constexpr f32 c5 = 0.247708f;

    const math::Vec3 result =
        harmonics.coefficients[8] * (c1 * (n.x * n.x - n.y * n.y)) +
        harmonics.coefficients[6] * (c3 * n.z * n.z) + harmonics.coefficients[0] * c4 -
        harmonics.coefficients[6] * c5 +
        (harmonics.coefficients[4] * (n.x * n.y) + harmonics.coefficients[7] * (n.x * n.z) +
         harmonics.coefficients[5] * (n.y * n.z)) *
            (2.0f * c1) +
        (harmonics.coefficients[3] * n.x + harmonics.coefficients[1] * n.y +
         harmonics.coefficients[2] * n.z) *
            (2.0f * c2);
    return FiniteNonNegative(result);
}

// ---------------------------------------------------------------------------
// Diffuse irradiance
// ---------------------------------------------------------------------------

namespace
{

struct IrradianceContext
{
    const Cubemap* environment = nullptr;
    Cubemap* output = nullptr;
    i32 outputSize = 0;
    i32 sourceSize = 0;
    u32 sourceLevel = 0;
};

void IntegrateIrradianceRow(void* userData, u32 index) noexcept
{
    auto* context = static_cast<IrradianceContext*>(userData);
    const i32 outputSize = context->outputSize;
    const u32 face = index / static_cast<u32>(outputSize);
    const i32 y = static_cast<i32>(index % static_cast<u32>(outputSize));
    const f32 inverseOutput = 1.0f / static_cast<f32>(outputSize);
    const i32 sourceSize = context->sourceSize;
    const f32 inverseSource = 1.0f / static_cast<f32>(sourceSize);

    for (i32 x = 0; x < outputSize; ++x)
    {
        const f32 u = (static_cast<f32>(x) + 0.5f) * inverseOutput;
        const f32 v = (static_cast<f32>(y) + 0.5f) * inverseOutput;
        const math::Vec3 normal = CubeDirectionFromFaceUv(face, u, v);

        math::Vec3 irradiance{};
        for (u32 sourceFace = 0; sourceFace < kCubeFaceCount; ++sourceFace)
        {
            for (i32 sourceY = 0; sourceY < sourceSize; ++sourceY)
            {
                for (i32 sourceX = 0; sourceX < sourceSize; ++sourceX)
                {
                    const f32 sourceU = (static_cast<f32>(sourceX) + 0.5f) * inverseSource;
                    const f32 sourceV = (static_cast<f32>(sourceY) + 0.5f) * inverseSource;
                    const math::Vec3 direction =
                        CubeDirectionFromFaceUv(sourceFace, sourceU, sourceV);
                    const f32 cosine = cyber::math::Dot(normal, direction);
                    if (cosine <= 0.0f)
                    {
                        continue;
                    }
                    const f32 solidAngle = CubeTexelSolidAngle(sourceSize, sourceU, sourceV);
                    const math::Vec3 radiance =
                        FiniteNonNegative(ToVec3(context->environment->GetTexel(
                            sourceFace, context->sourceLevel, sourceX, sourceY)));
                    irradiance = irradiance + radiance * (cosine * solidAngle);
                }
            }
        }
        context->output->SetTexel(face, 0, x, y, ToColor(irradiance));
    }
}

}  // namespace

core::Status<core::Error> BuildDiffuseIrradianceCubemap(const Cubemap& environment,
                                                        const IrradianceFilterDesc& desc,
                                                        Cubemap& output,
                                                        jobs::JobSystem* jobSystem) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (!environment.IsValid())
    {
        return StatusType::Err(core::ErrorCode::InvalidArgument, "environment cubemap invalid");
    }
    if (!output.IsValid())
    {
        return StatusType::Err(core::ErrorCode::InvalidArgument, "irradiance output invalid");
    }

    // The cosine convolution is band-limited, so integrating a coarse level of
    // the environment costs a fraction of the work and changes the result by
    // far less than the output resolution can represent.
    constexpr i32 kIntegrationSize = 32;
    u32 sourceLevel = 0;
    while (sourceLevel + 1u < environment.MipLevels() &&
           environment.MipSize(sourceLevel) > kIntegrationSize)
    {
        ++sourceLevel;
    }

    IrradianceContext context{};
    context.environment = &environment;
    context.output = &output;
    context.outputSize = output.FaceSize();
    context.sourceSize = environment.MipSize(sourceLevel);
    context.sourceLevel = sourceLevel;

    RunRange(jobSystem, desc.useJobs, static_cast<u32>(context.outputSize) * kCubeFaceCount,
             &IntegrateIrradianceRow, &context);
    output.GenerateMips();
    return StatusType::Ok();
}

// ---------------------------------------------------------------------------
// Specular prefilter
// ---------------------------------------------------------------------------

f32 SpecularMipRoughness(u32 level, u32 mipLevels) noexcept
{
    if (mipLevels <= 1)
    {
        return 0.0f;
    }
    const u32 clamped = std::min(level, mipLevels - 1u);
    return static_cast<f32>(clamped) / static_cast<f32>(mipLevels - 1u);
}

f32 SpecularRoughnessMip(f32 roughness, u32 mipLevels) noexcept
{
    if (mipLevels <= 1)
    {
        return 0.0f;
    }
    const f32 clamped = Saturate(roughness);
    return clamped * static_cast<f32>(mipLevels - 1u);
}

namespace
{

struct PrefilterContext
{
    const Cubemap* environment = nullptr;
    Cubemap* output = nullptr;
    u32 level = 0;
    i32 size = 0;
    f32 roughness = 0.0f;
    u32 sampleCount = 1;
    f32 sourceTexelSolidAngle = 0.0f;
};

void PrefilterRow(void* userData, u32 index) noexcept
{
    auto* context = static_cast<PrefilterContext*>(userData);
    const i32 size = context->size;
    const u32 face = index / static_cast<u32>(size);
    const i32 y = static_cast<i32>(index % static_cast<u32>(size));
    const f32 inverseSize = 1.0f / static_cast<f32>(size);
    const f32 roughness = context->roughness;
    const u32 sampleCount = context->sampleCount;

    for (i32 x = 0; x < size; ++x)
    {
        const f32 u = (static_cast<f32>(x) + 0.5f) * inverseSize;
        const f32 v = (static_cast<f32>(y) + 0.5f) * inverseSize;
        const math::Vec3 normal = CubeDirectionFromFaceUv(face, u, v);

        math::Vec3 sum{};
        f32 weightSum = 0.0f;
        for (u32 sample = 0; sample < sampleCount; ++sample)
        {
            const math::Vec2 random = Hammersley(sample, sampleCount);
            const math::Vec3 half = ImportanceSampleGgx(random, normal, roughness);
            const math::Vec3 light = cyber::math::Normalize(cyber::math::Reflect(-normal, half));
            const f32 ndotl = cyber::math::Dot(normal, light);
            if (ndotl <= 0.0f)
            {
                continue;
            }
            const f32 ndoth = Saturate(cyber::math::Dot(normal, half));
            const f32 distribution = DistributionGgx(ndoth, roughness);
            const f32 pdf = distribution * 0.25f + 1.0e-6f;
            const f32 sampleSolidAngle = 1.0f / (static_cast<f32>(sampleCount) * pdf);
            const f32 sourceLevel =
                roughness <= 0.0f
                    ? 0.0f
                    : std::max(0.0f,
                               0.5f * std::log2(sampleSolidAngle / context->sourceTexelSolidAngle));
            const math::Color radiance =
                SampleCubemapLevel(*context->environment, light, sourceLevel);
            sum = sum + FiniteNonNegative(ToVec3(radiance)) * ndotl;
            weightSum += ndotl;
        }

        const math::Vec3 filtered =
            weightSum > 0.0f
                ? sum * (1.0f / weightSum)
                : FiniteNonNegative(ToVec3(SampleCubemapFace(*context->environment, normal, 0)));
        context->output->SetTexel(face, context->level, x, y, ToColor(filtered));
    }
}

}  // namespace

core::Status<core::Error> PrefilterSpecularCubemap(const Cubemap& environment,
                                                   const SpecularPrefilterDesc& desc,
                                                   Cubemap& output,
                                                   jobs::JobSystem* jobSystem) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (!environment.IsValid())
    {
        return StatusType::Err(core::ErrorCode::InvalidArgument, "environment cubemap invalid");
    }
    if (!output.IsValid() || output.MipLevels() < 2)
    {
        return StatusType::Err(core::ErrorCode::InvalidArgument,
                               "prefiltered output needs at least two levels");
    }
    if (desc.sampleCount == 0 || desc.sampleCount > 4096)
    {
        return StatusType::Err(core::ErrorCode::InvalidArgument, "sample count out of range");
    }

    const i32 baseSize = output.FaceSize();
    const f32 sourceFaceSize = static_cast<f32>(environment.FaceSize());
    const f32 sourceTexelSolidAngle =
        4.0f * kPi / (6.0f * sourceFaceSize * sourceFaceSize + 1.0e-6f);

    // Level 0 is the mirror level and is resampled directly, so a perfectly
    // smooth surface reflects the environment itself rather than an estimate.
    {
        PrefilterContext context{};
        context.environment = &environment;
        context.output = &output;
        context.level = 0;
        context.size = baseSize;
        context.roughness = 0.0f;
        context.sampleCount = 1;
        context.sourceTexelSolidAngle = sourceTexelSolidAngle;
        RunRange(jobSystem, desc.useJobs, static_cast<u32>(baseSize) * kCubeFaceCount,
                 &PrefilterRow, &context);
    }

    for (u32 level = 1; level < output.MipLevels(); ++level)
    {
        PrefilterContext context{};
        context.environment = &environment;
        context.output = &output;
        context.level = level;
        context.size = output.MipSize(level);
        context.roughness = SpecularMipRoughness(level, output.MipLevels());
        context.sampleCount = desc.sampleCount;
        context.sourceTexelSolidAngle = sourceTexelSolidAngle;
        RunRange(jobSystem, desc.useJobs, static_cast<u32>(context.size) * kCubeFaceCount,
                 &PrefilterRow, &context);
    }
    return StatusType::Ok();
}

// ---------------------------------------------------------------------------
// Split-sum BRDF table
// ---------------------------------------------------------------------------

namespace
{

math::Vec2 IntegrateSplitSumBrdf(f32 ndotv, f32 roughness, u32 sampleCount) noexcept
{
    const f32 clampedNdotV = std::max(ndotv, 1.0e-4f);
    const math::Vec3 view{std::sqrt(std::max(0.0f, 1.0f - clampedNdotV * clampedNdotV)), 0.0f,
                          clampedNdotV};
    const math::Vec3 normal{0.0f, 0.0f, 1.0f};

    f32 scale = 0.0f;
    f32 bias = 0.0f;
    for (u32 sample = 0; sample < sampleCount; ++sample)
    {
        const math::Vec2 random = Hammersley(sample, sampleCount);
        const math::Vec3 half = ImportanceSampleGgx(random, normal, roughness);
        const math::Vec3 light = cyber::math::Reflect(-view, half);
        const f32 ndotl = light.z;
        if (ndotl <= 0.0f)
        {
            continue;
        }
        const f32 ndoth = Saturate(half.z);
        const f32 vdoth = Saturate(cyber::math::Dot(view, half));
        const f32 geometry = GeometrySmithIbl(clampedNdotV, ndotl, roughness);
        const f32 visibility = geometry * vdoth / (ndoth * clampedNdotV + 1.0e-6f);
        const f32 fresnel = std::pow(1.0f - vdoth, 5.0f);
        scale += (1.0f - fresnel) * visibility;
        bias += fresnel * visibility;
    }
    const f32 normalization = 1.0f / static_cast<f32>(sampleCount);
    return math::Vec2{scale * normalization, bias * normalization};
}

struct BrdfLutContext
{
    Texture* output = nullptr;
    i32 size = 0;
    u32 sampleCount = 0;
};

void BuildBrdfLutRow(void* userData, u32 index) noexcept
{
    auto* context = static_cast<BrdfLutContext*>(userData);
    const i32 size = context->size;
    const i32 y = static_cast<i32>(index);
    const f32 roughness = (static_cast<f32>(y) + 0.5f) / static_cast<f32>(size);
    for (i32 x = 0; x < size; ++x)
    {
        const f32 ndotv = (static_cast<f32>(x) + 0.5f) / static_cast<f32>(size);
        const math::Vec2 integrated = IntegrateSplitSumBrdf(ndotv, roughness, context->sampleCount);
        context->output->SetTexel(0, x, y, math::Color{integrated.x, integrated.y, 0.0f, 1.0f});
    }
}

// Karis mobile fit, used when no table is supplied.
math::Vec2 AnalyticSplitSumBrdf(f32 ndotv, f32 roughness) noexcept
{
    const f32 x = -1.0f * roughness + 1.0f;
    const f32 y = -0.0275f * roughness + 0.0425f;
    const f32 z = -0.572f * roughness + 1.04f;
    const f32 w = 0.022f * roughness - 0.04f;
    const f32 a004 = std::min(x * x, std::exp2(-9.28f * ndotv)) * x + y;
    return math::Vec2{-1.04f * a004 + z, 1.04f * a004 + w};
}

}  // namespace

core::Status<core::Error> BuildSplitSumBrdfLut(const BrdfLutDesc& desc, Texture& output,
                                               memory::Allocator& allocator,
                                               jobs::JobSystem* jobSystem) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (desc.size < 8 || desc.size > 512)
    {
        return StatusType::Err(core::ErrorCode::InvalidArgument, "brdf table size out of range");
    }
    if (desc.sampleCount < 16 || desc.sampleCount > 8192)
    {
        return StatusType::Err(core::ErrorCode::InvalidArgument, "sample count out of range");
    }

    TextureDesc textureDesc{};
    textureDesc.width = desc.size;
    textureDesc.height = desc.size;
    textureDesc.format = TextureFormat::RGBA32F;
    textureDesc.generateMips = false;
    auto created = Texture::Create(textureDesc, allocator);
    if (created.IsError())
    {
        return StatusType::Err(created.Error());
    }
    output = std::move(created).Value();

    BrdfLutContext context{};
    context.output = &output;
    context.size = desc.size;
    context.sampleCount = desc.sampleCount;
    RunRange(jobSystem, desc.useJobs, static_cast<u32>(desc.size), &BuildBrdfLutRow, &context);
    return StatusType::Ok();
}

core::Result<EnvironmentMap, core::Error>
EnvironmentMap::Create(const Framebuffer& equirectangular, const EnvironmentMapDesc& desc,
                       memory::Allocator& allocator, jobs::JobSystem* jobSystem) noexcept
{
    using ResultType = core::Result<EnvironmentMap, core::Error>;
    if (!equirectangular.IsValid() || desc.faceSize < 1 || desc.faceSize > kMaxCubemapFaceSize ||
        desc.diffuseFaceSize < 1 || desc.diffuseFaceSize > kMaxCubemapFaceSize ||
        desc.brdfLutSize < 8 || desc.brdfLutSize > 512 || desc.equirectangularSamplesPerAxis == 0 ||
        desc.equirectangularSamplesPerAxis > 8 || desc.specularSampleCount == 0 ||
        desc.specularSampleCount > 4096 || desc.brdfSampleCount < 16 || desc.brdfSampleCount > 8192)
    {
        return ResultType::Err(
            core::Error{core::ErrorCode::InvalidArgument, "invalid environment map descriptor"});
    }

    CubemapDesc sourceDesc{};
    sourceDesc.faceSize = desc.faceSize;
    auto source = Cubemap::Create(sourceDesc, allocator);
    if (!source)
    {
        return ResultType::Err(source.Error());
    }
    EquirectangularProjectionDesc projectionDesc{};
    projectionDesc.samplesPerAxis = desc.equirectangularSamplesPerAxis;
    projectionDesc.useJobs = desc.useJobs;
    auto projected =
        ProjectEquirectangularToCubemap(equirectangular, projectionDesc, source.Value(), jobSystem);
    if (!projected)
    {
        return ResultType::Err(projected.Error());
    }
    return CreateFromSource(std::move(source).Value(), desc, allocator, jobSystem);
}

core::Result<EnvironmentMap, core::Error>
EnvironmentMap::Create(const Cubemap& source, const EnvironmentMapDesc& desc,
                       memory::Allocator& allocator, jobs::JobSystem* jobSystem) noexcept
{
    using ResultType = core::Result<EnvironmentMap, core::Error>;
    if (!source.IsValid() || source.FaceSize() != desc.faceSize || desc.faceSize < 1 ||
        desc.faceSize > kMaxCubemapFaceSize || desc.diffuseFaceSize < 1 ||
        desc.diffuseFaceSize > kMaxCubemapFaceSize || desc.brdfLutSize < 8 ||
        desc.brdfLutSize > 512 || desc.specularSampleCount == 0 ||
        desc.specularSampleCount > 4096 || desc.brdfSampleCount < 16 || desc.brdfSampleCount > 8192)
    {
        return ResultType::Err(
            core::Error{core::ErrorCode::InvalidArgument, "invalid environment map descriptor"});
    }

    CubemapDesc sourceDesc{};
    sourceDesc.faceSize = desc.faceSize;
    auto sourceCopy = Cubemap::Create(sourceDesc, allocator);
    if (!sourceCopy)
    {
        return ResultType::Err(sourceCopy.Error());
    }
    Cubemap copiedSource = std::move(sourceCopy).Value();
    for (u32 face = 0; face < kCubeFaceCount; ++face)
    {
        for (i32 y = 0; y < desc.faceSize; ++y)
        {
            for (i32 x = 0; x < desc.faceSize; ++x)
            {
                copiedSource.SetTexel(face, 0, x, y, source.GetTexel(face, 0, x, y));
            }
        }
    }
    copiedSource.GenerateMips();
    return CreateFromSource(std::move(copiedSource), desc, allocator, jobSystem);
}

core::Result<EnvironmentMap, core::Error>
EnvironmentMap::CreateFromSource(Cubemap&& source, const EnvironmentMapDesc& desc,
                                 memory::Allocator& allocator, jobs::JobSystem* jobSystem) noexcept
{
    using ResultType = core::Result<EnvironmentMap, core::Error>;
    if (!source.IsValid() || source.FaceSize() != desc.faceSize)
    {
        return ResultType::Err(
            core::Error{core::ErrorCode::InvalidArgument, "invalid environment source cubemap"});
    }

    EnvironmentMap environment;
    environment.m_source = std::move(source);

    CubemapDesc specularDesc{};
    specularDesc.faceSize = desc.faceSize;
    auto specular = Cubemap::Create(specularDesc, allocator);
    if (!specular)
    {
        return ResultType::Err(specular.Error());
    }
    environment.m_specular = std::move(specular).Value();

    CubemapDesc diffuseDesc{};
    diffuseDesc.faceSize = desc.diffuseFaceSize;
    auto diffuse = Cubemap::Create(diffuseDesc, allocator);
    if (!diffuse)
    {
        return ResultType::Err(diffuse.Error());
    }
    environment.m_diffuseIrradiance = std::move(diffuse).Value();

    SpecularPrefilterDesc specularFilter{};
    specularFilter.sampleCount = desc.specularSampleCount;
    specularFilter.useJobs = desc.useJobs;
    auto prefiltered = PrefilterSpecularCubemap(environment.m_source, specularFilter,
                                                environment.m_specular, jobSystem);
    if (!prefiltered)
    {
        return ResultType::Err(prefiltered.Error());
    }

    IrradianceFilterDesc diffuseFilter{};
    diffuseFilter.useJobs = desc.useJobs;
    auto irradiance = BuildDiffuseIrradianceCubemap(environment.m_source, diffuseFilter,
                                                    environment.m_diffuseIrradiance, jobSystem);
    if (!irradiance)
    {
        return ResultType::Err(irradiance.Error());
    }

    BrdfLutDesc brdfDesc{};
    brdfDesc.size = desc.brdfLutSize;
    brdfDesc.sampleCount = desc.brdfSampleCount;
    brdfDesc.useJobs = desc.useJobs;
    auto brdf = BuildSplitSumBrdfLut(brdfDesc, environment.m_brdfLut, allocator, jobSystem);
    if (!brdf)
    {
        return ResultType::Err(brdf.Error());
    }
    return ResultType::Ok(std::move(environment));
}

core::Result<EnvironmentMap, core::Error> EnvironmentMap::Load(const char* utf8Path,
                                                               const EnvironmentMapDesc& desc,
                                                               memory::Allocator& allocator,
                                                               jobs::JobSystem* jobSystem) noexcept
{
    using ResultType = core::Result<EnvironmentMap, core::Error>;
    auto image = LoadImage(utf8Path, allocator);
    if (!image)
    {
        return ResultType::Err(image.Error());
    }
    return Create(image.Value(), desc, allocator, jobSystem);
}

bool EnvironmentMap::IsValid() const noexcept
{
    return m_source.IsValid() && m_specular.IsValid() && m_diffuseIrradiance.IsValid() &&
           m_brdfLut.IsValid();
}

EnvironmentLighting EnvironmentMap::Lighting() const noexcept
{
    EnvironmentLighting lighting{};
    lighting.specular = &m_specular;
    lighting.diffuseIrradiance = &m_diffuseIrradiance;
    lighting.brdfLut = &m_brdfLut;
    return lighting;
}

const Cubemap& EnvironmentMap::Source() const noexcept
{
    return m_source;
}

const Cubemap& EnvironmentMap::Specular() const noexcept
{
    return m_specular;
}

const Cubemap& EnvironmentMap::DiffuseIrradiance() const noexcept
{
    return m_diffuseIrradiance;
}

const Texture& EnvironmentMap::BrdfLut() const noexcept
{
    return m_brdfLut;
}

math::Vec2 SampleSplitSumBrdf(const Texture* lut, f32 ndotv, f32 roughness) noexcept
{
    const f32 clampedNdotV = Saturate(ndotv);
    const f32 clampedRoughness = Saturate(roughness);
    if (lut == nullptr || !lut->IsValid())
    {
        return AnalyticSplitSumBrdf(clampedNdotV, clampedRoughness);
    }
    SamplerDesc sampler{};
    sampler.addressU = AddressMode::ClampToEdge;
    sampler.addressV = AddressMode::ClampToEdge;
    sampler.mipFilter = FilterMode::Nearest;
    const math::Color texel =
        SampleTextureLevel(*lut, clampedNdotV, clampedRoughness, 0.0f, sampler);
    return math::Vec2{FiniteOrZero(texel.r), FiniteOrZero(texel.g)};
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

f32 ComputeSpecularOcclusion(f32 ndotv, f32 ambientOcclusion, f32 roughness) noexcept
{
    const f32 cosine = Saturate(ndotv);
    const f32 occlusion = Saturate(ambientOcclusion);
    const f32 clampedRoughness = Saturate(roughness);
    // Lagarde's ground-truth ambient-occlusion fit: a narrow lobe at low
    // roughness escapes the occluded region, a wide one does not.
    const f32 exponent = std::exp2(-16.0f * clampedRoughness - 1.0f);
    return Saturate(std::pow(cosine + occlusion, exponent) - 1.0f + occlusion);
}

namespace
{

struct IblTerms
{
    math::Vec3 diffuse{};
    math::Vec3 specular{};
};

math::Vec3 SampleEnvironmentIrradiance(const EnvironmentLighting& environment,
                                       const math::Vec3& normal) noexcept
{
    if (environment.diffuseIrradiance != nullptr && environment.diffuseIrradiance->IsValid())
    {
        return FiniteNonNegative(
            ToVec3(SampleCubemapFace(*environment.diffuseIrradiance, normal, 0)));
    }
    if (environment.diffuseSh != nullptr)
    {
        return EvaluateSphericalHarmonicsIrradiance(*environment.diffuseSh, normal);
    }
    return math::Vec3::Zero();
}

// Fdez-Aguera's multiple-scattering split-sum form. Without the second term a
// split-sum environment loses energy as roughness rises, which shows up as a
// rough metal that is visibly darker than its environment; with it a white
// furnace returns the environment exactly.
IblTerms EvaluateTerms(const EnvironmentLighting& environment, const IblSurface& surface,
                       const math::Vec3& reflectDirection, bool useGivenDirection,
                       const math::Vec3* radianceOverride = nullptr) noexcept
{
    IblTerms terms{};
    const f32 intensity = radianceOverride != nullptr ? 1.0f
                                                      : (math::IsFinite(environment.intensity) &&
                                                                 environment.intensity > 0.0f
                                                             ? environment.intensity
                                                             : 0.0f);
    if (intensity <= 0.0f)
    {
        return terms;
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
    const f32 ndotv = std::max(cyber::math::Dot(normal, view), 1.0e-4f);
    const f32 roughness = Saturate(surface.roughness);
    const f32 metallic = Saturate(surface.metallic);
    const f32 reflectance = Saturate(surface.reflectance);
    const math::Vec3 baseColor = FiniteNonNegative(surface.baseColor);
    const math::Vec3 f0 = math::Vec3{reflectance, reflectance, reflectance} * (1.0f - metallic) +
                          baseColor * metallic;
    const math::Vec3 diffuseAlbedo = baseColor * (1.0f - metallic);

    math::Vec3 irradiance = math::Vec3::Zero();
    if (radianceOverride == nullptr)
    {
        irradiance = surface.useIrradianceOverride
                         ? FiniteNonNegative(surface.irradianceOverride)
                         : SampleEnvironmentIrradiance(environment, normal) * intensity;
    }
    const math::Vec3 environmentRadiance = irradiance * kInversePi;

    const math::Vec2 split = SampleSplitSumBrdf(environment.brdfLut, ndotv, roughness);
    const f32 scale = std::max(0.0f, FiniteOrZero(split.x));
    const f32 bias = std::max(0.0f, FiniteOrZero(split.y));
    const math::Vec3 singleScatter = f0 * scale + math::Vec3{bias, bias, bias};
    const f32 singleScatterEnergy = Saturate(scale + bias);
    const f32 missingEnergy = 1.0f - singleScatterEnergy;
    const math::Vec3 averageFresnel = f0 + (math::Vec3::One() - f0) * (1.0f / 21.0f);
    const math::Vec3 multiScatterDenominator = math::Vec3::One() - averageFresnel * missingEnergy;
    const math::Vec3 multiScatter{
        multiScatterDenominator.x > 1.0e-5f
            ? singleScatter.x * averageFresnel.x / multiScatterDenominator.x
            : 0.0f,
        multiScatterDenominator.y > 1.0e-5f
            ? singleScatter.y * averageFresnel.y / multiScatterDenominator.y
            : 0.0f,
        multiScatterDenominator.z > 1.0e-5f
            ? singleScatter.z * averageFresnel.z / multiScatterDenominator.z
            : 0.0f};
    const math::Vec3 multiScatterEnergy = multiScatter * missingEnergy;

    const f32 occlusion = Saturate(surface.ambientOcclusion);
    const f32 specularOcclusion = environment.applySpecularOcclusion
                                      ? ComputeSpecularOcclusion(ndotv, occlusion, roughness)
                                      : 1.0f;

    // Energy left for the diffuse lobe once single and multiple specular
    // scattering have taken their share.
    const math::Vec3 diffuseWeight =
        math::Vec3{std::max(0.0f, 1.0f - singleScatter.x - multiScatterEnergy.x),
                   std::max(0.0f, 1.0f - singleScatter.y - multiScatterEnergy.y),
                   std::max(0.0f, 1.0f - singleScatter.z - multiScatterEnergy.z)};
    terms.diffuse =
        Multiply(Multiply(diffuseAlbedo, diffuseWeight), environmentRadiance) * occlusion;

    if (radianceOverride != nullptr)
    {
        terms.diffuse = math::Vec3::Zero();
        terms.specular =
            Multiply(FiniteNonNegative(*radianceOverride), singleScatter) * specularOcclusion;
        return terms;
    }

    if (environment.specular != nullptr && environment.specular->IsValid())
    {
        math::Vec3 direction = reflectDirection;
        if (!useGivenDirection || !IsFinite(direction) ||
            cyber::math::LengthSquared(direction) < 1.0e-12f)
        {
            direction = cyber::math::Reflect(-view, normal);
        }
        direction = cyber::math::Normalize(direction);
        const f32 level = SpecularRoughnessMip(roughness, environment.specular->MipLevels());
        const math::Vec3 prefiltered =
            FiniteNonNegative(ToVec3(SampleCubemapLevel(*environment.specular, direction, level))) *
            intensity;
        terms.specular = (Multiply(prefiltered, singleScatter) +
                          Multiply(multiScatterEnergy, environmentRadiance)) *
                         specularOcclusion;
    }
    return terms;
}

}  // namespace

math::Vec3 EvaluateIblDiffuse(const EnvironmentLighting& environment,
                              const IblSurface& surface) noexcept
{
    return EvaluateTerms(environment, surface, math::Vec3::Zero(), false).diffuse;
}

math::Vec3 EvaluateIblSpecular(const EnvironmentLighting& environment,
                               const IblSurface& surface) noexcept
{
    return EvaluateTerms(environment, surface, math::Vec3::Zero(), false).specular;
}

math::Vec3 EvaluateIblSpecularDirection(const EnvironmentLighting& environment,
                                        const IblSurface& surface,
                                        const math::Vec3& reflectDirection) noexcept
{
    return EvaluateTerms(environment, surface, reflectDirection, true).specular;
}

math::Vec3 EvaluateImageBasedLighting(const EnvironmentLighting& environment,
                                      const IblSurface& surface) noexcept
{
    const IblTerms terms = EvaluateTerms(environment, surface, math::Vec3::Zero(), false);
    return terms.diffuse + terms.specular;
}

math::Vec3 ApplySpecularEnvironmentWeight(const EnvironmentLighting& environment,
                                          const IblSurface& surface,
                                          const math::Vec3& radiance) noexcept
{
    return EvaluateTerms(environment, surface, math::Vec3::Zero(), false, &radiance).specular;
}

}  // namespace cyber::software
