#include <cyber/jobs/job_system.h>
#include <cyber/software/global_illumination.h>

#include <algorithm>
#include <cmath>
#include <new>
#include <utility>

namespace cyber::software
{

namespace
{

using cyber::core::Error;
using cyber::core::ErrorCode;

constexpr f32 kPi = cyber::math::kPi;
constexpr f32 kInversePi = 1.0f / kPi;

bool IsFinite(const math::Vec3& value) noexcept
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

math::Vec3 FiniteNonNegative(const math::Vec3& value) noexcept
{
    return math::Vec3{std::isfinite(value.x) && value.x > 0.0f ? value.x : 0.0f,
                      std::isfinite(value.y) && value.y > 0.0f ? value.y : 0.0f,
                      std::isfinite(value.z) && value.z > 0.0f ? value.z : 0.0f};
}

math::Vec3 Saturate(const math::Vec3& value) noexcept
{
    const math::Vec3 positive = FiniteNonNegative(value);
    return math::Vec3{std::min(positive.x, 1.0f), std::min(positive.y, 1.0f),
                      std::min(positive.z, 1.0f)};
}

math::Vec3 Multiply(const math::Vec3& lhs, const math::Vec3& rhs) noexcept
{
    return math::Vec3{lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
}

f32 MaxComponent(const math::Vec3& value) noexcept
{
    return std::max(value.x, std::max(value.y, value.z));
}

f32 SmoothRangeFade(f32 distance, f32 range) noexcept
{
    if (!std::isfinite(distance) || !std::isfinite(range) || distance < 0.0f || range <= 0.0f ||
        distance >= range)
    {
        return 0.0f;
    }
    const f32 ratio = distance / range;
    const f32 fade = 1.0f - ratio * ratio;
    return fade * fade;
}

// Counter-based randomness. lowbias32 is a well-mixed 32-bit integer hash;
// seeding a stream from (sequence, sample) and stepping a counter through it
// gives every sample an independent, reproducible stream that does not depend
// on which thread evaluates it or in what order.
u32 Hash(u32 value) noexcept
{
    value ^= value >> 16u;
    value *= 0x7FEB352Du;
    value ^= value >> 15u;
    value *= 0x846CA68Bu;
    value ^= value >> 16u;
    return value;
}

class SampleStream
{
public:
    SampleStream(u32 sequence, u32 sample) noexcept
        : m_seed(Hash(sequence * 0x9E3779B9u ^ Hash(sample + 0x632BE5ABu)))
    {
    }

    // Uniform in [0, 1).
    f32 Next() noexcept
    {
        const u32 bits = Hash(m_seed + m_counter * 0x85EBCA6Bu);
        ++m_counter;
        return static_cast<f32>(bits >> 8u) * (1.0f / 16777216.0f);
    }

private:
    u32 m_seed = 0;
    u32 m_counter = 0;
};

// Orthonormal basis around a unit normal (Duff et al. 2017), branch-free except
// for the sign, and continuous everywhere but the sign flip.
void BuildBasis(const math::Vec3& normal, math::Vec3& tangent, math::Vec3& bitangent) noexcept
{
    const f32 sign = std::copysign(1.0f, normal.z);
    const f32 a = -1.0f / (sign + normal.z);
    const f32 b = normal.x * normal.y * a;
    tangent = math::Vec3{1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x};
    bitangent = math::Vec3{b, sign + normal.y * normal.y * a, -normal.y};
}

math::Vec3 SampleCosineHemisphere(const math::Vec3& normal, f32 u1, f32 u2) noexcept
{
    math::Vec3 tangent{};
    math::Vec3 bitangent{};
    BuildBasis(normal, tangent, bitangent);
    const f32 radius = std::sqrt(u1);
    const f32 phi = 2.0f * kPi * u2;
    const f32 height = std::sqrt(std::max(0.0f, 1.0f - u1));
    return math::Normalize(tangent * (radius * std::cos(phi)) +
                           bitangent * (radius * std::sin(phi)) + normal * height);
}

// Spherical Fibonacci lattice. The golden-ratio increment is taken in double
// precision so that the azimuth stays well distributed past a few thousand
// points, where a binary32 product starts to repeat.
math::Vec3 FibonacciSphere(u32 index, u32 count) noexcept
{
    const f32 y = 1.0f - 2.0f * (static_cast<f32>(index) + 0.5f) / static_cast<f32>(count);
    const f32 radius = std::sqrt(std::max(0.0f, 1.0f - y * y));
    const double turns = static_cast<double>(index) * 0.6180339887498949;
    const f32 phi = static_cast<f32>(2.0 * 3.14159265358979323846 * (turns - std::floor(turns)));
    return math::Vec3{radius * std::cos(phi), y, radius * std::sin(phi)};
}

// A uniformly distributed rotation from three uniforms (Shoemake 1992), as the
// rows of a 3x3 matrix.
struct Rotation
{
    math::Vec3 row0{1.0f, 0.0f, 0.0f};
    math::Vec3 row1{0.0f, 1.0f, 0.0f};
    math::Vec3 row2{0.0f, 0.0f, 1.0f};

    [[nodiscard]] math::Vec3 Apply(const math::Vec3& value) const noexcept
    {
        return math::Vec3{math::Dot(row0, value), math::Dot(row1, value), math::Dot(row2, value)};
    }
};

Rotation RandomRotation(f32 u1, f32 u2, f32 u3) noexcept
{
    const f32 r1 = std::sqrt(std::max(0.0f, 1.0f - u1));
    const f32 r2 = std::sqrt(u1);
    const f32 t1 = 2.0f * kPi * u2;
    const f32 t2 = 2.0f * kPi * u3;
    const f32 x = r1 * std::sin(t1);
    const f32 y = r1 * std::cos(t1);
    const f32 z = r2 * std::sin(t2);
    const f32 w = r2 * std::cos(t2);
    Rotation rotation{};
    rotation.row0 =
        math::Vec3{1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - z * w), 2.0f * (x * z + y * w)};
    rotation.row1 =
        math::Vec3{2.0f * (x * y + z * w), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - x * w)};
    rotation.row2 =
        math::Vec3{2.0f * (x * z - y * w), 2.0f * (y * z + x * w), 1.0f - 2.0f * (x * x + y * y)};
    return rotation;
}

// The same real order-2 basis and coefficient order as ibl.cpp, so a probe's
// projection can be convolved by EvaluateSphericalHarmonicsIrradiance.
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

// Weight a ray contributes to a visibility texel: the cosine between the ray
// and the texel centre raised to the 32nd power, which falls to one half at
// about 12 degrees, the texel spacing. Narrower lobes would leave texels with
// no rays at low ray counts; wider ones reintroduce the averaging past the
// wall that made coarser maps leak.
f32 VisibilityLobeWeight(f32 cosine) noexcept
{
    if (!(cosine > 0.0f))
    {
        return 0.0f;
    }
    const f32 c2 = cosine * cosine;
    const f32 c4 = c2 * c2;
    const f32 c8 = c4 * c4;
    const f32 c16 = c8 * c8;
    return c16 * c16;
}

math::Vec3 VisibilityTexelDirection(u32 texel) noexcept
{
    const u32 u = texel % kIrradianceVisibilityResolution;
    const u32 v = texel / kIrradianceVisibilityResolution;
    const f32 scale = 1.0f / static_cast<f32>(kIrradianceVisibilityResolution);
    return DecodeOctahedral(
        math::Vec2{(static_cast<f32>(u) + 0.5f) * scale, (static_cast<f32>(v) + 0.5f) * scale});
}

// Distance moments in a direction, bilinear over the four nearest texels.
// Coordinates clamp at the map edge; the octahedral fold makes edge texels
// neighbours of texels on the opposite edge, and clamping rather than wrapping
// there costs a little smoothness across the -Z fold, which the bilinear
// weights already make small.
void ReadVisibility(const IrradianceProbe& probe, const math::Vec3& direction, f32& mean,
                    f32& meanSquared) noexcept
{
    const math::Vec2 uv = EncodeOctahedral(direction);
    const f32 resolution = static_cast<f32>(kIrradianceVisibilityResolution);
    const f32 maxCoordinate = resolution - 1.0f;
    const f32 x = std::clamp(uv.x * resolution - 0.5f, 0.0f, maxCoordinate);
    const f32 y = std::clamp(uv.y * resolution - 0.5f, 0.0f, maxCoordinate);
    const u32 x0 = std::min(static_cast<u32>(x), kIrradianceVisibilityResolution - 1);
    const u32 y0 = std::min(static_cast<u32>(y), kIrradianceVisibilityResolution - 1);
    const u32 x1 = std::min(x0 + 1, kIrradianceVisibilityResolution - 1);
    const u32 y1 = std::min(y0 + 1, kIrradianceVisibilityResolution - 1);
    const f32 fx = x - static_cast<f32>(x0);
    const f32 fy = y - static_cast<f32>(y0);
    const auto at = [](u32 tx, u32 ty, const f32* values) noexcept
    { return values[ty * kIrradianceVisibilityResolution + tx]; };
    const auto blend = [&](const f32* values) noexcept
    {
        const f32 top = at(x0, y0, values) + (at(x1, y0, values) - at(x0, y0, values)) * fx;
        const f32 bottom = at(x0, y1, values) + (at(x1, y1, values) - at(x0, y1, values)) * fx;
        return top + (bottom - top) * fy;
    };
    mean = blend(probe.meanDistance);
    meanSquared = blend(probe.meanDistanceSquared);
}

const GiSurfaceMaterial& MaterialFor(const GiScene& scene, u32 instanceId) noexcept
{
    static const GiSurfaceMaterial kDefaultMaterial{};
    if (instanceId < scene.materials.Size() && scene.materials.Data() != nullptr)
    {
        return scene.materials[instanceId];
    }
    return kDefaultMaterial;
}

math::Vec3 SkyRadiance(const GiScene& scene, const math::Vec3& direction) noexcept
{
    if (scene.skyCubemap != nullptr && scene.skyCubemap->IsValid())
    {
        const math::Color texel = SampleCubemapLevel(*scene.skyCubemap, direction, 0.0f);
        return FiniteNonNegative(math::Vec3{texel.r, texel.g, texel.b});
    }
    return FiniteNonNegative(scene.skyRadiance);
}

f32 OffsetScale(const GiScene& scene) noexcept
{
    return std::isfinite(scene.rayOffsetScale) && scene.rayOffsetScale > 0.0f ? scene.rayOffsetScale
                                                                              : 1.0f;
}

// Direct irradiance at a surface point from every analytic light, each tested
// with an any-hit shadow ray. Returned in W/m^2; the caller applies albedo/pi.
math::Vec3 GatherDirectIrradiance(const GiScene& scene, const math::Vec3& origin,
                                  const math::Vec3& shadingNormal,
                                  const math::Vec3& geometricNormal, PathTraceStats& stats) noexcept
{
    math::Vec3 irradiance{};
    const usize directionalCount =
        std::min<usize>(scene.directionalLights.Size(), kMaxGiDirectionalLights);
    for (usize index = 0; index < directionalCount; ++index)
    {
        const GiDirectionalLight& light = scene.directionalLights[index];
        const math::Vec3 toLight = math::Normalize(-light.direction);
        const f32 cosine = math::Dot(shadingNormal, toLight);
        if (!(cosine > 0.0f) || math::Dot(geometricNormal, toLight) <= 0.0f)
        {
            continue;
        }
        RayDesc shadow{};
        shadow.origin = origin;
        shadow.direction = toLight;
        ++stats.shadowRays;
        if (!scene.geometry->IntersectAny(shadow, kRayInstanceMaskAll, &stats.traversal))
        {
            irradiance += FiniteNonNegative(light.irradiance) * cosine;
        }
    }

    const usize pointCount = std::min<usize>(scene.pointLights.Size(), kMaxGiPointLights);
    for (usize index = 0; index < pointCount; ++index)
    {
        const GiPointLight& light = scene.pointLights[index];
        const math::Vec3 offset = light.position - origin;
        const f32 distanceSquared = math::LengthSquared(offset);
        if (!(distanceSquared > 1.0e-8f))
        {
            continue;
        }
        const f32 distance = std::sqrt(distanceSquared);
        const math::Vec3 toLight = offset * (1.0f / distance);
        const f32 cosine = math::Dot(shadingNormal, toLight);
        const f32 fade = SmoothRangeFade(distance, light.range);
        if (!(cosine > 0.0f) || fade <= 0.0f || math::Dot(geometricNormal, toLight) <= 0.0f)
        {
            continue;
        }
        ++stats.shadowRays;
        if (TraceVisibility(*scene.geometry, origin, light.position, kRayInstanceMaskAll, 1.0e-4f,
                            &stats.traversal))
        {
            irradiance += FiniteNonNegative(light.intensity) * (cosine * fade / distanceSquared);
        }
    }
    return irradiance;
}

// The integrator. firstHit, when supplied, receives the primary hit so a probe
// bake can read the distance and facing without tracing the ray twice.
math::Vec3 TracePath(const GiScene& scene, const RayDesc& ray, const PathTraceSettings& settings,
                     SampleStream& stream, PathTraceStats& stats, RayHit* firstHit) noexcept
{
    math::Vec3 radiance{};
    math::Vec3 throughput{1.0f, 1.0f, 1.0f};
    RayDesc current = ray;
    const f32 offsetScale = OffsetScale(scene);
    const u32 maxBounces = std::min(settings.maxBounces, 32u);

    for (u32 bounce = 0;; ++bounce)
    {
        RayHit hit{};
        ++stats.rays;
        if (!scene.geometry->Intersect(current, hit, kRayInstanceMaskAll, &stats.traversal))
        {
            radiance +=
                Multiply(throughput, SkyRadiance(scene, math::Normalize(current.direction)));
            break;
        }
        if (bounce == 0 && firstHit != nullptr)
        {
            *firstHit = hit;
        }

        const GiSurfaceMaterial& material = MaterialFor(scene, hit.instanceId);
        if (hit.frontFace)
        {
            radiance += Multiply(throughput, FiniteNonNegative(material.emission));
        }

        // Reflection is two-sided: the hit normals already face the ray, so a
        // single-sided quad such as a floor lights correctly from either side.
        // Emission stays front-only, which is how an area light is aimed.
        const math::Vec3 albedo = Saturate(material.albedo);
        const math::Vec3 origin = OffsetRayOrigin(hit.position, hit.geometricNormal, offsetScale);
        const math::Vec3 direct =
            GatherDirectIrradiance(scene, origin, hit.shadingNormal, hit.geometricNormal, stats);
        radiance += Multiply(Multiply(throughput, albedo), direct * kInversePi);

        if (bounce >= maxBounces)
        {
            break;
        }

        // Cosine-weighted sampling makes the Lambertian estimator albedo times
        // the incoming radiance: the cosine and the 1/pi cancel against the pdf.
        throughput = Multiply(throughput, albedo);
        if (settings.russianRoulette && bounce + 1 >= settings.rouletteStartBounce)
        {
            const f32 survival = std::clamp(MaxComponent(throughput), 0.05f, 0.95f);
            if (stream.Next() >= survival)
            {
                ++stats.terminatedByRoulette;
                // Consume the direction samples anyway, so the stream position
                // does not depend on the roulette outcome.
                (void)stream.Next();
                (void)stream.Next();
                break;
            }
            throughput = throughput * (1.0f / survival);
        }
        else
        {
            (void)stream.Next();
        }
        if (!(MaxComponent(throughput) > 0.0f))
        {
            break;
        }

        const f32 u1 = stream.Next();
        const f32 u2 = stream.Next();
        const math::Vec3 direction = SampleCosineHemisphere(hit.shadingNormal, u1, u2);
        if (math::Dot(direction, hit.geometricNormal) <= 0.0f)
        {
            // A shading normal tilted past the geometric one can sample under
            // the surface. Ending the path there loses a sliver of energy at
            // grazing normal-mapped angles and never leaks light through.
            break;
        }
        current = RayDesc{};
        current.origin = origin;
        current.direction = direction;
    }

    if (settings.maxPathRadiance > 0.0f && std::isfinite(settings.maxPathRadiance))
    {
        const f32 peak = MaxComponent(radiance);
        if (peak > settings.maxPathRadiance)
        {
            radiance = radiance * (settings.maxPathRadiance / peak);
        }
    }
    return FiniteNonNegative(radiance);
}

usize AlignUp(usize value, usize alignment) noexcept
{
    return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

core::Status<core::Error> ValidateGiScene(const GiScene& scene) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (scene.geometry == nullptr || !scene.geometry->IsValid())
    {
        return StatusType::Err(ErrorCode::InvalidArgument, "GI scene has no valid geometry");
    }
    if (scene.directionalLights.Size() > kMaxGiDirectionalLights)
    {
        return StatusType::Err(ErrorCode::InvalidArgument,
                               "GI scene has too many directional lights");
    }
    if (scene.pointLights.Size() > kMaxGiPointLights)
    {
        return StatusType::Err(ErrorCode::InvalidArgument, "GI scene has too many point lights");
    }
    for (usize index = 0; index < scene.directionalLights.Size(); ++index)
    {
        const GiDirectionalLight& light = scene.directionalLights[index];
        if (!IsFinite(light.direction) || math::LengthSquared(light.direction) <= 0.0f ||
            !IsFinite(light.irradiance))
        {
            return StatusType::Err(ErrorCode::InvalidArgument,
                                   "GI directional light is not finite");
        }
    }
    for (usize index = 0; index < scene.pointLights.Size(); ++index)
    {
        const GiPointLight& light = scene.pointLights[index];
        if (!IsFinite(light.position) || !IsFinite(light.intensity) || !std::isfinite(light.range))
        {
            return StatusType::Err(ErrorCode::InvalidArgument, "GI point light is not finite");
        }
    }
    if (!std::isfinite(scene.rayOffsetScale) || scene.rayOffsetScale < 0.0f ||
        !IsFinite(scene.skyRadiance))
    {
        return StatusType::Err(ErrorCode::InvalidArgument, "GI scene sky or offset is invalid");
    }
    return StatusType::Ok();
}

// ---------------------------------------------------------------------------
// Path tracing
// ---------------------------------------------------------------------------

math::Vec3 TracePathRadiance(const GiScene& scene, const RayDesc& ray,
                             const PathTraceSettings& settings, u32 sequence, u32 sample,
                             PathTraceStats* stats) noexcept
{
    if (scene.geometry == nullptr || !scene.geometry->IsValid())
    {
        return math::Vec3::Zero();
    }
    SampleStream stream(sequence, sample);
    PathTraceStats local{};
    const math::Vec3 radiance = TracePath(scene, ray, settings, stream, local, nullptr);
    if (stats != nullptr)
    {
        stats->Add(local);
    }
    return radiance;
}

math::Vec3 EstimateIrradiance(const GiScene& scene, const math::Vec3& position,
                              const math::Vec3& normal, const PathTraceSettings& settings,
                              u32 sampleCount, u32 sequence, PathTraceStats* stats) noexcept
{
    if (scene.geometry == nullptr || !scene.geometry->IsValid() || sampleCount == 0 ||
        !IsFinite(position) || !IsFinite(normal) || math::LengthSquared(normal) <= 0.0f)
    {
        return math::Vec3::Zero();
    }
    const math::Vec3 n = math::Normalize(normal);
    const math::Vec3 origin = OffsetRayOrigin(position, n, OffsetScale(scene));
    // The hemisphere directions and the paths they launch draw from separate
    // streams, so changing the bounce count does not reshuffle the directions.
    const u32 pathSequence = Hash(sequence ^ 0x68BC21EBu);
    PathTraceStats local{};
    math::Vec3 sum{};
    for (u32 sample = 0; sample < sampleCount; ++sample)
    {
        SampleStream directions(sequence, sample);
        const f32 u1 = directions.Next();
        const f32 u2 = directions.Next();
        RayDesc ray{};
        ray.origin = origin;
        ray.direction = SampleCosineHemisphere(n, u1, u2);
        SampleStream path(pathSequence, sample);
        sum += TracePath(scene, ray, settings, path, local, nullptr);
    }
    if (stats != nullptr)
    {
        stats->Add(local);
    }
    return sum * (kPi / static_cast<f32>(sampleCount));
}

// ---------------------------------------------------------------------------
// Octahedral mapping
// ---------------------------------------------------------------------------

math::Vec2 EncodeOctahedral(const math::Vec3& direction) noexcept
{
    const f32 sum = std::fabs(direction.x) + std::fabs(direction.y) + std::fabs(direction.z);
    if (!(sum > 0.0f) || !std::isfinite(sum))
    {
        return math::Vec2{0.5f, 0.5f};
    }
    f32 x = direction.x / sum;
    f32 y = direction.y / sum;
    if (direction.z < 0.0f)
    {
        const f32 foldedX = (1.0f - std::fabs(y)) * (x >= 0.0f ? 1.0f : -1.0f);
        const f32 foldedY = (1.0f - std::fabs(x)) * (y >= 0.0f ? 1.0f : -1.0f);
        x = foldedX;
        y = foldedY;
    }
    return math::Vec2{x * 0.5f + 0.5f, y * 0.5f + 0.5f};
}

math::Vec3 DecodeOctahedral(const math::Vec2& uv) noexcept
{
    const f32 x = uv.x * 2.0f - 1.0f;
    const f32 y = uv.y * 2.0f - 1.0f;
    const f32 z = 1.0f - std::fabs(x) - std::fabs(y);
    math::Vec3 direction{x, y, z};
    if (z < 0.0f)
    {
        direction.x = (1.0f - std::fabs(y)) * (x >= 0.0f ? 1.0f : -1.0f);
        direction.y = (1.0f - std::fabs(x)) * (y >= 0.0f ? 1.0f : -1.0f);
    }
    return math::Normalize(direction);
}

// ---------------------------------------------------------------------------
// Irradiance volume
// ---------------------------------------------------------------------------

IrradianceVolume::~IrradianceVolume()
{
    Release();
}

IrradianceVolume::IrradianceVolume(IrradianceVolume&& other) noexcept
{
    MoveFrom(other);
}

IrradianceVolume& IrradianceVolume::operator=(IrradianceVolume&& other) noexcept
{
    if (this != &other)
    {
        Release();
        MoveFrom(other);
    }
    return *this;
}

void IrradianceVolume::Release() noexcept
{
    if (m_allocator != nullptr && m_probes != nullptr)
    {
        m_allocator->Deallocate(m_probes);
    }
    m_allocator = nullptr;
    m_probes = nullptr;
    m_sizeBytes = 0;
    m_probeCount = 0;
    m_desc = IrradianceVolumeDesc{};
}

void IrradianceVolume::MoveFrom(IrradianceVolume& other) noexcept
{
    m_allocator = std::exchange(other.m_allocator, nullptr);
    m_probes = std::exchange(other.m_probes, nullptr);
    m_sizeBytes = std::exchange(other.m_sizeBytes, 0);
    m_probeCount = std::exchange(other.m_probeCount, 0);
    m_desc = std::exchange(other.m_desc, IrradianceVolumeDesc{});
}

core::Result<IrradianceVolume, core::Error>
IrradianceVolume::Create(const IrradianceVolumeDesc& desc, memory::Allocator& allocator) noexcept
{
    using ResultType = core::Result<IrradianceVolume, core::Error>;
    if (!IsFinite(desc.boundsMin) || !IsFinite(desc.boundsMax) ||
        !(desc.boundsMax.x > desc.boundsMin.x) || !(desc.boundsMax.y > desc.boundsMin.y) ||
        !(desc.boundsMax.z > desc.boundsMin.z))
    {
        return ResultType::Err(ErrorCode::InvalidArgument, "irradiance volume bounds are empty");
    }
    const auto validAxis = [](u32 count) noexcept
    { return count >= 2 && count <= kMaxIrradianceProbesPerAxis; };
    if (!validAxis(desc.probeCountX) || !validAxis(desc.probeCountY) ||
        !validAxis(desc.probeCountZ))
    {
        return ResultType::Err(ErrorCode::InvalidArgument,
                               "irradiance volume probe count per axis out of range");
    }
    const u64 total = u64{desc.probeCountX} * u64{desc.probeCountY} * u64{desc.probeCountZ};
    if (total > kMaxIrradianceProbes)
    {
        return ResultType::Err(ErrorCode::InvalidArgument, "irradiance volume has too many probes");
    }

    const u32 count = static_cast<u32>(total);
    const usize sizeBytes = sizeof(IrradianceProbe) * count;
    void* memory = allocator.Allocate(AlignUp(sizeBytes, 16), 16);
    if (memory == nullptr)
    {
        return ResultType::Err(ErrorCode::Internal, "irradiance volume allocation failed");
    }
    IrradianceVolume volume;
    volume.m_allocator = &allocator;
    volume.m_probes = static_cast<IrradianceProbe*>(memory);
    volume.m_sizeBytes = sizeBytes;
    volume.m_probeCount = count;
    volume.m_desc = desc;
    for (u32 index = 0; index < count; ++index)
    {
        new (&volume.m_probes[index]) IrradianceProbe{};
    }
    return ResultType::Ok(std::move(volume));
}

bool IrradianceVolume::IsValid() const noexcept
{
    return m_probes != nullptr && m_probeCount > 0;
}

const IrradianceVolumeDesc& IrradianceVolume::Desc() const noexcept
{
    return m_desc;
}

u32 IrradianceVolume::ProbeCount() const noexcept
{
    return m_probeCount;
}

math::Vec3 IrradianceVolume::ProbeSpacing() const noexcept
{
    if (!IsValid())
    {
        return math::Vec3::Zero();
    }
    const math::Vec3 extent = m_desc.boundsMax - m_desc.boundsMin;
    return math::Vec3{extent.x / static_cast<f32>(m_desc.probeCountX - 1),
                      extent.y / static_cast<f32>(m_desc.probeCountY - 1),
                      extent.z / static_cast<f32>(m_desc.probeCountZ - 1)};
}

usize IrradianceVolume::DataSizeBytes() const noexcept
{
    return m_sizeBytes;
}

memory::Allocator* IrradianceVolume::GetAllocator() const noexcept
{
    return m_allocator;
}

u32 IrradianceVolume::ProbeIndex(u32 x, u32 y, u32 z) const noexcept
{
    if (!IsValid() || x >= m_desc.probeCountX || y >= m_desc.probeCountY || z >= m_desc.probeCountZ)
    {
        return kInvalidRayIndex;
    }
    return x + m_desc.probeCountX * (y + m_desc.probeCountY * z);
}

math::Vec3 IrradianceVolume::ProbePosition(u32 x, u32 y, u32 z) const noexcept
{
    const math::Vec3 spacing = ProbeSpacing();
    return m_desc.boundsMin + math::Vec3{spacing.x * static_cast<f32>(x),
                                         spacing.y * static_cast<f32>(y),
                                         spacing.z * static_cast<f32>(z)};
}

math::Vec3 IrradianceVolume::ProbePosition(u32 index) const noexcept
{
    if (!IsValid() || index >= m_probeCount)
    {
        return math::Vec3::Zero();
    }
    const u32 x = index % m_desc.probeCountX;
    const u32 y = (index / m_desc.probeCountX) % m_desc.probeCountY;
    const u32 z = index / (m_desc.probeCountX * m_desc.probeCountY);
    return ProbePosition(x, y, z);
}

const IrradianceProbe& IrradianceVolume::Probe(u32 index) const noexcept
{
    static const IrradianceProbe kUnbaked{};
    if (!IsValid() || index >= m_probeCount)
    {
        return kUnbaked;
    }
    return m_probes[index];
}

void IrradianceVolume::SetProbe(u32 index, const IrradianceProbe& probe) noexcept
{
    if (IsValid() && index < m_probeCount)
    {
        m_probes[index] = probe;
    }
}

void IrradianceVolume::Clear() noexcept
{
    for (u32 index = 0; index < m_probeCount; ++index)
    {
        m_probes[index] = IrradianceProbe{};
    }
}

// ---------------------------------------------------------------------------
// Bake
// ---------------------------------------------------------------------------

namespace
{

struct BakeJob
{
    const GiScene* scene = nullptr;
    const IrradianceBakeDesc* desc = nullptr;
    const IrradianceVolume* volume = nullptr;
    IrradianceProbe* probes = nullptr;
    PathTraceStats* stats = nullptr;
    f32 maxDistance = 1.0f;
    f32 rayStart = kRayDefaultTMin;
};

void BakeProbe(void* userData, u32 index) noexcept
{
    const BakeJob& job = *static_cast<const BakeJob*>(userData);
    const GiScene& scene = *job.scene;
    const IrradianceBakeDesc& desc = *job.desc;
    const math::Vec3 position = job.volume->ProbePosition(index);
    const u32 rayCount = desc.raysPerProbe;
    // Separate seeds for the lattice rotation and for each ray's path, both
    // derived from the probe index, so the bake is a pure function of it.
    const u32 probeSequence = Hash(index * 0x27D4EB2Fu + 0x165667B1u);

    Rotation rotation{};
    if (desc.rotatePerProbe)
    {
        SampleStream rotationStream(probeSequence, 0xFFFFFFFFu);
        const f32 u1 = rotationStream.Next();
        const f32 u2 = rotationStream.Next();
        const f32 u3 = rotationStream.Next();
        rotation = RandomRotation(u1, u2, u3);
    }

    IrradianceProbe probe{};
    PathTraceStats stats{};
    math::Vec3 texelDirections[kIrradianceVisibilityTexels];
    for (u32 texel = 0; texel < kIrradianceVisibilityTexels; ++texel)
    {
        texelDirections[texel] = VisibilityTexelDirection(texel);
    }
    f32 texelWeight[kIrradianceVisibilityTexels] = {};
    f32 texelMean[kIrradianceVisibilityTexels] = {};
    f32 texelMeanSquared[kIrradianceVisibilityTexels] = {};
    f32 overallMean = 0.0f;
    f32 overallMeanSquared = 0.0f;
    u32 backfaces = 0;
    const f32 solidAngle = 4.0f * kPi / static_cast<f32>(rayCount);

    for (u32 rayIndex = 0; rayIndex < rayCount; ++rayIndex)
    {
        const math::Vec3 direction =
            math::Normalize(rotation.Apply(FibonacciSphere(rayIndex, rayCount)));
        RayDesc ray{};
        ray.origin = position;
        ray.direction = direction;
        ray.tMin = job.rayStart;
        RayHit first{};
        SampleStream stream(probeSequence, rayIndex);
        math::Vec3 radiance = TracePath(scene, ray, desc.path, stream, stats, &first);

        f32 distance = job.maxDistance;
        if (first.hit)
        {
            distance = std::min(first.distance, job.maxDistance);
            if (!first.frontFace)
            {
                // What a probe sees through a back face is the inside of a
                // solid. Counting it as black, at its true distance, is what
                // lets both the classification and the visibility test reject
                // the probe from the far side of the wall.
                ++backfaces;
                radiance = math::Vec3::Zero();
            }
        }

        f32 basis[9];
        EvaluateShBasis(direction, basis);
        for (u32 coefficient = 0; coefficient < 9; ++coefficient)
        {
            probe.radiance.coefficients[coefficient] +=
                radiance * (basis[coefficient] * solidAngle);
        }

        overallMean += distance;
        overallMeanSquared += distance * distance;
        for (u32 texel = 0; texel < kIrradianceVisibilityTexels; ++texel)
        {
            const f32 weight = VisibilityLobeWeight(math::Dot(direction, texelDirections[texel]));
            texelWeight[texel] += weight;
            texelMean[texel] += weight * distance;
            texelMeanSquared[texel] += weight * distance * distance;
        }
    }

    // A texel no ray came near -- possible only at very low ray counts --
    // takes the probe-wide moments rather than claiming open space.
    overallMean /= static_cast<f32>(rayCount);
    overallMeanSquared /= static_cast<f32>(rayCount);
    for (u32 texel = 0; texel < kIrradianceVisibilityTexels; ++texel)
    {
        if (texelWeight[texel] > 1.0e-6f)
        {
            probe.meanDistance[texel] = texelMean[texel] / texelWeight[texel];
            probe.meanDistanceSquared[texel] = texelMeanSquared[texel] / texelWeight[texel];
        }
        else
        {
            probe.meanDistance[texel] = overallMean;
            probe.meanDistanceSquared[texel] = overallMeanSquared;
        }
    }
    probe.backfaceFraction = static_cast<f32>(backfaces) / static_cast<f32>(rayCount);
    if (probe.backfaceFraction > desc.insideBackfaceThreshold)
    {
        probe.state = IrradianceProbeState::Inside;
        probe.radiance = SphericalHarmonicsL2{};
    }
    else
    {
        probe.state = IrradianceProbeState::Valid;
    }
    job.probes[index] = probe;
    job.stats[index] = stats;
}

}  // namespace

core::Status<core::Error> BakeIrradianceVolume(const GiScene& scene, const IrradianceBakeDesc& desc,
                                               IrradianceVolume& volume, IrradianceBakeStats* stats,
                                               jobs::JobSystem* jobSystem) noexcept
{
    using StatusType = core::Status<core::Error>;
    auto sceneStatus = ValidateGiScene(scene);
    if (sceneStatus.IsError())
    {
        return sceneStatus;
    }
    if (!volume.IsValid() || volume.GetAllocator() == nullptr)
    {
        return StatusType::Err(ErrorCode::InvalidArgument, "irradiance bake target is invalid");
    }
    if (desc.raysPerProbe < 16 || desc.raysPerProbe > 65536 || desc.path.maxBounces > 32 ||
        !std::isfinite(desc.insideBackfaceThreshold) || desc.insideBackfaceThreshold < 0.0f ||
        desc.insideBackfaceThreshold > 1.0f || !std::isfinite(desc.maxVisibilityDistance) ||
        desc.maxVisibilityDistance < 0.0f)
    {
        return StatusType::Err(ErrorCode::InvalidArgument,
                               "irradiance bake descriptor out of range");
    }

    const u32 count = volume.ProbeCount();
    memory::Allocator& allocator = *volume.GetAllocator();
    const usize probeBytes = AlignUp(sizeof(IrradianceProbe) * count, 16);
    const usize statsBytes = AlignUp(sizeof(PathTraceStats) * count, 16);
    void* scratch = allocator.Allocate(probeBytes + statsBytes, 16);
    if (scratch == nullptr)
    {
        return StatusType::Err(ErrorCode::Internal, "irradiance bake scratch allocation failed");
    }
    auto* probes = static_cast<IrradianceProbe*>(scratch);
    auto* probeStats =
        reinterpret_cast<PathTraceStats*>(static_cast<unsigned char*>(scratch) + probeBytes);
    for (u32 index = 0; index < count; ++index)
    {
        new (&probes[index]) IrradianceProbe{};
        new (&probeStats[index]) PathTraceStats{};
    }

    const math::Vec3 spacing = volume.ProbeSpacing();
    const f32 largestSpacing = std::max(spacing.x, std::max(spacing.y, spacing.z));
    BakeJob job{};
    job.scene = &scene;
    job.desc = &desc;
    job.volume = &volume;
    job.probes = probes;
    job.stats = probeStats;
    job.maxDistance =
        desc.maxVisibilityDistance > 0.0f ? desc.maxVisibilityDistance : 1.5f * largestSpacing;
    job.rayStart = kRayDefaultTMin * OffsetScale(scene);

    if (desc.useJobs && jobSystem != nullptr && jobSystem->IsRunning() && count > 1)
    {
        jobSystem->ParallelFor(count, 1, &BakeProbe, &job);
    }
    else
    {
        for (u32 index = 0; index < count; ++index)
        {
            BakeProbe(&job, index);
        }
    }

    IrradianceBakeStats totals{};
    totals.probeCount = count;
    for (u32 index = 0; index < count; ++index)
    {
        volume.SetProbe(index, probes[index]);
        totals.path.Add(probeStats[index]);
        if (probes[index].state == IrradianceProbeState::Valid)
        {
            ++totals.validProbes;
        }
        else if (probes[index].state == IrradianceProbeState::Inside)
        {
            ++totals.insideProbes;
        }
    }
    allocator.Deallocate(scratch);
    if (stats != nullptr)
    {
        *stats = totals;
    }
    return StatusType::Ok();
}

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------

IrradianceVolumeSample
SampleIrradianceVolumeDetailed(const IrradianceVolume& volume, const math::Vec3& position,
                               const math::Vec3& normal,
                               const IrradianceVolumeSampleDesc& desc) noexcept
{
    IrradianceVolumeSample result{};
    result.irradiance = FiniteNonNegative(desc.fallbackIrradiance);
    result.usedFallback = true;
    if (!volume.IsValid() || !IsFinite(position) || !IsFinite(normal) ||
        math::LengthSquared(normal) <= 0.0f)
    {
        return result;
    }

    const IrradianceVolumeDesc& grid = volume.Desc();
    const math::Vec3 spacing = volume.ProbeSpacing();
    const f32 smallestSpacing = std::min(spacing.x, std::min(spacing.y, spacing.z));
    const math::Vec3 n = math::Normalize(normal);
    const f32 bias = desc.normalBias > 0.0f && std::isfinite(desc.normalBias)
                         ? desc.normalBias
                         : 0.25f * smallestSpacing;
    const math::Vec3 point = position + n * bias;

    const math::Vec3 local = point - grid.boundsMin;
    const u32 counts[3] = {grid.probeCountX, grid.probeCountY, grid.probeCountZ};
    const f32 coordinates[3] = {local.x / spacing.x, local.y / spacing.y, local.z / spacing.z};
    u32 base[3] = {};
    f32 fraction[3] = {};
    for (u32 axis = 0; axis < 3; ++axis)
    {
        const f32 limit = static_cast<f32>(counts[axis] - 1);
        const f32 clamped = std::clamp(coordinates[axis], 0.0f, limit);
        const u32 cell = std::min(static_cast<u32>(clamped), counts[axis] - 2);
        base[axis] = cell;
        fraction[axis] = std::clamp(clamped - static_cast<f32>(cell), 0.0f, 1.0f);
    }

    // Variance floor for the Chebyshev bound, so a probe whose rays all hit
    // one flat wall at one distance does not produce a step function.
    const f32 varianceFloor = 1.0e-6f + (0.02f * smallestSpacing) * (0.02f * smallestSpacing);

    math::Vec3 weighted{};
    f32 totalWeight = 0.0f;
    math::Vec3 trilinearOnly{};
    f32 trilinearTotal = 0.0f;
    u32 contributing = 0;
    for (u32 corner = 0; corner < 8; ++corner)
    {
        const u32 offset[3] = {corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u};
        const u32 index =
            volume.ProbeIndex(base[0] + offset[0], base[1] + offset[1], base[2] + offset[2]);
        const IrradianceProbe& probe = volume.Probe(index);
        if (probe.state != IrradianceProbeState::Valid)
        {
            continue;
        }
        f32 trilinear = 1.0f;
        for (u32 axis = 0; axis < 3; ++axis)
        {
            trilinear *= offset[axis] != 0 ? fraction[axis] : 1.0f - fraction[axis];
        }
        const math::Vec3 irradiance =
            FiniteNonNegative(EvaluateSphericalHarmonicsIrradiance(probe.radiance, n));
        trilinearOnly += irradiance * trilinear;
        trilinearTotal += trilinear;

        const math::Vec3 probePosition = volume.ProbePosition(index);
        const math::Vec3 toProbe = probePosition - point;
        const f32 distance = math::Length(toProbe);
        const math::Vec3 direction = distance > 1.0e-6f ? toProbe * (1.0f / distance) : n;

        f32 weight = 1.0f;
        if (desc.useNormalWeighting)
        {
            // A smooth back-face term: full weight for a probe in front of the
            // surface, a small floor for one behind it, never a hard cut.
            const f32 facing = (math::Dot(direction, n) + 1.0f) * 0.5f;
            weight *= facing * facing + 0.2f;
        }
        if (desc.useVisibility)
        {
            f32 mean = 0.0f;
            f32 meanSquared = 0.0f;
            ReadVisibility(probe, -direction, mean, meanSquared);
            if (distance > mean)
            {
                const f32 variance = std::max(meanSquared - mean * mean, varianceFloor);
                const f32 excess = distance - mean;
                const f32 chebyshev = variance / (variance + excess * excess);
                // Cubing sharpens the one-tailed bound, which is loose by
                // construction, into something close to a visibility step.
                weight *= std::max(chebyshev * chebyshev * chebyshev, 0.0f);
            }
        }
        // Crush weights that are nearly zero rather than let several of them
        // add up to a visible contribution from probes that should not count.
        constexpr f32 kCrushThreshold = 0.2f;
        if (weight < kCrushThreshold)
        {
            weight *= weight * weight / (kCrushThreshold * kCrushThreshold);
        }
        weight *= trilinear;
        if (weight > 0.0f)
        {
            ++contributing;
        }
        weighted += irradiance * weight;
        totalWeight += weight;
    }

    result.contributingProbes = contributing;
    result.totalWeight = totalWeight;
    if (totalWeight > 1.0e-6f)
    {
        result.irradiance = weighted * (1.0f / totalWeight);
        result.usedFallback = false;
    }
    else if (trilinearTotal > 1.0e-6f)
    {
        // Every usable probe was rejected by visibility or facing. Falling back
        // to plain trilinear over the valid probes keeps the answer continuous
        // instead of snapping to the fallback colour.
        result.irradiance = trilinearOnly * (1.0f / trilinearTotal);
        result.usedFallback = false;
    }
    return result;
}

math::Vec3 SampleIrradianceVolume(const IrradianceVolume& volume, const math::Vec3& position,
                                  const math::Vec3& normal,
                                  const IrradianceVolumeSampleDesc& desc) noexcept
{
    return SampleIrradianceVolumeDetailed(volume, position, normal, desc).irradiance;
}

math::Vec3 EvaluateIrradianceVolume(const IrradianceVolumeView& view, const math::Vec3& position,
                                    const math::Vec3& normal) noexcept
{
    const f32 intensity =
        std::isfinite(view.intensity) && view.intensity > 0.0f ? view.intensity : 0.0f;
    if (view.volume == nullptr || !view.volume->IsValid())
    {
        return FiniteNonNegative(view.sampling.fallbackIrradiance) * intensity;
    }
    return SampleIrradianceVolume(*view.volume, position, normal, view.sampling) * intensity;
}

}  // namespace cyber::software
