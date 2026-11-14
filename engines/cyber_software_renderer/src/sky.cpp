#include <cyber/jobs/job_system.h>
#include <cyber/software/ibl.h>

#include <algorithm>
#include <cmath>

namespace cyber::software
{

namespace
{

constexpr f32 kPi = cyber::math::kPi;

f32 Saturate(f32 value) noexcept
{
    if (!std::isfinite(value))
    {
        return 0.0f;
    }
    return std::clamp(value, 0.0f, 1.0f);
}

bool IsFinite(const math::Vec3& value) noexcept
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

math::Vec3 Multiply(const math::Vec3& lhs, const math::Vec3& rhs) noexcept
{
    return math::Vec3{lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
}

math::Vec3 Exponential(const math::Vec3& value) noexcept
{
    return math::Vec3{std::exp(-value.x), std::exp(-value.y), std::exp(-value.z)};
}

// Validated and clamped copy of the configuration, so a caller cannot drive
// the march into a non-finite state.
struct SkyParameters
{
    math::Vec3 sunDirection{0.0f, -1.0f, 0.0f};
    math::Vec3 sunIrradiance{};
    math::Vec3 rayleighScattering{};
    math::Vec3 groundAlbedo{};
    f32 mieScattering = 0.0f;
    f32 mieExtinction = 0.0f;
    f32 mieAnisotropy = 0.0f;
    f32 rayleighScaleHeight = 8000.0f;
    f32 mieScaleHeight = 1200.0f;
    f32 planetRadius = 6360000.0f;
    f32 atmosphereRadius = 6420000.0f;
    f32 viewRadius = 6360500.0f;
    f32 sunCosineRadius = 1.0f;
    u32 viewSamples = 24;
    u32 lightSamples = 8;
    bool includeSunDisk = true;
    bool valid = false;
};

SkyParameters Validate(const ProceduralSkyConfig& config) noexcept
{
    SkyParameters parameters{};
    if (!IsFinite(config.sunDirection) ||
        cyber::math::LengthSquared(config.sunDirection) < 1.0e-12f)
    {
        return parameters;
    }
    if (!std::isfinite(config.planetRadiusMeters) || !std::isfinite(config.atmosphereRadiusMeters) ||
        config.planetRadiusMeters <= 0.0f || config.atmosphereRadiusMeters <= config.planetRadiusMeters)
    {
        return parameters;
    }
    if (!std::isfinite(config.viewAltitudeMeters) || config.viewAltitudeMeters < 0.0f)
    {
        return parameters;
    }
    if (!std::isfinite(config.rayleighScaleHeightMeters) ||
        config.rayleighScaleHeightMeters <= 0.0f || !std::isfinite(config.mieScaleHeightMeters) ||
        config.mieScaleHeightMeters <= 0.0f)
    {
        return parameters;
    }
    if (config.viewSampleCount < 2 || config.viewSampleCount > 128 ||
        config.lightSampleCount < 2 || config.lightSampleCount > 64)
    {
        return parameters;
    }

    parameters.sunDirection = cyber::math::Normalize(config.sunDirection);
    parameters.sunIrradiance = math::Vec3{std::max(0.0f, config.sunIrradiance.x),
                                          std::max(0.0f, config.sunIrradiance.y),
                                          std::max(0.0f, config.sunIrradiance.z)};
    parameters.rayleighScattering = math::Vec3{std::max(0.0f, config.rayleighScattering.x),
                                               std::max(0.0f, config.rayleighScattering.y),
                                               std::max(0.0f, config.rayleighScattering.z)};
    parameters.groundAlbedo = math::Vec3{Saturate(config.groundAlbedo.x),
                                         Saturate(config.groundAlbedo.y),
                                         Saturate(config.groundAlbedo.z)};
    parameters.mieScattering = std::max(0.0f, config.mieScattering);
    parameters.mieExtinction = std::max(parameters.mieScattering, config.mieExtinction);
    parameters.mieAnisotropy = std::clamp(config.mieAnisotropy, -0.9f, 0.9f);
    parameters.rayleighScaleHeight = config.rayleighScaleHeightMeters;
    parameters.mieScaleHeight = config.mieScaleHeightMeters;
    parameters.planetRadius = config.planetRadiusMeters;
    parameters.atmosphereRadius = config.atmosphereRadiusMeters;
    parameters.viewRadius =
        std::min(config.planetRadiusMeters + config.viewAltitudeMeters, config.atmosphereRadiusMeters);
    parameters.sunCosineRadius =
        std::cos(std::clamp(config.sunAngularRadiusRadians, 0.0f, 0.1f));
    parameters.viewSamples = config.viewSampleCount;
    parameters.lightSamples = config.lightSampleCount;
    parameters.includeSunDisk = config.includeSunDisk;
    parameters.valid = true;
    return parameters;
}

// Distance to the nearest positive intersection of a ray from `origin` with a
// sphere of `radius` centered at the origin of the frame. Returns -1 when the
// ray misses or only hits behind the start.
f32 IntersectSphere(const math::Vec3& origin, const math::Vec3& direction, f32 radius) noexcept
{
    const f32 b = cyber::math::Dot(origin, direction);
    const f32 c = cyber::math::LengthSquared(origin) - radius * radius;
    const f32 discriminant = b * b - c;
    if (discriminant < 0.0f)
    {
        return -1.0f;
    }
    const f32 root = std::sqrt(discriminant);
    const f32 nearHit = -b - root;
    if (nearHit > 0.0f)
    {
        return nearHit;
    }
    const f32 farHit = -b + root;
    return farHit > 0.0f ? farHit : -1.0f;
}

f32 Altitude(const SkyParameters& parameters, const math::Vec3& position) noexcept
{
    return std::max(0.0f, cyber::math::Length(position) - parameters.planetRadius);
}

// Optical depth from `position` toward the sun, or a negative value when the
// planet blocks the path.
bool SunOpticalDepth(const SkyParameters& parameters, const math::Vec3& position,
                     const math::Vec3& toSun, f32& rayleighDepth, f32& mieDepth) noexcept
{
    rayleighDepth = 0.0f;
    mieDepth = 0.0f;
    if (IntersectSphere(position, toSun, parameters.planetRadius) > 0.0f)
    {
        return false;
    }
    const f32 top = IntersectSphere(position, toSun, parameters.atmosphereRadius);
    if (top <= 0.0f)
    {
        return true;
    }
    const f32 step = top / static_cast<f32>(parameters.lightSamples);
    for (u32 index = 0; index < parameters.lightSamples; ++index)
    {
        const math::Vec3 sample =
            position + toSun * (step * (static_cast<f32>(index) + 0.5f));
        const f32 altitude = Altitude(parameters, sample);
        rayleighDepth += std::exp(-altitude / parameters.rayleighScaleHeight) * step;
        mieDepth += std::exp(-altitude / parameters.mieScaleHeight) * step;
    }
    return true;
}

math::Vec3 Transmittance(const SkyParameters& parameters, f32 rayleighDepth,
                         f32 mieDepth) noexcept
{
    const math::Vec3 extinction =
        parameters.rayleighScattering * rayleighDepth +
        math::Vec3{parameters.mieExtinction, parameters.mieExtinction, parameters.mieExtinction} *
            mieDepth;
    return Exponential(extinction);
}

f32 RayleighPhase(f32 cosine) noexcept
{
    return 3.0f / (16.0f * kPi) * (1.0f + cosine * cosine);
}

f32 HenyeyGreensteinPhase(f32 cosine, f32 anisotropy) noexcept
{
    const f32 g = anisotropy;
    const f32 denominator = 1.0f + g * g - 2.0f * g * cosine;
    if (denominator <= 0.0f)
    {
        return 0.0f;
    }
    return (1.0f - g * g) / (4.0f * kPi * denominator * std::sqrt(denominator));
}

math::Vec3 Scatter(const SkyParameters& parameters, const math::Vec3& viewDirection) noexcept
{
    const math::Vec3 origin{0.0f, parameters.viewRadius, 0.0f};
    const math::Vec3 toSun = -parameters.sunDirection;

    const f32 atmosphereDistance =
        IntersectSphere(origin, viewDirection, parameters.atmosphereRadius);
    if (atmosphereDistance <= 0.0f)
    {
        return math::Vec3::Zero();
    }
    const f32 groundDistance = IntersectSphere(origin, viewDirection, parameters.planetRadius);
    const bool hitsGround = groundDistance > 0.0f;
    const f32 marchDistance = hitsGround ? groundDistance : atmosphereDistance;

    const f32 cosine = cyber::math::Dot(viewDirection, toSun);
    const f32 rayleighPhase = RayleighPhase(cosine);
    const f32 miePhase = HenyeyGreensteinPhase(cosine, parameters.mieAnisotropy);

    const f32 step = marchDistance / static_cast<f32>(parameters.viewSamples);
    f32 viewRayleighDepth = 0.0f;
    f32 viewMieDepth = 0.0f;
    math::Vec3 radiance{};

    for (u32 index = 0; index < parameters.viewSamples; ++index)
    {
        const math::Vec3 sample =
            origin + viewDirection * (step * (static_cast<f32>(index) + 0.5f));
        const f32 altitude = Altitude(parameters, sample);
        const f32 rayleighDensity = std::exp(-altitude / parameters.rayleighScaleHeight);
        const f32 mieDensity = std::exp(-altitude / parameters.mieScaleHeight);
        viewRayleighDepth += rayleighDensity * step;
        viewMieDepth += mieDensity * step;

        f32 sunRayleighDepth = 0.0f;
        f32 sunMieDepth = 0.0f;
        if (!SunOpticalDepth(parameters, sample, toSun, sunRayleighDepth, sunMieDepth))
        {
            continue;
        }
        const math::Vec3 transmittance =
            Multiply(Transmittance(parameters, viewRayleighDepth, viewMieDepth),
                     Transmittance(parameters, sunRayleighDepth, sunMieDepth));
        const math::Vec3 inScatter =
            parameters.rayleighScattering * (rayleighDensity * rayleighPhase) +
            math::Vec3{parameters.mieScattering, parameters.mieScattering,
                       parameters.mieScattering} *
                (mieDensity * miePhase);
        radiance = radiance + Multiply(Multiply(inScatter, transmittance),
                                       parameters.sunIrradiance) *
                                  step;
    }

    if (hitsGround)
    {
        const math::Vec3 groundPoint = origin + viewDirection * groundDistance;
        const math::Vec3 groundNormal = cyber::math::Normalize(groundPoint);
        f32 sunRayleighDepth = 0.0f;
        f32 sunMieDepth = 0.0f;
        if (SunOpticalDepth(parameters, groundPoint, toSun, sunRayleighDepth, sunMieDepth))
        {
            const f32 ndotl = std::max(0.0f, cyber::math::Dot(groundNormal, toSun));
            const math::Vec3 sunAtGround =
                Multiply(Transmittance(parameters, sunRayleighDepth, sunMieDepth),
                         parameters.sunIrradiance);
            const math::Vec3 groundRadiance =
                Multiply(parameters.groundAlbedo, sunAtGround) * (ndotl / kPi);
            radiance =
                radiance + Multiply(groundRadiance,
                                    Transmittance(parameters, viewRayleighDepth, viewMieDepth));
        }
    }
    else if (parameters.includeSunDisk && cosine >= parameters.sunCosineRadius)
    {
        // Spread the sun's irradiance over its own solid angle so the disk
        // carries the correct radiance, which is what makes a prefiltered
        // highlight match the matching directional light.
        const f32 solidAngle = 2.0f * kPi * (1.0f - parameters.sunCosineRadius);
        if (solidAngle > 0.0f)
        {
            const math::Vec3 disk =
                Multiply(parameters.sunIrradiance,
                         Transmittance(parameters, viewRayleighDepth, viewMieDepth)) *
                (1.0f / solidAngle);
            radiance = radiance + disk;
        }
    }

    return math::Vec3{std::max(0.0f, radiance.x), std::max(0.0f, radiance.y),
                      std::max(0.0f, radiance.z)};
}

struct SkyContext
{
    const SkyParameters* parameters = nullptr;
    Cubemap* output = nullptr;
    i32 size = 0;
};

void RenderSkyRow(void* userData, u32 index) noexcept
{
    auto* context = static_cast<SkyContext*>(userData);
    const i32 size = context->size;
    const u32 face = index / static_cast<u32>(size);
    const i32 y = static_cast<i32>(index % static_cast<u32>(size));
    const f32 inverseSize = 1.0f / static_cast<f32>(size);
    for (i32 x = 0; x < size; ++x)
    {
        const f32 u = (static_cast<f32>(x) + 0.5f) * inverseSize;
        const f32 v = (static_cast<f32>(y) + 0.5f) * inverseSize;
        const math::Vec3 direction = CubeDirectionFromFaceUv(face, u, v);
        const math::Vec3 radiance = Scatter(*context->parameters, direction);
        context->output->SetTexel(face, 0, x, y,
                                  math::Color{radiance.x, radiance.y, radiance.z, 1.0f});
    }
}

}  // namespace

math::Vec3 EvaluateProceduralSky(const ProceduralSkyConfig& config,
                                 const math::Vec3& viewDirection) noexcept
{
    const SkyParameters parameters = Validate(config);
    if (!parameters.valid || !IsFinite(viewDirection) ||
        cyber::math::LengthSquared(viewDirection) < 1.0e-12f)
    {
        return math::Vec3::Zero();
    }
    return Scatter(parameters, cyber::math::Normalize(viewDirection));
}

math::Vec3 EvaluateSunTransmittance(const ProceduralSkyConfig& config) noexcept
{
    const SkyParameters parameters = Validate(config);
    if (!parameters.valid)
    {
        return math::Vec3::Zero();
    }
    const math::Vec3 origin{0.0f, parameters.viewRadius, 0.0f};
    const math::Vec3 toSun = -parameters.sunDirection;
    f32 rayleighDepth = 0.0f;
    f32 mieDepth = 0.0f;
    if (!SunOpticalDepth(parameters, origin, toSun, rayleighDepth, mieDepth))
    {
        return math::Vec3::Zero();
    }
    return Multiply(parameters.sunIrradiance,
                    Transmittance(parameters, rayleighDepth, mieDepth));
}

core::Status<core::Error> RenderProceduralSkyCubemap(const ProceduralSkyConfig& config,
                                                     Cubemap& output,
                                                     jobs::JobSystem* jobSystem) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (!output.IsValid())
    {
        return StatusType::Err(core::ErrorCode::InvalidArgument, "sky output invalid");
    }
    const SkyParameters parameters = Validate(config);
    if (!parameters.valid)
    {
        return StatusType::Err(core::ErrorCode::InvalidArgument, "sky configuration invalid");
    }

    SkyContext context{};
    context.parameters = &parameters;
    context.output = &output;
    context.size = output.FaceSize();

    const u32 rows = static_cast<u32>(context.size) * kCubeFaceCount;
    if (jobSystem != nullptr && jobSystem->IsRunning() && rows > 1)
    {
        jobSystem->ParallelFor(rows, 1, &RenderSkyRow, &context);
    }
    else
    {
        for (u32 row = 0; row < rows; ++row)
        {
            RenderSkyRow(&context, row);
        }
    }
    output.GenerateMips();
    return StatusType::Ok();
}

}  // namespace cyber::software
