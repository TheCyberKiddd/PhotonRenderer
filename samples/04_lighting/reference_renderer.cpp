#include "reference_renderer.h"

#include <cyber/math/constants.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cyber::software
{

namespace
{

constexpr core::f32 kPi = cyber::math::kPi;
constexpr core::f32 kInversePi = 1.0f / kPi;
constexpr core::f32 kMinimumRoughness = 0.045f;
constexpr core::f32 kRayEpsilon = 1.0e-3f;

struct Ray
{
    math::Vec3 origin;
    math::Vec3 direction;
};

struct Hit
{
    core::f32 distance = std::numeric_limits<core::f32>::max();
    core::u32 column = 0;
    core::u32 row = 0;
    math::Vec3 position;
    math::Vec3 normal;
    math::Vec3 tangent;
    math::Vec2 uv;
};

struct Random
{
    core::u64 state;

    [[nodiscard]] core::u32 NextU32() noexcept
    {
        core::u64 value = (state += 0x9E3779B97F4A7C15ull);
        value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
        value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
        value ^= value >> 31u;
        return static_cast<core::u32>(value >> 32u);
    }

    [[nodiscard]] core::f32 NextFloat() noexcept
    {
        return static_cast<core::f32>(NextU32() >> 8u) * (1.0f / 16777216.0f);
    }
};

[[nodiscard]] core::f32 ClampUnit(core::f32 value) noexcept
{
    return std::isfinite(value) ? std::clamp(value, 0.0f, 1.0f) : 0.0f;
}

[[nodiscard]] math::Vec3 ClampRadiance(const math::Vec3& value) noexcept
{
    const auto channel = [](core::f32 component) noexcept
    { return std::isfinite(component) ? std::max(component, 0.0f) : 0.0f; };
    return math::Vec3{channel(value.x), channel(value.y), channel(value.z)};
}

[[nodiscard]] math::Vec3 Hadamard(const math::Vec3& lhs, const math::Vec3& rhs) noexcept
{
    return math::Vec3{lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
}

[[nodiscard]] math::Vec3 ToVec3(const math::Color& color) noexcept
{
    return ClampRadiance(math::Vec3{color.r, color.g, color.b});
}

[[nodiscard]] math::Vec3 SphereCenter(const SphereGridPathTraceDesc& desc, core::u32 column,
                                      core::u32 row) noexcept
{
    const core::f32 half = static_cast<core::f32>(desc.gridSize - 1) * 0.5f;
    return math::Vec3{(static_cast<core::f32>(column) - half) * desc.sphereSpacing,
                      (half - static_cast<core::f32>(row)) * desc.sphereSpacing, 0.0f};
}

[[nodiscard]] bool IntersectScene(const Ray& ray, const SphereGridPathTraceDesc& desc,
                                  Hit& closest) noexcept
{
    bool found = false;
    for (core::u32 row = 0; row < desc.gridSize; ++row)
    {
        for (core::u32 column = 0; column < desc.gridSize; ++column)
        {
            const math::Vec3 center = SphereCenter(desc, column, row);
            const math::Vec3 originToCenter = ray.origin - center;
            const core::f32 halfB = cyber::math::Dot(originToCenter, ray.direction);
            const core::f32 c = cyber::math::Dot(originToCenter, originToCenter) -
                                desc.sphereRadius * desc.sphereRadius;
            const core::f32 discriminant = halfB * halfB - c;
            if (discriminant < 0.0f)
            {
                continue;
            }

            const core::f32 root = std::sqrt(discriminant);
            core::f32 distance = -halfB - root;
            if (distance <= kRayEpsilon)
            {
                distance = -halfB + root;
            }
            if (distance <= kRayEpsilon || distance >= closest.distance)
            {
                continue;
            }

            found = true;
            closest.distance = distance;
            closest.column = column;
            closest.row = row;
            closest.position = ray.origin + ray.direction * distance;
            closest.normal = cyber::math::Normalize(closest.position - center);

            core::f32 phi = std::atan2(closest.normal.z, closest.normal.x);
            if (phi < 0.0f)
            {
                phi += 2.0f * kPi;
            }
            const core::f32 theta = std::acos(std::clamp(closest.normal.y, -1.0f, 1.0f));
            closest.uv = math::Vec2{phi / (2.0f * kPi), theta / kPi};
            closest.tangent =
                cyber::math::Normalize(math::Vec3{-std::sin(phi), 0.0f, std::cos(phi)});
            if (cyber::math::LengthSquared(closest.tangent) < 1.0e-12f)
            {
                closest.tangent = math::Vec3{1.0f, 0.0f, 0.0f};
            }
        }
    }
    return found;
}

[[nodiscard]] math::Vec3 SurfaceNormal(const Hit& hit, const PbrLightingData& lighting) noexcept
{
    if (lighting.normalTexture == nullptr)
    {
        return hit.normal;
    }
    const math::Color sample = SampleTextureLevel(*lighting.normalTexture, hit.uv.x, hit.uv.y,
                                                  lighting.mipLevel, lighting.sampler);
    const math::Vec3 bitangent = cyber::math::Cross(hit.normal, hit.tangent);
    const math::Vec3 local{(sample.r * 2.0f - 1.0f) * lighting.material.normalScale,
                           (sample.g * 2.0f - 1.0f) * lighting.material.normalScale,
                           sample.b * 2.0f - 1.0f};
    const math::Vec3 mapped =
        cyber::math::Normalize(hit.tangent * local.x + bitangent * local.y + hit.normal * local.z);
    return cyber::math::LengthSquared(mapped) > 1.0e-12f ? mapped : hit.normal;
}

[[nodiscard]] math::Vec3 SurfaceColor(const Hit& hit, const PbrLightingData& lighting) noexcept
{
    math::Vec3 baseColor = ToVec3(lighting.material.baseColor);
    if (lighting.baseColorTexture != nullptr)
    {
        const math::Color textureColor = SampleTextureLevel(
            *lighting.baseColorTexture, hit.uv.x, hit.uv.y, lighting.mipLevel, lighting.sampler);
        baseColor = Hadamard(baseColor, ToVec3(textureColor));
    }
    return baseColor;
}

[[nodiscard]] core::f32 NormalDistribution(core::f32 nDotH, core::f32 roughness) noexcept
{
    const core::f32 alpha = std::max(roughness * roughness, kMinimumRoughness * kMinimumRoughness);
    const core::f32 alphaSquared = alpha * alpha;
    const core::f32 denominator = nDotH * nDotH * (alphaSquared - 1.0f) + 1.0f;
    return alphaSquared / (kPi * denominator * denominator);
}

[[nodiscard]] math::Vec3 EvaluateReferenceBrdf(const math::Vec3& normal, const math::Vec3& view,
                                               const math::Vec3& light,
                                               const PbrMaterial& material) noexcept
{
    const core::f32 nDotV = ClampUnit(cyber::math::Dot(normal, view));
    const core::f32 nDotL = ClampUnit(cyber::math::Dot(normal, light));
    if (nDotV <= 0.0f || nDotL <= 0.0f || !std::isfinite(material.metallic) ||
        !std::isfinite(material.roughness))
    {
        return math::Vec3::Zero();
    }

    const core::f32 metallic = ClampUnit(material.metallic);
    const core::f32 roughness = std::clamp(material.roughness, kMinimumRoughness, 1.0f);
    const math::Vec3 halfVector = cyber::math::Normalize(view + light);
    const core::f32 nDotH = ClampUnit(cyber::math::Dot(normal, halfVector));
    const core::f32 vDotH = ClampUnit(cyber::math::Dot(view, halfVector));
    const core::f32 alpha = roughness * roughness;
    const core::f32 alphaSquared = alpha * alpha;
    const auto masking = [alphaSquared](core::f32 cosine) noexcept
    {
        const core::f32 root = std::sqrt(alphaSquared + (1.0f - alphaSquared) * cosine * cosine);
        const core::f32 denominator = cosine + root;
        return denominator > 0.0f ? 2.0f * cosine / denominator : 0.0f;
    };

    const math::Vec3 albedo = ToVec3(material.baseColor);
    const math::Vec3 f0 = math::Lerp(math::Vec3{0.04f}, albedo, metallic);
    const core::f32 fresnelFactor = std::pow(1.0f - vDotH, 5.0f);
    const math::Vec3 fresnel = f0 + (math::Vec3::One() - f0) * fresnelFactor;
    const core::f32 distribution = NormalDistribution(nDotH, roughness);
    const core::f32 geometry = masking(nDotV) * masking(nDotL);
    const math::Vec3 specular =
        fresnel * (distribution * geometry / std::max(4.0f * nDotV * nDotL, 1.0e-6f));
    const math::Vec3 diffuse =
        Hadamard(math::Vec3::One() - fresnel, albedo) * ((1.0f - metallic) * kInversePi);
    return diffuse + specular;
}

[[nodiscard]] core::f32 RangeFade(core::f32 distance, core::f32 range) noexcept
{
    if (!std::isfinite(distance) || !std::isfinite(range) || range <= 0.0f || distance >= range)
    {
        return 0.0f;
    }
    const core::f32 ratio = distance / range;
    const core::f32 remaining = 1.0f - ratio * ratio;
    return remaining * remaining;
}

[[nodiscard]] bool Occluded(const Hit& hit, const math::Vec3& normal, const math::Vec3& direction,
                            core::f32 maximumDistance, const SphereGridPathTraceDesc& desc) noexcept
{
    Hit blocker{};
    const Ray shadowRay{hit.position + normal * kRayEpsilon, direction};
    return IntersectScene(shadowRay, desc, blocker) &&
           blocker.distance < maximumDistance - kRayEpsilon;
}

[[nodiscard]] math::Vec3 EstimateDirect(const Hit& hit, const math::Vec3& normal,
                                        const math::Vec3& view, const PbrLightingData& lighting,
                                        const PbrMaterial& material,
                                        const SphereGridPathTraceDesc& desc) noexcept
{
    math::Vec3 result = math::Vec3::Zero();
    const auto addLight = [&](const math::Vec3& direction, const math::Vec3& radiance) noexcept
    {
        const core::f32 nDotL = ClampUnit(cyber::math::Dot(normal, direction));
        if (nDotL <= 0.0f ||
            Occluded(hit, normal, direction, std::numeric_limits<core::f32>::max(), desc))
        {
            return;
        }
        result += Hadamard(EvaluateReferenceBrdf(normal, view, direction, material),
                           ClampRadiance(radiance)) *
                  nDotL;
    };

    for (core::u32 index = 0; index < lighting.directionalCount && index < kMaxPbrDirectionalLights;
         ++index)
    {
        const PbrDirectionalLight& light = lighting.directional[index];
        addLight(cyber::math::Normalize(-light.direction), light.radiance);
    }

    for (core::u32 index = 0; index < lighting.pointCount && index < kMaxPbrPointLights; ++index)
    {
        const PbrPointLight& light = lighting.points[index];
        const math::Vec3 offset = light.position - hit.position;
        const core::f32 distanceSquared = cyber::math::LengthSquared(offset);
        if (!(distanceSquared > 1.0e-8f) || !std::isfinite(distanceSquared))
        {
            continue;
        }
        const core::f32 distance = std::sqrt(distanceSquared);
        const core::f32 fade = RangeFade(distance, light.range);
        const math::Vec3 direction = offset / distance;
        if (fade > 0.0f && !Occluded(hit, normal, direction, distance, desc))
        {
            const core::f32 nDotL = ClampUnit(cyber::math::Dot(normal, direction));
            result += Hadamard(EvaluateReferenceBrdf(normal, view, direction, material),
                               ClampRadiance(light.intensity)) *
                      (nDotL * fade / std::max(distanceSquared, 1.0e-4f));
        }
    }

    for (core::u32 index = 0; index < lighting.spotCount && index < kMaxPbrSpotLights; ++index)
    {
        const PbrSpotLight& light = lighting.spots[index];
        const math::Vec3 offset = light.position - hit.position;
        const core::f32 distanceSquared = cyber::math::LengthSquared(offset);
        if (!(distanceSquared > 1.0e-8f) || !std::isfinite(distanceSquared))
        {
            continue;
        }
        const core::f32 distance = std::sqrt(distanceSquared);
        const core::f32 fade = RangeFade(distance, light.range);
        const math::Vec3 direction = offset / distance;
        const core::f32 inner = std::cos(std::clamp(light.innerConeRadians, 0.0f, kPi * 0.5f));
        const core::f32 outer = std::cos(std::clamp(light.outerConeRadians, 0.0f, kPi * 0.5f));
        const core::f32 cone =
            cyber::math::Dot(cyber::math::Normalize(light.direction), -direction);
        core::f32 coneFade = 0.0f;
        if (inner > outer)
        {
            const core::f32 t = ClampUnit((cone - outer) / (inner - outer));
            coneFade = t * t * (3.0f - 2.0f * t);
        }
        else if (cone >= inner)
        {
            coneFade = 1.0f;
        }
        if (fade * coneFade > 0.0f && !Occluded(hit, normal, direction, distance, desc))
        {
            const core::f32 nDotL = ClampUnit(cyber::math::Dot(normal, direction));
            result += Hadamard(EvaluateReferenceBrdf(normal, view, direction, material),
                               ClampRadiance(light.intensity)) *
                      (nDotL * fade * coneFade / std::max(distanceSquared, 1.0e-4f));
        }
    }
    return result;
}

[[nodiscard]] math::Vec3 CosineDirection(const math::Vec3& normal, Random& random) noexcept
{
    const core::f32 radius = std::sqrt(random.NextFloat());
    const core::f32 angle = 2.0f * kPi * random.NextFloat();
    const math::Vec3 axis =
        std::fabs(normal.z) < 0.999f ? math::Vec3{0.0f, 0.0f, 1.0f} : math::Vec3{0.0f, 1.0f, 0.0f};
    const math::Vec3 tangent = cyber::math::Normalize(cyber::math::Cross(axis, normal));
    const math::Vec3 bitangent = cyber::math::Cross(normal, tangent);
    const core::f32 z = std::sqrt(std::max(0.0f, 1.0f - radius * radius));
    return cyber::math::Normalize(tangent * (radius * std::cos(angle)) +
                                  bitangent * (radius * std::sin(angle)) + normal * z);
}

[[nodiscard]] math::Vec3 SampleGgxDirection(const math::Vec3& normal, const math::Vec3& view,
                                            core::f32 roughness, Random& random) noexcept
{
    const core::f32 alpha = std::max(roughness * roughness, kMinimumRoughness * kMinimumRoughness);
    const core::f32 alphaSquared = alpha * alpha;
    const core::f32 u = random.NextFloat();
    const core::f32 phi = 2.0f * kPi * random.NextFloat();
    const core::f32 cosine = std::sqrt((1.0f - u) / (1.0f + (alphaSquared - 1.0f) * u));
    const core::f32 sine = std::sqrt(std::max(0.0f, 1.0f - cosine * cosine));
    const math::Vec3 axis =
        std::fabs(normal.z) < 0.999f ? math::Vec3{0.0f, 0.0f, 1.0f} : math::Vec3{0.0f, 1.0f, 0.0f};
    const math::Vec3 tangent = cyber::math::Normalize(cyber::math::Cross(axis, normal));
    const math::Vec3 bitangent = cyber::math::Cross(normal, tangent);
    const math::Vec3 halfVector = cyber::math::Normalize(
        tangent * (sine * std::cos(phi)) + bitangent * (sine * std::sin(phi)) + normal * cosine);
    return cyber::math::Normalize(cyber::math::Reflect(-view, halfVector));
}

[[nodiscard]] core::f32 SpecularProbability(const PbrMaterial& material) noexcept
{
    const core::f32 metallic = ClampUnit(material.metallic);
    return std::clamp(0.15f + 0.7f * metallic, 0.1f, 0.9f);
}

[[nodiscard]] core::f32 DirectionPdf(const math::Vec3& normal, const math::Vec3& view,
                                     const math::Vec3& direction,
                                     const PbrMaterial& material) noexcept
{
    const core::f32 nDotL = ClampUnit(cyber::math::Dot(normal, direction));
    if (nDotL <= 0.0f)
    {
        return 0.0f;
    }
    const core::f32 diffusePdf = nDotL * kInversePi;
    const math::Vec3 halfVector = cyber::math::Normalize(view + direction);
    const core::f32 nDotH = ClampUnit(cyber::math::Dot(normal, halfVector));
    const core::f32 vDotH = ClampUnit(cyber::math::Dot(view, halfVector));
    const core::f32 specularPdf =
        vDotH > 1.0e-7f ? NormalDistribution(nDotH, material.roughness) * nDotH / (4.0f * vDotH)
                        : 0.0f;
    const core::f32 specularProbability = SpecularProbability(material);
    return (1.0f - specularProbability) * diffusePdf + specularProbability * specularPdf;
}

[[nodiscard]] math::Vec3 SampleDirection(const math::Vec3& normal, const math::Vec3& view,
                                         const PbrMaterial& material, Random& random) noexcept
{
    if (random.NextFloat() < SpecularProbability(material))
    {
        return SampleGgxDirection(normal, view,
                                  std::clamp(material.roughness, kMinimumRoughness, 1.0f), random);
    }
    return CosineDirection(normal, random);
}

[[nodiscard]] math::Vec3 TracePath(Ray ray, const SphereGridPathTraceDesc& desc,
                                   memory::Span<const PbrLightingData> materials, Random& random,
                                   bool& primaryHit) noexcept
{
    math::Vec3 radiance = math::Vec3::Zero();
    math::Vec3 throughput = math::Vec3::One();
    math::Vec3 environment{0.012f, 0.016f, 0.025f};
    for (core::u32 bounce = 0; bounce <= desc.maxBounces; ++bounce)
    {
        Hit hit{};
        if (!IntersectScene(ray, desc, hit))
        {
            radiance += Hadamard(throughput, environment);
            break;
        }

        const PbrLightingData& lighting =
            materials[static_cast<core::usize>(hit.row) * desc.gridSize + hit.column];
        if (bounce == 0)
        {
            primaryHit = true;
        }
        const math::Vec3 albedo = SurfaceColor(hit, lighting);
        PbrMaterial material = lighting.material;
        material.baseColor = math::Color{albedo.x, albedo.y, albedo.z, 1.0f};
        const math::Vec3 normal = SurfaceNormal(hit, lighting);
        const math::Vec3 view = cyber::math::Normalize(desc.eyePosition - hit.position);
        environment = ClampRadiance(lighting.ambientIrradiance) * kInversePi;
        radiance +=
            Hadamard(throughput, EstimateDirect(hit, normal, view, lighting, material, desc));

        if (bounce == desc.maxBounces)
        {
            break;
        }

        const math::Vec3 direction = SampleDirection(normal, view, material, random);
        const core::f32 nDotL = ClampUnit(cyber::math::Dot(normal, direction));
        const core::f32 pdf = DirectionPdf(normal, view, direction, material);
        if (nDotL <= 0.0f || pdf <= 1.0e-8f)
        {
            break;
        }
        const math::Vec3 brdf = EvaluateReferenceBrdf(normal, view, direction, material);
        throughput = Hadamard(throughput, brdf) * (nDotL / pdf);
        throughput = ClampRadiance(throughput);

        if (bounce >= 2)
        {
            const core::f32 survival = std::clamp(
                std::max(throughput.x, std::max(throughput.y, throughput.z)), 0.05f, 0.95f);
            if (random.NextFloat() >= survival)
            {
                break;
            }
            throughput /= survival;
        }
        ray = Ray{hit.position + normal * kRayEpsilon, direction};
    }
    return radiance;
}

}  // namespace

bool RenderSphereGridPathTrace(Framebuffer& target, memory::Span<const PbrLightingData> materials,
                               const SphereGridPathTraceDesc& desc) noexcept
{
    if (!target.IsValid() || desc.gridSize == 0 || desc.gridSize > 256 ||
        materials.Size() != static_cast<core::usize>(desc.gridSize) * desc.gridSize ||
        desc.samplesPerPixel == 0 || desc.samplesPerPixel > 4096 || desc.maxBounces == 0 ||
        desc.maxBounces > 32 || !std::isfinite(desc.sphereRadius) || desc.sphereRadius <= 0.0f ||
        !std::isfinite(desc.sphereSpacing) || desc.sphereSpacing <= 2.0f * desc.sphereRadius ||
        !std::isfinite(desc.verticalFovRadians) || desc.verticalFovRadians <= 0.0f ||
        desc.verticalFovRadians >= kPi || !std::isfinite(desc.eyePosition.x) ||
        !std::isfinite(desc.eyePosition.y) || !std::isfinite(desc.eyePosition.z))
    {
        return false;
    }

    const math::Vec3 forward = cyber::math::Normalize(-desc.eyePosition);
    const math::Vec3 right =
        cyber::math::Normalize(cyber::math::Cross(forward, math::Vec3{0.0f, 1.0f, 0.0f}));
    if (cyber::math::LengthSquared(forward) < 1.0e-12f ||
        cyber::math::LengthSquared(right) < 1.0e-12f)
    {
        return false;
    }
    const math::Vec3 up = cyber::math::Cross(right, forward);
    const core::f32 aspect =
        static_cast<core::f32>(target.Width()) / static_cast<core::f32>(target.Height());
    const core::f32 halfHeight = std::tan(desc.verticalFovRadians * 0.5f);
    target.Clear(math::Color{0.012f, 0.016f, 0.025f, 0.0f});

    for (core::i32 y = 0; y < target.Height(); ++y)
    {
        for (core::i32 x = 0; x < target.Width(); ++x)
        {
            const core::f32 ndcX =
                (2.0f * (static_cast<core::f32>(x) + 0.5f) / target.Width() - 1.0f) * aspect;
            const core::f32 ndcY =
                1.0f - 2.0f * (static_cast<core::f32>(y) + 0.5f) / target.Height();
            const math::Vec3 direction = cyber::math::Normalize(
                forward + right * (ndcX * halfHeight) + up * (ndcY * halfHeight));
            math::Vec3 sum = math::Vec3::Zero();
            bool primaryHit = false;
            for (core::u32 sample = 0; sample < desc.samplesPerPixel; ++sample)
            {
                const core::u64 pixelKey =
                    static_cast<core::u64>(y) * static_cast<core::u64>(target.Width()) +
                    static_cast<core::u64>(x);
                Random random{desc.seed ^ (pixelKey * 0xD6E8FEB86659FD93ull) ^
                              (static_cast<core::u64>(sample) * 0xA5A3564E27F8862Bull)};
                bool sampleHit = false;
                sum +=
                    TracePath(Ray{desc.eyePosition, direction}, desc, materials, random, sampleHit);
                primaryHit = primaryHit || sampleHit;
            }
            sum /= static_cast<core::f32>(desc.samplesPerPixel);
            target.SetPixel(x, y, math::Color{sum.x, sum.y, sum.z, primaryHit ? 1.0f : 0.0f});
        }
    }
    return true;
}

}  // namespace cyber::software
