#include <cyber/math/constants.h>
#include <cyber/software/global_illumination.h>
#include <cyber/software/pbr.h>

#include <algorithm>
#include <cmath>

namespace cyber::software
{

namespace
{

constexpr f32 kMinimumRoughness = 0.045f;
constexpr f32 kDielectricF0 = 0.04f;
constexpr f32 kInversePi = 1.0f / cyber::math::kPi;

f32 Saturate(f32 value) noexcept
{
    return math::IsFinite(value) ? std::clamp(value, 0.0f, 1.0f) : 0.0f;
}

f32 FiniteNonNegative(f32 value) noexcept
{
    return math::IsFinite(value) && value > 0.0f ? value : 0.0f;
}

bool IsFinite(const math::Vec3& value) noexcept
{
    return math::IsFinite(value.x) && math::IsFinite(value.y) && math::IsFinite(value.z);
}

math::Vec3 FiniteNonNegative(const math::Vec3& value) noexcept
{
    return math::Vec3{FiniteNonNegative(value.x), FiniteNonNegative(value.y),
                      FiniteNonNegative(value.z)};
}

f32 SmoothRangeFade(f32 distance, f32 range) noexcept
{
    if (!math::IsFinite(distance) || !math::IsFinite(range) || distance < 0.0f || range <= 0.0f ||
        distance >= range)
    {
        return 0.0f;
    }
    const f32 ratio = distance / range;
    const f32 fade = 1.0f - ratio * ratio;
    return fade * fade;
}

f32 SchlickGeometry(f32 nDotX, f32 alpha) noexcept
{
    const f32 alphaSquared = alpha * alpha;
    const f32 denominator = nDotX + std::sqrt(alphaSquared + (1.0f - alphaSquared) * nDotX * nDotX);
    return denominator > 0.0f ? (2.0f * nDotX) / denominator : 0.0f;
}

math::Vec3 FresnelSchlick(f32 vDotH, const math::Vec3& f0) noexcept
{
    // (1 - v.h)^5 by multiplication: std::pow is a library call per light per pixel.
    const f32 base = 1.0f - Saturate(vDotH);
    const f32 base2 = base * base;
    const f32 factor = base2 * base2 * base;
    return f0 + (math::Vec3::One() - f0) * factor;
}

math::Vec3 Multiply(const math::Vec3& lhs, const math::Vec3& rhs) noexcept
{
    return math::Vec3{lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
}

math::Vec3 ToVec3(const math::Color& color) noexcept
{
    return math::Vec3{Saturate(FiniteNonNegative(color.r)), Saturate(FiniteNonNegative(color.g)),
                      Saturate(FiniteNonNegative(color.b))};
}

math::Vec3 EvaluateDirect(const math::Vec3& normal, const math::Vec3& viewDirection,
                          const math::Vec3& lightDirection, const math::Vec3& radiance,
                          const PbrMaterial& material) noexcept
{
    const math::Color brdf = EvaluateGgxBrdf(normal, viewDirection, lightDirection, material);
    const f32 nDotL = Saturate(math::Dot(normal, lightDirection));
    return Multiply(math::Vec3{brdf.r, brdf.g, brdf.b}, FiniteNonNegative(radiance)) * nDotL;
}

f32 HenyeyGreenstein(f32 cosine, f32 anisotropy) noexcept
{
    const f32 g = std::clamp(math::IsFinite(anisotropy) ? anisotropy : 0.0f, -0.9f, 0.9f);
    const f32 clampedCosine = std::clamp(math::IsFinite(cosine) ? cosine : 0.0f, -1.0f, 1.0f);
    const f32 denominator = 1.0f + g * g - 2.0f * g * clampedCosine;
    return (1.0f - g * g) /
           (4.0f * cyber::math::kPi * std::pow(std::max(denominator, 1.0e-4f), 1.5f));
}

f32 EvaluateFogShadow(const PbrDirectionalLight& light, const PbrLightingData& data,
                      const math::Vec3& position, const math::Vec3& toLight) noexcept
{
    if (light.cascades.count > 0)
    {
        DirectionalShadowCascadeSet cascades = light.cascades;
        for (u32 index = 0; index < cascades.count && index < kMaxDirectionalShadowCascades;
             ++index)
        {
            cascades.cascades[index].shadow.normalOffsetBias = 0.0f;
        }
        const f32 viewDepth = cyber::math::Dot(position - data.eyePosition, data.cameraForward);
        return EvaluateDirectionalShadowVisibility(cascades, viewDepth, position, toLight, toLight);
    }

    ShadowMapView shadow = light.shadow;
    shadow.normalOffsetBias = 0.0f;
    return shadow.depth == nullptr ? 1.0f
                                   : EvaluateShadowVisibility(shadow, position, toLight, toLight);
}

f32 EvaluateFogShadow(const ShadowMapView& source, const math::Vec3& position,
                      const math::Vec3& toLight) noexcept
{
    ShadowMapView shadow = source;
    shadow.normalOffsetBias = 0.0f;
    return shadow.depth == nullptr ? 1.0f
                                   : EvaluateShadowVisibility(shadow, position, toLight, toLight);
}

f32 SpotConeAttenuation(const PbrSpotLight& light, const math::Vec3& sampleToLight) noexcept
{
    if (!math::IsFinite(light.innerConeRadians) || !math::IsFinite(light.outerConeRadians))
    {
        return 0.0f;
    }
    const f32 innerCos =
        std::cos(std::clamp(light.innerConeRadians, 0.0f, cyber::math::kPi * 0.5f));
    const f32 outerCos =
        std::cos(std::clamp(light.outerConeRadians, 0.0f, cyber::math::kPi * 0.5f));
    const f32 coneCos = cyber::math::Dot(cyber::math::Normalize(light.direction), -sampleToLight);
    if (innerCos > outerCos)
    {
        const f32 amount = Saturate((coneCos - outerCos) / (innerCos - outerCos));
        return amount * amount * (3.0f - 2.0f * amount);
    }
    return coneCos >= innerCos ? 1.0f : 0.0f;
}

math::Vec3 EvaluateFogSource(const PbrLightingData& data, const PbrHeightFog& fog,
                             const math::Vec3& position, const math::Vec3& rayDirection) noexcept
{
    const math::Vec3 albedo{Saturate(fog.scatteringAlbedo.x), Saturate(fog.scatteringAlbedo.y),
                            Saturate(fog.scatteringAlbedo.z)};
    const f32 intensity = std::min(FiniteNonNegative(fog.scatteringIntensity), 1.0e4f);
    const math::Vec3 toEye = -rayDirection;
    math::Vec3 source = Multiply(albedo, FiniteNonNegative(data.ambientIrradiance) * kInversePi) *
                        (intensity / (4.0f * cyber::math::kPi));

    for (u32 index = 0; index < data.directionalCount && index < kMaxPbrDirectionalLights; ++index)
    {
        const PbrDirectionalLight& light = data.directional[index];
        const math::Vec3 toLight = cyber::math::Normalize(-light.direction);
        const f32 phase =
            HenyeyGreenstein(cyber::math::Dot(light.direction, toEye), fog.anisotropy);
        const f32 visibility = EvaluateFogShadow(light, data, position, toLight);
        source +=
            Multiply(albedo, FiniteNonNegative(light.radiance)) * (phase * intensity * visibility);
    }

    const cyber::core::usize pointIterations =
        data.usePointLightIndices
            ? data.pointLightIndices.Size()
            : (data.pointLightSpan.Empty() ? std::min(data.pointCount, kMaxPbrPointLights)
                                           : data.pointLightSpan.Size());
    for (cyber::core::usize index = 0; index < pointIterations; ++index)
    {
        cyber::core::usize lightIndex = index;
        if (data.usePointLightIndices)
        {
            lightIndex = data.pointLightIndices[index];
            if (lightIndex >= data.pointLightSpan.Size())
            {
                continue;
            }
        }
        const PbrPointLight& light =
            data.pointLightSpan.Empty() ? data.points[lightIndex] : data.pointLightSpan[lightIndex];
        const math::Vec3 offset = light.position - position;
        const f32 distanceSquared = cyber::math::LengthSquared(offset);
        if (!(distanceSquared > 1.0e-8f) || !math::IsFinite(distanceSquared))
        {
            continue;
        }
        const f32 distance = std::sqrt(distanceSquared);
        const f32 fade = SmoothRangeFade(distance, light.range);
        if (fade <= 0.0f)
        {
            continue;
        }
        const math::Vec3 toLight = offset / distance;
        const f32 phase = HenyeyGreenstein(cyber::math::Dot(toLight, rayDirection), fog.anisotropy);
        const f32 visibility =
            EvaluatePointShadowVisibility(light.shadow, -offset, position, toLight, toLight);
        source += Multiply(albedo, FiniteNonNegative(light.intensity)) *
                  (fade * visibility * phase * intensity / std::max(distanceSquared, 1.0e-4f));
    }

    for (u32 index = 0; index < data.spotCount && index < kMaxPbrSpotLights; ++index)
    {
        const PbrSpotLight& light = data.spots[index];
        const math::Vec3 offset = light.position - position;
        const f32 distanceSquared = cyber::math::LengthSquared(offset);
        if (!(distanceSquared > 1.0e-8f) || !math::IsFinite(distanceSquared))
        {
            continue;
        }
        const f32 distance = std::sqrt(distanceSquared);
        const f32 attenuation = SmoothRangeFade(distance, light.range) *
                                SpotConeAttenuation(light, offset / distance) /
                                std::max(distanceSquared, 1.0e-4f);
        if (attenuation <= 0.0f)
        {
            continue;
        }
        const math::Vec3 toLight = offset / distance;
        const f32 phase = HenyeyGreenstein(cyber::math::Dot(toLight, rayDirection), fog.anisotropy);
        const f32 visibility = EvaluateFogShadow(light.shadow, position, toLight);
        source += Multiply(albedo, FiniteNonNegative(light.intensity)) *
                  (attenuation * visibility * phase * intensity);
    }
    return source;
}

void ApplyHeightFog(const PbrLightingData& data, const Varyings& varyings,
                    math::Vec3& surfaceRadiance) noexcept
{
    const PbrHeightFog& fog = data.fog;
    if (!math::IsFinite(fog.density) || fog.density <= 0.0f || !math::IsFinite(fog.heightFalloff) ||
        !math::IsFinite(fog.baseHeight) || !math::IsFinite(fog.startDistance) ||
        !math::IsFinite(fog.maxDistance) || fog.maxDistance <= fog.startDistance ||
        !IsFinite(varyings.worldPosition) || !IsFinite(data.eyePosition))
    {
        return;
    }

    const math::Vec3 ray = varyings.worldPosition - data.eyePosition;
    const f32 distance = cyber::math::Length(ray);
    if (!math::IsFinite(distance) || distance <= fog.startDistance)
    {
        return;
    }
    const f32 endDistance = std::min(distance, fog.maxDistance);
    if (endDistance <= fog.startDistance)
    {
        return;
    }

    const math::Vec3 rayDirection = ray / distance;
    const f32 stepLength = (endDistance - fog.startDistance) /
                           static_cast<f32>(std::clamp(fog.integrationSteps, 1U, 32U));
    const u32 steps = std::clamp(fog.integrationSteps, 1U, 32U);
    f32 transmittance = 1.0f;
    math::Vec3 inScattering = math::Vec3::Zero();
    for (u32 step = 0; step < steps; ++step)
    {
        const f32 sampleDistance = fog.startDistance + (static_cast<f32>(step) + 0.5f) * stepLength;
        const math::Vec3 samplePosition = data.eyePosition + rayDirection * sampleDistance;
        const f32 exponent =
            std::clamp(-fog.heightFalloff * (samplePosition.y - fog.baseHeight), -80.0f, 80.0f);
        const f32 density = std::min(fog.density * std::exp(exponent), 1.0e6f);
        const f32 segmentOpacity = 1.0f - std::exp(-density * stepLength);
        if (segmentOpacity > 0.0f)
        {
            inScattering += EvaluateFogSource(data, fog, samplePosition, rayDirection) *
                            (transmittance * segmentOpacity);
            transmittance *= 1.0f - segmentOpacity;
        }
    }

    surfaceRadiance = surfaceRadiance * transmittance + inScattering;
}

f32 ToneMapChannel(f32 value, f32 exposure) noexcept
{
    const f32 x = FiniteNonNegative(value) * exposure;
    if (!math::IsFinite(x) || x >= 1.0e10f)
    {
        return 1.0f;
    }
    const f32 numerator = x * (2.51f * x + 0.03f);
    const f32 denominator = x * (2.43f * x + 0.59f) + 0.14f;
    return denominator > 0.0f ? Saturate(numerator / denominator) : 0.0f;
}

}  // namespace

math::Color EvaluateGgxBrdf(const math::Vec3& normal, const math::Vec3& viewDirection,
                            const math::Vec3& lightDirection, const PbrMaterial& material) noexcept
{
    const math::Vec3 n = cyber::math::Normalize(normal);
    const math::Vec3 v = cyber::math::Normalize(viewDirection);
    const math::Vec3 l = cyber::math::Normalize(lightDirection);
    const f32 nDotV = Saturate(cyber::math::Dot(n, v));
    const f32 nDotL = Saturate(cyber::math::Dot(n, l));
    if (nDotV <= 0.0f || nDotL <= 0.0f)
    {
        return math::Color{0.0f, 0.0f, 0.0f, 1.0f};
    }

    if (!math::IsFinite(material.metallic) || !math::IsFinite(material.roughness))
    {
        return math::Color{0.0f, 0.0f, 0.0f, 1.0f};
    }
    const f32 metallic = Saturate(material.metallic);
    const f32 roughness = std::clamp(material.roughness, kMinimumRoughness, 1.0f);
    const f32 alpha = roughness * roughness;
    const math::Vec3 baseColor = ToVec3(material.baseColor);
    const math::Vec3 f0 = math::Lerp(math::Vec3{kDielectricF0}, baseColor, metallic);
    const math::Vec3 halfVector = cyber::math::Normalize(v + l);
    const f32 nDotH = Saturate(cyber::math::Dot(n, halfVector));
    const f32 vDotH = Saturate(cyber::math::Dot(v, halfVector));

    const f32 alphaSquared = alpha * alpha;
    const f32 distributionDenominator = nDotH * nDotH * (alphaSquared - 1.0f) + 1.0f;
    const f32 distribution =
        alphaSquared /
        std::max(cyber::math::kPi * distributionDenominator * distributionDenominator, 1.0e-7f);
    const f32 geometry = SchlickGeometry(nDotV, alpha) * SchlickGeometry(nDotL, alpha);
    const math::Vec3 fresnel = FresnelSchlick(vDotH, f0);
    const math::Vec3 specular =
        fresnel * (distribution * geometry / std::max(4.0f * nDotV * nDotL, 1.0e-6f));

    const math::Vec3 diffuseWeight = (math::Vec3::One() - fresnel) * (1.0f - metallic);
    const math::Vec3 diffuse = Multiply(diffuseWeight, baseColor) * kInversePi;
    const math::Vec3 result = diffuse + specular;
    return math::Color{result.x, result.y, result.z, 1.0f};
}

math::Vec3 ApplyTangentSpaceNormal(const math::Vec3& normal, const math::Vec3& tangent,
                                   f32 tangentSign, const math::Color& normalSample,
                                   f32 normalScale) noexcept
{
    math::Vec3 n = cyber::math::Normalize(normal);
    if (cyber::math::LengthSquared(n) < 1.0e-12f)
    {
        n = math::Vec3{0.0f, 0.0f, 1.0f};
    }

    math::Vec3 t = tangent - n * cyber::math::Dot(n, tangent);
    t = cyber::math::Normalize(t);
    if (cyber::math::LengthSquared(t) < 1.0e-12f)
    {
        const math::Vec3 axis =
            std::fabs(n.z) < 0.999f ? math::Vec3{0.0f, 0.0f, 1.0f} : math::Vec3{0.0f, 1.0f, 0.0f};
        t = cyber::math::Normalize(cyber::math::Cross(axis, n));
    }
    const math::Vec3 b = cyber::math::Cross(n, t) * (tangentSign < 0.0f ? -1.0f : 1.0f);
    const f32 scale = math::IsFinite(normalScale) ? normalScale : 1.0f;
    const math::Vec3 tangentNormal{(normalSample.r * 2.0f - 1.0f) * scale,
                                   (normalSample.g * 2.0f - 1.0f) * scale,
                                   normalSample.b * 2.0f - 1.0f};
    const math::Vec3 world =
        cyber::math::Normalize(t * tangentNormal.x + b * tangentNormal.y + n * tangentNormal.z);
    return cyber::math::LengthSquared(world) > 1.0e-12f ? world : n;
}

math::Color PbrPixelShader(const Varyings& varyings, const void* userData) noexcept
{
    const auto* data = static_cast<const PbrLightingData*>(userData);
    if (data == nullptr)
    {
        return varyings.color;
    }
    if (!math::IsFinite(data->material.metallic) || !math::IsFinite(data->material.roughness))
    {
        return math::Color{0.0f, 0.0f, 0.0f, 1.0f};
    }

    math::Color baseColor = data->material.baseColor;
    if (data->baseColorTexture != nullptr)
    {
        const math::Color texel = SampleTextureLevel(*data->baseColorTexture, varyings.uv.x,
                                                     varyings.uv.y, data->mipLevel, data->sampler);
        baseColor.r *= texel.r;
        baseColor.g *= texel.g;
        baseColor.b *= texel.b;
        baseColor.a *= texel.a;
    }
    baseColor.r *= varyings.color.r;
    baseColor.g *= varyings.color.g;
    baseColor.b *= varyings.color.b;
    baseColor.a *= varyings.color.a;
    if (!math::IsFinite(baseColor.a))
    {
        baseColor.a = 1.0f;
    }

    PbrMaterial material = data->material;
    material.baseColor = baseColor;

    math::Vec3 normal = cyber::math::Normalize(varyings.normal);
    if (cyber::math::LengthSquared(normal) < 1.0e-12f)
    {
        normal = math::Vec3{0.0f, 0.0f, 1.0f};
    }
    if (data->normalTexture != nullptr)
    {
        const math::Color normalSample = SampleTextureLevel(
            *data->normalTexture, varyings.uv.x, varyings.uv.y, data->mipLevel, data->sampler);
        normal = ApplyTangentSpaceNormal(normal, varyings.tangent, varyings.tangentSign,
                                         normalSample, material.normalScale);
    }
    const math::Vec3 viewDirection =
        cyber::math::Normalize(data->eyePosition - varyings.worldPosition);
    if (data->visualizeShadowCascades && data->directionalCount > 0 &&
        data->directional[0].cascades.count > 0)
    {
        const f32 viewDepth =
            cyber::math::Dot(varyings.worldPosition - data->eyePosition, data->cameraForward);
        const u32 cascadeIndex =
            GetDirectionalShadowCascadeIndex(data->directional[0].cascades, viewDepth);
        if (cascadeIndex < data->directional[0].cascades.count)
        {
            return DirectionalShadowCascadeDebugColor(cascadeIndex);
        }
        return math::Color{0.08f, 0.08f, 0.08f, 1.0f};
    }

    math::Vec3 output{0.0f, 0.0f, 0.0f};
    const math::Vec3 diffuseAlbedo = ToVec3(baseColor) * (1.0f - Saturate(material.metallic));

    const bool hasProbes =
        !data->reflectionProbes.probes.Empty() || data->reflectionProbes.fallback != nullptr;
    const bool hasVolume = data->irradianceVolume != nullptr &&
                           data->irradianceVolume->volume != nullptr &&
                           data->irradianceVolume->volume->IsValid();
    const math::Vec3 volumeIrradiance =
        hasVolume
            ? EvaluateIrradianceVolume(*data->irradianceVolume, varyings.worldPosition, normal)
            : math::Vec3{0.0f, 0.0f, 0.0f};
    if (hasProbes || data->environment != nullptr)
    {
        IblSurface iblSurface{};
        iblSurface.useIrradianceOverride = hasVolume;
        iblSurface.irradianceOverride = volumeIrradiance;
        iblSurface.normal = normal;
        iblSurface.viewDirection = viewDirection;
        iblSurface.baseColor = ToVec3(baseColor);
        iblSurface.metallic = Saturate(material.metallic);
        iblSurface.roughness = Saturate(material.roughness);
        iblSurface.ambientOcclusion = Saturate(data->ambientOcclusion);
        if (hasProbes)
        {
            output +=
                data->screenSpaceReflections != nullptr
                    ? ResolveReflections(*data->screenSpaceReflections, data->reflectionProbes,
                                         varyings.worldPosition, iblSurface)
                    : EvaluateReflectionProbes(data->reflectionProbes, varyings.worldPosition,
                                               iblSurface);
        }
        else
        {
            output += EvaluateImageBasedLighting(*data->environment, iblSurface);
        }
    }
    else if (hasVolume)
    {
        output += Multiply(diffuseAlbedo, volumeIrradiance * kInversePi) *
                  Saturate(data->ambientOcclusion);
    }
    else
    {
        const math::Vec3 ambient = FiniteNonNegative(data->ambientIrradiance) * kInversePi;
        output += Multiply(diffuseAlbedo, ambient);
    }

    for (u32 index = 0; index < data->directionalCount && index < kMaxPbrDirectionalLights; ++index)
    {
        const PbrDirectionalLight& light = data->directional[index];
        const math::Vec3 direction = cyber::math::Normalize(-light.direction);
        const f32 viewDepth =
            cyber::math::Dot(varyings.worldPosition - data->eyePosition, data->cameraForward);
        const f32 visibility =
            light.cascades.count > 0
                ? EvaluateDirectionalShadowVisibility(light.cascades, viewDepth,
                                                      varyings.worldPosition, normal, direction)
                : (light.shadow.depth == nullptr
                       ? 1.0f
                       : EvaluateShadowVisibility(light.shadow, varyings.worldPosition, normal,
                                                  direction));
        output += EvaluateDirect(normal, viewDirection, direction,
                                 FiniteNonNegative(light.radiance) * visibility, material);
    }

    const cyber::core::usize pointIterations =
        data->usePointLightIndices
            ? data->pointLightIndices.Size()
            : (data->pointLightSpan.Empty() ? std::min(data->pointCount, kMaxPbrPointLights)
                                            : data->pointLightSpan.Size());
    for (cyber::core::usize index = 0; index < pointIterations; ++index)
    {
        cyber::core::usize lightIndex = index;
        if (data->usePointLightIndices)
        {
            lightIndex = data->pointLightIndices[index];
            if (lightIndex >= data->pointLightSpan.Size())
            {
                continue;
            }
        }
        const PbrPointLight& light = data->pointLightSpan.Empty()
                                         ? data->points[lightIndex]
                                         : data->pointLightSpan[lightIndex];
        const math::Vec3 offset = light.position - varyings.worldPosition;
        const f32 distanceSquared = cyber::math::LengthSquared(offset);
        if (!(distanceSquared > 1.0e-8f) || !math::IsFinite(distanceSquared))
        {
            continue;
        }
        const f32 distance = std::sqrt(distanceSquared);
        const f32 fade = SmoothRangeFade(distance, light.range);
        if (fade > 0.0f)
        {
            const math::Vec3 direction = offset / distance;
            const f32 visibility = EvaluatePointShadowVisibility(
                light.shadow, -offset, varyings.worldPosition, normal, direction);
            output += EvaluateDirect(normal, viewDirection, offset / distance,
                                     FiniteNonNegative(light.intensity) *
                                         (fade * visibility / std::max(distanceSquared, 1.0e-4f)),
                                     material);
        }
    }

    for (u32 index = 0; index < data->spotCount && index < kMaxPbrSpotLights; ++index)
    {
        const PbrSpotLight& light = data->spots[index];
        const math::Vec3 offset = light.position - varyings.worldPosition;
        const f32 distanceSquared = cyber::math::LengthSquared(offset);
        if (!(distanceSquared > 1.0e-8f) || !math::IsFinite(distanceSquared) ||
            !math::IsFinite(light.innerConeRadians) || !math::IsFinite(light.outerConeRadians))
        {
            continue;
        }
        const f32 distance = std::sqrt(distanceSquared);
        const f32 rangeFade = SmoothRangeFade(distance, light.range);
        const math::Vec3 toLight = offset / distance;
        const f32 innerCos =
            std::cos(std::clamp(light.innerConeRadians, 0.0f, cyber::math::kPi * 0.5f));
        const f32 outerCos =
            std::cos(std::clamp(light.outerConeRadians, 0.0f, cyber::math::kPi * 0.5f));
        const f32 coneCos = cyber::math::Dot(cyber::math::Normalize(light.direction), -toLight);
        f32 coneFade = 0.0f;
        if (innerCos > outerCos)
        {
            const f32 coneAmount = Saturate((coneCos - outerCos) / (innerCos - outerCos));
            coneFade = coneAmount * coneAmount * (3.0f - 2.0f * coneAmount);
        }
        else if (coneCos >= innerCos)
        {
            coneFade = 1.0f;
        }
        const f32 attenuation = rangeFade * coneFade / std::max(distanceSquared, 1.0e-4f);
        if (attenuation > 0.0f)
        {
            const f32 visibility = light.shadow.depth == nullptr
                                       ? 1.0f
                                       : EvaluateShadowVisibility(
                                             light.shadow, varyings.worldPosition, normal, toLight);
            output += EvaluateDirect(
                normal, viewDirection, toLight,
                FiniteNonNegative(light.intensity) * (attenuation * visibility), material);
        }
    }

    ApplyHeightFog(*data, varyings, output);
    return math::Color{output.x, output.y, output.z, baseColor.a};
}

math::Color ToneMapAces(const math::Color& hdrColor, f32 exposureEv) noexcept
{
    const f32 stops = math::IsFinite(exposureEv) ? std::clamp(exposureEv, -32.0f, 32.0f) : 0.0f;
    const f32 exposure = std::exp2(stops);
    return math::Color{ToneMapChannel(hdrColor.r, exposure), ToneMapChannel(hdrColor.g, exposure),
                       ToneMapChannel(hdrColor.b, exposure), hdrColor.a};
}

}  // namespace cyber::software
