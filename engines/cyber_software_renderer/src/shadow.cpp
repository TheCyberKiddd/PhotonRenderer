#include <cyber/software/shader.h>
#include <cyber/software/shadow.h>

#include <algorithm>
#include <cmath>

namespace cyber::software
{

namespace
{

bool IsFinite(const math::Vec3& value) noexcept
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

f32 FiniteNonNegative(f32 value) noexcept
{
    return std::isfinite(value) && value > 0.0f ? value : 0.0f;
}

bool IsFinite(const math::Mat4& value) noexcept
{
    for (cyber::core::usize column = 0; column < 4; ++column)
    {
        const math::Vec4& vector = value[column];
        if (!std::isfinite(vector.x) || !std::isfinite(vector.y) || !std::isfinite(vector.z) ||
            !std::isfinite(vector.w))
        {
            return false;
        }
    }
    return true;
}

f32 LinearShadowDistance(const ShadowMapView& shadow, f32 reverseZDepth) noexcept
{
    if (shadow.perspectiveProjection)
    {
        return shadow.nearPlane * shadow.farPlane /
               (shadow.nearPlane + reverseZDepth * (shadow.farPlane - shadow.nearPlane));
    }
    return shadow.farPlane - reverseZDepth * (shadow.farPlane - shadow.nearPlane);
}

bool IsPcssConfigured(const ShadowMapView& shadow) noexcept
{
    if (!shadow.usePcss || !std::isfinite(shadow.nearPlane) ||
        !std::isfinite(shadow.farPlane) || shadow.nearPlane <= 0.0f ||
        shadow.farPlane <= shadow.nearPlane)
    {
        return false;
    }
    if (shadow.perspectiveProjection)
    {
        return std::isfinite(shadow.verticalFovRadians) && shadow.verticalFovRadians > 0.0f &&
               shadow.verticalFovRadians < cyber::math::kPi &&
               std::isfinite(shadow.sourceRadius) && shadow.sourceRadius > 0.0f;
    }
    return std::isfinite(shadow.worldUnitsPerTexel) && shadow.worldUnitsPerTexel > 0.0f &&
           std::isfinite(shadow.sourceAngularRadiusRadians) &&
           shadow.sourceAngularRadiusRadians > 0.0f &&
           shadow.sourceAngularRadiusRadians < cyber::math::kHalfPi;
}

}  // namespace

f32 EvaluateShadowVisibility(const ShadowMapView& shadow, const math::Vec3& worldPosition,
                             const math::Vec3& worldNormal,
                             const math::Vec3& directionToLight) noexcept
{
    if (shadow.depth == nullptr || !shadow.depth->IsValid() || !IsFinite(worldPosition) ||
        !IsFinite(worldNormal) || !IsFinite(directionToLight))
    {
        return 1.0f;
    }

    const math::Vec3 normal = math::Normalize(worldNormal);
    const math::Vec3 toLight = math::Normalize(directionToLight);
    const f32 normalLight = std::clamp(math::Dot(normal, toLight), 0.0f, 1.0f);
    const f32 normalBias = FiniteNonNegative(shadow.normalOffsetBias);
    const math::Vec3 biasedPosition = worldPosition + normal * normalBias;
    const math::Vec4 clip = shadow.clipFromWorld *
                            math::Vec4{biasedPosition.x, biasedPosition.y, biasedPosition.z, 1.0f};
    if (!std::isfinite(clip.x) || !std::isfinite(clip.y) || !std::isfinite(clip.z) ||
        !std::isfinite(clip.w) || clip.w <= 1.0e-8f)
    {
        return 1.0f;
    }

    const f32 inverseW = 1.0f / clip.w;
    const f32 ndcX = clip.x * inverseW;
    const f32 ndcY = clip.y * inverseW;
    const f32 receiverDepth = clip.z * inverseW;
    if (!std::isfinite(ndcX) || !std::isfinite(ndcY) || !std::isfinite(receiverDepth) ||
        ndcX < -1.0f || ndcX > 1.0f || ndcY < -1.0f || ndcY > 1.0f || receiverDepth < 0.0f ||
        receiverDepth > 1.0f)
    {
        return 1.0f;
    }

    const f32 grazing = 1.0f - normalLight;
    const f32 depthBias =
        FiniteNonNegative(shadow.constantDepthBias) +
        FiniteNonNegative(shadow.slopeDepthBias) * grazing / std::max(normalLight, 0.1f);
    const f32 biasedDepth = std::min(receiverDepth + depthBias, 1.0f);
    const f32 u = (ndcX * 0.5f + 0.5f) * static_cast<f32>(shadow.depth->Width());
    const f32 v = (0.5f - ndcY * 0.5f) * static_cast<f32>(shadow.depth->Height());
    const cyber::core::i32 centerX = std::min(
        static_cast<cyber::core::i32>(u), static_cast<cyber::core::i32>(shadow.depth->Width() - 1));
    const cyber::core::i32 centerY =
        std::min(static_cast<cyber::core::i32>(v),
                 static_cast<cyber::core::i32>(shadow.depth->Height() - 1));
    const u32 maximumRadius =
        (shadow.softFilter || shadow.usePcss) ? kMaxSoftShadowFilterRadius
                                              : kMaxShadowFilterRadius;
    const cyber::core::i32 radius =
        static_cast<cyber::core::i32>(std::min(shadow.filterRadius, maximumRadius));

    if (IsPcssConfigured(shadow) && radius > 0)
    {
        constexpr f32 kDiskOffsets[16][2] = {
            {-0.1305f, -0.9914f}, {0.4321f, -0.9018f},  {-0.7284f, -0.6842f},
            {0.8783f, -0.4301f},  {-0.9912f, -0.1320f}, {0.6834f, 0.1591f},
            {-0.3567f, 0.2553f},  {0.9732f, 0.2265f},   {-0.8691f, 0.4885f},
            {0.1712f, 0.5902f},   {-0.5856f, 0.8105f},  {0.5501f, 0.8344f},
            {0.3008f, -0.4794f},  {-0.0762f, 0.9621f},  {0.8173f, 0.7440f},
            {-0.4302f, -0.3107f},
        };
        const f32 maxRadius = static_cast<f32>(radius);
        const f32 receiverDistance = LinearShadowDistance(shadow, biasedDepth);
        f32 searchRadius = 0.0f;
        if (shadow.perspectiveProjection)
        {
            const f32 tanHalfFov = std::tan(shadow.verticalFovRadians * 0.5f);
            searchRadius =
                shadow.sourceRadius * (receiverDistance - shadow.nearPlane) /
                (receiverDistance * 2.0f * shadow.nearPlane * tanHalfFov) *
                static_cast<f32>(shadow.depth->Width());
        }
        else
        {
            searchRadius =
                std::tan(shadow.sourceAngularRadiusRadians) *
                std::max(receiverDistance - shadow.nearPlane, 0.0f) /
                shadow.worldUnitsPerTexel;
        }
        if (!std::isfinite(searchRadius) || !std::isfinite(receiverDistance) ||
            receiverDistance <= 0.0f)
        {
            searchRadius = 1.0f;
        }
        searchRadius = std::clamp(searchRadius, 1.0f, maxRadius);

        f32 blockerDistanceSum = 0.0f;
        u32 blockerCount = 0;
        for (const auto& offset : kDiskOffsets)
        {
            const cyber::core::i32 sampleX =
                centerX + static_cast<cyber::core::i32>(std::round(offset[0] * searchRadius));
            const cyber::core::i32 sampleY =
                centerY + static_cast<cyber::core::i32>(std::round(offset[1] * searchRadius));
            if (sampleX < 0 || sampleX >= shadow.depth->Width() || sampleY < 0 ||
                sampleY >= shadow.depth->Height())
            {
                continue;
            }
            const f32 blockerDepth = shadow.depth->Get(sampleX, sampleY);
            if (blockerDepth > biasedDepth)
            {
                blockerDistanceSum += LinearShadowDistance(shadow, blockerDepth);
                ++blockerCount;
            }
        }
        if (blockerCount == 0)
        {
            return 1.0f;
        }

        const f32 averageBlockerDistance =
            blockerDistanceSum / static_cast<f32>(blockerCount);
        f32 penumbraRadius = 0.0f;
        if (receiverDistance > averageBlockerDistance && averageBlockerDistance > 0.0f)
        {
            if (shadow.perspectiveProjection)
            {
                penumbraRadius =
                    shadow.sourceRadius * (receiverDistance - averageBlockerDistance) /
                    averageBlockerDistance *
                    static_cast<f32>(shadow.depth->Width()) /
                    (2.0f * receiverDistance * std::tan(shadow.verticalFovRadians * 0.5f));
            }
            else
            {
                penumbraRadius =
                    std::tan(shadow.sourceAngularRadiusRadians) *
                    (receiverDistance - averageBlockerDistance) / shadow.worldUnitsPerTexel;
            }
        }
        if (!std::isfinite(penumbraRadius))
        {
            penumbraRadius = maxRadius;
        }
        penumbraRadius = std::clamp(penumbraRadius, 0.0f, maxRadius);
        if (penumbraRadius < 0.5f)
        {
            return biasedDepth >= shadow.depth->Get(centerX, centerY) ? 1.0f : 0.0f;
        }

        u32 litSamples = 0;
        for (const auto& offset : kDiskOffsets)
        {
            const cyber::core::i32 sampleX =
                centerX + static_cast<cyber::core::i32>(std::round(offset[0] * penumbraRadius));
            const cyber::core::i32 sampleY =
                centerY + static_cast<cyber::core::i32>(std::round(offset[1] * penumbraRadius));
            if (sampleX < 0 || sampleX >= shadow.depth->Width() || sampleY < 0 ||
                sampleY >= shadow.depth->Height() ||
                biasedDepth >= shadow.depth->Get(sampleX, sampleY))
            {
                ++litSamples;
            }
        }
        return static_cast<f32>(litSamples) / 16.0f;
    }

    u32 litSamples = 0;
    u32 sampleCount = 0;
    if (shadow.softFilter && radius > 0)
    {
        constexpr f32 kStratifiedOffsets[4] = {-0.75f, -0.25f, 0.25f, 0.75f};
        for (const f32 yOffset : kStratifiedOffsets)
        {
            for (const f32 xOffset : kStratifiedOffsets)
            {
                ++sampleCount;
                const cyber::core::i32 sampleX =
                    centerX + static_cast<cyber::core::i32>(std::round(xOffset * radius));
                const cyber::core::i32 sampleY =
                    centerY + static_cast<cyber::core::i32>(std::round(yOffset * radius));
                if (sampleX < 0 || sampleX >= shadow.depth->Width() || sampleY < 0 ||
                    sampleY >= shadow.depth->Height() ||
                    biasedDepth >= shadow.depth->Get(sampleX, sampleY))
                {
                    ++litSamples;
                }
            }
        }
    }
    else
    {
        for (cyber::core::i32 y = -radius; y <= radius; ++y)
        {
            for (cyber::core::i32 x = -radius; x <= radius; ++x)
            {
                const cyber::core::i32 sampleX = centerX + x;
                const cyber::core::i32 sampleY = centerY + y;
                ++sampleCount;
                if (sampleX < 0 || sampleX >= shadow.depth->Width() || sampleY < 0 ||
                    sampleY >= shadow.depth->Height() ||
                    biasedDepth >= shadow.depth->Get(sampleX, sampleY))
                {
                    ++litSamples;
                }
            }
        }
    }
    return static_cast<f32>(litSamples) / static_cast<f32>(sampleCount);
}

core::Status<core::Error>
BuildDirectionalShadowCascades(const Camera& camera, f32 aspect, const math::Vec3& lightDirection,
                               const DirectionalShadowCascadeConfig& config,
                               DirectionalShadowCascadeSet& output) noexcept
{
    output = DirectionalShadowCascadeSet{};
    const f32 lightLengthSquared = cyber::math::LengthSquared(lightDirection);
    if (!IsFinite(camera.position) || !std::isfinite(camera.yawRadians) ||
        !std::isfinite(camera.pitchRadians) || !std::isfinite(camera.fovYRadians) ||
        camera.fovYRadians <= 0.0f || camera.fovYRadians >= cyber::math::kPi ||
        !std::isfinite(camera.nearPlane) || !std::isfinite(camera.farPlane) ||
        camera.nearPlane <= 0.0f || camera.farPlane <= camera.nearPlane || !std::isfinite(aspect) ||
        aspect <= 0.0f || !IsFinite(lightDirection) || !std::isfinite(lightLengthSquared) ||
        lightLengthSquared <= 1.0e-12f || config.cascadeCount == 0 ||
        config.cascadeCount > kMaxDirectionalShadowCascades || config.mapResolution < 8 ||
        config.mapResolution > 16384 || !std::isfinite(config.shadowDistance) ||
        config.shadowDistance <= camera.nearPlane || !std::isfinite(config.splitLambda) ||
        config.splitLambda < 0.0f || config.splitLambda > 1.0f ||
        !std::isfinite(config.blendFraction) || config.blendFraction < 0.0f ||
        config.blendFraction > 0.5f || !std::isfinite(config.depthPadding) ||
        config.depthPadding < 0.0f || !std::isfinite(config.constantDepthBias) ||
        config.constantDepthBias < 0.0f || !std::isfinite(config.slopeDepthBias) ||
        config.slopeDepthBias < 0.0f || !std::isfinite(config.normalOffsetBias) ||
        config.normalOffsetBias < 0.0f ||
        !std::isfinite(config.sourceAngularRadiusRadians) ||
        config.sourceAngularRadiusRadians < 0.0f ||
        config.sourceAngularRadiusRadians >= cyber::math::kHalfPi)
    {
        return core::Status<core::Error>::Err(
            core::Error{core::ErrorCode::InvalidArgument, "invalid directional shadow settings"});
    }

    const f32 farDistance = std::min(camera.farPlane, config.shadowDistance);
    if (!(farDistance > camera.nearPlane))
    {
        return core::Status<core::Error>::Err(
            core::Error{core::ErrorCode::InvalidArgument,
                        "shadow distance does not cover the camera near plane"});
    }

    const math::Vec3 forward = CameraForward(camera);
    const math::Vec3 right = CameraRight(camera);
    const math::Vec3 up = CameraUp(camera);
    const math::Vec3 lightForward = cyber::math::Normalize(lightDirection);
    const math::Vec3 upReference =
        std::fabs(cyber::math::Dot(lightForward, math::Vec3{0.0f, 1.0f, 0.0f})) > 0.99f
            ? math::Vec3{1.0f, 0.0f, 0.0f}
            : math::Vec3{0.0f, 1.0f, 0.0f};
    const math::Vec3 lightRight =
        cyber::math::Normalize(cyber::math::Cross(lightForward, upReference));
    const math::Vec3 lightUp = cyber::math::Cross(lightRight, lightForward);
    if (!IsFinite(forward) || !IsFinite(right) || !IsFinite(up) || !IsFinite(lightForward) ||
        !IsFinite(lightRight) || !IsFinite(lightUp))
    {
        return core::Status<core::Error>::Err(
            core::Error{core::ErrorCode::InvalidArgument, "camera or light basis is degenerate"});
    }

    const f32 tangentY = std::tan(camera.fovYRadians * 0.5f);
    const f32 logarithmicRatio = farDistance / camera.nearPlane;
    const f32 lightDistancePadding = config.depthPadding;
    f32 previousSplit = camera.nearPlane;
    output.count = config.cascadeCount;
    output.blendFraction = config.blendFraction;

    for (u32 cascadeIndex = 0; cascadeIndex < config.cascadeCount; ++cascadeIndex)
    {
        const f32 fraction =
            static_cast<f32>(cascadeIndex + 1) / static_cast<f32>(config.cascadeCount);
        const f32 logarithmicSplit = camera.nearPlane * std::pow(logarithmicRatio, fraction);
        const f32 uniformSplit = camera.nearPlane + (farDistance - camera.nearPlane) * fraction;
        const f32 splitFar =
            config.splitLambda * logarithmicSplit + (1.0f - config.splitLambda) * uniformSplit;
        const f32 splitMiddle = (previousSplit + splitFar) * 0.5f;
        const f32 nearHalfHeight = previousSplit * tangentY;
        const f32 farHalfHeight = splitFar * tangentY;
        const f32 nearHalfWidth = nearHalfHeight * aspect;
        const f32 farHalfWidth = farHalfHeight * aspect;
        const math::Vec3 center = camera.position + forward * splitMiddle;
        math::Vec3 corners[8];
        u32 cornerIndex = 0;
        for (u32 plane = 0; plane < 2; ++plane)
        {
            const f32 distance = plane == 0 ? previousSplit : splitFar;
            const f32 halfWidth = plane == 0 ? nearHalfWidth : farHalfWidth;
            const f32 halfHeight = plane == 0 ? nearHalfHeight : farHalfHeight;
            const math::Vec3 planeCenter = camera.position + forward * distance;
            for (const f32 ySign : {-1.0f, 1.0f})
            {
                for (const f32 xSign : {-1.0f, 1.0f})
                {
                    corners[cornerIndex++] =
                        planeCenter + right * (xSign * halfWidth) + up * (ySign * halfHeight);
                }
            }
        }

        f32 radius = 0.0f;
        for (const math::Vec3& corner : corners)
        {
            radius = std::max(radius, cyber::math::Length(corner - center));
        }
        radius = std::ceil(radius * 16.0f) / 16.0f;
        radius *=
            static_cast<f32>(config.mapResolution) / static_cast<f32>(config.mapResolution - 2);
        if (!std::isfinite(radius) || radius <= 0.0f)
        {
            output = DirectionalShadowCascadeSet{};
            return core::Status<core::Error>::Err(
                core::Error{core::ErrorCode::InvalidArgument, "cascade bounds are not finite"});
        }

        const f32 texelWorldSize = (2.0f * radius) / static_cast<f32>(config.mapResolution);
        const f32 lightCenterX = cyber::math::Dot(center, lightRight);
        const f32 lightCenterY = cyber::math::Dot(center, lightUp);
        const f32 snappedX = std::round(lightCenterX / texelWorldSize) * texelWorldSize;
        const f32 snappedY = std::round(lightCenterY / texelWorldSize) * texelWorldSize;
        const math::Vec3 snappedCenter =
            center + lightRight * (snappedX - lightCenterX) + lightUp * (snappedY - lightCenterY);
        const f32 eyeDistance = radius + lightDistancePadding + 0.1f;
        const math::Vec3 lightEye = snappedCenter - lightForward * eyeDistance;
        const math::Mat4 lightView = math::Mat4::LookAt(lightEye, snappedCenter, upReference);
        const f32 lightFar = 2.0f * eyeDistance;
        const math::Mat4 lightProjection =
            math::Mat4::OrthographicReverseZ(-radius, radius, -radius, radius, 0.1f, lightFar);
        const math::Mat4 clipFromWorld = lightProjection * lightView;
        if (!IsFinite(clipFromWorld))
        {
            output = DirectionalShadowCascadeSet{};
            return core::Status<core::Error>::Err(
                core::Error{core::ErrorCode::InvalidArgument, "cascade projection is not finite"});
        }

        DirectionalShadowCascade& cascade = output.cascades[cascadeIndex];
        cascade.splitNear = previousSplit;
        cascade.splitFar = splitFar;
        cascade.shadow.clipFromWorld = clipFromWorld;
        cascade.shadow.constantDepthBias = config.constantDepthBias;
        cascade.shadow.slopeDepthBias = config.slopeDepthBias;
        cascade.shadow.normalOffsetBias = config.normalOffsetBias;
        cascade.shadow.filterRadius = std::min(config.filterRadius, kMaxSoftShadowFilterRadius);
        cascade.shadow.softFilter = true;
        cascade.shadow.usePcss = config.sourceAngularRadiusRadians > 0.0f;
        cascade.shadow.nearPlane = 0.1f;
        cascade.shadow.farPlane = lightFar;
        cascade.shadow.worldUnitsPerTexel = texelWorldSize;
        cascade.shadow.sourceAngularRadiusRadians = config.sourceAngularRadiusRadians;
        previousSplit = splitFar;
    }

    return core::Status<core::Error>::Ok();
}

u32 GetDirectionalShadowCascadeIndex(const DirectionalShadowCascadeSet& cascades,
                                     f32 viewDepth) noexcept
{
    if (cascades.count == 0 || cascades.count > kMaxDirectionalShadowCascades ||
        !std::isfinite(viewDepth) || viewDepth < cascades.cascades[0].splitNear)
    {
        return kMaxDirectionalShadowCascades;
    }
    for (u32 index = 0; index < cascades.count; ++index)
    {
        if (viewDepth <= cascades.cascades[index].splitFar)
        {
            return index;
        }
    }
    return kMaxDirectionalShadowCascades;
}

math::Color DirectionalShadowCascadeDebugColor(u32 cascadeIndex) noexcept
{
    constexpr math::Color kColors[kMaxDirectionalShadowCascades] = {
        {1.0f, 0.2f, 0.2f, 1.0f},
        {0.2f, 0.8f, 0.25f, 1.0f},
        {0.2f, 0.45f, 1.0f, 1.0f},
        {1.0f, 0.75f, 0.15f, 1.0f},
    };
    return cascadeIndex < kMaxDirectionalShadowCascades ? kColors[cascadeIndex]
                                                        : math::Color{0.08f, 0.08f, 0.08f, 1.0f};
}

f32 EvaluateDirectionalShadowVisibility(const DirectionalShadowCascadeSet& cascades, f32 viewDepth,
                                        const math::Vec3& worldPosition,
                                        const math::Vec3& worldNormal,
                                        const math::Vec3& directionToLight) noexcept
{
    if (cascades.count == 0 || cascades.count > kMaxDirectionalShadowCascades ||
        !std::isfinite(viewDepth) || viewDepth < cascades.cascades[0].splitNear)
    {
        return 1.0f;
    }

    for (u32 index = 0; index < cascades.count; ++index)
    {
        const DirectionalShadowCascade& cascade = cascades.cascades[index];
        if (viewDepth > cascade.splitFar)
        {
            continue;
        }
        const f32 visibility =
            EvaluateShadowVisibility(cascade.shadow, worldPosition, worldNormal, directionToLight);
        if (index + 1 >= cascades.count || cascades.blendFraction <= 0.0f)
        {
            return visibility;
        }

        const f32 blendWidth = (cascade.splitFar - cascade.splitNear) * cascades.blendFraction;
        const f32 blendStart = cascade.splitFar - blendWidth;
        if (blendWidth <= 0.0f || viewDepth < blendStart)
        {
            return visibility;
        }
        const f32 nextVisibility = EvaluateShadowVisibility(
            cascades.cascades[index + 1].shadow, worldPosition, worldNormal, directionToLight);
        const f32 blend = std::clamp((viewDepth - blendStart) / blendWidth, 0.0f, 1.0f);
        return visibility + (nextVisibility - visibility) * blend;
    }
    return 1.0f;
}

core::Status<core::Error> BuildSpotShadowMap(const math::Vec3& position,
                                             const math::Vec3& direction,
                                             const SpotShadowConfig& config,
                                             ShadowMapView& output) noexcept
{
    output = ShadowMapView{};
    const f32 directionLengthSquared = cyber::math::LengthSquared(direction);
    if (!IsFinite(position) || !IsFinite(direction) || !std::isfinite(directionLengthSquared) ||
        directionLengthSquared <= 1.0e-12f || !std::isfinite(config.nearPlane) ||
        !std::isfinite(config.farPlane) || config.nearPlane <= 0.0f ||
        config.farPlane <= config.nearPlane || !std::isfinite(config.outerConeRadians) ||
        config.outerConeRadians <= 0.0f || config.outerConeRadians >= cyber::math::kHalfPi ||
        !std::isfinite(config.constantDepthBias) || config.constantDepthBias < 0.0f ||
        !std::isfinite(config.slopeDepthBias) || config.slopeDepthBias < 0.0f ||
        !std::isfinite(config.normalOffsetBias) || config.normalOffsetBias < 0.0f ||
        !std::isfinite(config.sourceRadius) || config.sourceRadius < 0.0f)
    {
        return core::Status<core::Error>::Err(
            core::Error{core::ErrorCode::InvalidArgument, "invalid spot shadow settings"});
    }

    const math::Vec3 forward = cyber::math::Normalize(direction);
    const math::Vec3 up = std::fabs(cyber::math::Dot(forward, math::Vec3{0.0f, 1.0f, 0.0f})) > 0.99f
                              ? math::Vec3{1.0f, 0.0f, 0.0f}
                              : math::Vec3{0.0f, 1.0f, 0.0f};
    const math::Mat4 view = math::Mat4::LookAt(position, position + forward, up);
    const math::Mat4 projection = math::Mat4::PerspectiveReverseZ(
        config.outerConeRadians * 2.0f, 1.0f, config.nearPlane, config.farPlane);
    output.clipFromWorld = projection * view;
    output.constantDepthBias = config.constantDepthBias;
    output.slopeDepthBias = config.slopeDepthBias;
    output.normalOffsetBias = config.normalOffsetBias;
    output.filterRadius = std::min(config.filterRadius, kMaxSoftShadowFilterRadius);
    output.softFilter = true;
    output.usePcss = config.sourceRadius > 0.0f;
    output.perspectiveProjection = true;
    output.nearPlane = config.nearPlane;
    output.farPlane = config.farPlane;
    output.verticalFovRadians = config.outerConeRadians * 2.0f;
    output.sourceRadius = config.sourceRadius;
    if (!IsFinite(output.clipFromWorld))
    {
        output = ShadowMapView{};
        return core::Status<core::Error>::Err(
            core::Error{core::ErrorCode::InvalidArgument, "spot shadow transform is not finite"});
    }
    return core::Status<core::Error>::Ok();
}

core::Status<core::Error> BuildPointShadowMaps(const math::Vec3& position,
                                               const PointShadowConfig& config,
                                               PointShadowMapSet& output) noexcept
{
    output = PointShadowMapSet{};
    if (!IsFinite(position) || !std::isfinite(config.nearPlane) ||
        !std::isfinite(config.farPlane) || config.nearPlane <= 0.0f ||
        config.farPlane <= config.nearPlane || !std::isfinite(config.constantDepthBias) ||
        config.constantDepthBias < 0.0f || !std::isfinite(config.slopeDepthBias) ||
        config.slopeDepthBias < 0.0f || !std::isfinite(config.normalOffsetBias) ||
        config.normalOffsetBias < 0.0f || !std::isfinite(config.sourceRadius) ||
        config.sourceRadius < 0.0f)
    {
        return core::Status<core::Error>::Err(
            core::Error{core::ErrorCode::InvalidArgument, "invalid point shadow settings"});
    }

    constexpr math::Vec3 kDirections[6] = {
        {1.0f, 0.0f, 0.0f},  {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
        {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f},  {0.0f, 0.0f, -1.0f},
    };
    constexpr math::Vec3 kUp[6] = {
        {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, -1.0f},
        {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
    };
    const math::Mat4 projection = math::Mat4::PerspectiveReverseZ(
        cyber::math::kHalfPi, 1.0f, config.nearPlane, config.farPlane);
    for (u32 face = 0; face < 6; ++face)
    {
        ShadowMapView& map = output.faces[face];
        map.clipFromWorld =
            projection * math::Mat4::LookAt(position, position + kDirections[face], kUp[face]);
        map.constantDepthBias = config.constantDepthBias;
        map.slopeDepthBias = config.slopeDepthBias;
        map.normalOffsetBias = config.normalOffsetBias;
        map.filterRadius = std::min(config.filterRadius, kMaxSoftShadowFilterRadius);
        map.softFilter = true;
        map.usePcss = config.sourceRadius > 0.0f;
        map.perspectiveProjection = true;
        map.nearPlane = config.nearPlane;
        map.farPlane = config.farPlane;
        map.verticalFovRadians = cyber::math::kHalfPi;
        map.sourceRadius = config.sourceRadius;
        if (!IsFinite(map.clipFromWorld))
        {
            output = PointShadowMapSet{};
            return core::Status<core::Error>::Err(core::Error{
                core::ErrorCode::InvalidArgument, "point shadow face transform is not finite"});
        }
    }
    return core::Status<core::Error>::Ok();
}

u32 SelectShadowCubeFace(const math::Vec3& lightToReceiver) noexcept
{
    if (!IsFinite(lightToReceiver))
    {
        return 0;
    }
    const f32 ax = std::fabs(lightToReceiver.x);
    const f32 ay = std::fabs(lightToReceiver.y);
    const f32 az = std::fabs(lightToReceiver.z);
    if (ax >= ay && ax >= az)
    {
        return lightToReceiver.x >= 0.0f ? 0U : 1U;
    }
    if (ay >= az)
    {
        return lightToReceiver.y >= 0.0f ? 2U : 3U;
    }
    return lightToReceiver.z >= 0.0f ? 4U : 5U;
}

f32 EvaluatePointShadowVisibility(const PointShadowMapSet& shadows,
                                  const math::Vec3& lightToReceiver,
                                  const math::Vec3& worldPosition,
                                  const math::Vec3& worldNormal,
                                  const math::Vec3& directionToLight) noexcept
{
    if (!IsFinite(lightToReceiver))
    {
        return 1.0f;
    }

    const f32 axes[3] = {std::fabs(lightToReceiver.x), std::fabs(lightToReceiver.y),
                         std::fabs(lightToReceiver.z)};
    const f32 maximumAxis = std::max({axes[0], axes[1], axes[2]});
    if (maximumAxis <= 1.0e-8f || !std::isfinite(maximumAxis))
    {
        return 1.0f;
    }
    const u32 faces[3] = {lightToReceiver.x >= 0.0f ? 0U : 1U,
                          lightToReceiver.y >= 0.0f ? 2U : 3U,
                          lightToReceiver.z >= 0.0f ? 4U : 5U};

    f32 weightedVisibility = 0.0f;
    f32 totalWeight = 0.0f;
    for (u32 axis = 0; axis < 3; ++axis)
    {
        const f32 relativeAxis = axes[axis] / maximumAxis;
        const f32 transition = std::clamp((relativeAxis - 0.85f) / 0.15f, 0.0f, 1.0f);
        const f32 weight = transition * transition * (3.0f - 2.0f * transition);
        if (weight <= 0.0f)
        {
            continue;
        }

        const ShadowMapView& shadow = shadows.faces[faces[axis]];
        const f32 visibility =
            shadow.depth == nullptr
                ? 1.0f
                : EvaluateShadowVisibility(shadow, worldPosition, worldNormal, directionToLight);
        weightedVisibility += visibility * weight;
        totalWeight += weight;
    }
    return totalWeight > 0.0f ? weightedVisibility / totalWeight : 1.0f;
}

core::Status<core::Error> InitializeShadowAtlas(u32 width, u32 height, ShadowAtlas& output) noexcept
{
    output = ShadowAtlas{};
    if (width < kShadowAtlasTileSize || height < kShadowAtlasTileSize ||
        width > kShadowAtlasTileSize * 64 || height > kShadowAtlasTileSize * 64 ||
        (width % kShadowAtlasTileSize) != 0 || (height % kShadowAtlasTileSize) != 0)
    {
        return core::Status<core::Error>::Err(
            core::Error{core::ErrorCode::InvalidArgument, "invalid shadow atlas dimensions"});
    }
    output.width = width;
    output.height = height;
    return core::Status<core::Error>::Ok();
}

core::Status<core::Error> SelectShadowResolution(f32 importance, u32 minimumResolution,
                                                 u32 maximumResolution, u32& output) noexcept
{
    output = 0;
    const auto isPowerOfTwo = [](u32 value) noexcept
    { return value != 0 && (value & (value - 1U)) == 0; };
    if (!std::isfinite(importance) || importance < 0.0f || importance > 1.0f ||
        minimumResolution < kShadowAtlasTileSize || maximumResolution < minimumResolution ||
        maximumResolution > kShadowAtlasTileSize * 64 || !isPowerOfTwo(minimumResolution) ||
        !isPowerOfTwo(maximumResolution))
    {
        return core::Status<core::Error>::Err(
            core::Error{core::ErrorCode::InvalidArgument, "invalid shadow resolution importance"});
    }

    const f32 target =
        static_cast<f32>(minimumResolution) *
        std::pow(static_cast<f32>(maximumResolution) / static_cast<f32>(minimumResolution),
                 importance);
    u32 resolution = minimumResolution;
    while (resolution < maximumResolution && static_cast<f32>(resolution) < target)
    {
        resolution *= 2;
    }
    if (importance >= 1.0f)
    {
        resolution = maximumResolution;
    }
    output = std::clamp(resolution, minimumResolution, maximumResolution);
    return core::Status<core::Error>::Ok();
}

core::Status<core::Error> AllocateShadowAtlas(ShadowAtlas& atlas, u32 requestedResolution,
                                              u32 guardBandTexels,
                                              ShadowAtlasAllocation& output) noexcept
{
    output = ShadowAtlasAllocation{};
    if (atlas.width < kShadowAtlasTileSize || atlas.height < kShadowAtlasTileSize ||
        requestedResolution == 0 || guardBandTexels > kShadowAtlasTileSize ||
        requestedResolution > atlas.width || requestedResolution > atlas.height ||
        requestedResolution > 0xFFFFFFFFU - 2U * guardBandTexels)
    {
        return core::Status<core::Error>::Err(
            core::Error{core::ErrorCode::InvalidArgument, "invalid shadow atlas request"});
    }

    u32 activeSlot = kMaxShadowAtlasAllocations;
    for (u32 slot = 0; slot < kMaxShadowAtlasAllocations; ++slot)
    {
        if (!atlas.allocations[slot].active)
        {
            activeSlot = slot;
            break;
        }
    }
    if (activeSlot == kMaxShadowAtlasAllocations)
    {
        return core::Status<core::Error>::Err(
            core::Error{core::ErrorCode::OutOfRange, "shadow atlas allocation table is full"});
    }

    const u32 required = requestedResolution + 2U * guardBandTexels;
    u32 allocationSize = kShadowAtlasTileSize;
    while (allocationSize < required)
    {
        if (allocationSize > 0x7FFFFFFFU)
        {
            return core::Status<core::Error>::Err(
                core::Error{core::ErrorCode::Overflow, "shadow atlas request size overflow"});
        }
        allocationSize *= 2;
    }
    if (allocationSize > atlas.width || allocationSize > atlas.height)
    {
        return core::Status<core::Error>::Err(
            core::Error{core::ErrorCode::OutOfRange, "shadow map does not fit in the atlas"});
    }

    const u32 tileWidth = atlas.width / kShadowAtlasTileSize;
    const u32 tileHeight = atlas.height / kShadowAtlasTileSize;
    const u32 tileSpan = allocationSize / kShadowAtlasTileSize;
    const u64 mask = tileSpan == 64 ? ~u64{0} : ((u64{1} << tileSpan) - 1U);
    for (u32 tileY = 0; tileY + tileSpan <= tileHeight; tileY += tileSpan)
    {
        for (u32 tileX = 0; tileX + tileSpan <= tileWidth; tileX += tileSpan)
        {
            const u64 shiftedMask = mask << tileX;
            bool free = true;
            for (u32 row = tileY; row < tileY + tileSpan; ++row)
            {
                if ((atlas.occupiedRows[row] & shiftedMask) != 0)
                {
                    free = false;
                    break;
                }
            }
            if (!free)
            {
                continue;
            }
            for (u32 row = tileY; row < tileY + tileSpan; ++row)
            {
                atlas.occupiedRows[row] |= shiftedMask;
            }
            u32 generation = atlas.nextGeneration++;
            if (generation == 0)
            {
                generation = atlas.nextGeneration++;
            }
            ShadowAtlas::AllocationRecord& record = atlas.allocations[activeSlot];
            record.x = tileX * kShadowAtlasTileSize;
            record.y = tileY * kShadowAtlasTileSize;
            record.size = allocationSize;
            record.generation = generation;
            record.active = true;
            output.x = record.x;
            output.y = record.y;
            output.size = allocationSize;
            output.innerOffset = guardBandTexels;
            output.innerResolution = requestedResolution;
            output.slot = activeSlot;
            output.generation = generation;
            return core::Status<core::Error>::Ok();
        }
    }
    return core::Status<core::Error>::Err(
        core::Error{core::ErrorCode::OutOfRange, "shadow atlas is out of free space"});
}

bool ReleaseShadowAtlas(ShadowAtlas& atlas, const ShadowAtlasAllocation& allocation) noexcept
{
    if (allocation.slot >= kMaxShadowAtlasAllocations || allocation.generation == 0)
    {
        return false;
    }
    ShadowAtlas::AllocationRecord& record = atlas.allocations[allocation.slot];
    if (!record.active || record.generation != allocation.generation || record.x != allocation.x ||
        record.y != allocation.y || record.size != allocation.size || record.size == 0 ||
        (record.x % kShadowAtlasTileSize) != 0 || (record.y % kShadowAtlasTileSize) != 0 ||
        (record.size % kShadowAtlasTileSize) != 0)
    {
        return false;
    }

    const u32 tileX = record.x / kShadowAtlasTileSize;
    const u32 tileY = record.y / kShadowAtlasTileSize;
    const u32 tileSpan = record.size / kShadowAtlasTileSize;
    if (tileSpan > 64 || tileX + tileSpan > atlas.width / kShadowAtlasTileSize ||
        tileY + tileSpan > atlas.height / kShadowAtlasTileSize)
    {
        return false;
    }
    const u64 mask = tileSpan == 64 ? ~u64{0} : ((u64{1} << tileSpan) - 1U);
    const u64 shiftedMask = mask << tileX;
    for (u32 row = tileY; row < tileY + tileSpan; ++row)
    {
        if ((atlas.occupiedRows[row] & shiftedMask) != shiftedMask)
        {
            return false;
        }
    }
    for (u32 row = tileY; row < tileY + tileSpan; ++row)
    {
        atlas.occupiedRows[row] &= ~shiftedMask;
    }
    record.active = false;
    return true;
}

math::Color ShadowDepthDebugColor(f32 reverseZDepth) noexcept
{
    const f32 value = std::isfinite(reverseZDepth) ? std::clamp(reverseZDepth, 0.0f, 1.0f) : 0.0f;
    return math::Color{value, value, value, 1.0f};
}

core::Status<core::Error>
RenderShadowMap(const ShadowRenderDesc& desc, cyber::memory::Span<const VertexInput> vertices,
                DepthBuffer& depth, jobs::JobSystem* jobSystem,
                cyber::memory::Allocator& allocator, PipelineStats* stats) noexcept
{
    const math::Mat4 modelViewProjection = desc.clipFromWorld * desc.modelToWorld;
    if (!depth.IsValid() || !IsFinite(desc.modelToWorld) || !IsFinite(desc.clipFromWorld) ||
        !IsFinite(modelViewProjection))
    {
        if (stats != nullptr)
        {
            *stats = PipelineStats{};
        }
        return core::Status<core::Error>::Err(core::Error{
            core::ErrorCode::InvalidArgument, "invalid shadow map render target or transforms"});
    }

    DepthOnlyShaderDesc shader{};
    shader.vertexShader = TransformDepthOnlyVertexShader;
    shader.userData = &modelViewProjection;
    shader.pipeline.viewport =
        Viewport{0.0f, 0.0f, static_cast<f32>(depth.Width()), static_cast<f32>(depth.Height())};
    shader.pipeline.cullMode = desc.cullMode;
    shader.pipeline.useJobs = desc.useJobs;

    depth.Clear();
    return DrawDepthOnlyTriangles(shader, vertices, depth, jobSystem, allocator, stats);
}

core::Status<core::Error>
RenderDirectionalShadowMaps(DirectionalShadowCascadeSet& cascades,
                            cyber::memory::Span<const VertexInput> vertices,
                            cyber::memory::Span<DepthBuffer> depthMaps,
                            cyber::memory::Allocator& allocator, jobs::JobSystem* jobSystem,
                            CullMode cullMode, bool useJobs,
                            cyber::memory::Span<PipelineStats> stats) noexcept
{
    const auto Fail = [&stats](const char* message) noexcept
    {
        for (u32 index = 0; index < stats.Size(); ++index)
        {
            stats[index] = PipelineStats{};
        }
        return core::Status<core::Error>::Err(
            core::Error{core::ErrorCode::InvalidArgument, message});
    };

    if (cascades.count == 0 || cascades.count > kMaxDirectionalShadowCascades ||
        depthMaps.Size() < cascades.count ||
        (!stats.Empty() && stats.Size() < cascades.count) ||
        (vertices.Size() != 0 && vertices.Data() == nullptr))
    {
        return Fail("invalid directional cascade render targets or count");
    }

    cyber::core::i32 width = 0;
    cyber::core::i32 height = 0;
    f32 previousSplit = cascades.cascades[0].splitNear;
    for (u32 index = 0; index < cascades.count; ++index)
    {
        const DirectionalShadowCascade& cascade = cascades.cascades[index];
        DepthBuffer& depth = depthMaps[index];
        if (!depth.IsValid() || !IsFinite(cascade.shadow.clipFromWorld) ||
            !std::isfinite(cascade.splitNear) || !std::isfinite(cascade.splitFar) ||
            cascade.splitNear < 0.0f || cascade.splitFar <= cascade.splitNear ||
            (index != 0 && cascade.splitNear < previousSplit))
        {
            return Fail("invalid directional cascade description or render target");
        }
        if (index == 0)
        {
            width = depth.Width();
            height = depth.Height();
        }
        else if (depth.Width() != width || depth.Height() != height)
        {
            return Fail("directional cascade depth maps must have equal dimensions");
        }
        previousSplit = cascade.splitFar;
    }

    for (u32 index = 0; index < cascades.count; ++index)
    {
        cascades.cascades[index].shadow.depth = nullptr;
    }

    for (u32 index = 0; index < cascades.count; ++index)
    {
        ShadowRenderDesc render{};
        render.clipFromWorld = cascades.cascades[index].shadow.clipFromWorld;
        render.cullMode = cullMode;
        render.useJobs = useJobs;
        PipelineStats* cascadeStats = stats.Empty() ? nullptr : &stats[index];
        const core::Status<core::Error> rendered =
            RenderShadowMap(render, vertices, depthMaps[index], jobSystem, allocator,
                            cascadeStats);
        if (!rendered)
        {
            for (u32 reset = index + 1; reset < stats.Size(); ++reset)
            {
                stats[reset] = PipelineStats{};
            }
            return rendered;
        }
    }
    for (u32 index = 0; index < cascades.count; ++index)
    {
        cascades.cascades[index].shadow.depth = &depthMaps[index];
    }
    return core::Status<core::Error>::Ok();
}

}  // namespace cyber::software
