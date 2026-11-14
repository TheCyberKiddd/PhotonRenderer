#include <cyber/software/clustered_pbr.h>

#include <cmath>

namespace cyber::software
{

core::Status<core::Error>
BuildClusterLightBounds(cyber::memory::Span<const PbrPointLight> pointLights,
                        const math::Mat4& viewFromWorld,
                        cyber::memory::Span<cyber::renderer::ClusterLightBound> output) noexcept
{
    using cyber::core::Error;
    using cyber::core::ErrorCode;
    using cyber::renderer::ClusterLightBound;

    if (output.Size() != pointLights.Size() ||
        (output.Size() != 0 && (pointLights.Data() == nullptr || output.Data() == nullptr)))
    {
        return core::Status<Error>::Err(
            Error{ErrorCode::InvalidArgument, "clustered PBR: invalid light-bound spans"});
    }

    for (core::usize index = 0; index < pointLights.Size(); ++index)
    {
        const PbrPointLight& light = pointLights[index];
        const math::Vec3 viewPosition = viewFromWorld.TransformPoint(light.position);
        if (!math::IsFinite(viewPosition.x) || !math::IsFinite(viewPosition.y) ||
            !math::IsFinite(viewPosition.z) || !math::IsFinite(light.range) || light.range <= 0.0f)
        {
            return core::Status<Error>::Err(
                Error{ErrorCode::InvalidArgument, "clustered PBR: invalid point-light bound"});
        }
    }

    for (core::usize index = 0; index < pointLights.Size(); ++index)
    {
        const PbrPointLight& light = pointLights[index];
        const math::Vec3 viewPosition = viewFromWorld.TransformPoint(light.position);
        output[index] =
            ClusterLightBound{viewPosition.x, viewPosition.y, -viewPosition.z, light.range};
    }
    return core::Status<Error>::Ok();
}

math::Color ClusteredPbrPixelShader(const Varyings& varyings, const void* userData) noexcept
{
    const auto* clustered = static_cast<const ClusteredPbrLightingData*>(userData);
    if (clustered == nullptr || clustered->lighting == nullptr)
    {
        return varyings.color;
    }

    PbrLightingData lighting = *clustered->lighting;
    lighting.pointLightSpan = clustered->pointLights;
    lighting.usePointLightIndices = false;

    if (clustered->clusters != nullptr)
    {
        const math::Vec3 viewPosition =
            clustered->viewFromWorld.TransformPoint(varyings.worldPosition);
        const cyber::core::u32 clusterIndex = clustered->clusters->ClusterAtPixel(
            varyings.pixelPosition.x, varyings.pixelPosition.y, -viewPosition.z);
        if (clusterIndex != cyber::renderer::kInvalidClusterIndex)
        {
            const auto indices = clustered->clusters->Lights(clusterIndex);
            bool validIndices = true;
            for (core::usize index = 0; index < indices.Size(); ++index)
            {
                if (indices[index] >= clustered->pointLights.Size())
                {
                    validIndices = false;
                    break;
                }
            }
            if (validIndices)
            {
                lighting.pointLightIndices = indices;
                lighting.usePointLightIndices = true;
            }
        }
    }
    return PbrPixelShader(varyings, &lighting);
}

}  // namespace cyber::software
