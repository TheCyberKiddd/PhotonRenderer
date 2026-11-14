#include <cyber/renderer/raytracing_path.h>

namespace cyber::renderer
{

namespace
{

RayTracingSelection Rasterized(RayTracingFallbackReason reason) noexcept
{
    RayTracingSelection selection{};
    selection.technique = RayTracingTechnique::Rasterized;
    selection.reason = reason;
    return selection;
}

RayTracingSelection Selected(RayTracingTechnique technique) noexcept
{
    RayTracingSelection selection{};
    selection.technique = technique;
    selection.reason = RayTracingFallbackReason::None;
    return selection;
}

}  // namespace

RayTracingSelection RayTracingPlan::For(RayTracedEffect effect) const noexcept
{
    const u32 index = static_cast<u32>(effect);
    if (index >= kRayTracedEffectCount)
    {
        return Rasterized(RayTracingFallbackReason::EffectDisabled);
    }
    return effects[index];
}

bool RayTracingPlan::UsesHardwareRayTracing() const noexcept
{
    for (const RayTracingSelection& selection : effects)
    {
        if (selection.technique == RayTracingTechnique::HardwarePipeline ||
            selection.technique == RayTracingTechnique::HardwareInline)
        {
            return true;
        }
    }
    return false;
}

RayTracingPlan SelectRayTracingPlan(const rhi::Capabilities& capabilities,
                                    const RayTracingPolicyDesc& desc) noexcept
{
    RayTracingPlan plan{};
    // RayQuery without RayTracing is not a configuration D3D12 can report, so
    // a capability set claiming it is malformed and is trusted for neither.
    const bool rayTracing = capabilities.features.Has(rhi::Feature::RayTracing);
    const bool rayQuery = rayTracing && capabilities.features.Has(rhi::Feature::RayQuery);
    plan.hardwareCapable = rayTracing;
    plan.inlineCapable = rayQuery;

    for (u32 index = 0; index < kRayTracedEffectCount; ++index)
    {
        RayTracingSelection& selection = plan.effects[index];
        if (desc.preference == RayTracingPreference::Off)
        {
            selection = Rasterized(RayTracingFallbackReason::PolicyDisabled);
            continue;
        }
        if (!desc.enabledEffects[index])
        {
            selection = Rasterized(RayTracingFallbackReason::EffectDisabled);
            continue;
        }
        if (desc.preference == RayTracingPreference::SoftwareReference)
        {
            selection = Selected(RayTracingTechnique::SoftwareReference);
            continue;
        }
        if (!rayTracing)
        {
            selection = Rasterized(RayTracingFallbackReason::Unsupported);
            continue;
        }
        if (!desc.hardwarePathsAvailable)
        {
            selection = Rasterized(RayTracingFallbackReason::NotImplemented);
            continue;
        }
        const bool wantInline = desc.preference != RayTracingPreference::PreferPipeline;
        selection = Selected(wantInline && rayQuery ? RayTracingTechnique::HardwareInline
                                                    : RayTracingTechnique::HardwarePipeline);
    }
    return plan;
}

RayTracingPlan SelectRayTracingPlan(const rhi::Device& device,
                                    const RayTracingPolicyDesc& desc) noexcept
{
    const rhi::Capabilities capabilities =
        device.IsValid() ? device.Capabilities() : rhi::Capabilities{};
    return SelectRayTracingPlan(capabilities, desc);
}

const char* RayTracedEffectName(RayTracedEffect effect) noexcept
{
    switch (effect)
    {
    case RayTracedEffect::Shadows:
        return "shadows";
    case RayTracedEffect::Reflections:
        return "reflections";
    case RayTracedEffect::GlobalIllumination:
        return "global_illumination";
    case RayTracedEffect::AmbientOcclusion:
        return "ambient_occlusion";
    case RayTracedEffect::Count:
        break;
    }
    return "unknown";
}

const char* RayTracingTechniqueName(RayTracingTechnique technique) noexcept
{
    switch (technique)
    {
    case RayTracingTechnique::Rasterized:
        return "rasterized";
    case RayTracingTechnique::HardwarePipeline:
        return "hardware_pipeline";
    case RayTracingTechnique::HardwareInline:
        return "hardware_inline";
    case RayTracingTechnique::SoftwareReference:
        return "software_reference";
    }
    return "unknown";
}

const char* RayTracingFallbackReasonName(RayTracingFallbackReason reason) noexcept
{
    switch (reason)
    {
    case RayTracingFallbackReason::None:
        return "none";
    case RayTracingFallbackReason::PolicyDisabled:
        return "policy_disabled";
    case RayTracingFallbackReason::EffectDisabled:
        return "effect_disabled";
    case RayTracingFallbackReason::Unsupported:
        return "unsupported";
    case RayTracingFallbackReason::NotImplemented:
        return "not_implemented";
    }
    return "unknown";
}

}  // namespace cyber::renderer
