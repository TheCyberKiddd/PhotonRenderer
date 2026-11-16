#include "test_backend.h"

#include <cyber/memory/heap_allocator.h>
#include <cyber/renderer/raytracing_path.h>
#include <cyber/rhi/backend.h>
#include <cyber/rhi/d3d12.h>
#include <cyber/rhi/device.h>
#include <cyber/test/test.h>

#include <cstdio>
#include <cstring>
#include <utility>

using namespace cyber::renderer;
using cyber::core::u32;
using cyber::rhi::Capabilities;
using cyber::rhi::Feature;

namespace
{

Capabilities WithFeatures(std::initializer_list<Feature> features)
{
    Capabilities capabilities{};
    for (const Feature feature : features)
    {
        capabilities.features.Add(feature);
    }
    return capabilities;
}

RayTracingPolicyDesc AllEffects(RayTracingPreference preference, bool hardwarePaths)
{
    RayTracingPolicyDesc desc{};
    desc.preference = preference;
    desc.hardwarePathsAvailable = hardwarePaths;
    for (bool& enabled : desc.enabledEffects)
    {
        enabled = true;
    }
    return desc;
}

void ExpectEvery(const RayTracingPlan& plan, RayTracingTechnique technique,
                 RayTracingFallbackReason reason)
{
    for (u32 index = 0; index < kRayTracedEffectCount; ++index)
    {
        const RayTracingSelection selection = plan.For(static_cast<RayTracedEffect>(index));
        CG_EXPECT_EQ(selection.technique, technique);
        CG_EXPECT_EQ(selection.reason, reason);
    }
}

}  // namespace

// The done-when line: the ray-traced path switches on only where the hardware
// supports it. With GPU paths available, a capability set without the feature
// falls back and one with it does not.
CG_TEST(renderer_raytracing_switches_on_only_when_supported)
{
    const RayTracingPolicyDesc desc = AllEffects(RayTracingPreference::Auto, true);

    const RayTracingPlan none = SelectRayTracingPlan(WithFeatures({Feature::Compute}), desc);
    CG_EXPECT_FALSE(none.hardwareCapable);
    CG_EXPECT_FALSE(none.UsesHardwareRayTracing());
    ExpectEvery(none, RayTracingTechnique::Rasterized, RayTracingFallbackReason::Unsupported);

    const RayTracingPlan tier10 =
        SelectRayTracingPlan(WithFeatures({Feature::Compute, Feature::RayTracing}), desc);
    CG_EXPECT_TRUE(tier10.hardwareCapable);
    CG_EXPECT_FALSE(tier10.inlineCapable);
    CG_EXPECT_TRUE(tier10.UsesHardwareRayTracing());
    ExpectEvery(tier10, RayTracingTechnique::HardwarePipeline, RayTracingFallbackReason::None);

    const RayTracingPlan tier11 = SelectRayTracingPlan(
        WithFeatures({Feature::Compute, Feature::RayTracing, Feature::RayQuery}), desc);
    CG_EXPECT_TRUE(tier11.inlineCapable);
    ExpectEvery(tier11, RayTracingTechnique::HardwareInline, RayTracingFallbackReason::None);
}

CG_TEST(renderer_raytracing_preferences)
{
    const Capabilities tier11 = WithFeatures({Feature::RayTracing, Feature::RayQuery});

    ExpectEvery(SelectRayTracingPlan(tier11, AllEffects(RayTracingPreference::Off, true)),
                RayTracingTechnique::Rasterized, RayTracingFallbackReason::PolicyDisabled);
    ExpectEvery(
        SelectRayTracingPlan(tier11, AllEffects(RayTracingPreference::PreferPipeline, true)),
        RayTracingTechnique::HardwarePipeline, RayTracingFallbackReason::None);
    ExpectEvery(SelectRayTracingPlan(tier11, AllEffects(RayTracingPreference::PreferInline, true)),
                RayTracingTechnique::HardwareInline, RayTracingFallbackReason::None);

    // PreferInline on a tier 1.0 adapter takes the pipeline rather than
    // nothing.
    const Capabilities tier10 = WithFeatures({Feature::RayTracing});
    ExpectEvery(SelectRayTracingPlan(tier10, AllEffects(RayTracingPreference::PreferInline, true)),
                RayTracingTechnique::HardwarePipeline, RayTracingFallbackReason::None);

    // The software reference answers whatever the adapter, and is never what
    // Auto chooses.
    ExpectEvery(SelectRayTracingPlan(Capabilities{},
                                     AllEffects(RayTracingPreference::SoftwareReference, false)),
                RayTracingTechnique::SoftwareReference, RayTracingFallbackReason::None);
    const RayTracingPlan software = SelectRayTracingPlan(
        Capabilities{}, AllEffects(RayTracingPreference::SoftwareReference, false));
    CG_EXPECT_FALSE(software.UsesHardwareRayTracing());
}

CG_TEST(renderer_raytracing_effects_are_opt_in)
{
    const Capabilities tier11 = WithFeatures({Feature::RayTracing, Feature::RayQuery});
    RayTracingPolicyDesc desc{};
    desc.hardwarePathsAvailable = true;
    // The default enables shadows only.
    const RayTracingPlan plan = SelectRayTracingPlan(tier11, desc);
    // Selections are held in named values: CG_EXPECT_EQ binds a reference to
    // its argument, and For() returns by value.
    const RayTracingSelection shadows = plan.For(RayTracedEffect::Shadows);
    const RayTracingSelection reflections = plan.For(RayTracedEffect::Reflections);
    const RayTracingSelection gi = plan.For(RayTracedEffect::GlobalIllumination);
    const RayTracingSelection occlusion = plan.For(RayTracedEffect::AmbientOcclusion);
    CG_EXPECT_EQ(shadows.technique, RayTracingTechnique::HardwareInline);
    CG_EXPECT_EQ(reflections.technique, RayTracingTechnique::Rasterized);
    CG_EXPECT_EQ(reflections.reason, RayTracingFallbackReason::EffectDisabled);
    CG_EXPECT_EQ(gi.technique, RayTracingTechnique::Rasterized);
    CG_EXPECT_EQ(occlusion.technique, RayTracingTechnique::Rasterized);

    // An out-of-range effect is a rasterized, disabled answer, not a read past
    // the table.
    const RayTracingSelection sentinel = plan.For(RayTracedEffect::Count);
    const RayTracingSelection outOfRange = plan.For(static_cast<RayTracedEffect>(200));
    CG_EXPECT_EQ(sentinel.technique, RayTracingTechnique::Rasterized);
    CG_EXPECT_EQ(outOfRange.reason, RayTracingFallbackReason::EffectDisabled);
}

CG_TEST(renderer_raytracing_malformed_capabilities_are_not_trusted)
{
    // RayQuery without RayTracing cannot come from D3D12; the policy treats it
    // as no support rather than picking an inline path with no pipeline.
    const RayTracingPlan plan = SelectRayTracingPlan(WithFeatures({Feature::RayQuery}),
                                                     AllEffects(RayTracingPreference::Auto, true));
    CG_EXPECT_FALSE(plan.hardwareCapable);
    CG_EXPECT_FALSE(plan.inlineCapable);
    ExpectEvery(plan, RayTracingTechnique::Rasterized, RayTracingFallbackReason::Unsupported);
}

// This build has no GPU implementation, so a capable adapter is reported as
// capable and still rasterizes, with a reason that says which of the two is
// missing.
CG_TEST(renderer_raytracing_build_without_gpu_paths_falls_back)
{
    CG_EXPECT_FALSE(kHardwareRayTracingCompiledIn);
    RayTracingPolicyDesc desc =
        AllEffects(RayTracingPreference::Auto, kHardwareRayTracingCompiledIn);
    const RayTracingPlan plan =
        SelectRayTracingPlan(WithFeatures({Feature::RayTracing, Feature::RayQuery}), desc);
    CG_EXPECT_TRUE(plan.hardwareCapable);
    CG_EXPECT_FALSE(plan.UsesHardwareRayTracing());
    ExpectEvery(plan, RayTracingTechnique::Rasterized, RayTracingFallbackReason::NotImplemented);

    // The default descriptor carries the build constant.
    const RayTracingPolicyDesc defaults{};
    CG_EXPECT_EQ(defaults.hardwarePathsAvailable, kHardwareRayTracingCompiledIn);
}

CG_TEST(renderer_raytracing_names)
{
    for (u32 index = 0; index < kRayTracedEffectCount; ++index)
    {
        CG_EXPECT_TRUE(
            std::strcmp(RayTracedEffectName(static_cast<RayTracedEffect>(index)), "unknown") != 0);
    }
    CG_EXPECT_STREQ(RayTracedEffectName(RayTracedEffect::Count), "unknown");
    CG_EXPECT_STREQ(RayTracingTechniqueName(RayTracingTechnique::HardwareInline),
                    "hardware_inline");
    CG_EXPECT_STREQ(RayTracingTechniqueName(static_cast<RayTracingTechnique>(99)), "unknown");
    CG_EXPECT_STREQ(RayTracingFallbackReasonName(RayTracingFallbackReason::NotImplemented),
                    "not_implemented");
    CG_EXPECT_STREQ(RayTracingFallbackReasonName(static_cast<RayTracingFallbackReason>(99)),
                    "unknown");
    CG_EXPECT_STREQ(cyber::rhi::FeatureName(Feature::RayTracing), "ray_tracing");
    CG_EXPECT_STREQ(cyber::rhi::FeatureName(Feature::RayQuery), "ray_query");
}

// The null backend is a validation device with no ray-tracing objects, so it
// must never claim the feature, and an invalid device reports nothing.
CG_TEST(renderer_raytracing_null_device_falls_back)
{
    cyber::memory::HeapAllocator heap;
    cyber::rhi::DeviceDesc deviceDesc{};
    deviceDesc.backend = cyber::rhi::NullBackend();
    auto created = cyber::rhi::Device::Create(deviceDesc, heap);
    CG_REQUIRE(created.IsOk());
    cyber::rhi::Device device = std::move(created.Value());
    const RayTracingPlan plan =
        SelectRayTracingPlan(device, AllEffects(RayTracingPreference::Auto, true));
    CG_EXPECT_FALSE(plan.hardwareCapable);
    ExpectEvery(plan, RayTracingTechnique::Rasterized, RayTracingFallbackReason::Unsupported);

    const cyber::rhi::Device invalid{};
    ExpectEvery(SelectRayTracingPlan(invalid, AllEffects(RayTracingPreference::Auto, true)),
                RayTracingTechnique::Rasterized, RayTracingFallbackReason::Unsupported);
}

// Every D3D12 adapter on this machine: whatever it reports, the reported set
// must be self-consistent and the plan must agree with it. On an adapter that
// cannot ray trace -- the development machine's Intel HD Graphics 530 and the
// WARP software adapter both report tier "not supported" -- every effect
// rasterizes.
CG_TEST(renderer_raytracing_d3d12_adapters_report_consistently)
{
    const cyber::rhi::Backend* backend = cyber::rhi::test::TestBackend();
    if (backend == nullptr)
    {
        CG_SKIP("this build does not contain the D3D12 backend");
    }
    const u32 adapterCount = backend->enumerateAdapters(nullptr, 0);
    if (adapterCount == 0)
    {
        CG_SKIP("no D3D12 adapter is available");
    }
    cyber::memory::HeapAllocator heap;
    u32 opened = 0;
    for (u32 adapter = 0; adapter < adapterCount; ++adapter)
    {
        cyber::rhi::DeviceDesc desc{};
        desc.backend = backend;
        desc.adapterIndex = adapter;
        auto created = cyber::rhi::Device::Create(desc, heap);
        if (created.IsError())
        {
            continue;
        }
        ++opened;
        cyber::rhi::Device device = std::move(created.Value());
        const Capabilities capabilities = device.Capabilities();
        const bool rayTracing = capabilities.features.Has(Feature::RayTracing);
        const bool rayQuery = capabilities.features.Has(Feature::RayQuery);
        CG_EXPECT_TRUE(rayTracing || !rayQuery);

        const RayTracingPlan plan =
            SelectRayTracingPlan(device, AllEffects(RayTracingPreference::Auto, true));
        CG_EXPECT_EQ(plan.hardwareCapable, rayTracing);
        CG_EXPECT_EQ(plan.UsesHardwareRayTracing(), rayTracing);
        std::printf("[raytracing] adapter %u \"%s\": ray_tracing=%d ray_query=%d -> shadows %s\n",
                    adapter, capabilities.adapter.name, rayTracing ? 1 : 0, rayQuery ? 1 : 0,
                    RayTracingTechniqueName(plan.For(RayTracedEffect::Shadows).technique));

        // What actually runs in this build: rasterized everywhere.
        const RayTracingPlan shipped = SelectRayTracingPlan(device, RayTracingPolicyDesc{});
        CG_EXPECT_FALSE(shipped.UsesHardwareRayTracing());
    }
    if (opened == 0)
    {
        CG_SKIP("no D3D12 adapter could create a device");
    }
}
