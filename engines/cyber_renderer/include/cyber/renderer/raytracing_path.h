#pragma once

#include <cyber/core/types.h>
#include <cyber/rhi/capabilities.h>
#include <cyber/rhi/device.h>

namespace cyber::renderer
{

using cyber::core::u32;
using cyber::core::u8;

/// Whether this build contains GPU implementations of the hardware ray-tracing
/// techniques.
///
/// Phase 31 (ADR 0028) reports ray-tracing capability from the adapter and
/// ships the selection policy, but the RHI has no acceleration-structure or
/// ray-dispatch objects yet. Until it does, a capable adapter still resolves to
/// the rasterized path, with `RayTracingFallbackReason::NotImplemented` saying
/// why, so a log line distinguishes "this GPU cannot" from "this build does
/// not". The value is a build constant.
inline constexpr bool kHardwareRayTracingCompiledIn = false;

/// Effects a ray-traced path can answer.
enum class RayTracedEffect : u8
{
    /// Rasterized answer: reverse-Z shadow maps with PCF/PCSS (Phase 28).
    Shadows = 0,
    /// Rasterized answer: screen-space reflections over reflection probes
    /// (Phase 29).
    Reflections,
    /// Rasterized answer: the baked irradiance volume (Phase 31).
    GlobalIllumination,
    /// Rasterized answer: screen-space ambient occlusion (Phase 30).
    AmbientOcclusion,
    /// Count of enumerators, for tables. Not an effect.
    Count
};

/// Number of real effects.
inline constexpr u32 kRayTracedEffectCount = static_cast<u32>(RayTracedEffect::Count);

/// The implementation that answers one effect.
enum class RayTracingTechnique : u8
{
    /// The rasterized or screen-space technique the effect always has.
    Rasterized = 0,
    /// A DXR ray-generation pipeline with shader tables (tier 1.0).
    HardwarePipeline,
    /// DXR 1.1 inline ray queries from a compute or pixel shader (tier 1.1).
    HardwareInline,
    /// The CPU ray tracer in the software renderer. Never chosen
    /// automatically: it is the reference the hardware paths are judged
    /// against, and is selected only by asking for it.
    SoftwareReference,
};

/// Why an effect did not get a hardware ray-traced technique. `None` means
/// the requested technique was selected.
enum class RayTracingFallbackReason : u8
{
    None = 0,
    /// The policy's preference is `Off`.
    PolicyDisabled,
    /// The effect is not enabled in the policy.
    EffectDisabled,
    /// The adapter does not report `rhi::Feature::RayTracing`, or reports
    /// `RayQuery` without it, which is treated as no support at all.
    Unsupported,
    /// The adapter is capable but this build has no GPU implementation; see
    /// `kHardwareRayTracingCompiledIn`.
    NotImplemented,
};

/// What the caller asks for.
enum class RayTracingPreference : u8
{
    /// Never ray trace. Every effect is rasterized.
    Off = 0,
    /// Use hardware ray tracing where the adapter supports it, preferring
    /// inline queries, which need no shader tables and suit the
    /// one-ray-per-pixel visibility work the first effects are.
    Auto,
    /// Use the ray-generation pipeline even when inline queries exist.
    PreferPipeline,
    /// Use inline queries where available, else the pipeline. Identical to
    /// `Auto` today; kept separate so `Auto` may grow heuristics later
    /// without changing what an explicit request means.
    PreferInline,
    /// Answer every enabled effect with the CPU reference tracer, whatever the
    /// adapter. For offline renders, validation, and machines without
    /// capable hardware.
    SoftwareReference,
};

/// Caller policy.
struct RayTracingPolicyDesc
{
    RayTracingPreference preference = RayTracingPreference::Auto;
    /// Per-effect opt-in, indexed by `RayTracedEffect`. The default enables
    /// shadows only, which is the effect the build guide says to start with.
    bool enabledEffects[kRayTracedEffectCount] = {true, false, false, false};
    /// Whether GPU implementations exist. Defaults to the build constant; a
    /// test sets it to exercise the capability gate the way a build with the
    /// GPU paths will.
    bool hardwarePathsAvailable = kHardwareRayTracingCompiledIn;
};

/// The decision for one effect.
struct RayTracingSelection
{
    RayTracingTechnique technique = RayTracingTechnique::Rasterized;
    RayTracingFallbackReason reason = RayTracingFallbackReason::None;
};

/// The decision for every effect, plus what the adapter reported.
struct RayTracingPlan
{
    RayTracingSelection effects[kRayTracedEffectCount]{};
    /// The adapter reports `rhi::Feature::RayTracing`.
    bool hardwareCapable = false;
    /// The adapter reports both `RayTracing` and `RayQuery`.
    bool inlineCapable = false;

    /// Decision for one effect. `RayTracedEffect::Count` or an out-of-range
    /// value returns a rasterized selection with `EffectDisabled`.
    [[nodiscard]] RayTracingSelection For(RayTracedEffect effect) const noexcept;

    /// True when at least one effect resolved to a hardware technique.
    [[nodiscard]] bool UsesHardwareRayTracing() const noexcept;
};

/// Resolve a policy against reported capabilities.
///
/// Pure: no device objects are touched, so the gate is unit tested with
/// fabricated capability sets on hardware that cannot ray trace.
///
/// @param capabilities What the device reports.
/// @param desc Caller policy.
/// @return A selection for every effect. A hardware technique is chosen only
///         when the effect is enabled, the preference allows it, the adapter
///         reports `Feature::RayTracing` (and `Feature::RayQuery` for the
///         inline technique), and `hardwarePathsAvailable` is set.
[[nodiscard]] RayTracingPlan SelectRayTracingPlan(const rhi::Capabilities& capabilities,
                                                  const RayTracingPolicyDesc& desc) noexcept;

/// Resolve a policy against a live device. An invalid device reports no
/// capabilities, so every effect falls back.
[[nodiscard]] RayTracingPlan SelectRayTracingPlan(const rhi::Device& device,
                                                  const RayTracingPolicyDesc& desc) noexcept;

/// Static, human-readable names for logs. Out-of-range values return
/// "unknown".
[[nodiscard]] const char* RayTracedEffectName(RayTracedEffect effect) noexcept;
[[nodiscard]] const char* RayTracingTechniqueName(RayTracingTechnique technique) noexcept;
[[nodiscard]] const char* RayTracingFallbackReasonName(RayTracingFallbackReason reason) noexcept;

}  // namespace cyber::renderer
