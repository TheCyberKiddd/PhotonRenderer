#pragma once

#include <cyber/core/types.h>

namespace cyber::rhi
{

struct Backend;

// Whether this build contains the Vulkan backend.
//
// The backend needs no SDK at build time, because it declares the Vulkan API
// subset it uses and loads the platform's Vulkan runtime when it is first asked
// for an adapter (ADR 0029). A build can therefore contain it on a machine that
// has no GPU driver; whether a device exists is a runtime question the adapter
// list answers. The value is a build constant, safe to read from any thread.
[[nodiscard]] constexpr bool IsVulkanCompiledIn() noexcept
{
#if defined(CYBER_RHI_HAS_VULKAN)
    return true;
#else
    return false;
#endif
}

// Returns the Vulkan backend, or null when this build does not contain it.
//
// Adapter enumeration is a pure function of the machine, as for Direct3D 12: each
// call loads the runtime, creates one instance, lists the physical devices, and
// releases everything, so the backend keeps no global state. Adapters are listed
// discrete GPU first and software implementations last. A machine with no Vulkan
// runtime reports zero adapters rather than failing.
//
// `Backend::createDevice` creates a complete RHI device: resources, bind groups,
// pipelines, command lists, submission, fences and swapchains. Shaders must be SPIR-V.
// Timestamp queries, pipeline caches and reserved textures answer
// `ErrorCode::Unsupported` by name; `docs/design/phase-34-vulkan.md` lists the gaps.
//
// Returns a pointer with static lifetime, or null when Vulkan is not part of
// this build.
[[nodiscard]] const Backend* VulkanBackend() noexcept;

}  // namespace cyber::rhi
