#include <cyber/rhi/vulkan.h>

#if defined(CYBER_RHI_HAS_VULKAN)

#include "vulkan_context.h"
#include "vulkan_device.h"

#include <cyber/rhi/backend.h>
#include <cyber/rhi/capabilities.h>
#include <cyber/rhi/device.h>

namespace cyber::rhi
{

const Backend* VulkanBackend() noexcept
{
    // Static and immutable after construction, like the D3D12 backend: adapters
    // are gathered per call and nothing is cached between calls.
    static const Backend backend = []() noexcept
    {
        Backend value{};
        value.kind = BackendKind::Vulkan;
        value.name = "vulkan";
        value.apiVersion = (1U * 100U) + 3U;
        value.interface = detail::VulkanDeviceInterface();
        value.enumerateAdapters = &detail::EnumerateVulkanAdapters;
        value.createDevice = &detail::CreateVulkanDevice;
        value.destroyDevice = &detail::DestroyVulkanDevice;
        return value;
    }();
    return &backend;
}

}  // namespace cyber::rhi

#else

namespace cyber::rhi
{

const Backend* VulkanBackend() noexcept
{
    return nullptr;
}

}  // namespace cyber::rhi

#endif
