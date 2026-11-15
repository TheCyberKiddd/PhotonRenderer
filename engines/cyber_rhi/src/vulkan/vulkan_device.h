#pragma once

// The Vulkan implementation of the RHI `DeviceInterface` (ADR 0013). Phase 34
// increment 6 (ADR 0029): resources, samplers, shaders, fences, queues and device
// reporting are implemented on the layers of increments 1 to 5. Bind groups,
// pipelines, command lists, submission and swapchains are the next increment; their
// table entries exist and answer `ErrorCode::Unsupported` with a message that says
// so, so `Device::Create` accepts the backend and a caller learns exactly what is
// missing.

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/memory/allocator.h>
#include <cyber/rhi/backend.h>
#include <cyber/rhi/descriptors.h>

#if defined(CYBER_RHI_HAS_VULKAN)

namespace cyber::rhi::detail
{

[[nodiscard]] const DeviceInterface* VulkanDeviceInterface() noexcept;

[[nodiscard]] cyber::core::Result<DeviceCreation, cyber::core::Error>
CreateVulkanDevice(const DeviceDesc& desc, memory::Allocator& allocator) noexcept;

void DestroyVulkanDevice(void* internal) noexcept;

}  // namespace cyber::rhi::detail

#endif
