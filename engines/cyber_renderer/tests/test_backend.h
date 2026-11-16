#pragma once

// The hardware backend a renderer GPU test runs on: D3D12 unless CYBER_RHI_BACKEND
// names another. CMake registers each GPU test a second time as `<name>_vulkan`
// with the variable set (cyber_add_vulkan_variant), so the same test body runs
// unchanged on D3D12 and on Vulkan.

#include <cyber/rhi/backend.h>
#include <cyber/rhi/backend_select.h>
#include <cyber/rhi/d3d12.h>

namespace cyber::rhi::test
{

[[nodiscard]] inline const Backend* TestBackend() noexcept
{
    return SelectBackend(0, nullptr, D3D12Backend());
}

}  // namespace cyber::rhi::test
