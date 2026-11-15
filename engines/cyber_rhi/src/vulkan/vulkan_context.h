#pragma once

#include "vk_api.h"
#include "vk_api_ext.h"

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/memory/allocator.h>
#include <cyber/rhi/capabilities.h>

namespace cyber::rhi::detail
{

using cyber::core::Error;
using cyber::core::Result;
using cyber::core::Status;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::usize;

// The Vulkan version this backend requires. Dynamic rendering, synchronization2
// and timeline semaphores are core in 1.3, so a device below it is listed by
// adapter enumeration (a caller should see that the machine has a GPU) but is
// refused at device creation with `ErrorCode::Unsupported`.
inline constexpr u32 kVulkanRequiredApiVersion = vk::kApiVersion13;

// The most physical devices the backend lists. A machine with more is truncated,
// never overflowed.
inline constexpr u32 kMaxVulkanAdapters = 16;

inline constexpr u32 kNoQueueFamily = 0xFFFFFFFFU;

// Queue family indices the backend selected. Dedicated families are used when
// the device has them; otherwise the compute and transfer indices repeat the
// graphics family, which every Vulkan device is required to have.
struct VulkanQueueFamilies
{
    u32 graphics = kNoQueueFamily;
    u32 compute = kNoQueueFamily;
    u32 transfer = kNoQueueFamily;
    bool dedicatedCompute = false;
    bool dedicatedTransfer = false;
};

// What the context learned about the device it created.
struct VulkanDeviceInfo
{
    char name[kAdapterNameBytes] = {};
    u32 vendorId = 0;
    u32 deviceId = 0;
    u32 driverVersion = 0;
    u32 apiVersion = 0;
    vk::PhysicalDeviceType type = vk::PhysicalDeviceType::Other;
    u64 deviceLocalBytes = 0;
    u64 hostVisibleBytes = 0;
    VulkanQueueFamilies queues{};
    bool validationLayerEnabled = false;
    bool synchronizationValidationEnabled = false;
    bool debugUtilsEnabled = false;
    bool bufferDeviceAddress = false;
    bool presentation = false;
    // Core 1.0 features the RHI device uses, enabled when the hardware has them.
    bool samplerAnisotropy = false;
    bool textureCompressionBc = false;
    bool multiDrawIndirect = false;
    bool shaderStorageImageExtendedFormats = false;
    u32 maxPushConstantBytes = 128;
    u32 maxImageDimension1D = 4096;
    u32 maxImageDimension2D = 4096;
    u32 maxImageDimension3D = 256;
    u32 maxImageDimensionCube = 4096;
    u32 maxImageArrayLayers = 256;
    bool drawIndirectCount = false;
};

struct VulkanContextDesc
{
    // Index into the list `EnumerateVulkanAdapters` returns.
    u32 adapterIndex = 0;
    // Loads the Khronos validation layer, with synchronization validation, when
    // it is installed. A machine without it runs unvalidated.
    bool enableValidation = true;
    // Turns "validation layer not installed" into an error. Tests and CI set it
    // so a missing layer cannot pass as a clean run.
    bool requireValidation = false;
    // Enables VK_KHR_surface, the platform surface, and VK_KHR_swapchain when the
    // machine has them. A context without them still renders offscreen.
    bool enablePresentation = true;
};

// Lists the Vulkan physical devices in the order `VulkanContext` indexes them:
// discrete GPUs, integrated GPUs, virtual GPUs, anything else, and software
// implementations last, so "first adapter" is real hardware and "last adapter"
// is the one CI can rely on. Follows the `Backend::enumerateAdapters` contract:
// writes what fits and returns the total. Returns 0 when no Vulkan loader is
// installed. Creates and destroys one instance per call and keeps no state.
[[nodiscard]] u32 EnumerateVulkanAdapters(AdapterInfo* out, u32 capacity) noexcept;

// Owns the loader, the instance, the validation messenger, and the logical
// device with its queues. This is the layer under the RHI device: it is created,
// validated, and torn down on its own, so the pieces above it (resources,
// pipelines, command lists, swapchains) are built on something already tested.
class VulkanContext
{
public:
    VulkanContext() noexcept = default;
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    VulkanContext(VulkanContext&& other) noexcept;
    VulkanContext& operator=(VulkanContext&& other) noexcept;
    ~VulkanContext();

    // Creates a context on adapter `desc.adapterIndex`.
    //
    // `ErrorCode::NotFound` when no loader is installed, or validation is
    // required and absent. `ErrorCode::OutOfRange` for an adapter index that does
    // not exist. `ErrorCode::Unsupported` when the device is below Vulkan 1.3 or
    // lacks a required feature, with the missing feature named in the message.
    [[nodiscard]] static Result<VulkanContext, Error> Create(const VulkanContextDesc& desc,
                                                             memory::Allocator& allocator) noexcept;

    [[nodiscard]] bool IsValid() const noexcept
    {
        return m_device != nullptr;
    }
    [[nodiscard]] const VulkanDeviceInfo& Info() const noexcept
    {
        return m_info;
    }

    [[nodiscard]] vk::Instance Instance() const noexcept
    {
        return m_instance;
    }
    [[nodiscard]] vk::PhysicalDevice PhysicalDevice() const noexcept
    {
        return m_physicalDevice;
    }
    [[nodiscard]] vk::Device Device() const noexcept
    {
        return m_device;
    }
    [[nodiscard]] const vk::DeviceApi& DeviceFunctions() const noexcept
    {
        return m_deviceApi;
    }
    // Resolves one device-level command by name, for optional entry points (such as
    // VK_EXT_debug_utils labels) that are not part of the required function tables.
    [[nodiscard]] vk::PFN_VoidFunction ResolveDeviceFunction(const char* name) const noexcept
    {
        return m_instanceApi.getDeviceProcAddr != nullptr
                   ? m_instanceApi.getDeviceProcAddr(m_device, name)
                   : nullptr;
    }
    [[nodiscard]] const vk::DeviceApiExt& DeviceFunctionsExt() const noexcept
    {
        return m_deviceExt;
    }
    [[nodiscard]] const vk::SurfaceApi& SurfaceFunctions() const noexcept
    {
        return m_surfaceApi;
    }
    [[nodiscard]] const vk::SwapchainApi& SwapchainFunctions() const noexcept
    {
        return m_swapchainApi;
    }
    [[nodiscard]] const vk::PhysicalDeviceMemoryProperties& MemoryProperties() const noexcept
    {
        return m_memoryProperties;
    }

    // Queue handles. Compute and transfer alias the graphics queue when the
    // device has no dedicated family for them.
    [[nodiscard]] vk::Queue GraphicsQueue() const noexcept
    {
        return m_graphicsQueue;
    }
    [[nodiscard]] vk::Queue ComputeQueue() const noexcept
    {
        return m_computeQueue;
    }
    [[nodiscard]] vk::Queue TransferQueue() const noexcept
    {
        return m_transferQueue;
    }

    // Validation messages seen since creation. A test treats any error as a
    // failure; warnings are counted separately so a test can choose.
    [[nodiscard]] u32 ValidationErrorCount() const noexcept;
    [[nodiscard]] u32 ValidationWarningCount() const noexcept;

    // Copies the first error message into `buffer` (null-terminated, truncated to
    // fit) and returns its length, or 0 when there has been none.
    [[nodiscard]] u32 FirstValidationError(char* buffer, u32 capacity) const noexcept;

    // Attaches a name a capture tool and the validation layer show for an object.
    // Succeeds without effect when VK_EXT_debug_utils is not enabled, so callers
    // name unconditionally.
    Status<Error> SetObjectName(vk::ObjectType type, u64 handle, const char* name) noexcept;

    // Blocks until every queue is idle.
    Status<Error> WaitIdle() noexcept;

private:
    struct ValidationState;
    struct Loader;

    void Release() noexcept;

    memory::Allocator* m_allocator = nullptr;
    Loader* m_loader = nullptr;
    ValidationState* m_validation = nullptr;
    vk::InstanceApi m_instanceApi{};
    vk::DeviceApi m_deviceApi{};
    vk::PhysicalDeviceMemoryProperties m_memoryProperties{};
    vk::DeviceApiExt m_deviceExt{};
    vk::SurfaceApi m_surfaceApi{};
    vk::SwapchainApi m_swapchainApi{};
    vk::Instance m_instance = nullptr;
    vk::PhysicalDevice m_physicalDevice = nullptr;
    vk::Device m_device = nullptr;
    vk::Queue m_graphicsQueue = nullptr;
    vk::Queue m_computeQueue = nullptr;
    vk::Queue m_transferQueue = nullptr;
    vk::DebugUtilsMessengerEXT m_messenger = 0;
    VulkanDeviceInfo m_info{};
};

}  // namespace cyber::rhi::detail
