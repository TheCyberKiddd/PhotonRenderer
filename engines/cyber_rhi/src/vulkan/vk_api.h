#pragma once

// The subset of the Vulkan 1.3 API this backend uses, declared here rather than
// taken from the Khronos headers. ADR 0029 records the decision: the loader
// (`vulkan-1.dll`, `libvulkan.so.1`) is a driver component like `d3d12.dll`, the
// declarations are a transcription of the specification, and nothing is linked
// at build time, so the backend compiles on a machine with no Vulkan SDK and
// is skipped at runtime on one with no driver.
//
// Every value below is the one the specification assigns. A declaration is added
// when code needs it, not speculatively, and a structure whose full layout the
// backend never reads is declared with an opaque tail whose size is an upper
// bound, so a driver can never write past it.

#include <cyber/core/types.h>

#if defined(_WIN32)
#define CYBER_VKAPI __stdcall
#else
#define CYBER_VKAPI
#endif

namespace cyber::rhi::detail::vk
{

using cyber::core::i32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::u8;
using cyber::core::usize;

using Bool32 = u32;
using Flags = u32;
using DeviceSize = u64;

inline constexpr Bool32 kTrue = 1;
inline constexpr Bool32 kFalse = 0;

[[nodiscard]] constexpr u32 MakeVersion(u32 variant, u32 major, u32 minor, u32 patch) noexcept
{
    return (variant << 29U) | (major << 22U) | (minor << 12U) | patch;
}
[[nodiscard]] constexpr u32 VersionMajor(u32 version) noexcept
{
    return (version >> 22U) & 0x7FU;
}
[[nodiscard]] constexpr u32 VersionMinor(u32 version) noexcept
{
    return (version >> 12U) & 0x3FFU;
}

inline constexpr u32 kApiVersion11 = MakeVersion(0, 1, 1, 0);
inline constexpr u32 kApiVersion13 = MakeVersion(0, 1, 3, 0);

// Opaque handles. Dispatchable handles are pointers, non-dispatchable ones are
// 64-bit integers on every platform this backend targets.
struct InstanceT;
struct PhysicalDeviceT;
struct DeviceT;
struct QueueT;
using Instance = InstanceT*;
using PhysicalDevice = PhysicalDeviceT*;
using Device = DeviceT*;
using Queue = QueueT*;
using DebugUtilsMessengerEXT = u64;
using DeviceMemory = u64;
using Buffer = u64;

enum class Result : i32
{
    Success = 0,
    NotReady = 1,
    Timeout = 2,
    Incomplete = 5,
    ErrorOutOfHostMemory = -1,
    ErrorOutOfDeviceMemory = -2,
    ErrorInitializationFailed = -3,
    ErrorDeviceLost = -4,
    ErrorMemoryMapFailed = -5,
    ErrorLayerNotPresent = -6,
    ErrorExtensionNotPresent = -7,
    ErrorFeatureNotPresent = -8,
    ErrorIncompatibleDriver = -9,
    ErrorTooManyObjects = -10,
    ErrorFormatNotSupported = -11,
};

enum class StructureType : u32
{
    ApplicationInfo = 0,
    InstanceCreateInfo = 1,
    DeviceQueueCreateInfo = 2,
    DeviceCreateInfo = 3,
    PhysicalDeviceVulkan11Features = 49,
    MemoryAllocateInfo = 5,
    MemoryAllocateFlagsInfo = 1000060000,
    BufferCreateInfo = 12,
    PhysicalDeviceVulkan12Features = 51,
    PhysicalDeviceVulkan13Features = 53,
    PhysicalDeviceFeatures2 = 1000059000,
    DebugUtilsObjectNameInfoEXT = 1000128000,
    DebugUtilsMessengerCallbackDataEXT = 1000128003,
    DebugUtilsMessengerCreateInfoEXT = 1000128004,
    ValidationFeaturesEXT = 1000247000,
};

// `VkValidationFeatureEnableEXT` values the backend turns on.
inline constexpr u32 kValidationFeatureSynchronization = 4;

enum class ObjectType : u32
{
    Unknown = 0,
    Instance = 1,
    PhysicalDevice = 2,
    Device = 3,
    Queue = 4,
    Semaphore = 5,
    CommandBuffer = 6,
    Fence = 7,
    DeviceMemory = 8,
    Buffer = 9,
    Image = 10,
    ImageView = 14,
    ShaderModule = 15,
    Pipeline = 19,
    PipelineLayout = 20,
    Sampler = 21,
    DescriptorSetLayout = 22,
    DescriptorSet = 23,
    CommandPool = 25,
};

enum class PhysicalDeviceType : u32
{
    Other = 0,
    IntegratedGpu = 1,
    DiscreteGpu = 2,
    VirtualGpu = 3,
    Cpu = 4,
};

namespace QueueFlag
{
inline constexpr Flags kGraphics = 0x1;
inline constexpr Flags kCompute = 0x2;
inline constexpr Flags kTransfer = 0x4;
inline constexpr Flags kSparseBinding = 0x8;
}  // namespace QueueFlag

namespace MemoryPropertyFlag
{
inline constexpr Flags kDeviceLocal = 0x1;
inline constexpr Flags kHostVisible = 0x2;
inline constexpr Flags kHostCoherent = 0x4;
inline constexpr Flags kHostCached = 0x8;
}  // namespace MemoryPropertyFlag

namespace MemoryHeapFlag
{
inline constexpr Flags kDeviceLocal = 0x1;
}

namespace DebugUtilsSeverity
{
inline constexpr Flags kVerbose = 0x1;
inline constexpr Flags kInfo = 0x10;
inline constexpr Flags kWarning = 0x100;
inline constexpr Flags kError = 0x1000;
}  // namespace DebugUtilsSeverity

namespace DebugUtilsMessageType
{
inline constexpr Flags kGeneral = 0x1;
inline constexpr Flags kValidation = 0x2;
inline constexpr Flags kPerformance = 0x4;
}  // namespace DebugUtilsMessageType

inline constexpr u32 kMaxPhysicalDeviceNameSize = 256;
inline constexpr u32 kUuidSize = 16;
inline constexpr u32 kMaxMemoryTypes = 32;
inline constexpr u32 kMaxMemoryHeaps = 16;
inline constexpr u32 kMaxExtensionNameSize = 256;
inline constexpr u32 kMaxDescriptionSize = 256;

struct ApplicationInfo
{
    StructureType sType = StructureType::ApplicationInfo;
    const void* pNext = nullptr;
    const char* pApplicationName = nullptr;
    u32 applicationVersion = 0;
    const char* pEngineName = nullptr;
    u32 engineVersion = 0;
    u32 apiVersion = 0;
};

struct InstanceCreateInfo
{
    StructureType sType = StructureType::InstanceCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    const ApplicationInfo* pApplicationInfo = nullptr;
    u32 enabledLayerCount = 0;
    const char* const* ppEnabledLayerNames = nullptr;
    u32 enabledExtensionCount = 0;
    const char* const* ppEnabledExtensionNames = nullptr;
};

struct LayerProperties
{
    char layerName[kMaxExtensionNameSize];
    u32 specVersion;
    u32 implementationVersion;
    char description[kMaxDescriptionSize];
};

struct ExtensionProperties
{
    char extensionName[kMaxExtensionNameSize];
    u32 specVersion;
};

// The head of `VkPhysicalDeviceProperties`, which is everything the backend
// reads. The real structure continues with the device limits and sparse
// properties (824 bytes in all on a 64-bit target); the tail here is larger than
// that, so a driver filling the real structure stays inside this one.
struct PhysicalDeviceProperties
{
    u32 apiVersion;
    u32 driverVersion;
    u32 vendorID;
    u32 deviceID;
    PhysicalDeviceType deviceType;
    char deviceName[kMaxPhysicalDeviceNameSize];
    u8 pipelineCacheUUID[kUuidSize];
    u64 limitsAndSparseProperties[(1024 - 296) / 8];
};
static_assert(sizeof(PhysicalDeviceProperties) >= 824,
              "must be at least as large as the specification's structure");

struct MemoryType
{
    Flags propertyFlags;
    u32 heapIndex;
};

struct MemoryHeap
{
    DeviceSize size;
    Flags flags;
};

struct PhysicalDeviceMemoryProperties
{
    u32 memoryTypeCount;
    MemoryType memoryTypes[kMaxMemoryTypes];
    u32 memoryHeapCount;
    MemoryHeap memoryHeaps[kMaxMemoryHeaps];
};

struct Extent3D
{
    u32 width;
    u32 height;
    u32 depth;
};

struct QueueFamilyProperties
{
    Flags queueFlags;
    u32 queueCount;
    u32 timestampValidBits;
    Extent3D minImageTransferGranularity;
};

// `VkPhysicalDeviceFeatures` is 55 `VkBool32` members. The backend does not
// request any of the core 1.0 features by name yet, so it is kept as an array.
inline constexpr u32 kCoreFeatureCount = 55;

struct PhysicalDeviceFeatures
{
    Bool32 flags[kCoreFeatureCount];
};

struct PhysicalDeviceFeatures2
{
    StructureType sType = StructureType::PhysicalDeviceFeatures2;
    void* pNext = nullptr;
    PhysicalDeviceFeatures features{};
};

// The Vulkan 1.2 and 1.3 feature structures are lists of `VkBool32` after the
// header. Indices below are the position of each member in the specification's
// declaration order; the array form keeps the layout exact without declaring
// forty-odd members the backend never names.
inline constexpr u32 kVulkan11FeatureCount = 12;
inline constexpr u32 kVulkan12FeatureCount = 47;
inline constexpr u32 kVulkan13FeatureCount = 15;

// `VkPhysicalDeviceFeatures` member positions the backend enables when present.
namespace Feature10
{
inline constexpr u32 kRobustBufferAccess = 0;
inline constexpr u32 kImageCubeArray = 2;
inline constexpr u32 kIndependentBlend = 3;
inline constexpr u32 kMultiDrawIndirect = 9;
inline constexpr u32 kDrawIndirectFirstInstance = 10;
inline constexpr u32 kDepthClamp = 11;
inline constexpr u32 kDepthBiasClamp = 12;
inline constexpr u32 kFillModeNonSolid = 13;
inline constexpr u32 kSamplerAnisotropy = 19;
inline constexpr u32 kTextureCompressionBc = 22;
inline constexpr u32 kVertexPipelineStoresAndAtomics = 25;
inline constexpr u32 kFragmentStoresAndAtomics = 26;
inline constexpr u32 kShaderImageGatherExtended = 28;
inline constexpr u32 kShaderStorageImageExtendedFormats = 29;
inline constexpr u32 kShaderStorageImageReadWithoutFormat = 31;
inline constexpr u32 kShaderStorageImageWriteWithoutFormat = 32;
inline constexpr u32 kShaderUniformBufferArrayDynamicIndexing = 33;
inline constexpr u32 kShaderSampledImageArrayDynamicIndexing = 34;
inline constexpr u32 kShaderStorageBufferArrayDynamicIndexing = 35;
inline constexpr u32 kShaderStorageImageArrayDynamicIndexing = 36;
inline constexpr u32 kShaderResourceResidency = 42;
inline constexpr u32 kShaderResourceMinLod = 43;
inline constexpr u32 kSparseBinding = 44;
inline constexpr u32 kSparseResidencyImage2D = 46;
inline constexpr u32 kSparseResidencyAliased = 52;
}  // namespace Feature10

namespace Feature11
{
inline constexpr u32 kShaderDrawParameters = 11;
}  // namespace Feature11

namespace Feature12
{
inline constexpr u32 kDrawIndirectCount = 1;
inline constexpr u32 kDescriptorIndexing = 9;
inline constexpr u32 kShaderUniformBufferArrayNonUniformIndexing = 13;
inline constexpr u32 kShaderSampledImageArrayNonUniformIndexing = 14;
inline constexpr u32 kShaderStorageBufferArrayNonUniformIndexing = 15;
inline constexpr u32 kShaderStorageImageArrayNonUniformIndexing = 16;
inline constexpr u32 kDescriptorBindingUniformBufferUpdateAfterBind = 20;
inline constexpr u32 kDescriptorBindingSampledImageUpdateAfterBind = 21;
inline constexpr u32 kDescriptorBindingStorageImageUpdateAfterBind = 22;
inline constexpr u32 kDescriptorBindingStorageBufferUpdateAfterBind = 23;
inline constexpr u32 kDescriptorBindingUpdateUnusedWhilePending = 26;
inline constexpr u32 kDescriptorBindingPartiallyBound = 27;
inline constexpr u32 kDescriptorBindingVariableDescriptorCount = 28;
inline constexpr u32 kRuntimeDescriptorArray = 29;
inline constexpr u32 kScalarBlockLayout = 31;
inline constexpr u32 kUniformBufferStandardLayout = 33;
inline constexpr u32 kHostQueryReset = 36;
inline constexpr u32 kTimelineSemaphore = 37;
inline constexpr u32 kBufferDeviceAddress = 38;
}  // namespace Feature12

struct PhysicalDeviceVulkan11Features
{
    StructureType sType = StructureType::PhysicalDeviceVulkan11Features;
    void* pNext = nullptr;
    Bool32 flags[kVulkan11FeatureCount]{};
};

// Byte offsets inside `PhysicalDeviceProperties::limitsAndSparseProperties`
// (the `VkPhysicalDeviceLimits` structure, then `VkPhysicalDeviceSparseProperties`
// at 504), for the members the backend reads.
inline constexpr u32 kLimitTimestampComputeAndGraphics = 420;
inline constexpr u32 kLimitTimestampPeriod = 424;
inline constexpr u32 kSparseResidencyStandard2DBlockShape = 504;
inline constexpr u32 kSparseResidencyNonResidentStrict = 520;

namespace Feature13
{
inline constexpr u32 kRobustImageAccess = 0;
inline constexpr u32 kSynchronization2 = 9;
inline constexpr u32 kDynamicRendering = 12;
}  // namespace Feature13

struct PhysicalDeviceVulkan12Features
{
    StructureType sType = StructureType::PhysicalDeviceVulkan12Features;
    void* pNext = nullptr;
    Bool32 flags[kVulkan12FeatureCount]{};
};

struct PhysicalDeviceVulkan13Features
{
    StructureType sType = StructureType::PhysicalDeviceVulkan13Features;
    void* pNext = nullptr;
    Bool32 flags[kVulkan13FeatureCount]{};
};

inline constexpr DeviceSize kWholeSize = ~0ULL;

namespace BufferUsage
{
inline constexpr Flags kTransferSrc = 0x1;
inline constexpr Flags kTransferDst = 0x2;
inline constexpr Flags kUniformTexelBuffer = 0x4;
inline constexpr Flags kStorageTexelBuffer = 0x8;
inline constexpr Flags kUniformBuffer = 0x10;
inline constexpr Flags kStorageBuffer = 0x20;
inline constexpr Flags kIndexBuffer = 0x40;
inline constexpr Flags kVertexBuffer = 0x80;
inline constexpr Flags kIndirectBuffer = 0x100;
inline constexpr Flags kShaderDeviceAddress = 0x20000;
}  // namespace BufferUsage

enum class SharingMode : u32
{
    Exclusive = 0,
    Concurrent = 1,
};

struct MemoryAllocateInfo
{
    StructureType sType = StructureType::MemoryAllocateInfo;
    const void* pNext = nullptr;
    DeviceSize allocationSize = 0;
    u32 memoryTypeIndex = 0;
};

struct MemoryAllocateFlagsInfo
{
    StructureType sType = StructureType::MemoryAllocateFlagsInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    u32 deviceMask = 0;
};

inline constexpr Flags kMemoryAllocateDeviceAddress = 0x2;

struct MemoryRequirements
{
    DeviceSize size;
    DeviceSize alignment;
    u32 memoryTypeBits;
};

struct BufferCreateInfo
{
    StructureType sType = StructureType::BufferCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    DeviceSize size = 0;
    Flags usage = 0;
    SharingMode sharingMode = SharingMode::Exclusive;
    u32 queueFamilyIndexCount = 0;
    const u32* pQueueFamilyIndices = nullptr;
};

struct DeviceQueueCreateInfo
{
    StructureType sType = StructureType::DeviceQueueCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    u32 queueFamilyIndex = 0;
    u32 queueCount = 0;
    const float* pQueuePriorities = nullptr;
};

struct DeviceCreateInfo
{
    StructureType sType = StructureType::DeviceCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    u32 queueCreateInfoCount = 0;
    const DeviceQueueCreateInfo* pQueueCreateInfos = nullptr;
    u32 enabledLayerCount = 0;
    const char* const* ppEnabledLayerNames = nullptr;
    u32 enabledExtensionCount = 0;
    const char* const* ppEnabledExtensionNames = nullptr;
    const PhysicalDeviceFeatures* pEnabledFeatures = nullptr;
};

struct ValidationFeaturesEXT
{
    StructureType sType = StructureType::ValidationFeaturesEXT;
    const void* pNext = nullptr;
    u32 enabledValidationFeatureCount = 0;
    const u32* pEnabledValidationFeatures = nullptr;
    u32 disabledValidationFeatureCount = 0;
    const u32* pDisabledValidationFeatures = nullptr;
};

inline constexpr const char* kValidationFeaturesExtensionName = "VK_EXT_validation_features";

struct DebugUtilsLabelEXT;

struct DebugUtilsObjectNameInfoEXT
{
    StructureType sType = StructureType::DebugUtilsObjectNameInfoEXT;
    const void* pNext = nullptr;
    ObjectType objectType = ObjectType::Unknown;
    u64 objectHandle = 0;
    const char* pObjectName = nullptr;
};

struct DebugUtilsMessengerCallbackDataEXT
{
    StructureType sType;
    const void* pNext;
    Flags flags;
    const char* pMessageIdName;
    i32 messageIdNumber;
    const char* pMessage;
    u32 queueLabelCount;
    const DebugUtilsLabelEXT* pQueueLabels;
    u32 cmdBufLabelCount;
    const DebugUtilsLabelEXT* pCmdBufLabels;
    u32 objectCount;
    const DebugUtilsObjectNameInfoEXT* pObjects;
};

using DebugUtilsMessengerCallbackEXT =
    Bool32(CYBER_VKAPI*)(Flags messageSeverity, Flags messageTypes,
                         const DebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);

struct DebugUtilsMessengerCreateInfoEXT
{
    StructureType sType = StructureType::DebugUtilsMessengerCreateInfoEXT;
    const void* pNext = nullptr;
    Flags flags = 0;
    Flags messageSeverity = 0;
    Flags messageType = 0;
    DebugUtilsMessengerCallbackEXT pfnUserCallback = nullptr;
    void* pUserData = nullptr;
};

// Function pointer types. The allocator argument is always null: the backend
// uses the driver's allocator, and host-memory tracking is a later concern.
using PFN_VoidFunction = void(CYBER_VKAPI*)();
using PFN_GetInstanceProcAddr = PFN_VoidFunction(CYBER_VKAPI*)(Instance, const char*);
using PFN_GetDeviceProcAddr = PFN_VoidFunction(CYBER_VKAPI*)(Device, const char*);

using PFN_EnumerateInstanceVersion = Result(CYBER_VKAPI*)(u32*);
using PFN_EnumerateInstanceLayerProperties = Result(CYBER_VKAPI*)(u32*, LayerProperties*);
using PFN_EnumerateInstanceExtensionProperties = Result(CYBER_VKAPI*)(const char*, u32*,
                                                                      ExtensionProperties*);
using PFN_CreateInstance = Result(CYBER_VKAPI*)(const InstanceCreateInfo*, const void*, Instance*);
using PFN_DestroyInstance = void(CYBER_VKAPI*)(Instance, const void*);
using PFN_EnumeratePhysicalDevices = Result(CYBER_VKAPI*)(Instance, u32*, PhysicalDevice*);
using PFN_GetPhysicalDeviceProperties = void(CYBER_VKAPI*)(PhysicalDevice,
                                                           PhysicalDeviceProperties*);
using PFN_GetPhysicalDeviceMemoryProperties = void(CYBER_VKAPI*)(PhysicalDevice,
                                                                 PhysicalDeviceMemoryProperties*);
using PFN_GetPhysicalDeviceQueueFamilyProperties = void(CYBER_VKAPI*)(PhysicalDevice, u32*,
                                                                      QueueFamilyProperties*);
using PFN_GetPhysicalDeviceFeatures2 = void(CYBER_VKAPI*)(PhysicalDevice, PhysicalDeviceFeatures2*);
// `VkPhysicalDeviceDescriptorIndexingProperties`: one u32 then seven VkBool32
// flags, then the update-after-bind limits, all 32-bit (indices below).
struct PhysicalDeviceDescriptorIndexingProperties
{
    StructureType sType = static_cast<StructureType>(1000161002);
    void* pNext = nullptr;
    u32 values[23]{};
};
inline constexpr u32 kUabPerStageSamplers = 8;
inline constexpr u32 kUabPerStageUniformBuffers = 9;
inline constexpr u32 kUabPerStageStorageBuffers = 10;
inline constexpr u32 kUabPerStageSampledImages = 11;
inline constexpr u32 kUabPerStageStorageImages = 12;
struct PhysicalDeviceProperties2
{
    StructureType sType = static_cast<StructureType>(1000059001);
    void* pNext = nullptr;
    PhysicalDeviceProperties properties{};
};
using PFN_GetPhysicalDeviceProperties2 = void(CYBER_VKAPI*)(PhysicalDevice,
                                                            PhysicalDeviceProperties2*);
using PFN_EnumerateDeviceExtensionProperties = Result(CYBER_VKAPI*)(PhysicalDevice, const char*,
                                                                    u32*, ExtensionProperties*);
using PFN_CreateDevice = Result(CYBER_VKAPI*)(PhysicalDevice, const DeviceCreateInfo*, const void*,
                                              Device*);
using PFN_DestroyDevice = void(CYBER_VKAPI*)(Device, const void*);
using PFN_GetDeviceQueue = void(CYBER_VKAPI*)(Device, u32, u32, Queue*);
using PFN_DeviceWaitIdle = Result(CYBER_VKAPI*)(Device);
using PFN_AllocateMemory = Result(CYBER_VKAPI*)(Device, const MemoryAllocateInfo*, const void*,
                                                DeviceMemory*);
using PFN_FreeMemory = void(CYBER_VKAPI*)(Device, DeviceMemory, const void*);
using PFN_MapMemory = Result(CYBER_VKAPI*)(Device, DeviceMemory, DeviceSize, DeviceSize, Flags,
                                           void**);
using PFN_UnmapMemory = void(CYBER_VKAPI*)(Device, DeviceMemory);
using PFN_CreateBuffer = Result(CYBER_VKAPI*)(Device, const BufferCreateInfo*, const void*,
                                              Buffer*);
using PFN_DestroyBuffer = void(CYBER_VKAPI*)(Device, Buffer, const void*);
using PFN_GetBufferMemoryRequirements = void(CYBER_VKAPI*)(Device, Buffer, MemoryRequirements*);
using PFN_BindBufferMemory = Result(CYBER_VKAPI*)(Device, Buffer, DeviceMemory, DeviceSize);
using PFN_SetDebugUtilsObjectNameEXT = Result(CYBER_VKAPI*)(Device,
                                                            const DebugUtilsObjectNameInfoEXT*);
using PFN_CreateDebugUtilsMessengerEXT = Result(CYBER_VKAPI*)(
    Instance, const DebugUtilsMessengerCreateInfoEXT*, const void*, DebugUtilsMessengerEXT*);
using PFN_DestroyDebugUtilsMessengerEXT = void(CYBER_VKAPI*)(Instance, DebugUtilsMessengerEXT,
                                                             const void*);

// Entry points the loader exports directly.
struct GlobalApi
{
    PFN_GetInstanceProcAddr getInstanceProcAddr = nullptr;
    PFN_EnumerateInstanceVersion enumerateInstanceVersion = nullptr;
    PFN_EnumerateInstanceLayerProperties enumerateInstanceLayerProperties = nullptr;
    PFN_EnumerateInstanceExtensionProperties enumerateInstanceExtensionProperties = nullptr;
    PFN_CreateInstance createInstance = nullptr;
};

// Entry points resolved through an instance.
struct InstanceApi
{
    PFN_DestroyInstance destroyInstance = nullptr;
    PFN_EnumeratePhysicalDevices enumeratePhysicalDevices = nullptr;
    PFN_GetPhysicalDeviceProperties getPhysicalDeviceProperties = nullptr;
    PFN_GetPhysicalDeviceMemoryProperties getPhysicalDeviceMemoryProperties = nullptr;
    PFN_GetPhysicalDeviceQueueFamilyProperties getPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_GetPhysicalDeviceFeatures2 getPhysicalDeviceFeatures2 = nullptr;
    PFN_GetPhysicalDeviceProperties2 getPhysicalDeviceProperties2 = nullptr;
    PFN_EnumerateDeviceExtensionProperties enumerateDeviceExtensionProperties = nullptr;
    PFN_CreateDevice createDevice = nullptr;
    PFN_GetDeviceProcAddr getDeviceProcAddr = nullptr;
    // Null when VK_EXT_debug_utils is not enabled.
    PFN_CreateDebugUtilsMessengerEXT createDebugUtilsMessenger = nullptr;
    PFN_DestroyDebugUtilsMessengerEXT destroyDebugUtilsMessenger = nullptr;
};

// Entry points resolved through a device, which skips the loader's dispatch.
struct DeviceApi
{
    PFN_DestroyDevice destroyDevice = nullptr;
    PFN_GetDeviceQueue getDeviceQueue = nullptr;
    PFN_DeviceWaitIdle deviceWaitIdle = nullptr;
    PFN_AllocateMemory allocateMemory = nullptr;
    PFN_FreeMemory freeMemory = nullptr;
    PFN_MapMemory mapMemory = nullptr;
    PFN_UnmapMemory unmapMemory = nullptr;
    PFN_CreateBuffer createBuffer = nullptr;
    PFN_DestroyBuffer destroyBuffer = nullptr;
    PFN_GetBufferMemoryRequirements getBufferMemoryRequirements = nullptr;
    PFN_BindBufferMemory bindBufferMemory = nullptr;
    // Null when VK_EXT_debug_utils is not enabled.
    PFN_SetDebugUtilsObjectNameEXT setDebugUtilsObjectName = nullptr;
};

inline constexpr const char* kValidationLayerName = "VK_LAYER_KHRONOS_validation";
inline constexpr const char* kDebugUtilsExtensionName = "VK_EXT_debug_utils";

}  // namespace cyber::rhi::detail::vk
