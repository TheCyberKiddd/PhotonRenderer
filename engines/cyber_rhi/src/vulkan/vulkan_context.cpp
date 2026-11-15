#include "vulkan_context.h"

#include "vk_api_ext.h"

#include <atomic>
#include <cstring>
#include <initializer_list>
#include <new>
#include <type_traits>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef interface
#undef interface
#endif
#else
#include <dlfcn.h>
#endif

namespace cyber::rhi::detail
{
namespace
{

using cyber::core::ErrorCode;
using cyber::core::usize;

// The loader entry points a context needs, found in the platform's Vulkan
// runtime. Opened per use: the operating system reference-counts the module, so
// two contexts share one mapping without the backend keeping a global.
struct LoaderLibrary
{
    void* module = nullptr;
    vk::GlobalApi global{};

    [[nodiscard]] static void* Symbol(void* module, const char* name) noexcept
    {
#if defined(_WIN32)
        return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(module), name));
#else
        return dlsym(module, name);
#endif
    }

    [[nodiscard]] bool Open() noexcept
    {
#if defined(_WIN32)
        module = LoadLibraryA("vulkan-1.dll");
#elif defined(__APPLE__)
        module = dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);
#else
        module = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
        if (module == nullptr)
        {
            return false;
        }
        global.getInstanceProcAddr =
            reinterpret_cast<vk::PFN_GetInstanceProcAddr>(Symbol(module, "vkGetInstanceProcAddr"));
        if (global.getInstanceProcAddr == nullptr)
        {
            Close();
            return false;
        }
        // Global commands are fetched through a null instance, as the
        // specification requires; `vkEnumerateInstanceVersion` is absent on a
        // 1.0 loader, which this backend does not support.
        const auto fetch = [this](const char* name) noexcept
        { return global.getInstanceProcAddr(nullptr, name); };
        global.enumerateInstanceVersion =
            reinterpret_cast<vk::PFN_EnumerateInstanceVersion>(fetch("vkEnumerateInstanceVersion"));
        global.enumerateInstanceLayerProperties =
            reinterpret_cast<vk::PFN_EnumerateInstanceLayerProperties>(
                fetch("vkEnumerateInstanceLayerProperties"));
        global.enumerateInstanceExtensionProperties =
            reinterpret_cast<vk::PFN_EnumerateInstanceExtensionProperties>(
                fetch("vkEnumerateInstanceExtensionProperties"));
        global.createInstance = reinterpret_cast<vk::PFN_CreateInstance>(fetch("vkCreateInstance"));
        if (global.enumerateInstanceVersion == nullptr ||
            global.enumerateInstanceLayerProperties == nullptr ||
            global.enumerateInstanceExtensionProperties == nullptr ||
            global.createInstance == nullptr)
        {
            Close();
            return false;
        }
        return true;
    }

    void Close() noexcept
    {
        if (module != nullptr)
        {
#if defined(_WIN32)
            FreeLibrary(static_cast<HMODULE>(module));
#else
            dlclose(module);
#endif
        }
        module = nullptr;
        global = vk::GlobalApi{};
    }
};

// Validation message accounting, shared with the driver thread that calls the
// messenger callback. Everything is atomic or behind the one-flag lock.
struct ValidationCounters
{
    std::atomic<u32> errors{0};
    std::atomic<u32> warnings{0};
    mutable std::atomic_flag lock;
    char firstError[512] = {};
    u32 firstErrorLength = 0;
};

void LockCounters(const ValidationCounters& counters) noexcept
{
    while (counters.lock.test_and_set(std::memory_order_acquire))
    {
    }
}

void UnlockCounters(const ValidationCounters& counters) noexcept
{
    counters.lock.clear(std::memory_order_release);
}

vk::Bool32 CYBER_VKAPI ValidationCallback(vk::Flags severity, vk::Flags /*types*/,
                                          const vk::DebugUtilsMessengerCallbackDataEXT* data,
                                          void* userData) noexcept
{
    auto* counters = static_cast<ValidationCounters*>(userData);
    if (counters == nullptr)
    {
        return vk::kFalse;
    }
    if ((severity & vk::DebugUtilsSeverity::kError) != 0)
    {
        if (counters->errors.fetch_add(1, std::memory_order_acq_rel) == 0 && data != nullptr &&
            data->pMessage != nullptr)
        {
            LockCounters(*counters);
            const usize length = std::strlen(data->pMessage);
            const usize copied = length < sizeof(counters->firstError) - 1
                                     ? length
                                     : sizeof(counters->firstError) - 1;
            std::memcpy(counters->firstError, data->pMessage, copied);
            counters->firstError[copied] = '\0';
            counters->firstErrorLength = static_cast<u32>(copied);
            UnlockCounters(*counters);
        }
    }
    else if ((severity & vk::DebugUtilsSeverity::kWarning) != 0)
    {
        counters->warnings.fetch_add(1, std::memory_order_acq_rel);
    }
    // False: the call that triggered the message is not aborted.
    return vk::kFalse;
}

// An instance with the function table resolved against it.
struct InstanceBundle
{
    vk::Instance instance = nullptr;
    vk::InstanceApi api{};
    bool validationLayer = false;
    bool synchronizationValidation = false;
    bool debugUtils = false;
    bool surface = false;
};

template <typename Function>
[[nodiscard]] bool Resolve(const vk::GlobalApi& global, vk::Instance instance, const char* name,
                           Function& out) noexcept
{
    out = reinterpret_cast<Function>(global.getInstanceProcAddr(instance, name));
    return out != nullptr;
}

[[nodiscard]] bool LoadInstanceApi(const vk::GlobalApi& global, InstanceBundle& bundle) noexcept
{
    vk::InstanceApi& api = bundle.api;
    const vk::Instance instance = bundle.instance;
    const bool required =
        Resolve(global, instance, "vkDestroyInstance", api.destroyInstance) &&
        Resolve(global, instance, "vkEnumeratePhysicalDevices", api.enumeratePhysicalDevices) &&
        Resolve(global, instance, "vkGetPhysicalDeviceProperties",
                api.getPhysicalDeviceProperties) &&
        Resolve(global, instance, "vkGetPhysicalDeviceMemoryProperties",
                api.getPhysicalDeviceMemoryProperties) &&
        Resolve(global, instance, "vkGetPhysicalDeviceQueueFamilyProperties",
                api.getPhysicalDeviceQueueFamilyProperties) &&
        Resolve(global, instance, "vkGetPhysicalDeviceFeatures2", api.getPhysicalDeviceFeatures2) &&
        Resolve(global, instance, "vkEnumerateDeviceExtensionProperties",
                api.enumerateDeviceExtensionProperties) &&
        Resolve(global, instance, "vkCreateDevice", api.createDevice) &&
        Resolve(global, instance, "vkGetDeviceProcAddr", api.getDeviceProcAddr);
    if (!required)
    {
        return false;
    }
    if (bundle.debugUtils)
    {
        if (!Resolve(global, instance, "vkCreateDebugUtilsMessengerEXT",
                     api.createDebugUtilsMessenger) ||
            !Resolve(global, instance, "vkDestroyDebugUtilsMessengerEXT",
                     api.destroyDebugUtilsMessenger))
        {
            bundle.debugUtils = false;
            api.createDebugUtilsMessenger = nullptr;
            api.destroyDebugUtilsMessenger = nullptr;
        }
    }
    return true;
}

// Creates an instance, with the validation layer and debug utils when the
// caller asked and the machine has them. Returns the Vulkan result of
// `vkCreateInstance` when it fails.
[[nodiscard]] ErrorCode CreateInstance(const LoaderLibrary& loader, bool wantValidation,
                                       bool wantPresentation, InstanceBundle& bundle) noexcept
{
    const vk::GlobalApi& global = loader.global;

    u32 loaderVersion = vk::MakeVersion(0, 1, 0, 0);
    if (global.enumerateInstanceVersion(&loaderVersion) != vk::Result::Success ||
        loaderVersion < vk::kApiVersion11)
    {
        return ErrorCode::Unsupported;
    }

    // Layers and extensions are listed into fixed arrays; a machine with more
    // than the capacity simply has the tail ignored, which can only turn off an
    // optional feature.
    constexpr u32 kLayerCapacity = 64;
    constexpr u32 kExtensionCapacity = 128;
    bool validationLayerPresent = false;
    if (wantValidation)
    {
        vk::LayerProperties layers[kLayerCapacity];
        u32 layerCount = kLayerCapacity;
        const vk::Result listed = global.enumerateInstanceLayerProperties(&layerCount, layers);
        if (listed == vk::Result::Success || listed == vk::Result::Incomplete)
        {
            for (u32 index = 0; index < layerCount && index < kLayerCapacity; ++index)
            {
                if (std::strcmp(layers[index].layerName, vk::kValidationLayerName) == 0)
                {
                    validationLayerPresent = true;
                    break;
                }
            }
        }
    }

    const char* extensions[6] = {};
    u32 extensionCount = 0;
    bool debugUtils = false;
    bool validationFeatures = false;
    bool surfaceExtension = false;
    bool platformSurfaceExtension = false;
    {
        vk::ExtensionProperties available[kExtensionCapacity];
        u32 count = kExtensionCapacity;
        const vk::Result listed =
            global.enumerateInstanceExtensionProperties(nullptr, &count, available);
        if (listed == vk::Result::Success || listed == vk::Result::Incomplete)
        {
            for (u32 index = 0; index < count && index < kExtensionCapacity; ++index)
            {
                if (std::strcmp(available[index].extensionName, vk::kDebugUtilsExtensionName) == 0)
                {
                    debugUtils = true;
                }
                if (std::strcmp(available[index].extensionName, vk::kSurfaceExtensionName) == 0)
                {
                    surfaceExtension = true;
                }
                if (std::strcmp(available[index].extensionName, vk::kWin32SurfaceExtensionName) ==
                    0)
                {
                    platformSurfaceExtension = true;
                }
            }
        }
    }
    if (validationLayerPresent)
    {
        vk::ExtensionProperties available[kExtensionCapacity];
        u32 count = kExtensionCapacity;
        const vk::Result listed = global.enumerateInstanceExtensionProperties(
            vk::kValidationLayerName, &count, available);
        if (listed == vk::Result::Success || listed == vk::Result::Incomplete)
        {
            for (u32 index = 0; index < count && index < kExtensionCapacity; ++index)
            {
                if (std::strcmp(available[index].extensionName,
                                vk::kValidationFeaturesExtensionName) == 0)
                {
                    validationFeatures = true;
                }
            }
        }
    }
    if (debugUtils)
    {
        extensions[extensionCount++] = vk::kDebugUtilsExtensionName;
    }
    if (validationFeatures)
    {
        extensions[extensionCount++] = vk::kValidationFeaturesExtensionName;
    }
    // Presentation is Win32-only for now; Linux and Android surfaces arrive with
    // their window backends. A machine without the extensions runs headless.
    const bool surface = wantPresentation && surfaceExtension && platformSurfaceExtension;
    if (surface)
    {
        extensions[extensionCount++] = vk::kSurfaceExtensionName;
        extensions[extensionCount++] = vk::kWin32SurfaceExtensionName;
    }

    // Synchronization validation is on whenever the layer is: it is the check
    // that catches a barrier translated wrongly from an RHI state.
    const u32 enabledFeatures[1] = {vk::kValidationFeatureSynchronization};
    vk::ValidationFeaturesEXT features{};
    features.enabledValidationFeatureCount = 1;
    features.pEnabledValidationFeatures = enabledFeatures;

    const char* layerNames[1] = {vk::kValidationLayerName};

    vk::ApplicationInfo application{};
    application.pApplicationName = "PhotonRenderer";
    application.pEngineName = "PhotonRenderer";
    application.apiVersion = vk::kApiVersion13;

    vk::InstanceCreateInfo info{};
    info.pApplicationInfo = &application;
    info.enabledExtensionCount = extensionCount;
    info.ppEnabledExtensionNames = extensions;
    if (validationLayerPresent)
    {
        info.enabledLayerCount = 1;
        info.ppEnabledLayerNames = layerNames;
        if (validationFeatures)
        {
            info.pNext = &features;
        }
    }

    vk::Instance instance = nullptr;
    const vk::Result created = global.createInstance(&info, nullptr, &instance);
    if (created != vk::Result::Success || instance == nullptr)
    {
        return created == vk::Result::ErrorIncompatibleDriver ? ErrorCode::Unsupported
                                                              : ErrorCode::Internal;
    }

    bundle.instance = instance;
    bundle.validationLayer = validationLayerPresent;
    bundle.synchronizationValidation = validationLayerPresent && validationFeatures;
    bundle.debugUtils = debugUtils;
    bundle.surface = surface;
    if (!LoadInstanceApi(global, bundle))
    {
        // The function table is incomplete, so even `vkDestroyInstance` may be
        // missing; resolve it by itself to clean up.
        vk::PFN_DestroyInstance destroy = nullptr;
        if (Resolve(global, instance, "vkDestroyInstance", destroy))
        {
            destroy(instance, nullptr);
        }
        bundle = InstanceBundle{};
        return ErrorCode::Internal;
    }
    return ErrorCode::Ok;
}

void DestroyInstance(InstanceBundle& bundle) noexcept
{
    if (bundle.instance != nullptr && bundle.api.destroyInstance != nullptr)
    {
        bundle.api.destroyInstance(bundle.instance, nullptr);
    }
    bundle = InstanceBundle{};
}

// Hardware before software, discrete before integrated. Stable, so devices of
// one type keep the order the loader reported them in.
[[nodiscard]] u32 DeviceRank(vk::PhysicalDeviceType type) noexcept
{
    switch (type)
    {
    case vk::PhysicalDeviceType::DiscreteGpu:
        return 0;
    case vk::PhysicalDeviceType::IntegratedGpu:
        return 1;
    case vk::PhysicalDeviceType::VirtualGpu:
        return 2;
    case vk::PhysicalDeviceType::Other:
        return 3;
    case vk::PhysicalDeviceType::Cpu:
        return 4;
    }
    return 3;
}

struct RankedDevice
{
    vk::PhysicalDevice device = nullptr;
    u32 rank = 0;
};

// Lists the physical devices in adapter order. `ranked` receives up to
// kMaxVulkanAdapters entries; the return value is how many.
[[nodiscard]] u32 CollectDevices(const InstanceBundle& bundle, RankedDevice* ranked) noexcept
{
    vk::PhysicalDevice devices[kMaxVulkanAdapters] = {};
    u32 count = kMaxVulkanAdapters;
    const vk::Result listed = bundle.api.enumeratePhysicalDevices(bundle.instance, &count, devices);
    if ((listed != vk::Result::Success && listed != vk::Result::Incomplete) || count == 0)
    {
        return 0;
    }
    if (count > kMaxVulkanAdapters)
    {
        count = kMaxVulkanAdapters;
    }

    for (u32 index = 0; index < count; ++index)
    {
        vk::PhysicalDeviceProperties properties{};
        bundle.api.getPhysicalDeviceProperties(devices[index], &properties);
        RankedDevice entry{devices[index], DeviceRank(properties.deviceType)};
        // Insertion sort: `count` is at most 16 and the sort must be stable.
        u32 position = index;
        while (position > 0 && ranked[position - 1].rank > entry.rank)
        {
            ranked[position] = ranked[position - 1];
            --position;
        }
        ranked[position] = entry;
    }
    return count;
}

void CopyName(char* destination, usize capacity, const char* source) noexcept
{
    usize length = 0;
    while (length + 1 < capacity && source[length] != '\0')
    {
        destination[length] = source[length];
        ++length;
    }
    destination[length] = '\0';
}

// Sums device-local and other heaps. Integrated GPUs report one device-local
// heap that is system memory, which is faithful: that is what the device uses.
void SumHeaps(const vk::PhysicalDeviceMemoryProperties& memory, u64& deviceLocal,
              u64& other) noexcept
{
    deviceLocal = 0;
    other = 0;
    for (u32 index = 0; index < memory.memoryHeapCount && index < vk::kMaxMemoryHeaps; ++index)
    {
        if ((memory.memoryHeaps[index].flags & vk::MemoryHeapFlag::kDeviceLocal) != 0)
        {
            deviceLocal += memory.memoryHeaps[index].size;
        }
        else
        {
            other += memory.memoryHeaps[index].size;
        }
    }
}

// Backend and adapter versions are reported as major * 100 + minor, matching
// the Direct3D 12 backend's 1200 for 12.0: Vulkan 1.3 is 103.
[[nodiscard]] u32 EncodeApiVersion(u32 vulkanVersion) noexcept
{
    return vk::VersionMajor(vulkanVersion) * 100U + vk::VersionMinor(vulkanVersion);
}

}  // namespace

struct VulkanContext::ValidationState
{
    ValidationCounters counters;
};

struct VulkanContext::Loader
{
    LoaderLibrary library;
};

u32 EnumerateVulkanAdapters(AdapterInfo* out, u32 capacity) noexcept
{
    LoaderLibrary loader;
    if (!loader.Open())
    {
        return 0;
    }

    InstanceBundle bundle;
    // Enumeration never needs validation: it creates an instance for one query.
    if (CreateInstance(loader, false, false, bundle) != ErrorCode::Ok)
    {
        loader.Close();
        return 0;
    }

    RankedDevice ranked[kMaxVulkanAdapters];
    const u32 count = CollectDevices(bundle, ranked);
    u32 written = 0;
    for (u32 index = 0; index < count; ++index)
    {
        if (out == nullptr || written >= capacity)
        {
            continue;
        }
        vk::PhysicalDeviceProperties properties{};
        bundle.api.getPhysicalDeviceProperties(ranked[index].device, &properties);
        vk::PhysicalDeviceMemoryProperties memory{};
        bundle.api.getPhysicalDeviceMemoryProperties(ranked[index].device, &memory);

        AdapterInfo info{};
        CopyName(info.name, sizeof(info.name), properties.deviceName);
        info.vendorId = properties.vendorID;
        info.deviceId = properties.deviceID;
        info.driverVersion = properties.driverVersion;
        u64 deviceLocal = 0;
        u64 other = 0;
        SumHeaps(memory, deviceLocal, other);
        info.dedicatedVideoMemoryBytes = deviceLocal;
        info.sharedSystemMemoryBytes = other;
        info.backend = BackendKind::Vulkan;
        info.apiVersion = EncodeApiVersion(properties.apiVersion);
        out[written++] = info;
    }

    DestroyInstance(bundle);
    loader.Close();
    return count;
}

VulkanContext::VulkanContext(VulkanContext&& other) noexcept
{
    *this = std::move(other);
}

VulkanContext& VulkanContext::operator=(VulkanContext&& other) noexcept
{
    if (this != &other)
    {
        Release();
        m_allocator = other.m_allocator;
        m_loader = other.m_loader;
        m_validation = other.m_validation;
        m_instanceApi = other.m_instanceApi;
        m_deviceApi = other.m_deviceApi;
        m_memoryProperties = other.m_memoryProperties;
        m_deviceExt = other.m_deviceExt;
        m_surfaceApi = other.m_surfaceApi;
        m_swapchainApi = other.m_swapchainApi;
        m_instance = other.m_instance;
        m_physicalDevice = other.m_physicalDevice;
        m_device = other.m_device;
        m_graphicsQueue = other.m_graphicsQueue;
        m_computeQueue = other.m_computeQueue;
        m_transferQueue = other.m_transferQueue;
        m_messenger = other.m_messenger;
        m_info = other.m_info;

        other.m_allocator = nullptr;
        other.m_loader = nullptr;
        other.m_validation = nullptr;
        other.m_instanceApi = vk::InstanceApi{};
        other.m_deviceApi = vk::DeviceApi{};
        other.m_instance = nullptr;
        other.m_physicalDevice = nullptr;
        other.m_device = nullptr;
        other.m_graphicsQueue = nullptr;
        other.m_computeQueue = nullptr;
        other.m_transferQueue = nullptr;
        other.m_messenger = 0;
        other.m_info = VulkanDeviceInfo{};
    }
    return *this;
}

VulkanContext::~VulkanContext()
{
    Release();
}

void VulkanContext::Release() noexcept
{
    // Reverse order of creation: the device first, then the messenger that
    // reports on it, then the instance, then the loader.
    if (m_device != nullptr && m_deviceApi.destroyDevice != nullptr)
    {
        if (m_deviceApi.deviceWaitIdle != nullptr)
        {
            (void)m_deviceApi.deviceWaitIdle(m_device);
        }
        m_deviceApi.destroyDevice(m_device, nullptr);
    }
    if (m_messenger != 0 && m_instanceApi.destroyDebugUtilsMessenger != nullptr)
    {
        m_instanceApi.destroyDebugUtilsMessenger(m_instance, m_messenger, nullptr);
    }
    if (m_instance != nullptr && m_instanceApi.destroyInstance != nullptr)
    {
        m_instanceApi.destroyInstance(m_instance, nullptr);
    }
    if (m_loader != nullptr)
    {
        m_loader->library.Close();
        m_loader->~Loader();
        m_allocator->Deallocate(m_loader);
    }
    // The messenger is gone, so nothing can call into the counters any more.
    if (m_validation != nullptr)
    {
        m_validation->~ValidationState();
        m_allocator->Deallocate(m_validation);
    }

    m_allocator = nullptr;
    m_loader = nullptr;
    m_validation = nullptr;
    m_instanceApi = vk::InstanceApi{};
    m_deviceApi = vk::DeviceApi{};
    m_instance = nullptr;
    m_physicalDevice = nullptr;
    m_device = nullptr;
    m_graphicsQueue = nullptr;
    m_computeQueue = nullptr;
    m_transferQueue = nullptr;
    m_messenger = 0;
    m_info = VulkanDeviceInfo{};
}

Result<VulkanContext, Error> VulkanContext::Create(const VulkanContextDesc& desc,
                                                   memory::Allocator& allocator) noexcept
{
    using Created = Result<VulkanContext, Error>;

    VulkanContext context;
    context.m_allocator = &allocator;

    void* loaderStorage = allocator.Allocate(sizeof(Loader), alignof(Loader));
    void* validationStorage = allocator.Allocate(sizeof(ValidationState), alignof(ValidationState));
    if (loaderStorage == nullptr || validationStorage == nullptr)
    {
        if (loaderStorage != nullptr)
        {
            allocator.Deallocate(loaderStorage);
        }
        if (validationStorage != nullptr)
        {
            allocator.Deallocate(validationStorage);
        }
        context.m_allocator = nullptr;
        return Created::Err(Error{ErrorCode::Internal, "failed to allocate the Vulkan context"});
    }
    context.m_loader = new (loaderStorage) Loader();
    context.m_validation = new (validationStorage) ValidationState();

    if (!context.m_loader->library.Open())
    {
        return Created::Err(Error{ErrorCode::NotFound, "no Vulkan loader is installed"});
    }
    const LoaderLibrary& loader = context.m_loader->library;

    InstanceBundle bundle;
    const ErrorCode instanceResult =
        CreateInstance(loader, desc.enableValidation, desc.enablePresentation, bundle);
    if (instanceResult != ErrorCode::Ok)
    {
        return Created::Err(Error{instanceResult, "vkCreateInstance failed"});
    }
    context.m_instance = bundle.instance;
    context.m_instanceApi = bundle.api;

    if (desc.requireValidation && !bundle.validationLayer)
    {
        return Created::Err(Error{ErrorCode::NotFound,
                                  "VK_LAYER_KHRONOS_validation is required and not installed"});
    }

    if (bundle.validationLayer && bundle.debugUtils)
    {
        vk::DebugUtilsMessengerCreateInfoEXT messenger{};
        messenger.messageSeverity =
            vk::DebugUtilsSeverity::kWarning | vk::DebugUtilsSeverity::kError;
        messenger.messageType =
            vk::DebugUtilsMessageType::kGeneral | vk::DebugUtilsMessageType::kValidation;
        messenger.pfnUserCallback = &ValidationCallback;
        messenger.pUserData = &context.m_validation->counters;
        if (bundle.api.createDebugUtilsMessenger(bundle.instance, &messenger, nullptr,
                                                 &context.m_messenger) != vk::Result::Success)
        {
            context.m_messenger = 0;
        }
    }

    RankedDevice ranked[kMaxVulkanAdapters];
    const u32 deviceCount = CollectDevices(bundle, ranked);
    if (desc.adapterIndex >= deviceCount)
    {
        return Created::Err(Error{ErrorCode::OutOfRange, "Vulkan adapter index does not exist"});
    }
    const vk::PhysicalDevice physical = ranked[desc.adapterIndex].device;
    const vk::InstanceApi& api = bundle.api;

    vk::PhysicalDeviceProperties properties{};
    api.getPhysicalDeviceProperties(physical, &properties);
    if (properties.apiVersion < kVulkanRequiredApiVersion)
    {
        return Created::Err(
            Error{ErrorCode::Unsupported, "the device does not support Vulkan 1.3"});
    }

    // Queue families. Graphics must also support compute (the specification
    // guarantees such a family exists); compute and transfer prefer a family
    // that is not the graphics one so uploads and async compute can overlap.
    vk::QueueFamilyProperties families[32] = {};
    u32 familyCount = 32;
    api.getPhysicalDeviceQueueFamilyProperties(physical, &familyCount, families);
    if (familyCount > 32)
    {
        familyCount = 32;
    }
    VulkanQueueFamilies queues{};
    for (u32 index = 0; index < familyCount; ++index)
    {
        const vk::Flags flags = families[index].queueFlags;
        if (queues.graphics == kNoQueueFamily && (flags & vk::QueueFlag::kGraphics) != 0 &&
            (flags & vk::QueueFlag::kCompute) != 0)
        {
            queues.graphics = index;
        }
    }
    if (queues.graphics == kNoQueueFamily)
    {
        return Created::Err(
            Error{ErrorCode::Unsupported, "the device has no graphics and compute queue family"});
    }
    queues.compute = queues.graphics;
    queues.transfer = queues.graphics;
    for (u32 index = 0; index < familyCount; ++index)
    {
        const vk::Flags flags = families[index].queueFlags;
        if (!queues.dedicatedCompute && (flags & vk::QueueFlag::kCompute) != 0 &&
            (flags & vk::QueueFlag::kGraphics) == 0)
        {
            queues.compute = index;
            queues.dedicatedCompute = true;
        }
        if (!queues.dedicatedTransfer && (flags & vk::QueueFlag::kTransfer) != 0 &&
            (flags & (vk::QueueFlag::kGraphics | vk::QueueFlag::kCompute)) == 0)
        {
            queues.transfer = index;
            queues.dedicatedTransfer = true;
        }
    }

    // Features. The required set is what the RHI's design depends on: bindless
    // resources need descriptor indexing, frame pacing needs timeline
    // semaphores, and barriers are written against synchronization2. A device
    // missing one is refused by name rather than failing later and obscurely.
    vk::PhysicalDeviceVulkan12Features supported12{};
    vk::PhysicalDeviceVulkan13Features supported13{};
    supported12.pNext = &supported13;
    vk::PhysicalDeviceFeatures2 supported{};
    supported.pNext = &supported12;
    api.getPhysicalDeviceFeatures2(physical, &supported);

    struct Required
    {
        bool is13;
        u32 index;
        const char* name;
    };
    constexpr Required kRequired[] = {
        {false, vk::Feature12::kTimelineSemaphore, "timelineSemaphore"},
        {false, vk::Feature12::kDescriptorIndexing, "descriptorIndexing"},
        {false, vk::Feature12::kRuntimeDescriptorArray, "runtimeDescriptorArray"},
        {false, vk::Feature12::kDescriptorBindingPartiallyBound, "descriptorBindingPartiallyBound"},
        {false, vk::Feature12::kDescriptorBindingVariableDescriptorCount,
         "descriptorBindingVariableDescriptorCount"},
        {false, vk::Feature12::kDescriptorBindingSampledImageUpdateAfterBind,
         "descriptorBindingSampledImageUpdateAfterBind"},
        {false, vk::Feature12::kShaderSampledImageArrayNonUniformIndexing,
         "shaderSampledImageArrayNonUniformIndexing"},
        {true, vk::Feature13::kSynchronization2, "synchronization2"},
        {true, vk::Feature13::kDynamicRendering, "dynamicRendering"},
    };
    vk::PhysicalDeviceVulkan12Features enabled12{};
    vk::PhysicalDeviceVulkan13Features enabled13{};
    enabled12.pNext = &enabled13;
    for (const Required& feature : kRequired)
    {
        const vk::Bool32* source = feature.is13 ? supported13.flags : supported12.flags;
        if (source[feature.index] == vk::kFalse)
        {
            return Created::Err(
                Error{ErrorCode::Unsupported, feature.name});  // names the missing feature
        }
        (feature.is13 ? enabled13.flags : enabled12.flags)[feature.index] = vk::kTrue;
    }
    // Optional features are enabled when present; the backend reports them so
    // the renderer can choose a path.
    const bool bufferDeviceAddress =
        supported12.flags[vk::Feature12::kBufferDeviceAddress] != vk::kFalse;
    const bool drawIndirectCount =
        supported12.flags[vk::Feature12::kDrawIndirectCount] != vk::kFalse;
    if (bufferDeviceAddress)
    {
        enabled12.flags[vk::Feature12::kBufferDeviceAddress] = vk::kTrue;
    }
    if (drawIndirectCount)
    {
        enabled12.flags[vk::Feature12::kDrawIndirectCount] = vk::kTrue;
    }

    // One queue per distinct family.
    const float priority = 1.0F;
    vk::DeviceQueueCreateInfo queueInfos[3] = {};
    u32 queueInfoCount = 0;
    const u32 wanted[3] = {queues.graphics, queues.compute, queues.transfer};
    for (const u32 family : wanted)
    {
        bool known = false;
        for (u32 index = 0; index < queueInfoCount; ++index)
        {
            known = known || queueInfos[index].queueFamilyIndex == family;
        }
        if (!known)
        {
            vk::DeviceQueueCreateInfo info{};
            info.queueFamilyIndex = family;
            info.queueCount = 1;
            info.pQueuePriorities = &priority;
            queueInfos[queueInfoCount++] = info;
        }
    }

    // The swapchain extension, when presentation was requested and the device has it.
    const char* deviceExtensions[1] = {};
    u32 deviceExtensionCount = 0;
    bool presentation = false;
    if (bundle.surface)
    {
        vk::ExtensionProperties available[256];
        u32 count = 256;
        const vk::Result listed =
            api.enumerateDeviceExtensionProperties(physical, nullptr, &count, available);
        if (listed == vk::Result::Success || listed == vk::Result::Incomplete)
        {
            for (u32 index = 0; index < count && index < 256; ++index)
            {
                if (std::strcmp(available[index].extensionName, vk::kSwapchainExtensionName) == 0)
                {
                    presentation = true;
                }
            }
        }
        if (presentation)
        {
            deviceExtensions[deviceExtensionCount++] = vk::kSwapchainExtensionName;
        }
    }

    vk::PhysicalDeviceFeatures2 enabled{};
    enabled.pNext = &enabled12;
    // Core 1.0 features the RHI exposes. Indices follow `VkPhysicalDeviceFeatures`:
    // multiDrawIndirect 9, samplerAnisotropy 19, textureCompressionBC 22.
    constexpr u32 kFeatureMultiDrawIndirect = 9;
    constexpr u32 kFeatureSamplerAnisotropy = 19;
    constexpr u32 kFeatureTextureCompressionBc = 22;
    for (const u32 index :
         {kFeatureMultiDrawIndirect, kFeatureSamplerAnisotropy, kFeatureTextureCompressionBc})
    {
        enabled.features.flags[index] = supported.features.flags[index];
    }
    vk::DeviceCreateInfo deviceInfo{};
    deviceInfo.pNext = &enabled;
    deviceInfo.queueCreateInfoCount = queueInfoCount;
    deviceInfo.pQueueCreateInfos = queueInfos;
    deviceInfo.enabledExtensionCount = deviceExtensionCount;
    deviceInfo.ppEnabledExtensionNames = deviceExtensions;

    vk::Device device = nullptr;
    const vk::Result created = api.createDevice(physical, &deviceInfo, nullptr, &device);
    if (created != vk::Result::Success || device == nullptr)
    {
        return Created::Err(Error{created == vk::Result::ErrorFeatureNotPresent ||
                                          created == vk::Result::ErrorExtensionNotPresent
                                      ? ErrorCode::Unsupported
                                      : ErrorCode::Internal,
                                  "vkCreateDevice failed"});
    }
    context.m_device = device;
    context.m_physicalDevice = physical;

    // Device-level entry points skip the loader's dispatch trampoline.
    vk::DeviceApi& deviceApi = context.m_deviceApi;
    const auto resolveDevice = [&](const char* name) noexcept
    { return api.getDeviceProcAddr(device, name); };
    deviceApi.destroyDevice =
        reinterpret_cast<vk::PFN_DestroyDevice>(resolveDevice("vkDestroyDevice"));
    deviceApi.getDeviceQueue =
        reinterpret_cast<vk::PFN_GetDeviceQueue>(resolveDevice("vkGetDeviceQueue"));
    deviceApi.deviceWaitIdle =
        reinterpret_cast<vk::PFN_DeviceWaitIdle>(resolveDevice("vkDeviceWaitIdle"));
    const auto bind = [&](const char* name, auto& slot) noexcept
    {
        slot = reinterpret_cast<std::remove_reference_t<decltype(slot)>>(resolveDevice(name));
        return slot != nullptr;
    };
    const bool memoryEntries =
        bind("vkAllocateMemory", deviceApi.allocateMemory) &&
        bind("vkFreeMemory", deviceApi.freeMemory) && bind("vkMapMemory", deviceApi.mapMemory) &&
        bind("vkUnmapMemory", deviceApi.unmapMemory) &&
        bind("vkCreateBuffer", deviceApi.createBuffer) &&
        bind("vkDestroyBuffer", deviceApi.destroyBuffer) &&
        bind("vkGetBufferMemoryRequirements", deviceApi.getBufferMemoryRequirements) &&
        bind("vkBindBufferMemory", deviceApi.bindBufferMemory);
    if (bundle.debugUtils)
    {
        // The naming function is an instance-level extension command; through
        // the device address it is dispatched to the device that owns the object.
        deviceApi.setDebugUtilsObjectName = reinterpret_cast<vk::PFN_SetDebugUtilsObjectNameEXT>(
            api.getDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
    }
    // The second slice of device commands, resolved in one pass.
    bool extEntries = true;
    const auto resolveInto = [&](auto& slot, const char* name) noexcept
    {
        slot = reinterpret_cast<std::remove_reference_t<decltype(slot)>>(resolveDevice(name));
        return slot != nullptr;
    };
    const auto resolveSurface = [&](auto& slot, const char* name) noexcept
    {
        slot = reinterpret_cast<std::remove_reference_t<decltype(slot)>>(
            loader.global.getInstanceProcAddr(bundle.instance, name));
        return slot != nullptr;
    };
#define CYBER_VK_RESOLVE_DEVICE(member, name, type)                                                \
    extEntries = resolveInto(context.m_deviceExt.member, name) && extEntries;
    CYBER_VK_DEVICE_FUNCTIONS_EXT(CYBER_VK_RESOLVE_DEVICE)
#undef CYBER_VK_RESOLVE_DEVICE
    if (presentation)
    {
#define CYBER_VK_RESOLVE_SWAPCHAIN(member, name, type)                                             \
    presentation = resolveInto(context.m_swapchainApi.member, name) && presentation;
        CYBER_VK_SWAPCHAIN_FUNCTIONS(CYBER_VK_RESOLVE_SWAPCHAIN)
#undef CYBER_VK_RESOLVE_SWAPCHAIN
#define CYBER_VK_RESOLVE_SURFACE(member, name, type)                                               \
    presentation = resolveSurface(context.m_surfaceApi.member, name) && presentation;
        CYBER_VK_SURFACE_FUNCTIONS(CYBER_VK_RESOLVE_SURFACE)
#undef CYBER_VK_RESOLVE_SURFACE
    }
    if (deviceApi.destroyDevice == nullptr || deviceApi.getDeviceQueue == nullptr ||
        deviceApi.deviceWaitIdle == nullptr || !memoryEntries || !extEntries)
    {
        return Created::Err(Error{ErrorCode::Internal, "device entry points are missing"});
    }

    deviceApi.getDeviceQueue(device, queues.graphics, 0, &context.m_graphicsQueue);
    deviceApi.getDeviceQueue(device, queues.compute, 0, &context.m_computeQueue);
    deviceApi.getDeviceQueue(device, queues.transfer, 0, &context.m_transferQueue);

    vk::PhysicalDeviceMemoryProperties memory{};
    api.getPhysicalDeviceMemoryProperties(physical, &memory);
    VulkanDeviceInfo& info = context.m_info;
    CopyName(info.name, sizeof(info.name), properties.deviceName);
    info.vendorId = properties.vendorID;
    info.deviceId = properties.deviceID;
    info.driverVersion = properties.driverVersion;
    info.apiVersion = properties.apiVersion;
    info.type = properties.deviceType;
    context.m_memoryProperties = memory;
    SumHeaps(memory, info.deviceLocalBytes, info.hostVisibleBytes);
    info.queues = queues;
    info.validationLayerEnabled = bundle.validationLayer;
    info.synchronizationValidationEnabled = bundle.synchronizationValidation;
    info.debugUtilsEnabled = bundle.debugUtils;
    info.bufferDeviceAddress = bufferDeviceAddress;
    info.presentation = presentation;
    info.samplerAnisotropy = supported.features.flags[kFeatureSamplerAnisotropy] != vk::kFalse;
    info.textureCompressionBc =
        supported.features.flags[kFeatureTextureCompressionBc] != vk::kFalse;
    info.multiDrawIndirect = supported.features.flags[kFeatureMultiDrawIndirect] != vk::kFalse;
    {
        // The first five members of `VkPhysicalDeviceLimits` are the image
        // dimensions and layer count, then three buffer ranges, then
        // `maxPushConstantsSize`; all `uint32_t`, so they are read from the head
        // of the opaque tail without declaring the rest of the structure.
        u32 head[9] = {};
        std::memcpy(head, properties.limitsAndSparseProperties, sizeof(head));
        info.maxImageDimension1D = head[0];
        info.maxImageDimension2D = head[1];
        info.maxImageDimension3D = head[2];
        info.maxImageDimensionCube = head[3];
        info.maxImageArrayLayers = head[4];
        info.maxPushConstantBytes = head[8];
    }
    info.drawIndirectCount = drawIndirectCount;

    // Name the queues so a capture and a validation message say which is which.
    (void)context.SetObjectName(vk::ObjectType::Queue,
                                reinterpret_cast<u64>(context.m_graphicsQueue), "graphics queue");
    if (queues.dedicatedCompute)
    {
        (void)context.SetObjectName(vk::ObjectType::Queue,
                                    reinterpret_cast<u64>(context.m_computeQueue), "compute queue");
    }
    if (queues.dedicatedTransfer)
    {
        (void)context.SetObjectName(vk::ObjectType::Queue,
                                    reinterpret_cast<u64>(context.m_transferQueue),
                                    "transfer queue");
    }

    return Created::Ok(std::move(context));
}

u32 VulkanContext::ValidationErrorCount() const noexcept
{
    return m_validation != nullptr ? m_validation->counters.errors.load(std::memory_order_acquire)
                                   : 0;
}

u32 VulkanContext::ValidationWarningCount() const noexcept
{
    return m_validation != nullptr ? m_validation->counters.warnings.load(std::memory_order_acquire)
                                   : 0;
}

u32 VulkanContext::FirstValidationError(char* buffer, u32 capacity) const noexcept
{
    if (buffer == nullptr || capacity == 0)
    {
        return 0;
    }
    buffer[0] = '\0';
    if (m_validation == nullptr || ValidationErrorCount() == 0)
    {
        return 0;
    }
    const ValidationCounters& counters = m_validation->counters;
    LockCounters(counters);
    const u32 length =
        counters.firstErrorLength < capacity - 1 ? counters.firstErrorLength : capacity - 1;
    std::memcpy(buffer, counters.firstError, length);
    buffer[length] = '\0';
    UnlockCounters(counters);
    return length;
}

Status<Error> VulkanContext::SetObjectName(vk::ObjectType type, u64 handle,
                                           const char* name) noexcept
{
    if (m_device == nullptr)
    {
        return Status<Error>::Err(Error{ErrorCode::InvalidArgument, "context is not valid"});
    }
    if (m_deviceApi.setDebugUtilsObjectName == nullptr || name == nullptr)
    {
        return Status<Error>::Ok();
    }
    vk::DebugUtilsObjectNameInfoEXT info{};
    info.objectType = type;
    info.objectHandle = handle;
    info.pObjectName = name;
    if (m_deviceApi.setDebugUtilsObjectName(m_device, &info) != vk::Result::Success)
    {
        return Status<Error>::Err(
            Error{ErrorCode::Internal, "vkSetDebugUtilsObjectNameEXT failed"});
    }
    return Status<Error>::Ok();
}

Status<Error> VulkanContext::WaitIdle() noexcept
{
    if (m_device == nullptr || m_deviceApi.deviceWaitIdle == nullptr)
    {
        return Status<Error>::Err(Error{ErrorCode::InvalidArgument, "context is not valid"});
    }
    const vk::Result result = m_deviceApi.deviceWaitIdle(m_device);
    if (result == vk::Result::ErrorDeviceLost)
    {
        return Status<Error>::Err(Error{ErrorCode::DeviceLost, "Vulkan device lost"});
    }
    if (result != vk::Result::Success)
    {
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkDeviceWaitIdle failed"});
    }
    return Status<Error>::Ok();
}

}  // namespace cyber::rhi::detail
