#pragma once

// GPU memory sub-allocation for the Vulkan backend. Phase 34 increment 2
// (ADR 0029, building on the Phase 15 design of ADR 0018): a few large
// `VkDeviceMemory` blocks per memory type, carved by a first-fit free list, so a
// scene of thousands of buffers stays under `maxMemoryAllocationCount` and
// allocation is a list walk rather than a driver call.

#include "vk_api.h"
#include "vk_api_ext.h"

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/rhi/memory.h>

namespace cyber::rhi::detail
{

using cyber::core::Error;
using cyber::core::Status;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::u8;
using cyber::core::usize;
using cyber::memory::DynamicArray;

// The three classes of the RHI, as for D3D12. Vulkan picks the memory type per
// class from the device's own table instead of fixed heap types.
enum class VulkanMemoryClass : u8
{
    DeviceLocal = 0,
    // Host-visible and coherent, preferably not device-local.
    Upload = 1,
    // Host-visible, coherent, preferably host-cached.
    Readback = 2,
    Count = 3,
};

struct VulkanMemoryRequest
{
    u64 sizeBytes = 0;
    // Power of two, from `VkMemoryRequirements::alignment`.
    u64 alignmentBytes = 1;
    // `VkMemoryRequirements::memoryTypeBits`.
    u32 memoryTypeBits = 0;
    VulkanMemoryClass memoryClass = VulkanMemoryClass::DeviceLocal;
    // Buffers and linear images are linear; optimal-tiling images are not. The
    // two are never placed in one block, which is how the allocator satisfies
    // `bufferImageGranularity` without reading the limit.
    bool linear = true;
};

// A reservation. `mapped` points at the first byte for host-visible classes and
// is null for device-local memory. Opaque to callers beyond these fields.
struct VulkanMemoryBlock
{
    vk::DeviceMemory memory = 0;
    u64 offset = 0;
    u64 sizeBytes = 0;
    void* mapped = nullptr;
    u32 pool = 0;
    u32 heap = 0;
    bool dedicated = false;
};

// A buffer and the memory bound to it.
struct VulkanBuffer
{
    vk::Buffer buffer = 0;
    VulkanMemoryBlock block{};
    u64 sizeBytes = 0;

    [[nodiscard]] bool IsValid() const noexcept
    {
        return buffer != 0;
    }
};

struct VulkanImageDesc
{
    u32 width = 1;
    u32 height = 1;
    u32 depth = 1;
    u32 mipLevels = 1;
    u32 arrayLayers = 1;
    vk::Format format = vk::Format::Undefined;
    // `vk::ImageUsage` bits.
    vk::Flags usage = 0;
    vk::ImageType type = vk::ImageType::Type2D;
    vk::ImageTiling tiling = vk::ImageTiling::Optimal;
    // Sample count: 1, 2, 4, 8, 16, 32 or 64, the same values as the Vulkan bits.
    u32 samples = 1;
    // A cube texture is a 2D array whose layers can be viewed as faces.
    bool cubeCompatible = false;
};

// An image and the memory bound to it. The image starts in the `Undefined`
// layout; the caller transitions it with a barrier.
struct VulkanImage
{
    vk::Image image = 0;
    VulkanMemoryBlock block{};
    VulkanImageDesc desc{};

    [[nodiscard]] bool IsValid() const noexcept
    {
        return image != 0;
    }
};

struct VulkanBufferDesc
{
    u64 sizeBytes = 0;
    // `vk::BufferUsage` bits.
    vk::Flags usage = 0;
    VulkanMemoryClass memoryClass = VulkanMemoryClass::DeviceLocal;
};

// The allocator of one device. Not thread-safe: every call is made under the
// owning device's lock, so the device decides the locking discipline once.
class VulkanMemoryAllocator
{
public:
    explicit VulkanMemoryAllocator(memory::Allocator& allocator) noexcept;
    VulkanMemoryAllocator(const VulkanMemoryAllocator&) = delete;
    VulkanMemoryAllocator& operator=(const VulkanMemoryAllocator&) = delete;
    ~VulkanMemoryAllocator();

    // Binds the allocator to a device. Exactly once, before any other call.
    // `memoryProperties` is copied.
    [[nodiscard]] Status<Error>
    Initialize(vk::Device device, const vk::DeviceApi& functions, const vk::DeviceApiExt& extended,
               const vk::PhysicalDeviceMemoryProperties& memoryProperties,
               bool deviceAddress = false, u64 largestBlockBytes = 256ULL * 1024 * 1024) noexcept;

    // Declares the distinct queue families that will touch resources. With more than
    // one, new buffers and images are created `CONCURRENT` across them, which
    // spares the RHI an ownership transfer on every cross-queue use (D3D12 has none).
    void SetQueueFamilies(const u32* families, u32 count) noexcept;

    // Releases every block. The GPU must be idle.
    void Shutdown() noexcept;

    // `ErrorCode::Unsupported` when no memory type satisfies the class and the
    // request's type bits; `ErrorCode::OutOfRange` for a zero size or a
    // non-power-of-two alignment; `ErrorCode::Internal` when the driver refuses
    // memory or the allocation count cap is reached.
    [[nodiscard]] Status<Error> Allocate(const VulkanMemoryRequest& request,
                                         VulkanMemoryBlock* out) noexcept;

    // Returns a block `Allocate` produced. A dedicated block's memory is released
    // to the driver; a sub-allocated range returns to its block's free list and
    // merges with a touching neighbour.
    void Free(const VulkanMemoryBlock& block) noexcept;

    // Creates a buffer and binds memory for it.
    [[nodiscard]] Status<Error> CreateBuffer(const VulkanBufferDesc& desc,
                                             VulkanBuffer* out) noexcept;
    void DestroyBuffer(VulkanBuffer& buffer) noexcept;

    // Creates an optimal- or linear-tiling image and binds device-local memory.
    [[nodiscard]] Status<Error> CreateImage(const VulkanImageDesc& desc, VulkanImage* out) noexcept;
    void DestroyImage(VulkanImage& image) noexcept;

    // Views and samplers are plain driver objects with no memory of their own.
    [[nodiscard]] Status<Error> CreateImageView(const VulkanImage& image, vk::ImageViewType type,
                                                vk::Flags aspect, u32 baseMip, u32 mipCount,
                                                u32 baseLayer, u32 layerCount,
                                                vk::ImageView* out) noexcept;
    void DestroyImageView(vk::ImageView view) noexcept;
    [[nodiscard]] Status<Error> CreateSampler(const vk::SamplerCreateInfo& info,
                                              vk::Sampler* out) noexcept;
    void DestroySampler(vk::Sampler sampler) noexcept;

    // The report `Device::MemoryStats()` reads.
    [[nodiscard]] MemoryStats Statistics() const noexcept;

    // Driver allocations alive now, dedicated included. Capped well below the
    // specification's guaranteed minimum for `maxMemoryAllocationCount` (4096).
    [[nodiscard]] u32 LiveDriverAllocations() const noexcept
    {
        return m_liveDriverAllocations;
    }

    static constexpr u32 kMaxDriverAllocations = 3500;
    static constexpr u64 kFirstBlockBytes = 8ULL * 1024 * 1024;

private:
    struct Range
    {
        u64 offset = 0;
        u64 sizeBytes = 0;
    };

    struct Heap
    {
        explicit Heap(memory::Allocator& allocator) noexcept : freeRanges(allocator) {}
        Heap(const Heap&) = delete;
        Heap& operator=(const Heap&) = delete;
        Heap(Heap&&) noexcept = default;
        Heap& operator=(Heap&&) noexcept = default;

        vk::DeviceMemory memory = 0;
        u64 sizeBytes = 0;
        u64 usedBytes = 0;
        void* mapped = nullptr;
        // Sorted by offset, never touching: a touching pair is merged on free.
        DynamicArray<Range> freeRanges;
    };

    struct Pool
    {
        explicit Pool(memory::Allocator& allocator) noexcept : heaps(allocator) {}
        Pool(const Pool&) = delete;
        Pool& operator=(const Pool&) = delete;
        Pool(Pool&&) noexcept = default;
        Pool& operator=(Pool&&) noexcept = default;

        DynamicArray<Heap> heaps;
        u32 memoryType = 0;
        VulkanMemoryClass memoryClass = VulkanMemoryClass::DeviceLocal;
        bool linear = true;
        u64 nextBlockBytes = kFirstBlockBytes;
    };

    struct Dedicated
    {
        vk::DeviceMemory memory = 0;
        VulkanMemoryClass memoryClass = VulkanMemoryClass::DeviceLocal;
        u64 sizeBytes = 0;
    };

    [[nodiscard]] bool SelectMemoryType(VulkanMemoryClass memoryClass, u32 typeBits,
                                        u32* out) const noexcept;
    [[nodiscard]] Status<Error> AllocateDedicated(const VulkanMemoryRequest& request,
                                                  u32 memoryType, VulkanMemoryBlock* out) noexcept;
    [[nodiscard]] Status<Error> OpenHeap(Pool& pool, u32 poolIndex, u64 bytes) noexcept;
    [[nodiscard]] bool CarveFrom(Heap& heap, u64 sizeBytes, u64 alignmentBytes,
                                 u64* outOffset) noexcept;
    [[nodiscard]] static bool IsHostVisible(VulkanMemoryClass memoryClass) noexcept
    {
        return memoryClass != VulkanMemoryClass::DeviceLocal;
    }

    memory::Allocator* m_allocator;
    vk::Device m_device = nullptr;
    vk::DeviceApi m_functions{};
    vk::DeviceApiExt m_ext{};
    vk::PhysicalDeviceMemoryProperties m_properties{};
    u64 m_largestBlockBytes = 0;
    bool m_deviceAddress = false;
    u32 m_sharedFamilies[3] = {};
    u32 m_sharedFamilyCount = 0;
    u32 m_liveDriverAllocations = 0;
    // One pool per (class, linear, memory type), created on first use.
    DynamicArray<Pool> m_pools;
    DynamicArray<u32> m_poolLookup;
    DynamicArray<Dedicated> m_dedicated;
};

}  // namespace cyber::rhi::detail
