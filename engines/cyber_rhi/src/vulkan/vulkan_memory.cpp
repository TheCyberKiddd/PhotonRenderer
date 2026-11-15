#include "vulkan_memory.h"

#include <utility>

namespace cyber::rhi::detail
{
namespace
{

using cyber::core::ErrorCode;

constexpr u32 kNoPool = 0xFFFFFFFFU;
constexpr u32 kMemoryTypeSlots = vk::kMaxMemoryTypes;
constexpr u32 kPoolSlots = static_cast<u32>(VulkanMemoryClass::Count) * 2U * kMemoryTypeSlots;

[[nodiscard]] constexpr u64 RoundUp(u64 value, u64 alignment) noexcept
{
    return (value + alignment - 1) & ~(alignment - 1);
}

[[nodiscard]] constexpr bool IsPowerOfTwo(u64 value) noexcept
{
    return value != 0 && (value & (value - 1)) == 0;
}

[[nodiscard]] constexpr u32 SlotOf(VulkanMemoryClass memoryClass, bool linear, u32 type) noexcept
{
    return ((static_cast<u32>(memoryClass) * 2U + (linear ? 1U : 0U)) * kMemoryTypeSlots) + type;
}

// Inserts `value` at `index`, keeping order. Grows first, so a failure leaves the
// array untouched.
template <typename T>
[[nodiscard]] bool InsertAt(DynamicArray<T>& array, usize index, const T& value) noexcept
{
    if (!array.TryPushBack(value))
    {
        return false;
    }
    for (usize slot = array.Size() - 1; slot > index; --slot)
    {
        array[slot] = array[slot - 1];
    }
    array[index] = value;
    return true;
}

template <typename T>
void RemoveAt(DynamicArray<T>& array, usize index) noexcept
{
    for (usize slot = index; slot + 1 < array.Size(); ++slot)
    {
        array[slot] = array[slot + 1];
    }
    array.PopBack();
}

}  // namespace

VulkanMemoryAllocator::VulkanMemoryAllocator(memory::Allocator& allocator) noexcept
    : m_allocator(&allocator), m_pools(allocator), m_poolLookup(allocator), m_dedicated(allocator)
{
}

VulkanMemoryAllocator::~VulkanMemoryAllocator()
{
    Shutdown();
}

Status<Error> VulkanMemoryAllocator::Initialize(
    vk::Device device, const vk::DeviceApi& functions,
    const vk::DeviceApiExt& extended,
    const vk::PhysicalDeviceMemoryProperties& memoryProperties, bool deviceAddress,
    u64 largestBlockBytes) noexcept
{
    if (device == nullptr || functions.allocateMemory == nullptr ||
        functions.freeMemory == nullptr || functions.mapMemory == nullptr ||
        functions.createBuffer == nullptr || largestBlockBytes < kFirstBlockBytes)
    {
        return Status<Error>::Err(
            Error{ErrorCode::InvalidArgument, "memory allocator needs a device and its functions"});
    }
    if (!m_poolLookup.TryResize(kPoolSlots))
    {
        return Status<Error>::Err(
            Error{ErrorCode::Internal, "failed to allocate the memory pool table"});
    }
    for (u32 slot = 0; slot < kPoolSlots; ++slot)
    {
        m_poolLookup[slot] = kNoPool;
    }
    m_device = device;
    m_functions = functions;
    m_ext = extended;
    m_properties = memoryProperties;
    m_deviceAddress = deviceAddress;
    m_largestBlockBytes = largestBlockBytes;
    return Status<Error>::Ok();
}

void VulkanMemoryAllocator::SetQueueFamilies(const u32* families, u32 count) noexcept
{
    m_sharedFamilyCount = 0;
    for (u32 index = 0; index < count && index < 3; ++index)
    {
        m_sharedFamilies[m_sharedFamilyCount++] = families[index];
    }
}

void VulkanMemoryAllocator::Shutdown() noexcept
{
    if (m_device == nullptr)
    {
        return;
    }
    for (usize poolIndex = 0; poolIndex < m_pools.Size(); ++poolIndex)
    {
        Pool& pool = m_pools[poolIndex];
        for (usize heapIndex = 0; heapIndex < pool.heaps.Size(); ++heapIndex)
        {
            // Freeing mapped memory unmaps it implicitly.
            m_functions.freeMemory(m_device, pool.heaps[heapIndex].memory, nullptr);
        }
        pool.heaps.Clear();
    }
    for (usize index = 0; index < m_dedicated.Size(); ++index)
    {
        m_functions.freeMemory(m_device, m_dedicated[index].memory, nullptr);
    }
    m_pools.Clear();
    m_dedicated.Clear();
    m_liveDriverAllocations = 0;
    m_device = nullptr;
}

bool VulkanMemoryAllocator::SelectMemoryType(VulkanMemoryClass memoryClass, u32 typeBits,
                                             u32* out) const noexcept
{
    vk::Flags required = 0;
    vk::Flags preferred = 0;
    vk::Flags avoided = 0;
    switch (memoryClass)
    {
    case VulkanMemoryClass::DeviceLocal:
        required = vk::MemoryPropertyFlag::kDeviceLocal;
        break;
    case VulkanMemoryClass::Upload:
        required = vk::MemoryPropertyFlag::kHostVisible | vk::MemoryPropertyFlag::kHostCoherent;
        avoided = vk::MemoryPropertyFlag::kDeviceLocal;
        break;
    case VulkanMemoryClass::Readback:
        required = vk::MemoryPropertyFlag::kHostVisible | vk::MemoryPropertyFlag::kHostCoherent;
        preferred = vk::MemoryPropertyFlag::kHostCached;
        avoided = vk::MemoryPropertyFlag::kDeviceLocal;
        break;
    case VulkanMemoryClass::Count:
        return false;
    }

    bool found = false;
    int bestScore = -1;
    for (u32 index = 0; index < m_properties.memoryTypeCount && index < vk::kMaxMemoryTypes;
         ++index)
    {
        if ((typeBits & (1U << index)) == 0)
        {
            continue;
        }
        const vk::Flags flags = m_properties.memoryTypes[index].propertyFlags;
        if ((flags & required) != required)
        {
            continue;
        }
        // Preferred and avoided properties break ties; the first type with the
        // best score wins, and the specification orders types so that the first
        // match is the most performant for a given property set.
        int score = 0;
        score += (flags & preferred) != 0 ? 2 : 0;
        score += (flags & avoided) == 0 ? 1 : 0;
        if (score > bestScore)
        {
            bestScore = score;
            *out = index;
            found = true;
        }
    }
    return found;
}

Status<Error> VulkanMemoryAllocator::AllocateDedicated(const VulkanMemoryRequest& request,
                                                       u32 memoryType,
                                                       VulkanMemoryBlock* out) noexcept
{
    if (m_liveDriverAllocations >= kMaxDriverAllocations)
    {
        return Status<Error>::Err(
            Error{ErrorCode::Internal, "the driver allocation count cap was reached"});
    }
    vk::MemoryAllocateFlagsInfo flags{};
    flags.flags = vk::kMemoryAllocateDeviceAddress;
    vk::MemoryAllocateInfo info{};
    info.allocationSize = request.sizeBytes;
    info.memoryTypeIndex = memoryType;
    if (m_deviceAddress && request.linear)
    {
        info.pNext = &flags;
    }
    vk::DeviceMemory memory = 0;
    if (m_functions.allocateMemory(m_device, &info, nullptr, &memory) != vk::Result::Success)
    {
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkAllocateMemory refused a block"});
    }
    void* mapped = nullptr;
    if (IsHostVisible(request.memoryClass) &&
        m_functions.mapMemory(m_device, memory, 0, vk::kWholeSize, 0, &mapped) !=
            vk::Result::Success)
    {
        m_functions.freeMemory(m_device, memory, nullptr);
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkMapMemory failed"});
    }
    if (!m_dedicated.TryPushBack(Dedicated{memory, request.memoryClass, request.sizeBytes}))
    {
        m_functions.freeMemory(m_device, memory, nullptr);
        return Status<Error>::Err(Error{ErrorCode::Internal, "failed to track a dedicated block"});
    }
    ++m_liveDriverAllocations;
    VulkanMemoryBlock block{};
    block.memory = memory;
    block.offset = 0;
    block.sizeBytes = request.sizeBytes;
    block.mapped = mapped;
    block.dedicated = true;
    *out = block;
    return Status<Error>::Ok();
}

Status<Error> VulkanMemoryAllocator::OpenHeap(Pool& pool, u32 /*poolIndex*/, u64 bytes) noexcept
{
    if (m_liveDriverAllocations >= kMaxDriverAllocations)
    {
        return Status<Error>::Err(
            Error{ErrorCode::Internal, "the driver allocation count cap was reached"});
    }
    vk::MemoryAllocateFlagsInfo flags{};
    flags.flags = vk::kMemoryAllocateDeviceAddress;
    vk::MemoryAllocateInfo info{};
    info.allocationSize = bytes;
    info.memoryTypeIndex = pool.memoryType;
    // Linear pools carry buffers, so they are the ones that may need an address.
    if (m_deviceAddress && pool.linear)
    {
        info.pNext = &flags;
    }

    vk::DeviceMemory memory = 0;
    if (m_functions.allocateMemory(m_device, &info, nullptr, &memory) != vk::Result::Success)
    {
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkAllocateMemory refused a block"});
    }
    void* mapped = nullptr;
    if (IsHostVisible(pool.memoryClass) &&
        m_functions.mapMemory(m_device, memory, 0, vk::kWholeSize, 0, &mapped) !=
            vk::Result::Success)
    {
        m_functions.freeMemory(m_device, memory, nullptr);
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkMapMemory failed"});
    }

    Heap* heap = pool.heaps.TryEmplaceBack(*m_allocator);
    if (heap == nullptr || !heap->freeRanges.TryPushBack(Range{0, bytes}))
    {
        if (heap != nullptr)
        {
            pool.heaps.PopBack();
        }
        m_functions.freeMemory(m_device, memory, nullptr);
        return Status<Error>::Err(Error{ErrorCode::Internal, "failed to track a memory block"});
    }
    heap->memory = memory;
    heap->sizeBytes = bytes;
    heap->mapped = mapped;
    ++m_liveDriverAllocations;
    return Status<Error>::Ok();
}

bool VulkanMemoryAllocator::CarveFrom(Heap& heap, u64 sizeBytes, u64 alignmentBytes,
                                      u64* outOffset) noexcept
{
    for (usize index = 0; index < heap.freeRanges.Size(); ++index)
    {
        const Range range = heap.freeRanges[index];
        const u64 aligned = RoundUp(range.offset, alignmentBytes);
        const u64 end = range.offset + range.sizeBytes;
        if (aligned + sizeBytes > end)
        {
            continue;
        }
        const u64 head = aligned - range.offset;
        const u64 tail = end - (aligned + sizeBytes);
        if (head != 0 && tail != 0)
        {
            // Split: the head stays in place, the tail is inserted after it.
            if (!InsertAt(heap.freeRanges, index + 1, Range{aligned + sizeBytes, tail}))
            {
                return false;
            }
            heap.freeRanges[index].sizeBytes = head;
        }
        else if (head != 0)
        {
            heap.freeRanges[index].sizeBytes = head;
        }
        else if (tail != 0)
        {
            heap.freeRanges[index] = Range{aligned + sizeBytes, tail};
        }
        else
        {
            RemoveAt(heap.freeRanges, index);
        }
        heap.usedBytes += sizeBytes;
        *outOffset = aligned;
        return true;
    }
    return false;
}

Status<Error> VulkanMemoryAllocator::Allocate(const VulkanMemoryRequest& request,
                                              VulkanMemoryBlock* out) noexcept
{
    if (out == nullptr || m_device == nullptr)
    {
        return Status<Error>::Err(
            Error{ErrorCode::InvalidArgument, "memory allocator is not initialized"});
    }
    if (request.sizeBytes == 0 || !IsPowerOfTwo(request.alignmentBytes) ||
        request.memoryClass >= VulkanMemoryClass::Count)
    {
        return Status<Error>::Err(
            Error{ErrorCode::OutOfRange, "memory request has a zero size or a bad alignment"});
    }
    u32 memoryType = 0;
    if (!SelectMemoryType(request.memoryClass, request.memoryTypeBits, &memoryType))
    {
        return Status<Error>::Err(
            Error{ErrorCode::Unsupported, "no memory type satisfies the request"});
    }

    // A request a quarter of the largest block or more would fragment a shared
    // block for little gain; it gets its own allocation, as in the D3D12 backend.
    if (request.sizeBytes >= m_largestBlockBytes / 4)
    {
        return AllocateDedicated(request, memoryType, out);
    }

    const u32 slot = SlotOf(request.memoryClass, request.linear, memoryType);
    if (m_poolLookup[slot] == kNoPool)
    {
        Pool* pool = m_pools.TryEmplaceBack(*m_allocator);
        if (pool == nullptr)
        {
            return Status<Error>::Err(Error{ErrorCode::Internal, "failed to create a memory pool"});
        }
        pool->memoryType = memoryType;
        pool->memoryClass = request.memoryClass;
        pool->linear = request.linear;
        m_poolLookup[slot] = static_cast<u32>(m_pools.Size() - 1);
    }
    const u32 poolIndex = m_poolLookup[slot];
    Pool& pool = m_pools[poolIndex];

    u64 offset = 0;
    for (usize heapIndex = 0; heapIndex < pool.heaps.Size(); ++heapIndex)
    {
        Heap& heap = pool.heaps[heapIndex];
        if (heap.sizeBytes - heap.usedBytes >= request.sizeBytes &&
            CarveFrom(heap, request.sizeBytes, request.alignmentBytes, &offset))
        {
            VulkanMemoryBlock block{};
            block.memory = heap.memory;
            block.offset = offset;
            block.sizeBytes = request.sizeBytes;
            block.mapped = heap.mapped != nullptr ? static_cast<u8*>(heap.mapped) + offset : nullptr;
            block.pool = poolIndex;
            block.heap = static_cast<u32>(heapIndex);
            *out = block;
            return Status<Error>::Ok();
        }
    }

    // Open a new block: the growth step, or the request plus its alignment slack
    // when that is larger.
    u64 bytes = pool.nextBlockBytes;
    const u64 needed = request.sizeBytes + request.alignmentBytes;
    while (bytes < needed)
    {
        bytes *= 2;
    }
    Status<Error> opened = OpenHeap(pool, poolIndex, bytes);
    if (!opened)
    {
        // The driver may refuse a large block but still serve exactly this
        // request, so the dedicated path is the fallback, not an error.
        return AllocateDedicated(request, memoryType, out);
    }
    if (pool.nextBlockBytes < m_largestBlockBytes)
    {
        pool.nextBlockBytes *= 2;
    }
    const usize heapIndex = pool.heaps.Size() - 1;
    Heap& heap = pool.heaps[heapIndex];
    if (!CarveFrom(heap, request.sizeBytes, request.alignmentBytes, &offset))
    {
        return Status<Error>::Err(Error{ErrorCode::Internal, "failed to carve a new block"});
    }
    VulkanMemoryBlock block{};
    block.memory = heap.memory;
    block.offset = offset;
    block.sizeBytes = request.sizeBytes;
    block.mapped = heap.mapped != nullptr ? static_cast<u8*>(heap.mapped) + offset : nullptr;
    block.pool = poolIndex;
    block.heap = static_cast<u32>(heapIndex);
    *out = block;
    return Status<Error>::Ok();
}

void VulkanMemoryAllocator::Free(const VulkanMemoryBlock& block) noexcept
{
    if (m_device == nullptr || block.memory == 0)
    {
        return;
    }
    if (block.dedicated)
    {
        for (usize index = 0; index < m_dedicated.Size(); ++index)
        {
            if (m_dedicated[index].memory == block.memory)
            {
                RemoveAt(m_dedicated, index);
                m_functions.freeMemory(m_device, block.memory, nullptr);
                --m_liveDriverAllocations;
                return;
            }
        }
        return;
    }
    if (block.pool >= m_pools.Size() || block.heap >= m_pools[block.pool].heaps.Size())
    {
        return;
    }
    Heap& heap = m_pools[block.pool].heaps[block.heap];
    if (heap.memory != block.memory || heap.usedBytes < block.sizeBytes)
    {
        return;
    }

    DynamicArray<Range>& ranges = heap.freeRanges;
    usize next = 0;
    while (next < ranges.Size() && ranges[next].offset < block.offset)
    {
        ++next;
    }
    const u64 end = block.offset + block.sizeBytes;
    const bool mergesPrevious =
        next > 0 && ranges[next - 1].offset + ranges[next - 1].sizeBytes == block.offset;
    const bool mergesNext = next < ranges.Size() && ranges[next].offset == end;
    if (mergesPrevious && mergesNext)
    {
        ranges[next - 1].sizeBytes += block.sizeBytes + ranges[next].sizeBytes;
        RemoveAt(ranges, next);
    }
    else if (mergesPrevious)
    {
        ranges[next - 1].sizeBytes += block.sizeBytes;
    }
    else if (mergesNext)
    {
        ranges[next] = Range{block.offset, block.sizeBytes + ranges[next].sizeBytes};
    }
    else if (!InsertAt(ranges, next, Range{block.offset, block.sizeBytes}))
    {
        // The bookkeeping could not grow, so the range stays reserved: it is
        // reported as used rather than lost, and Shutdown still releases the block.
        return;
    }
    heap.usedBytes -= block.sizeBytes;
}

Status<Error> VulkanMemoryAllocator::CreateBuffer(const VulkanBufferDesc& desc,
                                                  VulkanBuffer* out) noexcept
{
    if (out == nullptr || m_device == nullptr)
    {
        return Status<Error>::Err(
            Error{ErrorCode::InvalidArgument, "memory allocator is not initialized"});
    }
    if (desc.sizeBytes == 0 || desc.usage == 0)
    {
        return Status<Error>::Err(
            Error{ErrorCode::InvalidArgument, "a buffer needs a size and a usage"});
    }
    if ((desc.usage & vk::BufferUsage::kShaderDeviceAddress) != 0 && !m_deviceAddress)
    {
        return Status<Error>::Err(
            Error{ErrorCode::Unsupported, "buffer device address is not enabled on this device"});
    }

    vk::BufferCreateInfo info{};
    info.size = desc.sizeBytes;
    info.usage = desc.usage;
    if (m_sharedFamilyCount > 1)
    {
        info.sharingMode = vk::SharingMode::Concurrent;
        info.queueFamilyIndexCount = m_sharedFamilyCount;
        info.pQueueFamilyIndices = m_sharedFamilies;
    }
    vk::Buffer buffer = 0;
    if (m_functions.createBuffer(m_device, &info, nullptr, &buffer) != vk::Result::Success)
    {
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkCreateBuffer failed"});
    }
    vk::MemoryRequirements requirements{};
    m_functions.getBufferMemoryRequirements(m_device, buffer, &requirements);

    VulkanMemoryRequest request{};
    request.sizeBytes = requirements.size;
    request.alignmentBytes = requirements.alignment;
    request.memoryTypeBits = requirements.memoryTypeBits;
    request.memoryClass = desc.memoryClass;
    request.linear = true;
    VulkanMemoryBlock block{};
    Status<Error> allocated = Allocate(request, &block);
    if (!allocated)
    {
        m_functions.destroyBuffer(m_device, buffer, nullptr);
        return allocated;
    }
    if (m_functions.bindBufferMemory(m_device, buffer, block.memory, block.offset) !=
        vk::Result::Success)
    {
        Free(block);
        m_functions.destroyBuffer(m_device, buffer, nullptr);
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkBindBufferMemory failed"});
    }
    out->buffer = buffer;
    out->block = block;
    out->sizeBytes = desc.sizeBytes;
    return Status<Error>::Ok();
}

void VulkanMemoryAllocator::DestroyBuffer(VulkanBuffer& buffer) noexcept
{
    if (m_device == nullptr || buffer.buffer == 0)
    {
        return;
    }
    m_functions.destroyBuffer(m_device, buffer.buffer, nullptr);
    Free(buffer.block);
    buffer = VulkanBuffer{};
}

MemoryStats VulkanMemoryAllocator::Statistics() const noexcept
{
    MemoryStats stats{};
    MemoryHeapStats* classes[3] = {&stats.deviceLocal, &stats.upload, &stats.readback};
    for (usize poolIndex = 0; poolIndex < m_pools.Size(); ++poolIndex)
    {
        const Pool& pool = m_pools[poolIndex];
        MemoryHeapStats& target = *classes[static_cast<u8>(pool.memoryClass)];
        for (usize heapIndex = 0; heapIndex < pool.heaps.Size(); ++heapIndex)
        {
            const Heap& heap = pool.heaps[heapIndex];
            ++target.blockCount;
            target.reservedBytes += heap.sizeBytes;
            target.usedBytes += heap.usedBytes;
            for (usize rangeIndex = 0; rangeIndex < heap.freeRanges.Size(); ++rangeIndex)
            {
                if (heap.freeRanges[rangeIndex].sizeBytes > target.largestFreeBytes)
                {
                    target.largestFreeBytes = heap.freeRanges[rangeIndex].sizeBytes;
                }
            }
        }
    }
    for (usize index = 0; index < m_dedicated.Size(); ++index)
    {
        MemoryHeapStats& target = *classes[static_cast<u8>(m_dedicated[index].memoryClass)];
        ++target.blockCount;
        target.reservedBytes += m_dedicated[index].sizeBytes;
        target.usedBytes += m_dedicated[index].sizeBytes;
    }
    // Alignment padding returns to the free list instead of being lost, so
    // `wasteBytes` stays zero by construction.
    return stats;
}


Status<Error> VulkanMemoryAllocator::CreateImage(const VulkanImageDesc& desc,
                                                 VulkanImage* out) noexcept
{
    if (out == nullptr || m_device == nullptr)
    {
        return Status<Error>::Err(
            Error{ErrorCode::InvalidArgument, "memory allocator is not initialized"});
    }
    if (desc.width == 0 || desc.height == 0 || desc.depth == 0 || desc.mipLevels == 0 ||
        desc.arrayLayers == 0 || desc.format == vk::Format::Undefined || desc.usage == 0)
    {
        return Status<Error>::Err(
            Error{ErrorCode::InvalidArgument, "an image needs extent, mips, layers, format, usage"});
    }

    vk::ImageCreateInfo info{};
    info.imageType = desc.type;
    info.format = desc.format;
    info.extent = vk::Extent3D{desc.width, desc.height, desc.depth};
    info.mipLevels = desc.mipLevels;
    info.arrayLayers = desc.arrayLayers;
    info.tiling = desc.tiling;
    info.usage = desc.usage;
    info.samples = desc.samples;
    if (m_sharedFamilyCount > 1)
    {
        info.sharingMode = vk::SharingMode::Concurrent;
        info.queueFamilyIndexCount = m_sharedFamilyCount;
        info.pQueueFamilyIndices = m_sharedFamilies;
    }
    // Mutable format lets a view reinterpret the texels (an sRGB view of a unorm
    // texture), which the RHI allows; cube-compatible lets six layers be faces.
    info.flags = 0x8U | (desc.cubeCompatible ? 0x10U : 0U);
    vk::Image image = 0;
    if (m_ext.createImage(m_device, &info, nullptr, &image) != vk::Result::Success)
    {
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkCreateImage failed"});
    }
    vk::MemoryRequirements requirements{};
    m_ext.getImageMemoryRequirements(m_device, image, &requirements);

    VulkanMemoryRequest request{};
    request.sizeBytes = requirements.size;
    request.alignmentBytes = requirements.alignment;
    request.memoryTypeBits = requirements.memoryTypeBits;
    request.memoryClass = VulkanMemoryClass::DeviceLocal;
    request.linear = desc.tiling == vk::ImageTiling::Linear;
    VulkanMemoryBlock block{};
    Status<Error> allocated = Allocate(request, &block);
    if (!allocated)
    {
        m_ext.destroyImage(m_device, image, nullptr);
        return allocated;
    }
    if (m_ext.bindImageMemory(m_device, image, block.memory, block.offset) != vk::Result::Success)
    {
        Free(block);
        m_ext.destroyImage(m_device, image, nullptr);
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkBindImageMemory failed"});
    }
    out->image = image;
    out->block = block;
    out->desc = desc;
    return Status<Error>::Ok();
}

void VulkanMemoryAllocator::DestroyImage(VulkanImage& image) noexcept
{
    if (m_device == nullptr || image.image == 0)
    {
        return;
    }
    m_ext.destroyImage(m_device, image.image, nullptr);
    Free(image.block);
    image = VulkanImage{};
}

Status<Error> VulkanMemoryAllocator::CreateImageView(const VulkanImage& image,
                                                     vk::ImageViewType type, vk::Flags aspect,
                                                     u32 baseMip, u32 mipCount, u32 baseLayer,
                                                     u32 layerCount, vk::ImageView* out) noexcept
{
    if (out == nullptr || m_device == nullptr || !image.IsValid() || mipCount == 0 ||
        layerCount == 0 || baseMip + mipCount > image.desc.mipLevels ||
        baseLayer + layerCount > image.desc.arrayLayers)
    {
        return Status<Error>::Err(
            Error{ErrorCode::InvalidArgument, "image view range is outside the image"});
    }
    vk::ImageViewCreateInfo info{};
    info.image = image.image;
    info.viewType = type;
    info.format = image.desc.format;
    info.subresourceRange =
        vk::ImageSubresourceRange{aspect, baseMip, mipCount, baseLayer, layerCount};
    if (m_ext.createImageView(m_device, &info, nullptr, out) != vk::Result::Success)
    {
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkCreateImageView failed"});
    }
    return Status<Error>::Ok();
}

void VulkanMemoryAllocator::DestroyImageView(vk::ImageView view) noexcept
{
    if (m_device != nullptr && view != 0)
    {
        m_ext.destroyImageView(m_device, view, nullptr);
    }
}

Status<Error> VulkanMemoryAllocator::CreateSampler(const vk::SamplerCreateInfo& info,
                                                   vk::Sampler* out) noexcept
{
    if (out == nullptr || m_device == nullptr)
    {
        return Status<Error>::Err(
            Error{ErrorCode::InvalidArgument, "memory allocator is not initialized"});
    }
    if (m_ext.createSampler(m_device, &info, nullptr, out) != vk::Result::Success)
    {
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkCreateSampler failed"});
    }
    return Status<Error>::Ok();
}

void VulkanMemoryAllocator::DestroySampler(vk::Sampler sampler) noexcept
{
    if (m_device != nullptr && sampler != 0)
    {
        m_ext.destroySampler(m_device, sampler, nullptr);
    }
}

}  // namespace cyber::rhi::detail
