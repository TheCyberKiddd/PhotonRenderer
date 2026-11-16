#pragma once

// The state one Vulkan RHI device owns, shared by the translation units that
// implement its entry points (ADR 0029): slot tables with generation-checked
// handles, and the device itself. Private to `src/vulkan`.

#if defined(CYBER_RHI_HAS_VULKAN)

#include "hlsl_spirv.h"
#include "vulkan_context.h"
#include "vulkan_memory.h"
#include "vulkan_pipelines.h"
#include "vulkan_swapchain.h"

#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/handle.h>
#include <cyber/rhi/binding.h>
#include <cyber/rhi/buffer.h>
#include <cyber/rhi/capabilities.h>
#include <cyber/rhi/command_list.h>
#include <cyber/rhi/format.h>
#include <cyber/rhi/pipeline.h>
#include <cyber/rhi/render_pass.h>
#include <cyber/rhi/sampler.h>
#include <cyber/rhi/shader.h>
#include <cyber/rhi/swapchain.h>
#include <cyber/rhi/texture.h>

#include <cstring>
#include <mutex>
#include <new>

namespace cyber::rhi::detail
{
struct VulkanRhiDevice;
struct CommandListData;
struct SwapChainData;

namespace dev
{

using cyber::core::ErrorCode;
using cyber::memory::DynamicArray;

[[nodiscard]] inline Status<Error> Err(ErrorCode code, const char* message) noexcept
{
    return Status<Error>::Err(Error{code, message});
}

constexpr u32 kMaxTrackedSubresources = 64;
constexpr u32 kNameBytes = 128;

// A shader compiler diagnostic, copied to storage that outlives the compile so an
// `Error` can point at it. One buffer per thread: valid until that thread's next
// failed shader compile.
[[nodiscard]] inline const char* KeepShaderDiagnostic(const char* message) noexcept
{
    thread_local char buffer[384];
    usize length = 0;
    if (message != nullptr)
    {
        while (length + 1 < sizeof(buffer) && message[length] != '\0')
        {
            buffer[length] = message[length];
            ++length;
        }
    }
    buffer[length] = '\0';
    return buffer;
}

inline void CopyLabel(char (&destination)[kNameBytes], const char* label) noexcept
{
    destination[0] = '\0';
    if (label == nullptr)
    {
        return;
    }
    u32 length = 0;
    while (length + 1 < kNameBytes && label[length] != '\0')
    {
        destination[length] = label[length];
        ++length;
    }
    destination[length] = '\0';
}

// The Vulkan format of each RHI format, in `Format` order.
constexpr u32 kFormatTable[] = {
    9,    // R8Unorm
    16,   // RG8Unorm
    37,   // RGBA8Unorm
    43,   // RGBA8UnormSrgb
    44,   // BGRA8Unorm
    50,   // BGRA8UnormSrgb
    76,   // R16Float
    83,   // RG16Float
    97,   // RGBA16Float
    100,  // R32Float
    103,  // RG32Float
    106,  // R32G32B32Float
    109,  // RGBA32Float
    98,   // R32Uint
    101,  // RG32Uint
    107,  // RGBA32Uint
    64,   // R10G10B10A2Unorm (A2B10G10R10 pack: red in the low bits)
    122,  // RG11B10Float (B10G11R11 ufloat pack)
    124,  // D16Unorm
    126,  // D32Float
    129,  // D24UnormS8Uint
    130,  // D32FloatS8Uint
    133,  // BC1Unorm (RGBA variant)
    134,  // BC1UnormSrgb
    137,  // BC3Unorm
    138,  // BC3UnormSrgb
    141,  // BC5Unorm
    145,  // BC7Unorm
    146,  // BC7UnormSrgb
    143,  // BC6HUfloat
    144,  // BC6HSfloat
};
static_assert(sizeof(kFormatTable) / sizeof(kFormatTable[0]) == kFormatCount,
              "the Vulkan format table must cover every RHI format");

[[nodiscard]] inline vk::Format ToVulkanFormat(Format format) noexcept
{
    const u32 index = static_cast<u32>(format);
    return index < kFormatCount ? static_cast<vk::Format>(kFormatTable[index])
                                : vk::Format::Undefined;
}

[[nodiscard]] inline bool IsBlockCompressed(Format format) noexcept
{
    return FormatInfoOf(format).compressed;
}

// Slot storage with generation-checked handles, the same scheme as the D3D12
// backend's tables.
template <typename Slot>
class SlotTable
{
public:
    explicit SlotTable(memory::Allocator& allocator) noexcept
        : m_items(allocator), m_free(allocator)
    {
    }

    // Claims a slot, zeroed, and returns its index; false on allocation failure.
    [[nodiscard]] bool Acquire(u32* index) noexcept
    {
        if (!m_free.Empty())
        {
            *index = m_free.Back();
            m_free.PopBack();
        }
        else
        {
            if (!m_items.TryEmplaceBack())
            {
                return false;
            }
            *index = static_cast<u32>(m_items.Size() - 1);
        }
        Slot& slot = m_items[*index];
        const u32 generation = slot.generation == 0 ? 1U : slot.generation;
        slot = Slot{};
        slot.generation = generation;
        slot.alive = true;
        return true;
    }

    template <typename Tag>
    [[nodiscard]] Slot* Resolve(memory::Handle<Tag> handle) noexcept
    {
        if (!handle.IsValid() || handle.index >= m_items.Size())
        {
            return nullptr;
        }
        Slot& slot = m_items[handle.index];
        return slot.alive && slot.generation == handle.generation ? &slot : nullptr;
    }

    template <typename Tag>
    [[nodiscard]] const Slot* Resolve(memory::Handle<Tag> handle) const noexcept
    {
        return const_cast<SlotTable*>(this)->Resolve(handle);
    }

    template <typename Tag>
    [[nodiscard]] memory::Handle<Tag> HandleOf(u32 index) const noexcept
    {
        memory::Handle<Tag> handle;
        handle.index = index;
        handle.generation = m_items[index].generation;
        return handle;
    }

    // Retires a slot: the generation moves on so every old handle goes stale.
    void Release(u32 index) noexcept
    {
        Slot& slot = m_items[index];
        const u32 next = slot.generation + 1U == 0 ? 1U : slot.generation + 1U;
        slot = Slot{};
        slot.generation = next;
        (void)m_free.TryPushBack(index);
    }

    [[nodiscard]] usize Size() const noexcept
    {
        return m_items.Size();
    }
    [[nodiscard]] Slot& At(usize index) noexcept
    {
        return m_items[index];
    }
    [[nodiscard]] u32 AliveCount() const noexcept
    {
        u32 count = 0;
        for (usize index = 0; index < m_items.Size(); ++index)
        {
            count += m_items[index].alive ? 1U : 0U;
        }
        return count;
    }

private:
    DynamicArray<Slot> m_items;
    DynamicArray<u32> m_free;
};

struct BufferSlot
{
    VulkanBuffer buffer{};
    BufferDesc desc{};
    ResourceState state = ResourceState::Undefined;
    char name[kNameBytes] = {};
    bool alive = false;
    u32 generation = 0;
};

struct TextureSlot
{
    VulkanImage image{};
    // A swapchain image: owned by the swapchain, never destroyed or freed here.
    bool external = false;
    TextureDesc desc{};
    u64 sizeBytes = 0;
    // Reserved (sparse-resident) textures: the driver's tiling, where the packed
    // mip tail starts in the image's opaque address space, and one mapping entry
    // per tile in `validation::TileLinearIndex` order (device allocator).
    bool reserved = false;
    TextureTiling tiling{};
    u64 mipTailOffset = 0;
    u32 sparseTypeBits = 0;
    TileMappingInfo* tiles = nullptr;
    ResourceState states[kMaxTrackedSubresources] = {};
    char name[kNameBytes] = {};
    bool alive = false;
    u32 generation = 0;
};

struct ViewSlot
{
    TextureHandle texture{};
    TextureViewDesc desc{};
    vk::ImageView view = 0;
    char name[kNameBytes] = {};
    bool alive = false;
    u32 generation = 0;
};

struct SamplerSlot
{
    SamplerDesc desc{};
    vk::Sampler sampler = 0;
    char name[kNameBytes] = {};
    bool alive = false;
    u32 generation = 0;
};

struct ShaderSlot
{
    // SPIR-V shaders have a module from creation. HLSL shaders keep their source
    // and are compiled per pipeline, against that pipeline's layout
    // (`hlsl_spirv.h`), so `module` stays null for them.
    vk::ShaderModule module = 0;
    ShaderBytecodeFormat format = ShaderBytecodeFormat::SpirV;
    char* source = nullptr;  // device allocator, null-terminated
    usize sourceBytes = 0;
    char entry[64] = {};
    ShaderStage stage = ShaderStage::Vertex;
    u64 codeSize = 0;
    char name[kNameBytes] = {};
    bool alive = false;
    u32 generation = 0;
};

struct FenceSlot
{
    vk::Semaphore semaphore = 0;
    char name[kNameBytes] = {};
    bool alive = false;
    u32 generation = 0;
};

inline constexpr u32 kMaxLayoutEntries = 64;
inline constexpr u32 kMaxVertexBindings = 32;
inline constexpr u32 kMaxSwapChainImages = 4;

struct LayoutSlot
{
    vk::DescriptorSetLayout layout = 0;
    BindGroupLayoutEntry entries[kMaxLayoutEntries] = {};
    u32 entryCount = 0;
    // Descriptors in the set, arrays flattened: what `BindGroupDesc::resources` supplies.
    u32 descriptorCount = 0;
    u32 dynamicOffsetCount = 0;
    // Array bindings are update-after-bind and partially bound (bindless); sets of
    // this layout come from the update-after-bind pools.
    bool updateAfterBind = false;
    char name[kNameBytes] = {};
    bool alive = false;
    u32 generation = 0;
};

struct GroupSlot
{
    BindGroupLayoutHandle layout{};
    vk::DescriptorSet set = 0;
    u32 pool = 0;
    bool updateAfterBindPool = false;  // `pool` indexes `uabDescriptorPools`
    u32 descriptorCount = 0;
    char name[kNameBytes] = {};
    bool alive = false;
    u32 generation = 0;
};

struct PipelineLayoutSlot
{
    vk::PipelineLayout layout = 0;
    BindGroupLayoutHandle groups[kBindGroupSlotCount] = {};
    u32 pushConstantBytes = 0;
    char name[kNameBytes] = {};
    bool alive = false;
    u32 generation = 0;
};

struct PipelineSlot
{
    vk::Pipeline pipeline = 0;
    PipelineKind kind = PipelineKind::Graphics;
    u64 hash = 0;
    PipelineLayoutHandle layout{};
    char name[kNameBytes] = {};
    bool alive = false;
    u32 generation = 0;
};

// Command lists and swapchains hand the front end a stable `internal` pointer, so the
// table holds a pointer to heap data rather than the data itself (the table moves
// its slots when it grows). The data is defined in `vulkan_device_gpu.cpp`.
struct CommandListSlot
{
    CommandListData* data = nullptr;
    bool alive = false;
    u32 generation = 0;
};

struct SwapChainSlot
{
    SwapChainData* data = nullptr;
    bool alive = false;
    u32 generation = 0;
};

struct TimestampHeapSlot
{
    vk::QueryPool pool = 0;
    // Resolved values land here (Readback memory, persistently mapped).
    VulkanBuffer readback{};
    u32 queryCount = 0;
    char name[kNameBytes] = {};
    bool alive = false;
    u32 generation = 0;
};

struct PipelineCacheSlot
{
    vk::PipelineCache cache = 0;
    char name[kNameBytes] = {};
    bool alive = false;
    u32 generation = 0;
};

struct TileHeapSlot
{
    vk::DeviceMemory memory = 0;
    u32 tileCount = 0;
    u32 memoryType = 0;
    char name[kNameBytes] = {};
    bool alive = false;
    u32 generation = 0;
};

}  // namespace dev
using namespace dev;

// What one Vulkan RHI device owns. Two devices of this backend share nothing.
struct VulkanRhiDevice
{
    explicit VulkanRhiDevice(memory::Allocator& allocatorRef) noexcept
        : allocator(&allocatorRef), memory(allocatorRef), pipelines(context), buffers(allocatorRef),
          textures(allocatorRef), views(allocatorRef), samplers(allocatorRef),
          shaders(allocatorRef), fences(allocatorRef), layouts(allocatorRef), groups(allocatorRef),
          pipelineLayouts(allocatorRef), pipelineTable(allocatorRef), commandLists(allocatorRef),
          swapChains(allocatorRef), descriptorPools(allocatorRef), uabDescriptorPools(allocatorRef),
          initRetired(allocatorRef), timestampHeaps(allocatorRef), pipelineCaches(allocatorRef),
          tileHeaps(allocatorRef)
    {
    }
    VulkanRhiDevice(const VulkanRhiDevice&) = delete;
    VulkanRhiDevice& operator=(const VulkanRhiDevice&) = delete;

    ~VulkanRhiDevice()
    {
        ReleaseAll();
    }

    // Defined in `vulkan_device_gpu.cpp`: tears down everything the command half
    // created.
    void ReleaseGpuObjects() noexcept;

    // Destroys every live object, then the memory the resources lived in. Runs
    // before the context goes, because the driver objects need the device.
    void ReleaseAll() noexcept
    {
        if (!context.IsValid())
        {
            return;
        }
        (void)context.WaitIdle();
        const vk::DeviceApiExt& ext = context.DeviceFunctionsExt();
        for (usize index = 0; index < views.Size(); ++index)
        {
            if (views.At(index).alive)
            {
                memory.DestroyImageView(views.At(index).view);
            }
        }
        for (usize index = 0; index < samplers.Size(); ++index)
        {
            if (samplers.At(index).alive)
            {
                memory.DestroySampler(samplers.At(index).sampler);
            }
        }
        for (usize index = 0; index < shaders.Size(); ++index)
        {
            if (shaders.At(index).alive)
            {
                if (shaders.At(index).module != 0)
                {
                    pipelines.DestroyShader(shaders.At(index).module);
                }
                if (shaders.At(index).source != nullptr)
                {
                    allocator->Deallocate(shaders.At(index).source);
                    shaders.At(index).source = nullptr;
                }
            }
        }
        for (usize index = 0; index < fences.Size(); ++index)
        {
            if (fences.At(index).alive)
            {
                ext.destroySemaphore(context.Device(), fences.At(index).semaphore, nullptr);
            }
        }
        for (usize index = 0; index < timestampHeaps.Size(); ++index)
        {
            if (timestampHeaps.At(index).alive)
            {
                ext.destroyQueryPool(context.Device(), timestampHeaps.At(index).pool, nullptr);
                memory.DestroyBuffer(timestampHeaps.At(index).readback);
            }
        }
        for (usize index = 0; index < pipelineCaches.Size(); ++index)
        {
            if (pipelineCaches.At(index).alive)
            {
                ext.destroyPipelineCache(context.Device(), pipelineCaches.At(index).cache, nullptr);
            }
        }
        // Command lists, swapchains, pipelines, layouts, bind groups and the pools they
        // came from. After views and shaders, before the images they drew into.
        ReleaseGpuObjects();
        for (usize index = 0; index < textures.Size(); ++index)
        {
            if (textures.At(index).alive && !textures.At(index).external)
            {
                memory.DestroyImage(textures.At(index).image);
                if (textures.At(index).tiles != nullptr)
                {
                    allocator->Deallocate(textures.At(index).tiles);
                    textures.At(index).tiles = nullptr;
                }
            }
        }
        for (usize index = 0; index < tileHeaps.Size(); ++index)
        {
            if (tileHeaps.At(index).alive)
            {
                memory.FreeRaw(tileHeaps.At(index).memory);
            }
        }
        if (sparseZeroTile != 0)
        {
            memory.FreeRaw(sparseZeroTile);
            sparseZeroTile = 0;
        }
        for (usize index = 0; index < buffers.Size(); ++index)
        {
            if (buffers.At(index).alive)
            {
                memory.DestroyBuffer(buffers.At(index).buffer);
            }
        }
        memory.Shutdown();
    }

    memory::Allocator* allocator;
    VulkanContext context;
    VulkanMemoryAllocator memory;
    VulkanPipelineFactory pipelines;
    Capabilities capabilities{};
    mutable std::mutex mutex;
    SlotTable<BufferSlot> buffers;
    SlotTable<TextureSlot> textures;
    SlotTable<ViewSlot> views;
    SlotTable<SamplerSlot> samplers;
    SlotTable<ShaderSlot> shaders;
    SlotTable<FenceSlot> fences;
    SlotTable<LayoutSlot> layouts;
    SlotTable<GroupSlot> groups;
    SlotTable<PipelineLayoutSlot> pipelineLayouts;
    SlotTable<PipelineSlot> pipelineTable;
    SlotTable<CommandListSlot> commandLists;
    SlotTable<SwapChainSlot> swapChains;
    // Descriptor pools that bind groups allocate from, grown when one fills.
    DynamicArray<vk::DescriptorPool> descriptorPools;
    // Pools created with the update-after-bind flag, for bindless layouts.
    DynamicArray<vk::DescriptorPool> uabDescriptorPools;
    // A set layout with no bindings, standing in for an unused bind group slot so
    // set indices stay contiguous.
    vk::DescriptorSetLayout emptySetLayout = 0;
    // Orders swapchain work: acquire gates the graphics queue on the image being
    // available, present waits for every earlier graphics submit.
    // Pool for the device-level `Transition` and swapchain layout setup.
    vk::CommandPool transitionPool = 0;
    vk::Semaphore internalTimeline = 0;
    u64 internalValue = 0;
    u64 gateValue = 0;
    SubmissionInfo lastSubmission[kQueueKindCount] = {};
    u32 submissionCounter = 0;
    vk::PFN_CmdBeginDebugUtilsLabelEXT beginLabel = nullptr;
    vk::PFN_CmdEndDebugUtilsLabelEXT endLabel = nullptr;
    vk::PFN_CmdInsertDebugUtilsLabelEXT insertLabel = nullptr;
    bool drawIndirectCountSupported = false;
    // What generated SPIR-V may rely on, from the features the device enabled.
    hlsl::CompileFeatures shaderFeatures{};
    // Zero-initialisation of new resources (D3D12 committed resources start zeroed,
    // and on Intel an image placed in reused memory otherwise inherits stale
    // compression metadata, which can hang the GPU when sampled). Recorded into one
    // pending command buffer and submitted ahead of the next queue submission.
    vk::CommandBuffer initCommands = nullptr;
    struct RetiredInit
    {
        vk::CommandBuffer commands = nullptr;
        u64 value = 0;
    };
    DynamicArray<RetiredInit> initRetired;

    SlotTable<TimestampHeapSlot> timestampHeaps;
    SlotTable<PipelineCacheSlot> pipelineCaches;
    SlotTable<TileHeapSlot> tileHeaps;
    // Ordering for the compute and copy queues, as `internalTimeline` orders the
    // graphics queue: each submission to a kind signals its timeline, so a sparse
    // bind can wait for the work already on that queue and the next submission can
    // wait for the bind (D3D12's UpdateTileMappings is ordered on its queue).
    vk::Semaphore queueTimeline[kQueueKindCount] = {};
    u64 queueValue[kQueueKindCount] = {};
    // A bind the next submission on that kind must wait for (0 = none), and the
    // last bind per kind, waited for before a reserved texture or heap is freed.
    u64 pendingBind[kQueueKindCount] = {};
    u64 lastBind[kQueueKindCount] = {};
    // One zeroed tile that unmapped tiles alias when the device allows aliasing,
    // so they read as zero as D3D12 tier 2 guarantees (memory type in
    // `sparseZeroTileType`).
    vk::DeviceMemory sparseZeroTile = 0;
    u32 sparseZeroTileType = 0;
    bool sparseAliasing = false;
};

[[nodiscard]] inline VulkanRhiDevice& Self(void* internal) noexcept
{
    return *static_cast<VulkanRhiDevice*>(internal);
}
[[nodiscard]] inline const VulkanRhiDevice& Self(const void* internal) noexcept
{
    return *static_cast<const VulkanRhiDevice*>(internal);
}

// Defined in `vulkan_device_gpu.cpp`: the entries for bind groups, pipelines, command
// lists, submission, transitions and swapchains, and the objects they need.
void InstallGpuEntries(DeviceInterface& table) noexcept;
[[nodiscard]] Status<Error> InitializeGpuObjects(VulkanRhiDevice& device) noexcept;

// Defined in `vulkan_device_gpu.cpp`. All three run under the device mutex.
// Record zeroing of a new texture or device-local buffer into the pending batch.
void RecordTextureInit(VulkanRhiDevice& device, const TextureSlot& texture) noexcept;
void RecordBufferInit(VulkanRhiDevice& device, const BufferSlot& buffer) noexcept;
// Submits the pending batch on the graphics queue. `waitValue` receives the
// internal timeline value that marks its completion, or 0 when nothing was pending.
[[nodiscard]] Status<Error> FlushResourceInit(VulkanRhiDevice& device, u64* waitValue) noexcept;
// Flushes and waits, for a resource about to be destroyed while referenced by it.
[[nodiscard]] Status<Error> FlushResourceInitAndWait(VulkanRhiDevice& device) noexcept;
// Records a reset of queries [first, first + count) into the pending batch.
[[nodiscard]] bool RecordQueryReset(VulkanRhiDevice& device, vk::QueryPool pool, u32 first,
                                    u32 count) noexcept;

// Defined in `vulkan_device_features.cpp`: timestamps, pipeline caches, reserved
// textures. The `Vk*` entries lock the device; the helpers below expect it locked.
Status<Error> VkCreateTimestampQueryHeap(void* internal, const TimestampQueryHeapDesc& desc,
                                         TimestampQueryHeapHandle* out) noexcept;
Status<Error> VkDestroyTimestampQueryHeap(void* internal, TimestampQueryHeapHandle handle) noexcept;
Status<Error> VkReadTimestamps(void* internal, TimestampQueryHeapHandle handle, u32 firstQuery,
                               u32 queryCount, FenceHandle completionFence, u64 completionValue,
                               memory::Span<u64> out) noexcept;
u64 VkTimestampFrequency(const void* internal, QueueKind kind) noexcept;
Status<Error> VkCreatePipelineCache(void* internal, const PipelineCacheDesc& desc,
                                    PipelineCacheHandle* out) noexcept;
Status<Error> VkDestroyPipelineCache(void* internal, PipelineCacheHandle handle) noexcept;
Status<Error> VkPipelineCacheBytes(const void* internal, PipelineCacheHandle handle,
                                   memory::Allocator& allocator,
                                   memory::DynamicArray<memory::Byte>* out) noexcept;
Status<Error> VkCreateTileHeap(void* internal, const TileHeapDesc& desc,
                               TileHeapHandle* out) noexcept;
Status<Error> VkDestroyTileHeap(void* internal, TileHeapHandle handle) noexcept;
TextureTiling VkTextureTiling(const void* internal, TextureHandle handle) noexcept;
Status<Error> VkUpdateTileMappings(void* internal, QueueKind queue,
                                   memory::Span<const TileMapping> mappings) noexcept;
TileMappingInfo VkTileMapping(const void* internal, const TileMapping& tile) noexcept;
[[nodiscard]] vk::PipelineCache NativePipelineCache(const VulkanRhiDevice& device,
                                                    PipelineCacheHandle handle) noexcept;
// Creates the sparse image behind a reserved texture and fills the slot's tiling.
[[nodiscard]] Status<Error> CreateReservedImage(VulkanRhiDevice& device, const TextureDesc& desc,
                                                const VulkanImageDesc& native,
                                                TextureSlot& slot) noexcept;
// Points every tile of a new reserved texture at the zero tile (aliasing devices).
[[nodiscard]] Status<Error> AliasUnmappedTiles(VulkanRhiDevice& device,
                                               TextureHandle handle) noexcept;
// Waits until every sparse bind submitted so far has run.
[[nodiscard]] Status<Error> WaitForTileBinds(VulkanRhiDevice& device) noexcept;

}  // namespace cyber::rhi::detail

#endif
