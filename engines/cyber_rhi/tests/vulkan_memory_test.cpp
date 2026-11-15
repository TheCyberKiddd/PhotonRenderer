#include "../src/vulkan/vulkan_context.h"
#include "../src/vulkan/vulkan_memory.h"

#include <cyber/memory/heap_allocator.h>
#include <cyber/test/test.h>

#include <cstdio>
#include <cstring>
#include <utility>

using namespace cyber::rhi;
using namespace cyber::rhi::detail;
using cyber::core::ErrorCode;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::u8;
using cyber::memory::HeapAllocator;

namespace
{

constexpr u64 kMiB = 1024ULL * 1024ULL;

// Same policy as `vulkan_context_test.cpp`: validation required when installed,
// skipped (not failed) on a machine with no usable Vulkan 1.3 device.
[[nodiscard]] bool CreateContext(HeapAllocator& allocator, VulkanContext& out) noexcept
{
    VulkanContextDesc desc{};
    desc.requireValidation = true;
    auto created = VulkanContext::Create(desc, allocator);
    if (!created && created.Error().code == ErrorCode::NotFound)
    {
        desc.requireValidation = false;
        created = VulkanContext::Create(desc, allocator);
    }
    if (!created)
    {
        std::printf("  skipped: %s\n", created.Error().message);
        return false;
    }
    out = std::move(created.Value());
    return true;
}

[[nodiscard]] bool InitAllocator(VulkanContext& context, VulkanMemoryAllocator& memory) noexcept
{
    return memory
        .Initialize(context.Device(), context.DeviceFunctions(), context.DeviceFunctionsExt(),
                    context.MemoryProperties(), context.Info().bufferDeviceAddress)
        .IsOk();
}

constexpr vk::Flags kStorageAndTransfer =
    vk::BufferUsage::kStorageBuffer | vk::BufferUsage::kTransferSrc | vk::BufferUsage::kTransferDst;

}  // namespace

CG_TEST(vulkan_memory_serves_every_class)
{
    HeapAllocator heap;
    VulkanContext context;
    if (!CreateContext(heap, context))
    {
        return;
    }
    VulkanMemoryAllocator memory(heap);
    CG_REQUIRE(InitAllocator(context, memory));

    VulkanBuffer deviceLocal;
    VulkanBuffer upload;
    VulkanBuffer readback;
    CG_REQUIRE(memory
                   .CreateBuffer({64 * 1024, kStorageAndTransfer, VulkanMemoryClass::DeviceLocal},
                                 &deviceLocal)
                   .IsOk());
    CG_REQUIRE(
        memory.CreateBuffer({64 * 1024, kStorageAndTransfer, VulkanMemoryClass::Upload}, &upload)
            .IsOk());
    CG_REQUIRE(
        memory
            .CreateBuffer({64 * 1024, kStorageAndTransfer, VulkanMemoryClass::Readback}, &readback)
            .IsOk());

    // Device-local memory is not mapped; the host-visible classes are, for life.
    CG_EXPECT_TRUE(deviceLocal.block.mapped == nullptr);
    CG_REQUIRE(upload.block.mapped != nullptr);
    CG_REQUIRE(readback.block.mapped != nullptr);

    // The mapping is real memory: what is written is read back.
    auto* bytes = static_cast<u8*>(upload.block.mapped);
    for (u32 index = 0; index < 4096; ++index)
    {
        bytes[index] = static_cast<u8>(index * 31U + 7U);
    }
    bool intact = true;
    for (u32 index = 0; index < 4096; ++index)
    {
        intact = intact && bytes[index] == static_cast<u8>(index * 31U + 7U);
    }
    CG_EXPECT_TRUE(intact);

    const MemoryStats stats = memory.Statistics();
    CG_EXPECT_EQ(stats.deviceLocal.blockCount, u32{1});
    CG_EXPECT_EQ(stats.deviceLocal.usedBytes, deviceLocal.block.sizeBytes);
    CG_EXPECT_TRUE(stats.upload.reservedBytes >= 8 * kMiB);
    CG_EXPECT_TRUE(stats.readback.reservedBytes >= 8 * kMiB);

    memory.DestroyBuffer(deviceLocal);
    memory.DestroyBuffer(upload);
    memory.DestroyBuffer(readback);
    CG_EXPECT_EQ(memory.Statistics().UsedBytes(), u64{0});
    CG_EXPECT_EQ(context.ValidationErrorCount(), u32{0});
}

CG_TEST(vulkan_memory_suballocates_many_buffers_in_few_blocks)
{
    HeapAllocator heap;
    VulkanContext context;
    if (!CreateContext(heap, context))
    {
        return;
    }
    VulkanMemoryAllocator memory(heap);
    CG_REQUIRE(InitAllocator(context, memory));

    constexpr u32 kCount = 400;
    VulkanBuffer buffers[kCount];
    for (u32 index = 0; index < kCount; ++index)
    {
        // Sizes vary so alignment padding and splits are exercised.
        const u64 size = 256 + (index % 17) * 193;
        CG_REQUIRE(memory
                       .CreateBuffer({size, kStorageAndTransfer, VulkanMemoryClass::DeviceLocal},
                                     &buffers[index])
                       .IsOk());
    }

    // 400 buffers share the 8 MiB first block: far fewer driver allocations than
    // buffers, which is what keeps a real scene under the allocation count cap.
    CG_EXPECT_TRUE(memory.LiveDriverAllocations() <= 2);

    // No two live ranges in one block overlap.
    bool overlap = false;
    for (u32 a = 0; a < kCount; ++a)
    {
        for (u32 b = a + 1; b < kCount; ++b)
        {
            if (buffers[a].block.memory != buffers[b].block.memory)
            {
                continue;
            }
            const u64 aEnd = buffers[a].block.offset + buffers[a].block.sizeBytes;
            const u64 bEnd = buffers[b].block.offset + buffers[b].block.sizeBytes;
            overlap = overlap || (buffers[a].block.offset < bEnd && buffers[b].block.offset < aEnd);
        }
    }
    CG_EXPECT_FALSE(overlap);

    for (u32 index = 0; index < kCount; ++index)
    {
        memory.DestroyBuffer(buffers[index]);
    }
    CG_EXPECT_EQ(context.ValidationErrorCount(), u32{0});
}

CG_TEST(vulkan_memory_merges_freed_ranges_back_into_one)
{
    HeapAllocator heap;
    VulkanContext context;
    if (!CreateContext(heap, context))
    {
        return;
    }
    VulkanMemoryAllocator memory(heap);
    CG_REQUIRE(InitAllocator(context, memory));

    constexpr u32 kCount = 64;
    VulkanBuffer buffers[kCount];
    for (u32 index = 0; index < kCount; ++index)
    {
        CG_REQUIRE(memory
                       .CreateBuffer({4096, kStorageAndTransfer, VulkanMemoryClass::Upload},
                                     &buffers[index])
                       .IsOk());
    }
    const u64 reserved = memory.Statistics().upload.reservedBytes;

    // Free evens, then odds: every free after the first touches a neighbour on at
    // least one side, so a missing merge leaves the free list fragmented.
    for (u32 index = 0; index < kCount; index += 2)
    {
        memory.DestroyBuffer(buffers[index]);
    }
    for (u32 index = 1; index < kCount; index += 2)
    {
        memory.DestroyBuffer(buffers[index]);
    }

    const MemoryStats stats = memory.Statistics();
    CG_EXPECT_EQ(stats.upload.usedBytes, u64{0});
    // One merged range spans the whole block again.
    CG_EXPECT_EQ(stats.upload.largestFreeBytes, reserved);
    CG_EXPECT_EQ(context.ValidationErrorCount(), u32{0});
}

CG_TEST(vulkan_memory_reuses_freed_space_without_growing)
{
    HeapAllocator heap;
    VulkanContext context;
    if (!CreateContext(heap, context))
    {
        return;
    }
    VulkanMemoryAllocator memory(heap);
    CG_REQUIRE(InitAllocator(context, memory));

    VulkanBuffer buffer;
    for (u32 round = 0; round < 200; ++round)
    {
        CG_REQUIRE(
            memory
                .CreateBuffer({100 * 1024, kStorageAndTransfer, VulkanMemoryClass::DeviceLocal},
                              &buffer)
                .IsOk());
        memory.DestroyBuffer(buffer);
    }
    // Churn does not leak or open new blocks: one block, nothing used.
    const MemoryStats stats = memory.Statistics();
    CG_EXPECT_EQ(stats.deviceLocal.blockCount, u32{1});
    CG_EXPECT_EQ(stats.deviceLocal.usedBytes, u64{0});
}

CG_TEST(vulkan_memory_gives_large_requests_a_dedicated_block)
{
    HeapAllocator heap;
    VulkanContext context;
    if (!CreateContext(heap, context))
    {
        return;
    }
    VulkanMemoryAllocator memory(heap);
    CG_REQUIRE(InitAllocator(context, memory));

    // 64 MiB is a quarter of the largest block, the dedicated threshold.
    VulkanBuffer large;
    const auto created = memory.CreateBuffer(
        {64 * kMiB, kStorageAndTransfer, VulkanMemoryClass::DeviceLocal}, &large);
    if (!created)
    {
        std::printf("  skipped: %s\n", created.Error().message);
        return;
    }
    CG_EXPECT_TRUE(large.block.dedicated);
    CG_EXPECT_EQ(memory.LiveDriverAllocations(), u32{1});
    CG_EXPECT_EQ(memory.Statistics().deviceLocal.usedBytes, large.block.sizeBytes);

    memory.DestroyBuffer(large);
    CG_EXPECT_EQ(memory.LiveDriverAllocations(), u32{0});
    CG_EXPECT_EQ(memory.Statistics().deviceLocal.blockCount, u32{0});
    CG_EXPECT_EQ(context.ValidationErrorCount(), u32{0});
}

CG_TEST(vulkan_memory_rejects_bad_requests)
{
    HeapAllocator heap;
    VulkanContext context;
    if (!CreateContext(heap, context))
    {
        return;
    }
    VulkanMemoryAllocator memory(heap);
    CG_REQUIRE(InitAllocator(context, memory));

    VulkanBuffer buffer;
    CG_EXPECT_FALSE(
        memory.CreateBuffer({0, kStorageAndTransfer, VulkanMemoryClass::DeviceLocal}, &buffer)
            .IsOk());
    CG_EXPECT_FALSE(memory.CreateBuffer({256, 0, VulkanMemoryClass::DeviceLocal}, &buffer).IsOk());
    CG_EXPECT_FALSE(buffer.IsValid());

    VulkanMemoryBlock block;
    VulkanMemoryRequest request{};
    request.sizeBytes = 256;
    request.alignmentBytes = 3;  // not a power of two
    request.memoryTypeBits = 0xFFFFFFFFU;
    const auto badAlignment = memory.Allocate(request, &block);
    CG_REQUIRE(!badAlignment);
    CG_EXPECT_TRUE(badAlignment.Error().code == ErrorCode::OutOfRange);

    request.alignmentBytes = 256;
    request.memoryTypeBits = 0;  // no type allowed
    const auto noType = memory.Allocate(request, &block);
    CG_REQUIRE(!noType);
    CG_EXPECT_TRUE(noType.Error().code == ErrorCode::Unsupported);

    // Freeing something that was never allocated is ignored.
    memory.Free(VulkanMemoryBlock{});
    CG_EXPECT_EQ(memory.Statistics().UsedBytes(), u64{0});
}

CG_TEST(vulkan_memory_shutdown_releases_everything)
{
    HeapAllocator heap;
    const auto before = heap.Stats().used;
    {
        VulkanContext context;
        if (!CreateContext(heap, context))
        {
            return;
        }
        {
            VulkanMemoryAllocator memory(heap);
            CG_REQUIRE(InitAllocator(context, memory));
            VulkanBuffer buffer;
            // Deliberately never destroyed: Shutdown must release the block.
            CG_REQUIRE(
                memory.CreateBuffer({4096, kStorageAndTransfer, VulkanMemoryClass::Upload}, &buffer)
                    .IsOk());
        }
        CG_EXPECT_EQ(context.ValidationErrorCount(), u32{0});
    }
    CG_EXPECT_EQ(heap.Stats().used, before);
}
