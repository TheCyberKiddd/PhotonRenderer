#include "../src/vulkan/vulkan_context.h"

#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/rhi/backend.h>
#include <cyber/rhi/device.h>
#include <cyber/rhi/vulkan.h>
#include <cyber/test/test.h>

#include <cstdio>
#include <utility>

using namespace cyber::rhi;
using namespace cyber::rhi::detail;
using cyber::core::ErrorCode;
using cyber::core::u32;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;

namespace
{

inline constexpr u32 kMaxAdapters = 32;

[[nodiscard]] u32 CountAdapters(const Backend& backend) noexcept
{
    return backend.enumerateAdapters(nullptr, 0);
}

// Creates a context with validation required when the layer is installed and
// without it when it is not, so a machine without the Vulkan SDK still runs the
// test and a machine with it checks every message. Returns false and prints why
// when the machine cannot run it: no runtime, no adapter, or a device below
// Vulkan 1.3. The test is skipped there, not failed, because the machine is the
// variable and the code is not.
[[nodiscard]] bool CreateContext(HeapAllocator& allocator, VulkanContext& out) noexcept
{
    VulkanContextDesc desc{};
    desc.requireValidation = true;
    auto created = VulkanContext::Create(desc, allocator);
    if (!created && created.Error().code == ErrorCode::NotFound)
    {
        // Either no loader or no validation layer; the second attempt tells which.
        desc.requireValidation = false;
        created = VulkanContext::Create(desc, allocator);
    }
    if (!created)
    {
        std::printf("  skipped: %s (%s)\n", created.Error().message,
                    cyber::core::ErrorCodeName(created.Error().code));
        return false;
    }
    out = std::move(created.Value());
    return true;
}

}  // namespace

CG_TEST(vulkan_backend_presence_matches_the_build)
{
    CG_EXPECT_EQ(IsVulkanCompiledIn(), VulkanBackend() != nullptr);
}

CG_TEST(vulkan_backend_describes_itself)
{
    const Backend* backend = VulkanBackend();
    CG_REQUIRE(backend != nullptr);

    CG_EXPECT_TRUE(backend->kind == BackendKind::Vulkan);
    CG_EXPECT_STREQ(backend->name, "vulkan");
    CG_EXPECT_EQ(backend->apiVersion, u32{103});
    CG_EXPECT_TRUE(backend->enumerateAdapters != nullptr);
    CG_EXPECT_TRUE(VulkanBackend() == backend);
}

CG_TEST(vulkan_backend_enumerates_consistently)
{
    const Backend* backend = VulkanBackend();
    CG_REQUIRE(backend != nullptr);

    // A machine without a Vulkan runtime reports zero adapters, which is a valid
    // answer and not a failure.
    const u32 total = CountAdapters(*backend);
    if (total == 0)
    {
        std::printf("  skipped: no Vulkan adapters on this machine\n");
        return;
    }
    CG_REQUIRE(total <= kMaxAdapters);

    AdapterInfo adapters[kMaxAdapters]{};
    CG_EXPECT_EQ(EnumerateAdapters(*backend, Span<AdapterInfo>(adapters, kMaxAdapters)), total);
    for (u32 index = 0; index < total; ++index)
    {
        CG_EXPECT_TRUE(adapters[index].backend == BackendKind::Vulkan);
        CG_EXPECT_TRUE(adapters[index].name[kAdapterNameBytes - 1] == '\0');
        CG_EXPECT_TRUE(adapters[index].name[0] != '\0');
        CG_EXPECT_TRUE(adapters[index].apiVersion >= u32{100});
    }

    // A short span writes what fits and still reports the total, so a caller can
    // size its buffer from the first answer.
    AdapterInfo single{};
    single.vendorId = 0xDEADBEEF;
    CG_EXPECT_EQ(backend->enumerateAdapters(&single, 1), total);
    CG_EXPECT_NE(single.vendorId, u32{0xDEADBEEF});
    CG_EXPECT_STREQ(single.name, adapters[0].name);

    AdapterInfo untouched{};
    untouched.vendorId = 0xDEADBEEF;
    CG_EXPECT_EQ(backend->enumerateAdapters(&untouched, 0), total);
    CG_EXPECT_EQ(untouched.vendorId, u32{0xDEADBEEF});

    // The order is stable between calls, because `VulkanContext` indexes devices
    // by it.
    AdapterInfo again[kMaxAdapters]{};
    CG_EXPECT_EQ(backend->enumerateAdapters(again, kMaxAdapters), total);
    for (u32 index = 0; index < total; ++index)
    {
        CG_EXPECT_STREQ(again[index].name, adapters[index].name);
        CG_EXPECT_EQ(again[index].deviceId, adapters[index].deviceId);
    }
}

CG_TEST(vulkan_context_rejects_a_missing_adapter)
{
    HeapAllocator allocator;
    VulkanContextDesc desc{};
    desc.adapterIndex = 999;
    auto created = VulkanContext::Create(desc, allocator);
    CG_REQUIRE(!created);
    // No runtime is `NotFound`; with one, an index past the list is `OutOfRange`.
    CG_EXPECT_TRUE(created.Error().code == ErrorCode::NotFound ||
                   created.Error().code == ErrorCode::OutOfRange);
}

CG_TEST(vulkan_context_creates_and_reports_a_device)
{
    HeapAllocator allocator;
    VulkanContext context;
    if (!CreateContext(allocator, context))
    {
        return;
    }

    CG_REQUIRE(context.IsValid());
    const VulkanDeviceInfo& info = context.Info();
    std::printf("  device: %s, Vulkan %u.%u, validation %s, sync validation %s\n", info.name,
                vk::VersionMajor(info.apiVersion), vk::VersionMinor(info.apiVersion),
                info.validationLayerEnabled ? "on" : "off (layer not installed)",
                info.synchronizationValidationEnabled ? "on" : "off");

    CG_EXPECT_TRUE(info.name[0] != '\0');
    CG_EXPECT_TRUE(info.apiVersion >= vk::kApiVersion13);
    CG_EXPECT_TRUE(info.deviceLocalBytes > 0);

    // Every queue exists, and one that has no dedicated family aliases graphics.
    CG_EXPECT_TRUE(info.queues.graphics != kNoQueueFamily);
    CG_EXPECT_TRUE(context.GraphicsQueue() != nullptr);
    CG_EXPECT_TRUE(context.ComputeQueue() != nullptr);
    CG_EXPECT_TRUE(context.TransferQueue() != nullptr);
    if (!info.queues.dedicatedCompute)
    {
        CG_EXPECT_EQ(info.queues.compute, info.queues.graphics);
    }
    if (!info.queues.dedicatedTransfer)
    {
        CG_EXPECT_EQ(info.queues.transfer, info.queues.graphics);
    }
}

CG_TEST(vulkan_context_names_objects_and_idles_without_validation_errors)
{
    HeapAllocator allocator;
    VulkanContext context;
    if (!CreateContext(allocator, context))
    {
        return;
    }

    // Naming and idling are the first calls that pass handles through the layer;
    // a validation error here would mean the function table or the enabled
    // feature set is wrong.
    CG_EXPECT_TRUE(context
                       .SetObjectName(vk::ObjectType::Queue,
                                      reinterpret_cast<cyber::core::u64>(context.GraphicsQueue()),
                                      "test graphics queue")
                       .IsOk());
    CG_EXPECT_TRUE(context.WaitIdle().IsOk());

    char message[512];
    const u32 length = context.FirstValidationError(message, sizeof(message));
    if (length != 0)
    {
        std::printf("  validation: %s\n", message);
    }
    CG_EXPECT_EQ(context.ValidationErrorCount(), u32{0});
    CG_EXPECT_EQ(length, u32{0});
}

CG_TEST(vulkan_context_moves_ownership)
{
    HeapAllocator allocator;
    VulkanContext first;
    if (!CreateContext(allocator, first))
    {
        return;
    }

    const auto device = first.Device();
    VulkanContext second = std::move(first);
    CG_EXPECT_FALSE(first.IsValid());
    CG_REQUIRE(second.IsValid());
    CG_EXPECT_TRUE(second.Device() == device);

    // Assigning over a live context releases it before taking the new one.
    VulkanContext third;
    if (CreateContext(allocator, third))
    {
        second = std::move(third);
        CG_EXPECT_FALSE(third.IsValid());
        CG_EXPECT_TRUE(second.IsValid());
    }
    CG_EXPECT_TRUE(second.WaitIdle().IsOk());
}

CG_TEST(vulkan_context_creation_does_not_leak)
{
    // Each context allocates its loader state and validation counters from the
    // allocator; after teardown the allocator must be back where it started.
    HeapAllocator allocator;
    const auto before = allocator.Stats();
    {
        VulkanContext context;
        (void)CreateContext(allocator, context);
    }
    const auto after = allocator.Stats();
    CG_EXPECT_EQ(after.used, before.used);
}
