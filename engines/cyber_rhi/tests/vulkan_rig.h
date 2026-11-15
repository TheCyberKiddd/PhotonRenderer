#pragma once

// Shared fixture for the Vulkan backend tests: a context, its memory allocator and
// its command system, created together and torn down in reverse order.

#include "../src/vulkan/vulkan_barriers.h"
#include "../src/vulkan/vulkan_commands.h"
#include "../src/vulkan/vulkan_context.h"
#include "../src/vulkan/vulkan_memory.h"
#include "../src/vulkan/vulkan_pipelines.h"

#include <cyber/memory/heap_allocator.h>

#include <cstdio>
#include <utility>

namespace cyber::rhi::test
{

// Skips (not fails) a test on a machine with no usable Vulkan 1.3 device: the
// machine is the variable and the code is not. Validation is required when the
// layer is installed, so a machine that has it checks every message.
struct VulkanRig
{
    explicit VulkanRig(cyber::memory::HeapAllocator& heap) : memory(heap), pipelines(context) {}

    [[nodiscard]] bool Init(cyber::memory::HeapAllocator& heap, bool presentation = false) noexcept
    {
        detail::VulkanContextDesc desc{};
        desc.requireValidation = true;
        desc.enablePresentation = presentation;
        auto created = detail::VulkanContext::Create(desc, heap);
        if (!created && created.Error().code == cyber::core::ErrorCode::NotFound)
        {
            desc.requireValidation = false;
            created = detail::VulkanContext::Create(desc, heap);
        }
        if (!created)
        {
            std::printf("  skipped: %s\n", created.Error().message);
            return false;
        }
        context = std::move(created.Value());
        if (!memory
                 .Initialize(context.Device(), context.DeviceFunctions(),
                             context.DeviceFunctionsExt(), context.MemoryProperties(),
                             context.Info().bufferDeviceAddress)
                 .IsOk())
        {
            return false;
        }
        auto system = detail::VulkanCommandSystem::Create(context);
        if (!system)
        {
            std::printf("  command system failed: %s\n", system.Error().message);
            return false;
        }
        commands = std::move(system.Value());
        return true;
    }

    // Members are destroyed in reverse order: commands, then memory, then the
    // context they both use. `pipelines` only refers to the context.
    detail::VulkanContext context;
    detail::VulkanMemoryAllocator memory;
    detail::VulkanCommandSystem commands;
    detail::VulkanPipelineFactory pipelines;
};

// Makes device writes visible to the host: the last barrier before reading a
// readback buffer through its mapping.
inline void
HostReadBarrier(const detail::vk::DeviceApiExt& ext, detail::vk::CommandBuffer commands,
                detail::vk::Buffer buffer,
                detail::vk::Flags64 sourceStage = detail::vk::Stage2::kAllCommands,
                detail::vk::Flags64 sourceAccess = detail::vk::Access2::kMemoryWrite) noexcept
{
    detail::vk::BufferMemoryBarrier2 barrier{};
    barrier.srcStageMask = sourceStage;
    barrier.srcAccessMask = sourceAccess;
    barrier.dstStageMask = detail::vk::Stage2::kHost;
    barrier.dstAccessMask = detail::vk::Access2::kHostRead;
    barrier.buffer = buffer;
    detail::RecordBarriers(ext, commands, &barrier, 1, nullptr, 0);
}

}  // namespace cyber::rhi::test
