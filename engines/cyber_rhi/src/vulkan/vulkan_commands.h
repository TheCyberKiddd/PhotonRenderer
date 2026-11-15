#pragma once

// Command recording, submission and the timeline fence for the Vulkan backend.
// Phase 34 increment 3 (ADR 0029): one command pool and one timeline semaphore per
// device, with submission through `vkQueueSubmit2`. The timeline semaphore is the
// RHI fence of ADR 0016, a value that only moves forward, so a Vulkan fence wait
// takes a value exactly as the D3D12 one does.

#include "vulkan_context.h"

#include "vk_api_ext.h"

namespace cyber::rhi::detail
{

// Binary semaphores a submit may wait on or signal, for presentation. Both are
// optional; the timeline semaphore is always signalled.
struct VulkanSubmitSync
{
    vk::Semaphore waitBinary = 0;
    vk::Flags64 waitStage = vk::Stage2::kAllCommands;
    vk::Semaphore signalBinary = 0;
};

class VulkanCommandSystem
{
public:
    VulkanCommandSystem() noexcept = default;
    VulkanCommandSystem(const VulkanCommandSystem&) = delete;
    VulkanCommandSystem& operator=(const VulkanCommandSystem&) = delete;
    VulkanCommandSystem(VulkanCommandSystem&& other) noexcept;
    VulkanCommandSystem& operator=(VulkanCommandSystem&& other) noexcept;
    ~VulkanCommandSystem();

    // Creates the pool on the graphics family and the timeline semaphore. The
    // context must outlive the system.
    [[nodiscard]] static Result<VulkanCommandSystem, Error> Create(VulkanContext& context) noexcept;

    [[nodiscard]] bool IsValid() const noexcept
    {
        return m_pool != 0;
    }

    // Allocates a primary command buffer and begins it for one submission.
    [[nodiscard]] Status<Error> BeginRecording(vk::CommandBuffer* out) noexcept;

    // Ends the buffer and submits it on the graphics queue, signalling the next
    // timeline value, which is returned in `signalValue`. The buffer is not freed:
    // `Recycle` it once `CompletedValue()` has reached the returned value.
    [[nodiscard]] Status<Error> Submit(vk::CommandBuffer commands, const VulkanSubmitSync& sync,
                                       u64* signalValue) noexcept;

    // Blocks until the timeline reaches `value`. `ErrorCode::Busy` on timeout.
    [[nodiscard]] Status<Error> Wait(u64 value, u64 timeoutNanoseconds = ~0ULL) noexcept;

    // The highest value the GPU has signalled, without blocking.
    [[nodiscard]] u64 CompletedValue() const noexcept;

    // The value the next submit will signal.
    [[nodiscard]] u64 NextValue() const noexcept
    {
        return m_nextValue;
    }

    // Frees a buffer whose submission has completed.
    void Recycle(vk::CommandBuffer commands) noexcept;

    // Records with `record`, submits, waits for completion, and frees. For
    // uploads, readbacks and tests; a frame path batches instead.
    using RecordFunction = void (*)(vk::CommandBuffer commands, void* user);
    [[nodiscard]] Status<Error> RunImmediate(RecordFunction record, void* user) noexcept;

    [[nodiscard]] vk::Semaphore Timeline() const noexcept
    {
        return m_timeline;
    }

private:
    void Release() noexcept;

    VulkanContext* m_context = nullptr;
    vk::CommandPool m_pool = 0;
    vk::Semaphore m_timeline = 0;
    u64 m_nextValue = 1;
};

// Records one barrier batch.
inline void RecordBarriers(const vk::DeviceApiExt& ext, vk::CommandBuffer commands,
                           const vk::BufferMemoryBarrier2* buffers, u32 bufferCount,
                           const vk::ImageMemoryBarrier2* images, u32 imageCount) noexcept
{
    vk::DependencyInfo dependency{};
    dependency.bufferMemoryBarrierCount = bufferCount;
    dependency.pBufferMemoryBarriers = buffers;
    dependency.imageMemoryBarrierCount = imageCount;
    dependency.pImageMemoryBarriers = images;
    ext.cmdPipelineBarrier2(commands, &dependency);
}

}  // namespace cyber::rhi::detail
