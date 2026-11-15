#include "vulkan_commands.h"

#include <utility>

namespace cyber::rhi::detail
{
namespace
{
using cyber::core::ErrorCode;
}

VulkanCommandSystem::VulkanCommandSystem(VulkanCommandSystem&& other) noexcept
{
    *this = std::move(other);
}

VulkanCommandSystem& VulkanCommandSystem::operator=(VulkanCommandSystem&& other) noexcept
{
    if (this != &other)
    {
        Release();
        m_context = other.m_context;
        m_pool = other.m_pool;
        m_timeline = other.m_timeline;
        m_nextValue = other.m_nextValue;
        other.m_context = nullptr;
        other.m_pool = 0;
        other.m_timeline = 0;
        other.m_nextValue = 1;
    }
    return *this;
}

VulkanCommandSystem::~VulkanCommandSystem()
{
    Release();
}

void VulkanCommandSystem::Release() noexcept
{
    if (m_context != nullptr && m_context->IsValid())
    {
        // Commands may still be executing; the pool and semaphore cannot be
        // destroyed under them.
        (void)m_context->WaitIdle();
        const vk::DeviceApiExt& ext = m_context->DeviceFunctionsExt();
        if (m_pool != 0)
        {
            ext.destroyCommandPool(m_context->Device(), m_pool, nullptr);
        }
        if (m_timeline != 0)
        {
            ext.destroySemaphore(m_context->Device(), m_timeline, nullptr);
        }
    }
    m_context = nullptr;
    m_pool = 0;
    m_timeline = 0;
    m_nextValue = 1;
}

Result<VulkanCommandSystem, Error> VulkanCommandSystem::Create(VulkanContext& context) noexcept
{
    using Created = Result<VulkanCommandSystem, Error>;
    if (!context.IsValid())
    {
        return Created::Err(Error{ErrorCode::InvalidArgument, "context is not valid"});
    }
    VulkanCommandSystem system;
    system.m_context = &context;
    const vk::DeviceApiExt& ext = context.DeviceFunctionsExt();

    vk::CommandPoolCreateInfo poolInfo{};
    // Buffers are freed individually, so the pool allows per-buffer reset and frees.
    poolInfo.flags = vk::CommandPoolCreate::kResetCommandBuffer | vk::CommandPoolCreate::kTransient;
    poolInfo.queueFamilyIndex = context.Info().queues.graphics;
    if (ext.createCommandPool(context.Device(), &poolInfo, nullptr, &system.m_pool) !=
        vk::Result::Success)
    {
        system.m_pool = 0;
        return Created::Err(Error{ErrorCode::Internal, "vkCreateCommandPool failed"});
    }

    vk::SemaphoreTypeCreateInfo type{};
    vk::SemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.pNext = &type;
    if (ext.createSemaphore(context.Device(), &semaphoreInfo, nullptr, &system.m_timeline) !=
        vk::Result::Success)
    {
        system.m_timeline = 0;
        return Created::Err(Error{ErrorCode::Internal, "vkCreateSemaphore (timeline) failed"});
    }

    (void)context.SetObjectName(vk::ObjectType::Semaphore, system.m_timeline, "frame timeline");
    return Created::Ok(std::move(system));
}

Status<Error> VulkanCommandSystem::BeginRecording(vk::CommandBuffer* out) noexcept
{
    if (!IsValid() || out == nullptr)
    {
        return Status<Error>::Err(Error{ErrorCode::InvalidArgument, "command system is empty"});
    }
    const vk::DeviceApiExt& ext = m_context->DeviceFunctionsExt();
    vk::CommandBufferAllocateInfo allocate{};
    allocate.commandPool = m_pool;
    vk::CommandBuffer commands = nullptr;
    if (ext.allocateCommandBuffers(m_context->Device(), &allocate, &commands) !=
        vk::Result::Success)
    {
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkAllocateCommandBuffers failed"});
    }
    vk::CommandBufferBeginInfo begin{};
    begin.flags = vk::kCommandBufferOneTimeSubmit;
    if (ext.beginCommandBuffer(commands, &begin) != vk::Result::Success)
    {
        ext.freeCommandBuffers(m_context->Device(), m_pool, 1, &commands);
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkBeginCommandBuffer failed"});
    }
    *out = commands;
    return Status<Error>::Ok();
}

Status<Error> VulkanCommandSystem::Submit(vk::CommandBuffer commands, const VulkanSubmitSync& sync,
                                          u64* signalValue) noexcept
{
    if (!IsValid() || commands == nullptr)
    {
        return Status<Error>::Err(Error{ErrorCode::InvalidArgument, "command system is empty"});
    }
    const vk::DeviceApiExt& ext = m_context->DeviceFunctionsExt();
    if (ext.endCommandBuffer(commands) != vk::Result::Success)
    {
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkEndCommandBuffer failed"});
    }

    vk::CommandBufferSubmitInfo commandInfo{};
    commandInfo.commandBuffer = commands;

    vk::SemaphoreSubmitInfo waits[1] = {};
    u32 waitCount = 0;
    if (sync.waitBinary != 0)
    {
        waits[waitCount].semaphore = sync.waitBinary;
        waits[waitCount].stageMask = sync.waitStage;
        ++waitCount;
    }

    const u64 value = m_nextValue;
    vk::SemaphoreSubmitInfo signals[2] = {};
    u32 signalCount = 0;
    signals[signalCount].semaphore = m_timeline;
    signals[signalCount].value = value;
    signals[signalCount].stageMask = vk::Stage2::kAllCommands;
    ++signalCount;
    if (sync.signalBinary != 0)
    {
        signals[signalCount].semaphore = sync.signalBinary;
        signals[signalCount].stageMask = vk::Stage2::kAllCommands;
        ++signalCount;
    }

    vk::SubmitInfo2 submit{};
    submit.waitSemaphoreInfoCount = waitCount;
    submit.pWaitSemaphoreInfos = waits;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &commandInfo;
    submit.signalSemaphoreInfoCount = signalCount;
    submit.pSignalSemaphoreInfos = signals;

    const vk::Result result = ext.queueSubmit2(m_context->GraphicsQueue(), 1, &submit, 0);
    if (result == vk::Result::ErrorDeviceLost)
    {
        return Status<Error>::Err(Error{ErrorCode::DeviceLost, "Vulkan device lost on submit"});
    }
    if (result != vk::Result::Success)
    {
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkQueueSubmit2 failed"});
    }
    ++m_nextValue;
    if (signalValue != nullptr)
    {
        *signalValue = value;
    }
    return Status<Error>::Ok();
}

Status<Error> VulkanCommandSystem::Wait(u64 value, u64 timeoutNanoseconds) noexcept
{
    if (!IsValid())
    {
        return Status<Error>::Err(Error{ErrorCode::InvalidArgument, "command system is empty"});
    }
    vk::SemaphoreWaitInfo wait{};
    wait.semaphoreCount = 1;
    wait.pSemaphores = &m_timeline;
    wait.pValues = &value;
    const vk::Result result = m_context->DeviceFunctionsExt().waitSemaphores(
        m_context->Device(), &wait, timeoutNanoseconds);
    if (result == vk::Result::Timeout)
    {
        return Status<Error>::Err(Error{ErrorCode::Busy, "timeline wait timed out"});
    }
    if (result == vk::Result::ErrorDeviceLost)
    {
        return Status<Error>::Err(Error{ErrorCode::DeviceLost, "Vulkan device lost on wait"});
    }
    if (result != vk::Result::Success)
    {
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkWaitSemaphores failed"});
    }
    return Status<Error>::Ok();
}

u64 VulkanCommandSystem::CompletedValue() const noexcept
{
    u64 value = 0;
    if (IsValid())
    {
        (void)m_context->DeviceFunctionsExt().getSemaphoreCounterValue(m_context->Device(),
                                                                       m_timeline, &value);
    }
    return value;
}

void VulkanCommandSystem::Recycle(vk::CommandBuffer commands) noexcept
{
    if (IsValid() && commands != nullptr)
    {
        m_context->DeviceFunctionsExt().freeCommandBuffers(m_context->Device(), m_pool, 1,
                                                           &commands);
    }
}

Status<Error> VulkanCommandSystem::RunImmediate(RecordFunction record, void* user) noexcept
{
    vk::CommandBuffer commands = nullptr;
    Status<Error> begun = BeginRecording(&commands);
    if (!begun)
    {
        return begun;
    }
    record(commands, user);
    u64 value = 0;
    Status<Error> submitted = Submit(commands, VulkanSubmitSync{}, &value);
    if (!submitted)
    {
        Recycle(commands);
        return submitted;
    }
    Status<Error> waited = Wait(value);
    if (waited)
    {
        Recycle(commands);
    }
    return waited;
}

}  // namespace cyber::rhi::detail
