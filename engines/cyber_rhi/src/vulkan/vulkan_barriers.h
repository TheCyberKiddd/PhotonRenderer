#pragma once

// Translation of RHI resource states into Vulkan synchronization2 barriers.
// Phase 34 (ADR 0029, ADR 0014): the RHI tracks one `ResourceState` set per
// (mip, layer); Vulkan needs a pipeline-stage mask, an access mask and, for images,
// a layout for each side of a barrier. All three are derived here from the one
// state, so the renderer never names a Vulkan layout.

#include "vk_api_ext.h"

#include <cyber/rhi/resource_state.h>

namespace cyber::rhi::detail
{

// What one side of a barrier needs.
struct VulkanStateMapping
{
    vk::Flags64 stages = vk::Stage2::kNone;
    vk::Flags64 access = vk::Access2::kNone;
    vk::ImageLayout layout = vk::ImageLayout::Undefined;
};

// Shader stages a read or write may happen in. The RHI does not say which stage
// uses a resource, so the mapping is conservative: every stage the backend
// can run. A later refinement can narrow it using the pass's declared usage.
inline constexpr vk::Flags64 kVulkanAllShaderStages =
    vk::Stage2::kVertexShader | vk::Stage2::kFragmentShader | vk::Stage2::kComputeShader;

// `VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL`.
inline constexpr vk::ImageLayout kVulkanDepthStencilReadOnly = static_cast<vk::ImageLayout>(4);

// Maps a state set to stages, access and layout. `isTexture` selects whether the
// layout is meaningful; a buffer always reports `Undefined`.
//
// A single image layout has to serve every bit of the set, so:
//   * `Present`, `RenderTarget`, `DepthWrite`, `CopyDestination` are single-use
//     write states and map to their own layout;
//   * a lone read state maps to its optimal layout;
//   * `DepthRead` combined with `ShaderRead` keeps the depth read-only layout,
//     which Vulkan allows to be sampled;
//   * any other combination, and every `ShaderWrite`, maps to `General`, which is
//     valid for all of them at some cost on hardware that has per-layout
//     compression.
[[nodiscard]] constexpr VulkanStateMapping MapResourceState(ResourceState state,
                                                            bool isTexture) noexcept
{
    VulkanStateMapping mapping{};
    const u32 bits = ResourceStateBits(state);
    if (bits == 0)
    {
        return mapping;  // Undefined: nothing to wait for, nothing to make visible.
    }

    const auto has = [bits](ResourceState flag) constexpr noexcept
    { return (bits & ResourceStateBits(flag)) != 0; };

    if (has(ResourceState::CopySource))
    {
        mapping.stages |= vk::Stage2::kAllTransfer;
        mapping.access |= vk::Access2::kTransferRead;
    }
    if (has(ResourceState::CopyDestination))
    {
        mapping.stages |= vk::Stage2::kAllTransfer;
        mapping.access |= vk::Access2::kTransferWrite;
    }
    if (has(ResourceState::VertexBuffer))
    {
        mapping.stages |= vk::Stage2::kVertexInput;
        mapping.access |= vk::Access2::kVertexAttributeRead;
    }
    if (has(ResourceState::IndexBuffer))
    {
        mapping.stages |= vk::Stage2::kVertexInput;
        mapping.access |= vk::Access2::kIndexRead;
    }
    if (has(ResourceState::ConstantBuffer))
    {
        mapping.stages |= kVulkanAllShaderStages;
        mapping.access |= vk::Access2::kUniformRead;
    }
    if (has(ResourceState::ShaderRead))
    {
        mapping.stages |= kVulkanAllShaderStages;
        mapping.access |= vk::Access2::kShaderRead;
    }
    if (has(ResourceState::ShaderWrite))
    {
        mapping.stages |= kVulkanAllShaderStages;
        mapping.access |= vk::Access2::kShaderWrite | vk::Access2::kShaderRead;
    }
    if (has(ResourceState::RenderTarget))
    {
        mapping.stages |= vk::Stage2::kColorAttachmentOutput;
        mapping.access |= vk::Access2::kColorAttachmentRead | vk::Access2::kColorAttachmentWrite;
    }
    if (has(ResourceState::DepthRead))
    {
        mapping.stages |= vk::Stage2::kEarlyFragmentTests | vk::Stage2::kLateFragmentTests;
        mapping.access |= vk::Access2::kDepthStencilRead;
    }
    if (has(ResourceState::DepthWrite))
    {
        mapping.stages |= vk::Stage2::kEarlyFragmentTests | vk::Stage2::kLateFragmentTests;
        mapping.access |= vk::Access2::kDepthStencilRead | vk::Access2::kDepthStencilWrite;
    }
    if (has(ResourceState::IndirectArgument))
    {
        mapping.stages |= vk::Stage2::kDrawIndirect;
        mapping.access |= vk::Access2::kIndirectCommandRead;
    }
    // Present: the presentation engine reads after a semaphore, so the barrier
    // carries only the layout. Stage and access stay empty (NONE), which is what
    // the specification prescribes for the layout transition before present.

    if (!isTexture)
    {
        return mapping;
    }

    if (bits == ResourceStateBits(ResourceState::Present))
    {
        mapping.layout = vk::ImageLayout::PresentSrcKHR;
    }
    else if (bits == ResourceStateBits(ResourceState::CopySource))
    {
        mapping.layout = vk::ImageLayout::TransferSrcOptimal;
    }
    else if (bits == ResourceStateBits(ResourceState::CopyDestination))
    {
        mapping.layout = vk::ImageLayout::TransferDstOptimal;
    }
    else if (bits == ResourceStateBits(ResourceState::RenderTarget))
    {
        mapping.layout = vk::ImageLayout::ColorAttachmentOptimal;
    }
    else if (bits == ResourceStateBits(ResourceState::DepthWrite))
    {
        mapping.layout = vk::ImageLayout::DepthStencilAttachmentOptimal;
    }
    else if (bits == ResourceStateBits(ResourceState::ShaderRead))
    {
        mapping.layout = vk::ImageLayout::ShaderReadOnlyOptimal;
    }
    else if (bits == ResourceStateBits(ResourceState::DepthRead) ||
             bits == (ResourceStateBits(ResourceState::DepthRead) |
                      ResourceStateBits(ResourceState::ShaderRead)))
    {
        mapping.layout = kVulkanDepthStencilReadOnly;
    }
    else
    {
        mapping.layout = vk::ImageLayout::General;
    }
    return mapping;
}

// Builds the barrier that moves an image from `before` to `after`. A transition
// out of `Undefined` discards the contents, which is what `Undefined` means in
// the RHI and what the Vulkan old layout of `Undefined` does.
[[nodiscard]] constexpr vk::ImageMemoryBarrier2
MakeImageBarrier(vk::Image image, const vk::ImageSubresourceRange& range, ResourceState before,
                 ResourceState after) noexcept
{
    const VulkanStateMapping from = MapResourceState(before, true);
    const VulkanStateMapping to = MapResourceState(after, true);
    vk::ImageMemoryBarrier2 barrier{};
    barrier.srcStageMask = from.stages != vk::Stage2::kNone ? from.stages : vk::Stage2::kTopOfPipe;
    barrier.srcAccessMask = from.access;
    barrier.dstStageMask = to.stages != vk::Stage2::kNone ? to.stages : vk::Stage2::kBottomOfPipe;
    barrier.dstAccessMask = to.access;
    barrier.oldLayout = from.layout;
    barrier.newLayout = to.layout;
    barrier.image = image;
    barrier.subresourceRange = range;
    return barrier;
}

[[nodiscard]] constexpr vk::BufferMemoryBarrier2
MakeBufferBarrier(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
                  ResourceState before, ResourceState after) noexcept
{
    const VulkanStateMapping from = MapResourceState(before, false);
    const VulkanStateMapping to = MapResourceState(after, false);
    vk::BufferMemoryBarrier2 barrier{};
    barrier.srcStageMask = from.stages != vk::Stage2::kNone ? from.stages : vk::Stage2::kTopOfPipe;
    barrier.srcAccessMask = from.access;
    barrier.dstStageMask = to.stages != vk::Stage2::kNone ? to.stages : vk::Stage2::kBottomOfPipe;
    barrier.dstAccessMask = to.access;
    barrier.buffer = buffer;
    barrier.offset = offset;
    barrier.size = size;
    return barrier;
}

}  // namespace cyber::rhi::detail
