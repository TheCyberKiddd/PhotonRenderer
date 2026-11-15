#include "../src/vulkan/vulkan_barriers.h"
#include "../src/vulkan/vulkan_commands.h"
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

// A context, its allocator and its command system, torn down in reverse order.
// Skips (not fails) the test on a machine with no usable Vulkan 1.3 device.
struct Rig
{
    explicit Rig(HeapAllocator& heap) : memory(heap) {}

    [[nodiscard]] bool Init(HeapAllocator& heap) noexcept
    {
        VulkanContextDesc desc{};
        desc.requireValidation = true;
        auto created = VulkanContext::Create(desc, heap);
        if (!created && created.Error().code == ErrorCode::NotFound)
        {
            desc.requireValidation = false;
            created = VulkanContext::Create(desc, heap);
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
        auto system = VulkanCommandSystem::Create(context);
        if (!system)
        {
            std::printf("  command system failed: %s\n", system.Error().message);
            return false;
        }
        commands = std::move(system.Value());
        return true;
    }

    // Declaration order is destruction order reversed: commands, then memory, then
    // the context they both use.
    VulkanContext context;
    VulkanMemoryAllocator memory;
    VulkanCommandSystem commands;
};

constexpr vk::Flags kBufferUsage =
    vk::BufferUsage::kTransferSrc | vk::BufferUsage::kTransferDst | vk::BufferUsage::kStorageBuffer;

void HostReadBarrier(const vk::DeviceApiExt& ext, vk::CommandBuffer commands,
                     vk::Buffer buffer) noexcept
{
    vk::BufferMemoryBarrier2 barrier{};
    barrier.srcStageMask = vk::Stage2::kAllTransfer;
    barrier.srcAccessMask = vk::Access2::kTransferWrite;
    barrier.dstStageMask = vk::Stage2::kHost;
    barrier.dstAccessMask = vk::Access2::kHostRead;
    barrier.buffer = buffer;
    RecordBarriers(ext, commands, &barrier, 1, nullptr, 0);
}

}  // namespace

CG_TEST(vulkan_state_mapping_derives_stage_access_and_layout)
{
    // Pure logic, no device: runs everywhere.
    const auto undefined = MapResourceState(ResourceState::Undefined, true);
    CG_EXPECT_EQ(undefined.stages, vk::Stage2::kNone);
    CG_EXPECT_EQ(undefined.access, vk::Access2::kNone);
    CG_EXPECT_TRUE(undefined.layout == vk::ImageLayout::Undefined);

    const auto target = MapResourceState(ResourceState::RenderTarget, true);
    CG_EXPECT_TRUE(target.layout == vk::ImageLayout::ColorAttachmentOptimal);
    CG_EXPECT_EQ(target.stages, vk::Stage2::kColorAttachmentOutput);
    CG_EXPECT_TRUE((target.access & vk::Access2::kColorAttachmentWrite) != 0);

    CG_EXPECT_TRUE(MapResourceState(ResourceState::ShaderRead, true).layout ==
                   vk::ImageLayout::ShaderReadOnlyOptimal);
    CG_EXPECT_TRUE(MapResourceState(ResourceState::ShaderWrite, true).layout ==
                   vk::ImageLayout::General);
    CG_EXPECT_TRUE(MapResourceState(ResourceState::CopySource, true).layout ==
                   vk::ImageLayout::TransferSrcOptimal);
    CG_EXPECT_TRUE(MapResourceState(ResourceState::CopyDestination, true).layout ==
                   vk::ImageLayout::TransferDstOptimal);
    CG_EXPECT_TRUE(MapResourceState(ResourceState::DepthWrite, true).layout ==
                   vk::ImageLayout::DepthStencilAttachmentOptimal);

    // Present carries a layout and no stage or access.
    const auto present = MapResourceState(ResourceState::Present, true);
    CG_EXPECT_TRUE(present.layout == vk::ImageLayout::PresentSrcKHR);
    CG_EXPECT_EQ(present.stages, vk::Stage2::kNone);

    // One layout has to serve a set of reads: depth read plus shader read keeps the
    // depth read-only layout, an unrelated pair falls back to General.
    CG_EXPECT_TRUE(MapResourceState(ResourceState::DepthRead | ResourceState::ShaderRead, true)
                       .layout == kVulkanDepthStencilReadOnly);
    CG_EXPECT_TRUE(MapResourceState(ResourceState::CopySource | ResourceState::ShaderRead, true)
                       .layout == vk::ImageLayout::General);

    // Buffers never report a layout.
    const auto vertex = MapResourceState(ResourceState::VertexBuffer, false);
    CG_EXPECT_TRUE(vertex.layout == vk::ImageLayout::Undefined);
    CG_EXPECT_EQ(vertex.stages, vk::Stage2::kVertexInput);
    CG_EXPECT_EQ(vertex.access, vk::Access2::kVertexAttributeRead);
    CG_EXPECT_TRUE((MapResourceState(ResourceState::ConstantBuffer, false).access &
                    vk::Access2::kUniformRead) != 0);
    CG_EXPECT_EQ(MapResourceState(ResourceState::IndirectArgument, false).stages,
                 vk::Stage2::kDrawIndirect);

    // A barrier out of Undefined waits on nothing and discards the contents.
    const auto barrier = MakeImageBarrier(7, vk::ImageSubresourceRange{vk::ImageAspect::kColor},
                                          ResourceState::Undefined, ResourceState::CopyDestination);
    CG_EXPECT_EQ(barrier.srcStageMask, vk::Stage2::kTopOfPipe);
    CG_EXPECT_EQ(barrier.srcAccessMask, vk::Access2::kNone);
    CG_EXPECT_TRUE(barrier.oldLayout == vk::ImageLayout::Undefined);
    CG_EXPECT_TRUE(barrier.newLayout == vk::ImageLayout::TransferDstOptimal);
    CG_EXPECT_EQ(barrier.image, u64{7});

    // A write must be made available before a read: every writing state's barrier
    // has a write access on its source side.
    const auto toRead = MakeBufferBarrier(3, 0, vk::kWholeSize, ResourceState::CopyDestination,
                                          ResourceState::CopySource);
    CG_EXPECT_TRUE((toRead.srcAccessMask & vk::Access2::kTransferWrite) != 0);
    CG_EXPECT_TRUE((toRead.dstAccessMask & vk::Access2::kTransferRead) != 0);
}

CG_TEST(vulkan_timeline_advances_with_each_submit)
{
    HeapAllocator heap;
    Rig rig(heap);
    if (!rig.Init(heap))
    {
        return;
    }
    VulkanCommandSystem& commands = rig.commands;
    CG_EXPECT_EQ(commands.CompletedValue(), u64{0});
    CG_EXPECT_EQ(commands.NextValue(), u64{1});

    u64 last = 0;
    for (u32 round = 0; round < 8; ++round)
    {
        vk::CommandBuffer buffer = nullptr;
        CG_REQUIRE(commands.BeginRecording(&buffer).IsOk());
        u64 value = 0;
        CG_REQUIRE(commands.Submit(buffer, VulkanSubmitSync{}, &value).IsOk());
        CG_EXPECT_EQ(value, last + 1);
        last = value;
        CG_REQUIRE(commands.Wait(value).IsOk());
        CG_EXPECT_TRUE(commands.CompletedValue() >= value);
        commands.Recycle(buffer);
    }
    // Waiting for a value nobody will signal times out instead of hanging.
    const auto timedOut = commands.Wait(last + 100, 1000000ULL);
    CG_REQUIRE(!timedOut);
    CG_EXPECT_TRUE(timedOut.Error().code == ErrorCode::Busy);
    CG_EXPECT_EQ(rig.context.ValidationErrorCount(), u32{0});
}

CG_TEST(vulkan_buffer_copy_roundtrips_through_device_local_memory)
{
    HeapAllocator heap;
    Rig rig(heap);
    if (!rig.Init(heap))
    {
        return;
    }
    constexpr u32 kBytes = 64 * 1024;
    VulkanBuffer upload;
    VulkanBuffer device;
    VulkanBuffer readback;
    CG_REQUIRE(rig.memory.CreateBuffer({kBytes, kBufferUsage, VulkanMemoryClass::Upload}, &upload)
                   .IsOk());
    CG_REQUIRE(
        rig.memory.CreateBuffer({kBytes, kBufferUsage, VulkanMemoryClass::DeviceLocal}, &device)
            .IsOk());
    CG_REQUIRE(
        rig.memory.CreateBuffer({kBytes, kBufferUsage, VulkanMemoryClass::Readback}, &readback)
            .IsOk());

    auto* source = static_cast<u8*>(upload.block.mapped);
    for (u32 index = 0; index < kBytes; ++index)
    {
        source[index] = static_cast<u8>((index * 2654435761U) >> 24);
    }
    std::memset(readback.block.mapped, 0, kBytes);

    struct State
    {
        const vk::DeviceApiExt* ext;
        vk::Buffer upload;
        vk::Buffer device;
        vk::Buffer readback;
    } state{&rig.context.DeviceFunctionsExt(), upload.buffer, device.buffer, readback.buffer};

    CG_REQUIRE(rig.commands
                   .RunImmediate(
                       [](vk::CommandBuffer commands, void* user)
                       {
                           auto& s = *static_cast<State*>(user);
                           const vk::BufferCopy region{0, 0, kBytes};
                           s.ext->cmdCopyBuffer(commands, s.upload, s.device, 1, &region);
                           const auto toRead = MakeBufferBarrier(
                               s.device, 0, vk::kWholeSize, ResourceState::CopyDestination,
                               ResourceState::CopySource);
                           RecordBarriers(*s.ext, commands, &toRead, 1, nullptr, 0);
                           s.ext->cmdCopyBuffer(commands, s.device, s.readback, 1, &region);
                           HostReadBarrier(*s.ext, commands, s.readback);
                       },
                       &state)
                   .IsOk());

    CG_EXPECT_TRUE(std::memcmp(source, readback.block.mapped, kBytes) == 0);
    CG_EXPECT_EQ(rig.context.ValidationErrorCount(), u32{0});

    rig.memory.DestroyBuffer(upload);
    rig.memory.DestroyBuffer(device);
    rig.memory.DestroyBuffer(readback);
}

CG_TEST(vulkan_fill_buffer_writes_the_pattern)
{
    HeapAllocator heap;
    Rig rig(heap);
    if (!rig.Init(heap))
    {
        return;
    }
    VulkanBuffer readback;
    CG_REQUIRE(
        rig.memory.CreateBuffer({4096, kBufferUsage, VulkanMemoryClass::Readback}, &readback)
            .IsOk());
    std::memset(readback.block.mapped, 0, 4096);

    struct State
    {
        const vk::DeviceApiExt* ext;
        vk::Buffer buffer;
    } state{&rig.context.DeviceFunctionsExt(), readback.buffer};
    CG_REQUIRE(rig.commands
                   .RunImmediate(
                       [](vk::CommandBuffer commands, void* user)
                       {
                           auto& s = *static_cast<State*>(user);
                           s.ext->cmdFillBuffer(commands, s.buffer, 0, 4096, 0xA5A5A5A5U);
                           HostReadBarrier(*s.ext, commands, s.buffer);
                       },
                       &state)
                   .IsOk());

    const auto* words = static_cast<const u32*>(readback.block.mapped);
    bool all = true;
    for (u32 index = 0; index < 1024; ++index)
    {
        all = all && words[index] == 0xA5A5A5A5U;
    }
    CG_EXPECT_TRUE(all);
    rig.memory.DestroyBuffer(readback);
    CG_EXPECT_EQ(rig.context.ValidationErrorCount(), u32{0});
}

CG_TEST(vulkan_image_copy_roundtrips_with_state_barriers)
{
    HeapAllocator heap;
    Rig rig(heap);
    if (!rig.Init(heap))
    {
        return;
    }
    constexpr u32 kSize = 64;
    constexpr u32 kBytes = kSize * kSize * 4;

    VulkanImage image;
    VulkanImageDesc desc{};
    desc.width = kSize;
    desc.height = kSize;
    desc.format = vk::Format::R8G8B8A8Unorm;
    desc.usage = vk::ImageUsage::kTransferSrc | vk::ImageUsage::kTransferDst |
                 vk::ImageUsage::kSampled;
    CG_REQUIRE(rig.memory.CreateImage(desc, &image).IsOk());
    CG_EXPECT_TRUE(image.IsValid());

    VulkanBuffer upload;
    VulkanBuffer readback;
    CG_REQUIRE(rig.memory.CreateBuffer({kBytes, kBufferUsage, VulkanMemoryClass::Upload}, &upload)
                   .IsOk());
    CG_REQUIRE(
        rig.memory.CreateBuffer({kBytes, kBufferUsage, VulkanMemoryClass::Readback}, &readback)
            .IsOk());
    auto* pixels = static_cast<u8*>(upload.block.mapped);
    for (u32 y = 0; y < kSize; ++y)
    {
        for (u32 x = 0; x < kSize; ++x)
        {
            u8* texel = pixels + (y * kSize + x) * 4;
            texel[0] = static_cast<u8>(x * 4);
            texel[1] = static_cast<u8>(y * 4);
            texel[2] = static_cast<u8>((x ^ y) * 4);
            texel[3] = 255;
        }
    }
    std::memset(readback.block.mapped, 0, kBytes);

    struct State
    {
        const vk::DeviceApiExt* ext;
        vk::Image image;
        vk::Buffer upload;
        vk::Buffer readback;
    } state{&rig.context.DeviceFunctionsExt(), image.image, upload.buffer, readback.buffer};

    CG_REQUIRE(
        rig.commands
            .RunImmediate(
                [](vk::CommandBuffer commands, void* user)
                {
                    auto& s = *static_cast<State*>(user);
                    const vk::ImageSubresourceRange range{vk::ImageAspect::kColor, 0, 1, 0, 1};
                    vk::BufferImageCopy copy{};
                    copy.imageSubresource =
                        vk::ImageSubresourceLayers{vk::ImageAspect::kColor, 0, 0, 1};
                    copy.imageExtent = vk::Extent3D{kSize, kSize, 1};

                    auto toWrite = MakeImageBarrier(s.image, range, ResourceState::Undefined,
                                                    ResourceState::CopyDestination);
                    RecordBarriers(*s.ext, commands, nullptr, 0, &toWrite, 1);
                    s.ext->cmdCopyBufferToImage(commands, s.upload, s.image,
                                                vk::ImageLayout::TransferDstOptimal, 1, &copy);
                    auto toRead = MakeImageBarrier(s.image, range, ResourceState::CopyDestination,
                                                   ResourceState::CopySource);
                    RecordBarriers(*s.ext, commands, nullptr, 0, &toRead, 1);
                    s.ext->cmdCopyImageToBuffer(commands, s.image,
                                                vk::ImageLayout::TransferSrcOptimal, s.readback, 1,
                                                &copy);
                    HostReadBarrier(*s.ext, commands, s.readback);
                },
                &state)
            .IsOk());

    CG_EXPECT_TRUE(std::memcmp(pixels, readback.block.mapped, kBytes) == 0);

    // A view and a sampler over the image are plain driver objects; creating and
    // destroying them must be clean, and a view outside the image is refused.
    vk::ImageView view = 0;
    CG_REQUIRE(rig.memory
                   .CreateImageView(image, vk::ImageViewType::Type2D, vk::ImageAspect::kColor, 0, 1,
                                    0, 1, &view)
                   .IsOk());
    CG_EXPECT_TRUE(view != 0);
    vk::ImageView bad = 0;
    CG_EXPECT_FALSE(rig.memory
                        .CreateImageView(image, vk::ImageViewType::Type2D, vk::ImageAspect::kColor,
                                         0, 2, 0, 1, &bad)
                        .IsOk());
    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::Linear;
    samplerInfo.minFilter = vk::Filter::Linear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::ClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::ClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::ClampToEdge;
    vk::Sampler sampler = 0;
    CG_REQUIRE(rig.memory.CreateSampler(samplerInfo, &sampler).IsOk());
    CG_EXPECT_TRUE(sampler != 0);

    rig.memory.DestroySampler(sampler);
    rig.memory.DestroyImageView(view);
    rig.memory.DestroyImage(image);
    rig.memory.DestroyBuffer(upload);
    rig.memory.DestroyBuffer(readback);
    CG_EXPECT_EQ(rig.context.ValidationErrorCount(), u32{0});
}

CG_TEST(vulkan_mipmapped_image_views_respect_their_range)
{
    HeapAllocator heap;
    Rig rig(heap);
    if (!rig.Init(heap))
    {
        return;
    }
    VulkanImage image;
    VulkanImageDesc desc{};
    desc.width = 256;
    desc.height = 256;
    desc.mipLevels = 9;
    desc.arrayLayers = 4;
    desc.format = vk::Format::R8G8B8A8Unorm;
    desc.usage = vk::ImageUsage::kSampled | vk::ImageUsage::kTransferDst;
    CG_REQUIRE(rig.memory.CreateImage(desc, &image).IsOk());

    vk::ImageView whole = 0;
    vk::ImageView slice = 0;
    vk::ImageView outside = 0;
    CG_EXPECT_TRUE(rig.memory
                       .CreateImageView(image, vk::ImageViewType::Type2DArray,
                                        vk::ImageAspect::kColor, 0, 9, 0, 4, &whole)
                       .IsOk());
    CG_EXPECT_TRUE(rig.memory
                       .CreateImageView(image, vk::ImageViewType::Type2D, vk::ImageAspect::kColor,
                                        3, 2, 1, 1, &slice)
                       .IsOk());
    CG_EXPECT_FALSE(rig.memory
                        .CreateImageView(image, vk::ImageViewType::Type2D,
                                         vk::ImageAspect::kColor, 8, 2, 0, 1, &outside)
                        .IsOk());
    CG_EXPECT_FALSE(rig.memory
                        .CreateImageView(image, vk::ImageViewType::Type2D,
                                         vk::ImageAspect::kColor, 0, 1, 3, 2, &outside)
                        .IsOk());

    rig.memory.DestroyImageView(slice);
    rig.memory.DestroyImageView(whole);
    rig.memory.DestroyImage(image);
    CG_EXPECT_EQ(rig.memory.Statistics().deviceLocal.usedBytes, cyber::core::u64{0});
    CG_EXPECT_EQ(rig.context.ValidationErrorCount(), u32{0});
}
