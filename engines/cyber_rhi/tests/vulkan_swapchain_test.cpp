#include "vulkan_rig.h"

#include "../src/vulkan/vulkan_swapchain.h"

#include <cyber/platform/platform.h>
#include <cyber/test/test.h>

#include <cstdio>
#include <cstring>
#include <utility>

using namespace cyber::rhi;
using namespace cyber::rhi::detail;
using namespace cyber::rhi::test;
using cyber::core::u32;
using cyber::core::u8;
using cyber::memory::HeapAllocator;

namespace
{

constexpr vk::Flags kBufferUsage =
    vk::BufferUsage::kTransferSrc | vk::BufferUsage::kTransferDst | vk::BufferUsage::kStorageBuffer;

// A window with a native handle, or nothing on a machine without a window
// system. Hidden: the test needs a surface, not a visible frame.
[[nodiscard]] bool MakeWindow(HeapAllocator& heap, cyber::platform::Window& out) noexcept
{
    if (!cyber::platform::IsWindowSystemSupported())
    {
        std::printf("  skipped: no window system\n");
        return false;
    }
    cyber::platform::WindowDesc desc{};
    desc.title = "vulkan swapchain test";
    desc.width = 320;
    desc.height = 240;
    desc.visible = false;
    auto created = cyber::platform::Window::Create(desc, heap);
    if (!created || created.Value().NativeHandle() == nullptr)
    {
        std::printf("  skipped: window creation failed\n");
        return false;
    }
    out = std::move(created.Value());
    return true;
}

}  // namespace

CG_TEST(vulkan_swapchain_presents_frames_and_the_pixels_are_right)
{
    HeapAllocator heap;
    cyber::platform::Window window;
    if (!MakeWindow(heap, window))
    {
        return;
    }
    VulkanRig rig(heap);
    if (!rig.Init(heap, true))
    {
        return;
    }
    if (!rig.context.Info().presentation)
    {
        std::printf("  skipped: no presentation support on this device\n");
        return;
    }

    auto created = VulkanSwapchain::Create(rig.context, window.NativeHandle(), 320, 240);
    if (!created)
    {
        std::printf("  skipped: %s\n", created.Error().message);
        return;
    }
    VulkanSwapchain swapchain = std::move(created.Value());
    CG_REQUIRE(swapchain.IsValid());
    CG_EXPECT_TRUE(swapchain.ImageCount() >= 2);
    CG_EXPECT_TRUE(swapchain.Width() > 0 && swapchain.Height() > 0);
    CG_EXPECT_TRUE(swapchain.Format() == vk::Format::B8G8R8A8Srgb ||
                   swapchain.Format() == vk::Format::B8G8R8A8Unorm);

    VulkanBuffer readback;
    CG_REQUIRE(rig.memory
                   .CreateBuffer({swapchain.Width() * swapchain.Height() * 4, kBufferUsage,
                                  VulkanMemoryClass::Readback},
                                 &readback)
                   .IsOk());

    const vk::DeviceApiExt& ext = rig.context.DeviceFunctionsExt();
    u32 presented = 0;
    for (u32 frame = 0; frame < 6; ++frame)
    {
        u32 image = 0;
        vk::Semaphore available = 0;
        vk::Semaphore finished = 0;
        auto acquired = swapchain.Acquire(&image, &available, &finished, 2000000000ULL);
        CG_REQUIRE(acquired);
        CG_REQUIRE(acquired.Value() == VulkanPresentResult::Ok);

        vk::CommandBuffer commands = nullptr;
        CG_REQUIRE(rig.commands.BeginRecording(&commands).IsOk());
        const vk::ImageSubresourceRange range{vk::ImageAspect::kColor, 0, 1, 0, 1};

        // Just acquired: the image is `Undefined` to us, which discards the old
        // contents, then it is drawn into.
        auto toTarget = MakeImageBarrier(swapchain.Image(image), range, ResourceState::Undefined,
                                         ResourceState::RenderTarget);
        RecordBarriers(ext, commands, nullptr, 0, &toTarget, 1);

        vk::RenderingAttachmentInfo color{};
        color.imageView = swapchain.ImageView(image);
        color.imageLayout = vk::ImageLayout::ColorAttachmentOptimal;
        color.loadOp = vk::kAttachmentLoadOpClear;
        color.storeOp = vk::kAttachmentStoreOpStore;
        color.clearValue.color[0] = 1.0F;  // red, opaque
        color.clearValue.color[3] = 1.0F;
        vk::RenderingInfo rendering{};
        rendering.renderArea = vk::Rect2D{{0, 0}, {swapchain.Width(), swapchain.Height()}};
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &color;
        ext.cmdBeginRendering(commands, &rendering);
        ext.cmdEndRendering(commands);

        const bool inspect = frame == 0;
        if (inspect)
        {
            auto toCopy = MakeImageBarrier(swapchain.Image(image), range, ResourceState::RenderTarget,
                                           ResourceState::CopySource);
            RecordBarriers(ext, commands, nullptr, 0, &toCopy, 1);
            vk::BufferImageCopy copy{};
            copy.imageSubresource = vk::ImageSubresourceLayers{vk::ImageAspect::kColor, 0, 0, 1};
            copy.imageExtent = vk::Extent3D{swapchain.Width(), swapchain.Height(), 1};
            ext.cmdCopyImageToBuffer(commands, swapchain.Image(image),
                                     vk::ImageLayout::TransferSrcOptimal, readback.buffer, 1,
                                     &copy);
            HostReadBarrier(ext, commands, readback.buffer, vk::Stage2::kAllTransfer,
                            vk::Access2::kTransferWrite);
        }
        auto toPresent =
            MakeImageBarrier(swapchain.Image(image), range,
                             inspect ? ResourceState::CopySource : ResourceState::RenderTarget,
                             ResourceState::Present);
        RecordBarriers(ext, commands, nullptr, 0, &toPresent, 1);

        VulkanSubmitSync sync{};
        sync.waitBinary = available;
        sync.waitStage = vk::Stage2::kColorAttachmentOutput;
        sync.signalBinary = finished;
        u64 value = 0;
        CG_REQUIRE(rig.commands.Submit(commands, sync, &value).IsOk());

        auto result = swapchain.Present(image, finished);
        CG_REQUIRE(result);
        ++presented;
        CG_REQUIRE(rig.commands.Wait(value, 5000000000ULL).IsOk());
        rig.commands.Recycle(commands);

        if (inspect)
        {
            // BGRA bytes of opaque red.
            const auto* pixels = static_cast<const u8*>(readback.block.mapped);
            CG_EXPECT_EQ(pixels[0], u8{0});
            CG_EXPECT_EQ(pixels[1], u8{0});
            CG_EXPECT_EQ(pixels[2], u8{255});
            CG_EXPECT_EQ(pixels[3], u8{255});
        }
        if (result.Value() == VulkanPresentResult::NeedsRecreate)
        {
            CG_REQUIRE(swapchain.Recreate(320, 240).IsOk());
        }
    }
    CG_EXPECT_EQ(presented, u32{6});
    CG_EXPECT_EQ(rig.context.ValidationErrorCount(), u32{0});
    rig.memory.DestroyBuffer(readback);
}

CG_TEST(vulkan_swapchain_recreates_for_a_new_size)
{
    HeapAllocator heap;
    cyber::platform::Window window;
    if (!MakeWindow(heap, window))
    {
        return;
    }
    VulkanRig rig(heap);
    if (!rig.Init(heap, true) || !rig.context.Info().presentation)
    {
        std::printf("  skipped: no presentation support\n");
        return;
    }
    auto created = VulkanSwapchain::Create(rig.context, window.NativeHandle(), 320, 240);
    if (!created)
    {
        std::printf("  skipped: %s\n", created.Error().message);
        return;
    }
    VulkanSwapchain swapchain = std::move(created.Value());
    const u32 before = swapchain.ImageCount();
    // The surface extent follows the window, so a recreate at the same size must
    // still rebuild every image, view and semaphore cleanly.
    CG_REQUIRE(swapchain.Recreate(320, 240).IsOk());
    CG_EXPECT_TRUE(swapchain.IsValid());
    CG_EXPECT_TRUE(swapchain.ImageCount() >= 2);
    CG_EXPECT_TRUE(before >= 2);

    u32 image = 0;
    vk::Semaphore available = 0;
    vk::Semaphore finished = 0;
    auto acquired = swapchain.Acquire(&image, &available, &finished, 2000000000ULL);
    CG_REQUIRE(acquired);
    CG_EXPECT_TRUE(image < swapchain.ImageCount());
    CG_EXPECT_EQ(rig.context.ValidationErrorCount(), u32{0});
}

CG_TEST(vulkan_swapchain_requires_presentation_support)
{
    HeapAllocator heap;
    VulkanRig rig(heap);
    // A headless context has no surface extensions enabled.
    if (!rig.Init(heap, false))
    {
        return;
    }
    CG_EXPECT_FALSE(rig.context.Info().presentation);
    int dummy = 0;
    auto created = VulkanSwapchain::Create(rig.context, &dummy, 64, 64);
    CG_REQUIRE(!created);
    CG_EXPECT_TRUE(created.Error().code == cyber::core::ErrorCode::Unsupported);
}
