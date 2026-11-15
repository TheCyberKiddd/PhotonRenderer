#pragma once

// Presentation for the Vulkan backend. Phase 34 increment 5 (ADR 0029): a Win32
// surface and a FIFO swapchain, with the acquire and render-finished semaphore
// rings sized to the image count, recreation on out-of-date or resize, and the
// image/view handles the renderer targets. Linux and Android surfaces arrive with
// their window backends; `Create` reports `Unsupported` there.

#include "vulkan_context.h"

#include "vk_api_ext.h"

namespace cyber::rhi::detail
{

inline constexpr u32 kMaxSwapchainImages = 8;

enum class VulkanPresentResult : u8
{
    Ok,
    // The swapchain no longer matches the surface (resize, minimise): call
    // `Recreate` before the next frame. Not an error.
    NeedsRecreate,
};

class VulkanSwapchain
{
public:
    VulkanSwapchain() noexcept = default;
    VulkanSwapchain(const VulkanSwapchain&) = delete;
    VulkanSwapchain& operator=(const VulkanSwapchain&) = delete;
    VulkanSwapchain(VulkanSwapchain&& other) noexcept;
    VulkanSwapchain& operator=(VulkanSwapchain&& other) noexcept;
    ~VulkanSwapchain();

    // Creates a swapchain for the window with native handle `windowHandle` (an
    // `HWND` on Windows). The context must have been created with presentation
    // enabled and must outlive the swapchain. `ErrorCode::Unsupported` when the
    // context has no presentation support or the graphics queue cannot present to
    // this surface.
    [[nodiscard]] static Result<VulkanSwapchain, Error> Create(VulkanContext& context,
                                                               void* windowHandle, u32 width,
                                                               u32 height) noexcept;

    [[nodiscard]] bool IsValid() const noexcept
    {
        return m_swapchain != 0;
    }
    [[nodiscard]] vk::Format Format() const noexcept
    {
        return m_format;
    }
    [[nodiscard]] u32 Width() const noexcept
    {
        return m_extent.width;
    }
    [[nodiscard]] u32 Height() const noexcept
    {
        return m_extent.height;
    }
    [[nodiscard]] u32 ImageCount() const noexcept
    {
        return m_imageCount;
    }
    [[nodiscard]] vk::Image Image(u32 index) const noexcept
    {
        return m_images[index];
    }
    [[nodiscard]] vk::ImageView ImageView(u32 index) const noexcept
    {
        return m_views[index];
    }

    // Acquires the next image. `imageAvailable` is the binary semaphore the
    // submit that draws into the image must wait on; `renderFinished` is the one it
    // must signal, and `Present` waits on it. Both are owned by the swapchain.
    // Returns `NeedsRecreate` instead of an image when the surface changed.
    [[nodiscard]] Result<VulkanPresentResult, Error> Acquire(u32* imageIndex,
                                                             vk::Semaphore* imageAvailable,
                                                             vk::Semaphore* renderFinished,
                                                             u64 timeoutNanoseconds = ~0ULL) noexcept;

    [[nodiscard]] Result<VulkanPresentResult, Error> Present(u32 imageIndex,
                                                             vk::Semaphore renderFinished) noexcept;

    // Rebuilds the swapchain at a new size, waiting for the device to idle first.
    // A zero extent (a minimised window) leaves the old swapchain in place and
    // reports `ErrorCode::Busy`, so the caller skips the frame.
    [[nodiscard]] Status<Error> Recreate(u32 width, u32 height) noexcept;

private:
    void Release() noexcept;
    [[nodiscard]] Status<Error> Build(u32 width, u32 height) noexcept;
    void DestroyImagesAndSemaphores() noexcept;

    VulkanContext* m_context = nullptr;
    vk::SurfaceKHR m_surface = 0;
    vk::SwapchainKHR m_swapchain = 0;
    vk::Format m_format = vk::Format::Undefined;
    vk::Extent2D m_extent{};
    u32 m_imageCount = 0;
    vk::Image m_images[kMaxSwapchainImages] = {};
    vk::ImageView m_views[kMaxSwapchainImages] = {};
    // One acquire semaphore per image, handed out round-robin; one render-finished
    // semaphore per image, indexed by the image acquired.
    vk::Semaphore m_acquire[kMaxSwapchainImages] = {};
    vk::Semaphore m_finished[kMaxSwapchainImages] = {};
    u32 m_acquireCursor = 0;
};

}  // namespace cyber::rhi::detail
