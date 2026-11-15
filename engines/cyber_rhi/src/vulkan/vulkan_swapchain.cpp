#include "vulkan_swapchain.h"

#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef interface
#undef interface
#endif
#endif

namespace cyber::rhi::detail
{
namespace
{
using cyber::core::ErrorCode;

constexpr u32 kUndefinedExtent = 0xFFFFFFFFU;

[[nodiscard]] u32 ClampU32(u32 value, u32 low, u32 high) noexcept
{
    return value < low ? low : (value > high ? high : value);
}
}  // namespace

VulkanSwapchain::VulkanSwapchain(VulkanSwapchain&& other) noexcept
{
    *this = std::move(other);
}

VulkanSwapchain& VulkanSwapchain::operator=(VulkanSwapchain&& other) noexcept
{
    if (this != &other)
    {
        Release();
        m_context = other.m_context;
        m_surface = other.m_surface;
        m_swapchain = other.m_swapchain;
        m_format = other.m_format;
        m_extent = other.m_extent;
        m_imageCount = other.m_imageCount;
        for (u32 index = 0; index < kMaxSwapchainImages; ++index)
        {
            m_images[index] = other.m_images[index];
            m_views[index] = other.m_views[index];
            m_acquire[index] = other.m_acquire[index];
            m_finished[index] = other.m_finished[index];
        }
        m_acquireCursor = other.m_acquireCursor;
        other.m_context = nullptr;
        other.m_surface = 0;
        other.m_swapchain = 0;
        other.m_imageCount = 0;
        for (u32 index = 0; index < kMaxSwapchainImages; ++index)
        {
            other.m_images[index] = 0;
            other.m_views[index] = 0;
            other.m_acquire[index] = 0;
            other.m_finished[index] = 0;
        }
    }
    return *this;
}

VulkanSwapchain::~VulkanSwapchain()
{
    Release();
}

void VulkanSwapchain::DestroyImagesAndSemaphores() noexcept
{
    const vk::DeviceApiExt& ext = m_context->DeviceFunctionsExt();
    const vk::Device device = m_context->Device();
    for (u32 index = 0; index < kMaxSwapchainImages; ++index)
    {
        if (m_views[index] != 0)
        {
            ext.destroyImageView(device, m_views[index], nullptr);
        }
        if (m_acquire[index] != 0)
        {
            ext.destroySemaphore(device, m_acquire[index], nullptr);
        }
        if (m_finished[index] != 0)
        {
            ext.destroySemaphore(device, m_finished[index], nullptr);
        }
        m_views[index] = 0;
        m_acquire[index] = 0;
        m_finished[index] = 0;
        // The images belong to the swapchain and die with it.
        m_images[index] = 0;
    }
    m_imageCount = 0;
    m_acquireCursor = 0;
}

void VulkanSwapchain::Release() noexcept
{
    if (m_context != nullptr && m_context->IsValid())
    {
        // Presentation or a submit that waits on these semaphores may be pending.
        (void)m_context->WaitIdle();
        DestroyImagesAndSemaphores();
        if (m_swapchain != 0)
        {
            m_context->SwapchainFunctions().destroySwapchain(m_context->Device(), m_swapchain,
                                                             nullptr);
        }
        if (m_surface != 0)
        {
            m_context->SurfaceFunctions().destroySurface(m_context->Instance(), m_surface, nullptr);
        }
    }
    m_context = nullptr;
    m_surface = 0;
    m_swapchain = 0;
}

Result<VulkanSwapchain, Error> VulkanSwapchain::Create(VulkanContext& context, void* windowHandle,
                                                       u32 width, u32 height) noexcept
{
    using Created = Result<VulkanSwapchain, Error>;
    if (!context.IsValid() || windowHandle == nullptr)
    {
        return Created::Err(Error{ErrorCode::InvalidArgument, "swapchain needs a context and a window"});
    }
    if (!context.Info().presentation)
    {
        return Created::Err(
            Error{ErrorCode::Unsupported, "the context was created without presentation support"});
    }
#if defined(_WIN32)
    VulkanSwapchain swapchain;
    swapchain.m_context = &context;

    vk::Win32SurfaceCreateInfoKHR surfaceInfo{};
    surfaceInfo.hinstance = GetModuleHandleW(nullptr);
    surfaceInfo.hwnd = windowHandle;
    if (context.SurfaceFunctions().createWin32Surface(context.Instance(), &surfaceInfo, nullptr,
                                                      &swapchain.m_surface) != vk::Result::Success)
    {
        swapchain.m_surface = 0;
        return Created::Err(Error{ErrorCode::Internal, "vkCreateWin32SurfaceKHR failed"});
    }

    // The graphics queue presents, so its family has to be able to.
    vk::Bool32 supported = vk::kFalse;
    if (context.SurfaceFunctions().getSurfaceSupport(context.PhysicalDevice(),
                                                     context.Info().queues.graphics,
                                                     swapchain.m_surface, &supported) !=
            vk::Result::Success ||
        supported == vk::kFalse)
    {
        return Created::Err(
            Error{ErrorCode::Unsupported, "the graphics queue cannot present to this surface"});
    }

    Status<Error> built = swapchain.Build(width, height);
    if (!built)
    {
        return Created::Err(built.Error());
    }
    return Created::Ok(std::move(swapchain));
#else
    (void)width;
    (void)height;
    return Created::Err(
        Error{ErrorCode::Unsupported, "this platform has no Vulkan surface backend yet"});
#endif
}

Status<Error> VulkanSwapchain::Build(u32 width, u32 height) noexcept
{
    const vk::SurfaceApi& surfaceApi = m_context->SurfaceFunctions();
    const vk::SwapchainApi& swapchainApi = m_context->SwapchainFunctions();
    const vk::DeviceApiExt& ext = m_context->DeviceFunctionsExt();
    const vk::Device device = m_context->Device();

    vk::SurfaceCapabilitiesKHR caps{};
    if (surfaceApi.getSurfaceCapabilities(m_context->PhysicalDevice(), m_surface, &caps) !=
        vk::Result::Success)
    {
        return Status<Error>::Err(
            Error{ErrorCode::Internal, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed"});
    }

    vk::Extent2D extent{};
    if (caps.currentExtent.width != kUndefinedExtent)
    {
        extent = caps.currentExtent;
    }
    else
    {
        extent.width = ClampU32(width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = ClampU32(height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    if (extent.width == 0 || extent.height == 0)
    {
        // A minimised window has no drawable area.
        return Status<Error>::Err(Error{ErrorCode::Busy, "the surface has no drawable area"});
    }

    vk::SurfaceFormatKHR formats[32] = {};
    u32 formatCount = 32;
    const vk::Result listedFormats = surfaceApi.getSurfaceFormats(
        m_context->PhysicalDevice(), m_surface, &formatCount, formats);
    if ((listedFormats != vk::Result::Success && listedFormats != vk::Result::Incomplete) ||
        formatCount == 0)
    {
        return Status<Error>::Err(Error{ErrorCode::Internal, "the surface reports no formats"});
    }
    if (formatCount > 32)
    {
        formatCount = 32;
    }
    // Prefer BGRA8 sRGB, which the RHI's default swapchain format asks for, then
    // BGRA8 unorm, then whatever the surface lists first.
    vk::SurfaceFormatKHR chosen = formats[0];
    bool found = false;
    for (u32 index = 0; index < formatCount && !found; ++index)
    {
        if (formats[index].format == vk::Format::B8G8R8A8Srgb && formats[index].colorSpace == 0)
        {
            chosen = formats[index];
            found = true;
        }
    }
    for (u32 index = 0; index < formatCount && !found; ++index)
    {
        if (formats[index].format == vk::Format::B8G8R8A8Unorm && formats[index].colorSpace == 0)
        {
            chosen = formats[index];
            found = true;
        }
    }

    u32 imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount != 0 && imageCount > caps.maxImageCount)
    {
        imageCount = caps.maxImageCount;
    }
    if (imageCount > kMaxSwapchainImages)
    {
        imageCount = kMaxSwapchainImages;
    }

    vk::Flags usage = vk::ImageUsage::kColorAttachment;
    if ((caps.supportedUsageFlags & vk::ImageUsage::kTransferSrc) != 0)
    {
        usage |= vk::ImageUsage::kTransferSrc;
    }
    vk::Flags alpha = 1;  // opaque
    if ((caps.supportedCompositeAlpha & alpha) == 0)
    {
        alpha = caps.supportedCompositeAlpha & (~caps.supportedCompositeAlpha + 1);
    }

    vk::SwapchainCreateInfoKHR info{};
    info.surface = m_surface;
    info.minImageCount = imageCount;
    info.imageFormat = chosen.format;
    info.imageColorSpace = chosen.colorSpace;
    info.imageExtent = extent;
    info.imageUsage = usage;
    info.preTransform = caps.currentTransform;
    info.compositeAlpha = alpha;
    info.presentMode = 2;  // FIFO: the only mode every implementation has
    info.oldSwapchain = m_swapchain;
    vk::SwapchainKHR created = 0;
    const vk::Result result = swapchainApi.createSwapchain(device, &info, nullptr, &created);
    if (info.oldSwapchain != 0)
    {
        // The old swapchain is retired whether or not the new one was created.
        swapchainApi.destroySwapchain(device, info.oldSwapchain, nullptr);
        m_swapchain = 0;
    }
    if (result != vk::Result::Success)
    {
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkCreateSwapchainKHR failed"});
    }
    m_swapchain = created;
    m_format = chosen.format;
    m_extent = extent;

    u32 count = kMaxSwapchainImages;
    const vk::Result gotImages = swapchainApi.getSwapchainImages(device, m_swapchain, &count, m_images);
    if ((gotImages != vk::Result::Success && gotImages != vk::Result::Incomplete) || count == 0)
    {
        return Status<Error>::Err(Error{ErrorCode::Internal, "vkGetSwapchainImagesKHR failed"});
    }
    m_imageCount = count < kMaxSwapchainImages ? count : kMaxSwapchainImages;

    for (u32 index = 0; index < m_imageCount; ++index)
    {
        vk::ImageViewCreateInfo viewInfo{};
        viewInfo.image = m_images[index];
        viewInfo.format = m_format;
        viewInfo.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspect::kColor, 0, 1, 0, 1};
        if (ext.createImageView(device, &viewInfo, nullptr, &m_views[index]) != vk::Result::Success)
        {
            m_views[index] = 0;
            return Status<Error>::Err(Error{ErrorCode::Internal, "vkCreateImageView failed"});
        }
        vk::SemaphoreCreateInfo semaphoreInfo{};
        if (ext.createSemaphore(device, &semaphoreInfo, nullptr, &m_acquire[index]) !=
                vk::Result::Success ||
            ext.createSemaphore(device, &semaphoreInfo, nullptr, &m_finished[index]) !=
                vk::Result::Success)
        {
            return Status<Error>::Err(Error{ErrorCode::Internal, "vkCreateSemaphore failed"});
        }
    }
    return Status<Error>::Ok();
}

Status<Error> VulkanSwapchain::Recreate(u32 width, u32 height) noexcept
{
    if (m_context == nullptr || m_surface == 0)
    {
        return Status<Error>::Err(Error{ErrorCode::InvalidArgument, "swapchain is empty"});
    }
    Status<Error> idle = m_context->WaitIdle();
    if (!idle)
    {
        return idle;
    }
    DestroyImagesAndSemaphores();
    return Build(width, height);
}

Result<VulkanPresentResult, Error> VulkanSwapchain::Acquire(u32* imageIndex,
                                                            vk::Semaphore* imageAvailable,
                                                            vk::Semaphore* renderFinished,
                                                            u64 timeoutNanoseconds) noexcept
{
    using Outcome = Result<VulkanPresentResult, Error>;
    if (!IsValid() || imageIndex == nullptr || imageAvailable == nullptr ||
        renderFinished == nullptr)
    {
        return Outcome::Err(Error{ErrorCode::InvalidArgument, "swapchain is empty"});
    }
    const vk::Semaphore semaphore = m_acquire[m_acquireCursor % m_imageCount];
    u32 index = 0;
    const vk::Result result = m_context->SwapchainFunctions().acquireNextImage(
        m_context->Device(), m_swapchain, timeoutNanoseconds, semaphore, 0, &index);
    if (result == vk::kResultErrorOutOfDate)
    {
        return Outcome::Ok(VulkanPresentResult::NeedsRecreate);
    }
    if (result == vk::Result::Timeout || result == vk::Result::NotReady)
    {
        return Outcome::Err(Error{ErrorCode::Busy, "no swapchain image became available"});
    }
    if (result == vk::kResultErrorSurfaceLost)
    {
        return Outcome::Err(Error{ErrorCode::DeviceLost, "the surface was lost"});
    }
    if (result != vk::Result::Success && result != vk::kResultSuboptimal)
    {
        return Outcome::Err(Error{ErrorCode::Internal, "vkAcquireNextImageKHR failed"});
    }
    ++m_acquireCursor;
    *imageIndex = index;
    *imageAvailable = semaphore;
    *renderFinished = m_finished[index];
    return Outcome::Ok(VulkanPresentResult::Ok);
}

Result<VulkanPresentResult, Error> VulkanSwapchain::Present(u32 imageIndex,
                                                            vk::Semaphore renderFinished) noexcept
{
    using Outcome = Result<VulkanPresentResult, Error>;
    if (!IsValid() || imageIndex >= m_imageCount)
    {
        return Outcome::Err(Error{ErrorCode::InvalidArgument, "bad swapchain image"});
    }
    vk::PresentInfoKHR info{};
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &renderFinished;
    info.swapchainCount = 1;
    info.pSwapchains = &m_swapchain;
    info.pImageIndices = &imageIndex;
    const vk::Result result =
        m_context->SwapchainFunctions().queuePresent(m_context->GraphicsQueue(), &info);
    if (result == vk::kResultErrorOutOfDate || result == vk::kResultSuboptimal)
    {
        return Outcome::Ok(VulkanPresentResult::NeedsRecreate);
    }
    if (result == vk::kResultErrorSurfaceLost)
    {
        return Outcome::Err(Error{ErrorCode::DeviceLost, "the surface was lost"});
    }
    if (result == vk::Result::ErrorDeviceLost)
    {
        return Outcome::Err(Error{ErrorCode::DeviceLost, "Vulkan device lost on present"});
    }
    if (result != vk::Result::Success)
    {
        return Outcome::Err(Error{ErrorCode::Internal, "vkQueuePresentKHR failed"});
    }
    return Outcome::Ok(VulkanPresentResult::Ok);
}

}  // namespace cyber::rhi::detail
