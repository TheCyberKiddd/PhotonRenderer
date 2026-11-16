#pragma once

// The second slice of the Vulkan API subset (see `vk_api.h` and ADR 0029):
// images, samplers, commands, synchronization2, timeline semaphores,
// descriptors, shaders, pipelines with dynamic rendering, and presentation.
// Struct layouts and enumerator values were checked against the specification's
// reference pages (VkImageCreateInfo, VkImageMemoryBarrier2, VkRenderingInfo,
// VkSubmitInfo2) and, for the rest, against the registry's definitions.

#include "vk_api.h"

namespace cyber::rhi::detail::vk
{

using i32 = cyber::core::i32;

// Non-dispatchable handles.
using Image = u64;
using ImageView = u64;
using Sampler = u64;
using Semaphore = u64;
using CommandPool = u64;
using ShaderModule = u64;
using Pipeline = u64;
using PipelineLayout = u64;
using PipelineCache = u64;
using DescriptorSetLayout = u64;
using DescriptorPool = u64;
using DescriptorSet = u64;
using SurfaceKHR = u64;
using SwapchainKHR = u64;
struct CommandBufferT;
using CommandBuffer = CommandBufferT*;

// Structure types added in this slice.
namespace st
{
constexpr StructureType Make(u32 value) noexcept
{
    return static_cast<StructureType>(value);
}
inline constexpr StructureType kSemaphoreCreateInfo = Make(9);
inline constexpr StructureType kImageCreateInfo = Make(14);
inline constexpr StructureType kImageViewCreateInfo = Make(15);
inline constexpr StructureType kShaderModuleCreateInfo = Make(16);
inline constexpr StructureType kPipelineShaderStageCreateInfo = Make(18);
inline constexpr StructureType kPipelineVertexInputStateCreateInfo = Make(19);
inline constexpr StructureType kPipelineInputAssemblyStateCreateInfo = Make(20);
inline constexpr StructureType kPipelineViewportStateCreateInfo = Make(22);
inline constexpr StructureType kPipelineRasterizationStateCreateInfo = Make(23);
inline constexpr StructureType kPipelineMultisampleStateCreateInfo = Make(24);
inline constexpr StructureType kPipelineDepthStencilStateCreateInfo = Make(25);
inline constexpr StructureType kPipelineColorBlendStateCreateInfo = Make(26);
inline constexpr StructureType kPipelineDynamicStateCreateInfo = Make(27);
inline constexpr StructureType kGraphicsPipelineCreateInfo = Make(28);
inline constexpr StructureType kComputePipelineCreateInfo = Make(29);
inline constexpr StructureType kPipelineLayoutCreateInfo = Make(30);
inline constexpr StructureType kSamplerCreateInfo = Make(31);
inline constexpr StructureType kDescriptorSetLayoutCreateInfo = Make(32);
inline constexpr StructureType kDescriptorPoolCreateInfo = Make(33);
inline constexpr StructureType kDescriptorSetAllocateInfo = Make(34);
inline constexpr StructureType kWriteDescriptorSet = Make(35);
inline constexpr StructureType kCommandPoolCreateInfo = Make(39);
inline constexpr StructureType kCommandBufferAllocateInfo = Make(40);
inline constexpr StructureType kCommandBufferBeginInfo = Make(42);
inline constexpr StructureType kSwapchainCreateInfoKHR = Make(1000001000);
inline constexpr StructureType kPresentInfoKHR = Make(1000001001);
inline constexpr StructureType kWin32SurfaceCreateInfoKHR = Make(1000009000);
inline constexpr StructureType kPipelineRenderingCreateInfo = Make(1000044002);
inline constexpr StructureType kRenderingInfo = Make(1000044000);
inline constexpr StructureType kRenderingAttachmentInfo = Make(1000044001);
inline constexpr StructureType kDescriptorSetLayoutBindingFlagsCreateInfo = Make(1000161000);
inline constexpr StructureType kDescriptorSetVariableDescriptorCountAllocateInfo = Make(1000161003);
inline constexpr StructureType kSemaphoreTypeCreateInfo = Make(1000207002);
inline constexpr StructureType kSemaphoreWaitInfo = Make(1000207004);
inline constexpr StructureType kSemaphoreSignalInfo = Make(1000207005);
inline constexpr StructureType kMemoryBarrier2 = Make(1000314000);
inline constexpr StructureType kBufferMemoryBarrier2 = Make(1000314001);
inline constexpr StructureType kImageMemoryBarrier2 = Make(1000314002);
inline constexpr StructureType kDependencyInfo = Make(1000314003);
inline constexpr StructureType kSubmitInfo2 = Make(1000314004);
inline constexpr StructureType kSemaphoreSubmitInfo = Make(1000314005);
inline constexpr StructureType kCommandBufferSubmitInfo = Make(1000314006);
}  // namespace st

// Results beyond the base enum.
inline constexpr Result kResultSuboptimal = static_cast<Result>(1000001003);
inline constexpr Result kResultErrorOutOfDate = static_cast<Result>(-1000001004);
inline constexpr Result kResultErrorSurfaceLost = static_cast<Result>(-1000000000);

// Formats the backend uses. The full table arrives with the RHI format mapping.
enum class Format : u32
{
    Undefined = 0,
    R8Unorm = 9,
    R8G8B8A8Unorm = 37,
    R8G8B8A8Srgb = 43,
    B8G8R8A8Unorm = 44,
    B8G8R8A8Srgb = 50,
    R16G16B16A16Sfloat = 97,
    R32Sfloat = 100,
    R32G32Sfloat = 103,
    R32G32B32Sfloat = 106,
    R32G32B32A32Sfloat = 109,
    D32Sfloat = 126,
};

enum class ImageLayout : u32
{
    Undefined = 0,
    General = 1,
    ColorAttachmentOptimal = 2,
    DepthStencilAttachmentOptimal = 3,
    ShaderReadOnlyOptimal = 5,
    TransferSrcOptimal = 6,
    TransferDstOptimal = 7,
    PresentSrcKHR = 1000001002,
};

namespace ImageUsage
{
inline constexpr Flags kTransferSrc = 0x1;
inline constexpr Flags kTransferDst = 0x2;
inline constexpr Flags kSampled = 0x4;
inline constexpr Flags kStorage = 0x8;
inline constexpr Flags kColorAttachment = 0x10;
inline constexpr Flags kDepthStencilAttachment = 0x20;
}  // namespace ImageUsage

namespace ImageAspect
{
inline constexpr Flags kColor = 0x1;
inline constexpr Flags kDepth = 0x2;
inline constexpr Flags kStencil = 0x4;
}  // namespace ImageAspect

enum class ImageType : u32
{
    Type1D = 0,
    Type2D = 1,
    Type3D = 2,
};
enum class ImageViewType : u32
{
    Type1D = 0,
    Type2D = 1,
    Type3D = 2,
    Cube = 3,
    Type2DArray = 5,
};
enum class ImageTiling : u32
{
    Optimal = 0,
    Linear = 1,
};

struct Offset2D
{
    i32 x = 0;
    i32 y = 0;
};
struct Offset3D
{
    i32 x = 0;
    i32 y = 0;
    i32 z = 0;
};
struct Extent2D
{
    u32 width = 0;
    u32 height = 0;
};
struct Rect2D
{
    Offset2D offset{};
    Extent2D extent{};
};
struct Viewport
{
    float x = 0;
    float y = 0;
    float width = 0;
    float height = 0;
    float minDepth = 0;
    float maxDepth = 1;
};

struct ImageSubresourceRange
{
    Flags aspectMask = 0;
    u32 baseMipLevel = 0;
    u32 levelCount = 1;
    u32 baseArrayLayer = 0;
    u32 layerCount = 1;
};
struct ImageSubresourceLayers
{
    Flags aspectMask = 0;
    u32 mipLevel = 0;
    u32 baseArrayLayer = 0;
    u32 layerCount = 1;
};

struct ImageCreateInfo
{
    StructureType sType = st::kImageCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    ImageType imageType = ImageType::Type2D;
    Format format = Format::Undefined;
    Extent3D extent{1, 1, 1};
    u32 mipLevels = 1;
    u32 arrayLayers = 1;
    u32 samples = 1;
    ImageTiling tiling = ImageTiling::Optimal;
    Flags usage = 0;
    SharingMode sharingMode = SharingMode::Exclusive;
    u32 queueFamilyIndexCount = 0;
    const u32* pQueueFamilyIndices = nullptr;
    ImageLayout initialLayout = ImageLayout::Undefined;
};

struct ComponentMapping
{
    u32 r = 0;
    u32 g = 0;
    u32 b = 0;
    u32 a = 0;
};

struct ImageViewCreateInfo
{
    StructureType sType = st::kImageViewCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    Image image = 0;
    ImageViewType viewType = ImageViewType::Type2D;
    Format format = Format::Undefined;
    ComponentMapping components{};
    ImageSubresourceRange subresourceRange{};
};

enum class Filter : u32
{
    Nearest = 0,
    Linear = 1,
};
enum class SamplerMipmapMode : u32
{
    Nearest = 0,
    Linear = 1,
};
enum class SamplerAddressMode : u32
{
    Repeat = 0,
    MirroredRepeat = 1,
    ClampToEdge = 2,
    ClampToBorder = 3,
};
enum class CompareOp : u32
{
    Never = 0,
    Less = 1,
    Equal = 2,
    LessOrEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterOrEqual = 6,
    Always = 7,
};
enum class BorderColor : u32
{
    FloatTransparentBlack = 0,
    FloatOpaqueBlack = 2,
    FloatOpaqueWhite = 4,
};

struct SamplerCreateInfo
{
    StructureType sType = st::kSamplerCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    Filter magFilter = Filter::Nearest;
    Filter minFilter = Filter::Nearest;
    SamplerMipmapMode mipmapMode = SamplerMipmapMode::Nearest;
    SamplerAddressMode addressModeU = SamplerAddressMode::Repeat;
    SamplerAddressMode addressModeV = SamplerAddressMode::Repeat;
    SamplerAddressMode addressModeW = SamplerAddressMode::Repeat;
    float mipLodBias = 0;
    Bool32 anisotropyEnable = kFalse;
    float maxAnisotropy = 1;
    Bool32 compareEnable = kFalse;
    CompareOp compareOp = CompareOp::Never;
    float minLod = 0;
    float maxLod = 0;
    BorderColor borderColor = BorderColor::FloatTransparentBlack;
    Bool32 unnormalizedCoordinates = kFalse;
};

union ClearColorValue
{
    float float32[4];
    i32 int32[4];
    u32 uint32[4];
};
struct ClearDepthStencilValue
{
    float depth = 1.0F;
    u32 stencil = 0;
};
struct ImageCopy
{
    ImageSubresourceLayers srcSubresource{};
    Offset3D srcOffset{};
    ImageSubresourceLayers dstSubresource{};
    Offset3D dstOffset{};
    Extent3D extent{1, 1, 1};
};
// `VkDebugUtilsLabelEXT`, declared in full here (`vk_api.h` only forward-declares it).
struct DebugLabel
{
    StructureType sType = st::Make(1000128002);
    const void* pNext = nullptr;
    const char* pLabelName = nullptr;
    float color[4] = {0, 0, 0, 0};
};

// Commands.
namespace CommandPoolCreate
{
inline constexpr Flags kTransient = 0x1;
inline constexpr Flags kResetCommandBuffer = 0x2;
}  // namespace CommandPoolCreate

struct CommandPoolCreateInfo
{
    StructureType sType = st::kCommandPoolCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    u32 queueFamilyIndex = 0;
};
struct CommandBufferAllocateInfo
{
    StructureType sType = st::kCommandBufferAllocateInfo;
    const void* pNext = nullptr;
    CommandPool commandPool = 0;
    u32 level = 0;  // primary
    u32 commandBufferCount = 1;
};
struct CommandBufferBeginInfo
{
    StructureType sType = st::kCommandBufferBeginInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    const void* pInheritanceInfo = nullptr;
};
inline constexpr Flags kCommandBufferOneTimeSubmit = 0x1;

struct BufferCopy
{
    DeviceSize srcOffset = 0;
    DeviceSize dstOffset = 0;
    DeviceSize size = 0;
};
struct BufferImageCopy
{
    DeviceSize bufferOffset = 0;
    u32 bufferRowLength = 0;
    u32 bufferImageHeight = 0;
    ImageSubresourceLayers imageSubresource{};
    Offset3D imageOffset{};
    Extent3D imageExtent{1, 1, 1};
};

// Synchronization2. Stage and access masks are 64-bit.
using Flags64 = u64;
namespace Stage2
{
inline constexpr Flags64 kNone = 0;
inline constexpr Flags64 kTopOfPipe = 0x1;
inline constexpr Flags64 kDrawIndirect = 0x2;
inline constexpr Flags64 kVertexInput = 0x4;
inline constexpr Flags64 kVertexShader = 0x8;
inline constexpr Flags64 kFragmentShader = 0x80;
inline constexpr Flags64 kEarlyFragmentTests = 0x100;
inline constexpr Flags64 kLateFragmentTests = 0x200;
inline constexpr Flags64 kColorAttachmentOutput = 0x400;
inline constexpr Flags64 kComputeShader = 0x800;
inline constexpr Flags64 kAllTransfer = 0x1000;
inline constexpr Flags64 kBottomOfPipe = 0x2000;
inline constexpr Flags64 kHost = 0x4000;
inline constexpr Flags64 kAllGraphics = 0x8000;
inline constexpr Flags64 kAllCommands = 0x10000;
inline constexpr Flags64 kCopy = 0x100000000ULL;
inline constexpr Flags64 kClear = 0x800000000ULL;
}  // namespace Stage2
namespace Access2
{
inline constexpr Flags64 kNone = 0;
inline constexpr Flags64 kIndirectCommandRead = 0x1;
inline constexpr Flags64 kIndexRead = 0x2;
inline constexpr Flags64 kVertexAttributeRead = 0x4;
inline constexpr Flags64 kUniformRead = 0x8;
inline constexpr Flags64 kShaderRead = 0x20;
inline constexpr Flags64 kShaderWrite = 0x40;
inline constexpr Flags64 kColorAttachmentRead = 0x80;
inline constexpr Flags64 kColorAttachmentWrite = 0x100;
inline constexpr Flags64 kDepthStencilRead = 0x200;
inline constexpr Flags64 kDepthStencilWrite = 0x400;
inline constexpr Flags64 kTransferRead = 0x800;
inline constexpr Flags64 kTransferWrite = 0x1000;
inline constexpr Flags64 kHostRead = 0x2000;
inline constexpr Flags64 kHostWrite = 0x4000;
inline constexpr Flags64 kMemoryRead = 0x8000;
inline constexpr Flags64 kMemoryWrite = 0x10000;
inline constexpr Flags64 kShaderSampledRead = 0x100000000ULL;
inline constexpr Flags64 kShaderStorageRead = 0x200000000ULL;
inline constexpr Flags64 kShaderStorageWrite = 0x400000000ULL;
}  // namespace Access2

inline constexpr u32 kQueueFamilyIgnored = 0xFFFFFFFFU;

struct MemoryBarrier2
{
    StructureType sType = st::kMemoryBarrier2;
    const void* pNext = nullptr;
    Flags64 srcStageMask = 0;
    Flags64 srcAccessMask = 0;
    Flags64 dstStageMask = 0;
    Flags64 dstAccessMask = 0;
};
struct BufferMemoryBarrier2
{
    StructureType sType = st::kBufferMemoryBarrier2;
    const void* pNext = nullptr;
    Flags64 srcStageMask = 0;
    Flags64 srcAccessMask = 0;
    Flags64 dstStageMask = 0;
    Flags64 dstAccessMask = 0;
    u32 srcQueueFamilyIndex = kQueueFamilyIgnored;
    u32 dstQueueFamilyIndex = kQueueFamilyIgnored;
    Buffer buffer = 0;
    DeviceSize offset = 0;
    DeviceSize size = kWholeSize;
};
struct ImageMemoryBarrier2
{
    StructureType sType = st::kImageMemoryBarrier2;
    const void* pNext = nullptr;
    Flags64 srcStageMask = 0;
    Flags64 srcAccessMask = 0;
    Flags64 dstStageMask = 0;
    Flags64 dstAccessMask = 0;
    ImageLayout oldLayout = ImageLayout::Undefined;
    ImageLayout newLayout = ImageLayout::Undefined;
    u32 srcQueueFamilyIndex = kQueueFamilyIgnored;
    u32 dstQueueFamilyIndex = kQueueFamilyIgnored;
    Image image = 0;
    ImageSubresourceRange subresourceRange{};
};
struct DependencyInfo
{
    StructureType sType = st::kDependencyInfo;
    const void* pNext = nullptr;
    Flags dependencyFlags = 0;
    u32 memoryBarrierCount = 0;
    const MemoryBarrier2* pMemoryBarriers = nullptr;
    u32 bufferMemoryBarrierCount = 0;
    const BufferMemoryBarrier2* pBufferMemoryBarriers = nullptr;
    u32 imageMemoryBarrierCount = 0;
    const ImageMemoryBarrier2* pImageMemoryBarriers = nullptr;
};

struct SemaphoreSubmitInfo
{
    StructureType sType = st::kSemaphoreSubmitInfo;
    const void* pNext = nullptr;
    Semaphore semaphore = 0;
    u64 value = 0;
    Flags64 stageMask = 0;
    u32 deviceIndex = 0;
};
struct CommandBufferSubmitInfo
{
    StructureType sType = st::kCommandBufferSubmitInfo;
    const void* pNext = nullptr;
    CommandBuffer commandBuffer = nullptr;
    u32 deviceMask = 0;
};
struct SubmitInfo2
{
    StructureType sType = st::kSubmitInfo2;
    const void* pNext = nullptr;
    Flags flags = 0;
    u32 waitSemaphoreInfoCount = 0;
    const SemaphoreSubmitInfo* pWaitSemaphoreInfos = nullptr;
    u32 commandBufferInfoCount = 0;
    const CommandBufferSubmitInfo* pCommandBufferInfos = nullptr;
    u32 signalSemaphoreInfoCount = 0;
    const SemaphoreSubmitInfo* pSignalSemaphoreInfos = nullptr;
};

struct SemaphoreCreateInfo
{
    StructureType sType = st::kSemaphoreCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
};
struct SemaphoreTypeCreateInfo
{
    StructureType sType = st::kSemaphoreTypeCreateInfo;
    const void* pNext = nullptr;
    u32 semaphoreType = 1;  // timeline
    u64 initialValue = 0;
};
struct SemaphoreWaitInfo
{
    StructureType sType = st::kSemaphoreWaitInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    u32 semaphoreCount = 0;
    const Semaphore* pSemaphores = nullptr;
    const u64* pValues = nullptr;
};
struct SemaphoreSignalInfo
{
    StructureType sType = st::kSemaphoreSignalInfo;
    const void* pNext = nullptr;
    Semaphore semaphore = 0;
    u64 value = 0;
};

// Descriptors.
enum class DescriptorType : u32
{
    Sampler = 0,
    CombinedImageSampler = 1,
    SampledImage = 2,
    StorageImage = 3,
    UniformBuffer = 6,
    StorageBuffer = 7,
};
namespace ShaderStage
{
inline constexpr Flags kVertex = 0x1;
inline constexpr Flags kFragment = 0x10;
inline constexpr Flags kCompute = 0x20;
inline constexpr Flags kAllGraphics = 0x1F;
}  // namespace ShaderStage
namespace DescriptorBindingFlag
{
inline constexpr Flags kUpdateAfterBind = 0x1;
inline constexpr Flags kPartiallyBound = 0x4;
inline constexpr Flags kVariableDescriptorCount = 0x8;
}  // namespace DescriptorBindingFlag
inline constexpr Flags kDescriptorSetLayoutUpdateAfterBindPool = 0x2;
inline constexpr Flags kDescriptorPoolFreeDescriptorSet = 0x1;
inline constexpr Flags kDescriptorPoolUpdateAfterBind = 0x2;

struct DescriptorSetLayoutBinding
{
    u32 binding = 0;
    DescriptorType descriptorType = DescriptorType::StorageBuffer;
    u32 descriptorCount = 1;
    Flags stageFlags = 0;
    const Sampler* pImmutableSamplers = nullptr;
};
struct DescriptorSetLayoutCreateInfo
{
    StructureType sType = st::kDescriptorSetLayoutCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    u32 bindingCount = 0;
    const DescriptorSetLayoutBinding* pBindings = nullptr;
};
struct DescriptorSetLayoutBindingFlagsCreateInfo
{
    StructureType sType = st::kDescriptorSetLayoutBindingFlagsCreateInfo;
    const void* pNext = nullptr;
    u32 bindingCount = 0;
    const Flags* pBindingFlags = nullptr;
};
struct DescriptorPoolSize
{
    DescriptorType type = DescriptorType::StorageBuffer;
    u32 descriptorCount = 0;
};
struct DescriptorPoolCreateInfo
{
    StructureType sType = st::kDescriptorPoolCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    u32 maxSets = 0;
    u32 poolSizeCount = 0;
    const DescriptorPoolSize* pPoolSizes = nullptr;
};
struct DescriptorSetAllocateInfo
{
    StructureType sType = st::kDescriptorSetAllocateInfo;
    const void* pNext = nullptr;
    DescriptorPool descriptorPool = 0;
    u32 descriptorSetCount = 1;
    const DescriptorSetLayout* pSetLayouts = nullptr;
};
struct DescriptorSetVariableDescriptorCountAllocateInfo
{
    StructureType sType = st::kDescriptorSetVariableDescriptorCountAllocateInfo;
    const void* pNext = nullptr;
    u32 descriptorSetCount = 0;
    const u32* pDescriptorCounts = nullptr;
};
struct DescriptorBufferInfo
{
    Buffer buffer = 0;
    DeviceSize offset = 0;
    DeviceSize range = kWholeSize;
};
struct DescriptorImageInfo
{
    Sampler sampler = 0;
    ImageView imageView = 0;
    ImageLayout imageLayout = ImageLayout::Undefined;
};
struct WriteDescriptorSet
{
    StructureType sType = st::kWriteDescriptorSet;
    const void* pNext = nullptr;
    DescriptorSet dstSet = 0;
    u32 dstBinding = 0;
    u32 dstArrayElement = 0;
    u32 descriptorCount = 1;
    DescriptorType descriptorType = DescriptorType::StorageBuffer;
    const DescriptorImageInfo* pImageInfo = nullptr;
    const DescriptorBufferInfo* pBufferInfo = nullptr;
    const u64* pTexelBufferView = nullptr;
};
struct PushConstantRange
{
    Flags stageFlags = 0;
    u32 offset = 0;
    u32 size = 0;
};
struct PipelineLayoutCreateInfo
{
    StructureType sType = st::kPipelineLayoutCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    u32 setLayoutCount = 0;
    const DescriptorSetLayout* pSetLayouts = nullptr;
    u32 pushConstantRangeCount = 0;
    const PushConstantRange* pPushConstantRanges = nullptr;
};

// Shaders and pipelines.
struct ShaderModuleCreateInfo
{
    StructureType sType = st::kShaderModuleCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    usize codeSize = 0;
    const u32* pCode = nullptr;
};
struct PipelineShaderStageCreateInfo
{
    StructureType sType = st::kPipelineShaderStageCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    Flags stage = 0;
    ShaderModule module = 0;
    const char* pName = nullptr;
    const void* pSpecializationInfo = nullptr;
};
struct ComputePipelineCreateInfo
{
    StructureType sType = st::kComputePipelineCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    PipelineShaderStageCreateInfo stage{};
    PipelineLayout layout = 0;
    Pipeline basePipelineHandle = 0;
    i32 basePipelineIndex = -1;
};

struct VertexInputBindingDescription
{
    u32 binding = 0;
    u32 stride = 0;
    u32 inputRate = 0;
};
struct VertexInputAttributeDescription
{
    u32 location = 0;
    u32 binding = 0;
    Format format = Format::Undefined;
    u32 offset = 0;
};
struct PipelineVertexInputStateCreateInfo
{
    StructureType sType = st::kPipelineVertexInputStateCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    u32 vertexBindingDescriptionCount = 0;
    const VertexInputBindingDescription* pVertexBindingDescriptions = nullptr;
    u32 vertexAttributeDescriptionCount = 0;
    const VertexInputAttributeDescription* pVertexAttributeDescriptions = nullptr;
};
struct PipelineInputAssemblyStateCreateInfo
{
    StructureType sType = st::kPipelineInputAssemblyStateCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    u32 topology = 3;  // triangle list
    Bool32 primitiveRestartEnable = kFalse;
};
struct PipelineViewportStateCreateInfo
{
    StructureType sType = st::kPipelineViewportStateCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    u32 viewportCount = 1;
    const Viewport* pViewports = nullptr;
    u32 scissorCount = 1;
    const Rect2D* pScissors = nullptr;
};
struct PipelineRasterizationStateCreateInfo
{
    StructureType sType = st::kPipelineRasterizationStateCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    Bool32 depthClampEnable = kFalse;
    Bool32 rasterizerDiscardEnable = kFalse;
    u32 polygonMode = 0;  // fill
    Flags cullMode = 0;
    u32 frontFace = 0;  // counter-clockwise
    Bool32 depthBiasEnable = kFalse;
    float depthBiasConstantFactor = 0;
    float depthBiasClamp = 0;
    float depthBiasSlopeFactor = 0;
    float lineWidth = 1;
};
struct PipelineMultisampleStateCreateInfo
{
    StructureType sType = st::kPipelineMultisampleStateCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    u32 rasterizationSamples = 1;
    Bool32 sampleShadingEnable = kFalse;
    float minSampleShading = 0;
    const u32* pSampleMask = nullptr;
    Bool32 alphaToCoverageEnable = kFalse;
    Bool32 alphaToOneEnable = kFalse;
};
struct StencilOpState
{
    u32 failOp = 0;
    u32 passOp = 0;
    u32 depthFailOp = 0;
    CompareOp compareOp = CompareOp::Always;
    u32 compareMask = 0;
    u32 writeMask = 0;
    u32 reference = 0;
};
struct PipelineDepthStencilStateCreateInfo
{
    StructureType sType = st::kPipelineDepthStencilStateCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    Bool32 depthTestEnable = kFalse;
    Bool32 depthWriteEnable = kFalse;
    CompareOp depthCompareOp = CompareOp::Always;
    Bool32 depthBoundsTestEnable = kFalse;
    Bool32 stencilTestEnable = kFalse;
    StencilOpState front{};
    StencilOpState back{};
    float minDepthBounds = 0;
    float maxDepthBounds = 1;
};
struct PipelineColorBlendAttachmentState
{
    Bool32 blendEnable = kFalse;
    u32 srcColorBlendFactor = 1;  // one
    u32 dstColorBlendFactor = 0;  // zero
    u32 colorBlendOp = 0;         // add
    u32 srcAlphaBlendFactor = 1;
    u32 dstAlphaBlendFactor = 0;
    u32 alphaBlendOp = 0;
    Flags colorWriteMask = 0xF;
};
struct PipelineColorBlendStateCreateInfo
{
    StructureType sType = st::kPipelineColorBlendStateCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    Bool32 logicOpEnable = kFalse;
    u32 logicOp = 0;
    u32 attachmentCount = 0;
    const PipelineColorBlendAttachmentState* pAttachments = nullptr;
    float blendConstants[4] = {0, 0, 0, 0};
};
inline constexpr u32 kDynamicStateViewport = 0;
inline constexpr u32 kDynamicStateScissor = 1;
struct PipelineDynamicStateCreateInfo
{
    StructureType sType = st::kPipelineDynamicStateCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    u32 dynamicStateCount = 0;
    const u32* pDynamicStates = nullptr;
};
struct PipelineRenderingCreateInfo
{
    StructureType sType = st::kPipelineRenderingCreateInfo;
    const void* pNext = nullptr;
    u32 viewMask = 0;
    u32 colorAttachmentCount = 0;
    const Format* pColorAttachmentFormats = nullptr;
    Format depthAttachmentFormat = Format::Undefined;
    Format stencilAttachmentFormat = Format::Undefined;
};
struct GraphicsPipelineCreateInfo
{
    StructureType sType = st::kGraphicsPipelineCreateInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    u32 stageCount = 0;
    const PipelineShaderStageCreateInfo* pStages = nullptr;
    const PipelineVertexInputStateCreateInfo* pVertexInputState = nullptr;
    const PipelineInputAssemblyStateCreateInfo* pInputAssemblyState = nullptr;
    const void* pTessellationState = nullptr;
    const PipelineViewportStateCreateInfo* pViewportState = nullptr;
    const PipelineRasterizationStateCreateInfo* pRasterizationState = nullptr;
    const PipelineMultisampleStateCreateInfo* pMultisampleState = nullptr;
    const PipelineDepthStencilStateCreateInfo* pDepthStencilState = nullptr;
    const PipelineColorBlendStateCreateInfo* pColorBlendState = nullptr;
    const PipelineDynamicStateCreateInfo* pDynamicState = nullptr;
    PipelineLayout layout = 0;
    u64 renderPass = 0;
    u32 subpass = 0;
    Pipeline basePipelineHandle = 0;
    i32 basePipelineIndex = -1;
};

// Dynamic rendering.
union ClearValue
{
    float color[4];
    struct
    {
        float depth;
        u32 stencil;
    } depthStencil;
};
inline constexpr u32 kAttachmentLoadOpLoad = 0;
inline constexpr u32 kAttachmentLoadOpClear = 1;
inline constexpr u32 kAttachmentLoadOpDontCare = 2;
inline constexpr u32 kAttachmentStoreOpStore = 0;
inline constexpr u32 kAttachmentStoreOpDontCare = 1;
struct RenderingAttachmentInfo
{
    StructureType sType = st::kRenderingAttachmentInfo;
    const void* pNext = nullptr;
    ImageView imageView = 0;
    ImageLayout imageLayout = ImageLayout::Undefined;
    u32 resolveMode = 0;
    ImageView resolveImageView = 0;
    ImageLayout resolveImageLayout = ImageLayout::Undefined;
    u32 loadOp = kAttachmentLoadOpLoad;
    u32 storeOp = kAttachmentStoreOpStore;
    ClearValue clearValue{};
};
struct RenderingInfo
{
    StructureType sType = st::kRenderingInfo;
    const void* pNext = nullptr;
    Flags flags = 0;
    Rect2D renderArea{};
    u32 layerCount = 1;
    u32 viewMask = 0;
    u32 colorAttachmentCount = 0;
    const RenderingAttachmentInfo* pColorAttachments = nullptr;
    const RenderingAttachmentInfo* pDepthAttachment = nullptr;
    const RenderingAttachmentInfo* pStencilAttachment = nullptr;
};

// Presentation.
struct Win32SurfaceCreateInfoKHR
{
    StructureType sType = st::kWin32SurfaceCreateInfoKHR;
    const void* pNext = nullptr;
    Flags flags = 0;
    void* hinstance = nullptr;
    void* hwnd = nullptr;
};
struct SurfaceCapabilitiesKHR
{
    u32 minImageCount;
    u32 maxImageCount;
    Extent2D currentExtent;
    Extent2D minImageExtent;
    Extent2D maxImageExtent;
    u32 maxImageArrayLayers;
    Flags supportedTransforms;
    Flags currentTransform;
    Flags supportedCompositeAlpha;
    Flags supportedUsageFlags;
};
struct SurfaceFormatKHR
{
    Format format;
    u32 colorSpace;
};
struct SwapchainCreateInfoKHR
{
    StructureType sType = st::kSwapchainCreateInfoKHR;
    const void* pNext = nullptr;
    Flags flags = 0;
    SurfaceKHR surface = 0;
    u32 minImageCount = 0;
    Format imageFormat = Format::Undefined;
    u32 imageColorSpace = 0;
    Extent2D imageExtent{};
    u32 imageArrayLayers = 1;
    Flags imageUsage = 0;
    SharingMode imageSharingMode = SharingMode::Exclusive;
    u32 queueFamilyIndexCount = 0;
    const u32* pQueueFamilyIndices = nullptr;
    Flags preTransform = 1;    // identity
    Flags compositeAlpha = 1;  // opaque
    u32 presentMode = 2;       // FIFO
    Bool32 clipped = kTrue;
    SwapchainKHR oldSwapchain = 0;
};
struct PresentInfoKHR
{
    StructureType sType = st::kPresentInfoKHR;
    const void* pNext = nullptr;
    u32 waitSemaphoreCount = 0;
    const Semaphore* pWaitSemaphores = nullptr;
    u32 swapchainCount = 0;
    const SwapchainKHR* pSwapchains = nullptr;
    const u32* pImageIndices = nullptr;
    Result* pResults = nullptr;
};
inline constexpr const char* kSurfaceExtensionName = "VK_KHR_surface";
inline constexpr const char* kWin32SurfaceExtensionName = "VK_KHR_win32_surface";
inline constexpr const char* kSwapchainExtensionName = "VK_KHR_swapchain";

// ---- Queries, pipeline caches and sparse binding ------------------------------

using QueryPool = u64;
inline constexpr u32 kQueryTypeTimestamp = 2;
inline constexpr Flags kQueryResult64 = 0x1;
inline constexpr Flags kQueryResultWait = 0x2;

struct QueryPoolCreateInfo
{
    StructureType sType = static_cast<StructureType>(11);
    const void* pNext = nullptr;
    Flags flags = 0;
    u32 queryType = kQueryTypeTimestamp;
    u32 queryCount = 0;
    Flags pipelineStatistics = 0;
};

struct PipelineCacheCreateInfo
{
    StructureType sType = static_cast<StructureType>(17);
    const void* pNext = nullptr;
    Flags flags = 0;
    usize initialDataSize = 0;
    const void* pInitialData = nullptr;
};

// VkImageCreateFlagBits for reserved images.
inline constexpr Flags kImageCreateSparseBinding = 0x1;
inline constexpr Flags kImageCreateSparseResidency = 0x2;
// VkSparseImageFormatFlagBits.
inline constexpr Flags kSparseImageFormatSingleMipTail = 0x1;

struct SparseImageFormatProperties
{
    Flags aspectMask = 0;
    Extent3D imageGranularity{};
    Flags flags = 0;
};

struct SparseImageMemoryRequirements
{
    SparseImageFormatProperties formatProperties{};
    u32 imageMipTailFirstLod = 0;
    DeviceSize imageMipTailSize = 0;
    DeviceSize imageMipTailOffset = 0;
    DeviceSize imageMipTailStride = 0;
};

struct ImageSubresource
{
    Flags aspectMask = 0;
    u32 mipLevel = 0;
    u32 arrayLayer = 0;
};

struct SparseMemoryBind
{
    DeviceSize resourceOffset = 0;
    DeviceSize size = 0;
    DeviceMemory memory = 0;
    DeviceSize memoryOffset = 0;
    Flags flags = 0;
};

struct SparseImageMemoryBind
{
    ImageSubresource subresource{};
    Offset3D offset{};
    Extent3D extent{};
    DeviceMemory memory = 0;
    DeviceSize memoryOffset = 0;
    Flags flags = 0;
};

struct SparseImageOpaqueMemoryBindInfo
{
    Image image = 0;
    u32 bindCount = 0;
    const SparseMemoryBind* pBinds = nullptr;
};

struct SparseImageMemoryBindInfo
{
    Image image = 0;
    u32 bindCount = 0;
    const SparseImageMemoryBind* pBinds = nullptr;
};

struct TimelineSemaphoreSubmitInfo
{
    StructureType sType = static_cast<StructureType>(1000207003);
    const void* pNext = nullptr;
    u32 waitSemaphoreValueCount = 0;
    const u64* pWaitSemaphoreValues = nullptr;
    u32 signalSemaphoreValueCount = 0;
    const u64* pSignalSemaphoreValues = nullptr;
};

struct BindSparseInfo
{
    StructureType sType = static_cast<StructureType>(7);
    const void* pNext = nullptr;
    u32 waitSemaphoreCount = 0;
    const Semaphore* pWaitSemaphores = nullptr;
    u32 bufferBindCount = 0;
    const void* pBufferBinds = nullptr;
    u32 imageOpaqueBindCount = 0;
    const SparseImageOpaqueMemoryBindInfo* pImageOpaqueBinds = nullptr;
    u32 imageBindCount = 0;
    const SparseImageMemoryBindInfo* pImageBinds = nullptr;
    u32 signalSemaphoreCount = 0;
    const Semaphore* pSignalSemaphores = nullptr;
};

// Function pointer types.
#define CYBER_VK_PFN(name, ret, ...) using PFN_##name = ret(CYBER_VKAPI*)(__VA_ARGS__)

CYBER_VK_PFN(CreateQueryPool, Result, Device, const QueryPoolCreateInfo*, const void*, QueryPool*);
CYBER_VK_PFN(DestroyQueryPool, void, Device, QueryPool, const void*);
CYBER_VK_PFN(ResetQueryPool, void, Device, QueryPool, u32, u32);
CYBER_VK_PFN(CmdResetQueryPool, void, CommandBuffer, QueryPool, u32, u32);
CYBER_VK_PFN(CmdWriteTimestamp2, void, CommandBuffer, Flags64, QueryPool, u32);
CYBER_VK_PFN(CmdCopyQueryPoolResults, void, CommandBuffer, QueryPool, u32, u32, Buffer, DeviceSize,
             DeviceSize, Flags);
CYBER_VK_PFN(CreatePipelineCache, Result, Device, const PipelineCacheCreateInfo*, const void*,
             PipelineCache*);
CYBER_VK_PFN(DestroyPipelineCache, void, Device, PipelineCache, const void*);
CYBER_VK_PFN(GetPipelineCacheData, Result, Device, PipelineCache, usize*, void*);
CYBER_VK_PFN(GetImageSparseMemoryRequirements, void, Device, Image, u32*,
             SparseImageMemoryRequirements*);
CYBER_VK_PFN(QueueBindSparse, Result, Queue, u32, const BindSparseInfo*, u64);

CYBER_VK_PFN(CreateImage, Result, Device, const ImageCreateInfo*, const void*, Image*);
CYBER_VK_PFN(DestroyImage, void, Device, Image, const void*);
CYBER_VK_PFN(GetImageMemoryRequirements, void, Device, Image, MemoryRequirements*);
CYBER_VK_PFN(BindImageMemory, Result, Device, Image, DeviceMemory, DeviceSize);
CYBER_VK_PFN(CreateImageView, Result, Device, const ImageViewCreateInfo*, const void*, ImageView*);
CYBER_VK_PFN(DestroyImageView, void, Device, ImageView, const void*);
CYBER_VK_PFN(CreateSampler, Result, Device, const SamplerCreateInfo*, const void*, Sampler*);
CYBER_VK_PFN(DestroySampler, void, Device, Sampler, const void*);

CYBER_VK_PFN(CreateCommandPool, Result, Device, const CommandPoolCreateInfo*, const void*,
             CommandPool*);
CYBER_VK_PFN(DestroyCommandPool, void, Device, CommandPool, const void*);
CYBER_VK_PFN(ResetCommandPool, Result, Device, CommandPool, Flags);
CYBER_VK_PFN(AllocateCommandBuffers, Result, Device, const CommandBufferAllocateInfo*,
             CommandBuffer*);
CYBER_VK_PFN(FreeCommandBuffers, void, Device, CommandPool, u32, const CommandBuffer*);
CYBER_VK_PFN(BeginCommandBuffer, Result, CommandBuffer, const CommandBufferBeginInfo*);
CYBER_VK_PFN(EndCommandBuffer, Result, CommandBuffer);
CYBER_VK_PFN(ResetCommandBuffer, Result, CommandBuffer, Flags);

CYBER_VK_PFN(CmdCopyBuffer, void, CommandBuffer, Buffer, Buffer, u32, const BufferCopy*);
CYBER_VK_PFN(CmdCopyBufferToImage, void, CommandBuffer, Buffer, Image, ImageLayout, u32,
             const BufferImageCopy*);
CYBER_VK_PFN(CmdCopyImageToBuffer, void, CommandBuffer, Image, ImageLayout, Buffer, u32,
             const BufferImageCopy*);
CYBER_VK_PFN(CmdFillBuffer, void, CommandBuffer, Buffer, DeviceSize, DeviceSize, u32);
CYBER_VK_PFN(CmdPipelineBarrier2, void, CommandBuffer, const DependencyInfo*);
CYBER_VK_PFN(CmdBindPipeline, void, CommandBuffer, u32, Pipeline);
CYBER_VK_PFN(CmdBindDescriptorSets, void, CommandBuffer, u32, PipelineLayout, u32, u32,
             const DescriptorSet*, u32, const u32*);
CYBER_VK_PFN(CmdPushConstants, void, CommandBuffer, PipelineLayout, Flags, u32, u32, const void*);
CYBER_VK_PFN(CmdDispatch, void, CommandBuffer, u32, u32, u32);
CYBER_VK_PFN(CmdDraw, void, CommandBuffer, u32, u32, u32, u32);
CYBER_VK_PFN(CmdBeginRendering, void, CommandBuffer, const RenderingInfo*);
CYBER_VK_PFN(CmdEndRendering, void, CommandBuffer);
CYBER_VK_PFN(CmdSetViewport, void, CommandBuffer, u32, u32, const Viewport*);
CYBER_VK_PFN(CmdSetScissor, void, CommandBuffer, u32, u32, const Rect2D*);

CYBER_VK_PFN(QueueSubmit2, Result, Queue, u32, const SubmitInfo2*, u64);
CYBER_VK_PFN(CreateSemaphore, Result, Device, const SemaphoreCreateInfo*, const void*, Semaphore*);
CYBER_VK_PFN(DestroySemaphore, void, Device, Semaphore, const void*);
CYBER_VK_PFN(WaitSemaphores, Result, Device, const SemaphoreWaitInfo*, u64);
CYBER_VK_PFN(GetSemaphoreCounterValue, Result, Device, Semaphore, u64*);
CYBER_VK_PFN(SignalSemaphore, Result, Device, const SemaphoreSignalInfo*);

CYBER_VK_PFN(CreateDescriptorSetLayout, Result, Device, const DescriptorSetLayoutCreateInfo*,
             const void*, DescriptorSetLayout*);
CYBER_VK_PFN(DestroyDescriptorSetLayout, void, Device, DescriptorSetLayout, const void*);
CYBER_VK_PFN(CreateDescriptorPool, Result, Device, const DescriptorPoolCreateInfo*, const void*,
             DescriptorPool*);
CYBER_VK_PFN(DestroyDescriptorPool, void, Device, DescriptorPool, const void*);
CYBER_VK_PFN(AllocateDescriptorSets, Result, Device, const DescriptorSetAllocateInfo*,
             DescriptorSet*);
CYBER_VK_PFN(UpdateDescriptorSets, void, Device, u32, const WriteDescriptorSet*, u32, const void*);
CYBER_VK_PFN(CreatePipelineLayout, Result, Device, const PipelineLayoutCreateInfo*, const void*,
             PipelineLayout*);
CYBER_VK_PFN(DestroyPipelineLayout, void, Device, PipelineLayout, const void*);
CYBER_VK_PFN(CreateShaderModule, Result, Device, const ShaderModuleCreateInfo*, const void*,
             ShaderModule*);
CYBER_VK_PFN(DestroyShaderModule, void, Device, ShaderModule, const void*);
CYBER_VK_PFN(CreateComputePipelines, Result, Device, PipelineCache, u32,
             const ComputePipelineCreateInfo*, const void*, Pipeline*);
CYBER_VK_PFN(CreateGraphicsPipelines, Result, Device, PipelineCache, u32,
             const GraphicsPipelineCreateInfo*, const void*, Pipeline*);
CYBER_VK_PFN(DestroyPipeline, void, Device, Pipeline, const void*);

CYBER_VK_PFN(CmdBindVertexBuffers, void, CommandBuffer, u32, u32, const Buffer*, const DeviceSize*);
CYBER_VK_PFN(CmdBindIndexBuffer, void, CommandBuffer, Buffer, DeviceSize, u32);
CYBER_VK_PFN(CmdDrawIndexed, void, CommandBuffer, u32, u32, u32, i32, u32);
CYBER_VK_PFN(CmdDrawIndirect, void, CommandBuffer, Buffer, DeviceSize, u32, u32);
CYBER_VK_PFN(CmdDrawIndexedIndirect, void, CommandBuffer, Buffer, DeviceSize, u32, u32);
CYBER_VK_PFN(CmdDrawIndirectCount, void, CommandBuffer, Buffer, DeviceSize, Buffer, DeviceSize, u32,
             u32);
CYBER_VK_PFN(CmdDrawIndexedIndirectCount, void, CommandBuffer, Buffer, DeviceSize, Buffer,
             DeviceSize, u32, u32);
CYBER_VK_PFN(CmdDispatchIndirect, void, CommandBuffer, Buffer, DeviceSize);
CYBER_VK_PFN(CmdCopyImage, void, CommandBuffer, Image, ImageLayout, Image, ImageLayout, u32,
             const ImageCopy*);
CYBER_VK_PFN(CmdClearColorImage, void, CommandBuffer, Image, ImageLayout, const ClearColorValue*,
             u32, const ImageSubresourceRange*);
CYBER_VK_PFN(CmdClearDepthStencilImage, void, CommandBuffer, Image, ImageLayout,
             const ClearDepthStencilValue*, u32, const ImageSubresourceRange*);
CYBER_VK_PFN(FreeDescriptorSets, Result, Device, DescriptorPool, u32, const DescriptorSet*);
// VK_EXT_debug_utils command-buffer labels, resolved only when the extension is on.
CYBER_VK_PFN(CmdBeginDebugUtilsLabelEXT, void, CommandBuffer, const DebugLabel*);
CYBER_VK_PFN(CmdEndDebugUtilsLabelEXT, void, CommandBuffer);
CYBER_VK_PFN(CmdInsertDebugUtilsLabelEXT, void, CommandBuffer, const DebugLabel*);

// Instance-level presentation entry points.
CYBER_VK_PFN(CreateWin32SurfaceKHR, Result, Instance, const Win32SurfaceCreateInfoKHR*, const void*,
             SurfaceKHR*);
CYBER_VK_PFN(DestroySurfaceKHR, void, Instance, SurfaceKHR, const void*);
CYBER_VK_PFN(GetPhysicalDeviceSurfaceSupportKHR, Result, PhysicalDevice, u32, SurfaceKHR, Bool32*);
CYBER_VK_PFN(GetPhysicalDeviceSurfaceCapabilitiesKHR, Result, PhysicalDevice, SurfaceKHR,
             SurfaceCapabilitiesKHR*);
CYBER_VK_PFN(GetPhysicalDeviceSurfaceFormatsKHR, Result, PhysicalDevice, SurfaceKHR, u32*,
             SurfaceFormatKHR*);
CYBER_VK_PFN(GetPhysicalDeviceSurfacePresentModesKHR, Result, PhysicalDevice, SurfaceKHR, u32*,
             u32*);
// Device-level presentation entry points.
CYBER_VK_PFN(CreateSwapchainKHR, Result, Device, const SwapchainCreateInfoKHR*, const void*,
             SwapchainKHR*);
CYBER_VK_PFN(DestroySwapchainKHR, void, Device, SwapchainKHR, const void*);
CYBER_VK_PFN(GetSwapchainImagesKHR, Result, Device, SwapchainKHR, u32*, Image*);
CYBER_VK_PFN(AcquireNextImageKHR, Result, Device, SwapchainKHR, u64, Semaphore, u64, u32*);
CYBER_VK_PFN(QueuePresentKHR, Result, Queue, const PresentInfoKHR*);

#undef CYBER_VK_PFN

// Every device entry point of this slice: member, command name, type. The context
// resolves them all through `vkGetDeviceProcAddr` with one macro, so a command
// cannot be declared and forgotten.
#define CYBER_VK_DEVICE_FUNCTIONS_EXT(X)                                                           \
    X(createImage, "vkCreateImage", PFN_CreateImage)                                               \
    X(destroyImage, "vkDestroyImage", PFN_DestroyImage)                                            \
    X(getImageMemoryRequirements, "vkGetImageMemoryRequirements", PFN_GetImageMemoryRequirements)  \
    X(bindImageMemory, "vkBindImageMemory", PFN_BindImageMemory)                                   \
    X(createImageView, "vkCreateImageView", PFN_CreateImageView)                                   \
    X(destroyImageView, "vkDestroyImageView", PFN_DestroyImageView)                                \
    X(createSampler, "vkCreateSampler", PFN_CreateSampler)                                         \
    X(destroySampler, "vkDestroySampler", PFN_DestroySampler)                                      \
    X(createCommandPool, "vkCreateCommandPool", PFN_CreateCommandPool)                             \
    X(destroyCommandPool, "vkDestroyCommandPool", PFN_DestroyCommandPool)                          \
    X(resetCommandPool, "vkResetCommandPool", PFN_ResetCommandPool)                                \
    X(allocateCommandBuffers, "vkAllocateCommandBuffers", PFN_AllocateCommandBuffers)              \
    X(freeCommandBuffers, "vkFreeCommandBuffers", PFN_FreeCommandBuffers)                          \
    X(beginCommandBuffer, "vkBeginCommandBuffer", PFN_BeginCommandBuffer)                          \
    X(endCommandBuffer, "vkEndCommandBuffer", PFN_EndCommandBuffer)                                \
    X(resetCommandBuffer, "vkResetCommandBuffer", PFN_ResetCommandBuffer)                          \
    X(cmdCopyBuffer, "vkCmdCopyBuffer", PFN_CmdCopyBuffer)                                         \
    X(cmdCopyBufferToImage, "vkCmdCopyBufferToImage", PFN_CmdCopyBufferToImage)                    \
    X(cmdCopyImageToBuffer, "vkCmdCopyImageToBuffer", PFN_CmdCopyImageToBuffer)                    \
    X(cmdFillBuffer, "vkCmdFillBuffer", PFN_CmdFillBuffer)                                         \
    X(cmdPipelineBarrier2, "vkCmdPipelineBarrier2", PFN_CmdPipelineBarrier2)                       \
    X(cmdBindPipeline, "vkCmdBindPipeline", PFN_CmdBindPipeline)                                   \
    X(cmdBindDescriptorSets, "vkCmdBindDescriptorSets", PFN_CmdBindDescriptorSets)                 \
    X(cmdPushConstants, "vkCmdPushConstants", PFN_CmdPushConstants)                                \
    X(cmdDispatch, "vkCmdDispatch", PFN_CmdDispatch)                                               \
    X(cmdDraw, "vkCmdDraw", PFN_CmdDraw)                                                           \
    X(cmdBeginRendering, "vkCmdBeginRendering", PFN_CmdBeginRendering)                             \
    X(cmdEndRendering, "vkCmdEndRendering", PFN_CmdEndRendering)                                   \
    X(cmdSetViewport, "vkCmdSetViewport", PFN_CmdSetViewport)                                      \
    X(cmdSetScissor, "vkCmdSetScissor", PFN_CmdSetScissor)                                         \
    X(queueSubmit2, "vkQueueSubmit2", PFN_QueueSubmit2)                                            \
    X(createSemaphore, "vkCreateSemaphore", PFN_CreateSemaphore)                                   \
    X(destroySemaphore, "vkDestroySemaphore", PFN_DestroySemaphore)                                \
    X(waitSemaphores, "vkWaitSemaphores", PFN_WaitSemaphores)                                      \
    X(getSemaphoreCounterValue, "vkGetSemaphoreCounterValue", PFN_GetSemaphoreCounterValue)        \
    X(signalSemaphore, "vkSignalSemaphore", PFN_SignalSemaphore)                                   \
    X(createDescriptorSetLayout, "vkCreateDescriptorSetLayout", PFN_CreateDescriptorSetLayout)     \
    X(destroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout", PFN_DestroyDescriptorSetLayout)  \
    X(createDescriptorPool, "vkCreateDescriptorPool", PFN_CreateDescriptorPool)                    \
    X(destroyDescriptorPool, "vkDestroyDescriptorPool", PFN_DestroyDescriptorPool)                 \
    X(allocateDescriptorSets, "vkAllocateDescriptorSets", PFN_AllocateDescriptorSets)              \
    X(updateDescriptorSets, "vkUpdateDescriptorSets", PFN_UpdateDescriptorSets)                    \
    X(createPipelineLayout, "vkCreatePipelineLayout", PFN_CreatePipelineLayout)                    \
    X(destroyPipelineLayout, "vkDestroyPipelineLayout", PFN_DestroyPipelineLayout)                 \
    X(createShaderModule, "vkCreateShaderModule", PFN_CreateShaderModule)                          \
    X(destroyShaderModule, "vkDestroyShaderModule", PFN_DestroyShaderModule)                       \
    X(createComputePipelines, "vkCreateComputePipelines", PFN_CreateComputePipelines)              \
    X(createGraphicsPipelines, "vkCreateGraphicsPipelines", PFN_CreateGraphicsPipelines)           \
    X(destroyPipeline, "vkDestroyPipeline", PFN_DestroyPipeline)                                   \
    X(cmdBindVertexBuffers, "vkCmdBindVertexBuffers", PFN_CmdBindVertexBuffers)                    \
    X(cmdBindIndexBuffer, "vkCmdBindIndexBuffer", PFN_CmdBindIndexBuffer)                          \
    X(cmdDrawIndexed, "vkCmdDrawIndexed", PFN_CmdDrawIndexed)                                      \
    X(cmdDrawIndirect, "vkCmdDrawIndirect", PFN_CmdDrawIndirect)                                   \
    X(cmdDrawIndexedIndirect, "vkCmdDrawIndexedIndirect", PFN_CmdDrawIndexedIndirect)              \
    X(cmdDrawIndirectCount, "vkCmdDrawIndirectCount", PFN_CmdDrawIndirectCount)                    \
    X(cmdDrawIndexedIndirectCount, "vkCmdDrawIndexedIndirectCount",                                \
      PFN_CmdDrawIndexedIndirectCount)                                                             \
    X(cmdDispatchIndirect, "vkCmdDispatchIndirect", PFN_CmdDispatchIndirect)                       \
    X(cmdCopyImage, "vkCmdCopyImage", PFN_CmdCopyImage)                                            \
    X(cmdClearColorImage, "vkCmdClearColorImage", PFN_CmdClearColorImage)                          \
    X(cmdClearDepthStencilImage, "vkCmdClearDepthStencilImage", PFN_CmdClearDepthStencilImage)     \
    X(freeDescriptorSets, "vkFreeDescriptorSets", PFN_FreeDescriptorSets)                          \
    X(createQueryPool, "vkCreateQueryPool", PFN_CreateQueryPool)                                   \
    X(destroyQueryPool, "vkDestroyQueryPool", PFN_DestroyQueryPool)                                \
    X(resetQueryPool, "vkResetQueryPool", PFN_ResetQueryPool)                                      \
    X(cmdResetQueryPool, "vkCmdResetQueryPool", PFN_CmdResetQueryPool)                             \
    X(cmdWriteTimestamp2, "vkCmdWriteTimestamp2", PFN_CmdWriteTimestamp2)                          \
    X(cmdCopyQueryPoolResults, "vkCmdCopyQueryPoolResults", PFN_CmdCopyQueryPoolResults)           \
    X(createPipelineCache, "vkCreatePipelineCache", PFN_CreatePipelineCache)                       \
    X(destroyPipelineCache, "vkDestroyPipelineCache", PFN_DestroyPipelineCache)                    \
    X(getPipelineCacheData, "vkGetPipelineCacheData", PFN_GetPipelineCacheData)                    \
    X(getImageSparseMemoryRequirements, "vkGetImageSparseMemoryRequirements",                      \
      PFN_GetImageSparseMemoryRequirements)                                                        \
    X(queueBindSparse, "vkQueueBindSparse", PFN_QueueBindSparse)

// Device entry points that exist only when VK_KHR_swapchain is enabled.
#define CYBER_VK_SWAPCHAIN_FUNCTIONS(X)                                                            \
    X(createSwapchain, "vkCreateSwapchainKHR", PFN_CreateSwapchainKHR)                             \
    X(destroySwapchain, "vkDestroySwapchainKHR", PFN_DestroySwapchainKHR)                          \
    X(getSwapchainImages, "vkGetSwapchainImagesKHR", PFN_GetSwapchainImagesKHR)                    \
    X(acquireNextImage, "vkAcquireNextImageKHR", PFN_AcquireNextImageKHR)                          \
    X(queuePresent, "vkQueuePresentKHR", PFN_QueuePresentKHR)

// Instance entry points that exist only when VK_KHR_surface is enabled.
#define CYBER_VK_SURFACE_FUNCTIONS(X)                                                              \
    X(createWin32Surface, "vkCreateWin32SurfaceKHR", PFN_CreateWin32SurfaceKHR)                    \
    X(destroySurface, "vkDestroySurfaceKHR", PFN_DestroySurfaceKHR)                                \
    X(getSurfaceSupport, "vkGetPhysicalDeviceSurfaceSupportKHR",                                   \
      PFN_GetPhysicalDeviceSurfaceSupportKHR)                                                      \
    X(getSurfaceCapabilities, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR",                         \
      PFN_GetPhysicalDeviceSurfaceCapabilitiesKHR)                                                 \
    X(getSurfaceFormats, "vkGetPhysicalDeviceSurfaceFormatsKHR",                                   \
      PFN_GetPhysicalDeviceSurfaceFormatsKHR)                                                      \
    X(getSurfacePresentModes, "vkGetPhysicalDeviceSurfacePresentModesKHR",                         \
      PFN_GetPhysicalDeviceSurfacePresentModesKHR)

#define CYBER_VK_DECLARE_MEMBER(member, name, type) type member = nullptr;

struct DeviceApiExt
{
    CYBER_VK_DEVICE_FUNCTIONS_EXT(CYBER_VK_DECLARE_MEMBER)
};
struct SwapchainApi
{
    CYBER_VK_SWAPCHAIN_FUNCTIONS(CYBER_VK_DECLARE_MEMBER)
};
struct SurfaceApi
{
    CYBER_VK_SURFACE_FUNCTIONS(CYBER_VK_DECLARE_MEMBER)
};

#undef CYBER_VK_DECLARE_MEMBER

}  // namespace cyber::rhi::detail::vk
