#pragma once

// Descriptor layouts, pipeline layouts, shader modules and pipelines for the
// Vulkan backend. Phase 34 increment 4 (ADR 0029, ADR 0015): the pipeline-binding
// model maps to one descriptor set layout per bind group, push constants for root
// constants, one update-after-bind bindless set, and dynamic rendering, so a
// graphics pipeline names attachment formats instead of a render pass.

#include "vulkan_context.h"

#include "vk_api_ext.h"

namespace cyber::rhi::detail
{

struct VulkanBindingDesc
{
    u32 binding = 0;
    vk::DescriptorType type = vk::DescriptorType::StorageBuffer;
    u32 count = 1;
    vk::Flags stages = 0;
    // `vk::DescriptorBindingFlag` bits. A binding with any of them makes its layout
    // an update-after-bind layout.
    vk::Flags bindingFlags = 0;
};

struct VulkanGraphicsPipelineDesc
{
    vk::PipelineLayout layout = 0;
    vk::ShaderModule vertex = 0;
    vk::ShaderModule fragment = 0;
    vk::Format colorFormat = vk::Format::R8G8B8A8Unorm;
    vk::Format depthFormat = vk::Format::Undefined;
    // Raw `VkCullModeFlags`: 0 none, 1 front, 2 back.
    vk::Flags cullMode = 0;
    bool depthTest = false;
    bool depthWrite = false;
    bool alphaBlend = false;
};

// Creates and destroys the objects above on one device. Holds no ownership: each
// `Create*` is paired with a `Destroy*` by the caller, like the memory allocator.
class VulkanPipelineFactory
{
public:
    explicit VulkanPipelineFactory(VulkanContext& context) noexcept : m_context(&context) {}

    [[nodiscard]] Status<Error> CreateShader(const u32* words, usize wordCount,
                                             vk::ShaderModule* out) noexcept;
    void DestroyShader(vk::ShaderModule module) noexcept;

    [[nodiscard]] Status<Error> CreateSetLayout(const VulkanBindingDesc* bindings, u32 count,
                                                vk::DescriptorSetLayout* out) noexcept;
    void DestroySetLayout(vk::DescriptorSetLayout layout) noexcept;

    [[nodiscard]] Status<Error> CreatePipelineLayout(const vk::DescriptorSetLayout* sets,
                                                     u32 setCount,
                                                     const vk::PushConstantRange* ranges,
                                                     u32 rangeCount,
                                                     vk::PipelineLayout* out) noexcept;
    void DestroyPipelineLayout(vk::PipelineLayout layout) noexcept;

    [[nodiscard]] Status<Error> CreateComputePipeline(vk::PipelineLayout layout,
                                                      vk::ShaderModule module,
                                                      vk::Pipeline* out) noexcept;
    [[nodiscard]] Status<Error> CreateGraphicsPipeline(const VulkanGraphicsPipelineDesc& desc,
                                                       vk::Pipeline* out) noexcept;
    void DestroyPipeline(vk::Pipeline pipeline) noexcept;

    [[nodiscard]] Status<Error> CreateDescriptorPool(u32 maxSets, const vk::DescriptorPoolSize* sizes,
                                                     u32 sizeCount, bool updateAfterBind,
                                                     vk::DescriptorPool* out) noexcept;
    void DestroyDescriptorPool(vk::DescriptorPool pool) noexcept;

    // Allocates one set. `variableCount` is the size of the last binding when that
    // binding is variable-count, and ignored (pass 0) otherwise.
    [[nodiscard]] Status<Error> AllocateSet(vk::DescriptorPool pool, vk::DescriptorSetLayout layout,
                                            u32 variableCount, vk::DescriptorSet* out) noexcept;

private:
    VulkanContext* m_context;
};

}  // namespace cyber::rhi::detail
