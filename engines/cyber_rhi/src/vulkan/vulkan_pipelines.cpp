#include "vulkan_pipelines.h"

namespace cyber::rhi::detail
{
namespace
{
using cyber::core::ErrorCode;

constexpr u32 kMaxBindings = 32;

[[nodiscard]] Status<Error> Fail(const char* message) noexcept
{
    return Status<Error>::Err(Error{ErrorCode::Internal, message});
}
[[nodiscard]] Status<Error> Invalid(const char* message) noexcept
{
    return Status<Error>::Err(Error{ErrorCode::InvalidArgument, message});
}
}  // namespace

Status<Error> VulkanPipelineFactory::CreateShader(const u32* words, usize wordCount,
                                                  vk::ShaderModule* out) noexcept
{
    if (words == nullptr || wordCount < 5 || out == nullptr || words[0] != 0x07230203U)
    {
        return Invalid("shader code is not a SPIR-V module");
    }
    vk::ShaderModuleCreateInfo info{};
    info.codeSize = wordCount * sizeof(u32);
    info.pCode = words;
    if (m_context->DeviceFunctionsExt().createShaderModule(m_context->Device(), &info, nullptr,
                                                           out) != vk::Result::Success)
    {
        return Fail("vkCreateShaderModule failed");
    }
    return Status<Error>::Ok();
}

void VulkanPipelineFactory::DestroyShader(vk::ShaderModule module) noexcept
{
    if (module != 0)
    {
        m_context->DeviceFunctionsExt().destroyShaderModule(m_context->Device(), module, nullptr);
    }
}

Status<Error> VulkanPipelineFactory::CreateSetLayout(const VulkanBindingDesc* bindings, u32 count,
                                                     vk::DescriptorSetLayout* out) noexcept
{
    if ((bindings == nullptr && count != 0) || count > kMaxBindings || out == nullptr)
    {
        return Invalid("descriptor set layout has too many or no bindings");
    }
    vk::DescriptorSetLayoutBinding native[kMaxBindings] = {};
    vk::Flags flags[kMaxBindings] = {};
    bool updateAfterBind = false;
    for (u32 index = 0; index < count; ++index)
    {
        native[index].binding = bindings[index].binding;
        native[index].descriptorType = bindings[index].type;
        native[index].descriptorCount = bindings[index].count;
        native[index].stageFlags = bindings[index].stages;
        flags[index] = bindings[index].bindingFlags;
        updateAfterBind =
            updateAfterBind || (flags[index] & vk::DescriptorBindingFlag::kUpdateAfterBind) != 0;
    }
    vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};
    flagsInfo.bindingCount = count;
    flagsInfo.pBindingFlags = flags;
    vk::DescriptorSetLayoutCreateInfo info{};
    info.pNext = &flagsInfo;
    info.flags = updateAfterBind ? vk::kDescriptorSetLayoutUpdateAfterBindPool : 0;
    info.bindingCount = count;
    info.pBindings = native;
    if (m_context->DeviceFunctionsExt().createDescriptorSetLayout(m_context->Device(), &info,
                                                                  nullptr, out) !=
        vk::Result::Success)
    {
        return Fail("vkCreateDescriptorSetLayout failed");
    }
    return Status<Error>::Ok();
}

void VulkanPipelineFactory::DestroySetLayout(vk::DescriptorSetLayout layout) noexcept
{
    if (layout != 0)
    {
        m_context->DeviceFunctionsExt().destroyDescriptorSetLayout(m_context->Device(), layout,
                                                                   nullptr);
    }
}

Status<Error> VulkanPipelineFactory::CreatePipelineLayout(const vk::DescriptorSetLayout* sets,
                                                          u32 setCount,
                                                          const vk::PushConstantRange* ranges,
                                                          u32 rangeCount,
                                                          vk::PipelineLayout* out) noexcept
{
    if (out == nullptr || (sets == nullptr && setCount != 0) ||
        (ranges == nullptr && rangeCount != 0))
    {
        return Invalid("pipeline layout arguments are inconsistent");
    }
    vk::PipelineLayoutCreateInfo info{};
    info.setLayoutCount = setCount;
    info.pSetLayouts = sets;
    info.pushConstantRangeCount = rangeCount;
    info.pPushConstantRanges = ranges;
    if (m_context->DeviceFunctionsExt().createPipelineLayout(m_context->Device(), &info, nullptr,
                                                             out) != vk::Result::Success)
    {
        return Fail("vkCreatePipelineLayout failed");
    }
    return Status<Error>::Ok();
}

void VulkanPipelineFactory::DestroyPipelineLayout(vk::PipelineLayout layout) noexcept
{
    if (layout != 0)
    {
        m_context->DeviceFunctionsExt().destroyPipelineLayout(m_context->Device(), layout, nullptr);
    }
}

Status<Error> VulkanPipelineFactory::CreateComputePipeline(vk::PipelineLayout layout,
                                                           vk::ShaderModule module,
                                                           vk::Pipeline* out) noexcept
{
    if (layout == 0 || module == 0 || out == nullptr)
    {
        return Invalid("compute pipeline needs a layout and a shader");
    }
    vk::ComputePipelineCreateInfo info{};
    info.stage.stage = vk::ShaderStage::kCompute;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = layout;
    if (m_context->DeviceFunctionsExt().createComputePipelines(m_context->Device(), 0, 1, &info,
                                                               nullptr, out) != vk::Result::Success)
    {
        return Fail("vkCreateComputePipelines failed");
    }
    return Status<Error>::Ok();
}

Status<Error> VulkanPipelineFactory::CreateGraphicsPipeline(const VulkanGraphicsPipelineDesc& desc,
                                                            vk::Pipeline* out) noexcept
{
    if (desc.layout == 0 || desc.vertex == 0 || out == nullptr)
    {
        return Invalid("graphics pipeline needs a layout and a vertex shader");
    }
    vk::PipelineShaderStageCreateInfo stages[2] = {};
    u32 stageCount = 0;
    stages[stageCount].stage = vk::ShaderStage::kVertex;
    stages[stageCount].module = desc.vertex;
    stages[stageCount].pName = "main";
    ++stageCount;
    if (desc.fragment != 0)
    {
        stages[stageCount].stage = vk::ShaderStage::kFragment;
        stages[stageCount].module = desc.fragment;
        stages[stageCount].pName = "main";
        ++stageCount;
    }

    // Vertex data is pulled from buffers by the shader for now, so there is no
    // fixed-function vertex input; the RHI's vertex layouts map here later.
    vk::PipelineVertexInputStateCreateInfo vertexInput{};
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    vk::PipelineViewportStateCreateInfo viewport{};
    vk::PipelineRasterizationStateCreateInfo raster{};
    raster.cullMode = desc.cullMode;
    vk::PipelineMultisampleStateCreateInfo multisample{};
    vk::PipelineDepthStencilStateCreateInfo depth{};
    depth.depthTestEnable = desc.depthTest ? vk::kTrue : vk::kFalse;
    depth.depthWriteEnable = desc.depthWrite ? vk::kTrue : vk::kFalse;
    // The renderer's depth is reverse-Z (ADR 0002), so nearer means greater.
    depth.depthCompareOp = vk::CompareOp::GreaterOrEqual;

    vk::PipelineColorBlendAttachmentState attachment{};
    if (desc.alphaBlend)
    {
        attachment.blendEnable = vk::kTrue;
        attachment.srcColorBlendFactor = 6;  // source alpha
        attachment.dstColorBlendFactor = 7;  // one minus source alpha
        attachment.srcAlphaBlendFactor = 1;  // one
        attachment.dstAlphaBlendFactor = 7;
    }
    vk::PipelineColorBlendStateCreateInfo blend{};
    blend.attachmentCount = desc.colorFormat != vk::Format::Undefined ? 1 : 0;
    blend.pAttachments = &attachment;

    const u32 dynamicStates[2] = {vk::kDynamicStateViewport, vk::kDynamicStateScissor};
    vk::PipelineDynamicStateCreateInfo dynamic{};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;

    // Dynamic rendering: attachment formats live on the pipeline, there is no
    // render pass object.
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.colorAttachmentCount = blend.attachmentCount;
    rendering.pColorAttachmentFormats = &desc.colorFormat;
    rendering.depthAttachmentFormat = desc.depthFormat;

    vk::GraphicsPipelineCreateInfo info{};
    info.pNext = &rendering;
    info.stageCount = stageCount;
    info.pStages = stages;
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &inputAssembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = desc.layout;
    if (m_context->DeviceFunctionsExt().createGraphicsPipelines(m_context->Device(), 0, 1, &info,
                                                                nullptr, out) !=
        vk::Result::Success)
    {
        return Fail("vkCreateGraphicsPipelines failed");
    }
    return Status<Error>::Ok();
}

void VulkanPipelineFactory::DestroyPipeline(vk::Pipeline pipeline) noexcept
{
    if (pipeline != 0)
    {
        m_context->DeviceFunctionsExt().destroyPipeline(m_context->Device(), pipeline, nullptr);
    }
}

Status<Error> VulkanPipelineFactory::CreateDescriptorPool(u32 maxSets,
                                                          const vk::DescriptorPoolSize* sizes,
                                                          u32 sizeCount, bool updateAfterBind,
                                                          vk::DescriptorPool* out) noexcept
{
    if (maxSets == 0 || sizes == nullptr || sizeCount == 0 || out == nullptr)
    {
        return Invalid("descriptor pool needs sets and sizes");
    }
    vk::DescriptorPoolCreateInfo info{};
    info.flags = vk::kDescriptorPoolFreeDescriptorSet |
                 (updateAfterBind ? vk::kDescriptorPoolUpdateAfterBind : 0);
    info.maxSets = maxSets;
    info.poolSizeCount = sizeCount;
    info.pPoolSizes = sizes;
    if (m_context->DeviceFunctionsExt().createDescriptorPool(m_context->Device(), &info, nullptr,
                                                             out) != vk::Result::Success)
    {
        return Fail("vkCreateDescriptorPool failed");
    }
    return Status<Error>::Ok();
}

void VulkanPipelineFactory::DestroyDescriptorPool(vk::DescriptorPool pool) noexcept
{
    if (pool != 0)
    {
        m_context->DeviceFunctionsExt().destroyDescriptorPool(m_context->Device(), pool, nullptr);
    }
}

Status<Error> VulkanPipelineFactory::AllocateSet(vk::DescriptorPool pool,
                                                 vk::DescriptorSetLayout layout, u32 variableCount,
                                                 vk::DescriptorSet* out) noexcept
{
    if (pool == 0 || layout == 0 || out == nullptr)
    {
        return Invalid("descriptor set allocation needs a pool and a layout");
    }
    vk::DescriptorSetVariableDescriptorCountAllocateInfo variable{};
    variable.descriptorSetCount = 1;
    variable.pDescriptorCounts = &variableCount;
    vk::DescriptorSetAllocateInfo info{};
    info.pNext = variableCount != 0 ? &variable : nullptr;
    info.descriptorPool = pool;
    info.pSetLayouts = &layout;
    if (m_context->DeviceFunctionsExt().allocateDescriptorSets(m_context->Device(), &info, out) !=
        vk::Result::Success)
    {
        return Fail("vkAllocateDescriptorSets failed");
    }
    return Status<Error>::Ok();
}

}  // namespace cyber::rhi::detail
