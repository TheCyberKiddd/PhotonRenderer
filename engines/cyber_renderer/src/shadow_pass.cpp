#include <cyber/renderer/shadow_pass.h>
#include <cyber/rhi/command_list.h>
#include <cyber/rhi/resource_state.h>

namespace cyber::renderer
{

namespace
{

Status<Error> SetupShadowDepthPass(RenderPassBuilder& builder, void* userData) noexcept
{
    const auto& desc = *static_cast<const ShadowDepthPassDesc*>(userData);
    return builder.Write(desc.depthResource, cyber::rhi::ResourceState::DepthWrite);
}

Status<Error> RecordShadowDepthPass(RenderPassContext& context, void* userData) noexcept
{
    const auto& desc = *static_cast<const ShadowDepthPassDesc*>(userData);
    if (context.GetTexture(desc.depthResource) != desc.depthTexture)
    {
        return Status<Error>::Err(
            Error{cyber::core::ErrorCode::InvalidArgument,
                  "shadow depth pass resource does not match its depth texture"});
    }

    cyber::rhi::DepthStencilAttachment attachment{};
    attachment.view = desc.depthView;
    attachment.depthLoadOp = cyber::rhi::LoadOp::Clear;
    attachment.depthStoreOp = cyber::rhi::StoreOp::Store;
    attachment.clearValue = cyber::rhi::ClearDepthStencilValue::Of(0.0f);

    cyber::rhi::RenderPassDesc pass{};
    pass.hasDepthStencil = true;
    pass.depthStencil = attachment;
    pass.label = desc.name;

    const Status<Error> begun = context.Commands().BeginRenderPass(pass);
    if (!begun)
    {
        return begun;
    }

    const Status<Error> recorded = desc.record(context, desc.userData);
    const Status<Error> ended = context.Commands().EndRenderPass();
    return recorded ? ended : recorded;
}

Status<Error> SetupShadowSamplingPass(RenderPassBuilder& builder, void* userData) noexcept
{
    const auto& desc = *static_cast<const ShadowSamplingPassDesc*>(userData);
    for (cyber::core::usize index = 0; index < desc.shadowResources.Size(); ++index)
    {
        const Status<Error> read =
            builder.Read(desc.shadowResources[index], cyber::rhi::ResourceState::ShaderRead);
        if (!read)
        {
            return read;
        }
    }
    return Status<Error>::Ok();
}

Status<Error> RecordShadowSamplingPass(RenderPassContext& context, void* userData) noexcept
{
    const auto& desc = *static_cast<const ShadowSamplingPassDesc*>(userData);
    return desc.record(context, desc.userData);
}

}  // namespace

Status<Error> CreateShadowMapResources(cyber::rhi::Device& device, cyber::core::u32 width,
                                       cyber::core::u32 height, ShadowMapResources& output) noexcept
{
    if (!device.IsValid() || width == 0 || height == 0 || output.depthTexture.IsValid() ||
        output.depthView.IsValid() || output.sampledView.IsValid())
    {
        return Status<Error>::Err(Error{cyber::core::ErrorCode::InvalidArgument,
                                        "invalid shadow map resource description"});
    }

    cyber::rhi::TextureDesc textureDesc{};
    textureDesc.format = cyber::rhi::Format::D32Float;
    textureDesc.usage = cyber::rhi::TextureUsage::DepthStencil | cyber::rhi::TextureUsage::Sampled |
                        cyber::rhi::TextureUsage::CopySource;
    textureDesc.width = width;
    textureDesc.height = height;
    textureDesc.label = "CyberGraphics shadow depth";
    auto texture = device.CreateTexture(textureDesc);
    if (!texture)
    {
        return Status<Error>::Err(texture.Error());
    }

    cyber::rhi::TextureViewDesc viewDesc{};
    viewDesc.texture = texture.Value();
    viewDesc.format = cyber::rhi::Format::D32Float;
    viewDesc.range.aspect = cyber::rhi::TextureAspect::Depth;
    viewDesc.label = "CyberGraphics shadow depth view";
    auto view = device.CreateTextureView(viewDesc);
    if (!view)
    {
        const Status<Error> destroyed = device.DestroyTexture(texture.Value());
        return destroyed ? Status<Error>::Err(view.Error()) : destroyed;
    }

    cyber::rhi::TextureViewDesc sampledViewDesc = viewDesc;
    sampledViewDesc.usage = cyber::rhi::TextureViewUsage::Sampled;
    sampledViewDesc.label = "CyberGraphics sampled shadow depth view";
    auto sampledView = device.CreateTextureView(sampledViewDesc);
    if (!sampledView)
    {
        const Status<Error> destroyedView = device.DestroyTextureView(view.Value());
        if (!destroyedView)
        {
            return destroyedView;
        }
        const Status<Error> destroyedTexture = device.DestroyTexture(texture.Value());
        return destroyedTexture ? Status<Error>::Err(sampledView.Error()) : destroyedTexture;
    }

    output.depthTexture = texture.Value();
    output.depthView = view.Value();
    output.sampledView = sampledView.Value();
    output.width = width;
    output.height = height;
    return Status<Error>::Ok();
}

Result<GraphResourceHandle, Error> ImportShadowMap(RenderGraph& graph, const char* name,
                                                   const ShadowMapResources& resources) noexcept
{
    if (!resources.depthTexture.IsValid() || !resources.depthView.IsValid() ||
        resources.width == 0 || resources.height == 0)
    {
        return Result<GraphResourceHandle, Error>::Err(
            Error{cyber::core::ErrorCode::InvalidArgument, "invalid shadow map resources"});
    }
    return graph.ImportTexture(name, resources.depthTexture);
}

Status<Error> DestroyShadowMapResources(cyber::rhi::Device& device,
                                        ShadowMapResources& resources) noexcept
{
    if (resources.sampledView.IsValid())
    {
        const Status<Error> destroyed = device.DestroyTextureView(resources.sampledView);
        if (!destroyed)
        {
            return destroyed;
        }
        resources.sampledView = {};
    }
    if (resources.depthView.IsValid())
    {
        const Status<Error> destroyed = device.DestroyTextureView(resources.depthView);
        if (!destroyed)
        {
            return destroyed;
        }
        resources.depthView = {};
    }
    if (resources.depthTexture.IsValid())
    {
        const Status<Error> destroyed = device.DestroyTexture(resources.depthTexture);
        if (!destroyed)
        {
            return destroyed;
        }
        resources.depthTexture = {};
    }
    resources.width = 0;
    resources.height = 0;
    return Status<Error>::Ok();
}

Status<Error> AddShadowDepthPass(RenderGraph& graph, ShadowDepthPassDesc& desc) noexcept
{
    if (desc.name == nullptr || desc.name[0] == '\0' || !desc.depthResource.IsValid() ||
        !desc.depthTexture.IsValid() || !desc.depthView.IsValid() || desc.record == nullptr)
    {
        return Status<Error>::Err(Error{cyber::core::ErrorCode::InvalidArgument,
                                        "invalid shadow depth pass description"});
    }

    RenderPassDesc pass{};
    pass.name = desc.name;
    pass.setup = &SetupShadowDepthPass;
    pass.execute = &RecordShadowDepthPass;
    pass.userData = &desc;
    pass.sideEffect = true;
    return graph.AddPass(pass);
}

Status<Error> AddShadowSamplingPass(RenderGraph& graph, ShadowSamplingPassDesc& desc) noexcept
{
    if (desc.name == nullptr || desc.name[0] == '\0' || desc.record == nullptr ||
        desc.shadowResources.Empty() || desc.shadowResources.Data() == nullptr)
    {
        return Status<Error>::Err(Error{cyber::core::ErrorCode::InvalidArgument,
                                        "invalid shadow sampling pass description"});
    }
    for (cyber::core::usize index = 0; index < desc.shadowResources.Size(); ++index)
    {
        if (!desc.shadowResources[index].IsValid())
        {
            return Status<Error>::Err(
                Error{cyber::core::ErrorCode::InvalidArgument,
                      "shadow sampling pass contains an invalid graph resource"});
        }
    }

    RenderPassDesc pass{};
    pass.name = desc.name;
    pass.setup = &SetupShadowSamplingPass;
    pass.execute = &RecordShadowSamplingPass;
    pass.userData = &desc;
    pass.sideEffect = true;
    return graph.AddPass(pass);
}

cyber::rhi::SamplerDesc MakeReverseZShadowSamplerDesc() noexcept
{
    cyber::rhi::SamplerDesc desc{};
    desc.minFilter = cyber::rhi::FilterMode::Linear;
    desc.magFilter = cyber::rhi::FilterMode::Linear;
    desc.mipmapFilter = cyber::rhi::FilterMode::Nearest;
    desc.addressU = cyber::rhi::AddressMode::ClampToBorder;
    desc.addressV = cyber::rhi::AddressMode::ClampToBorder;
    desc.addressW = cyber::rhi::AddressMode::ClampToBorder;
    desc.borderColor = cyber::rhi::BorderColor::OpaqueBlack;
    desc.maxLod = 0.0f;
    desc.compareEnabled = true;
    desc.compareFunction = cyber::rhi::CompareFunction::GreaterEqual;
    desc.label = "Reverse-Z shadow comparison";
    return desc;
}

}  // namespace cyber::renderer
