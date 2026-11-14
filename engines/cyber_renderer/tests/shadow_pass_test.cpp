#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/renderer/shadow_pass.h>
#include <cyber/rhi/backend.h>
#include <cyber/rhi/command_list.h>
#include <cyber/rhi/d3d12.h>
#include <cyber/rhi/device.h>
#include <cyber/rhi/resource_state.h>
#include <cyber/rhi/submission.h>
#include <cyber/test/test.h>

#include <array>
#include <cstring>
#include <utility>

namespace
{

struct Fixture
{
    cyber::memory::HeapAllocator heap;
    cyber::rhi::Device device;

    bool Open() noexcept
    {
        cyber::rhi::DeviceDesc desc{};
        desc.backend = cyber::rhi::NullBackend();
        auto created = cyber::rhi::Device::Create(desc, heap);
        if (!created)
        {
            return false;
        }
        device = std::move(created.Value());
        return true;
    }

    bool BeginList(cyber::rhi::CommandList& list) noexcept
    {
        cyber::rhi::CommandListDesc desc{};
        auto created = device.CreateCommandList(desc);
        if (!created)
        {
            return false;
        }
        list = std::move(created.Value());
        return list.Begin().IsOk();
    }
};

struct DrawState
{
    cyber::core::u32 calls = 0;
};

cyber::core::Status<cyber::core::Error> RecordShadowGeometry(cyber::renderer::RenderPassContext&,
                                                             void* userData) noexcept
{
    ++static_cast<DrawState*>(userData)->calls;
    return cyber::core::Status<cyber::core::Error>::Ok();
}

struct ShadowSamplingProbe
{
    const cyber::renderer::GraphResourceHandle* graphResources = nullptr;
    cyber::rhi::TextureHandle expectedTextures[2]{};
    cyber::core::u32 calls = 0;
};

cyber::core::Status<cyber::core::Error>
RecordShadowSamplingProbe(cyber::renderer::RenderPassContext& context, void* userData) noexcept
{
    auto& probe = *static_cast<ShadowSamplingProbe*>(userData);
    for (cyber::core::u32 index = 0; index < 2; ++index)
    {
        if (context.GetTexture(probe.graphResources[index]) != probe.expectedTextures[index])
        {
            return cyber::core::Status<cyber::core::Error>::Err(
                cyber::core::Error{cyber::core::ErrorCode::Internal,
                                   "shadow sampling pass resolved the wrong resource"});
        }
    }
    ++probe.calls;
    return cyber::core::Status<cyber::core::Error>::Ok();
}

struct GpuShadowDraw
{
    cyber::rhi::PipelineHandle pipeline{};
    cyber::rhi::BufferHandle vertices{};
    cyber::core::u32 resolution = 0;
};

struct GpuShadowedLightingDraw
{
    cyber::rhi::PipelineHandle pipeline{};
    cyber::rhi::BindGroupHandle shadowBindings{};
    cyber::rhi::BufferHandle vertices{};
    cyber::rhi::TextureHandle shadowTexture{};
    cyber::renderer::GraphResourceHandle shadowResource{};
    cyber::rhi::TextureHandle colorTexture{};
    cyber::rhi::TextureViewHandle colorView{};
    cyber::rhi::BufferHandle readback{};
    cyber::core::u32 resolution = 0;
};

cyber::core::Status<cyber::core::Error>
RecordGpuShadowGeometry(cyber::renderer::RenderPassContext& context, void* userData) noexcept
{
    const auto& draw = *static_cast<const GpuShadowDraw*>(userData);
    auto& commands = context.Commands();
    const float extent = static_cast<float>(draw.resolution);
    if (!commands.SetViewport(cyber::rhi::Viewport{0.0f, 0.0f, extent, extent, 0.0f, 1.0f}) ||
        !commands.SetScissor(cyber::rhi::ScissorRect{0, 0, draw.resolution, draw.resolution}) ||
        !commands.SetPipeline(draw.pipeline) || !commands.SetVertexBuffer(0, draw.vertices) ||
        !commands.Draw(cyber::rhi::DrawArguments{3, 1, 0, 0}))
    {
        return cyber::core::Status<cyber::core::Error>::Err(cyber::core::Error{
            cyber::core::ErrorCode::Internal, "failed to record GPU shadow geometry"});
    }
    return cyber::core::Status<cyber::core::Error>::Ok();
}

cyber::core::Status<cyber::core::Error>
RecordGpuShadowedLighting(cyber::renderer::RenderPassContext& context, void* userData) noexcept
{
    using namespace cyber::rhi;
    const auto& draw = *static_cast<const GpuShadowedLightingDraw*>(userData);
    if (context.GetTexture(draw.shadowResource) != draw.shadowTexture)
    {
        return cyber::core::Status<cyber::core::Error>::Err(
            cyber::core::Error{cyber::core::ErrorCode::InvalidArgument,
                               "shadowed lighting received the wrong shadow map"});
    }

    auto& commands = context.Commands();
    const TextureBarrier targetReady{draw.colorTexture, SubresourceRange{},
                                     ResourceState::Undefined, ResourceState::RenderTarget};
    const BufferBarrier readbackReady{draw.readback, ResourceState::Undefined,
                                      ResourceState::CopyDestination};
    const Status<cyber::core::Error> barriers =
        commands.Barrier(cyber::memory::Span<const BufferBarrier>(&readbackReady, 1),
                         cyber::memory::Span<const TextureBarrier>(&targetReady, 1));
    if (!barriers)
    {
        return barriers;
    }

    ColorAttachment attachment{};
    attachment.view = draw.colorView;
    attachment.loadOp = LoadOp::Clear;
    attachment.storeOp = StoreOp::Store;
    attachment.clearValue = ClearColorValue::Of(0.0f, 0.0f, 0.0f, 1.0f);
    RenderPassDesc pass{};
    pass.colorAttachments = cyber::memory::Span<const ColorAttachment>(&attachment, 1);
    const Status<cyber::core::Error> begun = commands.BeginRenderPass(pass);
    if (!begun)
    {
        return begun;
    }
    const float extent = static_cast<float>(draw.resolution);
    const Status<cyber::core::Error> viewport =
        commands.SetViewport(Viewport{0.0f, 0.0f, extent, extent, 0.0f, 1.0f});
    const Status<cyber::core::Error> scissor =
        commands.SetScissor(ScissorRect{0, 0, draw.resolution, draw.resolution});
    const Status<cyber::core::Error> pipeline = commands.SetPipeline(draw.pipeline);
    const Status<cyber::core::Error> group =
        commands.SetBindGroup(BindGroupSlot::PerView, draw.shadowBindings);
    const Status<cyber::core::Error> vertexBuffer = commands.SetVertexBuffer(0, draw.vertices);
    const Status<cyber::core::Error> drawn = commands.Draw(DrawArguments{3, 1, 0, 0});
    const Status<cyber::core::Error> ended = commands.EndRenderPass();
    if (!viewport)
    {
        return viewport;
    }
    if (!scissor)
    {
        return scissor;
    }
    if (!pipeline)
    {
        return pipeline;
    }
    if (!group)
    {
        return group;
    }
    if (!vertexBuffer)
    {
        return vertexBuffer;
    }
    if (!drawn)
    {
        return drawn;
    }
    if (!ended)
    {
        return ended;
    }

    const TextureBarrier colorReady{draw.colorTexture, SubresourceRange{},
                                    ResourceState::RenderTarget, ResourceState::CopySource};
    const Status<cyber::core::Error> copyBarrier =
        commands.Barrier(cyber::memory::Span<const BufferBarrier>{},
                         cyber::memory::Span<const TextureBarrier>(&colorReady, 1));
    if (!copyBarrier)
    {
        return copyBarrier;
    }
    BufferTextureCopy copy{};
    copy.buffer = draw.readback;
    copy.bufferRowPitch = draw.resolution * 4;
    copy.bufferDepthPitch = draw.resolution * draw.resolution * 4;
    copy.texture = draw.colorTexture;
    copy.width = draw.resolution;
    copy.height = draw.resolution;
    return commands.CopyTextureToBuffer(copy);
}

cyber::memory::Span<const cyber::memory::Byte> ShaderBytes(const char* source) noexcept
{
    return cyber::memory::Span<const cyber::memory::Byte>(
        reinterpret_cast<const cyber::memory::Byte*>(source), std::strlen(source));
}

}  // namespace

CG_TEST(renderer_shadow_pass_records_depth_only_reverse_z_work)
{
    using namespace cyber::renderer;
    using namespace cyber::rhi;

    Fixture fixture;
    CG_REQUIRE(fixture.Open());

    ShadowMapResources resources{};
    CG_EXPECT_FALSE(static_cast<bool>(CreateShadowMapResources(fixture.device, 0, 64, resources)));
    CG_REQUIRE(CreateShadowMapResources(fixture.device, 64, 64, resources).IsOk());
    const TextureInfo textureInfo = fixture.device.TextureInfoOf(resources.depthTexture);
    CG_EXPECT_TRUE(textureInfo.valid);
    CG_EXPECT_EQ(textureInfo.format, Format::D32Float);
    CG_EXPECT_TRUE(HasUsage(textureInfo.usage, TextureUsage::DepthStencil));
    CG_EXPECT_TRUE(HasUsage(textureInfo.usage, TextureUsage::Sampled));
    CG_EXPECT_TRUE(fixture.device.TextureViewInfoOf(resources.depthView).valid);
    CG_EXPECT_EQ(fixture.device.TextureViewInfoOf(resources.depthView).usage,
                 TextureViewUsage::Automatic);
    CG_EXPECT_TRUE(fixture.device.TextureViewInfoOf(resources.sampledView).valid);
    CG_EXPECT_EQ(fixture.device.TextureViewInfoOf(resources.sampledView).usage,
                 TextureViewUsage::Sampled);

    RenderGraph graph(fixture.device, fixture.heap);
    auto depth = ImportShadowMap(graph, "shadow_depth", resources);
    CG_REQUIRE(static_cast<bool>(depth));

    DrawState drawState{};
    ShadowDepthPassDesc pass{};
    pass.name = "Directional shadow map";
    pass.depthResource = depth.Value();
    pass.depthTexture = resources.depthTexture;
    pass.depthView = resources.depthView;
    pass.record = &RecordShadowGeometry;
    pass.userData = &drawState;

    ShadowDepthPassDesc invalidPass = pass;
    invalidPass.record = nullptr;
    CG_EXPECT_FALSE(static_cast<bool>(AddShadowDepthPass(graph, invalidPass)));
    CG_REQUIRE(static_cast<bool>(AddShadowDepthPass(graph, pass)));
    CG_REQUIRE(graph.Compile().IsOk());

    CommandList commandList;
    CG_REQUIRE(fixture.BeginList(commandList));
    CG_REQUIRE(graph.Execute(commandList).IsOk());
    CG_EXPECT_EQ(drawState.calls, cyber::core::u32{1});
    CG_EXPECT_EQ(commandList.Stats().renderPasses, cyber::core::u32{1});
    CG_EXPECT_EQ(commandList.Stats().barriers, cyber::core::u32{1});
    CG_REQUIRE(commandList.End().IsOk());

    const CommandListHandle listHandle = commandList.Handle();
    SubmitDesc submit{};
    submit.queue = QueueKind::Graphics;
    submit.commandLists = cyber::memory::Span<const CommandListHandle>(&listHandle, 1);
    CG_REQUIRE(fixture.device.Submit(submit).IsOk());
    CG_REQUIRE(graph.AcknowledgeSubmission().IsOk());
    CG_EXPECT_EQ(fixture.device.TextureStateOf(resources.depthTexture, 0, 0),
                 ResourceState::DepthWrite);
    CG_REQUIRE(graph.Destroy().IsOk());
    CG_REQUIRE(DestroyShadowMapResources(fixture.device, resources).IsOk());
    CG_EXPECT_FALSE(resources.depthTexture.IsValid());
    CG_EXPECT_FALSE(resources.depthView.IsValid());
    CG_EXPECT_FALSE(resources.sampledView.IsValid());
}

CG_TEST(renderer_shadow_sampling_pass_transitions_all_shadow_maps)
{
    using namespace cyber::renderer;
    using namespace cyber::rhi;

    Fixture fixture;
    CG_REQUIRE(fixture.Open());
    ShadowMapResources maps[2]{};
    CG_REQUIRE(CreateShadowMapResources(fixture.device, 32, 32, maps[0]).IsOk());
    CG_REQUIRE(CreateShadowMapResources(fixture.device, 32, 32, maps[1]).IsOk());

    RenderGraph graph(fixture.device, fixture.heap);
    GraphResourceHandle resources[2]{};
    auto first = ImportShadowMap(graph, "cascade_near", maps[0]);
    auto second = ImportShadowMap(graph, "cascade_far", maps[1]);
    CG_REQUIRE(static_cast<bool>(first));
    CG_REQUIRE(static_cast<bool>(second));
    resources[0] = first.Value();
    resources[1] = second.Value();

    DrawState draws[2]{};
    ShadowDepthPassDesc depthPasses[2]{};
    depthPasses[0].name = "Render near cascade";
    depthPasses[0].depthResource = resources[0];
    depthPasses[0].depthTexture = maps[0].depthTexture;
    depthPasses[0].depthView = maps[0].depthView;
    depthPasses[0].record = &RecordShadowGeometry;
    depthPasses[0].userData = &draws[0];
    depthPasses[1].name = "Render far cascade";
    depthPasses[1].depthResource = resources[1];
    depthPasses[1].depthTexture = maps[1].depthTexture;
    depthPasses[1].depthView = maps[1].depthView;
    depthPasses[1].record = &RecordShadowGeometry;
    depthPasses[1].userData = &draws[1];
    CG_REQUIRE(AddShadowDepthPass(graph, depthPasses[0]).IsOk());
    CG_REQUIRE(AddShadowDepthPass(graph, depthPasses[1]).IsOk());

    ShadowSamplingProbe probe{};
    probe.graphResources = resources;
    probe.expectedTextures[0] = maps[0].depthTexture;
    probe.expectedTextures[1] = maps[1].depthTexture;
    ShadowSamplingPassDesc samplingPass{};
    samplingPass.name = "Sample directional cascades";
    samplingPass.shadowResources = cyber::memory::Span<const GraphResourceHandle>(resources, 2);
    samplingPass.record = &RecordShadowSamplingProbe;
    samplingPass.userData = &probe;
    CG_REQUIRE(AddShadowSamplingPass(graph, samplingPass).IsOk());

    GraphResourceHandle invalidResource{};
    ShadowSamplingPassDesc invalidSamplingPass{};
    invalidSamplingPass.shadowResources =
        cyber::memory::Span<const GraphResourceHandle>(&invalidResource, 1);
    invalidSamplingPass.record = &RecordShadowSamplingProbe;
    invalidSamplingPass.userData = &probe;
    CG_EXPECT_FALSE(AddShadowSamplingPass(graph, invalidSamplingPass).IsOk());
    CG_REQUIRE(graph.Compile().IsOk());

    cyber::rhi::CommandList commandList;
    CG_REQUIRE(fixture.BeginList(commandList));
    CG_REQUIRE(graph.Execute(commandList).IsOk());
    CG_EXPECT_EQ(draws[0].calls, cyber::core::u32{1});
    CG_EXPECT_EQ(draws[1].calls, cyber::core::u32{1});
    CG_EXPECT_EQ(probe.calls, cyber::core::u32{1});
    CG_EXPECT_EQ(commandList.Stats().renderPasses, cyber::core::u32{2});
    CG_REQUIRE(commandList.End().IsOk());
    const CommandListHandle listHandle = commandList.Handle();
    SubmitDesc submit{};
    submit.queue = QueueKind::Graphics;
    submit.commandLists = cyber::memory::Span<const CommandListHandle>(&listHandle, 1);
    CG_REQUIRE(fixture.device.Submit(submit).IsOk());
    CG_REQUIRE(graph.AcknowledgeSubmission().IsOk());
    CG_EXPECT_EQ(fixture.device.TextureStateOf(maps[0].depthTexture, 0, 0),
                 ResourceState::ShaderRead);
    CG_EXPECT_EQ(fixture.device.TextureStateOf(maps[1].depthTexture, 0, 0),
                 ResourceState::ShaderRead);

    CG_REQUIRE(graph.Destroy().IsOk());
    CG_REQUIRE(DestroyShadowMapResources(fixture.device, maps[0]).IsOk());
    CG_REQUIRE(DestroyShadowMapResources(fixture.device, maps[1]).IsOk());
}

CG_TEST(renderer_shadow_pass_uses_reverse_z_comparison_sampler)
{
    const cyber::rhi::SamplerDesc desc = cyber::renderer::MakeReverseZShadowSamplerDesc();
    CG_EXPECT_TRUE(desc.compareEnabled);
    CG_EXPECT_EQ(desc.compareFunction, cyber::rhi::CompareFunction::GreaterEqual);
    CG_EXPECT_EQ(desc.addressU, cyber::rhi::AddressMode::ClampToBorder);
    CG_EXPECT_EQ(desc.addressV, cyber::rhi::AddressMode::ClampToBorder);
    CG_EXPECT_EQ(desc.borderColor, cyber::rhi::BorderColor::OpaqueBlack);
    CG_EXPECT_EQ(desc.maxLod, 0.0f);

    cyber::memory::HeapAllocator heap;
    cyber::rhi::DeviceDesc deviceDesc{};
    deviceDesc.backend = cyber::rhi::NullBackend();
    auto device = cyber::rhi::Device::Create(deviceDesc, heap);
    CG_REQUIRE(static_cast<bool>(device));
    CG_EXPECT_TRUE(static_cast<bool>(device.Value().CreateSampler(desc)));
}

CG_TEST(renderer_shadow_pass_rasterizes_and_samples_depth_on_d3d12)
{
    using namespace cyber::renderer;
    using namespace cyber::rhi;

    const Backend* backend = D3D12Backend();
    if (backend == nullptr || backend->enumerateAdapters == nullptr ||
        backend->enumerateAdapters(nullptr, 0) == 0)
    {
        CG_SKIP("no Direct3D 12 adapter is available");
    }

    cyber::memory::HeapAllocator heap;
    DeviceDesc deviceDesc{};
    deviceDesc.backend = backend;
    deviceDesc.adapterIndex = backend->enumerateAdapters(nullptr, 0) - 1;
    auto createdDevice = Device::Create(deviceDesc, heap);
    if (!createdDevice)
    {
        CG_SKIP("Direct3D 12 device creation is unavailable");
    }
    Device device = std::move(createdDevice.Value());

    constexpr char kVertexShader[] =
        "struct Input { float3 position : TEXCOORD0; };\n"
        "float4 main(Input input) : SV_Position { return float4(input.position, 1.0); }\n";
    ShaderDesc vertexDesc{};
    vertexDesc.stage = ShaderStage::Vertex;
    vertexDesc.format = ShaderBytecodeFormat::HlslSource;
    vertexDesc.code = ShaderBytes(kVertexShader);
    vertexDesc.entryPoint = "main";
    auto vertexShader = device.CreateShader(vertexDesc);
    CG_REQUIRE(static_cast<bool>(vertexShader));
    PipelineLayoutDesc layoutDesc{};
    auto layout = device.CreatePipelineLayout(layoutDesc);
    CG_REQUIRE(static_cast<bool>(layout));
    VertexBinding binding{12, VertexInputRate::PerVertex};
    VertexAttribute attribute{0, 0, Format::R32G32B32Float, 0};
    GraphicsPipelineDesc pipelineDesc{};
    pipelineDesc.layout = layout.Value();
    pipelineDesc.vertexShader = vertexShader.Value();
    pipelineDesc.vertexLayout.bindings = cyber::memory::Span<const VertexBinding>(&binding, 1);
    pipelineDesc.vertexLayout.attributes =
        cyber::memory::Span<const VertexAttribute>(&attribute, 1);
    pipelineDesc.rasterizer.cullMode = CullMode::None;
    pipelineDesc.depthStencil.depthTestEnabled = true;
    pipelineDesc.depthStencil.depthWriteEnabled = true;
    pipelineDesc.depthStencil.depthCompare = CompareFunction::GreaterEqual;
    pipelineDesc.depthStencilFormat = Format::D32Float;
    pipelineDesc.label = "Phase 28 D3D12 shadow depth";
    auto pipeline = device.CreateGraphicsPipeline(pipelineDesc);
    CG_REQUIRE(static_cast<bool>(pipeline));

    constexpr cyber::core::u32 kResolution = 64;
    ShadowMapResources shadowResources{};
    CG_REQUIRE(CreateShadowMapResources(device, kResolution, kResolution, shadowResources).IsOk());

    constexpr char kShadowSamplePixelShader[] =
        "Texture2D<float> shadowDepth : register(t0, space1);\n"
        "SamplerComparisonState shadowSampler : register(s0, space1);\n"
        "float4 main(float4 position : SV_Position) : SV_Target\n"
        "{\n"
        "    float visibility = shadowDepth.SampleCmpLevelZero(shadowSampler, "
        "position.xy / 64.0, 0.6);\n"
        "    float outside = shadowDepth.SampleCmpLevelZero(shadowSampler, "
        "float2(1.5, 0.5), 0.6);\n"
        "    float3 albedo = float3(0.8, 0.7, 0.6);\n"
        "    float3 ambient = albedo * 0.05;\n"
        "    float3 direct = albedo * (0.70 * visibility);\n"
        "    return float4(ambient + direct, outside);\n"
        "}\n";
    ShaderDesc samplePixelDesc{};
    samplePixelDesc.stage = ShaderStage::Pixel;
    samplePixelDesc.format = ShaderBytecodeFormat::HlslSource;
    samplePixelDesc.code = ShaderBytes(kShadowSamplePixelShader);
    samplePixelDesc.entryPoint = "main";
    auto samplePixelShader = device.CreateShader(samplePixelDesc);
    CG_REQUIRE(static_cast<bool>(samplePixelShader));

    BindGroupLayoutEntry shadowEntries[2]{};
    shadowEntries[0].binding = 0;
    shadowEntries[0].type = BindingType::SampledTexture;
    shadowEntries[0].visibility = ShaderStageBit(ShaderStage::Pixel);
    shadowEntries[1].binding = 1;
    shadowEntries[1].type = BindingType::Sampler;
    shadowEntries[1].visibility = ShaderStageBit(ShaderStage::Pixel);
    BindGroupLayoutDesc shadowGroupDesc{};
    shadowGroupDesc.entries = cyber::memory::Span<const BindGroupLayoutEntry>(shadowEntries, 2);
    auto shadowGroupLayout = device.CreateBindGroupLayout(shadowGroupDesc);
    CG_REQUIRE(static_cast<bool>(shadowGroupLayout));

    PipelineLayoutDesc sampleLayoutDesc{};
    sampleLayoutDesc.groups[static_cast<cyber::core::u32>(BindGroupSlot::PerView)] =
        shadowGroupLayout.Value();
    auto sampleLayout = device.CreatePipelineLayout(sampleLayoutDesc);
    CG_REQUIRE(static_cast<bool>(sampleLayout));

    SamplerDesc shadowSamplerDesc = MakeReverseZShadowSamplerDesc();
    auto shadowSampler = device.CreateSampler(shadowSamplerDesc);
    CG_REQUIRE(static_cast<bool>(shadowSampler));

    BindingResource shadowBindings[2]{};
    shadowBindings[0].type = BindingType::SampledTexture;
    shadowBindings[0].textureView = shadowResources.sampledView;
    shadowBindings[1].type = BindingType::Sampler;
    shadowBindings[1].sampler = shadowSampler.Value();
    BindGroupDesc shadowBindGroupDesc{};
    shadowBindGroupDesc.layout = shadowGroupLayout.Value();
    shadowBindGroupDesc.resources = cyber::memory::Span<const BindingResource>(shadowBindings, 2);
    auto shadowBindGroup = device.CreateBindGroup(shadowBindGroupDesc);
    CG_REQUIRE(static_cast<bool>(shadowBindGroup));

    ColorAttachmentState sampleColorState{};
    sampleColorState.format = Format::RGBA8Unorm;
    GraphicsPipelineDesc samplePipelineDesc{};
    samplePipelineDesc.layout = sampleLayout.Value();
    samplePipelineDesc.vertexShader = vertexShader.Value();
    samplePipelineDesc.pixelShader = samplePixelShader.Value();
    samplePipelineDesc.rasterizer.cullMode = CullMode::None;
    samplePipelineDesc.vertexLayout.bindings =
        cyber::memory::Span<const VertexBinding>(&binding, 1);
    samplePipelineDesc.vertexLayout.attributes =
        cyber::memory::Span<const VertexAttribute>(&attribute, 1);
    samplePipelineDesc.depthStencil.depthTestEnabled = false;
    samplePipelineDesc.depthStencil.depthWriteEnabled = false;
    samplePipelineDesc.colorAttachments =
        cyber::memory::Span<const ColorAttachmentState>(&sampleColorState, 1);
    auto samplePipeline = device.CreateGraphicsPipeline(samplePipelineDesc);
    if (!samplePipeline)
    {
        CG_FAIL(samplePipeline.Error().message);
        return;
    }

    RenderGraph graph(device, heap);
    auto depthResource = ImportShadowMap(graph, "d3d12_shadow_depth", shadowResources);
    CG_REQUIRE(static_cast<bool>(depthResource));

    struct Vertex
    {
        float position[3];
    };
    constexpr std::array<Vertex, 3> kVertices{{
        {{-0.75f, -0.75f, 0.75f}},
        {{0.75f, -0.75f, 0.75f}},
        {{0.0f, 0.75f, 0.75f}},
    }};
    constexpr std::array<Vertex, 3> kFullscreenVertices{{
        {{-1.0f, -1.0f, 0.6f}},
        {{-1.0f, 3.0f, 0.6f}},
        {{3.0f, -1.0f, 0.6f}},
    }};
    BufferDesc stagingDesc{};
    stagingDesc.usage = BufferUsage::HostVisible | BufferUsage::CopySource;
    stagingDesc.sizeBytes = sizeof(kVertices) + sizeof(kFullscreenVertices);
    auto staging = device.CreateBuffer(stagingDesc);
    CG_REQUIRE(static_cast<bool>(staging));
    const auto vertexBytes = cyber::memory::Span<const cyber::memory::Byte>(
        reinterpret_cast<const cyber::memory::Byte*>(kVertices.data()), sizeof(kVertices));
    CG_REQUIRE(device.WriteBuffer(staging.Value(), 0, vertexBytes).IsOk());
    const auto fullscreenVertexBytes = cyber::memory::Span<const cyber::memory::Byte>(
        reinterpret_cast<const cyber::memory::Byte*>(kFullscreenVertices.data()),
        sizeof(kFullscreenVertices));
    CG_REQUIRE(
        device.WriteBuffer(staging.Value(), sizeof(kVertices), fullscreenVertexBytes).IsOk());
    BufferDesc vertexBufferDesc{};
    vertexBufferDesc.usage = BufferUsage::Vertex | BufferUsage::CopyDestination;
    vertexBufferDesc.sizeBytes = sizeof(kVertices);
    auto vertexBuffer = device.CreateBuffer(vertexBufferDesc);
    CG_REQUIRE(static_cast<bool>(vertexBuffer));
    vertexBufferDesc.sizeBytes = sizeof(kFullscreenVertices);
    auto fullscreenVertexBuffer = device.CreateBuffer(vertexBufferDesc);
    CG_REQUIRE(static_cast<bool>(fullscreenVertexBuffer));

    auto fence = device.CreateFence(FenceDesc{});
    CG_REQUIRE(static_cast<bool>(fence));
    CommandListDesc commandDesc{};
    auto commandListResult = device.CreateCommandList(commandDesc);
    CG_REQUIRE(static_cast<bool>(commandListResult));
    CommandList commandList = std::move(commandListResult.Value());
    CG_REQUIRE(commandList.Begin().IsOk());
    const BufferBarrier uploadBarriers[] = {
        {staging.Value(), ResourceState::Undefined, ResourceState::CopySource},
        {vertexBuffer.Value(), ResourceState::Undefined, ResourceState::CopyDestination},
        {fullscreenVertexBuffer.Value(), ResourceState::Undefined, ResourceState::CopyDestination},
    };
    CG_REQUIRE(
        commandList.Barrier(cyber::memory::Span<const BufferBarrier>(uploadBarriers, 3)).IsOk());
    CG_REQUIRE(
        commandList
            .CopyBuffer(BufferCopy{staging.Value(), 0, vertexBuffer.Value(), 0, sizeof(kVertices)})
            .IsOk());
    CG_REQUIRE(
        commandList
            .CopyBuffer(BufferCopy{staging.Value(), sizeof(kVertices),
                                   fullscreenVertexBuffer.Value(), 0, sizeof(kFullscreenVertices)})
            .IsOk());
    const BufferBarrier vertexReady[] = {
        {vertexBuffer.Value(), ResourceState::CopyDestination, ResourceState::VertexBuffer},
        {fullscreenVertexBuffer.Value(), ResourceState::CopyDestination,
         ResourceState::VertexBuffer},
    };
    CG_REQUIRE(
        commandList.Barrier(cyber::memory::Span<const BufferBarrier>(vertexReady, 2)).IsOk());

    GpuShadowDraw draw{pipeline.Value(), vertexBuffer.Value(), kResolution};
    ShadowDepthPassDesc pass{};
    pass.name = "D3D12 shadow map geometry";
    pass.depthResource = depthResource.Value();
    pass.depthTexture = shadowResources.depthTexture;
    pass.depthView = shadowResources.depthView;
    pass.record = &RecordGpuShadowGeometry;
    pass.userData = &draw;
    CG_REQUIRE(AddShadowDepthPass(graph, pass).IsOk());
    CG_REQUIRE(graph.Compile().IsOk());
    CG_REQUIRE(graph.Execute(commandList).IsOk());
    CG_REQUIRE(commandList.End().IsOk());

    const CommandListHandle firstList = commandList.Handle();
    const FenceSignal firstSignal{fence.Value(), 1};
    SubmitDesc firstSubmit{};
    firstSubmit.queue = QueueKind::Graphics;
    firstSubmit.commandLists = cyber::memory::Span<const CommandListHandle>(&firstList, 1);
    firstSubmit.signals = cyber::memory::Span<const FenceSignal>(&firstSignal, 1);
    CG_REQUIRE(device.Submit(firstSubmit).IsOk());
    CG_REQUIRE(graph.AcknowledgeSubmission().IsOk());
    CG_REQUIRE(device.WaitFence(fence.Value(), 1).IsOk());
    CG_EXPECT_EQ(commandList.Stats().draws, cyber::core::u32{1});

    constexpr cyber::core::u64 kReadbackBytes =
        static_cast<cyber::core::u64>(kResolution) * kResolution * sizeof(float);
    BufferDesc readbackDesc{};
    readbackDesc.usage = BufferUsage::Readback;
    readbackDesc.sizeBytes = kReadbackBytes;
    auto readback = device.CreateBuffer(readbackDesc);
    CG_REQUIRE(static_cast<bool>(readback));
    auto copyListResult = device.CreateCommandList(commandDesc);
    CG_REQUIRE(static_cast<bool>(copyListResult));
    CommandList copyList = std::move(copyListResult.Value());
    CG_REQUIRE(copyList.Begin().IsOk());
    const TextureBarrier depthToCopy{shadowResources.depthTexture, SubresourceRange{},
                                     ResourceState::DepthWrite, ResourceState::CopySource};
    const BufferBarrier readbackReady{readback.Value(), ResourceState::Undefined,
                                      ResourceState::CopyDestination};
    CG_REQUIRE(copyList
                   .Barrier(cyber::memory::Span<const BufferBarrier>(&readbackReady, 1),
                            cyber::memory::Span<const TextureBarrier>(&depthToCopy, 1))
                   .IsOk());
    BufferTextureCopy copy{};
    copy.buffer = readback.Value();
    copy.bufferRowPitch = kResolution * sizeof(float);
    copy.bufferDepthPitch = static_cast<cyber::core::u32>(kReadbackBytes);
    copy.texture = shadowResources.depthTexture;
    copy.width = kResolution;
    copy.height = kResolution;
    CG_REQUIRE(copyList.CopyTextureToBuffer(copy).IsOk());
    CG_REQUIRE(copyList.End().IsOk());
    const CommandListHandle copyHandle = copyList.Handle();
    const FenceSignal copySignal{fence.Value(), 2};
    SubmitDesc copySubmit{};
    copySubmit.queue = QueueKind::Graphics;
    copySubmit.commandLists = cyber::memory::Span<const CommandListHandle>(&copyHandle, 1);
    copySubmit.signals = cyber::memory::Span<const FenceSignal>(&copySignal, 1);
    CG_REQUIRE(device.Submit(copySubmit).IsOk());
    CG_REQUIRE(device.WaitFence(fence.Value(), 2).IsOk());

    std::array<cyber::memory::Byte, static_cast<std::size_t>(kReadbackBytes)> depthPixels{};
    CG_REQUIRE(device
                   .ReadBuffer(readback.Value(), 0,
                               cyber::memory::Span<cyber::memory::Byte>(depthPixels.data(),
                                                                        depthPixels.size()))
                   .IsOk());
    float centerDepth = 0.0f;
    std::memcpy(&centerDepth,
                depthPixels.data() +
                    (static_cast<std::size_t>(kResolution / 2) * kResolution + kResolution / 2) *
                        sizeof(float),
                sizeof(centerDepth));
    CG_EXPECT_TRUE(centerDepth > 0.70f && centerDepth < 0.80f);
    CG_REQUIRE(graph.Destroy().IsOk());

    TextureDesc sampleColorDesc{};
    sampleColorDesc.format = Format::RGBA8Unorm;
    sampleColorDesc.usage = TextureUsage::RenderTarget | TextureUsage::CopySource;
    sampleColorDesc.width = kResolution;
    sampleColorDesc.height = kResolution;
    auto sampleColor = device.CreateTexture(sampleColorDesc);
    CG_REQUIRE(static_cast<bool>(sampleColor));
    TextureViewDesc sampleColorViewDesc{};
    sampleColorViewDesc.texture = sampleColor.Value();
    sampleColorViewDesc.format = Format::RGBA8Unorm;
    auto sampleColorView = device.CreateTextureView(sampleColorViewDesc);
    CG_REQUIRE(static_cast<bool>(sampleColorView));

    BufferDesc sampleReadbackDesc{};
    sampleReadbackDesc.usage = BufferUsage::Readback | BufferUsage::CopyDestination;
    sampleReadbackDesc.sizeBytes = static_cast<cyber::core::u64>(kResolution) * kResolution * 4;
    auto sampleReadback = device.CreateBuffer(sampleReadbackDesc);
    CG_REQUIRE(static_cast<bool>(sampleReadback));

    RenderGraph sampleGraph(device, heap);
    auto sampledDepthResource =
        ImportShadowMap(sampleGraph, "d3d12_shadow_depth_for_lighting", shadowResources);
    CG_REQUIRE(static_cast<bool>(sampledDepthResource));
    const GraphResourceHandle sampledDepthResources[] = {sampledDepthResource.Value()};
    GpuShadowedLightingDraw lightingDraw{
        samplePipeline.Value(),       shadowBindGroup.Value(),      fullscreenVertexBuffer.Value(),
        shadowResources.depthTexture, sampledDepthResource.Value(), sampleColor.Value(),
        sampleColorView.Value(),      sampleReadback.Value(),       kResolution};
    ShadowSamplingPassDesc samplingPass{};
    samplingPass.name = "D3D12 shadowed receiver lighting";
    samplingPass.shadowResources =
        cyber::memory::Span<const GraphResourceHandle>(sampledDepthResources, 1);
    samplingPass.record = &RecordGpuShadowedLighting;
    samplingPass.userData = &lightingDraw;
    CG_REQUIRE(AddShadowSamplingPass(sampleGraph, samplingPass).IsOk());
    CG_REQUIRE(sampleGraph.Compile().IsOk());

    auto sampleListResult = device.CreateCommandList(commandDesc);
    CG_REQUIRE(static_cast<bool>(sampleListResult));
    CommandList sampleList = std::move(sampleListResult.Value());
    CG_REQUIRE(sampleList.Begin().IsOk());
    CG_REQUIRE(sampleGraph.Execute(sampleList).IsOk());
    CG_REQUIRE(sampleList.End().IsOk());
    const CommandListHandle sampleListHandle = sampleList.Handle();
    const FenceSignal sampleSignal{fence.Value(), 3};
    SubmitDesc sampleSubmit{};
    sampleSubmit.queue = QueueKind::Graphics;
    sampleSubmit.commandLists = cyber::memory::Span<const CommandListHandle>(&sampleListHandle, 1);
    sampleSubmit.signals = cyber::memory::Span<const FenceSignal>(&sampleSignal, 1);
    CG_REQUIRE(device.Submit(sampleSubmit).IsOk());
    CG_REQUIRE(sampleGraph.AcknowledgeSubmission().IsOk());
    CG_REQUIRE(device.WaitFence(fence.Value(), 3).IsOk());

    std::array<cyber::memory::Byte, static_cast<std::size_t>(kResolution) * kResolution * 4>
        samplePixels{};
    CG_REQUIRE(device
                   .ReadBuffer(sampleReadback.Value(), 0,
                               cyber::memory::Span<cyber::memory::Byte>(samplePixels.data(),
                                                                        samplePixels.size()))
                   .IsOk());
    const std::size_t centerPixel =
        (static_cast<std::size_t>(kResolution / 2) * kResolution + kResolution / 2) * 4;
    const std::size_t litPixel = (2 * static_cast<std::size_t>(kResolution) + 2) * 4;
    const auto channel = [&samplePixels](std::size_t pixel, std::size_t component)
    { return static_cast<cyber::core::u8>(samplePixels[pixel + component]); };
    const cyber::core::u32 shadowedLuminance =
        channel(centerPixel, 0) + channel(centerPixel, 1) + channel(centerPixel, 2);
    const cyber::core::u32 litLuminance =
        channel(litPixel, 0) + channel(litPixel, 1) + channel(litPixel, 2);
    CG_EXPECT_TRUE(litLuminance > shadowedLuminance + 100);
    CG_EXPECT_EQ(channel(centerPixel, 3), cyber::core::u8{255});
    CG_EXPECT_EQ(channel(litPixel, 3), cyber::core::u8{255});

    CG_REQUIRE(sampleGraph.Destroy().IsOk());
    CG_REQUIRE(DestroyShadowMapResources(device, shadowResources).IsOk());
}
